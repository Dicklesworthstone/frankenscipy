#![forbid(unsafe_code)]
//! Live SciPy diff for `minimize(method=Powell)` over seeded smooth problems
//! (frankenscipy-cjv9z, frankenscipy-de6qs).
//!
//! cjv9z: Powell stopped on an ABSOLUTE |Δf| test, so a scaled objective (1e-12·rosen) stopped
//! early; SciPy's test is relative. de6qs: the line search was a golden section on unit
//! directions, which capped x* accuracy near 2e-5; SciPy uses Brent on unnormalized directions.
//! This runs 20 seeded problems from five smooth families, plus the two beads' named rows, through
//! both `fsci_opt::minimize` and `scipy.optimize.minimize(method='Powell')`. Every objective is
//! written term by term identically on both sides (a vectorized sum would change the rounding).
//! Per row: the same success flag and x* within X_ABS_TOL of SciPy's. Named rows:
//! - 1e-12·rosen and rosen from (-1.2, 1) reach the same x* (to 1e-4, cjv9z);
//! - rosen from (-1.2, 1) with tol = 1e-6 lands within ROSEN_TOL6_ABS_TOL of (1, 1) (de6qs).
//!
//! The compared count is asserted.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_opt::{MinimizeOptions, OptimizeMethod, minimize};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// max |x_fsci - x_scipy| per row (de6qs asks 1e-8).
const X_ABS_TOL: f64 = 1e-8;
/// cjv9z: the scaled and unscaled Rosenbrock minima agree to this.
const SCALE_INVARIANCE_TOL: f64 = 1e-4;
/// de6qs: Rosenbrock at tol 1e-6 lands this close to (1, 1).
const ROSEN_TOL6_ABS_TOL: f64 = 1e-10;

/// kind: 0 scaled Rosenbrock (p[0] = scale), 1 quadratic 3-D (p = a, b, d, e, c0, c1, c2),
/// 2 Beale, 3 Rosenbrock 3-D, 4 exp of a 2-D quadratic (p = a, b, e, c0, c1).
#[derive(Debug, Clone, Serialize)]
struct Problem {
    name: String,
    kind: u32,
    p: Vec<f64>,
    x0: Vec<f64>,
    tol: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    x: Vec<f64>,
    success: bool,
    nfev: usize,
}

fn eval(kind: u32, p: &[f64], x: &[f64]) -> f64 {
    match kind {
        0 => {
            let (t1, t2) = (x[1] - x[0] * x[0], 1.0 - x[0]);
            p[0] * (100.0 * t1 * t1 + t2 * t2)
        }
        1 => {
            let (u, v, w) = (x[0] - p[4], x[1] - p[5], x[2] - p[6]);
            p[0] * u * u + p[1] * v * v + p[2] * w * w + p[3] * u * v
        }
        2 => {
            let (a, b) = (x[0], x[1]);
            let t1 = 1.5 - a + a * b;
            let t2 = 2.25 - a + a * b * b;
            let t3 = 2.625 - a + a * b * b * b;
            t1 * t1 + t2 * t2 + t3 * t3
        }
        3 => {
            let (t1, t2) = (x[1] - x[0] * x[0], 1.0 - x[0]);
            let (t3, t4) = (x[2] - x[1] * x[1], 1.0 - x[1]);
            100.0 * t1 * t1 + t2 * t2 + 100.0 * t3 * t3 + t4 * t4
        }
        _ => {
            let (u, v) = (x[0] - p[3], x[1] - p[4]);
            (0.1 * (p[0] * u * u + p[1] * v * v + p[2] * u * v)).exp()
        }
    }
}

/// splitmix64 on [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn around(&mut self, c: f64, r: f64) -> f64 {
        c + r * (2.0 * self.next() - 1.0)
    }
}

fn problems() -> Vec<Problem> {
    let mut rng = Rng(0x0907_2025);
    let mut out = Vec::new();
    for k in 0..20_u32 {
        let kind = k % 5;
        let (p, x0) = match kind {
            0 => {
                let scale = 10f64.powi(-3 * (k / 5) as i32);
                (
                    vec![scale],
                    vec![rng.around(-1.2, 0.3), rng.around(1.0, 0.3)],
                )
            }
            1 => {
                let (a, b, d) = (
                    rng.around(2.0, 1.5),
                    rng.around(3.0, 2.0),
                    rng.around(1.5, 1.0),
                );
                let e = (2.0 * rng.next() - 1.0) * (a * b).sqrt();
                let c: Vec<f64> = (0..3).map(|_| rng.around(0.0, 2.0)).collect();
                (
                    vec![a, b, d, e, c[0], c[1], c[2]],
                    (0..3).map(|_| rng.around(0.0, 3.0)).collect(),
                )
            }
            2 => (vec![], vec![rng.around(1.0, 0.5), rng.around(1.0, 0.5)]),
            3 => (
                vec![],
                vec![
                    rng.around(-1.0, 0.4),
                    rng.around(1.0, 0.4),
                    rng.around(1.0, 0.4),
                ],
            ),
            _ => {
                let (a, b) = (rng.around(2.0, 1.0), rng.around(2.0, 1.0));
                let e = (2.0 * rng.next() - 1.0) * (a * b).sqrt();
                (
                    vec![a, b, e, rng.around(0.5, 1.0), rng.around(-0.5, 1.0)],
                    vec![rng.around(0.0, 2.0), rng.around(0.0, 2.0)],
                )
            }
        };
        out.push(Problem {
            name: format!("seed{k}_kind{kind}"),
            kind,
            p,
            x0,
            tol: None,
        });
    }
    let rosen = |name: &str, scale: f64, tol: Option<f64>| Problem {
        name: name.to_string(),
        kind: 0,
        p: vec![scale],
        x0: vec![-1.2, 1.0],
        tol,
    };
    out.push(rosen("rosen_unscaled", 1.0, None));
    out.push(rosen("rosen_scaled_1e-12", 1e-12, None));
    out.push(rosen("rosen_tol_1e-6", 1.0, Some(1e-6)));
    out
}

fn scipy_answers(problems: &[Problem]) -> Option<Vec<Answer>> {
    let script = r#"
import json, sys, math
from scipy.optimize import minimize

def make(kind, p):
    if kind == 0:
        def f(x):
            t1, t2 = x[1] - x[0] * x[0], 1.0 - x[0]
            return p[0] * (100.0 * t1 * t1 + t2 * t2)
    elif kind == 1:
        def f(x):
            u, v, w = x[0] - p[4], x[1] - p[5], x[2] - p[6]
            return p[0] * u * u + p[1] * v * v + p[2] * w * w + p[3] * u * v
    elif kind == 2:
        def f(x):
            a, b = x[0], x[1]
            t1 = 1.5 - a + a * b
            t2 = 2.25 - a + a * b * b
            t3 = 2.625 - a + a * b * b * b
            return t1 * t1 + t2 * t2 + t3 * t3
    elif kind == 3:
        def f(x):
            t1, t2 = x[1] - x[0] * x[0], 1.0 - x[0]
            t3, t4 = x[2] - x[1] * x[1], 1.0 - x[1]
            return 100.0 * t1 * t1 + t2 * t2 + 100.0 * t3 * t3 + t4 * t4
    else:
        def f(x):
            u, v = x[0] - p[3], x[1] - p[4]
            return math.exp(0.1 * (p[0] * u * u + p[1] * v * v + p[2] * u * v))
    return lambda x: float(f(x))

out = []
for q in json.load(sys.stdin):
    r = minimize(make(q["kind"], q["p"]), q["x0"], method="Powell", tol=q["tol"])
    out.append({"x": [float(v) for v in r.x], "success": bool(r.success), "nfev": int(r.nfev)})
print(json.dumps(out))
"#;
    let query = serde_json::to_string(problems).expect("serialize problems");
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
                "failed to spawn the Powell oracle: {e}"
            );
            eprintln!("skipping Powell oracle: python not available ({e})");
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
        .expect("wait for the Powell oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "Powell oracle failed: {stderr}"
        );
        eprintln!("skipping Powell oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse Powell oracle JSON"))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_opt_powell() {
    let problems = problems();
    let Some(answers) = scipy_answers(&problems) else {
        return;
    };
    assert_eq!(
        answers.len(),
        problems.len(),
        "the oracle must answer every problem"
    );
    let mut compared = 0;
    let mut same_nfev = 0;
    let mut failures = Vec::new();
    let mut solutions = std::collections::HashMap::new();
    let mut ledger = CompareLedger::new("diff_opt_powell", &["powell"]);
    for (problem, answer) in problems.iter().zip(&answers) {
        let (kind, p) = (problem.kind, problem.p.clone());
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::Powell),
            tol: problem.tol,
            ..MinimizeOptions::default()
        };
        let result = minimize(move |x: &[f64]| eval(kind, &p, x), &problem.x0, options)
            .map_err(|e| format!("{e:?}"));
        let Some((answer, result)) =
            ledger.both("powell", &problem.name, Some(answer), result.as_ref().ok())
        else {
            if let Err(e) = &result {
                failures.push(format!("{}: fsci Err({e})", problem.name));
            }
            compared += 1;
            continue;
        };
        solutions.insert(problem.name.clone(), result.x.clone());
        // Rejects a length mismatch and a NaN in fsci's x that `max_abs_diff`'s fold would drop.
        let Some((expected_x, fsci_x)) = ledger.slices(
            "powell",
            &problem.name,
            Some(answer.x.as_slice()),
            Some(result.x.as_slice()),
        ) else {
            failures.push(format!(
                "{}: x {:?} vs SciPy {:?}",
                problem.name, result.x, answer.x
            ));
            compared += 1;
            continue;
        };
        let d = max_abs_diff(fsci_x, expected_x);
        let fsci_success = result.success;
        if result.nfev == answer.nfev {
            same_nfev += 1;
        }
        println!(
            "{}: |x - x_scipy| {d:.2e} | success fsci {fsci_success} SciPy {} | nfev fsci {} SciPy {}",
            problem.name, answer.success, result.nfev, answer.nfev
        );
        let row_fails = d.is_nan() || d > X_ABS_TOL || fsci_success != answer.success;
        ledger.compared("powell", &problem.name, !row_fails);
        if row_fails {
            failures.push(format!(
                "{}: x {:?} vs SciPy {:?}, success {fsci_success} vs {}",
                problem.name, result.x, answer.x, answer.success
            ));
        }
        compared += 1;
    }

    // cjv9z: scaling the objective does not change where Powell stops.
    let (unscaled, scaled) = (
        &solutions["rosen_unscaled"],
        &solutions["rosen_scaled_1e-12"],
    );
    let scale_gap = max_abs_diff(unscaled, scaled);
    println!("1e-12 * rosen vs rosen: x* gap {scale_gap:.2e}");
    if scale_gap.is_nan() || scale_gap > SCALE_INVARIANCE_TOL {
        failures.push(format!("scaled rosen stops elsewhere: gap {scale_gap:e}"));
    }
    // de6qs: at tol 1e-6 the Brent line search reaches (1, 1) to 1e-10.
    let tight = max_abs_diff(&solutions["rosen_tol_1e-6"], &[1.0, 1.0]);
    println!("rosen tol 1e-6: |x - (1, 1)| {tight:.2e}");
    if tight.is_nan() || tight > ROSEN_TOL6_ABS_TOL {
        failures.push(format!("rosen tol 1e-6 lands {tight:e} from (1, 1)"));
    }

    println!("{compared} problems compared; nfev equal to SciPy's on {same_nfev}");
    assert_eq!(compared, problems.len());
    assert!(failures.is_empty(), "Powell disagrees: {failures:#?}");
    ledger.finish(problems.len());
}
