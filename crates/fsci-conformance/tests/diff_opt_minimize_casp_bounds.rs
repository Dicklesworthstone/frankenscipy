#![forbid(unsafe_code)]
//! frankenscipy-szq1n.9: the CASP optimizer portfolio treats bounds as a feasibility
//! requirement. BFGS and Newton-CG ignore `options.bounds`, so a portfolio that scored them
//! lower than L-BFGS-B used to return an infeasible x reported as success (minimize (x − 5)²
//! on [−1, 1] came back at x ≈ 5). The candidates are now restricted before the argmin to the
//! methods that honour bounds: L-BFGS-B, Nelder-Mead (clips every trial point), and DIRECT
//! when every side is finite.
//!
//!   * `casp_bounded_scalar_lands_on_the_bound`: the bead's negative case, plus the same
//!     problem unbounded (x = 5), which shows the check can tell the two apart;
//!   * `casp_bounded_quadratics_are_feasible_and_optimal`: 200 seeded separable convex
//!     quadratics with random boxes; every answer lies inside its box exactly and matches the
//!     analytic optimum clip(c); a must-hit count keeps the property from passing vacuously;
//!   * `diff_opt_minimize_casp_bounded_rosenbrock_scipy`: live SciPy L-BFGS-B on Rosenbrock
//!     over [−2, 0.5]², x* to 1e-6, with the compared-case count asserted.

use std::process::Stdio;

use fsci_opt::types::Bound;
use fsci_opt::{MinimizeOptions, minimize_with_casp};
use fsci_runtime::{OptSolverAction, OptSolverPortfolio, RuntimeMode};
use serde::Deserialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// The bounded 1-D quadratic's minimizer sits exactly on its active bound x = 1.
const ACTIVE_BOUND_X_TOL: f64 = 1e-8;
/// The same quadratic without bounds, minimized at x = 5 (the must-differ arm).
const FREE_MINIMUM_X_TOL: f64 = 1e-6;
/// Worst relative error against the analytic optimum of the seeded boxed quadratics.
const SEEDED_BOX_REL_TOL: f64 = 1e-5;
/// Bounded Rosenbrock x against SciPy's L-BFGS-B x.
const SCIPY_X_ABS_TOL: f64 = 1e-6;

fn bounded_methods(action: OptSolverAction) -> bool {
    matches!(
        action,
        OptSolverAction::LBFGSB | OptSolverAction::NelderMead | OptSolverAction::DIRECT
    )
}

#[test]
fn casp_bounded_scalar_lands_on_the_bound() {
    let f = |x: &[f64]| (x[0] - 5.0).powi(2);
    let bounds: [Bound; 1] = [(Some(-1.0), Some(1.0))];
    let options = MinimizeOptions {
        bounds: Some(&bounds),
        ..MinimizeOptions::default()
    };
    let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 8);
    let out = minimize_with_casp(f, &[0.0], options, &mut portfolio).expect("bounded minimize");
    println!(
        "bounded: action={:?} posterior={:?} losses={:?} x={:?} success={}",
        out.chosen_action, out.posterior, out.expected_losses, out.result.x, out.result.success
    );
    assert!(
        bounded_methods(out.chosen_action),
        "{:?}",
        out.chosen_action
    );
    assert!(
        (-1.0..=1.0).contains(&out.result.x[0]),
        "infeasible x = {:?}",
        out.result.x
    );
    assert!(
        (out.result.x[0] - 1.0).abs() <= ACTIVE_BOUND_X_TOL,
        "x = {:?}, expected the bound 1",
        out.result.x
    );

    // The same problem without bounds is minimized at 5: the check above distinguishes a
    // solver that honours the box from one that ignores it.
    let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 8);
    let free = minimize_with_casp(f, &[0.0], MinimizeOptions::default(), &mut portfolio)
        .expect("unbounded minimize");
    println!(
        "unbounded: action={:?} x={:?}",
        free.chosen_action, free.result.x
    );
    assert!(
        (free.result.x[0] - 5.0).abs() <= FREE_MINIMUM_X_TOL,
        "x = {:?}",
        free.result.x
    );
}

/// splitmix64, for seeded cases.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

#[test]
fn casp_bounded_quadratics_are_feasible_and_optimal() {
    let mut rng = Rng(0x5A71_0009_B0D5_0001);
    let (mut cases, mut binding, mut worst) = (0_usize, 0_usize, 0.0_f64);
    for case in 0..200 {
        let n = 1 + (rng.next() % 6) as usize;
        let weights: Vec<f64> = (0..n).map(|_| rng.uniform(0.5, 20.0)).collect();
        let centres: Vec<f64> = (0..n).map(|_| rng.uniform(-8.0, 8.0)).collect();
        // Boxes: finite on both sides, or open on one side (DIRECT then drops out).
        let bounds: Vec<Bound> = (0..n)
            .map(|_| {
                let lo = rng.uniform(-4.0, 1.0);
                let hi = lo + rng.uniform(0.5, 4.0);
                match rng.next() % 4 {
                    0 => (None, Some(hi)),
                    1 => (Some(lo), None),
                    _ => (Some(lo), Some(hi)),
                }
            })
            .collect();
        let x0: Vec<f64> = bounds
            .iter()
            .map(|&(lo, hi)| match (lo, hi) {
                (Some(lo), Some(hi)) => 0.5 * (lo + hi),
                (Some(lo), None) => lo + 1.0,
                (None, Some(hi)) => hi - 1.0,
                (None, None) => 0.0,
            })
            .collect();
        let optimum: Vec<f64> = centres
            .iter()
            .zip(&bounds)
            .map(|(&c, &(lo, hi))| {
                c.max(lo.unwrap_or(f64::NEG_INFINITY))
                    .min(hi.unwrap_or(f64::INFINITY))
            })
            .collect();
        if optimum.iter().zip(&centres).any(|(o, c)| o != c) {
            binding += 1;
        }
        let f = |x: &[f64]| -> f64 {
            x.iter()
                .zip(&centres)
                .zip(&weights)
                .map(|((xi, ci), wi)| wi * (xi - ci).powi(2))
                .sum()
        };
        let options = MinimizeOptions {
            bounds: Some(&bounds),
            tol: Some(1e-12),
            maxiter: Some(5000),
            ..MinimizeOptions::default()
        };
        let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 4);
        let out = minimize_with_casp(f, &x0, options, &mut portfolio);
        assert!(out.is_ok(), "case {case}: {:?}", out.as_ref().err());
        let out = out.expect("checked above");
        assert!(
            bounded_methods(out.chosen_action),
            "case {case}: {:?} ignores bounds",
            out.chosen_action
        );
        for (i, (&xi, &(lo, hi))) in out.result.x.iter().zip(&bounds).enumerate() {
            assert!(
                lo.is_none_or(|lo| xi >= lo) && hi.is_none_or(|hi| xi <= hi),
                "case {case}: x[{i}] = {xi} outside {:?} ({:?})",
                (lo, hi),
                out.chosen_action
            );
        }
        let error = out
            .result
            .x
            .iter()
            .zip(&optimum)
            .map(|(x, o)| (x - o).abs() / o.abs().max(1.0))
            .fold(0.0_f64, f64::max);
        worst = worst.max(error);
        assert!(
            error <= SEEDED_BOX_REL_TOL,
            "case {case}: x = {:?}, optimum {optimum:?} ({:?})",
            out.result.x,
            out.chosen_action
        );
        cases += 1;
    }
    println!("{cases} bounded quadratics, {binding} with a binding bound, worst error {worst:e}");
    assert_eq!(cases, 200);
    // Must-hit: most cases have their unconstrained optimum outside the box.
    assert!(binding >= 100, "only {binding} cases bind a bound");
}

#[derive(Debug, Deserialize)]
struct ScipyMinimum {
    success: bool,
    x: Vec<f64>,
    fun: f64,
}

fn scipy_bounded_rosenbrock_or_skip() -> Option<ScipyMinimum> {
    let script = r#"
import json
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

res = minimize(rosen, [-1.2, 1.0], method='L-BFGS-B', bounds=[(-2.0, 0.5), (-2.0, 0.5)],
               tol=1e-12, options={'maxiter': 5000})
print(json.dumps({"success": bool(res.success), "x": [float(v) for v in res.x],
                  "fun": float(res.fun)}))
"#;
    let output = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "python spawn failed: {e}"
            );
            return None;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bounded Rosenbrock oracle failed: {stderr}"
        );
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse the oracle's JSON"))
}

#[test]
fn diff_opt_minimize_casp_bounded_rosenbrock_scipy() {
    let rosen = |x: &[f64]| -> f64 { 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2) };
    let bounds: [Bound; 2] = [(Some(-2.0), Some(0.5)), (Some(-2.0), Some(0.5))];
    let options = MinimizeOptions {
        bounds: Some(&bounds),
        tol: Some(1e-12),
        maxiter: Some(5000),
        ..MinimizeOptions::default()
    };
    let mut portfolio = OptSolverPortfolio::new(RuntimeMode::Strict, 8);
    let out = minimize_with_casp(rosen, &[-1.2, 1.0], options, &mut portfolio)
        .expect("bounded Rosenbrock");
    println!(
        "fsci: action={:?} x={:?} fun={:?}",
        out.chosen_action, out.result.x, out.result.fun
    );
    assert!(bounded_methods(out.chosen_action));
    assert!(
        out.result.x.iter().all(|&v| (-2.0..=0.5).contains(&v)),
        "infeasible x = {:?}",
        out.result.x
    );

    let mut compared = 0;
    if let Some(scipy) = scipy_bounded_rosenbrock_or_skip() {
        println!(
            "scipy: success={} x={:?} fun={}",
            scipy.success, scipy.x, scipy.fun
        );
        assert!(scipy.success, "SciPy's own run failed");
        for (fsci, reference) in out.result.x.iter().zip(&scipy.x) {
            assert!(
                (fsci - reference).abs() <= SCIPY_X_ABS_TOL,
                "x = {:?}, SciPy {:?}",
                out.result.x,
                scipy.x
            );
        }
        compared += 1;
    }
    if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
        assert_eq!(compared, 1, "the live SciPy row was not compared");
    } else if compared == 0 {
        println!("SciPy unavailable: the live row was not compared");
    }
}
