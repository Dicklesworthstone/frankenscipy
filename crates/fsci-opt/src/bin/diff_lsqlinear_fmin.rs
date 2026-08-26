//! Differential oracle probe for `scipy.optimize` entry points with NO differential coverage
//! (frankenscipy-ivxx6): `lsq_linear`, `fmin_bfgs`, `fmin_powell`, `fmin_cg`, `fmin_ncg`.
//!
//! Each was confirmed at zero referencing files from
//! `grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin/diff_*.rs`
//! across the FULL corpus -- both locations, since differential tests are split between them.
//!
//! `quadratic_assignment` is deliberately NOT here. SciPy's default `faq` method is a randomised
//! heuristic, so comparing permutations between two implementations would be comparing two draws
//! rather than two answers; a divergence would mean nothing and an agreement would be luck.
//!
//! FIXTURE NOTES.
//!   * `lsq_linear` is probed with bounds that ACTIVELY BIND and, as a control, with bounds so
//!     wide they cannot. The unbounded arm must reproduce the ordinary least-squares solution --
//!     if it does not, the bounded arm is measuring a broken solver rather than bound handling.
//!     The system is overdetermined and not square, since that is the case the function exists for.
//!   * the minimisers are compared on their CONVERGED LOCATION, not iteration-by-iteration. Two
//!     different descent algorithms will not trace the same path, but on a smooth convex problem
//!     they must reach the same minimiser. Rosenbrock is included because it is the standard
//!     stress case for exactly these methods.
//!
//! Lines: `name|v;v;v`. Inputs must match `python/diff_lsqlinear_fmin.py`.
use fsci_opt::{fmin_bfgs, fmin_cg, fmin_powell, lsq_linear};

fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}

fn main() {
    // Overdetermined 5x3 system.
    let a: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5, -0.25],
        vec![0.0, 2.0, 1.0],
        vec![3.0, -1.0, 0.5],
        vec![-1.0, 1.5, 2.0],
        vec![0.5, 0.5, -3.0],
    ];
    let b = vec![1.0, -2.0, 3.0, 0.5, -1.5];

    // Control: bounds so wide they cannot bind -> must equal ordinary least squares.
    let wide_lo = vec![-1.0e6; 3];
    let wide_hi = vec![1.0e6; 3];
    if let Ok(x) = lsq_linear(&a, &b, &wide_lo, &wide_hi) {
        dump("lsq_unbounded", &x);
    }

    // Bounds chosen to BIND: the unconstrained solution lies outside them.
    let lo = vec![-0.2, -0.5, 0.0];
    let hi = vec![0.5, 0.4, 1.0];
    if let Ok(x) = lsq_linear(&a, &b, &lo, &hi) {
        dump("lsq_bounded", &x);
    }

    // ---- minimisers -------------------------------------------------------------------------
    // Smooth convex quadratic with an exact minimiser at (1, -2, 0.5).
    let quad =
        |x: &[f64]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2) + 3.0 * (x[2] - 0.5).powi(2);
    // Rosenbrock: the standard stress case for these methods.
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);

    let q0 = [0.0, 0.0, 0.0];
    let r0 = [-1.2, 1.0];

    if let Ok(x) = fmin_bfgs(quad, &q0) {
        dump("bfgs_quad", &x);
    }
    if let Ok(x) = fmin_powell(quad, &q0) {
        dump("powell_quad", &x);
    }
    if let Ok(x) = fmin_cg(quad, &q0) {
        dump("cg_quad", &x);
    }
    if let Ok(x) = fmin_bfgs(rosen, &r0) {
        dump("bfgs_rosen", &x);
    }
    if let Ok(x) = fmin_powell(rosen, &r0) {
        dump("powell_rosen", &x);
    }
}
