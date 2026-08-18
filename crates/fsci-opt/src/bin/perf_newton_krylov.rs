//! Matrix-free Newton-Krylov against `scipy.optimize`'s `KrylovJacobian`.
//!
//! The headline number is a COUNT, not a time: for a matrix-free method every Krylov
//! product is one residual evaluation, and on the problems the method exists for that
//! evaluation dominates everything else. A count is also immune to the machine load
//! this box is under.
//!
//! Three arms share one outer Newton loop so the only difference is the inner solve:
//! ours with LGMRES augmentation, ours without it (the internal control that isolates
//! what the augmentation buys), and scipy's KrylovJacobian driven by the same loop in
//! the companion Python script.

use fsci_opt::{InnerMethod, KrylovJacobian};
use std::cell::Cell;
use std::time::Instant;

const MARKER: &str = "newton-krylov-bratu-v1";

/// 2D Bratu: `lap(u) + lambda * exp(u) = 0` on the unit square, zero Dirichlet
/// boundary, five-point Laplacian on an `m x m` interior grid. A standard
/// Newton-Krylov test problem: sparse, nonlinear, and genuinely coupled.
fn bratu_residual(u: &[f64], m: usize, lambda: f64) -> Vec<f64> {
    let h = 1.0 / (m as f64 + 1.0);
    let inv_h2 = 1.0 / (h * h);
    let at = |i: isize, j: isize| -> f64 {
        if i < 0 || j < 0 || i >= m as isize || j >= m as isize {
            0.0
        } else {
            u[i as usize * m + j as usize]
        }
    };
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let c = u[i * m + j];
            let lap = (at(i as isize + 1, j as isize)
                + at(i as isize - 1, j as isize)
                + at(i as isize, j as isize + 1)
                + at(i as isize, j as isize - 1)
                - 4.0 * c)
                * inv_h2;
            out[i * m + j] = lap + lambda * c.exp();
        }
    }
    out
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

struct Run {
    nfev: usize,
    newton_steps: usize,
    secs: f64,
    final_residual: f64,
    converged: bool,
}

fn run(m: usize, lambda: f64, method: InnerMethod, tol: f64, eta: f64, maxiter: usize) -> Run {
    let n = m * m;
    let counter = Cell::new(0usize);
    let f = |u: &[f64]| -> Vec<f64> {
        counter.set(counter.get() + 1);
        bratu_residual(u, m, lambda)
    };

    let mut x = vec![0.0; n];
    let mut fx = f(&x);
    let mut jac = KrylovJacobian::new(&f, None, 20, 10);
    jac.setup(&x, &fx);

    let t = Instant::now();
    let mut steps = 0usize;
    let mut converged = false;
    for _ in 0..maxiter {
        if norm2(&fx) < tol {
            converged = true;
            break;
        }
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = jac.solve(&rhs, eta, method);
        for (xi, di) in x.iter_mut().zip(&dx) {
            *xi += di;
        }
        fx = f(&x);
        jac.update(&x, &fx);
        steps += 1;
    }
    if norm2(&fx) < tol {
        converged = true;
    }
    Run {
        nfev: counter.get(),
        newton_steps: steps,
        secs: t.elapsed().as_secs_f64(),
        final_residual: norm2(&fx),
        converged,
    }
}

fn main() {
    println!("MARKER={MARKER}");
    let grids: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 32, 48]);
    let lambda = 1.0;
    let tol = 1e-8;
    let eta = 1e-3;
    let maxiter = 40;

    for &m in &grids {
        for (name, method) in [
            ("fsci_lgmres", InnerMethod::Lgmres),
            ("fsci_gmres", InnerMethod::Gmres),
        ] {
            let r = run(m, lambda, method, tol, eta, maxiter);
            println!(
                "m={m} n={} arm={name} nfev={} newton={} secs={:.6} resid={:.3e} converged={}",
                m * m,
                r.nfev,
                r.newton_steps,
                r.secs,
                r.final_residual,
                r.converged
            );
        }
    }
}
