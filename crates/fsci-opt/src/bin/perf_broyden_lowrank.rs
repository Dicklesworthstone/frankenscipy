//! Head-to-head: limited-memory Broyden Jacobian against `scipy.optimize.BroydenFirst`.
//!
//! Both arms run the SAME algorithm -- scipy's BroydenFirst is itself low-rank -- so
//! this measures our constant factor against the incumbent's, not a complexity
//! difference. The dense `broyden1` already in `root.rs` is timed as a third arm
//! because that is where the complexity claim actually lives.
//!
//! Prints a source-derived marker so a stale binary cannot be mistaken for a fresh one.

use fsci_opt::{BroydenJacobian, BroydenVariant, Jacobian, ReductionMethod};
use std::time::Instant;

/// Bump when the measured loop changes; printed so the log identifies the source.
const MARKER: &str = "broyden-lowrank-v1-steps30";
const STEPS: usize = 30;

/// A cheap O(n) nonlinear residual. Deliberately not separable -- the coupling term
/// stops the Jacobian being diagonal, which would make every method look identical.
fn residual(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| x[i] * x[i] - ((i % 7) as f64) - 1.0 + 0.1 * x[(i + 1) % n])
        .collect()
}

fn start_point(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + 0.01 * ((i % 13) as f64)).collect()
}

fn time_lowrank(n: usize) -> (f64, f64) {
    let x0 = start_point(n);
    let f0 = residual(&x0);
    let mut jac = BroydenJacobian::new(BroydenVariant::First, None, ReductionMethod::Restart, None);
    jac.setup(&x0, &f0);
    let mut x = x0;
    let mut f = f0;

    let t = Instant::now();
    for _ in 0..STEPS {
        let dir = jac.solve(&f);
        for (xi, di) in x.iter_mut().zip(&dir) {
            *xi -= di;
        }
        f = residual(&x);
        jac.update(&x, &f);
    }
    let secs = t.elapsed().as_secs_f64();
    let resid = f.iter().map(|v| v * v).sum::<f64>().sqrt();
    (secs, resid)
}

/// The dense arm: the same update rule stored as a full `n x n` inverse Jacobian, which
/// is what `root.rs`'s `broyden1` does internally. Written out here so both arms run the
/// identical step sequence and the only difference is the representation.
fn time_dense(n: usize) -> (f64, f64) {
    let x0 = start_point(n);
    let f0 = residual(&x0);
    let normf0 = f0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let normx0 = x0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let alpha = if normf0 != 0.0 {
        0.5 * normx0.max(1.0) / normf0
    } else {
        1.0
    };
    // Gm = -alpha * I, dense.
    let mut gm = vec![0.0; n * n];
    for i in 0..n {
        gm[i * n + i] = -alpha;
    }
    let mut x = x0;
    let mut f = f0;

    let t = Instant::now();
    for _ in 0..STEPS {
        // dir = Gm f
        let dir: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| gm[i * n + j] * f[j]).sum())
            .collect();
        let last_f = f.clone();
        let mut dx = vec![0.0; n];
        for i in 0..n {
            dx[i] = -dir[i];
            x[i] += dx[i];
        }
        f = residual(&x);
        let df: Vec<f64> = f.iter().zip(&last_f).map(|(a, b)| a - b).collect();

        // Broyden good: Gm += (dx - Gm df) (Gm^T dx)^T / (df . Gm^T dx)
        let gm_df: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| gm[i * n + j] * df[j]).sum())
            .collect();
        let v: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| gm[i * n + j] * dx[i]).sum())
            .collect();
        let denom: f64 = df.iter().zip(&v).map(|(a, b)| a * b).sum();
        if denom == 0.0 || !denom.is_finite() {
            continue;
        }
        let c: Vec<f64> = dx.iter().zip(&gm_df).map(|(a, b)| a - b).collect();
        for i in 0..n {
            let ci = c[i];
            if ci == 0.0 {
                continue;
            }
            for j in 0..n {
                gm[i * n + j] += ci * v[j] / denom;
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let resid = f.iter().map(|v| v * v).sum::<f64>().sqrt();
    (secs, resid)
}

fn main() {
    println!("MARKER={MARKER} STEPS={STEPS}");
    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 256, 1024, 4096]);
    let dense_limit: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    for &n in &sizes {
        let (lr, lr_res) = time_lowrank(n);
        println!("n={n} arm=fsci_lowrank secs={lr:.6} resid={lr_res:.6e}");
        if n <= dense_limit {
            let (d, d_res) = time_dense(n);
            println!("n={n} arm=fsci_dense secs={d:.6} resid={d_res:.6e}");
        }
    }
}
