//! Diagnostic probe for frankenscipy-sez4r: which `(n, seed)` makes `eig` hang?
//!
//! WHY A SEPARATE PROCESS PER CASE. The failure mode is a NON-TERMINATING loop inside
//! nalgebra's `Schur::do_decompose` (`max_niter == 0` means "continue indefinitely"), and a
//! hung thread cannot be killed from inside Rust. So each case runs in its own process and
//! the OS `timeout` command is the only reliable executioner.
//!
//! Usage:  eig_sweep_probe <n> <seed> [max_niter]
//!
//! Without `max_niter`: calls `fsci_linalg::eig` exactly as callers do. Exit 0 = converged,
//! prints `OK n seed <residual>`. A timeout kill (124 from `timeout`) is the signal.
//!
//! With `max_niter`: calls nalgebra's `Schur::try_new` DIRECTLY with that bound, which is
//! the only way to ask the question the timeout cannot answer — does the iteration NEVER
//! converge, or does it converge after some huge but finite number of sweeps? A `timeout`
//! kill proves neither. Prints `CONV n seed K` or `NOCONV n seed K`.
//!
//! `make_diag_dominant` is copied VERBATIM from
//! `crates/fsci-conformance/src/metamorphic.rs` so the fixture is bit-identical to the one
//! the hung test generates. Do not "clean it up" — its exact arithmetic is the point.

use fsci_linalg::{DecompOptions, eig};

fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let r =
                ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000) as f64 / 1000.0;
            a[i][j] = if i == j {
                (n as f64) * 2.0 + r
            } else {
                r - 0.5
            };
        }
    }
    a
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: eig_sweep_probe <n> <seed> [max_niter]");
        std::process::exit(2);
    }
    let n: usize = args[1].parse().expect("n");
    let seed: u64 = args[2].parse().expect("seed");

    let a = make_diag_dominant(n, seed);
    let trace: f64 = (0..n).map(|i| a[i][i]).sum();

    // Bounded mode: ask nalgebra directly whether it converges within K sweeps.
    if let Some(k) = args.get(3) {
        let max_niter: usize = k.parse().expect("max_niter");
        let m = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| a[i][j]);
        let eps = f64::EPSILON;
        match nalgebra::linalg::Schur::try_new(m, eps, max_niter) {
            Some(_) => println!("CONV {n} {seed} {max_niter}"),
            None => println!("NOCONV {n} {seed} {max_niter}"),
        }
        return;
    }

    match eig(&a, DecompOptions::default()) {
        Ok(res) => {
            let s: f64 = res.eigenvalues_re.iter().sum();
            println!("OK {n} {seed} {:.3e}", (s - trace).abs());
        }
        Err(e) => {
            // An ERROR is a different outcome from a HANG and must not be conflated:
            // scipy converges on all 7000 of these fixtures, so an error here would be
            // its own defect.
            println!("ERR {n} {seed} {e:?}");
        }
    }
}
