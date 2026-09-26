//! Same-process timing + bit-identity digest harness for the smoothing spline
//! (`UnivariateSpline::new` with s > 0).
//!
//! `UnivariateSpline::new` with s > 0 now runs the FITPACK `curfit` port (SciPy's knot-adding
//! fit). The penalized-fit `make_smoothing_spline_impl` this harness was written for (its
//! O(n^2) -> O(n*k^2) assembly lever, tests/artifacts/perf/2026-06-05-interpolate-smoothing-
//! banded-solve) no longer exists, so digests from before that change are not comparable. It
//! still dumps an FNV digest of the spline coefficients and times the construction.
//! Run: `cargo run -p fsci-interpolate --bin perf_smoothing_spline`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::UnivariateSpline;

fn problem(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (1.3 * t).sin() + 0.5 * (0.4 + 0.7 * t).cos())
        .collect();
    (x, y)
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let sizes = [800usize, 1500, 3000];
    let s = 0.5;

    // Probe grid: eval_many is a deterministic function of the coefficients, so an
    // identical digest here means byte-identical coefficients.
    let probe: Vec<f64> = (0..401).map(|i| i as f64 / 400.0 * 10.0).collect();

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let (x, y) = problem(n);
        let sp = UnivariateSpline::new(&x, &y, s).expect("UnivariateSpline");
        println!("n={n} digest={:016x}", digest(&sp.eval_many(&probe)));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let (x, y) = problem(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let sp = UnivariateSpline::new(black_box(&x), black_box(&y), s).expect("spline");
            acc += sp.eval(black_box(5.0));
        }
        let dt = t0.elapsed();
        println!("n={n:>5}  {:>10.3?}/build  (acc={acc:.6})", dt / reps);
    }
}
