//! Timing harness for the parallel separable filters (correlate1d / uniform_filter).
//! Run before and after the parallelization (via `git stash`) to get the speedup;
//! byte identity is proven by the in-crate `*_parallel_is_bit_identical` tests.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, correlate1d_with_origin, uniform_filter};

fn image(rows: usize, cols: usize) -> NdArray {
    let data: Vec<f64> = (0..rows * cols)
        .map(|k| ((k % 251) as f64 * 0.013).sin() + 0.5)
        .collect();
    NdArray::new(data, vec![rows, cols]).expect("image")
}

fn time_it(iters: usize, mut f: impl FnMut() -> NdArray) -> f64 {
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    let mode = BoundaryMode::Reflect;
    for &(n, ksz) in &[(800usize, 9usize), (1200, 21)] {
        let img = image(n, n);
        let weights: Vec<f64> = (0..ksz).map(|k| ((k as f64) * 0.3).cos() + 1.0).collect();
        let iters = (200_000_000 / (n * n + 1)).clamp(2, 50);

        let c1d = time_it(iters, || {
            correlate1d_with_origin(&img, &weights, 1, mode, 0.0, 0).expect("c1d")
        });
        let uf = time_it(iters, || uniform_filter(&img, ksz, mode, 0.0).expect("uf"));
        println!("{n}x{n} k={ksz}: correlate1d={c1d:>8.3}ms  uniform_filter={uf:>8.3}ms");
    }
}
