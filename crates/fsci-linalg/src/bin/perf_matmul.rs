//! Same-process timing for the multithreaded flat-workspace GEMM.
//!
//! Times the public `matmul` at large square sizes. Run on the same code before
//! and after the parallelization (via `git stash`) to get the speedup; byte
//! identity is proven separately by the in-crate row-split + golden tests.

use std::hint::black_box;
use std::time::Instant;

use fsci_linalg::matmul;

fn make(n: usize, seed: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| ((i as f64 * 0.0131 + j as f64 * 0.0071 + seed).sin()) + 0.5)
                .collect()
        })
        .collect()
}

fn main() {
    for &n in &[1024usize, 2048, 4096] {
        let a = make(n, 0.3);
        let b = make(n, 1.1);
        let iters = if n <= 1024 { 5 } else { 2 };
        // warm up
        black_box(matmul(&a, &b).expect("matmul"));
        let start = Instant::now();
        for _ in 0..iters {
            black_box(matmul(&a, &b).expect("matmul"));
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let gflops = 2.0 * (n as f64).powi(3) / (ms * 1e-3) / 1e9;
        println!("n={n:>5}: {ms:>10.3} ms/matmul  {gflops:>7.1} GFLOP/s");
    }
}
