//! Same-process A/B timing + tolerance proof for the polymul FFT lever.
//!
//! Avoids the criterion harness (whose dev-deps share a flaky build cache) by
//! timing the shipped `polymul` (FFT path for large inputs) against a verbatim
//! direct O(m·n) convolution, on the bench's 512x512 case.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::polymul;

fn direct(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] += ai * bj;
        }
    }
    r
}

fn main() {
    for &n in &[256usize, 512, 1024, 2048, 4096, 8192] {
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + 0.3).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).cos() - 0.2).collect();

        // Tolerance proof: shipped polymul (FFT path) vs verbatim direct.
        let got = polymul(&a, &b);
        let want = direct(&a, &b);
        assert_eq!(got.len(), want.len());
        let max_rel = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| {
                if w.abs() > 1e-12 {
                    (g - w).abs() / w.abs()
                } else {
                    0.0
                }
            })
            .fold(0.0_f64, f64::max);

        let iters = (2_000_000 / (n + 1)).max(20);
        let after = time_it_q(iters, || polymul(&a, &b));
        let before = time_it_q(iters, || direct(&a, &b));
        println!(
            "n={n:>5}: direct={before:>9.3}us  fft={after:>9.3}us  speedup={:>5.2}x  max_rel={max_rel:e}",
            before / after
        );
    }
}

fn time_it_q(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}
