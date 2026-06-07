//! Same-process A/B for the convolve FFT-threshold fix.
//!
//! In the medium regime (1000 < na·nb < ~2.1e5) the old `na·nb > 1000` gate forced
//! the slower FFT path; the cost-model gate now keeps it on the direct loop, which
//! is faster AND byte-identical to a verbatim direct convolution. This bin proves
//! both: bit-equality vs direct, and the speedup vs the old forced-FFT path
//! (fftconvolve).

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::{ConvolveMode, convolve, fftconvolve};

fn direct_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut full = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            full[i + j] += ai * bj;
        }
    }
    full
}

fn det(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.41 + seed).sin() + 0.3)
        .collect()
}

fn time_it(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    // Balanced sizes spanning the medium (was-FFT, now-direct) regime and beyond.
    for &n in &[40usize, 64, 100, 160, 256, 384, 512] {
        let a = det(n, 0.3);
        let b = det(n, 1.1);

        // Parity: shipped convolve vs verbatim direct (Full).
        let got = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        let want = direct_full(&a, &b);
        assert_eq!(got.len(), want.len());
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (g - w).abs())
            .fold(0.0_f64, f64::max);
        let exact = got
            .iter()
            .zip(&want)
            .all(|(&g, &w)| g.to_bits() == w.to_bits());

        let iters = (4_000_000 / (n * n + 1)).clamp(20, 5000);
        let now = time_it(iters, || {
            convolve(&a, &b, ConvolveMode::Full).expect("convolve")
        });
        let old = time_it(iters, || {
            fftconvolve(&a, &b, ConvolveMode::Full).expect("fft")
        });
        println!(
            "n={n:>4} (na*nb={:>7}): convolve={now:>8.3}us  forced-fft={old:>8.3}us  speedup={:>5.2}x  exact_vs_direct={exact} max_abs={max_abs:e}",
            n * n,
            old / now,
        );
    }
}
