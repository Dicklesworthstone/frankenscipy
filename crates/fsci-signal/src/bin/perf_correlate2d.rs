//! Same-process A/B timing + tolerance proof for the correlate2d FFT lever.
//!
//! Times the shipped `correlate2d` (FFT path for large inputs) against a verbatim
//! direct O(ar·ac·vr·vc) 2D correlation, and reports max relative error. Avoids
//! the criterion harness (whose shared dev-dep build cache is frequently skewed).

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::{ConvolveMode, correlate2d};

/// Verbatim direct 2D correlation (full mode), the pre-lever reference.
fn direct_full(a: &[f64], ar: usize, ac: usize, v: &[f64], vr: usize, vc: usize) -> Vec<f64> {
    let mut v_rev = vec![0.0; vr * vc];
    for i in 0..vr {
        for j in 0..vc {
            v_rev[i * vc + j] = v[(vr - 1 - i) * vc + (vc - 1 - j)];
        }
    }
    let full_r = ar + vr - 1;
    let full_c = ac + vc - 1;
    let mut full = vec![0.0; full_r * full_c];
    for i in 0..ar {
        for j in 0..ac {
            let aval = a[i * ac + j];
            for ki in 0..vr {
                for kj in 0..vc {
                    full[(i + ki) * full_c + (j + kj)] += aval * v_rev[ki * vc + kj];
                }
            }
        }
    }
    full
}

fn det_grid(rows: usize, cols: usize, seed: f64) -> Vec<f64> {
    (0..rows * cols)
        .map(|k| {
            let i = (k / cols) as f64;
            let j = (k % cols) as f64;
            (i * 0.13 + j * 0.27 + seed).sin() + 0.5 * (j * 0.05 - i * 0.02).cos()
        })
        .collect()
}

fn time_it(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    for _ in 0..2 {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    // (image side, kernel side)
    let cases = [
        (64usize, 8usize),
        (128, 16),
        (256, 16),
        (512, 32),
        (256, 64),
        (256, 128),
        (384, 96),
        (200, 200),
    ];
    for (n, k) in cases {
        let a = det_grid(n, n, 0.3);
        let v = det_grid(k, k, 1.1);

        // Parity: shipped (FFT path) vs verbatim direct, full mode.
        let got = correlate2d(&a, (n, n), &v, (k, k), ConvolveMode::Full).expect("correlate2d");
        let want = direct_full(&a, n, n, &v, k, k);
        assert_eq!(got.len(), want.len());
        let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
        for (&g, &w) in got.iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
            if w.abs() > 1e-9 {
                max_rel = max_rel.max((g - w).abs() / w.abs());
            }
        }

        let iters = (40_000_000 / (n * n * k * k + 1)).clamp(3, 200);
        let after = time_it(iters, || {
            correlate2d(&a, (n, n), &v, (k, k), ConvolveMode::Full).expect("correlate2d")
        });
        let before = time_it(iters, || direct_full(&a, n, n, &v, k, k));
        println!(
            "img={n:>4} ker={k:>3}: direct={before:>10.3}ms  fft={after:>9.3}ms  speedup={:>6.2}x  max_abs={max_abs:e} max_rel={max_rel:e}",
            before / after
        );
    }
}
