//! Same-process A/B + isomorphism harness for `spectral_contrast`.
//!
//! `old_spectral_contrast` is a verbatim copy of the original per-band full sort.
//! The library now derives each band's peak/valley with a single total_cmp pass
//! (no allocation), bit-identical to sorting then reading [len-1]/[0]. We assert
//! 0 mismatches across sizes/band counts/tie densities and time the win.
//! Run: `cargo run --release -p fsci-signal --bin perf_spectral_contrast`.

use fsci_signal::spectral_contrast;
use std::time::Instant;

fn old_spectral_contrast(magnitudes: &[f64], n_bands: usize) -> Vec<f64> {
    if magnitudes.is_empty() || n_bands == 0 {
        return vec![];
    }
    let n = magnitudes.len();
    let band_size = n / n_bands;
    if band_size == 0 {
        return vec![0.0; n_bands];
    }
    (0..n_bands)
        .map(|b| {
            let start = b * band_size;
            let end = ((b + 1) * band_size).min(n);
            let band = &magnitudes[start..end];
            if band.is_empty() {
                return 0.0;
            }
            let mut sorted = band.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let peak = sorted[sorted.len() - 1];
            let valley = sorted[0];
            if valley > 0.0 {
                (peak / valley).log10() * 20.0
            } else {
                0.0
            }
        })
        .collect()
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn make_mags(n: usize, grid: u64, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let v = rng.next_f64() * 4.0;
            if grid == 0 {
                v
            } else {
                (v * grid as f64).round()
            }
        })
        .collect()
}

fn main() {
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[0usize, 1, 2, 7, 64, 513, 4096] {
        for &nb in &[1usize, 2, 6, 12, 64] {
            for &grid in &[0u64, 2, 5] {
                for seed in 0..3u64 {
                    let m = make_mags(n, grid, seed * 911 + 1);
                    let got = spectral_contrast(&m, nb);
                    let want = old_spectral_contrast(&m, nb);
                    total += 1;
                    let ok = got.len() == want.len()
                        && got
                            .iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    if !ok {
                        mismatches += 1;
                        if payload.len() < 1500 {
                            payload.push_str(&format!(
                                "MISMATCH n={n} nb={nb} grid={grid} seed={seed}\n"
                            ));
                        }
                    }
                    if payload.len() < 1500 {
                        let chk = got.iter().map(|v| v.to_bits()).fold(0u64, |a, b| a ^ b);
                        payload.push_str(&format!(
                            "n={n} nb={nb} grid={grid} seed={seed} chk={chk:016x}\n"
                        ));
                    }
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} cases (0 == byte-identical)");

    // ---- Timing: many frames, few bands (sort dominates) ----
    for &(n, nb) in &[(2048usize, 6usize), (4096, 7), (8192, 8)] {
        let m = make_mags(n, 0, 7);
        let frames = 2000;

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..frames {
            acc += old_spectral_contrast(&m, nb).iter().sum::<f64>();
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..frames {
            acc += spectral_contrast(&m, nb).iter().sum::<f64>();
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>5} nb={nb:>3} x{frames}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>6.1}x  (acc={acc:.3})",
            old_t / frames,
            new_t / frames
        );
    }
}
