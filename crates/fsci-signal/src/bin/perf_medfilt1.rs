//! Same-process A/B + isomorphism harness for `medfilt1`.
//!
//! `old_medfilt1` is a verbatim copy of the original per-window-sort filter. The
//! library now routes large kernels through a sliding ordered-multiset median
//! (clipped windows, per-step rank = len/2), which is bit-identical: each output
//! is the rank-len/2 element of the same clipped window. We assert 0 mismatches
//! across sizes/kernels/tie densities and time the win.
//! Run: `cargo run --release -p fsci-signal --bin perf_medfilt1`.

use fsci_signal::medfilt1;
use std::time::Instant;

/// Verbatim copy of the original O(n*k log k) medfilt1.
fn old_medfilt1(x: &[f64], kernel_size: usize) -> Vec<f64> {
    if x.is_empty() || kernel_size == 0 {
        return x.to_vec();
    }
    let half = kernel_size / 2;
    let n = x.len();
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let mut window: Vec<f64> = x[start..end].to_vec();
            window.sort_by(|a, b| a.total_cmp(b));
            window[window.len() / 2]
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

/// `grid` controls tie density: values snapped to a `grid`-step lattice create
/// repeats and signed zeros (grid=0 => continuous).
fn make_signal(n: usize, grid: u64, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            let v = rng.next_f64() * 2.0 - 1.0;
            if grid == 0 {
                v
            } else {
                (v * grid as f64).round() / grid as f64
            }
        })
        .collect()
}

fn main() {
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[1usize, 2, 8, 33, 100, 512] {
        for &k in &[1usize, 2, 3, 8, 32, 33, 64, 65, 129] {
            for &grid in &[0u64, 2, 5] {
                for seed in 0..3u64 {
                    let x = make_signal(n, grid, seed * 131 + 1);
                    let got = medfilt1(&x, k);
                    let want = old_medfilt1(&x, k);
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
                                "MISMATCH n={n} k={k} grid={grid} seed={seed}\n"
                            ));
                        }
                    }
                    if payload.len() < 1500 {
                        let chk: u64 = got.iter().map(|v| v.to_bits()).fold(0u64, |a, b| a ^ b);
                        payload.push_str(&format!(
                            "n={n} k={k} grid={grid} seed={seed} chk={chk:016x}\n"
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

    // ---- Timing: large signal, large kernels (per-window sort dominates) ----
    for &(n, k) in &[(8192usize, 65usize), (8192, 257), (8192, 513)] {
        let x = make_signal(n, 0, 9);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += old_medfilt1(&x, k).iter().sum::<f64>();
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += medfilt1(&x, k).iter().sum::<f64>();
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>5} k={k:>4}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>6.1}x  (acc={acc:.3})",
            old_t / 5,
            new_t / 5
        );
    }
}
