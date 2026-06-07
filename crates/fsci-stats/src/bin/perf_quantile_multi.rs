//! Same-process A/B + isomorphism harness for the multi-quantile `quantile`.
//!
//! `naive` reproduces the original full-sort form; the library now partitions
//! per quantile when few. We prove the result vector is byte-identical
//! (`.to_bits()`) across q-counts (both gate paths) / ties / signed zeros, then
//! time the win. Run: `cargo run --release -p fsci-stats --bin perf_quantile_multi`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::quantile;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Verbatim original: full sort, evaluate each quantile.
fn naive(data: &[f64], q: &[f64]) -> Vec<f64> {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; q.len()];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    q.iter()
        .map(|&qi| {
            let qi = qi.clamp(0.0, 1.0);
            let idx = qi * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let frac = idx - lo as f64;
            if lo == hi || hi >= n {
                sorted[lo.min(n - 1)]
            } else {
                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
            }
        })
        .collect()
}

fn main() {
    let mut r = Lcg(0x51fe_a9c3_77b0_2d44);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    let q_sets: &[&[f64]] = &[
        &[0.5],
        &[0.25, 0.5, 0.75],
        &[0.0, 1.0],
        &[0.1, 0.333, 0.666, 0.9, 0.99],
        &[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    ];

    for trial in 0..3000 {
        let n = 1 + (r.next_u64() as usize % 400);
        let kind = trial % 3;
        let data: Vec<f64> = (0..n)
            .map(|_| match kind {
                0 => (r.next_u64() % 6) as f64,
                1 => r.unit() * 200.0 - 100.0,
                _ => {
                    let v = r.next_u64() % 3;
                    if v == 0 { -0.0 } else { (v as f64) - 1.0 }
                }
            })
            .collect();

        for &qs in q_sets {
            let g = quantile(&data, qs);
            let w = naive(&data, qs);
            total += 1;
            let eq =
                g.len() == w.len() && g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits());
            if !eq {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!("MISMATCH trial={trial} n={n} nq={}\n", qs.len()));
                }
            }
        }
        let g = quantile(&data, &[0.25, 0.5, 0.75]);
        let digest = g.iter().fold(1469598103934665603u64, |h, x| {
            (h ^ x.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} results (0 == byte-identical)");

    // ---- Timing: few quantiles (select path) and many (sort path) ----
    for &(n, nq) in &[(2_000_000usize, 3usize), (500_000, 3), (500_000, 40)] {
        let data: Vec<f64> = (0..n).map(|_| r.unit() * 1e6).collect();
        let qs: Vec<f64> = (0..nq).map(|i| (i as f64 + 0.5) / nq as f64).collect();

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += naive(&data, &qs)[0];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += quantile(&data, &qs)[0];
        }
        let new_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>9} nq={nq:>2}  sort={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            naive_t / 5,
            new_t / 5
        );
    }
}
