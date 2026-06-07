//! Same-process A/B + isomorphism harness for winsorize.
//!
//! `naive` reproduces the original full-sort form; the library now selects the
//! two cutoff ranks via select_nth. We prove the winsorized array is byte-identical
//! (`.to_bits()`) across random data incl. ties / signed zeros and a range of
//! limits, then time the win. Run: `cargo run --release -p fsci-stats --bin perf_winsorize`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::winsorize;
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

/// Verbatim original: full sort, read the two cutoff ranks, clamp.
fn naive(data: &[f64], limits: (f64, f64)) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let lo_idx = (limits.0.clamp(0.0, 1.0) * n as f64) as usize;
    let upidx = n.saturating_sub((limits.1.clamp(0.0, 1.0) * n as f64) as usize);
    let hi_idx = if upidx == 0 { 0 } else { upidx - 1 };
    let lo_idx = lo_idx.min(n - 1);
    let hi_idx = hi_idx.min(n - 1);
    let (lo_val, hi_val) = if lo_idx > hi_idx {
        let collapsed = sorted[lo_idx];
        (collapsed, collapsed)
    } else {
        (sorted[lo_idx], sorted[hi_idx])
    };
    data.iter().map(|&x| x.clamp(lo_val, hi_val)).collect()
}

fn vecs_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() {
    let mut r = Lcg(0x4d27_be91_05a3_cf68);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    let limit_sets: &[(f64, f64)] = &[
        (0.05, 0.05),
        (0.1, 0.2),
        (0.25, 0.25),
        (0.0, 0.1),
        (0.4, 0.4),
        (0.5, 0.5), // overlapping windows -> collapse path
        (0.0, 0.0),
    ];

    for trial in 0..3000 {
        let n = 1 + (r.next_u64() as usize % 400);
        let kind = trial % 3;
        let data: Vec<f64> = (0..n)
            .map(|_| match kind {
                0 => (r.next_u64() % 6) as f64, // heavy ties
                1 => r.unit() * 200.0 - 100.0,  // fine
                _ => {
                    let v = r.next_u64() % 3;
                    if v == 0 { -0.0 } else { (v as f64) - 1.0 } // signed zeros
                }
            })
            .collect();

        for &lim in limit_sets {
            let g = winsorize(&data, lim);
            let w = naive(&data, lim);
            total += 1;
            if !vecs_eq(&g, &w) {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!("MISMATCH trial={trial} n={n} lim={lim:?}\n"));
                }
            }
        }
        let g = winsorize(&data, (0.1, 0.1));
        let digest: u64 = g.iter().fold(1469598103934665603u64, |h, x| {
            (h ^ x.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism: {mismatches} mismatches / {total} winsorize results (0 == byte-identical)"
    );

    // ---- Timing: large arrays (sort dominates) ----
    for &n in &[100_000usize, 500_000, 2_000_000] {
        let data: Vec<f64> = (0..n).map(|_| r.unit() * 1e6).collect();
        let lim = (0.05, 0.05);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += naive(&data, lim)[0];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += winsorize(&data, lim)[0];
        }
        let new_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>9}  sort={:>10.3?}  select={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            naive_t / 5,
            new_t / 5
        );
    }
}
