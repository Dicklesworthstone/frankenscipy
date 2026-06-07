//! Same-process A/B + isomorphism harness for scoreatpercentile.
//!
//! `naive` reproduces the original full-sort form; the library now partitions
//! per percentile when few. We prove the result vector is byte-identical
//! (`.to_bits()`) across methods / per-counts / limits / ties / signed zeros,
//! then time the win. Run: `cargo run --release -p fsci-stats --bin perf_scoreperc`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::scoreatpercentile;
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

fn quantile_single(sorted: &[f64], per: f64, method: &str) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = per / 100.0 * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    match method {
        "fraction" => {
            let frac = idx - lower as f64;
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        }
        "lower" => sorted[lower],
        _ => sorted[upper],
    }
}

/// Verbatim original: filter by limit, full sort, evaluate each percentile.
fn naive(data: &[f64], per: &[f64], limit: Option<(f64, f64)>, method: &str) -> Vec<f64> {
    if data.is_empty() {
        return vec![f64::NAN; per.len()];
    }
    let mut filtered: Vec<f64> = match limit {
        Some((lo, hi)) => data
            .iter()
            .copied()
            .filter(|v| lo <= *v && *v <= hi)
            .collect(),
        None => data.to_vec(),
    };
    if filtered.is_empty() || filtered.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; per.len()];
    }
    filtered.sort_by(|a, b| a.total_cmp(b));
    per.iter()
        .map(|&p| quantile_single(&filtered, p, method))
        .collect()
}

fn main() {
    let mut r = Lcg(0x2bd9_6c41_e07f_a358);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    let per_sets: &[&[f64]] = &[
        &[50.0],
        &[25.0, 50.0, 75.0],
        &[0.0, 100.0],
        &[10.0, 33.3, 66.6, 90.0, 99.0],
        &[5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0],
    ];
    let methods = ["fraction", "lower", "higher"];

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
        let limit = if trial % 4 == 0 {
            Some((-10.0, 50.0))
        } else {
            None
        };

        for &per in per_sets {
            for &m in &methods {
                let g = scoreatpercentile(&data, per, limit, Some(m)).unwrap();
                let w = naive(&data, per, limit, m);
                total += 1;
                let eq =
                    g.len() == w.len() && g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits());
                if !eq {
                    mismatches += 1;
                    if payload.len() < 3000 {
                        payload.push_str(&format!(
                            "MISMATCH trial={trial} n={n} m={m} np={}\n",
                            per.len()
                        ));
                    }
                }
            }
        }
        let g = scoreatpercentile(&data, &[25.0, 75.0], None, None).unwrap();
        let digest = g.iter().fold(1469598103934665603u64, |h, x| {
            (h ^ x.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} results (0 == byte-identical)");

    // ---- Timing: large arrays, few percentiles (select path), and many (sort path) ----
    for &(n, np) in &[(2_000_000usize, 3usize), (500_000, 3), (500_000, 40)] {
        let data: Vec<f64> = (0..n).map(|_| r.unit() * 1e6).collect();
        let per: Vec<f64> = (0..np)
            .map(|i| (i as f64 + 0.5) * 100.0 / np as f64)
            .collect();

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += naive(&data, &per, None, "fraction")[0];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += scoreatpercentile(&data, &per, None, Some("fraction")).unwrap()[0];
        }
        let new_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>9} np={np:>2}  sort={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            naive_t / 5,
            new_t / 5
        );
    }
}
