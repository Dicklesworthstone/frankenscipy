//! Same-process A/B + isomorphism harness for median / percentile / iqr /
//! median_abs_deviation. The `naive_*` fns reproduce the original full-sort form;
//! the library now uses partial selection (select_nth). We prove every result is
//! byte-identical (`.to_bits()`) across random data incl. ties / signed zeros /
//! NaN, then time the win. Run: `cargo run --release -p fsci-stats --bin perf_quantile`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::{iqr, median, median_abs_deviation, percentile};
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

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn naive_median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n.is_multiple_of(2) {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    }
}

fn naive_percentile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() || q.is_nan() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let q_frac = (q / 100.0).clamp(0.0, 1.0);
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&s, q_frac)
}

fn naive_iqr(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25)
}

fn naive_mad(data: &[f64], scale: f64) -> f64 {
    if data.is_empty() || scale == 0.0 {
        return f64::NAN;
    }
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let med = quantile_sorted(&s, 0.5);
    let mut diffs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    diffs.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&diffs, 0.5) / scale
}

fn eqb(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn main() {
    let mut r = Lcg(0x9c2e_71ab_03d5_f481);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..2000 {
        let n = 1 + (r.next_u64() as usize % 300);
        let kind = trial % 4;
        let data: Vec<f64> = (0..n)
            .map(|_| match kind {
                0 => (r.next_u64() % 5) as f64, // heavy ties
                1 => r.unit() * 100.0 - 50.0,   // fine
                2 => {
                    let v = r.next_u64() % 3;
                    if v == 0 { -0.0 } else { (v as f64) - 1.0 } // signed zeros
                }
                _ => {
                    // occasionally inject NaN (median/percentile-with-nan paths)
                    if r.unit() < 0.1 {
                        f64::NAN
                    } else {
                        r.unit() * 10.0
                    }
                }
            })
            .collect();
        let has_nan = data.iter().any(|v| v.is_nan());

        for &q in &[0.0, 1.0, 25.0, 50.0, 73.3, 99.9, 100.0] {
            let g = percentile(&data, q);
            let w = naive_percentile(&data, q);
            total += 1;
            if !eqb(g, w) {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!("MISMATCH percentile trial={trial} n={n} q={q}\n"));
                }
            }
        }
        for (name, g, w) in [
            ("median", median(&data), naive_median(&data)),
            ("iqr", iqr(&data), naive_iqr(&data)),
            (
                "mad",
                median_abs_deviation(&data, 1.4826),
                naive_mad(&data, 1.4826),
            ),
        ] {
            total += 1;
            if !eqb(g, w) {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!("MISMATCH {name} trial={trial} n={n}\n"));
                }
            }
        }
        // golden digest over the non-NaN median+iqr
        if !has_nan {
            let dg = median(&data).to_bits() ^ iqr(&data).to_bits().rotate_left(1);
            payload.push_str(&format!("trial={trial} n={n} dg={dg:016x}\n"));
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} results (0 == byte-identical)");

    // ---- Timing: median + percentile on large arrays (sort dominates) ----
    for &n in &[100_000usize, 500_000, 2_000_000] {
        let data: Vec<f64> = (0..n).map(|_| r.unit() * 1e6).collect();

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..5 {
            acc += naive_median(&data) + naive_percentile(&data, 90.0);
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..5 {
            acc += median(&data) + percentile(&data, 90.0);
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
