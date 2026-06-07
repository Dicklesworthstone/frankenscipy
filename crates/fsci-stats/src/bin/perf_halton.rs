//! Same-process A/B + isomorphism harness for HaltonSampler::sample (general path).
//!
//! `old_radical` reproduces the original runtime-prime radical inverse; the
//! library now dispatches bundled primes to const-generic specialisations
//! (division strength-reduction). We prove the emitted points are byte-identical
//! (`.to_bits()`), then time the win.
//! Run: `cargo run --release -p fsci-stats --bin perf_halton`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::qmc::HaltonSampler;
use std::time::Instant;

const HALTON_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Verbatim copy of the original runtime-prime radical inverse.
fn old_radical(mut index: u64, prime: u64) -> f64 {
    let inv_prime = 1.0_f64 / prime as f64;
    let mut f = inv_prime;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % prime;
        result += digit as f64 * f;
        index /= prime;
        f *= inv_prime;
    }
    result
}

fn old_sample(d: usize, start: u64, n: usize) -> Vec<f64> {
    let primes = &HALTON_PRIMES[..d];
    let mut out = Vec::with_capacity(n * d);
    let mut idx = start;
    for _ in 0..n {
        for &p in primes {
            out.push(old_radical(idx, p));
        }
        idx = idx.saturating_add(1);
    }
    out
}

fn vec_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() {
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    // Cover dimensions both hitting the general path and the [2,3,5,7] sample_4d
    // special case (d=4), plus start/continuation sequencing.
    for &d in &[1usize, 2, 3, 4, 5, 8, 12, 20, 32] {
        for &(a, b) in &[
            (0usize, 64usize),
            (1, 100),
            (7, 300),
            (1000, 200),
            (0, 5000),
        ] {
            let mut s = HaltonSampler::new(d).unwrap();
            let got_a = s.sample(a);
            let got_b = s.sample(b);
            let want_a = old_sample(d, 0, a);
            let want_b = old_sample(d, a as u64, b);
            total += 2;
            if !vec_eq(&got_a, &want_a) {
                mismatches += 1;
                if payload.len() < 2000 {
                    payload.push_str(&format!("MISMATCH d={d} call=a a={a}\n"));
                }
            }
            if !vec_eq(&got_b, &want_b) {
                mismatches += 1;
                if payload.len() < 2000 {
                    payload.push_str(&format!("MISMATCH d={d} call=b a={a} b={b}\n"));
                }
            }
            let digest: u64 = got_a
                .iter()
                .chain(&got_b)
                .fold(1469598103934665603u64, |h, v| {
                    (h ^ v.to_bits()).wrapping_mul(1099511628211)
                });
            payload.push_str(&format!("d={d} a={a} b={b} digest={digest:016x}\n"));
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} sample blocks (0 == byte-identical)");

    // ---- Timing: general path (d != 4), many samples ----
    for &(d, n) in &[(8usize, 200_000usize), (16, 120_000), (32, 60_000)] {
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_sample(d, 0, n)[0];
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            let mut s = HaltonSampler::new(d).unwrap();
            acc += s.sample(n)[0];
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "d={d:>3} n={n:>6}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            old_t / 3,
            new_t / 3
        );
    }
}
