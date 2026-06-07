//! Same-process A/B + isomorphism harness for SobolSampler::sample (general path).
//!
//! `old_sample` reproduces the original from-scratch per-(sample,dim) Gray-loop
//! (with sobol_direction recomputed per call); the library now carries running
//! bits and flips one direction word per step (incremental Gray-code). We prove
//! the emitted points are byte-identical (`.to_bits()`), then time the win.
//! Run: `cargo run --release -p fsci-stats --bin perf_sobol`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::qmc::SobolSampler;
use std::time::Instant;

/// Verbatim copy of the original direction-number computation.
fn old_sobol_direction(dimension: usize, bit: usize) -> u64 {
    let mut direction = 1u64 << 63;
    if dimension == 0 {
        return direction >> bit.min(63);
    }
    for _ in 0..bit.min(63) {
        direction ^= direction >> 1;
    }
    direction
}

/// Verbatim copy of the original from-scratch Sobol bits.
fn old_sobol_bits(index: u64, dimension: usize) -> u64 {
    let mut gray = index ^ (index >> 1);
    let mut bit = 0usize;
    let mut value = 0u64;
    while gray != 0 {
        if gray & 1 == 1 {
            value ^= old_sobol_direction(dimension, bit);
        }
        gray >>= 1;
        bit += 1;
    }
    value
}

fn unit(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
}

/// Original from-scratch sample (unscrambled, starting at index `start`).
fn old_sample(d: usize, start: u64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n * d);
    let mut idx = start;
    for _ in 0..n {
        for dim in 0..d {
            out.push(unit(old_sobol_bits(idx, dim)));
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

    // Cover several dimensions (excluding 2, which routes to sample_2d) and both
    // a fresh start (next_index=0) and a continued start (two-call sequencing).
    // Constructor caps Sobol at dims 1..=2; d=1 exercises the (changed) general
    // path, d=2 the (unchanged) sample_2d. Both must match the from-scratch path.
    for &d in &[1usize, 2] {
        for &(a, b) in &[(0usize, 64usize), (1, 100), (7, 250), (300, 200), (0, 5000)] {
            let mut s = SobolSampler::new(d).unwrap();
            let got_a = s.sample(a); // advances next_index to a
            let got_b = s.sample(b); // starts at index a
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

    // ---- Timing: the d=1 general path (the one this commit changes) ----
    for &(d, n) in &[(1usize, 500_000usize), (1, 2_000_000), (1, 8_000_000)] {
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_sample(d, 0, n)[0];
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            let mut s = SobolSampler::new(d).unwrap();
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
