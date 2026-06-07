//! Same-process timing + bit-identity digest harness for `lombscargle`.
//!
//! Each frequency's Lomb-Scargle power is an independent O(n) reduction over the
//! samples; the library now splits the `m` frequencies across threads. This harness
//! dumps an FNV digest of the full power spectrum (compare across the stashed serial
//! build to prove byte-identity) and times the call. Run:
//! `cargo run -p fsci-signal --bin perf_lombscargle`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::lombscargle;

fn problem(n: usize, m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Irregularly sampled signal: two tones + a deterministic jitter on the times.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            t + 0.37 * (0.013 * t).sin()
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (0.20 * t).sin() + 0.5 * (0.53 * t + 0.4).cos())
        .collect();
    let freqs: Vec<f64> = (0..m)
        .map(|k| 0.001 + (k as f64 + 1.0) * (1.5 / m as f64))
        .collect();
    (x, y, freqs)
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let cases = [(2000usize, 2000usize), (4000, 4000), (8000, 6000)];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, m) in &cases {
        let (x, y, freqs) = problem(n, m);
        let p_raw = lombscargle(&x, &y, &freqs, false).expect("lombscargle");
        let p_norm = lombscargle(&x, &y, &freqs, true).expect("lombscargle");
        println!(
            "n={n} m={m} raw={:016x} norm={:016x}",
            digest(&p_raw),
            digest(&p_norm)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, m) in &cases {
        let (x, y, freqs) = problem(n, m);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let p = lombscargle(black_box(&x), black_box(&y), black_box(&freqs), false)
                .expect("lombscargle");
            acc += p[p.len() / 2];
        }
        let dt = t0.elapsed();
        println!("n={n:>5} m={m:>5}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
