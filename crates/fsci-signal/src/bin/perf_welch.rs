//! Same-process timing + bit-identity digest harness for `welch`.
//!
//! Welch's method splits the signal into overlapping segments and averages each
//! segment's (detrended, windowed) periodogram. The segment periodograms are
//! independent rffts, now split across threads; the averaging fold runs in segment
//! order so the PSD is bit-identical to the serial loop. This dumps an FNV digest of
//! the PSD (compare across the stashed serial build) and times the call.
//! Run: `cargo run -p fsci-signal --bin perf_welch`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::welch;

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            (0.10 * t).sin() + 0.5 * (0.37 * t + 0.4).cos() + 0.01 * t // ramp -> exercises detrend
        })
        .collect()
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    // (signal length, nperseg) — long signals with modest nperseg => many segments.
    let cases = [
        (1_000_000usize, 512usize),
        (2_000_000, 1024),
        (4_000_000, 256),
    ];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, nps) in &cases {
        let x = signal(n);
        let r = welch(&x, 1.0, Some("hann"), Some(nps), None).expect("welch");
        println!(
            "n={n} nperseg={nps} nfreq={} psd={:016x} freq={:016x}",
            r.psd.len(),
            digest(&r.psd),
            digest(&r.frequencies)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, nps) in &cases {
        let x = signal(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = welch(black_box(&x), 1.0, Some("hann"), Some(nps), None).expect("welch");
            acc += r.psd[r.psd.len() / 2];
        }
        let dt = t0.elapsed();
        println!("n={n:>8} nperseg={nps:>5}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
