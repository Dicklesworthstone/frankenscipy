//! Same-process timing + bit-identity digest harness for `csd` (cross-spectral density).
//!
//! csd averages independent per-segment cross-periodograms (detrend, window, two rffts,
//! then conj(X)*Y). The segment loop is now split across threads, and the averaging fold
//! runs in segment order so the result is bit-identical to the serial loop. Dumps an FNV
//! digest of the complex CSD (compare across the stashed serial build) and times the call.
//! Run: `cargo run -p fsci-signal --bin perf_csd`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::csd;

fn signals(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            (0.10 * t).sin() + 0.5 * (0.37 * t + 0.4).cos() + 0.01 * t
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            (0.10 * t + 0.3).sin() + 0.4 * (0.51 * t).cos() - 0.005 * t
        })
        .collect();
    (x, y)
}

fn digest(values: &[(f64, f64)]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, &(re, im)| {
        let h = (h ^ re.to_bits()).wrapping_mul(1099511628211);
        (h ^ im.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let cases = [
        (1_000_000usize, 512usize),
        (2_000_000, 1024),
        (4_000_000, 256),
    ];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, nps) in &cases {
        let (x, y) = signals(n);
        let r = csd(&x, &y, 1.0, Some("hann"), Some(nps), None).expect("csd");
        println!("n={n} nperseg={nps} nfreq={} csd={:016x}", r.csd.len(), digest(&r.csd));
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, nps) in &cases {
        let (x, y) = signals(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = csd(black_box(&x), black_box(&y), 1.0, Some("hann"), Some(nps), None)
                .expect("csd");
            acc += r.csd[r.csd.len() / 2].0;
        }
        let dt = t0.elapsed();
        println!("n={n:>8} nperseg={nps:>5}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
