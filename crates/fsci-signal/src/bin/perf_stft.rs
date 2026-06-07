//! Same-process A/B for the parallel STFT: times the shipped (parallel) `stft` against
//! a verbatim sequential frame loop in ONE process (immune to cross-worker variance),
//! and checks they are bit-identical. Parallelizing stft also covers spectrogram.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::{StftResult, get_window, stft};

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.001;
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
        })
        .collect()
}

/// Verbatim sequential STFT frame loop (window + rfft per frame). Uses the crate's
/// own `hann` so the windowing exactly matches `stft`'s `get_window("hann")`.
fn stft_seq(x: &[f64], nperseg: usize, noverlap: usize) -> Vec<Vec<(f64, f64)>> {
    let step = nperseg - noverlap;
    let win = get_window("hann", nperseg).expect("window");
    let n_freqs = nperseg / 2 + 1;
    let opts = fsci_fft::FftOptions::default();
    let mut zxx = Vec::new();
    let mut start = 0;
    while start + nperseg <= x.len() {
        let windowed: Vec<f64> = x[start..start + nperseg]
            .iter()
            .zip(&win)
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let spectrum = fsci_fft::rfft(&windowed, &opts).expect("rfft");
        zxx.push(spectrum[..n_freqs].to_vec());
        start += step;
    }
    zxx
}

fn digest(zxx: &[Vec<(f64, f64)>]) -> u64 {
    let mut d: u64 = 1469598103934665603;
    for row in zxx {
        for &(re, im) in row {
            d ^= re.to_bits();
            d = d.wrapping_mul(1099511628211);
            d ^= im.to_bits();
            d = d.wrapping_mul(1099511628211);
        }
    }
    d
}

fn time_par(iters: usize, x: &[f64], nperseg: usize, noverlap: usize) -> f64 {
    let f = || stft(x, 1000.0, Some("hann"), Some(nperseg), Some(noverlap)).expect("stft");
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn time_seq(iters: usize, x: &[f64], nperseg: usize, noverlap: usize) -> f64 {
    black_box(stft_seq(x, nperseg, noverlap));
    let start = Instant::now();
    for _ in 0..iters {
        black_box(stft_seq(x, nperseg, noverlap));
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    for &(n, nperseg) in &[
        (800_000usize, 1024usize),
        (1_200_000, 2048),
        (2_500_000, 1024),
    ] {
        let x = signal(n);
        let noverlap = nperseg / 4;
        let par: StftResult =
            stft(&x, 1000.0, Some("hann"), Some(nperseg), Some(noverlap)).expect("stft");
        let seq = stft_seq(&x, nperseg, noverlap);
        let exact = digest(&par.zxx) == digest(&seq);

        let iters = 8;
        let tp = time_par(iters, &x, nperseg, noverlap);
        let ts = time_seq(iters, &x, nperseg, noverlap);
        println!(
            "stft n={n} nperseg={nperseg}: seq={ts:>8.3}ms  par={tp:>8.3}ms  speedup={:>6.2}x  frames={}  bit_identical={exact}",
            ts / tp,
            par.zxx.len()
        );
    }
}
