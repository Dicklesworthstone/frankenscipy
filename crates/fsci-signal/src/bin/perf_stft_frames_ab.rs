// Same-process A/B for the ShortTimeFFT::stft frame loop: serial vs parallel
// over frames. Each frame is an independent windowed FFT (a faithful proxy of
// the struct's private fft_func, here via the public fsci_fft::fft), so the work
// is embarrassingly parallel and BYTE-IDENTICAL (distinct output columns, no
// reduction). Same process / same worker => no cross-worker noise; the serial
// and parallel column sets must be exactly equal.
use fsci_fft::{Complex64, FftOptions, fft};
use std::time::Instant;

fn build_window(m: usize) -> Vec<f64> {
    // Hann window.
    (0..m)
        .map(|j| {
            let t = std::f64::consts::PI * j as f64 / (m as f64 - 1.0);
            t.sin() * t.sin()
        })
        .collect()
}

fn build_signal(n: usize) -> Vec<f64> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn one_frame(x: &[f64], win: &[f64], hop: usize, m: usize, pi: usize) -> Vec<Complex64> {
    let k0 = pi * hop;
    let seg: Vec<Complex64> = (0..m)
        .map(|j| {
            let idx = k0 + j;
            (if idx < x.len() { x[idx] * win[j] } else { 0.0 }, 0.0)
        })
        .collect();
    fft(&seg, &FftOptions::default()).expect("fft")
}

fn frames_serial(
    x: &[f64],
    win: &[f64],
    hop: usize,
    m: usize,
    p_num: usize,
) -> Vec<Vec<Complex64>> {
    (0..p_num).map(|pi| one_frame(x, win, hop, m, pi)).collect()
}

fn frames_parallel(
    x: &[f64],
    win: &[f64],
    hop: usize,
    m: usize,
    p_num: usize,
    nthreads: usize,
) -> Vec<Vec<Complex64>> {
    let chunk = p_num.div_ceil(nthreads);
    let chunks: Vec<Vec<Vec<Complex64>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let c0 = t * chunk;
                if c0 >= p_num {
                    return None;
                }
                let c1 = (c0 + chunk).min(p_num);
                Some(scope.spawn(move || {
                    (c0..c1)
                        .map(|pi| one_frame(x, win, hop, m, pi))
                        .collect::<Vec<_>>()
                }))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut out = Vec::with_capacity(p_num);
    for c in chunks {
        out.extend(c);
    }
    out
}

fn best_of(
    reps: usize,
    mut f: impl FnMut() -> Vec<Vec<Complex64>>,
) -> (std::time::Duration, usize) {
    let mut best = std::time::Duration::MAX;
    let mut nf = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let out = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
        nf = out.len();
    }
    (best, nf)
}

fn main() {
    let nthreads = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(16);
    println!(
        "{:>8} {:>6} {:>7} {:>5} {:>11} {:>11} {:>8}  exact (nthreads={nthreads})",
        "n", "m", "frames", "hop", "serial_ms", "par_ms", "speedup"
    );
    for &(n, m, hop) in &[(500_000usize, 512usize, 256usize), (1_000_000, 256, 128)] {
        let win = build_window(m);
        let x = build_signal(n);
        let p_num = if n >= m { (n - m) / hop + 1 } else { 0 };
        let (t_ser, _) = best_of(3, || frames_serial(&x, &win, hop, m, p_num));
        let (t_par, _) = best_of(3, || frames_parallel(&x, &win, hop, m, p_num, nthreads));
        let exact = frames_serial(&x, &win, hop, m, p_num)
            == frames_parallel(&x, &win, hop, m, p_num, nthreads);
        let ser_ms = t_ser.as_secs_f64() * 1e3;
        let par_ms = t_par.as_secs_f64() * 1e3;
        println!(
            "{n:>8} {m:>6} {p_num:>7} {hop:>5} {ser_ms:>11.2} {par_ms:>11.2} {:>7.2}x  {}",
            ser_ms / par_ms,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
