// Same-process A/B for the ShortTimeFft::istft inner loop: per-frame inverse
// transform (independent, dominant) + overlap-add (serial). `serial` does
// transform+OLA interleaved per frame; `parallel` computes all per-frame
// transforms across threads then does the OLA serially in q order. The
// transform is a faithful proxy of the private ifft_func_onesided via the public
// fsci_fft::fft. BYTE-IDENTICAL: deterministic per-frame transform + unchanged
// OLA order. Same process / same worker => no cross-worker noise; outputs equal.
use fsci_fft::{Complex64, FftOptions, fft};
use std::time::Instant;

fn iframe(col: &[Complex64]) -> Vec<f64> {
    let spec = fft(col, &FftOptions::default()).expect("fft");
    spec.iter().map(|&(re, _im)| re).collect()
}

fn istft_serial(frames: &[Vec<Complex64>], m: usize, hop: usize, dual: &[f64]) -> Vec<f64> {
    let p = frames.len();
    let n = if p > 0 { (p - 1) * hop + m } else { 0 };
    let mut x = vec![0.0_f64; n];
    for (q, col) in frames.iter().enumerate() {
        let seg = iframe(col);
        let i0 = q * hop;
        for j in 0..m {
            x[i0 + j] += seg[j] * dual[j];
        }
    }
    x
}

fn istft_parallel(
    frames: &[Vec<Complex64>],
    m: usize,
    hop: usize,
    dual: &[f64],
    nthreads: usize,
) -> Vec<f64> {
    let p = frames.len();
    let n = if p > 0 { (p - 1) * hop + m } else { 0 };
    let chunk = p.div_ceil(nthreads);
    let chunks: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        let handles: Vec<_> =
            (0..nthreads)
                .filter_map(|t| {
                    let c0 = t * chunk;
                    if c0 >= p {
                        return None;
                    }
                    let c1 = (c0 + chunk).min(p);
                    Some(scope.spawn(move || {
                        frames[c0..c1].iter().map(|c| iframe(c)).collect::<Vec<_>>()
                    }))
                })
                .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut segs = Vec::with_capacity(p);
    for c in chunks {
        segs.extend(c);
    }
    let mut x = vec![0.0_f64; n];
    for (q, seg) in segs.iter().enumerate() {
        let i0 = q * hop;
        for j in 0..m {
            x[i0 + j] += seg[j] * dual[j];
        }
    }
    x
}

fn build(p_num: usize, m: usize) -> (Vec<Vec<Complex64>>, Vec<f64>) {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let frames: Vec<Vec<Complex64>> = (0..p_num)
        .map(|_| (0..m).map(|_| (u(), u())).collect())
        .collect();
    let dual: Vec<f64> = (0..m)
        .map(|j| {
            let t = std::f64::consts::PI * j as f64 / (m as f64 - 1.0);
            t.sin() * t.sin()
        })
        .collect();
    (frames, dual)
}

fn best_of(reps: usize, mut f: impl FnMut() -> Vec<f64>) -> (std::time::Duration, Vec<f64>) {
    let mut best = std::time::Duration::MAX;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        out = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, out)
}

fn main() {
    let nthreads = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(16);
    println!(
        "{:>7} {:>6} {:>5} {:>11} {:>11} {:>8}  exact (nthreads={nthreads})",
        "frames", "m", "hop", "serial_ms", "par_ms", "speedup"
    );
    for &(p_num, m, hop) in &[(2000usize, 512usize, 256usize), (7800, 256, 128)] {
        let (frames, dual) = build(p_num, m);
        let (t_ser, v_ser) = best_of(3, || istft_serial(&frames, m, hop, &dual));
        let (t_par, v_par) = best_of(3, || istft_parallel(&frames, m, hop, &dual, nthreads));
        let exact = v_ser == v_par;
        let ser_ms = t_ser.as_secs_f64() * 1e3;
        let par_ms = t_par.as_secs_f64() * 1e3;
        println!(
            "{p_num:>7} {m:>6} {hop:>5} {ser_ms:>11.2} {par_ms:>11.2} {:>7.2}x  {}",
            ser_ms / par_ms,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
