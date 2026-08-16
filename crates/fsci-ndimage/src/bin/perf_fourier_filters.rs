//! Median-null-gated A/B for the Fourier-domain filters: the ORIG serial per-element fill vs the
//! parallel-across-output-chunks fill. Both arms live in ONE binary, toggled by
//! `NDIMAGE_FOURIER_FORCE_SERIAL` and ALTERNATED per iteration inside one measured routine, so a
//! single `rch exec` invocation measures both on the same worker.
use fsci_fft::Complex64;
use fsci_ndimage::{
    NDIMAGE_FOURIER_FORCE_SERIAL, fourier_ellipsoid, fourier_gaussian, fourier_shift,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let which: String = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "gaussian".to_string());
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let shape = vec![side, side];
    let total = side * side;
    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let input: Vec<Complex64> = (0..total).map(|_| (r(), r())).collect();
    let param = vec![2.5, 3.0];
    // Named so the dispatch reads as a choice between filters rather than as a
    // signature (frankenscipy-iit9c).
    type FourierFilter = fn(&[Complex64], &[usize], &[f64]) -> Vec<Complex64>;
    let filt: FourierFilter = match which.as_str() {
        "ellipsoid" => fourier_ellipsoid,
        "shift" => fourier_shift,
        _ => fourier_gaussian,
    };

    // Parity: parallel must be byte-identical to serial.
    NDIMAGE_FOURIER_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = filt(&input, &shape, &param);
    NDIMAGE_FOURIER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = filt(&input, &shape, &param);
    let bitmism = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.0.to_bits() != y.0.to_bits() || x.1.to_bits() != y.1.to_bits())
        .count();

    let bench = |force_serial: bool| -> f64 {
        NDIMAGE_FOURIER_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || filt(black_box(&input), &shape, &param);
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    NDIMAGE_FOURIER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# fourier_{which} {side}x{side} ({total} elems)");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
