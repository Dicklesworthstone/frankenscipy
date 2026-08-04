//! Median-null-gated A/B for `opt::approx_fprime` (finite-difference gradient): the ORIG serial
//! per-component loop vs the parallel-across-components path. Both arms live in ONE binary, toggled by
//! `OPT_APPROX_FPRIME_FORCE_SERIAL` and ALTERNATED per iteration. Each gradient component perturbs one
//! parameter and evaluates the objective independently; a finite-difference gradient is used when the
//! objective is expensive, so the per-component evals dominate. scipy's approx_fprime is serial.
use fsci_opt::{OPT_APPROX_FPRIME_FORCE_SERIAL, approx_fprime};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn med(v: &mut [f64]) -> f64 {
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let work: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13);

    let x0: Vec<f64> = (0..n).map(|i| 0.1 + 0.01 * i as f64).collect();

    // Expensive scalar objective (deterministic, Sync): a transcendental accumulation over the
    // inputs repeated `work` times, mimicking a costly model.
    let f = move |x: &[f64]| -> f64 {
        let mut acc = 0.0_f64;
        for _ in 0..work {
            for (j, &xj) in x.iter().enumerate() {
                acc += (xj * (j + 1) as f64).sin();
            }
        }
        acc
    };

    // Parity: the full gradient vector must be bit-identical across the two arms.
    OPT_APPROX_FPRIME_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = approx_fprime(&x0, f, 1.49e-8).unwrap();
    OPT_APPROX_FPRIME_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = approx_fprime(&x0, f, 1.49e-8).unwrap();
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        OPT_APPROX_FPRIME_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || approx_fprime(black_box(&x0), f, 1.49e-8).unwrap();
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
        let f2 = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / f2);
        ov.push(o);
        fv.push(f2);
    }
    OPT_APPROX_FPRIME_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# opt::approx_fprime n={n} work={work}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
