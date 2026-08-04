//! Median-null-gated A/B for `opt::approx_derivative` (finite-difference Jacobian): the ORIG serial
//! per-column loop vs the parallel-across-columns path. Both arms live in ONE binary, toggled by
//! `OPT_APPROX_DERIV_FORCE_SERIAL` and ALTERNATED per iteration. Each Jacobian column perturbs one
//! parameter and evaluates `fun` independently; a finite-difference Jacobian is used precisely when
//! `fun` is expensive (ODE solve / simulation), so the per-column `fun` evals are the dominant cost.
use fsci_opt::{FiniteDiffMethod, OPT_APPROX_DERIV_FORCE_SERIAL, approx_derivative};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(48);
    let m: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let work: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(13);

    let x0: Vec<f64> = (0..n).map(|i| 0.1 + 0.01 * i as f64).collect();

    // Expensive vector function (deterministic, Sync): each output is a transcendental accumulation
    // over the inputs, repeated `work` times to mimic a costly model evaluation.
    let fun = move |x: &[f64]| -> Vec<f64> {
        (0..m)
            .map(|k| {
                let mut acc = 0.0_f64;
                for _ in 0..work {
                    for (j, &xj) in x.iter().enumerate() {
                        acc += (xj * (k + j + 1) as f64).sin();
                    }
                }
                acc
            })
            .collect()
    };

    // Parity: the full m×n Jacobian must be bit-identical across the two arms.
    OPT_APPROX_DERIV_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = approx_derivative(fun, &x0, FiniteDiffMethod::ThreePoint, None, None, None).unwrap();
    OPT_APPROX_DERIV_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = approx_derivative(fun, &x0, FiniteDiffMethod::ThreePoint, None, None, None).unwrap();
    let bitmism: usize = a
        .iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + usize::from(a.len() != b.len());

    let bench = |force_serial: bool| -> f64 {
        OPT_APPROX_DERIV_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || {
            approx_derivative(
                fun,
                black_box(&x0),
                FiniteDiffMethod::ThreePoint,
                None,
                None,
                None,
            )
            .unwrap()
        };
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
    OPT_APPROX_DERIV_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# opt::approx_derivative (ThreePoint) n={n} m={m} work={work}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
