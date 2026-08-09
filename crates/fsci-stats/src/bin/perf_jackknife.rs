//! Median-null-gated A/B for `stats::jackknife`: the ORIG serial per-replicate loop vs the
//! parallel-across-replicates path. Both arms live in ONE binary, toggled by
//! `STATS_JACKKNIFE_FORCE_SERIAL` and ALTERNATED per iteration. Jackknife is deterministic (no RNG),
//! so each leave-one-out replicate is independent; the `statistic` evals (n calls over ~n elements)
//! dominate. The statistic here is the sample median (sort-based) — a realistic jackknifed estimator.
use fsci_stats::{STATS_JACKKNIFE_FORCE_SERIAL, jackknife};
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

// Realistic jackknifed estimator: the sample median (sort + middle element).
fn median_stat(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(13);

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0
    };
    let data: Vec<f64> = (0..n).map(|_| r()).collect();

    // Parity: statistic/bias/se/replicates must all be bit-identical across the two arms.
    STATS_JACKKNIFE_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = jackknife(&data, median_stat);
    STATS_JACKKNIFE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = jackknife(&data, median_stat);
    let bitf = |x: f64, y: f64| usize::from(x.to_bits() != y.to_bits());
    let bitmism = bitf(a.statistic, b.statistic)
        + bitf(a.bias, b.bias)
        + bitf(a.se, b.se)
        + a.replicates
            .iter()
            .zip(&b.replicates)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count()
        + usize::from(a.replicates.len() != b.replicates.len());

    let bench = |force_serial: bool| -> f64 {
        STATS_JACKKNIFE_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || jackknife(black_box(&data), median_stat);
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
    STATS_JACKKNIFE_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# stats::jackknife (median statistic) n={n}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
