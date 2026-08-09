// Same-process A/B for the MatrixNormal logpdf maha term Σ_i ‖L_V⁻¹·y[i]‖².
// `maha_old` is the pre-change form: n separate single-RHS solves (one per y row)
// against the shared p×p col-Cholesky, summing ‖·‖². `maha_new` transposes Y so
// the n RHS columns are contiguous, solves all at once via multi-RHS forward
// substitution (acc[i] left-folds k in 0..r = the per-row .sum() order), then
// sums ‖·‖² i-outer/r-inner — BYTE-IDENTICAL. Same process / same worker => no
// cross-worker noise; the two maha values must be exactly equal.
use std::time::Instant;

fn maha_old(l: &[Vec<f64>], y: &[Vec<f64>], p: usize) -> f64 {
    let mut maha = 0.0_f64;
    for yi in y {
        let mut wi = vec![0.0_f64; p];
        for r in 0..p {
            let sum: f64 = (0..r).map(|k| l[r][k] * wi[k]).sum();
            wi[r] = (yi[r] - sum) / l[r][r];
        }
        maha += wi.iter().map(|&v| v * v).sum::<f64>();
    }
    maha
}

fn maha_new(l: &[Vec<f64>], y: &[Vec<f64>], n: usize, p: usize) -> f64 {
    let mut yt = vec![vec![0.0_f64; n]; p];
    for (i, yi) in y.iter().enumerate() {
        for (r, &v) in yi.iter().enumerate() {
            yt[r][i] = v;
        }
    }
    let mut wmat = vec![vec![0.0_f64; n]; p];
    let mut acc = vec![0.0_f64; n];
    for r in 0..p {
        let lrr = l[r][r];
        for av in acc.iter_mut() {
            *av = 0.0;
        }
        for k in 0..r {
            let lrk = l[r][k];
            let wk = &wmat[k];
            for (av, &wki) in acc.iter_mut().zip(wk.iter()) {
                *av += lrk * wki;
            }
        }
        let wr = &mut wmat[r];
        for i in 0..n {
            wr[i] = (yt[r][i] - acc[i]) / lrr;
        }
    }
    let mut maha = 0.0_f64;
    for i in 0..n {
        maha += (0..p).map(|r| wmat[r][i] * wmat[r][i]).sum::<f64>();
    }
    maha
}

fn build(n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let l: Vec<Vec<f64>> = (0..p)
        .map(|i| {
            (0..p)
                .map(|j| {
                    if j < i {
                        u() - 0.5
                    } else if j == i {
                        2.0 + u()
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let y: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..p).map(|_| u() - 0.5).collect())
        .collect();
    (l, y)
}

fn best_of(reps: usize, mut f: impl FnMut() -> f64) -> (std::time::Duration, f64) {
    let mut best = std::time::Duration::MAX;
    let mut val = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        val = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, val)
}

fn main() {
    println!(
        "{:>6} {:>5} {:>12} {:>12} {:>8}  exact",
        "n", "p", "old_us", "new_us", "speedup"
    );
    for &(n, p) in &[(512usize, 64usize), (512, 128), (256, 256)] {
        let (l, y) = build(n, p);
        let (t_old, v_old) = best_of(10, || maha_old(&l, &y, p));
        let (t_new, v_new) = best_of(10, || maha_new(&l, &y, n, p));
        let exact = v_old == v_new;
        let old_us = t_old.as_secs_f64() * 1e6;
        let new_us = t_new.as_secs_f64() * 1e6;
        println!(
            "{n:>6} {p:>5} {old_us:>12.2} {new_us:>12.2} {:>7.2}x  {}",
            old_us / new_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
