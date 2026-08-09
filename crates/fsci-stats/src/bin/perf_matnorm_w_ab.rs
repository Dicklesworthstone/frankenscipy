// Same-process A/B for the matrix-normal logpdf W build (W = L_U^-1 C, m x n).
// `w_old` is the pre-change form: n separate single-RHS lower-triangular solves
// (gather column, solve, scatter). `w_new` is one multi-RHS forward substitution
// over all n columns: acc[j] = sum_{k<i} L[i][k]*w[k][j] left-folds k in 0..i (the
// same order as the per-column `.sum()`), so the result is BYTE-IDENTICAL, while
// the contiguous inner j-loops vectorize and the gather/alloc/scatter is gone.
// Same process / same worker => no cross-worker noise; matrices must be exactly equal.
use std::time::Instant;

fn w_old(l: &[Vec<f64>], c: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut w = vec![vec![0.0_f64; n]; m];
    for j in 0..n {
        let col: Vec<f64> = (0..m).map(|i| c[i][j]).collect();
        let mut sol = vec![0.0_f64; m];
        for i in 0..m {
            let sum: f64 = (0..i).map(|k| l[i][k] * sol[k]).sum();
            sol[i] = (col[i] - sum) / l[i][i];
        }
        for i in 0..m {
            w[i][j] = sol[i];
        }
    }
    w
}

fn w_new(l: &[Vec<f64>], c: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut w = vec![vec![0.0_f64; n]; m];
    let mut acc = vec![0.0_f64; n];
    for i in 0..m {
        let lii = l[i][i];
        for a in acc.iter_mut() {
            *a = 0.0;
        }
        for k in 0..i {
            let lik = l[i][k];
            let wk = &w[k];
            for (a, &wkj) in acc.iter_mut().zip(wk.iter()) {
                *a += lik * wkj;
            }
        }
        let wi = &mut w[i];
        for j in 0..n {
            wi[j] = (c[i][j] - acc[j]) / lii;
        }
    }
    w
}

fn build(m: usize, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    // Lower-triangular L (m x m) with positive diagonal (well-conditioned).
    let l: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..m)
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
    let c: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| u() - 0.5).collect())
        .collect();
    (l, c)
}

fn best_of(
    reps: usize,
    mut f: impl FnMut() -> Vec<Vec<f64>>,
) -> (std::time::Duration, Vec<Vec<f64>>) {
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
    println!(
        "{:>6} {:>5} {:>12} {:>12} {:>8}  {}",
        "m", "n", "old_us", "new_us", "speedup"
    );
    for &(m, n) in &[(500usize, 64usize), (1000, 64), (1000, 128)] {
        let (l, c) = build(m, n);
        let (t_old, v_old) = best_of(5, || w_old(&l, &c, m, n));
        let (t_new, v_new) = best_of(5, || w_new(&l, &c, m, n));
        let exact = v_old == v_new;
        let old_us = t_old.as_secs_f64() * 1e6;
        let new_us = t_new.as_secs_f64() * 1e6;
        println!(
            "{m:>6} {n:>5} {old_us:>12.2} {new_us:>12.2} {:>7.2}x  {}",
            old_us / new_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
