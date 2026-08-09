// Same-process A/B for the Wishart/inverse-Wishart logpdf trace term
// Σ_j ‖L⁻¹·col_j(rhs)‖². `tr_old` is the pre-change form: p separate single-RHS
// lower-triangular solves (gather column, solve, ‖·‖²). `tr_new` is one
// multi-RHS forward substitution (acc[j] left-folds k in 0..i = the per-column
// .sum() order) with contiguous inner j-loops that vectorize, then a
// column-major ‖·‖² sum — BYTE-IDENTICAL. Same process / same worker => no
// cross-worker noise; the two traces must be exactly equal.
use std::time::Instant;

fn tr_old(l: &[Vec<f64>], rhs: &[Vec<f64>], p: usize) -> f64 {
    let mut tr = 0.0_f64;
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| rhs[i][j]).collect();
        let mut sol = vec![0.0_f64; p];
        for i in 0..p {
            let sum: f64 = (0..i).map(|k| l[i][k] * sol[k]).sum();
            sol[i] = (col[i] - sum) / l[i][i];
        }
        tr += sol.iter().map(|&v| v * v).sum::<f64>();
    }
    tr
}

fn tr_new(l: &[Vec<f64>], rhs: &[Vec<f64>], p: usize) -> f64 {
    let mut w = vec![vec![0.0_f64; p]; p];
    let mut acc = vec![0.0_f64; p];
    for i in 0..p {
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
        for j in 0..p {
            wi[j] = (rhs[i][j] - acc[j]) / lii;
        }
    }
    let mut tr = 0.0_f64;
    for j in 0..p {
        tr += (0..p).map(|i| w[i][j] * w[i][j]).sum::<f64>();
    }
    tr
}

fn build(p: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
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
    let rhs: Vec<Vec<f64>> = (0..p)
        .map(|i| {
            (0..p)
                .map(|j| if j <= i { u() - 0.5 } else { 0.0 })
                .collect()
        })
        .collect();
    (l, rhs)
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
        "{:>5} {:>12} {:>12} {:>8}  exact",
        "p", "old_us", "new_us", "speedup"
    );
    for &p in &[32usize, 64, 128, 200] {
        let (l, rhs) = build(p);
        // amortize tiny per-call cost across repeats inside the timed closure
        let (t_old, v_old) = best_of(20, || {
            let mut acc = 0.0;
            for _ in 0..16 {
                acc += tr_old(&l, &rhs, p);
            }
            acc
        });
        let (t_new, v_new) = best_of(20, || {
            let mut acc = 0.0;
            for _ in 0..16 {
                acc += tr_new(&l, &rhs, p);
            }
            acc
        });
        let exact = v_old == v_new;
        let old_us = t_old.as_secs_f64() * 1e6 / 16.0;
        let new_us = t_new.as_secs_f64() * 1e6 / 16.0;
        println!(
            "{p:>5} {old_us:>12.3} {new_us:>12.3} {:>7.2}x  {}",
            old_us / new_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
