// De-risk microbench for MultivariateNormal::logpdf_many: per-point single-RHS
// forward-substitution Mahalanobis (current) vs a multi-RHS batch solve that
// vectorizes the inner loop across points. BYTE-IDENTICAL (each point's
// `solved[i]` left-folds j in 0..i the same way; maha = Σ solved[i]² per point).
// If the batch form clearly wins, restructure the library; else don't.
use std::time::Instant;

// Current: one forward-substitution per point, then ||·||^2.
fn maha_perpoint(chol: &[Vec<f64>], centered: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; centered.len()];
    let mut solved = vec![0.0_f64; d];
    for (p, c) in centered.iter().enumerate() {
        for i in 0..d {
            let sum = (0..i).map(|j| chol[i][j] * solved[j]).sum::<f64>();
            solved[i] = (c[i] - sum) / chol[i][i];
        }
        out[p] = solved.iter().map(|v| v * v).sum::<f64>();
    }
    out
}

// Multi-RHS: solve all points at once (inner loop over points vectorizes), then
// per-point ||·||^2. `w[i][p]` = solved value for point p at coord i.
fn maha_multirhs(chol: &[Vec<f64>], centered: &[Vec<f64>], d: usize) -> Vec<f64> {
    let n = centered.len();
    let mut w = vec![vec![0.0_f64; n]; d];
    let mut acc = vec![0.0_f64; n];
    for i in 0..d {
        let lii = chol[i][i];
        for a in acc.iter_mut() {
            *a = 0.0;
        }
        for k in 0..i {
            let lik = chol[i][k];
            let wk = &w[k];
            for (a, &wkp) in acc.iter_mut().zip(wk.iter()) {
                *a += lik * wkp;
            }
        }
        let wi = &mut w[i];
        for p in 0..n {
            wi[p] = (centered[p][i] - acc[p]) / lii;
        }
    }
    let mut out = vec![0.0_f64; n];
    for (p, o) in out.iter_mut().enumerate() {
        *o = (0..d).map(|i| w[i][p] * w[i][p]).sum::<f64>();
    }
    out
}

fn build(n: usize, d: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut u = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let chol: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..d)
                .map(|j| {
                    if j < i {
                        u()
                    } else if j == i {
                        2.0 + u().abs()
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let centered: Vec<Vec<f64>> = (0..n).map(|_| (0..d).map(|_| u()).collect()).collect();
    (chol, centered)
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
    println!(
        "{:>7} {:>4} {:>11} {:>11} {:>8}  {}",
        "n", "d", "perpt_us", "batch_us", "speedup"
    );
    for &(n, d) in &[(50000usize, 4usize), (50000, 8), (20000, 16), (20000, 32)] {
        let (chol, centered) = build(n, d);
        let (t_pp, v_pp) = best_of(8, || maha_perpoint(&chol, &centered, d));
        let (t_mr, v_mr) = best_of(8, || maha_multirhs(&chol, &centered, d));
        let exact = v_pp == v_mr;
        let pp_us = t_pp.as_secs_f64() * 1e6;
        let mr_us = t_mr.as_secs_f64() * 1e6;
        println!(
            "{n:>7} {d:>4} {pp_us:>11.2} {mr_us:>11.2} {:>7.2}x  {}",
            pp_us / mr_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
