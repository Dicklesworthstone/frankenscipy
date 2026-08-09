// Same-process A/B for the matrix-normal logpdf Gram build (A = WᵀW, the
// dominant O(m·n²) term of MatrixNormal::logpdf). `gram_full` is the pre-change
// full-n² double loop; `gram_sym` computes the upper triangle once and mirrors
// (the Gram is symmetric: w[r][i]*w[r][j] commutes and is summed in the same
// r-order, so the mirrored entry is BYTE-IDENTICAL). Same process / same worker
// => no cross-worker noise; the two matrices must be exactly equal.
use std::time::Instant;

fn gram_full(w: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut va = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let a: f64 = (0..m).map(|r| w[r][i] * w[r][j]).sum();
            va[i][j] = a;
        }
    }
    va
}

fn gram_sym(w: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut va = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let a: f64 = (0..m).map(|r| w[r][i] * w[r][j]).sum();
            va[i][j] = a;
        }
    }
    // Mirrors the upper triangle into the lower one, so it READS va[j][i] while
    // WRITING va[i][j] (frankenscipy-023vy). Those are disjoint elements but the
    // borrow checker cannot see that through an iterator, so the index form is
    // the honest one here.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..i {
            va[i][j] = va[j][i];
        }
    }
    va
}

fn build_w(m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    (0..m)
        .map(|_| {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect()
        })
        .collect()
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
        "{:>6} {:>5} {:>12} {:>12} {:>8}  exact",
        "m", "n", "full_us", "sym_us", "speedup"
    );
    for &(m, n) in &[(2000usize, 64usize), (2000, 128), (4000, 150)] {
        let w = build_w(m, n);
        let (t_full, v_full) = best_of(5, || gram_full(&w, m, n));
        let (t_sym, v_sym) = best_of(5, || gram_sym(&w, m, n));
        let exact = v_full == v_sym;
        let full_us = t_full.as_secs_f64() * 1e6;
        let sym_us = t_sym.as_secs_f64() * 1e6;
        println!(
            "{m:>6} {n:>5} {full_us:>12.2} {sym_us:>12.2} {:>7.2}x  {}",
            full_us / sym_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
