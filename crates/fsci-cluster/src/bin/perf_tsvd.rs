// Correctness + A/B for truncated_svd (randomized) vs a full-SVD truncation, on a low-rank
// matrix. transformed·components must reconstruct X; the speedup is the wall-clock ratio.
use fsci_cluster::truncated_svd;
use fsci_linalg::{DecompOptions, svd};
use std::hint::black_box;
use std::time::Instant;

fn full_truncated_time(x: &[Vec<f64>], k: usize) -> f64 {
    let s = svd(x, DecompOptions::default()).expect("svd");
    s.s.iter().take(k).sum() // touch the result
}

fn main() {
    let n = 5000usize;
    let d = 300usize;
    let r = 20usize;
    let k = 25usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let u: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let v: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| rng()).collect()).collect();
    let x: Vec<Vec<f64>> = u
        .iter()
        .map(|ui| {
            (0..d)
                .map(|j| (0..r).map(|t| ui[t] * v[t][j]).sum::<f64>() + 1e-5 * rng())
                .collect()
        })
        .collect();

    let ts = truncated_svd(&x, k, 7).expect("truncated_svd");
    // reconstruction X ≈ transformed · components
    let kk = ts.singular_values.len();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, x_i) in x.iter().enumerate().take(n) {
        for (j, &xij) in x_i.iter().enumerate().take(d) {
            let approx: f64 = (0..kk)
                .map(|t| ts.transformed[i][t] * ts.components[t][j])
                .sum();
            num += (xij - approx).powi(2);
            den += xij * xij;
        }
    }
    println!(
        "truncated_svd rel_reconstruction_err={:.3e}  evr_sum={:.4}",
        (num / den).sqrt(),
        ts.explained_variance_ratio.iter().sum::<f64>()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(truncated_svd(&x, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_truncated_time(&x, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full SVD {f_ms:.2} ms | randomized truncated_svd {r_ms:.2} ms | speedup {:.2}x  (n={n} d={d} k={k})",
        f_ms / r_ms
    );
}
