// Correctness + A/B for classical_mds (randomized eigendecomposition of the double-centered
// Gram matrix) vs a full symmetric eigendecomposition of the same B. The embedding must
// reproduce the input pairwise distances; the speedup is the wall-clock ratio.
use fsci_cluster::classical_mds;
use fsci_linalg::{DecompOptions, eigh};
use std::hint::black_box;
use std::time::Instant;

// Baseline: build B = -1/2 J D^2 J and take its full eigendecomposition (what fsci would do
// without the randomized top-k route), then touch the k largest eigenvalues.
fn full_mds_time(dist: &[Vec<f64>], k: usize) -> f64 {
    let n = dist.len();
    let d2: Vec<Vec<f64>> = dist
        .iter()
        .map(|row| row.iter().map(|&v| v * v).collect())
        .collect();
    let row_mean: Vec<f64> = d2
        .iter()
        .map(|r| r.iter().sum::<f64>() / n as f64)
        .collect();
    let total: f64 = row_mean.iter().sum::<f64>() / n as f64;
    let b: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| -0.5 * (d2[i][j] - row_mean[i] - row_mean[j] + total))
                .collect()
        })
        .collect();
    let e = eigh(&b, DecompOptions::default()).expect("eigh");
    e.eigenvalues.iter().rev().take(k).sum()
}

fn main() {
    let n = 1600usize;
    let r = 8usize;
    let k = 8usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let pts: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let dist: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..r)
                        .map(|t| (pts[i][t] - pts[j][t]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect()
        })
        .collect();

    let mds = classical_mds(&dist, k, 7).expect("classical_mds");
    let kk = mds.eigenvalues.len();
    let mut maxerr = 0.0f64;
    for (i, dist_i) in dist.iter().enumerate().take(n) {
        for (j, &dij) in dist_i.iter().enumerate().take(n) {
            let de: f64 = (0..kk)
                .map(|t| (mds.embedding[i][t] - mds.embedding[j][t]).powi(2))
                .sum::<f64>()
                .sqrt();
            maxerr = maxerr.max((de - dij).abs());
        }
    }
    println!("classical_mds distance_reconstruction_maxerr={maxerr:.3e}");

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(classical_mds(&dist, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_mds_time(&dist, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full eigh MDS {f_ms:.2} ms | randomized classical_mds {r_ms:.2} ms | speedup {:.2}x  (n={n} r={r} k={k})",
        f_ms / r_ms
    );
}
