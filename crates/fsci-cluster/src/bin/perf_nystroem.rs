// Correctness + A/B for nystroem (column-sampling low-rank kernel approximation) vs a full
// eigendecomposition of the n×n kernel truncated to the same rank. Z·Zᵀ must reconstruct K;
// the speedup is the wall-clock ratio (Nyström only eig-decomposes the small m×m block).
use fsci_cluster::nystroem;
use fsci_linalg::{DecompOptions, eigh};
use std::hint::black_box;
use std::time::Instant;

// Baseline: full eigh of K, rebuild rank-m feature map Z = U_m·diag(sqrt(max(λ,0))).
fn full_lowrank_time(kernel: &[Vec<f64>], m: usize) -> f64 {
    let e = eigh(kernel, DecompOptions::default()).expect("eigh");
    let n = kernel.len();
    // top-m eigenpairs (eigh is ascending) → touch the reconstructed diagonal.
    let mut acc = 0.0;
    for c in 0..m {
        let idx = n - 1 - c;
        let lam = e.eigenvalues[idx].max(0.0).sqrt();
        acc += lam * e.eigenvectors[0][idx];
    }
    acc
}

fn main() {
    let n = 2000usize;
    let r = 30usize;
    let m = 60usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let b: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let kernel: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..r).map(|t| b[i][t] * b[j][t]).sum())
                .collect()
        })
        .collect();

    let ny = nystroem(&kernel, m, 7).expect("nystroem");
    let mp = ny.feature_map[0].len();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, krow) in kernel.iter().enumerate().take(n) {
        for (j, &kij) in krow.iter().enumerate().take(n) {
            let zz: f64 = (0..mp)
                .map(|t| ny.feature_map[i][t] * ny.feature_map[j][t])
                .sum();
            num += (zz - kij).powi(2);
            den += kij * kij;
        }
    }
    println!("nystroem rel_reconstruction_err={:.3e}", (num / den).sqrt());

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(nystroem(&kernel, m, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_lowrank_time(&kernel, m));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full eigh K {f_ms:.2} ms | nystroem {r_ms:.2} ms | speedup {:.2}x  (n={n} r={r} m={m})",
        f_ms / r_ms
    );
}
