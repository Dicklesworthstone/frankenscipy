// Correctness + A/B for kernel_pca (randomized_eigh) vs a full-eigh KernelPCA, on a
// low-rank PSD kernel. The projections must satisfy transformed·transformedᵀ ≈ centered
// kernel; the speedup is the wall-clock ratio (the eigendecomposition dominates).
use fsci_cluster::kernel_pca;
use fsci_linalg::{DecompOptions, eigh};
use std::hint::black_box;
use std::time::Instant;

fn double_center(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = k.len();
    let rm: Vec<f64> = k.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
    let tm: f64 = rm.iter().sum::<f64>() / n as f64;
    (0..n)
        .map(|i| (0..n).map(|j| k[i][j] - rm[i] - rm[j] + tm).collect())
        .collect()
}

fn full_kpca_transformed(kernel: &[Vec<f64>], kc_dim: usize) -> Vec<Vec<f64>> {
    let n = kernel.len();
    let kc = double_center(kernel);
    let e = eigh(&kc, DecompOptions::default()).expect("eigh"); // ascending
    (0..n)
        .map(|i| {
            (0..kc_dim)
                .map(|t| {
                    let idx = n - kc_dim + t;
                    e.eigenvectors[i][idx] * e.eigenvalues[idx].max(0.0).sqrt()
                })
                .collect()
        })
        .collect()
}

fn main() {
    let n = 1500usize;
    let r = 20usize;
    let k = 25usize; // > rank so the rank-k projection reconstructs the centered kernel exactly
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // PSD kernel K = B·Bᵀ (rank r).
    let b: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let kernel: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..r).map(|t| b[i][t] * b[j][t]).sum())
                .collect()
        })
        .collect();
    let kc = double_center(&kernel);

    let kp = kernel_pca(&kernel, k, 7).expect("kernel_pca");
    // transformed·transformedᵀ ≈ centered kernel.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let kk = kp.transformed[0].len();
    for (i, kc_i) in kc.iter().enumerate().take(n) {
        for (j, &kcij) in kc_i.iter().enumerate().take(n) {
            let approx: f64 = (0..kk)
                .map(|t| kp.transformed[i][t] * kp.transformed[j][t])
                .sum();
            num += (kcij - approx).powi(2);
            den += kcij * kcij;
        }
    }
    println!(
        "kernel_pca rel_reconstruction_err(transformed·Tᵀ vs centered K) = {:.3e}",
        (num / den).sqrt()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(kernel_pca(&kernel, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_kpca_transformed(&kernel, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full-eigh KernelPCA {f_ms:.2} ms | randomized kernel_pca {r_ms:.2} ms | speedup {:.2}x  (n={n} k={k})",
        f_ms / r_ms
    );
}
