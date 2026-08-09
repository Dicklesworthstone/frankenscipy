// Correctness + A/B for rbf_nystroem (data-based RBF feature map — never forms the n×n
// kernel) vs the path that materializes the dense n×n RBF kernel and feeds it to the
// precomputed `nystroem`. Both yield a feature map Z with Z·Zᵀ ≈ K_rbf; the speedup is the
// wall-clock ratio. The win is the O(n²)→O(n·m) drop from skipping the dense kernel.
use fsci_cluster::{nystroem, rbf_nystroem};
use std::hint::black_box;
use std::time::Instant;

fn build_rbf(data: &[Vec<f64>], gamma: f64) -> Vec<Vec<f64>> {
    let n = data.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d2: f64 = data[i]
                        .iter()
                        .zip(&data[j])
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    (-gamma * d2).exp()
                })
                .collect()
        })
        .collect()
}

fn main() {
    let n = 4000usize;
    let dim = 20usize;
    let m = 80usize;
    let gamma = 0.3f64;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let data: Vec<Vec<f64>> = (0..n).map(|_| (0..dim).map(|_| rng()).collect()).collect();

    // Correctness: rel reconstruction error of Z·Zᵀ vs the true RBF kernel.
    let z = rbf_nystroem(&data, m, gamma, 7).expect("rbf_nystroem");
    let kref = build_rbf(&data, gamma);
    let mp = z.feature_map[0].len();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, kref_i) in kref.iter().enumerate().take(n) {
        for (j, &kij) in kref_i.iter().enumerate().take(n) {
            let zz: f64 = (0..mp)
                .map(|t| z.feature_map[i][t] * z.feature_map[j][t])
                .sum();
            num += (zz - kij).powi(2);
            den += kij * kij;
        }
    }
    println!(
        "rbf_nystroem rel_reconstruction_err={:.3e}  (m={m})",
        (num / den).sqrt()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(rbf_nystroem(&data, m, gamma, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        // Full path: materialize n×n RBF kernel, then precomputed nystroem.
        let t = Instant::now();
        let k = build_rbf(&data, gamma);
        black_box(nystroem(&k, m, 7).unwrap());
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "dense-kernel + nystroem {f_ms:.2} ms | rbf_nystroem (data) {r_ms:.2} ms | speedup {:.2}x  (n={n} d={dim} m={m})",
        f_ms / r_ms
    );
}
