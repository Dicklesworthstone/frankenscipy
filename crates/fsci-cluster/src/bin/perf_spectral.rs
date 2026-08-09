// Correctness + A/B for spectral_clustering (randomized_eigh) vs a full-eigh spectral
// clustering, on well-separated RBF-affinity blobs. The randomized version must recover the
// blobs; the speedup is the wall-clock ratio (the eigendecomposition dominates).
use fsci_cluster::{kmeans, spectral_clustering};
use fsci_linalg::{DecompOptions, eigh};
use std::hint::black_box;
use std::time::Instant;

fn build(n_per: usize, n_clusters: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let n = n_per * n_clusters;
    let mut pts = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for c in 0..n_clusters {
        for _ in 0..n_per {
            pts.push(vec![20.0 * c as f64 + rng(), rng(), rng()]);
            truth.push(c);
        }
    }
    // RBF affinity, sigma=1.
    let aff: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d2: f64 = pts[i]
                        .iter()
                        .zip(&pts[j])
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    (-0.5 * d2).exp()
                })
                .collect()
        })
        .collect();
    (aff, truth)
}

fn purity(labels: &[usize], truth: &[usize], k: usize) -> f64 {
    let n = labels.len();
    let mut correct = 0usize;
    for pred in 0..k {
        let mut counts = vec![0usize; k];
        for i in 0..n {
            if labels[i] == pred {
                counts[truth[i]] += 1;
            }
        }
        correct += counts.iter().copied().max().unwrap_or(0);
    }
    correct as f64 / n as f64
}

fn full_eigh_spectral(aff: &[Vec<f64>], k: usize, seed: u64) -> Vec<usize> {
    let n = aff.len();
    let inv_sqrt: Vec<f64> = aff
        .iter()
        .map(|r| {
            let deg: f64 = r.iter().sum();
            if deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 }
        })
        .collect();
    let norm: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| aff[i][j] * inv_sqrt[i] * inv_sqrt[j])
                .collect()
        })
        .collect();
    let e = eigh(&norm, DecompOptions::default()).expect("eigh"); // ascending
    // Top-k eigenvectors = last k columns.
    let mut emb = vec![vec![0.0; k]; n];
    for (i, row) in emb.iter_mut().enumerate() {
        let mut nrm = 0.0;
        for (t, slot) in row.iter_mut().enumerate() {
            let v = e.eigenvectors[i][n - k + t];
            *slot = v;
            nrm += v * v;
        }
        let nrm = nrm.sqrt();
        if nrm > 1e-12 {
            for s in row.iter_mut() {
                *s /= nrm;
            }
        }
    }
    kmeans(&emb, k, 100, seed).unwrap().labels
}

fn main() {
    let k = 3usize;
    let (aff, truth) = build(500, k); // n = 1500
    let n = aff.len();

    let labels = spectral_clustering(&aff, k, 100, 7).expect("spectral");
    println!(
        "spectral_clustering purity = {:.4}  (n={n} k={k})",
        purity(&labels, &truth, k)
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(spectral_clustering(&aff, k, 100, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_eigh_spectral(&aff, k, 7));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!(
        "full-eigh spectral {f_ms:.2} ms | randomized spectral_clustering {r_ms:.2} ms | speedup {:.2}x",
        f_ms / r_ms
    );
}
