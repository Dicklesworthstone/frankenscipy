//! Timing + result-digest harness for the parallel kmeans assignment step.
//! Run before and after the parallelization (via `git stash`) to confirm the result
//! digest (labels + inertia + n_iter) is UNCHANGED (byte-identical) and to get the
//! speedup.

use std::hint::black_box;
use std::time::Instant;

use fsci_cluster::kmeans;

fn dataset(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| ((i as f64 * 0.013 + j as f64 * 0.7).sin() + (i % 50) as f64 * 0.1))
                .collect()
        })
        .collect()
}

fn main() {
    for &(n, k, d) in &[(20000usize, 50usize, 16usize), (40000, 80, 24)] {
        let data = dataset(n, d);
        let max_iter = 15;
        let seed = 12345;

        let r = kmeans(&data, k, max_iter, seed).expect("kmeans");
        // Result digest: FNV over labels + inertia bits + n_iter.
        let mut digest: u64 = 1469598103934665603;
        for &lab in &r.labels {
            digest ^= lab as u64;
            digest = digest.wrapping_mul(1099511628211);
        }
        digest ^= r.inertia.to_bits();
        digest = digest.wrapping_mul(1099511628211);
        digest ^= r.n_iter as u64;

        let iters = 3;
        let start = Instant::now();
        for _ in 0..iters {
            black_box(kmeans(&data, k, max_iter, seed).expect("kmeans"));
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "kmeans n={n} k={k} d={d}: {ms:>9.3} ms/run  n_iter={}  inertia={:.6e}  digest={digest:016x}",
            r.n_iter, r.inertia
        );
    }
}
