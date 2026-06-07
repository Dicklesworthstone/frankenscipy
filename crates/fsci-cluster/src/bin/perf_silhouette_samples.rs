//! Same-process timing + bit-identity digest harness for `silhouette_samples` /
//! `silhouette_score`.
//!
//! Each anchor's silhouette coefficient is an independent O(n·d) reduction over the
//! other points bucketed by cluster; the loop is now parallel across anchors. The
//! per-anchor bucket sum accumulates in increasing j order, identical to the former
//! upper-triangle matrix accumulation, so it is bit-identical. Dumps FNV digests of the
//! sample vector and the mean score (compare across the stashed serial build) and times
//! the calls. Run: `cargo run -p fsci-cluster --bin perf_silhouette_samples`.

use std::hint::black_box;
use std::time::Instant;

use fsci_cluster::{silhouette_samples, silhouette_score};

fn dataset(n: usize, d: usize, k: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let c = (i % k) as f64;
            (0..d)
                .map(|j| {
                    let t = (i * 7 + j * 13) as f64;
                    c * 5.0 + (0.013 * t).sin() + 0.3 * (0.07 * t + j as f64).cos()
                })
                .collect()
        })
        .collect();
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    (data, labels)
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let cases = [(1500usize, 8usize, 5usize), (3000, 12, 8), (5000, 16, 10)];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, d, k) in &cases {
        let (data, labels) = dataset(n, d, k);
        let s = silhouette_samples(&data, &labels).expect("silhouette_samples");
        let score = silhouette_score(&data, &labels).expect("silhouette_score");
        println!(
            "n={n} d={d} k={k} samples={:016x} score={:016x}",
            digest(&s),
            score.to_bits()
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, d, k) in &cases {
        let (data, labels) = dataset(n, d, k);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let s = silhouette_samples(black_box(&data), black_box(&labels)).expect("ss");
            acc += s[s.len() / 2];
        }
        let dt = t0.elapsed();
        println!("n={n:>5} d={d:>3} k={k:>3}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
