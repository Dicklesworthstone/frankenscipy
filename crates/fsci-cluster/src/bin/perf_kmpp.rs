//! Same-process A/B + isomorphism harness for kmeans (k-means++ init).
//!
//! `kmeans_old_init` reproduces the original O(n*k^2) rebuild-from-scratch init;
//! the library now carries the nearest-centroid distances across picks (O(n*k)).
//! We prove the full kmeans result (centroids, labels, inertia) is byte-identical
//! across random datasets and k, then time the win at large k.
//! Run: `cargo run --release -p fsci-cluster --bin perf_kmpp`.
#![allow(clippy::needless_range_loop)]

use fsci_cluster::kmeans;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Verbatim original O(n*k^2) k-means++ init (rebuild dists every pick), then a
/// Lloyd loop matching the library's so the whole result is comparable.
fn kmeans_old_init(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
    let n = data.len();
    let d = data[0].len();
    let mut rng = seed;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };

    // ---- old init ----
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let idx = (next(&mut rng) * n as f64) as usize % n;
    centroids.push(data[idx].clone());
    for _ in 1..k {
        let mut dists = vec![f64::INFINITY; n];
        for (i, point) in data.iter().enumerate() {
            for c in &centroids {
                let dd = sq_dist(point, c);
                dists[i] = dists[i].min(dd);
            }
        }
        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            let idx = (next(&mut rng) * n as f64) as usize % n;
            centroids.push(data[idx].clone());
            continue;
        }
        let threshold = next(&mut rng) * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &dd) in dists.iter().enumerate() {
            cumsum += dd;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    // ---- Lloyd (mirrors the library) ----
    let mut labels = vec![0usize; n];
    let mut inertia = f64::INFINITY;
    for _iter in 0..max_iter {
        let mut new_inertia = 0.0;
        for (i, point) in data.iter().enumerate() {
            let mut best_c = 0;
            let mut min_sq = sq_dist(point, &centroids[0]);
            for c in 0..k {
                let sd = sq_dist(point, &centroids[c]);
                if sd < min_sq || (sd == min_sq && c < best_c) {
                    min_sq = sd;
                    best_c = c;
                }
            }
            labels[i] = best_c;
            new_inertia += min_sq;
        }
        if (inertia - new_inertia).abs() < 1e-12 * inertia.abs().max(1.0) {
            return (centroids, labels, new_inertia);
        }
        inertia = new_inertia;
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![0.0; d]; k];
        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for (j, &val) in point.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for val in &mut new_centroids[c][..d] {
                    *val /= counts[c] as f64;
                }
            } else {
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }
    (centroids, labels, inertia)
}

fn main() {
    let mut r = Lcg(0x6d4e_a17c_9b30_5521);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..400 {
        let n = 50 + (r.next_u64() as usize % 400);
        let d = 1 + (r.next_u64() as usize % 5);
        let coarse = trial % 3 == 0; // duplicate points -> exercise total<=0 branch
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        if coarse {
                            (r.next_u64() % 4) as f64
                        } else {
                            r.unit() * 10.0
                        }
                    })
                    .collect()
            })
            .collect();
        let k = 1 + (r.next_u64() as usize % n.min(40));
        let seed = r.next_u64();
        let max_iter = 20;

        let got = kmeans(&data, k, max_iter, seed).unwrap();
        let (wc, wl, wi) = kmeans_old_init(&data, k, max_iter, seed);
        total += 1;
        let cent_eq = got.centroids.len() == wc.len()
            && got
                .centroids
                .iter()
                .zip(&wc)
                .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        if !cent_eq || got.labels != wl || got.inertia.to_bits() != wi.to_bits() {
            mismatches += 1;
            if payload.len() < 3000 {
                payload.push_str(&format!("MISMATCH trial={trial} n={n} d={d} k={k}\n"));
            }
        }
        let digest: u64 = got
            .centroids
            .iter()
            .flatten()
            .fold(1469598103934665603u64, |h, x| {
                (h ^ x.to_bits()).wrapping_mul(1099511628211)
            });
        payload.push_str(&format!(
            "trial={trial} n={n} d={d} k={k} digest={digest:016x}\n"
        ));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} kmeans runs (0 == byte-identical)");

    // ---- Timing: large k (init O(n*k^2) dominates), few iters ----
    for &(n, k) in &[(4000usize, 128usize), (4000, 256), (8000, 256)] {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![r.unit() * 50.0, r.unit() * 50.0])
            .collect();
        let seed = 12345;
        let iters = 3;

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += kmeans_old_init(&data, k, iters, seed).2;
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += kmeans(&data, k, iters, seed).unwrap().inertia;
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>5} k={k:>4}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            old_t / 3,
            new_t / 3
        );
    }
}
