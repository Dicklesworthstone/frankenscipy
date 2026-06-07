//! Same-process A/B + isomorphism harness for k_nearest_neighbors.
//!
//! `naive` reproduces the original full-sort-all-distances build; the library now
//! partitions (select_nth) and sorts only the k smallest. We prove the (indices,
//! distances) output is byte-identical across random/coarse data and a range of
//! k, then time the win. Run: `cargo run --release -p fsci-spatial --bin perf_knn`.
#![allow(clippy::needless_range_loop)]

use fsci_spatial::k_nearest_neighbors;
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

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Verbatim original: full stable sort of all distances, take k.
fn naive(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data.len();
    let mut all_indices = Vec::with_capacity(n);
    let mut all_distances = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean(&data[i], &data[j])))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        let k_actual = k.min(dists.len());
        all_indices.push(dists[..k_actual].iter().map(|&(idx, _)| idx).collect());
        all_distances.push(dists[..k_actual].iter().map(|&(_, d)| d).collect());
    }
    (all_indices, all_distances)
}

fn make(r: &mut Lcg, n: usize, d: usize, coarse: bool) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    if coarse {
                        (r.next_u64() % 5) as f64
                    } else {
                        r.unit() * 10.0
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut r = Lcg(0x7a3c_91de_4f02_8b15);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..500 {
        let n = 2 + (r.next_u64() as usize % 200);
        let d = 1 + (r.next_u64() as usize % 4);
        let coarse = trial % 2 == 0; // heavy ties / coincident points
        let data = make(&mut r, n, d, coarse);
        for &k in &[1usize, 3, 7, 20, 1000] {
            let (gi, gd) = k_nearest_neighbors(&data, k);
            let (wi, wd) = naive(&data, k);
            total += 1;
            let idx_eq = gi == wi;
            let dist_eq = gd.len() == wd.len()
                && gd.iter().zip(&wd).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                });
            if !idx_eq || !dist_eq {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!("MISMATCH trial={trial} n={n} d={d} k={k}\n"));
                }
            }
        }
        let (gi, _) = k_nearest_neighbors(&data, 5);
        let digest: u64 = gi.iter().flatten().fold(1469598103934665603u64, |h, &v| {
            (h ^ v as u64).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} d={d} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} knn results (0 == byte-identical)");

    // ---- Timing: small k, low d, growing n (full sort dominates) ----
    for &n in &[2000usize, 6000, 12000] {
        let data = make(&mut r, n, 2, false);
        let k = 5;

        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..3 {
            acc += naive(&data, k).0[0][0];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += k_nearest_neighbors(&data, k).0[0][0];
        }
        let new_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6} k={k}  fullsort={:>10.3?}  partial={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc})",
            naive_t / 3,
            new_t / 3
        );
    }
}
