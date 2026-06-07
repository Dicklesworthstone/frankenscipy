//! Same-process A/B + isomorphism harness for `k_nearest_neighbors`.
//!
//! `old_knn` is a verbatim copy of the original brute force (all m=n-1 distances
//! per query, then select_nth + sort the k-prefix). The library now descends a
//! KD-tree with a bounded composite-keyed set, which is byte-identical: same k
//! indices in the same composite order, identical distance bits. We assert 0
//! mismatches across shapes/dims/k/tie-densities and time the win.
//! Run: `cargo run --release -p fsci-spatial --bin perf_knn_all`.

use fsci_spatial::{euclidean, k_nearest_neighbors};
use std::time::Instant;

fn old_knn(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data.len();
    let mut all_indices = Vec::with_capacity(n);
    let mut all_distances = Vec::with_capacity(n);
    let by_dist_then_idx =
        |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean(&data[i], &data[j])))
            .collect();
        let k_actual = k.min(dists.len());
        if k_actual < dists.len() {
            dists.select_nth_unstable_by(k_actual, by_dist_then_idx);
        }
        dists[..k_actual].sort_by(by_dist_then_idx);
        all_indices.push(dists[..k_actual].iter().map(|&(idx, _)| idx).collect());
        all_distances.push(dists[..k_actual].iter().map(|&(_, d)| d).collect());
    }
    (all_indices, all_distances)
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn make_case(n: usize, dim: usize, grid: u64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let v = rng.next_f64();
                    if grid == 0 {
                        v
                    } else {
                        (v * grid as f64).floor()
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[1usize, 2, 5, 17, 64, 200] {
        for &dim in &[1usize, 2, 3] {
            for &k in &[1usize, 3, 7, 50] {
                for &grid in &[0u64, 4, 9] {
                    for seed in 0..3u64 {
                        let data = make_case(n, dim, grid, seed * 1301 + 1);
                        let (gi, gd) = k_nearest_neighbors(&data, k);
                        let (wi, wd) = old_knn(&data, k);
                        total += 1;
                        let idx_ok = gi == wi;
                        let dist_ok = gd.len() == wd.len()
                            && gd.iter().zip(&wd).all(|(a, b)| {
                                a.len() == b.len()
                                    && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                            });
                        if !idx_ok || !dist_ok {
                            mismatches += 1;
                            if payload.len() < 1500 {
                                payload.push_str(&format!(
                                    "MISMATCH n={n} dim={dim} k={k} grid={grid} seed={seed} idx_ok={idx_ok} dist_ok={dist_ok}\n"
                                ));
                            }
                        }
                        if payload.len() < 1500 {
                            let chk: usize = gi.iter().flatten().sum();
                            payload.push_str(&format!(
                                "n={n} dim={dim} k={k} grid={grid} seed={seed} idxsum={chk}\n"
                            ));
                        }
                    }
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism: {mismatches} mismatches / {total} cases (0 == byte-identical indices + distance bits)"
    );

    for &(n, dim, k) in &[(4000usize, 2usize, 5usize), (8000, 3, 10), (16000, 2, 5)] {
        let data = make_case(n, dim, 0, 7);

        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..3 {
            acc = acc.wrapping_add(old_knn(&data, k).0.iter().flatten().sum::<usize>());
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc = acc.wrapping_add(
                k_nearest_neighbors(&data, k)
                    .0
                    .iter()
                    .flatten()
                    .sum::<usize>(),
            );
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6} dim={dim} k={k:>3}  old={:>11.3?}  new={:>11.3?}  ratio={ratio:>7.1}x  (acc={acc})",
            old_t / 3,
            new_t / 3
        );
    }
}
