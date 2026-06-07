//! Same-process A/B + isomorphism harness for dbscan.
//!
//! `naive_dbscan` reproduces the original O(n^2) all-pairs region-query loop;
//! the library now uses a uniform spatial grid for low dimensions. We prove the
//! labels, core indices, and cluster count are identical across random and
//! clustered point sets and a range of eps / min_samples, then time the win.
//! Run: `cargo run --release -p fsci-cluster --bin perf_dbscan`.
#![allow(clippy::needless_range_loop)]

use fsci_cluster::dbscan;
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

/// Verbatim original O(n^2) DBSCAN (all-pairs region queries).
fn naive_dbscan(data: &[Vec<f64>], eps: f64, min_samples: usize) -> (Vec<i64>, Vec<usize>, usize) {
    let n = data.len();
    let d = data[0].len();
    let eps2 = eps * eps;
    let sqd = |a: &[f64], b: &[f64]| -> f64 { (0..d).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum() };
    let neighbors = |idx: usize| -> Vec<usize> {
        (0..n)
            .filter(|&j| sqd(&data[idx], &data[j]) <= eps2)
            .collect()
    };

    let mut labels = vec![-1i64; n];
    let mut visited = vec![false; n];
    let mut core_samples = Vec::new();
    let mut cluster_id = 0i64;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let nbrs = neighbors(i);
        if nbrs.len() < min_samples {
            continue;
        }
        core_samples.push(i);
        labels[i] = cluster_id;
        let mut queue = std::collections::VecDeque::from(nbrs);
        while let Some(j) = queue.pop_front() {
            if labels[j] == -1 {
                labels[j] = cluster_id;
            }
            if visited[j] {
                continue;
            }
            visited[j] = true;
            let j_nbrs = neighbors(j);
            if j_nbrs.len() >= min_samples {
                core_samples.push(j);
                for &nb in &j_nbrs {
                    if labels[nb] == -1 {
                        queue.push_back(nb);
                    }
                }
            }
        }
        cluster_id += 1;
    }
    core_samples.sort();
    core_samples.dedup();
    (labels, core_samples, cluster_id as usize)
}

/// n points: a few gaussian-ish blobs plus uniform noise.
fn make_points(r: &mut Lcg, n: usize, d: usize, blobs: usize, spread: f64) -> Vec<Vec<f64>> {
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(blobs);
    for _ in 0..blobs {
        let mut c = vec![0.0; d];
        for v in c.iter_mut() {
            *v = r.unit() * 20.0;
        }
        centers.push(c);
    }
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let mut p = vec![0.0; d];
        if r.unit() < 0.85 {
            let ci = (r.next_u64() as usize) % blobs;
            for k in 0..d {
                p[k] = centers[ci][k] + (r.unit() - 0.5) * spread;
            }
        } else {
            for v in p.iter_mut() {
                *v = r.unit() * 20.0;
            }
        }
        points.push(p);
    }
    points
}

fn main() {
    let mut r = Lcg(0x5151_a7c0_dead_1234);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..400 {
        let n = 256 + (r.next_u64() as usize % 700);
        let d = 2 + (r.next_u64() as usize % 4); // 2..=5
        let blobs = 2 + (r.next_u64() as usize % 5);
        let spread = 2.0 + r.unit() * 3.0;
        let data = make_points(&mut r, n, d, blobs, spread);

        for &(eps, ms) in &[(0.5, 3usize), (1.0, 4), (1.5, 5), (2.5, 8)] {
            let (la, ca, na) = naive_dbscan(&data, eps, ms);
            let got = dbscan(&data, eps, ms).unwrap();
            total += 1;
            if got.labels != la || got.core_sample_indices != ca || got.n_clusters != na {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!(
                        "MISMATCH trial={trial} n={n} d={d} eps={eps} ms={ms}\n"
                    ));
                }
            }
            let digest: u64 = got.labels.iter().fold(1469598103934665603u64, |h, &l| {
                (h ^ l as u64).wrapping_mul(1099511628211)
            });
            payload.push_str(&format!(
                "trial={trial} n={n} d={d} eps={eps} ms={ms} nclust={} digest={digest:016x}\n",
                got.n_clusters
            ));
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} dbscan runs (0 == identical)");

    // ---- Timing: clustered 2-D data, growing n ----
    for &n in &[2000usize, 8000, 20000] {
        let data = make_points(&mut r, n, 2, 6, 3.0);
        let eps = 1.0;
        let ms = 5;

        let t0 = Instant::now();
        let mut acc = 0i64;
        for _ in 0..3 {
            acc += naive_dbscan(&data, eps, ms).2 as i64;
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += dbscan(&data, eps, ms).unwrap().n_clusters as i64;
        }
        let grid_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / grid_t.as_secs_f64();
        println!(
            "n={n:>6}  naive={:>10.3?}  grid={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc})",
            naive_t / 3,
            grid_t / 3
        );
    }
}
