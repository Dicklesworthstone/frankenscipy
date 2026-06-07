//! Profiling-only harness for cluster vector-quantization hot paths.
//!
//! NOT a product binary. It exists so RCH, hyperfine, and sha256 checks can
//! attach to a tight deterministic nearest-centroid assignment scenario.
//! Build under `release-perf`:
//!
//! ```bash
//! cargo build -p fsci-cluster --profile release-perf --bin perf_cluster
//! ```
//!
//! Usage: `perf_cluster <mode> <n> <d> <k> <repeats>`
//!   mode    = vq | kmeans | dbscan | kmedoids | silhouette-samples | golden
//!   n       = number of observations
//!   d       = feature dimension
//!   k       = number of centroids / clusters
//!   repeats = timed iterations
//!
//! `golden` ignores the size args and emits bit-exact assignment output for a
//! fixed sweep of shapes so the optimization can be proven isomorphic.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_cluster::{dbscan, kmeans, kmedoids, silhouette_samples, vq};

/// Reference k-medoids mirroring the pre-optimization library algorithm: nearest
/// medoid assignment and the M×M intra-cluster distance matrix both index
/// scattered `Vec<Vec<f64>>` rows (`data[med]`, `data[members[i]]`). Used by the
/// `kmedoids-base` mode so this harness can A/B the contiguous-buffer flatten
/// lever within one binary build. Returns `(labels, inertia, n_iter)`.
fn kmedoids_baseline(
    data: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<usize>, f64, usize) {
    let n = data.len();
    let mut rng = seed;
    let mut medoid_indices: Vec<usize> = Vec::with_capacity(k);
    let mut used = vec![false; n];
    for _ in 0..k {
        loop {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng >> 33) as usize % n;
            if !used[idx] {
                used[idx] = true;
                medoid_indices.push(idx);
                break;
            }
        }
    }

    let mut labels = vec![0usize; n];
    let mut actual_iter = 0;
    for iter in 0..max_iter {
        actual_iter = iter + 1;
        for i in 0..n {
            let mut min_dist = f64::INFINITY;
            for (c, &med) in medoid_indices.iter().enumerate() {
                let dd = sq_dist_ref(&data[i], &data[med]);
                if dd < min_dist {
                    min_dist = dd;
                    labels[i] = c;
                }
            }
        }
        let mut changed = false;
        for (c, medoid_index) in medoid_indices.iter_mut().enumerate().take(k) {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            if members.is_empty() {
                continue;
            }
            let m = members.len();
            let mut dmat = vec![vec![0.0_f64; m]; m];
            for i in 0..m {
                for j in (i + 1)..m {
                    let dd = sq_dist_ref(&data[members[i]], &data[members[j]]).sqrt();
                    dmat[i][j] = dd;
                    dmat[j][i] = dd;
                }
            }
            let mut best_local = 0usize;
            let mut best_cost = dmat[0].iter().sum::<f64>();
            for (i, row) in dmat.iter().enumerate().skip(1) {
                let cost: f64 = row.iter().sum();
                if cost < best_cost {
                    best_cost = cost;
                    best_local = i;
                }
            }
            let best_med = members[best_local];
            if best_med != *medoid_index {
                *medoid_index = best_med;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let inertia: f64 = (0..n)
        .map(|i| sq_dist_ref(&data[i], &data[medoid_indices[labels[i]]]))
        .sum();
    (labels, inertia, actual_iter)
}

/// Reference (non-abandoning) squared Euclidean distance, mirroring the library's
/// pre-optimization `sq_dist`. Used only by the `vq-base` mode so this harness
/// can A/B the partial-distance early-abandonment lever within a single binary.
fn sq_dist_ref(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum()
}

/// Reference DBSCAN with the pre-optimization naive neighbor scan (scattered
/// `Vec<Vec<f64>>` rows, full unbounded `sq_dist`). Mirrors the library algorithm
/// exactly except for the contiguous-buffer + `eps2`-abandonment lever, so the
/// `dbscan-base` mode can A/B that single lever within one binary. Returns
/// `(n_clusters, labels, n_cores)`.
fn dbscan_baseline(data: &[Vec<f64>], eps: f64, min_samples: usize) -> (i64, Vec<i64>, usize) {
    let n = data.len();
    let eps2 = eps * eps;
    let mut labels = vec![-1i64; n];
    let mut visited = vec![false; n];
    let mut core_samples: Vec<usize> = Vec::new();
    let mut cluster_id = 0i64;

    let neighbors = |idx: usize| -> Vec<usize> {
        (0..n)
            .filter(|&j| sq_dist_ref(&data[idx], &data[j]) <= eps2)
            .collect()
    };

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
        let mut queue: std::collections::VecDeque<usize> = nbrs.into();
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
    core_samples.sort_unstable();
    core_samples.dedup();
    (cluster_id, labels, core_samples.len())
}

/// Reference nearest-centroid assignment with full distance evaluation.
fn vq_baseline(data: &[Vec<f64>], centroids: &[Vec<f64>]) -> (Vec<usize>, Vec<f64>) {
    let mut labels = Vec::with_capacity(data.len());
    let mut dists = Vec::with_capacity(data.len());
    for point in data {
        let mut min_sq = f64::INFINITY;
        let mut best_c = 0;
        for (c, centroid) in centroids.iter().enumerate() {
            let sd = sq_dist_ref(point, centroid);
            if sd < min_sq {
                min_sq = sd;
                best_c = c;
            }
        }
        labels.push(best_c);
        dists.push(min_sq.sqrt());
    }
    (labels, dists)
}

/// Reference silhouette-samples implementation mirroring the current library
/// algorithm before the bucket-pass lever. Used by `silhouette-samples-base` so
/// the post-optimization benchmark can compare old vs new in one binary.
fn silhouette_samples_baseline(data: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
    let n = data.len();
    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    (0..n)
        .map(|i| {
            let li = labels[i];

            let mut a_sum = 0.0;
            let mut a_count = 0usize;
            for j in 0..n {
                if i != j && labels[j] == li {
                    a_sum += sq_dist_ref(&data[i], &data[j]).sqrt();
                    a_count += 1;
                }
            }
            let a = if a_count > 0 {
                a_sum / a_count as f64
            } else {
                0.0
            };

            let mut b = f64::INFINITY;
            for c in 0..k {
                if c == li {
                    continue;
                }
                let mut c_sum = 0.0;
                let mut c_count = 0usize;
                for j in 0..n {
                    if labels[j] == c {
                        c_sum += sq_dist_ref(&data[i], &data[j]).sqrt();
                        c_count += 1;
                    }
                }
                if c_count > 0 {
                    b = b.min(c_sum / c_count as f64);
                }
            }

            if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            }
        })
        .collect()
}

/// Reference bucket-pass implementation matching the immediate pre-symmetric
/// library path: one `j = 0..n` scan per anchor, routed into cluster buckets.
fn silhouette_samples_bucket_baseline(data: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
    let n = data.len();
    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut samples = Vec::with_capacity(n);
    let mut cluster_sum = vec![0.0_f64; k];
    let mut cluster_count = vec![0usize; k];

    for i in 0..n {
        let li = labels[i];

        cluster_sum.fill(0.0);
        cluster_count.fill(0);
        for j in 0..n {
            if i == j {
                continue;
            }
            let lj = labels[j];
            cluster_sum[lj] += sq_dist_ref(&data[i], &data[j]).sqrt();
            cluster_count[lj] += 1;
        }

        let a = if cluster_count[li] > 0 {
            cluster_sum[li] / cluster_count[li] as f64
        } else {
            0.0
        };

        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == li || cluster_count[c] == 0 {
                continue;
            }
            let mean_c = cluster_sum[c] / cluster_count[c] as f64;
            if mean_c < b {
                b = mean_c;
            }
        }

        samples.push(if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        });
    }

    samples
}

/// Deterministic clustered data: `k` latent centers on a lattice, each point
/// drawn near one center with a reproducible LCG jitter. No external RNG so the
/// golden output is stable across machines.
fn make_clustered_data(n: usize, d: usize, k: usize) -> Vec<Vec<f64>> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let next = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|i| {
            let center = i % k;
            (0..d)
                .map(|j| {
                    let base = (center as f64) * 10.0 + (j as f64) * 0.5;
                    base + (next(&mut state) - 0.5) * 2.0
                })
                .collect()
        })
        .collect()
}

/// Fixed centroids on the same lattice the data clusters around.
fn make_centroids(d: usize, k: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|center| {
            (0..d)
                .map(|j| (center as f64) * 10.0 + (j as f64) * 0.5)
                .collect()
        })
        .collect()
}

fn make_labels(n: usize, k: usize) -> Vec<usize> {
    (0..n).map(|i| i % k).collect()
}

fn golden_text() -> String {
    let mut output = String::new();
    for &(n, d, k) in &[
        (64usize, 8usize, 4usize),
        (128, 16, 8),
        (200, 32, 12),
        (300, 64, 16),
    ] {
        let data = make_clustered_data(n, d, k);
        let centroids = make_centroids(d, k);
        let (labels, dists) = vq(&data, &centroids).expect("vq");
        write!(
            &mut output,
            "mode=vq n={n} d={d} k={k} len={} ",
            labels.len()
        )
        .expect("write vq header");
        for (&label, &dist) in labels.iter().zip(dists.iter()) {
            write!(&mut output, "{label}:{:016x} ", dist.to_bits()).expect("write vq bits");
        }
        output.push('\n');

        // kmeans is deterministic for a fixed seed; capture labels + inertia bits.
        let result = kmeans(&data, k, 50, 0x1234_5678).expect("kmeans");
        write!(
            &mut output,
            "mode=kmeans n={n} d={d} k={k} n_iter={} inertia={:016x} ",
            result.n_iter,
            result.inertia.to_bits()
        )
        .expect("write kmeans header");
        for &label in &result.labels {
            write!(&mut output, "{label} ").expect("write kmeans labels");
        }
        output.push('\n');

        // DBSCAN: eps chosen near the intra-cluster jitter radius so the scan
        // exercises both neighbor hits and (mostly) misses.
        let result = dbscan(&data, 3.0, 4).expect("dbscan");
        write!(
            &mut output,
            "mode=dbscan n={n} d={d} k={k} n_clusters={} cores={} ",
            result.n_clusters,
            result.core_sample_indices.len()
        )
        .expect("write dbscan header");
        for &label in &result.labels {
            write!(&mut output, "{label} ").expect("write dbscan labels");
        }
        for &idx in &result.core_sample_indices {
            write!(&mut output, "c{idx} ").expect("write dbscan cores");
        }
        output.push('\n');

        // k-medoids is deterministic for a fixed seed; capture labels + inertia
        // bits + n_iter so the flatten lever is proven bit-identical.
        let result = kmedoids(&data, k, 50, 0x1234_5678).expect("kmedoids");
        write!(
            &mut output,
            "mode=kmedoids n={n} d={d} k={k} n_iter={} inertia={:016x} ",
            result.n_iter,
            result.inertia.to_bits()
        )
        .expect("write kmedoids header");
        for &label in &result.labels {
            write!(&mut output, "{label} ").expect("write kmedoids labels");
        }
        output.push('\n');

        let labels = make_labels(n, k);
        let samples = silhouette_samples(&data, &labels).expect("silhouette_samples");
        write!(
            &mut output,
            "mode=silhouette-samples n={n} d={d} k={k} len={} ",
            samples.len()
        )
        .expect("write silhouette header");
        for &sample in &samples {
            write!(&mut output, "{:016x} ", sample.to_bits()).expect("write silhouette bits");
        }
        output.push('\n');
    }
    output
}

fn write_or_print_golden(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        print!("{output}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("vq");

    if mode == "golden" {
        write_or_print_golden(golden_text(), args.get(2).map(String::as_str));
        return;
    }

    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let d: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32);
    let repeats: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(50);

    let data = make_clustered_data(n, d, k);

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    if mode == "vq" {
        let centroids = make_centroids(d, k);
        for _ in 0..repeats {
            let (labels, dists) = vq(black_box(&data), black_box(&centroids)).expect("vq");
            checksum += dists.iter().sum::<f64>() + labels.iter().sum::<usize>() as f64;
            black_box(&labels);
        }
    } else if mode == "vq-base" {
        let centroids = make_centroids(d, k);
        for _ in 0..repeats {
            let (labels, dists) = vq_baseline(black_box(&data), black_box(&centroids));
            checksum += dists.iter().sum::<f64>() + labels.iter().sum::<usize>() as f64;
            black_box(&labels);
        }
    } else if mode == "kmeans" {
        for _ in 0..repeats {
            let result = kmeans(black_box(&data), k, 50, 0x1234_5678).expect("kmeans");
            checksum += result.inertia + result.labels.iter().sum::<usize>() as f64;
            black_box(&result.labels);
        }
    } else if mode == "dbscan" {
        for _ in 0..repeats {
            let result = dbscan(black_box(&data), 3.0, 4).expect("dbscan");
            checksum += result.n_clusters as f64
                + result.labels.iter().map(|&l| l as f64).sum::<f64>()
                + result.core_sample_indices.len() as f64;
            black_box(&result.labels);
        }
    } else if mode == "dbscan-base" {
        for _ in 0..repeats {
            let (n_clusters, labels, n_cores) = dbscan_baseline(black_box(&data), 3.0, 4);
            checksum +=
                n_clusters as f64 + labels.iter().map(|&l| l as f64).sum::<f64>() + n_cores as f64;
            black_box(&labels);
        }
    } else if mode == "kmedoids" {
        for _ in 0..repeats {
            let result = kmedoids(black_box(&data), k, 50, 0x1234_5678).expect("kmedoids");
            checksum += result.inertia + result.labels.iter().sum::<usize>() as f64;
            black_box(&result.labels);
        }
    } else if mode == "kmedoids-base" {
        for _ in 0..repeats {
            let (labels, inertia, n_iter) = kmedoids_baseline(black_box(&data), k, 50, 0x1234_5678);
            checksum += inertia + labels.iter().sum::<usize>() as f64 + n_iter as f64;
            black_box(&labels);
        }
    } else if mode == "silhouette-samples" {
        let labels = make_labels(n, k);
        for _ in 0..repeats {
            let samples =
                silhouette_samples(black_box(&data), black_box(&labels)).expect("silhouette");
            checksum += samples.iter().sum::<f64>();
            black_box(&samples);
        }
    } else if mode == "silhouette-samples-base" {
        let labels = make_labels(n, k);
        for _ in 0..repeats {
            let samples = silhouette_samples_baseline(black_box(&data), black_box(&labels));
            checksum += samples.iter().sum::<f64>();
            black_box(&samples);
        }
    } else if mode == "silhouette-samples-bucket-base" {
        let labels = make_labels(n, k);
        for _ in 0..repeats {
            let samples = silhouette_samples_bucket_baseline(black_box(&data), black_box(&labels));
            checksum += samples.iter().sum::<f64>();
            black_box(&samples);
        }
    } else {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_ms = total_ms / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"d\":{d},\"k\":{k},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
    );
}
