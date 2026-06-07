//! Same-process A/B + isomorphism harness for `fcluster` (maxclust).
//!
//! `old_fcluster` is a verbatim copy of the original O(n^2) per-merge relabel.
//! The library now agglomerates via union-find with per-set min-leaf tracking,
//! which is byte-identical (same partition, same min-leaf label, same 1..k
//! renumbering) at O(n·α(n)). We assert 0 mismatches across sizes and cut counts
//! and time the win on large linkages with few clusters.
//! Run: `cargo run --release -p fsci-cluster --bin perf_fcluster`.

use fsci_cluster::{LinkageMethod, fcluster, linkage};
use std::time::Instant;

/// Verbatim copy of the original O(n^2)-relabel fcluster maxclust cut.
fn old_fcluster(z: &[[f64; 4]], max_clusters: usize) -> Vec<usize> {
    let n = z.len() + 1;
    if max_clusters >= n || max_clusters == 0 {
        return (1..=n).collect();
    }
    let mut cluster_of = vec![0usize; 2 * n - 1];
    for (i, c) in cluster_of.iter_mut().enumerate().take(n) {
        *c = i;
    }
    let n_merges = n - max_clusters;
    for (step, row) in z.iter().enumerate().take(n_merges) {
        let new_id = n + step;
        let ci = row[0] as usize;
        let cj = row[1] as usize;
        let label = cluster_of[ci].min(cluster_of[cj]);
        let old_ci = cluster_of[ci];
        let old_cj = cluster_of[cj];
        for v in cluster_of.iter_mut().take(new_id + 1) {
            if *v == old_ci || *v == old_cj {
                *v = label;
            }
        }
        cluster_of[new_id] = label;
    }
    let leaf_labels: Vec<usize> = cluster_of[..n].to_vec();
    let mut unique: Vec<usize> = leaf_labels.clone();
    unique.sort_unstable();
    unique.dedup();
    leaf_labels
        .iter()
        .map(|&l| unique.binary_search(&l).unwrap_or(0) + 1)
        .collect()
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

fn make_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f64() * 10.0).collect())
        .collect()
}

fn main() {
    let methods = [
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
        LinkageMethod::Centroid,
        LinkageMethod::Median,
    ];
    let mut mismatches = 0usize;
    let mut total = 0usize;
    let mut payload = String::new();
    for &n in &[2usize, 3, 8, 33, 120] {
        for (mi, &m) in methods.iter().enumerate() {
            let data = make_data(n, 2, (n as u64) * 17 + mi as u64 + 1);
            let Ok(z) = linkage(&data, m) else { continue };
            for &k in &[1usize, 2, 3, 5, n / 2 + 1, n] {
                if k == 0 {
                    continue;
                }
                let got = fcluster(&z, k).unwrap();
                let want = old_fcluster(&z, k);
                total += 1;
                if got != want {
                    mismatches += 1;
                    if payload.len() < 1500 {
                        payload.push_str(&format!("MISMATCH n={n} m={mi} k={k}\n"));
                    }
                }
                if payload.len() < 1500 {
                    let chk: usize = got.iter().sum();
                    payload.push_str(&format!("n={n} m={mi} k={k} chk={chk}\n"));
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} cases (0 == identical labels)");

    // ---- Timing: large linkage, few clusters (the O(n^2) relabel worst case) ----
    for &n in &[1000usize, 2000, 4000] {
        let data = make_data(n, 3, 99);
        let z = linkage(&data, LinkageMethod::Average).unwrap();
        let k = 2usize;

        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..10 {
            acc = acc.wrapping_add(old_fcluster(&z, k).iter().sum::<usize>());
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..10 {
            acc = acc.wrapping_add(fcluster(&z, k).unwrap().iter().sum::<usize>());
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>5} k={k}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc})",
            old_t / 10,
            new_t / 10
        );
    }
}
