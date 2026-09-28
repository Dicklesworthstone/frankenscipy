//! Timing of `fcluster` maxclust cuts on large linkages, our arm only.
//!
//! This harness used to A/B a union-find cut against the O(n^2) per-merge relabel it
//! replaced (tests/artifacts/perf/2026-06-05-cluster-fcluster-unionfind). Both cut after
//! n - k merges and numbered clusters by their smallest member. SciPy does neither: it cuts at
//! a threshold, so tied heights merge together, and it numbers clusters in its depth-first
//! order. `fcluster` is now a port of SciPy's routine (frankenscipy-3h1yw), so neither old arm
//! is a reference any more.
//!
//! The comparison against SciPy is `perf_cluster_vs_scipy` with `FSCI_CLUSTER_OPS=fcluster`.
//! This bin only prints our per-call time over the sizes the old harness used, so the port's
//! O(n log n) search can be read against the old timings.
//! Run: `cargo run --release -p fsci-cluster --bin perf_fcluster`.

use fsci_cluster::{FclusterCriterion, LinkageMethod, fcluster, linkage};
use std::hint::black_box;
use std::time::Instant;

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
    for &n in &[1000usize, 2000, 4000] {
        let data = make_data(n, 3, 99);
        let z = linkage(&data, LinkageMethod::Average).unwrap();
        for k in [2usize, 32] {
            let started = Instant::now();
            let mut acc = 0usize;
            for _ in 0..10 {
                let labels = fcluster(black_box(&z), FclusterCriterion::MaxClust(k)).unwrap();
                acc = acc.wrapping_add(labels.iter().sum::<usize>());
            }
            println!(
                "n={n:>5} k={k:>2}  fcluster={:>10.3?}  (acc={acc})",
                started.elapsed() / 10
            );
        }
    }
}
