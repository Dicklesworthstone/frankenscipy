//! Same-process A/B + tolerance-parity harness for `silhouette_score_1d`.
//!
//! `old_silhouette` is a verbatim copy of the original O(n^2) all-pairs loop
//! (HashMap + Vec churn per point). The library now derives a(i) and each
//! candidate b(i) cluster-mean from per-cluster sorted prefix sums — O(n log n)
//! build + O(num_clusters * log m) per point. The two agree to f64 tolerance
//! (sum reassociation only; the metric is unchanged); we report the max abs diff
//! across many shapes and the speedup on large-n / small-k inputs.
//! Run: `cargo run --release -p fsci-stats --bin perf_silhouette`.

use fsci_stats::silhouette_score_1d;
use std::time::Instant;

/// Verbatim copy of the original O(n^2) silhouette_score_1d.
fn old_silhouette(data: &[f64], labels: &[f64]) -> f64 {
    if data.len() != labels.len() || data.len() < 2 {
        return f64::NAN;
    }
    let n = data.len();

    let unique_labels: std::collections::HashSet<i64> =
        labels.iter().map(|&l| l.round() as i64).collect();
    if unique_labels.len() < 2 {
        return 0.0;
    }

    let mut silhouettes = Vec::with_capacity(n);

    for i in 0..n {
        let li = labels[i].round() as i64;

        let mut same_cluster_dists = vec![];
        let mut other_cluster_dists: std::collections::HashMap<i64, Vec<f64>> =
            std::collections::HashMap::new();

        for j in 0..n {
            if i == j {
                continue;
            }
            let lj = labels[j].round() as i64;
            let dist = (data[i] - data[j]).abs();

            if lj == li {
                same_cluster_dists.push(dist);
            } else {
                other_cluster_dists.entry(lj).or_default().push(dist);
            }
        }

        let a = if same_cluster_dists.is_empty() {
            0.0
        } else {
            same_cluster_dists.iter().sum::<f64>() / same_cluster_dists.len() as f64
        };

        let b = other_cluster_dists
            .values()
            .map(|dists| dists.iter().sum::<f64>() / dists.len() as f64)
            .fold(f64::INFINITY, f64::min);

        let s = if a.max(b) == 0.0 {
            0.0
        } else {
            (b - a) / a.max(b)
        };
        silhouettes.push(s);
    }

    silhouettes.iter().sum::<f64>() / silhouettes.len() as f64
}

// Deterministic LCG so the harness needs no rng dependency.
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

fn make_case(n: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut data = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let c = (rng.next_f64() * k as f64) as usize % k.max(1);
        // Separated-ish clusters plus noise.
        data.push(c as f64 * 10.0 + rng.next_f64());
        labels.push(c as f64);
    }
    (data, labels)
}

fn main() {
    // ---- Tolerance parity across shapes ----
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut checked = 0usize;
    let mut payload = String::new();
    for &n in &[2usize, 3, 7, 16, 64, 257, 1000] {
        for &k in &[2usize, 3, 5, 9] {
            if k > n {
                continue;
            }
            for seed in 0..6u64 {
                let (data, labels) = make_case(n, k, seed * 7919 + 1);
                let got = silhouette_score_1d(&data, &labels);
                let want = old_silhouette(&data, &labels);
                checked += 1;
                let d = (got - want).abs();
                max_abs = max_abs.max(d);
                if want.abs() > 1e-12 {
                    max_rel = max_rel.max(d / want.abs());
                }
                if payload.len() < 1600 {
                    payload.push_str(&format!("n={n} k={k} seed={seed} got={got:.15e}\n"));
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "tolerance-parity: {checked} cases, max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e} (vs verbatim O(n^2))"
    );

    // ---- Timing: large n, modest k (point-distance work dominates) ----
    for &(n, k) in &[(4000usize, 8usize), (8000, 12), (16000, 20)] {
        let (data, labels) = make_case(n, k, 12345);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_silhouette(&data, &labels);
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += silhouette_score_1d(&data, &labels);
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6} k={k:>3}  old={:>11.3?}  new={:>11.3?}  ratio={ratio:>7.1}x  (acc={acc:.6})",
            old_t / 3,
            new_t / 3
        );
    }
}
