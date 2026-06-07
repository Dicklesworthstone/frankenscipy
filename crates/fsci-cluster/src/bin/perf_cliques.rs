//! Same-process A/B + isomorphism harness for proximity_cliques.
//!
//! `naive_cliques` reproduces the original O(n^2) all-pairs adjacency build (then
//! the SAME Bron-Kerbosch via the library); the library now builds adjacency with
//! a uniform spatial grid for low dimensions. We prove the returned clique list is
//! identical across random/clustered point sets and eps sweeps, then time it.
//! Run: `cargo run --release -p fsci-cluster --bin perf_cliques`.
#![allow(clippy::needless_range_loop)]

use fsci_cluster::proximity_cliques;
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
    let mut r = Lcg(0xc11f_e5ab_77d0_1234);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..250 {
        // Keep the proximity graph SPARSE so the (no-pivot, exponential)
        // Bron-Kerbosch stays trivial — the win being proven is the O(n^2)->O(n)
        // adjacency build, not the clique enumeration. n>=256 hits the grid path.
        let n = 256 + (r.next_u64() as usize % 80);
        let d = 2 + (r.next_u64() as usize % 4); // 2..=5
        let blobs = 4 + (r.next_u64() as usize % 6);
        let spread = 12.0 + r.unit() * 8.0;
        let data = make_points(&mut r, n, d, blobs, spread);

        for &eps in &[0.3, 0.5, 0.8] {
            let got = proximity_cliques(&data, eps);
            // Reference: the grid path is gated on n>=256; force the naive path
            // by calling with a tiny copy is not equivalent, so reconstruct the
            // naive adjacency + same library Bron-Kerbosch by using eps but on a
            // shadow that always uses the scan -- emulate by comparing against a
            // direct O(n^2) clique build below.
            let want = naive_cliques(&data, eps);
            total += 1;
            if got != want {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!(
                        "MISMATCH trial={trial} n={n} d={d} eps={eps} got={} want={}\n",
                        got.len(),
                        want.len()
                    ));
                }
            }
            let digest = digest_cliques(&got);
            payload.push_str(&format!(
                "trial={trial} n={n} d={d} eps={eps} ncliques={} digest={digest:016x}\n",
                got.len()
            ));
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} clique lists (0 == identical)");

    // ---- Timing: near-empty proximity graph (tiny eps) isolates the O(n^2)
    // adjacency build from the (trivial here) clique enumeration.
    for &n in &[2000usize, 6000, 12000] {
        let data = make_points(&mut r, n, 2, 8, 40.0);
        let eps = 0.05;

        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..3 {
            acc += naive_cliques(&data, eps).len();
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += proximity_cliques(&data, eps).len();
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

/// Naive O(n^2)-adjacency clique build, mirroring the original library code
/// (adjacency then maximal-clique enumeration). Used as the byte-for-byte oracle.
fn naive_cliques(data: &[Vec<f64>], eps: f64) -> Vec<Vec<usize>> {
    let n = data.len();
    if n == 0 || !eps.is_finite() || eps < 0.0 {
        return vec![];
    }
    let d = data[0].len();
    let eps2 = eps * eps;
    let mut adj = vec![vec![]; n];
    for i in 0..n {
        for j in i + 1..n {
            let dist: f64 = (0..d).map(|k| (data[i][k] - data[j][k]).powi(2)).sum();
            if dist <= eps2 {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut cliques = Vec::new();
    bron_kerbosch(
        &adj,
        &mut vec![],
        &mut (0..n).collect(),
        &mut vec![],
        &mut cliques,
    );
    cliques
}

/// Exact copy of the library's Bron-Kerbosch (no pivot) so the oracle's clique
/// list matches the library's for an identical adjacency.
fn bron_kerbosch(
    adj: &[Vec<usize>],
    r: &mut Vec<usize>,
    p: &mut Vec<usize>,
    x: &mut Vec<usize>,
    cliques: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if !r.is_empty() {
            cliques.push(r.clone());
        }
        return;
    }
    let p_copy = p.clone();
    for &v in &p_copy {
        r.push(v);
        let neighbors: std::collections::HashSet<usize> = adj[v].iter().cloned().collect();
        let mut new_p: Vec<usize> = p
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        let mut new_x: Vec<usize> = x
            .iter()
            .filter(|&&u| neighbors.contains(&u))
            .cloned()
            .collect();
        bron_kerbosch(adj, r, &mut new_p, &mut new_x, cliques);
        r.pop();
        p.retain(|&u| u != v);
        x.push(v);
    }
}

fn digest_cliques(cliques: &[Vec<usize>]) -> u64 {
    let mut h = 1469598103934665603u64;
    for clique in cliques {
        for &v in clique {
            h = (h ^ v as u64).wrapping_mul(1099511628211);
        }
        h = (h ^ 0xFFFF).wrapping_mul(1099511628211);
    }
    h
}
