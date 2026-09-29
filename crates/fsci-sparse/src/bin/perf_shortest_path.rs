//! Same-process A/B + isomorphism harness for `shortest_path` (point-to-point).
//!
//! `naive` reproduces the original O(V²) linear-scan Dijkstra; the library now
//! uses a heap Dijkstra O((V+E) log V) whose (cost, position) pop order + visited
//! finalization reproduce the linear scan's selection exactly, so `(distance,
//! path)` is byte-identical. We prove `(dist.to_bits, path)` equality across many
//! random graphs — including UNIT-weight graphs where ties abound — then time the
//! win on large sparse graphs.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_shortest_path`.

use std::time::Instant;

use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D, shortest_path};

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() >> 11) as usize % n
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Verbatim original: O(V²) linear-scan Dijkstra.
fn naive(graph: &CsrMatrix, source: usize, target: usize) -> (f64, Vec<usize>) {
    let n = graph.shape().rows;
    if source >= n || target >= n {
        return (f64::INFINITY, vec![]);
    }
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    dist[source] = 0.0;
    for _ in 0..n {
        let mut u = usize::MAX;
        let mut min_d = f64::INFINITY;
        for (i, (&d, &v)) in dist.iter().zip(visited.iter()).enumerate() {
            if !v && d < min_d {
                min_d = d;
                u = i;
            }
        }
        if u == usize::MAX || u == target {
            break;
        }
        visited[u] = true;
        for idx in graph.indptr()[u]..graph.indptr()[u + 1] {
            let v = graph.indices()[idx];
            let alt = dist[u] + graph.data()[idx];
            if alt < dist[v] {
                dist[v] = alt;
                prev[v] = u;
            }
        }
    }
    if dist[target] == f64::INFINITY {
        return (f64::INFINITY, vec![]);
    }
    let mut path = vec![target];
    let mut current = target;
    while current != source {
        current = prev[current];
        if current == usize::MAX {
            return (f64::INFINITY, vec![]);
        }
        path.push(current);
    }
    path.reverse();
    (dist[target], path)
}

// Random sparse directed graph, dedup edges, optional unit weights (tie stress).
fn random_graph(n: usize, avg_deg: usize, unit: bool, seed: u64) -> CsrMatrix {
    let mut g = Lcg(seed);
    let mut seen = std::collections::HashSet::new();
    let mut trips = Vec::new();
    for u in 0..n {
        for _ in 0..avg_deg {
            let v = g.below(n);
            if v == u || !seen.insert((u, v)) {
                continue;
            }
            let w = if unit { 1.0 } else { 0.5 + g.unit() * 9.5 };
            trips.push((u, v, w));
        }
    }
    let data: Vec<f64> = trips.iter().map(|t| t.2).collect();
    let rs: Vec<usize> = trips.iter().map(|t| t.0).collect();
    let cs: Vec<usize> = trips.iter().map(|t| t.1).collect();
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rs, cs, true).unwrap();
    coo.to_csr().unwrap()
}

fn main() {
    // ---- Isomorphism: naive vs library across many graphs (incl. tie-heavy) ----
    let mut mismatches = 0usize;
    let mut checks = 0usize;
    let mut digest = 1469598103934665603u64;
    for trial in 0..400u64 {
        let n = 5 + (trial as usize % 60);
        let unit = trial.is_multiple_of(2); // half unit-weight (ties), half random
        let g = random_graph(n, 3, unit, trial.wrapping_mul(2654435761) ^ 0x9e37);
        let mut gg = Lcg(trial ^ 0xabc);
        for _ in 0..6 {
            let s = gg.below(n);
            let t = gg.below(n);
            let (da, pa) = naive(&g, s, t);
            let (db, pb) = shortest_path(&g, true, s, t);
            checks += 1;
            if da.to_bits() != db.to_bits() || pa != pb {
                mismatches += 1;
            }
            digest = (digest ^ db.to_bits()).wrapping_mul(1099511628211);
            for &node in &pb {
                digest = (digest ^ node as u64).wrapping_mul(1099511628211);
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    println!("isomorphism: {mismatches} mismatches / {checks} (0 == byte-identical)");
    println!("path+dist digest={digest:016x}");
    println!("===GOLDEN_PAYLOAD_END===");

    // ---- Timing: large sparse graphs, naive O(V^2) vs library heap ----
    for &(n, deg) in &[(2000usize, 4usize), (8000, 4), (20000, 5)] {
        let g = random_graph(n, deg, false, 7);
        let reps = 20;
        let pairs: Vec<(usize, usize)> = {
            let mut p = Lcg(11);
            (0..reps).map(|_| (p.below(n), p.below(n))).collect()
        };
        let _ = shortest_path(&g, true, 0, n - 1);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for &(s, t) in &pairs {
            acc += naive(&g, s, t).0.min(1e18);
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for &(s, t) in &pairs {
            acc += shortest_path(&g, true, s, t).0.min(1e18);
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6} deg={deg}  naive={:>10.3?}  heap={:>10.3?}  ratio={ratio:>6.1}x  (acc={acc:.3})",
            old_t / reps as u32,
            new_t / reps as u32
        );
    }
}
