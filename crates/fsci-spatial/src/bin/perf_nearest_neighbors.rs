//! Same-process A/B + isomorphism harness for `nearest_neighbors`.
//!
//! `old_nearest_neighbors` is a verbatim copy of the original O(n^2) all-pairs
//! brute force. The library now routes through a KD-tree (O(n log n) average)
//! with a lowest-index tie-break, which is byte-identical: matched pairs give
//! identical distance bits and ties resolve to the same (smallest) index. We
//! assert 0 mismatches across shapes/dims/tie densities and time the win.
//! Run: `cargo run --release -p fsci-spatial --bin perf_nearest_neighbors`.

use fsci_spatial::{euclidean, nearest_neighbors};
use std::time::Instant;

/// Verbatim copy of the original O(n^2) nearest_neighbors.
fn old_nearest_neighbors(data: &[Vec<f64>]) -> (Vec<Option<usize>>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean(&data[i], &data[j]);
            if d < min_dist {
                min_dist = d;
                min_idx = Some(j);
            }
        }
        indices.push(min_idx);
        distances.push(min_dist);
    }
    (indices, distances)
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

/// `grid` controls tie density: coordinates are snapped to a `grid`-step lattice
/// so many points coincide / become equidistant (grid=0 => continuous).
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
    for &n in &[1usize, 2, 5, 17, 64, 256, 1000] {
        for &dim in &[1usize, 2, 3, 5] {
            for &grid in &[0u64, 3, 8] {
                for seed in 0..4u64 {
                    let data = make_case(n, dim, grid, seed * 1009 + 1);
                    let (gi, gd) = nearest_neighbors(&data);
                    let (wi, wd) = old_nearest_neighbors(&data);
                    total += 1;
                    let idx_ok = gi == wi;
                    let dist_ok = gd.len() == wd.len()
                        && gd.iter().zip(&wd).all(|(a, b)| a.to_bits() == b.to_bits());
                    if !idx_ok || !dist_ok {
                        mismatches += 1;
                        if payload.len() < 1500 {
                            payload.push_str(&format!(
                                "MISMATCH n={n} dim={dim} grid={grid} seed={seed} idx_ok={idx_ok} dist_ok={dist_ok}\n"
                            ));
                        }
                    }
                    if payload.len() < 1500 {
                        let chk: u64 = gi.iter().map(|o| o.map_or(0u64, |v| v as u64 + 1)).sum();
                        payload.push_str(&format!(
                            "n={n} dim={dim} grid={grid} seed={seed} idxsum={chk}\n"
                        ));
                    }
                }
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism: {mismatches} mismatches / {total} cases (0 == byte-identical index + distance bits)"
    );

    // ---- Timing: large n, low-dim continuous (KD-tree's sweet spot) ----
    for &(n, dim) in &[(4000usize, 2usize), (8000, 3), (16000, 2)] {
        let data = make_case(n, dim, 0, 7);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_nearest_neighbors(&data)
                .1
                .iter()
                .filter(|d| d.is_finite())
                .sum::<f64>();
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += nearest_neighbors(&data)
                .1
                .iter()
                .filter(|d| d.is_finite())
                .sum::<f64>();
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>6} dim={dim}  old={:>11.3?}  new={:>11.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            old_t / 3,
            new_t / 3
        );
    }
}
