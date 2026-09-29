//! Timing + checksum harness for N-D `HalfspaceIntersection::new`.
//!
//! The intersection is the dual convex hull of `m` dual points (the exact-predicate Quickhull),
//! replacing an enumeration of all C(m, ndim) halfspace subsets and a C(m, ndim+1) boundedness
//! scan. Dumps a checksum for fixed small inputs (compare across builds) and times large-m
//! builds.
//! Run: `cargo run --profile release-perf -p fsci-spatial --bin perf_halfspace_nd`.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::HalfspaceIntersection;

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

// m random outward halfspaces a·x <= 1 (row [a_0..a_{d-1}, -1]) with |a|=1, so the
// origin is strictly interior and the region is a bounded polytope (≈ unit ball).
fn halfspaces(m: usize, ndim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let a: Vec<f64> = (0..ndim).map(|_| g.unit() * 2.0 - 1.0).collect();
        let nrm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm < 1e-6 {
            continue;
        }
        let mut row: Vec<f64> = a.iter().map(|v| v / nrm).collect();
        row.push(-1.0);
        out.push(row);
    }
    out
}

fn checksum(hs: &HalfspaceIntersection) -> u64 {
    let mut h = 1469598103934665603u64;
    for v in &hs.intersections {
        for &c in v {
            h = (h ^ c.to_bits()).wrapping_mul(1099511628211);
        }
        h = h.wrapping_mul(31);
    }
    for f in &hs.dual_facets {
        for &idx in f {
            h = (h ^ idx as u64).wrapping_mul(1099511628211);
        }
        h = h.wrapping_mul(31);
    }
    for &dv in &hs.dual_vertices {
        h = (h ^ dv as u64).wrapping_mul(1099511628211);
    }
    h
}

fn main() {
    let interior_3 = vec![0.0, 0.0, 0.0];
    let interior_4 = vec![0.0, 0.0, 0.0, 0.0];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(m, ndim, seed) in &[(10usize, 3usize, 1u64), (20, 3, 2), (16, 4, 3)] {
        let hs = halfspaces(m, ndim, seed);
        let interior = if ndim == 3 { &interior_3 } else { &interior_4 };
        match HalfspaceIntersection::new(&hs, interior) {
            Ok(h) => println!(
                "m={m} ndim={ndim} seed={seed} nverts={} chk={:016x}",
                h.intersections.len(),
                checksum(&h)
            ),
            Err(e) => println!("m={m} ndim={ndim} seed={seed} err={e:?}"),
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(m, ndim) in &[(40usize, 3usize), (120, 3), (60, 4)] {
        let hs = halfspaces(m, ndim, 7);
        let interior = if ndim == 3 { &interior_3 } else { &interior_4 };
        let reps = 3;
        let _ = HalfspaceIntersection::new(&hs, interior);
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps {
            let h = HalfspaceIntersection::new(black_box(&hs), interior).unwrap();
            acc ^= h.intersections.len() as u64;
        }
        println!(
            "m={m:>4} ndim={ndim}  {:>10.3?}/build  (nverts_acc={acc})",
            t0.elapsed() / reps
        );
    }
}
