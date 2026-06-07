//! Same-process A/B + isomorphism harness for directed_hausdorff.
//!
//! `naive` is the original O(N·M) double loop; the library now uses the
//! Taha-Hanbury early-break. We prove the scalar result is byte-identical
//! (`.to_bits()`) across random and structured point clouds, then time the
//! random case. Run: `cargo run --release -p fsci-spatial --bin perf_hausdorff`.
#![allow(clippy::needless_range_loop)]

use fsci_spatial::{directed_hausdorff, hausdorff_distance};
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

fn sqeuclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Verbatim original O(N·M) directed Hausdorff (squared-distance inner loop).
fn naive(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> f64 {
    let mut max_dist_sq = 0.0_f64;
    for a in xa {
        let mut min_dist_sq = f64::INFINITY;
        for b in xb {
            let d_sq = sqeuclidean(a, b);
            if d_sq < min_dist_sq {
                min_dist_sq = d_sq;
            }
        }
        if min_dist_sq > max_dist_sq {
            max_dist_sq = min_dist_sq;
        }
    }
    max_dist_sq.sqrt()
}

fn cloud(r: &mut Lcg, n: usize, dim: usize, coarse: bool, scale: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    if coarse {
                        (r.next_u64() % 5) as f64
                    } else {
                        r.unit() * scale
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut r = Lcg(0xabcd_1234_5678_9f01);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..500 {
        let n = 1 + (r.next_u64() as usize % 60);
        let m = 1 + (r.next_u64() as usize % 60);
        let dim = 1 + (r.next_u64() as usize % 4);
        let coarse = trial % 3 == 0; // heavy ties / coincident points
        let xa = cloud(&mut r, n, dim, coarse, 10.0);
        let xb = cloud(&mut r, m, dim, coarse, 10.0);

        // directed (both directions) + symmetric
        let cases = [
            ("ab", directed_hausdorff(&xa, &xb).unwrap(), naive(&xa, &xb)),
            ("ba", directed_hausdorff(&xb, &xa).unwrap(), naive(&xb, &xa)),
            (
                "sym",
                hausdorff_distance(&xa, &xb).unwrap(),
                naive(&xa, &xb).max(naive(&xb, &xa)),
            ),
        ];
        for (name, got, want) in cases {
            total += 1;
            if got.to_bits() != want.to_bits() {
                mismatches += 1;
                if payload.len() < 3000 {
                    payload.push_str(&format!(
                        "MISMATCH {name} trial={trial} n={n} m={m} got={got:.17e} want={want:.17e}\n"
                    ));
                }
            }
        }
        let d = directed_hausdorff(&xa, &xb).unwrap();
        payload.push_str(&format!(
            "trial={trial} n={n} m={m} bits={:016x}\n",
            d.to_bits()
        ));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} results (0 == byte-identical)");

    // ---- Timing: random clouds, growing N=M ----
    for &n in &[1000usize, 4000, 12000] {
        let xa = cloud(&mut r, n, 3, false, 100.0);
        let xb = cloud(&mut r, n, 3, false, 100.0);

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += naive(&xa, &xb);
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += directed_hausdorff(&xa, &xb).unwrap();
        }
        let fast_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / fast_t.as_secs_f64();
        println!(
            "n=m={n:>6}  naive={:>10.3?}  earlybreak={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            naive_t / 3,
            fast_t / 3
        );
    }
}
