//! Same-process A/B + isomorphism harness for ridge_regression.
//!
//! `naive_build` reproduces the original FULL-matrix XtX build (per-sample alloc,
//! both triangles computed); the library now accumulates only the upper triangle
//! (mirror once, commutative-multiply byte-identical) + reused buffer. We prove
//! the XtX/Xty build is byte-identical (`.to_bits()`) across random problems, then
//! time the win. Run: `cargo run --release -p fsci-stats --bin perf_ridge`.
#![allow(clippy::needless_range_loop)]

use fsci_stats::ridge_regression;
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

/// Verbatim original ridge XtX/Xty build: per-sample alloc, full double loop.
fn naive_build(x: &[Vec<f64>], y: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let p = x[0].len();
    let p1 = p + 1;
    let mut xtx = vec![vec![0.0; p1]; p1];
    let mut xty = vec![0.0; p1];
    for i in 0..n {
        let mut row = vec![1.0];
        row.extend_from_slice(&x[i]);
        for j in 0..p1 {
            xty[j] += row[j] * y[i];
            for k in 0..p1 {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }
    (xtx, xty)
}

/// New build: upper triangle + mirror, reused buffer (matches the library).
fn new_build(x: &[Vec<f64>], y: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let p = x[0].len();
    let p1 = p + 1;
    let mut xtx = vec![vec![0.0; p1]; p1];
    let mut xty = vec![0.0; p1];
    let mut row = vec![1.0; p1];
    for i in 0..n {
        row[1..].copy_from_slice(&x[i][..p]);
        for j in 0..p1 {
            xty[j] += row[j] * y[i];
            for k in j..p1 {
                xtx[j][k] += row[j] * row[k];
            }
        }
    }
    for j in 0..p1 {
        for k in (j + 1)..p1 {
            xtx[k][j] = xtx[j][k];
        }
    }
    (xtx, xty)
}

fn mat_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn main() {
    let mut r = Lcg(0x3c8e_d145_9a07_2fb1);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..2000 {
        let n = 2 + (r.next_u64() as usize % 80);
        let p = 1 + (r.next_u64() as usize % n.clamp(1, 12));
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..p).map(|_| r.unit() * 10.0 - 5.0).collect())
            .collect();
        let y: Vec<f64> = (0..n).map(|_| r.unit() * 20.0 - 10.0).collect();
        let alpha = r.unit() * 2.0;

        let (xa, ya) = naive_build(&x, &y);
        let (xb, yb) = new_build(&x, &y);
        total += 1;
        if !mat_eq(&xa, &xb) || ya.iter().zip(&yb).any(|(a, b)| a.to_bits() != b.to_bits()) {
            mismatches += 1;
            if payload.len() < 2000 {
                payload.push_str(&format!("MISMATCH build trial={trial} n={n} p={p}\n"));
            }
        }
        let beta = ridge_regression(&x, &y, alpha);
        let digest: u64 = beta.iter().fold(1469598103934665603u64, |h, v| {
            (h ^ v.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} n={n} p={p} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!(
        "isomorphism (XtX/Xty build): {mismatches} mismatches / {total} (0 == byte-identical)"
    );

    // ---- Timing: build-dominated (many samples, moderate p) ----
    for &(n, p) in &[(200_000usize, 24usize), (100_000, 48), (50_000, 80)] {
        let x: Vec<Vec<f64>> = (0..n).map(|_| (0..p).map(|_| 1.0).collect()).collect();
        let y: Vec<f64> = vec![1.0; n];

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += naive_build(&x, &y).0[0][0];
        }
        let naive_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += new_build(&x, &y).0[0][0];
        }
        let new_t = t1.elapsed();

        let ratio = naive_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>7} p={p:>3}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            naive_t / 3,
            new_t / 3
        );
    }
}
