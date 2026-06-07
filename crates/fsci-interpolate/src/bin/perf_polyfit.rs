//! Same-process A/B + isomorphism harness for polyfit's A^T A build.
//!
//! `old_build` reproduces the original (per-sample xpow alloc + in-loop mirror);
//! `new_build` matches the library (upper triangle + reused buffer + single
//! mirror). We prove A^T A / A^T b is byte-identical (`.to_bits()`) across random
//! problems, then time the win; finally call polyfit end-to-end.
//! Run: `cargo run --release -p fsci-interpolate --bin perf_polyfit`.
#![allow(clippy::needless_range_loop)]

use fsci_interpolate::polyfit;
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

fn old_build(x: &[f64], y: &[f64], ncols: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let mut ata = vec![vec![0.0; ncols]; ncols];
    let mut atb = vec![0.0; ncols];
    for i in 0..n {
        let mut xpow = vec![1.0; ncols];
        for j in 1..ncols {
            xpow[j] = xpow[j - 1] * x[i];
        }
        for j in 0..ncols {
            atb[j] += xpow[j] * y[i];
            for k in j..ncols {
                ata[j][k] += xpow[j] * xpow[k];
                if k != j {
                    ata[k][j] = ata[j][k];
                }
            }
        }
    }
    (ata, atb)
}

fn new_build(x: &[f64], y: &[f64], ncols: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let mut ata = vec![vec![0.0; ncols]; ncols];
    let mut atb = vec![0.0; ncols];
    let mut xpow = vec![1.0; ncols];
    for i in 0..n {
        for j in 1..ncols {
            xpow[j] = xpow[j - 1] * x[i];
        }
        for j in 0..ncols {
            atb[j] += xpow[j] * y[i];
            for k in j..ncols {
                ata[j][k] += xpow[j] * xpow[k];
            }
        }
    }
    for j in 0..ncols {
        for k in (j + 1)..ncols {
            ata[k][j] = ata[j][k];
        }
    }
    (ata, atb)
}

fn mat_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn main() {
    let mut r = Lcg(0x12c7_55ab_e9d0_3f44);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..3000 {
        let deg = 1 + (r.next_u64() as usize % 14);
        let n = deg + 1 + (r.next_u64() as usize % 60);
        let ncols = deg + 1;
        // keep |x| small so high powers stay finite
        let x: Vec<f64> = (0..n).map(|_| r.unit() * 2.0 - 1.0).collect();
        let y: Vec<f64> = (0..n).map(|_| r.unit() * 4.0 - 2.0).collect();

        let (xa, ya) = old_build(&x, &y, ncols);
        let (xb, yb) = new_build(&x, &y, ncols);
        total += 1;
        if !mat_eq(&xa, &xb) || ya.iter().zip(&yb).any(|(a, b)| a.to_bits() != b.to_bits()) {
            mismatches += 1;
            if payload.len() < 2000 {
                payload.push_str(&format!("MISMATCH trial={trial} n={n} deg={deg}\n"));
            }
        }
        if let Ok(coeffs) = polyfit(&x, &y, deg) {
            let digest: u64 = coeffs.iter().fold(1469598103934665603u64, |h, v| {
                (h ^ v.to_bits()).wrapping_mul(1099511628211)
            });
            payload.push_str(&format!(
                "trial={trial} n={n} deg={deg} digest={digest:016x}\n"
            ));
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism (A^T A build): {mismatches} mismatches / {total} (0 == byte-identical)");

    // ---- Timing: build-dominated (many samples, high degree) ----
    for &(n, deg) in &[(200_000usize, 23usize), (100_000, 47), (60_000, 79)] {
        let ncols = deg + 1;
        let x: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let y: Vec<f64> = vec![1.0; n];

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_build(&x, &y, ncols).0[0][0];
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += new_build(&x, &y, ncols).0[0][0];
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "n={n:>7} deg={deg:>3}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            old_t / 3,
            new_t / 3
        );
    }
}
