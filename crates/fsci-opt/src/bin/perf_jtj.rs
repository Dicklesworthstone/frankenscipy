//! Same-process A/B + isomorphism harness for jtj_matrix (J^T J in LM curve fitting).
//!
//! `old_jtj` reproduces the original in-loop-mirror build; `new_jtj` matches the
//! library (upper triangle + single post-loop mirror). We prove J^T J is
//! byte-identical (`.to_bits()`) across random Jacobians, then time the win;
//! finally a least_squares fit confirms the library path still converges.
//! Run: `cargo run --release -p fsci-opt --bin perf_jtj`.
#![allow(clippy::needless_range_loop)]

use fsci_opt::{LeastSquaresOptions, least_squares};
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

/// Verbatim original: upper triangle + in-loop scattered mirror.
fn old_jtj(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = jac.first().map_or(0, Vec::len);
    let mut jtj = vec![vec![0.0; n]; n];
    for row in jac {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
        }
    }
    jtj
}

/// New: upper triangle only, mirror once afterwards (matches library).
fn new_jtj(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = jac.first().map_or(0, Vec::len);
    let mut jtj = vec![vec![0.0; n]; n];
    for row in jac {
        for i in 0..n {
            for j in i..n {
                jtj[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            jtj[j][i] = jtj[i][j];
        }
    }
    jtj
}

fn mat_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn main() {
    let mut r = Lcg(0x77a1_3e90_bc24_5fd8);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..3000 {
        let m = 1 + (r.next_u64() as usize % 60);
        let n = 1 + (r.next_u64() as usize % 14);
        let jac: Vec<Vec<f64>> = (0..m)
            .map(|_| (0..n).map(|_| r.unit() * 4.0 - 2.0).collect())
            .collect();
        let a = old_jtj(&jac);
        let b = new_jtj(&jac);
        total += 1;
        if !mat_eq(&a, &b) {
            mismatches += 1;
            if payload.len() < 2000 {
                payload.push_str(&format!("MISMATCH trial={trial} m={m} n={n}\n"));
            }
        }
        let digest: u64 = b.iter().flatten().fold(1469598103934665603u64, |h, v| {
            (h ^ v.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} m={m} n={n} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism (J^T J): {mismatches} mismatches / {total} (0 == byte-identical)");

    // ---- Timing: build-dominated (many residuals, moderate params) ----
    for &(m, n) in &[(200_000usize, 24usize), (100_000, 48), (60_000, 80)] {
        let jac: Vec<Vec<f64>> = (0..m).map(|_| vec![1.0; n]).collect();

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_jtj(&jac)[0][0];
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += new_jtj(&jac)[0][0];
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "m={m:>7} n={n:>3}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            old_t / 3,
            new_t / 3
        );
    }

    // ---- End-to-end: the library path (uses the new jtj) still converges ----
    // Fit a*exp(b*t): residuals r_k = a*exp(b*t_k) - y_k around a=2, b=0.5.
    let ts: Vec<f64> = (0..40).map(|k| k as f64 * 0.1).collect();
    let ys: Vec<f64> = ts.iter().map(|&t| 2.0 * (0.5 * t).exp()).collect();
    let resid = move |p: &[f64]| -> Vec<f64> {
        ts.iter()
            .zip(&ys)
            .map(|(&t, &y)| p[0] * (p[1] * t).exp() - y)
            .collect()
    };
    match least_squares(resid, &[1.0, 1.0], LeastSquaresOptions::default()) {
        Ok(res) => println!(
            "least_squares ok: params≈[{:.4}, {:.4}] (expect ~[2.0, 0.5])",
            res.x[0], res.x[1]
        ),
        Err(e) => println!("least_squares err: {e:?}"),
    }
}
