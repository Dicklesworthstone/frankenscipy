//! Same-process A/B + isomorphism harness for the ODR normal-equations build.
//!
//! `old_build` reproduces the original FULL J^T J double loop; `new_build`
//! matches the library (upper triangle + single mirror, commutative-multiply
//! byte-identical). We prove (normal, rhs) is byte-identical (`.to_bits()`)
//! across random Jacobians, then time the win; finally an end-to-end odr() fit
//! confirms the library path still converges.
//! Run: `cargo run --release -p fsci-odr --bin perf_odr_normal`.
#![allow(clippy::needless_range_loop)]

use fsci_odr::odr;
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

/// Verbatim original: full J^T J double loop.
fn old_build(jac: &[Vec<f64>], residuals: &[f64], damping: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.first().map_or(0, Vec::len);
    let mut normal = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for (row_idx, row) in jac.iter().enumerate() {
        for lhs in 0..n {
            rhs[lhs] -= row[lhs] * residuals[row_idx];
            for rhs_idx in 0..n {
                normal[lhs][rhs_idx] += row[lhs] * row[rhs_idx];
            }
        }
    }
    for (idx, row) in normal.iter_mut().enumerate() {
        row[idx] += damping;
    }
    (normal, rhs)
}

/// New: upper triangle + mirror, matching the library.
fn new_build(jac: &[Vec<f64>], residuals: &[f64], damping: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.first().map_or(0, Vec::len);
    let mut normal = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];
    for (row_idx, row) in jac.iter().enumerate() {
        for lhs in 0..n {
            rhs[lhs] -= row[lhs] * residuals[row_idx];
            for rhs_idx in lhs..n {
                normal[lhs][rhs_idx] += row[lhs] * row[rhs_idx];
            }
        }
    }
    for lhs in 0..n {
        for rhs_idx in (lhs + 1)..n {
            normal[rhs_idx][lhs] = normal[lhs][rhs_idx];
        }
    }
    for (idx, row) in normal.iter_mut().enumerate() {
        row[idx] += damping;
    }
    (normal, rhs)
}

fn mat_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn main() {
    let mut r = Lcg(0x4f9c_7a31_de02_b685);
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    for trial in 0..3000 {
        let rows = 1 + (r.next_u64() as usize % 80);
        let n = 1 + (r.next_u64() as usize % 14);
        let jac: Vec<Vec<f64>> = (0..rows)
            .map(|_| (0..n).map(|_| r.unit() * 4.0 - 2.0).collect())
            .collect();
        let residuals: Vec<f64> = (0..rows).map(|_| r.unit() * 6.0 - 3.0).collect();
        let damping = r.unit() * 1.5;

        let (na, ra) = old_build(&jac, &residuals, damping);
        let (nb, rb) = new_build(&jac, &residuals, damping);
        total += 1;
        if !mat_eq(&na, &nb) || ra.iter().zip(&rb).any(|(a, b)| a.to_bits() != b.to_bits()) {
            mismatches += 1;
            if payload.len() < 2000 {
                payload.push_str(&format!("MISMATCH trial={trial} rows={rows} n={n}\n"));
            }
        }
        let digest: u64 = nb.iter().flatten().chain(&rb).fold(1469598103934665603u64, |h, v| {
            (h ^ v.to_bits()).wrapping_mul(1099511628211)
        });
        payload.push_str(&format!("trial={trial} rows={rows} n={n} digest={digest:016x}\n"));
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism (normal+rhs): {mismatches} mismatches / {total} (0 == byte-identical)");

    // ---- Timing: build-dominated (many rows, moderate params) ----
    for &(rows, n) in &[(200_000usize, 24usize), (100_000, 48), (60_000, 80)] {
        let jac: Vec<Vec<f64>> = (0..rows).map(|_| vec![1.0; n]).collect();
        let residuals = vec![1.0; rows];

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_build(&jac, &residuals, 0.5).0[0][0];
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += new_build(&jac, &residuals, 0.5).0[0][0];
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "rows={rows:>7} n={n:>3}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.1})",
            old_t / 3,
            new_t / 3
        );
    }

    // ---- End-to-end: library odr() (uses new solve_lm_step) converges. ----
    // Fit y = b0 + b1*x to a noiseless line; expect b≈[1.0, 2.0].
    let xs: Vec<f64> = (0..30).map(|k| k as f64 * 0.2).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 2.0 * x).collect();
    let model = |beta: &[f64], x: &[f64]| -> Vec<f64> { x.iter().map(|&xi| beta[0] + beta[1] * xi).collect() };
    match odr(model, vec![0.0, 0.0], ys, xs) {
        Ok(out) => println!("odr ok: beta≈[{:.4}, {:.4}] (expect ~[1.0, 2.0])", out.beta[0], out.beta[1]),
        Err(e) => println!("odr err: {e:?}"),
    }
}
