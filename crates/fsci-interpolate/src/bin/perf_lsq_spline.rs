//! Same-process timing + bit-identity digest harness for `make_lsq_spline`.
//!
//! The library builds the least-squares normal equations A^T A by accumulating, for
//! each of m samples, the outer product of a length-n B-spline basis vector. The
//! basis has local support (~k+1 nonzeros), so the dense n^2 inner loop is O(m*n^2)
//! of which all but O(m*k^2) is `0*0`. The nz-restricted build skips those exact
//! zeros (bit-identical: `v + (+/-0.0) == v`). This harness dumps a coeff digest
//! (compare across the stashed pre-lever build to prove bit-identity) and times the
//! public path. Run: `cargo run -p fsci-interpolate --bin perf_lsq_spline`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::make_lsq_spline;

/// Clamped uniform knot vector with `ncoeff = t.len() - k - 1` coefficients.
fn clamped_knots(x0: f64, xn: f64, ncoeff: usize, k: usize) -> Vec<f64> {
    let interior = ncoeff - k - 1;
    let mut t = vec![x0; k + 1];
    for i in 1..=interior {
        t.push(x0 + (xn - x0) * i as f64 / (interior + 1) as f64);
    }
    t.extend(std::iter::repeat_n(xn, k + 1));
    t
}

fn problem(m: usize, ncoeff: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x0 = 0.0;
    let xn = m as f64;
    let x: Vec<f64> = (0..m).map(|i| x0 + (xn - x0) * i as f64 / (m - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| (0.013 * t).sin() + 0.5 * (0.4 + 0.07 * t).cos())
        .collect();
    let t = clamped_knots(x0, xn, ncoeff, k);
    (x, y, t)
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(1469598103934665603u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    let cases = [(1500usize, 200usize, 3usize), (3000, 400, 3), (5000, 600, 3)];

    // ---- bit-identity payload (compare across the stashed pre-lever build) ----
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(m, ncoeff, k) in &cases {
        let (x, y, t) = problem(m, ncoeff, k);
        let sp = make_lsq_spline(&x, &y, &t, k).expect("make_lsq_spline");
        println!(
            "m={m} ncoeff={ncoeff} k={k} ncoeffs={} digest={:016x}",
            sp.coeffs().len(),
            digest(sp.coeffs())
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    // ---- timing ----
    for &(m, ncoeff, k) in &cases {
        let (x, y, t) = problem(m, ncoeff, k);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let sp = make_lsq_spline(black_box(&x), black_box(&y), black_box(&t), k)
                .expect("make_lsq_spline");
            acc += sp.coeffs()[0];
        }
        let dt = t0.elapsed();
        println!(
            "m={m:>5} ncoeff={ncoeff:>4} k={k}  {:>9.3?}/build  (acc={acc:.6})",
            dt / reps
        );
    }
}
