//! Same-process A/B + isomorphism harness for qmc_quad (van der Corput dispatch).
//!
//! `old_qmc_quad` reproduces the original runtime-base van der Corput; the
//! library now dispatches the bundled primes to const-generic specialisations
//! (division strength-reduction). We prove qmc_quad's (integral, std_error) is
//! byte-identical (`.to_bits()`) for several integrands/dims, then time the win
//! on cheap integrands (where point generation dominates).
//! Run: `cargo run --release -p fsci-integrate --bin perf_qmcquad`.

use fsci_integrate::qmc_quad;
use std::time::Instant;

const QMC_PRIMES: [usize; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// Verbatim copy of the original runtime-base van der Corput.
fn old_vdc(mut idx: usize, base: usize) -> f64 {
    let mut q = 0.0;
    let mut bk = 1.0 / base as f64;
    while idx > 0 {
        q += (idx % base) as f64 * bk;
        idx /= base;
        bk /= base as f64;
    }
    q
}

/// Verbatim copy of the original qmc_quad (runtime van der Corput).
fn old_qmc_quad<F: Fn(&[f64]) -> f64>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    n_estimates: usize,
    n_points: usize,
) -> (f64, f64) {
    let d = lb.len();
    let mut volume = 1.0;
    for j in 0..d {
        volume *= ub[j] - lb[j];
    }
    let mut estimates = Vec::with_capacity(n_estimates);
    let mut point = vec![0.0; d];
    for est in 0..n_estimates {
        let start = 1 + est * n_points;
        let mut sum = 0.0;
        for i in 0..n_points {
            let idx = start + i;
            for j in 0..d {
                let u = old_vdc(idx, QMC_PRIMES[j]);
                point[j] = lb[j] + u * (ub[j] - lb[j]);
            }
            sum += f(&point);
        }
        estimates.push(sum / n_points as f64 * volume);
    }
    let n_est_f = n_estimates as f64;
    let mean = estimates.iter().sum::<f64>() / n_est_f;
    let se = if n_estimates > 1 {
        let var = estimates.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n_est_f - 1.0);
        (var / n_est_f).sqrt()
    } else {
        0.0
    };
    (mean, se)
}

fn main() {
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let mut payload = String::new();

    // Cheap integrands across several dims; the result must be byte-identical.
    let cases: &[(&str, fn(&[f64]) -> f64)] = &[
        ("sumsq", |x| x.iter().map(|&v| v * v).sum()),
        ("prod1p", |x| x.iter().map(|&v| 1.0 + v).product()),
        ("first", |x| x[0]),
    ];
    for (name, f) in cases {
        for d in 1..=8usize {
            let lb = vec![0.0; d];
            let ub: Vec<f64> = (0..d).map(|j| 1.0 + 0.1 * j as f64).collect();
            for &(ne, np) in &[(4usize, 256usize), (8, 1000), (16, 333)] {
                let got = qmc_quad(f, &lb, &ub, ne, np).unwrap();
                let (wi, we) = old_qmc_quad(f, &lb, &ub, ne, np);
                total += 1;
                if got.integral.to_bits() != wi.to_bits()
                    || got.standard_error.to_bits() != we.to_bits()
                {
                    mismatches += 1;
                    if payload.len() < 2000 {
                        payload.push_str(&format!("MISMATCH {name} d={d} ne={ne} np={np}\n"));
                    }
                }
                payload.push_str(&format!(
                    "{name} d={d} ne={ne} np={np} ibits={:016x} sebits={:016x}\n",
                    got.integral.to_bits(),
                    got.standard_error.to_bits()
                ));
            }
        }
    }
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    print!("{payload}");
    println!("===GOLDEN_PAYLOAD_END===");
    println!("isomorphism: {mismatches} mismatches / {total} qmc_quad runs (0 == byte-identical)");

    // ---- Timing: cheap integrand (point generation dominates) ----
    let cheap = |x: &[f64]| -> f64 { x.iter().map(|&v| v * v).sum() };
    for &(d, np) in &[(8usize, 20_000usize), (4, 40_000), (2, 80_000)] {
        let lb = vec![0.0; d];
        let ub = vec![1.0; d];
        let ne = 16;

        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..3 {
            acc += old_qmc_quad(cheap, &lb, &ub, ne, np).0;
        }
        let old_t = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += qmc_quad(cheap, &lb, &ub, ne, np).unwrap().integral;
        }
        let new_t = t1.elapsed();

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "d={d:>2} np={np:>6} ne={ne}  old={:>10.3?}  new={:>10.3?}  ratio={ratio:>7.1}x  (acc={acc:.3})",
            old_t / 3,
            new_t / 3
        );
    }
}
