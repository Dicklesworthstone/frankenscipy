//! Same-process serial/parallel benchmark for randomized QMC quadrature.
//!
//! Both paths use the same independently scrambled Halton estimates and seed.
//! Worker count must change scheduling only, preserving integral/error bits.
//! Run: `cargo run --release -p fsci-integrate --bin perf_qmcquad`.

use fsci_integrate::{QmcQuadOptions, qmc_quad_with_options};
use std::hint::black_box;
use std::time::Instant;

type Integrand = fn(&[f64]) -> f64;

fn main() {
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let cases: &[(&str, Integrand)] = &[
        ("sumsq", |x| x.iter().map(|&v| v * v).sum()),
        ("prod1p", |x| x.iter().map(|&v| 1.0 + v).product()),
        ("first", |x| x[0]),
    ];
    for &(name, f) in cases {
        for dimensions in 1..=8 {
            let lower = vec![0.0; dimensions];
            let upper: Vec<f64> = (0..dimensions).map(|j| 1.0 + 0.1 * j as f64).collect();
            for &(n_estimates, n_points) in &[(4, 256), (8, 1000), (16, 333)] {
                let options = QmcQuadOptions {
                    n_estimates,
                    n_points,
                    seed: 37,
                    workers: Some(1),
                };
                let serial = qmc_quad_with_options(f, &lower, &upper, options).unwrap();
                let parallel = qmc_quad_with_options(
                    f,
                    &lower,
                    &upper,
                    QmcQuadOptions {
                        workers: Some(4),
                        ..options
                    },
                )
                .unwrap();
                let identical = serial.integral.to_bits() == parallel.integral.to_bits()
                    && serial.standard_error.to_bits() == parallel.standard_error.to_bits();
                total += 1;
                mismatches += usize::from(!identical);
                println!(
                    "{name} d={dimensions} estimates={n_estimates} points={n_points} bit_identical={identical} integral={:.17e} standard_error={:.17e}",
                    parallel.integral, parallel.standard_error
                );
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches} of {total} worker comparisons differed"
    );
    println!("worker reproducibility: {total} comparisons passed");

    let cheap: Integrand = |x| x.iter().map(|value| value * value).sum();
    for &(dimensions, n_points) in &[(8, 20_000), (4, 40_000), (2, 80_000)] {
        let lower = vec![0.0; dimensions];
        let upper = vec![1.0; dimensions];
        let options = QmcQuadOptions {
            n_estimates: 16,
            n_points,
            seed: 37,
            workers: Some(1),
        };
        let serial_start = Instant::now();
        for _ in 0..3 {
            black_box(qmc_quad_with_options(cheap, &lower, &upper, options).unwrap());
        }
        let serial = serial_start.elapsed();
        let parallel_start = Instant::now();
        for _ in 0..3 {
            black_box(
                qmc_quad_with_options(
                    cheap,
                    &lower,
                    &upper,
                    QmcQuadOptions {
                        workers: None,
                        ..options
                    },
                )
                .unwrap(),
            );
        }
        let parallel = parallel_start.elapsed();
        println!(
            "d={dimensions:>2} points={n_points:>6} estimates=16 serial={:>10.3?} parallel={:>10.3?} speedup={:.2}x",
            serial / 3,
            parallel / 3,
            serial.as_secs_f64() / parallel.as_secs_f64(),
        );
    }
}
