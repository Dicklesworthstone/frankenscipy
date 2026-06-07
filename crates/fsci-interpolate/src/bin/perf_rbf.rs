//! Same-process A/B for the parallel RbfInterpolator::eval_many: proves the
//! multithreaded path is bit-identical to the sequential per-query map and times both.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::{RbfInterpolator, RbfKernel};

fn time_it(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    for &(ncenters, nqueries) in &[(500usize, 4000usize), (2000, 6000)] {
        let points: Vec<Vec<f64>> = (0..ncenters)
            .map(|i| {
                vec![
                    (i as f64 * 0.07).sin(),
                    (i as f64 * 0.11).cos(),
                    i as f64 * 0.001,
                ]
            })
            .collect();
        let values: Vec<f64> = (0..ncenters)
            .map(|i| (i as f64 * 0.05).sin() * 2.0)
            .collect();
        let rbf =
            RbfInterpolator::new(&points, &values, RbfKernel::Multiquadric, 1.3).expect("rbf");
        let queries: Vec<Vec<f64>> = (0..nqueries)
            .map(|i| {
                vec![
                    i as f64 * 0.001 - 3.0,
                    i as f64 * 0.002 - 2.0,
                    i as f64 * 0.0005,
                ]
            })
            .collect();

        let got = rbf.eval_many(&queries);
        let want: Vec<f64> = queries.iter().map(|q| rbf.eval(q)).collect();
        let exact = got.len() == want.len()
            && got
                .iter()
                .zip(&want)
                .all(|(&g, &w)| g.to_bits() == w.to_bits());

        let iters = (200_000_000 / (ncenters * nqueries + 1)).clamp(2, 50);
        let par = time_it(iters, || rbf.eval_many(&queries));
        let seq = time_it(iters, || queries.iter().map(|q| rbf.eval(q)).collect());
        println!(
            "rbf centers={ncenters} queries={nqueries}: seq={seq:>8.3}ms  par={par:>8.3}ms  speedup={:>6.2}x  bit_identical={exact}",
            seq / par
        );
    }
}
