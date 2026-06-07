//! Same-process A/B for the parallel GaussianKde::evaluate_many: proves the
//! multithreaded path is bit-identical to the sequential per-point map and times both.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::GaussianKde;

fn time_it(iters: usize, mut f: impl FnMut() -> Vec<f64>) -> f64 {
    black_box(f());
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    for &(ndata, npoints) in &[(1000usize, 4000usize), (5000, 8000)] {
        let data: Vec<f64> = (0..ndata)
            .map(|i| (i as f64 * 0.031).sin() * 3.0 + 1.0)
            .collect();
        let kde = GaussianKde::new(&data);
        let points: Vec<f64> = (0..npoints).map(|i| i as f64 * 0.002 - 8.0).collect();

        let got = kde.evaluate_many(&points);
        let want: Vec<f64> = points.iter().map(|&x| kde.evaluate(x)).collect();
        let exact = got.len() == want.len()
            && got
                .iter()
                .zip(&want)
                .all(|(&g, &w)| g.to_bits() == w.to_bits());

        let iters = (400_000_000 / (ndata * npoints + 1)).clamp(2, 50);
        let par = time_it(iters, || kde.evaluate_many(&points));
        let seq = time_it(iters, || points.iter().map(|&x| kde.evaluate(x)).collect());
        println!(
            "kde ndata={ndata} npoints={npoints}: seq={seq:>8.3}ms  par={par:>8.3}ms  speedup={:>6.2}x  bit_identical={exact}",
            seq / par
        );
    }
}
