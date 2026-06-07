//! Same-process timing + bit-identity digest harness for `chatterjeexi`.
//!
//! chatterjeexi's cost was dominated by `rank_max`, an O(n^2) "count values <= x"
//! double loop; it is now a sorted upper-bound (O(n log n), integer-exact). This dumps
//! the statistic/pvalue bits (compare across the stashed O(n^2) build) and times the
//! call. Run: `cargo run -p fsci-stats --bin perf_chatterjeexi`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::chatterjeexi;

fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            (0.013 * t).sin().powi(2) + 0.1 * (0.7 * t).cos()
        })
        .collect();
    (x, y)
}

fn main() {
    let sizes = [2000usize, 5000, 10000];

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let (x, y) = data(n);
        let r = chatterjeexi(&x, &y);
        println!(
            "n={n} stat={:016x} pvalue={:016x}",
            r.statistic.to_bits(),
            r.pvalue.to_bits()
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let (x, y) = data(n);
        let reps = 5;
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let r = chatterjeexi(black_box(&x), black_box(&y));
            acc += r.statistic;
        }
        let dt = t0.elapsed();
        println!("n={n:>6}  {:>10.3?}/call  (acc={acc:.6})", dt / reps);
    }
}
