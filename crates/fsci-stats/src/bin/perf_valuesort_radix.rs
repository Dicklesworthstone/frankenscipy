// Same-binary A/B for the radix VALUE sort in wasserstein/energy/ks_2samp.
use fsci_stats::{RANKDATA_RADIX_DISABLE, energy_distance, ks_2samp, wasserstein_distance};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let mut s = 0x1357_9bdf_2468_ace0u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &n in &[50_000usize, 200_000, 1_000_000] {
        let u: Vec<f64> = (0..n).map(|_| r() * 1e3 - 500.0).collect();
        let v: Vec<f64> = (0..n).map(|_| r() * 1e3 - 480.0).collect();
        let reps = if n <= 200_000 { 8 } else { 3 };
        // Named so `clippy::type_complexity` is satisfied by a definition rather
        // than an allow (frankenscipy-023vy).
        type TwoSampleCase<'a> = (&'a str, &'a dyn Fn(&[f64], &[f64]) -> f64);
        let cases: [TwoSampleCase; 3] = [
            ("wasserstein", &|a, b| wasserstein_distance(a, b)),
            ("energy", &|a, b| energy_distance(a, b)),
            ("ks_2samp", &|a, b| ks_2samp(a, b).statistic),
        ];
        for (name, f) in cases {
            let bench = |disable: bool| {
                RANKDATA_RADIX_DISABLE.store(disable, Ordering::Relaxed);
                let _ = black_box(f(&u, &v));
                let t = Instant::now();
                for _ in 0..reps {
                    let _ = black_box(f(black_box(&u), black_box(&v)));
                }
                t.elapsed().as_secs_f64() / reps as f64 * 1000.0
            };
            let base = bench(true).min(bench(true));
            let rdx = bench(false).min(bench(false));
            println!(
                "n={n:8} {name:12}: sort {base:8.2}ms -> radix {rdx:8.2}ms = {:.2}x",
                base / rdx
            );
        }
    }
    RANKDATA_RADIX_DISABLE.store(false, Ordering::Relaxed);
}
