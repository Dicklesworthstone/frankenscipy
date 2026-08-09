use fsci_stats::{WEIBULL_FIT_LN_REUSE_DISABLE, Weibull, fit};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let mut s = 42u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &n in &[10_000usize, 100_000, 500_000] {
        // Weibull-ish positive data (inverse-CDF sampling, shape 1.5 scale 2.0)
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let u = r().clamp(1e-12, 1.0 - 1e-12);
                2.0 * (-(1.0 - u).ln()).powf(1.0 / 1.5)
            })
            .collect();
        let run = |disable: bool| {
            WEIBULL_FIT_LN_REUSE_DISABLE.store(disable, Ordering::Relaxed);
            fit::<Weibull>(&data)
        };
        let re = run(false);
        let og = run(true);
        let dc = (re.c - og.c).abs();
        let ds = (re.scale - og.scale).abs();
        let bench = |disable: bool| {
            WEIBULL_FIT_LN_REUSE_DISABLE.store(disable, Ordering::Relaxed);
            let _ = black_box(fit::<Weibull>(&data));
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(fit::<Weibull>(black_box(&data)));
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let (mut ret, mut ogt) = (f64::MAX, f64::MAX);
        for _ in 0..4 {
            ogt = ogt.min(bench(true));
            ret = ret.min(bench(false));
        }
        println!(
            "n={n}: orig(powf) {ogt:.2}ms  reuse(ln) {ret:.2}ms  speedup {:.2}x  dc={dc:.1e} dscale={ds:.1e}",
            ogt / ret
        );
    }
}
