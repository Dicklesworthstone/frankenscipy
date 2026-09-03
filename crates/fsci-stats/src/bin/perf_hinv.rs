//! Same-invocation head-to-head for `NumericalInverseHermite` against
//! `scipy.stats.sampling.NumericalInverseHermite` (UNU.RAN HINV, C).
//!
//! Prints the ACCURACY achieved (node count and u-error) alongside the work, so
//! the two arms can be checked to be solving the problem to the same tolerance
//! before any cost is compared. A coarser table is cheaper; that is not a win.
//!
//! Usage: perf_hinv <n_draws> <u_resolution> [reps]
//!
//! `reps` repeats the draw loop in-process. Differencing instruction counts over
//! REPS rather than over n is what makes the per-draw cost measurable at all: the
//! Python arm carries a ~1.6e10 import whose run-to-run noise is ~1e9, which is
//! two orders above the marginal work of a single pass. Repeating the pass lifts
//! the signal above that floor without growing the array.
use fsci_stats::{Normal, NumericalInverseHermite};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let ures: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e-10);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let t0 = std::time::Instant::now();
    let h = NumericalInverseHermite::new(Normal::standard(), ures).expect("built");
    let setup = t0.elapsed();

    // Deterministic u stream, identical in both arms: no RNG difference can leak
    // into the comparison.
    let t1 = std::time::Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..reps {
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            acc += h.ppf(u);
        }
    }
    let draws = t1.elapsed();

    println!(
        "arm=fsci n={n} reps={reps} ures={ures:e} intervals={} u_error={:.6e} checksum={:.10e} setup_s={:.6} draws_s={:.6}",
        h.intervals(),
        h.u_error(),
        acc / (n * reps) as f64,
        setup.as_secs_f64(),
        draws.as_secs_f64()
    );
}
