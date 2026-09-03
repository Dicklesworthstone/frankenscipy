//! Where does the eigh constant sit? — `frankenscipy-ll0kk`
//!
//! The `EIGH_NATIVE_STAGE_TIMING` doc block states the live question exactly:
//! held at one implementation the vs-SciPy ratio is FLAT with size (1.82x at
//! n=512, 2.23x at 768, 1.81x at 1024 on hz2), so the deficit is a CONSTANT
//! FACTOR rather than an asymptotic kernel-quality wall, and the open question is
//! where that constant sits. Two candidates are already eliminated on that bead:
//! the blocked dsytrd reduction measured SLOWER (0.5-0.81x), and the
//! inverse-iteration clustering gate was refuted (minimum relative gap 2e-4 at
//! n=512 against a 1e-6 threshold, so the fast path always fires).
//!
//! The counters answer it directly instead of by elimination. This binary is the
//! driver they never had: it enables them, runs the native route at several sizes,
//! and prints each stage's SHARE of the total.
//!
//! ## Reading the output
//!
//! The three stages are the classical symmetric-eigen pipeline:
//!
//!   0  reduce  Householder tridiagonalisation           O(n^3), ~4/3 n^3 flops
//!   1  solve   tridiagonal eigensolve + inverse iteration
//!   2  back    back-transform of eigenvectors            O(n^3)
//!
//! A constant factor against SciPy should show up as one stage dominating at EVERY
//! size. A stage whose share GROWS with n is an asymptotic problem instead, and
//! would contradict the flat-ratio finding rather than explain it. Shares are what
//! matter here, not absolute times: this prints no comparison against SciPy and
//! makes no vs-incumbent claim, so it does not need a quiet host to be meaningful.
//!
//! ## What this is not
//!
//! Not a benchmark and not a certification. There is no incumbent arm, no A/A
//! null, and no margin. It is instrumentation readout, and the timing it reports
//! is wall time accumulated inside the three stages of one implementation.
//!
//! Usage:  perf_eigh_stages [sizes]     default 512,768,1024

use fsci_linalg::{
    DecompOptions, EIGH_NATIVE_STAGE_NANOS, EIGH_NATIVE_STAGE_TIMING,
    PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE, eigh,
};
use std::sync::atomic::Ordering;

/// Deterministic symmetric fixture. Symmetric by construction rather than by
/// symmetrising a random matrix, so the input is identical for a given n on every
/// host and the shares are comparable across runs.
fn symmetric_fixture(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                    let k = (lo as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(hi as u64);
                    let v = ((k % 100_003) as f64) / 100_003.0;
                    if i == j { v + n as f64 } else { v }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "512,768,1024".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(!sizes.is_empty(), "no sizes parsed");

    // Force the native route at every size, otherwise small n takes the nalgebra
    // path and the counters stay at zero -- which would print a clean, empty,
    // entirely misleading table.
    PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE.store(1, Ordering::Relaxed);
    EIGH_NATIVE_STAGE_TIMING.store(true, Ordering::Relaxed);

    println!("stage shares of native eigh, by size (frankenscipy-ll0kk)");
    println!(
        "{:>6} {:>12} {:>10} {:>10} {:>10}  note",
        "n", "total_ms", "reduce%", "solve%", "back%"
    );

    for &n in &sizes {
        let a = symmetric_fixture(n);
        for c in &EIGH_NATIVE_STAGE_NANOS {
            c.store(0, Ordering::Relaxed);
        }

        let started = std::time::Instant::now();
        let result = eigh(&a, DecompOptions::default());
        let wall = started.elapsed().as_secs_f64() * 1e3;
        assert!(result.is_ok(), "eigh failed at n={n}");

        let stage: Vec<u64> = EIGH_NATIVE_STAGE_NANOS
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let total: u64 = stage.iter().sum();

        // MUST-HIT: if the counters are all zero the native route never ran and
        // every share below would be 0/0. Refuse rather than print a clean table.
        assert!(
            total > 0,
            "all stage counters are zero at n={n}: the native route did not run, so \
             this table would report nothing while looking complete"
        );
        // The stages should account for most of the call. A large unaccounted
        // remainder means the interesting work is happening somewhere the counters
        // do not cover, which is worth knowing before drawing any conclusion.
        let accounted = (total as f64 / 1e6) / wall * 100.0;
        let note = if accounted < 70.0 {
            format!("only {accounted:.0}% of wall accounted -- counters miss a stage")
        } else {
            format!("{accounted:.0}% of wall accounted")
        };

        let pct = |s: u64| 100.0 * s as f64 / total as f64;
        println!(
            "{n:>6} {wall:>12.1} {:>9.1}% {:>9.1}% {:>9.1}%  {note}",
            pct(stage[0]),
            pct(stage[1]),
            pct(stage[2]),
        );
    }

    EIGH_NATIVE_STAGE_TIMING.store(false, Ordering::Relaxed);
    PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE.store(0, Ordering::Relaxed);
}
