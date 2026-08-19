//! `eig` against `scipy.linalg.eig`, with both Schur arms timed separately.
//!
//! Two questions in one harness. First, the standing one: how does our general
//! eigensolver compare to the incumbent. Second, the one blocking frankenscipy-sez4r:
//! the Francis arm recovers 230 fixtures nalgebra cannot converge on, and before that
//! ships it is worth knowing whether it COSTS anything, because "more robust but slower"
//! and "more robust and free" are different trades.
//!
//! Emits per-replicate timings rather than a mean, so the caller can take a median and a
//! bootstrap interval rather than trusting an average over a possibly-drifting window.
//! An A/A arm runs the SAME configuration twice under different labels; its spread is
//! the noise floor any real difference has to clear.

use fsci_linalg::{DecompOptions, EIG_USE_FRANCIS_SCHUR, eig};
use std::sync::atomic::Ordering;
use std::time::Instant;

const MARKER: &str = "eig-vs-scipy-v1";

/// Deterministic non-symmetric matrix. Non-symmetric on purpose: a symmetric input would
/// exercise a different, specialised path and say nothing about the Schur iteration.
fn matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { n as f64 + next() } else { next() })
                .collect()
        })
        .collect()
}

fn time_one(n: usize, seed: u64, reps: usize) -> Vec<f64> {
    let a = matrix(n, seed);
    // One untimed call first: the first run pays for lazily-initialised state that the
    // steady state does not, and folding that into replicate 1 would bias the median.
    let _ = eig(&a, DecompOptions::default());
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            let r = eig(&a, DecompOptions::default());
            let secs = t.elapsed().as_secs_f64();
            // Consume the result so the call cannot be optimised away.
            std::hint::black_box(&r);
            secs
        })
        .collect()
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![32, 64, 128, 256]);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);
    println!("MARKER={MARKER} reps={reps}");

    for &n in &sizes {
        for (label, francis) in [
            ("nalgebra", false),
            ("francis", true),
            // A/A: the same configuration again. Its spread against `nalgebra` is the
            // noise floor, measured rather than assumed.
            ("nalgebra_aa", false),
        ] {
            EIG_USE_FRANCIS_SCHUR.store(francis, Ordering::Relaxed);
            let ts = time_one(n, 12345, reps);
            let joined: Vec<String> = ts.iter().map(|t| format!("{t:.9}")).collect();
            println!("T n={n} arm={label} {}", joined.join(","));
        }
    }
    EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);
}
