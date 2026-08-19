//! frankenscipy-sez4r, PARITY half: does the Francis arm recover the spectra nalgebra
//! cannot converge on?
//!
//! `bounded_schur` was the robustness half — it turned a non-terminating loop into a
//! catchable `ConvergenceFailure`. That is strictly better than hanging and strictly
//! worse than SciPy, which converges on all 7000 fixtures. This binary measures whether
//! the from-scratch Francis double-shift QR closes that gap.
//!
//! THE HEADLINE METRIC IS A COUNT, not a time. "How many of the 7000 fixtures does each
//! arm return a spectrum for" is deterministic and identical on a loaded or idle host,
//! which is the property a convergence claim needs. Timings are reported alongside but
//! are not the claim.
//!
//! Both arms are iteration-bounded, so neither can hang and the whole grid runs in one
//! process — unlike `eig_sweep_probe`, which needed one process per case precisely
//! because the unbounded constructor could not be killed from inside Rust.
//!
//! `make_diag_dominant` is copied VERBATIM from `fsci-conformance`'s metamorphic suite
//! so the fixtures are bit-identical to the ones that exposed the hang.

use fsci_linalg::{DecompOptions, EIG_USE_FRANCIS_SCHUR, eig};
use std::sync::atomic::Ordering;
use std::time::Instant;

const MARKER: &str = "francis-vs-nalgebra-v1";

fn make_diag_dominant(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let r =
                ((seed.wrapping_mul(i as u64 + 1).wrapping_add(j as u64)) % 1000) as f64 / 1000.0;
            a[i][j] = if i == j { (n as f64) * 2.0 + r } else { r - 0.5 };
        }
    }
    a
}

/// Sorted eigenvalues, so two arms that order them differently still compare.
fn spectrum(a: &[Vec<f64>]) -> Option<Vec<(f64, f64)>> {
    match eig(a, DecompOptions::default()) {
        Ok(r) => {
            let mut v: Vec<(f64, f64)> = r
                .eigenvalues_re
                .iter()
                .zip(r.eigenvalues_im.iter())
                .map(|(&re, &im)| (re, im))
                .collect();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            Some(v)
        }
        Err(_) => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nmax: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    println!("MARKER={MARKER} nmax={nmax} seeds={seeds}");

    let mut nalgebra_ok = 0usize;
    let mut francis_ok = 0usize;
    let mut recovered: Vec<(usize, u64)> = Vec::new();
    let mut regressed: Vec<(usize, u64)> = Vec::new();
    let mut disagreed: Vec<(usize, u64, f64)> = Vec::new();
    let mut total = 0usize;

    let t_nalgebra = Instant::now();
    let mut nalgebra_spectra = Vec::new();
    EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);
    for n in 2..=nmax {
        for seed in 0..seeds {
            nalgebra_spectra.push(spectrum(&make_diag_dominant(n, seed)));
        }
    }
    let nalgebra_secs = t_nalgebra.elapsed().as_secs_f64();

    let t_francis = Instant::now();
    let mut francis_spectra = Vec::new();
    EIG_USE_FRANCIS_SCHUR.store(true, Ordering::Relaxed);
    for n in 2..=nmax {
        for seed in 0..seeds {
            francis_spectra.push(spectrum(&make_diag_dominant(n, seed)));
        }
    }
    let francis_secs = t_francis.elapsed().as_secs_f64();
    EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);

    let mut idx = 0usize;
    for n in 2..=nmax {
        for seed in 0..seeds {
            total += 1;
            let a = &nalgebra_spectra[idx];
            let b = &francis_spectra[idx];
            idx += 1;
            if a.is_some() {
                nalgebra_ok += 1;
            }
            if b.is_some() {
                francis_ok += 1;
            }
            match (a, b) {
                (None, Some(_)) => recovered.push((n, seed)),
                (Some(_), None) => regressed.push((n, seed)),
                (Some(x), Some(y)) => {
                    // Both converged: they must agree, or one of them is wrong.
                    let worst = x
                        .iter()
                        .zip(y.iter())
                        .map(|(p, q)| {
                            let d = ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
                            let s = (p.0 * p.0 + p.1 * p.1).sqrt().max(1.0);
                            d / s
                        })
                        .fold(0.0_f64, f64::max);
                    if worst > 1e-8 {
                        disagreed.push((n, seed, worst));
                    }
                }
                (None, None) => {}
            }
        }
    }

    println!("total={total}");
    println!("nalgebra_converged={nalgebra_ok} secs={nalgebra_secs:.3}");
    println!("francis_converged={francis_ok} secs={francis_secs:.3}");
    println!("recovered_by_francis={}", recovered.len());
    println!("regressed_under_francis={}", regressed.len());
    println!("disagreed_where_both_converged={}", disagreed.len());
    for (n, seed) in recovered.iter().take(40) {
        println!("RECOVERED n={n} seed={seed}");
    }
    for (n, seed) in regressed.iter().take(40) {
        println!("REGRESSED n={n} seed={seed}");
    }
    for (n, seed, w) in disagreed.iter().take(40) {
        println!("DISAGREE n={n} seed={seed} rel={w:.3e}");
    }

    // Emit BOTH spectra for every case where the two arms disagree. A disagreement is
    // not itself a verdict — it says one of them is wrong without saying which — so the
    // only useful output is both answers, side by side, for the incumbent to adjudicate.
    for (n, seed, _) in disagreed.iter() {
        EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);
        let na = spectrum(&make_diag_dominant(*n, *seed));
        EIG_USE_FRANCIS_SCHUR.store(true, Ordering::Relaxed);
        let fr = spectrum(&make_diag_dominant(*n, *seed));
        for (label, s) in [("NALGEBRA", &na), ("FRANCIS", &fr)] {
            if let Some(v) = s {
                let flat: Vec<String> =
                    v.iter().map(|(re, im)| format!("{re:.17e}:{im:.17e}")).collect();
                println!("CONTESTED {label} n={n} seed={seed} {}", flat.join(","));
            }
        }
    }

    // Emit spectra for the recovered cases so the Python arm can check them against
    // SciPy in the SAME invocation. Without this the claim would be "we converge",
    // which is not the same as "we converge to the right answer".
    EIG_USE_FRANCIS_SCHUR.store(true, Ordering::Relaxed);
    for (n, seed) in recovered.iter().take(25) {
        if let Some(s) = spectrum(&make_diag_dominant(*n, *seed)) {
            let flat: Vec<String> = s.iter().map(|(re, im)| format!("{re:.17e}:{im:.17e}")).collect();
            println!("SPECTRUM n={n} seed={seed} {}", flat.join(","));
        }
    }
    EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);
}
