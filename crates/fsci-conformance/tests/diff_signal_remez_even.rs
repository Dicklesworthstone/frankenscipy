#![forbid(unsafe_code)]
//! Live SciPy diff for `fsci_signal::remez` over seeded band specifications
//! (frankenscipy-szq1n.13). The even-numtaps path used to hand back a frequency-sampling
//! least-squares design when its Parks-McClellan exchange failed — a different, non-equiripple
//! filter returned as the minimax one; and the odd- and even-length exchanges, written by hand,
//! picked extremal points by their own rules and on low-order multiband specs returned designs
//! up to 38x worse than the minimax optimum. fsci now runs SciPy's own exchange (a port of
//! `_sigtoolsmodule.cc`). This compares 240 seeded well-posed specs (`specs`) and 240 arbitrary
//! ones, ill-posed included (`arbitrary_specs`), odd and even numtaps 8..=60, 2-4 bands, with
//! `scipy.signal.remez(numtaps, bands, desired, weight=w, fs=1)`: either both design a filter and
//! the taps agree to REMEZ_TAP_TOL relative to the largest tap, or both refuse.
//! The compared count is asserted, and so is that some specs are refused by both.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_signal::remez;
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// max |fsci tap - SciPy tap| / max |SciPy tap|.
const REMEZ_TAP_TOL: f64 = 1e-10;
const SPECS: usize = 240;

#[derive(Debug, Clone, Serialize)]
struct Spec {
    numtaps: usize,
    bands: Vec<f64>,
    desired: Vec<f64>,
    weight: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    /// SciPy's taps; a non-finite tap (SciPy returns NaN on some ill-posed specs) is `None`.
    taps: Option<Vec<Option<f64>>>,
    error: Option<String>,
}

/// splitmix64 on [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64) as usize % n
    }
}

/// Well-posed designs only: gains alternate 0 / 1 (lowpass, highpass, bandpass, bandstop,
/// multiband), every band at least MIN_BAND wide, every transition in [0.04, 0.12], weights in
/// [1, 10], and for even numtaps (type II, forced zero at Nyquist) a band touching 0.5 has gain 0.
/// Ill-posed specs (equal gains everywhere: zero minimax error, no alternation; or a nonzero
/// type-II Nyquist band: the Haar condition fails and the minimiser is not unique) make two
/// correct solvers' taps legitimately differ, and SciPy's exchange raises on many of them; a
/// referee LP (minimax on a dense grid) confirmed both classes. They are not taps-comparable.
fn specs() -> Vec<Spec> {
    const MIN_BAND: f64 = 0.03;
    let mut rng = Rng(0x5EED_2E3E);
    let mut out = Vec::new();
    while out.len() < SPECS {
        let numtaps = 8 + rng.below(53);
        let nbands = 2 + rng.below(3);
        let transitions: Vec<f64> = (0..nbands - 1).map(|_| 0.04 + 0.08 * rng.next()).collect();
        let spare = 0.5 - transitions.iter().sum::<f64>() - MIN_BAND * nbands as f64;
        if spare <= 0.0 {
            continue;
        }
        // Split the spare width among the bands.
        let mut cuts: Vec<f64> = (0..nbands - 1).map(|_| rng.next() * spare).collect();
        cuts.sort_by(f64::total_cmp);
        cuts.push(spare);
        let mut bands = vec![0.0];
        let (mut edge, mut previous_cut) = (0.0, 0.0);
        for k in 0..nbands {
            edge += MIN_BAND + (cuts[k] - previous_cut);
            previous_cut = cuts[k];
            bands.push(edge);
            if k + 1 < nbands {
                edge += transitions[k];
                bands.push(edge);
            }
        }
        *bands.last_mut().expect("edges") = 0.5;
        let first_gain = rng.below(2) as f64;
        let desired: Vec<f64> = (0..nbands)
            .map(|k| {
                if k.is_multiple_of(2) {
                    first_gain
                } else {
                    1.0 - first_gain
                }
            })
            .collect();
        if numtaps.is_multiple_of(2) && desired[nbands - 1] != 0.0 {
            continue;
        }
        let weight: Vec<f64> = (0..nbands).map(|_| 1.0 + 9.0 * rng.next()).collect();
        out.push(Spec {
            numtaps,
            bands,
            desired,
            weight,
        });
    }
    out
}

/// Arbitrary specs, ill-posed ones included: gains drawn from {0, 0.25, .., 1} per band
/// (equal gains, zero-error targets), bands as narrow as the draw makes them, nonzero type-II
/// Nyquist bands. SciPy's exchange raises on many of these. Since fsci runs SciPy's own
/// exchange, the same rule holds here too: equal taps, or both refuse.
fn arbitrary_specs() -> Vec<Spec> {
    let mut rng = Rng(0xA2B1_7EA5);
    (0..SPECS)
        .map(|_| {
            let numtaps = 8 + rng.below(53);
            let nbands = 2 + rng.below(3);
            let free = 0.5 - 0.04 * (nbands - 1) as f64;
            let mut cuts: Vec<f64> = (0..2 * nbands - 2).map(|_| rng.next() * free).collect();
            cuts.sort_by(f64::total_cmp);
            let mut bands = vec![0.0];
            for (i, c) in cuts.iter().enumerate() {
                // Even positions end a band, odd ones end a transition of at least 0.04.
                bands.push(c + 0.04 * i.div_ceil(2) as f64);
            }
            bands.push(0.5);
            let desired: Vec<f64> = (0..nbands)
                .map(|_| (rng.next() * 4.0).round() / 4.0)
                .collect();
            let weight: Vec<f64> = (0..nbands).map(|_| 0.5 + 9.5 * rng.next()).collect();
            Spec {
                numtaps,
                bands,
                desired,
                weight,
            }
        })
        .collect()
}

/// max over the bands of weight · | |H(f)| - desired | on 2000 points per band.
fn weighted_band_error(taps: &[f64], spec: &Spec) -> f64 {
    let mut worst = 0.0_f64;
    for (k, (&d, &w)) in spec.desired.iter().zip(&spec.weight).enumerate() {
        let (lo, hi) = (spec.bands[2 * k], spec.bands[2 * k + 1]);
        for j in 0..2000 {
            let f = lo + (hi - lo) * j as f64 / 1999.0;
            let (mut re, mut im) = (0.0, 0.0);
            for (n, &h) in taps.iter().enumerate() {
                let phase = -std::f64::consts::TAU * f * n as f64;
                re += h * phase.cos();
                im += h * phase.sin();
            }
            worst = worst.max(w * (re.hypot(im) - d).abs());
        }
    }
    worst
}

fn scipy_answers(specs: &[Spec]) -> Option<Vec<Answer>> {
    let script = r#"
import json, math, sys, warnings
from scipy.signal import remez

out = []
for s in json.load(sys.stdin):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            h = remez(s["numtaps"], s["bands"], s["desired"], weight=s["weight"], fs=1.0)
        out.append({"taps": [float(v) if math.isfinite(v) else None for v in h], "error": None})
    except Exception as e:
        out.append({"taps": None, "error": f"{type(e).__name__}: {e}"})
print(json.dumps(out))
"#;
    let query = serde_json::to_string(specs).expect("serialize specs");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the remez oracle: {e}"
            );
            eprintln!("skipping remez oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for the remez oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "remez oracle failed: {stderr}"
        );
        eprintln!("skipping remez oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse remez oracle JSON"))
}

#[test]
fn diff_signal_remez_even() {
    let specs: Vec<Spec> = specs().into_iter().chain(arbitrary_specs()).collect();
    let Some(answers) = scipy_answers(&specs) else {
        return;
    };
    assert_eq!(
        answers.len(),
        specs.len(),
        "the oracle must answer every spec"
    );
    let (mut compared, mut both_designed, mut both_refused) = (0, 0, 0);
    let mut worst = (0.0_f64, 0usize);
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new("diff_signal_remez_even", &["remez"]);
    for (i, (spec, answer)) in specs.iter().zip(&answers).enumerate() {
        let case_id = format!("spec_{i}");
        let fsci = remez(spec.numtaps, &spec.bands, &spec.desired, Some(&spec.weight));
        compared += 1;
        match &answer.taps {
            None => {
                // SciPy refused: its exchange raises on many of the arbitrary specs (see
                // `arbitrary_specs`), and the rule is that fsci then refuses too.
                ledger.expected_raise("remez", &case_id, fsci.is_err());
                if fsci.is_err() {
                    both_refused += 1;
                } else {
                    failures.push(format!(
                        "spec {i} (numtaps {}, bands {:?}): fsci designed, SciPy refused: {:?}",
                        spec.numtaps, spec.bands, answer.error
                    ));
                }
            }
            Some(want) => {
                let Some((want, taps)) =
                    ledger.both("remez", &case_id, Some(want), fsci.as_ref().ok())
                else {
                    failures.push(format!(
                        "spec {i} (numtaps {}, bands {:?}): SciPy designed, fsci refused: {:?}",
                        spec.numtaps,
                        spec.bands,
                        fsci.as_ref().err()
                    ));
                    continue;
                };
                both_designed += 1;
                // Non-finite taps must sit at the same positions on both sides.
                let same_finiteness = taps.len() == want.len()
                    && taps
                        .iter()
                        .zip(want)
                        .all(|(a, b)| a.is_finite() == b.is_some());
                let scale = want
                    .iter()
                    .flatten()
                    .fold(0.0_f64, |m, v| m.max(v.abs()))
                    .max(1e-300);
                let err = taps
                    .iter()
                    .zip(want)
                    .filter_map(|(a, b)| b.map(|b| (a - b).abs()))
                    .fold(0.0_f64, f64::max)
                    / scale;
                if err > worst.0 || err.is_nan() {
                    worst = (err, i);
                }
                let failed = !same_finiteness || err.is_nan() || err > REMEZ_TAP_TOL;
                ledger.compared("remez", &case_id, !failed);
                if failed {
                    let want: Vec<f64> = want.iter().map(|b| b.unwrap_or(f64::NAN)).collect();
                    // Which design is the minimax one? The smaller max weighted band error wins.
                    failures.push(format!(
                        "spec {i} (numtaps {}, bands {:?}, desired {:?}): taps differ by {err:.2e}; \
                         max weighted band error fsci {:.6e} vs SciPy {:.6e}",
                        spec.numtaps,
                        spec.bands,
                        spec.desired,
                        weighted_band_error(taps, spec),
                        weighted_band_error(&want, spec)
                    ));
                }
            }
        }
    }
    let even = specs.iter().filter(|s| s.numtaps.is_multiple_of(2)).count();
    println!(
        "{compared} specs ({even} even numtaps): {both_designed} designed by both, \
         {both_refused} refused by both; worst tap difference {:.2e} (spec {})",
        worst.0, worst.1
    );
    assert_eq!(compared, 2 * SPECS);
    assert!(even >= SPECS / 3, "too few even-numtaps specs: {even}");
    // Must-hit for the refusal arm: the arbitrary set contains specs SciPy refuses.
    assert!(
        both_refused > 0,
        "no spec was refused by both; the refusal arm is untested"
    );
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
    ledger.finish(specs.len());
}
