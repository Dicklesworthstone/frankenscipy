#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_stats::psd_welch` — power
//! spectral density via Welch's averaged-periodogram method
//! with a Hann window.
//!
//! Resolves [frankenscipy-p4l7b]. fsci's psd_welch uses an
//! ad-hoc `power / (window_size · fs)` per-bin normalisation
//! and averages across segments; this matches neither scipy's
//! `scaling='density'` (`/(fs · Σ window²)`) nor `scaling=
//! 'spectrum'` (`/(Σ window)²`) — the constants differ by
//! factors of 8/3 (Hann density) or N²/4 (Hann spectrum).
//!
//! A direct diff harness would require either picking and
//! pinning a scipy normalisation that fsci doesn't quite use,
//! or normalising both sides before comparison. Instead this
//! harness verifies normalisation-invariant METAMORPHIC
//! properties that any sensible PSD estimator must satisfy:
//!
//!   1. Output length is `window_size / 2 + 1` (one-sided FFT).
//!   2. For a pure-tone input at frequency f₀, the peak bin is
//!      at the bin closest to f₀.
//!   3. Linearity: `psd_welch(α·x) = α² · psd_welch(x)`.
//!
//! 5 fixtures × 3 invariants = 15 cases. The fixtures cover
//! pure tones at varied frequencies and window sizes, plus a
//! degenerate edge case.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::psd_welch;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const LINEARITY_TOL: f64 = 1.0e-12;
const PEAK_BIN_TOL: usize = 1; // peak should be within ±1 bin of the analytic frequency
// (Welch's averaged periodogram with a Hann window has a
// smearing kernel that can shift the peak by up to one bin
// when the tone falls between bins).

#[derive(Debug, Clone, Serialize)]
struct CaseLog {
    case_id: String,
    invariant: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MetamorphicLog {
    test_id: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseLog>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/metamorphic"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create psd_welch metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize psd_welch metamorphic log");
    fs::write(path, json).expect("write psd_welch metamorphic log");
}

struct Fixture {
    case_id: &'static str,
    fs: f64,
    n_samples: usize,
    window_size: usize,
    overlap: usize,
    tone_freq: f64,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            case_id: "tone_25hz_fs100_n512_w64",
            fs: 100.0,
            n_samples: 512,
            window_size: 64,
            overlap: 32,
            tone_freq: 25.0,
        },
        Fixture {
            case_id: "tone_10hz_fs100_n1024_w128",
            fs: 100.0,
            n_samples: 1024,
            window_size: 128,
            overlap: 64,
            tone_freq: 10.0,
        },
        Fixture {
            case_id: "tone_50hz_fs200_n2048_w256",
            fs: 200.0,
            n_samples: 2048,
            window_size: 256,
            overlap: 128,
            tone_freq: 50.0,
        },
        Fixture {
            case_id: "tone_5hz_fs50_n256_w32",
            fs: 50.0,
            n_samples: 256,
            window_size: 32,
            overlap: 16,
            tone_freq: 5.0,
        },
        Fixture {
            case_id: "tone_aligned_4hz_fs16_n128_w16",
            fs: 16.0,
            n_samples: 128,
            window_size: 16,
            overlap: 8,
            tone_freq: 4.0,
        },
    ]
}

fn make_tone(fs: f64, n: usize, freq: f64, amplitude: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()
        })
        .collect()
}

#[test]
fn metamorphic_stats_psd_welch() {
    let start = Instant::now();
    let mut cases = Vec::new();
    let fixtures = fixtures();

    for f in &fixtures {
        let x = make_tone(f.fs, f.n_samples, f.tone_freq, 1.0);
        let psd = psd_welch(&x, f.window_size, f.overlap, f.fs);

        // Invariant 1: output length is window_size / 2 + 1.
        let expected_len = f.window_size / 2 + 1;
        let length_pass = psd.len() == expected_len;
        cases.push(CaseLog {
            case_id: f.case_id.to_string(),
            invariant: "length_eq_n_over_2_plus_1".into(),
            detail: format!("psd.len()={}, expected={}", psd.len(), expected_len),
            pass: length_pass,
        });

        // Invariant 2: peak bin is closest to the analytic frequency.
        let bin_width = f.fs / f.window_size as f64;
        let expected_peak_bin = (f.tone_freq / bin_width).round() as usize;
        let actual_peak_bin = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let bin_distance = actual_peak_bin.abs_diff(expected_peak_bin);
        let peak_pass = bin_distance <= PEAK_BIN_TOL;
        cases.push(CaseLog {
            case_id: f.case_id.to_string(),
            invariant: "peak_bin_at_tone_freq".into(),
            detail: format!(
                "actual_bin={}, expected_bin={}, distance={}",
                actual_peak_bin, expected_peak_bin, bin_distance
            ),
            pass: peak_pass,
        });

        // Invariant 3: scale linearity. PSD is quadratic in amplitude,
        //              so 2x amplitude gives 4x PSD.
        let x2 = make_tone(f.fs, f.n_samples, f.tone_freq, 2.0);
        let psd2 = psd_welch(&x2, f.window_size, f.overlap, f.fs);
        let mut max_rel_diff = 0.0_f64;
        for (a, b) in psd.iter().zip(psd2.iter()) {
            if *a > 1e-15 {
                let expected = 4.0 * a;
                let rel_diff = (b - expected).abs() / expected.max(1e-15);
                max_rel_diff = max_rel_diff.max(rel_diff);
            }
        }
        let linearity_pass = max_rel_diff <= LINEARITY_TOL;
        cases.push(CaseLog {
            case_id: f.case_id.to_string(),
            invariant: "amplitude_quadratic_scaling".into(),
            detail: format!("max_rel_diff={max_rel_diff}"),
            pass: linearity_pass,
        });
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_psd_welch".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "psd_welch metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "psd_welch metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
