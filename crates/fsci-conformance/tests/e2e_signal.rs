#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-011 (Signal Processing).
//!
//! Implements conformance tests for scipy.signal parity:
//!   Happy-path (1-5): window functions, convolution, filtering
//!   Edge cases (6-8): peak detection, spectral analysis
//!   Cross-op consistency (9-11): Hilbert transform, correlation
//!   Performance boundary (12-14): large signals, CZT, CWT
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-011/e2e/`.

use fsci_conformance::PacketFamily;
use fsci_signal::{
    BaCoeffs, ConvolveMode, FilterType, FindPeaksOptions, FirWindow, PeakCondition, SosSection,
    SpectralScaling, autocorrelation, blackman, butter, butter_sos, convolve, correlate,
    csd_with_scaling, filtfilt, filtfilt_with_padtype, find_peaks, firwin, freqz, freqz_with_whole,
    gausspulse, get_window, get_window_with_fftbins, hamming, hann, hilbert_envelope, kaiser,
    lfilter, lfilter_zi, lfiltic, lombscargle, peak_prominences, resample,
    resample_poly_with_padtype, ricker, rms, savgol_coeffs, savgol_filter, sosfilt,
    spectral_centroid, spectral_flatness, stft, tf2sos, welch,
};
use serde::Serialize;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    signal_metadata: Option<SignalMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct SignalMetadata {
    operation: String,
    signal_length: usize,
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(PacketFamily::Signal.packet_id())
        .join("e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_signal -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).expect("failed to create e2e dir");
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).expect("failed to write bundle");
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0_f64,
            |acc, v| if v.is_nan() { f64::NAN } else { acc.max(v) },
        )
}

// ───────────────────── Scenario runner framework ──────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    passed: bool,
    error_chain: Option<String>,
    signal_meta: Option<SignalMetadata>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            passed: true,
            error_chain: None,
            signal_meta: None,
        }
    }

    fn set_signal_meta(&mut self, operation: &str, signal_length: usize, mode: &str) {
        self.signal_meta = Some(SignalMetadata {
            operation: operation.to_owned(),
            signal_length,
            mode: mode.to_owned(),
        });
    }

    fn record_step(
        &mut self,
        name: &str,
        action: &str,
        input_summary: &str,
        mode: &str,
        f: impl FnOnce() -> Result<String, String>,
    ) -> bool {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();
        let (outcome, output_summary) = match result {
            Ok(summary) => ("pass".to_owned(), summary),
            Err(err) => {
                self.passed = false;
                if self.error_chain.is_none() {
                    self.error_chain = Some(err.clone());
                }
                ("fail".to_owned(), err)
            }
        };
        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_owned(),
            action: action.to_owned(),
            input_summary: input_summary.to_owned(),
            output_summary,
            duration_ns,
            mode: mode.to_owned(),
            outcome: outcome.clone(),
        });
        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let total_duration_ns = self.start.elapsed().as_nanos();
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: Vec::new(),
            environment: make_env(),
            signal_metadata: self.signal_meta,
            overall: OverallResult {
                status: if self.passed { "pass" } else { "fail" }.to_owned(),
                total_duration_ns,
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//                       HAPPY-PATH SCENARIOS (1-5)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: Window functions - Hann, Hamming, Blackman
/// scipy.signal.windows.hann, hamming, blackman
#[test]
fn scenario_01_window_functions() {
    let mut runner = ScenarioRunner::new("scenario_01_window_functions");
    runner.set_signal_meta("window_functions", 64, "Strict");

    let n = 64;

    // Hann window
    runner.record_step(
        "hann_window",
        "hann(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = hann(n);
            // Verify symmetry: w[i] == w[n-1-i]
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            // Verify endpoints: w[0] = w[n-1] = 0 for Hann
            let endpoints_zero = w[0].abs() < 1e-14 && w[n - 1].abs() < 1e-14;
            // Verify peak at center
            let peak_at_center = w[n / 2] > 0.9;
            if symmetric && endpoints_zero && peak_at_center {
                Ok(format!("len={}, peak={:.4}", w.len(), w[n / 2]))
            } else {
                Err(format!(
                    "sym={symmetric}, endpoints={endpoints_zero}, peak={peak_at_center}"
                ))
            }
        },
    );

    // Hamming window
    runner.record_step(
        "hamming_window",
        "hamming(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = hamming(n);
            // Hamming has non-zero endpoints (0.08)
            let endpoints_correct = (w[0] - 0.08).abs() < 0.01;
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            if symmetric && endpoints_correct {
                Ok(format!("len={}, w[0]={:.4}", w.len(), w[0]))
            } else {
                Err(format!("sym={symmetric}, endpoints={endpoints_correct}"))
            }
        },
    );

    // Blackman window
    runner.record_step(
        "blackman_window",
        "blackman(64)",
        &format!("n={n}"),
        "Strict",
        || {
            let w = blackman(n);
            // Blackman has near-zero endpoints
            let endpoints_small = w[0].abs() < 0.01 && w[n - 1].abs() < 0.01;
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-14);
            if symmetric && endpoints_small {
                Ok(format!("len={}, w[0]={:.6}", w.len(), w[0]))
            } else {
                Err(format!(
                    "sym={symmetric}, endpoints_small={endpoints_small}"
                ))
            }
        },
    );

    // Kaiser window
    runner.record_step(
        "kaiser_window",
        "kaiser(64, 5.0)",
        &format!("n={n}, beta=5.0"),
        "Strict",
        || {
            let w = kaiser(n, 5.0);
            let symmetric = (0..n / 2).all(|i| (w[i] - w[n - 1 - i]).abs() < 1e-12);
            let peak_at_center = w[n / 2] > 0.9;
            if symmetric && peak_at_center {
                Ok(format!("len={}, peak={:.4}", w.len(), w[n / 2]))
            } else {
                Err(format!("sym={symmetric}, peak={peak_at_center}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_window_functions", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: Convolution modes - full, same, valid
/// scipy.signal.convolve
#[test]
fn scenario_02_convolution_modes() {
    let mut runner = ScenarioRunner::new("scenario_02_convolution_modes");
    runner.set_signal_meta("convolve", 10, "Strict");

    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel = vec![1.0, 1.0, 1.0];

    // Full convolution
    runner.record_step(
        "convolve_full",
        "convolve(signal, kernel, full)",
        "5-element signal, 3-element kernel",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Full).map_err(|e| format!("{e}"))?;
            // Full: output length = len(signal) + len(kernel) - 1 = 7
            if result.len() == 7 {
                // Check known values: sum filter
                let expected = vec![1.0, 3.0, 6.0, 9.0, 12.0, 9.0, 5.0];
                let diff = max_abs_diff(&result, &expected);
                if diff < 1e-12 {
                    Ok(format!("len={}, correct values", result.len()))
                } else {
                    Err(format!("values differ: {diff:.2e}"))
                }
            } else {
                Err(format!("expected len=7, got {}", result.len()))
            }
        },
    );

    // Same convolution
    runner.record_step(
        "convolve_same",
        "convolve(signal, kernel, same)",
        "same mode - output length = input length",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Same).map_err(|e| format!("{e}"))?;
            if result.len() == signal.len() {
                Ok(format!("len={}", result.len()))
            } else {
                Err(format!(
                    "expected len={}, got {}",
                    signal.len(),
                    result.len()
                ))
            }
        },
    );

    // Valid convolution
    runner.record_step(
        "convolve_valid",
        "convolve(signal, kernel, valid)",
        "valid mode - no boundary effects",
        "Strict",
        || {
            let result =
                convolve(&signal, &kernel, ConvolveMode::Valid).map_err(|e| format!("{e}"))?;
            // Valid: output length = len(signal) - len(kernel) + 1 = 3
            let expected_len = signal.len() - kernel.len() + 1;
            if result.len() == expected_len {
                // Valid region should be [6, 9, 12]
                let expected = vec![6.0, 9.0, 12.0];
                let diff = max_abs_diff(&result, &expected);
                if diff < 1e-12 {
                    Ok(format!("len={}, correct values", result.len()))
                } else {
                    Err(format!("values differ: {diff:.2e}"))
                }
            } else {
                Err(format!("expected len={expected_len}, got {}", result.len()))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_convolution_modes", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: Savitzky-Golay filter
/// scipy.signal.savgol_filter
#[test]
fn scenario_03_savgol_filter() {
    let mut runner = ScenarioRunner::new("scenario_03_savgol_filter");
    runner.set_signal_meta("savgol_filter", 100, "Strict");

    // Create noisy signal
    let n = 100;
    let clean: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
        .collect();
    // Add deterministic pseudo-random noise using sine at high frequency
    let noisy: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.2 * (i as f64 * 7.3).sin())
        .collect();

    // Get coefficients
    runner.record_step(
        "savgol_coeffs",
        "savgol_coeffs(7, 2, 0)",
        "window=7, polyorder=2, deriv=0",
        "Strict",
        || {
            let coeffs = savgol_coeffs(7, 2, 0).map_err(|e| format!("{e}"))?;
            // Coefficients should sum to 1 for smoothing (deriv=0)
            let sum: f64 = coeffs.iter().sum();
            if (sum - 1.0).abs() < 1e-10 {
                Ok(format!("len={}, sum={:.6}", coeffs.len(), sum))
            } else {
                Err(format!("coeffs sum={sum}, expected 1.0"))
            }
        },
    );

    // Apply filter
    runner.record_step(
        "apply_savgol",
        "savgol_filter(noisy, 7, 2)",
        &format!("filter {n}-point noisy signal"),
        "Strict",
        || {
            let filtered = savgol_filter(&noisy, 7, 2).map_err(|e| format!("{e}"))?;
            // Filtered signal should be smoother (lower variance from clean)
            let noise_var: f64 = noisy
                .iter()
                .zip(clean.iter())
                .map(|(&n, &c)| (n - c).powi(2))
                .sum::<f64>()
                / n as f64;
            let filtered_var: f64 = filtered
                .iter()
                .zip(clean.iter())
                .map(|(&f, &c)| (f - c).powi(2))
                .sum::<f64>()
                / n as f64;
            if filtered_var < noise_var * 0.5 {
                Ok(format!(
                    "noise_var={noise_var:.4}, filtered_var={filtered_var:.4}"
                ))
            } else {
                Err(format!(
                    "filtering not effective: {filtered_var:.4} >= {noise_var:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_savgol_filter", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: Correlation
/// scipy.signal.correlate
#[test]
fn scenario_04_correlation() {
    let mut runner = ScenarioRunner::new("scenario_04_correlation");
    runner.set_signal_meta("correlate", 20, "Strict");

    // Create test signals
    let signal: Vec<f64> = (0..20).map(|i| (i as f64 / 10.0).sin()).collect();
    let template: Vec<f64> = signal[5..10].to_vec(); // Extract a piece

    runner.record_step(
        "correlate_full",
        "correlate(signal, template, full)",
        "find template in signal",
        "Strict",
        || {
            let corr =
                correlate(&signal, &template, ConvolveMode::Full).map_err(|e| format!("{e}"))?;
            // Peak should be at position where template matches
            let max_idx = corr
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            // For correlation, peak indicates alignment
            // Expected peak around signal_len - 1 + template_start = 20-1+5 = 24? Let's check length
            // Full correlation length = len(a) + len(v) - 1 = 20 + 5 - 1 = 24
            if corr.len() == 24 {
                Ok(format!("len={}, peak_idx={}", corr.len(), max_idx))
            } else {
                Err(format!("expected len=24, got {}", corr.len()))
            }
        },
    );

    // Autocorrelation
    runner.record_step(
        "autocorrelation",
        "autocorrelation(signal, 10)",
        "self-correlation up to lag 10",
        "Strict",
        || {
            let acorr = autocorrelation(&signal, 10);
            // Autocorrelation at lag 0 should be maximum
            let max_val = acorr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (acorr[0] - max_val).abs() < 1e-12 {
                Ok(format!("len={}, r[0]={:.4}", acorr.len(), acorr[0]))
            } else {
                Err(format!(
                    "lag-0 not maximum: r[0]={:.4}, max={:.4}",
                    acorr[0], max_val
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_correlation", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: Gaussian pulse generation
/// scipy.signal.gausspulse
#[test]
fn scenario_05_gausspulse() {
    let mut runner = ScenarioRunner::new("scenario_05_gausspulse");
    runner.set_signal_meta("gausspulse", 101, "Strict");

    // Generate time vector centered at 0
    let t: Vec<f64> = (-50..=50).map(|i| i as f64 / 100.0).collect();
    let fc = 10.0; // Center frequency
    let bw = 0.5; // Fractional bandwidth

    runner.record_step(
        "generate_gausspulse",
        "gausspulse(t, fc=10, bw=0.5)",
        "101-point centered pulse",
        "Strict",
        || {
            let pulse = gausspulse(&t, fc, bw);
            // Pulse should be centered at t=0 (index 50)
            let max_idx = pulse
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            // Peak should be near center
            if (max_idx as i32 - 50).abs() <= 1 {
                Ok(format!("len={}, peak_idx={}", pulse.len(), max_idx))
            } else {
                Err(format!("peak not centered: idx={max_idx}"))
            }
        },
    );

    runner.record_step(
        "pulse_envelope",
        "check envelope is Gaussian",
        "verify decay shape",
        "Strict",
        || {
            let pulse = gausspulse(&t, fc, bw);
            // Envelope should decay away from center
            let center_val = pulse[50].abs();
            let edge_val = pulse[0].abs().max(pulse[100].abs());
            if edge_val < center_val * 0.1 {
                Ok(format!("center={center_val:.4}, edge={edge_val:.4}"))
            } else {
                Err(format!(
                    "envelope not decaying: center={center_val:.4}, edge={edge_val:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_gausspulse", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                       EDGE CASE SCENARIOS (6-8)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Peak finding
/// scipy.signal.find_peaks
#[test]
fn scenario_06_find_peaks() {
    let mut runner = ScenarioRunner::new("scenario_06_find_peaks");
    runner.set_signal_meta("find_peaks", 100, "Strict");

    // Create signal with known peaks
    let signal: Vec<f64> = (0..100)
        .map(|i| {
            let t = i as f64 / 100.0 * 4.0 * PI;
            t.sin() + 0.5 * (3.0 * t).sin()
        })
        .collect();

    runner.record_step(
        "find_basic_peaks",
        "find_peaks(signal, default options)",
        "find all local maxima",
        "Strict",
        || {
            let result = find_peaks(&signal, FindPeaksOptions::default())
                .map_err(|e| format!("find_peaks: {e}"))?;
            // Should find multiple peaks
            if result.peaks.len() >= 3 {
                Ok(format!(
                    "found {} peaks at {:?}",
                    result.peaks.len(),
                    &result.peaks[..3]
                ))
            } else {
                Err(format!("too few peaks: {}", result.peaks.len()))
            }
        },
    );

    runner.record_step(
        "find_peaks_with_height",
        "find_peaks with height threshold",
        "filter by minimum height",
        "Strict",
        || {
            let opts = FindPeaksOptions {
                height: Some(PeakCondition::at_least(0.5)),
                ..Default::default()
            };
            let result = find_peaks(&signal, opts).map_err(|e| format!("find_peaks: {e}"))?;
            // All found peaks should have height > 0.5
            let all_high = result.peaks.iter().all(|&i| signal[i] > 0.5);
            if all_high {
                Ok(format!("found {} peaks above 0.5", result.peaks.len()))
            } else {
                Err("some peaks below threshold".to_owned())
            }
        },
    );

    runner.record_step(
        "find_peaks_with_distance",
        "find_peaks with minimum distance",
        "enforce minimum spacing",
        "Strict",
        || {
            let opts = FindPeaksOptions {
                distance: Some(10),
                ..Default::default()
            };
            let result = find_peaks(&signal, opts).map_err(|e| format!("find_peaks: {e}"))?;
            // Check minimum distance between consecutive peaks
            let min_dist = result
                .peaks
                .windows(2)
                .map(|w| w[1] - w[0])
                .min()
                .unwrap_or(100);
            if min_dist >= 10 {
                Ok(format!(
                    "found {} peaks, min_dist={}",
                    result.peaks.len(),
                    min_dist
                ))
            } else {
                Err(format!("min_dist={min_dist} < 10"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_find_peaks", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Peak prominences
/// scipy.signal.peak_prominences
#[test]
fn scenario_07_peak_prominences() {
    let mut runner = ScenarioRunner::new("scenario_07_peak_prominences");
    runner.set_signal_meta("peak_prominences", 50, "Strict");

    // Create signal with peaks of different prominences
    let signal: Vec<f64> = (0..50)
        .map(|i| {
            let t = i as f64;
            if i == 10 {
                5.0
            } else if i == 30 {
                3.0
            } else {
                (t * 0.1).sin()
            }
        })
        .collect();
    let peaks = vec![10, 30];

    runner.record_step(
        "compute_prominences",
        "peak_prominences(signal, [10, 30])",
        "compute prominence for known peaks",
        "Strict",
        || {
            let (proms, left_bases, right_bases) = peak_prominences(&signal, &peaks);
            // Peak at 10 (height 5) should have higher prominence than peak at 30 (height 3)
            if proms.len() == 2 && proms[0] > proms[1] {
                Ok(format!(
                    "proms={:?}, left={:?}, right={:?}",
                    proms, left_bases, right_bases
                ))
            } else {
                Err(format!("unexpected prominences: {:?}", proms))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_peak_prominences", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Spectral features
/// scipy.signal spectral analysis
#[test]
fn scenario_08_spectral_features() {
    let mut runner = ScenarioRunner::new("scenario_08_spectral_features");
    runner.set_signal_meta("spectral", 64, "Strict");

    // Create magnitude spectrum (simulated FFT output)
    let n = 64;
    let freqs: Vec<f64> = (0..n).map(|i| i as f64 * 100.0 / n as f64).collect();
    // Magnitude spectrum with peak at index 16 (freq=25 Hz)
    let magnitudes: Vec<f64> = (0..n)
        .map(|i| {
            let dist = (i as f64 - 16.0).abs();
            (-dist * dist / 20.0).exp()
        })
        .collect();

    runner.record_step(
        "spectral_centroid",
        "spectral_centroid(magnitudes, freqs)",
        "compute frequency centroid",
        "Strict",
        || {
            let centroid = spectral_centroid(&magnitudes, &freqs);
            // Centroid should be near the peak frequency (25 Hz)
            if (centroid - 25.0).abs() < 5.0 {
                Ok(format!("centroid={centroid:.2} Hz"))
            } else {
                Err(format!("centroid={centroid:.2} Hz, expected ~25"))
            }
        },
    );

    runner.record_step(
        "spectral_flatness",
        "spectral_flatness(magnitudes)",
        "measure spectral flatness",
        "Strict",
        || {
            let flatness = spectral_flatness(&magnitudes);
            // Peaked spectrum should have low flatness (< 1)
            // Flatness = geom_mean / arith_mean, always <= 1
            if (0.0..=1.0).contains(&flatness) {
                Ok(format!("flatness={flatness:.4}"))
            } else {
                Err(format!("flatness={flatness:.4} out of range"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_08_spectral_features", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   CROSS-OP CONSISTENCY SCENARIOS (9-11)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Hilbert transform envelope
/// scipy.signal.hilbert
#[test]
fn scenario_09_hilbert_envelope() {
    let mut runner = ScenarioRunner::new("scenario_09_hilbert_envelope");
    runner.set_signal_meta("hilbert", 128, "Strict");

    // Create amplitude-modulated signal
    let n = 128;
    let carrier_freq = 20.0;
    let mod_freq = 2.0;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            // AM signal: (1 + 0.5*sin(mod)) * sin(carrier)
            (1.0 + 0.5 * (2.0 * PI * mod_freq * t).sin()) * (2.0 * PI * carrier_freq * t).sin()
        })
        .collect();

    runner.record_step(
        "compute_envelope",
        "hilbert_envelope(signal)",
        "extract amplitude envelope",
        "Strict",
        || {
            let envelope = hilbert_envelope(&signal).map_err(|e| format!("{e}"))?;
            // Envelope should be positive and follow the modulation
            let all_positive = envelope.iter().all(|&e| e >= 0.0);
            // RMS of envelope should be reasonable
            let env_rms = rms(&envelope);
            if all_positive && env_rms > 0.5 && env_rms < 2.0 {
                Ok(format!("len={}, rms={:.4}", envelope.len(), env_rms))
            } else {
                Err(format!("positive={all_positive}, rms={env_rms:.4}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_hilbert_envelope", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: Ricker wavelet
/// scipy.signal.ricker
#[test]
fn scenario_10_ricker_wavelet() {
    let mut runner = ScenarioRunner::new("scenario_10_ricker_wavelet");
    runner.set_signal_meta("ricker", 101, "Strict");

    runner.record_step(
        "generate_ricker",
        "ricker(101, 4.0)",
        "Mexican hat wavelet",
        "Strict",
        || {
            let wavelet = ricker(101, 4.0);
            // Ricker wavelet should be centered with positive peak
            let center = wavelet[50];
            // Should have negative sidelobes
            let has_negative = wavelet.iter().any(|&v| v < 0.0);
            // Should integrate to approximately zero
            let integral: f64 = wavelet.iter().sum();
            if center > 0.0 && has_negative && integral.abs() < 0.1 {
                Ok(format!("center={center:.4}, integral={integral:.4}"))
            } else {
                Err(format!(
                    "center={center:.4}, has_neg={has_negative}, integral={integral:.4}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_ricker_wavelet", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: RMS and peak-to-peak consistency
#[test]
fn scenario_11_signal_metrics() {
    let mut runner = ScenarioRunner::new("scenario_11_signal_metrics");
    runner.set_signal_meta("metrics", 100, "Strict");

    // Pure sine wave
    let signal: Vec<f64> = (0..100)
        .map(|i| (2.0 * PI * i as f64 / 100.0).sin())
        .collect();

    runner.record_step(
        "rms_sine",
        "rms(sin)",
        "RMS of pure sine should be 1/sqrt(2)",
        "Strict",
        || {
            let r = rms(&signal);
            let expected = 1.0 / 2.0_f64.sqrt(); // ~0.707
            if (r - expected).abs() < 0.02 {
                Ok(format!("rms={r:.4}, expected={expected:.4}"))
            } else {
                Err(format!("rms={r:.4}, expected={expected:.4}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_signal_metrics", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   PERFORMANCE BOUNDARY SCENARIOS (12-14)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large signal convolution
#[test]
fn scenario_12_large_convolution() {
    let mut runner = ScenarioRunner::new("scenario_12_large_convolution");
    runner.set_signal_meta("convolve", 10000, "Strict");

    let n = 10000;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let kernel: Vec<f64> = (0..101).map(|_| 1.0 / 101.0).collect(); // Moving average

    runner.record_step(
        "convolve_large",
        &format!("convolve({n}, 101, same)"),
        "large signal convolution",
        "Strict",
        || {
            let start = Instant::now();
            let result =
                convolve(&signal, &kernel, ConvolveMode::Same).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if result.len() == n && elapsed.as_millis() < 500 {
                Ok(format!("len={}, time={:?}", result.len(), elapsed))
            } else {
                Err(format!("len={}, time={:?}", result.len(), elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_convolution", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: Large window function
#[test]
fn scenario_13_large_window() {
    let mut runner = ScenarioRunner::new("scenario_13_large_window");
    runner.set_signal_meta("window", 65536, "Strict");

    let n = 65536;

    runner.record_step(
        "large_kaiser",
        &format!("kaiser({n}, 8.0)"),
        "large Kaiser window",
        "Strict",
        || {
            let start = Instant::now();
            let w = kaiser(n, 8.0);
            let elapsed = start.elapsed();
            // Check basic properties still hold
            let symmetric = (w[0] - w[n - 1]).abs() < 1e-10;
            if w.len() == n && symmetric && elapsed.as_millis() < 100 {
                Ok(format!("len={}, time={:?}", w.len(), elapsed))
            } else {
                Err(format!(
                    "len={}, symmetric={symmetric}, time={:?}",
                    w.len(),
                    elapsed
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_large_window", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: Savgol on large signal
#[test]
fn scenario_14_large_savgol() {
    let mut runner = ScenarioRunner::new("scenario_14_large_savgol");
    runner.set_signal_meta("savgol_filter", 100000, "Strict");

    let n = 100000;
    let signal: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.001).sin() + 0.1 * ((i * 17) as f64 % 1.0))
        .collect();

    runner.record_step(
        "savgol_large",
        &format!("savgol_filter({n}, 21, 3)"),
        "filter large signal",
        "Strict",
        || {
            let start = Instant::now();
            let filtered = savgol_filter(&signal, 21, 3).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if filtered.len() == n && elapsed.as_millis() < 1000 {
                Ok(format!("len={}, time={:?}", filtered.len(), elapsed))
            } else {
                Err(format!("len={}, time={:?}", filtered.len(), elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_large_savgol", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}

// ══════════════════ IIR FILTER DESIGN & FILTERING ══════════════════

/// Scenario 15: Butterworth filter design and lfilter
#[test]
fn scenario_15_butter_lfilter() {
    let mut runner = ScenarioRunner::new("scenario_15_butter_lfilter");
    runner.set_signal_meta("butter+lfilter", 1000, "Strict");

    // Generate test signal: 10Hz + 50Hz components
    let fs = 1000.0; // 1kHz sample rate
    let n = 1000;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * PI * 10.0 * ti).sin() + 0.5 * (2.0 * PI * 50.0 * ti).sin())
        .collect();

    let mut ba_coeffs: Option<BaCoeffs> = None;

    runner.record_step(
        "design_butter_lowpass",
        "butter(4, [0.03], lowpass)",
        "4th order Butterworth lowpass at 15Hz (wn=15/500=0.03)",
        "Strict",
        || {
            let wn = vec![0.03]; // Normalized cutoff (15Hz at 500Hz Nyquist)
            let ba = butter(4, &wn, FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            if ba.b.len() != 5 || ba.a.len() != 5 {
                return Err(format!(
                    "unexpected coefficient lengths: b={}, a={}",
                    ba.b.len(),
                    ba.a.len()
                ));
            }
            ba_coeffs = Some(ba);
            Ok("4th order Butterworth designed".to_string())
        },
    );

    runner.record_step(
        "apply_lfilter",
        "lfilter(b, a, signal)",
        "apply filter to mixed-frequency signal",
        "Strict",
        || {
            let ba = ba_coeffs.as_ref().unwrap();
            let filtered = lfilter(&ba.b, &ba.a, &signal, None).map_err(|e| format!("{e}"))?;
            if filtered.len() != n {
                return Err(format!("filtered length {} != {}", filtered.len(), n));
            }
            // After transient, the 50Hz component should be attenuated
            // Check that variance of latter half is reduced compared to original
            let orig_var: f64 = signal[500..].iter().map(|x| x * x).sum::<f64>() / 500.0;
            let filt_var: f64 = filtered[500..].iter().map(|x| x * x).sum::<f64>() / 500.0;
            if filt_var < orig_var {
                Ok(format!(
                    "filtering reduced variance: {:.4} -> {:.4}",
                    orig_var, filt_var
                ))
            } else {
                Err(format!(
                    "filtering did not reduce variance: {:.4} -> {:.4}",
                    orig_var, filt_var
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_15_butter_lfilter", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_15 failed");
}

/// Scenario 16: SOS filtering with sosfilt
#[test]
fn scenario_16_sosfilt() {
    let mut runner = ScenarioRunner::new("scenario_16_sosfilt");
    runner.set_signal_meta("sosfilt", 1000, "Strict");

    // Generate impulse response test
    let n = 100;
    let mut impulse = vec![0.0; n];
    impulse[0] = 1.0;

    // Simple 2nd order section: lowpass biquad
    // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    // SosSection = [b0, b1, b2, a0, a1, a2] where a0=1.0
    let sos: Vec<SosSection> = vec![
        [0.0675, 0.135, 0.0675, 1.0, -1.143, 0.4128], // ~0.2 normalized cutoff
    ];

    runner.record_step(
        "sosfilt_impulse_response",
        "sosfilt(sos, impulse)",
        "compute impulse response of biquad",
        "Strict",
        || {
            let h = sosfilt(&sos, &impulse).map_err(|e| format!("{e}"))?;
            if h.len() != n {
                return Err(format!("output length {} != {}", h.len(), n));
            }
            // First sample should equal b0
            if (h[0] - sos[0][0]).abs() > 1e-10 {
                return Err(format!("h[0]={} != b0={}", h[0], sos[0][0]));
            }
            // Impulse response should decay (stable filter)
            let energy_first_half: f64 = h[..50].iter().map(|x| x * x).sum();
            let energy_second_half: f64 = h[50..].iter().map(|x| x * x).sum();
            if energy_second_half < energy_first_half {
                Ok(format!("h[0]={:.4}, decaying response verified", h[0]))
            } else {
                Err("impulse response not decaying".to_string())
            }
        },
    );

    runner.record_step(
        "sosfilt_sine_attenuation",
        "sosfilt(sos, high_freq_sine)",
        "verify high frequencies are attenuated",
        "Strict",
        || {
            // High frequency sine (0.4 normalized = above cutoff of 0.2)
            let high_freq: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.4 * i as f64).sin()).collect();
            let filtered = sosfilt(&sos, &high_freq).map_err(|e| format!("{e}"))?;

            // Steady-state should be attenuated (skip transient)
            let input_power: f64 = high_freq[50..].iter().map(|x| x * x).sum::<f64>() / 50.0;
            let output_power: f64 = filtered[50..].iter().map(|x| x * x).sum::<f64>() / 50.0;
            let attenuation_db = 10.0 * (output_power / input_power).log10();

            if attenuation_db < -3.0 {
                Ok(format!("high freq attenuated by {:.1} dB", -attenuation_db))
            } else {
                Err(format!(
                    "insufficient attenuation: {:.1} dB",
                    -attenuation_db
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_16_sosfilt", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_16 failed");
}

/// Scenario 17: Zero-phase filtering with filtfilt
#[test]
fn scenario_17_filtfilt() {
    let mut runner = ScenarioRunner::new("scenario_17_filtfilt");
    runner.set_signal_meta("filtfilt", 500, "Strict");

    // Generate signal with sharp edge
    let n = 500;
    let mut signal = vec![0.0; n];
    for sample in signal.iter_mut().take(400).skip(100) {
        *sample = 1.0;
    }
    // Add noise
    for (i, sample) in signal.iter_mut().enumerate().take(n) {
        *sample += 0.1 * ((i * 37) as f64 % 1.0 - 0.5);
    }

    runner.record_step(
        "filtfilt_preserves_phase",
        "filtfilt(b, a, signal)",
        "zero-phase filtering preserves edge location",
        "Strict",
        || {
            // Simple moving average as IIR: b=[0.2,0.2,0.2,0.2,0.2], a=[1.0]
            let b = vec![0.2, 0.2, 0.2, 0.2, 0.2];
            let a = vec![1.0];

            let filtered = filtfilt(&b, &a, &signal).map_err(|e| format!("{e}"))?;

            if filtered.len() != n {
                return Err(format!("output length {} != {}", filtered.len(), n));
            }

            // Find edge locations (where signal crosses 0.5)
            let orig_edge = signal
                .windows(2)
                .position(|w| w[0] < 0.5 && w[1] >= 0.5)
                .unwrap_or(0);
            let filt_edge = filtered
                .windows(2)
                .position(|w| w[0] < 0.5 && w[1] >= 0.5)
                .unwrap_or(0);

            // Zero-phase filtering should preserve edge location
            let edge_shift = (orig_edge as i64 - filt_edge as i64).abs();
            if edge_shift <= 2 {
                Ok(format!(
                    "edge preserved: orig={}, filt={}, shift={}",
                    orig_edge, filt_edge, edge_shift
                ))
            } else {
                Err(format!(
                    "edge shifted too much: orig={}, filt={}, shift={}",
                    orig_edge, filt_edge, edge_shift
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_17_filtfilt", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_17 failed");
}

/// Scenario 18: FIR filter design with firwin
#[test]
fn scenario_18_firwin() {
    let mut runner = ScenarioRunner::new("scenario_18_firwin");
    runner.set_signal_meta("firwin", 101, "Strict");

    runner.record_step(
        "firwin_lowpass",
        "firwin(101, [0.3], hamming, pass_zero=true)",
        "design 101-tap lowpass FIR",
        "Strict",
        || {
            let h = firwin(101, &[0.3], FirWindow::Hamming, true).map_err(|e| format!("{e}"))?;
            if h.len() != 101 {
                return Err(format!("filter length {} != 101", h.len()));
            }
            // FIR lowpass should be symmetric
            let symmetric = (0..50).all(|i| (h[i] - h[100 - i]).abs() < 1e-12);
            if !symmetric {
                return Err("filter not symmetric".to_string());
            }
            // DC gain should be ~1.0 for lowpass
            let dc_gain: f64 = h.iter().sum();
            if (dc_gain - 1.0).abs() < 0.01 {
                Ok(format!(
                    "symmetric FIR, DC gain={:.4}, len={}",
                    dc_gain,
                    h.len()
                ))
            } else {
                Err(format!("DC gain {} != 1.0", dc_gain))
            }
        },
    );

    runner.record_step(
        "firwin_bandpass",
        "firwin(101, [0.2, 0.4], hamming, pass_zero=false)",
        "design 101-tap bandpass FIR",
        "Strict",
        || {
            let h =
                firwin(101, &[0.2, 0.4], FirWindow::Hamming, false).map_err(|e| format!("{e}"))?;
            if h.len() != 101 {
                return Err(format!("filter length {} != 101", h.len()));
            }
            // Bandpass should have ~0 DC gain
            let dc_gain: f64 = h.iter().sum();
            if dc_gain.abs() < 0.1 {
                Ok(format!("bandpass FIR, DC gain={:.4} (near zero)", dc_gain))
            } else {
                Err(format!("bandpass DC gain {} not near zero", dc_gain))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_18_firwin", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_18 failed");
}

/// Scenario 19: STFT (Short-Time Fourier Transform)
#[test]
fn scenario_19_stft() {
    let mut runner = ScenarioRunner::new("scenario_19_stft");
    runner.set_signal_meta("stft", 2048, "Strict");

    // Generate chirp signal (frequency increasing with time)
    let fs = 1000.0;
    let n = 2048;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = t
        .iter()
        .map(|&ti| {
            let freq = 10.0 + 200.0 * ti; // 10Hz to 210Hz over 2 seconds
            (2.0 * PI * freq * ti).sin()
        })
        .collect();

    runner.record_step(
        "compute_stft",
        "stft(signal, fs, hann, 256, 128)",
        "compute STFT of chirp signal",
        "Strict",
        || {
            let result = stft(&signal, fs, Some("hann"), Some(256), Some(128))
                .map_err(|e| format!("{e}"))?;

            // Check dimensions
            let n_freqs = result.frequencies.len();
            let n_times = result.times.len();

            if n_freqs != 129 {
                // nperseg/2 + 1 = 256/2 + 1 = 129
                return Err(format!("expected 129 frequency bins, got {}", n_freqs));
            }

            // Verify frequency range
            let f_max = result.frequencies.last().unwrap_or(&0.0);
            if (*f_max - fs / 2.0).abs() > 1.0 {
                return Err(format!("max freq {} != Nyquist {}", f_max, fs / 2.0));
            }

            Ok(format!(
                "STFT: {} freq bins, {} time frames, f_max={:.1}Hz",
                n_freqs, n_times, f_max
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_19_stft", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_19 failed");
}

/// Scenario 20: Resample
#[test]
fn scenario_20_resample() {
    let mut runner = ScenarioRunner::new("scenario_20_resample");
    runner.set_signal_meta("resample", 1000, "Strict");

    // Generate sinusoid at 50Hz sampled at 1kHz
    let fs_orig = 1000.0;
    let n_orig = 1000;
    let freq = 50.0;
    let signal: Vec<f64> = (0..n_orig)
        .map(|i| (2.0 * PI * freq * i as f64 / fs_orig).sin())
        .collect();

    runner.record_step(
        "resample_upsample",
        "resample(signal, 2000)",
        "upsample 1000 -> 2000 samples",
        "Strict",
        || {
            let resampled = resample(&signal, 2000).map_err(|e| format!("{e}"))?;
            if resampled.len() != 2000 {
                return Err(format!("expected 2000 samples, got {}", resampled.len()));
            }
            // Verify the signal energy is roughly preserved
            let orig_energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / n_orig as f64;
            let resamp_energy: f64 = resampled.iter().map(|x| x * x).sum::<f64>() / 2000.0;
            let energy_ratio = resamp_energy / orig_energy;
            // Energy should be roughly preserved (within 20%)
            if (energy_ratio - 1.0).abs() < 0.2 {
                Ok(format!(
                    "upsampled 1000->2000, energy ratio={:.3}",
                    energy_ratio
                ))
            } else {
                Err(format!("energy not preserved: ratio={:.3}", energy_ratio))
            }
        },
    );

    runner.record_step(
        "resample_downsample",
        "resample(signal, 500)",
        "downsample 1000 -> 500 samples",
        "Strict",
        || {
            let resampled = resample(&signal, 500).map_err(|e| format!("{e}"))?;
            if resampled.len() != 500 {
                return Err(format!("expected 500 samples, got {}", resampled.len()));
            }
            // Verify the signal energy is roughly preserved
            let orig_energy: f64 = signal.iter().map(|x| x * x).sum::<f64>() / n_orig as f64;
            let resamp_energy: f64 = resampled.iter().map(|x| x * x).sum::<f64>() / 500.0;
            let energy_ratio = resamp_energy / orig_energy;
            // Energy should be roughly preserved (within 20%)
            if (energy_ratio - 1.0).abs() < 0.2 {
                Ok(format!(
                    "downsampled 1000->500, energy ratio={:.3}",
                    energy_ratio
                ))
            } else {
                Err(format!("energy not preserved: ratio={:.3}", energy_ratio))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_20_resample", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_20 failed");
}

/// Scenario 21: Frequency response with freqz
#[test]
fn scenario_21_freqz() {
    let mut runner = ScenarioRunner::new("scenario_21_freqz");
    runner.set_signal_meta("freqz", 512, "Strict");

    runner.record_step(
        "freqz_butter_lowpass",
        "freqz(butter(4, 0.25))",
        "frequency response of 4th order Butterworth",
        "Strict",
        || {
            // Design 4th order Butterworth lowpass at normalized freq 0.25
            let ba = butter(4, &[0.25], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let resp = freqz(&ba.b, &ba.a, Some(512)).map_err(|e| format!("{e}"))?;

            if resp.w.len() != 512 || resp.h_mag.len() != 512 {
                return Err(format!(
                    "expected 512 points, got w={}, h={}",
                    resp.w.len(),
                    resp.h_mag.len()
                ));
            }

            // At DC (w=0), magnitude should be ~1
            let dc_mag = resp.h_mag[0];
            // At Nyquist (w=pi), magnitude should be much smaller
            let nyq_mag = resp.h_mag[511];

            if (dc_mag - 1.0).abs() < 0.01 && nyq_mag < 0.1 {
                Ok(format!("DC mag={:.4}, Nyquist mag={:.4}", dc_mag, nyq_mag))
            } else {
                Err(format!(
                    "unexpected response: DC={:.4}, Nyquist={:.4}",
                    dc_mag, nyq_mag
                ))
            }
        },
    );

    runner.record_step(
        "freqz_3db_point",
        "verify -3dB at cutoff",
        "Butterworth -3dB at cutoff frequency",
        "Strict",
        || {
            let ba = butter(4, &[0.25], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let resp = freqz(&ba.b, &ba.a, Some(512)).map_err(|e| format!("{e}"))?;

            // freqz returns normalized frequencies from 0 to pi
            // For cutoff at 0.25 (normalized to Nyquist), the actual angular freq is 0.25*pi
            // With 512 points from 0 to pi, index for 0.25*pi is ~128
            let cutoff_idx = (0.25 * 512.0) as usize; // ~128
            let mag_at_cutoff = resp.h_mag[cutoff_idx];
            let db_at_cutoff: f64 = 20.0 * mag_at_cutoff.log10();

            // Should be around -3dB (allow some tolerance for discrete sampling)
            if (db_at_cutoff + 3.0).abs() < 2.0 {
                Ok(format!("magnitude at cutoff = {:.2} dB", db_at_cutoff))
            } else {
                Err(format!(
                    "expected ~-3dB at cutoff, got {:.2} dB",
                    db_at_cutoff
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_21_freqz", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_21 failed");
}

/// Scenario 22: tf2sos conversion
#[test]
fn scenario_22_tf2sos() {
    let mut runner = ScenarioRunner::new("scenario_22_tf2sos");
    runner.set_signal_meta("tf2sos", 0, "Strict");

    runner.record_step(
        "butter_to_sos",
        "tf2sos(butter(4, 0.3))",
        "convert Butterworth to SOS form",
        "Strict",
        || {
            let ba = butter(4, &[0.3], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let sos = tf2sos(&ba.b, &ba.a).map_err(|e| format!("{e}"))?;

            // 4th order = 2 second-order sections
            if sos.len() != 2 {
                return Err(format!("expected 2 SOS sections, got {}", sos.len()));
            }

            // Each section should have 6 coefficients
            for (i, section) in sos.iter().enumerate() {
                if section.len() != 6 {
                    return Err(format!(
                        "section {} has {} coeffs, expected 6",
                        i,
                        section.len()
                    ));
                }
                // a0 should be 1.0
                if (section[3] - 1.0).abs() > 1e-10 {
                    return Err(format!("section {} a0={} != 1.0", i, section[3]));
                }
            }

            Ok(format!(
                "{} SOS sections, each properly normalized",
                sos.len()
            ))
        },
    );

    runner.record_step(
        "sos_filter_equivalence",
        "sosfilt(sos) == lfilter(b,a)",
        "SOS and TF forms give same result",
        "Strict",
        || {
            let ba = butter(4, &[0.3], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let sos = tf2sos(&ba.b, &ba.a).map_err(|e| format!("{e}"))?;

            // Test signal
            let signal: Vec<f64> = (0..100)
                .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
                .collect();

            let tf_result = lfilter(&ba.b, &ba.a, &signal, None).map_err(|e| format!("{e}"))?;
            let sos_result = sosfilt(&sos, &signal).map_err(|e| format!("{e}"))?;

            let max_diff = max_abs_diff(&tf_result, &sos_result);
            if max_diff < 1e-10 {
                Ok(format!(
                    "TF and SOS forms equivalent, max_diff={:.2e}",
                    max_diff
                ))
            } else {
                Err(format!("TF and SOS differ by {:.2e}", max_diff))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_22_tf2sos", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_22 failed");
}

/// Scenario 23: lfiltic reconstructs lfilter state from history.
#[test]
fn scenario_23_lfiltic() {
    let mut runner = ScenarioRunner::new("scenario_23_lfiltic");
    runner.set_signal_meta("lfiltic", 10, "Strict");

    let b = vec![0.5, 1.0, 0.2];
    let a = vec![2.0, 1.0, 3.0];
    let reference_y = vec![1.2, -0.7, 0.4];
    let reference_x = vec![0.25, -1.5];
    let y_history = vec![-2.72515625, 0.7971875, 1.499375, -0.70125, -0.3625, 0.725];
    let x_history = vec![0.25, -0.5, 1.1, 0.7, -0.2, 1.3];
    let x_future = vec![0.9, -0.1, 0.3, -0.8];
    let mut zi: Option<Vec<f64>> = None;

    runner.record_step(
        "lfiltic_reference_vector",
        "lfiltic(b, a, y_hist, x_hist)",
        "SciPy reference vector with a[0] != 1",
        "Strict",
        || {
            let got =
                lfiltic(&b, &a, &reference_y, Some(&reference_x)).map_err(|e| format!("{e}"))?;
            let expected = [0.425, -1.775];
            if got.len() != expected.len() {
                return Err(format!("zi length {} != {}", got.len(), expected.len()));
            }
            for (index, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
                if (actual - want).abs() > 1.0e-12 {
                    return Err(format!("zi[{index}] mismatch: {actual} vs {want}"));
                }
            }
            Ok(format!("zi={got:?}"))
        },
    );

    runner.record_step(
        "lfiltic_restore_state",
        "lfiltic(history) -> zi",
        "recover final Direct Form II transposed state from prior I/O",
        "Strict",
        || {
            let got = lfiltic(&b, &a, &y_history, Some(&x_history)).map_err(|e| format!("{e}"))?;
            let expected = [0.241796875, 4.112734375];
            if got.len() != expected.len() {
                return Err(format!("zi length {} != {}", got.len(), expected.len()));
            }
            for (index, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
                if (actual - want).abs() > 1.0e-12 {
                    return Err(format!("restored zi[{index}] mismatch: {actual} vs {want}"));
                }
            }
            zi = Some(got);
            Ok(format!("restored zi={expected:?}"))
        },
    );

    runner.record_step(
        "lfiltic_continuation_matches_reference",
        "lfilter(b, a, x_future, zi=lfiltic(...))",
        "continued filtering should match SciPy tail output",
        "Strict",
        || {
            let restored = zi
                .as_ref()
                .ok_or_else(|| "missing restored zi".to_string())?;
            let y_future =
                lfilter(&b, &a, &x_future, Some(restored)).map_err(|e| format!("{e}"))?;
            let expected = [0.466796875, 4.3043359375, -2.73736328125, -5.147822265625];
            if y_future.len() != expected.len() {
                return Err(format!(
                    "continued output length {} != {}",
                    y_future.len(),
                    expected.len()
                ));
            }
            for (index, (actual, want)) in y_future.iter().zip(expected.iter()).enumerate() {
                if (actual - want).abs() > 1.0e-12 {
                    return Err(format!("continued y[{index}] mismatch: {actual} vs {want}"));
                }
            }
            Ok(format!("continued output={y_future:?}"))
        },
    );

    runner.record_step(
        "lfiltic_invalid_a0",
        "lfiltic(b, a=[0,...], y, x)",
        "zero leading denominator coefficient should fail closed",
        "Strict",
        || match lfiltic(&[1.0, 2.0], &[0.0, 2.0], &[0.0, 0.0], Some(&[0.0, 1.0])) {
            Ok(value) => Err(format!("expected error, got {value:?}")),
            Err(err)
                if err
                    .to_string()
                    .contains("First `a` filter coefficient must be non-zero.") =>
            {
                Ok(err.to_string())
            }
            Err(err) => Err(format!("unexpected error: {err}")),
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_23_lfiltic", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_23 failed");
}

/// Scenario 24: get_window honors SciPy fftbins periodic/symmetric semantics.
#[test]
fn scenario_24_get_window_fftbins() {
    let mut runner = ScenarioRunner::new("scenario_24_get_window_fftbins");
    runner.set_signal_meta("get_window", 8, "Strict");

    runner.record_step(
        "hann_default_is_periodic",
        "get_window('hann', 8)",
        "SciPy default fftbins=True periodic Hann",
        "Strict",
        || {
            let got = get_window("hann", 8).map_err(|e| format!("{e}"))?;
            let expected = [
                0.0,
                0.1464466094067262,
                0.5,
                0.8535533905932737,
                1.0,
                0.8535533905932737,
                0.5,
                0.1464466094067262,
            ];
            let max_diff = max_abs_diff(&got, &expected);
            if max_diff < 1e-12 {
                Ok(format!("periodic hann max_diff={max_diff:.2e}"))
            } else {
                Err(format!("periodic hann mismatch {max_diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "hann_explicit_symmetric",
        "get_window_with_fftbins('hann', 8, false)",
        "fftbins=False should return symmetric Hann",
        "Strict",
        || {
            let got = get_window_with_fftbins("hann", 8, false).map_err(|e| format!("{e}"))?;
            let expected = [
                0.0,
                0.1882550990706332,
                0.6112604669781572,
                0.9504844339512095,
                0.9504844339512095,
                0.6112604669781572,
                0.1882550990706332,
                0.0,
            ];
            let max_diff = max_abs_diff(&got, &expected);
            if max_diff < 1e-12 {
                Ok(format!("symmetric hann max_diff={max_diff:.2e}"))
            } else {
                Err(format!("symmetric hann mismatch {max_diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "suffix_overrides_flag",
        "get_window_with_fftbins('hann_periodic', 8, false)",
        "_periodic/_symmetric suffixes should override fftbins",
        "Strict",
        || {
            let periodic =
                get_window_with_fftbins("hann_periodic", 8, false).map_err(|e| format!("{e}"))?;
            let symmetric =
                get_window_with_fftbins("hann_symmetric", 8, true).map_err(|e| format!("{e}"))?;
            let periodic_expected = [
                0.0,
                0.1464466094067262,
                0.5,
                0.8535533905932737,
                1.0,
                0.8535533905932737,
                0.5,
                0.1464466094067262,
            ];
            let symmetric_expected = [
                0.0,
                0.1882550990706332,
                0.6112604669781572,
                0.9504844339512095,
                0.9504844339512095,
                0.6112604669781572,
                0.1882550990706332,
                0.0,
            ];
            let periodic_diff = max_abs_diff(&periodic, &periodic_expected);
            let symmetric_diff = max_abs_diff(&symmetric, &symmetric_expected);
            if periodic_diff < 1e-12 && symmetric_diff < 1e-12 {
                Ok(format!(
                    "periodic_diff={periodic_diff:.2e}, symmetric_diff={symmetric_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "override mismatch periodic={periodic_diff:.2e} symmetric={symmetric_diff:.2e}"
                ))
            }
        },
    );

    runner.record_step(
        "parameterized_gaussian_fftbins",
        "get_window('gaussian,2.0', 8)",
        "parameterized windows should respect periodic vs symmetric forms",
        "Strict",
        || {
            let periodic = get_window("gaussian,2.0", 8).map_err(|e| format!("{e}"))?;
            let symmetric =
                get_window_with_fftbins("gaussian,2.0", 8, false).map_err(|e| format!("{e}"))?;
            let periodic_expected = [
                0.1353352832366127,
                0.32465246735834974,
                0.6065306597126334,
                0.8824969025845955,
                1.0,
                0.8824969025845955,
                0.6065306597126334,
                0.32465246735834974,
            ];
            let symmetric_expected = [
                0.2162651668298873,
                0.45783336177161427,
                0.7548396019890073,
                0.9692332344763441,
                0.9692332344763441,
                0.7548396019890073,
                0.45783336177161427,
                0.2162651668298873,
            ];
            let periodic_diff = max_abs_diff(&periodic, &periodic_expected);
            let symmetric_diff = max_abs_diff(&symmetric, &symmetric_expected);
            if periodic_diff < 1e-12 && symmetric_diff < 1e-12 {
                Ok(format!(
                    "periodic_diff={periodic_diff:.2e}, symmetric_diff={symmetric_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "gaussian mismatch periodic={periodic_diff:.2e} symmetric={symmetric_diff:.2e}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_24_get_window_fftbins", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_24 failed");
}

/// Scenario 25: welch clamps oversized nperseg to signal length.
#[test]
fn scenario_25_welch_clamps_oversized_nperseg() {
    let mut runner = ScenarioRunner::new("scenario_25_welch_clamps_oversized_nperseg");
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    runner.set_signal_meta("welch", x.len(), "Strict");

    runner.record_step(
        "welch_oversized_nperseg_clamps",
        "welch(x, fs=10.0, nperseg=16)",
        "oversized nperseg should clamp to signal length and match SciPy reference output",
        "Strict",
        || {
            let got = welch(&x, 10.0, None, Some(16), None).map_err(|e| format!("{e}"))?;
            let clamped =
                welch(&x, 10.0, None, Some(x.len()), None).map_err(|e| format!("{e}"))?;
            let expected_frequencies = [0.0, 2.0, 4.0];

            let freq_diff = max_abs_diff(&got.frequencies, &expected_frequencies);
            let clamp_freq_diff = max_abs_diff(&got.frequencies, &clamped.frequencies);
            let clamp_psd_diff = max_abs_diff(&got.psd, &clamped.psd);

            if freq_diff < 1e-12 && clamp_freq_diff < 1e-12 && clamp_psd_diff < 1e-12 {
                Ok(format!(
                    "freq_diff={freq_diff:.2e}, clamp_psd_diff={clamp_psd_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "welch clamp mismatch freq={freq_diff:.2e} clamp_freq={clamp_freq_diff:.2e} clamp_psd={clamp_psd_diff:.2e}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_25_welch_clamps_oversized_nperseg", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_25 failed");
}

/// Scenario 26: csd honors SciPy density/spectrum scaling with explicit noverlap.
#[test]
fn scenario_26_csd_scaling_modes() {
    let mut runner = ScenarioRunner::new("scenario_26_csd_scaling_modes");
    let fs = 16.0;
    let t: Vec<f64> = (0..32).map(|i| i as f64 / fs).collect();
    let x: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 2.0 * ti).sin()).collect();
    let y: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 2.0 * ti).cos()).collect();
    runner.set_signal_meta("csd", x.len(), "Strict");

    runner.record_step(
        "csd_density_vs_spectrum",
        "csd_with_scaling(x, y, fs=16, nperseg=8, noverlap=4)",
        "explicit noverlap should preserve SciPy density and spectrum scaling outputs",
        "Strict",
        || {
            let density = csd_with_scaling(
                &x,
                &y,
                fs,
                None,
                Some(8),
                Some(4),
                SpectralScaling::Density,
            )
            .map_err(|e| format!("{e}"))?;
            let spectrum = csd_with_scaling(
                &x,
                &y,
                fs,
                None,
                Some(8),
                Some(4),
                SpectralScaling::Spectrum,
            )
            .map_err(|e| format!("{e}"))?;

            let expected_frequencies = [0.0, 2.0, 4.0, 6.0, 8.0];
            let expected_density_re = [
                -3.962744417486146e-17,
                1.3346567516981065e-17,
                -7.103421885274257e-19,
                -2.5133161553135813e-33,
                -2.1146214215064937e-33,
            ];
            let expected_density_im = [
                0.0,
                0.16666666666666666,
                0.04166666666666669,
                2.363886302605754e-32,
                0.0,
            ];
            let expected_spectrum_re = [
                -1.127784089578213e-16,
                2.9738116731031224e-17,
                -9.686270956913913e-18,
                -2.9579909131093405e-34,
                -9.199850601359331e-33,
            ];
            let expected_spectrum_im = [0.0, 0.5, 0.12500000000000003, 6.980810313780984e-32, 0.0];

            let density_re: Vec<f64> = density.csd.iter().map(|(re, _)| *re).collect();
            let density_im: Vec<f64> = density.csd.iter().map(|(_, im)| *im).collect();
            let spectrum_re: Vec<f64> = spectrum.csd.iter().map(|(re, _)| *re).collect();
            let spectrum_im: Vec<f64> = spectrum.csd.iter().map(|(_, im)| *im).collect();

            let freq_diff = max_abs_diff(&density.frequencies, &expected_frequencies)
                .max(max_abs_diff(&spectrum.frequencies, &expected_frequencies));
            let density_re_diff = max_abs_diff(&density_re, &expected_density_re);
            let density_im_diff = max_abs_diff(&density_im, &expected_density_im);
            let spectrum_re_diff = max_abs_diff(&spectrum_re, &expected_spectrum_re);
            let spectrum_im_diff = max_abs_diff(&spectrum_im, &expected_spectrum_im);

            if freq_diff < 1e-12
                && density_re_diff < 1e-12
                && density_im_diff < 1e-12
                && spectrum_re_diff < 1e-12
                && spectrum_im_diff < 1e-12
            {
                Ok(format!(
                    "freq_diff={freq_diff:.2e}, density_im_diff={density_im_diff:.2e}, spectrum_im_diff={spectrum_im_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "csd scaling mismatch freq={freq_diff:.2e} density_re={density_re_diff:.2e} density_im={density_im_diff:.2e} spectrum_re={spectrum_re_diff:.2e} spectrum_im={spectrum_im_diff:.2e}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_26_csd_scaling_modes", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_26 failed");
}

/// Scenario 27: resample_poly honors SciPy padtype and cval edge handling.
#[test]
fn scenario_27_resample_poly_padtype_modes() {
    let mut runner = ScenarioRunner::new("scenario_27_resample_poly_padtype_modes");
    let x = vec![1.0, 2.0, 4.0, 8.0];
    runner.set_signal_meta("resample_poly", x.len(), "Strict");

    runner.record_step(
        "resample_poly_line_reference",
        "resample_poly_with_padtype(x, 3, 2, Some(\"line\"), None)",
        "global line extrapolation should match SciPy's line padtype output",
        "Strict",
        || {
            let got = resample_poly_with_padtype(&x, 3, 2, Some("line"), None)
                .map_err(|e| format!("{e}"))?;
            let expected = [
                1.000606173553777,
                1.780628511492595,
                2.331390913393649,
                4.002424694215109,
                6.739706761743146,
                8.971945980209963,
            ];
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-8 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("line padtype mismatch diff={diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "resample_poly_mean_reference",
        "resample_poly_with_padtype(x, 3, 2, Some(\"mean\"), None)",
        "background-subtracted mean padding should match SciPy's mean padtype output",
        "Strict",
        || {
            let got = resample_poly_with_padtype(&x, 3, 2, Some("mean"), None)
                .map_err(|e| format!("{e}"))?;
            let expected = [
                0.998333022727113,
                1.573448321613185,
                2.293752166053205,
                4.000151543388444,
                7.390581540874371,
                7.201952100936264,
            ];
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-8 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("mean padtype mismatch diff={diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "resample_poly_constant_cval_reference",
        "resample_poly_with_padtype(x, 3, 2, Some(\"constant\"), Some(5.0))",
        "constant padding with an explicit cval should match SciPy's nonzero boundary fill",
        "Strict",
        || {
            let got = resample_poly_with_padtype(&x, 3, 2, Some("constant"), Some(5.0))
                .map_err(|e| format!("{e}"))?;
            let expected = [
                1.000606173553777,
                1.39589024581026,
                2.44120445519666,
                4.002424694215109,
                7.179517790579072,
                7.616661693078465,
            ];
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-8 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("constant cval mismatch diff={diff:.2e}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_27_resample_poly_padtype_modes", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_27 failed");
}

/// Scenario 28: butter(..., output='sos') matches SciPy's SOS coefficients.
#[test]
fn scenario_28_butter_output_sos() {
    let mut runner = ScenarioRunner::new("scenario_28_butter_output_sos");
    runner.set_signal_meta("butter_sos", 0, "Strict");

    runner.record_step(
        "butter_sos_reference_sections",
        "butter_sos(4, [0.25], lowpass)",
        "4th-order Butterworth SOS sections should match SciPy's output='sos' coefficients",
        "Strict",
        || {
            let got = butter_sos(4, &[0.25], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let expected = [
                [
                    0.010209480791203138,
                    0.020418961582406275,
                    0.010209480791203138,
                    1.0,
                    -0.8553979327751704,
                    0.20971535775655478,
                ],
                [1.0, 2.0, 1.0, 1.0, -1.1130298541633479, 0.5740619150839545],
            ];

            if got.len() != expected.len() {
                return Err(format!(
                    "expected {} sections, got {}",
                    expected.len(),
                    got.len()
                ));
            }

            let mut max_diff = 0.0_f64;
            for (got_section, expected_section) in got.iter().zip(expected.iter()) {
                max_diff = max_diff.max(max_abs_diff(got_section, expected_section));
            }

            if max_diff < 1e-7 {
                Ok(format!("max_abs_diff={max_diff:.2e}"))
            } else {
                Err(format!("SciPy SOS mismatch diff={max_diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "butter_sos_filter_equivalence",
        "sosfilt(butter_sos(...), x) == lfilter(butter(...), x)",
        "returned SOS sections should reproduce the BA filter output on a representative signal",
        "Strict",
        || {
            let ba = butter(4, &[0.25], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let sos = butter_sos(4, &[0.25], FilterType::Lowpass).map_err(|e| format!("{e}"))?;
            let signal: Vec<f64> = (0..128)
                .map(|i| {
                    let t = i as f64;
                    (2.0 * PI * 0.05 * t).sin() + 0.35 * (2.0 * PI * 0.3 * t).sin()
                })
                .collect();

            let tf_result = lfilter(&ba.b, &ba.a, &signal, None).map_err(|e| format!("{e}"))?;
            let sos_result = sosfilt(&sos, &signal).map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&tf_result, &sos_result);

            if diff < 1e-10 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("SOS filtering mismatch diff={diff:.2e}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_28_butter_output_sos", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_28 failed");
}

/// Scenario 29: lombscargle(..., normalize=True) matches SciPy reference values.
#[test]
fn scenario_29_lombscargle_normalize() {
    let mut runner = ScenarioRunner::new("scenario_29_lombscargle_normalize");
    runner.set_signal_meta("lombscargle", 5, "Strict");

    runner.record_step(
        "lombscargle_normalized_reference_vector",
        "lombscargle(x, y, freqs, normalize=true)",
        "normalized Lomb-Scargle power should match SciPy for representative uneven samples",
        "Strict",
        || {
            let tol = 5e-9;
            let x = [0.0, 0.5, 1.1, 1.7, 2.4];
            let y = [1.0, -0.5, 0.75, 0.25, -1.25];
            let freqs = [0.5, 1.0, 1.5, 2.0];
            let expected = [
                0.411_635_223_256_532,
                0.379_193_542_532_48,
                0.341_939_227_868_014,
                0.226_926_597_192_203,
            ];

            let got = lombscargle(&x, &y, &freqs, true).map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&got, &expected);
            if diff < tol {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!(
                    "SciPy normalized lombscargle mismatch diff={diff:.2e}"
                ))
            }
        },
    );

    runner.record_step(
        "lombscargle_zero_energy_nan_sentinel",
        "lombscargle(x, zeros, freqs, normalize=true)",
        "zero-energy signals should return NaN under normalized SciPy semantics",
        "Strict",
        || {
            let x = [0.1, 0.4, 1.1, 2.0];
            let y = [0.0, 0.0, 0.0, 0.0];
            let freqs = [1.0, 2.0];
            let got = lombscargle(&x, &y, &freqs, true).map_err(|e| format!("{e}"))?;
            if got.iter().all(|value| value.is_nan()) {
                Ok(format!("nan_count={}", got.len()))
            } else {
                Err(format!("expected all NaN values, got {got:?}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_29_lombscargle_normalize", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_29 failed");
}

/// Scenario 30: freqz(..., whole=True) matches SciPy's full-unit-circle sampling.
#[test]
fn scenario_30_freqz_whole() {
    let mut runner = ScenarioRunner::new("scenario_30_freqz_whole");
    runner.set_signal_meta("freqz_whole", 8, "Strict");

    runner.record_step(
        "freqz_whole_reference_response",
        "freqz([1, 0.5, 0.25], [1, -0.3, 0.2], worN=8, whole=true)",
        "whole-spectrum response should match SciPy's sampled magnitudes and phases",
        "Strict",
        || {
            let tol = 1e-8;
            let expected_mag = [
                1.944_444_44,
                1.880_828_33,
                1.054_994_64,
                0.511_363_23,
                0.5,
                0.511_363_23,
                1.054_994_64,
                1.880_828_33,
            ];
            let expected_phase = [
                0.0,
                -0.434_838_91,
                -0.946_773_27,
                -0.486_582_96,
                0.0,
                0.486_582_96,
                0.946_773_27,
                0.434_838_91,
            ];

            let resp = freqz_with_whole(&[1.0, 0.5, 0.25], &[1.0, -0.3, 0.2], Some(8), true)
                .map_err(|e| format!("{e}"))?;
            let mag_diff = max_abs_diff(&resp.h_mag, &expected_mag);
            let phase_diff = max_abs_diff(&resp.h_phase, &expected_phase);

            if resp.w.len() != 8 {
                return Err(format!("expected 8 bins, got {}", resp.w.len()));
            }

            if (resp.w[4] - PI).abs() > 1e-12 {
                return Err(format!("expected pi at bin 4, got {}", resp.w[4]));
            }

            if mag_diff < tol && phase_diff < tol {
                Ok(format!(
                    "mag_diff={mag_diff:.2e}, phase_diff={phase_diff:.2e}"
                ))
            } else {
                Err(format!(
                    "SciPy whole freqz mismatch: mag_diff={mag_diff:.2e}, phase_diff={phase_diff:.2e}"
                ))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_30_freqz_whole", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_30 failed");
}

/// Scenario 31: filtfilt padtype matches SciPy's odd/even/constant padding modes.
#[test]
fn scenario_31_filtfilt_padtype_modes() {
    let mut runner = ScenarioRunner::new("scenario_31_filtfilt_padtype_modes");
    runner.set_signal_meta("filtfilt_padtype", 20, "Strict");

    runner.record_step(
        "filtfilt_odd_padtype_reference",
        "filtfilt_with_padtype(b, a, x, Some(\"odd\"))",
        "odd endpoint reflection should match SciPy's default forward-backward reference output",
        "Strict",
        || {
            let x = [
                0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0,
                -1.0, 0.0, 1.0, 2.0, 1.0,
            ];
            let expected = [
                0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28,
                -0.4, -0.28, 0.0, 0.36, 0.72, 1.0,
            ];
            let got = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("odd"))
                .map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-12 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("odd padtype mismatch diff={diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "filtfilt_even_padtype_reference",
        "filtfilt_with_padtype(b, a, x, Some(\"even\"))",
        "even endpoint extension should match SciPy's forward-backward reference output",
        "Strict",
        || {
            let x = [
                0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0,
                -1.0, 0.0, 1.0, 2.0, 1.0,
            ];
            let expected = [
                0.96, 0.92, 0.72, 0.36, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28,
                -0.4, -0.28, 0.08, 0.52, 0.88, 1.0,
            ];
            let got = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("even"))
                .map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-12 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("even padtype mismatch diff={diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "filtfilt_constant_padtype_reference",
        "filtfilt_with_padtype(b, a, x, Some(\"constant\"))",
        "constant endpoint extension should match SciPy's constant padtype output",
        "Strict",
        || {
            let x = [
                0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0,
                -1.0, 0.0, 1.0, 2.0, 1.0,
            ];
            let expected = [
                0.48, 0.6, 0.56, 0.32, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28,
                -0.4, -0.28, 0.04, 0.44, 0.8, 1.0,
            ];
            let got = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("constant"))
                .map_err(|e| format!("{e}"))?;
            let diff = max_abs_diff(&got, &expected);
            if diff < 1e-12 {
                Ok(format!("max_abs_diff={diff:.2e}"))
            } else {
                Err(format!("constant padtype mismatch diff={diff:.2e}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_31_filtfilt_padtype_modes", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_31 failed");
}

/// Scenario 32: lfilter_zi matches SciPy steady-state initialization semantics.
#[test]
fn scenario_32_lfilter_zi() {
    let mut runner = ScenarioRunner::new("scenario_32_lfilter_zi");
    runner.set_signal_meta("lfilter_zi", 12, "Strict");

    runner.record_step(
        "lfilter_zi_reference_vector",
        "lfilter_zi(b, a)",
        "SciPy steady-state vector for a normalized first-order IIR",
        "Strict",
        || {
            let got = lfilter_zi(&[0.5, 0.5], &[1.0, -0.5]).map_err(|e| format!("{e}"))?;
            let expected = [1.5];
            if got.len() != expected.len() {
                return Err(format!("zi length {} != {}", got.len(), expected.len()));
            }
            for (idx, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
                if (actual - want).abs() > 1.0e-12 {
                    return Err(format!("zi[{idx}] mismatch: {actual} vs {want}"));
                }
            }
            Ok(format!("zi={got:?}"))
        },
    );

    runner.record_step(
        "lfilter_zi_step_response",
        "lfilter(b, a, ones, zi=lfilter_zi(...))",
        "steady-state zi should make the unit-step response constant from the first sample",
        "Strict",
        || {
            let b = [0.2, 0.3, 0.5];
            let a = [1.0, -0.1, 0.2];
            let zi = lfilter_zi(&b, &a).map_err(|e| format!("{e}"))?;
            let got = lfilter(&b, &a, &[1.0, 1.0, 1.0], Some(&zi)).map_err(|e| format!("{e}"))?;
            let expected = [0.909_090_909_090_909_2; 3];
            let diff = max_abs_diff(&got, &expected);
            if diff < 1.0e-12 {
                Ok(format!("step response={got:?}"))
            } else {
                Err(format!("step response mismatch diff={diff:.2e}"))
            }
        },
    );

    runner.record_step(
        "lfilter_zi_trims_leading_zero_a",
        "lfilter_zi(b, a=[0, ...])",
        "SciPy trims leading zero denominator coefficients before solving the steady-state system",
        "Strict",
        || {
            let got = lfilter_zi(&[1.0, 2.0], &[0.0, 2.0, -0.6]).map_err(|e| format!("{e}"))?;
            let expected = [1.642_857_142_857_142_8];
            if got.len() != expected.len() {
                return Err(format!("zi length {} != {}", got.len(), expected.len()));
            }
            if (got[0] - expected[0]).abs() > 1.0e-12 {
                return Err(format!(
                    "trimmed zi mismatch: {} vs {}",
                    got[0], expected[0]
                ));
            }
            Ok(format!("trimmed zi={got:?}"))
        },
    );

    runner.record_step(
        "lfilter_zi_scalar_transfer_rejected",
        "lfilter_zi([b0], [a0])",
        "scalar transfer functions should fail with SciPy's degenerate-system error",
        "Strict",
        || match lfilter_zi(&[0.1], &[1.0]) {
            Ok(value) => Err(format!("expected error, got {value:?}")),
            Err(err)
                if err
                    .to_string()
                    .contains("The length of `a` along the last axis must be at least 2.") =>
            {
                Ok(err.to_string())
            }
            Err(err) => Err(format!("unexpected error: {err}")),
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_32_lfilter_zi", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_32 failed");
}
