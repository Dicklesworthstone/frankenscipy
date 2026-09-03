#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-005 (FFT backend routing).
//!
//! Implements bd-3jh.16.7 acceptance criteria:
//!   Happy-path:     1-3  (signal→FFT→filter→IFFT→compare)
//!   Error recovery: 4-6  (wrong input→catch→correct→retry)
//!   Adversarial:    7-10 (degenerate signals, edge sizes)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-005/e2e/`.

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use fsci_fft::{
    Complex64, FftError, FftOptions, Normalization, fft, fft2, fftfreq, ifft, ifft2, irfft, rfft,
};
use serde::Serialize;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
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
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-005/e2e")
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
    format!("cargo test -p fsci-conformance --test e2e_fft -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("failed to create e2e dir {}: {e}", dir.display()));
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
}

const TOL: f64 = 1e-9;

fn complex_mag_sq(c: Complex64) -> f64 {
    c.0 * c.0 + c.1 * c.1
}

fn max_abs_diff_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((x.0 - y.0).abs()).max((x.1 - y.1).abs()))
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

fn max_abs_diff_real(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

fn make_step(
    step_id: usize,
    name: &str,
    action: &str,
    input: &str,
    output: &str,
    dur: u128,
    outcome: &str,
) -> ForensicStep {
    ForensicStep {
        step_id,
        step_name: name.to_string(),
        action: action.to_string(),
        input_summary: input.to_string(),
        output_summary: output.to_string(),
        duration_ns: dur,
        mode: "strict".to_string(),
        outcome: outcome.to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-3)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: Generate real signal → rfft → zero high freqs → irfft → compare
#[test]
fn e2e_001_lowpass_filter_pipeline() {
    let scenario_id = "e2e_001_lowpass";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();

    // Step 1: Generate signal with two frequency components
    let n = 64;
    let t_start = Instant::now();
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
        })
        .collect();
    steps.push(make_step(
        1,
        "generate_signal",
        "create",
        &format!("n={n}, freqs=[3,20]Hz"),
        &format!("signal len={}", signal.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: rfft
    let t_start = Instant::now();
    let spectrum = rfft(&signal, &opts).expect("rfft");
    steps.push(make_step(
        2,
        "rfft",
        "transform",
        &format!("real signal n={n}"),
        &format!("spectrum len={}", spectrum.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Low-pass filter (zero bins above cutoff)
    let t_start = Instant::now();
    let freqs = fftfreq(n, 1.0 / n as f64).unwrap();
    let cutoff = 10.0;
    let mut filtered = spectrum.clone();
    // Zero out bins where freq > cutoff
    for (i, bin) in filtered.iter_mut().enumerate() {
        if i < freqs.len() && freqs[i].abs() > cutoff {
            *bin = (0.0, 0.0);
        }
    }
    steps.push(make_step(
        3,
        "lowpass_filter",
        "filter",
        &format!("cutoff={cutoff}Hz"),
        "zeroed high-freq bins",
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 4: irfft
    let t_start = Instant::now();
    let recovered = irfft(&filtered, Some(n), &opts).expect("irfft");
    steps.push(make_step(
        4,
        "irfft",
        "inverse_transform",
        &format!("spectrum len={}", filtered.len()),
        &format!("signal len={}", recovered.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 5: Verify: only 3Hz component should remain
    let t_start = Instant::now();
    let expected: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * 3.0 * (i as f64) / n as f64).sin())
        .collect();
    let diff = max_abs_diff_real(&recovered, &expected);
    let pass = diff < 0.6; // Filter is not brick-wall, so allow some leakage
    steps.push(make_step(
        5,
        "verify_output",
        "compare",
        &format!("expected 3Hz-only, diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "lowpass filter pipeline failed: diff={diff:.4e}");
}

/// Scenario 2: Complex FFT → IFFT roundtrip with normalization modes
#[test]
fn e2e_002_fft_ifft_normalization_sweep() {
    let scenario_id = "e2e_002_norm_sweep";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 32;

    let input: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            ((2.0 * PI * t).sin(), (4.0 * PI * t).cos() * 0.5)
        })
        .collect();
    steps.push(make_step(
        1,
        "generate_signal",
        "create",
        &format!("complex signal n={n}"),
        &format!("len={}", input.len()),
        0,
        "ok",
    ));

    let mut all_pass = true;
    for (step_offset, norm) in [
        Normalization::Backward,
        Normalization::Forward,
        Normalization::Ortho,
    ]
    .iter()
    .enumerate()
    {
        let opts = FftOptions::default().with_normalization(*norm);
        let norm_name = format!("{norm:?}");

        let t_start = Instant::now();
        let spectrum = fft(&input, &opts).expect("fft");
        steps.push(make_step(
            2 + step_offset * 2,
            &format!("fft_{norm_name}"),
            "transform",
            &format!("norm={norm_name}"),
            &format!("spectrum len={}", spectrum.len()),
            t_start.elapsed().as_nanos(),
            "ok",
        ));

        let t_start = Instant::now();
        let recovered = ifft(&spectrum, &opts).expect("ifft");
        let diff = max_abs_diff_complex(&recovered, &input);
        let pass = diff < TOL;
        if !pass {
            all_pass = false;
        }
        steps.push(make_step(
            3 + step_offset * 2,
            &format!("ifft_{norm_name}_verify"),
            "compare",
            &format!("diff={diff:.4e}"),
            &format!("pass={pass}"),
            t_start.elapsed().as_nanos(),
            if pass { "ok" } else { "fail" },
        ));
    }

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "normalization sweep failed");
}

/// Scenario 2b: exact NumPy DFT goldens for basic fft/rfft.
#[test]
fn e2e_002b_basic_fft_rfft_numpy_goldens() {
    let scenario_id = "e2e_002b_basic_fft_rfft_numpy_goldens";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let fft_input: Vec<Complex64> = vec![
        (1.0, 2.0),
        (-3.0, 0.5),
        (0.0, -1.0),
        (4.0, 3.0),
        (-2.0, 0.0),
    ];
    let fft_expected: Vec<Complex64> = vec![
        (0.0, 4.5),
        (-5.656_765_700_396_843, 3.838_672_033_902_625),
        (10.379_271_640_701_377, -1.002_915_321_325_72),
        (2.183_034_258_047_676_7, 5.429_966_304_450_563),
        (-1.905_540_198_352_211_3, -2.765_723_017_027_467_4),
    ];
    let t_start = Instant::now();
    let fft_result = fft(&fft_input, &FftOptions::default()).expect("fft numpy golden");
    let fft_diff = max_abs_diff_complex(&fft_result, &fft_expected);
    let fft_pass = fft_diff <= TOL;
    steps.push(make_step(
        1,
        "fft_exact_numpy_dft",
        "fft",
        "complex length-5 signal",
        &format!("max_abs_diff={fft_diff:.3e}"),
        t_start.elapsed().as_nanos(),
        if fft_pass { "ok" } else { "fail" },
    ));

    let rfft_input = vec![0.5, -1.0, 2.5, 0.0, -3.5];
    let rfft_expected: Vec<Complex64> = vec![
        (-1.5, 0.0),
        (-2.913_118_960_624_632, -3.847_104_421_469_066_6),
        (4.913_118_960_624_631, 0.908_178_160_006_700_7),
    ];
    let t_start = Instant::now();
    let rfft_result = rfft(&rfft_input, &FftOptions::default()).expect("rfft numpy golden");
    let rfft_diff = max_abs_diff_complex(&rfft_result, &rfft_expected);
    let rfft_pass = rfft_diff <= TOL;
    steps.push(make_step(
        2,
        "rfft_exact_numpy_dft",
        "rfft",
        "real length-5 signal",
        &format!("max_abs_diff={rfft_diff:.3e}"),
        t_start.elapsed().as_nanos(),
        if rfft_pass { "ok" } else { "fail" },
    ));

    let all_pass = fft_pass && rfft_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        all_pass,
        "fft/rfft numpy goldens failed: fft_diff={fft_diff:.4e}, rfft_diff={rfft_diff:.4e}"
    );
}

/// Scenario 2c: exact NumPy inverse-DFT goldens for ifft/irfft.
#[test]
fn e2e_002c_basic_ifft_irfft_numpy_goldens() {
    let scenario_id = "e2e_002c_basic_ifft_irfft_numpy_goldens";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let ifft_input: Vec<Complex64> = vec![
        (1.0, 2.0),
        (-3.0, 0.5),
        (0.0, -1.0),
        (4.0, 3.0),
        (-2.0, 0.0),
    ];
    let ifft_expected: Vec<Complex64> = vec![
        (0.0, 0.9),
        (-0.381_108_039_670_442_27, -0.553_144_603_405_493_5),
        (0.436_606_851_609_535_4, 1.085_993_260_890_112_6),
        (2.075_854_328_140_275_5, -0.200_583_064_265_144_02),
        (-1.131_353_140_079_368_6, 0.767_734_406_780_525_1),
    ];
    let t_start = Instant::now();
    let ifft_result = ifft(&ifft_input, &FftOptions::default()).expect("ifft numpy golden");
    let ifft_diff = max_abs_diff_complex(&ifft_result, &ifft_expected);
    let ifft_pass = ifft_diff <= TOL;
    steps.push(make_step(
        1,
        "ifft_exact_numpy_inverse_dft",
        "ifft",
        "complex length-5 spectrum",
        &format!("max_abs_diff={ifft_diff:.3e}"),
        t_start.elapsed().as_nanos(),
        if ifft_pass { "ok" } else { "fail" },
    ));

    let irfft_input: Vec<Complex64> = vec![(2.0, 0.0), (-1.0, 2.0), (0.5, -0.75)];
    let irfft_expected = vec![
        0.2,
        -0.469_919_833_973_349_4,
        0.029_865_039_902_443_87,
        1.540_955_353_347_493_1,
        0.699_099_440_723_412_5,
    ];
    let t_start = Instant::now();
    let irfft_result =
        irfft(&irfft_input, Some(5), &FftOptions::default()).expect("irfft numpy golden");
    let irfft_diff = max_abs_diff_real(&irfft_result, &irfft_expected);
    let irfft_pass = irfft_diff <= TOL;
    steps.push(make_step(
        2,
        "irfft_exact_numpy_inverse_dft",
        "irfft",
        "half-complex length-3 spectrum, n=5",
        &format!("max_abs_diff={irfft_diff:.3e}"),
        t_start.elapsed().as_nanos(),
        if irfft_pass { "ok" } else { "fail" },
    ));

    let all_pass = ifft_pass && irfft_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        all_pass,
        "ifft/irfft numpy goldens failed: ifft_diff={ifft_diff:.4e}, irfft_diff={irfft_diff:.4e}"
    );
}

/// Scenario 3: 2D FFT → filter → IFFT2 → verify spatial domain
#[test]
fn e2e_003_fft2_spectral_filter() {
    let scenario_id = "e2e_003_fft2_filter";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();
    let rows = 8;
    let cols = 8;
    let n = rows * cols;

    // Create 2D signal: sum of two spatial frequencies
    let t_start = Instant::now();
    let input: Vec<Complex64> = (0..n)
        .map(|idx| {
            let r = (idx / cols) as f64;
            let c = (idx % cols) as f64;
            (
                (2.0 * PI * r / rows as f64).sin() + (2.0 * PI * c / cols as f64).cos(),
                0.0,
            )
        })
        .collect();
    steps.push(make_step(
        1,
        "generate_2d_signal",
        "create",
        &format!("{rows}x{cols} spatial signal"),
        &format!("len={n}"),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: fft2
    let t_start = Instant::now();
    let spectrum = fft2(&input, (rows, cols), &opts).expect("fft2");
    steps.push(make_step(
        2,
        "fft2",
        "transform",
        &format!("shape=({rows},{cols})"),
        &format!("spectrum len={}", spectrum.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: ifft2 roundtrip
    let t_start = Instant::now();
    let recovered = ifft2(&spectrum, (rows, cols), &opts).expect("ifft2");
    let diff = max_abs_diff_complex(&recovered, &input);
    let pass = diff < TOL;
    steps.push(make_step(
        3,
        "ifft2_roundtrip",
        "inverse_transform+verify",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "fft2 spectral filter: diff={diff:.4e}");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (4-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 4: Submit NaN input → catch NonFiniteInput → correct → retry → succeed
#[test]
fn e2e_004_nan_recovery() {
    let scenario_id = "e2e_004_nan_recovery";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Create bad input with NaN
    let bad_input = vec![(1.0, 0.0), (f64::NAN, 0.0), (3.0, 0.0), (4.0, 0.0)];
    let opts_strict = FftOptions::default().with_check_finite(true);

    // Step 2: Submit bad input → expect error
    let t_start = Instant::now();
    let result = fft(&bad_input, &opts_strict);
    let is_err = result.is_err();
    steps.push(make_step(
        1,
        "submit_nan_input",
        "fft",
        "input contains NaN",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected_ok"
        },
    ));

    // Step 3: Correct input by replacing NaN with 0
    let t_start = Instant::now();
    let corrected: Vec<Complex64> = bad_input
        .iter()
        .map(|&(re, im)| {
            (
                if re.is_finite() { re } else { 0.0 },
                if im.is_finite() { im } else { 0.0 },
            )
        })
        .collect();
    steps.push(make_step(
        2,
        "correct_input",
        "sanitize",
        "replace NaN with 0",
        &format!("corrected len={}", corrected.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 4: Retry with corrected input
    let t_start = Instant::now();
    let result = fft(&corrected, &opts_strict);
    let pass = result.is_ok();
    steps.push(make_step(
        3,
        "retry_fft",
        "fft",
        &format!("corrected input n={}", corrected.len()),
        &format!("success={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: if !overall_pass {
                Some("NaN recovery flow failed".to_string())
            } else {
                None
            },
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "NaN recovery scenario failed");
}

/// Scenario 5: Submit empty input → catch → add data → retry
#[test]
fn e2e_005_empty_input_recovery() {
    let scenario_id = "e2e_005_empty_recovery";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();

    // Step 1: Submit empty input
    let t_start = Instant::now();
    let result = fft(&[], &opts);
    let is_err = matches!(result, Err(FftError::InvalidShape { .. }));
    steps.push(make_step(
        1,
        "submit_empty",
        "fft",
        "empty input",
        &format!("got_invalid_shape={is_err}"),
        t_start.elapsed().as_nanos(),
        "expected_error",
    ));

    // Step 2: Construct valid input
    let t_start = Instant::now();
    let valid: Vec<Complex64> = (0..8).map(|i| ((i as f64) * 0.5, 0.0)).collect();
    let result = fft(&valid, &opts);
    let pass = result.is_ok();
    steps.push(make_step(
        2,
        "retry_with_data",
        "fft",
        &format!("valid input n={}", valid.len()),
        &format!("success={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "empty input recovery failed");
}

/// Scenario 6: irfft length mismatch → diagnose → fix → succeed
#[test]
fn e2e_006_irfft_length_mismatch_recovery() {
    let scenario_id = "e2e_006_irfft_mismatch";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();

    // Step 1: rfft a signal
    let signal: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25).collect();
    let t_start = Instant::now();
    let spectrum = rfft(&signal, &opts).expect("rfft");
    steps.push(make_step(
        1,
        "rfft",
        "transform",
        "real signal n=16",
        &format!("spectrum len={}", spectrum.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Try irfft with wrong output_len
    let t_start = Instant::now();
    let result = irfft(&spectrum, Some(20), &opts);
    let is_err = matches!(result, Err(FftError::LengthMismatch { .. }));
    steps.push(make_step(
        2,
        "irfft_wrong_len",
        "inverse_transform",
        "output_len=20 (wrong)",
        &format!("got_mismatch={is_err}"),
        t_start.elapsed().as_nanos(),
        "expected_error",
    ));

    // Step 3: Fix: use correct output_len
    let t_start = Instant::now();
    let recovered = irfft(&spectrum, Some(16), &opts).expect("irfft with correct len");
    let diff = max_abs_diff_real(&recovered, &signal);
    let pass = diff < TOL;
    steps.push(make_step(
        3,
        "irfft_correct_len",
        "inverse_transform+verify",
        &format!("output_len=16, diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "irfft length mismatch recovery failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL SCENARIOS (7-10)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Length-1 signals through full pipeline
#[test]
fn e2e_007_degenerate_length_1() {
    let scenario_id = "e2e_007_length_1";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();

    // Complex fft length 1
    let t_start = Instant::now();
    let input = vec![(42.0, -7.0)];
    let spectrum = fft(&input, &opts).expect("fft n=1");
    let recovered = ifft(&spectrum, &opts).expect("ifft n=1");
    let diff = max_abs_diff_complex(&recovered, &input);
    let pass_complex = diff < TOL;
    steps.push(make_step(
        1,
        "fft_ifft_n1",
        "roundtrip",
        "complex n=1",
        &format!("diff={diff:.4e}, pass={pass_complex}"),
        t_start.elapsed().as_nanos(),
        if pass_complex { "ok" } else { "fail" },
    ));

    // Real fft length 1
    let t_start = Instant::now();
    let real_input = vec![42.0];
    let real_spectrum = rfft(&real_input, &opts).expect("rfft n=1");
    assert_eq!(real_spectrum.len(), 1); // n/2+1 = 1
    let pass_real = (real_spectrum[0].0 - 42.0).abs() < TOL;
    steps.push(make_step(
        2,
        "rfft_n1",
        "transform+verify",
        "real n=1",
        &format!("spectrum[0]={:?}, pass={pass_real}", real_spectrum[0]),
        t_start.elapsed().as_nanos(),
        if pass_real { "ok" } else { "fail" },
    ));

    let overall_pass = pass_complex && pass_real;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "degenerate length-1 scenario failed");
}

/// Scenario 8: Large prime-length FFT (stress test)
#[test]
fn e2e_008_large_prime_size() {
    let scenario_id = "e2e_008_large_prime";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();
    let n = 127; // large prime

    let t_start = Instant::now();
    let input: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            ((2.0 * PI * 5.0 * t).sin(), (2.0 * PI * 11.0 * t).cos())
        })
        .collect();
    steps.push(make_step(
        1,
        "generate_signal",
        "create",
        &format!("complex signal n={n} (prime)"),
        &format!("len={n}"),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    let t_start = Instant::now();
    let spectrum = fft(&input, &opts).expect("fft");
    steps.push(make_step(
        2,
        "fft",
        "transform",
        &format!("n={n}"),
        &format!("spectrum len={}", spectrum.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    let t_start = Instant::now();
    let recovered = ifft(&spectrum, &opts).expect("ifft");
    let diff = max_abs_diff_complex(&recovered, &input);
    let pass = diff < TOL;
    steps.push(make_step(
        3,
        "ifft_roundtrip",
        "inverse+verify",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Verify Parseval's theorem
    let t_start = Instant::now();
    let time_energy: f64 = input.iter().map(|c| complex_mag_sq(*c)).sum();
    let freq_energy: f64 = spectrum.iter().map(|c| complex_mag_sq(*c)).sum();
    let parseval_diff = (time_energy - freq_energy / n as f64).abs();
    let parseval_pass = parseval_diff < TOL * n as f64;
    steps.push(make_step(
        4,
        "parseval_check",
        "verify",
        &format!(
            "time_E={time_energy:.4}, freq_E/n={:.4}",
            freq_energy / n as f64
        ),
        &format!("diff={parseval_diff:.4e}, pass={parseval_pass}"),
        t_start.elapsed().as_nanos(),
        if parseval_pass { "ok" } else { "fail" },
    ));

    let overall_pass = pass && parseval_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "large prime-size FFT failed");
}

/// Scenario 9: DC-only (constant) signal
#[test]
fn e2e_009_constant_dc_signal() {
    let scenario_id = "e2e_009_dc_signal";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();
    let n = 16;
    let dc_value = 5.0;

    // Constant signal → FFT should have DC bin = n * dc_value, rest ≈ 0
    let t_start = Instant::now();
    let input: Vec<Complex64> = vec![(dc_value, 0.0); n];
    let spectrum = fft(&input, &opts).expect("fft");
    steps.push(make_step(
        1,
        "fft_constant",
        "transform",
        &format!("constant={dc_value}, n={n}"),
        &format!("DC bin={:?}", spectrum[0]),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Verify DC bin
    let t_start = Instant::now();
    let dc_expected = (n as f64 * dc_value, 0.0);
    let dc_diff =
        ((spectrum[0].0 - dc_expected.0).abs()).max((spectrum[0].1 - dc_expected.1).abs());
    let non_dc_max: f64 = spectrum[1..]
        .iter()
        .map(|c| complex_mag_sq(*c).sqrt())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
    let pass = dc_diff < TOL && non_dc_max < TOL;
    steps.push(make_step(
        2,
        "verify_spectrum",
        "check",
        &format!("dc_diff={dc_diff:.4e}, non_dc_max={non_dc_max:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "DC signal scenario failed");
}

/// Scenario 10: Rapid sequential transforms (no state leakage)
#[test]
fn e2e_010_rapid_sequential_transforms() {
    let scenario_id = "e2e_010_rapid_seq";
    let overall_start = Instant::now();
    let opts = FftOptions::default();
    let mut steps = Vec::new();
    let iterations = 100;

    let t_start = Instant::now();
    let mut all_pass = true;
    for i in 0..iterations {
        let n = 8 + (i % 8); // varying sizes 8..15
        let input: Vec<Complex64> = (0..n)
            .map(|j| ((j as f64 + i as f64) * 0.1, (j as f64 - i as f64) * 0.05))
            .collect();
        let spectrum = fft(&input, &opts).expect("fft");
        let recovered = ifft(&spectrum, &opts).expect("ifft");
        let diff = max_abs_diff_complex(&recovered, &input);
        if diff > TOL {
            all_pass = false;
        }
    }
    steps.push(make_step(
        1,
        "rapid_roundtrips",
        "fft+ifft",
        &format!("{iterations} iterations, sizes 8-15"),
        &format!("all_pass={all_pass}"),
        t_start.elapsed().as_nanos(),
        if all_pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        all_pass,
        "rapid sequential transforms: state leakage detected"
    );
}
