#![forbid(unsafe_code)]
//! Differential oracle + metamorphic + adversarial tests for FSCI-P2C-005 (FFT).
//!
//! Implements bd-3jh.16.6 acceptance criteria:
//!   Differential: >=15 cases comparing Rust FFT vs naive DFT reference
//!   Metamorphic:  >=6 relation tests
//!   Adversarial:  >=8 edge-case / hostile-input tests
//!
//! All tests emit structured JSON logs to
//! `fixtures/artifacts/FSCI-P2C-005/diff/`.
//!
//! Each test keeps a compared-case ledger: the reference side is the naive DFT, a closed-form
//! formula or the input itself (roundtrips), or another fsci output for the metamorphic
//! relations; the adversarial refusal tests record their expected-error verdict as a one-case
//! arm. The case id is the test's log id.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_fft::{
    Complex64, FftError, FftOptions, Normalization, fft, fft2, fftfreq, fftn, fftshift_1d, ifft,
    ifft2, ifftshift_1d, irfft, rfft, rfftfreq,
};
use serde::Serialize;

// ───────────────────── Structured log types ─────────────────────

#[derive(Debug, Clone, Serialize)]
struct DiffTestLog {
    test_id: String,
    category: String,
    input_summary: String,
    expected: String,
    actual: String,
    diff: f64,
    tolerance: f64,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
}

// ───────────────────── Helpers ─────────────────────

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-005/diff")
}

fn ensure_output_dir() {
    let dir = output_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).expect("create diff output dir");
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffTestLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

const TOL: f64 = 1e-9;

fn complex_mag_sq(c: Complex64) -> f64 {
    c.0 * c.0 + c.1 * c.1
}

fn max_abs_diff_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| nan_max((x.0 - y.0).abs(), (x.1 - y.1).abs()))
        .fold(0.0_f64, nan_max)
}

fn max_abs_diff_real(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, nan_max)
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

/// Sends a real reference and fsci's real output through [`CompareLedger::slices`], so the
/// ledger itself records a length mismatch or a non-finite element that does not match the
/// reference's. True when the case can be compared; the caller then records its verdict with
/// `ledger.compared`.
fn real_slices(
    ledger: &mut CompareLedger,
    arm: &str,
    case_id: &str,
    reference: &[f64],
    observed: &[f64],
) -> bool {
    ledger
        .slices(arm, case_id, Some(reference), Some(observed))
        .is_some()
}

/// [`real_slices`] for complex vectors, as interleaved re/im parts.
fn complex_slices(
    ledger: &mut CompareLedger,
    arm: &str,
    case_id: &str,
    reference: &[Complex64],
    observed: &[Complex64],
) -> bool {
    let re_im = |v: &[Complex64]| -> Vec<f64> { v.iter().flat_map(|&(re, im)| [re, im]).collect() };
    real_slices(ledger, arm, case_id, &re_im(reference), &re_im(observed))
}

/// Naive DFT reference implementation for oracle comparison.
fn naive_dft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut output = vec![(0.0, 0.0); n];
    for (k, out) in output.iter_mut().enumerate() {
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &(vr, vi)) in input.iter().enumerate() {
            let angle = sign * 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            let (s, c) = angle.sin_cos();
            re += vr * c - vi * s;
            im += vr * s + vi * c;
        }
        if inverse {
            re /= n as f64;
            im /= n as f64;
        }
        *out = (re, im);
    }
    output
}

/// Reference real FFT: compute full DFT and take first n/2+1 bins.
fn naive_rfft(input: &[f64]) -> Vec<Complex64> {
    let n = input.len();
    let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
    let full = naive_dft(&complex_input, false);
    full.into_iter().take(n / 2 + 1).collect()
}

/// Make a deterministic test signal: sum of harmonics.
fn test_signal_real(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * t).sin() + 0.5 * (6.0 * PI * t).cos()
        })
        .collect()
}

fn test_signal_complex(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (
                (2.0 * PI * t).sin() + 0.5 * (4.0 * PI * t).cos(),
                0.3 * (6.0 * PI * t).sin(),
            )
        })
        .collect()
}

/// Writes the test's log, with `ledger`'s compared counts, and asserts `diff <= tolerance`. The
/// caller has already recorded the case in the ledger: its `pair`/`slices` guard, then
/// `ledger.compared` with this same verdict when the guard accepted both sides.
fn run_diff_test(
    ledger: &CompareLedger,
    test_id: &str,
    category: &str,
    input_summary: &str,
    diff: f64,
    tolerance: f64,
) {
    let pass = diff <= tolerance;
    let start = Instant::now();
    let log = DiffTestLog {
        test_id: test_id.to_string(),
        category: category.to_string(),
        input_summary: input_summary.to_string(),
        expected: format!("diff <= {tolerance}"),
        actual: format!("diff = {diff:.2e}"),
        diff,
        tolerance,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    };
    emit_log(&log);
    assert!(pass, "{test_id}: diff {diff:.2e} > tol {tolerance:.2e}");
}

// ═══════════════════════════════════════════════════════════════════
// DIFFERENTIAL TESTS (diff_001 – diff_016)
// Compare Rust FFT output against naive DFT oracle
// ═══════════════════════════════════════════════════════════════════

#[test]
fn diff_001_fft_power_of_2() {
    let input = test_signal_complex(8);
    let opts = FftOptions::default();
    let result = fft(&input, &opts).unwrap();
    let expected = naive_dft(&input, false);
    let (arm, case) = ("fft_pow2", "diff_001_fft_pow2");
    let mut ledger = CompareLedger::new("diff_001_fft_power_of_2", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "complex signal n=8",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_002_fft_non_power_of_2() {
    let input = test_signal_complex(13);
    let opts = FftOptions::default();
    let result = fft(&input, &opts).unwrap();
    let expected = naive_dft(&input, false);
    let (arm, case) = ("fft_npow2", "diff_002_fft_npow2");
    let mut ledger = CompareLedger::new("diff_002_fft_non_power_of_2", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "complex signal n=13",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_003_fft_prime_size() {
    let input = test_signal_complex(23);
    let opts = FftOptions::default();
    let result = fft(&input, &opts).unwrap();
    let expected = naive_dft(&input, false);
    let (arm, case) = ("fft_prime", "diff_003_fft_prime");
    let mut ledger = CompareLedger::new("diff_003_fft_prime_size", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "complex signal n=23 (prime)",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_004_ifft_roundtrip() {
    let input = test_signal_complex(16);
    let opts = FftOptions::default();
    let spectrum = fft(&input, &opts).unwrap();
    let recovered = ifft(&spectrum, &opts).unwrap();
    let (arm, case) = ("ifft_roundtrip", "diff_004_ifft_roundtrip");
    let mut ledger = CompareLedger::new("diff_004_ifft_roundtrip", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &input, &recovered);
    let diff = max_abs_diff_complex(&recovered, &input);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "ifft(fft(x)) n=16",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_005_rfft_vs_oracle() {
    let input = test_signal_real(16);
    let opts = FftOptions::default();
    let result = rfft(&input, &opts).unwrap();
    let expected = naive_rfft(&input);
    let (arm, case) = ("rfft_oracle", "diff_005_rfft_oracle");
    let mut ledger = CompareLedger::new("diff_005_rfft_vs_oracle", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "real signal n=16 rfft",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_006_rfft_odd_length() {
    let input = test_signal_real(11);
    let opts = FftOptions::default();
    let result = rfft(&input, &opts).unwrap();
    let expected = naive_rfft(&input);
    let (arm, case) = ("rfft_odd", "diff_006_rfft_odd");
    let mut ledger = CompareLedger::new("diff_006_rfft_odd_length", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "real signal n=11 rfft",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_007_irfft_roundtrip() {
    let input = test_signal_real(16);
    let opts = FftOptions::default();
    let spectrum = rfft(&input, &opts).unwrap();
    let recovered = irfft(&spectrum, Some(16), &opts).unwrap();
    let (arm, case) = ("irfft_roundtrip", "diff_007_irfft_roundtrip");
    let mut ledger = CompareLedger::new("diff_007_irfft_roundtrip", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &input, &recovered);
    let diff = max_abs_diff_real(&recovered, &input);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "irfft(rfft(x)) n=16",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_008_fft2_vs_oracle() {
    let rows = 4;
    let cols = 4;
    let input: Vec<Complex64> = (0..rows * cols)
        .map(|i| {
            (
                ((i * 7 + 3) % 13) as f64 - 6.0,
                ((i * 5 + 1) % 11) as f64 - 5.0,
            )
        })
        .collect();
    let opts = FftOptions::default();
    let result = fft2(&input, (rows, cols), &opts).unwrap();

    // Reference: row-wise then column-wise DFT
    let mut expected = input.clone();
    // Row-wise DFT
    for r in 0..rows {
        let row: Vec<Complex64> = (0..cols).map(|c| expected[r * cols + c]).collect();
        let ft = naive_dft(&row, false);
        for c in 0..cols {
            expected[r * cols + c] = ft[c];
        }
    }
    // Column-wise DFT
    for c in 0..cols {
        let col: Vec<Complex64> = (0..rows).map(|r| expected[r * cols + c]).collect();
        let ft = naive_dft(&col, false);
        for r in 0..rows {
            expected[r * cols + c] = ft[r];
        }
    }
    let (arm, case) = ("fft2_oracle", "diff_008_fft2_oracle");
    let mut ledger = CompareLedger::new("diff_008_fft2_vs_oracle", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(&ledger, case, "differential", "4x4 complex fft2", diff, TOL);
    ledger.finish(1);
}

#[test]
fn diff_009_fft2_non_square() {
    let rows = 3;
    let cols = 5;
    let input: Vec<Complex64> = (0..rows * cols)
        .map(|i| ((i as f64) * 0.7 - 3.0, (i as f64) * 0.3))
        .collect();
    let opts = FftOptions::default();
    let result = fft2(&input, (rows, cols), &opts).unwrap();

    let mut expected = input.clone();
    for r in 0..rows {
        let row: Vec<Complex64> = (0..cols).map(|c| expected[r * cols + c]).collect();
        let ft = naive_dft(&row, false);
        for c in 0..cols {
            expected[r * cols + c] = ft[c];
        }
    }
    for c in 0..cols {
        let col: Vec<Complex64> = (0..rows).map(|r| expected[r * cols + c]).collect();
        let ft = naive_dft(&col, false);
        for r in 0..rows {
            expected[r * cols + c] = ft[r];
        }
    }
    let (arm, case) = ("fft2_nonsq", "diff_009_fft2_nonsq");
    let mut ledger = CompareLedger::new("diff_009_fft2_non_square", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(&ledger, case, "differential", "3x5 complex fft2", diff, TOL);
    ledger.finish(1);
}

#[test]
fn diff_010_ifft2_roundtrip() {
    let rows = 4;
    let cols = 6;
    let input = test_signal_complex(rows * cols);
    let opts = FftOptions::default();
    let spectrum = fft2(&input, (rows, cols), &opts).unwrap();
    let recovered = ifft2(&spectrum, (rows, cols), &opts).unwrap();
    let (arm, case) = ("ifft2_roundtrip", "diff_010_ifft2_roundtrip");
    let mut ledger = CompareLedger::new("diff_010_ifft2_roundtrip", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &input, &recovered);
    let diff = max_abs_diff_complex(&recovered, &input);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "ifft2(fft2(x)) 4x6",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_011_fftn_3d_oracle() {
    let shape = [2, 3, 2];
    let n = shape.iter().product::<usize>();
    let input: Vec<Complex64> = (0..n)
        .map(|i| ((i as f64) - 5.0, (i as f64) * 0.5 - 2.5))
        .collect();
    let opts = FftOptions::default();
    let result = fftn(&input, &shape, &opts).unwrap();

    // Reference: transform along each axis sequentially
    let mut expected = input.clone();
    // Axis 0: length 2, stride=6, repeats=1
    for offset in 0..6 {
        let v: Vec<Complex64> = (0..2).map(|i| expected[i * 6 + offset]).collect();
        let ft = naive_dft(&v, false);
        for (i, &val) in ft.iter().enumerate() {
            expected[i * 6 + offset] = val;
        }
    }
    // Axis 1: length 3, stride=2, repeats=2
    for outer in 0..2 {
        for offset in 0..2 {
            let v: Vec<Complex64> = (0..3)
                .map(|i| expected[outer * 6 + i * 2 + offset])
                .collect();
            let ft = naive_dft(&v, false);
            for (i, &val) in ft.iter().enumerate() {
                expected[outer * 6 + i * 2 + offset] = val;
            }
        }
    }
    // Axis 2: length 2, stride=1, repeats=6
    for outer in 0..6 {
        let v: Vec<Complex64> = (0..2).map(|i| expected[outer * 2 + i]).collect();
        let ft = naive_dft(&v, false);
        for (i, &val) in ft.iter().enumerate() {
            expected[outer * 2 + i] = val;
        }
    }
    let (arm, case) = ("fftn_3d", "diff_011_fftn_3d");
    let mut ledger = CompareLedger::new("diff_011_fftn_3d_oracle", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "2x3x2 complex fftn",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn diff_012_normalization_forward_vs_oracle() {
    let input = test_signal_complex(8);
    let opts = FftOptions::default().with_normalization(Normalization::Forward);
    let result = fft(&input, &opts).unwrap();
    // Forward normalization: divide by n
    let n = input.len() as f64;
    let expected: Vec<Complex64> = naive_dft(&input, false)
        .iter()
        .map(|&(re, im)| (re / n, im / n))
        .collect();
    let (arm, case) = ("norm_forward", "diff_012_norm_forward");
    let mut ledger = CompareLedger::new("diff_012_normalization_forward_vs_oracle", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(&ledger, case, "differential", "forward norm n=8", diff, TOL);
    ledger.finish(1);
}

#[test]
fn diff_013_normalization_ortho_vs_oracle() {
    let input = test_signal_complex(8);
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    let result = fft(&input, &opts).unwrap();
    let n = input.len() as f64;
    let scale = 1.0 / n.sqrt();
    let expected: Vec<Complex64> = naive_dft(&input, false)
        .iter()
        .map(|&(re, im)| (re * scale, im * scale))
        .collect();
    let (arm, case) = ("norm_ortho", "diff_013_norm_ortho");
    let mut ledger = CompareLedger::new("diff_013_normalization_ortho_vs_oracle", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(&ledger, case, "differential", "ortho norm n=8", diff, TOL);
    ledger.finish(1);
}

#[test]
fn diff_014_fftfreq_even_vs_oracle() {
    let n = 8;
    let d = 0.5;
    let result = fftfreq(n, d).unwrap();
    // Oracle: k/(n*d) for k=0..n/2-1, then (k-n)/(n*d) for k=n/2..n-1
    let expected: Vec<f64> = (0..n)
        .map(|k| {
            if k < n / 2 {
                k as f64 / (n as f64 * d)
            } else {
                (k as f64 - n as f64) / (n as f64 * d)
            }
        })
        .collect();
    let (arm, case) = ("fftfreq_even", "diff_014_fftfreq_even");
    let mut ledger = CompareLedger::new("diff_014_fftfreq_even_vs_oracle", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_real(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= 1e-15);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "fftfreq n=8 d=0.5",
        diff,
        1e-15,
    );
    ledger.finish(1);
}

#[test]
fn diff_015_rfftfreq_vs_oracle() {
    let n = 10;
    let d = 0.25;
    let result = rfftfreq(n, d).unwrap();
    let expected: Vec<f64> = (0..=n / 2).map(|k| k as f64 / (n as f64 * d)).collect();
    let (arm, case) = ("rfftfreq", "diff_015_rfftfreq");
    let mut ledger = CompareLedger::new("diff_015_rfftfreq_vs_oracle", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_real(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= 1e-15);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "rfftfreq n=10 d=0.25",
        diff,
        1e-15,
    );
    ledger.finish(1);
}

#[test]
fn diff_016_fftfreq_odd_vs_oracle() {
    let n = 7;
    let d = 1.0;
    let result = fftfreq(n, d).unwrap();
    let split = n.div_ceil(2);
    let expected: Vec<f64> = (0..n)
        .map(|k| {
            if k < split {
                k as f64 / (n as f64 * d)
            } else {
                (k as f64 - n as f64) / (n as f64 * d)
            }
        })
        .collect();
    let (arm, case) = ("fftfreq_odd", "diff_016_fftfreq_odd");
    let mut ledger = CompareLedger::new("diff_016_fftfreq_odd_vs_oracle", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_real(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= 1e-15);
    }
    run_diff_test(
        &ledger,
        case,
        "differential",
        "fftfreq n=7 d=1.0",
        diff,
        1e-15,
    );
    ledger.finish(1);
}

// ═══════════════════════════════════════════════════════════════════
// METAMORPHIC TESTS (meta_001 – meta_007)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn meta_001_parseval_energy_conservation() {
    let input = test_signal_complex(16);
    let opts = FftOptions::default();
    let spectrum = fft(&input, &opts).unwrap();
    let time_energy: f64 = input.iter().map(|c| complex_mag_sq(*c)).sum();
    let freq_energy: f64 = spectrum.iter().map(|c| complex_mag_sq(*c)).sum();
    let n = input.len() as f64;
    // Reference: the input's energy; fsci side: the spectrum's energy over n.
    let (arm, case) = ("parseval", "meta_001_parseval");
    let mut ledger = CompareLedger::new("meta_001_parseval_energy_conservation", &[arm]);
    let accepted = ledger
        .pair(arm, case, Some(time_energy), Some(freq_energy / n))
        .is_some();
    let diff = (time_energy - freq_energy / n).abs();
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "Parseval energy n=16",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn meta_002_linearity() {
    let n = 12;
    let a: Vec<Complex64> = (0..n)
        .map(|i| ((i as f64) * 0.5, -(i as f64) * 0.3))
        .collect();
    let b: Vec<Complex64> = (0..n)
        .map(|i| ((i as f64) * 0.2 - 1.0, (i as f64) * 0.1))
        .collect();
    let alpha = 2.5;
    let beta = -1.3;

    let opts = FftOptions::default();
    let fa = fft(&a, &opts).unwrap();
    let fb = fft(&b, &opts).unwrap();

    // alpha*a + beta*b
    let combined: Vec<Complex64> = a
        .iter()
        .zip(b.iter())
        .map(|(&(ar, ai), &(br, bi))| (alpha * ar + beta * br, alpha * ai + beta * bi))
        .collect();
    let fc = fft(&combined, &opts).unwrap();

    // alpha*F(a) + beta*F(b)
    let expected: Vec<Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&(ar, ai), &(br, bi))| (alpha * ar + beta * br, alpha * ai + beta * bi))
        .collect();

    // Reference: alpha*F(a) + beta*F(b) from fsci's own transforms; fsci side: F(alpha*a+beta*b).
    let (arm, case) = ("linearity", "meta_002_linearity");
    let mut ledger = CompareLedger::new("meta_002_linearity", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &fc);
    let diff = max_abs_diff_complex(&fc, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "F(a*x+b*y) = a*F(x)+b*F(y)",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn meta_003_circular_shift_magnitude_preservation() {
    let n = 10;
    let input = test_signal_complex(n);
    let opts = FftOptions::default();
    let spectrum = fft(&input, &opts).unwrap();
    let magnitudes_orig: Vec<f64> = spectrum.iter().map(|c| complex_mag_sq(*c).sqrt()).collect();

    // Circular shift by 3
    let mut shifted = vec![(0.0, 0.0); n];
    for i in 0..n {
        shifted[(i + 3) % n] = input[i];
    }
    let shifted_spectrum = fft(&shifted, &opts).unwrap();
    let magnitudes_shift: Vec<f64> = shifted_spectrum
        .iter()
        .map(|c| complex_mag_sq(*c).sqrt())
        .collect();

    // Reference: |F(x)|; fsci side: |F(shift(x))|.
    let (arm, case) = ("shift_mag", "meta_003_shift_mag");
    let mut ledger = CompareLedger::new("meta_003_circular_shift_magnitude_preservation", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &magnitudes_orig, &magnitudes_shift);
    let diff = max_abs_diff_real(&magnitudes_orig, &magnitudes_shift);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "|F(shift(x))| = |F(x)|",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn meta_004_conjugate_symmetry_real_input() {
    let input: Vec<Complex64> = test_signal_real(16).iter().map(|&x| (x, 0.0)).collect();
    let opts = FftOptions::default();
    let spectrum = fft(&input, &opts).unwrap();
    let n = spectrum.len();

    // Reference: conj(X[n-k]); fsci side: X[k]; over the k the check below covers.
    let mirrored_conj: Vec<Complex64> = (1..n / 2)
        .map(|k| (spectrum[n - k].0, -spectrum[n - k].1))
        .collect();
    let (arm, case) = ("conj_sym", "meta_004_conj_sym");
    let mut ledger = CompareLedger::new("meta_004_conjugate_symmetry_real_input", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &mirrored_conj, &spectrum[1..n / 2]);

    // For real input, X[k] = conj(X[n-k])
    let mut max_diff = 0.0_f64;
    for k in 1..n / 2 {
        let xk = spectrum[k];
        let xnk = spectrum[n - k];
        let d = nan_max((xk.0 - xnk.0).abs(), (xk.1 + xnk.1).abs());
        max_diff = nan_max(max_diff, d);
    }
    if accepted {
        ledger.compared(arm, case, max_diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "X[k]=conj(X[n-k]) for real input",
        max_diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn meta_005_fftshift_ifftshift_roundtrip() {
    let input = test_signal_real(15);
    let shifted = fftshift_1d(&input);
    let recovered = ifftshift_1d(&shifted);
    let (arm, case) = ("shift_roundtrip", "meta_005_shift_roundtrip");
    let mut ledger = CompareLedger::new("meta_005_fftshift_ifftshift_roundtrip", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &input, &recovered);
    let diff = max_abs_diff_real(&recovered, &input);
    if accepted {
        ledger.compared(arm, case, diff <= 0.0);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "ifftshift(fftshift(x)) = x",
        diff,
        0.0,
    );
    ledger.finish(1);
}

#[test]
fn meta_006_ortho_unitary_preservation() {
    // For ortho normalization: ifft(fft(x)) == x and ||fft(x)|| == ||x||
    let input = test_signal_complex(8);
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    let spectrum = fft(&input, &opts).unwrap();
    let recovered = ifft(&spectrum, &opts).unwrap();

    let input_energy: f64 = input.iter().map(|c| complex_mag_sq(*c)).sum();
    let spectrum_energy: f64 = spectrum.iter().map(|c| complex_mag_sq(*c)).sum();

    // One case, two guards: the roundtrip against x, then the spectrum's energy against x's.
    // `&&` consults the pair only once the slices accepted, so the case records one outcome.
    let (arm, case) = ("ortho_unitary", "meta_006_ortho_unitary");
    let mut ledger = CompareLedger::new("meta_006_ortho_unitary_preservation", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &input, &recovered)
        && ledger
            .pair(arm, case, Some(input_energy), Some(spectrum_energy))
            .is_some();

    let diff_roundtrip = max_abs_diff_complex(&recovered, &input);
    let energy_diff = (input_energy - spectrum_energy).abs();

    let diff = nan_max(diff_roundtrip, energy_diff);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "metamorphic",
        "ortho preserves energy + roundtrip",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn meta_007_rfft_output_length_invariant() {
    // rfft always returns n/2+1 bins
    // Structural: the reference is the bin count n/2+1, one case per n.
    let arm = "rfft_len";
    let mut ledger = CompareLedger::new("meta_007_rfft_output_length_invariant", &[arm]);
    let sizes = [4, 5, 7, 8, 15, 16, 31, 32];
    for n in sizes {
        let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let opts = FftOptions::default();
        let result = rfft(&input, &opts).unwrap();
        ledger.compared(arm, &format!("n={n}"), result.len() == n / 2 + 1);
        assert_eq!(
            result.len(),
            n / 2 + 1,
            "rfft(n={n}) should return {expected} bins, got {actual}",
            expected = n / 2 + 1,
            actual = result.len()
        );
    }
    run_diff_test(
        &ledger,
        "meta_007_rfft_len",
        "metamorphic",
        "rfft len = n/2+1 for various n",
        0.0,
        0.0,
    );
    ledger.finish(sizes.len());
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL TESTS (adv_001 – adv_010)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn adv_001_size_1_fft() {
    let input = vec![(42.0, -7.0)];
    let opts = FftOptions::default();
    let result = fft(&input, &opts).unwrap();
    let (arm, case) = ("size1_identity", "adv_001_size1");
    let mut ledger = CompareLedger::new("adv_001_size_1_fft", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &input, &result);
    let diff = max_abs_diff_complex(&result, &input);
    if accepted {
        ledger.compared(arm, case, diff <= TOL);
    }
    run_diff_test(
        &ledger,
        case,
        "adversarial",
        "fft of length-1 is identity",
        diff,
        TOL,
    );
    ledger.finish(1);
}

#[test]
fn adv_002_empty_input_rejected() {
    let opts = FftOptions::default();
    let result = fft(&[], &opts);
    let pass = matches!(result, Err(FftError::InvalidShape { .. }));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger = CompareLedger::new("adv_002_empty_input_rejected", &["empty_rejected"]);
    ledger.compared("empty_rejected", "adv_002_empty", pass);
    let log = DiffTestLog {
        test_id: "adv_002_empty".to_string(),
        category: "adversarial".to_string(),
        input_summary: "empty input".to_string(),
        expected: "InvalidShape error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "empty input should return InvalidShape");
    ledger.finish(1);
}

#[test]
fn adv_003_nan_rejected_by_check_finite() {
    let opts = FftOptions::default().with_check_finite(true);
    let input = vec![(1.0, f64::NAN), (2.0, 0.0)];
    let result = fft(&input, &opts);
    let pass = matches!(result, Err(FftError::NonFiniteInput));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger = CompareLedger::new("adv_003_nan_rejected_by_check_finite", &["nan_rejected"]);
    ledger.compared("nan_rejected", "adv_003_nan", pass);
    let log = DiffTestLog {
        test_id: "adv_003_nan".to_string(),
        category: "adversarial".to_string(),
        input_summary: "NaN in complex input".to_string(),
        expected: "NonFiniteInput error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "NaN should be rejected when check_finite=true");
    ledger.finish(1);
}

#[test]
fn adv_004_inf_rejected_rfft() {
    let opts = FftOptions::default().with_check_finite(true);
    let input = vec![1.0, f64::INFINITY, 3.0];
    let result = rfft(&input, &opts);
    let pass = matches!(result, Err(FftError::NonFiniteInput));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger = CompareLedger::new("adv_004_inf_rejected_rfft", &["inf_rfft_rejected"]);
    ledger.compared("inf_rfft_rejected", "adv_004_inf_rfft", pass);
    let log = DiffTestLog {
        test_id: "adv_004_inf_rfft".to_string(),
        category: "adversarial".to_string(),
        input_summary: "Inf in real rfft input".to_string(),
        expected: "NonFiniteInput error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "Inf should be rejected when check_finite=true");
    ledger.finish(1);
}

#[test]
fn adv_005_zero_workers_rejected() {
    use fsci_fft::WorkerPolicy;
    let opts = FftOptions::default().with_workers(WorkerPolicy::Exact(0));
    let input = vec![(1.0, 0.0)];
    let result = fft(&input, &opts);
    let pass = matches!(result, Err(FftError::InvalidWorkers { .. }));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger =
        CompareLedger::new("adv_005_zero_workers_rejected", &["zero_workers_rejected"]);
    ledger.compared("zero_workers_rejected", "adv_005_zero_workers", pass);
    let log = DiffTestLog {
        test_id: "adv_005_zero_workers".to_string(),
        category: "adversarial".to_string(),
        input_summary: "WorkerPolicy::Exact(0)".to_string(),
        expected: "InvalidWorkers error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "zero workers should be rejected");
    ledger.finish(1);
}

#[test]
fn adv_006_irfft_length_mismatch() {
    let opts = FftOptions::default();
    // For output_len=8, expects input of len 5 (8/2+1), but give 3
    let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
    let result = irfft(&input, Some(8), &opts);
    let pass = matches!(result, Err(FftError::LengthMismatch { .. }));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger = CompareLedger::new(
        "adv_006_irfft_length_mismatch",
        &["irfft_mismatch_rejected"],
    );
    ledger.compared("irfft_mismatch_rejected", "adv_006_irfft_mismatch", pass);
    let log = DiffTestLog {
        test_id: "adv_006_irfft_mismatch".to_string(),
        category: "adversarial".to_string(),
        input_summary: "irfft input len=3, output_len=8".to_string(),
        expected: "LengthMismatch error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "irfft length mismatch should be caught");
    ledger.finish(1);
}

#[test]
fn adv_007_fft2_shape_mismatch() {
    let opts = FftOptions::default();
    let input = vec![(1.0, 0.0); 10]; // 10 elements
    let result = fft2(&input, (3, 4), &opts); // expects 12
    let pass = matches!(result, Err(FftError::LengthMismatch { .. }));
    // Structural: the expected outcome is an error variant, not a value.
    let mut ledger = CompareLedger::new("adv_007_fft2_shape_mismatch", &["fft2_mismatch_rejected"]);
    ledger.compared("fft2_mismatch_rejected", "adv_007_fft2_mismatch", pass);
    let log = DiffTestLog {
        test_id: "adv_007_fft2_mismatch".to_string(),
        category: "adversarial".to_string(),
        input_summary: "fft2 input len=10, shape=3x4 (expects 12)".to_string(),
        expected: "LengthMismatch error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "fft2 shape mismatch should be caught");
    ledger.finish(1);
}

#[test]
fn adv_008_fftfreq_zero_n_rejected() {
    let result = fftfreq(0, 1.0);
    let pass = result.is_err();
    // Structural: the expected outcome is an error, not a value.
    let mut ledger = CompareLedger::new(
        "adv_008_fftfreq_zero_n_rejected",
        &["fftfreq_zero_rejected"],
    );
    ledger.compared("fftfreq_zero_rejected", "adv_008_fftfreq_zero", pass);
    let log = DiffTestLog {
        test_id: "adv_008_fftfreq_zero".to_string(),
        category: "adversarial".to_string(),
        input_summary: "fftfreq(n=0, d=1.0)".to_string(),
        expected: "error".to_string(),
        actual: format!("{result:?}"),
        diff: 0.0,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "fftfreq(0, _) should fail");
    ledger.finish(1);
}

#[test]
fn adv_009_fftfreq_negative_spacing_matches_scipy() {
    let n = 8;
    let d = -1.0;
    let result = fftfreq(n, d).unwrap();
    let expected = vec![-0.0, -0.125, -0.25, -0.375, 0.5, 0.375, 0.25, 0.125];
    let (arm, case) = ("fftfreq_neg_d", "adv_009_fftfreq_neg_d");
    let mut ledger = CompareLedger::new("adv_009_fftfreq_negative_spacing_matches_scipy", &[arm]);
    let accepted = real_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_real(&result, &expected);
    let pass = diff <= 1e-15;
    if accepted {
        ledger.compared(arm, case, pass);
    }
    let log = DiffTestLog {
        test_id: "adv_009_fftfreq_neg_d".to_string(),
        category: "differential".to_string(),
        input_summary: "fftfreq(n=8, d=-1.0)".to_string(),
        expected: format!("{expected:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: 1e-15,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(pass, "fftfreq negative spacing should match scipy");
    ledger.finish(1);
}

#[test]
fn adv_009b_fftfreq_non_finite_spacing_matches_scipy() {
    // Two arms, one case each: fftfreq(4, inf) against its signed zeros, and rfftfreq(4, nan)
    // against its n/2+1 NaN bins (slices matches a NaN reference element only with NaN, so an
    // empty or short output no longer passes the `all(is_nan)` check vacuously).
    let (inf_arm, nan_arm, case) = (
        "fftfreq_inf_d",
        "rfftfreq_nan_d",
        "adv_009b_fftfreq_nonfinite_d",
    );
    let mut ledger = CompareLedger::new(
        "adv_009b_fftfreq_non_finite_spacing_matches_scipy",
        &[inf_arm, nan_arm],
    );
    let inf_freqs = fftfreq(4, f64::INFINITY).unwrap();
    let inf_expected = vec![0.0, 0.0, -0.0, -0.0];
    let inf_accepted = real_slices(&mut ledger, inf_arm, case, &inf_expected, &inf_freqs);
    let inf_diff = max_abs_diff_real(&inf_freqs, &inf_expected);
    if inf_accepted {
        ledger.compared(inf_arm, case, inf_diff <= 0.0);
    }

    let nan_freqs = rfftfreq(4, f64::NAN).unwrap();
    let nan_expected = vec![f64::NAN; 4 / 2 + 1];
    let nan_accepted = real_slices(&mut ledger, nan_arm, case, &nan_expected, &nan_freqs);
    let nan_pass = nan_freqs.iter().all(|value| value.is_nan());
    if nan_accepted {
        ledger.compared(nan_arm, case, nan_pass);
    }
    let pass = inf_diff <= 0.0 && nan_pass;
    let log = DiffTestLog {
        test_id: "adv_009b_fftfreq_nonfinite_d".to_string(),
        category: "differential".to_string(),
        input_summary: "fftfreq(n=4, d=inf); rfftfreq(n=4, d=nan)".to_string(),
        expected: "scipy returns signed zero bins for inf and NaN bins for nan".to_string(),
        actual: format!("inf={inf_freqs:?}; nan={nan_freqs:?}"),
        diff: inf_diff,
        tolerance: 0.0,
        compared: ledger.counts().clone(),
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: 0,
    };
    emit_log(&log);
    assert!(
        pass,
        "non-finite spacing should match scipy/numpy propagation"
    );
    ledger.finish(1);
}

#[test]
fn adv_010_all_zeros_input() {
    let input = vec![(0.0, 0.0); 8];
    let opts = FftOptions::default();
    let result = fft(&input, &opts).unwrap();
    let expected = vec![(0.0, 0.0); 8];
    let (arm, case) = ("all_zeros", "adv_010_zeros");
    let mut ledger = CompareLedger::new("adv_010_all_zeros_input", &[arm]);
    let accepted = complex_slices(&mut ledger, arm, case, &expected, &result);
    let diff = max_abs_diff_complex(&result, &expected);
    if accepted {
        ledger.compared(arm, case, diff <= 0.0);
    }
    run_diff_test(
        &ledger,
        case,
        "adversarial",
        "fft of all-zeros = all-zeros",
        diff,
        0.0,
    );
    ledger.finish(1);
}
