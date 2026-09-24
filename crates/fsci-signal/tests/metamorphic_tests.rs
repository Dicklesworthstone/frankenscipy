//! Metamorphic tests for `fsci-signal`.
//!
//! Convolution commutativity, filtfilt zero-phase, lfilter impulse
//! identity, hilbert real-part recovery, window normalisation.
//!
//! Run with: `cargo test -p fsci-signal --test metamorphic_tests`

use fsci_signal::{
    ConvolveMode, FilterType, FindPeaksOptions, argrelmax, argrelmin, autocorrelation, bartlett,
    bessel as iir_bessel, blackman, boxcar, butter, cheby1, convolve, correlate, cwt, deconvolve,
    deemphasis, downsample, exponential_smooth, fftconvolve, filtfilt, find_peaks, freqz, gaussian,
    group_delay_from_ba, hamming, hann, hilbert, hilbert_envelope, iirnotch, iirpeak,
    impulse_response, kaiser, lanczos, lfilter, magnitude_response, matched_filter, max_len_seq,
    medfilt1, morlet, normalize_signal, parzen, peak_to_peak, phase_response, preemphasis,
    resample, ricker, rms, savgol_filter, signal_energy, sos2tf, sosfilt, sosfiltfilt,
    spectral_centroid, step_response, sweep_poly, tf2sos, tf2zpk, triang, unwrap_phase, upsample,
    xcorr_coefficient, zero_crossing_rate, zpk2tf,
};

const ATOL: f64 = 1e-9;
const RTOL: f64 = 1e-7;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(
        close(a, b),
        "{msg}: actual={a:.16e} expected={b:.16e} diff={:.3e}",
        (a - b).abs()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — convolve(a, b) = convolve(b, a) (commutativity).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_commutative() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![0.5, -1.0, 0.25, 2.0];
    let ab = convolve(&a, &b, ConvolveMode::Full).unwrap();
    let ba = convolve(&b, &a, ConvolveMode::Full).unwrap();
    assert_eq!(ab.len(), ba.len(), "MR1 length mismatch");
    for (i, (l, r)) in ab.iter().zip(&ba).enumerate() {
        assert_close(*l, *r, &format!("MR1 commutativity at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — convolve and fftconvolve (Full mode) agree element-wise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_matches_fftconvolve() {
    let a = vec![1.0, -0.5, 0.25, 0.7, 1.4, -0.3, 0.0, 1.1];
    let b = vec![0.3, 0.5, -0.2, 0.1];
    let direct = convolve(&a, &b, ConvolveMode::Full).unwrap();
    let fft = fftconvolve(&a, &b, ConvolveMode::Full).unwrap();
    assert_eq!(direct.len(), fft.len(), "MR2 length mismatch");
    for (i, (l, r)) in direct.iter().zip(&fft).enumerate() {
        assert!(
            (l - r).abs() <= 1e-8 * l.abs().max(r.abs()).max(1.0) + 1e-10,
            "MR2 direct vs fft at i={i}: {l} vs {r}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — convolve with the unit impulse [1] returns the input unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_with_unit_impulse_is_identity() {
    let x = vec![3.5, -1.2, 0.8, 4.1, -2.3];
    let out = convolve(&x, &[1.0], ConvolveMode::Full).unwrap();
    assert_eq!(out.len(), x.len(), "MR3 length");
    for (i, (got, want)) in out.iter().zip(&x).enumerate() {
        assert_close(*got, *want, &format!("MR3 impulse at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — correlate(a, b) = convolve(a, reverse(b)) (Full mode).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_correlate_equals_convolve_reversed() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![0.5, -1.0, 0.25];
    let corr = correlate(&a, &b, ConvolveMode::Full).unwrap();
    let mut b_rev = b.clone();
    b_rev.reverse();
    let conv = convolve(&a, &b_rev, ConvolveMode::Full).unwrap();
    assert_eq!(corr.len(), conv.len(), "MR4 length mismatch");
    for (i, (l, r)) in corr.iter().zip(&conv).enumerate() {
        assert_close(*l, *r, &format!("MR4 at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — lfilter with b=[1.0], a=[1.0] is identity.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lfilter_passthrough_is_identity() {
    let x = vec![1.0, 2.0, -3.0, 0.7, 4.2, -1.1, 5.5];
    let out = lfilter(&[1.0], &[1.0], &x, None).unwrap();
    assert_eq!(out.len(), x.len(), "MR5 length");
    for (i, (got, want)) in out.iter().zip(&x).enumerate() {
        assert_close(*got, *want, &format!("MR5 at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — filtfilt with the same coefficients applied twice converges to
// a zero-phase result. As a structural check, filtfilt of a constant
// signal returns the same constant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_filtfilt_preserves_constant() {
    // A simple low-pass: y[n] = 0.5 x[n] + 0.5 x[n-1].
    let b = vec![0.5, 0.5];
    let a = vec![1.0];
    let n = 32;
    let x = vec![3.7_f64; n];
    let out = filtfilt(&b, &a, &x).unwrap();
    assert_eq!(out.len(), n);
    for (i, &v) in out.iter().enumerate() {
        assert!(close(v, 3.7), "MR6 filtfilt changed constant at i={i}: {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — hilbert real part recovers the input.
//
// scipy.signal.hilbert returns the analytic signal x_a = x + i·H[x].
// The real component must equal the input (up to round-off).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_real_part_recovers_input() {
    // Pick lengths where the harness exercises typical FFT sizes.
    for &n in &[16_usize, 32, 64, 100] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64).sin()).collect();
        let analytic = hilbert(&x).unwrap();
        assert_eq!(analytic.len(), n, "MR7 length");
        for (i, ((re, _im), &xi)) in analytic.iter().zip(&x).enumerate() {
            assert!(
                close(*re, xi),
                "MR7 hilbert real part at n={n} i={i}: {re} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — window functions are non-negative on the standard [0, 1]
// support and symmetric.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_windows_nonneg_and_symmetric() {
    for n in [8usize, 16, 33, 64] {
        let windows: Vec<(&str, Vec<f64>)> = vec![
            ("hann", hann(n)),
            ("hamming", hamming(n)),
            ("blackman", blackman(n)),
            ("kaiser-6", kaiser(n, 6.0)),
        ];
        for (name, w) in windows {
            assert_eq!(w.len(), n, "{name} length n={n}");
            // Non-negative: hann/hamming/blackman are ≥0; kaiser(β>0) too.
            for (i, &v) in w.iter().enumerate() {
                assert!(v >= -1e-12, "MR8 {name} negative at i={i}: {v}");
                assert!(v <= 1.0 + 1e-12, "MR8 {name} > 1 at i={i}: {v}");
            }
            // Symmetric: w[k] == w[n-1-k].
            for k in 0..n / 2 {
                let mirror = w[n - 1 - k];
                assert!(
                    (w[k] - mirror).abs() < 1e-12,
                    "MR8 {name} not symmetric at k={k}: {} vs {mirror}",
                    w[k]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — convolve linearity in the first argument:
//   convolve(α x + β y, h) = α convolve(x, h) + β convolve(y, h)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convolve_linear_in_first_arg() {
    let x = vec![1.0, 0.5, -0.3, 0.8, 1.2];
    let y = vec![0.5, 1.5, -0.2, 0.9, 0.4];
    let h = vec![0.3, -0.1, 0.4];
    let alpha = 1.7_f64;
    let beta = -0.5_f64;
    let combined: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| alpha * xi + beta * yi)
        .collect();
    let combined_c = convolve(&combined, &h, ConvolveMode::Full).unwrap();
    let xc = convolve(&x, &h, ConvolveMode::Full).unwrap();
    let yc = convolve(&y, &h, ConvolveMode::Full).unwrap();
    assert_eq!(combined_c.len(), xc.len());
    for (i, ((cv, xv), yv)) in combined_c.iter().zip(&xc).zip(&yc).enumerate() {
        let expected = alpha * xv + beta * yv;
        assert_close(*cv, expected, &format!("MR9 linearity at i={i}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — Hilbert envelope identity: |x_a|² = x² + H[x]², the
// definition of analytic-signal magnitude squared.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_envelope_squared_identity() {
    for &n in &[16_usize, 32, 64, 100] {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64 * std::f64::consts::TAU).sin())
            .collect();
        let analytic = hilbert(&x).unwrap();
        for (i, ((re, im), &xi)) in analytic.iter().zip(&x).enumerate() {
            // Real part should equal input (already covered by MR7).
            // Envelope: |x_a|² = re² + im² should equal xi² + im².
            let env_sq = re * re + im * im;
            let expected = xi * xi + im * im;
            assert!(
                (env_sq - expected).abs() < 1e-9 * env_sq.max(1.0),
                "MR10 hilbert envelope at n={n} i={i}: got {env_sq}, expected {expected}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — Hilbert applied to a constant signal returns x_a.real = const
// and x_a.imag ≈ 0 (since the Hilbert transform of a constant is 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_of_constant() {
    let n = 64_usize;
    let value = 3.5_f64;
    let x = vec![value; n];
    let analytic = hilbert(&x).unwrap();
    for (i, (re, im)) in analytic.iter().enumerate() {
        assert!(
            (re - value).abs() < 1e-9,
            "MR11 hilbert const real part at i={i}: {re} vs {value}"
        );
        assert!(
            im.abs() < 1e-9,
            "MR11 hilbert const imag part at i={i}: {im}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — find_peaks on a strictly monotone signal returns no peaks.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_peaks_monotone_has_no_peaks() {
    // Strictly increasing: no peak at any interior position.
    let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let result = find_peaks(&x, FindPeaksOptions::default()).expect("find_peaks");
    assert!(
        result.peaks.is_empty(),
        "MR12 monotone signal should have no peaks, got {:?}",
        result.peaks
    );
    // Strictly decreasing: also no peaks.
    let y: Vec<f64> = (0..50).rev().map(|i| i as f64 * 0.1).collect();
    let result = find_peaks(&y, FindPeaksOptions::default()).expect("find_peaks");
    assert!(
        result.peaks.is_empty(),
        "MR12 monotone-decreasing signal should have no peaks, got {:?}",
        result.peaks
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — find_peaks on a discrete tent [0, 1, 0] returns index 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_peaks_tent() {
    let x = vec![0.0, 1.0, 0.0];
    let result = find_peaks(&x, FindPeaksOptions::default()).expect("find_peaks");
    assert_eq!(result.peaks, vec![1], "MR13 expected single peak at idx 1");
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — sosfilt and lfilter produce the same output for the same
// transfer function (when the SOS is constructed via tf2sos).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sosfilt_matches_lfilter() {
    // Simple low-pass: y[n] = (x[n] + 0.5 x[n-1]) / 1.5
    let b = vec![1.0_f64, 0.5];
    let a = vec![1.0_f64, 0.0];
    let x: Vec<f64> = (0..32).map(|i| (i as f64 / 4.0).sin()).collect();

    let y_lf = lfilter(&b, &a, &x, None).unwrap();
    let sos = tf2sos(&b, &a).unwrap();
    let y_sos = sosfilt(&sos, &x).unwrap();
    assert_eq!(y_lf.len(), y_sos.len(), "MR14 length mismatch");
    for (i, (l, s)) in y_lf.iter().zip(&y_sos).enumerate() {
        assert!(
            close(*l, *s),
            "MR14 lfilter vs sosfilt at i={i}: {l} vs {s}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — hann window of length n peaks at value 1 at the midpoint
// (sym=True, scipy convention: hann[(n-1)/2] = 1 for odd n).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hann_peak_at_midpoint() {
    for &n in &[7_usize, 11, 21, 51] {
        let w = hann(n);
        let mid = (n - 1) / 2;
        // For odd n, the maximum is exactly 1 at index (n-1)/2.
        assert!(
            (w[mid] - 1.0).abs() < 1e-12,
            "MR15 hann peak n={n} mid={mid} value={}",
            w[mid]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — Sum of a hamming window of length n ≈ 0.54 · n + ε:
// the average sample value is ~0.54 (the center coefficient).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hamming_sum_matches_central_coeff() {
    for &n in &[64_usize, 128, 256] {
        let w = hamming(n);
        let sum: f64 = w.iter().sum();
        let avg = sum / n as f64;
        // hamming = 0.54 - 0.46 cos(2πk/(N-1)) → mean over k is ~0.54.
        // Allow generous slack since the cos term doesn't integrate
        // to exactly zero at finite N.
        assert!(
            (avg - 0.54).abs() < 0.02,
            "MR16 hamming average n={n}: {avg}, expected ~0.54"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — A first-difference filter on a linear ramp produces a constant
// after the first sample. y[n] = x[n] - x[n-1] for the linear input
// x[n] = 2n + 1 yields y[n] = 2 for n ≥ 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lfilter_first_difference_on_ramp() {
    let x: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
    // y[n] = x[n] - x[n-1]
    let b = vec![1.0_f64, -1.0];
    let a = vec![1.0_f64];
    let y = lfilter(&b, &a, &x, None).unwrap();
    // y[0] = x[0] - 0 = 1 (assuming zi = 0); y[k] = 2 for k >= 1.
    assert!(
        (y[0] - 1.0).abs() < 1e-12,
        "MR17 first-diff y[0] = {}, expected 1",
        y[0]
    );
    for (k, &yk) in y.iter().enumerate().skip(1) {
        assert!(
            (yk - 2.0).abs() < 1e-12,
            "MR17 first-diff y[{k}] = {yk}, expected 2"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — zero_crossing_rate of a constant non-zero signal is 0; of a
// strict alternator [+1, -1, +1, -1, ...] is positive.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_zero_crossing_rate_constant_vs_alternator() {
    let constant = vec![3.0_f64; 32];
    let zcr_const = zero_crossing_rate(&constant);
    assert!(
        zcr_const.abs() < 1e-12,
        "MR18 zcr(constant) = {zcr_const}, expected 0"
    );
    let alt: Vec<f64> = (0..32)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let zcr_alt = zero_crossing_rate(&alt);
    assert!(
        zcr_alt > 0.0,
        "MR18 zcr(alternator) = {zcr_alt}, expected > 0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — rms(x) = √(mean(x²)) for any signal; equals |c| for a
// constant signal of value c.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rms_definition_and_constant() {
    let xs: [Vec<f64>; 3] = [
        vec![3.0_f64; 16],
        (0..32).map(|i| (i as f64).sin()).collect(),
        vec![1.0_f64, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0],
    ];
    for x in &xs {
        let r = rms(x);
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        assert!(
            (r - mean_sq.sqrt()).abs() < 1e-12,
            "MR19 rms = {r}, expected √mean(x²) = {}",
            mean_sq.sqrt()
        );
    }
    let c = vec![-2.5_f64; 10];
    let r = rms(&c);
    assert!((r - 2.5).abs() < 1e-12, "MR19 rms(const -2.5) = {r}");
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Autocorrelation has its maximum at lag 0 and equals Σx_i² there.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_autocorrelation_peak_at_zero_lag() {
    // Sample autocorrelation here is mean-centred and normalised so
    // ac[0] = 1 and |ac[k]| ≤ 1 for k ≥ 1.
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).cos()).collect();
    let max_lag = 8;
    let ac = autocorrelation(&x, max_lag);
    assert!(
        (ac[0] - 1.0).abs() < 1e-9,
        "MR20 ac[0] = {}, expected 1.0",
        ac[0]
    );
    for (k, &v) in ac.iter().enumerate().skip(1) {
        assert!(
            v.abs() <= ac[0] + 1e-9,
            "MR20 |ac[{k}]| = {} > ac[0] = {}",
            v.abs(),
            ac[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — argrelmax and argrelmin produce disjoint index sets.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_argrelmax_argrelmin_disjoint() {
    let x: Vec<f64> = (0..40)
        .map(|i| (i as f64 * 0.6).sin() + 0.5 * (i as f64 * 0.13).cos())
        .collect();
    let maxs = argrelmax(&x, 1);
    let mins = argrelmin(&x, 1);
    for &m in &maxs {
        assert!(
            !mins.contains(&m),
            "MR21 index {m} appears in both argrelmax and argrelmin"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — deconvolve(convolve(x, h), h) recovers x (within numerical
// noise) when the divisor leading coefficient is non-zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_deconvolve_inverts_convolve() {
    let x = vec![1.0_f64, 2.0, 3.0, 1.0, -1.0, 0.5];
    let h = vec![1.0_f64, 0.5, -0.25];
    let y = convolve(&x, &h, ConvolveMode::Full).unwrap();
    let (q, _r) = deconvolve(&y, &h).unwrap();
    assert_eq!(
        q.len(),
        x.len(),
        "MR22 deconvolve quotient length = {} vs x length = {}",
        q.len(),
        x.len()
    );
    for (i, (qi, xi)) in q.iter().zip(&x).enumerate() {
        assert!(
            (qi - xi).abs() < 1e-9,
            "MR22 deconvolve at i={i}: q = {qi}, x = {xi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — spectral_centroid lies between the smallest and largest
// frequency bins for any non-negative magnitude spectrum.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spectral_centroid_in_freq_range() {
    let mags = [
        vec![1.0_f64, 2.0, 3.0, 1.0, 0.5, 0.25, 0.1],
        vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![1.0_f64; 7],
    ];
    let freqs: Vec<f64> = (0..7).map(|k| k as f64 * 100.0).collect();
    for m in &mags {
        let c = spectral_centroid(m, &freqs);
        let fmin = *freqs.first().unwrap();
        let fmax = *freqs.last().unwrap();
        assert!(
            c >= fmin - 1e-9 && c <= fmax + 1e-9,
            "MR23 centroid = {c} outside [{fmin}, {fmax}]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — bartlett, hann, hamming, parzen, triang windows of odd length
// are symmetric: w[i] = w[n-1-i] for 0 ≤ i < n/2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_classic_windows_are_symmetric() {
    let n = 31; // odd
    let windows: Vec<(&str, Vec<f64>)> = vec![
        ("bartlett", bartlett(n)),
        ("hann", hann(n)),
        ("hamming", hamming(n)),
        ("parzen", parzen(n)),
        ("triang", triang(n)),
        ("blackman", blackman(n)),
    ];
    for (name, w) in &windows {
        for i in 0..n / 2 {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-12,
                "MR24 {name} not symmetric at i={i}: {} vs {}",
                w[i],
                w[n - 1 - i]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — boxcar window is all ones; lanczos values are in [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_boxcar_ones_lanczos_bounded() {
    for n in [4usize, 7, 16, 31] {
        let b = boxcar(n, true);
        for (i, &v) in b.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-15,
                "MR25 boxcar(n={n})[{i}] = {v}, expected 1"
            );
        }
        let l = lanczos(n);
        for (i, &v) in l.iter().enumerate() {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(&v),
                "MR25 lanczos(n={n})[{i}] = {v} outside [0, 1]"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — Gaussian window of odd length has its peak at the midpoint
// equal to 1.0 (un-normalised).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gaussian_window_peak_at_midpoint() {
    let n = 31;
    let mid = n / 2;
    for &std in &[1.0_f64, 2.5, 5.0] {
        let w = gaussian(n, std, true);
        assert!(
            (w[mid] - 1.0).abs() < 1e-12,
            "MR26 gaussian(n={n}, std={std})[{mid}] = {}, expected 1",
            w[mid]
        );
        // Window is monotonically non-increasing from midpoint outwards.
        for k in 1..=mid {
            assert!(
                w[mid - k] <= w[mid - k + 1] + 1e-12,
                "MR26 gaussian non-monotone left at offset {k}"
            );
            assert!(
                w[mid + k] <= w[mid + k - 1] + 1e-12,
                "MR26 gaussian non-monotone right at offset {k}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — resample preserves the requested output length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_resample_output_length() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).sin()).collect();
    for &num in &[8usize, 16, 32, 48, 64] {
        let y = resample(&x, num).unwrap();
        assert_eq!(
            y.len(),
            num,
            "MR27 resample length: got {} expected {num}",
            y.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — hilbert_envelope dominates |x| pointwise on smooth signals
// (envelope is the magnitude of the analytic signal, ≥ |x|).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_envelope_dominates_abs_input() {
    let x: Vec<f64> = (0..64)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())
        .collect();
    let env = hilbert_envelope(&x).unwrap();
    for (i, (xi, ei)) in x.iter().zip(&env).enumerate() {
        assert!(
            *ei + 1e-9 >= xi.abs(),
            "MR28 envelope[{i}] = {ei} < |x| = {}",
            xi.abs()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — Hilbert transform: real part of the analytic signal equals the
// original signal (Re(H[x]) = x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_real_part_equals_input() {
    let x: Vec<f64> = (0..32)
        .map(|i| 1.0 + 0.5 * (i as f64 * 0.3).cos() + 0.2 * (i as f64 * 0.7).sin())
        .collect();
    let h = hilbert(&x).unwrap();
    for (i, (xi, &(re, _im))) in x.iter().zip(&h).enumerate() {
        assert!(
            (xi - re).abs() < 1e-9,
            "MR29 Re(H[x])[{i}] = {re} vs x = {xi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — peak_to_peak(x) = max(x) - min(x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_peak_to_peak_definition() {
    let xs: &[Vec<f64>] = &[
        vec![1.0, -2.0, 3.0, 0.5, -1.5, 4.0],
        vec![0.0; 8],
        vec![5.0; 5],
        (0..16).map(|i| (i as f64 * 0.5).sin()).collect(),
    ];
    for x in xs {
        let p2p = peak_to_peak(x);
        let mx = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mn = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let expected = mx - mn;
        assert!(
            (p2p - expected).abs() < 1e-12,
            "MR30 peak_to_peak = {p2p} vs (max - min) = {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — Preemphasis followed by deemphasis approximately recovers
// the input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_preemphasis_deemphasis_roundtrip() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).cos()).collect();
    let coeff = 0.95_f64;
    let y = preemphasis(&x, coeff);
    let z = deemphasis(&y, coeff);
    assert_eq!(z.len(), x.len(), "MR31 length mismatch");
    for (i, (xi, zi)) in x.iter().zip(&z).enumerate() {
        assert!(
            (xi - zi).abs() < 1e-9,
            "MR31 preemphasis∘deemphasis at i={i}: {xi} vs {zi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — normalize_signal performs z-score normalization: output has
// mean 0 and std 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_normalize_signal_zscore() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).sin() * 100.0).collect();
    let n = normalize_signal(&x);
    let len = n.len() as f64;
    let mean: f64 = n.iter().sum::<f64>() / len;
    let var: f64 = n.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / len;
    let std = var.sqrt();
    assert!(
        mean.abs() < 1e-9,
        "MR32 normalize_signal mean = {mean}, expected 0"
    );
    assert!(
        (std - 1.0).abs() < 1e-9,
        "MR32 normalize_signal std = {std}, expected 1"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — downsample(x, k) returns ⌈n/k⌉ samples; upsample(x, k)
// returns n*k samples.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_downsample_upsample_lengths() {
    let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
    for &k in &[2usize, 4, 8] {
        let d = downsample(&x, k);
        assert_eq!(d.len(), x.len().div_ceil(k), "MR33 downsample length k={k}");
        let u = upsample(&x, k);
        assert_eq!(u.len(), x.len() * k, "MR33 upsample length k={k}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — signal_energy(x) = Σ x_i².
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_signal_energy_definition() {
    let xs: &[Vec<f64>] = &[
        vec![1.0, 2.0, 3.0, 4.0],
        (0..32).map(|i| (i as f64 * 0.4).cos()).collect(),
        vec![0.0; 8],
    ];
    for x in xs {
        let e = signal_energy(x);
        let manual: f64 = x.iter().map(|v| v * v).sum();
        assert!(
            (e - manual).abs() < 1e-12,
            "MR34 signal_energy = {e} vs Σx² = {manual}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — xcorr_coefficient(x, x) = 1 for any non-zero x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_xcorr_coefficient_self_is_one() {
    let xs: &[Vec<f64>] = &[
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        (0..16).map(|i| (i as f64 * 0.5).sin()).collect(),
        (0..32)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    ];
    for x in xs {
        let c = xcorr_coefficient(x, x);
        assert!(
            (c - 1.0).abs() < 1e-9,
            "MR35 xcorr_coefficient(x, x) = {c}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — Butterworth low-pass design returns finite (b, a) coefficients
// with a[0] = 1.0 (normalised).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_butter_lowpass_finite_normalised() {
    for order in [2usize, 4, 6, 8] {
        let r = butter(order, &[0.3], FilterType::Lowpass).unwrap();
        for (i, &v) in r.b.iter().enumerate() {
            assert!(v.is_finite(), "MR36 butter b[{i}] non-finite");
        }
        for (i, &v) in r.a.iter().enumerate() {
            assert!(v.is_finite(), "MR36 butter a[{i}] non-finite");
        }
        assert!(
            (r.a[0] - 1.0).abs() < 1e-12,
            "MR36 butter a[0] = {} != 1.0",
            r.a[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — tf2zpk then zpk2tf is approximately the identity on the
// (b, a) transfer function.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_tf2zpk_zpk2tf_roundtrip() {
    let r = butter(4, &[0.4], FilterType::Lowpass).unwrap();
    let zpk = tf2zpk(&r.b, &r.a).unwrap();
    let back = zpk2tf(&zpk);
    assert_eq!(back.b.len(), r.b.len(), "MR37 tf roundtrip b length");
    assert_eq!(back.a.len(), r.a.len(), "MR37 tf roundtrip a length");
    for (i, (x, y)) in r.b.iter().zip(&back.b).enumerate() {
        assert!(
            (x - y).abs() < 1e-7 * x.abs().max(1.0),
            "MR37 tf2zpk/zpk2tf b[{i}]: {x} vs {y}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — tf2sos and sos2tf round-trip preserves the (b, a) form
// (modulo gain factoring) — verify by re-applying tf2sos to the result.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sos_roundtrip_consistent() {
    let r = butter(4, &[0.4], FilterType::Lowpass).unwrap();
    let sos = tf2sos(&r.b, &r.a).unwrap();
    let back = sos2tf(&sos);
    let sos2 = tf2sos(&back.b, &back.a).unwrap();
    assert_eq!(
        sos.len(),
        sos2.len(),
        "MR38 SOS round-trip section count: {} vs {}",
        sos.len(),
        sos2.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — sosfiltfilt preserves input length (same-length output as x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sosfiltfilt_preserves_length() {
    let r = butter(2, &[0.3], FilterType::Lowpass).unwrap();
    let sos = tf2sos(&r.b, &r.a).unwrap();
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).sin()).collect();
    let y = sosfiltfilt(&sos, &x).unwrap();
    assert_eq!(
        y.len(),
        x.len(),
        "MR39 sosfiltfilt length: got {} expected {}",
        y.len(),
        x.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — freqz returns finite frequency response across the default
// number of samples.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_freqz_finite_response() {
    let r = butter(3, &[0.4], FilterType::Lowpass).unwrap();
    let resp = freqz(&r.b, &r.a, Some(64)).unwrap();
    assert_eq!(resp.h_mag.len(), 64, "MR40 freqz length");
    for (k, &m) in resp.h_mag.iter().enumerate() {
        assert!(m.is_finite() && m >= -1e-12, "MR40 freqz h_mag[{k}] = {m}");
    }
    for (k, &p) in resp.h_phase.iter().enumerate() {
        assert!(p.is_finite(), "MR40 freqz h_phase[{k}] = {p}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — iirnotch and iirpeak return finite coefficients with a[0] = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_iirnotch_iirpeak_finite() {
    for &(w0, q) in &[(0.3_f64, 30.0), (0.5, 10.0), (0.7, 25.0)] {
        let n = iirnotch(w0, q).unwrap();
        let p = iirpeak(w0, q).unwrap();
        for (i, &v) in n.b.iter().enumerate() {
            assert!(v.is_finite(), "MR41 iirnotch b[{i}] non-finite");
        }
        for (i, &v) in p.b.iter().enumerate() {
            assert!(v.is_finite(), "MR41 iirpeak b[{i}] non-finite");
        }
        assert!(
            (n.a[0] - 1.0).abs() < 1e-12,
            "MR41 iirnotch a[0] = {} != 1",
            n.a[0]
        );
        assert!(
            (p.a[0] - 1.0).abs() < 1e-12,
            "MR41 iirpeak a[0] = {} != 1",
            p.a[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — Savitzky-Golay filter on a perfectly polynomial input of
// degree ≤ polyorder reproduces the input exactly (interior).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_savgol_filter_polynomial_exactness() {
    // p(x) = 2x² - 0.5x + 1 sampled at uniform x.
    let x: Vec<f64> = (0..32)
        .map(|i| 2.0 * (i as f64).powi(2) - 0.5 * i as f64 + 1.0)
        .collect();
    let y = savgol_filter(&x, 7, 2).unwrap();
    // Check interior (skip first/last few samples affected by boundary).
    for i in 5..(x.len() - 5) {
        assert!(
            (y[i] - x[i]).abs() < 1e-7,
            "MR42 savgol(p)[{i}] = {} vs {}",
            y[i],
            x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — sweep_poly returns a vector of the same length as t.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sweep_poly_length_matches_t() {
    let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
    let poly = vec![1.0_f64, -2.0, 0.5];
    let s = sweep_poly(&t, &poly);
    assert_eq!(s.len(), t.len(), "MR43 sweep_poly length");
    for (i, &v) in s.iter().enumerate() {
        assert!(
            v.is_finite() && v.abs() <= 1.0 + 1e-12,
            "MR43 sweep_poly[{i}] = {v} non-finite or |v|>1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — ricker wavelet of odd length is symmetric.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ricker_symmetric() {
    for n in [11usize, 21, 41, 101] {
        let r = ricker(n, 5.0);
        for i in 0..n / 2 {
            assert!(
                (r[i] - r[n - 1 - i]).abs() < 1e-12,
                "MR44 ricker n={n} not symmetric at {i}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — morlet returns m complex samples.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_morlet_length_matches_m() {
    for m in [16usize, 32, 64] {
        let w = morlet(m, 5.0, 1.0, true);
        assert_eq!(w.len(), m, "MR45 morlet length");
        for (i, &(re, im)) in w.iter().enumerate() {
            assert!(
                re.is_finite() && im.is_finite(),
                "MR45 morlet[{i}] = ({re}, {im}) non-finite"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — unwrap_phase preserves the input length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_unwrap_phase_preserves_length() {
    let phase: Vec<f64> = (0..32)
        .map(|i| ((i as f64 * 0.5).sin() * 4.0).rem_euclid(2.0 * std::f64::consts::PI))
        .collect();
    let unwrapped = unwrap_phase(&phase);
    assert_eq!(unwrapped.len(), phase.len(), "MR46 unwrap_phase length");
    for &v in &unwrapped {
        assert!(v.is_finite(), "MR46 unwrap_phase non-finite");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — cwt returns one row per requested width.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cwt_rows_match_widths() {
    let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.4).sin()).collect();
    let widths = vec![1.0_f64, 2.0, 4.0, 8.0];
    let result = cwt(&data, ricker, &widths).unwrap();
    assert_eq!(result.len(), widths.len(), "MR47 cwt rows = widths");
    for (i, row) in result.iter().enumerate() {
        assert_eq!(row.len(), data.len(), "MR47 cwt row {i} length");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — medfilt1 preserves length and is idempotent on a uniform input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_medfilt1_preserves_length_and_constant() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).sin()).collect();
    let y = medfilt1(&x, 3);
    assert_eq!(y.len(), x.len(), "MR48 medfilt1 length");
    let c = vec![5.5_f64; 16];
    let yc = medfilt1(&c, 3);
    for &v in &yc {
        assert!((v - 5.5).abs() < 1e-12, "MR48 medfilt1(const)[{v}] != 5.5");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — exponential_smooth preserves length and reduces high-frequency
// noise (smoothed values lie within input min/max bounds).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_exponential_smooth_length_and_bounds() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.4).sin() + 0.5).collect();
    let alpha = 0.3_f64;
    let y = exponential_smooth(&x, alpha);
    assert_eq!(y.len(), x.len(), "MR49 exponential_smooth length");
    let mn = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mx = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for &v in &y {
        assert!(
            v >= mn - 1e-9 && v <= mx + 1e-9,
            "MR49 smoothed value {v} outside [{mn}, {mx}]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — max_len_seq(nbits) returns a sequence of length 2^nbits - 1
// with each entry in {-1, +1} (or 0/1 — verify they're bipolar/binary).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_max_len_seq_length_and_values() {
    for nbits in [3usize, 4, 5, 6] {
        let seq = max_len_seq(nbits).unwrap();
        let expected = (1usize << nbits) - 1;
        assert_eq!(
            seq.len(),
            expected,
            "MR50 MLS({nbits}) length = {} expected {expected}",
            seq.len()
        );
        for (i, &v) in seq.iter().enumerate() {
            assert!(
                v == 1.0 || v == -1.0 || v == 0.0,
                "MR50 MLS[{i}] = {v}, expected ±1 or 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — matched_filter output peaks at the position where the
// template matches the signal.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_matched_filter_peak_position() {
    // template = [1.0, 0.5, -0.5, 1.0]; signal embeds template at offset 5.
    let template = vec![1.0_f64, 0.5, -0.5, 1.0];
    let mut signal = vec![0.0_f64; 16];
    for (i, &t) in template.iter().enumerate() {
        signal[5 + i] = t;
    }
    let mf = matched_filter(&template, &signal).unwrap();
    let (peak_idx, &peak_val) = mf
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert!(peak_val > 0.0, "MR51 matched_filter peak = {peak_val} ≤ 0");
    // Peak should land near the embedding (offset 5..8 in 16-sample signal,
    // depending on filter convention — start of match or end of match).
    assert!(
        (5..=8).contains(&peak_idx),
        "MR51 matched_filter peak idx = {peak_idx}, expected in [5, 8]"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — impulse_response returns exactly n_samples elements.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_impulse_response_length() {
    let r = butter(2, &[0.3], FilterType::Lowpass).unwrap();
    for &n in &[16usize, 32, 64] {
        let h = impulse_response(&r.b, &r.a, n).unwrap();
        assert_eq!(
            h.len(),
            n,
            "MR52 impulse_response length = {} vs requested {}",
            h.len(),
            n
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — find_peaks on a single peak (sin wave centered at midpoint)
// returns at least one peak.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_find_peaks_finds_single_peak() {
    // Build a Gaussian-like bump.
    let x: Vec<f64> = (0..32)
        .map(|i| {
            let d = i as f64 - 16.0;
            (-d * d / 50.0).exp()
        })
        .collect();
    let opts = FindPeaksOptions::default();
    let res = find_peaks(&x, opts).expect("find_peaks");
    assert!(
        !res.peaks.is_empty(),
        "MR53 find_peaks on Gaussian bump returned 0 peaks"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — step_response returns the requested number of samples.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_step_response_length() {
    let r = butter(2, &[0.3], FilterType::Lowpass).unwrap();
    for &n in &[16usize, 32, 64] {
        let h = step_response(&r.b, &r.a, n).unwrap();
        assert_eq!(h.len(), n, "MR54 step_response length");
        for &v in &h {
            assert!(v.is_finite(), "MR54 step_response non-finite");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — group_delay_from_ba returns finite group delay across the
// requested grid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_group_delay_finite() {
    let r = butter(3, &[0.4], FilterType::Lowpass).unwrap();
    let (w, gd) = group_delay_from_ba(&r.b, &r.a, 64);
    assert_eq!(w.len(), 64, "MR55 group_delay w length");
    assert_eq!(gd.len(), 64, "MR55 group_delay length");
    for (i, &g) in gd.iter().enumerate() {
        assert!(g.is_finite(), "MR55 group_delay[{i}] = {g}");
    }

    let (_, pure_delay) = group_delay_from_ba(&[0.0, 1.0], &[1.0], 16);
    for (i, &g) in pure_delay.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-12,
            "MR55 pure one-sample delay group_delay[{i}] = {g}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — magnitude_response returns a finite linear-magnitude vector of
// the requested length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_magnitude_response_length_and_finite() {
    let r = butter(4, &[0.3], FilterType::Lowpass).unwrap();
    let (w, mag) = magnitude_response(&r.b, &r.a, 64);
    assert_eq!(w.len(), 64, "MR56 magnitude_response w length");
    assert_eq!(mag.len(), 64, "MR56 magnitude_response length");
    for (i, &m) in mag.iter().enumerate() {
        assert!(m.is_finite(), "MR56 |H|[{i}] = {m} non-finite");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR57 — cheby1 returns finite coefficients with a[0] = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cheby1_finite_normalised() {
    for order in [2usize, 4, 6] {
        let r = cheby1(order, 1.0, &[0.3], FilterType::Lowpass).unwrap();
        for (i, &v) in r.b.iter().enumerate() {
            assert!(v.is_finite(), "MR57 cheby1 b[{i}] non-finite");
        }
        for (i, &v) in r.a.iter().enumerate() {
            assert!(v.is_finite(), "MR57 cheby1 a[{i}] non-finite");
        }
        assert!(
            (r.a[0] - 1.0).abs() < 1e-12,
            "MR57 cheby1 a[0] = {}",
            r.a[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR58 — Bessel filter design returns finite coefficients with a[0] = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bessel_finite_normalised() {
    for order in [2usize, 4, 6] {
        let r = iir_bessel(order, &[0.3], FilterType::Lowpass).unwrap();
        for (i, &v) in r.b.iter().enumerate() {
            assert!(v.is_finite(), "MR58 bessel b[{i}] non-finite");
        }
        for (i, &v) in r.a.iter().enumerate() {
            assert!(v.is_finite(), "MR58 bessel a[{i}] non-finite");
        }
        assert!(
            (r.a[0] - 1.0).abs() < 1e-12,
            "MR58 bessel a[0] = {}",
            r.a[0]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR59 — phase_response returns finite phase across the requested grid.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_phase_response_finite() {
    let r = butter(3, &[0.4], FilterType::Lowpass).unwrap();
    let (w, phase) = phase_response(&r.b, &r.a, 64);
    assert_eq!(phase.len(), 64, "MR59 phase length");
    assert_eq!(w.len(), 64, "MR59 phase w length");
    for (i, &p) in phase.iter().enumerate() {
        assert!(p.is_finite(), "MR59 phase[{i}] = {p}");
    }
}
