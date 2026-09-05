//! Metamorphic tests for `fsci-fft`.
//!
//! Each test exercises a transform identity that holds independent of the
//! input contents, so no oracle is required. Examples: ifft∘fft = id,
//! Parseval's theorem, linearity, real-input round-trip via rfft/irfft.
//!
//! Run with: `cargo test -p fsci-fft --test metamorphic_tests`

use fsci_fft::{
    Complex64, FftOptions, Normalization, dct, dct_i, dct_iii, dct_iv, dctn, dst_i, dst_ii,
    dst_iii, dst_iv, dstn, fft, fft2, fftcorrelate, fftfreq, fftn, fftshift_1d, fht, fhtoffset,
    hamming_window, hann_window, hfft, hilbert, idct, idctn, idstn, ifft, ifft2, ifftn,
    ifftshift_1d, ifht, ihfft, irfft, irfft2, irfftn, next_fast_len, rfft, rfft2, rfftn,
};

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-10;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(close(a, b), "{msg}: {a:.16e} vs {b:.16e}");
}

fn close_complex(a: Complex64, b: Complex64) -> bool {
    close(a.0, b.0) && close(a.1, b.1)
}

/// Deterministic Lehmer-style PRNG. Avoids pulling in `rand` for tests.
fn prng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        // Map to [-1, 1)
        (bits as f64 / (1u64 << 32) as f64) * 2.0 - 1.0
    }
}

fn random_real(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = prng(seed);
    (0..n).map(|_| rng()).collect()
}

fn random_complex(n: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = prng(seed);
    (0..n).map(|_| (rng(), rng())).collect()
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — fft round-trip: ifft(fft(x)) == x for any x, any normalization.
// ─────────────────────────────────────────────────────────────────────

fn fft_round_trip_with(opts: &FftOptions, label: &str) {
    for &n in &[1, 2, 4, 8, 16, 17, 32, 64, 100] {
        let x = random_complex(n, 0xCAFE_F00D + n as u64);
        let xf = fft(&x, opts).unwrap();
        let xb = ifft(&xf, opts).unwrap();
        assert_eq!(xb.len(), n, "{label} length mismatch at n={n}");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close_complex(*got, *want),
                "{label} round-trip n={n} idx={i}: got=({}, {}) expected=({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }
}

#[test]
fn mr_fft_roundtrip_backward() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    fft_round_trip_with(&opts, "Backward");
}

#[test]
fn mr_fft_roundtrip_ortho() {
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    fft_round_trip_with(&opts, "Ortho");
}

#[test]
fn mr_fft_roundtrip_forward() {
    let opts = FftOptions::default().with_normalization(Normalization::Forward);
    fft_round_trip_with(&opts, "Forward");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — rfft / irfft round-trip on real inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_irfft_roundtrip() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 17, 32, 100] {
        let x = random_real(n, 0xDEAD_BEEF + n as u64);
        let xf = rfft(&x, &opts).unwrap();
        let xb = irfft(&xf, Some(n), &opts).unwrap();
        assert_eq!(xb.len(), n, "rfft round-trip length n={n}");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR2 rfft round-trip n={n} idx={i}: got={got} expected={want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — Parseval's theorem.
//
// Backward normalization (default): Σ|x_i|² = (1/N) · Σ|X_k|².
// Ortho normalization: Σ|x_i|² = Σ|X_k|² (FFT is unitary).
// ─────────────────────────────────────────────────────────────────────

fn complex_sq_norm(values: &[Complex64]) -> f64 {
    values.iter().map(|(re, im)| re * re + im * im).sum()
}

#[test]
fn mr_parseval_backward() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    for &n in &[8, 16, 32, 64, 100] {
        let x = random_complex(n, 0x1234_5678 + n as u64);
        let xf = fft(&x, &opts).unwrap();
        let lhs = complex_sq_norm(&x);
        let rhs = complex_sq_norm(&xf) / n as f64;
        assert_close(lhs, rhs, &format!("MR3 Parseval Backward n={n}"));
    }
}

#[test]
fn mr_parseval_ortho() {
    let opts = FftOptions::default().with_normalization(Normalization::Ortho);
    for &n in &[8, 16, 32, 64, 100] {
        let x = random_complex(n, 0xABCD_1234 + n as u64);
        let xf = fft(&x, &opts).unwrap();
        let lhs = complex_sq_norm(&x);
        let rhs = complex_sq_norm(&xf);
        assert_close(lhs, rhs, &format!("MR3 Parseval Ortho n={n}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — linearity: F(αx + βy) = α F(x) + β F(y)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_linearity() {
    let opts = FftOptions::default();
    let alpha = (1.5_f64, -0.3_f64);
    let beta = (-0.7_f64, 0.4_f64);
    let cmul = |a: Complex64, b: Complex64| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for &n in &[8, 16, 32, 64] {
        let x = random_complex(n, 0xFFEE_DDCC + n as u64);
        let y = random_complex(n, 0xCC11_22DD + n as u64);

        let combined: Vec<Complex64> = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| {
                let ax = cmul(alpha, *xi);
                let by = cmul(beta, *yi);
                (ax.0 + by.0, ax.1 + by.1)
            })
            .collect();

        let f_combined = fft(&combined, &opts).unwrap();
        let f_x = fft(&x, &opts).unwrap();
        let f_y = fft(&y, &opts).unwrap();

        for k in 0..n {
            let af = cmul(alpha, f_x[k]);
            let bf = cmul(beta, f_y[k]);
            let expected = (af.0 + bf.0, af.1 + bf.1);
            assert!(
                close_complex(f_combined[k], expected),
                "MR4 linearity n={n} k={k}: got=({}, {}) expected=({}, {})",
                f_combined[k].0,
                f_combined[k].1,
                expected.0,
                expected.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — DCT-II / IDCT round-trip
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_idct_roundtrip() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 32, 64] {
        let x = random_real(n, 0xC1C2_C3C4 + n as u64);
        let xf = dct(&x, &opts).unwrap();
        let xb = idct(&xf, &opts).unwrap();
        assert_eq!(xb.len(), n);
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR5 dct round-trip n={n} idx={i}: got={got} expected={want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — DST-II output structure: a periodic odd extension's DST is real.
//
// Specifically, DST-II of an all-zero input must be all-zero (linearity
// check at the zero of the operator). Combined with linearity it implies
// DST-II is well-defined for all inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dst_ii_zero_input_is_zero() {
    let opts = FftOptions::default();
    for &n in &[2, 4, 8, 16, 32] {
        let x = vec![0.0_f64; n];
        let xf = dst_ii(&x, &opts).unwrap();
        for (i, v) in xf.iter().enumerate() {
            assert!(v.abs() < 1e-12, "MR6 DST-II(0) n={n} idx={i}: {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — Conjugate symmetry of fft for real input.
// If x is real, then X[k] = conj(X[N-k]) for k = 1..N-1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_conjugate_symmetry_real_input() {
    let opts = FftOptions::default();
    for &n in &[8, 16, 32, 64] {
        let x_real = random_real(n, 0xAA_BB_CC_DD + n as u64);
        let x: Vec<Complex64> = x_real.iter().map(|&v| (v, 0.0)).collect();
        let xf = fft(&x, &opts).unwrap();
        for k in 1..n {
            let mirror = xf[n - k];
            let here = xf[k];
            assert!(
                close(here.0, mirror.0) && close(here.1, -mirror.1),
                "MR7 conjugate symmetry n={n} k={k}: X[k]=({}, {}) X[N-k]=({}, {})",
                here.0,
                here.1,
                mirror.0,
                mirror.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — fft2 / ifft2 round-trip on 2D inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft2_ifft2_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (12, 7), (3, 5)] {
        let n = rows * cols;
        let x = random_complex(n, 0xF00D_1234 + (rows * 1000 + cols) as u64);
        let xf = fft2(&x, (rows, cols), &opts).unwrap();
        let xb = ifft2(&xf, (rows, cols), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR8 fft2 round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close_complex(*got, *want),
                "MR8 fft2 round-trip rows={rows} cols={cols} i={i}: ({}, {}) vs ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — rfft2 / irfft2 round-trip on real 2D inputs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft2_irfft2_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (4, 8), (12, 16)] {
        let n = rows * cols;
        let x = random_real(n, 0xBEEF_3456 + (rows * 1000 + cols) as u64);
        let xf = rfft2(&x, (rows, cols), &opts).unwrap();
        let xb = irfft2(&xf, (rows, cols), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR9 rfft2 round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR9 rfft2 round-trip rows={rows} cols={cols} i={i}: {got} vs {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — Parseval's theorem in 2D for fft2 with Backward normalization:
// Σ|x|² = Σ|X|² / (rows * cols).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft2_parseval() {
    let opts = FftOptions::default().with_normalization(Normalization::Backward);
    for &(rows, cols) in &[(8_usize, 8), (4, 16), (12, 6)] {
        let n = rows * cols;
        let x = random_complex(n, 0xDADA_FACE + (rows * 1000 + cols) as u64);
        let xf = fft2(&x, (rows, cols), &opts).unwrap();
        let lhs: f64 = x.iter().map(|(re, im)| re * re + im * im).sum();
        let rhs: f64 = xf.iter().map(|(re, im)| re * re + im * im).sum::<f64>() / n as f64;
        assert_close(
            lhs,
            rhs,
            &format!("MR10 fft2 Parseval rows={rows} cols={cols}"),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — rfft output length is N/2 + 1 and matches fft Hermitian half.
//
// For a real input of length N:
//   rfft(x).len() == N/2 + 1
// and the rfft output equals the first N/2+1 entries of fft of the
// complex-promoted input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_matches_fft_hermitian_half() {
    let opts = FftOptions::default();
    for &n in &[2_usize, 4, 8, 16, 17, 32, 33, 64, 100] {
        let x_real = random_real(n, 0xCAFE_DEAD + n as u64);
        let r = rfft(&x_real, &opts).unwrap();
        let expected_len = n / 2 + 1;
        assert_eq!(
            r.len(),
            expected_len,
            "MR11 rfft length n={n}: got {}, expected {expected_len}",
            r.len()
        );

        // Compare to fft(complex(x))[..N/2+1].
        let x_complex: Vec<Complex64> = x_real.iter().map(|&v| (v, 0.0)).collect();
        let f = fft(&x_complex, &opts).unwrap();
        for k in 0..expected_len {
            assert!(
                close_complex(r[k], f[k]),
                "MR11 rfft[{k}] vs fft[{k}] at n={n}: ({}, {}) vs ({}, {})",
                r[k].0,
                r[k].1,
                f[k].0,
                f[k].1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — dctn / idctn 2D round-trip preserves the input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dctn_idctn_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (12, 8)] {
        let n = rows * cols;
        let x = random_real(n, 0xC1F2_3344 + (rows * 1000 + cols) as u64);
        let xf = dctn(&x, &[rows, cols], &opts).unwrap();
        let xb = idctn(&xf, &[rows, cols], &opts).unwrap();
        assert_eq!(xb.len(), n, "MR12 dctn round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR12 dctn round-trip rows={rows} cols={cols} i={i}: {got} vs {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — dstn / idstn 2D round-trip preserves the input.
// (Was previously failing: frankenscipy-oxl2 fixed the missing
// per-axis 1/(2N) factor in apply_dst_along_axis for Backward mode.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dstn_idstn_roundtrip() {
    let opts = FftOptions::default();
    for &(rows, cols) in &[(4_usize, 4), (8, 4), (16, 16), (12, 8)] {
        let n = rows * cols;
        let x = random_real(n, 0xD571_4567 + (rows * 1000 + cols) as u64);
        let xf = dstn(&x, &[rows, cols], &opts).unwrap();
        let xb = idstn(&xf, &[rows, cols], &opts).unwrap();
        assert_eq!(xb.len(), n, "MR13 dstn round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x).enumerate() {
            assert!(
                close(*got, *want),
                "MR13 dstn round-trip rows={rows} cols={cols} i={i}: {got} vs {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — hilbert preserves the input mean: Re(x_a).mean() == x.mean()
// since the real part of the analytic signal is x itself.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_preserves_mean() {
    let opts = FftOptions::default();
    for &n in &[16_usize, 32, 64, 100] {
        let x = random_real(n, 0xAB12_CD34 + n as u64);
        let analytic = hilbert(&x, &opts).unwrap();
        let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
        let re_mean: f64 = analytic.iter().map(|(re, _)| re).sum::<f64>() / n as f64;
        assert_close(re_mean, x_mean, &format!("MR14 hilbert mean n={n}"));
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — ihfft / hfft round-trip on a real input (Hermitian-symmetric
// path).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hfft_ihfft_roundtrip() {
    let opts = FftOptions::default();
    for &n in &[8_usize, 16, 32, 64] {
        let x_real = random_real(n, 0xDEAD_C0DE + n as u64);
        let xf = ihfft(&x_real, None, &opts).unwrap();
        let xb = hfft(&xf, Some(n), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR15 hfft round-trip length");
        for (i, (got, want)) in xb.iter().zip(&x_real).enumerate() {
            assert!(
                close(*got, *want),
                "MR15 hfft round-trip n={n} i={i}: {got} vs {want}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — ifftshift_1d ∘ fftshift_1d = identity for any length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftshift_ifftshift_identity() {
    for n in [4usize, 5, 8, 9, 16, 17, 32] {
        let mut rng = prng(n as u64 * 7);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let shifted = fftshift_1d(&x);
        let recovered = ifftshift_1d(&shifted);
        for (i, (a, b)) in recovered.iter().zip(&x).enumerate() {
            assert!(
                close(*a, *b),
                "MR16 ifftshift∘fftshift n={n} i={i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — For odd n, Σ fftfreq(n, d=1) = 0: positive and negative
// frequency bins pair up symmetrically (no Nyquist bin for odd n).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftfreq_odd_sum_is_zero() {
    for n in [3usize, 5, 7, 9, 17, 33, 65] {
        let f = fftfreq(n, 1.0).expect("fftfreq");
        let s: f64 = f.iter().sum();
        assert!(s.abs() < 1e-12, "MR17 Σfftfreq(n={n}) = {s}, expected 0");
        assert_eq!(f.len(), n, "MR17 fftfreq length n={n}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — dst_iii inverts dst_ii up to a 1/(2N) factor with Backward
// normalisation: ½N · idst_iii(dst_ii(x)) = x at every interior sample.
// (DST-II and DST-III are mutual inverses up to scaling under SciPy's
// default `Backward` convention.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dst_ii_dst_iii_roundtrip_backward() {
    let mut rng = prng(0x4444);
    let n = 16;
    let x: Vec<f64> = (0..n).map(|_| rng()).collect();
    let opts = FftOptions {
        normalization: Normalization::Backward,
        ..Default::default()
    };
    let y = dst_ii(&x, &opts).unwrap();
    let z = dst_iii(&y, &opts).unwrap();
    let scale = 1.0 / (2.0 * n as f64);
    for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
        let recovered = scale * zi;
        assert!(
            close(recovered, *xi),
            "MR18 dst_ii→dst_iii at i={i}: {recovered} vs {xi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — fft of a real, even-symmetric input has zero imaginary part
// (a known DFT symmetry: real-even input ⇒ real-even spectrum).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_real_even_input_has_real_spectrum() {
    // Construct a real, even-symmetric vector x[k] = cos(2π·k/N) sampled
    // for N=16 — even by construction.
    let n = 16;
    let x_real: Vec<f64> = (0..n)
        .map(|k| (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos())
        .collect();
    let x_complex: Vec<Complex64> = x_real.iter().map(|&v| (v, 0.0)).collect();
    let opts = FftOptions::default();
    let xf = fft(&x_complex, &opts).unwrap();
    for (k, &(re, im)) in xf.iter().enumerate() {
        assert!(
            im.abs() < 1e-9,
            "MR19 imag(fft[{k}]) = {im} for real-even input (re = {re})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Linearity of dct: dct(αx + βy) = α·dct(x) + β·dct(y).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_linearity() {
    let mut rng = prng(0x5555);
    let n = 32;
    let x: Vec<f64> = (0..n).map(|_| rng()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng()).collect();
    let alpha = 1.7_f64;
    let beta = -0.4_f64;
    let mixed: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(&xi, &yi)| alpha * xi + beta * yi)
        .collect();
    let opts = FftOptions::default();
    let dct_mixed = dct(&mixed, &opts).unwrap();
    let dct_x = dct(&x, &opts).unwrap();
    let dct_y = dct(&y, &opts).unwrap();
    for k in 0..n {
        let combined = alpha * dct_x[k] + beta * dct_y[k];
        assert!(
            close(dct_mixed[k], combined),
            "MR20 dct linearity at k={k}: {} vs {}",
            dct_mixed[k],
            combined
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — fft of an all-zero input is all zero; ifft recovers zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_zero_input_zero_spectrum() {
    let n = 16;
    let zeros = vec![(0.0_f64, 0.0_f64); n];
    let opts = FftOptions::default();
    let xf = fft(&zeros, &opts).unwrap();
    for (k, &(re, im)) in xf.iter().enumerate() {
        assert!(
            re.abs() < 1e-12 && im.abs() < 1e-12,
            "MR21 fft(0)[{k}] = ({re}, {im}), expected (0, 0)"
        );
    }
    let xb = ifft(&xf, &opts).unwrap();
    for (k, &(re, im)) in xb.iter().enumerate() {
        assert!(
            re.abs() < 1e-12 && im.abs() < 1e-12,
            "MR21 ifft(0)[{k}] = ({re}, {im}), expected (0, 0)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — next_fast_len(n) is at least n for any n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_next_fast_len_dominates_input() {
    for n in [0usize, 1, 2, 3, 5, 7, 11, 13, 17, 23, 100, 1000] {
        let m = next_fast_len(n, false);
        assert!(m >= n, "MR22 next_fast_len({n}) = {m} < {n}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — DCT-I is its own inverse up to a scale factor of 2(N-1):
// dct_i(dct_i(x)) ≈ 2(N-1) · x for N ≥ 2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_i_self_inverse_with_scale() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16] {
        let mut rng = prng((n * 31) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = dct_i(&x, &opts).unwrap();
        let z = dct_i(&y, &opts).unwrap();
        let scale = 2.0 * (n - 1) as f64;
        for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
            let recovered = zi / scale;
            assert!(
                close(recovered, *xi),
                "MR23 dct_i self-inverse n={n} i={i}: {recovered} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — DCT-IV is its own inverse up to a scale of 2N:
// dct_iv(dct_iv(x)) ≈ 2N · x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_iv_self_inverse_with_scale() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16] {
        let mut rng = prng((n * 17) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = dct_iv(&x, &opts).unwrap();
        let z = dct_iv(&y, &opts).unwrap();
        let scale = 2.0 * n as f64;
        for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
            let recovered = zi / scale;
            assert!(
                close(recovered, *xi),
                "MR24 dct_iv self-inverse n={n} i={i}: {recovered} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — fht then ifht with matching offset round-trips a smooth
// positive input within tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fht_ifht_roundtrip() {
    let opts = FftOptions::default();
    let n = 32;
    let dln = 0.05_f64;
    let mu = 0.0_f64;
    let bias = 0.0_f64;
    let offset = fhtoffset(dln, mu, 0.0, bias);
    // Smooth positive input (Gaussian-like).
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let t = ((i as f64) - (n as f64) / 2.0) * 0.2;
            (-t * t).exp()
        })
        .collect();
    let y = fht(&x, dln, mu, offset, bias, &opts).unwrap();
    let z = ifht(&y, dln, mu, offset, bias, &opts).unwrap();
    for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
        assert!(
            (zi - xi).abs() < 1e-6,
            "MR25 fht roundtrip at i={i}: {zi} vs {xi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — Length of rfft(x) for real input of length N is ⌊N/2⌋ + 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_output_length() {
    let opts = FftOptions::default();
    for n in [4usize, 5, 8, 9, 16, 17, 32] {
        let mut rng = prng(n as u64 + 1);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = rfft(&x, &opts).unwrap();
        assert_eq!(
            y.len(),
            n / 2 + 1,
            "MR26 rfft length n={n}: got {} expected {}",
            y.len(),
            n / 2 + 1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — fft is invertible: ifft(fft(x)) = x for any complex input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_ifft_inverse_complex_input() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32] {
        let mut rng = prng(n as u64 + 99);
        let x: Vec<Complex64> = (0..n).map(|_| (rng(), rng())).collect();
        let xf = fft(&x, &opts).unwrap();
        let xb = ifft(&xf, &opts).unwrap();
        for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
            assert!(
                close_complex(*a, *b),
                "MR27 ifft(fft(x)) at i={i}: ({},{}) vs ({},{})",
                a.0,
                a.1,
                b.0,
                b.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — fftn with shape [N] reduces to fft on the same input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftn_1d_matches_fft() {
    let opts = FftOptions::default();
    let n = 16;
    let mut rng = prng(0x6789);
    let x: Vec<Complex64> = (0..n).map(|_| (rng(), rng())).collect();
    let one_d = fft(&x, &opts).unwrap();
    let nd = fftn(&x, &[n], &opts).unwrap();
    assert_eq!(one_d.len(), nd.len(), "MR28 length");
    for (i, (a, b)) in one_d.iter().zip(&nd).enumerate() {
        assert!(
            close_complex(*a, *b),
            "MR28 fft vs fftn[1D] at {i}: ({}, {}) vs ({}, {})",
            a.0,
            a.1,
            b.0,
            b.1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — ifftn(fftn(x)) = x for any complex N-D input shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftn_ifftn_inverse() {
    let opts = FftOptions::default();
    let shapes: &[&[usize]] = &[&[8], &[4, 4], &[2, 2, 4]];
    for &shape in shapes {
        let total: usize = shape.iter().product();
        let mut rng = prng(total as u64 * 31);
        let x: Vec<Complex64> = (0..total).map(|_| (rng(), rng())).collect();
        let xf = fftn(&x, shape, &opts).unwrap();
        let xb = ifftn(&xf, shape, &opts).unwrap();
        for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
            assert!(
                close_complex(*a, *b),
                "MR29 ifftn∘fftn shape={shape:?} at {i}: ({}, {}) vs ({}, {})",
                a.0,
                a.1,
                b.0,
                b.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — hilbert returns a complex output of the same length as input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hilbert_preserves_length() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32, 64] {
        let mut rng = prng(n as u64 * 7);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let h = hilbert(&x, &opts).unwrap();
        assert_eq!(h.len(), n, "MR30 hilbert length");
        for (i, &(re, im)) in h.iter().enumerate() {
            assert!(
                re.is_finite() && im.is_finite(),
                "MR30 hilbert non-finite at {i}: ({re}, {im})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — fftcorrelate(x, x, "full") peaks at the centre lag (zero-lag
// = energy of x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftcorrelate_peak_at_zero_lag() {
    let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.4).cos()).collect();
    let c = fftcorrelate(&x, &x, "full").unwrap();
    // For "full" mode of correlate(x, x), output length is 2N - 1 with
    // peak at index N - 1.
    let mid = x.len() - 1;
    let energy: f64 = x.iter().map(|v| v * v).sum();
    assert_eq!(c.len(), 2 * x.len() - 1, "MR31 fftcorrelate length");
    assert!(
        (c[mid] - energy).abs() < 1e-7 * energy.abs().max(1.0),
        "MR31 fftcorrelate centre = {} vs energy = {energy}",
        c[mid]
    );
    // Peak is the maximum.
    for (k, &v) in c.iter().enumerate() {
        if k != mid {
            assert!(
                v.abs() <= c[mid].abs() + 1e-7,
                "MR31 fftcorrelate[{k}] = {v} > peak = {}",
                c[mid]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — hamming_window and hann_window are symmetric on odd length
// and produce values in [0, 1.0001] (small headroom for hamming).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hamming_hann_windows_symmetric() {
    for n in [5usize, 7, 13, 31, 65] {
        let h = hamming_window(n);
        let hn = hann_window(n);
        for i in 0..n / 2 {
            assert!(
                (h[i] - h[n - 1 - i]).abs() < 1e-12,
                "MR32 hamming not symmetric at i={i}: {} vs {}",
                h[i],
                h[n - 1 - i]
            );
            assert!(
                (hn[i] - hn[n - 1 - i]).abs() < 1e-12,
                "MR32 hann not symmetric at i={i}: {} vs {}",
                hn[i],
                hn[n - 1 - i]
            );
        }
        for &v in &h {
            assert!((-1e-12..=1.001).contains(&v), "MR32 hamming val = {v}");
        }
        for &v in &hn {
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "MR32 hann val = {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — fftshift_1d on an even-length vector swaps the two halves.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftshift_swaps_even_halves() {
    let n = 8;
    let mut rng = prng(0x9999);
    let x: Vec<f64> = (0..n).map(|_| rng()).collect();
    let s = fftshift_1d(&x);
    assert_eq!(s.len(), n, "MR33 fftshift length");
    let half = n / 2;
    for i in 0..half {
        // After shift: s[i] = x[i + half]; s[i + half] = x[i].
        assert!(
            (s[i] - x[i + half]).abs() < 1e-12,
            "MR33 fftshift first half at {i}: {} vs {}",
            s[i],
            x[i + half]
        );
        assert!(
            (s[i + half] - x[i]).abs() < 1e-12,
            "MR33 fftshift second half at {}: {} vs {}",
            i + half,
            s[i + half],
            x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — DST-I is its own inverse up to scale 2(N+1):
// dst_i(dst_i(x)) ≈ 2(N+1) · x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dst_i_self_inverse_with_scale() {
    let opts = FftOptions::default();
    for n in [4usize, 7, 12, 16] {
        let mut rng = prng((n * 41) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = dst_i(&x, &opts).unwrap();
        let z = dst_i(&y, &opts).unwrap();
        let scale = 2.0 * (n + 1) as f64;
        for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
            let recovered = zi / scale;
            assert!(
                close(recovered, *xi),
                "MR34 dst_i self-inverse n={n} i={i}: {recovered} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — DST-IV is its own inverse up to scale 2N:
// dst_iv(dst_iv(x)) ≈ 2N · x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dst_iv_self_inverse_with_scale() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16] {
        let mut rng = prng((n * 23) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = dst_iv(&x, &opts).unwrap();
        let z = dst_iv(&y, &opts).unwrap();
        let scale = 2.0 * n as f64;
        for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
            let recovered = zi / scale;
            assert!(
                close(recovered, *xi),
                "MR35 dst_iv self-inverse n={n} i={i}: {recovered} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — DCT-II and DCT-III are mutual inverses up to scale 2N
// (Backward normalisation): dct_iii(dct_ii(x)) ≈ 2N · x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dct_ii_dct_iii_mutual_inverse() {
    let opts = FftOptions {
        normalization: Normalization::Backward,
        ..Default::default()
    };
    for n in [4usize, 8, 16] {
        let mut rng = prng((n * 19) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = dct(&x, &opts).unwrap();
        let z = dct_iii(&y, &opts).unwrap();
        let scale = 2.0 * n as f64;
        for (i, (zi, xi)) in z.iter().zip(&x).enumerate() {
            let recovered = zi / scale;
            assert!(
                close(recovered, *xi),
                "MR36 dct_ii→dct_iii at n={n} i={i}: {recovered} vs {xi}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — idctn ∘ dctn = id on a smooth N-D signal under default options.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dctn_idctn_inverse_default() {
    let opts = FftOptions::default();
    let shape = [4usize, 4];
    let total: usize = shape.iter().product();
    let mut rng = prng(0xabcd);
    let x: Vec<f64> = (0..total).map(|_| rng()).collect();
    let y = dctn(&x, &shape, &opts).unwrap();
    let z = idctn(&y, &shape, &opts).unwrap();
    for (i, (a, b)) in x.iter().zip(&z).enumerate() {
        assert!(close(*a, *b), "MR37 dctn∘idctn at {i}: {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — idstn ∘ dstn = id on a smooth N-D signal under default options.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dstn_idstn_inverse_default() {
    let opts = FftOptions::default();
    let shape = [4usize, 4];
    let total: usize = shape.iter().product();
    let mut rng = prng(0xfeed);
    let x: Vec<f64> = (0..total).map(|_| rng()).collect();
    let y = dstn(&x, &shape, &opts).unwrap();
    let z = idstn(&y, &shape, &opts).unwrap();
    for (i, (a, b)) in x.iter().zip(&z).enumerate() {
        assert!(close(*a, *b), "MR38 dstn∘idstn at {i}: {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — ifftshift_1d ∘ fftshift_1d = id on odd-length input (the
// non-trivial parity case).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fftshift_odd_length_identity() {
    for n in [3usize, 5, 7, 9, 17] {
        let mut rng = prng((n * 11) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let s = fftshift_1d(&x);
        let r = ifftshift_1d(&s);
        for (i, (a, b)) in r.iter().zip(&x).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "MR39 fftshift∘ifftshift n={n} at {i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — irfftn(rfftn(x, shape), shape) = x for real-valued multi-D input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfftn_irfftn_inverse() {
    let opts = FftOptions::default();
    let shape = [4usize, 4];
    let total: usize = shape.iter().product();
    let mut rng = prng(0x1234);
    let x: Vec<f64> = (0..total).map(|_| rng()).collect();
    let xf = rfftn(&x, &shape, &opts).unwrap();
    let xb = irfftn(&xf, &shape, &opts).unwrap();
    for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
        assert!((a - b).abs() < 1e-9, "MR40 irfftn∘rfftn at {i}: {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — fft of unit impulse [1, 0, 0, …, 0] is the all-ones spectrum
// (every bin gets equal contribution from the impulse).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_impulse_spectrum_all_ones() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32] {
        let mut x = vec![(0.0_f64, 0.0_f64); n];
        x[0] = (1.0, 0.0);
        let xf = fft(&x, &opts).unwrap();
        for (k, &(re, im)) in xf.iter().enumerate() {
            assert!(
                (re - 1.0).abs() < 1e-12 && im.abs() < 1e-12,
                "MR41 fft(impulse)[{k}] = ({re}, {im}), expected (1, 0)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — Forward and Ortho normalisation roundtrips also yield x:
// for any normalisation, ifft(fft(x)) = x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_ifft_normalization_variants() {
    for &norm in &[
        Normalization::Backward,
        Normalization::Forward,
        Normalization::Ortho,
    ] {
        let opts = FftOptions {
            normalization: norm,
            ..Default::default()
        };
        let n = 16;
        let mut rng = prng(0x4321);
        let x: Vec<Complex64> = (0..n).map(|_| (rng(), rng())).collect();
        let xf = fft(&x, &opts).unwrap();
        let xb = ifft(&xf, &opts).unwrap();
        for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
            assert!(
                close_complex(*a, *b),
                "MR42 norm={norm:?} ifft∘fft at {i}: ({}, {}) vs ({}, {})",
                a.0,
                a.1,
                b.0,
                b.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — ihfft is the inverse of hfft on real-symmetric (Hermitian)
// input length n. (hfft(ihfft(x)) returns x within tolerance.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hfft_ihfft_normalisation_default() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16] {
        let mut rng = prng((n * 71) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let xf = ihfft(&x, None, &opts).unwrap();
        let xb = hfft(&xf, Some(n), &opts).unwrap();
        for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "MR43 hfft∘ihfft n={n} at {i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — rfft of length-N input has output length ⌊N/2⌋ + 1 and
// the DC bin is real-valued (imag = 0) for any real input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_dc_bin_real() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32] {
        let mut rng = prng((n * 19) as u64);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = rfft(&x, &opts).unwrap();
        assert_eq!(y.len(), n / 2 + 1, "MR44 rfft length");
        let (_re, im) = y[0];
        assert!(im.abs() < 1e-12, "MR44 rfft DC imag = {im} for real input");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — fft of constant input [c, c, c, …, c] has only the DC bin
// non-zero: spectrum[0] = N · c, spectrum[k] = 0 for k > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft_constant_input_only_dc_nonzero() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16] {
        let c = 2.5_f64;
        let x: Vec<Complex64> = vec![(c, 0.0); n];
        let xf = fft(&x, &opts).unwrap();
        // DC bin should equal n*c.
        assert!(
            (xf[0].0 - (n as f64) * c).abs() < 1e-9,
            "MR45 fft(const) DC = {} expected {}",
            xf[0].0,
            (n as f64) * c
        );
        assert!(
            xf[0].1.abs() < 1e-9,
            "MR45 fft(const) DC imag = {}",
            xf[0].1
        );
        for (k, val) in xf.iter().enumerate().take(n).skip(1) {
            assert!(
                val.0.abs() < 1e-9 && val.1.abs() < 1e-9,
                "MR45 fft(const)[{k}] = ({}, {}), expected (0, 0)",
                val.0,
                val.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — rfft of all-ones input has DC bin equal to N.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft_all_ones_dc_bin_is_n() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32] {
        let x: Vec<f64> = vec![1.0; n];
        let y = rfft(&x, &opts).unwrap();
        assert!(
            (y[0].0 - n as f64).abs() < 1e-9,
            "MR46 rfft(ones) DC = {} expected {}",
            y[0].0,
            n as f64
        );
        for (k, val) in y.iter().enumerate().skip(1) {
            assert!(
                val.0.abs() < 1e-9 && val.1.abs() < 1e-9,
                "MR46 rfft(ones)[{k}] = ({}, {})",
                val.0,
                val.1
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — fft2 of all-zero 2D input is all zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fft2_zero_input_zero_spectrum() {
    let opts = FftOptions::default();
    let rows = 4;
    let cols = 4;
    let x: Vec<Complex64> = vec![(0.0, 0.0); rows * cols];
    let y = fft2(&x, (rows, cols), &opts).unwrap();
    for &(re, im) in &y {
        assert!(re.abs() < 1e-12 && im.abs() < 1e-12, "MR47 fft2(0) ≠ 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — ifft2 of all-zero input is all zero.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ifft2_zero_input_zero_spectrum() {
    let opts = FftOptions::default();
    let rows = 4;
    let cols = 4;
    let x: Vec<Complex64> = vec![(0.0, 0.0); rows * cols];
    let y = ifft2(&x, (rows, cols), &opts).unwrap();
    for &(re, im) in &y {
        assert!(re.abs() < 1e-12 && im.abs() < 1e-12, "MR48 ifft2(0) ≠ 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — rfft2 returns expected output length: rows × (cols/2 + 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rfft2_output_length() {
    let opts = FftOptions::default();
    for &(r, c) in &[(4usize, 4usize), (4, 8), (8, 4), (16, 16)] {
        let mut rng = prng((r * c) as u64);
        let x: Vec<f64> = (0..r * c).map(|_| rng()).collect();
        let y = rfft2(&x, (r, c), &opts).unwrap();
        let expected = r * (c / 2 + 1);
        assert_eq!(
            y.len(),
            expected,
            "MR49 rfft2(r={r}, c={c}) length = {} expected {expected}",
            y.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — irfft of length-(N/2 + 1) output has length N (with explicit
// n parameter).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_irfft_explicit_length() {
    let opts = FftOptions::default();
    for n in [4usize, 8, 16, 32] {
        let mut rng = prng(n as u64 + 13);
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let y = rfft(&x, &opts).unwrap();
        let xb = irfft(&y, Some(n), &opts).unwrap();
        assert_eq!(xb.len(), n, "MR50 irfft length");
        for (i, (a, b)) in x.iter().zip(&xb).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "MR50 irfft round-trip n={n} at {i}: {a} vs {b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — dctn shape mismatch returns an error (input length must match
// product of shape).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dctn_shape_mismatch_errors() {
    let opts = FftOptions::default();
    let x: Vec<f64> = vec![1.0; 10];
    let result = dctn(&x, &[4, 4], &opts); // 16 expected, 10 supplied
    assert!(
        result.is_err(),
        "MR51 dctn(input_len=10, shape=[4,4]) should error"
    );
}
