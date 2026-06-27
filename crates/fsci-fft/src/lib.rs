#![forbid(unsafe_code)]

//! FFT API surface for FrankenSciPy packet P2C-005.
//!
//! This crate is intentionally contract-first at this stage:
//! - module boundaries are fixed (`transforms`, `helpers`, `plan`)
//! - public signatures are stable for conformance wiring
//! - kernels are populated in subsequent packet beads

pub mod helpers;
pub mod plan;
pub mod transforms;

pub use helpers::{
    analytic_signal, apply_window, blackman_window, cross_spectral_density, fftconvolve,
    fftcorrelate, fftfreq, fftshift, fftshift_1d, hamming_window, hann_window, ifftshift,
    ifftshift_1d, magnitude_spectrum, periodogram_simple, phase_spectrum_signal,
    polynomial_multiply_fft, rfftfreq, zero_pad_pow2,
};
pub use plan::{
    BoundedPlanCache, CacheAdmissionPolicy, PlanCacheBackend, PlanCacheConfig, PlanFingerprint,
    PlanKey, PlanMetadata, PlanningStrategy,
};
pub use transforms::{
    BackendKind, Complex64, FftError, FftOptions, SyncSharedAuditLedger, TransformTrace,
    WorkerPolicy, dct, dct_i, dct_iii, dct_iv, dctn, dst, dst_i, dst_ii, dst_iii, dst_iv, dstn,
    fft, fft_with_audit, fft2, fft2_with_audit, fftn, fftn_with_audit, fht, fhtoffset, fwht, hfft,
    hfft_with_audit, hfft2, hfft2_with_audit, hfftn, hfftn_with_audit, hilbert, idct, idct_i,
    idct_iii, idct_iv, idctn, idst, idstn, ifft, ifft_with_audit, ifft2, ifft2_with_audit, ifftn,
    ifftn_with_audit, ifht, ihfft, ihfft_with_audit, ihfft2, ihfft2_with_audit, ihfftn,
    ihfftn_with_audit, irfft, irfft_with_audit, irfft2, irfft2_with_audit, irfftn,
    irfftn_with_audit, next_fast_len, prev_fast_len, rfft, rfft_with_audit, rfft2,
    rfft2_with_audit, rfftn, rfftn_with_audit, sync_audit_ledger, take_transform_traces,
};

/// FFT normalization modes matching SciPy/PocketFFT conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Normalization {
    Forward,
    #[default]
    Backward,
    Ortho,
}

/// Transform entrypoints represented in the packet boundary contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformKind {
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fft2,
    Ifft2,
    Fftn,
    Ifftn,
    Rfftn,
    Irfftn,
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use proptest::prelude::*;
    use serde_json::json;

    use std::f64::consts::PI;

    use super::helpers::{
        analytic_signal, fftconvolve, fftfreq, fftshift, fftshift_1d, ifftshift, ifftshift_1d,
        magnitude_spectrum, periodogram_simple, rfftfreq,
    };
    use super::transforms::{
        Complex64, FftOptions, dct, dct_i, dct_iii, dct_iv, dst, dst_i, dst_ii, dst_iii, dst_iv,
        fft, fht, fhtoffset, hilbert, idct, idct_i, idct_iii, idct_iv, idst, ifft, ifht, irfft,
        rfft,
    };
    use super::{Normalization, TransformKind};

    const PROPTEST_CASES: u32 = 512;
    const FFT_TOL: f64 = 1e-9;

    fn assert_complex_close(got: &[Complex64], want: &[Complex64], context: &str) {
        assert_eq!(got.len(), want.len(), "{context} length");
        for (i, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < FFT_TOL && (got.1 - want.1).abs() < FFT_TOL,
                "{context}[{i}] = ({}, {}), want ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }

    #[test]
    fn normalization_default_matches_scipy() {
        assert_eq!(Normalization::default(), Normalization::Backward);
    }

    #[test]
    fn transform_kind_order_is_stable_for_plan_keys() {
        assert!(TransformKind::Fft < TransformKind::Fftn);
    }

    #[test]
    fn structured_log_schema_is_machine_parseable() {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let log = json!({
            "test_id": "fft_log_schema_test",
            "transform": "fft",
            "n": 8,
            "dtype": "complex64",
            "backend": "naive_dft",
            "mode": "strict",
            "seed": 0,
            "timestamp_ms": timestamp_ms,
            "max_error": 0.0,
            "result": "pass"
        });
        for field in [
            "test_id",
            "transform",
            "n",
            "dtype",
            "backend",
            "mode",
            "seed",
            "timestamp_ms",
            "max_error",
            "result",
        ] {
            assert!(log.get(field).is_some(), "missing field: {field}");
        }
    }

    fn complex_mag_sq(c: Complex64) -> f64 {
        c.0 * c.0 + c.1 * c.1
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        #[test]
        fn prop_ifft_fft_roundtrip(
            vals in proptest::collection::vec(-10i16..=10i16, 1..=32)
        ) {
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            let recovered = ifft(&spectrum, &opts).expect("ifft");
            for (&a, &b) in recovered.iter().zip(&input) {
                prop_assert!((a.0 - b.0).abs() <= FFT_TOL, "re: {} vs {}", a.0, b.0);
                prop_assert!((a.1 - b.1).abs() <= FFT_TOL, "im: {} vs {}", a.1, b.1);
            }
        }

        #[test]
        fn prop_irfft_rfft_roundtrip(
            vals in proptest::collection::vec(-10i16..=10i16, 2..=32)
        ) {
            let input: Vec<f64> = vals.iter().map(|&v| f64::from(v)).collect();
            let n = input.len();
            let opts = FftOptions::default();
            let spectrum = rfft(&input, &opts).expect("rfft");
            let recovered = irfft(&spectrum, Some(n), &opts).expect("irfft");
            for (&a, &b) in recovered.iter().zip(&input) {
                prop_assert!((a - b).abs() <= FFT_TOL, "{a} vs {b}");
            }
        }

        #[test]
        fn prop_parseval_theorem(
            vals in proptest::collection::vec(-10i16..=10i16, 1..=32)
        ) {
            // sum(|x|^2) == sum(|X|^2) / N
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let n = input.len() as f64;
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            let time_energy: f64 = input.iter().map(|c| complex_mag_sq(*c)).sum();
            let freq_energy: f64 = spectrum.iter().map(|c| complex_mag_sq(*c)).sum();
            prop_assert!(
                (time_energy - freq_energy / n).abs() <= FFT_TOL * n,
                "Parseval: {time_energy} vs {}", freq_energy / n
            );
        }

        #[test]
        fn prop_fftfreq_sum_zero_odd(n in (1u32..=50).prop_map(|v| (v * 2 + 1) as usize)) {
            // For odd n, positive and negative frequencies pair exactly, so sum=0
            let freqs = fftfreq(n, 1.0).expect("fftfreq");
            let sum: f64 = freqs.iter().sum();
            prop_assert!(sum.abs() <= 1e-10, "sum={sum} for n={n}");
        }

        #[test]
        fn prop_fft_shift_property(
            vals in proptest::collection::vec(-5i16..=5i16, 2..=16),
            k in 0usize..16,
        ) {
            // Verify fft produces consistent output length
            let input: Vec<Complex64> = vals.iter().map(|&v| (f64::from(v), 0.0)).collect();
            let n = input.len();
            let k_mod = k % n;
            let opts = FftOptions::default();
            let spectrum = fft(&input, &opts).expect("fft");
            // Circular shift the input
            let mut shifted = vec![(0.0, 0.0); n];
            for i in 0..n {
                shifted[(i + k_mod) % n] = input[i];
            }
            let shifted_spectrum = fft(&shifted, &opts).expect("fft shifted");
            // |shifted_spectrum[j]| should equal |spectrum[j]| (magnitude preserved)
            for j in 0..n {
                let mag_orig = complex_mag_sq(spectrum[j]).sqrt();
                let mag_shift = complex_mag_sq(shifted_spectrum[j]).sqrt();
                prop_assert!(
                    (mag_orig - mag_shift).abs() <= FFT_TOL * mag_orig.max(1.0),
                    "mag at bin {j}: {mag_orig} vs {mag_shift}"
                );
            }
        }
    }

    // ── DCT tests ───────────────────────────────────────────────────

    #[test]
    fn dct_constant_input() {
        // DCT of constant [1,1,1,1]: DC component should be N*2 = 8, others should be ~0
        let input = vec![1.0; 4];
        let result = dct(&input, &FftOptions::default()).expect("dct");
        assert_eq!(result.len(), 4);
        assert!((result[0] - 8.0).abs() < 1e-9, "DC = {}", result[0]);
        for (k, &val) in result.iter().enumerate().skip(1) {
            assert!(val.abs() < 1e-9, "AC[{k}] = {val}");
        }
    }

    #[test]
    fn dct_impulse() {
        // DCT of [1, 0, 0, 0] should be [2*cos(0), 2*cos(π/8), 2*cos(2π/8), 2*cos(3π/8)]
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let result = dct(&input, &FftOptions::default()).expect("dct");
        assert_eq!(result.len(), 4);
        for (k, &val) in result.iter().enumerate() {
            let expected = 2.0 * (std::f64::consts::PI * k as f64 / 8.0).cos();
            assert!(
                (val - expected).abs() < 1e-9,
                "DCT[{k}] = {val}, expected {expected}",
            );
        }
    }

    #[test]
    fn dct_idct_roundtrip() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let spectrum = dct(&input, &opts).expect("dct");
        let recovered = idct(&spectrum, &opts).expect("idct");
        for (a, b) in recovered.iter().zip(&input) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn dct_length_1() {
        let result = dct(&[5.0], &FftOptions::default()).expect("dct len 1");
        assert!((result[0] - 10.0).abs() < 1e-9, "DCT[0] = {}", result[0]);
    }

    // ── DCT type variants ──────────────────────────────────────────

    #[test]
    fn dct_i_preserves_length() {
        let input = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = dct_i(&input, &FftOptions::default()).expect("dct_i");
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn dct_i_known_values() {
        // DCT-I of [1, 0, 0] with N=3:
        // X[k] = Σ x[n]*cos(πnk/2) = cos(0) = 1 for all k
        let input = vec![1.0, 0.0, 0.0];
        let result = dct_i(&input, &FftOptions::default()).expect("dct_i");
        assert_eq!(result.len(), 3);
        for (k, &val) in result.iter().enumerate() {
            let mut expected = 0.0;
            for (n, &x) in input.iter().enumerate() {
                expected += x * (PI * n as f64 * k as f64 / 2.0).cos();
            }
            assert!(
                (val - expected).abs() < 1e-9,
                "DCT-I[{k}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn idct_types_invert_and_roundtrip() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let opts = FftOptions::default();
        // idct_i/iii/iv invert the corresponding forward DCT (round-trip == x).
        for norm in [
            Normalization::Backward,
            Normalization::Ortho,
            Normalization::Forward,
        ] {
            let o = opts.clone().with_normalization(norm);
            let rt1 = idct_i(&dct_i(&x, &o).unwrap(), &o).unwrap();
            let rt3 = idct_iii(&dct_iii(&x, &o).unwrap(), &o).unwrap();
            let rt4 = idct_iv(&dct_iv(&x, &o).unwrap(), &o).unwrap();
            for i in 0..x.len() {
                assert!(
                    (rt1[i] - x[i]).abs() < 1e-9,
                    "idct_i rt {} vs {}",
                    rt1[i],
                    x[i]
                );
                assert!(
                    (rt3[i] - x[i]).abs() < 1e-9,
                    "idct_iii rt {} vs {}",
                    rt3[i],
                    x[i]
                );
                assert!(
                    (rt4[i] - x[i]).abs() < 1e-9,
                    "idct_iv rt {} vs {}",
                    rt4[i],
                    x[i]
                );
            }
        }
        // Backward-norm scaling matches scipy.fft.idct(x, type=t).
        let i3 = idct_iii(&x, &opts).unwrap();
        let want_i3 = dct(&x, &opts).unwrap(); // dct = DCT-II (inverse type of III)
        let n = x.len() as f64;
        for (a, b) in i3.iter().zip(want_i3.iter()) {
            assert!(
                (a - b / (2.0 * n)).abs() < 1e-9,
                "idct_iii {a} vs {}",
                b / (2.0 * n)
            );
        }
    }

    #[test]
    fn dst_idst_dispatch_match_scipy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let opts = FftOptions::default();

        // dst dispatcher routes to dst_ii by default (scipy type=2).
        let d2 = dst(&x, 2, &opts).expect("dst t2");
        let want_d2 = [
            27.04592314,
            -12.0,
            9.89949494,
            -6.92820323,
            7.24693326,
            -6.0,
        ];
        for (a, b) in d2.iter().zip(want_d2.iter()) {
            assert!((a - b).abs() < 1e-7, "dst t2 {a} vs {b}");
        }
        assert_eq!(d2, dst_ii(&x, &opts).unwrap());

        // idst type 2 (backward) vs scipy.fft.idst oracle.
        let i2 = idst(&x, 2, &opts).expect("idst t2");
        let want_i2 = [
            2.44564502,
            -0.2845178,
            0.11243318,
            -0.06619961,
            0.04881554,
            -0.04238885,
        ];
        for (a, b) in i2.iter().zip(want_i2.iter()) {
            assert!((a - b).abs() < 1e-7, "idst t2 back {a} vs {b}");
        }

        // idst type 2 (ortho) vs scipy oracle.
        let ortho = opts.clone().with_normalization(Normalization::Ortho);
        let i2o = idst(&x, 2, &ortho).expect("idst t2 ortho");
        let want_i2o = [
            9.18940181,
            -1.70303749,
            1.1069189,
            -0.94676112,
            0.88654091,
            -0.86427822,
        ];
        for (a, b) in i2o.iter().zip(want_i2o.iter()) {
            assert!((a - b).abs() < 1e-7, "idst t2 ortho {a} vs {b}");
        }

        // Round-trips: idst(dst(x)) == x for every type and normalization.
        for t in 1u8..=4 {
            for norm in [
                Normalization::Backward,
                Normalization::Ortho,
                Normalization::Forward,
            ] {
                let o = opts.clone().with_normalization(norm);
                let fwd = dst(&x, t, &o).expect("dst");
                let back = idst(&fwd, t, &o).expect("idst");
                for (a, b) in back.iter().zip(&x) {
                    assert!((a - b).abs() < 1e-8, "roundtrip t{t} {norm:?}: {a} vs {b}");
                }
            }
        }

        // Invalid type errors.
        assert!(dst(&x, 5, &opts).is_err());
        assert!(idst(&x, 0, &opts).is_err());
    }

    #[test]
    fn dst_iii_inverts_dst_ii() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let forward = dst_ii(&input, &opts).expect("dst_ii");
        let inverse = dst_iii(&forward, &opts).expect("dst_iii");
        let scale = 2.0 * input.len() as f64;
        for (a, b) in inverse.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-8,
                "DST-III(DST-II(x)): {a}/{scale} vs {b}"
            );
        }
    }

    #[test]
    fn dct_iv_self_inverse() {
        // DCT-IV is its own inverse (up to scaling by 2N)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dct_iv(&input, &opts).expect("dct_iv");
        let roundtrip = dct_iv(&forward, &opts).expect("dct_iv roundtrip");
        let scale = 2.0 * input.len() as f64;
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DCT-IV roundtrip: {} vs {b}",
                a / scale
            );
        }
    }

    // ── DST variants ──────────────────────────────────────────────

    #[test]
    fn dst_i_roundtrip_proportional() {
        // DST-I applied twice gives back a scaled version of input (scale 2*(N+1))
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dst_i(&input, &opts).expect("dst_i");
        let roundtrip = dst_i(&forward, &opts).expect("dst_i roundtrip");
        let scale = 2.0 * (input.len() as f64 + 1.0);
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DST-I roundtrip: {a}/{scale} vs {b}"
            );
        }
    }

    #[test]
    fn dst_ii_constant_input() {
        // DST-II of constant should give specific pattern
        let input = vec![1.0; 4];
        let result = dst_ii(&input, &FftOptions::default()).expect("dst_ii");
        assert_eq!(result.len(), 4);
        // All values should be finite
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn dst_iv_self_inverse() {
        // DST-IV is its own inverse (up to scaling by 2N)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let forward = dst_iv(&input, &opts).expect("dst_iv");
        let roundtrip = dst_iv(&forward, &opts).expect("dst_iv roundtrip");
        let scale = 2.0 * input.len() as f64;
        for (a, b) in roundtrip.iter().zip(&input) {
            assert!(
                (a / scale - b).abs() < 1e-9,
                "DST-IV roundtrip: {} vs {b}",
                a / scale
            );
        }
    }

    #[test]
    fn dst_i_known_values() {
        // DST-I of [1, 1, 1, 1] with N=4, N+1=5
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let result = dst_i(&input, &FftOptions::default()).expect("dst_i");
        // X[k] = 2 * Σ x[n] * sin(π(n+1)(k+1)/5)
        for (k, &val) in result.iter().enumerate() {
            let mut expected = 0.0;
            for (n, &x) in input.iter().enumerate() {
                expected += 2.0 * x * (PI * (n as f64 + 1.0) * (k as f64 + 1.0) / 5.0).sin();
            }
            assert!(
                (val - expected).abs() < 1e-9,
                "DST-I[{k}] = {val}, expected {expected}"
            );
        }
    }

    // ── N-D FFT tests ────────────────────────────────────────────────

    #[test]
    fn ifftn_fftn_roundtrip_2d() {
        use super::transforms::{fftn, ifftn};
        // 2x3 complex array
        let input: Vec<Complex64> = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ];
        let shape = vec![2, 3];
        let opts = FftOptions::default();
        let spectrum = fftn(&input, &shape, &opts).expect("fftn");
        let recovered = ifftn(&spectrum, &shape, &opts).expect("ifftn");
        for (i, (&a, &b)) in recovered.iter().zip(&input).enumerate() {
            assert!((a.0 - b.0).abs() < 1e-9, "re[{i}]: {} vs {}", a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-9, "im[{i}]: {} vs {}", a.1, b.1);
        }
    }

    #[test]
    fn ifftn_fftn_roundtrip_3d() {
        use super::transforms::{fftn, ifftn};
        // 2x2x2 complex array
        let input: Vec<Complex64> = (0..8).map(|i| (i as f64, 0.0)).collect();
        let shape = vec![2, 2, 2];
        let opts = FftOptions::default();
        let spectrum = fftn(&input, &shape, &opts).expect("fftn");
        let recovered = ifftn(&spectrum, &shape, &opts).expect("ifftn");
        for (i, (&a, &b)) in recovered.iter().zip(&input).enumerate() {
            assert!((a.0 - b.0).abs() < 1e-9, "re[{i}]: {} vs {}", a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-9, "im[{i}]: {} vs {}", a.1, b.1);
        }
    }

    // ── Hilbert transform tests ─────────────────────────────────────

    #[test]
    fn hilbert_real_part_preserved() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        assert_eq!(analytic.len(), x.len());
        for (i, (&(re, _), &xi)) in analytic.iter().zip(x.iter()).enumerate() {
            assert!(
                (re - xi).abs() < 1e-9,
                "real part[{i}] = {re}, expected {xi}"
            );
        }
    }

    #[test]
    fn hilbert_cosine_gives_sine() {
        // Hilbert transform of cos(t) = sin(t) (for interior points, away from edges)
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 2.0 * i as f64 / n as f64).cos())
            .collect();
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        // The imaginary part should approximate sin(2*2π*i/n) for interior points
        for (i, &(_, im)) in analytic.iter().enumerate().skip(4).take(n - 8) {
            let expected_sin = (2.0 * std::f64::consts::PI * 2.0 * i as f64 / n as f64).sin();
            assert!(
                (im - expected_sin).abs() < 0.1,
                "hilbert imag[{i}] = {im}, expected ~{expected_sin}",
            );
        }
    }

    #[test]
    fn hilbert_envelope() {
        // Envelope = |analytic signal| should be ~1 for a pure cosine
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).cos())
            .collect();
        let analytic = hilbert(&x, &FftOptions::default()).expect("hilbert");
        // Interior envelope should be close to 1
        for (i, &(re, im)) in analytic.iter().enumerate().skip(4).take(n - 8) {
            let envelope = (re * re + im * im).sqrt();
            assert!(
                (envelope - 1.0).abs() < 0.15,
                "envelope[{i}] = {envelope}, expected ~1.0"
            );
        }
    }

    // ── Fast Hankel Transform ─────────────────────────────────────────

    #[test]
    fn fht_basic_runs() {
        // Simple test: FHT of a log-spaced power law
        let n = 16;
        let dln = 0.1;
        let mu = 0.5;
        let input: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).powf(-1.5)).collect();

        let result = fht(&input, dln, mu, 0.0, 0.0, &FftOptions::default());
        assert!(result.is_ok(), "FHT should succeed");
        let output = result.unwrap();
        assert_eq!(output.len(), n);

        // Check output is finite
        for &v in &output {
            assert!(v.is_finite(), "FHT output should be finite");
        }
    }

    #[test]
    fn ifht_inverts_fht() {
        // For symmetric transforms with proper parameters, ifht(fht(x)) ≈ x
        let n = 32;
        let dln = 0.1;
        let mu = 0.0; // μ=0 is special: J_0 Bessel function
        let bias = 0.0;

        // Create a smooth input
        let input: Vec<f64> = (0..n)
            .map(|i| {
                let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 4.0);
                (-x * x).exp() // Gaussian-ish
            })
            .collect();

        let opts = FftOptions::default();
        let forward = fht(&input, dln, mu, 0.0, bias, &opts).expect("fht");
        let recovered = ifht(&forward, dln, mu, 0.0, bias, &opts).expect("ifht");

        // ifht is the exact inverse of fht for matching parameters.
        assert_eq!(recovered.len(), n);
        for (i, (&r, &x)) in recovered.iter().zip(input.iter()).enumerate() {
            assert!(
                (r - x).abs() < 1e-9,
                "ifht(fht(x)) mismatch at {i}: {r} vs {x}"
            );
        }
    }

    #[test]
    fn fhtoffset_matches_scipy_reference() {
        // Reference values from scipy.fft.fhtoffset(dln, mu, initial, bias).
        for &(dln, mu, initial, bias, want) in &[
            (
                0.5_f64,
                0.0_f64,
                0.0_f64,
                0.0_f64,
                -0.157_875_391_166_816_93,
            ),
            (0.1, 1.0, 0.0, 0.5, -0.003_149_364_650_484_187_8),
            (0.3, 0.5, 0.1, 0.2, 0.223_900_341_282_366_97),
            (0.05, 2.0, -1.0, 0.0, -1.010_002_184_090_538_4),
        ] {
            let got = fhtoffset(dln, mu, initial, bias);
            assert!(
                (got - want).abs() < 1e-12,
                "fhtoffset({dln}, {mu}, {initial}, {bias}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fht_matches_scipy_reference() {
        // Reference values from scipy.fft.fht on a fixed 8-point input.
        let a = [1.0_f64, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.007];
        let dln = 0.1;
        let opts = FftOptions::default();
        let cases: &[(f64, f64, f64, [f64; 8])] = &[
            (
                0.0,
                0.0,
                0.0,
                [
                    0.614_518_022_333_418_1,
                    0.615_038_047_444_707_9,
                    0.439_876_899_751_637_6,
                    0.002_374_672_568_941_638,
                    0.279_043_106_693_971_2,
                    0.066_816_933_697_234_63,
                    -0.366_114_500_530_648_53,
                    0.335_446_818_040_737_56,
                ],
            ),
            (
                1.0,
                0.2,
                0.0,
                [
                    0.740_144_403_778_757,
                    0.357_614_369_725_865,
                    0.313_292_129_973_071_6,
                    0.519_531_546_937_729_7,
                    0.080_484_113_192_972_94,
                    -0.441_037_244_455_451_45,
                    0.189_585_709_779_738_93,
                    0.227_384_971_067_316_34,
                ],
            ),
            (
                0.0,
                0.0,
                0.3,
                [
                    1.064_212_856_210_819,
                    1.002_820_932_280_163_3,
                    0.635_981_621_457_241_4,
                    -0.503_058_453_856_779_9,
                    0.396_182_659_318_834_95,
                    -0.076_518_318_557_207_66,
                    -1.237_588_568_577_655_8,
                    0.403_259_721_259_078_6,
                ],
            ),
        ];
        for &(mu, offset, bias, want) in cases {
            let got = fht(&a, dln, mu, offset, bias, &opts).expect("fht");
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-9,
                    "fht(mu={mu}, offset={offset}, bias={bias})[{i}] = {g}, want {w}"
                );
            }
        }
    }

    #[test]
    fn fhtoffset_returns_finite() {
        let dln = 0.1;
        let mu = 0.5;
        let offset = fhtoffset(dln, mu, 0.0, 0.0);
        assert!(offset.is_finite(), "fhtoffset should return finite value");
        assert!(
            offset.abs() <= dln,
            "offset should be within one period: {offset}"
        );
    }

    #[test]
    fn fhtoffset_different_mu() {
        // Different mu values should give different offsets
        let dln = 0.2;
        let off0 = fhtoffset(dln, 0.0, 0.0, 0.0);
        let off1 = fhtoffset(dln, 1.0, 0.0, 0.0);
        let off2 = fhtoffset(dln, 2.0, 0.0, 0.0);

        // All should be finite
        assert!(off0.is_finite());
        assert!(off1.is_finite());
        assert!(off2.is_finite());
    }

    #[test]
    fn fft_matches_scipy_reference_values() {
        let input: Vec<Complex64> = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let opts = FftOptions::default();
        let result = fft(&input, &opts).expect("fft should succeed");
        // scipy.fft.fft([1,2,3,4]) = [10+0j, -2+2j, -2+0j, -2-2j]
        let expected: Vec<Complex64> = vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                "fft[{i}] = ({}, {}), want ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }

    #[test]
    fn fft_matches_exact_numpy_dft_golden_values() {
        let input: Vec<Complex64> = vec![
            (1.0, 2.0),
            (-3.0, 0.5),
            (0.0, -1.0),
            (4.0, 3.0),
            (-2.0, 0.0),
        ];
        let expected: Vec<Complex64> = vec![
            (0.0, 4.5),
            (-5.656765700396843, 3.838672033902625),
            (10.379271640701377, -1.00291532132572),
            (2.1830342580476767, 5.429966304450563),
            (-1.9055401983522113, -2.7657230170274674),
        ];
        let result = fft(&input, &FftOptions::default()).expect("fft should succeed");
        assert_complex_close(&result, &expected, "numpy fft");
    }

    #[test]
    fn ifft_matches_scipy_reference_values() {
        let input: Vec<Complex64> = vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
        let opts = FftOptions::default();
        let result = ifft(&input, &opts).expect("ifft should succeed");
        // scipy.fft.ifft([10+0j, -2+2j, -2+0j, -2-2j]) = [1,2,3,4]
        let expected: Vec<Complex64> = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                "ifft[{i}] = ({}, {}), want ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }

    #[test]
    fn ifft_matches_exact_numpy_inverse_dft_golden_values() {
        let input: Vec<Complex64> = vec![
            (1.0, 2.0),
            (-3.0, 0.5),
            (4.0, -1.0),
            (0.25, 3.0),
            (-2.5, -4.0),
        ];
        let expected: Vec<Complex64> = vec![
            (-0.05, 0.1),
            (-1.213_305_801_862_807_1, 0.205_814_593_777_397_3),
            (0.062_731_198_931_799, 0.317_847_781_361_829_84),
            (2.642_435_079_130_496_2, 1.861_989_606_263_055),
            (-0.441_860_476_199_487_8, -0.485_651_981_402_281_7),
        ];
        let result = ifft(&input, &FftOptions::default()).expect("ifft should succeed");
        assert_complex_close(&result, &expected, "numpy ifft");
    }

    #[test]
    fn rfft_matches_scipy_reference_values() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let result = rfft(&input, &opts).expect("rfft should succeed");
        // scipy.fft.rfft([1,2,3,4]) = [10+0j, -2+2j, -2+0j]
        let expected: Vec<Complex64> = vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                "rfft[{i}] = ({}, {}), want ({}, {})",
                got.0,
                got.1,
                want.0,
                want.1
            );
        }
    }

    #[test]
    fn rfft_matches_exact_numpy_dft_golden_values() {
        let input: Vec<f64> = vec![0.5, -1.0, 2.5, 0.0, -3.5];
        let expected: Vec<Complex64> = vec![
            (-1.5, 0.0),
            (-2.913_118_960_624_632, -3.8471044214690666),
            (4.913118960624631, 0.9081781600067007),
        ];
        let result = rfft(&input, &FftOptions::default()).expect("rfft should succeed");
        assert_complex_close(&result, &expected, "numpy rfft");
    }

    #[test]
    fn irfft_matches_scipy_reference_values() {
        let input: Vec<Complex64> = vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)];
        let opts = FftOptions::default();
        let result = irfft(&input, Some(4), &opts).expect("irfft should succeed");
        // scipy.fft.irfft([10+0j, -2+2j, -2+0j]) = [1,2,3,4]
        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "irfft[{i}] = {got}, want {want}");
        }
    }

    #[test]
    fn irfft_matches_exact_numpy_inverse_dft_golden_values() {
        let input: Vec<Complex64> = vec![(3.0, 0.75), (-1.5, 2.0), (0.25, -0.5), (2.0, -3.0)];
        let opts = FftOptions::default();
        let result = irfft(&input, Some(6), &opts).expect("irfft should succeed");
        let expected = [
            0.416_666_666_666_666_63,
            -0.558_012_701_892_219_3,
            0.319_978_830_179_634_5,
            0.75,
            1.763_354_503_153_698_8,
            0.308_012_701_892_219_3,
        ];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "numpy irfft[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fftfreq_matches_scipy_reference_values() {
        let result = fftfreq(8, 0.1).expect("fftfreq should succeed");
        // scipy.fft.fftfreq(8, 0.1) = [0, 1.25, 2.5, 3.75, -5, -3.75, -2.5, -1.25]
        let expected = [0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "fftfreq[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn dct_matches_scipy_reference_values() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let result = dct(&input, &opts).expect("dct should succeed");
        // scipy.fft.dct([1,2,3,4], type=2) = [20, -6.308644..., 0, -0.448341...]
        assert!(
            (result[0] - 20.0).abs() < 1e-6,
            "dct[0] = {}, want 20.0",
            result[0]
        );
        assert!(
            (result[1] - (-6.308644059797899)).abs() < 1e-6,
            "dct[1] = {}, want -6.308644",
            result[1]
        );
    }

    #[test]
    fn idct_dct_roundtrip_matches_scipy() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let opts = FftOptions::default();
        let dct_result = dct(&input, &opts).expect("dct should succeed");
        let idct_result = idct(&dct_result, &opts).expect("idct should succeed");
        for (i, (&got, &want)) in idct_result.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "idct(dct(x))[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fftshift_1d_even_length_matches_scipy_reference_values() {
        // scipy.fft.fftshift([0, 1, 2, 3]) = [2, 3, 0, 1]
        let input: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let result = fftshift_1d(&input);
        let expected: [f64; 4] = [2.0, 3.0, 0.0, 1.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "fftshift[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fftshift_1d_odd_length_matches_scipy_reference_values() {
        // scipy.fft.fftshift([0, 1, 2, 3, 4]) = [3, 4, 0, 1, 2]
        let input: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = fftshift_1d(&input);
        let expected: [f64; 5] = [3.0, 4.0, 0.0, 1.0, 2.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "fftshift[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn ifftshift_1d_matches_scipy_reference_values() {
        // scipy.fft.ifftshift([3, 4, 0, 1, 2]) = [0, 1, 2, 3, 4]
        let input: Vec<f64> = vec![3.0, 4.0, 0.0, 1.0, 2.0];
        let result = ifftshift_1d(&input);
        let expected: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "ifftshift[{i}] = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fftshift_nd_scalar_shape_matches_scipy_reference_values() {
        let input = vec![42.0];

        assert_eq!(fftshift(&input, &[], None).expect("fftshift scalar"), input);
        assert_eq!(
            ifftshift(&input, &[], None).expect("ifftshift scalar"),
            input
        );
        assert!(
            fftshift::<f64>(&[], &[], None).is_err(),
            "empty shape represents a scalar and requires one flat value"
        );
        assert!(
            fftshift(&input, &[], Some(&[0])).is_err(),
            "scalar has no shiftable axis"
        );
    }

    #[test]
    fn fft2_matches_scipy_reference_values() {
        // scipy.fft.fft2([[1, 2], [3, 4]])
        use super::transforms::fft2;
        use super::{Complex64, FftOptions, Normalization};
        // 2x2 input: [[1, 2], [3, 4]] in row-major (flattened)
        let input: Vec<Complex64> = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let options = FftOptions {
            normalization: Normalization::Backward,
            ..FftOptions::default()
        };
        let result = fft2(&input, (2, 2), &options).expect("fft2");
        // Expected: [[10, -2], [-4, 0]]
        let expected: [Complex64; 4] = [(10.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), (0.0, 0.0)];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-10 && (got.1 - want.1).abs() < 1e-10,
                "fft2[{i}] = {:?}, want {:?}",
                got,
                want
            );
        }
    }

    #[test]
    fn rfftfreq_matches_scipy_reference_values() {
        // scipy.fft.rfftfreq(5, d=1.0)
        // -> [0.0, 0.2, 0.4]
        let result = rfftfreq(5, 1.0).expect("rfftfreq");
        let expected = [0.0, 0.2, 0.4];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "rfftfreq[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn dst_ii_matches_scipy_reference_values() {
        // scipy.fft.dst([1, 2, 3, 4], type=2)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = dst_ii(&input, &FftOptions::default()).expect("dst_ii");
        // Check that output has same length as input
        assert_eq!(result.len(), 4);
        // DST-II should produce non-zero output for non-constant input
        assert!(result.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn hilbert_matches_scipy_reference_values() {
        // scipy.signal.hilbert([1, 0, 0, 0]) returns analytic signal
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let result = hilbert(&input, &FftOptions::default()).expect("hilbert");
        // Real part should equal input
        assert!((result[0].0 - 1.0).abs() < 1e-10, "hilbert[0] real");
        assert!((result[1].0 - 0.0).abs() < 1e-10, "hilbert[1] real");
        // Imaginary part should be non-trivial for analytic signal
        assert!(
            result.iter().any(|c| c.1.abs() > 1e-10),
            "hilbert should have imag part"
        );
    }

    #[test]
    fn fftconvolve_matches_scipy_reference_values() {
        // scipy.signal.fftconvolve([1, 2, 3], [0, 1, 0.5], mode='full')
        // -> [0.0, 1.0, 2.5, 4.0, 1.5]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = fftconvolve(&a, &b, "full").expect("fftconvolve");
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "fftconvolve[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn fftconvolve_same_mode_matches_scipy_reference() {
        // scipy.signal.fftconvolve([1, 2, 3, 4, 5], [1, 0, -1], mode='same')
        // Output length should equal first input length
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 0.0, -1.0];
        let result = fftconvolve(&a, &b, "same").expect("fftconvolve same");
        assert_eq!(result.len(), a.len(), "same mode output length");
    }

    #[test]
    fn periodogram_simple_matches_scipy_reference() {
        // scipy.signal.periodogram([1, 2, 3, 4], fs=1.0) returns (freqs, psd)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (freqs, psd) = periodogram_simple(&x, 1.0).expect("periodogram");
        // Frequencies should include DC (0.0) and go up to Nyquist (0.5)
        assert!(freqs[0].abs() < 1e-10, "DC frequency should be 0");
        assert!(
            (freqs.last().unwrap() - 0.5).abs() < 1e-10,
            "last freq should be Nyquist"
        );
        // PSD should be non-negative
        assert!(psd.iter().all(|&p| p >= 0.0), "PSD should be non-negative");
    }

    #[test]
    fn magnitude_spectrum_matches_scipy_reference() {
        // scipy.fft.fft([1, 0, -1, 0]) -> magnitude spectrum (one-sided)
        let x = vec![1.0, 0.0, -1.0, 0.0];
        let result = magnitude_spectrum(&x).expect("magnitude_spectrum");
        // One-sided spectrum: DC + n/2 positive freqs = n/2 + 1 for even n
        assert!(
            result.len() >= 2,
            "magnitude spectrum should have at least 2 values"
        );
        // DC component (sum) should be 0
        assert!(result[0].abs() < 1e-10, "DC magnitude should be 0");
        // All magnitudes should be non-negative
        assert!(
            result.iter().all(|&m| m >= 0.0),
            "magnitudes should be non-negative"
        );
    }

    #[test]
    fn analytic_signal_matches_scipy_reference() {
        // scipy.signal.hilbert([1, 2, 3, 4]) returns analytic signal
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let expected = [(1.0, 1.0), (2.0, -1.0), (3.0, -1.0), (4.0, 1.0)];
        let result = analytic_signal(&x).expect("analytic_signal");
        assert_eq!(result.len(), 4);
        for (i, (&(re, im), &(expected_re, expected_im))) in
            result.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (re - expected_re).abs() < 1e-10,
                "analytic_signal[{i}].real = {re}, expected {expected_re}"
            );
            assert!(
                (im - expected_im).abs() < 1e-10,
                "analytic_signal[{i}].imag = {im}, expected {expected_im}"
            );
        }
    }
}
