
/// frankenscipy-5f06d — driver for `ND_FILTER_FORCE_SCALAR`, which had none.
/// The default path processes 8 consecutive interior pixels with one SIMD
/// accumulator; the toggle restores the scalar interior. Documented BIT-IDENTICAL:
/// each lane sums the SAME taps in the SAME k-order, `+= w*x` with no FMA
/// contraction.
#[cfg(test)]
mod toggle_ab_nd_filter_scalar {
    use super::{BoundaryMode, ND_FILTER_FORCE_SCALAR, NdArray, correlate_with_origins};
    use std::sync::atomic::Ordering;

    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// This toggle has NO size gate -- it is predicated on `ndim >= 2` and an
    /// innermost stride of 1. What it DOES need is an interior box wide enough
    /// for the 8-wide path to trigger: the SIMD arm runs on runs of 8 consecutive
    /// interior pixels along the innermost axis, so a fixture narrower than
    /// `8 + kernel_width` would leave every pixel on the boundary fallback and
    /// compare the scalar path with itself.
    ///
    /// Driven under two boundary modes because the interior/border split is part
    /// of what the toggle changes: Constant and Reflect take different border
    /// paths, and a fixture that only exercised one would leave the other's
    /// interaction with the SIMD interior undriven.
    #[test]
    fn nd_filter_scalar_toggle_is_bit_identical() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

        const H: usize = 192;
        const W: usize = 192;
        const K: usize = 3;
        // Interior run along the innermost axis must comfortably exceed the 8-wide
        // block, or every pixel takes the border path and both arms agree trivially.
        const _: () = assert!(W >= 8 + K);

        assert!(
            f64::from_bits(4.5f64.to_bits() ^ 1).to_bits() != 4.5f64.to_bits()
                && 0.0f64.to_bits() != (-0.0f64).to_bits(),
            "the bit comparison is blind; the exact assertions below would pass \
             vacuously"
        );

        let data: Vec<f64> = (0..(H * W) as u64)
            .map(|i| {
                let k = i.wrapping_mul(2_654_435_761);
                ((k % 100_003) as f64) / 1000.0 - 50.0
            })
            .collect();
        let input = NdArray::new(data, vec![H, W]).expect("input");
        // Asymmetric weights: a symmetric kernel can mask a tap-ordering fault
        // because reversing the taps leaves the sum unchanged.
        let weights = NdArray::new(
            (0..(K * K) as u64).map(|i| 0.5 + i as f64).collect(),
            vec![K, K],
        )
        .expect("weights");
        let origins = vec![0i64, 0i64];

        for mode in [BoundaryMode::Constant, BoundaryMode::Reflect] {
            ND_FILTER_FORCE_SCALAR.store(true, Ordering::Relaxed);
            let scalar = correlate_with_origins(&input, &weights, &origins, mode, 0.0)
                .expect("correlate scalar");
            ND_FILTER_FORCE_SCALAR.store(false, Ordering::Relaxed);
            let simd = correlate_with_origins(&input, &weights, &origins, mode, 0.0)
                .expect("correlate simd");

            assert_eq!(
                scalar.data.len(),
                H * W,
                "correlate returned the wrong element count for {mode:?}"
            );
            assert!(
                scalar.data.iter().any(|v| v.is_finite() && *v != 0.0),
                "the {mode:?} fixture produced an all-zero result, which compares \
                 equal in both arms while proving nothing"
            );
            let diff = scalar
                .data
                .iter()
                .zip(&simd.data)
                .position(|(x, y)| x.to_bits() != y.to_bits());
            assert!(
                diff.is_none(),
                "ND_FILTER_FORCE_SCALAR is documented BIT-IDENTICAL and is not \
                 under {mode:?}: first difference at flat index {diff:?}. Each SIMD \
                 lane sums the same taps in the same k-order with no FMA \
                 contraction, so a difference here is a real defect in the \
                 vectorized interior"
            );
        }
        ND_FILTER_FORCE_SCALAR.store(false, Ordering::Relaxed);

        // Both arms agreeing cannot catch a fault they share. One interior pixel,
        // far from every border, is checked against the definition of correlation
        // computed directly from the input.
        let (r, c) = (H / 2, W / 2);
        let mut expect = 0.0f64;
        for kr in 0..K {
            for kc in 0..K {
                let ir = r + kr - K / 2;
                let ic = c + kc - K / 2;
                expect += weights.data[kr * K + kc] * input.data[ir * W + ic];
            }
        }
        let got = correlate_with_origins(
            &input,
            &weights,
            &origins,
            BoundaryMode::Constant,
            0.0,
        )
        .expect("correlate")
        .data[r * W + c];
        assert!(
            (got - expect).abs() <= 1e-9 * expect.abs().max(1.0),
            "the interior pixel disagrees with the correlation definition: got \
             {got}, expected {expect}"
        );
    }
}
