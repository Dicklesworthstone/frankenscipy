
/// frankenscipy-5f06d — driver for `MAHALANOBIS_ASSEMBLY_FORCE_SERIAL`, which
/// had none. Documented BYTE-IDENTICAL: the parallel arm splits the final
/// `sqrt` assembly across rows writing DISJOINT output slices, so no float is
/// combined across threads and rows are reassembled in order.
#[cfg(test)]
mod toggle_ab_mahalanobis_assembly {
    use super::{MAHALANOBIS_ASSEMBLY_FORCE_SERIAL, cdist_mahalanobis};
    use std::sync::atomic::Ordering;

    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The assembly is gated on CELLS, `na * nb >= 1 << 22` -- not on the point
    /// dimension and not on either input length alone. 2048 x 2048 clears it
    /// exactly; anything smaller sends both settings down the same serial loop
    /// and the comparison becomes a path compared with itself.
    ///
    /// `d` is deliberately 2. The gate is met by the CELL COUNT, so dimension is
    /// free to be small -- and the distance compute underneath is O(na*nb*d), so
    /// a larger `d` would only make an unoptimized test binary slower without
    /// making the comparison stronger.
    #[test]
    fn mahalanobis_assembly_toggle_is_byte_identical() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

        const NA: usize = 2048;
        const NB: usize = 2048;
        const D: usize = 2;
        const _: () = assert!(NA * NB >= (1 << 22));
        assert!(
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                >= 2,
            "the parallel arm is vacuous on a host reporting < 2 usable cores"
        );

        // Detector must-hit/must-miss: several assertions below are bit-exact, so
        // "nothing differed" is the passing outcome and would look identical to a
        // comparison that cannot see a difference. `to_bits`, since -0.0 == 0.0.
        assert!(
            f64::from_bits(3.5f64.to_bits() ^ 1).to_bits() != 3.5f64.to_bits()
                && 0.0f64.to_bits() != (-0.0f64).to_bits(),
            "the bit comparison is blind; the exact assertion below would pass \
             vacuously"
        );

        let pt = |i: u64, j: u64| {
            let k = i.wrapping_mul(2_654_435_761).wrapping_add(j.wrapping_mul(40_503));
            ((k % 100_003) as f64) / 1000.0
        };
        let xa: Vec<Vec<f64>> = (0..NA)
            .map(|i| (0..D).map(|j| pt(i as u64, j as u64)).collect())
            .collect();
        let xb: Vec<Vec<f64>> = (0..NB)
            .map(|i| (0..D).map(|j| pt(i as u64 + 7919, j as u64)).collect())
            .collect();
        // Identity inverse-covariance keeps the fixture well-conditioned; the
        // toggle gates the sqrt ASSEMBLY, not the covariance algebra.
        let vi: Vec<Vec<f64>> = (0..D)
            .map(|i| (0..D).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist serial");
        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let par = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist parallel");
        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(false, Ordering::Relaxed);

        assert_eq!(serial.len(), NA, "cdist returned the wrong row count");
        // A result that is all zeros -- or all NaN -- compares equal in both arms
        // while exercising nothing, so the fixture is checked for signal first.
        assert!(
            serial
                .iter()
                .flatten()
                .any(|v| v.is_finite() && *v > 0.0),
            "mahalanobis fixture is degenerate: no positive finite distance"
        );

        let mut first_diff = None;
        'outer: for (r, (rs, rp)) in serial.iter().zip(&par).enumerate() {
            assert_eq!(rs.len(), rp.len(), "row {r} length differs between arms");
            for (c, (x, y)) in rs.iter().zip(rp).enumerate() {
                if x.to_bits() != y.to_bits() {
                    first_diff = Some((r, c, *x, *y));
                    break 'outer;
                }
            }
        }
        assert!(
            first_diff.is_none(),
            "MAHALANOBIS_ASSEMBLY_FORCE_SERIAL is documented BYTE-IDENTICAL and \
             is not: first difference at {first_diff:?}. The parallel arm writes \
             disjoint row slices and combines no floats across threads, so a \
             difference here is a placement or bounds fault, not rounding"
        );

        // Checking only that the two arms AGREE cannot catch a fault they share --
        // e.g. both arms leaving the tail rows untouched. The last row is verified
        // against the closed form directly, since with an identity `vi` the
        // Mahalanobis distance IS the Euclidean one.
        let last = NA - 1;
        let expect: f64 = (0..D)
            .map(|j| (xa[last][j] - xb[NB - 1][j]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (par[last][NB - 1] - expect).abs() <= 1e-9 * expect.max(1.0),
            "the final cell disagrees with the closed form under an identity vi: \
             got {}, expected {expect}. Both arms agreeing on a wrong value is \
             exactly what an arms-only comparison cannot see",
            par[last][NB - 1]
        );
    }
}
