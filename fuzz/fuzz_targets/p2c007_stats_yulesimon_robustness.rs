#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{DiscreteDistribution, YuleSimon};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-x9qkq]:
// YuleSimon is the third discrete-distribution fuzz target in
// this session. Drive Arbitrary α ∈ [0.1, 30] and k_seed (u64).
// Assert:
//   1. Constructor never panics.
//   2. pmf finite, non-negative; pmf(k ≤ 0) = 0 (support starts at k=1).
//   3. cdf monotone non-decreasing across the support, in [0, 1].
//   4. cdf(k ≤ 0) = 0; cdf saturates to 1 for k far enough out.
//   5. pmf telescoping: pmf(k) = cdf(k) − cdf(k − 1).
//   6. Mean / var consistency with parameter regime: α > 1 → mean
//      finite; α ≤ 1 → mean infinite. Same threshold structure
//      for var at α = 2.
//
// Signed domain: `DiscreteDistribution::pmf`/`cdf` take `k: i64`.
// The in-support k is still reduced in u64 (1 ≤ k ≤ 200) and then
// converted with a checked `i64::try_from`, so the `Input` layout
// and existing seeds are unchanged. The same seed also maps
// bijectively onto a negative probe k ∈ [i64::MIN, −1] (seed 0 →
// −1), which extends the k = 0 below-support checks (pmf = 0,
// cdf = 0) to the negative integers.

const ALPHA_MIN: f64 = 0.1;
const ALPHA_MAX: f64 = 30.0;

#[derive(Debug, Arbitrary)]
struct Input {
    alpha_seed: f64,
    k_seed: u64,
}

fn sanitize_alpha(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.5;
    }
    seed.abs().clamp(ALPHA_MIN, ALPHA_MAX)
}

fn sanitize_k(seed: u64) -> i64 {
    // 1 ≤ k ≤ 200, so the checked conversion cannot fail.
    i64::try_from(1 + (seed % 200)).expect("k ≤ 200 fits in i64")
}

fn negative_k(seed: u64) -> i64 {
    // seed >> 1 ≤ i64::MAX, so the conversion cannot fail and
    // −1 − x ∈ [i64::MIN, −1] cannot overflow.
    -1 - i64::try_from(seed >> 1).expect("u64 >> 1 fits in i64")
}

fuzz_target!(|input: Input| {
    let alpha = sanitize_alpha(input.alpha_seed);
    let k = sanitize_k(input.k_seed);
    let k_neg = negative_k(input.k_seed);

    // 1. Constructor never panics.
    let dist = YuleSimon::new(alpha);

    // 2. pmf finite, non-negative on support; pmf(k ≤ 0) = 0.
    assert_eq!(dist.pmf(0), 0.0, "pmf(0) must be 0 (support starts at k=1)");
    assert_eq!(dist.pmf(k_neg), 0.0, "pmf({k_neg}) (k < 0) must be 0");
    let p = dist.pmf(k);
    assert!(p.is_finite(), "pmf({k}; α={alpha}) must be finite, got {p}");
    assert!(p >= 0.0, "pmf must be non-negative, got {p}");

    // 3-4. cdf monotone in [0, 1] + boundary.
    assert_eq!(dist.cdf(0), 0.0);
    assert_eq!(dist.cdf(k_neg), 0.0, "cdf({k_neg}) (k < 0) must be 0");
    let mut prev = 0.0;
    for j in 1..=20i64 {
        let cv = dist.cdf(j);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({j}; α={alpha}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({j}; α={alpha}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 4b. Far-tail saturation. cdf(k) = 1 - k·B(k, α+1). For large
    // k, k·B(k, α+1) ~ Γ(α+1)·k^(-α). So saturation rate is O(k^α).
    // For α = 0.1, need k ~ 10^15 for tail < 1e-3 — too large.
    // Skip saturation check for very small α; otherwise probe
    // moderately and check tail < 1.
    if alpha >= 0.5 {
        let probe = 10_000i64;
        let c_high = dist.cdf(probe);
        let tol = if alpha < 1.0 { 1.0e-1 } else { 1.0e-3 };
        assert!(
            c_high > 1.0 - tol,
            "cdf({probe}; α={alpha}) = {c_high} must approach 1 (tol {tol})"
        );
    }

    // 5. pmf telescoping.
    if k >= 1 {
        let prev = if k == 1 { 0.0 } else { dist.cdf(k - 1) };
        let diff = dist.cdf(k) - prev;
        assert!(
            (dist.pmf(k) - diff).abs() < 1e-12,
            "pmf({k}; α={alpha}) = {} ≠ cdf-diff {diff}",
            dist.pmf(k)
        );
    }

    // 6. Moments respect parameter boundaries.
    let m = dist.mean();
    let v = dist.var();
    if alpha > 1.0 {
        assert!(
            m.is_finite() && m > 0.0,
            "mean({alpha}) = {m} must be finite positive"
        );
    } else {
        assert!(m.is_infinite(), "mean({alpha}) = {m} must be ∞ for α ≤ 1");
    }
    if alpha > 2.0 {
        assert!(
            v.is_finite() && v > 0.0,
            "var({alpha}) = {v} must be finite positive"
        );
    } else {
        assert!(v.is_infinite(), "var({alpha}) = {v} must be ∞ for α ≤ 2");
    }
});
