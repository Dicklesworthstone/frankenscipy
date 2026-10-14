#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{DiscreteDistribution, Planck};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-hb381]:
// Planck is the second discrete-distribution fuzz target in
// this session. Drive Arbitrary λ ∈ [0.01, 50] and k_seed
// (u64), and assert:
//   1. Constructor never panics.
//   2. pmf finite, non-negative across grid; pmf = 0 below the
//      support (k < 0).
//   3. cdf monotone non-decreasing along ascending k, in [0, 1];
//      cdf = 0 below the support (k < 0).
//   4. cdf saturates: cdf(K) → 1 as K grows; specifically
//      1 − cdf(K) = e^(−λ(K + 1)).
//   5. pmf vs cdf-difference: pmf(k) = cdf(k) − cdf(k − 1).
//
// Signed domain: `DiscreteDistribution::pmf`/`cdf` take `k: i64`.
// The grid k is still reduced in u64 (k < 200) and then converted
// with a checked `i64::try_from`, so the `Input` layout and
// existing seeds are unchanged. The same seed also maps
// bijectively onto a negative probe k ∈ [i64::MIN, −1] (seed 0 →
// −1, the edge just below the support [0, ∞)), where the trait
// contract (pmf is 0 off the support; cdf = P(X ≤ k)) gives 0 for
// both. The unsigned API had no off-support k for Planck.

const LAMBDA_MIN: f64 = 0.01;
const LAMBDA_MAX: f64 = 50.0;

#[derive(Debug, Arbitrary)]
struct Input {
    lambda_seed: f64,
    k_seed: u64,
}

fn sanitize_lambda(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(LAMBDA_MIN, LAMBDA_MAX)
}

fn sanitize_k(seed: u64) -> i64 {
    // k < 200, so the checked conversion cannot fail.
    i64::try_from(seed % 200).expect("k < 200 fits in i64")
}

fn negative_k(seed: u64) -> i64 {
    // seed >> 1 ≤ i64::MAX, so the conversion cannot fail and
    // −1 − x ∈ [i64::MIN, −1] cannot overflow.
    -1 - i64::try_from(seed >> 1).expect("u64 >> 1 fits in i64")
}

fuzz_target!(|input: Input| {
    let lambda = sanitize_lambda(input.lambda_seed);
    let k = sanitize_k(input.k_seed);
    let k_neg = negative_k(input.k_seed);

    // 1. Constructor never panics within validated range.
    let dist = Planck::new(lambda);

    // 2. pmf finite, non-negative; 0 below the support.
    let p = dist.pmf(k);
    assert!(
        p.is_finite(),
        "pmf({k}; λ={lambda}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pmf must be non-negative, got {p}");
    assert_eq!(dist.pmf(k_neg), 0.0, "pmf({k_neg}) (k < 0) must be 0");

    // 3. cdf monotone non-decreasing across a grid, starting from
    // cdf = 0 below the support.
    assert_eq!(dist.cdf(k_neg), 0.0, "cdf({k_neg}) (k < 0) must be 0");
    let mut prev = 0.0;
    for j in 0..=20i64 {
        let cv = dist.cdf(j);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({j}; λ={lambda}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({j}; λ={lambda}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 4. Far-tail saturation: cdf(K) → 1 as K grows. The closed
    // form gives 1 − cdf(K) = e^(−λ(K + 1)). Probe at K such that
    // λ(K + 1) ≥ 50 → tail ≤ e^(−50) ≈ 2e-22. λ ≥ 0.01 bounds
    // K ≤ 5000, well inside i64.
    let kf = ((50.0_f64 / lambda).ceil() as i64).max(20);
    let c_high = dist.cdf(kf);
    assert!(
        c_high > 1.0 - 1.0e-15,
        "cdf({kf}; λ={lambda}) = {c_high} must be ≈ 1"
    );

    // 5. pmf(k) = cdf(k) − cdf(k − 1) for k ≥ 1.
    if k >= 1 {
        let pmf_via_diff = dist.cdf(k) - dist.cdf(k - 1);
        assert!(
            (dist.pmf(k) - pmf_via_diff).abs() < 1e-12,
            "pmf({k}; λ={lambda}) = {} ≠ cdf-diff {pmf_via_diff}",
            dist.pmf(k)
        );
    }
});
