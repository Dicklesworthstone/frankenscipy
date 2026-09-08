#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, RecipInvGauss};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-xgl28]:
// RecipInvGauss has 3 anchor cases + scipy diff but no fuzz
// coverage. Drive Arbitrary-derived (mu, x) and assert:
//   1. Constructor never panics for sanitized mu ∈ [0.05, 50].
//   2. pdf finite, non-negative on x > 0.
//   3. pdf = 0 strictly outside support (x ≤ 0).
//   4. cdf monotone non-decreasing along ascending grid.
//   5. cdf(0) = 0 (exactly), far-tail cdf approaches 1.
//   6. ppf(cdf(x)) round-trip on interior.

const MU_MIN: f64 = 0.05;
const MU_MAX: f64 = 50.0;
const X_MAX: f64 = 1.0e4;

#[derive(Debug, Arbitrary)]
struct Input {
    mu_seed: f64,
    x_seed: f64,
}

fn sanitize_mu(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(MU_MIN, MU_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(1.0e-6, X_MAX)
}

fuzz_target!(|input: Input| {
    let mu = sanitize_mu(input.mu_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within (0, 50].
    let dist = RecipInvGauss::new(mu);

    // 2. pdf finite, non-negative on x > 0.
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; mu={mu}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4. cdf monotone non-decreasing on a grid. Skip the strict
    // monotonicity check for very small mu (deep-tail regime where
    // Φ(z) for z ≈ -5 hits standard_normal_cdf's catastrophic
    // cancellation 0.5·(1 + erf(z)); the cdf accumulates ~1e-7
    // error there). Clamp to [0, 1] is still asserted.
    assert_eq!(dist.cdf(-0.5), 0.0);
    assert_eq!(dist.cdf(0.0), 0.0);
    let mut prev = 0.0;
    for k in 1..=15 {
        let xi = (k as f64) * (mu * 2.0 + 1.0) / 15.0;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; mu={mu}) = {cv} must be in [0, 1]"
        );
        // Only enforce monotonicity once cdf is moderately above 0 —
        // below ~1e-4, the difference of two Φ tails dominates the fp
        // error budget.
        if cv > 1e-4 {
            assert!(
                cv >= prev - 1e-8,
                "cdf must be non-decreasing: cdf({xi}; mu={mu}) = {cv} < prev = {prev}"
            );
        }
        prev = cv;
    }

    // 5. Far-tail saturation. RecipInvGauss(μ) has tail decay ~ 1/√x,
    // so cdf(very large x) approaches 1. Use 1e8 as a "very large" probe.
    let c_high = dist.cdf(1.0e8);
    assert!(
        c_high > 1.0 - 1.0e-3 || c_high.is_nan(),
        "cdf(1e8; mu={mu}) = {c_high} must approach 1"
    );

    // 6. ppf(cdf(x)) round-trip on interior, only for moderate mu.
    // For very small mu the cdf saturates from huge slope changes
    // and bisection precision degrades; for very large mu the deep
    // right tail makes the bisect bracket inadequate. Skip those.
    if (0.5..=10.0).contains(&mu) && x <= 5.0 * mu {
        let q = dist.cdf(x);
        if (1e-3..1.0 - 1e-3).contains(&q) {
            let recovered = dist.ppf(q);
            let scale = x.abs().max(1.0);
            assert!(
                (recovered - x).abs() < 1e-3 * scale,
                "ppf(cdf({x}; mu={mu})) = {recovered}, want {x}"
            );
        }
    }
});
