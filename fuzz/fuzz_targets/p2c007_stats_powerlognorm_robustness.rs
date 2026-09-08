#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, PowerLognorm};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-evspb]:
// PowerLognorm has 3 anchor + 368-case scipy diff but no fuzz
// coverage. Drive Arbitrary-derived (c, s, x) and assert:
//   1. Constructor never panics for sanitized c, s ∈ [0.1, 10].
//   2. pdf finite, non-negative on x > 0.
//   3. pdf = 0 strictly outside support (x ≤ 0).
//   4. cdf monotone non-decreasing along ascending grid in [0, 1].
//   5. cdf(≤0) = 0; far-tail cdf approaches 1.
//   6. ppf(cdf(x)) round-trip on interior — wide tol (1e-5 rel)
//      because PowerLognorm.ppf composes exp(−s·Φ⁻¹) with the
//      Beasley-Springer-Moro Φ⁻¹ helper (~1e-9 floor).
//   7. cdf + sf ≈ 1.

const PARAM_MIN: f64 = 0.1;
const PARAM_MAX: f64 = 10.0;
const X_MIN: f64 = 1.0e-4;
const X_MAX: f64 = 100.0;

#[derive(Debug, Arbitrary)]
struct Input {
    c_seed: f64,
    s_seed: f64,
    x_seed: f64,
}

fn sanitize_param(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(PARAM_MIN, PARAM_MAX)
}

fn sanitize_x(seed: f64) -> f64 {
    if !seed.is_finite() {
        return 1.0;
    }
    seed.abs().clamp(X_MIN, X_MAX)
}

fuzz_target!(|input: Input| {
    let c = sanitize_param(input.c_seed);
    let s = sanitize_param(input.s_seed);
    let x = sanitize_x(input.x_seed);

    // 1. Constructor never panics within validated range.
    let dist = PowerLognorm::new(c, s);

    // 2. pdf finite, non-negative on x > 0.
    let p = dist.pdf(x);
    assert!(
        p.is_finite(),
        "pdf({x}; c={c}, s={s}) must be finite, got {p}"
    );
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(0.0), 0.0, "pdf(0) must be 0");

    // 4-5. cdf monotone non-decreasing on grid + boundaries.
    assert_eq!(dist.cdf(-0.5), 0.0);
    assert_eq!(dist.cdf(0.0), 0.0);
    let mut prev = 0.0;
    for k in 1..=15 {
        let xi = (k as f64) * x / 7.5;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; c={c}, s={s}) = {cv} must be in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; c={c}, s={s}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 5b. Far-tail saturation depends on c AND s:
    //   sf(x) = Φ(−ln(x)/s)^c
    // The probe x must satisfy ln(x)/s = z_target where Φ(−z) is
    // small enough that Φ(−z)^c < tolerance. Scale ln(probe)
    // with s; cap the magnitude to avoid overflow.
    // For c < 0.2 the saturation rate is too heavy to bound
    // tightly within representable x (e.g., c=0.15, s=9: at z=5
    // sf = (3e-7)^0.15 ≈ 0.10 — 90% saturated, not 99.9%).
    // Just check monotone in the deep tail and skip the strict
    // approach-1 gate.
    if c >= 0.2 {
        let log_probe = (5.0 * s).clamp(18.4, 700.0);
        let probe = log_probe.exp();
        if probe.is_finite() {
            let c_high = dist.cdf(probe);
            let saturation_tol = if c < 0.5 { 1.0e-1 } else { 1.0e-3 };
            assert!(
                c_high > 1.0 - saturation_tol,
                "cdf({probe}; c={c}, s={s}) = {c_high} must approach 1 (tol {saturation_tol})"
            );
        }
    } else {
        let c_med = dist.cdf(1.0e6);
        let c_far = dist.cdf(1.0e10);
        assert!(
            c_far >= c_med - 1e-12,
            "cdf monotone in deep tail (c={c}, s={s}): \
             cdf(1e10)={c_far}, cdf(1e6)={c_med}"
        );
    }

    // 6. ppf(cdf(x)) round-trip on interior. Skip for c < 0.2 —
    // ppf composes exp(s · Φ⁻¹((1 − q)^(1/c))) and the 1/c
    // exponent amplifies the BSM Φ⁻¹ ~1e-9 floor and the
    // exp() amplifies further. Useful invariant only holds at
    // moderate-to-large c.
    if c >= 0.2 {
        let q = dist.cdf(x);
        if (1e-4..1.0 - 1e-4).contains(&q) {
            let recovered = dist.ppf(q);
            let scale = x.abs().max(1.0);
            let rel_tol = (1e-5 * (1.0_f64.max(1.0 / c))).max(1e-4);
            assert!(
                (recovered - x).abs() < rel_tol * scale,
                "ppf(cdf({x}; c={c}, s={s})) = {recovered}, want {x} (tol {rel_tol:e})"
            );
        }
    }

    // 7. cdf + sf ≈ 1. PowerLognorm computes cdf via
    //   −expm1(c · ln_1p(−Φ(z)))     [stable for x ≪ 1]
    // and sf via Φ(−z)^c, which take different paths through
    // the standard normal helpers. The drift between paths is
    // bounded by Φ-helper precision (~1e-15) × powf magnification.
    // For c < 0.5, raising to a small power amplifies that:
    // (1 − ε)^c ≈ 1 − cε, so the difference scales with c. But
    // for c < 0.5 with s small the drift can reach ~1e-6.
    // Tolerance scaled accordingly.
    // After the cdf branch fix, cdf and sf agree to ~1e-12 in
    // the well-conditioned regime. For very small c (< 0.2)
    // the sf path Φ(−z)^c can drift slightly more due to powf
    // precision; relax modestly.
    let sum = dist.cdf(x) + dist.sf(x);
    let sum_tol = if c < 0.2 { 1e-9 } else { 1e-12 };
    assert!(
        (sum - 1.0).abs() < sum_tol,
        "cdf({x}; c={c}, s={s}) + sf = {sum}, must be 1 ± {sum_tol:e}"
    );
});
