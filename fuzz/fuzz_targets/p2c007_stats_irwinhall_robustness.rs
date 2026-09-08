#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{ContinuousDistribution, IrwinHall};
use libfuzzer_sys::fuzz_target;

// Robustness oracle for [frankenscipy-mtg5c]:
// IrwinHall has 3 anchor cases + scipy diff but no fuzz coverage.
// Drive Arbitrary-derived (n, x) and assert:
//   1. Constructor never panics for sanitized n ∈ [1, N_MAX].
//   2. pdf finite, non-negative on the support [0, n].
//   3. pdf = 0 strictly outside support.
//   4. cdf monotone non-decreasing along ascending grid.
//   5. cdf(0) = 0 (n ≥ 2) and cdf(>n) = 1.
//   6. ppf round-trip on interior.
//   7. pdf is symmetric around n/2 for n ≥ 2 (Irwin-Hall is
//      symmetric by construction).

const N_MIN: u32 = 1;
const N_MAX: u32 = 20;

#[derive(Debug, Arbitrary)]
struct Input {
    n_seed: u8,
    x_seed: f64,
}

fn sanitize_n(seed: u8) -> u32 {
    ((seed as u32) % (N_MAX - N_MIN + 1)) + N_MIN
}

fn sanitize_x_in_support(seed: f64, n: u32) -> f64 {
    if !seed.is_finite() {
        return n as f64 / 2.0;
    }
    seed.abs().fract() * n as f64
}

fuzz_target!(|input: Input| {
    let n = sanitize_n(input.n_seed);
    let x = sanitize_x_in_support(input.x_seed, n);
    let nf = n as f64;

    // 1. Constructor never panics within [1, N_MAX].
    let dist = IrwinHall::new(n);

    // 2. pdf finite, non-negative on [0, n].
    let p = dist.pdf(x);
    assert!(p.is_finite(), "pdf({x}; n={n}) must be finite, got {p}");
    assert!(p >= 0.0, "pdf must be non-negative, got {p}");

    // 3. pdf = 0 strictly outside support.
    assert_eq!(dist.pdf(-0.5), 0.0, "pdf(<0) must be 0");
    assert_eq!(dist.pdf(nf + 0.5), 0.0, "pdf(>n) must be 0");

    // 4-5. cdf monotone non-decreasing + boundary values.
    if n >= 2 {
        // For n ≥ 2 the support is open at endpoints (pdf vanishes).
        let cdf_zero = dist.cdf(0.0);
        assert!(
            cdf_zero == 0.0 || cdf_zero.abs() < 1e-15,
            "cdf(0; n={n}) must be 0, got {cdf_zero}"
        );
    }
    assert!(
        dist.cdf(nf + 0.5) > 1.0 - 1e-15,
        "cdf(n + 0.5; n={n}) must be 1"
    );
    let mut prev = 0.0;
    for k in 0..=10 {
        let xi = (k as f64) * nf / 10.0;
        let cv = dist.cdf(xi);
        assert!(
            cv.is_finite() && (0.0..=1.0).contains(&cv),
            "cdf({xi}; n={n}) = {cv} must be finite and in [0, 1]"
        );
        assert!(
            cv >= prev - 1e-12,
            "cdf must be non-decreasing: cdf({xi}; n={n}) = {cv} < prev = {prev}"
        );
        prev = cv;
    }

    // 6. ppf round-trip on interior (skip near boundaries where the
    // alternating-sum cdf saturates fast).
    if n >= 2 && (0.05..0.95).contains(&(x / nf)) {
        let q = dist.cdf(x);
        if (1e-6..1.0 - 1e-6).contains(&q) {
            let recovered = dist.ppf(q);
            // Bisection-derived ppf — give it some slack.
            assert!(
                (recovered - x).abs() < 1e-6 * nf,
                "ppf(cdf({x}; n={n})) = {recovered}, want {x}"
            );
        }
    }

    // 7. Symmetric pdf around n/2 (Irwin-Hall is symmetric).
    if n >= 2 && x > 0.0 && x < nf {
        let mirror = nf - x;
        if mirror > 0.0 && mirror < nf {
            let p_x = dist.pdf(x);
            let p_mirror = dist.pdf(mirror);
            // Allow some fp slack for the alternating-sign sum.
            assert!(
                (p_x - p_mirror).abs() <= 1e-9 * p_x.abs().max(1.0),
                "pdf symmetry: pdf({x}; n={n}) = {p_x} vs pdf({mirror}; n={n}) = {p_mirror}"
            );
        }
    }
});
