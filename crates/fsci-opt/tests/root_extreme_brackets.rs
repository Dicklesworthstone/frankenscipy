#![forbid(unsafe_code)]
//! Regression coverage for finite-bracket arithmetic at extreme f64 scales.
//!
//! A sign-changing bracket with finite endpoints must not be rejected because
//! averaging its endpoints or squaring its residuals overflows or underflows.

use fsci_opt::{RootOptions, bisect, ridder};

fn opts() -> RootOptions {
    RootOptions {
        xtol: 1.0e-12,
        rtol: 1.0e-12,
        maxiter: 200,
        ..RootOptions::default()
    }
}

#[test]
fn bisection_does_not_overflow_the_midpoint_of_finite_endpoints() {
    for (a, b, expected) in [
        (1.4e308, 1.7e308, 1.5e308),
        (-1.7e308, -1.4e308, -1.5e308),
    ] {
        let f = |x: f64| x - expected;
        let result = bisect(f, (a, b), opts()).expect("finite bracket is valid");
        assert!(result.converged, "{result:?}");
        assert!(result.root.is_finite());
        assert!(
            (result.root / expected - 1.0).abs() < 2.0e-12,
            "bad root: {result:?}"
        );

        // Even an exhausted iteration budget must return a finite candidate,
        // not an exception caused by a nonfinite evaluation point.
        let limited = bisect(
            f,
            (a, b),
            RootOptions {
                maxiter: 1,
                ..opts()
            },
        )
        .expect("first bisection step must remain finite");
        assert!(limited.root.is_finite(), "{limited:?}");
        assert!(!limited.converged);
    }
}

#[test]
fn ridder_finds_roots_with_finite_huge_bracket_endpoints() {
    for (a, b, expected) in [
        (1.4e308, 1.7e308, 1.5e308),
        (-1.7e308, -1.4e308, -1.5e308),
    ] {
        let result = ridder(|x| x - expected, (a, b), opts())
            .expect("Ridder's interpolation stays inside the finite bracket");
        assert!(result.converged, "{result:?}");
        assert!(result.root.is_finite());
        assert!((result.root / expected - 1.0).abs() < 2.0e-12, "{result:?}");
    }
}

#[test]
fn ridder_rescales_discriminants_without_false_precision_loss() {
    // Before the fix 1e-200 squared to zero, giving an invalid discriminant
    // and false PrecisionLoss after one midpoint evaluation. At 1e200 the
    // discriminant overflowed to infinity and collapsed interpolation to a
    // slow bisection trace.
    for scale in [1.0e-200, 1.0, 1.0e200] {
        let result = ridder(|x| scale * (x - 0.3), (0.0, 1.0), opts())
            .expect("finite sign-changing residuals are a valid Ridder problem");
        assert!(result.converged, "scale={scale}: {result:?}");
        assert!((result.root - 0.3).abs() < 1.0e-9, "{result:?}");
        if scale >= 1.0 {
            assert!(
                result.function_calls <= 8,
                "interpolation should not degrade to bisection: {result:?}"
            );
        }
    }
}
