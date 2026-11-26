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
    for (a, b, expected) in [(1.4e308, 1.7e308, 1.5e308), (-1.7e308, -1.4e308, -1.5e308)] {
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
    for (a, b, expected) in [(1.4e308, 1.7e308, 1.5e308), (-1.7e308, -1.4e308, -1.5e308)] {
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

#[test]
fn secant_routes_match_live_scipy_across_extreme_units() {
    use std::process::Command;

    use fsci_opt::{newton, root_scalar, secant};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct OracleRow {
        unit: f64,
        scale: f64,
        root: f64,
        iterations: usize,
        function_calls: usize,
    }

    #[derive(Deserialize)]
    struct Oracle {
        scipy: String,
        numpy: String,
        rows: Vec<OracleRow>,
    }

    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| String::from("python3"));
    let available = Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP secant extreme units: missing SciPy; compared cases = 0");
        return;
    }

    let script = r#"
import json, os
import numpy as np
import scipy
from scipy.optimize import root_scalar
if os.environ.get("FSCI_REQUIRE_SCIPY_ORACLE") == "1":
    assert (scipy.__version__, np.__version__) == ("1.17.1", "2.4.3")
rows = []
for unit in [1e-100, 1., 1e200]:
    for scale in [1e-308, 1., 1e308]:
        result = root_scalar(lambda x: scale*(x/unit-1.), x0=0., x1=2.*unit,
                             method="secant", xtol=np.nextafter(0., 1.),
                             rtol=1e-12, maxiter=20)
        assert result.converged, (unit, scale, result)
        assert abs(result.root/unit-1.) < 2e-14, (unit, scale, result)
        rows.append(dict(unit=unit, scale=scale, root=result.root,
                         iterations=result.iterations, function_calls=result.function_calls))
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows),
                 allow_nan=False))
"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .output()
        .expect("launch secant SciPy oracle");
    assert!(
        output.status.success(),
        "secant SciPy oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).unwrap();
    if required {
        assert_eq!((&*oracle.scipy, &*oracle.numpy), ("1.17.1", "2.4.3"));
    }
    assert_eq!(
        oracle.rows.len(),
        9,
        "every generated case must be retained"
    );
    let options = RootOptions {
        xtol: f64::from_bits(1),
        rtol: 1e-12,
        maxiter: 20,
        ..RootOptions::default()
    };
    let mut compared = 0;
    for reference in &oracle.rows {
        let f = |x: f64| reference.scale * (x / reference.unit - 1.0);
        let direct = secant(f, 0.0, Some(2.0 * reference.unit), options).unwrap();
        let routed = root_scalar(f, None, Some(0.0), Some(2.0 * reference.unit), options).unwrap();
        let unified = newton(
            f,
            0.0,
            None,
            None,
            Some(2.0 * reference.unit),
            options.xtol,
            options.rtol,
            options.maxiter,
        )
        .unwrap();
        for result in [direct, routed, unified] {
            assert!(result.converged && result.root.is_finite(), "{result:?}");
            assert!(
                (result.root / reference.root - 1.0).abs() < 2e-14,
                "unit={}, scale={}: {result:?}",
                reference.unit,
                reference.scale
            );
            assert_eq!(
                (result.iterations, result.function_calls),
                (reference.iterations, reference.function_calls)
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 27);
    eprintln!(
        "secant extreme units: compared {compared} results against SciPy {} / NumPy {}",
        oracle.scipy, oracle.numpy
    );
}
