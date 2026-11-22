//! Differential oracle, metamorphic relation, and adversarial tests
//! for fsci-integrate validation functions (bd-3jh.12.6).
//!
//! Oracle values are hand-computed or derived from the fixture file
//! FSCI-P2C-001_validate_tol.json.  Every differential test verifies
//! that the Rust implementation matches the hand-computed oracle within
//! tolerance.

use fsci_integrate::{
    InitialStepRequest, IntegrateValidationError, MIN_RTOL, ToleranceValue, ToleranceWarning,
    select_initial_step, validate_first_step, validate_max_step, validate_tol,
};
use fsci_runtime::RuntimeMode;

// ═══════════════════════════════════════════════════════════════════
// §1  Differential Oracle Tests (18)
// ═══════════════════════════════════════════════════════════════════

// -- Fixture case 1: scalar_rtol_clamp --
// rtol=1e-30 below MIN_RTOL → clamped to MIN_RTOL with warning
#[test]
fn diff_fixture_scalar_rtol_clamp() {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("should succeed with clamping");

    // Oracle: rtol clamped to MIN_RTOL = 100 * f64::EPSILON = 2.220446049250313e-14
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert_eq!(result.atol, ToleranceValue::Scalar(1e-8));
    assert_eq!(
        result.warnings,
        vec![ToleranceWarning::RtolClamped { minimum: MIN_RTOL }]
    );
}

// -- Fixture case 2: vector_passthrough --
// Valid vector tolerances in Hardened mode → pass through unchanged
#[test]
fn diff_fixture_vector_passthrough() {
    let result = validate_tol(
        ToleranceValue::Vector(vec![1e-6, 2e-6, 3e-6]),
        ToleranceValue::Vector(vec![1e-9, 2e-9, 3e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect("should succeed");

    // Oracle: everything passes through unchanged, no warnings
    assert_eq!(result.rtol, ToleranceValue::Vector(vec![1e-6, 2e-6, 3e-6]));
    assert_eq!(result.atol, ToleranceValue::Vector(vec![1e-9, 2e-9, 3e-9]));
    assert!(result.warnings.is_empty());
    assert_eq!(result.mode, RuntimeMode::Hardened);
}

// -- Fixture case 3: wrong_shape --
// Vector atol length 2 but n=3 → AtolWrongShape error
#[test]
fn diff_fixture_wrong_shape() {
    let err = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9, 2e-9]),
        3,
        RuntimeMode::Strict,
    )
    .expect_err("should fail with wrong shape");

    // Oracle: error message "`atol` has wrong shape."
    assert_eq!(
        err,
        IntegrateValidationError::AtolWrongShape {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(err.to_string(), "`atol` has wrong shape.");
}

// -- Fixture case 4: negative_atol --
// atol vector with negative element → AtolMustBePositive error
#[test]
fn diff_fixture_negative_atol() {
    let err = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9, -2e-9, 3e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect_err("should fail with negative atol");

    // Oracle: error message "`atol` must be positive."
    assert_eq!(err, IntegrateValidationError::AtolMustBePositive);
    assert_eq!(err.to_string(), "`atol` must be positive.");
}

// -- Hand-computed oracle: rtol exactly at MIN_RTOL boundary --
#[test]
fn diff_rtol_exact_min() {
    let result = validate_tol(
        ToleranceValue::Scalar(MIN_RTOL),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("MIN_RTOL is valid");

    // Oracle: MIN_RTOL is NOT < MIN_RTOL so no clamping, no warning
    assert!(result.warnings.is_empty());
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
}

// -- Hand-computed oracle: rtol just below MIN_RTOL --
#[test]
fn diff_rtol_just_below_min() {
    let just_below = MIN_RTOL * (1.0 - f64::EPSILON);
    let result = validate_tol(
        ToleranceValue::Scalar(just_below),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("should clamp");

    // Oracle: just_below < MIN_RTOL → clamped to MIN_RTOL
    assert_eq!(result.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert!(!result.warnings.is_empty());
}

// -- Hand-computed oracle: validate_first_step within bounds --
#[test]
fn diff_first_step_within_bounds() {
    // Oracle: 0 < 0.5 <= |10.0 - 0.0| = 10.0 → Ok(0.5)
    let result = validate_first_step(0.5, 0.0, 10.0);
    assert_eq!(result.unwrap(), 0.5);
}

// -- Hand-computed oracle: validate_first_step at boundary --
#[test]
fn diff_first_step_at_boundary() {
    // Oracle: 0 < 10.0 <= |10.0 - 0.0| = 10.0 → Ok(10.0)
    let result = validate_first_step(10.0, 0.0, 10.0);
    assert_eq!(result.unwrap(), 10.0);
}

// -- Hand-computed oracle: validate_first_step exceeds bounds --
#[test]
fn diff_first_step_exceeds() {
    // Oracle: 10.1 > |10.0 - 0.0| = 10.0 → Err(FirstStepExceedsBounds)
    let err = validate_first_step(10.1, 0.0, 10.0).expect_err("exceeds bounds");
    assert_eq!(err, IntegrateValidationError::FirstStepExceedsBounds);
}

// -- Hand-computed oracle: validate_first_step zero --
#[test]
fn diff_first_step_zero() {
    // Oracle: 0.0 <= 0.0 → Err(FirstStepMustBePositive)
    let err = validate_first_step(0.0, 0.0, 10.0).expect_err("zero step");
    assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
}

// -- Hand-computed oracle: validate_first_step negative --
#[test]
fn diff_first_step_negative() {
    // Oracle: -1.0 <= 0.0 → Err(FirstStepMustBePositive)
    let err = validate_first_step(-1.0, 0.0, 10.0).expect_err("negative step");
    assert_eq!(err, IntegrateValidationError::FirstStepMustBePositive);
}

// -- Hand-computed oracle: validate_first_step backward integration --
#[test]
fn diff_first_step_backward() {
    // Oracle: 0 < 5.0 <= |0.0 - 10.0| = 10.0 → Ok(5.0)
    let result = validate_first_step(5.0, 10.0, 0.0);
    assert_eq!(result.unwrap(), 5.0);
}

// -- Hand-computed oracle: validate_max_step positive --
#[test]
fn diff_max_step_positive() {
    // Oracle: 1.0 > 0.0 → Ok(1.0)
    assert_eq!(validate_max_step(1.0).unwrap(), 1.0);
}

// -- Hand-computed oracle: validate_max_step infinity --
#[test]
fn diff_max_step_infinity() {
    // Oracle: INFINITY > 0.0 → Ok(INFINITY)
    assert_eq!(validate_max_step(f64::INFINITY).unwrap(), f64::INFINITY);
}

// -- Hand-computed oracle: validate_max_step zero --
#[test]
fn diff_max_step_zero() {
    // Oracle: 0.0 <= 0.0 → Err(MaxStepMustBePositive)
    let err = validate_max_step(0.0).expect_err("zero");
    assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
}

// -- Hand-computed oracle: validate_max_step negative --
#[test]
fn diff_max_step_negative() {
    // Oracle: -5.0 <= 0.0 → Err(MaxStepMustBePositive)
    let err = validate_max_step(-5.0).expect_err("negative");
    assert_eq!(err, IntegrateValidationError::MaxStepMustBePositive);
}

// -- Hand-computed oracle: vector rtol with partial clamping --
#[test]
fn diff_vector_rtol_partial_clamp() {
    let result = validate_tol(
        ToleranceValue::Vector(vec![1e-3, 1e-30, 1e-6]),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("should succeed with partial clamping");

    // Oracle: element 1 (1e-30) < MIN_RTOL, gets clamped to MIN_RTOL
    // Elements 0 and 2 are above MIN_RTOL so pass through
    // Warning is emitted because needs_clamp triggers on any(|x| x < MIN_RTOL)
    assert!(
        matches!(&result.rtol, ToleranceValue::Vector(_)),
        "expected vector"
    );
    if let ToleranceValue::Vector(v) = &result.rtol {
        assert_eq!(v[0], 1e-3);
        assert_eq!(v[1], MIN_RTOL);
        assert_eq!(v[2], 1e-6);
    }
    assert!(!result.warnings.is_empty());
}

// -- Hand-computed oracle: scalar atol with n=0 (empty system) --
#[test]
fn diff_empty_system() {
    // Oracle: scalar atol has no shape check, n=0 is valid → Ok
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(1e-6),
        0,
        RuntimeMode::Strict,
    )
    .expect("empty system should succeed");
    assert!(result.warnings.is_empty());
}

// -- Hand-computed oracle: Hardened select_initial_step rejects non-finite y0 --
#[test]
fn diff_select_initial_step_hardened_rejects_non_finite_y0() {
    let request = InitialStepRequest {
        t0: 0.0,
        y0: &[f64::NAN],
        t_bound: 1.0,
        max_step: f64::INFINITY,
        f0: &[1.0],
        direction: 1.0,
        order: 4.0,
        rtol: 1e-3,
        atol: ToleranceValue::Scalar(1e-6),
        mode: RuntimeMode::Hardened,
    };

    let mut rhs_called = false;
    let err = select_initial_step(
        &mut |_t, _y| {
            rhs_called = true;
            vec![0.0]
        },
        &request,
    )
    .expect_err("non-finite y0 should fail");

    assert!(
        !rhs_called,
        "hardened validation should fail before RHS evaluation"
    );
    assert_eq!(err, IntegrateValidationError::NonFiniteY0);
    assert_eq!(err.to_string(), "`y0` must be finite in Hardened mode.");
}

// -- Hand-computed oracle: Hardened select_initial_step rejects non-finite f0 --
#[test]
fn diff_select_initial_step_hardened_rejects_non_finite_f0() {
    let request = InitialStepRequest {
        t0: 0.0,
        y0: &[1.0],
        t_bound: 1.0,
        max_step: f64::INFINITY,
        f0: &[f64::NEG_INFINITY],
        direction: 1.0,
        order: 4.0,
        rtol: 1e-3,
        atol: ToleranceValue::Scalar(1e-6),
        mode: RuntimeMode::Hardened,
    };

    let mut rhs_called = false;
    let err = select_initial_step(
        &mut |_t, _y| {
            rhs_called = true;
            vec![0.0]
        },
        &request,
    )
    .expect_err("non-finite f0 should fail");

    assert!(
        !rhs_called,
        "hardened validation should fail before RHS evaluation"
    );
    assert_eq!(err, IntegrateValidationError::NonFiniteF0);
    assert_eq!(err.to_string(), "`f0` must be finite in Hardened mode.");
}

// ═══════════════════════════════════════════════════════════════════
// §2  Metamorphic Relation Tests (8)
// ═══════════════════════════════════════════════════════════════════

// MR1: validate_tol(rtol, atol * c, n) scales atol by c in result
#[test]
fn meta_atol_scaling() {
    let c = 3.0;
    let atol_base = vec![1e-8, 2e-8, 3e-8];
    let atol_scaled: Vec<f64> = atol_base.iter().map(|x| x * c).collect();

    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(atol_base.clone()),
        3,
        RuntimeMode::Strict,
    )
    .expect("base should succeed");

    let r2 = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(atol_scaled),
        3,
        RuntimeMode::Strict,
    )
    .expect("scaled should succeed");

    // Metamorphic: r2.atol == r1.atol * c
    assert!(
        matches!(
            (&r1.atol, &r2.atol),
            (ToleranceValue::Vector(_), ToleranceValue::Vector(_))
        ),
        "expected vectors"
    );
    if let (ToleranceValue::Vector(v1), ToleranceValue::Vector(v2)) = (&r1.atol, &r2.atol) {
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((b - a * c).abs() < 1e-20, "scaling not preserved");
        }
    }
}

// MR2: validate_tol in Strict == validate_tol in Hardened for valid inputs
// (both modes accept/reject the same valid inputs, produce same rtol/atol)
#[test]
fn meta_strict_equals_hardened_for_valid() {
    let cases: Vec<(f64, f64)> = vec![
        (1e-3, 1e-6),
        (1e-6, 1e-9),
        (1e-30, 1e-8), // triggers clamping
    ];

    for (rtol, atol) in cases {
        let r_strict = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            5,
            RuntimeMode::Strict,
        )
        .expect("strict should succeed");

        let r_hardened = validate_tol(
            ToleranceValue::Scalar(rtol),
            ToleranceValue::Scalar(atol),
            5,
            RuntimeMode::Hardened,
        )
        .expect("hardened should succeed");

        // Metamorphic: rtol and atol values are mode-independent
        assert_eq!(r_strict.rtol, r_hardened.rtol, "rtol should match");
        assert_eq!(r_strict.atol, r_hardened.atol, "atol should match");
        assert_eq!(
            r_strict.warnings.len(),
            r_hardened.warnings.len(),
            "warning count should match"
        );
    }
}

// MR3: validate_first_step(s, t0, tb) == validate_first_step(s, t0+k, tb+k)
// (translation invariance)
#[test]
fn meta_first_step_translation_invariance() {
    let offsets = [0.0, 100.0, -100.0, 1e10, -1e10];
    let step = 0.5;
    let t0 = 0.0;
    let tb = 10.0;

    let base = validate_first_step(step, t0, tb);

    for k in offsets {
        let shifted = validate_first_step(step, t0 + k, tb + k);
        assert_eq!(
            base.is_ok(),
            shifted.is_ok(),
            "translation by {k} changed result"
        );
        if let (Ok(a), Ok(b)) = (&base, &shifted) {
            assert_eq!(a, b, "translated values should match");
        }
    }
}

// MR4: validate_max_step(s) result independent of prior validate_tol calls
// (no state leakage)
#[test]
fn meta_max_step_no_state_leakage() {
    // Call validate_tol first to see if it leaks state
    let _ = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    );

    let r1 = validate_max_step(1.0);

    // Call a bunch more
    for _ in 0..100 {
        let _ = validate_tol(
            ToleranceValue::Scalar(1e-30),
            ToleranceValue::Vector(vec![1e-6; 100]),
            100,
            RuntimeMode::Hardened,
        );
    }

    let r2 = validate_max_step(1.0);
    assert_eq!(r1, r2, "state leakage detected");
}

// MR5: validate_tol idempotence — applying validate_tol to already-valid output
// produces the same result with no additional warnings
#[test]
fn meta_validate_tol_idempotence() {
    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-8),
        3,
        RuntimeMode::Strict,
    )
    .expect("first pass");

    // Apply again with the output values
    let r2 = validate_tol(r1.rtol.clone(), r1.atol.clone(), 3, RuntimeMode::Strict)
        .expect("second pass");

    // Metamorphic: output is a fixpoint — no further clamping needed
    assert_eq!(r1.rtol, r2.rtol, "rtol should be stable");
    assert_eq!(r1.atol, r2.atol, "atol should be stable");
    assert!(
        r2.warnings.is_empty(),
        "second pass should produce no warnings"
    );
}

// MR6: error kind preserved under mode change
// (wrong_shape and negative_atol errors don't depend on mode)
#[test]
fn meta_error_kind_mode_independent() {
    // Wrong shape
    let err_strict = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9]),
        3,
        RuntimeMode::Strict,
    )
    .expect_err("wrong shape");

    let err_hardened = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Vector(vec![1e-9]),
        3,
        RuntimeMode::Hardened,
    )
    .expect_err("wrong shape");

    assert_eq!(
        err_strict, err_hardened,
        "error kind should be mode-independent"
    );

    // Negative atol
    let err_strict2 = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(-1.0),
        1,
        RuntimeMode::Strict,
    )
    .expect_err("negative atol");

    let err_hardened2 = validate_tol(
        ToleranceValue::Scalar(1e-6),
        ToleranceValue::Scalar(-1.0),
        1,
        RuntimeMode::Hardened,
    )
    .expect_err("negative atol");

    assert_eq!(err_strict2, err_hardened2);
}

// MR7: validate_first_step symmetry — same step valid for (t0,tb) and (tb,t0)
#[test]
fn meta_first_step_direction_symmetry() {
    let step = 3.0;
    let fwd = validate_first_step(step, 0.0, 10.0);
    let bwd = validate_first_step(step, 10.0, 0.0);
    assert_eq!(
        fwd.is_ok(),
        bwd.is_ok(),
        "direction should not affect validity"
    );
    assert_eq!(fwd.unwrap(), bwd.unwrap());
}

// MR8: validate_tol monotonicity — if rtol1 < rtol2 and both need clamping,
// both produce the same clamped value
#[test]
fn meta_rtol_clamping_convergence() {
    let r1 = validate_tol(
        ToleranceValue::Scalar(1e-30),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("r1");

    let r2 = validate_tol(
        ToleranceValue::Scalar(1e-100),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    )
    .expect("r2");

    // Both are below MIN_RTOL → both clamped to the same value
    assert_eq!(
        r1.rtol, r2.rtol,
        "all sub-minimum rtols converge to MIN_RTOL"
    );
}

// ═══════════════════════════════════════════════════════════════════
// §3  Adversarial Vector Tests (10)
// ═══════════════════════════════════════════════════════════════════

// ADV1: rtol = f64::MIN_POSITIVE (near zero positive)
#[test]
fn adv_rtol_min_positive() {
    let result = validate_tol(
        ToleranceValue::Scalar(f64::MIN_POSITIVE),
        ToleranceValue::Scalar(1e-6),
        1,
        RuntimeMode::Strict,
    );
    // MIN_POSITIVE ≈ 5e-324, far below MIN_RTOL → should clamp
    let validated = result.expect("should clamp, not panic");
    assert_eq!(validated.rtol, ToleranceValue::Scalar(MIN_RTOL));
    assert!(!validated.warnings.is_empty());
}

// ADV2: atol vector with one NaN element
#[test]
fn adv_atol_nan_element() {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-8, f64::NAN, 1e-8]),
        3,
        RuntimeMode::Strict,
    );
    assert_eq!(
        result.expect_err("NaN atol should fail closed"),
        IntegrateValidationError::NonFiniteAtol
    );
}

// ADV2b: rtol is NaN
#[test]
fn adv_rtol_nan() {
    let result = validate_tol(
        ToleranceValue::Scalar(f64::NAN),
        ToleranceValue::Scalar(1e-8),
        1,
        RuntimeMode::Strict,
    );
    assert_eq!(
        result.expect_err("NaN rtol should fail closed"),
        IntegrateValidationError::NonFiniteRtol
    );
}

// ADV3: atol vector with one Inf element
#[test]
fn adv_atol_inf_element() {
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Vector(vec![1e-8, f64::INFINITY, 1e-8]),
        3,
        RuntimeMode::Strict,
    );
    // Inf is not < 0, so it's accepted (SciPy allows Inf tolerance)
    assert!(result.is_ok(), "Inf atol should be accepted");
}

// ADV4: n = usize::MAX (dimension overflow potential)
#[test]
fn adv_dimension_overflow() {
    // With scalar atol there's no length check, so this should succeed
    let result = validate_tol(
        ToleranceValue::Scalar(1e-3),
        ToleranceValue::Scalar(1e-6),
        usize::MAX,
        RuntimeMode::Strict,
    );
    assert!(result.is_ok(), "scalar atol with huge n should not panic");
}

// ADV5: first_step = f64::EPSILON (near zero positive)
#[test]
fn adv_first_step_epsilon() {
    let result = validate_first_step(f64::EPSILON, 0.0, 1.0);
    // EPSILON > 0 and EPSILON <= 1.0 → valid
    assert_eq!(result.unwrap(), f64::EPSILON);
}

// ADV6: t_bound - t0 = f64::EPSILON (extremely tight integration interval)
#[test]
fn adv_first_step_tight_interval() {
    let t0 = 0.0;
    let tb = f64::EPSILON;
    // Step exactly at boundary
    let result = validate_first_step(f64::EPSILON, t0, tb);
    assert!(result.is_ok(), "step == interval should be accepted");

    // Step exceeding the tiny interval
    let result2 = validate_first_step(2.0 * f64::EPSILON, t0, tb);
    assert!(result2.is_err(), "step > interval should be rejected");
}

// ADV7: NaN inputs to validate_first_step
#[test]
fn adv_first_step_nan() {
    let result = validate_first_step(f64::NAN, 0.0, 1.0);
    assert_eq!(
        result.expect_err("NaN first_step should fail closed"),
        IntegrateValidationError::NonFiniteFirstStep
    );
}

// ADV8: Inf inputs to validate_max_step
#[test]
fn adv_max_step_inf() {
    // SciPy uses INFINITY as default max_step
    let result = validate_max_step(f64::INFINITY);
    assert!(result.is_ok(), "Inf max_step should be accepted");
}

// ADV9: NaN to validate_max_step
#[test]
fn adv_max_step_nan() {
    let result = validate_max_step(f64::NAN);
    assert_eq!(
        result.expect_err("NaN max_step should fail closed"),
        IntegrateValidationError::NonFiniteMaxStep
    );
}

// ADV10: negative Inf to validate_max_step
#[test]
fn adv_max_step_neg_inf() {
    let result = validate_max_step(f64::NEG_INFINITY);
    assert!(result.is_err(), "negative Inf max_step should be rejected");
}

/// Check all sampled Simpson entry points on the same geometry in different
/// coordinate units. Extreme-scale oracle values use normalized coordinates;
/// the analytic change of variables avoids SciPy's own squared-width overflow.
#[test]
fn sampled_simpson_scipy() {
    use fsci_integrate::{cumulative_simpson, simpson, simpson_irregular};
    use std::process::Command;

    let python = std::env::var("FSCI_PYTHON")
        .or_else(|_| std::env::var("SCIPY_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let script = r#"
import numpy as np
import scipy
from scipy.integrate import simpson, cumulative_simpson
assert scipy.__version__ == '1.17.1', scipy.__version__
assert np.__version__ == '2.4.3', np.__version__
t = np.array([0., .25, .75, 1.5, 2., 3.])
for kind in range(3):
    values = (t*t, np.sin(t), np.exp(-t))[kind]
    for n in range(3, 7):
        for scale in [1e-200, 1e-20, 1., 1e20, 1e200]:
            oracle_scale = scale if 1e-100 <= scale <= 1e100 else 1.
            x, y = t[:n]*oracle_scale, values[:n]/oracle_scale
            print(kind, n, scale, float(simpson(y, x=x)),
                  *[float(v) for v in cumulative_simpson(y, x=x)])
"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output();
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let output = match output {
        Ok(output) if output.status.success() => output,
        other => {
            assert!(!required, "required pinned SciPy oracle failed: {other:?}");
            eprintln!("SKIP sampled_simpson_scipy: pinned SciPy oracle unavailable: {other:?}");
            return;
        }
    };
    let agrees = |actual: f64, expected: f64| {
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= 2e-13 * (1.0 + expected.abs())
    };
    assert!(agrees(1.0, 1.0));
    assert!(
        !agrees(1.0001, 1.0),
        "comparison must detect a wrong result"
    );
    assert!(!agrees(f64::NAN, 1.0));
    let grid = [0.0_f64, 0.25, 0.75, 1.5, 2.0, 3.0];
    let text = String::from_utf8(output.stdout).expect("oracle emits UTF-8");
    let mut compared = 0;
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert!(fields.len() >= 6, "invalid oracle row: {line}");
        let kind: usize = fields[0].parse().unwrap();
        let n: usize = fields[1].parse().unwrap();
        let scale: f64 = fields[2].parse().unwrap();
        assert!((3..=6).contains(&n));
        assert_eq!(fields.len(), n + 3, "invalid cumulative shape: {line}");
        let expected: f64 = fields[3].parse().unwrap();
        let x: Vec<_> = grid[..n].iter().map(|t| t * scale).collect();
        let y: Vec<_> = grid[..n]
            .iter()
            .map(|&t| {
                let value = match kind {
                    0 => t * t,
                    1 => t.sin(),
                    2 => (-t).exp(),
                    _ => panic!("unknown oracle family {kind}"),
                };
                value / scale
            })
            .collect();
        for (name, actual) in [
            ("simpson", simpson(&y, &x).expect("valid samples").integral),
            ("simpson_irregular", simpson_irregular(&y, &x)),
        ] {
            assert!(agrees(actual, expected), "{name}: {actual} != {line}");
        }
        let cumulative = cumulative_simpson(&y, &x).expect("increasing grid");
        assert_eq!(cumulative.len(), n - 1);
        for (i, value) in cumulative.iter().enumerate() {
            let expected: f64 = fields[4 + i].parse().unwrap();
            assert!(
                agrees(*value, expected),
                "cumulative[{i}]: {value} != {expected}, case={line}"
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 60, "every declared oracle case must be compared");
    eprintln!("sampled Simpson: {compared} cases, 180 routine comparisons against SciPy 1.17.1");
}

/// Exercise the public quadrature API against a required, versioned SciPy
/// process. The large-scale references are evaluated in normalized coordinates
/// so the oracle's own intermediate sum does not overflow.
#[test]
fn de_quadrature_scipy() {
    use fsci_integrate::{QuadResult, nsum, tanhsinh};
    use std::process::Command;

    let python = std::env::var("FSCI_PYTHON")
        .or_else(|_| std::env::var("SCIPY_PYTHON"))
        .unwrap_or_else(|_| "python3".into());
    let script = r#"
import numpy as np
import scipy
from scipy.integrate import tanhsinh, nsum
assert scipy.__version__ == '1.17.1', scipy.__version__
assert np.__version__ == '2.4.3', np.__version__
def integral(name, f, a, b, rtol=1e-12, scale=1.):
    with np.errstate(all='ignore'):
        result = tanhsinh(f, a, b, atol=0., rtol=rtol, maxlevel=10)
    print(name, float(result.integral)*scale, int(result.success),
          float(result.error)*abs(scale))
integral('smooth', np.exp, 0., 1.)
integral('tiny', np.exp, 0., 1., scale=1e-300)
integral('huge', np.exp, 0., 1., scale=5e307)
integral('reverse_huge', np.exp, 1., 0., scale=5e307)
integral('left_singular', lambda x: 1/np.sqrt(x), 0., 1.)
integral('shift_loose', lambda x: np.exp(-(x-1))/np.sqrt(x-1), 1., np.inf, 1e-6)
integral('shift_tight', lambda x: np.exp(-(x-1))/np.sqrt(x-1), 1., np.inf)
integral('mirror_tight', lambda x: np.exp(x+1)/np.sqrt(-1-x), -np.inf, -1.)
integral('right_singular', lambda x: (1-x)**(-0.75), 0., 1.)
integral('gaussian', lambda x: np.exp(-x*x), -np.inf, np.inf)
for name, f, a, b, step in [
    ('finite_series', lambda x: x*x, -2., 2., 0.5),
    ('infinite_series', lambda x: 1/(x*x), 1., np.inf, 1.),
]:
    result = nsum(f, a, b, step=step, tolerances={'atol': 0., 'rtol': 1e-11})
    print(name, float(result.sum), int(result.success), float(result.error))
"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output();
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let output = match output {
        Ok(output) if output.status.success() => output,
        other => {
            assert!(!required, "required pinned SciPy oracle failed: {other:?}");
            eprintln!("SKIP de_quadrature_scipy: pinned SciPy oracle unavailable: {other:?}");
            return;
        }
    };
    let text = String::from_utf8(output.stdout).expect("oracle emits UTF-8");
    let mut compared = 0;
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 4, "invalid oracle row: {line}");
        let expected: f64 = fields[1].parse().unwrap();
        let expected_success = match fields[2] {
            "0" => false,
            "1" => true,
            value => panic!("invalid oracle status: {value}"),
        };
        let oracle_error: f64 = fields[3].parse().unwrap();
        let result = match fields[0] {
            "smooth" => tanhsinh(f64::exp, 0.0, 1.0, 0.0, 1e-12, 10),
            "tiny" => tanhsinh(|x| 1e-300 * x.exp(), 0.0, 1.0, 0.0, 1e-12, 10),
            "huge" => tanhsinh(|x| 5e307 * x.exp(), 0.0, 1.0, 0.0, 1e-12, 10),
            "reverse_huge" => tanhsinh(|x| 5e307 * x.exp(), 1.0, 0.0, 0.0, 1e-12, 10),
            "left_singular" => tanhsinh(|x| 1.0 / x.sqrt(), 0.0, 1.0, 0.0, 1e-12, 10),
            "shift_loose" | "shift_tight" => tanhsinh(
                |x| (-(x - 1.0)).exp() / (x - 1.0).sqrt(),
                1.0,
                f64::INFINITY,
                0.0,
                if fields[0] == "shift_loose" {
                    1e-6
                } else {
                    1e-12
                },
                10,
            ),
            "mirror_tight" => tanhsinh(
                |x| (x + 1.0).exp() / (-1.0 - x).sqrt(),
                f64::NEG_INFINITY,
                -1.0,
                0.0,
                1e-12,
                10,
            ),
            "right_singular" => tanhsinh(|x| (1.0 - x).powf(-0.75), 0.0, 1.0, 0.0, 1e-12, 10),
            "gaussian" => tanhsinh(
                |x| (-x * x).exp(),
                f64::NEG_INFINITY,
                f64::INFINITY,
                0.0,
                1e-12,
                10,
            ),
            "finite_series" | "infinite_series" => {
                let sum = if fields[0] == "finite_series" {
                    nsum(|x| x * x, -2.0, 2.0, 0.5, 0.0, 1e-11)
                } else {
                    nsum(|x| 1.0 / (x * x), 1.0, f64::INFINITY, 1.0, 0.0, 1e-11)
                };
                QuadResult {
                    integral: sum.sum,
                    error: sum.error,
                    neval: sum.nfev,
                    converged: sum.converged,
                }
            }
            name => panic!("unexpected oracle case: {name}"),
        };
        assert!(expected.is_finite() && oracle_error.is_finite(), "{line}");
        assert!(result.integral.is_finite() && result.error.is_finite());
        assert_eq!(
            result.converged, expected_success,
            "{}: {result:?}",
            fields[0]
        );
        let relative_tolerance = if fields[0] == "shift_loose" {
            1e-6
        } else {
            5e-11
        };
        let allowance = if expected_success {
            relative_tolerance * expected.abs()
        } else {
            result.error.max(oracle_error)
        };
        let difference = (result.integral - expected).abs();
        assert!(
            difference <= allowance,
            "{}: Rust {result:?}, SciPy {expected}, difference {difference}",
            fields[0]
        );
        assert!(result.neval > 0);
        eprintln!(
            "{}: success={}, difference={difference:e}, error={:e}",
            fields[0], result.converged, result.error
        );
        compared += 1;
    }
    assert_eq!(
        compared, 12,
        "every case must be compared against the live oracle"
    );
}

#[test]
fn gauss_kronrod_global_error_matches_live_scipy() {
    use fsci_integrate::{QuadOptions, gauss_kronrod_quad};
    use std::process::Command;

    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = Command::new(&python)
        .args(["-c", "import scipy.integrate; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP gauss_kronrod: SciPy unavailable; compared cases = 0");
        return;
    }

    let script = r#"
import json, math, os, warnings
import numpy as np
import scipy
from scipy.integrate import quad
if os.environ.get('FSCI_REQUIRE_SCIPY_ORACLE') == '1':
    assert (scipy.__version__, np.__version__) == ('1.17.1', '2.4.3')
cusp = lambda x: abs(x-1/3)**.5 + abs(x-2/3)**.5
cases = [
    (math.exp, -2., 3., 1e-10, 1e-10, 128),
    (cusp, 0., 1., 1e-8, 0., 128),
    (lambda x: 1/math.sqrt(x), 0., 1., 1e-8, 0., 256),
    (lambda x: math.sin(100*x), 0., 1., 1e-10, 0., 128),
    (lambda x: math.exp(-x*x), -3., 2., 1e-11, 1e-11, 128),
    (cusp, 1., 0., 1e-8, 0., 128),
    (lambda x: x**12, 0., 1., 1e-12, 1e-12, 64),
    (lambda x: 1e-200*math.exp(x), 0., 1., 0., 1e-11, 64),
    (math.sin, 0., math.tau, 0., 1e-10, 64),
    (lambda x: math.sin(10000*x), 0., 1., 1e-12, 0., 6),
]
rows = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for index, (fun,a,b,atol,rtol,limit) in enumerate(cases):
        result = quad(fun,a,b,epsabs=atol,epsrel=rtol,limit=limit,full_output=True)
        rows.append({'id':index, 'integral':result[0], 'error':result[1],
                     'success':len(result)==3})
print(json.dumps({'scipy':scipy.__version__,'numpy':np.__version__,'rows':rows}))
"#;
    let output = Command::new(&python).args(["-c", script]).output().unwrap();
    assert!(
        output.status.success(),
        "SciPy quadrature failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let rows = oracle["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 10, "every oracle case must be present");
    let cases = [
        (-2.0, 3.0, 1e-10, 1e-10, 128),
        (0.0, 1.0, 1e-8, 0.0, 128),
        (0.0, 1.0, 1e-8, 0.0, 256),
        (0.0, 1.0, 1e-10, 0.0, 128),
        (-3.0, 2.0, 1e-11, 1e-11, 128),
        (1.0, 0.0, 1e-8, 0.0, 128),
        (0.0, 1.0, 1e-12, 1e-12, 64),
        (0.0, 1.0, 0.0, 1e-11, 64),
        (0.0, std::f64::consts::TAU, 0.0, 1e-10, 64),
        (0.0, 1.0, 1e-12, 0.0, 6),
    ];
    let mut compared = 0;
    for (id, (&(a, b, epsabs, epsrel, limit), row)) in cases.iter().zip(rows).enumerate() {
        assert_eq!(row["id"].as_u64().unwrap(), id as u64);
        let calls = std::cell::Cell::new(0usize);
        let result = gauss_kronrod_quad(
            |x| {
                calls.set(calls.get() + 1);
                match id {
                    0 => x.exp(),
                    1 | 5 => (x - 1.0 / 3.0).abs().sqrt() + (x - 2.0 / 3.0).abs().sqrt(),
                    2 => 1.0 / x.sqrt(),
                    3 => (100.0 * x).sin(),
                    4 => (-x * x).exp(),
                    6 => x.powi(12),
                    7 => 1e-200 * x.exp(),
                    8 => x.sin(),
                    9 => (10_000.0 * x).sin(),
                    _ => unreachable!(),
                }
            },
            a,
            b,
            QuadOptions {
                epsabs,
                epsrel,
                limit,
            },
        );
        let expected = row["integral"].as_f64().unwrap();
        let expected_success = row["success"].as_bool().unwrap();
        assert!(expected.is_finite() && row["error"].as_f64().unwrap().is_finite());
        assert!(result.integral.is_finite() && result.error.is_finite());
        assert_eq!(result.neval, calls.get());
        assert!(result.neval > 0 && result.neval <= 15 + 30 * (limit - 1));
        assert_eq!(result.converged, expected_success, "case {id}: {result:?}");
        if expected_success {
            let tolerance = epsabs.max(epsrel * expected.abs());
            assert!(
                (result.integral - expected).abs() <= tolerance,
                "case {id}: Rust {result:?} versus SciPy {expected}"
            );
            assert!(result.error <= epsabs.max(epsrel * result.integral.abs()));
        } else {
            // A capped oscillatory estimate has no accuracy certificate in
            // either algorithm. Compare the failure status, not two different
            // unfinished estimates; the evaluation budget is checked above.
            assert!(id >= 8);
        }
        compared += 1;
    }
    assert_eq!(compared, 10);
    eprintln!(
        "gauss_kronrod compared {compared}/10 cases; SciPy {}, NumPy {}",
        oracle["scipy"], oracle["numpy"]
    );
}
