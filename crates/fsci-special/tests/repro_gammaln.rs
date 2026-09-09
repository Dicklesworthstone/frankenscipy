use fsci_runtime::RuntimeMode;
use fsci_special::{SpecialTensor, gammaln};

#[test]
fn test_gammaln_overflow() {
    // gammaln(200) should be finite
    let x = SpecialTensor::RealScalar(200.0);
    let res = gammaln(&x, RuntimeMode::Strict).expect("gammaln(200)");
    if let SpecialTensor::RealScalar(v) = res {
        println!("gammaln(200) = {}", v);
        assert!(v.is_finite(), "gammaln(200) should be finite, got {}", v);
        assert!((v - 857.933).abs() < 1.0, "gammaln(200) ≈ 857.9, got {}", v);
    } else {
        panic!("expected scalar");
    }
}

#[test]
fn test_gammaln_near_negative_integer_accuracy() {
    let mode = RuntimeMode::Hardened;
    // Point near -1: -1.0 - 4*EPSILON. Catastrophic cancellation in (PI*x).sin() previously
    // corrupted gammaln_scalar here.
    let real = -1.0 - 4.0 * f64::EPSILON;
    let real_result = gammaln(&SpecialTensor::RealScalar(real), mode).expect("gammaln real");
    if let SpecialTensor::RealScalar(v) = real_result {
        // Scipy reference: scipy.special.gammaln(-1.0000000000000009) == 34.657359027997266
        let expected = 34.657359027997266;
        assert!(
            (v - expected).abs() <= 1.0e-12,
            "gammaln({real}) expected {expected}, got {v}"
        );
    } else {
        panic!("expected real scalar");
    }
}

#[test]
fn test_digamma_real_reduction_near_pole() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Hardened;
    for real in [
        -1.0 + 4.0 * f64::EPSILON,
        -1.0 - 4.0 * f64::EPSILON,
        -2.0 + 4.0 * f64::EPSILON,
        -0.5,
        0.1,
    ] {
        let real_res = fsci_special::digamma(&SpecialTensor::RealScalar(real), mode).expect("real");
        let comp_res = fsci_special::digamma(
            &SpecialTensor::ComplexScalar(Complex64::from_real(real)),
            mode,
        )
        .expect("complex");
        if let (SpecialTensor::RealScalar(r), SpecialTensor::ComplexScalar(c)) =
            (real_res, comp_res)
        {
            let scale = r.abs().max(c.re.abs());
            assert!(
                (r - c.re).abs() <= 1e-12 + 1e-12 * scale,
                "digamma({real}): real {r} != complex {c:?}"
            );
            assert!(c.im == 0.0, "digamma({real}): expected im == 0, got {c:?}");
        }
    }
}

#[test]
fn test_complex_gamma_exp_gammaln_consistency() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    let z = Complex64::new(-f64::MIN_POSITIVE, 0.0);
    let g_res = fsci_special::gamma(&SpecialTensor::ComplexScalar(z), mode).expect("gamma");
    let gl_res = fsci_special::gammaln(&SpecialTensor::ComplexScalar(z), mode).expect("gammaln");
    if let (SpecialTensor::ComplexScalar(gamma_z), SpecialTensor::ComplexScalar(gammaln_z)) =
        (g_res, gl_res)
    {
        let expected = gammaln_z.exp();
        let scale =
            (gamma_z.re.abs().max(gamma_z.im.abs())).max(expected.re.abs().max(expected.im.abs()));
        let tol = 1.0e-8 + 1.0e-6 * scale;
        assert!(
            (gamma_z.re - expected.re).abs() <= tol,
            "re mismatch: {} vs {}",
            gamma_z.re,
            expected.re
        );
        assert!(
            (gamma_z.im - expected.im).abs() <= tol,
            "im mismatch: {} vs {}",
            gamma_z.im,
            expected.im
        );
    } else {
        panic!("expected complex scalars");
    }
}
