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
