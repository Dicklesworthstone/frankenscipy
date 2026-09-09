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

#[test]
fn test_complex_gammaln_large_negative_and_non_finite_no_hang() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    for z in [
        Complex64::new(f64::NEG_INFINITY, 0.0),
        Complex64::new(f64::NEG_INFINITY, 1.0),
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::NAN),
        Complex64::new(-1.0e300, 1.0),
        Complex64::new(-1.0e300, 0.0),
        Complex64::new(-1000.5, 1.0),
        Complex64::new(-100.5, 1.0),
        Complex64::new(-50.5, 1.0),
    ] {
        let res = fsci_special::gammaln(&SpecialTensor::ComplexScalar(z), mode);
        assert!(res.is_ok(), "gammaln({z:?}) failed");
    }
}

#[test]
fn test_gammaln_subnormal_real_axis_reduction() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    for &x in &[
        1.6e-322,
        1.0e-320,
        1.0e-310,
        f64::MIN_POSITIVE * 0.5,
        f64::MIN_POSITIVE,
        1.0e-300,
        1.0e-100,
        1.0e-10,
        0.1,
        0.499,
    ] {
        let real_res = fsci_special::gammaln(&SpecialTensor::RealScalar(x), mode).expect("real");
        let comp_res =
            fsci_special::gammaln(&SpecialTensor::ComplexScalar(Complex64::from_real(x)), mode)
                .expect("complex");
        if let (SpecialTensor::RealScalar(r), SpecialTensor::ComplexScalar(c)) =
            (real_res, comp_res)
        {
            let scale = r.abs().max(c.re.abs());
            let diff = (r - c.re).abs();
            assert!(
                diff <= 1e-8 + 1e-6 * scale,
                "gammaln({x}): real {r} != complex {c:?}, diff {diff}"
            );
            assert_eq!(
                c.im, 0.0,
                "gammaln({x}): imaginary part expected 0.0, got {}",
                c.im
            );
        }
    }
}

#[test]
fn test_gammaln_negative_subnormal_real_axis_reduction() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    for &x in &[
        -8.0e-323,
        -1.6e-322,
        -1.0e-320,
        -1.0e-310,
        -f64::MIN_POSITIVE * 0.5,
        -f64::MIN_POSITIVE,
        -1.0e-300,
        -1.0e-100,
        -1.0e-10,
        -0.1,
        -0.499,
    ] {
        let real_res = fsci_special::gammaln(&SpecialTensor::RealScalar(x), mode).expect("real");
        let comp_res =
            fsci_special::gammaln(&SpecialTensor::ComplexScalar(Complex64::from_real(x)), mode)
                .expect("complex");
        if let (SpecialTensor::RealScalar(r), SpecialTensor::ComplexScalar(c)) =
            (real_res, comp_res)
        {
            let scale = r.abs().max(c.re.abs());
            let diff = (r - c.re).abs();
            assert!(
                diff <= 1e-8 + 1e-6 * scale,
                "gammaln({x}): real {r} != complex {c:?}, diff {diff}"
            );
            // Verify winding behavior: imaginary part on negative real axis must be integer multiple of π
            let winding = c.im / std::f64::consts::PI;
            assert!(
                (winding - winding.round()).abs() <= 1e-8 + 1e-6 * winding.abs().max(1.0),
                "gammaln({x}): expected integer-multiple of π imaginary part, got {c:?}"
            );
        }
    }
}

#[test]
fn test_complex_erfinv_conjugation_and_accuracy() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    for z in [
        Complex64::new(5.0e-322, 1360.2499972730875),
        Complex64::new(-5.0e-322, 1360.2499972730875),
        Complex64::new(5.0e-322, -1360.2499972730875),
        Complex64::new(-5.0e-322, -1360.2499972730875),
        Complex64::new(0.5, 2.0),
        Complex64::new(-0.5, 2.0),
        Complex64::new(0.5, -2.0),
        Complex64::new(-0.5, -2.0),
        Complex64::new(10.0, 5.0),
    ] {
        let z_conj = z.conj();
        let res = fsci_special::erfinv(&SpecialTensor::ComplexScalar(z), mode).expect("erfinv z");
        let res_conj = fsci_special::erfinv(&SpecialTensor::ComplexScalar(z_conj), mode)
            .expect("erfinv z_conj");

        if let (SpecialTensor::ComplexScalar(c), SpecialTensor::ComplexScalar(cc)) = (res, res_conj)
            && c.is_finite()
            && cc.is_finite()
        {
            assert_eq!(
                cc,
                c.conj(),
                "erfinv conjugation mismatch for {z:?}: {c:?} vs {cc:?}"
            );
        }
    }
}

#[test]
fn test_ellipkinc_conjugation() {
    use fsci_special::Complex64;
    let mode = RuntimeMode::Strict;
    let phi = 6.280026599764828;
    let m = 1.0;
    let real_res = fsci_special::ellipeinc(
        &SpecialTensor::RealScalar(phi),
        &SpecialTensor::RealScalar(m),
        mode,
    )
    .expect("real");
    let comp_res = fsci_special::ellipeinc(
        &SpecialTensor::ComplexScalar(Complex64::from_real(phi)),
        &SpecialTensor::ComplexScalar(Complex64::from_real(m)),
        mode,
    )
    .expect("complex");
    if let (SpecialTensor::RealScalar(r), SpecialTensor::ComplexScalar(c)) = (real_res, comp_res) {
        assert_eq!(r, c.re);
        assert_eq!(c.im, 0.0);
    }
    for phi in [
        Complex64::new(0.0, std::f64::consts::FRAC_PI_2),
        Complex64::new(0.5, 1.0),
        Complex64::new(-0.5, 1.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(1.0, 0.0),
    ] {
        for m in [
            Complex64::new(-0.5, 0.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(0.5, -0.25),
            Complex64::new(-2.0, 0.5),
            Complex64::new(2.0, -0.5),
        ] {
            let res_k = fsci_special::ellipkinc(
                &SpecialTensor::ComplexScalar(phi),
                &SpecialTensor::ComplexScalar(m),
                mode,
            )
            .expect("k phi, m");
            let res_k_conj = fsci_special::ellipkinc(
                &SpecialTensor::ComplexScalar(phi.conj()),
                &SpecialTensor::ComplexScalar(m.conj()),
                mode,
            )
            .expect("k conj");
            if let (SpecialTensor::ComplexScalar(c), SpecialTensor::ComplexScalar(cc)) =
                (res_k, res_k_conj)
                && c.is_finite()
                && cc.is_finite()
            {
                assert!(
                    (cc.re - c.conj().re).abs() <= 1e-7 && (cc.im - c.conj().im).abs() <= 1e-7,
                    "ellipkinc conjugation mismatch for phi={phi:?}, m={m:?}: {c:?} vs {cc:?}"
                );
            }

            let res_e = fsci_special::ellipeinc(
                &SpecialTensor::ComplexScalar(phi),
                &SpecialTensor::ComplexScalar(m),
                mode,
            )
            .expect("e phi, m");
            let res_e_conj = fsci_special::ellipeinc(
                &SpecialTensor::ComplexScalar(phi.conj()),
                &SpecialTensor::ComplexScalar(m.conj()),
                mode,
            )
            .expect("e conj");
            if let (SpecialTensor::ComplexScalar(c), SpecialTensor::ComplexScalar(cc)) =
                (res_e, res_e_conj)
                && c.is_finite()
                && cc.is_finite()
            {
                assert!(
                    (cc.re - c.conj().re).abs() <= 1e-7 && (cc.im - c.conj().im).abs() <= 1e-7,
                    "ellipeinc conjugation mismatch for phi={phi:?}, m={m:?}: {c:?} vs {cc:?}"
                );
            }
        }
    }
}
