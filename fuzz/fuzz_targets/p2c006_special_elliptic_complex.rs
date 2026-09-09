#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    ellipe, ellipeinc, ellipk, ellipkinc, ellipkm1,
};
use libfuzzer_sys::fuzz_target;

type UnarySpecialFn = fn(&SpecialTensor, RuntimeMode) -> SpecialResult;
type BinarySpecialFn = fn(&SpecialTensor, &SpecialTensor, RuntimeMode) -> SpecialResult;

const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum EdgeF64 {
    Finite(f64),
    Zero,
    NegZero,
    One,
    NegOne,
    Half,
    NegHalf,
    Tiny,
    NegTiny,
    PosInf,
    NegInf,
    Nan,
    PiOverTwo,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e3, 1.0e3),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::PosInf => f64::INFINITY,
            Self::NegInf => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
            Self::PiOverTwo => std::f64::consts::FRAC_PI_2,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct EllipticInput {
    m_re: EdgeF64,
    m_im: EdgeF64,
    phi_re: EdgeF64,
    phi_im: EdgeF64,
    hardened: bool,
    vectorize: bool,
    mismatch_lengths: bool,
}

fn mode_from_flag(hardened: bool) -> RuntimeMode {
    if hardened {
        RuntimeMode::Hardened
    } else {
        RuntimeMode::Strict
    }
}

fn approx_eq_scalar(lhs: f64, rhs: f64) -> bool {
    if !(lhs.is_finite() && rhs.is_finite()) {
        return false;
    }
    let scale = lhs.abs().max(rhs.abs());
    (lhs - rhs).abs() <= ABS_TOL + REL_TOL * scale
}

fn approx_eq_complex(lhs: Complex64, rhs: Complex64) -> bool {
    approx_eq_scalar(lhs.re, rhs.re) && approx_eq_scalar(lhs.im, rhs.im)
}

fn complex_from_result(result: SpecialResult) -> Option<Complex64> {
    match result.ok()? {
        RealScalar(value) => Some(Complex64::from_real(value)),
        ComplexScalar(value) => Some(value),
        _ => None,
    }
}

fn complex_vec_from_result(result: SpecialResult) -> Option<Vec<Complex64>> {
    match result.ok()? {
        RealVec(values) => Some(values.into_iter().map(Complex64::from_real).collect()),
        ComplexVec(values) => Some(values),
        _ => None,
    }
}

fn check_unary(name: &str, func: UnarySpecialFn, z: Complex64, mode: RuntimeMode, vectorize: bool) {
    let _ = func(&ComplexScalar(z), mode);

    if z.is_finite() && z.im != 0.0 {
        let lhs = complex_from_result(func(&ComplexScalar(z), mode));
        let rhs = complex_from_result(func(&ComplexScalar(z.conj()), mode));
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: conjugation mismatch for {z:?}"
            );
        }
    }

    if vectorize {
        let vec_result = complex_vec_from_result(func(&ComplexVec(vec![z, z.conj()]), mode));
        if let Some(values) = vec_result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            if values[0].is_finite() && values[1].is_finite() && z.is_finite() && z.im != 0.0 {
                assert!(
                    approx_eq_complex(values[1], values[0].conj()),
                    "{name}: vector conjugation mismatch for {z:?}"
                );
            }
        }
    }

    if z.im == 0.0 {
        let real = z.re;
        let real_result = func(&RealScalar(real), mode);
        let complex_result =
            complex_from_result(func(&ComplexScalar(Complex64::from_real(real)), mode));
        if let (Ok(RealScalar(real_value)), Some(complex_value)) = (real_result, complex_result)
            && real_value.is_finite()
            && complex_value.is_finite()
        {
            assert!(
                approx_eq_scalar(real_value, complex_value.re),
                "{name}: real-axis reduction mismatch for {real}"
            );
            assert!(
                complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                "{name}: expected near-real output on real axis for {real}"
            );
        }
    }
}

fn check_binary(
    name: &str,
    func: BinarySpecialFn,
    phi: Complex64,
    m: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
    mismatch_lengths: bool,
) {
    let _ = func(&ComplexScalar(phi), &ComplexScalar(m), mode);

    if phi.is_finite() && m.is_finite() && (phi.im != 0.0 || m.im != 0.0) {
        let lhs = complex_from_result(func(&ComplexScalar(phi), &ComplexScalar(m), mode));
        let rhs = complex_from_result(func(
            &ComplexScalar(phi.conj()),
            &ComplexScalar(m.conj()),
            mode,
        ));
        if let (Some(left), Some(right)) = (lhs, rhs)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(right, left.conj()),
                "{name}: conjugation mismatch for phi={phi:?}, m={m:?}"
            );
        }
    }

    if vectorize {
        let vec_result = complex_vec_from_result(func(
            &ComplexVec(vec![phi, phi.conj()]),
            &ComplexVec(vec![m, m.conj()]),
            mode,
        ));
        if let Some(values) = vec_result {
            assert_eq!(values.len(), 2, "{name}: vector output length mismatch");
            if values[0].is_finite()
                && values[1].is_finite()
                && phi.is_finite()
                && m.is_finite()
                && (phi.im != 0.0 || m.im != 0.0)
            {
                assert!(
                    approx_eq_complex(values[1], values[0].conj()),
                    "{name}: vector conjugation mismatch for phi={phi:?}, m={m:?}"
                );
            }
        }
    }

    if mismatch_lengths {
        assert!(
            func(
                &ComplexVec(vec![phi, phi.conj()]),
                &ComplexVec(vec![m]),
                mode
            )
            .is_err(),
            "{name}: mismatched vector lengths should fail closed"
        );
    }

    if phi.im == 0.0 && m.im == 0.0 {
        let real_result = func(&RealScalar(phi.re), &RealScalar(m.re), mode);
        let complex_result = complex_from_result(func(
            &ComplexScalar(Complex64::from_real(phi.re)),
            &ComplexScalar(Complex64::from_real(m.re)),
            mode,
        ));
        if let (Ok(RealScalar(real_value)), Some(complex_value)) = (real_result, complex_result)
            && real_value.is_finite()
            && complex_value.is_finite()
        {
            assert!(
                approx_eq_scalar(real_value, complex_value.re),
                "{name}: real-axis reduction mismatch for phi={:?}, m={:?}",
                phi.re,
                m.re
            );
            assert!(
                complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                "{name}: expected near-real output on real axis"
            );
        }
    }
}

fuzz_target!(|input: EllipticInput| {
    let mode = mode_from_flag(input.hardened);
    let m = Complex64::new(input.m_re.raw(), input.m_im.raw());
    let phi = Complex64::new(input.phi_re.raw(), input.phi_im.raw());

    for (name, func) in [
        ("ellipk", ellipk as UnarySpecialFn),
        ("ellipe", ellipe as UnarySpecialFn),
        ("ellipkm1", ellipkm1 as UnarySpecialFn),
    ] {
        check_unary(name, func, m, mode, input.vectorize);
    }

    for (name, func) in [
        ("ellipkinc", ellipkinc as BinarySpecialFn),
        ("ellipeinc", ellipeinc as BinarySpecialFn),
    ] {
        check_binary(
            name,
            func,
            phi,
            m,
            mode,
            input.vectorize,
            input.mismatch_lengths,
        );
    }

    if m.is_finite() {
        let p = Complex64::new(1.0 - m.re, -m.im);
        let ellipkm1_value = complex_from_result(ellipkm1(&ComplexScalar(p), mode));
        let ellipk_value = complex_from_result(ellipk(&ComplexScalar(m), mode));
        if let (Some(left), Some(right)) = (ellipkm1_value, ellipk_value)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(left, right),
                "ellipkm1 complement identity mismatch for p={p:?}, m={m:?}"
            );
        }

        let pi_over_two = Complex64::from_real(std::f64::consts::FRAC_PI_2);
        let f_complete = complex_from_result(ellipkinc(
            &ComplexScalar(pi_over_two),
            &ComplexScalar(m),
            mode,
        ));
        let k_complete = complex_from_result(ellipk(&ComplexScalar(m), mode));
        if let (Some(left), Some(right)) = (f_complete, k_complete)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(left, right),
                "ellipkinc(pi/2, m) mismatch for m={m:?}"
            );
        }

        let e_incomplete = complex_from_result(ellipeinc(
            &ComplexScalar(pi_over_two),
            &ComplexScalar(m),
            mode,
        ));
        let e_complete = complex_from_result(ellipe(&ComplexScalar(m), mode));
        if let (Some(left), Some(right)) = (e_incomplete, e_complete)
            && left.is_finite()
            && right.is_finite()
        {
            assert!(
                approx_eq_complex(left, right),
                "ellipeinc(pi/2, m) mismatch for m={m:?}"
            );
        }
    }
    let _ = fsci_special::take_special_traces();
});
