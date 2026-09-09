#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    hankel1, hankel2, jv, yv,
};
use libfuzzer_sys::fuzz_target;

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
        }
    }

    fn bounded(self) -> f64 {
        let raw = self.raw();
        if raw.is_finite() {
            raw.clamp(-16.0, 16.0)
        } else if raw.is_sign_positive() {
            16.0
        } else if raw.is_sign_negative() {
            -16.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct HankelInput {
    order: EdgeF64,
    re: EdgeF64,
    im: EdgeF64,
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
        ComplexScalar(value) => Some(value),
        RealScalar(value) => Some(Complex64::from_real(value)),
        _ => None,
    }
}

fn complex_vec_from_result(result: SpecialResult) -> Option<Vec<Complex64>> {
    match result.ok()? {
        ComplexVec(values) => Some(values),
        RealVec(values) => Some(values.into_iter().map(Complex64::from_real).collect()),
        _ => None,
    }
}

fn mul_i(value: Complex64) -> Complex64 {
    Complex64::new(-value.im, value.re)
}

fn mul_neg_i(value: Complex64) -> Complex64 {
    Complex64::new(value.im, -value.re)
}

fn check_identity(v: f64, z: Complex64, mode: RuntimeMode) {
    let v_tensor = RealScalar(v);
    let j = complex_from_result(jv(&v_tensor, &ComplexScalar(z), mode));
    let y = complex_from_result(yv(&v_tensor, &ComplexScalar(z), mode));
    let h1 = complex_from_result(hankel1(&v_tensor, &ComplexScalar(z), mode));
    let h2 = complex_from_result(hankel2(&v_tensor, &ComplexScalar(z), mode));
    if let (Some(j_value), Some(y_value), Some(h1_value), Some(h2_value)) = (j, y, h1, h2)
        && j_value.is_finite()
        && y_value.is_finite()
        && h1_value.is_finite()
        && h2_value.is_finite()
    {
        assert!(
            approx_eq_complex(h1_value, j_value + mul_i(y_value)),
            "hankel1 identity mismatch for v={v:?}, z={z:?}"
        );
        assert!(
            approx_eq_complex(h2_value, j_value + mul_neg_i(y_value)),
            "hankel2 identity mismatch for v={v:?}, z={z:?}"
        );
    }
}

fuzz_target!(|input: HankelInput| {
    let mode = mode_from_flag(input.hardened);
    let v = input.order.bounded();
    let z = Complex64::new(input.re.raw(), input.im.raw());
    let v_tensor = RealScalar(v);

    let _ = hankel1(&v_tensor, &ComplexScalar(z), mode);
    let _ = hankel2(&v_tensor, &ComplexScalar(z), mode);

    check_identity(v, z, mode);

    if z.is_finite() && z.im != 0.0 {
        let h1_conj = complex_from_result(hankel1(&v_tensor, &ComplexScalar(z.conj()), mode));
        let h2_conj = complex_from_result(hankel2(&v_tensor, &ComplexScalar(z.conj()), mode));
        let h1 = complex_from_result(hankel1(&v_tensor, &ComplexScalar(z), mode));
        let h2 = complex_from_result(hankel2(&v_tensor, &ComplexScalar(z), mode));
        if let (Some(h1c), Some(h2c), Some(h1z), Some(h2z)) = (h1_conj, h2_conj, h1, h2)
            && h1c.is_finite()
            && h2c.is_finite()
            && h1z.is_finite()
            && h2z.is_finite()
        {
            assert!(
                approx_eq_complex(h1c, h2z.conj()),
                "hankel1(conj(z)) mismatch for v={v:?}, z={z:?}"
            );
            assert!(
                approx_eq_complex(h2c, h1z.conj()),
                "hankel2(conj(z)) mismatch for v={v:?}, z={z:?}"
            );
        }
    }

    if input.vectorize {
        let h1_values =
            complex_vec_from_result(hankel1(&v_tensor, &ComplexVec(vec![z, z.conj()]), mode));
        let h2_values =
            complex_vec_from_result(hankel2(&v_tensor, &ComplexVec(vec![z, z.conj()]), mode));
        if let (Some(h1_vec), Some(h2_vec)) = (h1_values, h2_values) {
            assert_eq!(h1_vec.len(), 2, "hankel1: vector output length mismatch");
            assert_eq!(h2_vec.len(), 2, "hankel2: vector output length mismatch");
            if h1_vec[0].is_finite()
                && h1_vec[1].is_finite()
                && h2_vec[0].is_finite()
                && h2_vec[1].is_finite()
                && z.is_finite()
                && z.im != 0.0
            {
                assert!(
                    approx_eq_complex(h1_vec[1], h2_vec[0].conj()),
                    "hankel1 vector conjugation mismatch for v={v:?}, z={z:?}"
                );
                assert!(
                    approx_eq_complex(h2_vec[1], h1_vec[0].conj()),
                    "hankel2 vector conjugation mismatch for v={v:?}, z={z:?}"
                );
            }
        }
    }

    if input.mismatch_lengths {
        assert!(
            hankel1(&RealVec(vec![v, v + 1.0]), &ComplexVec(vec![z]), mode).is_err(),
            "hankel1: mismatched vector lengths should fail closed"
        );
        assert!(
            hankel2(&RealVec(vec![v, v + 1.0]), &ComplexVec(vec![z]), mode).is_err(),
            "hankel2: mismatched vector lengths should fail closed"
        );
    }
    let _ = fsci_special::take_special_traces();
});
