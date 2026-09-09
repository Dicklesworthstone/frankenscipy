#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult, SpecialTensor, beta, betainc, betaincinv, betaln, gammainc,
    gammaincc, gammaincinv,
};
use libfuzzer_sys::fuzz_target;

const ABS_TOL: f64 = 1.0e-7;
const REL_TOL: f64 = 1.0e-5;

#[derive(Clone, Copy, Debug, Arbitrary)]
enum EdgeF64 {
    Finite(f64),
    Zero,
    One,
    Half,
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
            Self::One => 1.0,
            Self::Half => 0.5,
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
            raw.clamp(-32.0, 32.0)
        } else if raw.is_sign_positive() {
            32.0
        } else if raw.is_sign_negative() {
            -32.0
        } else {
            0.0
        }
    }

    fn positive_param(self) -> f64 {
        self.bounded().abs().clamp(1.0e-6, 16.0)
    }

    fn unit(self) -> f64 {
        match self {
            Self::Zero => 0.0,
            Self::One => 1.0,
            Self::Half => 0.5,
            _ => {
                let bounded = self.bounded();
                0.5 + 0.5 * (bounded / (1.0 + bounded.abs()))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct IncompleteInput {
    a: EdgeF64,
    b: EdgeF64,
    x: EdgeF64,
    y: EdgeF64,
    alt: EdgeF64,
    hardened: bool,
    vectorize: bool,
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

fn real_from_result(result: SpecialResult) -> Option<f64> {
    match result.ok()? {
        SpecialTensor::RealScalar(value) => Some(value),
        _ => None,
    }
}

fn complex_from_result(result: SpecialResult) -> Option<Complex64> {
    match result.ok()? {
        SpecialTensor::RealScalar(value) => Some(Complex64::from_real(value)),
        SpecialTensor::ComplexScalar(value) => Some(value),
        _ => None,
    }
}

fn check_vec_len(result: SpecialResult, expected: usize) {
    match result {
        Ok(SpecialTensor::RealVec(values)) => assert_eq!(values.len(), expected),
        Ok(SpecialTensor::ComplexVec(values)) => assert_eq!(values.len(), expected),
        Ok(other) => assert!(
            matches!(
                other,
                SpecialTensor::RealVec(_) | SpecialTensor::ComplexVec(_)
            ),
            "expected vector result, got {other:?}"
        ),
        Err(_) => {}
    }
}

fuzz_target!(|input: IncompleteInput| {
    let mode = mode_from_flag(input.hardened);

    let a_real = input.a.positive_param();
    let b_real = input.b.positive_param();
    let x_real = input.x.positive_param();
    let x_beta = input.x.unit();
    let alt_real = input.alt.positive_param();

    let a_complex = Complex64::new(input.a.raw(), input.alt.raw());
    let b_complex = Complex64::new(input.b.raw(), -input.alt.raw());
    let x_complex = Complex64::new(input.x.raw(), input.y.raw());

    let _ = beta(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::ComplexScalar(b_complex),
        mode,
    );
    let _ = betaln(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::ComplexScalar(b_complex),
        mode,
    );
    let _ = betainc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::RealScalar(b_real),
        &SpecialTensor::ComplexScalar(x_complex),
        mode,
    );
    let _ = betainc(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::ComplexScalar(b_complex),
        &SpecialTensor::ComplexScalar(x_complex),
        mode,
    );
    let _ = gammainc(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::RealScalar(x_real),
        mode,
    );
    let _ = gammaincc(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::RealScalar(x_real),
        mode,
    );
    let _ = gammainc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::ComplexScalar(x_complex),
        mode,
    );
    let _ = gammaincc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::ComplexScalar(x_complex),
        mode,
    );

    if input.vectorize {
        check_vec_len(
            beta(
                &SpecialTensor::RealVec(vec![a_real, alt_real]),
                &SpecialTensor::ComplexScalar(b_complex),
                mode,
            ),
            2,
        );
        check_vec_len(
            gammainc(
                &SpecialTensor::RealVec(vec![a_real, alt_real]),
                &SpecialTensor::ComplexScalar(x_complex),
                mode,
            ),
            2,
        );
    }

    let beta_ab = complex_from_result(beta(
        &SpecialTensor::ComplexScalar(a_complex),
        &SpecialTensor::ComplexScalar(b_complex),
        mode,
    ));
    let beta_ba = complex_from_result(beta(
        &SpecialTensor::ComplexScalar(b_complex),
        &SpecialTensor::ComplexScalar(a_complex),
        mode,
    ));
    if let (Some(lhs), Some(rhs)) = (beta_ab, beta_ba)
        && lhs.is_finite()
        && rhs.is_finite()
    {
        assert!(approx_eq_complex(lhs, rhs), "beta symmetry mismatch");
    }

    let gamma_p = real_from_result(gammainc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::RealScalar(x_real),
        mode,
    ));
    let gamma_q = real_from_result(gammaincc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::RealScalar(x_real),
        mode,
    ));
    if let (Some(p), Some(q)) = (gamma_p, gamma_q)
        && p.is_finite()
        && q.is_finite()
    {
        assert!(approx_eq_scalar(p + q, 1.0), "gammainc complement mismatch");
    }

    let beta_y = real_from_result(betainc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::RealScalar(b_real),
        &SpecialTensor::RealScalar(x_beta),
        mode,
    ));
    if let Some(y) =
        beta_y.filter(|value| value.is_finite() && *value > 1.0e-9 && *value < 1.0 - 1.0e-9)
        && let Some(x_roundtrip) = real_from_result(betaincinv(
            &SpecialTensor::RealScalar(a_real),
            &SpecialTensor::RealScalar(b_real),
            &SpecialTensor::RealScalar(y),
            mode,
        ))
        && x_roundtrip.is_finite()
    {
        assert!(
            approx_eq_scalar(x_roundtrip, x_beta),
            "betaincinv roundtrip mismatch: expected {x_beta}, got {x_roundtrip}"
        );
    }

    let gamma_y = real_from_result(gammainc(
        &SpecialTensor::RealScalar(a_real),
        &SpecialTensor::RealScalar(x_real),
        mode,
    ));
    if let Some(y) =
        gamma_y.filter(|value| value.is_finite() && *value > 1.0e-9 && *value < 1.0 - 1.0e-9)
        && let Some(x_roundtrip) = real_from_result(gammaincinv(
            &SpecialTensor::RealScalar(a_real),
            &SpecialTensor::RealScalar(y),
            mode,
        ))
        && x_roundtrip.is_finite()
    {
        assert!(
            approx_eq_scalar(x_roundtrip, x_real),
            "gammaincinv roundtrip mismatch: expected {x_real}, got {x_roundtrip}"
        );
    }
    let _ = fsci_special::take_special_traces();
});
