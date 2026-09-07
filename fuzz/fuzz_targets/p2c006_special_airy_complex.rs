#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialTensor,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    ai, airy, bi,
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
            raw.clamp(-8.0, 8.0)
        } else if raw.is_sign_positive() {
            8.0
        } else if raw.is_sign_negative() {
            -8.0
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct AiryInput {
    re: EdgeF64,
    im: EdgeF64,
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

fn complex_scalar_from_tensor(tensor: &SpecialTensor) -> Option<Complex64> {
    match tensor {
        RealScalar(value) => Some(Complex64::from_real(*value)),
        ComplexScalar(value) => Some(*value),
        _ => None,
    }
}

fn complex_vec_from_tensor(tensor: &SpecialTensor) -> Option<Vec<Complex64>> {
    match tensor {
        RealVec(values) => Some(values.iter().copied().map(Complex64::from_real).collect()),
        ComplexVec(values) => Some(values.clone()),
        _ => None,
    }
}

fn airy_scalar_bundle(
    result: Result<Vec<SpecialTensor>, fsci_special::SpecialError>,
) -> Option<[Complex64; 4]> {
    let values = result.ok()?;
    if values.len() != 4 {
        return None;
    }
    Some([
        complex_scalar_from_tensor(&values[0])?,
        complex_scalar_from_tensor(&values[1])?,
        complex_scalar_from_tensor(&values[2])?,
        complex_scalar_from_tensor(&values[3])?,
    ])
}

fn airy_vec_bundle(
    result: Result<Vec<SpecialTensor>, fsci_special::SpecialError>,
) -> Option<[Vec<Complex64>; 4]> {
    let values = result.ok()?;
    if values.len() != 4 {
        return None;
    }
    Some([
        complex_vec_from_tensor(&values[0])?,
        complex_vec_from_tensor(&values[1])?,
        complex_vec_from_tensor(&values[2])?,
        complex_vec_from_tensor(&values[3])?,
    ])
}

fn check_wronskian(bundle: [Complex64; 4], z: Complex64) {
    let [ai_value, aip_value, bi_value, bip_value] = bundle;
    if ai_value.is_finite()
        && aip_value.is_finite()
        && bi_value.is_finite()
        && bip_value.is_finite()
    {
        let term1 = ai_value * bip_value;
        let term2 = aip_value * bi_value;
        let wronskian = term1 - term2;
        let expected = Complex64::from_real(1.0 / std::f64::consts::PI);
        // At large |z|, |term1| and |term2| reach ~10^9, so double-precision subtraction
        // of term1 - term2 experiences cancellation noise proportional to term magnitude * eps.
        // Scale tolerance by the cancellation magnitude.
        let cancellation_scale = term1.abs().max(term2.abs()).max(1.0);
        let tol = ABS_TOL + REL_TOL * cancellation_scale;
        assert!(
            (wronskian - expected).abs() <= tol,
            "airy wronskian mismatch for {z:?}: {wronskian:?} (tol={tol})"
        );
    }
}

fuzz_target!(|input: AiryInput| {
    let mode = mode_from_flag(input.hardened);
    let z = Complex64::new(input.re.bounded(), input.im.bounded());

    let scalar_bundle = airy_scalar_bundle(airy(&ComplexScalar(z), mode));
    if let Some(bundle) = scalar_bundle {
        check_wronskian(bundle, z);

        let ai_value = complex_scalar_from_tensor(
            &ai(&ComplexScalar(z), mode).unwrap_or(SpecialTensor::Empty),
        );
        let bi_value = complex_scalar_from_tensor(
            &bi(&ComplexScalar(z), mode).unwrap_or(SpecialTensor::Empty),
        );
        if let Some(ai_scalar) = ai_value
            && ai_scalar.is_finite()
        {
            assert!(
                approx_eq_complex(ai_scalar, bundle[0]),
                "ai convenience mismatch for {z:?}"
            );
        }
        if let Some(bi_scalar) = bi_value
            && bi_scalar.is_finite()
        {
            assert!(
                approx_eq_complex(bi_scalar, bundle[2]),
                "bi convenience mismatch for {z:?}"
            );
        }
    }

    if z.is_finite() && z.im != 0.0 {
        let conj_bundle = airy_scalar_bundle(airy(&ComplexScalar(z.conj()), mode));
        if let (Some(values), Some(conj_values)) = (scalar_bundle, conj_bundle) {
            for (index, (lhs, rhs)) in values.iter().zip(conj_values.iter()).enumerate() {
                if lhs.is_finite() && rhs.is_finite() {
                    assert!(
                        approx_eq_complex(*rhs, lhs.conj()),
                        "airy conjugation mismatch in component {index} for {z:?}"
                    );
                }
            }
        }
    }

    if input.vectorize {
        let vector_bundle = airy_vec_bundle(airy(&ComplexVec(vec![z, z.conj()]), mode));
        if let Some(bundle) = vector_bundle {
            for (index, component) in bundle.iter().enumerate() {
                assert_eq!(component.len(), 2, "airy component {index} length mismatch");
                if component[0].is_finite()
                    && component[1].is_finite()
                    && z.is_finite()
                    && z.im != 0.0
                {
                    assert!(
                        approx_eq_complex(component[1], component[0].conj()),
                        "airy vector conjugation mismatch in component {index} for {z:?}"
                    );
                }
            }
        }
    }

    if z.im == 0.0 {
        let real_bundle = airy_scalar_bundle(airy(&RealScalar(z.re), mode));
        if let (Some(complex_values), Some(real_values)) = (scalar_bundle, real_bundle) {
            for (index, (complex_value, real_value)) in
                complex_values.iter().zip(real_values.iter()).enumerate()
            {
                if complex_value.is_finite() && real_value.is_finite() {
                    assert!(
                        approx_eq_scalar(complex_value.re, real_value.re),
                        "airy real-axis reduction mismatch in component {index} for {z:?}"
                    );
                    assert!(
                        complex_value.im.abs() <= ABS_TOL + REL_TOL * complex_value.re.abs(),
                        "airy expected near-real output in component {index} for {z:?}"
                    );
                }
            }
        }
    }
});
