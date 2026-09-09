#![no_main]

use arbitrary::Arbitrary;
use fsci_runtime::RuntimeMode;
use fsci_special::{
    Complex64, SpecialResult,
    SpecialTensor::{ComplexScalar, ComplexVec, RealScalar, RealVec},
    hyp0f1, hyp1f1, hyp2f1,
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

    fn safe_parameter(self, base: f64) -> f64 {
        let bounded = self.bounded();
        base + bounded / (1.0 + bounded.abs())
    }

    fn unit_disk(self) -> f64 {
        let bounded = self.bounded();
        bounded / (4.0 * (1.0 + bounded.abs()))
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct HyperInput {
    a_re: EdgeF64,
    a_im: EdgeF64,
    b_re: EdgeF64,
    b_im: EdgeF64,
    c_re: EdgeF64,
    c_im: EdgeF64,
    z_re: EdgeF64,
    z_im: EdgeF64,
    hardened: bool,
    vectorize: bool,
    mismatch_lengths: bool,
}

#[derive(Clone, Copy)]
struct Hyp2f1Case {
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
    mismatch_lengths: bool,
    branch_z: Complex64,
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

fn check_hyp0f1(
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
    mismatch_lengths: bool,
) {
    let _ = hyp0f1(&ComplexScalar(b), &ComplexScalar(z), mode);

    let lhs = complex_from_result(hyp0f1(&ComplexScalar(b), &ComplexScalar(z), mode));
    let rhs = complex_from_result(hyp0f1(
        &ComplexScalar(b.conj()),
        &ComplexScalar(z.conj()),
        mode,
    ));
    if let (Some(left), Some(right)) = (lhs, rhs)
        && left.is_finite()
        && right.is_finite()
        && (b.im != 0.0 || z.im != 0.0)
    {
        assert!(
            approx_eq_complex(right, left.conj()),
            "hyp0f1 conjugation mismatch for b={b:?}, z={z:?}"
        );
    }

    if vectorize {
        let values = complex_vec_from_result(hyp0f1(
            &ComplexVec(vec![b, b.conj()]),
            &ComplexVec(vec![z, z.conj()]),
            mode,
        ));
        if let Some(vec_values) = values {
            assert_eq!(vec_values.len(), 2, "hyp0f1: vector output length mismatch");
            if vec_values[0].is_finite()
                && vec_values[1].is_finite()
                && (b.im != 0.0 || z.im != 0.0)
            {
                assert!(
                    approx_eq_complex(vec_values[1], vec_values[0].conj()),
                    "hyp0f1 vector conjugation mismatch for b={b:?}, z={z:?}"
                );
            }
        }
    }

    if mismatch_lengths {
        assert!(
            hyp0f1(&ComplexVec(vec![b, b.conj()]), &ComplexVec(vec![z]), mode).is_err(),
            "hyp0f1: mismatched vector lengths should fail closed"
        );
    }

    if b.im == 0.0 && z.im == 0.0 {
        let real_value = complex_from_result(hyp0f1(&RealScalar(b.re), &RealScalar(z.re), mode));
        let complex_value = complex_from_result(hyp0f1(&ComplexScalar(b), &ComplexScalar(z), mode));
        if let (Some(real_result), Some(complex_result)) = (real_value, complex_value)
            && real_result.is_finite()
            && complex_result.is_finite()
        {
            assert!(
                approx_eq_complex(complex_result, real_result),
                "hyp0f1 real-axis reduction mismatch for b={b:?}, z={z:?}"
            );
        }
    }

    let zero_result = complex_from_result(hyp0f1(
        &ComplexScalar(b),
        &ComplexScalar(Complex64::from_real(0.0)),
        mode,
    ));
    if let Some(zero_value) = zero_result
        && zero_value.is_finite()
    {
        assert!(
            approx_eq_complex(zero_value, Complex64::from_real(1.0)),
            "hyp0f1(b, 0) should equal 1 for b={b:?}"
        );
    }
}

fn check_hyp1f1(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
    vectorize: bool,
    mismatch_lengths: bool,
) {
    let _ = hyp1f1(
        &ComplexScalar(a),
        &ComplexScalar(b),
        &ComplexScalar(z),
        mode,
    );

    let lhs = complex_from_result(hyp1f1(
        &ComplexScalar(a),
        &ComplexScalar(b),
        &ComplexScalar(z),
        mode,
    ));
    let rhs = complex_from_result(hyp1f1(
        &ComplexScalar(a.conj()),
        &ComplexScalar(b.conj()),
        &ComplexScalar(z.conj()),
        mode,
    ));
    if let (Some(left), Some(right)) = (lhs, rhs)
        && left.is_finite()
        && right.is_finite()
        && (a.im != 0.0 || b.im != 0.0 || z.im != 0.0)
    {
        assert!(
            approx_eq_complex(right, left.conj()),
            "hyp1f1 conjugation mismatch for a={a:?}, b={b:?}, z={z:?}"
        );
    }

    if vectorize {
        let values = complex_vec_from_result(hyp1f1(
            &ComplexVec(vec![a, a.conj()]),
            &ComplexVec(vec![b, b.conj()]),
            &ComplexVec(vec![z, z.conj()]),
            mode,
        ));
        if let Some(vec_values) = values {
            assert_eq!(vec_values.len(), 2, "hyp1f1: vector output length mismatch");
            if vec_values[0].is_finite()
                && vec_values[1].is_finite()
                && (a.im != 0.0 || b.im != 0.0 || z.im != 0.0)
            {
                assert!(
                    approx_eq_complex(vec_values[1], vec_values[0].conj()),
                    "hyp1f1 vector conjugation mismatch for a={a:?}, b={b:?}, z={z:?}"
                );
            }
        }
    }

    if mismatch_lengths {
        assert!(
            hyp1f1(
                &ComplexVec(vec![a, a.conj()]),
                &ComplexVec(vec![b]),
                &ComplexScalar(z),
                mode
            )
            .is_err(),
            "hyp1f1: mismatched vector lengths should fail closed"
        );
    }

    if a.im == 0.0 && b.im == 0.0 && z.im == 0.0 {
        let real_value = complex_from_result(hyp1f1(
            &RealScalar(a.re),
            &RealScalar(b.re),
            &RealScalar(z.re),
            mode,
        ));
        let complex_value = complex_from_result(hyp1f1(
            &ComplexScalar(a),
            &ComplexScalar(b),
            &ComplexScalar(z),
            mode,
        ));
        if let (Some(real_result), Some(complex_result)) = (real_value, complex_value)
            && real_result.is_finite()
            && complex_result.is_finite()
        {
            assert!(
                approx_eq_complex(complex_result, real_result),
                "hyp1f1 real-axis reduction mismatch for a={a:?}, b={b:?}, z={z:?}"
            );
        }
    }

    let exp_identity = complex_from_result(hyp1f1(
        &ComplexScalar(a),
        &ComplexScalar(a),
        &ComplexScalar(z),
        mode,
    ));
    if let Some(value) = exp_identity
        && value.is_finite()
    {
        assert!(
            approx_eq_complex(value, z.exp()),
            "hyp1f1(a, a, z) should equal exp(z) for a={a:?}, z={z:?}"
        );
    }
}

fn check_hyp2f1(case: Hyp2f1Case) {
    let _ = hyp2f1(
        &ComplexScalar(case.a),
        &ComplexScalar(case.b),
        &ComplexScalar(case.c),
        &ComplexScalar(case.z),
        case.mode,
    );

    let lhs = complex_from_result(hyp2f1(
        &ComplexScalar(case.a),
        &ComplexScalar(case.b),
        &ComplexScalar(case.c),
        &ComplexScalar(case.z),
        case.mode,
    ));
    let rhs = complex_from_result(hyp2f1(
        &ComplexScalar(case.a.conj()),
        &ComplexScalar(case.b.conj()),
        &ComplexScalar(case.c.conj()),
        &ComplexScalar(case.z.conj()),
        case.mode,
    ));
    if let (Some(left), Some(right)) = (lhs, rhs)
        && left.is_finite()
        && right.is_finite()
        && (case.a.im != 0.0 || case.b.im != 0.0 || case.c.im != 0.0 || case.z.im != 0.0)
    {
        assert!(
            approx_eq_complex(right, left.conj()),
            "hyp2f1 conjugation mismatch for a={:?}, b={:?}, c={:?}, z={:?}",
            case.a,
            case.b,
            case.c,
            case.z
        );
    }

    if case.vectorize {
        let values = complex_vec_from_result(hyp2f1(
            &ComplexVec(vec![case.a, case.a.conj()]),
            &ComplexVec(vec![case.b, case.b.conj()]),
            &ComplexVec(vec![case.c, case.c.conj()]),
            &ComplexVec(vec![case.z, case.z.conj()]),
            case.mode,
        ));
        if let Some(vec_values) = values {
            assert_eq!(vec_values.len(), 2, "hyp2f1: vector output length mismatch");
            if vec_values[0].is_finite()
                && vec_values[1].is_finite()
                && (case.a.im != 0.0 || case.b.im != 0.0 || case.c.im != 0.0 || case.z.im != 0.0)
            {
                assert!(
                    approx_eq_complex(vec_values[1], vec_values[0].conj()),
                    "hyp2f1 vector conjugation mismatch for a={:?}, b={:?}, c={:?}, z={:?}",
                    case.a,
                    case.b,
                    case.c,
                    case.z
                );
            }
        }
    }

    if case.mismatch_lengths {
        assert!(
            hyp2f1(
                &ComplexVec(vec![case.a, case.a.conj()]),
                &ComplexVec(vec![case.b]),
                &ComplexScalar(case.c),
                &ComplexScalar(case.z),
                case.mode
            )
            .is_err(),
            "hyp2f1: mismatched vector lengths should fail closed"
        );
    }

    if case.a.im == 0.0 && case.b.im == 0.0 && case.c.im == 0.0 && case.z.im == 0.0 {
        let real_value = complex_from_result(hyp2f1(
            &RealScalar(case.a.re),
            &RealScalar(case.b.re),
            &RealScalar(case.c.re),
            &RealScalar(case.z.re),
            case.mode,
        ));
        let complex_value = complex_from_result(hyp2f1(
            &ComplexScalar(case.a),
            &ComplexScalar(case.b),
            &ComplexScalar(case.c),
            &ComplexScalar(case.z),
            case.mode,
        ));
        if let (Some(real_result), Some(complex_result)) = (real_value, complex_value)
            && real_result.is_finite()
            && complex_result.is_finite()
        {
            assert!(
                approx_eq_complex(complex_result, real_result),
                "hyp2f1 real-axis reduction mismatch for a={:?}, b={:?}, c={:?}, z={:?}",
                case.a,
                case.b,
                case.c,
                case.z
            );
        }
    }

    let linear_identity = complex_from_result(hyp2f1(
        &RealScalar(-1.0),
        &ComplexScalar(case.b),
        &ComplexScalar(case.b),
        &ComplexScalar(case.z),
        case.mode,
    ));
    if let Some(value) = linear_identity
        && value.is_finite()
    {
        assert!(
            approx_eq_complex(value, Complex64::from_real(1.0) - case.z),
            "hyp2f1(-1, b; b; z) should equal 1-z for b={:?}, z={:?}",
            case.b,
            case.z
        );
    }

    let branch_upper = case.branch_z;
    let branch_lower = case.branch_z.conj();
    let upper = hyp2f1(
        &RealScalar(1.0),
        &RealScalar(1.25),
        &RealScalar(1.75),
        &ComplexScalar(branch_upper),
        case.mode,
    );
    let lower = hyp2f1(
        &RealScalar(1.0),
        &RealScalar(1.25),
        &RealScalar(1.75),
        &ComplexScalar(branch_lower),
        case.mode,
    );
    if case.mode == RuntimeMode::Hardened {
        assert!(
            upper.is_err(),
            "hyp2f1 branch-cut upper lane should fail closed"
        );
        assert!(
            lower.is_err(),
            "hyp2f1 branch-cut lower lane should fail closed"
        );
    } else if let (Some(upper_value), Some(lower_value)) =
        (complex_from_result(upper), complex_from_result(lower))
        && upper_value.is_finite()
        && lower_value.is_finite()
    {
        assert!(
            approx_eq_complex(lower_value, upper_value.conj()),
            "hyp2f1 strict branch-cut lanes should preserve conjugation"
        );
    }
}

fuzz_target!(|input: HyperInput| {
    let mode = mode_from_flag(input.hardened);
    let a = Complex64::new(input.a_re.safe_parameter(0.75), input.a_im.bounded() / 8.0);
    let b = Complex64::new(input.b_re.safe_parameter(1.25), input.b_im.bounded() / 8.0);
    let c = Complex64::new(input.c_re.safe_parameter(1.75), input.c_im.bounded() / 8.0);
    let z = Complex64::new(input.z_re.unit_disk(), input.z_im.unit_disk());
    let branch_im = if input.z_im.raw().is_sign_negative() {
        -1.0e-9
    } else {
        1.0e-9
    };
    let branch_z = Complex64::new(1.0 + input.z_re.unit_disk().abs(), branch_im);

    check_hyp0f1(b, z, mode, input.vectorize, input.mismatch_lengths);
    check_hyp1f1(a, b, z, mode, input.vectorize, input.mismatch_lengths);
    check_hyp2f1(Hyp2f1Case {
        a,
        b,
        c,
        z,
        mode,
        vectorize: input.vectorize,
        mismatch_lengths: input.mismatch_lengths,
        branch_z,
    });
    let _ = fsci_special::take_special_traces();
});
