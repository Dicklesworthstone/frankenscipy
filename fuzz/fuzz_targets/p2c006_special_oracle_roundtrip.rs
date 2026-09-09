#![no_main]

use arbitrary::Arbitrary;
use fsci_conformance::{
    SpecialArgument, SpecialCase, SpecialCaseFunction, SpecialExpectedOutcome,
    SpecialPacketFixture, SpecialValueClass,
};
use fsci_runtime::RuntimeMode;
use libfuzzer_sys::fuzz_target;
use serde::{Deserialize, Serialize};

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
    Large,
    NegLarge,
}

impl EdgeF64 {
    fn raw(self) -> f64 {
        match self {
            Self::Finite(value) if value.is_finite() => value.clamp(-1.0e6, 1.0e6),
            Self::Finite(_) => 0.0,
            Self::Zero => 0.0,
            Self::NegZero => -0.0,
            Self::One => 1.0,
            Self::NegOne => -1.0,
            Self::Half => 0.5,
            Self::NegHalf => -0.5,
            Self::Tiny => f64::MIN_POSITIVE,
            Self::NegTiny => -f64::MIN_POSITIVE,
            Self::Large => 1.0e6,
            Self::NegLarge => -1.0e6,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ToleranceInput {
    None,
    Tight,
    Loose,
    Finite(f64),
}

impl ToleranceInput {
    fn into_option(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Tight => Some(1.0e-12),
            Self::Loose => Some(1.0e-4),
            Self::Finite(value) if value.is_finite() => Some(value.abs().clamp(1.0e-15, 1.0)),
            Self::Finite(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum FunctionChoice {
    Gamma,
    Gammaln,
    Digamma,
    Polygamma,
    Rgamma,
    Comb,
    Zeta,
    Erf,
    Erfc,
    Erfinv,
    Erfcinv,
    Exp1,
    Expi,
    Betainc,
    Beta,
    Betaln,
    Ellipe,
    Ellipkinc,
    EllipjSn,
    Hyp0f1,
    Hyp1f1,
    Jn,
    Iv,
    Kv,
    SphericalJn,
    WrightBessel,
    EvalChebyt,
    EvalHermite,
    EvalLegendre,
    EvalLaguerre,
    Lpmv,
    Expit,
    Logit,
    FresnelS,
    Dawsn,
    Gammainc,
    Lambertw,
    Ellipk,
    Hyp2f1,
    Jvp,
}

impl FunctionChoice {
    fn into_function(self) -> SpecialCaseFunction {
        match self {
            Self::Gamma => SpecialCaseFunction::Gamma,
            Self::Gammaln => SpecialCaseFunction::Gammaln,
            Self::Digamma => SpecialCaseFunction::Digamma,
            Self::Polygamma => SpecialCaseFunction::Polygamma,
            Self::Rgamma => SpecialCaseFunction::Rgamma,
            Self::Comb => SpecialCaseFunction::Comb,
            Self::Zeta => SpecialCaseFunction::Zeta,
            Self::Erf => SpecialCaseFunction::Erf,
            Self::Erfc => SpecialCaseFunction::Erfc,
            Self::Erfinv => SpecialCaseFunction::Erfinv,
            Self::Erfcinv => SpecialCaseFunction::Erfcinv,
            Self::Exp1 => SpecialCaseFunction::Exp1,
            Self::Expi => SpecialCaseFunction::Expi,
            Self::Betainc => SpecialCaseFunction::Betainc,
            Self::Beta => SpecialCaseFunction::Beta,
            Self::Betaln => SpecialCaseFunction::Betaln,
            Self::Ellipe => SpecialCaseFunction::Ellipe,
            Self::Ellipkinc => SpecialCaseFunction::Ellipkinc,
            Self::EllipjSn => SpecialCaseFunction::EllipjSn,
            Self::Hyp0f1 => SpecialCaseFunction::Hyp0f1,
            Self::Hyp1f1 => SpecialCaseFunction::Hyp1f1,
            Self::Jn => SpecialCaseFunction::Jn,
            Self::Iv => SpecialCaseFunction::Iv,
            Self::Kv => SpecialCaseFunction::Kv,
            Self::SphericalJn => SpecialCaseFunction::SphericalJn,
            Self::WrightBessel => SpecialCaseFunction::WrightBessel,
            Self::EvalChebyt => SpecialCaseFunction::EvalChebyt,
            Self::EvalHermite => SpecialCaseFunction::EvalHermite,
            Self::EvalLegendre => SpecialCaseFunction::EvalLegendre,
            Self::EvalLaguerre => SpecialCaseFunction::EvalLaguerre,
            Self::Lpmv => SpecialCaseFunction::Lpmv,
            Self::Expit => SpecialCaseFunction::Expit,
            Self::Logit => SpecialCaseFunction::Logit,
            Self::FresnelS => SpecialCaseFunction::FresnelS,
            Self::Dawsn => SpecialCaseFunction::Dawsn,
            Self::Gammainc => SpecialCaseFunction::Gammainc,
            Self::Lambertw => SpecialCaseFunction::Lambertw,
            Self::Ellipk => SpecialCaseFunction::Ellipk,
            Self::Hyp2f1 => SpecialCaseFunction::Hyp2f1,
            Self::Jvp => SpecialCaseFunction::Jvp,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionCoverageBucket {
    Gamma,
    Error,
    BetaAndDistribution,
    Elliptic,
    HyperAndBessel,
    Orthopoly,
    Convenience,
    Identity,
}

fn function_coverage_bucket(function: SpecialCaseFunction) -> FunctionCoverageBucket {
    match function {
        SpecialCaseFunction::Gamma
        | SpecialCaseFunction::Gammaln
        | SpecialCaseFunction::Multigammaln
        | SpecialCaseFunction::Digamma
        | SpecialCaseFunction::Polygamma
        | SpecialCaseFunction::Rgamma
        | SpecialCaseFunction::Gammainc
        | SpecialCaseFunction::Gammaincc
        | SpecialCaseFunction::Factorial
        | SpecialCaseFunction::Factorial2
        | SpecialCaseFunction::Comb
        | SpecialCaseFunction::Perm
        | SpecialCaseFunction::Poch
        | SpecialCaseFunction::Zeta
        | SpecialCaseFunction::Zetac
        | SpecialCaseFunction::HurwitzZeta => FunctionCoverageBucket::Gamma,
        SpecialCaseFunction::Erf
        | SpecialCaseFunction::Erfc
        | SpecialCaseFunction::Erfinv
        | SpecialCaseFunction::Erfcinv
        | SpecialCaseFunction::Erfcx
        | SpecialCaseFunction::Erfi
        | SpecialCaseFunction::OwensT
        | SpecialCaseFunction::Exp1
        | SpecialCaseFunction::Expi
        | SpecialCaseFunction::Expn => FunctionCoverageBucket::Error,
        SpecialCaseFunction::Beta
        | SpecialCaseFunction::Betaln
        | SpecialCaseFunction::Betainc
        | SpecialCaseFunction::Btdtr
        | SpecialCaseFunction::Btdtrc
        | SpecialCaseFunction::Btdtri
        | SpecialCaseFunction::Btdtria
        | SpecialCaseFunction::Btdtrib
        | SpecialCaseFunction::Fdtr
        | SpecialCaseFunction::Fdtrc
        | SpecialCaseFunction::Fdtri
        | SpecialCaseFunction::Fdtridfd
        | SpecialCaseFunction::Stdtr
        | SpecialCaseFunction::Stdtrc
        | SpecialCaseFunction::Stdtridf
        | SpecialCaseFunction::Stdtrit
        | SpecialCaseFunction::Bdtr
        | SpecialCaseFunction::Bdtrc
        | SpecialCaseFunction::Bdtri
        | SpecialCaseFunction::Nbdtr
        | SpecialCaseFunction::Nbdtrc
        | SpecialCaseFunction::Nbdtri
        | SpecialCaseFunction::Pdtr
        | SpecialCaseFunction::Pdtrc
        | SpecialCaseFunction::Pdtri
        | SpecialCaseFunction::Pdtrik
        | SpecialCaseFunction::Chdtr
        | SpecialCaseFunction::Chdtrc
        | SpecialCaseFunction::Chdtri
        | SpecialCaseFunction::Chdtriv
        | SpecialCaseFunction::Gdtr
        | SpecialCaseFunction::Gdtrc
        | SpecialCaseFunction::Gdtria
        | SpecialCaseFunction::Gdtrib
        | SpecialCaseFunction::Gdtrix => FunctionCoverageBucket::BetaAndDistribution,
        SpecialCaseFunction::Ellipk
        | SpecialCaseFunction::Ellipkm1
        | SpecialCaseFunction::Ellipe
        | SpecialCaseFunction::Ellipkinc
        | SpecialCaseFunction::Ellipeinc
        | SpecialCaseFunction::EllipjSn
        | SpecialCaseFunction::EllipjCn
        | SpecialCaseFunction::EllipjDn
        | SpecialCaseFunction::EllipjPh => FunctionCoverageBucket::Elliptic,
        SpecialCaseFunction::Lambertw
        | SpecialCaseFunction::Hyp0f1
        | SpecialCaseFunction::Hyp1f1
        | SpecialCaseFunction::Hyp2f1
        | SpecialCaseFunction::J0
        | SpecialCaseFunction::J1
        | SpecialCaseFunction::Jn
        | SpecialCaseFunction::Y0
        | SpecialCaseFunction::Y1
        | SpecialCaseFunction::Yn
        | SpecialCaseFunction::Iv
        | SpecialCaseFunction::Kv
        | SpecialCaseFunction::WrightBessel
        | SpecialCaseFunction::Jvp
        | SpecialCaseFunction::Yvp
        | SpecialCaseFunction::Ivp
        | SpecialCaseFunction::Kvp
        | SpecialCaseFunction::SphericalJn
        | SpecialCaseFunction::SphericalYn
        | SpecialCaseFunction::SphericalIn
        | SpecialCaseFunction::SphericalKn
        | SpecialCaseFunction::Lpmv
        | SpecialCaseFunction::SphHarmY
        | SpecialCaseFunction::Struve
        | SpecialCaseFunction::Modstruve
        | SpecialCaseFunction::Ber
        | SpecialCaseFunction::Bei
        | SpecialCaseFunction::Ker
        | SpecialCaseFunction::Kei => FunctionCoverageBucket::HyperAndBessel,
        SpecialCaseFunction::EvalLegendre
        | SpecialCaseFunction::EvalChebyt
        | SpecialCaseFunction::EvalChebyu
        | SpecialCaseFunction::EvalHermite
        | SpecialCaseFunction::EvalHermitenorm
        | SpecialCaseFunction::EvalLaguerre
        | SpecialCaseFunction::EvalGenlaguerre
        | SpecialCaseFunction::EvalJacobi
        | SpecialCaseFunction::EvalGegenbauer
        | SpecialCaseFunction::EvalShLegendre
        | SpecialCaseFunction::EvalShChebyt
        | SpecialCaseFunction::EvalShChebyu
        | SpecialCaseFunction::RootsChebytNode
        | SpecialCaseFunction::RootsChebytWeight
        | SpecialCaseFunction::RootsChebyuNode
        | SpecialCaseFunction::RootsChebyuWeight
        | SpecialCaseFunction::RootsHermiteNode
        | SpecialCaseFunction::RootsHermiteWeight
        | SpecialCaseFunction::RootsHermitenormNode
        | SpecialCaseFunction::RootsHermitenormWeight
        | SpecialCaseFunction::RootsLaguerreNode
        | SpecialCaseFunction::RootsLaguerreWeight
        | SpecialCaseFunction::RootsGenlaguerreNode
        | SpecialCaseFunction::RootsGenlaguerreWeight
        | SpecialCaseFunction::RootsJacobiNode
        | SpecialCaseFunction::RootsJacobiWeight
        | SpecialCaseFunction::RootsGegenbauerNode
        | SpecialCaseFunction::RootsGegenbauerWeight
        | SpecialCaseFunction::RootsLegendreNode
        | SpecialCaseFunction::RootsLegendreWeight => FunctionCoverageBucket::Orthopoly,
        SpecialCaseFunction::Sinc
        | SpecialCaseFunction::Xlogy
        | SpecialCaseFunction::Xlog1py
        | SpecialCaseFunction::Xlogx
        | SpecialCaseFunction::Kolmogorov
        | SpecialCaseFunction::Kolmogi
        | SpecialCaseFunction::Entr
        | SpecialCaseFunction::RelEntr
        | SpecialCaseFunction::KlDiv
        | SpecialCaseFunction::Ndtr
        | SpecialCaseFunction::Ndtri
        | SpecialCaseFunction::Nrdtrimn
        | SpecialCaseFunction::Nrdtrisd
        | SpecialCaseFunction::LogNdtr
        | SpecialCaseFunction::Logsumexp
        | SpecialCaseFunction::Log1p
        | SpecialCaseFunction::Expm1
        | SpecialCaseFunction::Logaddexp
        | SpecialCaseFunction::Logaddexp2
        | SpecialCaseFunction::Softplus
        | SpecialCaseFunction::Huber
        | SpecialCaseFunction::PseudoHuber
        | SpecialCaseFunction::Cosdg
        | SpecialCaseFunction::Sindg
        | SpecialCaseFunction::Tandg
        | SpecialCaseFunction::Cotdg
        | SpecialCaseFunction::Radian
        | SpecialCaseFunction::Cbrt
        | SpecialCaseFunction::Exp2
        | SpecialCaseFunction::Exp10
        | SpecialCaseFunction::Boxcox
        | SpecialCaseFunction::InvBoxcox
        | SpecialCaseFunction::Boxcox1p
        | SpecialCaseFunction::InvBoxcox1p
        | SpecialCaseFunction::FresnelS
        | SpecialCaseFunction::FresnelC
        | SpecialCaseFunction::Dawsn
        | SpecialCaseFunction::SiciSi
        | SpecialCaseFunction::SiciCi
        | SpecialCaseFunction::ShichiShi
        | SpecialCaseFunction::ShichiChi
        | SpecialCaseFunction::Spence
        | SpecialCaseFunction::Expit
        | SpecialCaseFunction::Logit
        | SpecialCaseFunction::Exprel => FunctionCoverageBucket::Convenience,
        SpecialCaseFunction::RelErfErfcIdentity
        | SpecialCaseFunction::RelGammaRecurrence
        | SpecialCaseFunction::RelBetaSymmetry
        | SpecialCaseFunction::RelGammaincComplement
        | SpecialCaseFunction::RelJnRecurrence
        | SpecialCaseFunction::RelErfinvComposition => FunctionCoverageBucket::Identity,
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum CategoryChoice {
    Differential,
    Regression,
    BranchCut,
    Domain,
    TensorBroadcast,
}

impl CategoryChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Differential => "differential",
            Self::Regression => "regression",
            Self::BranchCut => "branch_cut",
            Self::Domain => "domain",
            Self::TensorBroadcast => "tensor_broadcast",
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ValueClassChoice {
    Finite,
    Nan,
    PosInf,
    NegInf,
}

impl ValueClassChoice {
    fn into_class(self) -> SpecialValueClass {
        match self {
            Self::Finite => SpecialValueClass::Finite,
            Self::Nan => SpecialValueClass::Nan,
            Self::PosInf => SpecialValueClass::PosInf,
            Self::NegInf => SpecialValueClass::NegInf,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ErrorChoice {
    Domain,
    Overflow,
    NoConvergence,
    NotRepresentable,
}

impl ErrorChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Domain => "domain",
            Self::Overflow => "overflow",
            Self::NoConvergence => "no_convergence",
            Self::NotRepresentable => "not_representable",
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ComplexAtomInput {
    Finite(EdgeF64),
    Nan,
    PosInf,
    NegInf,
}

impl ComplexAtomInput {
    fn into_component(self) -> FutureComplexComponent {
        match self {
            Self::Finite(value) => FutureComplexComponent::Finite { value: value.raw() },
            Self::Nan => FutureComplexComponent::Nan,
            Self::PosInf => FutureComplexComponent::PosInf,
            Self::NegInf => FutureComplexComponent::NegInf,
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct ComplexPairInput {
    re: ComplexAtomInput,
    im: ComplexAtomInput,
}

impl ComplexPairInput {
    fn into_pair(self) -> FutureComplexPair {
        FutureComplexPair {
            re: self.re.into_component(),
            im: self.im.into_component(),
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct OutcomeMetadataInput {
    atol: ToleranceInput,
    rtol: ToleranceInput,
    include_contract_ref: bool,
}

impl OutcomeMetadataInput {
    fn atol(self) -> Option<f64> {
        self.atol.into_option()
    }

    fn rtol(self) -> Option<f64> {
        self.rtol.into_option()
    }

    fn contract_ref(self, case_id: &str) -> Option<String> {
        self.include_contract_ref
            .then(|| format!("p2c006/special-oracle/{case_id}"))
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum ExpectedInput {
    Scalar {
        value: EdgeF64,
        meta: OutcomeMetadataInput,
    },
    Vector {
        len: u8,
        values: [EdgeF64; 3],
        meta: OutcomeMetadataInput,
    },
    ComplexScalar {
        value: ComplexPairInput,
        meta: OutcomeMetadataInput,
    },
    ComplexVector {
        len: u8,
        values: [ComplexPairInput; 2],
        meta: OutcomeMetadataInput,
    },
    Class {
        class: ValueClassChoice,
    },
    ErrorKind {
        error: ErrorChoice,
    },
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct CaseInput {
    case_tag: u16,
    hardened: bool,
    function: FunctionChoice,
    category: CategoryChoice,
    arg_len: u8,
    args: [EdgeF64; 3],
    expected: ExpectedInput,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
enum FamilyChoice {
    Special,
    Oracle,
    Differential,
}

impl FamilyChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Special => "special",
            Self::Oracle => "special_oracle",
            Self::Differential => "special_differential",
        }
    }
}

#[derive(Clone, Copy, Debug, Arbitrary)]
struct PacketInput {
    packet_tag: u16,
    family: FamilyChoice,
    case_len: u8,
    cases: [CaseInput; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FuturePacketFixture {
    packet_id: String,
    family: String,
    cases: Vec<FutureCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FutureCase {
    case_id: String,
    category: String,
    mode: RuntimeMode,
    function: SpecialCaseFunction,
    #[serde(default)]
    args: Vec<f64>,
    expected: FutureExpectedOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FutureExpectedOutcome {
    Scalar {
        value: f64,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Vector {
        values: Vec<f64>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ComplexScalar {
        value: FutureComplexPair,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    ComplexVector {
        values: Vec<FutureComplexPair>,
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        rtol: Option<f64>,
        #[serde(default)]
        contract_ref: Option<String>,
    },
    Class {
        class: SpecialValueClass,
    },
    ErrorKind {
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FutureComplexPair {
    re: FutureComplexComponent,
    im: FutureComplexComponent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FutureComplexComponent {
    Finite { value: f64 },
    Nan,
    PosInf,
    NegInf,
}

fn build_future_packet(input: PacketInput) -> FuturePacketFixture {
    let case_count = 1 + (input.case_len as usize % input.cases.len());
    let cases = input.cases[..case_count]
        .iter()
        .map(|case| build_future_case(input.packet_tag, *case))
        .collect();

    FuturePacketFixture {
        packet_id: format!("p2c006-oracle-{}", input.packet_tag),
        family: input.family.as_str().to_string(),
        cases,
    }
}

fn build_future_case(packet_tag: u16, input: CaseInput) -> FutureCase {
    let case_id = format!("case-{packet_tag}-{}", input.case_tag);
    let args = input.args[..(input.arg_len as usize % (input.args.len() + 1))]
        .iter()
        .map(|value| value.raw())
        .collect();
    let function = input.function.into_function();
    let _coverage_bucket = function_coverage_bucket(function);

    FutureCase {
        case_id: case_id.clone(),
        category: input.category.as_str().to_string(),
        mode: if input.hardened {
            RuntimeMode::Hardened
        } else {
            RuntimeMode::Strict
        },
        function,
        args,
        expected: build_future_expected(input.expected, &case_id),
    }
}

fn build_future_expected(input: ExpectedInput, case_id: &str) -> FutureExpectedOutcome {
    match input {
        ExpectedInput::Scalar { value, meta } => FutureExpectedOutcome::Scalar {
            value: value.raw(),
            atol: meta.atol(),
            rtol: meta.rtol(),
            contract_ref: meta.contract_ref(case_id),
        },
        ExpectedInput::Vector { len, values, meta } => {
            let count = 1 + (len as usize % values.len());
            FutureExpectedOutcome::Vector {
                values: values[..count].iter().map(|value| value.raw()).collect(),
                atol: meta.atol(),
                rtol: meta.rtol(),
                contract_ref: meta.contract_ref(case_id),
            }
        }
        ExpectedInput::ComplexScalar { value, meta } => FutureExpectedOutcome::ComplexScalar {
            value: value.into_pair(),
            atol: meta.atol(),
            rtol: meta.rtol(),
            contract_ref: meta.contract_ref(case_id),
        },
        ExpectedInput::ComplexVector { len, values, meta } => {
            let count = 1 + (len as usize % values.len());
            FutureExpectedOutcome::ComplexVector {
                values: values[..count]
                    .iter()
                    .map(|value| value.into_pair())
                    .collect(),
                atol: meta.atol(),
                rtol: meta.rtol(),
                contract_ref: meta.contract_ref(case_id),
            }
        }
        ExpectedInput::Class { class } => FutureExpectedOutcome::Class {
            class: class.into_class(),
        },
        ExpectedInput::ErrorKind { error } => FutureExpectedOutcome::ErrorKind {
            error: error.as_str().to_string(),
        },
    }
}

fn same_f64_bits(lhs: f64, rhs: f64) -> bool {
    lhs.to_bits() == rhs.to_bits()
}

fn assert_same_optional_bits(lhs: Option<f64>, rhs: Option<f64>, context: &str) {
    match (lhs, rhs) {
        (Some(left), Some(right)) => {
            assert!(
                same_f64_bits(left, right),
                "{context}: mismatched float bits {left:?} vs {right:?}"
            );
        }
        (None, None) => {}
        _ => panic!("{context}: mismatched optional float presence"),
    }
}

fn assert_same_future_component(
    lhs: &FutureComplexComponent,
    rhs: &FutureComplexComponent,
    context: &str,
) {
    match (lhs, rhs) {
        (
            FutureComplexComponent::Finite { value: left },
            FutureComplexComponent::Finite { value: right },
        ) => {
            assert!(
                same_f64_bits(*left, *right),
                "{context}: mismatched complex component bits"
            );
        }
        (FutureComplexComponent::Nan, FutureComplexComponent::Nan)
        | (FutureComplexComponent::PosInf, FutureComplexComponent::PosInf)
        | (FutureComplexComponent::NegInf, FutureComplexComponent::NegInf) => {}
        _ => panic!("{context}: mismatched complex component variant"),
    }
}

fn assert_same_future_expected(
    lhs: &FutureExpectedOutcome,
    rhs: &FutureExpectedOutcome,
    context: &str,
) {
    match (lhs, rhs) {
        (
            FutureExpectedOutcome::Scalar {
                value: left_value,
                atol: left_atol,
                rtol: left_rtol,
                contract_ref: left_contract,
            },
            FutureExpectedOutcome::Scalar {
                value: right_value,
                atol: right_atol,
                rtol: right_rtol,
                contract_ref: right_contract,
            },
        ) => {
            assert!(
                same_f64_bits(*left_value, *right_value),
                "{context}: mismatched scalar bits"
            );
            assert_same_optional_bits(*left_atol, *right_atol, context);
            assert_same_optional_bits(*left_rtol, *right_rtol, context);
            assert_eq!(
                left_contract, right_contract,
                "{context}: contract ref mismatch"
            );
        }
        (
            FutureExpectedOutcome::Vector {
                values: left_values,
                atol: left_atol,
                rtol: left_rtol,
                contract_ref: left_contract,
            },
            FutureExpectedOutcome::Vector {
                values: right_values,
                atol: right_atol,
                rtol: right_rtol,
                contract_ref: right_contract,
            },
        ) => {
            assert_eq!(
                left_values.len(),
                right_values.len(),
                "{context}: vector len mismatch"
            );
            for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
                assert!(
                    same_f64_bits(*left, *right),
                    "{context}: vector scalar bits mismatch at {index}"
                );
            }
            assert_same_optional_bits(*left_atol, *right_atol, context);
            assert_same_optional_bits(*left_rtol, *right_rtol, context);
            assert_eq!(
                left_contract, right_contract,
                "{context}: contract ref mismatch"
            );
        }
        (
            FutureExpectedOutcome::ComplexScalar {
                value: left_value,
                atol: left_atol,
                rtol: left_rtol,
                contract_ref: left_contract,
            },
            FutureExpectedOutcome::ComplexScalar {
                value: right_value,
                atol: right_atol,
                rtol: right_rtol,
                contract_ref: right_contract,
            },
        ) => {
            assert_same_future_component(&left_value.re, &right_value.re, context);
            assert_same_future_component(&left_value.im, &right_value.im, context);
            assert_same_optional_bits(*left_atol, *right_atol, context);
            assert_same_optional_bits(*left_rtol, *right_rtol, context);
            assert_eq!(
                left_contract, right_contract,
                "{context}: contract ref mismatch"
            );
        }
        (
            FutureExpectedOutcome::ComplexVector {
                values: left_values,
                atol: left_atol,
                rtol: left_rtol,
                contract_ref: left_contract,
            },
            FutureExpectedOutcome::ComplexVector {
                values: right_values,
                atol: right_atol,
                rtol: right_rtol,
                contract_ref: right_contract,
            },
        ) => {
            assert_eq!(
                left_values.len(),
                right_values.len(),
                "{context}: complex vector len mismatch"
            );
            for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
                assert_same_future_component(&left.re, &right.re, &format!("{context}/re/{index}"));
                assert_same_future_component(&left.im, &right.im, &format!("{context}/im/{index}"));
            }
            assert_same_optional_bits(*left_atol, *right_atol, context);
            assert_same_optional_bits(*left_rtol, *right_rtol, context);
            assert_eq!(
                left_contract, right_contract,
                "{context}: contract ref mismatch"
            );
        }
        (
            FutureExpectedOutcome::Class { class: left_class },
            FutureExpectedOutcome::Class { class: right_class },
        ) => {
            assert_eq!(left_class, right_class, "{context}: class mismatch");
        }
        (
            FutureExpectedOutcome::ErrorKind { error: left_error },
            FutureExpectedOutcome::ErrorKind { error: right_error },
        ) => {
            assert_eq!(left_error, right_error, "{context}: error mismatch");
        }
        _ => panic!("{context}: mismatched expected outcome variant"),
    }
}

fn assert_same_future_packet(lhs: &FuturePacketFixture, rhs: &FuturePacketFixture) {
    assert_eq!(lhs.packet_id, rhs.packet_id, "future packet id mismatch");
    assert_eq!(lhs.family, rhs.family, "future family mismatch");
    assert_eq!(lhs.cases.len(), rhs.cases.len(), "future case len mismatch");

    for (index, (left, right)) in lhs.cases.iter().zip(rhs.cases.iter()).enumerate() {
        assert_eq!(left.case_id, right.case_id, "future case id mismatch");
        assert_eq!(left.category, right.category, "future category mismatch");
        assert_eq!(left.mode, right.mode, "future mode mismatch");
        assert_eq!(left.function, right.function, "future function mismatch");
        assert_eq!(left.args.len(), right.args.len(), "future arg len mismatch");
        for (arg_index, (left_arg, right_arg)) in
            left.args.iter().zip(right.args.iter()).enumerate()
        {
            assert!(
                same_f64_bits(*left_arg, *right_arg),
                "future arg bits mismatch at {index}/{arg_index}"
            );
        }
        assert_same_future_expected(
            &left.expected,
            &right.expected,
            &format!("future-expected-{index}"),
        );
    }
}

fn to_current_packet(packet: &FuturePacketFixture) -> Option<SpecialPacketFixture> {
    let cases = packet
        .cases
        .iter()
        .map(|case| {
            Some(SpecialCase {
                case_id: case.case_id.clone(),
                category: case.category.clone(),
                mode: case.mode,
                function: case.function,
                args: case
                    .args
                    .iter()
                    .copied()
                    .map(SpecialArgument::RealScalar)
                    .collect(),
                expected: match &case.expected {
                    FutureExpectedOutcome::Scalar {
                        value,
                        atol,
                        rtol,
                        contract_ref,
                    } => SpecialExpectedOutcome::Scalar {
                        value: *value,
                        atol: *atol,
                        rtol: *rtol,
                        contract_ref: contract_ref.clone(),
                    },
                    FutureExpectedOutcome::Vector {
                        values,
                        atol,
                        rtol,
                        contract_ref,
                    } => SpecialExpectedOutcome::Vector {
                        values: values.clone(),
                        atol: *atol,
                        rtol: *rtol,
                        contract_ref: contract_ref.clone(),
                    },
                    FutureExpectedOutcome::Class { class } => {
                        SpecialExpectedOutcome::Class { class: *class }
                    }
                    FutureExpectedOutcome::ErrorKind { error } => {
                        SpecialExpectedOutcome::ErrorKind {
                            error: error.clone(),
                        }
                    }
                    FutureExpectedOutcome::ComplexScalar { .. }
                    | FutureExpectedOutcome::ComplexVector { .. } => return None,
                },
            })
        })
        .collect::<Option<Vec<_>>>()?;

    Some(SpecialPacketFixture {
        packet_id: packet.packet_id.clone(),
        family: packet.family.clone(),
        cases,
    })
}

fn assert_same_current_packet(lhs: &SpecialPacketFixture, rhs: &SpecialPacketFixture) {
    assert_eq!(lhs.packet_id, rhs.packet_id, "current packet id mismatch");
    assert_eq!(lhs.family, rhs.family, "current family mismatch");
    assert_eq!(
        lhs.cases.len(),
        rhs.cases.len(),
        "current case len mismatch"
    );

    for (index, (left, right)) in lhs.cases.iter().zip(rhs.cases.iter()).enumerate() {
        assert_eq!(left.case_id, right.case_id, "current case id mismatch");
        assert_eq!(left.category, right.category, "current category mismatch");
        assert_eq!(left.mode, right.mode, "current mode mismatch");
        assert_eq!(left.function, right.function, "current function mismatch");
        assert_eq!(
            left.args.len(),
            right.args.len(),
            "current arg len mismatch"
        );
        for (arg_index, (left_arg, right_arg)) in
            left.args.iter().zip(right.args.iter()).enumerate()
        {
            match (left_arg, right_arg) {
                (SpecialArgument::RealScalar(left), SpecialArgument::RealScalar(right)) => {
                    assert!(
                        same_f64_bits(*left, *right),
                        "current arg bits mismatch at {index}/{arg_index}"
                    );
                }
                _ => panic!("current arg variant mismatch at {index}/{arg_index}"),
            }
        }

        match (&left.expected, &right.expected) {
            (
                SpecialExpectedOutcome::Scalar {
                    value: left_value,
                    atol: left_atol,
                    rtol: left_rtol,
                    contract_ref: left_contract,
                },
                SpecialExpectedOutcome::Scalar {
                    value: right_value,
                    atol: right_atol,
                    rtol: right_rtol,
                    contract_ref: right_contract,
                },
            ) => {
                assert!(
                    same_f64_bits(*left_value, *right_value),
                    "current scalar bits mismatch at case {index}"
                );
                assert_same_optional_bits(*left_atol, *right_atol, "current scalar atol");
                assert_same_optional_bits(*left_rtol, *right_rtol, "current scalar rtol");
                assert_eq!(
                    left_contract, right_contract,
                    "current scalar contract ref mismatch"
                );
            }
            (
                SpecialExpectedOutcome::Vector {
                    values: left_values,
                    atol: left_atol,
                    rtol: left_rtol,
                    contract_ref: left_contract,
                },
                SpecialExpectedOutcome::Vector {
                    values: right_values,
                    atol: right_atol,
                    rtol: right_rtol,
                    contract_ref: right_contract,
                },
            ) => {
                assert_eq!(
                    left_values.len(),
                    right_values.len(),
                    "current vector len mismatch at case {index}"
                );
                for (val_index, (left_val, right_val)) in
                    left_values.iter().zip(right_values.iter()).enumerate()
                {
                    assert!(
                        same_f64_bits(*left_val, *right_val),
                        "current vector bits mismatch at case {index}/{val_index}"
                    );
                }
                assert_same_optional_bits(*left_atol, *right_atol, "current vector atol");
                assert_same_optional_bits(*left_rtol, *right_rtol, "current vector rtol");
                assert_eq!(
                    left_contract, right_contract,
                    "current vector contract ref mismatch"
                );
            }
            (
                SpecialExpectedOutcome::Class { class: left_class },
                SpecialExpectedOutcome::Class { class: right_class },
            ) => {
                assert_eq!(left_class, right_class, "current class mismatch");
            }
            (
                SpecialExpectedOutcome::ErrorKind { error: left_error },
                SpecialExpectedOutcome::ErrorKind { error: right_error },
            ) => {
                assert_eq!(left_error, right_error, "current error mismatch");
            }
            _ => panic!("current expected outcome variant mismatch at case {index}"),
        }
    }
}

fuzz_target!(|input: PacketInput| {
    let future_packet = build_future_packet(input);
    let future_json =
        serde_json::to_string(&future_packet).expect("future packet should serialize");
    let roundtrip_future: FuturePacketFixture =
        serde_json::from_str(&future_json).expect("future packet should deserialize");
    assert_same_future_packet(&future_packet, &roundtrip_future);

    if let Some(current_packet) = to_current_packet(&future_packet) {
        let current_json =
            serde_json::to_string(&current_packet).expect("current packet should serialize");
        let roundtrip_current: SpecialPacketFixture =
            serde_json::from_str(&current_json).expect("current packet should deserialize");
        assert_same_current_packet(&current_packet, &roundtrip_current);

        let current_from_future: SpecialPacketFixture = serde_json::from_str(&future_json)
            .expect("current schema should accept current-compatible future payloads");
        assert_same_current_packet(&current_packet, &current_from_future);
    } else {
        assert!(
            serde_json::from_str::<SpecialPacketFixture>(&future_json).is_err(),
            "current schema unexpectedly accepted future-only complex payload"
        );
    }
});
