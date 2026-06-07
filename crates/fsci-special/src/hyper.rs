#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor,
};

pub const HYPER_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "hyp0f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 50 and b is stable away from nonpositive-integer poles",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "|z| >= 50 and b is stable away from nonpositive-integer poles",
            },
        ],
        notes: "CASP records the same magnitude split used by the scalar evaluator and guards near-pole lower parameters.",
    },
    DispatchPlan {
        function: "hyp1f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| <= 2 and moderate parameters",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "parameter shifting to stable region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z| or large parameters",
            },
        ],
        notes: "Fallback routing should preserve SciPy branch-selection semantics for strict mode.",
    },
    DispatchPlan {
        function: "hyperu",
        steps: &[
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "a > 0 and x > 0 integral representation over the Tricomi kernel",
            },
            DispatchStep {
                regime: KernelRegime::Series,
                when: "a is a nonpositive integer and U reduces to a Laguerre polynomial",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "noninteger b fallback through the M(a,b,x) connection formula",
            },
        ],
        notes: "Strict mode preserves SciPy's real-domain behavior: negative x and unsupported complex inputs fail closed to NaN/error surfaces.",
    },
    DispatchPlan {
        function: "hyp2f1",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|z| < 0.9 and c not near nonpositive integers",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "boundary neighborhoods near z=1",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "contiguous relation stabilization",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large-parameter asymptotic domains",
            },
        ],
        notes: "z=1 convergence edge cases and c-pole exclusions are explicit hardened guards.",
    },
];

/// Hypergeometric routine covered by the special-function CASP selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypergeometricFunction {
    /// Confluent hypergeometric limit 0F1(; b; z).
    Hyp0f1,
    /// Confluent hypergeometric 1F1(a; b; z).
    Hyp1f1,
    /// Gauss hypergeometric 2F1(a, b; c; z).
    Hyp2f1,
}

/// Evaluation branch selected for a hypergeometric problem instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypergeometricBranch {
    /// Power series in the original argument.
    DirectSeries,
    /// Kummer transformation for large negative real 1F1 arguments.
    KummerTransform,
    /// Finite polynomial when a numerator parameter is a nonpositive integer.
    TerminatingPolynomial,
    /// Gauss summation at z = 1 when c - a - b > 0.
    GaussSummation,
    /// Pfaff transform for real 2F1 arguments outside the direct-series disk.
    PfaffTransform,
    /// Closed-form reduction when a numerator parameter equals c.
    LinearFractionalIdentity,
    /// Real-axis branch cut for generic nonterminating 2F1 z > 1.
    RealBranchCutInfinity,
    /// Asymptotic expansion selected outside the stable direct-series region.
    AsymptoticExpansion,
    /// Divergent z = 1 boundary.
    DivergentAtUnitArgument,
    /// Lower parameter is too close to a pole for the requested precision.
    ParameterGuard,
    /// No stable analytic-continuation branch is implemented.
    UnsupportedAnalyticContinuation,
}

/// Scalar hypergeometric CASP input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperCaspProblem {
    /// Function family to select for.
    pub function: HypergeometricFunction,
    /// First numerator parameter.
    pub a: f64,
    /// Second parameter: denominator for 1F1, numerator for 2F1.
    pub b: f64,
    /// Optional denominator parameter for 2F1.
    pub c: Option<f64>,
    /// Real argument.
    pub z: f64,
    /// Absolute value of the real argument.
    pub z_abs: f64,
    /// Requested absolute branch-selection precision.
    pub precision_target: f64,
    /// Distance from the lower parameter to the nearest nonpositive-integer pole.
    pub parameter_stability_margin: f64,
}

impl HyperCaspProblem {
    /// Build a 0F1 CASP problem.
    pub fn hyp0f1(b: f64, z: f64, precision_target: f64) -> Self {
        Self {
            function: HypergeometricFunction::Hyp0f1,
            a: 0.0,
            b,
            c: None,
            z,
            z_abs: z.abs(),
            precision_target,
            parameter_stability_margin: lower_parameter_stability_margin(b),
        }
    }

    /// Build a 1F1 CASP problem.
    pub fn hyp1f1(a: f64, b: f64, z: f64, precision_target: f64) -> Self {
        Self {
            function: HypergeometricFunction::Hyp1f1,
            a,
            b,
            c: None,
            z,
            z_abs: z.abs(),
            precision_target,
            parameter_stability_margin: lower_parameter_stability_margin(b),
        }
    }

    /// Build a 2F1 CASP problem.
    pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64, precision_target: f64) -> Self {
        Self {
            function: HypergeometricFunction::Hyp2f1,
            a,
            b,
            c: Some(c),
            z,
            z_abs: z.abs(),
            precision_target,
            parameter_stability_margin: lower_parameter_stability_margin(c),
        }
    }
}

/// Branch decision plus the convergence and fallback metadata used by CASP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperCaspDecision {
    /// Selected evaluation branch.
    pub branch: HypergeometricBranch,
    /// Requested precision copied from the problem.
    pub precision_target: f64,
    /// Maximum terms expected for the branch's convergence check.
    pub max_terms: usize,
    /// Lower-parameter pole margin copied from the problem.
    pub parameter_stability_margin: f64,
    /// Ordered fallback chain for the selected branch.
    pub fallback_chain: &'static [HypergeometricBranch],
    /// Static explanation for audit logs and tests.
    pub reason: &'static str,
}

const HYP0F1_DIRECT_CHAIN: &[HypergeometricBranch] = &[HypergeometricBranch::DirectSeries];
const HYP0F1_ASYMPTOTIC_CHAIN: &[HypergeometricBranch] = &[
    HypergeometricBranch::AsymptoticExpansion,
    HypergeometricBranch::DirectSeries,
    HypergeometricBranch::UnsupportedAnalyticContinuation,
];
const HYP1F1_DIRECT_CHAIN: &[HypergeometricBranch] = &[HypergeometricBranch::DirectSeries];
const HYP1F1_KUMMER_CHAIN: &[HypergeometricBranch] = &[
    HypergeometricBranch::KummerTransform,
    HypergeometricBranch::DirectSeries,
    HypergeometricBranch::UnsupportedAnalyticContinuation,
];
const HYP1F1_ASYMPTOTIC_CHAIN: &[HypergeometricBranch] = &[
    HypergeometricBranch::AsymptoticExpansion,
    HypergeometricBranch::UnsupportedAnalyticContinuation,
];
/// Above this real argument the 1F1 direct series (500-term cap, peak term near
/// n ≈ z) can no longer converge, so large positive z switches to the DLMF
/// 13.7.2 asymptotic expansion. Chosen well inside the asymptotic's accurate
/// regime (~1e-14 rel for z ≥ 50) and far below the series' failure onset
/// (z ≈ 440). frankenscipy-8a4qg.
const HYP1F1_LARGE_Z_THRESHOLD: f64 = 200.0;
const HYP2F1_DIRECT_CHAIN: &[HypergeometricBranch] = &[HypergeometricBranch::DirectSeries];
const HYP2F1_TERMINATING_CHAIN: &[HypergeometricBranch] =
    &[HypergeometricBranch::TerminatingPolynomial];
const HYP2F1_GAUSS_CHAIN: &[HypergeometricBranch] = &[HypergeometricBranch::GaussSummation];
const HYP2F1_PFAFF_CHAIN: &[HypergeometricBranch] = &[
    HypergeometricBranch::PfaffTransform,
    HypergeometricBranch::DirectSeries,
    HypergeometricBranch::UnsupportedAnalyticContinuation,
];
const HYP2F1_IDENTITY_CHAIN: &[HypergeometricBranch] =
    &[HypergeometricBranch::LinearFractionalIdentity];
const HYP2F1_REAL_BRANCH_CUT_CHAIN: &[HypergeometricBranch] =
    &[HypergeometricBranch::RealBranchCutInfinity];
const HYPER_GUARD_CHAIN: &[HypergeometricBranch] = &[
    HypergeometricBranch::ParameterGuard,
    HypergeometricBranch::UnsupportedAnalyticContinuation,
];
const HYPER_UNSUPPORTED_CHAIN: &[HypergeometricBranch] =
    &[HypergeometricBranch::UnsupportedAnalyticContinuation];
const HYP2F1_DIVERGENT_CHAIN: &[HypergeometricBranch] =
    &[HypergeometricBranch::DivergentAtUnitArgument];
const HYPERU_QUADRATURE_STEPS: usize = 4096;
const HYPERU_LOG_UNDERFLOW: f64 = -745.0;
const HYPERU_LOG_OVERFLOW: f64 = 709.0;

/// Select the hypergeometric branch for a scalar special-function problem.
pub fn select_hypergeometric_branch(
    problem: HyperCaspProblem,
    mode: RuntimeMode,
) -> Result<HyperCaspDecision, SpecialError> {
    if !problem.precision_target.is_finite() || problem.precision_target <= 0.0 {
        return Err(SpecialError {
            function: hyper_function_name(problem.function),
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "precision_target must be positive and finite",
        });
    }

    if !problem.a.is_finite()
        || !problem.b.is_finite()
        || !problem.z.is_finite()
        || !problem.parameter_stability_margin.is_finite()
        || problem.c.is_some_and(|c| !c.is_finite())
    {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: hyper_function_name(problem.function),
                kind: SpecialErrorKind::NonFiniteInput,
                mode,
                detail: "hypergeometric CASP inputs must be finite",
            });
        }
        return Ok(hyper_casp_decision(
            HypergeometricBranch::UnsupportedAnalyticContinuation,
            problem,
            0,
            HYPER_UNSUPPORTED_CHAIN,
            "non-finite strict-mode input is delegated to the unsupported branch",
        ));
    }

    if problem.parameter_stability_margin <= problem.precision_target {
        return Ok(hyper_casp_decision(
            HypergeometricBranch::ParameterGuard,
            problem,
            0,
            HYPER_GUARD_CHAIN,
            "lower parameter is too close to a nonpositive-integer pole",
        ));
    }

    match problem.function {
        HypergeometricFunction::Hyp0f1 => Ok(select_hyp0f1_branch(problem)),
        HypergeometricFunction::Hyp1f1 => Ok(select_hyp1f1_branch(problem)),
        HypergeometricFunction::Hyp2f1 => select_hyp2f1_branch(problem, mode),
    }
}

/// Confluent hypergeometric function 1F1(a; b; z).
///
/// Also known as Kummer's function M(a, b, z).
/// Supports scalar or vector parameters with NumPy-style broadcasting.
/// Matches `scipy.special.hyp1f1(a, b, z)`.
pub fn hyp1f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    hyp1f1_dispatch("hyp1f1", a, b, z, mode)
}

/// Tricomi confluent hypergeometric function U(a, b, x).
///
/// Supports real scalar or vector parameters with NumPy-style broadcasting.
/// Complex inputs are intentionally rejected because SciPy's `hyperu` ufunc is
/// real-valued only.
pub fn hyperu(
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    hyperu_dispatch("hyperu", a, b, x, mode)
}

fn hyp1f1_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    // Check for empty tensors
    if matches!(a, SpecialTensor::Empty)
        || matches!(b, SpecialTensor::Empty)
        || matches!(z, SpecialTensor::Empty)
    {
        return Ok(SpecialTensor::Empty);
    }

    // Get vector lengths (0 = scalar)
    let a_len = tensor_vec_len(a);
    let b_len = tensor_vec_len(b);
    let z_len = tensor_vec_len(z);

    // Check broadcast compatibility
    let out_len = broadcast_len_3(a_len, b_len, z_len)
        .ok_or_else(|| broadcast_shape_error(function, mode))?;

    // Check if output is complex
    let is_complex = is_complex_tensor(a) || is_complex_tensor(b) || is_complex_tensor(z);

    if out_len == 0 {
        // All scalars
        if is_complex {
            let a_c = tensor_as_complex_scalar(a)?;
            let b_c = tensor_as_complex_scalar(b)?;
            let z_c = tensor_as_complex_scalar(z)?;
            let result = hyp1f1_complex_parameters(a_c, b_c, z_c, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        } else {
            let a_r = tensor_as_real_scalar(a)?;
            let b_r = tensor_as_real_scalar(b)?;
            let z_r = tensor_as_real_scalar(z)?;
            let result = hyp1f1_scalar(a_r, b_r, z_r, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
    } else {
        // At least one vector
        if is_complex {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_c = tensor_get_complex(a, i, a_len)?;
                let b_c = tensor_get_complex(b, i, b_len)?;
                let z_c = tensor_get_complex(z, i, z_len)?;
                results.push(hyp1f1_complex_parameters(a_c, b_c, z_c, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        } else {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_r = tensor_get_real(a, i, a_len)?;
                let b_r = tensor_get_real(b, i, b_len)?;
                let z_r = tensor_get_real(z, i, z_len)?;
                results.push(hyp1f1_scalar(a_r, b_r, z_r, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
    }
}

fn hyperu_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    if matches!(a, SpecialTensor::Empty)
        || matches!(b, SpecialTensor::Empty)
        || matches!(x, SpecialTensor::Empty)
    {
        return Ok(SpecialTensor::Empty);
    }

    if is_complex_tensor(a) || is_complex_tensor(b) || is_complex_tensor(x) {
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "hyperu supports real-valued inputs only",
        });
    }

    let a_len = tensor_vec_len(a);
    let b_len = tensor_vec_len(b);
    let x_len = tensor_vec_len(x);
    let out_len = broadcast_len_3(a_len, b_len, x_len)
        .ok_or_else(|| broadcast_shape_error(function, mode))?;

    if out_len == 0 {
        let a_r = tensor_as_real_scalar_for(function, a, mode)?;
        let b_r = tensor_as_real_scalar_for(function, b, mode)?;
        let x_r = tensor_as_real_scalar_for(function, x, mode)?;
        return hyperu_scalar(a_r, b_r, x_r, mode).map(SpecialTensor::RealScalar);
    }

    let mut results = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let a_r = tensor_get_real_for(function, a, i, a_len, mode)?;
        let b_r = tensor_get_real_for(function, b, i, b_len, mode)?;
        let x_r = tensor_get_real_for(function, x, i, x_len, mode)?;
        results.push(hyperu_scalar(a_r, b_r, x_r, mode)?);
    }
    Ok(SpecialTensor::RealVec(results))
}

// Helper functions for broadcast dispatch

fn tensor_vec_len(t: &SpecialTensor) -> usize {
    match t {
        SpecialTensor::RealScalar(_) | SpecialTensor::ComplexScalar(_) => 0,
        SpecialTensor::RealVec(v) => v.len(),
        SpecialTensor::ComplexVec(v) => v.len(),
        SpecialTensor::Empty => 0,
    }
}

fn is_complex_tensor(t: &SpecialTensor) -> bool {
    matches!(
        t,
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_)
    )
}

fn broadcast_len_4(a: usize, b: usize, c: usize, d: usize) -> Option<usize> {
    let abc = broadcast_len_3(a, b, c)?;
    broadcast_len_2(abc, d)
}

fn broadcast_len_3(a: usize, b: usize, c: usize) -> Option<usize> {
    let ab = broadcast_len_2(a, b)?;
    broadcast_len_2(ab, c)
}

fn broadcast_len_2(a: usize, b: usize) -> Option<usize> {
    match (a, b) {
        (0, 0) => Some(0),
        (0, n) | (n, 0) => Some(n),
        (m, n) if m == n => Some(m),
        _ => None,
    }
}

fn tensor_as_real_scalar(t: &SpecialTensor) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected real scalar",
        }),
    }
}

fn tensor_as_real_scalar_for(
    function: &'static str,
    t: &SpecialTensor,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "expected real scalar",
        }),
    }
}

fn tensor_as_complex_scalar(t: &SpecialTensor) -> Result<Complex64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(Complex64::from_real(*v)),
        SpecialTensor::ComplexScalar(v) => Ok(*v),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected scalar",
        }),
    }
}

fn tensor_get_real(t: &SpecialTensor, i: usize, len: usize) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        SpecialTensor::RealVec(vec) => Ok(vec[if len == 0 { 0 } else { i }]),
        _ => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "expected real tensor",
        }),
    }
}

fn tensor_get_real_for(
    function: &'static str,
    t: &SpecialTensor,
    i: usize,
    len: usize,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(*v),
        SpecialTensor::RealVec(vec) => {
            vec.get(if len == 0 { 0 } else { i })
                .copied()
                .ok_or(SpecialError {
                    function,
                    kind: SpecialErrorKind::ShapeMismatch,
                    mode,
                    detail: "real vector is empty or shorter than the broadcast output",
                })
        }
        _ => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "expected real tensor",
        }),
    }
}

fn tensor_get_complex(t: &SpecialTensor, i: usize, len: usize) -> Result<Complex64, SpecialError> {
    match t {
        SpecialTensor::RealScalar(v) => Ok(Complex64::from_real(*v)),
        SpecialTensor::ComplexScalar(v) => Ok(*v),
        SpecialTensor::RealVec(vec) => Ok(Complex64::from_real(vec[if len == 0 { 0 } else { i }])),
        SpecialTensor::ComplexVec(vec) => Ok(vec[if len == 0 { 0 } else { i }]),
        SpecialTensor::Empty => Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "empty tensor",
        }),
    }
}

/// Confluent hypergeometric limit function 0F1(; b; z).
///
/// Also known as the regularized confluent hypergeometric limit function.
/// Defined as: 0F1(; b; z) = Σ_{n=0}^∞ z^n / ((b)_n * n!)
///
/// Related to Bessel functions: J_n(x) = (x/2)^n / Γ(n+1) * 0F1(; n+1; -x²/4)
///
/// Supports scalar real or complex parameters with real or complex
/// scalar/vector `z` inputs.
/// Matches `scipy.special.hyp0f1(b, z)`.
///
/// # Arguments
/// * `b` - Parameter (must not be a non-positive integer)
/// * `z` - Argument
///
/// # Returns
/// Value of 0F1(; b; z)
pub fn hyp0f1(b: &SpecialTensor, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match (b, z) {
        (SpecialTensor::RealScalar(b_val), SpecialTensor::RealScalar(z_val)) => {
            let result = hyp0f1_scalar(*b_val, *z_val, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::RealVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_scalar(*b_val, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::RealScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_scalar(bi, *z_val, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::RealVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_scalar(bi, zi, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::RealScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(*b_val, Complex64::from_real(*z_val), mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::RealVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(
                    *b_val,
                    Complex64::from_real(zi),
                    mode,
                )?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::ComplexScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(*b_val, *z_val, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::ComplexScalar(b_val), SpecialTensor::ComplexVec(z_vec)) => {
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(*b_val, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::RealScalar(z_val)) => {
            let z_complex = Complex64::from_real(*z_val);
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(bi, z_complex, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::ComplexScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(bi, *z_val, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::RealVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(bi, Complex64::from_real(zi), mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::ComplexVec(b_vec), SpecialTensor::ComplexVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(bi, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::ComplexScalar(z_val)) => {
            let mut results = Vec::with_capacity(b_vec.len());
            for &bi in b_vec {
                results.push(hyp0f1_complex_scalar(
                    Complex64::from_real(bi),
                    *z_val,
                    mode,
                )?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealVec(b_vec), SpecialTensor::ComplexVec(z_vec)) => {
            if b_vec.len() != z_vec.len() {
                return Err(broadcast_shape_error("hyp0f1", mode));
            }
            let mut results = Vec::with_capacity(b_vec.len());
            for (&bi, &zi) in b_vec.iter().zip(z_vec.iter()) {
                results.push(hyp0f1_complex_scalar(Complex64::from_real(bi), zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::ComplexScalar(z_val)) => {
            let result = hyp0f1_complex_scalar(Complex64::from_real(*b_val), *z_val, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        }
        (SpecialTensor::RealScalar(b_val), SpecialTensor::ComplexVec(z_vec)) => {
            let b_complex = Complex64::from_real(*b_val);
            let mut results = Vec::with_capacity(z_vec.len());
            for &zi in z_vec {
                results.push(hyp0f1_complex_scalar(b_complex, zi, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        }
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Ok(SpecialTensor::Empty),
    }
}

/// Scalar implementation of 0F1(; b; z).
pub fn hyp0f1_scalar(b: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // Check for pole at non-positive integer b
    if b <= 0.0 && b == b.floor() {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp0f1",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "hyp0f1: b must not be a non-positive integer",
            });
        }
        return Ok(f64::NAN);
    }

    // Handle special cases
    if z == 0.0 {
        return Ok(1.0);
    }

    let decision = select_hypergeometric_branch(HyperCaspProblem::hyp0f1(b, z, 1.0e-14), mode)?;
    match decision.branch {
        HypergeometricBranch::DirectSeries => Ok(hyp0f1_series(b, z)),
        HypergeometricBranch::AsymptoticExpansion => Ok(hyp0f1_asymptotic(b, z)),
        HypergeometricBranch::ParameterGuard => {
            guarded_hypergeometric_parameter("hyp0f1", mode, decision.reason)
        }
        HypergeometricBranch::UnsupportedAnalyticContinuation => {
            unsupported_hypergeometric_branch("hyp0f1", mode, decision.reason)
        }
        _ => unsupported_hypergeometric_branch(
            "hyp0f1",
            mode,
            "selected branch is not valid for 0F1",
        ),
    }
}

/// Power series for 0F1(; b; z).
/// 0F1(; b; z) = Σ_{n=0}^∞ z^n / ((b)_n * n!)
fn hyp0f1_series(b: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 1..300 {
        let nf = n as f64;
        // term_n = term_{n-1} * z / (n * (b + n - 1))
        term *= z / (nf * (b + nf - 1.0));
        sum += term;

        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
        if !sum.is_finite() {
            break;
        }
    }

    sum
}

fn broadcast_shape_error(function: &'static str, mode: RuntimeMode) -> SpecialError {
    SpecialError {
        function,
        kind: SpecialErrorKind::ShapeMismatch,
        mode,
        detail: "vector parameters must have the same length for element-wise operations",
    }
}

fn complex_is_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn complex_is_nonpositive_integer(value: Complex64) -> bool {
    value.im == 0.0 && value.re <= 0.0 && value.re == value.re.floor()
}

fn complex_series_converged(term: Complex64, sum: Complex64, tol: f64) -> bool {
    term.abs() <= tol * sum.abs().max(1.0)
}

fn hyper_function_name(function: HypergeometricFunction) -> &'static str {
    match function {
        HypergeometricFunction::Hyp0f1 => "hyp0f1",
        HypergeometricFunction::Hyp1f1 => "hyp1f1",
        HypergeometricFunction::Hyp2f1 => "hyp2f1",
    }
}

fn hyper_casp_decision(
    branch: HypergeometricBranch,
    problem: HyperCaspProblem,
    max_terms: usize,
    fallback_chain: &'static [HypergeometricBranch],
    reason: &'static str,
) -> HyperCaspDecision {
    HyperCaspDecision {
        branch,
        precision_target: problem.precision_target,
        max_terms,
        parameter_stability_margin: problem.parameter_stability_margin,
        fallback_chain,
        reason,
    }
}

fn select_hyp0f1_branch(problem: HyperCaspProblem) -> HyperCaspDecision {
    if problem.z_abs < 50.0 {
        return hyper_casp_decision(
            HypergeometricBranch::DirectSeries,
            problem,
            300,
            HYP0F1_DIRECT_CHAIN,
            "moderate real 0F1 argument uses the direct power series",
        );
    }

    hyper_casp_decision(
        HypergeometricBranch::AsymptoticExpansion,
        problem,
        10,
        HYP0F1_ASYMPTOTIC_CHAIN,
        "large real 0F1 argument uses the asymptotic Bessel expansion",
    )
}

fn select_hyp1f1_branch(problem: HyperCaspProblem) -> HyperCaspDecision {
    if problem.z < -20.0 {
        // A nonpositive-integer a makes 1F1 a finite polynomial; the direct
        // series is exact and cheap, whereas Kummer would map it onto a
        // non-terminating inner series that overruns the 500-term cap.
        if is_nonpositive_integer(problem.a) {
            return hyper_casp_decision(
                HypergeometricBranch::DirectSeries,
                problem,
                500,
                HYP1F1_DIRECT_CHAIN,
                "nonpositive-integer a terminates 1F1; direct series is exact",
            );
        }
        // Past -threshold the Kummer inner series (argument -z) itself overruns
        // the 500-term cap, so use the DLMF 13.7.2 z -> -∞ asymptotic instead.
        if problem.z < -HYP1F1_LARGE_Z_THRESHOLD {
            return hyper_casp_decision(
                HypergeometricBranch::AsymptoticExpansion,
                problem,
                500,
                HYP1F1_ASYMPTOTIC_CHAIN,
                "large negative z uses the DLMF 13.7.2 asymptotic expansion",
            );
        }
        return hyper_casp_decision(
            HypergeometricBranch::KummerTransform,
            problem,
            500,
            HYP1F1_KUMMER_CHAIN,
            "large negative z is evaluated through Kummer's transformation",
        );
    }

    if problem.z > HYP1F1_LARGE_Z_THRESHOLD {
        return hyper_casp_decision(
            HypergeometricBranch::AsymptoticExpansion,
            problem,
            500,
            HYP1F1_ASYMPTOTIC_CHAIN,
            "large positive z uses the DLMF 13.7.2 asymptotic expansion",
        );
    }

    hyper_casp_decision(
        HypergeometricBranch::DirectSeries,
        problem,
        500,
        HYP1F1_DIRECT_CHAIN,
        "moderate real 1F1 argument uses the direct power series",
    )
}

fn select_hyp2f1_branch(
    problem: HyperCaspProblem,
    mode: RuntimeMode,
) -> Result<HyperCaspDecision, SpecialError> {
    let c = problem.c.ok_or(SpecialError {
        function: "hyp2f1",
        kind: SpecialErrorKind::DomainError,
        mode,
        detail: "2F1 CASP problem requires denominator parameter c",
    })?;

    if is_nonpositive_integer(problem.a) || is_nonpositive_integer(problem.b) {
        return Ok(hyper_casp_decision(
            HypergeometricBranch::TerminatingPolynomial,
            problem,
            500,
            HYP2F1_TERMINATING_CHAIN,
            "nonpositive-integer numerator parameter terminates the 2F1 series",
        ));
    }

    if problem.z_abs < 1.0 {
        // For z < 0 inside the unit disk the Pfaff transform z' = z/(z-1)
        // always lands inside (0, 1/2) for z ∈ (-1, 0), which converges
        // strictly faster than the direct series (and avoids cancellation
        // from alternating signs). The original dispatch only enabled
        // Pfaff for |z| ≥ 1, so e.g. z = -0.99 hit the 500-term cap on
        // the direct series and returned NaN — frankenscipy-3zzep.
        if problem.z < 0.0 {
            return Ok(hyper_casp_decision(
                HypergeometricBranch::PfaffTransform,
                problem,
                500,
                HYP2F1_PFAFF_CHAIN,
                "negative real argument uses the Pfaff transform for fast convergence",
            ));
        }
        return Ok(hyper_casp_decision(
            HypergeometricBranch::DirectSeries,
            problem,
            500,
            HYP2F1_DIRECT_CHAIN,
            "argument inside the unit disk uses the direct 2F1 series",
        ));
    }

    if (problem.z - 1.0).abs() < f64::EPSILON {
        if c - problem.a - problem.b > 0.0 {
            return Ok(hyper_casp_decision(
                HypergeometricBranch::GaussSummation,
                problem,
                0,
                HYP2F1_GAUSS_CHAIN,
                "z = 1 with c - a - b > 0 uses Gauss summation",
            ));
        }

        return Ok(hyper_casp_decision(
            HypergeometricBranch::DivergentAtUnitArgument,
            problem,
            0,
            HYP2F1_DIVERGENT_CHAIN,
            "2F1 diverges at z = 1 when c <= a + b",
        ));
    }

    if problem.z < 0.0 {
        return Ok(hyper_casp_decision(
            HypergeometricBranch::PfaffTransform,
            problem,
            500,
            HYP2F1_PFAFF_CHAIN,
            "negative real argument outside the unit disk uses the Pfaff transform",
        ));
    }

    if problem.z > 1.0 {
        // Euler's transformation 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z)
        // produces a *terminating* series whenever c-a or c-b is a nonpositive
        // integer, so the analytic continuation is an exact finite polynomial in
        // z divided by an integer power of (1-z). For real z > 1 the prefactor
        // (1-z)^{c-a-b} is real-valued only when the exponent is an integer; the
        // matching integrality requirement on the *other* parameter (b integer
        // when c-a terminates, a integer when c-b terminates) guarantees that.
        // This generalizes the old c==a / c==b linear-fractional identity (the
        // n=0 special case) to the full Euler-terminating family that
        // scipy.special.hyp2f1 returns finitely — frankenscipy-nwvrw. Inputs
        // outside this family stay on SciPy's branch cut and return +inf below.
        let terminates_b = is_nonpositive_integer(c - problem.a) && is_integer(problem.b);
        let terminates_a = is_nonpositive_integer(c - problem.b) && is_integer(problem.a);
        if terminates_a || terminates_b {
            return Ok(hyper_casp_decision(
                HypergeometricBranch::LinearFractionalIdentity,
                problem,
                0,
                HYP2F1_IDENTITY_CHAIN,
                "z > 1 Euler-terminating case reduces to a finite polynomial via \
                 (1-z)^{c-a-b} 2F1(c-a,c-b;c;z)",
            ));
        }
    }

    if problem.z > 1.0 {
        return Ok(hyper_casp_decision(
            HypergeometricBranch::RealBranchCutInfinity,
            problem,
            0,
            HYP2F1_REAL_BRANCH_CUT_CHAIN,
            "generic real z > 1 lies on SciPy's branch cut and returns positive infinity",
        ));
    }

    Ok(hyper_casp_decision(
        HypergeometricBranch::UnsupportedAnalyticContinuation,
        problem,
        0,
        HYPER_UNSUPPORTED_CHAIN,
        "2F1 analytic continuation for this real argument is not implemented",
    ))
}

fn lower_parameter_stability_margin(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }

    if x > 0.0 {
        return x;
    }

    let nearest = x.round();
    if nearest <= 0.0 {
        (x - nearest).abs()
    } else {
        x.abs()
    }
}

fn guarded_hypergeometric_parameter(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail,
        });
    }
    Ok(f64::NAN)
}

fn unsupported_hypergeometric_branch(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> Result<f64, SpecialError> {
    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail,
        });
    }
    Ok(f64::NAN)
}

fn divergent_hyp2f1_at_unit_argument(mode: RuntimeMode) -> Result<f64, SpecialError> {
    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "hyp2f1",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "2F1 diverges at z=1 when c <= a+b",
        });
    }
    Ok(f64::INFINITY)
}

fn hyp0f1_complex_scalar(
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if complex_is_nonpositive_integer(b) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp0f1",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "hyp0f1: b must not be a non-positive integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) {
        return Ok(Complex64::from_real(1.0));
    }

    // The ascending 0F1 series has its largest term at n ≈ √|z| with magnitude
    // ~e^{2√|z|}, while on the oscillatory (J-type) directions — z negative or
    // large |Im z| — the result is exponentially smaller, so the series cancels
    // catastrophically (rel ~2 by |z|=400, ~1e69 by |z|=10^4). For real b route
    // through the Bessel link 0F1(;b;z) = Γ(b) z^{(1−b)/2} I_{b−1}(2√z), using
    // the now-exact complex modified Bessel I (real order), which carries no
    // cancellation. scipy's hyp0f1 ufunc accepts only real b, matching this gate.
    if b.im == 0.0 && z.abs() >= 20.0 {
        return Ok(hyp0f1_via_bessel_i(b.re, z));
    }

    hyp0f1_series_complex(b, z, mode)
}

/// 0F1(;b;z) = Γ(b) z^{(1−b)/2} I_{b−1}(2√z) for real b, used at large |z|
/// where the ascending series cancels. Accurate across the whole z-plane
/// (~1e-14 vs mpmath) because complex_iv_scalar is exact for all arguments.
fn hyp0f1_via_bessel_i(b: f64, z: Complex64) -> Complex64 {
    let sqrt_z = z.powf(0.5);
    let arg = sqrt_z * 2.0;
    let i_bessel = crate::bessel::complex_iv_scalar(b - 1.0, arg);
    let gamma_b = crate::gamma::complex_gammaln(Complex64::from_real(b)).exp();
    let power = z.powc(Complex64::from_real((1.0 - b) * 0.5));
    gamma_b * power * i_bessel
}

fn hyp0f1_series_complex(
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 4_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 1..=max_terms {
        let nf = n as f64;
        let denom = (b + Complex64::from_real(nf - 1.0)) * nf;
        term = term * z / denom;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp0f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex-valued hyp0f1",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Asymptotic expansion for 0F1(; b; z) for large |z|.
/// Uses relation to Bessel functions.
fn hyp0f1_asymptotic(b: f64, z: f64) -> f64 {
    // 0F1(; b; z) is related to Bessel functions:
    // 0F1(; b; -x²/4) = Γ(b) * (x/2)^(1-b) * J_{b-1}(x)  for x > 0
    // 0F1(; b; x²/4) = Γ(b) * (x/2)^(1-b) * I_{b-1}(x)   for x > 0
    //
    // For large positive z: use modified Bessel I
    // For large negative z: use Bessel J

    // Compute gamma(b) via exp(gammaln(b))
    // Use gammaln_scalar with Strict mode (won't fail for b > 0)
    let ln_gamma_b = crate::gamma::gammaln_scalar(b, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let gamma_b = ln_gamma_b.exp();

    if z > 0.0 {
        // z = x²/4, so x = 2*sqrt(z)
        let x = 2.0 * z.sqrt();
        let nu = b - 1.0;

        // 0F1(; b; z) = Γ(b) * z^((1-b)/2) * I_{b-1}(x)
        let i_val = bessel_i_asymptotic(nu, x);
        gamma_b * z.powf((1.0 - b) / 2.0) * i_val
    } else {
        // z = -x²/4, so x = 2*sqrt(-z)
        let x = 2.0 * (-z).sqrt();
        let nu = b - 1.0;

        // 0F1(; b; z) = Γ(b) * (-z)^((1-b)/2) * J_{b-1}(x)
        let j_val = bessel_j_asymptotic(nu, x);
        gamma_b * (-z).powf((1.0 - b) / 2.0) * j_val
    }
}

/// Asymptotic approximation for I_nu(x) for large x.
fn bessel_i_asymptotic(nu: f64, x: f64) -> f64 {
    // I_nu(x) ~ exp(x) / sqrt(2*pi*x) * (1 - (4*nu²-1)/(8x) + ...)
    let coeff = (2.0 * std::f64::consts::PI * x).sqrt().recip();
    let mu = 4.0 * nu * nu;

    let mut sum = 1.0;
    let mut term = 1.0;
    let x_inv = 1.0 / x;

    for k in 1..10 {
        let kf = k as f64;
        term *= -(mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf) * x_inv;
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }

    x.exp() * coeff * sum
}

/// Asymptotic expansion of J_nu(x) for large x (DLMF 10.17.3):
///
///   J_ν(x) ~ sqrt(2/(πx)) [cos(ω) P(ν,x) - sin(ω) Q(ν,x)],   ω = x - νπ/2 - π/4,
///   P = Σ_j (-1)^j a_{2j}/x^{2j},   Q = Σ_j (-1)^j a_{2j+1}/x^{2j+1},
///   a_0 = 1,   a_k = a_{k-1} (4ν² - (2k-1)²) / (8k).
///
/// The previous implementation kept only the leading term (P = 1, Q = 0), which
/// is accurate to O(1/x) and left 0F1's oscillatory (z < 0) branch ~1-3% off
/// SciPy. The full series — summed to its smallest term, since it is asymptotic
/// (divergent) — restores ~1e-14 agreement. frankenscipy-o9ws0.
fn bessel_j_asymptotic(nu: f64, x: f64) -> f64 {
    let mu = 4.0 * nu * nu;
    let mut term = 1.0_f64; // a_k / x^k, a_0 = 1
    let mut prev_abs = 1.0_f64;
    let mut p = 1.0_f64; // k = 0 term of P
    let mut q = 0.0_f64;
    for k in 1..64 {
        let kf = k as f64;
        term *= (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf * x);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // asymptotic series past its smallest term — truncate
        }
        if k % 2 == 0 {
            let sign = if (k / 2) % 2 == 0 { 1.0 } else { -1.0 };
            p += sign * term;
        } else {
            let sign = if ((k - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            q += sign * term;
        }
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON {
            break;
        }
    }
    let omega = x - nu * std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4;
    let amplitude = (2.0 / (std::f64::consts::PI * x)).sqrt();
    amplitude * (omega.cos() * p - omega.sin() * q)
}

/// Gauss hypergeometric function 2F1(a, b; c; z).
///
/// Supports scalar or vector parameters with NumPy-style broadcasting.
/// Complex-valued evaluation uses the convergent `|z| < 1` series path.
/// Matches `scipy.special.hyp2f1(a, b, c, z)`.
pub fn hyp2f1(
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    hyp2f1_dispatch("hyp2f1", a, b, c, z, mode)
}

fn hyp2f1_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    b: &SpecialTensor,
    c: &SpecialTensor,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    // Check for empty tensors
    if matches!(a, SpecialTensor::Empty)
        || matches!(b, SpecialTensor::Empty)
        || matches!(c, SpecialTensor::Empty)
        || matches!(z, SpecialTensor::Empty)
    {
        return Ok(SpecialTensor::Empty);
    }

    // Get vector lengths (0 = scalar)
    let a_len = tensor_vec_len(a);
    let b_len = tensor_vec_len(b);
    let c_len = tensor_vec_len(c);
    let z_len = tensor_vec_len(z);

    // Check broadcast compatibility
    let out_len = broadcast_len_4(a_len, b_len, c_len, z_len)
        .ok_or_else(|| broadcast_shape_error(function, mode))?;

    // Check if output is complex
    let is_complex = is_complex_tensor(a)
        || is_complex_tensor(b)
        || is_complex_tensor(c)
        || is_complex_tensor(z);

    if out_len == 0 {
        // All scalars
        if is_complex {
            let a_c = tensor_as_complex_scalar(a)?;
            let b_c = tensor_as_complex_scalar(b)?;
            let c_c = tensor_as_complex_scalar(c)?;
            let z_c = tensor_as_complex_scalar(z)?;
            let result = hyp2f1_complex_parameters(a_c, b_c, c_c, z_c, mode)?;
            Ok(SpecialTensor::ComplexScalar(result))
        } else {
            let a_r = tensor_as_real_scalar(a)?;
            let b_r = tensor_as_real_scalar(b)?;
            let c_r = tensor_as_real_scalar(c)?;
            let z_r = tensor_as_real_scalar(z)?;
            let result = hyp2f1_scalar(a_r, b_r, c_r, z_r, mode)?;
            Ok(SpecialTensor::RealScalar(result))
        }
    } else {
        // At least one vector
        if is_complex {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_c = tensor_get_complex(a, i, a_len)?;
                let b_c = tensor_get_complex(b, i, b_len)?;
                let c_c = tensor_get_complex(c, i, c_len)?;
                let z_c = tensor_get_complex(z, i, z_len)?;
                results.push(hyp2f1_complex_parameters(a_c, b_c, c_c, z_c, mode)?);
            }
            Ok(SpecialTensor::ComplexVec(results))
        } else {
            let mut results = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let a_r = tensor_get_real(a, i, a_len)?;
                let b_r = tensor_get_real(b, i, b_len)?;
                let c_r = tensor_get_real(c, i, c_len)?;
                let z_r = tensor_get_real(z, i, z_len)?;
                results.push(hyp2f1_scalar(a_r, b_r, c_r, z_r, mode)?);
            }
            Ok(SpecialTensor::RealVec(results))
        }
    }
}

/// Scalar 1F1(a; b; z) via series summation.
///
/// 1F1(a; b; z) = Σ_{n=0}^∞ (a)_n z^n / ((b)_n n!)
/// where (a)_n = a(a+1)...(a+n-1) is the Pochhammer symbol.
fn hyp1f1_scalar(a: f64, b: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // A nonpositive-integer a terminates 1F1 into a degree-|a| polynomial that
    // is EXACT for every z. This must be decided before the b-pole guard and
    // before the large-|z| asymptotic, both of which mishandle it:
    //   * the z > 200 asymptotic carries a 1/Γ(a) factor that is 0 for a
    //     nonpositive integer, so it returned ~0 (hyp1f1(-1,0.5,300) gave 0 vs
    //     scipy -599);
    //   * the b-guard rejects negative-integer b outright, but the polynomial
    //     is well defined when b is a negative integer too — provided the
    //     (b)_k denominator does not vanish before the a-series terminates.
    // The a-series runs k = 0..|a|; for b = -|b| the denominator (b)_k first
    // hits 0 at k = |b|+1. Hence |a| <= |b| → finite polynomial, while
    // |a| > |b| → a genuine pole, where scipy returns +inf (even at z = 0).
    if is_nonpositive_integer(a) {
        if b <= 0.0 && b == b.floor() && b > a {
            return Ok(f64::INFINITY);
        }
        return hyp1f1_series(a, b, z, mode);
    }

    // b a nonpositive integer is a pole of 1/Γ(b): the series hits a zero
    // denominator (b)_k with a nonzero numerator (the terminating-a cases that
    // would survive it are already handled above), so the function diverges.
    // scipy.special.hyp1f1 returns +inf here for every z; reproduce that in the
    // permissive modes. Hardened keeps the fail-closed domain error.
    if b == 0.0 || (b < 0.0 && b == b.floor()) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp1f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "b must not be zero or a negative integer",
            });
        }
        return Ok(f64::INFINITY);
    }

    // Special cases
    if z == 0.0 {
        return Ok(1.0);
    }
    if a == 0.0 {
        return Ok(1.0);
    }
    if a == b {
        return Ok(z.exp());
    }

    let decision = select_hypergeometric_branch(HyperCaspProblem::hyp1f1(a, b, z, 1.0e-14), mode)?;
    match decision.branch {
        HypergeometricBranch::KummerTransform => {
            // M(a,b,z) = e^z M(b-a, b, -z)
            let inner = hyp1f1_series(b - a, b, -z, mode)?;
            Ok(z.exp() * inner)
        }
        HypergeometricBranch::DirectSeries => hyp1f1_series(a, b, z, mode),
        HypergeometricBranch::AsymptoticExpansion => {
            if z > 0.0 {
                hyp1f1_asymptotic(a, b, z)
            } else {
                hyp1f1_asymptotic_negative(a, b, z)
            }
        }
        HypergeometricBranch::ParameterGuard => {
            guarded_hypergeometric_parameter("hyp1f1", mode, decision.reason)
        }
        HypergeometricBranch::UnsupportedAnalyticContinuation => {
            unsupported_hypergeometric_branch("hyp1f1", mode, decision.reason)
        }
        _ => unsupported_hypergeometric_branch(
            "hyp1f1",
            mode,
            "selected branch is not valid for 1F1",
        ),
    }
}

/// Direct series summation for 1F1.
fn hyp1f1_series(a: f64, b: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let max_terms = 500;
    let eps = f64::EPSILON;

    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 0..max_terms {
        let nf = n as f64;
        // A nonpositive-integer a terminates the series at k = n: the (a + nf)
        // numerator factor is exactly zero, so this and every later term vanish.
        // Return before the multiply to avoid a 0/0 when b is also a negative
        // integer of the same magnitude (e.g. hyp1f1(-2,-2,z): (a+2)=(b+2)=0).
        if a + nf == 0.0 {
            return Ok(sum);
        }
        term *= (a + nf) * z / ((b + nf) * (nf + 1.0));

        if !term.is_finite() {
            return hyp1f1_unconverged(
                mode,
                "series term overflowed before convergence was established",
            );
        }

        sum += term;

        if !sum.is_finite() {
            return hyp1f1_unconverged(
                mode,
                "series sum overflowed before convergence was established",
            );
        }

        if term == 0.0 {
            return Ok(sum);
        }

        if term.abs() < eps * sum.abs() {
            return Ok(sum);
        }
    }

    hyp1f1_unconverged(mode, "series did not converge within 500 terms")
}

/// Large positive-z asymptotic for 1F1 (DLMF 13.7.2, dominant term as z → +∞):
///
///   M(a,b,z) ~ Γ(b)/Γ(a) · e^z · z^{a-b} · Σ_{s≥0} (b-a)_s (1-a)_s / (s! z^s).
///
/// The exponentially small e^{-z}-weighted companion term is negligible for the
/// z > 200 regime this serves. The series is asymptotic (divergent); we sum to
/// its smallest term (optimal truncation). The prefactor is built in log space
/// to avoid overflowing the intermediate e^z, which lets us return correct
/// finite values well past where the direct series fails.
///
/// Parity: SciPy evaluates e^z directly and therefore overflows to ±inf once
/// z exceeds ln(f64::MAX) ≈ 709.7827, even when the true value is representable.
/// We reproduce that boundary so conformance matches `scipy.special.hyp1f1`.
fn hyp1f1_asymptotic(a: f64, b: f64, z: f64) -> Result<f64, SpecialError> {
    const LN_F64_MAX: f64 = 709.782_712_893_384;

    // Prefactor sign and log-magnitude: Γ(b)/Γ(a) · e^z · z^{a-b}, z > 0.
    let (ln_gb, sign_b) = ln_gamma_with_sign(b);
    let (ln_ga, sign_a) = ln_gamma_with_sign(a);
    let sign = sign_a * sign_b;
    let ln_pref = ln_gb - ln_ga + z + (a - b) * z.ln();

    // Optimal-truncation asymptotic series Σ (b-a)_s (1-a)_s / (s! z^s).
    let mut series = 1.0_f64;
    let mut term = 1.0_f64;
    let mut prev_abs = 1.0_f64;
    for s in 0..1024 {
        let sf = s as f64;
        term *= (b - a + sf) * (1.0 - a + sf) / ((sf + 1.0) * z);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // asymptotic series past its smallest term — truncate
        }
        series += term;
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON * series.abs() {
            break;
        }
    }

    // Match SciPy's e^z overflow boundary.
    if z > LN_F64_MAX {
        return Ok(sign * series.signum() * f64::INFINITY);
    }
    Ok(sign * ln_pref.exp() * series)
}

/// Large negative-z asymptotic for 1F1 (DLMF 13.7.2 as z → -∞).
///
/// Generically the algebraic term dominates (the e^z companion is
/// exponentially small for z → -∞ and below f64 relative precision):
///
///   M(a,b,z) ~ Γ(b)/Γ(b-a) · (-z)^{-a} · Σ_{s≥0} (a)_s (a-b+1)_s / (s! (-z)^s).
///
/// When b - a is a nonpositive integer that term vanishes (1/Γ(b-a) = 0) and the
/// otherwise-subdominant exponential term is the whole answer (a - b = k is then
/// a nonnegative integer, so z^k is real):
///
///   M(a,b,z) ~ Γ(b)/Γ(a) · e^z · z^{a-b} · Σ_{s≥0} (b-a)_s (1-a)_s / (s! z^s).
///
/// Both branches build the prefactor in log space. (a is guaranteed non-(nonpos
/// integer) here — the polynomial case is routed to the direct series upstream.)
fn hyp1f1_asymptotic_negative(a: f64, b: f64, z: f64) -> Result<f64, SpecialError> {
    let neg_z = -z; // > 0
    let b_minus_a = b - a;

    if is_nonpositive_integer(b_minus_a) {
        // Exponential term; a - b = k is a nonnegative integer.
        let k = (a - b).round();
        let (ln_gb, sign_b) = ln_gamma_with_sign(b);
        let (ln_ga, sign_a) = ln_gamma_with_sign(a);
        let z_pow_sign = if (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        let sign = sign_a * sign_b * z_pow_sign;
        let ln_pref = ln_gb - ln_ga + z + k * neg_z.ln();
        let mut series = 1.0_f64;
        let mut term = 1.0_f64;
        let mut prev_abs = 1.0_f64;
        for s in 0..1024 {
            let sf = s as f64;
            term *= (b_minus_a + sf) * (1.0 - a + sf) / ((sf + 1.0) * z);
            let abs_term = term.abs();
            if abs_term > prev_abs {
                break;
            }
            series += term;
            prev_abs = abs_term;
            if abs_term <= f64::EPSILON * series.abs() {
                break;
            }
        }
        return Ok(sign * ln_pref.exp() * series);
    }

    // Algebraic term Γ(b)/Γ(b-a) (-z)^{-a} Σ (a)_s (a-b+1)_s / (s! (-z)^s).
    let (ln_gb, sign_b) = ln_gamma_with_sign(b);
    let (ln_gba, sign_ba) = ln_gamma_with_sign(b_minus_a);
    let sign = sign_b * sign_ba;
    let ln_pref = ln_gb - ln_gba - a * neg_z.ln();
    let mut series = 1.0_f64;
    let mut term = 1.0_f64;
    let mut prev_abs = 1.0_f64;
    for s in 0..1024 {
        let sf = s as f64;
        term *= (a + sf) * (a - b + 1.0 + sf) / ((sf + 1.0) * neg_z);
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break;
        }
        series += term;
        prev_abs = abs_term;
        if abs_term <= f64::EPSILON * series.abs() {
            break;
        }
    }
    Ok(sign * ln_pref.exp() * series)
}

fn hyp1f1_unconverged(mode: RuntimeMode, detail: &'static str) -> Result<f64, SpecialError> {
    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "hyp1f1",
            kind: SpecialErrorKind::OverflowRisk,
            mode,
            detail,
        });
    }
    Ok(f64::NAN)
}

/// DLMF 13.7.3 large-x asymptotic for U(a, b, x):
///
///   U(a,b,x) ~ x^{-a} Σ_{k≥0} (a)_k (a-b+1)_k / k! · (-1/x)^k.
///
/// The series is divergent, so we sum to its smallest term (optimal
/// truncation). Unlike the Γ-weighted connection formula — whose two 1F1 terms
/// each grow like e^x while U is recessive (~x^{-a}), so they cancel
/// catastrophically for large x — this expansion is cancellation-free. Returns
/// `Some(value)` only when the optimal-truncation floor is below ~1e-7 relative
/// to the partial sum (the asymptotic has resolved), otherwise `None` so the
/// caller falls back to the small-x-accurate connection formula.
fn hyperu_large_x_asymptotic(a: f64, b: f64, x: f64) -> Option<f64> {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut prev_abs = 1.0_f64;
    let mut min_abs = 1.0_f64;
    for k in 1..400 {
        let kf = k as f64;
        term *= (a + kf - 1.0) * (a - b + kf) / (kf * (-x));
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // divergence onset: the smallest term was the previous one
        }
        sum += term;
        prev_abs = abs_term;
        min_abs = abs_term;
        if abs_term <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    if min_abs <= 1e-7 * sum.abs() {
        Some(x.powf(-a) * sum)
    } else {
        None
    }
}

/// Scalar implementation of Tricomi's confluent hypergeometric U(a, b, x).
pub fn hyperu_scalar(a: f64, b: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return nonfinite_hyperu(mode);
    }

    if x.is_infinite() && x.is_sign_positive() && a.is_finite() && b.is_finite() {
        if a == 0.0 {
            return Ok(1.0);
        }
        if a > 0.0 {
            return Ok(0.0);
        }
        if is_nonpositive_integer(a) {
            return Ok(hyperu_terminating_polynomial(a, b, x));
        }
        return Ok(f64::NAN);
    }

    if !a.is_finite() || !b.is_finite() || !x.is_finite() {
        return nonfinite_hyperu(mode);
    }

    if x < 0.0 {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyperu",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "hyperu is real-valued only for x >= 0",
            });
        }
        return Ok(f64::NAN);
    }

    if x == 0.0 {
        return Ok(hyperu_at_zero(a, b));
    }

    if a == 0.0 {
        return Ok(1.0);
    }

    if is_nonpositive_integer(a) {
        return Ok(hyperu_terminating_polynomial(a, b, x));
    }

    if a > 0.0 {
        return hyperu_positive_a_integral(a, b, x, mode);
    }

    if !is_near_integer(b) {
        // For large x the connection formula's two 1F1 terms cancel
        // catastrophically (U(-0.5,-1.5,50) was -3e8 vs scipy 7.21); prefer the
        // cancellation-free DLMF 13.7.3 asymptotic whenever it resolves.
        if let Some(v) = hyperu_large_x_asymptotic(a, b, x) {
            return Ok(v);
        }
        return hyperu_connection_formula(a, b, x, mode);
    }

    // a < 0 (non-integer), b a positive integer, x > 0: the Gamma-weighted
    // connection formula is an indeterminate 0/0 here (Γ(1-b) and Γ(b-1) hit
    // poles). The confluent integral 1/Γ(a)∫₀^∞ e^{-xt} t^{a-1}(1+t)^{b-a-1} dt
    // is valid for any real b but only converges for a > 0, so seed two values
    // at a+m, a+m+1 (both > 0) and walk downward in a via A&S 13.4.15:
    //   U(a-1,b,x) = (2a + x - b) U(a,b,x) - a(a-b+1) U(a+1,b,x).
    // This stays on the dominant (recessive-free) downward direction and
    // recovers scipy.special.hyperu to ~1e-9 relative — frankenscipy-msy83.
    // b = 0 stays NaN to match scipy; b < 0 integer is a separate gap.
    if b.round() >= 1.0 {
        return hyperu_integer_b_recurrence(a, b, x, mode);
    }

    if b.round() <= -1.0 {
        // b a negative integer: the Kummer transform
        //   U(a, b, x) = x^{1-b} U(a-b+1, 2-b, x)
        // maps to b' = 2-b >= 3 (positive integer). The shifted first argument
        // a' = a-b+1 is either positive (positive-a integral) or negative
        // non-integer (handled by the b'>=1 downward recurrence above), so the
        // recursive call resolves in one level without re-entering this branch
        // — frankenscipy-4yh8z.
        let inner = hyperu_scalar(a - b + 1.0, 2.0 - b, x, mode)?;
        let value = x.powf(1.0 - b) * inner;
        if value.is_finite() {
            return Ok(value);
        }
        return unsupported_hypergeometric_branch(
            "hyperu",
            mode,
            "negative-integer-b hyperu Kummer transform produced a non-finite value",
        );
    }

    // b == 0 stays NaN/error to match scipy.special.hyperu.
    unsupported_hypergeometric_branch(
        "hyperu",
        mode,
        "hyperu with b = 0 and negative non-integer a is not finite-valued",
    )
}

/// U(a, b, x) for a < 0 non-integer and b a positive integer (x > 0) via the
/// confluent integral at positive a plus downward recurrence in a.
fn hyperu_integer_b_recurrence(
    a: f64,
    b: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    // Smallest integer shift m >= 1 placing a + m strictly positive.
    let mut m = 1_i32;
    while a + f64::from(m) <= 0.0 {
        m += 1;
    }
    let a_high = a + f64::from(m);

    // Seeds from the positive-a integral (valid for any real b).
    let mut u_ap1 = hyperu_positive_a_integral(a_high + 1.0, b, x, mode)?; // U(a_high + 1)
    let mut u_a = hyperu_positive_a_integral(a_high, b, x, mode)?; // U(a_high)

    let mut ai = a_high;
    for _ in 0..m {
        let u_am1 = (2.0 * ai + x - b) * u_a - ai * (ai - b + 1.0) * u_ap1;
        u_ap1 = u_a;
        u_a = u_am1;
        ai -= 1.0;
    }

    if u_a.is_finite() {
        return Ok(u_a);
    }
    unsupported_hypergeometric_branch(
        "hyperu",
        mode,
        "integer-b hyperu recurrence produced a non-finite value",
    )
}

fn nonfinite_hyperu(mode: RuntimeMode) -> Result<f64, SpecialError> {
    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "hyperu",
            kind: SpecialErrorKind::NonFiniteInput,
            mode,
            detail: "hyperu parameters must be finite",
        });
    }
    Ok(f64::NAN)
}

fn hyperu_at_zero(a: f64, b: f64) -> f64 {
    if b < 1.0 {
        return gamma_real(1.0 - b) * reciprocal_gamma_real(a - b + 1.0);
    }
    f64::INFINITY
}

fn hyperu_terminating_polynomial(a: f64, b: f64, x: f64) -> f64 {
    let n = (-a) as usize;
    if n == 0 {
        return 1.0;
    }

    let alpha = b - 1.0;
    let mut laguerre_nm2 = 1.0;
    let mut laguerre_nm1 = 1.0 + alpha - x;
    if n == 1 {
        return -laguerre_nm1;
    }

    for k in 2..=n {
        let kf = k as f64;
        let laguerre_n =
            ((2.0 * kf - 1.0 + alpha - x) * laguerre_nm1 - (kf - 1.0 + alpha) * laguerre_nm2) / kf;
        laguerre_nm2 = laguerre_nm1;
        laguerre_nm1 = laguerre_n;
    }

    let factorial = (1..=n).fold(1.0, |acc, k| acc * k as f64);
    if n.is_multiple_of(2) {
        factorial * laguerre_nm1
    } else {
        -factorial * laguerre_nm1
    }
}

fn hyperu_positive_a_integral(
    a: f64,
    b: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    let (ln_gamma_a, sign_gamma_a) = ln_gamma_with_sign(a);
    if !ln_gamma_a.is_finite() || sign_gamma_a <= 0.0 {
        return unsupported_hypergeometric_branch(
            "hyperu",
            mode,
            "positive-a hyperu integral requires finite Gamma(a)",
        );
    }

    let peak_scale = (a - 1.0).max(b - 1.0);
    let peak_s = if peak_scale > 0.0 {
        (peak_scale / x).ln()
    } else {
        0.0
    };
    let lower_tail = (-44.0 / a.max(0.25)).clamp(-180.0, -18.0);
    let lower = lower_tail.min(peak_s - 20.0).max(-180.0);
    let upper_decay = (60.0 / x).ln();
    let upper = upper_decay.max(peak_s + 20.0).clamp(18.0, 180.0);

    let h = (upper - lower) / HYPERU_QUADRATURE_STEPS as f64;
    let mut sum = hyperu_integrand_log_s(a, b, x, ln_gamma_a, lower)
        + hyperu_integrand_log_s(a, b, x, ln_gamma_a, upper);

    for i in 1..HYPERU_QUADRATURE_STEPS {
        let s = lower + h * i as f64;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * hyperu_integrand_log_s(a, b, x, ln_gamma_a, s);
    }

    let value = sum * h / 3.0;
    if value.is_finite() {
        return Ok(value);
    }

    unsupported_hypergeometric_branch(
        "hyperu",
        mode,
        "hyperu integral quadrature overflowed before convergence was established",
    )
}

fn hyperu_integrand_log_s(a: f64, b: f64, x: f64, ln_gamma_a: f64, s: f64) -> f64 {
    let t = s.exp();
    if !t.is_finite() {
        return 0.0;
    }

    let log_value = -x * t + a * s + (b - a - 1.0) * t.ln_1p() - ln_gamma_a;
    if log_value < HYPERU_LOG_UNDERFLOW {
        return 0.0;
    }
    if log_value > HYPERU_LOG_OVERFLOW {
        return f64::INFINITY;
    }
    log_value.exp()
}

fn hyperu_connection_formula(
    a: f64,
    b: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<f64, SpecialError> {
    let sin_pi_b = (std::f64::consts::PI * b).sin();
    if sin_pi_b.abs() < 1.0e-12 {
        return unsupported_hypergeometric_branch(
            "hyperu",
            mode,
            "connection formula is singular for integer b",
        );
    }

    let m_ab = hyp1f1_scalar(a, b, x, mode)?;
    let shifted_a = a - b + 1.0;
    let shifted_b = 2.0 - b;
    let m_shifted = hyp1f1_scalar(shifted_a, shifted_b, x, mode)?;
    let term1 = m_ab * reciprocal_gamma_real(shifted_a) * reciprocal_gamma_real(b);
    let term2 =
        x.powf(1.0 - b) * m_shifted * reciprocal_gamma_real(a) * reciprocal_gamma_real(shifted_b);
    let value = std::f64::consts::PI * (term1 - term2) / sin_pi_b;

    if value.is_finite() {
        return Ok(value);
    }

    unsupported_hypergeometric_branch(
        "hyperu",
        mode,
        "connection formula produced a non-finite value",
    )
}

fn is_near_integer(x: f64) -> bool {
    x.is_finite() && (x - x.round()).abs() <= 1.0e-12
}

fn reciprocal_gamma_real(x: f64) -> f64 {
    if is_nonpositive_integer(x) {
        return 0.0;
    }
    let (ln_abs, sign) = ln_gamma_with_sign(x);
    if !ln_abs.is_finite() || sign == 0.0 {
        return 0.0;
    }
    sign * (-ln_abs).exp()
}

fn gamma_real(x: f64) -> f64 {
    if is_nonpositive_integer(x) {
        return f64::INFINITY;
    }
    let (ln_abs, sign) = ln_gamma_with_sign(x);
    if !ln_abs.is_finite() || sign == 0.0 {
        return f64::NAN;
    }
    sign * ln_abs.exp()
}

fn complex_nan() -> Complex64 {
    Complex64::new(f64::NAN, f64::NAN)
}

fn hyp1f1_complex_parameters(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && a.im == 0.0 && b.im == 0.0 {
        return Ok(Complex64::from_real(hyp1f1_scalar(a.re, b.re, z.re, mode)?));
    }

    if complex_is_zero(b) || complex_is_nonpositive_integer(b) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp1f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "b must not be zero or a negative integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) || complex_is_zero(a) {
        return Ok(Complex64::from_real(1.0));
    }

    // For large |z| the convergent Maclaurin series cancels catastrophically
    // (its largest partial term is ~e^|z| while the result can be far smaller),
    // giving errors up to ~1e69 by |z|~100. Switch to the large-|z| asymptotic
    // expansion, which is summed to optimal truncation and so stays accurate
    // across the whole plane — including the Stokes band near arg z = ±π/2 that
    // a two-term truncation gets wrong. Below |z|~24 the series is still the
    // better choice; in the overlap (|z|>=18) prefer the asymptotic only when
    // its self-estimated remainder is already negligible.
    let absz = z.abs();
    if absz >= 18.0 {
        let (aval, aest) = hyp1f1_asymptotic_complex(a, b, z);
        if aval.is_finite() && (absz >= 24.0 || aest < 1.0e-12) {
            return Ok(aval);
        }
    }

    hyp1f1_series_complex(a, b, z, mode)
}

fn hyp1f1_series_complex(
    a: Complex64,
    b: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 2_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 0..max_terms {
        let nf = n as f64;
        let scale = (a + Complex64::from_real(nf)) / ((b + Complex64::from_real(nf)) * (nf + 1.0));
        term = term * scale * z;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp1f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex z",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Large-|z| asymptotic expansion of M(a,b,z) for complex parameters
/// (A&S 13.5.1 / DLMF 13.7.2):
///
/// ```text
///   M(a,b,z) ~ Γ(b)/Γ(b−a) (−z)^{−a} Σ_n (a)_n (1+a−b)_n / n! (−z)^{−n}
///            + Γ(b)/Γ(a) e^z z^{a−b}   Σ_n (b−a)_n (1−a)_n / n! z^{−n}
/// ```
///
/// Each block is a divergent asymptotic series summed to its optimal (smallest
/// term) truncation; carrying the full superasymptotic sum — not just two
/// terms — is what keeps the result accurate through the anti-Stokes band near
/// arg z = ±π/2. The principal branch of `(−z)^{−a}` encodes the e^{±iπa}
/// Stokes sign automatically, so no explicit half-plane sign switch is needed.
///
/// Returns the value together with a relative-error estimate (the leading
/// optimally-truncated remainder, inflated by any cancellation between the two
/// blocks), so the caller can defer to the convergent series when |z| is not
/// yet large compared with the parameters.
fn hyp1f1_asymptotic_complex(a: Complex64, b: Complex64, z: Complex64) -> (Complex64, f64) {
    let one = Complex64::from_real(1.0);
    let neg_z = -z;
    let gamma_b = crate::gamma::complex_gammaln(b).exp();

    // Subdominant (algebraic) block.
    let (s1, min1) = asymptotic_block_sum(a, one + a - b, neg_z.recip());
    let coef1 = gamma_b * complex_recip_gamma(b - a) * neg_z.powc(-a);
    let term1 = coef1 * s1;

    // Dominant (exponential) block.
    let (s2, min2) = asymptotic_block_sum(b - a, one - a, z.recip());
    let coef2 = gamma_b * complex_recip_gamma(a) * z.exp() * z.powc(a - b);
    let term2 = coef2 * s2;

    let result = term1 + term2;
    let abs_err = coef1.abs() * min1 + coef2.abs() * min2;
    let est_rel = abs_err / (result.abs() + f64::MIN_POSITIVE);
    (result, est_rel)
}

/// Sum a divergent asymptotic series Σ_n (pa)_n (pb)_n / n! · w^n by term
/// recurrence, stopping at the smallest term (optimal truncation). Returns the
/// partial sum and the magnitude of the last term retained, which estimates the
/// leading remainder of that block.
fn asymptotic_block_sum(pa: Complex64, pb: Complex64, w: Complex64) -> (Complex64, f64) {
    const NMAX: usize = 120;
    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);
    let mut prev_mag = 1.0;
    let mut min_mag = 1.0;
    for n in 1..NMAX {
        let k = (n - 1) as f64;
        let factor = (pa + Complex64::from_real(k))
            * (pb + Complex64::from_real(k))
            * Complex64::from_real(1.0 / n as f64);
        term = term * factor * w;
        if !term.is_finite() {
            break;
        }
        let mag = term.abs();
        if mag > prev_mag && n > 2 {
            break;
        }
        prev_mag = mag;
        min_mag = mag;
        sum = sum + term;
    }
    (sum, min_mag)
}

/// Scalar 2F1(a, b; c; z) via series summation and transformations.
///
/// 2F1(a, b; c; z) = Σ_{n=0}^∞ (a)_n (b)_n z^n / ((c)_n n!)
fn hyp2f1_scalar(a: f64, b: f64, c: f64, z: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    // c must not be zero or a negative integer (unless a or b is a negative
    // integer with |a| or |b| < |c|)
    if c == 0.0 || (c < 0.0 && c == c.floor()) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "c must not be zero or a negative integer",
            });
        }
        return Ok(f64::NAN);
    }

    // Special cases
    if z == 0.0 {
        return Ok(1.0);
    }
    if a == 0.0 || b == 0.0 {
        return Ok(1.0);
    }

    let decision =
        select_hypergeometric_branch(HyperCaspProblem::hyp2f1(a, b, c, z, 1.0e-14), mode)?;
    match decision.branch {
        HypergeometricBranch::DirectSeries | HypergeometricBranch::TerminatingPolynomial => {
            hyp2f1_series(a, b, c, z)
        }
        HypergeometricBranch::GaussSummation => {
            let cab = c - a - b;
            Ok(gamma_ratio_for_hyp2f1(c, cab, c - a, c - b))
        }
        HypergeometricBranch::PfaffTransform => {
            let z_new = z / (z - 1.0);
            let factor = (1.0 - z).powf(-a);
            let inner = hyp2f1_series(a, c - b, c, z_new)?;
            Ok(factor * inner)
        }
        HypergeometricBranch::LinearFractionalIdentity => {
            // Euler transformation: 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z).
            // Selection guarantees c-a or c-b is a nonpositive integer, so the
            // transformed series terminates exactly (hyp2f1_series stops when a
            // numerator parameter reaches zero), and c-a-b is an integer so the
            // (1-z) power is real-valued for z > 1. The old c==a / c==b identity
            // is recovered here: the inner 2F1 collapses to 1 and the exponent
            // becomes -b or -a respectively.
            let exponent = (c - a - b).round() as i32;
            let factor = (1.0 - z).powi(exponent);
            let inner = hyp2f1_series(c - a, c - b, c, z)?;
            Ok(factor * inner)
        }
        HypergeometricBranch::DivergentAtUnitArgument => divergent_hyp2f1_at_unit_argument(mode),
        HypergeometricBranch::ParameterGuard => {
            guarded_hypergeometric_parameter("hyp2f1", mode, decision.reason)
        }
        HypergeometricBranch::RealBranchCutInfinity => {
            if mode == RuntimeMode::Hardened {
                unsupported_hypergeometric_branch("hyp2f1", mode, decision.reason)
            } else {
                Ok(f64::INFINITY)
            }
        }
        HypergeometricBranch::UnsupportedAnalyticContinuation => {
            unsupported_hypergeometric_branch("hyp2f1", mode, decision.reason)
        }
        HypergeometricBranch::KummerTransform | HypergeometricBranch::AsymptoticExpansion => {
            unsupported_hypergeometric_branch(
                "hyp2f1",
                mode,
                "selected branch is not valid for 2F1",
            )
        }
    }
}

/// Direct series summation for 2F1.
///
/// `max_terms` is sized for |z| up to ~0.99: the term ratio approaches z
/// for large n, so the convergence criterion needs roughly
/// log(EPSILON)/log(|z|) ≈ 3.7 × 10³ iterations at z = 0.99. Smaller |z|
/// reaches the early-out within tens of iterations. For |z| close to 1
/// from above scipy uses the (1−z)-transform with gamma ratios, which
/// is not implemented here — frankenscipy-3zzep.
fn hyp2f1_series(a: f64, b: f64, c: f64, z: f64) -> Result<f64, SpecialError> {
    let max_terms = 5000;
    let eps = f64::EPSILON;

    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 0..max_terms {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) * z / ((c + nf) * (nf + 1.0));

        if !term.is_finite() {
            break;
        }

        sum += term;

        if term == 0.0 {
            return Ok(sum);
        }

        if term.abs() < eps * sum.abs() {
            return Ok(sum);
        }
    }

    Ok(f64::NAN)
}

fn is_integer(x: f64) -> bool {
    x.is_finite() && x == x.trunc() && x >= f64::from(i32::MIN) && x <= f64::from(i32::MAX)
}

fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && is_integer(x)
}

fn hyp2f1_complex_parameters(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    if z.im == 0.0 && a.im == 0.0 && b.im == 0.0 && c.im == 0.0 {
        return Ok(Complex64::from_real(hyp2f1_scalar(
            a.re, b.re, c.re, z.re, mode,
        )?));
    }

    if complex_is_zero(c) || complex_is_nonpositive_integer(c) {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "c must not be zero or a negative integer",
            });
        }
        return Ok(complex_nan());
    }

    if complex_is_zero(z) || complex_is_zero(a) || complex_is_zero(b) {
        return Ok(Complex64::from_real(1.0));
    }

    // Exact linear reductions when a Gauss parameter cancels the denominator:
    // 2F1(a,b;b;z) = (1-z)^{-a} and 2F1(a,b;a;z) = (1-z)^{-b}, valid for all z
    // (principal branch). These hold even outside the disk and cover the
    // doubly-degenerate corner where the generic connection weights are singular
    // (e.g. 2F1(1,2;2;z)).
    let one = Complex64::from_real(1.0);
    if complex_is_zero(c - b) {
        return Ok((one - z).powc(-a));
    }
    if complex_is_zero(c - a) {
        return Ok((one - z).powc(-b));
    }

    if z.abs() < 1.0 {
        return hyp2f1_series_complex(a, b, c, z, mode);
    }

    // Pfaff transformation 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1)).
    // The mapped argument w = z/(z-1) satisfies |w| < 1 exactly when Re(z) < 1/2,
    // so this extends the convergent complex series across the entire left
    // half-plane Re(z) < 1/2 (including |z| >= 1). (1-z)^{-a} uses the principal
    // branch, matching the standard analytic continuation. Re(z) >= 1/2 outside
    // the unit disk still needs the z -> 1/z connection formula — that remains a
    // documented gap (frankenscipy-f69ch).
    if z.re < 0.5 {
        let one = Complex64::from_real(1.0);
        let factor = (one - z).powc(-a);
        let w = z / (z - one);
        let inner = hyp2f1_series_complex(a, c - b, c, w, mode)?;
        let value = factor * inner;
        if value.is_finite() {
            return Ok(value);
        }
    }

    // z -> 1/z connection (DLMF 15.8.2) reaches the right half-plane Re(z) >= 1/2
    // outside the disk. The principal (-z)^{-a} is the correct analytic branch
    // only off the real axis — on real z > 1 the real-scalar path already
    // applies. When a - b is an exact integer the generic Γ(b-a)/Γ(a-b) weights
    // hit poles and the DLMF 15.8.8 logarithmic limit form is required; we reach
    // that limit by symmetric parameter perturbation. Closes frankenscipy-f69ch
    // and the degenerate corner frankenscipy-31a0c.
    let ab = a - b;
    let ab_is_integer = ab.im.abs() < 1.0e-12 && (ab.re - ab.re.round()).abs() < 1.0e-12;
    if z.im != 0.0 {
        let value = if ab_is_integer {
            hyp2f1_inv_z_connection_degenerate(a, b, c, z, mode)?
        } else {
            hyp2f1_inv_z_connection(a, b, c, z, mode)?
        };
        if value.is_finite() {
            return Ok(value);
        }
    }

    if mode == RuntimeMode::Hardened {
        return Err(SpecialError {
            function: "hyp2f1",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "complex-valued hyp2f1 outside |z| < 1 with Re(z) >= 1/2 \
                     failed to converge",
        });
    }

    Ok(complex_nan())
}

/// Linear z -> 1/z connection for 2F1 outside the unit disk (DLMF 15.8.2),
/// valid for a - b not an integer:
///   2F1(a,b;c;z) = w1 (-z)^{-a} 2F1(a, a-c+1; a-b+1; 1/z)
///                + w2 (-z)^{-b} 2F1(b, b-c+1; b-a+1; 1/z),
/// with w1 = Γ(c)Γ(b-a)/(Γ(b)Γ(c-a)), w2 = Γ(c)Γ(a-b)/(Γ(a)Γ(c-b)).
fn hyp2f1_inv_z_connection(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    use crate::gamma::complex_gammaln;

    let one = Complex64::from_real(1.0);
    let neg_z = -z;
    let inv_z = z.recip();

    let w1 = (complex_gammaln(c) + complex_gammaln(b - a)
        - complex_gammaln(b)
        - complex_gammaln(c - a))
    .exp();
    let w2 = (complex_gammaln(c) + complex_gammaln(a - b)
        - complex_gammaln(a)
        - complex_gammaln(c - b))
    .exp();

    let term1 = w1 * neg_z.powc(-a) * hyp2f1_series_complex(a, a - c + one, a - b + one, inv_z, mode)?;
    let term2 = w2 * neg_z.powc(-b) * hyp2f1_series_complex(b, b - c + one, b - a + one, inv_z, mode)?;

    Ok(term1 + term2)
}

/// Reciprocal Γ(z) for complex z, returning 0 at the nonpositive-integer poles
/// of Γ (where 1/Γ vanishes) rather than the NaN that `complex_gammaln` yields
/// there. Used for the *denominator* Γ's of the connection weights so a
/// further-degenerate c-a or c-b nonpositive integer collapses its term cleanly.
fn complex_recip_gamma(z: Complex64) -> Complex64 {
    if complex_is_zero(z) || complex_is_nonpositive_integer(z) {
        return Complex64::from_real(0.0);
    }
    (-crate::gamma::complex_gammaln(z)).exp()
}

/// Rising factorial (Pochhammer) (x)_k = x(x+1)...(x+k-1) for complex x.
fn complex_pochhammer(x: Complex64, k: u64) -> Complex64 {
    let mut acc = Complex64::from_real(1.0);
    for j in 0..k {
        acc = acc * (x + Complex64::from_real(j as f64));
    }
    acc
}

/// Degenerate a - b integer case of the z -> 1/z connection: the DLMF 15.8.8
/// logarithmic limit form, valid for |z| > 1 with the generic two-term weights
/// Γ(b-a)/Γ(a-b) replaced by the finite log limit. With b - a = m a nonnegative
/// integer (we order the parameters so this holds, 2F1 being symmetric in a, b),
///
///   F(a, a+m; c; z) = (-z)^{-a}/Γ(a+m) Σ_{k=0}^{m-1} (a)_k (m-k-1)!/(k! Γ(c-a-k)) z^{-k}
///                   + (-z)^{-a}/Γ(a)   Σ_{k=0}^{∞}  (a+m)_k (-1)^k z^{-k-m}
///                                         / (k! (k+m)! Γ(c-a-k-m))
///                                         × [ln(-z) + ψ(k+1) + ψ(k+m+1)
///                                            - ψ(a+k+m) - ψ(c-a-k-m)]
///
/// where F is the regularized 2F1/Γ(c). Using reciprocal-Γ for the denominator
/// Γ's keeps the infinite-sum terms finite (1/Γ → 0 at its poles), and a running
/// term recurrence avoids overflow of (k+m)! at large k. When c - a is *also* an
/// integer the sum's 1/Γ(c-a-k-m) hits poles where ψ(c-a-k-m) → ∞: the term is
/// then the finite 0·∞ limit Cp_k·(-1)^p p! (the fully logarithmic DLMF
/// 15.8.9/15.8.11 forms), so this one path also covers the triple-degenerate
/// corner. The c == a / c == b reductions are taken earlier in
/// `hyp2f1_complex_parameters`.
fn hyp2f1_inv_z_connection_degenerate(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    use crate::gamma::{complex_digamma_scalar, complex_gammaln, factorial};

    // Order parameters so the second exceeds the first by m >= 0.
    let (pa, pb) = if (a - b).re <= 0.0 { (a, b) } else { (b, a) };
    let m_f = (pb - pa).re.round();
    let m = m_f as u64;

    let neg_z = -z;
    let inv_z = z.recip();
    let ln_neg_z = neg_z.ln();

    // Finite sum, Σ_{k=0}^{m-1}, divided by Γ(a+m).
    let mut s1 = Complex64::from_real(0.0);
    let mut zinv_k = Complex64::from_real(1.0); // z^{-k}
    for k in 0..m {
        let kf = k as f64;
        let poch = complex_pochhammer(pa, k); // (a)_k
        let fact_ratio = factorial(m - k - 1) / factorial(k); // (m-k-1)!/k!
        let rg = complex_recip_gamma(c - pa - Complex64::from_real(kf)); // 1/Γ(c-a-k)
        s1 = s1 + poch * Complex64::from_real(fact_ratio) * rg * zinv_k;
        zinv_k = zinv_k * inv_z;
    }
    s1 = s1 * complex_recip_gamma(pa + Complex64::from_real(m_f)); // /Γ(a+m)

    // Infinite log sum, Σ_{k=0}^{∞}, divided by Γ(a). Track the regular core
    //   Cp_k = (a+m)_k (-1)^k z^{-k-m} / (k! (k+m)!)
    // (advanced by Cp_{k+1}/Cp_k = (a+m+k) / ((k+1)(k+1+m)) · (-1/z)) separately
    // from the 1/Γ(c-a-k-m) factor. For non-pole indices the term is
    //   Cp_k · rgamma(c-a-k-m) · [ln(-z) + ψ(k+1) + ψ(k+m+1) - ψ(a+k+m)
    //                             - ψ(c-a-k-m)].
    // When c - a is also an integer, x = c-a-k-m hits the nonpositive integer -p
    // for some k: there rgamma → 0 and ψ(x) → ∞ and the term is the finite 0·∞
    // limit  Cp_k · (-1)^p p!  (DLMF 15.8.9/15.8.11), since
    // lim_{x→-p} rgamma(x) ψ(x) = (-1)^{p+1} p!. This single path therefore
    // covers both the generic (c-a non-integer) and triple-degenerate corners.
    let tol = 1.0e-15;
    let max_terms = 4_000u64;
    let mut cp_core = Complex64::from_real(1.0 / factorial(m))
        * inv_z.powc(Complex64::from_real(m_f)); // Cp_0 = z^{-m}/m!
    let mut s2 = Complex64::from_real(0.0);
    let mut converged = false;
    for k in 0..max_terms {
        let kf = k as f64;
        if !cp_core.is_finite() {
            break;
        }
        let x = c - pa - Complex64::from_real(kf + m_f); // c - a - k - m
        let xr = x.re.round();
        let is_pole = x.im.abs() < 1.0e-9 && xr <= 0.0 && (x.re - xr).abs() < 1.0e-9;
        let term = if is_pole {
            let p = (-xr) as u64;
            let sign = if p.is_multiple_of(2) { 1.0 } else { -1.0 };
            cp_core * Complex64::from_real(sign * factorial(p)) // Cp_k·(-1)^p p!
        } else {
            let bracket = ln_neg_z
                + complex_digamma_scalar(Complex64::from_real(kf + 1.0))
                + complex_digamma_scalar(Complex64::from_real(kf + m_f + 1.0))
                - complex_digamma_scalar(pa + Complex64::from_real(kf + m_f))
                - complex_digamma_scalar(x);
            cp_core * complex_recip_gamma(x) * bracket
        };
        if !term.is_finite() {
            break;
        }
        s2 = s2 + term;
        if k > m + 2 && complex_series_converged(term, s2, tol) {
            converged = true;
            break;
        }
        // advance Cp core to k+1
        let num = pa + Complex64::from_real(m_f + kf);
        let den = (kf + 1.0) * (kf + 1.0 + m_f);
        cp_core = cp_core * num / Complex64::from_real(-den) * inv_z;
    }
    if !converged {
        if mode == RuntimeMode::Hardened {
            return Err(SpecialError {
                function: "hyp2f1",
                kind: SpecialErrorKind::OverflowRisk,
                mode,
                detail: "degenerate z -> 1/z log series failed to converge",
            });
        }
        return Ok(complex_nan());
    }
    s2 = s2 * complex_recip_gamma(pa); // /Γ(a)

    // F = (-z)^{-a} (s1 + s2);  2F1 = Γ(c) · F.
    let pref = neg_z.powc(-pa);
    let gamma_c = complex_gammaln(c).exp();
    Ok(gamma_c * pref * (s1 + s2))
}

fn hyp2f1_series_complex(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    let max_terms = 2_000;
    let tol = 1.0e-14;

    let mut sum = Complex64::from_real(1.0);
    let mut term = Complex64::from_real(1.0);

    for n in 0..max_terms {
        let nf = n as f64;
        let scale = ((a + Complex64::from_real(nf)) * (b + Complex64::from_real(nf)))
            / ((c + Complex64::from_real(nf)) * (nf + 1.0));
        term = term * scale * z;

        if !term.is_finite() || !sum.is_finite() {
            if mode == RuntimeMode::Hardened {
                return Err(SpecialError {
                    function: "hyp2f1",
                    kind: SpecialErrorKind::OverflowRisk,
                    mode,
                    detail: "series evaluation overflowed for complex z",
                });
            }
            return Ok(complex_nan());
        }

        sum = sum + term;
        if complex_series_converged(term, sum, tol) {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Compute Gamma(c)*Gamma(c-a-b) / (Gamma(c-a)*Gamma(c-b)) for the Gauss sum.
fn gamma_ratio_for_hyp2f1(c: f64, cab: f64, ca: f64, cb: f64) -> f64 {
    let (ln_c, sign_c) = ln_gamma_with_sign(c);
    let (ln_cab, sign_cab) = ln_gamma_with_sign(cab);
    let (ln_ca, sign_ca) = ln_gamma_with_sign(ca);
    let (ln_cb, sign_cb) = ln_gamma_with_sign(cb);

    if !ln_c.is_finite() || !ln_cab.is_finite() || sign_c == 0.0 || sign_cab == 0.0 {
        return f64::NAN;
    }
    if !ln_ca.is_finite() || !ln_cb.is_finite() || sign_ca == 0.0 || sign_cb == 0.0 {
        return 0.0;
    }

    let sign = sign_c * sign_cab * sign_ca * sign_cb;
    sign * (ln_c + ln_cab - ln_ca - ln_cb).exp()
}

/// Computes (ln(|Gamma(x)|), sign(Gamma(x))) via Lanczos approximation.
fn ln_gamma_with_sign(x: f64) -> (f64, f64) {
    if x <= 0.0 && x == x.floor() {
        return (f64::INFINITY, 0.0);
    }
    // Lanczos coefficients (g=7, n=9)
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        let sign = if sin_pi_x < 0.0 { -1.0 } else { 1.0 };
        let (ln_1_minus_x, _) = ln_gamma_with_sign(1.0 - x);
        let ln_abs = (pi / sin_pi_x.abs()).ln() - ln_1_minus_x;
        return (ln_abs, sign);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    let val = 0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln();
    (val, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn hyp1f1_nonpositive_integer_a_and_b_pole_match_scipy() {
        // frankenscipy-ztn7f: a nonpositive integer terminates 1F1 into an exact
        // degree-|a| polynomial. Previously the z>200 asymptotic (carrying a
        // 1/Γ(a)=0 factor) returned ~0, and a negative-integer b was rejected to
        // NaN even when a terminated the series first.
        // (a, b, z, scipy hyp1f1) from scipy.special 1.17.1.
        let poly: [(f64, f64, f64, f64); 7] = [
            // nonpos-int a, large z: the polynomial, not the ~0 asymptotic.
            (-1.0, 0.5, 300.0, -599.0),
            (-1.0, 1.0, 300.0, -299.0),
            (-1.0, 3.0, 300.0, -99.0),
            (-5.0, 1.0, 300.0, -18607051499.0),
            // nonpos-int a AND negative-integer b, |a|<=|b|: terminates before pole.
            (-1.0, -2.0, 5.0, 3.5),
            (-2.0, -2.0, 5.0, 18.5),
            (-2.0, -3.0, 5.0, 8.5),
        ];
        for (a, b, z, want) in poly {
            let got = hyp1f1_scalar(a, b, z, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "hyp1f1({a},{b},{z}) = {got}, scipy {want}"
            );
        }
        // Genuine poles → +inf (matching scipy): non-terminating a with
        // negative-integer b, and nonpos-int a with |a|>|b|.
        let poles: [(f64, f64, f64); 5] = [
            (0.5, -2.0, 5.0),   // non-integer a, b=-2
            (2.0, -2.0, 0.5),   // positive-int a, b=-2 (no termination)
            (-2.5, 0.0, 2.0),   // b=0
            (-3.0, -2.0, 0.5),  // |a|>|b|
            (-2.0, -1.0, 0.0),  // |a|>|b|, even at z=0
        ];
        for (a, b, z) in poles {
            let got = hyp1f1_scalar(a, b, z, RuntimeMode::Strict).unwrap();
            assert!(got == f64::INFINITY, "hyp1f1({a},{b},{z}) = {got}, expected +inf");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn hyperu_large_x_noninteger_b_asymptotic_matches_scipy() {
        // frankenscipy-w10iq: for a<0 non-integer and b non-integer, the
        // Γ-weighted connection formula sums two 1F1 terms that each grow like
        // e^x while U is recessive (~x^{-a}); they cancel catastrophically for
        // large x (U(-0.5,-1.5,50) was -3.16e8 vs scipy 7.21, and -5.3e74 at
        // x=200). The DLMF 13.7.3 asymptotic is cancellation-free.
        // (a, b, x, scipy hyperu) from scipy.special 1.17.1.
        let cases: [(f64, f64, f64, f64); 7] = [
            (-0.5, -1.5, 50.0, 7.210447801115636),
            (-0.5, -1.5, 200.0, 14.212583747872978),
            (-0.5, -0.5, 30.0, 5.567061595067197),
            (-1.5, -1.5, 40.0, 262.5862012751575),
            (-2.5, -2.5, 40.0, 10775.75401572671),
            (-0.5, -2.5, 20.0, 4.792553644800431),
            (-0.5, -1.5, 12.0, 3.7371524496535997), // connection-formula side of the crossover
        ];
        for (a, b, x, want) in cases {
            let got = hyperu_scalar(a, b, x, RuntimeMode::Strict).unwrap();
            assert!(
                (got - want).abs() <= 1e-7 * want.abs(),
                "hyperu({a},{b},{x}) = {got}, scipy {want}"
            );
        }
    }

    fn scalar(v: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(v)
    }

    fn complex(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn complex_vec(values: Vec<Complex64>) -> SpecialTensor {
        SpecialTensor::ComplexVec(values)
    }

    fn get_scalar(r: &SpecialResult) -> Option<f64> {
        match r.as_ref() {
            Ok(SpecialTensor::RealScalar(v)) => Some(*v),
            _ => None,
        }
    }

    fn get_complex_scalar(r: &SpecialResult) -> Option<Complex64> {
        match r.as_ref() {
            Ok(SpecialTensor::ComplexScalar(v)) => Some(*v),
            _ => None,
        }
    }

    fn get_real_vec(r: &SpecialResult) -> Option<&[f64]> {
        match r.as_ref() {
            Ok(SpecialTensor::RealVec(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    fn get_complex_vec(r: &SpecialResult) -> Option<&[Complex64]> {
        match r.as_ref() {
            Ok(SpecialTensor::ComplexVec(values)) => Some(values.as_slice()),
            _ => None,
        }
    }

    fn error_kind(r: &SpecialResult) -> Option<SpecialErrorKind> {
        match r {
            Err(err) => Some(err.kind),
            Ok(_) => None,
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "expected {}+{}i, got {}+{}i (|delta|={delta})",
            expected.re,
            expected.im,
            actual.re,
            actual.im
        );
    }

    fn select_casp_for_test(problem: HyperCaspProblem) -> HyperCaspDecision {
        let result = select_hypergeometric_branch(problem, RuntimeMode::Strict);
        assert!(
            result.is_ok(),
            "unexpected CASP selection error: {result:?}"
        );

        match result {
            Ok(decision) => decision,
            Err(_) => HyperCaspDecision {
                branch: HypergeometricBranch::UnsupportedAnalyticContinuation,
                precision_target: problem.precision_target,
                max_terms: 0,
                parameter_stability_margin: problem.parameter_stability_margin,
                fallback_chain: HYPER_UNSUPPORTED_CHAIN,
                reason: "unexpected CASP selection error",
            },
        }
    }

    #[test]
    fn hyper_casp_selects_direct_series_for_moderate_hyp0f1() {
        let problem = HyperCaspProblem::hyp0f1(2.0, 12.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::DirectSeries);
        assert_eq!(decision.max_terms, 300);
        assert_eq!(decision.fallback_chain, HYP0F1_DIRECT_CHAIN);
        assert!(decision.parameter_stability_margin > decision.precision_target);
    }

    #[test]
    fn hyper_casp_selects_asymptotic_for_large_hyp0f1_argument() {
        let problem = HyperCaspProblem::hyp0f1(2.0, 75.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::AsymptoticExpansion);
        assert_eq!(decision.max_terms, 10);
        assert_eq!(decision.fallback_chain, HYP0F1_ASYMPTOTIC_CHAIN);
    }

    #[test]
    fn hyper_casp_guards_hyp0f1_lower_parameter_near_pole() {
        let problem = HyperCaspProblem::hyp0f1(1.0e-16, 0.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::ParameterGuard);
        assert!(decision.parameter_stability_margin <= decision.precision_target);
    }

    #[test]
    fn hyper_casp_selects_direct_series_for_moderate_hyp1f1() {
        let problem = HyperCaspProblem::hyp1f1(1.0, 2.0, 0.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::DirectSeries);
        assert_eq!(decision.max_terms, 500);
        assert_eq!(decision.fallback_chain, HYP1F1_DIRECT_CHAIN);
        assert!(decision.parameter_stability_margin > decision.precision_target);
    }

    #[test]
    fn hyper_casp_selects_kummer_for_large_negative_hyp1f1() {
        let problem = HyperCaspProblem::hyp1f1(1.0, 2.0, -30.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::KummerTransform);
        assert_eq!(decision.fallback_chain, HYP1F1_KUMMER_CHAIN);
    }

    #[test]
    fn hyper_casp_guards_lower_parameter_near_pole() {
        let problem = HyperCaspProblem::hyp1f1(1.0, 1.0e-16, 0.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::ParameterGuard);
        assert!(decision.parameter_stability_margin <= decision.precision_target);
    }

    #[test]
    fn hyper_casp_selects_terminating_polynomial_for_hyp2f1() {
        let problem = HyperCaspProblem::hyp2f1(-2.0, 1.0, 1.5, 2.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::TerminatingPolynomial);
        assert_eq!(decision.fallback_chain, HYP2F1_TERMINATING_CHAIN);
    }

    #[test]
    fn hyper_casp_selects_direct_series_for_hyp2f1_inside_unit_disk() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 1.0, 2.0, 0.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::DirectSeries);
    }

    #[test]
    fn hyper_casp_selects_gauss_sum_at_unit_argument() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 1.0, 3.0, 1.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::GaussSummation);
        assert_eq!(decision.max_terms, 0);
    }

    #[test]
    fn hyper_casp_selects_pfaff_for_negative_outside_unit_disk() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 1.0, 2.0, -2.0, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::PfaffTransform);
        assert_eq!(decision.fallback_chain, HYP2F1_PFAFF_CHAIN);
    }

    #[test]
    fn hyper_casp_selects_identity_for_z_greater_than_one_reduction() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 1.0, 1.0, 1.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(
            decision.branch,
            HypergeometricBranch::LinearFractionalIdentity
        );
    }

    #[test]
    fn hyper_casp_selects_real_branch_cut_for_generic_z_greater_than_one() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 1.5, 1.0e-14);
        let decision = select_casp_for_test(problem);

        assert_eq!(decision.branch, HypergeometricBranch::RealBranchCutInfinity);
        assert_eq!(decision.max_terms, 0);
    }

    #[test]
    fn hyper_casp_hardened_rejects_nonfinite_inputs() {
        let problem = HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, f64::NAN, 1.0e-14);
        let result = select_hypergeometric_branch(problem, RuntimeMode::Hardened);
        assert!(
            result.is_err(),
            "non-finite CASP input should fail in hardened mode"
        );
        let kind = match result {
            Err(err) => err.kind,
            Ok(_) => SpecialErrorKind::DomainError,
        };

        assert_eq!(kind, SpecialErrorKind::NonFiniteInput);
    }

    #[test]
    fn hyp0f1_hardened_errors_when_lower_parameter_near_pole() {
        let result = hyp0f1(&scalar(1.0e-16), &scalar(0.5), RuntimeMode::Hardened);
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::PoleInput));
    }

    // ── hyp0f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp0f1_complex_parameter_zero_z_is_one() {
        let result = hyp0f1(
            &complex(1.25, -0.5),
            &complex(0.0, 0.0),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(1.0),
            1.0e-14,
        );
    }

    #[test]
    fn hyp0f1_complex_real_parameter_matches_real_path() {
        let complex_result = hyp0f1(&complex(1.5, 0.0), &scalar(0.25), RuntimeMode::Strict);
        let real_result = hyp0f1(&scalar(1.5), &scalar(0.25), RuntimeMode::Strict);
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp0f1_complex_parameter_preserves_parameter_conjugation_symmetry() {
        let b = Complex64::new(1.25, 0.5);
        let z = Complex64::new(0.2, -0.1);
        let lhs = hyp0f1(
            &complex(b.re, b.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        let rhs = hyp0f1(
            &complex(b.re, -b.im),
            &complex(z.re, -z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&rhs).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            get_complex_scalar(&lhs)
                .map_or(Complex64::new(f64::NAN, f64::NAN), |value| value.conj()),
            1.0e-12,
        );
    }

    #[test]
    fn hyp0f1_complex_parameter_pole_errors_hardened() {
        let result = hyp0f1(&complex(0.0, 0.0), &scalar(1.0), RuntimeMode::Hardened);
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::PoleInput));
    }

    #[test]
    fn hyp0f1_parameter_vector_broadcast_matches_scalar_evaluations() {
        let params = complex_vec(vec![Complex64::new(1.25, 0.5), Complex64::new(1.25, -0.5)]);
        let z = complex(0.2, -0.1);
        let result = hyp0f1(&params, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            hyp0f1_complex_scalar(
                Complex64::new(1.25, 0.5),
                Complex64::new(0.2, -0.1),
                RuntimeMode::Strict,
            )
            .unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            1.0e-12,
        );
        assert_complex_close(
            values[1],
            hyp0f1_complex_scalar(
                Complex64::new(1.25, -0.5),
                Complex64::new(0.2, -0.1),
                RuntimeMode::Strict,
            )
            .unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            1.0e-12,
        );
    }

    // ── hyp1f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp1f1_zero_z_is_one() {
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_zero_a_is_one() {
        let r = hyp1f1(
            &scalar(0.0),
            &scalar(2.0),
            &scalar(5.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp1f1_a_equals_b_is_exp() {
        // 1F1(a; a; z) = e^z
        let z = 1.5;
        let r = hyp1f1(&scalar(3.0), &scalar(3.0), &scalar(z), RuntimeMode::Strict);
        let expected = z.exp();
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "1F1(a;a;z) should be e^z: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp1f1_known_value() {
        // 1F1(1; 1; 1) = e
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - std::f64::consts::E).abs() < 1e-10,
            "1F1(1;1;1) should be e"
        );
    }

    #[test]
    fn hyp1f1_negative_z() {
        // 1F1(1; 2; -1) = (1 - e^(-1)) * 2 / 1 ... known value ≈ 0.6321205588
        // Actually: 1F1(1; 2; -1) = (e^(-1) - 1) / (-1) * 2/1...
        // Let's just check it's finite and reasonable
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(-1.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_finite(), "hyp1f1 should be finite for negative z");
        assert!(val > 0.0, "hyp1f1(1,2,-1) should be positive");
    }

    #[test]
    fn hyp1f1_large_negative_z_kummer() {
        // For large negative z, Kummer's transformation is used
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(-30.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_finite(), "should handle large negative z");
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn hyp1f1_large_positive_z_strict_uses_asymptotic() {
        // Formerly returned NaN (500-term direct-series cap); now the DLMF
        // 13.7.2 asymptotic computes it. scipy.special.hyp1f1(1,2,500).
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(500.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).expect("finite asymptotic value");
        let expected = 2.8071844357056744e214;
        assert!(((val - expected) / expected).abs() < 1e-12, "got {val:e}");
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn hyp1f1_large_positive_z_hardened_uses_asymptotic() {
        // Hardened mode no longer fails closed here — the value is computable.
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(500.0),
            RuntimeMode::Hardened,
        );
        let val = get_scalar(&r).expect("finite asymptotic value");
        let expected = 2.8071844357056744e214;
        assert!(((val - expected) / expected).abs() < 1e-12, "got {val:e}");
    }

    #[test]
    fn hyp1f1_b_zero_returns_inf_strict() {
        // b=0 is a pole of 1/Γ(b); scipy.special.hyp1f1 returns +inf there.
        // (Previously this returned NaN — a non-parity behavior. frankenscipy-ztn7f.)
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(0.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val == f64::INFINITY, "b=0 should return +inf in strict mode, got {val}");
    }

    #[test]
    fn hyp1f1_b_zero_errors_hardened() {
        let r = hyp1f1(
            &scalar(1.0),
            &scalar(0.0),
            &scalar(1.0),
            RuntimeMode::Hardened,
        );
        assert!(r.is_err(), "b=0 should error in hardened mode");
    }

    #[test]
    fn hyp1f1_vectorized() {
        let z_vec = SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]);
        let r = hyp1f1(&scalar(1.0), &scalar(1.0), &z_vec, RuntimeMode::Strict);
        let values = get_real_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-14); // e^0
        assert!((values[1] - std::f64::consts::E).abs() < 1e-10); // e^1
        assert!((values[2] - std::f64::consts::E.powi(2)).abs() < 1e-10); // e^2
    }

    #[test]
    fn hyp1f1_complex_matches_exp_identity() {
        let z = Complex64::new(0.5, 0.75);
        let r = hyp1f1(
            &scalar(2.0),
            &scalar(2.0),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            z.exp(),
            1.0e-10,
        );
    }

    #[test]
    fn hyp1f1_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.3, 0.4);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let r = hyp1f1(&scalar(1.0), &scalar(1.0), &input, RuntimeMode::Strict);
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[1], values[0].conj(), 1.0e-10);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn hyp0f1_complex_large_z_matches_mpmath() {
        // Golden values from mpmath.hyp0f1 (dps=30) on the oscillatory (J-type)
        // directions where the ascending series cancels catastrophically: the
        // negative-real cases used to return rel ~2.4 (z=-400) to ~1e69
        // (z=-10000). scipy.hyp0f1 accepts real b + complex z.
        let cases: &[(f64, (f64, f64), f64, f64)] = &[
            (2.0, (-400.0, 0.0), 0.00630191590187925, 0.0),
            (1.5, (0.0, -900.0), -1.5462761722273576e16, 1.5930749561190496e16),
            (2.5, (-2500.0, 300.0), -0.05794675493376355, 0.013103138447106267),
            (0.5, (-10000.0, 0.0), 0.48718767500700594, 0.0),
            (3.0, (-625.0, 625.0), -159626.04856509008, 867341.6324720286),
            (1.0, (-50.0, 50.0), -43.843811768703894, 42.19916419322264),
        ];
        for &(b, z, re, im) in cases {
            let got = hyp0f1_complex_scalar(
                Complex64::from_real(b),
                Complex64::new(z.0, z.1),
                RuntimeMode::Strict,
            )
            .unwrap();
            let want = Complex64::new(re, im);
            let rel = (got - want).abs() / (want.abs() + 1e-300);
            assert!(
                rel < 1.0e-9,
                "hyp0f1({b},{z:?}) = {got:?}, want {want:?}, rel={rel:.3e}"
            );
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn hyp1f1_complex_large_z_matches_mpmath() {
        // Golden values from mpmath.hyp1f1 (dps=30). These cover the regime
        // where the convergent Maclaurin series cancels catastrophically:
        // the negative-real-axis case used to return ~1e69, and the
        // arg z = ±pi/2 anti-Stokes band used to fail at rel ~1.8.
        let cases: &[((f64, f64), (f64, f64), (f64, f64), f64, f64)] = &[
            ((1.0, 0.5), (2.0, -0.3), (-99.0, 14.0), -0.011969417454155048, -0.007406056247135601),
            ((0.5, 0.0), (1.5, 0.0), (0.0, 60.0), 0.07842759028142188, 0.08885735047389137),
            ((0.5, 0.0), (1.5, 0.0), (-40.0, 40.0), 0.10886111845882578, 0.04509175168074972),
            ((1.2, 0.0), (3.4, 0.0), (0.0, -80.0), -0.004023506086242592, -0.013660756513785781),
            ((1.0, 1.0), (3.0, 0.5), (50.0, 30.0), 3.9816355065269386e18, 1.1345975153736888e18),
            ((0.5, 0.0), (10.0, 0.0), (-70.0, 20.0), 0.335215644799161, 0.04195012139845268),
            ((2.0, 0.0), (5.0, 0.0), (80.0, 0.0), 2.4997729898337908e30, 0.0),
        ];
        for &(a, b, z, re, im) in cases {
            let a = Complex64::new(a.0, a.1);
            let b = Complex64::new(b.0, b.1);
            let z = Complex64::new(z.0, z.1);
            let got = hyp1f1_complex_parameters(a, b, z, RuntimeMode::Strict).unwrap();
            let want = Complex64::new(re, im);
            let rel = (got - want).abs() / (want.abs() + 1.0e-300);
            assert!(
                rel < 1.0e-9,
                "hyp1f1({a:?},{b:?},{z:?}) = {got:?}, want {want:?}, rel={rel:.3e}"
            );
        }
    }

    #[test]
    fn hyp1f1_complex_parameters_match_exp_identity() {
        let a = Complex64::new(1.25, -0.5);
        let z = Complex64::new(0.25, -0.75);
        let r = hyp1f1(
            &complex(a.re, a.im),
            &complex(a.re, a.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            z.exp(),
            1.0e-10,
        );
    }

    #[test]
    fn hyp1f1_complex_parameter_vector_matches_exp_identity() {
        let a = Complex64::new(0.75, 0.4);
        let z0 = Complex64::new(0.15, 0.2);
        let z1 = z0.conj();
        let r = hyp1f1(
            &complex(a.re, a.im),
            &complex(a.re, a.im),
            &complex_vec(vec![z0, z1]),
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], z0.exp(), 1.0e-10);
        assert_complex_close(values[1], z1.exp(), 1.0e-10);
    }

    #[test]
    fn hyp1f1_complex_real_parameters_match_real_path() {
        let complex_result = hyp1f1(
            &complex(1.0, 0.0),
            &complex(2.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let real_result = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp1f1_complex_parameter_pole_errors_hardened() {
        let result = hyp1f1(
            &complex(1.0, 0.0),
            &complex(0.0, 0.0),
            &scalar(1.0),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp1f1_parameter_vectors_broadcast_with_scalar_z() {
        let a = complex_vec(vec![Complex64::new(0.75, 0.4), Complex64::new(0.75, -0.4)]);
        let b = complex_vec(vec![Complex64::new(0.75, 0.4), Complex64::new(0.75, -0.4)]);
        let z = complex(0.15, 0.2);
        let result = hyp1f1(&a, &b, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], Complex64::new(0.15, 0.2).exp(), 1.0e-10);
        assert_complex_close(values[1], Complex64::new(0.15, 0.2).exp(), 1.0e-10);
    }

    #[test]
    fn hyp1f1_parameter_vector_mismatch_is_shape_error() {
        let result = hyp1f1(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            &scalar(0.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::ShapeMismatch));
    }

    // ── hyperu tests ─────────────────────────────────────────────────

    #[test]
    fn hyperu_matches_scipy_reference_points() {
        let cases = [
            (1.0, 2.0, 3.0, 1.0 / 3.0),
            (2.0, 3.0, 1.0, 1.0),
            (0.5, 1.5, 2.0, std::f64::consts::FRAC_1_SQRT_2),
            (2.5, 1.25, 0.75, 0.167_391_934_337_980_03),
            (1.0, 1.0, 1.0, 0.596_347_362_323_194_1),
            (2.0, 2.0, 1.0, 0.403_652_637_676_805_7),
            (1.0, 3.0, 0.5, 6.0),
            (1.0, -1.0, 1.0, 0.298_173_681_161_597_04),
        ];

        for (a, b, x, expected) in cases {
            let actual = hyperu_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= 5.0e-7 * scale,
                "hyperu({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn hyperu_nonpositive_integer_a_uses_laguerre_polynomial() {
        let u0 = hyperu_scalar(0.0, 2.0, 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let u1 = hyperu_scalar(-1.0, 2.0, 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let u2 = hyperu_scalar(-2.0, 3.0, 4.0, RuntimeMode::Strict).unwrap_or(f64::NAN);

        assert_eq!(u0, 1.0);
        assert!((u1 + 1.0).abs() <= 1.0e-14);
        assert!((u2 + 4.0).abs() <= 1.0e-14);
    }

    #[test]
    fn hyperu_vector_dispatch_broadcasts_real_inputs() {
        let result = hyperu(
            &scalar(1.0),
            &SpecialTensor::RealVec(vec![2.0, 3.0]),
            &SpecialTensor::RealVec(vec![3.0, 0.5]),
            RuntimeMode::Strict,
        );
        let values = get_real_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0 / 3.0).abs() <= 5.0e-8);
        assert!((values[1] - 6.0).abs() <= 5.0e-7);
    }

    #[test]
    fn hyperu_preserves_scipy_real_domain_edges() {
        let strict_negative_x = hyperu_scalar(1.0, 2.0, -1.0, RuntimeMode::Strict).unwrap_or(0.0);
        let hardened_negative_x = hyperu_scalar(1.0, 2.0, -1.0, RuntimeMode::Hardened);
        let finite_zero = hyperu_scalar(1.0, 0.5, 0.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let singular_zero = hyperu_scalar(1.0, 2.0, 0.0, RuntimeMode::Strict).unwrap_or(f64::NAN);

        assert!(strict_negative_x.is_nan());
        assert!(hardened_negative_x.is_err());
        assert!((finite_zero - 2.0).abs() <= 1.0e-12);
        assert!(singular_zero.is_infinite());
    }

    // Golden parity for U(a, b, x) with a < 0 non-integer and b a positive
    // integer, x > 0 — frankenscipy-msy83. Previously these fell through to
    // UnsupportedAnalyticContinuation (NaN); the connection formula is 0/0 at
    // integer b. Now solved by the positive-a integral plus downward recurrence
    // in a. Golden values from scipy 1.17.1 (sp.hyperu).
    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy 1.17.1
    fn hyperu_integer_b_negative_a_matches_scipy() {
        let cases: [(f64, f64, f64, f64); 12] = [
            (-0.5, 1.0, 2.0, 1.2459478282260943),
            (-0.5, 2.0, 1.5, 0.5670844020302459),
            (-1.5, 1.0, 3.0, 1.4575309529103873),
            (-2.5, 3.0, 2.0, 6.027810851185301),
            (-1.5, 2.0, 2.0, -1.4420471910282342),
            (-0.5, 3.0, 4.0, 1.3129770242215841),
            (-0.5, 2.0, 0.5, -0.5583597698714023),
            (-3.5, 2.0, 5.0, -15.853761233741835),
            (-0.5, 1.0, 1.0, 0.7704036149704431),
            (-0.25, 1.0, 2.0, 1.1557629221557946),
            (-0.75, 4.0, 3.0, -0.1699230467955729),
            (-1.5, 5.0, 8.0, 2.8443411947620896),
        ];
        for (a, b, x, expected) in cases {
            let actual = hyperu_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= 5.0e-7 * scale,
                "hyperu({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }

        // b = 0 stays NaN to match scipy; no spurious finite continuation.
        for (a, b, x) in [(-0.5, 0.0, 2.0), (-1.5, 0.0, 2.0)] {
            let v = hyperu_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            assert!(v.is_nan(), "hyperu({a}, {b}, {x}) = {v}, expected NaN");
        }
    }

    // Golden parity for U(a, b, x) with a < 0 non-integer and b a NEGATIVE
    // integer, x > 0 — frankenscipy-4yh8z. Solved by the Kummer transform
    // U(a,b,x) = x^{1-b} U(a-b+1, 2-b, x) feeding the positive-integer-b path.
    // Golden values from scipy 1.17.1 (sp.hyperu).
    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy 1.17.1
    fn hyperu_negative_integer_b_negative_a_matches_scipy() {
        let cases: [(f64, f64, f64, f64); 9] = [
            (-0.5, -1.0, 2.0, 1.846201538916142),
            (-0.5, -2.0, 3.0, 2.323217629163707),
            (-1.5, -1.0, 2.0, 4.0606905588693),
            (-0.5, -3.0, 4.0, 2.718776736157855),
            (-2.5, -2.0, 5.0, 73.26510445153117),
            (-3.5, -1.0, 2.0, -7.845165460691863),
            (-0.75, -2.0, 1.5, 2.658320426135889),
            (-0.25, -1.0, 1.0, 1.2635630857188185),
            (-1.5, -3.0, 3.0, 13.282084015814732),
        ];
        for (a, b, x, expected) in cases {
            let actual = hyperu_scalar(a, b, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let scale = expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= 5.0e-7 * scale,
                "hyperu({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    // ── hyp2f1 tests ────────────────────────────────────────────────

    #[test]
    fn hyp2f1_zero_z_is_one() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(3.0),
            &scalar(0.0),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp2f1_a_zero_is_one() {
        let r = hyp2f1(
            &scalar(0.0),
            &scalar(2.0),
            &scalar(3.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hyp2f1_known_geometric() {
        // 2F1(1, 1; 2; z) = -ln(1-z)/z for |z| < 1
        let z = 0.5;
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = -(1.0 - z).ln() / z;
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "2F1(1,1;2;0.5) should be -ln(0.5)/0.5: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    /// 2F1(1, 2; 3; z) anchor (frankenscipy-3zzep).
    ///
    /// Pre-fix the dispatch sent z = -0.99 to the direct series, which
    /// caps at 500 terms; the term ratio approaches |z| = 0.99 so the
    /// convergence criterion was not met and the series returned NaN.
    /// The fix routes any z < 0 (inside the unit disk) through the
    /// Pfaff transform (where the inner series sees |w| = |z|/(1+|z|)
    /// ≤ 1/2 and converges in a few dozen iterations) and bumps the
    /// direct-series cap to 5000 so |z| → 1 from above also converges.
    ///
    /// Anchored against scipy.special.hyp2f1 values directly. The
    /// closed form -2(z + ln(1-z))/z² is mathematically correct but
    /// suffers catastrophic cancellation for |z| ≪ 1 and so is not a
    /// useful oracle near zero.
    #[test]
    fn hyp2f1_matches_scipy_across_unit_disk() {
        // (z, scipy.special.hyp2f1(1, 2, 3, z))
        let cases: [(f64, f64); 8] = [
            (-0.99, 0.6159889016704397),
            (-0.95, 0.6253088696384367),
            (-0.9, 0.6373978119200126),
            (-0.5, 0.7562791351346845),
            (-0.1, 0.9379640391350281),
            (-0.001, 0.9993338329336662),
            (0.5, 1.5451774444795623),
            (0.99, 7.3771455687952),
        ];
        for (z, expected) in cases {
            let r = hyp2f1(
                &scalar(1.0),
                &scalar(2.0),
                &scalar(3.0),
                &scalar(z),
                RuntimeMode::Strict,
            );
            let got = get_scalar(&r).unwrap_or(f64::NAN);
            let scale = expected.abs().max(1.0);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-12 * scale,
                "2F1(1,2;3;{z}) = {got}, expected {expected}, diff = {diff}"
            );
        }
    }

    /// Regression: hyp2f1 must not return NaN for any z in (-1, 1)
    /// when c-a-b ≥ 0 and the series is absolutely convergent.
    /// frankenscipy-3zzep — pre-fix the 500-term cap on the direct
    /// series produced NaN at z = ±0.99.
    #[test]
    fn hyp2f1_does_not_return_nan_inside_unit_disk() {
        for &z in &[-0.99_f64, -0.95, -0.5, 0.5, 0.95, 0.99] {
            let r = hyp2f1(
                &scalar(1.0),
                &scalar(2.0),
                &scalar(3.0),
                &scalar(z),
                RuntimeMode::Strict,
            );
            let got = get_scalar(&r).unwrap_or(f64::NAN);
            assert!(got.is_finite(), "2F1(1,2;3;{z}) returned non-finite {got}");
        }
    }

    #[test]
    fn hyp2f1_negative_a_polynomial() {
        // 2F1(-2, b; c; z) terminates after 3 terms (polynomial)
        // 2F1(-2, 1; 1; z) = 1 + (-2)(1)z/(1·1) + (-2)(-1)(1)(2)z²/(1·2·1·2)
        //                   = 1 - 2z + z²  = (1-z)²
        let z = 0.3;
        let r = hyp2f1(
            &scalar(-2.0),
            &scalar(1.0),
            &scalar(1.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = (1.0 - z).powi(2);
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-10,
            "2F1(-2,1;1;z) should be (1-z)²: got {} expected {expected}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp2f1_terminating_polynomial_allows_z_outside_unit_disk() {
        let z = 2.0;
        let r = hyp2f1(
            &scalar(-2.0),
            &scalar(1.0),
            &scalar(1.5),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = 1.0
            + (-2.0 * 1.0 / 1.5) * z
            + ((-2.0 * -1.0) * (1.0 * 2.0) / ((1.5 * 2.5) * 2.0)) * z * z;
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - expected).abs() < 1e-12,
            "terminating 2F1 polynomial should remain valid outside |z| < 1"
        );
    }

    #[test]
    fn hyp2f1_z_greater_than_one_identity_matches_scipy_sample() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(1.0),
            &scalar(1.5),
            RuntimeMode::Strict,
        );
        assert!((get_scalar(&r).unwrap_or(f64::NAN) + 2.0).abs() < 1e-12);
    }

    #[test]
    fn hyp2f1_strict_z_greater_than_one_branch_cut_returns_infinity() {
        let cases = [
            (1.0, 2.0, 3.0, 1.5),
            (0.25, 0.75, 2.5, 1.25),
            (2.0, 1.0, 5.0, 1.1),
            (0.5, 1.5, 4.0, 2.0),
        ];
        for (a, b, c, z) in cases {
            let r = hyp2f1(
                &scalar(a),
                &scalar(b),
                &scalar(c),
                &scalar(z),
                RuntimeMode::Strict,
            );
            let got = get_scalar(&r).unwrap_or(f64::NAN);
            assert!(
                got.is_infinite() && got.is_sign_positive(),
                "real branch-cut z > 1 should match scipy's positive infinity for \
                 a={a}, b={b}, c={c}, z={z}; got {got}"
            );
        }
    }

    #[test]
    fn hyp2f1_hardened_z_greater_than_one_unsupported_errors() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(3.0),
            &scalar(1.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&r), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp2f1_at_z_one_gauss_sum() {
        // 2F1(1, 1; 3; 1) = Gamma(3)*Gamma(1) / (Gamma(2)*Gamma(2)) = 2*1/(1*1) = 2
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(3.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        assert!(
            (get_scalar(&r).unwrap_or(f64::NAN) - 2.0).abs() < 1e-8,
            "2F1(1,1;3;1) should be 2: got {}",
            get_scalar(&r).unwrap_or(f64::NAN)
        );
    }

    #[test]
    fn hyp2f1_c_zero_returns_nan_strict() {
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(val.is_nan(), "c=0 should return NaN in strict mode");
    }

    #[test]
    fn hyp2f1_negative_z() {
        // 2F1(1, 1; 2; -0.5) should use Pfaff transformation
        let z = -0.5;
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(z),
            RuntimeMode::Strict,
        );
        let expected = -(1.0 - z).ln() / z; // = -ln(1.5)/(-0.5) = 2*ln(1.5)
        let val = get_scalar(&r).unwrap_or(f64::NAN);
        assert!(
            (val - expected).abs() < 1e-10,
            "2F1(1,1;2;-0.5): got {val} expected {expected}"
        );
    }

    #[test]
    fn hyp2f1_complex_matches_inverse_linear_identity_inside_unit_disk() {
        let z = Complex64::new(0.25, 0.25);
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        let expected = Complex64::from_real(1.0) / (Complex64::from_real(1.0) - z);
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            expected,
            1.0e-10,
        );
    }

    #[test]
    fn hyp2f1_complex_vector_preserves_conjugation_symmetry() {
        let z = Complex64::new(0.2, 0.3);
        let input = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let r = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &input,
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[1], values[0].conj(), 1.0e-10);
    }

    #[test]
    fn hyp2f1_complex_outside_unit_disk_degenerate_ab_integer() {
        // a - b = -1 (exact integer) with Re(z) >= 1/2 outside the disk: the
        // generic z->1/z weights hit poles, so this exercises the DLMF 15.8.8
        // degenerate limit (frankenscipy-31a0c). Doubly degenerate here since
        // c = b = 2, so 2F1(1,2;2;z) = (1-z)^{-1}. mpmath reference at
        // z = 1.25 + 0.5i is -0.8 + 1.6i exactly.
        let z = Complex64::new(1.25, 0.5);
        let result = hyp2f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(2.0),
            &complex(z.re, z.im),
            RuntimeMode::Hardened,
        );
        let value = get_complex_scalar(&result).unwrap_or(Complex64::new(f64::NAN, f64::NAN));
        let expected = (Complex64::from_real(1.0) - z).recip();
        assert_complex_close(value, expected, 1.0e-7);
    }

    #[test]
    fn hyp2f1_complex_outside_unit_disk_degenerate_matches_mpmath() {
        // Non-doubly-degenerate a - b integer cases (c-a, c-b generic) against
        // mpmath.hyp2f1 references computed at 30 digits. frankenscipy-31a0c.
        let cases = [
            // (a, b, c, z_re, z_im, ref_re, ref_im)
            (2.0, 3.0, 1.5, 1.4, 0.6, 0.782_543_846_311_524, 5.211_527_724_255_207),
            (1.5, 0.5, 2.0, 2.0, 1.0, 0.588_752_395_346_859, 0.805_805_439_764_103),
            (3.0, 1.0, 2.5, 1.1, 0.9, -0.419_516_836_979_255, 0.863_898_169_851_454),
        ];
        for (a, b, c, zr, zi, rr, ri) in cases {
            let result = hyp2f1(
                &scalar(a),
                &scalar(b),
                &scalar(c),
                &complex(zr, zi),
                RuntimeMode::Strict,
            );
            let value = get_complex_scalar(&result).unwrap_or(Complex64::new(f64::NAN, f64::NAN));
            assert_complex_close(value, Complex64::new(rr, ri), 1.0e-6);
        }
    }

    #[test]
    fn hyp2f1_complex_parameter_linear_identity_holds() {
        let b = Complex64::new(1.25, -0.5);
        let z = Complex64::new(0.2, 0.3);
        let r = hyp2f1(
            &scalar(-1.0),
            &complex(b.re, b.im),
            &complex(b.re, b.im),
            &complex(z.re, z.im),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&r).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(1.0) - z,
            1.0e-12,
        );
    }

    #[test]
    fn hyp2f1_complex_parameter_vector_matches_linear_identity() {
        let b = Complex64::new(0.75, 0.6);
        let z0 = Complex64::new(0.15, -0.25);
        let z1 = z0.conj();
        let r = hyp2f1(
            &scalar(-1.0),
            &complex(b.re, b.im),
            &complex(b.re, b.im),
            &complex_vec(vec![z0, z1]),
            RuntimeMode::Strict,
        );
        let values = get_complex_vec(&r).unwrap_or(&[]);
        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], Complex64::from_real(1.0) - z0, 1.0e-12);
        assert_complex_close(values[1], Complex64::from_real(1.0) - z1, 1.0e-12);
    }

    #[test]
    fn hyp2f1_complex_real_parameters_match_real_path() {
        let complex_result = hyp2f1(
            &complex(1.0, 0.0),
            &complex(1.0, 0.0),
            &complex(2.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        let real_result = hyp2f1(
            &scalar(1.0),
            &scalar(1.0),
            &scalar(2.0),
            &scalar(0.5),
            RuntimeMode::Strict,
        );
        assert_complex_close(
            get_complex_scalar(&complex_result).unwrap_or(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::from_real(get_scalar(&real_result).unwrap_or(f64::NAN)),
            1.0e-12,
        );
    }

    #[test]
    fn hyp2f1_complex_parameter_pole_errors_hardened() {
        let result = hyp2f1(
            &scalar(1.0),
            &complex(1.0, 0.0),
            &complex(0.0, 0.0),
            &scalar(0.5),
            RuntimeMode::Hardened,
        );
        assert_eq!(error_kind(&result), Some(SpecialErrorKind::DomainError));
    }

    #[test]
    fn hyp2f1_parameter_vectors_broadcast_with_scalar_z() {
        let b = complex_vec(vec![Complex64::new(0.75, 0.6), Complex64::new(0.75, -0.6)]);
        let c = complex_vec(vec![Complex64::new(0.75, 0.6), Complex64::new(0.75, -0.6)]);
        let z = complex(0.15, -0.25);
        let result = hyp2f1(&scalar(-1.0), &b, &c, &z, RuntimeMode::Strict);
        let values = get_complex_vec(&result).unwrap_or(&[]);

        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            Complex64::from_real(1.0) - Complex64::new(0.15, -0.25),
            1.0e-12,
        );
        assert_complex_close(
            values[1],
            Complex64::from_real(1.0) - Complex64::new(0.15, -0.25),
            1.0e-12,
        );
    }

    #[test]
    fn hyp1f1_matches_scipy_reference_values() {
        // scipy.special.hyp1f1(1, 2, 1) = (e-1) ≈ 1.7182818284
        // scipy.special.hyp1f1(0.5, 1.5, 1) ≈ 2.0179...
        let result = hyp1f1(
            &scalar(1.0),
            &scalar(2.0),
            &scalar(1.0),
            RuntimeMode::Strict,
        );
        let val = get_scalar(&result).expect("hyp1f1 result");
        let expected = std::f64::consts::E - 1.0;
        assert!(
            (val - expected).abs() < 1e-6,
            "hyp1f1(1, 2, 1) got {val}, expected {expected}"
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn hyp1f1_large_positive_z_matches_scipy() {
        // z > 200 exercises the DLMF 13.7.2 asymptotic path (frankenscipy-8a4qg):
        // the 500-term direct series returns NaN for z ≳ 440. Golden values from
        // scipy.special.hyp1f1 1.17.1 (cross-checked against mpmath 1.4.1).
        let cases = [
            (2.0, 3.0, 250.0, 2.98517503683573e106),
            (2.0, 3.0, 450.0, 1.2005165889159134e193),
            (2.0, 3.0, 500.0, 5.603140133668527e214),
            (2.0, 3.0, 700.0, 2.893666147999053e301),
            (0.5, 1.5, 300.0, 3.2428001599029676e127),
            (0.5, 1.5, 650.0, 1.5059293665485805e279),
            (1.0, 3.0, 220.0, 1.4486739567102264e91),
        ];
        for (a, b, z, expected) in cases {
            let r = hyp1f1(&scalar(a), &scalar(b), &scalar(z), RuntimeMode::Strict);
            let v = get_scalar(&r).expect("finite hyp1f1");
            let rel = ((v - expected) / expected).abs();
            assert!(rel < 1e-12, "hyp1f1({a},{b},{z}) = {v:e}, scipy {expected:e}, rel={rel:e}");
        }

        // SciPy overflows e^z to +inf for z past ln(f64::MAX) ≈ 709.78; match it.
        let inf = hyp1f1(&scalar(1.0), &scalar(2.0), &scalar(710.0), RuntimeMode::Strict);
        assert_eq!(get_scalar(&inf), Some(f64::INFINITY));
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn hyp1f1_large_negative_z_matches_scipy() {
        // z < -200 exercises the DLMF 13.7.2 z -> -∞ asymptotic (frankenscipy-k8kkf):
        // the Kummer inner series returned NaN for z ≲ -440. The last two cases
        // (b-a a nonpositive integer) take the exponential branch. scipy 1.17.1.
        let cases = [
            (2.0, 3.0, -500.0, 8.000000000000001e-06),
            (0.5, 1.5, -650.0, 0.03476067989503808),
            (1.0, 3.0, -460.0, 0.004338374291115312),
            (2.5, 4.0, -1000.0, 2.1382704062592586e-07),
            (5.0, 3.0, -500.0, 1.4606094091460309e-213),
            (4.0, 2.0, -460.0, 5.837316424310934e-196),
        ];
        for (a, b, z, expected) in cases {
            let r = hyp1f1(&scalar(a), &scalar(b), &scalar(z), RuntimeMode::Strict);
            let v = get_scalar(&r).expect("finite hyp1f1");
            let rel = ((v - expected) / expected).abs();
            assert!(rel < 1e-12, "hyp1f1({a},{b},{z}) = {v:e}, scipy {expected:e}, rel={rel:e}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn hyp0f1_large_negative_z_matches_scipy() {
        // Oscillatory (z < 0) branch via the full DLMF 10.17.3 J_ν asymptotic
        // (frankenscipy-o9ws0): the leading-order form was 1-3% off. scipy 1.17.1.
        let cases = [
            (2.0, -50.0, 2.1893548253388965e-02),
            (2.0, -200.0, 7.3270880329473410e-03),
            (0.5, -400.0, -6.6693806165226180e-01),
            (3.0, -800.0, -0.00019726526855692484),
            (1.5, -100.0, 0.04564726253638135),
            (2.5, -1000.0, -0.0006819625965429237),
        ];
        for (b, z, expected) in cases {
            let v = hyp0f1_scalar(b, z, RuntimeMode::Strict).expect("finite");
            let rel = ((v - expected) / expected).abs();
            assert!(rel < 1e-9, "hyp0f1({b},{z}) = {v:e}, scipy {expected:e}, rel={rel:e}");
        }
        // Positive z (modified Bessel I) branch stays correct.
        let pos = hyp0f1_scalar(2.0, 200.0, RuntimeMode::Strict).expect("finite");
        assert!(((pos - 1.0056860439881468e10) / 1.0056860439881468e10).abs() < 1e-9);
    }

    #[test]
    fn hyp0f1_matches_scipy_reference_values() {
        // scipy.special.hyp0f1(1, 0) = 1
        // scipy.special.hyp0f1(2, 1) ≈ 1.2660658478...
        let result0 = hyp0f1_scalar(1.0, 0.0, RuntimeMode::Strict).expect("hyp0f1(1, 0)");
        assert!(
            (result0 - 1.0).abs() < 1e-10,
            "hyp0f1(1, 0) got {result0}, expected 1.0"
        );

        let result1 = hyp0f1_scalar(2.0, 1.0, RuntimeMode::Strict).expect("hyp0f1(2, 1)");
        assert!(
            (result1 - 1.5906).abs() < 1e-3,
            "hyp0f1(2, 1) got {result1}, expected ~1.5906"
        );
    }

    type Hyp2f1ComplexCase = (f64, f64, f64, (f64, f64), (f64, f64));

    // Complex 2F1 analytic continuation to |z| >= 1 for Re(z) < 1/2 via the
    // Pfaff transform — frankenscipy-f69ch (the Re(z) >= 1/2 half is a separate
    // gap). scipy has no complex hyp2f1 ufunc; golden values are mpmath 1.4.1
    // (mp.dps=25), which is the standard high-precision reference.
    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from mpmath
    fn hyp2f1_complex_pfaff_continuation_matches_mpmath() {
        // (a, b, c, (z_re, z_im), (val_re, val_im))
        let cases: [Hyp2f1ComplexCase; 6] = [
            (0.5, 0.5, 1.5, (-1.0, 1.0), (0.862231298085739, 0.08130072049794629)),
            (0.5, 1.0, 2.0, (-2.0, 0.5), (0.7284461294343433, 0.038269144396779094)),
            (0.3, 0.7, 1.2, (-1.0, -1.0), (0.8593620385670224, -0.07934104644643247)),
            (1.0, 0.5, 1.5, (0.2, 1.5), (0.7910171811374038, 0.3217213255606514)),
            (0.5, 0.5, 2.0, (-3.0, 2.0), (0.7860012327422344, 0.0725236459589195)),
            (1.5, 0.5, 2.5, (-0.5, 3.0), (0.6267035512763195, 0.280313155049662)),
        ];
        for (a, b, c, (zr, zi), (vr, vi)) in cases {
            let result = hyp2f1(
                &complex(a, 0.0),
                &complex(b, 0.0),
                &complex(c, 0.0),
                &complex(zr, zi),
                RuntimeMode::Strict,
            );
            let got = get_complex_scalar(&result)
                .unwrap_or_else(|| Complex64::new(f64::NAN, f64::NAN));
            let expected = Complex64::new(vr, vi);
            let err = (got - expected).abs();
            let scale = expected.abs().max(1.0);
            assert!(
                err <= 1e-12 * scale,
                "hyp2f1({a},{b};{c};{zr}+{zi}i) = {got:?}, mpmath = {expected:?}, err={err:e}"
            );
        }

        // a == b (a-b = 0) with c-a = 1 also integer: the triple-degenerate
        // DLMF 15.8.9 corner, now resolved (frankenscipy-wwi45). mpmath 1.4.1
        // reference at z = 2+1i is 0.97633716024281776 + 0.44586478294123795i.
        let beyond = hyp2f1(
            &complex(0.5, 0.0),
            &complex(0.5, 0.0),
            &complex(1.5, 0.0),
            &complex(2.0, 1.0),
            RuntimeMode::Strict,
        );
        let v = get_complex_scalar(&beyond).unwrap_or_else(|| Complex64::new(f64::NAN, f64::NAN));
        assert_complex_close(
            v,
            Complex64::new(0.97633716024281776, 0.44586478294123795),
            1.0e-9,
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from mpmath
    fn hyp2f1_complex_triple_degenerate_matches_mpmath() {
        // a-b integer AND c-a integer (c != a, c != b) outside the unit disk:
        // the fully-logarithmic DLMF 15.8.9/15.8.11 corner (frankenscipy-wwi45).
        // Golden values from mpmath 1.4.1 (mp.dps=30).
        let cases: [Hyp2f1ComplexCase; 4] = [
            (0.5, 0.5, 2.5, (2.0, 1.0), (1.0764492159986923, 0.28361960044123111)),
            (0.3, 1.3, 2.3, (1.6, 0.7), (1.0667148233989199, 0.42844504985442885)),
            (1.5, 0.5, 2.5, (1.4, -0.9), (0.95593059253373275, -0.65513882101253395)),
            (0.5, 0.5, 3.5, (1.2, 1.1), (1.0525162721774637, 0.12780600458121074)),
        ];
        for (a, b, c, (zr, zi), (vr, vi)) in cases {
            let result = hyp2f1(
                &scalar(a),
                &scalar(b),
                &scalar(c),
                &complex(zr, zi),
                RuntimeMode::Strict,
            );
            let got =
                get_complex_scalar(&result).unwrap_or_else(|| Complex64::new(f64::NAN, f64::NAN));
            assert_complex_close(got, Complex64::new(vr, vi), 1.0e-9);
        }
    }

    // Complex 2F1 in the right half-plane Re(z) >= 1/2, |z| >= 1, via the
    // z -> 1/z connection formula (a-b non-integer) — frankenscipy-f69ch.
    // Golden values from mpmath 1.4.1 (mp.dps=25); scipy has no complex hyp2f1.
    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from mpmath
    fn hyp2f1_complex_inv_z_continuation_matches_mpmath() {
        let cases: [Hyp2f1ComplexCase; 6] = [
            (0.5, 0.3, 1.2, (2.0, 1.0), (0.9903704668763302, 0.32750091263584863)),
            (1.0, 0.4, 1.5, (2.5, 2.0), (0.6561515879441171, 0.5286478314518919)),
            (0.7, 0.2, 1.1, (5.0, -3.0), (0.7697473975684688, -0.32616546130854634)),
            (0.5, 0.25, 1.0, (1.5, 0.5), (1.0741936293080392, 0.30149211979729423)),
            (0.9, 0.3, 2.0, (2.0, 0.1), (1.147235558288442, 0.489377889798951)),
            (0.4, 0.7, 1.3, (4.0, -1.0), (0.6477257675533721, -0.6024687066512131)),
        ];
        for (a, b, c, (zr, zi), (vr, vi)) in cases {
            let result = hyp2f1(
                &complex(a, 0.0),
                &complex(b, 0.0),
                &complex(c, 0.0),
                &complex(zr, zi),
                RuntimeMode::Strict,
            );
            let got = get_complex_scalar(&result)
                .unwrap_or_else(|| Complex64::new(f64::NAN, f64::NAN));
            let expected = Complex64::new(vr, vi);
            let err = (got - expected).abs();
            let scale = expected.abs().max(1.0);
            assert!(
                err <= 1e-11 * scale,
                "hyp2f1({a},{b};{c};{zr}+{zi}i) = {got:?}, mpmath = {expected:?}, err={err:e}"
            );
        }
    }
}
