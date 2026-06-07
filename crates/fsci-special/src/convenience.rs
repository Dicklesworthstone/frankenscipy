#![forbid(unsafe_code)]

//! Convenience special functions commonly used in scientific computing.
//!
//! - `sinc` — Normalized sinc function: sin(πx)/(πx)
//! - `xlogy` — x * log(y) with correct handling of x=0
//! - `xlog1py` — x * log1p(y) with correct handling of x=0
//! - `logsumexp` — Log of sum of exponentials (numerically stable)
//! - `expit` — Logistic sigmoid: 1 / (1 + exp(-x))
//! - `logit` — Log-odds: log(p / (1 - p))
//! - `entr` — Elementwise entropy: -x * log(x)
//! - `rel_entr` — Relative entropy (KL divergence element): x * log(x/y)
//! - `ndtr` / `ndtri` / `ndtri_exp` — Standard normal CDF and inverse CDF
//! - `stirling2` — Stirling numbers of the second kind
//! - `nrdtrimn` — Recover the normal mean from a CDF value, scale, and quantile
//! - `kl_div` — KL divergence element with the `-x + y` correction

use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, record_special_trace,
};

pub const CONVENIENCE_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "sinc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "|x| < 1e-7: Taylor series to avoid 0/0",
            },
            DispatchStep {
                regime: KernelRegime::BackendDelegate,
                when: "general: sin(πx)/(πx)",
            },
        ],
        notes: "sinc(0) = 1 by convention. Matches numpy.sinc normalization.",
    },
    DispatchPlan {
        function: "xlogy",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "direct evaluation with 0*log(y)=0 convention",
        }],
        notes: "Strict mode propagates NaN even when x=0.",
    },
    DispatchPlan {
        function: "rel_entr",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "direct evaluation using x*log(x/y)",
        }],
        notes: "Matches SciPy domain rules including infinities.",
    },
    DispatchPlan {
        function: "stirling2",
        steps: &[DispatchStep {
            regime: KernelRegime::Recurrence,
            when: "integer n,k inputs: S(n,k) = k*S(n-1,k) + S(n-1,k-1)",
        }],
        notes: "Default floating output for scipy.special.stirling2(N, K, exact=False).",
    },
];

const DILOG_SERIES_MAX_TERMS: usize = 128;
const PI_SQUARED_OVER_SIX: f64 = PI * PI / 6.0;
const NDTRI_EXP_LOG_P_LOW: f64 = -3.719_338_661_598_645;
const NDTRI_EXP_LOG_P_HIGH: f64 = -0.024_548_872_921_412_7;

#[allow(clippy::excessive_precision)]
const NDTRI_EXP_A: [f64; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_690e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
];

#[allow(clippy::excessive_precision)]
const NDTRI_EXP_B: [f64; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
];

#[allow(clippy::excessive_precision)]
const NDTRI_EXP_C: [f64; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
];

#[allow(clippy::excessive_precision)]
const NDTRI_EXP_D: [f64; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
];

/// Normalized sinc function: sin(πx) / (πx).
///
/// sinc(0) = 1. Matches `numpy.sinc(x)` (not the unnormalized version).
pub fn sinc(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("sinc", x_tensor, mode, |x| Ok(sinc_scalar(x)))
}

/// Compute x * log(y), with the convention that 0 * log(0) = 0.
///
/// Matches `scipy.special.xlogy(x, y)`.
pub fn xlogy(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("xlogy", x_tensor, y_tensor, mode, |x, y| {
        Ok(xlogy_scalar(x, y))
    })
}

/// Compute x * log1p(y), with the convention that 0 * log1p(y) = 0 when x = 0.
///
/// Matches `scipy.special.xlog1py(x, y)`.
pub fn xlog1py(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("xlog1py", x_tensor, y_tensor, mode, |x, y| {
        Ok(xlog1py_scalar(x, y))
    })
}

/// Log of the sum of exponentials (numerically stable).
///
/// logsumexp(a) = log(sum(exp(a))) computed without overflow.
/// Matches `scipy.special.logsumexp(a)`.
pub fn logsumexp(data: &[f64]) -> f64 {
    let weights = vec![1.0; data.len()];
    logsumexp_weighted_unchecked(data, &weights)
}

/// Weighted log of the sum of exponentials (numerically stable).
///
/// Matches `scipy.special.logsumexp(a, b=b)`.
pub fn logsumexp_with_b(data: &[f64], b: &[f64]) -> Result<f64, SpecialError> {
    if data.len() != b.len() {
        return Err(SpecialError {
            function: "logsumexp_with_b",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "data and weights must have the same length",
        });
    }
    Ok(logsumexp_weighted_unchecked(data, b))
}

/// Log of the sum of exponentials reduced along a 2D axis.
///
/// Matches `scipy.special.logsumexp(a, axis=axis)` for 2D inputs.
pub fn logsumexp_axis_2d(data: &[Vec<f64>], axis: usize) -> Result<Vec<f64>, SpecialError> {
    logsumexp_axis_2d_impl(data, axis, None)
}

/// Weighted log of the sum of exponentials reduced along a 2D axis.
///
/// Matches `scipy.special.logsumexp(a, axis=axis, b=b)` for 2D inputs, including
/// NumPy-style broadcasting where each weight dimension is either 1 or matches
/// the corresponding data dimension.
pub fn logsumexp_axis_2d_with_b(
    data: &[Vec<f64>],
    axis: usize,
    b: &[Vec<f64>],
) -> Result<Vec<f64>, SpecialError> {
    logsumexp_axis_2d_impl(data, axis, Some(b))
}

fn logsumexp_axis_2d_impl(
    data: &[Vec<f64>],
    axis: usize,
    b: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>, SpecialError> {
    let (rows, cols) = rectangular_shape(data, "logsumexp_axis_2d")?;
    if axis > 1 {
        return Err(SpecialError {
            function: "logsumexp_axis_2d",
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "axis must be 0 or 1",
        });
    }

    let weight_shape = b
        .map(|weights| {
            let (weight_rows, weight_cols) =
                rectangular_shape(weights, "logsumexp_axis_2d_with_b")?;
            if !dimension_is_broadcastable(weight_rows, rows)
                || !dimension_is_broadcastable(weight_cols, cols)
            {
                return Err(SpecialError {
                    function: "logsumexp_axis_2d_with_b",
                    kind: SpecialErrorKind::DomainError,
                    mode: RuntimeMode::Strict,
                    detail: "weights are not broadcast-compatible with data",
                });
            }
            Ok((weight_rows, weight_cols))
        })
        .transpose()?;

    match axis {
        0 => {
            let mut reduced = Vec::with_capacity(cols);
            for col in 0..cols {
                let column = data
                    .iter()
                    .map(|row_values| row_values[col])
                    .collect::<Vec<_>>();
                let value =
                    if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                        let column_weights = data
                            .iter()
                            .enumerate()
                            .map(|(row, _)| weight_at(weights, weight_rows, weight_cols, row, col))
                            .collect::<Vec<_>>();
                        logsumexp_with_b(&column, &column_weights)?
                    } else {
                        logsumexp(&column)
                    };
                reduced.push(value);
            }
            Ok(reduced)
        }
        1 => {
            let mut reduced = Vec::with_capacity(rows);
            for (row, row_values) in data.iter().enumerate() {
                let value =
                    if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                        let row_weights = (0..cols)
                            .map(|col| weight_at(weights, weight_rows, weight_cols, row, col))
                            .collect::<Vec<_>>();
                        logsumexp_with_b(row_values, &row_weights)?
                    } else {
                        logsumexp(row_values)
                    };
                reduced.push(value);
            }
            Ok(reduced)
        }
        _ => unreachable!("axis validated above"),
    }
}

fn logsumexp_weighted_unchecked(data: &[f64], b: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max_val = f64::NEG_INFINITY;
    let mut saw_active_term = false;
    for (&value, &weight) in data.iter().zip(b.iter()) {
        if value.is_nan() || weight.is_nan() {
            return f64::NAN;
        }
        if weight == 0.0 {
            continue;
        }
        saw_active_term = true;
        max_val = max_val.max(value);
    }

    if !saw_active_term || max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    if max_val == f64::INFINITY {
        let infinite_weight_sum = data
            .iter()
            .zip(b.iter())
            .filter(|(value, weight)| **weight != 0.0 && **value == f64::INFINITY)
            .map(|(_, weight)| *weight)
            .sum::<f64>();
        return if infinite_weight_sum > 0.0 {
            f64::INFINITY
        } else if infinite_weight_sum == 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }

    let sum_exp = data
        .iter()
        .zip(b.iter())
        .filter(|(_, weight)| **weight != 0.0)
        .map(|(value, weight)| *weight * (*value - max_val).exp())
        .sum::<f64>();
    if sum_exp > 0.0 {
        max_val + sum_exp.ln()
    } else if sum_exp == 0.0 {
        f64::NEG_INFINITY
    } else {
        f64::NAN
    }
}

fn rectangular_shape(
    matrix: &[Vec<f64>],
    function: &'static str,
) -> Result<(usize, usize), SpecialError> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != cols) {
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode: RuntimeMode::Strict,
            detail: "matrix rows must all have the same length",
        });
    }
    Ok((rows, cols))
}

fn dimension_is_broadcastable(source: usize, target: usize) -> bool {
    source == target || source == 1 || source == 0 || target == 0
}

fn weight_at(
    weights: &[Vec<f64>],
    weight_rows: usize,
    weight_cols: usize,
    row: usize,
    col: usize,
) -> f64 {
    let source_row = if weight_rows <= 1 { 0 } else { row };
    let source_col = if weight_cols <= 1 { 0 } else { col };
    weights[source_row][source_col]
}

/// Logistic sigmoid function: 1 / (1 + exp(-x)).
///
/// Matches `scipy.special.expit(x)`.
pub fn expit(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("expit", x_tensor, mode, |x| Ok(expit_scalar(x)))
}

/// Log-odds function: log(p / (1 - p)).
///
/// Matches `scipy.special.logit(p)`.
/// Domain: p in (0, 1).
pub fn logit(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("logit", p_tensor, mode, |p| logit_scalar(p, mode))
}

/// Elementwise entropy: -x * log(x).
///
/// Returns 0 for x = 0, -inf for x < 0.
/// Matches `scipy.special.entr(x)`.
pub fn entr(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("entr", x_tensor, mode, |x| Ok(entr_scalar(x)))
}

/// Relative entropy element: x * log(x/y).
///
/// Matches `scipy.special.rel_entr(x, y)`.
pub fn rel_entr(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("rel_entr", x_tensor, y_tensor, mode, |x, y| {
        Ok(rel_entr_scalar(x, y))
    })
}

/// Standard normal cumulative distribution function Φ(x).
///
/// Matches `scipy.special.ndtr(x)`.
pub fn ndtr(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("ndtr", x_tensor, mode, |x| Ok(ndtr_scalar(x)))
}

#[must_use]
pub fn ndtr_scalar(x: f64) -> f64 {
    // Use erfc for improved tail accuracy (avoids catastrophic cancellation for x << 0).
    0.5 * crate::error::erfc_scalar(-x * FRAC_1_SQRT_2)
}

/// Inverse standard normal cumulative distribution function Φ⁻¹(y).
///
/// Matches `scipy.special.ndtri(y)`.
pub fn ndtri(y_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("ndtri", y_tensor, mode, |y| Ok(ndtri_scalar(y)))
}

#[must_use]
pub fn ndtri_scalar(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::NEG_INFINITY;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    // Route through erfcinv on the small complementary argument so the deep tail
    // stays finite (2y-1 rounds to ±1 for |y-½| → ½, killing the old form).
    if y <= 0.5 {
        -SQRT_2 * erfcinv_conv(2.0 * y)
    } else {
        SQRT_2 * erfcinv_conv(2.0 * (1.0 - y))
    }
}

/// Inverse of `log_ndtr`.
///
/// Finds `x` such that `log_ndtr(x) = y`, matching `scipy.special.ndtri_exp(y)`.
pub fn ndtri_exp(y_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("ndtri_exp", y_tensor, mode, |y| Ok(ndtri_exp_scalar(y)))
}

/// Scalar helper for `ndtri_exp`.
#[must_use]
pub fn ndtri_exp_scalar(log_p: f64) -> f64 {
    if log_p.is_nan() {
        return f64::NAN;
    }
    if log_p > 0.0 {
        return f64::NAN;
    }
    if log_p == 0.0 {
        return f64::INFINITY;
    }
    if log_p == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    if log_p <= NDTRI_EXP_LOG_P_LOW {
        return ndtri_exp_lower_tail_from_log_p(log_p);
    }
    if log_p >= NDTRI_EXP_LOG_P_HIGH {
        let upper_tail = -log_p.exp_m1();
        if upper_tail == 0.0 {
            return f64::INFINITY;
        }
        return -ndtri_exp_lower_tail_from_log_p(upper_tail.ln());
    }

    ndtri_exp_central(log_p.exp())
}

fn ndtri_exp_lower_tail_from_log_p(log_p: f64) -> f64 {
    let q = (-2.0 * log_p).sqrt();
    let numerator =
        (((((NDTRI_EXP_C[0] * q + NDTRI_EXP_C[1]) * q + NDTRI_EXP_C[2]) * q + NDTRI_EXP_C[3]) * q
            + NDTRI_EXP_C[4])
            * q)
            + NDTRI_EXP_C[5];
    let denominator =
        ((((NDTRI_EXP_D[0] * q + NDTRI_EXP_D[1]) * q + NDTRI_EXP_D[2]) * q + NDTRI_EXP_D[3]) * q)
            + 1.0;
    numerator / denominator
}

fn ndtri_exp_central(p: f64) -> f64 {
    let q = p - 0.5;
    let r = q * q;
    let numerator =
        (((((NDTRI_EXP_A[0] * r + NDTRI_EXP_A[1]) * r + NDTRI_EXP_A[2]) * r + NDTRI_EXP_A[3]) * r
            + NDTRI_EXP_A[4])
            * r
            + NDTRI_EXP_A[5])
            * q;
    let denominator =
        (((((NDTRI_EXP_B[0] * r + NDTRI_EXP_B[1]) * r + NDTRI_EXP_B[2]) * r + NDTRI_EXP_B[3]) * r
            + NDTRI_EXP_B[4])
            * r)
            + 1.0;
    numerator / denominator
}

/// Recover the mean of a normal distribution from a CDF value, standard deviation, and quantile.
///
/// Matches `scipy.special.nrdtrimn(p, std, x)`.
#[must_use]
pub fn nrdtrimn(p: f64, std: f64, x: f64) -> f64 {
    if p.is_nan() || std.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if std <= 0.0 || !(0.0 < p && p < 1.0) {
        return f64::NAN;
    }
    if std == f64::INFINITY {
        if x.is_finite() {
            return if p < 0.5 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        if x.is_sign_positive() {
            return if p < 0.5 { f64::INFINITY } else { f64::NAN };
        }
        return if p < 0.5 { f64::NAN } else { f64::NEG_INFINITY };
    }
    x - std * ndtri_scalar(p)
}

/// Recover the standard deviation of a normal distribution from a mean, CDF value, and quantile.
///
/// Matches `scipy.special.nrdtrisd(mn, p, x)`.
#[must_use]
pub fn nrdtrisd(mn: f64, p: f64, x: f64) -> f64 {
    const NRDTRISD_P50_DENOM: f64 = 6.637_989_419_862_078e-17;

    if mn.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0 < p && p < 1.0) {
        return f64::NAN;
    }
    let delta = x - mn;
    if p == 0.5 {
        return delta / NRDTRISD_P50_DENOM;
    }
    delta / ndtri_scalar(p)
}

/// KL divergence element `x * log(x / y) - x + y`.
///
/// Matches `scipy.special.kl_div(x, y)`.
#[must_use]
pub fn kl_div(x: f64, y: f64) -> f64 {
    kl_div_scalar(x, y)
}

// ══════════════════════════════════════════════════════════════════════
// Scalar Kernels
// ══════════════════════════════════════════════════════════════════════

fn sinc_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let px = PI * x;
    if px.abs() < 1.0e-7 {
        // Taylor series: sinc(x) ≈ 1 - (πx)²/6 + (πx)⁴/120
        let px2 = px * px;
        1.0 - px2 / 6.0 + px2 * px2 / 120.0
    } else {
        px.sin() / px
    }
}

fn xlogy_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

fn xlog1py_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    // scipy.special.xlog1py is x * log1p(y); use ln_1p so small |y| keeps full
    // precision (a naive (1.0 + y).ln() rounds 1.0 + y to 1.0 for y ≲ 1e-16,
    // returning 0 where scipy returns ~y).
    if x == 0.0 { 0.0 } else { x * y.ln_1p() }
}

fn expit_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn logit_scalar(p: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if p.is_nan() {
        return Ok(f64::NAN);
    }
    if p <= 0.0 || p >= 1.0 {
        return match mode {
            RuntimeMode::Strict => {
                if p <= 0.0 {
                    Ok(f64::NEG_INFINITY)
                } else {
                    Ok(f64::INFINITY)
                }
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    "logit",
                    mode,
                    "domain_error",
                    format!("p={p}"),
                    "fail_closed",
                    "p must be in (0, 1)",
                    false,
                );
                Err(SpecialError {
                    function: "logit",
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "p must be in (0, 1)",
                })
            }
        };
    }
    Ok((p / (1.0 - p)).ln())
}

fn entr_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        0.0
    } else if x < 0.0 {
        f64::NEG_INFINITY
    } else {
        -x * x.ln()
    }
}

fn rel_entr_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 && y >= 0.0 {
        0.0
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln()
    } else {
        f64::INFINITY
    }
}

fn kl_div_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 && y >= 0.0 {
        y
    } else if x > 0.0 && y > 0.0 {
        x * (x / y).ln() - x + y
    } else {
        f64::INFINITY
    }
}

// ══════════════════════════════════════════════════════════════════════
// Fresnel integrals
// ══════════════════════════════════════════════════════════════════════

/// Fresnel integrals S(z) and C(z).
///
/// S(z) = ∫₀ᶻ sin(πt²/2) dt
/// C(z) = ∫₀ᶻ cos(πt²/2) dt
///
/// Returns (S, C) as a pair of real scalars.
///
/// Uses rational approximation for small z and asymptotic expansion for large z.
pub fn fresnel(z: f64) -> (f64, f64) {
    if z.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let ax = z.abs();
    if ax < 1e-15 {
        return (0.0, 0.0);
    }

    let (s, c) = if ax < 1.6 {
        fresnel_series(ax)
    } else if ax < 6.0 {
        fresnel_mid(ax)
    } else {
        fresnel_asymptotic(ax)
    };

    if z < 0.0 { (-s, -c) } else { (s, c) }
}

/// Power series for Fresnel integrals (small arguments).
fn fresnel_series(x: f64) -> (f64, f64) {
    fresnel_taylor(x)
}

/// Taylor series computation of Fresnel integrals.
fn fresnel_taylor(x: f64) -> (f64, f64) {
    let x2 = x * x;
    let t = std::f64::consts::FRAC_PI_2 * x2;

    // S(x) = x * Σ (-1)^n t^{2n+1} / ((2n+1)!(4n+3))  where t = πx²/2
    // C(x) = x * Σ (-1)^n t^{2n} / ((2n)!(4n+1))
    let mut s = 0.0;
    let mut c = 0.0;
    let mut s_term = t / 3.0; // n=0: t/(1!*3)
    let mut c_term = 1.0; // n=0: 1/(0!*1)

    s += s_term;
    c += c_term;

    for n in 1..60 {
        let nf = n as f64;
        c_term *= -t * t / ((2.0 * nf) * (2.0 * nf - 1.0));
        c_term *= (4.0 * nf - 3.0) / (4.0 * nf + 1.0);
        c += c_term;

        s_term *= -t * t / ((2.0 * nf + 1.0) * (2.0 * nf));
        s_term *= (4.0 * nf - 1.0) / (4.0 * nf + 3.0);
        s += s_term;

        if s_term.abs() < 1e-16 && c_term.abs() < 1e-16 {
            break;
        }
    }

    (x * s, x * c)
}

/// Asymptotic expansion for Fresnel integrals (large arguments).
fn fresnel_asymptotic(x: f64) -> (f64, f64) {
    // Asymptotic: S(x) ≈ 1/2 - f(x)cos(πx²/2) - g(x)sin(πx²/2)
    //             C(x) ≈ 1/2 + f(x)sin(πx²/2) - g(x)cos(πx²/2)

    // A&S 7.3.27/7.3.28 auxiliary functions (the previous coefficients were
    // wrong — (2n)(2n-1) instead of (4m-1)(4m-3) for f, (2n+1)(2n) instead of
    // (4m+1)(4m-1) for g — and g was scaled by πx² instead of π²x³, leaving
    // C off ~1e-3 at large x). frankenscipy-2fpck.
    //   f(x) = 1/(πx)  Σ_m (-1)^m (4m-1)!! / (πx²)^{2m}
    //   g(x) = 1/(π²x³) Σ_m (-1)^m (4m+1)!! / (πx²)^{2m}
    let pix = std::f64::consts::PI * x;
    let pix2 = pix * x; // π x²
    let half_pix2 = std::f64::consts::FRAC_PI_2 * x * x;
    let inv_sq = 1.0 / (pix2 * pix2); // 1/(πx²)²

    let mut f_term = 1.0;
    let mut g_term = 1.0;
    let mut f = f_term;
    let mut g = g_term;
    let mut f_prev = 1.0;
    let mut g_prev = 1.0;

    for m in 1..40 {
        let mf = m as f64;
        // (4m-1)!!/(4m-5)!! = (4m-1)(4m-3);  (4m+1)!!/(4m-3)!! = (4m+1)(4m-1).
        f_term *= -(4.0 * mf - 1.0) * (4.0 * mf - 3.0) * inv_sq;
        g_term *= -(4.0 * mf + 1.0) * (4.0 * mf - 1.0) * inv_sq;
        // Asymptotic series: stop once a term stops shrinking.
        if f_term.abs() > f_prev && g_term.abs() > g_prev {
            break;
        }
        f += f_term;
        g += g_term;
        f_prev = f_term.abs();
        g_prev = g_term.abs();
        if f_term.abs() < 1e-18 && g_term.abs() < 1e-18 {
            break;
        }
    }

    f /= pix; // 1/(πx)
    g /= pix2 * pix; // 1/(π²x³)

    let sin_t = half_pix2.sin();
    let cos_t = half_pix2.cos();

    let s = 0.5 - f * cos_t - g * sin_t;
    let c = 0.5 + f * sin_t - g * cos_t;
    (s, c)
}

/// Mid-range Fresnel integrals (bridge between series and asymptotic).
fn fresnel_mid(x: f64) -> (f64, f64) {
    // For moderate x, use numerical integration via composite Simpson
    let n = 200;
    let h = x / n as f64;
    let half_pi = std::f64::consts::FRAC_PI_2;

    let mut s = 0.0;
    let mut c = 0.0;

    // Simpson's rule
    for i in 0..=n {
        let t = i as f64 * h;
        let arg = half_pi * t * t;
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        s += w * arg.sin();
        c += w * arg.cos();
    }

    s *= h / 3.0;
    c *= h / 3.0;
    (s, c)
}

// ══════════════════════════════════════════════════════════════════════
// Dawson function
// ══════════════════════════════════════════════════════════════════════

/// Dawson function D(x) = exp(-x²) ∫₀ˣ exp(t²) dt.
///
/// Related to the imaginary error function: D(x) = √π/2 * exp(-x²) * erfi(x)
///
/// Uses Rybicki's algorithm with a Gaussian kernel series for efficiency.
fn dawsn_impl(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    let sign = x.signum();
    let ax = x.abs();

    // For very small x, the Taylor series D(x) ≈ x - 2x³/3 + 4x⁵/15 - ... is
    // exact to machine precision; Rybicki covers everything above it.
    if ax < 0.025 {
        let x2 = ax * ax;
        let result = ax
            * (1.0 - 2.0 * x2 / 3.0 + 4.0 * x2 * x2 / 15.0 - 8.0 * x2 * x2 * x2 / 105.0
                + 16.0 * x2.powi(4) / 945.0);
        return sign * result;
    }

    // Mid-range via Rybicki's method (machine-accurate); large x via the
    // asymptotic series. The previous fixed-step Simpson quadrature was only
    // ~1e-7. frankenscipy-p43m1.
    if ax < 6.25 {
        return sign * dawsn_mid(ax);
    }

    // Asymptotic: D(x) ≈ 1/(2x) + 1/(4x³) + 3/(8x⁵) + ...
    sign * dawsn_asymptotic(ax)
}

/// Dawson function for moderate |x| via Rybicki's method (Numerical Recipes
/// §6.10): D(x) = (1/√π) Σ_{n odd} e^{-(x-nh)²}/n, evaluated with the standard
/// folded summation around the nearest even node n₀. Machine-accurate for
/// 0.025 ≤ x < 6.25 (the Rybicki truncation error is ~exp(-(π/2H)²), negligible
/// at H = 0.25), replacing the ~1e-7 fixed-step Simpson quadrature.
fn dawsn_mid(x: f64) -> f64 {
    const H: f64 = 0.25;
    const NMAX: usize = 58;

    let n0 = 2 * (0.5 * x / H + 0.5) as i64;
    let xp = x - n0 as f64 * H;
    let mut e1 = (2.0 * xp * H).exp();
    let e2 = e1 * e1;
    let mut d1 = (n0 + 1) as f64;
    let mut d2 = d1 - 2.0;
    let mut sum = 0.0;

    for i in 1..=NMAX {
        let arg = (2 * i - 1) as f64 * H;
        let c = (-arg * arg).exp();
        sum += c * (e1 / d1 + 1.0 / (d2 * e1));
        d1 += 2.0;
        d2 -= 2.0;
        e1 *= e2;
    }

    (-xp * xp).exp() * sum / std::f64::consts::PI.sqrt()
}

/// Dawson function asymptotic expansion for large arguments.
fn dawsn_asymptotic(x: f64) -> f64 {
    let x2 = x * x;
    let inv_2x2 = 0.5 / x2;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut prev_abs = 1.0;

    for n in 1..60 {
        term *= (2 * n - 1) as f64 * inv_2x2;
        // Divergent asymptotic series — stop at the smallest term.
        if term.abs() > prev_abs {
            break;
        }
        sum += term;
        prev_abs = term.abs();
        if term.abs() < 1e-16 * sum.abs() {
            break;
        }
    }

    sum / (2.0 * x)
}

// ══════════════════════════════════════════════════════════════════════
// Sine and Cosine Integrals
// ══════════════════════════════════════════════════════════════════════

/// Sine integral Si(x) and cosine integral Ci(x).
///
/// Si(x) = ∫₀ˣ sin(t)/t dt
/// Ci(x) = γ + ln|x| + ∫₀ˣ (cos(t)-1)/t dt
///
/// where γ ≈ 0.5772... is the Euler-Mascheroni constant.
///
/// Matches `scipy.special.sici(x)` which returns (Si(x), Ci(x)).
///
/// # Arguments
/// * `x` - Real argument
///
/// # Returns
/// Tuple (Si(x), Ci(x))
pub fn sici(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    let ax = x.abs();

    // The power series cancels catastrophically as x grows and the classical
    // asymptotic bottoms out at its optimal-truncation floor (~1e-9 near x≈20),
    // so above a small threshold evaluate Si/Ci from the complex exponential
    // integral E₁(ix) via a modified-Lentz continued fraction, which stays
    // machine-accurate. frankenscipy-dyni8.
    let (si, ci) = if ax < 6.0 {
        sici_series(ax)
    } else {
        sici_cf(ax)
    };

    // Si(-x) = -Si(x), Ci(-x) = Ci(x) + i*π (we ignore imaginary part for real x > 0)
    if x < 0.0 { (-si, ci) } else { (si, ci) }
}

/// Power series for Si and Ci.
fn sici_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.5772156649015329;

    let x2 = x * x;

    // Si(x) = x - x³/(3·3!) + x⁵/(5·5!) - x⁷/(7·7!) + ...
    //       = Σ (-1)^n x^{2n+1} / ((2n+1)·(2n+1)!)
    let mut si = x;
    let mut si_term = x;

    for n in 1..150 {
        let nf = n as f64;
        si_term *= -x2 / ((2.0 * nf) * (2.0 * nf + 1.0));
        let contribution = si_term / (2.0 * nf + 1.0);
        si += contribution;
        if contribution.abs() < 1e-16 * si.abs() {
            break;
        }
    }

    // Ci(x) = γ + ln(x) + Σ (-1)^n x^{2n} / ((2n)·(2n)!) for n ≥ 1
    //       = γ + ln(x) - x²/(2·2!) + x⁴/(4·4!) - ...
    let mut ci = EULER_GAMMA + x.ln();
    let mut ci_term = 1.0;

    for n in 1..150 {
        let nf = n as f64;
        ci_term *= -x2 / ((2.0 * nf - 1.0) * (2.0 * nf));
        let contribution = ci_term / (2.0 * nf);
        ci += contribution;
        if contribution.abs() < 1e-16 * ci.abs().max(1.0) {
            break;
        }
    }

    (si, ci)
}

/// Si(x) and Ci(x) for x ≥ 6 via the complex exponential integral:
/// E₁(ix) = -Ci(x) + i(Si(x) - π/2) for x > 0, so Ci(x) = -Re E₁(ix) and
/// Si(x) = π/2 + Im E₁(ix). E₁ is evaluated with the modified-Lentz continued
/// fraction (Numerical Recipes §6.3) on the imaginary axis — machine-accurate
/// where the power series cancels and the real asymptotic stalls.
fn sici_cf(x: f64) -> (f64, f64) {
    const EPS: f64 = 1e-16;
    const FPMIN: f64 = 1e-300;
    const MAXIT: usize = 400;

    let one = Complex64::from_real(1.0);
    let tiny = Complex64::from_real(FPMIN);
    let z = Complex64::new(0.0, x);
    let mut b = z + one;
    let mut c = Complex64::from_real(1.0 / FPMIN);
    let mut d = b.recip();
    let mut h = d;
    for i in 1..=MAXIT {
        let a = Complex64::from_real(-((i * i) as f64)); // -i·(nm1+i), nm1=0
        b = b + Complex64::from_real(2.0);
        let mut den = a * d + b;
        if den.re == 0.0 && den.im == 0.0 {
            den = tiny;
        }
        d = den.recip();
        c = b + a / c;
        if c.re == 0.0 && c.im == 0.0 {
            c = tiny;
        }
        let del = c * d;
        h = h * del;
        if (del - one).abs() <= EPS {
            break;
        }
    }
    let e1 = h * (-z).exp();
    (std::f64::consts::FRAC_PI_2 + e1.im, -e1.re)
}


/// Hyperbolic sine integral Shi(x) and hyperbolic cosine integral Chi(x).
///
/// Shi(x) = ∫₀ˣ sinh(t)/t dt
/// Chi(x) = γ + ln|x| + ∫₀ˣ (cosh(t)-1)/t dt
///
/// where γ ≈ 0.5772... is the Euler-Mascheroni constant.
///
/// Matches `scipy.special.shichi(x)` which returns (Shi(x), Chi(x)).
///
/// # Arguments
/// * `x` - Real argument
///
/// # Returns
/// Tuple (Shi(x), Chi(x))
pub fn shichi(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    let ax = x.abs();

    // Shi and Chi are computed using series for all x
    // (they don't have the oscillatory behavior of Si/Ci)
    let (shi, chi) = shichi_series(ax);

    // Shi(-x) = -Shi(x), Chi(-x) = Chi(x)
    if x < 0.0 { (-shi, chi) } else { (shi, chi) }
}

/// Power series for Shi and Chi.
fn shichi_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.5772156649015329;

    let x2 = x * x;

    // Shi(x) = x + x³/(3·3!) + x⁵/(5·5!) + x⁷/(7·7!) + ...
    //        = Σ x^{2n+1} / ((2n+1)·(2n+1)!)
    let mut shi = x;
    let mut shi_term = x;

    for n in 1..80 {
        let nf = n as f64;
        shi_term *= x2 / ((2.0 * nf) * (2.0 * nf + 1.0));
        let contribution = shi_term / (2.0 * nf + 1.0);
        shi += contribution;
        if contribution.abs() < 1e-16 * shi.abs() {
            break;
        }
    }

    // Chi(x) = γ + ln(x) + Σ x^{2n} / ((2n)·(2n)!) for n ≥ 1
    //        = γ + ln(x) + x²/(2·2!) + x⁴/(4·4!) + ...
    let mut chi = EULER_GAMMA + x.ln();
    let mut chi_term = 1.0;

    for n in 1..80 {
        let nf = n as f64;
        chi_term *= x2 / ((2.0 * nf - 1.0) * (2.0 * nf));
        let contribution = chi_term / (2.0 * nf);
        chi += contribution;
        if contribution.abs() < 1e-16 * chi.abs().max(1.0) {
            break;
        }
    }

    (shi, chi)
}

// ══════════════════════════════════════════════════════════════════════
// Struve functions
// ══════════════════════════════════════════════════════════════════════

/// Struve function H_v(x) for integer order v.
///
/// H_v(x) = (x/2)^{v+1} Σ_{k=0}^∞ (-1)^k (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
///
/// Appears in electromagnetics and acoustics (e.g., radiation impedance).
pub fn struve(v: f64, x: f64) -> f64 {
    if x.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        // Leading term (x/2)^(v+1) / (Γ(3/2)·Γ(v+3/2)). Resolves
        // [frankenscipy-udtt9].
        if v > -1.0 {
            return 0.0;
        }
        if v == -1.0 {
            // 1 / (Γ(3/2) · Γ(1/2)) = 1 / ((√π/2)·√π) = 2/π
            return 2.0 / PI;
        }
        // v < -1: leading term diverges; sign depends on Γ(v+3/2)
        // which oscillates at half-integer poles. Return NaN rather
        // than guessing.
        return f64::NAN;
    }
    if x.abs() > 30.0 && v.abs() < x.abs() / 2.0 {
        return struve_asymptotic(v, x);
    }
    struve_series(v, x)
}

/// Modified Struve function L_v(x).
///
/// L_v(x) = -i * exp(-i*v*π/2) * H_v(ix) (for real x, this is real)
/// L_v(x) = (x/2)^{v+1} Σ_{k=0}^∞ (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
///
/// Note: same as Struve series but without the (-1)^k alternating sign.
pub fn modstruve(v: f64, x: f64) -> f64 {
    if x.is_nan() || v.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        // Same leading-term behavior as struve at x=0; the only
        // difference between H_v and L_v is the (-1)^k alternation
        // which doesn't affect the k=0 term.
        if v > -1.0 {
            return 0.0;
        }
        if v == -1.0 {
            return 2.0 / PI;
        }
        return f64::NAN;
    }
    modstruve_series(v, x)
}

/// Integral of the Struve function H_0 from 0 to x.
///
/// Matches `scipy.special.itstruve0`. Because H_0 is odd, SciPy's integral is
/// even in x: integrating from 0 to a negative endpoint returns the same value
/// as integrating to the positive endpoint.
pub fn itstruve0(x: f64) -> f64 {
    struve_integral_abs(x, |t| struve(0.0, t))
}

/// Integral of the modified Struve function L_0 from 0 to x.
///
/// Matches `scipy.special.itmodstruve0`; L_0 has the same odd symmetry as H_0
/// for real inputs, so the integral is even in x.
pub fn itmodstruve0(x: f64) -> f64 {
    struve_integral_abs(x, |t| modstruve(0.0, t))
}

/// Integral of H_0(t) / t from x to infinity.
///
/// Matches `scipy.special.it2struve0`. The identity
/// ∫₀∞ H_0(t)/t dt = π/2 lets us compute the finite correction from 0 to |x|
/// while preserving SciPy's negative-argument symmetry.
pub fn it2struve0(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }

    let correction = struve_integral_abs(x, struve0_over_t);
    if x.is_sign_negative() {
        std::f64::consts::FRAC_PI_2 + correction
    } else {
        std::f64::consts::FRAC_PI_2 - correction
    }
}

fn struve_integral_abs<F>(x: f64, integrand: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if !x.is_finite() {
        return f64::NAN;
    }
    let upper = x.abs();
    if upper == 0.0 {
        return 0.0;
    }

    let raw_steps = (256.0 * upper.max(1.0)).ceil() as usize;
    let steps = raw_steps.clamp(256, 32_768);
    let steps = steps + (steps % 2);
    let h = upper / steps as f64;

    let mut sum = integrand(0.0) + integrand(upper);
    for i in 1..steps {
        let t = i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * integrand(t);
    }
    sum * h / 3.0
}

fn struve0_over_t(t: f64) -> f64 {
    if t.abs() < 1.0e-12 {
        2.0 / PI
    } else {
        struve(0.0, t) / t
    }
}

/// Struve function via power series.
fn struve_series(v: f64, x: f64) -> f64 {
    let half_x = x / 2.0;
    let half_x_sq = half_x * half_x;

    // H_v(x) = (x/2)^{v+1} Σ (-1)^k (x/2)^{2k} / (Γ(k+3/2) Γ(k+v+3/2))
    let mut sum = 0.0;
    let mut term = 1.0 / (gamma_fn(1.5) * gamma_fn(v + 1.5));

    for k in 0..100 {
        sum += term;
        let kf = k as f64;
        term *= -half_x_sq / ((kf + 1.5) * (kf + v + 1.5));
        if term.abs() < 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }

    sum * half_x.powf(v + 1.0)
}

/// Modified Struve function via power series.
fn modstruve_series(v: f64, x: f64) -> f64 {
    let half_x = x / 2.0;
    let half_x_sq = half_x * half_x;

    let mut sum = 0.0;
    let mut term = 1.0 / (gamma_fn(1.5) * gamma_fn(v + 1.5));

    for k in 0..100 {
        sum += term;
        let kf = k as f64;
        term *= half_x_sq / ((kf + 1.5) * (kf + v + 1.5));
        if term.abs() < 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }

    sum * half_x.powf(v + 1.0)
}

/// Struve asymptotic expansion for large x.
fn struve_asymptotic(v: f64, x: f64) -> f64 {
    // DLMF 11.6.1: H_v(x) = Y_v(x) + (1/π) Σ_{k≥0} Γ(k+1/2)/Γ(v+1/2-k) (x/2)^{v-2k-1}.
    // Y_v from the Bessel routine (accurate large-x asymptotic); the correction
    // series is asymptotic (divergent) — sum to its smallest term. The previous
    // code kept only the leading term for v≈0 (≈0.17% error) and fell back to the
    // power series for v≠0, which catastrophically cancels at large x (struve(1,50)
    // was -3531 vs 0.58; struve(1,200) was -5.7e83 vs 0.65). frankenscipy-3z6wd.
    let yv = crate::bessel::yv_scalar(v, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let half_x = 0.5 * x;
    // k = 0 term: Γ(1/2)/Γ(v+1/2) (x/2)^{v-1}.
    let mut term = gamma_fn(0.5) / gamma_fn(v + 0.5) * half_x.powf(v - 1.0);
    let mut sum = term;
    let mut prev_abs = term.abs();
    let two_over_x_sq = (2.0 / x) * (2.0 / x);
    for k in 0..200 {
        let kf = k as f64;
        // term_{k+1} = term_k (k+1/2)(v-1/2-k)(2/x)².
        term *= (kf + 0.5) * (v - 0.5 - kf) * two_over_x_sq;
        let abs_term = term.abs();
        if abs_term > prev_abs {
            break; // past the smallest term of the asymptotic series
        }
        sum += term;
        prev_abs = abs_term;
        if abs_term <= 1.0e-18 * sum.abs() {
            break;
        }
    }
    yv + sum / PI
}

/// Simple gamma function for use in Struve computation.
fn gamma_fn(x: f64) -> f64 {
    // Use Lanczos approximation
    if x <= 0.0 && x.fract().abs() < 1e-14 {
        return f64::INFINITY;
    }
    if (x - 0.5).abs() < 1e-14 {
        return std::f64::consts::PI.sqrt();
    }
    if (x - 1.5).abs() < 1e-14 {
        return std::f64::consts::PI.sqrt() / 2.0;
    }
    if (x - 2.5).abs() < 1e-14 {
        return 3.0 * std::f64::consts::PI.sqrt() / 4.0;
    }

    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        return PI / ((PI * x).sin() * gamma_fn(1.0 - x));
    }

    let z = x - 1.0;
    let mut s = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        s += coeff / (z + idx as f64);
    }
    let t = z + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * s
}

// ══════════════════════════════════════════════════════════════════════
// Number Theory Functions
// ══════════════════════════════════════════════════════════════════════

/// Stirling number of the second kind S(n, k).
///
/// Counts partitions of `n` labeled items into `k` non-empty unlabeled subsets.
/// This is the default floating-output path for `scipy.special.stirling2(N, K)`.
pub fn stirling2(
    n_tensor: &SpecialTensor,
    k_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("stirling2", n_tensor, k_tensor, mode, |n, k| {
        stirling2_real_scalar(n, k, mode)
    })
}

fn stirling2_real_scalar(n: f64, k: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if !n.is_finite() || !k.is_finite() {
        return Err(SpecialError {
            function: "stirling2",
            kind: SpecialErrorKind::NonFiniteInput,
            mode,
            detail: "stirling2 inputs must be finite integers",
        });
    }
    if n.fract() != 0.0 || k.fract() != 0.0 {
        return Err(SpecialError {
            function: "stirling2",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "stirling2 inputs must be integers",
        });
    }
    if n < i64::MIN as f64 || n > i64::MAX as f64 || k < i64::MIN as f64 || k > i64::MAX as f64 {
        return Err(SpecialError {
            function: "stirling2",
            kind: SpecialErrorKind::OverflowRisk,
            mode,
            detail: "stirling2 integer inputs exceed i64 range",
        });
    }
    Ok(stirling2_scalar(n as i64, k as i64))
}

/// Scalar floating-output Stirling number of the second kind.
///
/// Negative inputs and `k > n` return zero, matching SciPy. Results overflow to
/// infinity naturally in the `f64` default-output regime.
#[must_use]
pub fn stirling2_scalar(n: i64, k: i64) -> f64 {
    if n < 0 || k < 0 || k > n {
        return 0.0;
    }
    if n == 0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if k == 0 {
        return 0.0;
    }
    if k == 1 || k == n {
        return 1.0;
    }

    let n = n as usize;
    let k = k as usize;
    let mut row = vec![0.0; k + 1];
    row[0] = 1.0;

    for i in 1..=n {
        let upper = k.min(i);
        for j in (1..=upper).rev() {
            row[j] = (j as f64).mul_add(row[j], row[j - 1]);
        }
        row[0] = 0.0;
    }

    row[k]
}

/// Bernoulli number B_n.
///
/// Returns the nth Bernoulli number. B_0=1, B_1=-1/2, B_2=1/6, B_3=0, ...
/// Odd Bernoulli numbers beyond B_1 are zero.
///
/// Matches `scipy.special.bernoulli(n)`.
pub fn bernoulli(n: u32) -> f64 {
    // Precomputed for small n
    match n {
        0 => 1.0,
        1 => -0.5,
        2 => 1.0 / 6.0,
        4 => -1.0 / 30.0,
        6 => 1.0 / 42.0,
        8 => -1.0 / 30.0,
        10 => 5.0 / 66.0,
        12 => -691.0 / 2730.0,
        14 => 7.0 / 6.0,
        16 => -3617.0 / 510.0,
        18 => 43867.0 / 798.0,
        20 => -174611.0 / 330.0,
        _ => {
            if n % 2 == 1 {
                return 0.0; // Odd Bernoulli numbers (n >= 3) are zero
            }
            // Use the relationship with zeta function:
            // B_{2n} = (-1)^{n+1} * 2 * (2n)! / (2π)^{2n} * ζ(2n)
            let nf = n as f64;
            let sign = if (n / 2) & 1 == 0 { -1.0 } else { 1.0 };
            let zeta_val = crate::gamma::zeta(nf);
            sign * 2.0 * gamma_fn(nf + 1.0) / (2.0 * PI).powf(nf) * zeta_val
        }
    }
}

/// Euler number E_n.
///
/// Returns the nth Euler number. E_0=1, E_1=0, E_2=-1, E_3=0, E_4=5, ...
/// Odd Euler numbers are zero.
///
/// Matches `scipy.special.euler(n)`.
pub fn euler(n: u32) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    match n {
        0 => 1.0,
        2 => -1.0,
        4 => 5.0,
        6 => -61.0,
        8 => 1385.0,
        10 => -50521.0,
        12 => 2702765.0,
        14 => -199360981.0,
        _ => {
            // Compute via alternating sum formula or recurrence
            // E_{2n} = -Σ_{k=0}^{n-1} C(2n, 2k) E_{2k}
            let mut e = vec![0.0; (n / 2 + 1) as usize];
            e[0] = 1.0;
            for m in 1..=(n / 2) as usize {
                let mut sum = 0.0;
                for (k, &ek) in e.iter().enumerate().take(m) {
                    sum += comb_f64(2 * m as u64, 2 * k as u64) * ek;
                }
                e[m] = -sum;
            }
            e[(n / 2) as usize]
        }
    }
}

/// Hurwitz zeta function ζ(s, a) = Σ_{n=0}^∞ 1/(n+a)^s.
///
/// Generalizes the Riemann zeta function: ζ(s) = ζ(s, 1).
///
/// Matches `scipy.special.zeta(s, a)` (the two-argument form).
///
/// Uses Euler-Maclaurin summation:
///
/// ```text
///   ζ(s, a) ≈ Σ_{n=0}^{N-1} (n+a)^{-s}
///           + (N+a)^{1-s} / (s-1)               [integral tail]
///           - (1/2) (N+a)^{-s}                  [half-term]
///           + (s/12) (N+a)^{-s-1}               [B_2 correction]
///           - s(s+1)(s+2)/720 · (N+a)^{-s-3}    [B_4 correction]
///           + s..(s+4)/30240 · (N+a)^{-s-5}     [B_6 correction]
///           - s..(s+6)/1209600 · (N+a)^{-s-7}   [B_8 correction]
/// ```
///
/// Direct sum to N=20 with four Bernoulli terms gives ~1e-13 absolute
/// accuracy down to s≈1.1 (the previous implementation, which used a
/// 10 000-term direct sum and only the integral tail, missed the
/// half-term and Bernoulli corrections and so floored at ~1e-5 abs at
/// s=1.1) — frankenscipy-3u8ze.
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    if s.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 {
        // A nonpositive-integer a hits a pole (some a+k = 0) => +inf (scipy).
        if a == a.floor() {
            return f64::INFINITY;
        }
        // For negative non-integer a, scipy.special.zeta(s, a) is real only when s
        // is an integer (so the negative-base terms (a+j)^{-s} stay real), via the
        // shift recurrence ζ(s,a) = Σ_{j<m} (a+j)^{-s} + ζ(s, a+m), a+m ∈ (0,1].
        // Non-integer s with a<0 is NaN in scipy; reproduce that.
        if s != s.round() || s <= 1.0 {
            return f64::NAN;
        }
        let exp = -(s as i64);
        if !(i32::MIN as i64..=i32::MAX as i64).contains(&exp) {
            return f64::NAN;
        }
        let exp = exp as i32;
        let m = (-a).ceil() as i64;
        let mut acc = 0.0;
        for j in 0..m {
            acc += (a + j as f64).powi(exp);
        }
        return acc + hurwitz_zeta(s, a + m as f64);
    }
    if s <= 1.0 {
        return f64::INFINITY; // Pole at s=1
    }

    let n_direct: usize = 20;
    let mut sum = 0.0;
    for k in 0..n_direct {
        sum += (k as f64 + a).powf(-s);
    }

    let na = n_direct as f64 + a;
    let na_inv_sq = 1.0 / (na * na);

    // (N+a)^{-s} computed via powf for accuracy across the supported
    // s range; subsequent factors are obtained by multiplying by 1/na².
    let na_neg_s = na.powf(-s);

    // Integral tail: (N+a)^{1-s} / (s-1) = (N+a) · (N+a)^{-s} / (s-1).
    sum += na * na_neg_s / (s - 1.0);
    // Half-term: +(1/2) · (N+a)^{-s}. Direct sum covers k = 0..N-1,
    // the Euler-Maclaurin tail Σ_{k=N}^∞ f(k+a) starts at k = N,
    // so the half-term sits on the included left boundary y = N+a.
    sum += 0.5 * na_neg_s;

    // Bernoulli corrections. `term` tracks (N+a)^{-s-(2j-1)} for j=1,2,…
    // and `poch` tracks the Pochhammer symbol [s]_{2j-1}.
    let mut term = na_neg_s / na;
    let mut poch = s;
    sum += (1.0 / 12.0) * poch * term; // j=1
    term *= na_inv_sq;
    poch *= (s + 1.0) * (s + 2.0);
    sum -= (1.0 / 720.0) * poch * term; // j=2
    term *= na_inv_sq;
    poch *= (s + 3.0) * (s + 4.0);
    sum += (1.0 / 30240.0) * poch * term; // j=3
    term *= na_inv_sq;
    poch *= (s + 5.0) * (s + 6.0);
    sum -= (1.0 / 1209600.0) * poch * term; // j=4

    sum
}

fn comb_f64(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

fn map_real<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_type",
                "fail_closed",
                "unsupported input type for convenience scalar function",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type",
            })
        }
    }
}

fn map_real_or_complex<F, G>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError>,
    G: Fn(Complex64) -> Result<Complex64, SpecialError>,
{
    match input {
        SpecialTensor::RealScalar(x) => real_kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(real_kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        SpecialTensor::ComplexScalar(value) => {
            complex_kernel(*value).map(SpecialTensor::ComplexScalar)
        }
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(complex_kernel)
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        SpecialTensor::Empty => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid special-function input",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid special-function input",
            })
        }
    }
}

fn map_real_binary<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError>,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => left
            .iter()
            .copied()
            .map(|value| kernel(value, *right))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => right
            .iter()
            .copied()
            .map(|value| kernel(*left, value))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::RealVec),
        (SpecialTensor::RealVec(left), SpecialTensor::RealVec(right)) => {
            if left.len() != right.len() {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("lhs_len={},rhs_len={}", left.len(), right.len()),
                    "fail_closed",
                    "vector inputs must have matching lengths",
                    false,
                );
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            left.iter()
                .copied()
                .zip(right.iter().copied())
                .map(|(l, r)| kernel(l, r))
                .collect::<Result<Vec<_>, _>>()
                .map(SpecialTensor::RealVec)
        }
        _ => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "unsupported_types",
                "fail_closed",
                "unsupported input type combination for binary convenience function",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type combination",
            })
        }
    }
}

fn map_real_ternary<F>(
    function: &'static str,
    first: &SpecialTensor,
    second: &SpecialTensor,
    third: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError>,
{
    fn broadcast_len(tensor: &SpecialTensor) -> Option<usize> {
        match tensor {
            SpecialTensor::RealScalar(_) => Some(1),
            SpecialTensor::RealVec(values) => Some(values.len()),
            _ => None,
        }
    }

    fn broadcast_value(tensor: &SpecialTensor, idx: usize) -> Option<f64> {
        match tensor {
            SpecialTensor::RealScalar(value) => Some(*value),
            SpecialTensor::RealVec(values) => {
                if values.is_empty() {
                    None
                } else if values.len() == 1 {
                    Some(values[0])
                } else {
                    values.get(idx).copied()
                }
            }
            _ => None,
        }
    }

    let Some(first_len) = broadcast_len(first) else {
        record_special_trace(
            function,
            mode,
            "domain_error",
            "unsupported_first_input",
            "fail_closed",
            "unsupported input type for ternary convenience function",
            false,
        );
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        });
    };
    let Some(second_len) = broadcast_len(second) else {
        record_special_trace(
            function,
            mode,
            "domain_error",
            "unsupported_second_input",
            "fail_closed",
            "unsupported input type for ternary convenience function",
            false,
        );
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        });
    };
    let Some(third_len) = broadcast_len(third) else {
        record_special_trace(
            function,
            mode,
            "domain_error",
            "unsupported_third_input",
            "fail_closed",
            "unsupported input type for ternary convenience function",
            false,
        );
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        });
    };

    let target_len = first_len.max(second_len).max(third_len);
    if ![first_len, second_len, third_len]
        .into_iter()
        .all(|len| len == 1 || len == target_len)
    {
        record_special_trace(
            function,
            mode,
            "domain_error",
            format!("first_len={first_len},second_len={second_len},third_len={third_len}"),
            "fail_closed",
            "vector inputs must be broadcast-compatible",
            false,
        );
        return Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "vector inputs must be broadcast-compatible",
        });
    }

    if target_len == 1 {
        let first_value = broadcast_value(first, 0).ok_or(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        })?;
        let second_value = broadcast_value(second, 0).ok_or(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        })?;
        let third_value = broadcast_value(third, 0).ok_or(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "unsupported input type",
        })?;
        return kernel(first_value, second_value, third_value).map(SpecialTensor::RealScalar);
    }

    (0..target_len)
        .map(|idx| {
            let first_value = broadcast_value(first, idx).ok_or(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type",
            })?;
            let second_value = broadcast_value(second, idx).ok_or(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type",
            })?;
            let third_value = broadcast_value(third, idx).ok_or(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "unsupported input type",
            })?;
            kernel(first_value, second_value, third_value)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(SpecialTensor::RealVec)
}

// ══════════════════════════════════════════════════════════════════════
// Kelvin Functions
// ══════════════════════════════════════════════════════════════════════

/// Kelvin function ber(x): real part of J_0(x * sqrt(j)).
///
/// Matches `scipy.special.ber`.
pub fn ber(x: f64) -> f64 {
    // ber(x) = Re[J_0(x · e^{3πi/4})]. The ascending series Σ(-1)^k(x/2)^{4k}/((2k)!)²
    // has intermediate terms ~e^{x/√2} while |ber| ~ e^{x/√2}/√(2πx); for |x| ≳ 130
    // that loses >16 digits to cancellation (ber(150) was ~1e14× too large). The
    // exact complex Bessel J_0 has no such cancellation. frankenscipy-hsjhp.
    if x.abs() >= 80.0 {
        return kelvin_ber_bei_complex(x).re;
    }
    let x2 = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..50 {
        term *= -x2 * x2 / ((2 * k - 1) as f64 * (2 * k) as f64).powi(2);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}

/// ber(x) + i·bei(x) = J_0(x · e^{3πi/4}), via the exact complex Bessel J_0
/// (cancellation-free at large x where the ascending Kelvin series fails).
fn kelvin_ber_bei_complex(x: f64) -> Complex64 {
    let ax = x.abs();
    let arg = 3.0 * PI / 4.0;
    let z = Complex64::new(ax * arg.cos(), ax * arg.sin());
    crate::bessel::complex_jv_scalar(0.0, z)
}

/// Kelvin function derivative ber'(x).
///
/// Matches `scipy.special.berp`.
pub fn berp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    kelvin_derivatives(x).0
}

/// Kelvin function bei(x): imaginary part of J_0(x * sqrt(j)).
///
/// Matches `scipy.special.bei`.
pub fn bei(x: f64) -> f64 {
    // bei(x) = Im[J_0(x * e^{jπ/4})] = Σ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2 ... no
    // Actually: bei(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{4k+2} / ((2k)!(2k+1)!) ... no
    // Correct series: bei(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2
    // Wait, let me re-derive. J_0(z) = Σ (-1)^n (z/2)^{2n} / (n!)^2
    // With z = x*sqrt(j) = x*e^{jπ/4}:
    // (z/2)^{2n} = (x/2)^{2n} * e^{jnπ/2}
    // e^{jnπ/2} cycles: 1, j, -1, -j, 1, ...
    // J_0(x*e^{jπ/4}) = Σ (-1)^n (x/2)^{2n} / (n!)^2 * e^{jnπ/2}
    //
    // Real parts (n mod 4 == 0: factor 1, n mod 4 == 2: factor -1):
    // ber(x) = Σ_{k=0} (x/2)^{4k}/(2k)!^2 - (x/2)^{4k+2}/((2k+1)!)^2 ... hmm not quite
    //
    // Let me just use: n=0: e^0=1 real, n=1: e^{jπ/2}=j imag, n=2: e^{jπ}=-1 real, n=3: e^{j3π/2}=-j imag
    // So (-1)^n * e^{jnπ/2}: n=0: 1, n=1: -j, n=2: 1, n=3: j, n=4: 1, ...
    // Wait: (-1)^0 * e^0 = 1, (-1)^1 * e^{jπ/2} = -j, (-1)^2 * e^{jπ} = -1, (-1)^3 * e^{j3π/2} = j
    // Hmm that gives: real parts at n=0: 1, n=2: -1, n=4: 1, n=6: -1, ...
    // And: imag parts at n=1: -1, n=3: 1, n=5: -1, n=7: 1, ...
    //
    // ber(x) = 1 - (x/2)^4/(2!)^2 + (x/2)^8/(4!)^2 - ... = Σ_{k=0} (-1)^k (x/2)^{4k} / ((2k)!)^2
    // scipy.special.bei uses the positive-leading Kelvin convention:
    // bei(x) = (x/2)^2 - (x/2)^6/(3!)^2 + (x/2)^10/(5!)^2 - ...
    //        = Σ (-1)^k (x/2)^{4k+2} / ((2k+1)!)^2

    // Large |x|: the ascending series cancels catastrophically; use the exact
    // complex Bessel J_0 (bei = Im). frankenscipy-hsjhp.
    if x.abs() >= 80.0 {
        return kelvin_ber_bei_complex(x).im;
    }

    let x2 = x * x / 4.0; // (x/2)^2
    let mut term = x2;
    let mut sum = term;
    for k in 1..50 {
        // Ratio: next/current = -(x/2)^4 / ((2k+1) * (2k))^2 ... let me compute:
        // term_k = (-1)^{k+1} (x/2)^{4k+2} / ((2k+1)!)^2
        // term_{k+1}/term_k = -1 * (x/2)^4 / ((2k+2)*(2k+3))^2
        term *= -x2 * x2 / ((2 * k) as f64 * (2 * k + 1) as f64).powi(2);
        sum += term;
        if term.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}

/// Kelvin function derivative bei'(x).
///
/// Matches `scipy.special.beip`.
pub fn beip(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    kelvin_derivatives(x).1
}

/// Kelvin function ker(x): real part of K_0(x * sqrt(j)).
///
/// Matches `scipy.special.ker`.
/// (ker(x), kei(x)) for large x from the Kelvin identity
/// ker(x) + i·kei(x) = K₀(x e^{iπ/4}), evaluated with the complex DLMF 10.40.2
/// asymptotic K₀(z) ~ √(π/(2z)) e^{-z} Σ_k a_k/z^k, a₀ = 1,
/// a_k = a_{k-1}·(-(2k-1)²)/(8k) (the ν=0 coefficients). Summed to its smallest
/// term (asymptotic/divergent). This avoids the catastrophic cancellation of the
/// log/harmonic series, which loses ~x√2/ln(10) digits at large x.
fn kelvin_ker_kei_asymptotic(x: f64) -> (f64, f64) {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};
    let z = Complex64::new(x * FRAC_1_SQRT_2, x * FRAC_1_SQRT_2); // x e^{iπ/4}
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;
    let mut prev_abs = 1.0;
    for k in 1..60 {
        let kf = k as f64;
        let coef = -(2.0 * kf - 1.0).powi(2) / (8.0 * kf);
        term = term * Complex64::from_real(coef) / z;
        let a = term.abs();
        if a > prev_abs {
            break; // past the smallest term of the asymptotic series
        }
        sum = sum + term;
        prev_abs = a;
        if a < 1e-18 {
            break;
        }
    }
    let pref = Complex64::from_real((PI / 2.0).sqrt()) * z.powc(Complex64::from_real(-0.5));
    let k0 = pref * (-z).exp() * sum;
    (k0.re, k0.im)
}

/// I₁(z) for complex z via the ascending series Σ_{k≥0} (z/2)^{2k+1}/(k!(k+1)!).
fn complex_i1_series(z: Complex64) -> Complex64 {
    let half = z * Complex64::from_real(0.5);
    let h2 = half * half;
    let mut term = half; // k = 0
    let mut sum = Complex64::from_real(0.0);
    for k in 0..120u32 {
        sum = sum + term;
        let kf = k as f64;
        term = term * h2 / Complex64::from_real((kf + 1.0) * (kf + 2.0));
        if term.abs() < 1e-20 * sum.abs() {
            break;
        }
    }
    sum
}

/// I₁(z) via the large-|z| asymptotic e^z/√(2πz) Σ_k (-1)^k a_k/z^k (ν=1).
fn complex_i1_asymptotic(z: Complex64) -> Complex64 {
    let mu = 4.0;
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;
    let mut prev = 1.0;
    for k in 1..80u32 {
        let kf = k as f64;
        let coef = -(mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf);
        term = term * Complex64::from_real(coef) / z;
        let a = term.abs();
        if a > prev {
            break;
        }
        sum = sum + term;
        prev = a;
        if a < 1e-18 {
            break;
        }
    }
    let inv_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().recip();
    z.exp() * z.powc(Complex64::from_real(-0.5)) * Complex64::from_real(inv_sqrt_2pi) * sum
}

/// K₁(z) via the DLMF 10.40.2 asymptotic √(π/(2z)) e^{-z} Σ_k a_k/z^k (ν=1, the
/// all-positive coefficients (4-(2k-1)²)/(8k)).
fn complex_k1_asymptotic(z: Complex64) -> Complex64 {
    let mu = 4.0;
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;
    let mut prev = 1.0;
    for k in 1..80u32 {
        let kf = k as f64;
        let coef = (mu - (2.0 * kf - 1.0).powi(2)) / (8.0 * kf);
        term = term * Complex64::from_real(coef) / z;
        let a = term.abs();
        if a > prev {
            break;
        }
        sum = sum + term;
        prev = a;
        if a < 1e-18 {
            break;
        }
    }
    let pref = Complex64::from_real((std::f64::consts::PI / 2.0).sqrt())
        * z.powc(Complex64::from_real(-0.5));
    pref * (-z).exp() * sum
}

/// K₁(z) via the DLMF 10.31.1 logarithmic series (n = 1):
/// K₁(z) = 1/z + ln(z/2) I₁(z) - (z/4) Σ_k (ψ(k+1)+ψ(k+2)) (z²/4)^k/(k!(k+1)!).
fn complex_k1_series(z: Complex64) -> Complex64 {
    const EULER: f64 = 0.577_215_664_901_532_9;
    let i1 = complex_i1_series(z);
    let half = z * Complex64::from_real(0.5);
    let h2 = half * half;
    let mut sum = Complex64::from_real(0.0);
    let mut tk = Complex64::from_real(1.0); // (z²/4)^k/(k!(k+1)!), k = 0
    let mut h_k = 0.0; // H_k
    let mut h_k1 = 1.0; // H_{k+1}
    for k in 0..120u32 {
        let coef = -2.0 * EULER + h_k + h_k1; // ψ(k+1)+ψ(k+2)
        sum = sum + tk * Complex64::from_real(coef);
        let kf = k as f64;
        tk = tk * h2 / Complex64::from_real((kf + 1.0) * (kf + 2.0));
        h_k += 1.0 / (kf + 1.0);
        h_k1 += 1.0 / (kf + 2.0);
        if tk.abs() < 1e-20 * sum.abs().max(1e-300) {
            break;
        }
    }
    z.recip() + half.ln() * i1 - (z * Complex64::from_real(0.25)) * sum
}

/// (ber'(x), bei'(x), ker'(x), kei'(x)) from the analytic Kelvin-derivative
/// identities ber'+i·bei' = e^{iπ/4} I₁(z) and ker'+i·kei' = -e^{iπ/4} K₁(z),
/// z = x e^{iπ/4}. I₁/K₁ use complex series (small x) and asymptotics (large x).
/// Replaces the finite-difference / cancelling direct-series forms that were
/// ~1e-8..4e-6 off scipy. frankenscipy-l3kwr.
fn kelvin_derivatives(x: f64) -> (f64, f64, f64, f64) {
    use std::f64::consts::FRAC_1_SQRT_2;
    let rot = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2); // e^{iπ/4}
    let z = Complex64::new(x * FRAC_1_SQRT_2, x * FRAC_1_SQRT_2);
    let i1 = if x < 20.0 {
        complex_i1_series(z)
    } else {
        complex_i1_asymptotic(z)
    };
    let k1 = if x < 11.0 {
        complex_k1_series(z)
    } else {
        complex_k1_asymptotic(z)
    };
    let bb = rot * i1; // ber' + i·bei'
    let kk = -(rot * k1); // ker' + i·kei'
    (bb.re, bb.im, kk.re, kk.im)
}

pub fn ker(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    // Large x: the log/harmonic series below cancels catastrophically (ber/bei
    // grow like e^{x/√2} while ker decays like e^{-x/√2}), so use the complex
    // K₀ asymptotic. frankenscipy-rhilt.
    if x >= 11.0 {
        return kelvin_ker_kei_asymptotic(x).0;
    }
    // For small x, use the series representation:
    // ker(x) = -(ln(x/2) + γ) * ber(x) + (π/4) * bei(x) + Σ h(k) terms
    // where γ is the Euler-Mascheroni constant.
    //
    // For simplicity, use numerical integration of the integral representation:
    // ker(x) = ∫_0^∞ cos(x*sinh(t) - x*cosh(t)·something) dt (complex)
    //
    // Actually, use the relation: ker(x) + j·kei(x) = K_0(x·e^{jπ/4})
    // And K_0 can be computed from the series for small arguments.
    //
    // K_0(z) = -(ln(z/2) + γ) I_0(z) + Σ_{k=0}^∞ (z/2)^{2k} ψ(k+1) / (k!)^2
    // where ψ is the digamma function and ψ(1) = -γ, ψ(k+1) = -γ + Σ_{j=1}^k 1/j

    let gamma_em = 0.577_215_664_901_532_9;
    let x_half = x / 2.0;
    let ln_x2 = x_half.ln();

    // Compute ber(x) and bei(x) for the log term
    let ber_x = ber(x);
    let bei_x = bei(x);

    // ker(x) = -(ln(x/2) + γ) * ber(x) + (π/4) * bei(x) + series_correction
    // The series correction involves harmonic numbers.
    let x2 = x_half * x_half;
    let mut term = 1.0; // (x/2)^0 / (0!)^2 = 1
    let mut harmonic = 0.0; // H_0 = 0
    let mut correction = 0.0; // ψ(1) = -γ, but first term has k=0

    for k in 0..50 {
        if k > 0 {
            term *= x2 * x2 / ((2 * k - 1) as f64 * (2 * k) as f64).powi(2);
            harmonic += 1.0 / (2 * k - 1) as f64 + 1.0 / (2 * k) as f64;
        }
        // The correction uses (H_{2k} from the real part of ψ contributions)
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        correction += sign * term * harmonic;
    }

    -(ln_x2 + gamma_em) * ber_x + (std::f64::consts::PI / 4.0) * bei_x + correction
}

/// Kelvin function kei(x): imaginary part of K_0(x * sqrt(j)).
///
/// Matches `scipy.special.kei`.
pub fn kei(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return -std::f64::consts::PI / 4.0; // kei(0) = -π/4
    }
    if x >= 11.0 {
        return kelvin_ker_kei_asymptotic(x).1;
    }
    // kei(x) = -(ln(x/2) + γ) * bei(x) - (π/4) * ber(x) + series_correction

    let gamma_em = 0.577_215_664_901_532_9;
    let x_half = x / 2.0;
    let ln_x2 = x_half.ln();

    let ber_x = ber(x);
    let bei_x = bei(x);

    let x2 = x_half * x_half;
    let mut term = -x2; // first imaginary series term
    let mut harmonic = 1.0; // H_1 = 1
    let mut correction = term * harmonic;

    for k in 1..50 {
        term *= -x2 * x2 / ((2 * k) as f64 * (2 * k + 1) as f64).powi(2);
        harmonic += 1.0 / (2 * k) as f64 + 1.0 / (2 * k + 1) as f64;
        correction += term * harmonic;
    }

    -(ln_x2 + gamma_em) * bei_x - (std::f64::consts::PI / 4.0) * ber_x - correction
}

/// Kelvin function derivative ker'(x).
///
/// Matches `scipy.special.kerp` on the real domain.
pub fn kerp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    kelvin_derivatives(x).2
}

/// Kelvin function derivative kei'(x).
///
/// Matches `scipy.special.keip` on the real domain.
pub fn keip(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    kelvin_derivatives(x).3
}

/// Combined Kelvin functions `(Be, Ke, Be', Ke')`.
///
/// Matches `scipy.special.kelvin`, where `Be = ber + i bei` and
/// `Ke = ker + i kei`.
pub fn kelvin(x: f64) -> (Complex64, Complex64, Complex64, Complex64) {
    (
        Complex64::new(ber(x), bei(x)),
        Complex64::new(ker(x), kei(x)),
        Complex64::new(berp(x), beip(x)),
        Complex64::new(kerp(x), keip(x)),
    )
}

// ══════════════════════════════════════════════════════════════════════
// Exponential Integral Variants
// ══════════════════════════════════════════════════════════════════════

/// Generalized exponential integral E_n(x) = ∫_1^∞ t^{-n} e^{-xt} dt.
///
/// Matches `scipy.special.expn`.
pub fn expn(n: usize, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if n > 1 {
            1.0 / (n as f64 - 1.0)
        } else {
            f64::INFINITY
        };
    }
    if n == 0 {
        return (-x).exp() / x;
    }

    // E_n(x) via Numerical Recipes §6.3 `expint`: modified-Lentz continued
    // fraction for x > 1, power series for x ≤ 1, each with a real convergence
    // test. The previous E_1 path used a fixed 20-level recurrence with no
    // convergence check — ~7e-8 off at x≈1, which propagated to expi(-x) and
    // the upward E_n recurrence. frankenscipy-inkqr.
    const EULER: f64 = 0.577_215_664_901_532_9;
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;
    const MAXIT: usize = 200;
    let nm1 = n as f64 - 1.0;

    if x > 1.0 {
        let mut b = x + n as f64;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAXIT {
            let a = -(i as f64) * (nm1 + i as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() <= EPS {
                break;
            }
        }
        h * (-x).exp()
    } else {
        let mut ans = if nm1 != 0.0 { 1.0 / nm1 } else { -x.ln() - EULER };
        let mut fact = 1.0;
        for i in 1..=MAXIT {
            fact *= -x / i as f64;
            let del = if (i as f64 - nm1).abs() > 0.5 {
                -fact / (i as f64 - nm1)
            } else {
                // i == n-1: the logarithmic term.
                let mut psi = -EULER;
                for ii in 1..=(nm1 as usize) {
                    psi += 1.0 / ii as f64;
                }
                fact * (-x.ln() + psi)
            };
            ans += del;
            if del.abs() < ans.abs() * EPS {
                break;
            }
        }
        ans
    }
}

/// Exponential integral Ei(x) = PV ∫_{-∞}^{x} e^t/t dt (scalar version).
///
/// Matches `scipy.special.expi` for scalar inputs.
pub fn expi_scalar(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -expn(1, -x);
    }
    // The convergent series Ei(x) = γ + ln x + Σ_{k≥1} x^k/(k·k!) has its
    // largest term near k≈x, so a fixed 200-term cap truncates mid-ascent and
    // returns garbage once x≳200 (expi(500) was ~0 vs 2.8e214). For large x use
    // the divergent asymptotic Ei(x) ~ (e^x/x) Σ_{k≥0} k!/x^k with optimal
    // truncation (stop before the terms start growing); it is machine-accurate
    // for x ≥ 40 (smallest term ~ e^{-x}). e^x overflows to +inf past x≈709.78,
    // matching scipy's overflow→+inf.
    if x >= 40.0 {
        let inv_x = 1.0 / x;
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        let mut prev = f64::INFINITY;
        for k in 1..1000 {
            term *= k as f64 * inv_x;
            if term > prev {
                break; // asymptotic series is divergent: truncate at the smallest term
            }
            sum += term;
            prev = term;
        }
        return sum * x.exp() / x;
    }
    // For 0 < x < 40, the convergent series reaches its tail well within 200 terms.
    let gamma_em = 0.577_215_664_901_532_9;
    let mut sum = gamma_em + x.ln();
    let mut term = x;
    sum += term;
    for k in 2..200 {
        term *= x / k as f64;
        let contrib = term / k as f64;
        sum += contrib;
        if contrib.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum
}

// ══════════════════════════════════════════════════════════════════════
// Polygamma Functions
// ══════════════════════════════════════════════════════════════════════

/// Trigamma function ψ₁(x) = d²ln(Γ(x))/dx².
///
/// Matches `scipy.special.polygamma(1, x)`.
pub fn trigamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        return (pi * pi) / (sin_pi_x * sin_pi_x) - trigamma(1.0 - x);
    }

    let mut val = x;
    let mut result = 0.0;
    while val < 8.0 {
        result += 1.0 / (val * val);
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    result += inv_x + inv_x2 / 2.0 + inv_x2 * inv_x / 6.0 - inv_x2 * inv_x2 * inv_x / 30.0
        + inv_x2 * inv_x2 * inv_x2 * inv_x / 42.0;

    result
}

/// Pentagamma function ψ₃(x) = d⁴ln(Γ(x))/dx⁴.
///
/// Matches `scipy.special.polygamma(3, x)`. Same shift-then-asymptotic
/// structure as `tetragamma`, with the recurrence
///   ψ₃(x) = ψ₃(x + 1) + 6 / x⁴
/// applied until x ≥ 8 and the asymptotic series
///   ψ₃(x) ≈ 2/x³ + 3/x⁴ + 2/x⁵ − 1/x⁷ + 4/(3 x⁹)
/// (from B_2 = 1/6, B_4 = −1/30, B_6 = 1/42) truncated thereafter.
pub fn pentagamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        let s = (pi * x).sin();
        let c = (pi * x).cos();
        // d/dx [-π³ cos(πx)/sin³(πx)] · 2 — directly differentiate the
        // tetragamma reflection. Closed form for the reflection of ψ₃
        // is:  ψ₃(1 − x) − π⁴ (2 + 4 cos²(πx)) / sin⁴(πx).
        let s2 = s * s;
        return pentagamma(1.0 - x)
            - pi * pi * pi * pi * 2.0_f64.mul_add(c * c, 1.0) * 2.0 / (s2 * s2);
    }

    let mut val = x;
    let mut result = 0.0;
    while val < 8.0 {
        let v2 = val * val;
        result += 6.0 / (v2 * v2);
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    let inv_x5 = inv_x3 * inv_x2;
    let inv_x7 = inv_x5 * inv_x2;
    let inv_x9 = inv_x7 * inv_x2;
    result += 2.0 * inv_x3 + 3.0 * inv_x2 * inv_x2 + 2.0 * inv_x5 - inv_x7 + 4.0 / 3.0 * inv_x9;

    result
}

/// Tetragamma function ψ₂(x) = d³ln(Γ(x))/dx³.
///
/// Matches `scipy.special.polygamma(2, x)`.
pub fn tetragamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_x = (pi * x).sin();
        let cos_pi_x = (pi * x).cos();
        return tetragamma(1.0 - x)
            - 2.0 * pi * pi * pi * cos_pi_x / (sin_pi_x * sin_pi_x * sin_pi_x);
    }

    let mut val = x;
    let mut result = 0.0;
    while val < 8.0 {
        result -= 2.0 / (val * val * val);
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    let inv_x3 = inv_x2 * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let inv_x6 = inv_x4 * inv_x2;
    let inv_x8 = inv_x6 * inv_x2;
    let inv_x10 = inv_x8 * inv_x2;
    let inv_x12 = inv_x10 * inv_x2;
    // ψ''(x) ~ -1/x² - 1/x³ - 1/(2x⁴) + 1/(6x⁶) - 1/(6x⁸) + 3/(10x¹⁰) - 5/(6x¹²),
    // extended through the B₁₀ term (was truncated at x⁶, ~6e-9 residual at the
    // shift point). frankenscipy-luxsz.
    result += -inv_x2 - inv_x3 - inv_x4 / 2.0 + inv_x6 / 6.0 - inv_x8 / 6.0
        + 3.0 * inv_x10 / 10.0
        - 5.0 * inv_x12 / 6.0;

    result
}

/// Digamma function ψ(x) = d(ln Γ(x))/dx (scalar).
pub fn digamma_scalar(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }

    let mut val = x;
    let mut result = 0.0;

    if val < 0.0 {
        result -= std::f64::consts::PI / (std::f64::consts::PI * val).tan();
        val = 1.0 - val;
    }

    while val < 8.0 {
        result -= 1.0 / val;
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    result += val.ln() - inv_x / 2.0 - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0;

    result
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial & Utility Functions
// ══════════════════════════════════════════════════════════════════════

/// Rising factorial (Pochhammer symbol): (x)_n = x(x+1)...(x+n-1).
///
/// Matches `scipy.special.poch`.
pub fn poch(x: f64, n: f64) -> f64 {
    if n == 0.0 {
        return 1.0;
    }
    if n == n.floor() && n > 0.0 && n <= 20.0 {
        let ni = n as usize;
        let mut result = 1.0;
        for k in 0..ni {
            result *= x + k as f64;
        }
        return result;
    }
    // poch(x,n) = Γ(x+n)/Γ(x) is SIGNED; gammaln gives ln|·|, so restore the sign
    // from the gamma factors (scipy.special.poch(-4.3, 0.5) = -2.938, not +2.938).
    // For x>0 both signs are +1, so positive arguments are unchanged.
    let xn = x + n;
    let mode = fsci_runtime::RuntimeMode::Strict;
    let log_result = crate::gammaln_scalar(xn, mode).unwrap_or(f64::NAN)
        - crate::gammaln_scalar(x, mode).unwrap_or(f64::NAN);
    let sign = crate::gammasgn_scalar(xn, mode).unwrap_or(f64::NAN)
        * crate::gammasgn_scalar(x, mode).unwrap_or(f64::NAN);
    sign * log_result.exp()
}

/// Softmax function: exp(x_i) / Σ exp(x_j), numerically stable.
///
/// Matches `scipy.special.softmax`.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum_exp: f64 = exp_x.iter().sum();
    exp_x.iter().map(|&e| e / sum_exp).collect()
}

/// Log-softmax: log(softmax(x)), numerically stable.
///
/// Matches `scipy.special.log_softmax`.
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let shifted: Vec<f64> = x.iter().map(|&xi| xi - max_x).collect();
    let log_sum_exp = shifted.iter().map(|&s| s.exp()).sum::<f64>().ln();
    shifted.iter().map(|&s| s - log_sum_exp).collect()
}

/// Spence's function (dilogarithm): Li₂(1 - x).
///
/// Matches `scipy.special.spence`.
pub fn spence(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("spence", x_tensor, mode, |x| Ok(spence_scalar(x)))
}

#[must_use]
pub fn spence_scalar(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x < 0.0 {
        return f64::NAN;
    }
    dilog_real(1.0 - x)
}

fn dilog_real(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == 1.0 {
        return PI_SQUARED_OVER_SIX;
    }
    if z == 0.0 {
        return 0.0;
    }
    if z < 0.0 {
        let log_term = (1.0 - z).ln();
        let transformed = z / (z - 1.0);
        return -dilog_real(transformed) - 0.5 * log_term * log_term;
    }
    if z > 0.5 {
        let complement = 1.0 - z;
        return PI_SQUARED_OVER_SIX - z.ln() * complement.ln() - dilog_series(complement);
    }
    dilog_series(z)
}

fn dilog_series(z: f64) -> f64 {
    let mut term = z;
    let mut sum = z;
    for k in 2..=DILOG_SERIES_MAX_TERMS {
        term *= z;
        let kf = k as f64;
        let addend = term / (kf * kf);
        sum += addend;
        if addend.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

// ══════════════════════════════════════════════════════════════════════
// Additional Special Functions
// ══════════════════════════════════════════════════════════════════════

/// Wright Omega function: solution of y + ln(y) = z.
///
/// Matches `scipy.special.wrightomega`.
pub fn wrightomega(z_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "wrightomega",
        z_tensor,
        mode,
        |z| Ok(wrightomega_scalar(z)),
        |z| wrightomega_complex_scalar(z, mode),
    )
}

/// Scalar Wright Omega helper.
pub fn wrightomega_scalar(z: f64) -> f64 {
    let exp_z = z.exp();

    // Initial guess via Lambert W approximation
    let mut w = if z > 1.0 {
        z - z.ln()
    } else if z > -2.0 {
        z.exp() / (1.0 + z.exp())
    } else {
        if exp_z <= 1.0e-8 {
            return exp_z;
        }
        exp_z
    };

    // Newton iteration: f(w) = w + ln(w) - z, f'(w) = 1 + 1/w
    for _ in 0..50 {
        if z < 0.0 && (!w.is_finite() || w <= 0.0) {
            return exp_z;
        }
        let residual = w + w.ln() - z;
        if residual.abs() < 1e-15 {
            break;
        }
        let next = w - residual / (1.0 + 1.0 / w);
        if z < 0.0 && (!next.is_finite() || next <= 0.0) {
            return exp_z;
        }
        w = next;
    }

    w
}

fn wrightomega_complex_scalar(z: Complex64, _mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if z.re.is_nan() || z.im.is_nan() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if !z.is_finite() {
        if z.im == 0.0 && z.re == f64::INFINITY {
            return Ok(Complex64::from_real(f64::INFINITY));
        }
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if z.im == 0.0 {
        return Ok(Complex64::from_real(wrightomega_scalar(z.re)));
    }

    let one = Complex64::from_real(1.0);
    let two = Complex64::from_real(2.0);
    let mut w = wrightomega_complex_initial_guess(z);
    if w.norm_sqr() < 1.0e-24 {
        w = Complex64::new(1.0e-12, 0.0);
    }

    for _ in 0..100 {
        if w.norm_sqr() < 1.0e-24 {
            w = Complex64::new(1.0e-12, 0.0);
        }

        let f = w + w.ln() - z;
        if f.abs() < 1.0e-14 * (1.0 + z.abs()) {
            return Ok(w);
        }

        let fp = one + one / w;
        if fp.abs() < 1.0e-30 {
            break;
        }
        let fpp = -one / (w * w);
        let denom = two * fp * fp - f * fpp;
        let mut step = if denom.abs() < 1.0e-30 {
            f / fp
        } else {
            two * f * fp / denom
        };
        let mut next = w - step;

        if !next.is_finite() || next.norm_sqr() < 1.0e-24 {
            step = f / fp;
            next = w - step;
            if !next.is_finite() || next.norm_sqr() < 1.0e-24 {
                break;
            }
        }

        w = next;
        if step.abs() < 1.0e-14 * (1.0 + w.abs()) {
            return Ok(w);
        }
    }

    Ok(w)
}

fn wrightomega_complex_initial_guess(z: Complex64) -> Complex64 {
    if z.re < -2.0 && z.im.abs() < std::f64::consts::FRAC_PI_2 {
        let exp_z = z.exp();
        if exp_z.abs() < 1.0e-8 {
            return exp_z;
        }
    }

    if z.re > 1.0 || z.im.abs() > 1.0 {
        return z - z.ln();
    }

    let exp_z = z.exp();
    let denom = Complex64::from_real(1.0) + exp_z;
    if denom.norm_sqr() < 1.0e-24 {
        return Complex64::from_real(0.5);
    }

    exp_z / denom
}

/// Iterated exponential function (tetration): exp(exp(...exp(x)...)) applied n times.
///
/// exp2(x) = exp(exp(x))
pub fn exp2_iterated(x: f64) -> f64 {
    x.exp().exp()
}

/// Normalized sinc function squared: (sin(πx)/(πx))².
pub fn sinc_squared(x: f64) -> f64 {
    let s = sinc_scalar(x);
    s * s
}

/// Periodic sinc function (Dirichlet kernel).
///
/// diric(x, n) = sin(n*x/2) / (n * sin(x/2))
///
/// At x = 2πk (multiples of 2π), the limit is evaluated:
/// - For n odd: diric(2πk, n) = 1
/// - For n even: diric(2πk, n) = (-1)^k
///
/// Matches `scipy.special.diric(x, n)`.
///
/// # Arguments
/// * `x` - Input value (radians)
/// * `n` - Positive integer order (n >= 1)
///
/// # Examples
/// ```
/// use fsci_special::diric;
/// assert!((diric(0.0, 5) - 1.0).abs() < 1e-15);
/// assert!((diric(0.5, 5) - 0.767153947103405).abs() < 1e-12);
/// ```
pub fn diric(x: f64, n: i32) -> f64 {
    if x.is_nan() || n < 1 {
        return f64::NAN;
    }

    // n = 1 case: sin(x/2) / sin(x/2) = 1 always
    if n == 1 {
        return 1.0;
    }

    let n_f = n as f64;
    let half_x = x / 2.0;

    // Check if x is close to a multiple of 2π (where sin(x/2) ≈ 0)
    // sin(x/2) = 0 when x/2 = kπ, i.e., x = 2kπ
    let sin_half_x = half_x.sin();

    if sin_half_x.abs() < 1e-14 {
        // x ≈ 2kπ for some integer k
        // Use L'Hôpital's rule: lim = n * cos(n*x/2) / cos(x/2)
        // At x = 2kπ: cos(n*kπ) / cos(kπ) = (-1)^(nk) / (-1)^k = (-1)^((n-1)*k)
        let k = (x / (2.0 * std::f64::consts::PI)).round() as i64;
        if n % 2 == 1 {
            // n odd: (-1)^((n-1)*k) = (-1)^(even*k) = 1
            return 1.0;
        } else {
            // n even: (-1)^((n-1)*k) = (-1)^(odd*k) = (-1)^k
            return if k % 2 == 0 { 1.0 } else { -1.0 };
        }
    }

    // General case
    let sin_n_half_x = (n_f * half_x).sin();
    sin_n_half_x / (n_f * sin_half_x)
}

/// Inverse hyperbolic sine (sinh⁻¹): ln(x + √(x²+1)).
///
/// Matches `numpy.arcsinh`.
pub fn arcsinh(x: f64) -> f64 {
    x.asinh()
}

/// Inverse hyperbolic cosine (cosh⁻¹): ln(x + √(x²-1)).
///
/// Matches `numpy.arccosh`.
pub fn arccosh(x: f64) -> f64 {
    x.acosh()
}

/// Inverse hyperbolic tangent (tanh⁻¹): 0.5 * ln((1+x)/(1-x)).
///
/// Matches `numpy.arctanh`.
pub fn arctanh(x: f64) -> f64 {
    x.atanh()
}

/// Compute the squared modulus of the Gamma function |Γ(a + ib)|².
///
/// Useful in scattering theory and other physics applications.
pub fn gamma_mod_squared(a: f64, b: f64) -> f64 {
    // |Γ(a + ib)|² = π * b / (sinh(πb)) * Π_{k=0}^{∞} 1/(1 + b²/(a+k)²)
    // For simplicity, use |Γ(z)|² = Γ(z) * Γ(z̄) = Γ(a+ib) * Γ(a-ib)
    // This equals π / (a * Π_{k=1}^{N} ((a+k)² + b²) / k²) approximately
    //
    // Use the relation: |Γ(a+ib)|² = π*b / (sinh(πb)) for a = 0
    // General case: recurrence + asymptotic
    if b == 0.0 {
        let mode = fsci_runtime::RuntimeMode::Strict;
        let g = crate::gammaln_scalar(a, mode).unwrap_or(f64::NAN);
        return (2.0 * g).exp();
    }

    if a < 0.0 {
        let pi = std::f64::consts::PI;
        let sin_pi_a = (pi * a).sin();
        let cos_pi_a = (pi * a).cos();
        let sinh_pi_b = (pi * b).sinh();
        let cosh_pi_b = (pi * b).cosh();
        let sin_mod_sq = sin_pi_a * sin_pi_a * cosh_pi_b * cosh_pi_b
            + cos_pi_a * cos_pi_a * sinh_pi_b * sinh_pi_b;
        return (pi * pi) / (sin_mod_sq * gamma_mod_squared(1.0 - a, -b));
    }

    // Numerical: use Stirling's approximation shifted to large argument
    let mut val_a = a;
    let mut product = 1.0;
    while val_a < 8.0 {
        product *= val_a * val_a + b * b;
        val_a += 1.0;
    }

    // Stirling: ln|Γ(a+ib)| ≈ (a-0.5)*ln(a²+b²)/2 - b*atan(b/a) - a + 0.5*ln(2π) + ...
    let r2 = val_a * val_a + b * b;
    let theta = b.atan2(val_a);
    let log_mod =
        (val_a - 0.5) * r2.ln() / 2.0 - b * theta - val_a + 0.5 * (2.0 * std::f64::consts::PI).ln();

    (2.0 * log_mod).exp() / product
}

// ══════════════════════════════════════════════════════════════════════
// Convenience Wrappers
// ══════════════════════════════════════════════════════════════════════

/// Log of the binomial coefficient: ln(C(n, k)).
///
/// Matches `scipy.special.gammaln` combination.
pub fn log_comb(n: f64, k: f64) -> f64 {
    let mode = fsci_runtime::RuntimeMode::Strict;
    let lgn1 = crate::gammaln_scalar(n + 1.0, mode).unwrap_or(f64::NAN);
    let lgk1 = crate::gammaln_scalar(k + 1.0, mode).unwrap_or(f64::NAN);
    let lgnk1 = crate::gammaln_scalar(n - k + 1.0, mode).unwrap_or(f64::NAN);
    lgn1 - lgk1 - lgnk1
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Scalar convenience wrapper for `scipy.special.betainc`.
fn betainc_conv(a: f64, b: f64, x: f64) -> f64 {
    crate::betainc_scalar(a, b, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of the regularized incomplete beta function.
///
/// Finds x such that I_x(a, b) = y.
/// Matches `scipy.special.betaincinv`.
pub fn betaincinv(
    a_tensor: &SpecialTensor,
    b_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary(
        "betaincinv",
        a_tensor,
        b_tensor,
        y_tensor,
        mode,
        |a, b, y| Ok(betaincinv_scalar(a, b, y)),
    )
}

/// Scalar helper for the inverse regularized incomplete beta function.
pub fn betaincinv_scalar(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return 1.0;
    }
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }

    // Symmetry I_x(a,b) = 1 - I_{1-x}(b,a): solve the smaller tail so the small-y
    // seed and the dominant-term inverse stay well-conditioned at both ends.
    // frankenscipy-dmkvd.
    if y > 0.5 {
        return 1.0 - betaincinv_scalar(b, a, 1.0 - y);
    }

    let mode = fsci_runtime::RuntimeMode::Strict;
    let ln_beta = crate::betaln_scalar(a, b, mode).unwrap_or(f64::NAN);

    // Small-y seed: I_x(a,b) ~ x^a / (a·B(a,b)) ⟹ x ~ (y·a·B(a,b))^{1/a}.
    // The old x = a/(a+b) (the mean) is far from the root in the tail, which the
    // bracketed Newton could not reach before the (formerly absolute) tolerance
    // tripped.
    let mut x = (y * a * ln_beta.exp()).powf(1.0 / a);
    if !(x > 0.0 && x < 1.0) {
        x = a / (a + b);
    }

    // Bracketed Newton with bisection fallback
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;

    for _ in 0..120 {
        let val = betainc_conv(a, b, x);
        let err = val - y;
        // Relative tolerance: 1e-15 absolute is ~100% relative when y is tiny.
        if err.abs() <= 1e-15 * y.max(1e-300) {
            break;
        }

        // Update brackets
        if val < y {
            lo = x;
        } else {
            hi = x;
        }

        // Newton step: dI_x/dx = x^(a-1) * (1-x)^(b-1) / B(a,b)
        let dpx = if x > 0.0 && x < 1.0 {
            ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta).exp()
        } else {
            0.0
        };

        if dpx > 1e-30 {
            let x_new = x - err / dpx;
            if x_new > lo && x_new < hi {
                x = x_new;
            } else {
                x = 0.5 * (lo + hi);
            }
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    x
}

/// Regularized incomplete gamma function P(a, x).
///
/// Scalar wrapper matching `scipy.special.gammainc`.
pub fn gammainc_conv(a: f64, x: f64) -> f64 {
    crate::gammainc_scalar(a, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x).
///
/// Scalar wrapper matching `scipy.special.gammaincc`.
pub fn gammaincc_conv(a: f64, x: f64) -> f64 {
    crate::gammaincc_scalar(a, x, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of the regularized incomplete gamma function.
///
/// Finds x such that P(a, x) = y.
/// Matches `scipy.special.gammaincinv`.
pub fn gammaincinv(
    a_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("gammaincinv", a_tensor, y_tensor, mode, |a, y| {
        Ok(gammaincinv_scalar(a, y))
    })
}

/// Scalar helper for the inverse regularized incomplete gamma function.
pub fn gammaincinv_scalar(a: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }

    let mode = fsci_runtime::RuntimeMode::Strict;
    let ln_gamma_a = crate::gammaln_scalar(a, mode).unwrap_or(f64::NAN);

    // Initial guess using Wilson-Hilferty approximation for chi-squared quantiles
    let x0 = if y < 0.5 {
        // For small y, use inverse of the leading term: P(a,x) ~ x^a / (a * Gamma(a))
        // x ~ (y * a * Gamma(a))^(1/a)
        (y * a * ln_gamma_a.exp()).powf(1.0 / a)
    } else {
        // For larger y, start near a (the mean of Gamma(a,1))
        a
    };

    // Bracketed Newton: maintain [lo, hi] where P(a, lo) < y < P(a, hi)
    let mut lo = 0.0_f64;
    let mut hi = a + 4.0 * a.sqrt() + 10.0; // generous upper bound
    // Expand hi if needed
    while gammainc_conv(a, hi) < y {
        hi *= 2.0;
    }

    let mut x = x0.clamp(lo + 1e-300, hi);

    for _ in 0..100 {
        let p = gammainc_conv(a, x);
        let err = p - y;
        // Relative tolerance: an absolute 1e-14 stops far too early when y (and
        // hence P) is tiny (~1e-4 relative at y=1e-10). frankenscipy-lj6b2.
        if err.abs() <= 1e-15 * y.max(1e-300) {
            break;
        }

        // Update brackets
        if p < y {
            lo = x;
        } else {
            hi = x;
        }

        // Newton step: dP/dx = x^(a-1) * e^(-x) / Gamma(a)
        let dpx = x.powf(a - 1.0) * (-x).exp() / ln_gamma_a.exp();
        if dpx.abs() > 1e-30 {
            let x_new = x - err / dpx;
            // Accept Newton step only if it stays in bracket
            if x_new > lo && x_new < hi {
                x = x_new;
            } else {
                x = 0.5 * (lo + hi);
            }
        } else {
            x = 0.5 * (lo + hi);
        }
    }
    x
}

/// Inverse of the complemented regularized incomplete gamma function.
///
/// Finds x such that Q(a, x) = y. Matches `scipy.special.gammainccinv`.
pub fn gammainccinv(
    a_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("gammainccinv", a_tensor, y_tensor, mode, |a, y| {
        Ok(gammainccinv_scalar(a, y))
    })
}

/// Scalar helper for the inverse complemented regularized incomplete gamma function.
pub fn gammainccinv_scalar(a: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 1.0 {
        return 0.0;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }

    // Q(a,x) = y. Routing through gammaincinv(a, 1-y) computes P = 1-Q near 1,
    // so the small Q resolves to only ~1e-6 (and 1-y rounds to 1 for tiny y).
    // Instead refine on Q directly with Newton. frankenscipy-lj6b2.
    let ln_gamma_a =
        crate::gammaln_scalar(a, fsci_runtime::RuntimeMode::Strict).unwrap_or(f64::NAN);
    let mut x = if 1.0 - y < 1.0 {
        gammaincinv_scalar(a, 1.0 - y)
    } else {
        // Deep tail: Q(a,x) ~ x^{a-1} e^{-x}/Γ(a) ⟹ x ≈ -ln(y·Γ(a)) + (a-1)ln(x).
        let t = -y.ln() - ln_gamma_a;
        let mut s = t.max(1.0);
        for _ in 0..60 {
            s = t + (a - 1.0) * s.ln();
            if !s.is_finite() || s <= 0.0 {
                s = t;
                break;
            }
        }
        s
    };
    let gamma_a = ln_gamma_a.exp();
    for _ in 0..30 {
        let q = gammaincc_conv(a, x);
        let err = q - y;
        if err.abs() <= 1e-16 * y.max(1e-300) {
            break;
        }
        let dqx = -x.powf(a - 1.0) * (-x).exp() / gamma_a;
        if dqx == 0.0 || !dqx.is_finite() {
            break;
        }
        let x_new = x - err / dqx;
        x = if x_new > 0.0 { x_new } else { 0.5 * x };
    }
    x
}

/// Evaluate the complementary error function erfc(x) = 1 - erf(x).
///
/// Scalar convenience wrapper.
pub fn erfc_conv(x: f64) -> f64 {
    crate::erfc_scalar(x)
}

/// Inverse complementary error function.
///
/// Finds x such that erfc(x) = y.
/// Matches `scipy.special.erfcinv`.
pub fn erfcinv_conv(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return f64::INFINITY;
    }
    if y >= 2.0 {
        return f64::NEG_INFINITY;
    }
    if y > 1.0 {
        return -erfcinv_conv(2.0 - y);
    }
    if y > 0.0625 {
        // 1 - y is well-conditioned here; erfinv handles the central region.
        return crate::erfinv_scalar(1.0 - y, fsci_runtime::RuntimeMode::Strict)
            .unwrap_or(f64::NAN);
    }
    // Deep tail (x > ~1.3): the 1 - y form rounds to 1 for tiny y (erfcinv(1e-100)
    // was inf). Seed from the asymptotic and refine with log-space Newton on
    //   F(x) = -x² + ln(erfcx(x)) - ln(y),   F'(x) = -2/(√π·erfcx(x)),
    // using erfcx from the continued fraction so nothing under/overflows.
    // frankenscipy-l1jgv.
    let mut x = (-(y * 0.5).ln()).sqrt();
    let ln_y = y.ln();
    let sqrt_pi = std::f64::consts::PI.sqrt();
    for _ in 0..16 {
        let ex = crate::error::erfcx_cf_real(x);
        let f = -x * x + ex.ln() - ln_y;
        x += f * sqrt_pi * ex / 2.0;
        if f.abs() < 1e-16 {
            break;
        }
    }
    x
}

// ══════════════════════════════════════════════════════════════════════
// Additional Special Functions
// ══════════════════════════════════════════════════════════════════════

/// Scaled complementary error function: exp(x²) * erfc(x).
///
/// Avoids overflow for large x. Matches `scipy.special.erfcx`.
pub fn erfcx(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "erfcx",
        x_tensor,
        mode,
        |x| Ok(erfcx_scalar(x)),
        |z| erfcx_complex_scalar(z, mode),
    )
}

/// Scaled complementary error function for complex argument:
/// erfcx(z) = e^{z²} erfc(z) = w(iz), the Faddeeva function. frankenscipy-rkwu4.
fn erfcx_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    // i·z = (-Im(z)) + i·Re(z).
    wofz_scalar(Complex64::new(-z.im, z.re), mode)
}

pub fn erfcx_scalar(x: f64) -> f64 {
    if x < 0.0 {
        // For negative x, exp(x²) * erfc(x) = exp(x²) * (2 - erfc(-x))
        // This can overflow, but erfc(-x) is near 2 and exp(x²) grows
        (x * x).exp() * crate::erfc_scalar(x)
    } else if x < 25.0 {
        (x * x).exp() * crate::erfc_scalar(x)
    } else {
        // Asymptotic: erfcx(x) ≈ 1/(x√π) * (1 - 1/(2x²) + 3/(4x⁴) - ...)
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        inv_x / std::f64::consts::PI.sqrt() * (1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2)
    }
}

/// Imaginary error function: erfi(x) = -i * erf(ix) = 2/√π ∫₀ˣ exp(t²) dt.
///
/// Matches `scipy.special.erfi`.
fn erfi_impl(x: f64) -> f64 {
    // erfi(x) = 2x/√π * Σ_{k=0}^∞ x^{2k} / (k! * (2k+1))
    if x.abs() < 6.0 {
        let x2 = x * x;
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..100 {
            term *= x2 / k as f64;
            let contrib = term / (2 * k + 1) as f64;
            sum += contrib;
            if contrib.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        2.0 * x / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfi(x) = (2/√π) e^{x²} D(x), with Dawson's D(x) = e^{-x²}∫₀ˣe^{t²}dt
        // (O(1/x)). The old form multiplied erfcx(-|x|) — which already carries an
        // e^{x²} — by e^{x²} again, a spurious double exponential (erfi(10) was
        // 1.4e87 vs 1.5e42). For x² > ln(f64::MAX) the e^{x²} overflows to ±inf,
        // matching scipy. frankenscipy-sxr71.
        2.0 / std::f64::consts::PI.sqrt() * (x * x).exp() * dawsn_scalar(x)
    }
}

pub fn erfi(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "erfi",
        x_tensor,
        mode,
        |x| Ok(erfi_scalar(x)),
        |z| Ok(erfi_complex_scalar(z)),
    )
}

pub fn erfi_scalar(x: f64) -> f64 {
    erfi_impl(x)
}

/// Imaginary error function for complex argument: erfi(z) = -i·erf(iz).
/// frankenscipy-rkwu4.
fn erfi_complex_scalar(z: Complex64) -> Complex64 {
    // erf(iz) with iz = (-Im(z)) + i·Re(z); then multiply by -i.
    let e = crate::error::erf_complex_scalar(Complex64::new(-z.im, z.re));
    // -i·(e.re + i·e.im) = e.im - i·e.re.
    Complex64::new(e.im, -e.re)
}

/// Owen's T function: T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+t²)/2) / (1+t²) dt.
///
/// Used in bivariate normal distribution. Matches `scipy.special.owens_t`.
pub fn owens_t(
    h_tensor: &SpecialTensor,
    a_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("owens_t", h_tensor, a_tensor, mode, |h, a| {
        Ok(owens_t_scalar(h, a))
    })
}

#[must_use]
pub fn owens_t_scalar(h: f64, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    // T(h,a) is even in h and odd in a; reduce to h ≥ 0, a ≥ 0 and restore
    // the sign of a afterwards.
    let sign = if a < 0.0 { -1.0 } else { 1.0 };
    sign * owens_t_core(h.abs(), a.abs())
}

/// Owen's T for h ≥ 0, a ≥ 0.
fn owens_t_core(h: f64, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    if h == 0.0 {
        return a.atan() / (2.0 * std::f64::consts::PI);
    }
    if a > 1.0 {
        // The 10-point Gauss-Legendre rule below is only accurate over a short
        // interval; for a > 1 the integrand on [0, a] is sharply peaked near
        // t = 0 and 10 nodes miss it (owens_t(1,100) was 36% off). Owen's
        // reflection maps a > 1 to 1/a < 1:
        //   T(h,a) = ½Φ(−h) + ½Φ(−ah) − Φ(−h)Φ(−ah) − T(ah, 1/a).
        // The constant is written via the complementary CDF Φ(−x) so it stays
        // accurate when Φ(h), Φ(ah) ≈ 1 (the ½[Φ(h)+Φ(ah)]−Φ(h)Φ(ah) form lost
        // ~0.7% to cancellation at owens_t(8,3)).
        let qh = ndtr_scalar(-h);
        let qah = ndtr_scalar(-a * h);
        return 0.5 * qh + 0.5 * qah - qh * qah - owens_t_core(a * h, 1.0 / a);
    }

    // Numerical integration via Gauss-Legendre (10-point)
    let gl_nodes = [
        -0.973_906_528_517_171_7,
        -0.865_063_366_688_984_5,
        -0.679_409_568_299_024_4,
        -0.433_395_394_129_247_2,
        -0.148_874_338_981_631_2,
        0.148_874_338_981_631_2,
        0.433_395_394_129_247_2,
        0.679_409_568_299_024_4,
        0.865_063_366_688_984_5,
        0.973_906_528_517_171_7,
    ];
    let gl_weights = [
        0.066_671_344_308_688_1,
        0.149_451_349_150_580_6,
        0.219_086_362_515_982,
        0.269_266_719_309_996_4,
        0.295_524_224_714_752_9,
        0.295_524_224_714_752_9,
        0.269_266_719_309_996_4,
        0.219_086_362_515_982,
        0.149_451_349_150_580_6,
        0.066_671_344_308_688_1,
    ];

    let mid = a / 2.0;
    let half = a / 2.0;
    let h2 = h * h;

    let mut sum = 0.0;
    for (&node, &weight) in gl_nodes.iter().zip(gl_weights.iter()) {
        let t = mid + half * node;
        let integrand = (-0.5 * h2 * (1.0 + t * t)).exp() / (1.0 + t * t);
        sum += weight * integrand;
    }

    sum * half / (2.0 * std::f64::consts::PI)
}

/// Relative error exponential: (exp(x) - 1) / x, accurate near x=0.
///
/// Matches `scipy.special.exprel`.
pub fn exprel(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("exprel", x_tensor, mode, |x| Ok(exprel_scalar(x)))
}

pub fn exprel_scalar(x: f64) -> f64 {
    if x.abs() < 1e-5 {
        // Taylor series: 1 + x/2 + x²/6 + x³/24 + ...
        1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0
    } else {
        x.exp_m1() / x
    }
}

/// Box-Cox transformation: (x^λ - 1) / λ for λ ≠ 0, ln(x) for λ = 0.
///
/// Matches `scipy.special.boxcox`.
pub fn boxcox(
    x_tensor: &SpecialTensor,
    lam_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("boxcox", x_tensor, lam_tensor, mode, |x, lam| {
        Ok(boxcox_scalar(x, lam))
    })
}

pub fn boxcox_scalar(x: f64, lam: f64) -> f64 {
    boxcox_transform_scalar(x, lam)
}

/// Box-Cox transformation under the historical FrankenSciPy helper name.
///
/// Equivalent to [`boxcox`].
pub fn boxcox_transform(
    x_tensor: &SpecialTensor,
    lam_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("boxcox_transform", x_tensor, lam_tensor, mode, |x, lam| {
        Ok(boxcox_transform_scalar(x, lam))
    })
}

pub fn boxcox_transform_scalar(x: f64, lam: f64) -> f64 {
    if x.is_nan() || lam.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if lam == f64::INFINITY {
        if x < 1.0 {
            return -0.0;
        }
        return f64::NAN;
    }
    if lam == f64::NEG_INFINITY {
        if x > 1.0 {
            return 0.0;
        }
        return f64::NAN;
    }
    if x == 0.0 {
        if lam > 0.0 {
            return -1.0 / lam;
        }
        return f64::NEG_INFINITY;
    }
    if lam == 0.0 {
        x.ln()
    } else {
        (lam * x.ln()).exp_m1() / lam
    }
}

/// Inverse Box-Cox transformation.
///
/// Matches `scipy.special.inv_boxcox`.
pub fn inv_boxcox(
    y_tensor: &SpecialTensor,
    lam_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("inv_boxcox", y_tensor, lam_tensor, mode, |y, lam| {
        Ok(inv_boxcox_scalar(y, lam))
    })
}

pub fn inv_boxcox_scalar(y: f64, lam: f64) -> f64 {
    if lam.abs() < 1e-15 {
        y.exp()
    } else {
        (lam * y + 1.0).powf(1.0 / lam)
    }
}

/// Box-Cox transformation with offset: ((x+1)^λ - 1) / λ.
///
/// Matches `scipy.special.boxcox1p`.
pub fn boxcox1p(
    x_tensor: &SpecialTensor,
    lam_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("boxcox1p", x_tensor, lam_tensor, mode, |x, lam| {
        Ok(boxcox1p_scalar(x, lam))
    })
}

pub fn boxcox1p_scalar(x: f64, lam: f64) -> f64 {
    boxcox_transform_scalar(1.0 + x, lam)
}

/// Inverse Box-Cox transformation with offset.
///
/// Matches `scipy.special.inv_boxcox1p`.
pub fn inv_boxcox1p(
    y_tensor: &SpecialTensor,
    lam_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("inv_boxcox1p", y_tensor, lam_tensor, mode, |y, lam| {
        Ok(inv_boxcox1p_scalar(y, lam))
    })
}

pub fn inv_boxcox1p_scalar(y: f64, lam: f64) -> f64 {
    inv_boxcox_scalar(y, lam) - 1.0
}

/// Log of the standard normal CDF.
///
/// Matches `scipy.special.log_ndtr`.
pub fn log_ndtr(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("log_ndtr", x_tensor, mode, |x| Ok(log_ndtr_scalar(x)))
}

/// Scalar helper for `log_ndtr`.
pub fn log_ndtr_scalar(x: f64) -> f64 {
    // log(Φ(x)) where Φ is the standard normal CDF.
    if x >= -1.0 {
        // Φ ∈ [~0.16, 1]: log is well-conditioned directly.
        let t = ndtr_scalar(x);
        return if t > 0.0 { t.ln() } else { 0.0 };
    }
    if x > -8.0 {
        // Moderate left tail: log(½·erfc(-x/√2)) — accurate via erfc, avoiding
        // the rounding of the tiny ndtr value.
        return (0.5 * crate::erfc_scalar(-x * std::f64::consts::FRAC_1_SQRT_2)).ln();
    }
    // Deep left tail (DLMF 7.x Mills-ratio asymptotic):
    //   log Φ(x) = -x²/2 - ½ln(2π) - ln(-x) + ln(Σ_k (-1)^k (2k-1)!!/x^{2k}).
    // The previous form dropped that final correction series (≈ -1/x²), leaving
    // log_ndtr(-30) ~1e-3 off. frankenscipy-ar82j.
    let mut s = 1.0;
    let mut term = 1.0;
    let mut prev_abs = 1.0;
    for k in 1..60 {
        term *= -((2 * k - 1) as f64) / (x * x);
        if term.abs() > prev_abs {
            break; // asymptotic series past its smallest term
        }
        s += term;
        prev_abs = term.abs();
        if term.abs() < 1e-18 {
            break;
        }
    }
    -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln() - (-x).ln() + s.ln()
}

/// Compute the Dawson integral approximation for large arguments.
///
/// For small x, Dawson(x) ≈ x - 2x³/3 + ...
///
/// Matches `scipy.special.dawsn` (scalar convenience).
pub fn dawsn(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "dawsn",
        x_tensor,
        mode,
        |x| Ok(dawsn_scalar(x)),
        |z| dawsn_complex_scalar(z, mode),
    )
}

pub fn dawsn_scalar(x: f64) -> f64 {
    dawsn_impl(x)
}

/// Dawson's integral for complex argument. From w(z) = e^{-z²}(1 + erf(iz)) and
/// erf(iz) = i·erfi(z) = i·(2/√π)·D(z):
///   D(z) = -i·(√π/2)·(w(z) − e^{-z²}). frankenscipy-rkwu4.
fn dawsn_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    let w = wofz_scalar(z, mode)?;
    let diff = w - (-(z * z)).exp();
    let sq = PI.sqrt() / 2.0;
    // -i·(√π/2)·diff = (√π/2)·(diff.im − i·diff.re).
    Ok(Complex64::new(diff.im * sq, -diff.re * sq))
}

/// Compute the Struve function H_v(x) (scalar convenience).
pub fn struve_scalar(v: f64, x: f64) -> f64 {
    struve(v, x)
}

/// Compute the modified Struve function L_v(x) (scalar convenience).
pub fn modstruve_scalar(v: f64, x: f64) -> f64 {
    modstruve(v, x)
}

/// Compute the Debye function D_n(x) = (n/x^n) ∫₀ˣ t^n/(e^t - 1) dt.
///
/// Matches `scipy.special.debye` for n=1,2,3,4.
pub fn debye(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x < 0.0 {
        return f64::NAN;
    }

    // Numerical integration via Simpson's rule
    let npts = (200.0 * (1.0 + x / 5.0).min(10.0)) as usize;
    let npts = npts + (npts % 2);
    let h = x / npts as f64;

    let integrand = |t: f64| -> f64 {
        if t < 1e-15 {
            // L'Hôpital: t^n / (e^t - 1) → t^(n-1) for small t
            t.powi(n as i32 - 1)
        } else {
            t.powi(n as i32) / (t.exp() - 1.0)
        }
    };

    let mut sum = integrand(0.0) + integrand(x);
    for i in 1..npts {
        let t = i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * integrand(t);
    }
    let integral = sum * h / 3.0;

    n as f64 / x.powi(n as i32) * integral
}

/// Lambert W function principal branch W_0(x).
///
/// Finds w such that w * exp(w) = x.
/// Scalar convenience wrapper matching `scipy.special.lambertw`.
pub fn lambertw_scalar(x: f64) -> f64 {
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    if (x - (-1.0 / std::f64::consts::E)).abs() < f64::EPSILON {
        return -1.0;
    }
    if x < -1.0 / std::f64::consts::E {
        return f64::NAN;
    }

    // Initial guess. The asymptotic form x.ln() - ln(ln(x)) is only valid
    // when ln(x) > 1 (i.e. x > e). Below that, ln(ln(x)) is undefined or
    // -∞, so use a Padé-style approximation that is well-behaved for
    // small positive x.
    let mut w = if x < std::f64::consts::E {
        // Bürmann's series gives a good seed: W(x) ≈ x / (1 + x).
        x / (1.0 + x)
    } else {
        x.ln() - x.ln().ln()
    };

    // Halley's method
    for _ in 0..50 {
        let ew = w.exp();
        let wew = w * ew;
        let f = wew - x;
        if f.abs() < 1e-15 * x.abs().max(1.0) {
            break;
        }
        let fp = ew * (w + 1.0);
        let fpp = ew * (w + 2.0);
        w -= f / (fp - f * fpp / (2.0 * fp));
    }

    w
}

/// Zeta function for real s > 1 (convenience alias).
pub fn zeta_scalar(s: f64) -> f64 {
    if s <= 1.0 {
        return f64::NAN;
    }
    if s == f64::INFINITY {
        return 1.0;
    }
    crate::gamma::zeta(s)
}

/// Compute `ln(1 + x) − x` with stable evaluation near zero.
///
/// Matches `scipy.special.log1pmx(x)`. The naive form loses precision for
/// small `x` because `ln(1+x)` and `x` cancel each other. For `|x| ≤ 0.5`
/// this implementation uses the convergent Taylor expansion
///
///   log1pmx(x) = −x²/2 + x³/3 − x⁴/4 + x⁵/5 − …
///
/// truncated when the next term is below ULP-level relative to the current
/// partial sum (typically ≤ 25 terms for any |x| ≤ 0.5). For larger |x|
/// the direct `ln_1p(x) - x` form is accurate enough and is used directly.
///
/// Edge cases:
/// - `x = 0` returns `0.0` exactly (no subtraction).
/// - `x ≤ -1` returns NaN (ln domain), matching scipy.
/// - NaN propagates.
pub fn log1pmx_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x <= -1.0 {
        return f64::NAN;
    }
    if x.abs() <= 0.5 {
        // Taylor: -x²/2 + x³/3 - x⁴/4 + ...
        // Compute from k=2 upward; accumulator term at step k is (-x)^k / k.
        let mut term = -x * x / 2.0;
        let mut sum = term;
        let mut k = 2.0_f64;
        // We decrement |term| by |x| · k/(k+1) per step. With |x| ≤ 0.5 and
        // 24 iterations we already crush the term to < 0.5^24 / 24 ≈ 2.5e-9
        // relative; in practice an f64-relative test exits much earlier.
        for _ in 0..50 {
            term = -term * x * (k / (k + 1.0));
            sum += term;
            if term.abs() < f64::EPSILON * sum.abs() {
                return sum;
            }
            k += 1.0;
        }
        sum
    } else {
        x.ln_1p() - x
    }
}

/// Compute the Faddeeva function w(z) = exp(−z²) · erfc(−iz).
///
/// Matches `scipy.special.wofz` for real inputs and the stable upper-half-plane
/// complex region used by the Voigt-profile family. Real inputs are returned as
/// complex values because SciPy exposes `wofz` as a complex-valued function.
pub fn wofz(z_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z_tensor {
        SpecialTensor::RealScalar(x) => Ok(SpecialTensor::ComplexScalar(wofz_scalar(
            Complex64::from_real(*x),
            mode,
        )?)),
        SpecialTensor::RealVec(values) => values
            .iter()
            .copied()
            .map(|x| wofz_scalar(Complex64::from_real(x), mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        SpecialTensor::ComplexScalar(z) => wofz_scalar(*z, mode).map(SpecialTensor::ComplexScalar),
        SpecialTensor::ComplexVec(values) => values
            .iter()
            .copied()
            .map(|z| wofz_scalar(z, mode))
            .collect::<Result<Vec<_>, _>>()
            .map(SpecialTensor::ComplexVec),
        SpecialTensor::Empty => {
            record_special_trace(
                "wofz",
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid Faddeeva input",
                false,
            );
            Err(SpecialError {
                function: "wofz",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid Faddeeva input",
            })
        }
    }
}

pub fn wofz_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if !z.is_finite() {
        if mode == RuntimeMode::Hardened {
            record_special_trace(
                "wofz",
                mode,
                "domain_error",
                "nonfinite_input",
                "fail_closed",
                "complex Faddeeva input must be finite",
                false,
            );
            return Err(SpecialError {
                function: "wofz",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "complex Faddeeva input must be finite",
            });
        }
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.im == 0.0 {
        let (re, im) = wofz_real(z.re);
        return Ok(Complex64::new(re, im));
    }

    if z.im < 0.0 {
        let reflected = wofz_scalar(-z, mode)?;
        return Ok(Complex64::from_real(2.0) * (-(z * z)).exp() - reflected);
    }

    // Upper half plane (im > 0). Near the real axis the contour integral below
    // has a pole at t = Re(z) that sits within a single Simpson step (h ≈ 0.03),
    // so fixed-step quadrature misses it — wofz(0.5+0.001i) was ~10× off, with
    // error growing as im → 0 across |im| ≲ 0.15. For |z| < 4 use the exact
    // Faddeeva relation w(z) = e^{-z²} erfc(-iz); erfc(-iz) = 1 − erf(-iz) goes
    // through the pole-free erf Maclaurin series there (|-iz| = |z| < 4, so no
    // recursion back into wofz). frankenscipy-wsv5b.
    if z.abs() < 4.0 {
        // -i·z = Im(z) − i·Re(z).
        let erf_val = crate::error::erf_complex_scalar(Complex64::new(z.im, -z.re));
        return Ok((-(z * z)).exp() * (Complex64::from_real(1.0) - erf_val));
    }

    if z.abs() >= 8.0 {
        return Ok(wofz_asymptotic_upper_half_plane(z));
    }

    Ok(wofz_integral_upper_half_plane(z))
}

fn wofz_asymptotic_upper_half_plane(z: Complex64) -> Complex64 {
    let two_z_squared = (z * z) * 2.0;
    let mut term = Complex64::from_real(1.0);
    let mut sum = term;
    for k in 1..30 {
        term = term * f64::from(2 * k - 1) / two_z_squared;
        sum = sum + term;
        if term.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    Complex64::new(0.0, 1.0 / PI.sqrt()) * (sum / z)
}

fn wofz_integral_upper_half_plane(z: Complex64) -> Complex64 {
    const STEPS: usize = 768;
    const LOWER: f64 = -12.0;
    const UPPER: f64 = 12.0;

    let h = (UPPER - LOWER) / STEPS as f64;
    let mut sum = Complex64::new(0.0, 0.0);
    for i in 0..=STEPS {
        let t = LOWER + h * i as f64;
        let weight = if i == 0 || i == STEPS {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        let numerator = (-t * t).exp();
        let denominator = z - Complex64::from_real(t);
        sum = sum + denominator.recip() * (weight * numerator);
    }

    Complex64::new(0.0, 1.0 / PI) * (sum * (h / 3.0))
}

/// Compute the Faddeeva function w(z) = exp(−z²) · erfc(−iz) at a real
/// argument `x`, returning the real and imaginary parts as `(re, im)`.
///
/// On the real axis the closed form is
///   Re[w(x)] = exp(−x²)
///   Im[w(x)] = (2/√π) · F(x)
/// where F is the Dawson function `dawsn`. This matches
/// `scipy.special.wofz(x + 0j)` for real x.
pub fn wofz_real(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let re = (-x * x).exp();
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let im = two_over_sqrt_pi * dawsn_scalar(x);
    (re, im)
}

/// Voigt profile V(x; σ, γ) on the real axis.
///
/// Matches `scipy.special.voigt_profile` for scalar real inputs. The general
/// case is expressed through the Faddeeva function,
/// `Re[w((x + iγ)/(sqrt(2)σ))] / (σ sqrt(2π))`, with SciPy's point-mass and
/// Lorentzian edge cases when `sigma == 0`.
pub fn voigt_profile(x: f64, sigma: f64, gamma: f64) -> f64 {
    if x.is_nan() || sigma.is_nan() || gamma.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    if sigma == 0.0 {
        if gamma == 0.0 {
            return if x == 0.0 { f64::INFINITY } else { 0.0 };
        }
        return gamma / (PI * (x * x + gamma * gamma));
    }
    if gamma == 0.0 && sigma > 0.0 {
        return voigt_profile_real_gamma_zero(x, sigma);
    }

    let scale = SQRT_2 * sigma;
    let z = Complex64::new(x / scale, gamma / scale);
    match wofz_scalar(z, RuntimeMode::Strict) {
        Ok(w) => w.re / (sigma * (2.0 * PI).sqrt()),
        Err(_) => f64::NAN,
    }
}

/// Voigt profile V(x; σ, γ) on the real axis at γ = 0.
///
/// `scipy.special.voigt_profile(x, sigma, 0)` collapses to a Gaussian and is
/// a useful real-only fast path.
pub fn voigt_profile_real_gamma_zero(x: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || sigma.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    // Gaussian PDF with mean 0 and stddev sigma.
    let coeff = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    coeff * (-(x * x) / (2.0 * sigma * sigma)).exp()
}

/// Compute the Tukey-lambda CDF F(x; λ).
///
/// Matches `scipy.special.tklmbda(x, lam)`. The Tukey-lambda family has a
/// closed-form inverse-CDF (PPF) but no closed-form CDF. The PPF is
///
///   F⁻¹(p; λ) = (p^λ - (1-p)^λ) / λ  for λ ≠ 0
///   F⁻¹(p; 0) = ln(p / (1-p))         (logistic limit)
///
/// We invert it by bisection on `p ∈ (0, 1)`: the function p ↦ F⁻¹(p; λ) is
/// strictly increasing for any λ, so a single root exists when `x` is inside
/// the support. For λ > 0 the support is the bounded interval
/// `[-(1/λ), 1/λ]`; outside, the CDF saturates to 0 or 1.
///
/// Special cases:
/// - λ == 0: F(x) = 1 / (1 + e^{-x}) (logistic CDF, returned directly).
/// - x == 0: F(0; λ) = 1/2 for any λ (closed-form, by symmetry).
pub fn tklmbda(x: f64, lam: f64) -> f64 {
    if x.is_nan() || lam.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    if lam == 0.0 {
        // Logistic CDF: 1 / (1 + e^{-x}). Use the negative-x branch for
        // stability when x is very negative.
        if x >= 0.0 {
            return 1.0 / (1.0 + (-x).exp());
        }
        let ex = x.exp();
        return ex / (1.0 + ex);
    }
    // Bounded support for λ > 0: x ∈ [-1/λ, 1/λ].
    if lam > 0.0 {
        let bound = 1.0 / lam;
        if x <= -bound {
            return 0.0;
        }
        if x >= bound {
            return 1.0;
        }
    }
    // Bisection for p ∈ (eps, 1 - eps).
    let ppf = |p: f64| (p.powf(lam) - (1.0 - p).powf(lam)) / lam;
    let eps = 1.0e-300;
    let mut lo = eps;
    let mut hi = 1.0 - eps;
    let f_lo = ppf(lo);
    let f_hi = ppf(hi);
    // Verify the root is bracketed; if x is outside the closed form's
    // image (which happens for λ ≤ 0 with extreme x), return the saturated
    // tail value.
    if x <= f_lo {
        return 0.0;
    }
    if x >= f_hi {
        return 1.0;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let f_mid = ppf(mid);
        if (f_mid - x).abs() < 1.0e-14 || (hi - lo) < 1.0e-15 {
            return mid;
        }
        if f_mid < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Compute `x.powf(y) - 1` with extra accuracy near `x^y == 1`.
///
/// Matches `scipy.special.powm1`. The naive `x.powf(y) - 1` loses precision
/// catastrophically when `y * ln(x)` is small; this implementation uses
/// `expm1(y * ln(x))` which preserves the leading-order term.
///
/// Edge cases:
/// - `x == 1` or `y == 0`: returns `0.0` exactly (no expm1 cancellation).
/// - `x < 0`: returns NaN unless `y` is an integer; non-integer fractional
///   powers of negatives are not real-valued and scipy's powm1 surfaces NaN
///   the same way.
/// - `x == 0` and `y > 0`: returns `-1.0`.
/// - `x == 0` and `y < 0`: +∞.
/// - any non-finite input: propagates the standard f64 powf result minus 1
///   so behavior at ±∞ matches the IEEE convention.
pub fn powm1_scalar(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 1.0 || y == 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        if y > 0.0 {
            return -1.0;
        }
        return f64::INFINITY;
    }
    if x < 0.0 {
        // Integer y is well-defined; otherwise the result isn't real.
        if y.fract() == 0.0 && y.is_finite() {
            return x.powf(y) - 1.0;
        }
        return f64::NAN;
    }
    if !x.is_finite() || !y.is_finite() {
        return x.powf(y) - 1.0;
    }
    (y * x.ln()).exp_m1()
}

pub fn powm1(
    x_tensor: &SpecialTensor,
    y_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("powm1", x_tensor, y_tensor, mode, |x, y| {
        Ok(powm1_scalar(x, y))
    })
}

/// Compute `cos(x) - 1` with extra accuracy near `x == 0`.
///
/// Matches `scipy.special.cosm1`. The naive `cos(x) - 1` loses up to 16 bits
/// of precision when `x` is near a multiple of 2π; this implementation uses
/// the half-angle identity `cos(x) - 1 = -2 sin(x/2)^2` which has no
/// catastrophic cancellation.
pub fn cosm1_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let s = (x * 0.5).sin();
    -2.0 * s * s
}

pub fn cosm1(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("cosm1", x_tensor, mode, |x| Ok(cosm1_scalar(x)))
}

/// Arithmetic-geometric mean of two positive numbers.
pub fn agm(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    let mut an = a;
    let mut bn = b;
    for _ in 0..50 {
        let next_a = (an + bn) / 2.0;
        let next_b = (an * bn).sqrt();
        if (next_a - next_b).abs() < 1e-15 * next_a {
            return next_a;
        }
        an = next_a;
        bn = next_b;
    }
    (an + bn) / 2.0
}

/// Clausen function Cl₂(θ) = Σ_{k=1}^∞ sin(kθ)/k².
///
/// Pre-fix the loop bailed out on a single zero term (sin(kθ) = 0 for
/// k = 2 at θ = π/2 stopped the sum after one iteration, giving 1.0
/// instead of Catalan ≈ 0.9160). The fix: don't trust a single-term
/// magnitude check, instead drive the sum to a fixed iteration cap
/// (the series converges slowly enough that the cap is the dominant
/// cost) and exploit the period-2π and reflection symmetries to keep
/// |θ| ≤ π so the truncation error is bounded by Dirichlet's
/// criterion. frankenscipy-cho22.
pub fn clausen(theta: f64) -> f64 {
    if !theta.is_finite() {
        return f64::NAN;
    }

    // Reduce θ modulo 2π and exploit Cl₂ symmetries:
    //   Cl₂(θ + 2π) = Cl₂(θ)
    //   Cl₂(-θ) = -Cl₂(θ)
    //   Cl₂(2π - θ) = -Cl₂(θ)
    // After reduction we end up with t ∈ [0, π], where the series
    // alternates with a 1/N tail bound for non-pathological t.
    let two_pi = 2.0 * PI;
    let mut t = theta.rem_euclid(two_pi);
    let mut sign = 1.0;
    if t > PI {
        t = two_pi - t;
        sign = -1.0;
    }
    if t == 0.0 || t == PI {
        return 0.0;
    }

    // Direct sum to a fixed cap. For pathological t (rational multiples
    // of small π fractions) the tail bound is O(1/N); 10⁵ iterations
    // gives ≤1e-5 abs error in those cases, ≪1e-10 for the typical
    // alternating case (e.g. t = π/2).
    const MAX_TERMS: usize = 100_000;
    let mut sum = 0.0_f64;
    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        sum += (kf * t).sin() / (kf * kf);
    }
    sign * sum
}

/// Central difference derivative.
pub fn central_diff<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative via central difference.
pub fn central_diff2<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Gradient of a multivariate function via central differences.
pub fn gradient_approx<F>(f: F, x: &[f64], h: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad.push((f(&xp) - f(&xm)) / (2.0 * h));
    }
    grad
}

/// Jacobian of a vector function via central differences.
pub fn jacobian_approx<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![vec![0.0; n]; m];

    for j in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += h;
        xm[j] -= h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
        }
    }

    jac
}

/// Hessian of a scalar function via central differences.
pub fn hessian_approx<F>(f: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            let mut xmm = x.to_vec();

            xpp[i] += h;
            xpp[j] += h;
            xpm[i] += h;
            xpm[j] -= h;
            xmp[i] -= h;
            xmp[j] += h;
            xmm[i] -= h;
            xmm[j] -= h;

            hess[i][j] = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
            hess[j][i] = hess[i][j];
        }
    }

    hess
}

/// Kolmogorov distribution CDF.
///
/// Computes the complementary CDF of the Kolmogorov distribution,
/// P(D_n > x) where D_n is the Kolmogorov-Smirnov statistic.
///
/// Uses the series: K(x) = 1 - 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*x^2)
///
/// Matches `scipy.special.kolmogorov(y)`.
pub fn kolmogorov(y_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("kolmogorov", y_tensor, mode, |y| Ok(kolmogorov_scalar(y)))
}

#[must_use]
pub fn kolmogorov_scalar(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return 1.0;
    }
    if y >= 3.0 {
        // Asymptotic: K(y) ~ 2*exp(-2*y^2) for large y
        return 2.0 * (-2.0 * y * y).exp();
    }

    // Series expansion: K(y) = 1 - 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*y^2)
    let y2 = y * y;
    let mut sum = 0.0;
    let mut sign = 1.0;

    for k in 1..100 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * y2).exp();
        sum += term;
        if term.abs() < 1e-16 * sum.abs().max(1e-30) {
            break;
        }
        sign = -sign;
    }

    2.0 * sum
}

/// Inverse Kolmogorov distribution CDF.
///
/// Returns y such that kolmogorov(y) = p.
///
/// Matches `scipy.special.kolmogi(p)`.
pub fn kolmogi(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("kolmogi", p_tensor, mode, |p| Ok(kolmogi_scalar(p)))
}

#[must_use]
pub fn kolmogi_scalar(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p >= 1.0 {
        return 0.0;
    }

    // K(y) decreases monotonically from K(0) = 1 to 0, so the inverse is
    // bracketed by [lo, hi] with K(lo) ≥ p ≥ K(hi). K(0) = 1 ≥ p gives the
    // lower bound; grow the upper bound until the survival value drops
    // below p.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while kolmogorov_scalar(hi) > p {
        hi *= 2.0;
        if hi > 1.0e6 {
            return hi; // p indistinguishable from 0
        }
    }

    // Asymptotic seed from the dominant term p ≈ 2·exp(-2y²); valid for
    // every p ∈ (0, 1) since p/2 ≤ 1/2. A bare Newton iteration from a
    // fixed seed overshoots and diverges (notably at p = 0.5), so each
    // step is safeguarded: it is accepted only while it stays inside the
    // bracket, otherwise the method falls back to bisection.
    let mut y = (-(p / 2.0).ln() / 2.0).sqrt().clamp(1.0e-12, hi);
    for _ in 0..100 {
        let f = kolmogorov_scalar(y) - p;
        if f > 0.0 {
            lo = y; // K(y) > p ⇒ y below the root
        } else {
            hi = y;
        }
        if f.abs() < 1.0e-15 {
            break;
        }

        // dK/dy = -8y · Σ_{k≥1} (-1)^{k-1} k² exp(-2k²y²).
        let y2 = y * y;
        let mut dsum = 0.0;
        let mut sign = 1.0;
        for k in 1..100 {
            let kf = f64::from(k);
            let term = sign * kf * kf * (-2.0 * kf * kf * y2).exp();
            dsum += term;
            if term.abs() < 1.0e-18 {
                break;
            }
            sign = -sign;
        }
        let df = -8.0 * y * dsum;

        let next = if df.abs() > 1.0e-300 {
            let candidate = y - f / df;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };

        if (next - y).abs() < 1.0e-15 * (1.0 + y) {
            y = next;
            break;
        }
        y = next;
    }

    y
}

/// One-sided Kolmogorov-Smirnov distribution (Smirnov distribution).
///
/// Computes P(D_n^+ ≥ d) where D_n^+ is the one-sided KS statistic
/// for sample size n.
///
/// For n < 1000 uses the exact Birnbaum-Tingey series
///
/// ```text
///   P(D_n^+ ≥ d) = d · Σ_{v=0}^{m} C(n, v) · (d + v/n)^{v−1} · (1 − d − v/n)^{n−v}
/// ```
///
/// where m = ⌊n(1−d)⌋. The (v=0) term has the (d + 0)^{-1} = 1/d
/// factor, so v=0 contributes (1−d)^n; subsequent terms are positive.
/// For n ≥ 1000 the asymptotic exp(−2nd²) is used (relative error
/// O(1/n) per Smirnov).
///
/// Pre-fix the small-n branch only carried the asymptotic with a
/// hand-tuned correction and could be off by 0.3 absolute at n=1
/// (e.g. smirnov(1, 0.5) gave ≈0.39 instead of the exact 0.5)
/// — frankenscipy-5bura.
///
/// Matches `scipy.special.smirnov(n, d)`.
#[must_use]
pub fn smirnov(n: i32, d: f64) -> f64 {
    if n <= 0 || d.is_nan() {
        return f64::NAN;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;

    if n < 1000 {
        // Exact Birnbaum-Tingey series. Compute terms in log-space to
        // avoid overflow of C(n, v) for n up to ~1000.
        let m = ((nf * (1.0 - d)).floor() as i64).max(0);
        let lgam_np1 = crate::gammaln_scalar(nf + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
        let mut sum = 0.0_f64;
        for v in 0..=m {
            let vf = v as f64;
            let evn = d + vf / nf;
            if evn >= 1.0 {
                // (1 − d − v/n) ≤ 0; no real contribution.
                continue;
            }
            let omevn = 1.0 - evn;
            let log_binom = lgam_np1
                - crate::gammaln_scalar(vf + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
                - crate::gammaln_scalar(nf - vf + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
            let log_term = log_binom + (vf - 1.0) * evn.ln() + (nf - vf) * omevn.ln();
            sum += log_term.exp();
        }
        return (sum * d).clamp(0.0, 1.0);
    }

    // Asymptotic for large n.
    let x = 2.0 * nf * d * d;
    (-x).exp().clamp(0.0, 1.0)
}

/// Inverse one-sided Kolmogorov-Smirnov distribution.
///
/// Returns d such that smirnov(n, d) = p.
///
/// Matches `scipy.special.smirnovi(n, p)`.
#[must_use]
pub fn smirnovi(n: i32, p: f64) -> f64 {
    if n <= 0 || p.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;

    // Initial guess from asymptotic: p = exp(-2 * n * d^2) => d = sqrt(-ln(p) / (2n))
    let d0 = (-(p.ln()) / (2.0 * nf)).sqrt().min(0.99);

    // Newton-Raphson iteration
    let mut d = d0;
    for _ in 0..50 {
        let f = smirnov(n, d) - p;
        if f.abs() < 1e-14 {
            break;
        }

        // Numerical derivative
        let h = 1e-8 * d.max(1e-8);
        let df = (smirnov(n, d + h) - smirnov(n, d - h)) / (2.0 * h);

        if df.abs() < 1e-30 {
            break;
        }

        let delta = f / df;
        d -= delta;
        d = d.clamp(1e-15, 1.0 - 1e-15);

        if delta.abs() < 1e-14 * d {
            break;
        }
    }

    d
}

/// Cosine of angle given in degrees.
///
/// Matches `scipy.special.cosdg(x)`.
#[must_use]
pub fn cosdg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Convert degrees to radians and compute cosine
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(360.0);
    if x_mod == 0.0 || x_mod == 360.0 {
        return 1.0;
    }
    if x_mod == 90.0 || x_mod == 270.0 {
        return 0.0;
    }
    if x_mod == 180.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).cos()
}

/// Sine of angle given in degrees.
///
/// Matches `scipy.special.sindg(x)`.
#[must_use]
pub fn sindg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(360.0);
    if x_mod == 0.0 || x_mod == 180.0 || x_mod == 360.0 {
        return 0.0;
    }
    if x_mod == 90.0 {
        return 1.0;
    }
    if x_mod == 270.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).sin()
}

/// Tangent of angle given in degrees.
///
/// Matches `scipy.special.tandg(x)`.
#[must_use]
pub fn tandg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(180.0);
    if x_mod == 0.0 {
        return 0.0;
    }
    if x_mod == 90.0 {
        return f64::INFINITY;
    }
    if x_mod == 45.0 {
        return 1.0;
    }
    if x_mod == 135.0 {
        return -1.0;
    }
    (x * std::f64::consts::PI / 180.0).tan()
}

/// Cotangent of angle given in degrees.
///
/// Matches `scipy.special.cotdg(x)`.
#[must_use]
pub fn cotdg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Handle exact values for common angles
    let x_mod = x.rem_euclid(180.0);
    if x_mod == 0.0 {
        return f64::INFINITY;
    }
    if x_mod == 90.0 {
        return 0.0;
    }
    if x_mod == 45.0 {
        return 1.0;
    }
    if x_mod == 135.0 {
        return -1.0;
    }
    1.0 / (x * std::f64::consts::PI / 180.0).tan()
}

/// Convert angle from degrees to radians.
///
/// Matches `scipy.special.radian(d, m, s)` where d=degrees, m=minutes, s=seconds.
#[must_use]
pub fn radian(degrees: f64, minutes: f64, seconds: f64) -> f64 {
    if degrees.is_nan() || minutes.is_nan() || seconds.is_nan() {
        return f64::NAN;
    }
    let total_degrees = degrees + minutes / 60.0 + seconds / 3600.0;
    total_degrees * std::f64::consts::PI / 180.0
}

/// Cube root that handles negative numbers correctly.
///
/// Unlike `x.powf(1.0/3.0)`, this returns real values for negative x.
/// cbrt(-8) = -2, not NaN.
///
/// Matches `scipy.special.cbrt(x)`.
#[must_use]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Base-2 exponential: 2^x.
///
/// Matches `scipy.special.exp2(x)`.
#[must_use]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Base-10 exponential: 10^x.
///
/// Matches `scipy.special.exp10(x)`.
#[must_use]
pub fn exp10(x: f64) -> f64 {
    (x * std::f64::consts::LN_10).exp()
}

/// Base-2 logarithm.
///
/// Matches `scipy.special.log2(x)` (numpy ufunc).
#[must_use]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// Base-10 logarithm.
///
/// Matches `scipy.special.log10(x)` (numpy ufunc).
#[must_use]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Round to nearest integer.
///
/// Rounds half-way cases away from zero.
/// Matches `scipy.special.round(x)`.
#[must_use]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Floor function - largest integer not greater than x.
///
/// Matches numpy `floor(x)`.
#[must_use]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling function - smallest integer not less than x.
///
/// Matches numpy `ceil(x)`.
#[must_use]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Truncate towards zero.
///
/// Matches `scipy.special.fix(x)` and numpy `trunc(x)`.
#[must_use]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

/// Sign function.
///
/// Returns -1 for x < 0, 0 for x == 0, 1 for x > 0.
/// Returns NaN for NaN input.
///
/// Matches numpy `sign(x)`.
#[must_use]
pub fn sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Heaviside step function.
///
/// H(x) = 0 for x < 0, h0 for x == 0, 1 for x > 0.
///
/// Matches numpy `heaviside(x, h0)`.
#[must_use]
pub fn heaviside(x: f64, h0: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        0.0
    } else {
        h0
    }
}

/// Euclidean distance / hypotenuse: sqrt(x² + y²).
///
/// Computed without overflow for large inputs.
/// Matches numpy `hypot(x, y)`.
#[must_use]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Copy sign of y to magnitude of x.
///
/// Returns a value with magnitude of x and sign of y.
/// Matches numpy `copysign(x, y)`.
#[must_use]
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// Multiply x by 2 raised to the power exp.
///
/// ldexp(x, exp) = x * 2^exp
/// Matches numpy `ldexp(x, exp)`.
#[must_use]
pub fn ldexp(x: f64, exp: i32) -> f64 {
    // Use the formula: x * 2^exp
    // For safety, use libm's implementation via powi
    x * 2.0_f64.powi(exp)
}

/// Extract mantissa and exponent from x.
///
/// Returns (mantissa, exponent) such that x = mantissa * 2^exponent
/// where 0.5 <= |mantissa| < 1.0 (or mantissa == 0 if x == 0).
///
/// Matches numpy `frexp(x)`.
#[must_use]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }

    let bits = x.to_bits();
    let sign = bits >> 63;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;

    if exp == 0 {
        // Subnormal number - normalize it
        let normalized = x * 2.0_f64.powi(64);
        let (m, e) = frexp(normalized);
        return (m, e - 64);
    }

    // Normal number: reconstruct mantissa in [0.5, 1.0)
    let new_exp: u64 = 0x3fe; // Exponent for [0.5, 1.0)
    let new_bits = (sign << 63) | (new_exp << 52) | mantissa_bits;
    let mantissa = f64::from_bits(new_bits);

    (mantissa, exp - 0x3fe)
}

/// Absolute value.
///
/// Matches numpy `fabs(x)`.
#[must_use]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Clip/clamp value to range [min, max].
///
/// Matches numpy `clip(x, min, max)`.
#[must_use]
pub fn clip(x: f64, min: f64, max: f64) -> f64 {
    x.clamp(min, max)
}

/// Sine of π*x with higher accuracy for integer/half-integer arguments.
///
/// sinpi(x) = sin(π*x)
///
/// This function computes sin(π*x) more accurately than sin(PI*x),
/// especially for integer and half-integer values where the result
/// should be exactly 0 or ±1.
///
/// Matches `numpy.sinpi(x)`.
#[must_use]
pub fn sinpi(x: f64) -> f64 {
    use std::f64::consts::PI;

    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    // Reduce to [-1, 1) range for better accuracy
    let mut y = x % 2.0;
    if y > 1.0 {
        y -= 2.0;
    } else if y < -1.0 {
        y += 2.0;
    }

    // For exact integers, return 0
    if y == 0.0 || y == 1.0 || y == -1.0 {
        return 0.0;
    }

    // For half-integers, return ±1
    if y == 0.5 {
        return 1.0;
    }
    if y == -0.5 {
        return -1.0;
    }

    // Use sin(π*y) = sin(π*(1-y)) for y > 0.5
    if y > 0.5 {
        (PI * (1.0 - y)).sin()
    } else if y < -0.5 {
        -(PI * (1.0 + y)).sin()
    } else {
        (PI * y).sin()
    }
}

/// Cosine of π*x with higher accuracy for integer/half-integer arguments.
///
/// cospi(x) = cos(π*x)
///
/// This function computes cos(π*x) more accurately than cos(PI*x),
/// especially for integer and half-integer values where the result
/// should be exactly ±1 or 0.
///
/// Matches `numpy.cospi(x)`.
#[must_use]
pub fn cospi(x: f64) -> f64 {
    use std::f64::consts::PI;

    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    // Reduce to [-1, 1) range
    let mut y = x % 2.0;
    if y > 1.0 {
        y -= 2.0;
    } else if y < -1.0 {
        y += 2.0;
    }

    // For exact integers, return ±1
    if y == 0.0 {
        return 1.0;
    }
    if y == 1.0 || y == -1.0 {
        return -1.0;
    }

    // For half-integers, return 0
    if y == 0.5 || y == -0.5 {
        return 0.0;
    }

    // Use cos(π*y) = -cos(π*(1-y)) for y > 0.5
    if y > 0.5 {
        -(PI * (1.0 - y)).cos()
    } else if y < -0.5 {
        -(PI * (1.0 + y)).cos()
    } else {
        (PI * y).cos()
    }
}

/// Compute log(1 + x) with better accuracy for small x.
///
/// For small x, the direct computation log(1+x) loses precision.
/// This function uses a numerically stable algorithm.
///
/// Matches `numpy.log1p(x)`.
#[must_use]
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Compute exp(x) - 1 with better accuracy for small x.
///
/// For small x, exp(x) is close to 1, so exp(x)-1 loses precision.
/// This function uses a numerically stable algorithm.
///
/// Matches `numpy.expm1(x)`.
#[must_use]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Tangent of π*x with higher accuracy at integer arguments and
/// proper pole semantics at half-integers.
///
/// Computes tan(π·x) as sinpi(x) / cospi(x), so:
///   - tanpi(n) = ±0      for integer n   (preserves sign of sin(πx))
///   - tanpi(n + 0.5) = ±∞ for any integer n  (cospi vanishes)
///   - tanpi(NaN/±∞) = NaN
///
/// Matches `scipy.special.tanpi`. Resolves [frankenscipy-cd205].
#[must_use]
pub fn tanpi(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let s = sinpi(x);
    let c = cospi(x);
    if c == 0.0 {
        // Half-integer pole: sign matches sinpi(x)/0⁺ where 0⁺ is the
        // positive-side limit at the pole. Below we just return the
        // raw IEEE-754 division which propagates the correct ±∞.
        return s / c;
    }
    s / c
}

/// Compute log(exp(x) + exp(y)) without overflow.
///
/// This is useful for adding probabilities in log-space.
/// logaddexp(x, y) = log(exp(x) + exp(y))
///                 = max(x, y) + log1p(exp(-|x - y|))
///
/// Matches `numpy.logaddexp(x, y)`.
#[must_use]
pub fn logaddexp(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return y;
    }
    if y == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY || y == f64::INFINITY {
        return f64::INFINITY;
    }

    // Use the stable formula: max(x,y) + log1p(exp(-|x-y|))
    let (larger, smaller) = if x >= y { (x, y) } else { (y, x) };
    larger + (smaller - larger).exp().ln_1p()
}

/// Compute log2(2^x + 2^y) without overflow.
///
/// This is useful for adding values in log2-space.
/// logaddexp2(x, y) = log2(2^x + 2^y)
///                  = max(x, y) + log2(1 + 2^{-|x - y|})
///
/// Matches `numpy.logaddexp2(x, y)`.
#[must_use]
pub fn logaddexp2(x: f64, y: f64) -> f64 {
    use std::f64::consts::LN_2;

    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return y;
    }
    if y == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY || y == f64::INFINITY {
        return f64::INFINITY;
    }

    // Use the stable formula: max(x,y) + log2(1 + 2^(-|x-y|))
    let (larger, smaller) = if x >= y { (x, y) } else { (y, x) };
    let diff = smaller - larger;
    // 2^diff = exp(diff * ln(2))
    let two_pow_diff = (diff * LN_2).exp();
    larger + two_pow_diff.ln_1p() / LN_2
}

/// Return the next floating-point value after x towards y.
///
/// If x == y, returns y.
///
/// Matches `numpy.nextafter(x, y)`.
#[must_use]
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }

    // Use bit manipulation to find the next representable value
    if x == 0.0 {
        // From zero, step toward y
        let tiny = f64::from_bits(1);
        return if y > 0.0 { tiny } else { -tiny };
    }

    let bits = x.to_bits();
    let next_bits = if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

/// Return the spacing between x and the nearest adjacent number.
///
/// This is the positive distance to the next representable floating
/// point value larger in magnitude than x.
///
/// Matches `numpy.spacing(x)`.
#[must_use]
pub fn spacing(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax == 0.0 {
        // Smallest positive subnormal
        return f64::from_bits(1);
    }

    // Spacing is the difference to the next representable number
    let bits = ax.to_bits();
    let next = f64::from_bits(bits + 1);
    next - ax
}

/// Extract the fractional and integer parts of x.
///
/// Returns (fractional, integer) where x = fractional + integer
/// and fractional has the same sign as x with |fractional| < 1.
///
/// Matches `numpy.modf(x)` (which returns (frac, int)).
#[must_use]
pub fn modf(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x.is_infinite() {
        return (0.0_f64.copysign(x), x);
    }

    let int_part = x.trunc();
    let frac_part = x - int_part;
    (frac_part, int_part)
}

/// Test if x is negative (sign bit is set).
///
/// Returns true if sign bit is set, false otherwise.
/// Note: signbit(-0.0) is true, signbit(NaN) depends on the NaN's sign bit.
///
/// Matches `numpy.signbit(x)`.
#[must_use]
pub fn signbit(x: f64) -> bool {
    x.is_sign_negative()
}

/// Test if x is NaN.
///
/// Matches `numpy.isnan(x)`.
#[must_use]
pub fn isnan(x: f64) -> bool {
    x.is_nan()
}

/// Test if x is positive or negative infinity.
///
/// Matches `numpy.isinf(x)`.
#[must_use]
pub fn isinf(x: f64) -> bool {
    x.is_infinite()
}

/// Test if x is finite (not infinity or NaN).
///
/// Matches `numpy.isfinite(x)`.
#[must_use]
pub fn isfinite(x: f64) -> bool {
    x.is_finite()
}

/// Test if x is positive infinity.
///
/// Matches `numpy.isposinf(x)`.
#[must_use]
pub fn isposinf(x: f64) -> bool {
    x == f64::INFINITY
}

/// Test if x is negative infinity.
///
/// Matches `numpy.isneginf(x)`.
#[must_use]
pub fn isneginf(x: f64) -> bool {
    x == f64::NEG_INFINITY
}

/// Return 1/x (reciprocal).
///
/// Matches `numpy.reciprocal(x)`.
#[must_use]
pub fn reciprocal(x: f64) -> f64 {
    1.0 / x
}

/// Return x² (square).
///
/// Matches `numpy.square(x)`.
#[must_use]
pub fn square(x: f64) -> f64 {
    x * x
}

/// Return the positive part of x (max(x, 0)).
///
/// Matches `numpy.positive(x)` conceptually, though numpy.positive
/// just returns x. This matches the mathematical positive part `[x]⁺`.
#[must_use]
pub fn positive(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Return the negative part of x (max(-x, 0)).
///
/// This matches the mathematical negative part `[x]⁻ = max(-x, 0)`.
#[must_use]
pub fn negative(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x < 0.0 {
        -x
    } else {
        0.0
    }
}

/// Convert degrees to radians.
///
/// deg2rad(x) = x * π / 180
///
/// Matches `numpy.deg2rad(x)` and `numpy.radians(x)`.
#[must_use]
pub fn deg2rad(x: f64) -> f64 {
    x * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees.
///
/// rad2deg(x) = x * 180 / π
///
/// Matches `numpy.rad2deg(x)` and `numpy.degrees(x)`.
#[must_use]
pub fn rad2deg(x: f64) -> f64 {
    x * 180.0 / std::f64::consts::PI
}

/// Round to nearest integer (as float).
///
/// Uses round-half-to-even (banker's rounding) like numpy.rint.
/// This differs from `round` which uses round-half-away-from-zero.
///
/// Matches `numpy.rint(x)`.
#[must_use]
pub fn rint(x: f64) -> f64 {
    // Rust's round_ties_even provides banker's rounding
    x.round_ties_even()
}

/// Round towards zero (truncate to integer, return as float).
///
/// This is equivalent to trunc but emphasizes the "fix" semantic
/// from numpy where values are "fixed" towards zero.
///
/// Matches `numpy.fix(x)`.
#[must_use]
pub fn fix(x: f64) -> f64 {
    x.trunc()
}

/// Return quotient and remainder of division.
///
/// divmod(x, y) returns (floor(x/y), x % y) where the remainder
/// has the same sign as the divisor y (Python/numpy convention).
///
/// Matches `numpy.divmod(x, y)`.
#[must_use]
pub fn divmod(x: f64, y: f64) -> (f64, f64) {
    if y == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x.is_nan() || y.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    // Floor division and modulo (Python-style)
    let q = (x / y).floor();
    let r = x - q * y;
    (q, r)
}

/// Element-wise maximum, propagating NaNs.
///
/// If either input is NaN, returns NaN.
///
/// Matches `numpy.maximum(x, y)`.
#[must_use]
pub fn maximum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x >= y {
        x
    } else {
        y
    }
}

/// Element-wise minimum, propagating NaNs.
///
/// If either input is NaN, returns NaN.
///
/// Matches `numpy.minimum(x, y)`.
#[must_use]
pub fn minimum(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x <= y {
        x
    } else {
        y
    }
}

/// Element-wise maximum, ignoring NaNs.
///
/// If one input is NaN, returns the other. If both are NaN, returns NaN.
///
/// Matches `numpy.fmax(x, y)`.
#[must_use]
pub fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x >= y {
        x
    } else {
        y
    }
}

/// Element-wise minimum, ignoring NaNs.
///
/// If one input is NaN, returns the other. If both are NaN, returns NaN.
///
/// Matches `numpy.fmin(x, y)`.
#[must_use]
pub fn fmin(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        y
    } else if y.is_nan() || x <= y {
        x
    } else {
        y
    }
}

/// Compute x raised to the power y.
///
/// Handles special cases like 0^0 = 1 and negative bases with
/// non-integer exponents (returns NaN).
///
/// Matches `numpy.power(x, y)`.
#[must_use]
pub fn power(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Compute the absolute difference |x - y|.
///
/// Useful for computing distances and tolerances.
#[must_use]
pub fn fdiff(x: f64, y: f64) -> f64 {
    (x - y).abs()
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// nan_to_num(x, nan, posinf, neginf) replaces:
/// - NaN with `nan` (default 0.0)
/// - +inf with `posinf` (default f64::MAX)
/// - -inf with `neginf` (default f64::MIN)
///
/// Matches `numpy.nan_to_num(x)`.
#[must_use]
pub fn nan_to_num(x: f64, nan: f64, posinf: f64, neginf: f64) -> f64 {
    if x.is_nan() {
        nan
    } else if x == f64::INFINITY {
        posinf
    } else if x == f64::NEG_INFINITY {
        neginf
    } else {
        x
    }
}

/// Rectified Linear Unit (ReLU): max(0, x).
///
/// The ReLU activation function, commonly used in neural networks.
///
/// Matches `scipy.special.relu(x)` (proposed).
pub fn relu(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("relu", x_tensor, mode, |x| Ok(relu_scalar(x)))
}

#[must_use]
pub fn relu_scalar(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Softplus: log(1 + exp(x)).
///
/// A smooth approximation to ReLU. Computed in a numerically stable way.
///
/// Matches `scipy.special.softplus(x)` (proposed).
pub fn softplus(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("softplus", x_tensor, mode, |x| Ok(softplus_scalar(x)))
}

#[must_use]
pub fn softplus_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // For large positive x, softplus(x) ≈ x
    // For large negative x, softplus(x) ≈ exp(x) ≈ 0
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Huber loss function.
///
/// huber(delta, x) =
///   0.5 * x^2                  if |x| <= delta
///   delta * (|x| - 0.5*delta)  if |x| > delta
///
/// A robust loss function that is quadratic for small errors and
/// linear for large errors.
///
/// Matches `scipy.special.huber(delta, x)`.
pub fn huber(
    delta_tensor: &SpecialTensor,
    x_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("huber", delta_tensor, x_tensor, mode, |delta, x| {
        Ok(huber_scalar(delta, x))
    })
}

#[must_use]
pub fn huber_scalar(delta: f64, x: f64) -> f64 {
    if delta.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if delta <= 0.0 {
        return f64::NAN;
    }

    let ax = x.abs();
    if ax <= delta {
        0.5 * x * x
    } else {
        delta * (ax - 0.5 * delta)
    }
}

/// Pseudo-Huber loss function.
///
/// pseudo_huber(delta, x) = delta^2 * (sqrt(1 + (x/delta)^2) - 1)
///
/// A smooth approximation to the Huber loss. Unlike Huber, it has
/// continuous derivatives of all orders.
///
/// Matches `scipy.special.pseudo_huber(delta, x)`.
pub fn pseudo_huber(
    delta_tensor: &SpecialTensor,
    x_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("pseudo_huber", delta_tensor, x_tensor, mode, |delta, x| {
        Ok(pseudo_huber_scalar(delta, x))
    })
}

#[must_use]
pub fn pseudo_huber_scalar(delta: f64, x: f64) -> f64 {
    if delta.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if delta <= 0.0 {
        return f64::NAN;
    }

    let ratio = x / delta;
    delta * delta * ((1.0 + ratio * ratio).sqrt() - 1.0)
}

/// Exponential Linear Unit (ELU).
///
/// elu(x, alpha) =
///   x                    if x > 0
///   alpha * (exp(x) - 1) if x <= 0
///
/// A smooth activation function that allows negative outputs.
#[must_use]
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

/// Leaky Rectified Linear Unit.
///
/// leaky_relu(x, alpha) =
///   x         if x > 0
///   alpha * x if x <= 0
///
/// Unlike ReLU, allows small negative values to pass through.
#[must_use]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 { x } else { alpha * x }
}

/// Gaussian Error Linear Unit (GELU).
///
/// gelu(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
///
/// A smooth activation function used in transformers and modern NNs.
/// This is the exact formula; for the approximate version, use gelu_approx.
#[must_use]
pub fn gelu(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    let sqrt_2 = std::f64::consts::SQRT_2;
    x * 0.5 * (1.0 + crate::erf_scalar(x / sqrt_2))
}

/// Scaled Exponential Linear Unit (SELU).
///
/// selu(x) = scale * elu(x, alpha)
/// where scale ≈ 1.0507 and alpha ≈ 1.6733
///
/// Self-normalizing activation function.
#[must_use]
pub fn selu(x: f64) -> f64 {
    const ALPHA: f64 = 1.6732632423543772;
    const SCALE: f64 = 1.0507009873554805;

    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        SCALE * x
    } else {
        SCALE * ALPHA * (x.exp() - 1.0)
    }
}

/// Swish activation function.
///
/// swish(x, beta) = x * sigmoid(beta * x) = x / (1 + exp(-beta * x))
///
/// Also known as SiLU (Sigmoid Linear Unit) when beta = 1.
#[must_use]
pub fn swish(x: f64, beta: f64) -> f64 {
    if x.is_nan() || beta.is_nan() {
        return f64::NAN;
    }
    let bx = beta * x;
    if bx >= 0.0 {
        let ex = (-bx).exp();
        x / (1.0 + ex)
    } else {
        let ex = bx.exp();
        x * ex / (1.0 + ex)
    }
}

/// Mish activation function.
///
/// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// A self-regularized non-monotonic activation function.
pub fn mish(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("mish", x_tensor, mode, |x| Ok(mish_scalar(x)))
}

#[must_use]
pub fn mish_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // softplus(x) = ln(1 + exp(x)), computed stably
    let sp = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    };
    x * sp.tanh()
}

/// Hard sigmoid activation function.
///
/// hard_sigmoid(x) = clip((x + 3) / 6, 0, 1)
///
/// A piecewise linear approximation to sigmoid, faster to compute.
pub fn hard_sigmoid(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("hard_sigmoid", x_tensor, mode, |x| {
        Ok(hard_sigmoid_scalar(x))
    })
}

#[must_use]
pub fn hard_sigmoid_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Hard swish activation function.
///
/// hard_swish(x) = x * hard_sigmoid(x) = x * clip((x + 3) / 6, 0, 1)
///
/// A piecewise linear approximation to swish, used in MobileNetV3.
pub fn hard_swish(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("hard_swish", x_tensor, mode, |x| Ok(hard_swish_scalar(x)))
}

#[must_use]
pub fn hard_swish_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x * hard_sigmoid_scalar(x)
}

/// Hard tanh activation function.
///
/// hard_tanh(x, min, max) = clip(x, min, max)
///
/// A clipped version of the identity function, approximating tanh.
pub fn hard_tanh(
    x_tensor: &SpecialTensor,
    min_tensor: &SpecialTensor,
    max_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary(
        "hard_tanh",
        x_tensor,
        min_tensor,
        max_tensor,
        mode,
        |x, min_val, max_val| Ok(hard_tanh_scalar(x, min_val, max_val)),
    )
}

#[must_use]
pub fn hard_tanh_scalar(x: f64, min_val: f64, max_val: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x.clamp(min_val, max_val)
}

/// Log-cosh loss function.
///
/// log_cosh(x) = log(cosh(x))
///
/// A smooth approximation to absolute value loss. For large |x|,
/// log_cosh(x) ≈ |x| - ln(2).
pub fn log_cosh(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("log_cosh", x_tensor, mode, |x| Ok(log_cosh_scalar(x)))
}

#[must_use]
pub fn log_cosh_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // For numerical stability:
    // log(cosh(x)) = log((exp(x) + exp(-x))/2)
    //              = log(exp(x) + exp(-x)) - log(2)
    // For large |x|: log(cosh(x)) ≈ |x| - log(2)
    let ax = x.abs();
    if ax > 20.0 {
        ax - std::f64::consts::LN_2
    } else {
        x.cosh().ln()
    }
}

/// Softsign activation function.
///
/// softsign(x) = x / (1 + |x|)
///
/// A smooth, bounded activation function similar to tanh but with
/// slower saturation.
pub fn softsign(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("softsign", x_tensor, mode, |x| Ok(softsign_scalar(x)))
}

#[must_use]
pub fn softsign_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x / (1.0 + x.abs())
}

/// Threshold function.
///
/// threshold(x, threshold, value) =
///   x     if x > threshold
///   value otherwise
///
/// A simple step function with configurable threshold and fill value.
pub fn threshold(
    x_tensor: &SpecialTensor,
    thresh_tensor: &SpecialTensor,
    value_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_ternary(
        "threshold",
        x_tensor,
        thresh_tensor,
        value_tensor,
        mode,
        |x, thresh, value| Ok(threshold_scalar(x, thresh, value)),
    )
}

#[must_use]
pub fn threshold_scalar(x: f64, thresh: f64, value: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > thresh { x } else { value }
}

/// Sigmoid Linear Unit (SiLU), same as swish with beta=1.
///
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Also known as swish-1.
pub fn silu(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("silu", x_tensor, mode, |x| Ok(silu_scalar(x)))
}

#[must_use]
pub fn silu_scalar(x: f64) -> f64 {
    swish(x, 1.0)
}

/// Log-expit function (log of logistic sigmoid).
///
/// log_expit(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
///
/// Numerically stable computation of log(expit(x)).
pub fn log_expit(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("log_expit", x_tensor, mode, |x| Ok(log_expit_scalar(x)))
}

#[must_use]
pub fn log_expit_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // log(expit(x)) = -log(1 + exp(-x)) = -softplus(-x)
    -softplus_scalar(-x)
}

/// Complementary log-log function.
///
/// cloglog(p) = log(-log(1 - p))
///
/// The cloglog link function, used in survival analysis and
/// generalized linear models. Maps (0, 1) to (-∞, +∞).
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
#[must_use]
pub fn cloglog(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Use log1p for numerical stability when p is near 0
    if p < 0.5 {
        // log(-log(1-p)) where 1-p is close to 1
        // ln_1p(-p) = ln(1-p), then negate and take ln
        (-((-p).ln_1p())).ln()
    } else {
        (-((1.0 - p).ln())).ln()
    }
}

/// Inverse complementary log-log function.
///
/// cloglog_inv(x) = 1 - exp(-exp(x))
///
/// The inverse of cloglog. Maps (-∞, +∞) to (0, 1).
/// Also known as the Gumbel CDF.
#[must_use]
pub fn cloglog_inv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // 1 - exp(-exp(x))
    // Use expm1 for numerical stability
    if x < -37.0 {
        // exp(x) is essentially 0, so 1 - exp(-0) = 0
        0.0
    } else if x > 709.0 {
        // exp(x) overflows, result is 1
        1.0
    } else {
        -(-x.exp()).exp_m1()
    }
}

/// Log-log link function.
///
/// loglog(p) = -log(-log(p))
///
/// The log-log link function, used in extreme value distributions.
/// Maps (0, 1) to (-∞, +∞). The negative of the Gumbel quantile function.
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
#[must_use]
pub fn loglog(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    -(-p.ln()).ln()
}

/// Inverse log-log link function.
///
/// loglog_inv(x) = exp(-exp(-x))
///
/// The inverse of loglog. Maps (-∞, +∞) to (0, 1).
/// This is the standard Gumbel (minimum) CDF.
#[must_use]
pub fn loglog_inv(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // exp(-exp(-x))
    if x < -709.0 {
        // exp(-x) overflows, exp(-inf) = 0
        0.0
    } else if x > 37.0 {
        // exp(-x) is essentially 0, exp(-0) = 1
        1.0
    } else {
        (-(-x).exp()).exp()
    }
}

/// Cauchy link function (cauchit).
///
/// cauchit(p) = tan(π * (p - 0.5))
///
/// The inverse Cauchy CDF, used as a link function for heavy-tailed
/// distributions. Maps (0, 1) to (-∞, +∞).
///
/// Returns -∞ for p ≤ 0, +∞ for p ≥ 1.
pub fn cauchit(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("cauchit", p_tensor, mode, |p| Ok(cauchit_scalar(p)))
}

#[must_use]
pub fn cauchit_scalar(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    (std::f64::consts::PI * (p - 0.5)).tan()
}

/// Inverse Cauchy link function.
///
/// cauchit_inv(x) = 0.5 + arctan(x) / π
///
/// The Cauchy CDF. Maps (-∞, +∞) to (0, 1).
pub fn cauchit_inv(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("cauchit_inv", x_tensor, mode, |x| Ok(cauchit_inv_scalar(x)))
}

#[must_use]
pub fn cauchit_inv_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    0.5 + x.atan() / std::f64::consts::PI
}

/// Hard shrinkage function.
///
/// hardshrink(x, λ) = x if |x| > λ, else 0
///
/// A thresholding function that sets values with magnitude below λ to zero.
/// Used in sparse signal processing and neural networks.
pub fn hardshrink(
    x_tensor: &SpecialTensor,
    lambda_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("hardshrink", x_tensor, lambda_tensor, mode, |x, lambda| {
        Ok(hardshrink_scalar(x, lambda))
    })
}

#[must_use]
pub fn hardshrink_scalar(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || lambda.is_nan() {
        return f64::NAN;
    }
    if x.abs() > lambda { x } else { 0.0 }
}

/// Soft shrinkage function (soft thresholding).
///
/// softshrink(x, λ) = sign(x) * max(|x| - λ, 0)
///                  = x - λ  if x > λ
///                  = x + λ  if x < -λ
///                  = 0      otherwise
///
/// Shrinks values toward zero by λ. Used in LASSO regression,
/// wavelet denoising, and neural networks.
pub fn softshrink(
    x_tensor: &SpecialTensor,
    lambda_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("softshrink", x_tensor, lambda_tensor, mode, |x, lambda| {
        Ok(softshrink_scalar(x, lambda))
    })
}

#[must_use]
pub fn softshrink_scalar(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || lambda.is_nan() {
        return f64::NAN;
    }
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Tanh shrinkage function.
///
/// tanhshrink(x) = x - tanh(x)
///
/// A smooth shrinkage function that subtracts the bounded tanh.
/// Approaches 0 for small x, approaches x for large |x|.
pub fn tanhshrink(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("tanhshrink", x_tensor, mode, |x| Ok(tanhshrink_scalar(x)))
}

#[must_use]
pub fn tanhshrink_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    x - x.tanh()
}

/// CELU activation function (Continuously Differentiable ELU).
///
/// celu(x, α) = max(0, x) + min(0, α * (exp(x/α) - 1))
///
/// A continuously differentiable variant of ELU. Unlike ELU,
/// CELU is C¹ continuous (has continuous first derivative).
#[must_use]
pub fn celu(x: f64, alpha: f64) -> f64 {
    if x.is_nan() || alpha.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * ((x / alpha).exp() - 1.0)
    }
}

/// LogSigmoid activation function.
///
/// logsigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -softplus(-x)
///
/// Numerically stable log of the sigmoid function.
/// Equivalent to log_expit_scalar but with a more common ML name.
pub fn logsigmoid(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("logsigmoid", x_tensor, mode, |x| Ok(logsigmoid_scalar(x)))
}

#[must_use]
pub fn logsigmoid_scalar(x: f64) -> f64 {
    log_expit_scalar(x)
}

/// Log of 1 minus exp(x), computed in a numerically stable way.
///
/// log1mexp(x) = log(1 - exp(x))
///
/// For x < 0, this computes log(1 - exp(x)) stably.
/// Uses log1p for x close to 0 and direct computation otherwise.
///
/// Returns NaN for x > 0 (since 1 - exp(x) < 0).
pub fn log1mexp(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("log1mexp", x_tensor, mode, |x| Ok(log1mexp_scalar(x)))
}

#[must_use]
pub fn log1mexp_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        return f64::NAN; // log of negative number
    }
    if x == 0.0 {
        return f64::NEG_INFINITY; // log(0)
    }
    // For x < 0:
    // If x is close to 0 (say x > -0.693 = -ln(2)), use log1p(-exp(x))
    // Otherwise use log(1 - exp(x)) directly
    if x > -std::f64::consts::LN_2 {
        // x close to 0: exp(x) close to 1, use log1p for accuracy
        (-x.exp()).ln_1p()
    } else {
        // x far from 0: exp(x) small, direct computation is fine
        (1.0 - x.exp()).ln()
    }
}

/// Log of 1 plus exp(x), computed in a numerically stable way.
///
/// log1pexp(x) = log(1 + exp(x))
///
/// This is the same as softplus(x). Provided as an alias for
/// compatibility with other libraries.
pub fn log1pexp(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("log1pexp", x_tensor, mode, |x| Ok(log1pexp_scalar(x)))
}

#[must_use]
pub fn log1pexp_scalar(x: f64) -> f64 {
    softplus_scalar(x)
}

/// x * log(x) with proper handling of x = 0.
///
/// xlogx(x) = x * log(x) for x > 0
///          = 0         for x = 0
///          = NaN       for x < 0
///
/// Used in entropy calculations where 0 * log(0) = 0 by convention.
pub fn xlogx(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("xlogx", x_tensor, mode, |x| Ok(xlogx_scalar(x)))
}

#[must_use]
pub fn xlogx_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0; // By L'Hôpital's rule, lim x*log(x) as x->0+ = 0
    }
    x * x.ln()
}

/// Negative entropy function: x * log(x).
///
/// negentropy(x) = x * log(x)
///
/// The negation of entropy contribution. Same as xlogx.
/// Returns 0 for x = 0, NaN for x < 0.
pub fn negentropy(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("negentropy", x_tensor, mode, |x| Ok(negentropy_scalar(x)))
}

#[must_use]
pub fn negentropy_scalar(x: f64) -> f64 {
    xlogx_scalar(x)
}

/// Binary cross-entropy loss (logistic loss).
///
/// binary_cross_entropy(p, q) = -p * log(q) - (1-p) * log(1-q)
///
/// Computes the cross-entropy between true label p and predicted
/// probability q. Both p and q should be in [0, 1].
pub fn binary_cross_entropy(
    p_tensor: &SpecialTensor,
    q_tensor: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    map_real_binary("binary_cross_entropy", p_tensor, q_tensor, mode, |p, q| {
        Ok(binary_cross_entropy_scalar(p, q))
    })
}

#[must_use]
pub fn binary_cross_entropy_scalar(p: f64, q: f64) -> f64 {
    if p.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    if q <= 0.0 || q >= 1.0 {
        if (p == 0.0 && q == 0.0) || (p == 1.0 && q == 1.0) {
            return 0.0; // 0 * log(0) = 0 by convention
        }
        return f64::INFINITY; // log(0) or log(negative)
    }
    -p * q.ln() - (1.0 - p) * (1.0 - q).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn ber_bei_large_x_matches_scipy() {
        // frankenscipy-hsjhp: the ascending Kelvin series has intermediate terms
        // ~e^{x/√2} while |ber| ~ e^{x/√2}/√(2πx); for |x| ≳ 130 it loses >16
        // digits to cancellation (ber(150) was ~1e14× too large). The exact
        // complex Bessel identity ber+i·bei = J_0(x·e^{3πi/4}) is cancellation
        // free. (x, ber, bei) from scipy.special 1.17.1.
        let cases: [(f64, f64, f64); 5] = [
            (80.0, 1.5351532598029438e23, -6.023968476830469e22),
            (120.0, -2.417573872140967e35, 9.200088423774093e34),
            (150.0, 1.571856012161004e44, -3.4330393890016124e44),
            (200.0, -6.965727972432077e59, 2.4911521632799942e59),
            (300.0, -9.681529229336163e89, -2.9365929560916326e90),
        ];
        for (x, br, bi) in cases {
            assert!((ber(x) - br).abs() <= 1e-10 * br.abs(), "ber({x}) = {}, scipy {br}", ber(x));
            assert!((bei(x) - bi).abs() <= 1e-10 * bi.abs(), "bei({x}) = {}, scipy {bi}", bei(x));
            // even symmetry
            assert!((ber(-x) - ber(x)).abs() <= 1e-10 * br.abs());
            assert!((bei(-x) - bei(x)).abs() <= 1e-10 * bi.abs());
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    #[allow(clippy::type_complexity)] // flat (re,im,3×complex) golden rows
    fn complex_erfcx_erfi_dawsn_match_scipy() {
        // frankenscipy-rkwu4: erfcx/erfi/dawsn were real-only (map_real fails
        // closed on complex input) while scipy evaluates them for complex z.
        // Added via identities to the clean Faddeeva/erf siblings:
        //   erfcx(z)=w(iz), erfi(z)=-i erf(iz), dawsn(z)=-i(√π/2)(w(z)-e^{-z²}).
        // (re, im, erfcx.re, erfcx.im, erfi.re, erfi.im, dawsn.re, dawsn.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64, f64, f64, f64, f64); 5] = [
            (0.5, 1.0, 0.35490033286757783, -0.3428717191311008, 0.18797346722338337, 0.9507097283189572, 1.6914496078608425, 0.666961949487037),
            (2.0, 3.0, 0.09271076642644344, -0.1283169622282617, -1.1546724379290491e-05, 0.9989632788568172, -70.5023377945093, 110.8743213409972),
            (-1.0, 4.0, -0.03628154550758465, -0.1358395562946222, -3.79403296908907e-08, 1.0000000150962953, -2866261.1123123285, -421526.9848770045),
            (0.1, 8.0, 0.0009029126289383003, -0.07107654514487582, 1.1326489048167856e-29, 1.0, 5.468442077084464e27, -1.597440107434527e26),
            (5.0, -2.0, 0.0964981126066414, 0.037351653156368785, 101670558.35825253, -96103547.82551727, 0.08683899411315939, 0.036019520041904195),
        ];
        let cscalar = |re: f64, im: f64| SpecialTensor::ComplexScalar(Complex64::new(re, im));
        let getc = |r: SpecialResult| match r {
            Ok(SpecialTensor::ComplexScalar(v)) => v,
            _ => Complex64::new(f64::NAN, f64::NAN),
        };
        for (re, im, cxr, cxi, fir, fii, dwr, dwi) in cases {
            let z = cscalar(re, im);
            let cx = getc(erfcx(&z, RuntimeMode::Strict));
            let fi = getc(erfi(&z, RuntimeMode::Strict));
            let dw = getc(dawsn(&z, RuntimeMode::Strict));
            for (got, wr, wi, name) in [
                (cx, cxr, cxi, "erfcx"),
                (fi, fir, fii, "erfi"),
                (dw, dwr, dwi, "dawsn"),
            ] {
                let denom = wr.hypot(wi).max(1e-12);
                let err = (got.re - wr).hypot(got.im - wi) / denom;
                assert!(err <= 1e-9, "{name}({re}{im:+}i) = {got:?}, scipy ({wr},{wi}), rel {err:e}");
            }
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn owens_t_large_a_reflection_matches_scipy() {
        // frankenscipy-yyx6e family: the fixed 10-point Gauss-Legendre rule on
        // [0,a] missed the t=0-peaked integrand for a>1 (owens_t(1,100) was 36%
        // off) and lost ~0.7% to cancellation in the reflection constant at
        // large h (owens_t(8,3)). Owen's reflection to 1/a<1 with the stable
        // Φ(−x) constant now tracks scipy 1.17.1. (h, a, scipy).
        let cases: [(f64, f64, f64); 15] = [
            (0.5, 2.0, 0.1415806036539784),
            (3.0, 0.5, 0.0006051213785851948),
            (5.0, 1.0, 1.4332574485503542e-07),
            (8.0, 3.0, 3.1104802871359146e-16),
            (1.0, 100.0, 0.07932762696572854),
            (0.1, 0.99, 0.12341502312505477),
            (0.3, 5.0, 0.18887156345661174),
            (2.0, 1.5, 0.011365119947351746),
            (6.0, 2.0, 4.932938225188509e-10),
            (0.0, 10.0, 0.2341372412847232),
            (4.0, 0.25, 1.1219796518154963e-05),
            (10.0, 0.5, 3.8099247740170695e-24),
            (0.5, -3.0, -0.15108404307601844),
            (-2.0, 4.0, 0.011375065974089606),
            (1.5, 1.0, 0.031171999563740185),
        ];
        for (h, a, want) in cases {
            let got = owens_t_scalar(h, a);
            // relative 1e-7 with a tiny absolute floor: the largest-h cases sit
            // at ~1e-24 where the GL rule's ~1e-9 relative floor is absolutely
            // negligible.
            let tol = 1e-7 * want.abs() + 1e-15;
            assert!((got - want).abs() <= tol, "owens_t({h},{a}) = {got}, scipy {want}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn betaincinv_tail_matches_scipy() {
        // frankenscipy-dmkvd: relative-tolerance Newton + small-y seed + symmetry.
        // The absolute-tol / mean-seed form was 2.5x wrong at y=1e-15. scipy 1.17.1.
        let cases = [
            (2.0, 3.0, 1e-12, 4.082484015750327e-07),
            (0.5, 0.5, 1e-15, 2.46740110027234e-30),
            (1000.0, 2.0, 1e-10, 0.9740225210945375),
            (2.0, 1000.0, 0.9999999, 0.018928819114004375),
            (2.0, 3.0, 0.5, 0.3857275681323895),
            (5.0, 500.0, 0.02, 0.003042242102338533),
            (0.5, 0.5, 0.99, 0.9997532801828658),
        ];
        for (a, b, y, expected) in cases {
            let got = betaincinv_scalar(a, b, y);
            assert!(
                ((got - expected) / expected).abs() < 1e-11,
                "betaincinv({a},{b},{y}) = {got}, scipy {expected}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn gammaincinv_inverses_match_scipy() {
        // frankenscipy-lj6b2: relative-tolerance Newton (gammaincinv) and Q-Newton
        // (gammainccinv) replace the absolute-tol / P-near-1 forms. scipy 1.17.1.
        let p_cases = [
            (2.0, 1e-10, 1.414220229082974e-05),
            (100.0, 1e-8, 53.62144854308363),
            (0.5, 1e-6, 7.853981633978593e-13),
            (2.0, 0.5, 1.6783469900166612),
            (2.0, 0.999999, 16.68842079082944),
        ];
        for (a, y, expected) in p_cases {
            let got = gammaincinv_scalar(a, y);
            assert!(
                ((got - expected) / expected).abs() < 1e-11,
                "gammaincinv({a},{y}) = {got}, scipy {expected}"
            );
        }
        let q_cases = [
            (2.0, 1e-10, 26.33398160553087),
            (5.0, 1e-8, 28.831980813099847),
            (0.5, 0.5, 0.2274682115597862),
            (2.0, 1e-200, 466.6647703353572),
            (3.5, 1e-50, 126.03964554013531),
        ];
        for (a, y, expected) in q_cases {
            let got = gammainccinv_scalar(a, y);
            assert!(
                ((got - expected) / expected).abs() < 1e-11,
                "gammainccinv({a},{y}) = {got}, scipy {expected}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn ndtri_erfcinv_deep_tail_match_scipy() {
        // frankenscipy-l1jgv: ndtri/erfcinv now route the small complementary
        // argument through erfcinv's log-Newton, so the deep tail stays finite
        // (was -inf/inf for p<1e-16). scipy.special 1.17.1.
        let ndtri_cases = [
            (1e-10, -6.361340902404056),
            (1e-50, -14.933337534788487),
            (1e-200, -30.20559417957964),
            (0.5, 0.0),
            (0.99, 2.3263478740408408),
            (0.999999999, 5.997807019601637),
        ];
        for (p, expected) in ndtri_cases {
            let got = ndtri_scalar(p);
            assert!((got - expected).abs() <= 1e-12 * expected.abs().max(1e-9), "ndtri({p}) = {got}, scipy {expected}");
        }
        let erfcinv_cases = [
            (1e-3, 2.3267537655135246),
            (1e-10, 4.572824967389486),
            (1e-100, 15.065574702592647),
            (1e-300, 26.209469960516124),
            (0.5, 0.4769362762044699),
            (1.5, -0.4769362762044699),
        ];
        for (y, expected) in erfcinv_cases {
            let got = erfcinv_conv(y);
            assert!(((got - expected) / expected).abs() < 1e-12, "erfcinv({y}) = {got}, scipy {expected}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn log_ndtr_matches_scipy() {
        // frankenscipy-ar82j: deep-left-tail Mills-ratio asymptotic + erfc form.
        // The prior asymptotic dropped the log(1-1/x²+…) correction (~1e-3 at -30).
        let cases = [
            (-30.0, -454.32124395634327),
            (-20.0, -203.9171553710973),
            (-10.0, -53.23128515051248),
            (-8.0, -35.01343715991456),
            (-5.0, -15.064998393988727),
            (-2.0, -3.7831843336820317),
            (-1.0, -1.8410216450092634),
            (3.0, -0.0013508099647481925),
        ];
        for (x, expected) in cases {
            let got = log_ndtr_scalar(x);
            assert!((got - expected).abs() <= 1e-12 * expected.abs().max(1e-6), "log_ndtr({x}) = {got}, scipy {expected}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn kelvin_derivatives_match_scipy() {
        // frankenscipy-l3kwr: analytic ber'/bei'/ker'/kei' via complex I₁/K₁
        // replace finite-difference / cancelling-series forms (~1e-8..4e-6 off).
        let cases = [
            (1.0, -0.06244575217903096, 0.49739651146809727, -0.6946038911006908, 0.3523699133361705),
            (5.0, -3.8453394732621544, -4.354140514843111, 0.017193403828394, -0.0008199865436310269),
            (10.0, 51.19525834615495, 135.3093016566432, -0.00031559693447617284, 0.0001409138375599196),
            (11.0, -94.21185202497774, 264.11937428720506, -6.99034409619794e-05, 0.00014625254023259207),
            (15.0, 91.05533316965173, -4087.755236845389, 5.644678075956919e-06, -5.882222803057011e-06),
            (20.0, -48803.19784717074, 111855.02522349692, -7.501859210700294e-08, 1.906242756745313e-07),
            (50.0, -46498923792943.57, -118164845285863.83, 7.221202712795484e-17, -3.141565492509411e-17),
        ];
        for (x, berp_ref, beip_ref, kerp_ref, keip_ref) in cases {
            assert!((berp(x) - berp_ref).abs() <= 1e-7 * berp_ref.abs().max(1e-6), "berp({x})={}", berp(x));
            assert!((beip(x) - beip_ref).abs() <= 1e-7 * beip_ref.abs().max(1e-6), "beip({x})={}", beip(x));
            assert!((kerp(x) - kerp_ref).abs() <= 1e-7 * kerp_ref.abs().max(1e-12), "kerp({x})={}", kerp(x));
            assert!((keip(x) - keip_ref).abs() <= 1e-7 * keip_ref.abs().max(1e-12), "keip({x})={}", keip(x));
        }
    }


    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn ker_kei_large_x_matches_scipy() {
        // frankenscipy-rhilt: large x uses K₀(x e^{iπ/4}) asymptotic; the
        // log/harmonic series was ~3% off at x=20. scipy.special.ker/kei 1.17.1.
        let cases = [
            (11.0, -4.7791933610698554e-05, -0.00014953707794845456),
            (12.0, -6.307713705210742e-05, -3.899959497124564e-05),
            (15.0, -1.514347207267156e-08, 7.962894398377203e-06),
            (20.0, -7.715233109860963e-08, -1.8589415111194396e-07),
            (50.0, -2.9150770893968664e-17, 7.25581322036562e-17),
        ];
        for (x, ker_ref, kei_ref) in cases {
            let kr = ker(x);
            let ki = kei(x);
            assert!(((kr - ker_ref) / ker_ref).abs() < 1e-9, "ker({x}) = {kr:e}, scipy {ker_ref:e}");
            assert!(((ki - kei_ref) / kei_ref).abs() < 1e-9, "kei({x}) = {ki:e}, scipy {kei_ref:e}");
        }
        // Continuity / small-x series path stays correct.
        assert!((ker(5.0) - (-0.011511727199492405)).abs() < 1e-12);
        assert!((kei(5.0) - 0.011187586509870114).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn sici_matches_scipy() {
        // frankenscipy-dyni8: x≥6 now uses the complex-E₁ continued fraction
        // (was ~1e-9 off near x=20 from the asymptotic floor). scipy 1.17.1.
        let cases = [
            (1.0, 0.9460830703671831, 0.33740392290096816),
            (5.0, 1.549931244944674, -0.1900297496566439),
            (6.0, 1.4246875512805066, -0.06805724389324713),
            (10.0, 1.658347594218874, -0.04545643300445537),
            (20.0, 1.5482417010434397, 0.044419820845353314),
            (50.0, 1.551617072485936, -0.005628386324116305),
            (200.0, 1.5683823393394698, -0.004378446093027826),
            (1000.0, 1.5702331219687713, 0.0008263155110906821),
            (-20.0, -1.5482417010434397, 0.044419820845353314),
        ];
        for (x, si_ref, ci_ref) in cases {
            let (si, ci) = sici(x);
            assert!((si - si_ref).abs() < 1e-12, "Si({x}) = {si}, scipy {si_ref}");
            assert!((ci - ci_ref).abs() < 1e-12, "Ci({x}) = {ci}, scipy {ci_ref}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn expn_expi_match_scipy() {
        // frankenscipy-inkqr: E_n via NR modified-Lentz CF / series replaces the
        // fixed 20-level recurrence (E_1(1) was ~7e-8 off). scipy 1.17.1.
        let expn_cases = [
            (1usize, 0.3, 0.9056766516758468),
            (1, 1.0, 0.2193839343955205),
            (1, 2.0, 0.048900510708061146),
            (1, 5.0, 0.0011482955912753255),
            (1, 20.0, 9.835525290649882e-11),
            (2, 0.5, 0.3266438623245532),
            (2, 2.0, 0.03753426182049047),
            (2, 10.0, 3.8302404656316095e-06),
            (5, 1.0, 0.07045423746172041),
            (5, 3.0, 0.006697984917017044),
        ];
        for (n, x, expected) in expn_cases {
            let got = expn(n, x);
            assert!(((got - expected) / expected).abs() < 1e-12, "expn({n},{x})={got}, scipy {expected}");
        }
        let expi_neg = [
            (-0.5, -0.5597735947761608),
            (-1.0, -0.2193839343955205),
            (-3.0, -0.013048381094197039),
            (-10.0, -4.156968929685325e-06),
        ];
        for (x, expected) in expi_neg {
            let got = expi_scalar(x);
            assert!(((got - expected) / expected).abs() < 1e-12, "expi({x})={got}, scipy {expected}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn dawsn_matches_scipy() {
        // frankenscipy-p43m1: Rybicki mid-range replaces ~1e-7 Simpson; the
        // asymptotic gained optimal truncation. scipy.special.dawsn 1.17.1.
        let cases = [
            (0.03, 0.02998200647833413),
            (0.5, 0.4244363835020223),
            (1.0, 0.5380795069127684),
            (2.0, 0.301340388923792),
            (3.0, 0.17827103061055827),
            (3.9, 0.13292729108108925),
            (5.0, 0.10213407442427686),
            (6.25, 0.08106609406101171),
            (10.0, 0.05025384718759854),
            (30.0, 0.016675941401059196),
            (-3.0, -0.17827103061055827),
        ];
        for (x, expected) in cases {
            let got = dawsn_scalar(x);
            let rel = ((got - expected) / expected).abs();
            assert!(rel < 1e-13, "dawsn({x}) = {got}, scipy {expected}, rel={rel:e}");
        }
        // Propagation check: Im wofz(3+0i) = (2/√π) dawsn(3).
        let w = wofz_scalar(Complex64::new(3.0, 0.0), RuntimeMode::Strict).unwrap();
        assert!((w.im - 0.20115731703760037).abs() < 1e-13, "wofz(3).im = {}", w.im);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn fresnel_large_x_matches_scipy() {
        // frankenscipy-2fpck: corrected A&S 7.3.27/28 auxiliary-function series.
        // The cosine integral C was ~1e-3 off at large x. scipy.special.fresnel.
        let cases = [
            (6.0, 0.4469607612369303, 0.4995314678555011),
            (7.0, 0.49970478945344676, 0.5454670925469698),
            (10.0, 0.46816997858488224, 0.49989869420551575),
            (50.0, 0.49363380258593875, 0.49999918943072796),
            (200.0, 0.49840845056938343, 0.49999998733485207),
            (1000.0, 0.4996816901138163, 0.4999999998986788),
            (-50.0, -0.49363380258593875, -0.49999918943072796),
        ];
        for (x, sref, cref) in cases {
            let (s, c) = fresnel(x);
            assert!((s - sref).abs() < 1e-12, "fresnel({x}).S = {s}, scipy {sref}");
            assert!((c - cref).abs() < 1e-12, "fresnel({x}).C = {c}, scipy {cref}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn erfi_large_x_matches_scipy() {
        // frankenscipy-sxr71: erfi(x)=(2/√π)e^{x²}D(x) replaces the double-
        // exponential erfcx form (erfi(10) was 1.4e87 vs 1.5e42). scipy 1.17.1.
        let cases = [
            (6.0, 411275145582823.94),
            (7.0, 1.553486253460504e20),
            (10.0, 1.52430742270867e42),
            (15.0, 1.9613845638673805e96),
            (25.0, 6.135986249821945e269),
            (26.0, 8.31463716473099e291),
        ];
        for (x, expected) in cases {
            let got = erfi_scalar(x);
            let rel = ((got - expected) / expected).abs();
            assert!(rel < 1e-9, "erfi({x}) = {got:e}, scipy {expected:e}, rel={rel:e}");
            // Odd symmetry.
            assert!((erfi_scalar(-x) + got).abs() <= 1e-9 * got.abs());
        }
        // x² past ln(f64::MAX) ≈ 709.78 overflows to +inf, matching scipy.
        assert_eq!(erfi_scalar(27.0), f64::INFINITY);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy/mpmath
    fn struve_large_x_matches_scipy() {
        // frankenscipy-3z6wd: large-x H_v via DLMF 11.6.1 (Y_v + correction
        // series). v≠0 was catastrophically wrong (struve(1,200) gave -5.7e83
        // vs 0.65); v=0 was 0.17% off. scipy.special.struve 1.17.1.
        let cases = [
            (0.0, 35.0, 0.06397238222066917),
            (0.0, 50.0, -0.08533767482611902),
            (0.0, 200.0, -0.05108275594755782),
            (1.0, 35.0, 0.7646509379741863),
            (1.0, 50.0, 0.5800784479454417),
            (1.0, 200.0, 0.6519375112490656),
            (2.0, 100.0, 21.30386405267446),
            (0.5, 80.0, 0.0990534330000826),
            (3.0, 150.0, 955.1431304112394),
        ];
        for (v, x, expected) in cases {
            let got = struve(v, x);
            let rel = ((got - expected) / expected).abs();
            // ~1e-10 floor at moderate x is inherited from the Y_v asymptotic;
            // still a vast improvement over the prior catastrophic blow-up.
            assert!(rel < 1e-9, "struve({v},{x}) = {got:e}, scipy {expected:e}, rel={rel:e}");
        }
    }

    #[test]
    fn pentagamma_matches_scipy_polygamma_three() {
        // scipy.special.polygamma(3, x) at five spread-out positive
        // values. The shift-then-asymptotic implementation in
        // pentagamma() converges to ~1e-7 even at x = 0.5 where the
        // asymptotic truncation is least accurate.
        let cases = [
            (0.5_f64, 97.409_091_034_002_42),
            (1.0, 6.493_939_402_266_829),
            (2.0, 0.493_939_402_266_829),
            (5.0, 0.021_427_828_192_755),
            (10.0, 0.002_319_901_304_290),
        ];
        for &(x, want) in &cases {
            let got = pentagamma(x);
            assert!(
                (got - want).abs() < 1e-6,
                "pentagamma({x}) = {got}, scipy {want}",
            );
        }
    }

    #[test]
    fn tetragamma_matches_scipy_polygamma_two_negative_x() {
        // scipy.special.polygamma(2, x) at negative non-integer x; the
        // reflection branch in tetragamma() takes over here.
        let cases = [
            (-0.5_f64, -0.828_796_644_234_320),
            (-1.5, -0.236_204_051_641_728),
            (-2.5, -0.108_204_051_641_728),
        ];
        for &(x, want) in &cases {
            let got = tetragamma(x);
            assert!(
                (got - want).abs() < 1e-7,
                "tetragamma({x}) = {got}, scipy {want}",
            );
        }
    }

    #[test]
    fn logsumexp_with_weights_matches_scipy_contract_point() {
        let value =
            logsumexp_with_b(&[1.0, 2.0, 3.0], &[1.0, 2.0, 0.5]).expect("weighted logsumexp");
        assert!((value - 3.315_609_082_086_973_5).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_matches_scipy_axis_zero_contract_point() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let reduced = logsumexp_axis_2d(&data, 0).expect("axis logsumexp");
        assert_eq!(reduced.len(), 2);
        assert!((reduced[0] - 3.126_928_011_042_972_7).abs() < 1.0e-12);
        assert!((reduced[1] - 4.126_928_011_042_972).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_with_broadcast_weights_matches_scipy_contract_point() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![vec![1.0, 2.0]];
        let reduced =
            logsumexp_axis_2d_with_b(&data, 1, &weights).expect("axis logsumexp with weights");
        assert_eq!(reduced.len(), 2);
        assert!((reduced[0] - 2.861_994_804_058_251_2).abs() < 1.0e-12);
        assert!((reduced[1] - 4.861_994_804_058_251).abs() < 1.0e-12);
    }

    #[test]
    fn logsumexp_axis_2d_rejects_non_broadcastable_weights() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![vec![1.0, 2.0, 3.0]];
        let err = logsumexp_axis_2d_with_b(&data, 1, &weights).expect_err("shape mismatch");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn expi_scalar_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin Ei(x) at canonical
        // points. References from Abramowitz & Stegun Table 5.1 and
        // scipy.special.expi. Tolerance reflects the current
        // implementation's series-truncation accuracy (~1e-7 for the
        // E₁ branch via expn, ~1e-9 for the positive-x series); the
        // test fails cleanly if either branch regresses outside that
        // observed envelope.
        //   Ei(0) = −∞
        //   Ei(1) ≈ 1.895_117_816_355_937   (γ + Σ 1/(k·k!))
        //   Ei(2) ≈ 4.954_234_356_001_891
        //   Ei(−1) ≈ −0.219_383_934_395_520 (= −E₁(1))
        assert_eq!(expi_scalar(0.0), f64::NEG_INFINITY);
        assert!((expi_scalar(1.0) - 1.895_117_816_355_937).abs() < 1e-9);
        assert!((expi_scalar(2.0) - 4.954_234_356_001_891).abs() < 1e-9);
        assert!((expi_scalar(-1.0) - (-0.219_383_934_395_520)).abs() < 1e-6);
        // Large x: the convergent series truncated at 200 terms returned ~0 for
        // x≳200 (expi(500) was 0 vs 2.8e214); the divergent asymptotic branch
        // (x≥40) now tracks scipy 1.17.1 to ~1e-15. frankenscipy.
        for (x, want) in [
            (50.0_f64, 1.058563689713169e20),
            (100.0, 2.71555274485388e41),
            (500.0, 2.8128213978862945e214),
            (700.0, 1.4509787360525605e301),
        ] {
            let got = expi_scalar(x);
            assert!(((got - want) / want).abs() < 1e-13, "expi({x}) = {got}, scipy {want}");
        }
    }

    #[test]
    fn expit_logit_metamorphic_roundtrip() {
        // /testing-metamorphic: expit(logit(p)) = p for p in (0, 1).
        // Both inverses share the floating-point logistic kernel, so
        // a roundtrip identity catches branch drift in either path.
        for &p in &[0.001_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
            let l = logit_scalar(p, RuntimeMode::Strict).unwrap();
            let p_back = expit_scalar(l);
            let rel = ((p_back - p) / p).abs();
            assert!(
                rel < 1e-14,
                "expit(logit({p})) = {p_back}, expected {p} (rel = {rel})"
            );
        }
    }

    #[test]
    fn softmax_matches_scipy_reference_points() {
        // Skill rotation: /testing-conformance-harnesses (parity vs scipy
        // reference values). softmax is closed-form; pin three regimes:
        //
        //   softmax([1, 2, 3]) = [e/(e+e²+e³), e²/..., e³/...]
        //   softmax([0]) = [1.0]
        //   softmax([-1000, -1000, -1000]) = [1/3, 1/3, 1/3]
        //   (numerical stability via max subtraction)
        let r = softmax(&[1.0_f64, 2.0, 3.0]);
        let e1 = 1.0_f64.exp();
        let e2 = 2.0_f64.exp();
        let e3 = 3.0_f64.exp();
        let denom = e1 + e2 + e3;
        let expected = [e1 / denom, e2 / denom, e3 / denom];
        for (i, (got, want)) in r.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "softmax([1,2,3])[{i}] = {got}, expected {want}"
            );
        }
        // Sum-to-one invariant.
        let s: f64 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "softmax sum = {s}, expected 1.0");

        // Single-element softmax is always [1.0].
        assert_eq!(softmax(&[0.0]), vec![1.0]);
        assert_eq!(softmax(&[42.0]), vec![1.0]);

        // Numerical stability: large negative inputs should not underflow
        // to all-zero with NaN sum. Three identical entries → [1/3, 1/3, 1/3].
        let stable = softmax(&[-1000.0, -1000.0, -1000.0]);
        for (i, &v) in stable.iter().enumerate() {
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-12,
                "softmax(extreme)[{i}] = {v}, expected 1/3"
            );
        }
    }

    #[test]
    fn nrdtrimn_recovers_mean() {
        let mean = 3.0;
        let std = 2.0;
        let x = 6.0;
        let p = ndtr_scalar((x - mean) / std);
        let recovered = nrdtrimn(p, std, x);
        assert!(
            (recovered - mean).abs() <= 1.0e-12,
            "nrdtrimn mean recovery mismatch: expected={mean}, got={recovered}"
        );
    }

    #[test]
    fn nrdtrimn_matches_scipy_contract_points() {
        assert!((nrdtrimn(0.8, 2.0, 1.0) - (-0.683_242_467_145_828_8)).abs() <= 1.0e-12);
        assert!((nrdtrimn(0.2, 2.0, 1.0) - 2.683_242_467_145_828_6).abs() <= 1.0e-12);
        assert!((nrdtrimn(0.5, 1.0, 1.0) - 1.0).abs() <= 1.0e-12);
        assert!(nrdtrimn(0.2, f64::INFINITY, 1.0).is_infinite());
        assert!(nrdtrimn(0.2, f64::INFINITY, 1.0).is_sign_positive());
        assert!(nrdtrimn(0.5, f64::INFINITY, 1.0).is_infinite());
        assert!(nrdtrimn(0.5, f64::INFINITY, 1.0).is_sign_negative());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::INFINITY).is_nan());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::INFINITY).is_nan());
        assert!(nrdtrimn(0.2, f64::INFINITY, f64::NEG_INFINITY).is_nan());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::NEG_INFINITY).is_infinite());
        assert!(nrdtrimn(0.5, f64::INFINITY, f64::NEG_INFINITY).is_sign_negative());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::NEG_INFINITY).is_infinite());
        assert!(nrdtrimn(0.8, f64::INFINITY, f64::NEG_INFINITY).is_sign_negative());
    }

    #[test]
    fn nrdtrimn_rejects_invalid_inputs_like_scipy() {
        assert!(nrdtrimn(0.0, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(1.0, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, 0.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, -1.0, 1.0).is_nan());
        assert!(nrdtrimn(f64::NAN, 2.0, 1.0).is_nan());
        assert!(nrdtrimn(0.8, f64::NAN, 1.0).is_nan());
        assert!(nrdtrimn(0.8, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn nrdtrisd_recovers_std() {
        let mean = 3.0;
        let std = 2.0;
        let x = 6.0;
        let p = ndtr_scalar((x - mean) / std);
        let recovered = nrdtrisd(mean, p, x);
        assert!(
            (recovered - std).abs() <= 1.0e-12,
            "nrdtrisd std recovery mismatch: expected={std}, got={recovered}"
        );
    }

    #[test]
    fn nrdtrisd_matches_scipy_contract_points() {
        assert!(nrdtrisd(0.5, 0.5, 0.5) == 0.0);
        assert!((nrdtrisd(3.0, 0.933_192_798_731_141_9, 6.0) - 2.0).abs() <= 1.0e-12);
        assert!(
            (nrdtrisd(1.0, 0.5, 0.5) - (-7.532_401_279_578_852e15)).abs()
                <= 1.0e-12 * 7.532_401_279_578_852e15
        );
        assert!(nrdtrisd(1.0, 0.2, f64::INFINITY).is_infinite());
        assert!(nrdtrisd(1.0, 0.2, f64::INFINITY).is_sign_negative());
        assert!(nrdtrisd(1.0, 0.5, f64::INFINITY).is_infinite());
        assert!(nrdtrisd(1.0, 0.5, f64::INFINITY).is_sign_positive());
        assert!(nrdtrisd(f64::INFINITY, 0.2, 1.0).is_infinite());
        assert!(nrdtrisd(f64::INFINITY, 0.2, 1.0).is_sign_positive());
        assert!(nrdtrisd(f64::NEG_INFINITY, 0.8, 1.0).is_infinite());
        assert!(nrdtrisd(f64::NEG_INFINITY, 0.8, 1.0).is_sign_positive());
    }

    #[test]
    fn nrdtrisd_rejects_invalid_inputs_like_scipy() {
        assert!(nrdtrisd(0.0, 0.0, 1.0).is_nan());
        assert!(nrdtrisd(0.0, 1.0, 1.0).is_nan());
        assert!(nrdtrisd(f64::NAN, 0.8, 1.0).is_nan());
        assert!(nrdtrisd(0.0, f64::NAN, 1.0).is_nan());
        assert!(nrdtrisd(0.0, 0.8, f64::NAN).is_nan());
    }

    #[test]
    fn kelvin_functions_match_scipy_reference_points() {
        assert!((ber(1.0) - 0.984_381_781_213_087).abs() < 1e-12);
        assert!((bei(1.0) - 0.249_566_040_036_659_72).abs() < 1e-12);
        assert!((ker(1.0) - 0.286_706_208_728_316_04).abs() < 1e-10);
        assert!((kei(1.0) - (-0.494_994_636_518_72)).abs() < 1e-10);
    }

    #[test]
    fn kelvin_derivatives_match_scipy_reference_points() {
        let cases = [
            ("berp", 0.1, berp(0.1), -6.249_999_457_465_285e-5, 1.0e-12),
            ("berp", 3.0, berp(3.0), -1.569_846_632_229_404_2, 1.0e-12),
            ("beip", 0.1, beip(0.1), 0.049_999_973_958_334_02, 1.0e-12),
            ("beip", 3.0, beip(3.0), 0.880_482_324_057_861_4, 1.0e-12),
            ("kerp", 1.0, kerp(1.0), -0.694_603_891_100_690_8, 1.0e-3),
            ("keip", 1.0, keip(1.0), 0.352_369_913_336_170_5, 1.0e-3),
        ];
        for (func, x, actual, expected, tol) in cases {
            assert!(
                (actual - expected).abs() <= tol,
                "{func}({x}) = {actual}, expected {expected}"
            );
        }
        assert_eq!(berp(0.0), 0.0);
        assert_eq!(beip(0.0), 0.0);
        assert!(kerp(0.0).is_infinite() && kerp(0.0).is_sign_negative());
        assert_eq!(keip(0.0), 0.0);
        assert!(kerp(-1.0).is_nan());
        assert!(keip(-1.0).is_nan());
    }

    #[test]
    fn combined_kelvin_matches_component_reference_points() {
        let (be, ke, bep, kep) = kelvin(1.0);
        assert!((be.re - 0.984_381_781_213_087).abs() < 1.0e-12);
        assert!((be.im - 0.249_566_040_036_659_72).abs() < 1.0e-12);
        assert!((ke.re - 0.286_706_208_728_316_04).abs() < 1.0e-3);
        assert!((ke.im - (-0.494_994_636_518_72)).abs() < 1.0e-3);
        assert!((bep.re - (-0.062_445_752_179_030_96)).abs() < 1.0e-12);
        assert!((bep.im - 0.497_396_511_468_097_27).abs() < 1.0e-12);
        assert!((kep.re - (-0.694_603_891_100_690_8)).abs() < 1.0e-3);
        assert!((kep.im - 0.352_369_913_336_170_5).abs() < 1.0e-3);

        let (_be, ke_neg, _bep, kep_neg) = kelvin(-1.0);
        assert!(ke_neg.re.is_nan());
        assert!(ke_neg.im.is_nan());
        assert!(kep_neg.re.is_nan());
        assert!(kep_neg.im.is_nan());
    }

    #[test]
    fn kolmogorov_basic() {
        // kolmogorov(0) = 1 (survival function at 0)
        assert!((kolmogorov_scalar(0.0) - 1.0).abs() < 1e-10);

        // kolmogorov is monotonically decreasing
        assert!(kolmogorov_scalar(0.5) > kolmogorov_scalar(1.0));
        assert!(kolmogorov_scalar(1.0) > kolmogorov_scalar(1.5));
        assert!(kolmogorov_scalar(1.5) > kolmogorov_scalar(2.0));

        // Known value: kolmogorov(1.0) ≈ 0.27
        let k1 = kolmogorov_scalar(1.0);
        assert!(
            (k1 - 0.27).abs() < 0.02,
            "kolmogorov(1.0) = {k1}, expected ~0.27"
        );

        // For large y, kolmogorov(y) -> 0
        assert!(kolmogorov_scalar(3.0) < 0.001);
    }

    #[test]
    fn kolmogi_inverse() {
        // kolmogi should be inverse of kolmogorov
        for &y in &[0.5, 1.0, 1.5, 2.0, 2.5] {
            let p = kolmogorov_scalar(y);
            if p > 0.001 && p < 0.999 {
                let y_recovered = kolmogi_scalar(p);
                assert!(
                    (y_recovered - y).abs() < 0.001,
                    "kolmogi failed: y={y}, p={p}, y_recovered={y_recovered}"
                );
            }
        }
    }

    #[test]
    fn kolmogi_endpoints() {
        // kolmogi(0) = +inf
        assert!(kolmogi_scalar(0.0).is_infinite() && kolmogi_scalar(0.0).is_sign_positive());
        // kolmogi(1) = 0
        assert!((kolmogi_scalar(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn kolmogi_matches_scipy_reference() {
        // Reference values from scipy.special.kolmogi. p = 0.5 previously
        // diverged to ~3.8e10 because Newton overshot from a fixed seed
        // (frankenscipy-or0dc).
        for &(p, expected) in &[
            (0.5_f64, 0.827_573_555_189_905_9),
            (0.25, 1.019_184_720_253_685_7),
            (0.75, 0.676_447_691_502_820_1),
            (0.1, 1.223_847_870_217_082_3),
            (0.9, 0.571_173_265_106_340_1),
            (0.01, 1.627_623_611_518_950_4),
        ] {
            let got = kolmogi_scalar(p);
            assert!(
                (got - expected).abs() < 1e-9,
                "kolmogi({p}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn kolmogorov_tensor_dispatch_is_monotone() -> Result<(), String> {
        let scalar = kolmogorov(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - kolmogorov_scalar(1.0)).abs() < 1e-14);

        let vector = kolmogorov(
            &SpecialTensor::RealVec(vec![0.5, 1.0, 1.5, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert_eq!(values.len(), 4);
        assert!(values[0] > values[1]);
        assert!(values[1] > values[2]);
        assert!(values[2] > values[3]);
        Ok(())
    }

    #[test]
    fn kolmogi_tensor_dispatch_round_trips() -> Result<(), String> {
        let expected = vec![0.5, 1.0, 1.5];
        let probabilities = kolmogorov(
            &SpecialTensor::RealVec(expected.clone()),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let probability_values = expect_real_vec(probabilities)?;
        let recovered = kolmogi(
            &SpecialTensor::RealVec(probability_values),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let recovered_values = expect_real_vec(recovered)?;
        assert_eq!(recovered_values.len(), expected.len());
        for (actual, expected) in recovered_values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    fn smirnov_basic() {
        // smirnov(n, 0) = 1
        assert!((smirnov(10, 0.0) - 1.0).abs() < 1e-10);
        // smirnov(n, 1) = 0
        assert!((smirnov(10, 1.0) - 0.0).abs() < 1e-10);

        // smirnov is monotonically decreasing in d for larger d values
        for n in &[10, 20, 50] {
            assert!(
                smirnov(*n, 0.3) > smirnov(*n, 0.5),
                "n={n}: smirnov(0.3) > smirnov(0.5)"
            );
            assert!(
                smirnov(*n, 0.5) > smirnov(*n, 0.7),
                "n={n}: smirnov(0.5) > smirnov(0.7)"
            );
        }

        // smirnov(n, d) is in [0, 1]
        for n in &[10, 20, 50, 100] {
            for &d in &[0.1, 0.3, 0.5, 0.7] {
                let s = smirnov(*n, d);
                assert!(
                    (0.0..=1.0).contains(&s),
                    "smirnov({n}, {d}) = {s} out of range"
                );
            }
        }
    }

    #[test]
    fn smirnovi_inverse() {
        // smirnovi should be inverse of smirnov (within tolerance)
        for &n in &[20, 50, 100] {
            for &d in &[0.2, 0.3, 0.4, 0.5] {
                let p = smirnov(n, d);
                if p > 0.01 && p < 0.99 {
                    let d_recovered = smirnovi(n, p);
                    assert!(
                        (d_recovered - d).abs() < 0.05,
                        "smirnovi failed: n={n}, d={d}, p={p}, d_recovered={d_recovered}"
                    );
                }
            }
        }
    }

    /// Anchor smirnov against scipy.special.smirnov (frankenscipy-5bura).
    ///
    /// Pre-fix the asymptotic-with-correction was off by up to ~0.3 abs
    /// at small n (e.g. smirnov(1, 0.5) returned ≈0.39 vs the exact 0.5).
    /// The new Birnbaum-Tingey exact branch matches scipy to floating
    /// point for n ≤ 100. The asymptotic kicks in for n ≥ 1000 and
    /// stays within the well-known O(1/n) bound.
    #[test]
    fn smirnov_matches_scipy_at_small_n() {
        // (n, d, scipy.special.smirnov(n, d))
        let cases: [(i32, f64, f64); 14] = [
            (1, 0.1, 0.9),
            (1, 0.5, 0.5),
            (1, 0.8, 0.2),
            (2, 0.3, 0.61),
            (2, 0.5, 0.25),
            (5, 0.1, 0.85359),
            (5, 0.3, 0.34282),
            (5, 0.5, 0.056),
            (10, 0.1, 0.7642052309),
            (10, 0.3, 0.1354635556),
            (10, 0.5, 0.003888705),
            (50, 0.1, 0.34490701996888),
            (100, 0.1, 0.12659065846),
            (500, 0.1, 4.171146533e-5),
        ];
        for (n, d, expected) in cases {
            let got = smirnov(n, d);
            let scale = expected.abs().max(1e-12);
            let rel = (got - expected).abs() / scale;
            assert!(
                rel < 1e-9,
                "smirnov({n}, {d}) = {got}, expected {expected}, rel = {rel}"
            );
        }
    }

    /// Cl₂ accuracy after the truncation+stop fix (frankenscipy-cho22).
    ///
    /// Pre-fix the relative-to-sum convergence check returned the
    /// trivial partial sum (e.g. 1.0 at θ = π/2 instead of Catalan
    /// ≈0.916, since sin(2·π/2)=0 caused the loop to exit at k=2).
    /// After the fix the cap+symmetry path delivers ≤ 6e-8 abs across
    /// the supported range, with most cases ≤1e-10.
    #[test]
    fn clausen_matches_reference_after_truncation_fix() {
        let pi = std::f64::consts::PI;
        // (θ, reference Cl2(θ) from N=10⁶ direct sum)
        let cases: [(f64, f64); 8] = [
            (0.1, 0.330272346694),
            (0.5, 0.848311869671),
            (1.0, 1.013959139535),
            (pi / 3.0, 1.014941606410),
            (pi / 2.0, 0.915965594177),
            (2.0 * pi / 3.0, 0.676627737607),
            (pi, 0.0),
            (3.0 * pi / 2.0, -0.915965594177),
        ];
        for (t, expected) in cases {
            let got = clausen(t);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-6,
                "clausen({t}) = {got}, expected {expected}, diff = {diff}"
            );
        }
    }

    #[test]
    fn degree_trig_exact_values() {
        // Test exact values at common angles
        assert!((cosdg(0.0) - 1.0).abs() < 1e-15);
        assert!((cosdg(90.0) - 0.0).abs() < 1e-15);
        assert!((cosdg(180.0) - (-1.0)).abs() < 1e-15);
        assert!((cosdg(270.0) - 0.0).abs() < 1e-15);
        assert!((cosdg(360.0) - 1.0).abs() < 1e-15);

        assert!((sindg(0.0) - 0.0).abs() < 1e-15);
        assert!((sindg(90.0) - 1.0).abs() < 1e-15);
        assert!((sindg(180.0) - 0.0).abs() < 1e-15);
        assert!((sindg(270.0) - (-1.0)).abs() < 1e-15);

        assert!((tandg(0.0) - 0.0).abs() < 1e-15);
        assert!((tandg(45.0) - 1.0).abs() < 1e-15);
        assert!(tandg(90.0).is_infinite());

        assert!(cotdg(0.0).is_infinite());
        assert!((cotdg(45.0) - 1.0).abs() < 1e-15);
        assert!((cotdg(90.0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn degree_trig_general() {
        // Test general values match radians conversion
        for &deg in &[30.0, 45.0, 60.0, 120.0, 150.0, 210.0, 300.0] {
            let rad = deg * std::f64::consts::PI / 180.0;
            assert!(
                (cosdg(deg) - rad.cos()).abs() < 1e-14,
                "cosdg({deg}) failed"
            );
            assert!(
                (sindg(deg) - rad.sin()).abs() < 1e-14,
                "sindg({deg}) failed"
            );
        }
    }

    #[test]
    fn radian_basic() {
        // 180 degrees = π radians
        assert!((radian(180.0, 0.0, 0.0) - std::f64::consts::PI).abs() < 1e-14);

        // 90 degrees = π/2 radians
        assert!((radian(90.0, 0.0, 0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-14);

        // 45 degrees 30 minutes = 45.5 degrees
        let expected = 45.5 * std::f64::consts::PI / 180.0;
        assert!((radian(45.0, 30.0, 0.0) - expected).abs() < 1e-14);

        // 1 degree 1 minute 1 second
        let expected = (1.0 + 1.0 / 60.0 + 1.0 / 3600.0) * std::f64::consts::PI / 180.0;
        assert!((radian(1.0, 1.0, 1.0) - expected).abs() < 1e-14);
    }

    #[test]
    fn cbrt_basic() {
        // Positive values
        assert!((cbrt(8.0) - 2.0).abs() < 1e-14);
        assert!((cbrt(27.0) - 3.0).abs() < 1e-14);
        assert!((cbrt(1.0) - 1.0).abs() < 1e-14);

        // Negative values (key feature - handles negative inputs)
        assert!((cbrt(-8.0) - (-2.0)).abs() < 1e-14);
        assert!((cbrt(-27.0) - (-3.0)).abs() < 1e-14);

        // Zero
        assert!((cbrt(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn exp_log_functions() {
        // exp2: 2^x
        assert!((exp2(0.0) - 1.0).abs() < 1e-14);
        assert!((exp2(1.0) - 2.0).abs() < 1e-14);
        assert!((exp2(3.0) - 8.0).abs() < 1e-14);
        assert!((exp2(-1.0) - 0.5).abs() < 1e-14);

        // exp10: 10^x
        assert!((exp10(0.0) - 1.0).abs() < 1e-14);
        assert!((exp10(1.0) - 10.0).abs() < 1e-13);
        assert!((exp10(2.0) - 100.0).abs() < 1e-12);

        // log2
        assert!((log2(1.0) - 0.0).abs() < 1e-14);
        assert!((log2(2.0) - 1.0).abs() < 1e-14);
        assert!((log2(8.0) - 3.0).abs() < 1e-14);

        // log10
        assert!((log10(1.0) - 0.0).abs() < 1e-14);
        assert!((log10(10.0) - 1.0).abs() < 1e-14);
        assert!((log10(100.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn round_basic() {
        assert!((round(1.4) - 1.0).abs() < 1e-14);
        assert!((round(1.5) - 2.0).abs() < 1e-14);
        assert!((round(1.6) - 2.0).abs() < 1e-14);
        assert!((round(-1.4) - (-1.0)).abs() < 1e-14);
        assert!((round(-1.5) - (-2.0)).abs() < 1e-14);
        assert!((round(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn floor_ceil_trunc() {
        // floor
        assert!((floor(1.7) - 1.0).abs() < 1e-14);
        assert!((floor(-1.7) - (-2.0)).abs() < 1e-14);
        assert!((floor(2.0) - 2.0).abs() < 1e-14);

        // ceil
        assert!((ceil(1.3) - 2.0).abs() < 1e-14);
        assert!((ceil(-1.3) - (-1.0)).abs() < 1e-14);
        assert!((ceil(2.0) - 2.0).abs() < 1e-14);

        // trunc (towards zero)
        assert!((trunc(1.7) - 1.0).abs() < 1e-14);
        assert!((trunc(-1.7) - (-1.0)).abs() < 1e-14);
        assert!((trunc(2.0) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn sign_basic() {
        assert!((sign(5.0) - 1.0).abs() < 1e-14);
        assert!((sign(-5.0) - (-1.0)).abs() < 1e-14);
        assert!((sign(0.0) - 0.0).abs() < 1e-14);
        assert!(sign(f64::NAN).is_nan());
    }

    #[test]
    fn heaviside_basic() {
        assert!((heaviside(1.0, 0.5) - 1.0).abs() < 1e-14);
        assert!((heaviside(-1.0, 0.5) - 0.0).abs() < 1e-14);
        assert!((heaviside(0.0, 0.5) - 0.5).abs() < 1e-14);
        assert!((heaviside(0.0, 0.0) - 0.0).abs() < 1e-14);
        assert!((heaviside(0.0, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn hypot_basic() {
        // Classic 3-4-5 triangle
        assert!((hypot(3.0, 4.0) - 5.0).abs() < 1e-14);
        // 5-12-13 triangle
        assert!((hypot(5.0, 12.0) - 13.0).abs() < 1e-14);
        // Handles zeros
        assert!((hypot(3.0, 0.0) - 3.0).abs() < 1e-14);
        assert!((hypot(0.0, 4.0) - 4.0).abs() < 1e-14);
        // Negative values
        assert!((hypot(-3.0, 4.0) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn copysign_basic() {
        assert!((copysign(3.0, 1.0) - 3.0).abs() < 1e-14);
        assert!((copysign(3.0, -1.0) - (-3.0)).abs() < 1e-14);
        assert!((copysign(-3.0, 1.0) - 3.0).abs() < 1e-14);
        assert!((copysign(-3.0, -1.0) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn ldexp_basic() {
        // ldexp(x, n) = x * 2^n
        assert!((ldexp(1.0, 0) - 1.0).abs() < 1e-14);
        assert!((ldexp(1.0, 1) - 2.0).abs() < 1e-14);
        assert!((ldexp(1.0, 2) - 4.0).abs() < 1e-14);
        assert!((ldexp(1.0, -1) - 0.5).abs() < 1e-14);
        assert!((ldexp(3.0, 2) - 12.0).abs() < 1e-14);
    }

    #[test]
    fn frexp_basic() {
        // frexp returns (mantissa, exp) where x = mantissa * 2^exp
        // and 0.5 <= |mantissa| < 1.0
        let (m, e) = frexp(1.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 1);

        let (m, e) = frexp(2.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 2);

        let (m, e) = frexp(8.0);
        assert!((m - 0.5).abs() < 1e-14);
        assert_eq!(e, 4);

        let (m, e) = frexp(0.0);
        assert!((m - 0.0).abs() < 1e-14);
        assert_eq!(e, 0);

        // Verify roundtrip: ldexp(frexp(x)) == x
        for &x in &[0.5, 1.0, 2.0, std::f64::consts::PI, 100.0, 0.001] {
            let (m, e) = frexp(x);
            assert!((ldexp(m, e) - x).abs() < 1e-14);
        }
    }

    #[test]
    fn fabs_basic() {
        assert!((fabs(3.0) - 3.0).abs() < 1e-14);
        assert!((fabs(-3.0) - 3.0).abs() < 1e-14);
        assert!((fabs(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn clip_basic() {
        // Value within range
        assert!((clip(5.0, 0.0, 10.0) - 5.0).abs() < 1e-14);
        // Value below min
        assert!((clip(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-14);
        // Value above max
        assert!((clip(15.0, 0.0, 10.0) - 10.0).abs() < 1e-14);
        // At boundaries
        assert!((clip(0.0, 0.0, 10.0) - 0.0).abs() < 1e-14);
        assert!((clip(10.0, 0.0, 10.0) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn sinpi_basic() {
        // sinpi(0) = 0
        assert!((sinpi(0.0) - 0.0).abs() < 1e-14);
        // sinpi(0.5) = 1
        assert!((sinpi(0.5) - 1.0).abs() < 1e-14);
        // sinpi(1) = 0
        assert!((sinpi(1.0) - 0.0).abs() < 1e-14);
        // sinpi(-0.5) = -1
        assert!((sinpi(-0.5) - (-1.0)).abs() < 1e-14);
        // sinpi(1.5) = -1
        assert!((sinpi(1.5) - (-1.0)).abs() < 1e-14);
        // sinpi(2) = 0
        assert!((sinpi(2.0) - 0.0).abs() < 1e-14);
        // Non-special values
        assert!((sinpi(0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
        // Negative
        assert!((sinpi(-0.25) - (-std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-14);
    }

    #[test]
    fn cospi_basic() {
        // cospi(0) = 1
        assert!((cospi(0.0) - 1.0).abs() < 1e-14);
        // cospi(0.5) = 0
        assert!((cospi(0.5) - 0.0).abs() < 1e-14);
        // cospi(1) = -1
        assert!((cospi(1.0) - (-1.0)).abs() < 1e-14);
        // cospi(-0.5) = 0
        assert!((cospi(-0.5) - 0.0).abs() < 1e-14);
        // cospi(2) = 1
        assert!((cospi(2.0) - 1.0).abs() < 1e-14);
        // Non-special values
        assert!((cospi(0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
        // Negative
        assert!((cospi(-0.25) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
    }

    #[test]
    fn tanpi_integer_arguments_are_exact_zero() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-cd205]: scipy.special.tanpi at integer
        // arguments is exactly 0 (not the ±epsilon that tan(π·n) gives).
        for &n in &[-3.0_f64, -1.0, 0.0, 1.0, 2.0, 5.0] {
            assert_eq!(
                tanpi(n),
                0.0,
                "tanpi({n}) must be exactly 0, got {}",
                tanpi(n)
            );
        }
    }

    #[test]
    fn tanpi_quarter_integers_match_unit_value() {
        // tanpi(0.25) = tan(π/4) = 1; tanpi(-0.25) = -1.
        assert!((tanpi(0.25) - 1.0).abs() < 1e-14);
        assert!((tanpi(-0.25) - (-1.0)).abs() < 1e-14);
        // tanpi(0.75) = tan(3π/4) = -1; tanpi(-0.75) = +1.
        assert!((tanpi(0.75) - (-1.0)).abs() < 1e-14);
        assert!((tanpi(-0.75) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn tanpi_half_integers_are_infinite() {
        // tanpi at half-integers blows up because cospi vanishes.
        // Specifically: 0.5 → +∞ (sinpi=+1, cospi=0⁺); 1.5 → +∞;
        // -0.5 → -∞ (sinpi=-1, cospi=0⁺).
        assert!(tanpi(0.5).is_infinite());
        assert!(tanpi(-0.5).is_infinite());
        assert!(tanpi(1.5).is_infinite());
        assert!(tanpi(-1.5).is_infinite());
    }

    #[test]
    fn tanpi_propagates_nan_and_infinity() {
        assert!(tanpi(f64::NAN).is_nan());
        assert!(tanpi(f64::INFINITY).is_nan());
        assert!(tanpi(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn tanpi_matches_sinpi_over_cospi_for_regular_points() {
        // Direct identity: tanpi(x) = sinpi(x) / cospi(x) for regular
        // x. This is essentially the implementation, so the test is a
        // smoke check on a grid of non-special values.
        for &x in &[0.1_f64, 0.3, 0.7, 1.2, -0.4, -1.7] {
            let expected = sinpi(x) / cospi(x);
            assert!(
                (tanpi(x) - expected).abs() < 1e-14,
                "tanpi({x}) = {} != sinpi/cospi = {expected}",
                tanpi(x)
            );
        }
    }

    #[test]
    fn log1p_basic() {
        // log1p(0) = 0
        assert!((log1p(0.0) - 0.0).abs() < 1e-14);
        // log1p(e-1) = 1
        assert!((log1p(std::f64::consts::E - 1.0) - 1.0).abs() < 1e-14);
        // Small values - this is where log1p shines
        let small = 1e-15;
        assert!((log1p(small) - small).abs() < 1e-28);
    }

    #[test]
    fn expm1_basic() {
        // expm1(0) = 0
        assert!((expm1(0.0) - 0.0).abs() < 1e-14);
        // expm1(1) = e-1
        assert!((expm1(1.0) - (std::f64::consts::E - 1.0)).abs() < 1e-14);
        // Small values - this is where expm1 shines
        let small = 1e-15;
        assert!((expm1(small) - small).abs() < 1e-28);
    }

    #[test]
    fn logaddexp_basic() {
        // logaddexp(0, 0) = log(2)
        assert!((logaddexp(0.0, 0.0) - std::f64::consts::LN_2).abs() < 1e-14);
        // logaddexp(x, -inf) = x
        assert!((logaddexp(1.0, f64::NEG_INFINITY) - 1.0).abs() < 1e-14);
        // logaddexp(-inf, x) = x
        assert!((logaddexp(f64::NEG_INFINITY, 2.0) - 2.0).abs() < 1e-14);
        // For large difference, result ≈ max
        assert!((logaddexp(100.0, 0.0) - 100.0).abs() < 1e-10);
        // Symmetric
        assert!((logaddexp(1.0, 2.0) - logaddexp(2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn logaddexp2_basic() {
        // logaddexp2(0, 0) = 1  (log2(2^0 + 2^0) = log2(2) = 1)
        assert!((logaddexp2(0.0, 0.0) - 1.0).abs() < 1e-14);
        // logaddexp2(x, -inf) = x
        assert!((logaddexp2(1.0, f64::NEG_INFINITY) - 1.0).abs() < 1e-14);
        // logaddexp2(-inf, x) = x
        assert!((logaddexp2(f64::NEG_INFINITY, 2.0) - 2.0).abs() < 1e-14);
        // For large difference, result ≈ max
        assert!((logaddexp2(100.0, 0.0) - 100.0).abs() < 1e-10);
        // Symmetric
        assert!((logaddexp2(1.0, 2.0) - logaddexp2(2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn nextafter_basic() {
        // Moving toward larger value increases
        let next = nextafter(1.0, 2.0);
        assert!(next > 1.0);
        assert!(next < 1.0 + 1e-14);

        // Moving toward smaller value decreases
        let prev = nextafter(1.0, 0.0);
        assert!(prev < 1.0);
        assert!(prev > 1.0 - 1e-14);

        // From zero
        let tiny = nextafter(0.0, 1.0);
        assert!(tiny > 0.0);
        let neg_tiny = nextafter(0.0, -1.0);
        assert!(neg_tiny < 0.0);

        // Same value returns itself
        assert_eq!(nextafter(1.0, 1.0), 1.0);
    }

    #[test]
    fn spacing_basic() {
        // Spacing at 1.0 is machine epsilon
        let eps = spacing(1.0);
        assert!((eps - f64::EPSILON).abs() < 1e-30);

        // Spacing is always positive
        assert!(spacing(1.0) > 0.0);
        assert!(spacing(-1.0) > 0.0);
        assert!(spacing(0.0) > 0.0);

        // Spacing gets larger for larger numbers
        assert!(spacing(1e10) > spacing(1.0));
    }

    #[test]
    fn modf_basic() {
        // Positive values
        let (frac, int) = modf(3.5);
        assert!((frac - 0.5).abs() < 1e-14);
        assert!((int - 3.0).abs() < 1e-14);

        // Negative values
        let (frac, int) = modf(-3.5);
        assert!((frac - (-0.5)).abs() < 1e-14);
        assert!((int - (-3.0)).abs() < 1e-14);

        // Integer values
        let (frac, int) = modf(4.0);
        assert!((frac - 0.0).abs() < 1e-14);
        assert!((int - 4.0).abs() < 1e-14);

        // Zero
        let (frac, int) = modf(0.0);
        assert!((frac - 0.0).abs() < 1e-14);
        assert!((int - 0.0).abs() < 1e-14);
    }

    #[test]
    fn signbit_basic() {
        // Positive
        assert!(!signbit(1.0));
        // Negative
        assert!(signbit(-1.0));
        // Zero
        assert!(!signbit(0.0));
        // Negative zero
        assert!(signbit(-0.0));
    }

    #[test]
    fn float_classification() {
        // isnan
        assert!(isnan(f64::NAN));
        assert!(!isnan(1.0));
        assert!(!isnan(f64::INFINITY));

        // isinf
        assert!(isinf(f64::INFINITY));
        assert!(isinf(f64::NEG_INFINITY));
        assert!(!isinf(1.0));
        assert!(!isinf(f64::NAN));

        // isfinite
        assert!(isfinite(1.0));
        assert!(isfinite(0.0));
        assert!(!isfinite(f64::INFINITY));
        assert!(!isfinite(f64::NEG_INFINITY));
        assert!(!isfinite(f64::NAN));

        // isposinf
        assert!(isposinf(f64::INFINITY));
        assert!(!isposinf(f64::NEG_INFINITY));
        assert!(!isposinf(1.0));

        // isneginf
        assert!(isneginf(f64::NEG_INFINITY));
        assert!(!isneginf(f64::INFINITY));
        assert!(!isneginf(-1.0));
    }

    #[test]
    fn reciprocal_basic() {
        assert!((reciprocal(2.0) - 0.5).abs() < 1e-14);
        assert!((reciprocal(4.0) - 0.25).abs() < 1e-14);
        assert!((reciprocal(-2.0) - (-0.5)).abs() < 1e-14);
        assert!(reciprocal(0.0).is_infinite());
    }

    #[test]
    fn square_basic() {
        assert!((square(2.0) - 4.0).abs() < 1e-14);
        assert!((square(3.0) - 9.0).abs() < 1e-14);
        assert!((square(-2.0) - 4.0).abs() < 1e-14);
        assert!((square(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn positive_negative_parts() {
        // Positive part
        assert!((positive(3.0) - 3.0).abs() < 1e-14);
        assert!((positive(-3.0) - 0.0).abs() < 1e-14);
        assert!((positive(0.0) - 0.0).abs() < 1e-14);

        // Negative part
        assert!((negative(3.0) - 0.0).abs() < 1e-14);
        assert!((negative(-3.0) - 3.0).abs() < 1e-14);
        assert!((negative(0.0) - 0.0).abs() < 1e-14);

        // positive(x) + negative(x) == |x|
        for &x in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            assert!((positive(x) + negative(x) - x.abs()).abs() < 1e-14);
        }
    }

    #[test]
    fn deg2rad_rad2deg() {
        use std::f64::consts::PI;

        // deg2rad
        assert!((deg2rad(0.0) - 0.0).abs() < 1e-14);
        assert!((deg2rad(180.0) - PI).abs() < 1e-14);
        assert!((deg2rad(90.0) - PI / 2.0).abs() < 1e-14);
        assert!((deg2rad(360.0) - 2.0 * PI).abs() < 1e-14);
        assert!((deg2rad(-90.0) - (-PI / 2.0)).abs() < 1e-14);

        // rad2deg
        assert!((rad2deg(0.0) - 0.0).abs() < 1e-14);
        assert!((rad2deg(PI) - 180.0).abs() < 1e-12);
        assert!((rad2deg(PI / 2.0) - 90.0).abs() < 1e-12);
        assert!((rad2deg(2.0 * PI) - 360.0).abs() < 1e-12);

        // Roundtrip
        for &deg in &[0.0, 45.0, 90.0, 180.0, 270.0, 360.0] {
            assert!((rad2deg(deg2rad(deg)) - deg).abs() < 1e-12);
        }
    }

    #[test]
    fn rint_basic() {
        // Round to nearest, ties to even (banker's rounding)
        assert!((rint(1.4) - 1.0).abs() < 1e-14);
        assert!((rint(1.6) - 2.0).abs() < 1e-14);
        assert!((rint(-1.4) - (-1.0)).abs() < 1e-14);
        assert!((rint(-1.6) - (-2.0)).abs() < 1e-14);

        // Ties go to even
        assert!((rint(0.5) - 0.0).abs() < 1e-14); // 0 is even
        assert!((rint(1.5) - 2.0).abs() < 1e-14); // 2 is even
        assert!((rint(2.5) - 2.0).abs() < 1e-14); // 2 is even
        assert!((rint(3.5) - 4.0).abs() < 1e-14); // 4 is even
    }

    #[test]
    fn fix_basic() {
        // Round towards zero
        assert!((fix(1.7) - 1.0).abs() < 1e-14);
        assert!((fix(-1.7) - (-1.0)).abs() < 1e-14);
        assert!((fix(2.9) - 2.0).abs() < 1e-14);
        assert!((fix(-2.9) - (-2.0)).abs() < 1e-14);
        assert!((fix(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn divmod_basic() {
        // Basic division
        let (q, r) = divmod(7.0, 3.0);
        assert!((q - 2.0).abs() < 1e-14);
        assert!((r - 1.0).abs() < 1e-14);

        // Negative dividend
        let (q, r) = divmod(-7.0, 3.0);
        assert!((q - (-3.0)).abs() < 1e-14);
        assert!((r - 2.0).abs() < 1e-14);

        // Negative divisor
        let (q, r) = divmod(7.0, -3.0);
        assert!((q - (-3.0)).abs() < 1e-14);
        assert!((r - (-2.0)).abs() < 1e-14);

        // Both negative
        let (q, r) = divmod(-7.0, -3.0);
        assert!((q - 2.0).abs() < 1e-14);
        assert!((r - (-1.0)).abs() < 1e-14);

        // Division by zero
        let (q, r) = divmod(1.0, 0.0);
        assert!(q.is_nan());
        assert!(r.is_nan());
    }

    #[test]
    fn maximum_minimum_basic() {
        // maximum - basic
        assert!((maximum(3.0, 5.0) - 5.0).abs() < 1e-14);
        assert!((maximum(5.0, 3.0) - 5.0).abs() < 1e-14);
        assert!((maximum(-1.0, 1.0) - 1.0).abs() < 1e-14);

        // maximum - propagates NaN
        assert!(maximum(f64::NAN, 1.0).is_nan());
        assert!(maximum(1.0, f64::NAN).is_nan());
        assert!(maximum(f64::NAN, f64::NAN).is_nan());

        // minimum - basic
        assert!((minimum(3.0, 5.0) - 3.0).abs() < 1e-14);
        assert!((minimum(5.0, 3.0) - 3.0).abs() < 1e-14);
        assert!((minimum(-1.0, 1.0) - (-1.0)).abs() < 1e-14);

        // minimum - propagates NaN
        assert!(minimum(f64::NAN, 1.0).is_nan());
        assert!(minimum(1.0, f64::NAN).is_nan());
        assert!(minimum(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn fmax_fmin_basic() {
        // fmax - basic
        assert!((fmax(3.0, 5.0) - 5.0).abs() < 1e-14);
        assert!((fmax(5.0, 3.0) - 5.0).abs() < 1e-14);

        // fmax - ignores NaN
        assert!((fmax(f64::NAN, 1.0) - 1.0).abs() < 1e-14);
        assert!((fmax(1.0, f64::NAN) - 1.0).abs() < 1e-14);
        assert!(fmax(f64::NAN, f64::NAN).is_nan());

        // fmin - basic
        assert!((fmin(3.0, 5.0) - 3.0).abs() < 1e-14);
        assert!((fmin(5.0, 3.0) - 3.0).abs() < 1e-14);

        // fmin - ignores NaN
        assert!((fmin(f64::NAN, 1.0) - 1.0).abs() < 1e-14);
        assert!((fmin(1.0, f64::NAN) - 1.0).abs() < 1e-14);
        assert!(fmin(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn power_basic() {
        assert!((power(2.0, 3.0) - 8.0).abs() < 1e-14);
        assert!((power(3.0, 2.0) - 9.0).abs() < 1e-14);
        assert!((power(4.0, 0.5) - 2.0).abs() < 1e-14);
        assert!((power(2.0, -1.0) - 0.5).abs() < 1e-14);
        assert!((power(0.0, 0.0) - 1.0).abs() < 1e-14); // 0^0 = 1 by convention
    }

    #[test]
    fn fdiff_basic() {
        assert!((fdiff(5.0, 3.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(3.0, 5.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(-1.0, 1.0) - 2.0).abs() < 1e-14);
        assert!((fdiff(0.0, 0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn nan_to_num_basic() {
        // NaN replacement
        assert!((nan_to_num(f64::NAN, 0.0, f64::MAX, f64::MIN) - 0.0).abs() < 1e-14);
        assert!((nan_to_num(f64::NAN, 99.0, f64::MAX, f64::MIN) - 99.0).abs() < 1e-14);

        // Infinity replacement
        assert!((nan_to_num(f64::INFINITY, 0.0, 1e10, -1e10) - 1e10).abs() < 1e-14);
        assert!((nan_to_num(f64::NEG_INFINITY, 0.0, 1e10, -1e10) - (-1e10)).abs() < 1e-14);

        // Finite values unchanged
        assert!((nan_to_num(5.0, 0.0, f64::MAX, f64::MIN) - 5.0).abs() < 1e-14);
        assert!((nan_to_num(-3.0, 0.0, f64::MAX, f64::MIN) - (-3.0)).abs() < 1e-14);
    }

    #[test]
    fn relu_basic() {
        // Positive values pass through
        assert!((relu_scalar(5.0) - 5.0).abs() < 1e-14);
        assert!((relu_scalar(0.1) - 0.1).abs() < 1e-14);

        // Negative values become zero
        assert!((relu_scalar(-5.0) - 0.0).abs() < 1e-14);
        assert!((relu_scalar(-0.1) - 0.0).abs() < 1e-14);

        // Zero stays zero
        assert!((relu_scalar(0.0) - 0.0).abs() < 1e-14);

        // NaN propagates
        assert!(relu_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn relu_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = relu(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - relu_scalar(2.0)).abs() < 1e-14);

        let vector = relu(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(relu_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn relu_tensor_dispatch_preserves_nan() -> Result<(), String> {
        let vector = relu(
            &SpecialTensor::RealVec(vec![f64::NAN, -1.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 1.0);
        Ok(())
    }

    #[test]
    fn softplus_basic() {
        // softplus(0) = ln(2)
        assert!((softplus_scalar(0.0) - std::f64::consts::LN_2).abs() < 1e-14);

        // For large positive x, softplus(x) ≈ x
        assert!((softplus_scalar(100.0) - 100.0).abs() < 1e-10);

        // For large negative x, softplus(x) ≈ 0
        assert!(softplus_scalar(-100.0) < 1e-40);

        // softplus is always positive
        assert!(softplus_scalar(-10.0) > 0.0);
        assert!(softplus_scalar(0.0) > 0.0);
        assert!(softplus_scalar(10.0) > 0.0);

        // NaN propagates
        assert!(softplus_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn softplus_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = softplus(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - softplus_scalar(2.0)).abs() < 1e-14);

        let vector = softplus(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(softplus_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn softplus_tensor_dispatch_preserves_tail_stability() -> Result<(), String> {
        let vector = softplus(
            &SpecialTensor::RealVec(vec![f64::NAN, -100.0, 100.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1] < 1e-40);
        assert!((values[2] - 100.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn huber_basic() {
        let delta = 1.0;

        // For |x| <= delta, huber = 0.5 * x^2
        assert!((huber_scalar(delta, 0.5) - 0.125).abs() < 1e-14);
        assert!((huber_scalar(delta, -0.5) - 0.125).abs() < 1e-14);
        assert!((huber_scalar(delta, 0.0) - 0.0).abs() < 1e-14);

        // For |x| > delta, huber = delta * (|x| - 0.5*delta)
        assert!((huber_scalar(delta, 2.0) - 1.5).abs() < 1e-14);
        assert!((huber_scalar(delta, -2.0) - 1.5).abs() < 1e-14);

        // At boundary
        assert!((huber_scalar(delta, 1.0) - 0.5).abs() < 1e-14);

        // Invalid delta
        assert!(huber_scalar(0.0, 1.0).is_nan());
        assert!(huber_scalar(-1.0, 1.0).is_nan());
    }

    #[test]
    fn pseudo_huber_basic() {
        let delta = 1.0;

        // pseudo_huber(delta, 0) = 0
        assert!((pseudo_huber_scalar(delta, 0.0) - 0.0).abs() < 1e-14);

        // For small x, pseudo_huber ≈ 0.5 * x^2
        let small = 0.01;
        let expected = 0.5 * small * small;
        assert!((pseudo_huber_scalar(delta, small) - expected).abs() < 1e-6);

        // Symmetric
        assert!((pseudo_huber_scalar(delta, 2.0) - pseudo_huber_scalar(delta, -2.0)).abs() < 1e-14);

        // Invalid delta
        assert!(pseudo_huber_scalar(0.0, 1.0).is_nan());
        assert!(pseudo_huber_scalar(-1.0, 1.0).is_nan());
    }

    #[test]
    fn huber_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = huber(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(2.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - huber_scalar(1.0, 2.0)).abs() < 1e-14);

        let vector = huber(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealVec(vec![-2.0, -0.5, 0.5, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, -0.5, 0.5, 2.0].map(|x| huber_scalar(1.0, x));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }

        let broadcast = huber(
            &SpecialTensor::RealVec(vec![0.5, 1.0, 2.0]),
            &SpecialTensor::RealScalar(1.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let broadcast_values = expect_real_vec(broadcast)?;
        let broadcast_expected = [0.5, 1.0, 2.0].map(|delta| huber_scalar(delta, 1.5));
        assert_eq!(broadcast_values.len(), broadcast_expected.len());
        for (actual, expected) in broadcast_values.iter().zip(broadcast_expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn pseudo_huber_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = pseudo_huber(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(2.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - pseudo_huber_scalar(1.0, 2.0)).abs() < 1e-14);

        let vector = pseudo_huber(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealVec(vec![-2.0, -0.5, 0.5, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, -0.5, 0.5, 2.0].map(|x| pseudo_huber_scalar(1.0, x));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn huber_and_pseudo_huber_tensor_dispatch_preserve_domain_behavior() -> Result<(), String> {
        let huber_values = expect_real_vec(
            huber(
                &SpecialTensor::RealVec(vec![1.0, 0.0, -1.0, f64::NAN]),
                &SpecialTensor::RealScalar(1.0),
                RuntimeMode::Strict,
            )
            .map_err(|err| err.to_string())?,
        )?;
        assert_eq!(huber_values[0], 0.5);
        assert!(huber_values[1].is_nan());
        assert!(huber_values[2].is_nan());
        assert!(huber_values[3].is_nan());

        let pseudo_values = expect_real_vec(
            pseudo_huber(
                &SpecialTensor::RealVec(vec![1.0, 0.0, -1.0, f64::NAN]),
                &SpecialTensor::RealScalar(1.0),
                RuntimeMode::Strict,
            )
            .map_err(|err| err.to_string())?,
        )?;
        assert!(pseudo_values[0] > 0.0);
        assert!(pseudo_values[1].is_nan());
        assert!(pseudo_values[2].is_nan());
        assert!(pseudo_values[3].is_nan());
        Ok(())
    }

    #[test]
    fn elu_basic() {
        // Positive values pass through
        assert!((elu(2.0, 1.0) - 2.0).abs() < 1e-14);

        // Negative values: alpha * (exp(x) - 1)
        // elu(-1, 1) = 1 * (exp(-1) - 1) ≈ -0.632
        assert!((elu(-1.0, 1.0) - ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);

        // Zero
        assert!((elu(0.0, 1.0) - 0.0).abs() < 1e-14);

        // Different alpha
        assert!((elu(-1.0, 2.0) - 2.0 * ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);
    }

    #[test]
    fn leaky_relu_basic() {
        // Positive values pass through
        assert!((leaky_relu(2.0, 0.1) - 2.0).abs() < 1e-14);

        // Negative values scaled by alpha
        assert!((leaky_relu(-2.0, 0.1) - (-0.2)).abs() < 1e-14);

        // Zero
        assert!((leaky_relu(0.0, 0.1) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn gelu_basic() {
        // gelu(0) = 0 (since Φ(0) = 0.5)
        assert!((gelu(0.0) - 0.0).abs() < 1e-14);

        // For large positive x, gelu(x) ≈ x
        assert!((gelu(10.0) - 10.0).abs() < 1e-6);

        // For large negative x, gelu(x) ≈ 0
        assert!(gelu(-10.0).abs() < 1e-6);

        // gelu is smooth and monotonic for x > 0
        assert!(gelu(1.0) > gelu(0.5));
        assert!(gelu(2.0) > gelu(1.0));
    }

    #[test]
    fn selu_basic() {
        // Positive values scaled by ~1.0507
        let scale = 1.0507009873554805;
        assert!((selu(1.0) - scale * 1.0).abs() < 1e-10);

        // Negative values
        assert!(selu(-1.0) < 0.0);

        // Zero
        assert!((selu(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn swish_basic() {
        // swish(0, beta) = 0
        assert!((swish(0.0, 1.0) - 0.0).abs() < 1e-14);

        // For large positive x, swish(x, 1) ≈ x
        assert!((swish(10.0, 1.0) - 10.0).abs() < 1e-3);

        // For large negative x, swish(x, 1) ≈ 0
        assert!(swish(-10.0, 1.0).abs() < 1e-3);

        // swish is smooth
        assert!(swish(1.0, 1.0) > swish(0.0, 1.0));
    }

    #[test]
    fn swish_beta_zero_is_x_over_two() {
        // /testing-golden-artifacts for [frankenscipy-9rl85]:
        // swish(x, 0) = x · σ(0) = x · 0.5 = x/2 for all x.
        for &x in &[-3.0_f64, -0.5, 0.0, 0.5, 3.0, 100.0] {
            let actual = swish(x, 0.0);
            let expected = x / 2.0;
            assert!(
                (actual - expected).abs() < 1e-12,
                "swish({x}, 0) = {actual}, expected x/2 = {expected}"
            );
        }
    }

    #[test]
    fn swish_beta_one_equals_silu() {
        // swish(x, 1) ≡ silu_scalar(x) by definition.
        for &x in &[-3.0_f64, -0.5, 0.0, 0.5, 1.0, 3.0] {
            let s_swish = swish(x, 1.0);
            let s_silu = silu_scalar(x);
            assert!(
                (s_swish - s_silu).abs() < 1e-12,
                "swish({x}, 1) = {s_swish} ≠ silu({x}) = {s_silu}"
            );
        }
    }

    #[test]
    fn swish_joint_sign_flip_negates() {
        // swish(-x, -β) = (-x) · σ(βx) = -x·σ(βx) = -swish(x, β).
        for &x in &[1.0_f64, 2.5, 4.0] {
            for &beta in &[0.5_f64, 1.0, 2.0] {
                let s_pos = swish(x, beta);
                let s_neg = swish(-x, -beta);
                assert!(
                    (s_pos + s_neg).abs() < 1e-12,
                    "swish({x}, {beta}) = {s_pos}; swish(-x, -β) = {s_neg}; sum should be 0"
                );
            }
        }
    }

    #[test]
    fn mish_basic() {
        // mish(0) = 0
        assert!((mish_scalar(0.0) - 0.0).abs() < 1e-14);

        // For large positive x, mish(x) ≈ x
        assert!((mish_scalar(10.0) - 10.0).abs() < 1e-3);

        // For large negative x, mish(x) ≈ 0
        assert!(mish_scalar(-10.0).abs() < 1e-3);

        // mish is smooth and has slight negative region
        assert!(mish_scalar(-0.5) < 0.0);
    }

    #[test]
    fn mish_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = mish(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - mish_scalar(2.0)).abs() < 1e-14);

        let vector = mish(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(mish_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn mish_tensor_dispatch_preserves_negative_lobe_and_nan() -> Result<(), String> {
        let vector = mish(
            &SpecialTensor::RealVec(vec![f64::NAN, -0.5, 10.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1] < 0.0);
        assert!((values[2] - mish_scalar(10.0)).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn hard_sigmoid_basic() {
        // hard_sigmoid(-3) = 0
        assert!((hard_sigmoid_scalar(-3.0) - 0.0).abs() < 1e-14);
        // hard_sigmoid(3) = 1
        assert!((hard_sigmoid_scalar(3.0) - 1.0).abs() < 1e-14);
        // hard_sigmoid(0) = 0.5
        assert!((hard_sigmoid_scalar(0.0) - 0.5).abs() < 1e-14);
        // Linear region
        assert!((hard_sigmoid_scalar(1.0) - (4.0 / 6.0)).abs() < 1e-14);
    }

    #[test]
    fn hard_swish_basic() {
        // hard_swish(0) = 0 * 0.5 = 0
        assert!((hard_swish_scalar(0.0) - 0.0).abs() < 1e-14);
        // hard_swish(-3) = -3 * 0 = 0
        assert!((hard_swish_scalar(-3.0) - 0.0).abs() < 1e-14);
        // hard_swish(3) = 3 * 1 = 3
        assert!((hard_swish_scalar(3.0) - 3.0).abs() < 1e-14);
        // For large positive x, hard_swish(x) ≈ x
        assert!((hard_swish_scalar(10.0) - 10.0).abs() < 1e-14);
    }

    #[test]
    fn hard_tanh_basic() {
        // Clipped to range
        assert!((hard_tanh_scalar(0.5, -1.0, 1.0) - 0.5).abs() < 1e-14);
        assert!((hard_tanh_scalar(2.0, -1.0, 1.0) - 1.0).abs() < 1e-14);
        assert!((hard_tanh_scalar(-2.0, -1.0, 1.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn hard_tanh_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = hard_tanh(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - hard_tanh_scalar(2.0, -1.0, 1.0)).abs() < 1e-14);

        let vector = hard_tanh(
            &SpecialTensor::RealVec(vec![-2.0, 0.5, 2.0]),
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.5, 2.0].map(|x| hard_tanh_scalar(x, -1.0, 1.0));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn hard_tanh_tensor_dispatch_preserves_nan_and_bounds() -> Result<(), String> {
        let vector = hard_tanh(
            &SpecialTensor::RealVec(vec![f64::NAN, -2.0, 0.25, 2.0]),
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] + 1.0).abs() < 1e-14);
        assert!((values[2] - 0.25).abs() < 1e-14);
        assert!((values[3] - 1.0).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn log_cosh_basic() {
        // log_cosh(0) = 0
        assert!((log_cosh_scalar(0.0) - 0.0).abs() < 1e-14);
        // Symmetric
        assert!((log_cosh_scalar(2.0) - log_cosh_scalar(-2.0)).abs() < 1e-14);
        // For large |x|, log_cosh(x) ≈ |x| - ln(2)
        let large = 30.0;
        assert!((log_cosh_scalar(large) - (large - std::f64::consts::LN_2)).abs() < 1e-10);
        // Always non-negative
        assert!(log_cosh_scalar(-5.0) >= 0.0);
    }

    #[test]
    fn softsign_basic() {
        // softsign(0) = 0
        assert!((softsign_scalar(0.0) - 0.0).abs() < 1e-14);
        // Bounded by [-1, 1]
        assert!(softsign_scalar(100.0) < 1.0);
        assert!(softsign_scalar(-100.0) > -1.0);
        // Antisymmetric
        assert!((softsign_scalar(2.0) + softsign_scalar(-2.0)).abs() < 1e-14);
        // softsign(1) = 0.5
        assert!((softsign_scalar(1.0) - 0.5).abs() < 1e-14);
    }

    #[test]
    fn softsign_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = softsign(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - softsign_scalar(2.0)).abs() < 1e-14);

        let vector = softsign(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(softsign_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn softsign_tensor_dispatch_preserves_odd_symmetry_and_nan() -> Result<(), String> {
        let vector = softsign(
            &SpecialTensor::RealVec(vec![f64::NAN, -3.0, 3.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] + values[2]).abs() < 1e-14);
        assert!((values[2] - softsign_scalar(3.0)).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn threshold_basic() {
        // Above threshold: pass through
        assert!((threshold_scalar(5.0, 0.0, -1.0) - 5.0).abs() < 1e-14);
        // Below threshold: use value
        assert!((threshold_scalar(-5.0, 0.0, -1.0) - (-1.0)).abs() < 1e-14);
        // At threshold: use value (not strictly greater)
        assert!((threshold_scalar(0.0, 0.0, -1.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn threshold_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = threshold(
            &SpecialTensor::RealScalar(5.0),
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(-1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - threshold_scalar(5.0, 0.0, -1.0)).abs() < 1e-14);

        let vector = threshold(
            &SpecialTensor::RealVec(vec![-5.0, 0.0, 5.0]),
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(-1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-5.0, 0.0, 5.0].map(|x| threshold_scalar(x, 0.0, -1.0));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn threshold_tensor_dispatch_preserves_nan_and_boundaries() -> Result<(), String> {
        let vector = threshold(
            &SpecialTensor::RealVec(vec![f64::NAN, -1.0, 0.0, 2.0]),
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(-3.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] + 3.0).abs() < 1e-14);
        assert!((values[2] + 3.0).abs() < 1e-14);
        assert!((values[3] - 2.0).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn boxcox_scalar_helpers_cover_lambda_zero_and_roundtrip() {
        assert!((boxcox_transform_scalar(2.0, 0.0) - 2.0_f64.ln()).abs() < 1e-14);
        assert!((boxcox1p_scalar(1.0, 0.0) - 2.0_f64.ln()).abs() < 1e-14);

        let x = 3.5;
        let lam = 0.25;
        let y = boxcox_transform_scalar(x, lam);
        assert!((inv_boxcox_scalar(y, lam) - x).abs() < 1e-12);

        let x1p = 0.75;
        let y1p = boxcox1p_scalar(x1p, lam);
        assert!((inv_boxcox1p_scalar(y1p, lam) - x1p).abs() < 1e-12);
    }

    #[test]
    fn boxcox_alias_matches_scipy_edges() -> Result<(), String> {
        assert!((boxcox_scalar(2.0, 0.0) - 2.0_f64.ln()).abs() < 1e-14);
        assert_eq!(boxcox_scalar(0.0, 0.5), -2.0);
        assert!(boxcox_scalar(0.0, 0.0).is_infinite());
        assert!(boxcox_scalar(0.0, 0.0).is_sign_negative());
        assert!(boxcox_scalar(-1.0, 0.5).is_nan());
        assert_eq!(
            boxcox_scalar(0.5, f64::INFINITY).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert!(boxcox_scalar(2.0, f64::INFINITY).is_nan());
        assert_eq!(boxcox_scalar(2.0, f64::NEG_INFINITY), 0.0);

        let transformed = boxcox(
            &SpecialTensor::RealVec(vec![0.0, 2.0, 4.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(transformed)?;
        let expected = [0.0, 2.0, 4.0].map(|x| boxcox_transform_scalar(x, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn boxcox_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = boxcox_transform(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - boxcox_transform_scalar(2.0, 0.0)).abs() < 1e-14);

        let vector = boxcox1p(
            &SpecialTensor::RealVec(vec![0.0, 1.0, 3.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 1.0, 3.0].map(|x| boxcox1p_scalar(x, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn boxcox_tensor_dispatch_preserves_domain_nan_and_inverse_roundtrip() -> Result<(), String> {
        let transformed = boxcox_transform(
            &SpecialTensor::RealVec(vec![-1.0, 1.0, 4.0]),
            &SpecialTensor::RealScalar(0.25),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let transformed_values = expect_real_vec(transformed)?;
        assert!(transformed_values[0].is_nan());
        assert!((transformed_values[1] - boxcox_transform_scalar(1.0, 0.25)).abs() < 1e-14);

        let recovered = inv_boxcox(
            &SpecialTensor::RealVec(vec![transformed_values[1], transformed_values[2]]),
            &SpecialTensor::RealScalar(0.25),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let recovered_values = expect_real_vec(recovered)?;
        assert!((recovered_values[0] - 1.0).abs() < 1e-12);
        assert!((recovered_values[1] - 4.0).abs() < 1e-12);

        let recovered_1p = inv_boxcox1p(
            &SpecialTensor::RealVec(vec![boxcox1p_scalar(0.0, 0.0), boxcox1p_scalar(2.0, 0.0)]),
            &SpecialTensor::RealScalar(0.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let recovered_1p_values = expect_real_vec(recovered_1p)?;
        assert!(recovered_1p_values[0].abs() < 1e-12);
        assert!((recovered_1p_values[1] - 2.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn silu_basic() {
        // silu is just swish with beta=1
        assert!((silu_scalar(0.0) - swish(0.0, 1.0)).abs() < 1e-14);
        assert!((silu_scalar(2.0) - swish(2.0, 1.0)).abs() < 1e-14);
        assert!((silu_scalar(-2.0) - swish(-2.0, 1.0)).abs() < 1e-14);
    }

    #[test]
    fn silu_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = silu(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - silu_scalar(2.0)).abs() < 1e-14);

        let vector = silu(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(silu_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn silu_tensor_dispatch_preserves_nan_and_swish_reduction() -> Result<(), String> {
        let vector = silu(
            &SpecialTensor::RealVec(vec![f64::NAN, -1.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] - swish(-1.0, 1.0)).abs() < 1e-14);
        assert!((values[2] - swish(1.0, 1.0)).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn log_expit_scalar_basic() {
        // log_expit(0) = log(0.5) = -ln(2)
        assert!((log_expit_scalar(0.0) - (-std::f64::consts::LN_2)).abs() < 1e-14);
        // log_expit should equal log(expit_scalar(x))
        assert!((log_expit_scalar(2.0) - expit_scalar(2.0).ln()).abs() < 1e-14);
        // For large negative x, log_expit(x) ≈ x
        assert!((log_expit_scalar(-50.0) - (-50.0)).abs() < 1e-10);
        // Symmetric property: log_expit(x) + log_expit(-x) = -log(2) - |x|... actually just test consistency
        assert!((log_expit_scalar(5.0) - expit_scalar(5.0).ln()).abs() < 1e-14);
        assert!((log_expit_scalar(-5.0) - expit_scalar(-5.0).ln()).abs() < 1e-14);
    }

    #[test]
    fn log_expit_supports_real_tensor_dispatch() -> Result<(), String> {
        let scalar = log_expit(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        assert!((expect_real_scalar(scalar)? - log_expit_scalar(2.0)).abs() < 1e-14);

        let input = vec![f64::NAN, -50.0, 0.0, 50.0];
        let result = log_expit(&SpecialTensor::RealVec(input.clone()), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values.len(), input.len());
        for (actual, x) in values.into_iter().zip(input) {
            let expected = log_expit_scalar(x);
            if expected.is_nan() {
                assert!(actual.is_nan(), "log_expit({x}) should be NaN");
            } else {
                assert!((actual - expected).abs() < 1e-14, "log_expit({x})");
            }
        }
        Ok(())
    }

    fn expect_real_scalar(tensor: SpecialTensor) -> Result<f64, String> {
        match tensor {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected real scalar, got {other:?}")),
        }
    }

    fn expect_real_vec(tensor: SpecialTensor) -> Result<Vec<f64>, String> {
        match tensor {
            SpecialTensor::RealVec(values) => Ok(values),
            other => Err(format!("expected real vector, got {other:?}")),
        }
    }

    fn expect_complex_scalar(tensor: SpecialTensor) -> Result<Complex64, String> {
        match tensor {
            SpecialTensor::ComplexScalar(value) => Ok(value),
            other => Err(format!("expected complex scalar, got {other:?}")),
        }
    }

    fn expect_complex_vec(tensor: SpecialTensor) -> Result<Vec<Complex64>, String> {
        match tensor {
            SpecialTensor::ComplexVec(values) => Ok(values),
            other => Err(format!("expected complex vector, got {other:?}")),
        }
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64, label: &str) {
        let diff = (actual - expected).abs();
        let scale = expected.abs().max(1.0);
        assert!(
            diff <= tol * scale,
            "{label}: actual={actual:?}, expected={expected:?}, diff={diff}"
        );
    }

    #[test]
    fn exprel_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = exprel(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - exprel_scalar(1.0)).abs() < 1e-14);

        let vector = exprel(
            &SpecialTensor::RealVec(vec![-1.0e-6, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0e-6, 0.0, 1.0].map(exprel_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn exprel_near_zero_series_stays_stable() -> Result<(), String> {
        let result = exprel(&SpecialTensor::RealScalar(1.0e-8), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_real_scalar(result)?;
        let expected = 1.0 + 0.5e-8 + 1.0e-16 / 6.0 + 1.0e-24 / 24.0;
        assert!((value - expected).abs() < 1e-16);
        Ok(())
    }

    #[test]
    fn stirling2_scalar_matches_scipy_contract_points() {
        assert_eq!(stirling2_scalar(0, 0), 1.0);
        assert_eq!(stirling2_scalar(0, 1), 0.0);
        assert_eq!(stirling2_scalar(1, 0), 0.0);
        assert_eq!(stirling2_scalar(5, 2), 15.0);
        assert_eq!(stirling2_scalar(10, 3), 9330.0);
        assert_eq!(stirling2_scalar(-1, 2), 0.0);
        assert_eq!(stirling2_scalar(3, -1), 0.0);
        assert_eq!(stirling2_scalar(3, 4), 0.0);
    }

    #[test]
    fn stirling2_tensor_dispatch_broadcasts_scalar_k() -> Result<(), String> {
        let result = stirling2(
            &SpecialTensor::RealVec(vec![5.0, 6.0, 7.0]),
            &SpecialTensor::RealScalar(2.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values, vec![15.0, 31.0, 63.0]);
        Ok(())
    }

    #[test]
    fn stirling2_tensor_dispatch_rejects_non_integer_inputs() {
        let result = stirling2(
            &SpecialTensor::RealScalar(3.5),
            &SpecialTensor::RealScalar(2.0),
            RuntimeMode::Strict,
        );
        assert!(result.is_err());
    }

    #[test]
    fn ndtr_ndtri_tensor_dispatch_round_trip() -> Result<(), String> {
        let points = vec![-3.0, 0.0, 2.0];
        let probs = ndtr(&SpecialTensor::RealVec(points.clone()), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let prob_values = expect_real_vec(probs)?;
        let reconstructed = ndtri(&SpecialTensor::RealVec(prob_values), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let values = expect_real_vec(reconstructed)?;
        assert_eq!(values.len(), points.len());
        for (actual, expected) in values.iter().zip(points.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn ndtri_tensor_dispatch_preserves_endpoints() -> Result<(), String> {
        let result = ndtri(
            &SpecialTensor::RealVec(vec![0.0, 0.5, 1.0, -1.0, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values[0], f64::NEG_INFINITY);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], f64::INFINITY);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
        Ok(())
    }

    #[test]
    fn ndtri_exp_tensor_dispatch_matches_scipy_contract_points() -> Result<(), String> {
        let result = ndtri_exp(
            &SpecialTensor::RealVec(vec![-800.0, -1.0, -1.0e-20]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        let expected = [
            -39.884_694_838_256_68,
            -0.337_474_963_764_202_44,
            9.262_340_089_798_409,
        ];
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 2.0e-6 * expected.abs().max(1.0),
                "ndtri_exp contract point = {actual}, scipy {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn ndtri_exp_scalar_preserves_log_probability_boundaries() {
        assert_eq!(ndtri_exp_scalar(0.0), f64::INFINITY);
        assert_eq!(ndtri_exp_scalar(-0.0), f64::INFINITY);
        assert_eq!(ndtri_exp_scalar(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(ndtri_exp_scalar(1.0).is_nan());
        assert!(ndtri_exp_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn erfcx_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = erfcx(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - erfcx_scalar(1.0)).abs() < 1e-14);

        let vector = erfcx(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(erfcx_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn erfcx_large_x_uses_stable_asymptotic_path() -> Result<(), String> {
        let result = erfcx(&SpecialTensor::RealScalar(30.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_real_scalar(result)?;
        let expected = erfcx_scalar(30.0);
        assert!(value.is_finite());
        assert!((value - expected).abs() < 1e-16);
        Ok(())
    }

    #[test]
    fn dawsn_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = dawsn(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - dawsn_scalar(1.0)).abs() < 1e-14);

        let vector = dawsn(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(dawsn_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn dawsn_tensor_dispatch_preserves_odd_symmetry() -> Result<(), String> {
        let positive = dawsn(&SpecialTensor::RealScalar(2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let negative = dawsn(&SpecialTensor::RealScalar(-2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive)?;
        let negative_value = expect_real_scalar(negative)?;
        assert!((positive_value + negative_value).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn erfi_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = erfi(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - erfi_scalar(1.0)).abs() < 1e-14);

        let vector = erfi(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(erfi_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn erfi_tensor_dispatch_preserves_odd_symmetry() -> Result<(), String> {
        let positive = erfi(&SpecialTensor::RealScalar(2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let negative = erfi(&SpecialTensor::RealScalar(-2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive)?;
        let negative_value = expect_real_scalar(negative)?;
        assert!((positive_value + negative_value).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn owens_t_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = owens_t(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - owens_t_scalar(0.5, 1.0)).abs() < 1e-14);

        let vector = owens_t(
            &SpecialTensor::RealVec(vec![0.0, 0.5, 1.0]),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 0.5, 1.0].map(|h| owens_t_scalar(h, 1.0));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }

        let broadcast = owens_t(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealVec(vec![0.0, 0.5, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let broadcast_values = expect_real_vec(broadcast)?;
        let broadcast_expected = [0.0, 0.5, 1.0].map(|a| owens_t_scalar(0.5, a));
        assert_eq!(broadcast_values.len(), broadcast_expected.len());
        for (actual, expected) in broadcast_values.iter().zip(broadcast_expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn owens_t_tensor_dispatch_preserves_even_h_and_odd_a_symmetry() -> Result<(), String> {
        let positive_h = owens_t(
            &SpecialTensor::RealScalar(1.25),
            &SpecialTensor::RealScalar(0.75),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let negative_h = owens_t(
            &SpecialTensor::RealScalar(-1.25),
            &SpecialTensor::RealScalar(0.75),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let negative_a = owens_t(
            &SpecialTensor::RealScalar(1.25),
            &SpecialTensor::RealScalar(-0.75),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive_h)?;
        let negative_h_value = expect_real_scalar(negative_h)?;
        let negative_a_value = expect_real_scalar(negative_a)?;
        assert!((positive_value - negative_h_value).abs() < 1e-14);
        assert!((positive_value + negative_a_value).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn owens_t_tensor_dispatch_handles_zero_and_nan_inputs() -> Result<(), String> {
        let result = owens_t(
            &SpecialTensor::RealVec(vec![0.0, 0.0, f64::NAN]),
            &SpecialTensor::RealVec(vec![0.0, 1.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values[0], 0.0);
        assert!((values[1] - 0.125).abs() < 1e-14);
        assert!(values[2].is_nan());
        Ok(())
    }

    #[test]
    fn owens_t_closed_form_reference_values() {
        // /testing-golden-artifacts for [frankenscipy-puui1]: pin
        // Owen's T at boundary points where it has analytic closed
        // forms, in addition to the dispatch tests already in place.
        use std::f64::consts::PI;

        // T(0, a) = arctan(a) / (2π).
        for &a in &[0.5_f64, 1.0, 2.0, 5.0] {
            let expected = a.atan() / (2.0 * PI);
            assert!(
                (owens_t_scalar(0.0, a) - expected).abs() < 1e-12,
                "T(0, {a}) = {} != arctan({a})/(2π) = {expected}",
                owens_t_scalar(0.0, a)
            );
        }

        // T(h, 0) = 0 for any h.
        for &h in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            assert_eq!(owens_t_scalar(h, 0.0), 0.0, "T({h}, 0) must be 0");
        }

        // T(h, 1) = (1/2)·Φ(h)·[1 − Φ(h)] = (1/2)·Φ(h)·Φ(-h).
        for &h in &[-1.5_f64, -0.5, 0.5, 1.5, 2.0] {
            let phi_h = ndtr_scalar(h);
            let expected = 0.5 * phi_h * (1.0 - phi_h);
            assert!(
                (owens_t_scalar(h, 1.0) - expected).abs() < 1e-9,
                "T({h}, 1) = {} != (1/2)·Φ(h)·(1-Φ(h)) = {expected}",
                owens_t_scalar(h, 1.0)
            );
        }
    }

    #[test]
    fn owens_t_symmetry_and_antisymmetry_identities() {
        // /testing-metamorphic anchor for owens_t (paired with the
        // closed-form pins): T(-h, a) = T(h, a) (even in h) and
        // T(h, -a) = -T(h, a) (odd in a).
        for &h in &[-2.5_f64, -1.0, 0.5, 2.0] {
            for &a in &[-3.0_f64, -0.5, 0.5, 3.0] {
                let baseline = owens_t_scalar(h, a);
                let flipped_h = owens_t_scalar(-h, a);
                let flipped_a = owens_t_scalar(h, -a);
                assert!(
                    (baseline - flipped_h).abs() < 1e-12,
                    "T(-h, a) symmetry broken at h={h}, a={a}: \
                     baseline={baseline}, flipped_h={flipped_h}"
                );
                assert!(
                    (baseline + flipped_a).abs() < 1e-12,
                    "T(h, -a) antisymmetry broken at h={h}, a={a}: \
                     baseline={baseline}, flipped_a={flipped_a}"
                );
            }
        }
    }

    #[test]
    fn xlogx_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = xlogx(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - xlogx_scalar(2.0)).abs() < 1e-14);

        let vector = xlogx(
            &SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 1.0, 2.0].map(xlogx_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn xlogx_tensor_dispatch_preserves_domain_behavior() -> Result<(), String> {
        let vector = xlogx(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert!(values[2].is_nan());
        Ok(())
    }

    #[test]
    fn hard_sigmoid_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = hard_sigmoid(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - hard_sigmoid_scalar(0.0)).abs() < 1e-14);

        let vector = hard_sigmoid(
            &SpecialTensor::RealVec(vec![-4.0, 0.0, 4.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-4.0, 0.0, 4.0].map(hard_sigmoid_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn hard_sigmoid_tensor_dispatch_preserves_nan_and_bounds() -> Result<(), String> {
        let vector = hard_sigmoid(
            &SpecialTensor::RealVec(vec![f64::NAN, -10.0, 10.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 1.0);
        Ok(())
    }

    #[test]
    fn hard_swish_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = hard_swish(&SpecialTensor::RealScalar(3.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - hard_swish_scalar(3.0)).abs() < 1e-14);

        let vector = hard_swish(
            &SpecialTensor::RealVec(vec![-4.0, 0.0, 4.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-4.0, 0.0, 4.0].map(hard_swish_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn hard_swish_tensor_dispatch_preserves_nan_and_clipping_behavior() -> Result<(), String> {
        let vector = hard_swish(
            &SpecialTensor::RealVec(vec![f64::NAN, -10.0, -3.0, 10.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert!((values[3] - 10.0).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn log_cosh_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = log_cosh(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - log_cosh_scalar(0.5)).abs() < 1e-14);

        let vector = log_cosh(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(log_cosh_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn log_cosh_tensor_dispatch_preserves_large_tail_and_nan() -> Result<(), String> {
        let vector = log_cosh(
            &SpecialTensor::RealVec(vec![f64::NAN, -30.0, 30.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] - values[2]).abs() < 1e-14);
        let expected_tail = 30.0 - std::f64::consts::LN_2;
        assert!((values[2] - expected_tail).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn tanhshrink_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = tanhshrink(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - tanhshrink_scalar(0.5)).abs() < 1e-14);

        let vector = tanhshrink(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(tanhshrink_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn tanhshrink_tensor_dispatch_preserves_odd_symmetry() -> Result<(), String> {
        let positive = tanhshrink(&SpecialTensor::RealScalar(2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let negative = tanhshrink(&SpecialTensor::RealScalar(-2.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let positive_value = expect_real_scalar(positive)?;
        let negative_value = expect_real_scalar(negative)?;
        assert!((positive_value + negative_value).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn hardshrink_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = hardshrink(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - hardshrink_scalar(2.0, 0.5)).abs() < 1e-14);

        let vector = hardshrink(
            &SpecialTensor::RealVec(vec![-2.0, -0.5, 0.3, 2.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, -0.5, 0.3, 2.0].map(|x| hardshrink_scalar(x, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }

        let broadcast = hardshrink(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let broadcast_values = expect_real_vec(broadcast)?;
        let broadcast_expected = [0.0, 1.0, 2.0].map(|lambda| hardshrink_scalar(1.0, lambda));
        assert_eq!(broadcast_values.len(), broadcast_expected.len());
        for (actual, expected) in broadcast_values.iter().zip(broadcast_expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn hardshrink_tensor_dispatch_preserves_nan_and_boundary_behavior() -> Result<(), String> {
        let result = hardshrink(
            &SpecialTensor::RealVec(vec![f64::NAN, 0.5, -0.5, 0.6]),
            &SpecialTensor::RealVec(vec![0.1, 0.5, 0.5, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert!(values[3].is_nan());
        Ok(())
    }

    #[test]
    fn softshrink_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = softshrink(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - softshrink_scalar(2.0, 0.5)).abs() < 1e-14);

        let vector = softshrink(
            &SpecialTensor::RealVec(vec![-2.0, -0.5, 0.3, 2.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, -0.5, 0.3, 2.0].map(|x| softshrink_scalar(x, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }

        let broadcast = softshrink(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealVec(vec![0.0, 0.5, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let broadcast_values = expect_real_vec(broadcast)?;
        let broadcast_expected = [0.0, 0.5, 2.0].map(|lambda| softshrink_scalar(2.0, lambda));
        assert_eq!(broadcast_values.len(), broadcast_expected.len());
        for (actual, expected) in broadcast_values.iter().zip(broadcast_expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn softshrink_tensor_dispatch_preserves_nan_and_threshold_behavior() -> Result<(), String> {
        let result = softshrink(
            &SpecialTensor::RealVec(vec![f64::NAN, 0.5, -0.5, 2.0]),
            &SpecialTensor::RealVec(vec![0.1, 0.5, 0.5, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 0.0);
        assert!(values[3].is_nan());
        Ok(())
    }

    #[test]
    fn cloglog_basic() {
        // cloglog(0.5) = log(-log(0.5)) = log(ln(2)) ≈ -0.3665
        let expected = std::f64::consts::LN_2.ln();
        assert!((cloglog(0.5) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(cloglog(0.3) < cloglog(0.5));
        assert!(cloglog(0.5) < cloglog(0.7));

        // Boundary behavior
        assert!(cloglog(0.0).is_infinite() && cloglog(0.0) < 0.0);
        assert!(cloglog(1.0).is_infinite() && cloglog(1.0) > 0.0);

        // Near boundaries should still work
        assert!(cloglog(1e-10).is_finite());
        assert!(cloglog(1.0 - 1e-10).is_finite());
    }

    #[test]
    fn cloglog_inv_basic() {
        // cloglog_inv is the Gumbel CDF: 1 - exp(-exp(x))
        // cloglog_inv(0) = 1 - exp(-1) ≈ 0.6321
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((cloglog_inv(0.0) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(cloglog_inv(-2.0) < cloglog_inv(0.0));
        assert!(cloglog_inv(0.0) < cloglog_inv(2.0));

        // Bounded by (0, 1)
        assert!(cloglog_inv(-100.0) >= 0.0);
        assert!(cloglog_inv(100.0) <= 1.0);
    }

    #[test]
    fn cloglog_inverse_relationship() {
        // cloglog and cloglog_inv are inverses
        let p = 0.3;
        assert!((cloglog_inv(cloglog(p)) - p).abs() < 1e-14);

        let x = 1.5;
        assert!((cloglog(cloglog_inv(x)) - x).abs() < 1e-14);

        // Test more values
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((cloglog_inv(cloglog(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn loglog_basic() {
        // loglog(p) = -log(-log(p))
        // loglog(exp(-1)) = -log(-log(exp(-1))) = -log(1) = 0
        let p_at_zero = (-1.0_f64).exp();
        assert!((loglog(p_at_zero) - 0.0).abs() < 1e-14);

        // Monotonically increasing (like cloglog)
        assert!(loglog(0.3) < loglog(0.5));
        assert!(loglog(0.5) < loglog(0.7));

        // Boundary behavior (opposite direction of cloglog)
        assert!(loglog(0.0).is_infinite() && loglog(0.0) < 0.0);
        assert!(loglog(1.0).is_infinite() && loglog(1.0) > 0.0);
    }

    #[test]
    fn loglog_inv_basic() {
        // loglog_inv(x) = exp(-exp(-x)) - the Gumbel (minimum) CDF
        // loglog_inv(0) = exp(-exp(0)) = exp(-1) ≈ 0.3679
        let expected = (-1.0_f64).exp();
        assert!((loglog_inv(0.0) - expected).abs() < 1e-14);

        // Monotonically increasing
        assert!(loglog_inv(-2.0) < loglog_inv(0.0));
        assert!(loglog_inv(0.0) < loglog_inv(2.0));

        // Bounded by (0, 1)
        assert!(loglog_inv(-100.0) >= 0.0);
        assert!(loglog_inv(100.0) <= 1.0);
    }

    #[test]
    fn loglog_inverse_relationship() {
        // loglog and loglog_inv are inverses
        let p = 0.3;
        assert!((loglog_inv(loglog(p)) - p).abs() < 1e-14);

        let x = -1.5;
        assert!((loglog(loglog_inv(x)) - x).abs() < 1e-14);

        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((loglog_inv(loglog(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn cauchit_basic() {
        // cauchit(0.5) = tan(0) = 0
        assert!((cauchit_scalar(0.5) - 0.0).abs() < 1e-14);

        // cauchit(0.75) = tan(π/4) = 1
        assert!((cauchit_scalar(0.75) - 1.0).abs() < 1e-14);

        // cauchit(0.25) = tan(-π/4) = -1
        assert!((cauchit_scalar(0.25) - (-1.0)).abs() < 1e-14);

        // Monotonically increasing
        assert!(cauchit_scalar(0.3) < cauchit_scalar(0.5));
        assert!(cauchit_scalar(0.5) < cauchit_scalar(0.7));

        // Boundary behavior
        assert!(cauchit_scalar(0.0).is_infinite() && cauchit_scalar(0.0) < 0.0);
        assert!(cauchit_scalar(1.0).is_infinite() && cauchit_scalar(1.0) > 0.0);
    }

    #[test]
    fn cauchit_inv_basic() {
        // cauchit_inv(0) = 0.5
        assert!((cauchit_inv_scalar(0.0) - 0.5).abs() < 1e-14);

        // cauchit_inv(1) = 0.75
        assert!((cauchit_inv_scalar(1.0) - 0.75).abs() < 1e-14);

        // cauchit_inv(-1) = 0.25
        assert!((cauchit_inv_scalar(-1.0) - 0.25).abs() < 1e-14);

        // Bounded by (0, 1) for finite x
        assert!(cauchit_inv_scalar(-1000.0) > 0.0);
        assert!(cauchit_inv_scalar(1000.0) < 1.0);
    }

    #[test]
    fn cauchit_inverse_relationship() {
        // cauchit and cauchit_inv are inverses
        let p = 0.3;
        assert!((cauchit_inv_scalar(cauchit_scalar(p)) - p).abs() < 1e-14);

        let x = 2.5;
        assert!((cauchit_scalar(cauchit_inv_scalar(x)) - x).abs() < 1e-14);

        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert!((cauchit_inv_scalar(cauchit_scalar(p)) - p).abs() < 1e-13);
        }
    }

    #[test]
    fn cauchit_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = cauchit(&SpecialTensor::RealScalar(0.75), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - cauchit_scalar(0.75)).abs() < 1e-14);

        let vector = cauchit(
            &SpecialTensor::RealVec(vec![0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.25, 0.5, 0.75].map(cauchit_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(values[0] < values[1] && values[1] < values[2]);
        Ok(())
    }

    #[test]
    fn cauchit_inv_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = cauchit_inv(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - cauchit_inv_scalar(1.0)).abs() < 1e-14);

        let vector = cauchit_inv(
            &SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-1.0, 0.0, 1.0].map(cauchit_inv_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(values[0] < values[1] && values[1] < values[2]);
        Ok(())
    }

    #[test]
    fn cauchit_tensor_roundtrip_preserves_values() -> Result<(), String> {
        let probabilities = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let linked = cauchit(
            &SpecialTensor::RealVec(probabilities.clone()),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let linked_values = expect_real_vec(linked)?;
        let recovered = cauchit_inv(&SpecialTensor::RealVec(linked_values), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let recovered_values = expect_real_vec(recovered)?;
        assert_eq!(recovered_values.len(), probabilities.len());
        for (actual, expected) in recovered_values.iter().zip(probabilities.iter()) {
            assert!((actual - expected).abs() < 1e-13);
        }
        Ok(())
    }

    #[test]
    fn spence_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = spence(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - spence_scalar(0.5)).abs() < 1e-12);

        let vector = spence(
            &SpecialTensor::RealVec(vec![0.0, 0.5, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 0.5, 1.0].map(spence_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn spence_matches_scipy_reference_points() {
        let cases: &[(f64, f64)] = &[
            (0.0, 1.644_934_066_848_226_4),
            (1.0e-12, 1.644_934_066_819_596),
            (1.0e-6, 1.644_919_251_330_510_4),
            (0.01, 1.588_625_448_076_375_3),
            (0.1, 1.299_714_723_004_958_8),
            (0.5, 0.582_240_526_465_013_5),
            (0.99, 0.010_025_111_740_139_103),
            (1.0, 0.0),
            (1.01, -0.009_975_110_490_083_546),
            (1.5, -0.448_414_206_923_646_2),
            (2.0, -0.822_467_033_424_114_2),
            (4.0, -1.939_375_420_766_708_7),
            (10.0, -3.950_663_778_244_157),
            (1000.0, -25.495_564_102_428_908),
        ];

        for &(x, expected) in cases {
            let got = spence_scalar(x);
            let tol = if x <= 1.0e-9 { 3.0e-11 } else { 3.0e-13 };
            assert!(
                (got - expected).abs() < tol,
                "spence({x}) = {got}, expected {expected}"
            );
        }
        assert!(spence_scalar(-1.0).is_nan());
        assert!(spence_scalar(f64::INFINITY).is_nan());
    }

    #[test]
    fn spence_tensor_dispatch_handles_reflection_branch() -> Result<(), String> {
        let input = vec![2.5, 3.0];
        let result = spence(&SpecialTensor::RealVec(input.clone()), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values.len(), input.len());
        for (actual, x) in values.iter().zip(input.iter()) {
            let expected = spence_scalar(*x);
            assert!((actual - expected).abs() < 1e-12);
            let reflected = -spence_scalar(1.0 / x) - x.ln().powi(2) / 2.0;
            assert!((actual - reflected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn hardshrink_basic() {
        // hardshrink(x, λ) = x if |x| > λ, else 0
        assert!((hardshrink_scalar(2.0, 0.5) - 2.0).abs() < 1e-14);
        assert!((hardshrink_scalar(-2.0, 0.5) - (-2.0)).abs() < 1e-14);
        assert!((hardshrink_scalar(0.3, 0.5) - 0.0).abs() < 1e-14);
        assert!((hardshrink_scalar(-0.3, 0.5) - 0.0).abs() < 1e-14);
        // At threshold
        assert!((hardshrink_scalar(0.5, 0.5) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn softshrink_basic() {
        // softshrink shrinks toward zero by λ
        assert!((softshrink_scalar(2.0, 0.5) - 1.5).abs() < 1e-14);
        assert!((softshrink_scalar(-2.0, 0.5) - (-1.5)).abs() < 1e-14);
        assert!((softshrink_scalar(0.3, 0.5) - 0.0).abs() < 1e-14);
        assert!((softshrink_scalar(-0.3, 0.5) - 0.0).abs() < 1e-14);
        // At threshold
        assert!((softshrink_scalar(0.5, 0.5) - 0.0).abs() < 1e-14);
        assert!((softshrink_scalar(-0.5, 0.5) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn tanhshrink_basic() {
        // tanhshrink(x) = x - tanh(x)
        assert!((tanhshrink_scalar(0.0) - 0.0).abs() < 1e-14);
        // For small x, tanhshrink ≈ x^3/3
        assert!((tanhshrink_scalar(0.1) - (0.1 - 0.1_f64.tanh())).abs() < 1e-14);
        // For large |x|, tanhshrink ≈ x - sign(x)
        assert!((tanhshrink_scalar(10.0) - 9.0).abs() < 1e-5);
        assert!((tanhshrink_scalar(-10.0) - (-9.0)).abs() < 1e-5);
    }

    #[test]
    fn celu_basic() {
        // celu(x, α) is like elu but C¹ continuous
        // For x >= 0, celu(x) = x
        assert!((celu(2.0, 1.0) - 2.0).abs() < 1e-14);
        assert!((celu(0.0, 1.0) - 0.0).abs() < 1e-14);
        // For x < 0, celu(x, α) = α * (exp(x/α) - 1)
        let expected = 1.0 * ((-1.0_f64 / 1.0).exp() - 1.0);
        assert!((celu(-1.0, 1.0) - expected).abs() < 1e-14);
        // Different alpha
        let expected2 = 2.0 * ((-1.0_f64 / 2.0).exp() - 1.0);
        assert!((celu(-1.0, 2.0) - expected2).abs() < 1e-14);
    }

    #[test]
    fn logsigmoid_basic() {
        // logsigmoid is same as log_expit_scalar
        assert!((logsigmoid_scalar(0.0) - log_expit_scalar(0.0)).abs() < 1e-14);
        assert!((logsigmoid_scalar(2.0) - log_expit_scalar(2.0)).abs() < 1e-14);
        assert!((logsigmoid_scalar(-2.0) - log_expit_scalar(-2.0)).abs() < 1e-14);
        // logsigmoid(0) = log(0.5) = -ln(2)
        assert!((logsigmoid_scalar(0.0) - (-std::f64::consts::LN_2)).abs() < 1e-14);
    }

    #[test]
    fn logsigmoid_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = logsigmoid(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - logsigmoid_scalar(2.0)).abs() < 1e-14);

        let vector = logsigmoid(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(logsigmoid_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn logsigmoid_tensor_dispatch_preserves_extreme_tails_and_nan() -> Result<(), String> {
        let vector = logsigmoid(
            &SpecialTensor::RealVec(vec![f64::NAN, -100.0, 100.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!((values[1] - (-100.0)).abs() < 1e-10);
        assert!(values[2].abs() < 1e-40);
        Ok(())
    }

    #[test]
    fn log1mexp_basic() {
        // log1mexp(x) = log(1 - exp(x)) for x < 0
        // log1mexp(-ln(2)) = log(1 - 0.5) = log(0.5) = -ln(2)
        assert!(
            (log1mexp_scalar(-std::f64::consts::LN_2) - (-std::f64::consts::LN_2)).abs() < 1e-14
        );

        // For x -> -inf, exp(x) -> 0, so log1mexp(x) -> log(1) = 0
        assert!((log1mexp_scalar(-100.0) - 0.0).abs() < 1e-40);

        // log1mexp(0) = log(0) = -inf
        assert!(log1mexp_scalar(0.0).is_infinite() && log1mexp_scalar(0.0) < 0.0);

        // log1mexp(x) is NaN for x > 0
        assert!(log1mexp_scalar(1.0).is_nan());
    }

    #[test]
    fn log1pexp_basic() {
        // log1pexp is same as softplus
        assert!((log1pexp_scalar(0.0) - softplus_scalar(0.0)).abs() < 1e-14);
        assert!((log1pexp_scalar(2.0) - softplus_scalar(2.0)).abs() < 1e-14);
        assert!((log1pexp_scalar(-2.0) - softplus_scalar(-2.0)).abs() < 1e-14);
    }

    #[test]
    fn log1pexp_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = log1pexp(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - log1pexp_scalar(2.0)).abs() < 1e-14);

        let vector = log1pexp(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(log1pexp_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn log1pexp_tensor_dispatch_preserves_tail_stability() -> Result<(), String> {
        let vector = log1pexp(
            &SpecialTensor::RealVec(vec![f64::NAN, -100.0, 100.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1] < 1e-40);
        assert!((values[2] - 100.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn log1mexp_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = log1mexp(&SpecialTensor::RealScalar(-1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - log1mexp_scalar(-1.0)).abs() < 1e-14);

        let vector = log1mexp(
            &SpecialTensor::RealVec(vec![-2.0, -std::f64::consts::LN_2, -1.0e-6]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, -std::f64::consts::LN_2, -1.0e-6].map(log1mexp_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn log1mexp_tensor_dispatch_preserves_domain_edges() -> Result<(), String> {
        let vector = log1mexp(
            &SpecialTensor::RealVec(vec![f64::NAN, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert!(values[2].is_nan());
        Ok(())
    }

    #[test]
    fn xlogx_basic() {
        // xlogx(0) = 0 by convention
        assert!((xlogx_scalar(0.0) - 0.0).abs() < 1e-14);

        // xlogx(1) = 1 * log(1) = 0
        assert!((xlogx_scalar(1.0) - 0.0).abs() < 1e-14);

        // xlogx(e) = e * log(e) = e
        assert!((xlogx_scalar(std::f64::consts::E) - std::f64::consts::E).abs() < 1e-14);

        // xlogx(x) for x > 0
        assert!((xlogx_scalar(2.0) - 2.0 * 2.0_f64.ln()).abs() < 1e-14);

        // xlogx(x) is NaN for x < 0
        assert!(xlogx_scalar(-1.0).is_nan());
    }

    #[test]
    fn negentropy_basic() {
        // negentropy is same as xlogx
        assert!((negentropy_scalar(0.0) - xlogx_scalar(0.0)).abs() < 1e-14);
        assert!((negentropy_scalar(0.5) - xlogx_scalar(0.5)).abs() < 1e-14);
        assert!((negentropy_scalar(2.0) - xlogx_scalar(2.0)).abs() < 1e-14);
    }

    #[test]
    fn negentropy_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = negentropy(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - negentropy_scalar(0.5)).abs() < 1e-14);

        let vector = negentropy(
            &SpecialTensor::RealVec(vec![0.0, 0.5, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 0.5, 2.0].map(negentropy_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn negentropy_tensor_dispatch_preserves_domain_behavior() -> Result<(), String> {
        let vector = negentropy(
            &SpecialTensor::RealVec(vec![f64::NAN, -1.0, 0.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 0.0);
        Ok(())
    }

    #[test]
    fn binary_cross_entropy_basic() {
        // BCE(1, 1) = 0 (perfect prediction)
        assert!((binary_cross_entropy_scalar(1.0, 1.0) - 0.0).abs() < 1e-14);

        // BCE(0, 0) = 0 (perfect prediction)
        assert!((binary_cross_entropy_scalar(0.0, 0.0) - 0.0).abs() < 1e-14);

        // BCE(1, 0.5) = -log(0.5) = ln(2)
        assert!((binary_cross_entropy_scalar(1.0, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);

        // BCE(0, 0.5) = -log(0.5) = ln(2)
        assert!((binary_cross_entropy_scalar(0.0, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);

        // BCE(0.5, 0.5) = -0.5*log(0.5) - 0.5*log(0.5) = ln(2)
        assert!((binary_cross_entropy_scalar(0.5, 0.5) - std::f64::consts::LN_2).abs() < 1e-14);
    }

    #[test]
    fn binary_cross_entropy_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = binary_cross_entropy(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - binary_cross_entropy_scalar(1.0, 0.5)).abs() < 1e-14);

        let vector = binary_cross_entropy(
            &SpecialTensor::RealVec(vec![0.0, 0.5, 1.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [0.0, 0.5, 1.0].map(|p| binary_cross_entropy_scalar(p, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn binary_cross_entropy_tensor_dispatch_preserves_domain_behavior() -> Result<(), String> {
        let vector = binary_cross_entropy(
            &SpecialTensor::RealVec(vec![f64::NAN, 0.5, 1.0]),
            &SpecialTensor::RealVec(vec![0.5, -0.5, 0.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_nan());
        assert!(values[1].is_infinite());
        assert!(values[2].is_infinite());
        Ok(())
    }

    #[test]
    fn gammaincinv_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = gammaincinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - gammaincinv_scalar(2.0, 0.5)).abs() < 1e-12);

        let vector = gammaincinv(
            &SpecialTensor::RealVec(vec![1.0, 2.0, 3.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [1.0, 2.0, 3.0].map(|a| gammaincinv_scalar(a, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn gammaincinv_tensor_dispatch_preserves_boundary_behavior() -> Result<(), String> {
        let vector = gammaincinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealVec(vec![0.0, 1.0, -0.5, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert_eq!(values[0], 0.0);
        assert!(values[1].is_infinite());
        assert!(values[2].is_nan());
        assert!(values[3].is_nan());
        Ok(())
    }

    #[test]
    fn gammainccinv_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = gammainccinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - gammainccinv_scalar(2.0, 0.5)).abs() < 1e-12);

        let vector = gammainccinv(
            &SpecialTensor::RealVec(vec![1.0, 2.0, 3.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [1.0, 2.0, 3.0].map(|a| gammainccinv_scalar(a, 0.5));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn gammainccinv_tensor_dispatch_preserves_boundary_behavior() -> Result<(), String> {
        let vector = gammainccinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealVec(vec![1.0, 0.0, -0.5, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert_eq!(values[0], 0.0);
        assert!(values[1].is_infinite());
        assert!(values[2].is_nan());
        assert!(values[3].is_nan());
        Ok(())
    }

    #[test]
    fn gammainccinv_scalar_inverts_complement() {
        for &(a, q) in &[(0.5_f64, 0.25_f64), (2.0, 0.5), (5.0, 0.9), (10.0, 0.1)] {
            let x = gammainccinv_scalar(a, q);
            let actual = gammaincc_conv(a, x);
            assert!(
                (actual - q).abs() < 1e-10,
                "gammainccinv_scalar({a}, {q}) = {x}, Q(a,x) = {actual}"
            );
        }
    }

    #[test]
    fn betaincinv_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = betaincinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(3.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - betaincinv_scalar(2.0, 3.0, 0.5)).abs() < 1e-12);

        let vector = betaincinv(
            &SpecialTensor::RealVec(vec![2.0, 4.0, 6.0]),
            &SpecialTensor::RealScalar(3.0),
            &SpecialTensor::RealVec(vec![0.25, 0.5, 0.75]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected =
            [(2.0, 0.25), (4.0, 0.5), (6.0, 0.75)].map(|(a, y)| betaincinv_scalar(a, 3.0, y));
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn betaincinv_tensor_dispatch_preserves_boundary_and_broadcast_behavior() -> Result<(), String>
    {
        let vector = betaincinv(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(3.0),
            &SpecialTensor::RealVec(vec![0.0, 1.0, -0.5, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 1.0);
        assert!(values[2].is_nan());
        assert!(values[3].is_nan());

        let mismatch = betaincinv(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![3.0, 4.0, 5.0]),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        );
        assert!(mismatch.is_err());
        Ok(())
    }

    #[test]
    fn log_ndtr_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = log_ndtr(&SpecialTensor::RealScalar(-1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - log_ndtr_scalar(-1.0)).abs() < 1e-14);

        let vector = log_ndtr(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 2.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 2.0].map(log_ndtr_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn log_ndtr_tensor_dispatch_preserves_extreme_negative_tails() -> Result<(), String> {
        let vector = log_ndtr(
            &SpecialTensor::RealVec(vec![-50.0, -25.0, f64::NAN]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[0] < values[1]);
        assert!(values[2].is_nan());
        Ok(())
    }

    #[test]
    fn wrightomega_tensor_dispatch_matches_scalar_path() -> Result<(), String> {
        let scalar = wrightomega(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!((scalar_value - wrightomega_scalar(1.0)).abs() < 1e-14);

        let vector = wrightomega(
            &SpecialTensor::RealVec(vec![-2.0, 0.0, 1.0]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_real_vec(vector)?;
        let expected = [-2.0, 0.0, 1.0].map(wrightomega_scalar);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        Ok(())
    }

    #[test]
    fn wrightomega_negative_tail_matches_exponential_limit() {
        let expected_twenty = (-20.0f64).exp();
        let actual_twenty = wrightomega_scalar(-20.0);
        assert!(((actual_twenty / expected_twenty) - 1.0).abs() < 1.0e-6);

        let expected_hundred = (-100.0f64).exp();
        let actual_hundred = wrightomega_scalar(-100.0);
        assert!(((actual_hundred / expected_hundred) - 1.0).abs() < 1.0e-12);

        assert_eq!(wrightomega_scalar(-800.0), 0.0);
    }

    #[test]
    fn wrightomega_complex_real_axis_reduces_to_scalar_path() -> Result<(), String> {
        for x in [-100.0, -20.0, -2.0, 0.0, 1.0, 4.0] {
            let real_result = wrightomega(&SpecialTensor::RealScalar(x), RuntimeMode::Strict)
                .map_err(|err| err.to_string())?;
            let real_value = expect_real_scalar(real_result)?;
            let complex_result = wrightomega(
                &SpecialTensor::ComplexScalar(Complex64::from_real(x)),
                RuntimeMode::Strict,
            )
            .map_err(|err| err.to_string())?;
            let complex_value = expect_complex_scalar(complex_result)?;
            assert_complex_close(
                complex_value,
                Complex64::from_real(real_value),
                1e-12,
                "wrightomega real-axis reduction",
            );
        }
        Ok(())
    }

    #[test]
    fn wrightomega_complex_identity_principal_branch() -> Result<(), String> {
        let z = Complex64::new(0.5, 0.75);
        let result = wrightomega(&SpecialTensor::ComplexScalar(z), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_complex_scalar(result)?;
        assert_complex_close(
            value + value.ln(),
            z,
            1e-10,
            "wrightomega principal-branch identity",
        );
        Ok(())
    }

    #[test]
    fn wrightomega_complex_large_real_part_stays_finite() -> Result<(), String> {
        let z = Complex64::new(1000.0, 1.0);
        let result = wrightomega(&SpecialTensor::ComplexScalar(z), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_complex_scalar(result)?;
        let expected = Complex64::new(993.099_168_966_944_7, 0.998_994_064_481_437_6);
        assert!(value.re.is_finite());
        assert!(value.im.is_finite());
        assert_complex_close(
            value,
            expected,
            1e-10,
            "wrightomega large-real complex asymptotic",
        );
        assert_complex_close(
            value + value.ln(),
            z,
            1e-10,
            "wrightomega large-real complex identity",
        );
        Ok(())
    }

    #[test]
    fn wrightomega_complex_does_not_alias_two_pi_periods() -> Result<(), String> {
        let z = Complex64::new(0.0, 2.0 * std::f64::consts::PI);
        let result = wrightomega(&SpecialTensor::ComplexScalar(z), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let value = expect_complex_scalar(result)?;
        let expected = Complex64::new(-1.533_913_319_793_574_4, 4.375_185_153_061_897_5);
        assert_complex_close(
            value,
            expected,
            1e-10,
            "wrightomega outside principal strip should not collapse to omega(0)",
        );
        assert!(
            (value - Complex64::from_real(wrightomega_scalar(0.0))).abs() > 1.0,
            "wrightomega(z) must not reduce modulo 2πi to omega(0)",
        );
        assert_complex_close(value + value.ln(), z, 1e-10, "wrightomega 2πi identity");
        Ok(())
    }

    #[test]
    fn wrightomega_complex_vector_preserves_shape_and_conjugation() -> Result<(), String> {
        let z = Complex64::new(0.4, 0.7);
        let vector = wrightomega(
            &SpecialTensor::ComplexVec(vec![z, z.conj()]),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let values = expect_complex_vec(vector)?;
        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[1],
            values[0].conj(),
            1e-12,
            "wrightomega conjugation",
        );

        let scalar = wrightomega(&SpecialTensor::ComplexScalar(z), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let scalar_value = expect_complex_scalar(scalar)?;
        assert_complex_close(
            values[0],
            scalar_value,
            1e-12,
            "wrightomega vector lane matches scalar",
        );
        Ok(())
    }

    #[test]
    fn powm1_scalar_matches_naive_at_large_y_log_x() {
        // Where y * ln(x) is large, both formulations agree.
        for &x in &[2.0, 10.0, 100.0] {
            for &y in &[1.0, 2.5, 10.0] {
                let pm1 = powm1_scalar(x, y);
                let naive = x.powf(y) - 1.0;
                let rel = (pm1 - naive).abs() / naive.abs().max(1.0);
                assert!(rel < 1e-12, "powm1({x},{y})={pm1} vs naive={naive}");
            }
        }
    }

    #[test]
    fn powm1_scalar_metamorphic_beats_naive_near_one() {
        // For x.powf(y) ≈ 1 + ε with ε tiny, naive subtraction loses bits.
        // powm1 should preserve the linear ε term to full f64 precision.
        let x = 1.0 + 1.0e-15;
        let y = 1.0;
        let powm1 = powm1_scalar(x, y);
        let naive = x.powf(y) - 1.0;
        // The exact answer is 1.0e-15. Naive may round to 0; powm1 must not.
        assert!(
            powm1 > 0.5e-15 && powm1 < 1.5e-15,
            "powm1 should preserve 1e-15: got {powm1}"
        );
        // Sanity: prove naive is *worse*, i.e. powm1 is closer to truth.
        let truth = 1.0e-15;
        assert!(
            (powm1 - truth).abs() <= (naive - truth).abs(),
            "powm1 must beat naive near 1: powm1={powm1} naive={naive}"
        );
    }

    #[test]
    fn powm1_scalar_edge_cases() {
        assert_eq!(powm1_scalar(1.0, 5.0), 0.0);
        assert_eq!(powm1_scalar(2.5, 0.0), 0.0);
        assert_eq!(powm1_scalar(0.0, 2.0), -1.0);
        assert_eq!(powm1_scalar(0.0, -1.0), f64::INFINITY);
        assert_eq!(powm1_scalar(0.0, f64::NEG_INFINITY), f64::INFINITY);
        assert!(powm1_scalar(-2.0, 0.5).is_nan(), "negative^non-int = NaN");
        assert_eq!(powm1_scalar(-2.0, 3.0), -9.0, "(-2)^3 - 1 = -9");
        assert!(powm1_scalar(f64::NAN, 1.0).is_nan());
        assert!(powm1_scalar(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn powm1_supports_real_tensor_dispatch() -> Result<(), String> {
        let scalar = powm1(
            &SpecialTensor::RealScalar(1.0 + 1.0e-15),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .map_err(|err| err.to_string())?;
        let scalar_value = expect_real_scalar(scalar)?;
        assert!(scalar_value > 0.5e-15 && scalar_value < 1.5e-15);

        let bases = SpecialTensor::RealVec(vec![0.0, 1.0, 2.0, -2.0, -2.0]);
        let powers = SpecialTensor::RealVec(vec![-1.0, 5.0, 3.0, 3.0, 0.5]);
        let result = powm1(&bases, &powers, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        let expected = [f64::INFINITY, 0.0, powm1_scalar(2.0, 3.0), -9.0, f64::NAN];
        for (actual, expected) in values.into_iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn cosm1_scalar_matches_naive_at_moderate_x() {
        for &x in &[0.5, 1.0, 1.5, 2.0, 2.5] {
            let cm1 = cosm1_scalar(x);
            let naive = x.cos() - 1.0;
            assert!(
                (cm1 - naive).abs() < 1e-14,
                "cosm1({x})={cm1} vs naive={naive}"
            );
        }
    }

    #[test]
    fn cosm1_scalar_metamorphic_preserves_near_zero_quadratic() {
        // For x near 0, cos(x) - 1 ≈ -x²/2 + x⁴/24 - ...
        // The naive cos(x) - 1 returns 0 once cos(x) rounds to 1.0; cosm1
        // must return -x²/2 to leading order.
        let x = 1.0e-6;
        let cm1 = cosm1_scalar(x);
        let truth = -x * x * 0.5; // dominant quadratic term
        let rel = (cm1 - truth).abs() / truth.abs();
        assert!(rel < 1e-6, "cosm1({x})={cm1} vs truth≈{truth}, rel={rel}");
    }

    #[test]
    fn cosm1_scalar_edge_cases() {
        assert_eq!(cosm1_scalar(0.0), 0.0);
        assert!(cosm1_scalar(f64::NAN).is_nan());
        // Exactly π: cos(π) - 1 = -2.
        assert!((cosm1_scalar(std::f64::consts::PI) - (-2.0)).abs() < 1e-15);
    }

    #[test]
    fn cosm1_supports_real_tensor_dispatch() -> Result<(), String> {
        let scalar = cosm1(&SpecialTensor::RealScalar(0.5), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        assert!((expect_real_scalar(scalar)? - cosm1_scalar(0.5)).abs() < 1e-15);

        let input = vec![-1.0e-8, 0.0, 0.5, std::f64::consts::PI, f64::INFINITY];
        let result = cosm1(&SpecialTensor::RealVec(input.clone()), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let values = expect_real_vec(result)?;
        assert_eq!(values.len(), input.len());
        for (actual, x) in values.iter().zip(input) {
            let expected = cosm1_scalar(x);
            if expected.is_nan() {
                assert!(actual.is_nan(), "cosm1({x}) should be NaN");
            } else {
                assert!((actual - expected).abs() < 1e-15, "cosm1({x})");
            }
        }
        Ok(())
    }

    #[test]
    fn log1pmx_at_zero_is_exact_zero() {
        assert_eq!(log1pmx_scalar(0.0), 0.0);
    }

    #[test]
    fn log1pmx_metamorphic_matches_naive_at_moderate_x() {
        for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
            let lp = log1pmx_scalar(x);
            let naive = x.ln_1p() - x;
            assert!(
                (lp - naive).abs() < 1e-13 * naive.abs().max(1.0),
                "log1pmx({x}) = {lp} vs naive {naive}"
            );
        }
    }

    #[test]
    fn log1pmx_metamorphic_beats_naive_near_zero() {
        // At x = 1e-8, log1pmx(x) ≈ -x²/2 = -5e-17. The naive form
        // ln(1 + 1e-8) - 1e-8 has only ~7 significant digits left after
        // cancellation; the Taylor path keeps full precision.
        let x = 1.0e-8_f64;
        let lp = log1pmx_scalar(x);
        let truth = -x * x * 0.5; // dominant term
        let rel_err = (lp - truth).abs() / truth.abs();
        assert!(
            rel_err < 1e-6,
            "log1pmx({x}) = {lp}, expected ~{truth}, rel_err={rel_err}"
        );
    }

    #[test]
    fn log1pmx_propagates_nan_and_domain_error() {
        assert!(log1pmx_scalar(f64::NAN).is_nan());
        assert!(log1pmx_scalar(-1.0).is_nan());
        assert!(log1pmx_scalar(-2.0).is_nan());
    }

    #[test]
    fn log1pmx_metamorphic_signs() {
        // log1pmx(x) ≤ 0 for all x > -1 because ln(1+x) ≤ x by concavity.
        for &x in &[-0.5_f64, -0.1, -0.01, 0.01, 0.5, 1.0, 5.0, 100.0] {
            let v = log1pmx_scalar(x);
            assert!(v <= 0.0, "log1pmx({x}) = {v} should be ≤ 0");
        }
    }

    #[test]
    fn wofz_real_at_zero_is_one_zero() {
        let (re, im) = wofz_real(0.0);
        assert!((re - 1.0).abs() < 1e-15);
        assert!((im - 0.0).abs() < 1e-15);
    }

    #[test]
    fn wofz_real_metamorphic_real_part_is_gaussian() {
        // Re[w(x)] = exp(-x²) for all real x.
        for &x in &[-3.0_f64, -1.0, -0.3, 0.3, 1.0, 3.0] {
            let (re, _) = wofz_real(x);
            let expected = (-x * x).exp();
            assert!(
                (re - expected).abs() < 1e-15,
                "Re[w({x})] = {re}, expected {expected}"
            );
        }
    }

    #[test]
    fn wofz_real_metamorphic_imag_relates_to_dawson() {
        // Im[w(x)] = (2/√π) · dawsn(x).
        let scale = 2.0 / std::f64::consts::PI.sqrt();
        for &x in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            let (_, im) = wofz_real(x);
            let expected = scale * dawsn_scalar(x);
            assert!(
                (im - expected).abs() < 1e-12,
                "Im[w({x})] = {im}, expected {expected}"
            );
        }
    }

    #[test]
    fn wofz_real_metamorphic_odd_imag() {
        // Im[w(-x)] = -Im[w(x)] (because dawsn is odd).
        for &x in &[0.5_f64, 1.5, 3.0] {
            let (_, im_pos) = wofz_real(x);
            let (_, im_neg) = wofz_real(-x);
            assert!(
                (im_pos + im_neg).abs() < 1e-13,
                "Im[w({x})]={im_pos}, Im[w(-{x})]={im_neg} should be antisymmetric"
            );
        }
    }

    #[test]
    fn wofz_real_propagates_nan() {
        let (re, im) = wofz_real(f64::NAN);
        assert!(re.is_nan() && im.is_nan());
    }

    #[test]
    fn wofz_real_tensor_returns_complex_scipy_contract_point() -> Result<(), String> {
        let value = wofz(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let actual = expect_complex_scalar(value)?;
        let expected = Complex64::new(0.367_879_441_171_442_33, 0.607_157_705_841_393_7);
        assert_complex_close(actual, expected, 2.0e-10, "wofz(1 + 0j)");
        Ok(())
    }

    #[test]
    fn wofz_complex_matches_scipy_contract_points() -> Result<(), String> {
        let samples = [
            (
                Complex64::new(1.0, 0.5),
                Complex64::new(0.354_900_332_867_577_83, 0.342_871_719_131_100_8),
            ),
            (
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.354_900_332_867_577_83, -0.342_871_719_131_100_8),
            ),
            (
                Complex64::new(0.25, 1.25),
                Complex64::new(0.361_233_314_847_938_97, 0.051_429_596_928_865_56),
            ),
            (
                Complex64::new(2.0, 1.0),
                Complex64::new(0.140_239_581_366_277_98, 0.222_213_440_179_899_25),
            ),
            (
                Complex64::new(1.0, -0.5),
                Complex64::new(0.155_541_142_454_331_15, 1.137_837_215_781_686_5),
            ),
        ];

        for (z, expected) in samples {
            let actual = wofz_scalar(z, RuntimeMode::Strict).map_err(|err| err.to_string())?;
            assert_complex_close(actual, expected, 5.0e-8, "wofz complex SciPy contract");
        }
        Ok(())
    }

    #[test]
    fn wofz_complex_vector_preserves_shape_and_conjugation() -> Result<(), String> {
        let z = Complex64::new(0.75, 0.4);
        let values = expect_complex_vec(
            wofz(
                &SpecialTensor::ComplexVec(vec![z, -z.conj()]),
                RuntimeMode::Strict,
            )
            .map_err(|err| err.to_string())?,
        )?;
        assert_eq!(values.len(), 2);
        assert_complex_close(
            values[0],
            values[1].conj(),
            5.0e-8,
            "wofz conjugation for upper half-plane pair",
        );
        Ok(())
    }

    #[test]
    fn voigt_profile_real_gamma_zero_is_gaussian() {
        // V(x; σ, 0) = (1/(σ√(2π))) exp(-x²/(2σ²)).
        let sigma = 1.5;
        let coeff = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        for &x in &[-2.0_f64, 0.0, 1.0, 2.5] {
            let v = voigt_profile_real_gamma_zero(x, sigma);
            let expected = coeff * (-(x * x) / (2.0 * sigma * sigma)).exp();
            assert!((v - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn voigt_profile_real_gamma_zero_rejects_nonpositive_sigma() {
        assert!(voigt_profile_real_gamma_zero(1.0, 0.0).is_nan());
        assert!(voigt_profile_real_gamma_zero(1.0, -0.5).is_nan());
        assert!(voigt_profile_real_gamma_zero(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn voigt_profile_matches_scipy_contract_points() {
        let samples = [
            (0.0, 1.0, 0.0, 0.398_942_280_401_432_7),
            (1.0, 1.0, 0.0, 0.241_970_724_519_143_37),
            (0.0, 1.0, 1.0, 0.208_709_280_520_367_72),
            (1.0, 1.0, 1.0, 0.165_795_662_689_166_5),
            (0.0, 0.0, 1.0, std::f64::consts::FRAC_1_PI),
            (1.0, 0.0, 1.0, std::f64::consts::FRAC_1_PI / 2.0),
            (0.0, 0.0, 0.0, f64::INFINITY),
            (1.0, 0.0, 0.0, 0.0),
        ];
        for (x, sigma, gamma, expected) in samples {
            let actual = voigt_profile(x, sigma, gamma);
            if expected.is_infinite() {
                assert!(
                    actual.is_infinite(),
                    "voigt_profile({x}, {sigma}, {gamma}) = {actual}"
                );
            } else {
                assert!(
                    (actual - expected).abs() <= 5.0e-8,
                    "voigt_profile({x}, {sigma}, {gamma}) = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn tklmbda_at_origin_is_half_for_all_lambda() {
        // F(0; λ) = 1/2 for any λ by symmetry.
        for &lam in &[-0.5_f64, 0.0, 0.5, 1.0, 2.0] {
            let f = tklmbda(0.0, lam);
            assert!((f - 0.5).abs() < 1e-15, "F(0; {lam}) = {f}");
        }
    }

    #[test]
    fn tklmbda_lambda_zero_matches_logistic() {
        // λ = 0 collapses to the logistic CDF.
        for &x in &[-3.0_f64, -1.0, 0.0, 1.0, 3.0] {
            let logistic = 1.0 / (1.0 + (-x).exp());
            let f = tklmbda(x, 0.0);
            assert!(
                (f - logistic).abs() < 1e-12,
                "λ=0 logistic mismatch at x={x}: {f} vs {logistic}"
            );
        }
    }

    #[test]
    fn tklmbda_metamorphic_complement_symmetry() {
        // F(-x; λ) = 1 - F(x; λ) by symmetry of the Tukey-lambda family.
        for &lam in &[-0.5_f64, 0.5, 1.0, 1.5] {
            for &x in &[-1.0_f64, -0.3, 0.5, 1.0] {
                let a = tklmbda(x, lam);
                let b = tklmbda(-x, lam);
                assert!(
                    (a + b - 1.0).abs() < 1e-10,
                    "complement violated at (x={x}, λ={lam}): {a} + {b} = {}",
                    a + b
                );
            }
        }
    }

    #[test]
    fn tklmbda_lambda_one_uniform_image() {
        // For λ = 1, F⁻¹(p; 1) = (p^1 - (1-p)^1) / 1 = 2p - 1, so F(x; 1) = (x+1)/2 on [-1, 1].
        for &x in &[-0.8_f64, -0.3, 0.0, 0.5, 0.7] {
            let expected = (x + 1.0) * 0.5;
            let f = tklmbda(x, 1.0);
            assert!(
                (f - expected).abs() < 1e-10,
                "λ=1 affine mismatch at x={x}: {f} vs {expected}"
            );
        }
        // Saturation: x = ±1 must give 0 / 1 exactly.
        assert_eq!(tklmbda(-1.0, 1.0), 0.0);
        assert_eq!(tklmbda(1.0, 1.0), 1.0);
        assert_eq!(tklmbda(-2.0, 1.0), 0.0);
        assert_eq!(tklmbda(2.0, 1.0), 1.0);
    }

    #[test]
    fn tklmbda_propagates_nan() {
        assert!(tklmbda(f64::NAN, 0.5).is_nan());
        assert!(tklmbda(0.5, f64::NAN).is_nan());
    }

    #[test]
    fn modstruve_at_zero_handles_v_minus_one() {
        // Same leading-term behavior as struve: ride-along with
        // [frankenscipy-udtt9].
        assert_eq!(modstruve(0.0, 0.0), 0.0);
        assert_eq!(modstruve(1.5, 0.0), 0.0);
        let l_minus_one = modstruve(-1.0, 0.0);
        let expected = 2.0 / PI;
        assert!(
            (l_minus_one - expected).abs() < 1e-12,
            "modstruve(-1, 0) = {l_minus_one}, expected {expected}"
        );
        assert!(modstruve(-2.0, 0.0).is_nan());
    }

    #[test]
    fn struve_at_zero_handles_v_minus_one() {
        // [frankenscipy-udtt9] Regression: previously returned 0 for
        // every v at x=0; the leading series term gives a finite
        // value at v=-1 and 0 only for v > -1.
        // For v > -1, H_v(0) = 0.
        assert_eq!(struve(0.0, 0.0), 0.0);
        assert_eq!(struve(0.5, 0.0), 0.0);
        assert_eq!(struve(1.5, 0.0), 0.0);
        assert_eq!(struve(2.0, 0.0), 0.0);
        // For v = -1, H_{-1}(0) = 2/π.
        let h_minus_one = struve(-1.0, 0.0);
        let expected = 2.0 / PI;
        assert!(
            (h_minus_one - expected).abs() < 1e-12,
            "struve(-1, 0) = {h_minus_one}, expected {expected}"
        );
        // For v < -1, the leading term diverges and the limit sign is
        // undefined; we conservatively return NaN.
        assert!(struve(-1.5, 0.0).is_nan());
        assert!(struve(-2.0, 0.0).is_nan());
    }

    #[test]
    fn struve_integral_scalars_match_scipy_reference_values() {
        let cases = [
            (itstruve0 as fn(f64) -> f64, 0.0, 0.0, "itstruve0(0)"),
            (
                itstruve0 as fn(f64) -> f64,
                1.0,
                0.301_090_426_708_055_47,
                "itstruve0(1)",
            ),
            (
                itstruve0 as fn(f64) -> f64,
                -1.0,
                0.301_090_426_708_055_47,
                "itstruve0(-1)",
            ),
            (
                it2struve0 as fn(f64) -> f64,
                0.0,
                std::f64::consts::FRAC_PI_2,
                "it2struve0(0)",
            ),
            (
                it2struve0 as fn(f64) -> f64,
                1.0,
                0.957_197_350_638_352_4,
                "it2struve0(1)",
            ),
            (
                it2struve0 as fn(f64) -> f64,
                -1.0,
                2.184_395_302_951_440_7,
                "it2struve0(-1)",
            ),
            (
                itmodstruve0 as fn(f64) -> f64,
                1.0,
                0.336_472_628_644_038_4,
                "itmodstruve0(1)",
            ),
            (
                itmodstruve0 as fn(f64) -> f64,
                -1.0,
                0.336_472_628_644_038_4,
                "itmodstruve0(-1)",
            ),
        ];

        for &(func, x, expected, label) in &cases {
            let actual = func(x);
            let tol = 2.0e-7 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{label} = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn struve_integral_scalars_propagate_nonfinite_inputs() {
        for func in [
            itstruve0 as fn(f64) -> f64,
            it2struve0 as fn(f64) -> f64,
            itmodstruve0 as fn(f64) -> f64,
        ] {
            assert!(func(f64::NAN).is_nan());
            assert!(func(f64::INFINITY).is_nan());
            assert!(func(f64::NEG_INFINITY).is_nan());
        }
    }

    #[test]
    fn diric_at_zero_is_one() {
        // diric(0, n) = 1 for all n >= 1
        for n in 1..=10 {
            let val = diric(0.0, n);
            assert!(
                (val - 1.0).abs() < 1e-15,
                "diric(0, {n}) = {val}, expected 1.0"
            );
        }
    }

    #[test]
    fn diric_n_one_is_always_one() {
        // diric(x, 1) = sin(x/2) / sin(x/2) = 1 for all x
        for &x in &[0.0, 0.5, 1.0, 2.0, PI, 2.0 * PI, -1.0] {
            let val = diric(x, 1);
            assert!(
                (val - 1.0).abs() < 1e-15,
                "diric({x}, 1) = {val}, expected 1.0"
            );
        }
    }

    #[test]
    fn diric_general_case_matches_scipy() {
        // Values verified against scipy.special.diric
        let cases = [
            (0.1, 3, 0.9966694435186839),
            (0.1, 4, 0.9937606691655042),
            (0.5, 5, 0.767_153_947_103_405),
            (1.0, 3, 0.6935348705787598),
            (1.0, 5, 0.249_662_187_728_399),
        ];
        for (x, n, expected) in cases {
            let val = diric(x, n);
            assert!(
                (val - expected).abs() < 1e-12,
                "diric({x}, {n}) = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn diric_at_multiples_of_two_pi() {
        // At x = 2kπ:
        // - n odd: result = 1
        // - n even: result = (-1)^k
        let two_pi = 2.0 * PI;

        // x = 2π (k=1)
        assert!((diric(two_pi, 3) - 1.0).abs() < 1e-14, "odd n at 2π");
        assert!((diric(two_pi, 5) - 1.0).abs() < 1e-14, "odd n at 2π");
        assert!((diric(two_pi, 4) - (-1.0)).abs() < 1e-14, "even n at 2π");
        assert!((diric(two_pi, 6) - (-1.0)).abs() < 1e-14, "even n at 2π");

        // x = 4π (k=2)
        assert!((diric(4.0 * PI, 3) - 1.0).abs() < 1e-14, "odd n at 4π");
        assert!(
            (diric(4.0 * PI, 4) - 1.0).abs() < 1e-14,
            "even n at 4π, k=2"
        );

        // x = -2π (k=-1)
        assert!((diric(-two_pi, 3) - 1.0).abs() < 1e-14, "odd n at -2π");
        assert!((diric(-two_pi, 4) - (-1.0)).abs() < 1e-14, "even n at -2π");
    }

    #[test]
    fn diric_propagates_nan_and_rejects_invalid_n() {
        assert!(diric(f64::NAN, 5).is_nan());
        assert!(diric(0.5, 0).is_nan()); // n < 1
        assert!(diric(0.5, -1).is_nan()); // n < 1
    }

    #[test]
    fn ndtr_matches_scipy_reference_values() {
        // scipy.special.ndtr([-2.0, -1.0, 0.0, 1.0, 2.0])
        let cases = [
            (-2.0, 0.02275013194817921),
            (-1.0, 0.15865525393145707),
            (0.0, 0.5),
            (1.0, 0.8413447460685429),
            (2.0, 0.9772498680518208),
        ];
        for (x, expected) in cases {
            let result = super::ndtr_scalar(x);
            assert!(
                (result - expected).abs() < 1e-10,
                "ndtr({x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn ndtri_matches_scipy_reference_values() {
        // scipy.special.ndtri([0.1, 0.25, 0.5, 0.75, 0.9])
        let cases = [
            (0.1, -1.2815515655446004),
            (0.25, -0.6744897501960817),
            (0.5, 0.0),
            (0.75, 0.6744897501960817),
            (0.9, 1.2815515655446004),
        ];
        for (y, expected) in cases {
            let result = super::ndtri_scalar(y);
            assert!(
                (result - expected).abs() < 1e-10,
                "ndtri({y}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn fresnel_matches_scipy_reference_values() {
        // scipy.special.fresnel([0.5, 1.0, 2.0])
        let cases = [
            (0.5, 0.06473243285999929, 0.4923442258714464),
            (1.0, 0.4382591473903548, 0.7798934003768228),
            (2.0, 0.34341567836369824, 0.4882534060753408),
        ];
        for (x, expected_s, expected_c) in cases {
            let (s, c) = super::fresnel(x);
            assert!(
                (s - expected_s).abs() < 1e-6,
                "fresnel({x}).s = {s}, expected {expected_s}"
            );
            assert!(
                (c - expected_c).abs() < 1e-6,
                "fresnel({x}).c = {c}, expected {expected_c}"
            );
        }
    }

    #[test]
    fn sici_matches_scipy_reference_values() {
        // scipy.special.sici([1.0, 2.0, 5.0])
        let cases = [
            (1.0, 0.9460830703671831, 0.33740392290096817),
            (2.0, 1.6054129768026948, 0.422_980_828_084_050_6),
            (5.0, 1.5499312449446702, -0.19002974965664387),
        ];
        for (x, expected_si, expected_ci) in cases {
            let (si, ci) = super::sici(x);
            assert!(
                (si - expected_si).abs() < 1e-6,
                "sici({x}).si = {si}, expected {expected_si}"
            );
            assert!(
                (ci - expected_ci).abs() < 1e-6,
                "sici({x}).ci = {ci}, expected {expected_ci}"
            );
        }
    }

    #[test]
    fn shichi_matches_scipy_reference_values() {
        // scipy.special.shichi([0.5, 1.0, 2.0])
        let cases = [
            (0.5, 0.5069967498196671, -0.05277684495649361),
            (1.0, 1.0572508753757285, 0.8378669409802082),
            (2.0, 2.5015674311761847, 2.4529408862140567),
        ];
        for (x, expected_shi, expected_chi) in cases {
            let (shi, chi) = super::shichi(x);
            assert!(
                (shi - expected_shi).abs() < 1e-6,
                "shichi({x}).shi = {shi}, expected {expected_shi}"
            );
            assert!(
                (chi - expected_chi).abs() < 1e-3,
                "shichi({x}).chi = {chi}, expected {expected_chi}"
            );
        }
    }

    #[test]
    fn struve_matches_scipy_reference_values() {
        // scipy.special.struve([0, 1], [1.0, 2.0])
        let cases = [
            (0.0, 1.0, 0.5686246925337326),
            (1.0, 2.0, 0.645_931_651_099_601),
        ];
        for (v, x, expected) in cases {
            let result = super::struve(v, x);
            assert!(
                (result - expected).abs() < 1e-3,
                "struve({v}, {x}) = {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn sinc_matches_scipy_reference_values() {
        // scipy.special.sinc(0.5) = 0.6366197723675814
        use crate::SpecialTensor;
        use fsci_runtime::RuntimeMode;
        let x = SpecialTensor::RealScalar(0.5);
        let result = super::sinc(&x, RuntimeMode::Strict).expect("sinc");
        if let SpecialTensor::RealScalar(val) = result {
            assert!(
                (val - std::f64::consts::FRAC_2_PI).abs() < 1e-6,
                "sinc(0.5) = {val}, expected 0.6366197723675814"
            );
        } else {
            panic!("sinc should return scalar");
        }
    }

    #[test]
    fn expit_matches_scipy_reference_values() {
        // scipy.special.expit(0) = 0.5
        use crate::SpecialTensor;
        use fsci_runtime::RuntimeMode;
        let x = SpecialTensor::RealScalar(0.0);
        let result = super::expit(&x, RuntimeMode::Strict).expect("expit");
        if let SpecialTensor::RealScalar(val) = result {
            assert!((val - 0.5).abs() < 1e-10, "expit(0) = {val}, expected 0.5");
        } else {
            panic!("expit should return scalar");
        }
    }

    #[test]
    fn logit_matches_scipy_reference_values() {
        // scipy.special.logit(0.5) = 0.0
        use crate::SpecialTensor;
        use fsci_runtime::RuntimeMode;
        let p = SpecialTensor::RealScalar(0.5);
        let result = super::logit(&p, RuntimeMode::Strict).expect("logit");
        if let SpecialTensor::RealScalar(val) = result {
            assert!(val.abs() < 1e-10, "logit(0.5) = {val}, expected 0.0");
        } else {
            panic!("logit should return scalar");
        }
    }

    #[test]
    fn poch_negative_args_match_scipy_signed() {
        // scipy.special.poch = Γ(x+n)/Γ(x) is SIGNED; the gammaln fallback dropped
        // the sign for negative arguments (the integer-n product path was fine).
        let cases = [
            (-4.3, 0.5, -2.938123324828691_f64),
            (-2.5, 3.5, -1.057855469152043),
            (-0.5, 1.5, -0.28209479177387814),
            (-4.3, -1.5, -0.10553603896654783),
            (2.5, -1.5, 0.7522527780636751),
            (3.0, 4.0, 360.0), // positive args (product path): unchanged
        ];
        for (x, n, want) in cases {
            let got = super::poch(x, n);
            assert!(
                (got - want).abs() <= 1e-11 * want.abs().max(1.0),
                "poch({x},{n}) got {got}, want {want}"
            );
        }
    }

    #[test]
    fn kl_div_matches_scipy_reference_values() {
        // scipy.special.kl_div(1, 2) = 0.3068528194400546
        let result = super::kl_div(1.0, 2.0);
        assert!(
            (result - 0.3068528194400546).abs() < 1e-6,
            "kl_div(1, 2) = {result}, expected 0.3068528194400546"
        );
    }
}
