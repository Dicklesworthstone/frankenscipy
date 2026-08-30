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
#[cfg(feature = "ndtri-isafloor-bench")]
use std::simd::Simd;

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

// Retained REFERENCE implementation: the optimized path's own documentation
// cites it, and it is exercised by a test that checks the two agree. Dead only
// in non-test builds, so the test build still proves it is used
// (frankenscipy-e2ve2).
#[cfg_attr(not(test), allow(dead_code))]
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

    // Each reduced entry is an INDEPENDENT logsumexp over one column/row, so fan the
    // outputs across threads with the order-preserving `par_map_indices`. BYTE-IDENTICAL
    // to the serial loop: the within-reduction summation order is untouched (each call
    // is the same `logsumexp`/`logsumexp_with_b`), and results land in output order.
    // (scipy's vectorized 2D path already loses ~5x to fsci's fused per-row loop; this
    // widens that lead to ~25x by using the idle cores.)
    match axis {
        0 => par_map_indices(cols, |col| {
            let column = data
                .iter()
                .map(|row_values| row_values[col])
                .collect::<Vec<_>>();
            if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                let column_weights = (0..rows)
                    .map(|row| weight_at(weights, weight_rows, weight_cols, row, col))
                    .collect::<Vec<_>>();
                logsumexp_with_b(&column, &column_weights)
            } else {
                Ok(logsumexp(&column))
            }
        }),
        1 => par_map_indices(rows, |row| {
            let row_values = &data[row];
            if let (Some(weights), Some((weight_rows, weight_cols))) = (b, weight_shape) {
                let row_weights = (0..cols)
                    .map(|col| weight_at(weights, weight_rows, weight_cols, row, col))
                    .collect::<Vec<_>>();
                logsumexp_with_b(row_values, &row_weights)
            } else {
                Ok(logsumexp(row_values))
            }
        }),
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
    // expit is compute-bound (exp + reciprocal), so it parallelizes for large
    // arrays via the work-capped light path (was unconditionally serial, ~1.8x
    // slower than cephes at n≈200k-500k).
    map_real_light("expit", x_tensor, mode, |x| Ok(expit_scalar(x)))
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
    // ndtr(x) = ½·erfc(−x/√2); the scalar-per-element map lost to SciPy's SIMD ndtr
    // ufunc. Reuse the full-domain SIMD erfc chunk (byte-identical for |−x/√2| < 1,
    // a few ulp on the tail — far inside ndtr's tolerance). Below the 1<<20 gate.
    if let SpecialTensor::RealVec(values) = x_tensor
        && (64..(1 << 20)).contains(&values.len())
    {
        return Ok(SpecialTensor::RealVec(ndtr_real_vec_simd(values)));
    }
    map_real_wg("ndtr", x_tensor, mode, |x| Ok(ndtr_scalar(x)))
}

/// 8-wide ndtr: `½·erfc(−x/√2)` via the shared SIMD erfc chunk.
fn ndtr_real_vec_simd(values: &[f64]) -> Vec<f64> {
    use std::simd::Simd;
    const LANES: usize = 8;
    let scale = Simd::<f64, LANES>::splat(-FRAC_1_SQRT_2);
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let x = Simd::<f64, LANES>::from_slice(&values[i..i + LANES]);
        let erfc = crate::error::erfc_full_simd_chunk(x * scale);
        for j in 0..LANES {
            out[i + j] = 0.5 * erfc[j];
        }
        i += LANES;
    }
    while i < values.len() {
        out[i] = ndtr_scalar(values[i]);
        i += 1;
    }
    out
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
    map_real_wg("ndtri", y_tensor, mode, |y| Ok(ndtri_scalar(y)))
}

#[must_use]
pub fn ndtri_scalar(y: f64) -> f64 {
    ndtri_scalar_with(
        y,
        NDTRI_UNROLL_POLEVL.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Evaluate the Cephes rationals with a compile-time degree (`true`, shipping) instead of a
/// runtime slice length.
///
/// BIT-IDENTICAL: identical coefficients and identical Horner order in both arms. This is a
/// codegen-shape lever, not a numeric one.
pub static NDTRI_UNROLL_POLEVL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// BATCHES that took the unrolled arm — "enabled" is not "took effect". Counted per batch.
pub static NDTRI_UNROLL_POLEVL_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `ndtri_scalar` with the evaluator choice supplied by the caller, so the flag is read once
/// per batch rather than once per element.
#[must_use]
pub fn ndtri_scalar_with(y: f64, unrolled: bool) -> f64 {
    if unrolled {
        ndtri_kernel::<true>(y)
    } else {
        ndtri_kernel::<false>(y)
    }
}

fn ndtri_kernel<const UNROLL: bool>(y: f64) -> f64 {
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
    let mut p = y;
    let lower_tail = if p > 1.0 - CEPHES_NDTRI_EXP_NEG2 {
        p = 1.0 - p;
        false
    } else {
        true
    };

    if p > CEPHES_NDTRI_EXP_NEG2 {
        let w = p - 0.5;
        let w2 = w * w;
        return (w + w
            * (w2 * ndtri_pe::<_, UNROLL>(w2, &CEPHES_NDTRI_P0)
                / ndtri_p1e::<_, UNROLL>(w2, &CEPHES_NDTRI_Q0)))
            * CEPHES_NDTRI_SQRT_2PI;
    }

    let x = {
        let z = (-2.0 * p.ln()).sqrt();
        let z0 = z - z.ln() / z;
        let inv_z = 1.0 / z;
        let correction = if z < 8.0 {
            inv_z * ndtri_pe::<_, UNROLL>(inv_z, &CEPHES_NDTRI_P1)
                / ndtri_p1e::<_, UNROLL>(inv_z, &CEPHES_NDTRI_Q1)
        } else {
            inv_z * ndtri_pe::<_, UNROLL>(inv_z, &CEPHES_NDTRI_P2)
                / ndtri_p1e::<_, UNROLL>(inv_z, &CEPHES_NDTRI_Q2)
        };
        z0 - correction
    };

    if lower_tail { -x } else { x }
}

/// Scalar-map baseline for the pre-AVX2 `ndtri` SIMD retry.
///
/// This is benchmark-only evidence plumbing, not a second public algorithm.
#[cfg(feature = "ndtri-isafloor-bench")]
#[doc(hidden)]
#[must_use]
pub fn ndtri_isafloor_scalar_baseline(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(ndtri_scalar).collect()
}

/// Reconstructed central-rational SIMD candidate from the 2026-07-04 reject.
///
/// Tail lanes intentionally retain the rejected candidate's counted work:
/// the central rational is evaluated for the whole vector before those lanes
/// are replaced with the existing scalar log/sqrt result.
#[cfg(feature = "ndtri-isafloor-bench")]
#[doc(hidden)]
#[must_use]
pub fn ndtri_isafloor_simd_candidate(values: &[f64]) -> Vec<f64> {
    const LANES: usize = 8;

    let mut output = vec![0.0; values.len()];
    let mut offset = 0;
    while offset + LANES <= values.len() {
        let y = Simd::<f64, LANES>::from_slice(&values[offset..offset + LANES]);
        let w = y - Simd::splat(0.5);
        let w2 = w * w;
        let central = (w + w
            * (w2 * cephes_ndtri_polevl_simd(w2, &CEPHES_NDTRI_P0)
                / cephes_ndtri_p1evl_simd(w2, &CEPHES_NDTRI_Q0)))
            * Simd::splat(CEPHES_NDTRI_SQRT_2PI);
        let central = central.to_array();

        for lane in 0..LANES {
            let probability = values[offset + lane];
            output[offset + lane] = if probability > CEPHES_NDTRI_EXP_NEG2
                && probability < 1.0 - CEPHES_NDTRI_EXP_NEG2
            {
                central[lane]
            } else {
                ndtri_scalar(probability)
            };
        }
        offset += LANES;
    }
    for index in offset..values.len() {
        output[index] = ndtri_scalar(values[index]);
    }
    output
}

#[cfg(feature = "ndtri-isafloor-bench")]
fn cephes_ndtri_polevl_simd(x: Simd<f64, 8>, coefficients: &[f64]) -> Simd<f64, 8> {
    coefficients
        .iter()
        .copied()
        .fold(Simd::splat(0.0), |accumulator, coefficient| {
            accumulator * x + Simd::splat(coefficient)
        })
}

#[cfg(feature = "ndtri-isafloor-bench")]
fn cephes_ndtri_p1evl_simd(x: Simd<f64, 8>, coefficients: &[f64]) -> Simd<f64, 8> {
    let mut accumulator = x + Simd::splat(coefficients[0]);
    for &coefficient in &coefficients[1..] {
        accumulator = accumulator * x + Simd::splat(coefficient);
    }
    accumulator
}

const CEPHES_NDTRI_EXP_NEG2: f64 = 0.135_335_283_236_612_7;
const CEPHES_NDTRI_SQRT_2PI: f64 = 2.506_628_274_631_000_7;

const CEPHES_NDTRI_P0: [f64; 5] = [
    -5.996_335_010_141_079e1,
    9.800_107_541_859_997e1,
    -5.667_628_574_690_703e1,
    1.393_126_093_872_796_8e1,
    -1.239_165_838_673_812_5,
];

const CEPHES_NDTRI_Q0: [f64; 8] = [
    1.954_488_583_381_417_6,
    4.676_279_128_988_815,
    8.636_024_213_908_906e1,
    -2.254_626_878_541_193_8e2,
    2.002_602_123_800_606_6e2,
    -8.203_722_561_683_333e1,
    1.590_562_251_262_117e1,
    -1.183_316_211_213_300_1,
];

const CEPHES_NDTRI_P1: [f64; 9] = [
    4.055_448_923_059_624,
    3.152_510_945_998_938_5e1,
    5.716_281_922_464_213e1,
    4.408_050_738_932_008e1,
    1.468_495_619_288_580_2e1,
    2.186_633_068_507_902_5,
    -1.402_560_791_713_545e-1,
    -3.504_246_268_278_482e-2,
    -8.574_567_851_546_854e-4,
];

const CEPHES_NDTRI_Q1: [f64; 8] = [
    1.577_998_832_564_667_5e1,
    4.539_076_351_288_792e1,
    4.131_720_382_546_72e1,
    1.504_253_856_929_075e1,
    2.504_649_462_083_094,
    -1.421_829_228_547_877_8e-1,
    -3.808_064_076_915_783e-2,
    -9.332_594_808_954_574e-4,
];

const CEPHES_NDTRI_P2: [f64; 9] = [
    3.237_748_917_769_460_3,
    6.915_228_890_689_842,
    3.938_810_252_924_744_4,
    1.333_034_608_158_075_5,
    2.014_853_895_491_790_8e-1,
    1.237_166_348_178_200_2e-2,
    3.015_815_535_082_354e-4,
    2.658_069_746_867_375_5e-6,
    6.239_745_391_849_833e-9,
];

const CEPHES_NDTRI_Q2: [f64; 8] = [
    6.024_270_393_647_42,
    3.679_835_638_561_608_7,
    1.377_020_994_890_813_2,
    2.162_369_935_944_966_3e-1,
    1.342_040_060_885_431_8e-2,
    3.280_144_646_821_277_4e-4,
    2.892_478_647_453_806_8e-6,
    6.790_194_080_099_813e-9,
];

/// Horner with the degree known at COMPILE time, taking the table as an array rather than a
/// slice.
///
/// WHY BOTH FORMS EXIST. `cephes_ndtri_polevl` below takes `&[f64]`, so every call site's
/// fixed-size table (`[f64; 5]`, `[f64; 8]`, `[f64; 9]`) is coerced to a slice and its length
/// becomes a runtime value: a loop with a counter and a bounds-checked iterator. SciPy's
/// evaluator is a template whose degree is a compile-time constant and unrolls completely.
/// Whether LLVM already recovers that through inlining is not something to assume — it is what
/// the A/B measures.
///
/// BIT-IDENTICAL: same coefficients, same left-to-right Horner order, same rounding.
#[inline(always)]
fn cephes_ndtri_polevl_n<const N: usize>(x: f64, coef: &[f64; N]) -> f64 {
    coef.iter().copied().fold(0.0, |acc, c| acc * x + c)
}

/// Monic Horner with the degree known at compile time. See `cephes_ndtri_polevl_n`.
#[inline(always)]
fn cephes_ndtri_p1evl_n<const N: usize>(x: f64, coef: &[f64; N]) -> f64 {
    coef.iter().copied().fold(1.0, |acc, c| acc * x + c)
}

/// Pick the evaluator by a const flag, so each arm monomorphises with NO runtime branch.
#[inline(always)]
fn ndtri_pe<const N: usize, const UNROLL: bool>(x: f64, coef: &[f64; N]) -> f64 {
    if UNROLL {
        cephes_ndtri_polevl_n(x, coef)
    } else {
        cephes_ndtri_polevl(x, coef)
    }
}

/// Monic counterpart of `ndtri_pe`.
#[inline(always)]
fn ndtri_p1e<const N: usize, const UNROLL: bool>(x: f64, coef: &[f64; N]) -> f64 {
    if UNROLL {
        cephes_ndtri_p1evl_n(x, coef)
    } else {
        cephes_ndtri_p1evl(x, coef)
    }
}

fn cephes_ndtri_polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().copied().fold(0.0, |acc, c| acc * x + c)
}

fn cephes_ndtri_p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = x + coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
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
    let (s, c) = cephes_fresnl(z.abs());
    if z < 0.0 { (-s, -c) } else { (s, c) }
}

/// When `true`, [`erf_zeros`] computes its zeros serially (the ORIG behaviour); default `false` fans
/// the independent per-zero Newton root-finds across index-chunks. Byte-identical. `#[doc(hidden)]`.
#[doc(hidden)]
pub static ERF_ZEROS_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// First `nt` complex zeros of `erf` in the first quadrant, ordered by absolute
/// value, matching `scipy.special.erf_zeros`.
///
/// Each zero is found by Newton iteration `z ← z − erf(z)/erf'(z)`
/// (`erf'(z) = (2/√π) e^{-z²}`) from the Zhang–Jin specfun initial estimate
/// `z₀ = (½pu − ½ln(pv)/pu) + i(½pu + ½ln(pv)/pu)`, `pu = √(π(4nr−½))`,
/// `pv = π√(2nr−¼)`.
#[must_use]
pub fn erf_zeros(nt: usize) -> Vec<Complex64> {
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    // Each output zero `nr = i+1` is a pure function of its index: a closed-form asymptotic guess,
    // then a self-contained 60-step Newton refinement whose per-step cost is an `erf_complex_scalar`
    // plus a complex exp. No dependency on any other zero (no prev/shared state/scatter), so the
    // loop is embarrassingly parallel and each element is written to its own index slot →
    // BYTE-IDENTICAL to `(0..nt).map(compute).collect()`.
    let compute = |i: usize| -> Complex64 {
        let nrf = (i + 1) as f64;
        let pu = (PI * (4.0 * nrf - 0.5)).sqrt();
        let pv = PI * (2.0 * nrf - 0.25).sqrt();
        let ln_pv = pv.ln();
        let mut z = Complex64::new(0.5 * pu - 0.5 * ln_pv / pu, 0.5 * pu + 0.5 * ln_pv / pu);
        for _ in 0..60 {
            let f = crate::error::erf_complex_scalar(z);
            let fp = Complex64::from_real(two_over_sqrt_pi) * (-(z * z)).exp();
            let dz = f / fp;
            z = z - dz;
            if dz.abs() < 1e-15 * z.abs() {
                break;
            }
        }
        z
    };

    if ERF_ZEROS_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        return (0..nt).map(&compute).collect();
    }
    par_map_indices(nt, |i| Ok::<Complex64, SpecialError>(compute(i)))
        .expect("erf zeros are infallible")
}

/// Complex Fresnel integrals `(S(z), C(z))` for complex argument `z`, matching
/// `scipy.special.fresnel(z)` on the complex plane.
///
/// Evaluated through the complex error function via the exact identities
///
/// ```text
///   C(z) + i S(z) = ½(1+i) erf((√π/2)(1−i) z)
///   C(z) − i S(z) = ½(1−i) erf((√π/2)(1+i) z)
/// ```
///
/// so that `C = ½(p·e₁ + p̄·e₂)` and `S = −(i/2)(p·e₁ − p̄·e₂)` with
/// `p = (1+i)/2`, `e₁ = erf((√π/2)(1−i)z)`, `e₂ = erf((√π/2)(1+i)z)`. On the
/// real axis this reduces to the real [`fresnel`]; verified against
/// `scipy.special.fresnel` to ≤ 1e-14 relative error across the complex plane.
#[must_use]
pub fn fresnel_complex(z: Complex64) -> (Complex64, Complex64) {
    let h = PI.sqrt() / 2.0;
    let e1 = crate::error::erf_complex_scalar(Complex64::new(h, -h) * z);
    let e2 = crate::error::erf_complex_scalar(Complex64::new(h, h) * z);
    let p = Complex64::new(0.5, 0.5); // (1+i)/2
    let pc = Complex64::new(0.5, -0.5); // (1-i)/2
    let c = (p * e1 + pc * e2) / 2.0;
    // S = (p·e1 − pc·e2) / (2i) = (p·e1 − pc·e2) · (−i/2).
    let s = (p * e1 - pc * e2) * Complex64::new(0.0, -0.5);
    (s, c)
}

/// First `nt` complex zeros of the Fresnel integral `C` (`is_c = true`) or `S`
/// (`is_c = false`) in the first quadrant, ordered by absolute value.
/// When `true`, the Fresnel-zero finders ([`fresnelc_zeros`]/[`fresnels_zeros`]/[`fresnel_zeros`])
/// compute their zeros serially (the ORIG behaviour); default `false` fans the independent per-zero
/// Newton root-finds across index-chunks. Byte-identical. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static FRESNEL_ZEROS_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn fresnel_zeros_kind(nt: usize, is_c: bool) -> Vec<Complex64> {
    // Each output zero `nr = i+1` is a pure function of its index: a closed-form asymptotic guess,
    // then a self-contained 60-step Newton refinement whose per-step cost is a `fresnel_complex`
    // (two complex error functions) plus a complex cos/sin. No dependency on any other zero (no
    // prev, no shared state, no scatter), so the loop is embarrassingly parallel and each element
    // is written to its own index slot → BYTE-IDENTICAL to `(0..nt).map(compute).collect()`.
    let compute = |i: usize| -> Complex64 {
        let nrf = (i + 1) as f64;
        // C zeros cluster near √(4nr−1), S zeros near √(4nr) (where C'/S' vanish).
        let psq = if is_c { 4.0 * nrf - 1.0 } else { 4.0 * nrf }.sqrt();
        let ln_term = (PI * psq).ln();
        let px = psq - ln_term / (PI * PI * psq.powi(3));
        let py = ln_term / (PI * psq);
        let mut z = Complex64::new(px, py);
        for _ in 0..60 {
            let (s, c) = fresnel_complex(z);
            let arg = z * z * Complex64::from_real(PI / 2.0);
            let (f, fp) = if is_c { (c, arg.cos()) } else { (s, arg.sin()) };
            let dz = f / fp;
            z = z - dz;
            if dz.abs() < 1e-15 * z.abs() {
                break;
            }
        }
        z
    };

    if FRESNEL_ZEROS_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        return (0..nt).map(&compute).collect();
    }
    par_map_indices(nt, |i| Ok::<Complex64, SpecialError>(compute(i)))
        .expect("fresnel zeros are infallible")
}

/// First `nt` complex zeros of the Fresnel cosine integral `C`, matching
/// `scipy.special.fresnelc_zeros`.
#[must_use]
pub fn fresnelc_zeros(nt: usize) -> Vec<Complex64> {
    fresnel_zeros_kind(nt, true)
}

/// First `nt` complex zeros of the Fresnel sine integral `S`, matching
/// `scipy.special.fresnels_zeros`.
#[must_use]
pub fn fresnels_zeros(nt: usize) -> Vec<Complex64> {
    fresnel_zeros_kind(nt, false)
}

/// First `nt` complex zeros of the Fresnel `S` and `C` integrals as
/// `(zeros_of_S, zeros_of_C)`, matching `scipy.special.fresnel_zeros`.
#[must_use]
pub fn fresnel_zeros(nt: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    (fresnels_zeros(nt), fresnelc_zeros(nt))
}

// ── cephes fresnl.c port ──────────────────────────────────────────────────
// scipy.special.fresnel uses cephes for real arguments: small-x rational
// approximations, a large-x sentinel, and a single rational auxiliary-function
// branch for the wide middle range. Reproducing it gives parity with scipy
// (the prior 200-point Simpson mid-branch was ~1e-5 off near x∈[2.5,5]).

/// Horner evaluation of a polynomial with all coefficients listed,
/// `coef[0]` the highest degree (cephes `polevl`).
fn fresnel_polevl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
}

/// Like [`fresnel_polevl`] but with an implied leading coefficient of 1.0
/// (cephes `p1evl`).
fn fresnel_p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = x + coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
}

#[allow(clippy::excessive_precision)]
const FRESNEL_SN: [f64; 6] = [
    -2.991_819_194_010_198_537_26e3,
    7.088_400_452_577_385_768_63e5,
    -6.297_414_862_058_625_065_37e7,
    2.548_908_805_733_763_591_04e9,
    -4.429_795_180_596_977_791_03e10,
    3.180_162_978_765_678_179_86e11,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_SD: [f64; 6] = [
    2.813_762_688_899_943_156_96e2,
    4.558_478_108_065_325_816_75e4,
    5.173_438_887_700_964_007_30e6,
    4.193_202_458_981_112_311_29e8,
    2.244_117_956_453_409_209_40e10,
    6.073_663_894_900_846_390_49e11,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_CN: [f64; 6] = [
    -4.988_431_145_735_735_486_51e-8,
    9.504_280_628_298_596_051_34e-6,
    -6.451_914_356_839_650_509_62e-4,
    1.888_433_193_967_038_500_64e-2,
    -2.055_259_009_550_138_917_93e-1,
    9.999_999_999_999_999_988_22e-1,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_CD: [f64; 7] = [
    3.999_829_689_724_959_803_67e-12,
    9.154_392_157_746_574_787_99e-10,
    1.250_018_624_795_988_214_74e-7,
    1.222_627_890_241_790_309_97e-5,
    8.680_295_429_417_843_006_06e-4,
    4.121_420_907_221_997_929_36e-2,
    1.000_000_000_000_000_001_18e0,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_FN: [f64; 10] = [
    4.215_435_550_436_775_465_06e-1,
    1.434_079_197_807_588_852_61e-1,
    1.152_209_550_735_857_588_35e-2,
    3.450_179_397_825_740_279_00e-4,
    4.636_137_492_878_673_220_88e-6,
    3.055_689_837_902_576_058_27e-8,
    1.023_045_141_649_072_334_65e-10,
    1.720_107_432_681_618_288_79e-13,
    1.342_832_762_330_627_589_25e-16,
    3.763_297_112_699_878_890_06e-20,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_FD: [f64; 10] = [
    7.515_863_983_533_789_471_75e-1,
    1.168_889_258_591_913_821_42e-1,
    6.440_515_265_088_586_110_05e-3,
    1.559_344_091_641_530_208_73e-4,
    1.846_275_673_489_305_458_70e-6,
    1.126_992_247_639_990_352_61e-8,
    3.601_400_295_893_713_704_04e-11,
    5.887_545_336_215_784_100_10e-14,
    4.520_014_340_741_297_014_96e-17,
    1.254_432_370_900_112_643_84e-20,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_GN: [f64; 11] = [
    5.044_420_736_433_832_658_87e-1,
    1.971_028_335_255_234_117_09e-1,
    1.876_485_840_925_752_492_93e-2,
    6.840_793_809_153_930_901_72e-4,
    1.151_388_261_118_842_809_31e-5,
    9.828_524_436_884_222_238_54e-8,
    4.453_444_158_617_501_447_38e-10,
    1.082_680_411_390_208_703_18e-12,
    1.375_554_606_332_617_998_68e-15,
    8.363_544_356_306_774_215_31e-19,
    1.869_587_101_627_832_351_06e-22,
];
#[allow(clippy::excessive_precision)]
const FRESNEL_GD: [f64; 11] = [
    1.474_957_599_251_283_245_29e0,
    3.377_489_891_200_199_704_51e-1,
    2.536_037_414_203_387_951_22e-2,
    8.146_791_071_843_061_790_49e-4,
    1.275_450_756_677_291_187_02e-5,
    1.043_145_896_575_719_905_85e-7,
    4.606_807_281_465_204_282_11e-10,
    1.102_732_150_662_402_707_57e-12,
    1.387_965_312_595_788_712_58e-15,
    8.391_588_162_831_187_073_63e-19,
    1.869_587_101_627_832_363_42e-22,
];

/// Real Fresnel integrals `(S(x), C(x))` for `x >= 0`, ported from cephes
/// `fresnl.c` (the implementation behind `scipy.special.fresnel`).
fn cephes_fresnl(x: f64) -> (f64, f64) {
    let x2 = x * x;
    if x2 < 2.5625 {
        let t = x2 * x2;
        let ss = x * x2 * fresnel_polevl(t, &FRESNEL_SN) / fresnel_p1evl(t, &FRESNEL_SD);
        let cc = x * fresnel_polevl(t, &FRESNEL_CN) / fresnel_polevl(t, &FRESNEL_CD);
        (ss, cc)
    } else {
        // Asymptotic auxiliary functions for x >= 1.6. cephes truncates to
        // (0.5, 0.5) for x > 36974, but scipy keeps the 1/(πx) correction
        // there, so we retain this branch for all large x to match scipy.
        let t = std::f64::consts::PI * x2;
        let u = 1.0 / (t * t);
        let t_inv = 1.0 / t;
        let f = 1.0 - u * fresnel_polevl(u, &FRESNEL_FN) / fresnel_p1evl(u, &FRESNEL_FD);
        let g = t_inv * fresnel_polevl(u, &FRESNEL_GN) / fresnel_p1evl(u, &FRESNEL_GD);

        let t = std::f64::consts::FRAC_PI_2 * x2;
        let c = t.cos();
        let s = t.sin();
        let t = std::f64::consts::PI * x;
        let cc = 0.5 + (f * s - g * c) / t;
        let ss = 0.5 - (f * c + g * s) / t;
        (ss, cc)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Dawson function
// ══════════════════════════════════════════════════════════════════════

/// Dawson function D(x) = exp(-x²) ∫₀ˣ exp(t²) dt.
///
/// Related to the imaginary error function: D(x) = √π/2 * exp(-x²) * erfi(x).
///
/// Uses the Cephes rational approximations that SciPy carried before the XSF
/// Faddeeva rewrite: one branch for [0, 3.25), one for [3.25, 6.25), and one
/// for the tail. This keeps machine-precision parity while avoiding the
/// per-call exponentials in the former Rybicki summation.
fn dawsn_impl(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    let sign = x.signum();
    let ax = x.abs();

    if ax < 3.25 {
        let x2 = ax * ax;
        return sign * ax * dawsn_polevl(x2, &DAWSN_AN) / dawsn_polevl(x2, &DAWSN_AD);
    }

    if ax > 1.0e9 {
        return sign * 0.5 / ax;
    }

    let inv_x2 = 1.0 / (ax * ax);
    let correction = if ax < 6.25 {
        dawsn_polevl(inv_x2, &DAWSN_BN) / dawsn_p1evl(inv_x2, &DAWSN_BD)
    } else {
        dawsn_polevl(inv_x2, &DAWSN_CN) / dawsn_p1evl(inv_x2, &DAWSN_CD)
    };

    sign * 0.5 * (1.0 / ax + inv_x2 * correction / ax)
}

fn dawsn_polevl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
}

fn dawsn_p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = x + coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
}

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_AN: [f64; 10] = [
    1.13681498971755972054e-11,
    8.49262267667473811108e-10,
    1.94434204175553054283e-8,
    9.53151741254484363489e-7,
    3.07828309874913200438e-6,
    3.52513368520288738649e-4,
    -8.50149846724410912031e-4,
    4.22618223005546594270e-2,
    -9.17480371773452345351e-2,
    9.99999999999999994612e-1,
];

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_AD: [f64; 11] = [
    2.40372073066762605484e-11,
    1.48864681368493396752e-9,
    5.21265281010541664570e-8,
    1.27258478273186970203e-6,
    2.32490249820789513991e-5,
    3.25524741826057911661e-4,
    3.48805814657162590916e-3,
    2.79448531198828973716e-2,
    1.58874241960120565368e-1,
    5.74918629489320327824e-1,
    1.00000000000000000539,
];

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_BN: [f64; 11] = [
    5.08955156417900903354e-1,
    -2.44754418142697847934e-1,
    9.41512335303534411857e-2,
    -2.18711255142039025206e-2,
    3.66207612329569181322e-3,
    -4.23209114460388756528e-4,
    3.59641304793896631888e-5,
    -2.14640351719968974225e-6,
    9.10010780076391431042e-8,
    -2.40274520828250956942e-9,
    3.59233385440928410398e-11,
];

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_BD: [f64; 10] = [
    -6.31839869873368190192e-1,
    2.36706788228248691528e-1,
    -5.31806367003223277662e-2,
    8.48041718586295374409e-3,
    -9.47996768486665330168e-4,
    7.81025592944552338085e-5,
    -4.55875153252442634831e-6,
    1.89100358111421846170e-7,
    -4.91324691331920606875e-9,
    7.18466403235734541950e-11,
];

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_CN: [f64; 5] = [
    -5.90592860534773254987e-1,
    6.29235242724368800674e-1,
    -1.72858975380388136411e-1,
    1.64837047825189632310e-2,
    -4.86827613020462700845e-4,
];

#[allow(clippy::excessive_precision)] // Cephes dawsn.c coefficients.
const DAWSN_CD: [f64; 5] = [
    -2.69820057197544900361,
    1.73270799045947845857,
    -3.93708582281939493482e-1,
    3.44278924041233391079e-2,
    -9.73655226040941223894e-4,
];

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

    // Small/moderate x: the all-positive power series is exact. For large x it
    // needs O(x) terms and the previous 80-term cap truncated before convergence
    // (~1e-8 error by x≈100, and unusable past ~120). Switch to the exponential-
    // integral identities Shi=(Ei+E₁)/2, Chi=(Ei−E₁)/2 — no cancellation for
    // x≳1, accurate to ~1e-16 out to x≈700 (where Ei overflows). frankenscipy.
    let (shi, chi) = if ax >= 18.0 {
        let ei = expi_scalar(ax);
        let e1 = expn(1, ax);
        (0.5 * (ei + e1), 0.5 * (ei - e1))
    } else {
        shichi_series(ax)
    };

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
    if x.abs() > 18.0 && v.abs() < x.abs() / 2.0 {
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
    // modstruve_series caps at BESSEL_SERIES_MAX_TERMS (96); its peak term is at
    // k≈x/2, so for x≳190 it truncates BEFORE the dominant terms and grossly
    // underestimates (modstruve(0,300) gave 2.2e119 vs scipy 4.5e128). For large
    // x use the asymptotic L_v(x) = I_v(x) − correction instead (frankenscipy-kjtmn).
    if x.abs() > 40.0 && v.abs() < x.abs() / 2.0 {
        return modstruve_asymptotic(v, x);
    }
    modstruve_series(v, x)
}

/// Modified Struve L_v(x) large-x asymptotic (DLMF 11.6.2):
/// `L_v(x) = I_v(x) − (1/π) Σ_{k≥0} Γ(k+1/2)/Γ(v+1/2−k) (x/2)^{v−2k−1}`.
/// I_v from the (accurate large-x) Bessel routine; the correction is the same
/// asymptotic series as [`struve_asymptotic`] but SUBTRACTED from I_v. The
/// correction is exponentially small relative to I_v for large x, so its
/// asymptotic-truncation error is negligible in L_v.
fn modstruve_asymptotic(v: f64, x: f64) -> f64 {
    let iv = crate::bessel::iv_scalar(v, x);
    let half_x = 0.5 * x;
    let mut term = gamma_fn(0.5) / gamma_fn(v + 0.5) * half_x.powf(v - 1.0);
    let mut sum = term;
    let mut prev_abs = term.abs();
    let two_over_x_sq = (2.0 / x) * (2.0 / x);
    for k in 0..200 {
        let kf = k as f64;
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
    iv - sum / PI
}

/// Integral of the Struve function H_0 from 0 to x.
///
/// Matches `scipy.special.itstruve0`. Because H_0 is odd, SciPy's integral is
/// even in x: integrating from 0 to a negative endpoint returns the same value
/// as integrating to the positive endpoint.
pub fn itstruve0(x: f64) -> f64 {
    // For |x| ≤ 16 integrate the Struve H_0 series term-by-term (closed form,
    // ~1e-11 vs scipy) instead of running a Simpson quadrature that re-evaluates
    // struve() at every node. Beyond 16 the alternating series loses digits to
    // cancellation, so fall back to the (accuracy-preserving) quadrature.
    if x.abs() <= STRUVE_INT_SERIES_MAX {
        struve0_integral_series(x, -1.0)
    } else {
        struve_integral_abs(x, |t| struve(0.0, t))
    }
}

/// Integral of the modified Struve function L_0 from 0 to x.
///
/// Matches `scipy.special.itmodstruve0`; L_0 has the same odd symmetry as H_0
/// for real inputs, so the integral is even in x.
pub fn itmodstruve0(x: f64) -> f64 {
    // Modified-Struve L_0 series has all-positive terms (no cancellation); the
    // same term-by-term integral closed form is used for |x| ≤ 16, quadrature
    // beyond. See [`itstruve0`].
    if x.abs() <= STRUVE_INT_SERIES_MAX {
        struve0_integral_series(x, 1.0)
    } else {
        struve_integral_abs(x, |t| modstruve(0.0, t))
    }
}

/// Integrals of the modified Bessel functions `I₀` and `K₀` from 0 to `x`.
///
/// Returns `(∫₀ˣ I₀(t) dt, ∫₀ˣ K₀(t) dt)`, matching `scipy.special.iti0k0`.
///
/// `∫I₀` is evaluated from its (all-positive) Maclaurin series, exact for every
/// `x` — more accurate than scipy's large-`x` asymptotic. `∫K₀` uses the series
/// for `x ≤ 12` and, beyond that (where the series cancels against the `log`
/// term), the tail asymptotic `∫₀ˣ K₀ = π/2 − √(π/2x)·e^{−x}·Q(1/x)`. For `x < 0`
/// the `I₀` integral is odd (so it negates) and `∫K₀` is NaN (`K₀` is undefined
/// there), matching scipy.
#[must_use]
pub fn iti0k0(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 0.0);
    }
    if x < 0.0 {
        return (-iti0k0(-x).0, f64::NAN);
    }
    const EL: f64 = 0.577_215_664_901_532_9;
    let x24 = x * x / 4.0;

    // ∫₀ˣ I₀ = Σ_m x·(x²/4)^m / ((m!)²·(2m+1)); all terms positive → no cancellation.
    let mut term = x;
    let mut ti = 0.0_f64;
    let mut m = 0usize;
    loop {
        ti += term / (2 * m + 1) as f64;
        m += 1;
        term *= x24 / (m * m) as f64;
        if (term / ((2 * m + 1) as f64) < 1e-18 * ti && (m as f64) > 2.0 * x) || m > 5000 {
            break;
        }
    }

    let tk = if x <= 12.0 {
        // ∫₀ˣ K₀ = −(ln(x/2)+γ)·∫I₀ + Σ_m term_m/(2m+1)² + Σ_{m≥1} term_m·H_m/(2m+1).
        let mut term = x;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut h = 0.0_f64;
        let mut m = 0usize;
        loop {
            let d = (2 * m + 1) as f64;
            s2 += term / (d * d);
            if m >= 1 {
                h += 1.0 / m as f64;
                s3 += term * h / d;
            }
            m += 1;
            term *= x24 / (m * m) as f64;
            if (term.abs() / ((2 * m + 1) as f64) < 1e-18 * s2.abs().max(1.0)
                && (m as f64) > 2.0 * x)
                || m > 5000
            {
                break;
            }
        }
        -((x / 2.0).ln() + EL) * ti + s2 + s3
    } else {
        // Tail asymptotic. Q(1/x) coefficients (q_n = a_n^K − q_{n-1}(n−½)).
        const Q: [f64; 13] = [
            1.0,
            -6.25e-1,
            1.0078125,
            -2.592_773_437_5,
            9.186_859_130_859_375,
            -4.156_797_409_057_617e1,
            2.291_963_589_191_436_8e2,
            -1.491_504_060_477_018_4e3,
            1.119_235_449_557_891e4,
            -9.515_939_374_212_03e4,
            9.041_242_576_904_121e5,
            -9.493_856_041_645_449e6,
            1.091_823_825_694_336e8,
        ];
        let mut qx = 0.0_f64;
        let mut inv = 1.0_f64;
        for &c in &Q {
            qx += c * inv;
            inv /= x;
        }
        std::f64::consts::FRAC_PI_2 - (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp() * qx
    };
    (ti, tk)
}

/// Bessel-integral pair `(∫₀ˣ (I₀(t)−1)/t dt, ∫ₓ^∞ K₀(t)/t dt)`, matching
/// `scipy.special.it2i0k0`.
///
/// The first integral is the all-positive series `Σ_{m≥1} (x²/4)ᵐ/(2m·(m!)²)`.
/// The second is obtained from the ODE `d/dx = −K₀(x)/x` matched to the K₀
/// series, giving `∫ₓ^∞ K₀/t = ½L² + π²/24 + Σ_{m≥1}(pₘL + qₘ)(x²/4)ᵐ` with
/// `L = ln(x/2)+γ`, `pₘ = 1/(2m(m!)²)`, `qₘ = −(Hₘ/(2m) + 1/(4m²))/(m!)²`, for
/// `x ≤ 12`; beyond that (where the series cancels) the tail asymptotic
/// `√(π/2)·x^{−3/2}·e^{−x}·R(1/x)` is used. `x = 0 → (0, 1e300)` (scipy's
/// overflow sentinel for the divergent tail); for `x < 0` the first integral is
/// even and the K₀ tail is NaN. More accurate than scipy at large x.
#[must_use]
pub fn it2i0k0(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1e300);
    }
    if x < 0.0 {
        return (it2i0k0(-x).0, f64::NAN);
    }
    const EL: f64 = 0.577_215_664_901_532_9;
    let u = x * x / 4.0;

    // ∫₀ˣ (I₀−1)/t = Σ_{m≥1} uᵐ/(2m·(m!)²); aₘ = uᵐ/(m!)².
    let mut ii0 = 0.0_f64;
    let mut a = 1.0_f64;
    let mut m = 0usize;
    loop {
        m += 1;
        a *= u / (m * m) as f64;
        ii0 += a / (2 * m) as f64;
        if (a / ((2 * m) as f64) < 1e-20 * ii0.max(1.0) && (m as f64) > 2.0 * x) || m > 800 {
            break;
        }
    }

    let ik0 = if x <= 12.0 {
        let l = (x / 2.0).ln() + EL;
        let mut s = 0.0_f64;
        let mut a = 1.0_f64;
        let mut h = 0.0_f64;
        let mut m = 0usize;
        loop {
            m += 1;
            a *= u / (m * m) as f64;
            h += 1.0 / m as f64;
            let mf = m as f64;
            s += a / (2.0 * mf) * l - (h / (2.0 * mf) + 1.0 / (4.0 * mf * mf)) * a;
            if (a.abs() < 1e-22 * s.abs().max(1.0) && (m as f64) > 2.0 * x) || m > 800 {
                break;
            }
        }
        0.5 * l * l + std::f64::consts::PI * std::f64::consts::PI / 24.0 + s
    } else {
        const R: [f64; 13] = [
            1.0,
            -1.625,
            4.132_812_5,
            -1.453_808_593_75e1,
            6.553_353_881_835_937e1,
            -3.606_615_715_026_855e2,
            2.344_872_716_188_431e3,
            -1.758_827_309_891_581_5e4,
            1.495_063_953_827_857e5,
            -1.420_335_136_666_163_8e6,
            1.491_362_895_213_499e7,
            -1.715_072_842_854_485_2e8,
            2.143_844_091_658_617_3e9,
        ];
        let mut rx = 0.0_f64;
        let mut inv = 1.0_f64;
        for &c in &R {
            rx += c * inv;
            inv /= x;
        }
        (std::f64::consts::PI / 2.0).sqrt() * x.powf(-1.5) * (-x).exp() * rx
    };
    (ii0, ik0)
}

/// Bessel-integral pair `(∫₀ˣ (1−J₀(t))/t dt, ∫ₓ^∞ Y₀(t)/t dt)`, matching
/// `scipy.special.it2j0y0`.
///
/// For `x ≤ 20` both come from the alternating Maclaurin series:
/// `∫(1−J₀)/t = Σ_{m≥1}(−1)^{m+1}(x²/4)ᵐ/(2m(m!)²)` and (via ODE-matching
/// `d/dx[∫ₓ^∞ Y₀/t] = −Y₀(x)/x` against the Y₀ series)
/// `∫ₓ^∞ Y₀/t = −L²/π + π/6 + Σ_{m≥1}(pₘL + qₘ)(x²/4)ᵐ`, with `L = ln(x/2)+γ`,
/// `pₘ = (−1)^{m+1}/(πm(m!)²)`, `qₘ = (−1)ᵐ[Hₘ/(πm) + 1/(2πm²)]/(m!)²`. Beyond
/// `x = 20` (where the alternating series cancels) both switch to oscillatory
/// asymptotics: `∫(1−J₀)/t = γ + ln(x/2) + ∫ₓ^∞ J₀/t`, and the `Y₀`/`J₀` tails
/// are the standard `√(2/πt)` asymptotic integrated by parts. `x = 0` returns
/// scipy's `(0, −1e300)` sentinel; for `x < 0` the first integral is even and
/// the `Y₀` tail is NaN.
#[must_use]
pub fn it2j0y0(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, -1e300);
    }
    if x < 0.0 {
        return (it2j0y0(-x).0, f64::NAN);
    }
    const EL: f64 = 0.577_215_664_901_532_9;
    let pi = std::f64::consts::PI;

    if x <= 20.0 {
        let u = x * x / 4.0;
        let l = (x / 2.0).ln() + EL;
        // ∫(1−J₀)/t alternating series.
        let mut a = 1.0_f64;
        let mut ij0 = 0.0_f64;
        let mut m = 0usize;
        loop {
            m += 1;
            a *= u / (m * m) as f64;
            let sgn = if m % 2 == 1 { 1.0 } else { -1.0 }; // (−1)^{m+1}
            ij0 += sgn * a / (2 * m) as f64;
            if (a.abs() / ((2 * m) as f64) < 1e-22 * ij0.abs().max(1.0) && (m as f64) > 2.5 * x)
                || m > 2000
            {
                break;
            }
        }
        // ∫ₓ^∞ Y₀/t series.
        let mut a = 1.0_f64;
        let mut h = 0.0_f64;
        let mut s = 0.0_f64;
        let mut m = 0usize;
        loop {
            m += 1;
            a *= u / (m * m) as f64;
            h += 1.0 / m as f64;
            let mf = m as f64;
            let alt = if m % 2 == 1 { 1.0 } else { -1.0 }; // (−1)^{m+1}
            let pm = alt / (pi * mf) * a;
            let qm = -alt * (h / (pi * mf) + 1.0 / (2.0 * pi * mf * mf)) * a;
            s += pm * l + qm;
            if (a.abs() < 1e-23 * s.abs().max(1.0) && (m as f64) > 2.5 * x) || m > 2000 {
                break;
            }
        }
        let iy0 = -(l * l) / pi + pi / 6.0 + s;
        (ij0, iy0)
    } else {
        // Oscillatory asymptotics. AC[j] = (−1)^j (2j−1)!!²/(j!·8^j).
        const AC: [f64; 16] = [
            1.0,
            -1.25e-1,
            7.03125e-2,
            -7.32421875e-2,
            1.121_520_996_093_75e-1,
            -2.271_080_017_089_843_8e-1,
            5.725_014_209_747_314e-1,
            -1.727_727_502_584_457_4,
            6.074_042_001_273_483,
            -2.438_052_969_955_606_4e1,
            1.100_171_402_692_467_4e2,
            -5.513_358_961_220_206e2,
            3.038_090_510_922_384_5e3,
            -1.825_775_547_429_317_5e4,
            1.188_384_262_567_832_5e5,
            -8.328_593_040_162_893e5,
        ];
        let phi = x - pi / 4.0;
        let cphi = phi.cos();
        let sphi = phi.sin();
        let poch = |s: f64, j: usize| -> f64 {
            let mut r = 1.0;
            for t in 0..j {
                r *= s + t as f64;
            }
            r
        };
        let a_of = |s: f64| -> f64 {
            (0..8)
                .map(|k| {
                    let sg = if k % 2 == 0 { 1.0 } else { -1.0 };
                    poch(s, 2 * k) * sg * x.powf(-(s + (2 * k) as f64))
                })
                .sum()
        };
        let b_of = |s: f64| -> f64 {
            (0..8)
                .map(|k| {
                    let sg = if k % 2 == 0 { 1.0 } else { -1.0 };
                    poch(s, 2 * k + 1) * sg * x.powf(-(s + (2 * k + 1) as f64))
                })
                .sum()
        };
        let rc = (2.0 / pi).sqrt();
        // ∫ₓ^∞ J₀/t and ∫ₓ^∞ Y₀/t tails.
        let mut jtot = 0.0_f64;
        let mut ytot = 0.0_f64;
        for k in 0..8 {
            let sg = if k % 2 == 0 { 1.0 } else { -1.0 };
            let s_sin = 1.5 + 2.0 * k as f64;
            let s_cos = 2.5 + 2.0 * k as f64;
            let (a_s, b_s) = (a_of(s_sin), b_of(s_sin));
            let (a_c, b_c) = (a_of(s_cos), b_of(s_cos));
            // I_s(s) = cφ·A + sφ·B,  I_c(s) = cφ·B − sφ·A
            let is_s = cphi * a_s + sphi * b_s;
            let ic_s = cphi * b_s - sphi * a_s;
            let is_c = cphi * a_c + sphi * b_c;
            let ic_c = cphi * b_c - sphi * a_c;
            // Y₀ = √(2/πt)[P·sin + Q·cos]
            ytot += sg * AC[2 * k] * is_s + sg * AC[2 * k + 1] * ic_c;
            // J₀ = √(2/πt)[P·cos − Q·sin]
            jtot += sg * AC[2 * k] * ic_s - sg * AC[2 * k + 1] * is_c;
        }
        let ij0 = EL + (x / 2.0).ln() + rc * jtot;
        let iy0 = rc * ytot;
        (ij0, iy0)
    }
}

/// Integrals of the Bessel functions `J₀` and `Y₀` from 0 to `x`.
///
/// Returns `(∫₀ˣ J₀(t) dt, ∫₀ˣ Y₀(t) dt)`, matching `scipy.special.itj0y0`.
///
/// For `x ≤ 20` the integrals come from the (alternating) Maclaurin series:
/// `∫J₀ = Σ_m (−1)ᵐ x·(x²/4)ᵐ/((m!)²·(2m+1))` and
/// `∫Y₀ = (2/π)[(ln(x/2)+γ)·∫J₀ − Σ(−1)ᵐ tₘ/(2m+1)² − Σ_{m≥1}(−1)ᵐ tₘ·Hₘ/(2m+1)]`.
/// Beyond `x = 20` (where the alternating series cancels) it switches to the
/// Zhang-Jin oscillatory asymptotic. For `x < 0` the `J₀` integral is odd (so it
/// negates) and `∫Y₀` is NaN (`Y₀` is undefined there), matching scipy.
#[must_use]
pub fn itj0y0(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 0.0);
    }
    if x < 0.0 {
        return (-itj0y0(-x).0, f64::NAN);
    }
    const EL: f64 = 0.577_215_664_901_532_9;
    let pi = std::f64::consts::PI;

    if x <= 20.0 {
        let x24 = x * x / 4.0;
        let mut term = x;
        let mut tj = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut h = 0.0_f64;
        let mut m = 0usize;
        let mut sign = 1.0_f64;
        loop {
            let d = (2 * m + 1) as f64;
            tj += sign * term / d;
            s2 += sign * term / (d * d);
            if m >= 1 {
                h += 1.0 / m as f64;
                s3 += sign * term * h / d;
            }
            m += 1;
            sign = -sign;
            term *= x24 / (m * m) as f64;
            if (term.abs() / ((2 * m + 1) as f64) < 1e-20 * tj.abs().max(1.0)
                && (m as f64) > 2.0 * x)
                || m > 3000
            {
                break;
            }
        }
        let ty = (2.0 / pi) * (((x / 2.0).ln() + EL) * tj - s2 - s3);
        (tj, ty)
    } else {
        // Zhang-Jin oscillatory asymptotic.
        let mut a = [0.0_f64; 19];
        let (mut a0, mut a1) = (1.0_f64, 5.0 / 8.0);
        a[1] = a1;
        for k in 1..=16 {
            let kf = k as f64;
            let af = (1.5 * (kf + 0.5) * (kf + 5.0 / 6.0) * a1
                - 0.5 * (kf + 0.5) * (kf + 0.5) * (kf - 0.5) * a0)
                / (kf + 1.0);
            a[k + 1] = af;
            a0 = a1;
            a1 = af;
        }
        let mut bf = 1.0_f64;
        let mut r = 1.0_f64;
        for k in 1..=8 {
            r = -r / (x * x);
            bf += a[2 * k] * r;
        }
        let mut bg = a[1] / x;
        r = 1.0 / x;
        for k in 1..=8 {
            r = -r / (x * x);
            bg += a[2 * k + 1] * r;
        }
        let xp = x + 0.25 * pi;
        let rc = (2.0 / (pi * x)).sqrt();
        let tj = 1.0 - rc * (bf * xp.cos() + bg * xp.sin());
        let ty = rc * (bg * xp.cos() - bf * xp.sin());
        (tj, ty)
    }
}

/// Integrals of the Airy functions from 0 to `x`, matching `scipy.special.itairy`.
///
/// Returns `(Apt, Bpt, Ant, Bnt)` where
/// `Apt = ∫₀ˣ Ai(t) dt`, `Bpt = ∫₀ˣ Bi(t) dt`,
/// `Ant = ∫₀ˣ Ai(−t) dt`, `Bnt = ∫₀ˣ Bi(−t) dt`.
///
/// Evaluated by integrating the Airy Maclaurin series `Ai = c₁f − c₂g`,
/// `Bi = √3(c₁f + c₂g)` term-by-term, where `∫f = Σ_k a_k x^{3k+1}/(3k+1)!`,
/// `∫g = Σ_k b_k x^{3k+2}/(3k+2)!` (and the sign-alternated versions give the
/// negative-axis integrals). `c₁ = Ai(0)`, `c₂ = −Ai'(0)`. For `x < 0` the
/// integrals reflect: `itairy(−x) = (−Ant, −Bnt, −Apt, −Bpt)`.
///
/// This is accurate to ≈1e-9 (often full f64) for `|x| ≲ 10` — far more accurate
/// than scipy's specfun ITAIRY, which carries ~1e-5 error already by `x = 5`.
/// The cancellation in `Apt/Ant/Bnt` (which tend to finite limits while `f, g`
/// grow) erodes precision for larger `|x|`, as it does in scipy.
#[must_use]
pub fn itairy(x: f64) -> (f64, f64, f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    if x < 0.0 {
        let (apt, bpt, ant, bnt) = itairy(-x);
        return (-ant, -bnt, -apt, -bpt);
    }
    const C1: f64 = 0.355_028_053_887_817_2; // Ai(0)
    const C2: f64 = 0.258_819_403_792_806_8; // -Ai'(0)
    let sq3 = 3.0_f64.sqrt();
    let x3 = x * x * x;

    let (mut f, mut fa, mut g, mut ga) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut tf = x; // ∫f term, k=0
    let mut tg = x * x / 2.0; // ∫g term, k=0
    let mut k = 0usize;
    let mut sign = 1.0_f64;
    loop {
        f += tf;
        fa += sign * tf;
        g += tg;
        ga += sign * tg;
        let kf = k as f64;
        tf *= (3.0 * kf + 1.0) * x3 / ((3.0 * kf + 2.0) * (3.0 * kf + 3.0) * (3.0 * kf + 4.0));
        tg *= (3.0 * kf + 2.0) * x3 / ((3.0 * kf + 3.0) * (3.0 * kf + 4.0) * (3.0 * kf + 5.0));
        k += 1;
        sign = -sign;
        if (tf.abs() < 1e-20 * f.abs().max(1.0)
            && tg.abs() < 1e-20 * g.abs().max(1.0)
            && (k as f64) > 2.0 * x)
            || k > 400
        {
            break;
        }
    }
    let apt = C1 * f - C2 * g;
    let bpt = sq3 * (C1 * f + C2 * g);
    let ant = C1 * fa + C2 * ga;
    let bnt = sq3 * (C1 * fa - C2 * ga);
    (apt, bpt, ant, bnt)
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

    let correction = if x.abs() <= STRUVE_INT_SERIES_MAX {
        struve0_over_t_integral_series(x.abs())
    } else {
        struve_integral_abs(x, struve0_over_t)
    };
    if x.is_sign_negative() {
        std::f64::consts::FRAC_PI_2 + correction
    } else {
        std::f64::consts::FRAC_PI_2 - correction
    }
}

/// Above this |x| the alternating Struve-integral series loses too many digits
/// to cancellation, so the scalar integrals fall back to Simpson quadrature.
const STRUVE_INT_SERIES_MAX: f64 = 16.0;

/// `∫₀ˣ C_0(t) dt` where `C_0 = H_0` (`sgn = -1`, alternating) or `C_0 = L_0`
/// (`sgn = +1`, modified). Integrates the Struve/modified-Struve series
/// term-by-term: `Σ sgnᵏ · x^{2k+2} / (Γ(k+3/2)² · 2^{2k+1} · (2k+2))`, which is
/// even in `x`. Matches scipy to ~1e-11 for `|x| ≤ 16`. Zero for `x == 0`.
fn struve0_integral_series(x: f64, sgn: f64) -> f64 {
    let ax = x.abs();
    let x2 = ax * ax;
    let mut term = x2 / PI; // k = 0 term (Γ(3/2)² = π/4)
    let mut sum = term;
    let mut k = 0usize;
    loop {
        let kf = k as f64;
        // tₖ₊₁/tₖ = sgn · x²/(4(k+3/2)²) · (2k+2)/(2k+4)
        term *= sgn * x2 / (4.0 * (kf + 1.5) * (kf + 1.5)) * (2.0 * kf + 2.0) / (2.0 * kf + 4.0);
        sum += term;
        k += 1;
        if (term.abs() < 1e-18 * sum.abs().max(1.0) && kf > ax) || k > 2000 {
            break;
        }
    }
    sum
}

/// `∫₀ˣ H_0(t)/t dt` via term-by-term integration of the `H_0/t` series:
/// `Σ (-1)ᵏ · x^{2k+1} / (2^{2k+1} · Γ(k+3/2)² · (2k+1))`. `x` is `|x|`; the
/// caller applies scipy's `π/2 ∓ correction` sign convention. ~1e-10 for `x ≤ 16`.
fn struve0_over_t_integral_series(ax: f64) -> f64 {
    let x2 = ax * ax;
    let mut term = 2.0 * ax / PI; // k = 0 term
    let mut sum = term;
    let mut k = 0usize;
    loop {
        let kf = k as f64;
        // tₖ₊₁/tₖ = -x²/(4(k+3/2)²) · (2k+1)/(2k+3)
        term *= -x2 / (4.0 * (kf + 1.5) * (kf + 1.5)) * (2.0 * kf + 1.0) / (2.0 * kf + 3.0);
        sum += term;
        k += 1;
        if (term.abs() < 1e-18 * sum.abs().max(1.0) && kf > ax) || k > 2000 {
            break;
        }
    }
    sum
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

    // The Struve integrands oscillate with period ~2π (or, for L₀, rise smoothly),
    // so the old 256·|x| Simpson density was ~1600 points per oscillation — wildly
    // over-resolved. 64·|x| is ~400/oscillation: worst quadrature error vs
    // high-precision (mpmath) ground truth is 7.4e-11 for itstruve0 and 3.3e-10 for
    // itmodstruve0 over x∈[16,127], while cutting the per-node struve()/modstruve()
    // evaluations ~4×. (This path is already correct where SciPy's own itstruve0
    // is inaccurate at large x — e.g. itstruve0(50)=3.2445, SciPy returns 6.30.)
    let raw_steps = (64.0 * upper.max(1.0)).ceil() as usize;
    let steps = raw_steps.clamp(64, 32_768);
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
    map_real_binary_eager("stirling2", n_tensor, k_tensor, mode, |n, k| {
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
            let zeta_val = crate::gamma::zeta_scalar(nf);
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

/// Evaluate `f(0..n)` into a `Vec<f64>`, parallel over index chunks for large `n`.
/// Convenience-module kernels (ndtr/ndtri normal CDF/quantile, spence, kolmogorov, the ML
/// activations, ...) are non-trivial per element and each index writes its own slot, so
/// chunking across cores and concatenating in index order is bit-identical to
/// `(0..n).map(f).collect()` — including returning the first failing index's error in order.
fn par_map_indices<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 128)
            .max(1)
    };
    if nthreads <= 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("convenience array worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Above this length the cheap-but-COMPUTE-BOUND convenience kernels (e.g. `expit`'s
/// `exp`+reciprocal, ~8 ns/call) parallelize via [`par_map_light`]. The earlier
/// blanket-serial rule blamed "~40 ns/element of overhead", but that was the loose
/// `n/128` worker cap over-subscribing a ~8 ns kernel (64 OS threads, a flat spawn
/// floor). A WORK cap (≥~32k elements/worker) amortizes the spawn, flipping these
/// from a ~1.8x cephes loss to a win from ~128k up. Memory-bound kernels see no
/// benefit, so only the compute-bound ones (expit/logit/…) are routed here.
const LIGHT_KERNEL_PAR_MIN: usize = 1 << 18;

/// Work-capped parallel map for cheap compute-bound kernels — caps workers at
/// `min(cores, n/32768)` so each owns enough elements to amortize the OS-thread
/// spawn (the loose `n/128` cap of [`par_map_indices`] over-subscribes them).
/// Order-preserving → byte-identical to the serial map.
fn par_map_light<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 32768)
            .max(1)
    };
    if nthreads <= 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("convenience light worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Work-capped parallel map for MODERATE real kernels (~50-300 ns/elt:
/// erfcx/dawsn/erfi/spence/wrightomega). Caps workers at `min(cores, n/8192)` —
/// fewer elements/worker than [`par_map_light`] (these kernels are heavier so a
/// smaller chunk still amortizes the spawn) but far fewer threads than the loose
/// `n/128` of [`par_map_indices`], which over-subscribes ~64 OS threads onto a
/// sub-µs kernel and leaves a flat ~3 ms spawn floor (measured on `dawsn`:
/// constant ~2.8-3.3 ms across n=50k-500k). Order-preserving → byte-identical.
fn par_map_moderate<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 8192)
            .max(1)
    };
    if nthreads <= 1 {
        return (0..n).map(&f).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("convenience moderate worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Like [`map_real`] but parallelizes a real array of length ≥ [`LIGHT_KERNEL_PAR_MIN`]
/// through the work-capped [`par_map_light`]. For COMPUTE-bound cheap kernels only.
fn map_real_light<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    if let SpecialTensor::RealVec(values) = input
        && values.len() >= LIGHT_KERNEL_PAR_MIN
    {
        return par_map_light(values.len(), |i| kernel(values[i])).map(SpecialTensor::RealVec);
    }
    map_real_inner(function, input, mode, kernel, false)
}

/// Default-SERIAL elementwise map for convenience scalar functions. Nearly all of them are cheap
/// O(1) kernels (expit/logit/ndtr/relu/silu/… ~14-30ns/call); par_map_indices adds ~40ns/element of
/// overhead, so parallelizing them is a net LOSS at any practical length. The few genuinely heavy
/// kernels (kolmogorov/kolmogi series) use [`map_real_par`] instead.
fn map_real<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    map_real_inner(function, input, mode, kernel, false)
}

/// Parallel variant of [`map_real`] for heavy per-element kernels (kolmogorov/kolmogi).
fn map_real_par<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    map_real_inner(function, input, mode, kernel, true)
}

/// Work-gated variant of [`map_real`]: serial for short/medium arrays (the cheap-kernel
/// sweep proved `par_map_indices` over-subscribes them) but parallel at/above 1<<20, where a
/// mid-cost kernel (e.g. ndtr ~23ns, computed via erfc) finally dominates the thread-spawn +
/// concat overhead. Same-worker measurement flips ndtr from a slight SciPy loss to a multicore
/// win at n>=2M. Order-preserving, so byte-identical to the serial map either way.
fn map_real_wg<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    let parallel = matches!(input, SpecialTensor::RealVec(v) if v.len() >= 1 << 20);
    map_real_inner(function, input, mode, kernel, parallel)
}

fn map_real_inner<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
    parallel: bool,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            if parallel {
                par_map_indices(values.len(), |i| kernel(values[i])).map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| kernel(x))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
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
    real_par_min: usize,
    real_kernel: F,
    complex_kernel: G,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
    G: Fn(Complex64) -> Result<Complex64, SpecialError> + Sync,
{
    match input {
        SpecialTensor::RealScalar(x) => real_kernel(*x).map(SpecialTensor::RealScalar),
        // Moderate real kernels (erfcx/erfi/dawsn/spence/wrightomega, ~50-300ns/elt)
        // have per-fn break-evens far below the cheap-kernel 1<<20 but well above the
        // raw n/32 par_map_indices gate, which over-subscribes ~16 threads onto a
        // sub-µs kernel and pessimizes 2-14x at n=4096 (measured, NEGATIVE_EVIDENCE
        // 2026-06-22). Gate the real-array path on each caller's measured break-even.
        // Order-preserving ⇒ byte-identical either way. The COMPLEX kernels are much
        // heavier (Faddeeva/series), so their break-even is tiny — left eager.
        SpecialTensor::RealVec(values) => {
            par_map_indices_gated(values.len(), real_par_min, |i| real_kernel(values[i]))
                .map(SpecialTensor::RealVec)
        }
        SpecialTensor::ComplexScalar(value) => {
            complex_kernel(*value).map(SpecialTensor::ComplexScalar)
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| complex_kernel(values[i]))
                .map(SpecialTensor::ComplexVec)
        }
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

/// `par_map_indices` with a work-gate: stay serial (no thread spawn, no per-chunk
/// Vec alloc/concat) until the array is large enough to amortise par_map_indices'
/// overhead. The established break-even for this helper is ~1<<20 even for ~25-30ns
/// kernels (see GAMMA_FAMILY_PAR_MIN and the error.rs erf/erfc gate). Order-preserving
/// ⇒ byte-identical to the ungated call either way.
fn par_map_indices_gated<T, H>(n: usize, real_par_min: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    if n >= real_par_min {
        par_map_moderate(n, f)
    } else {
        (0..n).map(f).collect()
    }
}

/// Eager binary dispatch (real-array path always through `par_map_indices`). For
/// EXPENSIVE kernels (gammaincinv/gammainccinv/owens_t/stirling2 — µs-scale per call)
/// where parallelism amortises well below 1<<20.
fn map_real_binary_eager<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    map_real_binary_gated(function, lhs, rhs, mode, 0, kernel)
}

/// Binary dispatch for CHEAP kernels (≤ ~30ns/call: xlogy/boxcox/powm1/huber/…).
/// Real-array path is work-gated at 1<<20 so the well-vectorized serial map is used
/// until the array is huge — the ungated n/32 par_map_indices over-subscribed ~16
/// threads onto a ~4ns/elt kernel, measured 63x slower at n=4096 (frankenscipy: cc).
fn map_real_binary<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    map_real_binary_gated(function, lhs, rhs, mode, 1 << 20, kernel)
}

fn map_real_binary_gated<F>(
    function: &'static str,
    lhs: &SpecialTensor,
    rhs: &SpecialTensor,
    mode: RuntimeMode,
    real_par_min: usize,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64, f64) -> Result<f64, SpecialError> + Sync,
{
    match (lhs, rhs) {
        (SpecialTensor::RealScalar(left), SpecialTensor::RealScalar(right)) => {
            kernel(*left, *right).map(SpecialTensor::RealScalar)
        }
        (SpecialTensor::RealVec(left), SpecialTensor::RealScalar(right)) => {
            let right = *right;
            par_map_indices_gated(left.len(), real_par_min, |i| kernel(left[i], right))
                .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(left), SpecialTensor::RealVec(right)) => {
            let left = *left;
            par_map_indices_gated(right.len(), real_par_min, |i| kernel(left, right[i]))
                .map(SpecialTensor::RealVec)
        }
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
            par_map_indices_gated(left.len(), real_par_min, |i| kernel(left[i], right[i]))
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
    F: Fn(f64, f64, f64) -> Result<f64, SpecialError> + Sync,
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

    par_map_indices(target_len, |idx| {
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
/// Crossover where the Kelvin K-functions switch from the ascending log/harmonic
/// series to the K₀/K₁ asymptotic. The series cancels ~ε·e^{x√2} (ber/bei grow
/// while ker/kei decay) and the dominant-solution asymptotic floors at ~e^{-2x};
/// the two cross near x≈10, so 10 minimizes the worst-case error (at 11 the
/// series region [10,11] was ~1e-8 off scipy/mpmath vs the asymptotic's ~2e-9).
/// frankenscipy.
const KELVIN_ASYMP_X: f64 = 10.0;

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
    let k1 = if x < KELVIN_ASYMP_X {
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
    if x >= KELVIN_ASYMP_X {
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
    if x >= KELVIN_ASYMP_X {
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

/// Vectorized Kelvin functions over many arguments. fsci exposed these only as
/// scalars; SciPy's ber/bei/ker/kei ufuncs are ~640-710 ns/point (1.3 s for 2M),
/// while fsci's scalar kernels are ~33 ns. Fanning them across cores via the
/// crate's order-preserving parallel map is a ~150-250x win and fills the missing
/// vectorized API. Each `*_many` is bit-identical to a serial map over its scalar.
#[must_use]
pub fn ber_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(ber(x[i]))).expect("ber is infallible")
}
/// Vectorized Kelvin bei; see [`ber_many`].
#[must_use]
pub fn bei_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(bei(x[i]))).expect("bei is infallible")
}
/// Vectorized Kelvin ker; see [`ber_many`].
#[must_use]
pub fn ker_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(ker(x[i]))).expect("ker is infallible")
}
/// Vectorized Kelvin kei; see [`ber_many`].
#[must_use]
pub fn kei_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(kei(x[i]))).expect("kei is infallible")
}
/// Vectorized Kelvin berp (d/dx ber); see [`ber_many`].
#[must_use]
pub fn berp_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(berp(x[i]))).expect("berp is infallible")
}
/// Vectorized Kelvin beip (d/dx bei); see [`ber_many`].
#[must_use]
pub fn beip_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(beip(x[i]))).expect("beip is infallible")
}
/// Vectorized Kelvin kerp (d/dx ker); see [`ber_many`].
#[must_use]
pub fn kerp_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(kerp(x[i]))).expect("kerp is infallible")
}
/// Vectorized Kelvin keip (d/dx kei); see [`ber_many`].
#[must_use]
pub fn keip_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(keip(x[i]))).expect("keip is infallible")
}

/// Vectorized sine/cosine and Bessel/Airy integral special functions over many
/// arguments — fsci exposed these only as scalars while SciPy's ufuncs are slow
/// (iti0k0/itj0y0 ~580-601 ms, expn ~440, shichi ~288, fresnel ~165, sici/poch
/// ~90 for 2M points). Each fans the fast scalar kernel across cores via the
/// crate's order-preserving parallel map and is bit-identical to a serial map.
/// (itairy is intentionally NOT wrapped: its scalar disagrees with SciPy.)
#[must_use]
pub fn fresnel_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(fresnel(x[i])))
        .expect("fresnel is infallible")
}
/// Vectorized sine/cosine integrals (Si(x), Ci(x)); see [`fresnel_many`].
#[must_use]
pub fn sici_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(sici(x[i])))
        .expect("sici is infallible")
}
/// Vectorized hyperbolic sine/cosine integrals (Shi(x), Chi(x)); see [`fresnel_many`].
#[must_use]
pub fn shichi_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(shichi(x[i])))
        .expect("shichi is infallible")
}
/// Vectorized integrals of I_0/K_0 (∫₀ˣ I_0, ∫₀ˣ K_0); see [`fresnel_many`].
#[must_use]
pub fn iti0k0_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(iti0k0(x[i])))
        .expect("iti0k0 is infallible")
}
/// Vectorized integrals of J_0/Y_0 (∫₀ˣ J_0, ∫₀ˣ Y_0); see [`fresnel_many`].
#[must_use]
pub fn itj0y0_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(itj0y0(x[i])))
        .expect("itj0y0 is infallible")
}
/// Vectorized second integrals of I_0/K_0 (∫₀ˣ (I_0−1)/t, ∫ₓ^∞ K_0/t); see
/// [`fresnel_many`]. Matches a serial map of [`it2i0k0`] bit-for-bit.
#[must_use]
pub fn it2i0k0_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(it2i0k0(x[i])))
        .expect("it2i0k0 is infallible")
}
/// Vectorized second integrals of J_0/Y_0 (∫₀ˣ (1−J_0)/t, ∫ₓ^∞ Y_0/t); see
/// [`fresnel_many`]. Matches a serial map of [`it2j0y0`] bit-for-bit.
#[must_use]
pub fn it2j0y0_many(x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| Ok::<(f64, f64), SpecialError>(it2j0y0(x[i])))
        .expect("it2j0y0 is infallible")
}
/// Vectorized integrals of Airy functions `(∫₀ˣAi, ∫₀ˣBi, ∫₋∞..0 Ai-pair)`; see
/// [`fresnel_many`]. Bit-identical to a serial map of [`itairy`]. SciPy's specfun
/// `itairy` is looped ~1.8µs/call, so the parallel fan is a large gapfill win.
#[must_use]
pub fn itairy_many(x: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64, f64, f64), SpecialError>(itairy(x[i]))
    })
    .expect("itairy is infallible")
}
/// Vectorized parabolic cylinder `D_v(x)` and its derivative for fixed order `v`;
/// see [`fresnel_many`]. Bit-identical to a serial map of [`pbdv`](crate::pbdv).
#[must_use]
pub fn pbdv_many(v: f64, x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::pbdv(v, x[i]))
    })
    .expect("pbdv is infallible")
}
/// Vectorized parabolic cylinder `V_v(x)` and its derivative for fixed order `v`;
/// see [`fresnel_many`]. Bit-identical to a serial map of [`pbvv`](crate::pbvv).
#[must_use]
pub fn pbvv_many(v: f64, x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::pbvv(v, x[i]))
    })
    .expect("pbvv is infallible")
}
/// Vectorized parabolic cylinder `W(a, x)` and its derivative for fixed parameter
/// `a`; see [`fresnel_many`]. Bit-identical to a serial map of [`pbwa`](crate::pbwa).
#[must_use]
pub fn pbwa_many(a: f64, x: &[f64]) -> Vec<(f64, f64)> {
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::pbwa(a, x[i]))
    })
    .expect("pbwa is infallible")
}
/// Vectorized generalized exponential integral E_n(x) for fixed order `n`; see [`fresnel_many`].
#[must_use]
pub fn expn_many(n: usize, x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(expn(n, x[i])))
        .expect("expn is infallible")
}
/// Vectorized Pochhammer symbol (x)_a for fixed `a`; see [`fresnel_many`].
#[must_use]
pub fn poch_many(x: &[f64], a: f64) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(poch(x[i], a)))
        .expect("poch is infallible")
}
/// Vectorized integral of the Struve function H_0, `∫₀ˣ H_0(t) dt`; see [`fresnel_many`].
#[must_use]
pub fn itstruve0_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(itstruve0(x[i])))
        .expect("itstruve0 is infallible")
}
/// Vectorized integral of the modified Struve function L_0, `∫₀ˣ L_0(t) dt`; see [`fresnel_many`].
#[must_use]
pub fn itmodstruve0_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(itmodstruve0(x[i])))
        .expect("itmodstruve0 is infallible")
}
/// Vectorized second integral of the Struve function H_0, `∫₀ˣ (H_0(t)/t) dt`; see [`fresnel_many`].
#[must_use]
pub fn it2struve0_many(x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(it2struve0(x[i])))
        .expect("it2struve0 is infallible")
}
/// Vectorized weighted Bessel integral `besselpoly(a, λ, ν)` over many scale
/// factors `a` for a fixed `(λ, ν)`; see [`fresnel_many`]. fsci's convergent
/// series is already ~2.9x faster than SciPy's cephes ufunc even serial, so the
/// order-preserving parallel fan is a large win. Bit-identical to a serial map.
#[must_use]
pub fn besselpoly_many(a: &[f64], lambda: f64, nu: f64) -> Vec<f64> {
    par_map_indices(a.len(), |i| {
        Ok::<f64, SpecialError>(besselpoly(a[i], lambda, nu))
    })
    .expect("besselpoly is infallible")
}
/// Vectorized prolate-spheroidal characteristic value `pro_cv(m, n, c)` over many
/// `c` for a fixed `(m, n)`; see [`fresnel_many`]. Each `pro_cv` diagonalises a
/// ~50–110-wide tridiagonal, so SciPy's specfun ufunc is very slow (~9 µs/pt);
/// the order-preserving parallel fan of the (bit-identical) scalar wins large.
#[must_use]
pub fn pro_cv_many(m: u32, n: u32, c: &[f64]) -> Vec<f64> {
    par_map_indices(c.len(), |i| {
        Ok::<f64, SpecialError>(crate::orthopoly::pro_cv(m, n, c[i]))
    })
    .expect("pro_cv is infallible")
}
/// Vectorized oblate-spheroidal characteristic value `obl_cv(m, n, c)` over many
/// `c` for a fixed `(m, n)`; see [`pro_cv_many`].
#[must_use]
pub fn obl_cv_many(m: u32, n: u32, c: &[f64]) -> Vec<f64> {
    par_map_indices(c.len(), |i| {
        Ok::<f64, SpecialError>(crate::orthopoly::obl_cv(m, n, c[i]))
    })
    .expect("obl_cv is infallible")
}

/// Vectorized prolate-spheroidal angular function `pro_ang1(m, n, c, x)` (value
/// and derivative) over many `x` at a fixed `(m, n, c)`. The characteristic value
/// and expansion coefficients (the expensive tridiagonal solve, ~50–110 wide) are
/// INVARIANT in `x`, so they are computed ONCE and only the cheap Legendre series
/// is evaluated per `x` — whereas SciPy's `pro_ang1` ufunc re-solves the
/// eigenproblem for every element. Each entry equals [`pro_ang1`](crate::pro_ang1).
#[must_use]
pub fn pro_ang1_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    crate::orthopoly::spheroidal_ang1_many(m, n, c, x, true)
}

/// Vectorized oblate-spheroidal angular function `obl_ang1(m, n, c, x)` over many
/// `x` at a fixed `(m, n, c)`; see [`pro_ang1_many`].
#[must_use]
pub fn obl_ang1_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    crate::orthopoly::spheroidal_ang1_many(m, n, c, x, false)
}

/// Vectorized oblate-spheroidal radial function of the first kind `obl_rad1(m, n, c, x)`
/// (value and derivative) over many `x` at a fixed `(m, n, c)`; see [`pro_ang1_many`].
/// SciPy's specfun `obl_rad1` ufunc re-solves the spheroidal eigenproblem for every
/// element (~13µs/call), so a parallel fan over fsci's faster kernel is a large win.
/// Bit-identical to a serial map of [`obl_rad1`](crate::obl_rad1).
#[must_use]
pub fn obl_rad1_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    // The characteristic value `cv` is x-INVARIANT (the expensive spheroidal
    // eigenproblem). SciPy's ufunc — and a naive `obl_rad1` par-map — re-solve it for
    // every element; hoist it out of the loop and fan only the cheap per-x radial
    // series across threads. `obl_cv == spheroidal_cv(m,n,c,false)`, exactly the cv
    // `obl_rad1` computes internally, so this is bit-identical to a serial map.
    let cv = crate::orthopoly::obl_cv(m, n, c);
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::orthopoly::obl_rad1_cv(m, n, c, cv, x[i]))
    })
    .expect("obl_rad1_cv is infallible")
}

/// Vectorized oblate-spheroidal radial function of the second kind `obl_rad2(m, n, c, x)`
/// (value and derivative) over many `x` at a fixed `(m, n, c)`; see [`obl_rad1_many`].
/// SciPy's specfun `obl_rad2` is ~17µs/call. Bit-identical to a serial map of
/// [`obl_rad2`](crate::obl_rad2).
#[must_use]
pub fn obl_rad2_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    // See [`obl_rad1_many`]: hoist the x-invariant characteristic value out of the
    // per-element loop. Bit-identical to a serial map of [`obl_rad2`](crate::obl_rad2)
    // (`obl_rad2_cv` shares its `n<m` guard and driver).
    let cv = crate::orthopoly::obl_cv(m, n, c);
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::orthopoly::obl_rad2_cv(m, n, c, cv, x[i]))
    })
    .expect("obl_rad2_cv is infallible")
}

/// Vectorized prolate-spheroidal radial function of the first kind `pro_rad1(m, n, c, x)`
/// (value and derivative) over many `x` at a fixed `(m, n, c)`; the prolate sibling of
/// [`obl_rad1_many`]. SciPy's specfun `pro_rad1` ufunc re-solves the spheroidal
/// eigenproblem for every element, so a parallel fan over fsci's faster kernel — with the
/// x-invariant characteristic value hoisted out — is a large win. Bit-identical to a
/// serial map of [`pro_rad1`](crate::pro_rad1).
#[must_use]
pub fn pro_rad1_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    // `pro_cv == spheroidal_cv(m,n,c,true)` — exactly the cv `pro_rad1` computes
    // internally — is x-INVARIANT; hoist it out of the per-element loop and fan only the
    // cheap per-x radial series across threads (bit-identical to a serial `pro_rad1` map).
    let cv = crate::orthopoly::pro_cv(m, n, c);
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::orthopoly::pro_rad1_cv(m, n, c, cv, x[i]))
    })
    .expect("pro_rad1_cv is infallible")
}

/// Vectorized prolate-spheroidal radial function of the second kind `pro_rad2(m, n, c, x)`
/// (value and derivative) over many `x` at a fixed `(m, n, c)`; see [`pro_rad1_many`].
/// Bit-identical to a serial map of [`pro_rad2`](crate::pro_rad2) (`pro_rad2_cv` shares its
/// `n<m` guard and driver).
#[must_use]
pub fn pro_rad2_many(m: u32, n: u32, c: f64, x: &[f64]) -> Vec<(f64, f64)> {
    let cv = crate::orthopoly::pro_cv(m, n, c);
    par_map_indices(x.len(), |i| {
        Ok::<(f64, f64), SpecialError>(crate::orthopoly::pro_rad2_cv(m, n, c, cv, x[i]))
    })
    .expect("pro_rad2_cv is infallible")
}

/// Vectorized even periodic Mathieu function `mathieu_cem(m, q, x)` (value and
/// derivative) over many `x` (degrees) at a fixed `(m, q)`. The Fourier
/// coefficients (an x-invariant matrix solve) are computed ONCE and the cheap
/// cosine series is fanned across threads; SciPy's ufunc recomputes them per
/// element. Each entry equals [`mathieu_cem`](crate::mathieu_cem). (Large-`q`
/// values inherit the scalar's characteristic-value labeling — a pre-existing
/// concern, not introduced here.)
#[must_use]
pub fn mathieu_cem_many(m: u32, q: f64, x: &[f64]) -> Vec<(f64, f64)> {
    crate::orthopoly::mathieu_series_many(m, q, x, true)
}

/// Vectorized odd periodic Mathieu function `mathieu_sem(m, q, x)` over many `x`
/// (degrees) at a fixed `(m, q)`; see [`mathieu_cem_many`].
#[must_use]
pub fn mathieu_sem_many(m: u32, q: f64, x: &[f64]) -> Vec<(f64, f64)> {
    crate::orthopoly::mathieu_series_many(m, q, x, false)
}

/// Weighted integral of the Bessel function of the first kind,
/// `besselpoly(a, λ, ν) = ∫₀¹ xˡ Jᵥ(2·a·x) dx`.
///
/// Matches `scipy.special.besselpoly` (cephes `besselpoly`). Evaluated by the
/// term-by-term integration of the `Jᵥ` series:
///
/// ```text
///   besselpoly(a, λ, ν) = Σ_{m≥0} (−1)ᵐ a^{2m+ν}
///                                  / (m! · Γ(ν+m+1) · (λ+2m+ν+1))
/// ```
///
/// Special cases mirror cephes: `a = 0` gives `1/(λ+1)` for `ν = 0` and `0`
/// otherwise; a negative-integer `ν` is folded to `−ν` with an overall sign
/// `(−1)^ν` (Γ(ν+1) is a pole there). For negative non-integer `ν` the signed
/// Γ keeps the correct sign — hence `gammasgn·exp(gammaln)`, not `exp(gammaln)`.
#[must_use]
pub fn besselpoly(a: f64, lambda: f64, nu: f64) -> f64 {
    if a.is_nan() || lambda.is_nan() || nu.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 {
        return if nu == 0.0 { 1.0 / (lambda + 1.0) } else { 0.0 };
    }
    // Negative-integer ν: reflect to −ν and carry the (−1)^ν sign.
    let mut nu = nu;
    let mut factor = false;
    if nu < 0.0 && nu.floor() == nu {
        nu = -nu;
        factor = (nu as i64) % 2 != 0;
    }

    let mode = fsci_runtime::RuntimeMode::Strict;
    let g = nu + 1.0;
    let signed_gamma = crate::gammasgn_scalar(g, mode).unwrap_or(f64::NAN)
        * crate::gammaln_scalar(g, mode).unwrap_or(f64::NAN).exp();

    const EPS: f64 = 1e-17;
    let mut sm = (nu * a.ln()).exp() / (signed_gamma * (lambda + nu + 1.0));
    let mut sum = 0.0_f64;
    for m in 0..1000 {
        sum += sm;
        let mf = m as f64;
        sm *= -a * a * (lambda + nu + 1.0 + 2.0 * mf)
            / ((nu + mf + 1.0) * (mf + 1.0) * (lambda + nu + 3.0 + 2.0 * mf));
        if sm.abs() <= EPS * sum.abs() {
            break;
        }
    }
    if factor { -sum } else { sum }
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

/// First `nt` positive real zeros of a Kelvin function `f` (or its derivative).
///
/// All Kelvin functions are smooth and oscillatory on `x > 0` with simple zeros
/// spaced asymptotically `π√2 ≈ 4.443` apart, so a forward sign-change scan
/// (step `0.25`, far finer than the spacing) brackets every zero in order, and
/// bisection refines each to machine precision. Robust and reference-free; the
/// zeros are well-defined constants, so the result matches
/// `scipy.special.*_zeros` independently of any seed table.
fn kelvin_zeros_of(f: impl Fn(f64) -> f64, nt: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(nt as usize);
    if nt == 0 {
        return out;
    }
    let h = 0.25_f64;
    let limit = f64::from(nt) * 5.0 + 30.0;
    let x0 = 1e-2_f64;
    let mut x = x0;
    let mut fx = f(x);
    while (out.len() as u32) < nt && x < limit {
        let xn = x + h;
        let fn_ = f(xn);
        if fx * fn_ < 0.0 {
            // Refine the sign-change bracket [x, xn] with the superlinear Illinois
            // false-position method (~10-15 evals) instead of 80 bisection steps.
            // illinois_root wants an INCREASING g with g(lo) < 0 < g(hi); orient by
            // the sign at the low end (f rises through the zero when fx < 0, else
            // negate). Same guaranteed convergence as bisection, far fewer f-evals.
            let root = if fx < 0.0 {
                crate::beta::illinois_root(&f, x, xn, fx, fn_)
            } else {
                crate::beta::illinois_root(|t| -f(t), x, xn, -fx, -fn_)
            };
            out.push(root);
        }
        x = xn;
        fx = fn_;
    }
    out
}

/// First `nt` positive zeros of `ber`. Matches `scipy.special.ber_zeros`.
#[must_use]
pub fn ber_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(ber, nt)
}

/// First `nt` positive zeros of `bei`. Matches `scipy.special.bei_zeros`.
#[must_use]
pub fn bei_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(bei, nt)
}

/// First `nt` positive zeros of `ker`. Matches `scipy.special.ker_zeros`.
#[must_use]
pub fn ker_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(ker, nt)
}

/// First `nt` positive zeros of `kei`. Matches `scipy.special.kei_zeros`.
#[must_use]
pub fn kei_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(kei, nt)
}

/// First `nt` positive zeros of `ber'`. Matches `scipy.special.berp_zeros`.
#[must_use]
pub fn berp_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(berp, nt)
}

/// First `nt` positive zeros of `bei'`. Matches `scipy.special.beip_zeros`.
#[must_use]
pub fn beip_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(beip, nt)
}

/// First `nt` positive zeros of `ker'`. Matches `scipy.special.kerp_zeros`.
#[must_use]
pub fn kerp_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(kerp, nt)
}

/// First `nt` positive zeros of `kei'`. Matches `scipy.special.keip_zeros`.
#[must_use]
pub fn keip_zeros(nt: u32) -> Vec<f64> {
    kelvin_zeros_of(keip, nt)
}

/// First `nt` positive zeros of all eight Kelvin functions, returned in SciPy's
/// order `(ber, bei, ker, kei, ber', bei', ker', kei')`.
///
/// Matches `scipy.special.kelvin_zeros`.
#[must_use]
pub fn kelvin_zeros(nt: u32) -> [Vec<f64>; 8] {
    [
        ber_zeros(nt),
        bei_zeros(nt),
        ker_zeros(nt),
        kei_zeros(nt),
        berp_zeros(nt),
        beip_zeros(nt),
        kerp_zeros(nt),
        keip_zeros(nt),
    ]
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
        let mut ans = if nm1 != 0.0 {
            1.0 / nm1
        } else {
            -x.ln() - EULER
        };
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
    result += -inv_x2 - inv_x3 - inv_x4 / 2.0 + inv_x6 / 6.0 - inv_x8 / 6.0 + 3.0 * inv_x10 / 10.0
        - 5.0 * inv_x12 / 6.0;

    result
}

/// Digamma function ψ(x) = d(ln Γ(x))/dx (scalar).
pub fn digamma_scalar(x: f64) -> f64 {
    // ZERO IS A POLE WITH A SIGN, not a NaN. scipy 1.17.1:
    //   psi( 0.0) = -inf
    //   psi(-0.0) = +inf
    // In IEEE arithmetic `x == 0.0` is TRUE for -0.0, so the old guard
    // `x <= 0.0 && x == x.floor()` collapsed both zeros into one NaN
    // (frankenscipy-eaqem). The sign is INVERTED relative to intuition -- +0.0
    // gives NEGATIVE infinity -- so these are copied from the measurement, not
    // reasoned out.
    //
    // `gammasgn_scalar` in gamma.rs already handles the same input correctly
    // with `if x == 0.0 { if x.is_sign_negative() {..} }`, which is the positive
    // control proving this pattern is known here; digamma simply omitted it.
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    // The negative integers remain genuine NaN poles: psi(-1.0) = psi(-2.0) = nan.
    if x < 0.0 && x == x.floor() {
        return f64::NAN;
    }

    let mut val = x;
    let mut result = 0.0;

    if val < 0.0 {
        result -= std::f64::consts::PI / (std::f64::consts::PI * val).tan();
        val = 1.0 - val;
    }

    // Recur up to val ≥ 12 then apply the Stirling-type asymptotic through the
    // 1/x^10 term. The previous (shift-to-8, through 1/x^6) form left a ~2e-10
    // absolute error (worst ~4e-9 relative) that capped the accuracy of every
    // digamma-based series (e.g. the hyp2f1/hyperu logarithmic connection forms);
    // this reaches ~4e-14 worst relative — verified vs mpmath over x∈[0.01,1000]
    // and the reflected negative axis — for ~4 extra recur steps and 2 terms.
    while val < 12.0 {
        result -= 1.0 / val;
        val += 1.0;
    }

    let inv_x = 1.0 / val;
    let inv_x2 = inv_x * inv_x;
    let mut term = inv_x2;
    result += val.ln() - inv_x / 2.0 - term / 12.0; // 1/x^2
    term *= inv_x2;
    result += term / 120.0; // +1/(120 x^4)
    term *= inv_x2;
    result -= term / 252.0; // -1/(252 x^6)
    term *= inv_x2;
    result += term / 240.0; // +1/(240 x^8)
    term *= inv_x2;
    result -= term / 132.0; // -1/(132 x^10)

    result
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial & Utility Functions
// ══════════════════════════════════════════════════════════════════════

/// Rising factorial (Pochhammer symbol): (x)_n = x(x+1)...(x+n-1).
///
/// Matches `scipy.special.poch`.
pub fn poch(x: f64, n: f64) -> f64 {
    // poch(x, n) = (x)_n = Γ(x+n)/Γ(x). Matching scipy.special.poch requires
    // handling the Γ poles at non-positive integers, which the plain
    // gammaln/gammasgn ratio collapses to NaN:
    //   * numerator-only pole (x+n a non-positive integer, x not)  → +inf;
    //   * denominator-only pole (x a non-positive integer, x+n not) → 0;
    //   * both poles (⟺ n integer): the exact finite limit, computed as the
    //     signed (reciprocal) rising product — e.g. poch(-2,-2)=1/12,
    //     poch(-5,3)=-60 — which the gammaln ratio (∞/∞) cannot give.
    // scipy returns +∞ (never −∞) at every single-pole divergence; the
    // pole-cases are resolved before the integer product so the sign of the
    // product's zero factor can't flip +inf to −inf. frankenscipy-7o3a2
    if n == 0.0 {
        return 1.0;
    }
    let xn = x + n;
    let x_pole = x <= 0.0 && x == x.floor();
    let xn_pole = xn <= 0.0 && xn == xn.floor();
    if xn_pole && !x_pole {
        return f64::INFINITY; // numerator Γ pole, finite denominator
    }
    if x_pole && !xn_pole {
        return 0.0; // denominator Γ pole, finite numerator
    }
    if n == n.floor() {
        // Integer n: exact signed (reciprocal) product. Correct through the
        // both-pole finite limit and the ordinary case alike. The loop is
        // bounded — single-pole cases returned above, and the both-pole case
        // forces |n| ≤ |x|; the cap guards pathological large n (necessarily
        // neither-pole there, where the smooth Γ path below stays accurate).
        let ni = n as i64;
        if ni.abs() <= 4096 {
            let mut r = 1.0_f64;
            if ni > 0 {
                for i in 0..ni {
                    r *= x + i as f64;
                }
                return r;
            }
            for i in ni..0 {
                r *= x + i as f64;
            }
            return 1.0 / r;
        }
    }
    // Smooth signed Γ-ratio (neither argument hits a pole here). gammaln gives
    // ln|·|, so restore the sign from the gamma factors — scipy.special
    // .poch(-4.3, 0.5) = -2.938, not +2.938. For x>0 both signs are +1.
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
/// Matches `scipy.special.spence`. Real arguments take the closed-form real
/// dilogarithm; complex arguments evaluate Li₂(1 - z) via [`dilog_complex`]
/// (scipy accepts complex `spence` and `cspence`, returning finite values
/// across the whole plane — our previous real-only kernel fail-closed on
/// complex input).
pub fn spence(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "spence",
        x_tensor,
        mode,
        1 << 17, // spence break-even ~85k (BlackThrush A/B: 65536 loses 1.20x, 131072 wins 0.63x)
        |x| Ok(spence_scalar(x)),
        // Form 1 - z by negating the imaginary part directly rather than
        // subtracting: `0.0 - (+0.0)` would collapse to `+0.0` and lose the
        // sign of zero, but scipy's spence branch cut (z real ≤ 0) is continuous
        // from BELOW, so a real input like -0.4 must carry -0.0 into Li₂ to land
        // on scipy's lower-side value. For any nonzero imaginary part this is
        // identical to `1 - z`.
        |z| Ok(dilog_complex(Complex64::new(1.0 - z.re, -z.im))),
    )
}

/// Cephes `spence.c` rational-approximation coefficients (numerator A, denominator B;
/// degree 7). `scipy.special.spence` uses this exact form.
const SPENCE_A: [f64; 8] = [
    4.651_285_860_739_900_5e-5,
    7.315_890_452_380_947_1e-3,
    1.338_476_395_783_090_2e-1,
    8.796_913_117_545_303e-1,
    2.711_498_511_965_534_7e0,
    4.256_971_560_081_218e0,
    3.297_713_409_852_251e0,
    1.000_000_000_000_000_1e0,
];
const SPENCE_B: [f64; 8] = [
    6.909_904_889_125_533e-4,
    2.540_437_639_325_444e-2,
    2.829_748_606_025_681e-1,
    1.411_725_977_518_310_7e0,
    3.638_005_333_451_371e0,
    5.032_788_801_433_170e0,
    3.547_713_409_852_251e0,
    9.999_999_999_999_999e-1,
];

#[inline]
fn spence_polevl(x: f64, c: &[f64; 8]) -> f64 {
    let mut acc = c[0];
    for &ci in &c[1..] {
        acc = acc * x + ci;
    }
    acc
}

/// Spence's function `spence(x) = Li₂(1 − x)` for `x ≥ 0` via the Cephes rational
/// approximation: a fixed pair of degree-7 polynomials with argument reduction, instead
/// of the naïve ~50-term Li₂ power series (`dilog_real`). Validated against that series
/// path (`spence_cephes_matches_series_path`) to < 1e-11 over x ∈ (0, 60].
fn spence_cephes(x: f64) -> f64 {
    if x == 1.0 {
        return 0.0;
    }
    if x == 0.0 {
        return PI_SQUARED_OVER_SIX;
    }
    let mut x = x;
    let mut flag = 0u8;
    if x > 2.0 {
        x = 1.0 / x;
        flag |= 2;
    }
    let w = if x > 1.5 {
        flag |= 2;
        1.0 / x - 1.0
    } else if x < 0.5 {
        flag |= 1;
        -x
    } else {
        x - 1.0
    };
    let mut y = -w * spence_polevl(w, &SPENCE_A) / spence_polevl(w, &SPENCE_B);
    if flag & 1 != 0 {
        y = PI_SQUARED_OVER_SIX - x.ln() * (1.0 - x).ln() - y;
    }
    if flag & 2 != 0 {
        let z = x.ln();
        y = -0.5 * z * z - y;
    }
    y
}

#[must_use]
pub fn spence_scalar(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x < 0.0 {
        return f64::NAN;
    }
    spence_cephes(x)
}

/// Complex dilogarithm Li₂(z), principal branch (cut on the real axis for
/// z ≥ 1). Evaluated via the Bernoulli-number series
/// `Li₂(z) = Σ_{n≥1} (B_{n-1}/n!) · uⁿ` with `u = -ln(1 - z)`, which converges
/// rapidly (radius |u| < 2π) once the argument is reduced near 0. Two standard
/// transformations do the reduction:
///   * inversion `Li₂(z) = -Li₂(1/z) - π²/6 - ½·ln(-z)²` for |z| > 1,
///   * reflection `Li₂(z) = π²/6 - ln(z)·ln(1-z) - Li₂(1-z)` for Re(z) > ½.
///
/// After reduction the series argument has Re ≤ ½ and |·| ≤ 1, so |u| ≲ 1.05
/// and 20 terms reach full f64 precision. Verified against `scipy.special.spence`
/// (= Li₂(1-z)) to ≤ 2e-15 relative error over 400+ random points and along the
/// branch cut.
#[must_use]
pub fn dilog_complex(z: Complex64) -> Complex64 {
    // B_{n-1}/n! for n = 1..=20 (odd Bernoulli numbers above B_1 vanish, so the
    // even-index-n entries n ≥ 4 are exactly zero).
    const COEFFS: [f64; 20] = [
        1.000_000_000_000_000_0e0,    // n=1
        -2.500_000_000_000_000_0e-1,  // n=2
        2.777_777_777_777_777_6e-2,   // n=3
        0.0,                          // n=4
        -2.777_777_777_777_777_8e-4,  // n=5
        0.0,                          // n=6
        4.724_111_866_969_009_8e-6,   // n=7
        0.0,                          // n=8
        -9.185_773_074_661_964_1e-8,  // n=9
        0.0,                          // n=10
        1.897_886_998_897_100_1e-9,   // n=11
        0.0,                          // n=12
        -4.064_761_645_144_225_6e-11, // n=13
        0.0,                          // n=14
        8.921_691_020_456_452_3e-13,  // n=15
        0.0,                          // n=16
        -1.993_929_586_072_107_4e-14, // n=17
        0.0,                          // n=18
        4.518_980_029_619_918_2e-16,  // n=19
        0.0,                          // n=20
    ];

    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    if z.re == 1.0 && z.im == 0.0 {
        return Complex64::new(PI_SQUARED_OVER_SIX, 0.0);
    }

    // dilog_series(w) = Σ COEFFS[n-1] · (-ln(1-w))ⁿ
    let series = |w: Complex64| -> Complex64 {
        let u = -(Complex64::new(1.0, 0.0) - w).ln();
        let mut acc = Complex64::new(0.0, 0.0);
        let mut up = Complex64::new(1.0, 0.0);
        for &c in &COEFFS {
            up = up * u;
            if c != 0.0 {
                acc = acc + up * c;
            }
        }
        acc
    };

    let pi_sq_6 = Complex64::new(PI_SQUARED_OVER_SIX, 0.0);

    // Inversion: bring |z| ≤ 1.
    let (z, sign, offset) = if z.abs() > 1.0 {
        let log_neg = (-z).ln();
        let offset = -log_neg * log_neg * 0.5 - pi_sq_6;
        (z.recip(), -1.0, offset)
    } else {
        (z, 1.0, Complex64::new(0.0, 0.0))
    };

    // Reflection: push the series argument into Re ≤ ½.
    let one = Complex64::new(1.0, 0.0);
    let val = if z.re > 0.5 {
        pi_sq_6 - z.ln() * (one - z).ln() - series(one - z)
    } else {
        series(z)
    };

    val * sign + offset
}

// Retained REFERENCE implementation: the optimized path's own documentation
// cites it, and it is exercised by a test that checks the two agree. Dead only
// in non-test builds, so the test build still proves it is used
// (frankenscipy-e2ve2).
#[cfg_attr(not(test), allow(dead_code))]
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

// Retained REFERENCE implementation: the optimized path's own documentation
// cites it, and it is exercised by a test that checks the two agree. Dead only
// in non-test builds, so the test build still proves it is used
// (frankenscipy-e2ve2).
#[cfg_attr(not(test), allow(dead_code))]
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
        1 << 16, // wrightomega break-even ~50k (BlackThrush A/B: 32768 loses 1.54x, 65536 wins 0.79x)
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

    // ω is the Wright omega function. The previous Newton iteration on
    // w + Log(w) = z used a crude initial guess and converged to the wrong
    // sheet for moderate |z| (W·e^W and w+Log(w)=z have infinitely many roots),
    // so values were wrong by O(1) across the plane (maxrel ~3 vs scipy).
    //
    // Use the exact identity ω(z) = W_{K(z)}(e^z), where K = ⌈(Im z − π)/(2π)⌉
    // is the unwinding number, and evaluate the branch-K Lambert W with a
    // region-aware initial guess (the same machinery scipy's `_lambertw.pxd`
    // and the TOMS 917 reduction rely on). This matches scipy.special.wrightomega
    // to ≤ 1.2e-15 over the whole representable plane off the branch cuts.
    let pi = std::f64::consts::PI;

    // Exact singular branch points z = -1 ± iπ where ω = -1.
    if z.re == -1.0 && z.im.abs() == pi {
        return Ok(Complex64::from_real(-1.0));
    }

    let k = ((z.im - pi) / (2.0 * pi)).ceil();

    // e^z is representable: ω = W_K(e^z) directly.
    if z.re > -690.0 && z.re < 690.0 {
        return Ok(wrightomega_via_lambertw(z.exp(), k));
    }

    // e^z over/underflows. In the principal strip (|Im z| < π) ω collapses to
    // e^z (→ 0 as Re z → −∞); otherwise the large-|z| asymptotic ω ≈ z − Log(z)
    // holds (it is the principal-Log root of w + Log(w) = z off the cuts).
    if z.im.abs() < pi && z.re < 0.0 {
        return Ok(z.exp());
    }
    let mut w = z - z.ln();
    for _ in 0..60 {
        let f = w + w.ln() - z;
        if f.abs() < 1.0e-15 * (1.0 + z.abs()) {
            break;
        }
        let fp = Complex64::from_real(1.0) + Complex64::from_real(1.0) / w;
        w = w - f / fp;
    }
    Ok(w)
}

/// Branch-`k` Lambert W (W_k(ζ)) used by the Wright omega reduction: a
/// region-aware initial guess followed by Halley iteration on `w·e^w = ζ`.
fn wrightomega_via_lambertw(zeta: Complex64, k: f64) -> Complex64 {
    let mut w = lambertw_branch_guess(zeta, k);
    for _ in 0..100 {
        let ew = w.exp();
        let f = w * ew - zeta;
        let wp1 = w + Complex64::from_real(1.0);
        let denom = ew * wp1 - (w + Complex64::from_real(2.0)) * f / (wp1 * 2.0);
        if denom.abs() < 1.0e-300 {
            break;
        }
        let step = f / denom;
        w = w - step;
        if step.abs() <= 1.0e-16 * (1.0 + w.abs()) {
            break;
        }
    }
    w
}

/// Initial guess for W_k(ζ) (principal Lambert W when k = 0). Mirrors the
/// region split in scipy's `_lambertw.pxd`: branch-point series near ζ = −1/e
/// (only the principal W₀ basin meets the branch point there), a [2/2] Padé
/// of W₀ in the empirical region near 0, `Log ζ` for the rest of W₀, and the
/// asymptotic series `L₁ − L₂ + L₂/L₁` for every non-principal branch.
fn lambertw_branch_guess(zeta: Complex64, k: f64) -> Complex64 {
    let e = std::f64::consts::E;
    if k == 0.0 && (zeta + Complex64::from_real(1.0 / e)).abs() < 0.3 {
        let p = ((zeta * e + Complex64::from_real(1.0)) * 2.0).powf(0.5);
        let p2 = p * p;
        let p3 = p2 * p;
        return Complex64::from_real(-1.0) + p - p2 / 3.0 + p3 * (11.0 / 72.0);
    }
    if k == 0.0 {
        if -1.0 < zeta.re
            && zeta.re < 1.5
            && zeta.im.abs() < 1.0
            && zeta.re > -2.5 * zeta.im.abs() - 0.2
        {
            let z2 = zeta * zeta;
            let num = zeta * (Complex64::from_real(60.0) + zeta * 114.0 + z2 * 17.0);
            let den = Complex64::from_real(60.0) + zeta * 174.0 + z2 * 101.0;
            return num / den;
        }
        return zeta.ln();
    }
    let l1 = zeta.ln() + Complex64::new(0.0, 2.0 * std::f64::consts::PI * k);
    let l2 = l1.ln();
    l1 - l2 + l2 / l1
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

        let x_new = if dpx > 1e-30 {
            let step = x - err / dpx;
            if step > lo && step < hi {
                step
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };

        // Iterate-convergence break: once the estimate stops moving, x is the root
        // to machine precision. Without this, the (necessarily) tight `1e-15·y`
        // residual tolerance above is often unreachable (betainc carries ~1e-15
        // relative noise), so Newton oscillated at the ULP floor to the 120-iter
        // cap — the cause of a ~90-eval, ~18× SciPy loss on stdtrit / t- & F-quantiles
        // (each eval is a full betainc). Newton converges in ~5-7 iterates instead.
        if (x_new - x).abs() <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) {
            return x_new;
        }
        x = x_new;
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
    map_real_binary_eager("gammaincinv", a_tensor, y_tensor, mode, |a, y| {
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
        // Wilson-Hilferty: the Gamma(a,1) quantile ≈ a·(1 − 1/(9a) + z/√(9a))³ with z = Φ⁻¹(y).
        // Far better than the bare `a` seed → ~3-4 Newton iters instead of ~15 (compounds the
        // now-fast ndtri). Same converged root ⇒ accuracy unchanged.
        let z = ndtri_scalar(y);
        let h = 1.0 / (9.0 * a);
        let w = 1.0 - h + z * h.sqrt();
        (a * w * w * w).max(1e-300)
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
        let x_new = if dpx.abs() > 1e-30 {
            let step = x - err / dpx;
            // Accept Newton step only if it stays in bracket
            if step > lo && step < hi {
                step
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };

        // Iterate-convergence break (see betaincinv_scalar): the `1e-15·y` residual
        // tolerance above is often unreachable (gammainc carries ~1e-15 relative
        // noise), so without this Newton oscillated at the ULP floor toward the
        // 100-iter cap — each iter a full gammainc. Return as soon as the estimate
        // stops moving; x is then the root to machine precision (~3-4 iters).
        if (x_new - x).abs() <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) {
            return x_new;
        }
        x = x_new;
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
    map_real_binary_eager("gammainccinv", a_tensor, y_tensor, mode, |a, y| {
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
        let step = x - err / dqx;
        let x_new = if step > 0.0 { step } else { 0.5 * x };
        // Iterate-convergence break (see betaincinv_scalar): the `1e-16·y` residual
        // tolerance is unreachable (gammaincc carries ~1e-15 noise), so Newton else
        // oscillates at the ULP floor to the 30-iter cap — the chdtri / chi²-quantile
        // loss. Return once the estimate stops moving (root to machine precision).
        if (x_new - x).abs() <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) {
            return x_new;
        }
        x = x_new;
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
    if y >= 2e-3 {
        // Moderate tail: erfcinv(y) = -Φ⁻¹(y/2)/√2. ndtri's Halley path (y/2 ∈ [1e-3, 0.03125])
        // is fast (no erfcx continued-fraction) and accurate to ~1e-15. No recursion: ndtri only
        // calls back into erfcinv_conv for ITS extreme tail (y/2 < 1e-3, i.e. y < 2e-3, below).
        return -ndtri_scalar(0.5 * y) * std::f64::consts::FRAC_1_SQRT_2;
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
        let step = f * sqrt_pi * ex / 2.0;
        x += step;
        // Iterate-convergence break (see betaincinv_scalar): the absolute `1e-16`
        // log-space residual is unreachable for tiny y (~1e-15 relative noise), so
        // Newton otherwise ran all 16 iters — each a full erfcx continued fraction —
        // making the deep tail ~17× slower than SciPy. Stop once x stops moving.
        if step.abs() <= 4.0 * f64::EPSILON * x.abs().max(f64::MIN_POSITIVE) {
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
    // The scalar-per-element serial map lost ~1.6x to SciPy's SIMD-vectorized ufunc
    // (broad sweep 2a032a74): the erfcx Cephes rational is the dominant cost and it
    // SIMD-vectorises byte-identically (a lane-wise Horner is bit-for-bit the scalar
    // fold). Take the 8-wide path below the parallel gate — that is exactly the
    // serial regime where the loss lived; at/above the gate the parallel path already
    // wins, and small arrays keep the scalar path (SIMD setup ~= per-element cost).
    if let SpecialTensor::RealVec(values) = x_tensor
        && (64..(1 << 18)).contains(&values.len())
    {
        return Ok(SpecialTensor::RealVec(erfcx_real_vec_simd(values)));
    }
    map_real_or_complex(
        "erfcx",
        x_tensor,
        mode,
        1 << 18, // erfcx break-even ~205k (BlackThrush A/B: 131072 loses 1.52x, 262144 wins 0.85x)
        |x| Ok(erfcx_scalar(x)),
        |z| erfcx_complex_scalar(z, mode),
    )
}

/// 8-wide erfcx over a real slice. The Cephes rational (1 ≤ x < 25, its P/Q and R/S
/// branches) is evaluated SIMD; the exp path (x < 1) and the asymptotic (x ≥ 25) stay
/// scalar. Every lane is byte-identical to `erfcx_scalar`, so the returned Vec is
/// bit-for-bit `values.iter().map(erfcx_scalar).collect()`.
fn erfcx_real_vec_simd(values: &[f64]) -> Vec<f64> {
    use std::simd::Simd;
    const LANES: usize = 8;
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let x = Simd::<f64, LANES>::from_slice(&values[i..i + LANES]);
        let cephes = crate::error::erfcx_cephes_real_simd(x);
        for j in 0..LANES {
            let xj = values[i + j];
            out[i + j] = if (1.0..25.0).contains(&xj) {
                cephes[j]
            } else {
                erfcx_scalar(xj)
            };
        }
        i += LANES;
    }
    while i < values.len() {
        out[i] = erfcx_scalar(values[i]);
        i += 1;
    }
    out
}

/// Scaled complementary error function for complex argument:
/// erfcx(z) = e^{z²} erfc(z) = w(iz), the Faddeeva function. frankenscipy-rkwu4.
fn erfcx_complex_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    // i·z = (-Im(z)) + i·Re(z).
    wofz_scalar(Complex64::new(-z.im, z.re), mode)
}

pub fn erfcx_scalar(x: f64) -> f64 {
    if x < 1.0 {
        // Small / negative x: exp(x²)·erfc(x). For x<0, erfc(x)≈2 and exp(x²)
        // grows; for 0≤x<1, exp(x²) is near 1 and erfc via 1−erf is accurate.
        (x * x).exp() * crate::erfc_scalar(x)
    } else if x < 25.0 {
        // x ≥ 1: the Cephes rational IS erfcx (erfc = e^{−x²}·P/Q), so take it
        // directly — no exp(x²)·exp(−x²) round-trip (~2× faster, more accurate).
        crate::error::erfcx_cephes_real(x)
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
    // erfi(x) = (2/√π) e^{x²} D(x), with Dawson's D(x) = e^{-x²}∫₀ˣe^{t²}dt (O(1)
    // via the Cephes rational). Uniformly ~4-12× faster than the former Maclaurin
    // series `2x/√π Σ x^{2k}/(k!(2k+1))` (which needed ~60 terms near |x|=6) and
    // byte-identical to it — ≤2.4e-16 vs scipy over the whole |x| ≤ 6 range it
    // replaces. For x² > ln(f64::MAX) the e^{x²} overflows to ±inf (dawsn keeps its
    // sign), matching scipy. frankenscipy-sxr71.
    2.0 / std::f64::consts::PI.sqrt() * (x * x).exp() * dawsn_scalar(x)
}

pub fn erfi(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_or_complex(
        "erfi",
        x_tensor,
        mode,
        1 << 16, // erfi break-even ~45k (BlackThrush A/B: 32768 loses 1.15x, 65536 wins 0.59x)
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
    map_real_binary_eager("owens_t", h_tensor, a_tensor, mode, |h, a| {
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
    if lam == 0.0 {
        y.exp()
    } else {
        // scipy.special.inv_boxcox: exp(log1p(λy)/λ). This returns NaN when the
        // base λy+1 < 0, matching scipy — the old (λy+1).powf(1/λ) instead
        // returned a real power for integer 1/λ (e.g. inv_boxcox(5,-0.5) gave
        // 0.444 vs scipy NaN). frankenscipy-9ns59
        ((lam * y).ln_1p() / lam).exp()
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
    // scipy.special.boxcox1p uses log1p(x) so small |x| keeps full precision:
    // forming 1.0 + x explicitly (the old boxcox_transform(1.0 + x, λ)) loses
    // ~4 digits when |x| << 1 — boxcox1p(1e-12, λ) gave 1.0000889e-12 vs 1e-12.
    // The unified expm1(λ·log1p(x))/λ form also reproduces scipy's λ=±inf
    // sentinels and the x < -1 → NaN domain via IEEE arithmetic. frankenscipy-frqrf
    if x.is_nan() || lam.is_nan() {
        return f64::NAN;
    }
    let lgx = x.ln_1p(); // ln(1 + x); NaN for x < -1, -inf at x = -1
    if lam == 0.0 {
        lgx
    } else {
        (lam * lgx).exp_m1() / lam
    }
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
    // scipy.special.inv_boxcox1p: expm1(log1p(λy)/λ) (expm1(y) at λ=0). Keeps
    // precision near y=0 and yields NaN for negative base like inv_boxcox.
    // frankenscipy-9ns59
    if lam == 0.0 {
        y.exp_m1()
    } else {
        ((lam * y).ln_1p() / lam).exp_m1()
    }
}

/// Log of the standard normal CDF.
///
/// Matches `scipy.special.log_ndtr`.
pub fn log_ndtr(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if let SpecialTensor::RealVec(values) = x_tensor
        && (64..(1 << 20)).contains(&values.len())
    {
        return Ok(SpecialTensor::RealVec(log_ndtr_real_vec_simd(values)));
    }
    map_real_wg("log_ndtr", x_tensor, mode, |x| Ok(log_ndtr_scalar(x)))
}

fn log_ndtr_real_vec_simd(values: &[f64]) -> Vec<f64> {
    use std::simd::Simd;
    const LANES: usize = 8;
    let scale = Simd::<f64, LANES>::splat(-FRAC_1_SQRT_2);
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let x = Simd::<f64, LANES>::from_slice(&values[i..i + LANES]);
        let lanes = x.to_array();
        if lanes.iter().all(|x| x.is_finite() && *x >= -8.0) {
            let erfc = crate::error::erfc_full_simd_chunk(x * scale);
            for j in 0..LANES {
                out[i + j] = (0.5 * erfc[j]).ln();
            }
        } else {
            for j in 0..LANES {
                out[i + j] = log_ndtr_scalar(lanes[j]);
            }
        }
        i += LANES;
    }
    while i < values.len() {
        out[i] = log_ndtr_scalar(values[i]);
        i += 1;
    }
    out
}

/// Scalar helper for `log_ndtr`.
pub fn log_ndtr_scalar(x: f64) -> f64 {
    // log(Φ(x)) where Φ is the standard normal CDF.
    if x >= -1.0 {
        // Φ(x) = 1 − Φ(−x). For large positive x, Φ(x) rounds to exactly 1.0,
        // so the naive ln(Φ(x)) collapses to 0 even though log Φ(x) ≈ −Φ(−x) is
        // a tiny negative number (e.g. log_ndtr(15) ≈ −3.67e-51, not 0). Use
        // log1p(−Φ(−x)): Φ(−x) is the accurate tail value and log1p keeps the
        // result down to the denormal range — matches scipy.special.log_ndtr to
        // ~2e-16 across x ∈ [−1, 40]. frankenscipy-08o4z
        return (-ndtr_scalar(-x)).ln_1p();
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
        1 << 15, // dawsn break-even ~30k (BlackThrush A/B 2026-06-22: 16384 loses 1.96x, 32768 wins 0.95x)
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

/// Vectorized Struve function H_v(x) for a fixed order `v` over many arguments.
///
/// Matches `scipy.special.struve(v, x)` for scalar `v` and an array `x`. fsci
/// previously exposed only the scalar `struve`; SciPy's vectorized struve ufunc
/// is remarkably slow (~5 µs/point — 10.5 s for 2M points) because it drives a
/// per-point series/integral, whereas fsci's scalar kernel is ~47 ns. Fanning
/// that kernel across cores via the crate's order-preserving parallel map is a
/// ~500× win and fills the missing vectorized API. Bit-identical to a serial
/// `x.iter().map(|&xi| struve(v, xi))` (each element is an independent, pure call).
#[must_use]
pub fn struve_many(v: f64, x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(struve(v, x[i])))
        .expect("struve is infallible (NaN on out-of-domain, never Err)")
}

/// Vectorized modified Struve function L_v(x) for a fixed order `v` over many
/// arguments. Matches `scipy.special.modstruve(v, x)`; see [`struve_many`].
#[must_use]
pub fn modstruve_many(v: f64, x: &[f64]) -> Vec<f64> {
    par_map_indices(x.len(), |i| Ok::<f64, SpecialError>(modstruve(v, x[i])))
        .expect("modstruve is infallible (NaN on out-of-domain, never Err)")
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

/// Riemann zeta function ζ(s) for any real `s`, matching `scipy.special.zeta`.
///
/// scipy returns the analytic continuation over the whole real line — the pole
/// `ζ(1) = +∞`, `ζ(0) = -1/2`, the negative-`s` reflection (e.g. `ζ(-1) = -1/12`)
/// and the critical strip `0 < s < 1` (e.g. `ζ(1/2) ≈ -1.4603`). The underlying
/// `gamma::zeta_scalar` already implements all of these; this wrapper previously failed
/// closed to NaN for every `s ≤ 1`, diverging from scipy. frankenscipy.
pub fn zeta_scalar(s: f64) -> f64 {
    if s == f64::INFINITY {
        return 1.0;
    }
    crate::gamma::zeta_scalar(s)
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
        SpecialTensor::RealVec(values) => par_map_indices(values.len(), |i| {
            wofz_scalar(Complex64::from_real(values[i]), mode)
        })
        .map(SpecialTensor::ComplexVec),
        SpecialTensor::ComplexScalar(z) => wofz_scalar(*z, mode).map(SpecialTensor::ComplexScalar),
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| wofz_scalar(values[i], mode))
                .map(SpecialTensor::ComplexVec)
        }
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

/// Modified Fresnel positive integrals `(F₊, K₊)`, matching
/// `scipy.special.modfresnelp`.
///
/// `F₊(x) = ∫ₓ^∞ exp(i t²) dt` and the auxiliary
/// `K₊(x) = F₊(x)·exp(−i(x²+π/4)) / √π`.
///
/// Evaluated in closed form via the complex complementary error function,
/// `F₊(x) = (√π/2)·e^{iπ/4}·erfc(x·e^{−iπ/4})`, with `erfc(z) = e^{−z²}·w(iz)`
/// (the Faddeeva function `wofz`). Matches scipy to ~1e-15 for `x ≥ 0`. For
/// `x < 0` this returns the analytically-correct integral (scipy's specfun
/// `FFK` is inaccurate for negative arguments). frankenscipy
pub fn modfresnelp(x: f64) -> (Complex64, Complex64) {
    modfresnel_impl(x, true)
}

/// Modified Fresnel negative integrals `(F₋, K₋)`, matching
/// `scipy.special.modfresnelm`: `F₋(x) = ∫ₓ^∞ exp(−i t²) dt` and
/// `K₋(x) = F₋(x)·exp(i(x²+π/4)) / √π`. See [`modfresnelp`].
pub fn modfresnelm(x: f64) -> (Complex64, Complex64) {
    modfresnel_impl(x, false)
}

fn modfresnel_impl(x: f64, positive: bool) -> (Complex64, Complex64) {
    if x.is_nan() {
        let nan = Complex64::new(f64::NAN, f64::NAN);
        return (nan, nan);
    }
    let mode = RuntimeMode::Strict;
    let sqrt_pi_2 = std::f64::consts::PI.sqrt() / 2.0;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    // e^{±iπ/4}
    let rot = Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    let (front, arg_rot, k_phase_sign) = if positive {
        (rot, rot.conj(), -1.0) // F₊: e^{iπ/4}·erfc(x·e^{-iπ/4})
    } else {
        (rot.conj(), rot, 1.0) // F₋: e^{-iπ/4}·erfc(x·e^{iπ/4})
    };
    let z = arg_rot * x;
    let f = Complex64::new(sqrt_pi_2, 0.0) * front * erfc_via_wofz(z, mode);
    // K = F · exp(∓i(x²+π/4)) / √π
    let phase = Complex64::new(0.0, k_phase_sign * (x * x + std::f64::consts::FRAC_PI_4)).exp();
    let k = f * phase / sqrt_pi;
    (f, k)
}

/// Complex `erfc(z) = e^{−z²}·w(iz)` via the Faddeeva function, using the
/// reflection `erfc(z) = 2 − erfc(−z)` for `Re(z) < 0` to avoid `e^{−z²}`
/// overflow.
fn erfc_via_wofz(z: Complex64, mode: RuntimeMode) -> Complex64 {
    if z.re < 0.0 {
        return Complex64::new(2.0, 0.0) - erfc_via_wofz(-z, mode);
    }
    let iz = Complex64::new(-z.im, z.re); // i·z
    let w = wofz_scalar(iz, mode).unwrap_or(Complex64::new(f64::NAN, f64::NAN));
    (-z * z).exp() * w
}

/// Weideman's rational approximation of the Faddeeva function w(z) for the upper
/// half plane (Im z ≥ 0). One Horner pass over N=32 real coefficients plus a
/// Möbius map Z = (L+iz)/(L−iz): w = 2·p(Z)/(L−iz)² + (1/√π)/(L−iz). Uniformly
/// accurate (worst rel err 2.3e-13 over |z|<4, no near-real-axis pole), far
/// cheaper than the erf-Maclaurin series it replaces. J.A.C. Weideman, SIAM J.
/// Numer. Anal. 31 (1994) 1497.
fn wofz_weideman(z: Complex64) -> Complex64 {
    const L: f64 = 4.756_828_460_010_884;
    const A: [f64; 32] = [
        -1.303_179_786_305_008_7e-12,
        3.740_881_293_165_362_5e-12,
        8.030_367_899_963_89e-12,
        -2.154_363_207_783_877e-11,
        -5.544_235_948_166_462_4e-11,
        1.165_825_109_352_377_4e-10,
        4.153_743_091_833_453e-10,
        -5.231_020_481_196_329e-10,
        -3.208_015_091_723_369e-9,
        8.124_889_456_846_652e-10,
        2.379_755_677_989_741_7e-8,
        2.293_043_906_509_996_6e-8,
        -1.481_307_891_512_097_7e-7,
        -4.184_076_370_216_977_6e-7,
        4.255_833_137_575_008_5e-7,
        4.401_531_731_578_55e-6,
        6.821_031_944_001_985e-6,
        -2.140_961_920_171_075e-5,
        -1.307_544_925_461_534_6e-4,
        -2.453_298_027_002_143e-4,
        3.925_913_607_007_031e-4,
        4.519_541_105_349_217e-3,
        1.900_615_578_484_540_8e-2,
        5.730_440_352_983_722e-2,
        1.406_071_622_689_376_9e-1,
        2.954_445_107_150_873e-1,
        5.460_139_720_639_341e-1,
        9.019_254_893_647_999e-1,
        1.345_544_169_234_545,
        1.825_669_629_632_481_5,
        2.263_537_299_900_267_6,
        2.572_253_408_124_569_6,
    ];
    let iz = Complex64::new(-z.im, z.re); // i·z
    let l_minus = Complex64::new(L - iz.re, -iz.im); // L − iz
    let l_plus = Complex64::new(L + iz.re, iz.im); // L + iz
    let zz = l_plus / l_minus; // Z = (L+iz)/(L−iz)
    // Horner: p(Z) = ((…(a0·Z + a1)·Z + a2)…)·Z + a31.
    let mut p = Complex64::from_real(A[0]);
    for &c in &A[1..] {
        p = p * zz + Complex64::from_real(c);
    }
    let l_minus_sq = l_minus * l_minus;
    // 1/√π = 0.5641895835477563.
    const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;
    Complex64::from_real(2.0) * p / l_minus_sq + Complex64::from_real(INV_SQRT_PI) / l_minus
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
    if z.abs() < 0.5 {
        // Tiny |z|: the erf-Maclaurin series converges in ~2-3 terms — cheaper
        // than the fixed 32-term Weideman rational. -i·z = Im(z) − i·Re(z).
        let erf_val = crate::error::erf_complex_scalar(Complex64::new(z.im, -z.re));
        return Ok((-(z * z)).exp() * (Complex64::from_real(1.0) - erf_val));
    }
    if z.abs() < 4.0 {
        // Weideman (1994) rational approximation of the Faddeeva function on the
        // upper half plane — uniformly accurate (including near the real axis, no
        // pole issue) and ~2-4× faster than the former erf-Maclaurin path here
        // (which needed many terms as |z| → 4). N = 32 → worst rel err 2.3e-13 over
        // |z| < 4 (scipy's own wofz is ~1e-13); the complex-wofz tests use 5e-8.
        return Ok(wofz_weideman(z));
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
    // Gautschi/Laplace continued fraction for the Faddeeva function in the upper half-plane:
    //   w(z) = (i/√π) / (z − a₁/(z − a₂/(z − a₃/…))),   aₖ = k/2.
    // This band is gated to 4 ≤ |z| < 8, where the CF converges geometrically — ~24 terms reach ~1e-13
    // (more accurate than, and ~30× fewer ops than, the former 768-step Simpson quadrature, whose cost
    // made fsci's wofz/voigt ~70× slower per point than SciPy's Faddeeva). Evaluated bottom-up.
    const TERMS: usize = 24;
    let mut cf = Complex64::new(0.0, 0.0);
    for k in (1..=TERMS).rev() {
        cf = Complex64::from_real(k as f64 / 2.0) / (z - cf);
    }
    Complex64::new(0.0, 1.0 / PI.sqrt()) * (z - cf).recip()
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

/// Voigt profile evaluated over an array of `x` at fixed `(sigma, gamma)` — the batched form of
/// [`voigt_profile`]. `scipy.special.voigt_profile` is vectorized but SINGLE-THREADED (no `workers`
/// parameter), and the per-point cost is dominated by an expensive Faddeeva/`wofz` evaluation, so fanning
/// it across cores is a clean win. `out[i]` is bit-identical to `voigt_profile(xs[i], sigma, gamma)`;
/// parallel above 1<<14 points (the wofz kernel amortises the spawn floor well below that for huge arrays).
pub fn voigt_profile_many(xs: &[f64], sigma: f64, gamma: f64) -> Vec<f64> {
    par_map_indices_gated(xs.len(), 1 << 14, |i| {
        Ok(voigt_profile(xs[i], sigma, gamma))
    })
    .expect("voigt_profile is infallible")
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
    let lo = eps;
    let hi = 1.0 - eps;
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
    // ppf(p) is increasing in p, so f(p) = ppf(p) − x is increasing with
    // f(lo) < 0 < f(hi). Illinois false-position converges in ~10-15 ppf evals
    // (each is two `powf`) vs the ~40-step bisection this CDF ran on every call,
    // and lands on the machine-precision root rather than the 1e-14 residual
    // break. The endpoint ppf values are already computed above.
    crate::beta::illinois_root(|p| ppf(p) - x, lo, hi, f_lo - x, f_hi - x)
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
    // SciPy uses pow(x,y)-1 except where x^y is near 1 (catastrophic cancellation), where it
    // switches to expm1. The expm1 form is 1 ULP off for exact integer powers (powm1(2,3) gave
    // 6.999…98, not 7); route those through pow(x,y)-1 like SciPy.
    if (y * (x - 1.0)).abs() < 0.5 || y.abs() < 0.2 {
        (y * x.ln()).exp_m1()
    } else {
        x.powf(y) - 1.0
    }
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

    // Rapidly-convergent Bernoulli/log expansion on t ∈ (0, π) (valid |t| < 2π):
    //   Cl₂(t) = t − t·ln t + Σ_{n≥1} cₙ · t^{2n+1},  cₙ = (−1)^{n-1} B_{2n}/(2n·(2n+1)!)
    // The tail behaves like (t/2π)^{2n}, so ≤ ~26 terms reach machine precision
    // even at the worst point t → π (3.77e-13 vs mpmath over (0,π)) — replacing the
    // former 100 000-term direct `Σ sin(kt)/k²` sum, which was ~10⁴× slower AND only
    // ≤1e-5 accurate for near-rational t. cₙ > 0 for all n, so the sum is monotone.
    const CLAUSEN_C: [f64; 30] = [
        1.388_888_888_888_888_81e-2,
        6.944_444_444_444_444_44e-5,
        7.873_519_778_281_682_97e-7,
        1.148_221_634_332_745_51e-8,
        1.897_886_998_897_099_90e-10,
        3.387_301_370_953_521_20e-12,
        6.372_636_443_183_180_76e-14,
        1.246_205_991_295_067_15e-15,
        2.510_544_460_899_954_55e-17,
        5.178_258_806_090_623_20e-19,
        1.088_735_736_830_084_92e-20,
        2.325_744_114_302_087_08e-22,
        5.035_195_213_147_389_65e-24,
        1.102_649_929_438_121_50e-25,
        2.438_658_550_900_734_40e-27,
        5.440_142_678_856_252_74e-29,
        1.222_834_013_121_735_18e-30,
        2.767_263_468_967_950_83e-32,
        6.300_090_591_832_013_55e-34,
        1.442_086_838_841_847_64e-35,
        3.317_093_999_159_542_76e-37,
        7.663_913_557_920_658_38e-39,
        1.777_871_473_383_065_86e-40,
        4.139_605_898_234_137_51e-42,
        9.671_557_036_081_102_31e-44,
        2.266_718_701_676_612_31e-45,
        5.327_956_311_328_254_22e-47,
        1.255_724_838_956_433_59e-48,
        2.967_000_542_247_094_07e-50,
        7.026_787_317_600_742_43e-52,
    ];
    let mut sum = t - t * t.ln();
    let t2 = t * t;
    let mut p = t; // t^{2n+1}
    for &c in &CLAUSEN_C {
        p *= t2;
        let term = c * p;
        sum += term;
        if term.abs() < 1e-18 * sum.abs() {
            break;
        }
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
    map_real_par("kolmogorov", y_tensor, mode, |y| Ok(kolmogorov_scalar(y)))
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
    map_real_par("kolmogi", p_tensor, mode, |p| Ok(kolmogi_scalar(p)))
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
        // Exact Birnbaum-Tingey series. Compute terms in log-space to avoid
        // overflow of C(n, v). Stream log C(n, v) by the ratio recurrence
        //   log C(n, v+1) = log C(n, v) + ln(n − v) − ln(v + 1),
        // so each term costs two `ln` instead of two `gammaln` (~5-10× cheaper
        // per term); the dominant cost of this O(n) series. frankenscipy.
        let m = ((nf * (1.0 - d)).floor() as i64).max(0);
        let mut sum = 0.0_f64;
        let mut log_binom = 0.0_f64; // log C(n, 0) = 0
        for v in 0..=m {
            let vf = v as f64;
            let evn = d + vf / nf;
            if evn < 1.0 {
                // (1 − d − v/n) > 0; otherwise no real contribution.
                let omevn = 1.0 - evn;
                let log_term = log_binom + (vf - 1.0) * evn.ln() + (nf - vf) * omevn.ln();
                sum += log_term.exp();
            }
            if v < m {
                log_binom += (nf - vf).ln() - (vf + 1.0).ln();
            }
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

    // smirnov(n, ·) is continuous and strictly decreasing from 1 (at d=0) to 0
    // (at d=1), so invert by bisection on [0, 1]. The previous Newton iteration
    // overshot catastrophically when seeded in the flat upper tail (e.g. p=1e-6,
    // n=5: d0≈0.99 where d/dd smirnov ≈ −5e-8, so the step was ≈20, clamping to
    // the 1e-15 floor and returning ~0 instead of 0.937). Bisection is
    // unconditionally convergent here. frankenscipy-e6i43
    // smirnov(n, ·) is strictly decreasing from 1 (d=0) to 0 (d=1), and each
    // evaluation is an O(n) Birnbaum-Tingey series (gammaln-heavy). Solve the
    // INCREASING residual g(d) = p − smirnov(n, d) with Illinois false-position
    // (~13 smirnov evals) instead of the ~53-step bisection this ran — a ~4×
    // cut in the dominant cost. The bracket [0, 1] needs no kernel calls at the
    // ends: smirnov(n, 0) = 1 and smirnov(n, 1) = 0, so g(0) = p − 1 < 0 and
    // g(1) = p > 0. Bracket-preserving, so no Newton-style tail overshoot.
    crate::beta::illinois_root(|d| p - smirnov(n, d), 0.0, 1.0, p - 1.0, p)
}

// --- Cephes degree-trig support (sindg.c / tandg.c) ---
// scipy.special.{sindg,cosdg,tandg,cotdg} are the Cephes routines: reduce the
// angle in DEGREES (mod 360, then to an octant + small residual mod 45°) before
// converting that small residual to radians, so the polynomial argument stays
// tiny. A naive (x·π/180).cos() instead carries the full argument-reduction
// error (hundreds of ULP, and catastrophic near zero crossings). frankenscipy-3pzf7
const DEGTRIG_PI180: f64 = 1.745_329_251_994_329_576_92e-2; // π/180
const DEGTRIG_LOSSTH: f64 = 1.0e14;
#[allow(clippy::excessive_precision)]
const DEGTRIG_SINCOF: [f64; 6] = [
    1.589_623_015_722_184_479_52e-10,
    -2.505_074_776_285_035_401_35e-8,
    2.755_731_362_138_567_735_49e-6,
    -1.984_126_982_958_953_846_58e-4,
    8.333_333_333_322_118_588_62e-3,
    -1.666_666_666_666_663_072_95e-1,
];
#[allow(clippy::excessive_precision)]
const DEGTRIG_COSCOF: [f64; 6] = [
    -1.135_853_652_138_768_173_00e-11,
    2.087_570_084_197_473_167_78e-9,
    -2.755_731_417_929_673_881_12e-7,
    2.480_158_728_885_170_453_48e-5,
    -1.388_888_888_887_305_641_16e-3,
    4.166_666_666_666_659_292_18e-2,
];

#[inline]
fn degtrig_polevl(x: f64, coef: &[f64]) -> f64 {
    let mut acc = coef[0];
    for &c in &coef[1..] {
        acc = acc * x + c;
    }
    acc
}

/// Cosine of angle given in degrees.
///
/// Matches `scipy.special.cosdg(x)`.
#[must_use]
pub fn cosdg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let mut sign = 1.0_f64;
    let x = x.abs(); // cos is even
    if x > DEGTRIG_LOSSTH {
        return 0.0;
    }
    let mut y = (x / 45.0).floor();
    let mut z = (y / 16.0).floor();
    z = y - 16.0 * z;
    let mut j = z as i64;
    if j & 1 != 0 {
        j += 1;
        y += 1.0;
    }
    j &= 7;
    if j > 3 {
        j -= 4;
        sign = -sign;
    }
    if j > 1 {
        sign = -sign;
    }
    z = x - y * 45.0;
    z *= DEGTRIG_PI180;
    let zz = z * z;
    y = if j == 1 || j == 2 {
        z + z * (zz * degtrig_polevl(zz, &DEGTRIG_SINCOF))
    } else {
        1.0 - 0.5 * zz + zz * zz * degtrig_polevl(zz, &DEGTRIG_COSCOF)
    };
    if sign < 0.0 { -y } else { y }
}

/// Sine of angle given in degrees.
///
/// Matches `scipy.special.sindg(x)`.
#[must_use]
pub fn sindg(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let mut sign = 1.0_f64;
    let mut x = x;
    if x < 0.0 {
        x = -x;
        sign = -1.0;
    }
    if x > DEGTRIG_LOSSTH {
        return 0.0;
    }
    let mut y = (x / 45.0).floor();
    let mut z = (y / 16.0).floor();
    z = y - 16.0 * z;
    let mut j = z as i64;
    if j & 1 != 0 {
        j += 1;
        y += 1.0;
    }
    j &= 7;
    if j > 3 {
        sign = -sign;
        j -= 4;
    }
    z = x - y * 45.0;
    z *= DEGTRIG_PI180;
    let zz = z * z;
    y = if j == 1 || j == 2 {
        1.0 - 0.5 * zz + zz * zz * degtrig_polevl(zz, &DEGTRIG_COSCOF)
    } else {
        z + z * (zz * degtrig_polevl(zz, &DEGTRIG_SINCOF))
    };
    if sign < 0.0 { -y } else { y }
}

/// Cephes `tancot`: tangent (`cotflg=false`) or cotangent (`cotflg=true`)
/// of an angle in degrees, via exact modulo-180 octant reduction.
///
/// Mirrors `scipy/special/cephes/tandg.c` so the only floating-point rounding
/// is the final `tan` over the reduced argument in `[0, 90)`, matching scipy
/// bit-for-bit on the same libm. The naive `(x*pi/180).tan()` lost up to
/// ~1000 ULP because the degrees→radians scaling amplified the argument error.
fn degtrig_tancot(xx: f64, cotflg: bool) -> f64 {
    if xx.is_nan() {
        return f64::NAN;
    }
    let mut sign = if xx < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let mut x = xx.abs();
    if x > DEGTRIG_LOSSTH {
        return 0.0;
    }
    // Reduce modulo 180 degrees (tan/cot have period 180).
    x -= 180.0 * (x / 180.0).floor();
    if cotflg {
        if x <= 90.0 {
            x = 90.0 - x;
        } else {
            x -= 90.0;
            sign = -sign;
        }
    } else if x > 90.0 {
        x = 180.0 - x;
        sign = -sign;
    }
    // x is now folded into [0, 90]; the endpoints are exact.
    if x == 0.0 {
        return 0.0;
    } else if x == 45.0 {
        return sign;
    } else if x == 90.0 {
        // Singularity: scipy returns +inf regardless of the original sign.
        return f64::INFINITY;
    }
    sign * (x * DEGTRIG_PI180).tan()
}

/// Tangent of angle given in degrees.
///
/// Matches `scipy.special.tandg(x)`.
#[must_use]
pub fn tandg(x: f64) -> f64 {
    degtrig_tancot(x, false)
}

/// Cotangent of angle given in degrees.
///
/// Matches `scipy.special.cotdg(x)`.
#[must_use]
pub fn cotdg(x: f64) -> f64 {
    degtrig_tancot(x, true)
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
    // Non-negative integer powers up to 10^15 are exactly representable in f64 (< 2^53), and
    // scipy.special.exp10 returns them exactly (e.g. exp10(2.0) == 100.0). The generic
    // exp(x·ln10) form is 1 ULP off for these, so compute them directly.
    if (0.0..=15.0).contains(&x) && x == x.trunc() {
        return 10.0_f64.powi(x as i32);
    }
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
    // silu(x) = x·expit(x) — compute-bound (exp + reciprocal); parallelize large
    // arrays via the work-capped light path (was serial, ~1.7x slower than numpy).
    map_real_light("silu", x_tensor, mode, |x| Ok(silu_scalar(x)))
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
    // log_expit(x) = -log(1 + exp(-x)) — compute-bound (exp + log); parallelize
    // large arrays via the work-capped light path (was serial, ~1.2x slower).
    map_real_light("log_expit", x_tensor, mode, |x| Ok(log_expit_scalar(x)))
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

    /// `dilog_real` and `dilog_series` are the naive ~50-term Li₂ power series
    /// that `spence_cephes` REPLACED, and its own doc cites them: "Validated
    /// against that series". They were kept deliberately as the reference — and
    /// were dead code, which is how a reference quietly stops being checked
    /// (frankenscipy-e2ve2). This is that validation, executed rather than
    /// asserted in a comment.
    ///
    /// `spence(x)` is Li₂(1 − x), so the series is evaluated at `1 − x`.
    #[test]
    fn spence_agrees_with_the_retained_dilog_series_reference() {
        for &x in &[0.05_f64, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9] {
            let fast = super::spence_scalar(x);
            let reference = super::dilog_real(1.0 - x);
            assert!(
                (fast - reference).abs() <= 1.0e-12 * reference.abs().max(1.0),
                "spence({x}) = {fast} but the retained series reference gives {reference}"
            );
        }
        // The raw series arm, on the domain where it is actually accurate.
        // `dilog_series` is the NAIVE power series Σ zᵏ/k², whose truncation
        // error grows as |z| → 1: measured here, at z = -0.9 it differs from
        // `dilog_real` (which reflects negative arguments instead of summing
        // directly) by 4.0e-11 — -0.7521631791774027 against -0.7521631792172613.
        // That gap is the series' own convergence limit and is exactly why
        // `spence_cephes` replaced it, so the comparison is made where the
        // series is meant to be trusted rather than at the edge where it is not.
        for &z in &[-0.5_f64, -0.25, 0.0, 0.3, 0.5] {
            let direct = super::dilog_series(z);
            let via_real = super::dilog_real(z);
            assert!(
                (direct - via_real).abs() <= 1.0e-12 * via_real.abs().max(1.0),
                "dilog_series({z}) = {direct} vs dilog_real = {via_real}"
            );
        }
    }
    use super::*;

    #[test]
    fn ndtr_simd_matches_scalar_within_tol() {
        // ndtr SIMD vector path vs the scalar kernel across the useful CDF range.
        let mut xs: Vec<f64> = (0..2000)
            .map(|i| -8.0 + 16.0 * (i as f64) / 2000.0)
            .collect();
        for b in [-1.5, 1.5, -0.5, 0.5, 0.0, -37.0, 37.0] {
            xs.push(b);
        }
        let simd = match ndtr(&SpecialTensor::RealVec(xs.clone()), RuntimeMode::Strict).unwrap() {
            SpecialTensor::RealVec(v) => v,
            _ => unreachable!(),
        };
        let mut max_abs = 0.0f64;
        for (k, &x) in xs.iter().enumerate() {
            max_abs = max_abs.max((simd[k] - ndtr_scalar(x)).abs());
        }
        assert!(
            max_abs < 1e-15,
            "ndtr simd max abs diff vs scalar = {max_abs:e}"
        );
    }

    #[test]
    fn log_ndtr_simd_matches_scalar_within_tol() {
        let mut xs: Vec<f64> = (0..4096)
            .map(|i| -8.0 + 16.0 * (i as f64) / 4095.0)
            .collect();
        for b in [
            -50.0,
            -25.0,
            -8.0,
            -7.999_999,
            -1.0,
            0.0,
            6.0,
            12.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            xs.push(b);
        }
        let simd = match log_ndtr(&SpecialTensor::RealVec(xs.clone()), RuntimeMode::Strict).unwrap()
        {
            SpecialTensor::RealVec(v) => v,
            _ => unreachable!(),
        };
        let mut max_abs = 0.0f64;
        for (k, &x) in xs.iter().enumerate() {
            let expected = log_ndtr_scalar(x);
            if expected.is_nan() {
                assert!(simd[k].is_nan(), "log_ndtr simd preserved NaN at {x}");
            } else {
                max_abs = max_abs.max((simd[k] - expected).abs());
            }
        }
        assert!(
            max_abs < 1e-12,
            "log_ndtr simd max abs diff vs scalar = {max_abs:e}"
        );
    }

    #[test]
    fn erfi_dawsn_form_matches_scipy() {
        // erfi now always uses the O(1) (2/√π)e^{x²}dawsn(x) form (the |x|<6
        // Maclaurin series was removed). References from scipy.special.erfi
        // (1.17.1); asserted to 1e-12 relative, incl. negative x.
        let cases = [
            (0.5, 0.61495209469651),
            (1.0, 1.6504257587975),
            (2.0, 18.564802414576),
            (4.0, 1296959.7307176),
            (5.5, 1432099172039.8),
            (-3.0, -1629.9946226016),
        ];
        for (x, expected) in cases {
            let got = erfi_scalar(x);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs(),
                "erfi({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfcx_direct_cephes_matches_scipy() {
        // x ≥ 1 now takes the Cephes rational directly (no exp round-trip).
        // Covers both the P/Q (1≤x<8) and R/S (x≥8) branches, incl. the x=8 seam.
        // References from scipy.special.erfcx (1.17.1); asserted to 1e-13 relative.
        let cases = [
            (1.0, 0.42758357615581),
            (2.0, 0.25539567631051),
            (3.5, 0.15529365560889),
            (5.0, 0.11070463773307),
            (7.99, 0.070071436717953),
            (8.0, 0.069985166200881),
            (15.0, 0.037529606388506),
            (24.0, 0.023487546063683),
        ];
        for (x, expected) in cases {
            let got = erfcx_scalar(x);
            assert!(
                (got - expected).abs() <= 1e-13 * expected.abs(),
                "erfcx({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn wofz_weideman_region_matches_scipy() {
        // |z| < 4 now uses the Weideman rational (was the erf-Maclaurin series).
        // References from scipy.special.wofz (1.17.1); asserted to 1e-11 (well
        // inside Weideman N=32's 2.3e-13 and far tighter than the old 5e-8 test).
        // Includes a near-real-axis point (im=0.001) where a naive contour
        // quadrature would miss the pole.
        let cases = [
            (0.5, 0.5, 0.5331567079122, 0.2304882313845),
            (2.0, 3.0, 0.1307574696698, 0.08111265047746),
            (1.5, 0.001, 0.1057201587033, 0.4829111338981),
            (-1.0, 0.5, 0.3549003328676, -0.3428717191311),
            (3.5, 0.5, 0.02589697190059, 0.1644555405715),
            (0.0, 2.0, 0.2553956763105, 0.0),
            (2.5, 1.5, 0.1112334595626, 0.163236747198),
        ];
        for (re, im, wre, wim) in cases {
            let w = wofz_scalar(Complex64::new(re, im), RuntimeMode::Strict).unwrap();
            assert!(
                (w.re - wre).abs() <= 1e-11 && (w.im - wim).abs() <= 1e-11,
                "wofz({re}+{im}i) = {}+{}i, expected {wre}+{wim}i",
                w.re,
                w.im
            );
        }
    }

    #[test]
    fn signed_zero_is_preserved_across_the_odd_function_family() {
        // The remaining members of the signed-zero family enumerated by
        // scripts/scipy_signed_zero_probe.py. These are ODD or sign-preserving
        // at the origin, so scipy returns a SIGNED zero rather than flipping to
        // an infinity as psi and gamma do. Never asserted here before
        // (frankenscipy-eaqem follow-up).
        //
        // Measured, scipy 1.17.1: each f(+0.0) is +0.0 and each f(-0.0) is -0.0.
        let cases: [(&str, fn(f64) -> f64); 4] = [
            ("cbrt", cbrt),
            ("expm1", expm1),
            ("log1p", log1p),
            ("round", round),
        ];
        for (name, f) in cases {
            let pos = f(0.0);
            let neg = f(-0.0);
            assert_eq!(
                pos.to_bits(),
                0.0_f64.to_bits(),
                "{name}(+0.0) = {pos}, scipy +0.0 (bitwise)"
            );
            assert_eq!(
                neg.to_bits(),
                (-0.0_f64).to_bits(),
                "{name}(-0.0) = {neg}, scipy -0.0 (bitwise)"
            );
            // The whole point: a sign-losing implementation returns +0.0 for
            // both and passes any comparison written with ==, since
            // -0.0 == 0.0 is true. Only to_bits catches it.
            assert_ne!(
                pos.to_bits(),
                neg.to_bits(),
                "{name} collapsed the two zeros; == would not have caught this"
            );
        }

        // MUST-MISS: ordinary arguments are untouched, so this cannot pass by
        // some degenerate implementation that only handles zero.
        assert!((cbrt(8.0) - 2.0).abs() < 1e-12);
        assert!((expm1(0.0_f64.exp_m1()) - 0.0).abs() < 1e-12);
        assert!((log1p(1.0) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((round(2.5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn digamma_scalar_returns_a_signed_infinity_at_zero() {
        // MUST-HIT. `x == 0.0` is TRUE for -0.0 in IEEE, so the old guard
        // collapsed both zeros into one NaN. scipy 1.17.1 distinguishes them,
        // and the sign is INVERTED relative to intuition (frankenscipy-eaqem):
        //   psi( 0.0) = -inf
        //   psi(-0.0) = +inf
        let pos = digamma_scalar(0.0);
        let neg = digamma_scalar(-0.0);
        assert!(
            pos.is_infinite() && pos.is_sign_negative(),
            "psi(+0.0) = {pos}, scipy -inf"
        );
        assert!(
            neg.is_infinite() && neg.is_sign_positive(),
            "psi(-0.0) = {neg}, scipy +inf"
        );
        // And they must actually DIFFER -- if a future refactor collapses the
        // two zeros again this is the assertion that catches it.
        assert_ne!(
            pos.to_bits(),
            neg.to_bits(),
            "the two zeros must not collapse to one value"
        );

        // MUST-MISS: the negative integers are genuine NaN poles and must stay
        // NaN, so the fix cannot be "return an infinity for everything <= 0".
        for x in [-1.0_f64, -2.0, -3.0] {
            assert!(digamma_scalar(x).is_nan(), "psi({x}) must remain NaN");
        }
        // A non-integer negative argument is finite and unchanged:
        //   scipy.special.psi(-1.5) = 0.7031566406452434
        assert!(
            (digamma_scalar(-1.5) - 0.703_156_640_645_243_4).abs() < 1e-12,
            "psi(-1.5) = {}, scipy 0.7031566406452434",
            digamma_scalar(-1.5)
        );
        // ...as is the ordinary positive branch every existing test covers.
        //   scipy.special.psi(1.0) = -0.5772156649015329
        assert!(
            (digamma_scalar(1.0) - -0.577_215_664_901_532_9).abs() < 1e-12,
            "psi(1.0) = {}",
            digamma_scalar(1.0)
        );
    }

    #[test]
    fn digamma_scalar_matches_reference_high_accuracy() {
        // ψ(x) references from mpmath (30 digits). The Stirling asymptotic through
        // the 1/x^10 term (shift to x≥12) must resolve these to <1e-12 relative —
        // the earlier through-1/x^6 form left ~4e-9, capping digamma-based series.
        let cases = [
            (0.05, -20.497844991299869),
            (0.5, -1.9635100260214235),
            (1.0, -0.57721566490153286),
            (1.5, 0.036489973978576521),
            (2.0, 0.42278433509846714),
            (2.5, 0.70315664064524319),
            (3.7, 1.1671535393615114),
            (8.5, 2.0800908175794201),
            (50.0, 3.9019896734278922),
            (-0.5, 0.036489973978576521),
            (-2.3, 3.3173231575618227),
        ];
        for (x, expected) in cases {
            let got = digamma_scalar(x);
            assert!(
                (got - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
                "digamma({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn gammasgn_scalar_match_scipy() {
        // scipy.special.gammasgn: +1 for x>0, sign(sin pi x) on negative non-integer
        // intervals, +1 at x=0 (Gamma(0)=+inf), and NaN at negative-integer poles.
        use crate::gamma::gammasgn_scalar;
        let m = RuntimeMode::Strict;
        assert_eq!(gammasgn_scalar(3.0, m).unwrap(), 1.0);
        assert_eq!(gammasgn_scalar(0.5, m).unwrap(), 1.0);
        assert_eq!(gammasgn_scalar(-0.5, m).unwrap(), -1.0);
        assert_eq!(gammasgn_scalar(-1.5, m).unwrap(), 1.0);
        assert_eq!(gammasgn_scalar(-2.5, m).unwrap(), -1.0);
        assert_eq!(gammasgn_scalar(0.0, m).unwrap(), 1.0);
        assert!(gammasgn_scalar(-1.0, m).unwrap().is_nan());
    }

    #[test]
    fn expi_match_scipy() {
        // scipy.special.expi (exponential integral Ei).
        assert!(
            (expi_scalar(1.0) - 1.895_117_816_355_937).abs() < 1e-13,
            "expi(1)"
        );
        assert!(
            (expi_scalar(2.0) - 4.954_234_356_001_891).abs() < 1e-12,
            "expi(2)"
        );
        assert!(
            (expi_scalar(0.5) - 0.454_219_904_863_173_54).abs() < 1e-13,
            "expi(0.5)"
        );
        assert!(
            (expi_scalar(-1.0) - -0.219_383_934_395_520_5).abs() < 1e-13,
            "expi(-1)=-E1(1)"
        );
    }

    #[test]
    fn debye_erfc_exp2_iterated_match_analytic() {
        // Previously-untested helpers. debye(1,0)=1; debye(1,1)=int_0^1 t/(e^t-1) dt.
        assert!((debye(1, 0.0) - 1.0).abs() < 1e-12, "debye(1,0)");
        assert!(
            (debye(1, 1.0) - 0.777_504_634_112_248_5).abs() < 1e-7,
            "debye(1,1)"
        );
        // erfc_conv = erfc.
        assert!(
            (erfc_conv(1.0) - 0.157_299_207_050_285_16).abs() < 1e-12,
            "erfc_conv(1)"
        );
        // exp2_iterated(x) = exp(exp(x)).
        assert!(
            (exp2_iterated(0.0) - std::f64::consts::E).abs() < 1e-12,
            "exp2_iterated(0)=e"
        );
    }

    #[test]
    fn numerical_diff_helpers_match_analytic() {
        // Numerical-diff helpers, exact (within FD roundoff) for low-degree polys.
        // f=x^2: f''(2)=2.
        assert!(
            (central_diff2(|x: f64| x * x, 2.0, 0.01) - 2.0).abs() < 1e-6,
            "central_diff2"
        );
        // f=x0^2+x1^2: grad at [1,2]=[2,4]; Hessian diag 2, off-diag 0.
        let g = gradient_approx(|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], 0.001);
        assert!(
            (g[0] - 2.0).abs() < 1e-5 && (g[1] - 4.0).abs() < 1e-5,
            "gradient_approx"
        );
        let h = hessian_approx(|x: &[f64]| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0], 0.01);
        assert!(
            (h[0][0] - 2.0).abs() < 1e-4 && (h[1][1] - 2.0).abs() < 1e-4 && h[0][1].abs() < 1e-4,
            "hessian_approx"
        );
        // f(x)=[x0*x1, x0+x1] at [2,3]: Jacobian [[3,2],[1,1]].
        let j = jacobian_approx(
            |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]],
            &[2.0, 3.0],
            0.001,
        );
        assert!(
            (j[0][0] - 3.0).abs() < 1e-5
                && (j[0][1] - 2.0).abs() < 1e-5
                && (j[1][0] - 1.0).abs() < 1e-5
                && (j[1][1] - 1.0).abs() < 1e-5,
            "jacobian_approx"
        );
    }

    #[test]
    fn special_scalar_helpers_match_analytic() {
        // Previously-untested scalar helpers vs analytic/numpy values.
        assert!(
            (arcsinh(1.0) - (1.0 + 2.0_f64.sqrt()).ln()).abs() < 1e-12,
            "arcsinh(1)"
        );
        assert!(
            (arctanh(0.5) - 0.549_306_144_334_054_9).abs() < 1e-12,
            "arctanh(0.5)"
        );
        assert!(
            (log_comb(5.0, 2.0) - 10.0_f64.ln()).abs() < 1e-12,
            "log C(5,2)=ln10"
        );
        assert!((sinc_squared(0.0) - 1.0).abs() < 1e-12, "sinc^2(0)=1");
        // central_diff of x^2 at x=2 is exactly 2x=4 (quadratic -> O(h^2) error vanishes).
        assert!(
            (central_diff(|x: f64| x * x, 2.0, 0.1) - 4.0).abs() < 1e-12,
            "central_diff x^2"
        );
    }

    #[test]
    fn struve_match_scipy() {
        // scipy.special.struve (Struve function H_v(x)).
        assert!(
            (struve(0.0, 1.0) - 0.568_656_627_048_288_1).abs() < 1e-13,
            "struve(0,1)"
        );
        assert!(
            (struve(1.0, 2.0) - 0.646_763_728_283_562_2).abs() < 1e-13,
            "struve(1,2)"
        );
    }

    #[test]
    fn sici_shichi_match_scipy() {
        // scipy.special.sici (Si, Ci) and shichi (Shi, Chi) integrals at x=1.
        let (si, ci) = sici(1.0);
        assert!((si - 0.946_083_070_367_183_1).abs() < 1e-13, "Si(1)");
        assert!((ci - 0.337_403_922_900_968_16).abs() < 1e-13, "Ci(1)");
        let (shi, chi) = shichi(1.0);
        assert!((shi - 1.057_250_875_375_728_6).abs() < 1e-13, "Shi(1)");
        assert!((chi - 0.837_866_940_980_208_2).abs() < 1e-13, "Chi(1)");
    }

    #[test]
    fn fresnel_real_match_scipy() {
        // scipy.special.fresnel returns (S, C) Fresnel integrals.
        let (s, c) = fresnel(1.0);
        assert!((s - 0.438_259_147_390_354_7).abs() < 1e-13, "S(1)");
        assert!((c - 0.779_893_400_376_823).abs() < 1e-13, "C(1)");
        let (s2, c2) = fresnel(0.5);
        assert!((s2 - 0.064_732_432_859_999_29).abs() < 1e-13, "S(0.5)");
        assert!((c2 - 0.492_344_225_871_446_4).abs() < 1e-13, "C(0.5)");
    }

    #[test]
    fn spence_scalar_match_scipy() {
        // scipy.special.spence (Spence's dilogarithm): spence(0)=pi^2/6, spence(1)=0,
        // spence(2)=-pi^2/12.
        assert!(
            (spence_scalar(0.0) - 1.644_934_066_848_226_4).abs() < 1e-13,
            "spence(0)=pi^2/6"
        );
        assert_eq!(spence_scalar(1.0), 0.0);
        assert!(
            (spence_scalar(2.0) - -0.822_467_033_424_114_2).abs() < 1e-13,
            "spence(2)=-pi^2/12"
        );
    }

    #[test]
    fn spence_cephes_matches_series_path() {
        // The Cephes rational-approx port must reproduce the known-correct Li₂-series
        // path (`dilog_real`, itself validated vs scipy) across the whole domain — this
        // validates the ported A[8]/B[8] coefficients.
        let mut worst = 0.0f64;
        for i in 0..4000 {
            let x = 1e-5 + (i as f64) * 0.015; // x ∈ (0, 60]
            let c = super::spence_cephes(x);
            let s = super::dilog_real(1.0 - x);
            let rel = (c - s).abs() / s.abs().max(1e-12);
            worst = worst.max(rel);
        }
        assert!(
            worst < 1e-11,
            "spence cephes vs series worst rel err = {worst}"
        );
    }

    #[test]
    fn gammaincinv_betaincinv_match_scipy() {
        // scipy.special.gammaincinv/betaincinv (inverses of the regularized
        // incomplete gamma/beta).
        assert!(
            (gammaincinv_scalar(2.0, 0.5) - 1.678_346_990_016_661_2).abs() < 1e-11,
            "gammaincinv(2,0.5)"
        );
        assert!(
            (betaincinv_scalar(2.0, 3.0, 0.5) - 0.385_727_568_132_389_5).abs() < 1e-11,
            "betaincinv(2,3,0.5)"
        );
    }

    #[test]
    fn exp2_exp10_cosdg_sindg_match_scipy() {
        // scipy.special exp2/exp10 and degree trig (exact zeros at the axes).
        assert_eq!(exp2(3.0), 8.0);
        assert_eq!(exp10(2.0), 100.0);
        assert_eq!(cosdg(0.0), 1.0);
        assert!(cosdg(90.0).abs() < 1e-15, "cosdg(90)~0: {}", cosdg(90.0));
        assert!(sindg(180.0).abs() < 1e-15, "sindg(180)~0: {}", sindg(180.0));
        assert!((sindg(30.0) - 0.5).abs() < 1e-15, "sindg(30)");
    }

    #[test]
    fn gdtr_pdtr_match_scipy() {
        // scipy.special gamma CDF (gdtr) and Poisson CDF/inverse (pdtr/pdtri).
        use crate::gamma::{gdtr, pdtr, pdtri};
        assert!(
            (gdtr(1.0, 2.0, 3.0) - 0.800_851_726_528_544_2).abs() < 1e-13,
            "gdtr(1,2,3)"
        );
        assert!(
            (pdtr(2.0, 3.0) - 0.423_190_081_126_843_64).abs() < 1e-13,
            "pdtr(2,3)"
        );
        assert!(
            (pdtri(2.0, 0.4) - 3.105_378_597_263_35).abs() < 1e-11,
            "pdtri(2,0.4)"
        );
    }

    #[test]
    fn chdtr_fdtr_match_scipy() {
        // scipy.special chi-square CDF/inverse and F CDF.
        use crate::beta::fdtr;
        use crate::gamma::{chdtr, chdtri};
        assert!(
            (chdtr(5.0, 3.0) - 0.300_014_164_121_372_4).abs() < 1e-13,
            "chdtr(5,3)"
        );
        assert!(
            (chdtri(5.0, 0.3) - 6.064_429_984_154_905).abs() < 1e-11,
            "chdtri(5,0.3)"
        );
        assert!(
            (fdtr(5.0, 10.0, 2.0) - 0.835_805_049_100_261_1).abs() < 1e-13,
            "fdtr(5,10,2)"
        );
    }

    #[test]
    fn stdtr_stdtrit_match_scipy() {
        // scipy.special.stdtr (Student's t CDF) and stdtrit (its inverse).
        use crate::beta::{stdtr, stdtrit};
        assert!(
            (stdtr(5.0, 1.0) - 0.818_391_266_175_438_6).abs() < 1e-13,
            "stdtr(5,1)"
        );
        assert!(
            (stdtr(10.0, 2.0) - 0.963_305_982_614_629_9).abs() < 1e-13,
            "stdtr(10,2)"
        );
        assert!(
            (stdtrit(5.0, 0.95) - 2.015_048_373_333_023_3).abs() < 1e-11,
            "stdtrit(5,0.95)"
        );
    }

    #[test]
    fn factorial_match_scipy() {
        // scipy.special.factorial / factorial2 (double factorial).
        use crate::gamma::{factorial, factorial2};
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(10), 3_628_800.0);
        assert!(
            (factorial2(7) - 105.0).abs() < 1e-10,
            "factorial2(7)=7*5*3*1"
        );
        assert_eq!(factorial2(8), 384.0, "factorial2(8)=8*6*4*2");
    }

    #[test]
    fn gammaln_scalar_match_scipy() {
        // scipy.special.gammaln = ln|Gamma(x)|; stays finite where Gamma overflows.
        use crate::gamma::gammaln_scalar;
        let m = RuntimeMode::Strict;
        assert!(
            (gammaln_scalar(0.5, m).unwrap() - 0.572_364_942_924_7).abs() < 1e-12,
            "gammaln(0.5)=ln(sqrt pi)"
        );
        assert!(
            (gammaln_scalar(10.0, m).unwrap() - 12.801_827_480_081_469).abs() < 1e-11,
            "gammaln(10)=ln(9!)"
        );
        // Large arg: Gamma(100) overflows but gammaln stays finite.
        assert!(
            (gammaln_scalar(100.0, m).unwrap() - 359.134_205_369_575_4).abs() < 1e-9,
            "gammaln(100)"
        );
        assert!(
            (gammaln_scalar(0.1, m).unwrap() - 2.252_712_651_734_206).abs() < 1e-12,
            "gammaln(0.1)"
        );
    }

    #[test]
    fn zeta_zetac_match_scipy() {
        // scipy.special.zeta/zetac (Riemann zeta) at integer arguments.
        use crate::gamma::{zeta_scalar, zetac_scalar};
        assert!(
            (zeta_scalar(2.0) - 1.644_934_066_848_226_4).abs() < 1e-13,
            "zeta(2)=pi^2/6"
        );
        assert!(
            (zeta_scalar(3.0) - 1.202_056_903_159_594_2).abs() < 1e-13,
            "zeta(3) Apery"
        );
        assert!(
            (zeta_scalar(4.0) - 1.082_323_233_711_138_1).abs() < 1e-13,
            "zeta(4)=pi^4/90"
        );
        assert!(
            (zetac_scalar(2.0) - 0.644_934_066_848_226_4).abs() < 1e-13,
            "zetac(2)=zeta(2)-1"
        );
    }

    #[test]
    fn betaln_scalar_match_scipy() {
        // scipy.special.betaln = ln(B(a,b)); stays finite where B underflows.
        use crate::beta::betaln_scalar;
        let m = RuntimeMode::Strict;
        assert!(
            (betaln_scalar(2.0, 3.0, m).unwrap() - -2.484_906_649_788_000_4).abs() < 1e-13,
            "betaln(2,3)"
        );
        assert!(
            (betaln_scalar(0.5, 0.5, m).unwrap() - 1.144_729_885_849_4).abs() < 1e-12,
            "betaln(0.5,0.5)=ln(pi)"
        );
        // Large args: B(100,100) underflows but betaln stays finite.
        assert!(
            (betaln_scalar(100.0, 100.0, m).unwrap() - -139.665_259_086_706_67).abs() < 1e-10,
            "betaln(100,100)"
        );
    }

    #[test]
    fn poch_match_scipy() {
        // scipy.special.poch (Pochhammer rising factorial) = Gamma(x+n)/Gamma(x).
        assert!((poch(2.0, 3.0) - 24.0).abs() < 1e-12, "poch(2,3)");
        assert!((poch(0.5, 2.0) - 0.75).abs() < 1e-13, "poch(0.5,2)");
        assert!((poch(3.0, 0.0) - 1.0).abs() < 1e-14, "poch(3,0)");
        assert!(
            (poch(2.5, 1.5) - 4.513_516_668_382_049).abs() < 1e-12,
            "poch(2.5,1.5)"
        );
    }

    #[test]
    fn erfcx_scalar_match_scipy() {
        // scipy.special.erfcx = exp(x^2)*erfc(x) (scaled to avoid erfc underflow).
        assert!((erfcx_scalar(0.0) - 1.0).abs() < 1e-14, "erfcx(0)");
        assert!(
            (erfcx_scalar(1.0) - 0.427_583_576_155_807).abs() < 1e-13,
            "erfcx(1)"
        );
        assert!(
            (erfcx_scalar(10.0) - 0.056_140_992_743_822_59).abs() < 1e-15,
            "erfcx(10)"
        );
        assert!(
            (erfcx_scalar(-1.0) - 5.008_980_080_762_283).abs() < 1e-12,
            "erfcx(-1)"
        );
    }

    #[test]
    fn log_ndtr_scalar_tail_match_scipy() {
        // scipy.special.log_ndtr = log(Phi(x)), stable in BOTH tails. The left tail
        // is the key: Phi(-50)=0 underflows but log_ndtr stays finite (-1254.83).
        assert!(
            (log_ndtr_scalar(0.0) + std::f64::consts::LN_2).abs() < 1e-13,
            "log_ndtr(0)"
        );
        assert!(
            (log_ndtr_scalar(2.0) - -0.023_012_909_328_963_476).abs() < 1e-13,
            "log_ndtr(2)"
        );
        assert!(
            (log_ndtr_scalar(-50.0) - -1_254.831_361_139_419_9).abs() < 1e-9,
            "log_ndtr(-50) tail: {}",
            log_ndtr_scalar(-50.0)
        );
    }

    #[test]
    fn cbrt_handles_negatives_match_scipy() {
        // scipy.special.cbrt = real cube root, so cbrt(-8)=-2 (NOT NaN like the
        // (-8).powf(1/3) trap). Regression guard for the powf-domain class.
        assert_eq!(cbrt(-8.0), -2.0);
        assert_eq!(cbrt(27.0), 3.0);
        assert_eq!(cbrt(0.0), 0.0);
        assert!(
            (cbrt(2.0) - 1.259_921_049_894_873_2).abs() < 1e-15,
            "cbrt(2)"
        );
        assert!(
            (cbrt(-2.0) + 1.259_921_049_894_873_2).abs() < 1e-15,
            "cbrt(-2)"
        );
    }

    #[test]
    fn sinc_match_numpy() {
        // numpy.sinc (normalized): sin(pi*x)/(pi*x), sinc(0)=1, sinc(integer)~0.
        assert_eq!(sinc_scalar(0.0), 1.0);
        assert!(
            (sinc_scalar(0.5) - std::f64::consts::FRAC_2_PI).abs() < 1e-15,
            "sinc(0.5)=2/pi"
        );
        assert!(sinc_scalar(1.0).abs() < 1e-15, "sinc(1)~0");
        assert!(sinc_scalar(2.0).abs() < 1e-15, "sinc(2)~0");
    }

    #[test]
    fn exprel_match_scipy() {
        // scipy.special.exprel = (e^x - 1)/x, with the x->0 limit = 1 (Taylor near
        // 0 avoids the 0/0); expm1 elsewhere for accuracy.
        assert_eq!(exprel_scalar(0.0), 1.0);
        assert!(
            (exprel_scalar(1e-12) - 1.000_000_000_000_5).abs() < 1e-12,
            "exprel(1e-12)"
        );
        assert!(
            (exprel_scalar(1.0) - 1.718_281_828_459_045).abs() < 1e-15,
            "exprel(1)"
        );
        assert!(
            (exprel_scalar(2.0) - 3.194_528_049_465_325).abs() < 1e-15,
            "exprel(2)"
        );
    }

    #[test]
    fn entr_match_scipy() {
        // scipy.special.entr: -x*log(x), with entr(0)=0 and entr(x<0)=-inf.
        assert_eq!(entr_scalar(0.0), 0.0);
        assert_eq!(entr_scalar(-1.0), f64::NEG_INFINITY);
        assert!(
            (entr_scalar(0.5) - 0.346_573_590_279_972_64).abs() < 1e-15,
            "entr(0.5)"
        );
        assert!(
            (entr_scalar(2.0) - -1.386_294_361_119_890_6).abs() < 1e-15,
            "entr(2)"
        );
    }

    #[test]
    fn rel_entr_kl_div_match_scipy() {
        // scipy.special.rel_entr/kl_div edge cases (x=0, y=0, x<0) + a normal value.
        assert_eq!(rel_entr_scalar(0.0, 5.0), 0.0);
        assert_eq!(rel_entr_scalar(0.0, 0.0), 0.0);
        assert_eq!(rel_entr_scalar(2.0, 0.0), f64::INFINITY);
        assert_eq!(rel_entr_scalar(-1.0, 2.0), f64::INFINITY);
        assert!(
            (rel_entr_scalar(2.0, 3.0) - -0.810_930_216_216_328_8).abs() < 1e-15,
            "rel_entr(2,3)"
        );
        assert_eq!(kl_div_scalar(0.0, 5.0), 5.0);
        assert!(
            (kl_div_scalar(2.0, 3.0) - 0.189_069_783_783_671_23).abs() < 1e-15,
            "kl_div(2,3)"
        );
    }

    #[test]
    fn xlogy_xlog1py_match_scipy() {
        // scipy.special.xlogy/xlog1py: x==0 forces 0 (even 0*log(0)=0), else
        // x*log(y); xlog1py uses log1p for small-y precision.
        assert_eq!(xlogy_scalar(0.0, 0.0), 0.0);
        assert_eq!(xlogy_scalar(0.0, 5.0), 0.0);
        assert!(
            (xlogy_scalar(2.0, 3.0) - 2.197_224_577_336_219_6).abs() < 1e-15,
            "xlogy(2,3)"
        );
        assert_eq!(xlogy_scalar(3.0, 0.0), f64::NEG_INFINITY);
        assert_eq!(xlog1py_scalar(0.0, -1.0), 0.0);
        // log1p precision: x*log1p(1e-10) ~ 2e-10, not the 0 a naive (1+y).ln gives.
        assert!(
            (xlog1py_scalar(2.0, 1e-10) - 1.999_999_999_900_000_1e-10).abs() < 1e-25,
            "xlog1py precision"
        );
    }

    #[test]
    fn expit_logit_match_scipy() {
        // scipy.special.expit (stable two-branch form, no overflow) and logit.
        assert_eq!(expit_scalar(0.0), 0.5);
        assert_eq!(expit_scalar(-1000.0), 0.0); // overflow-safe (naive 1/(1+e^1000))
        assert_eq!(expit_scalar(1000.0), 1.0);
        assert!(
            (expit_scalar(2.0) - 0.880_797_077_977_882_3).abs() < 1e-15,
            "expit(2)"
        );
        let m = RuntimeMode::Strict;
        assert!(logit_scalar(0.5, m).unwrap().abs() < 1e-15, "logit(0.5)");
        assert!(
            (logit_scalar(0.880_797_077_977_882_3, m).unwrap() - 2.0).abs() < 1e-12,
            "logit round-trip"
        );
    }

    #[test]
    fn rgamma_scalar_poles_match_scipy() {
        // scipy.special.rgamma: 1/Γ is exactly 0 at the non-positive-integer poles
        // (including 0, where Γ→+inf so 1/Γ→0), and finite elsewhere.
        use crate::gamma::rgamma_scalar;
        let m = RuntimeMode::Strict;
        assert_eq!(rgamma_scalar(0.0, m).unwrap(), 0.0);
        assert_eq!(rgamma_scalar(-1.0, m).unwrap(), 0.0);
        assert_eq!(rgamma_scalar(-2.0, m).unwrap(), 0.0);
        assert!(
            (rgamma_scalar(5.0, m).unwrap() - 0.041_666_666_666_666_664).abs() < 1e-15,
            "rgamma(5)"
        );
        assert!(
            (rgamma_scalar(0.5, m).unwrap() - 0.564_189_583_547_756_3).abs() < 1e-15,
            "rgamma(0.5)"
        );
    }

    #[test]
    fn comb_perm_match_scipy() {
        // scipy.special.comb/perm: direct path, k>n edge, and the log-gamma path.
        use crate::gamma::{comb, perm};
        assert_eq!(comb(10, 3), 120.0);
        assert_eq!(comb(5, 7), 0.0); // k > n -> 0
        assert_eq!(comb(0, 0), 1.0);
        assert_eq!(perm(10, 3), 720.0);
        // Large values use the log-gamma branch; compare with relative tolerance.
        let rel = |g: f64, e: f64| (g - e).abs() / e < 1e-10;
        assert!(
            rel(comb(100, 50), 1.008_913_445_455_641_5e29),
            "comb(100,50)"
        );
        assert!(rel(perm(50, 30), 1.250_115_832_840_612_2e46), "perm(50,30)");
    }

    #[test]
    fn softmax_log_softmax_match_scipy() {
        // scipy.special.softmax/log_softmax, including shift-invariant stability.
        let es = [
            0.090_030_573_170_380_46,
            0.244_728_471_054_797_64,
            0.665_240_955_774_821_8,
        ];
        let s = softmax(&[1.0, 2.0, 3.0]);
        for (g, e) in s.iter().zip(&es) {
            assert!((g - e).abs() < 1e-12, "softmax: {g} vs {e}");
        }
        // Large values overflow a naive exp; result is identical (shift-invariant).
        let s2 = softmax(&[1000.0, 1001.0, 1002.0]);
        for (g, e) in s2.iter().zip(&es) {
            assert!((g - e).abs() < 1e-12, "softmax stable: {g} vs {e}");
        }
        let ls = log_softmax(&[1.0, 2.0, 3.0]);
        let els = [
            -2.407_605_964_444_380_6,
            -1.407_605_964_444_380_4,
            -0.407_605_964_444_380_4,
        ];
        for (g, e) in ls.iter().zip(&els) {
            assert!((g - e).abs() < 1e-12, "log_softmax: {g} vs {e}");
        }
    }

    #[test]
    fn logsumexp_match_scipy_with_stability() {
        // scipy.special.logsumexp. The large-value case would overflow a naive
        // exp(), but the max-subtraction keeps it finite (shift-invariant: the
        // two results differ by exactly 1000).
        assert!(
            (logsumexp(&[0.0, 1.0, 2.0]) - 2.407_605_964_444_38).abs() < 1e-12,
            "lse small"
        );
        assert!(
            (logsumexp(&[1000.0, 1001.0, 1002.0]) - 1_002.407_605_964_444_4).abs() < 1e-9,
            "lse stable"
        );
    }

    #[test]
    fn incomplete_gamma_beta_scalars_match_scipy() {
        // scipy.special regularized incomplete gamma/beta (1.17.1).
        use crate::beta::betainc_scalar;
        use crate::gamma::{gammainc_scalar, gammaincc_scalar};
        let m = RuntimeMode::Strict;
        assert!(
            (gammainc_scalar(2.0, 3.0, m).unwrap() - 0.800_851_726_528_544_2).abs() < 1e-12,
            "gammainc"
        );
        assert!(
            (gammaincc_scalar(2.0, 3.0, m).unwrap() - 0.199_148_273_471_455_8).abs() < 1e-12,
            "gammaincc"
        );
        assert!(
            (betainc_scalar(2.0, 3.0, 0.4, m).unwrap() - 0.524_799_999_999_999_9).abs() < 1e-12,
            "betainc"
        );
    }

    #[test]
    fn boxcox_scalar_match_scipy() {
        // scipy.special.boxcox/inv_boxcox/boxcox1p (1.17.1), including the
        // inv_boxcox NaN edge for a negative base (frankenscipy-9ns59).
        assert!(
            (boxcox_scalar(2.0, 0.5) - 0.828_427_124_746_190_1).abs() < 1e-12,
            "boxcox"
        );
        assert!(
            (boxcox_scalar(2.0, 0.0) - std::f64::consts::LN_2).abs() < 1e-12,
            "boxcox lam0"
        );
        assert!(
            (inv_boxcox_scalar(0.828_427_124_746_190_1, 0.5) - 2.0).abs() < 1e-12,
            "inv_boxcox round-trip"
        );
        assert!(
            inv_boxcox_scalar(5.0, -0.5).is_nan(),
            "inv_boxcox negative base -> NaN"
        );
        assert!(
            (boxcox1p_scalar(1.0, 0.5) - 0.828_427_124_746_190_1).abs() < 1e-12,
            "boxcox1p"
        );
    }

    #[test]
    fn scalar_special_functions_match_scipy() {
        // Golden values from scipy.special (1.17.1) for foundational scalar fns.
        let close = |got: f64, want: f64, name: &str| {
            assert!((got - want).abs() < 1e-12, "{name}: {got} != {want}");
        };
        close(
            crate::error::erf_scalar(0.5),
            0.520_499_877_813_046_5,
            "erf(0.5)",
        );
        close(
            crate::error::erfc_scalar(0.5),
            0.479_500_122_186_953_5,
            "erfc(0.5)",
        );
        close(
            crate::gamma::zeta_scalar(2.0),
            1.644_934_066_848_226_4,
            "zeta(2)",
        );
        close(
            crate::bessel::i0_scalar(1.0),
            1.266_065_877_752_008_2,
            "i0(1)",
        );
        close(
            crate::bessel::k0_scalar(1.0),
            0.421_024_438_240_708_23,
            "k0(1)",
        );
        close(ndtr_scalar(1.0), 0.841_344_746_068_542_9, "ndtr(1)");
        close(ndtri_scalar(0.975), 1.959_963_984_540_054, "ndtri(0.975)");
    }

    #[test]
    fn fresnel_complex_matches_scipy() {
        // frankenscipy: golden (S, C) from scipy.special.fresnel(z) (1.17.1) on
        // complex arguments. Reduces to the real fresnel on the real axis.
        let check = |z: Complex64, ws: (f64, f64), wc: (f64, f64), msg: &str| {
            let (s, c) = fresnel_complex(z);
            let tol = 1e-11 * (1.0 + ws.0.abs() + ws.1.abs() + wc.0.abs() + wc.1.abs());
            assert!(
                (s.re - ws.0).abs() < tol && (s.im - ws.1).abs() < tol,
                "{msg}: S {s:?}"
            );
            assert!(
                (c.re - wc.0).abs() < tol && (c.im - wc.1).abs() < tol,
                "{msg}: C {c:?}"
            );
        };
        check(
            Complex64::new(1.0, 0.0),
            (0.438_259_147_390_354_76, 0.0),
            (0.779_893_400_376_822_6, 0.0),
            "1+0i",
        );
        check(
            Complex64::new(1.0, 1.0),
            (-2.061_888_219_194_839_8, 2.061_888_219_194_839_8),
            (2.555_793_778_102_439, 2.555_793_778_102_439),
            "1+1i",
        );
        check(
            Complex64::new(2.0, -1.0),
            (-15.587_751_104_404_592, 36.725_464_883_991_435),
            (-36.225_687_992_881_66, -16.087_871_374_125_47),
            "2-1i",
        );
        check(
            Complex64::new(0.5, 0.5),
            (-0.136_781_657_729_138_8, 0.136_781_657_729_138_8),
            (0.531_735_955_007_170_3, 0.531_735_955_007_170_3),
            "0.5+0.5i",
        );
        check(
            Complex64::new(3.0, 0.2),
            (0.487_722_973_527_744_1, 0.341_141_758_244_58),
            (0.856_886_597_714_346_8, 0.009_686_387_597_120_427),
            "3+0.2i",
        );
    }

    #[test]
    fn erf_and_fresnel_zeros_match_scipy() {
        // frankenscipy: golden complex zeros from scipy.special.erf_zeros / fresnelc_zeros /
        // fresnels_zeros (1.17.1), first quadrant ordered by |z|.
        let close = |got: &[Complex64], want: &[(f64, f64)], msg: &str| {
            assert_eq!(got.len(), want.len(), "{msg}: len");
            for (g, &(re, im)) in got.iter().zip(want.iter()) {
                assert!(
                    (g.re - re).abs() < 1e-9 && (g.im - im).abs() < 1e-9,
                    "{msg}: got {}+{}i, want {re}+{im}i",
                    g.re,
                    g.im
                );
            }
        };
        close(
            &erf_zeros(3),
            &[
                (1.4506161632, 1.8809430002),
                (2.2446592738, 2.6165751407),
                (2.8397410469, 3.1756280996),
            ],
            "erf_zeros",
        );
        close(
            &fresnelc_zeros(3),
            &[
                (1.7436674862, 0.3057350636),
                (2.6514595973, 0.2529039555),
                (3.3203593363, 0.2239534581),
            ],
            "fresnelc_zeros",
        );
        close(
            &fresnels_zeros(3),
            &[
                (2.0092570118, 0.2885478973),
                (2.8334772325, 0.2442852408),
                (3.4675330835, 0.2184926805),
            ],
            "fresnels_zeros",
        );
        let (zs, zc) = fresnel_zeros(2);
        assert!((zs[0].re - 2.0092570118).abs() < 1e-9, "fresnel_zeros S");
        assert!((zc[0].re - 1.7436674862).abs() < 1e-9, "fresnel_zeros C");
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn kelvin_zeros_match_scipy() {
        // frankenscipy: golden from scipy.special.{ber,bei,ker,kei,berp,beip,kerp,keip}_zeros
        // (1.17.1). Our bisected zeros are at least as accurate as SciPy's specfun values
        // (which leave ~1e-10 residuals), so compare with a 1e-8 tolerance.
        type ZerosCase = (fn(u32) -> Vec<f64>, [f64; 4]);
        let cases: [ZerosCase; 8] = [
            (
                ber_zeros,
                [
                    2.8489178207951396,
                    7.238829447632408,
                    11.673963549647077,
                    16.11356382738789,
                ],
            ),
            (
                bei_zeros,
                [
                    5.026223951953151,
                    9.455406303277154,
                    13.893487852659412,
                    18.33398345577453,
                ],
            ),
            (
                ker_zeros,
                [
                    1.7185429596232313,
                    6.127279134970337,
                    10.562942708257847,
                    15.002688121534597,
                ],
            ),
            (
                kei_zeros,
                [
                    3.9146676068432655,
                    8.344225062948416,
                    12.782557148597336,
                    17.22314372343476,
                ],
            ),
            (
                berp_zeros,
                [
                    6.038710806721278,
                    10.513642514770426,
                    14.968445421840928,
                    19.417574926000736,
                ],
            ),
            (
                beip_zeros,
                [
                    3.772673304934953,
                    8.280987849760043,
                    12.742147523633703,
                    17.19343175251254,
                ],
            ),
            (
                kerp_zeros,
                [
                    2.6658397930175615,
                    7.17212212474824,
                    11.632186394772816,
                    16.08312024940579,
                ],
            ),
            (
                keip_zeros,
                [
                    4.93181194115222,
                    9.404054583281818,
                    13.858269159614109,
                    18.30717293559908,
                ],
            ),
        ];
        for (f, golden) in cases {
            let z = f(4);
            assert_eq!(z.len(), 4, "expected 4 zeros");
            for (got, want) in z.iter().zip(golden.iter()) {
                assert!(
                    (got - want).abs() < 1e-8,
                    "kelvin zero: got {got}, want {want}"
                );
            }
        }
        // kelvin_zeros stacks the eight families in SciPy's order.
        let all = kelvin_zeros(4);
        assert!(
            (all[0][0] - 2.8489178207951396).abs() < 1e-8,
            "ber zero in kelvin_zeros"
        );
        assert!(
            (all[7][3] - 18.30717293559908).abs() < 1e-8,
            "keip zero in kelvin_zeros"
        );
        assert!(ber_zeros(0).is_empty(), "nt=0 -> empty");
    }

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
            assert!(
                (ber(x) - br).abs() <= 1e-10 * br.abs(),
                "ber({x}) = {}, scipy {br}",
                ber(x)
            );
            assert!(
                (bei(x) - bi).abs() <= 1e-10 * bi.abs(),
                "bei({x}) = {}, scipy {bi}",
                bei(x)
            );
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
            (
                0.5,
                1.0,
                0.35490033286757783,
                -0.3428717191311008,
                0.18797346722338337,
                0.9507097283189572,
                1.6914496078608425,
                0.666961949487037,
            ),
            (
                2.0,
                3.0,
                0.09271076642644344,
                -0.1283169622282617,
                -1.1546724379290491e-05,
                0.9989632788568172,
                -70.5023377945093,
                110.8743213409972,
            ),
            (
                -1.0,
                4.0,
                -0.03628154550758465,
                -0.1358395562946222,
                -3.79403296908907e-08,
                1.0000000150962953,
                -2866261.1123123285,
                -421526.9848770045,
            ),
            (
                0.1,
                8.0,
                0.0009029126289383003,
                -0.07107654514487582,
                1.1326489048167856e-29,
                1.0,
                5.468442077084464e27,
                -1.597440107434527e26,
            ),
            (
                5.0,
                -2.0,
                0.0964981126066414,
                0.037351653156368785,
                101670558.35825253,
                -96103547.82551727,
                0.08683899411315939,
                0.036019520041904195,
            ),
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
                assert!(
                    err <= 1e-9,
                    "{name}({re}{im:+}i) = {got:?}, scipy ({wr},{wi}), rel {err:e}"
                );
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
            assert!(
                (got - want).abs() <= tol,
                "owens_t({h},{a}) = {got}, scipy {want}"
            );
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
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1e-9),
                "ndtri({p}) = {got}, scipy {expected}"
            );
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
            assert!(
                ((got - expected) / expected).abs() < 1e-12,
                "erfcinv({y}) = {got}, scipy {expected}"
            );
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
            // Large positive x: log Φ(x) ≈ −Φ(−x) is a tiny negative number.
            // The prior ln(Φ(x)) collapsed to 0 once Φ(x) rounded to 1.0
            // (x ≳ 9), so powernorm/lognorm left tails underflowed.
            (6.0, -9.86587645524372e-10),
            (10.0, -7.61985302416047e-24),
            (15.0, -3.6709661993126986e-51),
            (20.0, -2.7536241186061556e-89),
            (30.0, -4.906713927147908e-198),
        ];
        for (x, expected) in cases {
            let got = log_ndtr_scalar(x);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1e-6),
                "log_ndtr({x}) = {got}, scipy {expected}"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn kelvin_derivatives_match_scipy() {
        // frankenscipy-l3kwr: analytic ber'/bei'/ker'/kei' via complex I₁/K₁
        // replace finite-difference / cancelling-series forms (~1e-8..4e-6 off).
        let cases = [
            (
                1.0,
                -0.06244575217903096,
                0.49739651146809727,
                -0.6946038911006908,
                0.3523699133361705,
            ),
            (
                5.0,
                -3.8453394732621544,
                -4.354140514843111,
                0.017193403828394,
                -0.0008199865436310269,
            ),
            (
                10.0,
                51.19525834615495,
                135.3093016566432,
                -0.00031559693447617284,
                0.0001409138375599196,
            ),
            (
                11.0,
                -94.21185202497774,
                264.11937428720506,
                -6.99034409619794e-05,
                0.00014625254023259207,
            ),
            (
                15.0,
                91.05533316965173,
                -4087.755236845389,
                5.644678075956919e-06,
                -5.882222803057011e-06,
            ),
            (
                20.0,
                -48803.19784717074,
                111855.02522349692,
                -7.501859210700294e-08,
                1.906242756745313e-07,
            ),
            (
                50.0,
                -46498923792943.57,
                -118164845285863.83,
                7.221202712795484e-17,
                -3.141565492509411e-17,
            ),
        ];
        for (x, berp_ref, beip_ref, kerp_ref, keip_ref) in cases {
            assert!(
                (berp(x) - berp_ref).abs() <= 1e-7 * berp_ref.abs().max(1e-6),
                "berp({x})={}",
                berp(x)
            );
            assert!(
                (beip(x) - beip_ref).abs() <= 1e-7 * beip_ref.abs().max(1e-6),
                "beip({x})={}",
                beip(x)
            );
            assert!(
                (kerp(x) - kerp_ref).abs() <= 1e-7 * kerp_ref.abs().max(1e-12),
                "kerp({x})={}",
                kerp(x)
            );
            assert!(
                (keip(x) - keip_ref).abs() <= 1e-7 * keip_ref.abs().max(1e-12),
                "keip({x})={}",
                keip(x)
            );
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
            assert!(
                ((kr - ker_ref) / ker_ref).abs() < 1e-9,
                "ker({x}) = {kr:e}, scipy {ker_ref:e}"
            );
            assert!(
                ((ki - kei_ref) / kei_ref).abs() < 1e-9,
                "kei({x}) = {ki:e}, scipy {kei_ref:e}"
            );
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
            assert!(
                (si - si_ref).abs() < 1e-12,
                "Si({x}) = {si}, scipy {si_ref}"
            );
            assert!(
                (ci - ci_ref).abs() < 1e-12,
                "Ci({x}) = {ci}, scipy {ci_ref}"
            );
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
            assert!(
                ((got - expected) / expected).abs() < 1e-12,
                "expn({n},{x})={got}, scipy {expected}"
            );
        }
        let expi_neg = [
            (-0.5, -0.5597735947761608),
            (-1.0, -0.2193839343955205),
            (-3.0, -0.013048381094197039),
            (-10.0, -4.156968929685325e-06),
        ];
        for (x, expected) in expi_neg {
            let got = expi_scalar(x);
            assert!(
                ((got - expected) / expected).abs() < 1e-12,
                "expi({x})={got}, scipy {expected}"
            );
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
            assert!(
                rel < 1e-13,
                "dawsn({x}) = {got}, scipy {expected}, rel={rel:e}"
            );
        }
        // Propagation check: Im wofz(3+0i) = (2/√π) dawsn(3).
        let w = wofz_scalar(Complex64::new(3.0, 0.0), RuntimeMode::Strict).unwrap();
        assert!(
            (w.im - 0.20115731703760037).abs() < 1e-13,
            "wofz(3).im = {}",
            w.im
        );
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
            assert!(
                (s - sref).abs() < 1e-12,
                "fresnel({x}).S = {s}, scipy {sref}"
            );
            assert!(
                (c - cref).abs() < 1e-12,
                "fresnel({x}).C = {c}, scipy {cref}"
            );
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
            assert!(
                rel < 1e-9,
                "erfi({x}) = {got:e}, scipy {expected:e}, rel={rel:e}"
            );
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
            // Moderate-x band x ∈ [18, 30] that the old x>30 cutoff routed to the
            // catastrophically-cancelling series (struve(0,30) was ~1e-3 rel off).
            // frankenscipy-…: cutoff lowered to 18.
            (0.0, 20.0, 0.09439369808132349),
            (1.0, 25.0, 0.5388036213269298),
            (0.0, 30.0, -0.09609842155416415),
        ];
        for (v, x, expected) in cases {
            let got = struve(v, x);
            let rel = ((got - expected) / expected).abs();
            // ~1e-10 floor at moderate x is inherited from the Y_v asymptotic;
            // still a vast improvement over the prior catastrophic blow-up.
            assert!(
                rel < 1e-9,
                "struve({v},{x}) = {got:e}, scipy {expected:e}, rel={rel:e}"
            );
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
            assert!(
                ((got - want) / want).abs() < 1e-13,
                "expi({x}) = {got}, scipy {want}"
            );
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

    #[test]
    fn smirnovi_matches_scipy_reference_points() {
        // smirnovi(n, p) inverts smirnov(n, ·); now via Illinois (was ~53-step
        // bisection). References from scipy.special.smirnovi (1.17.1), 1e-10.
        let cases = [
            (10, 0.3, 0.23044687172686085),
            (50, 0.1, 0.14839812573875719),
            (100, 0.05, 0.12066568772965514),
            (20, 0.5, 0.12404490303120737),
            (200, 0.2, 0.062610868788876894),
            (5, 0.9, 0.074906194963475184),
            (1, 0.5, 0.5),
        ];
        for (n, p, expected) in cases {
            let got = smirnovi(n, p);
            assert!(
                (got - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                "smirnovi({n}, {p}) = {got}, expected {expected}"
            );
            // Round-trip: smirnov(n, smirnovi(n, p)) == p.
            let back = smirnov(n, got);
            assert!(
                (back - p).abs() <= 1e-9 * p.max(1e-12),
                "smirnovi round-trip: smirnov({n}, {got}) = {back} != {p}"
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
    fn spence_complex_matches_scipy_reference_points() {
        // scipy.special.spence(z) = Li₂(1 - z) for complex z (1.17.1). Previously
        // the complex arm fail-closed with DomainError; it now evaluates the
        // complex dilogarithm. Branch cut (z real ≤ 0) is continuous from below,
        // so on-cut reals carry a negative imaginary part (e.g. spence(-0.4)).
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.5, 0.3, 0.530_961_629_604_081_9, -0.403_436_824_175_195_05),
            (1.3, 0.7, -0.360_353_745_547_686_1, -0.593_105_767_669_333_9),
            (-1.5, 2.0, 0.473_611_502_970_342_27, -3.092_685_656_762_381),
            (2.0, -2.5, -1.269_201_542_040_707_2, 1.498_717_820_213_157_2),
            (-3.0, -1.0, 1.297_286_906_055_633_8, 4.170_387_594_940_814),
            (0.2, 4.0, -1.103_446_907_945_251_8, -2.730_578_639_321_245),
            (5.0, 0.1, -2.370_192_715_222_134, -0.040_233_399_077_564_97),
            (-0.4, 0.0, 2.319_073_036_309_661_4, -1.057_058_706_706_129_2),
            (0.0, 1.5, 0.254_920_670_834_846_6, -1.802_194_354_655_831),
        ];
        for &(re, im, want_re, want_im) in cases {
            let z = SpecialTensor::ComplexScalar(Complex64::new(re, im));
            let got = spence(&z, RuntimeMode::Strict).expect("complex spence");
            let SpecialTensor::ComplexScalar(c) = got else {
                panic!("spence({re}+{im}i) did not return a complex scalar");
            };
            let denom = want_re.hypot(want_im).max(1.0);
            let err = (c.re - want_re).hypot(c.im - want_im) / denom;
            assert!(
                err < 1e-12,
                "spence({re}+{im}i) = {}+{}i, want {want_re}+{want_im}i (relerr {err:e})",
                c.re,
                c.im
            );
        }
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
    fn wrightomega_complex_matches_scipy_reference_points() {
        // Regression: the complex Newton iteration on w + Log(w) = z converged to
        // a non-principal sheet for moderate |z| (the equation has infinitely
        // many roots), so values were wrong by O(1) — wrightomega(-1.5+2i) gave
        // -2.85-0.74i instead of scipy's -0.046+0.229i (maxrel ~3). Now evaluated
        // as ω(z) = W_{K(z)}(e^z) with the unwinding number K. These reference
        // values are scipy.special.wrightomega(z) (1.17.1); they pin the branch,
        // which the W·e^W = z identity tests cannot. Points span the principal
        // strip, |Im z| > π (K ≠ 0), and both deep tails.
        let cases: &[(f64, f64, f64, f64)] = &[
            (0.5, 0.3, 0.759_982_991_983_232_2, 0.130_255_991_959_168_8),
            (
                -1.5,
                2.0,
                -0.046_466_392_009_842_42,
                0.229_077_725_545_113_66,
            ),
            (2.0, -2.5, 1.281_057_065_675_128_1, -1.603_332_445_813_598_5),
            (0.2, 4.0, -0.605_890_980_974_417, 2.155_140_393_085_120_7),
            (0.0, 1.5, 0.392_558_477_115_177_44, 0.549_512_694_868_214_6),
            (-10.0, 4.0, -12.530_965_197_408_364, 0.932_702_135_820_647),
            (
                -3.0,
                -1.0,
                0.027_759_347_673_495_516,
                -0.039_677_501_694_402_574,
            ),
            (0.7, -0.6, 0.831_572_721_398_247_9, -0.277_699_309_499_649_2),
            (-8.0, 8.5, -10.486_305_311_771_067, 5.868_626_712_161_074),
            (5.0, -12.0, 2.603_767_504_361_378, -10.668_583_366_963_013),
        ];
        for &(re, im, wre, wim) in cases {
            let got = wrightomega(
                &SpecialTensor::ComplexScalar(Complex64::new(re, im)),
                RuntimeMode::Strict,
            )
            .expect("wrightomega complex");
            let SpecialTensor::ComplexScalar(c) = got else {
                panic!("wrightomega({re}+{im}i) did not return a complex scalar");
            };
            let denom = wre.hypot(wim).max(1.0);
            let err = (c.re - wre).hypot(c.im - wim) / denom;
            assert!(
                err < 1e-12,
                "wrightomega({re}+{im}i) = {}+{}i, scipy {wre}+{wim}i (relerr {err:e})",
                c.re,
                c.im
            );
        }
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
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy 1.17.1
    fn powm1_and_cosm1_match_scipy_golden_values() {
        let close = |got: f64, want: f64, tol: f64, label: &str| {
            assert!(
                (got - want).abs() <= tol,
                "{label}: got {got}, scipy {want}, tol {tol}"
            );
        };

        let pow_cases = [
            (1.000_000_000_001, 3.0, 3.000_266_701_750_023_6e-12),
            (1.000_000_1, 0.5, 4.999_999_877_919_342e-8),
            (2.0, 3.0, 7.0),
            (0.5, -2.0, 3.0),
            (-2.0, 3.0, -9.0),
        ];
        for (x, y, want) in pow_cases {
            close(powm1_scalar(x, y), want, 1e-18, "powm1");
        }

        let cos_cases = [
            (1.0e-8, -5.000_000_000_000_000_5e-17),
            (0.25, -0.031_087_578_289_355_215),
            (std::f64::consts::PI, -2.0),
            (std::f64::consts::TAU, 0.0),
        ];
        for (x, want) in cos_cases {
            close(cosm1_scalar(x), want, 1e-15, "cosm1");
        }
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
    fn voigt_profile_many_matches_scalar() {
        // Cross the 1<<14 parallel gate (n=20000) and a small serial case (n=7); both bit-identical.
        for &n in &[7usize, 20000usize] {
            let xs: Vec<f64> = (0..n).map(|i| (i as f64 - n as f64 / 2.0) * 0.01).collect();
            for &(sigma, gamma) in &[(1.0, 0.5), (0.0, 1.0), (2.0, 0.0), (0.7, 1.3)] {
                let many = voigt_profile_many(&xs, sigma, gamma);
                assert_eq!(many.len(), n);
                for (i, &x) in xs.iter().enumerate() {
                    assert_eq!(
                        many[i].to_bits(),
                        voigt_profile(x, sigma, gamma).to_bits(),
                        "voigt_profile_many[{i}] x={x} sigma={sigma} gamma={gamma}"
                    );
                }
            }
        }
        assert!(voigt_profile_many(&[], 1.0, 1.0).is_empty());
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
    fn tklmbda_matches_scipy_reference_points() {
        // λ ∉ {0, 1}: the CDF is found by inverting the quantile ppf(p) = x, now
        // via Illinois (was a ~40-step bisection). References from
        // scipy.special.tklmbda (1.17.1); asserted to 1e-12.
        let cases = [
            (0.5, 0.3, 0.65102782778284762),
            (1.2, 0.7, 0.94921331362257177),
            (-0.8, 0.5, 0.22870680067499194),
            (0.3, -0.5, 0.55266461633960517),
            (-1.5, 0.2, 0.13617300619727501),
            (2.0, 0.4, 0.98355336099790946),
            (-0.2, -1.0, 0.47506218943954792),
        ];
        for (x, lam, expected) in cases {
            let got = tklmbda(x, lam);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                "tklmbda({x}, {lam}) = {got}, expected {expected}"
            );
        }
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
    fn modstruve_large_x_matches_scipy() {
        // frankenscipy-kjtmn: modstruve_series caps at 96 terms, so for x≳190
        // it truncated before the dominant terms (modstruve(0,300) gave 2.2e119
        // vs scipy 4.5e128). Large x now via L_v = I_v − correction (DLMF 11.6.2).
        // scipy.special.modstruve 1.17.1.
        let cases = [
            (0.0, 50.0, 2.9325537838493355e20),
            (0.0, 100.0, 1.0737517071310736e42),
            (1.0, 300.0, 4.468381385036954e128),
            (2.0, 200.0, 2.019341357916405e85),
            (0.5, 80.0, 2.4712895036230834e33),
        ];
        for (v, x, expected) in cases {
            let got = modstruve(v, x);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-11,
                "modstruve({v},{x}) = {got:e}, scipy {expected:e}, rel={rel:e}"
            );
        }
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
    fn struve_integral_large_x_matches_high_precision_truth() {
        // |x| > 16 uses the Simpson quadrature (now 64·|x| steps, ~4× cheaper).
        // References are high-precision (mpmath 20-digit) ground truth — NOTE
        // SciPy's own itstruve0 is inaccurate here (itstruve0(50): truth 3.2445,
        // SciPy 6.30), so these lock in fsci's correctness, not SciPy parity.
        type ScalarCase = (fn(f64) -> f64, f64, f64, &'static str);
        let cases: [ScalarCase; 7] = [
            (itstruve0, 20.0, 2.548451692293957, "itstruve0(20)"),
            (itstruve0, 50.0, 3.244522325991607, "itstruve0(50)"),
            (itstruve0, 100.0, 3.720914252854685, "itstruve0(100)"),
            (it2struve0, 20.0, 0.04030795889403845, "it2struve0(20)"),
            (it2struve0, 50.0, 0.01378660549880537, "it2struve0(50)"),
            (itmodstruve0, 20.0, 44758596.19878038, "itmodstruve0(20)"),
            (itmodstruve0, 50.0, 2.962965929947215e20, "itmodstruve0(50)"),
        ];
        for &(func, x, expected, label) in &cases {
            let actual = func(x);
            assert!(
                (actual - expected).abs() <= 1.0e-8 * expected.abs().max(1.0),
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
    fn poch_gamma_pole_ratios_match_scipy() {
        // frankenscipy-7o3a2: poch hits Γ poles at non-positive integers. scipy
        // resolves them; the plain gammaln/gammasgn ratio returned NaN. Golden
        // values from scipy.special.poch 1.17.1.
        // Both arguments hit a pole (integer n) → finite limit:
        let finite = [
            (-2.0, -2.0, 0.08333333333333333_f64), // 1/12
            (-2.0, -1.0, -0.3333333333333333),
            (-3.0, -2.0, 0.05),
            (0.0, -1.0, -1.0),
            (-4.0, -3.0, -0.0047619047619047615),
            (-2.0, 2.0, 2.0),
            (-5.0, 3.0, -60.0),
            (0.0, 2.0, 0.0),
        ];
        for (x, n, want) in finite {
            let got = super::poch(x, n);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "poch({x},{n}) got {got}, want {want}"
            );
        }
        // Numerator-only pole (x+n a non-positive integer, x not) → +inf:
        for (x, n) in [
            (-3.5, 0.5),
            (-3.5, -0.5),
            (-0.5, -0.5),
            (1.0, -2.0),
            (-2.5, -1.5),
            (-4.5, -3.5),
        ] {
            assert!(
                super::poch(x, n) == f64::INFINITY,
                "poch({x},{n}) should be +inf"
            );
        }
        // Denominator-only pole (x a non-positive integer, x+n not) → 0:
        for (x, n) in [(-2.0, 0.5), (-2.0, 3.5), (-3.0, 0.25)] {
            assert_eq!(super::poch(x, n), 0.0, "poch({x},{n}) should be 0");
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

    #[test]
    fn it2j0y0_matches_reference_values() {
        // frankenscipy: the (∫(1-J0)/t, ∫Y0/t) pair was missing. Golden from
        // scipy.special.it2j0y0 1.17.1.
        let cases = [
            (0.5_f64, 0.031006986350915297_f64, 0.26968853860900577_f64),
            (1.0, 0.12116524699506871, 0.39527290169929336),
            (2.0, 0.44191940220810466, 0.16650134540454314),
            (5.0, 1.5403472199872168, -0.0463220552857449),
            (10.0, 2.17786642009344, -0.022987933564673504), // x≤20 series
            (25.0, 3.1082312595852026, 0.0035263393355635633), // x>20 asymptotic
        ];
        for (x, ij0, iy0) in cases {
            let (gj, gy) = super::it2j0y0(x);
            assert!(
                (gj - ij0).abs() <= 1e-8 * ij0.abs().max(1.0),
                "ij0({x}) = {gj}, want {ij0}"
            );
            assert!(
                (gy - iy0).abs() <= 1e-8 * iy0.abs().max(1.0),
                "iy0({x}) = {gy}, want {iy0}"
            );
        }
        // x=0 → (0, -1e300 scipy sentinel); x<0 → ij0 even, iy0 NaN.
        assert_eq!(super::it2j0y0(0.0), (0.0, -1e300));
        let (n0, ny) = super::it2j0y0(-1.0);
        assert!((n0 - 0.12116524699506871).abs() < 1e-9 && ny.is_nan());
    }

    #[test]
    fn it2i0k0_matches_reference_values() {
        // frankenscipy: the (∫(I0-1)/t, ∫K0/t) pair was missing. Golden from
        // scipy.special.it2i0k0 1.17.1 at x≤10 (where scipy is accurate; for
        // larger x our values beat scipy, verified vs mpmath to ~1e-12).
        let cases = [
            (0.5_f64, 0.031495274223672765_f64, 0.6657510156598185_f64),
            (1.0, 0.12897944249456852, 0.2085182909001295),
            (2.0, 0.5673537515648299, 0.03617748753402217),
            (5.0, 7.104776281843781, 0.0005863562610688433),
            (10.0, 340.81536680407874, 1.562928193088453e-06), // x≤12 series for ik0
        ];
        for (x, ii0, ik0) in cases {
            let (g0, gk) = super::it2i0k0(x);
            assert!(
                (g0 - ii0).abs() <= 1e-9 * ii0.abs().max(1.0),
                "ii0({x}) = {g0}, want {ii0}"
            );
            assert!(
                (gk - ik0).abs() <= 1e-9 * ik0.abs().max(1.0),
                "ik0({x}) = {gk}, want {ik0}"
            );
        }
        // x=0 → (0, 1e300 scipy sentinel); x<0 → ii0 even, ik0 NaN.
        assert_eq!(super::it2i0k0(0.0), (0.0, 1e300));
        let (n0, nk) = super::it2i0k0(-1.0);
        assert!((n0 - 0.12897944249456852).abs() < 1e-9 && nk.is_nan());
    }

    #[test]
    fn it2_second_integral_many_match_serial_bit_for_bit() {
        // The order-preserving parallel fan must be bit-identical to a serial map
        // (incl. the NaN/±1e300 sentinels and x<0 branches).
        let xs: Vec<f64> = (-40..=400).map(|i| i as f64 * 0.1).collect();

        let i0k0 = super::it2i0k0_many(&xs);
        let j0y0 = super::it2j0y0_many(&xs);
        assert_eq!(i0k0.len(), xs.len());
        assert_eq!(j0y0.len(), xs.len());
        for (idx, &x) in xs.iter().enumerate() {
            let (si, sk) = super::it2i0k0(x);
            assert_eq!(i0k0[idx].0.to_bits(), si.to_bits(), "it2i0k0 I0 at x={x}");
            assert_eq!(i0k0[idx].1.to_bits(), sk.to_bits(), "it2i0k0 K0 at x={x}");
            let (sj, sy) = super::it2j0y0(x);
            assert_eq!(j0y0[idx].0.to_bits(), sj.to_bits(), "it2j0y0 J0 at x={x}");
            assert_eq!(j0y0[idx].1.to_bits(), sy.to_bits(), "it2j0y0 Y0 at x={x}");
        }
        // Empty input is handled without spawning.
        assert!(super::it2i0k0_many(&[]).is_empty());
        assert!(super::it2j0y0_many(&[]).is_empty());
    }

    #[test]
    fn itairy_pbdv_pbvv_pbwa_many_match_serial_bit_for_bit() {
        // Order-preserving parallel fans must be bit-identical to a serial map
        // (incl. NaN branches and x<0). Covers the itairy 4-tuple and the three
        // parabolic-cylinder (value, derivative) pairs.
        let xs: Vec<f64> = (-30..=120).map(|i| i as f64 * 0.1).collect();

        let air = super::itairy_many(&xs);
        let dv = super::pbdv_many(2.0, &xs);
        let vv = super::pbvv_many(2.0, &xs);
        let wa = super::pbwa_many(1.0, &xs);
        assert_eq!(air.len(), xs.len());
        for (idx, &x) in xs.iter().enumerate() {
            let (a0, a1, a2, a3) = super::itairy(x);
            assert_eq!(air[idx].0.to_bits(), a0.to_bits(), "itairy.0 at x={x}");
            assert_eq!(air[idx].1.to_bits(), a1.to_bits(), "itairy.1 at x={x}");
            assert_eq!(air[idx].2.to_bits(), a2.to_bits(), "itairy.2 at x={x}");
            assert_eq!(air[idx].3.to_bits(), a3.to_bits(), "itairy.3 at x={x}");
            let (d, dp) = crate::pbdv(2.0, x);
            assert_eq!(dv[idx].0.to_bits(), d.to_bits(), "pbdv.0 at x={x}");
            assert_eq!(dv[idx].1.to_bits(), dp.to_bits(), "pbdv.1 at x={x}");
            let (v, vp) = crate::pbvv(2.0, x);
            assert_eq!(vv[idx].0.to_bits(), v.to_bits(), "pbvv.0 at x={x}");
            assert_eq!(vv[idx].1.to_bits(), vp.to_bits(), "pbvv.1 at x={x}");
            let (w, wp) = crate::pbwa(1.0, x);
            assert_eq!(wa[idx].0.to_bits(), w.to_bits(), "pbwa.0 at x={x}");
            assert_eq!(wa[idx].1.to_bits(), wp.to_bits(), "pbwa.1 at x={x}");
        }
        assert!(super::itairy_many(&[]).is_empty());
        assert!(super::pbdv_many(2.0, &[]).is_empty());
        assert!(super::pbvv_many(2.0, &[]).is_empty());
        assert!(super::pbwa_many(1.0, &[]).is_empty());
    }

    #[test]
    fn obl_rad_many_match_serial_bit_for_bit() {
        // Order-preserving parallel fan must equal a serial map of the scalar
        // spheroidal radial functions bit-for-bit (value + derivative).
        let (m, n, c) = (1u32, 2u32, 1.0f64);
        let xs: Vec<f64> = (1..=95).map(|i| i as f64 * 0.01).collect(); // |x|<1
        let r1 = super::obl_rad1_many(m, n, c, &xs);
        let r2 = super::obl_rad2_many(m, n, c, &xs);
        assert_eq!(r1.len(), xs.len());
        for (idx, &x) in xs.iter().enumerate() {
            let (s1v, s1d) = crate::orthopoly::obl_rad1(m, n, c, x);
            assert_eq!(r1[idx].0.to_bits(), s1v.to_bits(), "obl_rad1 val at x={x}");
            assert_eq!(r1[idx].1.to_bits(), s1d.to_bits(), "obl_rad1 der at x={x}");
            let (s2v, s2d) = crate::orthopoly::obl_rad2(m, n, c, x);
            assert_eq!(r2[idx].0.to_bits(), s2v.to_bits(), "obl_rad2 val at x={x}");
            assert_eq!(r2[idx].1.to_bits(), s2d.to_bits(), "obl_rad2 der at x={x}");
        }
        assert!(super::obl_rad1_many(m, n, c, &[]).is_empty());
        assert!(super::obl_rad2_many(m, n, c, &[]).is_empty());
    }

    #[test]
    fn pro_rad_many_match_serial_bit_for_bit() {
        // Order-preserving parallel fan (with the x-invariant cv hoisted) must equal a
        // serial map of the scalar prolate radial functions bit-for-bit (value + deriv).
        // Prolate radial coordinate ξ ≥ 1.
        let (m, n, c) = (1u32, 2u32, 1.0f64);
        let xs: Vec<f64> = (0..95).map(|i| 1.0 + i as f64 * 0.05).collect(); // ξ ≥ 1
        let r1 = super::pro_rad1_many(m, n, c, &xs);
        let r2 = super::pro_rad2_many(m, n, c, &xs);
        assert_eq!(r1.len(), xs.len());
        for (idx, &x) in xs.iter().enumerate() {
            let (s1v, s1d) = crate::orthopoly::pro_rad1(m, n, c, x);
            assert_eq!(r1[idx].0.to_bits(), s1v.to_bits(), "pro_rad1 val at x={x}");
            assert_eq!(r1[idx].1.to_bits(), s1d.to_bits(), "pro_rad1 der at x={x}");
            let (s2v, s2d) = crate::orthopoly::pro_rad2(m, n, c, x);
            assert_eq!(r2[idx].0.to_bits(), s2v.to_bits(), "pro_rad2 val at x={x}");
            assert_eq!(r2[idx].1.to_bits(), s2d.to_bits(), "pro_rad2 der at x={x}");
        }
        assert!(super::pro_rad1_many(m, n, c, &[]).is_empty());
        assert!(super::pro_rad2_many(m, n, c, &[]).is_empty());
    }

    #[test]
    fn itairy_matches_reference_values() {
        // frankenscipy: integrals of Airy functions were missing. Golden
        // (Apt, Bpt, Ant, Bnt) from mpmath (dps=25) — we are far more accurate
        // than scipy's specfun ITAIRY here, so the reference is the true integral.
        let cases = [
            (
                0.5_f64,
                0.14595330491185718,
                0.3653384655095201,
                0.20880954755731607,
                0.2500627554377472,
            ),
            (
                1.0,
                0.2363173419171098,
                0.8727691167380082,
                0.4656739834670686,
                0.37300500963429495,
            ),
            (
                2.0,
                0.3125327557806797,
                2.8734082599825452,
                0.9017728260386064,
                0.19354740799810316,
            ),
            (
                5.0,
                0.33328759030591787,
                321.47831857046515,
                0.7178822045478277,
                0.15873093858143864,
            ),
            (
                8.0,
                0.33333331724248355,
                440065.2580490418,
                0.7839825965711773,
                -0.014756446293227662,
            ),
        ];
        for (x, apt, bpt, ant, bnt) in cases {
            let (a, b, an, bn) = super::itairy(x);
            assert!(
                (a - apt).abs() <= 1e-9 * apt.abs().max(1.0),
                "Apt({x}) = {a}, want {apt}"
            );
            assert!(
                (b - bpt).abs() <= 1e-9 * bpt.abs().max(1.0),
                "Bpt({x}) = {b}, want {bpt}"
            );
            assert!(
                (an - ant).abs() <= 1e-9 * ant.abs().max(1.0),
                "Ant({x}) = {an}, want {ant}"
            );
            assert!(
                (bn - bnt).abs() <= 1e-9 * bnt.abs().max(1.0),
                "Bnt({x}) = {bn}, want {bnt}"
            );
        }
        // Reflection: itairy(-x) = (-Ant, -Bnt, -Apt, -Bpt). x=0 → all zero.
        let (a, b, an, bn) = super::itairy(-1.0);
        assert!((a + 0.4656739834670686).abs() < 1e-9 && (b + 0.37300500963429495).abs() < 1e-9);
        assert!((an + 0.2363173419171098).abs() < 1e-9 && (bn + 0.8727691167380082).abs() < 1e-9);
        assert_eq!(super::itairy(0.0), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn itj0y0_matches_scipy_reference_values() {
        // frankenscipy: integrals of J0/Y0 were missing. Golden (tj, ty) from
        // scipy.special.itj0y0 1.17.1.
        let cases = [
            (0.5_f64, 0.4896805066460451_f64, -0.5617954559146403_f64),
            (1.0, 0.9197304100897596, -0.637069376607422),
            (2.0, 1.4257702931970198, -0.28219285008510336),
            (5.0, 0.7153119177847658, 0.19971938762233765),
            (10.0, 1.067011303956721, 0.24129031832273223),
            (18.0, 0.8133057265998527, 0.01845775812398467), // near series tail
            (25.0, 0.8710149211549791, -0.09360792735177541), // x>20 → asymptotic
        ];
        for (x, tj, ty) in cases {
            let (gj, gy) = super::itj0y0(x);
            assert!(
                (gj - tj).abs() <= 1e-8 * tj.abs().max(1.0),
                "itj0y0({x}).0 = {gj}, want {tj}"
            );
            assert!(
                (gy - ty).abs() <= 1e-8 * ty.abs().max(1.0),
                "itj0y0({x}).1 = {gy}, want {ty}"
            );
        }
        // x < 0: J0 integral odd, Y0 integral NaN. x = 0 → (0, 0).
        let (nj, ny) = super::itj0y0(-2.0);
        assert!((nj + 1.4257702931970198).abs() < 1e-9 && ny.is_nan());
        assert_eq!(super::itj0y0(0.0), (0.0, 0.0));
    }

    #[test]
    fn iti0k0_matches_scipy_reference_values() {
        // frankenscipy: integrals of I0/K0 were missing. Golden (ti, tk) from
        // scipy.special.iti0k0 1.17.1 (small/moderate x where scipy is accurate).
        let cases = [
            (0.5_f64, 0.5105148087974031_f64, 0.9271025209311491_f64),
            (1.0, 1.0865210970235892, 1.2425098486237771),
            (2.0, 2.7750019054282458, 1.4736757343168283),
            (5.0, 31.84866777616979, 1.567387390728352),
            (8.0, 464.2437205810599, 1.5706574852890753),
            (11.0, 7694.03629203664, 1.5707903308855293),
            (15.0, 352620.47527856164, 1.5707962315468567), // x>12 → TK asymptotic
        ];
        for (x, ti, tk) in cases {
            let (gi, gk) = super::iti0k0(x);
            assert!(
                (gi - ti).abs() <= 1e-9 * ti.abs(),
                "iti0k0({x}).0 = {gi}, want {ti}"
            );
            assert!(
                (gk - tk).abs() <= 1e-9 * tk.abs().max(1.0),
                "iti0k0({x}).1 = {gk}, want {tk}"
            );
        }
        // x < 0: I0 integral is odd, K0 integral undefined (NaN). x = 0 → (0, 0).
        let (ni, nk) = super::iti0k0(-1.0);
        assert!((ni + 1.0865210970235892).abs() < 1e-9 && nk.is_nan());
        assert_eq!(super::iti0k0(0.0), (0.0, 0.0));
    }

    #[test]
    fn modfresnel_matches_scipy_reference_values() {
        // frankenscipy: scipy.special.modfresnelp/modfresnelm were missing.
        // Golden (fp.re, fp.im, kp.re, kp.im, fm.re, fm.im, km.re, km.im) from
        // scipy 1.17.1 at x = 0, 1, 2.
        let cases = [
            (
                0.0_f64,
                0.6266570686577501,
                0.6266570686577501,
                0.5,
                0.0,
                0.6266570686577501,
                -0.6266570686577501,
                0.5,
                0.0,
            ),
            (
                1.0,
                -0.27786716924252197,
                0.3163887669343689,
                0.20779404795392425,
                0.11515989377745535,
                -0.27786716924252197,
                -0.3163887669343689,
                0.20779404795392425,
                -0.11515989377745535,
            ),
            (
                2.0,
                0.16519560622453383,
                -0.17811942068600597,
                0.10702394153838506,
                0.08562294793588794,
                0.16519560622453383,
                0.17811942068600597,
                0.10702394153838506,
                -0.08562294793588794,
            ),
        ];
        for (x, fpr, fpi, kpr, kpi, fmr, fmi, kmr, kmi) in cases {
            let (fp, kp) = super::modfresnelp(x);
            let (fm, km) = super::modfresnelm(x);
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
            assert!(
                close(fp.re, fpr) && close(fp.im, fpi),
                "modfresnelp({x}).fp = {fp:?}"
            );
            assert!(
                close(kp.re, kpr) && close(kp.im, kpi),
                "modfresnelp({x}).kp = {kp:?}"
            );
            assert!(
                close(fm.re, fmr) && close(fm.im, fmi),
                "modfresnelm({x}).fm = {fm:?}"
            );
            assert!(
                close(km.re, kmr) && close(km.im, kmi),
                "modfresnelm({x}).km = {km:?}"
            );
        }
    }

    #[test]
    fn besselpoly_matches_scipy_reference_values() {
        // frankenscipy-6ccul: besselpoly(a,λ,ν) = ∫₀¹ xˡ Jᵥ(2ax) dx was missing.
        // Golden values from scipy.special.besselpoly 1.17.1.
        let cases = [
            (0.5, 1.5, 2.0, 0.026212998037515905_f64),
            (1.0, 1.0, 1.0, 0.24449718372863877),
            (0.0, 1.0, 0.0, 0.5), // a=0, ν=0 → 1/(λ+1)
            (0.0, 2.0, 1.0, 0.0), // a=0, ν≠0 → 0
            (2.0, 0.5, 0.0, 0.05519113378203583),
            (0.5, 1.0, -0.5, 0.4238384194901922), // negative non-integer ν (signed Γ)
            (0.5, 1.0, -1.0, -0.15453272353179368), // negative integer ν (reflection)
            (0.5, 1.0, -2.0, 0.02955404113913338),
            (3.0, 2.0, 3.0, 0.1026791022945383),
        ];
        for (a, lam, nu, want) in cases {
            let got = super::besselpoly(a, lam, nu);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "besselpoly({a},{lam},{nu}) = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn zeta_scalar_analytic_continuation_matches_scipy() {
        // frankenscipy-j3ks9: zeta_scalar failed closed to NaN for s <= 1, but
        // scipy.special.zeta is the analytic continuation over the whole real
        // line. Golden values from scipy.special.zeta 1.17.1.
        let cases = [
            (-5.0, -0.003968253968253968),
            (-2.5, 0.008516928777850334),
            (-1.0, -0.08333333333333333), // ζ(-1) = -1/12
            (-0.5, -0.2078862249773546),
            (0.0, -0.5), // ζ(0) = -1/2
            (0.5, -1.4603545088095868),
            (2.0, 1.6449340668482264), // π²/6
            (10.0, 1.0009945751278182),
        ];
        for (s, want) in cases {
            let got = super::zeta_scalar(s);
            assert!(
                (got - want).abs() < 1e-10 * want.abs().max(1.0),
                "zeta({s}) = {got}, expected {want}"
            );
        }
        // Pole at s = 1 → +∞; trivial zero at even negative s ≈ 0.
        assert!(super::zeta_scalar(1.0).is_infinite(), "ζ(1) is the pole");
        assert!(
            super::zeta_scalar(-10.0).abs() < 1e-12,
            "ζ(-10) is a trivial zero"
        );
        assert!(
            (super::zeta_scalar(f64::INFINITY) - 1.0).abs() < 1e-15,
            "ζ(∞) = 1"
        );
    }

    #[test]
    fn mathieu_series_many_matches_serial_and_scipy() {
        // mathieu_cem_many / mathieu_sem_many must be bit-identical to the serial
        // per-x loop (Fourier coeffs computed once, reused across all x).
        let (m, q) = (3u32, 5.0f64);
        let mut s = 0xC0FF_EE12_3456_789Bu64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            180.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let xs: Vec<f64> = (0..1500).map(|_| rng()).collect(); // above the 512 gate
        for &even in &[true, false] {
            let many = if even {
                mathieu_cem_many(m, q, &xs)
            } else {
                mathieu_sem_many(m, q, &xs)
            };
            assert_eq!(many.len(), xs.len());
            for (&x, &(gv, gd)) in xs.iter().zip(&many) {
                let (sv, sd) = if even {
                    crate::orthopoly::mathieu_cem(m, q, x)
                } else {
                    crate::orthopoly::mathieu_sem(m, q, x)
                };
                assert_eq!(gv.to_bits(), sv.to_bits(), "mathieu value mismatch");
                assert_eq!(gd.to_bits(), sd.to_bits(), "mathieu deriv mismatch");
            }
        }
        // Accuracy vs SciPy reference at a fixed point (moderate q).
        let (cv, cd) = mathieu_cem_many(m, q, &[40.0])[0];
        assert!((cv - 0.4016099012381087).abs() < 1e-11, "cem value {cv}");
        assert!((cd - (-2.484064740098615)).abs() < 1e-10, "cem deriv {cd}");
        let (sv, sd) = mathieu_sem_many(m, q, &[40.0])[0];
        assert!((sv - 1.0545080484646563).abs() < 1e-11, "sem value {sv}");
        assert!((sd - 0.32779280697063695).abs() < 1e-10, "sem deriv {sd}");
        // se_0 ≡ 0.
        assert_eq!(mathieu_sem_many(0, q, &[10.0, 20.0]), vec![(0.0, 0.0); 2]);
    }

    #[test]
    fn spheroidal_ang1_many_matches_serial_and_scipy() {
        // pro_ang1_many / obl_ang1_many must be bit-identical to the serial
        // per-x loop (the parallel eval reuses the once-computed cv+coeffs).
        let (m, n, c) = (1u32, 2u32, 1.5f64);
        let mut s = 0xA5A5_1234_9E37_79B9u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 1.8 - 0.9
        };
        let xs: Vec<f64> = (0..1500).map(|_| rng()).collect(); // above the 512 gate
        for &prolate in &[true, false] {
            let many = if prolate {
                pro_ang1_many(m, n, c, &xs)
            } else {
                obl_ang1_many(m, n, c, &xs)
            };
            assert_eq!(many.len(), xs.len());
            for (&x, &(gv, gd)) in xs.iter().zip(&many) {
                let (sv, sd) = if prolate {
                    crate::orthopoly::pro_ang1(m, n, c, x)
                } else {
                    crate::orthopoly::obl_ang1(m, n, c, x)
                };
                assert_eq!(gv.to_bits(), sv.to_bits(), "ang1_many value mismatch");
                assert_eq!(gd.to_bits(), sd.to_bits(), "ang1_many deriv mismatch");
            }
        }
        // Accuracy vs SciPy reference at a fixed point.
        let (pv, pd) = pro_ang1_many(m, n, c, &[0.3])[0];
        assert!(
            (pv - 0.8464455185558086).abs() < 1e-12,
            "pro_ang1 value {pv}"
        );
        assert!(
            (pd - 2.4622196095402016).abs() < 1e-11,
            "pro_ang1 deriv {pd}"
        );
        let (ov, od) = obl_ang1_many(m, n, c, &[0.3])[0];
        assert!(
            (ov - 0.8712913187137773).abs() < 1e-12,
            "obl_ang1 value {ov}"
        );
        assert!(
            (od - 2.7025234895588786).abs() < 1e-11,
            "obl_ang1 deriv {od}"
        );
    }
}
