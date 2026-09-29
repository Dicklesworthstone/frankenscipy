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
/// `log1p(-exp(-2))`, where xsf's `ndtri_exp` switches to `-ndtri(-expm1(y))`.
const NDTRI_EXP_UPPER: f64 = -0.14541345786885906;

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
    // Only a Hardened refusal can fail, so an array without one maps the kernel directly rather
    // than building a Result per element.
    if let SpecialTensor::RealVec(values) = p_tensor
        && (matches!(mode, RuntimeMode::Strict)
            || !values.iter().any(|&p| logit_hardened_refuses(p)))
    {
        return Ok(SpecialTensor::RealVec(
            values.iter().map(|&p| logit_value(p)).collect(),
        ));
    }
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
/// Matches `scipy.special.ndtri(y)`. Under `errstate`, `y` outside `[0, 1]` is SciPy's
/// "domain error".
pub fn ndtri(y_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    // The evaluator choice is read ONCE per call, never per element.
    let unrolled = NDTRI_UNROLL_POLEVL.load(std::sync::atomic::Ordering::Relaxed);
    if unrolled {
        NDTRI_UNROLL_POLEVL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // ndtri cannot fail per element (out of domain is NaN), so a serial batch maps the kernel
    // directly instead of collecting a `Result` per element. The parallel path (>= 2^20
    // elements) is unchanged.
    let value = match y_tensor {
        SpecialTensor::RealVec(values) if values.len() < 1 << 20 => SpecialTensor::RealVec(
            values
                .iter()
                .map(|&y| ndtri_scalar_with(y, unrolled))
                .collect(),
        ),
        _ => map_real_wg("ndtri", y_tensor, mode, |y| {
            Ok(ndtri_scalar_with(y, unrolled))
        })?,
    };
    crate::sf_error_unary("ndtri", y_tensor, mode, |y| {
        (!(0.0..=1.0).contains(&y) && !y.is_nan()).then_some(crate::SpecialErrorCode::Domain)
    })?;
    Ok(value)
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

// The Cephes tables verbatim from xsf/cephes/ndtri.h, which SciPy's ndtri uses. They were
// once each rounded to 16 significant digits, and seven entries of Q0, P1 and P2 then parsed
// to a double 1-2 ulp away from Cephes's. That made ndtri, erfcinv and every quantile built
// on them differ from SciPy in the last bit (frankenscipy-qbwth). The full decimals below
// parse to Cephes's doubles.
#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_P0: [f64; 5] = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];

#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_Q0: [f64; 8] = [
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];

#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_P1: [f64; 9] = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];

#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_Q1: [f64; 8] = [
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];

#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_P2: [f64; 9] = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];

#[allow(clippy::excessive_precision)]
const CEPHES_NDTRI_Q2: [f64; 8] = [
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
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

/// Scalar helper for `ndtri_exp`: xsf's `ndtri_exp`, bit-identical to SciPy 1.17.1 on 150,006
/// points (log-magnitudes 1e-300 to 1e300).
/// - y < -2: Cephes ndtri's tail rational applied directly to sqrt(-2y).
/// - y above log1p(-exp(-2)): -ndtri(-expm1(y)).
/// - Otherwise: ndtri(exp(y)).
///
/// It replaces Acklam's rational with no refinement, which was 1.1e-9 relative off SciPy
/// (frankenscipy-qbwth). NaN and y > 0 give NaN, y = 0 gives +inf and -inf gives -inf, as in
/// SciPy.
#[must_use]
pub fn ndtri_exp_scalar(log_p: f64) -> f64 {
    if log_p < -f64::MAX {
        return f64::NEG_INFINITY;
    }
    if log_p < -2.0 {
        return ndtri_exp_small_y(log_p);
    }
    if log_p > NDTRI_EXP_UPPER {
        return -ndtri_scalar(-log_p.exp_m1());
    }
    ndtri_scalar(log_p.exp())
}

/// xsf's `ndtri_exp_small_y`: `sqrt(-2y)` rather than `sqrt(-2 log p)`, since p itself would
/// underflow.
fn ndtri_exp_small_y(y: f64) -> f64 {
    let x = if y >= -f64::MAX * 0.5 {
        (-2.0 * y).sqrt()
    } else {
        std::f64::consts::SQRT_2 * (-y).sqrt()
    };
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * ndtri_pe::<_, true>(z, &CEPHES_NDTRI_P1) / ndtri_p1e::<_, true>(z, &CEPHES_NDTRI_Q1)
    } else {
        z * ndtri_pe::<_, true>(z, &CEPHES_NDTRI_P2) / ndtri_p1e::<_, true>(z, &CEPHES_NDTRI_Q2)
    };
    x1 - x0
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

/// Hardened refuses p outside (0, 1); NaN passes through as in Strict.
fn logit_hardened_refuses(p: f64) -> bool {
    !(p.is_nan() || (p > 0.0 && p < 1.0))
}

fn logit_scalar(p: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && logit_hardened_refuses(p) {
        record_special_trace(
            "logit",
            mode,
            "domain_error",
            format!("p={p}"),
            "fail_closed",
            "p must be in (0, 1)",
            false,
        );
        return Err(SpecialError {
            function: "logit",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "p must be in (0, 1)",
        });
    }
    Ok(logit_value(p))
}

/// xsf's `logit`, SciPy's kernel. log(p/(1-p)) loses relative precision as p nears 1/2, where
/// the result nears 0, so on [0.3, 0.65] it is log1p(2(p-1/2)) - log1p(-2(p-1/2)). The one
/// expression also gives Strict's edges: -inf at 0, inf at 1, NaN outside [0, 1] and at NaN.
fn logit_value(p: f64) -> f64 {
    if p < 0.3 || p > 0.65 {
        (p / (1.0 - p)).ln()
    } else {
        let s = 2.0 * (p - 0.5);
        s.ln_1p() - (-s).ln_1p()
    }
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

/// Struve function H_v(x), `scipy.special.struve(v, x)`.
///
/// A port of xsf 0d0a593f `cephes/struve.h`, the code SciPy 1.17.1 compiles. It tries the
/// large-x asymptotic expansion (DLMF 11.6.1, `Y_v` plus a divergent correction series), the
/// power series (DLMF 11.2.1, summed in double-double) and the Bessel-function series (DLMF
/// 11.4.19, a sum of `J_{n+v+1/2}`), each with an estimate of its own rounding and truncation
/// error, takes the first whose estimate is below 1e-12 of its value, and otherwise the most
/// accurate of the three if it is within 1e-7. Where the chosen expansion calls no Bessel
/// function (the power series, which covers most of `x ≲ 0.7v + 12` and the band beyond it
/// where the asymptotic expansion is not yet accurate) the result is SciPy's bit for bit.
/// Where it does (`Y_v` in the asymptotic expansion, `J_v` in the Bessel series and at
/// v = -n - 1/2) it carries the difference between fsci's `J_v`/`Y_v` and SciPy's AMOS ones:
/// usually a few ulp, but up to ~1.4e-9 relative for 10 ≲ x < 14, where fsci's `J_v` series
/// loses digits, and a finite value for v ≲ -200 at small x, where AMOS overflows to ±inf.
/// frankenscipy-00cad
///
/// The previous version summed the plain power series below x = 18 and the asymptotic
/// expansion to its smallest term above, which is a truncation error of ~e^{-x} relative to
/// a value that can be much smaller: struve(0.287, 19.57) was 1.25e-8 relative off.
///
/// For x < 0 the function is defined only for integer v, `H_v(-x) = (-1)^{v+1} H_v(x)`; any
/// other v gives NaN. At x = 0 it is 0 for v > -1, 2/π for v = -1, and below that the sign
/// of Γ(v + 3/2) times infinity (NaN where v + 3/2 is a negative integer), as in SciPy.
pub fn struve(v: f64, x: f64) -> f64 {
    xsf_struve::struve_hl(v, x, true)
}

/// Modified Struve function L_v(x), `scipy.special.modstruve(v, x)`.
///
/// The same port of xsf's `cephes/struve.h` as [`struve`], with the signs of the modified
/// function: the asymptotic expansion adds `I_v` (DLMF 11.6.2) and the Bessel series sums
/// `I_{n+v+1/2}` with alternating coefficients. Bit for bit with SciPy wherever the power
/// series is taken; where an expansion calls `I_v`, it carries the difference between fsci's
/// `I_v` and SciPy's Cephes one, up to ~1e-12 relative, and where SciPy overflows to ±inf
/// (v < -1 as x → 0, or v ≲ -200) fsci's `I_v` of negative order can return a finite value
/// instead. frankenscipy-00cad
pub fn modstruve(v: f64, x: f64) -> f64 {
    xsf_struve::struve_hl(v, x, false)
}

/// xsf `cephes/struve.h` (xsf 0d0a593f, as SciPy 1.17.1 compiles it), ported operation for
/// operation so that [`struve`] and [`modstruve`] round as SciPy's do. The Bessel functions it
/// calls are fsci's: `J_v` for xsf's AMOS `cyl_bessel_j`, `Y_v` for `cyl_bessel_y`, and `I_v`
/// for Cephes `iv`. A C `int` conversion keeps x86-64's result, `INT_MIN`, for NaN and out of
/// range values. frankenscipy-00cad
mod xsf_struve {
    use super::xsf_smirnov::Dd;
    use crate::bessel::{iv_scalar, jv_scalar, yv_scalar};
    use crate::gamma::{gammaln_scalar, gammasgn_scalar, rgamma_value};
    use fsci_runtime::RuntimeMode;
    use std::f64::consts::PI;

    const MAXITER: i32 = 10_000;
    /// A double-precision sum stops once a term is below this, relative to the sum.
    const SUM_EPS: f64 = 1e-16;
    /// The double-double power series stops once a term is below this, relative to the sum.
    const SUM_TINY: f64 = 1e-100;
    /// An expansion whose error estimate is below this, relative to its value, is taken.
    const GOOD_EPS: f64 = 1e-12;
    /// Failing that, the best of the three is taken if its estimate is below this...
    const ACCEPTABLE_EPS: f64 = 1e-7;
    /// ...relative to its value, or below this absolutely.
    const ACCEPTABLE_ATOL: f64 = 1e-300;

    /// C `(int)x` as compiled for x86-64: truncation, and `INT_MIN` for NaN or out of range.
    fn c_int(x: f64) -> i32 {
        if x.is_nan() || x >= 2_147_483_648.0 || x <= -2_147_483_649.0 {
            i32::MIN
        } else {
            x as i32
        }
    }

    /// Cephes `lgam`: fsci's `gammaln` is the same kernel, bit for bit (frankenscipy-bmyh2).
    fn lgam(x: f64) -> f64 {
        gammaln_scalar(x, RuntimeMode::Strict).unwrap_or(f64::NAN)
    }

    /// Cephes `gammasgn`.
    fn gammasgn(x: f64) -> f64 {
        gammasgn_scalar(x, RuntimeMode::Strict).unwrap_or(f64::NAN)
    }

    /// xsf `cyl_bessel_y` for real arguments.
    fn bessel_y(v: f64, x: f64) -> f64 {
        yv_scalar(v, x, RuntimeMode::Strict).unwrap_or(f64::NAN)
    }

    /// Large-x expansion for H and L (DLMF 11.6.1), summed up to its divergence point x/2.
    /// Returns the value and its error estimate.
    fn asymp_large_z(v: f64, z: f64, is_h: bool) -> (f64, f64) {
        let sgn: i32 = if is_h { -1 } else { 1 };
        let m = z / 2.0;
        let maxiter = if m <= 0.0 {
            0
        } else if m > f64::from(MAXITER) {
            MAXITER
        } else {
            c_int(m)
        };
        if maxiter == 0 {
            return (f64::NAN, f64::INFINITY);
        }
        if z < v {
            // The error estimate fails here.
            return (f64::NAN, f64::INFINITY);
        }

        let mut term = f64::from(-sgn) / PI.sqrt()
            * (-lgam(v + 0.5) + (v - 1.0) * (z / 2.0).ln()).exp()
            * gammasgn(v + 0.5);
        let mut sum = term;
        let mut maxterm = 0.0_f64;
        for n in 0..maxiter {
            term *= f64::from(sgn * (1 + 2 * n)) * (f64::from(1 + 2 * n) - 2.0 * v) / (z * z);
            sum += term;
            if term.abs() > maxterm {
                maxterm = term.abs();
            }
            if term.abs() < SUM_EPS * sum.abs() || term == 0.0 || !sum.is_finite() {
                break;
            }
        }
        if is_h {
            sum += bessel_y(v, z);
        } else {
            sum += iv_scalar(v, z);
        }
        // Strictly valid only for n > v - 1/2, but it works in practice (xsf).
        (sum, term.abs() + maxterm.abs() * SUM_EPS)
    }

    /// Power series for H and L (DLMF 11.2.1), summed in double-double. It converges from
    /// roughly n > |z|. Returns the value and its error estimate.
    fn power_series(v: f64, z: f64, is_h: bool) -> (f64, f64) {
        let sgn: i32 = if is_h { -1 } else { 1 };

        let mut tmp = -lgam(v + 1.5) + (v + 1.0) * (z / 2.0).ln();
        // A NaN `tmp` takes the scaling arm here and not in xsf; the sum is NaN either way.
        let scaleexp = if !(-600.0..=600.0).contains(&tmp) {
            // Scale the exponent to postpone underflow or overflow.
            let half = tmp / 2.0;
            tmp -= half;
            half
        } else {
            0.0
        };

        let mut term = 2.0 / PI.sqrt() * tmp.exp() * gammasgn(v + 1.5);
        let mut sum = term;
        let mut maxterm = 0.0_f64;

        let mut cterm = Dd::new(term);
        let mut csum = Dd::new(sum);
        let z2 = Dd::new(f64::from(sgn) * z * z);
        let c2v = Dd::new(2.0 * v);

        for n in 0..MAXITER {
            // cdiv = (3 + 2n)(3 + 2n + 2v)
            let cdiv = Dd::new(f64::from(3 + 2 * n));
            let ctmp = Dd::new(f64::from(3 + 2 * n)).add(c2v);
            let cdiv = cdiv.mul(ctmp);

            // cterm *= z2 / cdiv
            cterm = cterm.mul(z2).div(cdiv);
            csum = csum.add(cterm);

            term = cterm.hi;
            sum = csum.hi;

            if term.abs() > maxterm {
                maxterm = term.abs();
            }
            if term.abs() < SUM_TINY * sum.abs() || term == 0.0 || !sum.is_finite() {
                break;
            }
        }

        let mut err = term.abs() + maxterm.abs() * 1e-22;
        if scaleexp != 0.0 {
            sum *= scaleexp.exp();
            err *= scaleexp.exp();
        }
        if sum == 0.0 && term == 0.0 && v < 0.0 && !is_h {
            // Spurious underflow.
            return (f64::NAN, f64::INFINITY);
        }
        (sum, err)
    }

    /// Bessel-function series for H and L (DLMF 11.4.19). Returns the value and its error
    /// estimate.
    fn bessel_series(v: f64, z: f64, is_h: bool) -> (f64, f64) {
        if is_h && v < 0.0 {
            // Less reliable in this region.
            return (f64::NAN, f64::INFINITY);
        }

        let mut sum = 0.0_f64;
        let mut maxterm = 0.0_f64;
        let mut term = 0.0_f64;
        let mut cterm = (z / (2.0 * PI)).sqrt();

        for n in 0..MAXITER {
            let order = f64::from(n) + v + 0.5;
            let divisor = f64::from(n) + 0.5;
            if is_h {
                term = cterm * jv_scalar(order, z) / divisor;
                cterm *= z / 2.0 / f64::from(n + 1);
            } else {
                term = cterm * iv_scalar(order, z) / divisor;
                cterm *= -z / 2.0 / f64::from(n + 1);
            }
            sum += term;
            if term.abs() > maxterm {
                maxterm = term.abs();
            }
            if term.abs() < SUM_EPS * sum.abs() || term == 0.0 || !sum.is_finite() {
                break;
            }
        }

        let mut err = term.abs() + maxterm.abs() * 1e-16;
        // Account for potential underflow of the Bessel functions.
        err += 1e-300 * cterm.abs();
        (sum, err)
    }

    /// xsf `struve_hl`: H_v(z) when `is_h`, else L_v(z).
    pub(super) fn struve_hl(v: f64, z: f64, is_h: bool) -> f64 {
        if z < 0.0 {
            let n = c_int(v);
            if v == f64::from(n) {
                let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
                return sign * struve_hl(v, -z, is_h);
            }
            return f64::NAN;
        } else if z == 0.0 {
            if v < -1.0 {
                return gammasgn(v + 1.5) * f64::INFINITY;
            } else if v == -1.0 {
                return 2.0 / PI.sqrt() * rgamma_value(0.5, RuntimeMode::Strict);
            }
            return 0.0;
        }

        // v = -n - 1/2, n > 0: a spherical Bessel function.
        let n = c_int(-v - 0.5);
        if f64::from(n) == -v - 0.5 && n > 0 {
            let order = f64::from(n) + 0.5;
            if is_h {
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                return sign * jv_scalar(order, z);
            }
            return iv_scalar(order, z);
        }

        // The asymptotic expansion is not worth trying below z ~ 0.7v + 12.
        let asymp = if z >= 0.7 * v + 12.0 {
            let (value, err) = asymp_large_z(v, z, is_h);
            if err < GOOD_EPS * value.abs() {
                return value;
            }
            (value, err)
        } else {
            (f64::NAN, f64::INFINITY)
        };

        let power = power_series(v, z, is_h);
        if power.1 < GOOD_EPS * power.0.abs() {
            return power.0;
        }

        // The Bessel series tends to fail for |z| >~ |v|.
        let bessel = if z.abs() < v.abs() + 20.0 {
            let (value, err) = bessel_series(v, z, is_h);
            if err < GOOD_EPS * value.abs() {
                return value;
            }
            (value, err)
        } else {
            (f64::NAN, f64::INFINITY)
        };

        // The best of the three, if it is acceptable.
        let mut best = asymp;
        if power.1 < best.1 {
            best = power;
        }
        if bessel.1 < best.1 {
            best = bessel;
        }
        if best.1 < ACCEPTABLE_EPS * best.0.abs() || best.1 < ACCEPTABLE_ATOL {
            return best.0;
        }

        // Maybe it really is an overflow.
        let mut tmp = -lgam(v + 1.5) + (v + 1.0) * (z / 2.0).ln();
        if !is_h {
            tmp = tmp.abs();
        }
        if tmp > 700.0 {
            return f64::INFINITY * gammasgn(v + 1.5);
        }
        f64::NAN
    }
}

/// Integral of the Struve function H_0 from 0 to x.
///
/// Matches `scipy.special.itstruve0`. Because H_0 is odd, SciPy's integral is
/// even in x: integrating from 0 to a negative endpoint returns the same value
/// as integrating to the positive endpoint.
pub fn itstruve0(x: f64) -> f64 {
    // |x| ≤ 6: the term-by-term integral of the H_0 series (7.7e-16 at 6). Past that the
    // alternating series cancels (7e-15 by 8, 1e-12 by 14), so the rest takes the Laplace
    // form, which holds 1.4e-15 of the exact integral from 6 up (frankenscipy-ch0z1).
    let ax = x.abs();
    if ax <= ITSTRUVE0_SERIES_MAX {
        struve0_integral_series(x, -1.0)
    } else if ax.is_finite() {
        itstruve0_laplace(ax)
    } else {
        f64::NAN
    }
}

/// Integral of the modified Struve function L_0 from 0 to x.
///
/// Matches `scipy.special.itmodstruve0`; L_0 has the same odd symmetry as H_0
/// for real inputs, so the integral is even in x.
pub fn itmodstruve0(x: f64) -> f64 {
    // L_0's series has all-positive terms, so its term-by-term integral cannot cancel: summed
    // in f64 it is within 1.3e-15 of the exact integral at every x. It costs about x terms,
    // though, so from |x| = 45 the asymptotic form below takes over. The Simpson quadrature
    // this replaced past 16 was 3e-10 off and ~100x slower, and SciPy's own asymptotic past 20
    // is 7e-9 off at 25 (frankenscipy-ch0z1).
    let ax = x.abs();
    if ax < ITMODSTRUVE0_ASYMPTOTIC_MIN {
        struve0_integral_series(x, 1.0)
    } else if ax.is_finite() {
        itmodstruve0_asymptotic(ax)
    } else {
        f64::NAN
    }
}

/// From this |x| on, [`itmodstruve0`] takes [`itmodstruve0_asymptotic`].
const ITMODSTRUVE0_ASYMPTOTIC_MIN: f64 = 45.0;

/// `bₙ` of `∫₀ˣ I₀ ~ eˣ/√(2πx) Σ bₙ x⁻ⁿ`: `bₙ = Σ_{k≤n} aₖ (k + ½)_{n−k}`, with
/// `aₖ = ((2k−1)!!)² / (k! 8ᵏ)` the coefficients of I₀'s own large-argument expansion. Each
/// `aₖ t^{−k−½} eᵗ` integrates by parts to `eˣ x^{−k−½} Σ_m (k+½)_m x⁻ᵐ`. Summed in mpmath at
/// 60 digits and rounded once.
#[allow(clippy::excessive_precision)]
const ITI0_ASYMPTOTIC: [f64; 20] = [
    1.0,
    0.625,
    1.0078125,
    2.5927734375,
    9.186859130859375,
    41.56797409057617,
    229.19635891914368,
    1491.5040604770184,
    11192.354495578911,
    95159.3937421203,
    904124.2576904121,
    9493856.041645449,
    109182382.56943358,
    1364798039.8733943,
    18424892376.71708,
    267161772321.70163,
    4141013723937.8687,
    68326776514564.37,
    1195719014944093.0,
    2.21208056127209e16,
];

/// `∫₀ˣ L₀` for `x ≥ 45`: `∫₀ˣ I₀ − (2/π)(ln 2x + γ)`.
///
/// `I₀ − L₀ = (2/π) ∫₀¹ e^{−xt} (1−t²)^{−1/2} dt` (DLMF 11.5.4), so the difference of the two
/// integrals grows only like `(2/π)(ln 2x + γ)`, next to a value near `e^x`. `∫₀ˣ I₀` is 20
/// terms of its asymptotic series; the smallest term is below 1e-16 of the sum from x = 45. The
/// exponential is formed as `e^{x/2} · e^{x/2}/√(2πx)`, so it does not overflow before the value
/// does, near 713. Against the exact series at 50 digits the worst over x in [45, 712] is
/// 5.5e-16.
fn itmodstruve0_asymptotic(x: f64) -> f64 {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let inv_x = 1.0 / x;
    let mut sum = 0.0;
    for &b in ITI0_ASYMPTOTIC.iter().rev() {
        sum = sum * inv_x + b;
    }
    let half = (0.5 * x).exp();
    half * (half / (2.0 * PI * x).sqrt()) * sum - (2.0 / PI) * ((2.0 * x).ln() + EULER_GAMMA)
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
        -(x.ln() - (std::f64::consts::LN_2 - EL)) * ti + s2 + s3
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
        let l = x.ln() - (std::f64::consts::LN_2 - EL);
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
        let l = x.ln() - (std::f64::consts::LN_2 - EL);
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
        let ij0 = x.ln() - (std::f64::consts::LN_2 - EL) + rc * jtot;
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
        let ty = (2.0 / pi) * ((x.ln() - (std::f64::consts::LN_2 - EL)) * tj - s2 - s3);
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
/// Matches `scipy.special.it2struve0`, including its convention for negative `x`:
/// `π/2 + ∫₀^{|x|} H_0(t)/t dt` (`∫₀^∞ H_0(t)/t dt = π/2`), which is `π − it2struve0(|x|)`.
/// A non-finite `x` gives NaN, as SciPy's does.
pub fn it2struve0(x: f64) -> f64 {
    // Below |x| = 1.5 it is π/2 ∓ the term-by-term series of ∫₀^{|x|} H_0/t, within 6.5e-16 of
    // the exact integral. For positive x that difference cancels as the value falls towards
    // 2/(πx): 1.2e-15 off by 2, 1.3e-14 by 4.8 and 1e-10 by 16, where a Simpson quadrature at
    // 64|x| Struve evaluations per call took over (5e-9 off by 300, 3e-2 at 1e5, where its step
    // cap bit). Now [1.5, 6) is a Chebyshev fit of x·it2struve0(x) and [6, ∞) the
    // amplitude-phase form of [`it2struve0_tail`]: over 25000 points in [1.5, 1e5] both hold
    // 6.2e-16 of the exact integral, and 1.5e-16 over [−300, −1.5]. SciPy's own ITTH0 is 8e-11
    // off below 16 and 2e-7 between 16 and 40 (frankenscipy-ch0z1).
    let ax = x.abs();
    if ax < IT2STRUVE0_SERIES_MAX {
        let correction = struve0_over_t_integral_series(ax);
        return if x.is_sign_negative() {
            std::f64::consts::FRAC_PI_2 + correction
        } else {
            std::f64::consts::FRAC_PI_2 - correction
        };
    }
    let tail = if ax < IT2STRUVE0_TAIL_MIN {
        chebyshev_sum(&IT2STRUVE0_MID, (ax - 3.75) / 2.25) / ax
    } else if ax.is_finite() {
        it2struve0_tail(ax)
    } else {
        return f64::NAN;
    };
    if x.is_sign_negative() {
        PI - tail
    } else {
        tail
    }
}

/// Below this |x| [`it2struve0`] is `π/2 ∓` the series of `∫₀^{|x|} H_0/t`.
const IT2STRUVE0_SERIES_MAX: f64 = 1.5;

/// From this |x| on, [`it2struve0`] takes [`it2struve0_tail`]; below it (down to
/// [`IT2STRUVE0_SERIES_MAX`]) the Chebyshev fit [`IT2STRUVE0_MID`].
const IT2STRUVE0_TAIL_MIN: f64 = 6.0;

/// `x · it2struve0(x)` on `[1.5, 6]` as a Chebyshev series in `t = (x − 3.75) / 2.25`. Fitted at
/// 48 first-kind nodes to the exact integral (`π/2 −` the term-by-term series summed in mpmath at
/// 55 + 0.9x digits) and rounded once; the first dropped coefficient is 1.3e-18. `π/2 − ∫₀ˣ H_0/t`
/// is entire in `x`, so the coefficients fall faster than any geometric rate. The factor `x`
/// flattens the `2/(πx)` decay, so the sum does not cancel.
const IT2STRUVE0_MID: [f64; 20] = [
    0.6410305650603118,
    -0.21765286412528376,
    0.28001649276879725,
    0.052872010050828984,
    -0.04926783699426264,
    0.0014014551185209064,
    0.0017272693900763187,
    -7.583023703235011e-05,
    -3.329874912711984e-05,
    1.4972247475077957e-06,
    4.12169117122096e-07,
    -1.7593978429384766e-08,
    -3.5595867595764084e-09,
    1.4123629051813008e-10,
    2.2670581082508636e-11,
    -8.318456092471617e-13,
    -1.1085555484908106e-13,
    3.763722411325794e-15,
    4.2919021285321774e-16,
    -1.3519785850415636e-17,
];

/// `(A, P, Q)` of [`it2struve0_tail`] as Chebyshev series in `t = 2w − 1`, `w = 6/x ∈ (0, 1]`,
/// one row per order. Fitted at 64 first-kind nodes, where both Laplace integrals were evaluated
/// in mpmath at 36 digits, and rounded once. The dropped coefficients sum to below 3e-17.
#[rustfmt::skip]
const IT2STRUVE0_TAIL: [(f64, f64, f64); 27] = [
    (0.6345927470106502, 0.6127241376161029, 0.4767374481747674),
    (-0.0026588523281227314, 0.03933602309893508, -0.08910295572370847),
    (-0.0006016950537949631, -0.008928688130746343, -0.000557559214605177),
    (3.2105781285491515e-05, 0.0003902015758155304, 0.0009846060223677705),
    (1.428419138372598e-06, 9.477156954012659e-05, -0.00011614456041810242),
    (-5.016063575216223e-07, -2.4075304813807393e-05, -2.8690704323628355e-06),
    (4.8172278843853694e-08, 2.0655072923176477e-06, 3.83151917745201e-06),
    (2.255376798745148e-09, 3.493019810143679e-07, -7.752759228164026e-07),
    (-1.7372324533173867e-09, -1.720203123019696e-07, 4.653356094509801e-08),
    (3.3192488274382127e-10, 3.296046935208787e-08, 2.2024652309229824e-08),
    (-2.14953166709023e-11, -1.4329612308970617e-09, -9.17373119418164e-09),
    (-7.939238050821085e-12, -1.3737124225250654e-09, 1.8137011098703231e-09),
    (3.4617436050223598e-12, 5.745714344992165e-10, -8.13180113737295e-11),
    (-7.264216524963928e-13, -1.2465811580615054e-10, -8.940787131943638e-11),
    (5.698714745857356e-14, 8.457135849341004e-12, 4.102472623283286e-11),
    (2.3779762804221853e-14, 5.913655351959128e-12, -1.0186562480125185e-11),
    (-1.3026088275504654e-14, -3.2185866518862703e-12, 1.1034719927297996e-12),
    (3.612542560886082e-15, 9.391144737200432e-13, 3.6488321492202567e-13),
    (-5.515270094798337e-16, -1.4954935977341705e-13, -2.6609485254883205e-13),
    (-4.3917427836851755e-17, -1.5453947173176308e-14, 9.319071687082911e-14),
    (6.586961957784252e-17, 2.1996605616329332e-14, -2.013223474611025e-14),
    (-2.7049366162974344e-17, -9.553116770712703e-15, 8.262462789856021e-16),
    (7.057867499811132e-18, 2.660226057476041e-15, 1.6635483520693059e-15),
    (-9.358773538941577e-19, -3.832625757501277e-16, -9.706518484025748e-16),
    (-2.18871382347544e-19, -8.758905595111514e-17, 3.414513473980099e-16),
    (2.0951615652262967e-19, 9.234929192027303e-17, -7.791087930958881e-17),
    (-8.852697709642913e-20, -4.182190512192065e-17, 3.764211588296043e-18),
];

/// `it2struve0(x)` for `x ≥ 6` as `(A + (P sin x + Q cos x) / √x) / x`, with `A`, `P`, `Q` the
/// Chebyshev series [`IT2STRUVE0_TAIL`] in `w = 6/x`.
///
/// `H_0 = Y_0 + K_0` with `K_0(t) = H_0(t) − Y_0(t) = (2/π) ∫₀^∞ e^{−ts} (1+s²)^{−1/2} ds`
/// (DLMF 11.5.2), and each part of `∫ₓ^∞ H_0(t)/t dt` is one Laplace transform:
/// - `∫ₓ^∞ K_0(t)/t dt = (2/π) ∫₀^∞ e^{−xs} asinh(s)/s ds`, non-oscillatory and near `2/(πx)`.
///   This is `A/x`.
/// - `∫ₓ^∞ H_0⁽¹⁾(t)/t dt = (2/π) e^{ix} ∫₀^∞ e^{−xs} (−i) acosh(1+is)/(s−i) ds`, from the
///   representation of `H_0⁽¹⁾` that [`itstruve0_laplace`] uses: differentiating both sides in
///   `x` gives back `−H_0⁽¹⁾(x)/x`. It is `e^{ix} x^{−3/2} (P + iQ)`, and `∫ₓ^∞ Y_0(t)/t dt` is
///   its imaginary part, `x^{−3/2} (P sin x + Q cos x)`.
///
/// Both integrals agree with the exact series to 1e-41 on [0.5, 300]. `A`, `P` and `Q` are smooth
/// on `w ∈ (0, 1]` (at `w → 0` they tend to `2/π` and `√(2/π) e^{iπ/4}`). The phase is `x`
/// itself rather than `x − π/4`, so it carries no rounding: the π/4 is folded into `P` and `Q`.
fn it2struve0_tail(x: f64) -> f64 {
    let inv_x = 1.0 / x;
    let t = 2.0 * (IT2STRUVE0_TAIL_MIN * inv_x) - 1.0;
    let t2 = 2.0 * t;
    // Three Clenshaw recurrences side by side.
    let (mut a1, mut a2, mut p1, mut p2, mut q1, mut q2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(a, p, q) in IT2STRUVE0_TAIL[1..].iter().rev() {
        (a1, a2) = (a + t2 * a1 - a2, a1);
        (p1, p2) = (p + t2 * p1 - p2, p1);
        (q1, q2) = (q + t2 * q1 - q2, q1);
    }
    let (a0, p0, q0) = IT2STRUVE0_TAIL[0];
    let a = a0 + t * a1 - a2;
    let p = p0 + t * p1 - p2;
    let q = q0 + t * q1 - q2;
    let (sin_x, cos_x) = x.sin_cos();
    inv_x * (a + inv_x.sqrt() * (p * sin_x + q * cos_x))
}

/// Clenshaw sum `Σₖ cₖ Tₖ(t)` of a Chebyshev series, `c₀` taken at full weight.
fn chebyshev_sum(c: &[f64], t: f64) -> f64 {
    let t2 = 2.0 * t;
    let (mut b1, mut b2) = (0.0, 0.0);
    for &ck in c[1..].iter().rev() {
        (b1, b2) = (ck + t2 * b1 - b2, b1);
    }
    c[0] + t * b1 - b2
}

/// Above this |x| [`itstruve0`] takes [`itstruve0_laplace`] rather than the series.
const ITSTRUVE0_SERIES_MAX: f64 = 6.0;

/// The 24-point Gauss-Laguerre rule for the weight `e^{−u}`, as `(node, weight)`, cut after the
/// 18th node: the six dropped weights are below 6e-18 and their integrand in
/// [`itstruve0_laplace`] is bounded, so dropping them moves nothing. Computed at 60 digits (the
/// roots of `L₂₄` polished by Newton; `wᵢ = xᵢ / (25² L₂₅(xᵢ)²)`) and rounded once.
const GAUSS_LAGUERRE_24: [(f64, f64); 18] = [
    (0.05901985218150798, 0.14281197333478185),
    (0.31123914619848375, 0.2587741075174239),
    (0.7660969055459367, 0.2588067072728698),
    (1.4255975908036131, 0.18332268897777804),
    (2.2925620586321904, 0.0981662726299189),
    (3.3707742642089977, 0.040732478151408645),
    (4.665083703467171, 0.013226019405120156),
    (6.1815351187367655, 0.0033693490584783036),
    (7.927539247172152, 0.0006721625640935479),
    (9.912098015077706, 0.00010446121465927518),
    (12.146102711729766, 1.2544721977993332e-05),
    (14.642732289596674, 1.15131581273728e-06),
    (17.417992646508978, 7.96081295913363e-08),
    (20.491460082616424, 4.0728589875499996e-09),
    (23.887329848169735, 1.507008226292585e-10),
    (27.635937174332717, 3.917736515058451e-12),
    (31.776041352374722, 6.894181052958085e-14),
    (36.35840580165162, 7.819800382459448e-16),
];

/// The 24-point generalised Gauss-Laguerre rule for the weight `u^{−1/2} e^{−u}`, cut after the
/// 18th node as [`GAUSS_LAGUERRE_24`] is; weights `Γ(24.5) xᵢ / (24! · 25² · L₂₅^{(−1/2)}(xᵢ)²)`.
const GAUSS_LAGUERRE_24_HALF: [(f64, f64); 18] = [
    (0.02543799658568936, 0.6220020607559261),
    (0.22910231649262433, 0.5079230853295182),
    (0.6372902787326687, 0.33840894389128223),
    (1.2517406323627465, 0.18364459415857035),
    (2.075112909852381, 0.0809593539692077),
    (3.1110524551477132, 0.0288899231499622),
    (4.3642830769353065, 0.008306009823955105),
    (5.840733271323608, 0.0019127846396388305),
    (7.547704680023454, 0.00035030086360234567),
    (9.494095330026488, 5.0571980554969775e-05),
    (11.690695926056073, 5.694517383469696e-06),
    (14.150586187285759, 4.937317987339501e-07),
    (16.889671928527108, 3.2450282717915394e-08),
    (19.927425875242463, 1.5860934990330765e-09),
    (23.287932824879917, 5.630593075676338e-11),
    (27.001406056472355, 1.4093865163091777e-12),
    (31.106464709046566, 2.3951797309583587e-14),
    (35.653703516328214, 2.630319245316817e-16),
];

/// `∫₀ˣ H₀(t) dt` for `x > 6` as two Laplace transforms.
///
/// `H₀ = Y₀ + K₀` with `K₀(t) = H₀(t) − Y₀(t) = (2/π) ∫₀^∞ e^{−ts} (1+s²)^{−1/2} ds`
/// (DLMF 11.5.2). Integrating in `t`, with `∫₀^∞ Y₀ = 0`:
/// - `∫₀ˣ K₀ = (2/π)(ln 2x + γ + I(x))`, where `I(x) = ∫₀^∞ e^{−xs} (1 − (1+s²)^{−1/2}) / s ds`
///   is non-oscillatory.
/// - `∫₀ˣ Y₀ = −∫ₓ^∞ Y₀ = −Im[(2i/π) e^{ix} ∫₀^∞ e^{−xs} ds / ((1+is) √(2is − s²))]`. This
///   comes from rotating `H₀⁽¹⁾(t) = (2/(πi)) ∫₁^∞ e^{its} (s²−1)^{−1/2} ds` onto `s = 1 + iσ`.
///
/// With `u = xs` both are Gauss-Laguerre integrals. `I` uses the `e^{−u}` rule on
/// `(s/x) / (r(1+r))`, where `r = √(1+s²)`; this is `1 − 1/r` written so that it does not
/// cancel. The oscillatory part uses the `u^{−1/2} e^{−u}` rule and absorbs the `√s`
/// singularity. For `x ≥ 6` the integrands' singularities, at `u = ±ix` and `u = 2ix`, are far
/// enough from the positive axis for 18 nodes. Against the exact series at 40 + 0.9x digits,
/// the worst of 84 points in `[6, 300]` is 1.4e-15.
fn itstruve0_laplace(x: f64) -> f64 {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let inv_x = 1.0 / x;
    let mut i_sum = 0.0;
    for &(u, w) in &GAUSS_LAGUERRE_24 {
        let s = u * inv_x;
        let r = (1.0 + s * s).sqrt();
        i_sum += w * (s * inv_x) / (r * (1.0 + r));
    }
    // J = Σ w / ((1 + is) √(2i − s)); √(2i − s) is the principal root a + ib.
    let (mut j_re, mut j_im) = (0.0, 0.0);
    for &(u, w) in &GAUSS_LAGUERRE_24_HALF {
        let s = u * inv_x;
        let m = (s * s + 4.0).sqrt();
        let (a, b) = ((0.5 * (m - s)).sqrt(), (0.5 * (m + s)).sqrt());
        let (c, d) = (a - s * b, b + s * a);
        let inv_den = 1.0 / (c * c + d * d);
        j_re += w * c * inv_den;
        j_im -= w * d * inv_den;
    }
    // ∫ₓ^∞ Y₀ = Im[(2i/π) e^{ix} J / √x] = (2/π)(cos x · J_re − sin x · J_im) / √x.
    let (sin_x, cos_x) = x.sin_cos();
    let y0_tail = (2.0 / PI) * inv_x.sqrt() * (cos_x * j_re - sin_x * j_im);
    (2.0 / PI) * ((2.0 * x).ln() + EULER_GAMMA + i_sum) - y0_tail
}

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
/// caller applies scipy's `π/2 ∓ correction` sign convention, below
/// [`IT2STRUVE0_SERIES_MAX`] only.
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

/// Simple Lanczos gamma function.
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
/// Matches `scipy.special.zeta(s, a)` (the two-argument form) bit for bit: this is the
/// Cephes `zeta(x, q)` SciPy calls, checked on 30,000 points including negative `a`.
///
/// - s = 1 is a pole (+inf), and s < 1 is outside the domain (NaN). This implementation used
///   to return +inf for every s <= 1.
/// - A nonpositive-integer `a` is a pole (+inf). A negative non-integer `a` needs an integer
///   `s`, so that `(a + k)^-s` stays real; otherwise the result is NaN.
/// - For a > 1e8 it uses the asymptotic (1/(s-1) + 1/(2a))·a^(1-s) (DLMF 25.11.43).
/// - Otherwise it sums directly until a + k > 9 (at least nine terms), then applies up to
///   twelve Euler-Maclaurin Bernoulli corrections, stopping once a term is below 2^-53 of
///   the sum.
///
/// It replaces an N = 20 Euler-Maclaurin sum with four Bernoulli terms (~1e-13) and a separate
/// shift recurrence for negative `a` (frankenscipy-re34v).
pub fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    const MACHEP: f64 = 1.110_223_024_625_156_5e-16; // 2^-53, Cephes' MACHEP
    // (2k)! / B_2k
    #[allow(clippy::excessive_precision)]
    const EULER_MACLAURIN: [f64; 12] = [
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9,
        7.47242496e10,
        -2.950130727918164224e12,
        1.1646782814350067249e14,
        -4.5979787224074726105e15,
        1.8152105401943546773e17,
        -7.1661652561756670113e18,
    ];
    if s.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    if s < 1.0 {
        return f64::NAN;
    }
    if a <= 0.0 {
        if a == a.floor() {
            return f64::INFINITY;
        }
        if s != s.floor() {
            return f64::NAN;
        }
    }
    if a > 1e8 {
        return (1.0 / (s - 1.0) + 1.0 / (2.0 * a)) * a.powf(1.0 - s);
    }

    let mut sum = a.powf(-s);
    let mut base = a;
    let mut terms = 0;
    let mut b = 0.0;
    while terms < 9 || base <= 9.0 {
        terms += 1;
        base += 1.0;
        b = base.powf(-s);
        sum += b;
        if (b / sum).abs() < MACHEP {
            return sum;
        }
    }
    let w = base;
    sum += b * w / (s - 1.0);
    sum -= 0.5 * b;
    let mut poch = 1.0;
    let mut k = 0.0;
    for divisor in EULER_MACLAURIN {
        poch *= s + k;
        b /= w;
        let t = poch * b / divisor;
        sum += t;
        if (t / sum).abs() < MACHEP {
            return sum;
        }
        k += 1.0;
        poch *= s + k;
        b /= w;
        k += 1.0;
    }
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
        // Preallocated and filled in place, not `(0..n).map(f).collect()`: collecting
        // `Result`s goes through a shunt whose size hint is 0, so the Vec regrew as it went.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(f(i)?);
        }
        Ok(out)
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
    kelvin_ber_bei_prime(x).re
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
    kelvin_ber_bei_prime(x).im
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

/// The analytic Kelvin-derivative identities ber'+i·bei' = e^{iπ/4} I₁(z) and
/// ker'+i·kei' = -e^{iπ/4} K₁(z), z = x e^{iπ/4}. I₁/K₁ use complex series (small x) and
/// asymptotics (large x). They replace the finite-difference / cancelling direct-series forms
/// that were ~1e-8..4e-6 off scipy (frankenscipy-l3kwr).
///
/// The two pairs are separate functions: each derivative needs only one of I₁ and K₁, and the
/// K₁ series recomputes I₁ itself, so a shared helper made ber'/bei' evaluate three series for
/// the one they use.
fn kelvin_rotation(x: f64) -> (Complex64, Complex64) {
    use std::f64::consts::FRAC_1_SQRT_2;
    let rot = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2); // e^{iπ/4}
    let z = Complex64::new(x * FRAC_1_SQRT_2, x * FRAC_1_SQRT_2);
    (rot, z)
}

/// ber'(x) + i·bei'(x) = e^{iπ/4} I₁(x e^{iπ/4}).
fn kelvin_ber_bei_prime(x: f64) -> Complex64 {
    let (rot, z) = kelvin_rotation(x);
    let i1 = if x < 20.0 {
        complex_i1_series(z)
    } else {
        complex_i1_asymptotic(z)
    };
    rot * i1
}

/// ker'(x) + i·kei'(x) = -e^{iπ/4} K₁(x e^{iπ/4}).
fn kelvin_ker_kei_prime(x: f64) -> Complex64 {
    let (rot, z) = kelvin_rotation(x);
    let k1 = if x < KELVIN_ASYMP_X {
        complex_k1_series(z)
    } else {
        complex_k1_asymptotic(z)
    };
    -(rot * k1)
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
    kelvin_ker_kei_prime(x).re
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
    kelvin_ker_kei_prime(x).im
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
///
/// The crate's one digamma kernel, SciPy's xsf `digamma`. It keeps SciPy's signed pole at
/// zero, ψ(+0) = -inf and ψ(-0) = +inf (frankenscipy-eaqem), and NaN at the negative
/// integers. This was a second, separately maintained shift-to-12 asymptotic
/// (frankenscipy-re34v).
#[must_use]
pub fn digamma_scalar(x: f64) -> f64 {
    crate::gamma::digamma_core(x)
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial & Utility Functions
// ══════════════════════════════════════════════════════════════════════

/// Rising factorial (Pochhammer symbol): (x)_n = x(x+1)...(x+n-1).
///
/// Matches `scipy.special.poch`, to the bit: this is xsf's Cephes `poch` (the revision SciPy
/// 1.17.1 pins, 0d0a593f).
/// 1. Recurrences bring |m| below 1: multiply down while m ≥ 1, divide up while m ≤ −1.
///    Each loop stops early at a pole, an overflow or an underflow.
/// 2. m = 0 then returns the product.
/// 3. a > 10⁴ with |m| ≤ 1 takes an a^m series.
/// 4. The Γ poles are resolved: +∞ where only Γ(a + m) has one, 0 where only Γ(a) has one.
/// 5. Otherwise the result is the product times exp(lgam(a + m) − lgam(a)) and both gamma signs.
///
/// The form this replaced took an exact rising product for integer m and the log-gamma ratio
/// for everything else. Near a pole it was 7.4e-11 relative off at (−3.0000000013, −3.000006),
/// where SciPy is 3.5e-15 (frankenscipy-zw56i). A Python emulation of this code matched
/// scipy.special.poch on 45,007 of 45,007 points: pole-adjacent a, integer m, the large-a
/// series and the edges.
pub fn poch(x: f64, n: f64) -> f64 {
    let is_nonpos_int = |v: f64| v <= 0.0 && v == v.ceil() && v.abs() < 1e13;
    let (a, mut m) = (x, n);
    let mut r = 1.0_f64;
    // 1. Reduce |m| below 1 by the recurrences.
    while m >= 1.0 {
        if a + m == 1.0 {
            break;
        }
        m -= 1.0;
        r *= a + m;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }
    while m <= -1.0 {
        if a + m == 0.0 {
            break;
        }
        r /= a + m;
        m += 1.0;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }
    // 2. Evaluate with the reduced m.
    if m == 0.0 {
        return r;
    }
    if a > 1e4 && m.abs() <= 1.0 {
        return r
            * a.powf(m)
            * (1.0
                + m * (m - 1.0) / (2.0 * a)
                + m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a)
                + m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a));
    }
    if is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m {
        return f64::INFINITY;
    }
    if !is_nonpos_int(a + m) && is_nonpos_int(a) {
        return 0.0;
    }
    let mode = fsci_runtime::RuntimeMode::Strict;
    let lgam = |v: f64| crate::gammaln_scalar(v, mode).unwrap_or(f64::NAN);
    let sgn = |v: f64| crate::gammasgn_scalar(v, mode).unwrap_or(f64::NAN);
    r * (lgam(a + m) - lgam(a)).exp() * sgn(a + m) * sgn(a)
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
///
/// Under `errstate`, a negative real argument is SciPy's "domain error".
pub fn spence(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    let value = spence_dispatch(x_tensor, mode)?;
    crate::sf_error_unary("spence", x_tensor, mode, |x| {
        (x < 0.0).then_some(crate::SpecialErrorCode::Domain)
    })?;
    Ok(value)
}

fn spence_dispatch(x_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
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
// Verbatim from xsf/cephes/spence.h. Rounded to 16 digits, two entries parsed 1 ulp away from
// Cephes's doubles; the last, 9.99999999999999998740E-1, is exactly 1.0 as a double
// (frankenscipy-qbwth).
#[allow(clippy::excessive_precision)]
const SPENCE_B: [f64; 8] = [
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
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
    // xsf `wrightomega(double)`, bit-identical to SciPy 1.17.1 on 140,006 points. The seed is:
    // - e^z below -2;
    // - e^(2(z-1)/3) on [-2, 1);
    // - z - ln z + ln z / z beyond.
    // Then one Fritsch-Shafer-Crowley step, and a second when the condition estimate asks.
    // e^z alone is returned only below -50, where W(e^z) = e^z - e^2z + ... already rounds to
    // e^z. The previous Newton returned e^z from -18.4, 1e-8 relative off (frankenscipy-i20cg).
    if z.is_nan() {
        return z;
    }
    if z.is_infinite() {
        return if z > 0.0 { z } else { 0.0 };
    }
    if z < -50.0 {
        return z.exp();
    }
    if z > 1e20 {
        return z;
    }
    let mut w = if z < -2.0 {
        z.exp()
    } else if z < 1.0 {
        (2.0 * (z - 1.0) / 3.0).exp()
    } else {
        let l = z.ln();
        z - l + l / z
    };
    let fsc_step = |w: f64| -> (f64, f64, f64) {
        let r = z - w - w.ln();
        let wp1 = w + 1.0;
        let e = r / wp1 * (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - r)
            / (2.0 * wp1 * (wp1 + 2.0 / 3.0 * r) - 2.0 * r);
        (w * (1.0 + e), r, wp1)
    };
    let (next, r, wp1) = fsc_step(w);
    w = next;
    if ((2.0 * w * w - 8.0 * w - 1.0) * r.abs().powf(4.0)).abs()
        >= f64::EPSILON * 72.0 * wp1.abs().powf(6.0)
    {
        w = fsc_step(w).0;
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
///
/// SciPy's domain first: a shape that is not finite and positive is NaN at every `y`, the
/// endpoints included (`betaincinv(0, 3, 0)` and `betaincinv(inf, 3, 0.5)` are nan in SciPy
/// 1.17.1). The root is `crate::beta::ibeta_inv_pair`'s, which solves against the smaller of
/// `y` and `1 − y` and keeps `x` to full relative precision down to the 1e-300 tail
/// (frankenscipy-xzrpr).
pub fn betaincinv_scalar(a: f64, b: f64, y: f64) -> f64 {
    if a.is_nan() || b.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
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
    crate::beta::ibeta_inv_pair(a, b, y, 1.0 - y).0
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

/// Scalar helper for the inverse regularized incomplete gamma function: SciPy's
/// `gammaincinv`, xsf's `igami` bit for bit (DiDonato & Morris' estimate and at most three
/// Halley steps; `crate::igam_temme`, frankenscipy-449uv). It replaced a bracketed Newton
/// solve that took four to ten full `gammainc` evaluations.
#[must_use]
pub fn gammaincinv_scalar(a: f64, y: f64) -> f64 {
    crate::igam_temme::igami(a, y)
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

/// Scalar helper for the inverse complemented regularized incomplete gamma function:
/// SciPy's `gammainccinv`, xsf's `igamci` bit for bit (`crate::igam_temme`,
/// frankenscipy-449uv). It replaced a full `gammaincinv(a, 1 − y)` solve followed by a Newton
/// refinement on `Q`.
#[must_use]
pub fn gammainccinv_scalar(a: f64, y: f64) -> f64 {
    crate::igam_temme::igamci(a, y)
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
        // Asymptotic series erfcx(x) = 1/(x√π)·Σ_k (−1)^k·(2k−1)!!/(2x²)^k. From x = 25 each
        // term is at most (2k+1)/1250 of the one before, so eight terms reach ε: the ninth is
        // 5e-21 of the sum. Three terms stopped at 15/(8x⁶), 7.7e-9 at x = 25
        // (frankenscipy-k5qew).
        let inv_x = 1.0 / x;
        let t = 0.5 * inv_x * inv_x;
        let mut term = 1.0_f64;
        let mut sum = 1.0_f64;
        for k in 1..=8_u32 {
            term *= -f64::from(2 * k - 1) * t;
            sum += term;
        }
        inv_x / std::f64::consts::PI.sqrt() * sum
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

/// Owen's T, SciPy's bits: xsf's `cephes/owens_t.h` (the revision SciPy 1.17.1 pins,
/// 0d0a593f), which is Patefield and Tandy's algorithm ("Fast and accurate calculation of
/// Owen's T-function", J. Stat. Softw. 5(5), 2000). The former 10-point Gauss-Legendre rule
/// on [0, a] was up to 2.5e-9 relative off at (h, a) = (-4.872, -0.922), where SciPy is
/// 1.1e-13 (frankenscipy-nb55y).
///
/// T is even in h and odd in a, so the kernel sees h ≥ 0, a ≥ 0; a > 1 maps to 1/a through
/// Owen's reflection, taken with Φ or with the complementary Φ(−x) by whether ah ≤ 0.67.
/// Every operation keeps the C source's order and association; erf, erfc and ndtr are the
/// Cephes kernels xsf calls, and `expm1` is Cephes' rational, not libm's.
#[must_use]
pub fn owens_t_scalar(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    let h = h.abs();
    let fabs_a = a.abs();
    let fabs_ah = fabs_a * h;
    let result = if fabs_a == f64::INFINITY {
        // Patefield-Tandy p. 13.
        0.5 * owens_t_norm2(h)
    } else if h == f64::INFINITY {
        0.0
    } else if fabs_a <= 1.0 {
        owens_t_dispatch(h, fabs_a, fabs_ah)
    } else if fabs_ah <= 0.67 {
        let normh = owens_t_norm1(h);
        let normah = owens_t_norm1(fabs_ah);
        0.25 - normh * normah - owens_t_dispatch(fabs_ah, 1.0 / fabs_a, h)
    } else {
        let normh = owens_t_norm2(h);
        let normah = owens_t_norm2(fabs_ah);
        (normh + normah) / 2.0 - normh * normah - owens_t_dispatch(fabs_ah, 1.0 / fabs_a, h)
    };
    if a < 0.0 { -result } else { result }
}

/// Method index by (h, a) cell: 15 h intervals (`OWENS_T_HRANGE` upper bounds, then above
/// 4.8) times 8 a intervals (`OWENS_T_ARANGE`, then above 0.99999), row-major in a.
const OWENS_T_SELECT_METHOD: [usize; 120] = [
    0, 0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 8, 0, 1, 1, 2, 2, 4, 4, 13, 13, 14, 14,
    15, 15, 15, 8, 1, 1, 2, 2, 2, 4, 4, 14, 14, 14, 14, 15, 15, 15, 9, 1, 1, 2, 4, 4, 4, 4, 6, 6,
    15, 15, 15, 15, 15, 9, 1, 2, 2, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 10, 1, 2, 4, 4, 4, 5, 5,
    7, 7, 16, 16, 16, 11, 11, 11, 1, 2, 3, 3, 5, 5, 7, 7, 16, 16, 16, 16, 16, 11, 11, 1, 2, 3, 3,
    5, 5, 17, 17, 17, 17, 16, 16, 16, 11, 11,
];

const OWENS_T_HRANGE: [f64; 14] = [
    0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8,
];

const OWENS_T_ARANGE: [f64; 7] = [0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999];

/// Series order per method index (a double in the C source, compared against int counters).
const OWENS_T_ORD: [f64; 18] = [
    2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 18.0, 10.0, 20.0, 30.0, 0.0, 4.0, 7.0, 8.0, 20.0, 0.0, 0.0,
];

/// Which of T1..T6 each method index runs.
const OWENS_T_METHODS: [u8; 18] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6];

/// T3's Chebyshev-derived coefficients, verbatim from xsf.
const OWENS_T_C: [f64; 31] = [
    1.0,
    -1.0,
    1.0,
    -0.9999999999999998,
    0.9999999999999839,
    -0.9999999999993063,
    0.9999999999797337,
    -0.9999999995749584,
    0.9999999933226235,
    -0.9999999188923242,
    0.9999992195143483,
    -0.9999939351372067,
    0.9999613559769055,
    -0.9997955636651394,
    0.9990927896296171,
    -0.9965938374119182,
    0.9891001713838613,
    -0.9700785580406933,
    0.9291143868326319,
    -0.8542058695956156,
    0.737965260330301,
    -0.585234698828374,
    0.4159977761456763,
    -0.25882108752419436,
    0.13755358251638927,
    -0.060795276632595575,
    0.021633768329987153,
    -0.005934056934551867,
    0.0011743414818332946,
    -0.0001489155613350369,
    9.072354320794358e-06,
];

/// T5's 13-point Gauss rule on [0, 1] in t², nodes and weights verbatim from xsf.
const OWENS_T_PTS: [f64; 13] = [
    0.35082039676451715489E-02,
    0.31279042338030753740E-01,
    0.85266826283219451090E-01,
    0.16245071730812277011E+00,
    0.25851196049125434828E+00,
    0.36807553840697533536E+00,
    0.48501092905604697475E+00,
    0.60277514152618576821E+00,
    0.71477884217753226516E+00,
    0.81475510988760098605E+00,
    0.89711029755948965867E+00,
    0.95723808085944261843E+00,
    0.99178832974629703586E+00,
];

const OWENS_T_WTS: [f64; 13] = [
    0.18831438115323502887E-01,
    0.18567086243977649478E-01,
    0.18042093461223385584E-01,
    0.17263829606398753364E-01,
    0.16243219975989856730E-01,
    0.14994592034116704829E-01,
    0.13535474469662088392E-01,
    0.11886351605820165233E-01,
    0.10070377242777431897E-01,
    0.81130545742299586629E-02,
    0.60419009528470238773E-02,
    0.38862217010742057883E-02,
    0.16793031084546090448E-02,
];

/// xsf `get_method`: the first h and a interval whose upper bound is not exceeded.
fn owens_t_method(h: f64, a: f64) -> usize {
    let ihint = OWENS_T_HRANGE
        .iter()
        .position(|&bound| h <= bound)
        .unwrap_or(14);
    let iaint = OWENS_T_ARANGE
        .iter()
        .position(|&bound| a <= bound)
        .unwrap_or(7);
    OWENS_T_SELECT_METHOD[iaint * 15 + ihint]
}

/// xsf `owens_t_norm1`: erf(x/√2)/2 = Φ(x) − ½. A division by √2, as in the source; the
/// product with 1/√2 differs in the last bit.
fn owens_t_norm1(x: f64) -> f64 {
    crate::error::erf_scalar(x / SQRT_2) / 2.0
}

/// xsf `owens_t_norm2`: erfc(x/√2)/2 = Φ(−x).
fn owens_t_norm2(x: f64) -> f64 {
    crate::error::erfc_scalar(x / SQRT_2) / 2.0
}

fn owens_t_sqrt_2pi() -> f64 {
    (2.0 * PI).sqrt()
}

/// T1: the series in (a^(2j+1))/(2j+1) with the incomplete exponential sums, m + 1 terms.
fn owens_t1(h: f64, a: f64, m: f64) -> f64 {
    let hs = -0.5 * h * h;
    let dhs = hs.exp();
    let as_ = a * a;
    let mut aj = a / (2.0 * PI);
    let mut dj = xsf_smirnov::cephes_expm1(hs);
    let mut gj = hs * dhs;
    let mut val = a.atan() / (2.0 * PI);
    let mut j = 1.0;
    let mut jj = 1.0;
    loop {
        val += dj * aj / jj;
        if m <= j {
            break;
        }
        j += 1.0;
        jj += 2.0;
        aj *= as_;
        dj = gj - dj;
        gj *= hs / j;
    }
    val
}

/// T2: the series in powers of 1/h², 2m + 1 terms.
fn owens_t2(h: f64, a: f64, ah: f64, m: f64) -> f64 {
    let maxi = 2.0 * m + 1.0;
    let hs = h * h;
    let as_ = -a * a;
    let y = 1.0 / hs;
    let mut val = 0.0;
    let mut vi = a * (-0.5 * ah * ah).exp() / owens_t_sqrt_2pi();
    let mut z = (ndtr_scalar(ah) - 0.5) / h;
    let mut i = 1.0;
    loop {
        val += z;
        if maxi <= i {
            break;
        }
        z = y * (vi - i * z);
        vi *= as_;
        i += 2.0;
    }
    val * ((-0.5 * hs).exp() / owens_t_sqrt_2pi())
}

/// T3: T2's series with the 31 Chebyshev-economised coefficients.
fn owens_t3(h: f64, a: f64, ah: f64) -> f64 {
    let aa = a * a;
    let hh = h * h;
    let y = 1.0 / hh;
    let mut vi = a * (-ah * ah / 2.0).exp() / owens_t_sqrt_2pi();
    let mut zi = owens_t_norm1(ah) / h;
    let mut result = 0.0;
    let mut odd = 1.0;
    for &c in &OWENS_T_C {
        result += zi * c;
        zi = y * (odd * zi - vi);
        vi *= aa;
        odd += 2.0;
    }
    result * ((-hh / 2.0).exp() / owens_t_sqrt_2pi())
}

/// T4: the series in powers of −a², 2m + 1 terms.
fn owens_t4(h: f64, a: f64, m: f64) -> f64 {
    let maxi = 2.0 * m + 1.0;
    let hh = h * h;
    let naa = -a * a;
    let mut i = 1.0;
    let mut ai = a * (-hh * (1.0 - naa) / 2.0).exp() / (2.0 * PI);
    let mut yi = 1.0;
    let mut result = 0.0;
    loop {
        result += ai * yi;
        if maxi <= i {
            break;
        }
        i += 2.0;
        yi = (1.0 - hh * yi) / i;
        ai *= naa;
    }
    result
}

/// T5: 13-point Gauss quadrature of the defining integral.
fn owens_t5(h: f64, a: f64) -> f64 {
    let aa = a * a;
    let nhh = -0.5 * h * h;
    let mut result = 0.0;
    for (&pt, &wt) in OWENS_T_PTS.iter().zip(&OWENS_T_WTS) {
        let r = 1.0 + aa * pt;
        result += wt * (nhh * r).exp() / r;
    }
    result * a
}

/// T6: the a → 1 expansion about T(h, 1) = Φ(h)Φ(−h)/2.
fn owens_t6(h: f64, a: f64) -> f64 {
    let normh = owens_t_norm2(h);
    let y = 1.0 - a;
    let r = y.atan2(1.0 + a);
    let mut result = normh * (1.0 - normh) / 2.0;
    if r != 0.0 {
        result -= r * (-y * h * h / (2.0 * r)).exp() / (2.0 * PI);
    }
    result
}

/// xsf `owens_t_dispatch` for h ≥ 0 and 0 ≤ a ≤ 1; `ah` is a·h, or the original h under the
/// reflection.
fn owens_t_dispatch(h: f64, a: f64, ah: f64) -> f64 {
    if h == 0.0 {
        return a.atan() / (2.0 * PI);
    }
    if a == 0.0 {
        return 0.0;
    }
    if a == 1.0 {
        return owens_t_norm2(-h) * owens_t_norm2(h) / 2.0;
    }
    let index = owens_t_method(h, a);
    let m = OWENS_T_ORD[index];
    match OWENS_T_METHODS[index] {
        1 => owens_t1(h, a, m),
        2 => owens_t2(h, a, ah, m),
        3 => owens_t3(h, a, ah),
        4 => owens_t4(h, a, m),
        5 => owens_t5(h, a),
        6 => owens_t6(h, a),
        _ => f64::NAN,
    }
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

/// The SIMD path takes [`log_ndtr_scalar`]'s two branches lane by lane. For `x ≥ −1` that is
/// `log1p(−erfc(x/√2)/2)`, SciPy's form. The path once took `ln(erfc(−x/√2)/2)` for every
/// `x ≥ −8`, which is `ln Φ(x)` with Φ(x) rounded toward 1. It returned 0 from about x = 8.5
/// where the answer is −Φ(−x) (log_ndtr(10) = −7.6e-24), and lost digits from x ≈ 1. The
/// scalar kernel's fix (frankenscipy-08o4z) never reached this duplicate, so tensors of 64 to
/// 2^20 elements kept it (frankenscipy-iulu0).
fn log_ndtr_real_vec_simd(values: &[f64]) -> Vec<f64> {
    use std::simd::Simd;
    const LANES: usize = 8;
    let mut out = vec![0.0f64; values.len()];
    let mut i = 0;
    while i + LANES <= values.len() {
        let x = Simd::<f64, LANES>::from_slice(&values[i..i + LANES]);
        let lanes = x.to_array();
        if lanes.iter().all(|x| x.is_finite() && *x >= -8.0) {
            // erfc(x/√2) = 2Φ(−x) where x ≥ −1, erfc(−x/√2) = 2Φ(x) below.
            let u = lanes.map(|v| {
                if v >= -1.0 {
                    v * FRAC_1_SQRT_2
                } else {
                    -v * FRAC_1_SQRT_2
                }
            });
            let erfc = crate::error::erfc_full_simd_chunk(Simd::from_array(u));
            for j in 0..LANES {
                out[i + j] = if lanes[j] >= -1.0 {
                    (-(0.5 * erfc[j])).ln_1p()
                } else {
                    (0.5 * erfc[j]).ln()
                };
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
/// Matches `scipy.special.struve(v, x)` for scalar `v` and an array `x`, fanning the
/// scalar kernel across cores via the crate's order-preserving parallel map. The
/// kernel is SciPy's algorithm (see [`struve`]), often tens to hundreds of double-double
/// power-series terms per point, so the per-point cost is of SciPy's order; the
/// earlier ~47 ns kernel that this map was measured against was the inaccurate
/// series/asymptotic switch it replaced (frankenscipy-00cad). Bit-identical to a serial
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
            t.powi(n as i32) / t.exp_m1()
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
/// Matches `scipy.special.wofz` bit for bit over the whole complex plane (see
/// [`wofz_scalar`]). Real inputs are returned as complex values because SciPy
/// exposes `wofz` as a complex-valued function.
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

/// The Faddeeva function w(z) = exp(−z²) · erfc(−iz) at one complex point.
///
/// This is the Faddeeva package SciPy builds (`xsf/faddeeva.h`, see the private
/// `faddeeva` module), so it returns `scipy.special.wofz`'s bits everywhere, the
/// limits at ±∞ and NaN included. Against mpmath over 51,773 upper-half-plane
/// points (|x| ≤ 30 and 1e-12 ≤ y ≤ 10, plus |z| up to 1e3) that is at most
/// 3.0e-14 relative in Re w and 2.4e-13 in Im w. The dispatch it replaced
/// (erf series, Weideman N = 32, a 24-term continued fraction, an asymptotic
/// series) lost Re w near the real axis, where Re w ≈ e^{−x²} is far below |w|:
/// 100% of it at |x| ≈ 4.5, which put `voigt_profile` 2.3e-5 off
/// (frankenscipy-k64p7). Hardened mode fails closed on a non-finite argument.
pub fn wofz_scalar(z: Complex64, mode: RuntimeMode) -> Result<Complex64, SpecialError> {
    if !z.is_finite() && mode == RuntimeMode::Hardened {
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
    Ok(crate::faddeeva::w(z))
}

/// Compute the Faddeeva function w(z) = exp(−z²) · erfc(−iz) at a real
/// argument `x`, returning the real and imaginary parts as `(re, im)`.
///
/// On the real axis the closed form is
///   Re[w(x)] = exp(−x²)
///   Im[w(x)] = (2/√π) · F(x)
/// where F is the Dawson function. The imaginary part is the Faddeeva package's
/// `w_im`, so this is `scipy.special.wofz(x + 0j)` bit for bit, and equal to
/// `wofz_scalar(x + 0i)`.
pub fn wofz_real(x: f64) -> (f64, f64) {
    let w = crate::faddeeva::w(Complex64::from_real(x));
    (w.re, w.im)
}

/// `1/√2` and `√(2π)` as xsf's `voigt_profile` spells them.
const VOIGT_INV_SQRT_2: f64 = 0.707106781186547524401;
const VOIGT_SQRT_2PI: f64 = 2.5066282746310002416123552393401042;

/// Voigt profile V(x; σ, γ) on the real axis.
///
/// `scipy.special.voigt_profile`, bit for bit: xsf's `voigt_profile` operation for
/// operation, `Re[w((x + iγ)/(√2σ))] / σ / √(2π)` over the Faddeeva package's `w`,
/// with SciPy's point-mass (`σ = γ = 0`), Lorentzian (`σ = 0`) and Gaussian
/// (`γ = 0`) cases. Like SciPy it does not reject a negative `σ` or `γ`.
pub fn voigt_profile(x: f64, sigma: f64, gamma: f64) -> f64 {
    if sigma == 0.0 {
        if gamma == 0.0 {
            if x.is_nan() {
                return x;
            }
            return if x == 0.0 { f64::INFINITY } else { 0.0 };
        }
        return gamma / PI / (x * x + gamma * gamma);
    }
    if gamma == 0.0 {
        return voigt_gaussian(x, sigma);
    }
    let zreal = x / sigma * VOIGT_INV_SQRT_2;
    let zimag = gamma / sigma * VOIGT_INV_SQRT_2;
    let w = crate::faddeeva::w(Complex64::new(zreal, zimag));
    w.re / sigma / VOIGT_SQRT_2PI
}

/// xsf's `γ = 0` branch of `voigt_profile`, the normal density with mean 0 and
/// standard deviation `sigma`.
fn voigt_gaussian(x: f64, sigma: f64) -> f64 {
    1.0 / VOIGT_SQRT_2PI / sigma * (-(x / sigma) * (x / sigma) / 2.0).exp()
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
/// a useful real-only fast path; for `sigma > 0` it returns SciPy's bits. Unlike
/// `voigt_profile`, a non-positive or NaN `sigma` is NaN here.
pub fn voigt_profile_real_gamma_zero(x: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || sigma.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    voigt_gaussian(x, sigma)
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

/// Arithmetic-geometric mean.
///
/// SciPy 1.17.1's own `agm` (`scipy/special/_agm.pxd`), SciPy's bits. In the normal range it is
/// `(π/4)(a + b)/K(1 − e)` with `e = 4ab/(a + b)²` and K Cephes' `ellpk`, the kernel `ellipkm1`
/// is. Beyond `1/√(max/2)` .. `√(max/2)` the 20-step iteration runs instead, so nothing
/// overflows. SciPy's domain comes with it:
/// - NaN in or opposite signs: NaN;
/// - 0 against an infinity: NaN;
/// - any zero: 0;
/// - `a = b`: `a`;
/// - both negative: `−agm(−a, −b)`.
///
/// The iteration this replaced answered NaN for every zero or negative argument
/// (frankenscipy-89pgv).
pub fn agm(a: f64, b: f64) -> f64 {
    const SQRT_HALF_MAX: f64 = 9.480_751_908_109_176e153;
    const INV_SQRT_HALF_MAX: f64 = 1.054_768_661_486_3e-154;
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
        return f64::NAN;
    }
    if (a.is_infinite() || b.is_infinite()) && (a == 0.0 || b == 0.0) {
        return f64::NAN;
    }
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    if a == b {
        return a;
    }
    let (sgn, a, b) = if a < 0.0 { (-1.0, -a, -b) } else { (1.0, a, b) };
    if INV_SQRT_HALF_MAX < a && a < SQRT_HALF_MAX && INV_SQRT_HALF_MAX < b && b < SQRT_HALF_MAX {
        let e = 4.0 * a * b / (a + b).powi(2);
        return sgn * (PI / 4.0) * (a + b) / crate::elliptic::cephes_ellpk_x(e);
    }
    // SciPy's `_agm_iter`: at most 20 steps, stopping once the mean repeats an argument.
    let (mut a, mut b) = (a, b);
    let mut amean = 0.5 * a + 0.5 * b;
    let mut count = 20;
    while count > 0 && amean != a && amean != b {
        let gmean = a.sqrt() * b.sqrt();
        a = amean;
        b = gmean;
        amean = 0.5 * a + 0.5 * b;
        count -= 1;
    }
    sgn * amean
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

/// Survival function of the Kolmogorov distribution, P(sqrt(n) D_n > x) in the limit.
///
/// Matches `scipy.special.kolmogorov(y)`: xsf's `cephes::detail::_kolmogorov`, bit-identical to
/// SciPy 1.17.1 on 130,006 points.
pub fn kolmogorov(y_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_par("kolmogorov", y_tensor, mode, |y| Ok(kolmogorov_scalar(y)))
}

#[must_use]
pub fn kolmogorov_scalar(y: f64) -> f64 {
    kolmogorov_sf_cdf_pdf(y).0
}

/// (sf, cdf, pdf) of the Kolmogorov limit law: `scipy.special.kolmogorov`, SciPy's private
/// `_kolmogc`, and `-_kolmogp`. Ported from xsf `cephes::detail::_kolmogorov`.
/// - x <= 0.82: the Jacobi-theta dual series. SciPy returns exactly 1 below
///   pi / sqrt(8 * 746), where its terms underflow.
/// - Otherwise: the alternating series 2(v - v^4 + v^9 - ...), with v = exp(-2x^2).
///
/// This replaced a 100-term alternating series used everywhere. For x below ~0.035 its terms
/// stay near 1 for ~4.3/x of them, and it was up to 0.1376 off SciPy (frankenscipy-11wqg).
fn kolmogorov_sf_cdf_pdf(x: f64) -> (f64, f64, f64) {
    use std::f64::consts::PI;
    if x.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    // x <= pi / sqrt(8 · 746): exp(-pi^2/8x^2) underflows.
    if x <= 0.0 || x <= PI / f64::from(746 * 8).sqrt() {
        return (1.0, 0.0, 0.0);
    }
    let mut p = 1.0_f64;
    let mut d = 0.0_f64;
    let (sf, cdf);
    if x <= 0.82 {
        // P = w u (1 + u^8 + u^24 + u^48 + ...), u = e^(-pi^2/8x^2), w = sqrt(2pi)/x
        let w = (2.0 * PI).sqrt() / x;
        let logu8 = -PI * PI / (x * x);
        let u = (logu8 / 8.0).exp();
        if u == 0.0 {
            p = (logu8 / 8.0 + w.ln()).exp();
        } else {
            let u8 = logu8.exp();
            let u8cub = u8.powf(3.0);
            p = 1.0 + u8cub * p;
            d = 5.0 * 5.0 + u8cub * d;
            p = 1.0 + u8 * u8 * p;
            d = 3.0 * 3.0 + u8 * u8 * d;
            p = 1.0 + u8 * p;
            d = 1.0 * 1.0 + u8 * d;
            d = PI * PI / 4.0 / (x * x) * d - p;
            d *= w * u / x;
            p *= w * u;
        }
        cdf = p;
        sf = 1.0 - p;
    } else {
        // P = 2 (v - v^4 + v^9 - ...), v = e^(-2x^2)
        let v = (-2.0 * x * x).exp();
        let vsq = v * v;
        let v3 = v.powf(3.0);
        let mut vpwr = v3 * v3 * v;
        p = 1.0 - vpwr * p;
        d = 3.0 * 3.0 - vpwr * d;
        vpwr = v3 * vsq;
        p = 1.0 - vpwr * p;
        d = 2.0 * 2.0 - vpwr * d;
        vpwr = v3;
        p = 1.0 - vpwr * p;
        d = 1.0 * 1.0 - vpwr * d;
        p *= 2.0 * v;
        d *= 8.0 * v * x;
        sf = p;
        cdf = 1.0 - sf;
    }
    (sf.clamp(0.0, 1.0), cdf.clamp(0.0, 1.0), 0.0_f64.max(d))
}

/// Inverse of the Kolmogorov survival function: y with kolmogorov(y) = p.
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
    kolmogi_pair(p, 1.0 - p)
}

/// x with kolmogorov(x) = psf and the cdf at x = pcdf, where psf + pcdf = 1: xsf
/// `cephes::detail::_kolmogi`, a bracketed Newton iteration on `kolmogorov_sf_cdf_pdf`.
/// Giving both tails lets a caller holding the smaller one keep its precision. SciPy's
/// `kolmogi(p)` is `(p, 1 - p)`, and its private `_kolmogci(p)` is `(1 - p, p)`.
///
/// NaN when either is outside [0, 1] or they do not sum to 1 within 4 eps.
///
/// Against SciPy 1.17.1, `(p, 1 - p)` is bit-identical on 99.8% of 40,003 points and 1 ulp off
/// on the rest (p in 0.59-0.68). The xsf source SciPy pins is semantically this code, so the
/// cause is not known. The previous safeguarded Newton here was 6.3e-14 off.
#[must_use]
pub fn kolmogi_pair(psf: f64, pcdf: f64) -> f64 {
    use std::f64::consts::{PI, SQRT_2};
    #[allow(clippy::excessive_precision)]
    const LOGSQRT2PI: f64 = 9.189_385_332_046_727_417_803_297e-1;
    const XTOL: f64 = f64::EPSILON;
    const RTOL: f64 = 2.0 * f64::EPSILON;
    let within_tol = |x: f64, y: f64| (x - y).abs() <= XTOL + RTOL * y.abs();
    if !((0.0..=1.0).contains(&psf) && (0.0..=1.0).contains(&pcdf))
        || (1.0 - pcdf - psf).abs() > 4.0 * f64::EPSILON
    {
        return f64::NAN;
    }
    if pcdf == 0.0 {
        return 0.0;
    }
    if psf == 0.0 {
        return f64::INFINITY;
    }
    let (mut a, mut b, mut x);
    if pcdf <= 0.5 {
        // p ~ (sqrt(2pi)/x) exp(-pi^2/8x^2): two fixed-point steps for each bound.
        let logpcdf = pcdf.ln();
        let bound = |logx: f64| PI / (2.0 * SQRT_2 * (-(logpcdf + logx - LOGSQRT2PI)).sqrt());
        a = bound(logpcdf / 2.0);
        b = bound(0.0);
        a = bound(a.ln());
        b = bound(b.ln());
        x = (a + b) / 2.0;
    } else {
        // p ~ 2 exp(-2x^2), inverted as a power series in p/2.
        let jiggerb = 256.0 * f64::EPSILON;
        let pba = psf / (1.0 - (-4.0_f64).exp()) / 2.0;
        let pbb = psf * (1.0 - jiggerb) / 2.0;
        a = (-0.5 * pba.ln()).sqrt();
        b = (-0.5 * pbb.ln()).sqrt();
        let ph = psf / 2.0;
        let p2 = ph * ph;
        let p3 = ph * ph * ph;
        let q0 = (1.0
            + p3 * (1.0 + p3 * (4.0 + p2 * (-1.0 + ph * (22.0 + p2 * (-13.0 + 140.0 * ph))))))
            * ph;
        x = (-q0.ln() / 2.0).sqrt();
        if x < a || x > b {
            x = (a + b) / 2.0;
        }
    }
    for _ in 0..=500 {
        let x0 = x;
        let (sf_x, cdf_x, pdf_x) = kolmogorov_sf_cdf_pdf(x0);
        let df = if pcdf < 0.5 { pcdf - cdf_x } else { sf_x - psf };
        if df == 0.0 {
            break;
        }
        if df > 0.0 && x > a {
            a = x;
        } else if df < 0.0 && x < b {
            b = x;
        }
        let dfdx = -pdf_x;
        x = if dfdx.abs() <= 0.0 {
            (a + b) / 2.0
        } else {
            x0 - df / dfdx
        };
        if x >= a && x <= b {
            if within_tol(x, x0) {
                break;
            }
            if x == a || x == b {
                x = (a + b) / 2.0;
                if x == a || x == b {
                    break;
                }
            }
        } else {
            x = (a + b) / 2.0;
            if within_tol(x, x0) {
                break;
            }
        }
    }
    x
}

/// One-sided Kolmogorov–Smirnov survival function `P(D_n^+ ≥ d)` for sample size `n`.
///
/// A port of xsf `cephes::smirnov` (the code SciPy 1.17.1 compiles, xsf 0d0a593f): the
/// Birnbaum–Tingey sum
///
/// ```text
///   P(D_n^+ ≥ d) = d · Σ_{v=0}^{⌊n(1−d)⌋} C(n, v) · (d + v/n)^{v−1} · (1 − d − v/n)^{n−v}
/// ```
///
/// evaluated term by term in double-double arithmetic with an early exit once the remaining
/// terms cannot move the sum (or, when it has at most three terms, the alternating
/// complementary sum over `v > n(1−d)`), the closed forms `d(1+d)^{n−1}` for `d ≤ 1/n` and
/// `(1−d)^n` for `d ≥ 1 − 1/n`, zero where `−2nd²` is below `log(2^-1075)`, and
/// `exp(−(6nd + 1)² / 18n)` for `n > 10^6`. NaN for `n < 1`, a NaN `d`, or `d` outside
/// `[0, 1]`. frankenscipy-k4c2c
///
/// Matches `scipy.special.smirnov(n, d)` bit for bit.
#[must_use]
pub fn smirnov(n: i32, d: f64) -> f64 {
    if d.is_nan() {
        return f64::NAN;
    }
    xsf_smirnov::smirnov3(i64::from(n), d).sf
}

/// `(sf, cdf, pdf)` of the one-sided statistic `D_n^+` at `d`: `scipy.special.smirnov`,
/// SciPy's private `_smirnovc` and `-_smirnovp`, computed together by the kernel behind
/// [`smirnov`]. `n` is 64-bit so that `scipy.stats.kstwo`'s sample size is never narrowed.
///
/// The pdf is the true derivative. SciPy's differs from it for 46342 ≤ n ≤ 10^6, where xsf's
/// C `int` products overflow (see [`smirnovi`]); `sf` and `cdf` are SciPy's bit for bit.
/// frankenscipy-k4c2c
#[must_use]
pub fn smirnov_sf_cdf_pdf(n: i64, d: f64) -> (f64, f64, f64) {
    let p = xsf_smirnov::smirnov3(n, d);
    (p.sf, p.cdf, p.pdf)
}

/// Inverse of [`smirnov`] in `d`: the `d` with `smirnov(n, d) = p`.
///
/// A port of xsf `cephes::smirnovi` (SciPy 1.17.1): `(1 − p)^{1/n}` solved exactly when the
/// root lies above `1 − 1/n` (for `n < 150`), otherwise a bracketed Newton–Raphson iteration
/// on the double-double Birnbaum–Tingey sum, started from `d(1+d)^{n−1} = 1 − p` near zero or
/// from `sqrt(−log p / 2n) − 1/6n` elsewhere. `smirnovi(n, 0) = 1`, `smirnovi(n, 1) = 0`;
/// NaN for `n < 1` or `p` outside `[0, 1]`. frankenscipy-k4c2c
///
/// Matches `scipy.special.smirnovi(n, p)` bit for bit, except where SciPy's result comes from
/// C undefined behaviour, which fsci does not reproduce: xsf forms `n·(v−1)` and `(n−v)·n` in
/// the Newton derivative, and `6n` in the starting point, as C `int`s, which overflow for
/// 46342 ≤ n ≤ 10^6 and for n > 357913941. There fsci's products are exact and its root can
/// differ from SciPy's: in the first range fsci's is the nearer to the exact root (SciPy's was
/// up to ~1400 ulp off at the points checked against mpmath), in the second the two land 1–2
/// ulp apart.
#[must_use]
pub fn smirnovi(n: i32, p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    xsf_smirnov::smirnovi(i64::from(n), p, 1.0 - p)
}

/// Cephes' `expm1` for the crate's other xsf ports (`crate::igam_temme`'s `igamc_series`).
pub(crate) use xsf_smirnov::cephes_expm1;

/// xsf `cephes/kolmogorov.h` (the one-sided half, xsf 0d0a593f as compiled by SciPy 1.17.1)
/// with the parts of `cephes/dd_real.h` and `cephes/unity.h` it calls, ported operation for
/// operation: every `double_double` operator and libm call is reproduced in the order xsf
/// evaluates it, because [`smirnov`] and [`smirnovi`] are pinned to SciPy bit for bit. xsf's
/// C `int` arithmetic is done in `i64`, which agrees with it wherever it does not overflow;
/// where it does (undefined behaviour), fsci keeps the exact value. frankenscipy-k4c2c
mod xsf_smirnov {
    use super::ldexp;
    use std::f64::consts::LN_2;

    /// `n` above this uses `exp(−(6nx + 1)² / 18n)` instead of summing.
    const SMIRNOV_MAX_COMPUTE_N: i64 = 1_000_000;
    /// The alternating upper sum is used only when it has at most this many terms...
    const SM_UPPER_MAX_TERMS: i64 = 3;
    /// ...and only from this `n` up.
    const SM_UPPERSUM_MIN_N: i64 = 10;
    /// Largest power taken in one `pow` by `pow2_scaled` (below 1023 − 52, so both words of
    /// the double-double stay normal).
    const SM_MAX_EXPONENT: i64 = 960;
    const KOLMOG_MAXITER: i32 = 500;
    const XTOL: f64 = f64::EPSILON;
    const RTOL: f64 = 2.0 * XTOL;
    /// cephes `MINLOG`, log(2^-1075): `exp` of anything below it is 0.
    const MINLOG: f64 = -7.451_332_191_019_412_076_235e2;

    /// cephes `expm1` (`unity.h`): a rational approximation on [−0.5, 0.5], not libm's
    /// `expm1`, from which it differs in the last bit. Also Owen's T's T1 (`owens_t1`), and
    /// `igamc_series` (`crate::igam_temme`) through [`super::cephes_expm1`].
    pub(crate) fn cephes_expm1(x: f64) -> f64 {
        const EP: [f64; 3] = [
            1.261_771_930_748_105_908_779_8e-4,
            3.029_944_077_074_419_612_995_6e-2,
            9.999_999_999_999_999_999_102_5e-1,
        ];
        const EQ: [f64; 4] = [
            3.001_985_051_386_644_550_415_9e-6,
            2.524_483_403_496_841_041_922_4e-3,
            2.272_655_482_081_550_287_659_3e-1,
            2.000_000_000_000_000_000_089_7,
        ];
        let polevl = |x: f64, coef: &[f64]| coef[1..].iter().fold(coef[0], |ans, &c| ans * x + c);
        if !x.is_finite() {
            if x.is_nan() || x > 0.0 {
                return x;
            }
            return -1.0;
        }
        if !(-0.5..=0.5).contains(&x) {
            return x.exp() - 1.0;
        }
        let xx = x * x;
        let r = x * polevl(xx, &EP[..]);
        let r = r / (polevl(xx, &EQ[..]) - r);
        r + r
    }

    /// C `frexp`: significand in [0.5, 1) and binary exponent; zero, ±inf and NaN come back
    /// unchanged with exponent 0.
    fn frexp(x: f64) -> (f64, i32) {
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        if biased == 0 {
            if x == 0.0 {
                return (x, 0);
            }
            let (m, e) = frexp(x * f64::from_bits(0x43f0_0000_0000_0000)); // · 2^64
            return (m, e - 64);
        }
        if biased == 0x7ff {
            return (x, 0);
        }
        (
            f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000),
            biased - 0x3fe,
        )
    }

    /// `fl(a + b)` and its rounding error, for `|a| ≥ |b|` (`quick_two_sum`).
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let c = s - a;
        (s, b - c)
    }

    /// `fl(a + b)` and its rounding error (`two_sum`).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let c = s - a;
        let d = b - c;
        let e = s - c;
        (s, (a - e) + d)
    }

    /// `fl(a · b)` and its rounding error. xsf takes the error from `std::fma`; `mul_add` is
    /// the same exactly rounded operation.
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        (p, a.mul_add(b, -p))
    }

    /// xsf `double_double`: the unevaluated sum `hi + lo`. Each method is one C++ operator of
    /// `dd_real.h`; the suffix names the operand types where xsf overloads on them. The Struve
    /// power series ([`super::xsf_struve`]) sums in it too.
    #[derive(Clone, Copy, Debug)]
    pub(super) struct Dd {
        pub(super) hi: f64,
        pub(super) lo: f64,
    }

    /// e (`dd_real.h` `E`).
    const DD_E: Dd = Dd {
        hi: 2.718_281_828_459_045_091e0,
        lo: 1.445_646_891_729_250_158e-16,
    };
    /// log 2 (`dd_real.h` `LOG2`).
    const DD_LOG2: Dd = Dd {
        hi: 6.931_471_805_599_452_862e-1,
        lo: 2.319_046_813_846_299_558e-17,
    };
    /// 2^-104 (`dd_real.h` `EPS`).
    const DD_EPS: f64 = 4.930_380_657_631_32e-32;
    /// 1/3!, …, 1/8! (`dd_real.h` `inv_fact`; `exp` reads no further).
    const INV_FACT: [Dd; 6] = [
        Dd {
            hi: 1.666_666_666_666_666_57e-1,
            lo: 9.251_858_538_542_970_66e-18,
        },
        Dd {
            hi: 4.166_666_666_666_666_44e-2,
            lo: 2.312_964_634_635_742_66e-18,
        },
        Dd {
            hi: 8.333_333_333_333_333_22e-3,
            lo: 1.156_482_317_317_871_38e-19,
        },
        Dd {
            hi: 1.388_888_888_888_888_94e-3,
            lo: -5.300_543_954_373_577_06e-20,
        },
        Dd {
            hi: 1.984_126_984_126_984_13e-4,
            lo: 1.720_955_829_342_070_53e-22,
        },
        Dd {
            hi: 2.480_158_730_158_730_16e-5,
            lo: 2.151_194_786_677_588_16e-23,
        },
    ];

    impl Dd {
        pub(super) const fn new(hi: f64) -> Self {
            Self { hi, lo: 0.0 }
        }

        const fn splat(v: f64) -> Self {
            Self { hi: v, lo: v }
        }

        const fn neg(self) -> Self {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        }

        /// `dd == double`
        fn eq_f64(self, rhs: f64) -> bool {
            self.hi == rhs && self.lo == 0.0
        }

        /// `dd < double`
        fn lt_f64(self, rhs: f64) -> bool {
            if self.hi < rhs {
                return true;
            }
            if self.hi > rhs {
                return false;
            }
            self.lo < 0.0
        }

        /// `dd + dd` (the Briggs–Kahan IEEE-style sum).
        pub(super) fn add(self, rhs: Self) -> Self {
            let (s1, s2) = two_sum(self.hi, rhs.hi);
            let (t1, t2) = two_sum(self.lo, rhs.lo);
            let (s1, s2) = quick_two_sum(s1, s2 + t1);
            let (hi, lo) = quick_two_sum(s1, s2 + t2);
            Self { hi, lo }
        }

        /// `dd + double`
        fn add_f64(self, rhs: f64) -> Self {
            let (s1, s2) = two_sum(self.hi, rhs);
            let (hi, lo) = quick_two_sum(s1, s2 + self.lo);
            Self { hi, lo }
        }

        /// `dd - dd`, which xsf evaluates as `lhs + (-rhs)`.
        fn sub(self, rhs: Self) -> Self {
            self.add(rhs.neg())
        }

        /// `dd - double`
        fn sub_f64(self, rhs: f64) -> Self {
            let (s1, s2) = two_sum(self.hi, -rhs);
            let (hi, lo) = quick_two_sum(s1, s2 + self.lo);
            Self { hi, lo }
        }

        /// `double - dd`, i.e. `lhs - self`.
        fn rsub_f64(self, lhs: f64) -> Self {
            let (s1, s2) = two_sum(lhs, -self.hi);
            let (hi, lo) = quick_two_sum(s1, s2 - self.lo);
            Self { hi, lo }
        }

        /// `dd * dd`
        pub(super) fn mul(self, rhs: Self) -> Self {
            let (p1, p2) = two_prod(self.hi, rhs.hi);
            let (hi, lo) = quick_two_sum(p1, p2 + (self.hi * rhs.lo + self.lo * rhs.hi));
            Self { hi, lo }
        }

        /// `dd * double`; xsf's `double * dd` forms the same two exact products, so it is this
        /// too.
        fn mul_f64(self, rhs: f64) -> Self {
            let (p1, e1) = two_prod(self.hi, rhs);
            let (p2, e2) = two_prod(self.lo, rhs);
            let (hi, lo) = quick_two_sum(p1, e2 + p2 + e1);
            Self { hi, lo }
        }

        /// `dd / dd` (three quotient digits).
        pub(super) fn div(self, rhs: Self) -> Self {
            let q1 = self.hi / rhs.hi;
            let r = self.sub(rhs.mul_f64(q1));
            let q2 = r.hi / rhs.hi;
            let r = r.sub(rhs.mul_f64(q2));
            let q3 = r.hi / rhs.hi;
            let (hi, lo) = quick_two_sum(q1, q2);
            Self { hi, lo }.add_f64(q3)
        }

        /// `dd / double`, which xsf evaluates as `lhs / double_double(rhs)`.
        fn div_f64(self, rhs: f64) -> Self {
            self.div(Self::new(rhs))
        }

        /// `double / dd`, i.e. `double_double(lhs) / self`.
        fn rdiv_f64(self, lhs: f64) -> Self {
            Self::new(lhs).div(self)
        }

        /// `mul_pwr2`: scale both words by a power of two.
        fn mul_pwr2(self, rhs: f64) -> Self {
            Self {
                hi: self.hi * rhs,
                lo: self.lo * rhs,
            }
        }

        fn square(self) -> Self {
            let p1 = self.hi * self.hi;
            let mut p2 = self.hi.mul_add(self.hi, -p1);
            p2 += 2.0 * self.hi * self.lo;
            p2 += self.lo * self.lo;
            let (hi, lo) = quick_two_sum(p1, p2);
            Self { hi, lo }
        }

        fn floor(self) -> Self {
            let hi = self.hi.floor();
            if hi == self.hi {
                // The high word is an integer already: round the low word.
                let (hi, lo) = quick_two_sum(hi, self.lo.floor());
                return Self { hi, lo };
            }
            Self { hi, lo: 0.0 }
        }

        fn ldexp(self, exp: i32) -> Self {
            Self {
                hi: ldexp(self.hi, exp),
                lo: ldexp(self.lo, exp),
            }
        }

        /// `(b, e)` with `self = b · 2^e`, `0.5 ≤ |b.hi| < 1` (or `|b.hi| = 1` with the words of
        /// opposite sign).
        fn frexp(self) -> (Self, i32) {
            let (mut man, mut exponent) = frexp(self.hi);
            let mut b1 = ldexp(self.lo, -exponent);
            if man.abs() == 0.5 && man * b1 < 0.0 {
                man *= 2.0;
                b1 *= 2.0;
                exponent -= 1;
            }
            (Self { hi: man, lo: b1 }, exponent)
        }

        /// `exp`: reduce by `m log 2` and a factor 512, Taylor series, square nine times.
        fn exp(self) -> Self {
            const K: f64 = 512.0;
            const INV_K: f64 = 1.0 / K;
            if self.hi <= -709.0 {
                return Self::new(0.0);
            }
            if self.hi >= 709.0 {
                return Self::splat(f64::INFINITY);
            }
            if self.eq_f64(0.0) {
                return Self::new(1.0);
            }
            if self.eq_f64(1.0) {
                return DD_E;
            }
            let m = (self.hi / DD_LOG2.hi + 0.5).floor();
            let r = self.sub(DD_LOG2.mul_f64(m)).mul_pwr2(INV_K);
            let mut p = r.square();
            let mut s = r.add(p.mul_pwr2(0.5));
            p = p.mul(r);
            let mut t = p.mul(INV_FACT[0]);
            let mut i = 0;
            loop {
                s = s.add(t);
                p = p.mul(r);
                i += 1;
                t = p.mul(INV_FACT[i]);
                if !(t.hi.abs() > INV_K * DD_EPS && i < 5) {
                    break;
                }
            }
            s = s.add(t);
            for _ in 0..9 {
                s = s.mul_pwr2(2.0).add(s.square());
            }
            s.add_f64(1.0).ldexp(m as i32)
        }

        /// Natural log: one Newton step `x + a·exp(−x) − 1` from libm's `log(hi)`.
        fn ln(self) -> Self {
            if self.eq_f64(1.0) {
                return Self::new(0.0);
            }
            if self.hi <= 0.0 {
                return Self::splat(f64::NAN);
            }
            let x = Self::new(self.hi.ln());
            x.add(self.mul(x.neg().exp())).sub_f64(1.0)
        }

        fn ln_1p(self) -> Self {
            if self.hi <= -1.0 {
                return Self::splat(f64::NEG_INFINITY);
            }
            let la = self.hi.ln_1p();
            let elam1 = cephes_expm1(la);
            let mut ll = (self.lo / (1.0 + self.hi)).ln_1p();
            if self.hi > 0.0 {
                ll -= (elam1 - self.hi) / (elam1 + 1.0);
            }
            Self::new(la).add_f64(ll)
        }
    }

    /// An x87 80-bit `long double`, `±sig · 2^exp` with the top bit of `sig` set (or zero).
    ///
    /// `_smirnovi` evaluates three bracket expressions in `long double` (the constant
    /// `SCIPY_El` and the literals `2.0L` and `1.0L` promote them): each operation rounds to a
    /// 64-bit significand and the result rounds again to `double`. Plain `f64` arithmetic lands
    /// an ulp away often enough to change the Newton iterates (7 of 2016 SciPy roots over
    /// n ≤ 1000 moved by 1–2 ulp), so the rounding is emulated exactly here.
    #[derive(Clone, Copy, Debug)]
    struct X87 {
        neg: bool,
        sig: u64,
        exp: i32,
    }

    impl X87 {
        /// xsf `SCIPY_El`, e rounded to a 64-bit significand.
        const E: Self = Self {
            neg: false,
            sig: 0xadf8_5458_a2bb_4a9b,
            exp: -62,
        };

        /// The exact value `±mant · 2^exp` rounded to 64 bits, to nearest, ties to even.
        fn round(neg: bool, mant: u128, exp: i32) -> Self {
            if mant == 0 {
                return Self {
                    neg: false,
                    sig: 0,
                    exp: 0,
                };
            }
            let bits = 128 - mant.leading_zeros();
            if bits <= 64 {
                let shift = 64 - bits;
                return Self {
                    neg,
                    sig: (mant << shift) as u64,
                    exp: exp - shift as i32,
                };
            }
            let shift = bits - 64;
            let rem = mant & ((1_u128 << shift) - 1);
            let half = 1_u128 << (shift - 1);
            let mut sig = (mant >> shift) as u64;
            let mut exp = exp + shift as i32;
            if rem > half || (rem == half && sig & 1 == 1) {
                sig = sig.wrapping_add(1);
                if sig == 0 {
                    sig = 1 << 63;
                    exp += 1;
                }
            }
            Self { neg, sig, exp }
        }

        fn from_f64(x: f64) -> Self {
            let bits = x.to_bits();
            let biased = ((bits >> 52) & 0x7ff) as i32;
            let frac = bits & ((1 << 52) - 1);
            let (mant, exp) = if biased == 0 {
                (frac, -1074)
            } else {
                (frac | (1 << 52), biased - 1075)
            };
            Self::round(bits >> 63 == 1, u128::from(mant), exp)
        }

        fn from_i64(v: i64) -> Self {
            Self::round(v < 0, u128::from(v.unsigned_abs()), 0)
        }

        /// Round to `double` (every value `_smirnovi` converts lies in the normal range, where
        /// both parts below are exact and the one `f64` addition rounds the 64-bit value).
        fn to_f64(self) -> f64 {
            if self.sig == 0 {
                return 0.0;
            }
            let hi = ldexp((self.sig >> 11) as f64, self.exp + 11);
            let lo = ldexp((self.sig & 0x7ff) as f64, self.exp);
            if self.neg { -(hi + lo) } else { hi + lo }
        }

        fn add(self, rhs: Self) -> Self {
            if rhs.sig == 0 {
                return self;
            }
            if self.sig == 0 {
                return rhs;
            }
            let (big, small) = if (self.exp, self.sig) >= (rhs.exp, rhs.sig) {
                (self, rhs)
            } else {
                (rhs, self)
            };
            // 62 guard bits below both significands; bits shifted out of the smaller one are
            // folded into its last bit, which keeps every rounding decision exact.
            let a = u128::from(big.sig) << 62;
            let shifted = u128::from(small.sig) << 62;
            let d = (big.exp - small.exp) as u32;
            let b = if d >= 127 {
                1
            } else {
                (shifted >> d) | u128::from(shifted & ((1_u128 << d) - 1) != 0)
            };
            let mant = if big.neg == small.neg { a + b } else { a - b };
            Self::round(big.neg, mant, big.exp - 62)
        }

        fn sub(self, rhs: Self) -> Self {
            self.add(Self {
                neg: !rhs.neg,
                ..rhs
            })
        }

        fn div(self, rhs: Self) -> Self {
            // (sig·2^64) / rhs.sig has 64 or 65 bits; two more quotient bits and a sticky bit
            // below them keep the value on the correct side of every rounding boundary.
            let den = u128::from(rhs.sig);
            let num = u128::from(self.sig) << 64;
            let (q, r) = (num / den, num % den);
            let (q2, r2) = ((r << 2) / den, (r << 2) % den);
            let mant = (((q << 2) | q2) << 1) | u128::from(r2 != 0);
            Self::round(self.neg != rhs.neg, mant, self.exp - rhs.exp - 67)
        }

        fn sqrt(self) -> Self {
            // Radicand in [2^126, 2^128) with an even exponent; round the 64-bit integer root up
            // when the remainder exceeds the root (sqrt never lands exactly on a tie).
            let s = if (self.exp - 63) % 2 == 0 { 63 } else { 64 };
            let rad = u128::from(self.sig) << s;
            let root = rad.isqrt();
            let rem = rad - root * root;
            Self::round(false, root + u128::from(rem > root), (self.exp - s) / 2)
        }
    }

    /// C `std::clamp(v, lo, hi)`, which (unlike `f64::clamp`) does not require `lo <= hi`.
    fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
        if v < lo {
            lo
        } else if hi < v {
            hi
        } else {
            v
        }
    }

    /// `a^m`: libm `pow` of the high word, corrected to first order in `lo/hi` (`pow_D`).
    fn pow_dd(a: Dd, m: i64) -> Dd {
        if m <= 0 {
            if m == 0 {
                return Dd::new(1.0);
            }
            return pow_dd(a, -m).rdiv_f64(1.0);
        }
        if a.eq_f64(0.0) {
            return Dd::new(0.0);
        }
        let mf = m as f64;
        let ans = a.hi.powf(mf);
        let r = a.lo / a.hi;
        let mut adj = mf * r;
        if adj.abs() > 1e-8 {
            if adj.abs() < 1e-4 {
                // First two Taylor terms of (1 + r)^m.
                adj += (mf * r) * ((m - 1) as f64 / 2.0 * r);
            } else {
                adj = cephes_expm1(mf * r.ln_1p());
            }
        }
        Dd::new(ans).add_f64(ans * adj)
    }

    /// `(a + b)^m` rounded to `double` (`pow2`).
    fn pow2(a: f64, b: f64, m: i64) -> f64 {
        pow_dd(Dd::new(a).add_f64(b), m).hi
    }

    /// xsf `nextPowerOf2`: `|x + x·2^-52|` (its `int` round trip never changes the value).
    fn next_power_of_2(x: f64) -> f64 {
        let l = (ldexp(x, 1 - 53) + x).abs();
        if l == 0.0 { x.abs() } else { l }
    }

    /// `a^m` as `(significand, binary exponent)`, which cannot underflow (`pow2Scaled_D`).
    fn pow2_scaled(a: Dd, m: i64) -> (Dd, i64) {
        if m <= 0 {
            if m == 0 {
                return (Dd::new(1.0), 0);
            }
            let (ans, e1) = pow2_scaled(a, -m);
            let (ans, e2) = ans.rdiv_f64(1.0).frexp();
            return (ans, -e1 + i64::from(e2));
        }
        let (y, ye) = a.frexp();
        let ye = i64::from(ye);
        if m == 1 {
            return (y, ye);
        }
        let mut max_expt = SM_MAX_EXPONENT;
        let mf = m as f64;
        let neg_max = -(SM_MAX_EXPONENT as f64);
        // y^max_expt must stay >= 2^-960; a cheap test before calling log().
        if mf * (y.hi - 1.0) / y.hi < neg_max * LN_2 {
            let lg2y = y.hi.ln() / LN_2;
            let lg_ans = mf * lg2y;
            if lg_ans <= neg_max {
                max_expt = (next_power_of_2(neg_max / lg2y + 1.0) / 2.0) as i64;
            }
        }
        if m <= max_expt {
            let (ans, ans_e) = pow_dd(y, m).frexp();
            return (ans, i64::from(ans_e) + m * ye);
        }
        // y^m = (y^max_expt)^q · y^r
        let q = m / max_expt;
        let r = m % max_expt;
        let (y2r, y2r_e) = pow2_scaled(y, r);
        let (y2m, y2m_e) = pow2_scaled(y, max_expt);
        let (y2mq, y2mq_e) = pow2_scaled(y2m, q);
        let (ans, ans_e) = y2r.mul(y2mq).frexp();
        (
            ans,
            i64::from(ans_e) + (y2mq_e + y2m_e * q) + y2r_e + m * ye,
        )
    }

    /// `((a + b) / (c + d))^m` (`pow4_D`).
    fn pow4_dd(a: f64, b: f64, c: f64, d: f64, m: i64) -> Dd {
        if m <= 0 {
            if m == 0 {
                return Dd::new(1.0);
            }
            return pow4_dd(c, d, a, b, -m);
        }
        let num = Dd::new(a).add_f64(b);
        let den = Dd::new(c).add_f64(d);
        if num.eq_f64(0.0) {
            return if den.eq_f64(0.0) {
                Dd::splat(f64::NAN)
            } else {
                Dd::new(0.0)
            };
        }
        if den.eq_f64(0.0) {
            return Dd::splat(if num.lt_f64(0.0) {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        pow_dd(num.div(den), m)
    }

    /// `m · log((a + b) / (c + d))` rounded to `double` (`logpow4`).
    fn logpow4(a: f64, b: f64, c: f64, d: f64, m: i64) -> f64 {
        if m == 0 {
            return 0.0;
        }
        let num = Dd::new(a).add_f64(b);
        let den = Dd::new(c).add_f64(d);
        if num.eq_f64(0.0) {
            return if den.eq_f64(0.0) {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        if den.eq_f64(0.0) {
            return f64::INFINITY;
        }
        let x = num.div(den);
        let ans = if (0.5..=1.5).contains(&x.hi) {
            num.sub(den).div(den).ln_1p()
        } else {
            x.ln()
        };
        ans.mul_f64(m as f64).hi
    }

    /// `floor(n x)` and the remainder, exactly, as `(alpha, floor, n·x)`; a remainder that
    /// rounds to 1 carries into the floor (`modNX`).
    fn mod_nx(n: i64, x: f64) -> (f64, i64, f64) {
        let nx = Dd::new(x).mul_f64(n as f64);
        let nx_floor = nx.floor();
        let mut alpha = nx.sub(nx_floor).hi;
        let mut nxfloor = nx_floor.hi as i64;
        if alpha == 1.0 {
            nxfloor += 1;
            alpha = 0.0;
        }
        (alpha, nxfloor, nx.hi)
    }

    /// C(n, j) held as (significand, exponent), advanced to C(n, j + 1) (`updateBinomial`).
    fn update_binomial(cman: &mut Dd, cexpt: &mut i64, n: i64, j: i64) {
        let rat = Dd::new((n - j) as f64).div_f64(j as f64 + 1.0);
        let (man, expt) = cman.mul(rat).frexp();
        *cexpt += i64::from(expt);
        *cman = man;
    }

    /// `A_v(n, x) = C(n, v) (1 − x − v/n)^(n−v) (x + v/n)^(v−1)` (`computeAv`).
    fn compute_av(n: i64, x: f64, v: i64, cman: Dd, cexpt: i64) -> Dd {
        let nf = n as f64;
        let t2x = Dd::new((n - v) as f64).div_f64(nf).sub_f64(x);
        let (t2, t2e) = pow2_scaled(t2x, n - v);
        let t1x = Dd::new(v as f64).div_f64(nf).add_f64(x);
        let (t1, t1e) = pow2_scaled(t1x, v - 1);
        // The exponent stays far inside i32 for the n <= 10^6 that are summed; beyond ±2^30 the
        // value is 0 or inf either way.
        let expt = (cexpt + t1e + t2e).clamp(-(1 << 30), 1 << 30) as i32;
        t1.mul(t2).mul(cman).ldexp(expt)
    }

    /// (sf, cdf, pdf) of `D_n^+`, computed together (`ThreeProbs`).
    #[derive(Clone, Copy, Debug)]
    pub(super) struct ThreeProbs {
        pub(super) sf: f64,
        pub(super) cdf: f64,
        pub(super) pdf: f64,
    }

    /// xsf `_smirnov(n, x)`.
    pub(super) fn smirnov3(n: i64, x: f64) -> ThreeProbs {
        let probs = |sf, cdf, pdf| ThreeProbs { sf, cdf, pdf };
        if !(n > 0 && (0.0..=1.0).contains(&x)) {
            return probs(f64::NAN, f64::NAN, f64::NAN);
        }
        if n == 1 {
            return probs(1.0 - x, x, 1.0);
        }
        if x == 0.0 {
            return probs(1.0, 0.0, 1.0);
        }
        if x == 1.0 {
            return probs(0.0, 1.0, 0.0);
        }
        let (alpha, nxfl, nx) = mod_nx(n, x);
        let mut n1mxfl = n - nxfl - i64::from(alpha != 0.0);
        let mut n1mxceil = n - nxfl;
        // With alpha == 0 the last term belongs to neither sum.
        if alpha == 0.0 {
            n1mxfl -= 1;
            n1mxceil += 1;
        }
        // x <= 1/n
        if nxfl == 0 || (nxfl == 1 && alpha == 0.0) {
            let t = pow2(1.0, x, n - 1);
            let mut pdf = (nx + 1.0) * t / (1.0 + x);
            let cdf = x * t;
            // Adjust if x = 1/n exactly.
            if nxfl == 1 {
                pdf -= 0.5;
            }
            return probs(1.0 - cdf, cdf, pdf);
        }
        let nf = n as f64;
        // The sf underflows. (xsf forms -2n in a C int, which overflows above 2^30 and skips
        // this test; the branches it falls to return the same (0, 1, 0).)
        if -2.0 * nf * x * x < MINLOG {
            return probs(0.0, 1.0, 0.0);
        }
        // x >= 1 - 1/n
        if nxfl >= n - 1 {
            let sf = pow2(1.0, -x, n);
            return probs(sf, 1.0 - sf, nf * sf / (1.0 - x));
        }
        // n too large to sum: p ~ exp(-(6nx + 1)^2 / 18n).
        if n > SMIRNOV_MAX_COMPUTE_N {
            let s = 6.0 * nf * x + 1.0;
            // xsf writes std::pow(s, 2), which compiles to s·s.
            let logp = -(s * s) / 18.0 / nf;
            let (sf, cdf) = if logp < -LN_2 {
                let sf = logp.exp();
                (sf, 1.0 - sf)
            } else {
                let cdf = -cephes_expm1(logp);
                (1.0 - cdf, cdf)
            };
            return probs(sf, cdf, (6.0 * nf * x + 1.0) * 2.0 * sf / 3.0);
        }
        // The upper sum alternates in sign and loses ~1.6 bits per term: use it only when it
        // has very few terms.
        let n_upper_terms = n - n1mxceil + 1;
        let use_upper = (n_upper_terms <= 1 && x < 0.5)
            || (n >= SM_UPPERSUM_MIN_N
                && n_upper_terms <= SM_UPPER_MAX_TERMS
                && x <= 0.5 / nf.sqrt());
        let vmid = n / 2;
        let one_over_x = Dd::new(1.0).div_f64(x);
        let (start, step, n_terms, mut aj, daj_coeff) = if use_upper {
            let aj = pow4_dd(1.0, x, 1.0, 0.0, n - 1);
            let coeff = Dd::new(1.0)
                .add_f64(x)
                .rdiv_f64((n - 1) as f64)
                .add(one_over_x);
            (n, -1, n - n1mxceil + 1, aj, coeff)
        } else {
            let aj = pow4_dd(1.0, -x, 1.0, 0.0, n).div_f64(x);
            let coeff = Dd::new((n - 1) as f64)
                .mul_f64(x)
                .rsub_f64(-1.0)
                .div(Dd::new(1.0).sub_f64(x))
                .div_f64(x)
                .add(one_over_x);
            (0, 1, n1mxfl + 1, aj, coeff)
        };
        let daj = aj.mul(daj_coeff);
        let mut aj_sum = Dd::new(0.0).add(aj);
        let mut daj_sum = Dd::new(0.0).add(daj);
        let mut cman = Dd::new(1.0);
        let mut cexpt = 0;
        update_binomial(&mut cman, &mut cexpt, n, 0);
        let mut j = 1;
        while j < n_terms {
            let v = start + j * step;
            aj = compute_av(n, x, v, cman, cexpt);
            if aj.hi.is_finite() && !aj.eq_f64(0.0) {
                // coeff = 1/x + (v-1)/(x+v/n) - (n-v)/(1-x-v/n). SciPy's C forms n·(v-1)
                // and (n-v)·n as ints, which overflow from n = 46342 on (46341·46340 is the
                // last product below 2^31) and corrupt its pdf and smirnovi's Newton steps.
                // fsci deliberately does not reproduce that UB: the i64 products are exact.
                let coeff = Dd::new((nxfl + v) as f64)
                    .add_f64(alpha)
                    .rdiv_f64((n * (v - 1)) as f64)
                    .sub(
                        Dd::new((n - nxfl - v) as f64)
                            .sub_f64(alpha)
                            .rdiv_f64(((n - v) * n) as f64),
                    )
                    .add(one_over_x);
                aj_sum = aj_sum.add(aj);
                daj_sum = daj_sum.add(aj.mul(coeff));
            }
            // Safe to stop early?
            if !aj.eq_f64(0.0) {
                if (4 * (n_terms - j)) as f64 * aj.hi.abs() < f64::EPSILON * aj_sum.hi
                    && j != n_terms - 1
                {
                    break;
                }
            } else if j > vmid {
                break;
            }
            update_binomial(&mut cman, &mut cexpt, n, j);
            j += 1;
        }
        let deriv = daj_sum.mul_f64(x).hi;
        let prob = aj_sum.mul_f64(x).hi;
        let (sf, cdf, pdf) = if step < 0 {
            (1.0 - prob, prob, deriv)
        } else {
            (prob, 1.0 - prob, -deriv)
        };
        // std::fmax(0, pdf) sends NaN and -0.0 to +0.0 as well.
        let pdf = if pdf > 0.0 { pdf } else { 0.0 };
        probs(sf.clamp(0.0, 1.0), cdf.clamp(0.0, 1.0), pdf)
    }

    /// xsf `_smirnovi(n, psf, pcdf)`: the `x` with `smirnov(n, x) = psf` and
    /// `smirnovc(n, x) = pcdf`.
    // `x < a || x > b` is kept as xsf writes it: `!(a..=b).contains(&x)` differs on NaN.
    #[allow(clippy::manual_range_contains)]
    pub(super) fn smirnovi(n: i64, psf: f64, pcdf: f64) -> f64 {
        if !(n > 0 && (0.0..=1.0).contains(&psf) && (0.0..=1.0).contains(&pcdf)) {
            return f64::NAN;
        }
        if (1.0 - pcdf - psf).abs() > 4.0 * f64::EPSILON {
            return f64::NAN;
        }
        if pcdf == 0.0 {
            return 0.0;
        }
        if psf == 0.0 {
            return 1.0;
        }
        if n == 1 {
            return pcdf;
        }
        let nf = n as f64;
        // psf very close to 0: the root lies in ((n-1)/n, 1), where psf = (1-x)^n exactly.
        let psfrootn = psf.powf(1.0 / nf);
        if n < 150 && nf * psfrootn <= 1.0 {
            return 1.0 - psfrootn;
        }
        let logpcdf = if pcdf < 0.5 {
            pcdf.ln()
        } else {
            (-psf).ln_1p()
        };
        // Bracket and starting point for Newton-Raphson.
        let maxlogpcdf = logpow4(1.0, 0.0, nf, 0.0, 1) + logpow4(nf, 1.0, nf, 0.0, n - 1);
        let (mut a, mut b, mut x);
        if logpcdf <= maxlogpcdf {
            // 0 < x <= 1/n: pcdf = x (1+x)^(n-1). One Newton step on z e^(z-1) = R.
            let xmin = X87::from_f64(pcdf).div(X87::E).to_f64();
            let xmax = pcdf;
            let p1 = pow4_dd(nf, 1.0, nf, 0.0, n - 1).hi / nf;
            let r = pcdf / p1;
            if r >= 1.0 {
                // R > 1 is truncation error at x = 1/n.
                return 1.0 / nf;
            }
            let z0 = (r * r + r * (1.0 - r).exp()) / (1.0 + r);
            x = z0 / nf;
            a = (xmin * (1.0 - 4.0 * f64::EPSILON)).max(0.0);
            b = (xmax * (1.0 + 4.0 * f64::EPSILON)).min(1.0 / nf);
            x = clamp(x, a, b);
        } else {
            // 1/n < x < (n-1)/n. (xsf also scales xmin and xmax by 1 ∓ 4 eps, into
            // variables it then overwrites.)
            let xmin = 1.0 - psfrootn;
            let logpsf = if psf < 0.5 { psf.ln() } else { (-pcdf).ln_1p() };
            // std::sqrt(-logpsf / (2.0L * n)) and xmax - 1.0L / (6 * n). SciPy's C forms 6n as
            // an int, which overflows for n > 357913941 and moves its starting point; fsci
            // deliberately does not reproduce that UB and keeps 6n exact.
            let xmax = X87::from_f64(-logpsf)
                .div(X87::from_i64(2 * n))
                .sqrt()
                .to_f64();
            let xmax6 = X87::from_f64(xmax)
                .sub(X87::from_i64(1).div(X87::from_i64(6 * n)))
                .to_f64();
            a = xmin.max(1.0 / nf);
            b = xmax.min(1.0 - 1.0 / nf);
            x = xmax6;
        }
        if x < a || x > b {
            x = (a + b) / 2.0;
        }
        // Newton-Raphson on smirnov(n, x) - psf or pcdf - smirnovc(n, x), whichever has the
        // smaller p, falling back to bisection of the bracket.
        let mut dxold = b - a;
        let mut dx = dxold;
        let mut iterations = 0;
        loop {
            let x0 = x;
            let p = smirnov3(n, x0);
            let df = if pcdf < 0.5 { pcdf - p.cdf } else { p.sf - psf };
            let dfdx = -p.pdf;
            if df == 0.0 {
                return x;
            }
            if df > 0.0 && x > a {
                a = x;
            } else if df < 0.0 && x < b {
                b = x;
            }
            let deltax = if dfdx == 0.0 {
                x = (a + b) / 2.0;
                x0 - x
            } else {
                let deltax = df / dfdx;
                x = x0 - deltax;
                deltax
            };
            if (a..=b).contains(&x)
                && ((2.0 * deltax).abs() <= dxold.abs() || dxold.abs() < 256.0 * f64::EPSILON)
            {
                dxold = dx;
                dx = deltax;
            } else {
                dxold = dx;
                dx /= 2.0;
                x = (a + b) / 2.0;
            }
            // Not purely relative: near psf = 1 the root is close to 0.
            let atol = if psf < 0.5 { 0.0 } else { XTOL };
            if (x - x0).abs() <= atol + RTOL * x0.abs() {
                return x;
            }
            iterations += 1;
            if iterations > KOLMOG_MAXITER {
                return x;
            }
        }
    }
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

/// Multiply x by 2 raised to the power exp, with one rounding: C's `ldexp`, so numpy's.
///
/// musl's `scalbn`. `x * 2.0.powi(exp)` is wrong wherever 2^exp itself is not representable
/// but the product is: ldexp(0.5, 1024) came out inf, not 2^1023, and ldexp(2^52, -1100) came
/// out 0, not 2^-1048 (frankenscipy-9y1o1). Here the scaling is split into steps of 2^1023,
/// or of 2^-969 so a subnormal result is rounded once, before the final power-of-two
/// multiply, whose exponent is then always representable.
#[must_use]
pub fn ldexp(x: f64, exp: i32) -> f64 {
    const P1023: f64 = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
    const PM969: f64 = f64::from_bits(0x0360_0000_0000_0000); // 2^-1022 · 2^53
    let mut y = x;
    let mut n = exp;
    if n > 1023 {
        y *= P1023;
        n -= 1023;
        if n > 1023 {
            y *= P1023;
            n = (n - 1023).min(1023);
        }
    } else if n < -1022 {
        y *= PM969;
        n += 1022 - 53;
        if n < -1022 {
            y *= PM969;
            n = (n + 1022 - 53).max(-1022);
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
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
    // Numerically stable softplus(x) = ln(1 + exp(x)):
    // For x > 0: ln(1 + exp(x)) = x + ln(1 + exp(-x)) = x + ln_1p(exp(-x))
    // For x <= 0: ln(1 + exp(x)) = ln_1p(exp(x))
    // Avoids overflow for large positive x and maintains full 53-bit precision everywhere.
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
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
    if delta < 0.0 {
        return f64::INFINITY;
    }
    if delta == 0.0 {
        return 0.0;
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
    if delta < 0.0 {
        return f64::INFINITY;
    }
    if delta == 0.0 {
        return 0.0;
    }

    let ratio = x / delta;
    delta * delta * (0.5 * (ratio * ratio).ln_1p()).exp_m1()
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
    if x > 0.0 { x } else { alpha * x.exp_m1() }
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
        SCALE * ALPHA * x.exp_m1()
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
    x * softplus_scalar(x).tanh()
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
/// Returns -∞ for p = 0, +∞ for p = 1, and NaN for p < 0 or p > 1.
#[must_use]
pub fn cloglog(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
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
/// Returns -∞ for p = 0, +∞ for p = 1, and NaN for p < 0 or p > 1.
#[must_use]
pub fn loglog(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
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
/// Returns -∞ for p = 0, +∞ for p = 1, and NaN for p < 0 or p > 1.
pub fn cauchit(p_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real("cauchit", p_tensor, mode, |p| Ok(cauchit_scalar(p)))
}

#[must_use]
pub fn cauchit_scalar(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
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
        alpha * (x / alpha).exp_m1()
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
    if x.is_nan() || x > 0.0 {
        return f64::NAN; // log of negative number for x > 0
    }
    if x == 0.0 {
        return f64::NEG_INFINITY; // log(0)
    }
    // For x < 0: log(1 - exp(x))
    // Martin Mächler (2012) "Accurately Computing log(1 - exp(-a))":
    // For x > -ln(2) (x near 0): 1 - exp(x) = -expm1(x), so log(1 - exp(x)) = ln(-expm1(x))
    // For x <= -ln(2): exp(x) <= 0.5, so log(1 - exp(x)) = ln_1p(-exp(x))
    if x > -std::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
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
                // RELATIVE: an absolute 1e-12 passed the SIMD path's 0 against the scalar
                // -1.8e-33 at x = 12 (frankenscipy-iulu0).
                let rel = (simd[k] - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
                max_abs = max_abs.max(rel);
            }
        }
        assert!(
            max_abs < 1e-13,
            "log_ndtr simd max rel diff vs scalar = {max_abs:e}"
        );
    }

    #[test]
    fn log_ndtr_tensor_path_keeps_the_upper_tail() -> Result<(), String> {
        // frankenscipy-iulu0: 64 elements take the SIMD path. For x >= -1 it must be
        // log1p(-Phi(-x)), SciPy's form, not ln(Phi(x)) with Phi(x) rounded toward 1, which
        // was 0 at x = 10, 15 and 30. (x, scipy.special.log_ndtr(x)).
        let cases = [
            (-0.5, -1.1759117615936188),
            (0.0, -0.6931471805599453),
            (1.0, -0.1727537790234499),
            (3.0, -0.0013508099647481925),
            (5.0, -2.8665161296376294e-07),
            (10.0, -7.61985302416047e-24),
            (15.0, -3.6709661993126986e-51),
            (30.0, -4.906713927147908e-198),
            (-3.0, -6.60772622151035),
            (-7.9, -34.20622817098172),
        ];
        let xs: Vec<f64> = (0..64).map(|i| cases[i % cases.len()].0).collect();
        let out = log_ndtr(&SpecialTensor::RealVec(xs), RuntimeMode::Strict)
            .map_err(|err| err.to_string())?;
        let values = expect_real_vec(out)?;
        for (i, got) in values.iter().enumerate() {
            let (x, want) = cases[i % cases.len()];
            let rel = ((got - want) / want).abs();
            assert!(
                rel < 1e-13,
                "tensor log_ndtr({x}) = {got:e}, SciPy {want:e}"
            );
        }
        Ok(())
    }

    #[test]
    fn agm_is_scipys_agm_bit_for_bit() {
        // frankenscipy-89pgv: agm is SciPy's _agm.pxd. SciPy 1.17.1's bits through the
        // (pi/4)(a+b)/ellpk form, the extreme-magnitude iteration (1e-320, 1e300, 1e-200, inf),
        // zeros, a == b and both-negative arguments. The old iteration was NaN for every zero
        // or negative argument. (a, b, scipy.special.agm(a, b)).
        let cases: [(f64, f64, f64); 13] = [
            (24.0, 6.0, 13.458171481725614),
            (1.0, 2.0, 1.4567910310469068),
            (3.0, 3.0, 3.0),
            (0.0, 5.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1e-320, 5.0, 0.01061602831874734),
            (-2.0, -8.0, -4.486057160575205),
            (1e300, 1.0, 2.269406194157821e297),
            (1e-200, 1e-100, 6.781055745575451e-103),
            (f64::INFINITY, 5.0, f64::INFINITY),
            (0.5, 99.5, 23.398618868442995),
            (12.25, 88.0, 41.02213273165341),
        ];
        for (a, b, want) in cases {
            let got = agm(std::hint::black_box(a), std::hint::black_box(b));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "agm({a:?}, {b:?}) = {got:?}, SciPy {want:?}"
            );
        }
        for (a, b) in [(-1.0, 5.0), (0.0, f64::INFINITY), (f64::NAN, 1.0)] {
            assert!(agm(a, b).is_nan(), "agm({a}, {b}) is NaN in SciPy");
        }
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
    fn wofz_mid_region_matches_scipy() {
        // |z| < 4, the region a Weideman rational used to serve. References from
        // scipy.special.wofz (1.17.1), 13 digits, asserted to 1e-11. Includes a
        // near-real-axis point (im=0.001) where a naive contour quadrature would
        // miss the pole. Bit parity is `wofz_is_scipy_faddeeva_bit_for_bit`.
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
    fn poch_is_scipys_cephes_poch_bit_for_bit() {
        // frankenscipy-zw56i: poch is xsf's Cephes poch. SciPy 1.17.1's bits through the
        // reduction loops, m = 0, the large-a series, both pole outcomes (+inf, 0), the
        // both-pole finite limits and the gamma-ratio path. The first point is the sweep's
        // worst, 7.4e-11 off before. (a, m, scipy.special.poch(a, m)).
        let cases: [(f64, f64, f64); 14] = [
            (-3.00000000130809, -3.000005999982, -1.8163808136222019e-06),
            (2.0, 3.0, 24.0),
            (0.5, 2.0, 0.75),
            (-2.0, -2.0, 0.08333333333333333),
            (-5.0, 3.0, -60.0),
            (-4.3, 0.5, -2.938123324828691),
            (1.0, 0.0, 1.0),
            (-3.0, 0.5, 0.0),
            (0.5, -3.5, f64::INFINITY),
            (-2.5, 2.5, f64::INFINITY),
            (12345.6, 0.7, 731.2314255611542),
            (7.25, -2.75, 0.010067439447618393),
            (3.5, 4.25, 920.1022396922906),
            (
                -2.9999999999648965,
                -3.999003002990991,
                4.2000107014479106e-11,
            ),
        ];
        for (a, m, want) in cases {
            let got = poch(std::hint::black_box(a), std::hint::black_box(m));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "poch({a:?}, {m:?}) = {got:?}, SciPy {want:?}"
            );
        }
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
        assert_eq!(logit_scalar(0.0, m).unwrap(), f64::NEG_INFINITY);
        assert_eq!(logit_scalar(1.0, m).unwrap(), f64::INFINITY);
        assert!(logit_scalar(-0.1, m).unwrap().is_nan());
        assert!(logit_scalar(1.1, m).unwrap().is_nan());
        assert!(logit_scalar(f64::NAN, m).unwrap().is_nan());
        let mh = RuntimeMode::Hardened;
        assert!(logit_scalar(-0.1, mh).is_err());
        assert!(logit_scalar(0.0, mh).is_err());
        assert!(logit_scalar(1.0, mh).is_err());
        assert!(logit_scalar(1.1, mh).is_err());
    }

    #[test]
    fn logit_is_scipy_xsf_bit_for_bit_and_the_batch_is_the_scalar() -> Result<(), String> {
        // SciPy 1.17.1 values: either side of both branch points and near 1/2, where
        // log(p/(1-p)) alone was up to 3e-8 relative off (0.5 - 1e-9).
        let cases = [
            (0.299_999_999_999_999_93, -0.847_297_860_387_203_9),
            (0.3, -0.847_297_860_387_203_7),
            (0.4, -0.405_465_108_108_164_3),
            (0.500_000_000_001, 3.999_911_513_119_514e-12),
            (0.499_999_999, -4.000_000_108_916_879e-9),
            (0.500_000_1, 3.999_999_997_894_629_5e-7),
            (0.6, 0.405_465_108_108_164_3),
            (0.65, 0.619_039_208_406_223_5),
            (0.650_000_000_000_000_1, 0.619_039_208_406_224_1),
            (1e-300, -690.775_527_898_213_7),
            (0.999_999_999_999_999_9, 36.736_800_569_677_1),
        ];
        let xs: Vec<f64> = cases
            .iter()
            .map(|c| c.0)
            .chain([0.0, -0.0, 1.0, -0.1, 1.1, f64::INFINITY, f64::NAN])
            .collect();
        let batch = match logit(&SpecialTensor::RealVec(xs.clone()), RuntimeMode::Strict) {
            Ok(SpecialTensor::RealVec(v)) => v,
            other => return Err(format!("expected real vector, got {other:?}")),
        };
        for (i, &p) in xs.iter().enumerate() {
            let scalar = logit_scalar(p, RuntimeMode::Strict).map_err(|e| e.to_string())?;
            assert_eq!(batch[i].to_bits(), scalar.to_bits(), "logit({p:e})");
        }
        for (i, &(p, want)) in cases.iter().enumerate() {
            assert_eq!(batch[i].to_bits(), f64::to_bits(want), "logit({p:e})");
        }
        let n = cases.len();
        assert_eq!(batch[n], f64::NEG_INFINITY);
        assert_eq!(batch[n + 1], f64::NEG_INFINITY);
        assert_eq!(batch[n + 2], f64::INFINITY);
        assert!(batch[n + 3..].iter().all(|v| v.is_nan()));

        let refused = SpecialTensor::RealVec(vec![0.4, 1.0]);
        let err = logit(&refused, RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        match logit(
            &SpecialTensor::RealVec(vec![0.4, f64::NAN]),
            RuntimeMode::Hardened,
        ) {
            Ok(SpecialTensor::RealVec(v)) => {
                assert_eq!(v[0].to_bits(), f64::to_bits(-0.405_465_108_108_164_3));
                assert!(v[1].is_nan());
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
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
    fn owens_t_is_scipy_patefield_tandy_bit_for_bit() {
        // frankenscipy-nb55y: owens_t is xsf's Patefield-Tandy owens_t.h, so SciPy 1.17.1's
        // values are pinned to the bit. One (h, a) per path: T1..T6 directly, every method the
        // a > 1 reflection reaches under both of its branches (ah <= 0.67 and above), the
        // closed forms (h = 0, a = 0, a = 1, a = inf, h = inf), table-cell boundaries, and the
        // bead's worst point, which the Gauss-Legendre rule had 2.5e-9 relative off. Where
        // one exists the point was chosen so that libm's expm1 in T1, or x * (1/sqrt 2) in
        // place of x / sqrt 2 in norm1/norm2, gives different bits. (h, a, scipy).
        let cases: [(f64, f64, f64); 35] = [
            (0.715, 0.206, 0.024951259674922427),             // T1
            (6.431, 0.3657, 3.117647270473573e-11),           // T2
            (3.978, 0.6239, 1.721385520235399e-05),           // T3
            (0.838, 0.0907, 0.01012356113847244),             // T4
            (1.81, 0.6509, 0.014746297532343123),             // T5
            (1.279, 0.999995, 0.045179233085271504),          // T6
            (0.366, 1.443, 0.13900437617885206),              // reflection, ah <= 0.67, T1
            (0.0565, 11.591, 0.23346166537400592),            // reflection, ah <= 0.67, T4
            (0.4751, 1.818, 0.14083649699332967),             // reflection, ah > 0.67, T1
            (6.5729, 38.253, 1.2336150135792441e-11),         // reflection, ah > 0.67, T2
            (6.5933, 1.583, 1.0753902765021123e-11),          // reflection, ah > 0.67, T3
            (0.178, 13.859, 0.21461812741317024),             // reflection, ah > 0.67, T4
            (1.3436, 1.842, 0.04461648383326246),             // reflection, ah > 0.67, T5
            (1.279, 1.000008, 0.04517943459455167),           // reflection, ah > 0.67, T6
            (3.2, 1.000004, 0.00034333290105256097),          // reflection, 1/a just below 1
            (0.1, 1.0000001, 0.12420687956748308),            // reflection, ah <= 0.67, a ~ 1
            (-4.872, -0.922, -2.7618437205809556e-07),        // the bead's worst point
            (0.02, 0.025, 0.00397724926088619),               // on both cell bounds
            (4.8, 0.99999, 3.9666376122455044e-07),           // on both cell bounds
            (2.33, 0.5, 0.004025149112463264),                // on both cell bounds
            (2.75, 0.999995, 0.0014854419163221643),          // a above the last a bound
            (5.5, 0.999993, 9.494781052601517e-09),           // a above the last a bound
            (6.5, 0.9995, 2.0080002918471168e-11),            // h above the last h bound
            (10.0, 0.5, 3.8099247740170695e-24),              // h above the last h bound
            (1e-300, 0.3, 0.046386789538871175),              // tiny h
            (0.3, 1e-300, 1.5215172481714229e-301),           // tiny a
            (1.5, 1.0, 0.031171999563740185),                 // a = 1 closed form
            (37.0, 1.0, 2.8627856112626135e-300),             // a = 1 deep in the tail
            (0.0, 0.5, 0.07379180882521663),                  // h = 0: atan(a)/(2 pi)
            (0.0, 3.0, 0.19879180882521663),                  // h = 0 through the reflection
            (1.25, f64::INFINITY, 0.05282488683342765),       // a = inf: Phi(-h)/2
            (-1.25, f64::NEG_INFINITY, -0.05282488683342765), // odd in a, even in h
            (f64::INFINITY, -0.5, -0.0),                      // h = inf keeps the sign of a
            (0.5, -0.0, 0.0),                                 // a = -0 is +0
            (-0.715, -0.206, -0.024951259674922427),          // odd in a, even in h
        ];
        for (h, a, want) in cases {
            let got = owens_t_scalar(std::hint::black_box(h), std::hint::black_box(a));
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "owens_t({h:?}, {a:?}) = {got:?}, SciPy {want:?}"
            );
        }
        // NaN in either argument is NaN, including a = 0 (the former kernel returned 0).
        for (h, a) in [(f64::NAN, 0.0), (0.5, f64::NAN), (f64::NAN, f64::INFINITY)] {
            let got = owens_t_scalar(std::hint::black_box(h), std::hint::black_box(a));
            assert!(got.is_nan(), "owens_t({h:?}, {a:?}) = {got:?}, SciPy NaN");
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

    /// frankenscipy-qu5po. A NaN or negative `a` made the bracket `hi = a + 4·√a + 10` NaN,
    /// and `x0.clamp(lo + 1e-300, hi)` panicked on the NaN bound. gammainccinv panicked too,
    /// through its delegation to gammaincinv. SciPy 1.17.1 is nan for all of these:
    /// gammaincinv(nan, y), gammaincinv(-1, y), gammainccinv(nan, y) and gammainccinv(-1, y)
    /// at y = 0, 0.3, 0.95 and 1.
    ///
    /// Must not change. The signed zero is not negative, so SciPy's p = 0 / p = 1 edges still
    /// answer: gammaincinv(-0.0, 0) = 0.0, gammaincinv(-0.0, 1) = inf,
    /// gammainccinv(-0.0, 0) = inf and gammainccinv(-0.0, 1) = 0.0. A guard written as
    /// `a <= 0.0` or `is_sign_negative` would break these. The finite path is also unchanged
    /// (existing goldens): gammaincinv(2, 0.5) = 1.6783469900166612 and
    /// gammainccinv(0.5, 0.5) = 0.2274682115597862.
    #[test]
    fn gammaincinv_nan_or_negative_shape_is_nan_not_a_panic() {
        // Interior y first, where the clamp panicked. The y = 0 and y = 1 edges did not panic,
        // but they answered 0 or inf where SciPy gives nan.
        for a in [f64::NAN, -1.0] {
            for y in [0.3, 0.95, 0.0, 1.0] {
                let p = gammaincinv_scalar(a, y);
                let q = gammainccinv_scalar(a, y);
                assert!(
                    p.is_nan(),
                    "gammaincinv({a}, {y}) = {p}, SciPy 1.17.1 gives nan"
                );
                assert!(
                    q.is_nan(),
                    "gammainccinv({a}, {y}) = {q}, SciPy 1.17.1 gives nan"
                );
            }
        }

        assert_eq!(gammaincinv_scalar(-0.0, 0.0), 0.0);
        assert_eq!(gammaincinv_scalar(-0.0, 1.0), f64::INFINITY);
        assert_eq!(gammainccinv_scalar(-0.0, 0.0), f64::INFINITY);
        assert_eq!(gammainccinv_scalar(-0.0, 1.0), 0.0);
        let p = gammaincinv_scalar(2.0, 0.5);
        assert!(
            ((p - 1.678_346_990_016_661_2) / 1.678_346_990_016_661_2).abs() < 1e-11,
            "gammaincinv(2, 0.5) = {p}, SciPy 1.17.1 gives 1.6783469900166612"
        );
        let q = gammainccinv_scalar(0.5, 0.5);
        assert!(
            ((q - 0.227_468_211_559_786_2) / 0.227_468_211_559_786_2).abs() < 1e-11,
            "gammainccinv(0.5, 0.5) = {q}, SciPy 1.17.1 gives 0.2274682115597862"
        );
    }

    /// frankenscipy-g9yid. At a = 0, -0.0 and inf, SciPy 1.17.1 answers only the y edges;
    /// gammaincinv(a, y) and gammainccinv(a, y) are nan for y = 0.3, 0.5 and 0.95. fsci's
    /// Newton loop ran with P(0, x) = P(inf, x) = NaN and bisected on NaN residuals. A float
    /// emulation of that loop ends at about 1.5e-323, or at inf for a = inf with y ≥ 0.5.
    ///
    /// Must not change, SciPy 1.17.1: the edges gammaincinv(a, 0) = 0.0,
    /// gammaincinv(a, 1) = inf, gammainccinv(a, 0) = inf and gammainccinv(a, 1) = 0.0 for
    /// a = 0, -0.0 and inf; and the finite path, gammaincinv(2, 0.5) = 1.6783469900166612 and
    /// gammainccinv(0.5, 0.5) = 0.2274682115597862.
    #[test]
    fn gammaincinv_zero_or_infinite_shape_is_nan_inside_the_unit_interval() {
        for a in [0.0, -0.0, f64::INFINITY] {
            for y in [0.3, 0.5, 0.95] {
                let p = gammaincinv_scalar(a, y);
                let q = gammainccinv_scalar(a, y);
                assert!(
                    p.is_nan(),
                    "gammaincinv({a}, {y}) = {p}, SciPy 1.17.1 gives nan"
                );
                assert!(
                    q.is_nan(),
                    "gammainccinv({a}, {y}) = {q}, SciPy 1.17.1 gives nan"
                );
            }
            assert_eq!(gammaincinv_scalar(a, 0.0), 0.0, "gammaincinv({a}, 0)");
            assert_eq!(
                gammaincinv_scalar(a, 1.0),
                f64::INFINITY,
                "gammaincinv({a}, 1)"
            );
            assert_eq!(
                gammainccinv_scalar(a, 0.0),
                f64::INFINITY,
                "gammainccinv({a}, 0)"
            );
            assert_eq!(gammainccinv_scalar(a, 1.0), 0.0, "gammainccinv({a}, 1)");
        }
        let p = gammaincinv_scalar(2.0, 0.5);
        assert!(
            ((p - 1.678_346_990_016_661_2) / 1.678_346_990_016_661_2).abs() < 1e-11,
            "gammaincinv(2, 0.5) = {p}, SciPy 1.17.1 gives 1.6783469900166612"
        );
        let q = gammainccinv_scalar(0.5, 0.5);
        assert!(
            ((q - 0.227_468_211_559_786_2) / 0.227_468_211_559_786_2).abs() < 1e-11,
            "gammainccinv(0.5, 0.5) = {q}, SciPy 1.17.1 gives 0.2274682115597862"
        );
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
    fn wrightomega_real_is_scipy_xsf_bit_for_bit() {
        // scipy.special.wrightomega 1.17.1 on real input. -18.96 was the regression: e^z was
        // returned from -18.4 down, 1e-8 relative off.
        for (z, want) in [
            (-40.0, 4.248354255291589e-18),
            (-18.96, 5.831450863789683e-09),
            (-5.0, 0.0066930004977309955),
            (-1.0, 0.27846454276107374),
            (0.5, 0.7662486081617502),
            (3.0, 2.207940031569323),
            (50.0, 46.167719165492095),
            (1e21, 1e21),
        ] {
            let got = wrightomega_scalar(z);
            assert_eq!(
                got.to_bits(),
                f64::to_bits(want),
                "wrightomega({z}) = {got:e}, SciPy {want:e}"
            );
        }
    }

    #[test]
    fn kolmogorov_and_kolmogi_match_scipy_xsf() {
        // scipy.special.kolmogorov 1.17.1, bit for bit, across its branches. x = 0.02 was the
        // regression: the 100-term alternating series returned a value far from SciPy's 1.0.
        for (x, want) in [
            (0.02, 1.0),
            (0.0406, 1.0),
            (0.1, 1.0),
            (0.5, 0.9639452436648751),
            (0.82, 0.5119717052984973),
            (0.9, 0.3927307079406543),
            (1.5, 0.022217962616525127),
            (3.2, 2.5508152590520792e-09),
        ] {
            let got = kolmogorov_scalar(x);
            assert_eq!(
                got.to_bits(),
                f64::to_bits(want),
                "kolmogorov({x}) = {got:e}, SciPy {want:e}"
            );
        }
        // scipy.special.kolmogi 1.17.1, within 2 ulp: the port matches SciPy's bits on 99.8%
        // of points and is 1 ulp off on the rest.
        for (p, want) in [
            (1e-10, 3.4437623401231106),
            (0.1, 1.2238478702170823),
            (0.5, 0.8275735551899059),
            (0.9, 0.5711732651063401),
            (0.999999, 0.2775393539988728),
        ] {
            let got = kolmogi_scalar(p);
            assert!(
                (got - want).abs() <= 2.0 * f64::EPSILON * want,
                "kolmogi({p}) = {got:e}, SciPy {want:e}"
            );
        }
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
        // smirnovi(n, smirnov(n, d)) recovers d to a few ulp away from the flat tails.
        for &n in &[20, 50, 100, 1000] {
            for &d in &[0.05, 0.2, 0.3, 0.4, 0.5] {
                let p = smirnov(n, d);
                if p > 0.01 && p < 0.99 {
                    let d_recovered = smirnovi(n, p);
                    assert!(
                        (d_recovered - d).abs() <= 1e-12 * d,
                        "smirnovi failed: n={n}, d={d}, p={p}, d_recovered={d_recovered}"
                    );
                }
            }
        }
    }

    /// `smirnov` against `scipy.special.smirnov` (SciPy 1.17.1) bit for bit, through every
    /// branch of xsf `_smirnov`: n = 1; d = 0 and 1; d < 1/n; d = 1/n exactly; d ≥ 1 − 1/n;
    /// the lower sum at small and large n (the `int` products SciPy's C overflows from
    /// n = 46342 on feed only the pdf, so the sf stays SciPy's there too); the three-term
    /// upper sum; the underflow cut-off; the n > 10^6 approximation (both its `exp`
    /// and cephes `expm1` arms, and n = i32::MAX). The old exp(−2nd²) path for n ≥ 1000
    /// missed (1000, 0.03) by 3.3e-3. frankenscipy-k4c2c
    #[test]
    fn smirnov_matches_scipy_bit_for_bit() {
        // (n, d, scipy.special.smirnov(n, d).to_bits())
        let cases: [(i32, f64, u64); 26] = [
            (1, 0.3, 0x3fe6_6666_6666_6666),
            (7, 0.0, 0x3ff0_0000_0000_0000),
            (7, 1.0, 0x0000_0000_0000_0000),
            (10, 0.05, 0x3fed_8493_724b_3766),
            (10, 0.1, 0x3fe8_745e_8744_b960),
            (5, 0.85, 0x3f13_e814_50ef_dca0),
            (3, 0.5, 0x3fc5_5555_5555_5555),
            (5, 0.3, 0x3fd5_f0c3_4c1a_8ac6),
            (10, 0.3, 0x3fc1_56de_aa8d_0a54),
            (20, 0.25, 0x3fb1_9ea1_c70c_c735),
            (50, 0.1, 0x3fd6_12f4_e4bb_5691),
            (100, 0.2, 0x3f32_314b_50b0_7424),
            (500, 0.1, 0x3f05_de6b_0930_dbda),
            (1000, 0.03, 0x3fc4_bd74_882c_c070),
            (100, 0.025, 0x3feb_c807_3493_2cc0),
            (1000, 0.0025, 0x3fef_8ccd_2dfa_bb66),
            (100_000, 1.5e-5, 0x3fef_ff8c_f42c_852f),
            (1000, 0.7, 0x0000_0000_0000_0000),
            (46341, 0.004, 0x3fcc_f9bf_a7b3_8e06),
            (100_000, 0.003, 0x3fc5_1db1_a439_2120),
            (1_000_000, 0.001, 0x3fc1_4fb6_0abe_10e3),
            (2_000_000, 0.0005, 0x3fd7_8953_ebf7_ec61),
            (2_000_000, 1e-5, 0x3fef_fcab_45f9_dbe3),
            (2_000_000, 1e-7, 0x3fef_ffff_be6d_2b90),
            (i32::MAX, 1e-5, 0x3fe4_d39e_17ac_f1bb),
            (20, 1e-300, 0x3ff0_0000_0000_0000),
        ];
        let bad: Vec<String> = cases
            .iter()
            .filter_map(|&(n, d, bits)| {
                let got = smirnov(n, d);
                (got.to_bits() != bits).then(|| {
                    format!(
                        "smirnov({n}, {d:e}) = {got:e}, SciPy {:e}",
                        f64::from_bits(bits)
                    )
                })
            })
            .collect();
        assert!(
            bad.is_empty(),
            "{} of {} differ from SciPy:\n{}",
            bad.len(),
            cases.len(),
            bad.join("\n")
        );
    }

    /// `smirnovi` against `scipy.special.smirnovi` (SciPy 1.17.1) bit for bit wherever SciPy's
    /// C does not overflow an `int`: n = 1; p = 0 and 1; the exact `1 − p^{1/n}` root; the
    /// d ≤ 1/n start (p near 1); the Newton path at small and large n, up to n = 46341 (the
    /// last n whose derivative products fit an int), at 2·10^6 and at 357913941 (the last n
    /// whose 6n does); and five roots whose `long double` bracket arithmetic moves the answer:
    /// with the brackets in plain `f64` they come out 1–2 ulp low or high. frankenscipy-k4c2c
    #[test]
    fn smirnovi_matches_scipy_bit_for_bit() {
        // (n, p, scipy.special.smirnovi(n, p).to_bits())
        let cases: [(i32, f64, u64); 22] = [
            (1, 0.3, 0x3fe6_6666_6666_6666),
            (5, 0.0, 0x3ff0_0000_0000_0000),
            (5, 1.0, 0x0000_0000_0000_0000),
            (5, 1e-6, 0x3fed_fb1e_a780_eac1),
            (100, 0.99, 0x3f77_56aa_c1fe_04b1),
            (1000, 0.999, 0x3f42_97c0_ed83_5b70),
            (50, 0.1, 0x3fc2_feb5_b46a_f6d7),
            (20, 0.5, 0x3fbf_c168_21c1_28ad),
            (10, 0.3, 0x3fcd_7f48_78c4_24d8),
            (500, 0.797_590_610_435_077_3, 0x3f8e_2143_eeff_02f4),
            (10, 0.736_725_688_351_261_4, 0x3fbb_b29a_34a4_4c91),
            (150, 0.768_144_465_873_938_9, 0x3f9d_4435_2ba5_27ca),
            (999, 0.851_869_681_922_895_8, 0x3f82_0241_331e_832e),
            (500, 0.9, 0x3f84_5abf_afa1_aca5),
            (10_000, 0.05, 0x3f89_07da_95df_3c02),
            (46_341, 0.5, 0x3f66_5f9f_a334_6ed6),
            (2_000_000, 0.2, 0x3f44_c858_d5aa_0b96),
            (2_000_000, 0.9, 0x3f25_42f7_d480_fff6),
            (357_913_941, 0.75, 0x3ef5_053b_ed3a_6900),
            (1000, 1e-100, 0x3fd5_6ccd_c2db_7495),
            (200, 1e-300, 0x3fee_fcf2_3b15_fc64),
            (10, 1e-300, 0x3ff0_0000_0000_0000),
        ];
        let bad: Vec<String> = cases
            .iter()
            .filter_map(|&(n, p, bits)| {
                let got = smirnovi(n, p);
                (got.to_bits() != bits).then(|| {
                    format!(
                        "smirnovi({n}, {p:e}) = {got:e}, SciPy {:e}",
                        f64::from_bits(bits)
                    )
                })
            })
            .collect();
        assert!(
            bad.is_empty(),
            "{} of {} differ from SciPy:\n{}",
            bad.len(),
            cases.len(),
            bad.join("\n")
        );
    }

    /// Where SciPy's C overflows an `int` (`n·(v−1)`, `(n−v)·n` in the Newton derivative for
    /// 46342 ≤ n ≤ 10^6; `6n` in the starting point above n = 357913941), fsci keeps the exact
    /// products instead of reproducing that undefined behaviour, so these roots are pinned to
    /// fsci's own values. Each was checked against the root of the same function in mpmath
    /// (45 digits, one Newton step from fsci's root). In the derivative range fsci's root is
    /// the nearer one at every point: within 0.5 ulp for p ≤ 0.9, and 24–41 ulp near p = 1,
    /// where xsf's stopping test is absolute, against SciPy's 45–1421. In the 6n range the
    /// two land 1–2 ulp apart either way (SciPy is the nearer at (i32::MAX, 0.99)).
    /// frankenscipy-k4c2c
    #[test]
    fn smirnovi_keeps_exact_products_where_scipy_overflows_an_int() {
        // (n, p, fsci bits, scipy.special.smirnovi(n, p) bits); fsci − SciPy in ulp, then each
        // one's distance from the mpmath root in ulp.
        let cases: [(i32, f64, u64, u64); 10] = [
            // −21; fsci +24.5, SciPy +45.5
            (46_342, 0.99, 0x3f35_58b1_5355_08b2, 0x3f35_58b1_5355_08c7),
            // +381; fsci −0.33, SciPy −381.3
            (50_000, 0.5, 0x3f65_8a56_a052_7549, 0x3f65_8a56_a052_73cc),
            // −3; fsci +0.16, SciPy +3.16
            (100_000, 0.01, 0x3f73_a5dd_502c_0eb1, 0x3f73_a5dd_502c_0eb4),
            // −1; fsci +0.04, SciPy +1.04
            (100_000, 0.3, 0x3f64_15f4_f5a2_5a11, 0x3f64_15f4_f5a2_5a12),
            // −993; fsci +0.13, SciPy +993.1
            (100_000, 0.9, 0x3f47_ba97_4c26_9c89, 0x3f47_ba97_4c26_a06a),
            // −1380; fsci +40.8, SciPy +1420.8
            (100_000, 0.99, 0x3f2d_2a26_a6e2_2ce0, 0x3f2d_2a26_a6e2_3244),
            // +1; fsci +0.39, SciPy −0.61
            (
                357_913_942,
                0.75,
                0x3ef5_053b_ecbc_4a59,
                0x3ef5_053b_ecbc_4a58,
            ),
            // −2; fsci −0.85, SciPy +1.15
            (
                1_073_741_831,
                0.693_484_131_989_360_3,
                0x3eeb_6100_f3e3_e30b,
                0x3eeb_6100_f3e3_e30d,
            ),
            // +1; fsci +0.01, SciPy −0.99
            (i32::MAX, 0.75, 0x3ee1_29d1_d332_8d77, 0x3ee1_29d1_d332_8d76),
            // −1; fsci −0.53, SciPy +0.47
            (i32::MAX, 0.99, 0x3eb9_a9bd_71bb_df87, 0x3eb9_a9bd_71bb_df88),
        ];
        let bad: Vec<String> = cases
            .iter()
            .filter_map(|&(n, p, bits, scipy_bits)| {
                assert_ne!(
                    bits, scipy_bits,
                    "({n}, {p:e}) is not a point where SciPy differs"
                );
                let got = smirnovi(n, p);
                (got.to_bits() != bits).then(|| {
                    let scipy_note = if got.to_bits() == scipy_bits {
                        " (SciPy's overflowed value)"
                    } else {
                        ""
                    };
                    format!(
                        "smirnovi({n}, {p:e}) = {got:e}{scipy_note}, pinned {:e}",
                        f64::from_bits(bits)
                    )
                })
            })
            .collect();
        assert!(
            bad.is_empty(),
            "{} of {} differ from the exact-product roots:\n{}",
            bad.len(),
            cases.len(),
            bad.join("\n")
        );
    }

    /// `smirnov_sf_cdf_pdf` (the kernel fsci-stats' kstwo calls): sf is SciPy's `smirnov`
    /// bit for bit, cdf its complement, and pdf SciPy's `-_smirnovp` where xsf's `int`
    /// products fit (n ≤ 46341). Beyond that SciPy's pdf is corrupted by the overflow (it
    /// returns 0.0 at both points below); fsci's is the true derivative, pinned to values
    /// within 1.5e-17 relative of mpmath's (45 digits). frankenscipy-k4c2c
    #[test]
    fn smirnov_sf_cdf_pdf_is_scipys_sf_with_the_true_pdf() {
        // (n, d, pdf bits); SciPy's -_smirnovp is the same for the first two, 0.0 for the rest.
        let cases: [(i64, f64, u64); 4] = [
            (1000, 0.03, 0x4033_8e2c_68e4_429d),
            (46_341, 0.004, 0x4064_ffdf_0d76_8869),
            (50_000, 0.005, 0x4054_7777_f436_d159),
            (100_000, 0.003, 0x4068_c254_032b_ba80),
        ];
        for (n, d, pdf_bits) in cases {
            let (sf, cdf, pdf) = smirnov_sf_cdf_pdf(n, d);
            let small_n = i32::try_from(n).expect("test n fits i32");
            assert_eq!(sf.to_bits(), smirnov(small_n, d).to_bits(), "sf({n}, {d})");
            // All four take the lower sum, where xsf forms the cdf as 1 − sf.
            assert_eq!(cdf.to_bits(), (1.0 - sf).to_bits(), "cdf({n}, {d})");
            assert_eq!(
                pdf.to_bits(),
                pdf_bits,
                "pdf({n}, {d}) = {pdf:e}, pinned {:e}",
                f64::from_bits(pdf_bits)
            );
        }
        let (sf, cdf, pdf) = smirnov_sf_cdf_pdf(0, 0.5);
        assert!(sf.is_nan() && cdf.is_nan() && pdf.is_nan(), "n = 0");
    }

    /// SciPy's domain: NaN for n < 1, a NaN argument, or an argument outside [0, 1] (the old
    /// implementation returned 1 and 0 for d below 0 and above 1). frankenscipy-k4c2c
    #[test]
    fn smirnov_and_smirnovi_are_nan_outside_the_domain() {
        for (n, d) in [(0, 0.5), (-3, 0.5), (5, -0.1), (5, 1.5), (5, f64::NAN)] {
            assert!(smirnov(n, d).is_nan(), "smirnov({n}, {d})");
        }
        for (n, p) in [(0, 0.5), (-3, 0.5), (5, -0.1), (5, 1.1), (5, f64::NAN)] {
            assert!(smirnovi(n, p).is_nan(), "smirnovi({n}, {p})");
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
    fn ldexp_is_numpys_single_rounding_scaling_at_the_extremes() {
        // numpy.ldexp (C ldexp) values. The first two are where x * 2.0.powi(e) was wrong:
        // 2^e itself overflows or underflows though the product is representable.
        let cases = [
            (0.5, 1024, 8.988_465_674_311_58e307),
            (4_503_599_627_370_496.0, -1100, 3.315_618_4e-316),
            (1.0, -1074, 5e-324),
            (1.0, -1075, 0.0),
            (1.5, -1074, 1e-323),
            (3.0, -1076, 5e-324),
            (1.0, 1023, 8.988_465_674_311_58e307),
            (1.0, 1024, f64::INFINITY),
            (0.75, 1025, f64::INFINITY),
            (5e-324, 2098, f64::INFINITY),
            (1.797_693_134_862_315_7e308, -2098, 5e-324),
            (1.797_693_134_862_315_7e308, -2200, 0.0),
            (-2.5, 10, -2560.0),
            (-0.0, 50, -0.0),
            (f64::INFINITY, -5000, f64::INFINITY),
            (f64::NEG_INFINITY, 5000, f64::NEG_INFINITY),
            (1.0, i32::MAX, f64::INFINITY),
            (1.0, i32::MIN, 0.0),
            (2.225_073_858_507_201_4e-308, 1, 4.450_147_717_014_403e-308),
            (8.095e-320, 60, 9.332_636_185_032_189e-302),
        ];
        for (x, e, want) in cases {
            let got = ldexp(std::hint::black_box(x), std::hint::black_box(e));
            assert_eq!(got.to_bits(), f64::to_bits(want), "ldexp({x:e}, {e})");
        }
        assert!(ldexp(f64::NAN, 3).is_nan());
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

        // Domain edge cases matching SciPy
        assert_eq!(huber_scalar(0.0, 1.0), 0.0);
        assert_eq!(huber_scalar(-1.0, 1.0), f64::INFINITY);
        assert!(huber_scalar(f64::NAN, 1.0).is_nan());
        assert!(huber_scalar(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn pseudo_huber_basic() {
        let delta = 1.0;

        // pseudo_huber(delta, 0) = 0
        assert!((pseudo_huber_scalar(delta, 0.0) - 0.0).abs() < 1e-14);

        // For small x, pseudo_huber ≈ 0.5 * x^2 without catastrophic cancellation
        let small = 1e-10;
        let expected = 0.5 * small * small;
        assert!((pseudo_huber_scalar(delta, small) - expected).abs() < 1e-25);

        // Symmetric
        assert!((pseudo_huber_scalar(delta, 2.0) - pseudo_huber_scalar(delta, -2.0)).abs() < 1e-14);

        // Domain edge cases matching SciPy
        assert_eq!(pseudo_huber_scalar(0.0, 1.0), 0.0);
        assert_eq!(pseudo_huber_scalar(-1.0, 1.0), f64::INFINITY);
        assert!(pseudo_huber_scalar(f64::NAN, 1.0).is_nan());
        assert!(pseudo_huber_scalar(1.0, f64::NAN).is_nan());
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
        assert_eq!(huber_values[1], 0.0);
        assert_eq!(huber_values[2], f64::INFINITY);
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
        assert_eq!(pseudo_values[1], 0.0);
        assert_eq!(pseudo_values[2], f64::INFINITY);
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

        // Small negative x maintains full precision via exp_m1
        assert!((elu(-1e-10, 1.0) - (-1e-10_f64).exp_m1()).abs() < 1e-25);
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
    fn erfcx_large_x_asymptotic_series_matches_scipy() -> Result<(), String> {
        // scipy.special.erfcx 1.17.1, each within 2e-16 of mpmath at 40 digits. The x ≥ 25
        // branch was a three-term series and missed these by 7.7e-9 at x = 25 and 6.8e-9 at
        // 25.5 (frankenscipy-k5qew). This test used to compare erfcx with erfcx_scalar, which
        // could not see that.
        for (x, want) in [
            (24.999, 0.022_550_473_014_042_085),
            (25.0, 0.022_549_572_432_641_357),
            (25.5, 0.022_108_108_052_519_827),
            (26.0, 0.021_683_584_850_562_91),
            (30.0, 0.018_795_888_861_416_754),
            (40.0, 0.014_100_335_983_377_815),
            (50.0, 0.011_281_536_265_323_772),
            (100.0, 0.005_641_613_782_989_433),
            (1e3, 0.000_564_189_301_453_387_6),
            (1e8, 5.641_895_835_477_563e-9),
            (1e150, 5.641_895_835_477_563e-151),
        ] {
            let result = erfcx(&SpecialTensor::RealScalar(x), RuntimeMode::Strict)
                .map_err(|err| err.to_string())?;
            let value = expect_real_scalar(result)?;
            assert!(
                (value - want).abs() <= 1e-15 * want,
                "erfcx({x}) = {value}, SciPy {want}"
            );
        }
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
        assert!(cloglog(-0.1).is_nan());
        assert!(cloglog(1.1).is_nan());

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
        assert!(loglog(-0.1).is_nan());
        assert!(loglog(1.1).is_nan());
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
        assert!(cauchit_scalar(-0.1).is_nan());
        assert!(cauchit_scalar(1.1).is_nan());
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

        // High precision for small |x| near 0 (Martin Maechler algorithm)
        let small_x = -1.0e-15_f64;
        let expected_small = (-small_x.exp_m1()).ln();
        assert!((log1mexp_scalar(small_x) - expected_small).abs() < 1e-14);
        assert!((log1mexp_scalar(small_x) - small_x.abs().ln()).abs() < 1e-12);
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
    fn wofz_is_scipy_faddeeva_bit_for_bit() {
        // frankenscipy-k64p7. `wofz_scalar` is xsf's `Faddeeva::w`, so every branch of it
        // reproduces `scipy.special.wofz` to the bit. The dispatch it replaced (erf series,
        // Weideman N = 32, a 24-term continued fraction, an asymptotic series) lost Re w near
        // the real axis, where Re w ≈ e^{-x²} is tiny against |w|: 100% off at |x| ≈ 4.5,
        // 1e-2 at |x| ≈ 7, 1e-7 at |x| ≈ 3. Values are scipy.special.wofz 1.17.1; each comment
        // is SciPy's own relative error against mpmath at 80 digits.
        let cases: [(f64, f64, f64, f64); 21] = [
            // voigt_profile's worst point (x, sigma, gamma) = (-9.9622, 1.6646, 0.11926) -> z
            (
                -4.23185309219969,
                0.05065875922475325,
                0.0017530159382948228,
                -0.13738578027618212,
            ), // 7.3e-15, 7.3e-15
            // the old continued-fraction band 4 <= |z| < 8, near the axis
            (4.5, 0.001, 3.023933506934306e-05, 0.1287352036338017), // 4.9e-15, 4.9e-15
            (5.2, 1e-08, 2.2326832689018905e-10, 0.11062744390776813), // 6.7e-15, 7.5e-15
            (7.0, 1e-07, 1.1885945814534224e-09, 0.08144750806500302), // 1.0e-15, 8.4e-16
            // lower half plane, same band
            (4.8, -0.001, -2.6287682156816214e-05, 0.12027780130668135), // 6.3e-17, 2.7e-17
            // the old Weideman band 0.5 <= |z| < 4, near the axis
            (3.0, 1e-06, 0.0001234883688197116, 0.20115731629710729), // 3.8e-16, 1.3e-15
            // the old erf-series band |z| < 0.5, near the axis
            (0.3, 0.0001, 0.9138374897892162, 0.3188608529001802), // 2.1e-16, 2.7e-16
            // Re z == 0: erfcx(Im z)
            (0.0, 2.5, 0.2108063640611436, 0.0),  // 2.7e-17
            (0.0, -3.0, 16205.988853999586, 0.0), // 1.7e-17
            // Im z == 0: (exp(-x²), w_im(x))
            (3.0, 0.0, 0.00012340980408667956, 0.20115731703760037), // 9.5e-17, 8.6e-17
            // x < 5e-4: sum4 and sum5 by Taylor series
            (0.0001, 0.3, 0.7345993292845208, 6.876195628274159e-05), // 3.3e-16, 1.5e-16
            // x < 10, y > 5: the imaginary terms cancel
            (1.0, 6.0, 0.09042061181059989, 0.014686964935703239), // 3.2e-16, 5.9e-15
            // x < 10, lower half plane
            (2.0, -1.0, -0.20532558064658757, 0.14685548503016754), // 2.7e-16, 1.0e-15
            // x < 10, y < -6: 2·exp(y² − x²) replaces exp(−x²)·erfcx(y)
            (1.0, -6.5, 1.4910717386437524e+18, 6.903977257164115e+17), // 6.2e-17, 1.4e-16
            // 10 <= x <= 28, |y| <= 1e-10: sums in both directions from n0
            (15.0, 1e-12, 2.5244146785924235e-15, 0.03769678605913684), // 1.5e-16, 1.1e-16
            // general continued fraction, and its lower half plane (glibc cexp)
            (7.0, 0.5, 0.005910424131058674, 0.08101143885794782), // 1.1e-16, 8.1e-17
            (7.0, -0.5, -0.005910424131058674, 0.08101143885794782), // 1.1e-16, 8.1e-17
            // x > 28 takes the continued fraction even on the axis
            (29.0, 1e-12, 6.720557322671423e-16, 0.019466400393582408), // 1.9e-16, 4.6e-17
            // nu == 2 (x + |y| > 4000)
            (3000.0, 2000.0, 8.679840337528545e-05, 0.0001301975950477282), // 1.2e-15, 3.5e-15
            // nu == 1 (x + |y| > 1e7), both orderings of x and |y|
            (
                20000000.0,
                1.0,
                1.4104739588693873e-15,
                2.8209479177387746e-08,
            ), // 3.7e-15, 1.2e-15
            (
                1.0,
                20000000.0,
                2.8209479177387746e-08,
                1.4104739588693873e-15,
            ), // 1.3e-15, 3.8e-15
        ];
        for (re, im, wr, wi) in cases {
            let z = Complex64::new(std::hint::black_box(re), std::hint::black_box(im));
            let w = wofz_scalar(z, RuntimeMode::Strict).unwrap();
            assert_eq!(
                (w.re.to_bits(), w.im.to_bits()),
                (wr.to_bits(), wi.to_bits()),
                "wofz({re}{im:+}i) = {}{:+}i, scipy {wr}{wi:+}i",
                w.re,
                w.im
            );
            if im == 0.0 {
                let (rr, ri) = wofz_real(std::hint::black_box(re));
                assert_eq!((rr.to_bits(), ri.to_bits()), (wr.to_bits(), wi.to_bits()));
            }
        }
    }

    #[test]
    fn wofz_non_finite_follows_scipy_in_strict_and_fails_closed_in_hardened() {
        // scipy.special.wofz 1.17.1 at non-finite arguments. (−∞ − i) is (0, −0): the sign
        // of the zero is part of the contract.
        let inf = f64::INFINITY;
        let cases = [
            ((inf, 0.0), (0.0, 0.0)),
            ((0.0, inf), (0.0, 0.0)),
            ((inf, 1.0), (0.0, 0.0)),
            ((-inf, -1.0), (0.0, -0.0)),
        ];
        for ((re, im), (wr, wi)) in cases {
            let w = wofz_scalar(Complex64::new(re, im), RuntimeMode::Strict).unwrap();
            assert_eq!(
                (w.re.to_bits(), w.im.to_bits()),
                (f64::to_bits(wr), f64::to_bits(wi)),
                "wofz({re}{im:+}i) = {w:?}"
            );
        }
        // (NaN + 0i) → (NaN, NaN); (0 + NaN·i) → (NaN, 0), keeping Re z's zero; (1 − ∞i) → NaN.
        let w = wofz_scalar(Complex64::new(f64::NAN, 0.0), RuntimeMode::Strict).unwrap();
        assert!(w.re.is_nan() && w.im.is_nan());
        let w = wofz_scalar(Complex64::new(0.0, f64::NAN), RuntimeMode::Strict).unwrap();
        assert!(w.re.is_nan() && w.im.to_bits() == 0.0_f64.to_bits());
        let w = wofz_scalar(Complex64::new(1.0, -inf), RuntimeMode::Strict).unwrap();
        assert!(w.re.is_nan() && w.im.is_nan());
        assert!(wofz_scalar(Complex64::new(inf, 0.0), RuntimeMode::Hardened).is_err());
    }

    #[test]
    fn voigt_profile_is_scipy_bit_for_bit_near_the_real_axis() {
        // frankenscipy-k64p7: the first point was 2.3e-5 relative off (SciPy: 7.5e-15 against
        // mpmath). voigt_profile is xsf's, operation for operation, over Faddeeva::w. Values
        // are scipy.special.voigt_profile 1.17.1; comments are SciPy's error against mpmath.
        let cases: [(f64, f64, f64, f64); 7] = [
            (
                -9.96220011339966,
                1.6645991962024114,
                0.1192557222328333,
                0.00042013247248880344,
            ), // 7.5e-15
            (-7.25, 1.0, 0.001, 6.439693401395942e-06), // 2.9e-15
            (6.5, 1.2, 1e-09, 1.4144724764659558e-07),  // 3.2e-15
            (3.1, 0.7, 0.05, 0.0020520634639960597),    // 9.5e-16
            (0.4, 1.0, 1e-05, 0.3682674402029819),      // 8.7e-17
            (2.0, 1.5, 0.0, 0.10934004978399577),       // 1.8e-16, gamma == 0 Gaussian
            (2.0, 0.0, 0.5, 0.03744822190397538),       // 1.2e-16, sigma == 0 Lorentzian
        ];
        for (x, sigma, gamma, expected) in cases {
            let got = voigt_profile(
                std::hint::black_box(x),
                std::hint::black_box(sigma),
                std::hint::black_box(gamma),
            );
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voigt_profile({x}, {sigma}, {gamma}) = {got:e}, scipy {expected:e}"
            );
        }
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
        // For v < -1, SciPy returns sign(Γ(v + 3/2))·∞ (frankenscipy-00cad).
        assert_eq!(modstruve(-2.0, 0.0), f64::NEG_INFINITY);
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
        // For v < -1 the leading term diverges; SciPy returns sign(Γ(v + 3/2))·∞,
        // which is NaN where v + 3/2 is a negative integer (frankenscipy-00cad).
        assert_eq!(struve(-1.5, 0.0), f64::INFINITY);
        assert_eq!(struve(-2.0, 0.0), f64::NEG_INFINITY);
        assert!(struve(-2.5, 0.0).is_nan());
    }

    /// scipy.special.struve / modstruve 1.17.1 (xsf 0d0a593f `cephes/struve.h`) in every branch
    /// of `struve_hl`. The branches that never call a Bessel function are SciPy's bits: the
    /// double-double power series (the first H row is the point the old series/asymptotic
    /// switch had 1.25e-8 wrong), x < 0 and x = 0 on it, the best-of-three fallback when it
    /// picks the power series, overflow and failure. The branches that call J_v, Y_v or I_v
    /// (v = -n - 1/2, the asymptotic expansion, the Bessel series) carry fsci's Bessel values,
    /// which are not SciPy's AMOS/Cephes bits. They are held to mpmath at 60 digits instead:
    /// struve(-3.5, 7.5) is 3e-16 off mpmath here and 7.3e-15 in SciPy. Inputs go through
    /// `black_box` so no constant folding stands in for the runtime path. frankenscipy-00cad
    #[test]
    fn struve_and_modstruve_every_branch_is_scipys_bits_or_mpmath() {
        use std::hint::black_box;
        let h_cases: [(f64, f64, u64); 19] = [
            (0.287, 19.57, 0x3f71_f050_e6ef_1bd4),      // power series
            (0.0, 20.0, 0x3fb8_2a2f_7635_3139),         // power series
            (1.0, 25.0, 0x3fe1_3de1_1792_18d6),         // power series
            (0.287, 5.0, 0x3fa5_d32b_1e82_9132),        // power series
            (-40.0, 20.0, 0xc161_5253_0d72_37a3),       // power series
            (-299.856, 202.743, 0xc540_21d3_b019_7a8d), // power series
            (2.0, -3.0, 0xbfe7_c1a1_b068_0962),         // x < 0, power series
            (0.5, -1.0, 0x7ff8_0000_0000_0000),         // x < 0, non-integer v: NaN
            (-1.0, 0.0, 0x3fe4_5f30_6dc9_c882),         // x = 0
            (-1.5, 0.0, 0x7ff0_0000_0000_0000),         // x = 0
            (-2.0, 0.0, 0xfff0_0000_0000_0000),         // x = 0
            (-2.5, 0.0, 0x7ff8_0000_0000_0000),         // x = 0
            (0.5, 0.0, 0x0000_0000_0000_0000),          // x = 0
            (0.287, 24.0, 0xbfa0_5e99_1b98_ec2c),       // best of three: power series
            (-74.98, 68.57, 0x3fee_c711_e5aa_4c86),     // best of three: power series
            (262.957, 2.874, 0x0000_0000_0000_0000),    // best of three: power series
            (-259.432, 5.815, 0x7ff0_0000_0000_0000),   // overflow
            (f64::NAN, 1.0, 0x7ff8_0000_0000_0000),     // failure
            (1.0, f64::NAN, 0x7ff8_0000_0000_0000),     // failure
        ];
        let l_cases: [(f64, f64, u64); 15] = [
            (0.287, 5.0, 0x403a_bba1_6dcd_79e4),        // power series
            (12.0, 20.0, 0x4132_7a12_12a9_8ede),        // power series
            (-13.32, 10.27, 0x3ff3_1008_f6e3_d950),     // power series
            (206.653, 166.291, 0x4414_7e2f_97e4_6761),  // power series
            (2.0, -3.0, 0xbffb_f667_87b3_df84),         // x < 0, power series
            (0.5, -1.0, 0x7ff8_0000_0000_0000),         // x < 0, non-integer v: NaN
            (-1.0, 0.0, 0x3fe4_5f30_6dc9_c882),         // x = 0
            (-1.5, 0.0, 0x7ff0_0000_0000_0000),         // x = 0
            (-2.0, 0.0, 0xfff0_0000_0000_0000),         // x = 0
            (-2.5, 0.0, 0x7ff8_0000_0000_0000),         // x = 0
            (0.5, 0.0, 0x0000_0000_0000_0000),          // x = 0
            (-247.3, 166.6, 0x4162_1f6c_1899_9e65),     // best of three: power series
            (262.957, 2.874, 0x0000_0000_0000_0000),    // best of three: power series
            (-233.0, 0.97, 0x7ff0_0000_0000_0000),      // overflow
            (-299.856, 202.743, 0x7ff8_0000_0000_0000), // failure
        ];
        for (name, f, cases) in [
            ("struve", struve as fn(f64, f64) -> f64, &h_cases[..]),
            ("modstruve", modstruve, &l_cases[..]),
        ] {
            for &(v, x, bits) in cases {
                let want = f64::from_bits(bits);
                let got = f(black_box(v), black_box(x));
                if want.is_nan() {
                    assert!(got.is_nan(), "{name}({v}, {x}) = {got:e}, SciPy NaN");
                } else {
                    assert_eq!(
                        got.to_bits(),
                        bits,
                        "{name}({v}, {x}) = {got:e}, SciPy {want:e}"
                    );
                }
            }
        }
        // The branches that call J_v, Y_v or I_v, against mpmath's struveh / struvel at 60
        // digits. SciPy's error at each is at most 8.4e-16, except 7.3e-15 at H(-3.5, 7.5) and
        // 6.6e-13 at H(-3.3, 30).
        #[rustfmt::skip]
        let bessel_cases: [(&str, fn(f64, f64) -> f64, f64, f64, f64, f64); 16] = [
            ("struve", struve, 3.0, -30.0, 38.343491008657196, 1e-14),       // x < 0, asymptotic
            ("struve", struve, -3.5, 7.5, 0.13484950550869113, 1e-14),       // v = -n - 1/2
            ("struve", struve, -1.5, 31.0, 0.1329542636128643, 1e-14),       // v = -n - 1/2
            ("struve", struve, 2.5, 21.0, 9.57067437442845, 1e-14),          // asymptotic
            ("struve", struve, 1.0, 60.0, 0.7286660738055736, 1e-14),        // asymptotic
            ("struve", struve, 2.0, 100.0, 21.303864052674466, 1e-14),       // asymptotic
            ("struve", struve, 0.25, 100.0, -0.05453173410693158, 1e-14),    // asymptotic
            ("struve", struve, 189.5, 183.4, 9.035113849118622e19, 1e-14),   // Bessel series
            ("struve", struve, 283.0, 252.9, 1.4034988401917428e21, 1e-14),  // Bessel series
            // Best of three on the asymptotic expansion: 7.0e-13 here and 6.6e-13 in SciPy, the
            // truncation of Cephes' expansion itself.
            ("struve", struve, -3.3, 30.0, -0.002651512976579513, 1e-12),
            ("modstruve", modstruve, 3.0, -30.0, 671140461759.4525, 1e-14),  // x < 0, asymptotic
            ("modstruve", modstruve, -3.5, 0.5, 0.0006810359708579382, 1e-14), // v = -n - 1/2
            ("modstruve", modstruve, -3.5, 5.0, 7.417560126111555, 1e-14),   // v = -n - 1/2
            ("modstruve", modstruve, 0.0, 18.5, 10110921.471720433, 1e-14),  // asymptotic
            ("modstruve", modstruve, -0.75, 20.0, 42934125.4551164, 1e-14),  // asymptotic
            ("modstruve", modstruve, 2.5, 100.0, 1.0405531961408039e42, 1e-14), // asymptotic
        ];
        for (name, f, v, x, want, bound) in bessel_cases {
            let got = f(black_box(v), black_box(x));
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= bound,
                "{name}({v}, {x}) = {got:e}, mpmath {want:e}, rel {rel:e}"
            );
        }
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
        // Past 6 itstruve0 and it2struve0 are their Laplace-derived forms, and itmodstruve0 is
        // its series and from 45 its asymptotic form; each is pinned at 2e-15 in its own
        // `*_exact_integral*` test below (frankenscipy-ch0z1).
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

    /// frankenscipy-ch0z1. Past |x| = 6 `itstruve0` is the Laplace form. Each pin is the exact
    /// integral: the term-by-term series summed in mpmath at 40 + 0.9x digits, which is more than
    /// its cancellation costs, then rounded once. The old routes miss these at 2e-15: the double
    /// series was 1e-12 off by x = 14, and the Simpson quadrature past 16 was 7e-11 off. SciPy's
    /// own ITSH0 is 1e-12 off at 16, 9e-7 at 29, and O(1) wrong from 40, so SciPy is not the
    /// reference here.
    #[test]
    fn itstruve0_past_six_is_the_exact_integral() {
        #[rustfmt::skip]
        const PINS: [(f64, f64); 15] = [
            (6.25, 1.7909473149001491),
            (7.5, 1.8267382620256432),
            (-7.5, 1.8267382620256432),
            (9.0, 2.287536730163513),
            (11.3, 2.4751252870662794),
            (13.7, 2.271809228671129),
            (15.9, 2.731994117122735),
            (16.0, 2.74642640255143),
            (17.2, 2.7712380313606744),
            (23.7, 2.926592804551052),
            (29.0, 3.100335851303605),
            (40.0, 3.148417627447186),
            (55.5, 3.3947758689205614),
            (100.0, 3.720914252854685),
            (250.0, 4.349953826618994),
        ];
        for (x, want) in PINS {
            let got = itstruve0(std::hint::black_box(x));
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 2e-15,
                "itstruve0({x}) = {got:e}, exact {want:e}, rel {rel:e}"
            );
        }
        // The two routes meet at 6 without a step.
        let below = itstruve0(ITSTRUVE0_SERIES_MAX);
        let above = itstruve0(ITSTRUVE0_SERIES_MAX.next_up());
        assert!(
            ((above - below) / below).abs() <= 4e-15,
            "{below:e} | {above:e}"
        );
    }

    /// frankenscipy-ch0z1. `itmodstruve0` is its all-positive series below |x| = 45 and the
    /// asymptotic `∫I₀ − (2/π)(ln 2x + γ)` from there, including past 709.78, where `e^x` alone
    /// overflows. Each pin is the series summed in mpmath at 50 digits. The Simpson quadrature
    /// this replaced past 16 was 3e-10 off, and SciPy's asymptotic past 20 is 7e-9 off at 25.
    #[test]
    fn itmodstruve0_is_the_exact_integral_up_to_overflow() {
        #[rustfmt::skip]
        const PINS: [(f64, f64); 14] = [
            (16.5, 1499803.0549157115),
            (18.25, 8170935.521923119),
            (20.0, 44758596.19878038),
            (25.0, 5899173181.422938),
            (-25.0, 5899173181.422938),
            (30.0, 795538858181.7499),
            (45.0, 2.1075225881890097e18),
            (-45.5, 3.45501222001195e18),
            (50.0, 2.9629659299472146e20),
            (100.0, 1.079217066847346e42),
            (400.0, 1.0431665100985694e172),
            (700.0, 1.530688656412344e302),
            (709.9, 3.0293345555359286e306),
            (712.0, 2.470148769369148e307),
        ];
        for (x, want) in PINS {
            let got = itmodstruve0(std::hint::black_box(x));
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 2e-15,
                "itmodstruve0({x}) = {got:e}, exact {want:e}, rel {rel:e}"
            );
        }
        // Past the overflow it is inf, as SciPy's is.
        assert_eq!(itmodstruve0(std::hint::black_box(715.0)), f64::INFINITY);
    }

    /// frankenscipy-ch0z1. `it2struve0` is `π/2 ∓` its series below |x| = 1.5, a Chebyshev fit
    /// of `x·it2struve0(x)` on [1.5, 6) and the amplitude-phase tail from 6; negative x is
    /// `π − it2struve0(|x|)`. Each pin is the exact integral: `π/2 −` the term-by-term series
    /// summed in mpmath at 50 + 0.9x digits, which is more than its cancellation costs (π minus
    /// that for negative x), rounded once. The old routes miss 20 of these at 2e-15: `π/2 −` the
    /// f64 series was 4.9e-15 off at 3.9 and 6.5e-11 at 16, and the Simpson quadrature past 16
    /// was 8e-11 off at 17.2 and 5e-9 at 300. SciPy's own ITTH0 is 2e-11 off at 16 and 6e-8 at
    /// 23.7, so SciPy is not the reference here.
    #[test]
    fn it2struve0_is_the_exact_integral() {
        #[rustfmt::skip]
        const PINS: [(f64, f64); 24] = [
            (1.75, 0.5741925901547921),
            (-2.5, 2.844448633790081),
            (3.9, 0.07431575919885444),
            (4.8, 0.07278263269433576),
            (5.5, 0.09985040949792102),
            (5.999999999999999, 0.11833664178101615),
            (6.0, 0.11833664178101617),
            (6.25, 0.12496225938490622),
            (7.5, 0.12070378801861187),
            (-7.5, 3.020888865571181),
            (9.0, 0.06504320442147894),
            (11.3, 0.044244133188391564),
            (13.7, 0.061486120344981444),
            (15.9, 0.0304106017357131),
            (16.0, 0.029505695657558808),
            (17.2, 0.02791105859214424),
            (23.7, 0.022332752575750597),
            (29.0, 0.016885308982410245),
            (40.0, 0.016213313631284396),
            (-40.0, 3.125379339958509),
            (55.5, 0.01091346468259087),
            (100.0, 0.0065541908590652665),
            (250.0, 0.00244122142442305),
            (300.0, 0.0020105371670472018),
        ];
        let misses: Vec<String> = PINS
            .iter()
            .filter_map(|&(x, want)| {
                let got = it2struve0(std::hint::black_box(x));
                let rel = ((got - want) / want).abs();
                (rel.is_nan() || rel > 2e-15)
                    .then(|| format!("it2struve0({x}) = {got:e}, exact {want:e}, rel {rel:e}"))
            })
            .collect();
        assert!(
            misses.is_empty(),
            "{} of {} pins miss 2e-15:\n{}",
            misses.len(),
            PINS.len(),
            misses.join("\n")
        );
        // The three routes meet without a step.
        for edge in [IT2STRUVE0_SERIES_MAX, IT2STRUVE0_TAIL_MIN] {
            let below = it2struve0(edge.next_down());
            let above = it2struve0(edge);
            assert!(
                ((above - below) / below).abs() <= 4e-15,
                "at {edge}: {below:e} | {above:e}"
            );
        }
        // Far out it is 2/(πx) (its expansion is 2/(πx)·(1 − 1/(3x²) + ...)), and π for −x, as
        // SciPy's is.
        let huge = it2struve0(std::hint::black_box(1e300));
        assert!(
            ((huge - 6.366_197_723_675_814e-301) / huge).abs() <= 2e-15,
            "{huge:e}"
        );
        assert_eq!(it2struve0(std::hint::black_box(-1e300)), PI);
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
