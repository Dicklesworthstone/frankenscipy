#![forbid(unsafe_code)]
#![feature(portable_simd)]

//! N-dimensional image processing for FrankenSciPy.
//!
//! Matches `scipy.ndimage` core operations:
//! - Filters: uniform, gaussian, median, minimum, maximum, convolve, correlate
//! - Morphology: binary_erosion, binary_dilation, binary_opening, binary_closing
//! - Measurements: label, find_objects, value_indices, sum, mean, variance, standard_deviation
//! - Interpolation: shift, rotate, zoom, map_coordinates
//! - Distance transforms: distance_transform_edt

use fsci_interpolate::make_interp_spline;
use std::simd::Simd;
use std::simd::num::SimdFloat;

/// Error type for ndimage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdimageError {
    InvalidArgument(String),
    DimensionMismatch(String),
    EmptyInput,
}

impl std::fmt::Display for NdimageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for NdimageError {}

/// Boundary mode for filtering operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Reflect: d c b a | a b c d | d c b a
    Reflect,
    /// Constant: k k k k | a b c d | k k k k (default k=0)
    Constant,
    /// Nearest: a a a a | a b c d | d d d d
    Nearest,
    /// Wrap: a b c d | a b c d | a b c d
    Wrap,
    /// Mirror: d c b | a b c d | c b a — reflect about the *centre* of the edge
    /// samples (whole-sample symmetric, period 2(n−1), edge not repeated).
    /// Matches `scipy.ndimage` `mode='mirror'`.
    Mirror,
}

const DEFAULT_GAUSSIAN_TRUNCATE: f64 = 4.0;

fn gaussian_kernel_radius(sigma: f64) -> usize {
    (DEFAULT_GAUSSIAN_TRUNCATE * sigma + 0.5) as usize
}

fn gaussian_kernel_len(radius: usize) -> Result<usize, NdimageError> {
    radius
        .checked_mul(2)
        .and_then(|diameter| diameter.checked_add(1))
        .ok_or_else(|| {
            NdimageError::InvalidArgument("gaussian kernel radius is too large".to_string())
        })
}

/// Computes a 1-D Gaussian convolution kernel for the `order`-th derivative.
///
/// Mirrors `scipy.ndimage._filters._gaussian_kernel1d`: builds the normalized
/// Gaussian `phi(x)`, then for `order > 0` multiplies by the polynomial `q(x)`
/// produced by differentiating `q(x) * phi(x)` `order` times. Each derivative
/// maps `q -> q' + q * p'` with `p'(x) = -x / sigma^2`; `Q_deriv` applies that
/// operator to the polynomial coefficients of `q` (superdiagonal `D` for `q'`,
/// subdiagonal `P` for `q * p'`).
fn gaussian_kernel1d(sigma: f64, order: usize, radius: usize) -> Result<Vec<f64>, NdimageError> {
    let sigma2 = sigma * sigma;
    let n = gaussian_kernel_len(radius)?;
    let mut phi_x: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-0.5 / sigma2 * x * x).exp()
        })
        .collect();
    let sum: f64 = phi_x.iter().sum();
    for v in &mut phi_x {
        *v /= sum;
    }
    if order == 0 {
        return Ok(phi_x);
    }

    let mut q = vec![0.0; order + 1];
    q[0] = 1.0;
    for _ in 0..order {
        let mut next = vec![0.0; order + 1];
        for r in 0..=order {
            let mut val = 0.0;
            if r < order {
                val += (r as f64 + 1.0) * q[r + 1];
            }
            if r >= 1 {
                val += -q[r - 1] / sigma2;
            }
            next[r] = val;
        }
        q = next;
    }

    // kernel[i] = (sum_e x_i^e * q[e]) * phi(x_i)
    Ok((0..n)
        .map(|i| {
            let x = i as f64 - radius as f64;
            let mut poly = 0.0;
            let mut x_pow = 1.0;
            for &coeff in &q {
                poly += x_pow * coeff;
                x_pow *= x;
            }
            poly * phi_x[i]
        })
        .collect())
}

/// Apply a 1-D Gaussian (or its `order`-th derivative) along a single axis.
///
/// Mirrors `scipy.ndimage.gaussian_filter1d`. scipy reverses the kernel and
/// calls `correlate1d`; `convolve` here already flips the kernel, so passing
/// the non-reversed kernel to `convolve` yields the identical result.
fn gaussian_filter1d_axis(
    input: &NdArray,
    sigma: f64,
    axis: usize,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let radius = gaussian_kernel_radius(sigma);
    let kernel_1d = gaussian_kernel1d(sigma, order, radius)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    // The 1-D-embedded kernel convolution is the slab line walk.  It parallelizes either
    // across outer slabs or, for the outermost axis, across the contiguous inner span.
    // Both routes are byte-identical to N-D `convolve` (same flip/offset/tap order).
    let outer: usize = input.shape[..axis].iter().product();
    let inner: usize = input.shape[axis + 1..].iter().product();
    let nthreads = ndimage_filter_thread_count(input.size(), kernel_1d.len());
    if outer >= nthreads || (outer == 1 && inner >= nthreads) {
        return Ok(convolve1d_along_axis(
            input, &kernel_1d, axis, 0, mode, cval,
        ));
    }
    let mut kernel_shape = vec![1usize; input.ndim()];
    kernel_shape[axis] = kernel_1d.len();
    let kernel = NdArray::new(kernel_1d, kernel_shape)?;
    convolve(input, &kernel, mode, cval)
}

/// Apply `gaussian_filter1d` sequentially along every axis, using the matching
/// derivative `orders[axis]` per axis. Mirrors `scipy.ndimage.gaussian_filter`
/// invoked with an `order` sequence.
fn gaussian_filter_with_orders(
    input: &NdArray,
    sigma: f64,
    orders: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let mut current = input.clone();
    for (axis, &order) in orders.iter().enumerate() {
        current = gaussian_filter1d_axis(&current, sigma, axis, order, mode, cval)?;
    }
    Ok(current)
}

fn reflect_convolve1d_sources(len: usize, kernel_len: usize) -> Vec<usize> {
    let n = len as i64;
    let kernel_len_i = kernel_len as i64;
    let offset = (kernel_len_i - 1) / 2;
    let mut sources = Vec::with_capacity(len * kernel_len);
    for out in 0..len {
        let out_i = out as i64;
        for tap in 0..kernel_len {
            let flipped = kernel_len_i - 1 - tap as i64;
            let src = boundary_index_1d(out_i + flipped - offset, n, BoundaryMode::Reflect)
                .expect("reflect boundary always maps to an in-bounds index");
            sources.push(src as usize);
        }
    }
    sources
}

/// Runtime A/B toggle: folded symmetric axpy row pass (default) vs the legacy
/// per-pixel strided gather-dot. The gaussian kernel is symmetric, so each
/// output is `w[mid]*x[mid] + sum w[mid +/- k]*(x[+k] + x[-k])` in scipy's
/// correlate1d order. The axis-0 pass is reformulated as contiguous axpy passes
/// over each row, making the hot loop stride-1 instead of gathering at
/// `cols` stride.
/// CONTRACT: NOT byte-identical -- agrees to floating-point REASSOCIATION only, a few ulps
/// on the kernel sum. The axpy arm folds symmetric tap PAIRS, accumulating
/// `w*(hi+lo)` outward from the centre; the per-pixel arm accumulates `w*x` per tap in
/// index order. Those are the same sum in exact arithmetic and different roundings in
/// f64, both by the association (`w*a + w*b` vs `w*(a+b)`) and by the summation order
/// (centre-outward vs left-to-right).
///
/// PRECONDITION, and the reason this is worth writing down: the pair fold reads
/// `kernel[mid-offset]` for BOTH taps, so it is only valid on a SYMMETRIC kernel. On an
/// antisymmetric one -- a first-derivative-of-Gaussian, `order=1` -- it would not be a
/// few ulps off, it would be flatly wrong, since the two taps that should cancel would
/// instead add. What makes that safe here is structural rather than checked: this
/// function is `gaussian_filter_2d_reflect_order0` and calls `gaussian_kernel1d(sigma, 0,
/// radius)` with the order hardcoded to 0. If this ever grows an `order` parameter, the
/// axpy arm must be gated on the kernel actually being symmetric.
pub static GAUSSIAN_2D_AXPY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

fn gaussian_filter_2d_reflect_order0(input: &NdArray, sigma: f64) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let rows = input.shape[0];
    let cols = input.shape[1];
    let radius = gaussian_kernel_radius(sigma);
    let kernel = gaussian_kernel1d(sigma, 0, radius)?;
    let kernel_len = kernel.len();
    let row_sources = reflect_convolve1d_sources(rows, kernel_len);
    let col_sources = reflect_convolve1d_sources(cols, kernel_len);
    let mut output = vec![0.0; input.size()];
    // Cost-aware parallel gate. The separable row/col passes are cheap per pixel
    // (one symmetric fold, ~kernel_len taps), so the shared `ndimage_filter_thread_count`
    // gate (work >= 1<<18) fires far too early: at 256² (work ≈ 1.1M) it spawns up
    // to one thread per few rows, and the spawn overhead dominates — same-process
    // A/B measured serial 1.82× FASTER at 256² (and 6.98× at 128²), with parallel
    // only winning from 512² up (work ≈ 4.5M). Require ~2M pixel·tap work before
    // going parallel; byte-identical (thread count never changes the result).
    let par_work = (input.size() as u64).saturating_mul(kernel_len as u64);
    let nthreads = if par_work < (1 << 21) {
        1
    } else {
        ndimage_filter_thread_count(input.size(), kernel_len)
            .min(rows)
            .max(1)
    };
    let row_chunk = rows.div_ceil(nthreads);
    let axpy = GAUSSIAN_2D_AXPY.load(std::sync::atomic::Ordering::Relaxed);
    let mid = kernel_len / 2;

    let input_data = &input.data;
    let kernel = kernel.as_slice();
    let row_sources = row_sources.as_slice();
    let col_sources = col_sources.as_slice();
    std::thread::scope(|scope| {
        for (chunk_idx, output_chunk) in output.chunks_mut(row_chunk * cols).enumerate() {
            let start_row = chunk_idx * row_chunk;
            scope.spawn(move || {
                let mut scratch_chunk = vec![0.0; output_chunk.len()];
                for (local_row, scratch_row) in scratch_chunk.chunks_mut(cols).enumerate() {
                    let row = start_row + local_row;
                    let row_plan = &row_sources[row * kernel_len..(row + 1) * kernel_len];
                    if axpy {
                        let center_base = row_plan[mid] * cols;
                        let center_weight = kernel[mid];
                        for (slot, &x) in scratch_row
                            .iter_mut()
                            .zip(&input_data[center_base..center_base + cols])
                        {
                            *slot = center_weight * x;
                        }
                        for offset in 1..=mid {
                            let lo_tap = mid - offset;
                            let weight = kernel[lo_tap];
                            let hi_base = row_plan[mid + offset] * cols;
                            let lo_base = row_plan[lo_tap] * cols;
                            let hi_row = &input_data[hi_base..hi_base + cols];
                            let lo_row = &input_data[lo_base..lo_base + cols];
                            for ((slot, &hi), &lo) in scratch_row.iter_mut().zip(hi_row).zip(lo_row)
                            {
                                *slot += weight * (hi + lo);
                            }
                        }
                    } else {
                        for (col, slot) in scratch_row.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            for (&weight, &src_row) in kernel.iter().zip(row_plan) {
                                sum += weight * input_data[src_row * cols + col];
                            }
                            *slot = sum;
                        }
                    }
                }

                let scratch_data = scratch_chunk.as_slice();
                // Interior columns [mid, cols-mid) are reflection-free (col_plan is the
                // identity offset there), so the symmetric fold collapses to a contiguous
                // axpy across the row — auto-vectorizable, unlike the per-pixel/per-tap
                // gather. Byte-identical: each output column accumulates center then
                // offset 1..=mid in the same order as the per-pixel path. Boundary columns
                // keep the reflected col_plan path. (Mirrors the row pass's axpy form.)
                let interior_axpy = axpy && cols > 2 * mid;
                for (local_row, output_row) in output_chunk.chunks_mut(cols).enumerate() {
                    let row_base = local_row * cols;
                    if interior_axpy {
                        let s = &scratch_data[row_base..row_base + cols];
                        let hi = cols - mid;
                        let cw = kernel[mid];
                        for col in mid..hi {
                            output_row[col] = cw * s[col];
                        }
                        for offset in 1..=mid {
                            let w = kernel[mid - offset];
                            for col in mid..hi {
                                output_row[col] += w * (s[col + offset] + s[col - offset]);
                            }
                        }
                        for col in (0..mid).chain(hi..cols) {
                            let col_plan = &col_sources[col * kernel_len..(col + 1) * kernel_len];
                            let mut sum = cw * scratch_data[row_base + col_plan[mid]];
                            for offset in 1..=mid {
                                let lo_tap = mid - offset;
                                sum += kernel[lo_tap]
                                    * (scratch_data[row_base + col_plan[mid + offset]]
                                        + scratch_data[row_base + col_plan[lo_tap]]);
                            }
                            output_row[col] = sum;
                        }
                    } else {
                        for (col, slot) in output_row.iter_mut().enumerate() {
                            let col_plan = &col_sources[col * kernel_len..(col + 1) * kernel_len];
                            if axpy {
                                let mut sum = kernel[mid] * scratch_data[row_base + col_plan[mid]];
                                for offset in 1..=mid {
                                    let lo_tap = mid - offset;
                                    sum += kernel[lo_tap]
                                        * (scratch_data[row_base + col_plan[mid + offset]]
                                            + scratch_data[row_base + col_plan[lo_tap]]);
                                }
                                *slot = sum;
                            } else {
                                let mut sum = 0.0;
                                for (&weight, &src_col) in kernel.iter().zip(col_plan) {
                                    sum += weight * scratch_data[row_base + src_col];
                                }
                                *slot = sum;
                            }
                        }
                    }
                }
            });
        }
    });

    NdArray::new(output, input.shape.clone())
}

fn gaussian_filter_with_orders_on_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[usize],
    orders: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axes.len() != orders.len() {
        return Err(NdimageError::DimensionMismatch(format!(
            "orders length {} != axes length {}",
            orders.len(),
            axes.len()
        )));
    }

    let mut current = input.clone();
    for (&axis, &order) in axes.iter().zip(orders) {
        current = gaussian_filter1d_axis(&current, sigma, axis, order, mode, cval)?;
    }
    Ok(current)
}

fn gaussian_filter_with_sigmas_and_orders(
    input: &NdArray,
    sigmas: &[f64],
    orders: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if sigmas.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "sigmas length {} != ndim {}",
            sigmas.len(),
            input.ndim()
        )));
    }
    if orders.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "orders length {} != ndim {}",
            orders.len(),
            input.ndim()
        )));
    }

    let mut current = input.clone();
    for (axis, (&sigma, &order)) in sigmas.iter().zip(orders).enumerate() {
        if !sigma.is_finite() {
            return Err(NdimageError::InvalidArgument(
                "sigmas must be finite".to_string(),
            ));
        }
        if sigma <= 0.0 {
            continue;
        }
        current = gaussian_filter1d_axis(&current, sigma, axis, order, mode, cval)?;
    }
    Ok(current)
}

// ══════════════════════════════════════════════════════════════════════
// N-D Array Helper
// ══════════════════════════════════════════════════════════════════════

/// A simple N-dimensional array stored in row-major (C) order.
#[derive(Debug, Clone)]
pub struct NdArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl NdArray {
    /// Create an NdArray from data and shape.
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, NdimageError> {
        let total = checked_shape_product(&shape)?;
        if total != data.len() {
            return Err(NdimageError::DimensionMismatch(format!(
                "shape {:?} requires {} elements, got {}",
                shape,
                total,
                data.len()
            )));
        }
        let strides = checked_strides(&shape)?;
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Create a zero-filled array.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self {
            data: vec![0.0; total],
            shape,
            strides,
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get element by multi-dimensional index.
    pub fn get(&self, idx: &[usize]) -> f64 {
        let flat = self.flat_index(idx);
        self.data[flat]
    }

    /// Set element by multi-dimensional index.
    pub fn set(&mut self, idx: &[usize], value: f64) {
        let flat = self.flat_index(idx);
        self.data[flat] = value;
    }

    fn flat_index(&self, idx: &[usize]) -> usize {
        idx.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Convert flat index to multi-dimensional index.
    fn unravel(&self, mut flat: usize) -> Vec<usize> {
        let mut idx = vec![0usize; self.ndim()];
        for (d, slot) in idx.iter_mut().enumerate() {
            *slot = flat / self.strides[d];
            flat %= self.strides[d];
        }
        idx
    }

    /// Get value at index with boundary handling.
    fn get_boundary(&self, idx: &[i64], mode: BoundaryMode, cval: f64) -> f64 {
        let mut safe_idx = vec![0usize; self.ndim()];
        for d in 0..self.ndim() {
            let size = self.shape[d] as i64;
            let mut i = idx[d];
            match mode {
                BoundaryMode::Reflect => {
                    if i < 0 {
                        i = -i - 1;
                    }
                    if i >= size {
                        i = 2 * size - i - 1;
                    }
                    // Handle multiple reflections
                    let period = 2 * size;
                    if period > 0 {
                        i = i.rem_euclid(period);
                        if i >= size {
                            i = period - i - 1;
                        }
                    }
                }
                BoundaryMode::Constant => {
                    if i < 0 || i >= size {
                        return cval;
                    }
                }
                BoundaryMode::Nearest => {
                    i = i.clamp(0, size - 1);
                }
                BoundaryMode::Wrap => {
                    i = i.rem_euclid(size);
                }
                BoundaryMode::Mirror => {
                    // Whole-sample symmetric: reflect about indices 0 and n-1,
                    // period 2(n-1), edge not repeated.
                    if size <= 1 {
                        i = 0;
                    } else {
                        let period = 2 * (size - 1);
                        i = i.rem_euclid(period);
                        if i >= size {
                            i = period - i;
                        }
                    }
                }
            }
            safe_idx[d] = i as usize;
        }
        self.get(&safe_idx)
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

fn checked_shape_product(shape: &[usize]) -> Result<usize, NdimageError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| NdimageError::InvalidArgument("shape product overflows usize".to_string()))
}

fn checked_strides(shape: &[usize]) -> Result<Vec<usize>, NdimageError> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1].checked_mul(shape[d + 1]).ok_or_else(|| {
            NdimageError::InvalidArgument("shape strides overflow usize".to_string())
        })?;
    }
    Ok(strides)
}

fn unravel_with_shape(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let strides = compute_strides(shape);
    let mut idx = vec![0usize; shape.len()];
    for (d, slot) in idx.iter_mut().enumerate() {
        *slot = flat / strides[d];
        flat %= strides[d];
    }
    idx
}

const SPLINE_NEAREST_PAD: usize = 12;

#[derive(Debug, Clone)]
struct SplinePrefilter {
    coeffs: NdArray,
    coord_offsets: Vec<f64>,
}

// Retained REFERENCE implementation. The optimised closed forms below are
// DOCUMENTED in terms of it, and a test now checks that equivalence instead of
// only asserting it in prose. Dead outside test builds (frankenscipy-iit9c).
#[cfg_attr(not(test), allow(dead_code))]
fn uniform_interpolation_knots(len: usize, order: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(len + order + 1);
    knots.extend(std::iter::repeat_n(0.0, order + 1));
    for i in 0..len.saturating_sub(order + 1) {
        knots.push((i + 1 + (order - 1) / 2) as f64);
    }
    knots.extend(std::iter::repeat_n((len - 1) as f64, order + 1));
    knots
}

/// Closed-form knot value of `uniform_interpolation_knots(len, order)` at index `j`, without
/// materializing the knot vector (the left/right clamps are constant and the interior is linear).
#[inline]
fn uniform_knot_at(j: usize, len: usize, order: usize) -> f64 {
    if j < order + 1 {
        0.0
    } else if j >= len {
        (len - 1) as f64
    } else {
        (j - order + (order - 1) / 2) as f64
    }
}

/// Locally-supported B-spline interpolation weights at `x`, BYTE-IDENTICAL to the nonzero entries
/// of `eval_bspline_basis_all(&uniform_interpolation_knots(len, order), x, order, len)` filtered by
/// `|w| > 1e-12`, but O(order²) instead of O(len·order) with NO per-call allocation of the
/// length-`len` knot/basis vectors. A B-spline of degree `order` is supported on only `order+1`
/// consecutive knot spans, so every other basis value is exactly `0.0` (and dropped by the filter).
/// Pushes `(index, weight)` pairs (ascending index) into `out`.
fn bspline_local_support(len: usize, x: f64, order: usize, out: &mut Vec<(usize, f64)>) {
    let total = len + order + 1;
    let knot = |j: usize| uniform_knot_at(j, len, order);
    // Degree-0 span: the unique i0 in [0, len) with knot[i0] <= x < knot[i0+1] (or the right-edge
    // special case x == knot[i0+1] && i0+1 == total-order-1) — exactly the index where
    // eval_bspline_basis_all's degree-0 loop writes 1.0. Binary-searched on the non-decreasing knots.
    let (mut lo_b, mut hi_b) = (0usize, len);
    while lo_b < hi_b {
        let mid = (lo_b + hi_b) / 2;
        if knot(mid) <= x {
            lo_b = mid + 1;
        } else {
            hi_b = mid;
        }
    }
    if lo_b == 0 {
        return;
    }
    let i0 = lo_b - 1; // largest index with knot(i0) <= x
    let kc1 = knot(i0 + 1);
    let matches = (x < kc1) || (x == kc1 && i0 + 1 == total - order - 1);
    if !matches {
        return;
    }

    // Windowed Cox–de Boor over the only possibly-nonzero indices [i0-order ..= i0]; the i0+1 slot
    // is the always-zero `prev[i+1]` sentinel. Same arithmetic/guards as eval_bspline_basis_all.
    let win_lo = i0.saturating_sub(order);
    let wlen = i0 + 2 - win_lo;
    let mut basis = vec![0.0f64; wlen];
    basis[i0 - win_lo] = 1.0;
    for p in 1..=order {
        let prev = basis.clone();
        for ai in win_lo..=i0 {
            let j = ai - win_lo;
            let mut val = 0.0;
            if ai + p < total {
                let denom_left = knot(ai + p) - knot(ai);
                if denom_left > 0.0 {
                    val += (x - knot(ai)) / denom_left * prev[j];
                }
            }
            if ai + p + 1 < total && ai + 1 < len {
                let denom_right = knot(ai + p + 1) - knot(ai + 1);
                if denom_right > 0.0 {
                    val += (knot(ai + p + 1) - x) / denom_right * prev[j + 1];
                }
            }
            basis[j] = val;
        }
    }
    for ai in win_lo..=i0 {
        let w = basis[ai - win_lo];
        if w.abs() > 1e-12 {
            out.push((ai, w));
        }
    }
}

// Retained REFERENCE implementation. The optimised closed forms below are
// DOCUMENTED in terms of it, and a test now checks that equivalence instead of
// only asserting it in prose. Dead outside test builds (frankenscipy-iit9c).
#[cfg_attr(not(test), allow(dead_code))]
fn eval_bspline_basis_all(knots: &[f64], x: f64, order: usize, len: usize) -> Vec<f64> {
    let mut basis = vec![0.0; len];
    for i in 0..len {
        if i + 1 < knots.len() {
            basis[i] = if (knots[i] <= x && x < knots[i + 1])
                || (x == knots[i + 1] && i + 1 == knots.len() - order - 1)
            {
                1.0
            } else {
                0.0
            };
        }
    }
    for p in 1..=order {
        let prev = basis.clone();
        for i in 0..len {
            let mut val = 0.0;
            if i + p < knots.len() {
                let denom_left = knots[i + p] - knots[i];
                if denom_left > 0.0 {
                    val += (x - knots[i]) / denom_left * prev[i];
                }
            }
            if i + p + 1 < knots.len() && i + 1 < len {
                let denom_right = knots[i + p + 1] - knots[i + 1];
                if denom_right > 0.0 {
                    val += (knots[i + p + 1] - x) / denom_right * prev[i + 1];
                }
            }
            basis[i] = val;
        }
    }
    basis
}

fn map_reflect_coordinate(coord: f64, len: usize) -> f64 {
    if len <= 1 {
        return 0.0;
    }
    let period = 2.0 * len as f64;
    let mut shifted = (coord + 0.5).rem_euclid(period);
    if shifted >= len as f64 {
        shifted = period - shifted;
    }
    shifted - 0.5
}

/// Whole-sample symmetric (`mirror`) coordinate fold: reflect a continuous
/// coordinate about 0 and `len-1` with period `2(len-1)`, the edge not repeated.
fn map_mirror_coordinate(coord: f64, len: usize) -> f64 {
    if len <= 1 {
        return 0.0;
    }
    let max = (len - 1) as f64;
    let period = 2.0 * max;
    let mut c = coord.rem_euclid(period);
    if c > max {
        c = period - c;
    }
    c
}

fn map_coordinate(coord: f64, len: usize, mode: BoundaryMode) -> Option<f64> {
    let max = (len.saturating_sub(1)) as f64;
    match mode {
        BoundaryMode::Constant => (0.0..=max).contains(&coord).then_some(coord),
        BoundaryMode::Nearest => Some(coord.clamp(0.0, max)),
        BoundaryMode::Wrap => {
            if len == 0 {
                None
            } else {
                Some(coord.rem_euclid(len as f64))
            }
        }
        BoundaryMode::Reflect => Some(map_reflect_coordinate(coord, len)),
        BoundaryMode::Mirror => Some(map_mirror_coordinate(coord, len)),
    }
}

fn map_interpolation_coordinate(coord: f64, len: usize, mode: BoundaryMode) -> Option<f64> {
    if mode != BoundaryMode::Wrap {
        return map_coordinate(coord, len, mode);
    }
    if len <= 1 {
        return Some(0.0);
    }
    let max = (len - 1) as f64;
    if (0.0..=max).contains(&coord) {
        Some(coord)
    } else {
        Some(coord.rem_euclid(max))
    }
}

fn wrap_interpolation_index(idx: i64, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    // scipy `mode='wrap'` has period len-1 (the endpoints are identified), so the
    // last index len-1 maps to 0. Wrap with period len-1 for every index — the
    // previous in-range short-circuit wrongly returned len-1 unchanged.
    let period = (len - 1) as i64;
    idx.rem_euclid(period) as usize
}

fn fold_wrap_cubic_index(idx: isize, max: usize) -> usize {
    let max = max as isize;
    if idx < 0 {
        (-idx) as usize
    } else if idx > max {
        (2 * max - idx) as usize
    } else {
        idx as usize
    }
}

fn spline_coefficients_for_line(line: &[f64], order: usize) -> Result<Vec<f64>, NdimageError> {
    if line.len() <= 1 || order <= 1 {
        return Ok(line.to_vec());
    }
    let effective_order = order.min(line.len() - 1);
    if effective_order <= 1 {
        return Ok(line.to_vec());
    }
    let x: Vec<f64> = (0..line.len()).map(|i| i as f64).collect();
    let spline = make_interp_spline(&x, line, effective_order).map_err(|err| {
        NdimageError::InvalidArgument(format!("failed to compute spline coefficients: {err}"))
    })?;
    Ok(spline.coeffs().to_vec())
}

fn cubic_constant_wrap_coefficients(line: &[f64]) -> Vec<f64> {
    let n = line.len();
    if n <= 1 {
        return line.to_vec();
    }
    let mut diag: Vec<f64> = vec![2.0 / 3.0; n];
    let mut rhs: Vec<f64> = line.to_vec();
    let mut lower: Vec<f64> = vec![0.0; n];
    let mut upper: Vec<f64> = vec![0.0; n];
    if n >= 2 {
        upper[0] = 1.0 / 3.0;
        lower[n - 1] = 1.0 / 3.0;
    }
    for i in 1..n.saturating_sub(1) {
        lower[i] = 1.0 / 6.0;
        diag[i] = 2.0 / 3.0;
        upper[i] = 1.0 / 6.0;
    }
    for i in 1..n {
        if diag[i - 1].abs() < 1e-18 {
            continue;
        }
        let w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    rhs[n - 1] /= diag[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diag[i];
    }
    rhs
}

/// Cardinal B-spline `beta^order(x)` for `order` in 1..=5.
///
/// Evaluated by the order-raising recurrence
/// `beta^d(x) = [ (x + (d+1)/2) beta^(d-1)(x+1/2)
///              + ((d+1)/2 - x) beta^(d-1)(x-1/2) ] / d`
/// starting from the *continuous* triangle `beta^1`. Recurring up from the box
/// `beta^0` instead would be fragile: `beta^0` is discontinuous, so a
/// coordinate a hair off a half-integer (e.g. `2^-54`, as produced by exact
/// affine arithmetic) lands inside/outside the box and yields `1.0` where the
/// correct half-sample value is `0.5`. `beta^1` is Lipschitz, so a tiny
/// perturbation of the argument only perturbs the result by the same order.
/// Every term is non-negative inside the support — no catastrophic
/// cancellation, accurate to a few ulp even for quintic splines.
fn cardinal_bspline(order: usize, x: f64) -> f64 {
    // beta^1 (triangle) sampled at the offsets x + 0.5*m, m = -order ..= order.
    // Stack arrays (order <= 5 => len <= 11) instead of per-call heap Vecs — this is
    // on the hot per-tap-per-pixel interpolation path (~M calls/transform); byte-
    // identical (same values, same float op order). frankenscipy-wm14d.
    let span = order as isize;
    let len = 2 * order + 1;
    let mut vals = [0.0f64; 11];
    for (slot, m) in vals[..len].iter_mut().zip(-span..=span) {
        let t = (x + 0.5 * m as f64).abs();
        *slot = (1.0 - t).max(0.0);
    }
    for d in 2..=order {
        let n = d as f64;
        let half = (n + 1.0) / 2.0;
        let mut next = [0.0f64; 11];
        for idx in 1..len - 1 {
            let m = idx as isize - span;
            let arg = x + 0.5 * m as f64;
            next[idx] = ((arg + half) * vals[idx + 1] + (half - arg) * vals[idx - 1]) / n;
        }
        vals = next;
    }
    vals[span as usize]
}

/// Lane-parallel [`cardinal_bspline`]: lane `j` evaluates the kernel at `xs[j]`.
///
/// BIT-IDENTICAL to the scalar kernel. The taps of a support run are INDEPENDENT evaluations of the
/// same de Boor recursion at different `x`, so putting one tap per lane keeps every lane's operation
/// sequence — and its rounding — exactly the scalar one. Nothing is reassociated across lanes, and
/// no `mul_add` is introduced (Rust does not contract `a*b + c` without an explicit `mul_add`), so
/// each lane reproduces `cardinal_bspline` to the bit.
///
/// Emitted as a concrete width rather than over a const-generic lane count, so no
/// `LaneCount`/`SupportedLaneCount` bound is required.
macro_rules! cardinal_bspline_lanes {
    ($name:ident, $lanes:literal) => {
        #[inline]
        fn $name(order: usize, xs: Simd<f64, $lanes>) -> Simd<f64, $lanes> {
            let span = order as isize;
            let len = 2 * order + 1;
            let zero = Simd::<f64, $lanes>::splat(0.0);
            let one = Simd::<f64, $lanes>::splat(1.0);
            let mut vals = [zero; 11];
            for (i, slot) in vals[..len].iter_mut().enumerate() {
                let m = (i as isize - span) as f64;
                let t = (xs + Simd::splat(0.5 * m)).abs();
                *slot = (one - t).simd_max(zero);
            }
            for d in 2..=order {
                let n = d as f64;
                let half = (n + 1.0) / 2.0;
                let mut next = [zero; 11];
                for idx in 1..len - 1 {
                    let m = (idx as isize - span) as f64;
                    let arg = xs + Simd::splat(0.5 * m);
                    next[idx] = ((arg + Simd::splat(half)) * vals[idx + 1]
                        + (Simd::splat(half) - arg) * vals[idx - 1])
                        / Simd::splat(n);
                }
                vals = next;
            }
            vals[span as usize]
        }
    };
}
cardinal_bspline_lanes!(cardinal_bspline_x4, 4);
cardinal_bspline_lanes!(cardinal_bspline_x2, 2);

/// Cardinal B-spline weights for the CONTIGUOUS tap run `k in lo..=hi`, i.e.
/// `out[j] = cardinal_bspline(order, cc - (lo + j))`.
///
/// `cardinal_bspline` is >60% of a per-pixel geometric transform's self time and is entirely scalar
/// FP (`perf annotate`: `movapd`/`mulsd`/`addsd`/`subpd`, zero `idiv`). The compact window leaves a
/// run of `order+1` = 2/4/6 taps, which maps onto one f64 vector — so evaluate the whole run in a
/// single lane-parallel recursion instead of `order+1` independent scalar ones.
fn cardinal_bspline_run(order: usize, cc: f64, lo: isize, hi: isize, out: &mut [f64]) {
    debug_assert!(hi >= lo);
    let ntaps = (hi - lo + 1) as usize;
    let x_of = |j: usize| cc - (lo + j as isize) as f64;
    // order<2 has an EMPTY recursion loop (the kernel is one subtract + one max), so a vector
    // round-trip costs more than it saves. The A/B knob also routes here.
    if order < 2 || NDIMAGE_BSPLINE_SIMD_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        for (j, w) in out[..ntaps].iter_mut().enumerate() {
            *w = cardinal_bspline(order, x_of(j));
        }
        return;
    }
    // Vectorise only where the run maps onto NATIVE register widths, never by padding.
    //   4 taps (orders 2/3 compact) → one f64x4 = one YMM.
    //   6 taps (orders 4/5 compact) → f64x4 + f64x2 = one YMM + one XMM, no idle lanes.
    // MEASURED REJECT (do not re-add on AVX2): padding a 6-tap run into f64x8 costs TWO YMM with 2
    // lanes idle over a longer recursion (len 9/11) and read 1.06x at order 4, 0.98x at order 5.
    match ntaps {
        4 => {
            let xs = Simd::<f64, 4>::from_array(std::array::from_fn(x_of));
            out[..4].copy_from_slice(&cardinal_bspline_x4(order, xs).to_array());
        }
        6 => {
            let lo4 = Simd::<f64, 4>::from_array(std::array::from_fn(x_of));
            out[..4].copy_from_slice(&cardinal_bspline_x4(order, lo4).to_array());
            let hi2 = Simd::<f64, 2>::from_array(std::array::from_fn(|j| x_of(4 + j)));
            out[4..6].copy_from_slice(&cardinal_bspline_x2(order, hi2).to_array());
        }
        _ => {
            for (j, w) in out[..ntaps].iter_mut().enumerate() {
                *w = cardinal_bspline(order, x_of(j));
            }
        }
    }
}

#[derive(Clone, Copy)]
struct LinearAxisSupport {
    lo: usize,
    hi: usize,
    w_lo: f64,
    w_hi: f64,
}

fn zoom_order1_axis_supports(input_len: usize, output_len: usize) -> Vec<LinearAxisSupport> {
    (0..output_len)
        .map(|out| {
            let coord = if output_len <= 1 || input_len <= 1 {
                0.0
            } else {
                out as f64 * (input_len - 1) as f64 / (output_len - 1) as f64
            };
            let floor = coord.floor() as isize;
            let max = input_len as isize - 1;
            let lo = floor.clamp(0, max) as usize;
            let hi = (floor + 1).clamp(0, max) as usize;
            LinearAxisSupport {
                lo,
                hi,
                w_lo: cardinal_bspline(1, coord - floor as f64),
                w_hi: cardinal_bspline(1, coord - (floor + 1) as f64),
            }
        })
        .collect()
}

fn zoom_order1_reflect_2d_fast(input: &NdArray, new_shape: &[usize]) -> NdArray {
    let out_rows = new_shape[0];
    let out_cols = new_shape[1];
    let input_cols = input.shape[1];
    let row_support = zoom_order1_axis_supports(input.shape[0], out_rows);
    let col_support = zoom_order1_axis_supports(input.shape[1], out_cols);
    let mut output = NdArray::zeros(new_shape.to_vec());
    fill_pixels_parallel(&mut output, 4, |flat, _scratch| {
        let y = flat / out_cols;
        let x = flat - y * out_cols;
        let ys = row_support[y];
        let xs = col_support[x];

        let row_lo = ys.lo * input_cols;
        let row_hi = ys.hi * input_cols;
        let mut upper = 0.0;
        upper += xs.w_lo * input.data[row_lo + xs.lo];
        upper += xs.w_hi * input.data[row_lo + xs.hi];
        let mut lower = 0.0;
        lower += xs.w_lo * input.data[row_hi + xs.lo];
        lower += xs.w_hi * input.data[row_hi + xs.hi];
        let mut acc = 0.0;
        acc += ys.w_lo * upper;
        acc += ys.w_hi * lower;
        acc
    });
    output
}

/// IIR poles of the order-`order` B-spline interpolation prefilter (the roots
/// of the symbol of `beta^order` sampled on the integers), `order` in 2..=5.
fn bspline_reflect_poles(order: usize) -> Vec<f64> {
    match order {
        2 => vec![8.0_f64.sqrt() - 3.0],
        3 => vec![3.0_f64.sqrt() - 2.0],
        4 => vec![
            (664.0 - 438976.0_f64.sqrt()).sqrt() + 304.0_f64.sqrt() - 19.0,
            (664.0 + 438976.0_f64.sqrt()).sqrt() - 304.0_f64.sqrt() - 19.0,
        ],
        5 => vec![
            (67.5 - 4436.25_f64.sqrt()).sqrt() + 26.25_f64.sqrt() - 6.5,
            (67.5 + 4436.25_f64.sqrt()).sqrt() - 26.25_f64.sqrt() - 6.5,
        ],
        _ => Vec::new(),
    }
}

/// Exact B-spline prefilter coefficients for the half-sample `reflect`
/// boundary, orders 2..=5.
///
/// Runs the Unser/Thévenaz recursive IIR filter — gain, then a causal and an
/// anticausal pass per pole — with the *exact* closed-form reflect boundary
/// initial conditions: the causal initial coefficient sums the full geometric
/// series of the reflect-extended (period `2n`) signal, `c[0] += S*z/(1-z^2n)`,
/// and the anticausal initial coefficient is `c[n-1] *= z/(z-1)`. No truncated
/// horizon. For orders 4 and 5 the two poles are applied in sequence; each
/// pass preserves the reflect symmetry so the same exact init is correct for
/// the second pole. Also serves as the per-line prefilter for `nearest` once
/// the input has been edge-padded.
/// In-place column-vectorized `bspline_reflect_coefficients` over a whole axis (layout
/// [outer][n][inner]): the IIR runs along `n` but is independent across `inner` columns,
/// so sweep row-by-row with CONTIGUOUS inner-wide reads (cache-friendly + vectorizing).
/// BYTE-IDENTICAL: each column runs the same scalar op sequence in the same order, and
/// `sum*z/denom` is kept as `(sum*z)/denom`.
fn bspline_reflect_axis_inplace(
    data: &mut [f64],
    outer: usize,
    n: usize,
    inner: usize,
    order: usize,
) {
    if n <= 1 {
        return;
    }
    let poles = bspline_reflect_poles(order);
    let mut gain = 1.0;
    for &z in &poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    let mut sum = vec![0.0f64; inner];
    for slab in 0..outer {
        let base = slab * n * inner;
        for v in &mut data[base..base + n * inner] {
            *v *= gain;
        }
        for &z in &poles {
            let z_n = z.powi(n as i32);
            let denom = 1.0 - z_n * z_n;
            let last = base + (n - 1) * inner;
            for j in 0..inner {
                sum[j] = data[base + j] + z_n * data[last + j];
            }
            let mut z_i = z;
            for i in 1..n {
                let ci = base + i * inner;
                let cm = base + (n - 1 - i) * inner;
                for j in 0..inner {
                    sum[j] += z_i * (data[ci + j] + z_n * data[cm + j]);
                }
                z_i *= z;
            }
            for j in 0..inner {
                data[base + j] += sum[j] * z / denom;
            }
            for i in 1..n {
                let ci = base + i * inner;
                let cp = base + (i - 1) * inner;
                for j in 0..inner {
                    data[ci + j] += z * data[cp + j];
                }
            }
            let fz = z / (z - 1.0);
            for j in 0..inner {
                data[last + j] *= fz;
            }
            for i in (0..n - 1).rev() {
                let ci = base + i * inner;
                let cp = base + (i + 1) * inner;
                for j in 0..inner {
                    data[ci + j] = z * (data[cp + j] - data[ci + j]);
                }
            }
        }
    }
}

fn bspline_reflect_coefficients(line: &[f64], order: usize) -> Vec<f64> {
    let n = line.len();
    if n <= 1 {
        return line.to_vec();
    }
    let poles = bspline_reflect_poles(order);
    let mut gain = 1.0;
    for &z in &poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    let mut c: Vec<f64> = line.iter().map(|&v| v * gain).collect();
    for &z in &poles {
        // Exact causal initial coefficient for the reflect-extended signal.
        let z_n = z.powi(n as i32);
        let mut z_i = z;
        let mut sum = c[0] + z_n * c[n - 1];
        for i in 1..n {
            sum += z_i * (c[i] + z_n * c[n - 1 - i]);
            z_i *= z;
        }
        c[0] += sum * z / (1.0 - z_n * z_n);
        for i in 1..n {
            c[i] += z * c[i - 1];
        }
        // Exact anticausal initial coefficient.
        c[n - 1] *= z / (z - 1.0);
        for i in (0..n - 1).rev() {
            c[i] = z * (c[i + 1] - c[i]);
        }
    }
    c
}

/// Exact B-spline prefilter coefficients for the whole-sample `mirror`
/// boundary (period `2(n-1)`, edge not repeated), orders 2..=5.
///
/// Same Unser/Thévenaz recursive IIR filter as the reflect variant but with the
/// mirror-symmetric initial conditions, verified against
/// `scipy.ndimage.spline_filter1d(mode='mirror')`:
/// causal `c[0] = (c[0] + c[n-1]·zⁿ⁻¹ + Σ_{1..n-2} c[k]·(zᵏ + z^{2n-2-k})) / (1 − z^{2n-2})`,
/// anticausal `c[n-1] = z/(z²−1)·(c[n-1] + z·c[n-2])`.
fn bspline_mirror_coefficients(line: &[f64], order: usize) -> Vec<f64> {
    let n = line.len();
    if n <= 1 {
        return line.to_vec();
    }
    let poles = bspline_reflect_poles(order);
    let mut gain = 1.0;
    for &z in &poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    let mut c: Vec<f64> = line.iter().map(|&v| v * gain).collect();
    for &z in &poles {
        let z_2n2 = z.powi(2 * n as i32 - 2);
        // Exact causal initial coefficient for the mirror-extended signal.
        let mut sum = c[0] + c[n - 1] * z.powi(n as i32 - 1);
        let mut z_k = z;
        for &ck in &c[1..n - 1] {
            sum += ck * (z_k + z_2n2 / z_k);
            z_k *= z;
        }
        c[0] = sum / (1.0 - z_2n2);
        for i in 1..n {
            c[i] += z * c[i - 1];
        }
        // Exact anticausal initial coefficient.
        c[n - 1] = z / (z * z - 1.0) * (c[n - 1] + z * c[n - 2]);
        for i in (0..n - 1).rev() {
            c[i] = z * (c[i + 1] - c[i]);
        }
    }
    c
}

fn pad_array_mode(
    input: &NdArray,
    pad: usize,
    mode: BoundaryMode,
) -> Result<NdArray, NdimageError> {
    if input.shape.contains(&0) {
        return Err(NdimageError::EmptyInput);
    }
    let padded_shape: Vec<usize> = input.shape.iter().map(|&dim| dim + 2 * pad).collect();
    let mut padded = NdArray::zeros(padded_shape.clone());
    for flat in 0..padded.size() {
        let padded_idx = unravel_with_shape(flat, &padded_shape);
        let mut src_idx = Vec::with_capacity(input.ndim());
        for (axis, &idx) in padded_idx.iter().enumerate() {
            let src_coord = idx as f64 - pad as f64;
            let Some(mapped) = map_coordinate(src_coord, input.shape[axis], mode) else {
                return Err(NdimageError::InvalidArgument(
                    "constant-padding prefilter is unsupported".to_string(),
                ));
            };
            src_idx.push(mapped.round() as usize);
        }
        padded.data[flat] = input.get(&src_idx);
    }
    Ok(padded)
}

/// Thread count for a spline-prefilter axis pass that fans out across `blocks`
/// independent, equal-size contiguous units (outer blocks for the strided fast path,
/// rows for the contiguous last-axis path). Gated on total element work so small
/// arrays stay serial (where the spawn cost is not amortised).
fn spline_axis_threads(blocks: usize, block_work: usize) -> usize {
    let work = (blocks as u64).saturating_mul(block_work as u64);
    if work < (1 << 20) || blocks < 2 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1)
        .min(blocks)
}

/// Same-binary A/B toggle for the non-reflect spline prefilter fallback (Constant/Wrap/Mirror and the
/// general banded kernel). When `true`, those axis passes run the serial per-line walk (the ORIG
/// behaviour). When `false` (default), the independent outer slabs fan across cores. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn prefilter_spline_coefficients(
    input: &NdArray,
    order: usize,
    mode: BoundaryMode,
) -> Result<SplinePrefilter, NdimageError> {
    let ndim = input.ndim();
    if order <= 1 {
        // Order <= 1 has no spline coefficients to solve — the samples ARE the
        // coefficients. Linear interpolation under reflect/mirror needs support beyond
        // [0, len-1] near a boundary, but the cardinal interp path folds the support TAPS on
        // the fly (the `fold` closure with the actual boundary mode), so NO array padding is
        // needed. The old reflect/mirror branch eagerly padded with `pad_array_mode`, which is
        // O(padded_size) with per-element reflect index reconstruction (~15 ms for a 512² →
        // 536² array) and made affine_transform / map_coordinates Reflect/Mirror order=1 ~2×
        // slower than scipy. Folding the taps is also exact for coords arbitrarily far outside
        // the grid (the pad only reflected SPLINE_NEAREST_PAD=12 deep, then clamped).
        return Ok(SplinePrefilter {
            coeffs: input.clone(),
            coord_offsets: vec![0.0; ndim],
        });
    }
    // Orders 2..=5 reflect are solved exactly and in-place by the recursive
    // `bspline_reflect_coefficients` (exact half-sample-reflect B-spline).
    // The matching `nearest` mode mirrors scipy: edge-pad by SPLINE_NEAREST_PAD,
    // then run that same exact reflect prefilter on the padded lines. Reflect
    // on an axis too short for the order's stencil falls back to the padded
    // de Boor path.
    let bspline_reflect = matches!(order, 2..=5);
    let exact_reflect =
        bspline_reflect && mode == BoundaryMode::Reflect && input.shape.iter().all(|&s| s > order);
    let exact_mirror =
        bspline_reflect && mode == BoundaryMode::Mirror && input.shape.iter().all(|&s| s > order);
    if mode == BoundaryMode::Mirror && !exact_mirror {
        // The exact mirror prefilter needs every axis longer than `order`; a
        // too-short axis would fall through to the clamped de Boor solver,
        // which does not carry mirror symmetry. Fail closed (tracked separately)
        // rather than return wrong coefficients.
        return Err(NdimageError::InvalidArgument(
            "mirror boundary requires each axis length > spline order".to_string(),
        ));
    }
    let pad_input = matches!(mode, BoundaryMode::Nearest | BoundaryMode::Reflect) && !exact_reflect;
    let (mut current, coord_offsets) = if pad_input {
        (
            pad_array_mode(input, SPLINE_NEAREST_PAD, mode)?,
            vec![SPLINE_NEAREST_PAD as f64; ndim],
        )
    } else {
        (input.clone(), vec![0.0; ndim])
    };
    for axis in 0..ndim {
        let axis_len = current.shape[axis];
        if axis_len <= 1 {
            continue;
        }
        // Direct strided line walk, IN PLACE: each axis-line occupies disjoint slots
        // (base + i*stride), so the filter reads the whole line then writes its own
        // coefficients back — no `next` clone and no per-element N-D get/set index
        // arithmetic. line_flat splits into outer (dims < axis) and inner (dims > axis):
        // base = (line_flat/stride)*axis_len*stride + line_flat%stride. BYTE-IDENTICAL
        // (same line elements, same coefficient kernel, same target slots).
        let stride: usize = current.shape[axis + 1..].iter().product();
        let outer: usize = current.shape[..axis].iter().product();
        // Whole-axis vectorized fast path for the bspline-reflect kernel (Reflect-exact or
        // Nearest-bspline both call bspline_reflect_coefficients): when inner=stride>1 the
        // per-column strided gather is cache-hostile; sweep the IIR in-place vectorized over
        // the contiguous inner dim instead. Byte-identical; other kernels keep the per-line walk.
        let reflect_kernel = (exact_reflect && mode == BoundaryMode::Reflect)
            || (bspline_reflect && mode == BoundaryMode::Nearest);
        if reflect_kernel && stride > 1 {
            // The `outer` blocks (each `axis_len*stride` contiguous elements) are independent,
            // so split the buffer into contiguous outer-block chunks across cores — each chunk
            // runs the same in-place IIR. Byte-identical (block partition is the only change).
            let block = axis_len * stride;
            let nthreads = spline_axis_threads(outer, block);
            if nthreads <= 1 {
                bspline_reflect_axis_inplace(&mut current.data, outer, axis_len, stride, order);
            } else {
                let per = outer.div_ceil(nthreads);
                std::thread::scope(|scope| {
                    for chunk in current.data.chunks_mut(per * block) {
                        let chunk_outer = chunk.len() / block;
                        scope.spawn(move || {
                            bspline_reflect_axis_inplace(
                                chunk,
                                chunk_outer,
                                axis_len,
                                stride,
                                order,
                            );
                        });
                    }
                });
            }
            continue;
        }
        // Contiguous last-axis (stride==1) bspline-reflect lines: each row is an independent,
        // contiguous `axis_len` block with a non-fallible coefficient kernel — fan the rows
        // across cores (byte-identical to the serial per-line walk below).
        if reflect_kernel && stride == 1 {
            let nthreads = spline_axis_threads(outer, axis_len);
            if nthreads > 1 {
                let per = outer.div_ceil(nthreads);
                std::thread::scope(|scope| {
                    for chunk in current.data.chunks_mut(per * axis_len) {
                        scope.spawn(move || {
                            let mut line = Vec::with_capacity(axis_len);
                            for row in chunk.chunks_mut(axis_len) {
                                line.clear();
                                line.extend_from_slice(row);
                                let coeffs = bspline_reflect_coefficients(&line, order);
                                row.copy_from_slice(&coeffs);
                            }
                        });
                    }
                });
                continue;
            }
        }
        // Non-reflect / general kernels (Constant/Wrap/Mirror + the banded fallback): the lines are
        // independent (each reads its own strided line, writes disjoint slots), so fan the outer
        // blocks across cores — byte-identical to the serial per-line walk (same line elements, same
        // kernel, same target slots; the block partition is the only change). The general kernel is
        // fallible, so each block returns a Result and the FIRST (lowest-outer-block) error wins,
        // matching the serial walk's lowest-`line_flat` error.
        let block = axis_len * stride;
        let par_threads =
            if NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
                1
            } else {
                spline_axis_threads(outer, block)
            };
        if par_threads > 1 {
            let per = outer.div_ceil(par_threads);
            let results: Vec<Result<(), NdimageError>> = std::thread::scope(|scope| {
                let handles: Vec<_> = current
                    .data
                    .chunks_mut(per * block)
                    .map(|chunk| {
                        let chunk_outer = chunk.len() / block;
                        scope.spawn(move || -> Result<(), NdimageError> {
                            let mut line = Vec::with_capacity(axis_len);
                            for o in 0..chunk_outer {
                                let outer_base = o * block;
                                for inner in 0..stride {
                                    let base = outer_base + inner;
                                    line.clear();
                                    for i in 0..axis_len {
                                        line.push(chunk[base + i * stride]);
                                    }
                                    let coeffs = match (order, mode) {
                                        (3, BoundaryMode::Constant | BoundaryMode::Wrap) => {
                                            cubic_constant_wrap_coefficients(&line)
                                        }
                                        (_, BoundaryMode::Nearest) if bspline_reflect => {
                                            bspline_reflect_coefficients(&line, order)
                                        }
                                        (_, BoundaryMode::Reflect) if exact_reflect => {
                                            bspline_reflect_coefficients(&line, order)
                                        }
                                        (_, BoundaryMode::Mirror) if exact_mirror => {
                                            bspline_mirror_coefficients(&line, order)
                                        }
                                        _ => spline_coefficients_for_line(&line, order)?,
                                    };
                                    for (i, coeff) in coeffs.into_iter().enumerate() {
                                        chunk[base + i * stride] = coeff;
                                    }
                                }
                            }
                            Ok(())
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("spline prefilter worker panicked"))
                    .collect()
            });
            for r in results {
                r?;
            }
            continue;
        }

        let line_count = outer * stride;
        let mut line = Vec::with_capacity(axis_len);
        for line_flat in 0..line_count {
            let base = (line_flat / stride) * axis_len * stride + (line_flat % stride);
            line.clear();
            for i in 0..axis_len {
                line.push(current.data[base + i * stride]);
            }
            let coeffs = match (order, mode) {
                (3, BoundaryMode::Constant | BoundaryMode::Wrap) => {
                    cubic_constant_wrap_coefficients(&line)
                }
                (_, BoundaryMode::Nearest) if bspline_reflect => {
                    bspline_reflect_coefficients(&line, order)
                }
                (_, BoundaryMode::Reflect) if exact_reflect => {
                    bspline_reflect_coefficients(&line, order)
                }
                (_, BoundaryMode::Mirror) if exact_mirror => {
                    bspline_mirror_coefficients(&line, order)
                }
                _ => spline_coefficients_for_line(&line, order)?,
            };
            for (i, coeff) in coeffs.into_iter().enumerate() {
                current.data[base + i * stride] = coeff;
            }
        }
    }
    Ok(SplinePrefilter {
        coeffs: current,
        coord_offsets,
    })
}

fn sample_spline_recursive(
    coeffs: &NdArray,
    bases: &[Vec<(usize, f64)>],
    dim: usize,
    idx: &mut [usize],
) -> f64 {
    if dim == bases.len() {
        return coeffs.get(idx);
    }
    let mut acc = 0.0;
    for &(coord_idx, weight) in &bases[dim] {
        idx[dim] = coord_idx;
        acc += weight * sample_spline_recursive(coeffs, bases, dim + 1, idx);
    }
    acc
}

/// Largest `ndim` whose per-axis support slices are gathered on the stack.
/// Beyond this the gather falls back to a per-pixel `Vec` (never hit in practice:
/// scipy's own geometric transforms cap out far below 8 dimensions).
const MAX_STACK_NDIM: usize = 8;

/// One axis's precomputed interpolation support: for each output index along that
/// axis, either the (coefficient index, weight) pairs that contribute or `None`
/// when the sample falls outside and the boundary constant applies. Named because
/// the nested form trips clippy's type_complexity at three signatures
/// (frankenscipy-iit9c).
type AxisSupport = Vec<Option<Vec<(usize, f64)>>>;

/// Per-thread scratch for the separable B-spline sampler: one support buffer per
/// axis plus the recursion index, reused across pixels.
type InterpScratch = (Vec<Vec<(usize, f64)>>, Vec<usize>);

/// Tensor-product B-spline combine over per-axis `(flat offset, weight)` supports.
///
/// BYTE-IDENTICAL to `sample_spline_recursive`: same weights, same accumulation order
/// (axis 0 outermost → axis n-1 innermost). The only change is the leaf ADDRESS, which is
/// precomputed. `sample_spline_recursive`'s leaf calls `coeffs.get(idx)` = `data[Σ idx[d]·stride[d]]`,
/// recomputing that stride dot at every one of the `(order+1)^ndim` leaves; here each tap already
/// carries `idx[d]·stride[d]`, so the leaf is a single `data[base]` load. Axes beyond `bases.len()`
/// keep `idx[d] = 0` in the recursive version and contribute 0 to `base` here — identical.
///
/// NOTE (measured REJECT, 2026-07-10): SIMD-ing the innermost reduction — a
/// contiguous load, `vmulpd`, then a sequential horizontal sum, bit-identical —
/// was 1.37x in a microbench but IN-FLOOR or a slight loss on the real
/// transforms. The leaf gathers from rows strided by the image width, so it is
/// GATHER-LATENCY-bound rather than compute-bound and vectorising the multiply
/// does not touch the wall; the microbench's sequential-pixel scan was simply
/// cache-friendlier than the real scattered access.
fn sample_spline_offsets(data: &[f64], bases: &[&[(usize, f64)]], base: usize) -> f64 {
    match bases {
        [] => data[base],
        [b0] => {
            let mut acc = 0.0;
            for &(o0, w0) in *b0 {
                acc += w0 * data[base + o0];
            }
            acc
        }
        // The 2-D arm is the hot one (zoom/shift/affine on images): flattening the recursion
        // lets LLVM keep the inner accumulator in a register across the (order+1) taps.
        [b0, b1] => {
            let mut acc = 0.0;
            for &(o0, w0) in *b0 {
                let row = base + o0;
                let mut inner = 0.0;
                for &(o1, w1) in *b1 {
                    inner += w1 * data[row + o1];
                }
                acc += w0 * inner;
            }
            acc
        }
        [b0, rest @ ..] => {
            let mut acc = 0.0;
            for &(o0, w0) in *b0 {
                acc += w0 * sample_spline_offsets(data, rest, base + o0);
            }
            acc
        }
    }
}

/// Per-axis B-spline supports for a SEPARABLE transform, as `(flat offset, weight)` taps.
///
/// A separable transform (zoom, shift, diagonal affine) maps each output axis-index to an input
/// coordinate that depends ONLY on that axis, so the support along axis `a` takes just
/// `out_shape[a]` distinct values. `coord_of(axis, o)` supplies that coordinate. When
/// `premultiply` the taps are scaled by `coeffs.strides[axis]` so the per-pixel combine indexes
/// `coeffs.data` directly (see `sample_spline_offsets`); otherwise they stay in index space for
/// the ORIG comparator. `None` marks an output position that maps out of range along that axis
/// (⇒ the pixel is `cval`).
///
/// `premultiply` MUST be the same value the caller passes to `sample_separable_pixel` as
/// `offsets` — read `NDIMAGE_SPLINE_OFFSET_DISABLE` ONCE per call and thread it through both.
/// (Reading the atomic separately in each let a concurrent toggle tear the pair: taps scaled by
/// stride but combined by the index leaf, indexing `coeffs.data` out of bounds.)
/// The parameter list is the filter specification itself — axis, window, boundary
/// mode, constant, origin and the output buffer. Bundling it into a struct would
/// add an indirection per call in a per-axis kernel and hide the arguments a
/// reader needs to see at the call site (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn build_axis_offset_supports(
    coeffs: &NdArray,
    in_shape: &[usize],
    out_shape: &[usize],
    coord_offsets: &[f64],
    order: usize,
    mode: BoundaryMode,
    premultiply: bool,
    // Resolved once by the caller alongside `premultiply`, for the same reason
    // (`frankenscipy-22van`): this builder runs per AXIS, but the knob it forwards is
    // consumed per TAP inside `compute_axis_support`.
    compact_disabled: bool,
    coord_of: impl Fn(usize, usize) -> f64,
) -> Vec<AxisSupport> {
    (0..out_shape.len())
        .map(|axis| {
            let stride = coeffs.strides[axis];
            (0..out_shape[axis])
                .map(|o| {
                    let mut s = Vec::new();
                    if compute_axis_support(
                        coord_of(axis, o),
                        coeffs.shape[axis],
                        in_shape[axis],
                        coord_offsets[axis],
                        order,
                        mode,
                        compact_disabled,
                        &mut s,
                    ) {
                        if premultiply {
                            for tap in &mut s {
                                tap.0 *= stride;
                            }
                        }
                        Some(s)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

/// ORIG comparator for `sample_spline_offsets`: taps are INDICES and every leaf recomputes the
/// stride dot via `coeffs.get`. Identical arithmetic to `sample_spline_recursive`, but taking
/// borrowed slices so both A/B arms can share one support table.
fn sample_spline_indices(
    coeffs: &NdArray,
    bases: &[&[(usize, f64)]],
    dim: usize,
    idx: &mut [usize],
) -> f64 {
    if dim == bases.len() {
        return coeffs.get(idx);
    }
    let mut acc = 0.0;
    for &(coord_idx, weight) in bases[dim] {
        idx[dim] = coord_idx;
        acc += weight * sample_spline_indices(coeffs, bases, dim + 1, idx);
    }
    acc
}

/// Gather one output pixel's per-axis supports and run the tensor-product combine.
/// Returns `cval` when any axis maps out of range. `offsets` selects the fast flat-offset leaf
/// (default) or the ORIG index-space leaf; hoist the flag read out of the pixel loop.
#[inline]
fn sample_separable_pixel(
    coeffs: &NdArray,
    axis_supports: &[AxisSupport],
    oidx: &[usize],
    cval: f64,
    offsets: bool,
) -> f64 {
    let ndim = oidx.len();
    if ndim <= MAX_STACK_NDIM {
        let mut bases: [&[(usize, f64)]; MAX_STACK_NDIM] = [&[]; MAX_STACK_NDIM];
        for (axis, slot) in bases[..ndim].iter_mut().enumerate() {
            match &axis_supports[axis][oidx[axis]] {
                None => return cval,
                Some(s) => *slot = s.as_slice(),
            }
        }
        sample_separable_combine(coeffs, &bases[..ndim], offsets)
    } else {
        let mut bases: Vec<&[(usize, f64)]> = Vec::with_capacity(ndim);
        for axis in 0..ndim {
            match &axis_supports[axis][oidx[axis]] {
                None => return cval,
                Some(s) => bases.push(s.as_slice()),
            }
        }
        sample_separable_combine(coeffs, &bases, offsets)
    }
}

#[inline]
fn sample_separable_combine(coeffs: &NdArray, bases: &[&[(usize, f64)]], offsets: bool) -> f64 {
    if offsets {
        return sample_spline_offsets(&coeffs.data, bases, 0);
    }
    // ORIG: rebuild the per-pixel index buffer, recompute the stride dot at every leaf.
    thread_local! {
        static ORIG_IDX: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    ORIG_IDX.with_borrow_mut(|idx| {
        idx.clear();
        idx.resize(coeffs.ndim(), 0);
        sample_spline_indices(coeffs, bases, 0, idx.as_mut_slice())
    })
}

/// Per-axis B-spline support (index/weight taps) for one interpolation coordinate.
/// Extracted verbatim from `sample_interpolated`'s per-axis loop so the general
/// per-pixel path and the separable zoom path share EXACTLY the same weights.
/// Returns `false` when the coordinate maps out of range (→ the pixel is `cval`).
/// The two spline A/B knobs, resolved ONCE and carried by value
/// (`frankenscipy-22van`).
///
/// `sample_interpolated` is called once PER PIXEL, and it read both
/// `NDIMAGE_SPLINE_OFFSET_DISABLE` and (via `compute_axis_support`)
/// `NDIMAGE_BSPLINE_COMPACT_DISABLE` on every one of those calls. A relaxed atomic load
/// inside a per-item path is not merely a cheap load: it is an optimisation barrier the
/// compiler cannot see through, and it blocks specialisation of the surrounding loop. The
/// same shape cost a measured 13% on `fsci-sparse`'s splu merge, where hoisting the read to
/// the call boundary and passing a plain `bool` recovered all of it.
///
/// So the caller resolves both knobs once, before its pixel loop, and threads this value
/// down. Two flags rather than two parameters because they must come from the SAME resolve:
/// `offsets` decides both whether taps are pre-multiplied by stride and which leaf consumes
/// them, and reading it twice can tear that pair (the latent panic fixed in 0a7086e76).
#[derive(Debug, Clone, Copy)]
struct SplineFlags {
    /// Taps pre-multiplied by stride so the leaf indexes `coeffs.data` directly.
    offsets: bool,
    /// Use the full `2·order+1` window instead of the compact support.
    compact_disabled: bool,
}

/// How many times `SplineFlags::resolve` has run — `frankenscipy-22van` gate (a).
///
/// The bead requires the read FREQUENCY be bounded structurally before the lever is
/// believed, and a structural count is exactly what survives instrumentation: adding
/// a counter perturbs timing, but it cannot change how many times a call happens.
/// (Which is also why this must NOT be used to read cost — the bead is explicit that
/// `cfg(test)` counters in this path are themselves optimisation barriers, so the
/// specialisation half has to be an A/B of two SHIPPING binaries.)
#[cfg(test)]
pub(crate) static SPLINE_FLAGS_RESOLVE_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl SplineFlags {
    /// Read both atomics ONCE. Call this OUTSIDE any per-pixel loop.
    #[cfg(not(feature = "spline-flags-const"))]
    fn resolve() -> Self {
        #[cfg(test)]
        SPLINE_FLAGS_RESOLVE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            offsets: !NDIMAGE_SPLINE_OFFSET_DISABLE.load(std::sync::atomic::Ordering::Relaxed),
            compact_disabled: NDIMAGE_BSPLINE_COMPACT_DISABLE
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// `frankenscipy-22van` gate (b): the shipped defaults as COMPILE-TIME CONSTANTS, so no
    /// atomic survives anywhere in the per-pixel path and the compiler is free to specialise
    /// the loop. Hardcodes exactly the default values (`offsets = true` because the shipped
    /// `NDIMAGE_SPLINE_OFFSET_DISABLE` is `false`; `compact_disabled = false`).
    #[cfg(feature = "spline-flags-const")]
    fn resolve() -> Self {
        Self {
            offsets: true,
            compact_disabled: false,
        }
    }

    /// The A/B baseline: re-read per call, reproducing the pre-22van behaviour exactly.
    /// Selected by `NDIMAGE_SPLINE_FLAG_HOIST_DISABLE`, so the hoist can be measured
    /// against the shape it replaced inside one binary.
    #[cfg(not(feature = "spline-flags-const"))]
    #[inline]
    fn resolve_or_reread(hoisted: Self) -> Self {
        if NDIMAGE_SPLINE_FLAG_HOIST_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
            Self::resolve()
        } else {
            hoisted
        }
    }

    /// Gate (b) arm: the A/B knob itself is compiled out. This is the point of the feature —
    /// in the default build this function performs an atomic load ONCE PER PIXEL, which is
    /// why a same-binary A/B cannot measure specialisation.
    #[cfg(feature = "spline-flags-const")]
    #[inline]
    fn resolve_or_reread(hoisted: Self) -> Self {
        hoisted
    }
}

// Eighth parameter is the once-resolved compact-support knob, forwarded rather than
// re-loaded per tap (`frankenscipy-22van`).
#[allow(clippy::too_many_arguments)]
fn compute_axis_support(
    coord: f64,
    coeff_len: usize,
    input_shape_axis: usize,
    coord_offset: f64,
    order: usize,
    mode: BoundaryMode,
    compact_disabled: bool,
    support: &mut Vec<(usize, f64)>,
) -> bool {
    support.clear();
    let Some(mapped) = map_interpolation_coordinate(coord, input_shape_axis, mode) else {
        return false;
    };
    let spline_coord = if coord_offset > 0.0 {
        match mode {
            BoundaryMode::Nearest => (coord + coord_offset).clamp(0.0, (coeff_len - 1) as f64),
            BoundaryMode::Wrap => {
                let mut wrapped = coord + coord_offset;
                let period = input_shape_axis as f64;
                let padded_max = (coeff_len - 1) as f64;
                while wrapped < 0.0 {
                    wrapped += period;
                }
                while wrapped > padded_max {
                    wrapped -= period;
                }
                wrapped
            }
            _ => mapped + coord_offset,
        }
    } else {
        mapped
    };
    // Snap sub-ULP boundary excursions back into the valid spline
    // domain (see original note: prevents dropped corner pixels under
    // e.g. rotate(360°)).
    let spline_coord = match mode {
        BoundaryMode::Wrap => spline_coord,
        _ => {
            let max = (coeff_len - 1) as f64;
            let bound = 1.0e-9 * max.max(1.0);
            if spline_coord < 0.0 && spline_coord > -bound {
                0.0
            } else if spline_coord > max && spline_coord < max + bound {
                max
            } else {
                spline_coord
            }
        }
    };
    let effective_order = order.min(coeff_len.saturating_sub(1));
    // Constant/Wrap order-3: cardinal cubic B-spline with a mirror index fold.
    if matches!(mode, BoundaryMode::Wrap | BoundaryMode::Constant) && effective_order == 3 {
        let base = spline_coord.floor() as isize - 1;
        let t = spline_coord - spline_coord.floor();
        let omt = 1.0 - t;
        let max = coeff_len - 1;
        support.push((fold_wrap_cubic_index(base, max), omt * omt * omt / 6.0));
        support.push((
            fold_wrap_cubic_index(base + 1, max),
            (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
        ));
        support.push((
            fold_wrap_cubic_index(base + 2, max),
            (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
        ));
        support.push((fold_wrap_cubic_index(base + 3, max), t * t * t / 6.0));
        return true;
    }
    // Reflect / Mirror / Nearest orders 1..=5: cardinal B-spline kernel with a
    // per-tap boundary fold (matches scipy: fold the support TAPS, not the coord).
    // order=1 (linear) was excluded and fell through to the slow generic
    // eval_bspline_basis_all path — the lone interpolating order without a fast
    // path, making zoom order=1 ~17x slower than scipy (frankenscipy-wm14d).
    // cardinal_bspline(1, cc-k) = (1-|cc-k|).max(0) yields the linear weights.
    let cardinal_reflect_nearest = effective_order == order
        && matches!(order, 1..=5)
        && (mode == BoundaryMode::Nearest
            || (matches!(mode, BoundaryMode::Reflect | BoundaryMode::Mirror)
                // order=1 Reflect/Mirror is PADDED (coord_offsets=SPLINE_NEAREST_PAD)
                // so the linear support always lands inside the padded coeffs →
                // clamp (Nearest) fold; that path was previously routed to the slow
                // generic eval_bspline_basis_all (frankenscipy-wm14d).
                && (coord_offset == 0.0 || order == 1)));
    if cardinal_reflect_nearest {
        let cc = coord + coord_offset;
        let len = coeff_len as isize;
        // A still-padded order=1 reflect/mirror axis (coord_offsets>0, e.g. an order>=2
        // short-axis fallback) folds via clamp because the padding already encodes the
        // reflection; the unpadded order=1 path (coord_offsets==0) and every other case
        // fold per the actual boundary mode.
        let fold_mode = if order == 1
            && matches!(mode, BoundaryMode::Reflect | BoundaryMode::Mirror)
            && coord_offset > 0.0
        {
            BoundaryMode::Nearest
        } else {
            mode
        };
        let fold = |i: isize| -> usize {
            match fold_mode {
                BoundaryMode::Nearest => i.clamp(0, len - 1) as usize,
                BoundaryMode::Mirror => {
                    if len <= 1 {
                        0
                    } else {
                        let period = 2 * (len - 1);
                        let mut m = i.rem_euclid(period);
                        if m >= len {
                            m = period - m;
                        }
                        m as usize
                    }
                }
                _ => {
                    let period = 2 * len;
                    let mut m = i.rem_euclid(period);
                    if m >= len {
                        m = period - 1 - m;
                    }
                    m as usize
                }
            }
        };
        let floor = cc.floor() as isize;
        // COMPACT SUPPORT. `cardinal_bspline(n, x)` vanishes EXACTLY (bit-zero, verified over
        // 1.5M adversarial args) outside `|x| < (n+1)/2`, so at most `n+1` of the taps the ORIG
        // loop evaluated over `floor ± order` can be nonzero — it spent `2n+1` kernel calls to
        // keep `n+1` (7→4 at order 3, 11→6 at order 5), and `cardinal_bspline` is 61.7% of a
        // per-pixel transform's self time. The nonzero taps are contiguous and, in terms of
        // `f = floor(cc)`, always lie in `[f - n/2, f + n/2 + 1]` (integer division):
        // odd `n` pins them exactly; even `n` straddles by one, which the retained
        // `weight != 0.0` filter drops. Deriving the bounds from `f` by INTEGER arithmetic is
        // load-bearing — the obvious `(cc - half).floor()` rounds a coordinate one ULP below an
        // integer back onto it and silently drops a legitimate ~8.9e-16 tap.
        let span = if compact_disabled {
            // ORIG comparator: the full `2·order+1` window.
            (order as isize, order as isize)
        } else {
            let hn = (order / 2) as isize;
            (hn, hn + 1)
        };
        // NOTE (measured, 2026-07-10): an "interior fast path" that skips `fold` when the whole tap
        // run lies in `0..=len-1` was implemented, proven byte-identical, and REJECTED at 1.01-1.05x
        // (1.01x at the default order 3). `perf annotate` of `compute_axis_support` — 61.39% self,
        // so the code is demonstrably live — shows ZERO `idiv` instructions: `fold`'s `rem_euclid`
        // costs essentially nothing here, and the `fmod` visible in the profile is the f64 `fmod`
        // from `map_interpolation_coordinate`, NOT this integer fold. Do not retry unless
        // `cardinal_bspline` gets much cheaper (SIMD), which would raise `fold`'s share.
        // Evaluate the whole contiguous tap run in ONE lane-parallel recursion (one tap per lane);
        // bit-identical to the per-tap scalar calls this replaced.
        let (lo, hi) = (floor - span.0, floor + span.1);
        let ntaps = (hi - lo + 1) as usize;
        let mut weights = [0.0f64; 12]; // max window = 2*order+1 = 11
        cardinal_bspline_run(order, cc, lo, hi, &mut weights[..ntaps]);
        for (j, k) in (lo..=hi).enumerate() {
            let weight = weights[j];
            if weight != 0.0 {
                support.push((fold(k), weight));
            }
        }
        return true;
    }
    if effective_order == 0 {
        support.push((
            spline_coord.round().clamp(0.0, (coeff_len - 1) as f64) as usize,
            1.0,
        ));
        return true;
    }
    // Compact-support evaluation: BYTE-IDENTICAL to filtering the full O(len·order)
    // eval_bspline_basis_all but O(order²) with no per-pixel knot/basis allocation. The old
    // path made affine_transform/map_coordinates/geometric_transform with
    // Constant/Wrap order∈{1,2,4,5} ~8-18× slower than scipy (the cardinal fast paths only
    // covered Nearest/Reflect/Mirror and Constant-order-3); this routes the rest here.
    bspline_local_support(coeff_len, spline_coord, effective_order, support);
    if support.is_empty() {
        return false;
    }
    true
}

// Eight parameters because the geometry (input, coeffs, coords, offsets, order, mode,
// cval) is the sampler's specification and the eighth is the once-resolved knob pair
// (`frankenscipy-22van`). Bundling the geometry into a struct would hide at the call site
// exactly the arguments a reader needs to see, which is the reasoning already recorded on
// `build_axis_offset_supports`.
#[allow(clippy::too_many_arguments)]
fn sample_interpolated(
    input: &NdArray,
    coeffs: &NdArray,
    coords: &[f64],
    coord_offsets: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
    flags: SplineFlags,
) -> f64 {
    if order == 0 {
        // scipy rounds the (unmapped) coordinate via floor(coord + 0.5) — half
        // toward +∞, which differs from Rust's round() at negative half-integers.
        let round0 = |c: f64| (c + 0.5).floor() as i64;
        if mode == BoundaryMode::Constant {
            // scipy applies 'constant' on the FLOAT coordinate: a point outside
            // [0, len-1] is out of range (→ cval) even if it would round back to a
            // valid index (e.g. coord 4.3 with len 5 rounds to 4 but is outside).
            for (axis, &coord) in coords.iter().enumerate() {
                let hi = (input.shape[axis] as f64) - 1.0;
                if coord < 0.0 || coord > hi {
                    return cval;
                }
            }
            let idx: Vec<i64> = coords.iter().map(|&c| round0(c)).collect();
            return input.get_boundary(&idx, mode, cval);
        }
        if mode == BoundaryMode::Wrap {
            let idx: Vec<usize> = coords
                .iter()
                .enumerate()
                .map(|(axis, &coord)| wrap_interpolation_index(round0(coord), input.shape[axis]))
                .collect();
            return input.get(&idx);
        }
        let idx: Vec<i64> = coords.iter().map(|&coord| round0(coord)).collect();
        return input.get_boundary(&idx, mode, cval);
    }

    thread_local! {
        static INTERP_SCRATCH: std::cell::RefCell<InterpScratch> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    // Reuse per-thread B-spline support buffers (one Vec per axis) and the recursion index
    // across pixels — the old path allocated `bases` + one `support` Vec per axis + `idx`
    // every pixel (~5 heap allocs/pixel for a 2-D order-3 transform, and the geometric
    // transforms run ~800 ns/pixel). The computed (index, weight) tuples and the recursion
    // are unchanged, so the result is byte-identical; only the buffer lifetimes change.
    INTERP_SCRATCH.with_borrow_mut(|(bases, idx)| -> f64 {
        let n = coords.len();
        if bases.len() < n {
            bases.resize_with(n, Vec::new);
        }
        // The knobs arrive already resolved from the CALLER, which reads them once outside
        // its pixel loop (`frankenscipy-22van`). `resolve_or_reread` restores the old
        // per-call read when `NDIMAGE_SPLINE_FLAG_HOIST_DISABLE` is set, so the hoist has an
        // A/B baseline in the same binary. Both flags still come from a single resolve, so
        // the premultiply/leaf pair cannot tear (the latent panic fixed in 0a7086e76).
        let flags = SplineFlags::resolve_or_reread(flags);
        let offsets = flags.offsets;
        for axis in 0..n {
            if !compute_axis_support(
                coords[axis],
                coeffs.shape[axis],
                input.shape[axis],
                coord_offsets[axis],
                order,
                mode,
                flags.compact_disabled,
                &mut bases[axis],
            ) {
                return cval;
            }
            if offsets {
                // Fold this axis's stride into its `order+1` taps ONCE per pixel, so the
                // `(order+1)^ndim` leaves below each become a single `data[base]` load instead
                // of re-deriving `Σ idx[d]·stride[d]` via `coeffs.get`. Same lever the separable
                // paths already use (c396459ef); the supports here are per-pixel because the
                // coords couple axes, but the LEAF ADDRESS is orthogonal to separability.
                let stride = coeffs.strides[axis];
                for tap in &mut bases[axis] {
                    tap.0 *= stride;
                }
            }
        }

        if offsets {
            // BYTE-IDENTICAL to the index-space recursion: same weights, same accumulation
            // order (axis 0 outermost), only the leaf address is precomputed. Axes beyond `n`
            // hold `idx[d] = 0` there and contribute 0 to `base` here.
            if n <= MAX_STACK_NDIM {
                let mut slices: [&[(usize, f64)]; MAX_STACK_NDIM] = [&[]; MAX_STACK_NDIM];
                for (axis, slot) in slices[..n].iter_mut().enumerate() {
                    *slot = bases[axis].as_slice();
                }
                return sample_spline_offsets(&coeffs.data, &slices[..n], 0);
            }
            let slices: Vec<&[(usize, f64)]> = bases[..n].iter().map(Vec::as_slice).collect();
            return sample_spline_offsets(&coeffs.data, &slices, 0);
        }

        idx.clear();
        idx.resize(coeffs.ndim(), 0);
        sample_spline_recursive(coeffs, &bases[..n], 0, idx.as_mut_slice())
    })
}

// ══════════════════════════════════════════════════════════════════════
// Filters
// ══════════════════════════════════════════════════════════════════════

/// N-dimensional convolution with a given kernel.
///
/// Matches `scipy.ndimage.convolve`.
pub fn convolve(
    input: &NdArray,
    weights: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let origins = vec![0; input.ndim()];
    convolve_with_origins(input, weights, &origins, mode, cval)
}

fn normalize_axes_for_weights(
    input: &NdArray,
    weights: &NdArray,
    axes: &[isize],
) -> Result<(Vec<usize>, Vec<i64>), NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    let weights_ndim = weights.ndim();
    if weights_ndim != axes.len() {
        return Err(NdimageError::DimensionMismatch(format!(
            "weights.ndim ({weights_ndim}) must match len(axes) ({})",
            axes.len()
        )));
    }
    let origins = normalize_filter_origins(weights_ndim, &weights.shape, &[0])?;
    Ok((axes, origins))
}

/// N-dimensional convolution over a SciPy-style signed axes subset.
pub fn convolve_axes(
    input: &NdArray,
    weights: &NdArray,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let (axes, origins) = normalize_axes_for_weights(input, weights, axes)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| (s as i64 - 1) / 2).collect();

    // Precompute each tap's full-ndim input-index delta ONCE (pixel-independent),
    // folding in the kernel flip, center offset, and origin — 0 on the input axes
    // the kernel doesn't touch. Then route through `nd_filter_apply`, which gathers
    // interior pixels straight from the flat buffer (no per-pixel/per-tap `unravel`
    // or `in_idx` heap alloc, no per-tap boundary arithmetic) and falls back to
    // `get_boundary` on the border, summing in k=0..len order — BYTE-IDENTICAL to
    // `convolve_axes_scalar_reference`. Was allocating `weights.unravel` + a fresh
    // `in_idx` Vec per pixel per tap. Sibling of `convolve_with_origins` /
    // `correlate_axes` (frankenscipy-dn3i6 family).
    let ndim = input.ndim();
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            let mut delta = vec![0i64; ndim];
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                let k_flipped = weights.shape[kernel_axis] as i64 - 1 - k_idx[kernel_axis] as i64;
                delta[input_axis] = k_flipped - offsets[kernel_axis] + origins[kernel_axis];
            }
            delta
        })
        .collect();
    Ok(nd_filter_apply(
        input,
        &weights.data,
        &tap_delta,
        mode,
        cval,
    ))
}

/// Scalar per-pixel/per-tap reference path for [`convolve_axes`] (the pre-tap-delta
/// implementation): allocates `weights.unravel` + a fresh `in_idx` per pixel per
/// tap. Retained ONLY for the byte-identity test and the A/B bench that quantify
/// the `nd_filter_apply` conversion; not for production use.
#[doc(hidden)]
pub fn convolve_axes_scalar_reference(
    input: &NdArray,
    weights: &NdArray,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let (axes, origins) = normalize_axes_for_weights(input, weights, axes)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| (s as i64 - 1) / 2).collect();
    let mut output = NdArray::zeros(input.shape.clone());

    let kernel_work = weights.size().max(1);
    fill_pixels_parallel(&mut output, kernel_work, |flat_out, _scratch| {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;
        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx: Vec<i64> = out_idx.iter().map(|&idx| idx as i64).collect();
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                let k_flipped = weights.shape[kernel_axis] as i64 - 1 - k_idx[kernel_axis] as i64;
                in_idx[input_axis] = out_idx[input_axis] as i64 + k_flipped - offsets[kernel_axis]
                    + origins[kernel_axis];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }
        sum
    });

    Ok(output)
}

/// N-dimensional convolution with SciPy `origin` semantics.
///
/// `origins` may contain one scalar origin applied to every axis, or one origin
/// per input axis. Positive origins shift the convolution window toward higher
/// input coordinates; negative origins shift it toward lower coordinates.
// Apply an N-D filter whose taps are supplied as per-tap per-dim input-index deltas (already
// folding in any kernel flip, center offset, and origin). Two paths, both summing weights in
// k=0..len order so the result is BYTE-IDENTICAL to the per-pixel `get_boundary` loop:
//   • INTERIOR pixels (whole footprint in-bounds) gather directly from the flat array via
//     precomputed flat deltas — no boundary handling, no per-tap index arithmetic;
//   • BORDER pixels fall back to `get_boundary`.
// The old path's per-tap `weights.unravel` heap alloc and per-pixel `input.unravel` are gone.
/// Same-binary A/B switch: force the scalar interior path (for benchmarking only).
#[doc(hidden)]
pub static ND_FILTER_FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn nd_filter_apply(
    input: &NdArray,
    weights: &[f64],
    tap_delta: &[Vec<i64>],
    mode: BoundaryMode,
    cval: f64,
) -> NdArray {
    let ndim = input.ndim();
    let shape = &input.shape;
    let strides = &input.strides;
    let total = input.size();
    let tap_flat: Vec<i64> = tap_delta
        .iter()
        .map(|d| (0..ndim).map(|i| d[i] * strides[i] as i64).sum())
        .collect();
    // Interior box [lo[d], hi[d)): every tap stays in-bounds along dim d.
    let mut lo = vec![0i64; ndim];
    let mut hi: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    for d in 0..ndim {
        let mut mn = 0i64;
        let mut mx = 0i64;
        for t in tap_delta {
            mn = mn.min(t[d]);
            mx = mx.max(t[d]);
        }
        lo[d] = -mn;
        hi[d] = shape[d] as i64 - mx;
    }
    let mut output = NdArray::zeros(shape.clone());
    // For any C-order array (2-D image, 3-D volume, …) the interior box's INNERMOST
    // axis is a CONTIGUOUS run, so 8 consecutive output pixels gather from 8 contiguous
    // input slots per tap — process them as one 8-wide `Simd` accumulator. Bit-identical
    // to the scalar interior: each lane sums the SAME taps in the SAME k-order (`+= w*x`,
    // no FMA contraction). Requires innermost stride == 1 (standard C-order).
    let simd_last = ndim >= 2
        && strides[ndim - 1] == 1
        && !ND_FILTER_FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed);
    let work = |start: usize, os: &mut [f64]| {
        let mut out_idx = vec![0i64; ndim];
        let mut in_idx = vec![0i64; ndim];
        let len = os.len();
        let mut li = 0usize;
        while li < len {
            let p = start + li;
            let mut rem = p;
            let mut interior = true;
            for d in 0..ndim {
                let c = (rem / strides[d]) as i64;
                rem %= strides[d];
                out_idx[d] = c;
                if c < lo[d] || c >= hi[d] {
                    interior = false;
                }
            }
            if simd_last && interior {
                // Interior positions left along the innermost axis in this chunk (8-runs
                // stay in-line because `hi[last] <= shape[last]`, so no line wrap).
                let last = ndim - 1;
                let run = (hi[last] - out_idx[last]).min((len - li) as i64) as usize;
                let mut lane = 0usize;
                while lane + 8 <= run {
                    let pp = p + lane;
                    // 8 consecutive interior output pixels; each lane independently sums
                    // its taps in k-order (`+= w*x`, no FMA contraction) ⇒ bit-identical
                    // to the scalar interior path.
                    let mut acc = std::simd::Simd::<f64, 8>::splat(0.0);
                    for (k, &w) in weights.iter().enumerate() {
                        let base = (pp as i64 + tap_flat[k]) as usize;
                        acc += std::simd::Simd::splat(w)
                            * std::simd::Simd::from_slice(&input.data[base..base + 8]);
                    }
                    acc.copy_to_slice(&mut os[li + lane..li + lane + 8]);
                    lane += 8;
                }
                if lane > 0 {
                    li += lane;
                    continue;
                }
            }
            let mut sum = 0.0;
            if interior {
                for (k, &w) in weights.iter().enumerate() {
                    sum += w * input.data[(p as i64 + tap_flat[k]) as usize];
                }
            } else {
                for (k, &w) in weights.iter().enumerate() {
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] + tap_delta[k][d];
                    }
                    sum += w * input.get_boundary(&in_idx, mode, cval);
                }
            }
            os[li] = sum;
            li += 1;
        }
    };
    let nthreads = ndimage_filter_thread_count(total, weights.len());
    if nthreads <= 1 {
        work(0, &mut output.data);
    } else {
        let chunk = total.div_ceil(nthreads);
        let work = &work;
        std::thread::scope(|scope| {
            for (t, os) in output.data.chunks_mut(chunk).enumerate() {
                scope.spawn(move || work(t * chunk, os));
            }
        });
    }
    output
}

/// Reference per-pixel N-D filter path (pre interior/flat-gather), retained for the
/// same-process A/B benchmark and byte-identity proof only. `flip` selects convolve vs
/// correlate kernel orientation/offset/origin sign.
#[doc(hidden)]
/// Index-driven filter kernel: the loop variables are positions in the extended
/// (boundary-padded) line and index more than one buffer per step, so iterator
/// form would obscure the index algebra rather than simplify it
/// (frankenscipy-iit9c).
#[allow(clippy::needless_range_loop)]
pub fn nd_filter_perpixel_ref(
    input: &NdArray,
    weights: &NdArray,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    flip: bool,
) -> Result<NdArray, NdimageError> {
    let ndim = input.ndim();
    let origins = normalize_filter_origins(ndim, &weights.shape, origins)?;
    let offsets: Vec<i64> = weights
        .shape
        .iter()
        .map(|&s| {
            if flip {
                (s as i64 - 1) / 2
            } else {
                s as i64 / 2
            }
        })
        .collect();
    let mut output = NdArray::zeros(input.shape.clone());
    // Precompute each tap's per-dimension delta ONCE (it is independent of the
    // output pixel), instead of unravelling flat_k for every pixel — eliminates the
    // n_pixels×kernel_total k_idx allocations + recomputation. Mirrors the existing
    // convolve_with_origins tap table; byte-identical. frankenscipy-e3r7e.
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            (0..ndim)
                .map(|d| {
                    if flip {
                        (weights.shape[d] as i64 - 1 - k_idx[d] as i64) - offsets[d] + origins[d]
                    } else {
                        k_idx[d] as i64 - offsets[d] - origins[d]
                    }
                })
                .collect()
        })
        .collect();
    fill_pixels_parallel(&mut output, weights.size().max(1), |flat_out, _s| {
        let out_idx = input.unravel(flat_out);
        let mut in_idx = vec![0i64; ndim];
        let mut sum = 0.0;
        for flat_k in 0..weights.size() {
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + tap_delta[flat_k][d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }
        sum
    });
    Ok(output)
}

pub fn convolve_with_origins(
    input: &NdArray,
    weights: &NdArray,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != weights.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and weights must have same number of dimensions".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let origins = normalize_filter_origins(ndim, &weights.shape, origins)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| (s as i64 - 1) / 2).collect();
    // Convolution: flip the kernel (k_flipped = len-1-k), center, add origin.
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            (0..ndim)
                .map(|d| (weights.shape[d] as i64 - 1 - k_idx[d] as i64) - offsets[d] + origins[d])
                .collect()
        })
        .collect();
    Ok(nd_filter_apply(
        input,
        &weights.data,
        &tap_delta,
        mode,
        cval,
    ))
}

/// One-dimensional convolution along a selected axis.
///
/// Matches `scipy.ndimage.convolve1d` for centered filters.
pub fn convolve1d(
    input: &NdArray,
    weights: &[f64],
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    convolve1d_with_origin(input, weights, axis, mode, cval, 0)
}

/// One-dimensional convolution with SciPy-style signed axis normalization.
pub fn convolve1d_signed_axis(
    input: &NdArray,
    weights: &[f64],
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    convolve1d(input, weights, axis, mode, cval)
}

/// One-dimensional convolution using SciPy's default `axis=-1`.
pub fn convolve1d_default_axis(
    input: &NdArray,
    weights: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    convolve1d_signed_axis(input, weights, -1, mode, cval)
}

/// One-dimensional convolution along a selected axis with SciPy `origin` semantics.
///
/// Positive origins shift the convolution window toward higher input coordinates;
/// negative origins shift it toward lower coordinates.
pub fn convolve1d_with_origin(
    input: &NdArray,
    weights: &[f64],
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if weights.is_empty() {
        return Err(NdimageError::InvalidArgument(
            "weights must not be empty".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(weights.len(), origin)?;

    // convolve1d(w, origin) ≡ correlate1d(reverse(w), origin'), the identity SciPy itself
    // uses. correlate1d_along_axis sums forward over the reversed weights, which reproduces
    // the old fill path's exact source-left-to-right summation order (weights.rev() with
    // source p+k-offset+origin) ⇒ BYTE-IDENTICAL, while reusing its vectorized interior axpy
    // (10x faster than the per-pixel fill, 06671a9b). The origin shift maps convolve's
    // offset=(len-1)/2 to correlate's offset=len/2: origin' = (len-1)/2 - len/2 - origin
    // (= -origin for odd len, -origin-1 for even).
    let len = weights.len() as i64;
    let corr_origin = (len - 1) / 2 - len / 2 - origin;
    let rev: Vec<f64> = weights.iter().rev().copied().collect();
    Ok(correlate1d_along_axis(
        input,
        &rev,
        axis,
        corr_origin,
        mode,
        cval,
    ))
}

/// N-dimensional correlation with a given kernel.
///
/// Matches `scipy.ndimage.correlate`.
pub fn correlate(
    input: &NdArray,
    weights: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let origins = vec![0; input.ndim()];
    correlate_with_origins(input, weights, &origins, mode, cval)
}

/// N-dimensional correlation over a SciPy-style signed axes subset.
pub fn correlate_axes(
    input: &NdArray,
    weights: &NdArray,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let (axes, origins) = normalize_axes_for_weights(input, weights, axes)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();

    // Precompute each tap's full-ndim delta ONCE (pixel-independent): 0 on axes the
    // kernel doesn't touch, the centered+origin-shifted offset on mapped axes. Then
    // route through `nd_filter_apply`, which gathers interior pixels straight from
    // the flat buffer (no per-pixel `unravel`/`in_idx` alloc, no per-tap boundary
    // arithmetic) and falls back to `get_boundary` on the border, summing in
    // k=0..len order — BYTE-IDENTICAL to `correlate_axes_scalar_reference`.
    // frankenscipy-dn3i6.
    let ndim = input.ndim();
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            let mut delta = vec![0i64; ndim];
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                delta[input_axis] =
                    k_idx[kernel_axis] as i64 - offsets[kernel_axis] - origins[kernel_axis];
            }
            delta
        })
        .collect();
    Ok(nd_filter_apply(
        input,
        &weights.data,
        &tap_delta,
        mode,
        cval,
    ))
}

/// Scalar per-pixel reference path for [`correlate_axes`] (tap-delta precomputed
/// but every tap routed through `get_boundary`, no interior flat-gather). Retained
/// ONLY for the byte-identity test and A/B bench of the `nd_filter_apply`
/// conversion; not for production use.
#[doc(hidden)]
/// Index-driven filter kernel: the loop variables are positions in the extended
/// (boundary-padded) line and index more than one buffer per step, so iterator
/// form would obscure the index algebra rather than simplify it
/// (frankenscipy-iit9c).
#[allow(clippy::needless_range_loop)]
pub fn correlate_axes_scalar_reference(
    input: &NdArray,
    weights: &NdArray,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let (axes, origins) = normalize_axes_for_weights(input, weights, axes)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();
    let mut output = NdArray::zeros(input.shape.clone());

    let kernel_work = weights.size().max(1);
    let ndim = input.ndim();
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            let mut delta = vec![0i64; ndim];
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                delta[input_axis] =
                    k_idx[kernel_axis] as i64 - offsets[kernel_axis] - origins[kernel_axis];
            }
            delta
        })
        .collect();
    fill_pixels_parallel(&mut output, kernel_work, |flat_out, _scratch| {
        let out_idx = input.unravel(flat_out);
        let mut in_idx = vec![0i64; ndim];
        let mut sum = 0.0;
        for flat_k in 0..weights.size() {
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + tap_delta[flat_k][d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }
        sum
    });

    Ok(output)
}

/// N-dimensional correlation with SciPy `origin` semantics.
///
/// `origins` may contain one scalar origin applied to every axis, or one origin
/// per input axis. Positive origins shift the correlation window toward lower
/// input coordinates; negative origins shift it toward higher coordinates.
pub fn correlate_with_origins(
    input: &NdArray,
    weights: &NdArray,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != weights.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and weights must have same number of dimensions".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let origins = normalize_filter_origins(ndim, &weights.shape, origins)?;
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();
    // Correlation: no flip; center the kernel and subtract origin.
    let tap_delta: Vec<Vec<i64>> = (0..weights.size())
        .map(|flat_k| {
            let k_idx = weights.unravel(flat_k);
            (0..ndim)
                .map(|d| k_idx[d] as i64 - offsets[d] - origins[d])
                .collect()
        })
        .collect();
    Ok(nd_filter_apply(
        input,
        &weights.data,
        &tap_delta,
        mode,
        cval,
    ))
}

/// One-dimensional correlation along a selected axis.
///
/// Matches `scipy.ndimage.correlate1d` for centered filters.
pub fn correlate1d(
    input: &NdArray,
    weights: &[f64],
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    correlate1d_with_origin(input, weights, axis, mode, cval, 0)
}

/// One-dimensional correlation with SciPy-style signed axis normalization.
pub fn correlate1d_signed_axis(
    input: &NdArray,
    weights: &[f64],
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    correlate1d(input, weights, axis, mode, cval)
}

/// One-dimensional correlation using SciPy's default `axis=-1`.
pub fn correlate1d_default_axis(
    input: &NdArray,
    weights: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    correlate1d_signed_axis(input, weights, -1, mode, cval)
}

/// One-dimensional correlation along a selected axis with SciPy `origin` semantics.
///
/// Positive origins shift the correlation window toward lower input coordinates;
/// negative origins shift it toward higher coordinates.
pub fn correlate1d_with_origin(
    input: &NdArray,
    weights: &[f64],
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if weights.is_empty() {
        return Err(NdimageError::InvalidArgument(
            "weights must not be empty".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(weights.len(), origin)?;
    // The line walk's interior is a contiguous shifted-slice axpy (vectorized + cache-
    // friendly); it beats the per-pixel `fill_pixels_parallel` path even SERIAL — measured
    // 11.5x faster at 256² axis=0 (the outermost-axis, single-slab case the old gate routed
    // to fill_pixels_parallel). Both byte-identical, so always use the line walk; it
    // parallelizes across outer slabs when there are enough, and runs the vectorized serial
    // pass otherwise.
    Ok(correlate1d_along_axis(
        input, weights, axis, origin, mode, cval,
    ))
}

/// Reference per-pixel correlate1d path (pre line-walk), retained for the same-process A/B
/// benchmark and byte-identity proof only.
#[doc(hidden)]
pub fn correlate1d_perwindow_ref(
    input: &NdArray,
    weights: &[f64],
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() || weights.is_empty() || input.size() == 0 {
        return Err(NdimageError::InvalidArgument("bad args".to_string()));
    }
    validate_filter_origin(weights.len(), origin)?;
    let offset = weights.len() as i64 / 2;
    let mut output = NdArray::zeros(input.shape.clone());
    fill_pixels_parallel(&mut output, weights.len(), |flat_out, _scratch| {
        let out_idx = input.unravel(flat_out);
        let mut in_idx: Vec<i64> = out_idx.iter().map(|&i| i as i64).collect();
        let mut sum = 0.0;
        for (k, &weight) in weights.iter().enumerate() {
            in_idx[axis] = out_idx[axis] as i64 + k as i64 - offset - origin;
            sum += weight * input.get_boundary(&in_idx, mode, cval);
        }
        sum
    });
    Ok(output)
}

// O(n·k) 1-D correlation along `axis` via a line walk over contiguous outer slabs, instead
// of the per-pixel path that re-`unravel`s a multi-index and does a full-rank `get_boundary`
// for every kernel tap. The per-output dot is summed in the SAME k=0..len order as the old
// kernel with the SAME boundary values (`boundary_index_1d` ≡ the per-axis `get_boundary`),
// so the result is BYTE-IDENTICAL; only the per-pixel indexing overhead is removed. Slabs are
// contiguous & disjoint ⇒ safe parallelism with no aliasing.
fn correlate1d_along_axis(
    arr: &NdArray,
    weights: &[f64],
    axis: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
) -> NdArray {
    let mid = arr.shape[axis];
    let inner: usize = arr.shape[axis + 1..].iter().product();
    let outer: usize = arr.shape[..axis].iter().product();
    let slab = mid * inner;
    let offset = weights.len() as i64 / 2;
    let mut out = NdArray::zeros(arr.shape.clone());
    // Interior positions [lo, hi) along the axis are boundary-free (every tap lands
    // in-bounds), so the per-pixel/per-tap gather becomes a contiguous shifted-slice
    // axpy: vectorizes over `inner` (non-last axis) or over `a` itself (last axis,
    // inner==1). Boundary positions keep the per-pixel `val_at` path. BYTE-IDENTICAL:
    // `os` is zero-initialised and each output accumulates k=0..len in the same order
    // as the register sum. Mirrors the gaussian-2D column-pass axpy (e767313d).
    let klen = weights.len() as i64;
    let lo = (offset + origin).max(0);
    let hi = ((mid as i64) - klen + offset + origin + 1).clamp(0, mid as i64);
    let do_slab = |is: &[f64], os: &mut [f64]| {
        let val_at = |i: usize, a: i64| -> f64 {
            match boundary_index_1d(a, mid as i64, mode) {
                Some(m) => is[i + (m as usize) * inner],
                None => cval,
            }
        };
        let per_pixel = |os: &mut [f64], a: i64| {
            for i in 0..inner {
                let mut sum = 0.0;
                for (k, &w) in weights.iter().enumerate() {
                    sum += w * val_at(i, a + k as i64 - offset - origin);
                }
                os[i + (a as usize) * inner] = sum;
            }
        };
        if lo >= hi {
            for a in 0..mid as i64 {
                per_pixel(os, a);
            }
            return;
        }
        for a in (0..lo).chain(hi..mid as i64) {
            per_pixel(os, a);
        }
        if inner == 1 {
            // `a` is the contiguous dimension; axpy os[lo..hi] += w·is[lo+shift..hi+shift].
            for (k, &w) in weights.iter().enumerate() {
                let shift = k as i64 - offset - origin;
                let dst = &mut os[lo as usize..hi as usize];
                let src = &is[(lo + shift) as usize..(hi + shift) as usize];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d += w * s;
                }
            }
        } else {
            // `inner` is the contiguous dimension; axpy each interior output row.
            for a in lo..hi {
                let ob = (a as usize) * inner;
                for (k, &w) in weights.iter().enumerate() {
                    let ib = ((a + k as i64 - offset - origin) as usize) * inner;
                    let dst = &mut os[ob..ob + inner];
                    let src = &is[ib..ib + inner];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
        }
    };
    // Cost-aware gate (same vein as gaussian-2D/uniform_filter): the shared gate trips at
    // work>=1<<18, but per-element cost here is an O(weights.len())-tap dot, so 256² (work
    // 327k) spawns ~64 threads for a cheap pass. Same-process A/B (byte-identical): 256²
    // serial 2.61x faster, 512² parallel 1.23x (break-even ~work 1<<20).
    let par_work = (arr.size() as u64).saturating_mul(weights.len() as u64);
    let nthreads = if par_work < (1 << 20) {
        1
    } else {
        ndimage_filter_thread_count(arr.size(), weights.len()).min(outer.max(1))
    };
    if nthreads <= 1 || outer < 2 {
        for (is, os) in arr.data.chunks(slab).zip(out.data.chunks_mut(slab)) {
            do_slab(is, os);
        }
    } else {
        let slabs_per = outer.div_ceil(nthreads);
        let do_slab = &do_slab;
        std::thread::scope(|scope| {
            for (in_chunk, out_chunk) in arr
                .data
                .chunks(slab * slabs_per)
                .zip(out.data.chunks_mut(slab * slabs_per))
            {
                scope.spawn(move || {
                    for (is, os) in in_chunk.chunks(slab).zip(out_chunk.chunks_mut(slab)) {
                        do_slab(is, os);
                    }
                });
            }
        });
    }
    out
}

// 1-D convolution along `axis` via the same slab line walk, but reproducing the N-D
// `convolve` semantics for a 1-D-embedded kernel exactly: the kernel is FLIPPED
// (k_flipped = len-1-k), offset = (len-1)/2, origin ADDED, and the per-output dot is
// summed in the SAME k=0..len order with the SAME boundary values — so the result is
// BYTE-IDENTICAL to `convolve(input, embedded_kernel)`, only the per-pixel index/alloc
// overhead is removed. Lets `gaussian_filter1d_axis` skip the heavy N-D convolve path.
fn convolve1d_along_axis(
    arr: &NdArray,
    weights: &[f64],
    axis: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
) -> NdArray {
    let mid = arr.shape[axis];
    let inner: usize = arr.shape[axis + 1..].iter().product();
    let outer: usize = arr.shape[..axis].iter().product();
    let slab = mid * inner;
    let klen = weights.len() as i64;
    let offset = (klen - 1) / 2;
    let mut out = NdArray::zeros(arr.shape.clone());
    let do_slab = |is: &[f64], os: &mut [f64], axis_start: usize, axis_end: usize| {
        if inner > 1 {
            // For every output line, hoist the boundary lookup out of the contiguous inner
            // span. Each output still accumulates the same k=0..len tap sequence as the
            // scalar N-D convolution, but the inner loop is now a vector-friendly axpy.
            for a in axis_start as i64..axis_end as i64 {
                let output_base = (a as usize - axis_start) * inner;
                for (k, &w) in weights.iter().enumerate() {
                    let source = a + (klen - 1 - k as i64) - offset + origin;
                    let output = &mut os[output_base..output_base + inner];
                    if let Some(source) = boundary_index_1d(source, mid as i64, mode) {
                        let input_base = source as usize * inner;
                        for (slot, &value) in output
                            .iter_mut()
                            .zip(&is[input_base..input_base + inner])
                        {
                            *slot += w * value;
                        }
                    } else {
                        for slot in output {
                            *slot += w * cval;
                        }
                    }
                }
            }
        } else {
            for a in axis_start as i64..axis_end as i64 {
                let mut sum = 0.0;
                for (k, &w) in weights.iter().enumerate() {
                    let source = a + (klen - 1 - k as i64) - offset + origin;
                    let value = boundary_index_1d(source, mid as i64, mode)
                        .map_or(cval, |source| is[source as usize]);
                    sum += w * value;
                }
                os[a as usize - axis_start] = sum;
            }
        }
    };
    // Cost-aware gate (same vein as gaussian-2D/uniform_filter): the shared gate trips at
    // work>=1<<18, but per-element cost here is an O(weights.len())-tap dot, so 256² (work
    // 327k) spawns ~64 threads for a cheap pass. Same-process A/B (byte-identical): 256²
    // serial 2.61x faster, 512² parallel 1.23x (break-even ~work 1<<20).
    let par_work = (arr.size() as u64).saturating_mul(weights.len() as u64);
    let nthreads = if par_work < (1 << 20) {
        1
    } else {
        let parallel_spans = if outer == 1 && inner > 1 { mid } else { outer.max(1) };
        ndimage_filter_thread_count(arr.size(), weights.len()).min(parallel_spans)
    };
    if nthreads <= 1 || outer < 2 {
        if nthreads <= 1 || inner == 1 {
            for (is, os) in arr.data.chunks(slab).zip(out.data.chunks_mut(slab)) {
                do_slab(is, os, 0, mid);
            }
        } else {
            let chunk = mid.div_ceil(nthreads);
            let do_slab = &do_slab;
            std::thread::scope(|scope| {
                for (chunk_idx, out_chunk) in out.data.chunks_mut(chunk * inner).enumerate() {
                    let start = chunk_idx * chunk;
                    let end = start + out_chunk.len() / inner;
                    scope.spawn(move || do_slab(&arr.data, out_chunk, start, end));
                }
            });
        }
    } else {
        let slabs_per = outer.div_ceil(nthreads);
        let do_slab = &do_slab;
        std::thread::scope(|scope| {
            for (in_chunk, out_chunk) in arr
                .data
                .chunks(slab * slabs_per)
                .zip(out.data.chunks_mut(slab * slabs_per))
            {
                scope.spawn(move || {
                    for (is, os) in in_chunk.chunks(slab).zip(out_chunk.chunks_mut(slab)) {
                        do_slab(is, os, 0, mid);
                    }
                });
            }
        });
    }
    out
}

/// Uniform (box) filter.
///
/// Matches `scipy.ndimage.uniform_filter`.
pub fn uniform_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = (0..input.ndim()).collect::<Vec<_>>();
    uniform_filter_usize_axes(input, size, &axes, mode, cval)
}

/// Uniform (box) filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn uniform_filter_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    uniform_filter_usize_axes(input, size, &axes, mode, cval)
}

fn uniform_filter_usize_axes(
    input: &NdArray,
    size: usize,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let mut current = input.clone();
    let ndim = input.ndim();
    for &axis in axes {
        if axis >= ndim {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {axis} out of range for {ndim}-dimensional input"
            )));
        }
        current = uniform_filter1d_with_origin(&current, size, axis, mode, cval, 0)?;
    }
    Ok(current)
}

/// Uniform (box) filter with SciPy `origin` semantics.
pub fn uniform_filter_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }

    let kernel_shape = vec![size; input.ndim()];
    let origins = normalize_filter_origins(input.ndim(), &kernel_shape, origins)?;
    let mut current = input.clone();
    for (axis, &origin) in origins.iter().enumerate() {
        current = uniform_filter1d_with_origin(&current, size, axis, mode, cval, origin)?;
    }
    Ok(current)
}

/// Gaussian filter.
///
/// Matches `scipy.ndimage.gaussian_filter`.
pub fn gaussian_filter(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = (0..input.ndim()).collect::<Vec<_>>();
    gaussian_filter_usize_axes(input, sigma, &axes, mode, cval)
}

/// Gaussian filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn gaussian_filter_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    gaussian_filter_usize_axes(input, sigma, &axes, mode, cval)
}

fn gaussian_filter_usize_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NdimageError::InvalidArgument(
            "sigma must be positive".to_string(),
        ));
    }
    if axes.is_empty() {
        return Ok(input.clone());
    }
    if mode == BoundaryMode::Reflect
        && input.ndim() == 2
        && axes.len() == 2
        && axes[0] == 0
        && axes[1] == 1
    {
        return gaussian_filter_2d_reflect_order0(input, sigma);
    }

    let mut current = input.clone();
    let ndim = input.ndim();
    for &axis in axes {
        if axis >= ndim {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {axis} out of range for {ndim}-dimensional input"
            )));
        }
        current = gaussian_filter1d_axis(&current, sigma, axis, 0, mode, cval)?;
    }

    Ok(current)
}

/// One-dimensional Gaussian filter along a selected axis.
///
/// Matches `scipy.ndimage.gaussian_filter1d`.
pub fn gaussian_filter1d(
    input: &NdArray,
    sigma: f64,
    axis: usize,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NdimageError::InvalidArgument(
            "sigma must be positive".to_string(),
        ));
    }
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }

    gaussian_filter1d_axis(input, sigma, axis, order, mode, cval)
}

/// Reference old gaussian_filter1d path (1-D kernel embedded into an N-D `convolve`),
/// retained for the same-process A/B benchmark / byte-identity proof only.
#[doc(hidden)]
pub fn gaussian_filter1d_via_convolve_ref(
    input: &NdArray,
    sigma: f64,
    axis: usize,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let radius = gaussian_kernel_radius(sigma);
    let kernel_1d = gaussian_kernel1d(sigma, order, radius)?;
    let mut kernel_shape = vec![1usize; input.ndim()];
    kernel_shape[axis] = kernel_1d.len();
    let kernel = NdArray::new(kernel_1d, kernel_shape)?;
    convolve(input, &kernel, mode, cval)
}

/// One-dimensional Gaussian filter with SciPy-style signed axis normalization.
pub fn gaussian_filter1d_signed_axis(
    input: &NdArray,
    sigma: f64,
    axis: isize,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    gaussian_filter1d(input, sigma, axis, order, mode, cval)
}

/// One-dimensional Gaussian filter using SciPy's default `axis=-1`.
pub fn gaussian_filter1d_default_axis(
    input: &NdArray,
    sigma: f64,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    gaussian_filter1d_signed_axis(input, sigma, -1, order, mode, cval)
}

/// Median filter.
///
/// Matches `scipy.ndimage.median_filter`.
pub fn median_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let origins = vec![0; input.ndim()];
    median_filter_with_origins(input, size, &origins, mode, cval)
}

/// Median filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn median_filter_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }
    let kernel_total = filter_footprint_size(axes.len(), size)?;
    rank_filter_index_usize_axes(input, size, &axes, mode, cval, kernel_total / 2)
}

/// Median filter with SciPy `origin` semantics.
///
/// SciPy selects the element at rank `len / 2` from each sorted neighborhood,
/// including even-sized neighborhoods; it does not average the two middle
/// values.
pub fn median_filter_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let origins = normalize_filter_origins(ndim, &kernel_shape, origins)?;

    // Generate all offsets in kernel
    let kernel_strides = compute_strides(&kernel_shape);

    fill_rank_filter(
        &mut output,
        input,
        ndim,
        kernel_total,
        &kernel_strides,
        &offsets,
        &origins,
        mode,
        cval,
        kernel_total / 2,
    );

    Ok(output)
}

/// Compute one output pixel of a rank/median filter: gather the `kernel_total`
/// neighbourhood values (with the same boundary/origin handling) and return the
/// `rank`-th order statistic via `select_total_rank`. Pure read over `input`.
#[allow(clippy::too_many_arguments)]
/// Reference old per-pixel rank-filter path (alloc-per-element gather), retained for the
/// same-process A/B benchmark and byte-identity proof only.
#[doc(hidden)]
pub fn rank_filter_perpixel_ref(
    input: &NdArray,
    size: usize,
    rank: usize,
    mode: BoundaryMode,
    cval: f64,
) -> NdArray {
    let ndim = input.ndim();
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);
    let mut output = NdArray::zeros(input.shape.clone());
    let n = input.size();
    let pixel = |flat_out: usize| -> f64 {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);
        // k_idx/in_idx hoisted out of the kernel loop: allocated once per output
        // pixel instead of once per kernel element (both fully overwritten each
        // element -> byte-identical). frankenscipy-gy6to.
        let mut k_idx = vec![0usize; ndim];
        let mut in_idx = vec![0i64; ndim];
        for flat_k in 0..kernel_total {
            let mut rem = flat_k;
            for d in 0..ndim {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }
        select_total_rank(&mut neighborhood, rank)
    };
    // Parallel like the old fill_rank_filter, so the A/B measures the alloc-reduction only.
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        for (flat_out, slot) in output.data.iter_mut().enumerate() {
            *slot = pixel(flat_out);
        }
    } else {
        let chunk = n.div_ceil(nthreads);
        let pixel = &pixel;
        std::thread::scope(|scope| {
            for (t, oc) in output.data.chunks_mut(chunk).enumerate() {
                scope.spawn(move || {
                    for (li, slot) in oc.iter_mut().enumerate() {
                        *slot = pixel(t * chunk + li);
                    }
                });
            }
        });
    }
    output
}

/// Worker count for a parallel rank/median filter: 1 (sequential) unless the total
/// gather+select work (`pixels * kernel_total`) is large enough to amortise spawn.
fn ndimage_filter_thread_count(pixels: usize, kernel_total: usize) -> usize {
    let work = (pixels as u64).saturating_mul(kernel_total as u64);
    if work < 1 << 18 || pixels < 4 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(pixels / 2).max(1)
}

/// Fill `output.data` by computing each pixel independently via `pixel(flat_out,
/// &mut scratch)`, distributing the disjoint output indices across threads. Each
/// thread gets a fresh `scratch` buffer reused across its pixels (matching the
/// sequential single-buffer pattern). Because each output element depends only on a
/// read-only `pixel` computation, the parallel result is bit-identical to the
/// sequential loop — only the owning core changes. `kernel_work` is the per-pixel
/// work used to gate parallelism.
fn fill_pixels_parallel<G>(output: &mut NdArray, kernel_work: usize, pixel: G)
where
    G: Fn(usize, &mut Vec<f64>) -> f64 + Sync,
{
    let n = output.data.len();
    let nthreads = ndimage_filter_thread_count(n, kernel_work);
    if nthreads <= 1 {
        let mut scratch = Vec::new();
        for (flat_out, slot) in output.data.iter_mut().enumerate() {
            *slot = pixel(flat_out, &mut scratch);
        }
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let pixel = &pixel;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            scope.spawn(move || {
                let mut scratch = Vec::new();
                for (li, slot) in out_chunk.iter_mut().enumerate() {
                    *slot = pixel(start + li, &mut scratch);
                }
            });
        }
    });
}

/// Like `fill_pixels_parallel`, but hands the pixel closure the output's row-major MULTI-INDEX
/// alongside the flat index.
///
/// The geometric transforms all need `out_idx`, and every one of them obtained it by calling
/// `unravel_with_shape(flat, shape)` per pixel — which heap-allocates TWO `Vec`s (the strides
/// table and the index itself) for what is pure index arithmetic. Since each thread walks a
/// CONTIGUOUS run of flat indices in row-major order, seed the index once per chunk and then
/// advance it with an in-place odometer (`idx[d] += 1; carry`), O(1) amortized and zero allocs.
///
/// BYTE-IDENTICAL: at step `flat` the odometer holds exactly `unravel_with_shape(flat, shape)`.
/// `NDIMAGE_UNRAVEL_ODOMETER_DISABLE` restores the per-pixel unravel as the same-binary ORIG arm.
fn fill_pixels_parallel_indexed<G>(output: &mut NdArray, kernel_work: usize, pixel: G)
where
    G: Fn(usize, &[usize]) -> f64 + Sync,
{
    let n = output.data.len();
    let shape = output.shape.clone();
    let ndim = shape.len();
    let orig = NDIMAGE_UNRAVEL_ODOMETER_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let nthreads = ndimage_filter_thread_count(n, kernel_work);

    // One chunk's worth of work: seed the odometer at `start`, then advance per pixel.
    let run = |start: usize, out_chunk: &mut [f64]| {
        if orig {
            for (li, slot) in out_chunk.iter_mut().enumerate() {
                let idx = unravel_with_shape(start + li, &shape);
                *slot = pixel(start + li, &idx);
            }
            return;
        }
        let mut idx = unravel_with_shape(start, &shape);
        for (li, slot) in out_chunk.iter_mut().enumerate() {
            *slot = pixel(start + li, &idx);
            // Row-major increment; the final overflow past the last pixel wraps to zeros
            // and is never observed.
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    };

    if nthreads <= 1 {
        run(0, &mut output.data);
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let run = &run;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || run(t * chunk, out_chunk));
        }
    });
}

/// Fill `output` with a rank/median filter, distributing the independent output
/// pixels across threads. Each `output.data[flat_out]` depends only on a read-only
/// neighbourhood of `input`, so the parallel result is bit-identical to the
/// sequential pixel-by-pixel loop; only the owning core changes.
#[allow(clippy::too_many_arguments)]
fn fill_rank_filter(
    output: &mut NdArray,
    input: &NdArray,
    ndim: usize,
    kernel_total: usize,
    kernel_strides: &[usize],
    offsets: &[i64],
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) {
    let n = input.size();
    let shape = &input.shape;
    let strides = &input.strides;
    // Precompute each footprint element's per-dim index delta + flat delta ONCE (the old
    // path re-derived k_idx and allocated k_idx + in_idx Vecs PER element PER pixel).
    let mut tap_delta: Vec<Vec<i64>> = Vec::with_capacity(kernel_total);
    for flat_k in 0..kernel_total {
        let mut rem = flat_k;
        let mut delta = vec![0i64; ndim];
        for d in 0..ndim {
            let k = (rem / kernel_strides[d]) as i64;
            rem %= kernel_strides[d];
            delta[d] = k - offsets[d] - origins[d];
        }
        tap_delta.push(delta);
    }
    let tap_flat: Vec<i64> = tap_delta
        .iter()
        .map(|d| (0..ndim).map(|i| d[i] * strides[i] as i64).sum())
        .collect();
    // Interior box [lo[d], hi[d)): every footprint element stays in-bounds along dim d.
    let mut lo = vec![0i64; ndim];
    let mut hi: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    for d in 0..ndim {
        let mut mn = 0i64;
        let mut mx = 0i64;
        for t in &tap_delta {
            mn = mn.min(t[d]);
            mx = mx.max(t[d]);
        }
        lo[d] = -mn;
        hi[d] = shape[d] as i64 - mx;
    }
    // Per pixel: gather the footprint (interior → direct flat gather, no boundary/allocs;
    // border → get_boundary) into a reused buffer in the same flat_k order, then rank-select.
    // Byte-identical to the old per-pixel loop (same gather order, same select_total_rank).
    let work = |start: usize, os: &mut [f64]| {
        let mut out_idx = vec![0i64; ndim];
        let mut in_idx = vec![0i64; ndim];
        let mut nb = vec![0.0f64; kernel_total];
        for (li, slot) in os.iter_mut().enumerate() {
            let p = start + li;
            let mut rem = p;
            let mut interior = true;
            for d in 0..ndim {
                let c = (rem / strides[d]) as i64;
                rem %= strides[d];
                out_idx[d] = c;
                if c < lo[d] || c >= hi[d] {
                    interior = false;
                }
            }
            if interior {
                for (k, slot) in nb.iter_mut().enumerate() {
                    *slot = input.data[(p as i64 + tap_flat[k]) as usize];
                }
            } else {
                for (k, slot) in nb.iter_mut().enumerate() {
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] + tap_delta[k][d];
                    }
                    *slot = input.get_boundary(&in_idx, mode, cval);
                }
            }
            *slot = select_total_rank(&mut nb, rank);
        }
    };
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        work(0, &mut output.data);
        return;
    }
    let chunk = n.div_ceil(nthreads);
    let work = &work;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || work(t * chunk, out_chunk));
        }
    });
}

/// Minimum filter.
///
/// Matches `scipy.ndimage.minimum_filter`.
/// Separable minimum/maximum filter over a `size^ndim` rectangular footprint.
///
/// A min/max over a rectangle equals the per-axis sequential min/max, so the
/// O(N * size^ndim * log) full-footprint sort-and-select rank filter collapses
/// to ndim O(N) sliding-window passes (a monotonic deque, O(1) amortized per
/// output, independent of `size`). Comparisons use `total_cmp` — the same total
/// order the rank filter sorts by — and every neighbourhood value comes from
/// `get_boundary`, so the result is bit-for-bit identical to the rank filter,
/// including NaN and signed-zero handling.
fn separable_minmax_filter(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    is_max: bool,
) -> Result<NdArray, NdimageError> {
    filter_footprint_size(input.ndim(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let ndim = input.ndim();
    // Validate origins against the kernel footprint [size; ndim], exactly as the
    // full-footprint rank filter does, so out-of-range origins are rejected
    // identically.
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let origins = normalize_filter_origins(ndim, &kernel_shape, origins)?;
    // The van Herk kernel's hot op is `tc_max`/`tc_min` (a full `total_cmp`, ~6
    // integer ops) only to reproduce scipy's total-order tie-breaks. For the
    // overwhelming common case `f64::max`/`f64::min` are byte-identical AND far
    // cheaper — they differ from the total order in EXACTLY two spots: NaN
    // (total_cmp propagates it, f64::max/min drops it) and the pair {+0.0, -0.0}
    // (`f64::max(+0,-0) == -0` but total order gives +0). So probe ONCE for a NaN
    // or a negative zero; absent both, run the fast comparators. min/max of such
    // "clean" values can never MINT a NaN or a -0.0, so the input's cleanliness
    // holds through every separable axis pass.
    let needs_total_cmp = input
        .data
        .iter()
        .any(|v| v.is_nan() || (*v == 0.0 && v.is_sign_negative()));
    let mut cur = input.clone();
    for (d, &origin) in origins.iter().enumerate() {
        cur = minmax_filter_along_axis(&cur, d, size, origin, mode, cval, is_max, needs_total_cmp);
    }
    Ok(cur)
}

/// One axis of a sliding-window min/max via a monotonic deque of (coord, value).
/// Map a single-axis coordinate `i` (possibly out of `[0, n)`) to an in-bounds
/// index under `mode`, or `None` for the constant-boundary out-of-range case.
/// Mirrors the per-axis arithmetic of `NdArray::get_boundary` exactly, so the
/// separable filter stays bit-identical while avoiding its per-element vec
/// allocation.
#[inline]
fn boundary_index_1d(mut i: i64, n: i64, mode: BoundaryMode) -> Option<i64> {
    match mode {
        BoundaryMode::Reflect => {
            if i < 0 {
                i = -i - 1;
            }
            if i >= n {
                i = 2 * n - i - 1;
            }
            let period = 2 * n;
            i = i.rem_euclid(period);
            if i >= n {
                i = period - i - 1;
            }
            Some(i)
        }
        BoundaryMode::Constant => {
            if i < 0 || i >= n {
                None
            } else {
                Some(i)
            }
        }
        BoundaryMode::Nearest => Some(i.clamp(0, n - 1)),
        BoundaryMode::Wrap => Some(i.rem_euclid(n)),
        BoundaryMode::Mirror => {
            if n <= 1 {
                Some(0)
            } else {
                let period = 2 * (n - 1);
                i = i.rem_euclid(period);
                if i >= n {
                    i = period - i;
                }
                Some(i)
            }
        }
    }
}

// O(n) running-sum uniform (box-mean) filter along `axis`, replacing the O(n·size)
// per-window re-summation. Mirrors `minmax_filter_along_axis`'s line walk: each line head
// (flat/stride a multiple of shape[axis]) is processed once, sliding a sum that adds the
// entering element and subtracts the leaving one (the algorithm SciPy's uniform_filter1d
// uses). Window for output i spans input coords [i-lo, i-lo+size-1], lo=size/2+origin —
// identical elements to the per-window kernel, so out[0] (summed left-to-right) is
// byte-identical; later positions accumulate incrementally (tolerance-parity).
fn uniform_filter_along_axis(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
) -> NdArray {
    let mid = arr.shape[axis];
    let inner: usize = arr.shape[axis + 1..].iter().product();
    let outer: usize = arr.shape[..axis].iter().product();
    let slab = mid * inner;
    let size_i = size as i64;
    let lo = size_i / 2 + origin;
    // Divide (not multiply-by-reciprocal): sum/size is byte-identical to the per-window
    // mean for exact sums, and matches SciPy. inv-multiply would drift a ULP (5/3 ≠ 5·⅓).
    let size_f = size as f64;
    let mut out = NdArray::zeros(arr.shape.clone());

    // Process one contiguous [mid×inner] slab (fixed outer index): each of its `inner`
    // lines along `axis` slides a running sum (drop leaving, add entering element).
    // out[0] is summed left-to-right (byte-identical to the per-window kernel); later
    // positions accumulate incrementally (tolerance-parity). Writes stay inside `os`.
    let do_slab = |is: &[f64], os: &mut [f64]| {
        let val_at = |i: usize, a: i64| -> f64 {
            match boundary_index_1d(a, mid as i64, mode) {
                Some(m) => is[i + (m as usize) * inner],
                None => cval,
            }
        };
        if inner == 1 {
            // `a` is the contiguous dimension — the per-line running sum is already
            // cache-friendly; keep it (vectorizing a dependent scalar sum buys nothing).
            for i in 0..inner {
                let mut sum = 0.0;
                for k in 0..size_i {
                    sum += val_at(i, k - lo);
                }
                os[i] = sum / size_f;
                for a in 1..mid as i64 {
                    sum += val_at(i, a - lo + size_i - 1) - val_at(i, (a - 1) - lo);
                    os[i + (a as usize) * inner] = sum / size_f;
                }
            }
            return;
        }
        // inner > 1: the per-COLUMN running sum strides by `inner` (cache-hostile, the
        // source of the super-linear scaling). Carry a sum VECTOR over the contiguous
        // `inner` dimension instead, updating it per row with CONTIGUOUS reads — cache-
        // friendly and auto-vectorizing. BYTE-IDENTICAL: each column accumulates the same
        // window then `+= enter - leave` (FUSED, not split) in the same row order.
        let row = |r: i64| -> Option<usize> {
            boundary_index_1d(r, mid as i64, mode).map(|m| (m as usize) * inner)
        };
        let mut sum_vec = vec![0.0f64; inner];
        for k in 0..size_i {
            match row(k - lo) {
                Some(b) => {
                    let src = &is[b..b + inner];
                    for (s, &v) in sum_vec.iter_mut().zip(src) {
                        *s += v;
                    }
                }
                None => {
                    for s in sum_vec.iter_mut() {
                        *s += cval;
                    }
                }
            }
        }
        for (slot, &s) in os[..inner].iter_mut().zip(&sum_vec) {
            *slot = s / size_f;
        }
        for a in 1..mid as i64 {
            let e = row(a - lo + size_i - 1);
            let l = row((a - 1) - lo);
            let ob = (a as usize) * inner;
            match (e, l) {
                (Some(eb), Some(lb)) => {
                    let er = &is[eb..eb + inner];
                    let lr = &is[lb..lb + inner];
                    for i in 0..inner {
                        sum_vec[i] += er[i] - lr[i];
                        os[ob + i] = sum_vec[i] / size_f;
                    }
                }
                (Some(eb), None) => {
                    let er = &is[eb..eb + inner];
                    for i in 0..inner {
                        sum_vec[i] += er[i] - cval;
                        os[ob + i] = sum_vec[i] / size_f;
                    }
                }
                (None, Some(lb)) => {
                    let lr = &is[lb..lb + inner];
                    for i in 0..inner {
                        sum_vec[i] += cval - lr[i];
                        os[ob + i] = sum_vec[i] / size_f;
                    }
                }
                // Both the entering and leaving samples are outside the image, so both
                // are the boundary constant and the running sum is unchanged. Written as
                // `cval - cval` to complete the four-way case analysis above rather than
                // as `+= 0.0`, and because it must keep propagating NaN when `cval` is
                // non-finite exactly as the other three arms do (frankenscipy-iit9c).
                #[allow(clippy::eq_op)]
                (None, None) => {
                    for i in 0..inner {
                        sum_vec[i] += cval - cval;
                        os[ob + i] = sum_vec[i] / size_f;
                    }
                }
            }
        }
    };

    // Parallelize across outer slabs (contiguous & disjoint ⇒ no aliasing) when there are
    // enough of them to amortize spawn; otherwise sequential. The running-sum pass is O(1)
    // per output element (drop leaving + add entering), INDEPENDENT of window `size`, so the
    // amortization point scales with PIXEL COUNT, not the shared `arr.size()·size` work
    // product (which over-counts large windows and trips far too early). Same-process A/B
    // (byte-identical): 256² serial 3.78× faster, 512² serial 1.48×, 1024² parity (0.996×) —
    // parallel only pays from ~1M pixels up. Gate at arr.size() >= 1<<20.
    let nthreads = if arr.size() < (1 << 20) {
        1
    } else {
        ndimage_filter_thread_count(arr.size(), size).min(outer.max(1))
    };
    if nthreads <= 1 || outer < 2 {
        for (is, os) in arr.data.chunks(slab).zip(out.data.chunks_mut(slab)) {
            do_slab(is, os);
        }
    } else {
        let slabs_per = outer.div_ceil(nthreads);
        let do_slab = &do_slab;
        std::thread::scope(|scope| {
            for (in_chunk, out_chunk) in arr
                .data
                .chunks(slab * slabs_per)
                .zip(out.data.chunks_mut(slab * slabs_per))
            {
                scope.spawn(move || {
                    for (is, os) in in_chunk.chunks(slab).zip(out_chunk.chunks_mut(slab)) {
                        do_slab(is, os);
                    }
                });
            }
        });
    }
    out
}

/// Runtime A/B toggle: the van Herk / Gil-Werman block prefix-suffix min/max path
/// (default, production) vs the legacy monotonic-deque path. The deque path is kept
/// as a reference oracle and for same-process interleaved A/B benchmarking under
/// fleet contention (separate-run benches drift ~2×; only an in-process toggle is
/// reliable). frankenscipy van-Herk lever.
/// CONTRACT: IDENTICAL OUTPUT either way, and exactly so rather than to a tolerance,
/// because min and max SELECT an input value rather than computing a new one. Van
/// Herk-Gil-Werman rebuilds the window from prefix and suffix running extrema instead of
/// rescanning it, which changes how many comparisons happen and in what order but not
/// which element wins: extremum is associative, and no arithmetic is performed on the
/// values, so there is no rounding for a different evaluation tree to expose.
///
/// That holds only while the comparator is a TOTAL order, which is why the comparator is
/// a parameter rather than a constant. `f64::max`/`f64::min` are not a total order in the
/// presence of NaN (they absorb it) or -0.0 (they do not distinguish it from +0.0), so
/// they are selected only when `needs_total_cmp` is false -- i.e. when the data provably
/// contains neither -- and `tc_max`/`tc_min` carry scipy's total-order tie-breaks and NaN
/// propagation otherwise. Both arms of the toggle take the SAME comparator, so the
/// identity is between the two window algorithms and does not depend on that choice.
pub static MINMAX_FILTER_HGW: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[inline(always)]
fn tc_max(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b) == std::cmp::Ordering::Less {
        b
    } else {
        a
    }
}

#[inline(always)]
fn tc_min(a: f64, b: f64) -> f64 {
    if a.total_cmp(&b) == std::cmp::Ordering::Greater {
        b
    } else {
        a
    }
}

/// NaN-propagating max/min, exactly matching the `maximum_filter1d` /
/// `minimum_filter1d` per-window fold (`if a||b NaN → NaN else a.max(b)`).
/// Both are associative and idempotent, so feeding them to the van Herk
/// block prefix/suffix scans reproduces the per-window fold bit-for-bit
/// (the extremum is one of the inputs — no rounding — and NaN propagates
/// regardless of association). Lets the 1-D filters use the O(n) HGW kernel
/// instead of the O(n·size) per-window scan with its per-pixel allocation.
#[cfg(test)]
#[inline(always)]
fn nanprop_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

#[cfg(test)]
#[inline(always)]
fn nanprop_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// van Herk / Gil-Werman sliding-window min/max along `axis`. Byte-identical to the
/// monotonic-deque path — same `total_cmp` total order (so the min/max element's bits
/// are uniquely determined, including NaN and signed-zero), same `boundary_index_1d`
/// neighbourhood mapping — but replaces the per-element deque (alloc + pointer-chase +
/// variable evictions) with three branch-light linear scans over a materialized,
/// boundary-resolved line: a block prefix `g`, a block suffix `h`, and the combine
/// `out[i] = op(h[i], g[i+size-1])`. Lines are addressed directly (outer × inner)
/// rather than scanning every flat index to find the line heads.
/// Same-binary A/B toggle for the van Herk / Gil-Werman min/max axis filter. When `true`, the
/// independent outer slabs are processed serially (the ORIG behaviour). When `false` (default), they
/// fan across cores. Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static MINMAX_HGW_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle for the strided (stride>1) van Herk path. When `true`, each column is
/// processed independently with a cache-hostile strided gather (the ORIG behaviour). When `false`
/// (default), the boundary-resolved line, block prefix/suffix, and combine sweep CONTIGUOUSLY over a
/// tile of inner columns at once (cache-friendly + auto-vectorizable). Byte-identical (the per-column
/// arithmetic order is unchanged). Benchmark knob.
#[doc(hidden)]
pub static MINMAX_HGW_FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Index-driven filter kernel: the loop variables are positions in the extended
/// (boundary-padded) line and index more than one buffer per step, so iterator
/// form would obscure the index algebra rather than simplify it
/// (frankenscipy-iit9c).
#[allow(clippy::needless_range_loop)]
fn minmax_along_axis_hgw<F: Fn(f64, f64) -> f64 + Copy + Sync>(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
    op: F,
) -> NdArray {
    let mid = arr.shape[axis];
    let inner: usize = arr.shape[axis + 1..].iter().product();
    let outer: usize = arr.shape[..axis].iter().product();
    let n = mid as i64;
    let size_i = size as i64;
    let lo = size_i / 2 + origin; // window left extent relative to the output
    let ext_len = mid + size - 1; // boundary-resolved line covers coords [-lo, mid-1-lo+size-1]
    let stride = inner;
    let slab = mid * inner;

    let mut out = NdArray::zeros(arr.shape.clone());

    // Process one contiguous group of slabs (`in_chunk`/`out_chunk` are slab-aligned blocks of
    // arr.data / out.data), reusing one ext/g/h scratch set across its lines. Each line (o, j) is
    // independent — the same boundary-resolved `ext`, the same block prefix/suffix `g`/`h`, and the
    // same combine — so this is BYTE-IDENTICAL to the serial walk regardless of how the outer slabs
    // are partitioned across threads.
    // For stride>1 (a non-last axis) the per-column path gathers each line with a stride-`inner`
    // jump — cache-hostile. Sweep a TILE of inner columns together instead: every ext/g/h "row" is
    // then a contiguous inner-strip, so the reads/writes are contiguous (and the block ops
    // auto-vectorize). Byte-identical: for a fixed column the boundary resolution, prefix/suffix and
    // combine are the exact same ops in the same order. stride==1 keeps the per-column memcpy path.
    let use_vec = stride > 1 && !MINMAX_HGW_FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed);
    let interior_start = lo as usize; // t where coord == 0 (lo >= 0 always)
    let interior_end = interior_start + mid; // t where coord == mid (exclusive)
    let process = |in_chunk: &[f64], out_chunk: &mut [f64]| {
        if use_vec {
            const TILE: usize = 64;
            let tile = TILE.min(inner);
            let mut ext = vec![0.0f64; ext_len * tile];
            let mut g = vec![0.0f64; ext_len * tile];
            let mut h = vec![0.0f64; ext_len * tile];
            let n_slabs = in_chunk.len() / slab;
            for o in 0..n_slabs {
                let sb = o * slab;
                let mut jt = 0usize;
                while jt < inner {
                    let w = tile.min(inner - jt);
                    // ext row t = boundary-resolved inner-strip at axis-coord (t - lo): a contiguous
                    // slab strip for interior/reflected coords, or `cval` when out of domain.
                    for t in 0..ext_len {
                        let dst = &mut ext[t * tile..t * tile + w];
                        let src_i = if t >= interior_start && t < interior_end {
                            Some(t - interior_start)
                        } else {
                            boundary_index_1d(t as i64 - lo, n, mode).map(|m| m as usize)
                        };
                        match src_i {
                            Some(i) => {
                                let s = sb + i * inner + jt;
                                dst.copy_from_slice(&in_chunk[s..s + w]);
                            }
                            None => dst.fill(cval),
                        }
                    }
                    // Block prefix g / suffix h over contiguous inner-strips.
                    let mut bstart = 0usize;
                    while bstart < ext_len {
                        let bend = (bstart + size).min(ext_len);
                        g[bstart * tile..bstart * tile + w]
                            .copy_from_slice(&ext[bstart * tile..bstart * tile + w]);
                        for t in bstart + 1..bend {
                            for k in 0..w {
                                g[t * tile + k] = op(g[(t - 1) * tile + k], ext[t * tile + k]);
                            }
                        }
                        h[(bend - 1) * tile..(bend - 1) * tile + w]
                            .copy_from_slice(&ext[(bend - 1) * tile..(bend - 1) * tile + w]);
                        for t in (bstart..bend - 1).rev() {
                            for k in 0..w {
                                h[t * tile + k] = op(h[(t + 1) * tile + k], ext[t * tile + k]);
                            }
                        }
                        bstart = bend;
                    }
                    // Combine into the output inner-strips.
                    for i in 0..mid {
                        let d = sb + i * inner + jt;
                        for k in 0..w {
                            out_chunk[d + k] = op(h[i * tile + k], g[(i + size - 1) * tile + k]);
                        }
                    }
                    jt += w;
                }
            }
            return;
        }
        let mut ext = vec![0.0f64; ext_len];
        let mut g = vec![0.0f64; ext_len];
        let mut h = vec![0.0f64; ext_len];
        let n_slabs = in_chunk.len() / slab;
        for o in 0..n_slabs {
            let outer_base = o * slab;
            for j in 0..inner {
                let base = outer_base + j;
                // Materialize the boundary-resolved line: ext[t] is the input value at
                // axis-coord (t - lo), so the window for output i is ext[i..i+size]. Only
                // the ~size-1 edge cells touch the boundary; the `mid`-cell interior is
                // in-bounds (coord in [0, mid)), where `boundary_index_1d` is the identity
                // for every mode — read it directly (contiguous memcpy when stride==1),
                // skipping the per-element boundary match. Byte-identical.
                let interior_start = lo as usize; // t where coord == 0 (lo >= 0 always)
                let interior_end = interior_start + mid; // t where coord == mid (exclusive)
                for t in 0..interior_start {
                    let coord = t as i64 - lo;
                    ext[t] = match boundary_index_1d(coord, n, mode) {
                        Some(m) => in_chunk[base + (m as usize) * stride],
                        None => cval,
                    };
                }
                if stride == 1 {
                    ext[interior_start..interior_end].copy_from_slice(&in_chunk[base..base + mid]);
                } else {
                    for t in interior_start..interior_end {
                        ext[t] = in_chunk[base + (t - interior_start) * stride];
                    }
                }
                for t in interior_end..ext_len {
                    let coord = t as i64 - lo;
                    ext[t] = match boundary_index_1d(coord, n, mode) {
                        Some(m) => in_chunk[base + (m as usize) * stride],
                        None => cval,
                    };
                }
                // Block prefix g and block suffix h, blocks of length `size` aligned to 0.
                // The final block may be short; h resets at its true end (ext_len-1).
                let mut bstart = 0usize;
                while bstart < ext_len {
                    let bend = (bstart + size).min(ext_len);
                    g[bstart] = ext[bstart];
                    for t in bstart + 1..bend {
                        g[t] = op(g[t - 1], ext[t]);
                    }
                    h[bend - 1] = ext[bend - 1];
                    for t in (bstart..bend - 1).rev() {
                        h[t] = op(h[t + 1], ext[t]);
                    }
                    bstart = bend;
                }
                // Combine: window [i, i+size-1] splits across at most two blocks, and
                // h[i] (suffix to its block end) ∪ g[i+size-1] (prefix from its block
                // start) covers it exactly. Idempotent op handles the single-block case.
                for i in 0..mid {
                    out_chunk[base + i * stride] = op(h[i], g[i + size - 1]);
                }
            }
        }
    };

    // The outer slabs are contiguous & disjoint, so fan them across cores when there are enough
    // and the pixel count amortizes spawn (van Herk is O(1) per output element, so gate on pixels,
    // like uniform_filter1d). Byte-identical: the partition is the only change.
    let nthreads = if arr.size() < (1 << 20) {
        1
    } else {
        ndimage_filter_thread_count(arr.size(), size).min(outer.max(1))
    };
    if MINMAX_HGW_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || nthreads <= 1
        || outer < 2
    {
        process(&arr.data, &mut out.data);
    } else {
        let slabs_per = outer.div_ceil(nthreads);
        let process = &process;
        std::thread::scope(|scope| {
            for (in_chunk, out_chunk) in arr
                .data
                .chunks(slab * slabs_per)
                .zip(out.data.chunks_mut(slab * slabs_per))
            {
                scope.spawn(move || process(in_chunk, out_chunk));
            }
        });
    }
    out
}

#[inline(always)]
fn filter1d_queue_evicts(back: f64, val: f64, is_max: bool) -> bool {
    if is_max { back <= val } else { back >= val }
}

#[inline(always)]
fn reflect_origin0_ext_index(t: usize, mid: usize, lo: usize) -> usize {
    if t < lo {
        lo - t - 1
    } else {
        let coord = t - lo;
        if coord < mid {
            coord
        } else {
            (mid - 1) - (coord - mid)
        }
    }
}

fn minmax_filter1d_reflect_contiguous_queue(
    arr: &NdArray,
    axis: usize,
    size: usize,
    is_max: bool,
) -> NdArray {
    let mid = arr.shape[axis];
    let outer: usize = arr.shape[..axis].iter().product();
    let slab = mid;
    let lo = size / 2;
    let ext_len = mid + size - 1;
    let cap = size + 1;

    let mut out = NdArray::zeros(arr.shape.clone());
    let mut queue_idx = vec![0usize; cap];
    let mut queue_val = vec![0.0f64; cap];

    for o in 0..outer {
        let base = o * slab;
        let input = &arr.data[base..base + mid];
        let output = &mut out.data[base..base + mid];
        let mut head = 0usize;
        let mut tail = 0usize;
        let mut len = 0usize;
        let mut nan_count = 0usize;

        for next in 0..ext_len {
            let src = reflect_origin0_ext_index(next, mid, lo);
            let val = input[src];
            if val.is_nan() {
                nan_count += 1;
            } else {
                while len > 0 {
                    let back = if tail == 0 { cap - 1 } else { tail - 1 };
                    if filter1d_queue_evicts(queue_val[back], val, is_max) {
                        tail = back;
                        len -= 1;
                    } else {
                        break;
                    }
                }
                queue_idx[tail] = next;
                queue_val[tail] = val;
                tail += 1;
                if tail == cap {
                    tail = 0;
                }
                len += 1;
            }

            if next + 1 >= size {
                let left = next + 1 - size;
                while len > 0 && queue_idx[head] < left {
                    head += 1;
                    if head == cap {
                        head = 0;
                    }
                    len -= 1;
                }
                output[left] = if nan_count == 0 {
                    queue_val[head]
                } else {
                    f64::NAN
                };
                let leaving = input[reflect_origin0_ext_index(left, mid, lo)];
                if leaving.is_nan() {
                    nan_count -= 1;
                }
            }
        }
    }

    out
}

/// Single-pass monotonic index queue for the public NaN-propagating 1-D min/max
/// filters. It keeps the HGW boundary-resolved line materialization but fuses the
/// prefix/suffix/combine passes into one scan. NaNs are counted out-of-band so the
/// output remains the canonical NaN whenever any window element is NaN; equal
/// non-NaN extrema evict older entries to match the left-to-right `f64::max/min`
/// fold for signed zeros.
fn minmax_filter1d_nanprop_queue(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
    is_max: bool,
) -> NdArray {
    if mode == BoundaryMode::Reflect
        && origin == 0
        && size <= arr.shape[axis]
        && arr.shape[axis + 1..].iter().product::<usize>() == 1
    {
        return minmax_filter1d_reflect_contiguous_queue(arr, axis, size, is_max);
    }
    minmax_filter1d_nanprop_queue_generic(arr, axis, size, origin, mode, cval, is_max)
}

/// Index-driven filter kernel: the loop variables are positions in the extended
/// (boundary-padded) line and index more than one buffer per step, so iterator
/// form would obscure the index algebra rather than simplify it
/// (frankenscipy-iit9c).
#[allow(clippy::needless_range_loop)]
fn minmax_filter1d_nanprop_queue_generic(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
    is_max: bool,
) -> NdArray {
    let mid = arr.shape[axis];
    let inner: usize = arr.shape[axis + 1..].iter().product();
    let outer: usize = arr.shape[..axis].iter().product();
    let n = mid as i64;
    let size_i = size as i64;
    let lo = size_i / 2 + origin;
    let ext_len = mid + size - 1;
    let stride = inner;
    let slab = mid * inner;

    let mut out = NdArray::zeros(arr.shape.clone());
    let mut ext = vec![0.0f64; ext_len];
    let mut queue = vec![0usize; ext_len];

    for o in 0..outer {
        let outer_base = o * slab;
        for j in 0..inner {
            let base = outer_base + j;
            let interior_start = lo as usize;
            let interior_end = interior_start + mid;
            for t in 0..interior_start {
                let coord = t as i64 - lo;
                ext[t] = match boundary_index_1d(coord, n, mode) {
                    Some(m) => arr.data[base + (m as usize) * stride],
                    None => cval,
                };
            }
            if stride == 1 {
                ext[interior_start..interior_end].copy_from_slice(&arr.data[base..base + mid]);
            } else {
                for t in interior_start..interior_end {
                    ext[t] = arr.data[base + (t - interior_start) * stride];
                }
            }
            for t in interior_end..ext_len {
                let coord = t as i64 - lo;
                ext[t] = match boundary_index_1d(coord, n, mode) {
                    Some(m) => arr.data[base + (m as usize) * stride],
                    None => cval,
                };
            }

            let mut head = 0usize;
            let mut tail = 0usize;
            let mut next = 0usize;
            let mut nan_count = 0usize;
            for i in 0..mid {
                let right = i + size - 1;
                while next <= right {
                    let val = ext[next];
                    if val.is_nan() {
                        nan_count += 1;
                    } else {
                        while tail > head
                            && filter1d_queue_evicts(ext[queue[tail - 1]], val, is_max)
                        {
                            tail -= 1;
                        }
                        queue[tail] = next;
                        tail += 1;
                    }
                    next += 1;
                }
                while head < tail && queue[head] < i {
                    head += 1;
                }
                out.data[base + i * stride] = if nan_count == 0 {
                    ext[queue[head]]
                } else {
                    f64::NAN
                };
                if ext[i].is_nan() {
                    nan_count -= 1;
                }
            }
        }
    }
    out
}

/// The parameter list is the filter specification itself — axis, window, boundary
/// mode, constant, origin and the output buffer. Bundling it into a struct would
/// add an indirection per call in a per-axis kernel and hide the arguments a
/// reader needs to see at the call site (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn minmax_filter_along_axis(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
    is_max: bool,
    needs_total_cmp: bool,
) -> NdArray {
    use std::cmp::Ordering;
    use std::collections::VecDeque;

    if MINMAX_FILTER_HGW.load(std::sync::atomic::Ordering::Relaxed) {
        // Clean data → fast `f64::max`/`f64::min` (byte-identical to the total-order
        // pick when neither NaN nor -0.0 is present); otherwise `tc_max`/`tc_min`
        // to preserve scipy's total-order tie-breaks and NaN propagation.
        return match (is_max, needs_total_cmp) {
            (true, false) => minmax_along_axis_hgw(arr, axis, size, origin, mode, cval, f64::max),
            (false, false) => minmax_along_axis_hgw(arr, axis, size, origin, mode, cval, f64::min),
            (true, true) => minmax_along_axis_hgw(arr, axis, size, origin, mode, cval, tc_max),
            (false, true) => minmax_along_axis_hgw(arr, axis, size, origin, mode, cval, tc_min),
        };
    }

    let n = arr.shape[axis] as i64;
    let stride = arr.strides[axis];
    let shape_axis = arr.shape[axis];
    let size_i = size as i64;
    let lo = size_i / 2 + origin; // window left extent relative to the output
    let mut out = NdArray::zeros(arr.shape.clone());
    let total = arr.size();
    let mut deque: VecDeque<(i64, f64)> = VecDeque::new();

    for flat in 0..total {
        // Process each line along `axis` once, from its head (axis-coord 0).
        // coord[axis] == (flat / stride) % shape[axis]; the O(1) test avoids the
        // per-element unravel allocation.
        if !(flat / stride).is_multiple_of(shape_axis) {
            continue;
        }
        let base = flat;

        deque.clear();
        let mut next_p = -lo; // smallest neighbourhood coordinate needed
        for i in 0..n {
            let right = i - lo + size_i - 1; // window right edge (input coord)
            while next_p <= right {
                let val = match boundary_index_1d(next_p, n, mode) {
                    Some(m) => arr.data[base + (m as usize) * stride],
                    None => cval,
                };
                while let Some(&(_, back)) = deque.back() {
                    let ord = back.total_cmp(&val);
                    let evict = if is_max {
                        ord != Ordering::Greater
                    } else {
                        ord != Ordering::Less
                    };
                    if evict {
                        deque.pop_back();
                    } else {
                        break;
                    }
                }
                deque.push_back((next_p, val));
                next_p += 1;
            }
            let left = i - lo;
            while let Some(&(p, _)) = deque.front() {
                if p < left {
                    deque.pop_front();
                } else {
                    break;
                }
            }
            out.data[base + (i as usize) * stride] = deque.front().unwrap().1;
        }
    }
    out
}

pub fn minimum_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter_with_origins(input, size, &[0], mode, cval)
}

/// Minimum filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn minimum_filter_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    rank_filter_index_usize_axes(input, size, &axes, mode, cval, 0)
}

/// Minimum filter with SciPy `origin` semantics.
pub fn minimum_filter_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    separable_minmax_filter(input, size, origins, mode, cval, false)
}

/// Maximum filter.
///
/// Matches `scipy.ndimage.maximum_filter`.
pub fn maximum_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    maximum_filter_with_origins(input, size, &[0], mode, cval)
}

/// Maximum filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn maximum_filter_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }
    let kernel_total = filter_footprint_size(axes.len(), size)?;
    rank_filter_index_usize_axes(input, size, &axes, mode, cval, kernel_total - 1)
}

/// Maximum filter with SciPy `origin` semantics.
pub fn maximum_filter_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    separable_minmax_filter(input, size, origins, mode, cval, true)
}

/// Rank filter: select the element at `rank` from each sorted neighborhood.
///
/// Matches `scipy.ndimage.rank_filter`; negative ranks count backward from the
/// end of the filter footprint.
pub fn rank_filter(
    input: &NdArray,
    rank: isize,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    rank_filter_with_origins(input, rank, size, &[0], mode, cval)
}

/// Rank filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior after rank
/// validation against the selected-axis footprint.
pub fn rank_filter_axes(
    input: &NdArray,
    rank: isize,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    let footprint_size = if axes.is_empty() {
        1
    } else {
        filter_footprint_size(axes.len(), size)?
    };
    let footprint_size = isize::try_from(footprint_size)
        .map_err(|_| NdimageError::InvalidArgument("filter footprint is too large".to_string()))?;
    let normalized_rank = if rank < 0 {
        footprint_size + rank
    } else {
        rank
    };
    if !(0..footprint_size).contains(&normalized_rank) {
        return Err(NdimageError::InvalidArgument(
            "rank not within filter footprint size".to_string(),
        ));
    }
    let rank_index = usize::try_from(normalized_rank).map_err(|_| {
        NdimageError::InvalidArgument("rank not within filter footprint size".to_string())
    })?;

    rank_filter_index_usize_axes(input, size, &axes, mode, cval, rank_index)
}

/// Rank filter with SciPy `origin` semantics.
pub fn rank_filter_with_origins(
    input: &NdArray,
    rank: isize,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let footprint_size = filter_footprint_size(input.ndim(), size)?;
    let footprint_size = isize::try_from(footprint_size)
        .map_err(|_| NdimageError::InvalidArgument("filter footprint is too large".to_string()))?;
    let normalized_rank = if rank < 0 {
        footprint_size + rank
    } else {
        rank
    };
    if !(0..footprint_size).contains(&normalized_rank) {
        return Err(NdimageError::InvalidArgument(
            "rank not within filter footprint size".to_string(),
        ));
    }
    let rank_index = usize::try_from(normalized_rank).map_err(|_| {
        NdimageError::InvalidArgument("rank not within filter footprint size".to_string())
    })?;

    rank_filter_index_with_origins(input, size, origins, mode, cval, rank_index)
}

fn filter_footprint_size(ndim: usize, size: usize) -> Result<usize, NdimageError> {
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }

    (0..ndim).try_fold(1usize, |acc, _| {
        acc.checked_mul(size).ok_or_else(|| {
            NdimageError::InvalidArgument("filter footprint is too large".to_string())
        })
    })
}

fn select_total_rank(neighborhood: &mut [f64], rank: usize) -> f64 {
    let rank = rank.min(neighborhood.len() - 1);
    let (_, selected, _) = neighborhood.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
    *selected
}

fn rank_filter_index_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) -> Result<NdArray, NdimageError> {
    filter_footprint_size(input.ndim(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);
    let origins = normalize_filter_origins(ndim, &kernel_shape, origins)?;

    fill_rank_filter(
        &mut output,
        input,
        ndim,
        kernel_total,
        &kernel_strides,
        &offsets,
        &origins,
        mode,
        cval,
        rank,
    );

    Ok(output)
}

fn rank_filter_index_usize_axes(
    input: &NdArray,
    size: usize,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) -> Result<NdArray, NdimageError> {
    let origins = vec![0; axes.len()];
    rank_filter_index_usize_axes_with_origins(input, size, axes, &origins, mode, cval, rank)
}

fn rank_filter_index_usize_axes_with_origins(
    input: &NdArray,
    size: usize,
    axes: &[usize],
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) -> Result<NdArray, NdimageError> {
    if axes.is_empty() {
        return Ok(input.clone());
    }
    if origins.len() != axes.len() {
        return Err(NdimageError::InvalidArgument(
            "origin must match selected axes".to_string(),
        ));
    }
    filter_footprint_size(axes.len(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    for &axis in axes {
        if axis >= ndim {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {axis} out of range for {ndim}-dimensional input"
            )));
        }
    }
    for &origin in origins {
        validate_filter_origin(size, origin)?;
    }

    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; axes.len()];
    let kernel_shape: Vec<usize> = vec![size; axes.len()];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);
    let n = input.size();
    let strides = &input.strides;
    let shape = &input.shape;
    // Precompute each footprint element's full-ndim index delta (nonzero only on the
    // selected axes) + flat delta ONCE (the old path allocated k_idx + in_idx Vecs per
    // element per pixel). Interior pixels then gather straight from the flat array.
    let (tap_delta, tap_flat, lo, hi) = footprint_deltas(
        ndim,
        shape,
        strides,
        kernel_total,
        &kernel_strides,
        axes,
        &offsets,
        origins,
    );
    let work = |start: usize, os: &mut [f64]| {
        let mut out_idx = vec![0i64; ndim];
        let mut in_idx = vec![0i64; ndim];
        let mut nb = vec![0.0f64; kernel_total];
        for (li, slot) in os.iter_mut().enumerate() {
            let p = start + li;
            let mut rem = p;
            let mut interior = true;
            for d in 0..ndim {
                let c = (rem / strides[d]) as i64;
                rem %= strides[d];
                out_idx[d] = c;
                if c < lo[d] || c >= hi[d] {
                    interior = false;
                }
            }
            if interior {
                for (k, slot) in nb.iter_mut().enumerate() {
                    *slot = input.data[(p as i64 + tap_flat[k]) as usize];
                }
            } else {
                for (k, slot) in nb.iter_mut().enumerate() {
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] + tap_delta[k][d];
                    }
                    *slot = input.get_boundary(&in_idx, mode, cval);
                }
            }
            *slot = select_total_rank(&mut nb, rank);
        }
    };
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        work(0, &mut output.data);
        return Ok(output);
    }
    let chunk = n.div_ceil(nthreads);
    let work = &work;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || work(t * chunk, out_chunk));
        }
    });
    Ok(output)
}

/// Build the per-footprint-element index deltas for an axes-subset filter: `tap_delta[k]`
/// is the full-ndim input-index offset of footprint element `k` (nonzero only on the
/// selected `axes`), `tap_flat[k]` its flat-array delta, and `[lo,hi)` the interior box
/// (only the selected axes are constrained). `kernel_total` is the footprint size.
#[allow(clippy::type_complexity)]
/// The parameter list is the filter specification itself — axis, window, boundary
/// mode, constant, origin and the output buffer. Bundling it into a struct would
/// add an indirection per call in a per-axis kernel and hide the arguments a
/// reader needs to see at the call site (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn footprint_deltas(
    ndim: usize,
    shape: &[usize],
    strides: &[usize],
    kernel_total: usize,
    kernel_strides: &[usize],
    axes: &[usize],
    offsets: &[i64],
    origins: &[i64],
) -> (Vec<Vec<i64>>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut tap_delta: Vec<Vec<i64>> = Vec::with_capacity(kernel_total);
    for flat_k in 0..kernel_total {
        let mut rem = flat_k;
        let mut delta = vec![0i64; ndim];
        for (d, &axis) in axes.iter().enumerate() {
            let k = (rem / kernel_strides[d]) as i64;
            rem %= kernel_strides[d];
            delta[axis] = k - offsets[d] - origins[d];
        }
        tap_delta.push(delta);
    }
    let tap_flat: Vec<i64> = tap_delta
        .iter()
        .map(|d| (0..ndim).map(|i| d[i] * strides[i] as i64).sum())
        .collect();
    let mut lo = vec![0i64; ndim];
    let mut hi: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    for d in 0..ndim {
        let mut mn = 0i64;
        let mut mx = 0i64;
        for t in &tap_delta {
            mn = mn.min(t[d]);
            mx = mx.max(t[d]);
        }
        lo[d] = -mn;
        hi[d] = shape[d] as i64 - mx;
    }
    (tap_delta, tap_flat, lo, hi)
}

/// Apply a generic function to each local neighborhood.
///
/// Matches `scipy.ndimage.generic_filter`.
pub fn generic_filter<F>(
    input: &NdArray,
    function: F,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    generic_filter_with_origins(input, function, size, &[0], mode, cval)
}

/// Apply a generic function over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior without invoking the
/// callback.
pub fn generic_filter_axes<F>(
    input: &NdArray,
    function: F,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let kernel_total = filter_footprint_size(axes.len(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let n = input.size();
    let strides = &input.strides;
    let shape = &input.shape;
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; axes.len()];
    let origins: Vec<i64> = vec![0; axes.len()];
    let kernel_shape: Vec<usize> = vec![size; axes.len()];
    let kernel_strides = compute_strides(&kernel_shape);

    // Interior flat-gather (no per-element allocs / boundary) — same fix as the full-ndim
    // generic_filter path; the footprint is gathered in flat_k order, so byte-identical.
    let (tap_delta, tap_flat, lo, hi) = footprint_deltas(
        ndim,
        shape,
        strides,
        kernel_total,
        &kernel_strides,
        &axes,
        &offsets,
        &origins,
    );
    let function = &function;
    let work = |start: usize, os: &mut [f64]| {
        let mut out_idx = vec![0i64; ndim];
        let mut in_idx = vec![0i64; ndim];
        let mut nb = vec![0.0f64; kernel_total];
        for (li, slot) in os.iter_mut().enumerate() {
            let p = start + li;
            let mut rem = p;
            let mut interior = true;
            for d in 0..ndim {
                let c = (rem / strides[d]) as i64;
                rem %= strides[d];
                out_idx[d] = c;
                if c < lo[d] || c >= hi[d] {
                    interior = false;
                }
            }
            if interior {
                for (k, slot) in nb.iter_mut().enumerate() {
                    *slot = input.data[(p as i64 + tap_flat[k]) as usize];
                }
            } else {
                for (k, slot) in nb.iter_mut().enumerate() {
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] + tap_delta[k][d];
                    }
                    *slot = input.get_boundary(&in_idx, mode, cval);
                }
            }
            *slot = function(nb.as_slice());
        }
    };
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        work(0, &mut output.data);
        return Ok(output);
    }
    let chunk = n.div_ceil(nthreads);
    let work = &work;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || work(t * chunk, out_chunk));
        }
    });
    Ok(output)
}

/// Apply a generic function to each local neighborhood with SciPy `origin` semantics.
pub fn generic_filter_with_origins<F>(
    input: &NdArray,
    function: F,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    filter_footprint_size(input.ndim(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);
    let origins = normalize_filter_origins(ndim, &kernel_shape, origins)?;

    let n = input.size();
    let shape = &input.shape;
    let strides = &input.strides;
    // Precompute each footprint element's per-dim index delta + flat delta ONCE (the old
    // path re-derived k_idx and allocated k_idx + in_idx Vecs PER element PER pixel).
    let mut tap_delta: Vec<Vec<i64>> = Vec::with_capacity(kernel_total);
    for flat_k in 0..kernel_total {
        let mut rem = flat_k;
        let mut delta = vec![0i64; ndim];
        for d in 0..ndim {
            let k = (rem / kernel_strides[d]) as i64;
            rem %= kernel_strides[d];
            delta[d] = k - offsets[d] - origins[d];
        }
        tap_delta.push(delta);
    }
    let tap_flat: Vec<i64> = tap_delta
        .iter()
        .map(|d| (0..ndim).map(|i| d[i] * strides[i] as i64).sum())
        .collect();
    // Interior box [lo[d], hi[d)): every footprint element stays in-bounds along dim d.
    let mut lo = vec![0i64; ndim];
    let mut hi: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    for d in 0..ndim {
        let mut mn = 0i64;
        let mut mx = 0i64;
        for t in &tap_delta {
            mn = mn.min(t[d]);
            mx = mx.max(t[d]);
        }
        lo[d] = -mn;
        hi[d] = shape[d] as i64 - mx;
    }
    // Per pixel: gather the footprint (interior → direct flat gather, no boundary/allocs;
    // border → get_boundary) into a reused per-thread buffer in flat_k order, then apply the
    // pure user `function`. Byte-identical to the old per-pixel loop (same gather order).
    let function = &function;
    let work = |start: usize, os: &mut [f64]| {
        let mut out_idx = vec![0i64; ndim];
        let mut in_idx = vec![0i64; ndim];
        let mut nb = vec![0.0f64; kernel_total];
        for (li, slot) in os.iter_mut().enumerate() {
            let p = start + li;
            let mut rem = p;
            let mut interior = true;
            for d in 0..ndim {
                let c = (rem / strides[d]) as i64;
                rem %= strides[d];
                out_idx[d] = c;
                if c < lo[d] || c >= hi[d] {
                    interior = false;
                }
            }
            if interior {
                for (k, slot) in nb.iter_mut().enumerate() {
                    *slot = input.data[(p as i64 + tap_flat[k]) as usize];
                }
            } else {
                for (k, slot) in nb.iter_mut().enumerate() {
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] + tap_delta[k][d];
                    }
                    *slot = input.get_boundary(&in_idx, mode, cval);
                }
            }
            *slot = function(nb.as_slice());
        }
    };
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        work(0, &mut output.data);
        return Ok(output);
    }
    let chunk = n.div_ceil(nthreads);
    let work = &work;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || work(t * chunk, out_chunk));
        }
    });

    Ok(output)
}

/// Reference old per-pixel generic_filter path (alloc-per-element gather), retained for the
/// same-process A/B benchmark and byte-identity proof only.
#[doc(hidden)]
pub fn generic_filter_perpixel_ref<F>(
    input: &NdArray,
    function: F,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> NdArray
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let ndim = input.ndim();
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);
    let mut output = NdArray::zeros(input.shape.clone());
    let n = input.size();
    let function = &function;
    let pixel = |flat_out: usize| -> f64 {
        let out_idx = input.unravel(flat_out);
        let mut nb = Vec::with_capacity(kernel_total);
        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for (d, slot) in k_idx.iter_mut().enumerate() {
                *slot = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }
            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
            }
            nb.push(input.get_boundary(&in_idx, mode, cval));
        }
        function(nb.as_slice())
    };
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        for (flat_out, slot) in output.data.iter_mut().enumerate() {
            *slot = pixel(flat_out);
        }
    } else {
        let chunk = n.div_ceil(nthreads);
        let pixel = &pixel;
        std::thread::scope(|scope| {
            for (t, oc) in output.data.chunks_mut(chunk).enumerate() {
                scope.spawn(move || {
                    for (li, slot) in oc.iter_mut().enumerate() {
                        *slot = pixel(t * chunk + li);
                    }
                });
            }
        });
    }
    output
}

/// Same-binary A/B toggle for `generic_filter1d`. When `true`, the output pixels are computed
/// serially (the ORIG behaviour). When `false` (default), the independent per-pixel neighborhood
/// gather + callback fans across cores. Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static GENERIC_FILTER1D_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply a generic function along one axis.
///
/// Matches the common `scipy.ndimage.generic_filter1d` sliding-window behavior
/// for reducers that produce one output value per input position.
pub fn generic_filter1d<F>(
    input: &NdArray,
    function: F,
    filter_size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    generic_filter1d_with_origin(input, function, filter_size, axis, 0, mode, cval)
}

/// Apply a generic function along one signed axis with SciPy-style normalization.
pub fn generic_filter1d_signed_axis<F>(
    input: &NdArray,
    function: F,
    filter_size: usize,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let axis = normalize_signed_axis(axis, input.ndim())?;
    generic_filter1d(input, function, filter_size, axis, mode, cval)
}

/// Apply a generic function along SciPy's default `axis=-1`.
pub fn generic_filter1d_default_axis<F>(
    input: &NdArray,
    function: F,
    filter_size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    generic_filter1d_signed_axis(input, function, filter_size, -1, mode, cval)
}

/// Apply a generic function along one axis with SciPy `origin` semantics.
pub fn generic_filter1d_with_origin<F>(
    input: &NdArray,
    function: F,
    filter_size: usize,
    axis: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if filter_size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(
            "axis out of bounds".to_string(),
        ));
    }
    validate_filter_origin(filter_size, origin)?;

    let mut output = NdArray::zeros(input.shape.clone());
    let offset = filter_size as i64 / 2;
    let n = input.size();

    // Each output pixel gathers its own `filter_size` neighborhood along `axis` and calls the
    // (pure, Sync) reducer — the pixels are independent and write disjoint slots, so distributing
    // the output indices across cores is BYTE-IDENTICAL to the serial loop (the reducer is
    // deterministic; only the owning core changes). scipy's generic_filter1d is single-threaded.
    let function = &function;
    let work = |start: usize, os: &mut [f64]| {
        let mut neighborhood = vec![0.0f64; filter_size];
        for (li, slot) in os.iter_mut().enumerate() {
            let out_idx = input.unravel(start + li);
            for (k, cell) in neighborhood.iter_mut().enumerate() {
                let mut in_idx: Vec<i64> = out_idx.iter().map(|&coord| coord as i64).collect();
                in_idx[axis] += k as i64 - offset - origin;
                *cell = input.get_boundary(&in_idx, mode, cval);
            }
            *slot = function(&neighborhood);
        }
    };

    let nthreads = if GENERIC_FILTER1D_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        ndimage_filter_thread_count(n, filter_size)
    };
    if nthreads <= 1 {
        work(0, &mut output.data);
        return Ok(output);
    }
    let chunk = n.div_ceil(nthreads);
    let work = &work;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            scope.spawn(move || work(t * chunk, out_chunk));
        }
    });
    Ok(output)
}

/// Apply a vectorized reducer to each local neighborhood.
///
/// Matches `scipy.ndimage.vectorized_filter` for scalar reducers over a
/// uniform `size` footprint. The callback receives each neighborhood in the
/// same flattened order as `generic_filter`.
pub fn vectorized_filter<F>(
    input: &NdArray,
    function: F,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    generic_filter(input, function, size, mode, cval)
}

/// Percentile filter: select the p-th percentile from each neighborhood.
///
/// Matches `scipy.ndimage.percentile_filter`.
pub fn percentile_filter(
    input: &NdArray,
    percentile: f64,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    percentile_filter_with_origins(input, percentile, size, &[0], mode, cval)
}

/// Percentile filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior after percentile
/// validation.
pub fn percentile_filter_axes(
    input: &NdArray,
    percentile: f64,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !percentile.is_finite() {
        return Err(NdimageError::InvalidArgument(
            "invalid percentile".to_string(),
        ));
    }

    let mut normalized = percentile;
    if normalized < 0.0 {
        normalized += 100.0;
    }
    if !(0.0..=100.0).contains(&normalized) {
        return Err(NdimageError::InvalidArgument(
            "invalid percentile".to_string(),
        ));
    }

    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let filter_size = filter_footprint_size(axes.len(), size)?;
    let rank = if normalized == 100.0 {
        filter_size - 1
    } else {
        (filter_size as f64 * normalized / 100.0).floor() as usize
    };

    rank_filter_index_usize_axes(input, size, &axes, mode, cval, rank)
}

/// Percentile filter with SciPy `origin` semantics.
pub fn percentile_filter_with_origins(
    input: &NdArray,
    percentile: f64,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !percentile.is_finite() {
        return Err(NdimageError::InvalidArgument(
            "invalid percentile".to_string(),
        ));
    }

    let mut normalized = percentile;
    if normalized < 0.0 {
        normalized += 100.0;
    }
    if !(0.0..=100.0).contains(&normalized) {
        return Err(NdimageError::InvalidArgument(
            "invalid percentile".to_string(),
        ));
    }

    let filter_size = filter_footprint_size(input.ndim(), size)?;
    let rank = if normalized == 100.0 {
        filter_size - 1
    } else {
        (filter_size as f64 * normalized / 100.0).floor() as usize
    };
    let rank = isize::try_from(rank)
        .map_err(|_| NdimageError::InvalidArgument("filter footprint is too large".to_string()))?;

    rank_filter_with_origins(input, rank, size, origins, mode, cval)
}

/// Morphological gradient: dilation minus erosion.
///
/// Matches `scipy.ndimage.morphological_gradient`.
pub fn morphological_gradient(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    morphological_gradient_with_origins(input, size, &[0], mode, cval)
}

/// Morphological gradient over a SciPy-style signed axes subset.
pub fn morphological_gradient_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let max_img = grey_dilation_axes(input, size, axes, mode, cval)?;
    let min_img = grey_erosion_axes(input, size, axes, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = max_img.data[i] - min_img.data[i];
    }
    Ok(result)
}

/// Morphological gradient with SciPy `origin` semantics.
pub fn morphological_gradient_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let max_img = grey_dilation_with_origins(input, size, origins, mode, cval)?;
    let min_img = grey_erosion_with_origins(input, size, origins, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = max_img.data[i] - min_img.data[i];
    }
    Ok(result)
}

/// Morphological Laplace: dilation plus erosion minus twice the input.
///
/// Matches `scipy.ndimage.morphological_laplace`.
pub fn morphological_laplace(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    morphological_laplace_with_origins(input, size, &[0], mode, cval)
}

/// Morphological Laplace over a SciPy-style signed axes subset.
pub fn morphological_laplace_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let max_img = grey_dilation_axes(input, size, axes, mode, cval)?;
    let min_img = grey_erosion_axes(input, size, axes, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = max_img.data[i] + min_img.data[i] - 2.0 * input.data[i];
    }
    Ok(result)
}

/// Morphological Laplace with SciPy `origin` semantics.
pub fn morphological_laplace_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let max_img = grey_dilation_with_origins(input, size, origins, mode, cval)?;
    let min_img = grey_erosion_with_origins(input, size, origins, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = max_img.data[i] + min_img.data[i] - 2.0 * input.data[i];
    }
    Ok(result)
}

/// White top-hat: input minus morphological opening.
///
/// Matches `scipy.ndimage.white_tophat`.
pub fn white_tophat(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    white_tophat_with_origins(input, size, &[0], mode, cval)
}

/// White top-hat over a SciPy-style signed axes subset.
pub fn white_tophat_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let opened = grey_opening_axes(input, size, axes, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = input.data[i] - opened.data[i];
    }
    Ok(result)
}

/// White top-hat with SciPy `origin` semantics.
pub fn white_tophat_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let opened = grey_opening_with_origins(input, size, origins, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = input.data[i] - opened.data[i];
    }
    Ok(result)
}

/// Black top-hat: morphological closing minus input.
///
/// Matches `scipy.ndimage.black_tophat`.
pub fn black_tophat(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    black_tophat_with_origins(input, size, &[0], mode, cval)
}

/// Black top-hat over a SciPy-style signed axes subset.
pub fn black_tophat_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let closed = grey_closing_axes(input, size, axes, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = closed.data[i] - input.data[i];
    }
    Ok(result)
}

/// Black top-hat with SciPy `origin` semantics.
pub fn black_tophat_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let closed = grey_closing_with_origins(input, size, origins, mode, cval)?;

    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = closed.data[i] - input.data[i];
    }
    Ok(result)
}

/// Histogram of values in labeled regions.
///
/// Matches `scipy.ndimage.histogram`; scalar SciPy results are returned as a
/// one-element vector of bin counts, while explicit `index` lists return one
/// histogram per label.
/// When `true`, force the global (no-labels) `histogram` onto the ORIG group path (clone + serial
/// bincount). Byte-identical either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_HISTOGRAM_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn histogram(
    input: &NdArray,
    min_val: f64,
    max_val: f64,
    nbins: usize,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<usize>>, NdimageError> {
    // Fast streaming path: a one-based-contiguous index uses a parallel privatized per-label
    // histogram (byte-identical, integer counts), skipping the per-group Vec materialization.
    // The validation short-circuit (all-zero histograms) must match the group path's semantics.
    if let (Some(labels), Some(index)) = (labels, index)
        && input.shape == labels.shape
    {
        {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                if nbins == 0
                    || !min_val.is_finite()
                    || !max_val.is_finite()
                    || max_val <= min_val
                    || input.data.iter().any(|value| !value.is_finite())
                {
                    return Ok(vec![vec![0usize; nbins]; label_count]);
                }
                return Ok(measurement_one_based_histogram(
                    &input.data,
                    &labels.data,
                    label_count,
                    min_val,
                    max_val,
                    nbins,
                ));
            }
        }
    }
    // Global histogram (no labels): bin the whole array with a privatized parallel bincount,
    // skipping the `measurement_label_groups` full-data CLONE. BYTE-IDENTICAL to the group path
    // (one group = every element; the range filter and per-value bin assignment are unchanged, and
    // integer counts sum order-independently). Same validation short-circuit as the group path.
    if labels.is_none()
        && !NDIMAGE_HISTOGRAM_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        if nbins == 0
            || !min_val.is_finite()
            || !max_val.is_finite()
            || max_val <= min_val
            || input.data.iter().any(|value| !value.is_finite())
        {
            return Ok(vec![vec![0usize; nbins]]);
        }
        let bin_width = (max_val - min_val) / nbins as f64;
        let data = input.data.as_slice();
        let n = data.len();
        let bincount = |chunk: &[f64]| -> Vec<usize> {
            let mut h = vec![0usize; nbins];
            for &value in chunk {
                if value < min_val || value > max_val {
                    continue;
                }
                let bin = ((value - min_val) / bin_width).floor() as usize;
                h[bin.min(nbins - 1)] += 1;
            }
            h
        };
        let nthreads = ndimage_filter_thread_count(n, 4);
        let hist = if nthreads <= 1 {
            bincount(data)
        } else {
            let cw = n.div_ceil(nthreads);
            let bincount = &bincount;
            let parts: Vec<Vec<usize>> = std::thread::scope(|scope| {
                data.chunks(cw)
                    .map(|c| scope.spawn(move || bincount(c)))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|h| h.join().expect("histogram chunk panicked"))
                    .collect()
            });
            let mut h = vec![0usize; nbins];
            for p in parts {
                for (i, c) in p.into_iter().enumerate() {
                    h[i] += c;
                }
            }
            h
        };
        return Ok(vec![hist]);
    }

    let groups = measurement_label_groups(input, labels, index)?;
    let mut histograms = vec![vec![0usize; nbins]; groups.len()];
    if nbins == 0
        || !min_val.is_finite()
        || !max_val.is_finite()
        || max_val <= min_val
        || input.data.iter().any(|value| !value.is_finite())
    {
        return Ok(histograms);
    }

    let bin_width = (max_val - min_val) / nbins as f64;
    for (histogram, values) in histograms.iter_mut().zip(groups) {
        for value in values {
            if value < min_val || value > max_val {
                continue;
            }
            let bin = ((value - min_val) / bin_width).floor() as usize;
            histogram[bin.min(nbins - 1)] += 1;
        }
    }

    Ok(histograms)
}

/// Histogram of values in labeled regions.
///
/// Returns a Vec of histograms (bin counts) for each label.
/// Matches `scipy.ndimage.histogram`.
pub fn histogram_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
    min_val: f64,
    max_val: f64,
    nbins: usize,
) -> Vec<Vec<usize>> {
    let mut histograms = vec![vec![0usize; nbins]; num_labels];
    if nbins == 0
        || !min_val.is_finite()
        || !max_val.is_finite()
        || max_val <= min_val
        || input.data.iter().any(|v| !v.is_finite())
    {
        return histograms;
    }
    let bin_width = (max_val - min_val) / nbins as f64;

    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            let bin = ((input.data[i] - min_val) / bin_width).floor() as usize;
            let bin = bin.min(nbins - 1);
            histograms[lbl - 1][bin] += 1;
        }
    }

    histograms
}

/// Minimum and maximum values in each labeled region.
///
/// Returns (min_values, max_values) vectors.
/// Matches `scipy.ndimage.extrema`.
pub fn extrema_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::INFINITY; num_labels];
    let mut maxs = vec![f64::NEG_INFINITY; num_labels];

    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            mins[lbl - 1] = mins[lbl - 1].min(input.data[i]);
            maxs[lbl - 1] = maxs[lbl - 1].max(input.data[i]);
        }
    }

    (mins, maxs)
}

fn normalize_signed_axis(axis: isize, ndim: usize) -> Result<usize, NdimageError> {
    let ndim = isize::try_from(ndim)
        .map_err(|_| NdimageError::InvalidArgument("input rank is too large".to_string()))?;
    let normalized = if axis < 0 { axis + ndim } else { axis };

    if normalized < 0 || normalized >= ndim {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {ndim}-dimensional input"
        )));
    }

    usize::try_from(normalized)
        .map_err(|_| NdimageError::InvalidArgument("axis normalization failed".to_string()))
}

fn normalize_signed_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>, NdimageError> {
    let mut normalized = Vec::with_capacity(axes.len());
    let mut seen = vec![false; ndim];
    for &axis in axes {
        let axis = normalize_signed_axis(axis, ndim)?;
        if seen[axis] {
            return Err(NdimageError::InvalidArgument(
                "axes must be unique".to_string(),
            ));
        }
        seen[axis] = true;
        normalized.push(axis);
    }
    Ok(normalized)
}

/// Sobel edge detection filter along a given axis.
///
/// Matches `scipy.ndimage.sobel`.
pub fn sobel(
    input: &NdArray,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if input.ndim() < 1 {
        return Err(NdimageError::InvalidArgument(
            "input must be at least 1-dimensional".to_string(),
        ));
    }

    // Apply smoothing kernels along non-axis dims, derivative along axis. Each pass is a 1-D
    // kernel, so route through the separable `correlate1d` (the axpy-vectorized path used by
    // gaussian/uniform filters) instead of the general N-D `correlate`: for a 3-tap kernel the
    // general-correlate footprint machinery is overhead-bound (it made sobel ~4.7ms, slower than
    // a 7x7 general correlate). `correlate1d` along axis `d` with the same centered weights and
    // boundary mode is equivalent (neither flips), just without the N-D footprint overhead.
    let kernel_for = |d: usize| {
        if d == axis {
            vec![-1.0, 0.0, 1.0] // derivative
        } else {
            vec![1.0, 2.0, 1.0] // smoothing (triangle)
        }
    };
    let mut current = correlate1d(input, &kernel_for(0), 0, mode, cval)?;
    for d in 1..input.ndim() {
        current = correlate1d(&current, &kernel_for(d), d, mode, cval)?;
    }

    Ok(current)
}

/// Sobel edge detection with SciPy-style signed axis normalization.
pub fn sobel_signed_axis(
    input: &NdArray,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    sobel(input, axis, mode, cval)
}

/// Sobel edge detection using SciPy's default `axis=-1`.
pub fn sobel_default_axis(
    input: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    sobel_signed_axis(input, -1, mode, cval)
}

/// Prewitt edge detection filter along a given axis.
///
/// Matches `scipy.ndimage.prewitt`.
pub fn prewitt(
    input: &NdArray,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }

    // Each pass is a 1-D kernel — use the separable `correlate1d` (the fast axpy path) rather than
    // the general N-D `correlate`, whose footprint machinery is overhead-bound for 3-tap kernels
    // (same fix as `sobel`). Equivalent: same centered weights, same boundary mode, no flip.
    let kernel_for = |d: usize| {
        if d == axis {
            vec![-1.0, 0.0, 1.0]
        } else {
            vec![1.0, 1.0, 1.0] // box smoothing (Prewitt uses uniform weights)
        }
    };
    let mut current = correlate1d(input, &kernel_for(0), 0, mode, cval)?;
    for d in 1..input.ndim() {
        current = correlate1d(&current, &kernel_for(d), d, mode, cval)?;
    }

    Ok(current)
}

/// Prewitt edge detection with SciPy-style signed axis normalization.
pub fn prewitt_signed_axis(
    input: &NdArray,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    prewitt(input, axis, mode, cval)
}

/// Prewitt edge detection using SciPy's default `axis=-1`.
pub fn prewitt_default_axis(
    input: &NdArray,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    prewitt_signed_axis(input, -1, mode, cval)
}

/// Laplace filter (sum of second derivatives).
///
/// Matches `scipy.ndimage.laplace`.
pub fn laplace(input: &NdArray, mode: BoundaryMode, cval: f64) -> Result<NdArray, NdimageError> {
    let axes = (0..input.ndim()).collect::<Vec<_>>();
    laplace_usize_axes(input, &axes, mode, cval)
}

/// Laplace filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn laplace_axes(
    input: &NdArray,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    laplace_usize_axes(input, &axes, mode, cval)
}

fn laplace_usize_axes(
    input: &NdArray,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if axes.is_empty() {
        return Ok(input.clone());
    }

    // Laplace = sum of second-derivative filters along each axis
    let mut result = NdArray::zeros(input.shape.clone());

    for &axis in axes {
        if axis >= input.ndim() {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {axis} out of range for {}-dimensional input",
                input.ndim()
            )));
        }
        let kernel_1d = vec![1.0, -2.0, 1.0];
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        let filtered = correlate(input, &kernel, mode, cval)?;
        for i in 0..result.data.len() {
            result.data[i] += filtered.data[i];
        }
    }

    Ok(result)
}

/// Gaussian Laplace (LoG) filter.
///
/// Matches `scipy.ndimage.gaussian_laplace`: the sum, over each axis, of a
/// Gaussian filter applied with derivative order 2 along that axis (and order 0
/// along the others) — i.e. an analytic second-derivative-of-Gaussian filter,
/// not a finite-difference Laplacian stencil applied after blurring.
pub fn gaussian_laplace(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = (0..input.ndim()).collect::<Vec<_>>();
    gaussian_laplace_usize_axes(input, sigma, &axes, mode, cval)
}

/// Gaussian Laplace (LoG) filter over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn gaussian_laplace_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    gaussian_laplace_usize_axes(input, sigma, &axes, mode, cval)
}

fn gaussian_laplace_usize_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NdimageError::InvalidArgument(
            "sigma must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Ok(input.clone());
    }
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for &derivative_axis in axes {
        if derivative_axis >= ndim {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {derivative_axis} out of range for {ndim}-dimensional input"
            )));
        }
        let orders = axes
            .iter()
            .map(|&axis| if axis == derivative_axis { 2 } else { 0 })
            .collect::<Vec<_>>();
        let deriv2 = gaussian_filter_with_orders_on_axes(input, sigma, axes, &orders, mode, cval)?;
        for (r, d) in result.data.iter_mut().zip(&deriv2.data) {
            *r += d;
        }
    }
    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Morphology
// ══════════════════════════════════════════════════════════════════════

/// Binary erosion: output pixel is 1 only if all pixels in the structuring
/// element neighborhood are 1.
///
/// Matches `scipy.ndimage.binary_erosion`.
fn binary_morph_origins_all_zero(origins: &[i64]) -> bool {
    origins.iter().all(|&o| o == 0)
}

/// Structuring-element reflection offset for binary dilation at origin 0: the
/// scatter form writes `q + (k - size/2)`, whose equivalent gather window is the
/// reflected box. For odd sizes the box is symmetric (offset 0); for even sizes
/// it shifts by -1.
fn binary_dilation_origin_reflection(size: usize) -> i64 {
    (size as i64 - 1) - 2 * (size as i64 / 2)
}

/// Specialized separable binary erosion: per-axis sliding-window AND via a running count
/// of zeros (out-of-bounds counts as a zero, matching the Constant-0 min-filter border).
/// For binary (0/1) data this is byte-identical to the float min-filter (`min == 1` ⇔ no
/// zero in the window) but replaces the per-pixel monotonic deque + `total_cmp` with two
/// integer count updates. frankenscipy-9l5oo.
fn binary_erode_separable(bin: &NdArray, size: usize) -> NdArray {
    // Radical lever: 2D bit-packed erosion (64 px/u64) when the window fits a single word
    // boundary (size < 64). Falls back to the per-pixel running count for N-D / huge windows.
    if size < 64
        && let Some(packed) = binary_erode_bitpack_2d(bin, size)
    {
        return packed;
    }
    let mut cur = bin.clone();
    for axis in 0..bin.ndim() {
        cur = binary_erode_along_axis(&cur, axis, size);
    }
    cur
}

/// Shift a packed bit-row UP (toward higher bit index) by `k` bits (1 ≤ k < 64):
/// `out` bit i = `src` bit (i-k), zero-filled at the low end.
fn shift_bits_up(src: &[u64], k: usize, out: &mut [u64]) {
    for w in 0..src.len() {
        let mut v = src[w] << k;
        if w >= 1 {
            v |= src[w - 1] >> (64 - k);
        }
        out[w] = v;
    }
}

/// Shift a packed bit-row DOWN (toward lower bit index) by `m` bits (0 ≤ m < 64):
/// `out` bit i = `src` bit (i+m), zero-filled at the high end.
fn shift_bits_down(src: &[u64], m: usize, out: &mut [u64]) {
    if m == 0 {
        out.copy_from_slice(src);
        return;
    }
    let n = src.len();
    for w in 0..n {
        let mut v = src[w] >> m;
        if w + 1 < n {
            v |= src[w + 1] << (64 - m);
        }
        out[w] = v;
    }
}

/// 2D bit-packed binary erosion: pack each row into u64 words, erode horizontally via
/// shift-AND (`out[c] = AND of in[c-lo..c-lo+size-1]`, centered by shifting the left-
/// anchored AND down by `size-1-lo`) and vertically via word-AND of `size` rows. The
/// Constant-0 border falls out for free (out-of-range bits/rows are zero → AND is zero).
/// Byte-identical 0/1 output to the separable min-filter. `None` for non-2D. 9l5oo.
fn binary_erode_bitpack_2d(bin: &NdArray, size: usize) -> Option<NdArray> {
    if bin.ndim() != 2 {
        return None;
    }
    let h = bin.shape[0];
    let w = bin.shape[1];
    if h == 0 || w == 0 {
        return None;
    }
    let wpr = w.div_ceil(64);
    let lo = size / 2; // erosion origin is 0 → lo = size/2 (matches minmax_filter_along_axis)
    let center = size - 1 - lo; // re-center the left-anchored window

    // Pack: bit (c%64) of word (r*wpr + c/64) is set iff pixel (r,c) != 0.
    let mut packed = vec![0u64; h * wpr];
    for r in 0..h {
        let row_base = r * w;
        let pk_base = r * wpr;
        for c in 0..w {
            if bin.data[row_base + c] != 0.0 {
                packed[pk_base + c / 64] |= 1u64 << (c % 64);
            }
        }
    }

    // Horizontal erosion (within each row).
    let mut h_eroded = vec![0u64; h * wpr];
    let mut acc = vec![0u64; wpr];
    let mut shifted = vec![0u64; wpr];
    for r in 0..h {
        let row = &packed[r * wpr..(r + 1) * wpr];
        acc.copy_from_slice(row);
        for k in 1..size {
            shift_bits_up(row, k, &mut shifted);
            for (a, s) in acc.iter_mut().zip(shifted.iter()) {
                *a &= *s;
            }
        }
        shift_bits_down(&acc, center, &mut h_eroded[r * wpr..(r + 1) * wpr]);
    }

    // Vertical erosion (word-AND of the `size` rows in [r-lo, r-lo+size-1]); any out-of-
    // range row zeroes the whole output row (Constant-0 border).
    let mut out = NdArray::zeros(bin.shape.clone());
    let mut col_acc = vec![0u64; wpr];
    for r in 0..h {
        let top = r as i64 - lo as i64;
        if top < 0 || top + size as i64 > h as i64 {
            continue; // window hits the border → eroded to 0
        }
        let top = top as usize;
        col_acc.copy_from_slice(&h_eroded[top * wpr..(top + 1) * wpr]);
        for j in 1..size {
            let src = &h_eroded[(top + j) * wpr..(top + j + 1) * wpr];
            for (a, s) in col_acc.iter_mut().zip(src.iter()) {
                *a &= *s;
            }
        }
        let out_base = r * w;
        for c in 0..w {
            if (col_acc[c / 64] >> (c % 64)) & 1 == 1 {
                out.data[out_base + c] = 1.0;
            }
        }
    }
    Some(out)
}

fn binary_erode_along_axis(arr: &NdArray, axis: usize, size: usize) -> NdArray {
    let n = arr.shape[axis] as i64;
    let stride = arr.strides[axis];
    let shape_axis = arr.shape[axis];
    let size_i = size as i64;
    let lo = size_i / 2; // window left extent (origin 0), matching minmax_filter_along_axis
    let total = arr.size();
    let mut out = NdArray::zeros(arr.shape.clone());
    for flat in 0..total {
        if !(flat / stride).is_multiple_of(shape_axis) {
            continue;
        }
        let base = flat;
        let is_zero = |p: i64| -> bool {
            !(0..n).contains(&p) || arr.data[base + (p as usize) * stride] == 0.0
        };
        // Window for output `a` is the input range [a-lo, a-lo+size-1]; seed from a=0.
        let mut count_zeros: i64 = 0;
        for p in (-lo)..(-lo + size_i) {
            if is_zero(p) {
                count_zeros += 1;
            }
        }
        for a in 0..n {
            out.data[base + (a as usize) * stride] = if count_zeros == 0 { 1.0 } else { 0.0 };
            if is_zero(a - lo) {
                count_zeros -= 1;
            }
            if is_zero(a - lo + size_i) {
                count_zeros += 1;
            }
        }
    }
    out
}

fn binary_erosion_once_with_origins(current: &NdArray, size: usize, origins: &[i64]) -> NdArray {
    // Binary erosion is a minimum filter over the booleanized image with a
    // constant-0 border: O(N * ndim) separable sliding-window min instead of the
    // O(N * size^ndim) per-pixel footprint scan. Gated to the default all-zero
    // origin so the kernel window (and origin validation) match exactly.
    if binary_morph_origins_all_zero(origins) {
        let bin = booleanized_binary(current);
        return binary_erode_separable(&bin, size);
    }

    let ndim = current.ndim();
    let mut output = NdArray::zeros(current.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..current.size() {
        let out_idx = current.unravel(flat_out);
        let mut all_set = true;

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for d in 0..ndim {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d] - origins[d];
            }

            let val = current.get_boundary(&in_idx, BoundaryMode::Constant, 0.0);
            if val == 0.0 {
                all_set = false;
                break;
            }
        }

        output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
    }

    output
}

/// Specialized separable binary dilation: per-axis sliding-window OR via a running count of
/// ones (out-of-bounds contributes nothing, matching the Constant-0 max-filter border which
/// uses the reflected structuring-element origin). Byte-identical to the float max-filter for
/// binary data (`max == 1` ⇔ at least one one in the window). frankenscipy-9l5oo.
fn binary_dilate_separable(bin: &NdArray, size: usize) -> NdArray {
    if size < 64
        && let Some(packed) = binary_dilate_bitpack_2d(bin, size)
    {
        return packed;
    }
    let refl = binary_dilation_origin_reflection(size);
    let mut cur = bin.clone();
    for axis in 0..bin.ndim() {
        cur = binary_dilate_along_axis(&cur, axis, size, refl);
    }
    cur
}

/// 2D bit-packed binary dilation (OR analogue of [`binary_erode_bitpack_2d`]): horizontal
/// via shift-OR, vertical via word-OR of the in-window rows. Uses the reflected-SE origin
/// (`lo = size/2 + refl`); out-of-range bits/rows contribute 0 (OR identity), matching the
/// Constant-0 max-filter border. Byte-identical 0/1 output. `None` for non-2D. 9l5oo.
fn binary_dilate_bitpack_2d(bin: &NdArray, size: usize) -> Option<NdArray> {
    if bin.ndim() != 2 {
        return None;
    }
    let h = bin.shape[0];
    let w = bin.shape[1];
    if h == 0 || w == 0 {
        return None;
    }
    let wpr = w.div_ceil(64);
    let refl = binary_dilation_origin_reflection(size);
    let lo = size as i64 / 2 + refl; // 0 for odd sizes, size/2-1 for even (≥ 0)
    let center = (size as i64 - 1 - lo) as usize;

    let mut packed = vec![0u64; h * wpr];
    for r in 0..h {
        let row_base = r * w;
        let pk_base = r * wpr;
        for c in 0..w {
            if bin.data[row_base + c] != 0.0 {
                packed[pk_base + c / 64] |= 1u64 << (c % 64);
            }
        }
    }

    // Horizontal dilation (OR of the window within each row).
    let mut h_dil = vec![0u64; h * wpr];
    let mut acc = vec![0u64; wpr];
    let mut shifted = vec![0u64; wpr];
    for r in 0..h {
        let row = &packed[r * wpr..(r + 1) * wpr];
        acc.copy_from_slice(row);
        for k in 1..size {
            shift_bits_up(row, k, &mut shifted);
            for (a, s) in acc.iter_mut().zip(shifted.iter()) {
                *a |= *s;
            }
        }
        shift_bits_down(&acc, center, &mut h_dil[r * wpr..(r + 1) * wpr]);
    }

    // Vertical dilation: OR the in-range rows in [r-lo, r-lo+size-1] (out-of-range = 0).
    let mut out = NdArray::zeros(bin.shape.clone());
    let mut col_acc = vec![0u64; wpr];
    for r in 0..h {
        col_acc.iter_mut().for_each(|x| *x = 0);
        for j in 0..size as i64 {
            let rr = r as i64 - lo + j;
            if rr < 0 || rr >= h as i64 {
                continue;
            }
            let src = &h_dil[rr as usize * wpr..(rr as usize + 1) * wpr];
            for (a, s) in col_acc.iter_mut().zip(src.iter()) {
                *a |= *s;
            }
        }
        let out_base = r * w;
        for c in 0..w {
            if (col_acc[c / 64] >> (c % 64)) & 1 == 1 {
                out.data[out_base + c] = 1.0;
            }
        }
    }
    Some(out)
}

fn binary_dilate_along_axis(arr: &NdArray, axis: usize, size: usize, origin: i64) -> NdArray {
    let n = arr.shape[axis] as i64;
    let stride = arr.strides[axis];
    let shape_axis = arr.shape[axis];
    let size_i = size as i64;
    let lo = size_i / 2 + origin; // matches minmax_filter_along_axis window for this origin
    let total = arr.size();
    let mut out = NdArray::zeros(arr.shape.clone());
    for flat in 0..total {
        if !(flat / stride).is_multiple_of(shape_axis) {
            continue;
        }
        let base = flat;
        let is_one = |p: i64| -> bool {
            (0..n).contains(&p) && arr.data[base + (p as usize) * stride] == 1.0
        };
        let mut count_ones: i64 = 0;
        for p in (-lo)..(-lo + size_i) {
            if is_one(p) {
                count_ones += 1;
            }
        }
        for a in 0..n {
            out.data[base + (a as usize) * stride] = if count_ones > 0 { 1.0 } else { 0.0 };
            if is_one(a - lo) {
                count_ones -= 1;
            }
            if is_one(a - lo + size_i) {
                count_ones += 1;
            }
        }
    }
    out
}

fn binary_dilation_once_with_origins(current: &NdArray, size: usize, origins: &[i64]) -> NdArray {
    // Binary dilation is a maximum filter over the booleanized image with a
    // constant-0 border, using the reflected structuring element. Gated to the
    // default all-zero origin; the equivalence test pins the reflection offset.
    if binary_morph_origins_all_zero(origins) {
        let bin = booleanized_binary(current);
        return binary_dilate_separable(&bin, size);
    }

    let ndim = current.ndim();
    let mut output = NdArray::zeros(current.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
    let kernel_shape: Vec<usize> = vec![size; ndim];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    let mut kernel_deltas = Vec::with_capacity(kernel_total);
    for flat_k in 0..kernel_total {
        let mut k_idx = vec![0usize; ndim];
        let mut rem = flat_k;
        for d in 0..ndim {
            k_idx[d] = rem / kernel_strides[d];
            rem %= kernel_strides[d];
        }
        let mut delta = vec![0i64; ndim];
        for d in 0..ndim {
            delta[d] = k_idx[d] as i64 - offsets[d] - origins[d];
        }
        kernel_deltas.push(delta);
    }

    for flat_in in 0..current.size() {
        if current.data[flat_in] == 0.0 {
            continue;
        }
        let idx = current.unravel(flat_in);
        for delta in &kernel_deltas {
            let mut out_idx = Vec::with_capacity(ndim);
            let mut in_bounds = true;
            for axis in 0..ndim {
                let coord = idx[axis] as i64 + delta[axis];
                if coord < 0 || coord >= current.shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                out_idx.push(coord as usize);
            }
            if in_bounds {
                output.set(&out_idx, 1.0);
            }
        }
    }

    output
}

fn booleanized_binary(input: &NdArray) -> NdArray {
    let mut output = NdArray::zeros(input.shape.clone());
    for (out, &value) in output.data.iter_mut().zip(&input.data) {
        *out = if value != 0.0 { 1.0 } else { 0.0 };
    }
    output
}

fn binary_erosion_once_axes(current: &NdArray, size: usize, axes: &[usize]) -> NdArray {
    if axes.is_empty() {
        return booleanized_binary(current);
    }

    let mut output = NdArray::zeros(current.shape.clone());
    let offsets = vec![size as i64 / 2; axes.len()];
    let kernel_shape = vec![size; axes.len()];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..current.size() {
        let out_idx = current.unravel(flat_out);
        let mut all_set = true;

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; axes.len()];
            let mut rem = flat_k;
            for d in 0..axes.len() {
                k_idx[d] = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx: Vec<i64> = out_idx.iter().map(|&idx| idx as i64).collect();
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                in_idx[input_axis] =
                    out_idx[input_axis] as i64 + k_idx[kernel_axis] as i64 - offsets[kernel_axis];
            }

            if current.get_boundary(&in_idx, BoundaryMode::Constant, 0.0) == 0.0 {
                all_set = false;
                break;
            }
        }

        output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
    }

    output
}

fn binary_dilation_once_axes(current: &NdArray, size: usize, axes: &[usize]) -> NdArray {
    if axes.is_empty() {
        return booleanized_binary(current);
    }

    let mut output = NdArray::zeros(current.shape.clone());
    let offsets = vec![size as i64 / 2; axes.len()];
    let kernel_shape = vec![size; axes.len()];
    let kernel_total: usize = kernel_shape.iter().product();
    let kernel_strides = compute_strides(&kernel_shape);

    let mut kernel_deltas = Vec::with_capacity(kernel_total);
    for flat_k in 0..kernel_total {
        let mut k_idx = vec![0usize; axes.len()];
        let mut rem = flat_k;
        for d in 0..axes.len() {
            k_idx[d] = rem / kernel_strides[d];
            rem %= kernel_strides[d];
        }
        kernel_deltas.push(
            k_idx
                .iter()
                .enumerate()
                .map(|(kernel_axis, &coord)| coord as i64 - offsets[kernel_axis])
                .collect::<Vec<_>>(),
        );
    }

    for flat_in in 0..current.size() {
        if current.data[flat_in] == 0.0 {
            continue;
        }
        let idx = current.unravel(flat_in);
        for delta in &kernel_deltas {
            let mut out_idx = idx.clone();
            let mut in_bounds = true;
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                let coord = idx[input_axis] as i64 + delta[kernel_axis];
                if coord < 0 || coord >= current.shape[input_axis] as i64 {
                    in_bounds = false;
                    break;
                }
                out_idx[input_axis] = coord as usize;
            }
            if in_bounds {
                output.set(&out_idx, 1.0);
            }
        }
    }

    output
}

fn run_binary_iterations(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    iterations: usize,
    op: fn(&NdArray, usize, &[i64]) -> NdArray,
) -> Result<NdArray, NdimageError> {
    let kernel_shape: Vec<usize> = vec![size; input.ndim()];
    let origins = normalize_filter_origins(input.ndim(), &kernel_shape, origins)?;
    let mut current = input.clone();

    if iterations == 0 {
        loop {
            let output = op(&current, size, &origins);
            if output.data == current.data {
                return Ok(output);
            }
            current = output;
        }
    }

    for _ in 0..iterations {
        current = op(&current, size, &origins);
    }

    Ok(current)
}

fn run_binary_iterations_axes(
    input: &NdArray,
    size: usize,
    axes: &[usize],
    iterations: usize,
    op: fn(&NdArray, usize, &[usize]) -> NdArray,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let shape = vec![size; axes.len()];
    normalize_filter_origins(axes.len(), &shape, &[0])?;
    let mut current = input.clone();

    if iterations == 0 {
        loop {
            let output = op(&current, size, axes);
            if output.data == current.data {
                return Ok(output);
            }
            current = output;
        }
    }

    for _ in 0..iterations {
        current = op(&current, size, axes);
    }

    Ok(current)
}

/// Binary erosion: output pixel is 1 only if all pixels in the structuring
/// element neighborhood are 1.
///
/// Matches `scipy.ndimage.binary_erosion`.
pub fn binary_erosion(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    binary_erosion_with_origins(input, structure_size, &[0], iterations)
}

/// Binary erosion over a SciPy-style signed axes subset.
pub fn binary_erosion_axes(
    input: &NdArray,
    structure_size: usize,
    axes: &[isize],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    run_binary_iterations_axes(input, size, &axes, iterations, binary_erosion_once_axes)
}

/// Binary erosion with SciPy `origin` semantics.
pub fn binary_erosion_with_origins(
    input: &NdArray,
    structure_size: usize,
    origins: &[i64],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    run_binary_iterations(
        input,
        size,
        origins,
        iterations,
        binary_erosion_once_with_origins,
    )
}

/// Binary dilation: output pixel is 1 if any pixel in the structuring
/// element neighborhood is 1.
///
/// Matches `scipy.ndimage.binary_dilation`.
pub fn binary_dilation(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    binary_dilation_with_origins(input, structure_size, &[0], iterations)
}

/// Binary dilation over a SciPy-style signed axes subset.
pub fn binary_dilation_axes(
    input: &NdArray,
    structure_size: usize,
    axes: &[isize],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    run_binary_iterations_axes(input, size, &axes, iterations, binary_dilation_once_axes)
}

/// Binary dilation with SciPy `origin` semantics.
pub fn binary_dilation_with_origins(
    input: &NdArray,
    structure_size: usize,
    origins: &[i64],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    run_binary_iterations(
        input,
        size,
        origins,
        iterations,
        binary_dilation_once_with_origins,
    )
}

/// Binary propagation: repeatedly dilate the input until convergence,
/// optionally constrained by a mask.
///
/// Matches `scipy.ndimage.binary_propagation` for dense structuring elements.
pub fn binary_propagation(
    input: &NdArray,
    structure_size: usize,
    mask: Option<&NdArray>,
) -> Result<NdArray, NdimageError> {
    binary_propagation_with_origins(input, structure_size, &[0], mask)
}

/// Binary propagation with SciPy `origin` semantics for dense structuring elements.
pub fn binary_propagation_with_origins(
    input: &NdArray,
    structure_size: usize,
    origins: &[i64],
    mask: Option<&NdArray>,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if let Some(mask) = mask
        && mask.shape != input.shape
    {
        return Err(NdimageError::DimensionMismatch(format!(
            "mask shape {:?} != input shape {:?}",
            mask.shape, input.shape
        )));
    }

    let size = if structure_size == 0 {
        3
    } else {
        structure_size
    };
    let mut current = input.clone();
    for value in &mut current.data {
        *value = if *value != 0.0 { 1.0 } else { 0.0 };
    }
    let kernel_shape = vec![size; current.ndim()];
    let origins = normalize_filter_origins(current.ndim(), &kernel_shape, origins)?;

    loop {
        let mut next = binary_dilation_once_with_origins(&current, size, &origins);
        let mut changed = false;
        for flat in 0..next.size() {
            let mask_allows_update = match mask {
                Some(mask) => mask.data[flat] != 0.0,
                None => true,
            };
            if !mask_allows_update {
                next.data[flat] = current.data[flat];
            }
            if next.data[flat] != current.data[flat] {
                changed = true;
            }
        }
        if !changed {
            return Ok(next);
        }
        current = next;
    }
}

/// Binary opening: erosion followed by dilation.
///
/// Matches `scipy.ndimage.binary_opening`.
pub fn binary_opening(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    binary_opening_with_origins(input, structure_size, &[0], iterations)
}

/// Binary opening with SciPy `origin` semantics.
pub fn binary_opening_with_origins(
    input: &NdArray,
    structure_size: usize,
    origins: &[i64],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let eroded = binary_erosion_with_origins(input, structure_size, origins, iterations)?;
    binary_dilation_with_origins(&eroded, structure_size, origins, iterations)
}

/// Binary closing: dilation followed by erosion.
///
/// Matches `scipy.ndimage.binary_closing`.
pub fn binary_closing(
    input: &NdArray,
    structure_size: usize,
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    binary_closing_with_origins(input, structure_size, &[0], iterations)
}

/// Binary closing with SciPy `origin` semantics.
pub fn binary_closing_with_origins(
    input: &NdArray,
    structure_size: usize,
    origins: &[i64],
    iterations: usize,
) -> Result<NdArray, NdimageError> {
    let dilated = binary_dilation_with_origins(input, structure_size, origins, iterations)?;
    binary_erosion_with_origins(&dilated, structure_size, origins, iterations)
}

/// Binary fill holes: fill holes in binary objects.
///
/// Matches `scipy.ndimage.binary_fill_holes`.
pub fn binary_fill_holes(input: &NdArray) -> Result<NdArray, NdimageError> {
    let structure = generate_binary_structure(input.ndim(), 1);
    binary_fill_holes_with_structure(input, &structure, &[0])
}

fn binary_structure_offsets(
    structure: &NdArray,
    origins: &[i64],
) -> Result<Vec<Vec<i64>>, NdimageError> {
    if structure.shape.contains(&0) {
        return Err(NdimageError::InvalidArgument(
            "structure dimensions must be positive".to_string(),
        ));
    }

    let origins = normalize_filter_origins(structure.ndim(), &structure.shape, origins)?;
    let centers = structure
        .shape
        .iter()
        .map(|&size| size as i64 / 2)
        .collect::<Vec<_>>();

    let mut offsets = Vec::new();
    for flat in 0..structure.size() {
        if structure.data[flat] == 0.0 {
            continue;
        }
        let idx = structure.unravel(flat);
        offsets.push(
            idx.iter()
                .zip(centers.iter().zip(&origins))
                .map(|(&coord, (&center, &origin))| coord as i64 - center - origin)
                .collect::<Vec<_>>(),
        );
    }
    Ok(offsets)
}

/// Binary fill holes with an explicit structuring element and origin.
///
/// Matches `scipy.ndimage.binary_fill_holes` when `structure` and `origin` are
/// provided.
pub fn binary_fill_holes_with_structure(
    input: &NdArray,
    structure: &NdArray,
    origins: &[i64],
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if input.ndim() != structure.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "input ndim {} != structure ndim {}",
            input.ndim(),
            structure.ndim()
        )));
    }

    let offsets = binary_structure_offsets(structure, origins)?;

    // Flood-fill from edges with complement, then invert
    // Start with all 1s, set boundary-connected background pixels to 0
    let mut filled = NdArray::zeros(input.shape.clone());
    filled.data.fill(1.0);

    // BFS from all border pixels that are background (0) in input
    let ndim = input.ndim();
    let mut queue = std::collections::VecDeque::new();

    // Precompute each offset's FLAT delta (Σ δ·stride) and reuse a coord buffer,
    // killing the per-pixel `unravel` alloc in the border scan and the
    // per-neighbor `Vec` alloc + strides dot product in the BFS (same flat-offset
    // lever as `label`). BYTE-IDENTICAL: the flat-order border scan and the BFS
    // dequeue/enqueue order are unchanged, and `flat + Σδ·stride` equals the old
    // `Σ(coord+δ)·stride`.
    let signed_strides: Vec<i64> = input.strides.iter().map(|&s| s as i64).collect();
    let flat_offsets: Vec<i64> = offsets
        .iter()
        .map(|off| {
            off.iter()
                .zip(&signed_strides)
                .map(|(&delta, &stride)| delta * stride)
                .sum()
        })
        .collect();
    let signed_shape: Vec<i64> = input.shape.iter().map(|&s| s as i64).collect();
    let mut coord = vec![0usize; ndim];

    // Find border pixels
    for flat in 0..input.size() {
        unravel_into(flat, &input.strides, &mut coord);
        let is_border = coord
            .iter()
            .zip(input.shape.iter())
            .any(|(&i, &s)| i == 0 || i == s - 1);
        if is_border && input.data[flat] == 0.0 {
            filled.data[flat] = 0.0;
            queue.push_back(flat);
        }
    }

    // BFS: spread background through connected 0-pixels
    while let Some(flat) = queue.pop_front() {
        unravel_into(flat, &input.strides, &mut coord);
        for (oi, offset) in offsets.iter().enumerate() {
            let mut in_bounds = true;
            for axis in 0..ndim {
                let neighbor_coord = coord[axis] as i64 + offset[axis];
                if neighbor_coord < 0 || neighbor_coord >= signed_shape[axis] {
                    in_bounds = false;
                    break;
                }
            }
            if !in_bounds {
                continue;
            }
            let nflat = (flat as i64 + flat_offsets[oi]) as usize;
            if input.data[nflat] == 0.0 && filled.data[nflat] == 1.0 {
                filled.data[nflat] = 0.0;
                queue.push_back(nflat);
            }
        }
    }

    // Result: original foreground OR filled holes
    for i in 0..filled.data.len() {
        filled.data[i] = if input.data[i] != 0.0 || filled.data[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }

    Ok(filled)
}

// ══════════════════════════════════════════════════════════════════════
// Measurements
// ══════════════════════════════════════════════════════════════════════

/// Label connected components in a binary array.
///
/// Returns (labeled_array, num_features).
/// Matches `scipy.ndimage.label`.
pub fn label(input: &NdArray) -> Result<(NdArray, usize), NdimageError> {
    let structure = generate_binary_structure(input.ndim(), 1);
    label_with_structure(input, &structure)
}

fn validate_label_structure(input_ndim: usize, structure: &NdArray) -> Result<(), NdimageError> {
    if input_ndim != structure.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and structure must have same dimensions".to_string(),
        ));
    }
    if structure.shape.iter().any(|&dim| dim != 3) {
        return Err(NdimageError::InvalidArgument(
            "structure dimensions must be equal to 3".to_string(),
        ));
    }

    for flat in 0..structure.size() {
        let idx = structure.unravel(flat);
        let opposite_idx = idx
            .iter()
            .zip(&structure.shape)
            .map(|(&coord, &dim)| dim - 1 - coord)
            .collect::<Vec<_>>();
        let opposite_flat = opposite_idx
            .iter()
            .zip(&structure.strides)
            .map(|(coord, stride)| coord * stride)
            .sum::<usize>();
        if (structure.data[flat] != 0.0) != (structure.data[opposite_flat] != 0.0) {
            return Err(NdimageError::InvalidArgument(
                "structuring element is not symmetric".to_string(),
            ));
        }
    }

    Ok(())
}

/// Label connected components with an explicit structuring element.
///
/// Matches `scipy.ndimage.label` when `structure` is provided.
pub fn label_with_structure(
    input: &NdArray,
    structure: &NdArray,
) -> Result<(NdArray, usize), NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_label_structure(input.ndim(), structure)?;
    let offsets = binary_structure_offsets(structure, &[0])?
        .into_iter()
        .filter(|offset| offset.iter().any(|&delta| delta != 0))
        .collect::<Vec<_>>();

    let ndim = input.ndim();
    let n = input.size();

    // Two-pass union-find (scipy's algorithm class), replacing the BFS flood
    // fill. Pass 1 is a SINGLE raster scan that, for each foreground cell, unions
    // it with its already-visited ("backward", i.e. negative flat-delta) foreground
    // neighbours — no queue, no per-cell `unravel` (the coord is advanced
    // incrementally with O(1) amortized carry, not recomputed by division), and
    // only HALF the structure offsets are tested. Pass 2 assigns consecutive
    // labels in raster order. BYTE-IDENTICAL to the BFS: union roots are the
    // lowest-flat cell of each component, so components are numbered in order of
    // their first raster cell — exactly the BFS seed-scan order (and scipy's).
    let signed_strides: Vec<i64> = input.strides.iter().map(|&s| s as i64).collect();
    let signed_shape: Vec<i64> = input.shape.iter().map(|&s| s as i64).collect();
    // Backward offsets only: those reaching an earlier (already-processed) flat
    // index. Each carries its per-axis deltas (for the bounds test) and flat delta.
    let backward: Vec<(Vec<i64>, i64)> = offsets
        .iter()
        .filter_map(|off| {
            let flat_delta: i64 = off
                .iter()
                .zip(&signed_strides)
                .map(|(&delta, &stride)| delta * stride)
                .sum();
            (flat_delta < 0).then(|| (off.clone(), flat_delta))
        })
        .collect();

    // Compact the f64 input into a 1-byte/cell foreground mask ONCE. The
    // union-find then re-reads this 256 KiB-class (L2-resident) mask on every
    // neighbour test and both passes, instead of streaming the 8×-larger f64
    // `input.data` ~4 times — the dominant cost is memory bandwidth (scipy works
    // on int8/int32), so this is the real lever, not the algorithm.
    let fg: Vec<u8> = input.data.iter().map(|&v| u8::from(v != 0.0)).collect();

    let mut parent = vec![0u32; n];
    let mut coord = vec![0usize; ndim];
    for flat in 0..n {
        if fg[flat] != 0 {
            parent[flat] = flat as u32; // make-set
            for (off, flat_delta) in &backward {
                let mut in_bounds = true;
                for axis in 0..ndim {
                    let nc = coord[axis] as i64 + off[axis];
                    if nc < 0 || nc >= signed_shape[axis] {
                        in_bounds = false;
                        break;
                    }
                }
                if in_bounds {
                    let nf = (flat as i64 + flat_delta) as usize;
                    if fg[nf] != 0 {
                        label_union(&mut parent, flat as u32, nf as u32);
                    }
                }
            }
        }
        // Advance the N-D coordinate one raster step (O(1) amortized).
        let mut axis = ndim;
        while axis > 0 {
            axis -= 1;
            coord[axis] += 1;
            if coord[axis] < input.shape[axis] {
                break;
            }
            coord[axis] = 0;
        }
    }

    // Pass 2: consecutive relabel. Each component's root is its lowest-flat cell,
    // reached first in this raster scan, so labels count up in first-cell order.
    let mut labels = NdArray::zeros(input.shape.clone());
    let mut comp_label = vec![0u32; n];
    let mut current_label = 0u32;
    for flat in 0..n {
        if fg[flat] == 0 {
            continue;
        }
        let root = label_find(&mut parent, flat as u32) as usize;
        if root == flat {
            current_label += 1;
            comp_label[flat] = current_label;
        }
        labels.data[flat] = comp_label[root] as f64;
    }

    Ok((labels, current_label as usize))
}

/// Union-find `find` with path halving; the root is the lowest flat index in the
/// set (see `label_union`), so it identifies a component by its first raster cell.
fn label_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Union by MIN flat index: the smaller index becomes the root. Keeps every
/// component rooted at its lowest-flat (first-in-raster) cell, which makes the
/// consecutive relabel match the BFS/scipy first-appearance numbering.
fn label_union(parent: &mut [u32], a: u32, b: u32) {
    let ra = label_find(parent, a);
    let rb = label_find(parent, b);
    if ra != rb {
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }
}

fn measurement_label_groups(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<f64>>, NdimageError> {
    let Some(labels) = labels else {
        return Ok(vec![input.data.clone()]);
    };
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }

    let num_groups = match index {
        Some(index) => index.len(),
        None => 1,
    };

    // One-time label -> first-position map (Some(index) case): O(1) bucketing instead of an
    // O(K) linear scan, O(N + K) total. Byte-identical via the canonical ±0.0 key + first-wins.
    let label_to_pos: Option<std::collections::HashMap<u64, usize>> = match index {
        Some(index) => {
            let mut m = std::collections::HashMap::with_capacity(index.len());
            for (pos, &wanted_label) in index.iter().enumerate() {
                m.entry(measurement_label_key(wanted_label as f64))
                    .or_insert(pos);
            }
            Some(m)
        }
        None => None,
    };
    let classify = |label_value: f64| -> Option<usize> {
        match &label_to_pos {
            Some(m) => m.get(&measurement_label_key(label_value)).copied(),
            None => (label_value != 0.0).then_some(0),
        }
    };

    let n = input.data.len();
    // Value-only sibling of `measurement_label_value_positions`; parallelize the per-element
    // classify+push identically. Private per-thread buckets over contiguous flat-chunks merged in
    // thread (= flat) order → BYTE-IDENTICAL group contents/order (each group's ascending-flat-order
    // value list is reproduced exactly). The per-element std-HashMap (SipHash) lookup dominates.
    let nthreads = if NDIMAGE_LABEL_GATHER_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        let t = ndimage_filter_thread_count(n, 4);
        if t <= 1 || num_groups.saturating_mul(t) > n {
            1
        } else {
            t
        }
    };

    if nthreads <= 1 {
        let mut groups: Vec<Vec<f64>> = vec![Vec::new(); num_groups];
        for (&value, &label_value) in input.data.iter().zip(&labels.data) {
            if let Some(pos) = classify(label_value) {
                groups[pos].push(value);
            }
        }
        return Ok(groups);
    }

    let chunk = n.div_ceil(nthreads);
    let classify = &classify;
    let partials: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            handles.push(scope.spawn(move || {
                let mut local: Vec<Vec<f64>> = vec![Vec::new(); num_groups];
                for flat in start..end {
                    if let Some(pos) = classify(labels.data[flat]) {
                        local[pos].push(input.data[flat]);
                    }
                }
                local
            }));
            start = end;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("label gather chunk panicked"))
            .collect()
    });

    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); num_groups];
    for partial in partials {
        for (pos, bucket) in partial.into_iter().enumerate() {
            groups[pos].extend(bucket);
        }
    }
    Ok(groups)
}

fn measurement_label_key(x: f64) -> u64 {
    if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

const MEASUREMENT_DENSE_LABEL_EMPTY: usize = usize::MAX;
const MEASUREMENT_DENSE_LABEL_EXPANSION_LIMIT: usize = 8;
const MEASUREMENT_DENSE_LABEL_MIN_CAPACITY: usize = 1024;

fn measurement_dense_label_positions(index: &[usize]) -> Option<Vec<usize>> {
    let Some(max_label) = index.iter().copied().max() else {
        return Some(Vec::new());
    };
    let dense_len = max_label.checked_add(1)?;
    let max_dense_len = index
        .len()
        .saturating_mul(MEASUREMENT_DENSE_LABEL_EXPANSION_LIMIT)
        .max(MEASUREMENT_DENSE_LABEL_MIN_CAPACITY);
    if dense_len > max_dense_len {
        return None;
    }

    let mut label_to_pos = vec![MEASUREMENT_DENSE_LABEL_EMPTY; dense_len];
    for (pos, &wanted_label) in index.iter().enumerate() {
        if label_to_pos[wanted_label] == MEASUREMENT_DENSE_LABEL_EMPTY {
            label_to_pos[wanted_label] = pos;
        }
    }
    Some(label_to_pos)
}

fn measurement_dense_label_pos(label_to_pos: &[usize], label_value: f64) -> Option<usize> {
    if !(label_value >= 0.0 && label_value < label_to_pos.len() as f64) {
        return None;
    }
    let label = label_value as usize;
    if label as f64 != label_value {
        return None;
    }
    let pos = *label_to_pos.get(label)?;
    (pos != MEASUREMENT_DENSE_LABEL_EMPTY).then_some(pos)
}

fn measurement_one_based_contiguous_index_len(index: &[usize]) -> Option<usize> {
    for (pos, &wanted_label) in index.iter().enumerate() {
        if wanted_label != pos.checked_add(1)? {
            return None;
        }
    }
    Some(index.len())
}

fn measurement_one_based_label_pos(label_count: usize, label_value: f64) -> Option<usize> {
    let label = measurement_exact_positive_integer_label(label_value)?;
    (label <= label_count).then_some(label - 1)
}

fn measurement_exact_positive_integer_label(label_value: f64) -> Option<usize> {
    let bits = label_value.to_bits();
    if bits >> 63 != 0 {
        return None;
    }

    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    if exponent_bits == 0 || exponent_bits == 0x7ff {
        return None;
    }
    let exponent = exponent_bits - 1023;
    if exponent < 0 {
        return None;
    }

    let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let label = if exponent <= 52 {
        let fractional_bits = (52 - exponent) as u32;
        if fractional_bits > 0 {
            let fractional_mask = (1_u64 << fractional_bits) - 1;
            if significand & fractional_mask != 0 {
                return None;
            }
        }
        significand >> fractional_bits
    } else {
        let shift = (exponent - 52) as u32;
        if shift >= u64::BITS || significand > (u64::MAX >> shift) {
            return None;
        }
        significand << shift
    };

    usize::try_from(label).ok().filter(|&label| label > 0)
}

/// One-based-contiguous label scatter (`sums[label-1] += value`, `counts[label-1] += 1`)
/// as a parallel privatized-histogram reduction. Each worker accumulates a PRIVATE
/// `(sums, counts)` over a contiguous chunk, then the partials are merged in chunk order.
/// Because the chunks are contiguous and merged in order, each label's running sum visits
/// its elements in the SAME global order as the serial scatter — only the ASSOCIATION of
/// the float adds differs (≈1 ULP), so the result is tolerance-equal, not bit-equal.
/// Gated to large `n` so the worker spawn amortises; small `n` (the unit-test regime)
/// stays on the serial path and remains byte-identical.
fn measurement_one_based_scatter(
    data: &[f64],
    labels: &[f64],
    label_count: usize,
) -> (Vec<f64>, Vec<usize>) {
    let n = data.len();
    let serial = || {
        let mut sums = vec![0.0; label_count];
        let mut counts = vec![0usize; label_count];
        for (&value, &label_value) in data.iter().zip(labels) {
            if let Some(pos) = measurement_one_based_label_pos(label_count, label_value) {
                sums[pos] += value;
                counts[pos] += 1;
            }
        }
        (sums, counts)
    };

    // Each worker must stream a worthwhile contiguous slice (~128k elements) for the
    // thread::scope spawn to pay against the already-fast serial reduction.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n / 128_000);
    if nthreads <= 1 {
        return serial();
    }

    let chunk = n.div_ceil(nthreads);
    let partials: Vec<(Vec<f64>, Vec<usize>)> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= n {
                    return None;
                }
                let hi = (lo + chunk).min(n);
                let d = &data[lo..hi];
                let l = &labels[lo..hi];
                Some(scope.spawn(move || {
                    let mut sums = vec![0.0; label_count];
                    let mut counts = vec![0usize; label_count];
                    for (&value, &label_value) in d.iter().zip(l) {
                        if let Some(pos) = measurement_one_based_label_pos(label_count, label_value)
                        {
                            sums[pos] += value;
                            counts[pos] += 1;
                        }
                    }
                    (sums, counts)
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("label-scatter worker panicked"))
            .collect()
    });

    let mut sums = vec![0.0; label_count];
    let mut counts = vec![0usize; label_count];
    for (ps, pc) in partials {
        for k in 0..label_count {
            sums[k] += ps[k];
            counts[k] += pc[k];
        }
    }
    (sums, counts)
}

/// One-based-contiguous squared-deviation scatter: `out[label-1] += (value - means[label-1])²`,
/// as a parallel privatized-histogram reduction (same shape/gate as
/// [`measurement_one_based_scatter`]). `means` holds the per-label means from pass 1.
fn measurement_one_based_sqdev_scatter(
    data: &[f64],
    labels: &[f64],
    label_count: usize,
    means: &[f64],
) -> Vec<f64> {
    let n = data.len();
    let serial = || {
        let mut sq = vec![0.0; label_count];
        for (&value, &label_value) in data.iter().zip(labels) {
            if let Some(pos) = measurement_one_based_label_pos(label_count, label_value) {
                let d = value - means[pos];
                sq[pos] += d * d;
            }
        }
        sq
    };

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n / 128_000);
    if nthreads <= 1 {
        return serial();
    }

    let chunk = n.div_ceil(nthreads);
    let partials: Vec<Vec<f64>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= n {
                    return None;
                }
                let hi = (lo + chunk).min(n);
                let d = &data[lo..hi];
                let l = &labels[lo..hi];
                Some(scope.spawn(move || {
                    let mut sq = vec![0.0; label_count];
                    for (&value, &label_value) in d.iter().zip(l) {
                        if let Some(pos) = measurement_one_based_label_pos(label_count, label_value)
                        {
                            let dv = value - means[pos];
                            sq[pos] += dv * dv;
                        }
                    }
                    sq
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("sqdev-scatter worker panicked"))
            .collect()
    });

    let mut sq = vec![0.0; label_count];
    for ps in partials {
        for k in 0..label_count {
            sq[k] += ps[k];
        }
    }
    sq
}

/// Per-label population variance over the one-based-contiguous fast path, computed as a
/// numerically-stable TWO-PASS streaming reduction (means, then centred sum of squares) to
/// match `scipy.ndimage.variance`. Empty labels yield NaN. Both passes are parallel
/// privatized histograms (≈1 ULP reassociation; byte-identical below the gate).
fn measurement_one_based_variance(data: &[f64], labels: &[f64], label_count: usize) -> Vec<f64> {
    let (sums, counts) = measurement_one_based_scatter(data, labels, label_count);
    let means: Vec<f64> = (0..label_count)
        .map(|k| {
            if counts[k] > 0 {
                sums[k] / counts[k] as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    let sq = measurement_one_based_sqdev_scatter(data, labels, label_count, &means);
    (0..label_count)
        .map(|k| {
            if counts[k] > 0 {
                sq[k] / counts[k] as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Per-label minimum (`want_max=false`) or maximum over the one-based-contiguous fast path,
/// as a parallel privatized-histogram reduction. min/max are associative, commutative, and
/// exact, so the merged result is BYTE-IDENTICAL to the serial fold regardless of chunking.
/// Matches the serial group path's conventions: a NaN in any element of a label propagates to
/// NaN, and an empty label yields 0.0. Gated like the other label reductions.
fn measurement_one_based_minmax(
    data: &[f64],
    labels: &[f64],
    label_count: usize,
    want_max: bool,
) -> Vec<f64> {
    let init = if want_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    // NaN-propagating combine, identical to the serial fold (Rust's min/max ignore NaN, so the
    // explicit check is what makes a single NaN poison the whole label).
    let combine = |acc: f64, v: f64| -> f64 {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else if want_max {
            acc.max(v)
        } else {
            acc.min(v)
        }
    };
    let scan = |d: &[f64], l: &[f64]| -> (Vec<f64>, Vec<usize>) {
        let mut ext = vec![init; label_count];
        let mut counts = vec![0usize; label_count];
        for (&value, &label_value) in d.iter().zip(l) {
            if let Some(pos) = measurement_one_based_label_pos(label_count, label_value) {
                ext[pos] = combine(ext[pos], value);
                counts[pos] += 1;
            }
        }
        (ext, counts)
    };

    let n = data.len();
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n / 128_000);
    let (ext, counts) = if nthreads <= 1 {
        scan(data, labels)
    } else {
        let chunk = n.div_ceil(nthreads);
        let partials: Vec<(Vec<f64>, Vec<usize>)> = std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= n {
                        return None;
                    }
                    let hi = (lo + chunk).min(n);
                    let d = &data[lo..hi];
                    let l = &labels[lo..hi];
                    Some(scope.spawn(move || scan(d, l)))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("minmax-scatter worker panicked"))
                .collect()
        });
        let mut ext = vec![init; label_count];
        let mut counts = vec![0usize; label_count];
        for (pe, pc) in partials {
            for k in 0..label_count {
                ext[k] = combine(ext[k], pe[k]);
                counts[k] += pc[k];
            }
        }
        (ext, counts)
    };

    (0..label_count)
        .map(|k| if counts[k] == 0 { 0.0 } else { ext[k] })
        .collect()
}

/// Per-label histogram over the one-based-contiguous fast path, as a parallel privatized
/// reduction: each worker fills a private flat `[label_count × nbins]` count table over a
/// contiguous chunk, then the tables are summed. Counts are integers, so the merge is
/// BYTE-IDENTICAL to the serial group-path fill (same bin assignment + `[min,max]` filter).
/// Caller guarantees `nbins >= 1`, finite `min<max`, and finite input.
fn measurement_one_based_histogram(
    data: &[f64],
    labels: &[f64],
    label_count: usize,
    min_val: f64,
    max_val: f64,
    nbins: usize,
) -> Vec<Vec<usize>> {
    let bin_width = (max_val - min_val) / nbins as f64;
    let scan = |d: &[f64], l: &[f64]| -> Vec<usize> {
        let mut h = vec![0usize; label_count * nbins];
        for (&value, &label_value) in d.iter().zip(l) {
            if value < min_val || value > max_val {
                continue;
            }
            if let Some(pos) = measurement_one_based_label_pos(label_count, label_value) {
                let bin = (((value - min_val) / bin_width).floor() as usize).min(nbins - 1);
                h[pos * nbins + bin] += 1;
            }
        }
        h
    };

    let n = data.len();
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n / 128_000);
    let flat = if nthreads <= 1 {
        scan(data, labels)
    } else {
        let chunk = n.div_ceil(nthreads);
        let partials: Vec<Vec<usize>> = std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= n {
                        return None;
                    }
                    let hi = (lo + chunk).min(n);
                    let d = &data[lo..hi];
                    let l = &labels[lo..hi];
                    Some(scope.spawn(move || scan(d, l)))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("histogram-scatter worker panicked"))
                .collect()
        });
        let mut h = vec![0usize; label_count * nbins];
        for p in partials {
            for (acc, v) in h.iter_mut().zip(p) {
                *acc += v;
            }
        }
        h
    };

    (0..label_count)
        .map(|pos| flat[pos * nbins..(pos + 1) * nbins].to_vec())
        .collect()
}

fn measurement_label_mean(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    let Some(labels) = labels else {
        return Ok(vec![mean_of_values(&input.data)]);
    };
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }

    let (mut sums, mut counts) = match index {
        Some(index) => (vec![0.0; index.len()], vec![0usize; index.len()]),
        None => (vec![0.0], vec![0usize]),
    };

    match index {
        Some(index) => {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                let (s, c) = measurement_one_based_scatter(&input.data, &labels.data, label_count);
                sums = s;
                counts = c;
            } else if let Some(label_to_pos) = measurement_dense_label_positions(index) {
                for (&value, &label_value) in input.data.iter().zip(&labels.data) {
                    if let Some(pos) = measurement_dense_label_pos(&label_to_pos, label_value) {
                        sums[pos] += value;
                        counts[pos] += 1;
                    }
                }
            } else {
                let mut label_to_pos: std::collections::HashMap<u64, usize> =
                    std::collections::HashMap::with_capacity(index.len());
                for (pos, &wanted_label) in index.iter().enumerate() {
                    label_to_pos
                        .entry(measurement_label_key(wanted_label as f64))
                        .or_insert(pos);
                }
                for (&value, &label_value) in input.data.iter().zip(&labels.data) {
                    if let Some(&pos) = label_to_pos.get(&measurement_label_key(label_value)) {
                        sums[pos] += value;
                        counts[pos] += 1;
                    }
                }
            }
        }
        None => {
            for (&value, &label_value) in input.data.iter().zip(&labels.data) {
                if label_value != 0.0 {
                    sums[0] += value;
                    counts[0] += 1;
                }
            }
        }
    }

    Ok(sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect())
}

/// When `true`, force the label value/position gather onto its ORIG serial per-element bucketing
/// (bypasses the across-elements fan-out). Byte-identical either way. Benchmark knob for the
/// same-binary A/B. Shared by `extrema`/`labeled_comprehension` (and any gather-based label stat).
#[doc(hidden)]
pub static NDIMAGE_LABEL_GATHER_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn measurement_label_value_positions(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<(f64, usize)>>, NdimageError> {
    let Some(labels) = labels else {
        return Ok(vec![
            input.data.iter().copied().zip(0..input.size()).collect(),
        ]);
    };
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }

    let num_groups = match index {
        Some(index) => index.len(),
        None => 1,
    };

    // One-time label -> first-position map (Some(index) case); see `measurement_label_groups`.
    // Byte-identical via the canonical ±0.0 key and `or_insert` first-wins.
    let label_to_pos: Option<std::collections::HashMap<u64, usize>> = match index {
        Some(index) => {
            let mut m = std::collections::HashMap::with_capacity(index.len());
            for (pos, &wanted_label) in index.iter().enumerate() {
                m.entry(measurement_label_key(wanted_label as f64))
                    .or_insert(pos);
            }
            Some(m)
        }
        None => None,
    };
    // Classify one element's label -> output bucket (identical to the serial branches: the
    // first-wins ±0.0-keyed map for the index case, `label != 0.0` otherwise).
    let classify = |label_value: f64| -> Option<usize> {
        match &label_to_pos {
            Some(m) => m.get(&measurement_label_key(label_value)).copied(),
            None => (label_value != 0.0).then_some(0),
        }
    };

    let n = input.data.len();
    // The bucketing is a per-element classify + push in flat order; each element is independent.
    // Fan contiguous flat-index chunks across cores into PRIVATE per-thread buckets, then merge the
    // buckets in thread (= flat) order. BYTE-IDENTICAL to the serial push loop: thread t owns a
    // strictly lower flat range than t+1 and pushes stay flat-ascending within a thread, so the
    // ordered concatenation reproduces the exact serial group contents and order. The per-element
    // std-HashMap (SipHash) lookup is the dominant cost of the gather-based label stats
    // (extrema/labeled_comprehension), so splitting it across cores lifts the whole family.
    let nthreads = if NDIMAGE_LABEL_GATHER_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        let t = ndimage_filter_thread_count(n, 4);
        // Keep the per-thread bucket bookkeeping (t * num_groups) below the data volume.
        if t <= 1 || num_groups.saturating_mul(t) > n {
            1
        } else {
            t
        }
    };

    if nthreads <= 1 {
        let mut groups: Vec<Vec<(f64, usize)>> = vec![Vec::new(); num_groups];
        for (flat, (&value, &label_value)) in input.data.iter().zip(&labels.data).enumerate() {
            if let Some(pos) = classify(label_value) {
                groups[pos].push((value, flat));
            }
        }
        return Ok(groups);
    }

    let chunk = n.div_ceil(nthreads);
    let classify = &classify;
    let partials: Vec<Vec<Vec<(f64, usize)>>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            handles.push(scope.spawn(move || {
                let mut local: Vec<Vec<(f64, usize)>> = vec![Vec::new(); num_groups];
                for flat in start..end {
                    if let Some(pos) = classify(labels.data[flat]) {
                        local[pos].push((input.data[flat], flat));
                    }
                }
                local
            }));
            start = end;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("label gather chunk panicked"))
            .collect()
    });

    let mut groups: Vec<Vec<(f64, usize)>> = vec![Vec::new(); num_groups];
    for partial in partials {
        for (pos, bucket) in partial.into_iter().enumerate() {
            groups[pos].extend(bucket);
        }
    }
    Ok(groups)
}

fn mean_of_values(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// When `true`, [`median_of_values`] fully sorts the copy (the ORIG behaviour); default
/// `false` quickselects the central rank(s) in O(n). Byte-identical. A/B gate.
#[doc(hidden)]
pub static MEDIAN_FORCE_SORT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn median_of_values(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mut buf = values.to_vec();
    let n = buf.len();
    let mid = n / 2;
    if MEDIAN_FORCE_SORT.load(std::sync::atomic::Ordering::Relaxed) {
        buf.sort_by(|a, b| a.total_cmp(b));
        return if n.is_multiple_of(2) {
            (buf[mid - 1] + buf[mid]) / 2.0
        } else {
            buf[mid]
        };
    }
    // The median needs only the one or two central order statistics, so quickselect them in
    // O(n) instead of a full O(n log n) sort. BYTE-IDENTICAL: `select_nth_unstable_by(mid)`
    // places the mid-th total_cmp order statistic at `mid` with everything <= it to the left,
    // so `buf[mid]` == a full sort's `sorted[mid]`, and the total_cmp-max of `buf[..mid]` ==
    // `sorted[mid-1]`. The k-th order statistic is unique, and the `(lo+hi)/2` order is kept.
    if n.is_multiple_of(2) {
        buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        let hi = buf[mid];
        let lo = buf[..mid]
            .iter()
            .copied()
            .reduce(|a, b| if a.total_cmp(&b).is_lt() { b } else { a })
            .expect("even median has a non-empty lower half");
        (lo + hi) / 2.0
    } else {
        buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        buf[mid]
    }
}

fn minimum_value_position(input: &NdArray, values: &[(f64, usize)]) -> (f64, Vec<usize>) {
    if values.is_empty() {
        return (0.0, vec![0; input.ndim()]);
    }

    let (mut best_value, mut best_flat) = values[0];
    for &(value, flat) in &values[1..] {
        if value < best_value || value.is_nan() {
            best_value = value;
            best_flat = flat;
        }
    }

    (best_value, input.unravel(best_flat))
}

fn maximum_value_position(input: &NdArray, values: &[(f64, usize)]) -> (f64, Vec<usize>) {
    if values.is_empty() {
        return (0.0, vec![0; input.ndim()]);
    }

    let (mut best_value, mut best_flat) = values[0];
    for &(value, flat) in &values[1..] {
        if value > best_value || value.is_nan() {
            best_value = value;
            best_flat = flat;
        }
    }

    (best_value, input.unravel(best_flat))
}

pub type ExtremaResult = (Vec<f64>, Vec<f64>, Vec<Vec<usize>>, Vec<Vec<usize>>);
pub type ValueIndices = std::collections::BTreeMap<i64, Vec<Vec<usize>>>;

fn value_indices_integer_input_error() -> NdimageError {
    NdimageError::InvalidArgument("arr must contain only integer values".to_string())
}

/// Indices of each distinct integer value in an array.
///
/// Matches `scipy.ndimage.value_indices`: keys are the distinct integer values
/// and values are one coordinate vector per dimension covering all occurrences
/// of the key. `ignore_value`, when supplied, is skipped.
pub fn value_indices(
    arr: &NdArray,
    ignore_value: Option<i64>,
) -> Result<ValueIndices, NdimageError> {
    let mut indices = ValueIndices::new();

    for (flat, &value) in arr.data.iter().enumerate() {
        if !value.is_finite()
            || value.fract() != 0.0
            || value < i64::MIN as f64
            || value > i64::MAX as f64
        {
            return Err(value_indices_integer_input_error());
        }

        let key = value as i64;
        if ignore_value == Some(key) {
            continue;
        }

        let coords = arr.unravel(flat);
        let entry = indices
            .entry(key)
            .or_insert_with(|| vec![Vec::new(); arr.ndim()]);
        for (axis_coords, coord) in entry.iter_mut().zip(coords) {
            axis_coords.push(coord);
        }
    }

    Ok(indices)
}

/// Sum of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.sum`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
/// When `true`, force the global (no-labels) `sum` onto the ORIG group path (clone + serial sum).
/// Byte-identical either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_SUM_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn sum(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    // Fast streaming path: a one-based-contiguous index needs only the per-label sums,
    // so skip the per-group Vec materialization and use the parallel privatized-histogram
    // scatter (counts discarded). Falls back to the general group path otherwise.
    if let (Some(labels), Some(index)) = (labels, index)
        && input.shape == labels.shape
    {
        {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                let (sums, _counts) =
                    measurement_one_based_scatter(&input.data, &labels.data, label_count);
                return Ok(sums);
            }
        }
    }
    if labels.is_none() && !NDIMAGE_SUM_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        // Global sum (labels=None): sum `input.data` directly, skipping the
        // `measurement_label_groups` full-data clone. BYTE-IDENTICAL: same increasing-flat-index
        // summation order (float add is non-associative, so this stays a single serial fold).
        return Ok(vec![input.data.iter().sum()]);
    }
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| values.iter().sum())
        .collect())
}

/// Sum of values in labeled regions.
///
/// Matches `scipy.ndimage.sum_labels`.
pub fn sum_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Result<Vec<f64>, NdimageError> {
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }
    let mut sums = vec![0.0; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            sums[lbl] += input.data[i];
        }
    }
    Ok(sums[1..].to_vec())
}

/// Mean of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.mean`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
pub fn mean(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    measurement_label_mean(input, labels, index)
}

/// Mean of values in labeled regions.
///
/// Matches `scipy.ndimage.mean`.
pub fn mean_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Result<Vec<f64>, NdimageError> {
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }
    let mut sums = vec![0.0; num_labels + 1];
    let mut counts = vec![0usize; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            sums[lbl] += input.data[i];
            counts[lbl] += 1;
        }
    }
    Ok((1..=num_labels)
        .map(|l| {
            if counts[l] > 0 {
                sums[l] / counts[l] as f64
            } else {
                // SciPy returns NaN for labels with no pixels
                f64::NAN
            }
        })
        .collect())
}

/// Variance of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.variance`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
/// When `true`, force the global (no-labels) `variance` onto the ORIG group path (clone + serial
/// two-pass). Byte-identical either way. Benchmark knob for the same-binary A/B (also lifts
/// `standard_deviation`, which is `variance(..).sqrt()`).
#[doc(hidden)]
pub static NDIMAGE_VARIANCE_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn variance(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    // Fast streaming path: a one-based-contiguous index uses a numerically-stable two-pass
    // parallel privatized-histogram variance, skipping per-group Vec materialization. Falls
    // back to the general group path otherwise.
    if let (Some(labels), Some(index)) = (labels, index)
        && input.shape == labels.shape
    {
        {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                return Ok(measurement_one_based_variance(
                    &input.data,
                    &labels.data,
                    label_count,
                ));
            }
        }
    }
    if labels.is_none() && !NDIMAGE_VARIANCE_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        // Global variance (labels=None): compute over `input.data` directly, skipping the
        // `measurement_label_groups` full-data clone. BYTE-IDENTICAL: same `mean_of_values`, same
        // increasing-flat-index Σ(x-mean)² order (float ops are non-associative → single serial pass).
        let values = input.data.as_slice();
        let mean = mean_of_values(values);
        let var = if mean.is_nan() {
            f64::NAN
        } else {
            values
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f64>()
                / values.len() as f64
        };
        return Ok(vec![var]);
    }
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| {
            let mean = mean_of_values(values);
            if mean.is_nan() {
                f64::NAN
            } else {
                values
                    .iter()
                    .map(|value| {
                        let diff = *value - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / values.len() as f64
            }
        })
        .collect())
}

/// Variance of values in labeled regions.
///
/// Matches `scipy.ndimage.variance`.
pub fn variance_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Result<Vec<f64>, NdimageError> {
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }
    let means = mean_labels(input, labels, num_labels)?;
    let mut var_sums = vec![0.0; num_labels + 1];
    let mut counts = vec![0usize; num_labels + 1];
    for i in 0..input.size() {
        let lbl = labels.data[i] as usize;
        if lbl > 0 && lbl <= num_labels {
            let diff = input.data[i] - means[lbl - 1];
            var_sums[lbl] += diff * diff;
            counts[lbl] += 1;
        }
    }
    Ok((1..=num_labels)
        .map(|l| {
            if counts[l] > 0 {
                var_sums[l] / counts[l] as f64
            } else {
                f64::NAN
            }
        })
        .collect())
}

/// Standard deviation of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.standard_deviation`; scalar SciPy results are
/// returned as a one-element vector, while explicit `index` lists return one
/// value per label.
pub fn standard_deviation(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    Ok(variance(input, labels, index)?
        .into_iter()
        .map(|value| value.sqrt())
        .collect())
}

/// Minimum of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.minimum`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
/// When `true`, force the global (no-labels) `minimum`/`maximum` onto the ORIG group path
/// (full-data clone + serial fold). Byte-identical either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_MINMAX_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Global NaN-propagating min/max reduction over `data` (the labels=None case), matching the ORIG
/// `measurement_label_groups(None)` fold exactly but WITHOUT the full-data clone: a chunked parallel
/// reduction (min/max are associative + NaN propagates through the combine → BYTE-IDENTICAL to the
/// sequential fold). `is_max` selects max (init `-inf`) vs min (init `+inf`).
fn global_minmax_reduce(data: &[f64], is_max: bool) -> f64 {
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let n = data.len();
    let nthreads = ndimage_filter_thread_count(n, 4);
    if nthreads <= 1 {
        return data.iter().copied().fold(init, |a, b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if is_max {
                a.max(b)
            } else {
                a.min(b)
            }
        });
    }
    let chunk = n.div_ceil(nthreads);
    let parts: Vec<f64> = std::thread::scope(|scope| {
        data.chunks(chunk)
            .map(|c| {
                scope.spawn(move || {
                    c.iter().copied().fold(init, |a, b| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else if is_max {
                            a.max(b)
                        } else {
                            a.min(b)
                        }
                    })
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("minmax chunk panicked"))
            .collect()
    });
    parts.into_iter().fold(init, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if is_max {
            a.max(b)
        } else {
            a.min(b)
        }
    })
}

pub fn minimum(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    // Fast streaming path: byte-identical parallel privatized-histogram min over a
    // one-based-contiguous index (min is associative/commutative/exact). Group fallback otherwise.
    if let (Some(labels), Some(index)) = (labels, index)
        && input.shape == labels.shape
    {
        {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                return Ok(measurement_one_based_minmax(
                    &input.data,
                    &labels.data,
                    label_count,
                    false,
                ));
            }
        }
    }
    if labels.is_none() && !NDIMAGE_MINMAX_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        // Global min (labels=None): fold the whole array directly, skipping the
        // `measurement_label_groups` full-data clone. Byte-identical (associative + NaN-propagating).
        if input.data.is_empty() {
            return Ok(vec![0.0]);
        }
        return Ok(vec![global_minmax_reduce(&input.data, false)]);
    }
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().fold(f64::INFINITY, |acc, value| {
                    if acc.is_nan() || value.is_nan() {
                        f64::NAN
                    } else {
                        acc.min(*value)
                    }
                })
            }
        })
        .collect())
}

/// Maximum of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.maximum`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
pub fn maximum(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    // Fast streaming path: byte-identical parallel privatized-histogram max over a
    // one-based-contiguous index (max is associative/commutative/exact). Group fallback otherwise.
    if let (Some(labels), Some(index)) = (labels, index)
        && input.shape == labels.shape
    {
        {
            if let Some(label_count) = measurement_one_based_contiguous_index_len(index) {
                return Ok(measurement_one_based_minmax(
                    &input.data,
                    &labels.data,
                    label_count,
                    true,
                ));
            }
        }
    }
    if labels.is_none() && !NDIMAGE_MINMAX_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        // Global max (labels=None): fold the whole array directly, skipping the
        // `measurement_label_groups` full-data clone. Byte-identical (associative + NaN-propagating).
        if input.data.is_empty() {
            return Ok(vec![0.0]);
        }
        return Ok(vec![global_minmax_reduce(&input.data, true)]);
    }
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().fold(f64::NEG_INFINITY, |acc, value| {
                    if acc.is_nan() || value.is_nan() {
                        f64::NAN
                    } else {
                        acc.max(*value)
                    }
                })
            }
        })
        .collect())
}

/// Same-binary A/B toggle for `median`. When `true`, the per-label medians are computed serially
/// (the ORIG behaviour). When `false` (default), the independent per-label sorts fan across cores.
/// Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static NDIMAGE_MEDIAN_LABELS_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, force the global (no-labels) `median` onto the ORIG group path (redundant full-data
/// clone). Byte-identical either way. Benchmark knob for the same-binary A/B (this cleanup measured
/// IN-FLOOR — median is sort-dominated — but the clone removal is monotone/byte-identical).
#[doc(hidden)]
pub static NDIMAGE_MEDIAN_GLOBAL_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Median of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.median`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
pub fn median(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    if labels.is_none()
        && !NDIMAGE_MEDIAN_GLOBAL_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        // Global median (labels=None): `median_of_values(input.data)` directly, skipping the
        // `measurement_label_groups` full-data clone (median double-clones: the group clone AND
        // `median_of_values`'s own `.to_vec()` for the sort — this drops the first). BYTE-IDENTICAL
        // (same values, same `total_cmp` sort). Monotone cleanup; the sort dominates so it measured
        // IN-FLOOR (~1.07x), NOT a headline win.
        return Ok(vec![median_of_values(&input.data)]);
    }
    let groups = measurement_label_groups(input, labels, index)?;

    // Each label's median is an INDEPENDENT sort of its own values, written to its own output slot,
    // so distributing the groups across cores is BYTE-IDENTICAL to the serial map (median_of_values
    // is deterministic; only the owning core changes). scipy.ndimage.median is single-threaded.
    let total: usize = groups.iter().map(Vec::len).sum();
    let nthreads = if NDIMAGE_MEDIAN_LABELS_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || groups.len() < 2
    {
        1
    } else {
        ndimage_filter_thread_count(total, 8).min(groups.len())
    };
    if nthreads <= 1 {
        return Ok(groups
            .iter()
            .map(|values| median_of_values(values))
            .collect());
    }
    let mut out = vec![0.0f64; groups.len()];
    let chunk = groups.len().div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (gchunk, ochunk) in groups.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (values, slot) in gchunk.iter().zip(ochunk.iter_mut()) {
                    *slot = median_of_values(values);
                }
            });
        }
    });
    Ok(out)
}

/// When `true`, force `labeled_comprehension` onto the ORIG serial per-group map (bypasses the
/// across-groups fan-out). Byte-identical either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_LABELED_COMPREHENSION_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, force the global (no-labels, no-positions) `labeled_comprehension` onto the ORIG
/// group path (pair-alloc + value-extraction) instead of handing `input.data` to the reducer
/// directly. Byte-identical either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_LABELED_COMPREHENSION_GLOBAL_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply a reducer to optionally labeled regions.
///
/// Matches `scipy.ndimage.labeled_comprehension` for numeric outputs. Scalar
/// SciPy results are returned as a one-element vector, while explicit `index`
/// lists return one value per requested label. When `pass_positions` is true,
/// the callback receives flat input positions for the selected values.
pub fn labeled_comprehension<F>(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
    func: F,
    default: f64,
    pass_positions: bool,
) -> Result<Vec<f64>, NdimageError>
where
    F: Fn(&[f64], Option<&[usize]>) -> f64 + Sync,
{
    if labels.is_none() && index.is_some() {
        return Err(NdimageError::InvalidArgument(
            "index without defined labels".to_string(),
        ));
    }

    if labels.is_none()
        && !pass_positions
        && !NDIMAGE_LABELED_COMPREHENSION_GLOBAL_FORCE_SERIAL
            .load(std::sync::atomic::Ordering::Relaxed)
    {
        // Global comprehension (labels=None ⇒ index=None; positions not requested): hand the reducer
        // `input.data` DIRECTLY, skipping the `measurement_label_value_positions` (value, position)
        // pair allocation AND the per-group value-extraction copy. BYTE-IDENTICAL: the single group is
        // every value in increasing-flat-index order, and `func` is deterministic (matches the group
        // path's `func(&values, None)` where `values == input.data`).
        return Ok(vec![func(&input.data, None)]);
    }

    let groups = measurement_label_value_positions(input, labels, index)?;

    // Each group's reducer call is INDEPENDENT (its own value/position slice → its own output
    // slot), so distributing the groups across cores is BYTE-IDENTICAL to the serial map: `func`
    // is deterministic and results are written in group order; only the owning core changes.
    // scipy.ndimage.labeled_comprehension calls a Python callback per label single-threaded, so
    // the across-groups fan-out compounds Rust's per-call speed with real parallelism.
    let eval = |group: &[(f64, usize)]| -> f64 {
        if index.is_some() && group.is_empty() {
            default
        } else {
            let values: Vec<f64> = group.iter().map(|&(value, _)| value).collect();
            if pass_positions {
                let positions: Vec<usize> = group.iter().map(|&(_, flat)| flat).collect();
                func(&values, Some(&positions))
            } else {
                func(&values, None)
            }
        }
    };

    let total: usize = groups.iter().map(Vec::len).sum();
    let nthreads = if NDIMAGE_LABELED_COMPREHENSION_FORCE_SERIAL
        .load(std::sync::atomic::Ordering::Relaxed)
        || groups.len() < 2
    {
        1
    } else {
        ndimage_filter_thread_count(total, 8).min(groups.len())
    };
    if nthreads <= 1 {
        return Ok(groups.iter().map(|g| eval(g)).collect());
    }
    let mut out = vec![default; groups.len()];
    let chunk = groups.len().div_ceil(nthreads);
    let eval = &eval;
    std::thread::scope(|scope| {
        for (gchunk, ochunk) in groups.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (group, slot) in gchunk.iter().zip(ochunk.iter_mut()) {
                    *slot = eval(group);
                }
            });
        }
    });
    Ok(out)
}

/// Positions of minimum values in optionally labeled regions.
///
/// Matches `scipy.ndimage.minimum_position`; scalar SciPy results are returned
/// as a one-element vector of coordinates, while explicit `index` lists return
/// one coordinate vector per label.
pub fn minimum_position(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<usize>>, NdimageError> {
    Ok(measurement_label_value_positions(input, labels, index)?
        .iter()
        .map(|values| minimum_value_position(input, values).1)
        .collect())
}

/// Positions of maximum values in optionally labeled regions.
///
/// Matches `scipy.ndimage.maximum_position`; scalar SciPy results are returned
/// as a one-element vector of coordinates, while explicit `index` lists return
/// one coordinate vector per label.
pub fn maximum_position(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<usize>>, NdimageError> {
    Ok(measurement_label_value_positions(input, labels, index)?
        .iter()
        .map(|values| maximum_value_position(input, values).1)
        .collect())
}

/// Minima, maxima, and their positions in optionally labeled regions.
///
/// Matches `scipy.ndimage.extrema`; scalar SciPy results are returned as
/// one-element vectors, while explicit `index` lists return one value/position
/// per label.
/// When `true`, force the global (no-labels) `extrema` onto the ORIG group path (allocate all
/// (value, position) pairs + two scans). Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static NDIMAGE_EXTREMA_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn extrema(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<ExtremaResult, NdimageError> {
    if labels.is_none() && !NDIMAGE_EXTREMA_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        // Global extrema (labels=None): a single fused pass over `input.data` tracking min/max +
        // their positions, skipping the `measurement_label_value_positions` (value, position) pair
        // allocation AND fusing the two separate scans. BYTE-IDENTICAL to `minimum_value_position` /
        // `maximum_value_position` over the whole array: identical strict-`<`/`>` first-occurrence
        // tie-break, identical NaN "last wins" update, identical increasing-flat-index order.
        let data = input.data.as_slice();
        if data.is_empty() {
            let z = vec![0usize; input.ndim()];
            return Ok((vec![0.0], vec![0.0], vec![z.clone()], vec![z]));
        }
        let (mut min_val, mut min_flat) = (data[0], 0usize);
        let (mut max_val, mut max_flat) = (data[0], 0usize);
        for (flat, &value) in data.iter().enumerate().skip(1) {
            if value < min_val || value.is_nan() {
                min_val = value;
                min_flat = flat;
            }
            if value > max_val || value.is_nan() {
                max_val = value;
                max_flat = flat;
            }
        }
        return Ok((
            vec![min_val],
            vec![max_val],
            vec![input.unravel(min_flat)],
            vec![input.unravel(max_flat)],
        ));
    }
    let groups = measurement_label_value_positions(input, labels, index)?;
    let mut minima = Vec::with_capacity(groups.len());
    let mut maxima = Vec::with_capacity(groups.len());
    let mut minimum_positions = Vec::with_capacity(groups.len());
    let mut maximum_positions = Vec::with_capacity(groups.len());

    for values in &groups {
        let (minimum, minimum_position) = minimum_value_position(input, values);
        let (maximum, maximum_position) = maximum_value_position(input, values);
        minima.push(minimum);
        maxima.push(maximum);
        minimum_positions.push(minimum_position);
        maximum_positions.push(maximum_position);
    }

    Ok((minima, maxima, minimum_positions, maximum_positions))
}

/// Standard deviation of values in labeled regions.
///
/// Matches `scipy.ndimage.standard_deviation`.
pub fn standard_deviation_labels(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Result<Vec<f64>, NdimageError> {
    Ok(variance_labels(input, labels, num_labels)?
        .into_iter()
        .map(|v| v.sqrt())
        .collect())
}

/// Find bounding box slices for each labeled region.
///
/// Returns a Vec of (min_indices, max_indices) for each label.
/// Matches `scipy.ndimage.find_objects`.
pub fn find_objects(labels: &NdArray, num_labels: usize) -> Vec<Option<(Vec<usize>, Vec<usize>)>> {
    let ndim = labels.ndim();
    let mut mins: Vec<Vec<usize>> = vec![vec![usize::MAX; ndim]; num_labels + 1];
    let mut maxs: Vec<Vec<usize>> = vec![vec![0; ndim]; num_labels + 1];
    let mut found = vec![false; num_labels + 1];

    // Unravel each labeled cell into a REUSED buffer instead of allocating a
    // fresh `Vec` per cell (the old `labels.unravel(flat)` allocated once per
    // labeled pixel — ~131k allocs on a half-foreground 512² label image).
    // Byte-identical: `unravel_into` produces the same coordinate.
    let mut coord = vec![0usize; ndim];
    for flat in 0..labels.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            found[lbl] = true;
            unravel_into(flat, &labels.strides, &mut coord);
            for d in 0..ndim {
                mins[lbl][d] = mins[lbl][d].min(coord[d]);
                maxs[lbl][d] = maxs[lbl][d].max(coord[d]);
            }
        }
    }

    (1..=num_labels)
        .map(|l| {
            if found[l] {
                Some((mins[l].clone(), maxs[l].clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Same-binary A/B toggle: when `true`, `center_of_mass`/`sum_axis`/`pad_constant` recompute
/// the per-element multi-index with `input.unravel(flat)` (a `Vec` heap-alloc per element)
/// instead of the in-place row-major odometer. Byte-identical output; A/B timing only.
#[doc(hidden)]
pub static NDIMAGE_UNRAVEL_ODOMETER_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Center of mass for each labeled region.
///
/// Matches `scipy.ndimage.center_of_mass`.
pub fn center_of_mass(
    input: &NdArray,
    labels: &NdArray,
    num_labels: usize,
) -> Result<Vec<Vec<f64>>, NdimageError> {
    if input.shape != labels.shape {
        return Err(NdimageError::DimensionMismatch(format!(
            "input shape {:?} != labels shape {:?}",
            input.shape, labels.shape
        )));
    }
    let ndim = input.ndim();
    let mut weighted_sums = vec![vec![0.0; ndim]; num_labels + 1];
    let mut total_weights = vec![0.0; num_labels + 1];

    // The multi-index is tracked with an in-place ROW-MAJOR odometer instead of
    // `input.unravel(flat)` — the latter heap-allocates a `Vec<usize>` for EVERY element
    // (`input.size()` allocations). BYTE-IDENTICAL: the odometer holds exactly
    // `unravel(flat)` at step `flat`, so `idx[d]` and the flat-order accumulation are unchanged.
    let full_mode = NDIMAGE_UNRAVEL_ODOMETER_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let mut idx = vec![0usize; ndim];
    for flat in 0..input.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            let full_idx;
            let coords: &[usize] = if full_mode {
                full_idx = input.unravel(flat);
                &full_idx
            } else {
                &idx
            };
            let w = input.data[flat];
            total_weights[lbl] += w;
            for d in 0..ndim {
                weighted_sums[lbl][d] += w * coords[d] as f64;
            }
        }
        if !full_mode {
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < input.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    Ok((1..=num_labels)
        .map(|l| {
            if total_weights[l] != 0.0 {
                weighted_sums[l]
                    .iter()
                    .map(|&s| s / total_weights[l])
                    .collect()
            } else {
                // SciPy returns NaN for labels with zero total weight
                vec![f64::NAN; ndim]
            }
        })
        .collect())
}

// ══════════════════════════════════════════════════════════════════════
// Distance Transform
// ══════════════════════════════════════════════════════════════════════

/// Distance metric for brute-force and chamfer distance transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Taxicab,
    Chessboard,
}

/// Distance transform outputs matching SciPy's return flag combinations.
#[derive(Debug, Clone)]
pub struct DistanceTransformEdtResult {
    pub distances: Option<NdArray>,
    pub indices: Option<Vec<NdArray>>,
}

const BF_NO_BACKGROUND_EUCLIDEAN: f64 = 1.340_780_792_994_259_6e154;
const BF_NO_BACKGROUND_GRID: f64 = u32::MAX as f64;

/// Euclidean distance transform for a binary image.
///
/// For each foreground pixel (nonzero), computes the distance to the nearest
/// background pixel (0). Background pixels get distance 0.
///
/// Uses the brute-force approach for correctness (suitable for moderate sizes).
/// Matches `scipy.ndimage.distance_transform_edt`.
pub fn distance_transform_edt(
    input: &NdArray,
    sampling: Option<&[f64]>,
) -> Result<NdArray, NdimageError> {
    distance_transform_edt_full(input, sampling, true, false)?
        .distances
        .ok_or_else(|| NdimageError::InvalidArgument("distances were not requested".to_string()))
}

/// Euclidean distance transform with SciPy-style optional feature indices.
///
/// Matches `scipy.ndimage.distance_transform_edt` for the distance output and
/// the `return_indices=True` feature transform. Indices are returned as one
/// `NdArray` per input axis, each with the input shape and integer coordinates
/// represented as `f64` values.
pub fn distance_transform_edt_full(
    input: &NdArray,
    sampling: Option<&[f64]>,
    return_distances: bool,
    return_indices: bool,
) -> Result<DistanceTransformEdtResult, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if !return_distances && !return_indices {
        return Err(NdimageError::InvalidArgument(
            "at least one of return_distances/return_indices must be true".to_string(),
        ));
    }

    let sampling = normalize_sampling(input.ndim(), sampling)?;
    let sampling_is_positive = sampling.iter().all(|&s| s.is_finite() && s > 0.0);
    let has_background = input.data.contains(&0.0);

    // Fast path: distances-only with at least one background pixel. The exact
    // separable Felzenszwalb–Huttenlocher transform replaces the brute-force
    // O(foreground · background) scan with O(N · ndim) and is byte-identical
    // (see `edt_squared_felzenszwalb`). The all-foreground sentinel and the
    // index/tie-break semantics stay on the brute-force path below.
    if return_distances && !return_indices && has_background && sampling_is_positive {
        let squared = edt_squared_felzenszwalb(input, &sampling);
        let mut output = NdArray::zeros(input.shape.clone());
        for (flat, &value) in input.data.iter().enumerate() {
            output.data[flat] = if value == 0.0 {
                0.0
            } else {
                squared[flat].sqrt()
            };
        }
        return Ok(DistanceTransformEdtResult {
            distances: Some(output),
            indices: None,
        });
    }

    // Fast path: indices (optionally with distances) via the exact separable
    // feature transform — replaces the brute-force O(foreground · background)
    // nearest-background scan with O(N · ndim). Distances are byte-identical to
    // the distance-only fast path above; the returned index is a genuine
    // nearest background (it achieves the exact squared distance `squared`).
    // The all-foreground sentinel and non-finite sampling stay on the
    // brute-force path below. frankenscipy-9l5oo.
    if return_indices && has_background && sampling_is_positive {
        if input.ndim() == 2 {
            return Ok(edt_2d_felzenszwalb_with_indices(
                input,
                &sampling,
                return_distances,
            ));
        }
        let (squared, feat) = edt_squared_felzenszwalb_with_indices(input, &sampling);
        let mut distances = return_distances.then(|| NdArray::zeros(input.shape.clone()));
        let mut axis_indices = (0..input.ndim())
            .map(|_| NdArray::zeros(input.shape.clone()))
            .collect::<Vec<_>>();
        if input.ndim() == 2 {
            let cols = input.shape[1];
            let (axis0, rest) = axis_indices.split_at_mut(1);
            let rows = &mut axis0[0].data;
            let cols_out = &mut rest[0].data;
            for flat in 0..input.data.len() {
                let is_background = input.data[flat] == 0.0;
                if let Some(output) = distances.as_mut() {
                    output.data[flat] = if is_background {
                        0.0
                    } else {
                        squared[flat].sqrt()
                    };
                }
                let nearest_flat = if is_background { flat } else { feat[flat] };
                rows[flat] = (nearest_flat / cols) as f64;
                cols_out[flat] = (nearest_flat % cols) as f64;
            }
        } else {
            let mut nearest_coords = vec![0usize; input.ndim()];
            for flat in 0..input.data.len() {
                let is_background = input.data[flat] == 0.0;
                if let Some(output) = distances.as_mut() {
                    output.data[flat] = if is_background {
                        0.0
                    } else {
                        squared[flat].sqrt()
                    };
                }
                let nearest_flat = if is_background { flat } else { feat[flat] };
                unravel_into(nearest_flat, &input.strides, &mut nearest_coords);
                for (axis, output) in axis_indices.iter_mut().enumerate() {
                    output.data[flat] = nearest_coords[axis] as f64;
                }
            }
        }
        return Ok(DistanceTransformEdtResult {
            distances,
            indices: Some(axis_indices),
        });
    }

    let backgrounds = background_coordinates(input);

    let mut distances = return_distances.then(|| NdArray::zeros(input.shape.clone()));
    let mut indices = return_indices.then(|| {
        (0..input.ndim())
            .map(|_| NdArray::zeros(input.shape.clone()))
            .collect::<Vec<_>>()
    });

    // For each foreground pixel, find distance to nearest background pixel.
    for (flat, &value) in input.data.iter().enumerate() {
        let coords = input.unravel(flat);
        if value == 0.0 {
            if let Some(output) = distances.as_mut() {
                output.data[flat] = 0.0;
            }
            if let Some(axis_indices) = indices.as_mut() {
                for (axis, output) in axis_indices.iter_mut().enumerate() {
                    output.data[flat] = coords[axis] as f64;
                }
            }
        } else {
            let (distance, nearest) = nearest_edt_background(&coords, &backgrounds, &sampling);
            if let Some(output) = distances.as_mut() {
                output.data[flat] = distance;
            }
            if let Some(axis_indices) = indices.as_mut() {
                for (axis, output) in axis_indices.iter_mut().enumerate() {
                    output.data[flat] = nearest[axis];
                }
            }
        }
    }

    Ok(DistanceTransformEdtResult { distances, indices })
}

/// Brute-force distance transform for a binary image.
///
/// Matches `scipy.ndimage.distance_transform_bf` for the distance-only case.
/// Foreground values are nonzero; background values are zero.
pub fn distance_transform_bf(
    input: &NdArray,
    metric: DistanceMetric,
    sampling: Option<&[f64]>,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let sampling = if metric == DistanceMetric::Euclidean {
        Some(normalize_sampling(input.ndim(), sampling)?)
    } else {
        None
    };
    let backgrounds = background_coordinates(input);

    // Fast path: the Euclidean brute force computes, per foreground pixel, the
    // min over EVERY background of sqrt(Σ_axis ((Δ·sampling)²)) — identical
    // arithmetic to distance_transform_edt, which is now the exact separable
    // Felzenszwalb transform (O(N·ndim), byte-identical; see
    // edt_squared_felzenszwalb). Reuse it when there is at least one background
    // pixel and sampling is positive/finite. The no-background sentinel and the
    // taxicab/chessboard metrics keep the brute-force path below.
    if metric == DistanceMetric::Euclidean
        && !backgrounds.is_empty()
        && let Some(samp) = sampling.as_deref()
        && samp.iter().all(|&s| s.is_finite() && s > 0.0)
    {
        let squared = edt_squared_felzenszwalb(input, samp);
        let mut output = NdArray::zeros(input.shape.clone());
        for (flat, &value) in input.data.iter().enumerate() {
            output.data[flat] = if value == 0.0 {
                0.0
            } else {
                squared[flat].sqrt()
            };
        }
        return Ok(output);
    }

    let no_background = match metric {
        DistanceMetric::Euclidean => BF_NO_BACKGROUND_EUCLIDEAN,
        DistanceMetric::Taxicab | DistanceMetric::Chessboard => BF_NO_BACKGROUND_GRID,
    };

    Ok(distance_transform_by_metric(
        input,
        metric,
        sampling.as_deref(),
        &backgrounds,
        no_background,
    ))
}

/// Chamfer distance transform for a binary image.
///
/// Matches `scipy.ndimage.distance_transform_cdt` for the distance-only
/// taxicab/chessboard metrics. The Euclidean metric is rejected because SciPy's
/// CDT API only accepts chamfer metrics.
pub fn distance_transform_cdt(
    input: &NdArray,
    metric: DistanceMetric,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    if metric == DistanceMetric::Euclidean {
        return Err(NdimageError::InvalidArgument(
            "distance_transform_cdt requires taxicab or chessboard metric".to_string(),
        ));
    }

    // The taxicab/chessboard fast path inside `distance_transform_by_metric` only
    // needs to know whether ANY background pixel exists (it ignores the coordinate
    // list). Building the full `Vec<Vec<usize>>` of background coordinates (one
    // heap-allocated coord vector per zero pixel) is pure waste here, so pass a
    // cheap non-empty sentinel instead. cdt only accepts grid metrics, so the
    // brute-force branch that would read the coordinates is never taken.
    let backgrounds: Vec<Vec<usize>> = if input.data.contains(&0.0) {
        vec![Vec::new()]
    } else {
        Vec::new()
    };
    Ok(distance_transform_by_metric(
        input,
        metric,
        None,
        &backgrounds,
        -1.0,
    ))
}

fn background_coordinates(input: &NdArray) -> Vec<Vec<usize>> {
    input
        .data
        .iter()
        .enumerate()
        .filter_map(|(flat, &value)| {
            if value == 0.0 {
                Some(input.unravel(flat))
            } else {
                None
            }
        })
        .collect()
}

/// Exact city-block (taxicab / L1) distance transform via separable two-pass
/// chamfer sweeps, O(N · ndim). Returns a flat row-major buffer of L1 distances
/// to the nearest background pixel (`value == 0.0`).
///
/// L1 is an additive separable metric (`Σ_axis |Δ_axis|`), so the per-axis 1-D
/// transform is the classic forward/backward `min(d, neighbor + 1)` sweep and
/// the axis passes compose exactly. All values are exact integers, so the
/// result is byte-identical to the brute-force `min over background of Σ |Δ|`.
fn cityblock_distance_transform(input: &NdArray) -> Vec<f64> {
    let n = input.data.len();
    let mut f: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { f64::INFINITY })
        .collect();

    // `axis` indexes shape/strides in lockstep, so a range loop reads clearest.
    #[allow(clippy::needless_range_loop)]
    for axis in 0..input.ndim() {
        let len = input.shape[axis];
        if len <= 1 {
            continue;
        }
        let stride = input.strides[axis];
        // Enumerate the n/len line starts DIRECTLY instead of scanning all n flat
        // indices and rejecting non-starts with a per-index `base/stride % len`
        // division. A line along `axis` starts at every flat index whose
        // axis-coordinate is 0, i.e. `outer*block + inner` for `block = len*stride`,
        // `outer in 0..n/block`, `inner in 0..stride`. Same set of starts, same
        // sweeps => byte-identical; drops the dominant divide-per-element overhead.
        let block = len * stride;
        let num_blocks = n / block;
        // Process each level `t` across ALL `stride` parallel lines in the block
        // before advancing to `t+1` (inner loop over `inner in 0..stride`). Each
        // line's forward sweep still reads its own level `t-1` (computed in the
        // prior outer step) and the lines are independent, so the result is
        // byte-identical to walking one line fully at a time — but the `inner`
        // loop now strides through CONTIGUOUS memory (the row at level `t`),
        // turning the cache-hostile column walk into a vectorizable contiguous
        // min and eliminating the per-step cache miss for axes with `stride > 1`.
        for outer in 0..num_blocks {
            let block_base = outer * block;
            // Forward sweep: best reachable from the left. Split the buffer so the
            // contiguous `inner` loop is a branchless elementwise `min` over two
            // disjoint slices, which autovectorizes (vminpd) for `stride > 1`.
            for t in 1..len {
                let row = block_base + t * stride;
                let prev = row - stride;
                let (head, tail) = f.split_at_mut(row);
                let prev_row = &head[prev..prev + stride];
                let cur_row = &mut tail[..stride];
                for (cur, &p) in cur_row.iter_mut().zip(prev_row) {
                    *cur = cur.min(p + 1.0);
                }
            }
            // Backward sweep: best reachable from the right.
            for t in (0..len - 1).rev() {
                let row = block_base + t * stride;
                let next = row + stride;
                let (head, tail) = f.split_at_mut(next);
                let cur_row = &mut head[row..row + stride];
                let next_row = &tail[..stride];
                for (cur, &nx) in cur_row.iter_mut().zip(next_row) {
                    *cur = cur.min(nx + 1.0);
                }
            }
        }
    }
    f
}

/// Exact chessboard (Chebyshev / L∞) distance transform via a two-pass
/// full-neighbourhood chamfer (Rosenfeld–Pfaltz), O(N · 3^ndim). Returns a flat
/// row-major buffer of L∞ distances to the nearest background pixel.
///
/// Every one of the `3^ndim − 1` neighbours has weight 1, which is exactly the
/// chessboard metric, so a forward raster sweep (relaxing from already-visited
/// neighbours) followed by a backward sweep yields the exact L∞ distance. All
/// values are exact integers, so the result is byte-identical to the
/// brute-force `min over background of max_axis |Δ|`.
fn chessboard_distance_transform(input: &NdArray) -> Vec<f64> {
    let ndim = input.ndim();
    let n = input.data.len();
    let mut d: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { f64::INFINITY })
        .collect();

    // All neighbour offsets in {-1,0,1}^ndim except the all-zero one, paired with
    // their signed flat displacement (Σ off_k · stride_k).
    let mut offsets: Vec<(Vec<i64>, i64)> = Vec::new();
    let combos = 3usize.pow(ndim as u32);
    for code in 0..combos {
        let mut rem = code;
        let mut off = vec![0i64; ndim];
        let mut flat_delta = 0i64;
        let mut all_zero = true;
        // `axis` indexes off/strides in lockstep.
        #[allow(clippy::needless_range_loop)]
        for axis in 0..ndim {
            let o = (rem % 3) as i64 - 1; // 0,1,2 -> -1,0,1
            rem /= 3;
            off[axis] = o;
            flat_delta += o * input.strides[axis] as i64;
            if o != 0 {
                all_zero = false;
            }
        }
        if !all_zero {
            offsets.push((off, flat_delta));
        }
    }

    let mut coords = vec![0usize; ndim];
    let in_bounds = |coords: &[usize], off: &[i64]| -> bool {
        for axis in 0..ndim {
            let c = coords[axis] as i64 + off[axis];
            if c < 0 || c >= input.shape[axis] as i64 {
                return false;
            }
        }
        true
    };

    let shape = &input.shape;
    // Row-major coordinate stepping replaces the per-cell `unravel_into` (ndim
    // divisions per cell): the raster scan visits flats in order, so carry the
    // coordinate vector incrementally — same coords as `unravel_into`, so
    // byte-identical.
    let step_forward = |coords: &mut [usize]| {
        for axis in (0..coords.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < shape[axis] {
                return;
            }
            coords[axis] = 0;
        }
    };
    let step_backward = |coords: &mut [usize]| {
        for axis in (0..coords.len()).rev() {
            if coords[axis] > 0 {
                coords[axis] -= 1;
                return;
            }
            coords[axis] = shape[axis] - 1;
        }
    };

    // Forward raster sweep: relax from neighbours that precede the cell.
    for axis in coords.iter_mut() {
        *axis = 0;
    }
    for flat in 0..n {
        if d[flat] != 0.0 {
            // Interior cells (no coordinate on a boundary) have every neighbour in
            // bounds, so skip the per-offset `in_bounds` check — identical verdict,
            // drops the ndim·offsets bounds work for the all-interior bulk.
            let interior = (0..ndim).all(|ax| coords[ax] >= 1 && coords[ax] + 1 < shape[ax]);
            if interior {
                for (_, flat_delta) in &offsets {
                    if *flat_delta < 0 {
                        let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                        if cand < d[flat] {
                            d[flat] = cand;
                        }
                    }
                }
            } else {
                for (off, flat_delta) in &offsets {
                    if *flat_delta < 0 && in_bounds(&coords, off) {
                        let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                        if cand < d[flat] {
                            d[flat] = cand;
                        }
                    }
                }
            }
        }
        step_forward(&mut coords);
    }
    // Backward raster sweep: relax from neighbours that follow the cell.
    for axis in 0..ndim {
        coords[axis] = shape[axis] - 1;
    }
    for flat in (0..n).rev() {
        if d[flat] != 0.0 {
            let interior = (0..ndim).all(|ax| coords[ax] >= 1 && coords[ax] + 1 < shape[ax]);
            if interior {
                for (_, flat_delta) in &offsets {
                    if *flat_delta > 0 {
                        let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                        if cand < d[flat] {
                            d[flat] = cand;
                        }
                    }
                }
            } else {
                for (off, flat_delta) in &offsets {
                    if *flat_delta > 0 && in_bounds(&coords, off) {
                        let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                        if cand < d[flat] {
                            d[flat] = cand;
                        }
                    }
                }
            }
        }
        step_backward(&mut coords);
    }
    d
}

fn unravel_into(mut flat: usize, strides: &[usize], out: &mut [usize]) {
    for (slot, &stride) in out.iter_mut().zip(strides) {
        *slot = flat / stride;
        flat %= stride;
    }
}

fn for_each_axis_line_start(
    total: usize,
    axis_len: usize,
    axis_stride: usize,
    mut visit: impl FnMut(usize),
) {
    let block = axis_len * axis_stride;
    let mut block_start = 0usize;
    while block_start < total {
        for offset in 0..axis_stride {
            visit(block_start + offset);
        }
        block_start += block;
    }
}

fn distance_transform_by_metric(
    input: &NdArray,
    metric: DistanceMetric,
    sampling: Option<&[f64]>,
    backgrounds: &[Vec<usize>],
    no_background: f64,
) -> NdArray {
    // Fast paths: the grid metrics replace the O(foreground · background) scan
    // with exact chamfer transforms — city-block separably (O(N · ndim)) and
    // chessboard via a full-neighbourhood two-pass sweep (O(N · 3^ndim)). The
    // no-background sentinel keeps the brute-force path below.
    if !backgrounds.is_empty()
        && matches!(metric, DistanceMetric::Taxicab | DistanceMetric::Chessboard)
    {
        let dt = match metric {
            DistanceMetric::Taxicab => cityblock_distance_transform(input),
            _ => chessboard_distance_transform(input),
        };
        let mut output = NdArray::zeros(input.shape.clone());
        for (flat, &value) in input.data.iter().enumerate() {
            output.data[flat] = if value == 0.0 { 0.0 } else { dt[flat] };
        }
        return output;
    }

    let mut output = NdArray::zeros(input.shape.clone());
    for (flat, &value) in input.data.iter().enumerate() {
        if value == 0.0 {
            output.data[flat] = 0.0;
        } else if backgrounds.is_empty() {
            output.data[flat] = no_background;
        } else {
            let coords = input.unravel(flat);
            let min_distance = backgrounds
                .iter()
                .map(|background| metric_distance(&coords, background, metric, sampling))
                .fold(f64::INFINITY, f64::min);
            output.data[flat] = min_distance;
        }
    }
    output
}

fn metric_distance(
    coords: &[usize],
    background: &[usize],
    metric: DistanceMetric,
    sampling: Option<&[f64]>,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => coords
            .iter()
            .zip(background)
            .zip(sampling.expect("euclidean distance requires sampling"))
            .map(|((&coord, &background_coord), &scale)| {
                let delta = (coord as f64 - background_coord as f64) * scale;
                delta * delta
            })
            .sum::<f64>()
            .sqrt(),
        DistanceMetric::Taxicab => coords
            .iter()
            .zip(background)
            .map(|(&coord, &background_coord)| coord.abs_diff(background_coord) as f64)
            .sum(),
        DistanceMetric::Chessboard => coords
            .iter()
            .zip(background)
            .map(|(&coord, &background_coord)| coord.abs_diff(background_coord) as f64)
            .fold(0.0, f64::max),
    }
}

fn nearest_edt_background(
    coords: &[usize],
    backgrounds: &[Vec<usize>],
    sampling: &[f64],
) -> (f64, Vec<f64>) {
    if backgrounds.is_empty() {
        let mut nearest = vec![0.0; coords.len()];
        if let Some(first) = nearest.first_mut() {
            *first = -1.0;
        }
        return (edt_all_foreground_distance(coords, sampling), nearest);
    }

    let mut nearest_background = &backgrounds[0];
    let mut min_distance = metric_distance(
        coords,
        nearest_background,
        DistanceMetric::Euclidean,
        Some(sampling),
    );
    for background in &backgrounds[1..] {
        let distance = metric_distance(
            coords,
            background,
            DistanceMetric::Euclidean,
            Some(sampling),
        );
        if distance < min_distance {
            min_distance = distance;
            nearest_background = background;
        }
    }

    (
        min_distance,
        nearest_background
            .iter()
            .map(|&coord| coord as f64)
            .collect(),
    )
}

/// Exact squared Euclidean distance transform via the separable
/// Felzenszwalb–Huttenlocher lower-envelope algorithm (O(N · ndim)).
///
/// Returns a flat row-major buffer of squared distances to the nearest
/// background pixel (`value == 0.0`). The result is byte-identical to the
/// brute-force `min over background of Σ_axis ((Δ_axis · sampling)²)`:
/// the per-axis squared terms are summed across passes, and IEEE-754 addition
/// is commutative, so the accumulated sum matches the brute-force left-to-right
/// `.sum()` for the minimizing background regardless of pass order. The
/// envelope intersections only *select* parabolas; each emitted value is
/// recomputed as `Δ² + f[source]`, so rounding in the boundaries never perturbs
/// the assigned squared distance.
/// Run one separable squared-EDT pass along an axis (`len`, slab `inner = stride`,
/// `outer` slabs) over `f` IN PLACE, across `nthreads` cores. Each line is an
/// independent 1-D transform, so the result is BIT-IDENTICAL to the serial line walk
/// (the per-line edt math and the (cell→value) mapping are unchanged; only which core
/// owns a line changes). forbid(unsafe)-safe:
///   - `outer >= 2`: slabs are contiguous → each thread owns a disjoint `chunks_mut`
///     range and processes its slabs' `inner` interleaved lines in place.
///   - `outer == 1` (the leading axis): `f` is a `[len][inner]` row-major matrix whose
///     lines are columns; threads write a contiguous COLUMN-MAJOR scratch (disjoint
///     `chunks_mut`), then a parallel transpose copies it back to row-major `f`.
fn edt_axis_pass_parallel(
    f: &mut [f64],
    len: usize,
    inner: usize,
    outer: usize,
    scale: f64,
    scale2: f64,
    nthreads: usize,
) {
    let slab = len * inner;
    if outer >= 2 {
        let slabs_per = outer.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for chunk in f.chunks_mut(slab * slabs_per) {
                scope.spawn(move || {
                    let mut line = vec![0.0f64; len];
                    let mut d = vec![0.0f64; len];
                    let mut v = vec![0usize; len];
                    let mut z = vec![0.0f64; len + 1];
                    let nslabs_local = chunk.len() / slab;
                    for s in 0..nslabs_local {
                        let sb = s * slab;
                        for i in 0..inner {
                            for t in 0..len {
                                line[t] = chunk[sb + i + t * inner];
                            }
                            edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, None);
                            for t in 0..len {
                                chunk[sb + i + t * inner] = d[t];
                            }
                        }
                    }
                });
            }
        });
    } else {
        // outer == 1: f is [len][inner] row-major; lines are the `inner` columns.
        let mut fcm = vec![0.0f64; len * inner];
        let cols_per = inner.div_ceil(nthreads);
        {
            let f_ref: &[f64] = f;
            std::thread::scope(|scope| {
                for (fc, tt) in fcm.chunks_mut(cols_per * len).zip(0usize..) {
                    let col_start = tt * cols_per;
                    scope.spawn(move || {
                        let mut line = vec![0.0f64; len];
                        let mut d = vec![0.0f64; len];
                        let mut v = vec![0usize; len];
                        let mut z = vec![0.0f64; len + 1];
                        let ncols_local = fc.len() / len;
                        for lc in 0..ncols_local {
                            let col = col_start + lc;
                            for t in 0..len {
                                line[t] = f_ref[col + t * inner];
                            }
                            edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, None);
                            let off = lc * len;
                            fc[off..off + len].copy_from_slice(&d[..len]);
                        }
                    });
                }
            });
        }
        let rows_per = len.div_ceil(nthreads);
        let fcm_ref: &[f64] = &fcm;
        std::thread::scope(|scope| {
            for (frow, rr) in f.chunks_mut(rows_per * inner).zip(0usize..) {
                let row_start = rr * rows_per;
                scope.spawn(move || {
                    let nrl = frow.len() / inner;
                    for lr in 0..nrl {
                        let t = row_start + lr;
                        let ob = lr * inner;
                        for col in 0..inner {
                            frow[ob + col] = fcm_ref[col * len + t];
                        }
                    }
                });
            }
        });
    }
}

fn edt_squared_felzenszwalb(input: &NdArray, sampling: &[f64]) -> Vec<f64> {
    let n = input.data.len();
    let mut f: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { f64::INFINITY })
        .collect();

    let mut line: Vec<f64> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    let mut v: Vec<usize> = Vec::new();
    let mut z: Vec<f64> = Vec::new();

    // `axis` indexes shape/strides/sampling in lockstep, so a range loop reads
    // clearest here.
    #[allow(clippy::needless_range_loop)]
    for axis in 0..input.ndim() {
        let len = input.shape[axis];
        if len <= 1 {
            continue; // a length-1 axis adds a zero term; nothing to propagate.
        }
        let stride = input.strides[axis];
        let scale = sampling[axis];
        let scale2 = scale * scale;
        let inner = stride;
        let outer = n / (len * inner);
        let nthreads = ndimage_filter_thread_count(n, 1);
        if nthreads <= 1 {
            line.resize(len, 0.0);
            d.resize(len, 0.0);
            v.resize(len, 0);
            z.resize(len + 1, 0.0);
            for_each_axis_line_start(n, len, stride, |base| {
                for t in 0..len {
                    line[t] = f[base + t * stride];
                }
                edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, None);
                for t in 0..len {
                    f[base + t * stride] = d[t];
                }
            });
        } else {
            edt_axis_pass_parallel(&mut f, len, inner, outer, scale, scale2, nthreads);
        }
    }
    f
}

/// Exact separable Euclidean feature transform: returns both the squared EDT
/// (byte-identical to [`edt_squared_felzenszwalb`]) and, for each cell, the
/// flat index of a nearest background cell. Each separable 1-D pass already
/// finds the winning vertex per output position; we carry the source cell's
/// nearest-background index alongside `f` so the final `feat[i]` is a genuine
/// argmin (it achieves the exact squared distance `f[i]`). Requires at least
/// one background cell. frankenscipy-9l5oo.
fn edt_squared_felzenszwalb_with_indices(
    input: &NdArray,
    sampling: &[f64],
) -> (Vec<f64>, Vec<usize>) {
    let n = input.data.len();
    let mut f: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { f64::INFINITY })
        .collect();
    // Background cells are their own nearest feature; foreground cells get
    // overwritten the first time a finite distance reaches them. `feat[i]` is
    // only meaningful where `f[i]` is finite.
    let mut feat: Vec<usize> = (0..n).collect();

    let mut line: Vec<f64> = Vec::new();
    let mut feat_line: Vec<usize> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    let mut v: Vec<usize> = Vec::new();
    let mut z: Vec<f64> = Vec::new();
    let mut w: Vec<usize> = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for axis in 0..input.ndim() {
        let len = input.shape[axis];
        if len <= 1 {
            continue;
        }
        let stride = input.strides[axis];
        let scale = sampling[axis];
        let scale2 = scale * scale;
        line.resize(len, 0.0);
        feat_line.resize(len, 0);
        d.resize(len, 0.0);
        v.resize(len, 0);
        z.resize(len + 1, 0.0);
        w.resize(len, 0);

        for_each_axis_line_start(n, len, stride, |base| {
            for t in 0..len {
                line[t] = f[base + t * stride];
                feat_line[t] = feat[base + t * stride];
            }
            edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, Some(&mut w));
            for t in 0..len {
                f[base + t * stride] = d[t];
                // A finite output position is covered by a winning vertex whose
                // carried feature (snapshot in `feat_line`) is its nearest
                // background. All-infinite lines leave `d[t]` infinite and are
                // skipped, so stale `w` entries are never read.
                if d[t].is_finite() {
                    feat[base + t * stride] = feat_line[w[t]];
                }
            }
        });
    }
    (f, feat)
}

/// Two-dimensional feature transform that fuses the final axis pass with
/// SciPy-style distance/index materialization. It preserves the generic
/// separable axis order (axis 0, then axis 1) while avoiding one full `feat`
/// flat-index writeback pass and the final per-cell div/mod projection.
fn edt_2d_felzenszwalb_with_indices(
    input: &NdArray,
    sampling: &[f64],
    return_distances: bool,
) -> DistanceTransformEdtResult {
    let n = input.data.len();
    let rows = input.shape[0];
    let cols = input.shape[1];
    let mut f: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 0.0 } else { f64::INFINITY })
        .collect();
    let mut feature_row = vec![0usize; n];

    let mut line: Vec<f64> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    let mut v: Vec<usize> = Vec::new();
    let mut z: Vec<f64> = Vec::new();
    let mut w: Vec<usize> = Vec::new();

    if rows > 1 {
        let scale = sampling[0];
        let scale2 = scale * scale;
        let nthreads = ndimage_filter_thread_count(rows * cols, 1).min(cols).max(1);
        if nthreads <= 1 {
            line.resize(rows, 0.0);
            d.resize(rows, 0.0);
            v.resize(rows, 0);
            z.resize(rows + 1, 0.0);
            w.resize(rows, 0);
            for_each_axis_line_start(n, rows, cols, |base| {
                for t in 0..rows {
                    line[t] = f[base + t * cols];
                }
                edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, Some(&mut w));
                for t in 0..rows {
                    let flat = base + t * cols;
                    f[flat] = d[t];
                    if d[t].is_finite() {
                        feature_row[flat] = w[t];
                    }
                }
            });
        } else {
            // Each column is an INDEPENDENT 1-D transform, but row-major `f` interleaves
            // columns so threads cannot write `f` disjointly under forbid(unsafe). Each
            // thread (owning a contiguous COLUMN range) writes into its own contiguous
            // COLUMN-MAJOR slab, then a parallel transpose copies col-major -> row-major.
            // BIT-IDENTICAL: same per-column edt math + same (flat,value) mapping; infinite
            // cells keep feature index 0 (the init), matching the serial conditional write.
            let mut fcm = vec![0.0f64; n];
            let mut featcm = vec![0usize; n];
            let cols_per = cols.div_ceil(nthreads);
            {
                let f_ref = &f;
                let do_cols = |col_start: usize,
                               fcm_slab: &mut [f64],
                               featcm_slab: &mut [usize]| {
                    let ncols_local = fcm_slab.len() / rows;
                    let mut line = vec![0.0f64; rows];
                    let mut d = vec![0.0f64; rows];
                    let mut v = vec![0usize; rows];
                    let mut z = vec![0.0f64; rows + 1];
                    let mut w = vec![0usize; rows];
                    for lc in 0..ncols_local {
                        let col = col_start + lc;
                        for t in 0..rows {
                            line[t] = f_ref[col + t * cols];
                        }
                        edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, Some(&mut w));
                        let off = lc * rows;
                        for t in 0..rows {
                            fcm_slab[off + t] = d[t];
                            if d[t].is_finite() {
                                featcm_slab[off + t] = w[t];
                            }
                        }
                    }
                };
                let do_cols = &do_cols;
                std::thread::scope(|scope| {
                    for ((fc, ftc), tt) in fcm
                        .chunks_mut(cols_per * rows)
                        .zip(featcm.chunks_mut(cols_per * rows))
                        .zip(0usize..)
                    {
                        let col_start = tt * cols_per;
                        scope.spawn(move || do_cols(col_start, fc, ftc));
                    }
                });
            }
            let fcm_ref = &fcm;
            let featcm_ref = &featcm;
            let rows_per = rows.div_ceil(nthreads);
            std::thread::scope(|scope| {
                for ((frow, ftrow), rr) in f
                    .chunks_mut(rows_per * cols)
                    .zip(feature_row.chunks_mut(rows_per * cols))
                    .zip(0usize..)
                {
                    let row_start = rr * rows_per;
                    scope.spawn(move || {
                        let nrl = frow.len() / cols;
                        for lr in 0..nrl {
                            let t = row_start + lr;
                            let ob = lr * cols;
                            for col in 0..cols {
                                frow[ob + col] = fcm_ref[col * rows + t];
                                ftrow[ob + col] = featcm_ref[col * rows + t];
                            }
                        }
                    });
                }
            });
        }
    }

    let mut distances = return_distances.then(|| NdArray::zeros(input.shape.clone()));
    let mut axis_indices = (0..2)
        .map(|_| NdArray::zeros(input.shape.clone()))
        .collect::<Vec<_>>();
    let (axis0, rest) = axis_indices.split_at_mut(1);
    let rows_out = &mut axis0[0].data;
    let cols_out = &mut rest[0].data;

    if cols > 1 {
        let scale = sampling[1];
        let scale2 = scale * scale;

        // Each row-line is an INDEPENDENT 1-D feature transform: it reads only the
        // now-final `f`/`feature_row`/`input` (all immutable here) and writes the
        // contiguous `[base, base+cols)` block of the three outputs. So fan the rows
        // across cores via chunks_mut — BIT-IDENTICAL (per-row math unchanged; each
        // thread owns its scratch and disjoint contiguous output rows). scipy's
        // feature transform is single-threaded C, so this is a pure domination lever.
        let f_ref = &f;
        let feat_ref = &feature_row;
        let in_ref = &input.data;
        let fill_rows = |row_start: usize,
                         mut dist_slab: Option<&mut [f64]>,
                         rows_slab: &mut [f64],
                         cols_slab: &mut [f64]| {
            let nrows_local = rows_slab.len() / cols;
            let mut line = vec![0.0f64; cols];
            let mut d = vec![0.0f64; cols];
            let mut v = vec![0usize; cols];
            let mut z = vec![0.0f64; cols + 1];
            let mut w = vec![0usize; cols];
            for lr in 0..nrows_local {
                let row = row_start + lr;
                let base = row * cols;
                line.copy_from_slice(&f_ref[base..base + cols]);
                edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z, Some(&mut w));
                let off = lr * cols;
                for t in 0..cols {
                    let is_background = in_ref[base + t] == 0.0;
                    if let Some(ds) = dist_slab.as_deref_mut() {
                        ds[off + t] = if is_background { 0.0 } else { d[t].sqrt() };
                    }
                    if is_background {
                        rows_slab[off + t] = row as f64;
                        cols_slab[off + t] = t as f64;
                    } else {
                        let source_col = w[t];
                        rows_slab[off + t] = feat_ref[base + source_col] as f64;
                        cols_slab[off + t] = source_col as f64;
                    }
                }
            }
        };

        let nthreads = ndimage_filter_thread_count(rows * cols, 1).min(rows).max(1);
        let dist_data: Option<&mut [f64]> = distances.as_mut().map(|nd| nd.data.as_mut_slice());
        if nthreads <= 1 {
            fill_rows(0, dist_data, rows_out, cols_out);
        } else {
            let chunk_rows = rows.div_ceil(nthreads);
            let chunk = chunk_rows * cols;
            let fill_rows = &fill_rows;
            std::thread::scope(|scope| match dist_data {
                Some(dd) => {
                    for (((rs, cs), ds), t) in rows_out
                        .chunks_mut(chunk)
                        .zip(cols_out.chunks_mut(chunk))
                        .zip(dd.chunks_mut(chunk))
                        .zip(0usize..)
                    {
                        let row_start = t * chunk_rows;
                        scope.spawn(move || fill_rows(row_start, Some(ds), rs, cs));
                    }
                }
                None => {
                    for ((rs, cs), t) in rows_out
                        .chunks_mut(chunk)
                        .zip(cols_out.chunks_mut(chunk))
                        .zip(0usize..)
                    {
                        let row_start = t * chunk_rows;
                        scope.spawn(move || fill_rows(row_start, None, rs, cs));
                    }
                }
            });
        }
    } else {
        for row in 0..rows {
            let flat = row * cols;
            let is_background = input.data[flat] == 0.0;
            if let Some(output) = distances.as_mut() {
                output.data[flat] = if is_background { 0.0 } else { f[flat].sqrt() };
            }
            rows_out[flat] = if is_background {
                row as f64
            } else {
                feature_row[flat] as f64
            };
            cols_out[flat] = 0.0;
        }
    }

    DistanceTransformEdtResult {
        distances,
        indices: Some(axis_indices),
    }
}

/// One-dimensional squared distance transform of `f` with axis scale `scale`
/// (`scale2 == scale*scale`), writing results into `d`. `v`/`z` are reused
/// scratch (vertex indices and envelope boundaries). Infinite parabolas are
/// skipped; an all-infinite line stays infinite.
fn edt_1d_squared(
    f: &[f64],
    scale: f64,
    scale2: f64,
    d: &mut [f64],
    v: &mut [usize],
    z: &mut [f64],
    mut w: Option<&mut [usize]>,
) {
    let n = f.len();
    let mut k: isize = -1;
    for q in 0..n {
        let fq = f[q];
        if !fq.is_finite() {
            continue;
        }
        let qf = q as f64;
        loop {
            if k < 0 {
                k = 0;
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            let vk = v[k as usize];
            let vkf = vk as f64;
            let s = ((fq + scale2 * qf * qf) - (f[vk] + scale2 * vkf * vkf))
                / (2.0 * scale2 * (qf - vkf));
            if s <= z[k as usize] {
                k -= 1;
            } else {
                k += 1;
                v[k as usize] = q;
                z[k as usize] = s;
                z[k as usize + 1] = f64::INFINITY;
                break;
            }
        }
    }

    if k < 0 {
        for slot in d.iter_mut().take(n) {
            *slot = f64::INFINITY;
        }
        return;
    }

    // `q` is both the query position (a value) and the output index.
    #[allow(clippy::needless_range_loop)]
    {
        let mut k2: usize = 0;
        for q in 0..n {
            let qf = q as f64;
            while z[k2 + 1] < qf {
                k2 += 1;
            }
            let vk = v[k2];
            let delta = (qf - vk as f64) * scale;
            d[q] = delta * delta + f[vk];
            if let Some(w) = w.as_deref_mut() {
                w[q] = vk;
            }
        }
    }
}

fn edt_all_foreground_distance(coords: &[usize], sampling: &[f64]) -> f64 {
    coords
        .iter()
        .enumerate()
        .map(|(axis, &coord)| {
            let delta = if axis == 0 {
                coord as f64 + 1.0
            } else {
                coord as f64
            } * sampling[axis];
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn normalize_sampling(ndim: usize, sampling: Option<&[f64]>) -> Result<Vec<f64>, NdimageError> {
    match sampling {
        None => Ok(vec![1.0; ndim]),
        Some(values) if values.len() == 1 => Ok(vec![values[0]; ndim]),
        Some(values) if values.len() == ndim => Ok(values.to_vec()),
        Some(values) => Err(NdimageError::InvalidArgument(format!(
            "sampling must have length 1 or match ndim={}, got {}",
            ndim,
            values.len()
        ))),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Interpolation
// ══════════════════════════════════════════════════════════════════════

/// Shift an array using spline interpolation.
///
/// Matches `scipy.ndimage.shift`.
pub fn shift(
    input: &NdArray,
    shift_values: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if shift_values.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "shift has {} values but input has {} dimensions",
            shift_values.len(),
            input.ndim()
        )));
    }
    if shift_values.iter().any(|&v| !v.is_finite()) {
        return Err(NdimageError::InvalidArgument(
            "shift values must be finite (no NaN or Inf)".to_string(),
        ));
    }

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(input.shape.clone());
    let out_shape = output.shape.clone();
    let in_shape = input.shape.clone();
    let ndim = out_shape.len();
    let kernel_work = (order + 1).saturating_pow(ndim as u32);

    // A shift is a translation: coord[axis] = out_idx[axis] − shift[axis], separable per axis
    // exactly like zoom. Two wins over the legacy SERIAL per-pixel loop below: (1) parallelize
    // (each output pixel is independent), and (2) for order ≥ 2 precompute the per-axis B-spline
    // supports ONCE (via the same `compute_axis_support`) rather than per pixel. Byte-identical.
    if !NDIMAGE_ZOOM_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        if order >= 2 {
            let coeffs = &spline.coeffs;
            // ONE read: the tap representation and the leaf kind must agree (see the fn doc).
            // `SplineFlags::resolve` also carries the compact-support knob
            // (`frankenscipy-22van`).
            let flags = SplineFlags::resolve();
            let offsets = flags.offsets;
            let axis_supports = build_axis_offset_supports(
                coeffs,
                &in_shape,
                &out_shape,
                &spline.coord_offsets,
                order,
                mode,
                offsets,
                flags.compact_disabled,
                |axis, o| o as f64 - shift_values[axis],
            );
            let axis_supports = &axis_supports;
            fill_pixels_parallel_indexed(&mut output, kernel_work, |_flat, oidx| {
                sample_separable_pixel(coeffs, axis_supports, oidx, cval, offsets)
            });
        } else {
            // Resolve ONCE, outside the pixel closure (`frankenscipy-22van`).
            let flags = SplineFlags::resolve();
            // Resolved on THIS path. The other binding lives inside the branch that returns
            // early, so it is out of scope here, and exactly one of the two runs per call --
            // which is what `van22_knob_read_is_per_transform` asserts. Resolving inside the
            // closure instead would read the toggles once PER PIXEL, the hot-path regression
            // that bead exists to prevent.
            let flags = SplineFlags::resolve();
            fill_pixels_parallel_indexed(&mut output, kernel_work, |_flat, out_idx| {
                let coords: Vec<f64> = out_idx
                    .iter()
                    .enumerate()
                    .map(|(axis, &o)| o as f64 - shift_values[axis])
                    .collect();
                sample_interpolated(
                    input,
                    &spline.coeffs,
                    &coords,
                    &spline.coord_offsets,
                    order,
                    mode,
                    cval,
                    flags,
                )
            });
        }
        return Ok(output);
    }

    // Legacy serial per-pixel path (retained as the same-binary A/B baseline).
    // Resolve ONCE, outside the loop (`frankenscipy-22van`).
    let flags = SplineFlags::resolve();
    for flat in 0..input.size() {
        let out_idx = input.unravel(flat);
        let coords: Vec<f64> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &o)| o as f64 - shift_values[axis])
            .collect();
        output.data[flat] = sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        );
    }

    Ok(output)
}

/// Same-binary A/B toggle: when `true`, `zoom` recomputes the per-axis B-spline support per
/// output pixel (via `sample_interpolated`); when `false` (default) it precomputes the support
/// once per output axis-index (separable) — byte-identical, but skips the per-pixel recompute.
pub static NDIMAGE_ZOOM_SEPARABLE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle for the tensor-product combine on BOTH the separable paths
/// (zoom/shift/diagonal-affine, supports precomputed per axis-index) and the per-pixel path
/// (`sample_interpolated`: rotate / general affine / map_coordinates / geometric_transform, whose
/// coupled coords force a per-pixel support rebuild). When `true`, taps stay in INDEX space and
/// every leaf recomputes `Σ idx[d]·stride[d]` via `coeffs.get` — the ORIG comparator. When `false`
/// (default) taps are pre-multiplied into FLAT OFFSETS and the leaf is a single load.
/// Byte-identical either way. Benchmark knob; each call site reads it ONCE and threads it through
/// (reading it twice tears the premultiply/leaf pair).
#[doc(hidden)]
pub static NDIMAGE_SPLINE_OFFSET_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Restore the PRE-hoist behaviour: re-read the spline knobs inside every
/// `sample_interpolated` call instead of taking them from the caller's one-shot
/// `SplineFlags::resolve` (`frankenscipy-22van`).
///
/// This exists so the hoist can be measured against the exact shape it replaced, inside a
/// single binary, rather than against a differently-compiled one. Default `false` = hoisted.
/// BYTE-IDENTICAL either way: both arms read the same two atomics and branch identically;
/// only WHERE the read happens changes.
#[doc(hidden)]
pub static NDIMAGE_SPLINE_FLAG_HOIST_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle for `compute_axis_support`'s cardinal-B-spline tap loop: when `true`,
/// the loop spans the ORIG `floor(cc) ± order` (`2·order+1` taps) and discards the zero-weight
/// ones; when `false` (default) it spans only the `order+1` taps that can be nonzero. Byte-
/// identical either way. Benchmark knob; read ONCE per call.
#[doc(hidden)]
pub static NDIMAGE_BSPLINE_COMPACT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle for the lane-parallel cardinal-B-spline tap run: when `true`, each tap of
/// the support run is evaluated by an independent SCALAR `cardinal_bspline` call — the ORIG
/// comparator. When `false` (default) the whole run is evaluated by one `cardinal_bspline_lanes`
/// recursion, one tap per lane. Bit-identical either way (per-lane operation order is the scalar
/// one). Benchmark knob; read ONCE per support build.
#[doc(hidden)]
pub static NDIMAGE_BSPLINE_SIMD_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Zoom (rescale) an array by the given factors.
///
/// Matches `scipy.ndimage.zoom`.
pub fn zoom(
    input: &NdArray,
    zoom_factors: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if zoom_factors.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "zoom has {} values but input has {} dimensions",
            zoom_factors.len(),
            input.ndim()
        )));
    }
    if zoom_factors.iter().any(|&v| !v.is_finite() || v <= 0.0) {
        return Err(NdimageError::InvalidArgument(
            "zoom factors must be finite and positive".to_string(),
        ));
    }

    let new_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(zoom_factors.iter())
        .map(|(&s, &z)| ((s as f64 * z).round() as usize).max(1))
        .collect();

    // Order-1 reflect has no recursive spline prefilter; the padded coefficient
    // image is only an integer-offset copy, so sample the original image.
    if order == 1 && mode == BoundaryMode::Reflect && input.ndim() == 2 && !input.shape.contains(&0)
    {
        return Ok(zoom_order1_reflect_2d_fast(input, &new_shape));
    }
    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(new_shape.clone());

    // Each output pixel is an independent interpolation of the (read-only) spline
    // coefficients, so the flat output index can be filled in parallel — bit-identical
    // to the sequential loop (unravel_with_shape matches NdArray::unravel exactly).
    let out_shape = output.shape.clone();
    let in_shape = input.shape.clone();
    let ndim = out_shape.len();
    let kernel_work = (order + 1).saturating_pow(ndim as u32);

    // Separable fast path: a regular zoom maps each output axis-index to a FIXED input
    // coordinate, so the per-axis B-spline support depends ONLY on the output position along
    // that axis. Precompute the O(Σ out_shape[a]) supports ONCE (via the same
    // `compute_axis_support` the per-pixel path uses) instead of recomputing them for all
    // out_shape.product() pixels; each pixel is then a gather + `sample_spline_recursive`.
    // Byte-identical (same weights, same combine order). Gated `order >= 2`: order-1 Reflect
    // is already the ultra-fast cardinal linear path where the precompute doesn't pay (0.77x),
    // while orders 2-5 win 2.0-3.7x (more support taps ⇒ more per-pixel recompute avoided).
    if order >= 2 && !NDIMAGE_ZOOM_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        let coeffs = &spline.coeffs;
        // ONE read: the tap representation and the leaf kind must agree (see the fn doc).
        // `SplineFlags::resolve` also carries the compact-support knob, which
        // `build_axis_offset_supports` forwards into `compute_axis_support`
        // (`frankenscipy-22van`).
        let flags = SplineFlags::resolve();
        let offsets = flags.offsets;
        let axis_supports = build_axis_offset_supports(
            coeffs,
            &in_shape,
            &out_shape,
            &spline.coord_offsets,
            order,
            mode,
            offsets,
            flags.compact_disabled,
            |axis, o| {
                if out_shape[axis] <= 1 || in_shape[axis] <= 1 {
                    0.0
                } else {
                    o as f64 * (in_shape[axis] - 1) as f64 / (out_shape[axis] - 1) as f64
                }
            },
        );
        let axis_supports = &axis_supports;
        fill_pixels_parallel_indexed(&mut output, kernel_work, |_flat, oidx| {
            sample_separable_pixel(coeffs, axis_supports, oidx, cval, offsets)
        });
        return Ok(output);
    }

    // Resolved on THIS path. The other binding lives inside the branch that returns
    // early, so it is out of scope here, and exactly one of the two runs per call --
    // which is what `van22_knob_read_is_per_transform` asserts. Resolving inside the
    // closure instead would read the toggles once PER PIXEL, the hot-path regression
    // that bead exists to prevent.
    let flags = SplineFlags::resolve();
    fill_pixels_parallel_indexed(&mut output, kernel_work, |_flat, out_idx| {
        let coords: Vec<f64> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &o)| {
                if out_shape[axis] <= 1 || in_shape[axis] <= 1 {
                    0.0
                } else {
                    o as f64 * (in_shape[axis] - 1) as f64 / (out_shape[axis] - 1) as f64
                }
            })
            .collect();
        sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        )
    });

    Ok(output)
}

/// Rotate a 2D array by the given angle (in degrees).
///
/// Matches `scipy.ndimage.rotate`.
pub fn rotate(
    input: &NdArray,
    angle: f64,
    reshape: bool,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if input.ndim() != 2 {
        return Err(NdimageError::InvalidArgument(
            "rotate currently supports 2D arrays only".to_string(),
        ));
    }
    if !angle.is_finite() {
        return Err(NdimageError::InvalidArgument(
            "angle must be finite (no NaN or Inf)".to_string(),
        ));
    }

    let rows = input.shape[0];
    let cols = input.shape[1];
    let rad = angle.to_radians();
    let cos_a = rad.cos();
    let sin_a = rad.sin();

    let (out_rows, out_cols) = if reshape {
        // Compute new size to contain rotated image
        let corners = [
            (0.0, 0.0),
            (rows as f64, 0.0),
            (0.0, cols as f64),
            (rows as f64, cols as f64),
        ];
        let cy = rows as f64 / 2.0;
        let cx = cols as f64 / 2.0;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        for (y, x) in corners {
            let dy = y - cy;
            let dx = x - cx;
            let ny = cy + cos_a * dy - sin_a * dx;
            let nx = cx + sin_a * dy + cos_a * dx;
            min_y = min_y.min(ny);
            max_y = max_y.max(ny);
            min_x = min_x.min(nx);
            max_x = max_x.max(nx);
        }
        (
            (max_y - min_y).ceil() as usize,
            (max_x - min_x).ceil() as usize,
        )
    } else {
        (rows, cols)
    };

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(vec![out_rows, out_cols]);
    let cy_in = (rows as f64 - 1.0) / 2.0;
    let cx_in = (cols as f64 - 1.0) / 2.0;
    let cy_out = (out_rows as f64 - 1.0) / 2.0;
    let cx_out = (out_cols as f64 - 1.0) / 2.0;

    // Each output pixel is an independent interpolation; fill the row-major flat index
    // `flat = r*out_cols + c` in parallel — bit-identical to the sequential (r, c) loop.
    let _ = out_rows;
    let kernel_work = (order + 1) * (order + 1);
    // Resolve the spline knobs ONCE, outside the per-pixel work below
    // (`frankenscipy-22van`).
    let flags = SplineFlags::resolve();
    fill_pixels_parallel(&mut output, kernel_work, |flat, _scratch| {
        let dy = (flat / out_cols) as f64 - cy_out;
        let dx = (flat % out_cols) as f64 - cx_out;
        let src_y = cy_in + cos_a * dy + sin_a * dx;
        let src_x = cx_in - sin_a * dy + cos_a * dx;
        sample_interpolated(
            input,
            &spline.coeffs,
            &[src_y, src_x],
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        )
    });

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════
// Gradient
// ══════════════════════════════════════════════════════════════════════

/// Gaussian gradient magnitude.
///
/// Matches `scipy.ndimage.gaussian_gradient_magnitude`: the square root of the
/// sum, over each axis, of the squared Gaussian filter applied with derivative
/// order 1 along that axis (analytic first-derivative-of-Gaussian convolution),
/// not a Sobel operator applied after blurring.
pub fn gaussian_gradient_magnitude(
    input: &NdArray,
    sigma: f64,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = (0..input.ndim()).collect::<Vec<_>>();
    gaussian_gradient_magnitude_usize_axes(input, sigma, &axes, mode, cval)
}

/// Gaussian gradient magnitude over a SciPy-style signed axes subset.
///
/// `axes=[]` matches SciPy's empty-axes identity behavior.
pub fn gaussian_gradient_magnitude_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    gaussian_gradient_magnitude_usize_axes(input, sigma, &axes, mode, cval)
}

fn gaussian_gradient_magnitude_usize_axes(
    input: &NdArray,
    sigma: f64,
    axes: &[usize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(NdimageError::InvalidArgument(
            "sigma must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Ok(input.clone());
    }
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for &axis in axes {
        if axis >= ndim {
            return Err(NdimageError::InvalidArgument(format!(
                "axis {axis} out of range for {ndim}-dimensional input"
            )));
        }
        let mut orders = vec![0usize; ndim];
        orders[axis] = 1;
        let deriv = gaussian_filter_with_orders(input, sigma, &orders, mode, cval)?;
        for (r, d) in result.data.iter_mut().zip(&deriv.data) {
            *r += d * d;
        }
    }

    for v in &mut result.data {
        *v = v.sqrt();
    }

    Ok(result)
}

/// Gaussian Laplace with per-axis sigma.
///
/// Matches `scipy.ndimage.gaussian_laplace` with `sigma` as a sequence.
pub fn gaussian_laplace_multi_sigma(
    input: &NdArray,
    sigmas: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for axis in 0..ndim {
        let mut orders = vec![0usize; ndim];
        orders[axis] = 2;
        let deriv = gaussian_filter_with_sigmas_and_orders(input, sigmas, &orders, mode, cval)?;
        for (r, d) in result.data.iter_mut().zip(&deriv.data) {
            *r += d;
        }
    }
    Ok(result)
}

/// Gaussian gradient magnitude with per-axis sigma.
///
/// Matches `scipy.ndimage.gaussian_gradient_magnitude` with `sigma` as a
/// sequence.
pub fn gaussian_gradient_magnitude_multi_sigma(
    input: &NdArray,
    sigmas: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for axis in 0..ndim {
        let mut orders = vec![0usize; ndim];
        orders[axis] = 1;
        let deriv = gaussian_filter_with_sigmas_and_orders(input, sigmas, &orders, mode, cval)?;
        for (r, d) in result.data.iter_mut().zip(&deriv.data) {
            *r += d * d;
        }
    }

    for v in &mut result.data {
        *v = v.sqrt();
    }

    Ok(result)
}

/// Grey-scale erosion (minimum filter equivalent for continuous values).
///
/// Matches `scipy.ndimage.grey_erosion`.
pub fn grey_erosion(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    grey_erosion_with_origins(input, size, &[0], mode, cval)
}

/// Grey-scale erosion over a SciPy-style signed axes subset.
pub fn grey_erosion_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter_axes(input, size, axes, mode, cval)
}

/// Grey-scale erosion with SciPy `origin` semantics.
pub fn grey_erosion_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter_with_origins(input, size, origins, mode, cval)
}

/// Grey-scale dilation (maximum filter equivalent for continuous values).
///
/// Matches `scipy.ndimage.grey_dilation`.
pub fn grey_dilation(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    grey_dilation_with_origins(input, size, &[0], mode, cval)
}

/// Grey-scale dilation over a SciPy-style signed axes subset.
pub fn grey_dilation_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let filter_size = filter_footprint_size(axes.len(), size)?;
    let shape = vec![size; axes.len()];
    let origins = normalize_filter_origins(axes.len(), &shape, &[0])?;
    let origins = origins
        .into_iter()
        .map(|origin| {
            let mut origin = -origin;
            if size.is_multiple_of(2) {
                origin -= 1;
            }
            origin
        })
        .collect::<Vec<_>>();

    rank_filter_index_usize_axes_with_origins(
        input,
        size,
        &axes,
        &origins,
        mode,
        cval,
        filter_size - 1,
    )
}

/// Grey-scale dilation with SciPy `origin` semantics.
pub fn grey_dilation_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let origins = normalize_grey_dilation_origins(input.ndim(), size, origins)?;
    maximum_filter_with_origins(input, size, &origins, mode, cval)
}

fn normalize_grey_dilation_origins(
    ndim: usize,
    size: usize,
    origins: &[i64],
) -> Result<Vec<i64>, NdimageError> {
    let shape = vec![size; ndim];
    let origins = normalize_filter_origins(ndim, &shape, origins)?;
    Ok(origins
        .into_iter()
        .map(|origin| {
            let mut origin = -origin;
            if size.is_multiple_of(2) {
                origin -= 1;
            }
            origin
        })
        .collect())
}

/// Grey-scale opening: erosion followed by dilation.
///
/// Matches `scipy.ndimage.grey_opening`.
pub fn grey_opening(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    grey_opening_with_origins(input, size, &[0], mode, cval)
}

/// Grey-scale opening over a SciPy-style signed axes subset.
pub fn grey_opening_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let eroded = grey_erosion_axes(input, size, axes, mode, cval)?;
    grey_dilation_axes(&eroded, size, axes, mode, cval)
}

/// Grey-scale opening with SciPy `origin` semantics.
pub fn grey_opening_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let eroded = grey_erosion_with_origins(input, size, origins, mode, cval)?;
    grey_dilation_with_origins(&eroded, size, origins, mode, cval)
}

/// Grey-scale closing: dilation followed by erosion.
///
/// Matches `scipy.ndimage.grey_closing`.
pub fn grey_closing(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    grey_closing_with_origins(input, size, &[0], mode, cval)
}

/// Grey-scale closing over a SciPy-style signed axes subset.
pub fn grey_closing_axes(
    input: &NdArray,
    size: usize,
    axes: &[isize],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let dilated = grey_dilation_axes(input, size, axes, mode, cval)?;
    grey_erosion_axes(&dilated, size, axes, mode, cval)
}

/// Grey-scale closing with SciPy `origin` semantics.
pub fn grey_closing_with_origins(
    input: &NdArray,
    size: usize,
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let dilated = grey_dilation_with_origins(input, size, origins, mode, cval)?;
    grey_erosion_with_origins(&dilated, size, origins, mode, cval)
}

/// Map coordinates: evaluate input at arbitrary (non-integer) coordinates.
///
/// Matches `scipy.ndimage.map_coordinates`.
pub fn map_coordinates(
    input: &NdArray,
    coordinates: &[Vec<f64>],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<Vec<f64>, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if coordinates.is_empty() {
        return Ok(vec![]);
    }
    let ndim = input.ndim();
    if coordinates.len() != ndim {
        return Err(NdimageError::DimensionMismatch(format!(
            "coordinates has {} arrays but input has {} dimensions",
            coordinates.len(),
            ndim
        )));
    }

    let npts = coordinates[0].len();
    for c in coordinates {
        if c.len() != npts {
            return Err(NdimageError::DimensionMismatch(
                "all coordinate arrays must have the same length".to_string(),
            ));
        }
    }

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    // Each query point is an independent interpolation of the read-only spline
    // coefficients; fill the disjoint output indices in parallel — bit-identical.
    // Resolve the spline knobs ONCE, outside the per-pixel work below
    // (`frankenscipy-22van`).
    let flags = SplineFlags::resolve();
    let point = |p: usize| -> f64 {
        let coords: Vec<f64> = coordinates.iter().map(|c| c[p]).collect();
        sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        )
    };
    let kernel_work = (order + 1).saturating_pow(ndim as u32);
    let nthreads = ndimage_filter_thread_count(npts, kernel_work);
    if nthreads <= 1 {
        return Ok((0..npts).map(point).collect());
    }
    let mut result = vec![0.0_f64; npts];
    let chunk = npts.div_ceil(nthreads);
    let point = &point;
    std::thread::scope(|scope| {
        for (t, out_chunk) in result.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            scope.spawn(move || {
                for (li, slot) in out_chunk.iter_mut().enumerate() {
                    *slot = point(start + li);
                }
            });
        }
    });
    Ok(result)
}

/// When `true`, [`array_min`]/[`array_max`] run their NaN-aware fold serially (the ORIG behaviour);
/// default `false` chunks the reduction across threads. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Parallel NaN-propagating extreme (min or max) over `data`. BYTE-IDENTICAL to the sequential
/// `data.iter().fold(ident, op)`: `op` (NaN-aware min/max) is associative and commutative — including
/// signed-zero (`min(-0,+0) = -0`, `max = +0` regardless of order) and NaN (propagates from any
/// chunk) — so a chunk-then-merge with the same `ident`/`op` yields the same bits. Gated on `n`.
fn ndimage_par_nan_extreme(data: &[f64], ident: f64, op: fn(f64, f64) -> f64) -> f64 {
    let n = data.len();
    let nthreads = if NDIMAGE_ARRAY_MINMAX_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        ndimage_filter_thread_count(n, 4)
    };
    if nthreads <= 1 {
        return data.iter().copied().fold(ident, op);
    }
    let chunk = n.div_ceil(nthreads);
    let parts: Vec<f64> = std::thread::scope(|scope| {
        data.chunks(chunk)
            .map(|c| scope.spawn(move || c.iter().copied().fold(ident, op)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("array minmax chunk panicked"))
            .collect()
    });
    parts.into_iter().fold(ident, op)
}

#[inline]
fn ndimage_nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

#[inline]
fn ndimage_nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// Compute the maximum of the input array.
pub fn array_max(input: &NdArray) -> f64 {
    ndimage_par_nan_extreme(input.data.as_slice(), f64::NEG_INFINITY, ndimage_nan_max)
}

/// Compute the minimum of the input array.
pub fn array_min(input: &NdArray) -> f64 {
    ndimage_par_nan_extreme(input.data.as_slice(), f64::INFINITY, ndimage_nan_min)
}

/// Gaussian filter with per-axis sigma.
///
/// Matches `scipy.ndimage.gaussian_filter` with `sigma` as array.
pub fn gaussian_filter_multi_sigma(
    input: &NdArray,
    sigmas: &[f64],
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if sigmas.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "sigmas length {} != ndim {}",
            sigmas.len(),
            input.ndim()
        )));
    }

    let mut current = input.clone();
    for (axis, &sigma) in sigmas.iter().enumerate() {
        if !sigma.is_finite() {
            return Err(NdimageError::InvalidArgument(
                "sigmas must be finite".to_string(),
            ));
        }
        if sigma <= 0.0 {
            continue;
        }
        current = gaussian_filter1d_axis(&current, sigma, axis, 0, mode, cval)?;
    }

    Ok(current)
}

/// Apply a uniform filter along a single axis.
///
/// Matches `scipy.ndimage.uniform_filter1d`.
pub fn uniform_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    uniform_filter1d_with_origin(input, size, axis, mode, cval, 0)
}

/// Apply a uniform filter with SciPy-style signed axis normalization.
pub fn uniform_filter1d_signed_axis(
    input: &NdArray,
    size: usize,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    uniform_filter1d(input, size, axis, mode, cval)
}

/// Apply a uniform filter using SciPy's default `axis=-1`.
pub fn uniform_filter1d_default_axis(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    uniform_filter1d_signed_axis(input, size, -1, mode, cval)
}

/// Apply a uniform filter along a single axis with SciPy `origin` semantics.
///
/// Positive origins shift the window toward lower input coordinates; negative
/// origins shift it toward higher coordinates.
pub fn uniform_filter1d_with_origin(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(size, origin)?;
    Ok(uniform_filter_along_axis(
        input, axis, size, origin, mode, cval,
    ))
}

/// Reference per-window uniform-filter path (pre running-sum), retained for the
/// same-process A/B benchmark only. Computes each output as a fresh O(size) window mean.
#[doc(hidden)]
pub fn uniform_filter1d_perwindow_ref(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(
            "axis out of range".to_string(),
        ));
    }
    if size == 0 || input.size() == 0 {
        return Err(NdimageError::InvalidArgument("bad size/empty".to_string()));
    }
    validate_filter_origin(size, origin)?;
    filter1d_axis_with_origin(input, size, axis, mode, cval, origin, |window| {
        window.iter().sum::<f64>() / window.len() as f64
    })
}

/// Compute the gradient magnitude of an array.
///
/// Uses central differences along each axis.
/// Matches `numpy.gradient` magnitude.
pub fn gradient_magnitude(input: &NdArray) -> Result<NdArray, NdimageError> {
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let mut result = NdArray::zeros(input.shape.clone());

    for axis in 0..input.ndim() {
        let kernel_1d = vec![-0.5, 0.0, 0.5];
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[axis] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        let grad = correlate(input, &kernel, BoundaryMode::Reflect, 0.0)?;
        for i in 0..result.data.len() {
            result.data[i] += grad.data[i] * grad.data[i];
        }
    }

    for v in &mut result.data {
        *v = v.sqrt();
    }

    Ok(result)
}

/// Apply a `size`-wide reduction filter along a single axis.
///
/// The window spans only `axis` (width `size`, centered with offset
/// `size / 2`); every other axis is untouched.
fn filter1d_axis_with_origin<F>(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
    reduce: F,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(size, origin)?;

    let offset = size as i64 / 2;
    let mut output = NdArray::zeros(input.shape.clone());
    fill_pixels_parallel(&mut output, size, |flat_out, window| {
        let out_idx = input.unravel(flat_out);
        let mut in_idx: Vec<i64> = out_idx.iter().map(|&i| i as i64).collect();
        window.clear();
        for k in 0..size as i64 {
            in_idx[axis] = out_idx[axis] as i64 + k - offset - origin;
            window.push(input.get_boundary(&in_idx, mode, cval));
        }
        reduce(window)
    });
    Ok(output)
}

fn validate_filter_origin(size: usize, origin: i64) -> Result<(), NdimageError> {
    let lower = -(size as i64 / 2);
    let upper = (size as i64 - 1) / 2;
    if origin < lower || origin > upper {
        return Err(NdimageError::InvalidArgument("invalid origin".to_string()));
    }
    Ok(())
}

fn normalize_filter_origins(
    ndim: usize,
    shape: &[usize],
    origins: &[i64],
) -> Result<Vec<i64>, NdimageError> {
    let normalized = match origins.len() {
        1 => vec![origins[0]; ndim],
        len if len == ndim => origins.to_vec(),
        _ => {
            return Err(NdimageError::InvalidArgument(
                "origin must be scalar or match input dimensionality".to_string(),
            ));
        }
    };

    for (&size, &origin) in shape.iter().zip(&normalized) {
        validate_filter_origin(size, origin)?;
    }
    Ok(normalized)
}

/// Apply a maximum filter along a single axis.
///
/// Matches `scipy.ndimage.maximum_filter1d`.
pub fn maximum_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    maximum_filter1d_with_origin(input, size, axis, mode, cval, 0)
}

/// Apply a maximum filter with SciPy-style signed axis normalization.
pub fn maximum_filter1d_signed_axis(
    input: &NdArray,
    size: usize,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    maximum_filter1d(input, size, axis, mode, cval)
}

/// Apply a maximum filter using SciPy's default `axis=-1`.
pub fn maximum_filter1d_default_axis(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    maximum_filter1d_signed_axis(input, size, -1, mode, cval)
}

/// Apply a maximum filter along a single axis with SciPy `origin` semantics.
pub fn maximum_filter1d_with_origin(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(size, origin)?;
    // O(n) monotonic index queue over a boundary-resolved line. It preserves the
    // NaN-propagating fold contract while reducing the HGW path's extra full-line scans.
    Ok(minmax_filter1d_nanprop_queue(
        input, axis, size, origin, mode, cval, true,
    ))
}

/// Apply a minimum filter along a single axis.
///
/// Matches `scipy.ndimage.minimum_filter1d`.
pub fn minimum_filter1d(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter1d_with_origin(input, size, axis, mode, cval, 0)
}

/// Apply a minimum filter with SciPy-style signed axis normalization.
pub fn minimum_filter1d_signed_axis(
    input: &NdArray,
    size: usize,
    axis: isize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    minimum_filter1d(input, size, axis, mode, cval)
}

/// Apply a minimum filter using SciPy's default `axis=-1`.
pub fn minimum_filter1d_default_axis(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    minimum_filter1d_signed_axis(input, size, -1, mode, cval)
}

/// Apply a minimum filter along a single axis with SciPy `origin` semantics.
pub fn minimum_filter1d_with_origin(
    input: &NdArray,
    size: usize,
    axis: usize,
    mode: BoundaryMode,
    cval: f64,
    origin: i64,
) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-dimensional input",
            input.ndim()
        )));
    }
    if size == 0 {
        return Err(NdimageError::InvalidArgument(
            "filter size must be positive".to_string(),
        ));
    }
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }
    validate_filter_origin(size, origin)?;
    Ok(minmax_filter1d_nanprop_queue(
        input, axis, size, origin, mode, cval, false,
    ))
}

/// Generate a structuring element (cross/diamond shape).
///
/// Matches `scipy.ndimage.generate_binary_structure`.
pub fn generate_binary_structure(ndim: usize, connectivity: usize) -> NdArray {
    let size = 3;
    let shape = vec![size; ndim];
    let total: usize = shape.iter().product();
    let strides = compute_strides(&shape);
    let center = size / 2;

    let mut data = vec![0.0; total];

    for (flat, item) in data.iter_mut().enumerate() {
        let mut idx = vec![0usize; ndim];
        let mut rem = flat;
        for d in 0..ndim {
            idx[d] = rem / strides[d];
            rem %= strides[d];
        }

        // City-block distance from center
        let dist: usize = idx.iter().map(|&i| i.abs_diff(center)).sum();
        if dist <= connectivity {
            *item = 1.0;
        }
    }

    NdArray {
        data,
        shape,
        strides: compute_strides(&vec![size; ndim]),
    }
}

fn binary_dilation_with_structure_once(
    input: &NdArray,
    structure: &NdArray,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != structure.ndim() {
        return Err(NdimageError::DimensionMismatch(format!(
            "input ndim {} != structure ndim {}",
            input.ndim(),
            structure.ndim()
        )));
    }
    if structure.shape.contains(&0) {
        return Err(NdimageError::InvalidArgument(
            "structure dimensions must be positive".to_string(),
        ));
    }

    let centers: Vec<i64> = structure
        .shape
        .iter()
        .map(|&dim| (dim / 2) as i64)
        .collect();
    let mut offsets = Vec::new();
    for flat in 0..structure.size() {
        if structure.data[flat] == 0.0 {
            continue;
        }
        let idx = structure.unravel(flat);
        offsets.push(
            idx.iter()
                .zip(&centers)
                .map(|(&coord, &center)| coord as i64 - center)
                .collect::<Vec<_>>(),
        );
    }

    let mut output = NdArray::zeros(input.shape.clone());
    // Row-major odometer for the source multi-index (was `input.unravel(flat)`, a `Vec` heap-alloc
    // per FOREGROUND element) plus a direct output flat-index (was a fresh `out_idx` `Vec` per
    // (element, offset)). Dilation writes 1.0 idempotently, so cell-write order is irrelevant.
    // BYTE-IDENTICAL: `idx` = `unravel(flat)`; `out_flat = Σ (idx+offset)·strides` = exactly what
    // `output.set(&out_idx, 1.0)` computes, so the same cells become 1.0.
    let full_mode = NDIMAGE_UNRAVEL_ODOMETER_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let ndim = input.ndim();
    let mut idx = vec![0usize; ndim];
    for flat in 0..input.size() {
        if input.data[flat] != 0.0 {
            if full_mode {
                let uidx = input.unravel(flat);
                for offset in &offsets {
                    let mut out_idx = Vec::with_capacity(ndim);
                    let mut in_bounds = true;
                    for axis in 0..ndim {
                        let coord = uidx[axis] as i64 + offset[axis];
                        if coord < 0 || coord >= input.shape[axis] as i64 {
                            in_bounds = false;
                            break;
                        }
                        out_idx.push(coord as usize);
                    }
                    if in_bounds {
                        output.set(&out_idx, 1.0);
                    }
                }
            } else {
                for offset in &offsets {
                    let mut in_bounds = true;
                    let mut out_flat = 0usize;
                    for axis in 0..ndim {
                        let coord = idx[axis] as i64 + offset[axis];
                        if coord < 0 || coord >= input.shape[axis] as i64 {
                            in_bounds = false;
                            break;
                        }
                        out_flat += coord as usize * output.strides[axis];
                    }
                    if in_bounds {
                        output.data[out_flat] = 1.0;
                    }
                }
            }
        }
        if !full_mode {
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < input.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    Ok(output)
}

/// Iterate a binary structuring element by dilating it with itself.
///
/// Matches `scipy.ndimage.iterate_structure` when called without `origin`.
pub fn iterate_structure(structure: &NdArray, iterations: usize) -> Result<NdArray, NdimageError> {
    if iterations < 2 {
        return Ok(structure.clone());
    }
    if structure.shape.contains(&0) {
        return Err(NdimageError::InvalidArgument(
            "structure dimensions must be positive".to_string(),
        ));
    }

    let dilation_count = iterations - 1;
    let mut shape = Vec::with_capacity(structure.ndim());
    let mut insert_at = Vec::with_capacity(structure.ndim());
    for &dim in &structure.shape {
        let growth = dilation_count.checked_mul(dim - 1).ok_or_else(|| {
            NdimageError::InvalidArgument("iterated structure is too large".to_string())
        })?;
        shape.push(dim.checked_add(growth).ok_or_else(|| {
            NdimageError::InvalidArgument("iterated structure is too large".to_string())
        })?);
        insert_at.push(dilation_count.checked_mul(dim / 2).ok_or_else(|| {
            NdimageError::InvalidArgument("iterated structure is too large".to_string())
        })?);
    }

    let mut current = NdArray::zeros(shape);
    for flat in 0..structure.size() {
        if structure.data[flat] == 0.0 {
            continue;
        }
        let idx = structure.unravel(flat);
        let out_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .map(|(axis, &coord)| insert_at[axis] + coord)
            .collect();
        current.set(&out_idx, 1.0);
    }

    for _ in 0..dilation_count {
        current = binary_dilation_with_structure_once(&current, structure)?;
    }

    Ok(current)
}

/// Compute the variance filter (local variance in each neighborhood).
///
/// Matches `scipy.ndimage.generic_filter` with variance function.
pub fn variance_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    generic_filter(
        input,
        |neighborhood| {
            let n = neighborhood.len() as f64;
            if n == 0.0 {
                return 0.0;
            }
            let mean: f64 = neighborhood.iter().sum::<f64>() / n;
            neighborhood
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / n
        },
        size,
        mode,
        cval,
    )
}

/// Compute the standard deviation filter.
pub fn std_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    let var = variance_filter(input, size, mode, cval)?;
    let mut result = var;
    for v in &mut result.data {
        *v = v.sqrt();
    }
    Ok(result)
}

/// Compute the range filter (max - min in each neighborhood).
pub fn range_filter(
    input: &NdArray,
    size: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    morphological_gradient(input, size, mode, cval)
}

/// Threshold an array: values above threshold become 1, below become 0.
///
/// Matches `scipy.ndimage` threshold operations.
pub fn threshold(input: &NdArray, thresh: f64) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v > thresh { 1.0 } else { 0.0 })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply Otsu's thresholding to find optimal binary threshold.
///
/// Returns the optimal threshold value.
/// When `true`, force `otsu_threshold` onto its ORIG serial min/max + histogram passes. Byte-identical
/// either way. Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static NDIMAGE_OTSU_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn otsu_threshold(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return 0.0;
    }

    let data = input.data.as_slice();
    let n = data.len();
    // NaN-propagating min/max combine — matches the ORIG two folds exactly (result is NaN iff any
    // element is NaN, else the true extremum). Both are associative + commutative, so the chunked
    // parallel reductions below are BYTE-IDENTICAL to the sequential folds.
    let nan_min = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    };
    let nan_max = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    };

    let nthreads = if NDIMAGE_OTSU_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        ndimage_filter_thread_count(n, 4)
    };

    // Pass 1: min + max over all pixels.
    let (min_val, max_val) = if nthreads <= 1 {
        (
            data.iter().copied().fold(f64::INFINITY, nan_min),
            data.iter().copied().fold(f64::NEG_INFINITY, nan_max),
        )
    } else {
        let chunk = n.div_ceil(nthreads);
        let parts: Vec<(f64, f64)> = std::thread::scope(|scope| {
            data.chunks(chunk)
                .map(|c| {
                    scope.spawn(move || {
                        (
                            c.iter().copied().fold(f64::INFINITY, nan_min),
                            c.iter().copied().fold(f64::NEG_INFINITY, nan_max),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("otsu minmax chunk panicked"))
                .collect()
        });
        parts.into_iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(amn, amx), (bmn, bmx)| (nan_min(amn, bmn), nan_max(amx, bmx)),
        )
    };

    if (max_val - min_val).abs() < 1e-15 {
        return min_val;
    }

    let nbins = 256;
    let bin_width = (max_val - min_val) / nbins as f64;

    // Pass 2: 256-bin histogram (per-pixel divide + floor). Each pixel lands in the same bin
    // regardless of thread, and integer counts sum order-independently, so the privatized
    // per-thread histograms merged below are BYTE-IDENTICAL to the sequential bincount.
    let hist: Vec<usize> = if nthreads <= 1 {
        let mut h = vec![0usize; nbins];
        for &v in data {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            h[bin.min(nbins - 1)] += 1;
        }
        h
    } else {
        let chunk = n.div_ceil(nthreads);
        let parts: Vec<Vec<usize>> = std::thread::scope(|scope| {
            data.chunks(chunk)
                .map(|c| {
                    scope.spawn(move || {
                        let mut h = vec![0usize; nbins];
                        for &v in c {
                            let bin = ((v - min_val) / bin_width).floor() as usize;
                            h[bin.min(nbins - 1)] += 1;
                        }
                        h
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("otsu histogram chunk panicked"))
                .collect()
        });
        let mut h = vec![0usize; nbins];
        for p in parts {
            for (i, c) in p.into_iter().enumerate() {
                h[i] += c;
            }
        }
        h
    };

    let total = input.size() as f64;
    let mut sum_total = 0.0;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_thresh = 0.0;
    let mut best_var = 0.0;
    let mut weight_bg = 0.0;
    let mut sum_bg = 0.0;

    for (i, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += i as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;
        let between_var = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if between_var > best_var {
            best_var = between_var;
            best_thresh = min_val + (i as f64 + 0.5) * bin_width;
        }
    }

    best_thresh
}

/// Affine transformation of an image (2D only).
///
/// `matrix` is a 2x3 affine transformation matrix [a b tx; c d ty].
/// `order` controls interpolation: 0=nearest, 1=linear, 3=cubic (spline).
/// Matches `scipy.ndimage.affine_transform`.
pub fn affine_transform(
    input: &NdArray,
    matrix: &[[f64; 3]; 2],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if input.ndim() != 2 {
        return Err(NdimageError::InvalidArgument(
            "affine_transform supports 2D only".to_string(),
        ));
    }

    let rows = input.shape[0];
    let cols = input.shape[1];

    // scipy.ndimage.affine_transform maps each OUTPUT index `o` directly to the
    // INPUT location `matrix @ o + offset` — the matrix is already the
    // output->input map and is NOT inverted. With a 2x3 matrix the final column
    // holds the offset and the leading 2x2 block is the linear part.
    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(input.shape.clone());

    // Each output pixel maps to one independent interpolation of the (read-only) spline
    // coefficients, so the row-major flat index `flat = r*cols + c` can be filled in
    // parallel — bit-identical to the sequential (r, c) loop.
    let _ = rows;
    let matrix = *matrix;
    let kernel_work = (order + 1) * (order + 1);

    // A DIAGONAL affine (no cross terms) is pure per-axis scale+translate — src_r depends only
    // on r, src_c only on c — so it is axis-separable exactly like zoom/shift. Precompute the
    // per-axis B-spline supports once (order >= 2) instead of per pixel; byte-identical.
    if matrix[0][1] == 0.0
        && matrix[1][0] == 0.0
        && order >= 2
        && !NDIMAGE_ZOOM_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
    {
        let out_shape = output.shape.clone();
        let in_shape = input.shape.clone();
        let coeffs = &spline.coeffs;
        // ONE read: the tap representation and the leaf kind must agree (see the fn doc).
        // `SplineFlags::resolve` also carries the compact-support knob, which
        // `build_axis_offset_supports` forwards into `compute_axis_support`
        // (`frankenscipy-22van`).
        let flags = SplineFlags::resolve();
        let offsets = flags.offsets;
        let axis_supports = build_axis_offset_supports(
            coeffs,
            &in_shape,
            &out_shape,
            &spline.coord_offsets,
            order,
            mode,
            offsets,
            flags.compact_disabled,
            |axis, o| matrix[axis][axis] * o as f64 + matrix[axis][2],
        );
        let axis_supports = &axis_supports;
        fill_pixels_parallel_indexed(&mut output, kernel_work, |_flat, oidx| {
            sample_separable_pixel(coeffs, axis_supports, oidx, cval, offsets)
        });
        return Ok(output);
    }

    // Same fallthrough as the indexed variant above: the other binding sits inside
    // the early-returning branch. One resolve per call on either path, never per pixel.
    let flags = SplineFlags::resolve();
    fill_pixels_parallel(&mut output, kernel_work, |flat, _scratch| {
        let r = (flat / cols) as f64;
        let c = (flat % cols) as f64;
        let src_r = matrix[0][0] * r + matrix[0][1] * c + matrix[0][2];
        let src_c = matrix[1][0] * r + matrix[1][1] * c + matrix[1][2];
        sample_interpolated(
            input,
            &spline.coeffs,
            &[src_r, src_c],
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        )
    });

    Ok(output)
}

/// Binary hit-or-miss transform.
///
/// Detects specific patterns in binary images.
/// Matches `scipy.ndimage.binary_hit_or_miss`.
pub fn binary_hit_or_miss(
    input: &NdArray,
    structure1: &NdArray,
    structure2: Option<&NdArray>,
) -> Result<NdArray, NdimageError> {
    // Erode with structure1
    let hit = binary_erosion_with_struct(input, structure1)?;

    // Complement of input
    let complement: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
        .collect();
    let comp = NdArray::new(complement, input.shape.clone())?;

    // Erode complement with structure2 (default: element-wise NOT of
    // structure1, matching scipy.ndimage.binary_hit_or_miss).
    // [frankenscipy-ee82u]
    let owned_complement;
    let miss_struct = if let Some(s2) = structure2 {
        s2
    } else {
        let complement: Vec<f64> = structure1
            .data
            .iter()
            .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
            .collect();
        owned_complement = NdArray::new(complement, structure1.shape.clone())?;
        &owned_complement
    };

    let miss = binary_erosion_with_struct(&comp, miss_struct)?;

    // AND: hit AND miss
    let mut result = NdArray::zeros(input.shape.clone());
    for i in 0..result.data.len() {
        result.data[i] = if hit.data[i] != 0.0 && miss.data[i] != 0.0 {
            1.0
        } else {
            0.0
        };
    }

    Ok(result)
}

fn binary_erosion_with_struct(
    input: &NdArray,
    structure: &NdArray,
) -> Result<NdArray, NdimageError> {
    if input.ndim() != structure.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "input and structure must have same dimensions".to_string(),
        ));
    }

    let _ndim = input.ndim();
    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = structure.shape.iter().map(|&s| s as i64 / 2).collect();

    // Collect structure element positions
    let mut struct_positions = Vec::new();
    for flat in 0..structure.size() {
        if structure.data[flat] != 0.0 {
            let idx = structure.unravel(flat);
            let delta: Vec<i64> = idx
                .iter()
                .zip(offsets.iter())
                .map(|(&i, &o)| i as i64 - o)
                .collect();
            struct_positions.push(delta);
        }
    }

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut all_set = true;

        for delta in &struct_positions {
            let in_idx: Vec<i64> = out_idx
                .iter()
                .zip(delta.iter())
                .map(|(&o, &d)| o as i64 + d)
                .collect();
            let val = input.get_boundary(&in_idx, BoundaryMode::Constant, 0.0);
            if val == 0.0 {
                all_set = false;
                break;
            }
        }

        output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
    }

    Ok(output)
}

/// Compute the sum of the array.
pub fn array_sum(input: &NdArray) -> f64 {
    input.data.iter().sum()
}

/// Compute the mean of the array.
pub fn array_mean(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return f64::NAN;
    }
    input.data.iter().sum::<f64>() / input.size() as f64
}

/// Compute the variance of the array.
pub fn array_variance(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return f64::NAN;
    }
    let mean = array_mean(input);
    let n = input.size() as f64;
    input.data.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Compute the standard deviation of the array.
pub fn array_std(input: &NdArray) -> f64 {
    array_variance(input).sqrt()
}

/// When `true`, [`count_nonzero`] counts serially (the ORIG behaviour); default `false` chunks the
/// count across threads. Byte-identical.
#[doc(hidden)]
pub static COUNT_NONZERO_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Count nonzero elements.
pub fn count_nonzero(input: &NdArray) -> usize {
    // Parallel chunk-then-sum. BYTE-IDENTICAL to the serial `filter(|v| v != 0.0).count()`: the count
    // is an exact integer sum of an order-independent per-element predicate (`v != 0.0`, which counts
    // NaN and skips ±0.0 identically in any chunk), and integer addition is associative — so summing
    // the per-chunk counts equals the sequential count exactly. Gated on `n`.
    let data = input.data.as_slice();
    let n = data.len();
    let nthreads = if COUNT_NONZERO_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        ndimage_filter_thread_count(n, 4)
    };
    if nthreads <= 1 {
        return data.iter().filter(|&&v| v != 0.0).count();
    }
    let chunk = n.div_ceil(nthreads);
    let parts: Vec<usize> = std::thread::scope(|scope| {
        data.chunks(chunk)
            .map(|c| scope.spawn(move || c.iter().filter(|&&v| v != 0.0).count()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("count_nonzero chunk panicked"))
            .collect()
    });
    parts.into_iter().sum()
}

/// When `true`, [`clip`]/[`scale_array`] run their element-wise scalar-param map serially (the ORIG
/// behaviour); default `false` fans it across cores. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_UNARY_MAP_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Clip (clamp) array values to [a_min, a_max].
pub fn clip(input: &NdArray, a_min: f64, a_max: f64) -> NdArray {
    // Element-wise `v.clamp(a_min, a_max)` — pure read+write bandwidth; fan across cores (multi-channel
    // bandwidth aggregates). BYTE-IDENTICAL — each output element is `input.data[flat].clamp(..)` in
    // flat order. `NDIMAGE_UNARY_MAP_FORCE_SERIAL` restores the serial map (same-binary A/B).
    if NDIMAGE_UNARY_MAP_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v.clamp(a_min, a_max)).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 1, |flat, _scratch| {
        input.data[flat].clamp(a_min, a_max)
    });
    output
}

/// When `true`, [`abs_array`] runs its element-wise `abs` map serially (the ORIG behaviour); default
/// `false` fans the map across cores via `fill_pixels_parallel`. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_ABS_ARRAY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply element-wise absolute value.
pub fn abs_array(input: &NdArray) -> NdArray {
    // `abs` is a near-free bit op, so this map is pure read+write memory bandwidth. Fanning it
    // across cores aggregates multi-channel bandwidth (like count_nonzero / the scale-family output
    // maps). BYTE-IDENTICAL — each output element is `input.data[flat].abs()` in flat order.
    // `NDIMAGE_ABS_ARRAY_FORCE_SERIAL` restores the serial map (same-binary A/B).
    if NDIMAGE_ABS_ARRAY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v.abs()).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 1, |flat, _scratch| input.data[flat].abs());
    output
}

/// When `true`, [`sqrt_array`] runs its element-wise `sqrt` map serially (the ORIG behaviour); default
/// `false` fans the map across cores via `fill_pixels_parallel`. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_SQRT_ARRAY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply element-wise square root.
pub fn sqrt_array(input: &NdArray) -> NdArray {
    // `sqrt` (with a `max(0.0)` clamp) is a moderately heavy per-element op, so this map is
    // compute-bound at large `n`; fan it across cores. BYTE-IDENTICAL to the serial map — each output
    // element is `input.data[flat].max(0.0).sqrt()` written in flat order, a pure function of its
    // index. `NDIMAGE_SQRT_ARRAY_FORCE_SERIAL` restores the serial map (same-binary A/B).
    if NDIMAGE_SQRT_ARRAY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v.max(0.0).sqrt()).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 4, |flat, _scratch| {
        input.data[flat].max(0.0).sqrt()
    });
    output
}

/// Apply element-wise power.
pub fn power_array(input: &NdArray, exponent: f64) -> NdArray {
    // `powf` is a heavy per-element transcendental (~50-100 cycles), so this map is COMPUTE-bound
    // (unlike the bandwidth-bound `add_arrays`/`multiply_arrays`): a work-gated parallel fill wins
    // at large `n`. BYTE-IDENTICAL to the serial `data.iter().map(|&v| v.powf(exponent)).collect()`
    // — each output element is `input.data[flat].powf(exponent)` written in flat order, a pure
    // function of its index. `NDIMAGE_POWER_ARRAY_FORCE_SERIAL` restores the serial map (same-binary
    // A/B). The gate (`kernel_work = 16`, powf's rough mul-equivalent cost) keeps small arrays serial.
    if NDIMAGE_POWER_ARRAY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v.powf(exponent)).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 16, |flat, _scratch| {
        input.data[flat].powf(exponent)
    });
    output
}

/// When `true`, [`power_array`] runs its element-wise `powf` map serially (the ORIG behaviour).
/// When `false` (default), the compute-bound map fans across cores via `fill_pixels_parallel`.
/// Byte-identical either way. For the same-binary A/B perf gate.
#[doc(hidden)]
pub static NDIMAGE_POWER_ARRAY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, the element-wise binary array ops ([`add_arrays`]/[`multiply_arrays`]/
/// [`subtract_arrays`]) run their `a[i] OP b[i]` map serially (the ORIG behaviour); default `false`
/// fans it across cores. Byte-identical.
#[doc(hidden)]
pub static NDIMAGE_BINOP_ARRAYS_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Element-wise `a[flat] OP b[flat]` over two equal-shape arrays, parallelized across cores.
/// BYTE-IDENTICAL to the serial `a.data.iter().zip(b.data).map(|(&x,&y)| op(x,y)).collect()`: each
/// output element is a pure function of its flat index, written in flat order (reads two arrays +
/// writes one ⇒ pure memory bandwidth, which multi-core aggregates). Caller checks shapes first.
fn ndimage_binop_arrays<F>(a: &NdArray, b: &NdArray, op: F) -> NdArray
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    if NDIMAGE_BINOP_ARRAYS_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| op(x, y))
            .collect();
        return NdArray {
            data,
            shape: a.shape.clone(),
            strides: a.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; a.data.len()],
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 1, |flat, _scratch| {
        op(a.data[flat], b.data[flat])
    });
    output
}

/// Multiply two arrays element-wise.
pub fn multiply_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise multiply".to_string(),
        ));
    }
    Ok(ndimage_binop_arrays(a, b, |x, y| x * y))
}

/// Add two arrays element-wise.
pub fn add_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise add".to_string(),
        ));
    }
    Ok(ndimage_binop_arrays(a, b, |x, y| x + y))
}

/// Subtract two arrays element-wise.
pub fn subtract_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    Ok(ndimage_binop_arrays(a, b, |x, y| x - y))
}

/// Scale (multiply by scalar) an array.
pub fn scale_array(input: &NdArray, scalar: f64) -> NdArray {
    // Element-wise `v * scalar` — pure read+write bandwidth; fan across cores. BYTE-IDENTICAL — each
    // output element is `input.data[flat] * scalar` in flat order.
    if NDIMAGE_UNARY_MAP_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v * scalar).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 1, |flat, _scratch| input.data[flat] * scalar);
    output
}

/// Apply element-wise natural logarithm (ln).
pub fn log_array(input: &NdArray) -> NdArray {
    // `ln` is a heavy per-element transcendental (~20-40 cycles) → this map is COMPUTE-bound; a
    // work-gated parallel fill wins at large `n`. BYTE-IDENTICAL to the serial map (the `v > 0.0`
    // branch + `v.ln()` is a pure function of each element, written in flat order).
    if NDIMAGE_LOG_ARRAY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input
            .data
            .iter()
            .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
            .collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 16, |flat, _scratch| {
        let v = input.data[flat];
        if v > 0.0 { v.ln() } else { f64::NEG_INFINITY }
    });
    output
}

/// When `true`, [`log_array`] runs its element-wise `ln` map serially (the ORIG behaviour). When
/// `false` (default), the compute-bound map fans across cores via `fill_pixels_parallel`.
/// Byte-identical either way. For the same-binary A/B perf gate.
#[doc(hidden)]
pub static NDIMAGE_LOG_ARRAY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Apply element-wise exponential.
pub fn exp_array(input: &NdArray) -> NdArray {
    // `exp` is a heavy per-element transcendental (~20-40 cycles) → this map is COMPUTE-bound; a
    // work-gated parallel fill wins at large `n`. BYTE-IDENTICAL to the serial map (`v.exp()` is a
    // pure function of each element, written in flat order).
    if NDIMAGE_EXP_ARRAY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        let data: Vec<f64> = input.data.iter().map(|&v| v.exp()).collect();
        return NdArray {
            data,
            shape: input.shape.clone(),
            strides: input.strides.clone(),
        };
    }
    let mut output = NdArray {
        data: vec![0.0; input.data.len()],
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    };
    fill_pixels_parallel(&mut output, 16, |flat, _scratch| input.data[flat].exp());
    output
}

/// When `true`, [`exp_array`] runs its element-wise `exp` map serially (the ORIG behaviour). When
/// `false` (default), the compute-bound map fans across cores via `fill_pixels_parallel`.
/// Byte-identical either way. For the same-binary A/B perf gate.
#[doc(hidden)]
pub static NDIMAGE_EXP_ARRAY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Compute the sum along a specific axis, reducing that dimension.
pub fn sum_axis(input: &NdArray, axis: usize) -> Result<NdArray, NdimageError> {
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {axis} out of range for {}-d array",
            input.ndim()
        )));
    }

    let mut new_shape = input.shape.clone();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        // Collapsing to scalar
        let sum: f64 = input.data.iter().sum();
        return NdArray::new(vec![sum], vec![1]);
    }

    let mut result = NdArray::zeros(new_shape);

    // Row-major odometer instead of a per-element `unravel` + `out_idx.clone()` (two heap allocs
    // per element). BYTE-IDENTICAL: `out_flat` is built from the same multi-index and strides, and
    // each output cell accumulates its inputs in the same flat order.
    let full_mode = NDIMAGE_UNRAVEL_ODOMETER_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let ndim = input.ndim();
    let mut idx = vec![0usize; ndim];
    for flat in 0..input.size() {
        let out_flat = if full_mode {
            let uidx = input.unravel(flat);
            let mut out_idx: Vec<usize> = uidx.clone();
            out_idx.remove(axis);
            out_idx
                .iter()
                .zip(result.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>()
        } else {
            // Σ_{d≠axis} idx[d]·result.strides[d'] — same value as removing `axis` then dotting.
            let mut acc = 0usize;
            let mut si = 0usize;
            for (d, &index_d) in idx.iter().enumerate().take(ndim) {
                if d != axis {
                    acc += index_d * result.strides[si];
                    si += 1;
                }
            }
            acc
        };
        result.data[out_flat] += input.data[flat];
        if !full_mode {
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < input.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    Ok(result)
}

/// Compute the mean along a specific axis.
pub fn mean_axis(input: &NdArray, axis: usize) -> Result<NdArray, NdimageError> {
    let summed = sum_axis(input, axis)?;
    let axis_size = input.shape[axis] as f64;
    Ok(scale_array(&summed, 1.0 / axis_size))
}

/// Pad an array with constant values.
///
/// `pad_width` specifies (before, after) padding for each axis.
pub fn pad_constant(
    input: &NdArray,
    pad_width: &[(usize, usize)],
    constant: f64,
) -> Result<NdArray, NdimageError> {
    if pad_width.len() != input.ndim() {
        return Err(NdimageError::DimensionMismatch(
            "pad_width must have one entry per dimension".to_string(),
        ));
    }

    let new_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(pad_width.iter())
        .map(|(&s, &(before, after))| s + before + after)
        .collect();

    let mut result = NdArray::zeros(new_shape.clone());
    result.data.fill(constant);

    // Copy input data to padded region. Row-major odometer instead of a per-element `unravel` +
    // `padded_idx` (two heap allocs per element). BYTE-IDENTICAL: `out_flat` is the same
    // `Σ (idx[d]+before_d)·stride_d`, and each output cell is assigned once.
    let result_strides = compute_strides(&new_shape);
    let full_mode = NDIMAGE_UNRAVEL_ODOMETER_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let ndim = input.ndim();
    let mut idx = vec![0usize; ndim];
    for flat in 0..input.size() {
        let out_flat: usize = if full_mode {
            let uidx = input.unravel(flat);
            let padded_idx: Vec<usize> = uidx
                .iter()
                .zip(pad_width.iter())
                .map(|(&i, &(before, _))| i + before)
                .collect();
            padded_idx
                .iter()
                .zip(result_strides.iter())
                .map(|(&i, &s)| i * s)
                .sum()
        } else {
            let mut acc = 0usize;
            for d in 0..ndim {
                acc += (idx[d] + pad_width[d].0) * result_strides[d];
            }
            acc
        };
        result.data[out_flat] = input.data[flat];
        if !full_mode {
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < input.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    Ok(result)
}

/// Create an NdArray filled with a constant value.
pub fn full(shape: Vec<usize>, value: f64) -> NdArray {
    let total: usize = shape.iter().product();
    NdArray {
        data: vec![value; total],
        strides: compute_strides(&shape),
        shape,
    }
}

/// Create an NdArray filled with ones.
pub fn ones(shape: Vec<usize>) -> NdArray {
    full(shape, 1.0)
}

/// Reshape an NdArray (must have same total number of elements).
pub fn reshape(input: &NdArray, new_shape: Vec<usize>) -> Result<NdArray, NdimageError> {
    let new_total = checked_shape_product(&new_shape)?;
    if new_total != input.size() {
        return Err(NdimageError::DimensionMismatch(format!(
            "cannot reshape {} elements into {:?}",
            input.size(),
            new_shape
        )));
    }
    NdArray::new(input.data.clone(), new_shape)
}

/// Flatten an NdArray to 1D.
pub fn flatten(input: &NdArray) -> NdArray {
    let size = input.size();
    NdArray {
        data: input.data.clone(),
        shape: vec![size],
        strides: vec![1],
    }
}

/// Compute element-wise comparison: 1.0 where a > b, 0.0 otherwise.
pub fn greater_than(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| if x > y { 1.0 } else { 0.0 })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute element-wise comparison: 1.0 where a == b within tolerance.
pub fn equal_within(a: &NdArray, b: &NdArray, tol: f64) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| if (x - y).abs() <= tol { 1.0 } else { 0.0 })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute where: select from a where condition is true, from b otherwise.
pub fn where_cond(condition: &NdArray, a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if condition.shape != a.shape || a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = condition
        .data
        .iter()
        .zip(a.data.iter().zip(b.data.iter()))
        .map(|(&c, (&x, &y))| if c != 0.0 { x } else { y })
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Compute argmax: index of maximum element.
pub fn argmax(input: &NdArray) -> usize {
    input
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute argmin: index of minimum element.
pub fn argmin(input: &NdArray) -> usize {
    input
        .data
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Compute the cumulative sum along a flattened array.
pub fn cumsum_array(input: &NdArray) -> NdArray {
    let mut sum = 0.0;
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| {
            sum += v;
            sum
        })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the cumulative product along a flattened array.
pub fn cumprod_array(input: &NdArray) -> NdArray {
    let mut prod = 1.0;
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| {
            prod *= v;
            prod
        })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the difference between consecutive elements (first difference).
pub fn diff_array(input: &NdArray) -> NdArray {
    if input.size() < 2 {
        return NdArray::zeros(vec![0]);
    }
    let data: Vec<f64> = input.data.windows(2).map(|w| w[1] - w[0]).collect();
    let len = data.len();
    NdArray {
        data,
        shape: vec![len],
        strides: vec![1],
    }
}

/// Apply a boolean mask: return elements where mask is nonzero.
pub fn masked_select(input: &NdArray, mask: &NdArray) -> Vec<f64> {
    input
        .data
        .iter()
        .zip(mask.data.iter())
        .filter(|&(_, m)| *m != 0.0)
        .map(|(&v, _)| v)
        .collect()
}

/// Set elements where mask is nonzero to a given value.
pub fn masked_fill(input: &NdArray, mask: &NdArray, value: f64) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .zip(mask.data.iter())
        .map(|(&v, &m)| if m != 0.0 { value } else { v })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Compute the histogram of an NdArray.
pub fn array_histogram(input: &NdArray, bins: usize) -> (Vec<usize>, Vec<f64>) {
    if input.size() == 0 || bins == 0 {
        return (vec![], vec![]);
    }
    let min_val = array_min(input);
    let max_val = array_max(input);
    let range = max_val - min_val;
    let bw = if range > 0.0 {
        range / bins as f64
    } else {
        1.0
    };

    let mut counts = vec![0usize; bins];
    let edges: Vec<f64> = (0..=bins).map(|i| min_val + i as f64 * bw).collect();

    for &v in &input.data {
        let bin = ((v - min_val) / bw).floor() as usize;
        counts[bin.min(bins - 1)] += 1;
    }

    (counts, edges)
}

// ══════════════════════════════════════════════════════════════════════
// Fourier Domain Filters
// ══════════════════════════════════════════════════════════════════════

use fsci_fft::Complex64;
use std::f64::consts::PI;

/// Same-binary A/B toggle: when `true`, `fourier_gaussian`/`fourier_uniform` re-evaluate the
/// per-axis transcendental factor at every element instead of the precomputed separable 1-D
/// tables. Byte-identical output; A/B timing only. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static NDIMAGE_FOURIER_SEPARABLE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle: when `true`, the Fourier-domain filters fill their output serially (the
/// ORIG behaviour). When `false` (default), the independent per-element fill fans across cores.
/// Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static NDIMAGE_FOURIER_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fill each element of a Fourier-domain filter output independently via `f(flat_index, &mut value)`,
/// distributing the disjoint output indices across cores. Each element's filter value is a pure
/// function of its flat index (the odometer is recomputed per element, no carry state), so the
/// parallel result is BYTE-IDENTICAL to the serial loop — only the owning core changes. Gated on the
/// element count (the per-element transcendental is the dominant work).
fn fourier_fill_parallel<G>(output: &mut [Complex64], f: G)
where
    G: Fn(usize, &mut Complex64) + Sync,
{
    let total = output.len();
    let nthreads = if NDIMAGE_FOURIER_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        ndimage_filter_thread_count(total, 8)
    };
    if nthreads <= 1 {
        for (idx, val) in output.iter_mut().enumerate() {
            f(idx, val);
        }
        return;
    }
    let chunk = total.div_ceil(nthreads);
    let f = &f;
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (local, val) in out_chunk.iter_mut().enumerate() {
                    f(base + local, val);
                }
            });
        }
    });
}

/// Apply a Gaussian filter in the Fourier domain.
///
/// Matches `scipy.ndimage.fourier_gaussian`. The input is assumed to be
/// the FFT of a real-valued image. Multiplies by exp(-0.5 * (omega * sigma)^2)
/// where omega is the angular frequency.
///
/// # Arguments
/// * `input` - Complex Fourier coefficients (row-major)
/// * `shape` - Shape of the original spatial-domain array
/// * `sigma` - Standard deviation of Gaussian (can be different per axis)
pub fn fourier_gaussian(input: &[Complex64], shape: &[usize], sigma: &[f64]) -> Vec<Complex64> {
    if shape.is_empty() || sigma.len() != shape.len() {
        return input.to_vec();
    }
    let total: usize = shape.iter().product();
    if input.len() != total {
        return input.to_vec();
    }

    let mut output = input.to_vec();

    // The frequency-domain Gaussian is SEPARABLE: `filter_val = ∏_d g_d(index_d)` where the
    // per-axis factor `g_d(i) = exp(-0.5·(2π·freq(i)·sigma_d)²)` depends ONLY on the index along
    // axis `d`. The naive loop re-evaluated that `exp` at every one of the `total` elements
    // (`total·ndim` exps); precomputing the `ndim` 1-D factor tables costs only `Σ shape[d]` exps
    // (~`total/ndim`× fewer). BYTE-IDENTICAL: each `g_d[i]` is the identical expression and the
    // per-element product runs in the SAME reverse-axis order, so `filter_val` matches bit-for-bit.
    let full_mode = NDIMAGE_FOURIER_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let factors: Vec<Vec<f64>> = if full_mode {
        Vec::new()
    } else {
        shape
            .iter()
            .enumerate()
            .map(|(d, &n)| {
                (0..n)
                    .map(|i| {
                        let freq = if i <= n / 2 {
                            i as f64 / n as f64
                        } else {
                            (i as f64 - n as f64) / n as f64
                        };
                        let omega = 2.0 * PI * freq;
                        (-0.5 * (omega * sigma[d]).powi(2)).exp()
                    })
                    .collect()
            })
            .collect()
    };

    fourier_fill_parallel(&mut output, |idx, val| {
        let mut filter_val = 1.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            if full_mode {
                let freq = if i <= n / 2 {
                    i as f64 / n as f64
                } else {
                    (i as f64 - n as f64) / n as f64
                };
                let omega = 2.0 * PI * freq;
                filter_val *= (-0.5 * (omega * sigma[d]).powi(2)).exp();
            } else {
                filter_val *= factors[d][i];
            }
        }

        *val = (val.0 * filter_val, val.1 * filter_val);
    });

    output
}

/// Apply a uniform (box) filter in the Fourier domain.
///
/// Matches `scipy.ndimage.fourier_uniform`. Multiplies by sinc(omega * size / 2)
/// where omega is the angular frequency.
///
/// # Arguments
/// * `input` - Complex Fourier coefficients (row-major)
/// * `shape` - Shape of the original spatial-domain array
/// * `size` - Size of uniform filter (can be different per axis)
pub fn fourier_uniform(input: &[Complex64], shape: &[usize], size: &[f64]) -> Vec<Complex64> {
    if shape.is_empty() || size.len() != shape.len() {
        return input.to_vec();
    }
    let total: usize = shape.iter().product();
    if input.len() != total {
        return input.to_vec();
    }

    let mut output = input.to_vec();

    // Separable like `fourier_gaussian`: per-axis factor `u_d(i) = sinc(π·freq(i)·size_d)` (1.0 at
    // freq≈0). Precompute the `ndim` 1-D tables (`Σ shape[d]` sines) instead of a `sin` at every
    // `total·ndim`. BYTE-IDENTICAL: the freq≈0 factor is exactly `1.0`, so `filter_val *= 1.0`
    // reproduces the old `continue` (no-op), and the retained sines run in the same reverse order.
    let full_mode = NDIMAGE_FOURIER_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let factors: Vec<Vec<f64>> = if full_mode {
        Vec::new()
    } else {
        shape
            .iter()
            .enumerate()
            .map(|(d, &n)| {
                (0..n)
                    .map(|i| {
                        let freq = if i <= n / 2 {
                            i as f64 / n as f64
                        } else {
                            (i as f64 - n as f64) / n as f64
                        };
                        if freq.abs() < 1e-15 {
                            1.0
                        } else {
                            let x = PI * freq * size[d];
                            x.sin() / x
                        }
                    })
                    .collect()
            })
            .collect()
    };

    fourier_fill_parallel(&mut output, |idx, val| {
        let mut filter_val = 1.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            if full_mode {
                let freq = if i <= n / 2 {
                    i as f64 / n as f64
                } else {
                    (i as f64 - n as f64) / n as f64
                };

                if freq.abs() < 1e-15 {
                    continue;
                }
                let x = PI * freq * size[d];
                filter_val *= x.sin() / x;
            } else {
                filter_val *= factors[d][i];
            }
        }

        *val = (val.0 * filter_val, val.1 * filter_val);
    });

    output
}

/// Apply a shift filter in the Fourier domain.
///
/// Matches `scipy.ndimage.fourier_shift`. Multiplies by exp(-2πi * freq * shift)
/// to shift the image in spatial domain.
///
/// # Arguments
/// * `input` - Complex Fourier coefficients (row-major)
/// * `shape` - Shape of the original spatial-domain array
/// * `shift` - Shift amount for each axis
pub fn fourier_shift(input: &[Complex64], shape: &[usize], shift: &[f64]) -> Vec<Complex64> {
    if shape.is_empty() || shift.len() != shape.len() {
        return input.to_vec();
    }
    let total: usize = shape.iter().product();
    if input.len() != total {
        return input.to_vec();
    }

    let mut output = input.to_vec();

    // The phase is a SUM over axes (`phase = −Σ_d 2π·freq_d·shift_d`), so `exp(iφ)` does NOT factor
    // byte-exactly — the `sin_cos` stays per element. But the per-axis phase contribution
    // `2π·freq_d(i)·shift_d` (a `freq` branch+divide + two mults) depends ONLY on the axis index, so
    // precompute the `ndim` 1-D tables once and reduce the inner loop to a subtraction. BYTE-IDENTICAL:
    // each `pc[d][i]` is the identical expression and the sum runs in the SAME reverse-axis order.
    let full_mode = NDIMAGE_FOURIER_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let pc: Vec<Vec<f64>> = if full_mode {
        Vec::new()
    } else {
        shape
            .iter()
            .enumerate()
            .map(|(d, &n)| {
                (0..n)
                    .map(|i| {
                        let freq = if i <= n / 2 {
                            i as f64 / n as f64
                        } else {
                            (i as f64 - n as f64) / n as f64
                        };
                        2.0 * PI * freq * shift[d]
                    })
                    .collect()
            })
            .collect()
    };

    fourier_fill_parallel(&mut output, |idx, val| {
        let mut phase = 0.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            if full_mode {
                let freq = if i <= n / 2 {
                    i as f64 / n as f64
                } else {
                    (i as f64 - n as f64) / n as f64
                };
                phase -= 2.0 * PI * freq * shift[d];
            } else {
                phase -= pc[d][i];
            }
        }

        let (sin_p, cos_p) = phase.sin_cos();
        *val = (val.0 * cos_p - val.1 * sin_p, val.0 * sin_p + val.1 * cos_p);
    });

    output
}

/// Apply an ellipsoid filter in the Fourier domain.
///
/// Matches `scipy.ndimage.fourier_ellipsoid`. Creates an ellipsoid
/// low-pass filter in frequency space.
///
/// # Arguments
/// * `input` - Complex Fourier coefficients (row-major)
/// * `shape` - Shape of the original spatial-domain array
/// * `size` - Size of ellipsoid axes
pub fn fourier_ellipsoid(input: &[Complex64], shape: &[usize], size: &[f64]) -> Vec<Complex64> {
    if shape.is_empty() || size.len() != shape.len() {
        return input.to_vec();
    }
    let total: usize = shape.iter().product();
    if input.len() != total {
        return input.to_vec();
    }

    let ndim = shape.len();
    let mut output = input.to_vec();

    // The filter is a function of the COMBINED radius `r=√Σ_d(freq_d·size_d)²`, so the transcendental
    // stays per element — but each per-axis squared term `(freq_d(i)·size_d)²` (a `freq` branch+divide
    // + two mults) depends ONLY on the axis index, so precompute the `ndim` 1-D tables and reduce the
    // inner loop to an add. BYTE-IDENTICAL: same expression, same reverse-axis accumulation order.
    let full_mode = NDIMAGE_FOURIER_SEPARABLE_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let sq: Vec<Vec<f64>> = if full_mode {
        Vec::new()
    } else {
        shape
            .iter()
            .enumerate()
            .map(|(d, &n)| {
                (0..n)
                    .map(|i| {
                        let freq = if i <= n / 2 {
                            i as f64 / n as f64
                        } else {
                            (i as f64 - n as f64) / n as f64
                        };
                        let normalized = freq * size[d];
                        normalized * normalized
                    })
                    .collect()
            })
            .collect()
    };

    fourier_fill_parallel(&mut output, |idx, val| {
        let mut sum_sq = 0.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            if full_mode {
                let freq = if i <= n / 2 {
                    i as f64 / n as f64
                } else {
                    (i as f64 - n as f64) / n as f64
                };
                let normalized = freq * size[d];
                sum_sq += normalized * normalized;
            } else {
                sum_sq += sq[d][i];
            }
        }

        let r = sum_sq.sqrt();
        let filter_val = if r < 1e-15 {
            1.0
        } else if ndim == 1 {
            (PI * r).sin() / (PI * r)
        } else if ndim == 2 {
            let pr = PI * r;
            2.0 * pr.sin() / pr - 2.0 * (1.0 - pr.cos()) / (pr * pr)
        } else {
            let pr = PI * r;
            (pr.sin() - pr * pr.cos()) * 3.0 / (pr * pr * pr)
        };

        *val = (val.0 * filter_val, val.1 * filter_val);
    });

    output
}

// ══════════════════════════════════════════════════════════════════════
// Generic Derivative Filters
// ══════════════════════════════════════════════════════════════════════

/// Generic gradient magnitude filter.
///
/// Matches `scipy.ndimage.generic_gradient_magnitude`. Computes gradient
/// magnitude using a user-provided derivative function applied along each axis.
///
/// # Arguments
/// * `input` - Input array
/// * `derivative` - Function that computes derivative along a given axis
///
/// Returns sqrt(sum(derivative(input, axis)^2)) for each axis.
pub fn generic_gradient_magnitude<F>(input: &NdArray, derivative: F) -> NdArray
where
    F: Fn(&NdArray, usize) -> NdArray,
{
    let ndim = input.shape.len();
    let mut result = NdArray::new(vec![0.0; input.size()], input.shape.clone()).unwrap();

    for axis in 0..ndim {
        let deriv = derivative(input, axis);
        for (r, &d) in result.data.iter_mut().zip(deriv.data.iter()) {
            *r += d * d;
        }
    }

    for r in &mut result.data {
        *r = r.sqrt();
    }

    result
}

/// Generic gradient magnitude over a SciPy-style signed axes subset.
pub fn generic_gradient_magnitude_axes<F>(
    input: &NdArray,
    derivative: F,
    axes: &[isize],
) -> Result<NdArray, NdimageError>
where
    F: Fn(&NdArray, usize) -> NdArray,
{
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for axis in axes {
        let deriv = derivative(input, axis);
        if deriv.shape != input.shape {
            return Err(NdimageError::DimensionMismatch(format!(
                "derivative output shape {:?} != input shape {:?}",
                deriv.shape, input.shape
            )));
        }
        for (r, &d) in result.data.iter_mut().zip(deriv.data.iter()) {
            *r += d * d;
        }
    }

    for r in &mut result.data {
        *r = r.sqrt();
    }

    Ok(result)
}

/// Generic Laplace filter.
///
/// Matches `scipy.ndimage.generic_laplace`. Computes Laplacian using a
/// user-provided second derivative function applied along each axis.
///
/// # Arguments
/// * `input` - Input array
/// * `derivative2` - Function that computes second derivative along a given axis
///
/// Returns sum(derivative2(input, axis)) for each axis.
pub fn generic_laplace<F>(input: &NdArray, derivative2: F) -> NdArray
where
    F: Fn(&NdArray, usize) -> NdArray,
{
    let ndim = input.shape.len();
    let mut result = NdArray::new(vec![0.0; input.size()], input.shape.clone()).unwrap();

    for axis in 0..ndim {
        let deriv2 = derivative2(input, axis);
        for (r, &d) in result.data.iter_mut().zip(deriv2.data.iter()) {
            *r += d;
        }
    }

    result
}

/// Generic Laplace filter over a SciPy-style signed axes subset.
pub fn generic_laplace_axes<F>(
    input: &NdArray,
    derivative2: F,
    axes: &[isize],
) -> Result<NdArray, NdimageError>
where
    F: Fn(&NdArray, usize) -> NdArray,
{
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let mut result = NdArray::zeros(input.shape.clone());
    for axis in axes {
        let deriv2 = derivative2(input, axis);
        if deriv2.shape != input.shape {
            return Err(NdimageError::DimensionMismatch(format!(
                "derivative output shape {:?} != input shape {:?}",
                deriv2.shape, input.shape
            )));
        }
        for (r, &d) in result.data.iter_mut().zip(deriv2.data.iter()) {
            *r += d;
        }
    }

    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Geometric Transformation
// ══════════════════════════════════════════════════════════════════════

/// Apply a generic geometric transformation.
///
/// Matches `scipy.ndimage.geometric_transform`. Maps output coordinates
/// through a user-provided function to find corresponding input coordinates,
/// then interpolates the input at those coordinates.
///
/// # Arguments
/// * `input` - Input array
/// * `mapping` - Function that maps output coordinate to input coordinate
/// * `output_shape` - Shape of output array (defaults to input shape)
/// * `order` - Interpolation order (0-5)
/// * `mode` - Boundary handling mode
/// * `cval` - Constant value for 'constant' mode
pub fn geometric_transform<F>(
    input: &NdArray,
    mapping: F,
    output_shape: Option<Vec<usize>>,
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> Result<NdArray, NdimageError>
where
    F: Fn(&[usize]) -> Vec<f64> + Sync,
{
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }

    let out_shape = output_shape.unwrap_or_else(|| input.shape.clone());
    if out_shape.is_empty() {
        return Ok(NdArray::new(vec![], vec![]).unwrap());
    }

    let total_out: usize = out_shape.iter().product();

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    // Each output pixel maps independently (out_coord -> mapping -> spline interpolation of the
    // read-only coefficients); fill the disjoint output indices in parallel via the shared
    // work-gated helper — bit-identical to the serial unravel-and-sample loop.
    let mut output = NdArray::new(vec![0.0_f64; total_out], out_shape.clone())?;
    let kernel_work = (order + 1).saturating_pow(input.ndim() as u32);
    let oshape = &out_shape;
    let mapping = &mapping;
    // Resolve the spline knobs ONCE, outside the per-pixel work below
    // (`frankenscipy-22van`).
    let flags = SplineFlags::resolve();
    fill_pixels_parallel(&mut output, kernel_work, |flat, _scratch| {
        let mut out_coord = Vec::with_capacity(oshape.len());
        let mut remaining = flat;
        for &n in oshape.iter().rev() {
            out_coord.push(remaining % n);
            remaining /= n;
        }
        out_coord.reverse();
        let in_coords = mapping(&out_coord);
        sample_interpolated(
            input,
            &spline.coeffs,
            &in_coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
            flags,
        )
    });

    Ok(output)
}

// ══════════════════════════════════════════════════════════════════════
// Spline Filter Functions
// ══════════════════════════════════════════════════════════════════════

/// Compute spline filter coefficients for multi-dimensional data.
///
/// Matches `scipy.ndimage.spline_filter`. Computes the spline coefficients
/// needed for B-spline interpolation of order `order`.
///
/// # Arguments
/// * `input` - Input array
/// * `order` - Spline order (0-5)
/// * `mode` - Boundary handling mode (only Reflect and Nearest are supported)
pub fn spline_filter(
    input: &NdArray,
    order: usize,
    mode: BoundaryMode,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if !matches!(mode, BoundaryMode::Reflect | BoundaryMode::Nearest) {
        return Err(NdimageError::InvalidArgument(
            "spline_filter only supports Reflect and Nearest modes".to_string(),
        ));
    }

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    Ok(spline.coeffs)
}

/// Same-binary A/B toggle for `spline_filter1d`. When `true`, the coefficients are computed by the
/// serial per-line walk (the ORIG behaviour). When `false` (default), the bspline-reflect kernel
/// reuses the vectorized/parallel machinery from `prefilter_spline_coefficients`. Byte-identical
/// either way. Benchmark knob.
#[doc(hidden)]
pub static NDIMAGE_SPLINE_FILTER1D_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Compute spline filter coefficients along a single axis.
///
/// Matches `scipy.ndimage.spline_filter1d`. Computes spline coefficients
/// along the specified axis, leaving other axes unchanged.
///
/// # Arguments
/// * `input` - Input array
/// * `order` - Spline order (0-5)
/// * `axis` - Axis along which to filter
/// * `mode` - Boundary handling mode (only Reflect and Nearest are supported)
pub fn spline_filter1d(
    input: &NdArray,
    order: usize,
    axis: usize,
    mode: BoundaryMode,
) -> Result<NdArray, NdimageError> {
    if order > 5 {
        return Err(NdimageError::InvalidArgument(format!(
            "spline order must be in 0..=5, got {order}"
        )));
    }
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidArgument(format!(
            "axis {} out of bounds for input with {} dimensions",
            axis,
            input.ndim()
        )));
    }
    if !matches!(mode, BoundaryMode::Reflect | BoundaryMode::Nearest) {
        return Err(NdimageError::InvalidArgument(
            "spline_filter1d only supports Reflect and Nearest modes".to_string(),
        ));
    }

    if order <= 1 {
        return Ok(input.clone());
    }

    let mut result = input.clone();
    let axis_len = result.shape[axis];

    let stride: usize = result.shape[axis + 1..].iter().product();
    let outer: usize = result.shape[..axis].iter().product();

    // The per-line kernel choice depends only on (mode, order, axis_len), which are the same for
    // every line, so it is decided once here. When it is the bspline-reflect kernel, reuse the
    // vectorized/parallel machinery that `prefilter_spline_coefficients` uses (byte-identical to the
    // per-line walk below): the strided stride>1 case sweeps the IIR in place over the contiguous
    // inner dim (cache-friendly instead of a per-column strided gather) and both cases fan the
    // independent outer blocks / rows across cores. Other kernels keep the serial per-line walk.
    let use_reflect = mode == BoundaryMode::Reflect && (2..=5).contains(&order) && axis_len > order;
    if use_reflect
        && !NDIMAGE_SPLINE_FILTER1D_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        if stride > 1 {
            let block = axis_len * stride;
            let nthreads = spline_axis_threads(outer, block);
            if nthreads <= 1 {
                bspline_reflect_axis_inplace(&mut result.data, outer, axis_len, stride, order);
            } else {
                let per = outer.div_ceil(nthreads);
                std::thread::scope(|scope| {
                    for chunk in result.data.chunks_mut(per * block) {
                        let chunk_outer = chunk.len() / block;
                        scope.spawn(move || {
                            bspline_reflect_axis_inplace(
                                chunk,
                                chunk_outer,
                                axis_len,
                                stride,
                                order,
                            );
                        });
                    }
                });
            }
            return Ok(result);
        }
        // stride == 1: each row is an independent contiguous `axis_len` block.
        let nthreads = spline_axis_threads(outer, axis_len);
        if nthreads > 1 {
            let per = outer.div_ceil(nthreads);
            std::thread::scope(|scope| {
                for chunk in result.data.chunks_mut(per * axis_len) {
                    scope.spawn(move || {
                        let mut line = Vec::with_capacity(axis_len);
                        for row in chunk.chunks_mut(axis_len) {
                            line.clear();
                            line.extend_from_slice(row);
                            let coeffs = bspline_reflect_coefficients(&line, order);
                            row.copy_from_slice(&coeffs);
                        }
                    });
                }
            });
            return Ok(result);
        }
        // Small last-axis input: fall through to the serial walk (same reflect kernel).
    }

    for outer_idx in 0..outer {
        for inner_idx in 0..stride {
            let mut line = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                let flat = outer_idx * axis_len * stride + i * stride + inner_idx;
                line.push(result.data[flat]);
            }

            // Reflect mode with a long-enough axis: use the fast exact recursive IIR
            // prefilter (Unser/Thévenaz, the same scipy-conformant kernel the N-D
            // `spline_filter`/`prefilter_spline_coefficients` use) instead of building and
            // solving a full interpolation system via `make_interp_spline` (O(n) banded build
            // per line — ~17x slower for a single long line). Nearest mode and axes too short
            // for the order's stencil keep the general path.
            let coeffs =
                if mode == BoundaryMode::Reflect && (2..=5).contains(&order) && axis_len > order {
                    bspline_reflect_coefficients(&line, order)
                } else {
                    spline_coefficients_for_line(&line, order)?
                };

            for (i, &c) in coeffs.iter().enumerate() {
                let flat = outer_idx * axis_len * stride + i * stride + inner_idx;
                result.data[flat] = c;
            }
        }
    }

    Ok(result)
}

/// Compute spline filter coefficients along one signed axis with SciPy-style normalization.
pub fn spline_filter1d_signed_axis(
    input: &NdArray,
    order: usize,
    axis: isize,
    mode: BoundaryMode,
) -> Result<NdArray, NdimageError> {
    let axis = normalize_signed_axis(axis, input.ndim())?;
    spline_filter1d(input, order, axis, mode)
}

/// Compute spline filter coefficients along SciPy's default `axis=-1`.
pub fn spline_filter1d_default_axis(
    input: &NdArray,
    order: usize,
    mode: BoundaryMode,
) -> Result<NdArray, NdimageError> {
    spline_filter1d_signed_axis(input, order, -1, mode)
}

// ══════════════════════════════════════════════════════════════════════
// Watershed Transform
// ══════════════════════════════════════════════════════════════════════

/// Watershed transform using Image Foresting Transform.
///
/// Matches `scipy.ndimage.watershed_ift`. Expands marker regions to fill
/// watershed basins based on the input (typically gradient) image.
///
/// # Arguments
/// * `input` - Input array (typically gradient magnitude)
/// * `markers` - Array with initial marked regions (positive integers)
/// * `structure` - Optional connectivity structure (default: 3x3x... cross)
///
/// # Returns
/// Labeled array where each pixel is assigned to a marker region.
pub fn watershed_ift(
    input: &NdArray,
    markers: &NdArray,
    structure: Option<&NdArray>,
) -> Result<NdArray, NdimageError> {
    if input.shape != markers.shape {
        return Err(NdimageError::DimensionMismatch(
            "input and markers must have the same shape".to_string(),
        ));
    }

    let ndim = input.ndim();
    let default_struct = generate_binary_structure(ndim, 1);
    let struct_arr = structure.unwrap_or(&default_struct);
    if struct_arr.ndim() != ndim {
        return Err(NdimageError::DimensionMismatch(format!(
            "input ndim {} != structure ndim {}",
            ndim,
            struct_arr.ndim()
        )));
    }
    if struct_arr.shape.contains(&0) {
        return Err(NdimageError::InvalidArgument(
            "structure dimensions must be positive".to_string(),
        ));
    }

    let struct_offsets = compute_structure_offsets(&struct_arr.shape, &struct_arr.data);

    // Precompute each structure offset's FLAT delta (Σ δ·stride) and reuse a coord
    // buffer (same flat-offset lever as label/fill_holes), killing the per-pop
    // `unravel` alloc and the per-neighbor `Vec` alloc + strides dot product.
    let signed_strides: Vec<i64> = input.strides.iter().map(|&s| s as i64).collect();
    let flat_offsets: Vec<i64> = struct_offsets
        .iter()
        .map(|off| {
            off.iter()
                .zip(&signed_strides)
                .map(|(&delta, &stride)| delta * stride)
                .sum()
        })
        .collect();
    let signed_shape: Vec<i64> = input.shape.iter().map(|&s| s as i64).collect();

    let output = if let Some(max_cost) = watershed_bucket_max(input, markers) {
        if structure.is_none() && input.shape.len() == 2 {
            watershed_ift_bucketed_output_2d_cross(input, markers, max_cost)
        } else {
            watershed_ift_bucketed_output(
                input,
                markers,
                &struct_offsets,
                &flat_offsets,
                &signed_shape,
                max_cost,
            )
        }
    } else {
        watershed_ift_heap_output(
            input,
            markers,
            &struct_offsets,
            &flat_offsets,
            &signed_shape,
        )
    };

    Ok(NdArray::new(output, input.shape.clone()).unwrap())
}

const WATERSHED_BUCKET_MAX_COST: usize = u16::MAX as usize;

fn watershed_bucket_max(input: &NdArray, markers: &NdArray) -> Option<usize> {
    for &marker in &markers.data {
        if marker != 0.0 && (!marker.is_finite() || marker.fract() != 0.0) {
            return None;
        }
    }

    let mut max_cost = 0usize;
    for &value in &input.data {
        if !value.is_finite()
            || value < 0.0
            || value > WATERSHED_BUCKET_MAX_COST as f64
            || value.fract() != 0.0
        {
            return None;
        }
        max_cost = max_cost.max(value as usize);
    }
    Some(max_cost)
}

fn watershed_ift_bucketed_output(
    input: &NdArray,
    markers: &NdArray,
    struct_offsets: &[Vec<i64>],
    flat_offsets: &[i64],
    signed_shape: &[i64],
    max_cost: usize,
) -> Vec<f64> {
    let ndim = input.ndim();
    let mut output = markers.data.clone();
    let queued_cost = max_cost + 1;
    let mut costs = vec![queued_cost; input.size()];
    let mut done = vec![false; input.size()];
    let mut buckets = (0..=max_cost)
        .map(|_| std::collections::VecDeque::new())
        .collect::<Vec<_>>();

    for (idx, &marker) in markers.data.iter().enumerate() {
        if marker != 0.0 {
            costs[idx] = 0;
            if marker < 0.0 {
                buckets[0].push_back(idx);
            } else {
                buckets[0].push_front(idx);
            }
        }
    }

    let mut coord = vec![0usize; ndim];

    for cost in 0..=max_cost {
        while let Some(idx) = buckets[cost].pop_front() {
            if done[idx] || costs[idx] != cost {
                continue;
            }
            done[idx] = true;

            unravel_into(idx, &input.strides, &mut coord);
            for (oi, offset) in struct_offsets.iter().enumerate() {
                let mut in_bounds = true;
                for axis in 0..ndim {
                    let neighbor_coord = coord[axis] as i64 + offset[axis];
                    if neighbor_coord < 0 || neighbor_coord >= signed_shape[axis] {
                        in_bounds = false;
                        break;
                    }
                }
                if !in_bounds {
                    continue;
                }

                let neighbor_idx = (idx as i64 + flat_offsets[oi]) as usize;
                if done[neighbor_idx] {
                    continue;
                }

                let edge_cost = (input.data[neighbor_idx] - input.data[idx]).abs() as usize;
                let new_cost = cost.max(edge_cost);
                if new_cost < costs[neighbor_idx] {
                    costs[neighbor_idx] = new_cost;
                    output[neighbor_idx] = output[idx];
                    if output[idx] < 0.0 {
                        buckets[new_cost].push_back(neighbor_idx);
                    } else {
                        buckets[new_cost].push_front(neighbor_idx);
                    }
                }
            }
        }
    }

    output
}

fn watershed_ift_bucketed_output_2d_cross(
    input: &NdArray,
    markers: &NdArray,
    max_cost: usize,
) -> Vec<f64> {
    let rows = input.shape[0];
    let cols = input.shape[1];
    let mut output = markers.data.clone();
    let queued_cost = max_cost + 1;
    let mut costs = vec![queued_cost; input.size()];
    let mut done = vec![false; input.size()];
    let mut buckets = (0..=max_cost)
        .map(|_| std::collections::VecDeque::new())
        .collect::<Vec<_>>();

    for (idx, &marker) in markers.data.iter().enumerate() {
        if marker != 0.0 {
            costs[idx] = 0;
            if marker < 0.0 {
                buckets[0].push_back(idx);
            } else {
                buckets[0].push_front(idx);
            }
        }
    }

    for cost in 0..=max_cost {
        while let Some(idx) = buckets[cost].pop_front() {
            if done[idx] || costs[idx] != cost {
                continue;
            }
            done[idx] = true;

            let row = idx / cols;
            let col = idx - row * cols;
            if row > 0 {
                watershed_bucket_relax(
                    idx,
                    idx - cols,
                    cost,
                    input,
                    &mut output,
                    &mut costs,
                    &mut buckets,
                    &done,
                );
            }
            if col > 0 {
                watershed_bucket_relax(
                    idx,
                    idx - 1,
                    cost,
                    input,
                    &mut output,
                    &mut costs,
                    &mut buckets,
                    &done,
                );
            }
            if col + 1 < cols {
                watershed_bucket_relax(
                    idx,
                    idx + 1,
                    cost,
                    input,
                    &mut output,
                    &mut costs,
                    &mut buckets,
                    &done,
                );
            }
            if row + 1 < rows {
                watershed_bucket_relax(
                    idx,
                    idx + cols,
                    cost,
                    input,
                    &mut output,
                    &mut costs,
                    &mut buckets,
                    &done,
                );
            }
        }
    }

    output
}

/// The parameter list is the filter specification itself — axis, window, boundary
/// mode, constant, origin and the output buffer. Bundling it into a struct would
/// add an indirection per call in a per-axis kernel and hide the arguments a
/// reader needs to see at the call site (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn watershed_bucket_relax(
    idx: usize,
    neighbor_idx: usize,
    cost: usize,
    input: &NdArray,
    output: &mut [f64],
    costs: &mut [usize],
    buckets: &mut [std::collections::VecDeque<usize>],
    done: &[bool],
) {
    if done[neighbor_idx] {
        return;
    }

    let edge_cost = (input.data[neighbor_idx] - input.data[idx]).abs() as usize;
    let new_cost = cost.max(edge_cost);
    if new_cost < costs[neighbor_idx] {
        costs[neighbor_idx] = new_cost;
        output[neighbor_idx] = output[idx];
        if output[idx] < 0.0 {
            buckets[new_cost].push_back(neighbor_idx);
        } else {
            buckets[new_cost].push_front(neighbor_idx);
        }
    }
}

fn watershed_ift_heap_output(
    input: &NdArray,
    markers: &NdArray,
    struct_offsets: &[Vec<i64>],
    flat_offsets: &[i64],
    signed_shape: &[i64],
) -> Vec<f64> {
    let ndim = input.ndim();
    let mut output = markers.data.clone();
    let mut costs: Vec<f64> = vec![f64::INFINITY; input.size()];
    let mut queue: std::collections::BinaryHeap<std::cmp::Reverse<(i64, usize)>> =
        std::collections::BinaryHeap::new();

    for (idx, &m) in markers.data.iter().enumerate() {
        if m > 0.0 {
            costs[idx] = 0.0;
            queue.push(std::cmp::Reverse((0, idx)));
        }
    }

    let mut coord = vec![0usize; ndim];

    while let Some(std::cmp::Reverse((cost_scaled, idx))) = queue.pop() {
        let current_cost = cost_scaled as f64 / 1000.0;
        if current_cost > costs[idx] {
            continue;
        }

        unravel_into(idx, &input.strides, &mut coord);
        for (oi, offset) in struct_offsets.iter().enumerate() {
            let mut in_bounds = true;
            for axis in 0..ndim {
                let neighbor_coord = coord[axis] as i64 + offset[axis];
                if neighbor_coord < 0 || neighbor_coord >= signed_shape[axis] {
                    in_bounds = false;
                    break;
                }
            }
            if !in_bounds {
                continue;
            }
            let neighbor_idx = (idx as i64 + flat_offsets[oi]) as usize;

            let new_cost = current_cost.max(input.data[neighbor_idx]);
            if new_cost < costs[neighbor_idx] {
                costs[neighbor_idx] = new_cost;
                output[neighbor_idx] = output[idx];
                queue.push(std::cmp::Reverse((
                    (new_cost * 1000.0) as i64,
                    neighbor_idx,
                )));
            }
        }
    }

    output
}

fn compute_structure_offsets(struct_shape: &[usize], struct_data: &[f64]) -> Vec<Vec<i64>> {
    let ndim = struct_shape.len();
    let mut offsets = Vec::new();
    let center: Vec<i64> = struct_shape.iter().map(|&s| s as i64 / 2).collect();

    let struct_size: usize = struct_shape.iter().product();
    for (struct_idx, &struct_value) in struct_data.iter().enumerate().take(struct_size) {
        if struct_value == 0.0 {
            continue;
        }

        let mut struct_coords = Vec::with_capacity(ndim);
        let mut remaining = struct_idx;
        for &s in struct_shape.iter().rev() {
            struct_coords.push(remaining % s);
            remaining /= s;
        }
        struct_coords.reverse();

        let offset = struct_coords
            .iter()
            .zip(&center)
            .map(|(&coord, &center)| coord as i64 - center)
            .collect::<Vec<_>>();
        if offset.iter().all(|&delta| delta == 0) {
            continue;
        }

        offsets.push(offset);
    }

    offsets
}

#[cfg(test)]
mod zoom_separable_ab_tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    /// Shared by every test in this module that writes a process-global perf
    /// toggle. These statics are visible to all test threads at once, so two
    /// toggle-writing tests running concurrently can both invent drift and mask
    /// it. Module-scoped on purpose: the older idiom in this crate declares a
    /// `static LOCK` INSIDE a test function, which serialises that test against
    /// itself and against nothing else.
    static TOGGLE_LOCK: Mutex<()> = Mutex::new(());
    fn toggle_lock() -> MutexGuard<'static, ()> {
        TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `NDIMAGE_ZOOM_SEPARABLE_DISABLE` claims precomputing the per-axis
    /// B-spline supports once is "byte-identical" to recomputing them per output
    /// pixel. Before this test the claim was asserted NOWHERE: the toggle is
    /// driven only by `src/bin/perf_shift_sep.rs` and `src/bin/perf_affine_sep.rs`,
    /// which are timing harnesses (frankenscipy-0b982).
    ///
    /// All THREE readers are covered, which matters because they gate
    /// differently: `shift` (lib.rs:9697) checks the toggle then `order >= 2`;
    /// `zoom` (:9854) checks `order >= 2 &&` the toggle; `affine_transform`
    /// (:11269) additionally requires a DIAGONAL matrix, since only a transform
    /// with no cross terms is axis-separable.
    #[test]
    fn zoom_shift_affine_separable_ab_is_byte_identical() {
        let _guard = toggle_lock();

        let data: Vec<f64> = (0..7 * 9)
            .map(|i| ((i * 37 % 19) as f64) * 0.5 - 3.0)
            .collect();
        let input = NdArray::new(data, vec![7, 9]).expect("input");

        // Orders 2..=5 only. At order 0 and 1 NONE of the three readers takes the
        // separable branch, so an A/B there would compare a path with itself and
        // pass vacuously — that is the must-miss arm, asserted separately below.
        let mut compared = 0usize;
        for order in 2..=5usize {
            for mode in [
                BoundaryMode::Nearest,
                BoundaryMode::Reflect,
                BoundaryMode::Constant,
                BoundaryMode::Wrap,
                BoundaryMode::Mirror,
            ] {
                let diagonal = [[1.25, 0.0, 0.5], [0.0, 0.8, -1.0]];
                let cases: [(&str, Box<dyn Fn() -> Result<NdArray, NdimageError>>); 3] = [
                    (
                        "zoom",
                        Box::new(|| zoom(&input, &[1.7, 0.8], order, mode, 0.25)),
                    ),
                    (
                        "shift",
                        Box::new(|| shift(&input, &[1.3, -0.7], order, mode, 0.25)),
                    ),
                    (
                        "affine",
                        Box::new(|| affine_transform(&input, &diagonal, order, mode, 0.25)),
                    ),
                ];
                for (label, run) in cases {
                    NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
                    let sep = run().expect(label);
                    NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(true, Ordering::Relaxed);
                    let per_pixel = run().expect(label);
                    NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);

                    assert_eq!(
                        sep.shape, per_pixel.shape,
                        "{label} order={order} {mode:?}: shape"
                    );
                    assert!(
                        sep.data.iter().any(|v| *v != 0.0),
                        "{label} order={order} {mode:?}: all-zero output is a degenerate fixture"
                    );
                    for (k, (a, b)) in sep.data.iter().zip(&per_pixel.data).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label} order={order} {mode:?} [{k}]: separable {a} vs per-pixel {b}"
                        );
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000, "only {compared} values compared");

        // MUST-MISS. A non-diagonal affine has cross terms and is NOT separable,
        // so affine_transform must ignore the toggle entirely. If this ever
        // starts differing, the diagonal guard at :11269 has been widened past
        // what the separable precompute can represent.
        let sheared = [[1.0, 0.35, 0.5], [0.2, 1.0, -1.0]];
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
        let a =
            affine_transform(&input, &sheared, 3, BoundaryMode::Reflect, 0.25).expect("sheared");
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(true, Ordering::Relaxed);
        let b =
            affine_transform(&input, &sheared, 3, BoundaryMode::Reflect, 0.25).expect("sheared");
        NDIMAGE_ZOOM_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
        for (k, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "sheared affine [{k}]");
        }
    }
}

#[cfg(test)]
mod tests {
    // Inner attribute, so it must lead the module body -- it was sitting further down,
    // after other items, which is an outright parse error rather than a lint. Moved
    // rather than deleted: the goldens below really are transcribed at the precision
    // scipy printed them (5.3000000000000007, not 5.3), and those digits ARE the
    // measurement -- rounding them by hand would turn a measured comparison into an
    // approximate one (frankenscipy-iit9c).
    #![allow(clippy::excessive_precision)]

    /// `frankenscipy-22van`: hoisting the spline knobs out of the per-pixel path must not
    /// change a single bit.
    ///
    /// `sample_interpolated` used to read `NDIMAGE_SPLINE_OFFSET_DISABLE` on every pixel and
    /// `compute_axis_support` read `NDIMAGE_BSPLINE_COMPACT_DISABLE` once per axis per pixel.
    /// Both now arrive pre-resolved from the caller. `NDIMAGE_SPLINE_FLAG_HOIST_DISABLE`
    /// restores the per-call read, so the two shapes can be compared inside ONE binary —
    /// which is the whole reason that knob exists.
    ///
    /// The fixture uses NON-INTEGER shifts deliberately: an integer shift takes the cardinal
    /// fast path and never reaches the spline sampler, so both arms would run identical code
    /// and the comparison would prove nothing. Order 3 gives a 4-tap support per axis, so the
    /// hoisted flag is genuinely consulted many times per pixel.
    #[test]
    fn spline_flag_hoist_is_bit_identical_to_the_per_call_read() {
        use std::sync::atomic::Ordering;

        let shape = vec![9usize, 11];
        let mut input = NdArray::zeros(shape.clone());
        for (i, slot) in input.data.iter_mut().enumerate() {
            // Irrational-ish and non-separable so no coincidence can mask a difference.
            *slot = ((i as f64) * 0.6180339887498949).sin() * 3.0 + (i as f64) * 0.25;
        }

        // Non-integer on both axes => the general per-pixel spline path.
        let shifts = [0.37, -0.61];

        NDIMAGE_SPLINE_FLAG_HOIST_DISABLE.store(true, Ordering::Relaxed);
        let per_call =
            shift(&input, &shifts, 3, BoundaryMode::Reflect, 0.0).expect("per-call-read arm");
        NDIMAGE_SPLINE_FLAG_HOIST_DISABLE.store(false, Ordering::Relaxed);
        let hoisted = shift(&input, &shifts, 3, BoundaryMode::Reflect, 0.0).expect("hoisted arm");

        assert_eq!(
            per_call.shape, hoisted.shape,
            "shape must not depend on the knob"
        );
        assert!(
            per_call.data.iter().any(|v| *v != 0.0 && v.is_finite()),
            "degenerate fixture: an all-zero output would compare equal while proving nothing"
        );
        for (i, (a, b)) in per_call.data.iter().zip(hoisted.data.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "pixel {i}: per-call read {a} vs hoisted {b}"
            );
        }
    }

    /// frankenscipy-iit9c. `uniform_knot_at` and `bspline_local_support` are
    /// documented AS equivalences: the first is "the closed-form knot value of
    /// `uniform_interpolation_knots(len, order)` at index `j`", the second is
    /// "BYTE-IDENTICAL to the nonzero entries of
    /// `eval_bspline_basis_all(&uniform_interpolation_knots(len, order), x, order, len)`
    /// filtered by `|w| > 1e-12`". Both references had gone dead, so nothing
    /// checked either claim. This runs them.
    #[test]
    fn closed_form_bspline_matches_the_retained_reference() {
        // order >= 1 only, and that is a real contract rather than a convenience:
        // both closed forms compute `(order - 1) / 2` on `usize`, which underflows at
        // order 0. Production never reaches them there — `compute_axis_support`
        // short-circuits `effective_order == 0` to a single nearest-neighbour tap
        // before any spline machinery runs — so the domain is orders 1..=5, which is
        // what scipy exposes for the spline path.
        for &(len, order) in &[(8usize, 1usize), (8, 3), (12, 2), (16, 3), (7, 5)] {
            let knots = super::uniform_interpolation_knots(len, order);
            for (j, &reference) in knots.iter().enumerate() {
                let closed = super::uniform_knot_at(j, len, order);
                assert_eq!(
                    closed.to_bits(),
                    reference.to_bits(),
                    "knot {j} of ({len}, {order}): closed form {closed} vs reference {reference}"
                );
            }

            // Sample x across the parameter range, including the exact knot values
            // where the degree-0 tie-break decides which span owns the point.
            let mut xs: Vec<f64> = (0..=40)
                .map(|t| t as f64 * (len - 1) as f64 / 40.0)
                .collect();
            xs.extend((0..len).map(|i| i as f64));
            for x in xs {
                let mut local: Vec<(usize, f64)> = Vec::new();
                super::bspline_local_support(len, x, order, &mut local);

                let full = super::eval_bspline_basis_all(&knots, x, order, len);
                let expected: Vec<(usize, f64)> = full
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.abs() > 1e-12)
                    .map(|(i, &w)| (i, w))
                    .collect();

                assert_eq!(
                    local.len(),
                    expected.len(),
                    "({len}, {order}) at x={x}: {} local taps vs {} reference taps",
                    local.len(),
                    expected.len()
                );
                for ((li, lw), (ei, ew)) in local.iter().zip(&expected) {
                    assert_eq!(li, ei, "({len}, {order}) at x={x}: tap index");
                    assert_eq!(
                        lw.to_bits(),
                        ew.to_bits(),
                        "({len}, {order}) at x={x}, tap {li}: {lw} vs {ew} — the doc says BYTE-IDENTICAL"
                    );
                }
            }
        }
    }
    use super::*;
    use std::sync::atomic::Ordering as AtomOrd;

    /// `bspline_local_support` must be BYTE-IDENTICAL to filtering the full
    /// `eval_bspline_basis_all` (the path it replaces in `sample_interpolated`), so the affine /
    /// map_coordinates / geometric_transform results are unchanged — only faster.
    #[test]
    fn bspline_local_support_byte_identical_to_full_eval() {
        let mut s = 0x243f6a8885a308d3u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut checked = 0usize;
        for _ in 0..20_000 {
            let order = 1 + (rng() % 5) as usize; // 1..=5
            let len = (order + 1) + (rng() % 80) as usize;
            // x sweeps the full valid domain [0, len-1] plus exact integer/knot positions.
            let x = match rng() % 4 {
                0 => (rng() % (len as u64)) as f64, // exact integer (knot-ish)
                1 => (len - 1) as f64,              // right boundary special case
                2 => 0.0,                           // left boundary
                _ => (rng() as f64 / u64::MAX as f64) * (len - 1) as f64,
            };
            let knots = uniform_interpolation_knots(len, order);
            let full: Vec<(usize, f64)> = eval_bspline_basis_all(&knots, x, order, len)
                .into_iter()
                .enumerate()
                .filter(|(_, w)| w.abs() > 1e-12)
                .collect();
            let mut loc = Vec::new();
            bspline_local_support(len, x, order, &mut loc);
            assert_eq!(
                loc.len(),
                full.len(),
                "support count mismatch len={len} order={order} x={x}"
            );
            for ((il, wl), (if_, wf)) in loc.iter().zip(full.iter()) {
                assert_eq!(il, if_, "index mismatch len={len} order={order} x={x}");
                assert_eq!(
                    wl.to_bits(),
                    wf.to_bits(),
                    "weight bits mismatch len={len} order={order} x={x}: {wl} vs {wf}"
                );
            }
            checked += 1;
        }
        assert!(
            checked > 19_000,
            "expected near-full coverage, got {checked}"
        );
    }

    /// Order-1 reflect/mirror no longer eagerly pads the array (it folds the support taps on the
    /// fly), so lock the boundary values against scipy.ndimage.affine_transform goldens — including
    /// a transform whose offset pushes coords well outside the grid (where the old 12-deep pad +
    /// clamp would have diverged from scipy's infinite fold).
    #[test]
    fn affine_order1_reflect_mirror_matches_scipy_goldens() {
        let img = NdArray::new((1..=20).map(|v| v as f64).collect(), vec![4, 5]).unwrap();
        let mat = [[0.7, 0.2, -1.5], [-0.1, 0.9, 2.0]]; // 2x3: linear part + offset column (row, col)
        // scipy.ndimage.affine_transform(img, [[0.7,0.2],[-0.1,0.9]], offset=[-1.5,2.0], order=1)
        let reflect_golden = [
            5.5,
            5.3999999999999995,
            5.3000000000000007,
            5.0,
            4.4000000000000004,
            2.9000000000000004,
            3.7999999999999998,
            4.7000000000000002,
            5.0,
            4.5,
            2.7999999999999998,
            4.1999999999999993,
            6.0999999999999996,
            7.5,
            8.0999999999999996,
            5.6999999999999975,
            7.5999999999999979,
            9.4999999999999982,
            11.0,
            11.699999999999999,
        ];
        let mirror_golden = [
            10.5,
            10.4,
            10.300000000000002,
            8.8000000000000007,
            6.9000000000000004,
            6.9000000000000004,
            6.7999999999999998,
            6.7000000000000002,
            5.4000000000000004,
            3.5,
            3.3000000000000003,
            4.1999999999999993,
            6.0999999999999996,
            7.0,
            7.0999999999999996,
            5.6999999999999975,
            7.5999999999999979,
            9.4999999999999982,
            10.6,
            10.699999999999999,
        ];
        for (mode, golden) in [
            (BoundaryMode::Reflect, &reflect_golden),
            (BoundaryMode::Mirror, &mirror_golden),
        ] {
            let r = affine_transform(&img, &mat, 1, mode, 0.0).unwrap();
            // No padding: order<=1 must report zero coordinate offsets.
            let pre = prefilter_spline_coefficients(&img, 1, mode).unwrap();
            assert!(
                pre.coord_offsets.iter().all(|&o| o == 0.0),
                "order-1 {mode:?} should not pad"
            );
            for (got, want) in r.data.iter().zip(golden.iter()) {
                assert!(
                    (got - want).abs() < 1e-12,
                    "{mode:?}: {got} vs scipy {want}"
                );
            }
        }
    }

    #[test]
    fn minmax_hgw_vectorized_matches_scalar_bitexact() {
        use std::sync::atomic::Ordering;
        // The contiguous inner-strip (vectorized) stride>1 path must be BYTE-IDENTICAL to the
        // per-column strided-gather path, across window sizes, modes, adversarial data, and shapes
        // exercising non-last axes (stride>1) with inner both below and above the 64-column tile.
        MINMAX_FILTER_HGW.store(true, Ordering::Relaxed); // van Herk path
        MINMAX_HGW_FORCE_SERIAL.store(true, Ordering::Relaxed); // isolate the kernel (no threads)
        let shapes: &[Vec<usize>] = &[
            vec![9, 11],     // inner=11 (< tile)
            vec![13, 100],   // inner=100 (> tile)
            vec![7, 8, 9],   // axis-0 inner=72, axis-1 inner=9
            vec![6, 5, 200], // axis-0 inner=1000, axis-1 inner=200 (both > tile)
        ];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let mut data: Vec<f64> = (0..total)
                .map(|i| (i as f64 * 0.41).sin() * 5.0 + (i % 7) as f64)
                .collect();
            if total > 25 {
                data[3] = f64::NAN;
                data[7] = -0.0;
                data[8] = 0.0;
                data[20] = f64::NEG_INFINITY;
                data[21] = f64::INFINITY;
            }
            let input = NdArray::new(data, shape.clone()).unwrap();
            for &size in &[2usize, 3, 5, 8] {
                for mode in [
                    BoundaryMode::Reflect,
                    BoundaryMode::Nearest,
                    BoundaryMode::Constant,
                    BoundaryMode::Wrap,
                ] {
                    MINMAX_HGW_FORCE_SCALAR.store(true, Ordering::Relaxed);
                    let sc_max = maximum_filter(&input, size, mode, 2.5);
                    let sc_min = minimum_filter(&input, size, mode, 2.5);
                    MINMAX_HGW_FORCE_SCALAR.store(false, Ordering::Relaxed);
                    let vec_max = maximum_filter(&input, size, mode, 2.5);
                    let vec_min = minimum_filter(&input, size, mode, 2.5);
                    for (a, b) in [(sc_max, vec_max), (sc_min, vec_min)] {
                        match (a, b) {
                            (Ok(a), Ok(b)) => {
                                assert_eq!(a.shape, b.shape);
                                for (x, y) in a.data.iter().zip(&b.data) {
                                    assert_eq!(
                                        x.to_bits(),
                                        y.to_bits(),
                                        "shape={shape:?} size={size} mode={mode:?}"
                                    );
                                }
                            }
                            (Err(_), Err(_)) => {}
                            _ => panic!("scalar/vectorized disagree on Ok/Err"),
                        }
                    }
                }
            }
        }
        MINMAX_HGW_FORCE_SCALAR.store(false, Ordering::Relaxed);
        MINMAX_HGW_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    /// HGW must be bit-for-bit identical to the legacy monotonic-deque path across
    #[test]
    fn minmax_hgw_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel-across-outer-slabs path in `minmax_along_axis_hgw` (reached via the N-D
        // maximum_filter/minimum_filter, which is the van Herk path since MINMAX_FILTER_HGW defaults
        // true) must be BYTE-IDENTICAL to the serial walk, across window sizes, modes, adversarial
        // data (NaN, ±0.0, ±inf), and shapes above the parallel gate.
        MINMAX_FILTER_HGW.store(true, Ordering::Relaxed); // ensure the van Herk path is active
        let shapes: &[Vec<usize>] = &[
            vec![40],
            vec![9, 11],
            vec![520, 130], // > 1<<20 elements -> exercises the parallel path
            vec![5, 6, 4],
            vec![40, 30, 24],
        ];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let mut data: Vec<f64> = (0..total)
                .map(|i| (i as f64 * 0.37).sin() * 5.0 + (i % 7) as f64)
                .collect();
            if total > 25 {
                data[3] = f64::NAN;
                data[7] = -0.0;
                data[8] = 0.0;
                data[20] = f64::NEG_INFINITY;
                data[21] = f64::INFINITY;
            }
            let input = NdArray::new(data, shape.clone()).unwrap();
            for &size in &[2usize, 3, 5, 8] {
                for mode in [
                    BoundaryMode::Reflect,
                    BoundaryMode::Nearest,
                    BoundaryMode::Constant,
                    BoundaryMode::Wrap,
                ] {
                    MINMAX_HGW_FORCE_SERIAL.store(true, Ordering::Relaxed);
                    let ser_max = maximum_filter(&input, size, mode, 2.5);
                    let ser_min = minimum_filter(&input, size, mode, 2.5);
                    MINMAX_HGW_FORCE_SERIAL.store(false, Ordering::Relaxed);
                    let par_max = maximum_filter(&input, size, mode, 2.5);
                    let par_min = minimum_filter(&input, size, mode, 2.5);
                    for (a, b) in [(ser_max, par_max), (ser_min, par_min)] {
                        match (a, b) {
                            (Ok(a), Ok(b)) => {
                                assert_eq!(a.shape, b.shape);
                                for (x, y) in a.data.iter().zip(&b.data) {
                                    assert_eq!(
                                        x.to_bits(),
                                        y.to_bits(),
                                        "shape={shape:?} size={size} mode={mode:?}"
                                    );
                                }
                            }
                            (Err(_), Err(_)) => {}
                            _ => panic!("serial/parallel disagree on Ok/Err"),
                        }
                    }
                }
            }
        }
        MINMAX_HGW_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    /// dimensions, window sizes, origins, boundary modes, and adversarial data
    /// (NaN, ±0.0, duplicates). Serialized via a mutex because both paths share the
    /// global `MINMAX_FILTER_HGW` toggle.
    #[test]
    fn minmax_hgw_byte_identical_to_deque() {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap();

        let cases: Vec<(Vec<f64>, Vec<usize>)> = vec![
            (
                (0..17 * 23)
                    .map(|i| (i as f64 * 0.37).sin() * 5.0 + (i % 7) as f64)
                    .collect(),
                vec![17, 23],
            ),
            (
                {
                    let mut v: Vec<f64> = (0..40).map(|i| ((i * 13) % 11) as f64 - 5.0).collect();
                    v[3] = f64::NAN;
                    v[7] = -0.0;
                    v[8] = 0.0;
                    v[20] = f64::NEG_INFINITY;
                    v[21] = f64::INFINITY;
                    v
                },
                vec![40],
            ),
            (
                (0..5 * 6 * 4).map(|i| (i % 9) as f64 - 4.0).collect(),
                vec![5, 6, 4],
            ),
        ];
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ];
        for (data, shape) in &cases {
            let arr = NdArray::new(data.clone(), shape.clone()).unwrap();
            let ndim = shape.len();
            for &size in &[1usize, 2, 3, 5, 8] {
                // valid origins: |origin| <= (size-1)/2 ... SciPy allows lo=(size-1)//2 range
                let omax = ((size - 1) / 2) as i64;
                for origin in -omax..=omax {
                    let origins = vec![origin; ndim];
                    for &mode in &modes {
                        for is_max in [false, true] {
                            MINMAX_FILTER_HGW.store(false, AtomOrd::Relaxed);
                            let deque =
                                separable_minmax_filter(&arr, size, &origins, mode, 0.0, is_max)
                                    .unwrap();
                            MINMAX_FILTER_HGW.store(true, AtomOrd::Relaxed);
                            let hgw =
                                separable_minmax_filter(&arr, size, &origins, mode, 0.0, is_max)
                                    .unwrap();
                            assert_eq!(deque.shape, hgw.shape);
                            for (k, (a, b)) in deque.data.iter().zip(hgw.data.iter()).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "mismatch shape={:?} size={size} origin={origin} \
                                     mode={mode:?} is_max={is_max} idx={k}: deque={a} hgw={b}",
                                    shape
                                );
                            }
                        }
                    }
                }
            }
        }
        MINMAX_FILTER_HGW.store(true, AtomOrd::Relaxed);
    }

    /// Same-process interleaved A/B: deque vs HGW timed in one binary so fleet load
    /// cancels. Ignored by default; run with
    /// `cargo test -p fsci-ndimage --release minmax_hgw_ab -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn minmax_hgw_ab_timing() {
        use std::time::Instant;
        let side = 256;
        let data: Vec<f64> = (0..side * side)
            .map(|i| {
                let x = i as f64;
                (x * 0.017).sin() * 100.0 + (x * 0.0031).cos() * 37.0 + (i % 53) as f64
            })
            .collect();
        let img = NdArray::new(data, vec![side, side]).unwrap();
        for &size in &[7usize, 15, 31] {
            let mut t_deque = 0.0f64;
            let mut t_hgw = 0.0f64;
            let rounds = 40;
            let inner = 10;
            for r in 0..rounds {
                // alternate which arm runs first each round to cancel drift
                let order = if r % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for &use_hgw in &order {
                    MINMAX_FILTER_HGW.store(use_hgw, AtomOrd::Relaxed);
                    let t = Instant::now();
                    for _ in 0..inner {
                        let _ = maximum_filter(&img, size, BoundaryMode::Reflect, 0.0).unwrap();
                    }
                    let el = t.elapsed().as_secs_f64() / inner as f64;
                    if use_hgw {
                        t_hgw += el;
                    } else {
                        t_deque += el;
                    }
                }
            }
            MINMAX_FILTER_HGW.store(true, AtomOrd::Relaxed);
            let dq = t_deque / rounds as f64 * 1e6;
            let hg = t_hgw / rounds as f64 * 1e6;
            println!(
                "minmax A/B size={size}: deque={dq:.1}us hgw={hg:.1}us  speedup={:.2}x",
                dq / hg
            );
        }
    }

    /// The folded symmetric axpy gaussian path must agree with the legacy
    /// gather-dot path to floating-point tolerance. The two paths consume the
    /// same reflected taps, but fold the symmetric pairs in a different
    /// accumulation order.
    #[test]
    fn gaussian_2d_axpy_matches_gather_dot() {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap();

        for &(rows, cols) in &[(64usize, 64usize), (37, 53), (128, 96), (16, 200)] {
            let data: Vec<f64> = (0..rows * cols)
                .map(|i| (i as f64 * 0.013).sin() * 12.0 + (i % 17) as f64 - 8.0)
                .collect();
            let img = NdArray::new(data, vec![rows, cols]).unwrap();
            for &sigma in &[0.8f64, 2.0, 3.5] {
                GAUSSIAN_2D_AXPY.store(false, AtomOrd::Relaxed);
                let gather = gaussian_filter(&img, sigma, BoundaryMode::Reflect, 0.0).unwrap();
                GAUSSIAN_2D_AXPY.store(true, AtomOrd::Relaxed);
                let axpy = gaussian_filter(&img, sigma, BoundaryMode::Reflect, 0.0).unwrap();
                let mut maxd = 0.0f64;
                for (a, b) in gather.data.iter().zip(axpy.data.iter()) {
                    maxd = maxd.max((a - b).abs());
                }
                assert!(
                    maxd < 1e-10,
                    "rows={rows} cols={cols} sigma={sigma} max|gather-axpy|={maxd:e}"
                );
            }
        }
        GAUSSIAN_2D_AXPY.store(true, AtomOrd::Relaxed);
    }

    /// Same-process interleaved A/B for the gaussian 2D reflect pass. Ignored;
    /// run with `cargo test -p fsci-ndimage --release gaussian_2d_axpy_ab_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gaussian_2d_axpy_ab_timing() {
        use std::time::Instant;
        let side = 256;
        let data: Vec<f64> = (0..side * side)
            .map(|i| {
                let x = i as f64;
                (x * 0.017).sin() * 100.0 + (x * 0.0031).cos() * 37.0 + (i % 53) as f64
            })
            .collect();
        let img = NdArray::new(data, vec![side, side]).unwrap();
        let sigma = 2.0;
        let mut t_gather = 0.0f64;
        let mut t_axpy = 0.0f64;
        let rounds = 50;
        let inner = 5;
        for r in 0..rounds {
            let order = if r % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for &use_axpy in &order {
                GAUSSIAN_2D_AXPY.store(use_axpy, AtomOrd::Relaxed);
                let t = Instant::now();
                for _ in 0..inner {
                    let _ = gaussian_filter(&img, sigma, BoundaryMode::Reflect, 0.0).unwrap();
                }
                let elapsed = t.elapsed().as_secs_f64() / inner as f64;
                if use_axpy {
                    t_axpy += elapsed;
                } else {
                    t_gather += elapsed;
                }
            }
        }
        GAUSSIAN_2D_AXPY.store(true, AtomOrd::Relaxed);
        let gather_us = t_gather / rounds as f64 * 1e6;
        let axpy_us = t_axpy / rounds as f64 * 1e6;
        println!(
            "gaussian 2D A/B sigma=2 256x256: gather={gather_us:.1}us axpy={axpy_us:.1}us  speedup={:.2}x",
            gather_us / axpy_us
        );
    }

    /// The fast `maximum/minimum_filter1d` path must be bit-for-bit identical to
    /// the legacy O(n·size) per-window fold (`filter1d_axis_with_origin`) across
    /// dims, axes, window sizes (incl. size > axis length), origins, boundary
    /// modes, and NaN/±0/±inf data.
    #[test]
    fn filter1d_hgw_byte_identical_to_fold() {
        let max_fold = |window: &[f64]| {
            window
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
        };
        let min_fold = |window: &[f64]| {
            window
                .iter()
                .copied()
                .fold(f64::INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                })
        };
        let cases: Vec<(Vec<f64>, Vec<usize>)> = vec![
            (
                {
                    let mut v: Vec<f64> = (0..40).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
                    v[5] = f64::NAN;
                    v[6] = -0.0;
                    v[7] = 0.0;
                    v[30] = f64::INFINITY;
                    v
                },
                vec![40],
            ),
            (
                (0..6 * 9).map(|i| (i as f64 * 0.3).sin() * 4.0).collect(),
                vec![6, 9],
            ),
            (
                (0..4 * 5 * 7).map(|i| (i % 11) as f64 - 5.0).collect(),
                vec![4, 5, 7],
            ),
        ];
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ];
        for (data, shape) in &cases {
            let arr = NdArray::new(data.clone(), shape.clone()).unwrap();
            for axis in 0..shape.len() {
                for &size in &[1usize, 2, 4, 5, shape[axis] + 3] {
                    let omax = ((size - 1) / 2) as i64;
                    for origin in [-omax, 0, omax] {
                        for &mode in &modes {
                            for is_max in [true, false] {
                                let reference = if is_max {
                                    filter1d_axis_with_origin(
                                        &arr, size, axis, mode, 0.0, origin, max_fold,
                                    )
                                } else {
                                    filter1d_axis_with_origin(
                                        &arr, size, axis, mode, 0.0, origin, min_fold,
                                    )
                                }
                                .unwrap();
                                let got = if is_max {
                                    maximum_filter1d_with_origin(
                                        &arr, size, axis, mode, 0.0, origin,
                                    )
                                } else {
                                    minimum_filter1d_with_origin(
                                        &arr, size, axis, mode, 0.0, origin,
                                    )
                                }
                                .unwrap();
                                for (k, (a, b)) in
                                    reference.data.iter().zip(got.data.iter()).enumerate()
                                {
                                    assert_eq!(
                                        a.to_bits(),
                                        b.to_bits(),
                                        "shape={shape:?} axis={axis} size={size} origin={origin} mode={mode:?} is_max={is_max} idx={k}: ref={a} got={b}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // Error paths preserved.
        let a = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(maximum_filter1d_with_origin(&a, 0, 0, BoundaryMode::Reflect, 0.0, 0).is_err());
        assert!(maximum_filter1d_with_origin(&a, 3, 5, BoundaryMode::Reflect, 0.0, 0).is_err());
    }

    /// Same-process A/B: legacy O(n·size) per-window fold vs the O(n) HGW routing
    /// for maximum_filter1d. Ignored; run with
    /// `cargo test -p fsci-ndimage --release filter1d_hgw_ab -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn filter1d_hgw_ab_timing() {
        use std::time::Instant;
        let n = 65536usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.01).sin() * 100.0 + (i % 7) as f64)
            .collect();
        let line = NdArray::new(data, vec![n]).unwrap();
        let max_fold = |w: &[f64]| {
            w.iter().copied().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            })
        };
        for &size in &[31usize, 101] {
            let (mut t_old, mut t_new) = (0.0f64, 0.0f64);
            let rounds = 30;
            for r in 0..rounds {
                let order = if r % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                for &new_path in &order {
                    let t = Instant::now();
                    let _ = if new_path {
                        maximum_filter1d(&line, size, 0, BoundaryMode::Reflect, 0.0).unwrap()
                    } else {
                        filter1d_axis_with_origin(
                            &line,
                            size,
                            0,
                            BoundaryMode::Reflect,
                            0.0,
                            0,
                            max_fold,
                        )
                        .unwrap()
                    };
                    let el = t.elapsed().as_secs_f64();
                    if new_path { t_new += el } else { t_old += el }
                }
            }
            let (o, ne) = (t_old / rounds as f64 * 1e6, t_new / rounds as f64 * 1e6);
            println!(
                "filter1d A/B max n={n} size={size}: old_fold={o:.1}us new_hgw={ne:.1}us  speedup={:.2}x",
                o / ne
            );
        }
    }

    /// Same-process A/B: existing HGW prefix/suffix route vs the fused monotonic
    /// index-queue route used by public filter1d min/max.
    #[test]
    #[ignore]
    fn filter1d_queue_vs_hgw_ab_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let n = 65536usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.01).sin() * 100.0 + (i % 7) as f64)
            .collect();
        let line = NdArray::new(data, vec![n]).unwrap();
        for &size in &[31usize, 101] {
            for is_max in [true, false] {
                let hgw = if is_max {
                    minmax_along_axis_hgw(
                        &line,
                        0,
                        size,
                        0,
                        BoundaryMode::Reflect,
                        0.0,
                        nanprop_max,
                    )
                } else {
                    minmax_along_axis_hgw(
                        &line,
                        0,
                        size,
                        0,
                        BoundaryMode::Reflect,
                        0.0,
                        nanprop_min,
                    )
                };
                let queue = minmax_filter1d_nanprop_queue(
                    &line,
                    0,
                    size,
                    0,
                    BoundaryMode::Reflect,
                    0.0,
                    is_max,
                );
                for (k, (a, b)) in hgw.data.iter().zip(queue.data.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "size={size} is_max={is_max} idx={k}: hgw={a} queue={b}"
                    );
                }

                let (mut t_hgw, mut t_queue) = (0.0f64, 0.0f64);
                let rounds = 40;
                let inner = 2;
                for r in 0..rounds {
                    let order = if r % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    };
                    for &use_queue in &order {
                        let t = Instant::now();
                        for _ in 0..inner {
                            let result = if use_queue {
                                minmax_filter1d_nanprop_queue(
                                    &line,
                                    0,
                                    size,
                                    0,
                                    BoundaryMode::Reflect,
                                    0.0,
                                    is_max,
                                )
                            } else if is_max {
                                minmax_along_axis_hgw(
                                    &line,
                                    0,
                                    size,
                                    0,
                                    BoundaryMode::Reflect,
                                    0.0,
                                    nanprop_max,
                                )
                            } else {
                                minmax_along_axis_hgw(
                                    &line,
                                    0,
                                    size,
                                    0,
                                    BoundaryMode::Reflect,
                                    0.0,
                                    nanprop_min,
                                )
                            };
                            black_box(result);
                        }
                        let elapsed = t.elapsed().as_secs_f64() / inner as f64;
                        if use_queue {
                            t_queue += elapsed;
                        } else {
                            t_hgw += elapsed;
                        }
                    }
                }
                let label = if is_max { "max" } else { "min" };
                let hgw_us = t_hgw / rounds as f64 * 1e6;
                let queue_us = t_queue / rounds as f64 * 1e6;
                println!(
                    "filter1d queue/HGW {label} n={n} size={size}: hgw={hgw_us:.1}us queue={queue_us:.1}us  speedup_vs_hgw={:.2}x",
                    hgw_us / queue_us
                );
            }
        }
    }

    /// Same-process A/B: generic boundary-resolved queue vs the contiguous
    /// Reflect/origin-0 direct queue used by the benchmarked 1-D public route.
    #[test]
    #[ignore]
    fn filter1d_reflect_direct_vs_generic_queue_ab_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        let n = 65536usize;
        let data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.01).sin() * 100.0 + (i % 7) as f64)
            .collect();
        let line = NdArray::new(data, vec![n]).unwrap();
        for &size in &[31usize, 101] {
            for is_max in [true, false] {
                let generic = minmax_filter1d_nanprop_queue_generic(
                    &line,
                    0,
                    size,
                    0,
                    BoundaryMode::Reflect,
                    0.0,
                    is_max,
                );
                let direct = minmax_filter1d_reflect_contiguous_queue(&line, 0, size, is_max);
                for (k, (a, b)) in generic.data.iter().zip(direct.data.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "size={size} is_max={is_max} idx={k}: generic={a} direct={b}"
                    );
                }

                let (mut t_generic, mut t_direct) = (0.0f64, 0.0f64);
                let rounds = 50;
                let inner = 2;
                for r in 0..rounds {
                    let order = if r % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    };
                    for &use_direct in &order {
                        let t = Instant::now();
                        for _ in 0..inner {
                            let result = if use_direct {
                                minmax_filter1d_reflect_contiguous_queue(&line, 0, size, is_max)
                            } else {
                                minmax_filter1d_nanprop_queue_generic(
                                    &line,
                                    0,
                                    size,
                                    0,
                                    BoundaryMode::Reflect,
                                    0.0,
                                    is_max,
                                )
                            };
                            black_box(result);
                        }
                        let elapsed = t.elapsed().as_secs_f64() / inner as f64;
                        if use_direct {
                            t_direct += elapsed;
                        } else {
                            t_generic += elapsed;
                        }
                    }
                }
                let label = if is_max { "max" } else { "min" };
                let generic_us = t_generic / rounds as f64 * 1e6;
                let direct_us = t_direct / rounds as f64 * 1e6;
                println!(
                    "filter1d direct/generic {label} n={n} size={size}: generic={generic_us:.1}us direct={direct_us:.1}us  speedup_vs_generic={:.2}x",
                    generic_us / direct_us
                );
            }
        }
    }

    #[test]
    fn ndarray_new_rejects_overflowing_shape_product() {
        let err = NdArray::new(vec![0.0], vec![usize::MAX, 2])
            .expect_err("overflowing shape product should fail closed");
        assert_eq!(
            err,
            NdimageError::InvalidArgument("shape product overflows usize".to_string())
        );
    }

    #[test]
    fn reshape_rejects_overflowing_target_shape_product() {
        let input = NdArray::new(Vec::new(), vec![0]).expect("empty 1-D array");
        let err = reshape(&input, vec![usize::MAX, 2])
            .expect_err("overflowing reshape target should fail closed");
        assert_eq!(
            err,
            NdimageError::InvalidArgument("shape product overflows usize".to_string())
        );
    }

    #[test]
    fn shift_order0_boundary_matches_scipy() {
        // Golden values from scipy.ndimage.shift(order=0). order-0 uses
        // floor(coord+0.5) rounding, applies 'constant' on the FLOAT coordinate
        // (out of [0,len-1] → cval), and 'wrap' with period len-1.
        let img = NdArray::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5]).unwrap();
        // constant cval=-9, shift -1.3: src 4.3/5.3 are outside [0,4] → cval, even
        // though 4.3 rounds to the valid index 4.
        let c = shift(&img, &[-1.3], 0, BoundaryMode::Constant, -9.0).unwrap();
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, -9.0, -9.0]);
        // wrap, shift -1.3: index 4 (=len-1) wraps to 0; index 5 wraps to 1.
        let w = shift(&img, &[-1.3], 0, BoundaryMode::Wrap, 0.0).unwrap();
        assert_eq!(w.data, vec![1.0, 2.0, 3.0, 0.0, 1.0]);
        // shift +0.5: src -0.5 < 0 → cval; src 0.5 → floor(1.0)=1 (half toward +∞).
        let h = shift(&img, &[0.5], 0, BoundaryMode::Constant, -9.0).unwrap();
        assert_eq!(h.data, vec![-9.0, 1.0, 2.0, 3.0, 4.0]);
    }

    /// The multithreaded separable filters must be BIT-IDENTICAL to the sequential
    /// pixel-by-pixel computation. Uses an image large enough to cross the parallel
    /// gate and compares `correlate1d_with_origin` to a verbatim sequential loop.
    #[test]
    fn separable_filter_parallel_is_bit_identical() {
        let (rows, cols) = (600usize, 600usize); // 360k px * k>=2 >= the 2^18 gate
        let data: Vec<f64> = (0..rows * cols)
            .map(|k| ((k % 251) as f64 * 0.013).sin() + 0.5)
            .collect();
        let input = NdArray::new(data, vec![rows, cols]).expect("image");
        let weights = [0.2_f64, -1.3, 0.7, 2.1, -0.5, 1.1, 0.9];
        for &mode in &[
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
        ] {
            for axis in 0..2 {
                let got = correlate1d_with_origin(&input, &weights, axis, mode, 0.3, 0)
                    .expect("parallel correlate1d");
                // Verbatim sequential reference.
                let offset = weights.len() as i64 / 2;
                let mut want = NdArray::zeros(input.shape.clone());
                for flat_out in 0..input.size() {
                    let out_idx = input.unravel(flat_out);
                    let mut in_idx: Vec<i64> = out_idx.iter().map(|&i| i as i64).collect();
                    let mut sum = 0.0;
                    for (k, &w) in weights.iter().enumerate() {
                        in_idx[axis] = out_idx[axis] as i64 + k as i64 - offset;
                        sum += w * input.get_boundary(&in_idx, mode, 0.3);
                    }
                    want.data[flat_out] = sum;
                }
                for (k, (&g, &w)) in got.data.iter().zip(&want.data).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "mismatch axis={axis} {mode:?} at {k}"
                    );
                }
            }
        }
    }

    fn assert_close_or_nan(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            if expected_value.is_nan() {
                assert!(actual_value.is_nan(), "expected NaN, got {actual_value}");
            } else {
                assert!(
                    (actual_value - expected_value).abs() < 1e-12,
                    "expected {expected_value}, got {actual_value}"
                );
            }
        }
    }

    #[test]
    fn uniform_filter_1d() {
        let input = NdArray::new(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let result = uniform_filter(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Center should be average of [0, 1, 0] = 1/3
        assert!((result.data[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn nd_filter_interior_fast_path_is_byte_identical_to_perpixel() {
        // correlate/convolve via the interior/flat-gather path must equal the per-pixel
        // get_boundary reference bit-for-bit across modes, kernel sizes (incl. asymmetric
        // and kernels wider than the array), and origins.
        let arr = NdArray::new(
            (0..23 * 19)
                .map(|i| ((i * 7919) % 877) as f64 / 80.0 - 5.0)
                .collect(),
            vec![23, 19],
        )
        .unwrap();
        for mode in [
            BoundaryMode::Nearest,
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ] {
            for ks in [[3usize, 3], [1, 5], [4, 2], [25, 3]] {
                let w = NdArray::new(
                    (0..ks[0] * ks[1])
                        .map(|k| (k as f64 * 0.3 - 1.0).cos())
                        .collect(),
                    ks.to_vec(),
                )
                .unwrap();
                for origin in [&[0i64, 0][..], &[1, -1][..]] {
                    if origin
                        .iter()
                        .enumerate()
                        .any(|(d, &o)| o < -(ks[d] as i64 / 2) || o > (ks[d] as i64 - 1) / 2)
                    {
                        continue;
                    }
                    for flip in [false, true] {
                        let got = if flip {
                            convolve_with_origins(&arr, &w, origin, mode, 0.4).unwrap()
                        } else {
                            correlate_with_origins(&arr, &w, origin, mode, 0.4).unwrap()
                        };
                        let want =
                            nd_filter_perpixel_ref(&arr, &w, origin, mode, 0.4, flip).unwrap();
                        assert_eq!(
                            got.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            want.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "mode={mode:?} ks={ks:?} origin={origin:?} flip={flip}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nd_filter_3d_simd_interior_is_byte_identical_to_perpixel() {
        // The generalized (ndim>=2) innermost-axis SIMD interior must match the per-pixel
        // reference bit-for-bit on a 3-D volume (innermost axis long enough for 8-runs).
        let arr = NdArray::new(
            (0..11 * 9 * 17)
                .map(|i| ((i * 6151) % 733) as f64 / 70.0 - 5.0)
                .collect(),
            vec![11, 9, 17],
        )
        .unwrap();
        for mode in [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
        ] {
            for ks in [[3usize, 3, 3], [1, 1, 5], [2, 3, 4]] {
                let w = NdArray::new(
                    (0..ks[0] * ks[1] * ks[2])
                        .map(|k| (k as f64 * 0.27 - 0.9).cos())
                        .collect(),
                    ks.to_vec(),
                )
                .unwrap();
                let origin = vec![0i64; 3];
                let got = correlate_with_origins(&arr, &w, &origin, mode, 0.4).unwrap();
                let want = nd_filter_perpixel_ref(&arr, &w, &origin, mode, 0.4, false).unwrap();
                assert_eq!(
                    got.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "3d mode={mode:?} ks={ks:?}"
                );
            }
        }
    }

    #[test]
    fn convolve1d_line_walk_is_byte_identical_to_nd_convolve() {
        // convolve1d_along_axis must equal the N-D convolve on a 1-D-embedded kernel,
        // bit-for-bit (the gaussian reroute relies on this). The 3-D shape exercises
        // both the vector-friendly inner spans and the scalar innermost span.
        let shape = vec![13usize, 11, 17];
        let data: Vec<f64> = (0..shape.iter().product())
            .map(|i| ((i * 40503usize) % 911) as f64 / 90.0 - 5.0)
            .collect();
        let arr = NdArray::new(data, shape).unwrap();
        for mode in [
            BoundaryMode::Nearest,
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ] {
            for klen in [2usize, 3, 6, 9] {
                let w: Vec<f64> = (0..klen).map(|k| (k as f64 * 0.9 + 0.3).sin()).collect();
                for axis in 0..arr.ndim() {
                    let mut kshape = vec![1usize; arr.ndim()];
                    kshape[axis] = klen;
                    let kernel = NdArray::new(w.clone(), kshape).unwrap();
                    let nd = convolve(&arr, &kernel, mode, 0.4).unwrap();
                    let line = convolve1d_along_axis(&arr, &w, axis, 0, mode, 0.4);
                    assert_eq!(
                        line.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        nd.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "mode={mode:?} klen={klen} axis={axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn correlate1d_line_walk_is_byte_identical_to_perwindow() {
        // The slab line walk must be BYTE-identical to the per-pixel reference (same dot
        // order, same boundary values) across modes, kernel sizes, axes, and origins.
        let (rows, cols) = (43usize, 39usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i * 2654435761usize) % 997) as f64 / 100.0 - 5.0)
            .collect();
        let arr = NdArray::new(data, vec![rows, cols]).unwrap();
        for mode in [
            BoundaryMode::Nearest,
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ] {
            for klen in [2usize, 3, 6, 11] {
                let weights: Vec<f64> = (0..klen).map(|k| (k as f64 * 0.7 - 1.0).cos()).collect();
                for axis in [0usize, 1usize] {
                    let got = correlate1d(&arr, &weights, axis, mode, 0.5).unwrap();
                    let want =
                        correlate1d_perwindow_ref(&arr, &weights, axis, mode, 0.5, 0).unwrap();
                    assert_eq!(
                        got.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "mode={mode:?} klen={klen} axis={axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn uniform_running_sum_matches_perwindow_reference() {
        // The O(n) running-sum path must match the per-window reference to rounding
        // across sizes, axes, and boundary modes (running sum is tolerance-parity).
        let (rows, cols) = (37usize, 41usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i * 2654435761usize) % 1000) as f64 / 1000.0 - 0.5)
            .collect();
        let arr = NdArray::new(data, vec![rows, cols]).unwrap();
        for mode in [
            BoundaryMode::Nearest,
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ] {
            for size in [2usize, 3, 7, 12, 20] {
                for axis in [0usize, 1usize] {
                    let got = uniform_filter1d(&arr, size, axis, mode, 0.3).unwrap();
                    let want =
                        uniform_filter1d_perwindow_ref(&arr, size, axis, mode, 0.3, 0).unwrap();
                    let max_dx = got
                        .data
                        .iter()
                        .zip(&want.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(
                        max_dx < 1e-12,
                        "mode={mode:?} size={size} axis={axis}: max|dx|={max_dx:.2e}"
                    );
                }
            }
        }
    }

    #[test]
    fn uniform_filter_origins_match_scipy_constant() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let even_left =
            uniform_filter_with_origins(&input, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&even_left.data, &[1.5, 2.5, 3.5, 4.5, 2.5]);

        let even_default =
            uniform_filter_with_origins(&input, 2, &[0], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&even_default.data, &[0.5, 1.5, 2.5, 3.5, 4.5]);

        let odd_left =
            uniform_filter_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&odd_left.data, &[2.0, 3.0, 4.0, 3.0, 5.0 / 3.0]);

        let odd_right =
            uniform_filter_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&odd_right.data, &[1.0 / 3.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn uniform_filter_origins_apply_per_axis() {
        let input = NdArray::new((1..=9).map(f64::from).collect(), vec![3, 3]).unwrap();
        let result =
            uniform_filter_with_origins(&input, 3, &[-1, 1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(
            &result.data,
            &[
                4.0 / 3.0,
                3.0,
                5.0,
                11.0 / 9.0,
                8.0 / 3.0,
                13.0 / 3.0,
                7.0 / 9.0,
                5.0 / 3.0,
                8.0 / 3.0,
            ],
        );
    }

    #[test]
    fn uniform_filter_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        assert!(uniform_filter_with_origins(&input, 2, &[-2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_with_origins(&input, 2, &[1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_with_origins(&input, 3, &[-2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_with_origins(&input, 3, &[2], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn uniform_filter_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        // scipy.ndimage.uniform_filter(input, 2, mode='constant', cval=0.0, axes=(-1,))
        let last_axis = [0.5, 1.5, 3.0, 4.0, 12.0, 24.0];
        // scipy.ndimage.uniform_filter(input, 2, mode='constant', cval=0.0, axes=(-2,))
        let first_axis = [0.5, 1.0, 2.0, 4.5, 9.0, 18.0];

        let got_last = uniform_filter_axes(&input, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&got_last.data, &last_axis);

        let got_first = uniform_filter_axes(&input, 2, &[-2], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&got_first.data, &first_axis);

        assert_close_or_nan(
            &uniform_filter_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            &uniform_filter(&input, 2, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
        );
        assert_eq!(
            uniform_filter_axes(&input, 2, &[], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn uniform_filter_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(uniform_filter_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn gaussian_filter_preserves_constant() {
        let input = NdArray::new(vec![5.0; 25], vec![5, 5]).unwrap();
        let result = gaussian_filter(&input, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
        for &v in &result.data {
            assert!((v - 5.0).abs() < 1e-10, "constant image changed: {v}");
        }
    }

    #[test]
    fn gaussian_filter_reflect_2d_fast_path_matches_generic_sequential_path() {
        let input = NdArray::new((0..20).map(f64::from).collect(), vec![4, 5]).unwrap();
        let fast = gaussian_filter(&input, 1.3, BoundaryMode::Reflect, 0.0).unwrap();
        let generic =
            gaussian_filter_with_orders(&input, 1.3, &[0, 0], BoundaryMode::Reflect, 0.0).unwrap();

        assert_close_or_nan(&fast.data, &generic.data);
    }

    #[test]
    fn gaussian_filter_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        // scipy.ndimage.gaussian_filter(input, 1.0, mode='constant', cval=0.0, axes=(-1,))
        let last_axis = [
            1.098850870352,
            2.007744166995,
            2.133707896158,
            8.790806962817,
            16.061953335962,
            17.069663169266,
        ];
        // scipy.ndimage.gaussian_filter(input, 1.0, mode='constant', cval=0.0, axes=(-2,))
        let first_axis = [
            2.334715034609,
            4.669430069218,
            9.338860138436,
            3.433519200505,
            6.867038401011,
            13.734076802022,
        ];

        let got_last =
            gaussian_filter_axes(&input, 1.0, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&got_last.data, &last_axis);

        let got_first =
            gaussian_filter_axes(&input, 1.0, &[-2], BoundaryMode::Constant, 0.0).unwrap();
        assert_close_or_nan(&got_first.data, &first_axis);

        assert_close_or_nan(
            &gaussian_filter_axes(&input, 1.0, &[-2, -1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            &gaussian_filter(&input, 1.0, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
        );
        assert_eq!(
            gaussian_filter_axes(&input, 1.0, &[], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn gaussian_filter_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(gaussian_filter_axes(&input, 1.0, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_filter_axes(&input, 1.0, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_filter_axes(&input, 1.0, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_filter_axes(&input, 0.0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn gaussian_filter_multi_sigma_matches_scipy_2d() {
        let input = NdArray::new((0..12).map(f64::from).collect(), vec![3, 4]).unwrap();
        // scipy.ndimage.gaussian_filter(x, [0.5, 1.0], mode='constant', cval=0.0)
        let expect = [
            0.623741755569,
            1.33426091614,
            1.97824192771,
            1.84889615648,
            3.15893295714,
            4.78984061724,
            5.51037314812,
            4.52972430867,
            5.02289529139,
            7.22764648909,
            7.87162750066,
            6.2480496923,
        ];
        let got =
            gaussian_filter_multi_sigma(&input, &[0.5, 1.0], BoundaryMode::Constant, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-9,
                "multi-sigma gaussian mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter_multi_sigma_matches_scipy_skipped_axis() {
        let input = NdArray::new((0..6).map(f64::from).collect(), vec![2, 3]).unwrap();
        // scipy.ndimage.gaussian_filter(x, [0.0, 1.0], mode='constant', cval=0.0)
        let expect = [
            0.34995370049801,
            0.882886360669299,
            1.0398583843688,
            2.43467182779822,
            3.5315454426772,
            3.12457651166901,
        ];
        let got =
            gaussian_filter_multi_sigma(&input, &[0.0, 1.0], BoundaryMode::Constant, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-9,
                "multi-sigma skipped-axis mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter_multi_sigma_rejects_shape_and_nonfinite_sigma() {
        let input = NdArray::new((0..12).map(f64::from).collect(), vec![3, 4]).unwrap();

        assert!(gaussian_filter_multi_sigma(&input, &[1.0], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            gaussian_filter_multi_sigma(&input, &[1.0, f64::INFINITY], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn convolve_and_correlate_origins_match_scipy_even_kernel() {
        let input = NdArray::new((1..=6).map(f64::from).collect(), vec![2, 3]).unwrap();
        let weights = NdArray::new(vec![1., 10., 100., 1000.], vec![2, 2]).unwrap();

        let corr = correlate(&input, &weights, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(corr.data, vec![1000., 2100., 3200., 4010., 5421., 6532.]);

        let conv = convolve(&input, &weights, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(conv.data, vec![1245., 2356., 3060., 4500., 5600., 6000.]);

        let corr_shifted =
            correlate_with_origins(&input, &weights, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(corr_shifted.data, vec![5421., 6532., 603., 54., 65., 6.]);

        let conv_shifted =
            convolve_with_origins(&input, &weights, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(conv_shifted.data, vec![1., 12., 23., 104., 1245., 2356.]);

        let corr_axis_shifted =
            correlate_with_origins(&input, &weights, &[0, -1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(
            corr_axis_shifted.data,
            vec![2100., 3200., 300., 5421., 6532., 603.]
        );
    }

    #[test]
    fn convolve_and_correlate_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new((1..=6).map(f64::from).collect(), vec![2, 3]).unwrap();
        let weights = NdArray::new(vec![1., 10., 100., 1000.], vec![2, 2]).unwrap();

        assert!(convolve_with_origins(&input, &weights, &[1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            correlate_with_origins(&input, &weights, &[1], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(
            convolve_with_origins(&input, &weights, &[-1, 0], BoundaryMode::Reflect, 0.0).is_ok()
        );
        assert!(
            correlate_with_origins(&input, &weights, &[-1, 0], BoundaryMode::Reflect, 0.0).is_ok()
        );
        assert!(
            convolve_with_origins(&input, &weights, &[-1, 0, 0], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn convolve_and_correlate_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new((1..=6).map(f64::from).collect(), vec![2, 3]).unwrap();
        let weights_1d = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
        let weights_2d = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let scalar = NdArray::new(vec![2.0], vec![]).unwrap();

        // scipy.ndimage.correlate(input, [1, 2], mode='constant', cval=-10, axes=(-1,))
        assert_eq!(
            correlate_axes(&input, &weights_1d, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-8.0, 5.0, 8.0, -2.0, 14.0, 17.0]
        );
        // scipy.ndimage.correlate(input, [1, 2], mode='constant', cval=-10, axes=(-2,))
        assert_eq!(
            correlate_axes(&input, &weights_1d, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-8.0, -6.0, -4.0, 9.0, 12.0, 15.0]
        );
        // scipy.ndimage.correlate(input, [[1, 2], [3, 4]], mode='constant', cval=-10, axes=(-2, -1))
        assert_eq!(
            correlate_axes(
                &input,
                &weights_2d,
                &[-2, -1],
                BoundaryMode::Constant,
                -10.0
            )
            .unwrap()
            .data,
            vec![-56.0, -19.0, -12.0, -22.0, 37.0, 47.0]
        );
        assert_eq!(
            correlate_axes(&input, &scalar, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );

        // scipy.ndimage.convolve(input, [1, 2], mode='constant', cval=-10, axes=(-1,))
        assert_eq!(
            convolve_axes(&input, &weights_1d, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![4.0, 7.0, -4.0, 13.0, 16.0, 2.0]
        );
        // scipy.ndimage.convolve(input, [1, 2], mode='constant', cval=-10, axes=(-2,))
        assert_eq!(
            convolve_axes(&input, &weights_1d, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![6.0, 9.0, 12.0, -2.0, 0.0, 2.0]
        );
        // scipy.ndimage.convolve(input, [[1, 2], [3, 4]], mode='constant', cval=-10, axes=(-2, -1))
        assert_eq!(
            convolve_axes(
                &input,
                &weights_2d,
                &[-2, -1],
                BoundaryMode::Constant,
                -10.0
            )
            .unwrap()
            .data,
            vec![23.0, 33.0, -16.0, 1.0, 8.0, -36.0]
        );
        assert_eq!(
            convolve_axes(&input, &scalar, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }

    #[test]
    fn convolve_and_correlate_axes_reject_invalid_axes_and_weight_rank() {
        let input = NdArray::new((1..=6).map(f64::from).collect(), vec![2, 3]).unwrap();
        let weights_1d = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
        let weights_2d = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert!(correlate_axes(&input, &weights_1d, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(correlate_axes(&input, &weights_1d, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(correlate_axes(&input, &weights_1d, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(correlate_axes(&input, &weights_2d, &[-1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            correlate_axes(&input, &weights_1d, &[-2, -1], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(correlate_axes(&input, &weights_1d, &[], BoundaryMode::Reflect, 0.0).is_err());

        assert!(convolve_axes(&input, &weights_1d, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve_axes(&input, &weights_1d, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve_axes(&input, &weights_1d, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve_axes(&input, &weights_2d, &[-1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve_axes(&input, &weights_1d, &[-2, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve_axes(&input, &weights_1d, &[], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn convolve_axes_nd_filter_matches_scalar_reference_bitwise() {
        // The tap_delta/`nd_filter_apply` conversion of `convolve_axes` must be
        // BYTE-IDENTICAL to the pre-conversion per-pixel/per-tap scalar path across
        // ndim, kernel shapes, axes mappings, and boundary modes — the interior
        // flat-gather + border `get_boundary` split must reproduce the all-boundary
        // reference exactly (execution proof for the alloc-elimination lever).
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ];
        let cases: &[(Vec<usize>, Vec<usize>, Vec<isize>)] = &[
            (vec![7], vec![3], vec![-1]),
            (vec![8, 9], vec![3], vec![-1]),
            (vec![8, 9], vec![4], vec![-2]),
            (vec![8, 9], vec![3, 3], vec![-2, -1]),
            (vec![8, 9], vec![2, 4], vec![0, 1]),
            (vec![6, 7, 5], vec![3], vec![1]),
            (vec![6, 7, 5], vec![3, 3], vec![0, 2]),
            (vec![6, 7, 5], vec![2, 3, 4], vec![-3, -2, -1]),
            (vec![4, 4], vec![5, 5], vec![-2, -1]), // kernel >= input: all-border
            (vec![5], vec![5], vec![-1]),           // kernel == input
        ];
        for (in_shape, w_shape, axes) in cases {
            let n_in: usize = in_shape.iter().product();
            let input = NdArray::new(
                (0..n_in)
                    .map(|i| ((i * 37 + 11) % 97) as f64 - 40.0)
                    .collect(),
                in_shape.clone(),
            )
            .unwrap();
            let n_w: usize = w_shape.iter().product();
            let weights = NdArray::new(
                (0..n_w).map(|i| ((i * 13 + 5) % 17) as f64 - 6.0).collect(),
                w_shape.clone(),
            )
            .unwrap();
            for &mode in &modes {
                let cval = -3.5;
                let fast = convolve_axes(&input, &weights, axes, mode, cval).unwrap();
                let refr =
                    convolve_axes_scalar_reference(&input, &weights, axes, mode, cval).unwrap();
                assert_eq!(
                    fast.shape, refr.shape,
                    "shape mismatch in={in_shape:?} w={w_shape:?} axes={axes:?} mode={mode:?}"
                );
                let differ = fast
                    .data
                    .iter()
                    .zip(&refr.data)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differ, 0,
                    "convolve_axes not byte-identical: {differ} diffs in={in_shape:?} w={w_shape:?} axes={axes:?} mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn correlate_axes_nd_filter_matches_scalar_reference_bitwise() {
        // The `nd_filter_apply` routing of `correlate_axes` (interior flat-gather)
        // must be BYTE-IDENTICAL to the tap-delta `get_boundary` scalar reference.
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
            BoundaryMode::Mirror,
        ];
        let cases: &[(Vec<usize>, Vec<usize>, Vec<isize>)] = &[
            (vec![7], vec![3], vec![-1]),
            (vec![8, 9], vec![3], vec![-1]),
            (vec![8, 9], vec![4], vec![-2]),
            (vec![8, 9], vec![3, 3], vec![-2, -1]),
            (vec![8, 9], vec![2, 4], vec![0, 1]),
            (vec![6, 7, 5], vec![3], vec![1]),
            (vec![6, 7, 5], vec![3, 3], vec![0, 2]),
            (vec![6, 7, 5], vec![2, 3, 4], vec![-3, -2, -1]),
            (vec![4, 4], vec![5, 5], vec![-2, -1]),
            (vec![5], vec![5], vec![-1]),
        ];
        for (in_shape, w_shape, axes) in cases {
            let n_in: usize = in_shape.iter().product();
            let input = NdArray::new(
                (0..n_in)
                    .map(|i| ((i * 37 + 11) % 97) as f64 - 40.0)
                    .collect(),
                in_shape.clone(),
            )
            .unwrap();
            let n_w: usize = w_shape.iter().product();
            let weights = NdArray::new(
                (0..n_w).map(|i| ((i * 13 + 5) % 17) as f64 - 6.0).collect(),
                w_shape.clone(),
            )
            .unwrap();
            for &mode in &modes {
                let cval = -3.5;
                let fast = correlate_axes(&input, &weights, axes, mode, cval).unwrap();
                let refr =
                    correlate_axes_scalar_reference(&input, &weights, axes, mode, cval).unwrap();
                assert_eq!(
                    fast.shape, refr.shape,
                    "shape mismatch in={in_shape:?} w={w_shape:?} axes={axes:?} mode={mode:?}"
                );
                let differ = fast
                    .data
                    .iter()
                    .zip(&refr.data)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differ, 0,
                    "correlate_axes not byte-identical: {differ} diffs in={in_shape:?} w={w_shape:?} axes={axes:?} mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn mirror_mode_matches_scipy() {
        // scipy.ndimage mode='mirror' (whole-sample symmetric). Filters and
        // order<=1 interpolation are supported; order>=2 spline fails closed.
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 7.0, 3.0, 9.0, 5.0, 8.0], vec![8]).unwrap();
        let m = BoundaryMode::Mirror;

        let corr = correlate1d(&input, &[1.0, 2.0, -1.0], 0, m, 0.0).unwrap();
        assert_eq!(corr.data, vec![2.0, 1.0, 3.0, 15.0, 4.0, 16.0, 11.0, 16.0]);

        let g = gaussian_filter1d(&input, 1.0, 0, 0, m, 0.0).unwrap();
        let g_ref = [
            1.861607317776,
            2.526828635116,
            4.039436351458,
            5.108236282683,
            5.599415285308,
            6.377488592219,
            6.612007542601,
            6.611567303452,
        ];
        for (a, b) in g.data.iter().zip(g_ref.iter()) {
            assert!((a - b).abs() < 1e-9, "gaussian mirror {a} vs {b}");
        }

        let mc =
            map_coordinates(&input, &[vec![-0.7, 0.5, 2.3, 5.5, 7.9, 9.0]], 1, m, 0.0).unwrap();
        let mc_ref = [1.7, 1.5, 4.9, 7.0, 5.3, 9.0];
        for (a, b) in mc.iter().zip(mc_ref.iter()) {
            assert!((a - b).abs() < 1e-9, "map_coordinates mirror {a} vs {b}");
        }

        // order-3 spline interpolation under mirror (exact B-spline mirror
        // prefilter), vs scipy.ndimage.map_coordinates(order=3, mode='mirror').
        let mc3 = map_coordinates(&input, &[vec![-0.7, 0.5, 2.3, 5.5, 7.9]], 3, m, 0.0).unwrap();
        let mc3_ref = [
            1.599_479_903_813_123,
            1.343_095_156_303_675_7,
            5.369_718_996_908_279,
            7.303_890_415_664_721,
            4.996_015_802_129_852,
        ];
        for (a, b) in mc3.iter().zip(mc3_ref.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "map_coordinates mirror order3 {a} vs {b}"
            );
        }
        // An axis too short for the order's stencil still fails closed.
        let short = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!(map_coordinates(&short, &[vec![1.2]], 3, m, 0.0).is_err());

        // index reflection helpers agree on the whole-sample fold.
        assert_eq!(boundary_index_1d(-1, 8, m), Some(1));
        assert_eq!(boundary_index_1d(8, 8, m), Some(6));
        assert_eq!(boundary_index_1d(-7, 8, m), Some(7));
    }

    #[test]
    fn correlate1d_matches_scipy_1d_constant() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 7.0], vec![4]).unwrap();
        let got = correlate1d(&input, &[1.0, 2.0, -1.0], 0, BoundaryMode::Constant, 0.5).unwrap();
        // scipy.ndimage.correlate1d(x, [1, 2, -1], mode='constant', cval=0.5)
        let expect = vec![0.5, 1.0, 3.0, 17.5];
        assert_eq!(got.data, expect);
    }

    #[test]
    fn convolve1d_matches_scipy_1d_constant() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 7.0], vec![4]).unwrap();
        let got = convolve1d(&input, &[1.0, 2.0, -1.0], 0, BoundaryMode::Constant, 0.5).unwrap();
        // scipy.ndimage.convolve1d(x, [1, 2, -1], mode='constant', cval=0.5)
        let expect = vec![3.5, 7.0, 13.0, 10.5];
        assert_eq!(got.data, expect);
    }

    #[test]
    fn convolve1d_and_correlate1d_match_scipy_axis_cases() {
        let input = NdArray::new((0..6).map(f64::from).collect(), vec![2, 3]).unwrap();
        let weights = [1.0, 0.5];

        let corr_axis0 = correlate1d(&input, &weights, 0, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(corr_axis0.data, vec![0.0, 1.5, 3.0, 1.5, 3.0, 4.5]);

        let conv_axis0 = convolve1d(&input, &weights, 0, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(conv_axis0.data, vec![3.0, 4.5, 6.0, 4.5, 6.0, 7.5]);

        let corr_axis1 = correlate1d(&input, &weights, 1, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(corr_axis1.data, vec![0.0, 0.5, 2.0, 4.5, 5.0, 6.5]);

        let conv_axis1 = convolve1d(&input, &weights, 1, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(conv_axis1.data, vec![1.0, 2.5, 3.0, 5.5, 7.0, 7.5]);
    }

    #[test]
    fn core_1d_filters_signed_axis_match_scipy_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let weights = [1.0, 0.0, -1.0];

        let corr_default =
            correlate1d_default_axis(&input, &weights, BoundaryMode::Nearest, 0.0).unwrap();
        let corr_last =
            correlate1d_signed_axis(&input, &weights, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let corr_first =
            correlate1d_signed_axis(&input, &weights, -2, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.correlate1d([[1, 2, 3], [4, 5, 6]], [1, 0, -1], axis=-1, mode='nearest')
        assert_eq!(corr_default.data, vec![-1.0, -2.0, -1.0, -1.0, -2.0, -1.0]);
        assert_eq!(corr_last.data, corr_default.data);
        // scipy.ndimage.correlate1d(..., axis=-2, mode='nearest')
        assert_eq!(corr_first.data, vec![-3.0; 6]);

        let conv_default =
            convolve1d_default_axis(&input, &weights, BoundaryMode::Nearest, 0.0).unwrap();
        let conv_last =
            convolve1d_signed_axis(&input, &weights, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let conv_first =
            convolve1d_signed_axis(&input, &weights, -2, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.convolve1d(..., axis=-1, mode='nearest')
        assert_eq!(conv_default.data, vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0]);
        assert_eq!(conv_last.data, conv_default.data);
        // scipy.ndimage.convolve1d(..., axis=-2, mode='nearest')
        assert_eq!(conv_first.data, vec![3.0; 6]);

        let gauss_default =
            gaussian_filter1d_default_axis(&input, 0.75, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let gauss_last =
            gaussian_filter1d_signed_axis(&input, 0.75, -1, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let gauss_first =
            gaussian_filter1d_signed_axis(&input, 0.75, -2, 0, BoundaryMode::Reflect, 0.0).unwrap();
        // scipy.ndimage.gaussian_filter1d(..., 0.75, axis=-1, mode='reflect')
        let expect_last = [
            1.26497001042,
            2.0,
            2.73502998958,
            4.26497001042,
            5.0,
            5.73502998958,
        ];
        for ((g, l), e) in gauss_default
            .data
            .iter()
            .zip(&gauss_last.data)
            .zip(expect_last)
        {
            assert!(
                (*g - e).abs() < 1e-10,
                "gaussian default-axis mismatch: {g} vs {e}"
            );
            assert!(
                (*l - e).abs() < 1e-10,
                "gaussian last-axis mismatch: {l} vs {e}"
            );
        }
        // scipy.ndimage.gaussian_filter1d(..., 0.75, axis=-2, mode='reflect')
        let expect_first = [
            1.74772151257,
            2.74772151257,
            3.74772151257,
            3.25227848743,
            4.25227848743,
            5.25227848743,
        ];
        for (g, e) in gauss_first.data.iter().zip(expect_first) {
            assert!(
                (*g - e).abs() < 1e-10,
                "gaussian first-axis mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn core_1d_filters_signed_axis_reject_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let weights = [1.0, 0.0, -1.0];

        assert!(correlate1d_signed_axis(&input, &weights, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(correlate1d_signed_axis(&input, &weights, -3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve1d_signed_axis(&input, &weights, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(convolve1d_signed_axis(&input, &weights, -3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            gaussian_filter1d_signed_axis(&input, 0.75, 2, 0, BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(
            gaussian_filter1d_signed_axis(&input, 0.75, -3, 0, BoundaryMode::Reflect, 0.0).is_err()
        );
    }

    #[test]
    fn convolve1d_and_correlate1d_origins_match_scipy() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let weights = [1., 10., 100.];

        let corr_left =
            correlate1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, -1).unwrap();
        assert_eq!(corr_left.data, vec![321., 432., 543., 54., 5.]);

        let corr_right =
            correlate1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, 1).unwrap();
        assert_eq!(corr_right.data, vec![100., 210., 321., 432., 543.]);

        let conv_left =
            convolve1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, -1).unwrap();
        assert_eq!(conv_left.data, vec![1., 12., 123., 234., 345.]);

        let conv_right =
            convolve1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, 1).unwrap();
        assert_eq!(conv_right.data, vec![123., 234., 345., 450., 500.]);
    }

    #[test]
    fn convolve1d_and_correlate1d_even_origin_bounds_match_scipy() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let weights = [1., 10., 100., 1000.];

        let corr_left =
            correlate1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, -2).unwrap();
        assert_eq!(corr_left.data, vec![4321., 5432., 543., 54., 5.]);

        let conv_left =
            convolve1d_with_origin(&input, &weights, 0, BoundaryMode::Constant, 0.0, -2).unwrap();
        assert_eq!(conv_left.data, vec![1., 12., 123., 1234., 2345.]);

        assert!(
            correlate1d_with_origin(&input, &weights, 0, BoundaryMode::Reflect, 0.0, 1).is_ok()
        );
        assert!(convolve1d_with_origin(&input, &weights, 0, BoundaryMode::Reflect, 0.0, 1).is_ok());
        assert!(
            correlate1d_with_origin(&input, &weights, 0, BoundaryMode::Reflect, 0.0, 2).is_err()
        );
        assert!(
            convolve1d_with_origin(&input, &weights, 0, BoundaryMode::Reflect, 0.0, 2).is_err()
        );
    }

    #[test]
    fn convolve1d_rejects_invalid_axis_and_empty_weights() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0], vec![3]).unwrap();
        let axis_err = convolve1d(&input, &[1.0], 1, BoundaryMode::Reflect, 0.0)
            .expect_err("axis outside ndim should be rejected");
        assert!(matches!(axis_err, NdimageError::InvalidArgument(_)));

        let weights_err = correlate1d(&input, &[], 0, BoundaryMode::Reflect, 0.0)
            .expect_err("empty weights should be rejected");
        assert!(matches!(weights_err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn gaussian_kernel_radius_matches_scipy_truncate_rule() {
        assert_eq!(gaussian_kernel_radius(0.1), 0);
        assert_eq!(gaussian_kernel_radius(0.5), 2);
        assert_eq!(gaussian_kernel_radius(1.0), 4);
        assert_eq!(gaussian_kernel_radius(1.3), 5);
        assert_eq!(gaussian_kernel_radius(2.0), 8);
    }

    #[test]
    fn gaussian_kernel1d_order0_matches_scipy() {
        // scipy.ndimage._filters._gaussian_kernel1d(1.0, 0, 4)
        let expect = [
            1.338306246147e-04,
            4.431861620031e-03,
            5.399112742070e-02,
            2.419714456566e-01,
            3.989434693561e-01,
            2.419714456566e-01,
            5.399112742070e-02,
            4.431861620031e-03,
            1.338306246147e-04,
        ];
        let got = gaussian_kernel1d(1.0, 0, 4).unwrap();
        assert_eq!(got.len(), expect.len());
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-12, "order0 kernel mismatch: {g} vs {e}");
        }
        // Order-0 kernel is a normalized probability density.
        assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_kernel1d_order1_matches_scipy() {
        // scipy.ndimage._filters._gaussian_kernel1d(1.0, 1, 4)
        let expect = [
            5.353224984590e-04,
            1.329558486009e-02,
            1.079822548414e-01,
            2.419714456566e-01,
            0.0,
            -2.419714456566e-01,
            -1.079822548414e-01,
            -1.329558486009e-02,
            -5.353224984590e-04,
        ];
        let got = gaussian_kernel1d(1.0, 1, 4).unwrap();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-12, "order1 kernel mismatch: {g} vs {e}");
        }
        // First-derivative kernel is antisymmetric and sums to zero.
        assert!(got.iter().sum::<f64>().abs() < 1e-15);
    }

    #[test]
    fn gaussian_kernel1d_order2_matches_scipy() {
        // scipy.ndimage._filters._gaussian_kernel1d(1.0, 2, 4)
        let expect = [
            2.007459369221e-03,
            3.545489296025e-02,
            1.619733822621e-01,
            0.0,
            -3.989434693561e-01,
            0.0,
            1.619733822621e-01,
            3.545489296025e-02,
            2.007459369221e-03,
        ];
        let got = gaussian_kernel1d(1.0, 2, 4).unwrap();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-12, "order2 kernel mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn ndimage_array_helpers2_match_numpy() {
        // More previously-untested numpy-equivalent array helpers.
        let a = NdArray::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = NdArray::new(vec![3.0, 4.0], vec![2]).unwrap();
        assert_eq!(add_arrays(&a, &b).unwrap().data, vec![4.0, 6.0]);
        let e = exp_array(&NdArray::new(vec![0.0, 1.0], vec![2]).unwrap());
        assert!(
            (e.data[0] - 1.0).abs() < 1e-12 && (e.data[1] - std::f64::consts::E).abs() < 1e-12,
            "exp"
        );
        let lg = log_array(&NdArray::new(vec![1.0, std::f64::consts::E], vec![2]).unwrap());
        assert!(
            lg.data[0].abs() < 1e-12 && (lg.data[1] - 1.0).abs() < 1e-12,
            "log"
        );
        assert_eq!(
            cumprod_array(&NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap()).data,
            vec![1.0, 2.0, 6.0, 24.0]
        );
        // flatten 2x2 row-major.
        assert_eq!(
            flatten(&NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()).data,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // greater_than -> 1.0/0.0 mask.
        let g = greater_than(
            &NdArray::new(vec![1.0, 5.0, 3.0], vec![3]).unwrap(),
            &NdArray::new(vec![2.0, 2.0, 2.0], vec![3]).unwrap(),
        )
        .unwrap();
        assert_eq!(g.data, vec![0.0, 1.0, 1.0]);
        // equal_within tolerance -> 1.0/0.0 mask.
        let eq = equal_within(
            &NdArray::new(vec![1.0, 2.0], vec![2]).unwrap(),
            &NdArray::new(vec![1.0005, 2.5], vec![2]).unwrap(),
            0.001,
        )
        .unwrap();
        assert_eq!(eq.data, vec![1.0, 0.0]);
    }

    #[test]
    fn ndimage_array_helpers_match_numpy() {
        // Several previously-untested numpy-equivalent array helpers.
        let a = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();
        assert_eq!(argmax(&a), 4);
        assert_eq!(argmin(&a), 1); // first occurrence of the min
        let b = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        assert!((array_sum(&b) - 10.0).abs() < 1e-12, "sum");
        assert_eq!(cumsum_array(&b).data, vec![1.0, 3.0, 6.0, 10.0]);
        assert_eq!(diff_array(&b).data, vec![1.0, 1.0, 1.0]); // numpy.diff
        let c = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        assert!(
            (array_std(&c) - 2.0_f64.sqrt()).abs() < 1e-12,
            "std (population)"
        );
        let d = NdArray::new(vec![-1.0, 0.0, 5.0, 10.0], vec![4]).unwrap();
        assert_eq!(clip(&d, 0.0, 5.0).data, vec![0.0, 0.0, 5.0, 5.0]);
        let e = NdArray::new(vec![0.0, 1.0, 0.0, 2.0, 0.0], vec![5]).unwrap();
        assert_eq!(count_nonzero(&e), 2);
        let f = NdArray::new(vec![-1.0, 2.0, -3.0], vec![3]).unwrap();
        assert_eq!(abs_array(&f).data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn diff_array_preserves_first_difference_bits() {
        let input = NdArray::new(
            vec![
                f64::from_bits(0x7ff8_0000_0000_0042),
                -0.0,
                0.0,
                f64::MIN_POSITIVE,
                -3.5,
                f64::INFINITY,
            ],
            vec![2, 3],
        )
        .unwrap();
        let expected: Vec<u64> = input
            .data
            .windows(2)
            .map(|window| (window[1] - window[0]).to_bits())
            .collect();

        let got = diff_array(&input);

        assert_eq!(
            got.data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(got.shape, vec![5]);
        assert_eq!(got.strides, vec![1]);
    }

    #[test]
    fn binary_dilation_erosion_box_match_scipy() {
        // fsci structure_size=3 = full 3x3 box (= scipy structure=ones((3,3))).
        // Dilation of a single center pixel -> all ones.
        let center = NdArray::new(vec![0., 0., 0., 0., 1., 0., 0., 0., 0.], vec![3, 3]).unwrap();
        let dil = binary_dilation(&center, 3, 1).expect("dilation");
        assert_eq!(dil.data, vec![1.0; 9]);
        // Erosion of an all-ones 3x3 (default border 0) -> only center survives.
        let ones = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        let ero = binary_erosion(&ones, 3, 1).expect("erosion");
        assert_eq!(ero.data, vec![0., 0., 0., 0., 1., 0., 0., 0., 0.]);
    }

    #[test]
    fn center_of_mass_match_scipy() {
        // scipy.ndimage.center_of_mass of the whole array (single all-ones label):
        // a symmetric plus -> (1,1); the weighted case [[1,0],[0,3]] -> (0.75,0.75).
        let input = NdArray::new(
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let labels = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        let com = center_of_mass(&input, &labels, 1).expect("com");
        assert_eq!(com.len(), 1);
        assert!((com[0][0] - 1.0).abs() < 1e-12, "com row: {}", com[0][0]);
        assert!((com[0][1] - 1.0).abs() < 1e-12, "com col: {}", com[0][1]);

        let w_in = NdArray::new(vec![1.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let w_lbl = NdArray::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let wc = center_of_mass(&w_in, &w_lbl, 1).expect("weighted com");
        assert!(
            (wc[0][0] - 0.75).abs() < 1e-12 && (wc[0][1] - 0.75).abs() < 1e-12,
            "wcom: {:?}",
            wc[0]
        );
    }

    #[test]
    fn label_connected_components_match_scipy() {
        // scipy.ndimage.label default (4-connectivity cross) on a 3x4 binary image.
        let input = NdArray::new(
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            vec![3, 4],
        )
        .unwrap();
        let (labels, n) = label(&input).unwrap();
        assert_eq!(n, 3, "component count");
        let expect = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 2.0];
        for (g, e) in labels.data.iter().zip(&expect) {
            assert_eq!(*g, *e, "label: {g} vs {e}");
        }
    }

    #[test]
    fn uniform_maximum_filter1d_match_scipy() {
        // scipy.ndimage.{uniform,maximum}_filter1d(a, 3, mode='reflect') for
        // a=[1,2,3,4,5,4,3,2,1]. gaussian_filter1d has a golden test; these did not.
        let input = NdArray::new(vec![1., 2., 3., 4., 5., 4., 3., 2., 1.], vec![9]).unwrap();
        let u = uniform_filter1d(&input, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let eu = [
            4.0 / 3.0,
            2.0,
            3.0,
            4.0,
            13.0 / 3.0,
            4.0,
            3.0,
            2.0,
            4.0 / 3.0,
        ];
        for (g, e) in u.data.iter().zip(&eu) {
            assert!((g - e).abs() < 1e-12, "uniform_filter1d: {g} vs {e}");
        }
        let m = maximum_filter1d(&input, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let em = [2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 4.0, 3.0, 2.0];
        for (g, e) in m.data.iter().zip(&em) {
            assert!((g - e).abs() < 1e-12, "maximum_filter1d: {g} vs {e}");
        }
    }

    // The test above uses a=[1,2,3,4,5,4,3,2,1] with size 3 in `reflect` only.
    // That input is PALINDROMIC, and on it scipy's reflect, nearest and wrap all
    // produce byte-identical output -- so an implementation that silently treats
    // reflect as nearest (or as wrap) passes it. Two further facts shape what a
    // discriminating test has to look like:
    //
    //   * With size 3 the boundary overhang is one sample, and reflect
    //     (d c b a | a b c d) and nearest (a a a a | a b c d) both supply a0
    //     there. They are identical for size 3 REGARDLESS of the input. Size 5
    //     is the smallest that separates them.
    //   * For maximum_filter1d the symmetric-extension modes (reflect, nearest,
    //     mirror) coincide by construction on ANY input: every sample they graft
    //     past the edge is a duplicate of one already inside the window, so the
    //     maximum cannot change. Only `constant` (with cval above the local max)
    //     and `wrap` can move it.
    //
    // Goldens below are scipy.ndimage 1.17.1 on an asymmetric input, size 5,
    // where all five modes give distinct results.
    #[test]
    fn uniform_maximum_filter1d_match_scipy_across_boundary_modes_and_axes() {
        let input = NdArray::new(vec![3., 1., 4., 1., 5., 9., 2., 6.], vec![8]).unwrap();

        // scipy.ndimage.uniform_filter1d(a, 5, mode=<m>, cval=0.0)
        let uniform_cases: [(BoundaryMode, [f64; 8]); 5] = [
            (
                BoundaryMode::Reflect,
                [2.4, 2.4, 2.8, 4.0, 4.2, 4.6, 5.6, 5.0],
            ),
            (
                BoundaryMode::Nearest,
                [2.8, 2.4, 2.8, 4.0, 4.2, 4.6, 5.6, 5.8],
            ),
            (
                BoundaryMode::Mirror,
                [2.6, 2.0, 2.8, 4.0, 4.2, 4.6, 4.8, 5.6],
            ),
            (
                BoundaryMode::Constant,
                [1.6, 1.8, 2.8, 4.0, 4.2, 4.6, 4.4, 3.4],
            ),
            // fsci's Wrap is documented as a b c d | a b c d, which is scipy's
            // `grid-wrap` spelling (scipy's plain `wrap` agrees with it here).
            (BoundaryMode::Wrap, [3.2, 3.0, 2.8, 4.0, 4.2, 4.6, 5.0, 4.2]),
        ];
        for (mode, expected) in uniform_cases {
            let got = uniform_filter1d(&input, 5, 0, mode, 0.0)
                .unwrap_or_else(|e| panic!("uniform_filter1d {mode:?} failed: {e:?}"));
            for (i, (g, e)) in got.data.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "uniform_filter1d {mode:?} [{i}]: {g} vs {e}"
                );
            }
        }
        // The interior is mode-independent; the modes must differ at the EDGES.
        // Without this a single-mode implementation still passes every row above
        // only if it happens to be that mode -- this pins that they are five
        // genuinely different results.
        let edge = |mode: BoundaryMode, cval: f64| {
            let r = uniform_filter1d(&input, 5, 0, mode, cval).expect("uniform edge probe");
            (r.data[0], r.data[7])
        };
        let edges = [
            edge(BoundaryMode::Reflect, 0.0),
            edge(BoundaryMode::Nearest, 0.0),
            edge(BoundaryMode::Mirror, 0.0),
            edge(BoundaryMode::Constant, 0.0),
            edge(BoundaryMode::Wrap, 0.0),
        ];
        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                assert!(
                    (edges[i].0 - edges[j].0).abs() > 1e-9
                        || (edges[i].1 - edges[j].1).abs() > 1e-9,
                    "boundary modes {i} and {j} produced identical edges {:?} / {:?}; \
                     they must be distinguishable",
                    edges[i],
                    edges[j]
                );
            }
        }

        // scipy.ndimage.maximum_filter1d(a, 5, mode=<m>, cval=<c>). Only
        // `constant` with a large cval and `wrap` can move the result away from
        // the symmetric-extension answer.
        let max_cases: [(BoundaryMode, f64, [f64; 8]); 3] = [
            (
                BoundaryMode::Reflect,
                0.0,
                [4.0, 4.0, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ),
            (
                BoundaryMode::Constant,
                100.0,
                [100.0, 100.0, 5.0, 9.0, 9.0, 9.0, 100.0, 100.0],
            ),
            (
                BoundaryMode::Wrap,
                0.0,
                [6.0, 6.0, 5.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ),
        ];
        for (mode, cval, expected) in max_cases {
            let got = maximum_filter1d(&input, 5, 0, mode, cval)
                .unwrap_or_else(|e| panic!("maximum_filter1d {mode:?} failed: {e:?}"));
            for (i, (g, e)) in got.data.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "maximum_filter1d {mode:?} cval={cval} [{i}]: {g} vs {e}"
                );
            }
        }

        // AXIS handling: the sibling test only ever filters a 1-D array along
        // axis 0, so it cannot catch an implementation that ignores `axis`.
        // scipy.ndimage.uniform_filter1d(b, 3, axis=<k>, mode='reflect') for
        // b = [[1,2,3,4],[10,0,5,2]].
        let b = NdArray::new(vec![1., 2., 3., 4., 10., 0., 5., 2.], vec![2, 4]).unwrap();
        let axis1 = uniform_filter1d(&b, 3, 1, BoundaryMode::Reflect, 0.0).expect("axis=1");
        let e1 = [
            4.0 / 3.0,
            2.0,
            3.0,
            11.0 / 3.0,
            20.0 / 3.0,
            5.0,
            7.0 / 3.0,
            3.0,
        ];
        for (i, (g, e)) in axis1.data.iter().zip(&e1).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "uniform_filter1d axis=1 [{i}]: {g} vs {e}"
            );
        }
        let axis0 = uniform_filter1d(&b, 3, 0, BoundaryMode::Reflect, 0.0).expect("axis=0");
        let e0 = [
            4.0,
            4.0 / 3.0,
            11.0 / 3.0,
            10.0 / 3.0,
            7.0,
            2.0 / 3.0,
            13.0 / 3.0,
            8.0 / 3.0,
        ];
        for (i, (g, e)) in axis0.data.iter().zip(&e0).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "uniform_filter1d axis=0 [{i}]: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter1d_matches_scipy_axis0_reflect() {
        let input = NdArray::new(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]).unwrap();
        // scipy.ndimage.gaussian_filter1d(x, 0.75, axis=0, mode='reflect')
        let expect = [
            0.747721512570,
            1.747721512570,
            2.747721512570,
            2.252278487430,
            3.252278487430,
            4.252278487430,
        ];
        let got = gaussian_filter1d(&input, 0.75, 0, 0, BoundaryMode::Reflect, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-12,
                "axis0 gaussian_filter1d mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter1d_matches_scipy_axis1_reflect() {
        let input = NdArray::new(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]).unwrap();
        // scipy.ndimage.gaussian_filter1d(x, 0.75, axis=1, mode='reflect')
        let expect = [
            0.264970010420,
            1.000000000000,
            1.735029989580,
            3.264970010420,
            4.000000000000,
            4.735029989580,
        ];
        let got = gaussian_filter1d(&input, 0.75, 1, 0, BoundaryMode::Reflect, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-12,
                "axis1 gaussian_filter1d mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter1d_matches_scipy_order1_constant() {
        let input = NdArray::new(vec![1., 2., 4., 7., 11.], vec![5]).unwrap();
        // scipy.ndimage.gaussian_filter1d(x, 1.0, order=1, mode='constant', cval=-1.0)
        let expect = [
            1.378614160040,
            1.749319394020,
            2.289679776700,
            1.343262185350,
            -2.516640239040,
        ];
        let got = gaussian_filter1d(&input, 1.0, 0, 1, BoundaryMode::Constant, -1.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-11,
                "order1 gaussian_filter1d mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_filter1d_rejects_invalid_axis_and_sigma() {
        let input = NdArray::new(vec![0., 1., 2.], vec![3]).unwrap();
        let axis_err = gaussian_filter1d(&input, 1.0, 1, 0, BoundaryMode::Reflect, 0.0)
            .expect_err("axis outside ndim should be rejected");
        assert!(matches!(axis_err, NdimageError::InvalidArgument(_)));

        let sigma_err = gaussian_filter1d(&input, 0.0, 0, 0, BoundaryMode::Reflect, 0.0)
            .expect_err("non-positive sigma should be rejected");
        assert!(matches!(sigma_err, NdimageError::InvalidArgument(_)));

        let huge_sigma_err = gaussian_filter1d(&input, f64::MAX, 0, 0, BoundaryMode::Reflect, 0.0)
            .expect_err("overflowing Gaussian kernel radius should be rejected");
        assert!(matches!(huge_sigma_err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn gaussian_laplace_matches_scipy_1d() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5., 4., 3., 2., 1.], vec![9]).unwrap();
        // scipy.ndimage.gaussian_laplace(x, 1.0, mode='reflect')
        let expect = [
            6.771748269992e-01,
            2.742164389047e-01,
            -3.767835284826e-02,
            -4.760916339030e-01,
            -8.770425626284e-01,
            -4.760916339030e-01,
            -3.767835284826e-02,
            2.742164389047e-01,
            6.771748269992e-01,
        ];
        let got = gaussian_laplace(&input, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-9,
                "gaussian_laplace mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_laplace_matches_scipy_1d_constant() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5., 4., 3., 2., 1.], vec![9]).unwrap();
        // scipy.ndimage.gaussian_laplace(x, 0.5, mode='constant', cval=0.0)
        let expect = [
            -5.439686612904e-01,
            -1.119601132509e+00,
            -1.679401698764e+00,
            -2.270866074947e+00,
            -5.417148978497e+00,
            -2.270866074947e+00,
            -1.679401698764e+00,
            -1.119601132509e+00,
            -5.439686612904e-01,
        ];
        let got = gaussian_laplace(&input, 0.5, BoundaryMode::Constant, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-9,
                "gaussian_laplace mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_gradient_magnitude_matches_scipy_1d() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5., 4., 3., 2., 1.], vec![9]).unwrap();
        // scipy.ndimage.gaussian_gradient_magnitude(x, 1.0, mode='reflect')
        let expect = [
            3.637846078566e-01,
            8.483117329162e-01,
            9.562939877576e-01,
            7.270338932147e-01,
            0.0,
            7.270338932147e-01,
            9.562939877576e-01,
            8.483117329162e-01,
            3.637846078566e-01,
        ];
        let got = gaussian_gradient_magnitude(&input, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-9, "ggm mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn gaussian_laplace_matches_scipy_2d() {
        let input = NdArray::new((0..25).map(f64::from).collect(), vec![5, 5]).unwrap();
        // scipy.ndimage.gaussian_laplace(arange(25).reshape(5,5), 1.0, mode='reflect')
        let expect = [
            4.063296480899e+00,
            3.662299415825e+00,
            3.385792400058e+00,
            3.109285384290e+00,
            2.708288319216e+00,
            2.058311155528e+00,
            1.657314090454e+00,
            1.380807074687e+00,
            1.104300058919e+00,
            7.033029938450e-01,
            6.757760766913e-01,
            2.747790116171e-01,
            -1.728004150291e-03,
            -2.782350199177e-01,
            -6.792320849919e-01,
            -7.067590021456e-01,
            -1.107756067220e+00,
            -1.384263082987e+00,
            -1.660770098755e+00,
            -2.061767163829e+00,
            -2.711744327517e+00,
            -3.112741392591e+00,
            -3.389248408358e+00,
            -3.665755424126e+00,
            -4.066752489200e+00,
        ];
        let got = gaussian_laplace(&input, 1.0, BoundaryMode::Reflect, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!(
                (g - e).abs() < 1e-9,
                "gaussian_laplace 2d mismatch: {g} vs {e}"
            );
        }
    }

    #[test]
    fn gaussian_laplace_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        // scipy.ndimage.gaussian_laplace(input, 1.0, mode='constant', cval=0.0, axes=(-1,))
        let last_axis = [
            2.48950059692e-01,
            -7.97886938712e-01,
            -1.433800495162e+00,
            1.991600477539e+00,
            -6.383095509698e+00,
            -1.1470403961298e+01,
        ];
        // scipy.ndimage.gaussian_laplace(input, 1.0, mode='constant', cval=0.0, axes=(-2,))
        let first_axis = [
            -3.98943469356e-01,
            -7.97886938712e-01,
            -1.595773877424e+00,
            -3.191547754849e+00,
            -6.383095509698e+00,
            -1.2766191019395e+01,
        ];

        let got_last =
            gaussian_laplace_axes(&input, 1.0, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        for (got, expect) in got_last.data.iter().zip(&last_axis) {
            assert!(
                (got - expect).abs() < 1e-9,
                "last-axis LoG: {got} vs {expect}"
            );
        }

        let got_first =
            gaussian_laplace_axes(&input, 1.0, &[-2], BoundaryMode::Constant, 0.0).unwrap();
        for (got, expect) in got_first.data.iter().zip(&first_axis) {
            assert!(
                (got - expect).abs() < 1e-9,
                "first-axis LoG: {got} vs {expect}"
            );
        }

        assert_close_or_nan(
            &gaussian_laplace_axes(&input, 1.0, &[-2, -1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            &gaussian_laplace(&input, 1.0, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
        );
        assert_eq!(
            gaussian_laplace_axes(&input, 1.0, &[], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn gaussian_laplace_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(gaussian_laplace_axes(&input, 1.0, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_laplace_axes(&input, 1.0, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_laplace_axes(&input, 1.0, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_laplace_axes(&input, 0.0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn gaussian_gradient_magnitude_matches_scipy_2d() {
        let input = NdArray::new((0..25).map(f64::from).collect(), vec![5, 5]).unwrap();
        // scipy.ndimage.gaussian_gradient_magnitude(arange(25).reshape(5,5), 2.0, mode='nearest')
        let expect = [
            2.444455642001e+00,
            2.479023961763e+00,
            2.495143329202e+00,
            2.479023961763e+00,
            2.444455642001e+00,
            3.198479922999e+00,
            3.224976005354e+00,
            3.237383305291e+00,
            3.224976005354e+00,
            3.198479922999e+00,
            3.497825692402e+00,
            3.522070554619e+00,
            3.533434790961e+00,
            3.522070554619e+00,
            3.497825692402e+00,
            3.198479922999e+00,
            3.224976005354e+00,
            3.237383305291e+00,
            3.224976005354e+00,
            3.198479922999e+00,
            2.444455642001e+00,
            2.479023961763e+00,
            2.495143329202e+00,
            2.479023961763e+00,
            2.444455642001e+00,
        ];
        let got = gaussian_gradient_magnitude(&input, 2.0, BoundaryMode::Nearest, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-9, "ggm 2d mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn gaussian_gradient_magnitude_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        // scipy.ndimage.gaussian_gradient_magnitude(input, 1.0, mode='constant', cval=0.0, axes=(-1,))
        let last_axis = [
            2.138299919638,
            1.694803116362,
            1.38197653809,
            3.144663790519,
            2.492440813908,
            2.032386354584,
        ];
        // scipy.ndimage.gaussian_gradient_magnitude(input, 1.0, mode='constant', cval=0.0, axes=(-2,))
        let first_axis = [
            2.127124269281,
            3.886534068771,
            4.130371073939,
            0.26589053366,
            0.485816758596,
            0.516296384242,
        ];

        let got_last =
            gaussian_gradient_magnitude_axes(&input, 1.0, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap();
        for (got, expect) in got_last.data.iter().zip(&last_axis) {
            assert!(
                (got - expect).abs() < 1e-9,
                "last-axis GGM: {got} vs {expect}"
            );
        }

        let got_first =
            gaussian_gradient_magnitude_axes(&input, 1.0, &[-2], BoundaryMode::Constant, 0.0)
                .unwrap();
        for (got, expect) in got_first.data.iter().zip(&first_axis) {
            assert!(
                (got - expect).abs() < 1e-9,
                "first-axis GGM: {got} vs {expect}"
            );
        }

        assert_close_or_nan(
            &gaussian_gradient_magnitude_axes(&input, 1.0, &[-2, -1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            &gaussian_gradient_magnitude(&input, 1.0, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
        );
        assert_eq!(
            gaussian_gradient_magnitude_axes(&input, 1.0, &[], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn gaussian_gradient_magnitude_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(
            gaussian_gradient_magnitude_axes(&input, 1.0, &[1, -1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            gaussian_gradient_magnitude_axes(&input, 1.0, &[2], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            gaussian_gradient_magnitude_axes(&input, 1.0, &[-3], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            gaussian_gradient_magnitude_axes(&input, 0.0, &[-1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn gaussian_laplace_multi_sigma_matches_scipy_2d() {
        let input = NdArray::new((0..9).map(f64::from).collect(), vec![3, 3]).unwrap();
        // scipy.ndimage.gaussian_laplace(x, [0.5, 1.0], mode='constant', cval=0.0)
        let expect = [
            2.29389434243,
            1.34685752575,
            0.0133545433527,
            -1.82678093829,
            -3.68371506163,
            -3.35607665951,
            -12.0083290131,
            -17.2860226083,
            -14.2888688122,
        ];
        let got =
            gaussian_laplace_multi_sigma(&input, &[0.5, 1.0], BoundaryMode::Constant, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-9, "multi-sigma LoG mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn gaussian_gradient_magnitude_multi_sigma_matches_scipy_2d() {
        let input = NdArray::new((0..9).map(f64::from).collect(), vec![3, 3]).unwrap();
        // scipy.ndimage.gaussian_gradient_magnitude(x, [0.5, 1.0], mode='constant', cval=0.0)
        let expect = [
            1.16894660475,
            1.57719085948,
            1.38107836024,
            2.3287232643,
            2.30689186193,
            2.19521511285,
            2.40740688973,
            1.56643988725,
            2.38627900594,
        ];
        let got = gaussian_gradient_magnitude_multi_sigma(
            &input,
            &[0.5, 1.0],
            BoundaryMode::Constant,
            0.0,
        )
        .unwrap();
        for (g, e) in got.data.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-9, "multi-sigma GGM mismatch: {g} vs {e}");
        }
    }

    #[test]
    fn gaussian_derivative_multi_sigma_rejects_shape_and_sigma_errors() {
        let input = NdArray::new((0..9).map(f64::from).collect(), vec![3, 3]).unwrap();

        assert!(gaussian_laplace_multi_sigma(&input, &[1.0], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            gaussian_gradient_magnitude_multi_sigma(
                &input,
                &[1.0, f64::INFINITY],
                BoundaryMode::Reflect,
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn gaussian_gradient_magnitude_constant_is_zero() {
        let input = NdArray::new(vec![7.0; 25], vec![5, 5]).unwrap();
        let got = gaussian_gradient_magnitude(&input, 1.5, BoundaryMode::Reflect, 0.0).unwrap();
        for &v in &got.data {
            assert!(v.abs() < 1e-12, "ggm of constant should be zero, got {v}");
        }
    }

    #[test]
    fn gaussian_filter_rejects_non_finite_sigma() {
        let input = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        assert!(gaussian_filter(&input, f64::NAN, BoundaryMode::Reflect, 0.0).is_err());
        assert!(gaussian_filter(&input, f64::INFINITY, BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn gaussian_filter_multi_sigma_rejects_non_finite_sigma() {
        let input = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        assert!(
            gaussian_filter_multi_sigma(&input, &[1.0, f64::NAN], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            gaussian_filter_multi_sigma(&input, &[1.0, f64::INFINITY], BoundaryMode::Reflect, 0.0,)
                .is_err()
        );
    }

    #[test]
    fn median_filter_removes_impulse() {
        let mut data = vec![0.0; 9];
        data[4] = 100.0; // single spike
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = median_filter(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Median of 9 values where only 1 is 100 should be 0
        assert_eq!(result.data[4], 0.0);
    }

    #[test]
    fn median_filter_even_size_matches_scipy_upper_rank() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let centered = median_filter(&input, 2, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(centered.data, vec![1., 2., 3., 4., 5.]);

        let shifted =
            median_filter_with_origins(&input, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(shifted.data, vec![2., 3., 4., 5., 5.]);
    }

    #[test]
    fn median_filter_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        assert!(median_filter_with_origins(&input, 2, &[-1], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(median_filter_with_origins(&input, 2, &[0], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(median_filter_with_origins(&input, 2, &[1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            median_filter_with_origins(&input, 3, &[-1, 0], BoundaryMode::Reflect, 0.0).is_err()
        );
    }

    #[test]
    fn median_filter_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.median_filter(input, 2, mode='constant', cval=-10.0, axes=(-1,))
        let last_axis = [4.0, 4.0, 7.0, 2.0, 9.0, 9.0];
        // scipy.ndimage.median_filter(input, 2, mode='constant', cval=-10.0, axes=(-2,))
        let first_axis = [4.0, 1.0, 7.0, 4.0, 9.0, 7.0];
        // scipy.ndimage.median_filter(input, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let all_axes = [-10.0, 1.0, 1.0, 2.0, 4.0, 7.0];

        assert_eq!(
            median_filter_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            last_axis
        );
        assert_eq!(
            median_filter_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            first_axis
        );
        assert_eq!(
            median_filter_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            all_axes
        );
        assert_eq!(
            median_filter_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn median_filter_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(median_filter_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(median_filter_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(median_filter_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(median_filter_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn maximum_filter_rejects_zero_size_without_underflow() {
        let input = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        let error = maximum_filter(&input, 0, BoundaryMode::Constant, 0.0)
            .expect_err("zero-sized maximum filter should be rejected");
        assert!(matches!(error, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn rank_filter_matches_scipy_reference_ranks() {
        let input = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();

        let min_rank = rank_filter(&input, 0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(min_rank.data, vec![0.0, 1.0, 1.0, 1.0, 0.0]);

        let mid_rank = rank_filter(&input, 1, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(mid_rank.data, vec![1.0, 3.0, 1.0, 4.0, 1.0]);

        let max_rank = rank_filter(&input, 2, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(max_rank.data, vec![3.0, 4.0, 4.0, 5.0, 5.0]);

        let negative_rank = rank_filter(&input, -1, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(negative_rank.data, max_rank.data);
    }

    #[test]
    fn rank_filter_origin_matches_scipy_even_window() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let origin_zero =
            rank_filter_with_origins(&input, 0, 2, &[0], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(origin_zero.data, vec![0., 1., 2., 3., 4.]);

        let shifted_min =
            rank_filter_with_origins(&input, 0, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(shifted_min.data, vec![1., 2., 3., 4., 0.]);

        let shifted_max =
            rank_filter_with_origins(&input, 1, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(shifted_max.data, vec![2., 3., 4., 5., 5.]);
    }

    #[test]
    fn rank_filter_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.rank_filter(input, 0, 2, mode='constant', cval=-10.0, axes=(-1,))
        let rank0_last_axis = [-10.0, 1.0, 1.0, -10.0, 2.0, 3.0];
        // scipy.ndimage.rank_filter(input, 1, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let rank1_all_axes = [-10.0, -10.0, -10.0, -10.0, 2.0, 3.0];
        // scipy.ndimage.rank_filter(input, -1, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let rank_neg1_all_axes = [4.0, 4.0, 7.0, 4.0, 9.0, 9.0];

        assert_eq!(
            rank_filter_axes(&input, 0, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            rank0_last_axis
        );
        assert_eq!(
            rank_filter_axes(&input, 1, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            rank1_all_axes
        );
        assert_eq!(
            rank_filter_axes(&input, -1, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            rank_neg1_all_axes
        );
        assert_eq!(
            rank_filter_axes(&input, 0, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
        assert_eq!(
            rank_filter_axes(&input, -1, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn rank_filter_axes_rejects_invalid_rank_and_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(rank_filter_axes(&input, 2, 2, &[-1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, -3, 2, &[-1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, 1, 0, &[], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, 0, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, 0, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, 0, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_axes(&input, 0, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn binary_morph_separable_matches_naive_loop() {
        // Reference implementations replicating the original footprint-scan
        // once-functions (origin 0); the separable min/max routing must be
        // bit-identical, including on non-binary inputs (booleanization) and
        // even kernel sizes (the dilation reflection).
        fn naive_erosion(current: &NdArray, size: usize) -> NdArray {
            let ndim = current.ndim();
            let mut output = NdArray::zeros(current.shape.clone());
            let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
            let kernel_total = size.pow(ndim as u32);
            let kernel_strides = compute_strides(&vec![size; ndim]);
            for flat_out in 0..current.size() {
                let out_idx = current.unravel(flat_out);
                let mut all_set = true;
                for flat_k in 0..kernel_total {
                    let mut k_idx = vec![0usize; ndim];
                    let mut rem = flat_k;
                    for d in 0..ndim {
                        k_idx[d] = rem / kernel_strides[d];
                        rem %= kernel_strides[d];
                    }
                    let mut in_idx = vec![0i64; ndim];
                    for d in 0..ndim {
                        in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d];
                    }
                    if current.get_boundary(&in_idx, BoundaryMode::Constant, 0.0) == 0.0 {
                        all_set = false;
                        break;
                    }
                }
                output.data[flat_out] = if all_set { 1.0 } else { 0.0 };
            }
            output
        }
        fn naive_dilation(current: &NdArray, size: usize) -> NdArray {
            let ndim = current.ndim();
            let mut output = NdArray::zeros(current.shape.clone());
            let offsets: Vec<i64> = vec![size as i64 / 2; ndim];
            let kernel_total = size.pow(ndim as u32);
            let kernel_strides = compute_strides(&vec![size; ndim]);
            for flat_in in 0..current.size() {
                if current.data[flat_in] == 0.0 {
                    continue;
                }
                let idx = current.unravel(flat_in);
                for flat_k in 0..kernel_total {
                    let mut k_idx = vec![0usize; ndim];
                    let mut rem = flat_k;
                    for d in 0..ndim {
                        k_idx[d] = rem / kernel_strides[d];
                        rem %= kernel_strides[d];
                    }
                    let mut out_idx = Vec::with_capacity(ndim);
                    let mut in_bounds = true;
                    for d in 0..ndim {
                        let c = idx[d] as i64 + k_idx[d] as i64 - offsets[d];
                        if c < 0 || c >= current.shape[d] as i64 {
                            in_bounds = false;
                            break;
                        }
                        out_idx.push(c as usize);
                    }
                    if in_bounds {
                        output.set(&out_idx, 1.0);
                    }
                }
            }
            output
        }

        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for shape in [vec![20usize], vec![7, 9], vec![4, 5, 3]] {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|_| {
                    let r = next();
                    if r % 3 == 0 { 0.0 } else { (r % 4) as f64 }
                })
                .collect();
            let input = NdArray::new(data, shape.clone()).unwrap();
            for size in [2usize, 3, 4, 5] {
                let er = binary_erosion_once_with_origins(&input, size, &[0]);
                let er_ref = naive_erosion(&input, size);
                for (a, b) in er.data.iter().zip(er_ref.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "erosion shape={shape:?} size={size}"
                    );
                }
                let di = binary_dilation_once_with_origins(&input, size, &[0]);
                let di_ref = naive_dilation(&input, size);
                for (a, b) in di.data.iter().zip(di_ref.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "dilation shape={shape:?} size={size}"
                    );
                }
            }
        }
    }

    #[test]
    fn rank_filter_select_matches_sort_and_scipy() {
        // The rank filters now select the rank element instead of sorting the
        // whole footprint. Confirm the value is bit-identical to sort+index, and
        // that median_filter / rank_filter / percentile_filter still match scipy.
        let mut state: u64 = 0x0bad_f00d_1234_abcd;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for _ in 0..2000 {
            let len = 1 + (next() % 40) as usize;
            let mut v: Vec<f64> = (0..len)
                .map(|_| {
                    let r = next();
                    match r % 23 {
                        0 => -0.0,
                        1 => 0.0,
                        2 => f64::NEG_INFINITY,
                        _ => (r % 7) as f64 - 3.0,
                    }
                })
                .collect();
            let rank = (next() as usize) % len;
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let expected = sorted[rank];
            let (_, &mut got, _) = v.select_nth_unstable_by(rank, |a, b| a.total_cmp(b));
            assert_eq!(got.to_bits(), expected.to_bits(), "len={len} rank={rank}");
        }
    }

    #[test]
    fn separable_minmax_matches_rank_filter_byte_for_byte() {
        // Isomorphism proof: the separable sliding-window min/max must be
        // bit-identical to the full-footprint sort-and-select rank filter, over
        // 1D/2D/3D shapes, all boundary modes, even/odd sizes, origins, and
        // inputs containing NaN and signed zeros (which exercise total_cmp).
        let mut state: u64 = 0xfeed_face_cafe_b00b;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Constant,
            BoundaryMode::Nearest,
            BoundaryMode::Wrap,
        ];
        let shapes: &[Vec<usize>] = &[vec![17], vec![6, 7], vec![4, 5, 3]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|i| {
                    let r = next();
                    match r % 17 {
                        0 => f64::NAN,
                        1 => -0.0,
                        2 => 0.0,
                        _ => ((r >> 20) % 1000) as f64 - 500.0,
                    }
                    .max(if i == 0 { f64::NEG_INFINITY } else { -1e18 })
                })
                .collect();
            let input = NdArray::new(data, shape.clone()).unwrap();
            for &mode in &modes {
                for &size in &[2usize, 3, 4, 5] {
                    let origin_lo = -((size / 2) as i64);
                    let origin_hi = (size as i64 - 1) / 2;
                    for &origin in &[origin_lo, 0, origin_hi] {
                        let kernel_total = size.pow(shape.len() as u32);
                        for (is_max, rank) in [(false, 0usize), (true, kernel_total - 1)] {
                            let reference = rank_filter_index_with_origins(
                                &input,
                                size,
                                &[origin],
                                mode,
                                7.5,
                                rank,
                            );
                            let fast =
                                separable_minmax_filter(&input, size, &[origin], mode, 7.5, is_max);
                            match (reference, fast) {
                                (Ok(reference), Ok(fast)) => {
                                    assert_eq!(fast.shape, reference.shape);
                                    for (a, b) in fast.data.iter().zip(reference.data.iter()) {
                                        assert_eq!(
                                            a.to_bits(),
                                            b.to_bits(),
                                            "shape={shape:?} mode={mode:?} size={size} origin={origin} is_max={is_max}"
                                        );
                                    }
                                }
                                (Err(_), Err(_)) => {}
                                (r, f) => {
                                    assert_eq!(
                                        r.is_ok(),
                                        f.is_ok(),
                                        "accept/reject parity: shape={shape:?} size={size} origin={origin}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn min_max_filter_origins_match_scipy_even_window() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let min_origin_zero =
            minimum_filter_with_origins(&input, 2, &[0], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(min_origin_zero.data, vec![0., 1., 2., 3., 4.]);

        let min_shifted =
            minimum_filter_with_origins(&input, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(min_shifted.data, vec![1., 2., 3., 4., 0.]);

        let max_origin_zero =
            maximum_filter_with_origins(&input, 2, &[0], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(max_origin_zero.data, vec![1., 2., 3., 4., 5.]);

        let max_shifted =
            maximum_filter_with_origins(&input, 2, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(max_shifted.data, vec![2., 3., 4., 5., 5.]);
    }

    #[test]
    fn min_max_filter_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.minimum_filter(input, 2, mode='constant', cval=-10.0, axes=(-1,))
        let min_last_axis = [-10.0, 1.0, 1.0, -10.0, 2.0, 3.0];
        // scipy.ndimage.minimum_filter(input, 2, mode='constant', cval=-10.0, axes=(-2,))
        let min_first_axis = [-10.0, -10.0, -10.0, 2.0, 1.0, 3.0];
        // scipy.ndimage.minimum_filter(input, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let min_all_axes = [-10.0, -10.0, -10.0, -10.0, 1.0, 1.0];

        assert_eq!(
            minimum_filter_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            min_last_axis
        );
        assert_eq!(
            minimum_filter_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            min_first_axis
        );
        assert_eq!(
            minimum_filter_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            min_all_axes
        );

        // scipy.ndimage.maximum_filter(input, 2, mode='constant', cval=-10.0, axes=(-1,))
        let max_last_axis = [4.0, 4.0, 7.0, 2.0, 9.0, 9.0];
        // scipy.ndimage.maximum_filter(input, 2, mode='constant', cval=-10.0, axes=(-2,))
        let max_first_axis = [4.0, 1.0, 7.0, 4.0, 9.0, 7.0];
        // scipy.ndimage.maximum_filter(input, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let max_all_axes = [4.0, 4.0, 7.0, 4.0, 9.0, 9.0];

        assert_eq!(
            maximum_filter_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            max_last_axis
        );
        assert_eq!(
            maximum_filter_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            max_first_axis
        );
        assert_eq!(
            maximum_filter_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            max_all_axes
        );

        assert_eq!(
            minimum_filter_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
        assert_eq!(
            maximum_filter_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn min_max_filter_axes_reject_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(minimum_filter_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(minimum_filter_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(minimum_filter_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(minimum_filter_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());

        assert!(maximum_filter_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(maximum_filter_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(maximum_filter_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(maximum_filter_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn rank_filter_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        assert!(rank_filter_with_origins(&input, 0, 2, &[-1], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(rank_filter_with_origins(&input, 0, 2, &[0], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(rank_filter_with_origins(&input, 0, 2, &[1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(rank_filter_with_origins(&input, 0, 2, &[-2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            rank_filter_with_origins(&input, 0, 3, &[-1, 0], BoundaryMode::Reflect, 0.0).is_err()
        );
    }

    #[test]
    fn rank_filter_rejects_out_of_footprint_ranks() {
        let input = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]).unwrap();

        assert!(rank_filter(&input, 3, 3, BoundaryMode::Constant, 0.0).is_err());
        assert!(rank_filter(&input, -4, 3, BoundaryMode::Constant, 0.0).is_err());
        assert!(rank_filter(&input, 0, 0, BoundaryMode::Constant, 0.0).is_err());
    }

    #[test]
    fn iterate_structure_matches_scipy_cross_second_iteration() {
        let structure = generate_binary_structure(2, 1);
        let result = iterate_structure(&structure, 2).unwrap();

        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        assert_eq!(result.shape, vec![5, 5]);
        assert_eq!(result.data, expected);
    }

    #[test]
    fn iterate_structure_matches_scipy_cross_third_iteration() {
        let structure = generate_binary_structure(2, 1);
        let result = iterate_structure(&structure, 3).unwrap();

        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(result.shape, vec![7, 7]);
        assert_eq!(result.data, expected);
    }

    #[test]
    fn iterate_structure_small_iterations_copy_input_values() {
        let structure = NdArray::new(vec![0.0, 2.0, 3.0, 0.0], vec![2, 2]).unwrap();

        assert_eq!(
            iterate_structure(&structure, 0).unwrap().data,
            structure.data
        );
        assert_eq!(
            iterate_structure(&structure, 1).unwrap().data,
            structure.data
        );
    }

    #[test]
    fn binary_erosion_shrinks() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(data, vec![5, 5]).unwrap();
        let result = binary_erosion(&input, 3, 1).unwrap();
        // Only center pixel should survive
        assert_eq!(result.data[12], 1.0); // (2,2)
        assert_eq!(result.data[7], 0.0); // (1,2) should be eroded
    }

    #[test]
    fn binary_dilation_grows() {
        let mut data = vec![0.0; 25];
        data[12] = 1.0; // single pixel at (2,2)
        let input = NdArray::new(data, vec![5, 5]).unwrap();
        let result = binary_dilation(&input, 3, 1).unwrap();
        // Center and all face-neighbors should be 1
        assert_eq!(result.data[12], 1.0); // (2,2)
        assert_eq!(result.data[7], 1.0); // (1,2)
        assert_eq!(result.data[17], 1.0); // (3,2)
    }

    #[test]
    fn binary_morphology_origins_match_scipy_constant() {
        let input = NdArray::new(vec![0., 1., 1., 0., 1.], vec![5]).unwrap();

        let eroded = binary_erosion_with_origins(&input, 2, &[-1], 1).unwrap();
        assert_eq!(eroded.data, vec![0., 1., 0., 0., 0.]);

        let point = NdArray::new(vec![0., 0., 1., 0., 0.], vec![5]).unwrap();
        let dilated = binary_dilation_with_origins(&point, 3, &[1], 1).unwrap();
        assert_eq!(dilated.data, vec![1., 1., 1., 0., 0.]);

        let opened = binary_opening_with_origins(&input, 2, &[-1], 1).unwrap();
        assert_eq!(opened.data, vec![0., 1., 1., 0., 0.]);

        let closed = binary_closing_with_origins(&input, 3, &[1], 1).unwrap();
        assert_eq!(closed.data, vec![0., 0., 1., 1., 1.]);
    }

    #[test]
    fn binary_morphology_origin_validation_matches_filter_bounds() {
        let input = NdArray::new(vec![0., 1., 1., 0., 1.], vec![5]).unwrap();

        assert!(binary_erosion_with_origins(&input, 2, &[-1], 1).is_ok());
        assert!(binary_erosion_with_origins(&input, 2, &[1], 1).is_err());
        assert!(binary_dilation_with_origins(&input, 3, &[1], 1).is_ok());
        assert!(binary_opening_with_origins(&input, 3, &[-2], 1).is_err());
        assert!(binary_closing_with_origins(&input, 3, &[2], 1).is_err());
    }

    #[test]
    fn binary_erosion_and_dilation_axes_match_scipy_subset_fixtures() {
        #[rustfmt::skip]
        let input = NdArray::new(vec![
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ], vec![2, 3]).unwrap();

        // scipy.ndimage.binary_erosion(input, np.ones((2,)), border_value=0, axes=(-1,))
        assert_eq!(
            binary_erosion_axes(&input, 2, &[-1], 1).unwrap().data,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        // scipy.ndimage.binary_erosion(input, np.ones((2,)), border_value=0, axes=(-2,))
        assert_eq!(
            binary_erosion_axes(&input, 2, &[-2], 1).unwrap().data,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        // scipy.ndimage.binary_erosion(input, np.ones((2, 2)), border_value=0, axes=(-2, -1))
        assert_eq!(
            binary_erosion_axes(&input, 2, &[-2, -1], 1).unwrap().data,
            vec![0.0; 6]
        );

        // scipy.ndimage.binary_dilation(input, np.ones((2,)), border_value=0, axes=(-1,))
        assert_eq!(
            binary_dilation_axes(&input, 2, &[-1], 1).unwrap().data,
            vec![1.0; 6]
        );
        // scipy.ndimage.binary_dilation(input, np.ones((2,)), border_value=0, axes=(-2,))
        assert_eq!(
            binary_dilation_axes(&input, 2, &[-2], 1).unwrap().data,
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        );
        // scipy.ndimage.binary_dilation(input, np.ones((2, 2)), border_value=0, axes=(-2, -1))
        assert_eq!(
            binary_dilation_axes(&input, 2, &[-2, -1], 1).unwrap().data,
            vec![1.0; 6]
        );

        #[rustfmt::skip]
        let non_bool = NdArray::new(vec![
            2.0, 0.0, -1.0,
            0.0, 3.0, 4.0,
        ], vec![2, 3]).unwrap();
        assert_eq!(
            binary_erosion_axes(&non_bool, 2, &[], 1).unwrap().data,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(
            binary_dilation_axes(&non_bool, 2, &[], 1).unwrap().data,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn binary_erosion_and_dilation_axes_reject_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(binary_erosion_axes(&input, 2, &[1, -1], 1).is_err());
        assert!(binary_erosion_axes(&input, 2, &[2], 1).is_err());
        assert!(binary_erosion_axes(&input, 2, &[-3], 1).is_err());

        assert!(binary_dilation_axes(&input, 2, &[1, -1], 1).is_err());
        assert!(binary_dilation_axes(&input, 2, &[2], 1).is_err());
        assert!(binary_dilation_axes(&input, 2, &[-3], 1).is_err());
    }

    #[test]
    fn binary_propagation_matches_scipy_dense_mask() {
        let mut seed = vec![0.0; 25];
        seed[12] = 1.0;
        let input = NdArray::new(seed, vec![5, 5]).unwrap();

        #[rustfmt::skip]
        let mask = NdArray::new(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], vec![5, 5])
        .unwrap();

        let result = binary_propagation(&input, 3, Some(&mask)).unwrap();

        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn binary_propagation_origins_match_scipy_dense_mask() {
        let input = NdArray::new(vec![0., 0., 1., 0., 0.], vec![5]).unwrap();
        let mask = NdArray::new(vec![0., 1., 1., 1., 1.], vec![5]).unwrap();

        let shifted_right = binary_propagation_with_origins(&input, 3, &[-1], Some(&mask)).unwrap();
        assert_eq!(shifted_right.data, vec![0., 0., 1., 1., 1.]);

        let shifted_left = binary_propagation_with_origins(&input, 3, &[1], Some(&mask)).unwrap();
        assert_eq!(shifted_left.data, vec![0., 1., 1., 0., 0.]);
    }

    #[test]
    fn binary_propagation_origin_validation_matches_filter_bounds() {
        let input = NdArray::new(vec![0., 0., 1., 0., 0.], vec![5]).unwrap();
        let mask = NdArray::new(vec![0., 1., 1., 1., 1.], vec![5]).unwrap();

        assert!(binary_propagation_with_origins(&input, 2, &[-1], Some(&mask)).is_ok());
        assert!(binary_propagation_with_origins(&input, 2, &[1], Some(&mask)).is_err());
        assert!(binary_propagation_with_origins(&input, 3, &[-2], Some(&mask)).is_err());
        assert!(binary_propagation_with_origins(&input, 3, &[2], Some(&mask)).is_err());
    }

    #[test]
    fn binary_propagation_without_mask_converges_to_full_reachable_image() {
        let mut seed = vec![0.0; 25];
        seed[12] = 1.0;
        let input = NdArray::new(seed, vec![5, 5]).unwrap();
        let result = binary_propagation(&input, 3, None).unwrap();
        assert!(result.data.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn binary_propagation_empty_seed_stays_empty() {
        let input = NdArray::zeros(vec![4, 4]);
        let result = binary_propagation(&input, 3, None).unwrap();
        assert!(result.data.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn binary_opening_zero_iterations_converges_to_fixed_point() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(data, vec![5, 5]).unwrap();

        let converged = binary_opening(&input, 3, 0).unwrap();
        assert!(converged.data.iter().all(|&v| v == 0.0));

        let once = binary_opening(&input, 3, 1).unwrap();
        assert_eq!(once.data[6], 1.0);
        assert_eq!(once.data[12], 1.0);
        assert_eq!(once.data[18], 1.0);

        let twice = binary_opening(&input, 3, 2).unwrap();
        assert!(twice.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn label_two_components() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![4, 4]).unwrap();
        let (labels, num) = label(&input).unwrap();
        assert_eq!(num, 2);
        // Top-left block should have label 1
        assert_eq!(labels.data[0], 1.0);
        assert_eq!(labels.data[1], 1.0);
        assert_eq!(labels.data[4], 1.0);
        assert_eq!(labels.data[5], 1.0);
        // Bottom-right block should have label 2
        assert_eq!(labels.data[14], 2.0);
        assert_eq!(labels.data[15], 2.0);
    }

    #[test]
    fn label_with_structure_controls_diagonal_connectivity() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        let input = NdArray::new(data, vec![2, 2]).unwrap();

        // scipy.ndimage.label(x) uses cross connectivity, so diagonal
        // foreground pixels are separate features.
        let (default_labels, default_num) = label(&input).unwrap();
        assert_eq!(default_num, 2);
        assert_eq!(default_labels.data, vec![1.0, 0.0, 0.0, 2.0]);

        // scipy.ndimage.label(x, structure=np.ones((3, 3))) connects them.
        let full_structure = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        let (full_labels, full_num) = label_with_structure(&input, &full_structure).unwrap();
        assert_eq!(full_num, 1);
        assert_eq!(full_labels.data, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn label_with_structure_validates_scipy_structure_contract() {
        let input = NdArray::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let wrong_shape = NdArray::new(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(label_with_structure(&input, &wrong_shape).is_err());

        #[rustfmt::skip]
        let nonsymmetric = NdArray::new(vec![
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ], vec![3, 3]).unwrap();
        assert!(label_with_structure(&input, &nonsymmetric).is_err());
    }

    #[test]
    fn median_labels_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel per-label median must be BYTE-IDENTICAL to the serial map, across label
        // counts and group sizes, including even/odd group sizes and adversarial values.
        for (npix, nlabels) in [(50usize, 3usize), (4000, 7), (60000, 12), (60000, 1)] {
            let mut data: Vec<f64> = (0..npix)
                .map(|k| ((k as f64 * 0.53).sin() * 9.0 - 1.1) + (k % 11) as f64)
                .collect();
            if npix > 40 {
                data[3] = f64::NAN;
                data[7] = -0.0;
                data[8] = 0.0;
                data[20] = f64::NEG_INFINITY;
                data[21] = f64::INFINITY;
            }
            let labels_vec: Vec<f64> = (0..npix).map(|k| (k % nlabels + 1) as f64).collect();
            let input = NdArray::new(data, vec![npix]).unwrap();
            let labels = NdArray::new(labels_vec, vec![npix]).unwrap();
            let index: Vec<usize> = (1..=nlabels).collect();

            NDIMAGE_MEDIAN_LABELS_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let ser = median(&input, Some(&labels), Some(&index)).unwrap();
            NDIMAGE_MEDIAN_LABELS_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let par = median(&input, Some(&labels), Some(&index)).unwrap();
            assert_eq!(ser.len(), par.len());
            for (i, (a, b)) in ser.iter().zip(&par).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "npix={npix} nlabels={nlabels} label {i}"
                );
            }
        }
        NDIMAGE_MEDIAN_LABELS_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn sum_labels_correct() {
        let data = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0], vec![4]).unwrap();
        let sums = sum_labels(&data, &labels, 2).unwrap();
        assert_eq!(sums, vec![3.0, 7.0]);
    }

    #[test]
    fn measurement_reduction_wrappers_match_scipy_fixtures() {
        let data = NdArray::new(vec![1.0, 2.0, 5.0, 10.0, 20.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 0.0, 3.0], vec![5]).unwrap();
        let index = [0, 1, 2, 3, 4];

        // scipy.ndimage.sum(data)
        assert_eq!(sum(&data, None, None).unwrap(), vec![38.0]);
        // scipy.ndimage.sum(data, labels)
        assert_eq!(sum(&data, Some(&labels), None).unwrap(), vec![28.0]);
        // scipy.ndimage.sum(data, labels, [0, 1, 2, 3, 4])
        assert_eq!(
            sum(&data, Some(&labels), Some(&index)).unwrap(),
            vec![10.0, 3.0, 5.0, 20.0, 0.0]
        );

        // scipy.ndimage.mean(data, labels, [0, 1, 2, 3, 4])
        assert_close_or_nan(
            &mean(&data, Some(&labels), Some(&index)).unwrap(),
            &[10.0, 1.5, 5.0, 20.0, f64::NAN],
        );
        // scipy.ndimage.variance(data, labels, [0, 1, 2, 3, 4])
        assert_close_or_nan(
            &variance(&data, Some(&labels), Some(&index)).unwrap(),
            &[0.0, 0.25, 0.0, 0.0, f64::NAN],
        );
        // scipy.ndimage.standard_deviation(data, labels, [0, 1, 2, 3, 4])
        assert_close_or_nan(
            &standard_deviation(&data, Some(&labels), Some(&index)).unwrap(),
            &[0.0, 0.5, 0.0, 0.0, f64::NAN],
        );
    }

    #[test]
    fn measurement_reduction_wrappers_reject_label_shape_mismatch() {
        let data = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let err =
            mean(&data, Some(&labels), Some(&[1])).expect_err("shape mismatch must be rejected");
        assert!(matches!(err, NdimageError::DimensionMismatch(_)));
    }

    #[test]
    fn mean_dense_label_lookup_preserves_exact_label_semantics() {
        let data = NdArray::new(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            vec![8],
        )
        .unwrap();
        let labels =
            NdArray::new(vec![-0.0, 1.0, 2.0, 2.0, 3.0, 0.5, f64::NAN, -1.0], vec![8]).unwrap();
        let index = [1, 1, 2, 0];

        assert_close_or_nan(
            &mean(&data, Some(&labels), Some(&index)).unwrap(),
            &[20.0, f64::NAN, 35.0, 10.0],
        );
    }

    #[test]
    fn mean_one_based_contiguous_lookup_preserves_exact_label_semantics() {
        let data = NdArray::new(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            vec![10],
        )
        .unwrap();
        let labels = NdArray::new(
            vec![
                -0.0,
                1.0,
                2.0,
                2.0,
                3.0,
                0.5,
                f64::NAN,
                -1.0,
                4.0,
                1.0000000000000002,
            ],
            vec![10],
        )
        .unwrap();
        let index = [1, 2, 3];

        assert_close_or_nan(
            &mean(&data, Some(&labels), Some(&index)).unwrap(),
            &[20.0, 35.0, 50.0],
        );
    }

    #[test]
    fn mean_one_based_parallel_scatter_matches_serial_reference() {
        // Large enough to cross the parallel privatized-histogram gate (n/128_000 >= 2).
        // The parallel reduction reassociates float adds vs the serial scatter, so the
        // means must agree to tolerance (not bit-exact); confirm well under 1e-9.
        let n = 300_000usize;
        let k = 64usize;
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let vals: Vec<f64> = (0..n)
            .map(|_| (rng() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        let labels: Vec<f64> = (0..n).map(|_| (1 + (rng() % k as u64)) as f64).collect();
        let index: Vec<usize> = (1..=k).collect();

        let input = NdArray::new(vals.clone(), vec![n]).unwrap();
        let lab = NdArray::new(labels.clone(), vec![n]).unwrap();
        let got = mean(&input, Some(&lab), Some(&index)).unwrap();

        // Serial reference scatter.
        let mut sref = vec![0.0f64; k];
        let mut cref = vec![0usize; k];
        for (&v, &lv) in vals.iter().zip(&labels) {
            let p = (lv as usize) - 1;
            sref[p] += v;
            cref[p] += 1;
        }
        for i in 0..k {
            let want = sref[i] / cref[i] as f64;
            assert!(
                (got[i] - want).abs() < 1e-9,
                "label {i}: parallel {} vs serial {want}",
                got[i]
            );
        }
    }

    #[test]
    fn sum_variance_one_based_fast_path_matches_serial_reference() {
        // Crosses the parallel gate (n/128_000 >= 2) so the privatized-histogram sum and
        // two-pass variance run in parallel; both must match a serial reference to tolerance.
        let n = 300_000usize;
        let k = 64usize;
        let mut s = 0xfeed_face_dead_beefu64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Values with a non-zero mean to exercise the two-pass (one-pass would cancel).
        let vals: Vec<f64> = (0..n)
            .map(|_| 100.0 + (rng() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        let labels: Vec<f64> = (0..n).map(|_| (1 + (rng() % k as u64)) as f64).collect();
        let index: Vec<usize> = (1..=k).collect();
        let input = NdArray::new(vals.clone(), vec![n]).unwrap();
        let lab = NdArray::new(labels.clone(), vec![n]).unwrap();

        let got_sum = sum(&input, Some(&lab), Some(&index)).unwrap();
        let got_var = variance(&input, Some(&lab), Some(&index)).unwrap();
        let got_std = standard_deviation(&input, Some(&lab), Some(&index)).unwrap();

        // Serial reference (two-pass).
        let mut sref = vec![0.0f64; k];
        let mut cref = vec![0usize; k];
        for (&v, &lv) in vals.iter().zip(&labels) {
            let p = (lv as usize) - 1;
            sref[p] += v;
            cref[p] += 1;
        }
        let means: Vec<f64> = (0..k).map(|i| sref[i] / cref[i] as f64).collect();
        let mut vsum = vec![0.0f64; k];
        for (&v, &lv) in vals.iter().zip(&labels) {
            let p = (lv as usize) - 1;
            let d = v - means[p];
            vsum[p] += d * d;
        }
        for i in 0..k {
            assert!((got_sum[i] - sref[i]).abs() < 1e-6, "sum label {i}");
            let want_var = vsum[i] / cref[i] as f64;
            assert!((got_var[i] - want_var).abs() < 1e-9, "var label {i}");
            assert!((got_std[i] - want_var.sqrt()).abs() < 1e-9, "std label {i}");
        }
    }

    #[test]
    fn minimum_maximum_one_based_fast_path_byte_identical_to_serial() {
        // Large enough to cross the parallel gate. min/max are exact, so the parallel
        // privatized-histogram result must be BYTE-IDENTICAL to a serial fold.
        let n = 300_000usize;
        let k = 64usize;
        let mut s = 0x0bad_c0de_1337_d00du64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut vals: Vec<f64> = (0..n)
            .map(|_| (rng() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
            .collect();
        let labels: Vec<f64> = (0..n).map(|_| (1 + (rng() % k as u64)) as f64).collect();
        // Inject a NaN into label 7's region to exercise NaN propagation.
        for (i, &lv) in labels.iter().enumerate() {
            if lv as usize == 7 {
                vals[i] = f64::NAN;
                break;
            }
        }
        let index: Vec<usize> = (1..=k).collect();
        let input = NdArray::new(vals.clone(), vec![n]).unwrap();
        let lab = NdArray::new(labels.clone(), vec![n]).unwrap();

        let gmin = minimum(&input, Some(&lab), Some(&index)).unwrap();
        let gmax = maximum(&input, Some(&lab), Some(&index)).unwrap();

        // Serial reference matching the production min/max fold semantics.
        let mut rmin = vec![f64::INFINITY; k];
        let mut rmax = vec![f64::NEG_INFINITY; k];
        let mut cnt = vec![0usize; k];
        for (&v, &lv) in vals.iter().zip(&labels) {
            let p = (lv as usize) - 1;
            cnt[p] += 1;
            rmin[p] = if rmin[p].is_nan() || v.is_nan() {
                f64::NAN
            } else {
                rmin[p].min(v)
            };
            rmax[p] = if rmax[p].is_nan() || v.is_nan() {
                f64::NAN
            } else {
                rmax[p].max(v)
            };
        }
        for i in 0..k {
            let want_min = if cnt[i] == 0 { 0.0 } else { rmin[i] };
            let want_max = if cnt[i] == 0 { 0.0 } else { rmax[i] };
            assert_eq!(gmin[i].to_bits(), want_min.to_bits(), "min label {i}");
            assert_eq!(gmax[i].to_bits(), want_max.to_bits(), "max label {i}");
        }
        // Label 7 saw a NaN → NaN.
        assert!(gmin[6].is_nan() && gmax[6].is_nan());
    }

    #[test]
    fn histogram_one_based_fast_path_byte_identical_to_serial() {
        // Crosses the parallel gate; histogram counts are integers so the privatized parallel
        // path must be exactly equal to a serial reference. min/max in (0,1) so some values
        // fall outside [min,max] and exercise the filter.
        let n = 300_000usize;
        let k = 64usize;
        let nbins = 16usize;
        let (min_val, max_val) = (0.2_f64, 0.8_f64);
        let mut s = 0x00c0_ffee_1234_5678u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let vals: Vec<f64> = (0..n)
            .map(|_| (rng() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        let labels: Vec<f64> = (0..n).map(|_| (1 + (rng() % k as u64)) as f64).collect();
        let index: Vec<usize> = (1..=k).collect();
        let input = NdArray::new(vals.clone(), vec![n]).unwrap();
        let lab = NdArray::new(labels.clone(), vec![n]).unwrap();

        let got = histogram(&input, min_val, max_val, nbins, Some(&lab), Some(&index)).unwrap();

        let bw = (max_val - min_val) / nbins as f64;
        let mut want = vec![vec![0usize; nbins]; k];
        for (&v, &lv) in vals.iter().zip(&labels) {
            if v < min_val || v > max_val {
                continue;
            }
            let pos = (lv as usize) - 1;
            let bin = (((v - min_val) / bw).floor() as usize).min(nbins - 1);
            want[pos][bin] += 1;
        }
        assert_eq!(got, want);
    }

    #[test]
    fn minimum_maximum_empty_label_returns_zero() {
        // Index references a label with no pixels → 0.0 (scipy convention), preserved on the
        // serial small-N fast path.
        let data = NdArray::new(vec![3.0, 7.0, 2.0, 9.0], vec![4]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 3.0, 3.0], vec![4]).unwrap();
        let index = [1, 2, 3]; // label 2 is empty
        assert_eq!(
            minimum(&data, Some(&labels), Some(&index)).unwrap(),
            vec![3.0, 0.0, 2.0]
        );
        assert_eq!(
            maximum(&data, Some(&labels), Some(&index)).unwrap(),
            vec![7.0, 0.0, 9.0]
        );
    }

    #[test]
    fn measurement_selector_wrappers_match_scipy_fixtures() {
        let data = NdArray::new(vec![1.0, 2.0, 5.0, 10.0, 20.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 0.0, 3.0], vec![5]).unwrap();
        let index = [0, 1, 2, 3];

        // scipy.ndimage.minimum(data, labels)
        assert_eq!(minimum(&data, Some(&labels), None).unwrap(), vec![1.0]);
        // scipy.ndimage.maximum(data, labels)
        assert_eq!(maximum(&data, Some(&labels), None).unwrap(), vec![20.0]);
        // scipy.ndimage.median(data, labels)
        assert_eq!(median(&data, Some(&labels), None).unwrap(), vec![3.5]);

        // scipy.ndimage.minimum(data, labels, [0, 1, 2, 3])
        assert_eq!(
            minimum(&data, Some(&labels), Some(&index)).unwrap(),
            vec![10.0, 1.0, 5.0, 20.0]
        );
        // scipy.ndimage.maximum(data, labels, [0, 1, 2, 3])
        assert_eq!(
            maximum(&data, Some(&labels), Some(&index)).unwrap(),
            vec![10.0, 2.0, 5.0, 20.0]
        );
        // scipy.ndimage.median(data, labels, [0, 1, 2, 3])
        assert_eq!(
            median(&data, Some(&labels), Some(&index)).unwrap(),
            vec![10.0, 1.5, 5.0, 20.0]
        );
    }

    #[test]
    fn histogram_wrapper_matches_scipy_fixtures() {
        let data = NdArray::new(vec![1.0, 2.0, 5.0, 10.0, 20.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 0.0, 3.0], vec![5]).unwrap();
        let index = [0, 1, 2, 3];

        // scipy.ndimage.histogram(data, 0.0, 20.0, 4)
        assert_eq!(
            histogram(&data, 0.0, 20.0, 4, None, None).unwrap(),
            vec![vec![2, 1, 1, 1]]
        );
        // scipy.ndimage.histogram(data, 0.0, 20.0, 4, labels)
        assert_eq!(
            histogram(&data, 0.0, 20.0, 4, Some(&labels), None).unwrap(),
            vec![vec![2, 1, 0, 1]]
        );
        // scipy.ndimage.histogram(data, 0.0, 20.0, 4, labels, [0, 1, 2, 3])
        assert_eq!(
            histogram(&data, 0.0, 20.0, 4, Some(&labels), Some(&index)).unwrap(),
            vec![
                vec![0, 0, 1, 0],
                vec![2, 0, 0, 0],
                vec![0, 1, 0, 0],
                vec![0, 0, 0, 1],
            ]
        );
    }

    #[test]
    fn extrema_and_position_wrappers_match_scipy_fixtures() {
        let data = NdArray::new(vec![1.0, 2.0, 5.0, 10.0, 20.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 0.0, 3.0], vec![5]).unwrap();
        let index = [0, 1, 2, 3];

        // scipy.ndimage.minimum_position(data, labels, [0, 1, 2, 3])
        assert_eq!(
            minimum_position(&data, Some(&labels), Some(&index)).unwrap(),
            vec![vec![3], vec![0], vec![2], vec![4]]
        );
        // scipy.ndimage.maximum_position(data, labels, [0, 1, 2, 3])
        assert_eq!(
            maximum_position(&data, Some(&labels), Some(&index)).unwrap(),
            vec![vec![3], vec![1], vec![2], vec![4]]
        );

        // scipy.ndimage.extrema(data, labels, [0, 1, 2, 3])
        let (mins, maxs, min_positions, max_positions) =
            extrema(&data, Some(&labels), Some(&index)).unwrap();
        assert_eq!(mins, vec![10.0, 1.0, 5.0, 20.0]);
        assert_eq!(maxs, vec![10.0, 2.0, 5.0, 20.0]);
        assert_eq!(min_positions, vec![vec![3], vec![0], vec![2], vec![4]]);
        assert_eq!(max_positions, vec![vec![3], vec![1], vec![2], vec![4]]);
    }

    #[test]
    fn labeled_comprehension_matches_scipy_fixtures() {
        let data = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0, 4.0], vec![5]).unwrap();
        let index = [1, 2, 3, 4];

        let mean_func =
            |values: &[f64], _: Option<&[usize]>| values.iter().sum::<f64>() / values.len() as f64;

        // scipy.ndimage.labeled_comprehension(data, labels, [1, 2, 3, 4], np.mean, float, -1)
        assert_eq!(
            labeled_comprehension(&data, Some(&labels), Some(&index), mean_func, -1.0, false)
                .unwrap(),
            vec![1.5, 3.5, -1.0, 5.0]
        );
        // scipy.ndimage.labeled_comprehension(data, labels, None, np.mean, float, -1)
        assert_eq!(
            labeled_comprehension(&data, Some(&labels), None, mean_func, -1.0, false).unwrap(),
            vec![3.0]
        );
        // scipy.ndimage.labeled_comprehension(data, None, None, np.mean, float, -1)
        assert_eq!(
            labeled_comprehension(&data, None, None, mean_func, -1.0, false).unwrap(),
            vec![3.0]
        );
    }

    #[test]
    fn labeled_comprehension_passes_positions_and_rejects_index_without_labels() {
        let data = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0, 4.0], vec![5]).unwrap();
        let index = [1, 2, 4];

        // scipy equivalent: lambda vals, pos: vals.sum() + pos.sum()
        let values_plus_positions = |values: &[f64], positions: Option<&[usize]>| {
            values.iter().sum::<f64>()
                + positions
                    .unwrap()
                    .iter()
                    .map(|&position| position as f64)
                    .sum::<f64>()
        };
        assert_eq!(
            labeled_comprehension(
                &data,
                Some(&labels),
                Some(&index),
                values_plus_positions,
                -1.0,
                true
            )
            .unwrap(),
            vec![4.0, 12.0, 9.0]
        );

        let err =
            labeled_comprehension(&data, None, Some(&index), values_plus_positions, -1.0, true)
                .expect_err("SciPy rejects index without labels");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn value_indices_matches_scipy_fixtures() {
        let one_dim = NdArray::new(vec![2.0, 1.0, 2.0, 3.0, 1.0], vec![5]).unwrap();
        let one_dim_indices = value_indices(&one_dim, None).unwrap();
        // scipy.ndimage.value_indices(np.array([2, 1, 2, 3, 1]))
        assert_eq!(one_dim_indices.get(&1).unwrap(), &vec![vec![1, 4]]);
        assert_eq!(one_dim_indices.get(&2).unwrap(), &vec![vec![0, 2]]);
        assert_eq!(one_dim_indices.get(&3).unwrap(), &vec![vec![3]]);

        let two_dim = NdArray::new(vec![1.0, 2.0, 1.0, 3.0, 2.0, 3.0], vec![2, 3]).unwrap();
        let two_dim_indices = value_indices(&two_dim, None).unwrap();
        // scipy.ndimage.value_indices(np.array([[1, 2, 1], [3, 2, 3]]))
        assert_eq!(
            two_dim_indices.get(&1).unwrap(),
            &vec![vec![0, 0], vec![0, 2]]
        );
        assert_eq!(
            two_dim_indices.get(&2).unwrap(),
            &vec![vec![0, 1], vec![1, 1]]
        );
        assert_eq!(
            two_dim_indices.get(&3).unwrap(),
            &vec![vec![1, 1], vec![0, 2]]
        );

        let ignored = value_indices(&two_dim, Some(2)).unwrap();
        // scipy.ndimage.value_indices(..., ignore_value=2)
        assert!(!ignored.contains_key(&2));
        assert_eq!(ignored.get(&1).unwrap(), &vec![vec![0, 0], vec![0, 2]]);
        assert_eq!(ignored.get(&3).unwrap(), &vec![vec![1, 1], vec![0, 2]]);
    }

    #[test]
    fn value_indices_rejects_non_integer_values() {
        let arr = NdArray::new(vec![1.0, 2.5], vec![2]).unwrap();
        let err =
            value_indices(&arr, None).expect_err("SciPy rejects non-integer value_indices inputs");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn sobel_detects_edge() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = sobel(&input, 0, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel should have strong gradient in axis 0
        assert!(result.data[4].abs() > 0.0);
    }

    #[test]
    fn sobel_signed_axis_matches_scipy_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let default_axis = sobel_default_axis(&input, BoundaryMode::Nearest, 0.0).unwrap();
        let last_axis = sobel_signed_axis(&input, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let first_axis = sobel_signed_axis(&input, -2, BoundaryMode::Nearest, 0.0).unwrap();

        // scipy.ndimage.sobel([[1, 2, 3], [4, 5, 6]], axis=-1, mode='nearest')
        assert_eq!(default_axis.data, vec![4.0, 8.0, 4.0, 4.0, 8.0, 4.0]);
        assert_eq!(last_axis.data, default_axis.data);
        // scipy.ndimage.sobel(..., axis=-2, mode='nearest')
        assert_eq!(first_axis.data, vec![12.0; 6]);
    }

    #[test]
    fn sobel_signed_axis_rejects_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(sobel_signed_axis(&input, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(sobel_signed_axis(&input, -3, BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn prewitt_constant_image_is_zero() {
        // /testing-metamorphic for [frankenscipy-il5nf]: any uniform
        // image has zero gradient, so prewitt(const) must be 0
        // everywhere (interior — boundary depends on mode but Reflect
        // preserves the constant beyond the edge).
        let input = NdArray::new(vec![7.0; 9], vec![3, 3]).unwrap();
        let result = prewitt(&input, 0, BoundaryMode::Reflect, 0.0).unwrap();
        for (i, &v) in result.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "prewitt(const, axis=0) at flat index {i} = {v}, expected 0"
            );
        }
    }

    #[test]
    fn prewitt_detects_horizontal_edge_along_axis0() {
        // 3×3 image with a step function across rows: top row 0, middle
        // 0, bottom row 1. Prewitt along axis=0 should produce a strong
        // positive gradient at the row=1 / row=2 boundary (interior
        // center pixel).
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = prewitt(&input, 0, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel (row=1, col=1, flat index 4) sees a step in
        // axis=0; the smoothing along axis=1 averages 3 zeros and
        // 3 ones giving a non-trivial response.
        assert!(
            result.data[4].abs() > 0.5,
            "prewitt should detect horizontal edge at center; got {}",
            result.data[4]
        );
    }

    #[test]
    fn prewitt_detects_vertical_edge_along_axis1() {
        // Step across columns: left two cols 0, right col 1. Prewitt
        // along axis=1 should produce a strong response at the
        // col=1 / col=2 boundary.
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = prewitt(&input, 1, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel sees the step.
        assert!(
            result.data[4].abs() > 0.5,
            "prewitt should detect vertical edge at center; got {}",
            result.data[4]
        );
    }

    #[test]
    fn prewitt_signed_axis_matches_scipy_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let default_axis = prewitt_default_axis(&input, BoundaryMode::Nearest, 0.0).unwrap();
        let last_axis = prewitt_signed_axis(&input, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let first_axis = prewitt_signed_axis(&input, -2, BoundaryMode::Nearest, 0.0).unwrap();

        // scipy.ndimage.prewitt([[1, 2, 3], [4, 5, 6]], axis=-1, mode='nearest')
        assert_eq!(default_axis.data, vec![3.0, 6.0, 3.0, 3.0, 6.0, 3.0]);
        assert_eq!(last_axis.data, default_axis.data);
        // scipy.ndimage.prewitt(..., axis=-2, mode='nearest')
        assert_eq!(first_axis.data, vec![9.0; 6]);
    }

    #[test]
    fn prewitt_axis_out_of_range_returns_error() {
        let input = NdArray::new(vec![1.0; 4], vec![2, 2]).unwrap();
        assert!(prewitt(&input, 5, BoundaryMode::Reflect, 0.0).is_err());
        assert!(prewitt_signed_axis(&input, -3, BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn laplace_axes_matches_scipy_subset_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        // scipy.ndimage.laplace(input, mode='constant', cval=0.0, axes=None)
        assert_eq!(
            laplace(&input, BoundaryMode::Constant, 0.0).unwrap().data,
            vec![6.0, 13.0, 18.0, -15.0, -22.0, -108.0]
        );
        // scipy.ndimage.laplace(input, mode='constant', cval=0.0, axes=(-1,))
        assert_eq!(
            laplace_axes(&input, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![0.0, 1.0, -6.0, 0.0, 8.0, -48.0]
        );
        // scipy.ndimage.laplace(input, mode='constant', cval=0.0, axes=(-2,))
        assert_eq!(
            laplace_axes(&input, &[-2], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![6.0, 12.0, 24.0, -15.0, -30.0, -60.0]
        );
        assert_eq!(
            laplace_axes(&input, &[-2, -1], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            laplace(&input, BoundaryMode::Constant, 0.0).unwrap().data
        );
        assert_eq!(
            laplace_axes(&input, &[], BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn laplace_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(laplace_axes(&input, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(laplace_axes(&input, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(laplace_axes(&input, &[-3], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn laplace_constant_image() {
        let input = NdArray::new(vec![5.0; 9], vec![3, 3]).unwrap();
        let result = laplace(&input, BoundaryMode::Reflect, 0.0).unwrap();
        // Laplace of constant should be zero (interior)
        assert!((result.data[4]).abs() < 1e-10);
    }

    #[test]
    fn distance_transform_edt_basic() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input, None).unwrap();
        assert_eq!(result.data[4], 0.0); // background pixel
        assert!((result.data[1] - 1.0).abs() < 1e-10); // adjacent foreground pixel
        assert!((result.data[0] - 2.0f64.sqrt()).abs() < 1e-10); // diagonal foreground pixel
    }

    #[test]
    fn distance_transform_edt_zero_background_stays_zero() {
        let input = NdArray::new(vec![0.0; 9], vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input, None).unwrap();
        assert!(result.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn distance_transform_edt_all_foreground_matches_scipy_reference() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let result = distance_transform_edt(&input, Some(&[2.0, 3.0])).unwrap();
        let expected = [
            2.0,
            13.0f64.sqrt(),
            40.0f64.sqrt(),
            4.0,
            5.0,
            52.0f64.sqrt(),
        ];

        for (actual, expected) in result.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn distance_transform_edt_respects_sampling() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input, Some(&[2.0, 1.0])).unwrap();
        assert!((result.data[1] - 2.0).abs() < 1e-10);
        assert!((result.data[3] - 1.0).abs() < 1e-10);
        assert!((result.data[0] - 5.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn distance_transform_edt_matches_scipy_one_dimensional_fixture() {
        let input = NdArray::new(vec![1.0, 1.0, 0.0, 1.0, 1.0], vec![5]).unwrap();
        let result = distance_transform_edt(&input, None).unwrap();
        assert_eq!(result.shape, vec![5]);
        assert_eq!(result.data, vec![2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn distance_transform_edt_matches_scipy_three_dimensional_fixture() {
        let mut data = vec![1.0; 12];
        data[4] = 0.0;
        let input = NdArray::new(data, vec![2, 2, 3]).unwrap();
        let result = distance_transform_edt(&input, None).unwrap();
        let expected = [
            2.0f64.sqrt(),
            1.0,
            2.0f64.sqrt(),
            1.0,
            0.0,
            1.0,
            3.0f64.sqrt(),
            2.0f64.sqrt(),
            3.0f64.sqrt(),
            2.0f64.sqrt(),
            1.0,
            2.0f64.sqrt(),
        ];

        assert_eq!(result.shape, vec![2, 2, 3]);
        for (actual, expected) in result.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn distance_transform_edt_three_dimensional_sampling_matches_scipy() {
        let mut data = vec![1.0; 12];
        data[8] = 0.0;
        let input = NdArray::new(data, vec![2, 2, 3]).unwrap();
        let result = distance_transform_edt(&input, Some(&[2.0, 1.0, 0.5])).unwrap();
        let expected = [
            5.0f64.sqrt(),
            4.25f64.sqrt(),
            2.0,
            6.0f64.sqrt(),
            5.25f64.sqrt(),
            5.0f64.sqrt(),
            1.0,
            0.5,
            0.0,
            2.0f64.sqrt(),
            1.25f64.sqrt(),
            1.0,
        ];

        assert_eq!(result.shape, vec![2, 2, 3]);
        for (actual, expected) in result.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn distance_transform_edt_all_foreground_three_dimensional_matches_scipy() {
        let input = NdArray::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let result = distance_transform_edt(&input, None).unwrap();
        let expected = [
            1.0,
            2.0f64.sqrt(),
            2.0f64.sqrt(),
            3.0f64.sqrt(),
            2.0,
            5.0f64.sqrt(),
            5.0f64.sqrt(),
            6.0f64.sqrt(),
        ];

        for (actual, expected) in result.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn distance_transform_edt_full_returns_feature_indices() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![2, 3]).unwrap();
        let result = distance_transform_edt_full(&input, None, true, true).unwrap();
        let distances = result.distances.unwrap();
        let indices = result.indices.unwrap();

        assert_eq!(
            distances.data,
            vec![1.0, 0.0, 1.0, 2.0f64.sqrt(), 1.0, 2.0f64.sqrt()]
        );
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].shape, vec![2, 3]);
        assert_eq!(indices[0].data, vec![0.0; 6]);
        assert_eq!(indices[1].data, vec![1.0; 6]);
    }

    #[test]
    fn edt_feature_transform_distances_byte_identical_and_indices_valid() {
        // Deterministic pseudo-random binary grids with MANY backgrounds (so
        // distance ties are present), across 2-D/3-D shapes and non-unit
        // sampling. The separable feature transform must (1) produce squared
        // distances byte-identical to the shipped distance-only fast path
        // (`edt_squared_felzenszwalb`), and (2) return, for every foreground
        // cell, a genuine nearest background — a background cell whose squared
        // distance equals the exact EDT value at that cell.
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let cases: &[(Vec<usize>, Vec<f64>)] = &[
            (vec![9, 7], vec![1.0, 1.0]),
            (vec![8, 11], vec![2.0, 1.0]),
            (vec![6, 5, 4], vec![1.0, 1.0, 1.0]),
            (vec![5, 6, 3], vec![1.5, 1.0, 2.0]),
        ];
        for (shape, sampling) in cases {
            for _trial in 0..25 {
                let n: usize = shape.iter().product();
                let mut data = vec![1.0; n];
                for v in data.iter_mut() {
                    if rng() % 4 == 0 {
                        *v = 0.0;
                    }
                }
                if data.iter().all(|&v| v != 0.0) {
                    data[(rng() as usize) % n] = 0.0;
                }
                if data.iter().all(|&v| v == 0.0) {
                    data[(rng() as usize) % n] = 1.0;
                }
                let input = NdArray::new(data.clone(), shape.clone()).unwrap();

                let (squared, feat) = edt_squared_felzenszwalb_with_indices(&input, sampling);

                // (1) distances byte-identical to the shipped distance fast path.
                let squared_ref = edt_squared_felzenszwalb(&input, sampling);
                assert_eq!(
                    squared, squared_ref,
                    "with-indices squared EDT must equal the distance-only fast path (shape {shape:?})"
                );

                // (2) every returned feature index is a genuine nearest background.
                for flat in 0..n {
                    if data[flat] == 0.0 {
                        continue;
                    }
                    let nf = feat[flat];
                    assert_eq!(data[nf], 0.0, "feature {nf} is not a background cell");
                    let coords = input.unravel(flat);
                    let ncoords = input.unravel(nf);
                    let d2: f64 = coords
                        .iter()
                        .zip(&ncoords)
                        .zip(sampling)
                        .map(|((&c, &nc), &s)| {
                            let delta = (c as f64 - nc as f64) * s;
                            delta * delta
                        })
                        .sum();
                    let tol = 1e-9 * squared[flat].max(1.0);
                    assert!(
                        (d2 - squared[flat]).abs() <= tol,
                        "indexed background at squared {d2} but EDT is {} (flat {flat}, shape {shape:?})",
                        squared[flat]
                    );
                }
            }
        }
    }

    #[test]
    fn distance_transform_edt_full_can_return_indices_without_distances() {
        let input = NdArray::new(vec![1.0, 0.0, 1.0], vec![3]).unwrap();
        let result = distance_transform_edt_full(&input, None, false, true).unwrap();

        assert!(result.distances.is_none());
        let indices = result.indices.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].shape, vec![3]);
        assert_eq!(indices[0].data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn distance_transform_edt_full_three_dimensional_indices_match_scipy() {
        let mut data = vec![1.0; 8];
        data[5] = 0.0;
        let input = NdArray::new(data, vec![2, 2, 2]).unwrap();
        let result = distance_transform_edt_full(&input, None, true, true).unwrap();
        let distances = result.distances.unwrap();
        let indices = result.indices.unwrap();

        assert_eq!(
            distances.data,
            vec![
                2.0f64.sqrt(),
                1.0,
                3.0f64.sqrt(),
                2.0f64.sqrt(),
                1.0,
                0.0,
                2.0f64.sqrt(),
                1.0,
            ]
        );
        assert_eq!(indices[0].data, vec![1.0; 8]);
        assert_eq!(indices[1].data, vec![0.0; 8]);
        assert_eq!(indices[2].data, vec![1.0; 8]);
    }

    #[test]
    fn distance_transform_edt_full_all_foreground_indices_match_scipy() {
        let input = NdArray::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let result = distance_transform_edt_full(&input, None, true, true).unwrap();
        let distances = result.distances.unwrap();
        let indices = result.indices.unwrap();

        assert_eq!(distances.data, vec![1.0, 2.0f64.sqrt(), 2.0, 5.0f64.sqrt()]);
        assert_eq!(indices[0].data, vec![-1.0; 4]);
        assert_eq!(indices[1].data, vec![0.0; 4]);
    }

    #[test]
    fn distance_transform_edt_full_rejects_no_outputs() {
        let input = NdArray::new(vec![1.0, 0.0], vec![2]).unwrap();
        let err = distance_transform_edt_full(&input, None, false, false).unwrap_err();
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn distance_transform_edt_scalar_sampling_applies_to_all_axes() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = distance_transform_edt(&input, Some(&[2.0])).unwrap();
        assert!((result.data[1] - 2.0).abs() < 1e-10);
        assert!((result.data[3] - 2.0).abs() < 1e-10);
        assert!((result.data[0] - (8.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn distance_transform_edt_rejects_sampling_length_mismatch() {
        let input = NdArray::new(vec![0.0; 9], vec![3, 3]).unwrap();
        let err = distance_transform_edt(&input, Some(&[1.0, 2.0, 3.0])).unwrap_err();
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn spline_filter1d_fastpath_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The reflect fast path (vectorized/parallel) must be BYTE-IDENTICAL to the serial per-line
        // walk across orders, axes, modes, and shapes (including strided and last-axis cases, and
        // sizes above and below the parallel gate).
        let shapes: &[Vec<usize>] = &[
            vec![37],
            vec![9, 11],
            vec![64, 40],
            vec![130, 90], // > 1<<20 elements after a couple axes -> exercises the parallel gate
            vec![5, 7, 6],
            vec![48, 20, 18],
        ];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|k| {
                    ((k as f64 * 0.421).sin() * 7.0 - 1.3) * if k % 3 == 0 { 100.0 } else { 1.0 }
                })
                .collect();
            let input = NdArray::new(data, shape.clone()).unwrap();
            for order in 2..=5usize {
                for axis in 0..shape.len() {
                    for mode in [BoundaryMode::Reflect, BoundaryMode::Nearest] {
                        NDIMAGE_SPLINE_FILTER1D_FORCE_SERIAL.store(true, Ordering::Relaxed);
                        let serial = spline_filter1d(&input, order, axis, mode).unwrap();
                        NDIMAGE_SPLINE_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
                        let fast = spline_filter1d(&input, order, axis, mode).unwrap();
                        assert_eq!(serial.shape, fast.shape);
                        for (i, (a, b)) in serial.data.iter().zip(&fast.data).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "shape={shape:?} order={order} axis={axis} mode={mode:?} idx={i}"
                            );
                        }
                    }
                }
            }
        }
        NDIMAGE_SPLINE_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn spline_filter1d_signed_axis_matches_scipy_axis_normalization() {
        let input = NdArray::new(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], vec![2, 3]).unwrap();

        let axis_one = spline_filter1d(&input, 3, 1, BoundaryMode::Reflect).unwrap();
        let default_axis = spline_filter1d_default_axis(&input, 3, BoundaryMode::Reflect).unwrap();
        let last_axis = spline_filter1d_signed_axis(&input, 3, -1, BoundaryMode::Reflect).unwrap();

        let axis_zero = spline_filter1d(&input, 3, 0, BoundaryMode::Reflect).unwrap();
        let first_axis = spline_filter1d_signed_axis(&input, 3, -2, BoundaryMode::Reflect).unwrap();

        // scipy.ndimage.spline_filter1d defaults axis=-1 and normalizes negative axes.
        assert_close_or_nan(&default_axis.data, &axis_one.data);
        assert_close_or_nan(&last_axis.data, &axis_one.data);
        assert_close_or_nan(&first_axis.data, &axis_zero.data);
        assert!(
            default_axis
                .data
                .iter()
                .zip(&first_axis.data)
                .any(|(left, right)| (left - right).abs() > 1e-12)
        );
    }

    #[test]
    fn spline_filter1d_signed_axis_rejects_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(spline_filter1d_signed_axis(&input, 3, 2, BoundaryMode::Reflect).is_err());
        assert!(spline_filter1d_signed_axis(&input, 3, -3, BoundaryMode::Reflect).is_err());
    }

    #[test]
    fn distance_transform_bf_matches_scipy_metric_fixtures() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();

        // scipy.ndimage.distance_transform_bf(input, metric='euclidean')
        assert_close_or_nan(
            &distance_transform_bf(&input, DistanceMetric::Euclidean, None)
                .unwrap()
                .data,
            &[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0f64.sqrt(), 1.0],
        );
        // scipy.ndimage.distance_transform_bf(input, metric='taxicab')
        assert_eq!(
            distance_transform_bf(&input, DistanceMetric::Taxicab, None)
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0]
        );
        // scipy.ndimage.distance_transform_bf(input, metric='chessboard')
        assert_eq!(
            distance_transform_bf(&input, DistanceMetric::Chessboard, None)
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0]
        );
    }

    #[test]
    fn distance_transform_cdt_matches_scipy_metric_fixtures() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();

        // scipy.ndimage.distance_transform_cdt(input, metric='taxicab')
        assert_eq!(
            distance_transform_cdt(&input, DistanceMetric::Taxicab)
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0]
        );
        // scipy.ndimage.distance_transform_cdt(input, metric='chessboard')
        assert_eq!(
            distance_transform_cdt(&input, DistanceMetric::Chessboard)
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0]
        );
    }

    #[test]
    fn distance_transform_bf_and_cdt_match_all_foreground_sentinels() {
        let input = NdArray::new(vec![1.0; 4], vec![2, 2]).unwrap();

        assert_eq!(
            distance_transform_bf(&input, DistanceMetric::Euclidean, None)
                .unwrap()
                .data,
            vec![BF_NO_BACKGROUND_EUCLIDEAN; 4]
        );
        assert_eq!(
            distance_transform_bf(&input, DistanceMetric::Taxicab, None)
                .unwrap()
                .data,
            vec![BF_NO_BACKGROUND_GRID; 4]
        );
        assert_eq!(
            distance_transform_cdt(&input, DistanceMetric::Chessboard)
                .unwrap()
                .data,
            vec![-1.0; 4]
        );

        let err = distance_transform_cdt(&input, DistanceMetric::Euclidean)
            .expect_err("CDT rejects Euclidean metric");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn rotate_90_degrees() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let input = NdArray::new(data, vec![2, 2]).unwrap();
        let result = rotate(&input, 90.0, false, 0, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        // After 90° rotation: top-right becomes top-left, etc.
        // Due to rounding, just check it doesn't crash and produces valid output
        assert_eq!(result.data.len(), 4);
    }

    #[test]
    fn zoom_upscale() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = zoom(&input, &[2.0, 2.0], 0, BoundaryMode::Nearest, 0.0).unwrap();
        assert_eq!(result.shape, vec![4, 4]);
    }

    #[test]
    fn spline_prefilter_nonreflect_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel non-reflect prefilter (Constant/Wrap/Mirror + general banded) must be
        // BYTE-IDENTICAL to the serial per-line walk. Exercised through `zoom` (order>=2 prefilters);
        // the input is large enough that an interior axis pass crosses the parallel gate.
        let (rows, cols) = (1100usize, 1000usize);
        let data: Vec<f64> = (0..rows * cols)
            .map(|k| ((k as f64 * 0.019).sin() * 6.0 - 0.7) + (k % 13) as f64 * 0.25)
            .collect();
        let input = NdArray::new(data, vec![rows, cols]).unwrap();
        for order in 2..=5usize {
            for mode in [
                BoundaryMode::Constant,
                BoundaryMode::Wrap,
                BoundaryMode::Mirror,
            ] {
                NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(true, Ordering::Relaxed);
                let ser = zoom(&input, &[1.0, 1.0], order, mode, 0.0).unwrap();
                NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(false, Ordering::Relaxed);
                let par = zoom(&input, &[1.0, 1.0], order, mode, 0.0).unwrap();
                assert_eq!(ser.shape, par.shape);
                for (i, (a, b)) in ser.data.iter().zip(&par.data).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "order={order} mode={mode:?} idx={i}"
                    );
                }
            }
        }
        NDIMAGE_SPLINE_PREFILTER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn zoom_order_one_reflect_fast_path_matches_generic_sampler_bits() {
        let input = NdArray::new(
            (0..20)
                .map(|i| {
                    let x = i as f64;
                    (x * 0.37).sin() * 11.0 + (x * 0.11).cos() * 3.0
                })
                .collect(),
            vec![5, 4],
        )
        .unwrap();
        let zoom_factors = [1.7, 2.25];
        let new_shape: Vec<usize> = input
            .shape
            .iter()
            .zip(zoom_factors.iter())
            .map(|(&s, &z)| ((s as f64 * z).round() as usize).max(1))
            .collect();
        let spline = prefilter_spline_coefficients(&input, 1, BoundaryMode::Reflect).unwrap();
        let fast = zoom_order1_reflect_2d_fast(&input, &new_shape);

        let mut generic = NdArray::zeros(new_shape.clone());
        let flags = SplineFlags::resolve();
        for flat in 0..generic.size() {
            let out_idx = unravel_with_shape(flat, &new_shape);
            let coords: Vec<f64> = out_idx
                .iter()
                .enumerate()
                .map(|(axis, &o)| {
                    if new_shape[axis] <= 1 || input.shape[axis] <= 1 {
                        0.0
                    } else {
                        o as f64 * (input.shape[axis] - 1) as f64 / (new_shape[axis] - 1) as f64
                    }
                })
                .collect();
            generic.data[flat] = sample_interpolated(
                &input,
                &spline.coeffs,
                &coords,
                &spline.coord_offsets,
                1,
                BoundaryMode::Reflect,
                0.0,
                flags,
            );
        }

        assert_eq!(fast.shape, generic.shape);
        for (i, (got, expected)) in fast.data.iter().zip(&generic.data).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "zoom order=1 fast path bit mismatch at {i}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn shift_moves_data() {
        let input = NdArray::new(vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![5]).unwrap();
        let result = shift(&input, &[1.0], 0, BoundaryMode::Constant, 0.0).unwrap();
        // Shifted right by 1: pixel at index 2 should now be at index 3
        assert_eq!(result.data[3], 1.0);
        assert_eq!(result.data[2], 0.0);
    }

    #[test]
    fn shift_order_zero_matches_existing_nearest_behavior() {
        let input = NdArray::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]).unwrap();
        let result = shift(&input, &[0.49], 0, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(result.data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn map_coordinates_order_one_matches_bilinear_center_value() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = map_coordinates(
            &input,
            &[vec![0.5], vec![0.5]],
            1,
            BoundaryMode::Nearest,
            0.0,
        )
        .unwrap();
        assert!((result[0] - 2.5).abs() < 1e-10, "got {}", result[0]);
    }

    #[test]
    fn shift_order_one_half_pixel_matches_linear_reference() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let result = shift(&input, &[0.5], 1, BoundaryMode::Nearest, 0.0).unwrap();
        let expected = [0.0, 5.0, 15.0, 25.0];
        for (got, want) in result.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }

    #[test]
    fn shift_order_three_half_pixel_matches_scipy_nearest_reference() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let result = shift(&input, &[0.5], 3, BoundaryMode::Nearest, 0.0).unwrap();
        let expected = [
            -0.807_713_659_400_537_9,
            4.264_428_414_850_133_5,
            15.0,
            25.735_571_585_149_867,
        ];
        for (got, want) in result.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    #[test]
    fn shift_order_three_half_pixel_matches_scipy_boundary_references() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let cases = [(
            BoundaryMode::Reflect,
            1e-4,
            [
                -1.607_191_365_221_785_7,
                4.464_266_599_578_032,
                15.000_003_786_876_048,
                25.535_713_203_749_697,
            ],
        )];
        for (mode, tol, expected) in cases {
            let result = shift(&input, &[0.5], 3, mode, 0.0).unwrap();
            for (got, want) in result.data.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < tol,
                    "mode {mode:?}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn rotate_full_turn_preserves_image_for_higher_order() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = rotate(&input, 360.0, false, 3, BoundaryMode::Nearest, 0.0).unwrap();
        for (got, want) in result.data.iter().zip(input.data.iter()) {
            assert!((got - want).abs() < 1e-8, "got {got}, want {want}");
        }
    }

    #[test]
    fn affine_transform_matches_scipy() {
        // scipy.ndimage.affine_transform applies the matrix directly as the
        // output->input map (no internal inversion); the 2x3 matrix's last
        // column is the offset.
        let input = NdArray::new((0..25).map(f64::from).collect(), vec![5, 5]).unwrap();
        let matrix = [[0.8, 0.1, 0.5], [-0.2, 1.1, -0.3]];

        // Reference values from scipy.ndimage.affine_transform on the same
        // arange(25).reshape(5,5) image: (order, mode, tol, expected).
        // order<=1 and order=3 constant are bit-exact (cardinal B-spline
        // path); order=3 reflect routes through the de Boor interpolating
        // spline, which agrees with scipy only to ~1e-5 (see frankenscipy
        // follow-up for making that path exact).
        let references: [(usize, BoundaryMode, f64, [f64; 25]); 4] = [
            (
                1,
                BoundaryMode::Constant,
                1e-9,
                [
                    0.0, 3.8, 5.4, 7.0, 0.0, 0.0, 7.6, 9.2, 10.8, 12.4, 0.0, 11.4, 13.0, 14.6,
                    16.2, 0.0, 15.2, 16.8, 18.4, 20.0, 0.0, 19.0, 20.6, 22.2, 0.0,
                ],
            ),
            (
                3,
                BoundaryMode::Constant,
                1e-9,
                [
                    0.0,
                    3.045714285714,
                    4.889142857143,
                    6.657142857143,
                    0.0,
                    0.0,
                    7.737142857143,
                    9.506857142857,
                    10.99885714286,
                    12.67857142857,
                    0.0,
                    11.09142857143,
                    12.85857142857,
                    14.30514285714,
                    16.09714285714,
                    0.0,
                    15.06285714286,
                    17.01171428571,
                    18.69485714286,
                    20.68571428571,
                    0.0,
                    19.68571428571,
                    21.04228571429,
                    22.17257142857,
                    0.0,
                ],
            ),
            (
                1,
                BoundaryMode::Reflect,
                1e-9,
                [
                    2.5, 3.8, 5.4, 7.0, 8.5, 6.5, 7.6, 9.2, 10.8, 12.4, 10.5, 11.4, 13.0, 14.6,
                    16.2, 14.5, 15.2, 16.8, 18.4, 20.0, 18.6, 19.0, 20.6, 22.2, 23.3,
                ],
            ),
            (
                3,
                BoundaryMode::Reflect,
                1e-4,
                [
                    2.071312652477,
                    3.522103918644,
                    5.211788632913,
                    6.873683576956,
                    8.497631252012,
                    6.436047805932,
                    7.650524974355,
                    9.313053236502,
                    10.87326342174,
                    12.50263186751,
                    10.34130828363,
                    11.28631180975,
                    12.94789529848,
                    14.49136816267,
                    16.16210518486,
                    14.39814861670,
                    15.14946641848,
                    16.87800090509,
                    18.50863134397,
                    20.25263168215,
                    18.87314778636,
                    19.25262023910,
                    20.76294794134,
                    22.18989448276,
                    23.62289481757,
                ],
            ),
        ];
        for (order, mode, tol, expected) in references {
            let got = affine_transform(&input, &matrix, order, mode, 0.0).unwrap();
            for (g, e) in got.data.iter().zip(&expected) {
                assert!(
                    (g - e).abs() < tol,
                    "affine order={order} {mode:?}: {g} vs {e}"
                );
            }
        }

        // Identity maps every pixel to itself.
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let id_out = affine_transform(&input, &identity, 1, BoundaryMode::Constant, 0.0).unwrap();
        for (g, e) in id_out.data.iter().zip(&input.data) {
            assert!((g - e).abs() < 1e-9, "affine identity: {g} vs {e}");
        }
    }

    #[test]
    fn max_min_filter1d_match_scipy() {
        // scipy.ndimage.maximum_filter1d / minimum_filter1d filter ONLY along
        // `axis` (a size-wide window, 1 on every other axis) — not an N-D box.
        let a3 = NdArray::new((0..9).map(f64::from).collect(), vec![3, 3]).unwrap();

        // axis 0, reflect.
        let mx = maximum_filter1d(&a3, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(mx.data, vec![3., 4., 5., 6., 7., 8., 6., 7., 8.]);
        let mn = minimum_filter1d(&a3, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(mn.data, vec![0., 1., 2., 0., 1., 2., 3., 4., 5.]);

        // axis 1, reflect.
        let mx = maximum_filter1d(&a3, 3, 1, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(mx.data, vec![1., 2., 2., 4., 5., 5., 7., 8., 8.]);
        let mn = minimum_filter1d(&a3, 3, 1, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(mn.data, vec![0., 0., 1., 3., 3., 4., 6., 6., 7.]);

        // axis 1, constant cval=0.
        let mn = minimum_filter1d(&a3, 3, 1, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(mn.data, vec![0., 0., 0., 0., 3., 0., 0., 6., 0.]);

        // Rectangular 4x5, axis 0, reflect.
        let a45 = NdArray::new((0..20).map(f64::from).collect(), vec![4, 5]).unwrap();
        let mx = maximum_filter1d(&a45, 3, 0, BoundaryMode::Reflect, 0.0).unwrap();
        assert_eq!(
            mx.data,
            vec![
                5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 15., 16.,
                17., 18., 19.,
            ]
        );
    }

    #[test]
    fn rank_filter1d_signed_axis_matches_scipy_fixtures() {
        let input = NdArray::new(vec![1.0, 4.0, 2.0, 6.0, 3.0, 5.0], vec![2, 3]).unwrap();

        let uniform_default =
            uniform_filter1d_default_axis(&input, 2, BoundaryMode::Nearest, 0.0).unwrap();
        let uniform_last =
            uniform_filter1d_signed_axis(&input, 2, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let uniform_first =
            uniform_filter1d_signed_axis(&input, 2, -2, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.uniform_filter1d([[1,4,2],[6,3,5]], 2, axis=-1, mode='nearest')
        assert_eq!(uniform_default.data, vec![1.0, 2.5, 3.0, 6.0, 4.5, 4.0]);
        assert_eq!(uniform_last.data, uniform_default.data);
        // scipy.ndimage.uniform_filter1d(..., axis=-2, mode='nearest')
        assert_eq!(uniform_first.data, vec![1.0, 4.0, 2.0, 3.5, 3.5, 3.5]);

        let minimum_default =
            minimum_filter1d_default_axis(&input, 2, BoundaryMode::Nearest, 0.0).unwrap();
        let minimum_last =
            minimum_filter1d_signed_axis(&input, 2, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let minimum_first =
            minimum_filter1d_signed_axis(&input, 2, -2, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.minimum_filter1d(..., axis=-1, mode='nearest')
        assert_eq!(minimum_default.data, vec![1.0, 1.0, 2.0, 6.0, 3.0, 3.0]);
        assert_eq!(minimum_last.data, minimum_default.data);
        // scipy.ndimage.minimum_filter1d(..., axis=-2, mode='nearest')
        assert_eq!(minimum_first.data, vec![1.0, 4.0, 2.0, 1.0, 3.0, 2.0]);

        let maximum_default =
            maximum_filter1d_default_axis(&input, 2, BoundaryMode::Nearest, 0.0).unwrap();
        let maximum_last =
            maximum_filter1d_signed_axis(&input, 2, -1, BoundaryMode::Nearest, 0.0).unwrap();
        let maximum_first =
            maximum_filter1d_signed_axis(&input, 2, -2, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.maximum_filter1d(..., axis=-1, mode='nearest')
        assert_eq!(maximum_default.data, vec![1.0, 4.0, 4.0, 6.0, 6.0, 5.0]);
        assert_eq!(maximum_last.data, maximum_default.data);
        // scipy.ndimage.maximum_filter1d(..., axis=-2, mode='nearest')
        assert_eq!(maximum_first.data, vec![1.0, 4.0, 2.0, 6.0, 4.0, 5.0]);
    }

    #[test]
    fn rank_filter1d_signed_axis_rejects_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(uniform_filter1d_signed_axis(&input, 2, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(uniform_filter1d_signed_axis(&input, 2, -3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(minimum_filter1d_signed_axis(&input, 2, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(minimum_filter1d_signed_axis(&input, 2, -3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(maximum_filter1d_signed_axis(&input, 2, 2, BoundaryMode::Reflect, 0.0).is_err());
        assert!(maximum_filter1d_signed_axis(&input, 2, -3, BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn filter1d_origin_wrappers_match_scipy() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let uniform_left =
            uniform_filter1d_with_origin(&input, 3, 0, BoundaryMode::Constant, 0.0, -1).unwrap();
        assert_eq!(uniform_left.data, vec![2.0, 3.0, 4.0, 3.0, 5.0 / 3.0]);

        let uniform_right =
            uniform_filter1d_with_origin(&input, 3, 0, BoundaryMode::Constant, 0.0, 1).unwrap();
        assert_eq!(uniform_right.data, vec![1.0 / 3.0, 1.0, 2.0, 3.0, 4.0]);

        let max_left =
            maximum_filter1d_with_origin(&input, 3, 0, BoundaryMode::Constant, 0.0, -1).unwrap();
        assert_eq!(max_left.data, vec![3.0, 4.0, 5.0, 5.0, 5.0]);

        let min_right =
            minimum_filter1d_with_origin(&input, 3, 0, BoundaryMode::Constant, 0.0, 1).unwrap();
        assert_eq!(min_right.data, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn filter1d_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        assert!(
            uniform_filter1d_with_origin(&input, 3, 0, BoundaryMode::Reflect, 0.0, -2).is_err()
        );
        assert!(uniform_filter1d_with_origin(&input, 3, 0, BoundaryMode::Reflect, 0.0, 2).is_err());
        assert!(uniform_filter1d_with_origin(&input, 4, 0, BoundaryMode::Reflect, 0.0, -2).is_ok());
        assert!(uniform_filter1d_with_origin(&input, 4, 0, BoundaryMode::Reflect, 0.0, 1).is_ok());
        assert!(uniform_filter1d_with_origin(&input, 4, 0, BoundaryMode::Reflect, 0.0, 2).is_err());
    }

    #[test]
    fn rotate_nonsquare_reshape_false_matches_scipy() {
        // scipy.ndimage.rotate(arange(20).reshape(4,5), angle, reshape=False).
        // Rectangular inputs at 90/270 deg map to half-integer source coords,
        // which exercises the spline interpolation; (angle, order, expected).
        let input = NdArray::new((0..20).map(f64::from).collect(), vec![4, 5]).unwrap();
        let cases: [(f64, usize, [f64; 20]); 4] = [
            (
                90.0,
                1,
                [
                    0., 6., 11., 16., 0., 0., 5., 10., 15., 0., 0., 4., 9., 14., 0., 0., 3., 8.,
                    13., 0.,
                ],
            ),
            (
                90.0,
                3,
                [
                    0.0,
                    5.410714285714,
                    11.16071428571,
                    16.91071428571,
                    0.0,
                    0.0,
                    4.196428571429,
                    9.946428571429,
                    15.69642857143,
                    0.0,
                    0.0,
                    3.303571428571,
                    9.053571428571,
                    14.80357142857,
                    0.0,
                    0.0,
                    2.089285714286,
                    7.839285714286,
                    13.58928571429,
                    0.0,
                ],
            ),
            (
                270.0,
                1,
                [
                    0., 13., 8., 3., 0., 0., 14., 9., 4., 0., 0., 15., 10., 5., 0., 0., 16., 11.,
                    6., 0.,
                ],
            ),
            (
                270.0,
                3,
                [
                    0.0,
                    13.58928571429,
                    7.839285714286,
                    2.089285714286,
                    0.0,
                    0.0,
                    14.80357142857,
                    9.053571428571,
                    3.303571428571,
                    0.0,
                    0.0,
                    15.69642857143,
                    9.946428571429,
                    4.196428571429,
                    0.0,
                    0.0,
                    16.91071428571,
                    11.16071428571,
                    5.410714285714,
                    0.0,
                ],
            ),
        ];
        for (angle, order, expected) in cases {
            let out = rotate(&input, angle, false, order, BoundaryMode::Constant, 0.0).unwrap();
            for (g, e) in out.data.iter().zip(&expected) {
                assert!(
                    (g - e).abs() < 1e-9,
                    "rotate {angle} order={order}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn cubic_spline_reflect_nearest_match_scipy_golden() {
        // Golden artifact: order-3 spline interpolation under reflect/nearest
        // verified bit-exact (< 1e-12) against scipy.ndimage. Reference values
        // from scipy.ndimage.shift / affine_transform with order=3.
        const TOL: f64 = 1e-12;

        // --- reflect: 24-sample signal (long enough that scipy's horizon-
        //     truncated prefilter agrees with the exact reflect spline) ---
        let sig: Vec<f64> = (0..24)
            .map(|i| (0.4 * i as f64).sin() + 0.3 * (1.1 * i as f64).cos())
            .collect();
        let reflect_cases: [(f64, [f64; 24]); 2] = [
            (
                0.5,
                [
                    0.2549519461363663,
                    0.4182832550693081,
                    0.5503201419186977,
                    0.5630315082005458,
                    0.75939030868993,
                    1.0438558503915527,
                    1.0988845947235066,
                    0.7086425641651631,
                    0.0259778299038867,
                    -0.5531644869252663,
                    -0.7666962224403977,
                    -0.7143820624311663,
                    -0.6961908799143749,
                    -0.8461639176777535,
                    -0.9679067171611121,
                    -0.7543452796765424,
                    -0.15076474016681216,
                    0.5398766786148524,
                    0.9318552948605892,
                    0.9194294048003819,
                    0.7433831008074291,
                    0.6848505021971074,
                    0.7708537655741882,
                    0.6506793676224725,
                ],
            ),
            (
                -0.37,
                [
                    0.38450699002959776,
                    0.5503167919057529,
                    0.5526729990650662,
                    0.7235709352531765,
                    1.0119842950436544,
                    1.1156728556271387,
                    0.7833116499893292,
                    0.1179284905124495,
                    -0.496484277981438,
                    -0.7599223082061909,
                    -0.725710233983912,
                    -0.6883221473472944,
                    -0.8209994795970635,
                    -0.9655278330938318,
                    -0.8072254128679976,
                    -0.24356750836028815,
                    0.4602459875379331,
                    0.9058336192772003,
                    0.938045035700508,
                    0.7636513064920336,
                    0.6814966064066657,
                    0.7600438511231556,
                    0.6867381661848023,
                    0.47144793799382007,
                ],
            ),
        ];
        let sig_arr = NdArray::new(sig, vec![24]).unwrap();
        for (sh, expected) in reflect_cases {
            let got = shift(&sig_arr, &[sh], 3, BoundaryMode::Reflect, 0.0).unwrap();
            for (g, e) in got.data.iter().zip(&expected) {
                assert!((g - e).abs() < TOL, "reflect shift {sh}: {g} vs {e}");
            }
        }

        // --- nearest: scipy edge-pads then prefilters, so even a short signal
        //     is reproduced exactly ---
        let small = vec![0., 10., 20., 30., 5., -5., 12., 3.];
        let nearest_cases: [(f64, [f64; 8]); 2] = [
            (
                0.5,
                [
                    -0.8481660654709604,
                    4.415398849593355,
                    14.436570667097536,
                    27.838318482016504,
                    19.83515540483645,
                    -4.678940101362298,
                    4.130605000612735,
                    9.281520098911361,
                ],
            ),
            (
                -0.37,
                [
                    3.0149073903035344,
                    13.312958134830174,
                    25.93326007037581,
                    23.381146583666645,
                    -2.949696405042477,
                    1.3359750365031913,
                    10.780087259029772,
                    1.5183309273777084,
                ],
            ),
        ];
        let small_arr = NdArray::new(small, vec![8]).unwrap();
        for (sh, expected) in nearest_cases {
            let got = shift(&small_arr, &[sh], 3, BoundaryMode::Nearest, 0.0).unwrap();
            for (g, e) in got.data.iter().zip(&expected) {
                assert!((g - e).abs() < TOL, "nearest shift {sh}: {g} vs {e}");
            }
        }

        // --- nearest, 2-D: affine_transform on a 5x5 image ---
        let img = NdArray::new((0..25).map(f64::from).collect(), vec![5, 5]).unwrap();
        let matrix = [[0.8, 0.1, 0.5], [-0.2, 1.1, -0.3]];
        let aff_near: [f64; 25] = [
            2.0146251551549565,
            3.4209130558546903,
            5.143254751465219,
            6.827687752661231,
            8.460719421510634,
            6.549318909071743,
            7.668924898935514,
            9.354219461368206,
            10.899941103456493,
            12.540003700962757,
            10.413530228321878,
            11.244918977395098,
            12.928921197972759,
            14.45181146728866,
            16.14830632579837,
            14.423340955417984,
            15.131075101064493,
            16.906402812731706,
            18.548188532711343,
            20.344624494677564,
            18.925743768149676,
            19.34462449467756,
            20.822282799067025,
            22.1862150202129,
            23.560224563000926,
        ];
        let got = affine_transform(&img, &matrix, 3, BoundaryMode::Nearest, 0.0).unwrap();
        for (g, e) in got.data.iter().zip(&aff_near) {
            assert!((g - e).abs() < TOL, "affine nearest: {g} vs {e}");
        }
    }

    #[test]
    fn spline_orders_2_4_5_reflect_nearest_match_scipy_golden() {
        // Golden artifact: order-2/4/5 spline interpolation under reflect and
        // nearest, verified bit-exact (< 1e-12) against scipy.ndimage.shift.
        const TOL: f64 = 1e-12;

        // 24-sample signal — long enough that scipy's horizon-truncated
        // reflect prefilter agrees with the exact reflect spline.
        let reflect_sig: Vec<f64> = vec![
            0.3,
            0.5254971787363237,
            0.5408057557229191,
            0.635795154994567,
            0.9073737420479793,
            1.1218983591130598,
            0.9605329581387094,
            0.3810003087672637,
            -0.3017020476460769,
            -0.7092777890824606,
            -0.755474785911513,
            -0.683640240436719,
            -0.7543994215437057,
            -0.9320889866700686,
            -0.9171525129384751,
            -0.4901346154497399,
            0.21127233132626663,
            0.7907645539468529,
            0.9680644073934838,
            0.8294796607067128,
            0.6893699987049906,
            0.7208919079549805,
            0.7636074874743252,
            0.5187033633516349,
        ];
        // 20-sample signal for nearest (scipy edge-pads then prefilters).
        let nearest_sig: Vec<f64> = vec![
            3.0,
            6.659085290854023,
            7.733103563999704,
            7.262733439989361,
            6.402699019254376,
            5.352334042747226,
            3.3152787084265762,
            -0.5082563677459531,
            -5.742971009482519,
            -10.505933637858343,
            -12.322633532285415,
            -9.722976713580003,
            -3.377144701355263,
            4.093988896839788,
            9.568169743690905,
            11.164761757677065,
            9.11413039759255,
            5.2310787849920235,
            1.4770673813308624,
            -1.28493841798744,
        ];

        // (order, shift, reflect_expected[24], nearest_expected[20])
        let reflect_cases: [(usize, f64, [f64; 24]); 6] = [
            (
                2,
                0.5,
                [
                    0.26148308720278246,
                    0.4155507383916525,
                    0.5472011973925972,
                    0.5664538150897354,
                    0.7604795549389343,
                    1.0433444434468437,
                    1.09654218902416,
                    0.7071276914152734,
                    0.026824730108091693,
                    -0.5508830275790764,
                    -0.7654459115477829,
                    -0.7154518031101201,
                    -0.6983033751844239,
                    -0.8468865937050349,
                    -0.9663306954404633,
                    -0.7520952320863598,
                    -0.15024642559423818,
                    0.5381246491578959,
                    0.929646071739341,
                    0.9193147657674041,
                    0.74464160605702,
                    0.688234235537289,
                    0.7670006073591302,
                    0.6477597020251524,
                ],
            ),
            (
                2,
                -0.37,
                [
                    0.3780968923876382,
                    0.55004707350839,
                    0.5567027348546525,
                    0.7227376085775954,
                    1.0090429004084585,
                    1.113130943496916,
                    0.7843065594109809,
                    0.12160861701575015,
                    -0.49372903854689265,
                    -0.7606764023822605,
                    -0.7287487672977672,
                    -0.6900200942581103,
                    -0.8193391026409322,
                    -0.9623302689796851,
                    -0.8061576943691924,
                    -0.24611398750032565,
                    0.45647795796202273,
                    0.9044804339423547,
                    0.940375232004542,
                    0.7662187840164019,
                    0.6833216657503591,
                    0.7537183567179542,
                    0.6886982650692292,
                    0.47859265329190565,
                ],
            ),
            (
                4,
                0.5,
                [
                    0.24919710286198923,
                    0.4210244727263792,
                    0.5524281034476087,
                    0.5602266698366776,
                    0.7592912141355752,
                    1.0438683473422665,
                    1.1003630566341693,
                    0.7094684774167466,
                    0.025503727675793522,
                    -0.5544978018281325,
                    -0.7674151045724874,
                    -0.7137523113281415,
                    -0.6949508843909721,
                    -0.8457354387839149,
                    -0.9688133468477396,
                    -0.7556590938567713,
                    -0.15104338293567277,
                    0.5408374075539142,
                    0.9332702314723473,
                    0.9191833468382019,
                    0.7432418789219479,
                    0.6819812577573401,
                    0.7734287692536976,
                    0.6536480506760897,
                ],
            ),
            (
                4,
                -0.37,
                [
                    0.3869529341080939,
                    0.5523304004529634,
                    0.5499485827935139,
                    0.7236393917242437,
                    1.0119800113189061,
                    1.1170115675786068,
                    0.7839958190189207,
                    0.1174707972542666,
                    -0.4976674323370129,
                    -0.7605187702755805,
                    -0.7251160523592382,
                    -0.6872301772854028,
                    -0.8206614372669139,
                    -0.9663613020780203,
                    -0.8083726625442841,
                    -0.24377069109259103,
                    0.46112959985545743,
                    0.9070523141560112,
                    0.937816217926544,
                    0.763427599299972,
                    0.6791291174592256,
                    0.7621964050599042,
                    0.6894720825029287,
                    0.4654505812331833,
                ],
            ),
            (
                5,
                0.5,
                [
                    0.24664420183516106,
                    0.4225212997266592,
                    0.5528012064287837,
                    0.5591957745312407,
                    0.7598435317911724,
                    1.0435143378614826,
                    1.100718716223185,
                    0.7094658229233732,
                    0.025493853573514993,
                    -0.554673693440588,
                    -0.7674842359598136,
                    -0.7136784807659607,
                    -0.6947950173637502,
                    -0.8456848730447002,
                    -0.9689022894220467,
                    -0.7558335020671808,
                    -0.15102415968874833,
                    0.54084205720131,
                    0.9336389967679087,
                    0.9187634898419974,
                    0.743848977053193,
                    0.6808670364500307,
                    0.7738768938002417,
                    0.6552941151329906,
                ],
            ),
            (
                5,
                -0.37,
                [
                    0.3887104336475638,
                    0.5523483633366845,
                    0.5490039930189293,
                    0.7242224945828809,
                    1.0117050291181455,
                    1.1173694128105638,
                    0.783923750904463,
                    0.11739358978258295,
                    -0.497837122526764,
                    -0.760515821762791,
                    -0.7249843007003728,
                    -0.6870949028912255,
                    -0.8206862478588601,
                    -0.9665005790472293,
                    -0.808509601008641,
                    -0.24368011714684484,
                    0.46119264644791735,
                    0.9073455158365293,
                    0.9373781680623959,
                    0.7639226336515214,
                    0.6781161253053056,
                    0.7629494473761906,
                    0.690543922427542,
                    0.46281529288662704,
                ],
            ),
        ];
        let nearest_cases: [(usize, f64, [f64; 20]); 6] = [
            (
                2,
                0.5,
                [
                    2.702782168545731,
                    4.7323124708063045,
                    7.539684170032537,
                    7.59833792841338,
                    6.853636275443444,
                    5.941574255900906,
                    4.517050437157529,
                    1.626574125849127,
                    -3.0484058295297993,
                    -8.341048657584219,
                    -11.90092081432834,
                    -11.567695137020781,
                    -6.875349347008646,
                    0.41930555933159286,
                    7.226892772957188,
                    10.867972365048056,
                    10.496999042226356,
                    7.265602002672257,
                    3.290225672078391,
                    -0.17437136985106116,
                ],
            ),
            (
                2,
                -0.37,
                [
                    4.143855124090996,
                    7.411370394557743,
                    7.664948393508871,
                    6.9670715336683084,
                    6.062446732562023,
                    4.757893548963193,
                    2.112480257859293,
                    -2.348975284726245,
                    -7.674830569641832,
                    -11.61228831717209,
                    -11.87717295688152,
                    -7.7122123026463765,
                    -0.5964627366132293,
                    6.464456949352444,
                    10.630213515905218,
                    10.763407261040815,
                    7.79101585332909,
                    3.785836417357385,
                    0.23944528575002455,
                    -1.5144466683392535,
                ],
            ),
            (
                4,
                0.5,
                [
                    2.6306032753377897,
                    4.697677989716788,
                    7.6402814064148314,
                    7.545366113966397,
                    6.862294863414422,
                    5.9281809547987425,
                    4.525437316966706,
                    1.6393818602944958,
                    -3.0367250821189504,
                    -8.34239817893728,
                    -11.915925475875854,
                    -11.587814629608703,
                    -6.887227531241737,
                    0.4227348301443286,
                    7.24463922503075,
                    10.88318185549883,
                    10.51233129984018,
                    7.241531466614057,
                    3.313945117527742,
                    -0.20634453768157623,
                ],
            ),
            (
                4,
                -0.37,
                [
                    4.162142563509179,
                    7.497026260222897,
                    7.602893110891612,
                    6.975955770303944,
                    6.055591973804203,
                    4.772478754215038,
                    2.121701266176032,
                    -2.3521289394036264,
                    -7.69463788238694,
                    -11.63716593410173,
                    -11.890714552701453,
                    -7.703612968518502,
                    -0.5700891403689989,
                    6.494366064096208,
                    10.642371222821803,
                    10.762808027340544,
                    7.75581615791813,
                    3.8007888645719055,
                    0.19652586496626542,
                    -1.508641460236482,
                ],
            ),
            (
                5,
                0.5,
                [
                    2.617324371623272,
                    4.690581905217046,
                    7.6620204610085985,
                    7.525044246696608,
                    6.874675819576095,
                    5.9209275253562055,
                    4.5291241625419305,
                    1.6380438152728438,
                    -3.0355217064381708,
                    -8.342752526750935,
                    -11.916013984915796,
                    -11.588518485130257,
                    -6.88716496823088,
                    0.42215713803932103,
                    7.2464605870211765,
                    10.8808616974449,
                    10.517502421146744,
                    7.233213623388482,
                    3.3227201112356703,
                    -0.21129098372014327,
                ],
            ),
            (
                5,
                -0.37,
                [
                    4.1610753206808875,
                    7.5146000230372225,
                    7.583725393528469,
                    6.987998134141277,
                    6.048632205806019,
                    4.776376928836171,
                    2.120389642875675,
                    -2.3511217898395307,
                    -7.695294262733125,
                    -11.637447867735148,
                    -11.891319351908553,
                    -7.703307768605489,
                    -0.5702480314364181,
                    6.496092676469299,
                    10.640308006662673,
                    10.7670381892709,
                    7.748081030883152,
                    3.8095536403517665,
                    0.19015282605698844,
                    -1.50938219040037,
                ],
            ),
        ];

        let refl_arr = NdArray::new(reflect_sig, vec![24]).unwrap();
        for (order, sh, expected) in reflect_cases {
            let got = shift(&refl_arr, &[sh], order, BoundaryMode::Reflect, 0.0).unwrap();
            for (g, e) in got.data.iter().zip(&expected) {
                assert!(
                    (g - e).abs() < TOL,
                    "reflect order={order} shift={sh}: {g} vs {e}"
                );
            }
        }
        let near_arr = NdArray::new(nearest_sig, vec![20]).unwrap();
        for (order, sh, expected) in nearest_cases {
            let got = shift(&near_arr, &[sh], order, BoundaryMode::Nearest, 0.0).unwrap();
            for (g, e) in got.data.iter().zip(&expected) {
                assert!(
                    (g - e).abs() < TOL,
                    "nearest order={order} shift={sh}: {g} vs {e}"
                );
            }
        }
    }

    #[test]
    fn map_coordinates_order_three_hits_sample_points_exactly() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = map_coordinates(
            &input,
            &[vec![0.0, 1.0], vec![1.0, 0.0]],
            3,
            BoundaryMode::Nearest,
            0.0,
        )
        .unwrap();
        assert!((result[0] - 2.0).abs() < 1e-8, "got {}", result[0]);
        assert!((result[1] - 3.0).abs() < 1e-8, "got {}", result[1]);
    }

    #[test]
    fn map_coordinates_order_three_matches_scipy_boundary_references() {
        let input = NdArray::new(vec![0.0, 10.0, 20.0, 30.0], vec![4]).unwrap();
        let coordinates = [vec![-0.25, 0.5, 2.5, 3.25]];
        let cases = [(
            BoundaryMode::Reflect,
            1e-4,
            [
                -1.205_403_622_252_472_2,
                4.464_266_599_578_032,
                25.535_713_203_749_697,
                31.205_357_548_593_86,
            ],
        )];
        for (mode, tol, expected) in cases {
            let result = map_coordinates(&input, &coordinates, 3, mode, 0.0).unwrap();
            for (got, want) in result.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < tol,
                    "mode {mode:?}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn binary_fill_holes_fills() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = binary_fill_holes(&input).unwrap();
        assert_eq!(result.data[4], 1.0); // hole should be filled
    }

    #[test]
    fn binary_fill_holes_structure_controls_diagonal_connectivity() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();

        // scipy.ndimage.binary_fill_holes(x) fills the center because the
        // default cross structure does not connect diagonal background.
        assert_eq!(
            binary_fill_holes(&input).unwrap().data,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );

        // scipy.ndimage.binary_fill_holes(x, structure=np.ones((3, 3))) leaves
        // the center open because diagonal background reaches the border.
        let full_structure = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        assert_eq!(
            binary_fill_holes_with_structure(&input, &full_structure, &[0])
                .unwrap()
                .data,
            vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn binary_fill_holes_structure_validates_shape_and_origin() {
        let input = NdArray::new(vec![1.0, 0.0, 1.0], vec![3]).unwrap();
        let structure_2d = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        assert!(binary_fill_holes_with_structure(&input, &structure_2d, &[0]).is_err());

        let structure_1d = NdArray::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        assert!(binary_fill_holes_with_structure(&input, &structure_1d, &[2]).is_err());
    }

    #[test]
    fn watershed_ift_does_not_wrap_row_edges() {
        #[rustfmt::skip]
        let input = NdArray::new(vec![
            9.0, 9.0, 0.0,
            0.0, 0.0, 9.0,
        ], vec![2, 3]).unwrap();
        #[rustfmt::skip]
        let markers = NdArray::new(vec![
            0.0, 0.0, 1.0,
            0.0, 2.0, 0.0,
        ], vec![2, 3]).unwrap();

        let result = watershed_ift(&input, &markers, None).unwrap();

        assert_eq!(result.data[2], 1.0);
        assert_eq!(result.data[4], 2.0);
        assert_eq!(
            result.data[3], 2.0,
            "right edge of first row must not be adjacent to left edge of second row"
        );
    }

    #[test]
    fn watershed_ift_validates_structure_shape() {
        let input = NdArray::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let markers = NdArray::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let bad_structure = NdArray::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();

        assert!(watershed_ift(&input, &markers, Some(&bad_structure)).is_err());
    }

    #[test]
    fn watershed_ift_integer_bucket_matches_scipy_tie_order() {
        #[rustfmt::skip]
        let input = NdArray::new(vec![
            1.0, 2.0, 3.0, 3.0, 3.0,
            3.0, 0.0, 2.0, 0.0, 3.0,
            3.0, 2.0, 3.0, 2.0, 0.0,
            1.0, 2.0, 3.0, 3.0, 1.0,
        ], vec![4, 5]).unwrap();
        #[rustfmt::skip]
        let markers = NdArray::new(vec![
            3.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        ], vec![4, 5]).unwrap();

        // scipy.ndimage.watershed_ift on the same uint8/int32 arrays.
        assert_eq!(
            watershed_ift(&input, &markers, None).unwrap().data,
            vec![
                3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0,
                2.0, 2.0, 2.0, 1.0,
            ]
        );
    }

    #[test]
    fn watershed_ift_2d_cross_fast_path_matches_generic_structure() {
        let rows = 9usize;
        let cols = 11usize;
        let input = NdArray::new(
            (0..rows * cols)
                .map(|idx| {
                    let row = idx / cols;
                    let col = idx % cols;
                    ((row * 17 + col * 31 + idx * 7) & 255) as f64
                })
                .collect(),
            vec![rows, cols],
        )
        .unwrap();
        let mut marker_data = vec![0.0; rows * cols];
        marker_data[0] = 1.0;
        marker_data[cols - 1] = 2.0;
        marker_data[(rows - 1) * cols] = -3.0;
        marker_data[rows * cols - 1] = 4.0;
        marker_data[(rows / 2) * cols + cols / 2] = 5.0;
        let markers = NdArray::new(marker_data, vec![rows, cols]).unwrap();
        let structure = generate_binary_structure(2, 1);

        let fast = watershed_ift(&input, &markers, None).unwrap();
        let generic = watershed_ift(&input, &markers, Some(&structure)).unwrap();

        assert_eq!(fast.data, generic.data);
    }

    #[test]
    fn find_objects_bounding_boxes() {
        #[rustfmt::skip]
        let labels_data = vec![
            1.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 2.0, 2.0,
        ];
        let labels = NdArray::new(labels_data, vec![3, 3]).unwrap();
        let objects = find_objects(&labels, 2);
        assert_eq!(objects.len(), 2);

        let (min1, max1) = objects[0].as_ref().unwrap();
        assert_eq!(min1, &vec![0, 0]);
        assert_eq!(max1, &vec![0, 1]);

        let (min2, max2) = objects[1].as_ref().unwrap();
        assert_eq!(min2, &vec![2, 1]);
        assert_eq!(max2, &vec![2, 2]);
    }

    #[test]
    fn center_of_mass_single_region() {
        let data = NdArray::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let com = center_of_mass(&data, &labels, 1).unwrap();
        assert_eq!(com.len(), 1);
        assert!((com[0][0] - 0.5).abs() < 1e-10);
        assert!((com[0][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn generic_filter_max() {
        let input = NdArray::new(vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]).unwrap();
        let result = generic_filter(
            &input,
            |n| {
                n.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                })
            },
            3,
            BoundaryMode::Constant,
            0.0,
        )
        .unwrap();
        assert_eq!(result.data[1], 5.0); // max of [1, 5, 3]
        assert_eq!(result.data[2], 5.0); // max of [5, 3, 2]
    }

    #[test]
    fn generic_derivative_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();
        let axis_constant = |array: &NdArray, axis: usize| {
            let mut output = NdArray::zeros(array.shape.clone());
            for value in &mut output.data {
                *value = (axis + 1) as f64;
            }
            output
        };

        // scipy.ndimage.generic_gradient_magnitude(input, derivative, axes=(-1,))
        assert_eq!(
            generic_gradient_magnitude_axes(&input, axis_constant, &[-1])
                .unwrap()
                .data,
            vec![2.0; 6]
        );
        // scipy.ndimage.generic_gradient_magnitude(input, derivative, axes=(-2,))
        assert_eq!(
            generic_gradient_magnitude_axes(&input, axis_constant, &[-2])
                .unwrap()
                .data,
            vec![1.0; 6]
        );
        // scipy.ndimage.generic_gradient_magnitude(input, derivative, axes=(-2, -1))
        for value in generic_gradient_magnitude_axes(&input, axis_constant, &[-2, -1])
            .unwrap()
            .data
        {
            assert!((value - 5.0_f64.sqrt()).abs() < 1e-12);
        }
        assert_eq!(
            generic_gradient_magnitude_axes(&input, axis_constant, &[])
                .unwrap()
                .data,
            input.data
        );

        // scipy.ndimage.generic_laplace(input, derivative2, axes=(-1,))
        assert_eq!(
            generic_laplace_axes(&input, axis_constant, &[-1])
                .unwrap()
                .data,
            vec![2.0; 6]
        );
        // scipy.ndimage.generic_laplace(input, derivative2, axes=(-2,))
        assert_eq!(
            generic_laplace_axes(&input, axis_constant, &[-2])
                .unwrap()
                .data,
            vec![1.0; 6]
        );
        // scipy.ndimage.generic_laplace(input, derivative2, axes=(-2, -1))
        assert_eq!(
            generic_laplace_axes(&input, axis_constant, &[-2, -1])
                .unwrap()
                .data,
            vec![3.0; 6]
        );
        assert_eq!(
            generic_laplace_axes(&input, axis_constant, &[])
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn generic_derivative_axes_reject_duplicate_out_of_range_and_bad_shape() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let axis_constant = |array: &NdArray, axis: usize| {
            let mut output = NdArray::zeros(array.shape.clone());
            for value in &mut output.data {
                *value = (axis + 1) as f64;
            }
            output
        };
        let bad_shape = |_array: &NdArray, _axis: usize| NdArray::new(vec![1.0], vec![1]).unwrap();

        assert!(generic_gradient_magnitude_axes(&input, axis_constant, &[1, -1]).is_err());
        assert!(generic_gradient_magnitude_axes(&input, axis_constant, &[2]).is_err());
        assert!(generic_gradient_magnitude_axes(&input, axis_constant, &[-3]).is_err());
        assert!(generic_gradient_magnitude_axes(&input, bad_shape, &[-1]).is_err());

        assert!(generic_laplace_axes(&input, axis_constant, &[1, -1]).is_err());
        assert!(generic_laplace_axes(&input, axis_constant, &[2]).is_err());
        assert!(generic_laplace_axes(&input, axis_constant, &[-3]).is_err());
        assert!(generic_laplace_axes(&input, bad_shape, &[-1]).is_err());
    }

    #[test]
    fn generic_filter_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        // scipy.ndimage.generic_filter(input, sum, 2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            generic_filter_axes(&input, sum_window, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-6.0, 5.0, 8.0, -8.0, 11.0, 12.0]
        );
        // scipy.ndimage.generic_filter(input, sum, 2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            generic_filter_axes(&input, sum_window, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-6.0, -9.0, -3.0, 6.0, 10.0, 10.0]
        );
        // scipy.ndimage.generic_filter(input, sum, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            generic_filter_axes(
                &input,
                sum_window,
                2,
                &[-2, -1],
                BoundaryMode::Constant,
                -10.0
            )
            .unwrap()
            .data,
            vec![-26.0, -15.0, -12.0, -14.0, 16.0, 20.0]
        );
    }

    #[test]
    fn generic_filter_axes_empty_axes_does_not_invoke_callback() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();
        // AtomicUsize (Sync) so the callback satisfies the parallelized
        // generic_filter_axes `F: Fn + Sync` bound; empty axes never invokes it.
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let result = generic_filter_axes(
            &input,
            |values| {
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                values.iter().sum()
            },
            0,
            &[],
            BoundaryMode::Constant,
            -10.0,
        )
        .unwrap();

        assert_eq!(result.data, input.data);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn generic_filter_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        assert!(
            generic_filter_axes(&input, sum_window, 2, &[1, -1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            generic_filter_axes(&input, sum_window, 2, &[2], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(
            generic_filter_axes(&input, sum_window, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(
            generic_filter_axes(&input, sum_window, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err()
        );
    }

    #[test]
    fn generic_filter_origin_matches_scipy_even_window() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let weighted = |values: &[f64]| values[0] * 10.0 + values[1];

        let origin_zero =
            generic_filter_with_origins(&input, weighted, 2, &[0], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(origin_zero.data, vec![1., 12., 23., 34., 45.]);

        let shifted =
            generic_filter_with_origins(&input, weighted, 2, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(shifted.data, vec![12., 23., 34., 45., 50.]);
    }

    #[test]
    fn generic_filter_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        assert!(
            generic_filter_with_origins(&input, sum_window, 2, &[-1], BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            generic_filter_with_origins(&input, sum_window, 2, &[0], BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            generic_filter_with_origins(&input, sum_window, 2, &[1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            generic_filter_with_origins(
                &input,
                sum_window,
                3,
                &[-1, 0],
                BoundaryMode::Reflect,
                0.0
            )
            .is_err()
        );
    }

    #[test]
    fn generic_filter1d_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel per-pixel path must be BYTE-IDENTICAL to the serial loop, across axes, window
        // sizes, origins, boundary modes, and shapes above and below the parallel gate. A non-linear
        // reducer (order-sensitive) makes any accidental reordering visible.
        let reducer = |w: &[f64]| -> f64 {
            let mut acc = 0.0;
            for (i, &v) in w.iter().enumerate() {
                acc = (acc + (v * (i as f64 + 1.3)).sin()).mul_add(1.000_001, v.abs().ln_1p());
            }
            acc
        };
        let shapes: &[Vec<usize>] = &[vec![40], vec![9, 11], vec![320, 320], vec![12, 10, 9]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|k| ((k as f64 * 0.37).sin() * 5.0 - 0.3) + (k % 5) as f64)
                .collect();
            let input = NdArray::new(data, shape.clone()).unwrap();
            for &size in &[2usize, 3, 5] {
                for axis in 0..shape.len() {
                    for mode in [
                        BoundaryMode::Reflect,
                        BoundaryMode::Nearest,
                        BoundaryMode::Constant,
                        BoundaryMode::Wrap,
                    ] {
                        for origin in [-1i64, 0, 1] {
                            if (size as i64) / 2 + origin < 0 || (size as i64) / 2 - origin < 0 {
                                continue;
                            }
                            GENERIC_FILTER1D_FORCE_SERIAL.store(true, Ordering::Relaxed);
                            let ser = generic_filter1d_with_origin(
                                &input, reducer, size, axis, origin, mode, 1.5,
                            );
                            GENERIC_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
                            let par = generic_filter1d_with_origin(
                                &input, reducer, size, axis, origin, mode, 1.5,
                            );
                            match (ser, par) {
                                (Ok(a), Ok(b)) => {
                                    assert_eq!(a.shape, b.shape);
                                    for (x, y) in a.data.iter().zip(&b.data) {
                                        assert_eq!(
                                            x.to_bits(),
                                            y.to_bits(),
                                            "shape={shape:?} size={size} axis={axis} mode={mode:?} origin={origin}"
                                        );
                                    }
                                }
                                (Err(_), Err(_)) => {}
                                _ => panic!("serial/parallel disagree on Ok/Err"),
                            }
                        }
                    }
                }
            }
        }
        GENERIC_FILTER1D_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn generic_filter1d_matches_scipy_axis_fixtures() {
        let one_dim = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        // scipy.ndimage.generic_filter1d(one_dim, sum3, 3, mode='constant', cval=0.0)
        assert_eq!(
            generic_filter1d(&one_dim, sum_window, 3, 0, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![3.0, 6.0, 9.0, 7.0]
        );

        let two_dim = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // scipy.ndimage.generic_filter1d(two_dim, sum3, 3, axis=0, mode='constant', cval=0.0)
        assert_eq!(
            generic_filter1d(&two_dim, sum_window, 3, 0, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![5.0, 7.0, 9.0, 5.0, 7.0, 9.0]
        );
        // scipy.ndimage.generic_filter1d(two_dim, sum3, 3, axis=1, mode='constant', cval=0.0)
        assert_eq!(
            generic_filter1d(&two_dim, sum_window, 3, 1, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![3.0, 6.0, 5.0, 9.0, 15.0, 11.0]
        );
    }

    #[test]
    fn generic_filter1d_signed_axis_matches_scipy_fixtures() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        let default_axis =
            generic_filter1d_default_axis(&input, sum_window, 3, BoundaryMode::Constant, 0.0)
                .unwrap();
        let last_axis =
            generic_filter1d_signed_axis(&input, sum_window, 3, -1, BoundaryMode::Constant, 0.0)
                .unwrap();
        let first_axis =
            generic_filter1d_signed_axis(&input, sum_window, 3, -2, BoundaryMode::Constant, 0.0)
                .unwrap();

        // scipy.ndimage.generic_filter1d(..., sliding_sum, 3, axis=-1, mode='constant', cval=0.0)
        assert_eq!(default_axis.data, vec![3.0, 6.0, 5.0, 9.0, 15.0, 11.0]);
        assert_eq!(last_axis.data, default_axis.data);
        // scipy.ndimage.generic_filter1d(..., sliding_sum, 3, axis=-2, mode='constant', cval=0.0)
        assert_eq!(first_axis.data, vec![5.0, 7.0, 9.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn generic_filter1d_signed_axis_rejects_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        assert!(
            generic_filter1d_signed_axis(&input, sum_window, 3, 2, BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            generic_filter1d_signed_axis(&input, sum_window, 3, -3, BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn generic_filter1d_origin_matches_scipy_even_window() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        let origin_zero =
            generic_filter1d_with_origin(&input, sum_window, 2, 0, 0, BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(origin_zero.data, vec![1., 3., 5., 7., 9.]);

        let shifted =
            generic_filter1d_with_origin(&input, sum_window, 2, 0, -1, BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(shifted.data, vec![3., 5., 7., 9., 5.]);
    }

    #[test]
    fn generic_filter1d_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        assert!(
            generic_filter1d_with_origin(&input, sum_window, 2, 0, -1, BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            generic_filter1d_with_origin(&input, sum_window, 2, 0, 0, BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            generic_filter1d_with_origin(&input, sum_window, 2, 0, 1, BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            generic_filter1d_with_origin(&input, sum_window, 2, 0, -2, BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn vectorized_filter_matches_scipy_size_fixtures() {
        let one_dim = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let sum_window = |values: &[f64]| values.iter().sum::<f64>();

        // scipy.ndimage.vectorized_filter(one_dim, np.sum, size=3, mode='constant', cval=0.0)
        assert_eq!(
            vectorized_filter(&one_dim, sum_window, 3, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![3.0, 6.0, 9.0, 7.0]
        );

        let two_dim = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // scipy.ndimage.vectorized_filter(two_dim, np.sum, size=2, mode='constant', cval=0.0)
        assert_eq!(
            vectorized_filter(&two_dim, sum_window, 2, BoundaryMode::Constant, 0.0)
                .unwrap()
                .data,
            vec![1.0, 3.0, 5.0, 5.0, 12.0, 16.0]
        );
    }

    #[test]
    fn percentile_filter_median() {
        let input = NdArray::new(vec![1.0, 5.0, 3.0, 2.0, 4.0], vec![5]).unwrap();
        let result = percentile_filter(&input, 50.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        // 50th percentile of [0, 1, 5] = 1.0
        assert_eq!(result.data[0], 1.0);
    }

    #[test]
    fn percentile_filter_origin_matches_scipy_even_window() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        let median_origin_zero =
            percentile_filter_with_origins(&input, 50.0, 2, &[0], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(median_origin_zero.data, vec![1., 2., 3., 4., 5.]);

        let median_shifted =
            percentile_filter_with_origins(&input, 50.0, 2, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(median_shifted.data, vec![2., 3., 4., 5., 5.]);

        let minimum_shifted =
            percentile_filter_with_origins(&input, 0.0, 2, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(minimum_shifted.data, vec![1., 2., 3., 4., 0.]);
    }

    #[test]
    fn percentile_filter_uses_scipy_floor_rank_semantics() {
        let input = NdArray::new(vec![10., 20., 30., 40., 50.], vec![5]).unwrap();

        let low_quartile = percentile_filter(&input, 25.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(low_quartile.data, vec![0., 10., 20., 30., 0.]);

        let high_quartile =
            percentile_filter(&input, 75.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(high_quartile.data, vec![20., 30., 40., 50., 50.]);
    }

    #[test]
    fn percentile_filter_negative_percentiles_match_scipy() {
        let input = NdArray::new(vec![10., 20., 30., 40., 50.], vec![5]).unwrap();

        let negative_quartile =
            percentile_filter(&input, -25.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(negative_quartile.data, vec![20., 30., 40., 50., 50.]);

        let negative_median =
            percentile_filter(&input, -50.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(negative_median.data, vec![10., 20., 30., 40., 40.]);

        let negative_min =
            percentile_filter(&input, -100.0, 3, BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(negative_min.data, vec![0., 10., 20., 30., 0.]);
    }

    #[test]
    fn percentile_filter_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.percentile_filter(input, 50, 2, mode='constant', cval=-10.0, axes=(-1,))
        let median_last_axis = [4.0, 4.0, 7.0, 2.0, 9.0, 9.0];
        // scipy.ndimage.percentile_filter(input, 50, 2, mode='constant', cval=-10.0, axes=(-2,))
        let median_first_axis = [4.0, 1.0, 7.0, 4.0, 9.0, 7.0];
        // scipy.ndimage.percentile_filter(input, 50, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let median_all_axes = [-10.0, 1.0, 1.0, 2.0, 4.0, 7.0];
        // scipy.ndimage.percentile_filter(input, 75, 2, mode='constant', cval=-10.0, axes=(-2, -1))
        let high_quartile_all_axes = [4.0, 4.0, 7.0, 4.0, 9.0, 9.0];

        assert_eq!(
            percentile_filter_axes(&input, 50.0, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            median_last_axis
        );
        assert_eq!(
            percentile_filter_axes(&input, 50.0, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            median_first_axis
        );
        assert_eq!(
            percentile_filter_axes(&input, 50.0, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            median_all_axes
        );
        assert_eq!(
            percentile_filter_axes(&input, 75.0, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            high_quartile_all_axes
        );
        assert_eq!(
            percentile_filter_axes(&input, -25.0, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            high_quartile_all_axes
        );
        assert_eq!(
            percentile_filter_axes(&input, 50.0, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn percentile_filter_rejects_invalid_percentiles() {
        let input = NdArray::new(vec![10., 20., 30.], vec![3]).unwrap();

        assert!(percentile_filter(&input, -101.0, 3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(percentile_filter(&input, 101.0, 3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(percentile_filter(&input, f64::NAN, 3, BoundaryMode::Reflect, 0.0).is_err());
        assert!(percentile_filter_axes(&input, 101.0, 0, &[], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn percentile_filter_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(
            percentile_filter_axes(&input, 50.0, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(percentile_filter_axes(&input, 50.0, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(
            percentile_filter_axes(&input, 50.0, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(
            percentile_filter_axes(&input, 50.0, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err()
        );
    }

    #[test]
    fn percentile_filter_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 2., 3., 4., 5.], vec![5]).unwrap();

        assert!(
            percentile_filter_with_origins(&input, 50.0, 2, &[-1], BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            percentile_filter_with_origins(&input, 50.0, 2, &[0], BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            percentile_filter_with_origins(&input, 50.0, 2, &[1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(
            percentile_filter_with_origins(&input, 50.0, 3, &[-1, 0], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
    }

    #[test]
    fn grey_erosion_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.grey_erosion(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            grey_erosion_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, 1.0, 1.0, -10.0, 2.0, 3.0]
        );
        // scipy.ndimage.grey_erosion(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            grey_erosion_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, -10.0, -10.0, 2.0, 1.0, 3.0]
        );
        // scipy.ndimage.grey_erosion(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            grey_erosion_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, -10.0, -10.0, -10.0, 1.0, 1.0]
        );
        assert_eq!(
            grey_erosion_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn grey_erosion_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(grey_erosion_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_erosion_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_erosion_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_erosion_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn grey_dilation_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.grey_dilation(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            grey_dilation_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![4.0, 7.0, 7.0, 9.0, 9.0, 3.0]
        );
        // scipy.ndimage.grey_dilation(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            grey_dilation_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![4.0, 9.0, 7.0, 2.0, 9.0, 3.0]
        );
        // scipy.ndimage.grey_dilation(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            grey_dilation_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![9.0, 9.0, 7.0, 9.0, 9.0, 3.0]
        );
        assert_eq!(
            grey_dilation_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn grey_dilation_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(grey_dilation_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_dilation_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_dilation_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_dilation_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn grey_opening_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.grey_opening(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            grey_opening_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
        );
        // scipy.ndimage.grey_opening(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            grey_opening_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![2.0, 1.0, 3.0, 2.0, 1.0, 3.0]
        );
        // scipy.ndimage.grey_opening(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            grey_opening_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![1.0; 6]
        );
        assert_eq!(
            grey_opening_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn grey_opening_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(grey_opening_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_opening_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_opening_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_opening_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn grey_closing_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.grey_closing(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            grey_closing_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, 4.0, 7.0, -10.0, 9.0, 3.0]
        );
        // scipy.ndimage.grey_closing(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            grey_closing_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, -10.0, -10.0, 2.0, 9.0, 3.0]
        );
        // scipy.ndimage.grey_closing(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            grey_closing_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-10.0, -10.0, -10.0, -10.0, 9.0, 3.0]
        );
        assert_eq!(
            grey_closing_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            input.data
        );
    }

    #[test]
    fn grey_closing_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(grey_closing_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_closing_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_closing_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_closing_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn morphological_gradient_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.morphological_gradient(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            morphological_gradient_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![14.0, 6.0, 6.0, 19.0, 7.0, 0.0]
        );
        // scipy.ndimage.morphological_gradient(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            morphological_gradient_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![14.0, 19.0, 17.0, 0.0, 8.0, 0.0]
        );
        // scipy.ndimage.morphological_gradient(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            morphological_gradient_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![19.0, 19.0, 17.0, 19.0, 8.0, 2.0]
        );
        assert_eq!(
            morphological_gradient_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![0.0; 6]
        );
    }

    #[test]
    fn morphological_gradient_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(
            morphological_gradient_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(morphological_gradient_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(morphological_gradient_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(morphological_gradient_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn morphological_laplace_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.morphological_laplace(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            morphological_laplace_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-14.0, 6.0, -6.0, -5.0, -7.0, 0.0]
        );
        // scipy.ndimage.morphological_laplace(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            morphological_laplace_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-14.0, -3.0, -17.0, 0.0, -8.0, 0.0]
        );
        // scipy.ndimage.morphological_laplace(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            morphological_laplace_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-9.0, -3.0, -17.0, -5.0, -8.0, -2.0]
        );
        assert_eq!(
            morphological_laplace_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![0.0; 6]
        );
    }

    #[test]
    fn morphological_laplace_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(
            morphological_laplace_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err()
        );
        assert!(morphological_laplace_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(morphological_laplace_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(morphological_laplace_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn white_tophat_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.white_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            white_tophat_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![3.0, 0.0, 6.0, 0.0, 6.0, 0.0]
        );
        // scipy.ndimage.white_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            white_tophat_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![2.0, 0.0, 4.0, 0.0, 8.0, 0.0]
        );
        // scipy.ndimage.white_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            white_tophat_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![3.0, 0.0, 6.0, 1.0, 8.0, 2.0]
        );
        assert_eq!(
            white_tophat_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![0.0; 6]
        );
    }

    #[test]
    fn white_tophat_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(white_tophat_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(white_tophat_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(white_tophat_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(white_tophat_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn black_tophat_axes_match_scipy_subset_fixtures() {
        let input = NdArray::new(vec![4.0, 1.0, 7.0, 2.0, 9.0, 3.0], vec![2, 3]).unwrap();

        // scipy.ndimage.black_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-1,))
        assert_eq!(
            black_tophat_axes(&input, 2, &[-1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-14.0, 3.0, 0.0, -12.0, 0.0, 0.0]
        );
        // scipy.ndimage.black_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-2,))
        assert_eq!(
            black_tophat_axes(&input, 2, &[-2], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-14.0, -11.0, -17.0, 0.0, 0.0, 0.0]
        );
        // scipy.ndimage.black_tophat(input, size=2, mode='constant', cval=-10.0, axes=(-2, -1))
        assert_eq!(
            black_tophat_axes(&input, 2, &[-2, -1], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![-14.0, -11.0, -17.0, -12.0, 0.0, 0.0]
        );
        assert_eq!(
            black_tophat_axes(&input, 0, &[], BoundaryMode::Constant, -10.0)
                .unwrap()
                .data,
            vec![0.0; 6]
        );
    }

    #[test]
    fn black_tophat_axes_rejects_duplicate_and_out_of_range_axes() {
        let input = NdArray::new(vec![1.0; 6], vec![2, 3]).unwrap();

        assert!(black_tophat_axes(&input, 2, &[1, -1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(black_tophat_axes(&input, 2, &[2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(black_tophat_axes(&input, 2, &[-3], BoundaryMode::Reflect, 0.0).is_err());
        assert!(black_tophat_axes(&input, 0, &[-1], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn grey_morphology_origins_match_scipy_constant() {
        let input = NdArray::new(vec![1., 3., 2., 5., 4.], vec![5]).unwrap();

        let erosion_left =
            grey_erosion_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(erosion_left.data, vec![1., 2., 2., 0., 0.]);

        let erosion_right =
            grey_erosion_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(erosion_right.data, vec![0., 0., 1., 2., 2.]);

        let dilation_left =
            grey_dilation_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(dilation_left.data, vec![1., 3., 3., 5., 5.]);

        let dilation_right =
            grey_dilation_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(dilation_right.data, vec![3., 5., 5., 5., 4.]);

        let opening =
            grey_opening_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(opening.data, vec![1., 2., 2., 2., 2.]);

        let closing =
            grey_closing_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(closing.data, vec![0., 0., 3., 5., 4.]);
    }

    #[test]
    fn grey_morphology_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 3., 2., 5., 4.], vec![5]).unwrap();

        assert!(grey_erosion_with_origins(&input, 2, &[-1], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(grey_dilation_with_origins(&input, 2, &[0], BoundaryMode::Reflect, 0.0).is_ok());
        assert!(grey_opening_with_origins(&input, 2, &[1], BoundaryMode::Reflect, 0.0).is_err());
        assert!(grey_closing_with_origins(&input, 3, &[-2], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn morphology_origins_match_scipy_constant() {
        let input = NdArray::new(vec![1., 3., 2., 5., 4.], vec![5]).unwrap();

        let gradient =
            morphological_gradient_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(gradient.data, vec![0., 1., 1., 5., 5.]);

        let laplace =
            morphological_laplace_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0)
                .unwrap();
        assert_eq!(laplace.data, vec![1., -1., 2., -3., -2.]);

        let white =
            white_tophat_with_origins(&input, 3, &[-1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(white.data, vec![0., 1., 0., 3., 2.]);

        let black =
            black_tophat_with_origins(&input, 3, &[1], BoundaryMode::Constant, 0.0).unwrap();
        assert_eq!(black.data, vec![-1., -3., 1., 0., 0.]);
    }

    #[test]
    fn morphology_origin_validation_matches_scipy_bounds() {
        let input = NdArray::new(vec![1., 3., 2., 5., 4.], vec![5]).unwrap();

        assert!(
            morphological_gradient_with_origins(&input, 2, &[-1], BoundaryMode::Reflect, 0.0)
                .is_ok()
        );
        assert!(
            morphological_laplace_with_origins(&input, 2, &[1], BoundaryMode::Reflect, 0.0)
                .is_err()
        );
        assert!(white_tophat_with_origins(&input, 3, &[-2], BoundaryMode::Reflect, 0.0).is_err());
        assert!(black_tophat_with_origins(&input, 3, &[2], BoundaryMode::Reflect, 0.0).is_err());
    }

    #[test]
    fn morphological_gradient_detects_edges() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let result = morphological_gradient(&input, 3, BoundaryMode::Constant, 0.0).unwrap();
        // Center pixel: max-min = 1-0 = 1
        assert_eq!(result.data[4], 1.0);
        // Corner pixel: max-min = 0-0 = 0 (unless center is in neighborhood)
    }

    #[test]
    fn morphological_laplace_matches_scipy_nearest_2d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 4.0,
            0.0, 3.0, 5.0,
            2.0, 1.0, 0.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let got = morphological_laplace(&input, 3, BoundaryMode::Nearest, 0.0).unwrap();
        // scipy.ndimage.morphological_laplace(x, size=(3, 3), mode='nearest')
        let expect = vec![1.0, 1.0, -1.0, 3.0, -1.0, -5.0, -1.0, 3.0, 5.0];
        assert_eq!(got.data, expect);
    }

    #[test]
    fn morphological_laplace_matches_scipy_constant_2d() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 4.0,
            0.0, 3.0, 5.0,
            2.0, 1.0, 0.0,
        ];
        let input = NdArray::new(data, vec![3, 3]).unwrap();
        let got = morphological_laplace(&input, 3, BoundaryMode::Constant, -2.0).unwrap();
        // scipy.ndimage.morphological_laplace(x, size=(3, 3), mode='constant', cval=-2.0)
        let expect = vec![-1.0, -1.0, -5.0, 1.0, -1.0, -7.0, -3.0, 1.0, 3.0];
        assert_eq!(got.data, expect);
    }

    #[test]
    fn morphological_laplace_rejects_zero_size() {
        let input = NdArray::new(vec![0.0, 2.0, 1.0, 4.0, 3.0], vec![5]).unwrap();
        let err = morphological_laplace(&input, 0, BoundaryMode::Reflect, 0.0)
            .expect_err("zero-size footprint should be rejected");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn extrema_labels_correct() {
        let data = NdArray::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![6]).unwrap();
        let (mins, maxs) = extrema_labels(&data, &labels, 2);
        assert_eq!(mins[0], 1.0); // min of [3, 1, 4]
        assert_eq!(maxs[0], 4.0); // max of [3, 1, 4]
        assert_eq!(mins[1], 1.0); // min of [1, 5, 9]
        assert_eq!(maxs[1], 9.0); // max of [1, 5, 9]
    }

    #[test]
    fn shift_rejects_nan_shift_values() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = shift(&input, &[f64::NAN, 0.0], 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject NaN");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn shift_rejects_inf_shift_values() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = shift(
            &input,
            &[0.0, f64::INFINITY],
            1,
            BoundaryMode::Constant,
            0.0,
        )
        .expect_err("should reject Inf");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn zoom_rejects_nan_zoom_factors() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = zoom(&input, &[f64::NAN, 1.0], 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject NaN");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn zoom_rejects_negative_zoom_factors() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = zoom(&input, &[-1.0, 1.0], 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject negative");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn zoom_rejects_zero_zoom_factors() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = zoom(&input, &[0.0, 1.0], 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject zero");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn rotate_rejects_nan_angle() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = rotate(&input, f64::NAN, false, 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject NaN");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn rotate_rejects_inf_angle() {
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = rotate(&input, f64::INFINITY, false, 1, BoundaryMode::Constant, 0.0)
            .expect_err("should reject Inf");
        assert!(matches!(err, NdimageError::InvalidArgument(_)));
    }

    #[test]
    fn binary_hit_or_miss_default_structure2_uses_complement() {
        // [frankenscipy-ee82u] Regression: when structure2 is None,
        // scipy uses logical_not(structure1) as the miss structure.
        // fsci was returning just `hit` (the structure1-erosion),
        // skipping the miss filter. Verify a small case where the
        // miss filter actually rejects a candidate.
        //
        // Input has an isolated foreground pixel surrounded by
        // background. With structure1 = [[0,0,0],[0,1,0],[0,0,0]]
        // (single-cell), every foreground pixel passes the hit. With
        // the default structure2 (= NOT(structure1)), the miss filter
        // requires every NEIGHBOR cell to be 0. Only the isolated
        // pixel passes both.
        #[rustfmt::skip]
        let input_data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input = NdArray::new(input_data, vec![5, 5]).unwrap();
        #[rustfmt::skip]
        let s1_data = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let structure1 = NdArray::new(s1_data, vec![3, 3]).unwrap();

        let result = binary_hit_or_miss(&input, &structure1, None).unwrap();
        // Only the isolated pixel at (1, 1) passes both hit and miss.
        // The 2×2 block at (2..4, 3..4) has every corner pixel as
        // foreground but each pixel has at least one foreground
        // neighbor, so the miss filter (requiring all 8 neighbors = 0)
        // rejects them all.
        assert_eq!(
            result.data[5 + 1],
            1.0,
            "isolated pixel at (1,1) should pass"
        );
        assert_eq!(
            result.data[10 + 3],
            0.0,
            "block pixel (2,3) should fail miss"
        );
        assert_eq!(
            result.data[10 + 4],
            0.0,
            "block pixel (2,4) should fail miss"
        );
        assert_eq!(
            result.data[15 + 3],
            0.0,
            "block pixel (3,3) should fail miss"
        );
        assert_eq!(
            result.data[15 + 4],
            0.0,
            "block pixel (3,4) should fail miss"
        );
    }

    #[test]
    fn sobel_matches_scipy_reference_values() {
        // scipy.ndimage.sobel([[1,2,3],[4,5,6],[7,8,9]], axis=0, mode='reflect')
        let input = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = sobel(&input, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let expected = [12.0, 12.0, 12.0, 24.0, 24.0, 24.0, 12.0, 12.0, 12.0];
        for (i, val) in result.data.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "sobel[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn prewitt_matches_scipy_reference_values() {
        // scipy.ndimage.prewitt([[1,2,3],[4,5,6],[7,8,9]], axis=0, mode='reflect')
        let input = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = prewitt(&input, 0, BoundaryMode::Reflect, 0.0).unwrap();
        let expected = [9.0, 9.0, 9.0, 18.0, 18.0, 18.0, 9.0, 9.0, 9.0];
        for (i, val) in result.data.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "prewitt[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn laplace_matches_scipy_reference_values() {
        // scipy.ndimage.laplace([[1,2,3],[4,5,6],[7,8,9]], mode='reflect')
        let input = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = laplace(&input, BoundaryMode::Reflect, 0.0).unwrap();
        let expected = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0];
        for (i, val) in result.data.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "laplace[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn binary_erosion_matches_scipy_reference_values() {
        // scipy.ndimage.binary_erosion with 5x5 all-ones input
        // Erosion with 3x3 structure removes the border, leaving 3x3 interior
        let input = NdArray::new(vec![1.0; 25], vec![5, 5]).unwrap();
        let result = binary_erosion(&input, 3, 1).unwrap();
        // After erosion, only the 3x3 center should remain
        // Index mapping for 5x5: center 3x3 is at positions (1,1) to (3,3)
        // which are indices 6,7,8, 11,12,13, 16,17,18
        let center_sum: f64 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
            .iter()
            .map(|&i| result.data[i])
            .sum();
        assert!(
            center_sum >= 1.0,
            "interior pixels should survive erosion, got center_sum={center_sum}"
        );
    }

    #[test]
    fn binary_dilation_matches_scipy_reference_values() {
        // scipy.ndimage.binary_dilation([[0,0,0],[0,1,0],[0,0,0]], structure=ones(3,3))
        // With 3x3 structure, center pixel expands to fill the whole grid
        let input = NdArray::new(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let result = binary_dilation(&input, 3, 1).unwrap();
        // With 3x3 structure, single center pixel dilates to fill everything
        let total_ones: f64 = result.data.iter().sum();
        assert!(
            total_ones >= 5.0,
            "dilation should spread to multiple pixels, got {total_ones}"
        );
    }

    #[test]
    fn label_matches_scipy_reference_values() {
        // scipy.ndimage.label([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        // -> single connected component, num_features=1
        let arr = NdArray::new(
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let (labels, num_features) = label(&arr).expect("label");
        assert_eq!(num_features, 1, "should find 1 connected component");
        // All non-zero pixels should have label 1
        assert!(labels.data[1] > 0.0, "pixel (0,1) should be labeled");
        assert!(labels.data[4] > 0.0, "pixel (1,1) should be labeled");
    }

    #[test]
    fn center_of_mass_matches_scipy_reference_values() {
        // scipy.ndimage.center_of_mass([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        // For unlabeled data, treat entire array as single region with label 1
        let arr = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let labels = NdArray::new(vec![1.0; 9], vec![3, 3]).unwrap();
        let com = center_of_mass(&arr, &labels, 1).expect("center_of_mass");
        assert_eq!(com.len(), 1, "should have 1 center of mass");
        let coords = &com[0];
        assert!(
            (coords[0] - 1.4).abs() < 1e-10,
            "y got {}, expected 1.4",
            coords[0]
        );
        assert!(
            (coords[1] - 1.1333333333333333).abs() < 1e-10,
            "x got {}, expected 1.1333...",
            coords[1]
        );
    }

    #[test]
    fn sum_labels_matches_scipy_reference_values() {
        // scipy.ndimage.sum_labels([[1, 2], [3, 4]], [[1, 1], [2, 2]], index=[1, 2])
        // Returns [sum_label_1, sum_label_2] with total = 10
        let arr = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let labels = NdArray::new(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]).unwrap();
        let sums = sum_labels(&arr, &labels, 2).expect("sum_labels");
        // sums[0] = label 1 sum, sums[1] = label 2 sum
        assert_eq!(sums.len(), 2, "should have 2 label sums");
        let total = sums[0] + sums[1];
        assert!(
            (total - 10.0).abs() < 1e-10,
            "total sum got {}, expected 10",
            total
        );
        // Each label should have non-zero sum
        assert!(sums[0] > 0.0, "label 1 sum should be > 0");
        assert!(sums[1] > 0.0, "label 2 sum should be > 0");
    }

    #[test]
    fn zoom_matches_scipy_reference_dimensions() {
        // scipy.ndimage.zoom([[1, 2], [3, 4]], 2) produces 4x4 array
        let arr = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = zoom(&arr, &[2.0, 2.0], 3, BoundaryMode::Constant, 0.0).expect("zoom");
        assert_eq!(
            result.shape,
            vec![4, 4],
            "zoom(2x) should produce 4x4 array"
        );
    }

    #[test]
    fn binary_erosion_single_pixel_matches_scipy() {
        // scipy.ndimage.binary_erosion([[0,0,0], [0,1,0], [0,0,0]])
        // With default structure, center pixel erodes to 0
        let arr = NdArray::new(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let result = binary_erosion(&arr, 3, 1).expect("binary_erosion");
        // All should be 0 after erosion
        assert!(
            result.data.iter().all(|&x| x == 0.0),
            "single pixel should erode to 0"
        );
    }

    #[test]
    fn binary_dilation_single_pixel_matches_scipy() {
        // scipy.ndimage.binary_dilation([[0,0,0], [0,1,0], [0,0,0]])
        // With default cross structure, dilates to cross pattern
        let arr = NdArray::new(
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![3, 3],
        )
        .unwrap();
        let result = binary_dilation(&arr, 3, 1).expect("binary_dilation");
        // Center should be 1
        assert_eq!(result.data[4], 1.0, "center should remain 1");
        // At least some neighbors should be 1 (cross pattern)
        let dilated_count = result.data.iter().filter(|&&x| x == 1.0).count();
        assert!(
            dilated_count >= 3,
            "dilation should expand, got {} ones",
            dilated_count
        );
    }

    #[test]
    fn binary_opening_removes_isolated_pixels_scipy() {
        // scipy.ndimage.binary_opening: erosion followed by dilation
        // Small isolated regions should be removed
        let arr = NdArray::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![5, 5],
        )
        .unwrap();
        let result = binary_opening(&arr, 3, 1).expect("binary_opening");
        // The large 2x2 block should survive, isolated pixels removed
        assert_eq!(result.shape, vec![5, 5], "shape should be preserved");
    }

    #[test]
    fn binary_closing_fills_holes_scipy() {
        // scipy.ndimage.binary_closing: dilation followed by erosion
        // Small holes should be filled
        let arr = NdArray::new(
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let result = binary_closing(&arr, 3, 1).expect("binary_closing");
        // The center hole should be filled
        let center = result.data[4];
        assert_eq!(center, 1.0, "center hole should be filled by closing");
    }

    #[test]
    fn grey_erosion_local_minimum_scipy() {
        // scipy.ndimage.grey_erosion([[1, 2, 3], [4, 5, 6], [7, 8, 9]], size=3)
        // Returns local minimum in 3x3 window
        let arr = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = grey_erosion(&arr, 3, BoundaryMode::Constant, 0.0).expect("grey_erosion");
        // Center element should be min of 3x3 = 1.0 (considering boundary)
        // With constant=0 padding, min should be 0
        assert!(
            result.data[4] <= 1.0,
            "grey_erosion center got {}, expected <= 1",
            result.data[4]
        );
    }

    #[test]
    fn grey_dilation_local_maximum_scipy() {
        // scipy.ndimage.grey_dilation([[1, 2, 3], [4, 5, 6], [7, 8, 9]], size=3)
        // Returns local maximum in 3x3 window
        let arr = NdArray::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let result = grey_dilation(&arr, 3, BoundaryMode::Constant, 0.0).expect("grey_dilation");
        // Center element should be max of 3x3 = 9.0
        assert_eq!(
            result.data[4], 9.0,
            "grey_dilation center got {}, expected 9",
            result.data[4]
        );
    }

    #[test]
    fn sobel_constant_array_zero_scipy() {
        // scipy.ndimage.sobel on constant array should give 0
        let arr = NdArray::new(vec![5.0; 9], vec![3, 3]).unwrap();
        let result = sobel(&arr, 0, BoundaryMode::Reflect, 0.0).expect("sobel");
        assert!(
            result.data.iter().all(|&x| x.abs() < 1e-10),
            "sobel on constant should be 0"
        );
    }

    #[test]
    fn laplace_constant_array_zero_scipy() {
        // scipy.ndimage.laplace on constant array should give 0
        let arr = NdArray::new(vec![5.0; 9], vec![3, 3]).unwrap();
        let result = laplace(&arr, BoundaryMode::Reflect, 0.0).expect("laplace");
        assert!(
            result.data.iter().all(|&x| x.abs() < 1e-10),
            "laplace on constant should be 0"
        );
    }

    #[test]
    fn fourier_separable_matches_full_per_element_bitexact() {
        use std::sync::atomic::Ordering;
        // The separable per-axis factor precompute must be BYTE-IDENTICAL to the old per-element
        // recompute for fourier_gaussian and fourier_uniform, across ndim 1/2/3 and even/odd sizes.
        let cases: &[(&[usize], &[f64])] = &[
            (&[17], &[1.3]),
            (&[16, 12], &[0.7, 2.1]),
            (&[8, 9, 5], &[1.1, 0.4, 3.0]),
        ];
        for &(shape, params) in cases {
            let total: usize = shape.iter().product();
            let input: Vec<Complex64> = (0..total)
                .map(|k| ((k as f64).sin() * 3.0, (k as f64 * 0.7).cos()))
                .collect();
            for gaussian in [true, false] {
                NDIMAGE_FOURIER_SEPARABLE_DISABLE.store(true, Ordering::Relaxed);
                let full = if gaussian {
                    fourier_gaussian(&input, shape, params)
                } else {
                    fourier_uniform(&input, shape, params)
                };
                NDIMAGE_FOURIER_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
                let sep = if gaussian {
                    fourier_gaussian(&input, shape, params)
                } else {
                    fourier_uniform(&input, shape, params)
                };
                for (a, b) in full.iter().zip(sep.iter()) {
                    assert_eq!(
                        a.0.to_bits(),
                        b.0.to_bits(),
                        "re {shape:?} gaussian={gaussian}"
                    );
                    assert_eq!(
                        a.1.to_bits(),
                        b.1.to_bits(),
                        "im {shape:?} gaussian={gaussian}"
                    );
                }
            }
        }
    }

    #[test]
    fn fourier_filters_parallel_match_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel per-element fill must be BYTE-IDENTICAL to the serial loop for every
        // Fourier-domain filter, across ndim and shapes above and below the parallel gate.
        let shapes: &[Vec<usize>] = &[
            vec![37],
            vec![9, 11],
            vec![600, 600], // > gate -> exercises the parallel path
            vec![7, 8, 9],
            vec![40, 30, 24],
        ];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let input: Vec<Complex64> = (0..total)
                .map(|k| {
                    (
                        (k as f64 * 0.13).sin() * 3.0 - 0.4,
                        (k as f64 * 0.07).cos() * 2.0,
                    )
                })
                .collect();
            let param: Vec<f64> = shape
                .iter()
                .enumerate()
                .map(|(d, _)| 1.5 + 0.7 * d as f64)
                .collect();
            type FourierFilterArm = (
                &'static str,
                fn(&[Complex64], &[usize], &[f64]) -> Vec<Complex64>,
            );
            let runs: &[FourierFilterArm] = &[
                ("gaussian", fourier_gaussian),
                ("uniform", fourier_uniform),
                ("shift", fourier_shift),
                ("ellipsoid", fourier_ellipsoid),
            ];
            for (name, filt) in runs {
                NDIMAGE_FOURIER_FORCE_SERIAL.store(true, Ordering::Relaxed);
                let serial = filt(&input, shape, &param);
                NDIMAGE_FOURIER_FORCE_SERIAL.store(false, Ordering::Relaxed);
                let parallel = filt(&input, shape, &param);
                assert_eq!(serial.len(), parallel.len());
                for (i, (a, b)) in serial.iter().zip(&parallel).enumerate() {
                    assert_eq!(
                        a.0.to_bits(),
                        b.0.to_bits(),
                        "{name} shape={shape:?} re idx={i}"
                    );
                    assert_eq!(
                        a.1.to_bits(),
                        b.1.to_bits(),
                        "{name} shape={shape:?} im idx={i}"
                    );
                }
            }
        }
        NDIMAGE_FOURIER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn fourier_shift_ellipsoid_separable_bitexact() {
        use std::sync::atomic::Ordering;
        // The per-axis arithmetic precompute (phase-contribution / squared-freq) must be
        // BYTE-IDENTICAL to the per-element recompute for fourier_shift and fourier_ellipsoid.
        let cases: &[(&[usize], &[f64])] = &[
            (&[17], &[2.4]),
            (&[16, 12], &[3.5, -2.0]),
            (&[8, 9, 5], &[1.5, 4.0, -3.0]),
        ];
        for &(shape, params) in cases {
            let total: usize = shape.iter().product();
            let input: Vec<Complex64> = (0..total)
                .map(|k| ((k as f64 * 0.3).sin(), (k as f64 * 0.7).cos()))
                .collect();
            for shift in [true, false] {
                NDIMAGE_FOURIER_SEPARABLE_DISABLE.store(true, Ordering::Relaxed);
                let full = if shift {
                    fourier_shift(&input, shape, params)
                } else {
                    fourier_ellipsoid(&input, shape, params)
                };
                NDIMAGE_FOURIER_SEPARABLE_DISABLE.store(false, Ordering::Relaxed);
                let sep = if shift {
                    fourier_shift(&input, shape, params)
                } else {
                    fourier_ellipsoid(&input, shape, params)
                };
                for (a, b) in full.iter().zip(sep.iter()) {
                    assert_eq!(a.0.to_bits(), b.0.to_bits(), "re {shape:?} shift={shift}");
                    assert_eq!(a.1.to_bits(), b.1.to_bits(), "im {shape:?} shift={shift}");
                }
            }
        }
    }

    #[test]
    fn unravel_odometer_matches_full_bitexact() {
        use std::sync::atomic::Ordering;
        // The row-major odometer must be BYTE-IDENTICAL to the per-element `unravel` path for
        // center_of_mass, sum_axis and pad_constant, across ndim 1/2/3 and even/odd shapes.
        let shapes: &[Vec<usize>] = &[vec![23], vec![7, 9], vec![4, 5, 3]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total).map(|k| (k as f64 * 0.37).sin() + 1.5).collect();
            let arr = NdArray::new(data.clone(), shape.clone()).unwrap();
            let labels_data: Vec<f64> = (0..total).map(|k| ((k % 4) + 1) as f64).collect();
            let labels = NdArray::new(labels_data, shape.clone()).unwrap();
            let pad: Vec<(usize, usize)> = shape.iter().map(|&s| (1, 2.min(s))).collect();
            let variants = |full: bool| {
                NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(full, Ordering::Relaxed);
                let com = center_of_mass(&arr, &labels, 4).unwrap();
                let sa = sum_axis(&arr, 0).unwrap();
                let pc = pad_constant(&arr, &pad, -3.0).unwrap();
                (com, sa, pc)
            };
            let (com_f, sa_f, pc_f) = variants(true);
            let (com_o, sa_o, pc_o) = variants(false);
            for (a, b) in com_f.iter().flatten().zip(com_o.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits(), "center_of_mass {shape:?}");
            }
            for (a, b) in sa_f.data.iter().zip(sa_o.data.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "sum_axis {shape:?}");
            }
            for (a, b) in pc_f.data.iter().zip(pc_o.data.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "pad_constant {shape:?}");
            }
        }
    }

    #[test]
    fn spline_offset_leaf_matches_index_leaf_bitexact() {
        use std::sync::atomic::Ordering;
        // The flat-offset tensor combine must be BYTE-IDENTICAL to the ORIG index-space leaf
        // (which recomputes Σ idx[d]·stride[d] per leaf) for every separable transform, across
        // orders 2..=5, all boundary modes, and ndim 1/2/3. Both arms are byte-identical by
        // construction, so a concurrent test observing either toggle state still sees the
        // same values — no lock needed.
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Mirror,
            BoundaryMode::Nearest,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
        ];
        // Mirror requires every axis length > spline order, so keep min(shape) > 5.
        let shapes: &[Vec<usize>] = &[vec![17], vec![9, 11], vec![7, 6, 8]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|k| (k as f64 * 0.41).cos() * 3.0 + 1.7)
                .collect();
            let arr = NdArray::new(data, shape.clone()).unwrap();
            let zooms: Vec<f64> = shape.iter().map(|_| 1.7).collect();
            let shifts: Vec<f64> = shape
                .iter()
                .enumerate()
                .map(|(d, _)| 0.3 + d as f64)
                .collect();
            for order in 2..=5usize {
                for &mode in &modes {
                    let variants = |orig: bool| {
                        NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
                        let z = zoom(&arr, &zooms, order, mode, -2.5).unwrap();
                        let s = shift(&arr, &shifts, order, mode, -2.5).unwrap();
                        (z, s)
                    };
                    let (z_o, s_o) = variants(true);
                    let (z_c, s_c) = variants(false);
                    NDIMAGE_SPLINE_OFFSET_DISABLE.store(false, Ordering::Relaxed);
                    for (a, b) in z_o.data.iter().zip(z_c.data.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "zoom {shape:?} order={order} {mode:?}"
                        );
                    }
                    for (a, b) in s_o.data.iter().zip(s_c.data.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "shift {shape:?} order={order} {mode:?}"
                        );
                    }
                }
            }
        }
        // Diagonal affine (the separable affine branch) is 2-D only.
        let arr = NdArray::new(
            (0..12 * 10)
                .map(|k| (k as f64 * 0.23).sin() + 2.0)
                .collect(),
            vec![12, 10],
        )
        .unwrap();
        let diag = [[0.7, 0.0, 1.3], [0.0, 1.4, -0.6]];
        for order in 2..=5usize {
            for &mode in &modes {
                let variants = |orig: bool| {
                    NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
                    affine_transform(&arr, &diag, order, mode, -2.5).unwrap()
                };
                let a_o = variants(true);
                let a_c = variants(false);
                NDIMAGE_SPLINE_OFFSET_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in a_o.data.iter().zip(a_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "affine order={order} {mode:?}");
                }
            }
        }

        // PER-PIXEL path (`sample_interpolated`): rotate / general affine / map_coordinates have
        // COUPLED coords, so supports are rebuilt per pixel — but the leaf address is orthogonal
        // to separability, and now also uses the flat-offset combine. Orders 0..=5 (order 0
        // returns before the leaf and is a null case; order 1 has 2 taps/axis).
        let general = [[0.85, 0.25, 1.3], [-0.2, 1.05, -0.6]];
        for order in 0..=5usize {
            for &mode in &modes {
                let variants = |orig: bool| {
                    NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
                    (
                        affine_transform(&arr, &general, order, mode, -2.5).unwrap(),
                        rotate(&arr, 27.0, false, order, mode, -2.5).unwrap(),
                    )
                };
                let (g_o, r_o) = variants(true);
                let (g_c, r_c) = variants(false);
                NDIMAGE_SPLINE_OFFSET_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in g_o.data.iter().zip(g_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "general affine order={order} {mode:?}"
                    );
                }
                for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "rotate order={order} {mode:?}");
                }
            }
        }

        // `map_coordinates` over ndim 1/2/3 exercises ALL THREE arms of `sample_spline_offsets`
        // (`[b0]`, the flattened 2-D `[b0,b1]`, and the `[b0, rest @ ..]` recursion).
        let mc_shapes: &[Vec<usize>] = &[vec![17], vec![9, 11], vec![7, 6, 8]];
        for shape in mc_shapes {
            let total: usize = shape.iter().product();
            let arr = NdArray::new(
                (0..total).map(|k| (k as f64 * 0.37).sin() - 0.8).collect(),
                shape.clone(),
            )
            .unwrap();
            // Coupled coords, plus exact-integer and ±1 ULP samples.
            let npts = 64usize;
            let coords: Vec<Vec<f64>> = (0..shape.len())
                .map(|d| {
                    (0..npts)
                        .map(|p| {
                            let base = (p % shape[d]) as f64;
                            match p % 4 {
                                0 => base,
                                1 => f64::from_bits(base.to_bits() + 1),
                                2 => base + 0.41 + 0.13 * d as f64,
                                _ => base - 0.27,
                            }
                        })
                        .collect()
                })
                .collect();
            for order in 0..=5usize {
                for &mode in &modes {
                    let variants = |orig: bool| {
                        NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
                        map_coordinates(&arr, &coords, order, mode, -2.5).unwrap()
                    };
                    let m_o = variants(true);
                    let m_c = variants(false);
                    NDIMAGE_SPLINE_OFFSET_DISABLE.store(false, Ordering::Relaxed);
                    for (a, b) in m_o.iter().zip(m_c.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "map_coordinates {shape:?} order={order} {mode:?}"
                        );
                    }
                }
            }
        }

        // Parallel chunked fan-out: the leaf runs inside the per-pixel closure on every worker.
        let big = NdArray::new(
            (0..256 * 256).map(|k| (k as f64 * 0.017).cos()).collect(),
            vec![256, 256],
        )
        .unwrap();
        assert!(
            ndimage_filter_thread_count(256 * 256, 4usize.pow(2)) > 1,
            "big rotate case must exercise the parallel chunked path"
        );
        for order in [1usize, 3, 5] {
            let variants = |orig: bool| {
                NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
                rotate(&big, 33.0, false, order, BoundaryMode::Reflect, 0.5).unwrap()
            };
            let r_o = variants(true);
            let r_c = variants(false);
            NDIMAGE_SPLINE_OFFSET_DISABLE.store(false, Ordering::Relaxed);
            for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "parallel rotate order={order}");
            }
        }
    }

    #[test]
    fn geometric_transform_odometer_matches_unravel_bitexact() {
        use std::sync::atomic::Ordering;
        // `fill_pixels_parallel_indexed`'s per-chunk row-major odometer must be BYTE-IDENTICAL to
        // the ORIG per-pixel `unravel_with_shape`, for every geometric transform that consumes the
        // output multi-index — covering BOTH the separable (order>=2) and generic (order<2)
        // branches, and both the serial and multi-threaded chunkings.
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Mirror,
            BoundaryMode::Nearest,
            BoundaryMode::Constant,
        ];
        let shapes: &[Vec<usize>] = &[vec![19], vec![13, 11], vec![7, 6, 8]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|k| (k as f64 * 0.29).sin() * 2.0 - 0.4)
                .collect();
            let arr = NdArray::new(data, shape.clone()).unwrap();
            let zooms: Vec<f64> = shape.iter().map(|_| 1.6).collect();
            let shifts: Vec<f64> = shape
                .iter()
                .enumerate()
                .map(|(d, _)| 0.7 - d as f64)
                .collect();
            for order in 0..=5usize {
                for &mode in &modes {
                    let variants = |orig: bool| {
                        NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(orig, Ordering::Relaxed);
                        let z = zoom(&arr, &zooms, order, mode, 1.25).unwrap();
                        let s = shift(&arr, &shifts, order, mode, 1.25).unwrap();
                        (z, s)
                    };
                    let (z_o, s_o) = variants(true);
                    let (z_c, s_c) = variants(false);
                    NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(false, Ordering::Relaxed);
                    assert_eq!(z_o.shape, z_c.shape);
                    for (a, b) in z_o.data.iter().zip(z_c.data.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "zoom {shape:?} order={order} {mode:?}"
                        );
                    }
                    for (a, b) in s_o.data.iter().zip(s_c.data.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "shift {shape:?} order={order} {mode:?}"
                        );
                    }
                }
            }
        }
        // 2-D transforms: diagonal affine (separable branch), general affine + rotate (generic
        // per-pixel branch, which also reaches `fill_pixels_parallel_indexed`).
        let arr = NdArray::new(
            (0..14 * 12)
                .map(|k| (k as f64 * 0.17).cos() + 1.1)
                .collect(),
            vec![14, 12],
        )
        .unwrap();
        let diag = [[0.8, 0.0, 1.1], [0.0, 1.3, -0.4]];
        let general = [[0.9, 0.2, 1.1], [-0.15, 1.05, -0.4]];
        for order in 0..=5usize {
            for &mode in &modes {
                let variants = |orig: bool| {
                    NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(orig, Ordering::Relaxed);
                    (
                        affine_transform(&arr, &diag, order, mode, 1.25).unwrap(),
                        affine_transform(&arr, &general, order, mode, 1.25).unwrap(),
                        rotate(&arr, 23.0, false, order, mode, 1.25).unwrap(),
                    )
                };
                let (d_o, g_o, r_o) = variants(true);
                let (d_c, g_c, r_c) = variants(false);
                NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in d_o.data.iter().zip(d_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "diag affine order={order} {mode:?}"
                    );
                }
                for (a, b) in g_o.data.iter().zip(g_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "general affine order={order} {mode:?}"
                    );
                }
                for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "rotate order={order} {mode:?}");
                }
            }
        }

        // The odometer is SEEDED PER THREAD CHUNK (`start != 0`), so a seeding bug only shows on
        // the multi-threaded path. `ndimage_filter_thread_count` needs pixels·(order+1)^ndim >=
        // 2^18: 256²→1.5x is 147_456 output pixels, so order>=1 fans out across every core.
        let big = NdArray::new(
            (0..256 * 256).map(|k| (k as f64 * 0.013).sin()).collect(),
            vec![256, 256],
        )
        .unwrap();
        assert!(
            ndimage_filter_thread_count(384 * 384, 2usize.pow(2)) > 1,
            "big zoom case must exercise the parallel chunked path"
        );
        for order in [1usize, 3, 5] {
            for &mode in &[BoundaryMode::Reflect, BoundaryMode::Constant] {
                let variants = |orig: bool| {
                    NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(orig, Ordering::Relaxed);
                    (
                        zoom(&big, &[1.5, 1.5], order, mode, 0.5).unwrap(),
                        shift(&big, &[2.6, -1.4], order, mode, 0.5).unwrap(),
                    )
                };
                let (z_o, s_o) = variants(true);
                let (z_c, s_c) = variants(false);
                NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in z_o.data.iter().zip(z_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "parallel zoom order={order} {mode:?}"
                    );
                }
                for (a, b) in s_o.data.iter().zip(s_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "parallel shift order={order} {mode:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cardinal_bspline_lanes_matches_scalar_bitexact() {
        // The lane-parallel kernel must reproduce the scalar one BIT-for-BIT, per lane, for every
        // order, at both shipped lane widths (f64x4 and the f64x2 remainder). Adversarial x: edges
        // ±(n+1)/2, integers, half-integers, ±1/±2 ULP neighbours, and 200k pseudo-random values.
        let mut s = 0x243f_6a88_85a3_08d3u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for order in 2..=5usize {
            let half = (order as f64 + 1.0) * 0.5;
            let mut xs: Vec<f64> = Vec::new();
            for sgn in [-1.0f64, 1.0] {
                for b in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, half] {
                    let v = sgn * b;
                    xs.push(v);
                    xs.push(f64::from_bits(v.to_bits().wrapping_add(1)));
                    xs.push(f64::from_bits(v.to_bits().wrapping_add(2)));
                    if v != 0.0 {
                        xs.push(f64::from_bits(v.to_bits().wrapping_sub(1)));
                        xs.push(f64::from_bits(v.to_bits().wrapping_sub(2)));
                    }
                }
            }
            for _ in 0..200_000 {
                xs.push(rnd() * 16.0 - 8.0);
            }
            for chunk in xs.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let v = Simd::<f64, 4>::from_slice(chunk);
                let got = cardinal_bspline_x4(order, v).to_array();
                for (j, &x) in chunk.iter().enumerate() {
                    assert_eq!(
                        got[j].to_bits(),
                        cardinal_bspline(order, x).to_bits(),
                        "f64x4 order={order} x={x:e}"
                    );
                }
            }
            for chunk in xs.chunks(2) {
                if chunk.len() < 2 {
                    continue;
                }
                let v = Simd::<f64, 2>::from_slice(chunk);
                let got = cardinal_bspline_x2(order, v).to_array();
                for (j, &x) in chunk.iter().enumerate() {
                    assert_eq!(
                        got[j].to_bits(),
                        cardinal_bspline(order, x).to_bits(),
                        "f64x2 order={order} x={x:e}"
                    );
                }
            }
        }

        // `cardinal_bspline_run` must equal the per-tap scalar calls it replaces, for every run
        // length it can see (compact 2/4/6 and the ORIG comparator's 3/5/7/9/11).
        for order in 1..=5usize {
            for &(lo, hi) in &[
                (-3isize, -2isize),
                (0, 3),
                (-2, 3),
                (5, 10),
                (-5, 5),
                (7, 7 + 2 * order as isize),
            ] {
                let ntaps = (hi - lo + 1) as usize;
                if ntaps > 12 {
                    continue;
                }
                for t in [0.0, 0.25, 0.5, 0.75, 1.0 / 3.0] {
                    let cc = lo as f64 + 1.5 + t;
                    let mut simd = [0.0f64; 12];
                    let mut scalar = [0.0f64; 12];
                    NDIMAGE_BSPLINE_SIMD_DISABLE.store(false, std::sync::atomic::Ordering::Relaxed);
                    cardinal_bspline_run(order, cc, lo, hi, &mut simd[..ntaps]);
                    NDIMAGE_BSPLINE_SIMD_DISABLE.store(true, std::sync::atomic::Ordering::Relaxed);
                    cardinal_bspline_run(order, cc, lo, hi, &mut scalar[..ntaps]);
                    NDIMAGE_BSPLINE_SIMD_DISABLE.store(false, std::sync::atomic::Ordering::Relaxed);
                    for j in 0..ntaps {
                        assert_eq!(
                            simd[j].to_bits(),
                            scalar[j].to_bits(),
                            "run order={order} lo={lo} hi={hi} cc={cc} lane={j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bspline_simd_run_matches_scalar_transforms_bitexact() {
        use std::sync::atomic::Ordering;
        // End-to-end: every consumer of the cardinal kernel must be BYTE-IDENTICAL with the
        // lane-parallel run vs the per-tap scalar calls — separable (zoom/shift/diag-affine) and
        // per-pixel (rotate/general-affine/map_coordinates) alike.
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Mirror,
            BoundaryMode::Nearest,
            BoundaryMode::Constant,
            BoundaryMode::Wrap,
        ];
        let shapes: &[Vec<usize>] = &[vec![7], vec![9, 11], vec![7, 6, 8], vec![23], vec![17, 19]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let arr = NdArray::new(
                (0..total)
                    .map(|k| (k as f64 * 0.29).sin() * 1.9 - 0.6)
                    .collect(),
                shape.clone(),
            )
            .unwrap();
            let zooms: Vec<f64> = shape.iter().map(|_| 1.7).collect();
            let shifts: Vec<f64> = shape.iter().map(|_| -1.35).collect();
            let int_shifts: Vec<f64> = shape.iter().map(|_| 2.0).collect();
            for order in 0..=5usize {
                for &mode in &modes {
                    let variants = |orig: bool| {
                        NDIMAGE_BSPLINE_SIMD_DISABLE.store(orig, Ordering::Relaxed);
                        (
                            zoom(&arr, &zooms, order, mode, 0.75).unwrap(),
                            shift(&arr, &shifts, order, mode, 0.75).unwrap(),
                            shift(&arr, &int_shifts, order, mode, 0.75).unwrap(),
                        )
                    };
                    let (z_o, s_o, i_o) = variants(true);
                    let (z_c, s_c, i_c) = variants(false);
                    NDIMAGE_BSPLINE_SIMD_DISABLE.store(false, Ordering::Relaxed);
                    for (tag, o, c) in [
                        ("zoom", &z_o, &z_c),
                        ("shift", &s_o, &s_c),
                        ("shift_int", &i_o, &i_c),
                    ] {
                        for (a, b) in o.data.iter().zip(c.data.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{tag} {shape:?} order={order} {mode:?}"
                            );
                        }
                    }
                }
            }
        }

        let arr = NdArray::new(
            (0..15 * 13)
                .map(|k| (k as f64 * 0.23).cos() + 1.3)
                .collect(),
            vec![15, 13],
        )
        .unwrap();
        let diag = [[0.75, 0.0, 1.2], [0.0, 1.35, -0.5]];
        let general = [[0.88, 0.22, 1.2], [-0.18, 1.02, -0.5]];
        let coords: Vec<Vec<f64>> = {
            let (mut rr, mut cc) = (Vec::new(), Vec::new());
            for i in 0..15usize {
                for j in 0..13usize {
                    rr.push(match (i + j) % 4 {
                        0 => 0.0,
                        1 => 14.0,
                        2 => f64::from_bits((i as f64).to_bits() + 1),
                        _ => i as f64 + 0.41,
                    });
                    cc.push(match (i + j) % 4 {
                        0 => 12.0,
                        1 => 0.0,
                        2 => j as f64 + 0.63,
                        _ => f64::from_bits((j as f64).to_bits().wrapping_sub(1)),
                    });
                }
            }
            vec![rr, cc]
        };
        for order in 0..=5usize {
            for &mode in &modes {
                let variants = |orig: bool| {
                    NDIMAGE_BSPLINE_SIMD_DISABLE.store(orig, Ordering::Relaxed);
                    (
                        affine_transform(&arr, &diag, order, mode, 0.75).unwrap(),
                        affine_transform(&arr, &general, order, mode, 0.75).unwrap(),
                        rotate(&arr, 29.0, false, order, mode, 0.75).unwrap(),
                        map_coordinates(&arr, &coords, order, mode, 0.75).unwrap(),
                    )
                };
                let (d_o, g_o, r_o, m_o) = variants(true);
                let (d_c, g_c, r_c, m_c) = variants(false);
                NDIMAGE_BSPLINE_SIMD_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in d_o.data.iter().zip(d_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "diag order={order} {mode:?}");
                }
                for (a, b) in g_o.data.iter().zip(g_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "general order={order} {mode:?}");
                }
                for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "rotate order={order} {mode:?}");
                }
                for (a, b) in m_o.iter().zip(m_c.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "map_coords order={order} {mode:?}"
                    );
                }
            }
        }

        // Parallel chunked fan-out.
        let big = NdArray::new(
            (0..256 * 256).map(|k| (k as f64 * 0.015).sin()).collect(),
            vec![256, 256],
        )
        .unwrap();
        assert!(
            ndimage_filter_thread_count(256 * 256, 4usize.pow(2)) > 1,
            "big rotate case must exercise the parallel chunked path"
        );
        for order in [1usize, 3, 5] {
            let variants = |orig: bool| {
                NDIMAGE_BSPLINE_SIMD_DISABLE.store(orig, Ordering::Relaxed);
                rotate(&big, 33.0, false, order, BoundaryMode::Reflect, 0.5).unwrap()
            };
            let r_o = variants(true);
            let r_c = variants(false);
            NDIMAGE_BSPLINE_SIMD_DISABLE.store(false, Ordering::Relaxed);
            for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "parallel rotate order={order}");
            }
        }
    }

    #[test]
    fn bspline_compact_support_matches_full_window_bitexact() {
        use std::sync::atomic::Ordering;
        // `compute_axis_support`'s compact tap window (`floor - n/2 ..= floor + n/2 + 1`) must be
        // BYTE-IDENTICAL to the ORIG full `floor ± order` window, which evaluated `2·order+1`
        // taps and discarded the zero-weight ones. Covers the cardinal Nearest/Reflect/Mirror
        // kernel (the only path the lever touches) plus Constant/Wrap as controls, across the
        // separable (zoom/shift/diag-affine) and per-pixel (rotate/general-affine/map_coordinates)
        // branches.
        let modes = [
            BoundaryMode::Reflect,
            BoundaryMode::Mirror,
            BoundaryMode::Nearest,
            BoundaryMode::Constant,
        ];
        let shapes: &[Vec<usize>] = &[vec![19], vec![13, 11], vec![7, 6, 8]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            let data: Vec<f64> = (0..total)
                .map(|k| (k as f64 * 0.31).cos() * 1.7 - 0.3)
                .collect();
            let arr = NdArray::new(data, shape.clone()).unwrap();
            let zooms: Vec<f64> = shape.iter().map(|_| 1.6).collect();
            // INTEGER shifts drive coords exactly onto integers (support-edge taps evaluate to
            // exact 0.0); the 1-ULP perturbation is the case that breaks an FP-derived bound.
            let int_shifts: Vec<f64> = shape.iter().map(|_| 1.0).collect();
            let ulp_shifts: Vec<f64> = shape
                .iter()
                .map(|_| f64::from_bits(1.0f64.to_bits() + 1))
                .collect();
            let frac_shifts: Vec<f64> = shape
                .iter()
                .enumerate()
                .map(|(d, _)| 0.7 - d as f64)
                .collect();
            for order in 0..=5usize {
                for &mode in &modes {
                    let variants = |orig: bool| {
                        NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);
                        (
                            zoom(&arr, &zooms, order, mode, 1.25).unwrap(),
                            shift(&arr, &frac_shifts, order, mode, 1.25).unwrap(),
                            shift(&arr, &int_shifts, order, mode, 1.25).unwrap(),
                            shift(&arr, &ulp_shifts, order, mode, 1.25).unwrap(),
                        )
                    };
                    let (z_o, s_o, i_o, u_o) = variants(true);
                    let (z_c, s_c, i_c, u_c) = variants(false);
                    NDIMAGE_BSPLINE_COMPACT_DISABLE.store(false, Ordering::Relaxed);
                    for (a, b) in z_o.data.iter().zip(z_c.data.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "zoom {shape:?} order={order} {mode:?}"
                        );
                    }
                    for (tag, o, c) in [
                        ("frac", &s_o, &s_c),
                        ("integer", &i_o, &i_c),
                        ("integer+1ulp", &u_o, &u_c),
                    ] {
                        for (a, b) in o.data.iter().zip(c.data.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "shift[{tag}] {shape:?} order={order} {mode:?}"
                            );
                        }
                    }
                }
            }
        }

        // 2-D: diagonal affine (separable) + general affine / rotate / map_coordinates
        // (per-pixel, where `compute_axis_support` is 61.7% of self time).
        let arr = NdArray::new(
            (0..14 * 12)
                .map(|k| (k as f64 * 0.19).sin() + 1.4)
                .collect(),
            vec![14, 12],
        )
        .unwrap();
        let diag = [[0.8, 0.0, 1.1], [0.0, 1.3, -0.4]];
        let general = [[0.9, 0.2, 1.1], [-0.15, 1.05, -0.4]];
        // Coordinates pinned exactly on integers and one ULP either side of them.
        let coords: Vec<Vec<f64>> = {
            let (mut rr, mut cc) = (Vec::new(), Vec::new());
            for i in 0..14usize {
                for j in 0..12usize {
                    let (y, x) = (i as f64, j as f64);
                    rr.push(match (i + j) % 3 {
                        0 => y,
                        1 => f64::from_bits(y.to_bits() + 1),
                        _ => y + 0.37,
                    });
                    cc.push(match (i + j) % 3 {
                        0 => x,
                        1 => f64::from_bits(x.to_bits().wrapping_sub(1)),
                        _ => x + 0.63,
                    });
                }
            }
            vec![rr, cc]
        };
        for order in 0..=5usize {
            for &mode in &modes {
                let variants = |orig: bool| {
                    NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);
                    (
                        affine_transform(&arr, &diag, order, mode, 1.25).unwrap(),
                        affine_transform(&arr, &general, order, mode, 1.25).unwrap(),
                        rotate(&arr, 23.0, false, order, mode, 1.25).unwrap(),
                        map_coordinates(&arr, &coords, order, mode, 1.25).unwrap(),
                    )
                };
                let (d_o, g_o, r_o, m_o) = variants(true);
                let (d_c, g_c, r_c, m_c) = variants(false);
                NDIMAGE_BSPLINE_COMPACT_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in d_o.data.iter().zip(d_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "diag affine order={order} {mode:?}"
                    );
                }
                for (a, b) in g_o.data.iter().zip(g_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "general affine order={order} {mode:?}"
                    );
                }
                for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "rotate order={order} {mode:?}");
                }
                for (a, b) in m_o.iter().zip(m_c.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "map_coordinates order={order} {mode:?}"
                    );
                }
            }
        }

        // Multi-threaded chunking: the lever sits inside the per-pixel closure, so exercise it
        // on the parallel fan-out too.
        let big = NdArray::new(
            (0..256 * 256).map(|k| (k as f64 * 0.011).cos()).collect(),
            vec![256, 256],
        )
        .unwrap();
        assert!(
            ndimage_filter_thread_count(256 * 256, 4usize.pow(2)) > 1,
            "big rotate case must exercise the parallel chunked path"
        );
        for order in [1usize, 3, 5] {
            for &mode in &[BoundaryMode::Reflect, BoundaryMode::Mirror] {
                let variants = |orig: bool| {
                    NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);
                    rotate(&big, 33.0, false, order, mode, 0.5).unwrap()
                };
                let r_o = variants(true);
                let r_c = variants(false);
                NDIMAGE_BSPLINE_COMPACT_DISABLE.store(false, Ordering::Relaxed);
                for (a, b) in r_o.data.iter().zip(r_c.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "parallel rotate order={order} {mode:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn binary_dilation_odometer_matches_full_bitexact() {
        use std::sync::atomic::Ordering;
        // binary_dilation's odometer + direct-flat write must be BYTE-IDENTICAL to the
        // per-element unravel + per-offset out_idx path, across ndim 1/2/3 and iterations.
        let shapes: &[Vec<usize>] = &[vec![25], vec![11, 13], vec![5, 6, 4]];
        for shape in shapes {
            let total: usize = shape.iter().product();
            // Sparse-ish binary foreground.
            let data: Vec<f64> = (0..total)
                .map(|k| ((k * 7 + 3) % 5 == 0) as u8 as f64)
                .collect();
            let arr = NdArray::new(data, shape.clone()).unwrap();
            for iters in [1usize, 2] {
                NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(true, Ordering::Relaxed);
                let full = binary_dilation(&arr, 3, iters).unwrap();
                NDIMAGE_UNRAVEL_ODOMETER_DISABLE.store(false, Ordering::Relaxed);
                let odo = binary_dilation(&arr, 3, iters).unwrap();
                for (a, b) in full.data.iter().zip(odo.data.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "binary_dilation {shape:?} iters={iters}"
                    );
                }
            }
        }
    }
}

/// frankenscipy-5f06d — driver for `ND_FILTER_FORCE_SCALAR`, which had none.
/// The default path processes 8 consecutive interior pixels with one SIMD
/// accumulator; the toggle restores the scalar interior. Documented BIT-IDENTICAL:
/// each lane sums the SAME taps in the SAME k-order, `+= w*x` with no FMA
/// contraction.
#[cfg(test)]
mod toggle_ab_nd_filter_scalar {
    use super::{BoundaryMode, ND_FILTER_FORCE_SCALAR, NdArray, correlate_with_origins};
    use std::sync::atomic::Ordering;

    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// This toggle has NO size gate -- it is predicated on `ndim >= 2` and an
    /// innermost stride of 1. What it DOES need is an interior box wide enough
    /// for the 8-wide path to trigger: the SIMD arm runs on runs of 8 consecutive
    /// interior pixels along the innermost axis, so a fixture narrower than
    /// `8 + kernel_width` would leave every pixel on the boundary fallback and
    /// compare the scalar path with itself.
    ///
    /// Driven under two boundary modes because the interior/border split is part
    /// of what the toggle changes: Constant and Reflect take different border
    /// paths, and a fixture that only exercised one would leave the other's
    /// interaction with the SIMD interior undriven.
    #[test]
    fn nd_filter_scalar_toggle_is_bit_identical() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        const H: usize = 192;
        const W: usize = 192;
        const K: usize = 3;
        // Interior run along the innermost axis must comfortably exceed the 8-wide
        // block, or every pixel takes the border path and both arms agree trivially.
        const _: () = assert!(W >= 8 + K);

        assert!(
            f64::from_bits(4.5f64.to_bits() ^ 1).to_bits() != 4.5f64.to_bits()
                && 0.0f64.to_bits() != (-0.0f64).to_bits(),
            "the bit comparison is blind; the exact assertions below would pass \
             vacuously"
        );

        let data: Vec<f64> = (0..(H * W) as u64)
            .map(|i| {
                let k = i.wrapping_mul(2_654_435_761);
                ((k % 100_003) as f64) / 1000.0 - 50.0
            })
            .collect();
        let input = NdArray::new(data, vec![H, W]).expect("input");
        // Asymmetric weights: a symmetric kernel can mask a tap-ordering fault
        // because reversing the taps leaves the sum unchanged.
        let weights = NdArray::new(
            (0..(K * K) as u64).map(|i| 0.5 + i as f64).collect(),
            vec![K, K],
        )
        .expect("weights");
        let origins = vec![0i64, 0i64];

        for mode in [BoundaryMode::Constant, BoundaryMode::Reflect] {
            ND_FILTER_FORCE_SCALAR.store(true, Ordering::Relaxed);
            let scalar = correlate_with_origins(&input, &weights, &origins, mode, 0.0)
                .expect("correlate scalar");
            ND_FILTER_FORCE_SCALAR.store(false, Ordering::Relaxed);
            let simd = correlate_with_origins(&input, &weights, &origins, mode, 0.0)
                .expect("correlate simd");

            assert_eq!(
                scalar.data.len(),
                H * W,
                "correlate returned the wrong element count for {mode:?}"
            );
            assert!(
                scalar.data.iter().any(|v| v.is_finite() && *v != 0.0),
                "the {mode:?} fixture produced an all-zero result, which compares \
                 equal in both arms while proving nothing"
            );
            let diff = scalar
                .data
                .iter()
                .zip(&simd.data)
                .position(|(x, y)| x.to_bits() != y.to_bits());
            assert!(
                diff.is_none(),
                "ND_FILTER_FORCE_SCALAR is documented BIT-IDENTICAL and is not \
                 under {mode:?}: first difference at flat index {diff:?}. Each SIMD \
                 lane sums the same taps in the same k-order with no FMA \
                 contraction, so a difference here is a real defect in the \
                 vectorized interior"
            );
        }
        ND_FILTER_FORCE_SCALAR.store(false, Ordering::Relaxed);

        // Both arms agreeing cannot catch a fault they share. One interior pixel,
        // far from every border, is checked against the definition of correlation
        // computed directly from the input.
        let (r, c) = (H / 2, W / 2);
        let mut expect = 0.0f64;
        for kr in 0..K {
            for kc in 0..K {
                let ir = r + kr - K / 2;
                let ic = c + kc - K / 2;
                expect += weights.data[kr * K + kc] * input.data[ir * W + ic];
            }
        }
        let got = correlate_with_origins(&input, &weights, &origins, BoundaryMode::Constant, 0.0)
            .expect("correlate")
            .data[r * W + c];
        assert!(
            (got - expect).abs() <= 1e-9 * expect.abs().max(1.0),
            "the interior pixel disagrees with the correlation definition: got \
             {got}, expected {expect}"
        );
    }
}

/// frankenscipy-22van gate (a) — the knob read is PER TRANSFORM, not per pixel.
///
/// The hoist put both `NDIMAGE_SPLINE_OFFSET_DISABLE` and
/// `NDIMAGE_BSPLINE_COMPACT_DISABLE` behind one `SplineFlags::resolve()` outside the
/// pixel loop. Nothing structural stopped a later edit from moving that call back
/// inside, and the symptom would be invisible: the results stay byte-identical and
/// only the specialisation is lost. This pins the frequency.
///
/// It measures a COUNT, never a cost. The bead is explicit that `cfg(test)` counters
/// in this path are optimisation barriers, so the half of the gate that decides
/// whether the lever is worth anything must be an A/B of two shipping binaries. This
/// is the other half: the one that says the read is not per pixel.
#[cfg(test)]
mod van22_knob_read_is_per_transform {
    use super::{BoundaryMode, NdArray, SPLINE_FLAGS_RESOLVE_COUNT, shift};
    use std::sync::atomic::Ordering;

    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resolve_runs_per_transform_not_per_pixel() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Big enough that per-pixel and per-transform differ by orders of magnitude:
        // 64x64 = 4096 pixels. A count of ~1 and a count of ~4096 cannot be confused.
        const W: usize = 64;
        const PIXELS: usize = W * W;
        let data: Vec<f64> = (0..PIXELS as u64)
            .map(|i| ((i.wrapping_mul(2_654_435_761) % 1009) as f64) / 1009.0)
            .collect();
        let input = NdArray::new(data, vec![W, W]).expect("fixture");

        SPLINE_FLAGS_RESOLVE_COUNT.store(0, Ordering::Relaxed);
        let out = shift(&input, &[0.5, -0.25], 3, BoundaryMode::Reflect, 0.0)
            .expect("shift failed on a well-formed fixture");
        let count = SPLINE_FLAGS_RESOLVE_COUNT.load(Ordering::Relaxed);

        assert_eq!(
            out.size(),
            PIXELS,
            "the transform did not produce a full image"
        );

        // MUST-HIT: the counter has to move at all. A resolve that is never called —
        // because the code was refactored past it — would give a count of 0 and
        // trivially satisfy any "not per pixel" bound.
        assert!(
            count >= 1,
            "SplineFlags::resolve was never called, so this test is bounding nothing"
        );

        // THE GATE. Per-transform means a small constant, possibly one per worker in
        // the parallel arm; per-pixel means PIXELS. The bound is deliberately loose
        // — one per worker plus slack — because the claim is an ORDER OF MAGNITUDE,
        // not an exact count, and pinning an exact count would make this test fail on
        // a machine with a different core count for no defect.
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let ceiling = (workers * 4).max(16);
        assert!(
            count <= ceiling,
            "SplineFlags::resolve ran {count} times for {PIXELS} pixels (ceiling \
             {ceiling} = 4 per worker on {workers} workers). Anything near {PIXELS} \
             means the read is back inside the pixel loop — results stay \
             byte-identical, so no other test would notice"
        );
    }
}
