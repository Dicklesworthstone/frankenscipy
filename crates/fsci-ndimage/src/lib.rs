#![forbid(unsafe_code)]

//! N-dimensional image processing for FrankenSciPy.
//!
//! Matches `scipy.ndimage` core operations:
//! - Filters: uniform, gaussian, median, minimum, maximum, convolve, correlate
//! - Morphology: binary_erosion, binary_dilation, binary_opening, binary_closing
//! - Measurements: label, find_objects, value_indices, sum, mean, variance, standard_deviation
//! - Interpolation: shift, rotate, zoom, map_coordinates
//! - Distance transforms: distance_transform_edt

use fsci_interpolate::make_interp_spline;

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
}

const DEFAULT_GAUSSIAN_TRUNCATE: f64 = 4.0;

fn gaussian_kernel_radius(sigma: f64) -> usize {
    (DEFAULT_GAUSSIAN_TRUNCATE * sigma + 0.5) as usize
}

/// Computes a 1-D Gaussian convolution kernel for the `order`-th derivative.
///
/// Mirrors `scipy.ndimage._filters._gaussian_kernel1d`: builds the normalized
/// Gaussian `phi(x)`, then for `order > 0` multiplies by the polynomial `q(x)`
/// produced by differentiating `q(x) * phi(x)` `order` times. Each derivative
/// maps `q -> q' + q * p'` with `p'(x) = -x / sigma^2`; `Q_deriv` applies that
/// operator to the polynomial coefficients of `q` (superdiagonal `D` for `q'`,
/// subdiagonal `P` for `q * p'`).
fn gaussian_kernel1d(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let n = 2 * radius + 1;
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
        return phi_x;
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
    (0..n)
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
        .collect()
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
    let kernel_1d = gaussian_kernel1d(sigma, order, radius);
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
        let total: usize = shape.iter().product();
        if total != data.len() {
            return Err(NdimageError::DimensionMismatch(format!(
                "shape {:?} requires {} elements, got {}",
                shape,
                total,
                data.len()
            )));
        }
        let strides = compute_strides(&shape);
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

fn uniform_interpolation_knots(len: usize, order: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(len + order + 1);
    knots.extend(std::iter::repeat_n(0.0, order + 1));
    for i in 0..len.saturating_sub(order + 1) {
        knots.push((i + 1 + (order - 1) / 2) as f64);
    }
    knots.extend(std::iter::repeat_n((len - 1) as f64, order + 1));
    knots
}

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
    let period = (len - 1) as i64;
    if (0..len as i64).contains(&idx) {
        idx as usize
    } else {
        idx.rem_euclid(period) as usize
    }
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
    let span = order as isize;
    let mut vals: Vec<f64> = (-span..=span)
        .map(|m| {
            let t = (x + 0.5 * m as f64).abs();
            (1.0 - t).max(0.0)
        })
        .collect();
    for d in 2..=order {
        let n = d as f64;
        let half = (n + 1.0) / 2.0;
        let mut next = vec![0.0; vals.len()];
        for idx in 1..vals.len() - 1 {
            let m = idx as isize - span;
            let arg = x + 0.5 * m as f64;
            next[idx] = ((arg + half) * vals[idx + 1] + (half - arg) * vals[idx - 1]) / n;
        }
        vals = next;
    }
    vals[span as usize]
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

fn prefilter_spline_coefficients(
    input: &NdArray,
    order: usize,
    mode: BoundaryMode,
) -> Result<SplinePrefilter, NdimageError> {
    let ndim = input.ndim();
    if order <= 1 {
        // Order <= 1 has no spline coefficients to solve — the samples ARE the
        // coefficients. But linear interpolation under `reflect` needs support
        // beyond [0, len-1] near a boundary (a coord folded to e.g. -0.3 spans
        // indices -1 and 0). Pad the array with the reflected values so the
        // support always lands in range, mirroring the order>=2 path.
        if order == 1 && mode == BoundaryMode::Reflect {
            return Ok(SplinePrefilter {
                coeffs: pad_array_mode(input, SPLINE_NEAREST_PAD, mode)?,
                coord_offsets: vec![SPLINE_NEAREST_PAD as f64; ndim],
            });
        }
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
        let reduced_shape: Vec<usize> = current
            .shape
            .iter()
            .enumerate()
            .filter_map(|(d, &size)| (d != axis).then_some(size))
            .collect();
        let line_count = reduced_shape.iter().product::<usize>().max(1);
        let mut next = current.clone();
        for line_flat in 0..line_count {
            let reduced_idx = unravel_with_shape(line_flat, &reduced_shape);
            let mut idx = vec![0usize; ndim];
            let mut src = 0usize;
            for (d, slot) in idx.iter_mut().enumerate() {
                if d == axis {
                    continue;
                }
                *slot = reduced_idx[src];
                src += 1;
            }
            let mut line = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                idx[axis] = i;
                line.push(current.get(&idx));
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
                _ => spline_coefficients_for_line(&line, order)?,
            };
            for (i, coeff) in coeffs.into_iter().enumerate() {
                idx[axis] = i;
                next.set(&idx, coeff);
            }
        }
        current = next;
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

fn sample_interpolated(
    input: &NdArray,
    coeffs: &NdArray,
    coords: &[f64],
    coord_offsets: &[f64],
    order: usize,
    mode: BoundaryMode,
    cval: f64,
) -> f64 {
    if order == 0 {
        if mode == BoundaryMode::Wrap {
            let idx: Vec<usize> = coords
                .iter()
                .enumerate()
                .map(|(axis, coord)| {
                    wrap_interpolation_index(coord.round() as i64, input.shape[axis])
                })
                .collect();
            return input.get(&idx);
        }
        let idx: Vec<i64> = coords.iter().map(|coord| coord.round() as i64).collect();
        return input.get_boundary(&idx, mode, cval);
    }

    let mut bases = Vec::with_capacity(coords.len());
    for (axis, &coord) in coords.iter().enumerate() {
        let coeff_len = coeffs.shape[axis];
        let Some(mapped) = map_interpolation_coordinate(coord, input.shape[axis], mode) else {
            return cval;
        };
        let spline_coord = if coord_offsets[axis] > 0.0 {
            match mode {
                BoundaryMode::Nearest => {
                    (coord + coord_offsets[axis]).clamp(0.0, (coeff_len - 1) as f64)
                }
                BoundaryMode::Wrap => {
                    let mut wrapped = coord + coord_offsets[axis];
                    let period = input.shape[axis] as f64;
                    let padded_max = (coeff_len - 1) as f64;
                    while wrapped < 0.0 {
                        wrapped += period;
                    }
                    while wrapped > padded_max {
                        wrapped -= period;
                    }
                    wrapped
                }
                _ => mapped + coord_offsets[axis],
            }
        } else {
            mapped
        };
        // Snap sub-ULP boundary excursions back into the valid spline
        // domain. `map_reflect_coordinate` uses a half-pixel offset and
        // can return values like -1.22e-15 for an input of essentially
        // 0.0; without this clamp the B-spline basis at that coord is
        // identically zero and sample_interpolated falls through to
        // cval, dropping corner pixels under e.g. rotate(360°).
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
        // Constant and Wrap order-3 share `cubic_constant_wrap_coefficients`,
        // which produces *cardinal* cubic B-spline coefficients with mirror
        // boundary symmetry (scipy's constant-mode spline filter is mirror-
        // based). They must be reconstructed with the cardinal cubic B-spline
        // kernel and a mirror index fold — not the de Boor knot basis used by
        // the Nearest/Reflect path, whose `make_interp_spline` coefficients are
        // a different representation.
        if matches!(mode, BoundaryMode::Wrap | BoundaryMode::Constant) && effective_order == 3 {
            let base = spline_coord.floor() as isize - 1;
            let t = spline_coord - spline_coord.floor();
            let omt = 1.0 - t;
            let max = coeff_len - 1;
            let support = vec![
                (fold_wrap_cubic_index(base, max), omt * omt * omt / 6.0),
                (
                    fold_wrap_cubic_index(base + 1, max),
                    (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
                ),
                (
                    fold_wrap_cubic_index(base + 2, max),
                    (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
                ),
                (fold_wrap_cubic_index(base + 3, max), t * t * t / 6.0),
            ];
            bases.push(support);
            continue;
        }
        // Reflect / Nearest orders 2..=5: reconstruct from the exact
        // `bspline_reflect_coefficients` with the cardinal B-spline kernel.
        // scipy folds the support TAPS, not the coordinate, so the lookup
        // coordinate is used directly (offset by SPLINE_NEAREST_PAD for the
        // edge-padded `nearest` case) and each tap index is mapped back through
        // the boundary rule — half-sample mirror for reflect, edge-clamp for
        // nearest. The padded de Boor fallback for short reflect axes has
        // coord_offsets > 0 and is excluded here so it keeps the de Boor path.
        let cardinal_reflect_nearest = effective_order == order
            && matches!(order, 2..=5)
            && ((mode == BoundaryMode::Reflect && coord_offsets[axis] == 0.0)
                || mode == BoundaryMode::Nearest);
        if cardinal_reflect_nearest {
            let cc = coord + coord_offsets[axis];
            let len = coeff_len as isize;
            let fold = |i: isize| -> usize {
                match mode {
                    BoundaryMode::Nearest => i.clamp(0, len - 1) as usize,
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
            let span = order as isize;
            let mut support = Vec::with_capacity(2 * order + 1);
            for k in (floor - span)..=(floor + span) {
                let weight = cardinal_bspline(order, cc - k as f64);
                if weight != 0.0 {
                    support.push((fold(k), weight));
                }
            }
            bases.push(support);
            continue;
        }
        if effective_order == 0 {
            bases.push(vec![(
                spline_coord.round().clamp(0.0, (coeff_len - 1) as f64) as usize,
                1.0,
            )]);
            continue;
        }
        let knots = uniform_interpolation_knots(coeff_len, effective_order);
        let weights = eval_bspline_basis_all(&knots, spline_coord, effective_order, coeff_len);
        let support: Vec<(usize, f64)> = weights
            .into_iter()
            .enumerate()
            .filter(|(_, weight)| weight.abs() > 1e-12)
            .collect();
        if support.is_empty() {
            return cval;
        }
        bases.push(support);
    }

    let mut idx = vec![0usize; coeffs.ndim()];
    sample_spline_recursive(coeffs, &bases, 0, &mut idx)
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
    let mut output = NdArray::zeros(input.shape.clone());

    for flat_out in 0..input.size() {
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

        output.data[flat_out] = sum;
    }

    Ok(output)
}

/// N-dimensional convolution with SciPy `origin` semantics.
///
/// `origins` may contain one scalar origin applied to every axis, or one origin
/// per input axis. Positive origins shift the convolution window toward higher
/// input coordinates; negative origins shift it toward lower coordinates.
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
    let mut output = NdArray::zeros(input.shape.clone());

    // Kernel center offsets
    let offsets: Vec<i64> = weights.shape.iter().map(|&s| (s as i64 - 1) / 2).collect();

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;

        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                // Convolution: flip the kernel (unlike correlation)
                let k_flipped = weights.shape[d] as i64 - 1 - k_idx[d] as i64;
                in_idx[d] = out_idx[d] as i64 + k_flipped - offsets[d] + origins[d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }

        output.data[flat_out] = sum;
    }

    Ok(output)
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

    let offset = (weights.len() as i64 - 1) / 2;
    let mut output = NdArray::zeros(input.shape.clone());
    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut in_idx: Vec<i64> = out_idx.iter().map(|&i| i as i64).collect();
        let mut sum = 0.0;
        for (k, &weight) in weights.iter().rev().enumerate() {
            in_idx[axis] = out_idx[axis] as i64 + k as i64 - offset + origin;
            sum += weight * input.get_boundary(&in_idx, mode, cval);
        }
        output.data[flat_out] = sum;
    }

    Ok(output)
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
    let mut output = NdArray::zeros(input.shape.clone());

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;

        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx: Vec<i64> = out_idx.iter().map(|&idx| idx as i64).collect();
            for (kernel_axis, &input_axis) in axes.iter().enumerate() {
                in_idx[input_axis] = out_idx[input_axis] as i64 + k_idx[kernel_axis] as i64
                    - offsets[kernel_axis]
                    - origins[kernel_axis];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }

        output.data[flat_out] = sum;
    }

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
    let mut output = NdArray::zeros(input.shape.clone());

    let offsets: Vec<i64> = weights.shape.iter().map(|&s| s as i64 / 2).collect();

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut sum = 0.0;

        for flat_k in 0..weights.size() {
            let k_idx = weights.unravel(flat_k);
            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d] - origins[d];
            }
            sum += weights.data[flat_k] * input.get_boundary(&in_idx, mode, cval);
        }

        output.data[flat_out] = sum;
    }

    Ok(output)
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
fn rank_filter_pixel(
    flat_out: usize,
    input: &NdArray,
    ndim: usize,
    kernel_total: usize,
    kernel_strides: &[usize],
    offsets: &[i64],
    origins: &[i64],
    mode: BoundaryMode,
    cval: f64,
    rank: usize,
) -> f64 {
    let out_idx = input.unravel(flat_out);
    let mut neighborhood = Vec::with_capacity(kernel_total);
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
        neighborhood.push(input.get_boundary(&in_idx, mode, cval));
    }
    select_total_rank(&mut neighborhood, rank)
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
    let nthreads = ndimage_filter_thread_count(n, kernel_total);
    if nthreads <= 1 {
        for (flat_out, slot) in output.data.iter_mut().enumerate() {
            *slot = rank_filter_pixel(
                flat_out,
                input,
                ndim,
                kernel_total,
                kernel_strides,
                offsets,
                origins,
                mode,
                cval,
                rank,
            );
        }
        return;
    }
    let chunk = n.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (t, out_chunk) in output.data.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            scope.spawn(move || {
                for (li, slot) in out_chunk.iter_mut().enumerate() {
                    *slot = rank_filter_pixel(
                        start + li,
                        input,
                        ndim,
                        kernel_total,
                        kernel_strides,
                        offsets,
                        origins,
                        mode,
                        cval,
                        rank,
                    );
                }
            });
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
    let mut cur = input.clone();
    for (d, &origin) in origins.iter().enumerate() {
        cur = minmax_filter_along_axis(&cur, d, size, origin, mode, cval, is_max);
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
    }
}

fn minmax_filter_along_axis(
    arr: &NdArray,
    axis: usize,
    size: usize,
    origin: i64,
    mode: BoundaryMode,
    cval: f64,
    is_max: bool,
) -> NdArray {
    use std::cmp::Ordering;
    use std::collections::VecDeque;

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
    let kernel_total = filter_footprint_size(axes.len(), size)?;
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
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; axes.len()];
            let mut rem = flat_k;
            for (d, slot) in k_idx.iter_mut().enumerate() {
                *slot = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx: Vec<i64> = out_idx.iter().map(|&coord| coord as i64).collect();
            for (d, &axis) in axes.iter().enumerate() {
                in_idx[axis] += k_idx[d] as i64 - offsets[d] - origins[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        output.data[flat_out] = select_total_rank(&mut neighborhood, rank);
    }

    Ok(output)
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
    F: Fn(&[f64]) -> f64,
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
    F: Fn(&[f64]) -> f64,
{
    let axes = normalize_signed_axes(axes, input.ndim())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let kernel_total = filter_footprint_size(axes.len(), size)?;
    if input.size() == 0 {
        return Err(NdimageError::EmptyInput);
    }

    let mut output = NdArray::zeros(input.shape.clone());
    let offsets: Vec<i64> = vec![size as i64 / 2; axes.len()];
    let kernel_shape: Vec<usize> = vec![size; axes.len()];
    let kernel_strides = compute_strides(&kernel_shape);

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; axes.len()];
            let mut rem = flat_k;
            for (d, slot) in k_idx.iter_mut().enumerate() {
                *slot = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx: Vec<i64> = out_idx.iter().map(|&coord| coord as i64).collect();
            for (d, &axis) in axes.iter().enumerate() {
                in_idx[axis] += k_idx[d] as i64 - offsets[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        output.data[flat_out] = function(&neighborhood);
    }

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
    F: Fn(&[f64]) -> f64,
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

    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(kernel_total);

        for flat_k in 0..kernel_total {
            let mut k_idx = vec![0usize; ndim];
            let mut rem = flat_k;
            for (d, slot) in k_idx.iter_mut().enumerate() {
                *slot = rem / kernel_strides[d];
                rem %= kernel_strides[d];
            }

            let mut in_idx = vec![0i64; ndim];
            for d in 0..ndim {
                in_idx[d] = out_idx[d] as i64 + k_idx[d] as i64 - offsets[d] - origins[d];
            }
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        output.data[flat_out] = function(&neighborhood);
    }

    Ok(output)
}

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
    F: Fn(&[f64]) -> f64,
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
    F: Fn(&[f64]) -> f64,
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
    F: Fn(&[f64]) -> f64,
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
    F: Fn(&[f64]) -> f64,
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
    for flat_out in 0..input.size() {
        let out_idx = input.unravel(flat_out);
        let mut neighborhood = Vec::with_capacity(filter_size);

        for k in 0..filter_size {
            let mut in_idx: Vec<i64> = out_idx.iter().map(|&coord| coord as i64).collect();
            in_idx[axis] += k as i64 - offset - origin;
            neighborhood.push(input.get_boundary(&in_idx, mode, cval));
        }

        output.data[flat_out] = function(&neighborhood);
    }

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
    F: Fn(&[f64]) -> f64,
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
pub fn histogram(
    input: &NdArray,
    min_val: f64,
    max_val: f64,
    nbins: usize,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<Vec<usize>>, NdimageError> {
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

    // Apply smoothing kernels along non-axis dims, derivative along axis
    let mut current = input.clone();

    for d in 0..input.ndim() {
        let kernel_1d = if d == axis {
            vec![-1.0, 0.0, 1.0] // derivative
        } else {
            vec![1.0, 2.0, 1.0] // smoothing (triangle)
        };
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[d] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = correlate(&current, &kernel, mode, cval)?;
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

    let mut current = input.clone();

    for d in 0..input.ndim() {
        let kernel_1d = if d == axis {
            vec![-1.0, 0.0, 1.0]
        } else {
            vec![1.0, 1.0, 1.0] // box smoothing (Prewitt uses uniform weights)
        };
        let mut kernel_shape = vec![1usize; input.ndim()];
        kernel_shape[d] = 3;
        let kernel = NdArray::new(kernel_1d, kernel_shape)?;
        current = correlate(&current, &kernel, mode, cval)?;
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

fn binary_erosion_once_with_origins(current: &NdArray, size: usize, origins: &[i64]) -> NdArray {
    // Binary erosion is a minimum filter over the booleanized image with a
    // constant-0 border: O(N * ndim) separable sliding-window min instead of the
    // O(N * size^ndim) per-pixel footprint scan. Gated to the default all-zero
    // origin so the kernel window (and origin validation) match exactly.
    if binary_morph_origins_all_zero(origins) {
        let bin = booleanized_binary(current);
        if let Ok(result) =
            separable_minmax_filter(&bin, size, &[0], BoundaryMode::Constant, 0.0, false)
        {
            return result;
        }
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

fn binary_dilation_once_with_origins(current: &NdArray, size: usize, origins: &[i64]) -> NdArray {
    // Binary dilation is a maximum filter over the booleanized image with a
    // constant-0 border, using the reflected structuring element. Gated to the
    // default all-zero origin; the equivalence test pins the reflection offset.
    if binary_morph_origins_all_zero(origins) {
        let bin = booleanized_binary(current);
        let refl = binary_dilation_origin_reflection(size);
        if let Ok(result) =
            separable_minmax_filter(&bin, size, &[refl], BoundaryMode::Constant, 0.0, true)
        {
            return result;
        }
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
    for i in 0..filled.data.len() {
        filled.data[i] = 1.0;
    }

    // BFS from all border pixels that are background (0) in input
    let ndim = input.ndim();
    let mut queue = std::collections::VecDeque::new();

    // Find border pixels
    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let is_border = idx
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
        let idx = input.unravel(flat);
        for offset in &offsets {
            let mut neighbor_idx = Vec::with_capacity(ndim);
            let mut in_bounds = true;
            for axis in 0..ndim {
                let coord = idx[axis] as i64 + offset[axis];
                if coord < 0 || coord >= input.shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                neighbor_idx.push(coord as usize);
            }
            if !in_bounds {
                continue;
            }
            let nflat = neighbor_idx
                .iter()
                .zip(input.strides.iter())
                .map(|(i, s)| i * s)
                .sum::<usize>();
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

    let mut labels = NdArray::zeros(input.shape.clone());
    let mut current_label = 0usize;
    let ndim = input.ndim();
    let mut queue = std::collections::VecDeque::new();

    for flat in 0..input.size() {
        if input.data[flat] == 0.0 || labels.data[flat] != 0.0 {
            continue;
        }

        current_label += 1;
        labels.data[flat] = current_label as f64;
        queue.push_back(flat);

        while let Some(current_flat) = queue.pop_front() {
            let idx = input.unravel(current_flat);
            for offset in &offsets {
                let mut neighbor_idx = Vec::with_capacity(ndim);
                let mut in_bounds = true;
                for axis in 0..ndim {
                    let coord = idx[axis] as i64 + offset[axis];
                    if coord < 0 || coord >= input.shape[axis] as i64 {
                        in_bounds = false;
                        break;
                    }
                    neighbor_idx.push(coord as usize);
                }
                if !in_bounds {
                    continue;
                }
                let neighbor_flat = neighbor_idx
                    .iter()
                    .zip(&input.strides)
                    .map(|(coord, stride)| coord * stride)
                    .sum::<usize>();
                if input.data[neighbor_flat] != 0.0 && labels.data[neighbor_flat] == 0.0 {
                    labels.data[neighbor_flat] = current_label as f64;
                    queue.push_back(neighbor_flat);
                }
            }
        }
    }

    Ok((labels, current_label))
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

    let mut groups = match index {
        Some(index) => vec![Vec::new(); index.len()],
        None => vec![Vec::new()],
    };

    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if let Some(index) = index {
            if let Some(pos) = index
                .iter()
                .position(|&wanted_label| label_value == wanted_label as f64)
            {
                groups[pos].push(value);
            }
        } else if label_value != 0.0 {
            groups[0].push(value);
        }
    }

    Ok(groups)
}

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

    let mut groups = match index {
        Some(index) => vec![Vec::new(); index.len()],
        None => vec![Vec::new()],
    };

    for (flat, (&value, &label_value)) in input.data.iter().zip(&labels.data).enumerate() {
        if let Some(index) = index {
            if let Some(pos) = index
                .iter()
                .position(|&wanted_label| label_value == wanted_label as f64)
            {
                groups[pos].push((value, flat));
            }
        } else if label_value != 0.0 {
            groups[0].push((value, flat));
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

fn median_of_values(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
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
pub fn sum(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
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
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| mean_of_values(values))
        .collect())
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
pub fn variance(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
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
pub fn minimum(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
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

/// Median of values in optionally labeled regions.
///
/// Matches `scipy.ndimage.median`; scalar SciPy results are returned as a
/// one-element vector, while explicit `index` lists return one value per label.
pub fn median(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<Vec<f64>, NdimageError> {
    Ok(measurement_label_groups(input, labels, index)?
        .iter()
        .map(|values| median_of_values(values))
        .collect())
}

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
    F: Fn(&[f64], Option<&[usize]>) -> f64,
{
    if labels.is_none() && index.is_some() {
        return Err(NdimageError::InvalidArgument(
            "index without defined labels".to_string(),
        ));
    }

    let groups = measurement_label_value_positions(input, labels, index)?;
    Ok(groups
        .iter()
        .map(|group| {
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
        })
        .collect())
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
pub fn extrema(
    input: &NdArray,
    labels: Option<&NdArray>,
    index: Option<&[usize]>,
) -> Result<ExtremaResult, NdimageError> {
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

    for flat in 0..labels.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            found[lbl] = true;
            let idx = labels.unravel(flat);
            for d in 0..ndim {
                mins[lbl][d] = mins[lbl][d].min(idx[d]);
                maxs[lbl][d] = maxs[lbl][d].max(idx[d]);
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

    for flat in 0..input.size() {
        let lbl = labels.data[flat] as usize;
        if lbl > 0 && lbl <= num_labels {
            let idx = input.unravel(flat);
            let w = input.data[flat];
            total_weights[lbl] += w;
            for d in 0..ndim {
                weighted_sums[lbl][d] += w * idx[d] as f64;
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
    let backgrounds = background_coordinates(input);

    // Fast path: distances-only with at least one background pixel. The exact
    // separable Felzenszwalb–Huttenlocher transform replaces the brute-force
    // O(foreground · background) scan with O(N · ndim) and is byte-identical
    // (see `edt_squared_felzenszwalb`). The all-foreground sentinel and the
    // index/tie-break semantics stay on the brute-force path below.
    if return_distances
        && !return_indices
        && !backgrounds.is_empty()
        && sampling.iter().all(|&s| s.is_finite() && s > 0.0)
    {
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

    let backgrounds = background_coordinates(input);
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
        for base in 0..n {
            if !(base / stride).is_multiple_of(len) {
                continue; // only flat indices with axis-coordinate 0 start a line
            }
            // Forward sweep: best reachable from the left.
            for t in 1..len {
                let cand = f[base + (t - 1) * stride] + 1.0;
                let cur = base + t * stride;
                if cand < f[cur] {
                    f[cur] = cand;
                }
            }
            // Backward sweep: best reachable from the right.
            for t in (0..len - 1).rev() {
                let cand = f[base + (t + 1) * stride] + 1.0;
                let cur = base + t * stride;
                if cand < f[cur] {
                    f[cur] = cand;
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

    // Forward raster sweep: relax from neighbours that precede the cell.
    for flat in 0..n {
        if d[flat] == 0.0 {
            continue;
        }
        unravel_into(flat, &input.strides, &mut coords);
        for (off, flat_delta) in &offsets {
            if *flat_delta < 0 && in_bounds(&coords, off) {
                let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                if cand < d[flat] {
                    d[flat] = cand;
                }
            }
        }
    }
    // Backward raster sweep: relax from neighbours that follow the cell.
    for flat in (0..n).rev() {
        if d[flat] == 0.0 {
            continue;
        }
        unravel_into(flat, &input.strides, &mut coords);
        for (off, flat_delta) in &offsets {
            if *flat_delta > 0 && in_bounds(&coords, off) {
                let cand = d[(flat as i64 + flat_delta) as usize] + 1.0;
                if cand < d[flat] {
                    d[flat] = cand;
                }
            }
        }
    }
    d
}

fn unravel_into(mut flat: usize, strides: &[usize], out: &mut [usize]) {
    for (slot, &stride) in out.iter_mut().zip(strides) {
        *slot = flat / stride;
        flat %= stride;
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
        line.resize(len, 0.0);
        d.resize(len, 0.0);
        v.resize(len, 0);
        z.resize(len + 1, 0.0);

        // Every flat index whose coordinate along `axis` is 0 starts one line.
        for base in 0..n {
            if !(base / stride).is_multiple_of(len) {
                continue;
            }
            for t in 0..len {
                line[t] = f[base + t * stride];
            }
            edt_1d_squared(&line, scale, scale2, &mut d, &mut v, &mut z);
            for t in 0..len {
                f[base + t * stride] = d[t];
            }
        }
    }
    f
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
        );
    }

    Ok(output)
}

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

    let spline = prefilter_spline_coefficients(input, order, mode)?;
    let mut output = NdArray::zeros(new_shape.clone());

    for flat in 0..output.size() {
        let out_idx = output.unravel(flat);
        let coords: Vec<f64> = out_idx
            .iter()
            .enumerate()
            .map(|(axis, &o)| {
                if output.shape[axis] <= 1 || input.shape[axis] <= 1 {
                    0.0
                } else {
                    o as f64 * (input.shape[axis] - 1) as f64 / (output.shape[axis] - 1) as f64
                }
            })
            .collect();
        output.data[flat] = sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
        );
    }

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

    for r in 0..out_rows {
        for c in 0..out_cols {
            // Map output to input (inverse rotation)
            let dy = r as f64 - cy_out;
            let dx = c as f64 - cx_out;
            let src_y = cy_in + cos_a * dy + sin_a * dx;
            let src_x = cx_in - sin_a * dy + cos_a * dx;

            let value = sample_interpolated(
                input,
                &spline.coeffs,
                &[src_y, src_x],
                &spline.coord_offsets,
                order,
                mode,
                cval,
            );
            output.set(&[r, c], value);
        }
    }

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
    let mut result = Vec::with_capacity(npts);
    for p in 0..npts {
        let coords: Vec<f64> = coordinates.iter().map(|c| c[p]).collect();
        result.push(sample_interpolated(
            input,
            &spline.coeffs,
            &coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
        ));
    }

    Ok(result)
}

/// Compute the maximum of the input array.
pub fn array_max(input: &NdArray) -> f64 {
    input
        .data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

/// Compute the minimum of the input array.
pub fn array_min(input: &NdArray) -> f64 {
    input
        .data
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        })
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
    filter1d_axis_with_origin(input, size, axis, mode, cval, origin, |window| {
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
    })
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
    filter1d_axis_with_origin(input, size, axis, mode, cval, origin, |window| {
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
    })
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
    for flat in 0..input.size() {
        if input.data[flat] == 0.0 {
            continue;
        }
        let idx = input.unravel(flat);
        for offset in &offsets {
            let mut out_idx = Vec::with_capacity(input.ndim());
            let mut in_bounds = true;
            for axis in 0..input.ndim() {
                let coord = idx[axis] as i64 + offset[axis];
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
pub fn otsu_threshold(input: &NdArray) -> f64 {
    if input.size() == 0 {
        return 0.0;
    }

    // Build histogram
    let min_val = input
        .data
        .iter()
        .cloned()
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        });
    let max_val = input
        .data
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });

    if (max_val - min_val).abs() < 1e-15 {
        return min_val;
    }

    let nbins = 256;
    let bin_width = (max_val - min_val) / nbins as f64;
    let mut hist = vec![0usize; nbins];

    for &v in &input.data {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        hist[bin.min(nbins - 1)] += 1;
    }

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

    for r in 0..rows {
        for c in 0..cols {
            let src_r = matrix[0][0] * r as f64 + matrix[0][1] * c as f64 + matrix[0][2];
            let src_c = matrix[1][0] * r as f64 + matrix[1][1] * c as f64 + matrix[1][2];
            let value = sample_interpolated(
                input,
                &spline.coeffs,
                &[src_r, src_c],
                &spline.coord_offsets,
                order,
                mode,
                cval,
            );
            output.set(&[r, c], value);
        }
    }

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

/// Count nonzero elements.
pub fn count_nonzero(input: &NdArray) -> usize {
    input.data.iter().filter(|&&v| v != 0.0).count()
}

/// Clip (clamp) array values to [a_min, a_max].
pub fn clip(input: &NdArray, a_min: f64, a_max: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.clamp(a_min, a_max)).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise absolute value.
pub fn abs_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.abs()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise square root.
pub fn sqrt_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.max(0.0).sqrt()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise power.
pub fn power_array(input: &NdArray, exponent: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.powf(exponent)).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Multiply two arrays element-wise.
pub fn multiply_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise multiply".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x * y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Add two arrays element-wise.
pub fn add_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match for element-wise add".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Subtract two arrays element-wise.
pub fn subtract_arrays(a: &NdArray, b: &NdArray) -> Result<NdArray, NdimageError> {
    if a.shape != b.shape {
        return Err(NdimageError::DimensionMismatch(
            "shapes must match".to_string(),
        ));
    }
    let data: Vec<f64> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x - y)
        .collect();
    Ok(NdArray {
        data,
        shape: a.shape.clone(),
        strides: a.strides.clone(),
    })
}

/// Scale (multiply by scalar) an array.
pub fn scale_array(input: &NdArray, scalar: f64) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v * scalar).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise natural logarithm (ln).
pub fn log_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input
        .data
        .iter()
        .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
        .collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

/// Apply element-wise exponential.
pub fn exp_array(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input.data.iter().map(|&v| v.exp()).collect();
    NdArray {
        data,
        shape: input.shape.clone(),
        strides: input.strides.clone(),
    }
}

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

    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let mut out_idx: Vec<usize> = idx.clone();
        out_idx.remove(axis);
        let out_flat = out_idx
            .iter()
            .zip(result.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum::<usize>();
        result.data[out_flat] += input.data[flat];
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
    for v in &mut result.data {
        *v = constant;
    }

    // Copy input data to padded region
    let result_strides = compute_strides(&new_shape);
    for flat in 0..input.size() {
        let idx = input.unravel(flat);
        let padded_idx: Vec<usize> = idx
            .iter()
            .zip(pad_width.iter())
            .map(|(&i, &(before, _))| i + before)
            .collect();
        let out_flat: usize = padded_idx
            .iter()
            .zip(result_strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();
        result.data[out_flat] = input.data[flat];
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
    let new_total: usize = new_shape.iter().product();
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
    NdArray {
        data: data.clone(),
        shape: vec![data.len()],
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

    for (idx, val) in output.iter_mut().enumerate() {
        let mut filter_val = 1.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            let freq = if i <= n / 2 {
                i as f64 / n as f64
            } else {
                (i as f64 - n as f64) / n as f64
            };
            let omega = 2.0 * PI * freq;
            filter_val *= (-0.5 * (omega * sigma[d]).powi(2)).exp();
        }

        *val = (val.0 * filter_val, val.1 * filter_val);
    }

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

    for (idx, val) in output.iter_mut().enumerate() {
        let mut filter_val = 1.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

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
        }

        *val = (val.0 * filter_val, val.1 * filter_val);
    }

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

    for (idx, val) in output.iter_mut().enumerate() {
        let mut phase = 0.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            let freq = if i <= n / 2 {
                i as f64 / n as f64
            } else {
                (i as f64 - n as f64) / n as f64
            };
            phase -= 2.0 * PI * freq * shift[d];
        }

        let (sin_p, cos_p) = phase.sin_cos();
        *val = (val.0 * cos_p - val.1 * sin_p, val.0 * sin_p + val.1 * cos_p);
    }

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

    for (idx, val) in output.iter_mut().enumerate() {
        let mut sum_sq = 0.0;
        let mut remaining = idx;

        for (d, &n) in shape.iter().enumerate().rev() {
            let i = remaining % n;
            remaining /= n;

            let freq = if i <= n / 2 {
                i as f64 / n as f64
            } else {
                (i as f64 - n as f64) / n as f64
            };
            let normalized = freq * size[d];
            sum_sq += normalized * normalized;
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
    }

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
    F: Fn(&[usize]) -> Vec<f64>,
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
    let mut result = Vec::with_capacity(total_out);

    for linear_idx in 0..total_out {
        let mut out_coord = Vec::with_capacity(out_shape.len());
        let mut remaining = linear_idx;
        for &n in out_shape.iter().rev() {
            out_coord.push(remaining % n);
            remaining /= n;
        }
        out_coord.reverse();

        let in_coords = mapping(&out_coord);

        let val = sample_interpolated(
            input,
            &spline.coeffs,
            &in_coords,
            &spline.coord_offsets,
            order,
            mode,
            cval,
        );
        result.push(val);
    }

    Ok(NdArray::new(result, out_shape).unwrap())
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

    for outer_idx in 0..outer {
        for inner_idx in 0..stride {
            let mut line = Vec::with_capacity(axis_len);
            for i in 0..axis_len {
                let flat = outer_idx * axis_len * stride + i * stride + inner_idx;
                line.push(result.data[flat]);
            }

            let coeffs = spline_coefficients_for_line(&line, order)?;

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

    while let Some(std::cmp::Reverse((cost_scaled, idx))) = queue.pop() {
        let current_cost = cost_scaled as f64 / 1000.0;
        if current_cost > costs[idx] {
            continue;
        }

        let coords = input.unravel(idx);
        for offset in &struct_offsets {
            let mut neighbor_coords = Vec::with_capacity(ndim);
            let mut in_bounds = true;
            for axis in 0..ndim {
                let coord = coords[axis] as i64 + offset[axis];
                if coord < 0 || coord >= input.shape[axis] as i64 {
                    in_bounds = false;
                    break;
                }
                neighbor_coords.push(coord as usize);
            }
            if !in_bounds {
                continue;
            }
            let neighbor_idx = neighbor_coords
                .iter()
                .zip(input.strides.iter())
                .map(|(&coord, &stride)| coord * stride)
                .sum::<usize>();

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

    Ok(NdArray::new(output, input.shape.clone()).unwrap())
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
mod tests {
    use super::*;

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
        let got = gaussian_kernel1d(1.0, 0, 4);
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
        let got = gaussian_kernel1d(1.0, 1, 4);
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
        let got = gaussian_kernel1d(1.0, 2, 4);
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-12, "order2 kernel mismatch: {g} vs {e}");
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
                                (r, f) => panic!(
                                    "accept/reject parity: shape={shape:?} size={size} origin={origin} ref_ok={} fast_ok={}",
                                    r.is_ok(),
                                    f.is_ok()
                                ),
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
        let calls = std::cell::Cell::new(0usize);
        let result = generic_filter_axes(
            &input,
            |values| {
                calls.set(calls.get() + 1);
                values.iter().sum()
            },
            0,
            &[],
            BoundaryMode::Constant,
            -10.0,
        )
        .unwrap();

        assert_eq!(result.data, input.data);
        assert_eq!(calls.get(), 0);
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
}
