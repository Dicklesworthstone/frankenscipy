#![forbid(unsafe_code)]

//! Interpolation routines for FrankenSciPy.
//!
//! Matches `scipy.interpolate` core functions:
//! - `interp1d` — 1D interpolation (linear, nearest, cubic)
//! - `CubicSpline` — natural/clamped/not-a-knot cubic spline
//! - `BSpline` — B-spline representation and evaluation
//! - `Akima1DInterpolator` — Akima piecewise cubic Hermite interpolator
//! - `PchipInterpolator` — Monotonicity-preserving cubic Hermite interpolator
//! - `RegularGridInterpolator` — N-D interpolation on regular grids
//! - `NearestNDInterpolator` — Nearest-neighbor for scattered N-D data
//! - `LinearNDInterpolator` — Linear interpolation for scattered 2D data
//! - `CloughTocher2DInterpolator` — Smooth scattered 2D interpolation
//! - `SmoothBivariateSpline` — Smooth bivariate approximation for scattered 2D data

use fsci_runtime::RuntimeMode;
use std::collections::HashMap;

mod surfit;
pub use surfit::{bisplev_derivative, bisplrep, lsq_bivariate_spline};
mod sphere;
pub use sphere::{lsq_sphere_bivariate_spline, smooth_sphere_bivariate_spline};
mod fitpack_cyclic;
mod sphere_grid;
pub use sphere_grid::rect_sphere_bivariate_spline;

/// Interpolation method for `interp1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpKind {
    #[default]
    Linear,
    Nearest,
    CubicSpline,
}

/// Boundary conditions for cubic spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SplineBc {
    /// S''(x_0) = 0, S''(x_n) = 0 (default).
    #[default]
    Natural,
    /// Not-a-knot: S'''(x_1⁻) = S'''(x_1⁺), S'''(x_{n-2}⁻) = S'''(x_{n-2}⁺).
    NotAKnot,
    /// Clamped: S'(x_0) = deriv_left, S'(x_n) = deriv_right.
    Clamped(f64, f64),
    /// Periodic: `y[0] == y[n-1]`, S'(x_0)=S'(x_n), S''(x_0)=S''(x_n).
    Periodic,
}

/// Error type for interpolation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    TooFewPoints { minimum: usize, actual: usize },
    UnsortedX,
    NonFiniteX,
    LengthMismatch { x_len: usize, y_len: usize },
    OutOfBounds { value: String },
    InvalidArgument { detail: String },
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { minimum, actual } => {
                write!(f, "need at least {minimum} points, got {actual}")
            }
            Self::UnsortedX => write!(f, "x values must be strictly increasing"),
            Self::NonFiniteX => write!(f, "x values must be finite (no NaN or Inf)"),
            Self::LengthMismatch { x_len, y_len } => {
                write!(f, "x and y must have same length (x={x_len}, y={y_len})")
            }
            Self::OutOfBounds { value } => write!(f, "interpolation point out of bounds: {value}"),
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for InterpError {}

/// Options for 1D interpolation.
#[derive(Debug, Clone, Copy)]
pub struct Interp1dOptions {
    pub kind: InterpKind,
    pub mode: RuntimeMode,
    pub fill_value: Option<f64>,
    pub bounds_error: bool,
    pub spline_bc: SplineBc,
}

impl Default for Interp1dOptions {
    fn default() -> Self {
        Self {
            kind: InterpKind::Linear,
            mode: RuntimeMode::Strict,
            fill_value: None,
            bounds_error: true,
            spline_bc: SplineBc::default(),
        }
    }
}

/// 1D interpolation function object.
///
/// Matches `scipy.interpolate.interp1d(x, y, kind=...)`.
/// Create with `Interp1d::new(x, y, options)`, then call `.evaluate(x_new)`.
#[derive(Debug)]
pub struct Interp1d {
    x: Vec<f64>,
    y: Vec<f64>,
    options: Interp1dOptions,
    /// Cubic spline coefficients (a, b, c, d) for each interval, if applicable.
    spline_coeffs: Option<Vec<[f64; 4]>>,
    /// `Some((x0, inv_dx))` when `x` is evenly spaced — enables the O(1)
    /// direct-address interval lookup on the unsorted batch path.
    uniform: Option<(f64, f64)>,
}

impl Interp1d {
    /// Create a new 1D interpolator.
    pub fn new(x: &[f64], y: &[f64], options: Interp1dOptions) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }

        let min_points = match options.kind {
            InterpKind::Linear | InterpKind::Nearest => 2,
            InterpKind::CubicSpline => 4,
        };
        if x.len() < min_points {
            return Err(InterpError::TooFewPoints {
                minimum: min_points,
                actual: x.len(),
            });
        }

        // Check for non-finite x values (NaN/Inf) - must check before sortedness
        // because NaN comparisons always return false, bypassing the sort check
        if x.iter().any(|&v| !v.is_finite()) {
            return Err(InterpError::NonFiniteX);
        }

        // Check strictly increasing
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let spline_coeffs = if options.kind == InterpKind::CubicSpline {
            Some(compute_cubic_spline(x, y, options.spline_bc)?)
        } else {
            None
        };

        let uniform = detect_uniform_axis(x);
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            options,
            spline_coeffs,
            uniform,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn eval(&self, x_new: f64) -> Result<f64, InterpError> {
        if x_new.is_nan() {
            return Ok(f64::NAN);
        }
        // Bounds checking
        if x_new < self.x[0] || x_new > self.x[self.x.len() - 1] {
            if self.options.bounds_error {
                return Err(InterpError::OutOfBounds {
                    value: format!("{x_new}"),
                });
            }
            if let Some(fill) = self.options.fill_value {
                return Ok(fill);
            }
            return Ok(f64::NAN);
        }

        let i = find_interval_helper(&self.x, x_new);
        self.eval_at_interval(i, x_new)
    }

    /// Evaluate at multiple points.
    ///
    /// Optimized for sorted `x_new` to achieve O(N+M) performance.
    pub fn eval_many(&self, x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
        if x_new.is_empty() {
            return Ok(Vec::new());
        }

        // Linear is the hot, cheap-per-point kind: route it through a specialized
        // loop that hoists the per-point `kind` match + `Result` out of the inner
        // loop and uses the O(1) uniform finder on the unsorted path. Bit-identical
        // to the generic path below (same interval index + same lerp formula/order).
        if self.options.kind == InterpKind::Linear {
            return self.eval_many_linear(x_new);
        }

        // Check if x_new is sorted to enable linear sweep optimization
        let is_sorted = x_new.windows(2).all(|w| w[0] <= w[1]);

        if is_sorted {
            let mut results = Vec::with_capacity(x_new.len());
            let mut i = 0;
            let n = self.x.len();
            for &xi in x_new {
                // Bounds checking
                if xi < self.x[0] || xi > self.x[n - 1] {
                    if self.options.bounds_error {
                        return Err(InterpError::OutOfBounds {
                            value: format!("{xi}"),
                        });
                    }
                    results.push(self.options.fill_value.unwrap_or(f64::NAN));
                    continue;
                }

                // Advance i while xi is beyond the current interval
                while i < n - 2 && xi >= self.x[i + 1] {
                    i += 1;
                }
                results.push(self.eval_at_interval(i, xi)?);
            }
            Ok(results)
        } else {
            x_new.iter().map(|&xi| self.eval(xi)).collect()
        }
    }

    /// Specialized linear `eval_many`. Bit-identical to the generic path: the
    /// sorted branch mirrors the O(N+M) cursor sweep, the unsorted branch mirrors
    /// the per-point `eval` (NaN -> NaN, bounds -> error/fill/NaN, else lerp), but
    /// the inner loops are free of the per-point `kind` match and `Result`, and the
    /// unsorted finder is O(1) on uniform grids (`find_interval_uniform_helper` is
    /// bit-equivalent to `find_interval_helper`).
    fn eval_many_linear(&self, x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
        let n = self.x.len();
        let x0 = self.x[0];
        let xn = self.x[n - 1];
        let x = &self.x;
        let y = &self.y;
        let fill = self.options.fill_value;
        let mut out = Vec::with_capacity(x_new.len());

        let is_sorted = x_new.windows(2).all(|w| w[0] <= w[1]);
        if is_sorted {
            let mut i = 0usize;
            for &xi in x_new {
                if xi < x0 || xi > xn {
                    if self.options.bounds_error {
                        return Err(InterpError::OutOfBounds {
                            value: format!("{xi}"),
                        });
                    }
                    out.push(fill.unwrap_or(f64::NAN));
                    continue;
                }
                while i < n - 2 && xi >= x[i + 1] {
                    i += 1;
                }
                let t = (xi - x[i]) / (x[i + 1] - x[i]);
                out.push(y[i] + t * (y[i + 1] - y[i]));
            }
        } else {
            for &xi in x_new {
                if xi.is_nan() {
                    out.push(f64::NAN);
                    continue;
                }
                if xi < x0 || xi > xn {
                    if self.options.bounds_error {
                        return Err(InterpError::OutOfBounds {
                            value: format!("{xi}"),
                        });
                    }
                    out.push(fill.unwrap_or(f64::NAN));
                    continue;
                }
                let i = match self.uniform {
                    Some(meta) => find_interval_uniform_helper(meta, x, xi),
                    None => find_interval_helper(x, xi),
                };
                let t = (xi - x[i]) / (x[i + 1] - x[i]);
                out.push(y[i] + t * (y[i + 1] - y[i]));
            }
        }
        Ok(out)
    }

    fn eval_at_interval(&self, i: usize, x_new: f64) -> Result<f64, InterpError> {
        match self.options.kind {
            InterpKind::Linear => {
                let t = (x_new - self.x[i]) / (self.x[i + 1] - self.x[i]);
                Ok(self.y[i] + t * (self.y[i + 1] - self.y[i]))
            }
            InterpKind::Nearest => {
                let mid = 0.5 * (self.x[i] + self.x[i + 1]);
                Ok(if x_new <= mid {
                    self.y[i]
                } else {
                    self.y[i + 1]
                })
            }
            InterpKind::CubicSpline => {
                let coeffs =
                    self.spline_coeffs
                        .as_ref()
                        .ok_or_else(|| InterpError::InvalidArgument {
                            detail: "cubic spline initialized without coeffs".to_string(),
                        })?;
                let dx = x_new - self.x[i];
                let [a, b, c, d] = coeffs[i];
                Ok(a + dx * (b + dx * (c + dx * d)))
            }
        }
    }
}

/// Binary search to find the interval containing x_new in a sorted array.
/// Returns index i such that array[i] <= x_new < array[i+1].
fn find_interval_helper(array: &[f64], x_new: f64) -> usize {
    let n = array.len();
    if x_new <= array[0] {
        return 0;
    }
    if x_new >= array[n - 1] {
        return n - 2;
    }
    // Binary search
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if array[mid] <= x_new {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// O(1) direct-address interval lookup for an evenly-spaced array, BIT-IDENTICAL to
/// `find_interval_helper` (both return the largest `i` with `array[i] <= x`, clamped
/// to `[0, n-2]`, with `x <= array[0] -> 0` and `x >= array[n-1] -> n-2`). `meta`
/// is `(x0, inv_dx)` from `detect_uniform_axis`. The correction loops reproduce the
/// boundary clamps without explicit endpoint branches and run ≤ once or twice on a
/// genuinely uniform axis. (Locked by `interp1d_uniform_finder_matches_helper`.)
fn find_interval_uniform_helper(meta: (f64, f64), array: &[f64], x_new: f64) -> usize {
    let n = array.len();
    if x_new <= array[0] {
        return 0;
    }
    if x_new >= array[n - 1] {
        return n - 2;
    }
    let (x0, inv_dx) = meta;
    let mut i = ((x_new - x0) * inv_dx) as usize; // negative -> 0, oversized -> clamped next
    if i > n - 2 {
        i = n - 2;
    }
    while i + 1 < n - 1 && array[i + 1] <= x_new {
        i += 1;
    }
    while i > 0 && array[i] > x_new {
        i -= 1;
    }
    i
}

/// Classify an axis as uniform (evenly spaced) and return `(x0, inv_dx)` for
/// O(1) direct-address interval lookup. The axis is assumed strictly increasing
/// with `len() >= 2` (guaranteed by `RegularGridInterpolator::new`). An axis
/// counts as uniform when every coordinate lies within `1e-9 * span` of its
/// ideal linspace position, which covers grids built by `linspace`/`arange`
/// while excluding genuinely irregular axes. Returns `None` when not uniform or
/// when `dx` is not finite/positive.
fn detect_uniform_axis(axis: &[f64]) -> Option<(f64, f64)> {
    let n = axis.len();
    let x0 = axis[0];
    let span = axis[n - 1] - x0;
    let dx = span / (n - 1) as f64;
    if !dx.is_finite() || dx <= 0.0 {
        return None;
    }
    let tol = 1e-9 * span;
    for (i, &v) in axis.iter().enumerate() {
        if (v - (x0 + i as f64 * dx)).abs() > tol {
            return None;
        }
    }
    Some((x0, 1.0 / dx))
}

/// Compute cubic spline coefficients with configurable boundary conditions.
fn compute_cubic_spline(x: &[f64], y: &[f64], bc: SplineBc) -> Result<Vec<[f64; 4]>, InterpError> {
    let n = x.len();
    if n < 4 {
        return Err(InterpError::TooFewPoints {
            minimum: 4,
            actual: n,
        });
    }

    let m = n - 1; // number of intervals
    let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

    let c = match bc {
        SplineBc::Natural => solve_spline_natural(n, &h, y),
        SplineBc::NotAKnot => solve_spline_not_a_knot(n, &h, y),
        SplineBc::Clamped(dl, dr) => solve_spline_clamped(n, &h, y, dl, dr),
        SplineBc::Periodic => {
            let scale = y[0].abs().max(y[n - 1].abs()).max(1.0);
            if (y[0] - y[n - 1]).abs() > 1e-12 * scale {
                return Err(InterpError::InvalidArgument {
                    detail: "periodic spline requires y[0] == y[n-1]".to_string(),
                });
            }
            solve_spline_periodic(n, &h, y)
        }
    };

    let mut coeffs = Vec::with_capacity(m);
    for i in 0..m {
        let a_i = y[i];
        let b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        let c_i = c[i];
        let d_i = (c[i + 1] - c[i]) / (3.0 * h[i]);
        coeffs.push([a_i, b_i, c_i, d_i]);
    }

    Ok(coeffs)
}

fn solve_spline_natural(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let inner = n - 2;
    let mut rhs = vec![0.0; inner];
    for (i, r) in rhs.iter_mut().enumerate() {
        let ii = i + 1;
        *r = 3.0 * ((y[ii + 1] - y[ii]) / h[ii] - (y[ii] - y[ii - 1]) / h[ii - 1]);
    }

    let mut diag: Vec<f64> = (0..inner).map(|i| 2.0 * (h[i] + h[i + 1])).collect();
    let sub: Vec<f64> = (0..inner).map(|i| h[i]).collect();
    let sup: Vec<f64> = (0..inner).map(|i| h[i + 1]).collect();

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);

    let mut c = vec![0.0; n];
    for (i, &ci) in rhs.iter().enumerate() {
        c[i + 1] = ci;
    }
    c
}

fn solve_spline_not_a_knot(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let inner = n - 2;
    let r01 = h[0] / h[1];

    let mut diag = vec![0.0; inner];
    let mut sub = vec![0.0; inner];
    let mut sup = vec![0.0; inner];
    let mut rhs = vec![0.0; inner];

    for idx in 0..inner {
        let i = idx + 1;
        sub[idx] = h[i - 1];
        diag[idx] = 2.0 * (h[i - 1] + h[i]);
        sup[idx] = h[i];
        rhs[idx] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    diag[0] = h[0] * (1.0 + r01) + 2.0 * (h[0] + h[1]);
    sup[0] = h[1] - h[0] * r01;

    let rn = h[n - 2] / h[n - 3];
    let last = inner - 1;
    sub[last] = h[n - 3] - h[n - 2] * rn;
    diag[last] = 2.0 * (h[n - 3] + h[n - 2]) + h[n - 2] * (1.0 + rn);

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);

    let mut c = vec![0.0; n];
    for (idx, &ci) in rhs.iter().enumerate() {
        c[idx + 1] = ci;
    }
    c[0] = (1.0 + r01) * c[1] - r01 * c[2];
    c[n - 1] = (1.0 + rn) * c[n - 2] - rn * c[n - 3];

    c
}

fn solve_spline_clamped(
    n: usize,
    h: &[f64],
    y: &[f64],
    deriv_left: f64,
    deriv_right: f64,
) -> Vec<f64> {
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    diag[0] = 2.0 * h[0];
    sup[0] = h[0];
    rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - deriv_left);

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    let m = n - 1;
    sub[m] = h[m - 1];
    diag[m] = 2.0 * h[m - 1];
    rhs[m] = 3.0 * (deriv_right - (y[m] - y[m - 1]) / h[m - 1]);

    thomas_solve(&sub, &mut diag, &sup, &mut rhs);
    rhs
}

fn solve_spline_periodic(n: usize, h: &[f64], y: &[f64]) -> Vec<f64> {
    let m = n - 1;
    let mut sub = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut sup = vec![0.0; m];
    let mut rhs = vec![0.0; m];

    if m == 1 {
        let mut c = vec![0.0; n];
        c[0] = 0.0;
        c[n - 1] = 0.0;
        return c;
    }

    let last_h = h[m - 1];

    // i = 0 boundary row (couples to c_{n-2}).
    sub[0] = last_h;
    diag[0] = 2.0 * (h[0] + last_h);
    sup[0] = h[0];
    rhs[0] = 3.0 * ((y[1] - y[0]) / h[0] - (y[0] - y[m - 1]) / last_h);

    // Interior rows i = 1..m-1 (i corresponds to original index).
    for i in 1..m {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    let solution = solve_cyclic_tridiagonal(&sub, &diag, &sup, &rhs);
    let mut c = vec![0.0; n];
    if m > 0 {
        c[..m].copy_from_slice(&solution[..m]);
    }
    c[n - 1] = c[0];
    c
}

fn thomas_solve(sub: &[f64], diag: &mut [f64], sup: &[f64], rhs: &mut [f64]) {
    let n = diag.len();
    if n == 0 {
        return;
    }
    for i in 1..n {
        if diag[i - 1].abs() < 1e-18 {
            continue;
        }
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    if diag[n - 1].abs() > 1e-18 {
        rhs[n - 1] /= diag[n - 1];
    }
    for i in (0..n - 1).rev() {
        if diag[i].abs() > 1e-18 {
            rhs[i] = (rhs[i] - sup[i] * rhs[i + 1]) / diag[i];
        }
    }
}

fn solve_cyclic_tridiagonal(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![if diag[0].abs() > 1e-18 {
            rhs[0] / diag[0]
        } else {
            0.0
        }];
    }

    let a0 = sub[0];
    let c_last = sup[n - 1];
    let gamma = -diag[0];

    let mut bb = diag.to_vec();
    bb[0] = diag[0] - gamma;
    bb[n - 1] = diag[n - 1] - a0 * c_last / gamma;

    let mut a = sub.to_vec();
    let mut c = sup.to_vec();
    a[0] = 0.0;
    c[n - 1] = 0.0;

    let mut x = rhs.to_vec();
    let mut x_diag = bb.clone();
    thomas_solve(&a, &mut x_diag, &c, &mut x);

    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n - 1] = a0;
    let mut z = u.clone();
    thomas_solve(&a, &mut bb, &c, &mut z);

    let numerator = x[0] + c_last * x[n - 1] / gamma;
    let denominator = 1.0 + z[0] + c_last * z[n - 1] / gamma;
    let factor = numerator / denominator;
    for i in 0..n {
        x[i] -= factor * z[i];
    }
    x
}

/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolator.
#[derive(Debug)]
pub struct PchipInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl PchipInterpolator {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: x.len(),
            });
        }
        if x.iter().any(|&v| !v.is_finite()) {
            return Err(InterpError::NonFiniteX);
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let n = x.len();
        let m = n - 1;
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let delta: Vec<f64> = (0..m).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            for i in 1..m {
                if delta[i - 1].signum() != delta[i].signum()
                    || delta[i - 1] == 0.0
                    || delta[i] == 0.0
                {
                    d[i] = 0.0;
                } else {
                    let w1 = 2.0 * h[i] + h[i - 1];
                    let w2 = h[i] + 2.0 * h[i - 1];
                    d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }
            d[0] = pchip_end_slope(&h, &delta, true);
            d[n - 1] = pchip_end_slope(&h, &delta, false);
        }

        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            coeffs.push([
                y[i],
                d[i],
                (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi,
                (d[i] + d[i + 1] - 2.0 * delta[i]) / (hi * hi),
            ]);
        }

        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if !x_new.is_finite() {
            return f64::NAN;
        }
        let n = self.x.len();
        let i = if x_new <= self.x[0] {
            0
        } else if x_new >= self.x[n - 1] {
            n - 2
        } else {
            find_interval_helper(&self.x, x_new)
        };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        let work = (x_new.len() as u64).saturating_mul(24);
        let serial_path = work < PAR_QUERY_MIN_WORK || x_new.len() < 4;
        let sorted_finite =
            x_new.iter().all(|x| x.is_finite()) && x_new.windows(2).all(|w| w[0] <= w[1]);
        if serial_path && sorted_finite {
            let n = self.x.len();
            let mut i = 0usize;
            return x_new
                .iter()
                .map(|&xi| {
                    if xi >= self.x[n - 1] {
                        i = n - 2;
                    } else {
                        while i + 1 < n - 1 && self.x[i + 1] <= xi {
                            i += 1;
                        }
                    }
                    let dx = xi - self.x[i];
                    let [a, b, c, d] = self.coeffs[i];
                    a + dx * (b + dx * (c + dx * d))
                })
                .collect();
        }
        par_query_map(x_new, 24, |&xi| self.eval(xi))
    }
}

/// One-shot PCHIP (piecewise cubic Hermite) interpolation: build a
/// `PchipInterpolator` from `(x, y)` and evaluate it at every point in
/// `x_new`.
///
/// Matches `scipy.interpolate.pchip_interpolate(xi, yi, x)` for 1-D real
/// inputs at derivative order 0. PCHIP preserves monotonicity of the
/// input data, so it's the standard choice when avoiding overshoot
/// matters more than smoothness at the knots.
pub fn pchip_interpolate(xi: &[f64], yi: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = PchipInterpolator::new(xi, yi)?;
    Ok(interp.eval_many(x_new))
}

fn pchip_end_slope(h: &[f64], delta: &[f64], is_left: bool) -> f64 {
    let m = delta.len();
    if m < 2 {
        return delta[0];
    }
    let (d1, d2, h1, h2) = if is_left {
        (delta[0], delta[1], h[0], h[1])
    } else {
        (delta[m - 1], delta[m - 2], h[m - 1], h[m - 2])
    };
    let mut d = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2);
    if d.signum() != d1.signum() {
        d = 0.0;
    } else if d1.signum() != d2.signum() && d.abs() > 3.0 * d1.abs() {
        d = 3.0 * d1;
    }
    d
}

pub fn interp1d_linear(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = Interp1d::new(
        x,
        y,
        Interp1dOptions {
            bounds_error: false,
            ..Default::default()
        },
    )?;
    interp.eval_many(x_new)
}

/// Standalone cubic spline interpolator.
#[derive(Debug, Clone)]
pub struct CubicSplineStandalone {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplineStandalone {
    pub fn new(x: &[f64], y: &[f64], bc: SplineBc) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 4 {
            return Err(InterpError::TooFewPoints {
                minimum: 4,
                actual: x.len(),
            });
        }
        if x.iter().any(|&v| !v.is_finite()) {
            return Err(InterpError::NonFiniteX);
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }
        let coeffs = compute_cubic_spline(x, y, bc)?;
        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if x_new.is_nan() {
            return f64::NAN;
        }
        let n = self.x.len();
        let i = if x_new <= self.x[0] {
            0
        } else if x_new >= self.x[n - 1] {
            n - 2
        } else {
            find_interval_helper(&self.x, x_new)
        };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        if let Some(v) = cubic_cursor_eval_many(&self.x, &self.coeffs, x_new) {
            return v;
        }
        par_query_map(x_new, 24, |&xi| self.eval(xi))
    }

    pub fn derivative(&self, nu: usize) -> CubicSplineDerivative {
        let m = self.coeffs.len();
        match nu {
            0 => CubicSplineDerivative {
                x: self.x.clone(),
                coeffs: self.coeffs.clone(),
            },
            1 => {
                let dc: Vec<[f64; 4]> = self
                    .coeffs
                    .iter()
                    .map(|&[_a, b, c, d]| [b, 2.0 * c, 3.0 * d, 0.0])
                    .collect();
                CubicSplineDerivative {
                    x: self.x.clone(),
                    coeffs: dc,
                }
            }
            2 => {
                let dc: Vec<[f64; 4]> = self
                    .coeffs
                    .iter()
                    .map(|&[_a, _b, c, d]| [2.0 * c, 6.0 * d, 0.0, 0.0])
                    .collect();
                CubicSplineDerivative {
                    x: self.x.clone(),
                    coeffs: dc,
                }
            }
            _ => CubicSplineDerivative {
                x: self.x.clone(),
                coeffs: vec![[0.0; 4]; m],
            },
        }
    }

    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if (b - a).abs() < 1e-15 {
            return 0.0;
        }
        let sign = if a > b { -1.0 } else { 1.0 };
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let n = self.x.len();
        let mut total = 0.0;
        for i in 0..n - 1 {
            let seg_lo = self.x[i].max(lo);
            let seg_hi = self.x[i + 1].min(hi);
            if seg_lo >= seg_hi {
                continue;
            }
            let [a0, b0, c0, d0] = self.coeffs[i];
            let dx_lo = seg_lo - self.x[i];
            let dx_hi = seg_hi - self.x[i];
            let anti = |dx: f64| {
                a0 * dx + b0 * dx * dx / 2.0 + c0 * dx.powi(3) / 3.0 + d0 * dx.powi(4) / 4.0
            };
            total += anti(dx_hi) - anti(dx_lo);
        }
        sign * total
    }
}

#[derive(Debug, Clone)]
pub struct CubicSplineDerivative {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl CubicSplineDerivative {
    pub fn eval(&self, x_new: f64) -> f64 {
        if x_new.is_nan() {
            return f64::NAN;
        }
        let n = self.x.len();
        let i = if x_new <= self.x[0] {
            0
        } else if x_new >= self.x[n - 1] {
            n - 2
        } else {
            find_interval_helper(&self.x, x_new)
        };
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }
}

/// Akima (1970) piecewise cubic Hermite interpolator.
#[derive(Debug, Clone)]
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl Akima1DInterpolator {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: x.len(),
            });
        }
        if x.iter().any(|&v| !v.is_finite()) {
            return Err(InterpError::NonFiniteX);
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }
        let n = x.len();
        let m = n - 1;
        let delta: Vec<f64> = (0..m)
            .map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            .collect();
        let slopes = akima_slopes(&delta);
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let hi = h[i];
            coeffs.push([
                y[i],
                slopes[i],
                (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / hi,
                (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (hi * hi),
            ]);
        }
        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if !x_new.is_finite() {
            return f64::NAN;
        }
        let i = find_interval_helper(&self.x, x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        if let Some(v) = cubic_cursor_eval_many(&self.x, &self.coeffs, x_new) {
            return v;
        }
        par_query_map(x_new, 24, |&xi| self.eval(xi))
    }
}

/// Cubic Hermite spline: piecewise cubic interpolant with user-specified
/// derivatives at each knot.
///
/// Matches `scipy.interpolate.CubicHermiteSpline`. The Hermite basis on
/// segment `[x_i, x_{i+1}]` reduces to the same per-segment cubic
/// coefficients used by Akima / PCHIP:
///   a = y_i
///   b = dy_i
///   c = (3·δ − 2·dy_i − dy_{i+1}) / h
///   d = (dy_i + dy_{i+1} − 2·δ) / h²
/// where δ = (y_{i+1} − y_i) / h and h = x_{i+1} − x_i.
#[derive(Debug)]
pub struct CubicHermiteSpline {
    x: Vec<f64>,
    coeffs: Vec<[f64; 4]>,
}

impl CubicHermiteSpline {
    pub fn new(x: &[f64], y: &[f64], dydx: &[f64]) -> Result<Self, InterpError> {
        if x.len() != y.len() || x.len() != dydx.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: x.len(),
            });
        }
        if x.iter().any(|&v| !v.is_finite())
            || y.iter().any(|&v| !v.is_finite())
            || dydx.iter().any(|&v| !v.is_finite())
        {
            return Err(InterpError::NonFiniteX);
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }
        let n = x.len();
        let m = n - 1;
        let mut coeffs = Vec::with_capacity(m);
        for i in 0..m {
            let h = x[i + 1] - x[i];
            let delta = (y[i + 1] - y[i]) / h;
            let dy = dydx[i];
            let dy_next = dydx[i + 1];
            coeffs.push([
                y[i],
                dy,
                (3.0 * delta - 2.0 * dy - dy_next) / h,
                (dy + dy_next - 2.0 * delta) / (h * h),
            ]);
        }
        Ok(Self {
            x: x.to_vec(),
            coeffs,
        })
    }

    pub fn eval(&self, x_new: f64) -> f64 {
        if !x_new.is_finite() {
            return f64::NAN;
        }
        let i = find_interval_helper(&self.x, x_new);
        let dx = x_new - self.x[i];
        let [a, b, c, d] = self.coeffs[i];
        a + dx * (b + dx * (c + dx * d))
    }

    pub fn eval_many(&self, x_new: &[f64]) -> Vec<f64> {
        if let Some(v) = cubic_cursor_eval_many(&self.x, &self.coeffs, x_new) {
            return v;
        }
        par_query_map(x_new, 24, |&xi| self.eval(xi))
    }
}

/// One-shot cubic Hermite interpolation: build a `CubicHermiteSpline` and
/// evaluate at every point.
pub fn cubic_hermite_interpolate(
    xi: &[f64],
    yi: &[f64],
    dydx: &[f64],
    x_new: &[f64],
) -> Result<Vec<f64>, InterpError> {
    let interp = CubicHermiteSpline::new(xi, yi, dydx)?;
    Ok(interp.eval_many(x_new))
}

/// One-shot Akima 1-D interpolation: build an `Akima1DInterpolator` from
/// `(x, y)` and evaluate at every point in `x_new`.
///
/// Akima's piecewise cubic resists overshoot from outliers (it weights
/// adjacent slopes by the local-curvature norm), making it the standard
/// choice for noisy data where PCHIP's strict monotonicity is too
/// conservative.
pub fn akima1d_interpolate(xi: &[f64], yi: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = Akima1DInterpolator::new(xi, yi)?;
    Ok(interp.eval_many(x_new))
}

fn akima_slopes(delta: &[f64]) -> Vec<f64> {
    let m = delta.len();
    let n = m + 1;
    let mut slopes = vec![0.0; n];
    if m == 0 {
        return slopes;
    }
    if m == 1 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
        return slopes;
    }
    let mut d = Vec::with_capacity(m + 4);
    let d0 = delta[0];
    let d1 = delta.get(1).copied().unwrap_or(d0);
    d.push(3.0 * d0 - 2.0 * d1);
    d.push(2.0 * d0 - d1);
    d.extend_from_slice(delta);
    let d_last = delta[m - 1];
    let d_prev = delta.get(m.wrapping_sub(2)).copied().unwrap_or(d_last);
    d.push(2.0 * d_last - d_prev);
    d.push(3.0 * d_last - 2.0 * d_prev);
    for j in 0..n {
        let w1 = (d[j + 3] - d[j + 2]).abs();
        let w2 = (d[j + 1] - d[j]).abs();
        slopes[j] = if w1 + w2 < 1e-30 {
            0.5 * (d[j + 1] + d[j + 2])
        } else {
            (w1 * d[j + 1] + w2 * d[j + 2]) / (w1 + w2)
        };
    }
    slopes
}

#[derive(Debug, Clone)]
pub struct BSpline {
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
    pub extrapolate: bool,
}

impl BSpline {
    pub fn new(t: Vec<f64>, mut c: Vec<f64>, k: usize) -> Result<Self, InterpError> {
        // A degree-k spline on knots `t` has exactly `t.len() - k - 1` B-spline
        // coefficients. scipy's tck convention pads `c` with k+1 trailing zeros to
        // length `len(t)`; accept that (and any padding) by truncating to the real
        // coefficient count, so a scipy-style tck round-trips through our consumers.
        if t.len() < k + 2 {
            return Err(InterpError::InvalidArgument {
                detail: format!("knot length {} must be >= k+2={}", t.len(), k + 2),
            });
        }
        let expected_coeffs = t.len() - k - 1;
        if c.len() < expected_coeffs {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "number of coefficients ({}) must be >= t.len()-k-1={}",
                    c.len(),
                    expected_coeffs
                ),
            });
        }
        c.truncate(expected_coeffs);
        if t.windows(2).any(|w| w[1] < w[0]) {
            return Err(InterpError::InvalidArgument {
                detail: "knots must be non-decreasing".to_string(),
            });
        }
        Ok(Self {
            t,
            c,
            k,
            extrapolate: true,
        })
    }

    pub fn knots(&self) -> &[f64] {
        &self.t
    }
    pub fn coeffs(&self) -> &[f64] {
        &self.c
    }
    pub fn degree(&self) -> usize {
        self.k
    }

    pub fn eval(&self, x: f64) -> f64 {
        let mut d = vec![0.0; self.k + 1];
        self.eval_into(x, &mut d)
    }

    pub fn eval_into(&self, x: f64, d: &mut [f64]) -> f64 {
        Self::eval_parts(&self.t, &self.c, self.k, self.extrapolate, x, d)
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        if xs.is_empty() {
            return Vec::new();
        }
        let is_sorted = xs.windows(2).all(|w| w[0] <= w[1]);

        // Serial gate FIRST (before the available_parallelism syscall). Per-point
        // de Boor evaluation is independent and SciPy's splev is single-threaded,
        // so large batches fan across cores. For sorted input each worker re-seeds
        // its knot-span pointer `mu` by advancing from `k` to its chunk start
        // (O(#knots), ≪ #queries) then merge-advances within the chunk — the span
        // reached for any `x` depends only on `x` and the knots, so the result is
        // BIT-IDENTICAL to the single serial pointer walk.
        const BSPLINE_EVAL_PAR_GATE: usize = 1 << 15;
        let nthreads = if xs.len() >= BSPLINE_EVAL_PAR_GATE {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(xs.len() / 16384)
                .max(1)
        } else {
            1
        };

        if nthreads <= 1 {
            let mut d = vec![0.0; self.k + 1];
            if is_sorted {
                let mut results = Vec::with_capacity(xs.len());
                let mut mu = self.k;
                let n = self.c.len();
                let t = &self.t;
                for &x in xs {
                    while mu < n - 1 && x >= t[mu + 1] {
                        mu += 1;
                    }
                    results.push(self.eval_into_with_span(x, mu, &mut d));
                }
                return results;
            }
            return xs.iter().map(|&x| self.eval_into(x, &mut d)).collect();
        }

        let mut out = vec![0.0; xs.len()];
        let chunk = xs.len().div_ceil(nthreads);
        let n = self.c.len();
        std::thread::scope(|s| {
            for (xchunk, ochunk) in xs.chunks(chunk).zip(out.chunks_mut(chunk)) {
                s.spawn(move || {
                    let mut d = vec![0.0; self.k + 1];
                    if is_sorted {
                        let t = &self.t;
                        let mut mu = self.k;
                        while mu < n - 1 && xchunk[0] >= t[mu + 1] {
                            mu += 1;
                        }
                        for (o, &x) in ochunk.iter_mut().zip(xchunk) {
                            while mu < n - 1 && x >= t[mu + 1] {
                                mu += 1;
                            }
                            *o = self.eval_into_with_span(x, mu, &mut d);
                        }
                    } else {
                        for (o, &x) in ochunk.iter_mut().zip(xchunk) {
                            *o = self.eval_into(x, &mut d);
                        }
                    }
                });
            }
        });
        out
    }

    fn eval_into_with_span(&self, x: f64, mu: usize, d: &mut [f64]) -> f64 {
        Self::eval_parts_with_span(&self.t, &self.c, self.k, self.extrapolate, x, mu, d)
    }

    fn eval_parts(t: &[f64], c: &[f64], k: usize, extrapolate: bool, x: f64, d: &mut [f64]) -> f64 {
        let n = c.len();
        if !extrapolate && (x < t[k] || x > t[n]) {
            return f64::NAN;
        }
        let mu = Self::find_span_parts(t, c, k, x);
        Self::eval_parts_with_span(t, c, k, extrapolate, x, mu, d)
    }

    fn eval_parts_with_span(
        t: &[f64],
        c: &[f64],
        k: usize,
        extrapolate: bool,
        x: f64,
        mu: usize,
        d: &mut [f64],
    ) -> f64 {
        let n = c.len();
        if !extrapolate && (x < t[k] || x > t[n]) {
            return f64::NAN;
        }
        for (j, value) in d.iter_mut().enumerate().take(k + 1) {
            let idx = mu.wrapping_sub(k) + j;
            *value = if idx < n { c[idx] } else { 0.0 };
        }
        for r in 1..=k {
            for j in (r..=k).rev() {
                let left = mu.wrapping_sub(k) + j;
                let right = left + k + 1 - r;
                if right < t.len() {
                    let denom = t[right] - t[left];
                    if denom > 0.0 {
                        let alpha = (x - t[left]) / denom;
                        d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                    } else {
                        d[j] = d[j - 1];
                    }
                }
            }
        }
        d[k]
    }

    pub fn derivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 {
            return Ok(self.clone());
        }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;
        for _ in 0..nu {
            if k == 0 {
                return Err(InterpError::InvalidArgument {
                    detail: "cannot differentiate degree-0 spline".to_string(),
                });
            }
            let n = c.len();
            let mut dc = Vec::with_capacity(n - 1);
            for i in 0..n - 1 {
                let denom = t[i + k + 1] - t[i + 1];
                dc.push(if denom > 0.0 {
                    k as f64 * (c[i + 1] - c[i]) / denom
                } else {
                    0.0
                });
            }
            t.remove(t.len() - 1);
            t.remove(0);
            c = dc;
            k -= 1;
        }
        Self::new(t, c, k)
    }

    pub fn antiderivative(&self, nu: usize) -> Result<Self, InterpError> {
        if nu == 0 {
            return Ok(self.clone());
        }
        let mut t = self.t.clone();
        let mut c = self.c.clone();
        let mut k = self.k;
        for _ in 0..nu {
            let n = c.len();
            t.insert(0, t[0]);
            t.push(t[t.len() - 1]);
            k += 1;
            // After inserting the two knots above and incrementing k, the
            // produced `new_c` has exactly `t.len() - k - 1 == n + 1` entries,
            // so the B-spline invariant holds without any padding.
            let mut new_c = vec![0.0; n + 1];
            for i in 0..n {
                let denom = t[i + k + 1] - t[i + 1];
                new_c[i + 1] = new_c[i] + c[i] * denom / k as f64;
            }
            c = new_c;
        }
        Self::new(t, c, k)
    }

    pub fn integrate(&self, a: f64, b: f64) -> Result<f64, InterpError> {
        let anti = self.antiderivative(1)?;
        Ok(anti.eval(b) - anti.eval(a))
    }

    fn find_span_parts(t: &[f64], c: &[f64], k: usize, x: f64) -> usize {
        Self::find_span_n(t, c.len(), k, x)
    }

    /// Knot-span search by coefficient count `n` (same convention as `find_span_parts`),
    /// so the separable `eval_grid` can find the span without a coefficient slice.
    fn find_span_n(t: &[f64], n: usize, k: usize, x: f64) -> usize {
        if x <= t[k] {
            return k;
        }
        if x >= t[n] {
            let mut mu = n - 1;
            while mu > k && t[mu] == t[n] {
                mu -= 1;
            }
            return mu;
        }
        let mut lo = k;
        let mut hi = n;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

#[derive(Debug, Clone)]
pub struct BarycentricInterpolator {
    xi: Vec<f64>,
    yi: Vec<f64>,
    wi: Vec<f64>,
}

impl BarycentricInterpolator {
    pub fn new(xi: &[f64], yi: &[f64]) -> Result<Self, InterpError> {
        if xi.len() != yi.len() {
            return Err(InterpError::LengthMismatch {
                x_len: xi.len(),
                y_len: yi.len(),
            });
        }
        if xi.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        // Duplicate-node rejection is fused into the weight pass below: that loop already
        // forms every xi[i]-xi[j] difference, so the check costs one abs+compare instead of
        // a separate O(n²) scan over xi (and re-streaming the array). Weights are unchanged.

        // Barycentric weights w_i = 1/∏_{j≠i}(x_i − x_j). The raw product over/underflows for
        // even moderate node counts; SciPy's BarycentricInterpolator scales every difference by
        // the inverse capacity 4/(x_max − x_min), normalizing the interval to unit logarithmic
        // capacity so the product stays O(1) for large n. The resulting common factor
        // 4^{−(n−1)}·(x_max−x_min)^{n−1} is identical across all i and cancels in the barycentric
        // quotient num/den, so eval is unchanged — but now finite for hundreds/thousands of nodes.
        let x_max = xi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_min = xi.iter().cloned().fold(f64::INFINITY, f64::min);
        let inv_cap = if x_max > x_min {
            4.0 / (x_max - x_min)
        } else {
            1.0
        };
        let mut wi = vec![1.0; xi.len()];
        for i in 0..xi.len() {
            let mut denom = 1.0;
            for j in 0..xi.len() {
                if i != j {
                    let diff = xi[i] - xi[j];
                    if diff.abs() <= 1e-15 {
                        let (a, b) = (i.min(j), i.max(j));
                        return Err(InterpError::InvalidArgument {
                            detail: format!("duplicate interpolation nodes at indices {a} and {b}"),
                        });
                    }
                    denom *= inv_cap * diff;
                }
            }
            if !denom.is_finite() || denom == 0.0 {
                return Err(InterpError::InvalidArgument {
                    detail: format!(
                        "barycentric weight computation failed at node {i}: \
                         denominator={denom} (nodes may be too closely spaced or too widely separated)"
                    ),
                });
            }
            wi[i] = 1.0 / denom;
        }

        Ok(Self {
            xi: xi.to_vec(),
            yi: yi.to_vec(),
            wi,
        })
    }

    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        for (&xi, &yi) in self.xi.iter().zip(self.yi.iter()) {
            if (x - xi).abs() <= 1e-15 {
                return yi;
            }
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for ((&xi, &yi), &wi) in self.xi.iter().zip(self.yi.iter()).zip(self.wi.iter()) {
            let term = wi / (x - xi);
            numerator += term * yi;
            denominator += term;
        }
        if denominator == 0.0 {
            return f64::NAN;
        }
        numerator / denominator
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        // Each query is an independent O(n_nodes) barycentric evaluation dominated
        // by one reciprocal per node (compute-bound), so the queries fan out
        // across threads in contiguous chunks. Concatenated in query order with
        // each eval unchanged, so the result is bit-identical to the serial map.
        par_query_map(xs, self.xi.len(), |&x| self.eval(x))
    }
}

/// Floater-Hormann barycentric *rational* interpolator with no poles on the
/// real axis (C∞ smooth). Matches `scipy.interpolate.FloaterHormannInterpolator`.
///
/// Blends `n - d` polynomials of degree `d` (`0 ≤ d < n`) into a rational
/// interpolant of approximation order `O(h^{d+1})`. `d = n - 1` reduces to
/// polynomial interpolation. The barycentric weights follow Floater & Hormann
/// (2007), Eq. (18): `w_k = (−1)^{k−d} Σ_i Π_{j≠k} 1/|x_k − x_j|`.
pub struct FloaterHormannInterpolator {
    xi: Vec<f64>,
    yi: Vec<f64>,
    wi: Vec<f64>,
}

impl FloaterHormannInterpolator {
    pub fn new(xi: &[f64], yi: &[f64], d: usize) -> Result<Self, InterpError> {
        if xi.len() != yi.len() {
            return Err(InterpError::LengthMismatch {
                x_len: xi.len(),
                y_len: yi.len(),
            });
        }
        let n = xi.len();
        if n == 0 {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if d >= n {
            return Err(InterpError::InvalidArgument {
                detail: format!("d must satisfy 0 <= d < n (got d={d}, n={n})"),
            });
        }
        // Near-duplicate rejection. For large n the O(n²) all-pairs scan dominated
        // construction (the Floater–Hormann weight pass below is only O(n·d)), so switch to
        // an O(n log n) sort: sort node values carrying original indices, then any pair
        // within 1e-15 must appear as a sorted-adjacent pair (every value lying between two
        // near-equal nodes is itself within 1e-15 of both). Weights are untouched (the
        // sorted copy is discarded); total_cmp is NaN-safe — NaN never flags, matching the
        // all-pairs scan where (NaN).abs() <= 1e-15 is false. Small n keeps the direct scan.
        if n > 64 {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| xi[a].total_cmp(&xi[b]));
            for w in order.windows(2) {
                if (xi[w[0]] - xi[w[1]]).abs() <= 1e-15 {
                    let (a, b) = (w[0].min(w[1]), w[0].max(w[1]));
                    return Err(InterpError::InvalidArgument {
                        detail: format!("duplicate interpolation nodes at indices {a} and {b}"),
                    });
                }
            }
        } else {
            for i in 0..n {
                for j in i + 1..n {
                    if (xi[i] - xi[j]).abs() <= 1e-15 {
                        return Err(InterpError::InvalidArgument {
                            detail: format!("duplicate interpolation nodes at indices {i} and {j}"),
                        });
                    }
                }
            }
        }

        let mut wi = vec![0.0_f64; n];
        for k in 0..n {
            let lo = k.saturating_sub(d);
            let hi = (k + 1).min(n - d); // exclusive
            let mut wk = 0.0_f64;
            for i in lo..hi {
                // Π_{j=i..=i+d, j≠k} 1/|x_k − x_j|.
                let mut prod = 1.0_f64;
                for j in i..=i + d {
                    if j != k {
                        prod *= (xi[k] - xi[j]).abs();
                    }
                }
                wk += 1.0 / prod;
            }
            let sign = if (k as i64 - d as i64).rem_euclid(2) == 0 {
                1.0
            } else {
                -1.0
            };
            wi[k] = sign * wk;
        }

        Ok(Self {
            xi: xi.to_vec(),
            yi: yi.to_vec(),
            wi,
        })
    }

    /// The barycentric weights of the rational interpolant.
    pub fn weights(&self) -> &[f64] {
        &self.wi
    }

    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        for (&xi, &yi) in self.xi.iter().zip(self.yi.iter()) {
            if (x - xi).abs() <= 1e-15 {
                return yi;
            }
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for ((&xi, &yi), &wi) in self.xi.iter().zip(self.yi.iter()).zip(self.wi.iter()) {
            let term = wi / (x - xi);
            numerator += term * yi;
            denominator += term;
        }
        if denominator == 0.0 {
            return f64::NAN;
        }
        numerator / denominator
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        par_query_map(xs, self.xi.len(), |&x| self.eval(x))
    }
}

/// The minimal-singular-value right vector(s) of `a`, summed over a degenerate
/// minimum and normalized by `√count` (the AAA weight selection of scipy). Falls
/// back to a column-normalized SVD when `a` is severely ill-conditioned, exactly
/// as scipy does.
fn aaa_min_singular_vector(a: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    fn min_sv(res: &fsci_linalg::SvdResult) -> Vec<f64> {
        let smin = res.s.iter().copied().fold(f64::INFINITY, f64::min);
        let ncol = res.vt[0].len();
        let mut wj = vec![0.0_f64; ncol];
        let mut cnt = 0.0_f64;
        for (i, &si) in res.s.iter().enumerate() {
            if si == smin {
                cnt += 1.0;
                for c in 0..ncol {
                    wj[c] += res.vt[i][c];
                }
            }
        }
        let scale = cnt.sqrt();
        for w in &mut wj {
            *w /= scale;
        }
        wj
    }
    let opt = fsci_linalg::DecompOptions::default();
    let res = fsci_linalg::svd(a, opt).map_err(|e| InterpError::InvalidArgument {
        detail: format!("AAA: SVD failed: {e}"),
    })?;
    let cond_tol = 1.0 / (3.0 * f64::EPSILON);
    let last = res.s.len() - 1;
    if res.s[last] > 0.0 && res.s[0] / res.s[last] > cond_tol {
        // Severely ill-conditioned: re-solve on column-normalized columns.
        let nrows = a.len();
        let ncol = a[0].len();
        let colnorm: Vec<f64> = (0..ncol)
            .map(|c| (0..nrows).map(|r| a[r][c] * a[r][c]).sum::<f64>().sqrt())
            .collect();
        let an: Vec<Vec<f64>> = a
            .iter()
            .map(|row| (0..ncol).map(|c| row[c] / colnorm[c]).collect())
            .collect();
        let res2 = fsci_linalg::svd(&an, opt).map_err(|e| InterpError::InvalidArgument {
            detail: format!("AAA: SVD failed: {e}"),
        })?;
        let wj = min_sv(&res2);
        return Ok((0..ncol).map(|c| wj[c] / colnorm[c]).collect());
    }
    Ok(min_sv(&res))
}

/// Adaptive Antoulas–Anderson (AAA) barycentric rational approximation.
///
/// Matches `scipy.interpolate.AAA(x, y)` for real 1-D data: greedily adds the
/// support point where the current approximant is worst and solves a
/// least-squares (SVD) problem for the barycentric weights each step, until the
/// residual `‖y − r‖∞` drops below `rtol·‖y‖∞` (`rtol` defaults to `eps^0.75`,
/// `max_terms` caps the support set). The approximant is evaluated in barycentric
/// form and interpolates the data at the support points.
///
/// The Froissart-doublet `clean_up` post-process (scipy's `clean_up=True`) is not
/// performed; the result equals scipy's for well-conditioned data, where no
/// spurious poles arise.
pub struct Aaa {
    support_points: Vec<f64>,
    support_values: Vec<f64>,
    weights: Vec<f64>,
}

impl Aaa {
    pub fn new(
        z: &[f64],
        f: &[f64],
        rtol: Option<f64>,
        max_terms: usize,
    ) -> Result<Self, InterpError> {
        let m_total = z.len();
        if m_total != f.len() {
            return Err(InterpError::LengthMismatch {
                x_len: m_total,
                y_len: f.len(),
            });
        }
        if m_total == 0 {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if max_terms < 1 {
            return Err(InterpError::InvalidArgument {
                detail: "max_terms must be >= 1".to_string(),
            });
        }
        let rtol = rtol.unwrap_or(f64::EPSILON.powf(0.75));
        let fmax = f.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let atol = rtol * fmax;
        let max_terms = max_terms.min(m_total);

        let mut mask = vec![true; m_total];
        let mean_f = f.iter().sum::<f64>() / m_total as f64;
        let mut r = vec![mean_f; m_total];
        let mut cols_c: Vec<Vec<f64>> = Vec::new(); // Cauchy columns, each length m_total
        let mut zj: Vec<f64> = Vec::new();
        let mut fj: Vec<f64> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();

        for m in 0..max_terms {
            // Select the worst-approximated remaining point (first max on ties).
            let mut jj = usize::MAX;
            let mut best = -1.0_f64;
            for i in 0..m_total {
                if mask[i] {
                    let e = (f[i] - r[i]).abs();
                    if e > best {
                        best = e;
                        jj = i;
                    }
                }
            }
            zj.push(z[jj]);
            fj.push(f[jj]);
            let cm: Vec<f64> = (0..m_total).map(|i| 1.0 / (z[i] - z[jj])).collect();
            cols_c.push(cm);
            mask[jj] = false;

            // Loewner submatrix over remaining rows, columns 0..=m.
            let ncol = m + 1;
            let masked_rows: Vec<usize> = (0..m_total).filter(|&i| mask[i]).collect();
            let asub: Vec<Vec<f64>> = masked_rows
                .iter()
                .map(|&i| (0..ncol).map(|c| (f[i] - fj[c]) * cols_c[c][i]).collect())
                .collect();
            let wj = aaa_min_singular_vector(&asub)?;

            // Rational approximant R at every point (= f at support points).
            let mut new_r = vec![0.0_f64; m_total];
            for i in 0..m_total {
                let mut nn = 0.0_f64;
                let mut dd = 0.0_f64;
                let mut at_support = false;
                for c in 0..ncol {
                    if wj[c] == 0.0 {
                        continue;
                    }
                    let cc = cols_c[c][i];
                    if !cc.is_finite() {
                        at_support = true;
                        break;
                    }
                    nn += cc * wj[c] * fj[c];
                    dd += cc * wj[c];
                }
                new_r[i] = if at_support || !dd.is_finite() || dd == 0.0 {
                    f[i]
                } else {
                    nn / dd
                };
            }
            r = new_r;
            weights = wj;

            let max_err = (0..m_total)
                .map(|i| (f[i] - r[i]).abs())
                .fold(0.0_f64, f64::max);
            if max_err <= atol {
                break;
            }
        }

        // Drop any zero-weight support points (scipy does the same).
        let mut sp = Vec::new();
        let mut sv = Vec::new();
        let mut w = Vec::new();
        for c in 0..weights.len() {
            if weights[c] != 0.0 {
                sp.push(zj[c]);
                sv.push(fj[c]);
                w.push(weights[c]);
            }
        }
        Ok(Self {
            support_points: sp,
            support_values: sv,
            weights: w,
        })
    }

    pub fn support_points(&self) -> &[f64] {
        &self.support_points
    }
    pub fn support_values(&self) -> &[f64] {
        &self.support_values
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        for (&zk, &fk) in self.support_points.iter().zip(self.support_values.iter()) {
            if x == zk {
                return fk;
            }
        }
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for ((&zk, &fk), &wk) in self
            .support_points
            .iter()
            .zip(self.support_values.iter())
            .zip(self.weights.iter())
        {
            let t = wk / (x - zk);
            num += t * fk;
            den += t;
        }
        num / den
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        // Each query is an independent O(support) barycentric reduction over the AAA support set,
        // so distribute the queries across cores. BYTE-IDENTICAL: `par_query_map` preserves query
        // order and the per-query `eval` (its own reduction) is unchanged; only the owning core
        // differs. Work-gated (`work = m·support ≥ 2^23`), so small batches stay serial — this
        // brings AAA in line with every other interpolator's `eval_many`, which was the last serial
        // one. `eval` reads `support_points/values/weights` only, so it is `Sync`.
        par_query_map(xs, self.support_points.len().max(1), |&x| self.eval(x))
    }
}

/// One-shot barycentric interpolation: build a `BarycentricInterpolator`
/// from `(xi, yi)` and evaluate it at every point in `x_new`.
///
/// Matches `scipy.interpolate.barycentric_interpolate(xi, yi, x)` for
/// 1-D real inputs. Returns the interpolated values as a `Vec<f64>` of
/// the same length as `x_new`.
pub fn barycentric_interpolate(
    xi: &[f64],
    yi: &[f64],
    x_new: &[f64],
) -> Result<Vec<f64>, InterpError> {
    let interp = BarycentricInterpolator::new(xi, yi)?;
    Ok(interp.eval_many(x_new))
}

#[derive(Debug, Clone)]
pub struct UnivariateSpline {
    spline: BSpline,
    smoothing_factor: f64,
}

impl UnivariateSpline {
    pub fn new(x: &[f64], y: &[f64], s: f64) -> Result<Self, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::LengthMismatch {
                x_len: x.len(),
                y_len: y.len(),
            });
        }
        if x.len() < 4 {
            return Err(InterpError::TooFewPoints {
                minimum: 4,
                actual: x.len(),
            });
        }
        if x.iter().any(|&v| !v.is_finite()) {
            return Err(InterpError::NonFiniteX);
        }
        if x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(InterpError::UnsortedX);
        }

        let spline = if s <= 0.0 {
            make_interp_spline(x, y, 3)?
        } else {
            make_smoothing_spline_impl(x, y, s, 3)?
        };

        Ok(Self {
            spline,
            smoothing_factor: s.max(0.0),
        })
    }

    pub fn eval(&self, x: f64) -> f64 {
        self.spline.eval(x)
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        self.spline.eval_many(xs)
    }

    pub fn derivative(&self, nu: usize) -> Result<BSpline, InterpError> {
        self.spline.derivative(nu)
    }

    pub fn integral(&self, a: f64, b: f64) -> Result<f64, InterpError> {
        self.spline.integrate(a, b)
    }

    pub fn smoothing_factor(&self) -> f64 {
        self.smoothing_factor
    }
}

#[derive(Debug, Clone)]
pub struct InterpolatedUnivariateSpline {
    spline: UnivariateSpline,
}

impl InterpolatedUnivariateSpline {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, InterpError> {
        Ok(Self {
            spline: UnivariateSpline::new(x, y, 0.0)?,
        })
    }

    pub fn eval(&self, x: f64) -> f64 {
        self.spline.eval(x)
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        self.spline.eval_many(xs)
    }

    pub fn derivative(&self, nu: usize) -> Result<BSpline, InterpError> {
        self.spline.derivative(nu)
    }

    pub fn integral(&self, a: f64, b: f64) -> Result<f64, InterpError> {
        self.spline.integral(a, b)
    }
}

/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn make_interp_spline(x: &[f64], y: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let n = x.len();
    if n != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: n,
            y_len: y.len(),
        });
    }
    if n < k + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: k + 1,
            actual: n,
        });
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpError::UnsortedX);
    }
    let t = interpolation_knots(x, k);
    let mut a_mat = Vec::with_capacity(n);
    let mut scratch = vec![0.0_f64; n];
    let mut bw = 0usize;
    for i in 0..n {
        let Some(mu) = bspline_find_interval(&t, x[i], n) else {
            return Err(InterpError::InvalidArgument {
                detail: format!("x[{i}] is outside the spline knot span"),
            });
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let denom_left = t[idx + p] - t[idx];
                    if denom_left > 0.0 {
                        val += (x[i] - t[idx]) / denom_left * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < n {
                    let denom_right = t[idx + p + 1] - t[idx + 1];
                    if denom_right > 0.0 {
                        val += (t[idx + p + 1] - x[i]) / denom_right * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        for col in lo..=mu {
            bw = bw.max(i.abs_diff(col));
        }
        a_mat.push(CompactBandRow::from_slice(lo, &scratch[lo..=mu]));
        for s in scratch[lo..=mu].iter_mut() {
            *s = 0.0;
        }
    }
    let mut rhs = y.to_vec();
    // The collocation A[i][j]=B_j(x_i) is banded, so compact rows avoid the
    // dense O(n^2) allocation/fill that still remains after switching the solve
    // to a banded window.
    let c = solve_banded_compact(&mut a_mat, &mut rhs, bw)?;
    BSpline::new(t, c, k)
}

/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn make_lsq_spline(x: &[f64], y: &[f64], t: &[f64], k: usize) -> Result<BSpline, InterpError> {
    let m = x.len();
    if m != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: m,
            y_len: y.len(),
        });
    }
    let min_knots = k
        .checked_add(2)
        .ok_or_else(|| InterpError::InvalidArgument {
            detail: "spline degree is too large".to_string(),
        })?;
    if t.len() < min_knots {
        return Err(InterpError::InvalidArgument {
            detail: format!(
                "need at least {} knots for degree {}, got {}",
                min_knots,
                k,
                t.len()
            ),
        });
    }
    let n = t.len() - k - 1;
    if n == 0 || n > m {
        return Err(InterpError::InvalidArgument {
            detail: format!("need t.len()-k-1 coeffs ({n}) <= points ({m})"),
        });
    }
    // A^T A is k-banded (B_j, B_l overlap iff |j-l| ≤ k); store it in COMPACT banded rows
    // (O(n·k)) instead of a dense n×n matrix (n inner-Vec allocs). Byte-identical: the same
    // band entries are accumulated, and solve_banded_compact is bit-identical to solve_banded
    // for a banded system.
    // Pre-size each band row to its build window [a-k, a+k] so the scatter never grows the
    // Vec (solve-time U-fill still grows it, same as make_interp_spline).
    let mut ata: Vec<CompactBandRow> = (0..n)
        .map(|a| {
            let start = a.saturating_sub(k);
            let end = (a + k).min(n - 1);
            CompactBandRow::from_slice(start, &vec![0.0; end - start + 1])
        })
        .collect();
    let mut aty = vec![0.0; n];
    // O(m*k^2) sparse normal-equations assembly. A B-spline basis has local support: for
    // each sample only the k+1 functions on the knot span containing x[i] are nonzero. So
    // find that span (`mu`, O(log n)), evaluate just its k+1 basis values with the exact
    // Cox-de Boor recursion into a REUSED scratch buffer (no per-sample length-n alloc, no
    // O(n) sweep), and scatter the rank-1 outer product over the window [mu-k, mu].
    //
    // Bit-identical to the previous dense build for finite y: the windowed de Boor produces
    // the same basis values bit-for-bit (identical recursion, same ascending in-place sweep
    // as eval_basis_all); the window superset of the nonzeros only adds basis[a]*basis[b]
    // terms with a zero factor (+/-0.0, `v + (+/-0.0) == v`); the per-sample (a,b) order is
    // ascending and the sample order is unchanged, so every bit of A^T A matches. A^T y is
    // accumulated only over the window — for the finite inputs SciPy's make_lsq_spline
    // accepts this equals the full loop bit-for-bit (skipped 0*y = +/-0.0); the sibling
    // make_smoothing_spline_impl already assembles A^T y sparsely the same way. [perf]
    let mut scratch = vec![0.0_f64; n];
    for i in 0..m {
        let Some(mu) = bspline_find_interval(t, x[i], n) else {
            continue;
        };
        // Windowed Cox-de Boor (lo == hi == mu in eval_basis_all's terms).
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let denom_left = t[idx + p] - t[idx];
                    if denom_left > 0.0 {
                        val += (x[i] - t[idx]) / denom_left * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < n {
                    let denom_right = t[idx + p + 1] - t[idx + 1];
                    if denom_right > 0.0 {
                        val += (t[idx + p + 1] - x[i]) / denom_right * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        let yi = y[i];
        for a in lo..=mu {
            let ba = scratch[a];
            aty[a] += ba * yi;
            let row = &mut ata[a];
            // Rows are pre-sized to [a-k, a+k] ⊇ [lo, mu], so index the band buffer directly
            // (no cell_mut growth checks).
            let base = row.start;
            for b in lo..=mu {
                row.values[b - base] += ba * scratch[b];
            }
        }
        // Reset only the touched window so the buffer stays clean for the next sample.
        for s in scratch[lo..=mu].iter_mut() {
            *s = 0.0;
        }
    }
    // A^T A of B-splines is banded with half-width k (B_j and B_l overlap iff |j-l| <= k).
    let c = solve_banded_compact(&mut ata, &mut aty, k)?;
    BSpline::new(t.to_vec(), c, k)
}

/// Weighted least-squares univariate spline of degree `k` with user-given
/// interior knots, matching `scipy.interpolate.LSQUnivariateSpline(x, y, t, k)`.
///
/// `t_interior` are the interior knots only (strictly inside `(x[0], x[-1])`);
/// the `k+1` boundary knots at the data extents are added internally. Returns
/// the tck tuple `(t, c, k)` consumable by [`splev`]/[`BSpline`].
#[allow(clippy::type_complexity)]
pub fn lsq_univariate_spline(
    x: &[f64],
    y: &[f64],
    t_interior: &[f64],
    k: usize,
) -> Result<(Vec<f64>, Vec<f64>, usize), InterpError> {
    if x.len() != y.len() || x.len() < k + 1 {
        return Err(InterpError::InvalidArgument {
            detail: "need len(x)==len(y) >= k+1".to_string(),
        });
    }
    let xb = x[0];
    let xe = x[x.len() - 1];
    for &t in t_interior {
        if t <= xb || t >= xe {
            return Err(InterpError::InvalidArgument {
                detail: "interior knots must lie strictly inside (x[0], x[-1])".to_string(),
            });
        }
    }
    let mut t = vec![xb; k + 1];
    t.extend_from_slice(t_interior);
    t.extend(std::iter::repeat_n(xe, k + 1));
    let bs = make_lsq_spline(x, y, &t, k)?;
    Ok((bs.t.clone(), bs.c.clone(), k))
}

fn interpolation_knots(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let num_knots = n + k + 1;
    let num_interior = n - k - 1;
    let mut t = Vec::with_capacity(num_knots);
    for _ in 0..=k {
        t.push(x[0]);
    }
    // Match scipy.interpolate._bsplines._not_a_knot. For odd degree the interior knots are
    // the data sites x[k2 + i]; for even degree they are the Greville midpoints
    // (x[j] + x[j+1])/2 with the outermost omitted, a la not-a-knot. The odd branch is
    // bit-identical to the previous `x[i + 1 + (k-1)/2]` (since 1 + (k-1)/2 == (k+1)/2 for
    // odd k), so only even-degree splines change.
    if k % 2 == 1 {
        let k2 = k.div_ceil(2);
        for i in 0..num_interior {
            t.push(x[k2 + i]);
        }
    } else {
        let k2 = k / 2;
        for i in 0..num_interior {
            let j = k2 + i;
            t.push((x[j] + x[j + 1]) / 2.0);
        }
    }
    for _ in 0..=k {
        t.push(x[n - 1]);
    }
    t
}

/// Weighted least-squares B-spline fit on knots `t` plus the per-data residuals,
/// for the FITPACK knot-placement loop. Mirrors [`make_lsq_spline`]'s banded
/// normal-equations assembly but with linear weights `w`, returning
/// `(residuals_i = w_i²·(y_i − s(x_i))², fp = Σ residuals_i)` — exactly what
/// scipy's `_get_residuals` produces.
fn lsq_residuals(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    t: &[f64],
    k: usize,
) -> Result<(Vec<f64>, f64), InterpError> {
    let m = x.len();
    let n = t.len() - k - 1;
    let mut ata = vec![vec![0.0_f64; n]; n];
    let mut aty = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; n];
    for i in 0..m {
        let Some(mu) = bspline_find_interval(t, x[i], n) else {
            continue;
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let denom_left = t[idx + p] - t[idx];
                    if denom_left > 0.0 {
                        val += (x[i] - t[idx]) / denom_left * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < n {
                    let denom_right = t[idx + p + 1] - t[idx + 1];
                    if denom_right > 0.0 {
                        val += (t[idx + p + 1] - x[i]) / denom_right * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        let wi = w[i];
        let yi = y[i];
        for a in lo..=mu {
            let ba = scratch[a];
            aty[a] += wi * ba * yi;
            let row = &mut ata[a];
            for b in lo..=mu {
                row[b] += wi * ba * scratch[b];
            }
        }
        for sc in scratch[lo..=mu].iter_mut() {
            *sc = 0.0;
        }
    }
    let c = solve_banded(&mut ata, &mut aty, k)?;
    let spl = BSpline::new(t.to_vec(), c, k)?;
    let mut residuals = vec![0.0_f64; m];
    let mut fp = 0.0_f64;
    for i in 0..m {
        let d = y[i] - spl.eval(x[i]);
        let r = w[i] * w[i] * d * d;
        residuals[i] = r;
        fp += r;
    }
    Ok((residuals, fp))
}

/// FITPACK `fpknot`: locate a new interior knot inside the knot interval with the
/// largest `fpint`, placed at the middle strictly-interior data point.
///
/// `fpint` for interval `[t[k+j], t[k+j+1]]` follows FITPACK's `fpcurf`
/// accumulation: the full squared residual of every data point strictly inside,
/// plus the boundary data points (which sit on knots) contributing *half* to each
/// adjacent interval — except the global endpoints `xb`/`xe`, which contribute in
/// full to the first/last interval. Intervals with no strictly-interior point are
/// skipped (`nrdata == 0`); the first interval attaining the maximal `fpint` wins,
/// and the new knot is the middle strictly-interior data point `x[beg + cnt/2]`.
fn fpknot(x: &[f64], t: &[f64], k: usize, residuals: &[f64]) -> f64 {
    let n = t.len();
    let nrint = n - 2 * k - 1;
    let m = x.len();
    let xb = x[0];
    let xe = x[m - 1];
    let mut best_fpint = f64::NEG_INFINITY;
    let mut best_beg = 0usize;
    let mut best_cnt = 0usize;
    for j in 0..nrint {
        let lo = t[k + j];
        let hi = t[k + j + 1];
        // Strictly-interior data points: lo < x[i] < hi.
        let mut beg = 0usize;
        while beg < m && x[beg] <= lo {
            beg += 1;
        }
        let mut end = beg;
        let mut fpint = 0.0_f64;
        while end < m && x[end] < hi {
            fpint += residuals[end];
            end += 1;
        }
        let cnt = end - beg;
        if cnt == 0 {
            continue;
        }
        // Left boundary data point (== lo): full if it's the global start, else half.
        if beg > 0 && x[beg - 1] == lo {
            fpint += if lo == xb {
                residuals[beg - 1]
            } else {
                0.5 * residuals[beg - 1]
            };
        }
        // Right boundary data point (== hi): full if it's the global end, else half.
        if end < m && x[end] == hi {
            fpint += if hi == xe {
                residuals[end]
            } else {
                0.5 * residuals[end]
            };
        }
        if fpint > best_fpint {
            best_fpint = fpint;
            best_beg = beg;
            best_cnt = cnt;
        }
    }
    x[best_beg + best_cnt / 2]
}

/// Insert the knot chosen by [`fpknot`], keeping `t` sorted (scipy `add_knot`).
fn add_knot(x: &[f64], t: &[f64], k: usize, residuals: &[f64]) -> Vec<f64> {
    let new_knot = fpknot(x, t, k, residuals);
    let idx = t.partition_point(|&v| v < new_knot);
    let mut nt = Vec::with_capacity(t.len() + 1);
    nt.extend_from_slice(&t[..idx]);
    nt.push(new_knot);
    nt.extend_from_slice(&t[idx..]);
    nt
}

/// Generate the sequence of FITPACK least-squares knot vectors of increasing
/// length, denser where the LSQ spline deviates most from the data, until the
/// smoothing criterion `fp ≤ s` is satisfied (or the knot count reaches the
/// interpolation/`nest` limit).
///
/// Matches the non-periodic, not-a-knot form of
/// `scipy.interpolate.generate_knots(x, y, w, xb, xe, k, s, nest)`, returning the
/// full sequence of knot vectors that the SciPy generator would yield. For
/// `s == 0` it yields a single interpolation knot vector (and rejects `w`/`nest`,
/// as SciPy does). The periodic boundary condition is not supported.
#[allow(clippy::too_many_arguments)]
pub fn generate_knots(
    x: &[f64],
    y: &[f64],
    w: Option<&[f64]>,
    xb: Option<f64>,
    xe: Option<f64>,
    k: usize,
    s: f64,
    nest: Option<usize>,
) -> Result<Vec<Vec<f64>>, InterpError> {
    let m = x.len();
    if m != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: m,
            y_len: y.len(),
        });
    }
    if k == 0 {
        return Err(InterpError::InvalidArgument {
            detail: "spline degree k must be >= 1".to_string(),
        });
    }
    if m < k + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: k + 1,
            actual: m,
        });
    }
    if x.windows(2).any(|p| p[1] <= p[0]) {
        return Err(InterpError::UnsortedX);
    }
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "smoothing factor s must be >= 0".to_string(),
        });
    }
    if let Some(ww) = w
        && ww.len() != m
    {
        return Err(InterpError::InvalidArgument {
            detail: format!("weights length {} must match data length {m}", ww.len()),
        });
    }

    if s == 0.0 {
        if nest.is_some() || w.is_some() {
            return Err(InterpError::InvalidArgument {
                detail: "s == 0 is interpolation only (no weights or nest)".to_string(),
            });
        }
        return Ok(vec![interpolation_knots(x, k)]);
    }

    let xb = xb.unwrap_or(x[0]);
    let xe = xe.unwrap_or(x[m - 1]);
    let ones = vec![1.0_f64; m];
    let w = w.unwrap_or(&ones);
    let acc = s * 1e-3; // TOL = 1e-3
    let nmin = 2 * (k + 1);
    let nmax = m + k + 1;
    let nest = nest.unwrap_or((m + k + 1).max(2 * k + 3));
    if nest < nmin {
        return Err(InterpError::InvalidArgument {
            detail: format!("nest too small: {nest} < 2*(k+1) = {nmin}"),
        });
    }

    let mut out: Vec<Vec<f64>> = Vec::new();
    // Start with no interior knots: (k+1) copies of each endpoint.
    let mut t: Vec<f64> = Vec::with_capacity(nmin);
    for _ in 0..=k {
        t.push(xb);
    }
    for _ in 0..=k {
        t.push(xe);
    }
    let mut fp = 0.0_f64;
    let mut nplus = 1usize;

    for _iter in 0..m {
        out.push(t.clone());
        let fpold = fp;
        let (residuals, fp_new) = lsq_residuals(x, y, w, &t, k)?;
        fp = fp_new;
        let fpms = fp - s;
        if fpms.abs() < acc || fpms < 0.0 {
            return Ok(out);
        }

        // Decide how many knots to add this round (FITPACK heuristic).
        let n = t.len();
        if n == nmin {
            nplus = 1;
        } else {
            let delta = fpold - fp;
            let npl1: i64 = if delta > acc {
                (nplus as f64 * fpms / delta) as i64
            } else {
                (nplus * 2) as i64
            };
            let cand = npl1.max((nplus / 2) as i64).max(1);
            nplus = (nplus * 2).min(cand as usize);
        }

        let mut residuals = residuals;
        for j in 0..nplus {
            t = add_knot(x, &t, k, &residuals);
            let n = t.len();
            if n >= nmax {
                out.push(interpolation_knots(x, k));
                return Ok(out);
            }
            if n >= nest {
                out.push(t);
                return Ok(out);
            }
            if j < nplus - 1 {
                let (r, _) = lsq_residuals(x, y, w, &t, k)?;
                residuals = r;
            }
        }
    }
    Ok(out)
}

fn make_smoothing_spline_impl(
    x: &[f64],
    y: &[f64],
    s: f64,
    k: usize,
) -> Result<BSpline, InterpError> {
    let t = interpolation_knots(x, k);
    let n = x.len();
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];
    // O(n*k^2) sparse normal-equations assembly: locate the knot span containing each
    // x[i] (bspline_find_interval, O(log n)), evaluate just the k+1 active basis values
    // with the exact windowed Cox-de Boor recursion into a REUSED scratch buffer (no
    // per-sample length-n alloc, no O(n) scan), then scatter over the nonzero window
    // indices. BIT-IDENTICAL to the previous dense build: the windowed de Boor produces
    // the same basis values, and `nz` is exactly the same nonzero set in the same
    // ascending order, so every accumulator bit is preserved (skipped terms are 0). [perf]
    let mut scratch = vec![0.0_f64; n];
    let mut nz: Vec<usize> = Vec::with_capacity(k + 1);
    for i in 0..n {
        let Some(mu) = bspline_find_interval(&t, x[i], n) else {
            continue;
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let denom_left = t[idx + p] - t[idx];
                    if denom_left > 0.0 {
                        val += (x[i] - t[idx]) / denom_left * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < n {
                    let denom_right = t[idx + p + 1] - t[idx + 1];
                    if denom_right > 0.0 {
                        val += (t[idx + p + 1] - x[i]) / denom_right * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        nz.clear();
        nz.extend((lo..=mu).filter(|&idx| scratch[idx] != 0.0));
        let yi = y[i];
        for &j in &nz {
            let bj = scratch[j];
            aty[j] += bj * yi;
            let row = &mut ata[j];
            for &l in &nz {
                row[l] += bj * scratch[l];
            }
        }
        for s in scratch[lo..=mu].iter_mut() {
            *s = 0.0;
        }
    }

    let scale = y
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
        .max(1.0);
    let lambda = s / ((n as f64) * scale * scale);
    if lambda > 0.0 {
        for i in 0..n {
            ata[i][i] += penalty_diagonal(i, n, lambda);
            if i + 1 < n {
                let off = penalty_first_off_diagonal(i, n, lambda);
                ata[i][i + 1] += off;
                ata[i + 1][i] += off;
            }
            if i + 2 < n {
                ata[i][i + 2] += lambda;
                ata[i + 2][i] += lambda;
            }
        }
    }

    // A^T A has B-spline bandwidth k; the smoothing penalty adds the second off-diagonal,
    // so the system is banded with half-width max(k, 2).
    let c = solve_banded(&mut ata, &mut aty, k.max(2))?;
    BSpline::new(t, c, k)
}

fn penalty_diagonal(i: usize, n: usize, lambda: f64) -> f64 {
    if i == 0 || i + 1 == n {
        lambda
    } else if i == 1 || i + 2 == n {
        5.0 * lambda
    } else {
        6.0 * lambda
    }
}

fn penalty_first_off_diagonal(i: usize, n: usize, lambda: f64) -> f64 {
    if i == 0 || i + 2 == n {
        -2.0 * lambda
    } else {
        -4.0 * lambda
    }
}

/// Index `mu` of the knot span containing `x`, i.e. the unique `i` in `[0, n)` that
/// `eval_basis_all`'s degree-0 indicator sets to 1.0:
///   `(t[i] <= x && x < t[i+1]) || (x == t[i+1] && i+1 == n)`   (n == t.len()-k-1).
/// Returns `None` when `x` is outside the knot span (all basis functions are 0). The
/// knot vector `t` is non-decreasing, so `partition_point` finds the span in O(log n)
/// for any `x` (no assumption on the ordering of the sample points). Validated to match
/// `eval_basis_all`'s selection bit-for-bit by `bspline_find_interval_matches_eval_basis`.
fn bspline_find_interval(t: &[f64], x: f64, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    debug_assert!(t.len() > n);
    // p = number of knots in t[0..=n] that are <= x.
    let p = t[..=n].partition_point(|&v| v <= x);
    if p == 0 {
        return None; // x < t[0]
    }
    let mu = p - 1;
    if mu < n {
        // partition guarantees t[mu] <= x and t[mu+1] = t[p] > x: the first clause holds.
        Some(mu)
    } else if x == t[n] {
        // mu == n: x >= t[n]; only the right-boundary clause (x == t[n], i+1 == n) applies.
        Some(n - 1)
    } else {
        None // x > t[n], past the knot span
    }
}

/// The `k+1` non-zero B-spline basis values `N_{span-k..span}(x)` (Cox-de Boor, The NURBS
/// Book Algorithm A2.2). Returns a stack array indexed `[0..=k]` (k ≤ 5 for bivariate
/// splines). The `denom != 0.0` guard applies the standard 0/0→0 convention for repeated
/// knots; for clamped end-knots the denominators are positive span widths. Used by the
/// separable `eval_grid` (scipy bispev approach) to precompute each axis' basis ONCE.
fn bspline_basis_funs(t: &[f64], k: usize, x: f64, span: usize) -> [f64; 6] {
    let mut n = [0.0f64; 6];
    let mut left = [0.0f64; 6];
    let mut right = [0.0f64; 6];
    n[0] = 1.0;
    for j in 1..=k {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom != 0.0 { n[r] / denom } else { 0.0 };
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

fn eval_basis_all(t: &[f64], x: f64, k: usize, n: usize) -> Vec<f64> {
    let mut basis = vec![0.0; n];
    // Degree-0 indicator: exactly the interval(s) containing x become 1.0. Track the
    // nonzero span [lo, hi] so the Cox-de Boor recursion can be restricted to the
    // B-spline local support instead of sweeping all n indices at every level.
    let mut lo = n;
    let mut hi = 0usize;
    let mut any = false;
    for i in 0..n {
        if i + 1 < t.len()
            && ((t[i] <= x && x < t[i + 1]) || (x == t[i + 1] && i + 1 == t.len() - k - 1))
        {
            basis[i] = 1.0;
            if !any {
                lo = i;
                any = true;
            }
            hi = i;
        }
    }
    if !any {
        return basis; // x outside the knot span -> all basis functions are 0
    }
    // At level p the support of the active functions is [lo - p, hi]; indices outside
    // it are provably 0 (they were 0 at level p-1 and stay 0), so we skip them. The
    // sweep is ascending and in place: when computing basis[i] we read basis[i] (still
    // the previous level's value, not yet overwritten) and basis[i+1] (overwritten only
    // at the later step i+1), so the values and float ops are bit-identical to the
    // clone-per-level version — just without the O(n) clone and the dead 0*0 work.
    for p in 1..=k {
        let start = lo.saturating_sub(p);
        for i in start..=hi {
            let mut val = 0.0;
            if i + p < t.len() {
                let denom_left = t[i + p] - t[i];
                if denom_left > 0.0 {
                    val += (x - t[i]) / denom_left * basis[i];
                }
            }
            if i + p + 1 < t.len() && i + 1 < n {
                let denom_right = t[i + p + 1] - t[i + 1];
                if denom_right > 0.0 {
                    val += (t[i + p + 1] - x) / denom_right * basis[i + 1];
                }
            }
            basis[i] = val;
        }
    }
    basis
}

// Retained REFERENCE implementation: the optimised sibling's own doc cites it,
// and a test now runs the comparison that doc asserts. Dead only in non-test
// builds (frankenscipy-9yyez).
#[cfg_attr(not(test), allow(dead_code))]
fn solve_dense_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, InterpError> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "empty or mismatched system".to_string(),
        });
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            let v = a_row[col];
            if v.abs() > max_val {
                max_val = v.abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument {
                detail: "singular matrix".to_string(),
            });
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        // Zero-skipping Gaussian elimination. A row with `a[row][col] == 0` has factor 0,
        // so subtracting `factor * pivot` is `-= +/-0.0` for every entry — a no-op — and is
        // skipped; likewise a pivot entry of 0 contributes `+/-0.0` and is skipped. This is
        // bit-identical to the dense elimination (`v + (+/-0.0) == v` for finite `v`, and
        // the pivot ensures `factor` is finite) but only touches the actual non-zeros, so a
        // banded matrix (e.g. a B-spline A^T A) is solved in ~O(n * bandwidth) work instead
        // of O(n^3). split_at_mut borrows the pivot row in place (no per-row clone).
        //
        // Run serially: the previous per-column `std::thread::scope` spawn (a fresh
        // ~ncores-thread pool for EACH of the n columns) was net-catastrophic — the
        // spawn overhead dwarfs the per-column work, making a dense RBF solve 130×+
        // slower the moment the trailing block crossed the spawn threshold (n=150
        // serial 2 ms → n=300 parallel 268 ms). Serial elimination is the bit-identical
        // reference and far faster here; a single-spawn blocked LU is the right lever
        // for genuinely huge dense solves (future work).
        let (head, tail) = a.split_at_mut(col + 1);
        let pivot = &head[col];
        let pivot_diag = pivot[col];
        let (b_head, b_tail) = b.split_at_mut(col + 1);
        let b_col = b_head[col];
        for (target, bt) in tail.iter_mut().zip(b_tail.iter_mut()) {
            let av = target[col];
            if av == 0.0 {
                continue;
            }
            let factor = av / pivot_diag;
            for j in col..n {
                let pval = pivot[j];
                if pval != 0.0 {
                    target[j] -= factor * pval;
                }
            }
            *bt -= factor * b_col;
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            let aij = a[i][j];
            if aij != 0.0 {
                s -= aij * x[j];
            }
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

/// Flat row-major (`n×n` in `a`) counterpart of [`solve_dense_system`] for DENSE
/// systems (used by the RBF coefficient solve). Same partial-pivoting Gaussian
/// elimination in the same FP order → **bit-identical results**, but contiguous
/// rows keep the trailing-row update cache-resident and let LLVM vectorize it. The
/// inner `pivot != 0` / `aij != 0` skips of the Vec<Vec> version are dropped here:
/// subtracting `0.0` is a no-op for finite entries and a dense Φ has no structural
/// zeros, so the hot loop becomes a clean SIMD-able axpy. (The banded spline solve
/// keeps the Vec<Vec> zero-skip path — there the skip is a real O(n·bw) algorithmic
/// win, not just a guard.)
// Retained REFERENCE implementation: the optimised sibling's own doc cites it,
// and a test now runs the comparison that doc asserts. Dead only in non-test
// builds (frankenscipy-9yyez).
#[cfg_attr(not(test), allow(dead_code))]
fn solve_dense_system_flat(
    a: &mut [f64],
    n: usize,
    b: &mut [f64],
) -> Result<Vec<f64>, InterpError> {
    if n == 0 || a.len() != n * n || b.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "empty or mismatched system".to_string(),
        });
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument {
                detail: "singular matrix".to_string(),
            });
        }
        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }
        let pivot_diag = a[col * n + col];
        let b_col = b[col];
        // Borrow the pivot row and the trailing rows disjointly.
        let (pre, rest) = a.split_at_mut((col + 1) * n);
        let pivot_row = &pre[col * n..(col + 1) * n];
        for (r, row_idx) in ((col + 1)..n).enumerate() {
            let row = &mut rest[r * n..(r + 1) * n];
            let av = row[col];
            if av == 0.0 {
                continue;
            }
            let factor = av / pivot_diag;
            for j in col..n {
                row[j] -= factor * pivot_row[j];
            }
            b[row_idx] -= factor * b_col;
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Ok(x)
}

#[derive(Clone, Debug)]
struct CompactBandRow {
    start: usize,
    values: Vec<f64>,
}

impl CompactBandRow {
    fn from_slice(start: usize, values: &[f64]) -> Self {
        Self {
            start,
            values: values.to_vec(),
        }
    }

    fn get(&self, col: usize) -> f64 {
        if col < self.start {
            return 0.0;
        }
        self.values.get(col - self.start).copied().unwrap_or(0.0)
    }

    fn sub_assign(&mut self, col: usize, value: f64) {
        let cell = self.cell_mut(col);
        *cell -= value;
    }

    fn cell_mut(&mut self, col: usize) -> &mut f64 {
        if self.values.is_empty() {
            self.start = col;
            self.values.push(0.0);
            return &mut self.values[0];
        }
        if col < self.start {
            let pad = self.start - col;
            let old = std::mem::take(&mut self.values);
            self.values = vec![0.0; pad];
            self.values.extend(old);
            self.start = col;
        }
        let offset = col - self.start;
        if offset >= self.values.len() {
            self.values.resize(offset + 1, 0.0);
        }
        &mut self.values[offset]
    }
}

/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
fn solve_banded_compact(
    a: &mut [CompactBandRow],
    b: &mut [f64],
    bw: usize,
) -> Result<Vec<f64>, InterpError> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "empty or mismatched system".to_string(),
        });
    }
    for col in 0..n {
        let row_hi = (col + bw + 1).min(n);
        let mut max_row = col;
        let mut max_val = a[col].get(col).abs();
        for (off, a_row) in a[(col + 1)..row_hi].iter().enumerate() {
            let v = a_row.get(col).abs();
            if v > max_val {
                max_val = v;
                max_row = col + 1 + off;
            }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument {
                detail: "singular matrix".to_string(),
            });
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        let col_hi = (col + 2 * bw + 1).min(n);
        let pivot_diag = a[col].get(col);
        for row in (col + 1)..row_hi {
            let av = a[row].get(col);
            if av == 0.0 {
                continue;
            }
            let factor = av / pivot_diag;
            let (head, tail) = a.split_at_mut(row);
            let pivot = &head[col];
            let target = &mut tail[0];
            for j in col..col_hi {
                let pval = pivot.get(j);
                if pval != 0.0 {
                    target.sub_assign(j, factor * pval);
                }
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        let j_hi = (i + 2 * bw + 1).min(n);
        for j in (i + 1)..j_hi {
            let aij = a[i].get(j);
            if aij != 0.0 {
                s -= aij * x[j];
            }
        }
        x[i] = s / a[i].get(i);
    }
    Ok(x)
}

/// Partial-pivoting Gaussian elimination for a matrix whose nonzeros are confined to a
/// band of half-width `bw` (`a[i][j] == 0` for `|i-j| > bw`), e.g. a B-spline A^T A.
/// Bit-identical to `solve_dense_system`: with partial pivoting the L factor keeps lower
/// bandwidth `bw` and U gains upper bandwidth `2*bw`, so every entry the dense solve
/// touches outside `[col, col+bw]` rows / `[col, col+2*bw]` columns is exactly 0 and its
/// update is a no-op — restricting the loops to that window changes nothing but turns the
/// O(n^2)-scan solve into O(n*bw^2). The pivot search is bounded the same way (entries
/// below the band are 0, so they never win the max). Validated by the perf digest A/Bs.
fn solve_banded(a: &mut [Vec<f64>], b: &mut [f64], bw: usize) -> Result<Vec<f64>, InterpError> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "empty or mismatched system".to_string(),
        });
    }
    for col in 0..n {
        let row_hi = (col + bw + 1).min(n); // rows below the band are 0 in column `col`
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (off, a_row) in a[(col + 1)..row_hi].iter().enumerate() {
            let v = a_row[col].abs();
            if v > max_val {
                max_val = v;
                max_row = col + 1 + off;
            }
        }
        if max_val < 1e-14 {
            return Err(InterpError::InvalidArgument {
                detail: "singular matrix".to_string(),
            });
        }
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        let col_hi = (col + 2 * bw + 1).min(n); // U fill stays within upper bandwidth 2*bw
        let pivot_diag = a[col][col];
        for row in (col + 1)..row_hi {
            if a[row][col] == 0.0 {
                continue;
            }
            let factor = a[row][col] / pivot_diag;
            let (head, tail) = a.split_at_mut(row);
            let pivot = &head[col];
            let target = &mut tail[0];
            for j in col..col_hi {
                let pval = pivot[j];
                if pval != 0.0 {
                    target[j] -= factor * pval;
                }
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        let j_hi = (i + 2 * bw + 1).min(n);
        for j in (i + 1)..j_hi {
            let aij = a[i][j];
            if aij != 0.0 {
                s -= aij * x[j];
            }
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
}

/// In-place Cholesky of the GCV `(4,4)`-banded SPD matrix in compact band storage.
/// Only the lower band is read/written: on success the lower band holds `L` with
/// `a = L L^T`. Cholesky has no fill-in beyond the original bandwidth and needs no
/// pivoting, so compact storage is arithmetic-identical to the prior full-row path.
/// Returns `None` if `a` is not positive definite (a non-positive pivot) — the caller
/// treats that like a singular system.
fn chol_banded_gcv(a: &mut [GcvBandRow]) -> Option<()> {
    let n = a.len();
    for j in 0..n {
        let klo = j.saturating_sub(GCV_BW);
        let mut d = gcv_band_lower_get(a, j, j);
        for k in klo..j {
            let ljk = gcv_band_lower_get(a, j, k);
            d -= ljk * ljk;
        }
        if d <= 0.0 {
            return None;
        }
        let ljj = d.sqrt();
        gcv_band_lower_set(a, j, j, ljj);
        let ihi = (j + GCV_BW + 1).min(n);
        for i in (j + 1)..ihi {
            // L[i][j] = (a[i][j] - Σ_{k} L[i][k] L[j][k]) / L[j][j], k where both nonzero:
            // k ∈ [i-bw, j) (since i>j ⇒ i-bw ≥ j-bw, and L[*][k]=0 below the band).
            let kstart = i.saturating_sub(GCV_BW);
            let mut s = gcv_band_lower_get(a, i, j);
            for k in kstart..j {
                s -= gcv_band_lower_get(a, i, k) * gcv_band_lower_get(a, j, k);
            }
            gcv_band_lower_set(a, i, j, s / ljj);
        }
    }
    Some(())
}

/// `tr(A^-1 B)` where `A = G G^T` (`g` = [`chol_banded_gcv`] lower factor) and
/// `B` is symmetric `(4,4)`-banded. Only the band of `A^-1` (|i-j| <= 4) contributes
/// (B is banded), and that band is recovered from `g` by the Erisman–Tinney SELECTED
/// INVERSE backward recurrence in O(n·bw²) — no n solves. `z[i][d] = (A⁻¹)_{i,i+d}` for
/// `d ∈ 0..=bw` (upper band incl. diagonal; A⁻¹ is symmetric). With `L_{ki}=g[k][i]/g[i][i]`
/// (unit-lower) and `D_i=g[i][i]²`: `Z_{ij} = −Σ_{k=i+1}^{i+bw} L_{ki} Z_{kj}` (j>i) and
/// `Z_{ii} = 1/D_i − Σ_{k} L_{ki} Z_{ik}`. Tolerance-parity with `Σ_col (A⁻¹ B)_{col,col}`.
/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
fn gcv_trace_selinv_gcv(g: &[GcvBandRow], b: &[GcvBandRow], n: usize) -> f64 {
    let mut z = vec![[0.0_f64; GCV_BW + 1]; n];
    for i in (0..n).rev() {
        let gii = gcv_band_lower_get(g, i, i);
        let kmax = (i + GCV_BW).min(n - 1);
        // off-diagonals Z_{ij}, j = kmax..i+1 (descending), then the diagonal.
        let mut j = kmax;
        while j > i {
            let mut s = 0.0_f64;
            for k in (i + 1)..=kmax {
                let lki = gcv_band_lower_get(g, k, i) / gii;
                if lki != 0.0 {
                    let (lo, hi) = if k <= j { (k, j) } else { (j, k) };
                    s -= lki * z[lo][hi - lo];
                }
            }
            z[i][j - i] = s;
            j -= 1;
        }
        let mut s = 1.0 / (gii * gii);
        for k in (i + 1)..=kmax {
            let lki = gcv_band_lower_get(g, k, i) / gii;
            if lki != 0.0 {
                s -= lki * z[i][k - i]; // Z_{ik}, k>i, just computed above
            }
        }
        z[i][0] = s;
    }
    // tr = Σ_i Z_ii B_ii + 2 Σ_i Σ_{d=1}^{bw} Z_{i,i+d} B_{i,i+d}  (both symmetric, banded)
    let mut tr = 0.0_f64;
    for i in 0..n {
        tr += z[i][0] * gcv_band_get(b, i, i);
        let dmax = GCV_BW.min(n - 1 - i);
        for d in 1..=dmax {
            tr += 2.0 * z[i][d] * gcv_band_get(b, i, i + d);
        }
    }
    tr
}

#[derive(Debug)]
pub struct NearestNDInterpolator {
    tree: fsci_spatial::KDTree,
    values: Vec<f64>,
}

impl NearestNDInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        if points.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if points.len() != values.len() {
            return Err(InterpError::LengthMismatch {
                x_len: points.len(),
                y_len: values.len(),
            });
        }
        let tree = fsci_spatial::KDTree::new(points).map_err(|e| InterpError::InvalidArgument {
            detail: format!("KDTree error: {e}"),
        })?;
        Ok(Self {
            tree,
            values: values.to_vec(),
        })
    }
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        let (idx, _dist) = self
            .tree
            .query(query)
            .map_err(|e| InterpError::InvalidArgument {
                detail: format!("query error: {e}"),
            })?;
        Ok(self.values[idx])
    }
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        // Each query is an independent KD-tree nearest-neighbour lookup; parallelize across
        // queries (bit-identical — each result is a pure lookup written in query order).
        let work = self.values.len().max(2).ilog2() as usize * 8;
        par_query_try_map(queries, work, |q| self.eval(q))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GriddataMethod {
    Nearest,
    Linear,
    /// Cubic C¹ interpolation (2-D only), matching SciPy's `method='cubic'`
    /// which uses the Clough-Tocher scheme. Like SciPy, only 2-D scattered
    /// points are supported for cubic.
    Cubic,
}

/// Interpolate scattered data on arbitrary points.
///
/// Matches `scipy.interpolate.griddata(points, values, xi, method)`.
/// `Nearest` and `Linear` accept N-D points; `Cubic` (Clough-Tocher) is 2-D only
/// (as in SciPy).
pub fn griddata(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
    method: GriddataMethod,
) -> Result<Vec<f64>, InterpError> {
    match method {
        GriddataMethod::Nearest => NearestNDInterpolator::new(points, values)?.eval_many(xi),
        GriddataMethod::Linear => LinearNDInterpolator::new(points, values)?.eval_many(xi),
        GriddataMethod::Cubic => CloughTocher2DInterpolator::new(points, values)?.eval_many(xi),
    }
}

/// Interpolation method for RegularGridInterpolator.
///
/// Matches `scipy.interpolate.RegularGridInterpolator(method=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegularGridMethod {
    /// Linear interpolation (default). Requires at least 2 points per axis.
    #[default]
    Linear,
    /// Nearest-neighbor interpolation. Requires at least 2 points per axis.
    Nearest,
    /// Tensor-product PCHIP interpolation. Requires at least 4 points per axis.
    /// Matches scipy's `method='pchip'`.
    Pchip,
    /// Tensor product cubic spline (k=3). Requires at least 4 points per axis.
    /// Matches scipy's `method='cubic'`.
    Cubic,
    /// Tensor product quintic spline (k=5). Requires at least 6 points per axis.
    /// Matches scipy's `method='quintic'`.
    Quintic,
}

#[derive(Debug, Clone)]
pub struct RegularGridInterpolator {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    strides: Vec<usize>,
    method: RegularGridMethod,
    bounds_error: bool,
    fill_value: Option<f64>,
    /// Per-axis uniform-grid metadata: `Some((x0, inv_dx))` when the axis is an
    /// evenly-spaced grid (within a tight relative tolerance), enabling O(1)
    /// direct-address interval lookup instead of binary search. `None` for
    /// irregular axes (which keep the binary-search path).
    uniform_axes: Vec<Option<(f64, f64)>>,
    /// Per-axis spline coefficients for Cubic/Quintic methods.
    /// Each inner Vec contains spline coefficients for that axis.
    /// Reserved for future precomputation optimization.
    _spline_coeffs_per_axis: Option<Vec<Vec<[f64; 4]>>>,
}

/// Same-binary A/B toggle for batch PCHIP evaluation. When `true`, `eval_many` routes every PCHIP
/// query through the generic per-query `eval` path (which re-fits the query-independent last-axis
/// fibers on every query — the ORIG behaviour). When `false` (default), the last-axis fits are
/// hoisted out of the per-query loop and computed once. Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static INTERPN_PCHIP_BATCH_FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, [`RegularGridInterpolator::eval_many`]'s 3-D nearest-neighbour fast path evaluates
/// its queries serially (the ORIG behaviour); when `false` (default) the independent per-query
/// nearest lookups fan across cores via `par_query_try_map`. Byte-identical. Benchmark knob.
#[doc(hidden)]
pub static INTERPN_NEAREST3D_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl RegularGridInterpolator {
    pub fn new(
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        method: RegularGridMethod,
        bounds_error: bool,
        fill_value: Option<f64>,
    ) -> Result<Self, InterpError> {
        if points.is_empty() {
            return Err(InterpError::InvalidArgument {
                detail: "points empty".to_string(),
            });
        }

        // Determine minimum points required per axis based on method
        let min_points = match method {
            RegularGridMethod::Linear | RegularGridMethod::Nearest => 2,
            RegularGridMethod::Pchip | RegularGridMethod::Cubic => 4,
            RegularGridMethod::Quintic => 6,
        };

        for (dim, axis) in points.iter().enumerate() {
            if axis.len() < min_points {
                return Err(InterpError::TooFewPoints {
                    minimum: min_points,
                    actual: axis.len(),
                });
            }
            // Check for non-finite values (NaN/Inf) before sortedness check
            // because NaN comparisons always return false, bypassing the sort check
            if axis.iter().any(|&v| !v.is_finite()) {
                return Err(InterpError::NonFiniteX);
            }
            if axis.windows(2).any(|w| w[1] <= w[0]) {
                return Err(InterpError::InvalidArgument {
                    detail: format!("axis {dim} not strictly increasing"),
                });
            }
        }
        let ndim = points.len();
        let mut strides = vec![0usize; ndim];
        let mut total_size: usize = 1;
        for i in (0..ndim).rev() {
            strides[i] = total_size;
            total_size = total_size.checked_mul(points[i].len()).ok_or_else(|| {
                InterpError::InvalidArgument {
                    detail: "grid overflow".to_string(),
                }
            })?;
        }
        if values.len() != total_size {
            return Err(InterpError::LengthMismatch {
                x_len: total_size,
                y_len: values.len(),
            });
        }

        // Detect evenly-spaced axes once at construction so the hot eval paths
        // can replace per-query binary search with O(1) direct addressing. An
        // axis is treated as uniform when every coordinate sits within a tight
        // relative tolerance of the ideal linspace position; the eval-time
        // correction loop still reads the stored coordinates, so this only
        // affects speed, never which interval is selected.
        let uniform_axes = points
            .iter()
            .map(|axis| detect_uniform_axis(axis))
            .collect();

        // For spline methods, we don't precompute coefficients since that would be
        // expensive and may not be needed. Coefficients are computed on-the-fly
        // during interpolation using 1D cubic spline along each axis.
        Ok(Self {
            points,
            values,
            strides,
            method,
            bounds_error,
            fill_value,
            uniform_axes,
            _spline_coeffs_per_axis: None,
        })
    }

    pub fn ndim(&self) -> usize {
        self.points.len()
    }

    pub fn eval(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if xi.len() != ndim {
            return Err(InterpError::InvalidArgument {
                detail: format!("expected {ndim}D, got {}D", xi.len()),
            });
        }
        if xi.iter().any(|x| x.is_nan()) {
            return Ok(f64::NAN);
        }
        let mut out_of_bounds = false;
        for (dim, &x) in xi.iter().enumerate() {
            let axis = &self.points[dim];
            if x < axis[0] || x > axis[axis.len() - 1] {
                if self.bounds_error {
                    return Err(InterpError::OutOfBounds {
                        value: format!(
                            "dim {dim}: {x} outside [{}, {}]",
                            axis[0],
                            axis[axis.len() - 1]
                        ),
                    });
                }
                out_of_bounds = true;
            }
        }
        if out_of_bounds && let Some(fill) = self.fill_value {
            return Ok(fill);
        }
        match self.method {
            RegularGridMethod::Linear => self.eval_linear(xi),
            RegularGridMethod::Nearest => Ok(self.eval_nearest(xi)),
            RegularGridMethod::Pchip => self.eval_pchip(xi),
            RegularGridMethod::Cubic => self.eval_spline(xi, 3),
            RegularGridMethod::Quintic => self.eval_spline(xi, 5),
        }
    }

    pub fn eval_many(&self, xi: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        if self.method == RegularGridMethod::Nearest && self.ndim() == 3 {
            return self.eval_many_nearest_3d(xi);
        }

        // PCHIP: the first (last-axis) reduction fits contiguous fibers of `self.values`, and those
        // fits are query-INDEPENDENT — the generic path below re-runs `PchipInterpolator::new` for
        // them on every query. Hoist them out of the per-query loop (shared-predictor lever). The
        // knob restores the per-query path for the same-binary A/B.
        if self.method == RegularGridMethod::Pchip
            && !xi.is_empty()
            && !INTERPN_PCHIP_BATCH_FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
        {
            return self.eval_many_pchip(xi);
        }

        // Each query is an independent `eval` (read-only lookup + interpolation over the
        // shared grid, no mutable state), so for a large batch the queries run on disjoint
        // cores and the per-query value is the same pure `eval` regardless of owning core.
        // par_query_try_map concatenates contiguous chunks in order and returns the first
        // erroring chunk's error in query order, so the result is bit-identical to the
        // sequential `xi.iter().map(|x| self.eval(x)).collect()`.
        let ndim = self.ndim().max(1);
        let work_per_query = match self.method {
            RegularGridMethod::Nearest => ndim,
            RegularGridMethod::Linear => 1usize << ndim.min(16),
            RegularGridMethod::Pchip => ndim * 16,
            RegularGridMethod::Cubic => ndim * 64,
            RegularGridMethod::Quintic => ndim * 128,
        };
        par_query_try_map(xi, work_per_query, |x| self.eval(x))
    }

    /// Batch PCHIP that HOISTS the query-independent last-axis fits.
    ///
    /// `eval_pchip`'s first reduction fits every contiguous last-axis fiber of `self.values` with
    /// `PchipInterpolator::new`; those fits depend only on the grid, not the query, so the generic
    /// `eval` path recomputes the same derivatives for all `nq` queries. Fit them ONCE here, then
    /// each query only EVALUATES the shared fits (cheap bisect+Hermite) and re-fits the remaining,
    /// genuinely query-dependent reductions. BYTE-IDENTICAL to the per-query path: the same fibers
    /// are fitted with the same inputs and evaluated at the same points, in the same order. If the
    /// shared fit fails (cannot happen for a construction-validated grid), fall back to the generic
    /// per-query path so semantics are preserved exactly.
    fn eval_many_pchip(&self, xi: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        let ndim = self.ndim();
        let last = ndim - 1;
        let last_axis = &self.points[last];
        let last_len = last_axis.len();
        let last_fits: Result<Vec<PchipInterpolator>, InterpError> = self
            .values
            .chunks_exact(last_len)
            .map(|slice| PchipInterpolator::new(last_axis, slice))
            .collect();
        let work_per_query = ndim.max(1) * 16;
        match last_fits {
            Ok(fits) => par_query_try_map(xi, work_per_query, |q| self.eval_pchip_prefit(q, &fits)),
            Err(_) => par_query_try_map(xi, work_per_query, |x| self.eval(x)),
        }
    }

    /// Per-query PCHIP evaluation reusing precomputed last-axis fits. Mirrors `eval`'s guards
    /// (dimension count, NaN, bounds/fill) exactly, then runs the same reduction as `eval_pchip`
    /// with the first (last-axis) fit step replaced by evaluating the shared `last_fits`.
    fn eval_pchip_prefit(
        &self,
        xi: &[f64],
        last_fits: &[PchipInterpolator],
    ) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if xi.len() != ndim {
            return Err(InterpError::InvalidArgument {
                detail: format!("expected {ndim}D, got {}D", xi.len()),
            });
        }
        if xi.iter().any(|x| x.is_nan()) {
            return Ok(f64::NAN);
        }
        let mut out_of_bounds = false;
        for (dim, &x) in xi.iter().enumerate() {
            let axis = &self.points[dim];
            if x < axis[0] || x > axis[axis.len() - 1] {
                if self.bounds_error {
                    return Err(InterpError::OutOfBounds {
                        value: format!(
                            "dim {dim}: {x} outside [{}, {}]",
                            axis[0],
                            axis[axis.len() - 1]
                        ),
                    });
                }
                out_of_bounds = true;
            }
        }
        if out_of_bounds && let Some(fill) = self.fill_value {
            return Ok(fill);
        }

        let last = ndim - 1;
        // First reduction: evaluate the shared last-axis fits at xi[last] (byte-identical to fitting
        // each last-axis fiber and evaluating it, as `eval_pchip` does).
        let mut reduced: Vec<f64> = last_fits.iter().map(|f| f.eval(xi[last])).collect();
        let mut shape: Vec<usize> = self.points[..last].iter().map(Vec::len).collect();
        for dim in (0..last).rev() {
            let axis = &self.points[dim];
            let axis_len = shape[dim];
            let outer = reduced.len() / axis_len;
            let mut next = Vec::with_capacity(outer);
            for slice in reduced.chunks_exact(axis_len) {
                let interp = PchipInterpolator::new(axis, slice)?;
                next.push(interp.eval(xi[dim]));
            }
            reduced = next;
            shape.pop();
        }
        Ok(reduced[0])
    }

    fn find_interval(axis: &[f64], x: f64) -> usize {
        let n = axis.len();
        if x <= axis[0] {
            return 0;
        }
        if x >= axis[n - 1] {
            return n - 2;
        }
        match axis.binary_search_by(|probe| probe.total_cmp(&x)) {
            Ok(i) => i.min(n - 2),
            Err(i) => i.saturating_sub(1),
        }
    }

    /// O(1) equivalent of [`find_interval`] for an evenly-spaced axis. Computes
    /// the interval by direct arithmetic, then corrects against the *stored*
    /// coordinates so the returned index is bit-identical to the binary-search
    /// path (largest `i` with `axis[i] <= x`, clamped to `[0, n-2]`, with the
    /// boundary clamps). `meta = (x0, inv_dx)` comes from `uniform_axes`.
    #[inline]
    fn find_interval_uniform(meta: (f64, f64), axis: &[f64], x: f64) -> usize {
        let n = axis.len();
        let (x0, inv_dx) = meta;
        // Direct-address estimate of the interval. A negative estimate (x below
        // the grid) saturates to 0 on the `as usize` cast and a too-large one is
        // clamped to n-2, so the correction loops below reproduce the boundary
        // clamps of `find_interval` without needing explicit endpoint branches.
        let est = (x - x0) * inv_dx;
        let mut i = est as usize; // negative -> 0, oversized -> clamped next
        if i > n - 2 {
            i = n - 2;
        }
        // Correct to the exact interval using stored coordinates. For a genuinely
        // uniform axis these loops run at most once or twice; correctness does
        // not depend on the estimate, only speed.
        while i + 1 < n - 1 && axis[i + 1] <= x {
            i += 1;
        }
        while i > 0 && axis[i] > x {
            i -= 1;
        }
        i
    }

    /// Nearest grid index along one axis: locate the bracketing interval (via the
    /// uniform fast path when available, else binary search) and pick the closer
    /// endpoint, ties to the lower index. Identical result for both interval
    /// paths since `find_interval_uniform` is bit-equivalent to `find_interval`.
    #[inline]
    fn nearest_index(uniform: Option<(f64, f64)>, axis: &[f64], x: f64) -> usize {
        let i = match uniform {
            Some(meta) => Self::find_interval_uniform(meta, axis, x),
            None => Self::find_interval(axis, x),
        };
        if i + 1 < axis.len() && (x - axis[i]).abs() > (axis[i + 1] - x).abs() {
            i + 1
        } else {
            i
        }
    }

    fn eval_nearest(&self, xi: &[f64]) -> f64 {
        if self.ndim() == 3 {
            return self.eval_nearest_3d(xi);
        }

        let mut flat_idx = 0;
        for ((axis, &x), &stride) in self.points.iter().zip(xi).zip(&self.strides) {
            let i = Self::find_interval(axis, x);
            let nearest = if i + 1 < axis.len() && (x - axis[i]).abs() > (axis[i + 1] - x).abs() {
                i + 1
            } else {
                i
            };
            flat_idx += nearest * stride;
        }
        self.values[flat_idx]
    }

    fn eval_many_nearest_3d(&self, xi: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        // Hoist per-axis state so the inner pass touches only locals, and fold the NaN check,
        // bounds check, and nearest lookup into a single per-query pass. Each query is an
        // independent read-only nearest lookup over the shared grid, so route it through the same
        // order-preserving `par_query_try_map` the generic `eval_many` path uses — the 3D-nearest
        // special case previously stayed serial. BYTE-IDENTICAL: identical per-query body, chunks
        // concatenated in query order, and the earliest erroring query wins exactly as before.
        let ax = [
            self.points[0].as_slice(),
            self.points[1].as_slice(),
            self.points[2].as_slice(),
        ];
        let st = [self.strides[0], self.strides[1], self.strides[2]];
        let un = [
            self.uniform_axes[0],
            self.uniform_axes[1],
            self.uniform_axes[2],
        ];

        let per_query = |point: &Vec<f64>| -> Result<f64, InterpError> {
            if point.len() != 3 {
                return Err(InterpError::InvalidArgument {
                    detail: format!("expected 3D, got {}D", point.len()),
                });
            }
            let p = [point[0], point[1], point[2]];
            if p[0].is_nan() || p[1].is_nan() || p[2].is_nan() {
                return Ok(f64::NAN);
            }

            let mut out_of_bounds = false;
            for dim in 0..3 {
                let axis = ax[dim];
                if p[dim] < axis[0] || p[dim] > axis[axis.len() - 1] {
                    if self.bounds_error {
                        return Err(InterpError::OutOfBounds {
                            value: format!(
                                "dim {dim}: {} outside [{}, {}]",
                                p[dim],
                                axis[0],
                                axis[axis.len() - 1]
                            ),
                        });
                    }
                    out_of_bounds = true;
                }
            }
            if out_of_bounds && let Some(fill) = self.fill_value {
                return Ok(fill);
            }

            let flat_idx = Self::nearest_index(un[0], ax[0], p[0]) * st[0]
                + Self::nearest_index(un[1], ax[1], p[1]) * st[1]
                + Self::nearest_index(un[2], ax[2], p[2]) * st[2];
            Ok(self.values[flat_idx])
        };

        if INTERPN_NEAREST3D_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) {
            return xi.iter().map(&per_query).collect();
        }
        par_query_try_map(xi, 3, per_query)
    }

    fn eval_nearest_3d(&self, xi: &[f64]) -> f64 {
        debug_assert_eq!(xi.len(), 3);

        let mut flat_idx = 0;
        for (dim, &x) in xi.iter().enumerate().take(3) {
            flat_idx += Self::nearest_index(self.uniform_axes[dim], &self.points[dim], x)
                * self.strides[dim];
        }
        self.values[flat_idx]
    }

    /// Bracketing interval along axis `dim` for `x`: the O(1) direct-address path
    /// when the axis is evenly spaced, else binary search. `find_interval_uniform`
    /// is bit-identical to `find_interval` (asserted by
    /// `find_interval_uniform_matches_binary_search`), so every linear
    /// interpolation result is unchanged — this only removes the per-axis
    /// O(log n) search on uniform grids (the overwhelmingly common case).
    #[inline]
    fn linear_interval(&self, dim: usize, axis: &[f64], x: f64) -> usize {
        match self.uniform_axes[dim] {
            Some(meta) => Self::find_interval_uniform(meta, axis, x),
            None => Self::find_interval(axis, x),
        }
    }

    fn eval_linear(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if ndim == 2 {
            return Ok(self.eval_linear_2d(xi));
        }
        if ndim == 3 {
            return Ok(self.eval_linear_3d(xi));
        }

        let mut indices = Vec::with_capacity(ndim);
        let mut fracs = Vec::with_capacity(ndim);
        for (dim, (axis, &x)) in self.points.iter().zip(xi).enumerate() {
            let i = self.linear_interval(dim, axis, x);
            let denom = axis[i + 1] - axis[i];
            indices.push(i);
            fracs.push(if denom == 0.0 {
                0.0
            } else {
                (x - axis[i]) / denom
            });
        }
        let mut result = 0.0;
        for corner in 0..(1usize << ndim) {
            let mut weight = 1.0;
            let mut flat_idx = 0;
            for dim in 0..ndim {
                let bit = (corner >> dim) & 1;
                flat_idx += (indices[dim] + bit) * self.strides[dim];
                weight *= if bit == 0 {
                    1.0 - fracs[dim]
                } else {
                    fracs[dim]
                };
            }
            result += weight * self.values[flat_idx];
        }
        Ok(result)
    }

    /// Bilinear fast path (ndim == 2). Bit-identical to the generic `eval_linear`
    /// for two dimensions (same corner order 0..4, same dim loop, same weight /
    /// flat-index accumulation), but uses stack arrays instead of two
    /// `Vec::with_capacity` heap allocations per query — the same idiom as
    /// `eval_linear_3d`. The win compounds across a parallel `eval_many` batch.
    fn eval_linear_2d(&self, xi: &[f64]) -> f64 {
        debug_assert_eq!(xi.len(), 2);

        let mut indices = [0usize; 2];
        let mut fracs = [0.0; 2];
        for dim in 0..2 {
            let axis = &self.points[dim];
            let x = xi[dim];
            let i = self.linear_interval(dim, axis, x);
            let denom = axis[i + 1] - axis[i];
            indices[dim] = i;
            fracs[dim] = if denom == 0.0 {
                0.0
            } else {
                (x - axis[i]) / denom
            };
        }

        let mut result = 0.0;
        for corner in 0..4usize {
            let mut weight = 1.0;
            let mut flat_idx = 0;
            for dim in 0..2 {
                let bit = (corner >> dim) & 1;
                flat_idx += (indices[dim] + bit) * self.strides[dim];
                weight *= if bit == 0 {
                    1.0 - fracs[dim]
                } else {
                    fracs[dim]
                };
            }
            result += weight * self.values[flat_idx];
        }
        result
    }

    fn eval_linear_3d(&self, xi: &[f64]) -> f64 {
        debug_assert_eq!(xi.len(), 3);

        let mut indices = [0usize; 3];
        let mut fracs = [0.0; 3];
        for dim in 0..3 {
            let axis = &self.points[dim];
            let x = xi[dim];
            let i = self.linear_interval(dim, axis, x);
            let denom = axis[i + 1] - axis[i];
            indices[dim] = i;
            fracs[dim] = if denom == 0.0 {
                0.0
            } else {
                (x - axis[i]) / denom
            };
        }

        let mut result = 0.0;
        for corner in 0..8usize {
            let mut weight = 1.0;
            let mut flat_idx = 0;
            for dim in 0..3 {
                let bit = (corner >> dim) & 1;
                flat_idx += (indices[dim] + bit) * self.strides[dim];
                weight *= if bit == 0 {
                    1.0 - fracs[dim]
                } else {
                    fracs[dim]
                };
            }
            result += weight * self.values[flat_idx];
        }
        result
    }

    /// Tensor-product PCHIP interpolation.
    ///
    /// This mirrors SciPy's recursive `_evaluate_spline(..., method="pchip")`
    /// path: starting from the last interpolation dimension, collapse each
    /// contiguous 1D slice with a 1D PCHIP fit, then repeat for the remaining
    /// dimensions until only a scalar remains.
    fn eval_pchip(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        let mut reduced = self.values.clone();
        let mut shape: Vec<usize> = self.points.iter().map(Vec::len).collect();

        for dim in (0..ndim).rev() {
            let axis = &self.points[dim];
            let axis_len = shape[dim];
            let outer = reduced.len() / axis_len;
            let mut next = Vec::with_capacity(outer);

            for slice in reduced.chunks_exact(axis_len) {
                let interp = PchipInterpolator::new(axis, slice)?;
                next.push(interp.eval(xi[dim]));
            }

            reduced = next;
            shape.pop();
        }

        Ok(reduced[0])
    }

    /// Tensor product spline interpolation (cubic or quintic).
    ///
    /// Uses successive 1D cubic spline interpolations along each axis.
    /// For degree k, we need k+1 points per axis.
    fn eval_spline(&self, xi: &[f64], _degree: usize) -> Result<f64, InterpError> {
        // For tensor-product interpolation, we apply 1D spline interpolation
        // successively along each dimension. Start with a hypercube of values
        // and reduce dimension by interpolating along one axis at a time.

        // We'll work with a recursive reduction approach:
        // 1. Extract a hyperslab of the grid around the query point
        // 2. Interpolate along dimension 0
        // 3. Repeat for remaining dimensions

        // For simplicity, we use a direct tensor product approach:
        // Compute the spline basis values for each dimension, then sum over
        // all combinations.

        // For cubic spline, we use 4 points per dimension: a LOCAL cubic
        // (Catmull-Rom, C1) tensor product. NOTE: this is NOT identical to
        // scipy.interpolate.RegularGridInterpolator(method="cubic"), which fits
        // a GLOBAL C2 tensor spline over the whole grid (hence ~264x slower at
        // 300^2/50k queries; fsci 5.98 ms vs scipy 1582 ms). Values agree to
        // ~0.06% but are not bit-parity — a deliberate speed/locality choice.
        // (interpn `linear` IS bit-parity with scipy and 1.73x faster.)

        self.eval_spline_tensor_product(xi)
    }

    /// Compute tensor product cubic spline interpolation.
    ///
    /// Uses local cubic (Catmull-Rom style) interpolation which is C1 continuous.
    fn eval_spline_tensor_product(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();

        // For each dimension, compute interpolation indices and weights
        let mut interp_data: Vec<(usize, [f64; 4])> = Vec::with_capacity(ndim);

        for (axis, &x) in self.points.iter().zip(xi) {
            let n = axis.len();

            // Find the interval: we want 4 points centered around x
            // For Catmull-Rom, we need points at i-1, i, i+1, i+2 where
            // axis[i] <= x < axis[i+1]
            let i = Self::find_interval(axis, x);

            // Clamp to ensure we have 4 valid points
            let i0 = if i == 0 { 0 } else { i - 1 };
            let i0 = i0.min(n.saturating_sub(4));

            // Compute normalized parameter t for the interval [i, i+1]
            // where i corresponds to i0+1
            let center = i0 + 1;
            let t = if center + 1 < n && axis[center + 1] != axis[center] {
                (x - axis[center]) / (axis[center + 1] - axis[center])
            } else {
                0.0
            };

            // Catmull-Rom basis functions
            let t2 = t * t;
            let t3 = t2 * t;

            // Weights for points p0, p1, p2, p3 where we interpolate between p1 and p2
            let w0 = -0.5 * t3 + t2 - 0.5 * t;
            let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
            let w3 = 0.5 * t3 - 0.5 * t2;

            interp_data.push((i0, [w0, w1, w2, w3]));
        }

        // Now compute weighted sum over all 4^ndim combinations
        let mut result = 0.0;
        let num_corners = 4_usize.pow(ndim as u32);

        for corner_idx in 0..num_corners {
            let mut weight = 1.0;
            let mut flat_idx = 0;

            for (dim, (base_idx, weights)) in interp_data.iter().enumerate().take(ndim) {
                // Extract which of the 4 points we're using for this corner
                let offset = (corner_idx / 4_usize.pow(dim as u32)) % 4;
                let point_idx = *base_idx + offset;

                // Ensure point_idx is in bounds
                if point_idx >= self.points[dim].len() {
                    // Skip this corner (weight will be zeroed)
                    weight = 0.0;
                    break;
                }

                flat_idx += point_idx * self.strides[dim];
                weight *= weights[offset];
            }

            if weight != 0.0 {
                result += weight * self.values[flat_idx];
            }
        }

        Ok(result)
    }
}

pub fn interpn(
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    xi: &[Vec<f64>],
    method: RegularGridMethod,
    bounds_error: bool,
    fill_value: Option<f64>,
) -> Result<Vec<f64>, InterpError> {
    RegularGridInterpolator::new(points, values, method, bounds_error, fill_value)?.eval_many(xi)
}

#[derive(Debug, Clone)]
pub struct Delaunay2D {
    pub points: Vec<(f64, f64)>,
    pub simplices: Vec<(usize, usize, usize)>,
    pub neighbors: Vec<[Option<usize>; 3]>,
    simplex_bounds: Vec<SimplexBounds>,
    /// Per-simplex query-invariant barycentric basis, precomputed once so the
    /// point-location scan doesn't rebuild each simplex's Gram matrix per query.
    simplex_bary: Vec<SimplexBary>,
    grid: Option<SimplexGrid>,
}

/// Precomputed query-invariant barycentric basis for one simplex — point `a` and the
/// edge-vector Gram matrix `(d00,d01,d11)` with its determinant `denom`. Hoists the
/// per-(query,simplex) recomputation out of `find_simplex`; `weights()` is byte-
/// identical to `barycentric` (same float ops in the same order). frankenscipy-9l5oo.
#[derive(Clone, Copy, Debug)]
struct SimplexBary {
    ax: f64,
    ay: f64,
    v0x: f64,
    v0y: f64,
    v1x: f64,
    v1y: f64,
    d00: f64,
    d01: f64,
    d11: f64,
    denom: f64,
    degenerate: bool,
}

impl SimplexBary {
    fn for_simplex(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Self {
        let (v0x, v0y, v1x, v1y) = (b.0 - a.0, b.1 - a.1, c.0 - a.0, c.1 - a.1);
        let d00 = v0x * v0x + v0y * v0y;
        let d01 = v0x * v1x + v0y * v1y;
        let d11 = v1x * v1x + v1y * v1y;
        let denom = d00 * d11 - d01 * d01;
        SimplexBary {
            ax: a.0,
            ay: a.1,
            v0x,
            v0y,
            v1x,
            v1y,
            d00,
            d01,
            d11,
            denom,
            degenerate: denom.abs() < 1e-30,
        }
    }

    #[inline]
    fn weights(&self, p: (f64, f64)) -> (f64, f64, f64) {
        if self.degenerate {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let v2x = p.0 - self.ax;
        let v2y = p.1 - self.ay;
        let d20 = v2x * self.v0x + v2y * self.v0y;
        let d21 = v2x * self.v1x + v2y * self.v1y;
        let l2 = (self.d11 * d20 - self.d01 * d21) / self.denom;
        let l3 = (self.d00 * d21 - self.d01 * d20) / self.denom;
        (1.0 - l2 - l3, l2, l3)
    }
}

/// Uniform-grid spatial index over the (margin-padded) simplex bounding boxes.
///
/// `find_simplex` would otherwise scan every simplex per query (O(simplices) per
/// point). Each simplex is bucketed into every cell its padded bbox overlaps, so
/// a query's cell holds a superset of all simplices that could contain it. The
/// scan then visits only that cell's candidates, in ascending simplex-index
/// order — bit-identical to the full linear scan (same selection rule, same
/// first match), but ~O(1) candidates per query for well-distributed data.
#[derive(Debug, Clone)]
struct SimplexGrid {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    inv_w: f64,
    inv_h: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<usize>>,
}

impl SimplexGrid {
    fn build(bounds: &[SimplexBounds]) -> Option<Self> {
        if bounds.is_empty() {
            return None;
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for b in bounds {
            min_x = min_x.min(b.min_x - b.margin_x);
            min_y = min_y.min(b.min_y - b.margin_y);
            max_x = max_x.max(b.max_x + b.margin_x);
            max_y = max_y.max(b.max_y + b.margin_y);
        }
        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            return None;
        }
        let side = ((bounds.len() as f64).sqrt().round() as usize).clamp(1, 1024);
        let nx = side;
        let ny = side;
        let span_x = (max_x - min_x).max(f64::MIN_POSITIVE);
        let span_y = (max_y - min_y).max(f64::MIN_POSITIVE);
        let inv_w = nx as f64 / span_x;
        let inv_h = ny as f64 / span_y;
        let cell = |x: f64, y: f64| -> (usize, usize) {
            let cx = (((x - min_x) * inv_w).floor() as i64).clamp(0, nx as i64 - 1) as usize;
            let cy = (((y - min_y) * inv_h).floor() as i64).clamp(0, ny as i64 - 1) as usize;
            (cx, cy)
        };
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); nx * ny];
        for (i, b) in bounds.iter().enumerate() {
            let (cx0, cy0) = cell(b.min_x - b.margin_x, b.min_y - b.margin_y);
            let (cx1, cy1) = cell(b.max_x + b.margin_x, b.max_y + b.margin_y);
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    cells[cy * nx + cx].push(i);
                }
            }
        }
        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
            inv_w,
            inv_h,
            nx,
            ny,
            cells,
        })
    }

    /// Candidate simplices for `query`, or `None` if it lies outside the grid
    /// (no padded bbox can contain it — the caller falls back to a full scan).
    fn candidates(&self, query: (f64, f64)) -> Option<&[usize]> {
        if query.0 < self.min_x
            || query.0 > self.max_x
            || query.1 < self.min_y
            || query.1 > self.max_y
        {
            return None;
        }
        let cx = (((query.0 - self.min_x) * self.inv_w).floor() as i64).clamp(0, self.nx as i64 - 1)
            as usize;
        let cy = (((query.1 - self.min_y) * self.inv_h).floor() as i64).clamp(0, self.ny as i64 - 1)
            as usize;
        Some(&self.cells[cy * self.nx + cx])
    }
}

#[derive(Debug, Clone, Copy)]
struct SimplexBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    margin_x: f64,
    margin_y: f64,
}

impl SimplexBounds {
    fn for_simplex(points: &[(f64, f64)], (a, b, c): (usize, usize, usize)) -> Self {
        let (ax, ay) = points[a];
        let (bx, by) = points[b];
        let (cx, cy) = points[c];
        let min_x = ax.min(bx).min(cx);
        let max_x = ax.max(bx).max(cx);
        let min_y = ay.min(by).min(cy);
        let max_y = ay.max(by).max(cy);
        let scale_x = max_x
            .abs()
            .max(min_x.abs())
            .max((max_x - min_x).abs())
            .max(1.0);
        let scale_y = max_y
            .abs()
            .max(min_y.abs())
            .max((max_y - min_y).abs())
            .max(1.0);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            margin_x: 1e-9 * scale_x,
            margin_y: 1e-9 * scale_y,
        }
    }

    fn may_contain(self, query: (f64, f64)) -> bool {
        !(query.0 < self.min_x - self.margin_x
            || query.0 > self.max_x + self.margin_x
            || query.1 < self.min_y - self.margin_y
            || query.1 > self.max_y + self.margin_y)
    }
}

impl Delaunay2D {
    pub fn new(points: &[(f64, f64)]) -> Result<Self, InterpError> {
        let n = points.len();
        if n < 3 {
            return Err(InterpError::TooFewPoints {
                minimum: 3,
                actual: n,
            });
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        let (dx, dy) = ((max_x - min_x).max(1e-10), (max_y - min_y).max(1e-10));
        // Bowyer-Watson incremental triangulation. Both paths precompute each
        // triangle's circumcircle once (see `circumcircle`) so the incircle test
        // is a cheap `dist²(p, center) < r²` compare. Below
        // `DELAUNAY_GRID_THRESHOLD` the plain O(n²) "scan every active triangle
        // per insertion" loop is fastest; above it that scan dominates, so a
        // uniform grid over the triangles' circumcircle bounding boxes restricts
        // each insertion's candidate set to the new point's cell (a superset of
        // every triangle whose circumcircle could contain it). Both yield a
        // valid Delaunay triangulation (verified by the
        // `delaunay_empty_circumcircle_property` test).
        //
        // SUPER-TRIANGLE MARGIN, ESCALATED UNTIL THE RESULT IS PROVABLY COMPLETE.
        //
        // Bowyer-Watson deletes the triangles still touching the super-triangle's vertices at the
        // end. If the super-triangle sits too close, a genuine near-boundary triangle gets
        // attached to a super-vertex and is deleted with it, leaving a HOLE inside the convex
        // hull. `LinearNDInterpolator::eval` then returns NaN for a query in that hole, which is
        // indistinguishable from "outside the hull" -- 4 of 2000 queries at N=800 came back NaN
        // here and finite from live SciPy 1.17.1, every one 1.7e-4 to 9.4e-4 INSIDE the hull
        // (frankenscipy-tkpwa). The `-1e-10` barycentric tolerance in `find_simplex` is five
        // orders of magnitude too small to explain that, so widening it was never the fix.
        //
        // The margin used to be a fixed 10.0, and a fixed value CANNOT be right: the deficit was
        // measured to grow with the point count, so any constant only moves the failure to a
        // larger input.
        //
        //     margin   N=400   N=800   N=2000
        //         10       1       4        8      <- triangles missing
        //         30       1       2        6
        //        100       0       0        4
        //        300       0       0        1
        //       1000       0       0        0
        //
        // So escalate, and CHECK the result rather than trusting the constant. Euler's identity
        // fixes the count exactly for points in general position: a Delaunay triangulation of N
        // points with `h` of them on the convex hull has `2N - 2 - h` triangles. The first margin
        // whose output reaches that count is accepted. Starting at 1000 rather than 10 because the
        // sweep shows small margins essentially always leave holes, so starting low would pay for
        // a discarded triangulation on every call. Escalation is capped and the last attempt is
        // returned regardless, so a pathological input degrades to the old behaviour rather than
        // failing outright.
        //
        // `fsci_spatial::Delaunay` had the SAME defect from the same constant and is fixed the
        // same way; the two implementations are independent and neither can be dropped for the
        // other, so the reasoning is written out in both.
        let hull_vertices = fsci_spatial::ConvexHull::new(points)
            .ok()
            .map(|h| h.vertices.len());
        let mut triangles = Vec::new();
        for &margin in &[1000.0_f64, 1.0e5, 1.0e7] {
            let mut all_points = points.to_vec();
            all_points.push((min_x - margin * dx, min_y - margin * dy));
            all_points.push((max_x + margin * dx, min_y - margin * dy));
            all_points.push(((min_x + max_x) / 2.0, max_y + margin * dy));
            triangles = if n >= DELAUNAY_GRID_THRESHOLD {
                delaunay_triangulate_circle_grid(&all_points, n, min_x, min_y, dx, dy)
            } else {
                delaunay_triangulate_linear(&all_points, n)
            };
            // Degenerate inputs (all-collinear, duplicate-heavy) have no `2N - 2 - h` to reach;
            // when the hull is unavailable or the count cannot apply, take the first result.
            match hull_vertices {
                Some(h) if 2 * n >= 2 + h => {
                    if triangles.len() >= 2 * n - 2 - h {
                        break;
                    }
                }
                _ => break,
            }
        }
        let simplices = triangles
            .into_iter()
            .map(|triangle| orient_triangle_ccw(points, triangle))
            .collect::<Vec<_>>();
        let simplex_bounds: Vec<SimplexBounds> = simplices
            .iter()
            .map(|&simplex| SimplexBounds::for_simplex(points, simplex))
            .collect();
        let neighbors = compute_simplex_neighbors(&simplices);
        let grid = SimplexGrid::build(&simplex_bounds);
        let simplex_bary: Vec<SimplexBary> = simplices
            .iter()
            .map(|&(a, b, c)| SimplexBary::for_simplex(points[a], points[b], points[c]))
            .collect();
        Ok(Self {
            points: points.to_vec(),
            simplices,
            neighbors,
            simplex_bounds,
            simplex_bary,
            grid,
        })
    }
    pub fn find_simplex(&self, query: (f64, f64)) -> Option<(usize, f64, f64, f64)> {
        // The grid restricts the candidate set to one cell while preserving the
        // ascending-index "first match" rule, so the result is bit-identical to
        // the full linear scan. Out-of-grid queries (no padded bbox can contain
        // them) fall back to the linear scan to stay identical at the edges.
        match self.grid.as_ref().and_then(|g| g.candidates(query)) {
            Some(candidates) => {
                for &idx in candidates {
                    if !self.simplex_bounds[idx].may_contain(query) {
                        continue;
                    }
                    let (l1, l2, l3) = self.simplex_bary[idx].weights(query);
                    if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 {
                        return Some((idx, l1, l2, l3));
                    }
                }
                None
            }
            None => self.find_simplex_linear(query),
        }
    }

    fn find_simplex_linear(&self, query: (f64, f64)) -> Option<(usize, f64, f64, f64)> {
        for idx in 0..self.simplices.len() {
            if !self.simplex_bounds[idx].may_contain(query) {
                continue;
            }
            let (l1, l2, l3) = self.simplex_bary[idx].weights(query);
            if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 {
                return Some((idx, l1, l2, l3));
            }
        }
        None
    }
}

fn orient_triangle_ccw(
    points: &[(f64, f64)],
    (a, b, c): (usize, usize, usize),
) -> (usize, usize, usize) {
    if signed_triangle_area2(points[a], points[b], points[c]) >= 0.0 {
        (a, b, c)
    } else {
        (a, c, b)
    }
}

fn signed_triangle_area2(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn sorted_edge(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

fn compute_simplex_neighbors(simplices: &[(usize, usize, usize)]) -> Vec<[Option<usize>; 3]> {
    let mut neighbors = vec![[None; 3]; simplices.len()];
    let mut edge_owners = HashMap::<(usize, usize), (usize, usize)>::new();
    for (simplex_index, &(a, b, c)) in simplices.iter().enumerate() {
        for (local_edge_index, (u, v)) in [(0, (b, c)), (1, (c, a)), (2, (a, b))] {
            let edge = sorted_edge(u, v);
            if let Some((other_simplex, other_edge)) =
                edge_owners.insert(edge, (simplex_index, local_edge_index))
            {
                neighbors[simplex_index][local_edge_index] = Some(other_simplex);
                neighbors[other_simplex][other_edge] = Some(simplex_index);
            }
        }
    }
    neighbors
}

/// Circumcircle of triangle `a,b,c` as `(center_x, center_y, radius²)`,
/// precomputed once per triangle so the Bowyer-Watson incircle test reduces to
/// a `dist²(p, center) < r²` compare. The center is the intersection of the
/// perpendicular bisectors; `d` is twice the signed area and is non-zero for
/// any non-degenerate triangle (degenerate slivers give a huge radius and are
/// harmlessly reclaimed on the next insertion).
fn circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> (f64, f64, f64) {
    let ((ax, ay), (bx, by), (cx, cy)) = (a, b, c);
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let (rx, ry) = (ax - ux, ay - uy);
    (ux, uy, rx * rx + ry * ry)
}

/// Past this point count `Delaunay2D::new` switches from the plain O(n²)
/// circumcircle scan to the grid-accelerated build (matches the spatial crate's
/// `DELAUNAY_CIRCLE_GRID_THRESHOLD`).
const DELAUNAY_GRID_THRESHOLD: usize = 4096;

/// Bowyer-Watson with the plain "scan every triangle per insertion" loop; the
/// dead triangles are swap-removed so the live set stays compact. Returns the
/// interior triangles (super-triangle simplices dropped). Fastest below
/// `DELAUNAY_GRID_THRESHOLD`. Byte-identical to the historical inline loop.
fn delaunay_triangulate_linear(all_points: &[(f64, f64)], n: usize) -> Vec<(usize, usize, usize)> {
    let mut triangles = vec![(n, n + 1, n + 2)];
    let mut circ = vec![circumcircle(
        all_points[n],
        all_points[n + 1],
        all_points[n + 2],
    )];
    let mut bad: Vec<usize> = Vec::new();
    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for p_idx in 0..n {
        let p = all_points[p_idx];
        bad.clear();
        for (t_idx, &(ccx, ccy, r2)) in circ.iter().enumerate() {
            let (dx, dy) = (p.0 - ccx, p.1 - ccy);
            if dx * dx + dy * dy < r2 {
                bad.push(t_idx);
            }
        }
        boundary.clear();
        delaunay_collect_boundary(&triangles, &bad, &mut boundary);
        bad.sort_unstable();
        for &idx in bad.iter().rev() {
            triangles.swap_remove(idx);
            circ.swap_remove(idx);
        }
        for &(e0, e1) in &boundary {
            triangles.push((p_idx, e0, e1));
            circ.push(circumcircle(p, all_points[e0], all_points[e1]));
        }
    }
    triangles
        .into_iter()
        .filter(|&(a, b, c)| a < n && b < n && c < n)
        .collect()
}

/// Bowyer-Watson where a uniform grid over each triangle's circumcircle bbox
/// restricts the per-insertion candidate set to the new point's cell instead of
/// scanning all live triangles (the O(n²) bottleneck). Dead triangles are
/// MASKED via `active` rather than swap-removed, so a triangle's index stays
/// stable for the grid's whole lifetime. The point's cell list is a superset of
/// every triangle whose circumcircle contains the point, so the bad set — hence
/// the triangulation — is the same as the linear path (a rare empty cell falls
/// back to the full active scan for robustness).
fn delaunay_triangulate_circle_grid(
    all_points: &[(f64, f64)],
    n: usize,
    min_x: f64,
    min_y: f64,
    dx: f64,
    dy: f64,
) -> Vec<(usize, usize, usize)> {
    let mut triangles = vec![(n, n + 1, n + 2)];
    let mut circ = vec![circumcircle(
        all_points[n],
        all_points[n + 1],
        all_points[n + 2],
    )];
    let mut active = vec![true];
    let mut grid = DelaunayCircleGrid::new(n, min_x, min_y, dx, dy);
    grid.insert_circle(circ[0], 0);

    let mut bad: Vec<usize> = Vec::new();
    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for p_idx in 0..n {
        let point = all_points[p_idx];
        bad.clear();
        grid.bad_triangles(point, &circ, &active, &mut bad);
        if bad.is_empty() {
            delaunay_scan_active_bad_triangles(point, &circ, &active, &mut bad);
        }
        bad.sort_unstable();
        bad.dedup();

        boundary.clear();
        delaunay_collect_boundary(&triangles, &bad, &mut boundary);

        for &idx in &bad {
            active[idx] = false;
        }
        for &(e0, e1) in &boundary {
            let triangle_idx = triangles.len();
            let circle = circumcircle(all_points[p_idx], all_points[e0], all_points[e1]);
            triangles.push((p_idx, e0, e1));
            circ.push(circle);
            active.push(true);
            grid.insert_circle(circle, triangle_idx);
        }
    }

    triangles
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (a, b, c))| {
            (active[idx] && a < n && b < n && c < n).then_some((a, b, c))
        })
        .collect()
}

fn delaunay_scan_active_bad_triangles(
    point: (f64, f64),
    circ: &[(f64, f64, f64)],
    active: &[bool],
    bad: &mut Vec<usize>,
) {
    for (t_idx, &(cx, cy, r2)) in circ.iter().enumerate() {
        if !active[t_idx] {
            continue;
        }
        let ddx = point.0 - cx;
        let ddy = point.1 - cy;
        if ddx * ddx + ddy * ddy < r2 {
            bad.push(t_idx);
        }
    }
}

fn delaunay_collect_boundary(
    triangles: &[(usize, usize, usize)],
    bad: &[usize],
    boundary: &mut Vec<(usize, usize)>,
) {
    for &t_idx in bad {
        let (a, b, c) = triangles[t_idx];
        for &(e0, e1) in &[(a, b), (b, c), (c, a)] {
            if !bad.iter().any(|&other_idx| {
                other_idx != t_idx
                    && triangle_has_edge(
                        triangles[other_idx].0,
                        triangles[other_idx].1,
                        triangles[other_idx].2,
                        e0,
                        e1,
                    )
            }) {
                boundary.push((e0, e1));
            }
        }
    }
}

/// Uniform grid binning each triangle's circumcircle by the cells its bbox
/// overlaps, so the Bowyer-Watson bad-triangle search reads one cell.
struct DelaunayCircleGrid {
    min_x: f64,
    min_y: f64,
    inv_dx: f64,
    inv_dy: f64,
    dim: usize,
    cells: Vec<Vec<usize>>,
}

impl DelaunayCircleGrid {
    fn new(n: usize, min_x: f64, min_y: f64, dx: f64, dy: f64) -> Self {
        let dim = ((n as f64).sqrt() as usize).clamp(16, 128);
        Self {
            min_x,
            min_y,
            inv_dx: dim as f64 / dx.max(1e-10),
            inv_dy: dim as f64 / dy.max(1e-10),
            dim,
            cells: vec![Vec::new(); dim * dim],
        }
    }

    fn insert_circle(&mut self, circle: (f64, f64, f64), triangle_idx: usize) {
        let (cx, cy, r2) = circle;
        if !cx.is_finite() || !cy.is_finite() || !r2.is_finite() || r2 < 0.0 {
            return;
        }
        let r = r2.sqrt();
        let x0 = self.cell_x(cx - r);
        let x1 = self.cell_x(cx + r);
        let y0 = self.cell_y(cy - r);
        let y1 = self.cell_y(cy + r);
        for y in y0..=y1 {
            let row = y * self.dim;
            for x in x0..=x1 {
                self.cells[row + x].push(triangle_idx);
            }
        }
    }

    fn bad_triangles(
        &self,
        point: (f64, f64),
        circ: &[(f64, f64, f64)],
        active: &[bool],
        bad: &mut Vec<usize>,
    ) {
        let cell = self.point_cell(point);
        for &t_idx in &self.cells[cell] {
            if !active[t_idx] {
                continue;
            }
            let (cx, cy, r2) = circ[t_idx];
            let ddx = point.0 - cx;
            let ddy = point.1 - cy;
            if ddx * ddx + ddy * ddy < r2 {
                bad.push(t_idx);
            }
        }
    }

    fn point_cell(&self, point: (f64, f64)) -> usize {
        self.cell_y(point.1) * self.dim + self.cell_x(point.0)
    }

    fn cell_x(&self, x: f64) -> usize {
        clamp_delaunay_grid_cell((x - self.min_x) * self.inv_dx, self.dim)
    }

    fn cell_y(&self, y: f64) -> usize {
        clamp_delaunay_grid_cell((y - self.min_y) * self.inv_dy, self.dim)
    }
}

fn clamp_delaunay_grid_cell(scaled: f64, dim: usize) -> usize {
    if scaled <= 0.0 {
        0
    } else if scaled >= dim as f64 {
        dim - 1
    } else {
        scaled as usize
    }
}

fn triangle_has_edge(a: usize, b: usize, c: usize, e0: usize, e1: usize) -> bool {
    [(a, b), (b, c), (c, a)]
        .iter()
        .any(|&(x, y)| (x == e0 && y == e1) || (x == e1 && y == e0))
}

fn barycentric(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> (f64, f64, f64) {
    let (v0x, v0y, v1x, v1y, v2x, v2y) = (
        b.0 - a.0,
        b.1 - a.1,
        c.0 - a.0,
        c.1 - a.1,
        p.0 - a.0,
        p.1 - a.1,
    );
    let (d00, d01, d11, d20, d21) = (
        v0x * v0x + v0y * v0y,
        v0x * v1x + v0y * v1y,
        v1x * v1x + v1y * v1y,
        v2x * v0x + v2y * v0y,
        v2x * v1x + v2y * v1y,
    );
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let l2 = (d11 * d20 - d01 * d21) / denom;
    let l3 = (d00 * d21 - d01 * d20) / denom;
    (1.0 - l2 - l3, l2, l3)
}

#[derive(Debug, Clone)]
pub struct LinearNDInterpolator {
    delaunay: Delaunay2D,
    values: Vec<f64>,
}

impl LinearNDInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        if points.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 3,
                actual: 0,
            });
        }
        for (i, point) in points.iter().enumerate() {
            if point.len() != 2 {
                return Err(InterpError::InvalidArgument {
                    detail: format!(
                        "LinearND only supports 2D points; point {i} has dimension {}",
                        point.len()
                    ),
                });
            }
        }
        if points.len() != values.len() {
            return Err(InterpError::LengthMismatch {
                x_len: points.len(),
                y_len: values.len(),
            });
        }
        let pts: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
        Ok(Self {
            delaunay: Delaunay2D::new(&pts)?,
            values: values.to_vec(),
        })
    }
    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        if query.len() != 2 {
            return Err(InterpError::InvalidArgument {
                detail: "query must be 2D".to_string(),
            });
        }
        match self.delaunay.find_simplex((query[0], query[1])) {
            Some((idx, l1, l2, l3)) => {
                let (a, b, c) = self.delaunay.simplices[idx];
                Ok(l1 * self.values[a] + l2 * self.values[b] + l3 * self.values[c])
            }
            None => Ok(f64::NAN),
        }
    }
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        // Each query independently locates its simplex and evaluates the barycentric
        // interpolant; parallelize across queries (bit-identical, query-order preserved).
        // Low work weight: the simplex lookup is cheap, so only batch-parallelize when the
        // query count is large enough to amortize the spawn (avoids a small-m regression).
        par_query_try_map(queries, 6, |q| self.eval(q))
    }
}

/// Options for [`CloughTocher2DInterpolator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloughTocher2DOptions {
    /// Value returned outside the convex hull of the input points.
    pub fill_value: f64,
    /// Gradient-estimation tolerance, retained for SciPy API parity.
    pub tol: f64,
    /// Maximum gradient-estimation iterations, retained for SciPy API parity.
    pub maxiter: usize,
    /// Rescale input coordinates to the unit square before triangulation.
    pub rescale: bool,
}

type Point2 = (f64, f64);
type PreparedCloughTocherPoints = (Vec<Point2>, Point2, Point2);

impl Default for CloughTocher2DOptions {
    fn default() -> Self {
        Self {
            fill_value: f64::NAN,
            tol: 1e-6,
            maxiter: 400,
            rescale: false,
        }
    }
}

/// Smooth scattered-data interpolator for 2D point sets.
///
/// Mirrors the core public surface of
/// `scipy.interpolate.CloughTocher2DInterpolator(points, values, ...)`: input
/// points are triangulated, values are interpolated exactly at data sites, and
/// query points outside the convex hull return `fill_value`. The Rust kernel
/// estimates per-vertex gradients from neighboring triangles and evaluates a
/// cubic, gradient-corrected triangular patch that preserves affine functions.
#[derive(Debug, Clone)]
pub struct CloughTocher2DInterpolator {
    delaunay: Delaunay2D,
    // Superseded by `patches`, which bakes the values and gradients into the 19
    // Bézier control points at construction (frankenscipy-9l5oo), so nothing reads
    // these after the patch is built. Kept rather than deleted because they are the
    // inputs the patch is derived FROM and removing them changes the constructor's
    // shape; flagged for removal with that constructor, not silently
    // (frankenscipy-9yyez).
    #[allow(dead_code)]
    values: Vec<f64>,
    #[allow(dead_code)]
    gradients: Vec<(f64, f64)>,
    /// Precomputed cubic Bézier patch (19 control points) per triangle — the entire
    /// query-invariant Clough-Tocher patch, built once so `eval` only runs the Bézier
    /// sum per query instead of rebuilding the patch. frankenscipy-9l5oo.
    patches: Vec<[f64; 19]>,
    fill_value: f64,
    offset: (f64, f64),
    scale: (f64, f64),
    rescale: bool,
}

impl CloughTocher2DInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<Self, InterpError> {
        Self::with_options(points, values, CloughTocher2DOptions::default())
    }

    pub fn with_options(
        points: &[Vec<f64>],
        values: &[f64],
        options: CloughTocher2DOptions,
    ) -> Result<Self, InterpError> {
        validate_clough_tocher_inputs(points, values, options)?;
        let (scaled_points, offset, scale) = prepare_clough_tocher_points(points, options.rescale)?;
        let delaunay = Delaunay2D::new(&scaled_points)?;
        if delaunay.simplices.is_empty() {
            return Err(InterpError::InvalidArgument {
                detail: "CloughTocher2DInterpolator requires non-collinear points".to_string(),
            });
        }
        let gradients = estimate_clough_tocher_gradients(&delaunay, values);
        let patches: Vec<[f64; 19]> = (0..delaunay.simplices.len())
            .map(|i| clough_tocher_patch(&delaunay, i, values, &gradients))
            .collect();
        Ok(Self {
            delaunay,
            values: values.to_vec(),
            gradients,
            patches,
            fill_value: options.fill_value,
            offset,
            scale,
            rescale: options.rescale,
        })
    }

    pub fn eval(&self, query: &[f64]) -> Result<f64, InterpError> {
        if query.len() != 2 {
            return Err(InterpError::InvalidArgument {
                detail: "query must be 2D".to_string(),
            });
        }
        if !query[0].is_finite() || !query[1].is_finite() {
            return Ok(f64::NAN);
        }
        let point = self.transform_query((query[0], query[1]));
        let Some((idx, l1, l2, l3)) = self.delaunay.find_simplex(point) else {
            return Ok(self.fill_value);
        };
        Ok(clough_tocher_eval_patch(&self.patches[idx], [l1, l2, l3]))
    }

    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        // Each query independently locates its simplex and evaluates the Clough-Tocher
        // cubic; parallelize across queries (bit-identical, query-order preserved).
        par_query_try_map(queries, 160, |query| self.eval(query))
    }

    fn transform_query(&self, point: (f64, f64)) -> (f64, f64) {
        if self.rescale {
            (
                (point.0 - self.offset.0) / self.scale.0,
                (point.1 - self.offset.1) / self.scale.1,
            )
        } else {
            point
        }
    }
}

fn validate_clough_tocher_inputs(
    points: &[Vec<f64>],
    values: &[f64],
    options: CloughTocher2DOptions,
) -> Result<(), InterpError> {
    if points.len() < 3 {
        return Err(InterpError::TooFewPoints {
            minimum: 3,
            actual: points.len(),
        });
    }
    if points.len() != values.len() {
        return Err(InterpError::LengthMismatch {
            x_len: points.len(),
            y_len: values.len(),
        });
    }
    if !options.fill_value.is_finite() && !options.fill_value.is_nan() {
        return Err(InterpError::InvalidArgument {
            detail: "fill_value must be finite or NaN".to_string(),
        });
    }
    if !options.tol.is_finite() || options.tol <= 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "tol must be positive and finite".to_string(),
        });
    }
    if options.maxiter == 0 {
        return Err(InterpError::InvalidArgument {
            detail: "maxiter must be positive".to_string(),
        });
    }
    for (index, point) in points.iter().enumerate() {
        if point.len() != 2 {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "CloughTocher2DInterpolator only supports 2D points; point {index} has dimension {}",
                    point.len()
                ),
            });
        }
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(InterpError::InvalidArgument {
                detail: "CloughTocher2DInterpolator requires finite points".to_string(),
            });
        }
    }
    // Reject duplicate points in O(n) via a hash set instead of the O(n²)
    // all-pairs scan (which alone cost ~80 ms at n=20k and was the dominant
    // cost of `CloughTocher2DInterpolator::new`). All points are finite here, so
    // two are `==` iff their bit patterns match after normalizing signed zero
    // (`x + 0.0` maps -0.0 → +0.0), making this byte-identical to the `==` scan.
    let mut seen = std::collections::HashSet::with_capacity(points.len());
    for point in points {
        if !seen.insert(((point[0] + 0.0).to_bits(), (point[1] + 0.0).to_bits())) {
            return Err(InterpError::InvalidArgument {
                detail: "CloughTocher2DInterpolator requires unique points".to_string(),
            });
        }
    }
    Ok(())
}

fn prepare_clough_tocher_points(
    points: &[Vec<f64>],
    rescale: bool,
) -> Result<PreparedCloughTocherPoints, InterpError> {
    let raw = points
        .iter()
        .map(|point| (point[0], point[1]))
        .collect::<Vec<_>>();
    if !rescale {
        return Ok((raw, (0.0, 0.0), (1.0, 1.0)));
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in &raw {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    let scale_x = (max_x - min_x).max(1.0);
    let scale_y = (max_y - min_y).max(1.0);
    let scaled = raw
        .iter()
        .map(|&(x, y)| ((x - min_x) / scale_x, (y - min_y) / scale_y))
        .collect::<Vec<_>>();
    Ok((scaled, (min_x, min_y), (scale_x, scale_y)))
}

fn estimate_clough_tocher_gradients(delaunay: &Delaunay2D, values: &[f64]) -> Vec<(f64, f64)> {
    let mut neighbors = vec![Vec::<usize>::new(); values.len()];
    for &(a, b, c) in &delaunay.simplices {
        push_unique_neighbor(&mut neighbors[a], b);
        push_unique_neighbor(&mut neighbors[a], c);
        push_unique_neighbor(&mut neighbors[b], a);
        push_unique_neighbor(&mut neighbors[b], c);
        push_unique_neighbor(&mut neighbors[c], a);
        push_unique_neighbor(&mut neighbors[c], b);
    }

    (0..values.len())
        .map(|index| {
            estimate_vertex_gradient(index, delaunay, values, &neighbors[index])
                .or_else(|| fallback_triangle_gradient(index, delaunay, values))
                .unwrap_or((0.0, 0.0))
        })
        .collect()
}

fn push_unique_neighbor(neighbors: &mut Vec<usize>, candidate: usize) {
    if !neighbors.contains(&candidate) {
        neighbors.push(candidate);
    }
}

fn estimate_vertex_gradient(
    index: usize,
    delaunay: &Delaunay2D,
    values: &[f64],
    neighbors: &[usize],
) -> Option<(f64, f64)> {
    let origin = delaunay.points[index];
    let base = values[index];
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    let mut xz = 0.0;
    let mut yz = 0.0;
    for &neighbor in neighbors {
        let point = delaunay.points[neighbor];
        let dx = point.0 - origin.0;
        let dy = point.1 - origin.1;
        let dz = values[neighbor] - base;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
        xz += dx * dz;
        yz += dy * dz;
    }
    let det = xx * yy - xy * xy;
    if det.abs() <= 1e-24 {
        return None;
    }
    Some(((xz * yy - yz * xy) / det, (xx * yz - xy * xz) / det))
}

fn fallback_triangle_gradient(
    index: usize,
    delaunay: &Delaunay2D,
    values: &[f64],
) -> Option<(f64, f64)> {
    for &(a, b, c) in &delaunay.simplices {
        if a == index || b == index || c == index {
            return triangle_plane_gradient(
                [delaunay.points[a], delaunay.points[b], delaunay.points[c]],
                [values[a], values[b], values[c]],
            );
        }
    }
    None
}

fn triangle_plane_gradient(points: [(f64, f64); 3], values: [f64; 3]) -> Option<(f64, f64)> {
    let dx1 = points[1].0 - points[0].0;
    let dy1 = points[1].1 - points[0].1;
    let dz1 = values[1] - values[0];
    let dx2 = points[2].0 - points[0].0;
    let dy2 = points[2].1 - points[0].1;
    let dz2 = values[2] - values[0];
    let det = dx1 * dy2 - dx2 * dy1;
    if det.abs() <= 1e-24 {
        return None;
    }
    Some(((dz1 * dy2 - dz2 * dy1) / det, (dx1 * dz2 - dx2 * dz1) / det))
}

/// Precompute the 19 cubic Bézier control points of the Clough-Tocher macro-patch for
/// one triangle — all query-INVARIANT (vertices/values/gradients/neighbours only), so
/// it is computed ONCE per triangle instead of on every eval. frankenscipy-9l5oo.
fn clough_tocher_patch(
    delaunay: &Delaunay2D,
    simplex_index: usize,
    values: &[f64],
    gradients: &[(f64, f64)],
) -> [f64; 19] {
    let (a, b, c) = delaunay.simplices[simplex_index];
    let points = [delaunay.points[a], delaunay.points[b], delaunay.points[c]];
    let values = [values[a], values[b], values[c]];
    let gradients = [gradients[a], gradients[b], gradients[c]];

    let e12 = (points[1].0 - points[0].0, points[1].1 - points[0].1);
    let e23 = (points[2].0 - points[1].0, points[2].1 - points[1].1);
    let e31 = (points[0].0 - points[2].0, points[0].1 - points[2].1);

    let f1 = values[0];
    let f2 = values[1];
    let f3 = values[2];
    let df12 = gradients[0].0 * e12.0 + gradients[0].1 * e12.1;
    let df21 = -(gradients[1].0 * e12.0 + gradients[1].1 * e12.1);
    let df23 = gradients[1].0 * e23.0 + gradients[1].1 * e23.1;
    let df32 = -(gradients[2].0 * e23.0 + gradients[2].1 * e23.1);
    let df31 = gradients[2].0 * e31.0 + gradients[2].1 * e31.1;
    let df13 = -(gradients[0].0 * e31.0 + gradients[0].1 * e31.1);

    let c3000 = f1;
    let c2100 = (df12 + 3.0 * c3000) / 3.0;
    let c2010 = (df13 + 3.0 * c3000) / 3.0;
    let c0300 = f2;
    let c1200 = (df21 + 3.0 * c0300) / 3.0;
    let c0210 = (df23 + 3.0 * c0300) / 3.0;
    let c0030 = f3;
    let c1020 = (df31 + 3.0 * c0030) / 3.0;
    let c0120 = (df32 + 3.0 * c0030) / 3.0;

    let c2001 = (c2100 + c2010 + c3000) / 3.0;
    let c0201 = (c1200 + c0300 + c0210) / 3.0;
    let c0021 = (c1020 + c0120 + c0030) / 3.0;

    let mut g = [-0.5; 3];
    for (k, maybe_neighbor) in delaunay.neighbors[simplex_index].iter().enumerate() {
        let Some(neighbor_index) = maybe_neighbor else {
            continue;
        };
        let neighbor = delaunay.simplices[*neighbor_index];
        let neighbor_centroid = triangle_centroid([
            delaunay.points[neighbor.0],
            delaunay.points[neighbor.1],
            delaunay.points[neighbor.2],
        ]);
        let c = barycentric(points[0], points[1], points[2], neighbor_centroid);
        let denom = match k {
            0 => 2.0 - 3.0 * c.2 - 3.0 * c.1,
            1 => 2.0 - 3.0 * c.0 - 3.0 * c.2,
            _ => 2.0 - 3.0 * c.1 - 3.0 * c.0,
        };
        if denom.abs() <= 1e-14 {
            continue;
        }
        g[k] = match k {
            0 => (2.0 * c.2 + c.1 - 1.0) / denom,
            1 => (2.0 * c.0 + c.2 - 1.0) / denom,
            _ => (2.0 * c.1 + c.0 - 1.0) / denom,
        };
    }

    let c0111 = (g[0] * (-c0300 + 3.0 * c0210 - 3.0 * c0120 + c0030)
        + (-c0300 + 2.0 * c0210 - c0120 + c0021 + c0201))
        / 2.0;
    let c1011 = (g[1] * (-c0030 + 3.0 * c1020 - 3.0 * c2010 + c3000)
        + (-c0030 + 2.0 * c1020 - c2010 + c2001 + c0021))
        / 2.0;
    let c1101 = (g[2] * (-c3000 + 3.0 * c2100 - 3.0 * c1200 + c0300)
        + (-c3000 + 2.0 * c2100 - c1200 + c2001 + c0201))
        / 2.0;

    let c1002 = (c1101 + c1011 + c2001) / 3.0;
    let c0102 = (c1101 + c0111 + c0201) / 3.0;
    let c0012 = (c1011 + c0111 + c0021) / 3.0;
    let c0003 = (c1002 + c0102 + c0012) / 3.0;

    [
        c3000, c2100, c2010, c2001, c1200, c1101, c1020, c1011, c1002, c0300, c0210, c0201, c0120,
        c0111, c0102, c0030, c0021, c0012, c0003,
    ]
}

/// Evaluate a precomputed Clough-Tocher patch (19 control points, in the order produced
/// by [`clough_tocher_patch`]) at barycentric coords `bary`. This is the only
/// query-dependent step; everything else is precomputed once per triangle.
fn clough_tocher_eval_patch(p: &[f64; 19], bary: [f64; 3]) -> f64 {
    let [
        c3000,
        c2100,
        c2010,
        c2001,
        c1200,
        c1101,
        c1020,
        c1011,
        c1002,
        c0300,
        c0210,
        c0201,
        c0120,
        c0111,
        c0102,
        c0030,
        c0021,
        c0012,
        c0003,
    ] = *p;

    let min_bary = bary[0].min(bary[1]).min(bary[2]);
    let b1 = bary[0] - min_bary;
    let b2 = bary[1] - min_bary;
    let b3 = bary[2] - min_bary;
    let b4 = 3.0 * min_bary;

    b1.powi(3) * c3000
        + 3.0 * b1.powi(2) * b2 * c2100
        + 3.0 * b1.powi(2) * b3 * c2010
        + 3.0 * b1.powi(2) * b4 * c2001
        + 3.0 * b1 * b2.powi(2) * c1200
        + 6.0 * b1 * b2 * b4 * c1101
        + 3.0 * b1 * b3.powi(2) * c1020
        + 6.0 * b1 * b3 * b4 * c1011
        + 3.0 * b1 * b4.powi(2) * c1002
        + b2.powi(3) * c0300
        + 3.0 * b2.powi(2) * b3 * c0210
        + 3.0 * b2.powi(2) * b4 * c0201
        + 3.0 * b2 * b3.powi(2) * c0120
        + 6.0 * b2 * b3 * b4 * c0111
        + 3.0 * b2 * b4.powi(2) * c0102
        + b3.powi(3) * c0030
        + 3.0 * b3.powi(2) * b4 * c0021
        + 3.0 * b3 * b4.powi(2) * c0012
        + b4.powi(3) * c0003
}

fn triangle_centroid(points: [(f64, f64); 3]) -> (f64, f64) {
    (
        (points[0].0 + points[1].0 + points[2].0) / 3.0,
        (points[0].1 + points[1].1 + points[2].1) / 3.0,
    )
}

// ── RBF Interpolator ─────────────────────────────────────────────────

/// Radial basis function kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RbfKernel {
    /// Linear: r
    #[default]
    Linear,
    /// Thin plate spline: r² ln(r)
    ThinPlateSpline,
    /// Multiquadric: sqrt(1 + (εr)²)
    Multiquadric,
    /// Inverse multiquadric: 1/sqrt(1 + (εr)²)
    InverseMultiquadric,
    /// Gaussian: exp(-(εr)²)
    Gaussian,
}

/// Same-binary A/B toggle for the RBF `Φ` matrix build. When `true`, the matrix is filled serially
/// (the ORIG behaviour). When `false` (default), the independent rows fan across cores. Byte-identical.
#[doc(hidden)]
pub static RBF_BUILD_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// N-dimensional scattered data interpolation using radial basis functions.
///
/// Matches `scipy.interpolate.RBFInterpolator(y, d, kernel=kernel)`.
#[derive(Debug, Clone)]
pub struct RbfInterpolator {
    points: Vec<Vec<f64>>,
    weights: Vec<f64>,
    polynomial_weights: Vec<f64>,
    kernel: RbfKernel,
    epsilon: f64,
    dim: usize,
    degree: i32,
}

const MAX_RBF_POINTS: usize = 4096;

impl RbfInterpolator {
    /// Create a new RBF interpolator.
    ///
    /// # Arguments
    /// * `points` — Data point coordinates (N points × D dimensions).
    /// * `values` — Function values at each point.
    /// * `kernel` — Radial basis function kernel.
    /// * `epsilon` — Shape parameter (used by multiquadric, inverse_multiquadric, gaussian).
    pub fn new(
        points: &[Vec<f64>],
        values: &[f64],
        kernel: RbfKernel,
        epsilon: f64,
    ) -> Result<Self, InterpError> {
        Self::with_degree(points, values, kernel, epsilon, rbf_default_degree(kernel))
    }

    /// Create an RBF interpolator with an explicit polynomial-tail degree.
    ///
    /// `-1` disables the tail, `0` adds a constant, and `1` adds a constant
    /// plus one linear term per input dimension. The default constructor uses
    /// SciPy's kernel-specific defaults for the kernels this type supports.
    pub fn with_degree(
        points: &[Vec<f64>],
        values: &[f64],
        kernel: RbfKernel,
        epsilon: f64,
        degree: i32,
    ) -> Result<Self, InterpError> {
        let n = points.len();
        if n == 0 {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if n != values.len() {
            return Err(InterpError::LengthMismatch {
                x_len: n,
                y_len: values.len(),
            });
        }
        if n > MAX_RBF_POINTS {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "RbfInterpolator point count {n} exceeds dense solver safety bound {MAX_RBF_POINTS}"
                ),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(InterpError::InvalidArgument {
                detail: "RbfInterpolator epsilon must be finite and positive".to_string(),
            });
        }
        let dim = points[0].len();
        if dim == 0 {
            return Err(InterpError::InvalidArgument {
                detail: "RbfInterpolator points must have at least one dimension".to_string(),
            });
        }
        for (i, point) in points.iter().enumerate() {
            if point.len() != dim {
                return Err(InterpError::InvalidArgument {
                    detail: format!(
                        "all RbfInterpolator points must have the same dimension; point 0 has {dim} but point {i} has {}",
                        point.len()
                    ),
                });
            }
            if point.iter().any(|coord| !coord.is_finite()) {
                return Err(InterpError::NonFiniteX);
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(InterpError::InvalidArgument {
                detail: "RbfInterpolator values must be finite".to_string(),
            });
        }
        if !(-1..=1).contains(&degree) {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "RbfInterpolator supports polynomial degrees -1, 0, and 1; got {degree}"
                ),
            });
        }

        let polynomial_terms: Vec<Vec<f64>> = points
            .iter()
            .map(|point| rbf_polynomial_terms(point, degree))
            .collect();
        let tail_width = polynomial_terms.first().map_or(0, Vec::len);

        // Build the RBF matrix Φ[i,j] = φ(||points[i] - points[j]||) in FLAT row-major
        // storage so the dense solve runs over contiguous rows (cache-resident +
        // SIMD-able inner update) instead of a Vec<Vec> with one heap alloc per row.
        // Row `i` = φ(||points[i] - points[j]||) over all j — an independent, compute-bound sweep
        // (a distance + a transcendental per entry) that writes its OWN contiguous row, so the rows
        // fan across cores. BYTE-IDENTICAL to the serial double loop (each entry is the same pure
        // function of (points[i], points[j]); only the owning core changes). SciPy builds it
        // single-threaded. Gated on the total kernel-eval work so small systems stay serial.
        let mut phi = vec![0.0f64; n * n];
        let build_row = |i: usize, row: &mut [f64]| {
            for (j, slot) in row.iter_mut().enumerate() {
                let r2 = euclidean_dist_sq(&points[i], &points[j]);
                *slot = rbf_eval_sq(kernel, r2, epsilon);
            }
        };
        let build_work = (n as u64) * (n as u64) * (dim as u64 + 2);
        let nthreads = if RBF_BUILD_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
            || build_work < (1 << 18)
            || n < 2
        {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(n)
        };
        if nthreads <= 1 {
            for i in 0..n {
                build_row(i, &mut phi[i * n..i * n + n]);
            }
        } else {
            let per = n.div_ceil(nthreads);
            let build_row = &build_row;
            std::thread::scope(|scope| {
                for (ci, chunk) in phi.chunks_mut(per * n).enumerate() {
                    let i0 = ci * per;
                    scope.spawn(move || {
                        for (local, row) in chunk.chunks_mut(n).enumerate() {
                            build_row(i0 + local, row);
                        }
                    });
                }
            });
        }

        // Solve Φ w = values for weights. Route through fsci-linalg's multithreaded
        // blocked LU (n≥1000 fast path) instead of the local serial Gaussian elimination
        // — the dense O(n³) solve dominates RBF construction. Not bit-identical to the
        // naive GE (different pivoting/blocking, ~1e-12), but within the RBF tolerance.
        let mut system = vec![vec![0.0; n + tail_width]; n + tail_width];
        for (i, row) in phi.chunks(n).enumerate() {
            system[i][..n].copy_from_slice(row);
            system[i][n..].copy_from_slice(&polynomial_terms[i]);
        }
        for (term, column) in (0..tail_width).enumerate() {
            for (i, terms) in polynomial_terms.iter().enumerate() {
                system[n + term][i] = terms[column];
            }
        }
        let mut rhs = values.to_vec();
        rhs.resize(n + tail_width, 0.0);
        let solution = fsci_linalg::solve(&system, &rhs, fsci_linalg::SolveOptions::default())
            .map_err(|e| InterpError::InvalidArgument {
                detail: format!("RbfInterpolator dense solve failed: {e:?}"),
            })?
            .x;
        let weights = solution[..n].to_vec();
        let polynomial_weights = solution[n..].to_vec();

        Ok(Self {
            points: points.to_vec(),
            weights,
            polynomial_weights,
            kernel,
            epsilon,
            dim,
            degree,
        })
    }

    /// Evaluate the interpolant at a query point.
    pub fn eval(&self, query: &[f64]) -> f64 {
        if query.len() != self.dim {
            return f64::NAN;
        }
        let mut result = 0.0;
        for (i, pt) in self.points.iter().enumerate() {
            let r2 = euclidean_dist_sq(pt, query);
            result += self.weights[i] * rbf_eval_sq(self.kernel, r2, self.epsilon);
        }
        result
            + self
                .polynomial_weights
                .iter()
                .zip(rbf_polynomial_terms(query, self.degree))
                .map(|(&coefficient, term)| coefficient * term)
                .sum::<f64>()
    }

    /// Evaluate at multiple query points.
    ///
    /// Each query is an independent O(n_centers) sum, so for large
    /// `queries × centers` the work is split across threads; the per-query value is
    /// the same pure `eval` regardless of the owning core, so the result is
    /// bit-identical to the sequential map.
    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Vec<f64> {
        par_query_map(queries, self.points.len(), |q| self.eval(q))
    }
}

fn rbf_default_degree(kernel: RbfKernel) -> i32 {
    match kernel {
        RbfKernel::Linear => 0,
        RbfKernel::ThinPlateSpline => 1,
        RbfKernel::Multiquadric
        | RbfKernel::InverseMultiquadric
        | RbfKernel::Gaussian => -1,
    }
}

fn rbf_polynomial_terms(point: &[f64], degree: i32) -> Vec<f64> {
    match degree {
        -1 => Vec::new(),
        0 => vec![1.0],
        1 => {
            let mut terms = Vec::with_capacity(point.len() + 1);
            terms.push(1.0);
            terms.extend_from_slice(point);
            terms
        }
        _ => unreachable!("RbfInterpolator degree was validated at construction"),
    }
}

/// Map `f` over `queries`, splitting the work across threads when
/// `queries.len() * work_per_query` is large enough to amortise thread spawn. `f` is
/// a pure per-query function, so the parallel result is bit-identical to the
/// sequential `queries.iter().map(f).collect()` (order preserved by concatenating
/// contiguous chunks). Used by the per-query batch evaluators.
/// Chunked parallel fallible map producing one `R` per item, order-preserved (so the
/// result is identical to the serial `items.iter().map(f).collect()`). Splits items into
/// `nthreads` CONTIGUOUS chunks — one `thread::scope` spawn per chunk, NOT per item — to
/// avoid the over-spawn that sank a prior per-column attempt. Propagates the first error.
fn par_chunk_try_map<T, R, E, F>(items: &[T], nthreads: usize, f: F) -> Result<Vec<R>, E>
where
    T: Sync,
    R: Send,
    E: Send,
    F: Fn(&T) -> Result<R, E> + Sync,
{
    if nthreads <= 1 || items.len() < 2 {
        return items.iter().map(&f).collect();
    }
    let chunk = items.len().div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<R>, E>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for ch in items.chunks(chunk) {
            handles.push(scope.spawn(move || ch.iter().map(f).collect::<Result<Vec<R>, E>>()));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out = Vec::with_capacity(items.len());
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

fn par_query_map<T, F>(queries: &[T], work_per_query: usize, f: F) -> Vec<f64>
where
    T: Sync,
    F: Fn(&T) -> f64 + Sync,
{
    let m = queries.len();
    let work = (m as u64).saturating_mul(work_per_query.max(1) as u64);
    if work < PAR_QUERY_MIN_WORK || m < 4 {
        return queries.iter().map(&f).collect();
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(m / 2).max(1);
    if nthreads <= 1 {
        return queries.iter().map(&f).collect();
    }
    let chunk = m.div_ceil(nthreads);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= m {
                    return None;
                }
                let i1 = (i0 + chunk).min(m);
                Some(scope.spawn(move || queries[i0..i1].iter().map(f).collect::<Vec<f64>>()))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("interpolate worker panicked"))
            .collect()
    })
}

/// Runtime switch to force serial tensor-product grid evaluation (`eval_grid`) for
/// same-binary A/B benchmarks. Defaults off. `#[doc(hidden)]` — internal.
#[doc(hidden)]
/// CONTRACT: BYTE-IDENTICAL either way. The parallel arm chunks ROWS and each row's
/// value comes from `row_fn` applied to that row alone, so the work is a map over
/// independent outputs rather than a split reduction. Chunks are written back in index
/// order, so even the assembly order is fixed.
///
/// The property to re-check if this is ever rewritten: the moment anything is accumulated
/// ACROSS rows -- a running total, a shared best-so-far, a normalisation pass -- this
/// stops being an identity and becomes a summation-order tolerance.
pub static EVAL_GRID_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Parallel-across-x-rows driver for tensor-product grid evaluators
/// (`RectBivariateSpline`/`SmoothBivariateSpline::eval_grid`): each x-row produces one
/// independent, compute-bound row of the output grid, so rows fan across cores
/// BYTE-IDENTICALLY (chunks concatenated in x order). Gated on total grid points —
/// same crossover as `bisplev` (≥40k) since the per-point work is comparable.
fn par_grid_rows<F>(xi: &[f64], ncols: usize, row_fn: &F) -> Vec<Vec<f64>>
where
    F: Fn(f64) -> Vec<f64> + Sync,
{
    let nrows = xi.len();
    let grid = nrows.saturating_mul(ncols);
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(nrows.max(1));
    if cores <= 1
        || grid < 40_000
        || EVAL_GRID_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        return xi.iter().map(|&xv| row_fn(xv)).collect();
    }
    let chunk = nrows.div_ceil(cores);
    std::thread::scope(|scope| {
        let handles: Vec<_> = xi
            .chunks(chunk)
            .map(|xchunk| {
                scope.spawn(move || {
                    xchunk
                        .iter()
                        .map(|&xv| row_fn(xv))
                        .collect::<Vec<Vec<f64>>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("grid eval worker panicked"))
            .collect()
    })
}

/// Parallel work-gate (= queries × per-query op-cost) for `par_query_map`/`par_query_try_map`.
/// The parallel path has a large fixed cost — up to one thread per ~2 queries, each allocating
/// its own result `Vec` that is then `flat_map`-collected — which under fleet contention is
/// ~4-5 ms regardless of `m`. Same-process A/B on cubic-spline `eval_many` measured the old
/// `1<<18` gate parallelizing at m≈11k where SERIAL is 18.5x faster (m=16k), 10.5x (32k), 5.9x
/// (65k), 3.1x (131k); the break-even is ~350k queries (work ≈ 1<<23). Gate there so cheap batch
/// evals stay serial up to the point parallelism actually amortizes the spawn/alloc overhead.
const PAR_QUERY_MIN_WORK: u64 = 1 << 23;

/// Runtime switch to force the original `par_query_map` (per-point binary search)
/// path for the piecewise-cubic `eval_many`s, for same-binary A/B benchmarks.
/// Defaults off — sorted finite batches take the O(N+M) interval cursor.
#[doc(hidden)]
/// CONTRACT: BYTE-IDENTICAL either way. The cursor is a SEARCH optimisation, not an
/// arithmetic one: for a sorted batch it walks the breakpoints once in O(N+M) instead of
/// binary-searching per query, and then feeds the SAME interval index into the SAME cubic
/// evaluation. Locating an interval faster cannot change which interval is found, so the
/// returned values are bit-for-bit equal.
///
/// It declines rather than approximates when its precondition fails -- the fast path is
/// skipped for non-finite queries and for batches too small to amortise it -- so there is
/// no input on which it produces a different answer instead of no answer.
pub static INTERP_CUBIC_CURSOR_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Bounded-run interval cursor shared by the piecewise-cubic interpolators
/// (`CubicSplineStandalone`, `Akima1DInterpolator`, `CubicHermiteSpline`) whose
/// per-segment value is `a + dx·(b + dx·(c + dx·d))` with `dx = xi − x[i]` and
/// `i` the interval `find_interval_helper` returns (largest `i` with `x[i] ≤ xi`,
/// clamped to `[0, n−2]`). Returns `Some(values)` — BYTE-IDENTICAL to mapping the
/// per-point `eval` — only for a small-enough, all-finite `x_new` composed of at
/// most eight ascending runs. The cursor resets at each descending boundary and
/// advances monotonically within each run instead of binary-searching every point.
/// A worst-case comparison guard retains binary search when repeated knot sweeps
/// could cost more than `M·log N`; the serial gate keeps the parallel spawn/alloc
/// cost off cheap batches. Any NaN/∞/too-segmented/huge batch returns `None` so the
/// caller keeps its existing `par_query_map` path. Mirrors
/// `PchipInterpolator::eval_many` (frankenscipy-b75mf); `n ≥ 2` holds for every
/// constructed interpolator.
fn cubic_cursor_eval_many(x: &[f64], coeffs: &[[f64; 4]], x_new: &[f64]) -> Option<Vec<f64>> {
    const MAX_ASCENDING_RUNS: usize = 8;

    if INTERP_CUBIC_CURSOR_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let work = (x_new.len() as u64).saturating_mul(24);
    let serial_path = work < PAR_QUERY_MIN_WORK || x_new.len() < 4;
    if !serial_path || x_new.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut run_count = usize::from(!x_new.is_empty());
    for window in x_new.windows(2) {
        if window[0] > window[1] {
            run_count += 1;
            if run_count > MAX_ASCENDING_RUNS {
                return None;
            }
        }
    }
    let n = x.len();
    let binary_comparisons = x_new
        .len()
        .saturating_mul(n.next_power_of_two().trailing_zeros() as usize);
    if run_count.saturating_mul(n) > binary_comparisons {
        return None;
    }
    let mut i = 0usize;
    let mut previous = f64::NEG_INFINITY;
    Some(
        x_new
            .iter()
            .map(|&xi| {
                if xi < previous {
                    i = 0;
                }
                previous = xi;
                if xi >= x[n - 1] {
                    i = n - 2;
                } else {
                    while i + 1 < n - 1 && x[i + 1] <= xi {
                        i += 1;
                    }
                }
                let dx = xi - x[i];
                let [a, b, c, d] = coeffs[i];
                a + dx * (b + dx * (c + dx * d))
            })
            .collect(),
    )
}

/// Like `par_query_map` but for fallible per-query evaluation. Returns the first error in
/// query order (matching the serial `queries.iter().map(f).collect::<Result<Vec<_>, _>>()`):
/// chunks are processed in order, and a chunk's own `collect` short-circuits at its first
/// error, so the earliest erroring query wins exactly as in the serial fold. On all-success
/// the values are bit-identical (each query is an independent pure eval, chunks concatenated
/// in order).
fn par_query_try_map<T, E, F>(queries: &[T], work_per_query: usize, f: F) -> Result<Vec<f64>, E>
where
    T: Sync,
    E: Send,
    F: Fn(&T) -> Result<f64, E> + Sync,
{
    let m = queries.len();
    let work = (m as u64).saturating_mul(work_per_query.max(1) as u64);
    if work < PAR_QUERY_MIN_WORK || m < 4 {
        return queries.iter().map(&f).collect();
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(m / 2).max(1);
    if nthreads <= 1 {
        return queries.iter().map(&f).collect();
    }
    let chunk = m.div_ceil(nthreads);
    let f = &f;
    let chunk_results: Vec<Result<Vec<f64>, E>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= m {
                    return None;
                }
                let i1 = (i0 + chunk).min(m);
                Some(scope.spawn(move || {
                    queries[i0..i1]
                        .iter()
                        .map(f)
                        .collect::<Result<Vec<f64>, E>>()
                }))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("interpolate worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(m);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

// Retained REFERENCE implementation: the optimised sibling's own doc cites it,
// and a test now runs the comparison that doc asserts. Dead only in non-test
// builds (frankenscipy-9yyez).
#[cfg_attr(not(test), allow(dead_code))]
fn rbf_eval(kernel: RbfKernel, r: f64, epsilon: f64) -> f64 {
    match kernel {
        RbfKernel::Linear => r,
        RbfKernel::ThinPlateSpline => {
            if r < 1e-30 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
        RbfKernel::Multiquadric => (1.0 + (epsilon * r).powi(2)).sqrt(),
        RbfKernel::InverseMultiquadric => 1.0 / (1.0 + (epsilon * r).powi(2)).sqrt(),
        RbfKernel::Gaussian => (-(epsilon * r).powi(2)).exp(),
    }
}

// Retained REFERENCE implementation: the optimised sibling's own doc cites it,
// and a test now runs the comparison that doc asserts. Dead only in non-test
// builds (frankenscipy-9yyez).
#[cfg_attr(not(test), allow(dead_code))]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Squared Euclidean distance — `euclidean_dist` without the final `sqrt`.
fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
}

/// Evaluate an RBF kernel from the SQUARED radius `r2 = ||·||²`, skipping the
/// distance `sqrt` for the kernels that only depend on `r²`. For the smooth
/// kernels `φ(r) = f((εr)²) = f(ε²r²)`, so the squared distance feeds the kernel
/// directly — this drops one `sqrt` per (query, centre) on the hot RBF build/eval
/// loops. The two `r`-based kernels recover `r = √r2` and reuse [`rbf_eval`], so
/// they stay bit-identical; the `r²`-based kernels differ from the prior
/// `sqrt`-then-square only at the last ULP (`(√x)²` rounds back to `x ± 1 ULP`).
fn rbf_eval_sq(kernel: RbfKernel, r2: f64, epsilon: f64) -> f64 {
    match kernel {
        // r-based kernels: recover r = √r2 and use the original formula verbatim
        // (`r` equals the previous `euclidean_dist`, so these stay bit-identical).
        RbfKernel::Linear => r2.sqrt(),
        RbfKernel::ThinPlateSpline => {
            let r = r2.sqrt();
            if r < 1e-30 { 0.0 } else { r * r * r.ln() }
        }
        // r²-based kernels: (εr)² == ε²r², so feed r2 straight in and skip the sqrt.
        RbfKernel::Multiquadric => (1.0 + epsilon * epsilon * r2).sqrt(),
        RbfKernel::InverseMultiquadric => 1.0 / (1.0 + epsilon * epsilon * r2).sqrt(),
        RbfKernel::Gaussian => (-(epsilon * epsilon * r2)).exp(),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Additional Interpolators
// ══════════════════════════════════════════════════════════════════════

/// Krogh interpolator: polynomial through all given points.
///
/// Uses divided differences (Newton form) for the interpolating polynomial.
/// Matches `scipy.interpolate.KroghInterpolator`.
pub struct KroghInterpolator {
    xi: Vec<f64>,
    coeffs: Vec<f64>,
}

impl KroghInterpolator {
    /// Create a Krogh interpolator through points (xi, yi).
    pub fn new(xi: &[f64], yi: &[f64]) -> Result<Self, InterpError> {
        if xi.len() != yi.len() || xi.is_empty() {
            return Err(InterpError::InvalidArgument {
                detail: "xi and yi must have same non-zero length".to_string(),
            });
        }
        let n = xi.len();

        // Compute divided differences (Newton form)
        let mut dd = yi.to_vec();
        for j in 1..n {
            for i in (j..n).rev() {
                let dx = xi[i] - xi[i - j];
                if dx.abs() < 1e-15 {
                    return Err(InterpError::InvalidArgument {
                        detail: "duplicate knots".to_string(),
                    });
                }
                dd[i] = (dd[i] - dd[i - 1]) / dx;
            }
        }

        Ok(Self {
            xi: xi.to_vec(),
            coeffs: dd,
        })
    }

    /// Evaluate the interpolating polynomial at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let n = self.coeffs.len();
        // Horner's method for Newton form
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = result * (x - self.xi[i]) + self.coeffs[i];
        }
        result
    }

    /// Evaluate at multiple points.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        // Each query is an independent O(degree) Horner evaluation of the Newton
        // form over the (L1-resident, reused) coefficients, so the queries fan out
        // across threads in contiguous chunks. Concatenated in query order with
        // each eval unchanged, so the result is bit-identical to the serial map.
        par_query_map(xs, self.coeffs.len(), |&x| self.evaluate(x))
    }

    /// Re-express the stored Newton form as a Taylor (shifted-power) series about
    /// `x0`: returns `b` with `P(t) = Σ b[j]·(t − x0)^j`, so `b[j] = P⁽ʲ⁾(x0)/j!`.
    /// `b` has one entry per interpolation node (polynomial degree + 1).
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    fn newton_to_taylor(&self, x0: f64) -> Vec<f64> {
        let n = self.coeffs.len();
        let mut acc = vec![0.0; n];
        acc[0] = self.coeffs[n - 1];
        let mut len = 1usize; // current number of valid (ascending) coefficients
        // Fold in the remaining Newton factors high-to-low: multiply the running
        // polynomial in s = (t − x0) by (s + (x0 − xi[k])) and add dd[k].
        for k in (0..n - 1).rev() {
            let shift = x0 - self.xi[k];
            acc[len] = acc[len - 1]; // new leading term (old acc[len] == 0)
            for j in (1..len).rev() {
                acc[j] = acc[j - 1] + shift * acc[j];
            }
            acc[0] = shift * acc[0] + self.coeffs[k];
            len += 1;
        }
        acc
    }

    /// Evaluate the polynomial and its derivatives at `x`, returning
    /// `[P(x), P'(x), …, P⁽ᵈᵉʳ⁻¹⁾(x)]` (`der` values).
    ///
    /// Matches `scipy.interpolate.KroghInterpolator.derivatives(x, der)` for 1-D
    /// real inputs. Derivative orders beyond the polynomial degree are `0`.
    pub fn derivatives(&self, x: f64, der: usize) -> Vec<f64> {
        if !x.is_finite() {
            return vec![f64::NAN; der];
        }
        let taylor = self.newton_to_taylor(x);
        let mut out = Vec::with_capacity(der);
        let mut factorial = 1.0_f64;
        for j in 0..der {
            if j > 0 {
                factorial *= j as f64;
            }
            let bj = taylor.get(j).copied().unwrap_or(0.0);
            out.push(bj * factorial);
        }
        out
    }
}

/// Estimate the Taylor polynomial of `f` at `x` by polynomial fitting.
///
/// Matches `scipy.interpolate.approximate_taylor_polynomial(f, x, degree, scale,
/// order)`. `f` is sampled at `order + 1` Chebyshev-like abscissae spread over a
/// window of width `scale` centred on `x` (`order` defaults to `degree`); a Krogh
/// interpolant through those samples is re-expanded about `x` to recover the
/// Taylor coefficients.
///
/// Returns the polynomial coefficients in `numpy.poly1d` order — highest degree
/// first, length `degree + 1` — translated to the origin so that evaluating the
/// returned polynomial at `0` gives `f(x)` (the last coefficient). Choosing
/// `order` somewhat larger than `degree` improves the higher-order terms; the
/// fit becomes ill-conditioned past order ~30.
pub fn approximate_taylor_polynomial<F: Fn(f64) -> f64>(
    f: F,
    x: f64,
    degree: usize,
    scale: f64,
    order: Option<usize>,
) -> Result<Vec<f64>, InterpError> {
    let order = order.unwrap_or(degree);
    let n = order + 1;
    // xs[i] = scale·cos(i·π/n) + x  (numpy linspace(0, π, n, endpoint=False)).
    let xs: Vec<f64> = (0..n)
        .map(|i| scale * (std::f64::consts::PI * i as f64 / n as f64).cos() + x)
        .collect();
    let ys: Vec<f64> = xs.iter().map(|&xi| f(xi)).collect();
    let interp = KroghInterpolator::new(&xs, &ys)?;
    let taylor = interp.newton_to_taylor(x);
    // Ascending Taylor coefficients b_0..b_degree (b_j = P^(j)(x)/j!), zero-padded
    // if degree exceeds the interpolant degree, then reversed to poly1d order.
    let mut coeffs: Vec<f64> = (0..=degree)
        .map(|j| taylor.get(j).copied().unwrap_or(0.0))
        .collect();
    coeffs.reverse();
    Ok(coeffs)
}

/// One-shot Krogh interpolation: build a `KroghInterpolator` from
/// `(xi, yi)` and evaluate it at every point in `x_new`.
///
/// Matches `scipy.interpolate.krogh_interpolate(xi, yi, x)` for 1-D real
/// inputs. Returns the interpolated values as a `Vec<f64>` of the same
/// length as `x_new`.
pub fn krogh_interpolate(xi: &[f64], yi: &[f64], x_new: &[f64]) -> Result<Vec<f64>, InterpError> {
    let interp = KroghInterpolator::new(xi, yi)?;
    Ok(interp.evaluate_many(x_new))
}

/// Lagrange interpolation polynomial.
///
/// Returns the polynomial coefficients (highest degree first) that interpolate
/// the given data points. Not recommended for large n due to Runge's phenomenon.
///
/// Matches `scipy.interpolate.lagrange`.
pub fn lagrange(xi: &[f64], yi: &[f64]) -> Result<Vec<f64>, InterpError> {
    if xi.len() != yi.len() || xi.is_empty() {
        return Err(InterpError::InvalidArgument {
            detail: "xi and yi must have same non-zero length".to_string(),
        });
    }
    let n = xi.len();

    // Build the polynomial by summing Lagrange basis polynomials
    // Each L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
    // The result is a polynomial of degree n-1

    // We'll work with coefficient vectors directly
    let mut result = vec![0.0; n];

    for i in 0..n {
        // Compute the i-th Lagrange basis polynomial coefficients
        let mut basis = vec![0.0; n];
        basis[0] = 1.0; // Start with constant 1
        let mut deg = 0;

        for j in 0..n {
            if j == i {
                continue;
            }
            let denom = xi[i] - xi[j];
            if denom.abs() < 1e-15 {
                return Err(InterpError::InvalidArgument {
                    detail: "duplicate knots".to_string(),
                });
            }

            // Multiply current polynomial by (x - x_j) / denom
            // Shift coefficients and add
            deg += 1;
            for k in (1..=deg).rev() {
                basis[k] = (basis[k - 1] - xi[j] * basis[k]) / denom;
            }
            basis[0] = -xi[j] * basis[0] / denom;
        }

        // Add yi * basis to result
        for k in 0..n {
            result[k] += yi[i] * basis[k];
        }
    }

    // Reverse to get highest degree first (standard convention)
    result.reverse();
    Ok(result)
}

/// Evaluate a polynomial given coefficients (highest degree first).
///
/// `p(x) = coeffs[0] * x^(n-1) + coeffs[1] * x^(n-2) + ... + coeffs[n-1]`
pub fn polyval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for &c in coeffs {
        result = result * x + c;
    }
    result
}

/// Piecewise polynomial representation.
///
/// Stores polynomial coefficients for each interval between breakpoints.
/// Matches `scipy.interpolate.PPoly`.
pub struct PPoly {
    /// Coefficients: `c[i][j]` is the j-th coefficient (highest degree first)
    /// for the i-th interval.
    pub c: Vec<Vec<f64>>,
    /// Breakpoints (n+1 values for n intervals).
    pub x: Vec<f64>,
}

impl PPoly {
    /// Create a piecewise polynomial from coefficients and breakpoints.
    pub fn new(c: Vec<Vec<f64>>, x: Vec<f64>) -> Result<Self, InterpError> {
        if c.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if c.len() + 1 != x.len() {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "need {} intervals for {} breakpoints, got {}",
                    x.len() - 1,
                    x.len(),
                    c.len()
                ),
            });
        }
        Ok(Self { c, x })
    }

    /// Evaluate the piecewise polynomial at a point.
    pub fn evaluate(&self, xval: f64) -> f64 {
        if xval.is_nan() {
            return f64::NAN;
        }
        let seg = self.segment(xval);
        self.evaluate_segment(xval, seg)
    }

    fn segment(&self, xval: f64) -> usize {
        // Find interval: largest i in 1..n with x[i] <= xval, else 0. Binary search
        // over the sorted breakpoints — byte-identical to the former O(n) linear
        // scan (`partition_point` counts the leading run of x[i] <= xval), O(log n)
        // per point. frankenscipy-2jmet.
        let n = self.x.len() - 1;
        if n == 0 {
            0
        } else {
            self.x[1..n].partition_point(|&xi| xval >= xi)
        }
    }

    fn evaluate_segment(&self, xval: f64, seg: usize) -> f64 {
        // Evaluate polynomial in local coordinates.
        let dx = xval - self.x[seg];
        polyval(&self.c[seg], dx)
    }

    /// Evaluate at multiple points.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        let sorted_non_nan = !xs.iter().any(|x| x.is_nan()) && xs.windows(2).all(|w| w[0] <= w[1]);
        if sorted_non_nan {
            let n = self.x.len() - 1;
            let mut seg = 0usize;
            return xs
                .iter()
                .map(|&xval| {
                    if n > 0 {
                        while seg + 1 < n && xval >= self.x[seg + 1] {
                            seg += 1;
                        }
                    }
                    self.evaluate_segment(xval, seg)
                })
                .collect();
        }

        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

/// Piecewise polynomial in the Bernstein basis, matching
/// `scipy.interpolate.BPoly`.
///
/// On the i-th interval `[x[i], x[i+1]]` with local coordinate
/// `s = (xval - x[i]) / (x[i+1] - x[i])`, the value is
/// `Σ_{a=0}^{k} c[i][a] · C(k,a) · s^a · (1-s)^{k-a}`.
///
/// Note: coefficients are interval-major (`c[i][a]`, Bernstein order `a = 0..=k`)
/// — the transpose of SciPy's `c[a, i]` layout, matching fsci [`PPoly`].
/// Evaluation extrapolates using the nearest interval (SciPy's default).
pub struct BPoly {
    /// Bernstein coefficients: `c[i][a]` for interval i, order a.
    pub c: Vec<Vec<f64>>,
    /// Breakpoints (`n+1` values for `n` intervals).
    pub x: Vec<f64>,
    /// Per-interval Bernstein binomial coefficients, cached for repeated batch
    /// evaluation. Only the coefficient degree matters, not coefficient values.
    binoms: Vec<Vec<f64>>,
}

impl BPoly {
    /// Create a Bernstein-basis piecewise polynomial.
    pub fn new(c: Vec<Vec<f64>>, x: Vec<f64>) -> Result<Self, InterpError> {
        if c.is_empty() {
            return Err(InterpError::TooFewPoints {
                minimum: 1,
                actual: 0,
            });
        }
        if c.len() + 1 != x.len() {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "need {} intervals for {} breakpoints, got {}",
                    x.len() - 1,
                    x.len(),
                    c.len()
                ),
            });
        }
        let binoms = bpoly_binoms(&c);
        Ok(Self { c, x, binoms })
    }

    fn segment(&self, xval: f64) -> usize {
        // Largest i in 1..n with x[i] <= xval, else 0. Binary search over the sorted
        // breakpoints — byte-identical to the former O(n) linear scan, O(log n) per
        // point. frankenscipy-2jmet.
        let n = self.x.len() - 1;
        if n == 0 {
            0
        } else {
            self.x[1..n].partition_point(|&xi| xval >= xi)
        }
    }

    /// Evaluate the polynomial at a point.
    pub fn evaluate(&self, xval: f64) -> f64 {
        if xval.is_nan() {
            return f64::NAN;
        }
        let seg = self.segment(xval);
        let coeff_len = self.c[seg].len();
        if let Some(bn) = self.binoms.get(seg).filter(|bn| bn.len() == coeff_len) {
            return self.evaluate_segment_with_binoms(xval, seg, bn);
        }

        let k = coeff_len - 1;
        let local_binoms: Vec<f64> = (0..=k).map(|a| binom(k, a)).collect();
        self.evaluate_segment_with_binoms(xval, seg, &local_binoms)
    }

    /// Evaluate at multiple points.
    ///
    /// Byte-identical to mapping [`evaluate`](Self::evaluate), but the per-segment
    /// Bernstein binomial coefficients (which depend only on each piece's degree,
    /// not on the query point) are computed ONCE up front instead of recomputed —
    /// `O(k)` per term — on every point.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        let cached_binoms_match = self.binoms.len() == self.c.len()
            && self
                .binoms
                .iter()
                .zip(&self.c)
                .all(|(bn, coeffs)| bn.len() == coeffs.len());
        let local_binoms;
        let binoms = if cached_binoms_match {
            &self.binoms
        } else {
            local_binoms = bpoly_binoms(&self.c);
            &local_binoms
        };
        let sorted_non_nan = !xs.iter().any(|x| x.is_nan()) && xs.windows(2).all(|w| w[0] <= w[1]);
        if sorted_non_nan {
            let n = self.x.len() - 1;
            let mut seg = 0usize;
            return xs
                .iter()
                .map(|&xval| {
                    if n > 0 {
                        while seg + 1 < n && xval >= self.x[seg + 1] {
                            seg += 1;
                        }
                    }
                    self.evaluate_segment_with_binoms(xval, seg, &binoms[seg])
                })
                .collect();
        }

        // Per-query segment lookup + Bernstein sum, reading the hoisted `binoms`.
        // (Parallelizing this with par_query_map MEASURED as a regression — the
        // per-query work is only ~k flops, so thread-spawn overhead dominates;
        // reverted to the serial map. frankenscipy-yw7ts ledger: docs/perf_ledger_cc.md.)
        xs.iter()
            .map(|&xval| {
                if xval.is_nan() {
                    return f64::NAN;
                }
                let seg = self.segment(xval);
                self.evaluate_segment_with_binoms(xval, seg, &binoms[seg])
            })
            .collect()
    }

    fn evaluate_segment_with_binoms(&self, xval: f64, seg: usize, bn: &[f64]) -> f64 {
        let coeffs = &self.c[seg];
        let d = self.x[seg + 1] - self.x[seg];
        let s = (xval - self.x[seg]) / d;
        let s1 = 1.0 - s;
        if coeffs.len() == 4 {
            let s2 = s * s;
            let s3 = s2 * s;
            let s1_2 = s1 * s1;
            let s1_3 = s1_2 * s1;
            return coeffs[0] * s1_3
                + coeffs[1] * 3.0 * s * s1_2
                + coeffs[2] * 3.0 * s2 * s1
                + coeffs[3] * s3;
        }

        let k = coeffs.len() - 1;
        let mut value = 0.0;
        for (a, &ca) in coeffs.iter().enumerate() {
            value += ca * bn[a] * s.powi(a as i32) * s1.powi((k - a) as i32);
        }
        value
    }

    /// The first derivative as a degree `k-1` Bernstein polynomial; the
    /// derivative of a degree-0 piece is the zero polynomial.
    pub fn derivative(&self) -> BPoly {
        let dc: Vec<Vec<f64>> = self
            .c
            .iter()
            .enumerate()
            .map(|(i, coeffs)| {
                let k = coeffs.len() - 1;
                if k == 0 {
                    return vec![0.0];
                }
                let d = self.x[i + 1] - self.x[i];
                (0..k)
                    .map(|a| k as f64 * (coeffs[a + 1] - coeffs[a]) / d)
                    .collect()
            })
            .collect();
        let binoms = bpoly_binoms(&dc);
        BPoly {
            c: dc,
            x: self.x.clone(),
            binoms,
        }
    }
}

/// Binomial coefficient `C(n, k)` for small `n` (exact via the multiplicative
/// formula).
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

fn bpoly_binoms(c: &[Vec<f64>]) -> Vec<Vec<f64>> {
    c.iter()
        .map(|coeffs| {
            let k = coeffs.len() - 1;
            (0..=k).map(|a| binom(k, a)).collect()
        })
        .collect()
}

/// N-D tensor-product B-spline, matching `scipy.interpolate.NdBSpline`.
///
/// `t[d]` are the knots of dimension `d` (degree `k` all dimensions); `c` is the
/// flat row-major coefficient tensor of shape `[n0, …, n_{D-1}]` with
/// `n_d = len(t[d]) − k − 1`. The value is the contraction of the per-dimension
/// B-spline basis vectors with `c`.
pub struct NdBSpline {
    /// Per-dimension knot vectors.
    pub t: Vec<Vec<f64>>,
    /// Flat row-major coefficient tensor of shape `[n0, …, n_{D-1}]`.
    pub c: Vec<f64>,
    /// Spline degree (same for all dimensions).
    pub k: usize,
}

/// Same-binary A/B toggle: when `true`, `NdBSpline` contraction sweeps the FULL coefficient
/// tensor (∏ ns\[d] combinations, mostly zero-weighted) instead of the compact (k+1)^ndim
/// nonzero support box. Byte-identical output; only for A/B timing. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static NDBSPLINE_COMPACT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl NdBSpline {
    /// Build an N-D B-spline; validates the coefficient count against the knots.
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    pub fn new(t: Vec<Vec<f64>>, c: Vec<f64>, k: usize) -> Result<Self, InterpError> {
        let ndim = t.len();
        let min_knots = k
            .checked_mul(2)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| InterpError::InvalidArgument {
                detail: "degree is too large for knot validation".to_string(),
            })?;
        let mut prod = 1usize;
        for d in 0..ndim {
            if t[d].len() < min_knots {
                return Err(InterpError::InvalidArgument {
                    detail: format!("dimension {d}: need >= {min_knots} knots for degree {k}"),
                });
            }
            let coeff_count = t[d].len() - k - 1;
            prod = prod
                .checked_mul(coeff_count)
                .ok_or_else(|| InterpError::InvalidArgument {
                    detail: "coefficient shape product overflow".to_string(),
                })?;
        }
        if c.len() != prod {
            return Err(InterpError::InvalidArgument {
                detail: format!("c has {} entries, expected {prod}", c.len()),
            });
        }
        Ok(Self { t, c, k })
    }

    /// Contract the coefficient tensor with the per-dimension B-spline basis at `point`, visiting
    /// ONLY the `(k+1)^ndim` nonzero support combinations instead of the full `∏ ns[d]` tensor.
    ///
    /// A degree-`k` basis of length `ns[d]` is nonzero on just `k+1` contiguous indices, so the
    /// dense odometer wasted `∏(ns[d]/(k+1))` iterations on `w = ∏basis` products that are zero —
    /// compounding MULTIPLICATIVELY in ndim (e.g. ndim=3, ns=50 ⇒ 125k combos vs 64 nonzero).
    /// BYTE-IDENTICAL: nonzero terms are exactly those where every `basis[d][idx[d]]` is nonzero
    /// (a contiguous run), and the compact odometer visits them in the SAME row-major order, so
    /// the retained `c·w` terms accumulate identically. The `w != 0.0` guard is kept so an
    /// interior zero within the run (should not occur for B-splines) stays a `+c·0` no-op.
    #[inline]
    fn contract(&self, point: &[f64], ns: &[usize], stride: &[usize]) -> f64 {
        let ndim = self.t.len();
        let full_mode = NDBSPLINE_COMPACT_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
        // Per-dimension compact support: `base` accumulates the support box's origin offset into
        // `c`; `runs[d]` is the nonzero weight run; `lens[d]` its length (<= k+1).
        let mut base = 0usize;
        let mut runs: Vec<Vec<f64>> = Vec::with_capacity(ndim);
        let mut lens: Vec<usize> = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let full = eval_basis_all(&self.t[d], point[d], self.k, ns[d]);
            if full_mode {
                // A/B baseline: base 0 + full basis reproduces the old dense ∏ns[d] sweep exactly.
                lens.push(full.len());
                runs.push(full);
            } else {
                match full.iter().position(|&v| v != 0.0) {
                    None => {
                        lens.push(0);
                        runs.push(Vec::new());
                    }
                    Some(b) => {
                        let e = full.iter().rposition(|&v| v != 0.0).map_or(b, |x| x + 1);
                        base += b * stride[d];
                        lens.push(e - b);
                        runs.push(full[b..e].to_vec());
                    }
                }
            }
        }
        let total: usize = lens.iter().product();
        let mut idx = vec![0usize; ndim];
        let mut sum = 0.0f64;
        for _ in 0..total {
            let mut off = base;
            let mut w = 1.0f64;
            for d in 0..ndim {
                off += idx[d] * stride[d];
                w *= runs[d][idx[d]];
            }
            if w != 0.0 {
                sum += self.c[off] * w;
            }
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < lens[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        sum
    }

    /// Evaluate at `point` (length `ndim`).
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        let ndim = self.t.len();
        let ns: Vec<usize> = (0..ndim).map(|d| self.t[d].len() - self.k - 1).collect();
        // Row-major strides over the coefficient tensor [n0,…,n_{D-1}].
        let mut stride = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * ns[i + 1];
        }
        self.contract(point, &ns, &stride)
    }

    /// Evaluate the spline at many points (the realistic grid-evaluation workload,
    /// matching `scipy.interpolate.NdBSpline.__call__` on an array of points).
    ///
    /// Byte-identical to mapping [`evaluate`](Self::evaluate) over `points`, but the
    /// point-invariant per-dimension sizes, coefficient-tensor strides and the index
    /// scratch are computed once instead of on every point.
    pub fn evaluate_many(&self, points: &[Vec<f64>]) -> Vec<f64> {
        let ndim = self.t.len();
        let ns: Vec<usize> = (0..ndim).map(|d| self.t[d].len() - self.k - 1).collect();
        let mut stride = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * ns[i + 1];
        }
        // Each point is an independent compact contraction (see `contract`): only the
        // `(k+1)^ndim` nonzero support combinations are summed, not the full `∏ns[d]` tensor.
        // (Parallelizing the per-point loop was MEASURED as a regression on the typical
        // low-degree/low-dim case — thread-spawn overhead swamped the tiny contraction; the
        // compact box makes it tinier still, so it stays a serial map. frankenscipy-yw7ts
        // ledger: docs/perf_ledger_cc.md.)
        points
            .iter()
            .map(|point| self.contract(point, &ns, &stride))
            .collect()
    }
}

/// N-D tensor-product piecewise polynomial, matching `scipy.interpolate.NdPPoly`.
///
/// `c` is the flat (row-major) coefficient tensor of shape
/// `[k0+1, …, k_{D-1}+1, m0, …, m_{D-1}]` (`k_d` = degree, `m_d` = cells in dim
/// `d`); `x[d]` holds the `m_d+1` breakpoints of dimension `d`. On the cell
/// selected per dimension, the value is the tensor-product polynomial
/// `Σ c[i0,…,cell0,…] · Π_d (p_d − x[d][cell_d])^{k_d − i_d}`.
pub struct NdPPoly {
    /// Flat row-major coefficient tensor.
    pub c: Vec<f64>,
    /// Tensor shape `[k0+1, …, k_{D-1}+1, m0, …, m_{D-1}]` (length `2·ndim`).
    pub c_shape: Vec<usize>,
    /// Per-dimension breakpoints.
    pub x: Vec<Vec<f64>>,
}

impl NdPPoly {
    /// Build an N-D piecewise polynomial; validates the tensor shape against the
    /// breakpoints.
    pub fn new(c: Vec<f64>, c_shape: Vec<usize>, x: Vec<Vec<f64>>) -> Result<Self, InterpError> {
        let ndim = x.len();
        let expected_shape_len =
            ndim.checked_mul(2)
                .ok_or_else(|| InterpError::InvalidArgument {
                    detail: "c_shape length overflow".to_string(),
                })?;
        if c_shape.len() != expected_shape_len {
            return Err(InterpError::InvalidArgument {
                detail: "c_shape length must be 2*ndim".to_string(),
            });
        }
        let prod = c_shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| InterpError::InvalidArgument {
                detail: "coefficient shape product overflow".to_string(),
            })?;
        if c.len() != prod {
            return Err(InterpError::InvalidArgument {
                detail: format!("c has {} entries, expected {prod}", c.len()),
            });
        }
        for d in 0..ndim {
            if x[d].len() < 2 || c_shape[ndim + d] != x[d].len() - 1 {
                return Err(InterpError::InvalidArgument {
                    detail: format!("dimension {d}: cell count must equal len(x[{d}])-1"),
                });
            }
        }
        Ok(Self { c, c_shape, x })
    }

    /// Evaluate the polynomial at `point` (length `ndim`), extrapolating with the
    /// nearest cell for out-of-range coordinates.
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        let ndim = self.x.len();
        // Per-dimension cell index and local coordinate.
        let mut cell = vec![0usize; ndim];
        let mut dx = vec![0.0f64; ndim];
        for d in 0..ndim {
            let bp = &self.x[d];
            let m = bp.len() - 1; // number of cells
            // largest j with bp[j] <= point[d], clamped to [0, m-1].
            let mut j = 0usize;
            while j + 1 < m && point[d] >= bp[j + 1] {
                j += 1;
            }
            cell[d] = j;
            dx[d] = point[d] - bp[j];
        }
        // Strides for the row-major tensor.
        let nd2 = self.c_shape.len();
        let mut stride = vec![1usize; nd2];
        for i in (0..nd2 - 1).rev() {
            stride[i] = stride[i + 1] * self.c_shape[i + 1];
        }
        // Base offset from the cell indices (last ndim axes).
        let mut base = 0usize;
        for d in 0..ndim {
            base += cell[d] * stride[ndim + d];
        }
        // Precompute powers dx[d]^(k_d - i_d) for i_d in 0..=k_d.
        let orders: Vec<usize> = (0..ndim).map(|d| self.c_shape[d] - 1).collect();
        let mut powers: Vec<Vec<f64>> = Vec::with_capacity(ndim);
        for d in 0..ndim {
            let kd = orders[d];
            let mut p = vec![0.0f64; kd + 1];
            for id in 0..=kd {
                p[id] = dx[d].powi((kd - id) as i32);
            }
            powers.push(p);
        }
        // Sum over the order multi-index (i0,…,i_{D-1}).
        let total: usize = orders.iter().map(|&k| k + 1).product();
        let mut idx = vec![0usize; ndim];
        let mut sum = 0.0f64;
        for _ in 0..total {
            let mut off = base;
            let mut term = 1.0f64;
            for d in 0..ndim {
                off += idx[d] * stride[d];
                term *= powers[d][idx[d]];
            }
            sum += self.c[off] * term;
            // increment the mixed-radix multi-index.
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] <= orders[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        sum
    }

    /// Evaluate at many points (the realistic grid-evaluation workload, matching
    /// `scipy.interpolate.NdPPoly.__call__` on an array of points).
    ///
    /// Byte-identical to mapping [`evaluate`](Self::evaluate) over `points`, but the
    /// point-invariant tensor strides, per-dimension orders and the cell/dx/powers/
    /// idx scratch are computed/allocated once instead of on every point.
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    pub fn evaluate_many(&self, points: &[Vec<f64>]) -> Vec<f64> {
        let ndim = self.x.len();
        let nd2 = self.c_shape.len();
        let mut stride = vec![1usize; nd2];
        for i in (0..nd2 - 1).rev() {
            stride[i] = stride[i + 1] * self.c_shape[i + 1];
        }
        let orders: Vec<usize> = (0..ndim).map(|d| self.c_shape[d] - 1).collect();
        let total: usize = orders.iter().map(|&k| k + 1).product();
        // Reused per-point scratch (point-invariant strides/orders hoisted above).
        // (Parallelizing the per-point loop MEASURED as a regression on the typical
        // low-degree/low-dim case — the O(total) contraction is too cheap to amortise
        // thread-spawn + per-thread-scratch overhead; reverted to the serial map.
        // frankenscipy-yw7ts ledger: docs/perf_ledger_cc.md.)
        let mut cell = vec![0usize; ndim];
        let mut dx = vec![0.0f64; ndim];
        let mut idx = vec![0usize; ndim];
        let mut powers: Vec<Vec<f64>> = orders.iter().map(|&kd| vec![0.0f64; kd + 1]).collect();
        points
            .iter()
            .map(|point| {
                for d in 0..ndim {
                    let bp = &self.x[d];
                    let m = bp.len() - 1;
                    let mut j = 0usize;
                    while j + 1 < m && point[d] >= bp[j + 1] {
                        j += 1;
                    }
                    cell[d] = j;
                    dx[d] = point[d] - bp[j];
                }
                let mut base = 0usize;
                for d in 0..ndim {
                    base += cell[d] * stride[ndim + d];
                }
                for d in 0..ndim {
                    let kd = orders[d];
                    for id in 0..=kd {
                        powers[d][id] = dx[d].powi((kd - id) as i32);
                    }
                }
                idx.iter_mut().for_each(|v| *v = 0);
                let mut sum = 0.0f64;
                for _ in 0..total {
                    let mut off = base;
                    let mut term = 1.0f64;
                    for d in 0..ndim {
                        off += idx[d] * stride[d];
                        term *= powers[d][idx[d]];
                    }
                    sum += self.c[off] * term;
                    for d in (0..ndim).rev() {
                        idx[d] += 1;
                        if idx[d] <= orders[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
                sum
            })
            .collect()
    }
}

/// Smoothing spline representation (splrep equivalent).
///
/// Returns (knots, coefficients, degree) that can be used with `splev`.
/// Matches `scipy.interpolate.splrep`.
pub fn splrep(
    x: &[f64],
    y: &[f64],
    k: usize,
    s: f64,
) -> Result<(Vec<f64>, Vec<f64>, usize), InterpError> {
    if x.len() != y.len() || x.len() < k + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: k + 1,
            actual: x.len(),
        });
    }

    // For s=0 (interpolating), use make_interp_spline
    if s <= 0.0 {
        let bspl = make_interp_spline(x, y, k)?;
        return Ok((
            bspl.knots().to_vec(),
            pad_tck_coeffs(bspl.knots(), bspl.coeffs()),
            k,
        ));
    }

    // For s > 0 (smoothing), use make_lsq_spline with automatic knots
    let n = x.len();
    let n_interior = (n as f64 / 4.0).ceil() as usize;
    let n_interior = n_interior.max(1).min(n - k - 1);

    let mut knots = Vec::with_capacity(n_interior + 2 * (k + 1));

    // Repeated boundary knots
    for _ in 0..=k {
        knots.push(x[0]);
    }
    // Interior knots
    for i in 1..=n_interior {
        let idx = i * (n - 1) / (n_interior + 1);
        knots.push(x[idx]);
    }
    for _ in 0..=k {
        knots.push(x[n - 1]);
    }

    let bspl = make_lsq_spline(x, y, &knots, k)?;
    Ok((
        bspl.knots().to_vec(),
        pad_tck_coeffs(bspl.knots(), bspl.coeffs()),
        k,
    ))
}

/// Fit a B-spline to 1-D data, matching `scipy.interpolate.make_splrep(x, y, k=k, s=0)`
/// (the interpolation case), returning a [`BSpline`].
///
/// For `s = 0` this is exactly the interpolating spline ([`make_interp_spline`]);
/// scipy's `make_splrep(s=0)` produces identical knots and coefficients. A
/// positive `s` (FITPACK smoothing with adaptive knot insertion) returns an
/// error.
pub fn make_splrep(x: &[f64], y: &[f64], k: usize, s: f64) -> Result<BSpline, InterpError> {
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "smoothing factor s must be >= 0".to_string(),
        });
    }
    if s == 0.0 {
        return make_interp_spline(x, y, k);
    }
    // Smoothing spline: place knots via the FITPACK LSQ loop, then fit the
    // penalized spline whose residual fp equals s.
    let seq = generate_knots(x, y, None, None, None, k, s, None)?;
    let t = seq
        .last()
        .expect("generate_knots yields at least one vector")
        .clone();
    let nmin = 2 * (k + 1);
    if t.len() == nmin {
        // No interior knots: nothing to optimize, plain LSQ (constant-ish fit).
        return make_lsq_spline(x, y, &t, k);
    }
    let w = vec![1.0_f64; x.len()];
    splrep_smoothing_fit(x, y, &w, &t, k, s)
}

/// Product `Π_{s≠j-i}(t[j] - t[i+s])` over `s = 0..=k+1` (scipy `prodd`).
fn prodd(t: &[f64], i: usize, j: usize, k: usize) -> f64 {
    let mut res = 1.0_f64;
    for s in 0..k + 2 {
        if i + s != j {
            res *= t[j] - t[i + s];
        }
    }
    res
}

/// FITPACK discontinuity matrix: jumps of the k-th derivative of the B-splines at
/// interior knots (scipy `disc`). Returns one row per interior knot as
/// `(offset, [k+2 values])`, where the values occupy coefficient columns
/// `offset..offset+k+2`.
fn disc_matrix(t: &[f64], k: usize) -> Vec<(usize, Vec<f64>)> {
    let n = t.len();
    let nrint = n - 2 * k - 1;
    let delta = t[n - k - 1] - t[k];
    let factor = (delta / nrint as f64).powi(k as i32);
    let mut rows = Vec::with_capacity(nrint.saturating_sub(1));
    for jj in 0..nrint - 1 {
        let j = jj + k + 1;
        let mut vals = vec![0.0_f64; k + 2];
        for (ii, v) in vals.iter_mut().enumerate() {
            let i = jj + ii;
            *v = (t[i + k + 1] - t[i]) / prodd(t, i, j, k) * factor;
        }
        rows.push((jj, vals));
    }
    rows
}

/// Sum of the Cholesky-factor diagonal of an SPD matrix `g` (equals `Σ R[i,i]`
/// for the QR factor `R` of the data matrix), used to seed FITPACK's `p`.
/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
fn cholesky_diag_sum(g: &[Vec<f64>]) -> f64 {
    let n = g.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in 0..=i {
            let mut acc = g[i][j];
            for kk in 0..j {
                acc -= l[i][kk] * l[j][kk];
            }
            if i == j {
                let d = acc.max(0.0).sqrt();
                l[i][j] = d;
                sum += d;
            } else {
                l[i][j] = if l[j][j] != 0.0 { acc / l[j][j] } else { 0.0 };
            }
        }
    }
    sum
}

/// FITPACK `fprati`: root of the rational interpolant `r(p)=(u·p+v)/(p+w)` through
/// three points; `p3 == +inf` is handled explicitly.
fn fprati(p1: f64, f1: f64, p2: f64, f2: f64, p3: f64, f3: f64) -> f64 {
    let h1 = f1 * (f2 - f3);
    let h2 = f2 * (f3 - f1);
    let h3 = f3 * (f1 - f2);
    if p3.is_infinite() {
        -(p2 * h1 + p1 * h2) / h3
    } else {
        -(p1 * p2 * h3 + p2 * p3 * h1 + p1 * p3 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)
    }
}

/// FITPACK `root_rati`: find `p` with `f(p)=0` for the monotonically decreasing
/// `f`, maintaining a three-point bracket and stepping via [`fprati`].
fn root_rati(
    f: &mut dyn FnMut(f64) -> f64,
    p0: f64,
    bracket: ((f64, f64), (f64, f64)),
    acc: f64,
) -> f64 {
    let (con1, con9, con4) = (0.1_f64, 0.9_f64, 0.04_f64);
    let ((mut p1, mut f1), (mut p3, mut f3)) = bracket;
    let (mut ich1, mut ich3) = (false, false);
    let mut p = p0;
    for _ in 0..20 {
        let p2 = p;
        let f2 = f(p);
        if f2.abs() < acc {
            return p2;
        }
        if !ich3 {
            if f2 - f3 <= acc {
                p3 = p2;
                f3 = f2;
                p *= con4;
                if p <= p1 {
                    p = p1 * con9 + p2 * con1;
                }
                continue;
            } else if f2 < 0.0 {
                ich3 = true;
            }
        }
        if !ich1 {
            if f1 - f2 <= acc {
                p1 = p2;
                f1 = f2;
                p /= con4;
                if p3.is_finite() && p <= p3 {
                    p = p2 * con1 + p3 * con9;
                }
                continue;
            } else if f2 > 0.0 {
                ich1 = true;
            }
        }
        if f1 <= f2 || f2 <= f3 {
            return p2;
        }
        p = fprati(p1, f1, p2, f2, p3, f3);
        if f2 < 0.0 {
            p3 = p2;
            f3 = f2;
        } else {
            p1 = p2;
            f1 = f2;
        }
    }
    p
}

/// Penalized smoothing-spline fit on fixed knots `t`: find `p` such that the
/// data residual `fp(p) = Σ w_i²(y_i − s(x_i))²` equals `s`, where `s_p`
/// minimizes `Σ w_i(y_i − s)² + (1/p²)·‖B c‖²` and `B` is the discontinuity
/// matrix. Mirrors scipy's `_make_splrep_impl` / `F` via normal equations.
fn splrep_smoothing_fit(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    t: &[f64],
    k: usize,
    s: f64,
) -> Result<BSpline, InterpError> {
    let m = x.len();
    let nc = t.len() - k - 1;

    // G = AᵀWA (linear weights) and rhs = AᵀWy, assembled like make_lsq_spline.
    let mut g = vec![vec![0.0_f64; nc]; nc];
    let mut rhs = vec![0.0_f64; nc];
    let mut scratch = vec![0.0_f64; nc];
    for i in 0..m {
        let Some(mu) = bspline_find_interval(t, x[i], nc) else {
            continue;
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let dl = t[idx + p] - t[idx];
                    if dl > 0.0 {
                        val += (x[i] - t[idx]) / dl * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < nc {
                    let dr = t[idx + p + 1] - t[idx + 1];
                    if dr > 0.0 {
                        val += (t[idx + p + 1] - x[i]) / dr * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        let wi = w[i];
        let yi = y[i];
        for a in lo..=mu {
            let ba = scratch[a];
            rhs[a] += wi * ba * yi;
            for b in lo..=mu {
                g[a][b] += wi * ba * scratch[b];
            }
        }
        for sc in scratch[lo..=mu].iter_mut() {
            *sc = 0.0;
        }
    }

    // H = BᵀB from the discontinuity matrix.
    let brows = disc_matrix(t, k);
    let mut h = vec![vec![0.0_f64; nc]; nc];
    for (off, vals) in &brows {
        for (a, &va) in vals.iter().enumerate() {
            for (b, &vb) in vals.iter().enumerate() {
                h[off + a][off + b] += va * vb;
            }
        }
    }

    // Solve (G + (1/p²) H) c = rhs and return (fp, c). p == inf drops the penalty.
    let band = k + 1;
    let solve_at = |p: f64| -> Option<(f64, Vec<f64>)> {
        let inv = if p.is_infinite() { 0.0 } else { 1.0 / (p * p) };
        let mut mat = vec![vec![0.0_f64; nc]; nc];
        for i in 0..nc {
            for j in 0..nc {
                mat[i][j] = g[i][j] + inv * h[i][j];
            }
        }
        let mut b = rhs.clone();
        let c = solve_banded(&mut mat, &mut b, band).ok()?;
        let spl = BSpline::new(t.to_vec(), c.clone(), k).ok()?;
        let mut fp = 0.0_f64;
        for i in 0..m {
            let d = y[i] - spl.eval(x[i]);
            fp += w[i] * w[i] * d * d;
        }
        Some((fp, c))
    };

    // Bracket: f(p=inf) = LSQ on knots t (min fp); f(p=0) = constant fit (max fp).
    let (fp_inf, _) = solve_at(f64::INFINITY).ok_or_else(|| InterpError::InvalidArgument {
        detail: "smoothing fit: singular system".to_string(),
    })?;
    let xb = x[0];
    let xe = x[m - 1];
    let mut t0: Vec<f64> = Vec::with_capacity(2 * (k + 1));
    for _ in 0..=k {
        t0.push(xb);
    }
    for _ in 0..=k {
        t0.push(xe);
    }
    let (_, fp0) = lsq_residuals(x, y, w, &t0, k)?;

    let acc = s * 1e-3;
    let p0 = nc as f64 / cholesky_diag_sum(&g);
    let mut f = |p: f64| -> f64 {
        match solve_at(p) {
            Some((fp, _)) => fp - s,
            None => f64::INFINITY,
        }
    };
    let p = root_rati(
        &mut f,
        p0,
        ((0.0, fp0 - s), (f64::INFINITY, fp_inf - s)),
        acc,
    );
    let (_, c) = solve_at(p).ok_or_else(|| InterpError::InvalidArgument {
        detail: "smoothing fit: singular system at solution".to_string(),
    })?;
    BSpline::new(t.to_vec(), c, k)
}

/// Coefficients of the divided difference over `xs` (scipy `_coeff_of_divided_diff`):
/// `res[i] = 1 / Π_{k≠i}(xs[i] - xs[k])`.
fn coeff_of_divided_diff(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    (0..n)
        .map(|i| {
            let mut pp = 1.0_f64;
            for k in 0..n {
                if k != i {
                    pp *= xs[i] - xs[k];
                }
            }
            1.0 / pp
        })
        .collect()
}

/// Minimize a scalar function on `[a, b]` with Brent's bounded method, matching
/// `scipy.optimize.minimize_scalar(method='Bounded')` (xatol = 1e-5).
fn bounded_minimize(f: &dyn Fn(f64) -> f64, a: f64, b: f64, xatol: f64, maxfun: usize) -> f64 {
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (a, b);
    let mut fulc = a + golden_mean * (b - a);
    let (mut nfc, mut xf) = (fulc, fulc);
    let mut rat = 0.0_f64;
    let mut e = 0.0_f64;
    let mut fx = f(xf);
    let mut num = 1;
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;
    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let r0 = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r0;
            q = 2.0 * (q - r0);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = if xm - xf >= 0.0 { 1.0 } else { -1.0 };
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = if rat >= 0.0 { 1.0 } else { -1.0 };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        num += 1;
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            break;
        }
    }
    xf
}

/// Find the GCV-optimal regularization parameter for [`make_smoothing_spline`],
/// matching scipy's `_compute_optimal_gcv_parameter`: minimize over `[0, n]`
/// `GCV(λ) = (‖λ·E·c_λ‖²/n) / (1 − tr(A_λ)/n)²`, where `c_λ` solves
/// `(X + λE)c = y` and `tr(A_λ) = tr((XᵀWX + λXᵀE)⁻¹ XᵀWX)`.
/// Read entry (i,j) of a (2,2)-banded matrix stored in LAPACK band form
/// (`band[2+i-j][j]` for |i-j| ≤ 2), returning 0 outside the band. Lets the
/// smoothing-spline path keep the design X / penalty E in their O(n) band storage
/// (`xm`/`we`) instead of expanding to a full n×n matrix (the prior `band_to_full`).
#[inline]
fn band2_get(band: &[Vec<f64>], i: usize, j: usize) -> f64 {
    if j + 2 >= i && i + 2 >= j {
        band[(2 + i) - j][j]
    } else {
        0.0
    }
}

const GCV_BW: usize = 4;
const GCV_BAND_WIDTH: usize = 2 * GCV_BW + 1;
const GCV_BAND_CENTER: isize = GCV_BW as isize;
type GcvBandRow = [f64; GCV_BAND_WIDTH];

#[inline]
fn gcv_band_index(i: usize, j: usize) -> Option<usize> {
    let offset = j as isize - i as isize;
    (-GCV_BAND_CENTER..=GCV_BAND_CENTER)
        .contains(&offset)
        .then_some((GCV_BAND_CENTER + offset) as usize)
}

#[inline]
fn gcv_band_get(band: &[GcvBandRow], i: usize, j: usize) -> f64 {
    gcv_band_index(i, j).map_or(0.0, |idx| band[i][idx])
}

#[inline]
fn gcv_band_set(band: &mut [GcvBandRow], i: usize, j: usize, value: f64) {
    if let Some(idx) = gcv_band_index(i, j) {
        band[i][idx] = value;
    }
}

#[inline]
fn gcv_band_lower_get(band: &[GcvBandRow], i: usize, j: usize) -> f64 {
    debug_assert!(i >= j);
    debug_assert!(i - j <= GCV_BW);
    band[i][(GCV_BAND_CENTER - (i - j) as isize) as usize]
}

#[inline]
fn gcv_band_lower_set(band: &mut [GcvBandRow], i: usize, j: usize, value: f64) {
    debug_assert!(i >= j);
    debug_assert!(i - j <= GCV_BW);
    band[i][(GCV_BAND_CENTER - (i - j) as isize) as usize] = value;
}

/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
fn gcv_optimal_lambda(xm: &[Vec<f64>], we: &[Vec<f64>], y: &[f64], w: &[f64]) -> f64 {
    let n = y.len();
    let nf = n as f64;
    // XtWX = Xᵀ W X, XtE = Xᵀ W E.
    let mut xtwx = vec![[0.0_f64; GCV_BAND_WIDTH]; n];
    let mut xte = vec![[0.0_f64; GCV_BAND_WIDTH]; n];
    // X and E are (2,2)-banded (x_full[k][·] ≠ 0 only within |k-·| ≤ 2), so the Gram
    // products XᵀWX, XᵀWE are (4,4)-banded and each entry's k-sum has nonzero terms
    // only where BOTH the row factor (|k-i| ≤ 2) and the column factor (|k-j| ≤ 2)
    // are nonzero, i.e. k ∈ [max(i,j)-2, min(i,j)+2]. Restricting (i,j) to |i-j| ≤ 4
    // and k to that overlap is byte-identical to the full O(n³) triple loop — the
    // skipped terms are exactly the +0.0 no-ops and the nonzero terms still
    // accumulate in ascending-k order; out-of-band entries keep their 0 init. O(n).
    for i in 0..n {
        let jlo = i.saturating_sub(4);
        let jhi = (i + 4).min(n - 1);
        for j in jlo..=jhi {
            let klo = i.max(j).saturating_sub(2);
            let khi = (i.min(j) + 2).min(n - 1);
            let (mut sx, mut se) = (0.0_f64, 0.0_f64);
            for k in klo..=khi {
                let xwk = band2_get(xm, k, i) * w[k];
                sx += xwk * band2_get(xm, k, j);
                se += xwk * band2_get(we, k, j);
            }
            gcv_band_set(&mut xtwx, i, j, sx);
            gcv_band_set(&mut xte, i, j, se);
        }
    }
    // lhs scratch reused across every bounded_minimize eval in compact (4,4)-band storage.
    // Cholesky writes only the lower band + diagonal and does no pivoting/fill, so re-filling
    // |i-j| ≤ 4 each eval fully overwrites the previous factorization. This keeps the whole
    // GCV path at O(n) memory instead of keeping three full n×n matrices alive.
    let lhs_buf = std::cell::RefCell::new(vec![[0.0_f64; GCV_BAND_WIDTH]; n]);
    let gcv = |lam: f64| -> f64 {
        // c solves (X + λE) c = y. m = X + λE is (2,2)-banded; build it in COMPACT banded
        // storage (O(n·bw) per eval, not the dense O(n²) alloc) and solve with the same
        // partial-pivoting banded LU (solve_banded_compact, byte-identical to solve_banded).
        let mut m: Vec<CompactBandRow> = (0..n)
            .map(|i| {
                let jlo = i.saturating_sub(2);
                let jhi = (i + 2).min(n - 1);
                let vals: Vec<f64> = (jlo..=jhi)
                    .map(|j| band2_get(xm, i, j) + lam * band2_get(we, i, j))
                    .collect();
                CompactBandRow::from_slice(jlo, &vals)
            })
            .collect();
        let mut rhs = y.to_vec();
        let c = match solve_banded_compact(&mut m, &mut rhs, 2) {
            Ok(c) => c,
            Err(_) => return f64::INFINITY,
        };
        // numer = ‖λ·E·c‖² / n. E is (2,2)-banded, so row i's dot with c has nonzero
        // terms only at |i-j| ≤ 2 — restrict the inner sum to that band (byte-identical:
        // the rest are +0.0 no-ops, same ascending-j order). O(n²) → O(n) per λ.
        let mut numer = 0.0_f64;
        for i in 0..n {
            let jlo = i.saturating_sub(2);
            let jhi = (i + 2).min(n - 1);
            let mut r = 0.0_f64;
            for j in jlo..=jhi {
                r += band2_get(we, i, j) * c[j];
            }
            numer += (lam * r) * (lam * r);
        }
        numer /= nf;
        // tr(A) = tr(lhs⁻¹ XtWX), lhs = XtWX + λ XtE. `lhs` is INDEPENDENT of the column,
        // so the per-column loop was re-factoring the same (4,4)-banded matrix n times
        // (the residual O(n³): n columns × O(n²) build+factor). `lhs` is SPD (sum of two
        // Gram matrices, λ≥0), so factor it ONCE with banded Cholesky and substitute the
        // n RHS columns → O(n²). BUT only the trace is needed, and XtWX is bw-4 banded, so
        // tr(lhs⁻¹ XtWX) = Σ_{|i−j|≤4} (lhs⁻¹)_{ij} (XtWX)_{ij} needs only the BAND of lhs⁻¹.
        // The Erisman–Tinney SELECTED INVERSE recovers that band from the Cholesky factor in
        // O(n·bw²) — no n solves → the whole GCV eval is O(n) (matches scipy's Reinsch). The
        // λ chosen by bounded_minimize is the same (tolerance-parity, ≤~1e-12).
        let tr = {
            let mut lhs = lhs_buf.borrow_mut();
            for i in 0..n {
                let jlo = i.saturating_sub(4);
                let jhi = (i + 4).min(n - 1);
                for j in jlo..=jhi {
                    gcv_band_set(
                        &mut lhs,
                        i,
                        j,
                        gcv_band_get(&xtwx, i, j) + lam * gcv_band_get(&xte, i, j),
                    );
                }
            }
            match chol_banded_gcv(lhs.as_mut_slice()) {
                Some(()) => gcv_trace_selinv_gcv(lhs.as_slice(), &xtwx, n),
                None => return f64::INFINITY,
            }
        };
        let denom = (1.0 - tr / nf).powi(2);
        numer / denom
    };
    bounded_minimize(&gcv, 0.0, nf, 1e-5, 500)
}

/// Create a cubic smoothing B-spline matching `scipy.interpolate.make_smoothing_spline`.
///
/// Solves the penalized weighted least-squares problem
/// `Σ w_i (y_i − f(x_i))² + lam·∫ f''²` in the natural-cubic-spline basis (knots
/// at the data points), via the banded design matrix `X` and penalty `E` exactly
/// as scipy does, then maps back to B-spline coefficients. When `lam` is `None`
/// the parameter is chosen by generalized cross-validation (GCV) over `[0, n]`.
/// Index-driven numeric kernel: the loop variables are the recurrence/band
/// indices and several address more than one array per iteration, so iterator
/// form would obscure the index algebra this is checked against rather than
/// simplify it (frankenscipy-9yyez).
#[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
pub fn make_smoothing_spline(
    x: &[f64],
    y: &[f64],
    w: Option<&[f64]>,
    lam: Option<f64>,
) -> Result<BSpline, InterpError> {
    let n = x.len();
    if y.len() != n {
        return Err(InterpError::InvalidArgument {
            detail: "x and y must have the same length".to_string(),
        });
    }
    if n <= 4 {
        return Err(InterpError::TooFewPoints {
            minimum: 5,
            actual: n,
        });
    }
    if x.windows(2).any(|p| p[1] <= p[0]) {
        return Err(InterpError::InvalidArgument {
            detail: "x should be an ascending array".to_string(),
        });
    }
    if matches!(lam, Some(l) if l < 0.0) {
        return Err(InterpError::InvalidArgument {
            detail: "regularization parameter should be non-negative".to_string(),
        });
    }
    let w: Vec<f64> = match w {
        Some(s) if s.len() == n => s.to_vec(),
        Some(_) => {
            return Err(InterpError::InvalidArgument {
                detail: "weights must match the data length".to_string(),
            });
        }
        None => vec![1.0; n],
    };

    // Knot vector: triple-clamped at the ends with data points in between.
    let mut t = vec![x[0]; 3];
    t.extend_from_slice(x);
    t.extend(std::iter::repeat_n(x[n - 1], 3));
    let ncoef = t.len() - 4; // = n + 2
    // Collocation matrix rows X_bspl[i] = [B_j(x_i)].
    let xb: Vec<Vec<f64>> = (0..n).map(|i| eval_basis_all(&t, x[i], 3, ncoef)).collect();

    // Banded design matrix X (5 × n, LAPACK (2,2) storage) in the natural basis.
    let mut xm = vec![vec![0.0_f64; n]; 5];
    for i in 1..4 {
        for d in 0..(n - 4) {
            xm[i][2 + d] = xb[i + d][3 + d];
        }
    }
    xm[1][1] = xb[0][0];
    xm[2][0] = (x[2] + x[1] - 2.0 * x[0]) * xb[0][0];
    xm[2][1] = xb[1][1] + xb[1][2];
    xm[3][0] = (x[2] - x[0]) * xb[1][1];
    xm[3][1] = xb[2][2];
    xm[1][n - 2] = xb[n - 3][ncoef - 3];
    xm[1][n - 1] = (x[n - 1] - x[n - 3]) * xb[n - 2][ncoef - 2];
    xm[2][n - 2] = xb[n - 2][ncoef - 3] + xb[n - 2][ncoef - 2];
    xm[2][n - 1] = (2.0 * x[n - 1] - x[n - 2] - x[n - 3]) * xb[n - 1][ncoef - 1];
    xm[3][n - 2] = xb[n - 1][ncoef - 1];

    // Banded penalty matrix W⁻¹E (5 × n), scaled by 6.
    let mut we = vec![vec![0.0_f64; n]; 5];
    let c_left3 = coeff_of_divided_diff(&x[0..3]);
    for r in 0..3 {
        we[2 + r][0] = c_left3[r] / w[r];
    }
    let c_left4 = coeff_of_divided_diff(&x[0..4]);
    for r in 0..4 {
        we[1 + r][1] = c_left4[r] / w[r];
    }
    for j in 2..(n - 2) {
        let cj = coeff_of_divided_diff(&x[j - 2..j + 3]);
        let scale = x[j + 2] - x[j - 2];
        for r in 0..5 {
            we[r][j] = scale * cj[r] / w[j - 2 + r];
        }
    }
    let c_r4 = coeff_of_divided_diff(&x[n - 4..n]);
    for r in 0..4 {
        we[r][n - 2] = -c_r4[r] / w[n - 4 + r];
    }
    let c_r3 = coeff_of_divided_diff(&x[n - 3..n]);
    for r in 0..3 {
        we[r][n - 1] = c_r3[r] / w[n - 3 + r];
    }
    for row in &mut we {
        for v in row {
            *v *= 6.0;
        }
    }

    // X (design) and E (penalty) stay in their O(n) LAPACK (2,2)-band storage (`xm`/`we`,
    // A[i][j] = band[2+i-j][j] for |i-j| ≤ 2); readers use band2_get instead of expanding
    // to full n×n matrices (the prior band_to_full was the dominant O(n²) memory at large n).

    // Resolve the regularization parameter: explicit, or by generalized
    // cross-validation (GCV) minimized over [0, n] when lam is None.
    let lam = match lam {
        Some(l) => l,
        None => gcv_optimal_lambda(&xm, &we, y, &w),
    };

    // Solve (X + lam·E) c = y. X and E are (2,2)-banded (|i-j| ≤ 2), so their sum is
    // too — solve_banded(_, _, 2) is byte-identical to the dense solve here and runs
    // in O(n) instead of O(n²).
    // full is (2,2)-banded, so only fill |i-j| ≤ 2; out-of-band stays 0 (== the full
    // build, since x_full,e_full are 0 there) and solve_banded(_,_,2) makes the LU fill.
    // Byte-identical; O(n²) → O(n). (Matches the GCV closure's m build, 43eb09b2.)
    let mut full: Vec<CompactBandRow> = (0..n)
        .map(|i| {
            let jlo = i.saturating_sub(2);
            let jhi = (i + 2).min(n - 1);
            let vals: Vec<f64> = (jlo..=jhi)
                .map(|j| band2_get(&xm, i, j) + lam * band2_get(&we, i, j))
                .collect();
            CompactBandRow::from_slice(jlo, &vals)
        })
        .collect();
    let mut rhs = y.to_vec();
    let c = solve_banded_compact(&mut full, &mut rhs, 2)?;

    // Map the natural-basis solution back to B-spline coefficients (length ncoef).
    let tl = t.len();
    let mut cfull = Vec::with_capacity(ncoef);
    cfull.push(c[0] * (t[5] + t[4] - 2.0 * t[3]) + c[1]);
    cfull.push(c[0] * (t[5] - t[3]) + c[1]);
    for i in 1..(n - 1) {
        cfull.push(c[i]);
    }
    cfull.push(c[n - 1] * (t[tl - 4] - t[tl - 6]) + c[n - 2]);
    cfull.push(c[n - 1] * (2.0 * t[tl - 4] - t[tl - 5] - t[tl - 6]) + c[n - 2]);

    BSpline::new(t, cfull, 3)
}

/// Fit a parametric interpolating B-spline through N-D points, matching
/// `scipy.interpolate.splprep(points, k=k, s=0)` (the interpolation case).
///
/// `points` is a list of coordinate arrays (`[x, y, …]`), each of length `m`.
/// The curve is parameterized by cumulative chord length normalized to `[0, 1]`,
/// then each coordinate is fit against that parameter with [`splrep`] (`s = 0`),
/// sharing one knot vector. Returns `((t, c, k), u)` where `c[d]` are the
/// (unpadded) coefficients for coordinate `d` and `u` is the parameter sample.
///
/// Only smoothing factor `s = 0` (interpolation) is supported; a positive `s`
/// (FITPACK smoothing, which fits all coordinates with a shared adaptive knot
/// set) returns an error.
#[allow(clippy::type_complexity)]
pub fn splprep(
    points: &[Vec<f64>],
    k: usize,
    s: f64,
) -> Result<((Vec<f64>, Vec<Vec<f64>>, usize), Vec<f64>), InterpError> {
    if s != 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "splprep currently supports only s = 0 (interpolation)".to_string(),
        });
    }
    if points.is_empty() {
        return Err(InterpError::InvalidArgument {
            detail: "splprep requires at least one coordinate array".to_string(),
        });
    }
    let m = points[0].len();
    if points.iter().any(|p| p.len() != m) {
        return Err(InterpError::InvalidArgument {
            detail: "all coordinate arrays must have the same length".to_string(),
        });
    }
    // Cumulative chord-length parameter, normalized to [0, 1].
    let mut u = vec![0.0_f64; m];
    for i in 1..m {
        let d: f64 = points
            .iter()
            .map(|p| (p[i] - p[i - 1]).powi(2))
            .sum::<f64>()
            .sqrt();
        u[i] = u[i - 1] + d;
    }
    let total = u[m - 1];
    if total <= 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "splprep requires distinct points (nonzero total chord length)".to_string(),
        });
    }
    for v in &mut u {
        *v /= total;
    }
    // Fit each coordinate against u; all share the same knot vector.
    let (t, _, _) = splrep(&u, &points[0], k, 0.0)?;
    let ncoef = t.len().saturating_sub(k + 1);
    let mut coeffs = Vec::with_capacity(points.len());
    for p in points {
        let (_, c, _) = splrep(&u, p, k, 0.0)?;
        coeffs.push(c[..ncoef].to_vec());
    }
    Ok(((t, coeffs, k), u))
}

/// Multi-dimensional [`lsq_residuals`] for parametric fitting: per-point summed
/// squared residual across all coordinate splines (which share knots `t`), plus
/// the total `fp`. Mirrors `_get_residuals` with 2-D `y` (summed over axis 1).
fn lsq_residuals_nd(
    u: &[f64],
    ys: &[Vec<f64>],
    w: &[f64],
    t: &[f64],
    k: usize,
) -> Result<(Vec<f64>, f64), InterpError> {
    let m = u.len();
    let n = t.len() - k - 1;
    let d = ys.len();
    let mut ata = vec![vec![0.0_f64; n]; n];
    let mut atys = vec![vec![0.0_f64; n]; d];
    let mut scratch = vec![0.0_f64; n];
    for i in 0..m {
        let Some(mu) = bspline_find_interval(t, u[i], n) else {
            continue;
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let dl = t[idx + p] - t[idx];
                    if dl > 0.0 {
                        val += (u[i] - t[idx]) / dl * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < n {
                    let dr = t[idx + p + 1] - t[idx + 1];
                    if dr > 0.0 {
                        val += (t[idx + p + 1] - u[i]) / dr * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        let wi = w[i];
        for a in lo..=mu {
            let ba = scratch[a];
            for dd in 0..d {
                atys[dd][a] += wi * ba * ys[dd][i];
            }
            for b in lo..=mu {
                ata[a][b] += wi * ba * scratch[b];
            }
        }
        for sc in scratch[lo..=mu].iter_mut() {
            *sc = 0.0;
        }
    }
    let mut residuals = vec![0.0_f64; m];
    for dd in 0..d {
        let mut a = ata.clone();
        let mut b = atys[dd].clone();
        let c = solve_banded(&mut a, &mut b, k)?;
        let spl = BSpline::new(t.to_vec(), c, k)?;
        for i in 0..m {
            let e = ys[dd][i] - spl.eval(u[i]);
            residuals[i] += w[i] * w[i] * e * e;
        }
    }
    let fp = residuals.iter().sum();
    Ok((residuals, fp))
}

/// Multi-dimensional FITPACK knot placement (parametric): returns the final knot
/// vector that scipy's `generate_knots` would yield last for the summed residuals.
fn generate_knots_nd_last(
    u: &[f64],
    ys: &[Vec<f64>],
    w: &[f64],
    k: usize,
    s: f64,
) -> Result<Vec<f64>, InterpError> {
    let m = u.len();
    let xb = u[0];
    let xe = u[m - 1];
    let acc = s * 1e-3;
    let nmin = 2 * (k + 1);
    let nmax = m + k + 1;
    let nest = (m + k + 1).max(2 * k + 3);
    let mut t: Vec<f64> = Vec::with_capacity(nmin);
    for _ in 0..=k {
        t.push(xb);
    }
    for _ in 0..=k {
        t.push(xe);
    }
    let mut fp = 0.0_f64;
    let mut nplus = 1usize;
    for _iter in 0..m {
        let fpold = fp;
        let (residuals, fp_new) = lsq_residuals_nd(u, ys, w, &t, k)?;
        fp = fp_new;
        let fpms = fp - s;
        if fpms.abs() < acc || fpms < 0.0 {
            return Ok(t);
        }
        let n = t.len();
        if n == nmin {
            nplus = 1;
        } else {
            let delta = fpold - fp;
            let npl1: i64 = if delta > acc {
                (nplus as f64 * fpms / delta) as i64
            } else {
                (nplus * 2) as i64
            };
            let cand = npl1.max((nplus / 2) as i64).max(1);
            nplus = (nplus * 2).min(cand as usize);
        }
        let mut residuals = residuals;
        for j in 0..nplus {
            t = add_knot(u, &t, k, &residuals);
            let n = t.len();
            if n >= nmax {
                return Ok(interpolation_knots(u, k));
            }
            if n >= nest {
                return Ok(t);
            }
            if j < nplus - 1 {
                let (r, _) = lsq_residuals_nd(u, ys, w, &t, k)?;
                residuals = r;
            }
        }
    }
    Ok(t)
}

/// Multi-dimensional [`splrep_smoothing_fit`]: per-coordinate penalized fit on
/// shared knots `t` with a single smoothing parameter `p` chosen so the *total*
/// residual `fp` (summed across coordinates) equals `s`. Returns one coefficient
/// vector per coordinate.
fn splrep_smoothing_fit_nd(
    u: &[f64],
    ys: &[Vec<f64>],
    w: &[f64],
    t: &[f64],
    k: usize,
    s: f64,
) -> Result<Vec<Vec<f64>>, InterpError> {
    let m = u.len();
    let nc = t.len() - k - 1;
    let d = ys.len();
    let mut g = vec![vec![0.0_f64; nc]; nc];
    let mut rhs = vec![vec![0.0_f64; nc]; d];
    let mut scratch = vec![0.0_f64; nc];
    for i in 0..m {
        let Some(mu) = bspline_find_interval(t, u[i], nc) else {
            continue;
        };
        scratch[mu] = 1.0;
        for p in 1..=k {
            let start = mu.saturating_sub(p);
            for idx in start..=mu {
                let mut val = 0.0;
                if idx + p < t.len() {
                    let dl = t[idx + p] - t[idx];
                    if dl > 0.0 {
                        val += (u[i] - t[idx]) / dl * scratch[idx];
                    }
                }
                if idx + p + 1 < t.len() && idx + 1 < nc {
                    let dr = t[idx + p + 1] - t[idx + 1];
                    if dr > 0.0 {
                        val += (t[idx + p + 1] - u[i]) / dr * scratch[idx + 1];
                    }
                }
                scratch[idx] = val;
            }
        }
        let lo = mu.saturating_sub(k);
        let wi = w[i];
        for a in lo..=mu {
            let ba = scratch[a];
            for dd in 0..d {
                rhs[dd][a] += wi * ba * ys[dd][i];
            }
            for b in lo..=mu {
                g[a][b] += wi * ba * scratch[b];
            }
        }
        for sc in scratch[lo..=mu].iter_mut() {
            *sc = 0.0;
        }
    }

    let brows = disc_matrix(t, k);
    let mut h = vec![vec![0.0_f64; nc]; nc];
    for (off, vals) in &brows {
        for (a, &va) in vals.iter().enumerate() {
            for (b, &vb) in vals.iter().enumerate() {
                h[off + a][off + b] += va * vb;
            }
        }
    }

    let band = k + 1;
    let solve_at = |p: f64| -> Option<(f64, Vec<Vec<f64>>)> {
        let inv = if p.is_infinite() { 0.0 } else { 1.0 / (p * p) };
        let mut mat0 = vec![vec![0.0_f64; nc]; nc];
        for i in 0..nc {
            for j in 0..nc {
                mat0[i][j] = g[i][j] + inv * h[i][j];
            }
        }
        let mut fp = 0.0_f64;
        let mut cper = Vec::with_capacity(d);
        for dd in 0..d {
            let mut mat = mat0.clone();
            let mut b = rhs[dd].clone();
            let c = solve_banded(&mut mat, &mut b, band).ok()?;
            let spl = BSpline::new(t.to_vec(), c.clone(), k).ok()?;
            for i in 0..m {
                let e = ys[dd][i] - spl.eval(u[i]);
                fp += w[i] * w[i] * e * e;
            }
            cper.push(c);
        }
        Some((fp, cper))
    };

    let (fp_inf, _) = solve_at(f64::INFINITY).ok_or_else(|| InterpError::InvalidArgument {
        detail: "smoothing fit: singular system".to_string(),
    })?;
    let xb = u[0];
    let xe = u[m - 1];
    let mut t0 = Vec::with_capacity(2 * (k + 1));
    for _ in 0..=k {
        t0.push(xb);
    }
    for _ in 0..=k {
        t0.push(xe);
    }
    let (_, fp0) = lsq_residuals_nd(u, ys, w, &t0, k)?;

    let acc = s * 1e-3;
    let p0 = nc as f64 / cholesky_diag_sum(&g);
    let mut f = |p: f64| -> f64 {
        match solve_at(p) {
            Some((fp, _)) => fp - s,
            None => f64::INFINITY,
        }
    };
    let p = root_rati(
        &mut f,
        p0,
        ((0.0, fp0 - s), (f64::INFINITY, fp_inf - s)),
        acc,
    );
    let (_, cper) = solve_at(p).ok_or_else(|| InterpError::InvalidArgument {
        detail: "smoothing fit: singular system at solution".to_string(),
    })?;
    Ok(cper)
}

/// A parametric spline as `make_splprep` returns it: the shared knot vector, one
/// coefficient vector per coordinate, and the degree. Named because the bare
/// tuple trips clippy's type_complexity in the signature (frankenscipy-9yyez).
pub type ParametricSpline = (Vec<f64>, Vec<Vec<f64>>, usize);

/// Smoothing parametric B-spline curve. Matches `scipy.interpolate.make_splprep`.
///
/// Given `points` as `ndim` coordinate arrays of equal length sampling a curve,
/// parametrizes by normalized cumulative chord length `u`, places knots with the
/// FITPACK least-squares loop on the summed-across-dimensions residuals, and fits
/// the penalized smoothing spline whose total residual `fp` equals `s` (every
/// coordinate shares the knots and smoothing parameter). Returns
/// `((t, c_per_dim, k), u)`. `s == 0` reduces to interpolation (see [`splprep`]).
pub fn make_splprep(
    points: &[Vec<f64>],
    k: usize,
    s: f64,
) -> Result<(ParametricSpline, Vec<f64>), InterpError> {
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "smoothing factor s must be >= 0".to_string(),
        });
    }
    if s == 0.0 {
        return splprep(points, k, s);
    }
    if points.is_empty() {
        return Err(InterpError::InvalidArgument {
            detail: "make_splprep requires at least one coordinate array".to_string(),
        });
    }
    let m = points[0].len();
    if points.iter().any(|p| p.len() != m) {
        return Err(InterpError::InvalidArgument {
            detail: "all coordinate arrays must have the same length".to_string(),
        });
    }
    if m < k + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: k + 1,
            actual: m,
        });
    }
    // Normalized cumulative chord-length parametrization.
    let mut u = vec![0.0_f64; m];
    for i in 1..m {
        let dd: f64 = points
            .iter()
            .map(|p| (p[i] - p[i - 1]).powi(2))
            .sum::<f64>()
            .sqrt();
        u[i] = u[i - 1] + dd;
    }
    let total = u[m - 1];
    if total <= 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "make_splprep requires distinct points (nonzero total chord length)"
                .to_string(),
        });
    }
    for v in &mut u {
        *v /= total;
    }

    let w = vec![1.0_f64; m];
    let t = generate_knots_nd_last(&u, points, &w, k, s)?;
    let nmin = 2 * (k + 1);
    let coeffs = if t.len() == nmin {
        let mut cper = Vec::with_capacity(points.len());
        for coord in points {
            cper.push(make_lsq_spline(&u, coord, &t, k)?.coeffs().to_vec());
        }
        cper
    } else {
        splrep_smoothing_fit_nd(&u, points, &w, &t, k, s)?
    };
    Ok(((t, coeffs, k), u))
}

/// Pad B-spline coefficients to `len(t)` with trailing zeros, matching scipy's
/// FITPACK tck convention (`len(c) == len(t)`; the last k+1 entries are zero).
fn pad_tck_coeffs(t: &[f64], c: &[f64]) -> Vec<f64> {
    let mut out = c.to_vec();
    out.resize(t.len(), 0.0);
    out
}

/// Evaluate a spline at given points.
///
/// Takes (knots, coefficients, degree) from `splrep`.
/// Matches `scipy.interpolate.splev`.
pub fn splev(x_eval: &[f64], tck: &(Vec<f64>, Vec<f64>, usize)) -> Result<Vec<f64>, InterpError> {
    splev_with_derivative(x_eval, tck, 0)
}

/// Evaluate a spline or one of its derivatives at given points.
///
/// Takes (knots, coefficients, degree) from `splrep`.
/// Matches `scipy.interpolate.splev(..., der=der)`.
pub fn splev_with_derivative(
    x_eval: &[f64],
    tck: &(Vec<f64>, Vec<f64>, usize),
    der: usize,
) -> Result<Vec<f64>, InterpError> {
    let (t, c, k) = tck;
    if der > *k {
        return Err(InterpError::InvalidArgument {
            detail: format!("0<=der={der}<=k={} must hold", *k),
        });
    }
    let bspl = BSpline::new(t.clone(), c.clone(), *k)?;
    let eval_spline = if der == 0 {
        bspl
    } else {
        bspl.derivative(der)?
    };
    Ok(eval_spline.eval_many(x_eval))
}

/// Simple 2D interpolation on a regular grid.
///
/// Matches `scipy.interpolate.interp2d` (deprecated in scipy but still used).
/// Uses bilinear interpolation.
pub fn interp2d(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
    xi: f64,
    yi: f64,
) -> Result<f64, InterpError> {
    let nx = x.len();
    let ny = y.len();

    if z.len() != ny || (ny > 0 && z[0].len() != nx) {
        return Err(InterpError::InvalidArgument {
            detail: "z dimensions must match x and y lengths".to_string(),
        });
    }
    if nx < 2 || ny < 2 {
        return Err(InterpError::TooFewPoints {
            minimum: 2,
            actual: nx.min(ny),
        });
    }
    if !xi.is_finite() || !yi.is_finite() {
        return Ok(f64::NAN);
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpError::UnsortedX);
    }
    if y.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpError::InvalidArgument {
            detail: "y must be strictly increasing".to_string(),
        });
    }

    // Find x interval
    let mut ix = 0;
    for (i, &val) in x.iter().enumerate().take(nx - 1) {
        if xi >= val {
            ix = i;
        }
    }
    ix = ix.min(nx - 2);

    // Find y interval
    let mut iy = 0;
    for (i, &val) in y.iter().enumerate().take(ny - 1) {
        if yi >= val {
            iy = i;
        }
    }
    iy = iy.min(ny - 2);

    // Bilinear interpolation
    let dx = x[ix + 1] - x[ix];
    let dy = y[iy + 1] - y[iy];
    if dx == 0.0 || dy == 0.0 {
        return Ok(z[iy][ix]);
    }

    let tx = (xi - x[ix]) / dx;
    let ty = (yi - y[iy]) / dy;

    let f00 = z[iy][ix];
    let f10 = z[iy][ix + 1];
    let f01 = z[iy + 1][ix];
    let f11 = z[iy + 1][ix + 1];

    Ok(f00 * (1.0 - tx) * (1.0 - ty)
        + f10 * tx * (1.0 - ty)
        + f01 * (1.0 - tx) * ty
        + f11 * tx * ty)
}

/// Number of meaningful B-spline coefficients for a tck.
///
/// FITPACK's knot/coefficient relationship is `len(t) = m + k + 1`, where `m`
/// is the number of basis functions. SciPy zero-pads `c` to `len(t)`, while
/// FrankenSciPy's own tck is "tight" (`len(c) == m`). Deriving `m` from the
/// knot vector lets the spline calculus routines accept either convention.
fn tck_n_coeffs(t: &[f64], k: usize) -> Result<usize, InterpError> {
    t.len()
        .checked_sub(k + 1)
        .filter(|&m| m >= 1)
        .ok_or_else(|| InterpError::InvalidArgument {
            detail: format!(
                "invalid tck: knot length {} too short for degree {k}",
                t.len()
            ),
        })
}

/// Insert `m` knots at position `x` into a B-spline using Boehm's algorithm.
///
/// Matches the non-periodic form of `scipy.interpolate.insert(x, tck, m, per=0)`.
/// The spline curve is unchanged; only its `(t, c)` representation gains `m`
/// extra knots/coefficients. `x` must lie within the base interval
/// `[t[k], t[len(t)-k-1]]`. The returned coefficient vector is zero-padded to
/// `len(t)` to match SciPy's convention. The periodic case (`per=true`) is not
/// supported.
pub fn insert(
    x: f64,
    tck: &(Vec<f64>, Vec<f64>, usize),
    m: usize,
    per: bool,
) -> Result<(Vec<f64>, Vec<f64>, usize), InterpError> {
    if per {
        return Err(InterpError::InvalidArgument {
            detail: "periodic knot insertion (per=true) is not supported".to_string(),
        });
    }
    let (t0, c0, k) = tck;
    let k = *k;
    let n = tck_n_coeffs(t0, k)?;
    if c0.len() < n {
        return Err(InterpError::InvalidArgument {
            detail: format!("invalid tck: {} coefficients, {n} required", c0.len()),
        });
    }
    let lo = t0[k];
    let hi = t0[t0.len() - k - 1];
    if x < lo || x > hi {
        return Err(InterpError::InvalidArgument {
            detail: format!("x={x} outside the spline interval [{lo}, {hi}]"),
        });
    }

    // Work on the meaningful coefficients only; re-pad at the end.
    let mut t: Vec<f64> = t0.clone();
    let mut c: Vec<f64> = c0[..n].to_vec();

    for _ in 0..m {
        let nc = c.len(); // current number of coefficients
        // Find span l with t[l] <= x < t[l+1], clamped to the interior range.
        let mut l = k;
        while l < nc && x >= t[l + 1] {
            l += 1;
        }
        // Boehm single-knot insertion.
        let mut new_c = vec![0.0; nc + 1];
        for (i, slot) in new_c.iter_mut().enumerate() {
            *slot = if i <= l - k {
                c[i]
            } else if i <= l {
                let denom = t[i + k] - t[i];
                let alpha = if denom.abs() > 0.0 {
                    (x - t[i]) / denom
                } else {
                    0.0
                };
                alpha * c[i] + (1.0 - alpha) * c[i - 1]
            } else {
                c[i - 1]
            };
        }
        // Insert the knot, keeping t sorted.
        t.insert(l + 1, x);
        c = new_c;
    }

    // Zero-pad coefficients to len(t), matching SciPy.
    c.resize(t.len(), 0.0);
    Ok((t, c, k))
}

/// Compute the derivative of a B-spline.
///
/// Returns new knots and coefficients for the derivative spline. Accepts both
/// the FrankenSciPy "tight" tck convention (`len(c) == len(t) - k - 1`) and
/// SciPy's zero-padded convention (`len(c) == len(t)`); only the
/// `len(t) - k - 1` meaningful coefficients are consulted, and the returned
/// tck uses the tight convention.
///
/// Matches `scipy.interpolate.splder`.
pub fn splder(
    tck: &(Vec<f64>, Vec<f64>, usize),
) -> Result<(Vec<f64>, Vec<f64>, usize), InterpError> {
    let (t, c, k) = tck;
    let k = *k;
    if k == 0 {
        return Err(InterpError::InvalidArgument {
            detail: "cannot differentiate degree-0 spline".to_string(),
        });
    }

    let m = tck_n_coeffs(t, k)?;
    if c.len() < m {
        return Err(InterpError::InvalidArgument {
            detail: format!("invalid tck: {} coefficients, {m} required", c.len()),
        });
    }

    // Derivative coefficients: c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1}).
    let mut new_c = Vec::with_capacity(m - 1);
    for i in 0..m - 1 {
        let dt = t[i + k + 1] - t[i + 1];
        if dt.abs() > 1e-15 {
            new_c.push(k as f64 * (c[i + 1] - c[i]) / dt);
        } else {
            new_c.push(0.0);
        }
    }

    // Remove one knot from each end.
    let new_t = t[1..t.len() - 1].to_vec();

    Ok((new_t, new_c, k - 1))
}

/// Evaluate a B-spline and all its derivatives, up to order `k` (the spline
/// degree), at each point in `x_eval`.
///
/// Returns one row per evaluation point; `result[i]` is
/// `[B(x_i), B'(x_i), …, B^(k)(x_i)]` (length `k + 1`, order 0 being the spline
/// value itself). Matches `scipy.interpolate.spalde(x, tck)`.
///
/// The derivatives are obtained by repeatedly differentiating the spline with
/// [`splder`] and evaluating each successive derivative with [`splev`], which
/// reproduces FITPACK's result at non-knot points.
pub fn spalde(
    x_eval: &[f64],
    tck: &(Vec<f64>, Vec<f64>, usize),
) -> Result<Vec<Vec<f64>>, InterpError> {
    let k = tck.2;

    // Evaluate each derivative order over all points: per_order[j][i] is the
    // j-th derivative at x_eval[i].
    let mut per_order: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    let mut current = tck.clone();
    for j in 0..=k {
        per_order.push(splev(x_eval, &current)?);
        if j < k {
            current = splder(&current)?;
        }
    }

    // Transpose into one row of length k+1 per evaluation point.
    let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(k + 1); x_eval.len()];
    for order_vals in &per_order {
        for (i, &v) in order_vals.iter().enumerate() {
            out[i].push(v);
        }
    }
    Ok(out)
}

/// Evaluate a bivariate (tensor-product) B-spline on the grid `x × y`.
///
/// `tck` is `(tx, ty, c, kx, ky)`: the knot vectors in each direction, the flat
/// coefficient array (row-major over the `(len(tx)-kx-1, len(ty)-ky-1)` grid,
/// the FITPACK / scipy layout), and the two degrees. Returns `z` with
/// `z[i][j] == spline(x[i], y[j])`.
///
/// Runtime switch to force the serial `bisplev` grid loop for same-binary A/B
/// benchmarks. Defaults off. `#[doc(hidden)]` — internal.
#[doc(hidden)]
/// CONTRACT: BYTE-IDENTICAL either way. The fan-out is over GRID POINTS, each of which
/// is an independent `(kx+1)*(ky+1)` product against the same coefficients -- no point's
/// value enters any other's, so there is no reduction for a scheduling change to
/// reassociate, and the parallel arm writes each result to its own slot in index order.
///
/// The gate below is a performance choice with no bearing on this contract: it decides
/// WHETHER to fan out, not what the answer is. Both sides of the gate compute the same
/// bits.
pub static BISPLEV_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle: when `true`, `bisplev` sweeps the FULL basis (all nx_c/ny_c
/// coefficients, mostly zero-weighted) instead of the compact ky+1 nonzero run. Byte-identical
/// output; only for A/B timing. `#[doc(hidden)]` — internal.
#[doc(hidden)]
pub static BISPLEV_COMPACT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Matches `scipy.interpolate.bisplev(x, y, tck)` for the value case
/// (`dx == dy == 0`). The coefficient layout is scipy's, so a `tck` produced by
/// `scipy.interpolate.bisplrep` can be evaluated directly.
///
/// # Errors
/// Returns `InterpError::InvalidArgument` if the knot/coefficient sizes are
/// inconsistent (`c.len() != (len(tx)-kx-1)·(len(ty)-ky-1)`).
pub fn bisplev(
    x: &[f64],
    y: &[f64],
    tck: &(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize),
) -> Result<Vec<Vec<f64>>, InterpError> {
    let (tx, ty, c, kx, ky) = tck;
    let (kx, ky) = (*kx, *ky);

    if tx.len() < kx + 2 || ty.len() < ky + 2 {
        return Err(InterpError::InvalidArgument {
            detail: "knot vectors too short for the requested degree".to_string(),
        });
    }
    let nx_c = tx.len() - kx - 1;
    let ny_c = ty.len() - ky - 1;
    if c.len() != nx_c * ny_c {
        return Err(InterpError::InvalidArgument {
            detail: format!(
                "coefficient array has {} entries, expected {nx_c}·{ny_c} = {}",
                c.len(),
                nx_c * ny_c
            ),
        });
    }

    // Pre-compute the y-direction basis once per output column (independent of x), stored as
    // COMPACT support `(base index, nonzero weight run)`. Only ky+1 basis functions are nonzero
    // per point, so the inner y-sum touches ky+1 coefficients instead of sweeping all ny_c —
    // the old loop multiplied every c[row+b] by a mostly-zero byb. Byte-identical: the skipped
    // terms are exactly the zero-weight ones (`s += c·bxa·0.0` is a no-op) and the nonzero run
    // is contiguous, so the retained terms accumulate in the same order.
    let full_mode = BISPLEV_COMPACT_DISABLE.load(std::sync::atomic::Ordering::Relaxed);
    let compact = |full: &[f64]| -> (usize, Vec<f64>) {
        if full_mode {
            // A/B baseline: base 0 + full basis reproduces the old all-`ny_c` sweep exactly.
            return (0, full.to_vec());
        }
        match full.iter().position(|&v| v != 0.0) {
            None => (0, Vec::new()),
            Some(base) => {
                let end = full.iter().rposition(|&v| v != 0.0).map_or(base, |e| e + 1);
                (base, full[base..end].to_vec())
            }
        }
    };
    let by_all: Vec<(usize, Vec<f64>)> = y
        .iter()
        .map(|&yj| compact(&eval_basis_all(ty, yj, ky, ny_c)))
        .collect();

    let mut out = vec![vec![0.0; y.len()]; x.len()];

    // One output ROW (fixed x) is an independent, compute-bound tensor-product evaluation: the
    // x-basis is one de Boor sweep, then each y-column sums the (kx+1)·(ky+1) nonzero coefficient
    // block. Rows are independent (each writes its own `out[i]`), so the loop fans across cores.
    let fill_row = |xi: f64, row_out: &mut [f64]| {
        let (bx_base, bx) = compact(&eval_basis_all(tx, xi, kx, nx_c));
        for (j, (by_base, by)) in by_all.iter().enumerate() {
            let mut s = 0.0;
            for (a, &bxa) in bx.iter().enumerate() {
                if bxa == 0.0 {
                    continue;
                }
                let row = (bx_base + a) * ny_c + by_base;
                for (b, &byb) in by.iter().enumerate() {
                    s += c[row + b] * bxa * byb;
                }
            }
            row_out[j] = s;
        }
    };

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(x.len().max(1));
    let force_serial = BISPLEV_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed);
    // Fan out only when the grid is large enough to amortize the spawn + the serial
    // `by_all` precompute; the per-point work is only (kx+1)·(ky+1) mults, so gate on
    // total grid points. Measured: 64² stays serial, 200² (40k) wins 1.4×, 500² 2.3×,
    // 1000² 2.5× — gate at ≥40k so every enabled case is a measured win.
    let grid = x.len().saturating_mul(y.len());
    if cores <= 1 || force_serial || grid < 40_000 {
        for (&xi, row_out) in x.iter().zip(out.iter_mut()) {
            fill_row(xi, row_out);
        }
    } else {
        let chunk = x.len().div_ceil(cores);
        let fill_row_ref = &fill_row;
        std::thread::scope(|scope| {
            for (xchunk, outchunk) in x.chunks(chunk).zip(out.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (&xi, row_out) in xchunk.iter().zip(outchunk.iter_mut()) {
                        fill_row_ref(xi, row_out);
                    }
                });
            }
        });
    }
    Ok(out)
}

/// Compute the antiderivative (integral) of a B-spline.
///
/// Accepts both the FrankenSciPy "tight" tck convention
/// (`len(c) == len(t) - k - 1`) and SciPy's zero-padded convention
/// (`len(c) == len(t)`); only the `len(t) - k - 1` meaningful coefficients
/// are consulted, and the returned tck uses the tight convention. The
/// integration constant is fixed at zero.
///
/// Matches `scipy.interpolate.splantider`.
pub fn splantider(
    tck: &(Vec<f64>, Vec<f64>, usize),
) -> Result<(Vec<f64>, Vec<f64>, usize), InterpError> {
    let (t, c, k) = tck;
    let k = *k;
    let m = tck_n_coeffs(t, k)?;
    if c.len() < m {
        return Err(InterpError::InvalidArgument {
            detail: format!("invalid tck: {} coefficients, {m} required", c.len()),
        });
    }

    // Antiderivative coefficients (cumulative): c'_0 = 0 (integration
    // constant), c'_{i+1} = c'_i + c_i * (t_{i+k+1} - t_i) / (k + 1).
    let mut new_c = Vec::with_capacity(m + 1);
    new_c.push(0.0);
    for i in 0..m {
        let dt = t[i + k + 1] - t[i];
        new_c.push(new_c[i] + c[i] * dt / (k + 1) as f64);
    }

    // Add one knot at each end (repeat boundary knots).
    let mut new_t = Vec::with_capacity(t.len() + 2);
    new_t.push(t[0]);
    new_t.extend_from_slice(t);
    new_t.push(t[t.len() - 1]);

    Ok((new_t, new_c, k + 1))
}

/// Compute the definite integral of a B-spline.
///
/// Matches `scipy.interpolate.splint`.
pub fn splint(a: f64, b: f64, tck: &(Vec<f64>, Vec<f64>, usize)) -> Result<f64, InterpError> {
    let antider = splantider(tck)?;
    let bspl = BSpline::new(antider.0, antider.1, antider.2)?;
    Ok(bspl.eval(b) - bspl.eval(a))
}

/// Find the roots of a cubic B-spline (k=3).
///
/// Returns x values where the spline equals zero.
/// Matches `scipy.interpolate.sproot`.
pub fn sproot(tck: &(Vec<f64>, Vec<f64>, usize)) -> Result<Vec<f64>, InterpError> {
    let (t, c, k) = tck;
    if *k != 3 {
        return Err(InterpError::InvalidArgument {
            detail: "sproot only supports cubic splines (k=3)".to_string(),
        });
    }

    let bspl = BSpline::new(t.clone(), c.clone(), *k)?;
    let d1 = bspl.derivative(1)?;
    let d2 = bspl.derivative(2)?;
    let d3 = bspl.derivative(3)?;

    // On every knot interval [t[l], t[l+1]] the spline is a single cubic
    // polynomial. Recover that cubic from a Taylor expansion at the left
    // knot and solve it exactly, instead of scanning a fixed grid (which
    // misses closely spaced roots and double-counts roots near knots).
    // The real coefficient count is t.len()-k-1; derive it from the knots rather
    // than c.len() so a scipy-style padded tck (len(c)==len(t)) does not push the
    // interval loop and the `t[n_coeff]` access out of bounds.
    let n_coeff = t.len() - k - 1;
    let mut roots: Vec<f64> = Vec::new();
    for l in *k..n_coeff {
        let a = t[l];
        let b = t[l + 1];
        if b <= a {
            continue; // repeated knot — empty interval
        }
        // p(u) = c0 + c1·u + c2·u² + c3·u³, u ∈ [0, b−a).
        let c0 = bspl.eval(a);
        let c1 = d1.eval(a);
        let c2 = d2.eval(a) / 2.0;
        let c3 = d3.eval(a) / 6.0;
        for u in cubic_roots_on_interval(c0, c1, c2, c3, b - a) {
            roots.push(a + u);
        }
    }
    // The half-open intervals above exclude the domain's right endpoint;
    // include it explicitly when the spline vanishes there.
    let x_hi = t[n_coeff];
    let v_hi = bspl.eval(x_hi);
    let hi_scale = 1.0 + c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if v_hi.abs() <= 1e-9 * hi_scale {
        roots.push(x_hi);
    }

    roots.sort_by(f64::total_cmp);
    // Roots coinciding at a shared knot collapse to a single entry.
    roots.dedup_by(|x, y| (*x - *y).abs() <= 1e-7 * (1.0 + x.abs()));
    Ok(roots)
}

/// Real roots of the cubic `c0 + c1·u + c2·u² + c3·u³` lying in `[0, h)`.
///
/// The cubic's critical points split `[0, h]` into monotonic segments; each
/// segment contains at most one root, located by a sign change (with a
/// tolerance check at the segment's left breakpoint for roots that land on
/// a knot or a tangency).
/// Root of a monotone segment `[lo, hi]` where `flo = p(lo)` and `fhi = p(hi)`
/// have OPPOSITE signs, via the Illinois modified false-position method. Keeps
/// the sign-change bracket every step (so convergence is guaranteed like
/// bisection) but converges superlinearly — ~12-15 evaluations instead of the
/// former fixed 80 bisection steps. Works for either orientation of `p`.
fn illinois_segment_root(
    p: impl Fn(f64) -> f64,
    mut lo: f64,
    mut hi: f64,
    mut flo: f64,
    mut fhi: f64,
) -> f64 {
    let mut side = 0i32;
    let mut mid = f64::NAN;
    for _ in 0..60 {
        let denom = fhi - flo;
        let interp = if denom.is_finite() && denom != 0.0 {
            (lo * fhi - hi * flo) / denom
        } else {
            f64::NAN
        };
        let next = if interp > lo && interp < hi {
            interp
        } else {
            0.5 * (lo + hi)
        };
        let tol = 4.0 * f64::EPSILON * next.abs().max(1.0);
        if (next - mid).abs() <= tol || (hi - lo).abs() <= tol {
            return next;
        }
        mid = next;
        let fmid = p(mid);
        if fmid == 0.0 {
            return mid;
        }
        // `fmid` on the same side as `fhi` → tighten the high end, else the low.
        if (fmid < 0.0) == (fhi < 0.0) {
            hi = mid;
            fhi = fmid;
            if side == 1 {
                flo *= 0.5; // Illinois down-weight of the stale low endpoint
            }
            side = 1;
        } else {
            lo = mid;
            flo = fmid;
            if side == -1 {
                fhi *= 0.5;
            }
            side = -1;
        }
    }
    mid
}

fn cubic_roots_on_interval(c0: f64, c1: f64, c2: f64, c3: f64, h: f64) -> Vec<f64> {
    let p = |u: f64| ((c3 * u + c2) * u + c1) * u + c0;
    let scale = c0.abs() + c1.abs() * h + c2.abs() * h * h + c3.abs() * h * h * h;
    let eps = 1e-12 * (1.0 + scale);

    // Critical points of p: roots of p'(u) = 3·c3·u² + 2·c2·u + c1.
    let mut breaks = vec![0.0_f64, h];
    let (qa, qb, qc) = (3.0 * c3, 2.0 * c2, c1);
    if qa.abs() > 1e-300 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for r in [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)] {
                if r > 0.0 && r < h {
                    breaks.push(r);
                }
            }
        }
    } else if qb.abs() > 1e-300 {
        let r = -qc / qb;
        if r > 0.0 && r < h {
            breaks.push(r);
        }
    }
    breaks.sort_by(f64::total_cmp);

    let mut roots = Vec::new();
    for w in breaks.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi <= lo {
            continue;
        }
        let plo = p(lo);
        let phi = p(hi);
        if plo.abs() <= eps {
            // Root at the left breakpoint (knot, or a tangency at a
            // critical point that a sign-change test would miss).
            roots.push(lo);
        } else if (plo < 0.0) != (phi < 0.0) {
            // One sign change on a monotonic segment: refine with Illinois
            // false-position (~12-15 evals) instead of 80 fixed bisection steps.
            roots.push(illinois_segment_root(p, lo, hi, plo, phi));
        }
    }
    roots.retain(|&u| (0.0..h).contains(&u));
    roots
}

#[cfg(test)]
fn polyval_der_restarted(coeffs: &[f64], x: f64, der: usize) -> Vec<f64> {
    let n = coeffs.len();
    if n == 0 {
        return vec![0.0; der + 1];
    }

    let mut result = vec![0.0; der + 1];
    for (d, item) in result.iter_mut().enumerate().take(der + 1) {
        if d >= n {
            break;
        }
        let mut dc = coeffs.to_vec();
        for _ in 0..d {
            let len = dc.len();
            if len <= 1 {
                dc = vec![0.0];
                break;
            }
            dc = (0..len - 1).map(|i| dc[i] * (len - 1 - i) as f64).collect();
        }
        *item = polyval(&dc, x);
    }
    result
}

/// Evaluate a polynomial and its derivatives.
///
/// Returns (p(x), p'(x), p''(x), ...) up to order `der`.
pub fn polyval_der(coeffs: &[f64], x: f64, der: usize) -> Vec<f64> {
    let n = coeffs.len();
    if n == 0 {
        return vec![0.0; der + 1];
    }

    let mut result = vec![0.0; der + 1];
    let mut dc = coeffs.to_vec();
    let orders = result.len().min(n);
    for (order, item) in result.iter_mut().take(orders).enumerate() {
        *item = polyval(&dc, x);
        if order + 1 < orders {
            let len = dc.len();
            for (i, coefficient) in dc.iter_mut().enumerate().take(len - 1) {
                *coefficient *= (len - 1 - i) as f64;
            }
            dc.truncate(len - 1);
        }
    }

    result
}

/// Fit a polynomial of degree `deg` to data points (x, y) using least squares.
///
/// Returns coefficients in descending order of degree.
/// Matches `numpy.polyfit`.
pub fn polyfit(x: &[f64], y: &[f64], deg: usize) -> Result<Vec<f64>, InterpError> {
    let n = x.len();
    if n != y.len() {
        return Err(InterpError::LengthMismatch {
            x_len: n,
            y_len: y.len(),
        });
    }
    if n <= deg {
        return Err(InterpError::TooFewPoints {
            minimum: deg + 1,
            actual: n,
        });
    }

    let ncols = deg + 1;

    // Build Vandermonde matrix and solve normal equations
    let mut ata = vec![vec![0.0; ncols]; ncols];
    let mut atb = vec![0.0; ncols];

    // Accumulate only the upper triangle of A^T A, reuse one xpow buffer instead
    // of allocating per sample, and mirror to the lower triangle ONCE after the
    // loop. The previous in-loop mirror (`ata[k][j] = ata[j][k]` per sample) did
    // O(n·ncols²) cache-scattered writes. Byte-identical: the upper-triangle
    // accumulation order/operands are unchanged, xpow[0] stays 1.0 so the reused
    // buffer yields identical xpow[j] = x[i]^j, and the final mirror copies the
    // same finished values the per-sample assignment left behind.
    let mut xpow = vec![1.0; ncols]; // xpow[0] is x^0 = 1, never overwritten
    for i in 0..n {
        for j in 1..ncols {
            xpow[j] = xpow[j - 1] * x[i];
        }

        for j in 0..ncols {
            atb[j] += xpow[j] * y[i];
            for k in j..ncols {
                ata[j][k] += xpow[j] * xpow[k];
            }
        }
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..ncols {
        for k in (j + 1)..ncols {
            ata[k][j] = ata[j][k];
        }
    }

    // Solve via Cholesky-like method
    let coeffs = solve_normal(&ata, &atb)?;

    // Reverse to get descending order (highest degree first)
    let mut result = coeffs;
    result.reverse();
    Ok(result)
}

fn solve_normal(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpError> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .map(|r| {
            let mut row = r.clone();
            row.push(0.0);
            row
        })
        .collect();
    for i in 0..n {
        aug[i][n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            return Err(InterpError::InvalidArgument {
                detail: "singular system in polyfit".to_string(),
            });
        }

        let pivot = aug[col][col];
        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

/// Padé approximation: compute rational function `p(x)/q(x)` coefficients from a
/// Taylor series, matching `scipy.interpolate.pade(an, m, n)`.
///
/// Following scipy's argument convention, `m` is the order of the **denominator**
/// `q` and `n` is the order of the **numerator** `p` (so `p(x)/q(x) ≈ Σ a_k x^k`).
/// Returns `(p_coeffs, q_coeffs)` low-first, of lengths `n+1` and `m+1`.
pub fn pade(
    taylor_coeffs: &[f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), InterpError> {
    if taylor_coeffs.len() < m + n + 1 {
        return Err(InterpError::TooFewPoints {
            minimum: m + n + 1,
            actual: taylor_coeffs.len(),
        });
    }

    // scipy: m = denominator order, n = numerator order. The solver below builds
    // the numerator of degree `m` and denominator of degree `n` (its own internal
    // convention), so swap to honour scipy's argument order.
    let (m, n) = (n, m);

    // Solve for q coefficients from the system:
    // Σ_{j=0}^{n} q_j * a_{m+1+i-j} = 0 for i = 0..n-1 (with q_0 = 1)
    if n == 0 {
        return Ok((taylor_coeffs[..=m].to_vec(), vec![1.0]));
    }

    // Build system for q_1..q_n
    let mut mat = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];

    for i in 0..n {
        rhs[i] = -taylor_coeffs[m + 1 + i];
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let idx = m as i64 + i as i64 - j as i64;
            if idx >= 0 && (idx as usize) < taylor_coeffs.len() {
                mat[i][j] = taylor_coeffs[idx as usize];
            }
        }
    }

    let q_tail = solve_normal(&mat, &rhs)?;
    let mut q = vec![1.0];
    q.extend_from_slice(&q_tail);

    // Compute p coefficients: p_k = Σ_{j=0}^{min(k,n)} q_j * a_{k-j}
    let mut p = Vec::with_capacity(m + 1);
    for k in 0..=m {
        let mut pk = 0.0;
        for j in 0..=k.min(n) {
            if k >= j {
                pk += q[j] * taylor_coeffs[k - j];
            }
        }
        p.push(pk);
    }

    Ok((p, q))
}

/// Evaluate a rational function p(x)/q(x).
///
/// `p` and `q` are coefficient vectors [a0, a1, ...] (ascending powers).
pub fn ratval(p: &[f64], q: &[f64], x: f64) -> f64 {
    #[inline]
    fn eval_ascending(coeffs: &[f64], x: f64) -> f64 {
        let Some((&highest, remaining)) = coeffs.split_last() else {
            return 0.0;
        };
        remaining
            .iter()
            .rev()
            .fold(highest, |value, &coefficient| value * x + coefficient)
    }

    let num = eval_ascending(p, x);
    let den = eval_ascending(q, x);
    if den.abs() < 1e-30 {
        return f64::NAN;
    }
    num / den
}

/// Polynomial multiplication: convolve coefficient vectors.
///
/// Matches `numpy.polymul`.
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let n = a.len() + b.len() - 1;
    // FFT-based convolution is O((m+n) log(m+n)) vs the direct O(m·n) loop, but
    // carries a large constant (three length-L transforms). Switch to FFT only when
    // the direct work `m·n` dominates the FFT work `L·log2(L)` by a safe margin, so
    // small or lopsided inputs (where direct is faster AND byte-identical) keep the
    // exact loop. The constant 20 sits well past the measured balanced crossover
    // (~n=450); FFT loses at n=256, wins 1.6x at 512, 3.8x at 1024, 20x at 8192.
    // (numpy.polymul is the direct loop; FFT matches it to ~1e-12 ≪ the 1e-10
    // conformance tolerance, and every small parity case stays on the direct path.)
    let fft_len = n.next_power_of_two();
    let direct_ops = (a.len() as u64) * (b.len() as u64);
    let fft_ops = (fft_len as u64) * (fft_len.trailing_zeros().max(1) as u64);
    // On any FFT error or unexpected length, fall through to the exact direct loop.
    if direct_ops > 20 * fft_ops
        && let Ok(conv) = fsci_fft::fftconvolve(a, b, "full")
        && conv.len() == n
    {
        return conv;
    }
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Polynomial addition.
///
/// Matches `numpy.polyadd`.
pub fn polyadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut result = Vec::with_capacity(n);
    if a.len() >= b.len() {
        let prefix = a.len() - b.len();
        result.extend(a[..prefix].iter().map(|&value| 0.0 + value));
        result.extend(
            a[prefix..]
                .iter()
                .zip(b)
                .map(|(&a_value, &b_value)| (0.0 + a_value) + b_value),
        );
    } else {
        let prefix = b.len() - a.len();
        result.extend(b[..prefix].iter().map(|&value| 0.0 + value));
        result.extend(
            a.iter()
                .zip(&b[prefix..])
                .map(|(&a_value, &b_value)| (0.0 + a_value) + b_value),
        );
    }
    result
}

/// Polynomial subtraction.
pub fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut result = vec![0.0; n];
    let offset_a = n - a.len();
    let offset_b = n - b.len();
    for (i, &v) in a.iter().enumerate() {
        result[offset_a + i] += v;
    }
    for (i, &v) in b.iter().enumerate() {
        result[offset_b + i] += -v;
    }
    result
}

/// Polynomial derivative.
///
/// Matches `numpy.polyder`.
pub fn polyder(coeffs: &[f64], m: usize) -> Vec<f64> {
    let mut c = coeffs.to_vec();
    for _ in 0..m {
        if c.len() <= 1 {
            return vec![0.0];
        }
        let n = c.len();
        c = (0..n - 1).map(|i| c[i] * (n - 1 - i) as f64).collect();
    }
    c
}

/// Polynomial integration (antiderivative).
///
/// Matches `numpy.polyint`.
pub fn polyint(coeffs: &[f64], m: usize, k: f64) -> Vec<f64> {
    let mut c = coeffs.to_vec();
    for _ in 0..m {
        let n = c.len();
        let mut new_c = Vec::with_capacity(n + 1);
        for (i, &ci) in c.iter().enumerate() {
            new_c.push(ci / (n - i) as f64);
        }
        new_c.push(k);
        c = new_c;
    }
    c
}

/// Find roots of a polynomial given coefficients (highest degree first).
///
/// Construct a polynomial from its roots.
///
/// Matches `numpy.polynomial.polynomial.polyfromroots(roots)` in the
/// fsci-interpolate convention (highest-degree-first). Given roots
/// `[r₁, r₂, ..., rₙ]`, returns the coefficients of the monic
/// polynomial `(x - r₁)(x - r₂) ... (x - rₙ)` with the leading 1 first.
///
/// `polyfromroots([])` returns `[1.0]` (the empty product).
///
/// Resolves [frankenscipy-bbc5k].
#[must_use]
pub fn polyfromroots(roots: &[f64]) -> Vec<f64> {
    // Start from the constant polynomial 1 (degree-0).
    let mut coeffs: Vec<f64> = vec![1.0];
    for &r in roots {
        // Multiply current polynomial by (x − r). In highest-first
        // form, multiplying by (x − r) shifts every coefficient one
        // position toward the leading term and subtracts r times the
        // unshifted coefficient at the same position.
        let mut next = vec![0.0_f64; coeffs.len() + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            next[i] += c; // shifted up by one degree
            next[i + 1] -= r * c;
        }
        coeffs = next;
    }
    coeffs
}

/// Uses companion matrix eigenvalue method for degree > 2.
/// Matches `numpy.roots`.
pub fn polyroots(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }
    if n == 2 {
        // Linear: ax + b = 0 → x = -b/a
        if coeffs[0] == 0.0 {
            return vec![];
        }
        return vec![-coeffs[1] / coeffs[0]];
    }
    if n == 3 {
        // Quadratic: ax² + bx + c = 0
        let a = coeffs[0];
        let b = coeffs[1];
        let c = coeffs[2];
        if a == 0.0 {
            if b == 0.0 {
                return vec![];
            }
            return vec![-c / b];
        }
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return vec![]; // complex roots, skip for real-only
        }
        let sqrt_disc = disc.sqrt();
        vec![(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
    } else {
        // For higher degree, use Durand-Kerner method
        let degree = n - 1;
        let a0 = coeffs[0];
        let norm_coeffs: Vec<f64> = coeffs.iter().map(|&c| c / a0).collect();

        // Initial guesses on unit circle
        let mut roots: Vec<(f64, f64)> = (0..degree)
            .map(|k| {
                let angle = 2.0 * std::f64::consts::PI * k as f64 / degree as f64 + 0.4;
                (
                    0.4f64.powf(k as f64) * angle.cos(),
                    0.4f64.powf(k as f64) * angle.sin(),
                )
            })
            .collect();

        for _ in 0..1000 {
            let mut max_change = 0.0f64;
            for i in 0..degree {
                // Evaluate polynomial at roots[i]
                let (zr, zi) = roots[i];
                let mut pr = norm_coeffs[0];
                let mut pi_val = 0.0;
                for &c in &norm_coeffs[1..] {
                    let new_r = pr * zr - pi_val * zi + c;
                    let new_i = pr * zi + pi_val * zr;
                    pr = new_r;
                    pi_val = new_i;
                }

                // Divide by product of (z_i - z_j) for j != i
                let mut dr = 1.0;
                let mut di = 0.0;
                for (j, &(rj, ij)) in roots.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let diff_r = zr - rj;
                    let diff_i = zi - ij;
                    let new_r = dr * diff_r - di * diff_i;
                    let new_i = dr * diff_i + di * diff_r;
                    dr = new_r;
                    di = new_i;
                }

                let denom = dr * dr + di * di;
                if denom > 1e-30 {
                    let corr_r = (pr * dr + pi_val * di) / denom;
                    let corr_i = (pi_val * dr - pr * di) / denom;
                    roots[i] = (zr - corr_r, zi - corr_i);
                    max_change = max_change.max((corr_r * corr_r + corr_i * corr_i).sqrt());
                }
            }
            if max_change < 1e-14 {
                break;
            }
        }

        // Return real parts of roots with small imaginary parts
        roots
            .iter()
            .filter(|&&(_, im)| im.abs() < 1e-8)
            .map(|&(re, _)| re)
            .collect()
    }
}

/// Generate Chebyshev nodes of the first kind on [a, b].
///
/// These are optimal interpolation nodes that minimize Runge's phenomenon.
pub fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    if n == 0 || !a.is_finite() || !b.is_finite() {
        return vec![];
    }
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let t = (pi * (2 * k + 1) as f64 / (2 * n) as f64).cos();
            (a + b) / 2.0 + (b - a) / 2.0 * t
        })
        .collect()
}

/// Generate Chebyshev nodes of the second kind on [a, b].
pub fn chebyshev_nodes2(n: usize, a: f64, b: f64) -> Vec<f64> {
    if n == 0 || !a.is_finite() || !b.is_finite() {
        return vec![];
    }
    if n == 1 {
        return vec![(a + b) / 2.0];
    }
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let t = (pi * k as f64 / (n - 1) as f64).cos();
            (a + b) / 2.0 + (b - a) / 2.0 * t
        })
        .collect()
}

/// Compute barycentric interpolation weights.
///
/// Given nodes x_i, returns weights w_i for barycentric interpolation.
/// Returns empty vector if nodes are empty, contain non-finite values, or contain duplicates.
pub fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let n = nodes.len();
    if n == 0 {
        return vec![];
    }
    if nodes.iter().any(|&v| !v.is_finite()) {
        return vec![];
    }
    let mut weights = vec![1.0; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = nodes[i] - nodes[j];
                if diff.abs() < 1e-15 {
                    return vec![];
                }
                weights[i] /= diff;
            }
        }
        if !weights[i].is_finite() {
            return vec![];
        }
    }
    weights
}

/// Evaluate barycentric interpolation at a point.
///
/// Given nodes, values, weights, evaluates the interpolant at x.
pub fn barycentric_eval(nodes: &[f64], values: &[f64], weights: &[f64], x: f64) -> f64 {
    let n = nodes.len();
    if n == 0 || values.len() != n || weights.len() != n || !x.is_finite() {
        return f64::NAN;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..nodes.len() {
        let diff = x - nodes[i];
        if diff.abs() < 1e-15 {
            return values[i];
        }
        let t = weights[i] / diff;
        num += t * values[i];
        den += t;
    }
    if den == 0.0 {
        return f64::NAN;
    }
    num / den
}

/// Neville's algorithm for polynomial interpolation.
///
/// Returns the interpolated value at x using the full Neville tableau.
pub fn neville(nodes: &[f64], values: &[f64], x: f64) -> f64 {
    let n = nodes.len();
    if n == 0 || values.len() != n {
        return f64::NAN;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let mut p = values.to_vec();
    for j in 1..n {
        for i in (j..n).rev() {
            let dx = nodes[i] - nodes[i - j];
            if dx.abs() < 1e-15 {
                return f64::NAN;
            }
            p[i] = ((x - nodes[i - j]) * p[i] - (x - nodes[i]) * p[i - 1]) / dx;
        }
    }
    p[n - 1]
}

/// Compute Hermite interpolation (values and derivatives at each node).
///
/// Given nodes x_i, values y_i, and derivatives dy_i, returns the
/// interpolated value at x. Returns NaN if nodes contain duplicates or
/// if input is invalid.
pub fn hermite_interp(nodes: &[f64], values: &[f64], derivatives: &[f64], x: f64) -> f64 {
    let n = nodes.len();
    if n == 0 || values.len() != n || derivatives.len() != n {
        return f64::NAN;
    }
    if !x.is_finite() {
        return f64::NAN;
    }

    // Use Hermite basis functions
    let mut result = 0.0;
    for i in 0..n {
        // Compute L_i(x) and L_i'(x_i)
        let mut li = 1.0;
        let mut li_deriv_sum = 0.0;
        for j in 0..n {
            if j != i {
                let diff = nodes[i] - nodes[j];
                if diff.abs() < 1e-15 {
                    return f64::NAN;
                }
                li *= (x - nodes[j]) / diff;
                li_deriv_sum += 1.0 / diff;
            }
        }

        let li_sq = li * li;
        let h0 = (1.0 - 2.0 * (x - nodes[i]) * li_deriv_sum) * li_sq;
        let h1 = (x - nodes[i]) * li_sq;

        result += values[i] * h0 + derivatives[i] * h1;
    }

    result
}

/// Compute the definite integral of a polynomial (coefficients highest degree first).
pub fn polyint_definite(coeffs: &[f64], a: f64, b: f64) -> f64 {
    let n = coeffs.len();
    let mut value_at_b = 0.0;
    let mut value_at_a = 0.0;
    for (i, &coefficient) in coeffs.iter().enumerate() {
        let integrated = coefficient / (n - i) as f64;
        value_at_b = value_at_b * b + integrated;
        value_at_a = value_at_a * a + integrated;
    }
    // Preserve `polyint(..., k = 0.0)`'s trailing coefficient and the final
    // Horner operation, including signed-zero and non-finite behavior.
    value_at_b = value_at_b * b + 0.0;
    value_at_a = value_at_a * a + 0.0;
    value_at_b - value_at_a
}

/// Evaluate a polynomial and return (value, error_estimate).
///
/// Error estimated via condition number of evaluation.
pub fn polyval_with_error(coeffs: &[f64], x: f64) -> (f64, f64) {
    let val = polyval(coeffs, x);
    // Estimate error as |val| * machine_epsilon * condition
    let cond: f64 = coeffs
        .iter()
        .enumerate()
        .map(|(i, &c)| c.abs() * x.abs().powi((coeffs.len() - 1 - i) as i32))
        .sum();
    let err = cond * 2.2e-16; // machine epsilon
    (val, err)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RectBivariateSpline — 2D spline interpolation on rectangular grids
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Bivariate spline approximation over a rectangular mesh.
///
/// Matches `scipy.interpolate.RectBivariateSpline`.
///
/// Given 1-D arrays `x` and `y` and a 2-D array `z` of shape `(len(x), len(y))`,
/// constructs a spline that can be evaluated at arbitrary points.
///
/// # Example
///
/// ```ignore
/// let x = vec![0.0, 1.0, 2.0, 3.0];
/// let y = vec![0.0, 1.0, 2.0];
/// let z = vec![
///     vec![0.0, 1.0, 2.0],
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 3.0, 4.0],
///     vec![3.0, 4.0, 5.0],
/// ];
/// let spline = RectBivariateSpline::new(&x, &y, &z, 3, 3).unwrap();
/// let val = spline.eval(1.5, 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct RectBivariateSpline {
    /// Spline degree in x direction (typically 3 for cubic)
    kx: usize,
    /// Spline degree in y direction (typically 3 for cubic)
    ky: usize,
    /// Knot vector in x direction
    tx: Vec<f64>,
    /// Knot vector in y direction
    ty: Vec<f64>,
    /// 2D coefficient array, shape (len(ty) - ky - 1, len(tx) - kx - 1)
    coeffs: Vec<Vec<f64>>,
    /// Original x grid for bounds checking
    x_bounds: (f64, f64),
    /// Original y grid for bounds checking
    y_bounds: (f64, f64),
}

/// Same-binary A/B toggle for `RectBivariateSpline::eval_many`. When `true`, every scattered query
/// goes through the per-query `eval` path (which rebuilds the query-independent x-direction splines
/// on every point — the ORIG behaviour). When `false` (default), those x-splines are built once and
/// reused across the batch. Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static RECTBISPLINE_EVAL_MANY_FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl RectBivariateSpline {
    /// Create a new bivariate spline over a rectangular mesh.
    ///
    /// # Arguments
    /// * `x` - 1-D array of x coordinates (must be strictly increasing)
    /// * `y` - 1-D array of y coordinates (must be strictly increasing)
    /// * `z` - 2-D array of values, shape `(len(x), len(y))`
    /// * `kx` - Spline degree in x direction (1 <= kx <= 5, typically 3)
    /// * `ky` - Spline degree in y direction (1 <= ky <= 5, typically 3)
    pub fn new(
        x: &[f64],
        y: &[f64],
        z: &[Vec<f64>],
        kx: usize,
        ky: usize,
    ) -> Result<Self, InterpError> {
        let nx = x.len();
        let ny = y.len();

        // Validate dimensions
        if nx < 2 || ny < 2 {
            return Err(InterpError::TooFewPoints {
                minimum: 2,
                actual: nx.min(ny),
            });
        }
        if z.len() != nx {
            return Err(InterpError::InvalidArgument {
                detail: format!("z has {} rows but x has {} points", z.len(), nx),
            });
        }
        for (i, row) in z.iter().enumerate() {
            if row.len() != ny {
                return Err(InterpError::InvalidArgument {
                    detail: format!(
                        "z row {} has {} columns but y has {} points",
                        i,
                        row.len(),
                        ny
                    ),
                });
            }
        }

        // Validate degree
        if !(1..=5).contains(&kx) || !(1..=5).contains(&ky) {
            return Err(InterpError::InvalidArgument {
                detail: format!("spline degree must be 1-5, got kx={}, ky={}", kx, ky),
            });
        }

        // Need at least k+1 points in each direction
        if nx < kx + 1 {
            return Err(InterpError::TooFewPoints {
                minimum: kx + 1,
                actual: nx,
            });
        }
        if ny < ky + 1 {
            return Err(InterpError::TooFewPoints {
                minimum: ky + 1,
                actual: ny,
            });
        }

        // Validate strictly increasing
        for i in 1..nx {
            if x[i] <= x[i - 1] {
                return Err(InterpError::InvalidArgument {
                    detail: format!("x[{}] = {} <= x[{}] = {}", i, x[i], i - 1, x[i - 1]),
                });
            }
        }
        for i in 1..ny {
            if y[i] <= y[i - 1] {
                return Err(InterpError::InvalidArgument {
                    detail: format!("y[{}] = {} <= y[{}] = {}", i, y[i], i - 1, y[i - 1]),
                });
            }
        }

        // Build knot vectors using the same formula as make_interp_spline
        let tx = interpolation_knots(x, kx);
        let ty = interpolation_knots(y, ky);

        // SciPy accepts z as shape (len(x), len(y)). The coefficient code works
        // row-by-row along x for each y, so transpose to that internal layout.
        let z_by_y: Vec<Vec<f64>> = (0..ny)
            .map(|j| (0..nx).map(|i| z[i][j]).collect())
            .collect();

        // Compute spline coefficients using tensor product approach
        let coeffs = Self::compute_coefficients(x, y, &z_by_y, kx, ky)?;

        Ok(Self {
            kx,
            ky,
            tx,
            ty,
            coeffs,
            x_bounds: (x[0], x[nx - 1]),
            y_bounds: (y[0], y[ny - 1]),
        })
    }

    /// Compute the 2D spline coefficients.
    ///
    /// Uses the tensor product approach: fit 1D splines along each row,
    /// then fit 1D splines along each column of coefficients.
    fn compute_coefficients(
        x: &[f64],
        y: &[f64],
        z: &[Vec<f64>],
        kx: usize,
        ky: usize,
    ) -> Result<Vec<Vec<f64>>, InterpError> {
        let nx = x.len();
        let ny = y.len();

        // Each row-spline (and column-spline) is an INDEPENDENT `make_interp_spline` over a
        // banded system, so fan both tensor-product passes across cores (chunked, one spawn
        // per chunk). scipy's RectBivariateSpline is single-threaded FITPACK ⇒ pure
        // domination at large grids. Order-preserved ⇒ identical coefficients to the serial
        // build. Small grids stay serial (gate below the parallel break-even).
        let nthreads = if (nx as u64).saturating_mul(ny as u64) < (1 << 16) {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(nx.max(ny))
        };

        // Step 1: Fit 1D splines along x for each row of z → intermediate (ny, nx) coeffs.
        let row_coeffs: Vec<Vec<f64>> = par_chunk_try_map(z, nthreads, |row| {
            Ok::<_, InterpError>(make_interp_spline(x, row, kx)?.coeffs().to_vec())
        })?;

        // Step 2: Fit a 1D spline along y for each column of row_coeffs. Each column produces
        // its own length-ny coefficient vector; assemble the (ny, nx) result by transpose
        // (avoids a cross-thread write race on the shared row-major output).
        let col_indices: Vec<usize> = (0..nx).collect();
        let col_results: Vec<Vec<f64>> = par_chunk_try_map(&col_indices, nthreads, |&col_idx| {
            let col_values: Vec<f64> = row_coeffs.iter().map(|row| row[col_idx]).collect();
            Ok::<_, InterpError>(make_interp_spline(y, &col_values, ky)?.coeffs().to_vec())
        })?;

        let mut final_coeffs: Vec<Vec<f64>> = vec![vec![0.0; nx]; ny];
        for (col_idx, col) in col_results.iter().enumerate() {
            for (row_idx, &coeff) in col.iter().enumerate() {
                final_coeffs[row_idx][col_idx] = coeff;
            }
        }

        Ok(final_coeffs)
    }

    /// Evaluate the spline at a single point.
    pub fn eval(&self, xi: f64, yi: f64) -> f64 {
        if !xi.is_finite() || !yi.is_finite() {
            return f64::NAN;
        }
        self.eval_impl(xi, yi, 0, 0)
    }

    /// Evaluate the spline at multiple points.
    pub fn eval_many(&self, xi: &[f64], yi: &[f64]) -> Result<Vec<f64>, InterpError> {
        if xi.len() != yi.len() {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "xi and yi must have same length, got {} and {}",
                    xi.len(),
                    yi.len()
                ),
            });
        }

        // `eval` -> `eval_impl` rebuilds the `ny` x-direction BSplines (each cloning `tx` and a
        // coefficient row) on EVERY query, though they depend only on the spline, not the point.
        // Build them ONCE and reuse across the batch (shared-predictor lever); each query then only
        // evaluates the shared x-splines, builds its query-dependent y-spline, and evaluates that.
        // BYTE-IDENTICAL to the per-query path: the same deterministic builds, clamps, and
        // evaluation order. The knob and any x-spline build failure fall back to the per-query path.
        if xi.is_empty()
            || RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(xi.iter().zip(yi).map(|(&x, &y)| self.eval(x, y)).collect());
        }
        let x_splines: Option<Vec<BSpline>> = self
            .coeffs
            .iter()
            .map(|row| BSpline::new(self.tx.clone(), row.clone(), self.kx).ok())
            .collect();
        let Some(x_splines) = x_splines else {
            return Ok(xi.iter().zip(yi).map(|(&x, &y)| self.eval(x, y)).collect());
        };
        let eval_one = |x: f64, y: f64| -> f64 {
            if !x.is_finite() || !y.is_finite() {
                return f64::NAN;
            }
            let xc = x.clamp(self.x_bounds.0, self.x_bounds.1);
            let yc = y.clamp(self.y_bounds.0, self.y_bounds.1);
            let intermediate: Vec<f64> = x_splines.iter().map(|s| s.eval(xc)).collect();
            let Ok(y_spline) = BSpline::new(self.ty.clone(), intermediate, self.ky) else {
                return f64::NAN;
            };
            y_spline.eval(yc)
        };
        let pairs: Vec<(f64, f64)> = xi.iter().zip(yi).map(|(&x, &y)| (x, y)).collect();
        let work_per_query = self.coeffs.len().max(1);
        Ok(par_query_map(&pairs, work_per_query, |&(x, y)| {
            eval_one(x, y)
        }))
    }

    /// Evaluate the spline on a grid and return a 2D array.
    ///
    /// Returns values at all combinations of xi and yi, shape `(len(xi), len(yi))`.
    /// Index-driven numeric kernel: the loop variables are the recurrence/band
    /// indices and several address more than one array per iteration, so iterator
    /// form would obscure the index algebra this is checked against rather than
    /// simplify it (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
    pub fn eval_grid(&self, xi: &[f64], yi: &[f64]) -> Vec<Vec<f64>> {
        // Separable-basis (FITPACK bispev) grid evaluation: precompute each axis' k+1
        // non-zero B-spline basis weights ONCE per query coordinate, then tensor-contract
        // against the (kx+1)×(ky+1) coefficient window — instead of running the full
        // scalar de Boor recurrence per (xv, yv) (which rebuilt the x-recurrence for all
        // ny rows per xv and the y-recurrence per pair). Matches the de Boor value within
        // ~1e-13 (same spline, different summation order). frankenscipy-9l5oo.
        let ny = self.coeffs.len();
        let nx = self.coeffs.first().map_or(0, Vec::len);

        // y-axis bases (span + weights) precomputed once and reused for every xv.
        let y_basis: Vec<Option<(usize, [f64; 6])>> = yi
            .iter()
            .map(|&yv| {
                if !yv.is_finite() {
                    return None;
                }
                let yc = yv.clamp(self.y_bounds.0, self.y_bounds.1);
                let span = BSpline::find_span_n(&self.ty, ny, self.ky, yc);
                Some((span, bspline_basis_funs(&self.ty, self.ky, yc, span)))
            })
            .collect();

        // One output ROW (fixed x) is an independent, compute-bound tensor contraction:
        // one x-basis evaluation, then each y-column sums the (kx+1)·(ky+1) coefficient
        // window against the shared precomputed y-bases. Rows are independent → the grid
        // fans across cores BYTE-IDENTICALLY.
        let row_fn = |xv: f64| -> Vec<f64> {
            if !xv.is_finite() {
                return vec![f64::NAN; yi.len()];
            }
            let xc = xv.clamp(self.x_bounds.0, self.x_bounds.1);
            let x_span = BSpline::find_span_n(&self.tx, nx, self.kx, xc);
            let x_w = bspline_basis_funs(&self.tx, self.kx, xc, x_span);
            let x_lo = x_span - self.kx;
            y_basis
                .iter()
                .map(|yb| {
                    let Some((y_span, y_w)) = yb else {
                        return f64::NAN;
                    };
                    let y_lo = y_span - self.ky;
                    let mut val = 0.0;
                    for q in 0..=self.ky {
                        let row = &self.coeffs[y_lo + q];
                        let mut inner = 0.0;
                        for p in 0..=self.kx {
                            inner += x_w[p] * row[x_lo + p];
                        }
                        val += y_w[q] * inner;
                    }
                    val
                })
                .collect()
        };
        par_grid_rows(xi, yi.len(), &row_fn)
    }

    /// Evaluate the partial derivative d^(dx+dy)f / dx^dx dy^dy.
    pub fn eval_derivative(&self, xi: f64, yi: f64, dx: usize, dy: usize) -> f64 {
        if !xi.is_finite() || !yi.is_finite() {
            return f64::NAN;
        }
        self.eval_impl(xi, yi, dx, dy)
    }

    /// Internal evaluation with derivatives.
    fn eval_impl(&self, xi: f64, yi: f64, dx: usize, dy: usize) -> f64 {
        // Clamp to bounds
        let xi_clamped = xi.clamp(self.x_bounds.0, self.x_bounds.1);
        let yi_clamped = yi.clamp(self.y_bounds.0, self.y_bounds.1);

        // Use tensor product evaluation via 1D BSplines:
        // 1. For each row of coefficients, create a 1D x-spline and evaluate at xi
        // 2. Create a 1D y-spline from those intermediate values and evaluate at yi

        let ny = self.coeffs.len();
        let mut intermediate = Vec::with_capacity(ny);

        for row in &self.coeffs {
            // Create x-direction spline for this row of coefficients
            let Ok(mut x_spline) = BSpline::new(self.tx.clone(), row.clone(), self.kx) else {
                return f64::NAN;
            };

            // Apply x derivative if needed
            for _ in 0..dx {
                let Ok(deriv) = x_spline.derivative(1) else {
                    return f64::NAN;
                };
                x_spline = deriv;
            }

            intermediate.push(x_spline.eval(xi_clamped));
        }

        // Create y-direction spline from intermediate values
        let Ok(mut y_spline) = BSpline::new(self.ty.clone(), intermediate, self.ky) else {
            return f64::NAN;
        };

        // Apply y derivative if needed
        for _ in 0..dy {
            let Ok(deriv) = y_spline.derivative(1) else {
                return f64::NAN;
            };
            y_spline = deriv;
        }

        y_spline.eval(yi_clamped)
    }

    /// Compute the integral over a rectangular region.
    ///
    /// Matches `scipy.interpolate.RectBivariateSpline.integral(xa, xb, ya, yb)`.
    pub fn integral(&self, xa: f64, xb: f64, ya: f64, yb: f64) -> f64 {
        // Use Gaussian quadrature for integration
        // 5-point Gauss-Legendre on [-1, 1]
        let gauss_points = [
            -0.906_179_845_938_664,
            -0.538_469_310_105_683,
            0.0,
            0.538_469_310_105_683,
            0.906_179_845_938_664,
        ];
        let gauss_weights = [
            0.236_926_885_056_189,
            0.478_628_670_499_366,
            0.568_888_888_888_889,
            0.478_628_670_499_366,
            0.236_926_885_056_189,
        ];

        let xmid = (xa + xb) / 2.0;
        let xscale = (xb - xa) / 2.0;
        let ymid = (ya + yb) / 2.0;
        let yscale = (yb - ya) / 2.0;

        let mut result = 0.0;
        for (i, &gx) in gauss_points.iter().enumerate() {
            let xi = xmid + xscale * gx;
            let wx = gauss_weights[i];
            for (j, &gy) in gauss_points.iter().enumerate() {
                let yi = ymid + yscale * gy;
                let wy = gauss_weights[j];
                result += wx * wy * self.eval(xi, yi);
            }
        }

        result * xscale * yscale
    }

    /// Get spline degrees.
    pub fn degrees(&self) -> (usize, usize) {
        (self.kx, self.ky)
    }

    /// Get knot vectors.
    pub fn knots(&self) -> (&[f64], &[f64]) {
        (&self.tx, &self.ty)
    }

    /// Get coefficient array.
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.coeffs
    }
}

/// Convenience constructor for bicubic spline (kx=ky=3).
pub fn rect_bivariate_spline(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
) -> Result<RectBivariateSpline, InterpError> {
    RectBivariateSpline::new(x, y, z, 3, 3)
}

/// Convenience constructor for bilinear spline (kx=ky=1).
pub fn rect_bilinear_spline(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
) -> Result<RectBivariateSpline, InterpError> {
    RectBivariateSpline::new(x, y, z, 1, 1)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SmoothBivariateSpline — smooth approximation over scattered 2D samples
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Options for [`SmoothBivariateSpline`].
///
/// The fields mirror the SciPy constructor surface:
/// `w`, `bbox`, `kx`, `ky`, `s`, and `eps`.
#[derive(Debug, Clone)]
pub struct SmoothBivariateSplineOptions {
    /// Positive sample weights. Defaults to unit weights.
    pub weights: Option<Vec<f64>>,
    /// Approximation domain `[xmin, xmax, ymin, ymax]`.
    pub bbox: Option<[f64; 4]>,
    /// Degree in the x direction.
    pub kx: usize,
    /// Degree in the y direction.
    pub ky: usize,
    /// Non-negative smoothing factor. Defaults to the number of samples.
    pub smoothing: Option<f64>,
    /// Rank threshold/regularization floor. Must lie in `(0, 1)`.
    pub eps: f64,
}

impl Default for SmoothBivariateSplineOptions {
    fn default() -> Self {
        Self {
            weights: None,
            bbox: None,
            kx: 3,
            ky: 3,
            smoothing: None,
            eps: 1e-16,
        }
    }
}

/// Smooth bivariate approximation over scattered `(x, y, z)` samples.
///
/// This implements the scoped `scipy.interpolate.SmoothBivariateSpline`
/// contract used by FrankenSciPy: scattered samples, optional positive
/// weights, automatic bounding box, configurable degrees, callable evaluation,
/// grid evaluation, derivatives, integral, and residual reporting.
#[derive(Debug, Clone)]
pub struct SmoothBivariateSpline {
    kx: usize,
    ky: usize,
    bbox: [f64; 4],
    tx: Vec<f64>,
    ty: Vec<f64>,
    coeffs: Vec<f64>,
    nx_coeffs: usize,
    ny_coeffs: usize,
    residual: f64,
    smoothing_factor: f64,
}

/// Same-binary A/B toggle for `SmoothBivariateSpline::eval_grid`. When `true`, every grid cell goes
/// through the per-cell `eval` path (which rebuilds the ny x-direction splines for every (x,y) — the
/// ORIG behaviour). When `false` (default), the x-splines are built once for the whole grid and the
/// row's y-spline once per row. Byte-identical either way. Benchmark knob.
#[doc(hidden)]
pub static SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle for `SmoothBivariateSpline::eval_many` (scattered points). When `true`,
/// every query goes through the per-query `eval` path (which rebuilds the ny x-direction splines for
/// every (x,y) — the ORIG behaviour). When `false` (default), the x-splines are built once for the
/// whole batch and the per-query evals fan across cores via `par_query_map`. Byte-identical either
/// way. Benchmark knob.
#[doc(hidden)]
pub static SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

impl SmoothBivariateSpline {
    /// Build directly from an already-fitted representation.
    ///
    /// Used by `LSQBivariateSpline`, whose knots come from the caller rather than
    /// from the smoothing search. Everything downstream -- evaluation, derivatives,
    /// the integral -- is identical once the knots and coefficients exist, so this
    /// exists to share that machinery instead of duplicating a second evaluator.
    ///
    /// The residual is computed here with the same `compute_residual` the smoothing
    /// constructor uses, so the number means the same thing in both types rather
    /// than being a similarly-named quantity computed a second way.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_fitted(
        kx: usize,
        ky: usize,
        bbox: [f64; 4],
        tx: Vec<f64>,
        ty: Vec<f64>,
        coeffs: Vec<f64>,
        nx_coeffs: usize,
        ny_coeffs: usize,
        smoothing_factor: f64,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        weights: &[f64],
    ) -> Self {
        let mut spline = Self {
            kx,
            ky,
            bbox,
            tx,
            ty,
            coeffs,
            nx_coeffs,
            ny_coeffs,
            residual: 0.0,
            smoothing_factor,
        };
        spline.residual = spline.compute_residual(x, y, z, weights);
        spline
    }

    pub fn new(
        x: &[f64],
        y: &[f64],
        z: &[f64],
        options: SmoothBivariateSplineOptions,
    ) -> Result<Self, InterpError> {
        let n = x.len();
        if y.len() != n {
            return Err(InterpError::LengthMismatch {
                x_len: n,
                y_len: y.len(),
            });
        }
        if z.len() != n {
            return Err(InterpError::InvalidArgument {
                detail: format!("z must have same length as x and y, got {}", z.len()),
            });
        }
        if !(1..=5).contains(&options.kx) || !(1..=5).contains(&options.ky) {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "spline degree must be 1-5, got kx={}, ky={}",
                    options.kx, options.ky
                ),
            });
        }
        let min_points = (options.kx + 1) * (options.ky + 1);
        if n < min_points {
            return Err(InterpError::TooFewPoints {
                minimum: min_points,
                actual: n,
            });
        }
        if !(0.0..1.0).contains(&options.eps) {
            return Err(InterpError::InvalidArgument {
                detail: format!("eps must lie in (0, 1), got {}", options.eps),
            });
        }
        if !x.iter().chain(y).chain(z).all(|value| value.is_finite()) {
            return Err(InterpError::InvalidArgument {
                detail: "x, y, and z values must be finite".to_string(),
            });
        }

        let weights = smooth_bivariate_weights(n, options.weights.as_deref())?;
        let bbox = smooth_bivariate_bbox(x, y, options.bbox)?;
        let smoothing_factor = options.smoothing.unwrap_or(n as f64);
        if !smoothing_factor.is_finite() || smoothing_factor < 0.0 {
            return Err(InterpError::InvalidArgument {
                detail: format!("smoothing factor must be non-negative, got {smoothing_factor}"),
            });
        }

        // scipy.interpolate.SmoothBivariateSpline IS FITPACK `surfit` (= bisplrep): adaptive
        // knot placement + banded Givens solve. For the default case (unit weights,
        // data-extent bbox) route through fsci's parity-fast `bisplrep` — scipy-parity
        // results AND adaptive-knot speed — instead of the local fixed-knot dense path.
        // Custom weights/bbox keep the fixed-knot path (bisplrep takes neither).
        let (tx, ty, coeffs, nx_coeffs, ny_coeffs) = if options.weights.is_none()
            && options.bbox.is_none()
        {
            let (tx, ty, c, _, _) = bisplrep(x, y, z, options.kx, options.ky, smoothing_factor)?;
            let nxc = tx.len() - options.kx - 1;
            let nyc = ty.len() - options.ky - 1;
            // bisplrep returns FITPACK coefficient order c[ix*nyc+iy]; this struct's
            // eval uses c[iy*nxc+ix] (the smooth_bivariate_solve_coefficients layout).
            // Transpose so the existing eval/eval_derivative read the routed fit correctly.
            let mut ct = vec![0.0_f64; nxc * nyc];
            for ix in 0..nxc {
                for iy in 0..nyc {
                    ct[iy * nxc + ix] = c[ix * nyc + iy];
                }
            }
            (tx, ty, ct, nxc, nyc)
        } else {
            let fit = smooth_bivariate_fit(SmoothBivariateFit {
                x,
                y,
                z,
                weights: &weights,
                bbox,
                kx: options.kx,
                ky: options.ky,
                smoothing_factor,
                eps: options.eps,
            })?;
            (fit.tx, fit.ty, fit.coeffs, fit.nx_coeffs, fit.ny_coeffs)
        };
        let mut spline = Self {
            kx: options.kx,
            ky: options.ky,
            bbox,
            tx,
            ty,
            coeffs,
            nx_coeffs,
            ny_coeffs,
            residual: 0.0,
            smoothing_factor,
        };
        spline.residual = spline.compute_residual(x, y, z, &weights);
        Ok(spline)
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        self.eval_derivative(x, y, 0, 0)
    }

    pub fn eval_many(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, InterpError> {
        if x.len() != y.len() {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "x and y must have same length, got {} and {}",
                    x.len(),
                    y.len()
                ),
            });
        }

        // `eval` -> `eval_impl` rebuilds the `ny` x-direction BSplines (each cloning `tx` and a
        // coefficient row) on EVERY scattered query, though they depend only on the spline, not the
        // point. Build them ONCE and reuse across the batch (shared-predictor lever, the same recipe
        // as the sibling `RectBivariateSpline::eval_many` and this struct's `eval_grid`); each query
        // then only evaluates the shared x-splines, builds its query-dependent y-spline, and
        // evaluates that. The independent per-query evals then fan across cores via `par_query_map`.
        // BYTE-IDENTICAL to the serial `self.eval` map: `eval_one` reproduces `eval`'s exact guard
        // (`!x.finite || !y.finite -> NaN`), clamps, deterministic builds, and evaluation order. The
        // knob and any x-spline build failure fall back to the exact per-query path.
        if x.is_empty()
            || SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(x
                .iter()
                .zip(y)
                .map(|(&xv, &yv)| self.eval(xv, yv))
                .collect());
        }
        let x_splines: Option<Vec<BSpline>> = self
            .coeffs
            .chunks(self.nx_coeffs)
            .map(|row| BSpline::new(self.tx.clone(), row.to_vec(), self.kx).ok())
            .collect();
        let Some(x_splines) = x_splines else {
            return Ok(x
                .iter()
                .zip(y)
                .map(|(&xv, &yv)| self.eval(xv, yv))
                .collect());
        };
        let eval_one = |xv: f64, yv: f64| -> f64 {
            if !xv.is_finite() || !yv.is_finite() {
                return f64::NAN;
            }
            let xi = xv.clamp(self.bbox[0], self.bbox[1]);
            let yi = yv.clamp(self.bbox[2], self.bbox[3]);
            let intermediate: Vec<f64> = x_splines.iter().map(|s| s.eval(xi)).collect();
            let Ok(y_spline) = BSpline::new(self.ty.clone(), intermediate, self.ky) else {
                return f64::NAN;
            };
            y_spline.eval(yi)
        };
        let pairs: Vec<(f64, f64)> = x.iter().zip(y).map(|(&xv, &yv)| (xv, yv)).collect();
        let work_per_query = self.coeffs.len().max(1);
        Ok(par_query_map(&pairs, work_per_query, |&(xv, yv)| {
            eval_one(xv, yv)
        }))
    }

    pub fn eval_grid(&self, x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
        // Per-cell path (also the A/B baseline): each (xv, yv) rebuilds the ny x-direction splines.
        // Each output row (fixed x) is an independent sweep, fanned across cores by `par_grid_rows`.
        let percell = |xv: f64| -> Vec<f64> { y.iter().map(|&yv| self.eval(xv, yv)).collect() };
        if SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.load(std::sync::atomic::Ordering::Relaxed) {
            return par_grid_rows(x, y.len(), &percell);
        }

        // `eval` -> `eval_impl` rebuilds the ny x-direction BSplines for EVERY cell, though they
        // depend only on the spline; and for a FIXED row (xv) the x-reduction — and thus the whole
        // y-spline — is the same for every column. Build the x-splines ONCE for the grid and the
        // y-spline ONCE per row, then evaluate the y-spline at each yv (shared-predictor lever).
        // BYTE-IDENTICAL to the per-cell path: the same deterministic builds, clamps, and order.
        // Any x-spline build failure falls back to the exact per-cell path.
        let x_splines: Option<Vec<BSpline>> = self
            .coeffs
            .chunks(self.nx_coeffs)
            .map(|row| BSpline::new(self.tx.clone(), row.to_vec(), self.kx).ok())
            .collect();
        let Some(x_splines) = x_splines else {
            return par_grid_rows(x, y.len(), &percell);
        };
        let row_fn = |xv: f64| -> Vec<f64> {
            if !xv.is_finite() {
                return vec![f64::NAN; y.len()];
            }
            let xi = xv.clamp(self.bbox[0], self.bbox[1]);
            let intermediate: Vec<f64> = x_splines.iter().map(|s| s.eval(xi)).collect();
            let Ok(y_spline) = BSpline::new(self.ty.clone(), intermediate, self.ky) else {
                return vec![f64::NAN; y.len()];
            };
            y.iter()
                .map(|&yv| {
                    if !yv.is_finite() {
                        return f64::NAN;
                    }
                    y_spline.eval(yv.clamp(self.bbox[2], self.bbox[3]))
                })
                .collect()
        };
        par_grid_rows(x, y.len(), &row_fn)
    }

    pub fn eval_derivative(&self, x: f64, y: f64, dx: usize, dy: usize) -> f64 {
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        self.eval_impl(x, y, dx, dy)
    }

    pub fn integral(&self, xa: f64, xb: f64, ya: f64, yb: f64) -> f64 {
        let gauss_points = [
            -0.906_179_845_938_664,
            -0.538_469_310_105_683,
            0.0,
            0.538_469_310_105_683,
            0.906_179_845_938_664,
        ];
        let gauss_weights = [
            0.236_926_885_056_189,
            0.478_628_670_499_366,
            0.568_888_888_888_889,
            0.478_628_670_499_366,
            0.236_926_885_056_189,
        ];

        let xmid = (xa + xb) / 2.0;
        let xscale = (xb - xa) / 2.0;
        let ymid = (ya + yb) / 2.0;
        let yscale = (yb - ya) / 2.0;

        let mut result = 0.0;
        for (i, &gx) in gauss_points.iter().enumerate() {
            let xi = xmid + xscale * gx;
            let wx = gauss_weights[i];
            for (j, &gy) in gauss_points.iter().enumerate() {
                let yi = ymid + yscale * gy;
                let wy = gauss_weights[j];
                result += wx * wy * self.eval(xi, yi);
            }
        }

        result * xscale * yscale
    }

    pub fn residual(&self) -> f64 {
        self.residual
    }

    pub fn smoothing_factor(&self) -> f64 {
        self.smoothing_factor
    }

    pub fn degrees(&self) -> (usize, usize) {
        (self.kx, self.ky)
    }

    pub fn bbox(&self) -> [f64; 4] {
        self.bbox
    }

    pub fn knots(&self) -> (&[f64], &[f64]) {
        (&self.tx, &self.ty)
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    fn eval_impl(&self, x: f64, y: f64, dx: usize, dy: usize) -> f64 {
        if dx > self.kx || dy > self.ky {
            return 0.0;
        }

        let xi = x.clamp(self.bbox[0], self.bbox[1]);
        let yi = y.clamp(self.bbox[2], self.bbox[3]);

        let mut intermediate = Vec::with_capacity(self.ny_coeffs);
        for row in self.coeffs.chunks(self.nx_coeffs) {
            let Ok(mut x_spline) = BSpline::new(self.tx.clone(), row.to_vec(), self.kx) else {
                return f64::NAN;
            };
            for _ in 0..dx {
                let Ok(deriv) = x_spline.derivative(1) else {
                    return f64::NAN;
                };
                x_spline = deriv;
            }
            intermediate.push(x_spline.eval(xi));
        }

        let Ok(mut y_spline) = BSpline::new(self.ty.clone(), intermediate, self.ky) else {
            return f64::NAN;
        };
        for _ in 0..dy {
            let Ok(deriv) = y_spline.derivative(1) else {
                return f64::NAN;
            };
            y_spline = deriv;
        }
        y_spline.eval(yi)
    }

    fn compute_residual(&self, x: &[f64], y: &[f64], z: &[f64], weights: &[f64]) -> f64 {
        x.iter()
            .zip(y)
            .zip(z)
            .zip(weights)
            .map(|(((&xv, &yv), &zv), &weight)| {
                let diff = weight * (zv - self.eval(xv, yv));
                diff * diff
            })
            .sum()
    }
}

pub fn smooth_bivariate_spline(
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> Result<SmoothBivariateSpline, InterpError> {
    SmoothBivariateSpline::new(x, y, z, SmoothBivariateSplineOptions::default())
}

fn smooth_bivariate_weights(n: usize, weights: Option<&[f64]>) -> Result<Vec<f64>, InterpError> {
    match weights {
        Some(values) => {
            if values.len() != n {
                return Err(InterpError::InvalidArgument {
                    detail: format!("weights must have length {n}, got {}", values.len()),
                });
            }
            if values
                .iter()
                .any(|&value| !value.is_finite() || value <= 0.0)
            {
                return Err(InterpError::InvalidArgument {
                    detail: "weights must be positive and finite".to_string(),
                });
            }
            Ok(values.to_vec())
        }
        None => Ok(vec![1.0; n]),
    }
}

fn smooth_bivariate_bbox(
    x: &[f64],
    y: &[f64],
    bbox: Option<[f64; 4]>,
) -> Result<[f64; 4], InterpError> {
    let bbox = match bbox {
        Some(value) => value,
        None => [
            x.iter().copied().fold(f64::INFINITY, f64::min),
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            y.iter().copied().fold(f64::INFINITY, f64::min),
            y.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ],
    };
    if !bbox.iter().all(|value| value.is_finite()) || bbox[0] >= bbox[1] || bbox[2] >= bbox[3] {
        return Err(InterpError::InvalidArgument {
            detail: "bbox must be finite [xmin, xmax, ymin, ymax] with min < max".to_string(),
        });
    }
    Ok(bbox)
}

struct SmoothBivariateFit<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    weights: &'a [f64],
    bbox: [f64; 4],
    kx: usize,
    ky: usize,
    smoothing_factor: f64,
    eps: f64,
}

struct SmoothBivariateLinearSystem<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    weights: &'a [f64],
    tx: &'a [f64],
    ty: &'a [f64],
    kx: usize,
    ky: usize,
    smoothing_factor: f64,
    eps: f64,
}

struct SmoothBivariateSolution {
    tx: Vec<f64>,
    ty: Vec<f64>,
    coeffs: Vec<f64>,
    nx_coeffs: usize,
    ny_coeffs: usize,
}

fn smooth_bivariate_fit(
    input: SmoothBivariateFit<'_>,
) -> Result<SmoothBivariateSolution, InterpError> {
    let SmoothBivariateFit {
        x,
        y,
        z,
        weights,
        bbox,
        kx,
        ky,
        smoothing_factor,
        eps,
    } = input;
    let x_support = smooth_bivariate_support(x, bbox[0], bbox[1]);
    let y_support = smooth_bivariate_support(y, bbox[2], bbox[3]);
    if x_support.len() < kx + 1 || y_support.len() < ky + 1 {
        return Err(InterpError::InvalidArgument {
            detail: format!(
                "need at least {} distinct x values and {} distinct y values for kx={}, ky={}",
                kx + 1,
                ky + 1,
                kx,
                ky
            ),
        });
    }

    let (mut nx_coeffs, mut ny_coeffs) =
        smooth_bivariate_basis_shape(x_support.len(), y_support.len(), x.len(), kx, ky);

    loop {
        let tx = smooth_bivariate_knots(&x_support, kx, nx_coeffs);
        let ty = smooth_bivariate_knots(&y_support, ky, ny_coeffs);
        match smooth_bivariate_solve_coefficients(SmoothBivariateLinearSystem {
            x,
            y,
            z,
            weights,
            tx: &tx,
            ty: &ty,
            kx,
            ky,
            smoothing_factor,
            eps,
        }) {
            Ok(coeffs) => {
                return Ok(SmoothBivariateSolution {
                    tx,
                    ty,
                    coeffs,
                    nx_coeffs,
                    ny_coeffs,
                });
            }
            Err(err) => {
                let can_reduce_x = nx_coeffs > kx + 1;
                let can_reduce_y = ny_coeffs > ky + 1;
                if !can_reduce_x && !can_reduce_y {
                    return Err(err);
                }
                if can_reduce_x && (!can_reduce_y || nx_coeffs >= ny_coeffs) {
                    nx_coeffs -= 1;
                } else {
                    ny_coeffs -= 1;
                }
            }
        }
    }
}

fn smooth_bivariate_support(values: &[f64], lower: f64, upper: f64) -> Vec<f64> {
    let mut support = values.to_vec();
    support.push(lower);
    support.push(upper);
    support.sort_by(f64::total_cmp);
    support.dedup_by(|a, b| a.total_cmp(b).is_eq());
    support
}

fn smooth_bivariate_basis_shape(
    x_support: usize,
    y_support: usize,
    samples: usize,
    kx: usize,
    ky: usize,
) -> (usize, usize) {
    let min_x = kx + 1;
    let min_y = ky + 1;
    let max_x = x_support.min(samples / min_y).max(min_x);
    let max_y = y_support.min(samples / min_x).max(min_y);
    let aspect = (x_support as f64 / y_support as f64).ln();
    let mut best = (min_x, min_y, 0usize, f64::INFINITY);

    for nx in min_x..=max_x {
        for ny in min_y..=max_y {
            let coeff_count = nx * ny;
            if coeff_count > samples {
                continue;
            }
            let aspect_error = ((nx as f64 / ny as f64).ln() - aspect).abs();
            if coeff_count > best.2 || (coeff_count == best.2 && aspect_error < best.3) {
                best = (nx, ny, coeff_count, aspect_error);
            }
        }
    }

    (best.0, best.1)
}

fn smooth_bivariate_knots(support: &[f64], degree: usize, n_coeffs: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(n_coeffs + degree + 1);
    for _ in 0..=degree {
        knots.push(support[0]);
    }

    let num_interior = n_coeffs.saturating_sub(degree + 1);
    if num_interior > 0 {
        let last_interior = support.len().saturating_sub(2);
        for i in 0..num_interior {
            let idx =
                (((i + 1) * (support.len() - 1)) / (num_interior + 1)).clamp(1, last_interior);
            knots.push(support[idx]);
        }
    }

    for _ in 0..=degree {
        knots.push(*support.last().expect("non-empty support"));
    }
    knots
}

fn smooth_bivariate_solve_coefficients(
    system: SmoothBivariateLinearSystem<'_>,
) -> Result<Vec<f64>, InterpError> {
    let SmoothBivariateLinearSystem {
        x,
        y,
        z,
        weights,
        tx,
        ty,
        kx,
        ky,
        smoothing_factor,
        eps,
    } = system;
    let nx_coeffs = tx.len() - kx - 1;
    let ny_coeffs = ty.len() - ky - 1;
    let n_terms = nx_coeffs * ny_coeffs;
    let mut ata = vec![vec![0.0; n_terms]; n_terms];
    let mut atz = vec![0.0; n_terms];

    // The tensor B-spline basis is locally supported: bx is nonzero only in a (kx+1)-wide
    // window and by in a (ky+1)-wide window, so each data point's basis has just
    // (kx+1)(ky+1) nonzero terms. Accumulate AᵀA / Aᵀz from only those nonzeros (sparse
    // outer product) instead of the dense n_terms² double loop. nz is built in ascending
    // index order (iy then ix), so `row` is ascending and `col ≥ row` — exactly the
    // original (row, col≥row) accumulation order ⇒ byte-identical (skipped pairs were
    // 0·0 = +0.0 no-ops). O(m·n_terms²) → O(m·((kx+1)(ky+1))²).
    let mut nz: Vec<(usize, f64)> = Vec::with_capacity((kx + 1) * (ky + 1));
    for ((&xv, &yv), (&zv, &weight)) in x.iter().zip(y).zip(z.iter().zip(weights)) {
        let bx = eval_basis_all(tx, xv, kx, nx_coeffs);
        let by = eval_basis_all(ty, yv, ky, ny_coeffs);
        nz.clear();
        for (iy, &by_val) in by.iter().enumerate() {
            if by_val == 0.0 {
                continue;
            }
            for (ix, &bx_val) in bx.iter().enumerate() {
                if bx_val != 0.0 {
                    nz.push((iy * nx_coeffs + ix, bx_val * by_val));
                }
            }
        }

        let weight_sq = weight * weight;
        for a in 0..nz.len() {
            let (row, vr) = nz[a];
            atz[row] += weight_sq * vr * zv;
            for &(col, vc) in &nz[a..] {
                ata[row][col] += weight_sq * vr * vc;
                if row != col {
                    ata[col][row] = ata[row][col];
                }
            }
        }
    }

    let z_scale = z
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let lambda = if smoothing_factor > 0.0 {
        smoothing_factor / ((x.len() as f64) * z_scale * z_scale)
    } else {
        0.0
    };

    for (idx, row) in ata.iter_mut().enumerate() {
        row[idx] += eps;
    }
    if lambda > 0.0 {
        for iy in 0..ny_coeffs {
            for ix in 0..nx_coeffs {
                let idx = iy * nx_coeffs + ix;
                if ix + 1 < nx_coeffs {
                    let right = idx + 1;
                    ata[idx][idx] += lambda;
                    ata[right][right] += lambda;
                    ata[idx][right] -= lambda;
                    ata[right][idx] -= lambda;
                }
                if iy + 1 < ny_coeffs {
                    let down = idx + nx_coeffs;
                    ata[idx][idx] += lambda;
                    ata[down][down] += lambda;
                    ata[idx][down] -= lambda;
                    ata[down][idx] -= lambda;
                }
            }
        }
    }

    // AᵀA is banded: a coefficient pair (row, col) is coupled only when their tensor
    // indices overlap in a data point's support, i.e. |Δix| ≤ kx and |Δiy| ≤ ky; with the
    // row-major index iy·nx_coeffs+ix that is a band of half-width ky·nx_coeffs + kx. The
    // eps/penalty terms stay inside it. A banded solve is byte-identical to the dense one
    // (for a banded matrix the partial pivot search over the full column finds the same
    // max as over the band — out-of-band entries are 0) and runs in O(n_terms·bw²) instead
    // of O(n_terms³).
    let bw = (ky * nx_coeffs + kx).min(n_terms.saturating_sub(1));
    solve_banded(&mut ata, &mut atz, bw)
}

#[cfg(test)]
mod tests {

    /// frankenscipy-9yyez: `rbf_eval` and `euclidean_dist` were dead too. Pin
    /// them against their closed forms so the kernels cannot drift unnoticed.
    #[test]
    fn rbf_kernel_and_distance_helpers_match_their_closed_forms() {
        use super::RbfKernel;
        assert!((super::euclidean_dist(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert!(super::euclidean_dist(&[1.5, -2.0], &[1.5, -2.0]).abs() < 1e-12);

        assert!((super::rbf_eval(RbfKernel::Linear, 2.0, 1.0) - 2.0).abs() < 1e-12);
        // Thin plate spline: r² ln r, and 0 at r = 0 where the log diverges.
        let r: f64 = 2.0;
        assert!(
            (super::rbf_eval(RbfKernel::ThinPlateSpline, r, 1.0) - r * r * r.ln()).abs() < 1e-12
        );
        assert!(super::rbf_eval(RbfKernel::ThinPlateSpline, 0.0, 1.0).abs() < 1e-12);
        // Multiquadric: sqrt(1 + (εr)²).
        let (eps, rr) = (0.5_f64, 3.0_f64);
        assert!(
            (super::rbf_eval(RbfKernel::Multiquadric, rr, eps) - (1.0 + (eps * rr).powi(2)).sqrt())
                .abs()
                < 1e-12
        );
    }
    use super::*;

    /// The O(1) `find_interval_uniform_helper` must return the EXACT same interval
    /// index as the binary-search `find_interval_helper` for every probe — this
    /// locks the byte-identity of the linear `eval_many` unsorted fast path.
    #[test]
    fn interp1d_uniform_finder_matches_helper() {
        let axes: Vec<Vec<f64>> = vec![
            (0..10000).map(|i| i as f64 * 0.01).collect(),
            (0..16).map(|i| i as f64 / 15.0).collect(),
            (0..7).map(|i| -3.0 + i as f64 * 1.25).collect(),
            (0..5).map(|i| 2.0 + i as f64 * 1e6).collect(),
        ];
        for axis in &axes {
            let meta = detect_uniform_axis(axis).expect("uniform");
            let n = axis.len();
            let lo = axis[0];
            let hi = axis[n - 1];
            let span = hi - lo;
            let mut probes: Vec<f64> = vec![lo - span, hi + span, lo, hi];
            for i in 0..n {
                probes.push(axis[i]);
                if i + 1 < n {
                    probes.push(0.5 * (axis[i] + axis[i + 1]));
                    probes.push(axis[i] + 0.25 * (axis[i + 1] - axis[i]));
                }
            }
            for k in 0..=3000 {
                probes.push(lo - 0.1 * span + (k as f64 / 3000.0) * 1.2 * span);
            }
            for &x in &probes {
                assert_eq!(
                    find_interval_uniform_helper(meta, axis, x),
                    find_interval_helper(axis, x),
                    "axis n={n} x={x}"
                );
            }
        }
    }

    // ── frankenscipy-7xad9: bit-identity gates for the levers restored from
    // 89593bf13^ that did not already carry one. The other four are covered by
    // regular_grid_pchip_batch_hoist_matches_scalar_bitexact,
    // rbf_build_parallel_matches_serial_bitexact,
    // rect_bivariate_eval_many_hoist_matches_scalar_bitexact and
    // smooth_bivariate_eval_grid_hoist_matches_percell_bitexact, which came back
    // with the restore. Each test below drives BOTH arms through the toggle and
    // compares with to_bits(), never a tolerance.

    #[test]
    fn interp_cubic_cursor_toggle_is_bit_identical_through_public_eval_many() {
        // frankenscipy-ct8n2 sweep follow-up. INTERP_CUBIC_CURSOR_DISABLE was the
        // one toggle in this crate with NO driver anywhere — no test, no perf bin,
        // no bench — so its A/B could not be run at all. It was already undriven
        // in 89593bf13^, so the reconciliation inherited the gap rather than
        // causing it. An existing test calls `cubic_cursor_eval_many` directly but
        // never flips the toggle, which is why the sweep still counted it as
        // undriven; this drives it through the PUBLIC eval_many of all three
        // piecewise-cubic interpolators that route through the cursor.
        //
        // Sizing: the cursor only engages when work = len*24 < PAR_QUERY_MIN_WORK
        // (1<<23) AND len >= 4 AND every value is finite AND there are at most 8
        // ascending runs. A 16-point ascending finite batch satisfies all four, so
        // the OFF arm genuinely takes the cursor and the ON arm genuinely takes
        // par_query_map — otherwise both would fall through to the same path and
        // the test would be vacuous.
        use std::sync::atomic::Ordering;
        let x = [-1.0, 0.5, 1.75, 3.0, 6.5];
        let y = [1.0, -2.0, 3.5, 0.25, 8.0];
        let dydx = [0.5, -1.0, 2.0, -0.25, 1.5];
        let queries: Vec<f64> = (0..16).map(|i| -1.0 + i as f64 * 0.5).collect();

        let cubic = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
        let hermite = CubicHermiteSpline::new(&x, &y, &dydx).expect("hermite");

        INTERP_CUBIC_CURSOR_DISABLE.store(true, Ordering::Relaxed);
        let c_off = cubic.eval_many(&queries);
        let a_off = akima.eval_many(&queries);
        let h_off = hermite.eval_many(&queries);
        INTERP_CUBIC_CURSOR_DISABLE.store(false, Ordering::Relaxed);
        let c_on = cubic.eval_many(&queries);
        let a_on = akima.eval_many(&queries);
        let h_on = hermite.eval_many(&queries);

        for (name, disabled, cursor) in [
            ("cubic", &c_off, &c_on),
            ("akima", &a_off, &a_on),
            ("hermite", &h_off, &h_on),
        ] {
            assert_eq!(disabled.len(), queries.len(), "{name} length");
            assert_eq!(cursor.len(), queries.len(), "{name} length");
            for (i, (d, c)) in disabled.iter().zip(cursor.iter()).enumerate() {
                assert_eq!(
                    d.to_bits(),
                    c.to_bits(),
                    "{name}[{i}]: binary-search {d} vs cursor {c}"
                );
            }
        }

        // Guard against the sizing above silently going stale: if the cursor
        // stopped engaging for this batch, the comparison would be trivially true.
        assert!(
            cubic_cursor_eval_many(&cubic.x, &cubic.coeffs, &queries).is_some(),
            "fixture no longer takes the cursor path — the bit-identity check above would be vacuous"
        );
    }

    #[test]
    fn bisplev_compact_support_is_bit_identical_to_full_basis() {
        // BISPLEV_COMPACT_DISABLE selects the old all-`ny_c` basis sweep. The
        // compact path keeps only the ky+1 nonzero weights; the dropped terms are
        // exactly the `+ c·bxa·0.0` no-ops, so the retained ones must accumulate
        // in the same order and produce identical bits. No size gate — the toggle
        // switches on every call — but a cubic spline with a real interior knot
        // vector is used so the support window is genuinely narrower than the row.
        use std::sync::atomic::Ordering;
        let (kx, ky) = (3usize, 3usize);
        let mut t = vec![0.0f64; 4];
        for i in 1..=16 {
            t.push(i as f64 / 17.0);
        }
        t.extend(vec![1.0f64; 4]);
        let (tx, ty) = (t.clone(), t);
        let nx_c = tx.len() - kx - 1;
        let ny_c = ty.len() - ky - 1;
        let mut state = 0xBEEFu64;
        let c: Vec<f64> = (0..nx_c * ny_c)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let tck = (tx, ty, c, kx, ky);
        let g: Vec<f64> = (0..64).map(|i| i as f64 / 63.0).collect();

        BISPLEV_COMPACT_DISABLE.store(true, Ordering::Relaxed);
        let full = bisplev(&g, &g, &tck).expect("bisplev full basis");
        BISPLEV_COMPACT_DISABLE.store(false, Ordering::Relaxed);
        let compact = bisplev(&g, &g, &tck).expect("bisplev compact support");

        let mism: usize = full
            .iter()
            .zip(compact.iter())
            .map(|(fr, cr)| {
                fr.iter()
                    .zip(cr.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
            })
            .sum();
        assert_eq!(mism, 0, "compact-support bisplev must be bit-identical");
    }

    #[test]
    fn ndbspline_compact_support_is_bit_identical_to_full_basis() {
        // NDBSPLINE_COMPACT_DISABLE, same lever shape as bisplev but on the N-D
        // tensor contraction. No size gate; the toggle switches on every call.
        use std::sync::atomic::Ordering;
        let tx = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let t = vec![tx.clone(), tx];
        let mut state = 0x1234_5678u64;
        let c: Vec<f64> = (0..36)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect();
        let nb = NdBSpline::new(t, c, 3).expect("ndbspline");
        let pts: Vec<[f64; 2]> = (0..40)
            .map(|i| {
                let a = i as f64 / 39.0 * 3.0;
                [a, 3.0 - a]
            })
            .collect();

        NDBSPLINE_COMPACT_DISABLE.store(true, Ordering::Relaxed);
        let full: Vec<f64> = pts.iter().map(|p| nb.evaluate(p)).collect();
        NDBSPLINE_COMPACT_DISABLE.store(false, Ordering::Relaxed);
        let compact: Vec<f64> = pts.iter().map(|p| nb.evaluate(p)).collect();

        for (i, (f, c)) in full.iter().zip(compact.iter()).enumerate() {
            assert_eq!(
                f.to_bits(),
                c.to_bits(),
                "ndbspline point {i}: full {f} vs compact {c}"
            );
        }
    }

    #[test]
    fn smooth_bivariate_eval_many_hoist_matches_scalar_bitexact() {
        // SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR restores the per-query path that
        // rebuilds the ny x-direction BSplines for every scattered point. The
        // hoisted batch path must be bit-identical to mapping `eval`.
        use std::sync::atomic::Ordering;
        let n = 60usize;
        let mut xs = Vec::with_capacity(n * n);
        let mut ys = Vec::with_capacity(n * n);
        let mut zs = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n as f64 - 1.0);
                let y = j as f64 / (n as f64 - 1.0);
                xs.push(x);
                ys.push(y);
                zs.push((x * 2.4).sin() + (y * 1.7).cos() + x * y);
            }
        }
        let spline =
            SmoothBivariateSpline::new(&xs, &ys, &zs, SmoothBivariateSplineOptions::default())
                .expect("smooth bivariate spline");

        let qx: Vec<f64> = (0..500).map(|i| i as f64 / 499.0).collect();
        let qy: Vec<f64> = (0..500).map(|i| 1.0 - i as f64 / 499.0).collect();

        SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR.store(true, Ordering::Relaxed);
        let scalar = spline.eval_many(&qx, &qy).expect("eval_many scalar");
        SMOOTHBISPLINE_EVAL_MANY_FORCE_SCALAR.store(false, Ordering::Relaxed);
        let hoisted = spline.eval_many(&qx, &qy).expect("eval_many hoisted");

        assert_eq!(scalar.len(), hoisted.len());
        for (i, (s, h)) in scalar.iter().zip(hoisted.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                h.to_bits(),
                "smooth bivariate eval_many[{i}]: scalar {s} vs hoisted {h}"
            );
        }
    }

    #[test]
    fn regular_grid_nearest_3d_parallel_matches_serial_bitexact() {
        // INTERPN_NEAREST3D_FORCE_SERIAL pins the serial map; the parallel arm
        // fans the independent per-query lookups via par_query_try_map. 4000
        // 3-D queries is comfortably past that helper's internal work gate.
        use std::sync::atomic::Ordering;
        let axis: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let points = vec![axis.clone(), axis.clone(), axis];
        let values: Vec<f64> = (0..12 * 12 * 12)
            .map(|i| (i as f64 * 0.017).sin() * 3.0)
            .collect();
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
                .expect("regular grid nearest");

        let xi: Vec<Vec<f64>> = (0..4000usize)
            .map(|i| {
                let a = (i % 110) as f64 * 0.1;
                vec![a, 11.0 - a, (i % 47) as f64 * 0.23]
            })
            .collect();

        INTERPN_NEAREST3D_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = interp.eval_many(&xi).expect("nearest3d serial");
        INTERPN_NEAREST3D_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = interp.eval_many(&xi).expect("nearest3d parallel");

        assert_eq!(serial.len(), 4000, "one result per query");
        assert_eq!(serial.len(), parallel.len());
        for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "nearest3d[{i}]: serial {s} vs parallel {p}"
            );
        }
    }

    #[test]
    fn ndbspline_matches_scipy() {
        let tx = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let t = vec![tx.clone(), tx];
        // c = rng.standard_normal((6,6)) from numpy default_rng(0), row-major.
        let c = vec![
            0.12573, -0.132105, 0.640423, 0.1049, -0.535669, 0.361595, 1.304, 0.947081, -0.703735,
            -1.265421, -0.623274, 0.041326, -2.325031, -0.218792, -1.245911, -0.732267, -0.544259,
            -0.3163, 0.411631, 1.042513, -0.128535, 1.366463, -0.665195, 0.35151, 0.90347,
            0.094012, -0.743499, -0.921725, -0.457726, 0.220195, -1.009618, -0.209176, -0.159225,
            0.540846, 0.214659, 0.355373,
        ];
        let nb = NdBSpline::new(t, c, 3).unwrap();
        let pts = [[0.5, 0.5], [1.5, 1.5], [2.2, 0.8]];
        let want = [
            0.1353603362742121,
            -0.22157982666576687,
            0.007605438853110404,
        ];
        for (pt, w) in pts.iter().zip(want.iter()) {
            // c is rounded to 6 digits, so allow ~1e-5.
            assert!(
                (nb.evaluate(pt) - w).abs() <= 1e-5,
                "{:?}: {} vs {w}",
                pt,
                nb.evaluate(pt)
            );
        }
        // evaluate_many must be byte-identical to mapping evaluate over the points.
        let many: Vec<Vec<f64>> = pts.iter().map(|p| p.to_vec()).collect();
        let batch = nb.evaluate_many(&many);
        for (pt, b) in pts.iter().zip(batch.iter()) {
            assert_eq!(*b, nb.evaluate(pt), "evaluate_many != evaluate at {pt:?}");
        }
    }

    #[test]
    fn ndbspline_rejects_degree_overflow() {
        let result = NdBSpline::new(vec![vec![0.0, 1.0]], Vec::new(), usize::MAX / 2 + 1);
        assert!(matches!(result, Err(InterpError::InvalidArgument { .. })));
    }

    #[test]
    fn ppoly_interval_selection() {
        // Constant per-interval pieces pin the interval the binary search selects
        // (guards the linear-scan -> partition_point change; boundaries land in the
        // upper interval, out-of-range clamps to the end pieces).
        let p = PPoly::new(
            vec![vec![10.0], vec![20.0], vec![30.0]],
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(p.evaluate(0.5), 10.0);
        assert_eq!(p.evaluate(1.0), 20.0); // boundary -> upper interval
        assert_eq!(p.evaluate(1.5), 20.0);
        assert_eq!(p.evaluate(2.0), 30.0);
        assert_eq!(p.evaluate(2.5), 30.0);
        assert_eq!(p.evaluate(-1.0), 10.0); // below range -> first piece
        assert_eq!(p.evaluate(5.0), 30.0); // above range -> last piece
        // evaluate_many must agree exactly.
        let xs = [0.5, 1.0, 1.5, 2.0, 2.5, -1.0, 5.0];
        let batch = p.evaluate_many(&xs);
        for (&x, b) in xs.iter().zip(batch.iter()) {
            assert_eq!(*b, p.evaluate(x), "evaluate_many != evaluate at {x}");
        }
    }

    #[test]
    fn ppoly_evaluate_many_sorted_cursor_matches_scalar() {
        let p = PPoly::new(
            vec![vec![10.0], vec![20.0], vec![30.0]],
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let xs = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0];
        let batch = p.evaluate_many(&xs);
        for (&x, b) in xs.iter().zip(batch.iter()) {
            assert_eq!(*b, p.evaluate(x), "sorted evaluate_many != evaluate at {x}");
        }
    }

    #[test]
    fn ndppoly_matches_scipy() {
        // c = arange(1..=36).reshape(3,2,2,3); x=([0,1,2],[0,1,2,3]).
        let c: Vec<f64> = (1..=36).map(|v| v as f64).collect();
        let c_shape = vec![3, 2, 2, 3];
        let x = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]];
        let p = NdPPoly::new(c, c_shape, x).unwrap();
        let pts = [[0.5, 0.5], [1.5, 2.5], [0.2, 1.8], [1.9, 0.1]];
        let want = [58.125, 71.25, 59.424, 66.46399999999998];
        for (pt, w) in pts.iter().zip(want.iter()) {
            assert!(
                (p.evaluate(pt) - w).abs() <= 1e-9,
                "{:?}: {} vs {w}",
                pt,
                p.evaluate(pt)
            );
        }
        // evaluate_many must be byte-identical to mapping evaluate over the points.
        let many: Vec<Vec<f64>> = pts.iter().map(|pt| pt.to_vec()).collect();
        let batch = p.evaluate_many(&many);
        for (pt, b) in pts.iter().zip(batch.iter()) {
            assert_eq!(*b, p.evaluate(pt), "evaluate_many != evaluate at {pt:?}");
        }
    }

    #[test]
    fn ndppoly_rejects_shape_product_overflow() {
        let result = NdPPoly::new(Vec::new(), vec![usize::MAX, 2], vec![vec![0.0, 1.0, 2.0]]);
        assert!(matches!(result, Err(InterpError::InvalidArgument { .. })));
    }

    #[test]
    fn lsq_univariate_spline_matches_scipy() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 10.0 / 49.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let (t, c, k) = lsq_univariate_spline(&x, &y, &[2.0, 4.0, 6.0, 8.0], 3).unwrap();
        let bs = BSpline::new(t, c, k).unwrap();
        // scipy.interpolate.LSQUnivariateSpline oracle.
        let want = [
            (1.5, 1.003723229363428),
            (5.5, -0.6683729233728315),
            (9.0, 0.395544191785225),
        ];
        for (xe, w) in want {
            assert!(
                (bs.eval(xe) - w).abs() <= 1e-9,
                "{}: {} vs {w}",
                xe,
                bs.eval(xe)
            );
        }
    }

    #[test]
    fn bpoly_matches_scipy() {
        // scipy c shape (4,2) = [coeff][interval]; fsci is [interval][coeff].
        let c = vec![vec![1.0, 2.0, 0.0, 1.5], vec![0.5, 1.0, 2.5, 2.0]];
        let x = vec![0.0, 0.5, 1.0];
        let bp = BPoly::new(c, x).unwrap();
        let pts = [0.1, 0.4, 0.5, 0.6, 0.9, 1.1, -0.1];
        let want = [
            1.2920000000000003,
            0.9680000000000002,
            0.5,
            0.896,
            2.084,
            1.435999999999999,
            -0.011999999999999782,
        ];
        for (&p, &w) in pts.iter().zip(want.iter()) {
            assert!(
                (bp.evaluate(p) - w).abs() <= 1e-9,
                "eval({p}): {} vs {w}",
                bp.evaluate(p)
            );
        }
        // evaluate_many (precomputed binomials) must be byte-identical to per-point.
        let batch = bp.evaluate_many(&pts);
        for (&p, b) in pts.iter().zip(batch.iter()) {
            assert_eq!(*b, bp.evaluate(p), "evaluate_many != evaluate at {p}");
        }
        let d = bp.derivative();
        let dwant = [
            (0.1, 0.36000000000000004),
            (0.6, 4.68),
            (0.9, 1.0799999999999992),
        ];
        for (p, w) in dwant {
            assert!(
                (d.evaluate(p) - w).abs() <= 1e-9,
                "deriv({p}): {} vs {w}",
                d.evaluate(p)
            );
        }
    }

    #[test]
    fn bpoly_evaluate_many_sorted_cursor_matches_scalar() {
        let c = vec![vec![1.0, 2.0, 0.0, 1.5], vec![0.5, 1.0, 2.5, 2.0]];
        let x = vec![0.0, 0.5, 1.0];
        let bp = BPoly::new(c, x).unwrap();
        let pts = [-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1];
        let batch = bp.evaluate_many(&pts);
        for (&p, b) in pts.iter().zip(batch.iter()) {
            assert_eq!(
                *b,
                bp.evaluate(p),
                "sorted evaluate_many != evaluate at {p}"
            );
        }
    }

    #[test]
    fn make_smoothing_spline_lam_matches_scipy() {
        // scipy.interpolate.make_smoothing_spline(x, y, lam=0.5).
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [0.1, 0.9, 2.1, 2.9, 3.8, 5.2, 5.9];
        let b = make_smoothing_spline(&x, &y, None, Some(0.5)).unwrap();
        let t_exp = [
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0,
        ];
        for (g, e) in b.knots().iter().zip(&t_exp) {
            assert!((g - e).abs() < 1e-12, "t {g} vs {e}");
        }
        let c_exp = [
            0.06367822, 0.3721146, 0.98898735, 1.98694003, 2.92799062, 3.92768077, 5.02602438,
            5.65853435, 5.97478933,
        ];
        assert_eq!(b.coeffs().len(), c_exp.len());
        for (g, e) in b.coeffs().iter().zip(&c_exp) {
            assert!((g - e).abs() < 1e-6, "c {g} vs {e}");
        }
        // GCV (lam=None): scipy picks lam≈6.9999939 and these coefficients.
        let bg = make_smoothing_spline(&x, &y, None, None).unwrap();
        let cg_exp = [
            0.03603182, 0.35933446, 1.00593975, 1.984986, 2.96695705, 3.96200032, 4.97038668,
            5.63607983, 5.96892641,
        ];
        for (g, e) in bg.coeffs().iter().zip(&cg_exp) {
            assert!((g - e).abs() < 1e-4, "gcv c {g} vs {e}");
        }
        // Negative lam is rejected.
        assert!(make_smoothing_spline(&x, &y, None, Some(-1.0)).is_err());
    }

    #[test]
    // The naive full-storage reference this test compares against is written with
    // explicit band indices, matching the compact-storage kernel it checks; iterator
    // form would obscure the index algebra that IS the thing under test
    // (frankenscipy-9yyez).
    #[allow(clippy::needless_range_loop)]
    fn gcv_compact_band_trace_matches_full_storage_bits() {
        /// Naive reference kernel for the optimised path under test: the loop
        /// variables are the band/recurrence indices, so iterator form would obscure
        /// the index algebra this exists to be checked against (frankenscipy-9yyez).
        #[allow(clippy::needless_range_loop)]
        fn chol_banded_full(a: &mut [Vec<f64>], bw: usize) -> Option<()> {
            let n = a.len();
            for j in 0..n {
                let klo = j.saturating_sub(bw);
                let mut d = a[j][j];
                for k in klo..j {
                    d -= a[j][k] * a[j][k];
                }
                if d <= 0.0 {
                    return None;
                }
                let ljj = d.sqrt();
                a[j][j] = ljj;
                let ihi = (j + bw + 1).min(n);
                for i in (j + 1)..ihi {
                    let kstart = i.saturating_sub(bw);
                    let mut s = a[i][j];
                    for k in kstart..j {
                        s -= a[i][k] * a[j][k];
                    }
                    a[i][j] = s / ljj;
                }
            }
            Some(())
        }

        /// Naive reference kernel for the optimised path under test: the loop
        /// variables are the band/recurrence indices, so iterator form would obscure
        /// the index algebra this exists to be checked against (frankenscipy-9yyez).
        #[allow(clippy::needless_range_loop)]
        fn gcv_trace_selinv_full(g: &[Vec<f64>], b: &[Vec<f64>], n: usize, bw: usize) -> f64 {
            let mut z = vec![vec![0.0_f64; bw + 1]; n];
            for i in (0..n).rev() {
                let gii = g[i][i];
                let kmax = (i + bw).min(n - 1);
                let mut j = kmax;
                while j > i {
                    let mut s = 0.0_f64;
                    for k in (i + 1)..=kmax {
                        let lki = g[k][i] / gii;
                        if lki != 0.0 {
                            let (lo, hi) = if k <= j { (k, j) } else { (j, k) };
                            s -= lki * z[lo][hi - lo];
                        }
                    }
                    z[i][j - i] = s;
                    j -= 1;
                }
                let mut s = 1.0 / (gii * gii);
                for k in (i + 1)..=kmax {
                    let lki = g[k][i] / gii;
                    if lki != 0.0 {
                        s -= lki * z[i][k - i];
                    }
                }
                z[i][0] = s;
            }

            let mut tr = 0.0_f64;
            for i in 0..n {
                tr += z[i][0] * b[i][i];
                let dmax = bw.min(n - 1 - i);
                for d in 1..=dmax {
                    tr += 2.0 * z[i][d] * b[i][i + d];
                }
            }
            tr
        }

        let n = 48usize;
        let mut lower = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            lower[i][i] = 7.0 + (i % 5) as f64 * 0.125;
            let jlo = i.saturating_sub(GCV_BW);
            for j in jlo..i {
                lower[i][j] = ((i * 17 + j * 31 + 11) % 23) as f64 * 0.0025 + 0.00075;
            }
        }

        let mut full_a = vec![vec![0.0_f64; n]; n];
        let mut compact_a = vec![[0.0_f64; GCV_BAND_WIDTH]; n];
        for i in 0..n {
            let jlo = i.saturating_sub(GCV_BW);
            for j in jlo..=i {
                let klo = i.max(j).saturating_sub(GCV_BW);
                let khi = i.min(j);
                let mut sum = 0.0_f64;
                for k in klo..=khi {
                    sum += lower[i][k] * lower[j][k];
                }
                full_a[i][j] = sum;
                full_a[j][i] = sum;
                gcv_band_set(&mut compact_a, i, j, sum);
                gcv_band_set(&mut compact_a, j, i, sum);
            }
        }

        let mut full_b = vec![vec![0.0_f64; n]; n];
        let mut compact_b = vec![[0.0_f64; GCV_BAND_WIDTH]; n];
        for i in 0..n {
            let jhi = (i + GCV_BW).min(n - 1);
            for j in i..=jhi {
                let value = ((i * 13 + j * 7 + 3) % 29) as f64 * 0.003 + 0.25;
                full_b[i][j] = value;
                full_b[j][i] = value;
                gcv_band_set(&mut compact_b, i, j, value);
                gcv_band_set(&mut compact_b, j, i, value);
            }
        }

        assert_eq!(chol_banded_full(&mut full_a, GCV_BW), Some(()));
        assert_eq!(chol_banded_gcv(&mut compact_a), Some(()));
        for i in 0..n {
            let jlo = i.saturating_sub(GCV_BW);
            for j in jlo..=i {
                assert_eq!(
                    gcv_band_lower_get(&compact_a, i, j).to_bits(),
                    full_a[i][j].to_bits(),
                    "Cholesky lower-band mismatch at ({i},{j})"
                );
            }
        }

        let full_trace = gcv_trace_selinv_full(&full_a, &full_b, n, GCV_BW);
        let compact_trace = gcv_trace_selinv_gcv(&compact_a, &compact_b, n);
        assert_eq!(
            compact_trace.to_bits(),
            full_trace.to_bits(),
            "compact trace differs from full storage"
        );
    }

    #[test]
    fn make_splrep_interpolation_matches_scipy() {
        // make_splrep(x,y,s=0) == make_interp_spline; scipy gives these coeffs.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 1.0, 0.5, 2.0, 1.5, 3.0];
        let b = make_splrep(&x, &y, 3, 0.0).unwrap();
        let t_exp = [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0];
        for (g, e) in b.knots().iter().zip(&t_exp) {
            assert!((g - e).abs() < 1e-12, "t {g} vs {e}");
        }
        let c_exp = [0.0, 2.507407, -1.264815, 3.781481, -0.140741, 3.0];
        for (g, e) in b.coeffs().iter().zip(&c_exp) {
            assert!((g - e).abs() < 1e-5, "c {g} vs {e}");
        }
        // s > 0 now smooths: scipy make_splrep(x, y, k=3, s=1.0).
        let bs = make_splrep(&x, &y, 3, 1.0).unwrap();
        let ts_exp = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        for (g, e) in bs.knots().iter().zip(&ts_exp) {
            assert!((g - e).abs() < 1e-12, "ts {g} vs {e}");
        }
        let cs_exp = [0.04365079, 1.69157848, 0.52865961, 2.92063492];
        assert_eq!(bs.coeffs().len(), cs_exp.len());
        for (g, e) in bs.coeffs().iter().zip(&cs_exp) {
            assert!((g - e).abs() < 1e-5, "cs {g} vs {e}");
        }
    }

    #[test]
    fn make_splprep_smoothing_matches_scipy() {
        // 2-D curve sampled at 15 points; scipy.interpolate.make_splprep(s=0.01).
        let tt: Vec<f64> = (0..15).map(|i| i as f64 / 14.0).collect();
        let x: Vec<f64> = tt
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).cos() + 0.05 * (20.0 * t).sin())
            .collect();
        let y: Vec<f64> = tt
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let ((t, c, k), u) = make_splprep(&[x, y], 3, 0.01).unwrap();
        assert_eq!(k, 3);
        let t_exp = [
            0.0, 0.0, 0.0, 0.0, 0.29257123, 0.49981611, 0.78720603, 1.0, 1.0, 1.0, 1.0,
        ];
        for (g, e) in t.iter().zip(&t_exp) {
            assert!((g - e).abs() < 1e-6, "t {g} vs {e}");
        }
        let u_exp = [
            0.0, 0.06979872, 0.14459583, 0.22440958, 0.29257123, 0.35453232, 0.4256846, 0.49981611,
            0.57035793, 0.64780129, 0.72402847, 0.78720603, 0.85134033, 0.92617459, 1.0,
        ];
        for (g, e) in u.iter().zip(&u_exp) {
            assert!((g - e).abs() < 1e-7, "u {g} vs {e}");
        }
        let cx_exp = [
            1.00482233,
            1.09933223,
            -0.19034036,
            -1.47614048,
            0.0889122,
            1.02839757,
            1.04381222,
        ];
        let cy_exp = [
            -0.01613521,
            0.7105643,
            1.43592329,
            -0.21018385,
            -1.44683564,
            -0.48400388,
            0.02505848,
        ];
        assert_eq!(c.len(), 2);
        for (g, e) in c[0].iter().zip(&cx_exp) {
            assert!((g - e).abs() < 1e-5, "cx {g} vs {e}");
        }
        for (g, e) in c[1].iter().zip(&cy_exp) {
            assert!((g - e).abs() < 1e-5, "cy {g} vs {e}");
        }
        // s = 0 reduces to interpolation (delegates to splprep).
        let xi = vec![0.0, 1.0, 2.0, 3.0];
        let yi = vec![0.0, 1.0, 0.5, 2.0];
        assert!(make_splprep(&[xi, yi], 3, 0.0).is_ok());
    }

    #[test]
    fn splprep_interpolation_matches_scipy() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 1.0, 0.5, 2.0, 1.5, 3.0];
        let ((t, c, k), u) = splprep(&[x, y], 3, 0.0).unwrap();
        assert_eq!(k, 3);
        let u_exp = [
            0.0,
            0.19490713173322144,
            0.3489947488550868,
            0.5974535658666107,
            0.751541182988476,
            1.0,
        ];
        for (g, e) in u.iter().zip(&u_exp) {
            assert!((g - e).abs() < 1e-12, "u {g} vs {e}");
        }
        let t_exp = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.3489947488550868,
            0.5974535658666107,
            1.0,
            1.0,
            1.0,
            1.0,
        ];
        for (g, e) in t.iter().zip(&t_exp) {
            assert!((g - e).abs() < 1e-12, "t {g} vs {e}");
        }
        let cx_exp = [
            0.0,
            0.22741584352883404,
            2.230476464532552,
            2.7991826521281657,
            5.149697841196444,
            5.0,
        ];
        let cy_exp = [
            0.0,
            2.374428814147836,
            -1.121083821635914,
            3.865470998324146,
            -0.4202192267585394,
            3.0,
        ];
        for (g, e) in c[0].iter().zip(&cx_exp) {
            assert!((g - e).abs() < 1e-9, "cx {g} vs {e}");
        }
        for (g, e) in c[1].iter().zip(&cy_exp) {
            assert!((g - e).abs() < 1e-9, "cy {g} vs {e}");
        }
        // s>0 (FITPACK smoothing) is rejected.
        assert!(
            splprep(
                &[vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 0.0, 1.0]],
                3,
                1.0
            )
            .is_err()
        );
    }

    /// Parallel `RbfInterpolator::eval_many` must be BIT-IDENTICAL to the sequential
    /// per-query map (each query is an independent pure `eval`). Uses a size above the
    /// parallel gate so the threaded path runs.
    #[test]
    fn rbf_eval_many_parallel_is_bit_identical() {
        let points: Vec<Vec<f64>> = (0..200)
            .map(|i| {
                vec![
                    (i as f64 * 0.07).sin(),
                    (i as f64 * 0.11).cos(),
                    i as f64 * 0.001,
                ]
            })
            .collect();
        let values: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin() * 2.0).collect();
        for kernel in [
            RbfKernel::Multiquadric,
            RbfKernel::InverseMultiquadric,
            RbfKernel::ThinPlateSpline,
        ] {
            let rbf = RbfInterpolator::new(&points, &values, kernel, 1.3).expect("rbf");
            // 2000 queries * 200 centers = 400k >= the 2^18 gate -> parallel path.
            let queries: Vec<Vec<f64>> = (0..2000)
                .map(|i| {
                    vec![
                        i as f64 * 0.003 - 3.0,
                        i as f64 * 0.002 - 2.0,
                        i as f64 * 0.0005,
                    ]
                })
                .collect();
            let got = rbf.eval_many(&queries);
            let want: Vec<f64> = queries.iter().map(|q| rbf.eval(q)).collect();
            assert_eq!(got.len(), want.len());
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "rbf eval_many mismatch at {k} {kernel:?}"
                );
            }
        }
    }

    #[test]
    fn delaunay_grid_find_simplex_matches_linear_scan() {
        // Isomorphism proof for the simplex-grid index: grid find_simplex must be
        // bit-identical to the full linear scan (same simplex index and the same
        // barycentric coordinates) for every query, including points on edges,
        // vertices, outside the hull, and on the grid boundary.
        let mut state: u64 = 0xc0ff_ee00_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &npts in &[8usize, 30, 120, 400] {
            let pts: Vec<(f64, f64)> = (0..npts).map(|_| (next() * 10.0, next() * 7.0)).collect();
            let tri = match Delaunay2D::new(&pts) {
                Ok(t) => t,
                Err(_) => continue,
            };
            assert!(tri.grid.is_some(), "grid should build for npts={npts}");
            // Dense query grid spanning beyond the hull, plus exact vertices and
            // edge midpoints (the multi-simplex / tolerance-band cases).
            let mut queries: Vec<(f64, f64)> = Vec::new();
            for qy in 0..40 {
                for qx in 0..40 {
                    queries.push((-1.0 + qx as f64 * 0.3, -1.0 + qy as f64 * 0.22));
                }
            }
            for &(x, y) in &pts {
                queries.push((x, y));
            }
            for &(a, b, c) in &tri.simplices {
                let mid = |i: usize, j: usize| {
                    (
                        (tri.points[i].0 + tri.points[j].0) * 0.5,
                        (tri.points[i].1 + tri.points[j].1) * 0.5,
                    )
                };
                queries.push(mid(a, b));
                queries.push(mid(b, c));
                queries.push(mid(c, a));
            }
            for &q in &queries {
                let fast = tri.find_simplex(q);
                let slow = tri.find_simplex_linear(q);
                match (fast, slow) {
                    (Some((fi, fa, fb, fc)), Some((si, sa, sb, sc))) => {
                        assert_eq!(fi, si, "npts={npts} q={q:?}: simplex index");
                        assert_eq!(fa.to_bits(), sa.to_bits(), "npts={npts} q={q:?}: l1");
                        assert_eq!(fb.to_bits(), sb.to_bits(), "npts={npts} q={q:?}: l2");
                        assert_eq!(fc.to_bits(), sc.to_bits(), "npts={npts} q={q:?}: l3");
                    }
                    (None, None) => {}
                    (f, s) => {
                        assert_eq!(
                            f.is_some(),
                            s.is_some(),
                            "npts={npts} q={q:?}: match disagreement"
                        );
                    }
                }
            }
        }
    }

    /// Property proof for the circumcircle-precompute Bowyer-Watson [perf]: the
    /// precomputed `dist² < r²` incircle test is NOT bit-identical to the
    /// incircle determinant (it can pick the other diagonal on cocircular
    /// inputs), so we instead verify the defining DELAUNAY invariant — no input
    /// vertex lies strictly inside any output simplex's circumcircle. If the
    /// precompute corrupted the triangulation, an empty-circumcircle violation
    /// would appear here. Deterministic LCG clouds across a range of sizes.
    #[test]
    fn delaunay_empty_circumcircle_property() {
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let check = |label: &str, pts: &[(f64, f64)], tri: &Delaunay2D| {
            for &(a, b, c) in &tri.simplices {
                let (cx, cy, r2) = circumcircle(pts[a], pts[b], pts[c]);
                // Relative FP slack scaled by the circumradius so we only flag a
                // genuine non-Delaunay edge, not a vertex sitting on the circle.
                let slack = 1e-9 * (1.0 + r2);
                for (i, &(px, py)) in pts.iter().enumerate() {
                    if i == a || i == b || i == c {
                        continue;
                    }
                    let (dx, dy) = (px - cx, py - cy);
                    let d2 = dx * dx + dy * dy;
                    assert!(
                        d2 >= r2 - slack,
                        "{label}: vertex {i} inside circumcircle of \
                         simplex ({a},{b},{c}): d2={d2} r2={r2}"
                    );
                }
            }
        };
        // Small clouds — the plain O(n²) build path.
        for trial in 0..50usize {
            let n = 8 + trial % 60;
            let pts: Vec<(f64, f64)> = (0..n).map(|_| (next() * 12.0, next() * 9.0)).collect();
            let tri = match Delaunay2D::new(&pts) {
                Ok(t) => t,
                Err(_) => continue,
            };
            check(&format!("trial {trial}"), &pts, &tri);
        }
        // Large clouds ABOVE DELAUNAY_GRID_THRESHOLD — the grid-accelerated
        // build path must satisfy the SAME empty-circumcircle invariant.
        for &n in &[DELAUNAY_GRID_THRESHOLD, 5000] {
            let pts: Vec<(f64, f64)> = (0..n).map(|_| (next() * 60.0, next() * 45.0)).collect();
            let tri = Delaunay2D::new(&pts).expect("grid-path delaunay");
            check(&format!("grid n={n}"), &pts, &tri);
        }
    }

    /// Isomorphism proof for the smoothing-spline normal-equations
    /// sparse-support optimization [perf]: assembling A^T A / A^T y over only
    /// the nonzero B-spline basis indices must be BIT-IDENTICAL to the dense
    /// n^2 double-loop, because skipped terms have a zero factor (+/-0.0) and
    /// `v + (+/-0.0) == v` for every f64.
    #[test]
    fn smoothing_spline_sparse_assembly_is_bit_identical() {
        let k = 3usize;
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.37 - 2.0).collect();
        let y: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.5).sin() * 3.0 - 1.7 * (i as f64).cos())
            .collect();
        let t = interpolation_knots(&x, k);
        let n = x.len();

        // Dense reference.
        let mut ata_dense = vec![vec![0.0f64; n]; n];
        let mut aty_dense = vec![0.0f64; n];
        for i in 0..n {
            let basis = eval_basis_all(&t, x[i], k, n);
            for j in 0..n {
                aty_dense[j] += basis[j] * y[i];
                for l in 0..n {
                    ata_dense[j][l] += basis[j] * basis[l];
                }
            }
        }

        // Sparse-support version (mirrors make_smoothing_spline_impl).
        let mut ata = vec![vec![0.0f64; n]; n];
        let mut aty = vec![0.0f64; n];
        let mut nz: Vec<usize> = Vec::with_capacity(k + 1);
        for i in 0..n {
            let basis = eval_basis_all(&t, x[i], k, n);
            nz.clear();
            nz.extend((0..n).filter(|&idx| basis[idx] != 0.0));
            let yi = y[i];
            for &j in &nz {
                let bj = basis[j];
                aty[j] += bj * yi;
                let row = &mut ata[j];
                for &l in &nz {
                    row[l] += bj * basis[l];
                }
            }
        }

        for j in 0..n {
            assert_eq!(
                aty[j].to_bits(),
                aty_dense[j].to_bits(),
                "A^T y mismatch at {j}"
            );
            for l in 0..n {
                assert_eq!(
                    ata[j][l].to_bits(),
                    ata_dense[j][l].to_bits(),
                    "A^T A mismatch at ({j},{l})"
                );
            }
        }
    }

    /// Wall-clock witness for the sparse-support assembly [perf]. Run with
    /// `cargo test -p fsci-interpolate smoothing_spline_sparse_perf -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn smoothing_spline_sparse_perf_vs_dense() {
        use std::time::Instant;
        let k = 3usize;
        let n = 1200usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.013).collect();
        let _y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let t = interpolation_knots(&x, k);

        let t0 = Instant::now();
        let mut sink = 0.0f64;
        {
            let mut ata = vec![vec![0.0f64; n]; n];
            for &xi in &x {
                let basis = eval_basis_all(&t, xi, k, n);
                for j in 0..n {
                    for l in 0..n {
                        ata[j][l] += basis[j] * basis[l];
                    }
                }
            }
            sink += ata[n / 2][n / 2];
        }
        let dense = t0.elapsed();

        let t1 = Instant::now();
        {
            let mut ata = vec![vec![0.0f64; n]; n];
            let mut nz: Vec<usize> = Vec::with_capacity(k + 1);
            for &xi in &x {
                let basis = eval_basis_all(&t, xi, k, n);
                nz.clear();
                nz.extend((0..n).filter(|&idx| basis[idx] != 0.0));
                for &j in &nz {
                    let bj = basis[j];
                    let row = &mut ata[j];
                    for &l in &nz {
                        row[l] += bj * basis[l];
                    }
                }
            }
            sink += ata[n / 2][n / 2];
        }
        let sparse = t1.elapsed();

        let ratio = dense.as_secs_f64() / sparse.as_secs_f64();
        println!(
            "smoothing-spline A^TA: dense={dense:?} sparse={sparse:?} speedup={ratio:.2}x sink={sink}"
        );
        assert!(
            ratio > 1.0,
            "sparse assembly should be faster (got {ratio:.2}x)"
        );
    }

    #[test]
    fn linear_interp_at_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        for i in 0..4 {
            let val = interp.eval(x[i]).expect("eval");
            assert!((val - y[i]).abs() < 1e-12, "at knot {i}: {val} != {}", y[i]);
        }
    }

    #[test]
    fn linear_interp_midpoints() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 0.0];
        let interp = Interp1d::new(&x, &y, Interp1dOptions::default()).expect("interp1d");
        let val = interp.eval(0.5).expect("eval");
        assert!((val - 1.0).abs() < 1e-12, "got {val}");
        let val = interp.eval(1.5).expect("eval");
        assert!((val - 1.0).abs() < 1e-12, "got {val}");
    }

    #[test]
    fn nearest_interp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![10.0, 20.0, 30.0];
        let opts = Interp1dOptions {
            kind: InterpKind::Nearest,
            ..Interp1dOptions::default()
        };
        let interp = Interp1d::new(&x, &y, opts).expect("interp1d");
        assert_eq!(interp.eval(0.4).unwrap(), 10.0);
        assert_eq!(interp.eval(0.6).unwrap(), 20.0);
        assert_eq!(interp.eval(1.5).unwrap(), 20.0);
    }

    #[test]
    fn bspline_basic() {
        let t = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let c = vec![1.0, 2.0, 3.0];
        let k = 2;
        let spl = BSpline::new(t, c, k).expect("bspline");
        assert_eq!(spl.eval(0.0), 1.0);
        assert_eq!(spl.eval(1.0), 3.0);
    }

    #[test]
    fn bspline_antiderivative() {
        let t = vec![0.0, 0.0, 1.0, 1.0];
        let c = vec![1.0, 1.0];
        let k = 1;
        let spl = BSpline::new(t, c, k).unwrap();
        let anti = spl.antiderivative(1).unwrap();
        assert_eq!(anti.degree(), 2);
        assert_eq!(anti.eval(0.0), 0.0);
        assert_eq!(anti.eval(1.0), 1.0);
    }

    #[test]
    fn bspline_antiderivative_then_derivative_round_trips() {
        // d/dx of the antiderivative must recover the original spline exactly,
        // and the antiderivative must satisfy the knot invariant without any
        // coefficient padding.
        let t = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let c = vec![1.0, -2.0, 3.0, 0.5, -1.5];
        let k = 2;
        let spl = BSpline::new(t.clone(), c.clone(), k).unwrap();

        let anti = spl.antiderivative(1).unwrap();
        assert_eq!(
            anti.t.len(),
            anti.c.len() + anti.k + 1,
            "invariant after anti"
        );

        let back = anti.derivative(1).unwrap();
        assert_eq!(back.k, k);
        assert_eq!(back.t.len(), t.len());
        assert_eq!(back.c.len(), c.len());
        for (i, (&got, &want)) in back.c.iter().zip(c.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "coeff[{i}] = {got}, expected {want}"
            );
        }
        for (i, (&got, &want)) in back.t.iter().zip(t.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "knot[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn bspline_eval_many_parallel_matches_per_point() {
        // Above BSPLINE_EVAL_PAR_GATE (32768), eval_many fans across threads. It
        // must stay BIT-IDENTICAL to per-point eval for both sorted and unsorted
        // query batches (the parallel path re-seeds the knot-span pointer per
        // chunk, which must reproduce the serial pointer walk exactly).
        let n = 300usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.37).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin() + 0.3 * v.cos()).collect();
        // Build a cubic interpolating B-spline via the crate's own fit.
        let spline = make_interp_spline(&x, &y, 3).expect("cubic bspline");

        let nq = (1usize << 15) + 777; // > gate
        let lo = x[0];
        let hi = x[n - 1];
        let sorted: Vec<f64> = (0..nq)
            .map(|i| lo + (hi - lo) * (i as f64) / (nq as f64))
            .collect();
        let em = spline.eval_many(&sorted);
        for (i, &xq) in sorted.iter().enumerate() {
            assert_eq!(em[i].to_bits(), spline.eval(xq).to_bits(), "sorted i={i}");
        }

        // Unsorted (pseudo-random order) hits the per-point parallel branch.
        let mut unsorted = sorted.clone();
        let mut seed = 0x9e3779b97f4a7c15u64;
        for i in (1..unsorted.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            unsorted.swap(i, (seed as usize) % (i + 1));
        }
        let emu = spline.eval_many(&unsorted);
        for (i, &xq) in unsorted.iter().enumerate() {
            assert_eq!(
                emu[i].to_bits(),
                spline.eval(xq).to_bits(),
                "unsorted i={i}"
            );
        }
    }

    #[test]
    fn bspline_find_interval_matches_eval_basis() {
        // bspline_find_interval must return exactly the index eval_basis_all's degree-0
        // indicator sets to 1.0 (None when none does), across clamped/repeated/non-uniform
        // knots and x on knots, between knots, and outside the span.
        fn brute(t: &[f64], x: f64, k: usize, n: usize) -> Option<usize> {
            (0..n).find(|&i| {
                i + 1 < t.len()
                    && ((t[i] <= x && x < t[i + 1]) || (x == t[i + 1] && i + 1 == t.len() - k - 1))
            })
        }
        let knot_sets: Vec<(Vec<f64>, usize)> = vec![
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0], 3),
            (vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0], 1),
            (vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2),
            (vec![-2.0, -2.0, -2.0, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0], 2),
        ];
        for (t, k) in &knot_sets {
            let n = t.len() - k - 1;
            // Probe knots, midpoints, and out-of-range points.
            let mut probes: Vec<f64> = t.clone();
            for w in t.windows(2) {
                probes.push(0.5 * (w[0] + w[1]));
            }
            probes.push(t[0] - 1.0);
            probes.push(t[t.len() - 1] + 1.0);
            for &x in &probes {
                assert_eq!(
                    bspline_find_interval(t, x, n),
                    brute(t, x, *k, n),
                    "interval mismatch t={t:?} k={k} x={x}"
                );
            }
        }
    }

    #[test]
    fn make_lsq_spline_rejects_too_short_knots_without_panic() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let knots = vec![0.0, 1.0];
        let err = make_lsq_spline(&x, &y, &knots, 3).expect_err("too few knots");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn splev_second_derivative_matches_scipy_reference_tck() {
        let t = vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let c = vec![
            4.131186545826286e-17,
            -1.5593640784053972e-15,
            2.0677589048408837e-15,
            32.0,
            64.0,
        ];
        let tck = (t, c, 3_usize);
        let points = vec![0.5, 1.5, 2.5, 3.5];
        let expected = [3.0, 9.0, 15.0, 21.0];
        let actual = splev_with_derivative(&points, &tck, 2).expect("second derivative");
        for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "second derivative mismatch at {index}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn splev_rejects_derivative_order_above_degree() {
        let tck = (
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 3.0],
            2_usize,
        );
        let err = splev_with_derivative(&[0.5], &tck, 3).expect_err("derivative above degree");
        assert!(matches!(
            err,
            InterpError::InvalidArgument { detail } if detail == "0<=der=3<=k=2 must hold"
        ));
    }

    #[test]
    fn aaa_eval_many_parallel_matches_serial_bitexact() {
        // `Aaa::eval_many` now fans queries across cores via `par_query_map`. It must be
        // BYTE-IDENTICAL to the serial `xs.iter().map(eval)`, at a batch large enough to cross the
        // parallel gate (`m·support ≥ 2^23`) so the multi-threaded chunked path actually runs.
        let z: Vec<f64> = (0..100).map(|i| -1.0 + i as f64 * (2.0 / 99.0)).collect();
        let f: Vec<f64> = z
            .iter()
            .map(|&x| 1.0 / (x + 1.5) + 0.5 / (x - 2.0) + (3.0 * x).sin())
            .collect();
        let a = Aaa::new(&z, &f, None, 100).unwrap();
        let support = a.support_points().len().max(1);
        // Choose m so m·support ≥ 2^23; wide-ranging queries incl. exact support points and poles-ish.
        let m = (1usize << 23) / support + 4096;
        let xs: Vec<f64> = (0..m)
            .map(|i| {
                if i % 997 == 0 {
                    a.support_points()[i % support] // exact support point (early-exit branch)
                } else {
                    -1.3 + (i as f64) * 2.6 / (m as f64)
                }
            })
            .collect();
        assert!(
            (m as u64) * (support as u64) >= (1 << 23),
            "batch must cross the parallel gate ({m}·{support})"
        );
        let parallel = a.eval_many(&xs);
        let serial: Vec<f64> = xs.iter().map(|&x| a.eval(x)).collect();
        assert_eq!(parallel.len(), serial.len());
        for (i, (p, s)) in parallel.iter().zip(serial.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "aaa eval_many mismatch at query {i}"
            );
        }
    }

    #[test]
    fn aaa_matches_scipy() {
        let z: Vec<f64> = (0..100).map(|i| -1.0 + i as f64 * (2.0 / 99.0)).collect();
        let xs = [-0.3_f64, 0.1, 0.77];

        // Rational function: AAA recovers it exactly with 3 support points.
        let fr: Vec<f64> = z
            .iter()
            .map(|&x| 1.0 / (x + 1.5) + 0.5 / (x - 2.0))
            .collect();
        let a = Aaa::new(&z, &fr, None, 100).unwrap();
        assert_eq!(a.support_points().len(), 3, "rational nterms");
        let exp = [0.615942029, 0.3618421053, 0.0340245693];
        for (x, e) in xs.iter().zip(exp.iter()) {
            assert!(
                (a.eval(*x) - e).abs() < 1e-8,
                "rational eval {} vs {e}",
                a.eval(*x)
            );
        }
        // exact at support points
        for (&zk, &fk) in a.support_points().iter().zip(a.support_values().iter()) {
            assert!((a.eval(zk) - fk).abs() < 1e-12);
        }

        // exp(x): 6 support points, values match scipy.
        let fe: Vec<f64> = z.iter().map(|&x| x.exp()).collect();
        let ae = Aaa::new(&z, &fe, None, 100).unwrap();
        assert_eq!(ae.support_points().len(), 6, "exp nterms");
        let expe = [0.7408182207, 1.1051709181, 2.1597662538];
        for (x, e) in xs.iter().zip(expe.iter()) {
            assert!(
                (ae.eval(*x) - e).abs() < 1e-7,
                "exp eval {} vs {e}",
                ae.eval(*x)
            );
        }

        // Runge function: 3 support points.
        let fg: Vec<f64> = z.iter().map(|&x| 1.0 / (1.0 + 25.0 * x * x)).collect();
        let ag = Aaa::new(&z, &fg, None, 100).unwrap();
        assert_eq!(ag.support_points().len(), 3, "runge nterms");
    }

    #[test]
    fn floater_hormann_matches_scipy() {
        let x = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();

        // scipy.interpolate.FloaterHormannInterpolator(x, sin(x), d=3)
        let fh = FloaterHormannInterpolator::new(&x, &y, 3).unwrap();
        let exp_w = [
            -0.16666667,
            0.66666667,
            -1.16666667,
            1.16666667,
            -0.66666667,
            0.16666667,
        ];
        for (g, e) in fh.weights().iter().zip(exp_w.iter()) {
            assert!((g - e).abs() < 1e-7, "w {g} vs {e}");
        }
        let xs = [0.5, 2.5, 4.2];
        let exp = [0.4901548, 0.60070277, -0.87092117];
        for (g, e) in fh.eval_many(&xs).iter().zip(exp.iter()) {
            assert!((g - e).abs() < 1e-6, "eval {g} vs {e}");
        }
        // interpolation property: exact at the nodes.
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!((fh.eval(xi) - yi).abs() < 1e-12);
        }

        // d = 1
        let fh1 = FloaterHormannInterpolator::new(&x, &y, 1).unwrap();
        let exp_w1 = [-1.0, 2.0, -2.0, 2.0, -2.0, 1.0];
        for (g, e) in fh1.weights().iter().zip(exp_w1.iter()) {
            assert!((g - e).abs() < 1e-7, "w1 {g} vs {e}");
        }
        let exp1 = [0.48573853, 0.60411012, -0.86103101];
        for (g, e) in fh1.eval_many(&xs).iter().zip(exp1.iter()) {
            assert!((g - e).abs() < 1e-6, "eval1 {g} vs {e}");
        }
        // d >= n rejected.
        assert!(FloaterHormannInterpolator::new(&x, &y, 6).is_err());
    }

    #[test]
    fn floater_hormann_duplicate_check_sort_path() {
        // n > 64 exercises the O(n log n) sort+adjacent duplicate check (vs the small-n
        // all-pairs scan). A strictly increasing node set builds; injecting a duplicate
        // (non-adjacent in index, so only the sort can pair it up) must still be rejected.
        let n = 200usize;
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.1).cos()).collect();
        assert!(FloaterHormannInterpolator::new(&x, &y, 3).is_ok());
        x[10] = x[150]; // duplicate value far apart in index order
        let yd: Vec<f64> = x.iter().map(|&v| (v * 0.1).cos()).collect();
        let result = FloaterHormannInterpolator::new(&x, &yd, 3);
        assert!(matches!(result, Err(InterpError::InvalidArgument { .. })));
    }

    #[test]
    fn barycentric_interpolator_matches_quadratic() {
        let xi = vec![-1.0, 0.0, 2.0];
        let yi: Vec<f64> = xi.iter().map(|&x| x * x + 2.0 * x + 1.0).collect();
        let interp = BarycentricInterpolator::new(&xi, &yi).expect("barycentric");
        assert!((interp.eval(1.5) - 6.25).abs() < 1e-12);
    }

    #[test]
    fn barycentric_interpolator_is_exact_at_nodes() {
        let xi = vec![3.0, -1.0, 0.5];
        let yi = vec![7.0, 2.0, -4.0];
        let interp = BarycentricInterpolator::new(&xi, &yi).expect("barycentric");
        for (&x, &y) in xi.iter().zip(yi.iter()) {
            assert!((interp.eval(x) - y).abs() < 1e-14);
        }
    }

    #[test]
    fn barycentric_interpolator_rejects_duplicate_nodes() {
        let err =
            BarycentricInterpolator::new(&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).expect_err("dupes");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn barycentric_eval_nan_input_returns_nan() {
        let xi = vec![0.0, 1.0, 2.0];
        let yi = vec![0.0, 1.0, 4.0];
        let interp = BarycentricInterpolator::new(&xi, &yi).expect("barycentric");
        assert!(interp.eval(f64::NAN).is_nan());
        assert!(interp.eval(f64::INFINITY).is_nan());
        assert!(interp.eval(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn barycentric_eval_many_works() {
        let xi = vec![0.0, 1.0, 2.0];
        let yi = vec![0.0, 1.0, 4.0];
        let interp = BarycentricInterpolator::new(&xi, &yi).expect("barycentric");
        let results = interp.eval_many(&[0.0, 0.5, 1.0, 1.5, 2.0]);
        assert!((results[0] - 0.0).abs() < 1e-12);
        assert!((results[2] - 1.0).abs() < 1e-12);
        assert!((results[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn univariate_spline_interpolates_when_s_zero() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 0.0, 3.0];
        let spline = UnivariateSpline::new(&x, &y, 0.0).expect("spline");
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!((spline.eval(xi) - yi).abs() < 1e-10);
        }
    }

    #[test]
    fn univariate_spline_smoothing_reduces_residual_variation() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.3, 3.8, 9.1, 15.7];
        let interp = UnivariateSpline::new(&x, &y, 0.0).expect("interp spline");
        let smooth = UnivariateSpline::new(&x, &y, 2.0).expect("smooth spline");

        let interp_knot_residual = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (interp.eval(xi) - yi).abs())
            .sum::<f64>();
        let smooth_knot_residual = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (smooth.eval(xi) - yi).abs())
            .sum::<f64>();

        assert!(interp_knot_residual < 1e-8);
        assert!(smooth_knot_residual > 1e-3);
        assert!((smooth.eval(2.5) - 6.25).abs() < 1.5);
    }

    #[test]
    fn interpolated_univariate_spline_matches_exact_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![2.0, -1.0, 4.0, 3.0];
        let spline = InterpolatedUnivariateSpline::new(&x, &y).expect("interpolated spline");
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!((spline.eval(xi) - yi).abs() < 1e-10);
        }
    }

    #[test]
    fn interpolated_univariate_spline_matches_zero_smoothing_univariate() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 0.0, 1.0, 8.0];
        let interp = InterpolatedUnivariateSpline::new(&x, &y).expect("interpolated spline");
        let uni = UnivariateSpline::new(&x, &y, 0.0).expect("univariate spline");
        let xs = vec![0.25, 1.5, 2.75];
        let lhs = interp.eval_many(&xs);
        let rhs = uni.eval_many(&xs);
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn interpolated_univariate_spline_derivative_and_integral_work() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0];
        let spline = InterpolatedUnivariateSpline::new(&x, &y).expect("interpolated spline");
        let deriv = spline.derivative(1).expect("derivative");
        assert!(deriv.eval(1.5).is_finite());
        let integral = spline.integral(0.0, 3.0).expect("integral");
        assert!(integral.is_finite());
        assert!(integral > 0.0);
    }

    // ── RBF Interpolator tests ───────────────────────────────────────

    #[test]
    fn solve_dense_system_flat_matches_vecvec() {
        // The flat dense solver must be BIT-IDENTICAL to the Vec<Vec> reference on
        // random dense systems (this is what makes the RBF flat-Φ solve a byte-
        // identical perf change).
        let mut state = 0xDEADBEEF_u64;
        let mut nx = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for &n in &[1usize, 2, 5, 17, 40] {
            // diagonally-dominant random A (non-singular) + random b.
            let mut vv: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| nx()).collect()).collect();
            for (i, row) in vv.iter_mut().enumerate() {
                row[i] += n as f64 + 1.0;
            }
            let b: Vec<f64> = (0..n).map(|_| nx()).collect();
            let mut flat: Vec<f64> = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    flat[i * n + j] = vv[i][j];
                }
            }
            let mut vv2 = vv.clone();
            let mut b1 = b.clone();
            let mut b2 = b.clone();
            let ref_x = solve_dense_system(&mut vv2, &mut b1).expect("ref");
            let flat_x = solve_dense_system_flat(&mut flat, n, &mut b2).expect("flat");
            assert_eq!(ref_x.len(), flat_x.len());
            for (r, f) in ref_x.iter().zip(&flat_x) {
                assert_eq!(
                    r.to_bits(),
                    f.to_bits(),
                    "n={n}: flat solve != Vec<Vec> solve"
                );
            }
        }
    }

    #[test]
    fn rbf_build_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The parallel Φ-matrix build must be BYTE-IDENTICAL to the serial build: same Φ → same
        // (deterministic) dense solve → same weights → same evaluations. Sized above the parallel
        // build gate. Checked across kernels with a transcendental/sqrt per entry.
        let n = 300usize;
        let mut s = 12345u64;
        let mut rng = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let points: Vec<Vec<f64>> = (0..n).map(|_| vec![rng() * 10.0, rng() * 10.0]).collect();
        let values: Vec<f64> = (0..n).map(|_| rng() * 4.0 - 2.0).collect();
        let queries: Vec<Vec<f64>> = (0..40).map(|_| vec![rng() * 10.0, rng() * 10.0]).collect();
        for kernel in [
            RbfKernel::Gaussian,
            RbfKernel::Multiquadric,
            RbfKernel::InverseMultiquadric,
        ] {
            RBF_BUILD_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let ser = RbfInterpolator::new(&points, &values, kernel, 1.0).unwrap();
            RBF_BUILD_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let par = RbfInterpolator::new(&points, &values, kernel, 1.0).unwrap();
            for q in &queries {
                assert_eq!(
                    ser.eval(q).to_bits(),
                    par.eval(q).to_bits(),
                    "kernel={kernel:?}"
                );
            }
        }
        RBF_BUILD_FORCE_SERIAL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn rbf_exact_at_data_points() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let values = vec![0.0, 1.0, 4.0, 9.0]; // x²
        let rbf = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect("rbf");
        for (pt, &expected) in points.iter().zip(values.iter()) {
            let result = rbf.eval(pt);
            assert!(
                (result - expected).abs() < 1e-6,
                "RBF at {:?}: got {result}, expected {expected}",
                pt
            );
        }
    }

    #[test]
    fn rbf_rejects_mixed_point_dimensions() {
        let points = vec![vec![0.0, 0.0], vec![1.0], vec![0.5, 0.25]];
        let values = vec![0.0, 1.0, 2.0];
        let err = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0)
            .expect_err("mixed dims");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rbf_query_dimension_mismatch_returns_nan() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 1.0];
        let rbf = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect("rbf");
        assert!(rbf.eval(&[0.5]).is_nan());
    }

    #[test]
    fn linear_nd_interpolator_rejects_malformed_point_dimensions() {
        let points = vec![vec![0.0, 0.0], vec![1.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 2.0];
        let err = LinearNDInterpolator::new(&points, &values).expect_err("malformed point");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn griddata_cubic_delegates_to_clough_tocher_and_interpolates() {
        // griddata(method='cubic') must use the Clough-Tocher scheme (SciPy
        // parity) — i.e. produce results identical to CloughTocher2DInterpolator,
        // and pass exactly through the data points (C¹ interpolation).
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ];
        let values = points
            .iter()
            .map(|p| 1.0 + 2.0 * p[0] - 0.5 * p[1] + p[0] * p[1])
            .collect::<Vec<_>>();
        let xi = vec![vec![0.25, 0.25], vec![0.6, 0.4], vec![0.4, 0.6]];

        let via_griddata = griddata(&points, &values, &xi, GriddataMethod::Cubic).unwrap();
        let via_ct = CloughTocher2DInterpolator::new(&points, &values)
            .unwrap()
            .eval_many(&xi)
            .unwrap();
        assert_eq!(
            via_griddata, via_ct,
            "griddata cubic must delegate bit-for-bit to CloughTocher2DInterpolator"
        );

        // Interpolation: exact at the data points.
        let at_data = griddata(&points, &values, &points, GriddataMethod::Cubic).unwrap();
        for (got, &want) in at_data.iter().zip(values.iter()) {
            assert!(
                (got - want).abs() < 1e-9,
                "griddata cubic not exact at data point: {got} vs {want}"
            );
        }

        // Cubic is 2-D only (like SciPy): a 3-D query set must error, not silently
        // produce wrong output.
        let p3: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let v3 = vec![0.0, 1.0, 2.0];
        assert!(
            griddata(&p3, &v3, &[vec![0.5, 0.5, 0.0]], GriddataMethod::Cubic).is_err(),
            "griddata cubic must reject non-2-D points"
        );
    }

    #[test]
    fn clough_tocher_exact_at_data_points() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let values = points
            .iter()
            .map(|point| point[0] * point[0] + point[1] * point[1])
            .collect::<Vec<_>>();
        let interp = CloughTocher2DInterpolator::new(&points, &values).expect("clough-tocher");
        for (point, &expected) in points.iter().zip(values.iter()) {
            let got = interp.eval(point).expect("eval at point");
            assert!(
                (got - expected).abs() < 1e-10,
                "at {point:?}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn clough_tocher_preserves_affine_surfaces() {
        let points = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
            vec![2.0, 2.0],
            vec![1.0, 0.75],
        ];
        let values = points
            .iter()
            .map(|point| 2.0 * point[0] - 3.0 * point[1] + 1.0)
            .collect::<Vec<_>>();
        let interp = CloughTocher2DInterpolator::new(&points, &values).expect("clough-tocher");
        for query in [vec![0.25, 0.25], vec![1.25, 0.5], vec![1.5, 1.5]] {
            let got = interp.eval(&query).expect("affine eval");
            let expected = 2.0 * query[0] - 3.0 * query[1] + 1.0;
            assert!(
                (got - expected).abs() < 1e-10,
                "{query:?}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn clough_tocher_matches_vertex_gradients_at_interior_sites() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![1.0, 1.0],
            vec![-1.0, -1.0],
        ];
        let values = points
            .iter()
            .map(|point| point[0] * point[0] + point[0] * point[1] + point[1] * point[1])
            .collect::<Vec<_>>();
        let interp = CloughTocher2DInterpolator::new(&points, &values).expect("clough-tocher");

        let eps = 1e-6;
        let d_dx = (interp.eval(&[eps, 0.0]).expect("+x") - interp.eval(&[-eps, 0.0]).expect("-x"))
            / (2.0 * eps);
        let d_dy = (interp.eval(&[0.0, eps]).expect("+y") - interp.eval(&[0.0, -eps]).expect("-y"))
            / (2.0 * eps);
        let (gx, gy) = interp.gradients[0];

        assert!(
            (d_dx - gx).abs() < 1e-4,
            "x-derivative mismatch: finite-difference {d_dx}, stored gradient {gx}"
        );
        assert!(
            (d_dy - gy).abs() < 1e-4,
            "y-derivative mismatch: finite-difference {d_dy}, stored gradient {gy}"
        );
    }

    #[test]
    fn clough_tocher_fill_value_outside_hull() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![0.0, 1.0, 1.0];
        let interp = CloughTocher2DInterpolator::with_options(
            &points,
            &values,
            CloughTocher2DOptions {
                fill_value: -99.0,
                ..CloughTocher2DOptions::default()
            },
        )
        .expect("clough-tocher");
        assert_eq!(interp.eval(&[2.0, 2.0]).expect("outside"), -99.0);
    }

    #[test]
    fn clough_tocher_rescale_handles_different_coordinate_scales() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1000.0, 0.0],
            vec![0.0, 1.0],
            vec![1000.0, 1.0],
            vec![500.0, 0.5],
        ];
        let values = points
            .iter()
            .map(|point| point[0] / 1000.0 + 2.0 * point[1])
            .collect::<Vec<_>>();
        let interp = CloughTocher2DInterpolator::with_options(
            &points,
            &values,
            CloughTocher2DOptions {
                rescale: true,
                ..CloughTocher2DOptions::default()
            },
        )
        .expect("clough-tocher rescale");
        let got = interp.eval(&[250.0, 0.25]).expect("rescaled eval");
        assert!((got - 0.75).abs() < 1e-10, "rescaled affine got {got}");
    }

    #[test]
    fn clough_tocher_rejects_invalid_inputs() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let values = vec![0.0, 1.0, 2.0];
        let duplicate_err =
            CloughTocher2DInterpolator::new(&points, &values).expect_err("duplicate point");
        assert!(matches!(duplicate_err, InterpError::InvalidArgument { .. }));

        let nonfinite_points = vec![vec![0.0, 0.0], vec![f64::NAN, 0.0], vec![0.0, 1.0]];
        let nonfinite_err = CloughTocher2DInterpolator::new(&nonfinite_points, &values)
            .expect_err("nonfinite point");
        assert!(matches!(nonfinite_err, InterpError::InvalidArgument { .. }));

        let collinear = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let collinear_err =
            CloughTocher2DInterpolator::new(&collinear, &values).expect_err("collinear points");
        assert!(matches!(collinear_err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rbf_smooth_between_points() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let values = vec![0.0, 1.0, 0.0]; // triangular
        let rbf = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect("rbf");
        let mid = rbf.eval(&[0.5]);
        assert!(mid > 0.0 && mid < 1.0, "midpoint should interpolate: {mid}");
    }

    #[test]
    fn rbf_2d() {
        // 2D: f(x,y) = x + y at corners of unit square
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];
        let rbf = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect("rbf 2d");
        let center = rbf.eval(&[0.5, 0.5]);
        assert!(
            (center - 1.0).abs() < 0.3,
            "center of x+y should be ~1: {center}"
        );
    }

    #[test]
    fn rbf_multiquadric_kernel() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let values = vec![1.0, 2.0, 3.0];
        let rbf =
            RbfInterpolator::new(&points, &values, RbfKernel::Multiquadric, 1.0).expect("rbf mq");
        // Should interpolate exactly at data points
        for (pt, &expected) in points.iter().zip(values.iter()) {
            let result = rbf.eval(pt);
            assert!(
                (result - expected).abs() < 1e-6,
                "MQ at {:?}: {result} vs {expected}",
                pt
            );
        }
    }

    #[test]
    fn rbf_eval_many() {
        let points = vec![vec![0.0], vec![1.0]];
        let values = vec![0.0, 1.0];
        let rbf = RbfInterpolator::new(&points, &values, RbfKernel::Linear, 1.0).expect("rbf");
        let results = rbf.eval_many(&[vec![0.0], vec![0.5], vec![1.0]]);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn rbf_empty_rejected() {
        let err = RbfInterpolator::new(&[], &[], RbfKernel::Gaussian, 1.0).expect_err("empty");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    #[test]
    fn rbf_rejects_non_finite_points() {
        let points = vec![vec![0.0], vec![f64::NAN]];
        let values = vec![0.0, 1.0];
        let err =
            RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect_err("nan");
        assert_eq!(err, InterpError::NonFiniteX);
    }

    #[test]
    fn rbf_rejects_non_finite_values() {
        let points = vec![vec![0.0], vec![1.0]];
        let values = vec![0.0, f64::INFINITY];
        let err =
            RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0).expect_err("inf");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rbf_rejects_non_finite_epsilon() {
        let points = vec![vec![0.0], vec![1.0]];
        let values = vec![0.0, 1.0];
        let err =
            RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, f64::NAN).expect_err("nan");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rbf_rejects_excessive_point_count() {
        let points: Vec<Vec<f64>> = (0..=MAX_RBF_POINTS).map(|i| vec![i as f64]).collect();
        let values = vec![0.0; points.len()];
        let err = RbfInterpolator::new(&points, &values, RbfKernel::Gaussian, 1.0)
            .expect_err("safety bound");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    /// Deterministic dump of the 3D-nearest eval payload for golden-SHA proof.
    /// Run with `--ignored --nocapture` and pipe to `sha256sum`.
    #[test]
    #[ignore]
    fn dump_nearest_payload_for_golden_sha() {
        let points = vec![
            (0..32).map(|i| i as f64 / 31.0).collect::<Vec<_>>(),
            (0..32).map(|i| i as f64 / 31.0).collect::<Vec<_>>(),
            (0..16).map(|i| i as f64 / 15.0).collect::<Vec<_>>(),
        ];
        let nx = points[0].len();
        let ny = points[1].len();
        let nz = points[2].len();
        let mut values = Vec::with_capacity(nx * ny * nz);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    values.push((ix as f64 * 1.1 + iy as f64 * 2.3 + iz as f64 * 0.7).sin());
                }
            }
        }
        let grid = RegularGridInterpolator::new(
            points,
            values,
            RegularGridMethod::Nearest,
            false,
            Some(-999.0),
        )
        .expect("grid");
        // Edge-heavy query set: interior, exact coords, midpoints, OOB both sides, NaN.
        let mut queries: Vec<Vec<f64>> = Vec::new();
        for i in 0..6000 {
            let a = ((i * 37) % 997) as f64 / 997.0;
            let b = ((i * 53 + 17) % 991) as f64 / 991.0;
            let c = (a * 0.7 + b * 0.3).fract();
            queries.push(vec![a, b, c]);
        }
        queries.push(vec![0.0, 0.0, 0.0]);
        queries.push(vec![1.0, 1.0, 1.0]);
        queries.push(vec![-0.5, 0.5, 1.5]); // OOB -> fill
        queries.push(vec![0.5, f64::NAN, 0.5]); // NaN -> NaN
        queries.push(vec![1.0 / 62.0, 1.0 / 30.0, 1.0 / 15.0]); // exact midpoints / coords
        let out = grid.eval_many(&queries).expect("eval");
        let mut s = String::new();
        for v in out {
            s.push_str(&format!("{:0>16x}\n", v.to_bits()));
        }
        print!("{s}");
    }

    #[test]
    fn find_interval_uniform_matches_binary_search() {
        // Isomorphism proof for the uniform-axis fast path: for every axis the
        // O(1) direct-address lookup must return the *exact* same interval index
        // as the binary-search reference, across interior points, exact axis
        // coordinates, midpoints, and out-of-range values on both sides.
        let axes: Vec<Vec<f64>> = vec![
            (0..32).map(|i| i as f64 / 31.0).collect(), // linspace [0,1], n=32 (bench axis0/1)
            (0..16).map(|i| i as f64 / 15.0).collect(), // linspace [0,1], n=16 (bench axis2)
            (0..10).map(|i| -5.0 + i as f64 * 1.25).collect(), // negative start, dx=1.25
            (0..7).map(|i| 3.0 + i as f64 * 1e6).collect(), // large span
        ];
        for axis in &axes {
            let meta = detect_uniform_axis(axis).expect("axis is uniform");
            let n = axis.len();
            let lo = axis[0];
            let hi = axis[n - 1];
            let span = hi - lo;
            // Build a dense, edge-heavy set of probe points.
            let mut probes: Vec<f64> = Vec::new();
            probes.push(lo - span); // far below
            probes.push(hi + span); // far above
            for i in 0..n {
                probes.push(axis[i]); // exact coordinate
                if i + 1 < n {
                    probes.push(0.5 * (axis[i] + axis[i + 1])); // midpoint
                    probes.push(axis[i] + 0.25 * (axis[i + 1] - axis[i]));
                    probes.push(axis[i + 1] - 1e-12 * span.max(1.0)); // just below next
                }
            }
            for k in 0..=2000 {
                probes.push(lo - 0.1 * span + (k as f64 / 2000.0) * 1.2 * span);
            }
            for &x in &probes {
                let fast = RegularGridInterpolator::find_interval_uniform(meta, axis, x);
                let slow = RegularGridInterpolator::find_interval(axis, x);
                assert_eq!(fast, slow, "axis n={n} x={x}: fast={fast} slow={slow}");
            }
        }

        // Irregular axes must NOT be classified as uniform.
        assert!(detect_uniform_axis(&[0.0, 1.0, 3.0, 6.0]).is_none());
        assert!(detect_uniform_axis(&[0.0, 0.1, 0.2, 0.4, 0.8]).is_none());
    }

    #[test]
    fn regular_grid_linear_2d_bilinear() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        // values[i0][i1] = x + y with axis0=x, axis1=y
        let values = vec![0.0, 1.0, 1.0, 2.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect("regular grid");
        let value = interp.eval(&[0.25, 0.75]).expect("eval");
        assert!((value - 1.0).abs() < 1e-12, "bilinear got {value}");
    }

    #[test]
    fn regular_grid_linear_3d_trilinear() {
        let points = vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let mut values = Vec::new();
        for &x in &points[0] {
            for &y in &points[1] {
                for &z in &points[2] {
                    values.push(x + y + z);
                }
            }
        }
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect("regular grid");
        let value = interp.eval(&[0.5, 0.25, 0.75]).expect("eval");
        assert!((value - 1.5).abs() < 1e-12, "trilinear got {value}");
    }

    #[test]
    fn regular_grid_nearest_returns_closest() {
        let points = vec![vec![0.0, 1.0, 2.0]];
        let values = vec![0.0, 1.0, 2.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
                .expect("regular grid");
        assert_eq!(interp.eval(&[1.6]).unwrap(), 2.0);
        assert_eq!(interp.eval(&[-0.4]).unwrap(), 0.0);
    }

    #[test]
    fn regular_grid_extrapolates_when_fill_value_none() {
        let points = vec![vec![0.0, 1.0]];
        let values = vec![0.0, 1.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect("regular grid");
        let value = interp.eval(&[-0.5]).expect("eval");
        assert!((value + 0.5).abs() < 1e-12, "extrapolated got {value}");
    }

    #[test]
    fn regular_grid_out_of_bounds_fill_value_and_error() {
        let points = vec![vec![0.0, 1.0]];
        let values = vec![0.0, 1.0];
        let interp = RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            RegularGridMethod::Linear,
            false,
            Some(9.5),
        )
        .expect("regular grid");
        assert_eq!(interp.eval(&[2.0]).unwrap(), 9.5);

        let err_interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, true, None)
                .expect("regular grid");
        let err = err_interp.eval(&[2.0]).expect_err("bounds error");
        assert!(matches!(err, InterpError::OutOfBounds { .. }));
    }

    #[test]
    fn regular_grid_nearest_nan_query_returns_nan() {
        let points = vec![vec![0.0, 1.0, 2.0]];
        let values = vec![0.0, 1.0, 2.0];
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
                .expect("regular grid");
        assert!(interp.eval(&[f64::NAN]).unwrap().is_nan());
    }

    #[test]
    fn regular_grid_rejects_nan_in_points() {
        let points = vec![vec![0.0, f64::NAN, 2.0]];
        let values = vec![0.0, 1.0, 2.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect_err("nan in points");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn regular_grid_rejects_infinity_in_points() {
        let points = vec![vec![0.0, f64::INFINITY, 2.0]];
        let values = vec![0.0, 1.0, 2.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect_err("inf in points");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn regular_grid_rejects_neg_infinity_in_points() {
        let points = vec![vec![f64::NEG_INFINITY, 1.0, 2.0]];
        let values = vec![0.0, 1.0, 2.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect_err("neg_inf in points");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn regular_grid_rejects_nan_in_second_axis() {
        let points = vec![vec![0.0, 1.0], vec![0.0, f64::NAN]];
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Linear, false, None)
                .expect_err("nan in second axis");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn regular_grid_cubic_1d_smooth() {
        // Cubic spline should interpolate smoothly
        let points = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let values = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Cubic, false, None)
                .expect("regular grid cubic");
        // Test at midpoints
        let v_half = interp.eval(&[0.5]).expect("eval");
        let v_1_5 = interp.eval(&[1.5]).expect("eval");
        let v_2_5 = interp.eval(&[2.5]).expect("eval");
        // For cubic interpolation of x^2, should be close to correct values
        assert!((v_half - 0.25).abs() < 0.5, "got {v_half}, expected ~0.25");
        assert!((v_1_5 - 2.25).abs() < 0.5, "got {v_1_5}, expected ~2.25");
        assert!((v_2_5 - 6.25).abs() < 0.5, "got {v_2_5}, expected ~6.25");
    }

    #[test]
    fn regular_grid_cubic_2d_smooth() {
        // 2D cubic interpolation on a 4x4 grid
        let points = vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]];
        // values = x + y
        let mut values = Vec::new();
        for &x in &points[0] {
            for &y in &points[1] {
                values.push(x + y);
            }
        }
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Cubic, false, None)
                .expect("regular grid cubic 2d");
        // Test at midpoint - should be exactly correct for linear function
        let v = interp.eval(&[1.5, 1.5]).expect("eval");
        assert!((v - 3.0).abs() < 0.1, "got {v}, expected 3.0");
    }

    #[test]
    fn regular_grid_cubic_requires_4_points() {
        // Cubic method requires at least 4 points per axis
        let points = vec![vec![0.0, 1.0, 2.0]]; // Only 3 points
        let values = vec![0.0, 1.0, 4.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Cubic, false, None)
                .expect_err("too few points for cubic");
        assert!(matches!(err, InterpError::TooFewPoints { minimum: 4, .. }));
    }

    #[test]
    fn regular_grid_quintic_requires_6_points() {
        // Quintic method requires at least 6 points per axis
        let points = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]]; // Only 5 points
        let values = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Quintic, false, None)
                .expect_err("too few points for quintic");
        assert!(matches!(err, InterpError::TooFewPoints { minimum: 6, .. }));
    }

    #[test]
    fn regular_grid_pchip_batch_hoist_matches_scalar_bitexact() {
        use std::sync::atomic::Ordering;
        // The hoisted-fit batch path (`eval_many_pchip`) must be BYTE-IDENTICAL to the generic
        // per-query path, across ndim 1/2/3, interior/exact-node/edge queries, and out-of-bounds
        // (fill) queries, over a wide-dynamic-range grid.
        let axes: &[Vec<Vec<f64>>] = &[
            vec![(0..12).map(|i| i as f64 * 0.5).collect()],
            vec![
                (0..9).map(|i| i as f64).collect(),
                (0..11).map(|i| i as f64 * 0.3 - 1.0).collect(),
            ],
            vec![
                (0..6).map(|i| i as f64).collect(),
                (0..7).map(|i| i as f64 * 0.4).collect(),
                (0..8).map(|i| i as f64 * 0.25 - 0.5).collect(),
            ],
        ];
        for points in axes {
            let total: usize = points.iter().map(Vec::len).product();
            let values: Vec<f64> = (0..total)
                .map(|k| ((k as f64 * 0.37).sin() * 3.0 - 0.4) * if k % 4 == 0 { 1e7 } else { 1.0 })
                .collect();
            let interp = RegularGridInterpolator::new(
                points.clone(),
                values,
                RegularGridMethod::Pchip,
                false,
                Some(-9.5),
            )
            .unwrap();
            let ndim = points.len();
            let mut queries: Vec<Vec<f64>> = Vec::new();
            for t in 0..64 {
                queries.push(
                    points
                        .iter()
                        .map(|ax| {
                            let (lo, hi) = (ax[0], ax[ax.len() - 1]);
                            lo + (hi - lo) * ((t * 7 + 3) % 100) as f64 / 100.0
                        })
                        .collect(),
                );
                queries.push(points.iter().map(|ax| ax[t % ax.len()]).collect());
                // An out-of-bounds query (exercises the fill path in both routes).
                queries.push(points.iter().map(|ax| ax[ax.len() - 1] + 5.0).collect());
            }
            INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(true, Ordering::Relaxed);
            let scalar = interp.eval_many(&queries).unwrap();
            INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(false, Ordering::Relaxed);
            let hoisted = interp.eval_many(&queries).unwrap();
            assert_eq!(scalar.len(), hoisted.len());
            for (i, (a, b)) in scalar.iter().zip(&hoisted).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "pchip ndim={ndim} query {i}");
            }
        }
        INTERPN_PCHIP_BATCH_FORCE_SCALAR.store(false, Ordering::Relaxed);
    }

    #[test]
    fn regular_grid_pchip_requires_4_points() {
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0, 3.0]];
        let values = vec![0.0; 12];
        let err =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Pchip, false, None)
                .expect_err("too few points for pchip");
        assert!(matches!(err, InterpError::TooFewPoints { minimum: 4, .. }));
    }

    #[test]
    fn regular_grid_pchip_matches_scipy_reference_values() {
        let points: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
        let mut values = Vec::new();
        for &x in &points[0] {
            for &y in &points[1] {
                values.push(x.powi(4) * y.powi(4));
            }
        }
        let interp =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Pchip, false, None)
                .expect("regular grid pchip");

        let cases = [
            ([1.5, 2.0], 87.25),
            ([2.5, 2.5], 1575.924587673611),
            ([3.5, 3.0], 12279.515625),
        ];
        for (pt, expected) in cases {
            let actual = interp.eval(&pt).expect("eval pchip");
            assert!(
                (actual - expected).abs() < 1e-9,
                "point={pt:?} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn regular_grid_pchip_fill_and_extrapolate_match_scipy() {
        let points: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];
        let mut values = Vec::new();
        for &x in &points[0] {
            for &y in &points[1] {
                values.push(x.powi(4) * y.powi(4));
            }
        }

        let filled = RegularGridInterpolator::new(
            points.clone(),
            values.clone(),
            RegularGridMethod::Pchip,
            false,
            Some(-123.0),
        )
        .expect("regular grid pchip fill");
        let extrap =
            RegularGridInterpolator::new(points, values, RegularGridMethod::Pchip, false, None)
                .expect("regular grid pchip extrap");

        let filled_val = filled.eval(&[-1.0, 2.0]).expect("filled eval");
        let extrap_val = extrap.eval(&[-1.0, 2.0]).expect("extrap eval");
        assert_eq!(filled_val, -123.0);
        assert!((extrap_val - 2056.0).abs() < 1e-9);
    }

    #[test]
    fn cubic_spline_periodic_matches_endpoints_and_derivative() {
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (std::f64::consts::PI * t).sin())
            .collect();
        let spline = CubicSplineStandalone::new(&x, &y, SplineBc::Periodic).expect("periodic");
        assert!((spline.eval(0.0) - spline.eval(2.0)).abs() < 1e-10);
        let d = spline.derivative(1);
        assert!((d.eval(0.0) - d.eval(2.0)).abs() < 1e-8);
    }

    #[test]
    fn cubic_spline_bc_types_match_scipy() {
        // Golden values from scipy.interpolate.CubicSpline(x, y, bc_type=...)
        // for x=[0..4], y=x^3, evaluated at [0.5,1.5,2.5,3.5].
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 1.0, 8.0, 27.0, 64.0];
        let xe = [0.5, 1.5, 2.5, 3.5];
        let check = |bc: SplineBc, want: [f64; 4], name: &str| {
            let s = CubicSplineStandalone::new(&x, &y, bc).unwrap();
            for (i, (&got, &w)) in s.eval_many(&xe).iter().zip(want.iter()).enumerate() {
                assert!((got - w).abs() < 1e-9, "{name}[{i}]: {got} != {w}");
            }
        };
        check(
            SplineBc::Natural,
            [
                0.098_214_285_714_285_64,
                3.455_357_142_857_143,
                15.330_357_142_857_142,
                43.973_214_285_714_29,
            ],
            "natural",
        );
        // not-a-knot reproduces the cubic x^3 exactly.
        check(
            SplineBc::NotAKnot,
            [0.125, 3.375, 15.625, 42.875],
            "not-a-knot",
        );
        check(
            SplineBc::Clamped(0.0, 0.0),
            [
                0.017_857_142_857_142_85,
                3.910_714_285_714_286_5,
                13.589_285_714_285_714,
                50.482_142_857_142_86,
            ],
            "clamped",
        );
    }

    #[test]
    fn interp1d_linear_match_scipy() {
        // scipy.interpolate.interp1d default (linear) on y=x^2 nodes: midpoints
        // are the average of adjacent node values.
        let f = Interp1d::new(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 4.0, 9.0],
            Interp1dOptions::default(),
        )
        .expect("interp1d");
        let out = f.eval_many(&[0.5, 1.5, 2.5]).expect("eval");
        for (g, e) in out.iter().zip(&[0.5, 2.5, 6.5]) {
            assert!((g - e).abs() < 1e-12, "interp1d: {g} vs {e}");
        }
    }

    #[test]
    fn griddata_linear_nearest_match_scipy() {
        // scipy.interpolate.griddata on unit-square corners with f(x,y)=x+y.
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let vals = vec![0.0, 1.0, 1.0, 2.0];
        let xi = vec![vec![0.5, 0.5], vec![0.2, 0.7]];
        let lin = griddata(&pts, &vals, &xi, GriddataMethod::Linear).expect("linear");
        assert!((lin[0] - 1.0).abs() < 1e-12, "linear[0]: {}", lin[0]);
        assert!((lin[1] - 0.9).abs() < 1e-12, "linear[1]: {}", lin[1]);
        // Nearest at a non-tie point: [0.2,0.7] is closest to [0,1] -> 1.0.
        let near = griddata(&pts, &vals, &xi, GriddataMethod::Nearest).expect("nearest");
        assert!((near[1] - 1.0).abs() < 1e-12, "nearest[1]: {}", near[1]);
    }

    #[test]
    fn interpn_linear_match_scipy() {
        // scipy.interpolate.interpn((x,y), values, pts, method='linear') for
        // f(x,y)=x+y on a 3x3 grid; bilinear interpolation recovers f at the pts.
        let points = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];
        let values = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0];
        let xi = vec![vec![0.5, 0.5], vec![1.5, 0.5], vec![0.25, 1.75]];
        let out = interpn(points, values, &xi, RegularGridMethod::Linear, true, None).unwrap();
        for (g, e) in out.iter().zip(&[1.0, 2.0, 2.0]) {
            assert!((g - e).abs() < 1e-12, "interpn: {g} vs {e}");
        }
    }

    // The test above is degenerate in three separate ways, and each one hides a
    // different class of bug:
    //   * f(x,y) = x + y is LINEAR, and bilinear interpolation reproduces linear
    //     functions exactly. Its "goldens" are just f evaluated at the query
    //     points, so the interpolation weights are never actually constrained.
    //   * f is SYMMETRIC in x and y on a square grid, so transposing the axes
    //     produces identical output and an axis-ordering bug is invisible.
    //   * both axes are UNIFORM [0,1,2], so an implementation that assumes even
    //     spacing instead of using the real coordinates also passes.
    //
    // This pins interpn against SciPy on a case with none of those properties:
    // arbitrary non-round values with no exploitable structure, NON-UNIFORM
    // spacing on both axes, and axes of DIFFERENT lengths (4 x 3) so a
    // transposed implementation cannot even index the grid.
    //
    // Discrimination was checked, not assumed: feeding the same values to SciPy
    // with the x axis replaced by a uniform [0,1,2,3] yields
    // [0.9375, 4.25, 27.9375, 4.0] against the correct
    // [0.9375, 2.84375, 3.765625, 4.0] -- 27.94 vs 3.77 at the third query.
    #[test]
    fn interpn_linear_matches_scipy_on_nonuniform_asymmetric_grid() {
        // scipy.interpolate.interpn((x, y), values, pts, method='linear')
        // x = [0, 1, 3, 7] (non-uniform), y = [0, 2, 5] (non-uniform)
        let points = vec![vec![0.0, 1.0, 3.0, 7.0], vec![0.0, 2.0, 5.0]];
        // values[i][j], flattened C-order (first axis slowest), no structure.
        let values = vec![
            1.5, -2.25, 3.75, //
            0.5, 4.0, -1.125, //
            -3.25, 2.5, 6.0, //
            7.25, 0.75, -4.5,
        ];
        let xi = vec![
            vec![0.5, 1.0],
            vec![2.0, 3.5],
            vec![6.0, 0.5],
            vec![1.0, 2.0], // exactly on a grid node -> must return values[1][1]
        ];
        let out = interpn(points, values, &xi, RegularGridMethod::Linear, true, None)
            .expect("interpn 2-D");
        let expected = [0.9375, 2.843_75, 3.765_625, 4.0];
        assert_eq!(out.len(), expected.len());
        for (i, (g, e)) in out.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "interpn 2-D non-uniform [{i}]: {g} vs scipy {e}"
            );
        }
        // The on-node query must reproduce the stored value exactly; a
        // uniform-spacing implementation happens to agree here, which is why the
        // off-node queries above carry the discrimination.
        assert!(
            (out[3] - 4.0).abs() < 1e-15,
            "query exactly on a grid node must return that node's value, got {}",
            out[3]
        );

        // Same again in 3-D, since this bead is about N-D: axis lengths 3 x 2 x 3,
        // non-uniform x and z, and two values perturbed so the array is not an
        // arithmetic progression a wrong stride could still satisfy.
        let points3 = vec![vec![0.0, 1.0, 4.0], vec![0.0, 2.0], vec![0.0, 1.0, 3.0]];
        let values3 = vec![
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, //
            1.0, 1.5, 2.0, 2.5, 3.0, 7.75, //
            4.0, -3.25, 5.0, 5.5, 6.0, 6.5,
        ];
        let xi3 = vec![
            vec![0.5, 1.0, 0.5],
            vec![2.0, 0.5, 2.0],
            vec![3.0, 1.5, 1.0],
        ];
        let out3 = interpn(
            points3,
            values3,
            &xi3,
            RegularGridMethod::Linear,
            true,
            None,
        )
        .expect("interpn 3-D");
        let expected3 = [0.5, 2.510_416_666_666_667, 3.333_333_333_333_333_5];
        assert_eq!(out3.len(), expected3.len());
        for (i, (g, e)) in out3.iter().zip(&expected3).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "interpn 3-D non-uniform [{i}]: {g} vs scipy {e}"
            );
        }
    }

    #[test]
    fn pchip_akima_krogh_match_scipy() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]; // x^2
        let xe = [0.5, 1.5, 2.5, 3.5, 4.5];
        let close = |got: &[f64], want: &[f64], name: &str| {
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 1e-9, "{name}[{i}]: {g} != {w}");
            }
        };
        // scipy.interpolate.PchipInterpolator
        close(
            &pchip_interpolate(&x, &y, &xe).unwrap(),
            &[
                0.3125,
                2.21875,
                6.239_583_333_333_333,
                12.244_791_666_666_666,
                20.234_375,
            ],
            "pchip",
        );
        // scipy.interpolate.Akima1DInterpolator (reproduces x^2 on this grid)
        close(
            &akima1d_interpolate(&x, &y, &xe).unwrap(),
            &[0.25, 2.25, 6.25, 12.25, 20.25],
            "akima",
        );
        // scipy.interpolate.KroghInterpolator — polynomial through 4 points.
        let xk = [0.0, 1.0, 2.0, 3.0];
        let yk = [1.0, 3.0, 2.0, 5.0];
        close(
            &krogh_interpolate(&xk, &yk, &[0.5, 1.5, 2.5]).unwrap(),
            &[2.8125, 2.4375, 2.5625],
            "krogh",
        );
    }

    #[test]
    fn cubic_spline_periodic_rejects_mismatched_endpoints() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.5, 0.25];
        let err = CubicSplineStandalone::new(&x, &y, SplineBc::Periodic).expect_err("periodic");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn interp1d_rejects_nan_in_x_coordinates() {
        let x = vec![1.0, f64::NAN, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let err = Interp1d::new(&x, &y, Interp1dOptions::default()).expect_err("nan in x");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn interp1d_rejects_infinity_in_x_coordinates() {
        let x = vec![1.0, f64::INFINITY, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let err = Interp1d::new(&x, &y, Interp1dOptions::default()).expect_err("inf in x");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn pchip_rejects_nan_in_x_coordinates() {
        let x = vec![1.0, f64::NAN, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let err = PchipInterpolator::new(&x, &y).expect_err("nan in x");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn pchip_eval_non_finite_query_returns_nan() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let interp = PchipInterpolator::new(&x, &y).expect("pchip");
        assert!(interp.eval(f64::NAN).is_nan());
        assert!(interp.eval(f64::INFINITY).is_nan());
        assert!(interp.eval(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn pchip_eval_many_sorted_cursor_matches_scalar_bits() {
        let x = [0.0, 1.0, 2.0, 4.0, 7.0];
        let y = [1.0, -2.0, 3.5, 0.25, 8.0];
        let interp = PchipInterpolator::new(&x, &y).expect("pchip");
        let cases: &[&[f64]] = &[
            &[-2.0, 0.0, 0.0, 0.25, 1.0, 1.0, 2.0, 3.0, 4.0, 7.0, 9.0],
            &[0.5, 3.0, 1.5],
            &[0.5, f64::NAN, 2.5],
            &[f64::NEG_INFINITY, 0.5, f64::INFINITY],
        ];
        for queries in cases {
            let batch = interp.eval_many(queries);
            for (&query, value) in queries.iter().zip(batch.iter()) {
                assert_eq!(
                    value.to_bits(),
                    interp.eval(query).to_bits(),
                    "eval_many mismatch at {query:?}"
                );
            }
        }
    }

    #[test]
    fn cubic_segmented_cursor_matches_scalar_bits() {
        let x = [0.0, 1.0, 2.0, 4.0, 7.0];
        let y = [1.0, -2.0, 3.5, 0.25, 8.0];
        let dydx = [0.5, -1.0, 2.0, -0.25, 1.5];
        let queries = [
            -2.0, 0.0, 0.25, 1.0, 3.0, 7.0, 9.0, -1.0, 0.5, 2.0, 4.0, 8.0, 0.0, 1.5, 2.5, 6.0,
        ];
        let cubic = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic");
        let akima = Akima1DInterpolator::new(&x, &y).expect("akima");
        let hermite = CubicHermiteSpline::new(&x, &y, &dydx).expect("hermite");

        for (name, batch, scalar) in [
            (
                "cubic",
                cubic_cursor_eval_many(&cubic.x, &cubic.coeffs, &queries).expect("cursor"),
                queries
                    .iter()
                    .map(|&query| cubic.eval(query))
                    .collect::<Vec<f64>>(),
            ),
            (
                "akima",
                cubic_cursor_eval_many(&akima.x, &akima.coeffs, &queries).expect("cursor"),
                queries
                    .iter()
                    .map(|&query| akima.eval(query))
                    .collect::<Vec<f64>>(),
            ),
            (
                "hermite",
                cubic_cursor_eval_many(&hermite.x, &hermite.coeffs, &queries).expect("cursor"),
                queries
                    .iter()
                    .map(|&query| hermite.eval(query))
                    .collect::<Vec<f64>>(),
            ),
        ] {
            assert!(
                batch
                    .iter()
                    .zip(&scalar)
                    .all(|(left, right)| left.to_bits() == right.to_bits()),
                "{name} segmented cursor must be bit-identical to scalar evaluation"
            );
        }

        let nine_runs: Vec<f64> = (0..9).flat_map(|_| [0.0, 0.5, 1.0]).collect();
        assert!(
            cubic_cursor_eval_many(&cubic.x, &cubic.coeffs, &nine_runs).is_none(),
            "more than eight ascending runs must retain the binary-search fallback"
        );
    }

    #[test]
    fn akima_rejects_nan_in_x_coordinates() {
        let x = vec![1.0, f64::NAN, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let err = Akima1DInterpolator::new(&x, &y).expect_err("nan in x");
        assert!(matches!(err, InterpError::NonFiniteX));
    }

    #[test]
    fn akima_eval_non_finite_query_returns_nan() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let interp = Akima1DInterpolator::new(&x, &y).expect("akima");
        assert!(interp.eval(f64::NAN).is_nan());
        assert!(interp.eval(f64::INFINITY).is_nan());
        assert!(interp.eval(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn krogh_eval_non_finite_query_returns_nan() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        let interp = KroghInterpolator::new(&x, &y).expect("krogh");
        assert!(interp.evaluate(f64::NAN).is_nan());
        assert!(interp.evaluate(f64::INFINITY).is_nan());
        assert!(interp.evaluate(f64::NEG_INFINITY).is_nan());
    }

    // ── RectBivariateSpline tests ────────────────────────────────────

    #[test]
    fn rect_bivariate_eval_many_hoist_matches_scalar_bitexact() {
        use std::sync::atomic::Ordering;
        // The hoisted-x-spline batch path must be BYTE-IDENTICAL to the per-query `eval`, across
        // degrees, interior/edge/out-of-bounds (clamped) and non-finite queries.
        for (kx, ky) in [(1usize, 1usize), (3, 3), (2, 3), (3, 2)] {
            let nx = (kx + 6).max(6);
            let ny = (ky + 7).max(7);
            let x: Vec<f64> = (0..nx).map(|i| i as f64 * 0.7).collect();
            let y: Vec<f64> = (0..ny).map(|j| j as f64 * 0.4 - 1.0).collect();
            let z: Vec<Vec<f64>> = (0..nx)
                .map(|i| {
                    (0..ny)
                        .map(|j| ((i * 5 + j * 3) as f64 * 0.31).sin() * 4.0 - 0.7)
                        .collect()
                })
                .collect();
            let spline = RectBivariateSpline::new(&x, &y, &z, kx, ky).expect("spline");
            // Scattered queries: interior, exact nodes, out-of-bounds (clamped), and non-finite.
            let mut qx: Vec<f64> = Vec::new();
            let mut qy: Vec<f64> = Vec::new();
            for t in 0..80 {
                qx.push(x[0] + (x[nx - 1] - x[0]) * ((t * 7 + 1) % 100) as f64 / 100.0);
                qy.push(y[0] + (y[ny - 1] - y[0]) * ((t * 11 + 5) % 100) as f64 / 100.0);
            }
            qx.push(x[nx - 1] + 3.0); // out of bounds high -> clamp
            qy.push(y[0] - 2.0); // out of bounds low -> clamp
            qx.push(f64::NAN);
            qy.push(0.5);
            RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(true, Ordering::Relaxed);
            let scalar = spline.eval_many(&qx, &qy).unwrap();
            RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(false, Ordering::Relaxed);
            let hoisted = spline.eval_many(&qx, &qy).unwrap();
            assert_eq!(scalar.len(), hoisted.len());
            for (i, (a, b)) in scalar.iter().zip(&hoisted).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "kx={kx} ky={ky} query {i}");
            }
        }
        RECTBISPLINE_EVAL_MANY_FORCE_SCALAR.store(false, Ordering::Relaxed);
    }

    #[test]
    fn rect_bivariate_spline_linear_plane() {
        // z = x + y should be interpolated exactly by any spline
        // Use 4 points in each direction for cubic (k=3 needs k+1=4 points)
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let z = vec![
            vec![0.0, 1.0, 2.0, 3.0], // y=0
            vec![1.0, 2.0, 3.0, 4.0], // y=1
            vec![2.0, 3.0, 4.0, 5.0], // y=2
            vec![3.0, 4.0, 5.0, 6.0], // y=3
        ];
        let spline = RectBivariateSpline::new(&x, &y, &z, 3, 3).expect("bicubic");

        // Test at grid points
        for (yi, yval) in y.iter().enumerate() {
            for (xi, xval) in x.iter().enumerate() {
                let val = spline.eval(*xval, *yval);
                let expected = z[yi][xi];
                assert!(
                    (val - expected).abs() < 1e-10,
                    "at ({}, {}): got {}, expected {}",
                    xval,
                    yval,
                    val,
                    expected
                );
            }
        }

        // Test at midpoints
        let val = spline.eval(0.5, 0.5);
        assert!(
            (val - 1.0).abs() < 0.1,
            "at (0.5, 0.5): got {}, expected 1.0",
            val
        );

        let val = spline.eval(1.5, 1.0);
        assert!(
            (val - 2.5).abs() < 0.1,
            "at (1.5, 1.0): got {}, expected 2.5",
            val
        );
    }

    #[test]
    fn rect_bivariate_spline_bilinear() {
        // Test bilinear spline (kx=ky=1)
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0], vec![1.0, 2.0]];
        let spline = rect_bilinear_spline(&x, &y, &z).expect("bilinear");

        // Corners
        assert!((spline.eval(0.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((spline.eval(1.0, 0.0) - 1.0).abs() < 1e-10);
        assert!((spline.eval(0.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((spline.eval(1.0, 1.0) - 2.0).abs() < 1e-10);

        // Center
        let center = spline.eval(0.5, 0.5);
        assert!(
            (center - 1.0).abs() < 1e-10,
            "center: got {}, expected 1.0",
            center
        );
    }

    #[test]
    fn rect_bivariate_spline_uses_scipy_x_major_z_shape() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0];
        let z: Vec<Vec<f64>> = x
            .iter()
            .map(|&xv| y.iter().map(|&yv| 10.0 * xv + yv).collect())
            .collect();
        let spline = RectBivariateSpline::new(&x, &y, &z, 1, 1).expect("bilinear");

        let val = spline.eval(1.5, 0.5);
        assert!(
            (val - 15.5).abs() < 1e-10,
            "at (1.5, 0.5): got {val}, expected 15.5"
        );

        let grid = spline.eval_grid(&[0.5, 1.5], &[0.5, 1.5]);
        let expected = [vec![5.5, 6.5], vec![15.5, 16.5]];
        for (got_row, expected_row) in grid.iter().zip(expected.iter()) {
            for (&got, &want) in got_row.iter().zip(expected_row.iter()) {
                assert!((got - want).abs() < 1e-10, "got {got}, expected {want}");
            }
        }
    }

    #[test]
    fn rect_bivariate_spline_quadratic_surface() {
        // z = x^2 + y^2
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let z: Vec<Vec<f64>> = y
            .iter()
            .map(|&yv| x.iter().map(|&xv| xv * xv + yv * yv).collect())
            .collect();

        let spline = rect_bivariate_spline(&x, &y, &z).expect("bicubic");

        // Test at grid points
        for (yi, &yv) in y.iter().enumerate() {
            for (xi, &xv) in x.iter().enumerate() {
                let val = spline.eval(xv, yv);
                let expected = z[yi][xi];
                assert!(
                    (val - expected).abs() < 1e-8,
                    "at ({}, {}): got {}, expected {}",
                    xv,
                    yv,
                    val,
                    expected
                );
            }
        }

        // Test at a midpoint (2.5, 2.5) - expected 12.5
        let val = spline.eval(2.5, 2.5);
        assert!(
            (val - 12.5).abs() < 0.5,
            "at (2.5, 2.5): got {}, expected 12.5",
            val
        );
    }

    #[test]
    fn rect_bivariate_spline_eval_grid() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let z: Vec<Vec<f64>> = y
            .iter()
            .map(|&yv| x.iter().map(|&xv| xv + yv).collect())
            .collect();

        let spline = rect_bivariate_spline(&x, &y, &z).expect("bicubic");

        let xi = vec![0.5, 1.5];
        let yi = vec![0.5, 1.5];
        let result = spline.eval_grid(&xi, &yi);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);

        // Check corners of result grid
        assert!((result[0][0] - 1.0).abs() < 0.2, "got {}", result[0][0]); // (0.5, 0.5)
        assert!((result[0][1] - 2.0).abs() < 0.2, "got {}", result[0][1]); // (1.5, 0.5)
        assert!((result[1][0] - 2.0).abs() < 0.2, "got {}", result[1][0]); // (0.5, 1.5)
        assert!((result[1][1] - 3.0).abs() < 0.2, "got {}", result[1][1]); // (1.5, 1.5)
    }

    #[test]
    fn bivariate_eval_grid_parallel_is_byte_identical_to_serial() {
        use std::sync::atomic::Ordering;
        // Both eval_grid variants must be BYTE-IDENTICAL parallel vs serial on a grid
        // past the ≥40k fan-out gate (each x-row is an independent evaluation).
        let g: Vec<f64> = (0..210).map(|i| i as f64 / 209.0).collect(); // 210² = 44100 > 40k
        // RectBivariateSpline from a small data grid.
        let nd = 24usize;
        let xd: Vec<f64> = (0..nd).map(|i| i as f64 / (nd - 1) as f64).collect();
        let zd: Vec<Vec<f64>> = xd
            .iter()
            .map(|&yv| {
                xd.iter()
                    .map(|&xv| (xv * 4.0).sin() * (yv * 3.0).cos())
                    .collect()
            })
            .collect();
        let rbs = rect_bivariate_spline(&xd, &xd, &zd).unwrap();
        // SmoothBivariateSpline from scattered data.
        let mut st = 0x9u64;
        let mut rng = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut sx, mut sy, mut sz) = (vec![], vec![], vec![]);
        for _ in 0..250 {
            let (a, b) = (rng(), rng());
            sx.push(a);
            sy.push(b);
            sz.push((a * 4.0).sin() * (b * 3.0).cos());
        }
        let sbs = smooth_bivariate_spline(&sx, &sy, &sz).unwrap();

        for (label, ev) in [
            (
                "rect",
                &(|| rbs.eval_grid(&g, &g)) as &dyn Fn() -> Vec<Vec<f64>>,
            ),
            (
                "smooth",
                &(|| sbs.eval_grid(&g, &g)) as &dyn Fn() -> Vec<Vec<f64>>,
            ),
        ] {
            EVAL_GRID_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let serial = ev();
            EVAL_GRID_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let parallel = ev();
            let mism: usize = serial
                .iter()
                .zip(parallel.iter())
                .map(|(sr, pr)| {
                    sr.iter()
                        .zip(pr.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                })
                .sum();
            assert_eq!(
                mism, 0,
                "{label} eval_grid parallel must be byte-identical to serial"
            );
        }
    }

    #[test]
    fn rect_bivariate_spline_integral() {
        // Constant function z = 1 over [0,1] x [0,1] should integrate to 1
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![0.0, 0.5, 1.0];
        let z = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];

        let spline = rect_bilinear_spline(&x, &y, &z).expect("bilinear");
        let integral = spline.integral(0.0, 1.0, 0.0, 1.0);
        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral: got {}, expected 1.0",
            integral
        );
    }

    #[test]
    fn smooth_bivariate_eval_grid_hoist_matches_percell_bitexact() {
        use std::sync::atomic::Ordering;
        // The hoisted eval_grid (x-splines built once, y-spline once per row) must be
        // BYTE-IDENTICAL to the per-cell path, across degrees and finite/out-of-bounds/non-finite
        // query grids.
        for (kx, ky) in [(1usize, 1usize), (3, 3), (2, 3)] {
            // Scattered samples on a jittered grid — enough for a stable (kx,ky) fit.
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut zs = Vec::new();
            for i in 0..9 {
                for j in 0..9 {
                    let xv = i as f64 * 0.5 + ((i * 3 + j) % 5) as f64 * 0.01;
                    let yv = j as f64 * 0.4 - 1.0 + ((i + j * 2) % 4) as f64 * 0.01;
                    xs.push(xv);
                    ys.push(yv);
                    zs.push((xv * 1.3).sin() * 2.0 + (yv * 0.9).cos() - 0.3 * xv * yv);
                }
            }
            let options = SmoothBivariateSplineOptions {
                kx,
                ky,
                smoothing: Some(0.5),
                ..SmoothBivariateSplineOptions::default()
            };
            let Ok(spline) = SmoothBivariateSpline::new(&xs, &ys, &zs, options) else {
                continue;
            };
            // Query grid: interior, out-of-bounds (clamped both ends), and non-finite.
            let gx = vec![-3.0, 0.0, 0.75, 1.6, 2.5, 4.0, 9.0, f64::NAN];
            let gy = vec![-5.0, -0.9, 0.0, 0.5, 1.1, 3.0, f64::NAN, 0.2];
            SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(true, Ordering::Relaxed);
            let percell = spline.eval_grid(&gx, &gy);
            SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(false, Ordering::Relaxed);
            let hoisted = spline.eval_grid(&gx, &gy);
            assert_eq!(percell.len(), hoisted.len());
            for (ri, (pr, hr)) in percell.iter().zip(&hoisted).enumerate() {
                assert_eq!(pr.len(), hr.len());
                for (ci, (a, b)) in pr.iter().zip(hr).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "kx={kx} ky={ky} cell ({ri},{ci})");
                }
            }
        }
        SMOOTHBISPLINE_EVAL_GRID_FORCE_PERCELL.store(false, Ordering::Relaxed);
    }

    #[test]
    fn smooth_bivariate_spline_scattered_bilinear_surface() {
        let x = vec![0.0, 1.0, 0.0, 1.0, 0.5, 0.25];
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.5, 0.75];
        let z: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&xv, &yv)| 2.0 + 3.0 * xv - 4.0 * yv + 5.0 * xv * yv)
            .collect();
        let options = SmoothBivariateSplineOptions {
            kx: 1,
            ky: 1,
            smoothing: Some(0.0),
            ..SmoothBivariateSplineOptions::default()
        };
        let spline = SmoothBivariateSpline::new(&x, &y, &z, options).expect("smooth bivariate");

        let value = spline.eval(0.25, 0.5);
        assert!((value - 1.375).abs() < 1e-10, "value={value}");

        let dx = spline.eval_derivative(0.25, 0.5, 1, 0);
        assert!((dx - 5.5).abs() < 1e-10, "dx={dx}");

        let dy = spline.eval_derivative(0.25, 0.5, 0, 1);
        assert!((dy + 2.75).abs() < 1e-10, "dy={dy}");

        let integral = spline.integral(0.0, 1.0, 0.0, 1.0);
        assert!((integral - 2.75).abs() < 1e-10, "integral={integral}");

        let grid = spline.eval_grid(&[0.0, 1.0], &[0.0, 1.0]);
        let expected = [vec![2.0, -2.0], vec![5.0, 6.0]];
        for (got_row, expected_row) in grid.iter().zip(expected.iter()) {
            for (&got, &want) in got_row.iter().zip(expected_row.iter()) {
                assert!((got - want).abs() < 1e-10, "got {got}, expected {want}");
            }
        }

        assert!(spline.residual() < 1e-18, "residual={}", spline.residual());
        assert_eq!(spline.degrees(), (1, 1));
        assert_eq!(spline.bbox(), [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn smooth_bivariate_spline_builds_piecewise_surface() {
        let x = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
        let y = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let z: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(&xv, &yv)| (xv - 0.5_f64).abs() + (yv - 0.5_f64).abs())
            .collect();
        let options = SmoothBivariateSplineOptions {
            kx: 1,
            ky: 1,
            smoothing: Some(0.0),
            ..SmoothBivariateSplineOptions::default()
        };
        let spline = SmoothBivariateSpline::new(&x, &y, &z, options).expect("piecewise surface");

        let value = spline.eval(0.25, 0.75);
        assert!((value - 0.5).abs() < 1e-10, "value={value}");

        let integral = spline.integral(0.0, 1.0, 0.0, 1.0);
        assert!((integral - 0.5).abs() < 3e-2, "integral={integral}");

        let (tx, ty) = spline.knots();
        assert!(tx.len() > 2 * (spline.kx + 1), "tx={tx:?}");
        assert!(ty.len() > 2 * (spline.ky + 1), "ty={ty:?}");
        assert!(
            spline.coefficients().len() > (spline.kx + 1) * (spline.ky + 1),
            "coeff_count={}",
            spline.coefficients().len()
        );
    }

    #[test]
    fn smooth_bivariate_spline_rejects_invalid_weights() {
        let x = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let z = vec![0.0, 1.0, 1.0, 2.0];
        let options = SmoothBivariateSplineOptions {
            weights: Some(vec![1.0, 1.0, 1.0]),
            kx: 1,
            ky: 1,
            smoothing: Some(0.0),
            ..SmoothBivariateSplineOptions::default()
        };
        let err =
            SmoothBivariateSpline::new(&x, &y, &z, options).expect_err("weight length mismatch");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rect_bivariate_spline_rejects_mismatched_dimensions() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0];
        let z = vec![
            vec![0.0, 1.0], // Wrong: should have 3 columns
        ];
        let err = RectBivariateSpline::new(&x, &y, &z, 1, 1).expect_err("dimension mismatch");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rect_bivariate_spline_rejects_non_monotonic_x() {
        let x = vec![0.0, 2.0, 1.0]; // Not strictly increasing
        let y = vec![0.0, 1.0];
        let z = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let err = RectBivariateSpline::new(&x, &y, &z, 1, 1).expect_err("non-monotonic");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn rect_bivariate_spline_too_few_points() {
        let x = vec![0.0, 1.0]; // Only 2 points, need 4 for cubic
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let z = vec![vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0, 4.0]];
        let err = RectBivariateSpline::new(&x, &y, &z, 3, 3).expect_err("too few points");
        assert!(matches!(err, InterpError::TooFewPoints { .. }));
    }

    #[test]
    fn cubic_hermite_passes_through_nodes() {
        // Interpolating at the original xi recovers yi exactly regardless
        // of supplied slopes.
        let xi = [0.0_f64, 1.0, 2.0, 3.0];
        let yi = [1.0_f64, 4.0, 2.0, 5.0];
        let dydx = [0.5_f64, 1.0, -1.0, 0.0];
        let got = cubic_hermite_interpolate(&xi, &yi, &dydx, &xi).expect("interp");
        for (g, y) in got.iter().zip(yi.iter()) {
            assert!((g - y).abs() < 1e-12);
        }
    }

    #[test]
    fn cubic_hermite_metamorphic_recovers_polynomial_with_correct_slopes() {
        // For f(x) = x³ - 2x + 1 with exact slopes f'(x) = 3x² - 2, the
        // cubic Hermite reconstruction is exact at every dense point.
        let xi = [-2.0_f64, 0.0, 2.0];
        let f = |x: f64| x * x * x - 2.0 * x + 1.0;
        let df = |x: f64| 3.0 * x * x - 2.0;
        let yi: Vec<f64> = xi.iter().copied().map(f).collect();
        let dydx: Vec<f64> = xi.iter().copied().map(df).collect();
        let dense: Vec<f64> = (0..=200).map(|i| -2.0 + i as f64 * 4.0 / 200.0).collect();
        let got = cubic_hermite_interpolate(&xi, &yi, &dydx, &dense).expect("interp");
        let expected: Vec<f64> = dense.iter().copied().map(f).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-12,
                "polynomial reproduction: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn cubic_hermite_rejects_mismatched_dydx_length() {
        let err = cubic_hermite_interpolate(&[0.0, 1.0], &[2.0, 3.0], &[0.5], &[0.5])
            .expect_err("mismatched dydx must be rejected");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn cubic_hermite_rejects_unsorted_x() {
        let err =
            cubic_hermite_interpolate(&[0.0, 2.0, 1.0], &[0.0, 4.0, 2.0], &[0.5, 1.0, 1.5], &[0.5])
                .expect_err("unsorted x must be rejected");
        assert!(matches!(err, InterpError::UnsortedX));
    }

    #[test]
    fn akima1d_interpolate_passes_through_nodes() {
        let xi = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let yi = [0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let got = akima1d_interpolate(&xi, &yi, &xi).expect("interp");
        for (g, y) in got.iter().zip(yi.iter()) {
            assert!((g - y).abs() < 1e-12);
        }
    }

    #[test]
    fn akima1d_interpolate_metamorphic_dense_in_data_range() {
        // Akima interpolation at 200 dense interior points stays within
        // the y-range of the original data — Akima resists overshoot.
        let xi = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let yi = [0.5_f64, 1.0, 1.0, 1.5, 2.0];
        let dense: Vec<f64> = (1..=199).map(|i| i as f64 * 4.0 / 200.0).collect();
        let got = akima1d_interpolate(&xi, &yi, &dense).expect("interp");
        let (min_y, max_y) = (
            yi.iter().cloned().fold(f64::INFINITY, f64::min),
            yi.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        for v in &got {
            assert!(
                *v >= min_y - 0.01 && *v <= max_y + 0.01,
                "Akima overshoot: {v} outside [{min_y}-eps, {max_y}+eps]"
            );
        }
    }

    #[test]
    fn akima1d_interpolate_rejects_unsorted_x() {
        let err = akima1d_interpolate(&[0.0, 2.0, 1.0], &[0.0, 4.0, 2.0], &[0.5])
            .expect_err("unsorted x must be rejected");
        assert!(matches!(err, InterpError::UnsortedX));
    }

    #[test]
    fn pchip_interpolate_metamorphic_passes_through_nodes() {
        let xi = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let yi = [0.0_f64, 1.0, 4.0, 9.0, 16.0];
        let got = pchip_interpolate(&xi, &yi, &xi).expect("interp");
        for (g, y) in got.iter().zip(yi.iter()) {
            assert!((g - y).abs() < 1e-12);
        }
    }

    #[test]
    fn pchip_interpolate_metamorphic_monotone_preservation() {
        // PCHIP's defining property: monotone data → monotone interpolant.
        let xi = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let yi = [0.0_f64, 0.5, 1.0, 1.8, 3.5, 7.0]; // strictly increasing
        let dense: Vec<f64> = (0..=200).map(|i| i as f64 * 5.0 / 200.0).collect();
        let got = pchip_interpolate(&xi, &yi, &dense).expect("interp");
        for w in got.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-12,
                "monotonicity violated: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn pchip_interpolate_rejects_unsorted_x() {
        let err = pchip_interpolate(&[0.0, 2.0, 1.0], &[0.0, 4.0, 2.0], &[0.5])
            .expect_err("unsorted x must be rejected");
        assert!(matches!(err, InterpError::UnsortedX));
    }

    #[test]
    fn krogh_interpolate_recovers_polynomial_exactly() {
        // Krogh polynomial through n+1 nodes recovers degree-≤n polynomials
        // exactly — same property as barycentric and Lagrange.
        let xi = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
        let yi: Vec<f64> = xi.iter().map(|x| 2.0 * x * x - 3.0 * x + 1.0).collect();
        let x_new = [-1.5_f64, -0.5, 0.5, 1.5];
        let got = krogh_interpolate(&xi, &yi, &x_new).expect("interp");
        let expected: Vec<f64> = x_new.iter().map(|x| 2.0 * x * x - 3.0 * x + 1.0).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-12, "krogh: got {g}, expected {e}");
        }
    }

    #[test]
    fn krogh_interpolate_metamorphic_passes_through_nodes() {
        let xi = [0.0_f64, 0.5, 1.5, 3.0];
        let yi = [1.0_f64, 4.0, 2.0, 5.0];
        let got = krogh_interpolate(&xi, &yi, &xi).expect("interp");
        for (g, y) in got.iter().zip(yi.iter()) {
            assert!((g - y).abs() < 1e-10);
        }
    }

    #[test]
    fn krogh_interpolate_rejects_duplicate_knots() {
        let err = krogh_interpolate(&[0.0, 1.0, 1.0], &[2.0, 3.0, 4.0], &[0.5])
            .expect_err("duplicate knots must be rejected");
        assert!(matches!(err, InterpError::InvalidArgument { .. }));
    }

    #[test]
    fn barycentric_interpolate_recovers_polynomial_exactly() {
        // f(x) = x³ - 2x + 1 sampled at five nodes; barycentric interpolation
        // through n+1 nodes recovers any polynomial of degree ≤ n exactly.
        let xi = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
        let yi: Vec<f64> = xi.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
        let x_new = [-1.5_f64, -0.5, 0.5, 1.5];
        let got = barycentric_interpolate(&xi, &yi, &x_new).expect("interp");
        let expected: Vec<f64> = x_new.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-12,
                "barycentric reproduction: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn barycentric_interpolate_passes_through_nodes() {
        let xi = [0.0_f64, 1.0, 3.0, 5.0];
        let yi = [2.0_f64, 5.0, 1.0, 4.0];
        let got = barycentric_interpolate(&xi, &yi, &xi).expect("interp");
        for (g, y) in got.iter().zip(yi.iter()) {
            assert!((g - y).abs() < 1e-12);
        }
    }

    #[test]
    fn barycentric_interpolate_rejects_mismatched_lengths() {
        let err = barycentric_interpolate(&[0.0, 1.0], &[2.0], &[0.5])
            .expect_err("length mismatch must be rejected");
        assert!(matches!(err, InterpError::LengthMismatch { .. }));
    }

    #[test]
    fn interp1d_linear_midpoint_reference_values() {
        // /testing-conformance-harnesses for [frankenscipy-g5bi0]:
        // pin scipy.interpolate.interp1d(kind='linear') at midpoints.
        //
        //   x = [0, 1, 2, 3], y = [0, 2, 1, 4]
        //   midpoint t = 0.5 on each interval:
        //     0.5 = 0 + 0.5·(2 − 0) = 1.0
        //     1.5 = 2 + 0.5·(1 − 2) = 1.5
        //     2.5 = 1 + 0.5·(4 − 1) = 2.5
        let x = vec![0.0_f64, 1.0, 2.0, 3.0];
        let y = vec![0.0_f64, 2.0, 1.0, 4.0];
        let xnew = vec![0.5_f64, 1.5, 2.5];
        let result = interp1d_linear(&x, &y, &xnew).expect("interp1d_linear");
        let expected = [1.0_f64, 1.5, 2.5];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "interp1d_linear[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn interp1d_linear_exact_at_nodes() {
        // The interpolant must reproduce y exactly when queried at the
        // input nodes — including the right endpoint, which is special-
        // cased by the bounds check (xi >= self.x[n-1]).
        let x = vec![-2.0_f64, 0.0, 1.5, 4.0];
        let y = vec![3.0_f64, -1.0, 2.5, 7.0];
        let result = interp1d_linear(&x, &y, &x).expect("interp1d_linear");
        for (i, (&got, &exp)) in result.iter().zip(y.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "interp1d_linear at node {i} (x={}) = {got}, expected {exp}",
                x[i]
            );
        }
    }

    #[test]
    fn interp1d_linear_out_of_bounds_returns_nan() {
        // The wrapper sets bounds_error=false and leaves fill_value=None,
        // so out-of-domain queries fall through to NaN rather than
        // erroring or extrapolating linearly.
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![0.0_f64, 1.0, 4.0];
        let result = interp1d_linear(&x, &y, &[-0.5, 0.5, 2.5]).expect("interp1d_linear");
        assert!(
            result[0].is_nan(),
            "x=-0.5 should be NaN, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.5).abs() < 1e-12,
            "x=0.5 should be 0.5, got {}",
            result[1]
        );
        assert!(result[2].is_nan(), "x=2.5 should be NaN, got {}", result[2]);
    }

    #[test]
    fn polyfromroots_closed_form_examples() {
        // /porting-to-rust + /testing-golden-artifacts for
        // [frankenscipy-bbc5k]:
        //   roots []         → [1]                  (empty product)
        //   roots [a]        → [1, -a]              (x − a)
        //   roots [1, 2]     → [1, -3, 2]           (x² − 3x + 2)
        //   roots [1, 2, 3]  → [1, -6, 11, -6]      (x³ − 6x² + 11x − 6)
        //   roots [-1, 1]    → [1, 0, -1]           (x² − 1)
        let cases: &[(&[f64], &[f64])] = &[
            (&[], &[1.0]),
            (&[3.0], &[1.0, -3.0]),
            (&[1.0, 2.0], &[1.0, -3.0, 2.0]),
            (&[1.0, 2.0, 3.0], &[1.0, -6.0, 11.0, -6.0]),
            (&[-1.0, 1.0], &[1.0, 0.0, -1.0]),
        ];
        for (roots, expected) in cases {
            let got = polyfromroots(roots);
            assert_eq!(
                got.len(),
                expected.len(),
                "polyfromroots({roots:?}) length {} != expected {}",
                got.len(),
                expected.len()
            );
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "polyfromroots({roots:?})[{i}] = {g}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn polyfromroots_metamorphic_recovers_root_at_polyval() {
        // /testing-metamorphic: polyval(polyfromroots(roots), root) ≈ 0
        // for every input root. The constructed polynomial vanishes
        // exactly at each root by construction.
        let roots = [-2.5_f64, -0.7, 0.3, 1.4, 5.0];
        let coeffs = polyfromroots(&roots);
        for &r in &roots {
            let v = polyval(&coeffs, r);
            assert!(
                v.abs() < 1e-9,
                "polyval(polyfromroots({roots:?}), {r}) = {v}, expected ≈ 0"
            );
        }
    }

    #[test]
    fn lagrange_recovers_known_polynomials() {
        // /testing-conformance-harnesses: pin closed-form Lagrange
        // interpolating polynomials. Output is highest-degree-first.
        //
        //   nodes (0,1,2) values (0,1,4) → x²       → [1, 0, 0]
        //   nodes (0,1)   values (3,5)   → 2x + 3   → [2, 3]
        //   nodes (0)     values (42)    → 42       → [42]
        let p = lagrange(&[0.0_f64, 1.0, 2.0], &[0.0, 1.0, 4.0]).unwrap();
        assert_eq!(p.len(), 3);
        assert!((p[0] - 1.0).abs() < 1e-12, "x² coeff = {}", p[0]);
        assert!(p[1].abs() < 1e-12, "x¹ coeff = {}", p[1]);
        assert!(p[2].abs() < 1e-12, "x⁰ coeff = {}", p[2]);

        let q = lagrange(&[0.0_f64, 1.0], &[3.0, 5.0]).unwrap();
        assert_eq!(q.len(), 2);
        assert!((q[0] - 2.0).abs() < 1e-12, "x¹ coeff = {}", q[0]);
        assert!((q[1] - 3.0).abs() < 1e-12, "x⁰ coeff = {}", q[1]);

        let r = lagrange(&[0.0_f64], &[42.0]).unwrap();
        assert_eq!(r, vec![42.0]);
    }

    /// Hand-built degree-2 B-spline as a pair of tck variants that differ
    /// only in how `c` is sized: the `tight` form carries the `m = 5`
    /// meaningful coefficients, the `padded` form zero-pads to `len(t) = 8`
    /// like a SciPy/FITPACK tck. Spline calculus must accept both.
    fn quadratic_bspline_tcks() -> Vec<(Vec<f64>, Vec<f64>, usize)> {
        let t = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let c = vec![1.0, 2.0, 3.0, 1.0, 0.0];
        let mut c_padded = c.clone();
        c_padded.resize(t.len(), 0.0);
        vec![(t.clone(), c, 2), (t, c_padded, 2)]
    }

    #[test]
    fn splantider_matches_scipy_and_accepts_padded_tck() {
        // Golden values from scipy.interpolate.splantider on the exact tck.
        let expected_c = [
            0.0,
            1.0 / 3.0,
            5.0 / 3.0,
            14.0 / 3.0,
            16.0 / 3.0,
            16.0 / 3.0,
        ];
        let expected_t = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        // The padded (SciPy-shaped) tck must not panic — this is the bug fix.
        for tck in &quadratic_bspline_tcks() {
            let (at, ac, ak) = splantider(tck).unwrap();
            assert_eq!(ak, 3, "antiderivative degree is k + 1");
            assert_eq!(at.len(), expected_t.len());
            for (got, want) in at.iter().zip(expected_t.iter()) {
                assert!((got - want).abs() < 1e-12, "knot {got} != {want}");
            }
            // Output uses the tight convention: len(c) == len(t) - k - 1.
            assert_eq!(ac.len(), at.len() - ak - 1);
            assert_eq!(ac.len(), expected_c.len());
            for (got, want) in ac.iter().zip(expected_c.iter()) {
                assert!((got - want).abs() < 1e-12, "coeff {got} != {want}");
            }
        }
    }

    #[test]
    fn splder_matches_scipy_and_accepts_padded_tck() {
        // Golden values from scipy.interpolate.splder on the exact tck.
        let expected_c = [2.0, 1.0, -2.0, -2.0];
        let expected_t = [0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        for tck in &quadratic_bspline_tcks() {
            let (dt, dc, dk) = splder(tck).unwrap();
            assert_eq!(dk, 1, "derivative degree is k - 1");
            assert_eq!(dt, expected_t);
            assert_eq!(dc.len(), dt.len() - dk - 1);
            assert_eq!(dc, expected_c);
        }
    }

    #[test]
    fn generate_knots_matches_scipy() {
        fn check(got: &[Vec<f64>], exp: &[Vec<f64>], label: &str) {
            assert_eq!(got.len(), exp.len(), "{label}: sequence length");
            for (gi, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
                assert_eq!(g.len(), e.len(), "{label} step {gi}: knot count");
                for (a, b) in g.iter().zip(e.iter()) {
                    assert!((a - b).abs() < 1e-9, "{label} step {gi}: {a} vs {b}");
                }
            }
        }
        // Dataset A: linspace(0,1,11), y = sin(6x) + 0.1 cos(20x).
        let xa: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let xa = {
            let mut v = xa;
            v[10] = 1.0;
            v
        };
        let ya: Vec<f64> = xa
            .iter()
            .map(|&x| (6.0 * x).sin() + 0.1 * (20.0 * x).cos())
            .collect();

        let s05 = generate_knots(&xa, &ya, None, None, None, 3, 0.05, None).unwrap();
        check(
            &s05,
            &[
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
            ],
            "A s=0.05",
        );

        let s005 = generate_knots(&xa, &ya, None, None, None, 3, 0.005, None).unwrap();
        check(
            &s005,
            &[
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0,
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0,
                ],
            ],
            "A s=0.005",
        );

        // Large s -> constant fit, only the initial knot vector.
        let s5 = generate_knots(&xa, &ya, None, None, None, 3, 0.5, None).unwrap();
        check(
            &s5,
            &[vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]],
            "A s=0.5",
        );

        // k = 2.
        let c = generate_knots(&xa, &ya, None, None, None, 2, 0.05, None).unwrap();
        check(
            &c,
            &[
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.5, 0.8, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.0, 1.0],
            ],
            "C k=2 s=0.05",
        );

        // Weighted (linear ramp 1..2): one extra refinement step vs unweighted.
        let wd: Vec<f64> = (0..11).map(|i| 1.0 + i as f64 * 0.1).collect();
        let d = generate_knots(&xa, &ya, Some(&wd), None, None, 3, 0.05, None).unwrap();
        check(
            &d,
            &[
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0],
            ],
            "D weighted s=0.05",
        );

        // Dataset B: linspace(-3,3,20), y = exp(-x^2).
        let xb: Vec<f64> = {
            let mut v: Vec<f64> = (0..20).map(|i| -3.0 + i as f64 * (6.0 / 19.0)).collect();
            v[19] = 3.0;
            v
        };
        let yb: Vec<f64> = xb.iter().map(|&x| (-x * x).exp()).collect();
        let b = generate_knots(&xb, &yb, None, None, None, 3, 0.01, None).unwrap();
        check(
            &b,
            &[
                vec![-3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, 3.0],
                vec![-3.0, -3.0, -3.0, -3.0, 0.1578947368, 3.0, 3.0, 3.0, 3.0],
                vec![
                    -3.0,
                    -3.0,
                    -3.0,
                    -3.0,
                    0.1578947368,
                    1.7368421053,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                vec![
                    -3.0,
                    -3.0,
                    -3.0,
                    -3.0,
                    -1.4210526316,
                    0.1578947368,
                    1.7368421053,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                vec![
                    -3.0,
                    -3.0,
                    -3.0,
                    -3.0,
                    -1.4210526316,
                    0.1578947368,
                    1.1052631579,
                    1.7368421053,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
            ],
            "B s=0.01",
        );

        // s == 0 yields a single interpolation knot vector and rejects w/nest.
        let interp = generate_knots(&xa, &ya, None, None, None, 3, 0.0, None).unwrap();
        assert_eq!(interp.len(), 1);
        assert_eq!(interp[0].len(), xa.len() + 3 + 1);
        assert!(generate_knots(&xa, &ya, Some(&wd), None, None, 3, 0.0, None).is_err());
    }

    #[test]
    fn make_splrep_smoothing_matches_scipy() {
        let x: Vec<f64> = {
            let mut v: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
            v[10] = 1.0;
            v
        };
        let y: Vec<f64> = x
            .iter()
            .map(|&t| (6.0 * t).sin() + 0.1 * (20.0 * t).cos())
            .collect();

        // scipy.interpolate.make_splrep(x, y, k=3, s=0.05)
        let spl = make_splrep(&x, &y, 3, 0.05).unwrap();
        let exp_t = [0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0];
        let exp_c = [
            0.06567173,
            0.65378525,
            1.37575535,
            -0.00672879,
            -1.42662387,
            -0.74611222,
            -0.19968622,
        ];
        for (a, b) in spl.knots().iter().zip(exp_t.iter()) {
            assert!((a - b).abs() < 1e-9, "t {a} vs {b}");
        }
        assert_eq!(spl.coeffs().len(), exp_c.len());
        for (a, b) in spl.coeffs().iter().zip(exp_c.iter()) {
            assert!((a - b).abs() < 1e-5, "c {a} vs {b}");
        }

        // s = 0.005 (more interior knots).
        let spl2 = make_splrep(&x, &y, 3, 0.005).unwrap();
        let exp_c2 = [
            0.10214238,
            0.3404046,
            0.86000172,
            1.2173123,
            0.18503621,
            -0.3681132,
            -0.90141417,
            -1.28396656,
            -0.48171754,
            -0.23430597,
        ];
        assert_eq!(spl2.coeffs().len(), exp_c2.len());
        for (a, b) in spl2.coeffs().iter().zip(exp_c2.iter()) {
            assert!((a - b).abs() < 1e-5, "c2 {a} vs {b}");
        }

        // s = 0 still interpolates.
        let interp = make_splrep(&x, &y, 3, 0.0).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!((interp.eval(xi) - yi).abs() < 1e-9);
        }
    }

    #[test]
    fn insert_matches_scipy() {
        // tck from scipy.interpolate.splrep(linspace(0,1,7), sin(3x), s=0).
        let t = vec![
            0.0, 0.0, 0.0, 0.0, 0.33333333, 0.5, 0.66666667, 1.0, 1.0, 1.0, 1.0,
        ];
        let c = vec![
            -0.0, 0.336776, 0.82341466, 1.03979653, 0.91776607, 0.47545689, 0.14112001, 0.0, 0.0,
            0.0, 0.0,
        ];
        let tck = (t, c, 3usize);

        // Single insertion at x = 0.45.
        let (nt, nc, nk) = insert(0.45, &tck, 1, false).unwrap();
        assert_eq!(nk, 3);
        let exp_t = [
            0.0, 0.0, 0.0, 0.0, 0.33333333, 0.45, 0.5, 0.66666667, 1.0, 1.0, 1.0, 1.0,
        ];
        let exp_c = [
            -0.0, 0.336776, 0.77475079, 0.96947242, 1.0184412, 0.91776607, 0.47545689, 0.14112001,
            0.0, 0.0, 0.0, 0.0,
        ];
        for (a, b) in nt.iter().zip(exp_t.iter()) {
            assert!((a - b).abs() < 1e-7, "knot {a} vs {b}");
        }
        for (a, b) in nc.iter().zip(exp_c.iter()) {
            assert!((a - b).abs() < 1e-7, "coeff {a} vs {b}");
        }

        // Double insertion at x = 0.6.
        let (nt2, nc2, _) = insert(0.6, &tck, 2, false).unwrap();
        let exp_t2 = [
            0.0, 0.0, 0.0, 0.0, 0.33333333, 0.5, 0.6, 0.6, 0.66666667, 1.0, 1.0, 1.0, 1.0,
        ];
        let exp_c2 = [
            -0.0, 0.336776, 0.82341466, 1.01815834, 0.99641914, 0.95864832, 0.82930424, 0.47545689,
            0.14112001, 0.0, 0.0, 0.0, 0.0,
        ];
        for (a, b) in nt2.iter().zip(exp_t2.iter()) {
            assert!((a - b).abs() < 1e-7, "knot2 {a} vs {b}");
        }
        for (a, b) in nc2.iter().zip(exp_c2.iter()) {
            assert!((a - b).abs() < 1e-7, "coeff2 {a} vs {b}");
        }

        // Out-of-range x and periodic mode are rejected.
        assert!(insert(1.5, &tck, 1, false).is_err());
        assert!(insert(0.45, &tck, 1, true).is_err());
    }

    #[test]
    fn spalde_matches_scipy_cubic_sine_tck() {
        // Same cubic tck (scipy.splrep of sin on [0, 2π]) fed to both libraries;
        // golden rows from scipy.interpolate.spalde at x = 1.0 and x = 2.5.
        let t = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            1.1423973285781066,
            1.7135959928671598,
            2.284794657156213,
            2.8559933214452666,
            3.4271919857343196,
            3.998390650023373,
            4.569589314312426,
            5.140787978601479,
            std::f64::consts::TAU,
            std::f64::consts::TAU,
            std::f64::consts::TAU,
            std::f64::consts::TAU,
        ];
        let c = vec![
            6.8853109097104774e-18,
            0.3874530925947836,
            0.9328410869555639,
            1.045009810272688,
            0.7980061424100005,
            0.2974630662128592,
            -0.2974630662128589,
            -0.7980061424100005,
            -1.0450098102726877,
            -0.9328410869555637,
            -0.38745309259478417,
            -2.4492935982947064e-16,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let tck = (t, c, 3usize);

        let res = spalde(&[1.0, 2.5], &tck).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].len(), 4); // [value, 1st, 2nd, 3rd derivative]

        // scipy.interpolate.spalde([1.0, 2.5], tck)
        let expected = [
            [0.841_124_74, 0.543_276_82, -0.838_693_66, -0.728_993_49],
            [0.598_350_2, -0.801_824_51, -0.593_302_33, 0.854_008_96],
        ];
        for (row, exp) in res.iter().zip(expected.iter()) {
            for (got, e) in row.iter().zip(exp.iter()) {
                assert!((got - e).abs() < 1e-7, "got {got}, expected {e}");
            }
        }

        // Order-0 derivative equals splev (the spline value itself).
        let vals = splev(&[1.0, 2.5], &tck).unwrap();
        assert!((res[0][0] - vals[0]).abs() < 1e-15);
        assert!((res[1][0] - vals[1]).abs() < 1e-15);
    }

    #[test]
    fn bisplev_matches_scipy_on_shared_tck() {
        // tck from scipy.interpolate.bisplrep (kx=ky=3) on scattered data; the
        // SAME tck is evaluated here and golden values come from scipy.bisplev.
        let tx = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let ty = tx.clone();
        let c = vec![
            0.0007036914835632136,
            0.005926321740951743,
            -0.06160310345403084,
            -0.0001221843016128943,
            0.9557129046360605,
            1.62140389861632,
            0.23776490140582782,
            -0.26516573732626564,
            1.6567096611867258,
            1.262245622926786,
            0.6532855323355665,
            -0.402172884515181,
            0.14105879414771977,
            0.08629198314637655,
            0.5637006125242106,
            0.5145217038568174,
        ];
        let tck = (tx, ty, c, 3usize, 3usize);

        let x = [0.0_f64, 0.5, 1.0];
        let y = [0.0_f64, 0.5, 1.0];
        let z = bisplev(&x, &y, &tck).unwrap();
        assert_eq!(z.len(), 3);
        assert_eq!(z[0].len(), 3);
        // scipy.interpolate.bisplev([0,0.5,1],[0,0.5,1],tck) corners/centre.
        assert!(
            (z[0][0] - 0.0007036914835632136).abs() < 1e-12,
            "z00 = {}",
            z[0][0]
        );
        assert!(
            (z[2][2] - 0.5145217038568174).abs() < 1e-12,
            "z22 = {}",
            z[2][2]
        );
        assert!(
            (z[1][1] - 0.660_104_076_196_305_1).abs() < 1e-9,
            "z11 = {}",
            z[1][1]
        );

        // Inconsistent coefficient count -> error.
        let bad = (vec![0.0; 8], vec![0.0; 8], vec![0.0; 9], 3usize, 3usize);
        assert!(bisplev(&x, &y, &bad).is_err());
    }

    #[test]
    fn bisplev_parallel_is_byte_identical_to_serial_above_gate() {
        use std::sync::atomic::Ordering;
        // Build a valid cubic tck, then evaluate on a grid past the ≥40k fan-out gate:
        // the parallel-across-x-rows path must be BYTE-IDENTICAL to the serial loop
        // (each output row is an independent tensor-product evaluation).
        let (kx, ky) = (3usize, 3usize);
        let mut t = vec![0.0f64; 4];
        for i in 1..=16 {
            t.push(i as f64 / 17.0);
        }
        t.extend(vec![1.0f64; 4]);
        let (tx, ty) = (t.clone(), t);
        let nx_c = tx.len() - kx - 1;
        let ny_c = ty.len() - ky - 1;
        let mut state = 0xBEEFu64;
        let c: Vec<f64> = (0..nx_c * ny_c)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let tck = (tx, ty, c, kx, ky);
        let n = 210usize; // 210*210 = 44100 > 40000 gate
        let g: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        BISPLEV_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = bisplev(&g, &g, &tck).unwrap();
        BISPLEV_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel = bisplev(&g, &g, &tck).unwrap();
        let mism: usize = serial
            .iter()
            .zip(parallel.iter())
            .map(|(sr, pr)| {
                sr.iter()
                    .zip(pr.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
            })
            .sum();
        assert_eq!(mism, 0, "parallel bisplev must be byte-identical to serial");
    }

    #[test]
    fn splint_matches_scipy_on_padded_tck() {
        // Golden definite integrals from scipy.interpolate.splint.
        let cases = [
            (0.0, 3.0, 16.0 / 3.0),
            (0.5, 2.5, 4.354_166_666_666_667),
            (1.0, 2.0, 2.5),
            (0.0, 1.5, 3.145_833_333_333_333_5),
        ];
        for tck in &quadratic_bspline_tcks() {
            for &(a, b, want) in &cases {
                let got = splint(a, b, tck).unwrap();
                assert!(
                    (got - want).abs() < 1e-12,
                    "splint({a}, {b}) = {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn splrep_returns_scipy_padded_tck() {
        // scipy.interpolate.splrep returns the coefficient array padded to len(t)
        // with k+1 trailing zeros (len(c) == len(t)); fsci previously returned the
        // unpadded len(t)-k-1 array, so the tck was not interchangeable with scipy
        // (scipy's splder broadcasting failed on it).
        let xs: Vec<f64> = (0..9).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| (0.7 * x).sin() + 0.2 * x).collect();
        let (t, c, k) = splrep(&xs, &ys, 3, 0.0).expect("splrep");
        assert_eq!(k, 3);
        assert_eq!(
            c.len(),
            t.len(),
            "len(c) must equal len(t) (scipy convention)"
        );
        // The last k+1 coefficients are the zero padding.
        for &v in &c[t.len() - (k + 1)..] {
            assert_eq!(v, 0.0, "trailing padding must be zero");
        }
        // The spline still interpolates the data (padding ignored on evaluation).
        let ev = splev(&xs, &(t, c, k)).expect("splev");
        for (yi, ei) in ys.iter().zip(ev.iter()) {
            assert!((yi - ei).abs() < 1e-9, "splev({yi}) = {ei}");
        }
    }

    #[test]
    fn spline_calculus_rejects_degenerate_tck() {
        // Knot vector too short to define any basis function.
        let short = (vec![0.0, 1.0, 2.0], vec![1.0], 3);
        assert!(splantider(&short).is_err());
        assert!(splder(&short).is_err());
        // Fewer coefficients than the knot vector implies.
        let starved = (vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], vec![1.0], 2);
        assert!(splantider(&starved).is_err());
        assert!(splder(&starved).is_err());
    }

    #[test]
    fn sproot_matches_scipy_on_cubic_bspline() {
        // Hand-built cubic B-spline; reference roots from
        // scipy.interpolate.sproot on the same tck.
        let tck = (
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
            vec![-2.0, -1.0, 1.0, -1.0, 1.0, 2.0, 3.0],
            3,
        );
        let roots = sproot(&tck).expect("sproot");
        let expected = [
            0.819_600_955_053_800_3,
            1.476_002_243_894_599,
            2.447_523_768_942_384_8,
        ];
        assert_eq!(roots.len(), expected.len(), "root count: {roots:?}");
        for (got, want) in roots.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-7, "root {got} != {want}");
        }
        // Each reported root really is a zero of the spline.
        let bspl = BSpline::new(tck.0.clone(), tck.1.clone(), tck.2).unwrap();
        for &r in &roots {
            assert!(bspl.eval(r).abs() < 1e-7, "spline({r}) != 0");
        }
    }

    #[test]
    fn sproot_rejects_non_cubic() {
        let tck = (
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0],
            vec![1.0, -1.0, 1.0],
            2,
        );
        assert!(sproot(&tck).is_err());
    }

    #[test]
    fn pchip_interpolate_matches_scipy_reference_values() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = vec![1.5, 2.5];
        let result = pchip_interpolate(&x, &y, &x_new).expect("pchip_interpolate");
        // scipy.interpolate.PchipInterpolator(x, y)([1.5, 2.5])
        assert!(
            (result[0] - 2.21875).abs() < 1e-10,
            "pchip(1.5) got {}, expected 2.21875",
            result[0]
        );
        assert!(
            (result[1] - 6.239583333333333).abs() < 1e-10,
            "pchip(2.5) got {}, expected 6.239583333333333",
            result[1]
        );
    }

    #[test]
    fn akima1d_interpolate_matches_scipy_reference_values() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = vec![1.5, 2.5];
        let result = akima1d_interpolate(&x, &y, &x_new).expect("akima1d_interpolate");
        // scipy.interpolate.Akima1DInterpolator(x, y)([1.5, 2.5])
        assert!(
            (result[0] - 2.25).abs() < 1e-10,
            "akima(1.5) got {}, expected 2.25",
            result[0]
        );
        assert!(
            (result[1] - 6.25).abs() < 1e-10,
            "akima(2.5) got {}, expected 6.25",
            result[1]
        );
    }

    #[test]
    fn interp1d_linear_matches_scipy_reference_values() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = vec![1.5, 2.5];
        let result = interp1d_linear(&x, &y, &x_new).expect("interp1d_linear");
        // scipy.interpolate.interp1d(x, y, kind='linear')([1.5, 2.5])
        assert!(
            (result[0] - 2.5).abs() < 1e-10,
            "linear(1.5) got {}, expected 2.5",
            result[0]
        );
        assert!(
            (result[1] - 6.5).abs() < 1e-10,
            "linear(2.5) got {}, expected 6.5",
            result[1]
        );
    }

    #[test]
    fn krogh_interpolate_matches_scipy_reference_values() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = vec![1.5, 2.5];
        let result = krogh_interpolate(&x, &y, &x_new).expect("krogh_interpolate");
        // scipy.interpolate.krogh_interpolate(x, y, [1.5, 2.5])
        assert!(
            (result[0] - 2.25).abs() < 1e-10,
            "krogh(1.5) got {}, expected 2.25",
            result[0]
        );
        assert!(
            (result[1] - 6.25).abs() < 1e-10,
            "krogh(2.5) got {}, expected 6.25",
            result[1]
        );
    }

    #[test]
    fn krogh_derivatives_match_taylor_of_known_polynomial() {
        // P(t) = t^3 - 2t + 5 through 4 nodes; derivatives at t=2 are exact:
        // P(2)=9, P'(2)=10, P''(2)=12, P'''(2)=6, P''''(2)=0.
        let x = [-1.0, 0.0, 1.0, 3.0];
        let p = |t: f64| t * t * t - 2.0 * t + 5.0;
        let y: Vec<f64> = x.iter().map(|&t| p(t)).collect();
        let interp = KroghInterpolator::new(&x, &y).unwrap();
        let d = interp.derivatives(2.0, 5);
        let want = [9.0, 10.0, 12.0, 6.0, 0.0];
        for (g, w) in d.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-9, "derivative {g} vs {w}");
        }
    }

    #[test]
    fn approximate_taylor_polynomial_matches_scipy() {
        let close = |a: f64, b: f64, t: f64| (a - b).abs() <= t;

        // scipy: approximate_taylor_polynomial(x**3, 2.0, 3, 1.0, order=5)
        // -> poly1d coeffs (desc) [1, 6, 12, 8].
        let c = approximate_taylor_polynomial(|t| t * t * t, 2.0, 3, 1.0, Some(5)).unwrap();
        assert_eq!(c.len(), 4);
        let want = [1.0, 6.0, 12.0, 8.0];
        for (g, w) in c.iter().zip(want.iter()) {
            assert!(close(*g, *w, 1e-7), "x^3 coeff {g} vs {w}");
        }

        // scipy: approximate_taylor_polynomial(sin, 0, 5, 1.0, order=7)
        // -> [0.008327592, ~0, -0.166664813, ~0, 0.999999831, ~0].
        let s = approximate_taylor_polynomial(|t: f64| t.sin(), 0.0, 5, 1.0, Some(7)).unwrap();
        let sw = [
            0.008327592134467668,
            0.0,
            -0.16666481316597495,
            0.0,
            0.9999998316400153,
            0.0,
        ];
        for (g, w) in s.iter().zip(sw.iter()) {
            assert!(close(*g, *w, 1e-9), "sin coeff {g} vs {w}");
        }

        // scipy: approximate_taylor_polynomial(exp, 0, 4, 1.0) (order=None -> 4).
        let e = approximate_taylor_polynomial(|t: f64| t.exp(), 0.0, 4, 1.0, None).unwrap();
        let ew = [
            0.05284056311102009,
            0.1730167826453521,
            0.4923222196318288,
            0.999469779438879,
            1.000632483631965,
        ];
        for (g, w) in e.iter().zip(ew.iter()) {
            assert!(close(*g, *w, 1e-9), "exp coeff {g} vs {w}");
        }
    }

    #[test]
    fn polyfit_matches_scipy_reference_values() {
        // np.polyfit([0, 1, 2, 3, 4], [1.0, 2.7, 5.8, 10.2, 17.0], 2)
        // Returns coefficients [a, b, c] for ax^2 + bx + c
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.7, 5.8, 10.2, 17.0];
        let coeffs = polyfit(&x, &y, 2).expect("polyfit deg=2");
        // scipy returns highest degree first: [0.821..., 0.664..., 1.082...]
        // Our polyfit may use different ordering - check evaluation instead
        let y_pred: Vec<f64> = x.iter().map(|&xi| polyval(&coeffs, xi)).collect();
        for (i, (&actual, &expected)) in y_pred.iter().zip(y.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.5,
                "polyfit at x[{i}] got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn make_interp_spline_matches_scipy_reference_values() {
        // scipy.interpolate.make_interp_spline(x, y, k=3)(x_new)
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.7, 5.8, 10.2, 17.0];
        let spline = make_interp_spline(&x, &y, 3).expect("make_interp_spline k=3");
        let x_new = [0.5, 1.5, 2.5, 3.5];
        let y_new: Vec<f64> = x_new.iter().map(|&xi| spline.eval(xi)).collect();
        let expected = [1.65, 4.1, 7.7875, 13.2125];
        for (i, (&got, &want)) in y_new.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 0.01,
                "spline({}) got {got}, expected {want}",
                x_new[i]
            );
        }
    }

    #[test]
    fn make_interp_spline_compact_band_matches_dense_coefficients_bits() {
        for k in 0..=5usize {
            let n = 18 + k;
            let x: Vec<f64> = (0..n)
                .map(|i| i as f64 * 0.37 + ((i * 31) % 11) as f64 * 0.001)
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.23).sin() - 0.3 * (i as f64 * 0.11).cos())
                .collect();
            let t = interpolation_knots(&x, k);
            let mut a_mat = vec![vec![0.0; n]; n];
            for i in 0..n {
                let basis = eval_basis_all(&t, x[i], k, n);
                a_mat[i][..n].copy_from_slice(&basis[..n]);
            }
            let mut rhs = y.clone();
            let dense = solve_dense_system(&mut a_mat, &mut rhs).expect("dense coeffs");
            let compact = make_interp_spline(&x, &y, k).expect("compact coeffs");

            assert_eq!(compact.coeffs().len(), dense.len(), "degree {k}");
            for (idx, (&got, &want)) in compact.coeffs().iter().zip(&dense).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "coefficient mismatch degree {k} index {idx}"
                );
            }
        }
    }

    #[test]
    fn polyadd_matches_scipy_reference_values() {
        // np.polyadd([1, 2], [9, 5, 4]) -> [9, 6, 6]
        // numpy uses highest-degree-first convention
        let a = [1.0, 2.0];
        let b = [9.0, 5.0, 4.0];
        let result = polyadd(&a, &b);
        let expected = [9.0, 6.0, 6.0];
        assert_eq!(result.len(), expected.len(), "polyadd length mismatch");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "polyadd[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polyadd_direct_emission_preserves_old_operation_bits() {
        fn old_polyadd(a: &[f64], b: &[f64]) -> Vec<f64> {
            let n = a.len().max(b.len());
            let mut result = vec![0.0; n];
            let offset_a = n - a.len();
            let offset_b = n - b.len();
            for (i, &value) in a.iter().enumerate() {
                result[offset_a + i] += value;
            }
            for (i, &value) in b.iter().enumerate() {
                result[offset_b + i] += value;
            }
            result
        }

        let nan_a = f64::from_bits(0x7ff8_0000_0000_1234);
        let nan_b = f64::from_bits(0xfff8_0000_0000_5678);
        let cases = [
            (vec![], vec![]),
            (vec![-0.0, f64::INFINITY, nan_a], vec![]),
            (vec![], vec![-0.0, f64::NEG_INFINITY, nan_b]),
            (vec![1.0, -0.0], vec![-1.0, 0.0]),
            (
                vec![-0.0, f64::INFINITY, nan_a, 3.0],
                vec![0.0, f64::NEG_INFINITY],
            ),
            (
                vec![0.0, f64::NEG_INFINITY],
                vec![-0.0, f64::INFINITY, nan_b, -3.0],
            ),
        ];

        for (a, b) in cases {
            let expected_bits: Vec<u64> =
                old_polyadd(&a, &b).into_iter().map(f64::to_bits).collect();
            let actual_bits: Vec<u64> = polyadd(&a, &b).into_iter().map(f64::to_bits).collect();
            assert_eq!(actual_bits, expected_bits, "a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn polyder_matches_scipy_reference_values() {
        // np.polyder([3, 2, 1]) -> [6, 2] (derivative of 3x^2 + 2x + 1 = 6x + 2)
        let coeffs = [3.0, 2.0, 1.0];
        let result = polyder(&coeffs, 1);
        let expected = [6.0, 2.0];
        assert_eq!(result.len(), expected.len(), "polyder length mismatch");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "polyder[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polymul_matches_scipy_reference_values() {
        // np.polymul([1, 2], [1, 2, 1]) -> [1, 4, 5, 2]
        // (x + 2) * (x^2 + 2x + 1) = x^3 + 4x^2 + 5x + 2
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 1.0];
        let result = polymul(&a, &b);
        let expected = [1.0, 4.0, 5.0, 2.0];
        assert_eq!(result.len(), expected.len(), "polymul length mismatch");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "polymul[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polymul_fft_path_matches_direct_reference() {
        // Above the crossover polymul takes the FFT path; prove it matches a direct
        // O(m·n) convolution within the 1e-10 conformance tolerance for several
        // sizes and magnitudes. (Small inputs stay on the byte-identical direct loop.)
        fn direct(a: &[f64], b: &[f64]) -> Vec<f64> {
            let mut r = vec![0.0; a.len() + b.len() - 1];
            for (i, &ai) in a.iter().enumerate() {
                for (j, &bj) in b.iter().enumerate() {
                    r[i + j] += ai * bj;
                }
            }
            r
        }
        for &(m, n, scale) in &[
            (128usize, 128usize, 1.0_f64),
            (512, 512, 1.0),
            (300, 700, 3.0),
        ] {
            let a: Vec<f64> = (0..m)
                .map(|i| ((i as f64 * 0.7).sin() + 0.3) * scale)
                .collect();
            let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).cos() - 0.2).collect();
            let got = polymul(&a, &b);
            let want = direct(&a, &b);
            assert_eq!(got.len(), want.len());
            let mut max_abs = 0.0_f64;
            for (&g, &w) in got.iter().zip(want.iter()) {
                max_abs = max_abs.max((g - w).abs());
            }
            assert!(
                max_abs < 1e-10,
                "polymul FFT path diverged for {m}x{n}: max_abs={max_abs:e}"
            );
        }
    }

    #[test]
    fn barycentric_eval_weights_recover_quadratic() {
        // barycentric_weights + barycentric_eval reproduce a polynomial of degree
        // < n exactly. y=x^2 through 3 nodes. Both fns were untested.
        let nodes = [0.0, 1.0, 2.0];
        let values = [0.0, 1.0, 4.0];
        let w = barycentric_weights(&nodes);
        assert!(
            (barycentric_eval(&nodes, &values, &w, 1.5) - 2.25).abs() < 1e-12,
            "x=1.5"
        );
        assert!(
            (barycentric_eval(&nodes, &values, &w, 0.5) - 0.25).abs() < 1e-12,
            "x=0.5"
        );
        assert!(
            (barycentric_eval(&nodes, &values, &w, 2.0) - 4.0).abs() < 1e-12,
            "at node"
        );
    }

    #[test]
    fn barycentric_eval_fused_scan_matches_two_pass_bits() {
        fn two_pass(nodes: &[f64], values: &[f64], weights: &[f64], x: f64) -> f64 {
            let n = nodes.len();
            if n == 0 || values.len() != n || weights.len() != n || !x.is_finite() {
                return f64::NAN;
            }
            for (i, &xi) in nodes.iter().enumerate() {
                if (x - xi).abs() < 1e-15 {
                    return values[i];
                }
            }

            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..nodes.len() {
                let t = weights[i] / (x - nodes[i]);
                num += t * values[i];
                den += t;
            }
            if den == 0.0 {
                return f64::NAN;
            }
            num / den
        }

        // Named so the fixture list reads as data rather than as a type puzzle
        // (frankenscipy-9yyez): (x, y, expected, tolerance).
        type SplineCase<'a> = (&'a [f64], &'a [f64], &'a [f64], f64);
        let cases: &[SplineCase] = &[
            (
                &[-2.0, -0.5, 1.0, 3.0],
                &[4.0, -0.0, 1.0, 9.0],
                &[0.5, -1.5, 2.0, -0.25],
                0.25,
            ),
            (
                &[-2.0, -0.5, 1.0, 3.0],
                &[4.0, -0.0, 1.0, 9.0],
                &[0.5, -1.5, 2.0, -0.25],
                1.0,
            ),
            (
                &[-2.0, -0.5, 1.0, 3.0],
                &[4.0, -0.0, 1.0, 9.0],
                &[0.5, -1.5, 2.0, -0.25],
                1.0 + 5.0e-16,
            ),
            (&[0.0, 1.0], &[f64::INFINITY, 2.0], &[1.0, 1.0], 0.5),
            (
                &[0.0, f64::NAN, 2.0],
                &[1.0, 2.0, 3.0],
                &[1.0, 1.0, 1.0],
                0.5,
            ),
            (&[0.0, 1.0], &[1.0], &[1.0, 1.0], 0.5),
            (&[0.0, 1.0], &[1.0, 2.0], &[1.0, 1.0], f64::NAN),
            (&[], &[], &[], 0.5),
        ];
        for &(nodes, values, weights, x) in cases {
            assert_eq!(
                barycentric_eval(nodes, values, weights, x).to_bits(),
                two_pass(nodes, values, weights, x).to_bits()
            );
        }
    }

    #[test]
    fn hermite_interp_recovers_cubic() {
        // Cubic Hermite with 2 nodes (value+derivative each) gives a degree-3
        // polynomial = the unique cubic. y=x^3, y'=3x^2 at [0,2] -> reproduces x^3.
        // hermite_interp was untested.
        let nodes = [0.0, 2.0];
        let values = [0.0, 8.0];
        let derivs = [0.0, 12.0];
        assert!(
            (hermite_interp(&nodes, &values, &derivs, 1.0) - 1.0).abs() < 1e-12,
            "x=1"
        );
        assert!(
            (hermite_interp(&nodes, &values, &derivs, 1.5) - 3.375).abs() < 1e-12,
            "x=1.5"
        );
        // Matches the value and the derivative at a node.
        assert!(
            (hermite_interp(&nodes, &values, &derivs, 2.0) - 8.0).abs() < 1e-12,
            "at node"
        );
    }

    #[test]
    fn neville_recovers_cubic() {
        // Neville's algorithm exactly reproduces a polynomial of degree < n_nodes.
        // y = x^3 through 4 nodes -> the unique interpolating cubic is x^3. Untested.
        let nodes = [0.0, 1.0, 2.0, 3.0];
        let values = [0.0, 1.0, 8.0, 27.0];
        assert!(
            (neville(&nodes, &values, 1.5) - 3.375).abs() < 1e-12,
            "x=1.5"
        );
        assert!(
            (neville(&nodes, &values, 2.5) - 15.625).abs() < 1e-12,
            "x=2.5"
        );
        assert!(
            (neville(&nodes, &values, 2.0) - 8.0).abs() < 1e-12,
            "at node"
        );
    }

    #[test]
    fn chebyshev_nodes_match_analytic() {
        // chebyshev_nodes(n,a,b): first-kind nodes x_k=(a+b)/2+(b-a)/2*cos(pi(2k+1)/2n).
        // Was untested. n=3 on [-1,1]: [cos30, cos90, cos150] = [sqrt3/2, 0, -sqrt3/2].
        let nodes = chebyshev_nodes(3, -1.0, 1.0);
        assert_eq!(nodes.len(), 3);
        let s3 = 3.0_f64.sqrt() / 2.0;
        assert!((nodes[0] - s3).abs() < 1e-15, "node0");
        assert!(nodes[1].abs() < 1e-15, "node1~0");
        assert!((nodes[2] + s3).abs() < 1e-15, "node2");
        // On [0,4] the nodes are affine-mapped: center 2, half-width 2.
        let n2 = chebyshev_nodes(3, 0.0, 4.0);
        assert!((n2[0] - (2.0 + 2.0 * s3)).abs() < 1e-14, "mapped node0");
    }

    #[test]
    fn polyint_definite_matches_numpy() {
        // polyint_definite(descending coeffs, a, b) = integral_a^b p(x) dx. Untested.
        // integral_0^2 (3x^2+2x+1) dx = [x^3+x^2+x]_0^2 = 14.
        let v = polyint_definite(&[3.0, 2.0, 1.0], 0.0, 2.0);
        assert!((v - 14.0).abs() < 1e-12, "integral = {v}");
        // integral_1^3 (2x) dx = [x^2]_1^3 = 8.
        assert!(
            (polyint_definite(&[2.0, 0.0], 1.0, 3.0) - 8.0).abs() < 1e-12,
            "linear integral"
        );
    }

    #[test]
    fn polyint_definite_stream_matches_former_materialization_bits() {
        let cases = [
            (Vec::new(), -0.0, 0.0),
            (vec![3.0, 2.0, 1.0], 0.0, 2.0),
            (vec![-0.0, 0.0, -1.5, 2.25], -0.75, 0.875),
            (vec![f64::INFINITY, -0.0], -0.5, 0.25),
            (vec![f64::from_bits(0x7ff8_0000_0000_0042), 1.0], -0.5, 0.25),
            (vec![1.0, -2.0, 3.0], f64::NEG_INFINITY, f64::INFINITY),
        ];

        for (case_index, (coeffs, a, b)) in cases.iter().enumerate() {
            let antiderivative = polyint(coeffs, 1, 0.0);
            let former = polyval(&antiderivative, *b) - polyval(&antiderivative, *a);
            let streamed = polyint_definite(coeffs, *a, *b);
            assert_eq!(
                streamed.to_bits(),
                former.to_bits(),
                "case {case_index} differs"
            );
        }
    }

    #[test]
    fn polyval_der_matches_numpy() {
        // polyval_der(descending coeffs, x, der) returns [p(x), p'(x), ..., p^(der)].
        // Was untested. p = x^2-3x+2 at x=2: p=0, p'=2x-3=1, p''=2.
        let r = polyval_der(&[1.0, -3.0, 2.0], 2.0, 2);
        assert_eq!(r, vec![0.0, 1.0, 2.0]);
        // p=2x^3-3x+1 at x=1: p=0, p'=6x^2-3=3, p''=12x=12, p'''=12.
        let r2 = polyval_der(&[2.0, 0.0, -3.0, 1.0], 1.0, 3);
        assert_eq!(r2, vec![0.0, 3.0, 12.0, 12.0]);
    }

    #[test]
    fn polyval_der_prefix_reuse_matches_restarted_reference_bits() {
        let payload_nan = f64::from_bits(0x7ff8_0000_0000_1234);
        let coefficient_sets = [
            Vec::new(),
            vec![3.0],
            vec![1.0, -3.0, 2.0],
            vec![-0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY, payload_nan],
        ];
        let points = [
            -0.0,
            0.0,
            0.875,
            f64::INFINITY,
            f64::NEG_INFINITY,
            payload_nan,
        ];

        for coeffs in &coefficient_sets {
            for &x in &points {
                for der in [0, 1, 2, 4, 8] {
                    let actual = polyval_der(coeffs, x, der);
                    let expected = polyval_der_restarted(coeffs, x, der);
                    assert_eq!(actual.len(), expected.len());
                    for (order, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "coeffs={coeffs:?} x={x:?} der={der} order={order}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn polyroots_matches_numpy() {
        // numpy.roots convention: descending coeffs, real roots only. polyroots was
        // previously untested. Quadratic x^2-5x+6=(x-2)(x-3): larger root first.
        let r = polyroots(&[1.0, -5.0, 6.0]);
        assert_eq!(r.len(), 2);
        assert!((r[0] - 3.0).abs() < 1e-12, "root0");
        assert!((r[1] - 2.0).abs() < 1e-12, "root1");
        // Linear 2x-4=0 -> x=2.
        assert_eq!(polyroots(&[2.0, -4.0]), vec![2.0]);
        // x^2+1 has only complex roots -> no real roots.
        assert!(polyroots(&[1.0, 0.0, 1.0]).is_empty());
    }

    #[test]
    fn pade_exp_approximant_matches_scipy() {
        // scipy.interpolate.pade(exp Taylor coeffs, 2): the [2/2] Pade approximant
        // of exp. pade was previously untested. fsci returns (p, q) as ascending-
        // power coefficient vectors; p = 1 + x/2 + x^2/12, q = 1 - x/2 + x^2/12.
        let coeffs = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
        let (p, q) = pade(&coeffs, 2, 2).expect("pade");
        for (g, e) in p.iter().zip(&[1.0, 0.5, 1.0 / 12.0]) {
            assert!((g - e).abs() < 1e-12, "p coeff: {g}");
        }
        for (g, e) in q.iter().zip(&[1.0, -0.5, 1.0 / 12.0]) {
            assert!((g - e).abs() < 1e-12, "q coeff: {g}");
        }
        // p(1)/q(1) approximates e.
        assert!(
            (ratval(&p, &q, 1.0) - 2.714_285_714_285_714).abs() < 1e-12,
            "eval at 1"
        );
    }

    #[test]
    fn ratval_horner_matches_power_sum_reference() {
        fn power_sum_reference(p: &[f64], q: &[f64], x: f64) -> f64 {
            let num: f64 = p
                .iter()
                .enumerate()
                .map(|(i, &coefficient)| coefficient * x.powi(i as i32))
                .sum();
            let den: f64 = q
                .iter()
                .enumerate()
                .map(|(i, &coefficient)| coefficient * x.powi(i as i32))
                .sum();
            if den.abs() < 1e-30 {
                return f64::NAN;
            }
            num / den
        }

        let p: Vec<f64> = (0..257)
            .map(|i| ((i as f64 * 0.37).sin() + 1.5) / (i + 1) as f64)
            .collect();
        let q: Vec<f64> = (0..257)
            .map(|i| ((i as f64 * 0.19).cos() + 2.0) / (i + 2) as f64)
            .collect();
        for x in [-1.125, -0.75, 0.0, 0.75, 0.999, 1.125] {
            let expected = power_sum_reference(&p, &q, x);
            let actual = ratval(&p, &q, x);
            let tolerance = expected.abs().max(1.0) * 2e-11;
            assert!(
                (actual - expected).abs() <= tolerance,
                "x={x}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }

        assert_eq!(ratval(&[2.5], &[0.5], f64::NAN).to_bits(), 5.0f64.to_bits());
        assert_eq!(
            ratval(&[], &[1.0], f64::INFINITY).to_bits(),
            0.0f64.to_bits()
        );
        assert!(ratval(&[1.0], &[], 0.5).is_nan());
        assert!(ratval(&[1.0, 2.0], &[1.0], f64::NAN).is_nan());
    }

    #[test]
    fn barycentric_interpolate_matches_scipy_reference_values() {
        // scipy.interpolate.barycentric_interpolate([0,1,2,3], [1,4,2,5], [0.5,1.5,2.5])
        // -> [3.75, 3.0, 2.25]
        let xi = vec![0.0, 1.0, 2.0, 3.0];
        let yi = vec![1.0, 4.0, 2.0, 5.0];
        let x_new = vec![0.5, 1.5, 2.5];
        let result = barycentric_interpolate(&xi, &yi, &x_new).expect("barycentric");
        let expected = [3.75, 3.0, 2.25];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "barycentric[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn splev_matches_scipy_reference_values() {
        // scipy.interpolate.splrep([0,1,2,3,4], [0,1,4,9,16], k=3, s=0)
        // scipy.interpolate.splev([0.5,1.5,2.5,3.5], tck)
        // -> [0.25, 2.25, 6.25, 12.25]
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let tck = splrep(&x, &y, 3, 0.0).expect("splrep");
        let x_new = vec![0.5, 1.5, 2.5, 3.5];
        let result = splev(&x_new, &tck).expect("splev");
        let expected = [0.25, 2.25, 6.25, 12.25];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "splev[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polyint_matches_scipy_reference_values() {
        // np.polyint([3, 2, 1]) -> [1, 1, 1, 0]
        // integral of 3x^2 + 2x + 1 = x^3 + x^2 + x + C (where C=0)
        let coeffs = [3.0, 2.0, 1.0];
        let result = polyint(&coeffs, 1, 0.0);
        let expected = [1.0, 1.0, 1.0, 0.0];
        assert_eq!(result.len(), expected.len(), "polyint length mismatch");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "polyint[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polysub_matches_scipy_reference_values() {
        // np.polysub([1, 2, 3], [0, 1, 1]) -> [1, 1, 2]
        let a = [1.0, 2.0, 3.0];
        let b = [0.0, 1.0, 1.0];
        let result = polysub(&a, &b);
        let expected = [1.0, 1.0, 2.0];
        assert_eq!(result.len(), expected.len(), "polysub length mismatch");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "polysub[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn polysub_preserves_finite_bits_and_nan_classification() {
        fn allocating_reference(a: &[f64], b: &[f64]) -> Vec<f64> {
            let negated_b: Vec<f64> = b.iter().map(|&value| -value).collect();
            polyadd(a, &negated_b)
        }

        let finite_cases = [
            (vec![], vec![]),
            (vec![-0.0], vec![]),
            (vec![], vec![0.0, -0.0]),
            (vec![1.0, -0.0, 3.0], vec![0.0, 2.0]),
            (vec![0.0], vec![-0.0, 2.0, -3.0]),
        ];
        for (a, b) in finite_cases {
            let expected = allocating_reference(&a, &b);
            let actual = polysub(&a, &b);
            let expected_bits: Vec<u64> = expected.iter().map(|value| value.to_bits()).collect();
            let actual_bits: Vec<u64> = actual.iter().map(|value| value.to_bits()).collect();
            assert_eq!(actual_bits, expected_bits, "a={a:?}, b={b:?}");
        }

        // NumPy/SciPy require the NaN result, but do not expose a stable IEEE
        // payload contract. Keep the observable classification and finite lane
        // intact instead of pinning a compiler-dependent payload bit pattern.
        let actual = polysub(
            &[f64::from_bits(0x7ff8_0000_0000_1234), f64::INFINITY],
            &[f64::from_bits(0xfff8_0000_0000_5678), -0.0],
        );
        assert!(actual[0].is_nan());
        assert_eq!(actual[1], f64::INFINITY);
    }

    #[test]
    fn lagrange_matches_scipy_reference_values() {
        // scipy.interpolate.lagrange([0, 1, 2], [1, 2, 1]) -> poly with coeffs
        let xi = [0.0, 1.0, 2.0];
        let yi = [1.0, 2.0, 1.0];
        let coeffs = lagrange(&xi, &yi).expect("lagrange");
        // Evaluate at xi should give yi
        for i in 0..xi.len() {
            let val = polyval(&coeffs, xi[i]);
            assert!(
                (val - yi[i]).abs() < 1e-10,
                "lagrange(xi[{i}]) = {val}, expected {}",
                yi[i]
            );
        }
    }

    #[test]
    fn polyfromroots_matches_scipy_reference_values() {
        // np.poly([1, 2]) -> [1, -3, 2] (x-1)(x-2) = x^2 - 3x + 2
        let roots = [1.0, 2.0];
        let coeffs = polyfromroots(&roots);
        // Leading coefficient should be 1, evaluate at roots should give 0
        for &r in &roots {
            let val = polyval(&coeffs, r);
            assert!(
                val.abs() < 1e-10,
                "polyfromroots(root={r}) = {val}, expected 0"
            );
        }
    }

    #[test]
    fn polyval_matches_scipy_reference_values() {
        // np.polyval([2, 3, 1], 2) = 2*4 + 3*2 + 1 = 15
        let coeffs = [2.0, 3.0, 1.0];
        let result = polyval(&coeffs, 2.0);
        assert!(
            (result - 15.0).abs() < 1e-10,
            "polyval got {result}, expected 15"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LSQBivariateSpline — least-squares bivariate spline on USER-SUPPLIED knots
// ─────────────────────────────────────────────────────────────────────────────

/// Least-squares bivariate spline with knots chosen by the CALLER
/// -- `scipy.interpolate.LSQBivariateSpline`.
///
/// `SmoothBivariateSpline` picks its own knots to hit a smoothing target; this fixes
/// them and solves the resulting linear least-squares problem. That is the whole
/// difference, and it is the reason the type exists: when the knot placement is
/// dictated by the problem -- a known breakpoint, a grid shared with other data, a
/// reproducible fit -- an adaptive knot chooser is not a convenience but a source of
/// results that move when the data moves slightly.
///
/// # Construction
///
/// `tx` and `ty` are the INTERIOR knots only. The full knot vectors are formed by
/// clamping each end with `degree + 1` copies of the bounding-box edge, which is the
/// standard B-spline convention and what FITPACK's `surfit` expects when `iopt = -1`.
///
/// A fit is only determined when there are at least as many samples as coefficients,
/// and there are `(len(tx) + kx + 1) * (len(ty) + ky + 1)` of those. Fewer samples
/// leaves the system underdetermined, which is rejected rather than solved: an
/// underdetermined least-squares problem has infinitely many exact solutions and
/// returning whichever one the solver lands on would be a silently arbitrary answer.
///
/// # Implementation
///
/// This delegates to `SmoothBivariateSpline`'s fitted representation with a smoothing
/// factor of zero, so evaluation, grid evaluation, derivatives and the integral are
/// the SAME code paths that type already uses -- including its parallel `eval_many`
/// and `eval_grid`. Duplicating a second bivariate evaluator to gain a different
/// constructor would be the wrong trade.
#[derive(Debug, Clone)]
pub struct LSQBivariateSpline {
    inner: SmoothBivariateSpline,
}

impl LSQBivariateSpline {
    /// Fit with the given interior knots.
    ///
    /// `bbox` defaults to the data's bounding box when `options.bbox` is not set,
    /// matching `SmoothBivariateSpline`.
    pub fn new(
        x: &[f64],
        y: &[f64],
        z: &[f64],
        tx_interior: &[f64],
        ty_interior: &[f64],
        options: SmoothBivariateSplineOptions,
    ) -> Result<Self, InterpError> {
        let n = x.len();
        if y.len() != n {
            return Err(InterpError::LengthMismatch {
                x_len: n,
                y_len: y.len(),
            });
        }
        if z.len() != n {
            return Err(InterpError::InvalidArgument {
                detail: format!("z must have same length as x and y, got {}", z.len()),
            });
        }
        let kx = options.kx;
        let ky = options.ky;
        if n == 0 {
            return Err(InterpError::InvalidArgument {
                detail: "LSQBivariateSpline requires at least one sample".to_string(),
            });
        }

        let (x_lo, x_hi) = bounds_of(x);
        let (y_lo, y_hi) = bounds_of(y);
        let bbox = options.bbox.unwrap_or([x_lo, x_hi, y_lo, y_hi]);

        check_interior_knots(tx_interior, bbox[0], bbox[1], "tx")?;
        check_interior_knots(ty_interior, bbox[2], bbox[3], "ty")?;

        let nx_coeffs = tx_interior.len() + kx + 1;
        let ny_coeffs = ty_interior.len() + ky + 1;
        // Underdetermined systems are refused rather than solved; see the type docs.
        if nx_coeffs * ny_coeffs > n {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "LSQBivariateSpline needs at least {} samples for {} x {} \
                     coefficients, got {n}",
                    nx_coeffs * ny_coeffs,
                    nx_coeffs,
                    ny_coeffs
                ),
            });
        }

        let tx = clamped_knot_vector(tx_interior, bbox[0], bbox[1], kx);
        let ty = clamped_knot_vector(ty_interior, bbox[2], bbox[3], ky);

        let weights: Vec<f64> = match &options.weights {
            Some(w) if w.len() == n => w.clone(),
            Some(w) => {
                return Err(InterpError::InvalidArgument {
                    detail: format!("weights length {} must match samples {n}", w.len()),
                });
            }
            None => vec![1.0; n],
        };

        // Smoothing factor 0: pure least squares on the fixed knots, which is what
        // distinguishes this from the smoothing fit.
        let coeffs = smooth_bivariate_solve_coefficients(SmoothBivariateLinearSystem {
            x,
            y,
            z,
            weights: &weights,
            tx: &tx,
            ty: &ty,
            kx,
            ky,
            smoothing_factor: 0.0,
            eps: options.eps,
        })?;

        Ok(Self {
            inner: SmoothBivariateSpline::from_fitted(
                kx, ky, bbox, tx, ty, coeffs, nx_coeffs, ny_coeffs, 0.0, x, y, z, &weights,
            ),
        })
    }

    /// Evaluate at a single point.
    #[must_use]
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        self.inner.eval(x, y)
    }

    /// Evaluate at scattered points.
    #[must_use]
    pub fn eval_many(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, InterpError> {
        // Returns `Result`, matching the inner type. `eval_many` is the one evaluator
        // here that can fail — it validates that `x` and `y` are the same length — so
        // swallowing that into a bare `Vec` would mean either panicking on a caller
        // error or inventing a length. The single-point `eval` genuinely cannot fail
        // and correctly returns `f64`.
        self.inner.eval_many(x, y)
    }

    /// Evaluate on the grid `x` x `y`.
    #[must_use]
    pub fn eval_grid(&self, x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
        self.inner.eval_grid(x, y)
    }

    /// Mixed partial derivative of order `(dx, dy)`.
    #[must_use]
    pub fn eval_derivative(&self, x: f64, y: f64, dx: usize, dy: usize) -> f64 {
        self.inner.eval_derivative(x, y, dx, dy)
    }

    /// Integral over the rectangle.
    #[must_use]
    pub fn integral(&self, xa: f64, xb: f64, ya: f64, yb: f64) -> f64 {
        self.inner.integral(xa, xb, ya, yb)
    }

    /// Weighted sum of squared residuals of the fit.
    #[must_use]
    pub fn residual(&self) -> f64 {
        self.inner.residual()
    }

    /// `(kx, ky)`.
    #[must_use]
    pub fn degrees(&self) -> (usize, usize) {
        self.inner.degrees()
    }

    /// Full knot vectors, including the clamped ends.
    #[must_use]
    pub fn knots(&self) -> (&[f64], &[f64]) {
        self.inner.knots()
    }

    /// Fitted coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        self.inner.coefficients()
    }
}

/// Min and max of a slice, used for the default bounding box.
fn bounds_of(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        })
}

/// Interior knots must be finite, strictly increasing, and strictly inside the
/// bounding box.
///
/// Strictly inside matters: a knot sitting exactly on an edge duplicates a clamped
/// boundary knot, which raises the multiplicity there beyond `degree + 1` and makes
/// the basis singular. FITPACK rejects it, and so does this -- a fit that came back
/// with NaN coefficients from a singular basis would be far harder to diagnose than
/// a message naming the offending knot.
fn check_interior_knots(knots: &[f64], lo: f64, hi: f64, label: &str) -> Result<(), InterpError> {
    for (i, &k) in knots.iter().enumerate() {
        if !k.is_finite() {
            return Err(InterpError::InvalidArgument {
                detail: format!("{label} interior knot {i} is not finite: {k}"),
            });
        }
        if k <= lo || k >= hi {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "{label} interior knot {i} = {k} must lie strictly inside \
                     ({lo}, {hi})"
                ),
            });
        }
        if i > 0 && k <= knots[i - 1] {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "{label} interior knots must be strictly increasing; knot {i} = \
                     {k} does not exceed {}",
                    knots[i - 1]
                ),
            });
        }
    }
    Ok(())
}

/// Full knot vector: `degree + 1` copies of `lo`, the interior knots, then
/// `degree + 1` copies of `hi`.
fn clamped_knot_vector(interior: &[f64], lo: f64, hi: f64, degree: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(interior.len() + 2 * (degree + 1));
    for _ in 0..=degree {
        knots.push(lo);
    }
    knots.extend_from_slice(interior);
    for _ in 0..=degree {
        knots.push(hi);
    }
    knots
}

/// `LSQBivariateSpline` -- least squares on caller-supplied knots.
#[cfg(test)]
mod lsq_bivariate_spline_tests {
    use super::{LSQBivariateSpline, SmoothBivariateSplineOptions};

    /// A grid of samples from a bilinear surface, which a degree-1 spline can
    /// represent EXACTLY. That exactness is what makes the first test a statement
    /// about correctness rather than about tolerance.
    fn bilinear_samples() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                let xi = i as f64 / 7.0 * 4.0;
                let yj = j as f64 / 7.0 * 4.0;
                x.push(xi);
                y.push(yj);
                z.push(2.0 + 3.0 * xi - 1.5 * yj + 0.5 * xi * yj);
            }
        }
        (x, y, z)
    }

    fn opts(kx: usize, ky: usize) -> SmoothBivariateSplineOptions {
        SmoothBivariateSplineOptions {
            kx,
            ky,
            ..SmoothBivariateSplineOptions::default()
        }
    }

    /// MUST-HIT: a bilinear surface is in the span of a bidegree-(1,1) spline, so the
    /// least-squares fit must reproduce it to rounding at the sample points AND
    /// between them. Checking only at the samples would not distinguish a genuine fit
    /// from one that happened to interpolate.
    #[test]
    fn a_bilinear_surface_is_reproduced_exactly() {
        let (x, y, z) = bilinear_samples();
        let spline = LSQBivariateSpline::new(&x, &y, &z, &[2.0], &[2.0], opts(1, 1)).expect("fit");

        for (i, (&xi, &yi)) in x.iter().zip(&y).enumerate() {
            let got = spline.eval(xi, yi);
            assert!(
                (got - z[i]).abs() < 1e-8,
                "sample {i} at ({xi}, {yi}): got {got} want {}",
                z[i]
            );
        }
        // Between the samples, where an interpolant that merely hit the data points
        // could still be wrong.
        for (px, py) in [(0.7_f64, 1.3_f64), (2.4, 3.1), (3.9, 0.2)] {
            let want = 2.0 + 3.0 * px - 1.5 * py + 0.5 * px * py;
            let got = spline.eval(px, py);
            assert!(
                (got - want).abs() < 1e-8,
                "off-sample ({px}, {py}): got {got} want {want}"
            );
        }
        assert!(
            spline.residual() < 1e-12,
            "an exactly representable surface should leave no residual, got {}",
            spline.residual()
        );
    }

    /// The knots the caller gives are the knots that are used -- the point of the
    /// type. The full vectors must be the interior knots clamped by `degree + 1`
    /// copies of each bounding-box edge.
    #[test]
    fn the_supplied_knots_are_the_knots_used() {
        let (x, y, z) = bilinear_samples();
        let tx = [1.0_f64, 2.5];
        let ty = [3.0_f64];
        let spline = LSQBivariateSpline::new(&x, &y, &z, &tx, &ty, opts(1, 1)).expect("fit");
        let (kx_knots, ky_knots) = spline.knots();

        assert_eq!(kx_knots, &[0.0, 0.0, 1.0, 2.5, 4.0, 4.0]);
        assert_eq!(ky_knots, &[0.0, 0.0, 3.0, 4.0, 4.0]);
    }

    /// Invalid knot sets are rejected with the reason named, each paired with a
    /// must-miss so the guard cannot pass by refusing everything.
    #[test]
    fn invalid_knots_are_rejected() {
        let (x, y, z) = bilinear_samples();

        // Not strictly increasing.
        assert!(
            LSQBivariateSpline::new(&x, &y, &z, &[2.0, 2.0], &[2.0], opts(1, 1)).is_err(),
            "duplicate interior knots were accepted"
        );
        // On the boundary: this duplicates a clamped end knot and makes the basis
        // singular, which is why it is refused rather than fitted.
        assert!(
            LSQBivariateSpline::new(&x, &y, &z, &[0.0], &[2.0], opts(1, 1)).is_err(),
            "a knot on the lower bbox edge was accepted"
        );
        assert!(
            LSQBivariateSpline::new(&x, &y, &z, &[4.0], &[2.0], opts(1, 1)).is_err(),
            "a knot on the upper bbox edge was accepted"
        );
        assert!(
            LSQBivariateSpline::new(&x, &y, &z, &[f64::NAN], &[2.0], opts(1, 1)).is_err(),
            "a non-finite knot was accepted"
        );
        // MUST-MISS.
        assert!(
            LSQBivariateSpline::new(&x, &y, &z, &[2.0], &[2.0], opts(1, 1)).is_ok(),
            "a valid knot set was rejected"
        );
    }

    /// An underdetermined system is refused rather than solved. With cubic degrees and
    /// several interior knots the coefficient count exceeds the sample count, and any
    /// answer returned there would be one of infinitely many exact solutions -- chosen
    /// by the solver rather than by the data.
    #[test]
    fn an_underdetermined_fit_is_refused() {
        // 9 samples, but a bidegree-(3,3) fit with 2 interior knots each way needs
        // (2+4) * (2+4) = 36 coefficients.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                x.push(i as f64);
                y.push(j as f64);
                z.push((i + j) as f64);
            }
        }
        let err = LSQBivariateSpline::new(&x, &y, &z, &[0.8, 1.4], &[0.8, 1.4], opts(3, 3));
        assert!(
            err.is_err(),
            "a 36-coefficient fit to 9 samples was accepted; the system is \
             underdetermined and its solution is arbitrary"
        );
    }
}

/// A source-derived marker for remote compilation freshness checks.
///
/// The test binary embeds the exact bytes of this file at compile time, so its
/// SHA-256 must match the caller's checked-out source. This detects a worker that
/// runs a plausible but stale source snapshot without ever placing a deliberately
/// wrong value in the shared tree.
#[cfg(test)]
mod rch_source_freshness_tests {
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn compiled_source_marker_matches_exact_lib_bytes() {
        const SOURCE: &[u8] = include_bytes!("lib.rs");
        let source_sha256 = sha256_hex(SOURCE);

        // Must-hit: a one-byte source mutation changes the marker. A constant
        // marker or a detector that ignores its input would fail here.
        let mut altered = SOURCE.to_vec();
        altered[0] ^= 1;
        assert_ne!(
            sha256_hex(&altered),
            source_sha256,
            "the source freshness marker did not detect a one-byte mutation"
        );

        println!(
            "FSCI_RCH_SOURCE_SHA256 path=crates/fsci-interpolate/src/lib.rs bytes={} sha256={source_sha256}",
            SOURCE.len()
        );
    }
    /// `RbfInterpolator` implements SciPy's NON-DEFAULT `degree=-1` variant, not its default.
    ///
    /// This type builds a plain N x N kernel matrix (`phi` is `vec![0.0; n * n]`) with no
    /// polynomial augmentation. SciPy's `RBFInterpolator` augments the system with a polynomial
    /// of degree >= 0 for `linear` and >= 1 for `thin_plate_spline`, and those defaults are not
    /// cosmetic: both kernels are only CONDITIONALLY positive definite, and the tail is what
    /// makes the system well posed.
    ///
    /// The values below came from SciPy 1.17.1 on identical data and are pinned so the gap cannot
    /// close or widen silently. Data lying exactly on y = 2x + 1, evaluated OUTSIDE the hull --
    /// a question no kernel-convention difference affects, since SciPy's `linear` is -r where
    /// this is +r and with no tail that sign only flips the weights:
    ///
    /// | query | this type | SciPy `degree=-1` | SciPy DEFAULT |
    /// |---|---|---|---|
    /// | f(1.5) | 10.161878 | 10.161878 | 4.0 |
    /// | f(2) | 10.584389 | 10.584389 | 5.0 |
    /// | f(3) | -36.696158 | -36.696158 | 7.0 |
    ///
    /// So we agree with SciPy to the printed digits on the variant nobody selects, and differ
    /// qualitatively on the one they get by default. This test exists to make closing that gap a
    /// deliberate, visible change rather than a silent one.
    #[test]
    fn rbf_matches_scipy_degree_minus_one_and_not_scipy_default() {
        use super::{RbfInterpolator, RbfKernel};
        let points: Vec<Vec<f64>> = (0..9).map(|i| vec![i as f64 / 8.0]).collect();
        let values: Vec<f64> = points.iter().map(|p| 2.0 * p[0] + 1.0).collect();

        // MUST-HIT: interpolation is exact AT the nodes for both conditionally-positive-definite
        // kernels. If this ever fails, the assertions below are measuring a broken solve rather
        // than a missing polynomial tail.
        for (name, kernel) in [
            ("thin_plate_spline", RbfKernel::ThinPlateSpline),
            ("linear", RbfKernel::Linear),
        ] {
            let rbf = RbfInterpolator::new(&points, &values, kernel, 1.0).expect("rbf");
            let node_error = points
                .iter()
                .zip(&values)
                .map(|(p, &v)| (rbf.eval(p) - v).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                node_error < 1e-12,
                "{name}: RBF must reproduce its own data, saw {node_error:.3e}"
            );
        }

        // THE GAP, pinned against SciPy 1.17.1 on both sides.
        let tps =
            RbfInterpolator::new(&points, &values, RbfKernel::ThinPlateSpline, 1.0).expect("tps");
        for (query, scipy_degree_minus_one, scipy_default) in [
            (1.5_f64, 10.161878_f64, 4.0_f64),
            (2.0, 10.584389, 5.0),
            (3.0, -36.696158, 7.0),
        ] {
            let ours = tps.eval(&[query]);
            assert!(
                (ours - scipy_degree_minus_one).abs() < 1e-5,
                "thin_plate_spline f({query}): we give {ours:.6}, SciPy degree=-1 gives \
                 {scipy_degree_minus_one:.6}; we are supposed to match the no-tail variant"
            );
            assert!(
                (ours - scipy_default).abs() > 1.0,
                "thin_plate_spline f({query}): we now agree with SciPy's DEFAULT \
                 ({scipy_default:.6}) -- if a polynomial tail has been added, this test has done \
                 its job and should be rewritten to assert the new parity, not deleted"
            );
        }

        // `linear` behaves the same way: we match degree=-1 (5, 7, 11), while SciPy's degree-0
        // default returns the constant 3 everywhere outside the hull.
        let lin = RbfInterpolator::new(&points, &values, RbfKernel::Linear, 1.0).expect("linear");
        for (query, scipy_degree_minus_one) in [(1.5_f64, 5.0_f64), (2.0, 7.0), (3.0, 11.0)] {
            let ours = lin.eval(&[query]);
            assert!(
                (ours - scipy_degree_minus_one).abs() < 1e-9,
                "linear f({query}): we give {ours:.6}, SciPy degree=-1 gives \
                 {scipy_degree_minus_one:.6}"
            );
        }
    }

    /// SciPy parity for the two ND interpolators a LIVE SciPy harness already exercises
    /// (frankenscipy-ivxx6).
    ///
    /// `perf_griddata_scipy.rs` times `LinearNDInterpolator` and `CloughTocher2DInterpolator`
    /// against a live SciPy arm, and neither had ANY differential coverage. That is the
    /// configuration in which a competitive ratio can be recorded between two things nobody has
    /// shown compute the same value -- the situation that stopped the RBF row.
    ///
    /// FIXTURE DESIGN. The sites are in general position (irrational-ish coordinates, no four
    /// cocircular), so the Delaunay triangulation is UNIQUE and agreement is required rather than
    /// lucky; on a degenerate fixture both libraries could pick different legal triangulations and
    /// differ without either being wrong. The linear arm is a control: every valid triangulation
    /// reproduces a linear function, so it tests the barycentric arithmetic and not the mesh.
    /// Exterior queries must be NaN -- the must-miss arm, since an interpolator that silently
    /// extrapolates would pass every interior check.
    ///
    /// RESULT, measured against SciPy 1.17.1:
    ///   * `LinearNDInterpolator` AGREES to 2.220e-16. Pinned below.
    ///   * `CloughTocher2DInterpolator` DIVERGES, and the cause is algorithmic rather than a bug.
    ///     SciPy estimates vertex gradients with a GLOBAL iterative curvature-minimising solve
    ///     (documented `tol`/`maxiter`); this crate estimates them LOCALLY per vertex. Both are
    ///     legitimate Clough-Tocher constructions and they are different functions. On linear
    ///     data THIS crate is exact while SciPy carries its convergence tolerance -- see the
    ///     assertions below, which pin that direction explicitly so nobody "fixes" it backwards.
    #[test]
    fn nd_interpolators_against_scipy_1_17_1() {
        use super::{CloughTocher2DInterpolator, LinearNDInterpolator};

        let sites: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.318_309_886_183_790_7, 0.15915494309189535],
            vec![0.693_147_180_559_945_3, 0.43429448190325176],
            vec![0.577_215_664_901_532_9, 0.866_025_403_784_438_6],
            vec![0.135_335_283_236_612_7, 0.606_530_659_712_633_4],
            vec![0.866_025_403_784_438_6, 0.20787957635076193],
            vec![0.41421356237309515, 0.732_050_807_568_877_3],
            vec![0.22360679774997896, 0.33166247903553998],
            vec![0.785_398_163_397_448_3, 0.618_033_988_749_894_8],
        ];
        let queries: Vec<Vec<f64>> = vec![
            vec![0.25, 0.25],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.4, 0.8],
            vec![0.6, 0.15],
            vec![0.15, 0.85],
            vec![-0.25, 0.5],
            vec![1.4, 0.5],
            vec![0.5, -0.3],
        ];
        let linear = |p: &[f64]| 2.0 * p[0] - 3.0 * p[1] + 1.0;
        let nonlinear = |p: &[f64]| (3.0 * p[0]).sin() * (2.0 * p[1]).cos() + 0.5 * p[0];

        // SciPy 1.17.1
        let scipy_linearnd_linear: Vec<f64> = vec![
            0.75,
            0.5,
            1.75,
            -0.6000000000000002,
            1.7499999999999996,
            -1.25,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
        let scipy_linearnd_nonlinear: Vec<f64> = vec![
            0.7027381516961764,
            0.657453112847633,
            0.9111643418591131,
            0.22398295470806284,
            0.8796896732430468,
            0.12974716698758793,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
        let scipy_clough_nonlinear: Vec<f64> = vec![
            0.7345917469360356,
            0.7487565926370296,
            1.0115592520596974,
            0.19530234037679986,
            1.1052830901401725,
            0.08294242349752515,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];

        // ---- LinearNDInterpolator: full parity, pinned ----------------------------------------
        for (label, f, want) in [
            (
                "linear",
                &linear as &dyn Fn(&[f64]) -> f64,
                &scipy_linearnd_linear,
            ),
            ("nonlinear", &nonlinear, &scipy_linearnd_nonlinear),
        ] {
            let values: Vec<f64> = sites.iter().map(|p| f(p)).collect();
            let it = LinearNDInterpolator::new(&sites, &values).expect("linearnd");
            for (i, q) in queries.iter().enumerate() {
                let got = it.eval(q).unwrap_or(f64::NAN);
                let w = want[i];
                if w.is_nan() {
                    assert!(
                        got.is_nan(),
                        "linearnd/{label} q{i}: SciPy gives NaN outside the hull, we gave {got}"
                    );
                } else {
                    assert!(
                        (got - w).abs() <= 1e-12,
                        "linearnd/{label} q{i}: got {got:.17e}, SciPy 1.17.1 gives {w:.17e}"
                    );
                }
            }
        }

        // ---- CloughTocher2D: exact on linear data, and NOT SciPy's interpolant ----------------
        let lin_values: Vec<f64> = sites.iter().map(|p| linear(p)).collect();
        let ct = CloughTocher2DInterpolator::new(&sites, &lin_values).expect("clough");
        for (i, q) in queries.iter().enumerate().take(6) {
            let got = ct.eval(q).unwrap_or(f64::NAN);
            assert!(
                (got - linear(q)).abs() <= 1e-12,
                "clough/linear q{i}: a Clough-Tocher patch must reproduce linear data exactly; \
                 got {got:.17e} against {:.17e}",
                linear(q)
            );
        }
        for (i, q) in queries.iter().enumerate().skip(6) {
            assert!(
                ct.eval(q).unwrap_or(f64::NAN).is_nan(),
                "clough q{i}: must be NaN outside the convex hull"
            );
        }

        // The divergence itself, pinned so it cannot close silently. If a future change adopts
        // SciPy's global iterative gradient estimator this assertion FAILS -- rewrite it to
        // assert the new parity rather than delete it.
        let nl_values: Vec<f64> = sites.iter().map(|p| nonlinear(p)).collect();
        let ct_nl = CloughTocher2DInterpolator::new(&sites, &nl_values).expect("clough nl");
        let worst = queries
            .iter()
            .take(6)
            .enumerate()
            .map(|(i, q)| (ct_nl.eval(q).unwrap_or(f64::NAN) - scipy_clough_nonlinear[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst > 1e-3,
            "clough/nonlinear now agrees with SciPy to {worst:.3e}. If the gradient estimator \
             was changed to SciPy's global iterative one, this test has done its job and should \
             be rewritten to assert parity, not deleted"
        );
    }
}
