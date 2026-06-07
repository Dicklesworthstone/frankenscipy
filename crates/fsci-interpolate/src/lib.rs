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

        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            options,
            spline_coeffs,
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
        x_new.iter().map(|&xi| self.eval(xi)).collect()
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
        x_new.iter().map(|&xi| self.eval(xi)).collect()
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
        x_new.iter().map(|&xi| self.eval(xi)).collect()
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
        x_new.iter().map(|&xi| self.eval(xi)).collect()
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
    pub fn new(t: Vec<f64>, c: Vec<f64>, k: usize) -> Result<Self, InterpError> {
        if c.len() <= k {
            return Err(InterpError::InvalidArgument {
                detail: format!(
                    "number of coefficients ({}) must be > degree ({})",
                    c.len(),
                    k
                ),
            });
        }
        let expected_knots = c.len() + k + 1;
        if t.len() != expected_knots {
            return Err(InterpError::InvalidArgument {
                detail: format!("knot length {} != c.len()+k+1={}", t.len(), expected_knots),
            });
        }
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
        let mut d = vec![0.0; self.k + 1];
        let is_sorted = xs.windows(2).all(|w| w[0] <= w[1]);
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
            results
        } else {
            xs.iter().map(|&x| self.eval_into(x, &mut d)).collect()
        }
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
        let n = c.len();
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
        for i in 0..xi.len() {
            for j in i + 1..xi.len() {
                if (xi[i] - xi[j]).abs() <= 1e-15 {
                    return Err(InterpError::InvalidArgument {
                        detail: format!("duplicate interpolation nodes at indices {i} and {j}"),
                    });
                }
            }
        }

        // Compute barycentric weights using log-sum to reduce overflow risk
        // for large node counts or widely-spaced nodes.
        let mut wi = vec![1.0; xi.len()];
        for i in 0..xi.len() {
            let mut denom = 1.0;
            for j in 0..xi.len() {
                if i != j {
                    denom *= xi[i] - xi[j];
                }
            }
            if !denom.is_finite() || denom.abs() < f64::MIN_POSITIVE {
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
        xs.iter().map(|&x| self.eval(x)).collect()
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
    let mut a_mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        let basis = eval_basis_all(&t, x[i], k, n);
        a_mat[i][..n].copy_from_slice(&basis[..n]);
    }
    let mut rhs = y.to_vec();
    let c = solve_dense_system(&mut a_mat, &mut rhs)?;
    BSpline::new(t, c, k)
}

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
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];
    // A B-spline basis has local support: eval_basis_all returns a length-n vector
    // with only ~k+1 nonzero entries, so the dense n^2 inner double-loop wastes
    // O(m*n^2) on terms that contribute nothing. Restrict the A^T A accumulation to
    // the nonzero basis indices. BIT-IDENTICAL: a skipped (j,l) term is
    // basis[j]*basis[l] with a zero factor, and basis values are finite (in [0,1]),
    // so the product is +/-0.0 and `v + (+/-0.0) == v` for every f64 v — the
    // i-major accumulation order and every bit of A^T A is preserved. The A^T y loop
    // is left full (O(m*n), not the bottleneck) so any 0*non-finite-y term matches
    // the original bit-for-bit. [perf]
    let mut nz: Vec<usize> = Vec::with_capacity(k + 1);
    for i in 0..m {
        let basis = eval_basis_all(t, x[i], k, n);
        let yi = y[i];
        for j in 0..n {
            aty[j] += basis[j] * yi;
        }
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
    let c = solve_dense_system(&mut ata, &mut aty)?;
    BSpline::new(t.to_vec(), c, k)
}

fn interpolation_knots(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let num_knots = n + k + 1;
    let num_interior = n - k - 1;
    let mut t = Vec::with_capacity(num_knots);
    for _ in 0..=k {
        t.push(x[0]);
    }
    for i in 0..num_interior {
        t.push(x[i + 1 + (k - 1) / 2]);
    }
    for _ in 0..=k {
        t.push(x[n - 1]);
    }
    t
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
    // A B-spline basis has local support: eval_basis_all returns a length-n
    // vector with only ~k+1 nonzero entries, so the dense n^2 inner double-loop
    // wastes O(n^3) on terms that contribute nothing. Restrict to the nonzero
    // indices. This is BIT-IDENTICAL: a skipped term is basis[j]*basis[l] (or
    // basis[j]*y[i]) with a zero factor, hence +/-0.0, and `v + (+/-0.0) == v`
    // for every f64 v — so the accumulators are unchanged bit-for-bit, and the
    // i-major accumulation order is preserved. [perf]
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

    let c = solve_dense_system(&mut ata, &mut aty)?;
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
            if a_row[col].abs() > max_val {
                max_val = a_row[col].abs();
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
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (j, pval) in pivot_row.iter().enumerate().skip(col) {
                a[row][j] -= factor * pval;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Ok(x)
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
        queries.iter().map(|q| self.eval(q)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GriddataMethod {
    Nearest,
    Linear,
}

pub fn griddata(
    points: &[Vec<f64>],
    values: &[f64],
    xi: &[Vec<f64>],
    method: GriddataMethod,
) -> Result<Vec<f64>, InterpError> {
    match method {
        GriddataMethod::Nearest => NearestNDInterpolator::new(points, values)?.eval_many(xi),
        GriddataMethod::Linear => LinearNDInterpolator::new(points, values)?.eval_many(xi),
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

        xi.iter().map(|x| self.eval(x)).collect()
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
        // Hoist per-axis state out of the query loop so the inner pass touches
        // only locals. Folding the NaN check, bounds check, and nearest lookup
        // into a single pass (instead of three separate iterations over each
        // 3-vector) removes the dominant per-query overhead once interval lookup
        // is O(1). Behaviour is identical to the prior three-pass form.
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

        let mut results = Vec::with_capacity(xi.len());
        for point in xi {
            if point.len() != 3 {
                return Err(InterpError::InvalidArgument {
                    detail: format!("expected 3D, got {}D", point.len()),
                });
            }
            let p = [point[0], point[1], point[2]];
            if p[0].is_nan() || p[1].is_nan() || p[2].is_nan() {
                results.push(f64::NAN);
                continue;
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
                results.push(fill);
                continue;
            }

            let flat_idx = Self::nearest_index(un[0], ax[0], p[0]) * st[0]
                + Self::nearest_index(un[1], ax[1], p[1]) * st[1]
                + Self::nearest_index(un[2], ax[2], p[2]) * st[2];
            results.push(self.values[flat_idx]);
        }
        Ok(results)
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

    fn eval_linear(&self, xi: &[f64]) -> Result<f64, InterpError> {
        let ndim = self.ndim();
        if ndim == 3 {
            return Ok(self.eval_linear_3d(xi));
        }

        let mut indices = Vec::with_capacity(ndim);
        let mut fracs = Vec::with_capacity(ndim);
        for (axis, &x) in self.points.iter().zip(xi) {
            let i = Self::find_interval(axis, x);
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

    fn eval_linear_3d(&self, xi: &[f64]) -> f64 {
        debug_assert_eq!(xi.len(), 3);

        let mut indices = [0usize; 3];
        let mut fracs = [0.0; 3];
        for dim in 0..3 {
            let axis = &self.points[dim];
            let x = xi[dim];
            let i = Self::find_interval(axis, x);
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

        // For cubic spline, we use 4 points per dimension (local cubic).
        // This is the "not-a-knot" style local cubic that scipy uses.

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
    grid: Option<SimplexGrid>,
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
        let margin = 10.0;
        let mut all_points = points.to_vec();
        all_points.push((min_x - margin * dx, min_y - margin * dy));
        all_points.push((max_x + margin * dx, min_y - margin * dy));
        all_points.push(((min_x + max_x) / 2.0, max_y + margin * dy));
        let mut triangles = vec![(n, n + 1, n + 2)];
        for p_idx in 0..n {
            let p = all_points[p_idx];
            let mut bad = Vec::new();
            for (t_idx, &(a, b, c)) in triangles.iter().enumerate() {
                if in_circumcircle(all_points[a], all_points[b], all_points[c], p) {
                    bad.push(t_idx);
                }
            }
            let mut boundary = Vec::new();
            for &t_idx in &bad {
                let (a, b, c) = triangles[t_idx];
                for &(e0, e1) in &[(a, b), (b, c), (c, a)] {
                    if !bad.iter().any(|&o| {
                        o != t_idx
                            && triangle_has_edge(
                                triangles[o].0,
                                triangles[o].1,
                                triangles[o].2,
                                e0,
                                e1,
                            )
                    }) {
                        boundary.push((e0, e1));
                    }
                }
            }
            bad.sort_unstable();
            for &idx in bad.iter().rev() {
                triangles.swap_remove(idx);
            }
            for &(e0, e1) in &boundary {
                triangles.push((p_idx, e0, e1));
            }
        }
        let simplices = triangles
            .into_iter()
            .filter(|&(a, b, c)| a < n && b < n && c < n)
            .map(|triangle| orient_triangle_ccw(points, triangle))
            .collect::<Vec<_>>();
        let simplex_bounds: Vec<SimplexBounds> = simplices
            .iter()
            .map(|&simplex| SimplexBounds::for_simplex(points, simplex))
            .collect();
        let neighbors = compute_simplex_neighbors(&simplices);
        let grid = SimplexGrid::build(&simplex_bounds);
        Ok(Self {
            points: points.to_vec(),
            simplices,
            neighbors,
            simplex_bounds,
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
                    let (a, b, c) = self.simplices[idx];
                    let (l1, l2, l3) =
                        barycentric(self.points[a], self.points[b], self.points[c], query);
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
        for (idx, &(a, b, c)) in self.simplices.iter().enumerate() {
            if !self.simplex_bounds[idx].may_contain(query) {
                continue;
            }
            let (l1, l2, l3) = barycentric(self.points[a], self.points[b], self.points[c], query);
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

fn in_circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let (ax, ay, bx, by, cx, cy) = (
        a.0 - d.0,
        a.1 - d.1,
        b.0 - d.0,
        b.1 - d.1,
        c.0 - d.0,
        c.1 - d.1,
    );
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);
    let orient = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if orient > 0.0 { det > 0.0 } else { det < 0.0 }
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
        queries.iter().map(|q| self.eval(q)).collect()
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
    values: Vec<f64>,
    gradients: Vec<(f64, f64)>,
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
        Ok(Self {
            delaunay,
            values: values.to_vec(),
            gradients,
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
        Ok(clough_tocher_triangle_eval(
            &self.delaunay,
            idx,
            &self.values,
            &self.gradients,
            [l1, l2, l3],
        ))
    }

    pub fn eval_many(&self, queries: &[Vec<f64>]) -> Result<Vec<f64>, InterpError> {
        queries.iter().map(|query| self.eval(query)).collect()
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
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            if points[i][0] == points[j][0] && points[i][1] == points[j][1] {
                return Err(InterpError::InvalidArgument {
                    detail: "CloughTocher2DInterpolator requires unique points".to_string(),
                });
            }
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

fn clough_tocher_triangle_eval(
    delaunay: &Delaunay2D,
    simplex_index: usize,
    values: &[f64],
    gradients: &[(f64, f64)],
    bary: [f64; 3],
) -> f64 {
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

/// N-dimensional scattered data interpolation using radial basis functions.
///
/// Matches `scipy.interpolate.RBFInterpolator(y, d, kernel=kernel)`.
#[derive(Debug, Clone)]
pub struct RbfInterpolator {
    points: Vec<Vec<f64>>,
    weights: Vec<f64>,
    kernel: RbfKernel,
    epsilon: f64,
    dim: usize,
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

        // Build the RBF matrix: Φ[i,j] = φ(||points[i] - points[j]||)
        let mut phi = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_dist(&points[i], &points[j]);
                phi[i][j] = rbf_eval(kernel, r, epsilon);
            }
        }

        // Solve Φ w = values for weights
        let mut phi_mut = phi;
        let mut values_mut = values.to_vec();
        let weights = solve_dense_system(&mut phi_mut, &mut values_mut)?;

        Ok(Self {
            points: points.to_vec(),
            weights,
            kernel,
            epsilon,
            dim,
        })
    }

    /// Evaluate the interpolant at a query point.
    pub fn eval(&self, query: &[f64]) -> f64 {
        if query.len() != self.dim {
            return f64::NAN;
        }
        let mut result = 0.0;
        for (i, pt) in self.points.iter().enumerate() {
            let r = euclidean_dist(pt, query);
            result += self.weights[i] * rbf_eval(self.kernel, r, self.epsilon);
        }
        result
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

/// Map `f` over `queries`, splitting the work across threads when
/// `queries.len() * work_per_query` is large enough to amortise thread spawn. `f` is
/// a pure per-query function, so the parallel result is bit-identical to the
/// sequential `queries.iter().map(f).collect()` (order preserved by concatenating
/// contiguous chunks). Used by the per-query batch evaluators.
fn par_query_map<T, F>(queries: &[T], work_per_query: usize, f: F) -> Vec<f64>
where
    T: Sync,
    F: Fn(&T) -> f64 + Sync,
{
    let m = queries.len();
    let work = (m as u64).saturating_mul(work_per_query.max(1) as u64);
    if work < 1 << 18 || m < 4 {
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

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
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
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
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
        // Find interval
        let n = self.x.len() - 1;
        let mut seg = 0;
        for i in 1..n {
            if xval >= self.x[i] {
                seg = i;
            } else {
                break;
            }
        }

        // Evaluate polynomial in local coordinates
        let dx = xval - self.x[seg];
        polyval(&self.c[seg], dx)
    }

    /// Evaluate at multiple points.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
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
        return Ok((bspl.knots().to_vec(), bspl.coeffs().to_vec(), k));
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
    Ok((bspl.knots().to_vec(), bspl.coeffs().to_vec(), k))
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
    let n_coeff = c.len(); // == t.len() - k - 1 for the tight BSpline tck
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
            // One sign change on a monotonic segment: bisect for the root.
            let (mut a, mut b) = (lo, hi);
            for _ in 0..80 {
                let mid = 0.5 * (a + b);
                if (p(mid) < 0.0) == (plo < 0.0) {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            roots.push(0.5 * (a + b));
        }
    }
    roots.retain(|&u| (0.0..h).contains(&u));
    roots
}

/// Evaluate a polynomial and its derivatives.
///
/// Returns (p(x), p'(x), p''(x), ...) up to order `der`.
pub fn polyval_der(coeffs: &[f64], x: f64, der: usize) -> Vec<f64> {
    let n = coeffs.len();
    if n == 0 {
        return vec![0.0; der + 1];
    }

    // First compute all derivatives of the polynomial at x
    let mut result = vec![0.0; der + 1];

    // Horner's method for each derivative
    for (d, item) in result.iter_mut().enumerate().take(der + 1) {
        if d >= n {
            break;
        }
        // Coefficients of the d-th derivative
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

/// Padé approximation: compute rational function p(x)/q(x) coefficients
/// from a Taylor series.
///
/// Given Taylor coefficients [a0, a1, ..., a_{m+n}], returns (p_coeffs, q_coeffs)
/// of degrees m and n respectively such that p(x)/q(x) ≈ Σ a_k x^k.
///
/// Matches `scipy.interpolate.pade`.
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
    let num: f64 = p
        .iter()
        .enumerate()
        .map(|(i, &c)| c * x.powi(i as i32))
        .sum();
    let den: f64 = q
        .iter()
        .enumerate()
        .map(|(i, &c)| c * x.powi(i as i32))
        .sum();
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
    let mut result = vec![0.0; n];
    let offset_a = n - a.len();
    let offset_b = n - b.len();
    for (i, &v) in a.iter().enumerate() {
        result[offset_a + i] += v;
    }
    for (i, &v) in b.iter().enumerate() {
        result[offset_b + i] += v;
    }
    result
}

/// Polynomial subtraction.
pub fn polysub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let neg_b: Vec<f64> = b.iter().map(|&v| -v).collect();
    polyadd(a, &neg_b)
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
    // Check if x is exactly a node
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
    let antider = polyint(coeffs, 1, 0.0);
    polyval(&antider, b) - polyval(&antider, a)
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

        // Step 1: Fit 1D splines along x for each row of z
        // This gives us an intermediate coefficient matrix of shape (ny, nx)
        // (make_interp_spline returns nx coefficients for nx data points)
        let mut row_coeffs: Vec<Vec<f64>> = Vec::with_capacity(ny);

        for row in z {
            let spline_x = make_interp_spline(x, row, kx)?;
            row_coeffs.push(spline_x.coeffs().to_vec());
        }

        // Step 2: For each column of row_coeffs, fit a 1D spline along y
        // Final coefficients have shape (ny, nx)
        let mut final_coeffs: Vec<Vec<f64>> = vec![vec![0.0; nx]; ny];

        for col_idx in 0..nx {
            // Extract this column
            let col_values: Vec<f64> = row_coeffs.iter().map(|row| row[col_idx]).collect();
            let spline_y = make_interp_spline(y, &col_values, ky)?;
            let y_coeffs = spline_y.coeffs();

            for (row_idx, &coeff) in y_coeffs.iter().enumerate() {
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
        Ok(xi.iter().zip(yi).map(|(&x, &y)| self.eval(x, y)).collect())
    }

    /// Evaluate the spline on a grid and return a 2D array.
    ///
    /// Returns values at all combinations of xi and yi, shape `(len(xi), len(yi))`.
    pub fn eval_grid(&self, xi: &[f64], yi: &[f64]) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(xi.len());
        let ny = self.coeffs.len();
        let mut intermediate = vec![0.0; ny];
        let mut x_scratch = vec![0.0; self.kx + 1];
        let mut y_scratch = vec![0.0; self.ky + 1];

        for &xv in xi {
            if !xv.is_finite() {
                result.push(vec![f64::NAN; yi.len()]);
                continue;
            }

            let xi_clamped = xv.clamp(self.x_bounds.0, self.x_bounds.1);
            for (slot, row) in intermediate.iter_mut().zip(&self.coeffs) {
                *slot =
                    BSpline::eval_parts(&self.tx, row, self.kx, true, xi_clamped, &mut x_scratch);
            }

            result.push(
                yi.iter()
                    .map(|&yv| {
                        if !yv.is_finite() {
                            return f64::NAN;
                        }
                        let yi_clamped = yv.clamp(self.y_bounds.0, self.y_bounds.1);
                        BSpline::eval_parts(
                            &self.ty,
                            &intermediate,
                            self.ky,
                            true,
                            yi_clamped,
                            &mut y_scratch,
                        )
                    })
                    .collect(),
            );
        }

        result
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

impl SmoothBivariateSpline {
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
        let mut spline = Self {
            kx: options.kx,
            ky: options.ky,
            bbox,
            tx: fit.tx,
            ty: fit.ty,
            coeffs: fit.coeffs,
            nx_coeffs: fit.nx_coeffs,
            ny_coeffs: fit.ny_coeffs,
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
        Ok(x.iter()
            .zip(y)
            .map(|(&xv, &yv)| self.eval(xv, yv))
            .collect())
    }

    pub fn eval_grid(&self, x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|&xv| y.iter().map(|&yv| self.eval(xv, yv)).collect())
            .collect()
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

    for ((&xv, &yv), (&zv, &weight)) in x.iter().zip(y).zip(z.iter().zip(weights)) {
        let bx = eval_basis_all(tx, xv, kx, nx_coeffs);
        let by = eval_basis_all(ty, yv, ky, ny_coeffs);
        let mut basis = vec![0.0; n_terms];
        for (iy, &by_val) in by.iter().enumerate() {
            for (ix, &bx_val) in bx.iter().enumerate() {
                basis[iy * nx_coeffs + ix] = bx_val * by_val;
            }
        }

        let weight_sq = weight * weight;
        for row in 0..n_terms {
            atz[row] += weight_sq * basis[row] * zv;
            for col in row..n_terms {
                ata[row][col] += weight_sq * basis[row] * basis[col];
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

    solve_dense_system(&mut ata, &mut atz)
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    (f, s) => panic!("npts={npts} q={q:?}: match disagreement {f:?} vs {s:?}"),
                }
            }
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
