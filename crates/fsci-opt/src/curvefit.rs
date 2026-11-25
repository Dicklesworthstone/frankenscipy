#![forbid(unsafe_code)]

//! Nonlinear least squares: `least_squares` (SciPy's default Trust Region Reflective method,
//! rectangular Dogbox, bounds, robust losses, or Levenberg–Marquardt), `curve_fit`, and the
//! batched variants.

use fsci_runtime::RuntimeMode;
use std::borrow::Cow;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use crate::FiniteDiffMethod;
use crate::lsq_matrix::LeastSquaresMatrix;
use crate::trf::{self, Loss, LossFunction, LossKind, XScale};
use crate::trf_lsmr::{TrustRegionLsmrOptions, TrustRegionSolver};
use crate::types::OptError;
use fsci_sparse::{CsrMatrix, LinearOperator, Shape2D};

/// SciPy's `least_squares(method=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeastSquaresMethod {
    /// Trust Region Reflective (Branch, Coleman & Li): bounds, robust losses, `x_scale`.
    Trf,
    /// Powell dogleg in rectangular trust regions: bounds, robust losses, `x_scale`.
    Dogbox,
    /// Levenberg–Marquardt: no bounds, `m ≥ n`, linear loss only.
    Lm,
}

/// SciPy's `least_squares` termination messages by status.
fn termination_message(status: i32) -> &'static str {
    match status {
        -2 => "Stopped by the least-squares iteration callback.",
        0 => "The maximum number of function evaluations is exceeded.",
        1 => "`gtol` termination condition is satisfied.",
        2 => "`ftol` termination condition is satisfied.",
        3 => "`xtol` termination condition is satisfied.",
        _ => "Both `ftol` and `xtol` termination conditions are satisfied.",
    }
}

/// Result from `least_squares`.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresResult {
    /// Solution parameters.
    pub x: Vec<f64>,
    /// Residual vector at the solution.
    pub fun: Vec<f64>,
    /// Objective value: `0.5 * ||fun||²` for linear loss, or the robust loss
    /// applied to the residuals with the requested `f_scale`.
    pub cost: f64,
    /// Whether the solver converged.
    pub success: bool,
    /// Status message.
    pub message: String,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Number of Jacobian evaluations.
    pub njev: usize,
    /// Number of outer iterations. A trust-region iteration may try several
    /// steps before accepting one, or finish without accepting a step.
    pub nit: usize,
    /// Jacobian at the solution, retaining sparse or matrix-free storage when supplied
    /// by the caller or estimated using `jac_sparsity`. For a robust loss this is
    /// SciPy's loss-scaled Jacobian. Use [`LeastSquaresMatrix::to_dense`] when
    /// an explicitly materialized dense matrix is needed.
    pub jac: LeastSquaresMatrix,
    /// SciPy's status: -2 callback stop, 0 max_nfev, 1 gtol, 2 ftol,
    /// 3 xtol, 4 ftol and xtol. A callback stop retains the latest iterate
    /// but does not report convergence (`success` is false).
    pub status: i32,
    /// First-order optimality: ‖Jᵀf‖∞, scaled by Coleman–Li for bounded `Trf`, or
    /// restricted to the free variables for `Dogbox`.
    pub optimality: f64,
    /// Gradient `Jᵀf` of the cost at the solution.
    pub grad: Vec<f64>,
    /// −1 / 1 where a lower / upper bound is active, 0 otherwise.
    pub active_mask: Vec<i8>,
}

/// A read-only view of the current iterate passed to a least-squares callback.
///
/// The slices borrow solver storage only for the duration of the callback.
/// Copy the values you want to retain. `fun` contains the residuals at `x`,
/// before robust-loss scaling; for `curve_fit` they include its measurement
/// weights. `cost` is the requested objective, including any robust loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeastSquaresIteration<'a> {
    pub x: &'a [f64],
    pub fun: &'a [f64],
    pub cost: f64,
    /// Completed outer iterations, starting at one on the first callback.
    pub nit: usize,
    /// Residual evaluations, excluding finite-difference Jacobian calls.
    pub nfev: usize,
    pub njev: usize,
}

/// Iteration monitoring and cooperative cancellation for TRF and Dogbox.
///
/// Return [`ControlFlow::Continue`] to continue or [`ControlFlow::Break`] to
/// stop with status -2. The callback runs after each outer iteration, including
/// the final one, with the latest accepted iterate. It is not called at `x0`
/// or for individual rejected trial steps. No residual or Jacobian evaluation
/// is performed after a callback requests a stop.
///
/// Clones share the same callback. Batched fits may call it concurrently, so
/// use synchronization for shared progress or cancellation state. LM does not
/// support callbacks; passing one to LM returns an explicit option error.
///
/// ```
/// use std::ops::ControlFlow;
/// use fsci_opt::{LeastSquaresCallback, LeastSquaresOptions, least_squares};
///
/// let result = least_squares(
///     |x| vec![x[0] - 10.0],
///     &[0.0],
///     LeastSquaresOptions {
///         callback: Some(LeastSquaresCallback::new(|state| {
///             if state.nit >= 2 {
///                 ControlFlow::Break(())
///             } else {
///                 ControlFlow::Continue(())
///             }
///         })),
///         ..LeastSquaresOptions::default()
///     },
/// ).unwrap();
/// assert_eq!(result.status, -2);
/// assert!(!result.success);
/// ```
#[derive(Clone)]
pub struct LeastSquaresCallback {
    callback: Arc<dyn Fn(&LeastSquaresIteration<'_>) -> ControlFlow<()> + Send + Sync>,
}

impl std::fmt::Debug for LeastSquaresCallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LeastSquaresCallback")
            .finish_non_exhaustive()
    }
}

impl LeastSquaresCallback {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&LeastSquaresIteration<'_>) -> ControlFlow<()> + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
        }
    }

    pub(crate) fn call(&self, iteration: LeastSquaresIteration<'_>) -> ControlFlow<()> {
        (self.callback)(&iteration)
    }
}

/// A caller-defined robust loss for nonlinear least squares.
///
/// The callback receives `z[i] = (residual[i] / f_scale)^2` and returns three
/// rows: `rho(z)`, its first derivative, and its second derivative with respect
/// to `z`. Each row must contain exactly one finite value per residual. The
/// solver applies `f_scale` itself, so callbacks return the unscaled loss.
/// This is SciPy's callable `loss` contract. A vector callback can use different
/// loss functions for observations and regularization residuals.
///
/// Set [`LeastSquaresOptions::custom_loss`] and leave `loss` as `Linear`.
/// TRF and Dogbox support custom losses with dense, sparse, and matrix-free
/// Jacobians; LM rejects them. Clones share the callback, which can be invoked
/// concurrently by batched fits. Return an error from [`Self::new_fallible`]
/// to abort the solve; malformed or nonfinite rows also return a typed error.
/// If a nonzero residual's squared scaled value overflows or rounds to zero,
/// the solve returns an error instead of feeding a different objective to the
/// callback. Choose residual units and `f_scale` that keep this input representable.
///
/// ```
/// use fsci_opt::{LeastSquaresLoss, LeastSquaresOptions, least_squares};
///
/// // A smooth loss distinct from the built-ins: rho(z) = 4*((1+z)^0.25-1).
/// let loss = LeastSquaresLoss::new(|z| [
///     z.iter().map(|&z| 4.0 * (0.25 * z.ln_1p()).exp_m1()).collect(),
///     z.iter().map(|&z| (1.0 + z).powf(-0.75)).collect(),
///     z.iter().map(|&z| -0.75 * (1.0 + z).powf(-1.75)).collect(),
/// ]);
/// let fit = least_squares(
///     |x| vec![x[0] - 1.0, x[0] - 1.1, x[0] - 20.0],
///     &[1.0],
///     LeastSquaresOptions {
///         custom_loss: Some(loss),
///         f_scale: 0.1,
///         ..LeastSquaresOptions::default()
///     },
/// ).unwrap();
/// assert!(fit.success);
/// assert!((fit.x[0] - 1.05).abs() < 0.01);
/// ```
#[derive(Clone)]
pub struct LeastSquaresLoss {
    callback: Arc<dyn Fn(&[f64]) -> Result<[Vec<f64>; 3], OptError> + Send + Sync>,
}

impl std::fmt::Debug for LeastSquaresLoss {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LeastSquaresLoss { callback }")
    }
}

impl LeastSquaresLoss {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&[f64]) -> [Vec<f64>; 3] + Send + Sync + 'static,
    {
        Self::new_fallible(move |z| Ok(callback(z)))
    }

    pub fn new_fallible<F>(callback: F) -> Self
    where
        F: Fn(&[f64]) -> Result<[Vec<f64>; 3], OptError> + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
        }
    }

    pub(crate) fn evaluate(&self, z: &[f64]) -> Result<[Vec<f64>; 3], OptError> {
        let rho = (self.callback)(z)?;
        for (order, row) in rho.iter().enumerate() {
            if row.len() != z.len() {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "custom loss row {order} has {} values; expected {} residuals",
                        row.len(),
                        z.len()
                    ),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(OptError::NonFiniteInput {
                    detail: format!("custom loss row {order} must not contain NaN or Inf"),
                });
            }
        }
        Ok(rho)
    }
}

/// An analytic residual Jacobian shared by individual and batched fits.
///
/// Rows correspond to residuals and columns to parameters. Capturing closures can own
/// model data; the callback is validated at every evaluation before a solver uses it.
/// A callback shared by [`least_squares_many`] must be valid for every parameter row;
/// the callback receives the fitted variables, not the batch's per-fit data.
#[derive(Clone)]
pub struct LeastSquaresJacobian {
    callback: Arc<dyn Fn(&[f64]) -> Result<LeastSquaresMatrix, OptError> + Send + Sync>,
    matrix_free: bool,
}

impl std::fmt::Debug for LeastSquaresJacobian {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LeastSquaresJacobian { callback }")
    }
}

impl LeastSquaresJacobian {
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(&[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(move |x| Ok(LeastSquaresMatrix::Dense(callback(x)))),
            matrix_free: false,
        }
    }

    /// Supply an analytic CSR Jacobian. The automatic trust-region solver
    /// selects LSMR; all subsequent scaling and products retain sparse storage.
    /// Duplicate entries are summed and unsorted rows are canonicalized before
    /// computing column norms. LM requires a dense Jacobian.
    pub fn new_sparse<F>(callback: F) -> Self
    where
        F: Fn(&[f64]) -> CsrMatrix + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(move |x| Ok(LeastSquaresMatrix::Sparse(callback(x)))),
            matrix_free: false,
        }
    }

    /// Supply a matrix-free analytic Jacobian at each parameter vector.
    ///
    /// The returned [`LinearOperator`] must implement both `matvec_into` and
    /// `rmatvec_into` and own the state defining the Jacobian at this point. It must
    /// keep its shape fixed. Each product is checked for errors and nonfinite outputs.
    /// Automatic subproblem selection uses LSMR; TRF and Dogbox preserve products
    /// through bounds, parameter scaling, and robust losses without probing entries.
    ///
    /// `method='lm'`, `tr_solver='exact'`, and `x_scale='jac'` require explicit
    /// entries and reject this representation. `curve_fit` also requires explicit
    /// entries for its parameter-covariance calculation; use `least_squares` for
    /// matrix-free fitting. Batched solves require `Send + Sync` operators.
    pub fn new_operator<F, O>(callback: F) -> Self
    where
        F: Fn(&[f64]) -> O + Send + Sync + 'static,
        O: LinearOperator + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(move |x| Ok(LeastSquaresMatrix::from_operator(callback(x)))),
            matrix_free: true,
        }
    }

    fn evaluate(&self, x: &[f64], residual_count: usize) -> Result<LeastSquaresMatrix, OptError> {
        let matrix = (self.callback)(x)?;
        let valid_shape = match &matrix {
            LeastSquaresMatrix::Dense(rows) => {
                rows.len() == residual_count && rows.iter().all(|row| row.len() == x.len())
            }
            LeastSquaresMatrix::Sparse(csr) => csr.shape() == Shape2D::new(residual_count, x.len()),
            LeastSquaresMatrix::Operator(operator) => {
                operator.shape() == Shape2D::new(residual_count, x.len())
            }
        };
        if !valid_shape {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "Jacobian must have shape ({residual_count}, {}), with one row per residual",
                    x.len()
                ),
            });
        }
        if !matrix.all_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("Jacobian must not contain NaN or Inf"),
            });
        }
        let matrix = match matrix {
            LeastSquaresMatrix::Sparse(csr) => LeastSquaresMatrix::from_sparse(csr)?,
            dense => dense,
        };
        if !matrix.all_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("summing duplicate Jacobian entries produced NaN or Inf"),
            });
        }
        Ok(matrix)
    }
}

/// Structural nonzeros of a residual Jacobian, with a cached deterministic
/// column coloring for grouped finite differences.
///
/// Each row lists the parameters that can affect that residual. Columns whose
/// row supports are disjoint are perturbed together, reducing residual calls
/// from the parameter count to the number of colors (twice that for three-point
/// differences). Storage and construction never allocate a dense conflict graph.
/// The pattern must include every possible nonzero throughout the solve; an
/// omitted dependency is a promise that derivative is identically zero.
#[derive(Debug, Clone)]
pub struct JacobianSparsity {
    shape: Shape2D,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    column_entries: Vec<Vec<(usize, usize)>>,
    groups: Vec<Vec<usize>>,
}

impl JacobianSparsity {
    /// Build a pattern from one list of parameter indices per residual.
    /// Unsorted and repeated indices are accepted and canonicalized.
    pub fn from_rows(mut rows: Vec<Vec<usize>>, columns: usize) -> Result<Self, OptError> {
        let shape = Shape2D::new(rows.len(), columns);
        let mut indices = Vec::new();
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        let mut column_entries = vec![Vec::new(); columns];
        indptr.push(0);
        for (r, row) in rows.iter_mut().enumerate() {
            row.sort_unstable();
            row.dedup();
            for &c in row.iter() {
                if c >= columns {
                    return Err(OptError::InvalidArgument {
                        detail: format!("jac_sparsity column {c} is outside {columns} parameters"),
                    });
                }
                column_entries[c].push((r, indices.len()));
                indices.push(c);
            }
            indptr.push(indices.len());
        }
        // Greedy coloring on the row incidence lists. A generation marker
        // avoids clearing a color set for every column. Empty columns require
        // no residual evaluation at all.
        let mut colors = vec![usize::MAX; columns];
        let mut seen = vec![usize::MAX; columns];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for c in 0..columns {
            if column_entries[c].is_empty() {
                continue;
            }
            for &(r, _) in &column_entries[c] {
                for &neighbor in &rows[r] {
                    if neighbor >= c {
                        break;
                    }
                    seen[colors[neighbor]] = c;
                }
            }
            let color = (0..groups.len())
                .find(|&color| seen[color] != c)
                .unwrap_or(groups.len());
            if color == groups.len() {
                groups.push(Vec::new());
            }
            groups[color].push(c);
            colors[c] = color;
        }
        Ok(Self {
            shape,
            indices,
            indptr,
            column_entries,
            groups,
        })
    }

    /// Use the nonzero entries of a CSR matrix as the structural pattern.
    pub fn from_csr(matrix: &CsrMatrix) -> Result<Self, OptError> {
        let rows = (0..matrix.shape().rows)
            .map(|r| {
                (matrix.indptr()[r]..matrix.indptr()[r + 1])
                    .filter(|&p| matrix.data()[p] != 0.0)
                    .map(|p| matrix.indices()[p])
                    .collect()
            })
            .collect();
        Self::from_rows(rows, matrix.shape().cols)
    }

    pub fn shape(&self) -> Shape2D {
        self.shape
    }

    /// Number of residual calls per two-point Jacobian evaluation.
    pub fn color_count(&self) -> usize {
        self.groups.len()
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Options for `least_squares`: SciPy's keyword arguments.
#[derive(Debug, Clone)]
pub struct LeastSquaresOptions {
    /// `None` is SciPy's default: `Trf` for `least_squares`, `Lm` for an unbounded
    /// `curve_fit`, `Trf` for a bounded one.
    pub method: Option<LeastSquaresMethod>,
    /// Linear solver for TRF and Dogbox. `Auto` uses `Exact` for dense
    /// Jacobians and `Lsmr` for sparse or matrix-free ones. An explicit `Exact` with a
    /// sparse or matrix-free Jacobian is rejected. LM does not accept the LSMR backend.
    pub tr_solver: TrustRegionSolver,
    /// Gradient convergence tolerance. LM uses the maximum absolute cosine
    /// between the residual vector and each Jacobian column; the trust-region
    /// methods use their scaled gradient norms.
    pub gtol: f64,
    /// Convergence tolerance on the change in parameters (xtol).
    pub xtol: f64,
    /// Convergence tolerance on the change in cost (ftol).
    pub ftol: f64,
    /// Maximum number of function evaluations, excluding numerical Jacobian calls.
    pub max_nfev: Option<usize>,
    /// Numerical Jacobian scheme, used when `jac` is absent. `ThreePoint` uses
    /// central differences away from bounds and second-order one-sided differences
    /// near a bound, using the actual representable node spacing. An interval
    /// containing only two representable points uses their secant. Applies to
    /// `Trf`, `Dogbox`, and `Lm`.
    pub diff_method: FiniteDiffMethod,
    /// Relative finite-difference step for all methods. `None` selects SciPy's
    /// `ε^(1/2)` (two-point) or `ε^(1/3)` (three-point), scaled by `max(1, |x|)`.
    /// An explicit step is scaled by `|x|`; a zero coordinate or an unrepresentable
    /// requested displacement uses the automatic step. Derivatives use the actual
    /// rounded displacement, not the requested one.
    pub diff_step: Option<f64>,
    /// Optional analytic Jacobian of the residuals. Numerical differences are used
    /// when absent. Analytic calls contribute to `njev`, not `nfev`.
    pub jac: Option<LeastSquaresJacobian>,
    /// Structural sparsity for numerical Jacobians, ignored with a callable
    /// `jac`. Disjoint columns are differentiated together and stored in CSR
    /// format. Requires TRF or Dogbox and an automatic or LSMR subproblem solver.
    pub jac_sparsity: Option<JacobianSparsity>,
    /// Robust loss (`Trf` and `Dogbox`).
    pub loss: LossKind,
    /// A vectorized custom robust loss (`Trf` and `Dogbox`). Leave `loss` as
    /// `Linear` when supplying this callback; specifying both is an error.
    /// `f_scale` applies to the custom loss as it does to built-in robust losses.
    pub custom_loss: Option<LeastSquaresLoss>,
    /// Soft margin between inlier and outlier residuals for a robust loss.
    pub f_scale: f64,
    /// Parameter scaling: fixed positive scales or `'jac'`. `None` uses unit
    /// scale for `Trf`/`Dogbox` and Jacobian column scaling for `Lm`, as in SciPy.
    pub x_scale: Option<XScale>,
    /// Inspect each TRF/Dogbox iteration or stop early with status -2.
    /// LM rejects this option. See [`LeastSquaresCallback`] for call timing
    /// and the synchronization requirements of batched fits.
    pub callback: Option<LeastSquaresCallback>,
    /// Runtime mode.
    pub mode: RuntimeMode,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            method: None,
            tr_solver: TrustRegionSolver::Auto,
            gtol: 1.0e-8,
            xtol: 1.0e-8,
            ftol: 1.0e-8,
            max_nfev: None,
            diff_method: FiniteDiffMethod::TwoPoint,
            diff_step: None,
            jac: None,
            jac_sparsity: None,
            loss: LossKind::Linear,
            custom_loss: None,
            f_scale: 1.0,
            x_scale: None,
            callback: None,
            mode: RuntimeMode::Strict,
        }
    }
}

const SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

/// Result from `curve_fit`.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveFitResult {
    /// Optimal parameters.
    pub popt: Vec<f64>,
    /// Estimated covariance matrix of parameters (n x n).
    pub pcov: Vec<Vec<f64>>,
    /// Underlying least-squares result. Its residuals and Jacobian include the
    /// measurement-error transformation specified by [`CurveFitOptions::sigma`].
    pub ls_result: LeastSquaresResult,
}

/// Measurement-error specification for [`curve_fit`] and [`curve_fit_bounded`].
///
/// Standard deviations minimize `Σ((f(xᵢ, p) - yᵢ) / sigmaᵢ)²`; a covariance
/// matrix minimizes `rᵀ sigma⁻¹ r`. Correlated errors are handled by a Cholesky
/// solve, without forming the inverse. The same transformation is applied to
/// an analytic model Jacobian supplied through [`LeastSquaresOptions::jac`].
#[derive(Debug, Clone, PartialEq)]
pub enum CurveFitSigma {
    /// A common, finite nonzero standard deviation for every observation.
    Scalar(f64),
    /// One finite nonzero standard deviation per observation. A single entry
    /// broadcasts to all observations, as in SciPy.
    StandardDeviations(Vec<f64>),
    /// A finite, positive-definite observation covariance matrix. Its shape
    /// must be `m × m`, where `m` is the number of observations. As in SciPy's
    /// lower Cholesky factorization, only the lower triangle is used.
    Covariance(Vec<Vec<f64>>),
}

/// How [`curve_fit`] handles missing observations in `xdata` or `ydata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveFitNanPolicy {
    /// Reject NaN observations before evaluating the model.
    Raise,
    /// Drop an observation when either its x or y value is NaN. Infinities
    /// are retained, as in SciPy; use `check_finite` to reject them explicitly.
    Omit,
}

/// Options for `curve_fit`.
#[derive(Debug, Clone, Default)]
pub struct CurveFitOptions {
    /// Initial guess for parameters.
    pub p0: Option<Vec<f64>>,
    /// Least-squares options. Analytic Jacobians and `jac_sparsity` describe
    /// the unweighted model; measurement-error whitening transforms both.
    pub ls_options: LeastSquaresOptions,
    /// Measurement standard deviations or covariance; `None` gives unit weights.
    pub sigma: Option<CurveFitSigma>,
    /// If true, return covariance in the units of `sigma`. Otherwise multiply
    /// it by the weighted residual variance `2 * cost / (m - n)`.
    pub absolute_sigma: bool,
    /// Missing-observation policy. `None` performs no omission. `Omit` applies
    /// the same row selection to x/y, standard deviations, both covariance
    /// axes, analytic Jacobians and `jac_sparsity`. Jacobian callbacks and
    /// sparsity patterns describe the original observation grid; the fitter
    /// selects their retained rows. Covariance uses the retained sample count.
    /// A batch may have a different missing-data mask in each y row.
    /// Custom loss and iteration callbacks receive only retained residuals.
    pub nan_policy: Option<CurveFitNanPolicy>,
    /// Check x/y for NaNs and infinities before fitting. `None` means true
    /// when `nan_policy` is absent, false when it is explicit, matching SciPy.
    /// Explicit true rejects NaNs before an `Omit` policy can remove them.
    /// Model residuals and retained Jacobian entries must still be finite.
    pub check_finite: Option<bool>,
}

struct FitObservations<'a> {
    x: Cow<'a, [f64]>,
    y: Cow<'a, [f64]>,
    retained: Option<Vec<usize>>,
}

impl<'a> FitObservations<'a> {
    fn new(x: &'a [f64], y: &'a [f64], options: &CurveFitOptions) -> Result<Self, OptError> {
        if x.len() != y.len() {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "xdata and ydata must have same length (got {} and {})",
                    x.len(),
                    y.len()
                ),
            });
        }
        if x.is_empty() {
            return Err(OptError::InvalidArgument {
                detail: String::from("xdata must not be empty"),
            });
        }
        if options.check_finite.unwrap_or(options.nan_policy.is_none())
            && x.iter().chain(y).any(|value| !value.is_finite())
        {
            return Err(OptError::NonFiniteInput {
                detail: String::from("xdata and ydata must not contain NaN or Inf"),
            });
        }
        let has_nan = x.iter().chain(y).any(|value| value.is_nan());
        if has_nan && options.nan_policy == Some(CurveFitNanPolicy::Raise) {
            return Err(OptError::NonFiniteInput {
                detail: String::from("nan_policy='raise' rejects NaNs in xdata or ydata"),
            });
        }
        if has_nan && options.nan_policy == Some(CurveFitNanPolicy::Omit) {
            let retained: Vec<usize> = x
                .iter()
                .zip(y)
                .enumerate()
                .filter_map(|(i, (x, y))| (!x.is_nan() && !y.is_nan()).then_some(i))
                .collect();
            if retained.is_empty() {
                return Err(OptError::InvalidArgument {
                    detail: String::from("no observations remain after omitting NaNs"),
                });
            }
            return Ok(Self {
                x: Cow::Owned(retained.iter().map(|&i| x[i]).collect()),
                y: Cow::Owned(retained.iter().map(|&i| y[i]).collect()),
                retained: Some(retained),
            });
        }
        Ok(Self {
            x: Cow::Borrowed(x),
            y: Cow::Borrowed(y),
            retained: None,
        })
    }
}

/// A batch reuses its latest covariance factorization for the same mask. Keeping
/// only one cached factor bounds storage when every observation row has a distinct
/// missing-data pattern; active workers retain their own Arc while solving.
/// Mask before Cholesky: a principal submatrix of sigma describes the retained
/// observations; selecting rows from its full Cholesky factor does not.
struct FitWhiteningCache {
    sigma: Option<CurveFitSigma>,
    observations: usize,
    factor: Mutex<Option<(Vec<usize>, Arc<FitWhitening>)>>,
}

impl FitWhiteningCache {
    fn new(sigma: Option<CurveFitSigma>, observations: usize) -> Self {
        Self {
            sigma,
            observations,
            factor: Mutex::new(None),
        }
    }

    fn get(&self, retained: Option<&[usize]>) -> Result<Arc<FitWhitening>, OptError> {
        let key = retained.unwrap_or(&[]);
        let mut cached = self.factor.lock().map_err(|_| OptError::InvalidArgument {
            detail: String::from("curve-fit covariance preparation was interrupted"),
        })?;
        if let Some((mask, factor)) = cached.as_ref()
            && mask == key
        {
            return Ok(Arc::clone(factor));
        }
        let factor = Arc::new(match retained {
            None => FitWhitening::new(self.sigma.as_ref(), self.observations)?,
            Some(retained) => {
                let sigma = match self.sigma.as_ref() {
                    None => None,
                    Some(CurveFitSigma::Scalar(value)) => Some(CurveFitSigma::Scalar(*value)),
                    Some(CurveFitSigma::StandardDeviations(values)) => {
                        if values.len() != 1 && values.len() != self.observations {
                            return Err(OptError::InvalidArgument {
                                detail: format!(
                                    "sigma must contain one or {} standard deviations before omission",
                                    self.observations
                                ),
                            });
                        }
                        Some(CurveFitSigma::StandardDeviations(if values.len() == 1 {
                            values.clone()
                        } else {
                            retained.iter().map(|&i| values[i]).collect()
                        }))
                    }
                    Some(CurveFitSigma::Covariance(matrix)) => {
                        if matrix.len() != self.observations
                            || matrix.iter().any(|row| row.len() != self.observations)
                        {
                            return Err(OptError::InvalidArgument {
                                detail: format!(
                                    "sigma covariance must have shape ({0}, {0}) before omission",
                                    self.observations
                                ),
                            });
                        }
                        Some(CurveFitSigma::Covariance(
                            retained
                                .iter()
                                .map(|&i| retained.iter().map(|&j| matrix[i][j]).collect())
                                .collect(),
                        ))
                    }
                };
                FitWhitening::new(sigma.as_ref(), retained.len())?
            }
        });
        *cached = Some((key.to_vec(), Arc::clone(&factor)));
        Ok(factor)
    }
}

/// Prepared measurement-error transform, shared by fits using the same mask.
/// Factorizing an `m × m` covariance is O(m³), so it must not be repeated for
/// each residual/Jacobian evaluation or each independent batch member.
enum FitWhitening {
    Identity,
    Diagonal(Vec<f64>),
    Cholesky(Vec<Vec<f64>>),
}

impl FitWhitening {
    fn new(sigma: Option<&CurveFitSigma>, observations: usize) -> Result<Self, OptError> {
        let deviation = |value: f64| -> Result<f64, OptError> {
            if !value.is_finite() || value == 0.0 {
                return Err(OptError::InvalidArgument {
                    detail: String::from("sigma standard deviations must be finite and nonzero"),
                });
            }
            Ok(value)
        };
        match sigma {
            None => Ok(Self::Identity),
            Some(CurveFitSigma::Scalar(value)) => {
                Ok(Self::Diagonal(vec![deviation(*value)?; observations]))
            }
            Some(CurveFitSigma::StandardDeviations(values)) => {
                if values.len() == 1 {
                    return Ok(Self::Diagonal(vec![deviation(values[0])?; observations]));
                }
                if values.len() != observations {
                    return Err(OptError::InvalidArgument {
                        detail: format!(
                            "sigma must contain one or {observations} standard deviations"
                        ),
                    });
                }
                values
                    .iter()
                    .map(|&value| deviation(value))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Self::Diagonal)
            }
            Some(CurveFitSigma::Covariance(matrix)) => {
                if matrix.len() != observations
                    || matrix.iter().any(|row| row.len() != observations)
                {
                    return Err(OptError::InvalidArgument {
                        detail: format!(
                            "sigma covariance must have shape ({observations}, {observations})"
                        ),
                    });
                }
                if matrix.iter().flatten().any(|value| !value.is_finite()) {
                    return Err(OptError::NonFiniteInput {
                        detail: String::from("sigma covariance must not contain NaN or Inf"),
                    });
                }
                let lower = cholesky_decompose(matrix, observations)
                    .filter(|factor| factor.iter().flatten().all(|value| value.is_finite()));
                lower
                    .map(Self::Cholesky)
                    .ok_or_else(|| OptError::InvalidArgument {
                        detail: String::from("sigma covariance must be positive definite"),
                    })
            }
        }
    }

    fn residuals(&self, residuals: &mut [f64]) {
        match self {
            Self::Identity => {}
            Self::Diagonal(deviations) => {
                // Divide directly: for subnormal sigma, 1/sigma may overflow
                // even when the scaled residual itself is finite.
                for (value, &deviation) in residuals.iter_mut().zip(deviations) {
                    *value /= deviation;
                }
            }
            Self::Cholesky(lower) => {
                // Forward substitution L z = r, so ||z||² = rᵀ sigma⁻¹ r.
                for i in 0..residuals.len() {
                    let mut value = residuals[i];
                    for j in 0..i {
                        value -= lower[i][j] * residuals[j];
                    }
                    residuals[i] = value / lower[i][i];
                }
            }
        }
    }

    fn jacobian(&self, jacobian: &mut [Vec<f64>]) {
        match self {
            Self::Identity => {}
            Self::Diagonal(deviations) => {
                for (row, &deviation) in jacobian.iter_mut().zip(deviations) {
                    for value in row {
                        *value /= deviation;
                    }
                }
            }
            Self::Cholesky(lower) => {
                let parameters = jacobian.first().map_or(0, Vec::len);
                for i in 0..jacobian.len() {
                    for column in 0..parameters {
                        let mut value = jacobian[i][column];
                        for j in 0..i {
                            value -= lower[i][j] * jacobian[j][column];
                        }
                        jacobian[i][column] = value / lower[i][i];
                    }
                }
            }
        }
    }

    fn jacobian_matrix(&self, matrix: LeastSquaresMatrix) -> Result<LeastSquaresMatrix, OptError> {
        Ok(match (self, matrix) {
            (Self::Identity, matrix) => matrix,
            (Self::Diagonal(deviations), LeastSquaresMatrix::Sparse(csr)) => {
                // Divide directly, preserving finite ratios for subnormal sigma.
                let mut data = csr.data().to_vec();
                for (r, &deviation) in deviations.iter().enumerate() {
                    for value in &mut data[csr.indptr()[r]..csr.indptr()[r + 1]] {
                        *value /= deviation;
                    }
                }
                LeastSquaresMatrix::Sparse(
                    CsrMatrix::from_components(
                        csr.shape(),
                        data,
                        csr.indices().to_vec(),
                        csr.indptr().to_vec(),
                        false,
                    )
                    .expect("row division preserves validated CSR structure"),
                )
            }
            (_, matrix) => {
                // A full covariance mixes observation rows. Its whitened
                // analytic Jacobian is generally dense, even for sparse input.
                let mut dense = matrix.to_dense()?;
                self.jacobian(&mut dense);
                LeastSquaresMatrix::Dense(dense)
            }
        })
    }

    fn sparsity(&self, pattern: &JacobianSparsity) -> Result<JacobianSparsity, OptError> {
        let Self::Cholesky(lower) = self else {
            return Ok(pattern.clone());
        };
        if pattern.shape.rows != lower.len() {
            return Err(OptError::InvalidArgument {
                detail: String::from("jac_sparsity must have one row per observation"),
            });
        }
        // Forward substitution makes row i depend on every earlier whitened
        // row j with L[i,j] != 0. Preserve this transitive dependency closure;
        // reusing the unweighted pattern would silently drop real derivatives.
        let mut rows: Vec<Vec<usize>> = Vec::with_capacity(lower.len());
        for (i, low_row) in lower.iter().enumerate() {
            let mut dependencies = std::collections::BTreeSet::new();
            dependencies.extend(&pattern.indices[pattern.indptr[i]..pattern.indptr[i + 1]]);
            for (j, previous) in rows.iter().enumerate() {
                if low_row[j] != 0.0 {
                    dependencies.extend(previous);
                }
            }
            rows.push(dependencies.into_iter().copied().collect());
        }
        JacobianSparsity::from_rows(rows, pattern.shape.cols)
    }
}

/// `scipy.optimize.least_squares(fun, x0, method=...)` without bounds: minimize
/// `0.5·Σρ(residuals(x)²)`. The default method is SciPy's `trf`; `Dogbox` selects rectangular
/// trust-region dogleg and `Lm` selects Levenberg–Marquardt. See [`least_squares_bounded`]
/// for bounds.
pub fn least_squares<F>(
    residuals: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    match options.method.unwrap_or(LeastSquaresMethod::Trf) {
        LeastSquaresMethod::Trf | LeastSquaresMethod::Dogbox => {
            least_squares_trust_region(&residuals, x0, None, options)
        }
        LeastSquaresMethod::Lm => least_squares_lm(residuals, x0, options),
    }
}

/// The result of an LM termination: gradient, optimality and (empty) active mask filled in.
#[allow(clippy::too_many_arguments)]
fn lm_result(
    x: Vec<f64>,
    fun: Vec<f64>,
    cost: f64,
    status: i32,
    message: String,
    nfev: usize,
    njev: usize,
    nit: usize,
    jac: Vec<Vec<f64>>,
) -> LeastSquaresResult {
    let n = x.len();
    let mut grad = vec![0.0; n];
    for (row, &fi) in jac.iter().zip(&fun) {
        for (g, &j) in grad.iter_mut().zip(row) {
            *g += j * fi;
        }
    }
    LeastSquaresResult {
        optimality: grad.iter().fold(0.0_f64, |m, g| m.max(g.abs())),
        grad,
        active_mask: vec![0; n],
        // status: SciPy's code for the test that stopped the iteration (0 = max_nfev)
        success: status > 0,
        status,
        x,
        fun,
        cost,
        message,
        nfev,
        njev,
        nit,
        jac: LeastSquaresMatrix::Dense(jac),
    }
}

/// The LM damping metric. Store adaptive column norms directly, avoiding the
/// overflowing reciprocal of a small derivative. Fixed characteristic scales
/// remain in the units supplied by the caller.
enum LmScaling {
    Fixed(Vec<f64>),
    Jacobian(Vec<f64>),
}

impl LmScaling {
    fn new(scale: Option<&XScale>, n: usize) -> Result<Self, OptError> {
        match scale {
            Some(XScale::Fixed(values)) => {
                if values.len() != n {
                    return Err(OptError::InvalidArgument {
                        detail: String::from("Inconsistent shapes between x_scale and `x0`."),
                    });
                }
                if values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(OptError::InvalidArgument {
                        detail: String::from("x_scale entries must be positive and finite"),
                    });
                }
                Ok(Self::Fixed(values.clone()))
            }
            Some(XScale::Jac) | None => Ok(Self::Jacobian(Vec::new())),
        }
    }

    fn update(&mut self, jac: &[Vec<f64>]) -> Result<(), OptError> {
        if let Self::Jacobian(norms) = self {
            let initial = norms.is_empty();
            norms.resize(jac[0].len(), 0.0);
            for (column, norm) in norms.iter_mut().enumerate() {
                let current = jac
                    .iter()
                    .fold(0.0_f64, |value, row| value.hypot(row[column]));
                if !current.is_finite() {
                    return Err(OptError::NonFiniteInput {
                        detail: String::from("Jacobian column norm is not finite"),
                    });
                }
                // MINPACK's adaptive diagonal never decreases: a later small
                // derivative must not suddenly enlarge this parameter's steps.
                *norm = if initial && current == 0.0 {
                    1.0
                } else {
                    (*norm).max(current)
                };
            }
        }
        Ok(())
    }

    fn physical_step(&self, step: &[f64]) -> Vec<f64> {
        match self {
            Self::Fixed(scale) => step.iter().zip(scale).map(|(s, d)| s * d).collect(),
            Self::Jacobian(norms) => step.iter().zip(norms).map(|(s, d)| s / d).collect(),
        }
    }

    fn parameter_norm(&self, x: &[f64]) -> f64 {
        match self {
            Self::Fixed(scale) => x
                .iter()
                .zip(scale)
                .fold(0.0_f64, |norm, (value, d)| norm.hypot(value / d)),
            Self::Jacobian(norms) => x
                .iter()
                .zip(norms)
                .fold(0.0_f64, |norm, (value, d)| norm.hypot(value * d)),
        }
    }

    fn jacobian(&self, jac: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, OptError> {
        // J times the characteristic parameter scale uses the same columnwise
        // transformation as converting a scaled step back to physical units.
        let scaled: Vec<Vec<f64>> = jac.iter().map(|row| self.physical_step(row)).collect();
        if scaled.iter().flatten().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: String::from("scaled Jacobian must not contain NaN or Inf"),
            });
        }
        Ok(scaled)
    }
}

fn lm_max_cosine(jac: &[Vec<f64>], residuals: &[f64]) -> f64 {
    let residual_norm = residuals
        .iter()
        .fold(0.0_f64, |norm, &value| norm.hypot(value));
    if residual_norm == 0.0 {
        return 0.0;
    }
    if !residual_norm.is_finite() {
        return f64::INFINITY;
    }
    let mut maximum = 0.0_f64;
    for column in 0..jac[0].len() {
        let column_norm = jac
            .iter()
            .fold(0.0_f64, |norm, row| norm.hypot(row[column]));
        if !column_norm.is_finite() {
            return f64::INFINITY;
        }
        if column_norm != 0.0 {
            // Normalize before multiplying: Jᵀr and ||J||*||r|| can underflow
            // even when their ratio is an ordinary, nonzero cosine.
            let cosine: f64 = jac
                .iter()
                .zip(residuals)
                .map(|(row, &value)| (row[column] / column_norm) * (value / residual_norm))
                .sum();
            maximum = maximum.max(cosine.abs());
        }
    }
    maximum
}

/// Levenberg–Marquardt (`method='lm'`).
fn least_squares_lm<F>(
    residuals: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x0.len();
    if options.callback.is_some() {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "method 'lm' does not support iteration callbacks; use 'trf' or 'dogbox'",
            ),
        });
    }
    if options.loss != LossKind::Linear || options.custom_loss.is_some() {
        return Err(OptError::InvalidArgument {
            detail: String::from("method 'lm' supports only loss='linear'"),
        });
    }
    if !matches!(
        options.tr_solver,
        TrustRegionSolver::Auto | TrustRegionSolver::Exact
    ) {
        return Err(OptError::InvalidArgument {
            detail: String::from("method 'lm' does not support tr_solver='lsmr'"),
        });
    }
    if options.jac.is_none() && options.jac_sparsity.is_some() {
        return Err(OptError::InvalidArgument {
            detail: String::from("method 'lm' does not support jac_sparsity"),
        });
    }
    let diff_step = options.diff_step.unwrap_or(SQRT_EPS);
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must have at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    if !diff_step.is_finite() || diff_step <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("diff_step must be finite and > 0"),
        });
    }
    for (name, tolerance) in [
        ("gtol", options.gtol),
        ("xtol", options.xtol),
        ("ftol", options.ftol),
    ] {
        if !tolerance.is_finite() || tolerance < f64::EPSILON {
            return Err(OptError::InvalidArgument {
                detail: format!("{name} must be finite and at least machine epsilon for 'lm'"),
            });
        }
    }
    if options.max_nfev == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: String::from("max_nfev must be positive"),
        });
    }
    let mut x_perturbed = Vec::new();
    lm_iterations(
        |x| Ok(residuals(x)),
        |x, r, jac| lm_jacobian_into(&residuals, x, r, &options, jac, &mut x_perturbed),
        x0,
        &options,
    )
}

/// Shared QR trust-region iteration. The caller supplies the Jacobian so the
/// root interface can retain serial callbacks and least_squares can evaluate
/// independent difference columns in parallel.
pub(super) fn lm_iterations(
    residuals: impl Fn(&[f64]) -> Result<Vec<f64>, OptError>,
    mut jacobian: impl FnMut(&[f64], &[f64], &mut Vec<Vec<f64>>) -> Result<(), OptError>,
    x0: &[f64],
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError> {
    let n = x0.len();
    let mut scaling = LmScaling::new(options.x_scale.as_ref(), n)?;
    let max_nfev = options.max_nfev.unwrap_or(100 * n);

    let mut x = x0.to_vec();
    let mut r = residuals(&x)?;
    let m = r.len();
    let mut nfev = 1usize;
    let mut njev = 0usize;

    if m < n {
        return Err(OptError::InvalidArgument {
            detail: format!("number of residuals ({m}) must be >= number of parameters ({n})"),
        });
    }

    // Check for non-finite residuals
    if r.iter().any(|v| !v.is_finite()) {
        return match options.mode {
            RuntimeMode::Strict => Err(OptError::NonFiniteInput {
                detail: String::from("residuals contain non-finite values at x0"),
            }),
            RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite residuals"),
            }),
        };
    }

    let mut cost = 0.5 * dot_vec(&r, &r);
    let mut residual_norm = r.iter().fold(0.0_f64, |norm, &value| norm.hypot(value));
    if !residual_norm.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("residual norm is not finite"),
        });
    }
    let mut jac = Vec::new();
    jacobian(&x, &r, &mut jac)?;
    njev += 1;
    scaling.update(&jac)?;
    let mut scaled_jac = scaling.jacobian(&jac)?;
    // Reduce the Jacobian itself, not JᵀJ: squaring the condition number can
    // erase an identifiable direction and produce a plausible but wrong fit.
    // The reduction is reused until an accepted step changes the linear model.
    let mut qr = LmQr::new(&scaled_jac, &r)?;
    let mut scratch = LmQrScratch::default();
    let mut max_cosine = lm_max_cosine(&jac, &r);
    let mut jstep = Vec::new();
    let mut x_norm = scaling.parameter_norm(&x);
    if !x_norm.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("scaled LM parameter norm is not finite"),
        });
    }
    // Include the residual/model scale so changing residual units does not
    // change the first physical step. A fixed radius at x0 == 0 can be tiny
    // in Jacobian-scaled coordinates and falsely satisfy ftol without moving.
    // Store damping in square-root units so neither the Jacobian norms nor
    // the regularization parameter need to be squared.
    let initial_norm = if qr.max_column_norm > 0.0 {
        x_norm.max(residual_norm / qr.max_column_norm)
    } else {
        x_norm
    };
    let mut radius = if initial_norm == 0.0 {
        100.0
    } else {
        100.0 * initial_norm.min(f64::MAX / 100.0)
    };
    let mut damping = 0.0;
    let mut nit = 0usize;

    for _ in 0..max_nfev {
        nit += 1;

        // LM's gtol is the maximum absolute cosine between a Jacobian column
        // and the residual vector. An absolute Jᵀr test falsely converges at x0
        // when residuals (or curve_fit's 1/sigma weights) have small units.
        if max_cosine <= options.gtol {
            return Ok(lm_result(
                x,
                r,
                cost,
                1,
                String::from("residual is orthogonal to the Jacobian columns (gtol)"),
                nfev,
                njev,
                nit,
                jac,
            ));
        }

        let step_norm;
        (damping, step_norm) = qr.trust_region_step(radius, damping, &mut scratch)?;
        if nit == 1 {
            radius = radius.min(step_norm);
        }

        // Trial step
        let physical_step = scaling.physical_step(&scratch.step);
        let x_new: Vec<f64> = x.iter().zip(&physical_step).map(|(a, b)| a + b).collect();
        let r_new = residuals(&x_new)?;
        nfev += 1;
        if r_new.len() != m {
            return Err(OptError::InvalidArgument {
                detail: format!("residuals changed length from {m} to {}", r_new.len()),
            });
        }
        let cost_new = 0.5 * dot_vec(&r_new, &r_new);

        // Compare relative reductions in units of the current residual norm.
        // Squaring tiny residuals first would underflow both costs to zero and
        // prevent otherwise valid steps (even with a correctly scaled solve).
        let trial_norm = r_new.iter().fold(0.0_f64, |norm, &value| norm.hypot(value));
        let norm_ratio = trial_norm / residual_norm;
        let actual_reduction = (1.0 - norm_ratio) * (1.0 + norm_ratio);
        let predicted_reduction = {
            mat_vec_into(&scaled_jac, &scratch.step, &mut jstep);
            let linear: f64 = r
                .iter()
                .zip(&jstep)
                .map(|(f, j)| (f / residual_norm) * (j / residual_norm))
                .sum();
            let quadratic: f64 = jstep
                .iter()
                .map(|value| (value / residual_norm).powi(2))
                .sum();
            -2.0 * linear - quadratic
        };

        let ratio = if predicted_reduction > 0.0 && actual_reduction.is_finite() {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };
        if ratio <= 0.25 {
            let contraction = if actual_reduction.is_finite() && actual_reduction < 0.0 {
                (0.5 * predicted_reduction / (predicted_reduction - actual_reduction))
                    .clamp(0.1, 0.5)
            } else if !trial_norm.is_finite() {
                0.1
            } else {
                0.5
            };
            radius = contraction * radius.min(10.0 * step_norm);
        } else if damping == 0.0 || ratio >= 0.75 {
            radius = 2.0 * step_norm.min(f64::MAX / 2.0);
        }

        if ratio >= 1.0e-4 {
            x = x_new;
            r = r_new;
            cost = cost_new;
            residual_norm = trial_norm;
            jacobian(&x, &r, &mut jac)?;
            njev += 1;
            scaling.update(&jac)?;
            scaled_jac = scaling.jacobian(&jac)?;
            qr = LmQr::new(&scaled_jac, &r)?;
            max_cosine = lm_max_cosine(&jac, &r);
            x_norm = scaling.parameter_norm(&x);
            if !x_norm.is_finite() {
                return Err(OptError::NonFiniteInput {
                    detail: String::from("scaled LM parameter norm is not finite"),
                });
            }
        }

        // A heavily damped step can be arbitrarily small far from a solution.
        // LM's xtol therefore tests the trust radius, not that step alone.
        // ftol additionally requires agreement of actual and model reductions.
        let ftol_satisfied = actual_reduction.abs() <= options.ftol
            && predicted_reduction >= 0.0
            && predicted_reduction <= options.ftol
            && ratio <= 2.0;
        let xtol_satisfied = radius <= options.xtol * x_norm;
        if ftol_satisfied || xtol_satisfied {
            let status = if ftol_satisfied && xtol_satisfied {
                4
            } else if ftol_satisfied {
                2
            } else {
                3
            };
            return Ok(lm_result(
                x,
                r,
                cost,
                status,
                termination_message(status).to_string(),
                nfev,
                njev,
                nit,
                jac,
            ));
        }

        if nfev >= max_nfev {
            return Ok(lm_result(
                x,
                r,
                cost,
                0,
                format!("max function evaluations exceeded ({max_nfev})"),
                nfev,
                njev,
                nit,
                jac,
            ));
        }
    }

    Ok(lm_result(
        x,
        r,
        cost,
        0,
        String::from("max iterations exceeded"),
        nfev,
        njev,
        nit,
        jac,
    ))
}

/// SciPy's initial absolute step, before adjusting the stencil to bounds.
fn fd_step_scipy(value: f64, method: FiniteDiffMethod, rel_step: Option<f64>) -> f64 {
    let sign = if value >= 0.0 { 1.0 } else { -1.0 };
    let default_step = method.rel_eps() * sign * value.abs().max(1.0);
    match rel_step {
        None => default_step,
        Some(relative) => {
            let step = relative * sign * value.abs();
            if (value + step) - value == 0.0 {
                default_step
            } else {
                step
            }
        }
    }
}

/// Bound-aware, representable points shared by dense and grouped Jacobians.
/// A two-point stencil is also used when a bounded interval contains only two
/// floating-point coordinates: a distinct third sample does not exist there.
#[derive(Clone, Copy)]
struct FiniteDifferenceStencil {
    first: f64,
    second: Option<f64>,
}

fn fd_points_scipy(
    x: &[f64],
    lb: &[f64],
    ub: &[f64],
    method: FiniteDiffMethod,
    rel_step: Option<f64>,
) -> Result<Vec<FiniteDifferenceStencil>, OptError> {
    let mut points = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        // Even an unbounded problem can only sample finite coordinates.
        let lower_bound = lb[i].max(-f64::MAX);
        let upper_bound = ub[i].min(f64::MAX);
        let mut h = fd_step_scipy(x[i], method, rel_step);
        if !h.is_finite() {
            h = fd_step_scipy(x[i], method, None);
        }
        let mut one_sided = method == FiniteDiffMethod::TwoPoint;
        if method == FiniteDiffMethod::ThreePoint {
            h = h.abs();
        }
        let lower = x[i] - lower_bound;
        let upper = upper_bound - x[i];
        if method == FiniteDiffMethod::TwoPoint {
            let trial = x[i] + h;
            let violated = trial < lower_bound || trial > upper_bound;
            let fitting = h.abs() <= lower.max(upper);
            if violated && fitting {
                h = -h;
            } else if !fitting {
                h = if upper >= lower { upper } else { -lower };
            }
        } else if lower < h || upper < h {
            // A three-point one-sided stencil needs two steps into the box.
            // Prefer a central stencil when the shortened step permits it.
            h = if upper >= lower {
                h.min(0.5 * upper)
            } else {
                -h.min(0.5 * lower)
            };
            one_sided = true;
            if h.abs() <= lower.min(upper) {
                h = lower.min(upper);
                one_sided = false;
            }
        }
        let mut first = if one_sided { x[i] + h } else { x[i] - h };
        let mut second = if one_sided { x[i] + 2.0 * h } else { x[i] + h };
        if one_sided && !second.is_finite() {
            // 2*h can overflow even when the final coordinate is finite.
            second = (x[i] + h) + h;
        }
        first = first.clamp(lower_bound, upper_bound);
        second = second.clamp(lower_bound, upper_bound);
        let admissible = |point: f64| {
            point.is_finite() && point >= lower_bound && point <= upper_bound && point != x[i]
        };
        if !admissible(first) {
            if method == FiniteDiffMethod::ThreePoint && admissible(second) {
                first = second;
            } else {
                let candidates = if h >= 0.0 {
                    [x[i].next_up(), x[i].next_down()]
                } else {
                    [x[i].next_down(), x[i].next_up()]
                };
                first = candidates.into_iter().find(|&point| admissible(point)).ok_or_else(|| {
                    OptError::InvalidArgument {
                        detail: String::from(
                            "numerical differentiation requires a distinct finite point within bounds; supply an analytic Jacobian",
                        ),
                    }
                })?;
            }
        }
        let second = if method == FiniteDiffMethod::TwoPoint {
            None
        } else if admissible(second) && second != first {
            Some(second)
        } else {
            // Rounding can collapse half-ULP or uneven one-sided steps.
            // Use a third representable point if the interval contains one.
            let candidates = if first > x[i] {
                [first.next_up(), x[i].next_down()]
            } else {
                [first.next_down(), x[i].next_up()]
            };
            candidates
                .into_iter()
                .find(|&point| admissible(point) && point != first)
        };
        points.push(FiniteDifferenceStencil { first, second });
    }
    Ok(points)
}

/// Preserve the ordinary difference quotient while avoiding overflow in a
/// subtraction when the quotient itself is representable.
fn fd_quotient(fa: f64, fb: f64, xa: f64, xb: f64) -> f64 {
    let df = fa - fb;
    let dx = xa - xb;
    if df.is_finite() && dx.is_finite() {
        df / dx
    } else if dx.is_finite() {
        fa / dx - fb / dx
    } else {
        (0.5 * fa - 0.5 * fb) / (0.5 * xa - 0.5 * xb)
    }
}

/// Differentiate the quadratic through the actual floating-point nodes.
/// Combining secant slopes removes a constant residual offset before the
/// weighted sum, unlike the nominal -3*f0 + 4*f1 - f2 formula.
fn fd_three_point(x: f64, first: f64, second: f64, f0: f64, f1: f64, f2: f64) -> f64 {
    let d1 = first - x;
    let d2 = second - x;
    if d1 == -d2 {
        return fd_quotient(f2, f1, second, first);
    }
    let span = second - first;
    let weight = if d1.is_finite() && span.is_finite() {
        d1 / span
    } else {
        (0.5 * first - 0.5 * x) / (0.5 * second - 0.5 * first)
    };
    let slope1 = fd_quotient(f1, f0, first, x);
    let slope2 = fd_quotient(f2, f0, second, x);
    let derivative = slope1 + weight * (slope1 - slope2);
    if derivative.is_finite() || !slope1.is_finite() || !slope2.is_finite() {
        return derivative;
    }
    // Opposite large slopes can overflow their difference even when the
    // weighted derivative is finite.
    let scale = slope1.abs().max(slope2.abs());
    scale * (slope1 / scale + weight * (slope1 / scale - slope2 / scale))
}

/// SciPy's dense two-/three-point `approx_derivative`.
fn fd_jacobian_scipy<F>(
    residuals: &F,
    x: &[f64],
    f0: &[f64],
    lb: &[f64],
    ub: &[f64],
    method: FiniteDiffMethod,
    rel_step: Option<f64>,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let points = fd_points_scipy(x, lb, ub, method, rel_step)?;
    let mut columns = Vec::with_capacity(x.len());
    let mut x1 = x.to_vec();
    let mut x2 = x.to_vec();
    for (i, &FiniteDifferenceStencil { first, second }) in points.iter().enumerate() {
        x1[i] = first;
        let f1 = residuals(&x1);
        check_jacobian_residual_shape(&f1, f0.len())?;
        let column: Vec<f64> = if let Some(second) = second {
            x2[i] = second;
            let f2 = residuals(&x2);
            check_jacobian_residual_shape(&f2, f0.len())?;
            f0.iter()
                .zip(&f1)
                .zip(&f2)
                .map(|((&f0, &f1), &f2)| fd_three_point(x[i], first, second, f0, f1, f2))
                .collect()
        } else {
            f1.iter()
                .zip(f0)
                .map(|(&a, &b)| fd_quotient(a, b, first, x[i]))
                .collect()
        };
        x1[i] = x[i];
        x2[i] = x[i];
        columns.push(column);
    }
    Ok((0..f0.len())
        .map(|r| columns.iter().map(|col| col[r]).collect())
        .collect())
}

/// Sparse finite differences with the same per-coordinate stencils as the
/// dense path. Only declared nonzeros are formed; neither differentiation nor
/// the returned matrix allocates an m-by-n array.
fn fd_jacobian_grouped<F>(
    residuals: &F,
    x: &[f64],
    f0: &[f64],
    lb: &[f64],
    ub: &[f64],
    method: FiniteDiffMethod,
    rel_step: Option<f64>,
    sparsity: &JacobianSparsity,
) -> Result<LeastSquaresMatrix, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let points = fd_points_scipy(x, lb, ub, method, rel_step)?;
    let mut data = vec![0.0; sparsity.nnz()];
    let mut x1 = x.to_vec();
    let mut x2 = x.to_vec();
    for group in &sparsity.groups {
        for &c in group {
            x1[c] = points[c].first;
            x2[c] = points[c].second.unwrap_or(x[c]);
        }
        let f1 = residuals(&x1);
        check_jacobian_residual_shape(&f1, f0.len())?;
        let f2 = if group.iter().any(|&c| points[c].second.is_some()) {
            let values = residuals(&x2);
            check_jacobian_residual_shape(&values, f0.len())?;
            Some(values)
        } else {
            None
        };
        for &c in group {
            for &(r, entry) in &sparsity.column_entries[c] {
                data[entry] = if let Some(second) = points[c].second {
                    fd_three_point(
                        x[c],
                        points[c].first,
                        second,
                        f0[r],
                        f1[r],
                        f2.as_ref().expect("three-point group has a second sample")[r],
                    )
                } else {
                    fd_quotient(f1[r], f0[r], points[c].first, x[c])
                };
            }
        }
        for &c in group {
            x1[c] = x[c];
            x2[c] = x[c];
        }
    }
    Ok(LeastSquaresMatrix::Sparse(CsrMatrix::from_components(
        sparsity.shape,
        data,
        sparsity.indices.clone(),
        sparsity.indptr.clone(),
        true,
    )?))
}

/// Fill the LM Jacobian. As in SciPy >= 1.16, residual calls made solely for
/// numerical differentiation do not consume `max_nfev` or contribute to `nfev`.
fn lm_jacobian_into<F>(
    residuals: &F,
    x: &[f64],
    residual: &[f64],
    options: &LeastSquaresOptions,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    if let Some(analytic) = &options.jac {
        *jac = match analytic.evaluate(x, residual.len())? {
            LeastSquaresMatrix::Dense(matrix) => matrix,
            LeastSquaresMatrix::Sparse(_) | LeastSquaresMatrix::Operator(_) => {
                return Err(OptError::InvalidArgument {
                    detail: String::from("method 'lm' requires a dense Jacobian"),
                });
            }
        };
    } else if options.diff_method == FiniteDiffMethod::ThreePoint {
        *jac = fd_jacobian_scipy(
            residuals,
            x,
            residual,
            &vec![f64::NEG_INFINITY; x.len()],
            &vec![f64::INFINITY; x.len()],
            options.diff_method,
            options.diff_step,
        )?;
    } else {
        finite_diff_jacobian_parallel_into(
            residuals,
            x,
            residual,
            options.diff_step,
            jac,
            x_perturbed,
        )?;
    }
    if jac.iter().flatten().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("Jacobian must not contain NaN or Inf"),
        });
    }
    Ok(())
}

/// SciPy raises `ValueError` with this text when its `trf` SVD meets a non-finite matrix.
impl From<trf::NonFiniteJacobian> for OptError {
    fn from(_: trf::NonFiniteJacobian) -> Self {
        Self::NonFiniteInput {
            detail: String::from("array must not contain infs or NaNs"),
        }
    }
}

impl From<fsci_sparse::SparseError> for OptError {
    fn from(error: fsci_sparse::SparseError) -> Self {
        match error {
            fsci_sparse::SparseError::NonFiniteInput { message } => {
                Self::NonFiniteInput { detail: message }
            }
            error => Self::NotConverged {
                detail: format!("LSMR subproblem failed: {error}"),
            },
        }
    }
}

/// `trf`/`dogbox` setup: validate tolerances and bounds, require finite initial
/// residuals, and move `x0` off the bounds only for TRF (Dogbox accepts exact bounds).
fn least_squares_trust_region<F>(
    residuals: &F,
    x0: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must have at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    let (lb, ub) = match bounds {
        Some((lo, hi)) => (lo.to_vec(), hi.to_vec()),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };
    if lb.len() != n || ub.len() != n {
        return Err(OptError::InvalidArgument {
            detail: String::from("Inconsistent shapes between bounds and `x0`."),
        });
    }
    if lb
        .iter()
        .zip(&ub)
        .any(|(l, u)| l.partial_cmp(u) != Some(std::cmp::Ordering::Less))
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("Each lower bound must be strictly less than each upper bound."),
        });
    }
    if x0
        .iter()
        .zip(lb.iter().zip(&ub))
        .any(|(x, (l, u))| x < l || x > u)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("Initial guess is outside of provided bounds"),
        });
    }
    for (name, tol) in [
        ("ftol", options.ftol),
        ("xtol", options.xtol),
        ("gtol", options.gtol),
    ] {
        if tol.is_nan() || tol < 0.0 {
            return Err(OptError::InvalidArgument {
                detail: format!("{name} must be >= 0"),
            });
        }
    }
    if options.ftol < f64::EPSILON && options.xtol < f64::EPSILON && options.gtol < f64::EPSILON {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "At least one of the tolerances must be higher than machine epsilon ({:.2e}).",
                f64::EPSILON
            ),
        });
    }
    if !(options.f_scale > 0.0 && options.f_scale.is_finite()) {
        return Err(OptError::InvalidArgument {
            detail: String::from("f_scale must be positive and finite"),
        });
    }
    if options.custom_loss.is_some() && options.loss != LossKind::Linear {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "custom_loss and a non-linear built-in loss are mutually exclusive",
            ),
        });
    }
    if options.max_nfev == Some(0) {
        return Err(OptError::InvalidArgument {
            detail: String::from("max_nfev must be positive"),
        });
    }
    let x_scale = match &options.x_scale {
        Some(XScale::Fixed(scale)) => {
            if scale.len() != n {
                return Err(OptError::InvalidArgument {
                    detail: String::from("Inconsistent shapes between x_scale and `x0`."),
                });
            }
            if scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(OptError::InvalidArgument {
                    detail: String::from("x_scale entries must be positive and finite"),
                });
            }
            XScale::Fixed(scale.clone())
        }
        Some(XScale::Jac) => XScale::Jac,
        None => XScale::Fixed(vec![1.0; n]),
    };
    if let Some(r) = options.diff_step
        && !(r.is_finite() && r > 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("diff_step must be finite and > 0"),
        });
    }
    if let TrustRegionSolver::Lsmr(lsmr) = options.tr_solver
        && !lsmr.validate()
    {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "LSMR atol, btol and conlim must be finite and nonnegative; damp must be finite",
            ),
        });
    }
    let dogbox = options.method == Some(LeastSquaresMethod::Dogbox);
    let x0 = if dogbox {
        x0.to_vec()
    } else {
        trf::make_strictly_feasible(x0, &lb, &ub, 1e-10)
    };
    let f0 = residuals(&x0);
    if f0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("Residuals are not finite in the initial point."),
        });
    }
    let m = f0.len();
    // A callable Jacobian takes precedence, as in SciPy: its values and
    // storage are authoritative and a numerical sparsity hint is ignored.
    let sparsity = options
        .jac
        .is_none()
        .then_some(options.jac_sparsity.as_ref())
        .flatten();
    if let Some(pattern) = sparsity {
        if pattern.shape() != Shape2D::new(m, n) {
            return Err(OptError::InvalidArgument {
                detail: format!("jac_sparsity must have shape ({m}, {n})"),
            });
        }
        if options.tr_solver == TrustRegionSolver::Exact {
            return Err(OptError::InvalidArgument {
                detail: String::from("tr_solver='exact' is incompatible with jac_sparsity"),
            });
        }
    }
    let evaluate_jacobian = |x: &[f64], f: &[f64]| -> Result<LeastSquaresMatrix, OptError> {
        if let Some(analytic) = &options.jac {
            analytic.evaluate(x, m)
        } else if let Some(pattern) = sparsity {
            fd_jacobian_grouped(
                residuals,
                x,
                f,
                &lb,
                &ub,
                options.diff_method,
                options.diff_step,
                pattern,
            )
        } else {
            fd_jacobian_scipy(
                residuals,
                x,
                f,
                &lb,
                &ub,
                options.diff_method,
                options.diff_step,
            )
            .map(LeastSquaresMatrix::Dense)
        }
    };
    let j0 = evaluate_jacobian(&x0, &f0)?;
    if j0.is_operator() && matches!(x_scale, XScale::Jac) {
        return Err(OptError::InvalidArgument {
            detail: String::from("x_scale='jac' is not supported for a matrix-free Jacobian"),
        });
    }
    let solver = match options.tr_solver {
        TrustRegionSolver::Auto if j0.as_dense().is_none() => {
            TrustRegionSolver::Lsmr(TrustRegionLsmrOptions::default())
        }
        TrustRegionSolver::Auto => TrustRegionSolver::Exact,
        TrustRegionSolver::Exact if j0.as_dense().is_none() => {
            return Err(OptError::InvalidArgument {
                detail: String::from("tr_solver='exact' requires a dense Jacobian"),
            });
        }
        solver => solver,
    };
    let loss = if let Some(callback) = &options.custom_loss {
        Some(LossFunction::Custom {
            callback: callback.clone(),
            f_scale: options.f_scale,
        })
    } else {
        (options.loss != LossKind::Linear).then_some(LossFunction::Builtin(Loss {
            kind: options.loss,
            f_scale: options.f_scale,
        }))
    };
    let mut fun = |x: &[f64]| -> Result<Vec<f64>, OptError> {
        let f = residuals(x);
        if f.len() != m {
            return Err(OptError::InvalidArgument {
                detail: format!("residuals changed length from {m} to {}", f.len()),
            });
        }
        Ok(f)
    };
    let mut jac = |x: &[f64], f: &[f64]| -> Result<LeastSquaresMatrix, OptError> {
        let matrix = evaluate_jacobian(x, f)?;
        if solver == TrustRegionSolver::Exact && matrix.as_dense().is_none() {
            return Err(OptError::InvalidArgument {
                detail: String::from(
                    "tr_solver='exact' requires a dense Jacobian at every evaluation",
                ),
            });
        }
        if matrix.is_operator() && matches!(x_scale, XScale::Jac) {
            return Err(OptError::InvalidArgument {
                detail: String::from("x_scale='jac' is not supported for a matrix-free Jacobian"),
            });
        }
        Ok(matrix)
    };
    let solve = if dogbox { trf::dogbox } else { trf::trf };
    let out = solve(
        &mut fun,
        &mut jac,
        &x0,
        f0,
        j0,
        &lb,
        &ub,
        options.ftol,
        options.xtol,
        options.gtol,
        options.max_nfev,
        &x_scale,
        loss.as_ref(),
        solver,
        options.mode,
        options.callback.as_ref(),
    )?;
    Ok(LeastSquaresResult {
        // status: SciPy's code; success = status > 0 (a gtol/ftol/xtol test passed)
        success: out.status > 0,
        message: termination_message(out.status).to_string(),
        nit: out.nit,
        x: out.x,
        fun: out.fun,
        cost: out.cost,
        nfev: out.nfev,
        njev: out.njev,
        jac: out.jac,
        status: out.status,
        optimality: out.optimality,
        grad: out.grad,
        active_mask: out.active_mask,
    })
}

/// Nonlinear curve fitting.
///
/// Finds the parameters `popt` such that `f(xdata, popt)` best fits `ydata`
/// in the least-squares sense.
///
/// Equivalent to `scipy.optimize.curve_fit`.
///
/// # Arguments
/// * `f` - Model function `f(x, params) -> y` where `x` is a single data point
/// * `xdata` - Independent variable data
/// * `ydata` - Dependent variable data (same length as xdata)
/// * `options` - Curve fitting options (initial guess, tolerances, etc.)
pub fn curve_fit<F>(
    f: F,
    xdata: &[f64],
    ydata: &[f64],
    mut options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let whitening = FitWhiteningCache::new(options.sigma.take(), xdata.len());
    curve_fit_impl(&f, xdata, ydata, None, options, &whitening)
}

/// Shared scalar/batched and bounded/unbounded curve fitting. Select observations
/// before preparing their measurement-error transform, then use that transform
/// consistently for residuals and analytic/numerical Jacobians.
fn curve_fit_impl<F>(
    f: &F,
    xdata: &[f64],
    ydata: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    options: CurveFitOptions,
    whitening: &FitWhiteningCache,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let original_observations = xdata.len();
    let observations = FitObservations::new(xdata, ydata, &options)?;
    let xdata = observations.x.as_ref();
    let ydata = observations.y.as_ref();
    let whitening = whitening.get(observations.retained.as_deref())?;

    let p0 = options.p0.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("p0 (initial parameter guess) is required"),
    })?;
    let n_params = p0.len();
    let method = options.ls_options.method.unwrap_or(if bounds.is_some() {
        LeastSquaresMethod::Trf
    } else {
        LeastSquaresMethod::Lm
    });

    if method == LeastSquaresMethod::Lm && xdata.len() < n_params {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "number of data points ({}) must be >= number of parameters ({n_params})",
                xdata.len()
            ),
        });
    }

    if options
        .ls_options
        .jac
        .as_ref()
        .is_some_and(|jacobian| jacobian.matrix_free)
    {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "curve_fit requires explicit Jacobian entries for covariance; use least_squares for a matrix-free fit",
            ),
        });
    }

    // Differentiate the weighted residuals when no analytic Jacobian is supplied.
    // Analytic model Jacobians need the identical linear transform below.
    let residuals = |params: &[f64]| -> Vec<f64> {
        let mut values: Vec<f64> = xdata
            .iter()
            .zip(ydata.iter())
            .map(|(&xi, &yi)| f(xi, params) - yi)
            .collect();
        whitening.residuals(&mut values);
        values
    };

    // SciPy's curve_fit(method=None) runs 'lm' when there are no bounds.
    let mut ls_options = LeastSquaresOptions {
        method: Some(method),
        ..options.ls_options
    };
    if ls_options.jac.is_none()
        && let Some(pattern) = &ls_options.jac_sparsity
    {
        let selected;
        let pattern = if let Some(retained) = &observations.retained {
            if pattern.shape != Shape2D::new(original_observations, n_params) {
                return Err(OptError::InvalidArgument {
                    detail: String::from(
                        "jac_sparsity must describe the original observation grid",
                    ),
                });
            }
            selected = JacobianSparsity::from_rows(
                retained
                    .iter()
                    .map(|&r| pattern.indices[pattern.indptr[r]..pattern.indptr[r + 1]].to_vec())
                    .collect(),
                n_params,
            )?;
            &selected
        } else {
            pattern
        };
        ls_options.jac_sparsity = Some(whitening.sparsity(pattern)?);
    }
    if (observations.retained.is_some() || !matches!(whitening.as_ref(), FitWhitening::Identity))
        && let Some(jacobian) = ls_options.jac.take()
    {
        let whitening = Arc::clone(&whitening);
        let retained = observations.retained.clone();
        ls_options.jac = Some(LeastSquaresJacobian {
            callback: Arc::new(move |params| {
                let matrix = (jacobian.callback)(params)?;
                // Validate the original grid before selecting rows. Nonfinite
                // values in dropped rows are harmless; retained values are checked
                // by LeastSquaresJacobian::evaluate after selection and whitening.
                let valid_shape = match &matrix {
                    LeastSquaresMatrix::Dense(rows) => {
                        rows.len() == original_observations
                            && rows.iter().all(|row| row.len() == params.len())
                    }
                    LeastSquaresMatrix::Sparse(csr) => {
                        csr.shape() == Shape2D::new(original_observations, params.len())
                    }
                    LeastSquaresMatrix::Operator(operator) => {
                        operator.shape() == Shape2D::new(original_observations, params.len())
                    }
                };
                if !valid_shape {
                    return Err(OptError::InvalidArgument {
                        detail: format!(
                            "Jacobian must have shape ({original_observations}, {}) before omission",
                            params.len()
                        ),
                    });
                }
                let matrix = if let Some(retained) = &retained {
                    match matrix {
                        LeastSquaresMatrix::Dense(rows) => LeastSquaresMatrix::Dense(
                            retained.iter().map(|&r| rows[r].clone()).collect(),
                        ),
                        LeastSquaresMatrix::Sparse(csr) => {
                            let mut data = Vec::new();
                            let mut indices = Vec::new();
                            let mut indptr = vec![0];
                            for &r in retained {
                                let range = csr.indptr()[r]..csr.indptr()[r + 1];
                                data.extend_from_slice(&csr.data()[range.clone()]);
                                indices.extend_from_slice(&csr.indices()[range]);
                                indptr.push(data.len());
                            }
                            LeastSquaresMatrix::Sparse(CsrMatrix::from_components(
                                Shape2D::new(retained.len(), params.len()),
                                data,
                                indices,
                                indptr,
                                false,
                            )?)
                        }
                        LeastSquaresMatrix::Operator(_) => {
                            return Err(OptError::InvalidArgument {
                                detail: String::from(
                                    "curve_fit requires explicit Jacobian entries",
                                ),
                            });
                        }
                    }
                } else {
                    matrix
                };
                whitening.jacobian_matrix(matrix)
            }),
            matrix_free: false,
        });
    }
    let ls_result = match bounds {
        Some((lower, upper)) => least_squares_bounded(residuals, &p0, lower, upper, ls_options)?,
        None => least_squares(residuals, &p0, ls_options)?,
    };
    require_fit_converged(&ls_result)?;

    let pcov = curve_fit_covariance(&ls_result, method, options.absolute_sigma)?;
    warn_if_covariance_not_estimated(&pcov);

    Ok(CurveFitResult {
        popt: ls_result.x.clone(),
        pcov,
        ls_result,
    })
}

/// SciPy's `curve_fit` fills `pcov` with inf and warns `OptimizeWarning` when the covariance
/// is indeterminate (a singular LM Jacobian) or, with relative sigma, when there are no more
/// data points than parameters. TRF and Dogbox instead use a pseudoinverse for a singular
/// Jacobian.
fn warn_if_covariance_not_estimated(pcov: &[Vec<f64>]) {
    if pcov.iter().flatten().any(|v| !v.is_finite()) {
        fsci_runtime::warn(
            fsci_runtime::WarningCategory::OptimizeWarning,
            "Covariance of the parameters could not be estimated",
        );
    }
}

/// SciPy's `curve_fit` raises `RuntimeError("Optimal parameters not found: " + res.message)`
/// when the least-squares solve fails; returning its last iterate as `popt` hands the caller an
/// unconverged fit with no signal short of inspecting `ls_result` (frankenscipy-szq1n.7).
fn require_fit_converged(ls_result: &LeastSquaresResult) -> Result<(), OptError> {
    if ls_result.success {
        Ok(())
    } else {
        Err(OptError::NotConverged {
            detail: format!("Optimal parameters not found: {}", ls_result.message),
        })
    }
}

/// Box-constrained nonlinear least squares, `scipy.optimize.least_squares(fun, x0,
/// bounds=(lower, upper))`: Trust Region Reflective by default, or rectangular Dogbox.
/// Active bounds are reported in `active_mask` (`±inf` = no bound).
/// `method = Lm` is refused, as SciPy refuses it ("Method 'lm' doesn't support bounds").
///
/// This used to reparameterise each bounded coordinate (logistic / softplus) and run LM,
/// which only approached an active bound asymptotically (frankenscipy-1ksfv.6).
pub fn least_squares_bounded<F>(
    residuals: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastSquaresResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x0.len();
    if lower.len() != n || upper.len() != n {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "lower ({}) and upper ({}) bounds must match x0 length ({n})",
                lower.len(),
                upper.len()
            ),
        });
    }
    if options.method == Some(LeastSquaresMethod::Lm)
        && (lower.iter().any(|v| *v != f64::NEG_INFINITY)
            || upper.iter().any(|v| *v != f64::INFINITY))
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("Method 'lm' doesn't support bounds."),
        });
    }
    if options.method == Some(LeastSquaresMethod::Lm) {
        return least_squares_lm(residuals, x0, options);
    }
    least_squares_trust_region(&residuals, x0, Some((lower, upper)), options)
}

/// Box-constrained curve fitting: the bounded analogue of [`curve_fit`], matching
/// `scipy.optimize.curve_fit(f, xdata, ydata, p0=..., bounds=(lower, upper))`.
///
/// Runs [`least_squares_bounded`], i.e. SciPy's `trf`, as `curve_fit` does whenever bounds
/// are given.
pub fn curve_fit_bounded<F>(
    f: F,
    xdata: &[f64],
    ydata: &[f64],
    lower: &[f64],
    upper: &[f64],
    mut options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let whitening = FitWhiteningCache::new(options.sigma.take(), xdata.len());
    curve_fit_impl(&f, xdata, ydata, Some((lower, upper)), options, &whitening)
}

/// Batched curve fitting: fit the same model `f` (and shared `xdata`) to MANY independent
/// `ydata` rows, returning one optimal-parameter vector per row. This is the vmap-over-solver
/// primitive SciPy lacks — there you loop `curve_fit` in Python, paying the per-call overhead
/// N times serially; here the N independent fits are fanned across cores and the model is an
/// inlined Rust closure. Row `i` of the output is byte-identical to `curve_fit(f, xdata,
/// &ydata_rows[i], options).popt`.
///
/// Common in imaging / signal processing (a decay or peak fit per pixel / channel / trace).
pub fn curve_fit_many<F>(
    f: F,
    xdata: &[f64],
    ydata_rows: &[Vec<f64>],
    mut options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let whitening = FitWhiteningCache::new(options.sigma.take(), xdata.len());
    // The factored transform is shared. Do not also copy an O(m²) covariance
    // input into every worker's per-fit options.
    let fit_options = CurveFitOptions {
        sigma: None,
        ..options
    };
    let opts_ref = &fit_options;
    let fit_one = move |row: &[f64]| {
        curve_fit_impl(f_ref, xdata, row, None, opts_ref.clone(), &whitening).map(|r| r.popt)
    };

    // Each fit is an independent ~0.1 ms LM solve (heavy per item) → fan whole rows across
    // cores, capped by the row count; a tiny batch stays serial to dodge the spawn floor.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 8 {
        return ydata_rows.iter().map(|row| fit_one(row)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let fit_one = &fit_one;
    let chunk_results: Vec<Result<Vec<Vec<f64>>, OptError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| fit_one(&ydata_rows[i]))
                        .collect::<Result<Vec<_>, _>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("curve_fit_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Vectorised [`least_squares`] over `n` independent parameter rows — the
/// vmap-over-solver form for `scipy.optimize.least_squares`. `residuals(x, params)`
/// is an inlined Rust closure (no per-iteration Python callback, unlike a looped
/// scipy `least_squares`), and the independent (heavy, iterative LM/finite-diff)
/// solves fan whole-row across cores; a tiny batch stays serial. Entry `k` equals
/// `least_squares(|x| residuals(x, &param_rows[k]), x0, options)`.
pub fn least_squares_many<F>(
    residuals: F,
    x0: &[f64],
    param_rows: &[Vec<f64>],
    options: LeastSquaresOptions,
) -> Vec<Result<LeastSquaresResult, OptError>>
where
    F: Fn(&[f64], &[f64]) -> Vec<f64> + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let r_ref = &residuals;
    let solve_one = move |params: &[f64]| least_squares(|x| r_ref(x, params), x0, options.clone());

    // Each solve is a heavy iterative LM fit → fan whole rows across cores, capped
    // by the row count; a tiny batch stays serial to dodge the spawn floor.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<LeastSquaresResult, OptError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("least_squares_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Batched box-constrained curve fitting — [`curve_fit_bounded`] fanned across many `ydata`
/// rows, the bounded analogue of [`curve_fit_many`]. Row `i` is byte-identical to
/// `curve_fit_bounded(f, xdata, &ydata_rows[i], lower, upper, options).popt`.
pub fn curve_fit_bounded_many<F>(
    f: F,
    xdata: &[f64],
    ydata_rows: &[Vec<f64>],
    lower: &[f64],
    upper: &[f64],
    mut options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let whitening = FitWhiteningCache::new(options.sigma.take(), xdata.len());
    let fit_options = CurveFitOptions {
        sigma: None,
        ..options
    };
    let opts_ref = &fit_options;
    let fit_one = move |row: &[f64]| {
        curve_fit_impl(
            f_ref,
            xdata,
            row,
            Some((lower, upper)),
            opts_ref.clone(),
            &whitening,
        )
        .map(|r| r.popt)
    };

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 8 {
        return ydata_rows.iter().map(|row| fit_one(row)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let fit_one = &fit_one;
    let chunk_results: Vec<Result<Vec<Vec<f64>>, OptError>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| fit_one(&ydata_rows[i]))
                        .collect::<Result<Vec<_>, _>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("curve_fit_bounded_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr?);
    }
    Ok(out)
}

/// Result from [`leastsq`] — the classic MINPACK-style interface.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastsqResult {
    /// Solution parameters (scipy's `x`).
    pub x: Vec<f64>,
    /// Unscaled covariance estimate `cov_x = (JᵀJ)⁻¹`, or `None` if the
    /// Jacobian is rank-deficient at the solution (matching scipy, which
    /// returns `None` for `cov_x` in that case).
    pub cov_x: Option<Vec<Vec<f64>>>,
    /// Number of function evaluations (scipy's `infodict['nfev']`).
    pub nfev: usize,
    /// Human-readable termination message (scipy's `mesg`).
    pub mesg: String,
    /// MINPACK termination flag (scipy's `ier`). A value in `1..=4` means a
    /// solution was found.
    pub ier: i32,
}

/// Minimize the sum of squares of a set of equations (Levenberg-Marquardt).
///
/// Mirrors `scipy.optimize.leastsq(func, x0)`: `func` maps the parameter vector
/// to a vector of residuals, and the solver returns the optimal parameters
/// together with the unscaled covariance `cov_x = (JᵀJ)⁻¹` and a MINPACK-style
/// `ier` termination flag (`ier in {1, 2, 3, 4}` indicates success).
///
/// This is the legacy interface that predates [`least_squares`]; both wrap the
/// same Levenberg-Marquardt core, so they converge to the same minimum.
pub fn leastsq<F>(
    func: F,
    x0: &[f64],
    options: LeastSquaresOptions,
) -> Result<LeastsqResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    // leastsq is SciPy's MINPACK (Levenberg–Marquardt) interface.
    let ls = least_squares(
        func,
        x0,
        LeastSquaresOptions {
            method: Some(LeastSquaresMethod::Lm),
            ..options
        },
    )?;
    let n = x0.len();

    // scipy's `cov_x` is the *unscaled* inverse (JᵀJ)⁻¹; reuse the
    // `absolute_sigma` branch of the covariance helper. A rank-deficient
    // Jacobian yields a non-finite inverse -> report `None` like scipy.
    let jac = ls.jac.as_dense().ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("leastsq requires a dense LM Jacobian"),
    })?;
    let cov = compute_covariance(jac, &ls.fun, n, true);
    let cov_x = if cov.iter().flatten().all(|v| v.is_finite()) {
        Some(cov)
    } else {
        None
    };

    // Map the convergence reason onto the MINPACK `ier` codes scipy reports.
    let (ier, mesg) = if ls.success {
        if ls.message.contains("ftol") {
            (
                1,
                "Both actual and predicted relative reductions in the sum of squares\n  are at most ftol.",
            )
        } else if ls.message.contains("xtol") {
            (
                2,
                "The relative error between two consecutive iterates is at most xtol.",
            )
        } else if ls.message.contains("gtol") {
            (
                4,
                "The cosine of the angle between func(x) and any column of the\n  Jacobian is at most gtol in absolute value.",
            )
        } else {
            (1, "The solution converged.")
        }
    } else {
        (5, "Number of calls to function has reached maxfev = 0.")
    };

    Ok(LeastsqResult {
        x: ls.x,
        cov_x,
        nfev: ls.nfev,
        mesg: mesg.to_string(),
        ier,
    })
}

// ────────────────────────── helpers ──────────────────────────

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn resize_square(matrix: &mut Vec<Vec<f64>>, n: usize) {
    if matrix.len() != n {
        matrix.resize_with(n, Vec::new);
    }
    for row in matrix.iter_mut() {
        row.resize(n, 0.0);
        row.fill(0.0);
    }
}

/// Reject a changing residual shape before indexing a numerical Jacobian column.
fn check_jacobian_residual_shape(residual: &[f64], expected: usize) -> Result<(), OptError> {
    if residual.len() != expected {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "residuals changed length from {expected} to {} during Jacobian evaluation",
                residual.len()
            ),
        });
    }
    Ok(())
}

/// Unbounded two-point stencils for LM, including their representable displacement.
/// Compute them before any worker calls the model so a nonfinite point cannot be
/// mistaken for a zero derivative when the model ignores that coordinate.
fn lm_difference_points(x: &[f64], rel_step: Option<f64>) -> Result<Vec<(f64, f64)>, OptError> {
    x.iter()
        .map(|&value| {
            let point = value + fd_step_scipy(value, FiniteDiffMethod::TwoPoint, rel_step);
            let displacement = point - value;
            if !point.is_finite() || !displacement.is_finite() || displacement == 0.0 {
                return Err(OptError::NonFiniteInput {
                    detail: String::from(
                        "finite-difference perturbation must have a finite nonzero displacement",
                    ),
                });
            }
            Ok((point, displacement))
        })
        .collect()
}

/// Compute finite-difference Jacobian into fixed-shape scratch.
pub(super) fn finite_diff_jacobian_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    rel_step: Option<f64>,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = r0.len();
    let points = lm_difference_points(x, rel_step)?;
    if jac.len() != m {
        jac.resize_with(m, Vec::new);
    }
    for row in jac.iter_mut() {
        row.resize(n, 0.0);
    }
    x_perturbed.resize(n, 0.0);
    x_perturbed.copy_from_slice(x);

    for j in 0..n {
        let (point, step) = points[j];
        let original = x_perturbed[j];
        x_perturbed[j] = point;
        let r_plus = residuals(x_perturbed);
        x_perturbed[j] = original; // restore
        check_jacobian_residual_shape(&r_plus, m)?;
        for i in 0..m {
            jac[i][j] = (r_plus[i] - r0[i]) / step;
        }
    }
    Ok(())
}

/// Finite-difference Jacobian with the independent columns evaluated in parallel.
///
/// Each column `j` perturbs one parameter and re-evaluates the full `m`-residual
/// vector — the columns are independent, so their (expensive) residual evals run
/// on a worker pool and the `jac` matrix is filled serially afterwards. The result
/// is byte-identical to [`finite_diff_jacobian_into`] (each column uses the same
/// relative-step policy and rounded displacement in the one-sided difference).
///
/// Parallelism pays only when `n_params * m_data * cost_per_eval` is large enough
/// to amortize thread spawn (measured: a hard loss below ~30K work, 2.75–13.66x
/// above it), so below the work-gate this falls back to the serial routine — which
/// also keeps `available_parallelism()` (a syscall) off the small-problem hot path.
fn finite_diff_jacobian_parallel_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    rel_step: Option<f64>,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    let n = x.len();
    let m = r0.len();
    // Work-gate BEFORE any syscall: below it, the serial routine is faster.
    if n < 4 || m < 8192 || n.saturating_mul(m) < 131_072 {
        return finite_diff_jacobian_into(residuals, x, r0, rel_step, jac, x_perturbed);
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n);
    if nthreads <= 1 {
        return finite_diff_jacobian_into(residuals, x, r0, rel_step, jac, x_perturbed);
    }

    let points = lm_difference_points(x, rel_step)?;
    if jac.len() != m {
        jac.resize_with(m, Vec::new);
    }
    for row in jac.iter_mut() {
        row.resize(n, 0.0);
    }

    // Each worker owns a private perturbed-parameter buffer and returns
    // `(column index, perturbed residual vector, step)` for its column range.
    let per = n.div_ceil(nthreads);
    let columns: Vec<(usize, Vec<f64>, f64)> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * per;
                if lo >= n {
                    return None;
                }
                let hi = (lo + per).min(n);
                let points = &points;
                Some(scope.spawn(move || {
                    let mut xp = x.to_vec();
                    (lo..hi)
                        .map(|j| {
                            let (point, step) = points[j];
                            let original = xp[j];
                            xp[j] = point;
                            let r_plus = residuals(&xp);
                            xp[j] = original;
                            (j, r_plus, step)
                        })
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|h| h.join().expect("finite_diff_jacobian worker panicked"))
            .collect()
    });

    for (j, r_plus, step) in &columns {
        check_jacobian_residual_shape(r_plus, m)?;
        let step = *step;
        for i in 0..m {
            jac[i][*j] = (r_plus[i] - r0[i]) / step;
        }
    }
    let _ = x_perturbed;
    Ok(())
}

/// Compute J * v into fixed-shape scratch.
fn mat_vec_into(jac: &[Vec<f64>], v: &[f64], result: &mut Vec<f64>) {
    result.resize(jac.len(), 0.0);
    for (out, row) in result.iter_mut().zip(jac.iter()) {
        *out = dot_vec(row, v);
    }
}

/// Orthogonal reduction of the scaled residual Jacobian and right-hand side.
/// Adding diagonal rows to R solves the regularized least-squares problem
/// without forming JᵀJ, including when its rounded normal matrix is singular.
struct LmQr {
    upper: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    max_column_norm: f64,
}

#[derive(Default)]
struct LmQrScratch {
    upper: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    row: Vec<f64>,
    step: Vec<f64>,
}

/// Eliminate a new row with Givens rotations. `hypot` forms each rotation
/// without squaring the entries; the same orthogonal operations transform f.
fn lm_qr_add_row(upper: &mut [Vec<f64>], rhs: &mut [f64], row: &mut [f64], mut value: f64) {
    for k in 0..rhs.len() {
        if row[k] == 0.0 {
            continue;
        }
        let diagonal = upper[k][k].hypot(row[k]);
        let cosine = upper[k][k] / diagonal;
        let sine = row[k] / diagonal;
        upper[k][k] = diagonal;
        for (entry, incoming) in upper[k][k + 1..].iter_mut().zip(&mut row[k + 1..]) {
            let old = *entry;
            *entry = cosine * old + sine * *incoming;
            *incoming = cosine * *incoming - sine * old;
        }
        let old = rhs[k];
        rhs[k] = cosine * old + sine * value;
        value = cosine * value - sine * old;
    }
}

impl LmQr {
    fn new(jac: &[Vec<f64>], residuals: &[f64]) -> Result<Self, OptError> {
        let n = jac[0].len();
        let mut upper = vec![vec![0.0; n]; n];
        let mut rhs = vec![0.0; n];
        let mut row = vec![0.0; n];
        for (source, &value) in jac.iter().zip(residuals) {
            row.copy_from_slice(source);
            lm_qr_add_row(&mut upper, &mut rhs, &mut row, value);
        }
        let max_column_norm = (0..n)
            .map(|column| {
                upper
                    .iter()
                    .take(column + 1)
                    .fold(0.0_f64, |norm, row| norm.hypot(row[column]))
            })
            .fold(0.0_f64, f64::max);
        if upper.iter().flatten().any(|value| !value.is_finite())
            || rhs.iter().any(|value| !value.is_finite())
            || !max_column_norm.is_finite()
        {
            return Err(OptError::NonFiniteInput {
                detail: String::from("LM orthogonal Jacobian factorization is not finite"),
            });
        }
        Ok(Self {
            upper,
            rhs,
            max_column_norm,
        })
    }

    /// Minimize ‖R p + Qᵀf‖² + damping²·‖p‖². An exactly empty row of the
    /// undamped factor leaves its free variable at zero. Positive damping
    /// retains the full triangular factor, including dependent directions.
    fn solve(&self, damping: f64, scratch: &mut LmQrScratch) -> Option<f64> {
        let n = self.rhs.len();
        scratch.upper.clone_from(&self.upper);
        scratch.rhs.clone_from(&self.rhs);
        scratch.row.resize(n, 0.0);
        if damping > 0.0 {
            for column in 0..n {
                scratch.row.fill(0.0);
                scratch.row[column] = damping;
                lm_qr_add_row(&mut scratch.upper, &mut scratch.rhs, &mut scratch.row, 0.0);
            }
        }
        // An overflowing augmented diagonal can turn a nonzero residual into
        // an apparently zero step. Reject that factor before back-substitution
        // so the bounded damping search cannot report false convergence.
        if scratch
            .upper
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
            || scratch.rhs.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        scratch.step.resize(n, 0.0);
        for row in (0..n).rev() {
            let mut value = -scratch.rhs[row];
            for (&entry, &step) in scratch.upper[row][row + 1..]
                .iter()
                .zip(&scratch.step[row + 1..])
            {
                value -= entry * step;
            }
            let diagonal = scratch.upper[row][row];
            if diagonal == 0.0 {
                // A row receives entries only after a nonzero pivot arrives,
                // so a zero diagonal here means the whole row is zero.
                scratch.step[row] = 0.0;
            } else {
                scratch.step[row] = value / diagonal;
            }
        }
        let norm = scratch
            .step
            .iter()
            .fold(0.0_f64, |norm, &value| norm.hypot(value));
        norm.is_finite().then_some(norm)
    }

    /// Find an LM step in the scaled trust region. The Gauss–Newton step is
    /// used whenever it fits. Otherwise the step norm decreases monotonically
    /// with positive damping; a bounded bracket and geometric bisection locate
    /// the boundary to one percent, including damping far below machine epsilon.
    fn trust_region_step(
        &self,
        radius: f64,
        previous_damping: f64,
        scratch: &mut LmQrScratch,
    ) -> Result<(f64, f64), OptError> {
        if let Some(norm) = self.solve(0.0, scratch)
            && norm <= radius
        {
            return Ok((0.0, norm));
        }
        if !(radius > 0.0 && radius.is_finite()) {
            return Err(OptError::NotConverged {
                detail: String::from("LM trust radius is no longer positive and finite"),
            });
        }
        let mut lower = 0.0;
        let mut upper = if previous_damping > 0.0 && previous_damping.is_finite() {
            previous_damping
        } else {
            self.max_column_norm
        };
        if upper == 0.0 {
            return Err(OptError::NotConverged {
                detail: String::from("LM Jacobian has no resolvable descent direction"),
            });
        }
        // Multiplying/dividing by ten reaches every representable damping
        // magnitude in at most 632 iterations, even from opposite extremes.
        for _ in 0..640 {
            let norm = self.solve(upper, scratch).unwrap_or(f64::INFINITY);
            if norm / radius >= 0.99 && norm / radius <= 1.01 {
                return Ok((upper, norm));
            }
            if norm < radius {
                break;
            }
            lower = upper;
            let next = (upper * 10.0).min(f64::MAX);
            if next == upper {
                return Err(OptError::NotConverged {
                    detail: String::from("LM could not bound its regularized step"),
                });
            }
            upper = next;
        }
        if lower == 0.0 {
            for _ in 0..640 {
                let next = upper * 0.1;
                if next == 0.0 || next == upper {
                    break;
                }
                let norm = self.solve(next, scratch).unwrap_or(f64::INFINITY);
                if norm > radius {
                    lower = next;
                    break;
                }
                upper = next;
                if norm / radius >= 0.99 {
                    return Ok((upper, norm));
                }
            }
        }
        // The bracket is at most a decade wide. Geometric bisection avoids
        // overflowing its product and retains subnormal damping parameters.
        if lower > 0.0 {
            for _ in 0..64 {
                let candidate = lower.sqrt() * upper.sqrt();
                if candidate <= lower || candidate >= upper {
                    break;
                }
                let norm = self.solve(candidate, scratch).unwrap_or(f64::INFINITY);
                if norm / radius >= 0.99 && norm / radius <= 1.01 {
                    return Ok((candidate, norm));
                }
                if norm > radius {
                    lower = candidate;
                } else {
                    upper = candidate;
                }
            }
        }
        self.solve(upper, scratch)
            .filter(|&norm| norm <= radius * 1.01)
            .map(|norm| (upper, norm))
            .ok_or_else(|| OptError::NotConverged {
                detail: String::from("LM could not resolve a finite trust-region step"),
            })
    }
}

/// Compute Cholesky decomposition A = L * L^T.
fn cholesky_decompose(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut low = Vec::new();
    if cholesky_decompose_into(a, n, &mut low) {
        Some(low)
    } else {
        None
    }
}

/// Compute Cholesky decomposition A = L * L^T into fixed-shape scratch.
fn cholesky_decompose_into(a: &[Vec<f64>], n: usize, low: &mut Vec<Vec<f64>>) -> bool {
    resize_square(low, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (&low_ik, &low_jk) in low[i].iter().zip(low[j].iter()).take(j) {
                sum -= low_ik * low_jk;
            }
            if i == j {
                if sum <= 0.0 {
                    return false; // Not positive definite
                }
                low[i][j] = sum.sqrt();
            } else {
                low[i][j] = sum / low[j][j];
            }
        }
    }
    true
}

/// SciPy's curve-fit covariance: LM retains the full inverse, while TRF and
/// Dogbox truncate numerically singular directions in their pseudoinverse.
/// Neither covariance path forms the condition-squaring normal matrix.
fn curve_fit_covariance(
    result: &LeastSquaresResult,
    method: LeastSquaresMethod,
    absolute_sigma: bool,
) -> Result<Vec<Vec<f64>>, OptError> {
    let m = result.fun.len();
    let n = result.x.len();
    if !absolute_sigma && m <= n {
        return Ok(vec![vec![f64::INFINITY; n]; n]);
    }
    if !result.jac.all_finite() {
        return Ok(vec![vec![f64::INFINITY; n]; n]);
    }

    // curve_fit returns a full parameter covariance and explicitly performs a
    // dense SVD here. Existing dense Jacobians are borrowed without copying;
    // least_squares itself retains CSR storage end to end.
    let dense_storage;
    let jac = match result.jac.as_dense() {
        Some(matrix) => matrix,
        None => {
            dense_storage = result.jac.to_dense()?;
            &dense_storage
        }
    };
    if method == LeastSquaresMethod::Lm {
        return Ok(compute_covariance(jac, &result.fun, n, absolute_sigma));
    }
    // Normalize before the Jacobi SVD: its column inner products otherwise overflow
    // or underflow for large/small model derivatives, even when the covariance is finite.
    let scale = jac
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let mut covariance = vec![vec![0.0; n]; n];
    if scale != 0.0 {
        let normalized: Vec<Vec<f64>> = jac
            .iter()
            .map(|row| row.iter().map(|value| value / scale).collect())
            .collect();
        let (_, singular_values, right_vectors) = trf::svd_thin(&normalized, n);
        let threshold = f64::EPSILON * m.max(n) as f64 * singular_values[0];
        for (&singular, vector) in singular_values.iter().zip(&right_vectors) {
            if singular > threshold {
                for (i, row) in covariance.iter_mut().enumerate() {
                    for (j, entry) in row.iter_mut().enumerate() {
                        *entry += (vector[i] / singular) * (vector[j] / singular);
                    }
                }
            }
        }
        // Divide twice instead of forming scale², which can overflow for a finite J.
        for entry in covariance.iter_mut().flatten() {
            *entry /= scale;
            *entry /= scale;
        }
    }
    if covariance.iter().flatten().any(|value| value.is_nan()) {
        return Ok(vec![vec![f64::INFINITY; n]; n]);
    }
    if !absolute_sigma {
        let residual_variance = 2.0 * result.cost / (m - n) as f64;
        for entry in covariance.iter_mut().flatten() {
            *entry *= residual_variance;
        }
    }
    Ok(covariance)
}

/// Compute LM's full parameter covariance directly from its Jacobian.
///
/// Forming JᵀJ can erase a small but resolved singular direction or overflow
/// perfectly finite derivatives. The shared one-sided Jacobi SVD retains the
/// Jacobian's original scales. Unlike TRF/Dogbox covariance, LM does not discard
/// nonzero singular values using a relative pseudoinverse threshold.
fn compute_covariance(
    jac: &[Vec<f64>],
    residuals: &[f64],
    n: usize,
    absolute_sigma: bool,
) -> Vec<Vec<f64>> {
    let m = residuals.len();
    if !absolute_sigma && m <= n {
        return vec![vec![f64::INFINITY; n]; n];
    }

    let (_, singular_values, right_vectors) = trf::svd_thin(jac, n);
    if singular_values.len() != n
        || singular_values
            .iter()
            .any(|value| *value == 0.0 || !value.is_finite())
    {
        return vec![vec![f64::INFINITY; n]; n];
    }

    // Apply the residual standard deviation before taking outer products.
    // The squared cost can underflow or overflow even when this ratio, and
    // consequently the final relative covariance, are ordinary finite values.
    let residual_scale = if absolute_sigma {
        1.0
    } else {
        residuals
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value))
            / ((m - n) as f64).sqrt()
    };
    let mut pcov = vec![vec![0.0; n]; n];
    for (&singular, vector) in singular_values.iter().zip(&right_vectors) {
        let scale = residual_scale / singular;
        for (i, row) in pcov.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry += (vector[i] * scale) * (vector[j] * scale);
            }
        }
    }
    if pcov.iter().flatten().any(|value| value.is_nan()) {
        return vec![vec![f64::INFINITY; n]; n];
    }
    pcov
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsci_sparse::{SparseError, SparseResult};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn custom_weighted_loss(weights: Vec<f64>) -> LeastSquaresLoss {
        LeastSquaresLoss::new(move |z| {
            assert_eq!(z.len(), weights.len());
            [
                z.iter().zip(&weights).map(|(z, w)| z * w).collect(),
                weights.clone(),
                vec![0.0; z.len()],
            ]
        })
    }

    #[test]
    fn least_squares_custom_loss_uses_vector_values_and_exact_bounds() {
        // Zero weight suppresses an outlier; the other two observations give
        // x = (1*0 + 3*2)/(1+3) = 1.5. The constrained minimizer is x = 1.
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for bounded in [false, true] {
                let options = LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]; 3])),
                    custom_loss: Some(custom_weighted_loss(vec![1.0, 3.0, 0.0])),
                    f_scale: 0.2,
                    gtol: 1e-12,
                    ftol: 1e-12,
                    xtol: 1e-12,
                    ..LeastSquaresOptions::default()
                };
                let residuals = |x: &[f64]| vec![x[0], x[0] - 2.0, x[0] - 100.0];
                let result = if bounded {
                    least_squares_bounded(residuals, &[0.5], &[0.0], &[1.0], options)
                } else {
                    least_squares(residuals, &[0.5], options)
                }
                .unwrap();
                assert!(result.success, "{method:?}, bounded={bounded}: {result:?}");
                let expected_x = if bounded { 1.0 } else { 1.5 };
                let expected_cost = if bounded { 2.0 } else { 1.5 };
                assert!((result.x[0] - expected_x).abs() < 1e-10, "{result:?}");
                assert!((result.cost - expected_cost).abs() < 1e-10, "{result:?}");
                assert_eq!(result.active_mask, vec![if bounded { 1 } else { 0 }]);
                assert!((result.grad[0] - if bounded { -2.0 } else { 0.0 }).abs() < 1e-9);
                assert_eq!(result.fun[2], result.x[0] - 100.0);
            }
        }
    }

    #[test]
    fn least_squares_custom_loss_scales_cost_gradient_and_jacobian() {
        // rho(z)=z², C=2 gives cost=sum(f_i^4)/8,
        // gradient=sum(f_i^3)/2 and robust J_i=sqrt(3/2)*|f_i|.
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares(
                |x| vec![x[0], x[0] - 3.0],
                &[1.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]; 2])),
                    custom_loss: Some(LeastSquaresLoss::new(|z| {
                        assert_eq!(z, [0.25, 1.0]);
                        [
                            z.iter().map(|z| z * z).collect(),
                            z.iter().map(|z| 2.0 * z).collect(),
                            vec![2.0; z.len()],
                        ]
                    })),
                    f_scale: 2.0,
                    max_nfev: Some(1),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert_eq!(result.status, 0);
            assert_eq!(result.x, [1.0]);
            assert_eq!(result.fun, [1.0, -2.0]);
            assert_eq!(result.cost, 17.0 / 8.0);
            assert!((result.grad[0] + 3.5).abs() < 1e-14);
            let jac = result.jac.to_dense().unwrap();
            assert!((jac[0][0] - 1.5_f64.sqrt()).abs() < 1e-14);
            assert!((jac[1][0] - 6.0_f64.sqrt()).abs() < 1e-14);
        }
    }

    #[test]
    fn least_squares_custom_loss_preserves_finite_extreme_scale_models() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for case in 0..5 {
                let (residual, f_scale, expected_cost, expected_gradient, expected_jacobian) =
                    match case {
                        // Multiplying rho'' by two before z overflows, although
                        // rho' + 2*rho''*z is approximately three.
                        0 => (1e-154, 1.0, 2.5e-309, 1e-154, 3.0_f64.sqrt()),
                        // Dividing rho'' by C² first would underflow and erase
                        // its contribution to the local curvature.
                        1 => (1e200, 1e200, 0.0, 1e200, 3.0_f64.sqrt()),
                        // rho'/sqrt(EPS) overflows; multiplying by the tiny
                        // residual first still gives a finite local model.
                        2 => (1e-100, 1e-100, 5e-201, 1e208, f64::EPSILON.sqrt()),
                        // Doubling rho''*z overflows before cancellation with
                        // rho', although the final curvature is finite.
                        3 => (1e-100, 1e-100, 5e-201, 1e208, f64::EPSILON.sqrt()),
                        _ => (1e-100, 1e-100, 5e-201, -1e208, 1e154),
                    };
                let result = least_squares(
                    |_| vec![residual],
                    &[1.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]])),
                        custom_loss: Some(LeastSquaresLoss::new(move |z| {
                            let z = z[0];
                            let shift = z - 1.0;
                            match case {
                                0 => [vec![(5e307 * z) * z], vec![1e308 * z], vec![1e308]],
                                1 => [
                                    vec![0.5 * shift * shift + shift],
                                    vec![1.0 + shift],
                                    vec![1.0],
                                ],
                                2 => [
                                    vec![1.0 + 1e308 * shift - (2.5e307 * shift) * shift],
                                    vec![1e308 - 5e307 * shift],
                                    vec![-5e307],
                                ],
                                _ => {
                                    let first = if case == 3 { 1e308 } else { -1e308 };
                                    [
                                        vec![1.0 + first * shift - (0.5 * first * shift) * shift],
                                        vec![first - first * shift],
                                        vec![-first],
                                    ]
                                }
                            }
                        })),
                        f_scale,
                        gtol: 0.0,
                        max_nfev: Some(1),
                        ..LeastSquaresOptions::default()
                    },
                )
                .unwrap_or_else(|error| panic!("{method:?}, case={case}: {error}"));
                assert_eq!(result.status, 0, "{method:?}, case={case}: {result:?}");
                if expected_cost == 0.0 {
                    assert_eq!(result.cost, 0.0);
                } else {
                    assert!((result.cost / expected_cost - 1.0).abs() < 1e-13);
                }
                assert!((result.grad[0] / expected_gradient - 1.0).abs() < 1e-13);
                let jac = result.jac.to_dense().unwrap();
                assert!((jac[0][0] / expected_jacobian - 1.0).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn least_squares_custom_loss_preserves_finite_cost_when_unhalved_sum_overflows() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares(
                |_| vec![1e154; 2],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![0.0]; 2])),
                    custom_loss: Some(custom_weighted_loss(vec![1.0; 2])),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert!(result.success, "{method:?}: {result:?}");
            assert!((result.cost / 1e308 - 1.0).abs() < 1e-14);
            assert_eq!(result.grad, [0.0]);
        }
    }

    #[test]
    fn least_squares_custom_loss_validates_all_rows_on_every_call() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for row in 0..3 {
                for malformed_shape in [false, true] {
                    for after_initial in [false, true] {
                        let calls = Arc::new(AtomicUsize::new(0));
                        let loss_calls = Arc::clone(&calls);
                        let error = least_squares(
                            |x| vec![x[0] - 3.0],
                            &[0.0],
                            LeastSquaresOptions {
                                method: Some(method),
                                jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]])),
                                custom_loss: Some(LeastSquaresLoss::new(move |z| {
                                    let call = loss_calls.fetch_add(1, Ordering::Relaxed);
                                    let mut rho =
                                        [z.to_vec(), vec![1.0; z.len()], vec![0.0; z.len()]];
                                    if !after_initial || call > 0 {
                                        if malformed_shape {
                                            rho[row].clear();
                                        } else {
                                            rho[row][0] = f64::NAN;
                                        }
                                    }
                                    rho
                                })),
                                ..LeastSquaresOptions::default()
                            },
                        )
                        .expect_err("every custom-loss evaluation validates all three rows");
                        if malformed_shape {
                            assert!(matches!(error, OptError::InvalidArgument { .. }));
                        } else {
                            assert!(matches!(error, OptError::NonFiniteInput { .. }));
                        }
                        assert_eq!(
                            calls.load(Ordering::Relaxed),
                            if after_initial { 2 } else { 1 }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn least_squares_custom_loss_propagates_callback_errors_without_more_evaluations() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let calls = Arc::new(AtomicUsize::new(0));
            let loss_calls = Arc::clone(&calls);
            let model_calls = AtomicUsize::new(0);
            let jac_calls = Arc::new(AtomicUsize::new(0));
            let jac_counter = Arc::clone(&jac_calls);
            let error = least_squares(
                |x| {
                    model_calls.fetch_add(1, Ordering::Relaxed);
                    vec![x[0] - 3.0]
                },
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(move |_| {
                        jac_counter.fetch_add(1, Ordering::Relaxed);
                        vec![vec![1.0]]
                    })),
                    custom_loss: Some(LeastSquaresLoss::new_fallible(move |z| {
                        if loss_calls.fetch_add(1, Ordering::Relaxed) != 0 {
                            return Err(OptError::InvalidArgument {
                                detail: String::from("custom loss rejected the trial"),
                            });
                        }
                        Ok([z.to_vec(), vec![1.0; z.len()], vec![0.0; z.len()]])
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect_err("callback errors abort the solve");
            assert!(matches!(
                error,
                OptError::InvalidArgument { detail } if detail == "custom loss rejected the trial"
            ));
            assert_eq!(calls.load(Ordering::Relaxed), 2);
            assert_eq!(model_calls.load(Ordering::Relaxed), 2);
            assert_eq!(jac_calls.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn least_squares_custom_loss_rejects_conflicting_options_before_model_calls() {
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            let error = least_squares(
                |_| panic!("invalid custom-loss options must not evaluate the model"),
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    loss: if method == LeastSquaresMethod::Lm {
                        LossKind::Linear
                    } else {
                        LossKind::Cauchy
                    },
                    custom_loss: Some(LeastSquaresLoss::new(|_| {
                        panic!("invalid custom-loss options must not evaluate the loss")
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect_err("ambiguous losses and non-linear LM losses are unsupported");
            assert!(matches!(error, OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn least_squares_custom_loss_rejects_unrepresentable_normalization() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for scale in [1e-200, 1e200] {
                let error = least_squares(
                    |_| vec![1.0],
                    &[0.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new(|_| vec![vec![0.0]])),
                        custom_loss: Some(LeastSquaresLoss::new(|_| {
                            panic!("rounded zero/infinity must not reach a custom loss")
                        })),
                        f_scale: scale,
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect_err("a rounded custom-loss input cannot certify convergence");
                assert!(matches!(error, OptError::NonFiniteInput { .. }));
            }
        }
    }

    #[test]
    fn least_squares_custom_loss_accepts_empty_residuals() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares(
                |_| vec![],
                &[1.0, 2.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![])),
                    custom_loss: Some(custom_weighted_loss(vec![])),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert!(result.success);
            assert_eq!(result.x, [1.0, 2.0]);
            assert_eq!(result.grad, [0.0, 0.0]);
            assert_eq!(result.cost, 0.0);
        }
    }

    #[test]
    fn curve_fit_and_batched_least_squares_preserve_custom_loss() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let fit = curve_fit_bounded(
                |_, p| p[0],
                &[0.0, 1.0, 2.0],
                &[1.0, 3.0, 9.0],
                &[0.0],
                &[2.0],
                CurveFitOptions {
                    p0: Some(vec![0.5]),
                    sigma: Some(CurveFitSigma::StandardDeviations(vec![1.0, 2.0, 1.0])),
                    absolute_sigma: true,
                    ls_options: LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]; 3])),
                        custom_loss: Some(custom_weighted_loss(vec![4.0, 2.0, 0.0])),
                        gtol: 1e-12,
                        ftol: 1e-12,
                        xtol: 1e-12,
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                },
            )
            .unwrap();
            // Whitening changes the physical weights to [4, 1/2, 0].
            assert!((fit.popt[0] - 11.0 / 9.0).abs() < 1e-9, "{fit:?}");
            assert!((fit.ls_result.cost - 8.0 / 9.0).abs() < 1e-9);
            assert!((fit.pcov[0][0] - 2.0 / 9.0).abs() < 1e-9);

            let rows = [vec![0.0, 2.0, 100.0], vec![10.0, 12.0, -100.0]];
            let fitted = least_squares_many(
                |p, observations| observations.iter().map(|y| p[0] - y).collect(),
                &[0.5],
                &rows,
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]; 3])),
                    custom_loss: Some(custom_weighted_loss(vec![1.0, 3.0, 0.0])),
                    ..LeastSquaresOptions::default()
                },
            );
            assert_eq!(fitted.len(), 2);
            for (result, expected) in fitted.into_iter().zip([1.5, 11.5]) {
                let result = result.unwrap();
                assert!(result.success, "{result:?}");
                assert!((result.x[0] - expected).abs() < 1e-8, "{result:?}");
            }
        }
    }

    #[test]
    fn least_squares_callback_stops_with_current_bounded_robust_state() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for sparse in [false, true] {
                let calls = Arc::new(AtomicUsize::new(0));
                let jac_calls = Arc::new(AtomicUsize::new(0));
                let snapshot = Arc::new(Mutex::new(None));
                let callback_calls = Arc::clone(&calls);
                let callback_jac_calls = Arc::clone(&jac_calls);
                let callback_snapshot = Arc::clone(&snapshot);
                let jac_counter = Arc::clone(&jac_calls);
                let jac = if sparse {
                    LeastSquaresJacobian::new_sparse(move |_| {
                        jac_counter.fetch_add(1, Ordering::Relaxed);
                        CsrMatrix::from_components(
                            Shape2D::new(2, 2),
                            vec![1.0, 1.0],
                            vec![0, 1],
                            vec![0, 1, 2],
                            true,
                        )
                        .unwrap()
                    })
                } else {
                    LeastSquaresJacobian::new(move |_| {
                        jac_counter.fetch_add(1, Ordering::Relaxed);
                        vec![vec![1.0, 0.0], vec![0.0, 1.0]]
                    })
                };
                let result = least_squares_bounded(
                    |x| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        vec![x[0] - 5.0, x[1] + 3.0]
                    },
                    &[0.0, 0.0],
                    &[-2.0, -1.0],
                    &[3.0, 2.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(jac),
                        loss: LossKind::SoftL1,
                        f_scale: 0.5,
                        callback: Some(LeastSquaresCallback::new(move |state| {
                            assert_eq!(state.nit, 1);
                            assert_eq!(state.fun, [state.x[0] - 5.0, state.x[1] + 3.0]);
                            let expected_cost: f64 = state
                                .fun
                                .iter()
                                .map(|f| 0.25 * ((1.0 + 4.0 * f * f).sqrt() - 1.0))
                                .sum();
                            assert!((state.cost - expected_cost).abs() < 1.0e-14);
                            assert_eq!(state.nfev, callback_calls.load(Ordering::Relaxed));
                            assert_eq!(state.njev, callback_jac_calls.load(Ordering::Relaxed));
                            *callback_snapshot.lock().unwrap() = Some((
                                state.x.to_vec(),
                                state.fun.to_vec(),
                                state.cost,
                                state.nfev,
                                state.njev,
                            ));
                            ControlFlow::Break(())
                        })),
                        ..LeastSquaresOptions::default()
                    },
                )
                .unwrap();
                let (x, fun, cost, nfev, njev) = snapshot.lock().unwrap().clone().unwrap();
                assert_eq!(result.status, -2);
                assert!(!result.success);
                assert!(result.message.contains("callback"));
                assert_eq!(result.nit, 1);
                assert_eq!(result.x, x);
                assert_eq!(result.fun, fun);
                assert_eq!(result.cost, cost);
                assert_eq!(result.nfev, nfev);
                assert_eq!(result.njev, njev);
                // There must be no residual or Jacobian call after cancellation.
                assert_eq!(calls.load(Ordering::Relaxed), nfev);
                assert_eq!(jac_calls.load(Ordering::Relaxed), njev);
                assert_eq!(result.jac.as_sparse().is_some(), sparse);
                assert!(result.optimality.is_finite());
            }
        }
    }

    #[test]
    fn least_squares_callback_continue_preserves_results_and_evaluations() {
        let rosenbrock = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let options = LeastSquaresOptions {
                method: Some(method),
                ..LeastSquaresOptions::default()
            };
            let baseline = least_squares(rosenbrock, &[-1.2, 1.0], options.clone()).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let callback_calls = Arc::clone(&calls);
            let monitored = least_squares(
                rosenbrock,
                &[-1.2, 1.0],
                LeastSquaresOptions {
                    callback: Some(LeastSquaresCallback::new(move |state| {
                        let previous = callback_calls.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(state.nit, previous + 1);
                        assert_eq!(state.fun, rosenbrock(state.x));
                        ControlFlow::Continue(())
                    })),
                    ..options
                },
            )
            .unwrap();
            assert_eq!(monitored, baseline);
            assert_eq!(calls.load(Ordering::Relaxed), monitored.nit);
            assert!(monitored.nit > 1);
        }
    }

    #[test]
    fn least_squares_callback_stop_has_priority_over_convergence() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares(
                |x| vec![x[0] - 1.0],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]])),
                    callback: Some(LeastSquaresCallback::new(|state| {
                        assert_eq!(state.x, [1.0]);
                        ControlFlow::Break(())
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert_eq!(result.status, -2);
            assert!(!result.success);
            assert_eq!(result.nit, 1);
            assert_eq!(result.optimality, 0.0);
            assert_eq!(result.nfev, 2);
        }
    }

    #[test]
    fn least_squares_callback_is_not_called_without_an_iteration() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for start in [0.0, 1.0] {
                let result = least_squares(
                    |x| vec![x[0] - 1.0],
                    &[start],
                    LeastSquaresOptions {
                        method: Some(method),
                        max_nfev: Some(1),
                        callback: Some(LeastSquaresCallback::new(|_| {
                            panic!("the solver has not taken an iteration")
                        })),
                        ..LeastSquaresOptions::default()
                    },
                )
                .unwrap();
                assert_eq!(result.nit, 0);
                assert_eq!(result.nfev, 1);
                assert_eq!(result.status, if start == 1.0 { 1 } else { 0 });
            }
        }
        let error = least_squares(
            |_| panic!("unsupported callbacks must fail before evaluating a model"),
            &[0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                callback: Some(LeastSquaresCallback::new(|_| ControlFlow::Break(()))),
                ..LeastSquaresOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_callback_is_shared_safely_by_batched_solves() {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let targets: Vec<_> = (2..10).map(|i| vec![f64::from(i)]).collect();
        let results = least_squares_many(
            |x, target| vec![x[0] - target[0]],
            &[0.0],
            &targets,
            LeastSquaresOptions {
                callback: Some(LeastSquaresCallback::new(move |state| {
                    assert_eq!(state.nit, 1);
                    callback_calls.fetch_add(1, Ordering::Relaxed);
                    ControlFlow::Break(())
                })),
                ..LeastSquaresOptions::default()
            },
        );
        assert_eq!(calls.load(Ordering::Relaxed), targets.len());
        assert!(
            results
                .into_iter()
                .all(|result| result.unwrap().status == -2)
        );
    }

    #[test]
    fn least_squares_callback_stop_propagates_through_curve_fit() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let error = curve_fit_bounded(
                |x, p| p[0] * x,
                &[1.0, 2.0, 3.0],
                &[10.0, 20.0, 30.0],
                &[0.0],
                &[20.0],
                CurveFitOptions {
                    p0: Some(vec![1.0]),
                    ls_options: LeastSquaresOptions {
                        method: Some(method),
                        callback: Some(LeastSquaresCallback::new(|_| ControlFlow::Break(()))),
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                },
            )
            .unwrap_err();
            assert!(matches!(error, OptError::NotConverged { .. }));
            assert!(error.to_string().contains("callback"));
        }
    }

    struct IdentityJacobianProbe {
        dimension: usize,
        products: Arc<AtomicUsize>,
        basis_probes: Arc<AtomicUsize>,
    }

    impl LinearOperator for IdentityJacobianProbe {
        fn shape(&self) -> Shape2D {
            Shape2D::new(self.dimension, self.dimension)
        }

        fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            if x.len() != self.dimension || y.len() != self.dimension {
                return Err(SparseError::IncompatibleShape {
                    message: String::from("identity probe dimensions"),
                });
            }
            self.products.fetch_add(1, Ordering::Relaxed);
            if x.iter().filter(|&&value| value != 0.0).count() == 1 && x.contains(&1.0) {
                self.basis_probes.fetch_add(1, Ordering::Relaxed);
            }
            y.copy_from_slice(x);
            Ok(())
        }

        fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            self.matvec_into(x, y)
        }
    }

    fn identity_operator_jacobian(n: usize) -> LeastSquaresJacobian {
        LeastSquaresJacobian::new_operator(move |_| IdentityJacobianProbe {
            dimension: n,
            products: Arc::new(AtomicUsize::new(0)),
            basis_probes: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[test]
    fn least_squares_callback_preserves_matrix_free_storage() {
        let n = 128;
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let products = Arc::new(AtomicUsize::new(0));
            let probes = Arc::new(AtomicUsize::new(0));
            let product_counter = Arc::clone(&products);
            let probe_counter = Arc::clone(&probes);
            let result = least_squares_bounded(
                |x| x.iter().map(|value| value - 3.0).collect(),
                &vec![0.0; n],
                &vec![-1.0; n],
                &vec![4.0; n],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new_operator(move |_| {
                        IdentityJacobianProbe {
                            dimension: n,
                            products: Arc::clone(&product_counter),
                            basis_probes: Arc::clone(&probe_counter),
                        }
                    })),
                    callback: Some(LeastSquaresCallback::new(move |state| {
                        assert_eq!(state.x.len(), n);
                        assert!(state.fun.iter().zip(state.x).all(|(f, x)| *f == x - 3.0));
                        ControlFlow::Break(())
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .unwrap();
            assert_eq!(result.status, -2);
            assert!(result.jac.is_operator());
            assert_eq!(probes.load(Ordering::Relaxed), 0);
            assert!(products.load(Ordering::Relaxed) > 0);
        }
    }

    #[test]
    fn matrix_free_least_squares_never_materializes_or_probes_entries() {
        // A dense conversion takes n basis applications; the iterative solution takes
        // only a small, dimension-independent number of full-vector applications.
        let n = 1024;
        let target: Vec<_> = (0..n).map(|i| 1.0 + i as f64 / n as f64).collect();
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let products = Arc::new(AtomicUsize::new(0));
            let basis_probes = Arc::new(AtomicUsize::new(0));
            let product_counter = Arc::clone(&products);
            let basis_counter = Arc::clone(&basis_probes);
            let result = least_squares(
                |x| x.iter().zip(&target).map(|(a, b)| a - b).collect(),
                &vec![0.0; n],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new_operator(move |_| {
                        IdentityJacobianProbe {
                            dimension: n,
                            products: Arc::clone(&product_counter),
                            basis_probes: Arc::clone(&basis_counter),
                        }
                    })),
                    gtol: 1e-12,
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("matrix-free default selects LSMR");
            assert!(result.success, "{method:?}: {}", result.message);
            assert!(
                result
                    .x
                    .iter()
                    .zip(&target)
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            );
            assert!(result.jac.as_operator().is_some());
            assert_eq!(basis_probes.load(Ordering::Relaxed), 0);
            let count = products.load(Ordering::Relaxed);
            assert!(
                (1..128).contains(&count),
                "{method:?}: {count} products for {n} columns"
            );
        }
    }

    #[test]
    fn matrix_free_jacobian_is_retained_in_parallel_batch_results() {
        let targets = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, -1.0, 0.5],
            vec![-2.0, 3.0, 4.0],
            vec![0.5, 1.0, 2.0],
        ];
        let results = least_squares_many(
            |x, target| x.iter().zip(target).map(|(a, b)| a - b).collect(),
            &[0.0; 3],
            &targets,
            LeastSquaresOptions {
                jac: Some(identity_operator_jacobian(3)),
                ..LeastSquaresOptions::default()
            },
        );
        for (result, target) in results.into_iter().zip(targets) {
            let result = result.expect("operator result transfers from worker");
            assert!(result.success);
            assert!(
                result
                    .x
                    .iter()
                    .zip(target)
                    .all(|(a, b)| (a - b).abs() < 1e-8)
            );
            assert_eq!(
                result
                    .jac
                    .as_operator()
                    .unwrap()
                    .matvec(&[1.0, -2.0, 0.5])
                    .unwrap(),
                vec![1.0, -2.0, 0.5]
            );
        }
    }

    #[test]
    fn matrix_free_least_squares_rejects_entry_dependent_options_and_wrong_shapes() {
        for options in [
            LeastSquaresOptions {
                tr_solver: TrustRegionSolver::Exact,
                ..LeastSquaresOptions::default()
            },
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                ..LeastSquaresOptions::default()
            },
            LeastSquaresOptions {
                x_scale: Some(XScale::Jac),
                ..LeastSquaresOptions::default()
            },
        ] {
            let result = least_squares(
                |x| vec![x[0] - 1.0],
                &[0.0],
                LeastSquaresOptions {
                    jac: Some(identity_operator_jacobian(1)),
                    ..options
                },
            );
            assert!(
                matches!(result, Err(OptError::InvalidArgument { .. })),
                "{result:?}"
            );
        }
        let wrong_shape = least_squares(
            |x| vec![x[0] - 1.0],
            &[0.0],
            LeastSquaresOptions {
                jac: Some(identity_operator_jacobian(2)),
                ..LeastSquaresOptions::default()
            },
        );
        assert!(matches!(wrong_shape, Err(OptError::InvalidArgument { .. })));

        for sigma in [None, Some(CurveFitSigma::Scalar(2.0))] {
            let fitted = curve_fit(
                |_t, p| p[0],
                &[0.0, 1.0],
                &[1.0, 2.0],
                CurveFitOptions {
                    p0: Some(vec![0.0]),
                    sigma,
                    ls_options: LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Trf),
                        jac: Some(identity_operator_jacobian(1)),
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                },
            );
            assert!(
                matches!(fitted, Err(OptError::InvalidArgument { detail }) if detail.contains("covariance"))
            );
        }
    }

    struct ForwardOnlyJacobian;

    impl LinearOperator for ForwardOnlyJacobian {
        fn shape(&self) -> Shape2D {
            Shape2D::new(1, 1)
        }

        fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    struct FaultyJacobian {
        failure: u8,
    }

    impl LinearOperator for FaultyJacobian {
        fn shape(&self) -> Shape2D {
            Shape2D::new(1, 1)
        }

        fn matvec_into(&self, _x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            match self.failure {
                0 => Err(SparseError::Unsupported {
                    feature: String::from("user product failure"),
                }),
                1 => {
                    y[0] = f64::INFINITY;
                    Ok(())
                }
                _ => Ok(()), // Deliberately fail to write the output.
            }
        }

        fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            y.copy_from_slice(x);
            Ok(())
        }
    }

    #[test]
    fn matrix_free_products_fail_closed_on_callback_errors_and_incomplete_outputs() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let no_transpose = least_squares(
                |x| vec![x[0] - 1.0],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new_operator(|_| ForwardOnlyJacobian)),
                    ..LeastSquaresOptions::default()
                },
            );
            assert!(
                matches!(no_transpose, Err(OptError::NotConverged { detail }) if detail.contains("rmatvec"))
            );
            for failure in 0..3 {
                let result = least_squares(
                    |x| vec![x[0] - 1.0],
                    &[0.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new_operator(move |_| {
                            FaultyJacobian { failure }
                        })),
                        ..LeastSquaresOptions::default()
                    },
                );
                match failure {
                    0 => assert!(
                        matches!(result, Err(OptError::NotConverged { detail }) if detail.contains("user product failure"))
                    ),
                    _ => assert!(
                        matches!(result, Err(OptError::NonFiniteInput { .. })),
                        "{method:?}/{failure}: {result:?}"
                    ),
                }
            }
        }
    }

    #[test]
    fn parallel_fd_jacobian_byte_identical_to_serial() {
        // Exercise the parallel path (m >= 8192, n >= 4 clears the work-gate) and
        // prove it produces a bit-for-bit identical Jacobian to the serial routine.
        let n = 16usize;
        let m = 9000usize;
        let params: Vec<f64> = (0..n).map(|k| 0.3 + 0.13 * k as f64).collect();
        let xs: Vec<f64> = (0..m).map(|i| i as f64 * 0.001).collect();
        let residuals = move |p: &[f64]| -> Vec<f64> {
            xs.iter()
                .map(|&x| {
                    p.iter()
                        .enumerate()
                        .map(|(k, &c)| c * ((k as f64 + 1.0) * x).sin())
                        .sum()
                })
                .collect()
        };
        let r0 = residuals(&params);
        let mut jac_serial = Vec::new();
        let mut scratch_s = Vec::new();
        finite_diff_jacobian_into(
            &residuals,
            &params,
            &r0,
            None,
            &mut jac_serial,
            &mut scratch_s,
        )
        .expect("serial Jacobian");

        let mut jac_par = Vec::new();
        let mut scratch_p = Vec::new();
        finite_diff_jacobian_parallel_into(
            &residuals,
            &params,
            &r0,
            None,
            &mut jac_par,
            &mut scratch_p,
        )
        .expect("parallel Jacobian");

        assert_eq!(jac_serial.len(), jac_par.len());
        for (rs, rp) in jac_serial.iter().zip(&jac_par) {
            assert_eq!(rs.len(), rp.len());
            for (a, b) in rs.iter().zip(rp) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "parallel Jacobian must be byte-identical"
                );
            }
        }
    }

    #[test]
    fn finite_difference_jacobian_uses_actual_displacement() {
        let point = [1.0];
        let residual = |x: &[f64]| x.to_vec();
        let mut jac = Vec::new();
        finite_diff_jacobian_into(
            &residual,
            &point,
            &point,
            Some(0.75 * f64::EPSILON),
            &mut jac,
            &mut Vec::new(),
        )
        .expect("a rounded but nonzero displacement is valid");
        // The derivative of the identity is exactly one. Dividing by the
        // requested step instead of the rounded displacement gives 4/3 here.
        assert_eq!(jac, vec![vec![1.0]]);
    }

    #[test]
    fn parallel_fd_jacobian_cannot_erase_sub_ulp_perturbations() {
        const N: usize = 16;
        const M: usize = 8192;
        const { assert!(N >= 4 && M >= 8192 && N * M >= 131_072) };
        let point: Vec<f64> = (0..N)
            .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let residual = |x: &[f64]| (0..M).map(|i| x[i % N]).collect::<Vec<_>>();
        let mut jac = Vec::new();
        finite_diff_jacobian_parallel_into(
            &residual,
            &point,
            &residual(&point),
            Some(1e-18),
            &mut jac,
            &mut Vec::new(),
        )
        .expect("parallel perturbations retain each identity column");
        for (i, row) in jac.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                assert_eq!(value, if i % N == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn least_squares_linear_fit() {
        // Fit y = a*x + b to data
        let xdata = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ydata = [2.1, 3.9, 6.1, 7.9, 10.1]; // approx y = 2x + 0.1

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * x + p[1] - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 0.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.1,
            "slope ~ 2.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.1).abs() < 0.2,
            "intercept ~ 0.1, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_exponential_decay() {
        // Fit y = a * exp(-b * x)
        let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 3.0 * (-0.5 * x).exp()).collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-p[1] * x).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[1.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 1.0e-4,
            "a ~ 3.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.5).abs() < 1.0e-4,
            "b ~ 0.5, got {}",
            result.x[1]
        );
    }

    #[test]
    fn least_squares_rejects_empty_x0() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![] };
        let err = least_squares(residuals, &[], LeastSquaresOptions::default())
            .expect_err("should reject empty x0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    // SciPy: method='lm' refuses m < n; the default 'trf' accepts it (a constant residual has
    // zero gradient: status 1, x unchanged).
    #[test]
    fn least_squares_lm_rejects_underdetermined_and_trf_accepts_it() {
        let residuals = |_: &[f64]| -> Vec<f64> { vec![1.0] };
        let lm = LeastSquaresOptions {
            method: Some(LeastSquaresMethod::Lm),
            ..LeastSquaresOptions::default()
        };
        let err = least_squares(residuals, &[1.0, 2.0, 3.0], lm)
            .expect_err("lm should reject underdetermined");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
        let trf = least_squares(residuals, &[1.0, 2.0, 3.0], LeastSquaresOptions::default())
            .expect("trf accepts m < n");
        assert_eq!((trf.status, trf.x.clone()), (1, vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn least_squares_lm_does_not_converge_just_because_residual_units_are_small() {
        // The unique solution is x = 1 in every case. The last two magnitudes
        // also underflow the unscaled sum of squared residuals to zero.
        for magnitude in [1.0, 1.0e-6, 1.0e-12, 1.0e-30, 1.0e-200, 1.0e-300] {
            for derivative in 0..3 {
                let result = least_squares(
                    |x| vec![magnitude * (x[0] - 1.0)],
                    &[0.0],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        diff_method: if derivative == 2 {
                            FiniteDiffMethod::ThreePoint
                        } else {
                            FiniteDiffMethod::TwoPoint
                        },
                        jac: if derivative == 0 {
                            Some(LeastSquaresJacobian::new(move |_| vec![vec![magnitude]]))
                        } else {
                            None
                        },
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("scaled linear least squares");
                assert!(
                    result.success,
                    "magnitude={magnitude}, derivative={derivative}: {result:?}"
                );
                assert!(
                    (result.x[0] - 1.0).abs() < 1e-10,
                    "magnitude={magnitude}, derivative={derivative}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn least_squares_lm_difference_step_cannot_erase_the_jacobian() {
        // Each requested perturbation rounds back to the starting coordinate.
        // Treating that as a zero derivative falsely reported gtol success at x0.
        for start in [-1.0, 0.0, 1.0] {
            for diff_step in [1.0e-18, 1.0e-300] {
                let result = least_squares(
                    |x| vec![x[0] - 3.0],
                    &[start],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        diff_step: Some(diff_step),
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("a sub-ULP requested step uses the default representable step");
                assert!(
                    result.success,
                    "start={start}, step={diff_step}: {result:?}"
                );
                assert!((result.x[0] - 3.0).abs() < 1e-10, "{result:?}");
                assert!(result.fun[0].abs() < 1e-10, "{result:?}");
                assert_eq!(result.jac.to_dense().unwrap(), vec![vec![1.0]]);
            }
        }
    }

    #[test]
    fn least_squares_lm_difference_step_respects_small_parameter_units() {
        // An explicitly requested relative step scales with |x|, not 1 + |x|.
        // Adding 1e-4 to a 1e-12 parameter samples an entirely different regime.
        for sign in [-1.0, 1.0] {
            let result = least_squares(
                |x| vec![(x[0] / 1e-12).powi(2) - 4.0],
                &[sign * 1e-12],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    diff_step: Some(1e-4),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("relative finite differences preserve parameter units");
            assert!(result.success, "{result:?}");
            assert!(
                (result.x[0] / 1e-12 - sign * 2.0).abs() < 1e-8,
                "{result:?}"
            );
            assert!(result.fun[0].abs() < 1e-7, "{result:?}");
        }
    }

    #[test]
    fn least_squares_lm_scaled_nonlinear_residuals_keep_iterating() {
        for magnitude in [1.0, 1.0e-12, 1.0e-200] {
            let result = least_squares(
                |x| vec![magnitude * (x[0] * x[0] - 2.0)],
                &[1.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    jac: Some(LeastSquaresJacobian::new(move |x| {
                        vec![vec![2.0 * magnitude * x[0]]]
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("scaled nonlinear least squares");
            assert!(result.success, "magnitude={magnitude}: {result:?}");
            assert!(
                (result.x[0] - 2.0_f64.sqrt()).abs() < 1e-8,
                "magnitude={magnitude}: {result:?}"
            );
        }
    }

    #[test]
    fn least_squares_lm_resolves_correlated_parameters_without_normal_equations() {
        // J is full rank, but forming JᵀJ rounds its last diagonal to 1
        // for the smaller weak directions. The former Cholesky/diagonal
        // fallback returned success near [0.5, 0.5] instead of [0, 1].
        for weak in [1.0e-6, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-12] {
            for magnitude in [1.0e-200, 1.0e-120, 1.0, 1.0e100, 1.0e200] {
                let result = least_squares(
                    move |x| {
                        vec![
                            magnitude * (x[0] + x[1] - 1.0),
                            magnitude * weak * (x[1] - 1.0),
                        ]
                    },
                    &[0.0, 0.0],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        jac: Some(LeastSquaresJacobian::new(move |_| {
                            vec![vec![magnitude, magnitude], vec![0.0, magnitude * weak]]
                        })),
                        ftol: 1.0e-14,
                        xtol: 1.0e-14,
                        gtol: 1.0e-14,
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("identifiable linear model");
                assert!(
                    result.success,
                    "weak={weak}, magnitude={magnitude}: {result:?}"
                );
                assert!(
                    result.x[0].abs() < 1.0e-8 && (result.x[1] - 1.0).abs() < 1.0e-8,
                    "weak={weak}, magnitude={magnitude}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn least_squares_lm_initial_radius_preserves_large_residual_units() {
        for magnitude in [1.0e100, 1.0e200] {
            for start in [0.0, 1.0e-100] {
                let result = least_squares(
                    move |x| vec![magnitude * (x[0] - 1.0)],
                    &[start],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        jac: Some(LeastSquaresJacobian::new(move |_| vec![vec![magnitude]])),
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("the initial trust radius respects scaled residual units");
                assert!(
                    result.success && (result.x[0] - 1.0).abs() < 1.0e-12,
                    "magnitude={magnitude}, start={start}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn least_squares_lm_fixed_scaling_does_not_square_jacobian_units() {
        // With fixed x_scale, adaptive column normalization cannot hide an
        // overflow or underflow caused by squaring the physical Jacobian.
        for magnitude in [1.0e-200, 1.0e-100, 1.0, 1.0e100, 1.0e200] {
            let result = least_squares(
                move |x| vec![magnitude * (x[0] - 1.0)],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    x_scale: Some(XScale::Fixed(vec![1.0])),
                    jac: Some(LeastSquaresJacobian::new(move |_| vec![vec![magnitude]])),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("finite unsquared least-squares system");
            assert!(result.success, "magnitude={magnitude}: {result:?}");
            assert!((result.x[0] - 1.0).abs() < 1.0e-12, "{result:?}");
        }
    }

    #[test]
    fn least_squares_lm_resolves_rotated_and_permuted_weak_directions() {
        for weak in [1.0e-8, 1.0e-10] {
            for weak_column in [0, 1] {
                let result = least_squares(
                    move |x| {
                        let a = x[0] + x[1] - 1.0;
                        let b = weak * (x[weak_column] - 1.0);
                        vec![
                            (a - b) * std::f64::consts::FRAC_1_SQRT_2,
                            (a + b) * std::f64::consts::FRAC_1_SQRT_2,
                        ]
                    },
                    &[0.0, 0.0],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        jac: Some(LeastSquaresJacobian::new(move |_| {
                            let mut rows = vec![vec![1.0; 2]; 2];
                            rows[0][weak_column] -= weak;
                            rows[1][weak_column] += weak;
                            for entry in rows.iter_mut().flatten() {
                                *entry *= std::f64::consts::FRAC_1_SQRT_2;
                            }
                            rows
                        })),
                        ftol: 1.0e-14,
                        xtol: 1.0e-14,
                        gtol: 1.0e-14,
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("rotating residuals and permuting parameters preserves identifiability");
                assert!(result.success, "{result:?}");
                assert!((result.x[weak_column] - 1.0).abs() < 1.0e-8, "{result:?}");
                assert!(result.x[1 - weak_column].abs() < 1.0e-8, "{result:?}");
            }
        }
    }

    #[test]
    fn least_squares_lm_trust_radius_recovers_after_rejected_nonlinear_steps() {
        let result = least_squares(
            |x| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
            &[-1.2, 1.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|x| {
                    vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]]
                })),
                ftol: 1.0e-12,
                xtol: 1.0e-12,
                gtol: 1.0e-12,
                ..LeastSquaresOptions::default()
            },
        )
        .expect("Rosenbrock residuals need a bounded step after the first rejected trial");
        assert!(result.success, "{result:?}");
        assert!(
            result.x.iter().all(|x| (x - 1.0).abs() < 1.0e-9),
            "{result:?}"
        );
        assert!(result.cost < 1.0e-20, "{result:?}");
        assert!(
            result.nfev > result.njev,
            "the rejected step must be counted"
        );
    }

    #[test]
    fn least_squares_lm_rank_deficient_model_keeps_full_least_squares_objective() {
        let result = least_squares(
            |x| vec![x[0] + x[1] - 1.0, x[0] + x[1] - 3.0],
            &[0.0, 0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0, 1.0]; 2])),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("a rank-deficient model still has a finite least-squares minimum");
        assert!(result.success, "{result:?}");
        assert!(
            (result.x.iter().sum::<f64>() - 2.0).abs() < 1.0e-9,
            "{result:?}"
        );
        assert!((result.cost - 1.0).abs() < 1.0e-12, "{result:?}");

        let zero_column = least_squares(
            |x| vec![x[0] - 2.0, 1.0, -1.0],
            &[0.0, 7.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|_| {
                    vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]
                })),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("an unused parameter remains at its initial value");
        assert!(zero_column.success, "{zero_column:?}");
        assert_eq!(zero_column.x, [2.0, 7.0]);
        assert_eq!(zero_column.cost, 1.0);
    }

    #[test]
    fn least_squares_lm_rejects_nonfinite_trials_and_recovers() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let invalid_trials = AtomicUsize::new(0);
        let result = least_squares(
            |x| {
                if x[0] <= 0.0 {
                    invalid_trials.fetch_add(1, Ordering::Relaxed);
                }
                vec![x[0].ln() + 10.0]
            },
            &[1.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|x| vec![vec![1.0 / x[0]]])),
                ftol: 1.0e-12,
                xtol: 1.0e-12,
                gtol: 1.0e-12,
                ..LeastSquaresOptions::default()
            },
        )
        .expect("the trust radius shrinks before retrying outside the model domain");
        assert!(invalid_trials.load(Ordering::Relaxed) > 0);
        assert!(result.success, "{result:?}");
        assert!(
            (result.x[0] / (-10.0_f64).exp() - 1.0).abs() < 1.0e-9,
            "{result:?}"
        );
        assert!(result.cost < 1.0e-20, "{result:?}");

        let stationary = least_squares(
            |x| vec![x[0] * x[0] - 1.0],
            &[0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|x| vec![vec![2.0 * x[0]]])),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("LM may stop at a stationary point with a nonzero residual, as SciPy does");
        assert_eq!(stationary.status, 1);
        assert_eq!(stationary.x, [0.0]);
        assert_eq!(stationary.cost, 0.5);
    }

    #[test]
    fn least_squares_lm_rejects_nonfinite_augmented_factor() {
        let fit = |scale, mode| {
            least_squares(
                |x| {
                    let z = 1.0e308 * x[0];
                    vec![1.0 + z + 100.0 * z * z]
                },
                &[0.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    jac: Some(LeastSquaresJacobian::new(|x| {
                        let z = 1.0e308 * x[0];
                        vec![vec![1.0e308 * (1.0 + 200.0 * z)]]
                    })),
                    x_scale: Some(scale),
                    mode,
                    ..LeastSquaresOptions::default()
                },
            )
        };
        for mode in [RuntimeMode::Strict, RuntimeMode::Hardened] {
            // The first Gauss–Newton trial is rejected. Fixed parameter units
            // then require damping beyond the finite range; an infinite QR
            // diagonal must not masquerade as a converged zero step at x0.
            let error = fit(XScale::Fixed(vec![1.0]), mode)
                .expect_err("nonfinite augmentation cannot establish convergence");
            assert!(matches!(error, OptError::NotConverged { .. }), "{error}");

            // The same finite model remains solvable with Jacobian scaling.
            // Its exact minimizer is z = -1/200 and residual is 399/400.
            let scaled = fit(XScale::Jac, mode).expect("scaled damping stays finite");
            assert!(scaled.success, "{scaled:?}");
            assert!((1.0e308 * scaled.x[0] + 0.005).abs() < 1.0e-7);
            assert!((scaled.cost - 0.497_503_125).abs() < 1.0e-12);
        }

        // Checking factors must not normalize away an independently resolved
        // weak column when the undamped step is already representable.
        let mixed = least_squares(
            |x| vec![1.0e308 * (x[0] - 1.0), 1.0e-120 * (x[1] - 2.0)],
            &[0.0, 0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                jac: Some(LeastSquaresJacobian::new(|_| {
                    vec![vec![1.0e308, 0.0], vec![0.0, 1.0e-120]]
                })),
                x_scale: Some(XScale::Fixed(vec![1.0, 1.0])),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("finite mixed-scale directions remain resolved");
        assert!(mixed.success, "{mixed:?}");
        assert_eq!(mixed.x, [1.0, 2.0]);
        assert_eq!(mixed.cost, 0.0);
    }

    #[test]
    fn least_squares_lm_handles_parameters_with_different_units() {
        for x_scale in [
            None,
            Some(XScale::Jac),
            Some(XScale::Fixed(vec![1.0e6, 1.0e-6])),
        ] {
            let result = least_squares(
                |x| {
                    let a = x[0] * 1.0e-6 - 1.0;
                    let b = x[1] * 1.0e6 - 2.0;
                    vec![a + b, 2.0 * a - b, a - 2.0 * b]
                },
                &[0.0, 0.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    x_scale,
                    jac: Some(LeastSquaresJacobian::new(|_| {
                        vec![
                            vec![1.0e-6, 1.0e6],
                            vec![2.0e-6, -1.0e6],
                            vec![1.0e-6, -2.0e6],
                        ]
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("LM accepts fixed and adaptive parameter scales");
            assert!(result.success, "{result:?}");
            assert!((result.x[0] * 1.0e-6 - 1.0).abs() < 1e-9, "{result:?}");
            assert!((result.x[1] * 1.0e6 - 2.0).abs() < 1e-9, "{result:?}");
        }
    }

    #[test]
    fn least_squares_lm_evaluation_budget_excludes_numerical_jacobian_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            for max_nfev in [1, 2, 3] {
                let calls = AtomicUsize::new(0);
                let result = least_squares(
                    |x| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        vec![x[0] * x[0] - 2.0]
                    },
                    &[0.1],
                    LeastSquaresOptions {
                        method: Some(LeastSquaresMethod::Lm),
                        max_nfev: Some(max_nfev),
                        diff_method,
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("a spent budget is a result with status zero");
                // MINPACK checks maxfev after its first trial: a budget of one
                // still reports two evaluations unless x0 already converges.
                assert_eq!(result.nfev, max_nfev.max(2), "{result:?}");
                assert_eq!(result.status, 0, "{result:?}");
                let stencil_calls = if diff_method == FiniteDiffMethod::ThreePoint {
                    2
                } else {
                    1
                };
                assert_eq!(
                    calls.load(Ordering::Relaxed),
                    result.nfev + stencil_calls * result.njev
                );
            }
        }
    }

    #[test]
    fn least_squares_lm_validates_tolerances_and_budget_before_callbacks() {
        for invalid in [0.0, f64::EPSILON / 2.0, f64::NAN, f64::INFINITY] {
            for field in 0..3 {
                let mut options = LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    ..LeastSquaresOptions::default()
                };
                match field {
                    0 => options.ftol = invalid,
                    1 => options.xtol = invalid,
                    _ => options.gtol = invalid,
                }
                let result = least_squares(
                    |_| panic!("invalid LM options must fail before calling the model"),
                    &[0.0],
                    options,
                );
                assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
            }
        }
        let result = least_squares(
            |_| panic!("zero evaluation budget must fail before calling the model"),
            &[0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                max_nfev: Some(0),
                ..LeastSquaresOptions::default()
            },
        );
        assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
        let at_epsilon = least_squares(
            |x| vec![x[0]],
            &[0.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                ftol: f64::EPSILON,
                xtol: f64::EPSILON,
                gtol: f64::EPSILON,
                ..LeastSquaresOptions::default()
            },
        )
        .expect("SciPy accepts tolerances equal to machine epsilon");
        assert_eq!(at_epsilon.status, 1);
    }

    #[test]
    fn least_squares_lm_rejects_nonfinite_numerical_jacobian() {
        for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
            let result = least_squares(
                |x| vec![if x[0] == 0.0 { 1.0 } else { f64::NAN }],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    diff_method,
                    ..LeastSquaresOptions::default()
                },
            );
            assert!(matches!(result, Err(OptError::NonFiniteInput { .. })));
        }
    }

    #[test]
    fn curve_fit_lm_large_measurement_uncertainties_do_not_stop_the_fit() {
        for sigma in [
            CurveFitSigma::Scalar(1.0e12),
            CurveFitSigma::StandardDeviations(vec![1.0e12, 2.0e12, 3.0e12, 4.0e12]),
            CurveFitSigma::Covariance(vec![
                vec![1.0e24, 0.25e24, 0.25e24, 0.25e24],
                vec![0.25e24, 1.0e24, 0.25e24, 0.25e24],
                vec![0.25e24, 0.25e24, 1.0e24, 0.25e24],
                vec![0.25e24, 0.25e24, 0.25e24, 1.0e24],
            ]),
        ] {
            let result = curve_fit(
                |x, p| p[0] * x + p[1],
                &[0.0, 1.0, 2.0, 3.0],
                &[3.0, 5.0, 7.0, 9.0],
                CurveFitOptions {
                    p0: Some(vec![0.0, 0.0]),
                    sigma: Some(sigma),
                    ..CurveFitOptions::default()
                },
            )
            .expect("uncertainties change the weights, not this exact-fit optimum");
            assert!(result.ls_result.success, "{result:?}");
            assert!((result.popt[0] - 2.0).abs() < 1e-8, "{result:?}");
            assert!((result.popt[1] - 3.0).abs() < 1e-8, "{result:?}");
        }
    }

    #[test]
    fn curve_fit_lm_preserves_full_rank_covariance_for_correlated_parameters() {
        // The exact normal matrix is
        // [[4, 4], [4, 4 + 10*delta^2]], but all four entries round to 4
        // when it is formed in f64. The Jacobian itself remains full rank.
        let delta = 2.0_f64.powi(-28);
        let offsets = [-delta, delta, -2.0 * delta, 2.0 * delta];
        let model = |t: f64, p: &[f64]| p[0] + (1.0 + t) * p[1];
        let observations: Vec<f64> = offsets.iter().map(|&t| model(t, &[1.0, 2.0])).collect();
        let jacobian = || {
            LeastSquaresJacobian::new(move |_| offsets.iter().map(|t| vec![1.0, 1.0 + t]).collect())
        };
        let fit = curve_fit(
            model,
            &offsets,
            &observations,
            CurveFitOptions {
                p0: Some(vec![0.0, 0.0]),
                absolute_sigma: true,
                ls_options: LeastSquaresOptions {
                    jac: Some(jacobian()),
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            },
        )
        .expect("a full-rank correlated model has finite covariance");
        assert!((fit.popt[0] - 1.0).abs() < 1.0e-7, "{fit:?}");
        assert!((fit.popt[1] - 2.0).abs() < 1.0e-7, "{fit:?}");
        let inverse_variance = 1.0 / (10.0 * delta * delta);
        let expected = [
            [inverse_variance + 0.25, -inverse_variance],
            [-inverse_variance, inverse_variance],
        ];
        for (actual_row, expected_row) in fit.pcov.iter().zip(expected) {
            for (&actual, reference) in actual_row.iter().zip(expected_row) {
                assert!(actual.is_finite(), "{:?}", fit.pcov);
                assert!(
                    (actual / reference - 1.0).abs() < 1e-6,
                    "covariance={:?}, expected={expected:?}",
                    fit.pcov
                );
            }
        }

        let result = leastsq(
            |p| {
                offsets
                    .iter()
                    .map(|t| (p[0] - 1.0) + (1.0 + t) * (p[1] - 2.0))
                    .collect()
            },
            &[1.0, 2.0],
            LeastSquaresOptions {
                jac: Some(jacobian()),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("leastsq uses the same stable covariance");
        let covariance = result
            .cov_x
            .expect("a resolved full-rank Jacobian has an inverse");
        assert!((covariance[0][0] / inverse_variance - 1.0).abs() < 1e-6);
        assert!((covariance[0][1] / -inverse_variance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn curve_fit_lm_covariance_handles_independent_derivative_scales() {
        let large = 1.0e160;
        let small = 1.0e-150;
        let fit = curve_fit(
            |t, p| {
                if t == 0.0 {
                    large * p[0]
                } else if t == 1.0 {
                    small * p[1]
                } else {
                    0.0
                }
            },
            &[0.0, 1.0, 2.0],
            &[0.0, 0.0, 0.0],
            CurveFitOptions {
                p0: Some(vec![0.0, 0.0]),
                absolute_sigma: true,
                ls_options: LeastSquaresOptions {
                    jac: Some(LeastSquaresJacobian::new(move |_| {
                        vec![vec![large, 0.0], vec![0.0, small], vec![0.0, 0.0]]
                    })),
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            },
        )
        .expect("independent derivative scales must not overflow the covariance");
        assert!(fit.pcov.iter().flatten().all(|value| value.is_finite()));
        // The first entry is subnormal; its rounding unit is about 5e-4 relative.
        assert!((fit.pcov[0][0] / 1.0e-320 - 1.0).abs() < 1e-3);
        assert!((fit.pcov[1][1] / 1.0e300 - 1.0).abs() < 1e-12);
        assert_eq!(fit.pcov[0][1], 0.0);
        assert_eq!(fit.pcov[1][0], 0.0);
    }

    #[test]
    fn curve_fit_lm_relative_covariance_uses_residual_norm() {
        for magnitude in [1.0e-200, 1.0e200] {
            // The optimum is p=0, RSS=2*magnitude^2, and JᵀJ=2*magnitude^2.
            // With one residual degree of freedom, the relative covariance is 1.
            // Both squared quantities underflow or overflow at these scales.
            let fit = curve_fit(
                |_, p| magnitude * p[0],
                &[0.0, 1.0],
                &[-magnitude, magnitude],
                CurveFitOptions {
                    p0: Some(vec![0.0]),
                    ls_options: LeastSquaresOptions {
                        jac: Some(LeastSquaresJacobian::new(move |_| {
                            vec![vec![magnitude], vec![magnitude]]
                        })),
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                },
            )
            .expect("relative covariance is finite in the original parameter units");
            assert!(
                (fit.pcov[0][0] - 1.0).abs() < 1e-12,
                "magnitude={magnitude}, covariance={:?}",
                fit.pcov
            );
        }
    }

    #[test]
    fn curve_fit_lm_covariance_reports_rank_deficiency() {
        let fit = curve_fit(
            |_, p| p[0] + p[1],
            &[0.0, 1.0, 2.0],
            &[3.0, 3.0, 3.0],
            CurveFitOptions {
                p0: Some(vec![1.0, 2.0]),
                absolute_sigma: true,
                ls_options: LeastSquaresOptions {
                    jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0, 1.0]; 3])),
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            },
        )
        .expect("an unidentifiable fit returns an infinite LM covariance");
        assert!(
            fit.pcov
                .iter()
                .flatten()
                .all(|value| *value == f64::INFINITY)
        );
    }

    #[test]
    fn curve_fit_quadratic() {
        // Fit y = a*x^2 + b*x + c
        let xdata: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x * x - 3.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.0).abs() < 1.0e-6,
            "a ~ 2.0, got {}",
            result.popt[0]
        );
        assert!(
            (result.popt[1] - (-3.0)).abs() < 1.0e-6,
            "b ~ -3.0, got {}",
            result.popt[1]
        );
        assert!(
            (result.popt[2] - 1.0).abs() < 1.0e-6,
            "c ~ 1.0, got {}",
            result.popt[2]
        );
    }

    // frankenscipy-szq1n.7: a least-squares solve that stops on its evaluation budget used to be
    // returned as Ok(popt). SciPy raises "Optimal parameters not found".
    #[test]
    fn curve_fit_refuses_an_unconverged_fit_like_scipy() {
        let xdata: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.25).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 3.0 * (-1.3 * x).exp()).collect();
        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp();
        let starved = CurveFitOptions {
            p0: Some(vec![1.0, 0.1]),
            ls_options: LeastSquaresOptions {
                max_nfev: Some(2),
                ..LeastSquaresOptions::default()
            },
            ..CurveFitOptions::default()
        };
        let err = curve_fit(model, &xdata, &ydata, starved).expect_err("2 evaluations cannot fit");
        println!("curve_fit starved: {err}");
        assert!(
            matches!(&err, OptError::NotConverged { detail } if detail.starts_with("Optimal parameters not found")),
            "{err:?}"
        );
        let fine = CurveFitOptions {
            p0: Some(vec![1.0, 0.1]),
            ..CurveFitOptions::default()
        };
        let fit = curve_fit(model, &xdata, &ydata, fine).expect("converges");
        assert!((fit.popt[0] - 3.0).abs() < 1e-6 && (fit.popt[1] - 1.3).abs() < 1e-6);
    }

    #[test]
    fn curve_fit_bounded_interior_optimum_and_active_bound() {
        // Exponential-plus-offset; noiseless data => the least-squares minimum is the
        // exact generating parameters, so a correct bounded solve recovers them.
        let xdata: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let model = |x: f64, p: &[f64]| p[0] * (-p[1] * x).exp() + p[2];
        let truep = [3.0_f64, 0.7, 1.0];
        let ydata: Vec<f64> = xdata.iter().map(|&x| model(x, &truep)).collect();

        // (1) Interior optimum: bounds strictly contain the truth -> exact recovery,
        //     identical to scipy's trf minimiser.
        let lower = [0.0, 0.0, -5.0];
        let upper = [10.0, 5.0, 5.0];
        let opts = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let r = curve_fit_bounded(model, &xdata, &ydata, &lower, &upper, opts).expect("converges");
        for k in 0..3 {
            assert!(
                (r.popt[k] - truep[k]).abs() < 1.0e-5,
                "interior popt[{k}] = {} expected {}",
                r.popt[k],
                truep[k]
            );
            assert!(
                r.popt[k] >= lower[k] && r.popt[k] <= upper[k],
                "popt[{k}] out of bounds"
            );
        }
        // pcov is finite and symmetric (recomputed in parameter space).
        assert!(r.pcov.iter().flatten().all(|v| v.is_finite()));

        // (2) Active bound: cap the amplitude below the truth -> TRF pins it ON the bound and
        //     reports it active, as SciPy's trf does (the old logistic reparameterisation only
        //     approached it asymptotically).
        let upper2 = [2.0, 5.0, 5.0];
        let opts2 = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let r2 =
            curve_fit_bounded(model, &xdata, &ydata, &lower, &upper2, opts2).expect("converges");
        // SciPy 1.17.1 curve_fit(bounds=...) stops at 1.999999997496993 (ftol), active_mask
        // [1, 0, 0]; the port takes the same path.
        assert!(
            (r2.popt[0] - 1.999_999_997_496_993).abs() < 1.0e-12,
            "active-bound popt[0] = {} (SciPy 1.999999997496993)",
            r2.popt[0]
        );
        assert_eq!(r2.ls_result.active_mask, vec![1, 0, 0]);
    }

    // frankenscipy-1ksfv.6: least_squares defaults to SciPy's trf. SciPy 1.17.1 on
    // p0·e^(−p1·t) − y, t = linspace(0, 4, 40), y = 2e^(−0.3t) with y[5] += 3, y[20] −= 2.5,
    // x0 = [1, 1]: bounds ([0, 0.5], [10, 5]) -> x = [2.56323078, 0.5], active_mask [0, −1],
    // status 4, 10 function evaluations; loss soft_l1 with f_scale 0.1 -> [2.01080623,
    // 0.30341512], status 2, 11 evaluations. The port takes SciPy's path (same nfev).
    #[test]
    fn least_squares_trf_matches_scipy_with_bounds_and_robust_loss() {
        let t: Vec<f64> = (0..40).map(|i| 4.0 * f64::from(i) / 39.0).collect();
        let mut y: Vec<f64> = t.iter().map(|ti| 2.0 * (-0.3 * ti).exp()).collect();
        y[5] += 3.0;
        y[20] -= 2.5;
        let residuals = |p: &[f64]| -> Vec<f64> {
            t.iter()
                .zip(&y)
                .map(|(ti, yi)| p[0] * (-p[1] * ti).exp() - yi)
                .collect()
        };
        let bounded = least_squares_bounded(
            residuals,
            &[1.0, 1.0],
            &[0.0, 0.5],
            &[10.0, 5.0],
            LeastSquaresOptions::default(),
        )
        .expect("trf");
        assert!(bounded.success, "{}", bounded.message);
        assert!(
            (bounded.x[0] - 2.563_230_78).abs() < 1e-7,
            "{:?}",
            bounded.x
        );
        assert!((bounded.x[1] - 0.5).abs() < 1e-12, "{:?}", bounded.x);
        assert_eq!(bounded.active_mask, vec![0, -1]);
        assert_eq!((bounded.status, bounded.nfev), (4, 10));

        let robust = least_squares(
            residuals,
            &[1.0, 1.0],
            LeastSquaresOptions {
                loss: LossKind::SoftL1,
                f_scale: 0.1,
                ..LeastSquaresOptions::default()
            },
        )
        .expect("trf soft_l1");
        assert!((robust.x[0] - 2.010_806_23).abs() < 1e-7, "{:?}", robust.x);
        assert!((robust.x[1] - 0.303_415_12).abs() < 1e-7, "{:?}", robust.x);
        assert_eq!((robust.status, robust.nfev), (2, 11));

        // A plain linear fit with the outliers is pulled far from 2e^(−0.3t); the robust loss
        // is what recovers it.
        let plain = least_squares(residuals, &[1.0, 1.0], LeastSquaresOptions::default())
            .expect("trf linear");
        assert!((plain.x[1] - 0.3).abs() > 3.0 * (robust.x[1] - 0.3).abs());

        let lm_refused = least_squares_bounded(
            residuals,
            &[1.0, 1.0],
            &[0.0, 0.5],
            &[10.0, 5.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                ..LeastSquaresOptions::default()
            },
        )
        .expect_err("lm has no bounds");
        assert!(
            matches!(lm_refused, OptError::InvalidArgument { detail } if detail.contains("'lm'"))
        );
    }

    #[test]
    fn least_squares_dogbox_reaches_active_bounds_and_preserves_full_gradient() {
        let result = least_squares_bounded(
            |x| vec![x[0] - 3.0, x[1] + 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Dogbox),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("Dogbox solves the constrained quadratic");
        assert_eq!(result.x, vec![2.0, 0.0]);
        assert_eq!(result.active_mask, vec![1, -1]);
        assert_eq!(result.status, 1);
        assert_eq!(result.optimality, 0.0);
        assert!((result.grad[0] + 1.0).abs() < 1e-7);
        assert!((result.grad[1] - 1.0).abs() < 1e-7);
        assert!((result.cost - 1.0).abs() < 1e-14);
    }

    #[test]
    fn least_squares_dogbox_underdetermined_and_rank_deficient() {
        for duplicate in [false, true] {
            let result = least_squares(
                |x| {
                    let r = x[0] + 2.0 * x[1] - 3.0;
                    if duplicate { vec![r, 2.0 * r] } else { vec![r] }
                },
                &[0.0, 0.0],
                LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Dogbox),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("Dogbox accepts a deficient dense Jacobian");
            assert!(result.success);
            assert!(result.cost < 1e-20);
            assert!((result.x[0] - 0.6).abs() < 1e-7);
            assert!((result.x[1] - 1.2).abs() < 1e-7);
        }
    }

    #[test]
    fn least_squares_dogbox_keeps_initial_bound_and_honors_evaluation_budget() {
        let result = least_squares_bounded(
            |x| vec![x[0] + 1.0],
            &[0.0],
            &[0.0],
            &[1.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Dogbox),
                max_nfev: Some(1),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("initial bound optimum");
        assert_eq!(result.x, vec![0.0]);
        assert_eq!(result.active_mask, vec![-1]);
        assert_eq!(result.nfev, 1);
        assert_eq!(result.status, 1);

        let starved = least_squares(
            |x| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]],
            &[-1.2, 1.0],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Dogbox),
                max_nfev: Some(3),
                ..LeastSquaresOptions::default()
            },
        )
        .expect("budget exhaustion returns the current iterate");
        assert_eq!(starved.nfev, 3);
        assert_eq!(starved.status, 0);
        assert!(!starved.success);
    }

    #[test]
    fn least_squares_trust_regions_reject_zero_evaluation_budget() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares(
                |x| vec![x[0] - 1.0],
                &[0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    max_nfev: Some(0),
                    ..LeastSquaresOptions::default()
                },
            );
            assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
        }
    }

    #[test]
    fn least_squares_analytic_jacobian_counts_only_actual_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            let residual_calls = AtomicUsize::new(0);
            let jacobian_calls = Arc::new(AtomicUsize::new(0));
            let jacobian_count = Arc::clone(&jacobian_calls);
            let slopes = [1.0, 2.0, 4.0];
            let result = least_squares(
                |x| {
                    residual_calls.fetch_add(1, Ordering::Relaxed);
                    vec![x[0] - 2.0, 2.0 * x[0] - 4.0, 4.0 * x[0] - 8.0]
                },
                &[0.5],
                LeastSquaresOptions {
                    method: Some(method),
                    diff_method: FiniteDiffMethod::ThreePoint,
                    jac: Some(LeastSquaresJacobian::new(move |_| {
                        jacobian_count.fetch_add(1, Ordering::Relaxed);
                        slopes.iter().map(|&slope| vec![slope]).collect()
                    })),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("fit with a captured analytic Jacobian");
            assert!(result.success, "{method:?}: {result:?}");
            assert!((result.x[0] - 2.0).abs() < 1e-7);
            assert_eq!(result.nfev, residual_calls.load(Ordering::Relaxed));
            assert_eq!(result.njev, jacobian_calls.load(Ordering::Relaxed));
            assert_eq!(
                result.jac.to_dense().unwrap(),
                vec![vec![1.0], vec![2.0], vec![4.0]]
            );
        }
    }

    #[test]
    fn least_squares_validates_analytic_jacobian_every_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            for bad in [
                vec![],
                vec![vec![1.0]],
                vec![vec![1.0, 0.0], vec![0.0]],
                vec![vec![1.0, 0.0], vec![0.0, f64::NAN]],
            ] {
                let result = least_squares(
                    |x| vec![x[0] - 1.0, x[1] - 2.0],
                    &[0.0, 0.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new(move |_| bad.clone())),
                        ..LeastSquaresOptions::default()
                    },
                );
                assert!(matches!(
                    result,
                    Err(OptError::InvalidArgument { .. } | OptError::NonFiniteInput { .. })
                ));
            }
            let calls = AtomicUsize::new(0);
            let result = least_squares(
                |x| vec![x[0] - 1.0, x[1] - 2.0],
                &[0.0, 0.0],
                LeastSquaresOptions {
                    method: Some(method),
                    jac: Some(LeastSquaresJacobian::new(move |_| {
                        if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                            vec![vec![1.0, 0.0], vec![0.0, 1.0]]
                        } else {
                            vec![vec![1.0]]
                        }
                    })),
                    ..LeastSquaresOptions::default()
                },
            );
            assert!(
                matches!(result, Err(OptError::InvalidArgument { .. })),
                "refresh must validate: {method:?}: {result:?}"
            );
        }
    }

    #[test]
    fn least_squares_validates_fixed_scale_before_callbacks() {
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            for scale in [
                vec![],
                vec![1.0, 2.0],
                vec![0.0],
                vec![-1.0],
                vec![f64::INFINITY],
                vec![f64::NAN],
            ] {
                let result = least_squares(
                    |_| panic!("invalid scales must be rejected before evaluating the model"),
                    &[0.0],
                    LeastSquaresOptions {
                        method: Some(method),
                        x_scale: Some(XScale::Fixed(scale)),
                        ..LeastSquaresOptions::default()
                    },
                );
                assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
            }
        }
    }

    #[test]
    fn curve_fit_bounded_uses_analytic_jacobian_and_fixed_scaling() {
        let xdata: Vec<f64> = (0..6).map(f64::from).collect();
        let ydata: Vec<f64> = xdata.iter().map(|x| 2.0 * x + 1.0).collect();
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let jac_x = xdata.clone();
            let fit = curve_fit_bounded(
                |x, p| p[0] * x + p[1],
                &xdata,
                &ydata,
                &[0.0, 0.0],
                &[1.5, 10.0],
                CurveFitOptions {
                    p0: Some(vec![1.0, 1.0]),
                    ls_options: LeastSquaresOptions {
                        method: Some(method),
                        x_scale: Some(XScale::Fixed(vec![0.5, 2.0])),
                        jac: Some(LeastSquaresJacobian::new(move |_| {
                            jac_x.iter().map(|&x| vec![x, 1.0]).collect()
                        })),
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                },
            )
            .expect("bounded analytic curve fit");
            assert!((fit.popt[0] - 1.5).abs() < 1e-8);
            assert!((fit.popt[1] - 2.25).abs() < 1e-7);
            assert_eq!(fit.ls_result.active_mask, vec![1, 0]);
        }
    }

    #[test]
    fn least_squares_many_shares_analytic_jacobian() {
        let results = least_squares_many(
            |x, params| vec![x[0] - params[0]],
            &[0.0],
            &[vec![1.0], vec![2.0], vec![3.0]],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Dogbox),
                jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0]])),
                x_scale: Some(XScale::Fixed(vec![2.0])),
                ..LeastSquaresOptions::default()
            },
        );
        for (index, result) in results.iter().enumerate() {
            assert!(
                (result.as_ref().expect("batch member").x[0] - (index + 1) as f64).abs() < 1e-8
            );
        }
    }

    #[test]
    fn least_squares_detects_residual_shape_change_during_differencing() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
                for malformed_call in [1, 2] {
                    let calls = AtomicUsize::new(0);
                    let result = least_squares(
                        |x| {
                            if calls.fetch_add(1, Ordering::Relaxed) == malformed_call {
                                vec![]
                            } else {
                                vec![x[0] - 1.0]
                            }
                        },
                        &[0.0],
                        LeastSquaresOptions {
                            method: Some(method),
                            diff_method,
                            ..LeastSquaresOptions::default()
                        },
                    );
                    assert!(
                        matches!(result, Err(OptError::InvalidArgument { .. })),
                        "{method:?}/{diff_method:?} call {malformed_call}: {result:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn least_squares_lm_parallel_differencing_rejects_shape_change() {
        // Cross the production parallel threshold; a short column must be an error
        // even when it was evaluated by a worker rather than the serial path.
        let result = least_squares(
            |x| {
                if x.iter().any(|&value| value != 0.0) {
                    vec![]
                } else {
                    vec![1.0; 8192]
                }
            },
            &[0.0; 16],
            LeastSquaresOptions {
                method: Some(LeastSquaresMethod::Lm),
                ..LeastSquaresOptions::default()
            },
        );
        assert!(matches!(result, Err(OptError::InvalidArgument { .. })));
    }

    #[test]
    fn least_squares_three_point_improves_numerical_jacobian_accuracy() {
        let x = [-0.75_f64, 0.5];
        let residuals =
            |x: &[f64]| vec![(1.3 * x[0]).exp() + x[1].sin(), x[0].cos() - (-x[1]).exp()];
        let exact = [
            [1.3 * (1.3 * x[0]).exp(), x[1].cos()],
            [-x[0].sin(), (-x[1]).exp()],
        ];
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let mut errors = Vec::new();
            for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
                let result = least_squares(
                    residuals,
                    &x,
                    LeastSquaresOptions {
                        method: Some(method),
                        diff_method,
                        max_nfev: Some(1),
                        ..LeastSquaresOptions::default()
                    },
                )
                .expect("return initial numerical Jacobian");
                let error = result
                    .jac
                    .to_dense()
                    .unwrap()
                    .iter()
                    .flatten()
                    .zip(exact.iter().flatten())
                    .map(|(actual, exact)| (actual - exact).abs())
                    .fold(0.0_f64, f64::max);
                errors.push(error);
            }
            // Independent analytic derivatives distinguish the requested scheme
            // from silently continuing to use the two-point implementation.
            assert!(errors[1] < 2e-10, "{method:?}: {errors:?}");
            assert!(errors[0] > 1e-9, "two-point control: {errors:?}");
        }
    }

    #[test]
    fn least_squares_three_point_stencils_stay_within_bounds() {
        use std::sync::Mutex;
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let visited = Mutex::new(Vec::<Vec<f64>>::new());
            // Exact bounds require one-sided stencils; a narrow interval forces
            // the nominal cube-root-epsilon step to shrink.
            let lower = [0.0, 0.0, 1.0 - 1e-7];
            let upper = [1.0, 1.0, 1.0 + 1e-7];
            let result = least_squares_bounded(
                |x| {
                    assert!((0..3).all(|i| lower[i] <= x[i] && x[i] <= upper[i]));
                    visited.lock().expect("record stencil").push(x.to_vec());
                    vec![x[0].exp(), x[1].exp(), x[2].sin()]
                },
                &[0.0, 1.0, 1.0],
                &lower,
                &upper,
                LeastSquaresOptions {
                    method: Some(method),
                    diff_method: FiniteDiffMethod::ThreePoint,
                    max_nfev: Some(1),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("three-point Jacobian at a bounded initial point");
            let visited = visited.lock().expect("inspect stencil");
            assert_eq!(
                visited.len(),
                7,
                "one base evaluation plus two per variable"
            );
            assert!(visited[1][0] > visited[0][0] && visited[2][0] > visited[1][0]);
            assert!(visited[3][1] < visited[0][1] && visited[4][1] < visited[3][1]);
            assert!(visited[5][2] < visited[0][2] && visited[6][2] > visited[0][2]);
            for (i, exact) in [result.x[0].exp(), result.x[1].exp(), result.x[2].cos()]
                .into_iter()
                .enumerate()
            {
                assert!((result.jac.to_dense().unwrap()[i][i] - exact).abs() < 2e-9);
            }
            assert_eq!(
                result.nfev, 1,
                "numerical Jacobian calls are counted separately"
            );
            assert_eq!(result.njev, 1);
        }
    }

    #[test]
    fn least_squares_three_point_does_not_reverse_gradients_in_one_ulp_bounds() {
        let lower = 1.0_f64;
        let upper = lower.next_up();
        let width = upper - lower;
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
                for sparse in [false, true] {
                    let result = least_squares_bounded(
                        |x| {
                            assert!(lower <= x[0] && x[0] <= upper);
                            vec![(x[0] - lower) / width - 1.0]
                        },
                        &[lower],
                        &[lower],
                        &[upper],
                        LeastSquaresOptions {
                            method: Some(method),
                            diff_method,
                            jac_sparsity: sparse
                                .then(|| JacobianSparsity::from_rows(vec![vec![0]], 1).unwrap()),
                            ..LeastSquaresOptions::default()
                        },
                    )
                    .expect("a nonconstant residual in a one-ULP interval can be minimized");
                    // The old three-point stencil sampled the base point twice,
                    // reversed the gradient, and falsely declared the lower bound
                    // optimal. A feasible secant recovers the exact linear slope.
                    assert!(
                        result.success,
                        "{method:?}/{diff_method:?}, sparse={sparse}: {result:?}"
                    );
                    assert_eq!(result.x, vec![upper]);
                    assert_eq!(result.fun, vec![0.0]);
                    assert_eq!(result.cost, 0.0);
                    assert_eq!(result.jac.to_dense().unwrap(), vec![vec![1.0 / width]]);
                }
            }
        }
    }

    #[test]
    fn least_squares_three_point_uses_actual_unequal_node_spacing() {
        let lower = 1.0_f64;
        let ulp = lower.next_up() - lower;
        let upper = lower + 3.0 * ulp;
        let residuals = |x: &[f64]| {
            let t = (x[0] - lower) / ulp;
            vec![t * t + 2.0 * t + 3.0]
        };
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for sparse in [false, true] {
                let options = LeastSquaresOptions {
                    method: Some(method),
                    diff_method: FiniteDiffMethod::ThreePoint,
                    jac_sparsity: sparse
                        .then(|| JacobianSparsity::from_rows(vec![vec![0]], 1).unwrap()),
                    max_nfev: Some(1),
                    ..LeastSquaresOptions::default()
                };
                for start in [lower, upper] {
                    let result = least_squares_bounded(
                        |x| {
                            assert!(lower <= x[0] && x[0] <= upper);
                            residuals(x)
                        },
                        &[start],
                        &[lower],
                        &[upper],
                        options.clone(),
                    )
                    .expect("quadratic derivative in a three-ULP interval");
                    let t = (result.x[0] - lower) / ulp;
                    let exact = (2.0 * t + 2.0) / ulp;
                    let actual = result.jac.to_dense().unwrap()[0][0];
                    assert!(
                        (actual / exact - 1.0).abs() < 1e-14,
                        "{method:?}, sparse={sparse}, start={start}: {actual} != {exact}"
                    );
                }
                // At a power of two the lower ULP is half the upper ULP.
                // This nominally central stencil therefore has unequal spacing.
                let result = least_squares(
                    residuals,
                    &[lower],
                    LeastSquaresOptions {
                        diff_step: Some(0.6 * ulp),
                        ..options
                    },
                )
                .expect("quadratic derivative on an asymmetric central stencil");
                let exact = 2.0 / ulp;
                let actual = result.jac.to_dense().unwrap()[0][0];
                assert!(
                    (actual / exact - 1.0).abs() < 1e-14,
                    "{method:?}, sparse={sparse}: {actual} != {exact}"
                );
            }
        }
    }

    #[test]
    fn least_squares_three_point_groups_mix_secants_and_quadratics() {
        let lower = 1.0_f64;
        let upper = lower.next_up();
        let width = upper - lower;
        let pattern = JacobianSparsity::from_rows(vec![vec![0], vec![1]], 2).unwrap();
        assert_eq!(pattern.color_count(), 1);
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            let result = least_squares_bounded(
                |x| {
                    assert!(lower <= x[0] && x[0] <= upper);
                    assert!(0.0 <= x[1] && x[1] <= 2.0);
                    vec![(x[0] - lower) / width - 1.0, x[1] * x[1] + 2.0 * x[1] + 3.0]
                },
                &[lower, 1.0],
                &[lower, 0.0],
                &[upper, 2.0],
                LeastSquaresOptions {
                    method: Some(method),
                    diff_method: FiniteDiffMethod::ThreePoint,
                    diff_step: Some(1.0),
                    jac_sparsity: Some(pattern.clone()),
                    max_nfev: Some(1),
                    ..LeastSquaresOptions::default()
                },
            )
            .expect("a sparse color can combine two-point and three-point columns");
            assert_eq!(
                result.jac.to_dense().unwrap(),
                vec![vec![1.0 / width, 0.0], vec![0.0, 4.0]]
            );
        }
    }

    #[test]
    fn finite_difference_stencils_preserve_finite_slopes_and_constant_offsets() {
        let residuals = |x: &[f64]| vec![1.0e308 * (x[0] - 1.0), 1.0e308];
        let pattern = JacobianSparsity::from_rows(vec![vec![0], vec![0]], 1).unwrap();
        for initial in [0.0, 1.0] {
            let x = [initial];
            let f0 = residuals(&x);
            let dense = fd_jacobian_scipy(
                &residuals,
                &x,
                &f0,
                &[0.0],
                &[2.0],
                FiniteDiffMethod::ThreePoint,
                Some(1.0),
            )
            .expect("finite values and derivatives must not overflow intermediate differences");
            let sparse = fd_jacobian_grouped(
                &residuals,
                &x,
                &f0,
                &[0.0],
                &[2.0],
                FiniteDiffMethod::ThreePoint,
                Some(1.0),
                &pattern,
            )
            .unwrap()
            .to_dense()
            .unwrap();
            assert_eq!(dense, sparse);
            assert!((dense[0][0] / 1.0e308 - 1.0).abs() < 1e-9);
            assert_eq!(dense[1][0], 0.0);
        }
    }

    #[test]
    fn least_squares_numerical_jacobians_never_sample_infinite_parameters() {
        for initial in [-f64::MAX, f64::MAX] {
            for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
                for diff_method in [FiniteDiffMethod::TwoPoint, FiniteDiffMethod::ThreePoint] {
                    for sparse in [false, true] {
                        let result = least_squares(
                            |x| {
                                assert!(x[0].is_finite(), "numerical sample must remain finite");
                                vec![x[0] / f64::MAX]
                            },
                            &[initial],
                            LeastSquaresOptions {
                                method: Some(method),
                                diff_method,
                                jac_sparsity: sparse.then(|| {
                                    JacobianSparsity::from_rows(vec![vec![0]], 1).unwrap()
                                }),
                                max_nfev: Some(1),
                                ..LeastSquaresOptions::default()
                            },
                        )
                        .expect("finite one-sided samples exist at either end of f64 range");
                        let actual = result.jac.to_dense().unwrap()[0][0];
                        let exact = 1.0 / f64::MAX;
                        assert!(
                            (actual / exact - 1.0).abs() < 1e-8,
                            "{method:?}/{diff_method:?}, sparse={sparse}, initial={initial}: {actual}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn curve_fit_three_point_nonlinear_fit_all_methods() {
        let t: Vec<f64> = (0..30).map(|i| f64::from(i) / 10.0).collect();
        let y: Vec<f64> = t.iter().map(|&t| 2.5 * (-0.7 * t).exp() + 0.3).collect();
        for method in [
            LeastSquaresMethod::Trf,
            LeastSquaresMethod::Dogbox,
            LeastSquaresMethod::Lm,
        ] {
            let options = CurveFitOptions {
                p0: Some(vec![1.5, 0.4, 0.1]),
                ls_options: LeastSquaresOptions {
                    method: Some(method),
                    diff_method: FiniteDiffMethod::ThreePoint,
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            };
            let model = |t: f64, p: &[f64]| p[0] * (-p[1] * t).exp() + p[2];
            let fit = curve_fit(model, &t, &y, options.clone()).expect("unbounded three-point fit");
            for (actual, expected) in fit.popt.iter().zip([2.5, 0.7, 0.3]) {
                assert!((actual - expected).abs() < 5e-6, "{method:?}: {fit:?}");
            }
            if method != LeastSquaresMethod::Lm {
                let bounded =
                    curve_fit_bounded(model, &t, &y, &[0.0, 0.0, 0.0], &[5.0, 2.0, 1.0], options)
                        .expect("bounded three-point fit");
                for (actual, expected) in bounded.popt.iter().zip([2.5, 0.7, 0.3]) {
                    assert!((actual - expected).abs() < 5e-6, "{method:?}: {bounded:?}");
                }
            }
        }
    }

    #[test]
    fn least_squares_empty_residuals_preserve_parameter_dimensions() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for analytic in [false, true] {
                for bounded in [false, true] {
                    let options = LeastSquaresOptions {
                        method: Some(method),
                        x_scale: Some(XScale::Jac),
                        jac: analytic.then(|| LeastSquaresJacobian::new(|_| vec![])),
                        ..LeastSquaresOptions::default()
                    };
                    let result = if bounded {
                        least_squares_bounded(
                            |_| vec![],
                            &[0.25, 0.75],
                            &[0.0, 0.0],
                            &[1.0, 1.0],
                            options,
                        )
                    } else {
                        least_squares(|_| vec![], &[0.25, 0.75], options)
                    }
                    .expect("SciPy accepts an empty residual system");
                    assert_eq!(result.x, vec![0.25, 0.75]);
                    assert_eq!(result.grad, vec![0.0, 0.0]);
                    assert_eq!(result.active_mask, vec![0, 0]);
                    assert_eq!(result.status, 1);
                    assert_eq!(result.cost, 0.0);
                    assert_eq!(result.optimality, 0.0);
                    assert_eq!(result.nfev, 1);
                    assert_eq!(result.njev, 1);
                }
            }
        }
    }

    #[test]
    fn curve_fit_many_byte_identical_to_per_row() {
        // Batched fit must equal looping curve_fit per row, bit-for-bit (each fit is
        // independent; parallelism only distributes them).
        let x: Vec<f64> = (0..80).map(|i| i as f64 * 5.0 / 79.0).collect();
        let model = |xi: f64, p: &[f64]| p[0] * (-p[1] * xi).exp() + p[2];
        let mut s = 12345u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let nrows = 40usize; // crosses the serial->parallel gate
        let rows: Vec<Vec<f64>> = (0..nrows)
            .map(|_| {
                let (a, b, c) = (1.0 + 2.0 * rng(), 0.3 + rng(), rng());
                x.iter()
                    .map(|&xi| a * (-b * xi).exp() + c + 0.02 * (rng() - 0.5))
                    .collect()
            })
            .collect();
        let opts = || CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let batched = curve_fit_many(model, &x, &rows, opts()).expect("batched");
        assert_eq!(batched.len(), nrows);
        for (i, row) in rows.iter().enumerate() {
            let single = curve_fit(model, &x, row, opts()).expect("single").popt;
            for k in 0..3 {
                assert_eq!(
                    batched[i][k].to_bits(),
                    single[k].to_bits(),
                    "row {i} param {k}: batched {} vs single {}",
                    batched[i][k],
                    single[k]
                );
            }
        }
        // bounded batched is likewise bit-identical to per-row bounded.
        let lo = [0.0, 0.0, -2.0];
        let hi = [6.0, 4.0, 2.0];
        let bb = curve_fit_bounded_many(model, &x, &rows, &lo, &hi, opts()).expect("batched bnd");
        for (i, row) in rows.iter().enumerate() {
            let single = curve_fit_bounded(model, &x, row, &lo, &hi, opts())
                .expect("single bnd")
                .popt;
            for k in 0..3 {
                assert_eq!(
                    bb[i][k].to_bits(),
                    single[k].to_bits(),
                    "bnd row {i} param {k}"
                );
            }
        }
        assert!(curve_fit_many(model, &x, &[], opts()).unwrap().is_empty());
    }

    #[test]
    fn least_squares_many_byte_identical_to_per_param() {
        // Fit y = a*exp(-b*t): residuals(x, y) = a*exp(-b*t) - y, swept over many
        // noisy datasets. The batched solve must equal looping least_squares per
        // param, bit-for-bit, and converge near the generating (a, b).
        let t: Vec<f64> = (0..40).map(|i| 3.0 * i as f64 / 39.0).collect();
        let resid = |x: &[f64], y: &[f64]| -> Vec<f64> {
            t.iter()
                .zip(y)
                .map(|(&ti, &yi)| x[0] * (-x[1] * ti).exp() - yi)
                .collect()
        };
        let mut s = 19u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let nrows = 40usize;
        let truth: Vec<(f64, f64)> = (0..nrows)
            .map(|_| (1.0 + 2.0 * rng(), 0.5 + 1.5 * rng()))
            .collect();
        let ys: Vec<Vec<f64>> = truth
            .iter()
            .map(|&(a, b)| t.iter().map(|&ti| a * (-b * ti).exp()).collect())
            .collect();
        let opt = LeastSquaresOptions::default();

        let batched = least_squares_many(resid, &[1.5, 1.0], &ys, opt.clone());
        assert_eq!(batched.len(), nrows);
        for (i, y) in ys.iter().enumerate() {
            let single = least_squares(|x| resid(x, y), &[1.5, 1.0], opt.clone()).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.x[0].to_bits(), single.x[0].to_bits(), "a mismatch {i}");
            assert_eq!(many.x[1].to_bits(), single.x[1].to_bits(), "b mismatch {i}");
            // noiseless data => recovers the generating parameters.
            assert!((many.x[0] - truth[i].0).abs() < 1e-6, "a {i}");
            assert!((many.x[1] - truth[i].1).abs() < 1e-6, "b {i}");
        }
        assert!(least_squares_many(resid, &[1.5, 1.0], &[], opt).is_empty());
    }

    #[test]
    fn curve_fit_bounded_validates_bounds() {
        let xdata = [0.0, 1.0, 2.0, 3.0];
        let ydata = [0.0, 1.0, 2.0, 3.0];
        let model = |x: f64, p: &[f64]| p[0] * x + p[1];
        // lower >= upper is rejected.
        let opts = CurveFitOptions {
            p0: Some(vec![1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let err = curve_fit_bounded(model, &xdata, &ydata, &[1.0, 0.0], &[1.0, 5.0], opts);
        assert!(err.is_err(), "equal lower/upper must be rejected");
    }

    #[test]
    fn curve_fit_sinusoidal() {
        // Fit y = A * sin(omega * x + phi)
        let xdata: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.5 * (1.5 * x + 0.3).sin()).collect();

        let model = |x: f64, p: &[f64]| p[0] * (p[1] * x + p[2]).sin();

        let options = CurveFitOptions {
            p0: Some(vec![2.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };

        let result = curve_fit(model, &xdata, &ydata, options).expect("should converge");
        assert!(result.ls_result.success, "{}", result.ls_result.message);
        assert!(
            (result.popt[0] - 2.5).abs() < 0.1,
            "A ~ 2.5, got {}",
            result.popt[0]
        );
    }

    #[test]
    fn curve_fit_returns_covariance() {
        let xdata: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Linear: y = 2x + 1 with noise
        let ydata: Vec<f64> = xdata.iter().map(|&x| 2.0 * x + 1.0).collect();

        let model = |x: f64, p: &[f64]| p[0] * x + p[1];

        let options = CurveFitOptions {
            p0: Some(vec![1.0, 0.0]),
            absolute_sigma: true,
            ..CurveFitOptions::default()
        };

        let (result, warnings) =
            fsci_runtime::catch_warnings(|| curve_fit(model, &xdata, &ydata, options));
        let result = result.expect("should converge");
        // Covariance should be 2x2 and finite
        assert_eq!(result.pcov.len(), 2);
        assert_eq!(result.pcov[0].len(), 2);
        assert!(result.pcov[0][0].is_finite());
        assert!(result.pcov[1][1].is_finite());
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn curve_fit_trust_region_accepts_underdetermined_models() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for bounded in [false, true] {
                for absolute_sigma in [false, true] {
                    let options = CurveFitOptions {
                        p0: Some(vec![0.0, 0.0]),
                        absolute_sigma,
                        ls_options: LeastSquaresOptions {
                            method: Some(method),
                            gtol: 1e-12,
                            xtol: 1e-12,
                            ftol: 1e-12,
                            jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0, 1.0]])),
                            ..LeastSquaresOptions::default()
                        },
                        ..CurveFitOptions::default()
                    };
                    let (fit, warnings) = fsci_runtime::catch_warnings(|| {
                        if bounded {
                            curve_fit_bounded(
                                |_x, p| p[0] + p[1],
                                &[0.0],
                                &[1.0],
                                &[-2.0, -2.0],
                                &[2.0, 2.0],
                                options,
                            )
                        } else {
                            curve_fit(|_x, p| p[0] + p[1], &[0.0], &[1.0], options)
                        }
                    });
                    let fit =
                        fit.expect("TRF and Dogbox support fewer observations than parameters");
                    for parameter in &fit.popt {
                        assert!((parameter - 0.5).abs() < 1e-7, "{method:?}: {:?}", fit.popt);
                    }
                    assert_eq!(fit.pcov.len(), 2);
                    assert!(fit.pcov.iter().all(|row| row.len() == 2));
                    if absolute_sigma {
                        assert!(
                            fit.pcov
                                .iter()
                                .flatten()
                                .all(|value| (value - 0.25).abs() < 1e-14)
                        );
                        assert!(warnings.is_empty(), "{warnings:?}");
                    } else {
                        assert!(fit.pcov.iter().flatten().all(|value| value.is_infinite()));
                        assert_eq!(warnings.len(), 1);
                        assert_eq!(
                            warnings[0].category,
                            fsci_runtime::WarningCategory::OptimizeWarning
                        );
                    }
                }
            }
        }

        for bounded in [false, true] {
            let options = CurveFitOptions {
                p0: Some(vec![0.0, 0.0]),
                ls_options: LeastSquaresOptions {
                    method: Some(LeastSquaresMethod::Lm),
                    ..LeastSquaresOptions::default()
                },
                ..CurveFitOptions::default()
            };
            let result = if bounded {
                curve_fit_bounded(
                    |_x, p| p[0] + p[1],
                    &[0.0],
                    &[1.0],
                    &[f64::NEG_INFINITY; 2],
                    &[f64::INFINITY; 2],
                    options,
                )
            } else {
                curve_fit(|_x, p| p[0] + p[1], &[0.0], &[1.0], options)
            };
            assert!(
                matches!(result, Err(OptError::InvalidArgument { detail }) if detail.contains("data points"))
            );
        }
    }

    #[test]
    fn curve_fit_trust_region_rank_deficient_covariance() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for bounded in [false, true] {
                for absolute_sigma in [false, true] {
                    let options = CurveFitOptions {
                        p0: Some(vec![1.0, 1.0]),
                        absolute_sigma,
                        ls_options: LeastSquaresOptions {
                            method: Some(method),
                            jac: Some(LeastSquaresJacobian::new(|_| vec![vec![1.0, 1.0]; 3])),
                            ..LeastSquaresOptions::default()
                        },
                        ..CurveFitOptions::default()
                    };
                    let (fit, warnings) = fsci_runtime::catch_warnings(|| {
                        if bounded {
                            curve_fit_bounded(
                                |_x, p| p[0] + p[1],
                                &[0.0, 1.0, 2.0],
                                &[1.0, 2.0, 3.0],
                                &[-2.0, -2.0],
                                &[2.0, 2.0],
                                options,
                            )
                        } else {
                            curve_fit(
                                |_x, p| p[0] + p[1],
                                &[0.0, 1.0, 2.0],
                                &[1.0, 2.0, 3.0],
                                options,
                            )
                        }
                    });
                    let fit = fit.expect("rank-deficient fit has a pseudoinverse covariance");
                    let expected = if absolute_sigma {
                        1.0 / 12.0
                    } else {
                        1.0 / 6.0
                    };
                    for &entry in fit.pcov.iter().flatten() {
                        assert!(
                            (entry - expected).abs() < 1e-14,
                            "{method:?}: {:?}",
                            fit.pcov
                        );
                    }
                    assert!(warnings.is_empty(), "{warnings:?}");
                }
            }
        }
    }

    #[test]
    fn curve_fit_trust_region_zero_rank_and_scipy_singular_value_cutoff() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for weak_derivative in [0.0, 5e-16, 1e-14] {
                let options = CurveFitOptions {
                    p0: Some(vec![0.0, 0.0]),
                    absolute_sigma: true,
                    ls_options: LeastSquaresOptions {
                        method: Some(method),
                        jac: Some(LeastSquaresJacobian::new(move |_| {
                            vec![vec![1.0, 0.0], vec![0.0, weak_derivative], vec![0.0, 0.0]]
                        })),
                        ..LeastSquaresOptions::default()
                    },
                    ..CurveFitOptions::default()
                };
                let fit = curve_fit(
                    |x, p| {
                        if x == 0.0 {
                            p[0]
                        } else if x == 1.0 {
                            weak_derivative * p[1]
                        } else {
                            0.0
                        }
                    },
                    &[0.0, 1.0, 2.0],
                    &[0.0, 0.0, 0.0],
                    options,
                )
                .expect("small singular values are discarded by SciPy's dimension-aware cutoff");
                assert!((fit.pcov[0][0] - 1.0).abs() < 1e-14);
                if weak_derivative <= 5e-16 {
                    assert_eq!(fit.pcov[1][1], 0.0);
                } else {
                    assert!((fit.pcov[1][1] / 1e28 - 1.0).abs() < 1e-14);
                }
                assert_eq!(fit.pcov[0][1], 0.0);
                assert_eq!(fit.pcov[1][0], 0.0);
            }
            let (fit, warnings) = fsci_runtime::catch_warnings(|| {
                curve_fit(
                    |_x, _p| 0.0,
                    &[0.0, 1.0, 2.0],
                    &[1.0, 2.0, 3.0],
                    CurveFitOptions {
                        p0: Some(vec![0.0, 0.0]),
                        absolute_sigma: true,
                        ls_options: LeastSquaresOptions {
                            method: Some(method),
                            jac: Some(LeastSquaresJacobian::new(|_| vec![vec![0.0; 2]; 3])),
                            ..LeastSquaresOptions::default()
                        },
                        ..CurveFitOptions::default()
                    },
                )
            });
            assert_eq!(
                fit.expect("zero-rank pseudoinverse").pcov,
                vec![vec![0.0; 2]; 2]
            );
            assert!(warnings.is_empty(), "{warnings:?}");
        }
    }

    #[test]
    fn curve_fit_trust_region_covariance_preserves_large_jacobian_units() {
        for method in [LeastSquaresMethod::Trf, LeastSquaresMethod::Dogbox] {
            for derivative_scale in [1e80, 1e160] {
                let fit = curve_fit(
                    |x, p| {
                        derivative_scale
                            * if x == 0.0 {
                                p[0] + p[1]
                            } else if x == 1.0 {
                                p[1]
                            } else {
                                p[0]
                            }
                    },
                    &[0.0, 1.0, 2.0],
                    &[0.0, 0.0, 0.0],
                    CurveFitOptions {
                        p0: Some(vec![0.0, 0.0]),
                        absolute_sigma: true,
                        ls_options: LeastSquaresOptions {
                            method: Some(method),
                            jac: Some(LeastSquaresJacobian::new(move |_| {
                                vec![
                                    vec![derivative_scale, derivative_scale],
                                    vec![0.0, derivative_scale],
                                    vec![derivative_scale, 0.0],
                                ]
                            })),
                            ..LeastSquaresOptions::default()
                        },
                        ..CurveFitOptions::default()
                    },
                )
                .expect("finite covariance despite large derivatives");
                for (i, row) in fit.pcov.iter().enumerate() {
                    for (j, &entry) in row.iter().enumerate() {
                        let expected = if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 };
                        let restored = entry * derivative_scale * derivative_scale;
                        // At scale 1e160 the covariance is subnormal, with ~4 decimal digits.
                        let tolerance = if derivative_scale > 1e100 {
                            5e-4
                        } else {
                            1e-14
                        };
                        assert!(
                            (restored - expected).abs() < tolerance,
                            "{method:?}: {:?}",
                            fit.pcov
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn curve_fit_warns_when_the_jacobian_is_singular() {
        // SciPy: curve_fit(lambda x, a, b: a*x + 0*b, [1, 2, 3], [2, 4, 6]) returns
        // popt [2, 1], pcov all inf, and warns OptimizeWarning. SciPy's default p0 is ones;
        // fsci cannot read a closure's arity, so it is passed.
        let model = |x: f64, p: &[f64]| p[0] * x + 0.0 * p[1];
        let (result, warnings) = fsci_runtime::catch_warnings(|| {
            curve_fit(
                model,
                &[1.0, 2.0, 3.0],
                &[2.0, 4.0, 6.0],
                CurveFitOptions {
                    p0: Some(vec![1.0, 1.0]),
                    ..CurveFitOptions::default()
                },
            )
        });
        let result = result.expect("fit");
        assert!((result.popt[0] - 2.0).abs() < 1e-10, "{:?}", result.popt);
        assert!(result.pcov.iter().flatten().all(|v| v.is_infinite()));
        let categories: Vec<_> = warnings.iter().map(|w| w.category).collect();
        assert_eq!(categories, [fsci_runtime::WarningCategory::OptimizeWarning]);
        assert_eq!(
            warnings[0].message,
            "Covariance of the parameters could not be estimated"
        );
    }

    #[test]
    fn curve_fit_zero_dof_returns_infinite_covariance() {
        let xdata = [0.0, 1.0];
        let ydata = [1.0, 3.0];
        let model = |x: f64, p: &[f64]| p[0] * x + p[1];

        let (result, warnings) = fsci_runtime::catch_warnings(|| {
            curve_fit(
                model,
                &xdata,
                &ydata,
                CurveFitOptions {
                    p0: Some(vec![1.0, 1.0]),
                    absolute_sigma: false,
                    ..CurveFitOptions::default()
                },
            )
        });
        let result = result.expect("should fit exactly");

        assert!(
            result
                .pcov
                .iter()
                .flatten()
                .all(|entry| entry.is_infinite())
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].category,
            fsci_runtime::WarningCategory::OptimizeWarning
        );
    }

    #[test]
    fn curve_fit_xdata_ydata_length_mismatch() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(
            model,
            &[1.0, 2.0],
            &[1.0],
            CurveFitOptions {
                p0: Some(vec![1.0]),
                ..CurveFitOptions::default()
            },
        )
        .expect_err("should reject mismatched lengths");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn curve_fit_requires_p0() {
        let model = |_x: f64, _p: &[f64]| 0.0;
        let err = curve_fit(model, &[1.0], &[1.0], CurveFitOptions::default())
            .expect_err("should require p0");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    #[test]
    fn least_squares_rejects_non_positive_diff_step() {
        let residuals = |x: &[f64]| vec![x[0] - 1.0];
        let err = least_squares(
            residuals,
            &[0.0],
            LeastSquaresOptions {
                diff_step: Some(0.0),
                ..LeastSquaresOptions::default()
            },
        )
        .expect_err("should reject zero diff_step");
        assert!(matches!(err, OptError::InvalidArgument { .. }));
    }

    /// A forward-difference step that crosses a domain edge puts a NaN in the Jacobian while the
    /// residuals stay finite. SciPy 1.17.1 on `[√(1 − x), x − 0.5]` from x0 = 1 − 1e-9 (default
    /// step √ε·max(1, |x|) lands past 1) raises `ValueError: array must not contain infs or NaNs`
    /// from trf's SVD, bounded or not; from x0 = 0.5 or 0.9 the step never crosses and SciPy runs
    /// to max_nfev (status 0) without raising. fsci used to fold the NaN out of ‖g·v‖∞, read 0,
    /// and report gtol convergence (status 1) at x0 (frankenscipy-qdb8s).
    #[test]
    fn least_squares_trf_refuses_a_nan_jacobian_like_scipy() {
        let residuals = |x: &[f64]| vec![(1.0 - x[0]).sqrt(), x[0] - 0.5];
        let edge = [1.0 - 1e-9];
        let unbounded = least_squares(residuals, &edge, LeastSquaresOptions::default());
        let bounded = least_squares_bounded(
            residuals,
            &edge,
            &[-10.0],
            &[10.0],
            LeastSquaresOptions::default(),
        );
        for result in [unbounded, bounded] {
            assert!(
                matches!(&result, Err(OptError::NonFiniteInput { detail })
                    if detail == "array must not contain infs or NaNs"),
                "SciPy raises here; fsci returned {result:?}"
            );
        }
        for x0 in [0.5, 0.9] {
            let r = least_squares(residuals, &[x0], LeastSquaresOptions::default())
                .expect("the step never crosses the edge, so SciPy does not raise");
            assert_eq!((r.status, r.success), (0, false), "x0 = {x0}: {r:?}");
        }
    }

    #[test]
    fn least_squares_bounded_rejects_unordered_bounds_before_evaluating() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let evaluated = AtomicBool::new(false);
        for (lower, upper) in [(f64::NAN, 1.0), (0.0, 0.0)] {
            let err = least_squares_bounded(
                |_| {
                    evaluated.store(true, Ordering::Relaxed);
                    Vec::new()
                },
                &[0.0],
                &[lower],
                &[upper],
                LeastSquaresOptions::default(),
            )
            .expect_err("unordered bounds must fail");
            assert!(matches!(err, OptError::InvalidArgument { .. }));
            assert!(!evaluated.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn least_squares_gaussian_fit() {
        // Fit y = A * exp(-((x - mu)/sigma)^2 / 2)
        let xdata: Vec<f64> = (-30..=30).map(|i| i as f64 * 0.2).collect();
        let ydata: Vec<f64> = xdata
            .iter()
            .map(|&x| 5.0 * (-((x - 1.0) / 2.0).powi(2) / 2.0).exp())
            .collect();

        let residuals = |p: &[f64]| -> Vec<f64> {
            xdata
                .iter()
                .zip(ydata.iter())
                .map(|(&x, &y)| p[0] * (-((x - p[1]) / p[2]).powi(2) / 2.0).exp() - y)
                .collect()
        };

        let result = least_squares(residuals, &[3.0, 0.0, 1.0], LeastSquaresOptions::default())
            .expect("should converge");
        assert!(result.success, "{}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.1,
            "A ~ 5.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "mu ~ 1.0, got {}",
            result.x[1]
        );
        assert!(
            (result.x[2].abs() - 2.0).abs() < 0.1,
            "|sigma| ~ 2.0, got {}",
            result.x[2]
        );
    }

    #[test]
    fn leastsq_matches_scipy_rosenbrock_and_linear() {
        // Rosenbrock residuals; minimum at (1, 1) — scipy.optimize.leastsq agrees.
        let r1 = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
        let res = leastsq(r1, &[-1.2, 1.0], LeastSquaresOptions::default()).unwrap();
        assert!((res.x[0] - 1.0).abs() < 1e-6, "x0 = {}", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 1e-6, "x1 = {}", res.x[1]);
        // A converged run reports a MINPACK success flag (1..=4).
        assert!((1..=4).contains(&res.ier), "ier = {}", res.ier);
        // cov_x is the unscaled (JᵀJ)⁻¹ and is finite for a full-rank Jacobian.
        let cov = res
            .cov_x
            .expect("cov_x should be Some for full-rank Jacobian");
        assert_eq!(cov.len(), 2);
        assert!(cov.iter().flatten().all(|v| v.is_finite()));

        // Overdetermined linear fit r_i = a*u_i + b - v_i; scipy gives a≈0.97714286.
        let us = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let vs = [0.1_f64, 0.9, 2.1, 2.9, 4.2, 4.8];
        let r3 = move |p: &[f64]| -> Vec<f64> {
            us.iter()
                .zip(vs.iter())
                .map(|(&u, &v)| p[0] * u + p[1] - v)
                .collect()
        };
        let lin = leastsq(r3, &[0.0, 0.0], LeastSquaresOptions::default()).unwrap();
        assert!(
            (lin.x[0] - 0.977_142_857).abs() < 1e-6,
            "slope = {}",
            lin.x[0]
        );
        assert!(
            (lin.x[1] - 0.057_142_857).abs() < 1e-6,
            "intercept = {}",
            lin.x[1]
        );
    }
}
