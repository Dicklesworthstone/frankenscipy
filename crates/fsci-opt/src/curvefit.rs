#![forbid(unsafe_code)]

//! Nonlinear least squares: `least_squares` (SciPy's default Trust Region Reflective method,
//! rectangular Dogbox, bounds, robust losses, or Levenberg–Marquardt), `curve_fit`, and the
//! batched variants.

use fsci_runtime::RuntimeMode;
use std::sync::Arc;

use crate::FiniteDiffMethod;
use crate::lsq_matrix::LeastSquaresMatrix;
use crate::trf::{self, Loss, LossKind, XScale};
use crate::trf_lsmr::{TrustRegionLsmrOptions, TrustRegionSolver};
use crate::types::OptError;
use fsci_sparse::{CsrMatrix, Shape2D};

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
    /// Number of iterations (accepted steps for `Trf` and `Dogbox`).
    pub nit: usize,
    /// Jacobian at the solution, retaining sparse storage when supplied by the
    /// caller or estimated using `jac_sparsity`. For a robust loss this is
    /// SciPy's loss-scaled Jacobian. Use [`LeastSquaresMatrix::to_dense`] when
    /// an explicitly materialized dense matrix is needed.
    pub jac: LeastSquaresMatrix,
    /// SciPy's status: 0 max_nfev, 1 gtol, 2 ftol, 3 xtol, 4 ftol and xtol.
    pub status: i32,
    /// First-order optimality: ‖Jᵀf‖∞, scaled by Coleman–Li for bounded `Trf`, or
    /// restricted to the free variables for `Dogbox`.
    pub optimality: f64,
    /// Gradient `Jᵀf` of the cost at the solution.
    pub grad: Vec<f64>,
    /// −1 / 1 where a lower / upper bound is active, 0 otherwise.
    pub active_mask: Vec<i8>,
}

/// An analytic residual Jacobian shared by individual and batched fits.
///
/// Rows correspond to residuals and columns to parameters. Capturing closures can own
/// model data; the callback is validated at every evaluation before a solver uses it.
/// A callback shared by [`least_squares_many`] must be valid for every parameter row;
/// the callback receives the fitted variables, not the batch's per-fit data.
#[derive(Clone)]
pub struct LeastSquaresJacobian {
    callback: Arc<dyn Fn(&[f64]) -> LeastSquaresMatrix + Send + Sync>,
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
            callback: Arc::new(move |x| LeastSquaresMatrix::Dense(callback(x))),
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
            callback: Arc::new(move |x| LeastSquaresMatrix::Sparse(callback(x))),
        }
    }

    fn evaluate(&self, x: &[f64], residual_count: usize) -> Result<LeastSquaresMatrix, OptError> {
        let matrix = (self.callback)(x);
        let valid_shape = match &matrix {
            LeastSquaresMatrix::Dense(rows) => {
                rows.len() == residual_count && rows.iter().all(|row| row.len() == x.len())
            }
            LeastSquaresMatrix::Sparse(csr) => csr.shape() == Shape2D::new(residual_count, x.len()),
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
    /// Jacobians and `Lsmr` for sparse ones. An explicit `Exact` with a sparse
    /// Jacobian is rejected. LM does not accept the LSMR backend.
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
    /// near a bound. Applies to `Trf`, `Dogbox`, and `Lm`.
    pub diff_method: FiniteDiffMethod,
    /// Relative finite-difference step. For `Trf`/`Dogbox` and three-point `Lm`,
    /// `None` selects SciPy's `ε^(1/2)` (two-point) or `ε^(1/3)` (three-point)
    /// relative step. Two-point `Lm` retains its `√ε·(1 + |x|)` step.
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
    /// Soft margin between inlier and outlier residuals for a robust loss.
    pub f_scale: f64,
    /// Parameter scaling: fixed positive scales or `'jac'`. `None` uses unit
    /// scale for `Trf`/`Dogbox` and Jacobian column scaling for `Lm`, as in SciPy.
    pub x_scale: Option<XScale>,
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
            f_scale: 1.0,
            x_scale: None,
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
}

/// Prepared measurement-error transform, shared by every fit in a batch.
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

    fn jacobian_matrix(&self, matrix: LeastSquaresMatrix) -> LeastSquaresMatrix {
        match (self, matrix) {
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
                let mut dense = matrix.to_dense();
                self.jacobian(&mut dense);
                LeastSquaresMatrix::Dense(dense)
            }
        }
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
    if options.loss != LossKind::Linear {
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
    let mut scaling = LmScaling::new(options.x_scale.as_ref(), n)?;
    let max_nfev = options.max_nfev.unwrap_or(100 * n);

    let mut x = x0.to_vec();
    let mut r = residuals(&x);
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
    let mut x_perturbed = x.clone();
    lm_jacobian_into(&residuals, &x, &r, &options, &mut jac, &mut x_perturbed)?;
    njev += 1;
    scaling.update(&jac)?;
    let mut scaled_jac = scaling.jacobian(&jac)?;
    // Solve in scaled parameter coordinates. Cache the linear model until an
    // accepted step changes residuals, derivatives, or adaptive column scales.
    let mut jtj = Vec::new();
    let mut jtr = Vec::new();
    jtj_matrix_into(&scaled_jac, &mut jtj);
    jt_vec_into(&scaled_jac, &r, &mut jtr);
    let mut max_cosine = lm_max_cosine(&jac, &r);
    let mut damped_normal = Vec::new();
    let mut cholesky_low = Vec::new();
    let mut solve_y = Vec::new();
    let mut step = Vec::new();
    let mut jstep = Vec::new();

    // Initial damping parameter (Marquardt strategy)
    let mut mu = 1.0e-3 * max_diag_jtj(&scaled_jac);
    if mu == 0.0 {
        mu = 1.0;
    }
    let mut nu = 2.0;
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

        // Solve (J^T J + mu I) * step = -J^T r
        solve_damped_normal_equations_into(
            &jtj,
            &jtr,
            mu,
            n,
            DampedNormalScratch {
                a: &mut damped_normal,
                low: &mut cholesky_low,
                y: &mut solve_y,
                step: &mut step,
            },
        );

        let step_norm = step.iter().fold(0.0_f64, |norm, &value| norm.hypot(value));
        let x_norm = scaling.parameter_norm(&x);
        if !step_norm.is_finite() || !x_norm.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("scaled LM step or parameter norm is not finite"),
            });
        }

        // Use the same parameter metric as the damped solve. An additive
        // xtol² floor would prematurely stop small scaled parameter steps.
        if step_norm <= options.xtol * x_norm {
            return Ok(lm_result(
                x,
                r,
                cost,
                3,
                String::from("parameter change converged (xtol)"),
                nfev,
                njev,
                nit,
                jac,
            ));
        }

        // Trial step
        let physical_step = scaling.physical_step(&step);
        let x_new: Vec<f64> = x.iter().zip(&physical_step).map(|(a, b)| a + b).collect();
        let r_new = residuals(&x_new);
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
            mat_vec_into(&scaled_jac, &step, &mut jstep);
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

        if predicted_reduction > 0.0 {
            let rho = actual_reduction / predicted_reduction;

            if rho > 0.25 {
                // Accept step
                x = x_new;
                r = r_new;
                cost = cost_new;
                residual_norm = trial_norm;
                lm_jacobian_into(&residuals, &x, &r, &options, &mut jac, &mut x_perturbed)?;
                njev += 1;

                // LM requires small actual and predicted relative reductions;
                // the accepted step also establishes adequate model agreement.
                if actual_reduction <= options.ftol && predicted_reduction <= options.ftol {
                    return Ok(lm_result(
                        x,
                        r,
                        cost,
                        2,
                        String::from("cost change converged (ftol)"),
                        nfev,
                        njev,
                        nit,
                        jac,
                    ));
                }

                // Refresh the scaled model only after an accepted step.
                scaling.update(&jac)?;
                scaled_jac = scaling.jacobian(&jac)?;
                jtj_matrix_into(&scaled_jac, &mut jtj);
                jt_vec_into(&scaled_jac, &r, &mut jtr);
                max_cosine = lm_max_cosine(&jac, &r);

                // Decrease damping
                mu *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
            } else {
                // Reject step, increase damping
                mu *= nu;
                nu *= 2.0;
            }
        } else {
            // Reject step, increase damping
            mu *= nu;
            nu *= 2.0;
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

/// Bound-aware points shared by dense and grouped numerical Jacobians. The
/// boolean marks a one-sided stencil; the points are actual representable
/// coordinates, so the denominator includes rounding of the perturbation.
fn fd_points_scipy(
    x: &[f64],
    lb: &[f64],
    ub: &[f64],
    method: FiniteDiffMethod,
    rel_step: Option<f64>,
) -> Vec<(f64, f64, bool)> {
    let n = x.len();
    let unbounded =
        lb.iter().all(|v| *v == f64::NEG_INFINITY) && ub.iter().all(|v| *v == f64::INFINITY);
    let mut points = Vec::with_capacity(n);
    let relative_epsilon = method.rel_eps();
    for i in 0..n {
        let sign = if x[i] >= 0.0 { 1.0 } else { -1.0 };
        let default_step = relative_epsilon * sign * x[i].abs().max(1.0);
        let mut h = match rel_step {
            None => default_step,
            Some(r) => {
                let h = r * sign * x[i].abs();
                if (x[i] + h) - x[i] == 0.0 {
                    default_step
                } else {
                    h
                }
            }
        };
        let mut one_sided = method == FiniteDiffMethod::TwoPoint;
        if method == FiniteDiffMethod::ThreePoint {
            h = h.abs();
        }
        if !unbounded {
            let lower = x[i] - lb[i];
            let upper = ub[i] - x[i];
            if method == FiniteDiffMethod::TwoPoint {
                let trial = x[i] + h;
                let violated = trial < lb[i] || trial > ub[i];
                let fitting = h.abs() <= lower.max(upper);
                if violated && fitting {
                    h = -h;
                } else if !fitting {
                    h = if upper >= lower { upper } else { -lower };
                }
            } else if lower < h || upper < h {
                // A three-point one-sided stencil needs two steps into the box.
                // If shortening that step also permits a central stencil, use
                // the latter: it avoids cancellation in -3*f0 + 4*f1 - f2.
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
        }
        let first = if one_sided { x[i] + h } else { x[i] - h };
        let second = if one_sided { x[i] + 2.0 * h } else { x[i] + h };
        points.push((first, second, one_sided));
    }
    points
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
    let points = fd_points_scipy(x, lb, ub, method, rel_step);
    let mut columns = Vec::with_capacity(x.len());
    let mut x1 = x.to_vec();
    let mut x2 = x.to_vec();
    for (i, &(first, second, one_sided)) in points.iter().enumerate() {
        x1[i] = first;
        let f1 = residuals(&x1);
        check_jacobian_residual_shape(&f1, f0.len())?;
        let column: Vec<f64> = if method == FiniteDiffMethod::TwoPoint {
            let dx = x1[i] - x[i];
            f1.iter().zip(f0).map(|(a, b)| (a - b) / dx).collect()
        } else {
            x2[i] = second;
            let f2 = residuals(&x2);
            check_jacobian_residual_shape(&f2, f0.len())?;
            let dx = if one_sided {
                x2[i] - x[i]
            } else {
                x2[i] - x1[i]
            };
            f0.iter()
                .zip(&f1)
                .zip(&f2)
                .map(|((f0, f1), f2)| {
                    if one_sided {
                        (-3.0 * f0 + 4.0 * f1 - f2) / dx
                    } else {
                        (f2 - f1) / dx
                    }
                })
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
    let points = fd_points_scipy(x, lb, ub, method, rel_step);
    let mut data = vec![0.0; sparsity.nnz()];
    let mut x1 = x.to_vec();
    let mut x2 = x.to_vec();
    for group in &sparsity.groups {
        for &c in group {
            x1[c] = points[c].0;
            x2[c] = points[c].1;
        }
        let f1 = residuals(&x1);
        check_jacobian_residual_shape(&f1, f0.len())?;
        if method == FiniteDiffMethod::TwoPoint {
            for &c in group {
                let dx = x1[c] - x[c];
                for &(r, entry) in &sparsity.column_entries[c] {
                    data[entry] = (f1[r] - f0[r]) / dx;
                }
            }
        } else {
            let f2 = residuals(&x2);
            check_jacobian_residual_shape(&f2, f0.len())?;
            for &c in group {
                let one_sided = points[c].2;
                let dx = if one_sided {
                    x2[c] - x[c]
                } else {
                    x2[c] - x1[c]
                };
                for &(r, entry) in &sparsity.column_entries[c] {
                    data[entry] = if one_sided {
                        (-3.0 * f0[r] + 4.0 * f1[r] - f2[r]) / dx
                    } else {
                        (f2[r] - f1[r]) / dx
                    };
                }
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
            LeastSquaresMatrix::Sparse(_) => {
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
            options.diff_step.unwrap_or(SQRT_EPS),
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
    let solver = match options.tr_solver {
        TrustRegionSolver::Auto if j0.is_sparse() => {
            TrustRegionSolver::Lsmr(TrustRegionLsmrOptions::default())
        }
        TrustRegionSolver::Auto => TrustRegionSolver::Exact,
        TrustRegionSolver::Exact if j0.is_sparse() => {
            return Err(OptError::InvalidArgument {
                detail: String::from("tr_solver='exact' requires a dense Jacobian"),
            });
        }
        solver => solver,
    };
    let loss = (options.loss != LossKind::Linear).then_some(Loss {
        kind: options.loss,
        f_scale: options.f_scale,
    });
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
        if solver == TrustRegionSolver::Exact && matrix.is_sparse() {
            return Err(OptError::InvalidArgument {
                detail: String::from(
                    "tr_solver='exact' requires a dense Jacobian at every evaluation",
                ),
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
    )?;
    Ok(LeastSquaresResult {
        // status: SciPy's code; success = status > 0 (a gtol/ftol/xtol test passed)
        success: out.status > 0,
        message: termination_message(out.status).to_string(),
        nit: out.njev.saturating_sub(1),
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
    options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let whitening = Arc::new(FitWhitening::new(options.sigma.as_ref(), xdata.len())?);
    curve_fit_impl(&f, xdata, ydata, None, options, &whitening)
}

/// Shared scalar/batched and bounded/unbounded curve fitting. `whitening` has
/// already been validated and factored exactly once for this observation grid.
fn curve_fit_impl<F>(
    f: &F,
    xdata: &[f64],
    ydata: &[f64],
    bounds: Option<(&[f64], &[f64])>,
    options: CurveFitOptions,
    whitening: &Arc<FitWhitening>,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    if xdata.len() != ydata.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "xdata and ydata must have same length (got {} and {})",
                xdata.len(),
                ydata.len()
            ),
        });
    }
    if xdata.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xdata must not be empty"),
        });
    }

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
        ls_options.jac_sparsity = Some(whitening.sparsity(pattern)?);
    }
    if !matches!(whitening.as_ref(), FitWhitening::Identity)
        && let Some(jacobian) = ls_options.jac.take()
    {
        let whitening = Arc::clone(whitening);
        let observations = xdata.len();
        ls_options.jac = Some(LeastSquaresJacobian {
            callback: Arc::new(move |params| {
                let matrix = (jacobian.callback)(params);
                // Leave malformed matrices intact for LeastSquaresJacobian::evaluate
                // to report its normal shape error, before indexing any row here.
                let valid_shape = match &matrix {
                    LeastSquaresMatrix::Dense(rows) => {
                        rows.len() == observations
                            && rows.iter().all(|row| row.len() == params.len())
                    }
                    LeastSquaresMatrix::Sparse(csr) => {
                        csr.shape() == Shape2D::new(observations, params.len())
                    }
                };
                if valid_shape {
                    whitening.jacobian_matrix(matrix)
                } else {
                    matrix
                }
            }),
        });
    }
    let ls_result = match bounds {
        Some((lower, upper)) => least_squares_bounded(residuals, &p0, lower, upper, ls_options)?,
        None => least_squares(residuals, &p0, ls_options)?,
    };
    require_fit_converged(&ls_result)?;

    let pcov = curve_fit_covariance(&ls_result, method, options.absolute_sigma);
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
    options: CurveFitOptions,
) -> Result<CurveFitResult, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let whitening = Arc::new(FitWhitening::new(options.sigma.as_ref(), xdata.len())?);
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
    options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let whitening = Arc::new(FitWhitening::new(options.sigma.as_ref(), xdata.len())?);
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
    options: CurveFitOptions,
) -> Result<Vec<Vec<f64>>, OptError>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = ydata_rows.len();
    if nrows == 0 {
        return Ok(Vec::new());
    }
    let f_ref = &f;
    let whitening = Arc::new(FitWhitening::new(options.sigma.as_ref(), xdata.len())?);
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
    let m = ls.fun.len();

    // scipy's `cov_x` is the *unscaled* inverse (JᵀJ)⁻¹; reuse the
    // `absolute_sigma` branch of the covariance helper. A rank-deficient
    // Jacobian yields a non-finite inverse -> report `None` like scipy.
    let jac = ls.jac.as_dense().ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("leastsq requires a dense LM Jacobian"),
    })?;
    let cov = compute_covariance(jac, ls.cost, m, n, true);
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

/// Compute finite-difference Jacobian into fixed-shape scratch.
fn finite_diff_jacobian_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    eps: f64,
    jac: &mut Vec<Vec<f64>>,
    x_perturbed: &mut Vec<f64>,
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = r0.len();
    if jac.len() != m {
        jac.resize_with(m, Vec::new);
    }
    for row in jac.iter_mut() {
        row.resize(n, 0.0);
    }
    x_perturbed.resize(n, 0.0);
    x_perturbed.copy_from_slice(x);

    for j in 0..n {
        let step = eps * (1.0 + x[j].abs());
        let original = x_perturbed[j];
        x_perturbed[j] += step;
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
/// per-column `step = eps*(1+|x[j]|)` and the same one-sided difference).
///
/// Parallelism pays only when `n_params * m_data * cost_per_eval` is large enough
/// to amortize thread spawn (measured: a hard loss below ~30K work, 2.75–13.66x
/// above it), so below the work-gate this falls back to the serial routine — which
/// also keeps `available_parallelism()` (a syscall) off the small-problem hot path.
fn finite_diff_jacobian_parallel_into<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    eps: f64,
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
        return finite_diff_jacobian_into(residuals, x, r0, eps, jac, x_perturbed);
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(n);
    if nthreads <= 1 {
        return finite_diff_jacobian_into(residuals, x, r0, eps, jac, x_perturbed);
    }

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
                Some(scope.spawn(move || {
                    let mut xp = x.to_vec();
                    (lo..hi)
                        .map(|j| {
                            let step = eps * (1.0 + x[j].abs());
                            let original = xp[j];
                            xp[j] += step;
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

/// Compute J^T * J (n x n).
fn jtj_matrix(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut jtj = Vec::new();
    jtj_matrix_into(jac, &mut jtj);
    jtj
}

/// Compute J^T * J (n x n) into fixed-shape scratch.
fn jtj_matrix_into(jac: &[Vec<f64>], jtj: &mut Vec<Vec<f64>>) {
    let n = jac.first().map_or(0, Vec::len);
    resize_square(jtj, n);
    for row in jac {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
        }
    }
}

/// Compute J^T * v into fixed-shape scratch.
fn jt_vec_into(jac: &[Vec<f64>], v: &[f64], result: &mut Vec<f64>) {
    let n = jac.first().map_or(0, Vec::len);
    result.resize(n, 0.0);
    result.fill(0.0);
    for (row, &vi) in jac.iter().zip(v.iter()) {
        for (j, &jval) in row.iter().enumerate() {
            result[j] += jval * vi;
        }
    }
}

/// Compute J * v into fixed-shape scratch.
fn mat_vec_into(jac: &[Vec<f64>], v: &[f64], result: &mut Vec<f64>) {
    result.resize(jac.len(), 0.0);
    for (out, row) in result.iter_mut().zip(jac.iter()) {
        *out = dot_vec(row, v);
    }
}

/// Max diagonal element of J^T * J.
fn max_diag_jtj(jac: &[Vec<f64>]) -> f64 {
    let n = jac.first().map_or(0, Vec::len);
    let mut max_val = 0.0_f64;
    for j in 0..n {
        let mut diag = 0.0;
        for row in jac {
            diag += row[j] * row[j];
        }
        if max_val.is_nan() || diag.is_nan() {
            max_val = f64::NAN;
        } else {
            max_val = max_val.max(diag);
        }
    }
    max_val
}

/// Solve (A + mu * I) * x = -b into fixed-shape scratch.
/// Falls back to diagonal if matrix is singular.
struct DampedNormalScratch<'a> {
    a: &'a mut Vec<Vec<f64>>,
    low: &'a mut Vec<Vec<f64>>,
    y: &'a mut Vec<f64>,
    step: &'a mut Vec<f64>,
}

fn solve_damped_normal_equations_into(
    jtj: &[Vec<f64>],
    jtr: &[f64],
    mu: f64,
    n: usize,
    scratch: DampedNormalScratch<'_>,
) {
    resize_square(scratch.a, n);
    for (dst_row, src_row) in scratch.a.iter_mut().zip(jtj.iter()) {
        dst_row.copy_from_slice(src_row);
    }
    for (i, row) in scratch.a.iter_mut().enumerate() {
        row[i] += mu;
    }

    // Solve via Cholesky decomposition
    if cholesky_decompose_into(scratch.a, n, scratch.low) {
        cholesky_solve_with_l_into(scratch.low, jtr, n, scratch.y, scratch.step);
        // Return -step (we solve A*step = -jtr, but passed jtr not -jtr).
        for value in scratch.step.iter_mut() {
            *value = -*value;
        }
    } else {
        // Fallback: diagonal solve
        scratch.step.resize(n, 0.0);
        for i in 0..n {
            scratch.step[i] = -jtr[i] / (jtj[i][i] + mu);
        }
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

/// Solve L * L^T * x = b given L.
fn cholesky_solve_with_l(low: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = Vec::new();
    let mut x = Vec::new();
    cholesky_solve_with_l_into(low, b, n, &mut y, &mut x);
    x
}

/// Solve L * L^T * x = b given L into fixed-shape scratch.
fn cholesky_solve_with_l_into(
    low: &[Vec<f64>],
    b: &[f64],
    n: usize,
    y: &mut Vec<f64>,
    x: &mut Vec<f64>,
) {
    // Forward substitution: L * y = b
    y.resize(n, 0.0);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= low[i][j] * y[j];
        }
        y[i] = sum / low[i][i];
    }

    // Back substitution: L^T * x = y
    x.resize(n, 0.0);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= low[j][i] * x[j];
        }
        x[i] = sum / low[i][i];
    }
}

/// SciPy's curve-fit covariance: LM retains its inverse, while TRF and Dogbox use
/// the SVD pseudoinverse of JᵀJ without forming the condition-squaring normal matrix.
fn curve_fit_covariance(
    result: &LeastSquaresResult,
    method: LeastSquaresMethod,
    absolute_sigma: bool,
) -> Vec<Vec<f64>> {
    let m = result.fun.len();
    let n = result.x.len();
    if !absolute_sigma && m <= n {
        return vec![vec![f64::INFINITY; n]; n];
    }
    if !result.jac.all_finite() {
        return vec![vec![f64::INFINITY; n]; n];
    }

    // curve_fit returns a full parameter covariance and explicitly performs a
    // dense SVD here. Existing dense Jacobians are borrowed without copying;
    // least_squares itself retains CSR storage end to end.
    let dense_storage;
    let jac = match result.jac.as_dense() {
        Some(matrix) => matrix,
        None => {
            dense_storage = result.jac.to_dense();
            &dense_storage
        }
    };
    if method == LeastSquaresMethod::Lm {
        return compute_covariance(jac, result.cost, m, n, absolute_sigma);
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
        return vec![vec![f64::INFINITY; n]; n];
    }
    if !absolute_sigma {
        let residual_variance = 2.0 * result.cost / (m - n) as f64;
        for entry in covariance.iter_mut().flatten() {
            *entry *= residual_variance;
        }
    }
    covariance
}

/// Compute the existing LM parameter covariance matrix from its Jacobian.
fn compute_covariance(
    jac: &[Vec<f64>],
    cost: f64,
    m: usize,
    n: usize,
    absolute_sigma: bool,
) -> Vec<Vec<f64>> {
    if !absolute_sigma && m <= n {
        return vec![vec![f64::INFINITY; n]; n];
    }

    let jtj = jtj_matrix(jac);

    // Invert J^T J via Cholesky
    let mut pcov = vec![vec![0.0; n]; n];
    if let Some(low) = cholesky_decompose(&jtj, n) {
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;
            let column = cholesky_solve_with_l(&low, &e, n);
            for (row, val) in column.iter().enumerate() {
                pcov[row][col] = *val;
            }
        }
    } else {
        // If J^T J is singular, return infinity on diagonal
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val = f64::INFINITY;
            }
        }
        return pcov;
    }

    // Scale by residual variance if not absolute_sigma
    if !absolute_sigma && m > n {
        let s2 = 2.0 * cost / (m - n) as f64;
        for row in &mut pcov {
            for val in row.iter_mut() {
                *val *= s2;
            }
        }
    }

    pcov
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let eps = 1.4901161193847656e-8;

        let mut jac_serial = Vec::new();
        let mut scratch_s = Vec::new();
        finite_diff_jacobian_into(
            &residuals,
            &params,
            &r0,
            eps,
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
            eps,
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
            let slopes = vec![1.0, 2.0, 4.0];
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
            assert_eq!(result.jac.to_dense(), vec![vec![1.0], vec![2.0], vec![4.0]]);
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
                assert!((result.jac.to_dense()[i][i] - exact).abs() < 2e-9);
            }
            assert_eq!(
                result.nfev, 1,
                "numerical Jacobian calls are counted separately"
            );
            assert_eq!(result.njev, 1);
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
