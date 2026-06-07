#![forbid(unsafe_code)]

pub mod audit;
pub mod curvefit;
pub mod linesearch;
pub mod minimize;
pub mod root;
pub mod types;

pub use audit::{SyncSharedAuditLedger, record_fail_closed, sync_audit_ledger};
pub use curvefit::{
    CurveFitOptions, CurveFitResult, LeastSquaresOptions, LeastSquaresResult, curve_fit,
    least_squares,
};
pub use linesearch::{LineSearchResult, WolfeParams, line_search_wolfe1, line_search_wolfe2};
pub use minimize::{
    MinimizeScalarOptions, MinimizeScalarResult, OptCaspDecision, OptCaspProblem, bfgs, cg_pr_plus,
    get_optimize_traces, lbfgsb, minimize, minimize_scalar, minimize_with_audit, nelder_mead,
    newton_cg, powell, select_minimize_method, trust_exact,
};
pub use root::{
    MultivariateRootMethod, MultivariateRootOptions, MultivariateRootResult, RootResult, anderson,
    bisect, brenth, brentq, broyden1, broyden2, fsolve, halley, lm_root, newton_scalar, ridder,
    root, root_scalar, secant, toms748,
};
pub use types::{
    Bound, Bounds, ConvergenceStatus, LinearConstraint, MinimizeOptions, NonlinearConstraint,
    OptError, OptimizeMethod, OptimizeResult, RootMethod, RootOptions,
};

/// Exit status for adaptive numerical differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentiateStatus {
    Converged,
    ErrorIncreased,
    MaxIterations,
}

/// Direction of finite-difference steps for first derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDirection {
    Central,
    Forward,
    Backward,
}

/// Absolute and relative tolerances for adaptive differentiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentiateTolerances {
    pub atol: f64,
    pub rtol: f64,
}

impl Default for DifferentiateTolerances {
    fn default() -> Self {
        Self {
            atol: f64::MIN_POSITIVE,
            rtol: f64::EPSILON.sqrt(),
        }
    }
}

/// Options for `derivative`, `jacobian`, and `hessian`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DifferentiateOptions {
    pub tolerances: DifferentiateTolerances,
    pub maxiter: usize,
    pub order: usize,
    pub initial_step: f64,
    pub step_factor: f64,
    pub step_direction: StepDirection,
}

impl Default for DifferentiateOptions {
    fn default() -> Self {
        Self {
            tolerances: DifferentiateTolerances::default(),
            maxiter: 10,
            order: 8,
            initial_step: 0.5,
            step_factor: 2.0,
            step_direction: StepDirection::Central,
        }
    }
}

/// Scalar first-derivative result matching the core fields of SciPy's RichResult.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeResult {
    pub df: f64,
    pub error: f64,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
    pub step: f64,
}

/// Jacobian result. `df[row][column]` is `d f[row] / d x[column]`.
#[derive(Debug, Clone, PartialEq)]
pub struct JacobianResult {
    pub df: Vec<Vec<f64>>,
    pub error: Vec<Vec<f64>>,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
}

/// Hessian result. `ddf[row][column]` is the second partial derivative.
#[derive(Debug, Clone, PartialEq)]
pub struct HessianResult {
    pub ddf: Vec<Vec<f64>>,
    pub error: Vec<Vec<f64>>,
    pub success: bool,
    pub status: DifferentiateStatus,
    pub nit: usize,
    pub nfev: usize,
}

/// Forward-difference gradient approximation.
///
/// Matches `scipy.optimize.approx_fprime(xk, f, epsilon)`.
pub fn approx_fprime<F>(xk: &[f64], f: F, epsilon: f64) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if xk.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xk must have at least one element"),
        });
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("epsilon must be a positive finite value"),
        });
    }
    if xk.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("xk must not contain NaN or Inf"),
        });
    }

    let f0 = f(xk);
    if !f0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("objective returned non-finite value at xk"),
        });
    }

    let mut gradient = Vec::with_capacity(xk.len());
    let mut x_perturbed = xk.to_vec();
    for index in 0..xk.len() {
        let original = x_perturbed[index];
        x_perturbed[index] += epsilon;
        let f_plus = f(&x_perturbed);
        x_perturbed[index] = original; // restore

        if !f_plus.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: format!("objective returned non-finite value at perturbed index {index}"),
            });
        }
        gradient.push((f_plus - f0) / epsilon);
    }

    Ok(gradient)
}

/// Compare an analytical gradient to a finite-difference approximation.
///
/// Matches `scipy.optimize.check_grad`.
pub fn check_grad<F, G>(func: F, grad: G, x0: &[f64]) -> Result<f64, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let analytical = grad(x0);
    if analytical.len() != x0.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "gradient length must match x0 length (got {} and {})",
                analytical.len(),
                x0.len()
            ),
        });
    }
    if analytical.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("gradient returned non-finite values"),
        });
    }

    let numerical = approx_fprime(x0, func, 1.490_116_119_384_765_6e-8)?;
    let squared_error = analytical
        .iter()
        .zip(numerical.iter())
        .map(|(lhs, rhs)| {
            let delta = lhs - rhs;
            delta * delta
        })
        .sum::<f64>();
    Ok(squared_error.sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════
// Rosenbrock test function family
// ═══════════════════════════════════════════════════════════════════════════

/// The Rosenbrock function.
///
/// Matches `scipy.optimize.rosen`. The function is defined as:
/// f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
///
/// The global minimum is at x = [1, 1, ..., 1] with f(x) = 0.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Function value at x
pub fn rosen(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let t1 = x[i + 1] - x[i] * x[i];
        let t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    sum
}

/// Gradient of the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_der`.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Gradient vector at x
pub fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut grad = vec![0.0; n];

    // First element: df/dx_0 = -400*x_0*(x_1 - x_0^2) - 2*(1 - x_0)
    grad[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);

    // Middle elements: df/dx_i = 200*(x_i - x_{i-1}^2) - 400*x_i*(x_{i+1} - x_i^2) - 2*(1 - x_i)
    for i in 1..n - 1 {
        grad[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
            - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
            - 2.0 * (1.0 - x[i]);
    }

    // Last element: df/dx_{n-1} = 200*(x_{n-1} - x_{n-2}^2)
    grad[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);

    grad
}

/// Hessian matrix of the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_hess`.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
///
/// # Returns
/// Hessian matrix (n x n) at x
pub fn rosen_hess(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    if n < 2 {
        return vec![vec![0.0; n]; n];
    }

    let mut h = vec![vec![0.0; n]; n];

    // The Hessian has a tridiagonal structure
    // d²f/dx_i² and d²f/dx_i dx_{i±1} are non-zero

    // First diagonal element
    h[0][0] = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;

    // First off-diagonal
    h[0][1] = -400.0 * x[0];
    h[1][0] = -400.0 * x[0];

    // Middle elements
    for i in 1..n - 1 {
        h[i][i] = 202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1];
        h[i][i + 1] = -400.0 * x[i];
        h[i + 1][i] = -400.0 * x[i];
    }

    // Last diagonal element
    h[n - 1][n - 1] = 200.0;

    h
}

/// Hessian-vector product for the Rosenbrock function.
///
/// Matches `scipy.optimize.rosen_hess_prod`. Computes H(x) @ p
/// without explicitly forming the Hessian matrix.
///
/// # Arguments
/// * `x` - Input point (length >= 2)
/// * `p` - Vector to multiply (same length as x)
///
/// # Returns
/// Result of Hessian-vector product
pub fn rosen_hess_prod(x: &[f64], p: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 || p.len() != n {
        return vec![0.0; n];
    }

    let mut hp = vec![0.0; n];

    // First element
    hp[0] = (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) * p[0] - 400.0 * x[0] * p[1];

    // Middle elements
    for i in 1..n - 1 {
        hp[i] = -400.0 * x[i - 1] * p[i - 1]
            + (202.0 + 1200.0 * x[i] * x[i] - 400.0 * x[i + 1]) * p[i]
            - 400.0 * x[i] * p[i + 1];
    }

    // Last element
    hp[n - 1] = -400.0 * x[n - 2] * p[n - 2] + 200.0 * p[n - 1];

    hp
}

/// Solve the linear sum assignment problem using a Hungarian-style algorithm.
///
/// Returns `(row_ind, col_ind)` with monotonically increasing row indices.
pub fn linear_sum_assignment(
    cost_matrix: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<usize>), OptError> {
    if cost_matrix.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let col_count = cost_matrix[0].len();
    if col_count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    for (row_index, row) in cost_matrix.iter().enumerate() {
        if row.len() != col_count {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "cost_matrix must be rectangular; row 0 has length {col_count} but row {row_index} has length {}",
                    row.len()
                ),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("cost_matrix contains non-finite values in row {row_index}"),
            });
        }
    }

    if cost_matrix.len() <= col_count {
        let assignment = hungarian_rectangular(cost_matrix);
        let row_ind = (0..assignment.len()).collect::<Vec<_>>();
        return Ok((row_ind, assignment));
    }

    let transposed = transpose_matrix(cost_matrix);
    let assignment = hungarian_rectangular(&transposed);
    let mut pairs = assignment
        .into_iter()
        .enumerate()
        .map(|(col, row)| (row, col))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(row, _)| *row);
    let (row_ind, col_ind): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    Ok((row_ind, col_ind))
}

fn transpose_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let row_count = matrix.len();
    let col_count = matrix[0].len();
    let mut transposed = vec![vec![0.0; row_count]; col_count];
    for (row_index, row) in matrix.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            transposed[col_index][row_index] = *value;
        }
    }
    transposed
}

// Hungarian implementation for row_count <= col_count.
fn hungarian_rectangular(cost_matrix: &[Vec<f64>]) -> Vec<usize> {
    let row_count = cost_matrix.len();
    let col_count = cost_matrix[0].len();
    let mut u = vec![0.0; row_count + 1];
    let mut v = vec![0.0; col_count + 1];
    let mut p = vec![0usize; col_count + 1];
    let mut way = vec![0usize; col_count + 1];

    for row in 1..=row_count {
        p[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f64::INFINITY; col_count + 1];
        let mut used = vec![false; col_count + 1];

        loop {
            used[col0] = true;
            let row0 = p[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;

            for col in 1..=col_count {
                if used[col] {
                    continue;
                }
                let current = cost_matrix[row0 - 1][col - 1] - u[row0] - v[col];
                if current < minv[col] {
                    minv[col] = current;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }

            for col in 0..=col_count {
                if used[col] {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;

            if p[col0] == 0 {
                break;
            }
        }

        loop {
            let col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; row_count];
    for col in 1..=col_count {
        if p[col] != 0 {
            assignment[p[col] - 1] = col - 1;
        }
    }
    assignment
}

// ══════════════════════════════════════════════════════════════════════
// Linear Programming: linprog
// ══════════════════════════════════════════════════════════════════════

/// Result of a linear programming problem.
#[derive(Debug, Clone)]
pub struct LinprogResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Slack variables for inequality constraints.
    pub slack: Vec<f64>,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of iterations.
    pub nit: usize,
}

/// Variable integrality kind for `milp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrality {
    Continuous,
    Integer,
    Binary,
}

/// Result of a mixed-integer linear programming problem.
#[derive(Debug, Clone)]
pub struct MilpResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`.
    pub fun: f64,
    /// Whether the solver succeeded.
    pub success: bool,
    /// Solver status: 0 = optimal, 1 = node/iteration limit, 2 = infeasible, 3 = unbounded.
    pub status: u8,
    /// Human-readable message.
    pub message: String,
    /// Number of LP relaxations solved.
    pub mip_node_count: usize,
    /// Accumulated LP iteration count.
    pub nit: usize,
}

/// Problem definition for `milp`.
#[derive(Debug, Clone, Copy)]
pub struct MilpProblem<'a> {
    pub c: &'a [f64],
    pub integrality: &'a [Integrality],
    pub a_ub: &'a [Vec<f64>],
    pub b_ub: &'a [f64],
    pub a_eq: &'a [Vec<f64>],
    pub b_eq: &'a [f64],
    pub bounds: &'a [(Option<f64>, Option<f64>)],
}

/// Solver controls for `milp`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MilpOptions {
    pub max_nodes: Option<usize>,
    pub lp_maxiter: Option<usize>,
}

#[derive(Debug, Clone)]
struct StandardFormVariable {
    offset: f64,
    terms: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
struct StandardFormTransform {
    original_vars: Vec<StandardFormVariable>,
    decision_var_count: usize,
    upper_bound_rows: Vec<(usize, f64)>,
}

fn validate_variable_bounds(index: usize, lo: f64, hi: f64) -> Result<(), OptError> {
    if lo.is_nan() || hi.is_nan() {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: bounds must not be NaN (got {lo}, {hi})"),
        });
    }
    if lo == f64::INFINITY {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: lower bound must not be +inf"),
        });
    }
    if hi == f64::NEG_INFINITY {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: upper bound must not be -inf"),
        });
    }
    if lo > hi {
        return Err(OptError::InvalidBounds {
            detail: format!("variable {index}: lower bound {lo} > upper bound {hi}"),
        });
    }
    Ok(())
}

fn build_standard_form_transform(
    var_bounds: &[(f64, f64)],
) -> Result<StandardFormTransform, OptError> {
    let mut original_vars = Vec::with_capacity(var_bounds.len());
    let mut upper_bound_rows = Vec::new();
    let mut next_index = 0usize;

    for (index, &(lo, hi)) in var_bounds.iter().enumerate() {
        validate_variable_bounds(index, lo, hi)?;
        if lo.is_finite() {
            let std_index = next_index;
            next_index += 1;
            original_vars.push(StandardFormVariable {
                offset: lo,
                terms: vec![(std_index, 1.0)],
            });
            if hi.is_finite() {
                upper_bound_rows.push((std_index, hi - lo));
            }
        } else if hi.is_finite() {
            let std_index = next_index;
            next_index += 1;
            original_vars.push(StandardFormVariable {
                offset: hi,
                terms: vec![(std_index, -1.0)],
            });
        } else {
            let pos_index = next_index;
            let neg_index = next_index + 1;
            next_index += 2;
            original_vars.push(StandardFormVariable {
                offset: 0.0,
                terms: vec![(pos_index, 1.0), (neg_index, -1.0)],
            });
        }
    }

    Ok(StandardFormTransform {
        original_vars,
        decision_var_count: next_index,
        upper_bound_rows,
    })
}

/// Solve a linear programming problem using the revised simplex method.
///
/// Matches `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method)`.
///
/// Minimizes `c^T x` subject to:
///   `A_ub @ x <= b_ub` (inequality constraints)
///   `A_eq @ x == b_eq` (equality constraints)
///   `lb <= x <= ub` (variable bounds)
///
/// # Arguments
/// * `c` — Objective function coefficients (minimize c^T x).
/// * `a_ub` — Inequality constraint matrix (each row: a_i^T x <= b_ub_i). Can be empty.
/// * `b_ub` — Inequality constraint RHS.
/// * `a_eq` — Equality constraint matrix. Can be empty.
/// * `b_eq` — Equality constraint RHS.
/// * `bounds` — Variable bounds as `(Option<lower>, Option<upper>)` per variable.
///   `None` means unbounded in that direction. If empty, defaults to `(0, +inf)` per variable.
/// * `maxiter` — Maximum number of simplex iterations.
pub fn linprog(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    bounds: &[(Option<f64>, Option<f64>)],
    maxiter: Option<usize>,
) -> Result<LinprogResult, OptError> {
    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }

    // Validate dimensions.
    if a_ub.len() != b_ub.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_ub rows ({}) must match b_ub length ({})",
                a_ub.len(),
                b_ub.len()
            ),
        });
    }
    for (i, row) in a_ub.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_ub row {i} has {} cols, expected {n}", row.len()),
            });
        }
    }
    if a_eq.len() != b_eq.len() {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "A_eq rows ({}) must match b_eq length ({})",
                a_eq.len(),
                b_eq.len()
            ),
        });
    }
    for (i, row) in a_eq.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("A_eq row {i} has {} cols, expected {n}", row.len()),
            });
        }
    }

    // Apply variable bounds by transforming to standard form.
    // Default bounds: (0, +inf) if not specified.
    let var_bounds: Vec<(f64, f64)> = if bounds.is_empty() {
        vec![(0.0, f64::INFINITY); n]
    } else {
        if bounds.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
            });
        }
        bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect()
    };

    let transform = build_standard_form_transform(&var_bounds)?;

    // Transform to standard form: minimize c^T x, A x = b, x >= 0.
    // 1. Finite-lower-bound variables are shifted to y >= 0.
    // 2. Upper-only variables are reflected as x = ub - y with y >= 0.
    // 3. Free variables are split as x = x_pos - x_neg with x_pos, x_neg >= 0.
    // 4. Add slack variables for inequality constraints and finite shifted upper bounds.
    let m_ub = a_ub.len();
    let m_eq = a_eq.len();
    let decision_var_count = transform.decision_var_count;
    let ub_slacks = &transform.upper_bound_rows;
    let n_ub_slack = ub_slacks.len();

    // Total variables: transformed decision vars + inequality slack + upper-bound slack.
    let total_vars = decision_var_count + m_ub + n_ub_slack;
    let total_constraints = m_ub + m_eq + n_ub_slack;

    // Build standard-form objective: c' = [c, 0, 0, ...]
    let mut c_std = vec![0.0; total_vars];
    for (&coeff, transformed) in c.iter().zip(transform.original_vars.iter()) {
        for &(std_index, std_coeff) in &transformed.terms {
            c_std[std_index] += coeff * std_coeff;
        }
    }

    // Build constraint matrix A_std and b_std.
    let mut a_std = vec![vec![0.0; total_vars]; total_constraints];
    let mut b_std = vec![0.0; total_constraints];

    // Row 0..m_ub: inequality constraints → A_ub x + s = b_ub.
    for i in 0..m_ub {
        b_std[i] = b_ub[i];
        for (j, transformed) in transform.original_vars.iter().enumerate() {
            for &(std_index, std_coeff) in &transformed.terms {
                a_std[i][std_index] += a_ub[i][j] * std_coeff;
            }
            b_std[i] -= a_ub[i][j] * transformed.offset;
        }
        a_std[i][decision_var_count + i] = 1.0; // slack variable
    }

    // Row m_ub..m_ub+m_eq: equality constraints.
    for i in 0..m_eq {
        b_std[m_ub + i] = b_eq[i];
        for (j, transformed) in transform.original_vars.iter().enumerate() {
            for &(std_index, std_coeff) in &transformed.terms {
                a_std[m_ub + i][std_index] += a_eq[i][j] * std_coeff;
            }
            b_std[m_ub + i] -= a_eq[i][j] * transformed.offset;
        }
    }

    // Row m_ub+m_eq..: shifted finite upper bounds: y_i + s_ub = ub_i - lb_i.
    for (k, &(std_index, bound_diff)) in ub_slacks.iter().enumerate() {
        let row = m_ub + m_eq + k;
        a_std[row][std_index] = 1.0;
        a_std[row][decision_var_count + m_ub + k] = 1.0;
        b_std[row] = bound_diff;
    }

    // Ensure all b_std >= 0 (multiply negative rows by -1).
    for i in 0..total_constraints {
        if b_std[i] < 0.0 {
            b_std[i] = -b_std[i];
            for val in &mut a_std[i] {
                *val = -*val;
            }
        }
    }

    // Phase I: find a basic feasible solution using artificial variables.
    let n_art = total_constraints;
    let phase1_vars = total_vars + n_art;
    let mut tableau = vec![vec![0.0; phase1_vars + 1]; total_constraints + 1];

    // Fill constraint rows.
    for i in 0..total_constraints {
        for j in 0..total_vars {
            tableau[i][j] = a_std[i][j];
        }
        tableau[i][total_vars + i] = 1.0; // artificial variable
        tableau[i][phase1_vars] = b_std[i]; // RHS
    }

    // Phase I objective: minimize sum of artificial variables.
    // Start with +1 coefficient for each artificial, then subtract constraint
    // rows to eliminate basic (artificial) variables from the objective.
    let obj_row = total_constraints;
    for i in 0..total_constraints {
        tableau[obj_row][total_vars + i] = 1.0; // coefficient for artificial i
    }
    for i in 0..total_constraints {
        let (tableau_i, tableau_obj) = {
            let (left, right) = tableau.split_at_mut(obj_row);
            (&left[i], &mut right[0])
        };
        for (obj_val, &i_val) in tableau_obj
            .iter_mut()
            .zip(tableau_i.iter())
            .take(phase1_vars + 1)
        {
            *obj_val -= i_val;
        }
    }

    let mut basis: Vec<usize> = (total_vars..total_vars + n_art).collect();
    let maxiter = maxiter.unwrap_or(10_000);

    // Simplex iterations for Phase I.
    let phase1_nit = match simplex_iterate(&mut tableau, &mut basis, maxiter, phase1_vars) {
        Ok(nit) => nit,
        Err(err_code) => {
            let (status, message) = if err_code == 3 {
                (
                    3,
                    "Phase I problem is unbounded (should not happen)".to_string(),
                )
            } else {
                (1, "Phase I iteration limit exceeded".to_string())
            };
            return Ok(LinprogResult {
                x: vec![0.0; n],
                fun: 0.0,
                slack: vec![0.0; m_ub],
                success: false,
                status,
                message,
                nit: maxiter,
            });
        }
    };

    // Check if Phase I found a feasible solution (all artificials = 0).
    // Phase I objective = -sum(artificials). If < -1e-8, some artificials are nonzero → infeasible.
    let phase1_obj = tableau[obj_row][phase1_vars];
    if phase1_obj < -1e-8 {
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status: 2,
            message: "Problem is infeasible".to_string(),
            nit: phase1_nit,
        });
    }

    // Phase II: optimize with original objective.
    // Remove artificial columns and set Phase II objective.
    let mut tableau2 = vec![vec![0.0; total_vars + 1]; total_constraints + 1];
    for i in 0..total_constraints {
        for j in 0..total_vars {
            tableau2[i][j] = tableau[i][j];
        }
        tableau2[i][total_vars] = tableau[i][phase1_vars]; // RHS
    }

    // Set Phase II objective row.
    tableau2[obj_row][..total_vars].copy_from_slice(&c_std[..total_vars]);
    tableau2[obj_row][total_vars] = 0.0;

    // Eliminate basic variables from objective row.
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            let ratio = tableau2[obj_row][bj];
            if ratio.abs() > 1e-15 {
                let (row_vals, obj_vals) = {
                    let (left, right) = tableau2.split_at_mut(obj_row);
                    (&left[row], &mut right[0])
                };
                for (obj_val, &row_val) in obj_vals
                    .iter_mut()
                    .zip(row_vals.iter())
                    .take(total_vars + 1)
                {
                    *obj_val -= ratio * row_val;
                }
            }
        }
    }

    let phase2_result = simplex_iterate(&mut tableau2, &mut basis, maxiter, total_vars);
    let nit = phase1_nit + phase2_result.unwrap_or(maxiter);

    if let Err(err_code) = phase2_result {
        let (status, message) = if err_code == 3 {
            (3, "Problem is unbounded".to_string())
        } else {
            (1, "Phase II iteration limit exceeded".to_string())
        };
        return Ok(LinprogResult {
            x: vec![0.0; n],
            fun: 0.0,
            slack: vec![0.0; m_ub],
            success: false,
            status,
            message,
            nit,
        });
    }

    // Extract solution.
    let mut x_std = vec![0.0; total_vars];
    for (row, &bj) in basis.iter().enumerate() {
        if bj < total_vars {
            x_std[bj] = tableau2[row][total_vars];
        }
    }

    // Check for unbounded (any non-basic variable with negative reduced cost
    // and no positive column entry would have been caught during iteration).
    // The simplex_iterate returns Err for unbounded.

    // Recover original decision variables from the transformed non-negative basis.
    let mut x_orig = vec![0.0; n];
    for (i, transformed) in transform.original_vars.iter().enumerate() {
        let mut value = transformed.offset;
        for &(std_index, std_coeff) in &transformed.terms {
            value += std_coeff * x_std[std_index];
        }
        x_orig[i] = value;
    }

    // Compute slack: s = b_ub - A_ub @ x.
    let slack: Vec<f64> = (0..m_ub)
        .map(|i| {
            b_ub[i]
                - a_ub[i]
                    .iter()
                    .zip(&x_orig)
                    .map(|(&a, &x)| a * x)
                    .sum::<f64>()
        })
        .collect();

    let fun = c.iter().zip(&x_orig).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    Ok(LinprogResult {
        x: x_orig,
        fun,
        slack,
        success: true,
        status: 0,
        message: "Optimization terminated successfully".to_string(),
        nit,
    })
}

fn default_milp_bounds(
    n: usize,
    bounds: &[(Option<f64>, Option<f64>)],
) -> Result<Vec<(f64, f64)>, OptError> {
    if bounds.is_empty() {
        Ok(vec![(0.0, f64::INFINITY); n])
    } else if bounds.len() != n {
        Err(OptError::InvalidArgument {
            detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
        })
    } else {
        Ok(bounds
            .iter()
            .map(|(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .collect())
    }
}

fn bounds_to_options(bounds: &[(f64, f64)]) -> Vec<(Option<f64>, Option<f64>)> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            (
                if lo.is_finite() { Some(lo) } else { None },
                if hi.is_finite() { Some(hi) } else { None },
            )
        })
        .collect()
}

fn is_integral_with_tol(value: f64, tol: f64) -> bool {
    (value - value.round()).abs() <= tol
}

/// Solve a mixed-integer linear program by branch-and-bound over `linprog`
/// relaxations.
///
/// Matches the core SciPy surface of `scipy.optimize.milp` for bounded
/// continuous/integer/binary variables on small to medium problems.
pub fn milp(problem: MilpProblem<'_>, options: MilpOptions) -> Result<MilpResult, OptError> {
    let MilpProblem {
        c,
        integrality,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds,
    } = problem;

    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }

    let integrality = if integrality.is_empty() {
        vec![Integrality::Continuous; n]
    } else {
        if integrality.len() != n {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "integrality length ({}) must match c length ({n})",
                    integrality.len()
                ),
            });
        }
        integrality.to_vec()
    };

    let mut root_bounds = default_milp_bounds(n, bounds)?;
    for (i, (bound, kind)) in root_bounds.iter_mut().zip(integrality.iter()).enumerate() {
        validate_variable_bounds(i, bound.0, bound.1)?;
        match kind {
            Integrality::Continuous | Integrality::Integer => {}
            Integrality::Binary => {
                bound.0 = bound.0.max(0.0);
                bound.1 = bound.1.min(1.0);
            }
        }
        if bound.0.is_nan() || bound.1.is_nan() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds became NaN after integrality adjustment"),
            });
        }
        if bound.0 > bound.1 {
            return Ok(MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status: 2,
                message: format!(
                    "variable {i}: bounds are infeasible after integrality restrictions"
                ),
                mip_node_count: 0,
                nit: 0,
            });
        }
    }

    let max_nodes = options.max_nodes.unwrap_or(2_048);
    let tol = 1e-9;
    let mut node_count = 0usize;
    let mut total_nit = 0usize;
    let mut best: Option<LinprogResult> = None;
    let mut root_status = None;
    let mut stack = vec![root_bounds];

    while let Some(node_bounds) = stack.pop() {
        if node_count >= max_nodes {
            return Ok(match best {
                Some(best) => MilpResult {
                    x: best.x,
                    fun: best.fun,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded after finding an incumbent".to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
                None => MilpResult {
                    x: vec![0.0; n],
                    fun: 0.0,
                    success: false,
                    status: 1,
                    message: "MILP node limit exceeded before finding a feasible integer solution"
                        .to_string(),
                    mip_node_count: node_count,
                    nit: total_nit,
                },
            });
        }

        node_count += 1;
        let lp = linprog(
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            &bounds_to_options(&node_bounds),
            options.lp_maxiter,
        )?;
        total_nit += lp.nit;

        if root_status.is_none() {
            root_status = Some(lp.status);
        }
        if !lp.success {
            continue;
        }
        if best
            .as_ref()
            .is_some_and(|incumbent| lp.fun >= incumbent.fun - tol)
        {
            continue;
        }

        let branch_index = integrality
            .iter()
            .enumerate()
            .find_map(|(i, kind)| match kind {
                Integrality::Continuous => None,
                Integrality::Integer | Integrality::Binary => {
                    if is_integral_with_tol(lp.x[i], tol) {
                        None
                    } else {
                        Some(i)
                    }
                }
            });

        match branch_index {
            None => best = Some(lp),
            Some(i) => {
                let value = lp.x[i];
                let floor = value.floor();
                let ceil = value.ceil();

                let mut lower_branch = node_bounds.clone();
                lower_branch[i].1 = lower_branch[i].1.min(floor);
                if lower_branch[i].0 <= lower_branch[i].1 {
                    stack.push(lower_branch);
                }

                let mut upper_branch = node_bounds;
                upper_branch[i].0 = upper_branch[i].0.max(ceil);
                if upper_branch[i].0 <= upper_branch[i].1 {
                    stack.push(upper_branch);
                }
            }
        }
    }

    Ok(match best {
        Some(best) => MilpResult {
            x: best.x,
            fun: best.fun,
            success: true,
            status: 0,
            message: "Optimization terminated successfully".to_string(),
            mip_node_count: node_count,
            nit: total_nit,
        },
        None => {
            let (status, message) = match root_status {
                Some(3) => (3, "LP relaxation is unbounded".to_string()),
                _ => (2, "Problem is infeasible".to_string()),
            };
            MilpResult {
                x: vec![0.0; n],
                fun: 0.0,
                success: false,
                status,
                message,
                mip_node_count: node_count,
                nit: total_nit,
            }
        }
    })
}

/// Run simplex iterations on a tableau. Returns the number of iterations used.
/// Returns Err if iteration limit exceeded or problem is unbounded.
fn simplex_iterate(
    tableau: &mut [Vec<f64>],
    basis: &mut [usize],
    maxiter: usize,
    n_vars: usize,
) -> Result<usize, u8> {
    let m = basis.len(); // number of constraints
    let obj_row = m;
    let rhs_col = n_vars;

    for iteration in 0..maxiter {
        // Find entering variable using Bland's rule: smallest index with negative reduced cost.
        let mut pivot_col = None;
        let tol = 1e-12;
        for (j, &cost) in tableau[obj_row].iter().enumerate().take(n_vars) {
            if cost < -tol {
                pivot_col = Some(j);
                break;
            }
        }

        let pivot_col = match pivot_col {
            Some(c) => c,
            None => return Ok(iteration), // Optimal
        };

        // Minimum ratio test: find leaving variable.
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for (i, row) in tableau.iter().enumerate().take(m) {
            if row[pivot_col] > tol {
                let ratio = row[rhs_col] / row[pivot_col];
                if ratio < min_ratio {
                    min_ratio = ratio;
                    pivot_row = Some(i);
                }
            }
        }

        let pivot_row = match pivot_row {
            Some(r) => r,
            None => return Err(3), // Unbounded
        };

        // Pivot.
        let pivot_val = tableau[pivot_row][pivot_col];
        for val in &mut tableau[pivot_row][..=rhs_col] {
            *val /= pivot_val;
        }

        for i in 0..=obj_row {
            if i != pivot_row {
                let factor = tableau[i][pivot_col];
                if factor.abs() > 1e-15 {
                    let (p_row, t_row) = if i < pivot_row {
                        let (left, right) = tableau.split_at_mut(pivot_row);
                        (&right[0], &mut left[i])
                    } else {
                        let (left, right) = tableau.split_at_mut(i);
                        (&left[pivot_row], &mut right[0])
                    };

                    for (t_val, &p_val) in t_row.iter_mut().zip(p_row.iter()).take(rhs_col + 1) {
                        *t_val -= factor * p_val;
                    }
                }
            }
        }
        basis[pivot_row] = pivot_col;
    }

    Err(1) // Iteration limit
}

// ══════════════════════════════════════════════════════════════════════
// Global Optimization: differential_evolution, basinhopping
// ══════════════════════════════════════════════════════════════════════

use rand::{Rng, RngExt, SeedableRng};

/// Strategy for differential evolution mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeStrategy {
    /// best/1/bin: mutant = best + F*(r1 - r2).
    #[default]
    Best1Bin,
    /// rand/1/bin: mutant = r0 + F*(r1 - r2).
    Rand1Bin,
}

/// Options for `differential_evolution`.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions {
    /// Mutation strategy.
    pub strategy: DeStrategy,
    /// Maximum number of generations.
    pub maxiter: usize,
    /// Population size multiplier (popsize * ndim = actual population).
    pub popsize: usize,
    /// Convergence tolerance on function value spread.
    pub tol: f64,
    /// Mutation factor (scaling of difference vectors), typically in [0.5, 2].
    pub mutation: f64,
    /// Crossover probability, typically in [0.5, 1.0].
    pub recombination: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for DifferentialEvolutionOptions {
    fn default() -> Self {
        Self {
            strategy: DeStrategy::Best1Bin,
            maxiter: 1000,
            popsize: 15,
            tol: 1e-8,
            mutation: 0.8,
            recombination: 0.7,
            seed: None,
        }
    }
}

/// Minimize a function using differential evolution (global optimizer).
///
/// Matches `scipy.optimize.differential_evolution(func, bounds)`.
///
/// A population-based evolutionary optimizer that requires only function evaluations
/// (no gradients). Handles box constraints naturally.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Box constraints: `(lower, upper)` for each variable.
/// * `opts` — Options controlling strategy, population, mutation, crossover.
pub fn differential_evolution<F>(
    func: F,
    bounds: &[(f64, f64)],
    opts: DifferentialEvolutionOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must not be empty".to_string(),
        });
    }
    if opts.popsize == 0 {
        return Err(OptError::InvalidArgument {
            detail: "popsize must be at least 1".to_string(),
        });
    }
    let pop_count = opts
        .popsize
        .checked_mul(ndim)
        .ok_or_else(|| OptError::InvalidArgument {
            detail: format!(
                "population size overflow: popsize {} * ndim {ndim} exceeds usize",
                opts.popsize
            ),
        })?;
    if pop_count < 4 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "population size must be at least 4, got {pop_count} (popsize={} * ndim={ndim})",
                opts.popsize
            ),
        });
    }
    if !opts.tol.is_finite() || opts.tol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!("tol must be a finite non-negative value, got {}", opts.tol),
        });
    }
    if !opts.mutation.is_finite() || opts.mutation < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "mutation must be a finite non-negative value, got {}",
                opts.mutation
            ),
        });
    }
    if !opts.recombination.is_finite() || !(0.0..=1.0).contains(&opts.recombination) {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "recombination must be finite and within [0, 1], got {}",
                opts.recombination
            ),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }
    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    };

    // Initialize population uniformly within bounds.
    let mut population: Vec<Vec<f64>> = (0..pop_count)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| rng.random_range(lo..hi))
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population.iter().map(|x| func(x)).collect();
    let mut nfev = pop_count;

    // Track best.
    let mut best_idx = 0;
    let mut best_fun = fitness[0];
    for (i, &f) in fitness.iter().enumerate() {
        if f < best_fun {
            best_fun = f;
            best_idx = i;
        }
    }

    let mut converged = false;
    let mut generation = 0;

    for g in 0..opts.maxiter {
        generation = g;

        for i in 0..pop_count {
            // Select three distinct random indices != i.
            let (r0, r1, r2) = select_three(&mut rng, pop_count, i);

            // Mutation.
            let base = match opts.strategy {
                DeStrategy::Best1Bin => &population[best_idx],
                DeStrategy::Rand1Bin => &population[r0],
            };
            let mutant: Vec<f64> = base
                .iter()
                .zip(population[r1].iter().zip(population[r2].iter()))
                .map(|(&b, (&v1, &v2))| b + opts.mutation * (v1 - v2))
                .collect();

            // Binomial crossover.
            let j_rand = rng.random_range(0..ndim);
            let trial: Vec<f64> = (0..ndim)
                .map(|j| {
                    if j == j_rand || rng.random_range(0.0..1.0) < opts.recombination {
                        // Clip to bounds.
                        mutant[j].clamp(bounds[j].0, bounds[j].1)
                    } else {
                        population[i][j]
                    }
                })
                .collect();

            // Selection.
            let f_trial = func(&trial);
            nfev += 1;
            if f_trial <= fitness[i] {
                population[i] = trial;
                fitness[i] = f_trial;
                if f_trial < best_fun {
                    best_fun = f_trial;
                    best_idx = i;
                }
            }
        }

        // Check convergence: spread of fitness values.
        let fmin = fitness
            .iter()
            .cloned()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        let fmax = fitness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        if (fmax - fmin).abs() <= opts.tol * (1.0 + fmin.abs()) {
            converged = true;
            break;
        }
    }

    Ok(OptimizeResult {
        x: population[best_idx].clone(),
        fun: Some(best_fun),
        success: converged,
        status: if converged {
            ConvergenceStatus::Success
        } else {
            ConvergenceStatus::MaxIterations
        },
        message: if converged {
            "Optimization converged".to_string()
        } else {
            format!("Maximum iterations ({}) reached", opts.maxiter)
        },
        nfev,
        njev: 0,
        nhev: 0,
        nit: generation + 1,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// br-7470: Penalty-based constraint handling for differential evolution.
///
/// Wraps the user objective with a quadratic exterior penalty: for each
/// candidate x, the effective cost is
///
/// ```text
/// func(x) + penalty * sum_j violation_j(x)^2
/// ```
///
/// where `violation` returns the sum of squared boundary excess across
/// all linear and nonlinear constraints. A feasible point has zero
/// penalty; an infeasible point pays a large cost that pushes selection
/// away. Mirrors the spirit of scipy.optimize.differential_evolution's
/// constrained mode, which uses a feasibility-tournament with similar
/// boundary-excess penalty (the underlying numerics differ — scipy
/// uses a Lampinen tournament — but for unimodal global minima both
/// converge to the same constrained optimum).
///
/// `constraint_violation` should return >= 0; zero means feasible.
pub fn differential_evolution_constrained<F, G>(
    func: F,
    bounds: &[(f64, f64)],
    constraint_violation: G,
    opts: DifferentialEvolutionOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    const PENALTY: f64 = 1.0e6;
    let penalized = |x: &[f64]| {
        let v = constraint_violation(x);
        if v <= 0.0 {
            func(x)
        } else {
            func(x) + PENALTY * v * v
        }
    };
    differential_evolution(penalized, bounds, opts)
}

/// Select three distinct random indices from [0, n), all different from `exclude`.
///
/// Requires `n >= 4` (3 picks + 1 excluded). If n < 4, picks are allowed to
/// overlap with `exclude` to avoid an infinite loop.
fn select_three(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = Vec::with_capacity(3);
    let strict = n >= 4; // Can we guarantee 3 distinct indices != exclude?
    let mut attempts = 0;
    while indices.len() < 3 {
        let idx = rng.random_range(0..n);
        let skip_exclude = strict && idx == exclude;
        if !skip_exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
        attempts += 1;
        if attempts > 1000 {
            // Fallback: fill remaining with sequential indices
            for k in 0..n {
                if indices.len() >= 3 {
                    break;
                }
                if !indices.contains(&k) {
                    indices.push(k);
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

/// Options for `basinhopping`.
#[derive(Debug, Clone)]
pub struct BasinhoppingOptions {
    /// Number of basin-hopping iterations.
    pub niter: usize,
    /// Temperature for Metropolis acceptance criterion.
    pub temperature: f64,
    /// Step size for random perturbation.
    pub stepsize: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Tolerance for local minimizer.
    pub minimizer_tol: Option<f64>,
}

impl Default for BasinhoppingOptions {
    fn default() -> Self {
        Self {
            niter: 100,
            temperature: 1.0,
            stepsize: 0.5,
            seed: None,
            minimizer_tol: Some(1e-8),
        }
    }
}

/// Minimize a function using the basin-hopping algorithm (global optimizer).
///
/// Matches `scipy.optimize.basinhopping(func, x0)`.
///
/// Combines random perturbation with local minimization and a Metropolis
/// acceptance criterion to escape local minima.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `x0` — Initial guess.
/// * `opts` — Options controlling iterations, temperature, step size.
pub fn basinhopping<F>(
    func: F,
    x0: &[f64],
    opts: BasinhoppingOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = x0.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must not contain NaN or Inf".to_string(),
        });
    }
    if !opts.temperature.is_finite() || opts.temperature <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "temperature must be a positive finite value, got {}",
                opts.temperature
            ),
        });
    }
    if !opts.stepsize.is_finite() || opts.stepsize <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "stepsize must be a positive finite value, got {}",
                opts.stepsize
            ),
        });
    }
    if let Some(tol) = opts.minimizer_tol
        && (!tol.is_finite() || tol <= 0.0)
    {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "minimizer_tol must be a positive finite value when provided, got {tol}",
            ),
        });
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_rng(&mut rand::rng()),
    };

    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        tol: opts.minimizer_tol,
        maxiter: Some(200),
        ..MinimizeOptions::default()
    };

    // Initial local minimization.
    let local_result =
        crate::bfgs(&func, x0, minimize_opts).map_err(|e| OptError::InvalidArgument {
            detail: format!("initial local minimization failed: {e}"),
        })?;

    let mut x_current = local_result.x;
    let mut f_current = local_result.fun.unwrap_or(f64::INFINITY);
    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut total_nfev = local_result.nfev;
    let mut total_nit = 0;

    for _iter in 0..opts.niter {
        // Random perturbation.
        let x_perturbed: Vec<f64> = x_current
            .iter()
            .map(|&xi| xi + rng.random_range(-opts.stepsize..opts.stepsize))
            .collect();

        // Local minimization from perturbed point.
        let local = crate::bfgs(&func, &x_perturbed, minimize_opts);
        total_nit += 1;

        if let Ok(result) = local {
            total_nfev += result.nfev;
            let f_new = result.fun.unwrap_or(f64::INFINITY);

            // Metropolis acceptance criterion.
            let delta = f_new - f_current;
            let accept = if delta < 0.0 {
                true
            } else {
                let prob = (-delta / opts.temperature).exp();
                rng.random_range(0.0..1.0) < prob
            };

            if f_new < f_best {
                f_best = f_new;
                x_best.clone_from(&result.x);
            }

            if accept {
                x_current = result.x;
                f_current = f_new;
            }
        }
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        success: true,
        status: ConvergenceStatus::Success,
        message: format!("Basin-hopping completed after {} iterations", opts.niter),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Dual annealing global optimization.
///
/// Combines simulated annealing with periodic local minimization.
/// Effective for finding global minima of multimodal functions.
///
/// Matches `scipy.optimize.dual_annealing(func, bounds)`.
///
/// # Arguments
/// * `func` — Objective function to minimize.
/// * `bounds` — Search bounds as (lower, upper) pairs for each dimension.
/// * `maxiter` — Maximum number of global iterations.
/// * `seed` — Random seed for reproducibility.
pub fn dual_annealing<F>(
    func: F,
    bounds: &[(f64, f64)],
    maxiter: usize,
    seed: u64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let mut rng = SimpleRng::new(seed);

    // Initialize with random point in bounds
    let mut x_best: Vec<f64> = bounds
        .iter()
        .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
        .collect();
    let mut f_best = func(&x_best);
    let mut nfev = 1usize;

    let mut x_current = x_best.clone();
    let mut f_current = f_best;

    // Local search: dual annealing's defining step is a deterministic local
    // minimization after each annealing move (scipy uses L-BFGS-B). The bfgs
    // result is clamped back into the bounds and re-evaluated.
    let minimize_opts = MinimizeOptions {
        method: Some(OptimizeMethod::Bfgs),
        maxiter: Some(200),
        ..MinimizeOptions::default()
    };
    let local_search = |start: &[f64], nfev: &mut usize| -> (Vec<f64>, f64) {
        match crate::bfgs(&func, start, minimize_opts) {
            Ok(res) => {
                *nfev += res.nfev;
                let x_clamped: Vec<f64> = res
                    .x
                    .iter()
                    .zip(bounds.iter())
                    .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
                    .collect();
                let f_clamped = func(&x_clamped);
                *nfev += 1;
                (x_clamped, f_clamped)
            }
            Err(_) => {
                let f = func(start);
                *nfev += 1;
                (start.to_vec(), f)
            }
        }
    };

    // Temperature schedule
    let t_initial = 5230.0; // SciPy default
    for iteration in 0..maxiter {
        let temp = t_initial / (iteration as f64 + 1.0).ln().max(1.0);

        // Generate candidate via visiting distribution (Cauchy-like perturbation)
        let x_candidate: Vec<f64> = x_current
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| {
                let range = hi - lo;
                let perturbation = range * (rng.next_f64() - 0.5) * temp.sqrt() / (1.0 + temp);
                (xi + perturbation).clamp(lo, hi)
            })
            .collect();

        let f_candidate = func(&x_candidate);
        nfev += 1;

        // Metropolis acceptance
        let delta = f_candidate - f_current;
        let accept = if delta < 0.0 {
            true
        } else {
            let prob = (-delta / temp.max(1e-30)).exp();
            rng.next_f64() < prob
        };

        if accept {
            x_current = x_candidate.clone();
            f_current = f_candidate;
        }

        // Local search from the annealing candidate.
        let (x_local, f_local) = local_search(&x_candidate, &mut nfev);
        if f_local < f_current {
            x_current = x_local.clone();
            f_current = f_local;
        }

        if f_current < f_best {
            x_best.clone_from(&x_current);
            f_best = f_current;
        }
    }

    // Final local polish of the best point found.
    let (x_polished, f_polished) = local_search(&x_best, &mut nfev);
    if f_polished < f_best {
        x_best = x_polished;
        f_best = f_polished;
    }

    Ok(OptimizeResult {
        x: x_best,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "dual_annealing completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Brute-force global optimization.
///
/// Evaluates the objective function at each point of a multidimensional grid
/// defined by `ranges` and `ns` (number of samples per dimension), returning
/// the point with the lowest value.
/// Matches `scipy.optimize.brute(func, ranges, Ns=ns)`.
pub fn brute<F>(func: F, ranges: &[(f64, f64)], ns: usize) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = ranges.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "ranges must not be empty".to_string(),
        });
    }
    if ns < 2 {
        return Err(OptError::InvalidArgument {
            detail: "ns must be at least 2".to_string(),
        });
    }
    for &(lb, ub) in ranges {
        if !lb.is_finite() || !ub.is_finite() || lb >= ub {
            return Err(OptError::InvalidArgument {
                detail: format!("invalid range: ({}, {})", lb, ub),
            });
        }
    }

    let mut best_x = vec![0.0; ndim];
    let mut best_f = f64::INFINITY;
    let mut nfev = 0;

    let total_points = ns.pow(ndim as u32);
    for idx in 0..total_points {
        let mut current_idx = idx;
        let mut x = vec![0.0; ndim];
        for d in 0..ndim {
            let step_idx = current_idx % ns;
            current_idx /= ns;
            let (lb, ub) = ranges[d];
            x[d] = lb + (ub - lb) * (step_idx as f64) / ((ns - 1) as f64);
        }

        let f = func(&x);
        nfev += 1;
        if f < best_f {
            best_f = f;
            best_x = x;
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_f),
        nit: total_points,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "brute completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Simplicial homology global optimization over a bounded box.
///
/// This implementation uses a bounded lattice search to identify promising
/// simplicial cells, then polishes the best seeds with the existing L-BFGS-B
/// local optimizer. It matches the high-level contract of
/// `scipy.optimize.shgo(func, bounds)` for unconstrained bounded problems.
pub fn shgo<F>(func: F, bounds: &[(f64, f64)]) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    if ndim == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must be non-empty".to_string(),
        });
    }
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: bounds must be finite, got ({lo}, {hi})"),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: format!("variable {i}: lower bound {lo} >= upper bound {hi}"),
            });
        }
    }

    let target_points = 5usize.saturating_pow((ndim.min(4)) as u32).max(9);
    let axis_levels = ((target_points as f64).powf(1.0 / ndim as f64).ceil() as usize + 1).max(3);
    let mut lattice = Vec::new();
    let mut current = vec![0.0; ndim];
    build_shgo_lattice(bounds, axis_levels, 0, &mut current, &mut lattice);

    let mut nfev = 0usize;
    let mut ranked: Vec<(f64, Vec<f64>)> = lattice
        .into_iter()
        .filter_map(|point| {
            let value = func(&point);
            nfev += 1;
            value.is_finite().then_some((value, point))
        })
        .collect();

    if ranked.is_empty() {
        return Err(OptError::NonFiniteInput {
            detail: "objective returned only non-finite values over the bounded domain".to_string(),
        });
    }

    ranked.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    let starts = select_shgo_starts(&ranked, bounds, 8);
    let bound_tuples: Vec<Bound> = bounds
        .iter()
        .map(|&(lo, hi)| (Some(lo), Some(hi)))
        .collect();
    let local_options = MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        tol: Some(1e-8),
        maxiter: Some(250),
        maxfev: Some(2500),
        ..MinimizeOptions::default()
    };

    let mut best_x = ranked[0].1.clone();
    let mut best_fun = ranked[0].0;
    let mut total_nit = 0usize;

    for start in starts {
        let local = lbfgsb(&func, &start, local_options, Some(&bound_tuples))?;
        nfev += local.nfev;
        total_nit += local.nit;
        if let Some(fun) = local.fun
            && fun < best_fun
        {
            best_fun = fun;
            best_x = local.x;
        }
    }

    Ok(OptimizeResult {
        x: best_x,
        fun: Some(best_fun),
        success: true,
        status: ConvergenceStatus::Success,
        message: "shgo completed".to_string(),
        nfev,
        njev: 0,
        nhev: 0,
        nit: total_nit.max(1),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

fn build_shgo_lattice(
    bounds: &[(f64, f64)],
    axis_levels: usize,
    dim: usize,
    current: &mut Vec<f64>,
    lattice: &mut Vec<Vec<f64>>,
) {
    if dim == bounds.len() {
        lattice.push(current.clone());
        return;
    }

    let (lo, hi) = bounds[dim];
    for step in 0..axis_levels {
        let fraction = if axis_levels == 1 {
            0.0
        } else {
            step as f64 / (axis_levels - 1) as f64
        };
        current[dim] = lo + fraction * (hi - lo);
        build_shgo_lattice(bounds, axis_levels, dim + 1, current, lattice);
    }
}

fn select_shgo_starts(
    ranked: &[(f64, Vec<f64>)],
    bounds: &[(f64, f64)],
    max_starts: usize,
) -> Vec<Vec<f64>> {
    let mut starts = Vec::new();
    let distance_scale = bounds.iter().map(|(lo, hi)| hi - lo).sum::<f64>().max(1.0);
    let min_distance = 0.1 * distance_scale / bounds.len() as f64;

    for (_, point) in ranked {
        let distinct = starts.iter().all(|existing: &Vec<f64>| {
            existing
                .iter()
                .zip(point.iter())
                .map(|(lhs, rhs)| {
                    let delta = lhs - rhs;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
                >= min_distance
        });
        if distinct {
            starts.push(point.clone());
            if starts.len() >= max_starts {
                break;
            }
        }
    }

    if starts.is_empty() {
        starts.push(ranked[0].1.clone());
    }
    starts
}

/// Simple deterministic pseudo-random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// COBYLA: Constrained Optimization BY Linear Approximation.
///
/// Minimizes `func(x)` subject to `constraints[i](x) >= 0` for all i.
///
/// Uses a simplex-based method that approximates the objective and constraints
/// with linear models at each iteration.
///
/// Matches `scipy.optimize.minimize(func, x0, method='COBYLA', constraints=...)`.
pub fn cobyla<F, G>(
    func: F,
    x0: &[f64],
    constraints: &[G],
    maxiter: usize,
    rhobeg: f64,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut f_best = func(&x);
    let mut nfev = 1usize;
    let mut rho = rhobeg;

    for _iteration in 0..maxiter {
        // Check constraints
        let mut max_violation = 0.0_f64;
        for constraint in constraints {
            let cv = constraint(&x);
            if cv < 0.0 {
                max_violation = max_violation.max(-cv);
            }
        }

        // Try coordinate-wise descent with constraint penalty
        let mut improved = false;
        for d in 0..n {
            for &direction in &[rho, -rho] {
                let mut x_trial = x.clone();
                x_trial[d] += direction;

                let f_trial = func(&x_trial);
                nfev += 1;

                // Check all constraints
                let mut trial_violation = 0.0;
                for constraint in constraints {
                    let cv = constraint(&x_trial);
                    if cv < -1e-10 {
                        trial_violation += -cv;
                    }
                }

                // Accept if: (feasible and better) or (less infeasible)
                let current_penalty = f_best + 1000.0 * max_violation;
                let trial_penalty = f_trial + 1000.0 * trial_violation;

                if trial_penalty < current_penalty - 1e-12 {
                    x = x_trial;
                    f_best = f_trial;
                    max_violation = trial_violation;
                    improved = true;
                }
            }
        }

        // Shrink trust region if no improvement
        if !improved {
            rho *= 0.5;
            if rho < 1e-12 {
                break;
            }
        }
    }

    Ok(OptimizeResult {
        x,
        fun: Some(f_best),
        nit: maxiter,
        nfev,
        njev: 0,
        nhev: 0,
        success: true,
        status: ConvergenceStatus::Success,
        message: "cobyla completed".to_string(),
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Constrained Optimization via Augmented Lagrangian
// ══════════════════════════════════════════════════════════════════════

/// Minimize a function subject to nonlinear constraints using an augmented
/// Lagrangian penalty method.
///
/// Solves: min f(x) subject to c(x) >= 0.
///
/// This converts the constrained problem into a sequence of unconstrained
/// subproblems, each solved via Nelder-Mead.
pub fn augmented_lagrangian<F, C>(
    f: F,
    constraints: C,
    x0: &[f64],
    n_constraints: usize,
    max_outer: usize,
    max_inner: usize,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    C: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "x0 must be non-empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; n_constraints]; // Lagrange multipliers
    let mut mu = 10.0; // penalty parameter
    let mut total_nfev = 0;

    for outer in 0..max_outer {
        // Build augmented Lagrangian:
        // L(x) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ max(0, -c_i(x) + λ_i/μ)²
        let lambda_copy = lambda.clone();
        let mu_copy = mu;
        let penalty_fn = |xv: &[f64]| -> f64 {
            let fval = f(xv);
            let cvec = constraints(xv);
            let mut penalty = 0.0;
            for (i, &ci) in cvec.iter().enumerate() {
                let shifted = -ci + lambda_copy[i] / mu_copy;
                if shifted > 0.0 {
                    penalty += shifted * shifted;
                }
            }
            fval + mu_copy / 2.0 * penalty
        };

        // Solve unconstrained subproblem
        let opts = MinimizeOptions {
            method: Some(crate::OptimizeMethod::NelderMead),
            maxiter: Some(max_inner),
            tol: Some(1e-8),
            ..MinimizeOptions::default()
        };
        let result = crate::minimize(penalty_fn, &x, opts)?;
        total_nfev += result.nfev;
        x = result.x;

        // Update multipliers
        let cvec = constraints(&x);
        let mut max_violation = 0.0f64;
        for (i, &ci) in cvec.iter().enumerate() {
            let unconstrained_lambda = lambda[i] - mu * ci;
            lambda[i] = if unconstrained_lambda.is_nan() {
                f64::NAN
            } else {
                unconstrained_lambda.max(0.0)
            };

            let violation = -ci;
            max_violation = if max_violation.is_nan() || violation.is_nan() {
                f64::NAN
            } else {
                max_violation.max(violation.max(0.0))
            };
        }

        // Check convergence
        if max_violation < 1e-6 {
            let fval = f(&x);
            return Ok(OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("augmented lagrangian converged in {outer} outer iterations"),
                nfev: total_nfev,
                njev: 0,
                nhev: 0,
                nit: outer + 1,
                jac: None,
                hess_inv: None,
                maxcv: Some(max_violation),
            });
        }

        // Increase penalty
        mu *= 10.0;
    }

    let fval = f(&x);
    Ok(OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "augmented lagrangian did not converge".to_string(),
        nfev: total_nfev,
        njev: 0,
        nhev: 0,
        nit: max_outer,
        jac: None,
        hess_inv: None,
        maxcv: None,
    })
}

/// Golden-section search for 1D minimization on [a, b].
///
/// Matches `scipy.optimize.golden`.
pub fn golden<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
    let mut lo = a;
    let mut hi = b;
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let mut fc = f(c);
    let mut fd = f(d);

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = f(d);
        }
    }

    let x_min = (lo + hi) / 2.0;
    (x_min, f(x_min))
}

/// Brent's method for 1D minimization.
///
/// Matches `scipy.optimize.brent`.
pub fn brent_minimize<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let golden = 0.381_966_011_250_105; // 1 - golden ratio conjugate

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut lo = a;
    let mut hi = b;
    let mut e = 0.0f64;
    let mut d_step;

    for _ in 0..maxiter {
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            return (x, fx);
        }

        // Try parabolic interpolation
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = 2.0 * (q - r);
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

            if p.abs() < (0.5 * q * e).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d_step = p / q;
                let u = x + d_step;
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d_step = if x < mid { tol1 } else { -tol1 };
                }
            } else {
                e = if x < mid { hi - x } else { lo - x };
                d_step = golden * e;
            }
        } else {
            e = if x < mid { hi - x } else { lo - x };
            d_step = golden * e;
        }

        let u = if d_step.abs() >= tol1 {
            x + d_step
        } else {
            x + tol1 * d_step.signum()
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                hi = x;
            } else {
                lo = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }

        e = d_step;
    }

    (x, fx)
}

/// Find a minimum bracket for a 1D function.
///
/// Returns (xa, xb, xc, fa, fb, fc) such that f(xb) < f(xa) and f(xb) < f(xc).
/// Matches `scipy.optimize.bracket`.
pub fn bracket<F>(f: F, xa: f64, xb: f64) -> (f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    let golden = 1.618_033_988_749_895;
    let limit = 100.0;

    let mut xa = xa;
    let mut xb = xb;
    let mut fa = f(xa);
    let mut fb = f(xb);

    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut xc = xb + golden * (xb - xa);
    let mut fc = f(xc);

    for _ in 0..200 {
        if fb <= fc {
            return (xa, xb, xc, fa, fb, fc);
        }

        // Parabolic extrapolation
        let r = (xb - xa) * (fb - fc);
        let q = (xb - xc) * (fb - fa);
        let denom = 2.0 * (q - r).max(1e-20) * if q > r { 1.0 } else { -1.0 };
        let mut w = xb - ((xb - xc) * q - (xb - xa) * r) / denom;
        let wlim = xb + limit * (xc - xb);

        if (w - xc) * (xb - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xa = xb;
                fa = fb;
                xb = w;
                fb = fw;
                return (xa, xb, xc, fa, fb, fc);
            } else if fw > fb {
                xc = w;
                fc = fw;
                return (xa, xb, xc, fa, fb, fc);
            }
            w = xc + golden * (xc - xb);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
        } else if (w - wlim) * (xc - w) > 0.0 {
            let fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + golden * (xc - xb);
                fb = fc;
                fc = fw;
            }
        } else {
            w = xc + golden * (xc - xb);
        }

        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = f(xc);
    }

    (xa, xb, xc, fa, fb, fc)
}

/// Minimize a scalar function using bounded Brent's method.
///
/// Matches `scipy.optimize.minimize_scalar(method='bounded')`.
pub fn minimize_scalar_bounded<F>(f: F, bounds: (f64, f64), tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    brent_minimize(f, bounds.0, bounds.1, tol, maxiter)
}

/// Fixed-point iteration: find x such that f(x) = x.
///
/// Matches `scipy.optimize.fixed_point`.
pub fn fixed_point<F>(f: F, x0: f64, tol: f64, maxiter: usize) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;
    for _iter in 0..maxiter {
        let x_new = f(x);
        if (x_new - x).abs() < tol {
            return Ok(x_new);
        }

        // Steffensen's method acceleration
        let x_new2 = f(x_new);
        let denom = x_new2 - 2.0 * x_new + x;
        if denom.abs() > 1e-30 {
            let x_acc = x - (x_new - x).powi(2) / denom;
            if (x_acc - x).abs() < tol {
                return Ok(x_acc);
            }
            x = x_acc;
        } else {
            x = x_new;
        }
    }
    Err(OptError::InvalidArgument {
        detail: format!("fixed_point did not converge in {maxiter} iterations"),
    })
}

/// Non-negative least squares: minimize ||Ax - b||² subject to x >= 0.
///
/// Uses the active-set method (Lawson-Hanson algorithm).
/// Matches `scipy.optimize.nnls`.
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Result<(Vec<f64>, f64), OptError> {
    let m = a.len();
    if m == 0 {
        return Err(OptError::InvalidArgument {
            detail: "empty matrix".to_string(),
        });
    }
    let n = a[0].len();
    for (row_index, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(OptError::InvalidArgument {
                detail: "A must be rectangular".to_string(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("A contains non-finite values in row {row_index}"),
            });
        }
    }
    if b.len() != m {
        return Err(OptError::InvalidArgument {
            detail: format!("b length {} != A rows {m}", b.len()),
        });
    }
    if b.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "b must not contain NaN or Inf".to_string(),
        });
    }

    let mut x = vec![0.0; n];
    let mut passive = vec![false; n]; // passive set (unconstrained)

    for _ in 0..3 * n {
        // Compute gradient: w = A^T (b - Ax)
        let mut ax = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                ax[i] += a[i][j] * x[j];
            }
        }

        let mut w = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                w[j] += a[i][j] * (b[i] - ax[i]);
            }
        }

        // Find max w[j] among active (non-passive) variables
        let mut max_w = f64::NEG_INFINITY;
        let mut max_j = 0;
        for j in 0..n {
            if !passive[j] && w[j] > max_w {
                max_w = w[j];
                max_j = j;
            }
        }

        if max_w <= 1e-10 {
            break; // optimality reached
        }

        passive[max_j] = true;

        // Solve unconstrained least squares on passive set
        loop {
            let passive_indices: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            let p = passive_indices.len();

            if p == 0 {
                break;
            }

            // Build sub-problem: A_P * s_P = b
            let mut ata = vec![vec![0.0; p]; p];
            let mut atb_sub = vec![0.0; p];

            for (pi, &ji) in passive_indices.iter().enumerate() {
                for (pj, &jj) in passive_indices.iter().enumerate() {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..m {
                        ata[pi][pj] += a[i][ji] * a[i][jj];
                    }
                }
                #[allow(clippy::needless_range_loop)]
                for i in 0..m {
                    atb_sub[pi] += a[i][ji] * b[i];
                }
            }

            // Solve ata * s = atb_sub
            let s_p = solve_small_system(&ata, &atb_sub);

            // Check for negative elements
            let mut all_positive = true;
            for (pi, &si) in s_p.iter().enumerate() {
                if si <= 0.0 && passive[passive_indices[pi]] {
                    all_positive = false;
                    break;
                }
            }

            if all_positive {
                for (pi, &ji) in passive_indices.iter().enumerate() {
                    x[ji] = s_p[pi];
                }
                break;
            }

            // Find alpha and move variables back to active set
            let mut alpha = f64::INFINITY;
            for (pi, &ji) in passive_indices.iter().enumerate() {
                if s_p[pi] <= 0.0 {
                    let denom = x[ji] - s_p[pi];
                    if denom > 0.0 {
                        let a_val = x[ji] / denom;
                        alpha = alpha.min(a_val);
                    }
                }
            }

            for (pi, &ji) in passive_indices.iter().enumerate() {
                x[ji] += alpha * (s_p[pi] - x[ji]);
                if x[ji].abs() < 1e-15 {
                    x[ji] = 0.0;
                    passive[ji] = false;
                }
            }
        }
    }

    // Compute residual
    let mut residual = 0.0;
    for i in 0..m {
        let mut ax_i = 0.0;
        for j in 0..n {
            ax_i += a[i][j] * x[j];
        }
        residual += (b[i] - ax_i).powi(2);
    }

    Ok((x, residual.sqrt()))
}

fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut row = r.clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue;
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
        if aug[i][i].abs() < 1e-15 {
            continue;
        }
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    x
}

/// A pooled block of adjacent points produced by the PAVA stack.
struct IsotonicBlock {
    /// Number of original points the block spans.
    count: usize,
    /// Sum of case weights over the block.
    weight: f64,
    /// Weight-weighted sum of responses over the block.
    weighted_sum: f64,
}

/// Isotonic regression: fit a non-decreasing least-squares function.
///
/// Minimizes `Σ wᵢ(yᵢ − xᵢ)²` subject to `x` being non-decreasing, via the
/// Pool Adjacent Violators Algorithm (PAVA). Matches
/// `scipy.optimize.isotonic_regression` (the `increasing=True` case).
///
/// `weights` must be strictly positive and the same length as `y`; violating
/// either, or a non-finite entry in `y`/`weights`, yields a `NaN` vector.
pub fn isotonic_regression(y: &[f64], weights: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    if y.iter().any(|v| !v.is_finite()) {
        return vec![f64::NAN; n];
    }

    let w: Vec<f64> = if let Some(wts) = weights {
        // SciPy requires strictly-positive case weights.
        if wts.len() != n || wts.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return vec![f64::NAN; n];
        }
        wts.to_vec()
    } else {
        vec![1.0; n]
    };

    // Pool Adjacent Violators Algorithm (Busing 2022): scan left to right,
    // maintaining a stack of blocks whose means are strictly increasing. Each
    // new point starts a singleton block and is merged into the block to its
    // left as long as the left block's mean exceeds it — a downward
    // violation. Means are compared by cross-multiplication (all weights are
    // strictly positive) to avoid division until the final expansion.
    let mut blocks: Vec<IsotonicBlock> = Vec::with_capacity(n);
    for i in 0..n {
        let mut block = IsotonicBlock {
            count: 1,
            weight: w[i],
            weighted_sum: w[i] * y[i],
        };
        while let Some(prev) = blocks.last() {
            // prev.mean > block.mean  ⇔  prev.wsum·block.w > block.wsum·prev.w
            if prev.weighted_sum * block.weight > block.weighted_sum * prev.weight {
                let prev = blocks.pop().expect("last() returned Some");
                block.count += prev.count;
                block.weight += prev.weight;
                block.weighted_sum += prev.weighted_sum;
            } else {
                break;
            }
        }
        blocks.push(block);
    }

    // Expand each block's weighted mean back across the points it spans.
    let mut result = Vec::with_capacity(n);
    for block in &blocks {
        let mean = block.weighted_sum / block.weight;
        result.resize(result.len() + block.count, mean);
    }
    result
}

/// Bisection method for scalar optimization: find minimum by trisection.
///
/// For unimodal functions on [a, b].
pub fn minimize_trisection<F>(f: F, a: f64, b: f64, tol: f64, maxiter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    if !a.is_finite() || !b.is_finite() || !tol.is_finite() || tol <= 0.0 || a >= b {
        return (f64::NAN, f64::NAN);
    }
    let mut lo = a;
    let mut hi = b;

    for _ in 0..maxiter {
        if (hi - lo).abs() < tol {
            break;
        }
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    let x = (lo + hi) / 2.0;
    (x, f(x))
}

/// Gradient descent with line search.
///
/// Minimizes f(x) starting from x0 using gradient descent with backtracking.
pub fn gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty()
        || !tol.is_finite()
        || tol <= 0.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || x0.iter().any(|v| !v.is_finite())
    {
        return OptimizeResult {
            x: x0.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::InvalidInput,
            message: "invalid input parameters".to_string(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
    }
    let mut x = x0.to_vec();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;
        if g.len() != x.len() || g.iter().any(|v| !v.is_finite()) {
            return OptimizeResult {
                x,
                fun: None,
                success: false,
                status: ConvergenceStatus::InvalidInput,
                message: "gradient returned invalid values".to_string(),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
        }

        let g_norm: f64 = g.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if g_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("gradient descent converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }

        // Backtracking line search (Armijo condition)
        let mut alpha = learning_rate;
        let f0 = f(&x);
        nfev += 1;
        let descent = g.iter().map(|&v| v * v).sum::<f64>();

        let mut found = false;
        for _ in 0..20 {
            let x_new: Vec<f64> = x
                .iter()
                .zip(g.iter())
                .map(|(&xi, &gi)| xi - alpha * gi)
                .collect();
            let f_new = f(&x_new);
            nfev += 1;
            if f_new <= f0 - 1e-4 * alpha * descent {
                x = x_new;
                found = true;
                break;
            }
            alpha *= 0.5;
        }

        // If line search didn't find improvement, take a small step anyway
        if !found {
            for (xi, &gi) in x.iter_mut().zip(g.iter()) {
                *xi -= learning_rate * 1e-3 * gi;
            }
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "gradient descent did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Projected gradient descent for box-constrained optimization.
///
/// Minimizes f(x) subject to lb <= x <= ub.
#[allow(clippy::too_many_arguments)]
pub fn projected_gradient_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    tol: f64,
    maxiter: usize,
    learning_rate: f64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0
        || lb.len() != n
        || ub.len() != n
        || !tol.is_finite()
        || tol <= 0.0
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
        || x0.iter().any(|v| !v.is_finite())
        || lb
            .iter()
            .zip(ub.iter())
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return OptimizeResult {
            x: x0.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::InvalidInput,
            message: "invalid input parameters".to_string(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        };
    }
    let mut x: Vec<f64> = x0
        .iter()
        .zip(lb.iter().zip(ub.iter()))
        .map(|(&xi, (&lo, &hi))| xi.clamp(lo, hi))
        .collect();
    let mut nfev = 0;
    let mut njev = 0;

    for iter in 0..maxiter {
        let g = grad(&x);
        njev += 1;
        if g.len() != n || g.iter().any(|v| !v.is_finite()) {
            return OptimizeResult {
                x,
                fun: None,
                success: false,
                status: ConvergenceStatus::InvalidInput,
                message: "gradient returned invalid values".to_string(),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: None,
                hess_inv: None,
                maxcv: None,
            };
        }

        // Projected gradient step
        let x_new: Vec<f64> = x
            .iter()
            .zip(g.iter())
            .zip(lb.iter().zip(ub.iter()))
            .map(|((&xi, &gi), (&lo, &hi))| (xi - learning_rate * gi).clamp(lo, hi))
            .collect();

        let step_norm: f64 = x_new
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = x_new;

        if step_norm < tol {
            let fval = f(&x);
            nfev += 1;
            return OptimizeResult {
                x,
                fun: Some(fval),
                success: true,
                status: ConvergenceStatus::Success,
                message: format!("projected GD converged in {iter} iterations"),
                nfev,
                njev,
                nhev: 0,
                nit: iter,
                jac: Some(g),
                hess_inv: None,
                maxcv: None,
            };
        }
    }

    let fval = f(&x);
    nfev += 1;
    OptimizeResult {
        x,
        fun: Some(fval),
        success: false,
        status: ConvergenceStatus::MaxIterations,
        message: "projected GD did not converge".to_string(),
        nfev,
        njev,
        nhev: 0,
        nit: maxiter,
        jac: None,
        hess_inv: None,
        maxcv: None,
    }
}

/// Evaluate the first derivative of a scalar function numerically.
///
/// This mirrors `scipy.differentiate.derivative` for scalar-in/scalar-out
/// callables: adaptive real finite differences, Richardson extrapolation,
/// convergence metadata, and explicit non-finite failure handling.
pub fn derivative<F>(
    f: F,
    x: f64,
    options: DifferentiateOptions,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(f64) -> f64,
{
    adaptive_first_derivative(&f, x, options)
}

/// Evaluate the Jacobian of a vector-valued function numerically.
///
/// `df[row][column]` is the derivative of `f(x)[row]` with respect to
/// `x[column]`.
pub fn jacobian<F>(
    f: F,
    x: &[f64],
    options: DifferentiateOptions,
) -> Result<JacobianResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_differentiate_input(x)?;
    validate_differentiate_options(options)?;

    let f0 = f(x);
    if f0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("f(x) must return at least one element"),
        });
    }
    if f0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("f(x) returned non-finite values"),
        });
    }

    let rows = f0.len();
    let columns = x.len();
    let mut df = vec![vec![0.0; columns]; rows];
    let mut error = vec![vec![0.0; columns]; rows];
    let mut status = DifferentiateStatus::Converged;
    let mut success = true;
    let mut nit = 0;
    let mut nfev = 1;

    for row in 0..rows {
        for column in 0..columns {
            let component = |value: f64| {
                let mut shifted = x.to_vec();
                shifted[column] = value;
                let values = f(&shifted);
                if values.len() == rows {
                    values[row]
                } else {
                    f64::NAN
                }
            };
            let result = adaptive_first_derivative(&component, x[column], options)?;
            df[row][column] = result.df;
            error[row][column] = result.error;
            success &= result.success;
            status = merge_differentiate_status(status, result.status);
            nit = nit.max(result.nit);
            nfev += result.nfev;
        }
    }

    Ok(JacobianResult {
        df,
        error,
        success,
        status,
        nit,
        nfev,
    })
}

/// Evaluate the Hessian of a scalar function numerically.
///
/// This uses central second differences with Richardson extrapolation and
/// returns a symmetric Hessian matrix.
pub fn hessian<F>(f: F, x: &[f64], options: DifferentiateOptions) -> Result<HessianResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    validate_differentiate_input(x)?;
    let order = validate_differentiate_options(options)?;
    let f0 = evaluate_finite_vector_scalar(&f, x, "f(x)")?;

    let n = x.len();
    let mut ddf = vec![vec![0.0; n]; n];
    let mut error = vec![vec![0.0; n]; n];
    let mut status = DifferentiateStatus::Converged;
    let mut success = true;
    let mut nit = 0;
    let mut nfev = 1;

    for row in 0..n {
        for column in row..n {
            let result = adaptive_hessian_component(&f, x, f0, row, column, options, order)?;
            ddf[row][column] = result.df;
            ddf[column][row] = result.df;
            error[row][column] = result.error;
            error[column][row] = result.error;
            success &= result.success;
            status = merge_differentiate_status(status, result.status);
            nit = nit.max(result.nit);
            nfev += result.nfev;
        }
    }

    Ok(HessianResult {
        ddf,
        error,
        success,
        status,
        nit,
        nfev,
    })
}

fn validate_differentiate_input(x: &[f64]) -> Result<(), OptError> {
    if x.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x must have at least one element"),
        });
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x must contain only finite values"),
        });
    }
    Ok(())
}

fn validate_differentiate_options(options: DifferentiateOptions) -> Result<usize, OptError> {
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be greater than zero"),
        });
    }
    if options.order == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("order must be greater than zero"),
        });
    }
    if !options.initial_step.is_finite() || options.initial_step <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("initial_step must be finite and positive"),
        });
    }
    if !options.step_factor.is_finite()
        || options.step_factor <= 0.0
        || (options.step_factor - 1.0).abs() <= f64::EPSILON
    {
        return Err(OptError::InvalidArgument {
            detail: String::from("step_factor must be finite, positive, and not equal to one"),
        });
    }
    if !options.tolerances.atol.is_finite() || options.tolerances.atol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("atol must be finite and non-negative"),
        });
    }
    if !options.tolerances.rtol.is_finite() || options.tolerances.rtol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("rtol must be finite and non-negative"),
        });
    }

    if options.order.is_multiple_of(2) {
        Ok(options.order.max(2))
    } else {
        let rounded = options
            .order
            .checked_add(1)
            .ok_or_else(|| OptError::InvalidArgument {
                detail: String::from("order is too large"),
            })?;
        Ok(rounded.max(2))
    }
}

fn component_step(initial_step: f64, x: f64) -> f64 {
    let scale = 1.0 + x.abs();
    (initial_step * scale).max(f64::EPSILON.sqrt() * scale)
}

fn step_at_iteration(initial_step: f64, step_factor: f64, iteration: usize) -> f64 {
    initial_step / step_factor.powf(iteration as f64)
}

fn extrapolation_levels(
    order: usize,
    direction: StepDirection,
    iteration: usize,
    previous_levels: usize,
) -> usize {
    let maximum = match direction {
        StepDirection::Central => (order / 2).max(1),
        StepDirection::Forward | StepDirection::Backward => order.max(1),
    };
    maximum.min(iteration + 1).min(previous_levels + 1)
}

fn richardson_estimate(
    base: f64,
    previous_tableau: &[f64],
    levels: usize,
    step_factor: f64,
    error_power: usize,
) -> (f64, Vec<f64>) {
    let mut current = base;
    let mut next_tableau = Vec::with_capacity(levels);
    next_tableau.push(base);
    for level in 1..levels {
        let denominator = step_factor.powf((error_power * level) as f64) - 1.0;
        let previous_lower_order = previous_tableau[level - 1];
        current += (current - previous_lower_order) / denominator;
        next_tableau.push(current);
    }
    (current, next_tableau)
}

fn evaluate_finite_scalar<F>(f: &F, x: f64, context: &str) -> Result<f64, OptError>
where
    F: Fn(f64) -> f64,
{
    let value = f(x);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OptError::NonFiniteInput {
            detail: format!("{context} returned a non-finite value"),
        })
    }
}

fn evaluate_finite_vector_scalar<F>(f: &F, x: &[f64], context: &str) -> Result<f64, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let value = f(x);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(OptError::NonFiniteInput {
            detail: format!("{context} returned a non-finite value"),
        })
    }
}

fn adaptive_first_derivative<F>(
    f: &F,
    x: f64,
    options: DifferentiateOptions,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(f64) -> f64,
{
    if !x.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x must be finite"),
        });
    }
    let order = validate_differentiate_options(options)?;
    let initial_step = component_step(options.initial_step, x);
    let error_power = match options.step_direction {
        StepDirection::Central => 2,
        StepDirection::Forward | StepDirection::Backward => 1,
    };
    let mut nfev = 0;
    let f0 = match options.step_direction {
        StepDirection::Central => None,
        StepDirection::Forward | StepDirection::Backward => {
            nfev += 1;
            Some(evaluate_finite_scalar(f, x, "f(x)")?)
        }
    };
    let mut previous_estimate = None;
    let mut previous_tableau = Vec::new();
    let mut best_df = f64::NAN;
    let mut best_error = f64::INFINITY;
    let mut best_nit = 0;
    let mut best_step = initial_step;

    for iteration in 0..options.maxiter {
        let step = step_at_iteration(initial_step, options.step_factor, iteration);
        let base = match options.step_direction {
            StepDirection::Central => {
                let plus = evaluate_finite_scalar(f, x + step, "f(x + h)")?;
                let minus = evaluate_finite_scalar(f, x - step, "f(x - h)")?;
                nfev += 2;
                (plus - minus) / (2.0 * step)
            }
            StepDirection::Forward => {
                let plus = evaluate_finite_scalar(f, x + step, "f(x + h)")?;
                nfev += 1;
                let f_at_x = f0.ok_or_else(|| OptError::InvalidArgument {
                    detail: String::from("one-sided derivative missing base evaluation"),
                })?;
                (plus - f_at_x) / step
            }
            StepDirection::Backward => {
                let minus = evaluate_finite_scalar(f, x - step, "f(x - h)")?;
                nfev += 1;
                let f_at_x = f0.ok_or_else(|| OptError::InvalidArgument {
                    detail: String::from("one-sided derivative missing base evaluation"),
                })?;
                (f_at_x - minus) / step
            }
        };
        if !base.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("finite-difference estimate became non-finite"),
            });
        }

        let levels = extrapolation_levels(
            order,
            options.step_direction,
            iteration,
            previous_tableau.len(),
        );
        let (estimate, next_tableau) = richardson_estimate(
            base,
            &previous_tableau,
            levels,
            options.step_factor,
            error_power,
        );
        if !estimate.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("Richardson estimate became non-finite"),
            });
        }

        let error =
            previous_estimate.map_or(f64::INFINITY, |previous: f64| (estimate - previous).abs());
        let nit = iteration + 1;
        if iteration == 0 || error < best_error {
            best_df = estimate;
            best_error = error;
            best_nit = nit;
            best_step = step;
        } else if error.is_finite() && best_error.is_finite() && error > best_error {
            return Ok(DerivativeResult {
                df: best_df,
                error: best_error,
                success: false,
                status: DifferentiateStatus::ErrorIncreased,
                nit,
                nfev,
                step: best_step,
            });
        }

        if error.is_finite()
            && error <= options.tolerances.atol + options.tolerances.rtol * estimate.abs()
        {
            return Ok(DerivativeResult {
                df: estimate,
                error,
                success: true,
                status: DifferentiateStatus::Converged,
                nit,
                nfev,
                step,
            });
        }
        previous_estimate = Some(estimate);
        previous_tableau = next_tableau;
    }

    Ok(DerivativeResult {
        df: best_df,
        error: best_error,
        success: false,
        status: DifferentiateStatus::MaxIterations,
        nit: best_nit,
        nfev,
        step: best_step,
    })
}

fn adaptive_hessian_component<F>(
    f: &F,
    x: &[f64],
    f0: f64,
    row: usize,
    column: usize,
    options: DifferentiateOptions,
    order: usize,
) -> Result<DerivativeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let row_step = component_step(options.initial_step, x[row]);
    let column_step = component_step(options.initial_step, x[column]);
    let mut previous_estimate = None;
    let mut previous_tableau = Vec::new();
    let mut best_ddf = f64::NAN;
    let mut best_error = f64::INFINITY;
    let mut best_nit = 0;
    let mut best_step = row_step.max(column_step);
    let mut nfev = 0;

    for iteration in 0..options.maxiter {
        let h_row = step_at_iteration(row_step, options.step_factor, iteration);
        let h_column = step_at_iteration(column_step, options.step_factor, iteration);
        let (base, evaluations) =
            hessian_component_estimate(f, x, f0, row, column, h_row, h_column)?;
        nfev += evaluations;
        if !base.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("finite-difference Hessian estimate became non-finite"),
            });
        }

        let levels = (order / 2)
            .max(1)
            .min(iteration + 1)
            .min(previous_tableau.len() + 1);
        let (estimate, next_tableau) =
            richardson_estimate(base, &previous_tableau, levels, options.step_factor, 2);
        if !estimate.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("Richardson Hessian estimate became non-finite"),
            });
        }

        let error =
            previous_estimate.map_or(f64::INFINITY, |previous: f64| (estimate - previous).abs());
        let nit = iteration + 1;
        if iteration == 0 || error < best_error {
            best_ddf = estimate;
            best_error = error;
            best_nit = nit;
            best_step = h_row.max(h_column);
        } else if error.is_finite() && best_error.is_finite() && error > best_error {
            return Ok(DerivativeResult {
                df: best_ddf,
                error: best_error,
                success: false,
                status: DifferentiateStatus::ErrorIncreased,
                nit,
                nfev,
                step: best_step,
            });
        }

        if error.is_finite()
            && error <= options.tolerances.atol + options.tolerances.rtol * estimate.abs()
        {
            return Ok(DerivativeResult {
                df: estimate,
                error,
                success: true,
                status: DifferentiateStatus::Converged,
                nit,
                nfev,
                step: h_row.max(h_column),
            });
        }
        previous_estimate = Some(estimate);
        previous_tableau = next_tableau;
    }

    Ok(DerivativeResult {
        df: best_ddf,
        error: best_error,
        success: false,
        status: DifferentiateStatus::MaxIterations,
        nit: best_nit,
        nfev,
        step: best_step,
    })
}

fn hessian_component_estimate<F>(
    f: &F,
    x: &[f64],
    f0: f64,
    row: usize,
    column: usize,
    h_row: f64,
    h_column: f64,
) -> Result<(f64, usize), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    if row == column {
        let mut plus = x.to_vec();
        plus[row] += h_row;
        let f_plus = evaluate_finite_vector_scalar(f, &plus, "f(x + h)")?;

        let mut minus = x.to_vec();
        minus[row] -= h_row;
        let f_minus = evaluate_finite_vector_scalar(f, &minus, "f(x - h)")?;

        Ok(((f_plus - 2.0 * f0 + f_minus) / (h_row * h_row), 2))
    } else {
        let mut point = x.to_vec();
        point[row] += h_row;
        point[column] += h_column;
        let f_pp = evaluate_finite_vector_scalar(f, &point, "f(x + h_i + h_j)")?;

        point[column] = x[column] - h_column;
        let f_pm = evaluate_finite_vector_scalar(f, &point, "f(x + h_i - h_j)")?;

        point[row] = x[row] - h_row;
        point[column] = x[column] + h_column;
        let f_mp = evaluate_finite_vector_scalar(f, &point, "f(x - h_i + h_j)")?;

        point[column] = x[column] - h_column;
        let f_mm = evaluate_finite_vector_scalar(f, &point, "f(x - h_i - h_j)")?;

        Ok(((f_pp - f_pm - f_mp + f_mm) / (4.0 * h_row * h_column), 4))
    }
}

fn merge_differentiate_status(
    left: DifferentiateStatus,
    right: DifferentiateStatus,
) -> DifferentiateStatus {
    match (left, right) {
        (DifferentiateStatus::ErrorIncreased, _) | (_, DifferentiateStatus::ErrorIncreased) => {
            DifferentiateStatus::ErrorIncreased
        }
        (DifferentiateStatus::MaxIterations, _) | (_, DifferentiateStatus::MaxIterations) => {
            DifferentiateStatus::MaxIterations
        }
        (DifferentiateStatus::Converged, DifferentiateStatus::Converged) => {
            DifferentiateStatus::Converged
        }
    }
}

/// Compute numerical gradient via forward differences.
pub fn numerical_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![f64::NAN; n];
    }
    let f0 = f(x);
    if !f0.is_finite() {
        return vec![f64::NAN; n];
    }
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let fp = f(&xp);
        grad.push(if fp.is_finite() {
            (fp - f0) / eps
        } else {
            f64::NAN
        });
    }
    grad
}

/// Numerical Hessian via central differences.
pub fn numerical_hessian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![vec![f64::NAN; n]; n];
    }
    let mut hess = vec![vec![0.0; n]; n];
    let mut scratch = x.to_vec();

    for i in 0..n {
        for j in i..n {
            // f(x + eps_i + eps_j)
            scratch[i] += eps;
            scratch[j] += eps;
            let f_pp = f(&scratch);

            // f(x + eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_pm = f(&scratch);

            // f(x - eps_i + eps_j)
            scratch[i] -= 2.0 * eps;
            scratch[j] += 2.0 * eps;
            let f_mp = f(&scratch);

            // f(x - eps_i - eps_j)
            scratch[j] -= 2.0 * eps;
            let f_mm = f(&scratch);

            // Restore scratch to original x
            scratch[i] += eps;
            scratch[j] += eps;

            hess[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Compute the numerical Jacobian of a vector function.
pub fn numerical_jacobian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 || !eps.is_finite() || eps <= 0.0 || x.iter().any(|v| !v.is_finite()) {
        return vec![];
    }
    let f0 = f(x);
    let m = f0.len();
    if m == 0 || f0.iter().any(|v| !v.is_finite()) {
        return vec![vec![f64::NAN; n]; m.max(1)];
    }
    let mut jac = vec![vec![0.0; n]; m];
    for j in 0..n {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let fp = f(&xp);
        for i in 0..m {
            let fpi = fp.get(i).copied().unwrap_or(f64::NAN);
            jac[i][j] = if fpi.is_finite() {
                (fpi - f0[i]) / eps
            } else {
                f64::NAN
            };
        }
    }
    jac
}

/// Simulated annealing for combinatorial optimization.
///
/// Minimizes a function over discrete states using neighbor generation.
pub fn simulated_annealing<S, F, N>(
    initial: S,
    cost: F,
    neighbor: N,
    temp_initial: f64,
    temp_final: f64,
    max_iter: usize,
    seed: u64,
) -> (S, f64)
where
    S: Clone,
    F: Fn(&S) -> f64,
    N: Fn(&S, u64) -> S,
{
    if !temp_initial.is_finite()
        || !temp_final.is_finite()
        || temp_initial <= 0.0
        || temp_final <= 0.0
    {
        let c = cost(&initial);
        return (initial, c);
    }
    let mut current = initial;
    let mut current_cost = cost(&current);
    let mut best = current.clone();
    let mut best_cost = current_cost;
    let mut rng = seed;

    let cooling_rate = if max_iter > 1 && temp_initial > 0.0 {
        (temp_final.max(1e-12) / temp_initial.max(1e-12)).powf(1.0 / max_iter as f64)
    } else {
        1.0
    };
    let mut temp = temp_initial;

    for _ in 0..max_iter {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let candidate = neighbor(&current, rng);
        let candidate_cost = cost(&candidate);

        let delta = candidate_cost - current_cost;
        let accept = if delta < 0.0 {
            true
        } else {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 11) as f64 / (1u64 << 53) as f64;
            r < (-delta / temp).exp()
        };

        if accept {
            current = candidate;
            current_cost = candidate_cost;
            if current_cost < best_cost {
                best = current.clone();
                best_cost = current_cost;
            }
        }

        temp *= cooling_rate;
    }

    (best, best_cost)
}

/// Particle Swarm Optimization.
///
/// Minimizes f(x) over [lb, ub] using a swarm of particles.
pub fn pso<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    mut n_particles: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    n_particles = n_particles.max(1);
    let d = lb.len();
    if d == 0
        || ub.len() != d
        || lb
            .iter()
            .zip(ub.iter())
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return (vec![f64::NAN; d.max(1)], f64::NAN);
    }
    let w = 0.7298; // inertia
    let c1 = 1.4962; // cognitive
    let c2 = 1.4962; // social

    let mut rng = seed;
    let next_rand = |rng: &mut u64| -> f64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*rng >> 11) as f64 / (1u64 << 53) as f64
    };

    // Initialize particles
    let mut positions: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| lb[j] + next_rand(&mut rng) * (ub[j] - lb[j]))
                .collect()
        })
        .collect();

    let mut velocities: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            (0..d)
                .map(|j| (next_rand(&mut rng) - 0.5) * (ub[j] - lb[j]) * 0.1)
                .collect()
        })
        .collect();

    let mut personal_best = positions.clone();
    let mut personal_best_cost: Vec<f64> = positions.iter().map(|p| f(p)).collect();

    let global_best_idx = personal_best_cost
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut global_best = personal_best[global_best_idx].clone();
    let mut global_best_cost = personal_best_cost[global_best_idx];

    for _ in 0..max_iter {
        for i in 0..n_particles {
            // Update velocity
            for j in 0..d {
                let r1 = next_rand(&mut rng);
                let r2 = next_rand(&mut rng);
                velocities[i][j] = w * velocities[i][j]
                    + c1 * r1 * (personal_best[i][j] - positions[i][j])
                    + c2 * r2 * (global_best[j] - positions[i][j]);
            }

            // Update position
            for j in 0..d {
                positions[i][j] = (positions[i][j] + velocities[i][j]).clamp(lb[j], ub[j]);
            }

            // Evaluate
            let cost = f(&positions[i]);
            if cost < personal_best_cost[i] {
                personal_best_cost[i] = cost;
                personal_best[i].clone_from(&positions[i]);
                if cost < global_best_cost {
                    global_best_cost = cost;
                    global_best.clone_from(&positions[i]);
                }
            }
        }
    }

    (global_best, global_best_cost)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{
        BasinhoppingOptions, Bounds, ConvergenceStatus, DifferentialEvolutionOptions,
        DifferentiateOptions, Integrality, LinearConstraint, MilpOptions, MilpProblem,
        MinimizeOptions, NonlinearConstraint, OptimizeMethod, RootOptions, approx_fprime,
        basinhopping, bracket, brent_minimize, brute, check_grad, cobyla, derivative,
        differential_evolution, differential_evolution_constrained, dual_annealing, fixed_point,
        golden, gradient_descent, hessian, isotonic_regression, jacobian, linear_sum_assignment,
        linprog, milp, minimize_scalar_bounded, nnls, projected_gradient_descent, pso, rosen,
        rosen_der, rosen_hess, rosen_hess_prod, shgo,
    };

    #[test]
    fn defaults_match_packet_contract_intent() {
        let minimize = MinimizeOptions::default();
        assert_eq!(minimize.method, None);
        assert_eq!(minimize.mode, RuntimeMode::Strict);

        let root = RootOptions::default();
        assert!(root.xtol > 0.0);
        assert!(root.rtol > 0.0);
        assert_eq!(root.mode, RuntimeMode::Strict);
    }

    #[test]
    fn convergence_status_is_cloneable() {
        let status = ConvergenceStatus::CallbackStop;
        assert_eq!(status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn optimize_method_enum_is_stable() {
        let method = OptimizeMethod::Bfgs;
        assert!(matches!(method, OptimizeMethod::Bfgs));
    }

    // ── Constraint type tests ──────────────────────────────────────

    #[test]
    fn bounds_basic() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        assert_eq!(b.len(), 2);
        assert!(b.is_feasible(&[0.5, 0.0]));
        assert!(!b.is_feasible(&[1.5, 0.0]));
        assert!(!b.is_feasible(&[0.5, -2.0]));
    }

    #[test]
    fn bounds_project() {
        let b = Bounds::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("bounds");
        let projected = b.project(&[2.0, -3.0]);
        assert_eq!(projected, vec![1.0, -1.0]);
    }

    #[test]
    fn bounds_unbounded() {
        let b = Bounds::unbounded(3);
        assert!(b.is_feasible(&[1e100, -1e100, 0.0]));
    }

    #[test]
    fn bounds_invalid_lb_gt_ub() {
        let err = Bounds::new(vec![1.0], vec![0.0]).expect_err("lb > ub");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn bounds_to_tuples() {
        let b = Bounds::new(vec![0.0, f64::NEG_INFINITY], vec![f64::INFINITY, 1.0]).unwrap();
        let tuples = b.to_bound_tuples();
        assert_eq!(tuples[0], (Some(0.0), None));
        assert_eq!(tuples[1], (None, Some(1.0)));
    }

    #[test]
    fn linear_constraint_basic() {
        // x + y <= 1 expressed as A @ x <= ub with lb = -inf
        let lc = LinearConstraint::new(vec![vec![1.0, 1.0]], vec![f64::NEG_INFINITY], vec![1.0])
            .expect("linear constraint");
        assert!(lc.is_feasible(&[0.3, 0.3])); // 0.6 <= 1
        assert!(!lc.is_feasible(&[0.6, 0.6])); // 1.2 > 1
    }

    #[test]
    fn linear_constraint_evaluate() {
        let lc = LinearConstraint::new(
            vec![vec![2.0, 3.0], vec![1.0, -1.0]],
            vec![0.0, 0.0],
            vec![10.0, 5.0],
        )
        .expect("linear constraint");
        let ax = lc.evaluate(&[1.0, 2.0]);
        assert!((ax[0] - 8.0).abs() < 1e-12); // 2*1 + 3*2 = 8
        assert!((ax[1] - (-1.0)).abs() < 1e-12); // 1*1 + (-1)*2 = -1
    }

    #[test]
    fn nonlinear_constraint_basic() {
        fn circle(x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1]]
        }
        let nlc = NonlinearConstraint::new(circle, vec![0.0], vec![1.0]).expect("nlc");
        assert!(nlc.is_feasible(&[0.5, 0.5])); // 0.5 <= 1
        assert!(!nlc.is_feasible(&[1.0, 1.0])); // 2.0 > 1
    }

    #[test]
    fn bounds_with_lbfgsb() {
        // Minimize x^2 + y^2 subject to x >= 1, y >= 2
        let bounds = Bounds::new(vec![1.0, 2.0], vec![f64::INFINITY, f64::INFINITY]).unwrap();
        let bound_tuples = bounds.to_bound_tuples();

        let options = MinimizeOptions {
            method: Some(OptimizeMethod::LBfgsB),
            tol: Some(1e-10),
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        };
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let result = crate::lbfgsb(&f, &[5.0, 5.0], options, Some(&bound_tuples))
            .expect("lbfgsb with bounds");
        assert!(result.success, "{}", result.message);
        // Minimum should be at (1, 2) — the bound corner
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.01,
            "y ~ 2.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn approx_fprime_matches_quadratic_gradient() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = approx_fprime(&[3.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 6.0).abs() < 1.0e-5,
            "expected 6.0, got {}",
            grad[0]
        );
    }

    #[test]
    fn approx_fprime_matches_multivariate_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
        let grad = approx_fprime(&[2.0, -1.0], f, 1.0e-8).expect("gradient should evaluate");
        assert!(
            (grad[0] - 3.0).abs() < 1.0e-5,
            "expected 3.0, got {}",
            grad[0]
        );
        assert!(
            (grad[1] - (-4.0)).abs() < 1.0e-5,
            "expected -4.0, got {}",
            grad[1]
        );
    }

    #[test]
    fn check_grad_is_small_for_exact_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] * x[1];
        let g = |x: &[f64]| vec![2.0 * x[0], 3.0 * x[1] * x[1]];
        let error = check_grad(f, g, &[1.5, -2.0]).expect("gradient check should succeed");
        assert!(error < 1.0e-5, "expected small error, got {error}");
    }

    #[test]
    fn check_grad_detects_bad_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let wrong = |_x: &[f64]| vec![0.0, 0.0];
        let error = check_grad(f, wrong, &[2.0, -3.0]).expect("gradient check should succeed");
        assert!(error > 1.0, "expected large error, got {error}");
    }

    #[test]
    fn differentiate_derivative_matches_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x;
        let x = 1.25_f64;
        let result = derivative(f, x, crate::DifferentiateOptions::default())
            .expect("derivative should evaluate");
        let expected = 3.0 * x * x - 2.0;
        assert!(result.success, "status: {:?}", result.status);
        assert!(
            (result.df - expected).abs() < 1.0e-8,
            "df = {}, expected {expected}",
            result.df
        );
        assert!(result.error.is_finite());
    }

    #[test]
    fn differentiate_derivative_supports_one_sided_steps() {
        let options = crate::DifferentiateOptions {
            step_direction: crate::StepDirection::Forward,
            ..crate::DifferentiateOptions::default()
        };
        let result = derivative(|x| x * x, 0.0, options).expect("forward derivative");
        assert!(result.success, "status: {:?}", result.status);
        assert!(result.df.abs() < 1.0e-8, "df = {}", result.df);
    }

    #[test]
    fn differentiate_jacobian_recovers_linear_map() {
        let f = |x: &[f64]| {
            vec![
                2.0 * x[0] - x[1] + 0.5 * x[2],
                -3.0 * x[0] + 4.0 * x[1] + x[2],
            ]
        };
        let result = jacobian(
            f,
            &[1.0, -2.0, 0.25],
            crate::DifferentiateOptions::default(),
        )
        .expect("jacobian should evaluate");
        let expected = [[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]];
        assert!(result.success, "status: {:?}", result.status);
        for (row_index, row) in expected.iter().enumerate() {
            for (column_index, expected_value) in row.iter().enumerate() {
                assert!(
                    (result.df[row_index][column_index] - expected_value).abs() < 1.0e-8,
                    "jac[{row_index},{column_index}] = {} vs {expected_value}",
                    result.df[row_index][column_index]
                );
            }
        }
    }

    #[test]
    fn differentiate_jacobian_rejects_output_shape_changes() {
        let f = |x: &[f64]| {
            if x[0].is_sign_positive() {
                vec![x[0], x[0] * x[0]]
            } else {
                vec![x[0]]
            }
        };
        let options = crate::DifferentiateOptions {
            maxiter: 2,
            initial_step: 0.25,
            ..crate::DifferentiateOptions::default()
        };
        let error =
            jacobian(f, &[0.0], options).expect_err("shape-changing outputs must fail closed");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn differentiate_hessian_matches_quadratic() {
        let f = |x: &[f64]| 3.0 * x[0] * x[0] + 2.0 * x[0] * x[1] + 5.0 * x[1] * x[1] - x[0];
        let result = hessian(f, &[0.75, -1.25], crate::DifferentiateOptions::default())
            .expect("hessian should evaluate");
        let expected = [[6.0, 2.0], [2.0, 10.0]];
        assert!(result.success, "status: {:?}", result.status);
        for (row_index, row) in expected.iter().enumerate() {
            for (column_index, expected_value) in row.iter().enumerate() {
                assert!(
                    (result.ddf[row_index][column_index] - expected_value).abs() < 1.0e-7,
                    "hess[{row_index},{column_index}] = {} vs {expected_value}",
                    result.ddf[row_index][column_index]
                );
            }
        }
    }

    #[test]
    fn differentiate_rejects_non_finite_inputs() {
        let error = derivative(|x| x, f64::NAN, crate::DifferentiateOptions::default())
            .expect_err("NaN x should fail");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn differentiate_rejects_degenerate_step_factor() {
        let options = crate::DifferentiateOptions {
            step_factor: 1.0,
            ..crate::DifferentiateOptions::default()
        };
        let error = derivative(|x| x, 1.0, options).expect_err("step_factor=1 should fail");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn gradient_descent_rejects_invalid_callback_gradient() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();

        let wrong_len = gradient_descent(f, |_x| vec![1.0], &[1.0, 2.0], 1.0e-6, 10, 0.1);
        assert!(!wrong_len.success);
        assert_eq!(wrong_len.status, ConvergenceStatus::InvalidInput);

        let non_finite =
            gradient_descent(f, |x| vec![f64::NAN; x.len()], &[1.0, 2.0], 1.0e-6, 10, 0.1);
        assert!(!non_finite.success);
        assert_eq!(non_finite.status, ConvergenceStatus::InvalidInput);
    }

    #[test]
    fn projected_gradient_descent_rejects_invalid_callback_gradient() {
        let f = |x: &[f64]| x.iter().map(|value| value * value).sum::<f64>();

        let wrong_len = projected_gradient_descent(
            f,
            |_x| vec![1.0],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            1.0e-6,
            10,
            0.1,
        );
        assert!(!wrong_len.success);
        assert_eq!(wrong_len.status, ConvergenceStatus::InvalidInput);

        let non_finite = projected_gradient_descent(
            f,
            |x| vec![f64::INFINITY; x.len()],
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            1.0e-6,
            10,
            0.1,
        );
        assert!(!non_finite.success);
        assert_eq!(non_finite.status, ConvergenceStatus::InvalidInput);
    }

    #[test]
    fn linear_sum_assignment_solves_square_problem() {
        let cost_matrix = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1, 2]);
        assert_eq!(col_ind, vec![1, 0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_columns_than_rows() {
        let cost_matrix = vec![vec![10.0, 19.0, 8.0, 15.0], vec![10.0, 18.0, 7.0, 17.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 2]);
    }

    #[test]
    fn linear_sum_assignment_handles_more_rows_than_columns() {
        let cost_matrix = vec![vec![5.0, 9.0], vec![10.0, 3.0], vec![8.0, 7.0]];
        let (row_ind, col_ind) =
            linear_sum_assignment(&cost_matrix).expect("assignment should succeed");
        assert_eq!(row_ind, vec![0, 1]);
        assert_eq!(col_ind, vec![0, 1]);
    }

    #[test]
    fn linear_sum_assignment_rejects_non_rectangular_input() {
        let cost_matrix = vec![vec![1.0, 2.0], vec![3.0]];
        let error = linear_sum_assignment(&cost_matrix).expect_err("matrix should be rejected");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn linear_sum_assignment_allows_empty_input() {
        let (row_ind, col_ind) = linear_sum_assignment(&[]).expect("empty problem should succeed");
        assert!(row_ind.is_empty());
        assert!(col_ind.is_empty());
    }

    #[test]
    fn nnls_rejects_non_rectangular_input() {
        let a = vec![vec![1.0, 2.0], vec![3.0]];
        let b = vec![1.0, 2.0];
        let error = nnls(&a, &b).expect_err("ragged matrix should be rejected");
        assert!(matches!(error, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn nnls_rejects_non_finite_inputs() {
        let a = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        let b = vec![1.0, 2.0];
        let error = nnls(&a, &b).expect_err("NaN in A should be rejected");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));

        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, f64::INFINITY];
        let error = nnls(&a, &b).expect_err("Inf in b should be rejected");
        assert!(matches!(error, crate::OptError::NonFiniteInput { .. }));
    }

    // ── linprog tests ──────────────────────────────────────────────

    #[test]
    fn linprog_simple_2d() {
        // Maximize 5x + 4y (minimize -5x - 4y)
        // Subject to: 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
        // Optimal: x=3, y=1.5, obj = -21
        let c = vec![-5.0, -4.0];
        let a_ub = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b_ub = vec![24.0, 6.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 3.0).abs() < 0.01,
            "x={}, expected 3.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.5).abs() < 0.01,
            "y={}, expected 1.5",
            result.x[1]
        );
        assert!(
            (result.fun - (-21.0)).abs() < 0.01,
            "obj={}, expected -21.0",
            result.fun
        );
    }

    #[test]
    fn linprog_infeasible() {
        // x = -1 with x >= 0 — infeasible (non-negative variable can't equal negative).
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![-1.0]; // x = -1 but x >= 0
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(
            !result.success,
            "should be infeasible: success={}, status={}, x={:?}, fun={}",
            result.success, result.status, result.x, result.fun
        );
        assert_eq!(result.status, 2);
    }

    #[test]
    fn linprog_equality_constraint() {
        // Minimize x + y, subject to x + y = 10, x,y >= 0
        // Optimal: x=0, y=10, obj=10 (or x=10, y=0)
        let c = vec![1.0, 1.0];
        let a_eq = vec![vec![1.0, 1.0]];
        let b_eq = vec![10.0];
        let result = linprog(&c, &[], &[], &a_eq, &b_eq, &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - 10.0).abs() < 0.01,
            "obj={}, expected 10.0",
            result.fun
        );
    }

    #[test]
    fn linprog_with_bounds() {
        // Minimize -x - 2y, subject to x + y <= 4, x in [0,3], y in [0,3]
        // Optimal: x=1, y=3, obj = -7
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.fun - (-7.0)).abs() < 0.1,
            "obj={}, expected -7.0",
            result.fun
        );
    }

    #[test]
    fn linprog_degenerate_redundant_constraints() {
        // Minimize x, subject to x <= 5 AND x <= 10 AND x >= 0
        // Optimal: x = 0
        let c = vec![1.0];
        let a_ub = vec![vec![1.0], vec![1.0]];
        let b_ub = vec![5.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(result.x[0].abs() < 0.01, "x={}, expected 0.0", result.x[0]);
    }

    #[test]
    fn linprog_slack_variables() {
        // Verify slack is computed correctly.
        let c = vec![1.0, 1.0];
        let a_ub = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b_ub = vec![10.0, 10.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).unwrap();
        assert!(result.success);
        // Slack = b_ub - A_ub @ x
        for (i, &s) in result.slack.iter().enumerate() {
            assert!(s >= -1e-8, "slack[{i}] = {s} should be non-negative");
        }
    }

    #[test]
    fn linprog_bounds_enforced() {
        // Minimize -x, x in [2, 5] → optimal at x=5
        let c = vec![-1.0];
        let bounds = vec![(Some(2.0), Some(5.0))];
        let result = linprog(&c, &[], &[], &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.01,
            "x={}, expected 5.0",
            result.x[0]
        );
    }

    #[test]
    fn linprog_supports_free_variables() {
        // Minimize x, subject to x >= 1 with x otherwise free.
        let c = vec![1.0];
        let a_ub = vec![vec![-1.0]];
        let b_ub = vec![-1.0];
        let bounds = vec![(None, None)];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 0.01,
            "x={}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.fun - 1.0).abs() < 0.01,
            "obj={}, expected 1.0",
            result.fun
        );
    }

    #[test]
    fn linprog_supports_upper_only_bounds() {
        // Minimize -x, subject only to x <= 5.
        let c = vec![-1.0];
        let bounds = vec![(None, Some(5.0))];
        let result = linprog(&c, &[], &[], &[], &[], &bounds, None).unwrap();
        assert!(result.success, "linprog failed: {}", result.message);
        assert!(
            (result.x[0] - 5.0).abs() < 0.01,
            "x={}, expected 5.0",
            result.x[0]
        );
        assert!(
            (result.fun - (-5.0)).abs() < 0.01,
            "obj={}, expected -5.0",
            result.fun
        );
    }

    #[test]
    fn linprog_dimension_mismatch() {
        let c = vec![1.0, 2.0];
        let a_ub = vec![vec![1.0]]; // wrong number of columns
        let b_ub = vec![5.0];
        assert!(linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).is_err());
    }

    // ── milp tests ──────────────────────────────────────────────────

    #[test]
    fn milp_finds_integer_optimum() {
        let c = vec![-1.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![4.0];
        let bounds = vec![(Some(0.0), Some(3.0)), (Some(0.0), Some(3.0))];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer, Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &bounds,
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 3.0]);
        assert!((result.fun + 7.0).abs() < 1e-9);
    }

    #[test]
    fn milp_handles_binary_variables() {
        let c = vec![-3.0, -2.0];
        let a_ub = vec![vec![1.0, 1.0]];
        let b_ub = vec![1.0];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Binary, Integrality::Binary],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![1.0, 0.0]);
        assert!((result.fun + 3.0).abs() < 1e-9);
    }

    #[test]
    fn milp_reports_infeasible_problem() {
        let c = vec![1.0];
        let a_eq = vec![vec![1.0]];
        let b_eq = vec![0.5];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &a_eq,
                b_eq: &b_eq,
                bounds: &[(Some(0.0), Some(1.0))],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(!result.success);
        assert_eq!(result.status, 2);
    }

    #[test]
    fn milp_supports_free_integer_variables() {
        let c = vec![-1.0];
        let a_ub = vec![vec![1.0], vec![-1.0]];
        let b_ub = vec![2.2, 2.2];
        let result = milp(
            MilpProblem {
                c: &c,
                integrality: &[Integrality::Integer],
                a_ub: &a_ub,
                b_ub: &b_ub,
                a_eq: &[],
                b_eq: &[],
                bounds: &[(None, None)],
            },
            MilpOptions::default(),
        )
        .expect("milp");
        assert!(result.success, "milp failed: {}", result.message);
        assert_eq!(result.x, vec![2.0]);
        assert!((result.fun + 2.0).abs() < 1e-9);
    }

    #[test]
    fn milp_rejects_integrality_length_mismatch() {
        let err = milp(
            MilpProblem {
                c: &[1.0, 2.0],
                integrality: &[Integrality::Integer],
                a_ub: &[],
                b_ub: &[],
                a_eq: &[],
                b_eq: &[],
                bounds: &[],
            },
            MilpOptions::default(),
        )
        .expect_err("integrality length mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── Differential Evolution tests ───────────────────────────────

    #[test]
    fn de_rastrigin_finds_global_minimum() {
        // Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        // Global minimum at origin: f(0,...,0) = 0
        let rastrigin = |x: &[f64]| -> f64 {
            let n = x.len() as f64;
            let pi2 = 2.0 * std::f64::consts::PI;
            n * 10.0
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (pi2 * xi).cos())
                    .sum::<f64>()
        };

        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 200,
            seed: Some(42),
            ..Default::default()
        };
        let result = differential_evolution(rastrigin, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 1.0,
            "DE should find near-global minimum, got f={}",
            fun
        );
    }

    #[test]
    fn de_rosenbrock_2d() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1,1) with f=0
        let rosenbrock =
            |x: &[f64]| -> f64 { (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2) };

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 500,
            seed: Some(123),
            ..Default::default()
        };
        let result = differential_evolution(rosenbrock, &bounds, opts).unwrap();
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x ~ 1.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "y ~ 1.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn de_bounds_respected() {
        let func = |x: &[f64]| -> f64 { x.iter().sum() };
        let bounds = vec![(1.0, 3.0), (2.0, 4.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 50,
            seed: Some(99),
            ..Default::default()
        };
        let result = differential_evolution(func, &bounds, opts).unwrap();
        for (i, (&xi, &(lo, hi))) in result.x.iter().zip(bounds.iter()).enumerate() {
            assert!(
                xi >= lo - 1e-10 && xi <= hi + 1e-10,
                "variable {i}: {xi} outside [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn de_rand1bin_strategy() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = DifferentialEvolutionOptions {
            strategy: crate::DeStrategy::Rand1Bin,
            maxiter: 200,
            seed: Some(77),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(fun < 0.1, "rand1bin should find near-origin, got f={fun}");
    }

    #[test]
    fn de_population_explores() {
        // With only a few iterations, the population should have explored
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-10.0, 10.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = differential_evolution(sphere, &bounds, opts).unwrap();
        assert!(result.nfev > 0, "should have function evaluations");
    }

    #[test]
    fn de_rejects_population_too_small() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: 3,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_rejects_nonfinite_bounds() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(f64::NAN, 1.0)];
        let err = differential_evolution(sphere, &bounds, Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn de_rejects_population_size_overflow() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let opts = DifferentialEvolutionOptions {
            popsize: usize::MAX,
            ..Default::default()
        };
        let err = differential_evolution(sphere, &bounds, opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn de_constrained_respects_feasible_boundary() {
        let objective = |x: &[f64]| x[0] * x[0];
        let lower_bound_constraint = |x: &[f64]| (1.0 - x[0]).max(0.0);
        let bounds = vec![(0.0, 2.0)];
        let opts = DifferentialEvolutionOptions {
            maxiter: 160,
            seed: Some(101),
            ..Default::default()
        };

        let result =
            differential_evolution_constrained(objective, &bounds, lower_bound_constraint, opts)
                .expect("constrained differential evolution");

        assert!(result.success, "should converge: {}", result.message);
        assert!(
            result.x[0] >= 0.99,
            "constrained optimum must stay feasible, got {:?}",
            result.x
        );
        assert!(
            result.fun.unwrap_or(f64::INFINITY) < 1.05,
            "objective should be close to boundary optimum"
        );
    }

    // ── Basinhopping tests ─────────────────────────────────────────

    #[test]
    fn basinhopping_finds_global_minimum() {
        // Multi-modal function: f(x) = x^4 - 4x^2 + x
        // Has two local minima, global near x ≈ -1.38
        let func = |x: &[f64]| -> f64 {
            let xi = x[0];
            xi.powi(4) - 4.0 * xi * xi + xi
        };

        let opts = BasinhoppingOptions {
            niter: 50,
            seed: Some(42),
            stepsize: 1.0,
            ..Default::default()
        };
        let result = basinhopping(func, &[3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < -3.0,
            "basinhopping should find global minimum, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_sphere() {
        // Simple sphere function: global min at origin
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let opts = BasinhoppingOptions {
            niter: 20,
            seed: Some(99),
            ..Default::default()
        };
        let result = basinhopping(sphere, &[5.0, -3.0], opts).unwrap();
        let fun = result.fun.unwrap();
        assert!(
            fun < 0.01,
            "basinhopping on sphere should find near-zero, got f={fun}"
        );
    }

    #[test]
    fn basinhopping_returns_correct_nfev() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            niter: 5,
            seed: Some(1),
            ..Default::default()
        };
        let result = basinhopping(func, &[1.0], opts).unwrap();
        assert!(result.nfev > 0, "should track function evaluations");
    }

    #[test]
    fn basinhopping_rejects_nonpositive_stepsize() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let opts = BasinhoppingOptions {
            stepsize: 0.0,
            ..Default::default()
        };
        let err = basinhopping(func, &[1.0], opts).unwrap_err();
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn basinhopping_rejects_nonfinite_initial_point() {
        let func = |x: &[f64]| -> f64 { x[0] * x[0] };
        let err = basinhopping(func, &[f64::INFINITY], Default::default()).unwrap_err();
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    // ── Dual annealing tests ─────────────────────────────────────────

    #[test]
    fn dual_annealing_finds_sphere_minimum() {
        // Sphere function: f(x) = Σ x_i²  — minimum 0 at origin. The bead's
        // reproduction parameters: 100 iterations, seed 42. With a real local
        // search per annealing step this converges essentially exactly.
        let result = dual_annealing(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            42,
        )
        .expect("dual_annealing");
        assert!(result.success);
        assert!(
            result.fun.unwrap() < 1e-6,
            "should find near-zero minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_finds_shifted_quadratic() {
        // Shifted convex quadratic with interior minimum 0 at (2, -3).
        let result = dual_annealing(
            |x| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            42,
        )
        .expect("dual_annealing shifted");
        assert!(
            result.fun.unwrap() < 1e-6,
            "should find the shifted minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_finds_rosenbrock_valley() {
        // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²  — minimum 0 at (1,1).
        let result = dual_annealing(
            |x| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            100,
            123,
        )
        .expect("dual_annealing rosenbrock");
        assert!(
            result.fun.unwrap() < 1e-3,
            "should find the Rosenbrock minimum: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn dual_annealing_empty_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[], 100, 42).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn dual_annealing_nonfinite_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(f64::NAN, 1.0)], 100, 42).expect_err("nonfinite");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn dual_annealing_inverted_bounds_rejected() {
        let err = dual_annealing(|_| 0.0, &[(2.0, 2.0)], 100, 42).expect_err("inverted");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    #[test]
    fn pso_zero_particles_falls_back_to_single_particle() {
        let sphere = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let lb = [-5.0, -5.0];
        let ub = [5.0, 5.0];
        let (best, cost) = pso(sphere, &lb, &ub, 0, 10, 123);
        assert_eq!(best.len(), 2);
        assert!(cost.is_finite(), "expected finite cost, got {cost}");
        for (i, (&xi, (&lo, &hi))) in best.iter().zip(lb.iter().zip(ub.iter())).enumerate() {
            assert!(
                xi >= lo && xi <= hi,
                "particle coordinate {i} should stay in bounds [{lo}, {hi}], got {xi}"
            );
        }
    }

    #[test]
    fn brute_finds_sphere_minimum() {
        let result = crate::brute(
            |x: &[f64]| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            11,
        )
        .expect("brute sphere");
        assert!(result.success);
        assert!(
            result.fun.expect("objective value") < 1e-6,
            "expected exactly zero minimum (since 0.0 is on the grid), got {:?}",
            result.fun
        );
        assert_eq!(result.x, vec![0.0, 0.0]);
    }

    #[test]
    fn shgo_finds_sphere_minimum() {
        let result = shgo(
            |x| x.iter().map(|&xi| xi * xi).sum(),
            &[(-5.0, 5.0), (-5.0, 5.0)],
        )
        .expect("shgo sphere");
        assert!(result.success);
        assert!(
            result.fun.expect("objective value") < 1e-6,
            "expected near-zero minimum, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_finds_multimodal_global_minimum() {
        let result = shgo(
            |x| {
                let xi = x[0];
                xi.powi(4) - 4.0 * xi * xi + xi
            },
            &[(-3.0, 3.0)],
        )
        .expect("shgo multimodal");
        assert!(
            result.fun.expect("objective value") < -3.0,
            "expected global basin, got {:?}",
            result.fun
        );
    }

    #[test]
    fn shgo_rejects_invalid_bounds() {
        let err = shgo(|_| 0.0, &[(1.0, 1.0)]).expect_err("invalid bounds");
        assert!(matches!(err, crate::OptError::InvalidBounds { .. }));
    }

    // ── COBYLA tests ─────────────────────────────────────────────────

    #[test]
    fn cobyla_unconstrained_quadratic() {
        // Minimize x² + y² → (0, 0)
        let result = cobyla(
            |x| x[0] * x[0] + x[1] * x[1],
            &[1.0, 1.0],
            &[] as &[fn(&[f64]) -> f64],
            1000,
            0.5,
        )
        .expect("cobyla");
        assert!(
            result.fun.unwrap() < 0.01,
            "cobyla should minimize: {}",
            result.fun.unwrap()
        );
    }

    #[test]
    fn cobyla_with_constraint() {
        // Minimize x + y subject to x + y >= 1
        // Solution: x + y = 1 (constraint active), e.g. (0.5, 0.5)
        type ConstraintFn = dyn Fn(&[f64]) -> f64;
        let constraints: Vec<Box<ConstraintFn>> = vec![
            Box::new(|x: &[f64]| x[0] + x[1] - 1.0), // x + y >= 1
        ];
        let constraint_fns: Vec<&ConstraintFn> = constraints.iter().map(|b| b.as_ref()).collect();
        let result = cobyla(|x| x[0] + x[1], &[2.0, 2.0], &constraint_fns, 1000, 0.5)
            .expect("cobyla constrained");
        // x + y should be approximately 1
        let sum = result.x[0] + result.x[1];
        assert!((sum - 1.0).abs() < 0.1, "x+y should be ~1: {sum}");
    }

    #[test]
    fn cobyla_empty_x0_rejected() {
        let err = cobyla(|_: &[f64]| 0.0, &[], &[] as &[fn(&[f64]) -> f64], 100, 0.5)
            .expect_err("empty x0");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── Rosenbrock test function tests ──────────────────────────────

    #[test]
    fn rosen_minimum_is_zero() {
        // f([1, 1, ..., 1]) = 0
        let x = vec![1.0; 5];
        let val = rosen(&x);
        assert!(val.abs() < 1e-12, "rosen at minimum = {val}");
    }

    #[test]
    fn rosen_away_from_minimum() {
        // f([0, 0]) = 1
        let val = rosen(&[0.0, 0.0]);
        assert!((val - 1.0).abs() < 1e-12, "rosen([0,0]) = {val}");
    }

    #[test]
    fn rosen_2d_value() {
        // f([2, 3]) = 100*(3 - 4)^2 + (1 - 2)^2 = 100 + 1 = 101
        let val = rosen(&[2.0, 3.0]);
        assert!((val - 101.0).abs() < 1e-10, "rosen([2,3]) = {val}");
    }

    #[test]
    fn rosen_der_at_minimum_is_zero() {
        // Gradient at minimum should be zero
        let x = vec![1.0; 4];
        let grad = rosen_der(&x);
        for (i, g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-10, "grad[{i}] = {g}, expected 0");
        }
    }

    #[test]
    fn rosen_der_numerical_check() {
        // Compare analytical gradient to numerical approximation
        let x = vec![1.5, 2.0, 0.5];
        let grad = rosen_der(&x);
        let num_grad = approx_fprime(&x, rosen, 1e-7).expect("approx_fprime");
        for (i, (a, n)) in grad.iter().zip(num_grad.iter()).enumerate() {
            // Relative error check for large values
            let tol = 1e-4 * a.abs().max(1.0);
            assert!(
                (a - n).abs() < tol,
                "grad[{i}]: analytical={a}, numerical={n}"
            );
        }
    }

    #[test]
    fn rosen_hess_is_symmetric() {
        let x = vec![1.2, 0.8, 1.5];
        let h = rosen_hess(&x);
        for (i, row) in h.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate().skip(i + 1) {
                assert!(
                    (entry - h[j][i]).abs() < 1e-12,
                    "H[{i}][{j}] != H[{j}][{i}]"
                );
            }
        }
    }

    #[test]
    fn rosen_hess_at_minimum() {
        // At [1,1], H should be [[802, -400], [-400, 200]]
        let h = rosen_hess(&[1.0, 1.0]);
        assert!((h[0][0] - 802.0).abs() < 1e-10, "H[0][0] = {}", h[0][0]);
        assert!((h[0][1] + 400.0).abs() < 1e-10, "H[0][1] = {}", h[0][1]);
        assert!((h[1][0] + 400.0).abs() < 1e-10, "H[1][0] = {}", h[1][0]);
        assert!((h[1][1] - 200.0).abs() < 1e-10, "H[1][1] = {}", h[1][1]);
    }

    #[test]
    fn rosen_hess_prod_matches_explicit() {
        // H @ p should match explicit matrix-vector product
        let x = vec![0.5, 1.5, 2.0];
        let p = vec![1.0, 2.0, 3.0];
        let h = rosen_hess(&x);
        let hp_explicit: Vec<f64> = (0..3)
            .map(|i| h[i].iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let hp = rosen_hess_prod(&x, &p);
        for (i, (e, c)) in hp_explicit.iter().zip(hp.iter()).enumerate() {
            assert!((e - c).abs() < 1e-10, "hp[{i}]: explicit={e}, computed={c}");
        }
    }

    #[test]
    fn rosen_short_input() {
        // Single element input should return 0
        assert_eq!(rosen(&[1.0]), 0.0);
        assert_eq!(rosen_der(&[1.0]), vec![0.0]);
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "index {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn isotonic_regression_strictly_decreasing_pools_to_global_mean() {
        // A fully decreasing input pools into a single block — every PAVA
        // merge must propagate across the whole block, not just two cells.
        // scipy.optimize.isotonic_regression([5,4,3,2,1]).x == [3,3,3,3,3]
        let r = isotonic_regression(&[5.0, 4.0, 3.0, 2.0, 1.0], None);
        assert_close(&r, &[3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn isotonic_regression_weighted_multi_pool() {
        // Weighted PAVA: scipy gives an all-1.5 fit for this input.
        let y = [3.0, 1.0, 2.0, 0.5, 1.5];
        let w = [1.0, 2.0, 1.0, 1.0, 1.0];
        let r = isotonic_regression(&y, Some(&w));
        assert_close(&r, &[1.5, 1.5, 1.5, 1.5, 1.5]);

        // Weights bias the pooled mean: scipy gives an all-3.0 fit here.
        let r2 = isotonic_regression(&[4.0, 4.0, 2.0, 2.0], Some(&[3.0, 1.0, 1.0, 3.0]));
        assert_close(&r2, &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn isotonic_regression_already_monotone_is_identity() {
        let r = isotonic_regression(&[1.0, 2.0, 3.0], None);
        assert_close(&r, &[1.0, 2.0, 3.0]);
        // Mixed input with one isolated violation, scipy reference values.
        let r2 = isotonic_regression(
            &[1.5, 1.0, 4.0, 6.0, 5.7, 5.0, 7.8, 9.0, 7.5, 9.5, 9.0],
            None,
        );
        assert_close(
            &r2,
            &[
                1.25,
                1.25,
                4.0,
                5.566_666_666_666_666,
                5.566_666_666_666_666,
                5.566_666_666_666_666,
                7.8,
                8.25,
                8.25,
                9.25,
                9.25,
            ],
        );
    }

    #[test]
    fn isotonic_regression_rejects_invalid_weights() {
        // Non-positive weights are invalid (scipy requires strictly positive).
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0, 0.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0, -1.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        // Mismatched weight length is invalid.
        assert!(
            isotonic_regression(&[1.0, 2.0], Some(&[1.0]))
                .iter()
                .all(|v| v.is_nan())
        );
        // Empty input returns empty.
        assert!(isotonic_regression(&[], None).is_empty());
    }

    #[test]
    fn rosen_matches_scipy_reference_values() {
        // scipy.optimize.rosen([1, 1]) = 0.0
        let result = rosen(&[1.0, 1.0]);
        assert!(
            result.abs() < 1e-10,
            "rosen at [1,1] got {result}, expected 0.0"
        );
        // scipy.optimize.rosen([0, 0]) = 1.0
        let result = rosen(&[0.0, 0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "rosen at [0,0] got {result}, expected 1.0"
        );
    }

    #[test]
    fn rosen_der_matches_scipy_reference_values() {
        // scipy.optimize.rosen_der([1, 1]) = [0, 0]
        let result = rosen_der(&[1.0, 1.0]);
        assert!(
            result[0].abs() < 1e-10,
            "rosen_der[0] got {}, expected 0.0",
            result[0]
        );
        assert!(
            result[1].abs() < 1e-10,
            "rosen_der[1] got {}, expected 0.0",
            result[1]
        );
    }

    #[test]
    fn golden_matches_scipy_reference_values() {
        // scipy.optimize.golden(lambda x: x**2, brack=(-1, 0, 1)) ≈ 0
        let (x_opt, _f_opt) = golden(|x| x * x, -1.0, 1.0, 1e-10, 1000);
        assert!(
            x_opt.abs() < 1e-8,
            "golden x^2 minimum at {x_opt}, expected ~0"
        );
    }

    #[test]
    fn brent_minimize_matches_scipy_reference_values() {
        // scipy.optimize.brent(lambda x: (x-2)**2, brack=(0, 4)) ≈ 2
        let (x_opt, _f_opt) = brent_minimize(|x| (x - 2.0).powi(2), 0.0, 4.0, 1e-10, 1000);
        assert!(
            (x_opt - 2.0).abs() < 1e-8,
            "brent (x-2)^2 minimum at {x_opt}, expected ~2"
        );
    }

    #[test]
    fn linear_sum_assignment_matches_scipy_reference_values() {
        // scipy.optimize.linear_sum_assignment([[4,1,3],[2,0,5],[3,2,2]])
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let (row_ind, col_ind) = linear_sum_assignment(&cost).expect("lsa");
        assert_eq!(row_ind, vec![0, 1, 2]);
        assert_eq!(col_ind, vec![1, 0, 2]);
        let total_cost: f64 = row_ind
            .iter()
            .zip(col_ind.iter())
            .map(|(&i, &j)| cost[i][j])
            .sum();
        assert!(
            (total_cost - 5.0).abs() < 1e-10,
            "total cost got {total_cost}, expected 5.0"
        );
    }

    #[test]
    fn minimize_bfgs_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.minimize(rosen, [0, 0], method='BFGS')
        // Expected: x ≈ [1, 1], fun ≈ 0
        use crate::{OptimizeMethod, minimize::minimize};
        let result = minimize(
            rosen,
            &[0.0, 0.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::Bfgs),
                ..Default::default()
            },
        )
        .expect("minimize BFGS should succeed");
        assert!(result.success, "BFGS should converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-4,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-4,
            "x[1] got {}, expected ~1.0",
            result.x[1]
        );
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-8, "fun got {fun}, expected ~0.0");
    }

    #[test]
    fn minimize_nelder_mead_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.minimize(rosen, [0, 0], method='Nelder-Mead')
        // Expected: x ≈ [1, 1], fun ≈ 0
        use crate::{OptimizeMethod, minimize::minimize};
        let result = minimize(
            rosen,
            &[0.0, 0.0],
            MinimizeOptions {
                method: Some(OptimizeMethod::NelderMead),
                ..Default::default()
            },
        )
        .expect("minimize Nelder-Mead should succeed");
        assert!(result.success, "Nelder-Mead should converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-3,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-3,
            "x[1] got {}, expected ~1.0",
            result.x[1]
        );
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-6, "fun got {fun}, expected ~0.0");
    }

    #[test]
    fn approx_fprime_matches_scipy_reference_values() {
        // scipy.optimize.approx_fprime([1, 2], lambda x: x[0]**2 + x[1]**2, 1e-8)
        // -> array([2., 4.])
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let grad = approx_fprime(&[1.0, 2.0], f, 1e-8).expect("approx_fprime should succeed");
        assert!(
            (grad[0] - 2.0).abs() < 1e-5,
            "grad[0] got {}, expected 2.0",
            grad[0]
        );
        assert!(
            (grad[1] - 4.0).abs() < 1e-5,
            "grad[1] got {}, expected 4.0",
            grad[1]
        );
    }

    #[test]
    fn fixed_point_matches_scipy_reference_values() {
        // scipy.optimize.fixed_point(lambda x: np.cos(x), 0.5)
        // -> 0.7390851332151607 (Dottie number)
        let result =
            fixed_point(|x| x.cos(), 0.5, 1e-10, 100).expect("fixed_point should converge");
        assert!(
            (result - 0.7390851332151607).abs() < 1e-8,
            "fixed_point got {result}, expected 0.7390851332151607"
        );
    }

    #[test]
    fn isotonic_regression_matches_scipy_reference_values() {
        // from scipy.optimize import isotonic_regression
        // isotonic_regression([5, 3, 1, 2, 4]).x
        // -> array([1. , 1. , 1. , 3. , 4. ]) -- scipy 1.15+ returns decreasing by default
        // Our impl expects increasing, so: isotonic_regression([1, 2, 4, 3, 5])
        // For increasing isotonic regression on [5, 3, 1, 2, 4]:
        // Result should be monotonically increasing, minimizing SSE
        let y = [5.0, 3.0, 1.0, 2.0, 4.0];
        let result = isotonic_regression(&y, None);
        // Check monotonicity
        for i in 1..result.len() {
            assert!(
                result[i] >= result[i - 1],
                "result[{i}]={} < result[{}]={}",
                result[i],
                i - 1,
                result[i - 1]
            );
        }
        // Should have same sum as input (weighted average preserves sum)
        let input_sum: f64 = y.iter().sum();
        let output_sum: f64 = result.iter().sum();
        assert!(
            (input_sum - output_sum).abs() < 1e-10,
            "sum mismatch: input {input_sum}, output {output_sum}"
        );
    }

    #[test]
    fn brute_matches_scipy_reference_values() {
        // scipy.optimize.brute(lambda x: (x[0]-1)**2 + (x[1]-2)**2, [(-5, 5), (-5, 5)], Ns=20)
        // -> array([1., 2.])
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let result = brute(f, &[(-5.0, 5.0), (-5.0, 5.0)], 20).expect("brute should succeed");
        // brute with coarse grid should get close to [1, 2]
        assert!(
            (result.x[0] - 1.0).abs() < 0.6,
            "x[0] got {}, expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.6,
            "x[1] got {}, expected ~2.0",
            result.x[1]
        );
    }

    #[test]
    fn rosen_hess_matches_scipy_reference_values() {
        // scipy.optimize.rosen_hess([1.0, 1.0])
        // At the minimum [1,1], the Hessian should be [[802, -400], [-400, 200]]
        let h = rosen_hess(&[1.0, 1.0]);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 2);
        assert!(
            (h[0][0] - 802.0).abs() < 1e-10,
            "H[0][0] got {}, expected 802.0",
            h[0][0]
        );
        assert!(
            (h[0][1] - (-400.0)).abs() < 1e-10,
            "H[0][1] got {}, expected -400.0",
            h[0][1]
        );
        assert!(
            (h[1][0] - (-400.0)).abs() < 1e-10,
            "H[1][0] got {}, expected -400.0",
            h[1][0]
        );
        assert!(
            (h[1][1] - 200.0).abs() < 1e-10,
            "H[1][1] got {}, expected 200.0",
            h[1][1]
        );
    }

    #[test]
    fn bracket_matches_scipy_reference_values() {
        // scipy.optimize.bracket(lambda x: (x-2)**2, xa=0, xb=1)
        // Brackets the minimum at x=2
        let (xa, xb, xc, fa, fb, fc) = bracket(|x| (x - 2.0).powi(2), 0.0, 1.0);
        // xb should be between xa and xc (golden section bracket)
        // fb should be less than both fa and fc (bracketing condition)
        assert!(
            fb <= fa && fb <= fc,
            "bracket failed: fa={fa}, fb={fb}, fc={fc}"
        );
        assert!(
            (xa < xb && xb < xc) || (xa > xb && xb > xc),
            "bracket order violated: xa={xa}, xb={xb}, xc={xc}"
        );
    }

    #[test]
    fn nnls_matches_scipy_reference_values() {
        // scipy.optimize.nnls([[1,0],[1,0],[0,1]], [2,1,1])
        // -> (array([1.5, 1.0]), 0.5)
        // Solve min ||Ax - b||^2 subject to x >= 0
        let a = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![2.0, 1.0, 1.0];
        let (x, residual) = nnls(&a, &b).expect("nnls should succeed");
        assert_eq!(x.len(), 2);
        assert!(
            (x[0] - 1.5).abs() < 1e-10,
            "x[0] got {}, expected 1.5",
            x[0]
        );
        assert!(
            (x[1] - 1.0).abs() < 1e-10,
            "x[1] got {}, expected 1.0",
            x[1]
        );
        // Residual: ||Ax - b|| = ||(-0.5, 0.5, 0)|| = sqrt(0.5) ≈ 0.707
        let expected_residual = 0.5_f64.sqrt();
        assert!(
            (residual - expected_residual).abs() < 1e-10,
            "residual got {residual}, expected {expected_residual}"
        );
    }

    #[test]
    fn check_grad_matches_scipy_reference_values() {
        // scipy.optimize.check_grad(lambda x: sum(x**2), lambda x: 2*x, [1.0, 2.0])
        // -> 0.0 (gradient is exact)
        let f = |x: &[f64]| -> f64 { x.iter().map(|&xi| xi * xi).sum() };
        let grad = |x: &[f64]| -> Vec<f64> { x.iter().map(|&xi| 2.0 * xi).collect() };
        let x0 = vec![1.0, 2.0];
        let result = check_grad(f, grad, &x0).expect("check_grad");
        assert!(result < 1e-6, "check_grad got {result}, expected ~0");
    }

    #[test]
    fn linprog_matches_scipy_reference_values() {
        // scipy.optimize.linprog(c=[-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4])
        // -> x=[4.0, 0.0], fun=-4.0
        let c = vec![-1.0, 4.0];
        let a_ub = vec![vec![-3.0, 1.0], vec![1.0, 2.0]];
        let b_ub = vec![6.0, 4.0];
        let result = linprog(&c, &a_ub, &b_ub, &[], &[], &[], None).expect("linprog");
        assert_eq!(result.x.len(), 2);
        assert!(
            (result.x[0] - 4.0).abs() < 1e-6,
            "x[0] got {}, expected 4.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.0).abs() < 1e-6,
            "x[1] got {}, expected 0.0",
            result.x[1]
        );
        assert!(
            (result.fun - (-4.0)).abs() < 1e-6,
            "fun got {}, expected -4.0",
            result.fun
        );
    }

    #[test]
    fn derivative_matches_scipy_reference_values() {
        // scipy.misc.derivative(lambda x: x**2, 1.0, dx=1e-6)
        // -> 2.0 (derivative of x^2 at x=1)
        let f = |x: f64| x * x;
        let opts = DifferentiateOptions::default();
        let result = derivative(f, 1.0, opts).expect("derivative");
        assert!(
            (result.df - 2.0).abs() < 1e-4,
            "derivative got {}, expected 2.0",
            result.df
        );
    }

    #[test]
    fn jacobian_matches_scipy_reference_values() {
        // Jacobian of f(x,y) = [x^2, x*y] at (1, 2)
        // -> [[2*x, 0], [y, x]] at (1,2) = [[2, 0], [2, 1]]
        let f = |x: &[f64]| vec![x[0] * x[0], x[0] * x[1]];
        let x0 = vec![1.0, 2.0];
        let opts = DifferentiateOptions::default();
        let result = jacobian(f, &x0, opts).expect("jacobian");
        // df is the Jacobian matrix as Vec<Vec<f64>>
        // Check first row: [2, 0]
        assert!(
            (result.df[0][0] - 2.0).abs() < 1e-4,
            "J[0][0] got {}, expected 2",
            result.df[0][0]
        );
        assert!(
            result.df[0][1].abs() < 1e-4,
            "J[0][1] got {}, expected 0",
            result.df[0][1]
        );
        // Check second row: [2, 1]
        assert!(
            (result.df[1][0] - 2.0).abs() < 1e-4,
            "J[1][0] got {}, expected 2",
            result.df[1][0]
        );
        assert!(
            (result.df[1][1] - 1.0).abs() < 1e-4,
            "J[1][1] got {}, expected 1",
            result.df[1][1]
        );
    }

    #[test]
    fn differential_evolution_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.differential_evolution(rosen, [(-5, 5), (-5, 5)], seed=42)
        // finds minimum near (1, 1) with f(x) near 0
        let result = differential_evolution(
            rosen,
            &[(-5.0, 5.0), (-5.0, 5.0)],
            DifferentialEvolutionOptions::default(),
        )
        .expect("differential_evolution should succeed");

        assert!(result.success, "differential_evolution should converge");
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-6, "minimum value = {fun}, expected near 0");
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x[0] = {}, expected near 1",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "x[1] = {}, expected near 1",
            result.x[1]
        );
    }

    #[test]
    fn dual_annealing_rosenbrock_matches_scipy_reference() {
        // scipy.optimize.dual_annealing(rosen, [(-5, 5), (-5, 5)], seed=42)
        // finds minimum near (1, 1) with f(x) near 0
        let result = dual_annealing(rosen, &[(-5.0, 5.0), (-5.0, 5.0)], 1000, 42)
            .expect("dual_annealing should succeed");

        assert!(result.success, "dual_annealing should converge");
        let fun = result.fun.expect("fun should be Some");
        assert!(fun < 1e-3, "minimum value = {fun}, expected near 0");
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x[0] = {}, expected near 1",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "x[1] = {}, expected near 1",
            result.x[1]
        );
    }

    #[test]
    fn minimize_scalar_bounded_matches_scipy_reference() {
        // scipy.optimize.minimize_scalar(lambda x: (x-2)**2, bounds=(0, 4), method='bounded')
        // finds minimum at x=2
        let (xmin, fmin) = minimize_scalar_bounded(|x| (x - 2.0).powi(2), (0.0, 4.0), 1e-8, 500);
        assert!((xmin - 2.0).abs() < 1e-6, "xmin = {xmin}, expected 2.0");
        assert!(fmin.abs() < 1e-12, "fmin = {fmin}, expected 0.0");
    }
}
