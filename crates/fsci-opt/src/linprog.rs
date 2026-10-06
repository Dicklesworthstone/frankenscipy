//! Linear programming: `scipy.optimize.linprog` with HiGHS's result shape (frankenscipy-1ksfv.5).
//!
//! SciPy's default `linprog(method='highs')` hands the problem to HiGHS and returns `x`, `fun`,
//! `slack`, `con`, `status`, `message`, `nit`, `success` and four `{residual, marginals}` records:
//! `ineqlin`, `eqlin`, `lower` and `upper`. [`linprog`] returns the same shape from two solvers
//! written here (HiGHS is not linked):
//!
//! * a **bounded primal revised simplex** ([`LinprogMethod::HighsDs`], and
//!   [`LinprogMethod::Highs`], which like HiGHS on an LP picks the simplex). It works on
//!   `A x + s = b` with one logical `s_i` per row — `s_i ∈ [0, ∞)` for a `≤` row, `s_i = 0` for an
//!   equality — and the user's bounds on `x` taken as they are: a free variable stays one
//!   variable and a boxed one never becomes an extra row. The basis is a dense LU with partial
//!   pivoting, updated in product form and refactorized every 50 updates.
//!   Phase 1 minimizes the sum of infeasibilities from the slack basis; pricing is Dantzig's,
//!   the ratio test is Harris's two-pass test, and a run of degenerate pivots switches pricing
//!   and the ratio test to Bland's rule until the objective moves again (Beale's cycling
//!   example terminates). A final strict pass without Harris's relaxation removes the
//!   tolerance-sized infeasibilities the relaxation can leave, so the reported vertex and duals
//!   satisfy the optimality conditions to rounding.
//! * a **Mehrotra predictor–corrector interior point method** ([`LinprogMethod::HighsIpm`])
//!   on the same bounds (upper bounds through `x + w = u`, free variables regularized), followed
//!   by a crossover: the interior solution ranks the variables, a basis is built from the most
//!   interior independent columns, and the simplex above finishes from that basis
//!   (`crossover_nit` counts its iterations, as HiGHS's crossover does). If the IPM stalls or
//!   diverges (the signature of an infeasible or unbounded LP) the simplex decides the status.
//!
//! At an optimum the duals come from the final basis: `y = B⁻ᵀ c_B` gives the row marginals
//! (`∂fun/∂b_ub ≤ 0`, `∂fun/∂b_eq`), and each structural reduced cost `d_j = c_j − a_jᵀy` is the
//! `lower` marginal (`≥ 0`) when `x_j` is nonbasic at its lower bound and the `upper` marginal
//! (`≤ 0`) at its upper bound — HiGHS's split, including a fixed variable whose marginal goes to
//! `lower` when `d_j ≥ 0` and to `upper` otherwise. Like SciPy, a result that is not optimal
//! carries no solution: `x`, `slack`, `con` and every `residual`/`marginals` vector are empty
//! and `fun` is NaN where SciPy returns `None`.
//!
//! Differences from HiGHS that can show: there is no presolve and no scaling; HiGHS's
//! `highs-ds` is a *dual* simplex while this one is primal, so on an LP with several optimal
//! vertices (or several optimal duals) the two can return different, equally optimal answers;
//! `nit` counts this solver's iterations, not HiGHS's; and the `primal_status` word inside a
//! failure message reports this solver's state (HiGHS's presolve often reports `None` there).

use crate::types::OptError;

/// Product-form updates applied to the LU before the basis is refactorized from scratch.
const REFACTOR_INTERVAL: usize = 50;

/// Tolerance of the final strict simplex pass (no Harris relaxation): primal bound violation
/// and dual sign violation, relative to `1 + |bound|` and `1 + |c_j|`.
const STRICT_TOL: f64 = 1e-11;

/// IPM iterations allowed when the caller sets no `maxiter`; reaching it hands the problem to
/// the simplex instead of reporting an iteration limit the caller never asked for.
const IPM_DEFAULT_ITERATIONS: usize = 200;

/// Regularization of a free variable's diagonal in the IPM normal equations.
const IPM_FREE_REGULARIZATION: f64 = 1e-8;

/// Fraction of the step to the boundary an IPM iterate takes.
const IPM_STEP_FRACTION: f64 = 0.995;

const MSG_OPTIMAL: &str = "Optimization terminated successfully. (HiGHS Status 7: Optimal)";
const MSG_UNBOUNDED: &str = "The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is Feasible)";
const MSG_MODEL_ERROR: &str = "(HiGHS Status 2: Model error)";
const MSG_SOLVE_ERROR: &str = "(HiGHS Status 4: Solve error)";
/// SciPy's `_check_result` message when a reported optimum violates the constraints by more
/// than `sqrt(tol) * 10` with SciPy's `tol = 1e-9`.
const MSG_CHECK_FAILED: &str = "The solution does not satisfy the constraints within the required tolerance of 3.16E-04, yet no errors were raised and there is no certificate of infeasibility or unboundedness. Check whether the slack and constraint residuals are acceptable; if not, consider enabling presolve, adjusting the tolerance option(s), and/or using a different method. Please consider submitting a bug report.";

fn msg_infeasible(primal_status: &str) -> String {
    format!(
        "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is {primal_status})"
    )
}

fn msg_iteration_limit(feasible: bool) -> String {
    let primal_status = if feasible { "Feasible" } else { "Infeasible" };
    format!(
        "Iteration limit reached. (HiGHS Status 14: model_status is Iteration limit reached; primal_status is {primal_status})"
    )
}

/// `scipy.optimize.linprog`'s `method=` for the HiGHS family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinprogMethod {
    /// `method='highs'`, SciPy's default: HiGHS chooses, and for an LP it chooses the simplex.
    /// fsci runs the bounded revised simplex.
    #[default]
    Highs,
    /// `method='highs-ds'`: the bounded revised simplex (primal here; HiGHS's is dual).
    HighsDs,
    /// `method='highs-ipm'`: Mehrotra's predictor–corrector interior point method, then a
    /// crossover to an optimal basis.
    HighsIpm,
}

/// Solver controls for [`linprog`]; the defaults are SciPy's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinprogOptions {
    pub method: LinprogMethod,
    /// Iteration limit of the simplex (all phases) or of the IPM (crossover not included).
    /// `None` is HiGHS's unlimited default; the simplex then stops at a safeguard of
    /// `100·(m + n) + 10000` iterations, reported as an iteration limit, and an IPM still
    /// unconverged after 200 iterations hands the problem to the simplex.
    pub maxiter: Option<usize>,
    /// Bound and row violation the simplex treats as feasible (SciPy default `1e-7`).
    pub primal_feasibility_tolerance: f64,
    /// Reduced-cost sign violation the simplex treats as optimal (SciPy default `1e-7`).
    pub dual_feasibility_tolerance: f64,
    /// Relative duality gap at which the IPM stops (SciPy default `1e-8`).
    pub ipm_optimality_tolerance: f64,
}

impl Default for LinprogOptions {
    fn default() -> Self {
        Self {
            method: LinprogMethod::Highs,
            maxiter: None,
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            ipm_optimality_tolerance: 1e-8,
        }
    }
}

/// One of SciPy's `ineqlin` / `eqlin` / `lower` / `upper` records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinprogSensitivity {
    /// `b_ub − A_ub x` (ineqlin), `b_eq − A_eq x` (eqlin), `x − lb` (lower) or `ub − x`
    /// (upper); an infinite bound gives an infinite residual, as in SciPy.
    pub residual: Vec<f64>,
    /// Partial derivative of `fun` with respect to `b_ub` (`≤ 0`), `b_eq`, the lower bounds
    /// (`≥ 0`) or the upper bounds (`≤ 0`).
    pub marginals: Vec<f64>,
}

/// Result of [`linprog`], shaped like SciPy's HiGHS result.
///
/// Every solution field is empty (and `fun` is NaN) unless `status == 0` or the solution
/// failed SciPy's post-check (`status == 4` with `x` present): SciPy returns `None` there.
#[derive(Debug, Clone)]
pub struct LinprogResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Optimal objective value `c^T x`; NaN when there is no solution.
    pub fun: f64,
    /// `b_ub − A_ub x`.
    pub slack: Vec<f64>,
    /// `b_eq − A_eq x`.
    pub con: Vec<f64>,
    /// `status == 0`.
    pub success: bool,
    /// 0 optimal, 1 iteration limit, 2 infeasible (or a model error), 3 unbounded,
    /// 4 numerical difficulties.
    pub status: u8,
    /// SciPy's message for the status, HiGHS suffix included.
    pub message: String,
    /// Simplex iterations (all phases), or IPM iterations for [`LinprogMethod::HighsIpm`].
    pub nit: usize,
    /// Simplex iterations after the IPM's crossover basis; 0 for the simplex methods.
    pub crossover_nit: usize,
    pub ineqlin: LinprogSensitivity,
    pub eqlin: LinprogSensitivity,
    pub lower: LinprogSensitivity,
    pub upper: LinprogSensitivity,
}

impl LinprogResult {
    fn without_solution(status: u8, message: String, nit: usize, crossover_nit: usize) -> Self {
        Self {
            x: Vec::new(),
            fun: f64::NAN,
            slack: Vec::new(),
            con: Vec::new(),
            success: false,
            status,
            message,
            nit,
            crossover_nit,
            ineqlin: LinprogSensitivity::default(),
            eqlin: LinprogSensitivity::default(),
            lower: LinprogSensitivity::default(),
            upper: LinprogSensitivity::default(),
        }
    }
}

/// Solve a linear program, as `scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
/// method=..., options=...)` with a HiGHS method.
///
/// Minimizes `c^T x` subject to `A_ub x ≤ b_ub`, `A_eq x = b_eq` and `lb ≤ x ≤ ub`.
///
/// * `a_ub`, `a_eq` — dense rows of length `c.len()`; either may be empty.
/// * `bounds` — `(lower, upper)` per variable, `None` meaning unbounded on that side; an empty
///   slice means `(0, None)` for every variable, SciPy's default.
///
/// See the [module documentation](self) for the algorithms. A lower bound of `+inf`, an upper
/// bound of `-inf` or a lower bound above the upper one is a result with `status == 2`, as in
/// SciPy (HiGHS's "Model error" for the first two).
///
/// # Errors
/// [`OptError::InvalidArgument`] for an empty `c`, mismatched dimensions or a non-positive
/// tolerance; [`OptError::NonFiniteInput`] for a non-finite entry in `c`, `A_ub`, `b_ub`,
/// `A_eq` or `b_eq` (SciPy raises `ValueError` for all of these); [`OptError::InvalidBounds`]
/// for a NaN bound (SciPy silently reads NaN as `None`; `None` is the explicit spelling here).
pub fn linprog(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    bounds: &[(Option<f64>, Option<f64>)],
    options: LinprogOptions,
) -> Result<LinprogResult, OptError> {
    validate_problem(c, a_ub, b_ub, a_eq, b_eq)?;
    validate_options(&options)?;
    let n = c.len();
    let (lb, ub) = parse_bounds(n, bounds)?;
    if lb.contains(&f64::INFINITY) || ub.contains(&f64::NEG_INFINITY) {
        return Ok(LinprogResult::without_solution(
            2,
            MSG_MODEL_ERROR.to_string(),
            0,
            0,
        ));
    }
    if lb.iter().zip(&ub).any(|(l, u)| l > u) {
        return Ok(LinprogResult::without_solution(
            2,
            msg_infeasible("None"),
            0,
            0,
        ));
    }

    let lp = Lp::new(c, a_ub, b_ub, a_eq, b_eq, &lb, &ub);
    let tolerances = Tolerances {
        primal: options.primal_feasibility_tolerance,
        dual: options.dual_feasibility_tolerance,
    };
    let solved = match options.method {
        LinprogMethod::Highs | LinprogMethod::HighsDs => {
            let limit = options.maxiter.unwrap_or_else(|| simplex_safeguard(&lp));
            let run = simplex_from(Simplex::from_slack_basis(&lp, tolerances), 0, limit);
            Solved {
                end: run.end,
                nit: run.crossover_nit,
                crossover_nit: 0,
            }
        }
        LinprogMethod::HighsIpm => solve_ipm(
            &lp,
            tolerances,
            options.ipm_optimality_tolerance,
            options.maxiter,
        ),
    };
    Ok(assemble(&lp, c, a_ub, b_ub, a_eq, b_eq, &lb, &ub, solved))
}

fn validate_problem(
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
) -> Result<(), OptError> {
    let n = c.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "c must not be empty".to_string(),
        });
    }
    if c.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "c must contain only finite values".to_string(),
        });
    }
    for (name, rows, rhs) in [("A_ub", a_ub, b_ub), ("A_eq", a_eq, b_eq)] {
        let b_name = if name == "A_ub" { "b_ub" } else { "b_eq" };
        if rows.len() != rhs.len() {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "{name} rows ({}) must match {b_name} length ({})",
                    rows.len(),
                    rhs.len()
                ),
            });
        }
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(OptError::InvalidArgument {
                    detail: format!("{name} row {i} has {} cols, expected {n}", row.len()),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(OptError::NonFiniteInput {
                    detail: format!("{name} row {i} contains non-finite values"),
                });
            }
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(OptError::NonFiniteInput {
                detail: format!("{b_name} must contain only finite values"),
            });
        }
    }
    Ok(())
}

fn validate_options(options: &LinprogOptions) -> Result<(), OptError> {
    for (name, value) in [
        (
            "primal_feasibility_tolerance",
            options.primal_feasibility_tolerance,
        ),
        (
            "dual_feasibility_tolerance",
            options.dual_feasibility_tolerance,
        ),
        ("ipm_optimality_tolerance", options.ipm_optimality_tolerance),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(OptError::InvalidArgument {
                detail: format!("{name} must be finite and positive (got {value})"),
            });
        }
    }
    Ok(())
}

fn parse_bounds(
    n: usize,
    bounds: &[(Option<f64>, Option<f64>)],
) -> Result<(Vec<f64>, Vec<f64>), OptError> {
    if bounds.is_empty() {
        return Ok((vec![0.0; n], vec![f64::INFINITY; n]));
    }
    if bounds.len() != n {
        return Err(OptError::InvalidArgument {
            detail: format!("bounds length ({}) must match c length ({n})", bounds.len()),
        });
    }
    let mut lb = Vec::with_capacity(n);
    let mut ub = Vec::with_capacity(n);
    for (index, &(lo, hi)) in bounds.iter().enumerate() {
        let lo = lo.unwrap_or(f64::NEG_INFINITY);
        let hi = hi.unwrap_or(f64::INFINITY);
        if lo.is_nan() || hi.is_nan() {
            return Err(OptError::InvalidBounds {
                detail: format!(
                    "variable {index}: bounds must not be NaN (got {lo}, {hi}); use None for no bound"
                ),
            });
        }
        lb.push(lo);
        ub.push(hi);
    }
    Ok((lb, ub))
}

fn simplex_safeguard(lp: &Lp) -> usize {
    100 * (lp.n + lp.m) + 10_000
}

#[derive(Debug, Clone, Copy)]
struct Tolerances {
    primal: f64,
    dual: f64,
}

/// The LP in the simplex's form: `A x + s = b`, columns `0..n` structural and `n..n+m` the
/// logicals (`s_i ∈ [0, ∞)` for the first `m_ub` rows, `s_i = 0` for the equality rows).
struct Lp {
    n: usize,
    m_ub: usize,
    m: usize,
    /// Column-major `m × n`.
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl Lp {
    fn new(
        c: &[f64],
        a_ub: &[Vec<f64>],
        b_ub: &[f64],
        a_eq: &[Vec<f64>],
        b_eq: &[f64],
        lb: &[f64],
        ub: &[f64],
    ) -> Self {
        let n = c.len();
        let m_ub = a_ub.len();
        let m = m_ub + a_eq.len();
        let mut a = vec![0.0; m * n];
        for (i, row) in a_ub.iter().chain(a_eq).enumerate() {
            for (j, &value) in row.iter().enumerate() {
                a[j * m + i] = value;
            }
        }
        let mut lo = lb.to_vec();
        let mut hi = ub.to_vec();
        for i in 0..m {
            lo.push(0.0);
            hi.push(if i < m_ub { f64::INFINITY } else { 0.0 });
        }
        Self {
            n,
            m_ub,
            m,
            a,
            b: b_ub.iter().chain(b_eq).copied().collect(),
            c: c.to_vec(),
            lo,
            hi,
        }
    }

    fn ncols(&self) -> usize {
        self.n + self.m
    }

    fn cost(&self, j: usize) -> f64 {
        if j < self.n { self.c[j] } else { 0.0 }
    }

    fn column(&self, j: usize) -> &[f64] {
        &self.a[j * self.m..(j + 1) * self.m]
    }

    fn dot_col(&self, j: usize, v: &[f64]) -> f64 {
        if j < self.n {
            self.column(j).iter().zip(v).map(|(a, b)| a * b).sum()
        } else {
            v[j - self.n]
        }
    }

    /// `out += scale * column(j)`.
    fn axpy_col(&self, j: usize, scale: f64, out: &mut [f64]) {
        if j < self.n {
            for (o, &a) in out.iter_mut().zip(self.column(j)) {
                *o += scale * a;
            }
        } else {
            out[j - self.n] += scale;
        }
    }

    fn is_fixed(&self, j: usize) -> bool {
        self.lo[j] == self.hi[j]
    }
}

/// Dense LU with partial pivoting, `P B = L U` (row-major, `L` unit lower).
struct DenseLu {
    m: usize,
    lu: Vec<f64>,
    perm: Vec<usize>,
}

impl DenseLu {
    /// `None` when a pivot is negligible against the largest entry: the matrix is singular to
    /// working precision.
    fn factor(mut lu: Vec<f64>, m: usize) -> Option<Self> {
        let scale = lu.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())).max(1.0);
        let mut perm: Vec<usize> = (0..m).collect();
        for k in 0..m {
            let mut p = k;
            let mut best = lu[k * m + k].abs();
            for i in k + 1..m {
                let v = lu[i * m + k].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if !(best > 1e-12 * scale) {
                return None;
            }
            if p != k {
                for j in 0..m {
                    lu.swap(k * m + j, p * m + j);
                }
                perm.swap(k, p);
            }
            let pivot = lu[k * m + k];
            for i in k + 1..m {
                let factor = lu[i * m + k] / pivot;
                lu[i * m + k] = factor;
                if factor != 0.0 {
                    for j in k + 1..m {
                        lu[i * m + j] -= factor * lu[k * m + j];
                    }
                }
            }
        }
        Some(Self { m, lu, perm })
    }

    /// Overwrites `r` with `B⁻¹ r`.
    fn solve(&self, r: &mut [f64]) {
        let m = self.m;
        let mut z: Vec<f64> = self.perm.iter().map(|&p| r[p]).collect();
        for i in 0..m {
            let mut s = z[i];
            for j in 0..i {
                s -= self.lu[i * m + j] * z[j];
            }
            z[i] = s;
        }
        for i in (0..m).rev() {
            let mut s = z[i];
            for j in i + 1..m {
                s -= self.lu[i * m + j] * z[j];
            }
            z[i] = s / self.lu[i * m + i];
        }
        r.copy_from_slice(&z);
    }

    /// Overwrites `r` with `B⁻ᵀ r` (`Bᵀ = Uᵀ Lᵀ P`).
    fn solve_transpose(&self, r: &mut [f64]) {
        let m = self.m;
        let mut w = r.to_vec();
        for i in 0..m {
            let mut s = w[i];
            for j in 0..i {
                s -= self.lu[j * m + i] * w[j];
            }
            w[i] = s / self.lu[i * m + i];
        }
        for i in (0..m).rev() {
            let mut s = w[i];
            for j in i + 1..m {
                s -= self.lu[j * m + i] * w[j];
            }
            w[i] = s;
        }
        for (k, &p) in self.perm.iter().enumerate() {
            r[p] = w[k];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarState {
    Basic,
    AtLower,
    AtUpper,
    /// A free nonbasic variable, held at zero.
    Zero,
}

/// How a simplex run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit { feasible: bool },
    Breakdown,
}

/// How the ratio test picks the leaving variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeavingRule {
    /// Harris's two-pass test: bounds relaxed by the primal tolerance, then the largest pivot
    /// among the variables blocking within the relaxed step.
    Harris,
    /// The exact minimum ratio; ties to the largest pivot.
    LargestPivot,
    /// The exact minimum ratio; ties to the smallest variable index (Bland's leaving rule).
    LowestIndex,
}

enum Step {
    Flip { theta: f64 },
    Pivot { row: usize, theta: f64, bound: f64 },
    Unbounded,
}

/// The final state of a solve.
enum End {
    Optimal(Box<Vertex>),
    Infeasible,
    Unbounded,
    IterationLimit { feasible: bool },
    SolveError,
}

/// An optimal basic solution: structural values, row duals and structural reduced costs.
struct Vertex {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
    state: Vec<VarState>,
}

struct Solved {
    end: End,
    nit: usize,
    crossover_nit: usize,
}

/// Bounded primal revised simplex on an [`Lp`].
struct Simplex<'a> {
    lp: &'a Lp,
    basis: Vec<usize>,
    state: Vec<VarState>,
    x: Vec<f64>,
    lu: DenseLu,
    etas: Vec<(usize, Vec<f64>)>,
    tol_p: f64,
    tol_d: f64,
    /// Leaving rule outside Bland's mode (which always uses [`LeavingRule::LowestIndex`]).
    leaving: LeavingRule,
    /// Consecutive degenerate iterations after which Bland's rule takes over.
    bland_after: usize,
    nit: usize,
}

impl<'a> Simplex<'a> {
    /// The all-logical basis. A nonbasic structural starts at its lower bound when it has
    /// one — at its upper bound instead when boxed with a negative cost, HiGHS's choice —
    /// at its upper bound when it has only that, and at zero when free.
    fn from_slack_basis(lp: &'a Lp, tolerances: Tolerances) -> Option<Self> {
        let mut state = Vec::with_capacity(lp.ncols());
        let mut x = vec![0.0; lp.ncols()];
        for j in 0..lp.n {
            let (lo, hi) = (lp.lo[j], lp.hi[j]);
            let s = if lo.is_finite() && hi.is_finite() {
                if lp.c[j] < 0.0 && lo != hi {
                    VarState::AtUpper
                } else {
                    VarState::AtLower
                }
            } else if lo.is_finite() {
                VarState::AtLower
            } else if hi.is_finite() {
                VarState::AtUpper
            } else {
                VarState::Zero
            };
            x[j] = match s {
                VarState::AtLower => lo,
                VarState::AtUpper => hi,
                VarState::Zero | VarState::Basic => 0.0,
            };
            state.push(s);
        }
        state.extend(std::iter::repeat_n(VarState::Basic, lp.m));
        let basis: Vec<usize> = (lp.n..lp.n + lp.m).collect();
        Self::with_basis(lp, tolerances, basis, state, x)
    }

    /// A simplex at `basis`; nonbasic values are taken from `x` and must sit at the bound
    /// their `state` names (or at zero for [`VarState::Zero`]).
    fn with_basis(
        lp: &'a Lp,
        tolerances: Tolerances,
        basis: Vec<usize>,
        state: Vec<VarState>,
        x: Vec<f64>,
    ) -> Option<Self> {
        let lu = DenseLu::factor(Self::basis_matrix(lp, &basis), lp.m)?;
        let mut simplex = Self {
            lp,
            basis,
            state,
            x,
            lu,
            etas: Vec::new(),
            tol_p: tolerances.primal,
            tol_d: tolerances.dual,
            leaving: LeavingRule::Harris,
            bland_after: lp.m + 16,
            nit: 0,
        };
        simplex.recompute_primal();
        Some(simplex)
    }

    fn basis_matrix(lp: &Lp, basis: &[usize]) -> Vec<f64> {
        let m = lp.m;
        let mut mat = vec![0.0; m * m];
        let mut col = vec![0.0; m];
        for (k, &j) in basis.iter().enumerate() {
            col.iter_mut().for_each(|v| *v = 0.0);
            lp.axpy_col(j, 1.0, &mut col);
            for i in 0..m {
                mat[i * m + k] = col[i];
            }
        }
        mat
    }

    fn refactor(&mut self) -> bool {
        match DenseLu::factor(Self::basis_matrix(self.lp, &self.basis), self.lp.m) {
            Some(lu) => {
                self.lu = lu;
                self.etas.clear();
                self.recompute_primal();
                true
            }
            None => false,
        }
    }

    fn ftran(&self, v: &mut [f64]) {
        self.lu.solve(v);
        for (r, alpha) in &self.etas {
            let vr = v[*r] / alpha[*r];
            for (i, (vi, &ai)) in v.iter_mut().zip(alpha).enumerate() {
                if i != *r {
                    *vi -= ai * vr;
                }
            }
            v[*r] = vr;
        }
    }

    fn btran(&self, v: &mut [f64]) {
        for (r, alpha) in self.etas.iter().rev() {
            let mut s = 0.0;
            for (i, (&vi, &ai)) in v.iter().zip(alpha).enumerate() {
                if i != *r {
                    s += ai * vi;
                }
            }
            v[*r] = (v[*r] - s) / alpha[*r];
        }
        self.lu.solve_transpose(v);
    }

    /// `x_B = B⁻¹ (b − N x_N)`.
    fn recompute_primal(&mut self) {
        let mut rhs = self.lp.b.clone();
        for j in 0..self.lp.ncols() {
            if self.state[j] != VarState::Basic && self.x[j] != 0.0 {
                self.lp.axpy_col(j, -self.x[j], &mut rhs);
            }
        }
        self.ftran(&mut rhs);
        for (k, &j) in self.basis.iter().enumerate() {
            self.x[j] = rhs[k];
        }
    }

    /// Basic costs of the current phase: the infeasibility gradient (−1 below a lower bound,
    /// +1 above an upper bound) while any basic variable is infeasible, else `c_B`.
    fn phase_costs(&self) -> (Vec<f64>, bool) {
        let mut cb = vec![0.0; self.lp.m];
        let mut infeasible = false;
        for (k, &j) in self.basis.iter().enumerate() {
            let xj = self.x[j];
            if xj < self.lp.lo[j] - self.tol_p {
                cb[k] = -1.0;
                infeasible = true;
            } else if xj > self.lp.hi[j] + self.tol_p {
                cb[k] = 1.0;
                infeasible = true;
            }
        }
        if !infeasible {
            for (k, &j) in self.basis.iter().enumerate() {
                cb[k] = self.lp.cost(j);
            }
        }
        (cb, infeasible)
    }

    /// Entering variable and its reduced cost: Dantzig's largest `|d_j|`, or under Bland's
    /// rule the smallest eligible index. Fixed variables never enter.
    fn price(&self, y: &[f64], phase1: bool, bland: bool) -> Option<(usize, f64)> {
        let mut best = None;
        let mut best_abs = 0.0;
        for j in 0..self.lp.ncols() {
            let st = self.state[j];
            if st == VarState::Basic || self.lp.is_fixed(j) {
                continue;
            }
            let cost = if phase1 { 0.0 } else { self.lp.cost(j) };
            let d = cost - self.lp.dot_col(j, y);
            let eligible = match st {
                VarState::AtLower => d < -self.tol_d,
                VarState::AtUpper => d > self.tol_d,
                VarState::Zero => d.abs() > self.tol_d,
                VarState::Basic => false,
            };
            if !eligible {
                continue;
            }
            if bland {
                return Some((j, d));
            }
            if d.abs() > best_abs {
                best_abs = d.abs();
                best = Some((j, d));
            }
        }
        best
    }

    /// The bound basic position `k` runs into while moving at `rate` per unit step, or `None`
    /// when nothing stops it. An infeasible variable moving toward its violated bound stops
    /// there (the first breakpoint of the phase-1 objective); one moving away never stops.
    fn blocking_bound(&self, k: usize, rate: f64) -> Option<f64> {
        let j = self.basis[k];
        let (xj, lo, hi) = (self.x[j], self.lp.lo[j], self.lp.hi[j]);
        if rate > 0.0 {
            if xj < lo - self.tol_p {
                Some(lo)
            } else if xj > hi + self.tol_p || hi == f64::INFINITY {
                None
            } else {
                Some(hi)
            }
        } else if xj > hi + self.tol_p {
            Some(hi)
        } else if xj < lo - self.tol_p || lo == f64::NEG_INFINITY {
            None
        } else {
            Some(lo)
        }
    }

    fn ratio_test(&self, q: usize, dir: f64, alpha: &[f64], bland: bool) -> Step {
        let flip = if self.state[q] == VarState::Zero {
            f64::INFINITY
        } else {
            self.lp.hi[q] - self.lp.lo[q]
        };
        let amax = alpha.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let pivot_tol = 1e-9 * amax.max(1.0);
        let candidates = || {
            alpha.iter().enumerate().filter_map(move |(k, &a)| {
                if a.abs() <= pivot_tol {
                    return None;
                }
                let rate = -dir * a;
                self.blocking_bound(k, rate).map(|bound| (k, rate, bound))
            })
        };
        let rule = if bland {
            LeavingRule::LowestIndex
        } else {
            self.leaving
        };
        if rule == LeavingRule::Harris {
            // Pass 1: the largest step that keeps every blocking variable within its bound
            // relaxed by the primal tolerance.
            let mut theta_max = flip;
            for (k, rate, bound) in candidates() {
                let relaxed = bound + rate.signum() * self.tol_p;
                let t = (relaxed - self.x[self.basis[k]]) / rate;
                theta_max = theta_max.min(t);
            }
            if theta_max == f64::INFINITY {
                return Step::Unbounded;
            }
            if flip <= theta_max {
                return Step::Flip { theta: flip };
            }
            // Pass 2: among the variables blocking within that step, the largest pivot.
            let mut chosen: Option<(usize, f64, f64)> = None;
            let mut best_abs = 0.0;
            for (k, rate, bound) in candidates() {
                let t = (bound - self.x[self.basis[k]]) / rate;
                if t <= theta_max && alpha[k].abs() > best_abs {
                    best_abs = alpha[k].abs();
                    chosen = Some((k, t, bound));
                }
            }
            match chosen {
                Some((row, t, bound)) => Step::Pivot {
                    row,
                    theta: t.max(0.0),
                    bound,
                },
                None => Step::Unbounded,
            }
        } else {
            let mut theta_min = flip;
            for (k, rate, bound) in candidates() {
                let t = ((bound - self.x[self.basis[k]]) / rate).max(0.0);
                theta_min = theta_min.min(t);
            }
            if theta_min == f64::INFINITY {
                return Step::Unbounded;
            }
            if flip <= theta_min {
                return Step::Flip { theta: flip };
            }
            let tie = theta_min + 1e-12 * (1.0 + theta_min);
            let mut chosen: Option<(usize, f64, f64)> = None;
            for (k, rate, bound) in candidates() {
                let t = ((bound - self.x[self.basis[k]]) / rate).max(0.0);
                if t > tie {
                    continue;
                }
                let better = match chosen {
                    None => true,
                    Some((prev, _, _)) => {
                        if rule == LeavingRule::LowestIndex {
                            self.basis[k] < self.basis[prev]
                        } else {
                            alpha[k].abs() > alpha[prev].abs()
                        }
                    }
                };
                if better {
                    chosen = Some((k, t, bound));
                }
            }
            match chosen {
                Some((row, t, bound)) => Step::Pivot {
                    row,
                    theta: t,
                    bound,
                },
                None => Step::Unbounded,
            }
        }
    }

    /// Iterates until optimal, infeasible, unbounded or `limit` total iterations. A verdict is
    /// only returned on a freshly refactorized basis, so drift in the product-form updates
    /// cannot decide it.
    fn run(&mut self, limit: usize) -> Run {
        let m = self.lp.m;
        let mut degenerate_run = 0usize;
        loop {
            if self.etas.len() >= REFACTOR_INTERVAL && !self.refactor() {
                return Run::Breakdown;
            }
            let (mut y, phase1) = self.phase_costs();
            self.btran(&mut y);
            let bland = degenerate_run >= self.bland_after;
            let Some((q, dq)) = self.price(&y, phase1, bland) else {
                if !self.etas.is_empty() {
                    if !self.refactor() {
                        return Run::Breakdown;
                    }
                    continue;
                }
                return if phase1 {
                    Run::Infeasible
                } else {
                    Run::Optimal
                };
            };
            if self.nit >= limit {
                return Run::IterationLimit { feasible: !phase1 };
            }
            let dir = if dq < 0.0 { 1.0 } else { -1.0 };
            let mut alpha = vec![0.0; m];
            self.lp.axpy_col(q, 1.0, &mut alpha);
            self.ftran(&mut alpha);
            let theta = match self.ratio_test(q, dir, &alpha, bland) {
                Step::Unbounded => {
                    if !self.etas.is_empty() {
                        if !self.refactor() {
                            return Run::Breakdown;
                        }
                        continue;
                    }
                    // Phase 1 is bounded below by zero: no blocking variable there is a
                    // numerical failure, not an unbounded LP.
                    return if phase1 {
                        Run::Breakdown
                    } else {
                        Run::Unbounded
                    };
                }
                Step::Flip { theta } => {
                    self.move_basics(theta, dir, &alpha);
                    if self.state[q] == VarState::AtLower {
                        self.state[q] = VarState::AtUpper;
                        self.x[q] = self.lp.hi[q];
                    } else {
                        self.state[q] = VarState::AtLower;
                        self.x[q] = self.lp.lo[q];
                    }
                    theta
                }
                Step::Pivot { row, theta, bound } => {
                    self.move_basics(theta, dir, &alpha);
                    let p = self.basis[row];
                    self.x[q] += dir * theta;
                    self.x[p] = bound;
                    self.state[p] = if bound == self.lp.lo[p] {
                        VarState::AtLower
                    } else {
                        VarState::AtUpper
                    };
                    self.basis[row] = q;
                    self.state[q] = VarState::Basic;
                    self.etas.push((row, alpha));
                    theta
                }
            };
            self.nit += 1;
            if theta <= 1e-12 {
                degenerate_run += 1;
            } else {
                degenerate_run = 0;
            }
        }
    }

    fn move_basics(&mut self, theta: f64, dir: f64, alpha: &[f64]) {
        if theta == 0.0 {
            return;
        }
        for (k, &a) in alpha.iter().enumerate() {
            let j = self.basis[k];
            self.x[j] -= theta * dir * a;
        }
    }

    /// Row duals `y = B⁻ᵀ c_B` and the reduced cost of every column.
    fn duals(&self) -> (Vec<f64>, Vec<f64>) {
        let mut y: Vec<f64> = self.basis.iter().map(|&j| self.lp.cost(j)).collect();
        self.btran(&mut y);
        let d = (0..self.lp.ncols())
            .map(|j| {
                if self.state[j] == VarState::Basic {
                    0.0
                } else {
                    self.lp.cost(j) - self.lp.dot_col(j, &y)
                }
            })
            .collect();
        (y, d)
    }

    /// Whether the optimal basis is feasible and dual feasible only up to the run's
    /// tolerances rather than to [`STRICT_TOL`].
    fn needs_cleanup(&self) -> bool {
        let lp = self.lp;
        for &j in &self.basis {
            let xj = self.x[j];
            if (lp.lo[j] - xj) > STRICT_TOL * (1.0 + lp.lo[j].abs())
                || (xj - lp.hi[j]) > STRICT_TOL * (1.0 + lp.hi[j].abs())
            {
                return true;
            }
        }
        let (_, d) = self.duals();
        (0..lp.ncols()).any(|j| {
            if lp.is_fixed(j) {
                return false;
            }
            let scale = STRICT_TOL * (1.0 + lp.cost(j).abs());
            match self.state[j] {
                VarState::Basic => false,
                VarState::AtLower => d[j] < -scale,
                VarState::AtUpper => d[j] > scale,
                VarState::Zero => d[j].abs() > scale,
            }
        })
    }

    /// Runs to optimality, then (when needed) a strict pass with no Harris relaxation so the
    /// returned vertex satisfies the optimality conditions to rounding. A strict pass that
    /// fails to finish leaves the tolerance-level optimum in place.
    fn solve(&mut self, limit: usize) -> End {
        match self.run(limit) {
            Run::Optimal => {
                if self.needs_cleanup() {
                    let saved = (
                        self.basis.clone(),
                        self.state.clone(),
                        self.x.clone(),
                        self.tol_p,
                        self.tol_d,
                    );
                    self.leaving = LeavingRule::LargestPivot;
                    self.tol_p = STRICT_TOL;
                    self.tol_d = STRICT_TOL;
                    if self.run(limit) != Run::Optimal {
                        // The strict pass's iterations stay counted in `nit`.
                        (self.basis, self.state, self.x, self.tol_p, self.tol_d) = saved;
                        if !self.refactor() {
                            return End::SolveError;
                        }
                    }
                }
                let (y, d) = self.duals();
                End::Optimal(Box::new(Vertex {
                    x: self.x[..self.lp.n].to_vec(),
                    y,
                    d: d[..self.lp.n].to_vec(),
                    state: self.state[..self.lp.n].to_vec(),
                }))
            }
            Run::Infeasible => End::Infeasible,
            Run::Unbounded => End::Unbounded,
            Run::IterationLimit { feasible } => End::IterationLimit { feasible },
            Run::Breakdown => End::SolveError,
        }
    }
}

/// Builds SciPy's result from a solve.
fn assemble(
    lp: &Lp,
    c: &[f64],
    a_ub: &[Vec<f64>],
    b_ub: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    lb: &[f64],
    ub: &[f64],
    solved: Solved,
) -> LinprogResult {
    let Solved {
        end,
        nit,
        crossover_nit,
    } = solved;
    let vertex = match end {
        End::Optimal(vertex) => vertex,
        End::Infeasible => {
            return LinprogResult::without_solution(
                2,
                msg_infeasible("Infeasible"),
                nit,
                crossover_nit,
            );
        }
        End::Unbounded => {
            return LinprogResult::without_solution(
                3,
                MSG_UNBOUNDED.to_string(),
                nit,
                crossover_nit,
            );
        }
        End::IterationLimit { feasible } => {
            return LinprogResult::without_solution(
                1,
                msg_iteration_limit(feasible),
                nit,
                crossover_nit,
            );
        }
        End::SolveError => {
            return LinprogResult::without_solution(
                4,
                MSG_SOLVE_ERROR.to_string(),
                nit,
                crossover_nit,
            );
        }
    };
    let Vertex { x, y, d, state } = *vertex;
    let residual = |rows: &[Vec<f64>], rhs: &[f64]| -> Vec<f64> {
        rows.iter()
            .zip(rhs)
            .map(|(row, &bi)| bi - row.iter().zip(&x).map(|(a, xj)| a * xj).sum::<f64>())
            .collect()
    };
    let slack = residual(a_ub, b_ub);
    let con = residual(a_eq, b_eq);
    let n = x.len();
    let mut lower_marginals = vec![0.0; n];
    let mut upper_marginals = vec![0.0; n];
    for j in 0..n {
        let to_lower = match state[j] {
            VarState::Basic | VarState::Zero => continue,
            _ if lp.is_fixed(j) => d[j] >= 0.0,
            VarState::AtLower => true,
            VarState::AtUpper => false,
        };
        if to_lower {
            lower_marginals[j] = d[j];
        } else {
            upper_marginals[j] = d[j];
        }
    }
    let fun = c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum::<f64>();
    let valid = check_result(&x, fun, &slack, &con, lb, ub);
    let (status, message) = if valid {
        (0, MSG_OPTIMAL.to_string())
    } else {
        (4, MSG_CHECK_FAILED.to_string())
    };
    LinprogResult {
        lower: LinprogSensitivity {
            residual: x.iter().zip(lb).map(|(xj, l)| xj - l).collect(),
            marginals: lower_marginals,
        },
        upper: LinprogSensitivity {
            residual: x.iter().zip(ub).map(|(xj, u)| u - xj).collect(),
            marginals: upper_marginals,
        },
        ineqlin: LinprogSensitivity {
            residual: slack.clone(),
            marginals: y[..lp.m_ub].to_vec(),
        },
        eqlin: LinprogSensitivity {
            residual: con.clone(),
            marginals: y[lp.m_ub..].to_vec(),
        },
        x,
        fun,
        slack,
        con,
        success: status == 0,
        status,
        message,
        nit,
        crossover_nit,
    }
}

/// SciPy's `_check_result` on an optimum: bounds, slack and equality residuals within
/// `sqrt(1e-9) * 10`, and nothing NaN.
fn check_result(x: &[f64], fun: f64, slack: &[f64], con: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    let tol = 1e-9_f64.sqrt() * 10.0;
    if x.iter().chain(slack).chain(con).any(|v| v.is_nan()) || fun.is_nan() {
        return false;
    }
    let bounds_ok = x
        .iter()
        .zip(lb.iter().zip(ub))
        .all(|(&xj, (&l, &u))| xj >= l - tol && xj <= u + tol);
    bounds_ok && slack.iter().all(|&s| s >= -tol) && con.iter().all(|&r| r.abs() <= tol)
}

// ══════════════════════════════════════════════════════════════════════
// Interior point: Mehrotra predictor–corrector, then crossover
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IpmKind {
    /// `x' ≥ 0`.
    Lower,
    /// `0 ≤ x' ≤ u'`, the upper bound through `x' + w = u'` with `w ≥ 0`.
    Boxed,
    Free,
}

/// One IPM variable: simplex column `j` with value `shift + sign · x'`.
#[derive(Debug, Clone, Copy)]
struct IpmVar {
    j: usize,
    kind: IpmKind,
    sign: f64,
    shift: f64,
    upper: f64,
}

/// The IPM's last iterate, in its own coordinates.
struct IpmPoint {
    x: Vec<f64>,
    z: Vec<f64>,
    v: Vec<f64>,
}

enum IpmEnd {
    Converged,
    Limit {
        feasible: bool,
    },
    /// Stalled, diverged or produced a non-finite iterate.
    Failed,
}

/// Dense Cholesky of a symmetric positive semidefinite matrix (row-major, overwritten by
/// `L`). A pivot below `1e-13 ×` the largest diagonal marks a direction the rows do not
/// determine (dependent constraints); it becomes `1e64`, so solves set that component to zero.
fn cholesky(mat: &mut [f64], m: usize) {
    let max_diag = (0..m).map(|i| mat[i * m + i]).fold(0.0_f64, f64::max);
    let tiny = 1e-13 * max_diag;
    for k in 0..m {
        let mut d = mat[k * m + k];
        for p in 0..k {
            d -= mat[k * m + p] * mat[k * m + p];
        }
        let lkk = if d > tiny && d.is_finite() {
            d.sqrt()
        } else {
            1e64
        };
        mat[k * m + k] = lkk;
        for i in k + 1..m {
            let mut s = mat[i * m + k];
            for p in 0..k {
                s -= mat[i * m + p] * mat[k * m + p];
            }
            mat[i * m + k] = s / lkk;
        }
    }
}

fn cholesky_solve(l: &[f64], m: usize, rhs: &mut [f64]) {
    for i in 0..m {
        let mut s = rhs[i];
        for p in 0..i {
            s -= l[i * m + p] * rhs[p];
        }
        rhs[i] = s / l[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = rhs[i];
        for p in i + 1..m {
            s -= l[p * m + i] * rhs[p];
        }
        rhs[i] = s / l[i * m + i];
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The IPM problem `min c'ᵀx' : A'x' = b'` over the non-fixed structurals and the `≤`-row
/// logicals, with the bounds of [`IpmKind`].
struct IpmProblem {
    vars: Vec<IpmVar>,
    m: usize,
    /// Column-major `m × vars.len()`.
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl IpmProblem {
    fn new(lp: &Lp) -> Self {
        let m = lp.m;
        let mut b = lp.b.clone();
        let mut vars = Vec::new();
        for j in 0..lp.n + lp.m_ub {
            let (lo, hi) = (lp.lo[j], lp.hi[j]);
            if lo == hi {
                lp.axpy_col(j, -lo, &mut b);
                continue;
            }
            let var = match (lo.is_finite(), hi.is_finite()) {
                (true, true) => IpmVar {
                    j,
                    kind: IpmKind::Boxed,
                    sign: 1.0,
                    shift: lo,
                    upper: hi - lo,
                },
                (true, false) => IpmVar {
                    j,
                    kind: IpmKind::Lower,
                    sign: 1.0,
                    shift: lo,
                    upper: f64::INFINITY,
                },
                (false, true) => IpmVar {
                    j,
                    kind: IpmKind::Lower,
                    sign: -1.0,
                    shift: hi,
                    upper: f64::INFINITY,
                },
                (false, false) => IpmVar {
                    j,
                    kind: IpmKind::Free,
                    sign: 1.0,
                    shift: 0.0,
                    upper: f64::INFINITY,
                },
            };
            if var.shift != 0.0 {
                lp.axpy_col(j, -var.shift, &mut b);
            }
            vars.push(var);
        }
        let mut a = vec![0.0; m * vars.len()];
        let mut c = Vec::with_capacity(vars.len());
        for (k, var) in vars.iter().enumerate() {
            lp.axpy_col(var.j, var.sign, &mut a[k * m..(k + 1) * m]);
            c.push(var.sign * lp.cost(var.j));
        }
        Self { vars, m, a, b, c }
    }

    fn column(&self, k: usize) -> &[f64] {
        &self.a[k * self.m..(k + 1) * self.m]
    }

    /// Cholesky factor of `A diag(theta) Aᵀ`.
    fn normal_factor(&self, theta: &[f64]) -> Vec<f64> {
        let m = self.m;
        let mut mat = vec![0.0; m * m];
        for (k, &t) in theta.iter().enumerate() {
            let col = self.column(k);
            for i in 0..m {
                let ci = col[i] * t;
                if ci == 0.0 {
                    continue;
                }
                for p in 0..=i {
                    mat[i * m + p] += ci * col[p];
                }
            }
        }
        for i in 0..m {
            for p in 0..i {
                mat[p * m + i] = mat[i * m + p];
            }
        }
        cholesky(&mut mat, m);
        mat
    }

    fn a_times(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.m];
        for (k, &xk) in x.iter().enumerate() {
            if xk != 0.0 {
                for (o, &a) in out.iter_mut().zip(self.column(k)) {
                    *o += a * xk;
                }
            }
        }
        out
    }

    fn at_times(&self, y: &[f64]) -> Vec<f64> {
        (0..self.vars.len())
            .map(|k| dot(self.column(k), y))
            .collect()
    }
}

/// Newton direction of the bounded primal–dual system.
struct Direction {
    dx: Vec<f64>,
    dw: Vec<f64>,
    dy: Vec<f64>,
    dz: Vec<f64>,
    dv: Vec<f64>,
}

/// Residuals of the Newton system: `A dx = rp`, `dx + dw = ru`, `Aᵀdy + dz − dv = rd`,
/// `Z dx + X dz = rxz`, `V dw + W dv = rwv`.
struct Rhs<'r> {
    rp: &'r [f64],
    ru: &'r [f64],
    rd: &'r [f64],
    rxz: &'r [f64],
    rwv: &'r [f64],
}

struct IpmState {
    x: Vec<f64>,
    w: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    v: Vec<f64>,
}

impl IpmState {
    fn direction(&self, prob: &IpmProblem, theta: &[f64], chol: &[f64], rhs: &Rhs) -> Direction {
        let nv = prob.vars.len();
        let mut h = vec![0.0; nv];
        for k in 0..nv {
            let mut hk = rhs.rd[k];
            match prob.vars[k].kind {
                IpmKind::Free => {}
                IpmKind::Lower => hk -= rhs.rxz[k] / self.x[k],
                IpmKind::Boxed => {
                    hk -= rhs.rxz[k] / self.x[k];
                    hk += (rhs.rwv[k] - self.v[k] * rhs.ru[k]) / self.w[k];
                }
            }
            h[k] = hk;
        }
        let th: Vec<f64> = theta.iter().zip(&h).map(|(t, hk)| t * hk).collect();
        let mut dy = prob.a_times(&th);
        for (d, &r) in dy.iter_mut().zip(rhs.rp) {
            *d += r;
        }
        cholesky_solve(chol, prob.m, &mut dy);
        let aty = prob.at_times(&dy);
        let mut dir = Direction {
            dx: vec![0.0; nv],
            dw: vec![0.0; nv],
            dy,
            dz: vec![0.0; nv],
            dv: vec![0.0; nv],
        };
        for k in 0..nv {
            let dx = theta[k] * (aty[k] - h[k]);
            dir.dx[k] = dx;
            match prob.vars[k].kind {
                IpmKind::Free => {}
                IpmKind::Lower => dir.dz[k] = (rhs.rxz[k] - self.z[k] * dx) / self.x[k],
                IpmKind::Boxed => {
                    dir.dz[k] = (rhs.rxz[k] - self.z[k] * dx) / self.x[k];
                    let dw = rhs.ru[k] - dx;
                    dir.dw[k] = dw;
                    dir.dv[k] = (rhs.rwv[k] - self.v[k] * dw) / self.w[k];
                }
            }
        }
        dir
    }

    /// Largest primal and dual steps in `(0, ∞)` that keep the bounded variables and their
    /// duals nonnegative.
    fn max_steps(&self, prob: &IpmProblem, dir: &Direction) -> (f64, f64) {
        let ratio = |value: f64, delta: f64| {
            if delta < 0.0 {
                -value / delta
            } else {
                f64::INFINITY
            }
        };
        let mut ap = f64::INFINITY;
        let mut ad = f64::INFINITY;
        for (k, var) in prob.vars.iter().enumerate() {
            if var.kind == IpmKind::Free {
                continue;
            }
            ap = ap.min(ratio(self.x[k], dir.dx[k]));
            ad = ad.min(ratio(self.z[k], dir.dz[k]));
            if var.kind == IpmKind::Boxed {
                ap = ap.min(ratio(self.w[k], dir.dw[k]));
                ad = ad.min(ratio(self.v[k], dir.dv[k]));
            }
        }
        (ap, ad)
    }

    /// Complementarity `xᵀz + wᵀv` after steps `ap`, `ad` along `dir`.
    fn complementarity(&self, prob: &IpmProblem, dir: &Direction, ap: f64, ad: f64) -> f64 {
        let mut total = 0.0;
        for (k, var) in prob.vars.iter().enumerate() {
            match var.kind {
                IpmKind::Free => {}
                IpmKind::Lower => {
                    total += (self.x[k] + ap * dir.dx[k]) * (self.z[k] + ad * dir.dz[k]);
                }
                IpmKind::Boxed => {
                    total += (self.x[k] + ap * dir.dx[k]) * (self.z[k] + ad * dir.dz[k]);
                    total += (self.w[k] + ap * dir.dw[k]) * (self.v[k] + ad * dir.dv[k]);
                }
            }
        }
        total
    }
}

/// Mehrotra's starting point, moved into the interior of the bounds.
fn ipm_start(prob: &IpmProblem) -> IpmState {
    let nv = prob.vars.len();
    let ones = vec![1.0; nv];
    let chol = prob.normal_factor(&ones);
    // Least-norm primal: x = Aᵀ (AAᵀ)⁻¹ b.
    let mut t = prob.b.clone();
    cholesky_solve(&chol, prob.m, &mut t);
    let mut x = prob.at_times(&t);
    // Least-squares dual: y = (AAᵀ)⁻¹ A c, r = c − Aᵀy.
    let mut y = prob.a_times(&prob.c);
    cholesky_solve(&chol, prob.m, &mut y);
    let aty = prob.at_times(&y);
    let r: Vec<f64> = prob.c.iter().zip(&aty).map(|(c, a)| c - a).collect();
    let mut w = vec![0.0; nv];
    let mut z = vec![0.0; nv];
    let mut v = vec![0.0; nv];
    let mut min_primal = f64::INFINITY;
    let mut min_dual = f64::INFINITY;
    for (k, var) in prob.vars.iter().enumerate() {
        match var.kind {
            IpmKind::Free => {}
            IpmKind::Lower => {
                z[k] = r[k];
                min_primal = min_primal.min(x[k]);
                min_dual = min_dual.min(z[k]);
            }
            IpmKind::Boxed => {
                w[k] = var.upper - x[k];
                z[k] = r[k].max(0.0);
                v[k] = (-r[k]).max(0.0);
                min_primal = min_primal.min(x[k]).min(w[k]);
                min_dual = min_dual.min(z[k]).min(v[k]);
            }
        }
    }
    let shift_p = (-1.5 * min_primal).max(0.0);
    let shift_d = (-1.5 * min_dual).max(0.0);
    let (mut xz, mut sum_p, mut sum_d) = (0.0, 0.0, 0.0);
    for (k, var) in prob.vars.iter().enumerate() {
        if var.kind == IpmKind::Free {
            continue;
        }
        x[k] += shift_p;
        z[k] += shift_d;
        xz += x[k] * z[k];
        sum_p += x[k];
        sum_d += z[k];
        if var.kind == IpmKind::Boxed {
            w[k] += shift_p;
            v[k] += shift_d;
            xz += w[k] * v[k];
            sum_p += w[k];
            sum_d += v[k];
        }
    }
    let second_p = if sum_d > 0.0 { 0.5 * xz / sum_d } else { 0.0 };
    let second_d = if sum_p > 0.0 { 0.5 * xz / sum_p } else { 0.0 };
    for (k, var) in prob.vars.iter().enumerate() {
        if var.kind == IpmKind::Free {
            continue;
        }
        let bump = |value: f64, extra: f64| {
            let shifted = value + extra;
            if shifted > 1e-8 && shifted.is_finite() {
                shifted
            } else {
                1.0
            }
        };
        x[k] = bump(x[k], second_p);
        z[k] = bump(z[k], second_d);
        if var.kind == IpmKind::Boxed {
            w[k] = bump(w[k], second_p);
            v[k] = bump(v[k], second_d);
        }
    }
    IpmState { x, w, y, z, v }
}

/// Runs the IPM; `limit` is the caller's iteration limit when `user_limit`.
fn ipm_iterate(
    prob: &IpmProblem,
    tolerances: Tolerances,
    optimality_tol: f64,
    limit: usize,
    user_limit: bool,
) -> (IpmEnd, IpmPoint, usize) {
    let nv = prob.vars.len();
    let feas_tol = tolerances.primal.min(tolerances.dual);
    let mut s = ipm_start(prob);
    let n_comp = prob
        .vars
        .iter()
        .map(|var| match var.kind {
            IpmKind::Free => 0,
            IpmKind::Lower => 1,
            IpmKind::Boxed => 2,
        })
        .sum::<usize>() as f64;
    let upper: Vec<f64> = prob
        .vars
        .iter()
        .map(|var| {
            if var.kind == IpmKind::Boxed {
                var.upper
            } else {
                0.0
            }
        })
        .collect();
    let b_norm = inf_norm(&prob.b);
    let c_norm = inf_norm(&prob.c);
    let u_norm = inf_norm(&upper);
    let data_scale = 1.0 + b_norm + c_norm + u_norm;
    let mut stalled = 0usize;
    let mut iterations = 0usize;
    let end = loop {
        // Residuals.
        let ax = prob.a_times(&s.x);
        let rp: Vec<f64> = prob.b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let mut ru = vec![0.0; nv];
        for (k, var) in prob.vars.iter().enumerate() {
            if var.kind == IpmKind::Boxed {
                ru[k] = var.upper - s.x[k] - s.w[k];
            }
        }
        let aty = prob.at_times(&s.y);
        let rd: Vec<f64> = (0..nv)
            .map(|k| prob.c[k] - aty[k] - s.z[k] + s.v[k])
            .collect();
        let comp = dot(&s.x, &s.z) + dot(&s.w, &s.v);
        let mu = comp / n_comp;
        let pobj = dot(&prob.c, &s.x);
        let dobj = dot(&prob.b, &s.y) - dot(&upper, &s.v);
        let pinf = (inf_norm(&rp) / (1.0 + b_norm)).max(inf_norm(&ru) / (1.0 + u_norm));
        let dinf = inf_norm(&rd) / (1.0 + c_norm);
        let gap = (pobj - dobj).abs() / (1.0 + pobj.abs());
        if !(pinf.is_finite() && dinf.is_finite() && gap.is_finite() && mu.is_finite()) {
            break IpmEnd::Failed;
        }
        if pinf <= feas_tol && dinf <= feas_tol && gap <= optimality_tol {
            break IpmEnd::Converged;
        }
        if iterations >= limit {
            break if user_limit {
                IpmEnd::Limit {
                    feasible: pinf <= feas_tol,
                }
            } else {
                IpmEnd::Failed
            };
        }
        let primal_size = inf_norm(&s.x).max(inf_norm(&s.w));
        let dual_size = inf_norm(&s.y).max(inf_norm(&s.z)).max(inf_norm(&s.v));
        if primal_size > 1e10 * data_scale || dual_size > 1e10 * data_scale {
            break IpmEnd::Failed;
        }
        iterations += 1;

        let theta: Vec<f64> = (0..nv)
            .map(|k| match prob.vars[k].kind {
                IpmKind::Free => 1.0 / IPM_FREE_REGULARIZATION,
                IpmKind::Lower => 1.0 / (s.z[k] / s.x[k]),
                IpmKind::Boxed => 1.0 / (s.z[k] / s.x[k] + s.v[k] / s.w[k]),
            })
            .collect();
        let chol = prob.normal_factor(&theta);

        // Predictor (affine scaling).
        let rxz_aff: Vec<f64> = (0..nv).map(|k| -s.x[k] * s.z[k]).collect();
        let rwv_aff: Vec<f64> = (0..nv).map(|k| -s.w[k] * s.v[k]).collect();
        let aff = s.direction(
            prob,
            &theta,
            &chol,
            &Rhs {
                rp: &rp,
                ru: &ru,
                rd: &rd,
                rxz: &rxz_aff,
                rwv: &rwv_aff,
            },
        );
        let (ap_aff, ad_aff) = s.max_steps(prob, &aff);
        let mu_aff = s.complementarity(prob, &aff, ap_aff.min(1.0), ad_aff.min(1.0)) / n_comp;
        let sigma = (mu_aff / mu).clamp(0.0, 1.0).powi(3);

        // Corrector (centering plus the second-order term).
        let rxz: Vec<f64> = (0..nv)
            .map(|k| sigma * mu - s.x[k] * s.z[k] - aff.dx[k] * aff.dz[k])
            .collect();
        let rwv: Vec<f64> = (0..nv)
            .map(|k| sigma * mu - s.w[k] * s.v[k] - aff.dw[k] * aff.dv[k])
            .collect();
        let dir = s.direction(
            prob,
            &theta,
            &chol,
            &Rhs {
                rp: &rp,
                ru: &ru,
                rd: &rd,
                rxz: &rxz,
                rwv: &rwv,
            },
        );
        let (ap_max, ad_max) = s.max_steps(prob, &dir);
        let ap = (IPM_STEP_FRACTION * ap_max).min(1.0);
        let ad = (IPM_STEP_FRACTION * ad_max).min(1.0);
        for k in 0..nv {
            s.x[k] += ap * dir.dx[k];
            s.w[k] += ap * dir.dw[k];
            s.z[k] += ad * dir.dz[k];
            s.v[k] += ad * dir.dv[k];
        }
        for (yi, dyi) in s.y.iter_mut().zip(&dir.dy) {
            *yi += ad * dyi;
        }
        if ap < 1e-8 && ad < 1e-8 {
            stalled += 1;
            if stalled >= 3 {
                break IpmEnd::Failed;
            }
        } else {
            stalled = 0;
        }
    };
    let point = IpmPoint {
        x: s.x,
        z: s.z,
        v: s.v,
    };
    (end, point, iterations)
}

fn solve_ipm(
    lp: &Lp,
    tolerances: Tolerances,
    optimality_tol: f64,
    maxiter: Option<usize>,
) -> Solved {
    let simplex_limit = maxiter.unwrap_or_else(|| simplex_safeguard(lp));
    let prob = IpmProblem::new(lp);
    let has_bounded = prob.vars.iter().any(|var| var.kind != IpmKind::Free);
    if lp.m == 0 || !has_bounded {
        // Nothing for an interior method to do (no rows, or no bound to stay inside):
        // HiGHS's presolve settles such problems without IPM iterations; the simplex does here.
        return simplex_from(Simplex::from_slack_basis(lp, tolerances), 0, simplex_limit);
    }
    let (limit, user_limit) = match maxiter {
        Some(limit) => (limit, true),
        None => (IPM_DEFAULT_ITERATIONS, false),
    };
    let (end, point, iterations) =
        ipm_iterate(&prob, tolerances, optimality_tol, limit, user_limit);
    match end {
        IpmEnd::Limit { feasible } => Solved {
            end: End::IterationLimit { feasible },
            nit: iterations,
            crossover_nit: 0,
        },
        IpmEnd::Converged => {
            let start = crossover_basis(lp, &prob, &point, tolerances)
                .or_else(|| Simplex::from_slack_basis(lp, tolerances));
            simplex_from(start, iterations, simplex_limit)
        }
        IpmEnd::Failed => simplex_from(
            Simplex::from_slack_basis(lp, tolerances),
            iterations,
            simplex_limit,
        ),
    }
}

fn simplex_from(start: Option<Simplex<'_>>, ipm_nit: usize, limit: usize) -> Solved {
    match start {
        Some(mut simplex) => {
            let end = simplex.solve(limit);
            Solved {
                end,
                nit: ipm_nit,
                crossover_nit: simplex.nit,
            }
        }
        None => Solved {
            end: End::SolveError,
            nit: ipm_nit,
            crossover_nit: 0,
        },
    }
}

/// Crossover: a starting basis for the simplex from an IPM optimum. Every non-fixed column is
/// ranked by how interior it is — its distance to the nearest bound over the dual of that
/// bound (free columns first) — and the most interior linearly independent columns form the
/// basis, completed with logicals; the rest sit at the bound the IPM approached.
fn crossover_basis<'a>(
    lp: &'a Lp,
    prob: &IpmProblem,
    point: &IpmPoint,
    tolerances: Tolerances,
) -> Option<Simplex<'a>> {
    let ncols = lp.ncols();
    let m = lp.m;
    // Simplex-coordinate values and bound duals from the IPM point.
    let mut value = vec![0.0; ncols];
    let mut dual_lo = vec![0.0; ncols];
    let mut dual_hi = vec![0.0; ncols];
    let mut in_ipm = vec![false; ncols];
    for j in 0..ncols {
        if lp.is_fixed(j) {
            value[j] = lp.lo[j];
        }
    }
    for (k, var) in prob.vars.iter().enumerate() {
        let j = var.j;
        in_ipm[j] = true;
        value[j] = var.shift + var.sign * point.x[k];
        if var.sign > 0.0 {
            dual_lo[j] = point.z[k];
            dual_hi[j] = point.v[k];
        } else {
            dual_hi[j] = point.z[k];
        }
    }
    // Ranking: the interior measure, highest first; logicals of equality rows last.
    let mut order: Vec<(usize, f64)> = Vec::with_capacity(ncols);
    for j in 0..ncols {
        if !in_ipm[j] {
            continue;
        }
        let (lo, hi) = (lp.lo[j], lp.hi[j]);
        let below = value[j] - lo;
        let above = hi - value[j];
        let measure = if lo.is_infinite() && hi.is_infinite() {
            f64::INFINITY
        } else if below <= above {
            below / dual_lo[j].max(1e-300)
        } else {
            above / dual_hi[j].max(1e-300)
        };
        order.push((j, measure));
    }
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut ranked: Vec<usize> = order.into_iter().map(|(j, _)| j).collect();
    ranked.extend(lp.n + lp.m_ub..ncols);

    // Greedy independent columns via re-orthogonalized Gram–Schmidt.
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut basis = Vec::with_capacity(m);
    let mut col = vec![0.0; m];
    for &j in &ranked {
        if basis.len() == m {
            break;
        }
        col.iter_mut().for_each(|v| *v = 0.0);
        lp.axpy_col(j, 1.0, &mut col);
        let norm0 = inf_norm(&col);
        if norm0 == 0.0 {
            continue;
        }
        for _ in 0..2 {
            for qv in &q {
                let proj = dot(qv, &col);
                for (c, &qi) in col.iter_mut().zip(qv) {
                    *c -= proj * qi;
                }
            }
        }
        let norm = dot(&col, &col).sqrt();
        if norm > 1e-7 * norm0 {
            q.push(col.iter().map(|v| v / norm).collect());
            basis.push(j);
        }
    }
    if basis.len() < m {
        return None;
    }
    let mut state = vec![VarState::AtLower; ncols];
    let mut x = vec![0.0; ncols];
    for &j in &basis {
        state[j] = VarState::Basic;
    }
    for j in 0..ncols {
        if state[j] == VarState::Basic {
            continue;
        }
        let (lo, hi) = (lp.lo[j], lp.hi[j]);
        let (s, xj) = if lo.is_finite() && hi.is_finite() {
            if lo == hi || value[j] - lo <= hi - value[j] {
                (VarState::AtLower, lo)
            } else {
                (VarState::AtUpper, hi)
            }
        } else if lo.is_finite() {
            (VarState::AtLower, lo)
        } else if hi.is_finite() {
            (VarState::AtUpper, hi)
        } else {
            (VarState::Zero, 0.0)
        };
        state[j] = s;
        x[j] = xj;
    }
    Simplex::with_basis(lp, tolerances, basis, state, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_METHODS: [LinprogMethod; 3] = [
        LinprogMethod::Highs,
        LinprogMethod::HighsDs,
        LinprogMethod::HighsIpm,
    ];

    fn opts(method: LinprogMethod) -> LinprogOptions {
        LinprogOptions {
            method,
            ..LinprogOptions::default()
        }
    }

    struct Problem {
        c: Vec<f64>,
        a_ub: Vec<Vec<f64>>,
        b_ub: Vec<f64>,
        a_eq: Vec<Vec<f64>>,
        b_eq: Vec<f64>,
        bounds: Vec<(Option<f64>, Option<f64>)>,
    }

    impl Problem {
        fn solve(&self, options: LinprogOptions) -> LinprogResult {
            linprog(
                &self.c,
                &self.a_ub,
                &self.b_ub,
                &self.a_eq,
                &self.b_eq,
                &self.bounds,
                options,
            )
            .expect("linprog")
        }

        fn bound_pairs(&self) -> Vec<(f64, f64)> {
            if self.bounds.is_empty() {
                return vec![(0.0, f64::INFINITY); self.c.len()];
            }
            self.bounds
                .iter()
                .map(|(l, u)| (l.unwrap_or(f64::NEG_INFINITY), u.unwrap_or(f64::INFINITY)))
                .collect()
        }

        /// Asserts the KKT conditions of an optimal result to `tol`: primal feasibility, the
        /// marginal signs, stationarity `c = A_ubᵀy_ub + A_eqᵀy_eq + lower + upper`,
        /// complementary slackness and a zero duality gap.
        fn assert_optimal(&self, res: &LinprogResult, tol: f64) {
            assert_eq!(res.status, 0, "{}", res.message);
            assert!(res.success);
            let n = self.c.len();
            let bounds = self.bound_pairs();
            let scale = 1.0 + res.fun.abs();
            for (j, &(l, u)) in bounds.iter().enumerate() {
                assert!(res.x[j] >= l - tol && res.x[j] <= u + tol, "x[{j}]");
                assert!(res.lower.marginals[j] >= -tol, "lower marginal sign {j}");
                assert!(res.upper.marginals[j] <= tol, "upper marginal sign {j}");
                if l.is_infinite() {
                    assert_eq!(res.lower.marginals[j], 0.0);
                    assert_eq!(res.lower.residual[j], f64::INFINITY);
                } else {
                    assert!((res.lower.marginals[j] * (res.x[j] - l)).abs() <= tol * scale);
                    assert!((res.lower.residual[j] - (res.x[j] - l)).abs() <= 1e-15 * scale);
                }
                if u.is_infinite() {
                    assert_eq!(res.upper.marginals[j], 0.0);
                    assert_eq!(res.upper.residual[j], f64::INFINITY);
                } else {
                    assert!((res.upper.marginals[j] * (u - res.x[j])).abs() <= tol * scale);
                }
            }
            for (i, row) in self.a_ub.iter().enumerate() {
                let s = self.b_ub[i] - dot(row, &res.x);
                assert!((s - res.slack[i]).abs() <= tol * scale);
                assert!(s >= -tol, "slack {i} = {s}");
                assert!(res.ineqlin.marginals[i] <= tol, "ineqlin sign {i}");
                assert!((res.ineqlin.marginals[i] * s).abs() <= tol * scale);
            }
            for (i, row) in self.a_eq.iter().enumerate() {
                let r = self.b_eq[i] - dot(row, &res.x);
                assert!(r.abs() <= tol * (1.0 + self.b_eq[i].abs()), "con {i} = {r}");
            }
            assert_eq!(res.slack, res.ineqlin.residual);
            assert_eq!(res.con, res.eqlin.residual);
            for j in 0..n {
                let mut grad = res.lower.marginals[j] + res.upper.marginals[j];
                for (i, row) in self.a_ub.iter().enumerate() {
                    grad += row[j] * res.ineqlin.marginals[i];
                }
                for (i, row) in self.a_eq.iter().enumerate() {
                    grad += row[j] * res.eqlin.marginals[i];
                }
                assert!(
                    (grad - self.c[j]).abs() <= tol * (1.0 + self.c[j].abs()),
                    "stationarity {j}: {grad} vs {}",
                    self.c[j]
                );
            }
            let mut dual_obj =
                dot(&self.b_ub, &res.ineqlin.marginals) + dot(&self.b_eq, &res.eqlin.marginals);
            for (j, &(l, u)) in bounds.iter().enumerate() {
                if l.is_finite() {
                    dual_obj += l * res.lower.marginals[j];
                }
                if u.is_finite() {
                    dual_obj += u * res.upper.marginals[j];
                }
            }
            assert!(
                (dual_obj - res.fun).abs() <= tol * scale,
                "duality gap {} vs {}",
                dual_obj,
                res.fun
            );
            assert!((dot(&self.c, &res.x) - res.fun).abs() <= 1e-15 * scale);
        }
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{got:?} vs {want:?}"
            );
        }
    }

    /// The example of the task and of SciPy's `linprog` documentation of marginals:
    /// `min -x - 2y : x + y ≤ 4, x − y ≤ 2, 0 ≤ x, 0 ≤ y ≤ 3`. SciPy 1.17.1 (all three HiGHS
    /// methods): x = [1, 3], ineqlin.marginals = [-1, -0], upper.marginals = [0, -1].
    #[test]
    fn marginals_match_scipy_on_the_documented_example() {
        let p = Problem {
            c: vec![-1.0, -2.0],
            a_ub: vec![vec![1.0, 1.0], vec![1.0, -1.0]],
            b_ub: vec![4.0, 2.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), None), (Some(0.0), Some(3.0))],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_eq!(res.message, MSG_OPTIMAL);
            assert_close(&res.x, &[1.0, 3.0], 1e-14);
            assert!((res.fun + 7.0).abs() < 1e-14);
            assert_close(&res.slack, &[0.0, 4.0], 1e-14);
            assert!(res.con.is_empty());
            assert_close(&res.ineqlin.marginals, &[-1.0, 0.0], 1e-14);
            assert_close(&res.upper.marginals, &[0.0, -1.0], 1e-14);
            assert_close(&res.lower.marginals, &[0.0, 0.0], 1e-14);
            assert_eq!(res.upper.residual[0], f64::INFINITY);
            assert_close(&res.lower.residual, &[1.0, 3.0], 1e-14);
            assert!(res.eqlin.residual.is_empty() && res.eqlin.marginals.is_empty());
            if method == LinprogMethod::HighsIpm {
                assert!(res.nit > 0, "the IPM must iterate");
            } else {
                assert_eq!(res.crossover_nit, 0);
            }
        }
    }

    /// SciPy's linprog docstring: `c = [-1, 4]`, `A = [[-3, 1], [1, 2]]`, `b = [6, 4]`,
    /// `x0 ∈ (-inf, inf)`, `x1 ∈ [-3, inf)` → x = [10, -3], fun = -22,
    /// ineqlin.marginals = [-0, -1], lower.marginals = [0, 6].
    #[test]
    fn scipy_docstring_example_with_a_free_variable() {
        let p = Problem {
            c: vec![-1.0, 4.0],
            a_ub: vec![vec![-3.0, 1.0], vec![1.0, 2.0]],
            b_ub: vec![6.0, 4.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(None, None), (Some(-3.0), None)],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[10.0, -3.0], 1e-14);
            assert!((res.fun + 22.0).abs() < 1e-13);
            assert_close(&res.ineqlin.residual, &[39.0, 0.0], 1e-14);
            assert_close(&res.ineqlin.marginals, &[0.0, -1.0], 1e-14);
            assert_close(&res.lower.marginals, &[0.0, 6.0], 1e-14);
            assert_eq!(res.lower.residual[0], f64::INFINITY);
        }
    }

    /// `min -5x - 4y : 6x + 4y ≤ 24, x + 2y ≤ 6` → (3, 1.5), fun −21, duals (−0.75, −0.5).
    #[test]
    fn happy_path_textbook_lp() {
        let p = Problem {
            c: vec![-5.0, -4.0],
            a_ub: vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            b_ub: vec![24.0, 6.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[3.0, 1.5], 1e-14);
            assert_close(&res.ineqlin.marginals, &[-0.75, -0.5], 1e-14);
        }
    }

    /// Three constraints active at (1, 1) in the plane: primal degenerate, so the dual is not
    /// unique ([-1, -1, 0] and [0, 0, -1] are both optimal). Only the optimality conditions
    /// are fixed.
    #[test]
    fn degenerate_vertex_gives_a_valid_dual() {
        let p = Problem {
            c: vec![-1.0, -1.0],
            a_ub: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
            b_ub: vec![1.0, 1.0, 2.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[1.0, 1.0], 1e-14);
            assert!((res.fun + 2.0).abs() < 1e-14);
        }
    }

    /// Beale's example (1955), on which Dantzig's rule with lowest-index tie breaking cycles.
    /// SciPy: x = [1, 0, 1, 0], fun = −1.25, ineqlin.marginals = [0, −1.5, −1.25],
    /// lower.marginals = [0, 2, 0, 10.5].
    fn beale() -> Problem {
        Problem {
            c: vec![-0.75, 20.0, -0.5, 6.0],
            a_ub: vec![
                vec![0.25, -8.0, -1.0, 9.0],
                vec![0.5, -12.0, -0.5, 3.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            b_ub: vec![0.0, 0.0, 1.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![],
        }
    }

    #[test]
    fn beales_cycling_example_terminates() {
        let p = beale();
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[1.0, 0.0, 1.0, 0.0], 1e-14);
            assert!((res.fun + 1.25).abs() < 1e-14);
            assert_close(&res.ineqlin.marginals, &[0.0, -1.5, -1.25], 1e-14);
            assert_close(&res.lower.marginals, &[0.0, 2.0, 0.0, 10.5], 1e-14);
        }
    }

    /// Must-hit arm of the anti-cycling claim. The default path (Harris, ties to the largest
    /// pivot) happens not to cycle on Beale's example, so that test alone would not show the
    /// fallback works. Under the classic cycling configuration — Dantzig's entering rule with
    /// ties in the ratio test going to the lowest index — Beale's example does cycle: with the
    /// Bland fallback disabled it never reaches the optimum in 1000 iterations, and with the
    /// fallback enabled the same configuration finishes at the optimum.
    #[test]
    fn beale_cycles_without_the_bland_fallback() {
        let p = beale();
        let lb = vec![0.0; 4];
        let ub = vec![f64::INFINITY; 4];
        let lp = Lp::new(&p.c, &p.a_ub, &p.b_ub, &[], &[], &lb, &ub);
        let tol = Tolerances {
            primal: 1e-7,
            dual: 1e-7,
        };
        let mut cycling = Simplex::from_slack_basis(&lp, tol).expect("basis");
        cycling.leaving = LeavingRule::LowestIndex;
        cycling.bland_after = usize::MAX;
        assert_eq!(cycling.run(1000), Run::IterationLimit { feasible: true });
        assert_eq!(cycling.nit, 1000);
        let mut guarded = Simplex::from_slack_basis(&lp, tol).expect("basis");
        guarded.leaving = LeavingRule::LowestIndex;
        assert_eq!(guarded.run(1000), Run::Optimal);
        assert!(
            guarded.nit > guarded.bland_after,
            "the fallback must have engaged"
        );
        let (y, _) = guarded.duals();
        assert_eq!(guarded.x[..4].to_vec(), vec![1.0, 0.0, 1.0, 0.0]);
        assert_close(&y, &[0.0, -1.5, -1.25], 1e-14);
    }

    #[test]
    fn infeasible_problems_report_status_2() {
        let cases = [
            Problem {
                c: vec![1.0],
                a_ub: vec![],
                b_ub: vec![],
                a_eq: vec![vec![1.0]],
                b_eq: vec![-1.0],
                bounds: vec![],
            },
            Problem {
                c: vec![1.0, 1.0],
                a_ub: vec![vec![1.0, 1.0], vec![-1.0, -1.0]],
                b_ub: vec![1.0, -2.0],
                a_eq: vec![],
                b_eq: vec![],
                bounds: vec![],
            },
        ];
        for p in &cases {
            for method in ALL_METHODS {
                let res = p.solve(opts(method));
                assert_eq!(res.status, 2, "{method:?}: {}", res.message);
                assert!(!res.success);
                assert!(res.message.starts_with(
                    "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is "
                ));
                assert!(res.x.is_empty() && res.slack.is_empty() && res.fun.is_nan());
                assert!(res.lower.marginals.is_empty() && res.ineqlin.residual.is_empty());
            }
        }
    }

    #[test]
    fn inconsistent_bounds_are_infeasible_and_infinite_wrong_side_bounds_a_model_error() {
        let crossed = linprog(
            &[1.0],
            &[],
            &[],
            &[],
            &[],
            &[(Some(2.0), Some(1.0))],
            opts(LinprogMethod::Highs),
        )
        .expect("linprog");
        assert_eq!(crossed.status, 2);
        assert_eq!(crossed.message, msg_infeasible("None"));
        for bound in [
            (Some(f64::INFINITY), None),
            (None, Some(f64::NEG_INFINITY)),
            (Some(f64::INFINITY), Some(f64::INFINITY)),
        ] {
            let res = linprog(
                &[1.0],
                &[],
                &[],
                &[],
                &[],
                &[bound],
                opts(LinprogMethod::HighsIpm),
            )
            .expect("linprog");
            assert_eq!(res.status, 2);
            assert_eq!(res.message, "(HiGHS Status 2: Model error)");
        }
    }

    #[test]
    fn unbounded_problems_report_status_3() {
        let cases = [
            Problem {
                c: vec![-1.0, 0.0],
                a_ub: vec![vec![1.0, -1.0]],
                b_ub: vec![1.0],
                a_eq: vec![],
                b_eq: vec![],
                bounds: vec![],
            },
            Problem {
                c: vec![1.0],
                a_ub: vec![],
                b_ub: vec![],
                a_eq: vec![],
                b_eq: vec![],
                bounds: vec![(None, None)],
            },
            // x1 = x2, both free: c·x = 2·x1 has no lower bound.
            Problem {
                c: vec![1.0, 1.0],
                a_ub: vec![],
                b_ub: vec![],
                a_eq: vec![vec![1.0, -1.0]],
                b_eq: vec![0.0],
                bounds: vec![(None, None), (None, None)],
            },
        ];
        for p in &cases {
            for method in ALL_METHODS {
                let res = p.solve(opts(method));
                assert_eq!(res.status, 3, "{method:?}: {}", res.message);
                assert_eq!(res.message, MSG_UNBOUNDED);
                assert!(res.x.is_empty() && res.fun.is_nan());
            }
        }
    }

    /// `min t : |x1 − 2| ≤ t, |x2 + 1| ≤ t, x1 + x2 = 3` with x1, x2 free → x = (3, 0), t = 1.
    #[test]
    fn free_variables_stay_single_columns() {
        let p = Problem {
            c: vec![0.0, 0.0, 1.0],
            a_ub: vec![
                vec![1.0, 0.0, -1.0],
                vec![-1.0, 0.0, -1.0],
                vec![0.0, 1.0, -1.0],
                vec![0.0, -1.0, -1.0],
            ],
            b_ub: vec![2.0, -2.0, -1.0, 1.0],
            a_eq: vec![vec![1.0, 1.0, 0.0]],
            b_eq: vec![3.0],
            bounds: vec![(None, None), (None, None), (Some(0.0), None)],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[3.0, 0.0, 1.0], 1e-13);
            assert_eq!(res.upper.residual, vec![f64::INFINITY; 3]);
        }
    }

    /// `min x + 2y + 3z : x + y + z = 6, x − y = 2` → (4, 2, 0) with eqlin marginals (1.5, −0.5).
    #[test]
    fn equality_only_problem() {
        let p = Problem {
            c: vec![1.0, 2.0, 3.0],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![vec![1.0, 1.0, 1.0], vec![1.0, -1.0, 0.0]],
            b_eq: vec![6.0, 2.0],
            bounds: vec![],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert_close(&res.x, &[4.0, 2.0, 0.0], 1e-14);
            assert_close(&res.eqlin.marginals, &[1.5, -0.5], 1e-14);
            assert_close(&res.con, &[0.0, 0.0], 1e-14);
            assert_close(&res.lower.marginals, &[0.0, 0.0, 1.5], 1e-14);
        }
    }

    /// No rows: every variable goes to the bound its cost prefers; marginals are `c`.
    #[test]
    fn bounds_only_problem() {
        let p = Problem {
            c: vec![1.0, -2.0, 0.5],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![
                (Some(1.0), Some(3.0)),
                (Some(-2.0), Some(4.0)),
                (Some(-1.0), Some(1.0)),
            ],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-14);
            assert_eq!(res.x, vec![1.0, 4.0, -1.0]);
            assert_eq!(res.lower.marginals, vec![1.0, 0.0, 0.5]);
            assert_eq!(res.upper.marginals, vec![0.0, -2.0, 0.0]);
            if method == LinprogMethod::HighsIpm {
                // No rows: nothing for the IPM to do; the simplex settles it.
                assert_eq!(res.nit, 0);
            }
        }
    }

    /// Default bounds `(0, None)` and no rows at all.
    #[test]
    fn empty_constraints_use_default_bounds() {
        for method in ALL_METHODS {
            let res =
                linprog(&[2.0, 0.0, 1.0], &[], &[], &[], &[], &[], opts(method)).expect("linprog");
            assert_eq!(res.status, 0);
            assert_eq!(res.x, vec![0.0, 0.0, 0.0]);
            assert_eq!(res.fun, 0.0);
            assert_eq!(res.lower.marginals, vec![2.0, 0.0, 1.0]);
            assert!(res.slack.is_empty() && res.con.is_empty());
        }
    }

    /// Fixed variables: SciPy puts a fixed variable's reduced cost in `lower` when it is
    /// nonnegative and in `upper` otherwise (fixed: lower = [1, 0, 2]; fixedneg: upper[0] = −1).
    #[test]
    fn fixed_variable_marginals_follow_highs_split() {
        let p = Problem {
            c: vec![1.0, -1.0, 2.0],
            a_ub: vec![vec![1.0, 1.0, 1.0]],
            b_ub: vec![10.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![
                (Some(2.0), Some(2.0)),
                (Some(0.0), Some(5.0)),
                (Some(-1.0), Some(-1.0)),
            ],
        };
        let q = Problem {
            c: vec![-1.0, 1.0],
            a_ub: vec![vec![1.0, 1.0]],
            b_ub: vec![10.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(2.0), Some(2.0)), (Some(0.0), Some(5.0))],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-13);
            assert_eq!(res.x, vec![2.0, 5.0, -1.0]);
            assert_eq!(res.lower.marginals, vec![1.0, 0.0, 2.0]);
            assert_eq!(res.upper.marginals, vec![0.0, -1.0, 0.0]);
            let res = q.solve(opts(method));
            q.assert_optimal(&res, 1e-13);
            assert_eq!(res.lower.marginals, vec![0.0, 1.0]);
            assert_eq!(res.upper.marginals, vec![-1.0, 0.0]);
        }
    }

    /// A balanced transportation problem keeps all of its (linearly dependent) equality rows.
    #[test]
    fn redundant_equality_rows_are_handled() {
        // 2 sources (supply 3, 5) × 3 sinks (demand 2, 4, 2); cost[i][j].
        let cost = [[4.0, 6.0, 9.0], [5.0, 3.0, 8.0]];
        let mut a_eq = Vec::new();
        let mut b_eq = Vec::new();
        for (i, supply) in [3.0, 5.0].into_iter().enumerate() {
            let mut row = vec![0.0; 6];
            for j in 0..3 {
                row[i * 3 + j] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(supply);
        }
        for (j, demand) in [2.0, 4.0, 2.0].into_iter().enumerate() {
            let mut row = vec![0.0; 6];
            for i in 0..2 {
                row[i * 3 + j] = 1.0;
            }
            a_eq.push(row);
            b_eq.push(demand);
        }
        let p = Problem {
            c: cost.iter().flatten().copied().collect(),
            a_ub: vec![],
            b_ub: vec![],
            a_eq,
            b_eq,
            bounds: vec![],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            // Optimum by hand: x = [[2, 0, 1], [0, 4, 1]] → 8 + 9 + 12 + 8 = 37.
            assert!((res.fun - 37.0).abs() < 1e-12, "{}", res.fun);
        }
    }

    /// Several optimal vertices: the objective is still exact and the answer optimal.
    #[test]
    fn non_unique_primal_optimum() {
        let p = Problem {
            c: vec![1.0, 1.0],
            a_ub: vec![],
            b_ub: vec![],
            a_eq: vec![vec![1.0, 1.0]],
            b_eq: vec![10.0],
            bounds: vec![],
        };
        for method in ALL_METHODS {
            let res = p.solve(opts(method));
            p.assert_optimal(&res, 1e-12);
            assert!((res.fun - 10.0).abs() < 1e-13);
            assert_close(&res.eqlin.marginals, &[1.0], 1e-14);
        }
    }

    /// SciPy reports `status = 1` with no solution when the limit is reached, including
    /// `maxiter = 0` (live SciPy: `Iteration limit reached. (HiGHS Status 14: …)`). The
    /// simplex needs two pivots here and the IPM more than one iteration.
    #[test]
    fn iteration_limit_reports_status_1() {
        let p = Problem {
            c: vec![-5.0, -4.0],
            a_ub: vec![vec![6.0, 4.0], vec![1.0, 2.0]],
            b_ub: vec![24.0, 6.0],
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![],
        };
        for method in ALL_METHODS {
            for maxiter in [0, 1] {
                let res = p.solve(LinprogOptions {
                    method,
                    maxiter: Some(maxiter),
                    ..LinprogOptions::default()
                });
                assert_eq!(
                    res.status, 1,
                    "{method:?} maxiter={maxiter}: {}",
                    res.message
                );
                assert!(res.message.starts_with(
                    "Iteration limit reached. (HiGHS Status 14: model_status is Iteration limit reached; primal_status is "
                ));
                assert_eq!(res.nit, maxiter);
                assert!(res.x.is_empty() && !res.success);
            }
            let enough = p.solve(LinprogOptions {
                method,
                maxiter: Some(100),
                ..LinprogOptions::default()
            });
            assert_eq!(enough.status, 0);
        }
    }

    #[test]
    fn invalid_input_is_rejected() {
        let ok = opts(LinprogMethod::Highs);
        assert!(matches!(
            linprog(&[], &[], &[], &[], &[], &[], ok),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linprog(&[1.0, 2.0], &[vec![1.0]], &[5.0], &[], &[], &[], ok),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[vec![1.0]], &[], &[], &[], &[], ok),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[], &[], &[vec![1.0], vec![2.0]], &[1.0], &[], ok),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linprog(&[1.0, 2.0], &[], &[], &[], &[], &[(None, None)], ok),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linprog(&[f64::NAN], &[], &[], &[], &[], &[], ok),
            Err(OptError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[vec![f64::INFINITY]], &[1.0], &[], &[], &[], ok),
            Err(OptError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[vec![1.0]], &[f64::INFINITY], &[], &[], &[], ok),
            Err(OptError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[], &[], &[vec![1.0]], &[f64::NAN], &[], ok),
            Err(OptError::NonFiniteInput { .. })
        ));
        assert!(matches!(
            linprog(&[1.0], &[], &[], &[], &[], &[(Some(f64::NAN), None)], ok),
            Err(OptError::InvalidBounds { .. })
        ));
        for bad in [0.0, -1e-7, f64::NAN, f64::INFINITY] {
            let options = LinprogOptions {
                primal_feasibility_tolerance: bad,
                ..LinprogOptions::default()
            };
            assert!(matches!(
                linprog(&[1.0], &[], &[], &[], &[], &[], options),
                Err(OptError::InvalidArgument { .. })
            ));
        }
    }

    /// A seeded dense LP (20 `≤` rows, 10 equality rows, 50 mixed-bound variables) built to be
    /// feasible (x0 is interior) and bounded (c is a dual-feasible combination). Both
    /// methods must reach the same objective and satisfy the optimality conditions.
    #[test]
    fn random_dense_lp_methods_agree() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let (m_ub, m_eq, n) = (20, 10, 50);
        let a_ub: Vec<Vec<f64>> = (0..m_ub)
            .map(|_| (0..n).map(|_| next()).collect())
            .collect();
        let a_eq: Vec<Vec<f64>> = (0..m_eq)
            .map(|_| (0..n).map(|_| next()).collect())
            .collect();
        let bounds: Vec<(Option<f64>, Option<f64>)> = (0..n)
            .map(|j| match j % 5 {
                0 | 1 => (Some(-1.0), Some(1.0)),
                2 => (Some(0.0), None),
                3 => (None, Some(2.0)),
                _ => (Some(-2.0), Some(0.5)),
            })
            .collect();
        let x0: Vec<f64> = (0..n)
            .map(|j| 0.3 * next() + if j % 5 == 2 { 0.5 } else { 0.0 })
            .collect();
        let b_ub: Vec<f64> = a_ub
            .iter()
            .map(|row| dot(row, &x0) + 0.5 + 0.5 * next().abs())
            .collect();
        let b_eq: Vec<f64> = a_eq.iter().map(|row| dot(row, &x0)).collect();
        let y_ub: Vec<f64> = (0..m_ub).map(|_| -next().abs()).collect();
        let y_eq: Vec<f64> = (0..m_eq).map(|_| next()).collect();
        let c: Vec<f64> = (0..n)
            .map(|j| {
                let mut cj = 0.0;
                for i in 0..m_ub {
                    cj += a_ub[i][j] * y_ub[i];
                }
                for i in 0..m_eq {
                    cj += a_eq[i][j] * y_eq[i];
                }
                match j % 5 {
                    2 => cj + next().abs(),
                    3 => cj - next().abs(),
                    _ => cj + next(),
                }
            })
            .collect();
        let p = Problem {
            c,
            a_ub,
            b_ub,
            a_eq,
            b_eq,
            bounds,
        };
        let simplex = p.solve(opts(LinprogMethod::HighsDs));
        p.assert_optimal(&simplex, 1e-9);
        let ipm = p.solve(opts(LinprogMethod::HighsIpm));
        p.assert_optimal(&ipm, 1e-9);
        assert!((simplex.fun - ipm.fun).abs() <= 1e-10 * (1.0 + simplex.fun.abs()));
        assert_close(&simplex.x, &ipm.x, 1e-9);
        assert!(ipm.nit > 0);
    }

    /// The product-form update must agree with a fresh factorization: a 60-row problem forces
    /// at least one refactorization (more than [`REFACTOR_INTERVAL`] pivots).
    #[test]
    fn refactorization_keeps_the_solution_exact() {
        let n = 60;
        // min −Σx subject to x_i + x_{i+1} ≤ 1 + i/n and 0 ≤ x ≤ 1.
        let mut a_ub = Vec::new();
        let mut b_ub = Vec::new();
        for i in 0..n {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row[(i + 1) % n] = 1.0;
            a_ub.push(row);
            b_ub.push(1.0 + i as f64 / n as f64);
        }
        let p = Problem {
            c: (0..n).map(|j| -1.0 - (j % 7) as f64 * 0.1).collect(),
            a_ub,
            b_ub,
            a_eq: vec![],
            b_eq: vec![],
            bounds: vec![(Some(0.0), Some(1.0)); n],
        };
        let simplex = p.solve(opts(LinprogMethod::HighsDs));
        assert!(simplex.nit > REFACTOR_INTERVAL, "nit = {}", simplex.nit);
        p.assert_optimal(&simplex, 1e-12);
        let ipm = p.solve(opts(LinprogMethod::HighsIpm));
        p.assert_optimal(&ipm, 1e-12);
        assert!((simplex.fun - ipm.fun).abs() <= 1e-12 * simplex.fun.abs());
    }

    #[test]
    fn dense_lu_solves_and_transposed_solves() {
        let b = vec![2.0, 1.0, 0.0, 4.0, 3.0, 1.0, 0.0, 5.0, 7.0];
        let lu = DenseLu::factor(b.clone(), 3).expect("nonsingular");
        let mut x = vec![1.0, 2.0, 3.0];
        lu.solve(&mut x);
        for i in 0..3 {
            let bx: f64 = (0..3).map(|j| b[i * 3 + j] * x[j]).sum();
            assert!((bx - [1.0, 2.0, 3.0][i]).abs() < 1e-14);
        }
        let mut y = vec![1.0, -1.0, 2.0];
        lu.solve_transpose(&mut y);
        for j in 0..3 {
            let bty: f64 = (0..3).map(|i| b[i * 3 + j] * y[i]).sum();
            assert!((bty - [1.0, -1.0, 2.0][j]).abs() < 1e-14);
        }
        assert!(DenseLu::factor(vec![1.0, 2.0, 2.0, 4.0], 2).is_none());
    }
}
