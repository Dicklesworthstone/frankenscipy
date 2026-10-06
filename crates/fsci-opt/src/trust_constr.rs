#![forbid(unsafe_code)]
//! `scipy.optimize.minimize(method='trust-constr')`, transcribed from SciPy 1.17's
//! `scipy/optimize/_trustregion_constr` (BSD): the driver
//! (`minimize_trustregion_constr.py`), Byrd–Omojokun trust-region SQP for equality constraints
//! (`equality_constrained_sqp.py`), the trust-region interior-point (barrier) method that wraps
//! it when inequalities are present (`tr_interior_point.py`), the projected-CG and modified
//! dogleg subproblem solvers (`qp_subproblem.py`), the null-space / least-squares / row-space
//! projections (`projections.py`) and the canonical constraint form (`canonical_constraint.py`).
//!
//! The memoisation of SciPy's `ScalarFunction` / `VectorFunction` is reproduced as well,
//! because it is part of the algorithm: their quasi-Newton (BFGS) Hessians are updated whenever
//! the evaluation point moves — trial points included, and a constraint's again whenever its
//! multipliers change — and their finite-difference derivatives are SciPy's `'2-point'` scheme
//! with its relative step. Evaluation counts are SciPy's.
//!
//! Dense Jacobians only. Where SciPy's sparse path (`AugmentedSystem`, its default when the
//! Jacobian is sparse — a bounds-only problem, or `sparse_jacobian=True`) factorises the
//! augmented system with SuperLU, this module uses a dense LU with partial pivoting of the same
//! system: the same projections, differing only in rounding.

use fsci_runtime::RuntimeMode;

use crate::types::{
    Bound, Bounds, Constraint, ConstraintType, ConvergenceStatus, GradientFunc, HessFunc,
    HesspFunc, LinearConstraint, MinimizeCallback, MinimizeOptions, NonlinearConstraint, OptError,
    OptimizeResult,
};
use crate::{BfgsExceptionStrategy, BfgsHessian, HessianApproxType};

/// SciPy's `TERMINATION_MESSAGES`, indexed by status.
const TERMINATION_MESSAGES: [&str; 5] = [
    "The maximum number of function evaluations is exceeded.",
    "`gtol` termination condition is satisfied.",
    "`xtol` termination condition is satisfied.",
    "`callback` raised `StopIteration`.",
    "Constraint violation exceeds 'gtol'",
];

/// `np.finfo(np.float64).eps ** 0.5`, the relative step of SciPy's `'2-point'` differences.
const FD_REL_STEP: f64 = 1.490_116_119_384_765_6e-8;

/// SciPy's `projections(..., orth_tol=1e-12, max_refin=3, tol=1e-15)` defaults.
const ORTH_TOL: f64 = 1e-12;
const MAX_REFIN: usize = 3;
const PROJECTION_TOL: f64 = 1e-15;

/// The message SciPy raises for a dense Jacobian with more rows than columns.
const EXPECTED_SQUARE_MATRIX: &str = "The 'expected square matrix' error can occur if there are \
more equality constraints than independent variables. Consider how your constraints are set up, \
or use factorization_method='SVDFactorization'.";

// ══════════════════════════════════════════════════════════════════════
// Public types
// ══════════════════════════════════════════════════════════════════════

/// SciPy's `factorization_method`: how the projections onto the null space and row space of
/// the constraint Jacobian are computed.
///
/// `QrFactorization` (the default for dense Jacobians) and `SvdFactorization` are SciPy's
/// dense methods; `AugmentedSystem` (the default for sparse ones) and `NormalEquation` its
/// sparse methods. SciPy uses `NormalEquation` only with scikit-sparse installed and
/// otherwise falls back to `AugmentedSystem`, as this crate always does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorizationMethod {
    NormalEquation,
    AugmentedSystem,
    QrFactorization,
    SvdFactorization,
}

/// The algorithm trust-constr ran: SciPy's `result.method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustConstrMethod {
    /// Byrd–Omojokun trust-region SQP: no inequality constraints.
    EqualityConstrainedSqp,
    /// Trust-region interior point (barrier) method wrapping the SQP.
    TrInteriorPoint,
}

impl TrustConstrMethod {
    /// SciPy's name for the method.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EqualityConstrainedSqp => "equality_constrained_sqp",
            Self::TrInteriorPoint => "tr_interior_point",
        }
    }
}

/// Vector-valued constraint function.
pub type TrustConstraintFn<'a> = dyn Fn(&[f64]) -> Vec<f64> + 'a;
/// Constraint Jacobian: one row per component.
pub type TrustConstraintJacFn<'a> = dyn Fn(&[f64]) -> Vec<Vec<f64>> + 'a;
/// SciPy's constraint `hess(x, v)`: the Hessian of `dot(fun(x), v)`, `n × n`.
pub type TrustConstraintHessFn<'a> = dyn Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + 'a;

/// A constraint `lb <= fun(x) <= ub` as trust-constr consumes it natively: SciPy's
/// `LinearConstraint(A, lb, ub, keep_feasible)` (constant Jacobian `A`, zero Hessian) or
/// `NonlinearConstraint(fun, lb, ub, jac, hess, keep_feasible, finite_diff_rel_step)`. Without
/// `jac` the Jacobian is SciPy's `'2-point'` difference; without `hess` its Hessian term is a
/// BFGS approximation, as in SciPy. A component with `lb == ub` is an equality.
///
/// `lb`, `ub` and `keep_feasible` hold one entry per component, or a single entry that is
/// broadcast (SciPy's scalar bounds). SciPy does not require `lb <= ub` here, and neither does
/// this type.
pub struct TrustConstraint<'a> {
    kind: TrustConstraintKind<'a>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    keep_feasible: Vec<bool>,
}

enum TrustConstraintKind<'a> {
    Linear(Vec<Vec<f64>>),
    Nonlinear {
        fun: &'a TrustConstraintFn<'a>,
        jac: Option<&'a TrustConstraintJacFn<'a>>,
        hess: Option<&'a TrustConstraintHessFn<'a>>,
        rel_step: Option<f64>,
    },
}

impl<'a> TrustConstraint<'a> {
    /// `LinearConstraint(A, lb, ub)`: `lb <= A x <= ub`, `A` given by rows.
    #[must_use]
    pub fn linear(a: Vec<Vec<f64>>, lb: Vec<f64>, ub: Vec<f64>) -> Self {
        Self {
            kind: TrustConstraintKind::Linear(a),
            lb,
            ub,
            keep_feasible: vec![false],
        }
    }

    /// The crate's [`LinearConstraint`] as a native trust-constr constraint.
    #[must_use]
    pub fn from_linear(con: &LinearConstraint) -> Self {
        Self::linear(con.a.clone(), con.lb.clone(), con.ub.clone())
    }

    /// `NonlinearConstraint(fun, lb, ub)`: `lb <= fun(x) <= ub`.
    #[must_use]
    pub fn nonlinear(fun: &'a TrustConstraintFn<'a>, lb: Vec<f64>, ub: Vec<f64>) -> Self {
        Self {
            kind: TrustConstraintKind::Nonlinear {
                fun,
                jac: None,
                hess: None,
                rel_step: None,
            },
            lb,
            ub,
            keep_feasible: vec![false],
        }
    }

    /// The crate's [`NonlinearConstraint`] as a native trust-constr constraint.
    #[must_use]
    pub fn from_nonlinear(con: &'a NonlinearConstraint) -> Self {
        Self::nonlinear(&con.fun, con.lb.clone(), con.ub.clone())
    }

    /// SciPy's `old_constraint_to_new`: `{'type': 'eq', ...}` is `0 <= fun(x) <= 0` and
    /// `{'type': 'ineq', ...}` is `0 <= fun(x) <= inf`, with the dict's `jac` when given and
    /// `'2-point'` differences otherwise.
    #[must_use]
    pub fn from_dict(con: &'a Constraint<'a>) -> Self {
        let ub = match con.kind {
            ConstraintType::Eq => 0.0,
            ConstraintType::Ineq => f64::INFINITY,
        };
        let fun: &'a TrustConstraintFn<'a> = &*con.fun;
        let jac: Option<&'a TrustConstraintJacFn<'a>> = match &con.jac {
            Some(jac) => Some(&**jac),
            None => None,
        };
        Self {
            kind: TrustConstraintKind::Nonlinear {
                fun,
                jac,
                hess: None,
                rel_step: None,
            },
            lb: vec![0.0],
            ub: vec![ub],
            keep_feasible: vec![false],
        }
    }

    /// The Jacobian of a nonlinear constraint (no effect on a linear one, whose Jacobian is `A`).
    #[must_use]
    pub fn with_jac(mut self, jac: &'a TrustConstraintJacFn<'a>) -> Self {
        if let TrustConstraintKind::Nonlinear { jac: slot, .. } = &mut self.kind {
            *slot = Some(jac);
        }
        self
    }

    /// SciPy's `hess(x, v)` of a nonlinear constraint (no effect on a linear one).
    #[must_use]
    pub fn with_hess(mut self, hess: &'a TrustConstraintHessFn<'a>) -> Self {
        if let TrustConstraintKind::Nonlinear { hess: slot, .. } = &mut self.kind {
            *slot = Some(hess);
        }
        self
    }

    /// SciPy's `finite_diff_rel_step` of a nonlinear constraint's `'2-point'` Jacobian.
    #[must_use]
    pub fn with_finite_diff_rel_step(mut self, step: f64) -> Self {
        if let TrustConstraintKind::Nonlinear { rel_step, .. } = &mut self.kind {
            *rel_step = Some(step);
        }
        self
    }

    /// SciPy's `keep_feasible`: the inequality components to keep feasible along the
    /// iterations (one flag per component, or one for all).
    #[must_use]
    pub fn with_keep_feasible(mut self, keep_feasible: Vec<bool>) -> Self {
        self.keep_feasible = keep_feasible;
        self
    }
}

impl std::fmt::Debug for TrustConstraint<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.kind {
            TrustConstraintKind::Linear(_) => "linear",
            TrustConstraintKind::Nonlinear { .. } => "nonlinear",
        };
        f.debug_struct("TrustConstraint")
            .field("kind", &kind)
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("keep_feasible", &self.keep_feasible)
            .finish()
    }
}

/// SciPy's `Bounds(lb, ub, keep_feasible)` for trust-constr. Bounds enter the problem as one
/// more constraint, placed after the others; `keep_feasible` bounds are also never crossed by
/// the iterates or by finite-difference steps.
#[derive(Debug, Clone, PartialEq)]
pub struct TrustBounds {
    pub lb: Vec<f64>,
    pub ub: Vec<f64>,
    /// One flag per variable, or one for all.
    pub keep_feasible: Vec<bool>,
}

impl TrustBounds {
    /// `Bounds(lb, ub)`; use `±inf` for a missing side.
    #[must_use]
    pub fn new(lb: Vec<f64>, ub: Vec<f64>) -> Self {
        Self {
            lb,
            ub,
            keep_feasible: vec![false],
        }
    }

    /// The crate's [`Bounds`].
    #[must_use]
    pub fn from_bounds(bounds: &Bounds) -> Self {
        Self::new(bounds.lb.clone(), bounds.ub.clone())
    }

    /// SciPy's `old_bound_to_new` for `(min, max)` pairs, `None` being unbounded.
    #[must_use]
    pub fn from_tuples(bounds: &[Bound]) -> Self {
        let (lb, ub) = bounds
            .iter()
            .map(|&(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .unzip();
        Self::new(lb, ub)
    }

    /// SciPy's `keep_feasible`.
    #[must_use]
    pub fn with_keep_feasible(mut self, keep_feasible: Vec<bool>) -> Self {
        self.keep_feasible = keep_feasible;
        self
    }
}

/// `scipy.optimize.minimize(method='trust-constr')`'s full result: the common
/// [`OptimizeResult`] (`maxcv` is `constr_violation`, `jac` the objective gradient) plus the
/// fields only trust-constr reports. Per-constraint lists follow the constraint order, with the
/// bounds last.
#[derive(Debug, Clone, PartialEq)]
pub struct TrustConstrResult {
    pub result: OptimizeResult,
    /// SciPy's status: 0 iteration limit, 1 `gtol`, 2 `xtol`, 3 callback, 4 constraint
    /// violation exceeds `gtol` at an otherwise successful stop.
    pub status: u8,
    pub method: TrustConstrMethod,
    /// ∞-norm of the Lagrangian gradient.
    pub optimality: f64,
    /// Largest constraint violation.
    pub constr_violation: f64,
    /// Objective gradient.
    pub grad: Vec<f64>,
    pub lagrangian_grad: Vec<f64>,
    /// Lagrange multipliers per constraint: a positive multiplier of an inequality means its
    /// upper bound is active, a negative one its lower bound.
    pub v: Vec<Vec<f64>>,
    /// Constraint values per constraint.
    pub constr: Vec<Vec<f64>>,
    /// Constraint Jacobians per constraint (rows).
    pub jac: Vec<Vec<Vec<f64>>>,
    pub constr_nfev: Vec<usize>,
    pub constr_njev: Vec<usize>,
    pub constr_nhev: Vec<usize>,
    pub niter: usize,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    /// Total conjugate-gradient iterations.
    pub cg_niter: usize,
    /// Why the last CG subproblem stopped: 0 not evaluated, 1 iteration limit, 2 trust-region
    /// boundary, 3 negative curvature, 4 tolerance.
    pub cg_stop_cond: u8,
    pub tr_radius: f64,
    pub constr_penalty: f64,
    /// `None` for `equality_constrained_sqp`.
    pub barrier_parameter: Option<f64>,
    /// `None` for `equality_constrained_sqp`.
    pub barrier_tolerance: Option<f64>,
}

// ══════════════════════════════════════════════════════════════════════
// Small dense helpers (numpy semantics where they matter)
// ══════════════════════════════════════════════════════════════════════

/// Dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// `rows` as a `rows.len() × cols` matrix; `None` when a row has the wrong length.
    fn from_rows(rows: &[Vec<f64>], cols: usize) -> Option<Self> {
        if rows.iter().any(|r| r.len() != cols) {
            return None;
        }
        Some(Self {
            rows: rows.len(),
            cols,
            data: rows.iter().flatten().copied().collect(),
        })
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] = value;
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.rows).map(|i| self.row(i).to_vec()).collect()
    }

    /// `A x`.
    fn dot(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows).map(|i| dot(self.row(i), x)).collect()
    }

    /// `Aᵀ y`.
    fn tdot(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.cols];
        for (i, &yi) in y.iter().enumerate() {
            for (o, &a) in out.iter_mut().zip(self.row(i)) {
                *o += a * yi;
            }
        }
        out
    }

    fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.at(i, j));
            }
        }
        t
    }

    /// The rows `idx`, negated when `negate`.
    fn select_rows(&self, idx: &[usize], negate: bool) -> Self {
        let sign = if negate { -1.0 } else { 1.0 };
        let mut data = Vec::with_capacity(idx.len() * self.cols);
        for &i in idx {
            data.extend(self.row(i).iter().map(|v| sign * v));
        }
        Self {
            rows: idx.len(),
            cols: self.cols,
            data,
        }
    }

    fn negated(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| -v).collect(),
        }
    }

    fn vstack(parts: &[Self], cols: usize) -> Self {
        let mut data = Vec::new();
        let mut rows = 0;
        for p in parts {
            data.extend_from_slice(&p.data);
            rows += p.rows;
        }
        Self { rows, cols, data }
    }

    fn fro_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `np.linalg.norm(x)`: `sqrt(x·x)`.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `np.linalg.norm(x, np.inf)` (NaN-propagating); 0 for an empty vector.
fn norm_inf(a: &[f64]) -> f64 {
    np_max(a.iter().map(|v| v.abs()))
}

/// `np.max` of a non-empty sequence: NaN-propagating.
fn np_max(values: impl Iterator<Item = f64>) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let mut any = false;
    for v in values {
        if v.is_nan() {
            return f64::NAN;
        }
        if !any || v > best {
            best = v;
        }
        any = true;
    }
    if any { best } else { 0.0 }
}

/// `np.maximum` of two scalars (NaN-propagating).
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `np.minimum` of two scalars (NaN-propagating).
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// Python's builtin `max(a, b)`: `a` unless `b > a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's builtin `min(a, b)`: `a` unless `b < a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// numpy's pairwise summation (`np.sum` of a float64 vector).
fn np_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= BLOCK {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// `np.array_equal`: equal lengths and `==` elementwise (so `-0.0 == 0.0`, `NaN != NaN`).
fn array_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn neg(a: &[f64]) -> Vec<f64> {
    a.iter().map(|v| -v).collect()
}

fn scaled(alpha: f64, a: &[f64]) -> Vec<f64> {
    a.iter().map(|v| alpha * v).collect()
}

fn invalid(detail: impl Into<String>) -> OptError {
    OptError::InvalidArgument {
        detail: detail.into(),
    }
}

fn check_finite(values: &[f64], mode: RuntimeMode, what: &str) -> Result<(), OptError> {
    if mode == RuntimeMode::Hardened && values.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: format!("hardened mode rejects non-finite {what}"),
        });
    }
    Ok(())
}

fn dense_from_user(rows: Vec<Vec<f64>>, r: usize, c: usize, what: &str) -> Result<Mat, OptError> {
    if rows.len() != r {
        return Err(invalid(format!(
            "{what} must have {r} rows of length {c}, got {} rows",
            rows.len()
        )));
    }
    Mat::from_rows(&rows, c)
        .ok_or_else(|| invalid(format!("{what} must have {r} rows of length {c}")))
}

/// Broadcast a per-component vector given with one entry or `m` entries (`np.broadcast_to`).
fn broadcast<T: Copy>(values: &[T], m: usize, what: &str) -> Result<Vec<T>, OptError> {
    match values.len() {
        1 => Ok(vec![values[0]; m]),
        len if len == m => Ok(values.to_vec()),
        len => Err(invalid(format!(
            "{what} has {len} entries; expected 1 or {m}"
        ))),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Finite differences: SciPy's approx_derivative(method='2-point')
// ══════════════════════════════════════════════════════════════════════

/// `_compute_absolute_step` followed by `_adjust_scheme_to_bounds(..., 1, '1-sided', lb, ub)`.
fn fd_steps(x0: &[f64], rel_step: Option<f64>, lb: &[f64], ub: &[f64]) -> Vec<f64> {
    let mut h: Vec<f64> = x0
        .iter()
        .map(|&x| {
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let default = FD_REL_STEP * sign * np_maximum(1.0, x.abs());
            match rel_step {
                None => default,
                Some(r) => {
                    let step = r * sign * x.abs();
                    if (x + step) - x == 0.0 { default } else { step }
                }
            }
        })
        .collect();
    if lb.iter().all(|v| *v == f64::NEG_INFINITY) && ub.iter().all(|v| *v == f64::INFINITY) {
        return h;
    }
    for i in 0..h.len() {
        let lower = x0[i] - lb[i];
        let upper = ub[i] - x0[i];
        let trial = x0[i] + h[i];
        let violated = trial < lb[i] || trial > ub[i];
        let fitting = h[i].abs() <= np_maximum(lower, upper);
        if violated && fitting {
            h[i] = -h[i];
        }
        if !fitting {
            if upper >= lower {
                h[i] = upper;
            } else if upper < lower {
                h[i] = -lower;
            }
        }
    }
    h
}

// ══════════════════════════════════════════════════════════════════════
// The objective: SciPy's ScalarFunction
// ══════════════════════════════════════════════════════════════════════

enum ObjHessKind {
    Bfgs(BfgsHessian),
    Dense(HessFunc),
    Hessp(HesspFunc),
}

/// What `objective.hess(x)` returned: the live BFGS approximation, or a snapshot.
#[derive(Debug, Clone)]
enum ObjHess {
    Bfgs,
    Dense(Mat),
    /// `HessianLinearOperator(hessp)(x)`: products `hessp(x, p)` at the captured `x`.
    Hessp(Vec<f64>),
}

struct ScalarFunction<'f, F> {
    fun: &'f F,
    grad: Option<GradientFunc>,
    rel_step: Option<f64>,
    fd_lb: Vec<f64>,
    fd_ub: Vec<f64>,
    hess: ObjHessKind,
    h_dense: Mat,
    h_x: Vec<f64>,
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
    f_updated: bool,
    g_updated: bool,
    h_updated: bool,
    x_prev: Vec<f64>,
    g_prev: Vec<f64>,
    nfev: usize,
    ngev: usize,
    nhev: usize,
    mode: RuntimeMode,
}

impl<'f, F> ScalarFunction<'f, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn new(
        fun: &'f F,
        x0: &[f64],
        grad: Option<GradientFunc>,
        hess: ObjHessKind,
        rel_step: Option<f64>,
        fd_bounds: (&[f64], &[f64]),
        mode: RuntimeMode,
    ) -> Result<Self, OptError> {
        let n = x0.len();
        let mut sf = Self {
            fun,
            grad,
            rel_step,
            fd_lb: fd_bounds.0.to_vec(),
            fd_ub: fd_bounds.1.to_vec(),
            hess,
            h_dense: Mat::zeros(n, n),
            h_x: x0.to_vec(),
            x: x0.to_vec(),
            f: f64::NAN,
            g: vec![f64::NAN; n],
            f_updated: false,
            g_updated: false,
            h_updated: false,
            x_prev: Vec::new(),
            g_prev: Vec::new(),
            nfev: 0,
            ngev: 0,
            nhev: 0,
            mode,
        };
        sf.update_fun()?;
        sf.update_grad()?;
        if let ObjHessKind::Bfgs(bfgs) = &mut sf.hess {
            bfgs.initialize(n, HessianApproxType::Hess);
        } else if let ObjHessKind::Dense(hess) = sf.hess {
            let h = hess(x0);
            sf.nhev += 1;
            sf.h_dense = dense_from_user(h, n, n, "hess")?;
            check_finite(&sf.h_dense.data, mode, "Hessian values")?;
        } else {
            sf.nhev += 1;
        }
        sf.h_updated = true;
        Ok(sf)
    }

    fn call_fun(&mut self, x: &[f64]) -> Result<f64, OptError> {
        self.nfev += 1;
        let value = (self.fun)(x);
        if !value.is_finite() && self.mode == RuntimeMode::Hardened {
            return Err(OptError::NonFiniteInput {
                detail: String::from("hardened mode rejects non-finite objective values"),
            });
        }
        Ok(value)
    }

    fn update_fun(&mut self) -> Result<(), OptError> {
        if !self.f_updated {
            let x = self.x.clone();
            self.f = self.call_fun(&x)?;
            self.f_updated = true;
        }
        Ok(())
    }

    fn update_grad(&mut self) -> Result<(), OptError> {
        if !self.g_updated {
            let n = self.x.len();
            if let Some(grad) = self.grad {
                let g = grad(&self.x);
                if g.len() != n {
                    return Err(invalid(format!(
                        "gradient returned {} values for {n} variables",
                        g.len()
                    )));
                }
                check_finite(&g, self.mode, "gradient values")?;
                self.g = g;
            } else {
                self.update_fun()?;
                let x = self.x.clone();
                let f0 = self.f;
                let h = fd_steps(&x, self.rel_step, &self.fd_lb, &self.fd_ub);
                let mut g = vec![0.0; n];
                let mut x1 = x.clone();
                for i in 0..n {
                    x1[i] = x[i] + h[i];
                    let fi = self.call_fun(&x1)?;
                    x1[i] = x[i];
                    let dx = (x[i] + h[i]) - x[i];
                    g[i] = (fi - f0) / dx;
                }
                self.g = g;
            }
            self.ngev += 1;
            self.g_updated = true;
        }
        Ok(())
    }

    fn update_hess(&mut self) -> Result<(), OptError> {
        if !self.h_updated {
            if matches!(self.hess, ObjHessKind::Bfgs(_)) {
                self.update_grad()?;
                let delta_x = sub(&self.x, &self.x_prev);
                let delta_g = sub(&self.g, &self.g_prev);
                if let ObjHessKind::Bfgs(bfgs) = &mut self.hess {
                    bfgs.update(&delta_x, &delta_g);
                }
            } else if let ObjHessKind::Dense(hess) = self.hess {
                let n = self.x.len();
                let h = hess(&self.x);
                self.nhev += 1;
                self.h_dense = dense_from_user(h, n, n, "hess")?;
                check_finite(&self.h_dense.data, self.mode, "Hessian values")?;
            } else {
                self.nhev += 1;
                self.h_x.clone_from(&self.x);
            }
            self.h_updated = true;
        }
        Ok(())
    }

    fn update_x(&mut self, x: &[f64]) -> Result<(), OptError> {
        if matches!(self.hess, ObjHessKind::Bfgs(_)) {
            self.update_grad()?;
            self.x_prev = std::mem::replace(&mut self.x, x.to_vec());
            self.g_prev.clone_from(&self.g);
            self.f_updated = false;
            self.g_updated = false;
            self.h_updated = false;
            self.update_hess()?;
        } else {
            self.x = x.to_vec();
            self.f_updated = false;
            self.g_updated = false;
            self.h_updated = false;
        }
        Ok(())
    }

    fn fun(&mut self, x: &[f64]) -> Result<f64, OptError> {
        if !array_equal(x, &self.x) {
            self.update_x(x)?;
        }
        self.update_fun()?;
        Ok(self.f)
    }

    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        if !array_equal(x, &self.x) {
            self.update_x(x)?;
        }
        self.update_grad()?;
        Ok(self.g.clone())
    }

    fn hess(&mut self, x: &[f64]) -> Result<ObjHess, OptError> {
        if !array_equal(x, &self.x) {
            self.update_x(x)?;
        }
        self.update_hess()?;
        Ok(match self.hess {
            ObjHessKind::Bfgs(_) => ObjHess::Bfgs,
            ObjHessKind::Dense(_) => ObjHess::Dense(self.h_dense.clone()),
            ObjHessKind::Hessp(_) => ObjHess::Hessp(self.h_x.clone()),
        })
    }

    fn hess_dot(&self, h: &ObjHess, p: &[f64]) -> Result<Vec<f64>, OptError> {
        match (h, &self.hess) {
            (ObjHess::Bfgs, ObjHessKind::Bfgs(bfgs)) => Ok(bfgs.dot(p)),
            (ObjHess::Dense(m), _) => Ok(m.dot(p)),
            (ObjHess::Hessp(x), ObjHessKind::Hessp(hessp)) => {
                let hp = hessp(x, p);
                if hp.len() != p.len() {
                    return Err(invalid(format!(
                        "hessp returned {} values for {} variables",
                        hp.len(),
                        p.len()
                    )));
                }
                check_finite(&hp, self.mode, "Hessian-vector products")?;
                Ok(hp)
            }
            _ => Err(invalid("inconsistent objective Hessian state")),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Constraint functions: SciPy's VectorFunction / LinearVectorFunction
// ══════════════════════════════════════════════════════════════════════

enum VecHessKind<'a> {
    Bfgs(BfgsHessian),
    Callable(&'a TrustConstraintHessFn<'a>),
}

/// SciPy's `VectorFunction` for a `NonlinearConstraint`.
struct VectorFunction<'a> {
    fun: &'a TrustConstraintFn<'a>,
    jac: Option<&'a TrustConstraintJacFn<'a>>,
    hess: VecHessKind<'a>,
    h_dense: Mat,
    rel_step: Option<f64>,
    fd_lb: Vec<f64>,
    fd_ub: Vec<f64>,
    x: Vec<f64>,
    f: Vec<f64>,
    j: Mat,
    v: Vec<f64>,
    m: usize,
    f_updated: bool,
    j_updated: bool,
    h_updated: bool,
    x_prev: Option<Vec<f64>>,
    j_prev: Option<Mat>,
    nfev: usize,
    njev: usize,
    nhev: usize,
    mode: RuntimeMode,
}

impl<'a> VectorFunction<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fun: &'a TrustConstraintFn<'a>,
        jac: Option<&'a TrustConstraintJacFn<'a>>,
        hess: Option<&'a TrustConstraintHessFn<'a>>,
        rel_step: Option<f64>,
        x0: &[f64],
        fd_bounds: (&[f64], &[f64]),
        mode: RuntimeMode,
    ) -> Result<Self, OptError> {
        let n = x0.len();
        let f0 = fun(x0);
        check_finite(&f0, mode, "constraint values")?;
        let m = f0.len();
        if m == 0 {
            return Err(invalid("a constraint must have at least one component"));
        }
        let mut vf = Self {
            fun,
            jac,
            hess: match hess {
                Some(h) => VecHessKind::Callable(h),
                None => {
                    VecHessKind::Bfgs(BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None))
                }
            },
            h_dense: Mat::zeros(n, n),
            rel_step,
            fd_lb: fd_bounds.0.to_vec(),
            fd_ub: fd_bounds.1.to_vec(),
            x: x0.to_vec(),
            f: f0,
            j: Mat::zeros(m, n),
            v: vec![0.0; m],
            m,
            f_updated: true,
            j_updated: false,
            h_updated: false,
            x_prev: None,
            j_prev: None,
            nfev: 1,
            njev: 0,
            nhev: 0,
            mode,
        };
        if vf.jac.is_some() {
            vf.njev += 1;
        }
        vf.j = vf.eval_jac()?;
        vf.j_updated = true;
        if let VecHessKind::Bfgs(bfgs) = &mut vf.hess {
            bfgs.initialize(n, HessianApproxType::Hess);
        } else {
            vf.eval_hess()?;
        }
        vf.h_updated = true;
        Ok(vf)
    }

    fn call_fun(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        self.nfev += 1;
        let y = (self.fun)(x);
        if y.len() != self.m {
            return Err(invalid(format!(
                "constraint returned {} values, {} at x0",
                y.len(),
                self.m
            )));
        }
        check_finite(&y, self.mode, "constraint values")?;
        Ok(y)
    }

    /// The Jacobian at `self.x`: the user's, or `'2-point'` differences about `self.f`.
    fn eval_jac(&mut self) -> Result<Mat, OptError> {
        let n = self.x.len();
        if let Some(jac) = self.jac {
            let j = dense_from_user(jac(&self.x), self.m, n, "constraint jac")?;
            check_finite(&j.data, self.mode, "constraint Jacobian values")?;
            return Ok(j);
        }
        let x = self.x.clone();
        let f0 = self.f.clone();
        let h = fd_steps(&x, self.rel_step, &self.fd_lb, &self.fd_ub);
        let mut j = Mat::zeros(self.m, n);
        let mut x1 = x.clone();
        for i in 0..n {
            x1[i] = x[i] + h[i];
            let fi = self.call_fun(&x1)?;
            x1[i] = x[i];
            let dx = (x[i] + h[i]) - x[i];
            for r in 0..self.m {
                j.set(r, i, (fi[r] - f0[r]) / dx);
            }
        }
        Ok(j)
    }

    fn eval_hess(&mut self) -> Result<(), OptError> {
        if let VecHessKind::Callable(hess) = self.hess {
            let n = self.x.len();
            let h = hess(&self.x, &self.v);
            self.nhev += 1;
            self.h_dense = dense_from_user(h, n, n, "constraint hess")?;
            check_finite(&self.h_dense.data, self.mode, "constraint Hessian values")?;
        }
        Ok(())
    }

    fn update_fun(&mut self) -> Result<(), OptError> {
        if !self.f_updated {
            let x = self.x.clone();
            self.f = self.call_fun(&x)?;
            self.f_updated = true;
        }
        Ok(())
    }

    fn update_jac(&mut self) -> Result<(), OptError> {
        if !self.j_updated {
            if self.jac.is_none() {
                self.update_fun()?;
            } else {
                self.njev += 1;
            }
            self.j = self.eval_jac()?;
            self.j_updated = true;
        }
        Ok(())
    }

    fn update_hess(&mut self) -> Result<(), OptError> {
        if !self.h_updated {
            if matches!(self.hess, VecHessKind::Callable(_)) {
                self.eval_hess()?;
            } else {
                self.update_jac()?;
                if let (Some(x_prev), Some(j_prev)) = (&self.x_prev, &self.j_prev) {
                    let delta_x = sub(&self.x, x_prev);
                    let delta_g = sub(&self.j.tdot(&self.v), &j_prev.tdot(&self.v));
                    if let VecHessKind::Bfgs(bfgs) = &mut self.hess {
                        bfgs.update(&delta_x, &delta_g);
                    }
                }
            }
            self.h_updated = true;
        }
        Ok(())
    }

    fn update_x(&mut self, x: &[f64]) -> Result<(), OptError> {
        if !array_equal(x, &self.x) {
            if matches!(self.hess, VecHessKind::Bfgs(_)) {
                self.update_jac()?;
                self.x_prev = Some(std::mem::replace(&mut self.x, x.to_vec()));
                self.j_prev = Some(self.j.clone());
                self.f_updated = false;
                self.j_updated = false;
                self.h_updated = false;
                self.update_hess()?;
            } else {
                self.x = x.to_vec();
                self.f_updated = false;
                self.j_updated = false;
                self.h_updated = false;
            }
        }
        Ok(())
    }

    fn fun(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        self.update_x(x)?;
        self.update_fun()?;
        Ok(self.f.clone())
    }

    fn jac(&mut self, x: &[f64]) -> Result<Mat, OptError> {
        self.update_x(x)?;
        self.update_jac()?;
        Ok(self.j.clone())
    }

    fn hess(&mut self, x: &[f64], v: Vec<f64>) -> Result<(), OptError> {
        if !array_equal(&v, &self.v) {
            self.v = v;
            self.h_updated = false;
        }
        self.update_x(x)?;
        self.update_hess()
    }
}

/// SciPy's `LinearVectorFunction` (`A` given) and `IdentityVectorFunction` (`A = None`, the
/// bounds).
struct LinearFunction {
    a: Option<Mat>,
    x: Vec<f64>,
    f: Vec<f64>,
    f_updated: bool,
    v: Vec<f64>,
    j: Mat,
}

impl LinearFunction {
    fn new(a: Option<Mat>, x0: &[f64]) -> Self {
        let j = a.clone().unwrap_or_else(|| Mat::identity(x0.len()));
        let f = match &a {
            Some(a) => a.dot(x0),
            None => x0.to_vec(),
        };
        let m = f.len();
        Self {
            a,
            x: x0.to_vec(),
            f,
            f_updated: true,
            v: vec![0.0; m],
            j,
        }
    }

    fn update_x(&mut self, x: &[f64]) {
        if !array_equal(x, &self.x) {
            self.x = x.to_vec();
            self.f_updated = false;
        }
    }

    fn fun(&mut self, x: &[f64]) -> Vec<f64> {
        self.update_x(x);
        if !self.f_updated {
            self.f = match &self.a {
                Some(a) => a.dot(x),
                None => x.to_vec(),
            };
            self.f_updated = true;
        }
        self.f.clone()
    }
}

enum ConstraintFunction<'a> {
    Vector(Box<VectorFunction<'a>>),
    Linear(LinearFunction),
}

/// What a constraint's `hess(x, v)` returned.
#[derive(Debug, Clone)]
enum ConHess {
    Zero,
    /// The live BFGS approximation of constraint `idx`.
    Bfgs(usize),
    Dense(Mat),
}

// ══════════════════════════════════════════════════════════════════════
// PreparedConstraint and the canonical form
// ══════════════════════════════════════════════════════════════════════

/// How `CanonicalConstraint.from_PreparedConstraint` splits `lb <= f <= ub` into
/// `c_eq = 0`, `c_ineq <= 0`.
#[derive(Debug, Clone)]
enum CanonKind {
    Empty,
    Equal,
    Less {
        idx: Vec<usize>,
        all: bool,
    },
    Greater {
        idx: Vec<usize>,
        all: bool,
    },
    Interval {
        equal: Vec<usize>,
        less: Vec<usize>,
        greater: Vec<usize>,
        interval: Vec<usize>,
    },
}

impl CanonKind {
    fn classify(lb: &[f64], ub: &[f64]) -> Self {
        let m = lb.len();
        let all_lb_inf = lb.iter().all(|v| *v == f64::NEG_INFINITY);
        let all_ub_inf = ub.iter().all(|v| *v == f64::INFINITY);
        if all_lb_inf && all_ub_inf {
            Self::Empty
        } else if lb.iter().zip(ub).all(|(l, u)| l == u) {
            Self::Equal
        } else if all_lb_inf {
            let idx: Vec<usize> = (0..m).filter(|&i| ub[i] < f64::INFINITY).collect();
            let all = idx.len() == m;
            Self::Less { idx, all }
        } else if all_ub_inf {
            let idx: Vec<usize> = (0..m).filter(|&i| lb[i] > f64::NEG_INFINITY).collect();
            let all = idx.len() == m;
            Self::Greater { idx, all }
        } else {
            let lb_inf = |i: usize| lb[i] == f64::NEG_INFINITY;
            let ub_inf = |i: usize| ub[i] == f64::INFINITY;
            let equal = (0..m).filter(|&i| lb[i] == ub[i]).collect();
            let less = (0..m).filter(|&i| lb_inf(i) && !ub_inf(i)).collect();
            let greater = (0..m).filter(|&i| ub_inf(i) && !lb_inf(i)).collect();
            let interval = (0..m)
                .filter(|&i| lb[i] != ub[i] && !lb_inf(i) && !ub_inf(i))
                .collect();
            Self::Interval {
                equal,
                less,
                greater,
                interval,
            }
        }
    }

    fn sizes(&self, m: usize) -> (usize, usize) {
        match self {
            Self::Empty => (0, 0),
            Self::Equal => (m, 0),
            Self::Less { idx, .. } | Self::Greater { idx, .. } => (0, idx.len()),
            Self::Interval {
                equal,
                less,
                greater,
                interval,
            } => (equal.len(), less.len() + greater.len() + 2 * interval.len()),
        }
    }

    fn keep_feasible(&self, kf: &[bool]) -> Vec<bool> {
        match self {
            Self::Empty | Self::Equal => Vec::new(),
            Self::Less { idx, all } | Self::Greater { idx, all } => {
                if *all {
                    kf.to_vec()
                } else {
                    idx.iter().map(|&i| kf[i]).collect()
                }
            }
            Self::Interval {
                less,
                greater,
                interval,
                ..
            } => less
                .iter()
                .chain(greater)
                .chain(interval)
                .chain(interval)
                .map(|&i| kf[i])
                .collect(),
        }
    }

    /// `(c_eq, c_ineq)` from the constraint values `f`.
    fn map_fun(&self, f: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
        match self {
            Self::Empty => (Vec::new(), Vec::new()),
            Self::Equal => (sub(f, lb), Vec::new()),
            Self::Less { idx, .. } => (Vec::new(), idx.iter().map(|&i| f[i] - ub[i]).collect()),
            Self::Greater { idx, .. } => (Vec::new(), idx.iter().map(|&i| lb[i] - f[i]).collect()),
            Self::Interval {
                equal,
                less,
                greater,
                interval,
            } => {
                let eq = equal.iter().map(|&i| f[i] - lb[i]).collect();
                let ineq = less
                    .iter()
                    .map(|&i| f[i] - ub[i])
                    .chain(greater.iter().map(|&i| lb[i] - f[i]))
                    .chain(interval.iter().map(|&i| f[i] - ub[i]))
                    .chain(interval.iter().map(|&i| lb[i] - f[i]))
                    .collect();
                (eq, ineq)
            }
        }
    }

    /// `(J_eq, J_ineq)` from the constraint Jacobian `j`.
    fn map_jac(&self, j: &Mat) -> (Mat, Mat) {
        let n = j.cols;
        match self {
            Self::Empty => (Mat::zeros(0, n), Mat::zeros(0, n)),
            Self::Equal => (j.clone(), Mat::zeros(0, n)),
            Self::Less { idx, all } => (
                Mat::zeros(0, n),
                if *all {
                    j.clone()
                } else {
                    j.select_rows(idx, false)
                },
            ),
            Self::Greater { idx, all } => (
                Mat::zeros(0, n),
                if *all {
                    j.negated()
                } else {
                    j.select_rows(idx, true)
                },
            ),
            Self::Interval {
                equal,
                less,
                greater,
                interval,
            } => {
                let eq = j.select_rows(equal, false);
                let ineq = Mat::vstack(
                    &[
                        j.select_rows(less, false),
                        j.select_rows(greater, true),
                        j.select_rows(interval, false),
                        j.select_rows(interval, true),
                    ],
                    n,
                );
                (eq, ineq)
            }
        }
    }

    /// The multipliers of the constraint's components from those of its canonical rows.
    fn map_multipliers(&self, v_eq: &[f64], v_ineq: &[f64], m: usize) -> Vec<f64> {
        match self {
            Self::Empty => vec![0.0; m],
            Self::Equal => v_eq.to_vec(),
            Self::Less { idx, all } => {
                if *all {
                    v_ineq.to_vec()
                } else {
                    let mut v = vec![0.0; m];
                    for (k, &i) in idx.iter().enumerate() {
                        v[i] = v_ineq[k];
                    }
                    v
                }
            }
            Self::Greater { idx, all } => {
                if *all {
                    neg(v_ineq)
                } else {
                    let mut v = vec![0.0; m];
                    for (k, &i) in idx.iter().enumerate() {
                        v[i] = -v_ineq[k];
                    }
                    v
                }
            }
            Self::Interval {
                equal,
                less,
                greater,
                interval,
            } => {
                let (nl, ng, ni) = (less.len(), greater.len(), interval.len());
                let v_l = &v_ineq[..nl];
                let v_g = &v_ineq[nl..nl + ng];
                let v_il = &v_ineq[nl + ng..nl + ng + ni];
                let v_ig = &v_ineq[nl + ng + ni..nl + ng + 2 * ni];
                let mut v = vec![0.0; m];
                for (k, &i) in equal.iter().enumerate() {
                    v[i] = v_eq[k];
                }
                for (k, &i) in less.iter().enumerate() {
                    v[i] = v_l[k];
                }
                for (k, &i) in greater.iter().enumerate() {
                    v[i] = -v_g[k];
                }
                for (k, &i) in interval.iter().enumerate() {
                    v[i] = v_il[k] - v_ig[k];
                }
                v
            }
        }
    }
}

/// SciPy's `PreparedConstraint` with its `CanonicalConstraint`.
struct Prepared<'a> {
    fun: ConstraintFunction<'a>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    keep_feasible: Vec<bool>,
    canon: CanonKind,
    m: usize,
}

impl Prepared<'_> {
    fn values(&self) -> &[f64] {
        match &self.fun {
            ConstraintFunction::Vector(vf) => &vf.f,
            ConstraintFunction::Linear(lf) => &lf.f,
        }
    }

    fn jacobian(&self) -> &Mat {
        match &self.fun {
            ConstraintFunction::Vector(vf) => &vf.j,
            ConstraintFunction::Linear(lf) => &lf.j,
        }
    }

    fn multipliers(&self) -> &[f64] {
        match &self.fun {
            ConstraintFunction::Vector(vf) => &vf.v,
            ConstraintFunction::Linear(lf) => &lf.v,
        }
    }

    fn counts(&self) -> (usize, usize, usize) {
        match &self.fun {
            ConstraintFunction::Vector(vf) => (vf.nfev, vf.njev, vf.nhev),
            ConstraintFunction::Linear(_) => (0, 0, 0),
        }
    }

    fn eval(&mut self, x: &[f64]) -> Result<Vec<f64>, OptError> {
        match &mut self.fun {
            ConstraintFunction::Vector(vf) => vf.fun(x),
            ConstraintFunction::Linear(lf) => Ok(lf.fun(x)),
        }
    }

    fn eval_jac(&mut self, x: &[f64]) -> Result<Mat, OptError> {
        match &mut self.fun {
            ConstraintFunction::Vector(vf) => vf.jac(x),
            ConstraintFunction::Linear(lf) => {
                lf.update_x(x);
                Ok(lf.j.clone())
            }
        }
    }

    fn eval_hess(&mut self, x: &[f64], v: Vec<f64>, idx: usize) -> Result<ConHess, OptError> {
        match &mut self.fun {
            ConstraintFunction::Vector(vf) => {
                vf.hess(x, v)?;
                Ok(match &vf.hess {
                    VecHessKind::Bfgs(_) => ConHess::Bfgs(idx),
                    VecHessKind::Callable(_) => ConHess::Dense(vf.h_dense.clone()),
                })
            }
            ConstraintFunction::Linear(lf) => {
                lf.update_x(x);
                lf.v = v;
                Ok(ConHess::Zero)
            }
        }
    }
}

/// `LagrangianHessian(x, v_eq, v_ineq)`: the objective's Hessian plus the constraints'.
#[derive(Debug, Clone)]
struct LagrHess {
    obj: ObjHess,
    cons: Vec<ConHess>,
}

// ══════════════════════════════════════════════════════════════════════
// Projections (projections.py)
// ══════════════════════════════════════════════════════════════════════

/// LU factorisation with partial pivoting of a square matrix.
struct Lu {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl Lu {
    /// `None` when a pivot is exactly zero (SuperLU's "Factor is exactly singular").
    fn factor(k: &Mat) -> Option<Self> {
        let n = k.rows;
        let mut lu = k.data.clone();
        let mut piv: Vec<usize> = (0..n).collect();
        for col in 0..n {
            let mut p = col;
            let mut best = lu[col * n + col].abs();
            for r in col + 1..n {
                let v = lu[r * n + col].abs();
                if v > best {
                    best = v;
                    p = r;
                }
            }
            if best == 0.0 || best.is_nan() {
                return None;
            }
            if p != col {
                for j in 0..n {
                    lu.swap(p * n + j, col * n + j);
                }
                piv.swap(p, col);
            }
            let d = lu[col * n + col];
            for r in col + 1..n {
                let l = lu[r * n + col] / d;
                lu[r * n + col] = l;
                if l != 0.0 {
                    for j in col + 1..n {
                        lu[r * n + j] -= l * lu[col * n + j];
                    }
                }
            }
        }
        Some(Self { n, lu, piv })
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            let mut s = y[i];
            for j in 0..i {
                s -= self.lu[i * n + j] * y[j];
            }
            y[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in i + 1..n {
                s -= self.lu[i * n + j] * y[j];
            }
            y[i] = s / self.lu[i * n + i];
        }
        y
    }
}

/// LAPACK `dlapy2`: `sqrt(x² + y²)` without destructive overflow.
fn lapy2(x: f64, y: f64) -> f64 {
    let xa = x.abs();
    let ya = y.abs();
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// `scipy.linalg.qr(at, pivoting=True, mode='economic')` (LAPACK `dgeqp3` unblocked, then
/// `dorgqr`): `(Q, R, P)` with `at[:, P] = Q R`.
fn qr_pivoted(at: &Mat) -> (Mat, Mat, Vec<usize>) {
    let (rows, cols) = (at.rows, at.cols);
    let k = rows.min(cols);
    let mut a = at.clone();
    let mut jpvt: Vec<usize> = (0..cols).collect();
    let mut tau = vec![0.0; k];
    let col_norm = |a: &Mat, j: usize, from: usize| -> f64 {
        (from..rows)
            .map(|i| a.at(i, j) * a.at(i, j))
            .sum::<f64>()
            .sqrt()
    };
    let mut vn1: Vec<f64> = (0..cols).map(|j| col_norm(&a, j, 0)).collect();
    let mut vn2 = vn1.clone();
    let tol3z = (f64::EPSILON / 2.0).sqrt();
    for i in 0..k {
        // Pivot: the remaining column of largest norm (first on ties, as idamax).
        let mut pvt = i;
        for j in i + 1..cols {
            if vn1[j] > vn1[pvt] {
                pvt = j;
            }
        }
        if pvt != i {
            for r in 0..rows {
                let tmp = a.at(r, pvt);
                a.set(r, pvt, a.at(r, i));
                a.set(r, i, tmp);
            }
            jpvt.swap(pvt, i);
            vn1[pvt] = vn1[i];
            vn2[pvt] = vn2[i];
        }
        // dlarfg on a[i.., i].
        if i + 1 < rows {
            let alpha = a.at(i, i);
            let xnorm = col_norm(&a, i, i + 1);
            if xnorm != 0.0 {
                let beta = -lapy2(alpha, xnorm).copysign(alpha);
                tau[i] = (beta - alpha) / beta;
                let scal = 1.0 / (alpha - beta);
                for r in i + 1..rows {
                    a.set(r, i, a.at(r, i) * scal);
                }
                a.set(i, i, beta);
            }
        }
        // dlarf: apply H(i) to a[i.., i+1..] from the left.
        if i + 1 < cols && tau[i] != 0.0 {
            for j in i + 1..cols {
                let mut w = a.at(i, j);
                for r in i + 1..rows {
                    w += a.at(r, i) * a.at(r, j);
                }
                let tw = tau[i] * w;
                a.set(i, j, a.at(i, j) - tw);
                for r in i + 1..rows {
                    a.set(r, j, a.at(r, j) - a.at(r, i) * tw);
                }
            }
        }
        // Update the partial column norms.
        for j in i + 1..cols {
            if vn1[j] != 0.0 {
                let ratio = a.at(i, j).abs() / vn1[j];
                let temp = (1.0 - ratio * ratio).max(0.0);
                let ratio2 = vn1[j] / vn2[j];
                let temp2 = temp * ratio2 * ratio2;
                if temp2 <= tol3z {
                    if i + 1 < rows {
                        vn1[j] = col_norm(&a, j, i + 1);
                        vn2[j] = vn1[j];
                    } else {
                        vn1[j] = 0.0;
                        vn2[j] = 0.0;
                    }
                } else {
                    vn1[j] *= temp.sqrt();
                }
            }
        }
    }
    let mut r = Mat::zeros(k, cols);
    for i in 0..k {
        for j in i..cols {
            r.set(i, j, a.at(i, j));
        }
    }
    // dorg2r: Q = H(0) H(1) ... H(k-1) applied to the first k columns of the identity.
    let mut q = Mat::zeros(rows, k);
    for i in (0..k).rev() {
        if i + 1 < k {
            q.set(i, i, 1.0);
            for j in i + 1..k {
                let mut w = q.at(i, j);
                for r2 in i + 1..rows {
                    w += a.at(r2, i) * q.at(r2, j);
                }
                let tw = tau[i] * w;
                q.set(i, j, q.at(i, j) - tw);
                for r2 in i + 1..rows {
                    q.set(r2, j, q.at(r2, j) - a.at(r2, i) * tw);
                }
            }
        }
        for r2 in i + 1..rows {
            q.set(r2, i, -tau[i] * a.at(r2, i));
        }
        q.set(i, i, 1.0 - tau[i]);
        for r2 in 0..i {
            q.set(r2, i, 0.0);
        }
    }
    (q, r, jpvt)
}

/// One-sided Jacobi SVD of `w` (`rows >= cols`): `w = U diag(s) Vᵀ`, `s` descending.
fn jacobi_svd(w: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (rows, cols) = (w.rows, w.cols);
    let mut u = w.clone();
    let mut v = Mat::identity(cols);
    for _sweep in 0..80 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..rows {
                    let (up, uq) = (u.at(i, p), u.at(i, q));
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..rows {
                    let (up, uq) = (u.at(i, p), u.at(i, q));
                    u.set(i, p, c * up - s * uq);
                    u.set(i, q, s * up + c * uq);
                }
                for i in 0..cols {
                    let (vp, vq) = (v.at(i, p), v.at(i, q));
                    v.set(i, p, c * vp - s * vq);
                    v.set(i, q, s * vp + c * vq);
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| u.at(i, j) * u.at(i, j))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));
    let mut u_out = Mat::zeros(rows, cols);
    let mut v_out = Mat::zeros(cols, cols);
    let mut s_out = Vec::with_capacity(cols);
    for (k, &j) in order.iter().enumerate() {
        let sj = sigma[j];
        s_out.push(sj);
        for i in 0..rows {
            u_out.set(i, k, if sj > 0.0 { u.at(i, j) / sj } else { 0.0 });
        }
        for i in 0..cols {
            v_out.set(i, k, v.at(i, j));
        }
    }
    (u_out, s_out, v_out)
}

/// `scipy.linalg.svd(a, full_matrices=False)`: `(U, s, Vt)`.
fn svd_thin(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    if a.rows >= a.cols {
        let (u, s, v) = jacobi_svd(a);
        (u, s, v.transpose())
    } else {
        // aᵀ = U' S V'ᵀ, so a = V' S U'ᵀ.
        let (u, s, v) = jacobi_svd(&a.transpose());
        (v, s, u.transpose())
    }
}

/// `orthogonality(A, g)`: `‖A g‖ / (‖A‖_F ‖g‖)`, 0 when either norm is 0.
fn orthogonality(a: &Mat, g: &[f64]) -> f64 {
    let norm_g = norm(g);
    let norm_a = a.fro_norm();
    if norm_g == 0.0 || norm_a == 0.0 {
        return 0.0;
    }
    norm(&a.dot(g)) / (norm_a * norm_g)
}

/// SciPy's `(Z, LS, Y)` operators of a constraint Jacobian `A` (`m × n`).
enum Projections {
    Qr {
        a: Mat,
        q: Mat,
        r: Mat,
        perm: Vec<usize>,
    },
    Svd {
        a: Mat,
        u: Mat,
        s: Vec<f64>,
        vt: Mat,
    },
    Augmented {
        a: Mat,
        k: Mat,
        lu: Lu,
    },
}

/// Back substitution with the upper triangular `r` (`solve_triangular(R, b)`), or with `rᵀ`
/// (`trans='T'`).
fn solve_upper(r: &Mat, b: &[f64], transpose: bool) -> Vec<f64> {
    let k = r.rows;
    let mut x = b.to_vec();
    if transpose {
        for i in 0..k {
            let mut s = x[i];
            for j in 0..i {
                s -= r.at(j, i) * x[j];
            }
            x[i] = s / r.at(i, i);
        }
    } else {
        for i in (0..k).rev() {
            let mut s = x[i];
            for j in i + 1..k {
                s -= r.at(i, j) * x[j];
            }
            x[i] = s / r.at(i, i);
        }
    }
    x
}

impl Projections {
    fn new(a: &Mat, sparse: bool, method: Option<FactorizationMethod>) -> Result<Self, OptError> {
        let sparse = sparse || a.rows * a.cols == 0;
        let method = if sparse {
            match method {
                None
                | Some(
                    FactorizationMethod::AugmentedSystem | FactorizationMethod::NormalEquation,
                ) => FactorizationMethod::AugmentedSystem,
                Some(_) => return Err(invalid("Method not allowed for sparse array.")),
            }
        } else {
            match method {
                None | Some(FactorizationMethod::QrFactorization) => {
                    FactorizationMethod::QrFactorization
                }
                Some(FactorizationMethod::SvdFactorization) => {
                    FactorizationMethod::SvdFactorization
                }
                Some(_) => return Err(invalid("Method not allowed for dense array.")),
            }
        };
        match method {
            FactorizationMethod::AugmentedSystem => Ok(Self::augmented(a)),
            FactorizationMethod::QrFactorization => Self::qr(a),
            _ => Ok(Self::svd(a)),
        }
    }

    fn augmented(a: &Mat) -> Self {
        let (m, n) = (a.rows, a.cols);
        let mut k = Mat::zeros(n + m, n + m);
        for i in 0..n {
            k.set(i, i, 1.0);
        }
        for i in 0..m {
            for j in 0..n {
                k.set(n + i, j, a.at(i, j));
                k.set(j, n + i, a.at(i, j));
            }
        }
        match Lu::factor(&k) {
            Some(lu) => Self::Augmented {
                a: a.clone(),
                k,
                lu,
            },
            // SciPy warns "Singular Jacobian matrix. Using dense SVD decomposition".
            None => Self::svd(a),
        }
    }

    fn qr(a: &Mat) -> Result<Self, OptError> {
        let (q, r, perm) = qr_pivoted(&a.transpose());
        let last = r.rows.saturating_sub(1);
        if r.rows > 0 && norm_inf(r.row(last)) < PROJECTION_TOL {
            // SciPy warns "Singular Jacobian matrix. Using SVD decomposition".
            return Ok(Self::svd(a));
        }
        if r.rows != r.cols {
            return Err(invalid(EXPECTED_SQUARE_MATRIX));
        }
        Ok(Self::Qr {
            a: a.clone(),
            q,
            r,
            perm,
        })
    }

    fn svd(a: &Mat) -> Self {
        let (u, s, vt) = svd_thin(a);
        let keep: Vec<usize> = (0..s.len()).filter(|&i| s[i] > PROJECTION_TOL).collect();
        let mut u_k = Mat::zeros(u.rows, keep.len());
        for (c, &i) in keep.iter().enumerate() {
            for r in 0..u.rows {
                u_k.set(r, c, u.at(r, i));
            }
        }
        Self::Svd {
            a: a.clone(),
            u: u_k,
            s: keep.iter().map(|&i| s[i]).collect(),
            vt: vt.select_rows(&keep, false),
        }
    }

    fn a(&self) -> &Mat {
        match self {
            Self::Qr { a, .. } | Self::Svd { a, .. } | Self::Augmented { a, .. } => a,
        }
    }

    /// `inv(A Aᵀ) A x` by the factorisation (QR and SVD).
    fn ls_core(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Qr { q, r, perm, .. } => {
                let aux1 = q.tdot(x);
                let aux2 = solve_upper(r, &aux1, false);
                let mut z = vec![0.0; perm.len()];
                for (i, &p) in perm.iter().enumerate() {
                    z[p] = aux2[i];
                }
                z
            }
            Self::Svd { u, s, vt, .. } => {
                let aux1 = vt.dot(x);
                let aux2: Vec<f64> = aux1.iter().zip(s).map(|(a, si)| 1.0 / si * a).collect();
                u.dot(&aux2)
            }
            Self::Augmented { .. } => Vec::new(),
        }
    }

    /// `Z x = x - Aᵀ inv(A Aᵀ) A x` with SciPy's iterative refinement.
    fn null_space(&self, x: &[f64]) -> Vec<f64> {
        let a = self.a();
        match self {
            Self::Augmented { k, lu, .. } => {
                let n = a.cols;
                let mut v = x.to_vec();
                v.resize(n + a.rows, 0.0);
                let mut lu_sol = lu.solve(&v);
                let mut z = lu_sol[..n].to_vec();
                let mut refin = 0;
                while orthogonality(a, &z) > ORTH_TOL {
                    if refin >= MAX_REFIN {
                        break;
                    }
                    let new_v = sub(&v, &k.dot(&lu_sol));
                    let update = lu.solve(&new_v);
                    for (s, u) in lu_sol.iter_mut().zip(&update) {
                        *s += u;
                    }
                    z = lu_sol[..n].to_vec();
                    refin += 1;
                }
                z
            }
            _ => {
                let v = self.ls_core(x);
                let mut z = sub(x, &a.tdot(&v));
                let mut refin = 0;
                while orthogonality(a, &z) > ORTH_TOL {
                    if refin >= MAX_REFIN {
                        break;
                    }
                    let v = self.ls_core(&z);
                    z = sub(&z, &a.tdot(&v));
                    refin += 1;
                }
                z
            }
        }
    }

    /// `LS x = inv(A Aᵀ) A x`.
    fn least_squares(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Augmented { a, lu, .. } => {
                let n = a.cols;
                let mut v = x.to_vec();
                v.resize(n + a.rows, 0.0);
                lu.solve(&v)[n..].to_vec()
            }
            _ => self.ls_core(x),
        }
    }

    /// `Y x = Aᵀ inv(A Aᵀ) x`.
    fn row_space(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Self::Augmented { a, lu, .. } => {
                let n = a.cols;
                let mut v = vec![0.0; n];
                v.extend_from_slice(x);
                lu.solve(&v)[..n].to_vec()
            }
            Self::Qr { q, r, perm, .. } => {
                let aux1: Vec<f64> = perm.iter().map(|&p| x[p]).collect();
                let aux2 = solve_upper(r, &aux1, true);
                q.dot(&aux2)
            }
            Self::Svd { u, s, vt, .. } => {
                let aux1 = u.tdot(x);
                let aux2: Vec<f64> = aux1.iter().zip(s).map(|(a, si)| 1.0 / si * a).collect();
                vt.tdot(&aux2)
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// QP subproblems (qp_subproblem.py)
// ══════════════════════════════════════════════════════════════════════

/// `sphere_intersections(z, d, trust_radius, entire_line)`.
fn sphere_intersections(
    z: &[f64],
    d: &[f64],
    trust_radius: f64,
    entire_line: bool,
) -> (f64, f64, bool) {
    if norm(d) == 0.0 {
        return (0.0, 0.0, false);
    }
    if trust_radius.is_infinite() {
        return if entire_line {
            (f64::NEG_INFINITY, f64::INFINITY, true)
        } else {
            (0.0, 1.0, true)
        };
    }
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - trust_radius * trust_radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return (0.0, 0.0, false);
    }
    let sqrt_discriminant = discriminant.sqrt();
    let aux = b + sqrt_discriminant.copysign(b);
    let mut ta = -aux / (2.0 * a);
    let mut tb = -2.0 * c / aux;
    if tb < ta {
        std::mem::swap(&mut ta, &mut tb);
    }
    if entire_line {
        (ta, tb, true)
    } else if tb < 0.0 || ta > 1.0 {
        (0.0, 0.0, false)
    } else {
        (py_max(0.0, ta), py_min(1.0, tb), true)
    }
}

/// `box_intersections(z, d, lb, ub, entire_line)`.
fn box_intersections(
    z: &[f64],
    d: &[f64],
    lb: &[f64],
    ub: &[f64],
    entire_line: bool,
) -> (f64, f64, bool) {
    if norm(d) == 0.0 {
        return (0.0, 0.0, false);
    }
    for i in 0..d.len() {
        if d[i] == 0.0 && (z[i] < lb[i] || z[i] > ub[i]) {
            return (0.0, 0.0, false);
        }
    }
    let mut ta: Option<f64> = None;
    let mut tb: Option<f64> = None;
    for i in 0..d.len() {
        if d[i] == 0.0 {
            continue;
        }
        let t_lb = (lb[i] - z[i]) / d[i];
        let t_ub = (ub[i] - z[i]) / d[i];
        let lo = np_minimum(t_lb, t_ub);
        let hi = np_maximum(t_lb, t_ub);
        // Python's builtin max / min over the arrays.
        ta = Some(match ta {
            None => lo,
            Some(best) => py_max(best, lo),
        });
        tb = Some(match tb {
            None => hi,
            Some(best) => py_min(best, hi),
        });
    }
    let (mut ta, mut tb) = (ta.unwrap_or(f64::NAN), tb.unwrap_or(f64::NAN));
    let mut intersect = ta <= tb;
    if !entire_line {
        if tb < 0.0 || ta > 1.0 {
            intersect = false;
            ta = 0.0;
            tb = 0.0;
        } else {
            ta = py_max(0.0, ta);
            tb = py_min(1.0, tb);
        }
    }
    (ta, tb, intersect)
}

/// `box_sphere_intersections(z, d, lb, ub, trust_radius, entire_line)`.
fn box_sphere_intersections(
    z: &[f64],
    d: &[f64],
    lb: &[f64],
    ub: &[f64],
    trust_radius: f64,
    entire_line: bool,
) -> (f64, f64, bool) {
    let (ta_b, tb_b, intersect_b) = box_intersections(z, d, lb, ub, entire_line);
    let (ta_s, tb_s, intersect_s) = sphere_intersections(z, d, trust_radius, entire_line);
    let ta = np_maximum(ta_b, ta_s);
    let tb = np_minimum(tb_b, tb_s);
    (ta, tb, intersect_b && intersect_s && ta <= tb)
}

fn inside_box_boundaries(x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    x.iter().zip(lb).all(|(xi, l)| l <= xi) && x.iter().zip(ub).all(|(xi, u)| xi <= u)
}

fn reinforce_box_boundaries(x: &[f64], lb: &[f64], ub: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&xi, (&l, &u))| np_minimum(np_maximum(xi, l), u))
        .collect()
}

/// `modified_dogleg(A, Y, b, trust_radius, lb, ub)`: approximately minimise `½‖A x + b‖²`
/// in the trust region and the box.
fn modified_dogleg(
    a: &Mat,
    proj: &Projections,
    b: &[f64],
    trust_radius: f64,
    lb: &[f64],
    ub: &[f64],
) -> Vec<f64> {
    let newton_point = neg(&proj.row_space(b));
    if inside_box_boundaries(&newton_point, lb, ub) && norm(&newton_point) <= trust_radius {
        return newton_point;
    }
    let g = a.tdot(b);
    let a_g = a.dot(&g);
    let coef = -dot(&g, &g) / dot(&a_g, &a_g);
    let cauchy_point = scaled(coef, &g);
    let origin = vec![0.0; cauchy_point.len()];

    let p = sub(&newton_point, &cauchy_point);
    let (_, alpha, intersect) =
        box_sphere_intersections(&cauchy_point, &p, lb, ub, trust_radius, false);
    let x1 = if intersect {
        add(&cauchy_point, &scaled(alpha, &p))
    } else {
        let (_, alpha, _) =
            box_sphere_intersections(&origin, &cauchy_point, lb, ub, trust_radius, false);
        add(&origin, &scaled(alpha, &cauchy_point))
    };
    let (_, alpha, _) =
        box_sphere_intersections(&origin, &newton_point, lb, ub, trust_radius, false);
    let x2 = add(&origin, &scaled(alpha, &newton_point));
    if norm(&add(&a.dot(&x1), b)) < norm(&add(&a.dot(&x2), b)) {
        x1
    } else {
        x2
    }
}

/// SciPy's `cg_info`.
#[derive(Debug, Clone, Copy)]
struct CgInfo {
    niter: usize,
    stop_cond: u8,
}

/// `projected_cg(H, c, Z, Y, b, trust_radius, lb, ub)`: the equality-constrained QP
/// `min ½ xᵀHx + cᵀx  s.t.  A x + b = 0, ‖x‖ <= Δ, lb <= x <= ub` by projected CG.
fn projected_cg<M: SqpModel>(
    model: &mut M,
    h: &M::Hess,
    c: &[f64],
    proj: &Projections,
    b: &[f64],
    trust_radius: f64,
    lb: &[f64],
    ub: &[f64],
) -> Result<(Vec<f64>, CgInfo), OptError> {
    const CLOSE_TO_ZERO: f64 = 1e-25;
    let n = c.len();
    let m = b.len();
    let mut x = proj.row_space(&neg(b));
    let hx = model.hess_dot(h, &x)?;
    let mut r = proj.null_space(&add(&hx, c));
    let mut g = proj.null_space(&r);
    let mut p = neg(&g);
    let mut h_p = model.hess_dot(h, &p)?;
    let norm_g = norm(&g);
    let mut rt_g = norm_g * norm_g;

    let tr_distance = trust_radius - norm(&x);
    if tr_distance < 0.0 {
        return Err(invalid("Trust region problem does not have a solution."));
    } else if tr_distance < CLOSE_TO_ZERO {
        return Ok((
            x,
            CgInfo {
                niter: 0,
                stop_cond: 2,
            },
        ));
    }
    let tol = py_max(py_min(0.01 * rt_g.sqrt(), 0.1 * rt_g), CLOSE_TO_ZERO);
    let max_iter = n.saturating_sub(m);
    let max_infeasible_iter = n.saturating_sub(m);

    let mut stop_cond = 1;
    let mut counter = 0usize;
    let mut last_feasible_x = vec![0.0; n];
    let mut k = 0;
    for _ in 0..max_iter {
        if rt_g < tol {
            stop_cond = 4;
            break;
        }
        k += 1;
        let pt_h_p = dot(&h_p, &p);
        if pt_h_p <= 0.0 {
            if trust_radius.is_infinite() {
                return Err(invalid(
                    "Negative curvature not allowed for unrestricted problems.",
                ));
            }
            let (_, alpha, intersect) =
                box_sphere_intersections(&x, &p, lb, ub, trust_radius, true);
            if intersect {
                x = add(&x, &scaled(alpha, &p));
            }
            x = reinforce_box_boundaries(&x, lb, ub);
            stop_cond = 3;
            break;
        }
        let alpha = rt_g / pt_h_p;
        let alpha_p = scaled(alpha, &p);
        let x_next = add(&x, &alpha_p);
        if norm(&x_next) >= trust_radius {
            let (_, theta, intersect) =
                box_sphere_intersections(&x, &alpha_p, lb, ub, trust_radius, false);
            if intersect {
                x = add(&x, &scaled(theta * alpha, &p));
            }
            x = reinforce_box_boundaries(&x, lb, ub);
            stop_cond = 2;
            break;
        }
        if inside_box_boundaries(&x_next, lb, ub) {
            counter = 0;
        } else {
            counter += 1;
        }
        if counter > 0 {
            let (_, theta, intersect) =
                box_sphere_intersections(&x, &alpha_p, lb, ub, trust_radius, false);
            if intersect {
                last_feasible_x =
                    reinforce_box_boundaries(&add(&x, &scaled(theta * alpha, &p)), lb, ub);
                counter = 0;
            }
        }
        if counter > max_infeasible_iter {
            break;
        }
        let r_next = add(&r, &scaled(alpha, &h_p));
        let g_next = proj.null_space(&r_next);
        let norm_g_next = norm(&g_next);
        let rt_g_next = norm_g_next * norm_g_next;
        let beta = rt_g_next / rt_g;
        p = add(&neg(&g_next), &scaled(beta, &p));
        x = x_next;
        g = g_next;
        r.clone_from(&g);
        let norm_g = norm(&g);
        rt_g = norm_g * norm_g;
        h_p = model.hess_dot(h, &p)?;
    }
    if !inside_box_boundaries(&x, lb, ub) {
        x = last_feasible_x;
    }
    Ok((
        x,
        CgInfo {
            niter: k,
            stop_cond,
        },
    ))
}

// ══════════════════════════════════════════════════════════════════════
// Byrd–Omojokun trust-region SQP (equality_constrained_sqp.py)
// ══════════════════════════════════════════════════════════════════════

/// The callbacks `equality_constrained_sqp` takes.
trait SqpModel {
    type Hess;
    /// `fun_and_constr(x)`; may rewrite `x` (the barrier problem's `keep_feasible` slacks).
    fn fun_and_constr(&mut self, x: &mut [f64]) -> Result<(f64, Vec<f64>), OptError>;
    fn grad_and_jac(&mut self, x: &[f64]) -> Result<(Vec<f64>, Mat), OptError>;
    fn lagr_hess(&mut self, x: &[f64], v: &[f64]) -> Result<Self::Hess, OptError>;
    fn hess_dot(&mut self, h: &Self::Hess, p: &[f64]) -> Result<Vec<f64>, OptError>;
    /// `scaling(x).dot(d)`.
    fn scale(&self, x: &[f64], d: &[f64]) -> Vec<f64>;
    fn projections(&self, a: &Mat) -> Result<Projections, OptError>;
    #[allow(clippy::too_many_arguments)]
    fn stop_criteria(
        &mut self,
        x: &[f64],
        last_iteration_failed: bool,
        optimality: f64,
        constr_violation: f64,
        trust_radius: f64,
        penalty: f64,
        cg_info: CgInfo,
    ) -> Result<bool, OptError>;
}

/// `equality_constrained_sqp`: minimise `fun(x)` subject to `constr(x) = 0`.
#[allow(clippy::too_many_arguments)]
fn equality_constrained_sqp<M: SqpModel>(
    model: &mut M,
    x0: Vec<f64>,
    fun0: f64,
    grad0: Vec<f64>,
    constr0: Vec<f64>,
    jac0: Mat,
    initial_penalty: f64,
    initial_trust_radius: f64,
    trust_lb: Option<&[f64]>,
    trust_ub: Option<&[f64]>,
) -> Result<Vec<f64>, OptError> {
    const PENALTY_FACTOR: f64 = 0.3;
    const LARGE_REDUCTION_RATIO: f64 = 0.9;
    const INTERMEDIARY_REDUCTION_RATIO: f64 = 0.3;
    const SUFFICIENT_REDUCTION_RATIO: f64 = 1e-8;
    const TRUST_ENLARGEMENT_FACTOR_L: f64 = 7.0;
    const TRUST_ENLARGEMENT_FACTOR_S: f64 = 2.0;
    const MAX_TRUST_REDUCTION: f64 = 0.5;
    const MIN_TRUST_REDUCTION: f64 = 0.1;
    const SOC_THRESHOLD: f64 = 0.1;
    const TR_FACTOR: f64 = 0.8;
    const BOX_FACTOR: f64 = 0.5;

    let n = x0.len();
    let trust_lb = trust_lb.map_or_else(|| vec![f64::NEG_INFINITY; n], <[f64]>::to_vec);
    let trust_ub = trust_ub.map_or_else(|| vec![f64::INFINITY; n], <[f64]>::to_vec);
    let box_lb = scaled(BOX_FACTOR, &trust_lb);
    let box_ub = scaled(BOX_FACTOR, &trust_ub);

    let mut x = x0;
    let mut trust_radius = initial_trust_radius;
    let mut penalty = initial_penalty;
    let mut f = fun0;
    let mut c = grad0;
    let mut b = constr0;
    let mut a = jac0;
    let mut proj = model.projections(&a)?;
    let mut v = neg(&proj.least_squares(&c));
    let mut h = model.lagr_hess(&x, &v)?;
    let mut optimality = norm_inf(&add(&c, &a.tdot(&v)));
    let mut constr_violation = if b.is_empty() { 0.0 } else { norm_inf(&b) };
    let mut cg_info = CgInfo {
        niter: 0,
        stop_cond: 0,
    };
    let mut last_iteration_failed = false;
    while !model.stop_criteria(
        &x,
        last_iteration_failed,
        optimality,
        constr_violation,
        trust_radius,
        penalty,
        cg_info,
    )? {
        // Normal step: minimise ½‖A dn + b‖² in 0.8Δ and the half box.
        let dn = modified_dogleg(&a, &proj, &b, TR_FACTOR * trust_radius, &box_lb, &box_ub);

        // Tangential step: the QP in the null space of A.
        let c_t = add(&model.hess_dot(&h, &dn)?, &c);
        let b_t = vec![0.0; b.len()];
        let norm_dn = norm(&dn);
        let trust_radius_t = (trust_radius * trust_radius - norm_dn * norm_dn).sqrt();
        let lb_t = sub(&trust_lb, &dn);
        let ub_t = sub(&trust_ub, &dn);
        let (dt, info) = projected_cg(model, &h, &c_t, &proj, &b_t, trust_radius_t, &lb_t, &ub_t)?;
        cg_info = info;

        let d = add(&dn, &dt);
        let quadratic_model = 0.5 * dot(&model.hess_dot(&h, &d)?, &d) + dot(&c, &d);
        let linearized_constr = add(&a.dot(&d), &b);
        let vpred = py_max(1e-16, norm(&b) - norm(&linearized_constr));
        let previous_penalty = penalty;
        if quadratic_model > 0.0 {
            let new_penalty = quadratic_model / ((1.0 - PENALTY_FACTOR) * vpred);
            penalty = py_max(penalty, new_penalty);
        }
        let predicted_reduction = -quadratic_model + penalty * vpred;

        let merit_function = f + penalty * norm(&b);
        let mut x_next = add(&x, &model.scale(&x, &d));
        let (mut f_next, mut b_next) = model.fun_and_constr(&mut x_next)?;
        let merit_function_next = f_next + penalty * norm(&b_next);
        let actual_reduction = merit_function - merit_function_next;
        let mut reduction_ratio = actual_reduction / predicted_reduction;

        // Second-order correction.
        if reduction_ratio < SUFFICIENT_REDUCTION_RATIO && norm(&dn) <= SOC_THRESHOLD * norm(&dt) {
            let y = neg(&proj.row_space(&b_next));
            let (_, t, intersect) = box_intersections(&d, &y, &trust_lb, &trust_ub, false);
            let d_soc = add(&d, &scaled(t, &y));
            let mut x_soc = add(&x, &model.scale(&x, &d_soc));
            let (f_soc, b_soc) = model.fun_and_constr(&mut x_soc)?;
            let merit_function_soc = f_soc + penalty * norm(&b_soc);
            let actual_reduction_soc = merit_function - merit_function_soc;
            let reduction_ratio_soc = actual_reduction_soc / predicted_reduction;
            if intersect && reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO {
                x_next = x_soc;
                f_next = f_soc;
                b_next = b_soc;
                reduction_ratio = reduction_ratio_soc;
            }
        }

        // Trust-region update.
        if reduction_ratio >= LARGE_REDUCTION_RATIO {
            trust_radius = py_max(TRUST_ENLARGEMENT_FACTOR_L * norm(&d), trust_radius);
        } else if reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO {
            trust_radius = py_max(TRUST_ENLARGEMENT_FACTOR_S * norm(&d), trust_radius);
        } else if reduction_ratio < SUFFICIENT_REDUCTION_RATIO {
            let trust_reduction = (1.0 - SUFFICIENT_REDUCTION_RATIO) / (1.0 - reduction_ratio);
            let new_trust_radius = trust_reduction * norm(&d);
            if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius {
                trust_radius *= MAX_TRUST_REDUCTION;
            } else if new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius {
                trust_radius = new_trust_radius;
            } else {
                trust_radius *= MIN_TRUST_REDUCTION;
            }
        }

        if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO {
            x = x_next;
            f = f_next;
            b = b_next;
            let (c_new, a_new) = model.grad_and_jac(&x)?;
            c = c_new;
            a = a_new;
            proj = model.projections(&a)?;
            v = neg(&proj.least_squares(&c));
            h = model.lagr_hess(&x, &v)?;
            last_iteration_failed = false;
            optimality = norm_inf(&add(&c, &a.tdot(&v)));
            constr_violation = if b.is_empty() { 0.0 } else { norm_inf(&b) };
        } else {
            penalty = previous_penalty;
            last_iteration_failed = true;
        }
    }
    Ok(x)
}

// ══════════════════════════════════════════════════════════════════════
// The problem: objective, constraints, state and stop criteria
// ══════════════════════════════════════════════════════════════════════

/// SciPy's `state` `OptimizeResult`, updated by `update_state_sqp` / `update_state_ip`.
struct State {
    nit: usize,
    nfev: usize,
    njev: usize,
    nhev: usize,
    cg_niter: usize,
    cg_stop_cond: u8,
    x: Vec<f64>,
    fun: f64,
    grad: Vec<f64>,
    lagrangian_grad: Vec<f64>,
    v: Vec<Vec<f64>>,
    constr: Vec<Vec<f64>>,
    jac: Vec<Mat>,
    constr_nfev: Vec<usize>,
    constr_njev: Vec<usize>,
    constr_nhev: Vec<usize>,
    optimality: f64,
    constr_violation: f64,
    tr_radius: f64,
    constr_penalty: f64,
    barrier_parameter: f64,
    barrier_tolerance: f64,
    status: Option<u8>,
}

struct Problem<'a, 'f, F> {
    objective: ScalarFunction<'f, F>,
    prepared: Vec<Prepared<'a>>,
    n_vars: usize,
    n_eq: usize,
    n_ineq: usize,
    keep_feasible: Vec<bool>,
    sparse: bool,
    factorization: Option<FactorizationMethod>,
    gtol: f64,
    xtol: f64,
    barrier_tol: f64,
    maxiter: usize,
    callback: Option<MinimizeCallback>,
    state: State,
}

impl<F> Problem<'_, '_, F>
where
    F: Fn(&[f64]) -> f64,
{
    /// `canonical.fun(x)`: `(c_eq, c_ineq)`.
    fn canon_fun(&mut self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>), OptError> {
        let mut eq = Vec::with_capacity(self.n_eq);
        let mut ineq = Vec::with_capacity(self.n_ineq);
        for p in &mut self.prepared {
            if matches!(p.canon, CanonKind::Empty) {
                continue;
            }
            let f = p.eval(x)?;
            let (e, i) = p.canon.map_fun(&f, &p.lb, &p.ub);
            eq.extend(e);
            ineq.extend(i);
        }
        Ok((eq, ineq))
    }

    /// `canonical.jac(x)`: `(J_eq, J_ineq)`.
    fn canon_jac(&mut self, x: &[f64]) -> Result<(Mat, Mat), OptError> {
        let n = self.n_vars;
        let mut eq = Vec::new();
        let mut ineq = Vec::new();
        for p in &mut self.prepared {
            if matches!(p.canon, CanonKind::Empty) {
                continue;
            }
            let j = p.eval_jac(x)?;
            let (e, i) = p.canon.map_jac(&j);
            eq.push(e);
            ineq.push(i);
        }
        Ok((Mat::vstack(&eq, n), Mat::vstack(&ineq, n)))
    }

    /// `LagrangianHessian(x, v_eq, v_ineq)`.
    fn lagr_hess(&mut self, x: &[f64], v_eq: &[f64], v_ineq: &[f64]) -> Result<LagrHess, OptError> {
        let obj = self.objective.hess(x)?;
        let mut cons = Vec::with_capacity(self.prepared.len());
        let (mut ie, mut ii) = (0, 0);
        for (idx, p) in self.prepared.iter_mut().enumerate() {
            if matches!(p.canon, CanonKind::Empty) {
                cons.push(ConHess::Zero);
                continue;
            }
            let (ne, ni) = p.canon.sizes(p.m);
            let v = p
                .canon
                .map_multipliers(&v_eq[ie..ie + ne], &v_ineq[ii..ii + ni], p.m);
            cons.push(p.eval_hess(x, v, idx)?);
            ie += ne;
            ii += ni;
        }
        Ok(LagrHess { obj, cons })
    }

    fn lagr_hess_dot(&self, h: &LagrHess, p: &[f64]) -> Result<Vec<f64>, OptError> {
        let h_obj = self.objective.hess_dot(&h.obj, p)?;
        let mut h_cons = vec![0.0; p.len()];
        for ch in &h.cons {
            let term = match ch {
                ConHess::Zero => continue,
                ConHess::Dense(m) => m.dot(p),
                ConHess::Bfgs(idx) => match &self.prepared[*idx].fun {
                    ConstraintFunction::Vector(vf) => match &vf.hess {
                        VecHessKind::Bfgs(bfgs) => bfgs.dot(p),
                        VecHessKind::Callable(_) => {
                            return Err(invalid("inconsistent constraint Hessian state"));
                        }
                    },
                    ConstraintFunction::Linear(_) => {
                        return Err(invalid("inconsistent constraint Hessian state"));
                    }
                },
            };
            for (s, t) in h_cons.iter_mut().zip(&term) {
                *s += t;
            }
        }
        Ok(add(&h_obj, &h_cons))
    }

    /// `update_state_sqp`.
    fn update_state(
        &mut self,
        x: &[f64],
        last_iteration_failed: bool,
        tr_radius: f64,
        constr_penalty: f64,
        cg_info: CgInfo,
    ) {
        let state = &mut self.state;
        state.nit += 1;
        state.nfev = self.objective.nfev;
        state.njev = self.objective.ngev;
        state.nhev = self.objective.nhev;
        state.constr_nfev = self.prepared.iter().map(|p| p.counts().0).collect();
        state.constr_njev = self.prepared.iter().map(|p| p.counts().1).collect();
        state.constr_nhev = self.prepared.iter().map(|p| p.counts().2).collect();
        if !last_iteration_failed {
            state.x = x.to_vec();
            state.fun = self.objective.f;
            state.grad.clone_from(&self.objective.g);
            state.v = self
                .prepared
                .iter()
                .map(|p| p.multipliers().to_vec())
                .collect();
            state.constr = self.prepared.iter().map(|p| p.values().to_vec()).collect();
            state.jac = self.prepared.iter().map(|p| p.jacobian().clone()).collect();
            let mut lagrangian_grad = state.grad.clone();
            for p in &self.prepared {
                let jt_v = p.jacobian().tdot(p.multipliers());
                for (l, t) in lagrangian_grad.iter_mut().zip(&jt_v) {
                    *l += t;
                }
            }
            state.optimality = norm_inf(&lagrangian_grad);
            state.lagrangian_grad = lagrangian_grad;
            let mut violation = 0.0;
            for p in &self.prepared {
                let c = p.values();
                let below = np_max(p.lb.iter().zip(c).map(|(l, ci)| l - ci));
                let above = np_max(c.iter().zip(&p.ub).map(|(ci, u)| ci - u));
                violation = np_max([violation, below, above].into_iter());
            }
            state.constr_violation = violation;
        }
        state.tr_radius = tr_radius;
        state.constr_penalty = constr_penalty;
        state.cg_niter += cg_info.niter;
        state.cg_stop_cond = cg_info.stop_cond;
    }

    /// The `stop_criteria` closure of `_minimize_trustregion_constr` (`barrier` is
    /// `Some((barrier_parameter, barrier_tolerance))` for `tr_interior_point`).
    fn global_stop(
        &mut self,
        x: &[f64],
        last_iteration_failed: bool,
        tr_radius: f64,
        constr_penalty: f64,
        cg_info: CgInfo,
        barrier: Option<(f64, f64)>,
    ) -> bool {
        self.update_state(x, last_iteration_failed, tr_radius, constr_penalty, cg_info);
        if let Some((barrier_parameter, barrier_tolerance)) = barrier {
            self.state.barrier_parameter = barrier_parameter;
            self.state.barrier_tolerance = barrier_tolerance;
        }
        let state = &mut self.state;
        state.status = None;
        if let Some(callback) = self.callback
            && callback(&state.x)
        {
            state.status = Some(3);
            return true;
        }
        if state.optimality < self.gtol && state.constr_violation < self.gtol {
            state.status = Some(1);
        } else if state.tr_radius < self.xtol
            && barrier.is_none_or(|(barrier_parameter, _)| barrier_parameter < self.barrier_tol)
        {
            state.status = Some(2);
        } else if state.nit >= self.maxiter {
            state.status = Some(0);
        }
        state.status.is_some()
    }
}

/// The original problem as `equality_constrained_sqp` sees it (no inequalities).
struct EqualityModel<'p, 'a, 'f, F> {
    problem: &'p mut Problem<'a, 'f, F>,
}

impl<F> SqpModel for EqualityModel<'_, '_, '_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Hess = LagrHess;

    fn fun_and_constr(&mut self, x: &mut [f64]) -> Result<(f64, Vec<f64>), OptError> {
        let f = self.problem.objective.fun(x)?;
        let (c_eq, _) = self.problem.canon_fun(x)?;
        Ok((f, c_eq))
    }

    fn grad_and_jac(&mut self, x: &[f64]) -> Result<(Vec<f64>, Mat), OptError> {
        let g = self.problem.objective.grad(x)?;
        let (j_eq, _) = self.problem.canon_jac(x)?;
        Ok((g, j_eq))
    }

    fn lagr_hess(&mut self, x: &[f64], v: &[f64]) -> Result<LagrHess, OptError> {
        self.problem.lagr_hess(x, v, &[])
    }

    fn hess_dot(&mut self, h: &LagrHess, p: &[f64]) -> Result<Vec<f64>, OptError> {
        self.problem.lagr_hess_dot(h, p)
    }

    fn scale(&self, _x: &[f64], d: &[f64]) -> Vec<f64> {
        d.to_vec()
    }

    fn projections(&self, a: &Mat) -> Result<Projections, OptError> {
        Projections::new(a, self.problem.sparse, self.problem.factorization)
    }

    fn stop_criteria(
        &mut self,
        x: &[f64],
        last_iteration_failed: bool,
        _optimality: f64,
        _constr_violation: f64,
        trust_radius: f64,
        penalty: f64,
        cg_info: CgInfo,
    ) -> Result<bool, OptError> {
        Ok(self.problem.global_stop(
            x,
            last_iteration_failed,
            trust_radius,
            penalty,
            cg_info,
            None,
        ))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Trust-region interior point (tr_interior_point.py)
// ══════════════════════════════════════════════════════════════════════

/// SciPy's `BarrierSubproblem`: `min fun(x) - μ Σ log s  s.t.  c_eq(x) = 0, c_ineq(x) + s = 0`
/// over `z = [x, s]`.
struct BarrierSubproblem<'p, 'a, 'f, F> {
    problem: &'p mut Problem<'a, 'f, F>,
    n_vars: usize,
    n_ineq: usize,
    n_eq: usize,
    barrier_parameter: f64,
    tolerance: f64,
    enforce_feasibility: Vec<bool>,
    xtol: f64,
    terminate: bool,
    lb: Vec<f64>,
    ub: Vec<f64>,
}

/// `S Hs S` and the Lagrangian Hessian in `x` of the barrier problem.
struct BarrierHess {
    hx: LagrHess,
    s_hs_s: Vec<f64>,
}

impl<F> BarrierSubproblem<'_, '_, '_, F>
where
    F: Fn(&[f64]) -> f64,
{
    fn compute_function(&self, f: f64, c_ineq: &[f64], s: &mut [f64]) -> f64 {
        for (i, &enforce) in self.enforce_feasibility.iter().enumerate() {
            if enforce {
                s[i] = -c_ineq[i];
            }
        }
        let log_s: Vec<f64> = s
            .iter()
            .map(|&si| if si > 0.0 { si.ln() } else { f64::NEG_INFINITY })
            .collect();
        f - self.barrier_parameter * np_sum(&log_s)
    }

    fn compute_constr(c_ineq: &[f64], c_eq: &[f64], s: &[f64]) -> Vec<f64> {
        let mut out = c_eq.to_vec();
        out.extend(c_ineq.iter().zip(s).map(|(c, si)| c + si));
        out
    }

    fn compute_gradient(&self, g: &[f64]) -> Vec<f64> {
        let mut out = g.to_vec();
        out.extend(std::iter::repeat_n(-self.barrier_parameter, self.n_ineq));
        out
    }

    fn compute_jacobian(&self, j_eq: &Mat, j_ineq: &Mat, s: &[f64]) -> Mat {
        if self.n_ineq == 0 {
            return j_eq.clone();
        }
        let (n, ni, ne) = (self.n_vars, self.n_ineq, self.n_eq);
        let mut j = Mat::zeros(ne + ni, n + ni);
        for i in 0..ne {
            for k in 0..n {
                j.set(i, k, j_eq.at(i, k));
            }
        }
        for i in 0..ni {
            for k in 0..n {
                j.set(ne + i, k, j_ineq.at(i, k));
            }
            j.set(ne + i, n + i, s[i]);
        }
        j
    }

    /// `function_and_constraints(z)`.
    fn function_and_constraints(&mut self, z: &mut [f64]) -> Result<(f64, Vec<f64>), OptError> {
        let n = self.n_vars;
        let x = z[..n].to_vec();
        let outside = x
            .iter()
            .zip(self.lb.iter().zip(&self.ub))
            .any(|(xi, (l, u))| xi < l || xi > u);
        let (f, c_eq, c_ineq) = if outside {
            (f64::INFINITY, vec![0.0; self.n_eq], vec![0.0; self.n_ineq])
        } else {
            let f = self.problem.objective.fun(&x)?;
            let (c_eq, c_ineq) = self.problem.canon_fun(&x)?;
            (f, c_eq, c_ineq)
        };
        let s = &mut z[n..n + self.n_ineq];
        let fb = self.compute_function(f, &c_ineq, s);
        let cb = Self::compute_constr(&c_ineq, &c_eq, s);
        Ok((fb, cb))
    }

    /// `gradient_and_jacobian(z)`.
    fn gradient_and_jacobian(&mut self, z: &[f64]) -> Result<(Vec<f64>, Mat), OptError> {
        let n = self.n_vars;
        let x = &z[..n];
        let s = &z[n..n + self.n_ineq];
        let g = self.problem.objective.grad(x)?;
        let (j_eq, j_ineq) = self.problem.canon_jac(x)?;
        Ok((
            self.compute_gradient(&g),
            self.compute_jacobian(&j_eq, &j_ineq, s),
        ))
    }
}

impl<F> SqpModel for BarrierSubproblem<'_, '_, '_, F>
where
    F: Fn(&[f64]) -> f64,
{
    type Hess = BarrierHess;

    fn fun_and_constr(&mut self, z: &mut [f64]) -> Result<(f64, Vec<f64>), OptError> {
        self.function_and_constraints(z)
    }

    fn grad_and_jac(&mut self, z: &[f64]) -> Result<(Vec<f64>, Mat), OptError> {
        self.gradient_and_jacobian(z)
    }

    fn lagr_hess(&mut self, z: &[f64], v: &[f64]) -> Result<BarrierHess, OptError> {
        let (n, ne, ni) = (self.n_vars, self.n_eq, self.n_ineq);
        let hx = self.problem.lagr_hess(&z[..n], &v[..ne], &v[ne..ne + ni])?;
        let s_hs_s = if ni > 0 {
            let s = &z[n..n + ni];
            let v_ineq = &v[v.len() - ni..];
            v_ineq
                .iter()
                .zip(s)
                .map(|(&vi, &si)| {
                    if vi > 0.0 {
                        vi * si
                    } else {
                        self.barrier_parameter
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(BarrierHess { hx, s_hs_s })
    }

    fn hess_dot(&mut self, h: &BarrierHess, p: &[f64]) -> Result<Vec<f64>, OptError> {
        let n = self.n_vars;
        let mut out = self.problem.lagr_hess_dot(&h.hx, &p[..n])?;
        if self.n_ineq > 0 {
            out.extend(
                h.s_hs_s
                    .iter()
                    .zip(&p[n..n + self.n_ineq])
                    .map(|(a, b)| a * b),
            );
        }
        Ok(out)
    }

    fn scale(&self, z: &[f64], d: &[f64]) -> Vec<f64> {
        let n = self.n_vars;
        d.iter()
            .enumerate()
            .map(|(i, &di)| if i < n { di } else { z[i] * di })
            .collect()
    }

    fn projections(&self, a: &Mat) -> Result<Projections, OptError> {
        Projections::new(a, self.problem.sparse, self.problem.factorization)
    }

    fn stop_criteria(
        &mut self,
        z: &[f64],
        last_iteration_failed: bool,
        optimality: f64,
        constr_violation: f64,
        trust_radius: f64,
        penalty: f64,
        cg_info: CgInfo,
    ) -> Result<bool, OptError> {
        let x = &z[..self.n_vars];
        if self.problem.global_stop(
            x,
            last_iteration_failed,
            trust_radius,
            penalty,
            cg_info,
            Some((self.barrier_parameter, self.tolerance)),
        ) {
            self.terminate = true;
            return Ok(true);
        }
        let g_cond = optimality < self.tolerance && constr_violation < self.tolerance;
        let x_cond = trust_radius < self.xtol;
        Ok(g_cond || x_cond)
    }
}

/// The values at `x0` that `_minimize_trustregion_constr` hands the inner solvers.
struct Initial {
    x0: Vec<f64>,
    fun0: f64,
    grad0: Vec<f64>,
    c_eq0: Vec<f64>,
    c_ineq0: Vec<f64>,
    j_eq0: Mat,
    j_ineq0: Mat,
}

/// The `tr_interior_point` driver: a sequence of barrier problems with decreasing `μ`.
#[allow(clippy::too_many_arguments)]
fn tr_interior_point<F>(
    problem: &mut Problem<'_, '_, F>,
    init: Initial,
    initial_barrier_parameter: f64,
    initial_tolerance: f64,
    initial_penalty: f64,
    initial_trust_radius: f64,
    fd_bounds: (Vec<f64>, Vec<f64>),
) -> Result<(), OptError>
where
    F: Fn(&[f64]) -> f64,
{
    const BOUNDARY_PARAMETER: f64 = 0.995;
    const BARRIER_DECAY_RATIO: f64 = 0.2;
    const TRUST_ENLARGEMENT: f64 = 5.0;

    let (n_vars, n_eq, n_ineq) = (problem.n_vars, problem.n_eq, problem.n_ineq);
    let enforce_feasibility = problem.keep_feasible.clone();
    let xtol = problem.xtol;
    let mut trust_radius = initial_trust_radius;
    let mut s0: Vec<f64> = init
        .c_ineq0
        .iter()
        .map(|&c| np_maximum(-1.5 * c, 1.0))
        .collect();
    let mut subprob = BarrierSubproblem {
        problem,
        n_vars,
        n_ineq,
        n_eq,
        barrier_parameter: initial_barrier_parameter,
        tolerance: initial_tolerance,
        enforce_feasibility,
        xtol,
        terminate: false,
        lb: fd_bounds.0,
        ub: fd_bounds.1,
    };
    let mut fun0 = subprob.compute_function(init.fun0, &init.c_ineq0, &mut s0);
    let mut grad0 = subprob.compute_gradient(&init.grad0);
    let mut constr0 = BarrierSubproblem::<F>::compute_constr(&init.c_ineq0, &init.c_eq0, &s0);
    let mut jac0 = subprob.compute_jacobian(&init.j_eq0, &init.j_ineq0, &s0);
    let mut z = init.x0;
    z.extend_from_slice(&s0);
    let mut trust_lb = vec![f64::NEG_INFINITY; n_vars];
    trust_lb.extend(std::iter::repeat_n(-BOUNDARY_PARAMETER, n_ineq));
    let trust_ub = vec![f64::INFINITY; n_vars + n_ineq];

    loop {
        z = equality_constrained_sqp(
            &mut subprob,
            z,
            fun0,
            grad0,
            constr0,
            jac0,
            initial_penalty,
            trust_radius,
            Some(&trust_lb),
            Some(&trust_ub),
        )?;
        if subprob.terminate {
            break;
        }
        trust_radius = py_max(
            initial_trust_radius,
            TRUST_ENLARGEMENT * subprob.problem.state.tr_radius,
        );
        subprob.barrier_parameter *= BARRIER_DECAY_RATIO;
        subprob.tolerance *= BARRIER_DECAY_RATIO;
        let (f, c) = subprob.function_and_constraints(&mut z)?;
        let (g, j) = subprob.gradient_and_jacobian(&z)?;
        fun0 = f;
        constr0 = c;
        grad0 = g;
        jac0 = j;
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════
// Driver (minimize_trustregion_constr.py)
// ══════════════════════════════════════════════════════════════════════

/// The trust-constr options with SciPy's defaults filled in.
struct Settings {
    gtol: f64,
    xtol: f64,
    barrier_tol: f64,
    initial_tr_radius: f64,
    initial_constr_penalty: f64,
    initial_barrier_parameter: f64,
    initial_barrier_tolerance: f64,
    maxiter: usize,
}

fn settings(options: &MinimizeOptions<'_>) -> Result<Settings, OptError> {
    let mo = options.method_options;
    let s = Settings {
        gtol: mo.gtol.or(options.tol).unwrap_or(1e-8),
        xtol: mo.xtol.or(options.tol).unwrap_or(1e-8),
        barrier_tol: mo.barrier_tol.or(options.tol).unwrap_or(1e-8),
        initial_tr_radius: mo.initial_tr_radius.unwrap_or(1.0),
        initial_constr_penalty: mo.initial_constr_penalty.unwrap_or(1.0),
        initial_barrier_parameter: mo.initial_barrier_parameter.unwrap_or(0.1),
        initial_barrier_tolerance: mo.initial_barrier_tolerance.unwrap_or(0.1),
        maxiter: options.maxiter.unwrap_or(1000),
    };
    if options.mode == RuntimeMode::Hardened {
        for (name, value) in [
            ("gtol", s.gtol),
            ("xtol", s.xtol),
            ("barrier_tol", s.barrier_tol),
            ("initial_tr_radius", s.initial_tr_radius),
            ("initial_constr_penalty", s.initial_constr_penalty),
            ("initial_barrier_parameter", s.initial_barrier_parameter),
            ("initial_barrier_tolerance", s.initial_barrier_tolerance),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "hardened mode requires a finite positive {name}, got {value}"
                )));
            }
        }
    }
    if let Some(step) = mo.finite_diff_rel_step
        && !step.is_finite()
    {
        return Err(invalid("finite_diff_rel_step must be finite"));
    }
    Ok(s)
}

/// SciPy's bounds preprocessing: `nextafter` outward on the finite sides, and the strict
/// (`keep_feasible`) bounds that confine finite-difference steps and iterates.
fn prepare_bounds(
    bounds: &TrustBounds,
    n: usize,
) -> Result<(TrustBounds, Vec<f64>, Vec<f64>), OptError> {
    let lb = broadcast(&bounds.lb, n, "bounds lb")?;
    let ub = broadcast(&bounds.ub, n, "bounds ub")?;
    let keep_feasible = broadcast(&bounds.keep_feasible, n, "bounds keep_feasible")?;
    for i in 0..n {
        if lb[i].is_nan() || ub[i].is_nan() {
            return Err(OptError::InvalidBounds {
                detail: format!("bound {i} is NaN"),
            });
        }
        if ub[i] < lb[i] {
            return Err(OptError::InvalidBounds {
                detail: String::from("An upper bound is less than the corresponding lower bound."),
            });
        }
        if lb[i] == f64::INFINITY || ub[i] == f64::NEG_INFINITY {
            return Err(OptError::InvalidBounds {
                detail: format!("bound {i} excludes every value"),
            });
        }
    }
    let modified_lb: Vec<f64> = lb
        .iter()
        .map(|&l| if l.is_finite() { l.next_down() } else { l })
        .collect();
    let modified_ub: Vec<f64> = ub
        .iter()
        .map(|&u| if u.is_finite() { u.next_up() } else { u })
        .collect();
    let strict_lb = (0..n)
        .map(|i| {
            if keep_feasible[i] {
                modified_lb[i]
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let strict_ub = (0..n)
        .map(|i| {
            if keep_feasible[i] {
                modified_ub[i]
            } else {
                f64::INFINITY
            }
        })
        .collect();
    Ok((
        TrustBounds {
            lb: modified_lb,
            ub: modified_ub,
            keep_feasible,
        },
        strict_lb,
        strict_ub,
    ))
}

/// `PreparedConstraint(constraint, x0, sparse_jacobian, finite_diff_bounds)`.
fn prepare_constraint<'a>(
    con: &TrustConstraint<'a>,
    x0: &[f64],
    fd_bounds: (&[f64], &[f64]),
    mode: RuntimeMode,
) -> Result<Prepared<'a>, OptError> {
    let n = x0.len();
    let fun = match &con.kind {
        TrustConstraintKind::Linear(a) => {
            let a = dense_from_user(a.clone(), a.len(), n, "LinearConstraint A")?;
            if a.rows == 0 {
                return Err(invalid("a constraint must have at least one component"));
            }
            check_finite(&a.data, mode, "LinearConstraint A")?;
            ConstraintFunction::Linear(LinearFunction::new(Some(a), x0))
        }
        TrustConstraintKind::Nonlinear {
            fun,
            jac,
            hess,
            rel_step,
        } => ConstraintFunction::Vector(Box::new(VectorFunction::new(
            *fun, *jac, *hess, *rel_step, x0, fd_bounds, mode,
        )?)),
    };
    let m = match &fun {
        ConstraintFunction::Vector(vf) => vf.m,
        ConstraintFunction::Linear(lf) => lf.f.len(),
    };
    let lb = broadcast(&con.lb, m, "constraint lb")?;
    let ub = broadcast(&con.ub, m, "constraint ub")?;
    if lb.iter().chain(&ub).any(|v| v.is_nan()) {
        return Err(invalid("constraint bounds must not be NaN"));
    }
    let keep_feasible = broadcast(&con.keep_feasible, m, "constraint keep_feasible")?;
    finish_prepared(fun, lb, ub, keep_feasible)
}

/// The rest of `PreparedConstraint.__init__` (SciPy refuses an `x0` that violates a
/// `keep_feasible` inequality) and the canonical classification.
fn finish_prepared<'a>(
    fun: ConstraintFunction<'a>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    keep_feasible: Vec<bool>,
) -> Result<Prepared<'a>, OptError> {
    let canon = CanonKind::classify(&lb, &ub);
    let prepared = Prepared {
        fun,
        m: lb.len(),
        lb,
        ub,
        keep_feasible,
        canon,
    };
    let f0 = prepared.values();
    for i in 0..prepared.m {
        let mask = prepared.keep_feasible[i] && prepared.lb[i] != prepared.ub[i];
        if mask && (f0[i] < prepared.lb[i] || f0[i] > prepared.ub[i]) {
            return Err(invalid(
                "`x0` is infeasible with respect to some inequality constraint with \
                 `keep_feasible` set to True.",
            ));
        }
    }
    Ok(prepared)
}

/// The problem's options, functions and constraints, before any evaluation.
struct Spec<'s, 'a> {
    x0: &'s [f64],
    gradient: Option<GradientFunc>,
    hess: Option<HessFunc>,
    hessp: Option<HesspFunc>,
    callback: Option<MinimizeCallback>,
    constraints: Vec<&'s TrustConstraint<'a>>,
    bounds: Option<&'s TrustBounds>,
    settings: Settings,
    rel_step: Option<f64>,
    sparse_jacobian: Option<bool>,
    factorization: Option<FactorizationMethod>,
    mode: RuntimeMode,
}

/// `_minimize_trustregion_constr`.
fn minimize_trustregion_constr<F>(
    fun: &F,
    spec: Spec<'_, '_>,
) -> Result<TrustConstrResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    let x0 = spec.x0.to_vec();
    let n = x0.len();
    let settings = spec.settings;
    let hess = if let Some(hess) = spec.hess {
        ObjHessKind::Dense(hess)
    } else if let Some(hessp) = spec.hessp {
        ObjHessKind::Hessp(hessp)
    } else {
        ObjHessKind::Bfgs(BfgsHessian::new(BfgsExceptionStrategy::SkipUpdate, None))
    };
    let (bounds, fd_lb, fd_ub) = match spec.bounds {
        Some(b) => {
            let (modified, strict_lb, strict_ub) = prepare_bounds(b, n)?;
            (Some(modified), strict_lb, strict_ub)
        }
        None => (None, vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };

    let objective = ScalarFunction::new(
        fun,
        &x0,
        spec.gradient,
        hess,
        spec.rel_step,
        (&fd_lb, &fd_ub),
        spec.mode,
    )?;

    let mut prepared = Vec::with_capacity(spec.constraints.len() + 1);
    for con in &spec.constraints {
        prepared.push(prepare_constraint(con, &x0, (&fd_lb, &fd_ub), spec.mode)?);
    }
    // Every Jacobian is dense here, so a constraint is "sparse" only when `sparse_jacobian`
    // says so; the bounds are sparse unless another constraint fixed the format as dense.
    let mut sparse_jacobian = spec.sparse_jacobian;
    if !prepared.is_empty() {
        sparse_jacobian = Some(sparse_jacobian == Some(true));
    }
    if let Some(b) = bounds {
        if sparse_jacobian.is_none() {
            sparse_jacobian = Some(true);
        }
        let identity = ConstraintFunction::Linear(LinearFunction::new(None, &x0));
        prepared.push(finish_prepared(identity, b.lb, b.ub, b.keep_feasible)?);
    }
    let sparse = sparse_jacobian == Some(true);

    let (mut n_eq, mut n_ineq) = (0, 0);
    let mut keep_feasible = Vec::new();
    let mut c_eq0 = Vec::new();
    let mut c_ineq0 = Vec::new();
    let mut j_eq0 = Vec::new();
    let mut j_ineq0 = Vec::new();
    for p in &prepared {
        let (ne, ni) = p.canon.sizes(p.m);
        n_eq += ne;
        n_ineq += ni;
        keep_feasible.extend(p.canon.keep_feasible(&p.keep_feasible));
        let (e, i) = p.canon.map_fun(p.values(), &p.lb, &p.ub);
        c_eq0.extend(e);
        c_ineq0.extend(i);
        let (je, ji) = p.canon.map_jac(p.jacobian());
        j_eq0.push(je);
        j_ineq0.push(ji);
    }
    let method = if n_ineq == 0 {
        TrustConstrMethod::EqualityConstrainedSqp
    } else {
        TrustConstrMethod::TrInteriorPoint
    };

    let n_con = prepared.len();
    let state = State {
        nit: 0,
        nfev: 0,
        njev: 0,
        nhev: 0,
        cg_niter: 0,
        cg_stop_cond: 0,
        x: x0.clone(),
        fun: objective.f,
        grad: objective.g.clone(),
        lagrangian_grad: objective.g.clone(),
        v: prepared.iter().map(|p| p.multipliers().to_vec()).collect(),
        constr: prepared.iter().map(|p| p.values().to_vec()).collect(),
        jac: prepared.iter().map(|p| p.jacobian().clone()).collect(),
        constr_nfev: vec![0; n_con],
        constr_njev: vec![0; n_con],
        constr_nhev: vec![0; n_con],
        optimality: f64::NAN,
        constr_violation: f64::NAN,
        tr_radius: settings.initial_tr_radius,
        constr_penalty: settings.initial_constr_penalty,
        barrier_parameter: settings.initial_barrier_parameter,
        barrier_tolerance: settings.initial_barrier_tolerance,
        status: None,
    };
    let fun0 = objective.f;
    let grad0 = objective.g.clone();
    let mut problem = Problem {
        objective,
        prepared,
        n_vars: n,
        n_eq,
        n_ineq,
        keep_feasible,
        sparse,
        factorization: spec.factorization,
        gtol: settings.gtol,
        xtol: settings.xtol,
        barrier_tol: settings.barrier_tol,
        maxiter: settings.maxiter,
        callback: spec.callback,
        state,
    };
    let init = Initial {
        x0,
        fun0,
        grad0,
        c_eq0,
        c_ineq0,
        j_eq0: Mat::vstack(&j_eq0, n),
        j_ineq0: Mat::vstack(&j_ineq0, n),
    };

    match method {
        TrustConstrMethod::EqualityConstrainedSqp => {
            let mut model = EqualityModel {
                problem: &mut problem,
            };
            equality_constrained_sqp(
                &mut model,
                init.x0,
                init.fun0,
                init.grad0,
                init.c_eq0,
                init.j_eq0,
                settings.initial_constr_penalty,
                settings.initial_tr_radius,
                None,
                None,
            )?;
        }
        TrustConstrMethod::TrInteriorPoint => {
            tr_interior_point(
                &mut problem,
                init,
                settings.initial_barrier_parameter,
                settings.initial_barrier_tolerance,
                settings.initial_constr_penalty,
                settings.initial_tr_radius,
                (fd_lb, fd_ub),
            )?;
        }
    }

    let state = problem.state;
    let mut status = state.status.unwrap_or(0);
    if matches!(status, 1 | 2) && state.constr_violation > settings.gtol {
        status = 4;
    }
    let success = matches!(status, 1 | 2);
    let convergence = match status {
        1 | 2 => ConvergenceStatus::Success,
        3 => ConvergenceStatus::CallbackStop,
        4 => ConvergenceStatus::Infeasible,
        _ => ConvergenceStatus::MaxIterations,
    };
    let message = TERMINATION_MESSAGES[usize::from(status)].to_string();
    let is_ip = method == TrustConstrMethod::TrInteriorPoint;
    let result = OptimizeResult {
        x: state.x,
        fun: Some(state.fun),
        success,
        status: convergence,
        message,
        nfev: state.nfev,
        njev: state.njev,
        nhev: state.nhev,
        nit: state.nit,
        jac: Some(state.grad.clone()),
        hess_inv: None,
        maxcv: Some(state.constr_violation),
    };
    Ok(TrustConstrResult {
        result,
        status,
        method,
        optimality: state.optimality,
        constr_violation: state.constr_violation,
        grad: state.grad,
        lagrangian_grad: state.lagrangian_grad,
        v: state.v,
        constr: state.constr,
        jac: state.jac.iter().map(Mat::to_rows).collect(),
        constr_nfev: state.constr_nfev,
        constr_njev: state.constr_njev,
        constr_nhev: state.constr_nhev,
        niter: state.nit,
        nfev: state.nfev,
        njev: state.njev,
        nhev: state.nhev,
        cg_niter: state.cg_niter,
        cg_stop_cond: state.cg_stop_cond,
        tr_radius: state.tr_radius,
        constr_penalty: state.constr_penalty,
        barrier_parameter: is_ip.then_some(state.barrier_parameter),
        barrier_tolerance: is_ip.then_some(state.barrier_tolerance),
    })
}

/// `scipy.optimize.minimize(fun, x0, method='trust-constr', ...)` with SciPy's full result.
///
/// The constraints are `options.constraints` (SciPy's old-style dicts, converted by
/// `old_constraint_to_new`: an equality is `0 <= fun(x) <= 0`, an inequality
/// `0 <= fun(x) <= inf`) followed by `constraints` (new-style `LinearConstraint` /
/// `NonlinearConstraint`), and the bounds are `options.bounds` or `bounds` (not both). Only
/// equality constraints (or none) run SciPy's `equality_constrained_sqp`; any inequality or
/// bound runs `tr_interior_point`.
///
/// Options, with SciPy's defaults: `method_options.gtol`, `.xtol`, `.barrier_tol` (1e-8 each;
/// `tol` fills the ones left unset), `maxiter` (1000), `.initial_tr_radius` (1),
/// `.initial_constr_penalty` (1), `.initial_barrier_parameter` and
/// `.initial_barrier_tolerance` (0.1 each), `.factorization_method`, `.sparse_jacobian`,
/// `.finite_diff_rel_step`. Without `gradient` the gradient is SciPy's `'2-point'` difference
/// with a relative step; without `hess`/`hessp` the objective's Hessian is SciPy's BFGS
/// approximation (and each nonlinear constraint's, unless it has `hess`). `callback` is called
/// with the current `x` at every stop test, the first before any step; returning `true`
/// stops with status 3.
///
/// SciPy's trust-constr takes no `maxfev` and no `eps`: Strict mode records them in the
/// optimize trace as unknown options and ignores them, Hardened refuses them (see
/// [`crate::minimize::trust_constr`]). Hardened also rejects non-finite function, derivative
/// and constraint values, which Strict passes to the algorithm as SciPy does.
///
/// # Errors
///
/// [`OptError::InvalidArgument`] / [`OptError::InvalidBounds`] for malformed input (empty or
/// non-finite `x0`, mismatched lengths, an upper bound below a lower one, an `x0` infeasible
/// for a `keep_feasible` inequality, a factorization method the Jacobian format does not
/// allow), and the errors the evaluations raise.
pub fn trust_constr_full<F>(
    fun: &F,
    x0: &[f64],
    constraints: &[TrustConstraint<'_>],
    bounds: Option<&TrustBounds>,
    options: MinimizeOptions<'_>,
) -> Result<TrustConstrResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    crate::minimize::check_trust_constr_options(options)?;
    if x0.is_empty() {
        return Err(invalid(
            "x0 must be a finite 1-D vector with at least one element",
        ));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    let settings = settings(&options)?;
    let dict_constraints: Vec<TrustConstraint<'_>> = options
        .constraints
        .iter()
        .map(TrustConstraint::from_dict)
        .collect();
    let tuple_bounds = match (options.bounds, bounds) {
        (Some(_), Some(_)) => {
            return Err(invalid(
                "bounds given both in options.bounds and as an argument",
            ));
        }
        (Some(tuples), None) => {
            if tuples.len() != x0.len() {
                return Err(OptError::InvalidBounds {
                    detail: String::from(
                        "The number of bounds is not compatible with the length of `x0`.",
                    ),
                });
            }
            Some(TrustBounds::from_tuples(tuples))
        }
        _ => None,
    };
    let mo = options.method_options;
    let spec = Spec {
        x0,
        gradient: options.gradient,
        hess: options.hess,
        hessp: options.hessp,
        callback: options.callback,
        constraints: dict_constraints.iter().chain(constraints).collect(),
        bounds: tuple_bounds.as_ref().or(bounds),
        settings,
        rel_step: mo.finite_diff_rel_step,
        sparse_jacobian: mo.sparse_jacobian,
        factorization: mo.factorization_method,
        mode: options.mode,
    };
    minimize_trustregion_constr(fun, spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MinimizeMethodOptions, OptimizeMethod};

    fn sq(t: f64) -> f64 {
        t * t
    }

    fn rosen(v: &[f64]) -> f64 {
        sq(1.0 - v[0]) + 100.0 * sq(v[1] - v[0] * v[0])
    }

    fn hs71(v: &[f64]) -> f64 {
        v[0] * v[3] * (v[0] + v[1] + v[2]) + v[2]
    }

    fn hs71_constraints() -> Vec<Constraint<'static>> {
        vec![
            Constraint::ineq(|v: &[f64]| vec![v[0] * v[1] * v[2] * v[3] - 25.0]),
            Constraint::eq(|v: &[f64]| {
                vec![v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] - 40.0]
            }),
        ]
    }

    const HS71_BOUNDS: [Bound; 4] = [(Some(1.0), Some(5.0)); 4];

    fn solve(
        fun: &dyn Fn(&[f64]) -> f64,
        x0: &[f64],
        options: MinimizeOptions<'_>,
    ) -> TrustConstrResult {
        trust_constr_full(&fun, x0, &[], None, options).expect("trust-constr runs")
    }

    fn quad_grad(v: &[f64]) -> Vec<f64> {
        vec![2.0 * (v[0] - 2.0), 2.0 * (v[1] - 1.0)]
    }

    // SciPy 1.17.1: minimize(rosen, [-1.2, 1], method='trust-constr') -> method
    // 'equality_constrained_sqp' (no constraints at all), status 2, nit 68, nfev 189, njev 63.
    #[test]
    fn unconstrained_problem_runs_the_sqp_with_no_constraints() {
        let r = solve(&rosen, &[-1.2, 1.0], MinimizeOptions::default());
        assert_eq!(r.method, TrustConstrMethod::EqualityConstrainedSqp);
        assert_eq!(r.status, 2, "{}", r.result.message);
        assert!(r.result.success);
        assert_eq!(
            r.result.message,
            "`xtol` termination condition is satisfied."
        );
        assert_eq!((r.niter, r.nfev, r.njev), (68, 189, 63));
        assert!(r.v.is_empty() && r.constr.is_empty());
        assert_eq!(r.barrier_parameter, None);
        for xi in &r.result.x {
            assert!((xi - 1.0).abs() < 1e-4, "x = {:?}", r.result.x);
        }
    }

    // SciPy 1.17.1: Hock–Schittkowski #6 -> status 1, nit 12, nfev 36, x ≈ [1, 1].
    #[test]
    fn equality_only_problem_uses_byrd_omojokun_sqp() {
        let cons = [Constraint::eq(|v: &[f64]| {
            vec![10.0 * (v[1] - v[0] * v[0])]
        })];
        let r = solve(
            &|v: &[f64]| sq(1.0 - v[0]),
            &[-1.2, 1.0],
            MinimizeOptions {
                constraints: &cons,
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.method, TrustConstrMethod::EqualityConstrainedSqp);
        assert_eq!(r.status, 1, "{}", r.result.message);
        assert_eq!((r.niter, r.nfev), (12, 36));
        assert!((r.result.x[0] - 1.0).abs() < 1e-7 && (r.result.x[1] - 1.0).abs() < 1e-7);
        assert!(r.constr_violation < 1e-8 && r.optimality < 1e-8);
        assert_eq!(r.result.maxcv, Some(r.constr_violation));
        assert_eq!(r.v.len(), 1);
        assert!(r.v[0][0].abs() < 1e-8, "v = {:?}", r.v);
    }

    // min (x-2)² + (y-1)² s.t. x + y <= 1: the solution [1, 0] has the upper side active, so
    // its multiplier is positive (SciPy: v = 2).
    #[test]
    fn inequality_only_problem_uses_the_interior_point_method() {
        let f = |v: &[f64]| vec![v[0] + v[1]];
        let con = TrustConstraint::nonlinear(&f, vec![f64::NEG_INFINITY], vec![1.0]);
        let r = trust_constr_full(
            &|v: &[f64]| sq(v[0] - 2.0) + sq(v[1] - 1.0),
            &[0.0, 0.0],
            &[con],
            None,
            MinimizeOptions {
                gradient: Some(quad_grad),
                ..MinimizeOptions::default()
            },
        )
        .expect("runs");
        assert_eq!(r.method, TrustConstrMethod::TrInteriorPoint);
        assert_eq!(r.status, 1, "{}", r.result.message);
        assert!((r.result.x[0] - 1.0).abs() < 1e-6 && r.result.x[1].abs() < 1e-6);
        assert!((r.v[0][0] - 2.0).abs() < 1e-5, "v = {:?}", r.v);
        assert!(r.barrier_parameter.is_some_and(|mu| mu < 0.1));
        assert!(r.barrier_tolerance.is_some());
    }

    // (x-3)² on [0, 2] used to come back as x = 3 under success (frankenscipy-szq1n.7). SciPy
    // 1.17.1 stops at x = 1.9995973376185818 with status 1: the barrier is not driven to zero
    // once the gtol test passes.
    #[test]
    fn bounds_only_problem_honours_the_bounds() {
        let bounds = [(Some(0.0), Some(2.0))];
        let r = solve(
            &|v: &[f64]| (v[0] - 3.0) * (v[0] - 3.0),
            &[1.0],
            MinimizeOptions {
                bounds: Some(&bounds),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.method, TrustConstrMethod::TrInteriorPoint);
        assert_eq!(r.status, 1, "{}", r.result.message);
        assert!(
            (r.result.x[0] - 1.999_597_337_618_581_8).abs() < 1e-9,
            "x = {:?}",
            r.result.x
        );
        assert!(r.result.x[0] <= 2.0);
        // The bound is the only constraint; its upper side is active.
        assert_eq!(r.v.len(), 1);
        assert!(r.v[0][0] > 1.9, "v = {:?}", r.v);
    }

    // Hock–Schittkowski #71 from its infeasible start [1, 5, 5, 1]: equality, inequality and
    // bounds together. SciPy: x ≈ [1, 4.7430, 3.8211, 1.3794], fun ≈ 17.014017.
    #[test]
    fn mixed_constraints_and_bounds_from_an_infeasible_start() {
        let cons = hs71_constraints();
        let r = solve(
            &hs71,
            &[1.0, 5.0, 5.0, 1.0],
            MinimizeOptions {
                constraints: &cons,
                bounds: Some(&HS71_BOUNDS),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.method, TrustConstrMethod::TrInteriorPoint);
        assert_eq!(r.status, 1, "{}", r.result.message);
        let want = [1.0, 4.742_999_6, 3.821_150_0, 1.379_408_3];
        for (x, w) in r.result.x.iter().zip(want) {
            assert!((x - w).abs() < 1e-6, "x = {:?}", r.result.x);
        }
        assert!((r.result.fun.unwrap() - 17.014_017_3).abs() < 1e-6);
        assert!(r.constr_violation < 1e-8);
        // Constraints in order, bounds last: one entry per component.
        assert_eq!(r.v.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1, 4]);
        // The product constraint is active at its lower side (negative multiplier) and x0 = 1
        // at its lower bound.
        assert!(r.v[0][0] < -0.5 && r.v[2][0] < -1.0, "v = {:?}", r.v);
    }

    // HS21 started outside its bounds and inequality.
    #[test]
    fn infeasible_start_reaches_a_feasible_optimum() {
        let cons = [Constraint::ineq(|v: &[f64]| {
            vec![10.0 * v[0] - v[1] - 10.0]
        })];
        let bounds = [(Some(2.0), Some(50.0)), (Some(-50.0), Some(50.0))];
        let r = solve(
            &|v: &[f64]| 0.01 * v[0] * v[0] + v[1] * v[1] - 100.0,
            &[-1.0, -1.0],
            MinimizeOptions {
                constraints: &cons,
                bounds: Some(&bounds),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.status, 1, "{}", r.result.message);
        assert!(r.constr_violation <= 1e-8);
        assert!((r.result.x[0] - 2.0).abs() < 1e-4 && r.result.x[1].abs() < 1e-6);
        assert!((r.result.fun.unwrap() + 99.96).abs() < 1e-6);
    }

    #[test]
    fn maxiter_one_stops_at_x0_with_status_zero() {
        let cons = hs71_constraints();
        let r = solve(
            &hs71,
            &[1.0, 5.0, 5.0, 1.0],
            MinimizeOptions {
                constraints: &cons,
                bounds: Some(&HS71_BOUNDS),
                maxiter: Some(1),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.status, 0);
        assert!(!r.result.success);
        assert_eq!(r.result.status, ConvergenceStatus::MaxIterations);
        assert_eq!(
            r.result.message,
            "The maximum number of function evaluations is exceeded."
        );
        assert_eq!(r.niter, 1);
        assert_eq!(r.result.x, vec![1.0, 5.0, 5.0, 1.0]);
    }

    // SciPy 1.17.1: x = 1 and x = 2 together end at x = [1.5, 0] by the xtol test with a
    // violation of 0.5, which is status 4 (success False).
    #[test]
    fn incompatible_constraints_report_status_four() {
        let cons = [
            Constraint::eq(|v: &[f64]| vec![v[0] - 1.0]),
            Constraint::eq(|v: &[f64]| vec![v[0] - 2.0]),
        ];
        let r = solve(
            &|v: &[f64]| v[0] * v[0] + v[1] * v[1],
            &[0.0, 0.0],
            MinimizeOptions {
                constraints: &cons,
                gradient: Some(|v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]]),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.status, 4);
        assert!(!r.result.success);
        assert_eq!(r.result.status, ConvergenceStatus::Infeasible);
        assert_eq!(r.result.message, "Constraint violation exceeds 'gtol'");
        assert!((r.constr_violation - 0.5).abs() < 1e-12);
        assert!((r.result.x[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn callback_stops_with_status_three() {
        let r = solve(
            &rosen,
            &[-1.2, 1.0],
            MinimizeOptions {
                callback: Some(|x: &[f64]| x[0] > -1.0),
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(r.status, 3);
        assert!(!r.result.success);
        assert_eq!(r.result.status, ConvergenceStatus::CallbackStop);
        assert_eq!(r.result.message, "`callback` raised `StopIteration`.");
        assert!(r.result.x[0] > -1.0);
    }

    #[test]
    fn repeated_runs_are_bit_identical() {
        let cons = hs71_constraints();
        let options = MinimizeOptions {
            constraints: &cons,
            bounds: Some(&HS71_BOUNDS),
            ..MinimizeOptions::default()
        };
        let a = solve(&hs71, &[1.0, 5.0, 5.0, 1.0], options);
        let b = solve(&hs71, &[1.0, 5.0, 5.0, 1.0], options);
        let bits =
            |r: &TrustConstrResult| r.result.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b));
        assert_eq!((a.niter, a.nfev, a.cg_niter), (b.niter, b.nfev, b.cg_niter));
        assert_eq!(a.v, b.v);
    }

    // A duplicated equality makes the Jacobian rank deficient: QR detects it and SciPy falls
    // back to the SVD projections, whose projected CG then gets n - m = 0 iterations, so SciPy
    // 1.17.1 never leaves the (feasible) x0 and stops by xtol (status 2, nit 10, nfev 1) with
    // multipliers [-0.5, -0.5].
    #[test]
    fn rank_deficient_jacobian_falls_back_to_svd_like_scipy() {
        let con = TrustConstraint::linear(
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        );
        let r = trust_constr_full(
            &|v: &[f64]| v[0] * v[0] + v[1] * v[1],
            &[2.0, -1.0],
            &[con],
            None,
            MinimizeOptions {
                gradient: Some(|v: &[f64]| vec![2.0 * v[0], 2.0 * v[1]]),
                ..MinimizeOptions::default()
            },
        )
        .expect("runs");
        assert_eq!((r.status, r.niter, r.nfev), (2, 10, 1));
        assert_eq!(r.result.x, vec![2.0, -1.0]);
        for v in &r.v[0] {
            assert!((v + 0.5).abs() < 1e-12, "v = {:?}", r.v);
        }
    }

    // The three projection methods compute the same operators.
    #[test]
    fn qr_svd_and_augmented_projections_agree() {
        let a = Mat::from_rows(&[vec![1.0, 2.0, 0.5, -1.0], vec![0.3, -1.0, 2.0, 0.7]], 4)
            .expect("matrix");
        let x = [0.4, -1.3, 2.2, 0.9];
        let b = [1.5, -0.25];
        let qr = Projections::new(&a, false, None).expect("qr");
        let svd =
            Projections::new(&a, false, Some(FactorizationMethod::SvdFactorization)).expect("svd");
        let aug = Projections::new(&a, true, None).expect("augmented");
        assert!(matches!(qr, Projections::Qr { .. }));
        assert!(matches!(svd, Projections::Svd { .. }));
        assert!(matches!(aug, Projections::Augmented { .. }));
        let z = qr.null_space(&x);
        assert!(norm(&a.dot(&z)) < 1e-14, "A Z x = {:?}", a.dot(&z));
        let y = qr.row_space(&b);
        assert!(norm(&sub(&a.dot(&y), &b)) < 1e-14);
        for other in [&svd, &aug] {
            assert!(norm(&sub(&other.null_space(&x), &z)) < 1e-13);
            assert!(norm(&sub(&other.row_space(&b), &y)) < 1e-13);
            assert!(norm(&sub(&other.least_squares(&x), &qr.least_squares(&x))) < 1e-13);
        }
    }

    // A bounds-only problem has a sparse Jacobian in SciPy, where only the augmented system
    // (and NormalEquation) are allowed.
    #[test]
    fn factorization_method_must_suit_the_jacobian_format() {
        let bounds = [(Some(0.0), Some(2.0))];
        let options = |method| MinimizeOptions {
            bounds: Some(&bounds),
            method_options: MinimizeMethodOptions {
                factorization_method: Some(method),
                ..MinimizeMethodOptions::default()
            },
            ..MinimizeOptions::default()
        };
        let quad = |v: &[f64]| (v[0] - 3.0) * (v[0] - 3.0);
        let err = trust_constr_full(
            &quad,
            &[1.0],
            &[],
            None,
            options(FactorizationMethod::QrFactorization),
        )
        .expect_err("QR is a dense method");
        assert_eq!(err.to_string(), "Method not allowed for sparse array.");
        let r = trust_constr_full(
            &quad,
            &[1.0],
            &[],
            None,
            options(FactorizationMethod::NormalEquation),
        )
        .expect("NormalEquation falls back to the augmented system");
        assert_eq!(r.status, 1);
        // A dense constraint Jacobian refuses the sparse methods.
        let cons = [Constraint::eq(|v: &[f64]| vec![v[0] - 1.0])];
        let err = trust_constr_full(
            &quad,
            &[1.0],
            &[],
            None,
            MinimizeOptions {
                constraints: &cons,
                method_options: MinimizeMethodOptions {
                    factorization_method: Some(FactorizationMethod::AugmentedSystem),
                    ..MinimizeMethodOptions::default()
                },
                ..MinimizeOptions::default()
            },
        )
        .expect_err("augmented system is a sparse method");
        assert_eq!(err.to_string(), "Method not allowed for dense array.");
    }

    #[test]
    fn keep_feasible_bounds_refuse_an_infeasible_x0_and_hold_the_iterates() {
        let quad = |v: &[f64]| (v[0] - 3.0) * (v[0] - 3.0);
        let bounds = TrustBounds::new(vec![0.0], vec![2.0]).with_keep_feasible(vec![true]);
        let err = trust_constr_full(
            &quad,
            &[2.5],
            &[],
            Some(&bounds),
            MinimizeOptions::default(),
        )
        .expect_err("x0 outside a keep_feasible bound");
        assert!(err.to_string().contains("keep_feasible"), "{err}");
        let log = std::cell::RefCell::new(Vec::new());
        let traced = |v: &[f64]| {
            log.borrow_mut().push(v[0]);
            quad(v)
        };
        let r = trust_constr_full(
            &traced,
            &[1.0],
            &[],
            Some(&bounds),
            MinimizeOptions::default(),
        )
        .expect("runs");
        assert_eq!(r.status, 1, "{}", r.result.message);
        // Every evaluation, finite-difference steps included, stays inside the bounds.
        assert!(log.borrow().iter().all(|&x| (0.0..=2.0).contains(&x)));
    }

    #[test]
    fn invalid_inputs_are_refused() {
        let o = MinimizeOptions::default;
        assert!(matches!(
            trust_constr_full(&rosen, &[], &[], None, o()),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            trust_constr_full(&rosen, &[f64::NAN, 1.0], &[], None, o()),
            Err(OptError::NonFiniteInput { .. })
        ));
        let reversed = [(Some(1.0), Some(0.0)), (None, None)];
        assert!(matches!(
            trust_constr_full(
                &rosen,
                &[0.5, 0.5],
                &[],
                None,
                MinimizeOptions {
                    bounds: Some(&reversed),
                    ..o()
                }
            ),
            Err(OptError::InvalidBounds { .. })
        ));
        let short = [(Some(0.0), Some(1.0))];
        assert!(matches!(
            trust_constr_full(
                &rosen,
                &[0.5, 0.5],
                &[],
                None,
                MinimizeOptions {
                    bounds: Some(&short),
                    ..o()
                }
            ),
            Err(OptError::InvalidBounds { .. })
        ));
        let both = TrustBounds::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        assert!(matches!(
            trust_constr_full(
                &rosen,
                &[0.5, 0.5],
                &[],
                Some(&both),
                MinimizeOptions {
                    bounds: Some(&short),
                    ..o()
                }
            ),
            Err(OptError::InvalidArgument { .. })
        ));
        let wide = TrustConstraint::linear(vec![vec![1.0, 1.0, 1.0]], vec![0.0], vec![1.0]);
        assert!(matches!(
            trust_constr_full(&rosen, &[0.5, 0.5], &[wide], None, o()),
            Err(OptError::InvalidArgument { .. })
        ));
        let bad_lb = TrustConstraint::linear(vec![vec![1.0, 1.0]], vec![0.0, 0.0, 0.0], vec![1.0]);
        assert!(matches!(
            trust_constr_full(&rosen, &[0.5, 0.5], &[bad_lb], None, o()),
            Err(OptError::InvalidArgument { .. })
        ));
        let growing = std::cell::Cell::new(1);
        let f = |_: &[f64]| {
            growing.set(growing.get() + 1);
            vec![0.0; growing.get()]
        };
        let changing = TrustConstraint::nonlinear(&f, vec![0.0], vec![1.0]);
        assert!(matches!(
            trust_constr_full(&rosen, &[0.5, 0.5], &[changing], None, o()),
            Err(OptError::InvalidArgument { .. })
        ));
    }

    // SciPy's trust-constr has no `maxfev` and no `eps`: Strict ignores them (an unknown-option
    // warning), Hardened refuses them, and Hardened rejects non-finite objective values.
    #[test]
    fn unknown_options_and_non_finite_values_follow_the_mode_split() {
        let strict = solve(
            &rosen,
            &[-1.2, 1.0],
            MinimizeOptions {
                maxfev: Some(5),
                ..MinimizeOptions::default()
            },
        );
        let plain = solve(&rosen, &[-1.2, 1.0], MinimizeOptions::default());
        assert_eq!(strict.result.x, plain.result.x);
        for options in [
            MinimizeOptions {
                maxfev: Some(5),
                mode: RuntimeMode::Hardened,
                ..MinimizeOptions::default()
            },
            MinimizeOptions {
                gradient_eps: Some(1e-6),
                mode: RuntimeMode::Hardened,
                ..MinimizeOptions::default()
            },
        ] {
            assert!(matches!(
                trust_constr_full(&rosen, &[-1.2, 1.0], &[], None, options),
                Err(OptError::InvalidArgument { .. })
            ));
        }
        let nan_far = |v: &[f64]| if v[0] > 0.5 { f64::NAN } else { rosen(v) };
        let hardened = MinimizeOptions {
            mode: RuntimeMode::Hardened,
            ..MinimizeOptions::default()
        };
        assert!(matches!(
            trust_constr_full(&nan_far, &[-1.2, 1.0], &[], None, hardened),
            Err(OptError::NonFiniteInput { .. })
        ));
        let r = trust_constr_full(
            &nan_far,
            &[-1.2, 1.0],
            &[],
            None,
            MinimizeOptions::default(),
        )
        .expect("Strict passes NaN to the algorithm, as SciPy does");
        assert!(r.result.x.iter().all(|v| v.is_finite()));
    }

    // `tol` fills gtol, xtol and barrier_tol; an explicit option wins.
    #[test]
    fn tol_fills_the_unset_tolerances() {
        let loose = solve(
            &rosen,
            &[-1.2, 1.0],
            MinimizeOptions {
                tol: Some(1e-3),
                ..MinimizeOptions::default()
            },
        );
        let explicit = solve(
            &rosen,
            &[-1.2, 1.0],
            MinimizeOptions {
                method_options: MinimizeMethodOptions {
                    gtol: Some(1e-3),
                    xtol: Some(1e-3),
                    barrier_tol: Some(1e-3),
                    ..MinimizeMethodOptions::default()
                },
                ..MinimizeOptions::default()
            },
        );
        let default = solve(&rosen, &[-1.2, 1.0], MinimizeOptions::default());
        assert_eq!(loose.result.x, explicit.result.x);
        assert!(loose.niter < default.niter);
        let overridden = solve(
            &rosen,
            &[-1.2, 1.0],
            MinimizeOptions {
                tol: Some(1e-3),
                method_options: MinimizeMethodOptions {
                    gtol: Some(1e-8),
                    xtol: Some(1e-8),
                    ..MinimizeMethodOptions::default()
                },
                ..MinimizeOptions::default()
            },
        );
        assert_eq!(overridden.result.x, default.result.x);
    }

    // Exact Hessians: objective through `hess`, a constraint through its own `hess(x, v)`.
    #[test]
    fn exact_hessians_are_used() {
        let calls = std::cell::Cell::new(0);
        let f = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1]];
        let j = |v: &[f64]| vec![vec![2.0 * v[0], 2.0 * v[1]]];
        let h = |_: &[f64], l: &[f64]| {
            calls.set(calls.get() + 1);
            vec![vec![2.0 * l[0], 0.0], vec![0.0, 2.0 * l[0]]]
        };
        // min x + y on the unit circle: x = y = -1/√2.
        let con = TrustConstraint::nonlinear(&f, vec![1.0], vec![1.0])
            .with_jac(&j)
            .with_hess(&h);
        let r = trust_constr_full(
            &|v: &[f64]| v[0] + v[1],
            &[1.0, 0.0],
            &[con],
            None,
            MinimizeOptions {
                gradient: Some(|_: &[f64]| vec![1.0, 1.0]),
                hess: Some(|_: &[f64]| vec![vec![0.0, 0.0], vec![0.0, 0.0]]),
                ..MinimizeOptions::default()
            },
        )
        .expect("runs");
        assert_eq!(r.status, 1, "{}", r.result.message);
        let s = -std::f64::consts::FRAC_1_SQRT_2;
        assert!((r.result.x[0] - s).abs() < 1e-8 && (r.result.x[1] - s).abs() < 1e-8);
        assert!(calls.get() > 0 && r.constr_nhev[0] == calls.get());
        assert!(r.nhev > 0);
        // ∇f + v ∇c = 0 at the solution (SciPy's sign convention): v = -1 / (2x) = 1/√2.
        assert!((r.v[0][0] - 1.0 / (2.0 * -s)).abs() < 1e-6, "v = {:?}", r.v);
    }

    // `minimize(method=TrustConstr)` now honours bounds and constraints (the old kernel refused
    // them) and reports the constraint violation as `maxcv`.
    #[test]
    fn minimize_dispatches_constrained_trust_constr() {
        let cons = hs71_constraints();
        let options = MinimizeOptions {
            method: Some(OptimizeMethod::TrustConstr),
            constraints: &cons,
            bounds: Some(&HS71_BOUNDS),
            ..MinimizeOptions::default()
        };
        let r = crate::minimize::minimize(hs71, &[1.0, 5.0, 5.0, 1.0], options).expect("runs");
        let full = solve(&hs71, &[1.0, 5.0, 5.0, 1.0], options);
        assert_eq!(r, full.result);
        assert!(r.success && r.maxcv.is_some_and(|cv| cv < 1e-8));
    }
}
