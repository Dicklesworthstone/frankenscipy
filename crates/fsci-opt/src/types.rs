#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;
use serde::{Deserialize, Serialize};

pub type MinimizeCallback = fn(&[f64]) -> bool;
pub type GradientFunc = fn(&[f64]) -> Vec<f64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizeMethod {
    Bfgs,
    ConjugateGradient,
    Powell,
    NelderMead,
    LBfgsB,
    NewtonCg,
    TrustExact,
    /// SciPy `trust-ncg`: Newton conjugate-gradient trust region (Steihaug–Toint).
    TrustNcg,
    /// SciPy `dogleg`: Powell's dogleg trust region; needs a positive-definite Hessian.
    Dogleg,
    Tnc,
    Slsqp,
    TrustConstr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootMethod {
    Brentq,
    Brenth,
    Bisect,
    Ridder,
    Toms748,
    Newton,
    Secant,
    Halley,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    Success,
    MaxIterations,
    MaxEvaluations,
    PrecisionLoss,
    NanEncountered,
    OutOfBounds,
    CallbackStop,
    NotImplemented,
    InvalidInput,
    /// Stopped at a point that violates the constraints beyond tolerance (SciPy:
    /// "Did not converge to a solution satisfying the constraints").
    Infeasible,
    /// A factorization the method depends on failed (SciPy trust-region status 3: "A linalg
    /// error occurred, such as a non-psd Hessian").
    LinAlgError,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeTraceEntry {
    pub ts_unix_ms: u64,
    pub event: String,
    pub method: OptimizeMethod,
    pub iter_num: usize,
    pub f_val: Option<f64>,
    pub grad_norm: Option<f64>,
    pub step_size: Option<f64>,
    pub mode: RuntimeMode,
    pub reason: Option<String>,
    pub final_x: Option<Vec<f64>>,
    pub final_f: Option<f64>,
    pub total_nfev: usize,
    pub fixture_id: Option<String>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: Option<f64>,
    pub success: bool,
    pub status: ConvergenceStatus,
    pub message: String,
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    pub nit: usize,
    pub jac: Option<Vec<f64>>,
    pub hess_inv: Option<Vec<Vec<f64>>>,
    pub maxcv: Option<f64>,
}

impl OptimizeResult {
    #[must_use]
    pub fn not_implemented(seed: &[f64], message: impl Into<String>) -> Self {
        Self {
            x: seed.to_vec(),
            fun: None,
            success: false,
            status: ConvergenceStatus::NotImplemented,
            message: message.into(),
            nfev: 0,
            njev: 0,
            nhev: 0,
            nit: 0,
            jac: None,
            hess_inv: None,
            maxcv: None,
        }
    }
}

pub type HesspFunc = fn(&[f64], &[f64]) -> Vec<f64>;
/// SciPy `hess=`: the dense Hessian at `x`, as `n` rows of length `n`.
pub type HessFunc = fn(&[f64]) -> Vec<Vec<f64>>;

/// Bound constraint: (lower, upper) for one optimization variable.
///
/// `None` means unbounded in that direction.
pub type Bound = (Option<f64>, Option<f64>);

#[derive(Debug, Clone, Copy)]
pub struct MinimizeOptions<'a> {
    pub method: Option<OptimizeMethod>,
    pub tol: Option<f64>,
    pub maxiter: Option<usize>,
    pub maxfev: Option<usize>,
    pub gradient_eps: f64,
    pub callback: Option<MinimizeCallback>,
    pub gradient: Option<GradientFunc>,
    /// SciPy `hess=`: used by trust-exact, dogleg, trust-ncg and Newton-CG.
    pub hess: Option<HessFunc>,
    /// SciPy `hessp=`: used by trust-ncg and Newton-CG.
    pub hessp: Option<HesspFunc>,
    pub bounds: Option<&'a [Bound]>,
    /// SciPy `constraints=`: equality and inequality constraints. With `method: None` their
    /// presence routes to SLSQP, as `scipy.optimize.minimize` does; a method that cannot
    /// honour them refuses them.
    pub constraints: &'a [Constraint<'a>],
    pub gradient_available: bool,
    pub fixture_id: Option<&'static str>,
    pub seed: Option<u64>,
    pub mode: RuntimeMode,
}

impl Default for MinimizeOptions<'_> {
    fn default() -> Self {
        Self {
            method: None,
            tol: None,
            maxiter: None,
            maxfev: None,
            gradient_eps: 1.0e-8,
            callback: None,
            gradient: None,
            hess: None,
            hessp: None,
            bounds: None,
            constraints: &[],
            gradient_available: true,
            fixture_id: None,
            seed: None,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootOptions {
    pub method: Option<RootMethod>,
    pub xtol: f64,
    pub rtol: f64,
    pub maxiter: usize,
    pub fixture_id: Option<&'static str>,
    pub seed: Option<u64>,
    pub mode: RuntimeMode,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            method: None,
            xtol: 2.0e-12,
            rtol: 8.881_784_197_001_252e-16,
            maxiter: 100,
            fixture_id: None,
            seed: None,
            mode: RuntimeMode::Strict,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptError {
    InvalidArgument {
        detail: String,
    },
    InvalidBounds {
        detail: String,
    },
    SignChangeRequired {
        detail: String,
    },
    NonFiniteInput {
        detail: String,
    },
    EvaluationBudgetExceeded {
        detail: String,
    },
    NotImplemented {
        detail: String,
    },
    /// The solver stopped without meeting its convergence criterion, where SciPy raises rather
    /// than return a result (e.g. `curve_fit`'s "Optimal parameters not found").
    NotConverged {
        detail: String,
    },
}

impl std::fmt::Display for OptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument { detail } => write!(f, "{detail}"),
            Self::InvalidBounds { detail } => write!(f, "{detail}"),
            Self::SignChangeRequired { detail } => write!(f, "{detail}"),
            Self::NonFiniteInput { detail } => write!(f, "{detail}"),
            Self::EvaluationBudgetExceeded { detail } => write!(f, "{detail}"),
            Self::NotImplemented { detail } => write!(f, "{detail}"),
            Self::NotConverged { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for OptError {}

// ══════════════════════════════════════════════════════════════════════
// Constraint Types
// ══════════════════════════════════════════════════════════════════════

/// Box constraints on optimization variables.
///
/// Matches `scipy.optimize.Bounds(lb, ub)`.
///
/// Each element constrains one variable: `lb[i] <= x[i] <= ub[i]`.
/// Use `f64::NEG_INFINITY` / `f64::INFINITY` for unbounded.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower bounds per variable. Length must match x0.
    pub lb: Vec<f64>,
    /// Upper bounds per variable. Length must match x0.
    pub ub: Vec<f64>,
}

impl Bounds {
    /// Create bounds from lower and upper bound vectors.
    /// Validates that `lb[i] <= ub[i]` for all i.
    pub fn new(lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        if lb.len() != ub.len() {
            return Err(OptError::InvalidBounds {
                detail: format!(
                    "lb and ub must have same length (got {} and {})",
                    lb.len(),
                    ub.len()
                ),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("lb[{i}]={lo} > ub[{i}]={hi}"),
                });
            }
        }
        Ok(Self { lb, ub })
    }

    /// Create unbounded constraints for n variables.
    #[must_use]
    pub fn unbounded(n: usize) -> Self {
        Self {
            lb: vec![f64::NEG_INFINITY; n],
            ub: vec![f64::INFINITY; n],
        }
    }

    /// Number of constrained variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lb.len()
    }

    /// Whether there are no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lb.is_empty()
    }

    /// Check if a point is feasible (within all bounds).
    /// Returns false if x has different length than bounds.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        if x.len() != self.lb.len() {
            return false;
        }
        x.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&xi, (&lo, &hi))| xi >= lo && xi <= hi)
    }

    /// Project a point onto the feasible set (clip to bounds).
    /// Panics if x has different length than bounds.
    #[must_use]
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(
            x.len(),
            self.lb.len(),
            "project: x length {} != bounds length {}",
            x.len(),
            self.lb.len()
        );
        x.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .map(|(&xi, (&lo, &hi))| xi.clamp(lo, hi))
            .collect()
    }

    /// Convert to the legacy `Bound` tuple format used by lbfgsb.
    #[must_use]
    pub fn to_bound_tuples(&self) -> Vec<(Option<f64>, Option<f64>)> {
        self.lb
            .iter()
            .zip(self.ub.iter())
            .map(|(&lo, &hi)| {
                let lb = if lo == f64::NEG_INFINITY {
                    None
                } else {
                    Some(lo)
                };
                let ub = if hi == f64::INFINITY { None } else { Some(hi) };
                (lb, ub)
            })
            .collect()
    }
}

/// Linear constraint: lb <= A @ x <= ub.
///
/// Matches `scipy.optimize.LinearConstraint(A, lb, ub)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearConstraint {
    /// Constraint matrix A (m rows × n cols, row-major).
    /// m = number of constraints, n = number of variables.
    pub a: Vec<Vec<f64>>,
    /// Lower bounds on A @ x. Length = m.
    pub lb: Vec<f64>,
    /// Upper bounds on A @ x. Length = m.
    pub ub: Vec<f64>,
}

impl LinearConstraint {
    /// Create a linear constraint. Validates dimensions.
    pub fn new(a: Vec<Vec<f64>>, lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        let m = a.len();
        if m == 0 {
            return Err(OptError::InvalidArgument {
                detail: "constraint matrix A must have at least one row".to_string(),
            });
        }
        if lb.len() != m || ub.len() != m {
            return Err(OptError::InvalidArgument {
                detail: format!(
                    "lb/ub length ({}/{}) must match number of constraint rows ({m})",
                    lb.len(),
                    ub.len()
                ),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("constraint {i}: lb={lo} > ub={hi}"),
                });
            }
        }
        Ok(Self { a, lb, ub })
    }

    /// Number of variables (columns of A).
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.a.first().map_or(0, Vec::len)
    }

    /// Evaluate A @ x. Panics if x length doesn't match columns of A.
    pub fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n_vars();
        assert_eq!(
            x.len(),
            n,
            "evaluate: x length {} != A column count {n}",
            x.len()
        );
        self.a
            .iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi).sum())
            .collect()
    }

    /// Check if x satisfies all constraints.
    /// Returns false if x has wrong dimension.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        if x.len() != self.n_vars() {
            return false;
        }
        let ax = self.evaluate(x);
        ax.iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&v, (&lo, &hi))| v >= lo - 1e-10 && v <= hi + 1e-10)
    }
}

/// Nonlinear constraint: lb <= fun(x) <= ub.
///
/// Matches `scipy.optimize.NonlinearConstraint(fun, lb, ub)`.
#[derive(Clone)]
pub struct NonlinearConstraint {
    /// Constraint function. Returns vector of constraint values.
    pub fun: fn(&[f64]) -> Vec<f64>,
    /// Lower bounds on fun(x).
    pub lb: Vec<f64>,
    /// Upper bounds on fun(x).
    pub ub: Vec<f64>,
}

impl NonlinearConstraint {
    /// Create a nonlinear constraint.
    pub fn new(fun: fn(&[f64]) -> Vec<f64>, lb: Vec<f64>, ub: Vec<f64>) -> Result<Self, OptError> {
        if lb.len() != ub.len() {
            return Err(OptError::InvalidArgument {
                detail: "lb and ub must have same length".to_string(),
            });
        }
        for (i, (&lo, &hi)) in lb.iter().zip(ub.iter()).enumerate() {
            if lo > hi {
                return Err(OptError::InvalidBounds {
                    detail: format!("constraint {i}: lb={lo} > ub={hi}"),
                });
            }
        }
        Ok(Self { fun, lb, ub })
    }

    /// Evaluate fun(x) and check feasibility.
    #[must_use]
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        let values = (self.fun)(x);
        values
            .iter()
            .zip(self.lb.iter().zip(self.ub.iter()))
            .all(|(&v, (&lo, &hi))| v >= lo - 1e-10 && v <= hi + 1e-10)
    }
}

impl std::fmt::Debug for NonlinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonlinearConstraint")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("fun", &"<function>")
            .finish()
    }
}

/// The `type` key of a SciPy constraint dict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// `fun(x) == 0` componentwise.
    Eq,
    /// `fun(x) >= 0` componentwise.
    Ineq,
}

/// Vector-valued constraint function.
pub type ConstraintFn<'a> = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'a>;
/// Constraint Jacobian: one row per constraint component, one column per variable.
pub type ConstraintJacFn<'a> = Box<dyn Fn(&[f64]) -> Vec<Vec<f64>> + Send + Sync + 'a>;

/// A SciPy constraint dict `{'type': 'eq' | 'ineq', 'fun': fun, 'jac': jac}`, as `minimize`
/// takes through [`MinimizeOptions::constraints`]. Without `jac`, SLSQP differentiates `fun`
/// by forward differences with the step `gradient_eps`, as SciPy does with `eps`.
///
/// `LinearConstraint` and `NonlinearConstraint` convert with [`Constraint::from_linear`] and
/// [`Constraint::from_nonlinear`], exactly as SciPy's `new_constraint_to_old` does for SLSQP.
pub struct Constraint<'a> {
    pub kind: ConstraintType,
    pub fun: ConstraintFn<'a>,
    pub jac: Option<ConstraintJacFn<'a>>,
}

impl<'a> Constraint<'a> {
    /// `fun(x) == 0`.
    pub fn eq(fun: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'a) -> Self {
        Self {
            kind: ConstraintType::Eq,
            fun: Box::new(fun),
            jac: None,
        }
    }

    /// `fun(x) >= 0`.
    pub fn ineq(fun: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'a) -> Self {
        Self {
            kind: ConstraintType::Ineq,
            fun: Box::new(fun),
            jac: None,
        }
    }

    /// Attach the Jacobian of `fun` (one row per component).
    #[must_use]
    pub fn with_jac(mut self, jac: impl Fn(&[f64]) -> Vec<Vec<f64>> + Send + Sync + 'a) -> Self {
        self.jac = Some(Box::new(jac));
        self
    }

    /// SciPy `new_constraint_to_old` for `LinearConstraint(A, lb, ub)`: the rows with
    /// `lb == ub` become one equality constraint `A_eq·x − lb_eq`, the finite sides of the rest
    /// one inequality constraint `[A_lo·x − lb_lo, ub_hi − A_hi·x]`, both with the exact
    /// Jacobian.
    #[must_use]
    pub fn from_linear(con: &'a LinearConstraint) -> Vec<Self> {
        let rows = |x: &[f64]| -> Vec<f64> {
            con.a
                .iter()
                .map(|row| row.iter().zip(x).map(|(a, xi)| a * xi).sum())
                .collect()
        };
        let jac = |_: &[f64]| con.a.clone();
        split_constraint(&con.lb, &con.ub, rows, Some(jac))
    }

    /// SciPy `new_constraint_to_old` for `NonlinearConstraint(fun, lb, ub)` (finite-difference
    /// Jacobian, as SciPy uses when `jac` is not callable).
    #[must_use]
    pub fn from_nonlinear(con: &'a NonlinearConstraint) -> Vec<Self> {
        let fun = con.fun;
        split_constraint(&con.lb, &con.ub, fun, None::<fn(&[f64]) -> Vec<Vec<f64>>>)
    }
}

/// Split `lb <= fun(x) <= ub` into SciPy's old-style equality and inequality constraints.
fn split_constraint<'a, F, J>(lb: &[f64], ub: &[f64], fun: F, jac: Option<J>) -> Vec<Constraint<'a>>
where
    F: Fn(&[f64]) -> Vec<f64> + Clone + Send + Sync + 'a,
    J: Fn(&[f64]) -> Vec<Vec<f64>> + Clone + Send + Sync + 'a,
{
    let is_eq: Vec<bool> = lb.iter().zip(ub).map(|(l, u)| l == u).collect();
    let below: Vec<usize> = (0..lb.len())
        .filter(|&i| !is_eq[i] && lb[i] != f64::NEG_INFINITY)
        .collect();
    let above: Vec<usize> = (0..ub.len())
        .filter(|&i| !is_eq[i] && ub[i] != f64::INFINITY)
        .collect();
    let eq_rows: Vec<usize> = (0..lb.len()).filter(|&i| is_eq[i]).collect();
    let mut out = Vec::new();
    if !eq_rows.is_empty() {
        let (rows, lo, f) = (eq_rows.clone(), lb.to_vec(), fun.clone());
        let mut c = Constraint {
            kind: ConstraintType::Eq,
            fun: Box::new(move |x: &[f64]| {
                let y = f(x);
                rows.iter().map(|&i| y[i] - lo[i]).collect()
            }),
            jac: None,
        };
        if let Some(j) = jac.clone() {
            let rows = eq_rows;
            c.jac = Some(Box::new(move |x: &[f64]| {
                let dy = j(x);
                rows.iter().map(|&i| dy[i].clone()).collect()
            }));
        }
        out.push(c);
    }
    if !below.is_empty() || !above.is_empty() {
        let (lo, hi) = (lb.to_vec(), ub.to_vec());
        let (b1, a1) = (below.clone(), above.clone());
        let mut c = Constraint {
            kind: ConstraintType::Ineq,
            fun: Box::new(move |x: &[f64]| {
                let y = fun(x);
                b1.iter()
                    .map(|&i| y[i] - lo[i])
                    .chain(a1.iter().map(|&i| hi[i] - y[i]))
                    .collect()
            }),
            jac: None,
        };
        if let Some(j) = jac {
            c.jac = Some(Box::new(move |x: &[f64]| {
                let dy = j(x);
                below
                    .iter()
                    .map(|&i| dy[i].clone())
                    .chain(
                        above
                            .iter()
                            .map(|&i| dy[i].iter().map(|v| -v).collect::<Vec<f64>>()),
                    )
                    .collect()
            }));
        }
        out.push(c);
    }
    out
}

impl std::fmt::Debug for Constraint<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("kind", &self.kind)
            .field("fun", &"<function>")
            .field("jac", &self.jac.as_ref().map(|_| "<function>"))
            .finish()
    }
}
