#![forbid(unsafe_code)]

//! Radau IIA implicit Runge-Kutta solver for stiff ODEs (3-stage, order 5): SciPy's
//! `scipy.integrate.Radau` (`_ivp/radau.py`), step for step.
//!
//! The collocation system is solved by simplified Newton in SciPy's transformed variables
//! `W = T⁻¹Z`, with one real factor `(MU_REAL/h)I − J` and one complex factor
//! `(MU_COMPLEX/h)I − J`; the Newton start `Z0` is extrapolated from the previous step's
//! dense output. `J` is SciPy's finite-difference `num_jac`, taken at `(t0, y0)` on
//! construction, refreshed at the current point when Newton fails on a Jacobian that is not
//! current, and after an accepted step that converged slowly. The factor pair is kept while
//! the step size is held (predicted factor below 1.2 and no Jacobian refresh), and the step
//! size comes from SciPy's two-step predictive controller. The embedded order-3 error
//! estimate, its re-evaluation after a rejection, the dense output and every `nfev` / `njev` /
//! `nlu` count follow SciPy.
//!
//! Retired with this port (they selected behaviour SciPy does not have): the elementary
//! step-size controller and its `RADAU_ENABLE_PREDICTIVE_CONTROLLER` opt-in, the
//! `RADAU_FORCE_DENSE_LU_REBUILD` / `RADAU_DENSE_LU_REUSE_HITS` single-step LU retention,
//! and the `RADAU_HELD_STEP_HITS` / `RADAU_POST_STEP_JAC_REFRESH_HITS` counters for them.

use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::step_size::{InitialStepRequest, num_jac, select_initial_step};
use crate::validation::{
    ToleranceValue, validate_first_step, validate_max_step, validate_rhs_shape, validate_tol,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{Complex, DMatrix, DVector};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const NEWTON_MAXITER: usize = 6;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERR_EXP: f64 = -0.25; // embedded estimator is order 3 → 1/(3+1).
/// Contraction rate above which SciPy refreshes the Jacobian after an accepted
/// step (`recompute_jac`).
const JAC_REFRESH_RATE: f64 = 1e-3;

/// When `true`, the simplified-Newton corrector re-allocates its scratch (the stage
/// point, the stage derivatives and the two right-hand sides) on EVERY iteration, the
/// original behaviour, kept only for the same-binary A/B. When `false` (default), those
/// buffers are hoisted above the Newton loop and reused; every entry is overwritten before
/// it is read, so the trajectory is bit-identical. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_FORCE_PER_ITER_ALLOC: AtomicBool = AtomicBool::new(false);
/// When `true`, rediscover a diagonal Jacobian by scanning it at every factorization
/// instead of reading the `jac_diagonal` cached when the Jacobian was taken.
/// `#[doc(hidden)]`.
#[doc(hidden)]
/// CONTRACT: BIT-IDENTICAL either way, and `true` selects the SLOW path: both arms feed
/// the same diagonal entries into the same divisions, so any difference is evidence that
/// the cached structure no longer describes the Jacobian it is filed under. A/B agreement
/// is the assertion; the timing is incidental.
pub static RADAU_FORCE_DIAGONAL_RESCAN: AtomicBool = AtomicBool::new(false);
/// Count of Newton factor pairs built on the exact-diagonal path. Execution proof for
/// live-incumbent harnesses: a diagonal fixture reporting zero hits did not exercise the
/// structural lever. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_DIAG_NEWTON_HITS: AtomicUsize = AtomicUsize::new(0);

/// SciPy's `predict_factor`: the two-step step-size prediction of Hairer &
/// Wanner II, Sec. IV.8, falling back to the one-step rule when no previous step
/// is on record.
///
/// The `min(1, multiplier)` clamp is the load-bearing part. It caps growth by the
/// ratio the PREVIOUS step actually achieved, so a solver that is already at its
/// comfortable step size predicts a factor near 1, which lets the step be held and the
/// factor pair survive into the next step.
fn predict_factor(
    h_abs: f64,
    h_abs_old: Option<f64>,
    error_norm: f64,
    error_norm_old: Option<f64>,
) -> f64 {
    let multiplier = match (h_abs_old, error_norm_old) {
        (Some(h_old), Some(e_old)) if error_norm != 0.0 && h_old != 0.0 => {
            h_abs / h_old * (e_old / error_norm).powf(0.25)
        }
        _ => 1.0,
    };
    multiplier.min(1.0) * error_norm.powf(ERR_EXP)
}

/// SciPy's iteration-damped safety factor: a step that needed many Newton
/// iterations is trusted less.
fn newton_safety(n_iter: usize) -> f64 {
    let maxiter = NEWTON_MAXITER as f64;
    0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + n_iter as f64)
}

// SciPy's Radau IIA eigen-transform constants (`scipy/integrate/_ivp/radau.py`).
// `MU_REAL` (see `Tableau`) and `MU_COMPLEX` are the eigenvalues of the inverse
// collocation matrix A⁻¹; `T`/`TI = T⁻¹` are the real similarity transform that
// block-diagonalises A.
const MU_COMPLEX: Complex<f64> = Complex::new(2.6810828736277523, -3.050430199247411);
const RADAU_T: [[f64; 3]; 3] = [
    [
        0.09443876248897524,
        -0.1412552950209542,
        0.03002919410514742,
    ],
    [0.2502131229653333, 0.20412935229379994, -0.3829421127572619],
    [1.0, 1.0, 0.0],
];
const RADAU_TI: [[f64; 3]; 3] = [
    [4.178718591551904, 0.32768282076106237, 0.5233764454994495],
    [
        -4.178718591551904,
        -0.32768282076106237,
        0.47662355450055044,
    ],
    [0.5028726349457868, -2.571926949855605, 0.5960392048282249],
];

/// SciPy's module constants that derive from `S6 = 6 ** 0.5`: the collocation nodes `C`,
/// the error weights `E`, the dense-output coefficients `P`, and `MU_REAL`. Evaluated with
/// the same operations and the same `pow` SciPy uses.
struct Tableau {
    c: [f64; 3],
    e: [f64; 3],
    p: [[f64; 3]; 3],
    mu_real: f64,
}

impl Tableau {
    fn scipy() -> Self {
        let s6 = 6.0_f64.powf(0.5);
        Self {
            c: [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0],
            e: [
                (-13.0 - 7.0 * s6) / 3.0,
                (-13.0 + 7.0 * s6) / 3.0,
                -1.0 / 3.0,
            ],
            p: [
                [
                    13.0 / 3.0 + 7.0 * s6 / 3.0,
                    -23.0 / 3.0 - 22.0 * s6 / 3.0,
                    10.0 / 3.0 + 5.0 * s6,
                ],
                [
                    13.0 / 3.0 - 7.0 * s6 / 3.0,
                    -23.0 / 3.0 + 22.0 * s6 / 3.0,
                    10.0 / 3.0 - 5.0 * s6,
                ],
                [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0],
            ],
            mu_real: 3.0 + 3.0_f64.powf(2.0 / 3.0) - 3.0_f64.powf(1.0 / 3.0),
        }
    }
}

type DenseRealLu = nalgebra::linalg::LU<f64, nalgebra::Dyn, nalgebra::Dyn>;
type DenseComplexLu = nalgebra::linalg::LU<Complex<f64>, nalgebra::Dyn, nalgebra::Dyn>;

/// SciPy's `LU_real` / `LU_complex` pair: factors of `(MU_REAL/h)I − J` and
/// `(MU_COMPLEX/h)I − J`, built and dropped together.
enum NewtonFactors {
    Dense {
        real: DenseRealLu,
        complex: DenseComplexLu,
    },
    /// `J` is exactly diagonal, so both matrices are: their diagonals. Solving is one
    /// division per component, which is what the dense LU's triangular solves reduce to on a
    /// diagonal matrix (no pivoting, zero off-diagonal updates).
    Diagonal {
        real: Vec<f64>,
        complex: Vec<Complex<f64>>,
    },
}

impl NewtonFactors {
    /// `diagonal`: `J`'s diagonal when `J` is exactly diagonal, else `None` for the dense
    /// factorization. The matrices are formed as SciPy forms `M / h * I - J`.
    fn build(jac: &DMatrix<f64>, diagonal: Option<&[f64]>, mu_real: f64, h: f64) -> Self {
        let m_real = mu_real / h;
        let m_complex = Complex::new(MU_COMPLEX.re / h, MU_COMPLEX.im / h);
        match diagonal {
            Some(diagonal) => Self::Diagonal {
                real: diagonal.iter().map(|&j| m_real - j).collect(),
                complex: diagonal
                    .iter()
                    .map(|&j| Complex::new(m_complex.re - j, m_complex.im))
                    .collect(),
            },
            None => {
                let n = jac.nrows();
                let complex = DMatrix::<Complex<f64>>::from_fn(n, n, |row, col| {
                    if row == col {
                        Complex::new(m_complex.re - jac[(row, col)], m_complex.im)
                    } else {
                        Complex::new(0.0 - jac[(row, col)], 0.0)
                    }
                });
                Self::Dense {
                    real: dense_real_newton_matrix(jac, mu_real, h).lu(),
                    complex: complex.lu(),
                }
            }
        }
    }

    /// Solve with the real factor in place; `false` when it is singular.
    fn solve_real(&self, b: &mut DVector<f64>) -> bool {
        match self {
            Self::Dense { real, .. } => real.solve_mut(b),
            Self::Diagonal { real, .. } => {
                if real.contains(&0.0) {
                    return false;
                }
                for (bi, &d) in b.iter_mut().zip(real) {
                    *bi /= d;
                }
                true
            }
        }
    }

    /// Solve with the complex factor in place; `false` when it is singular.
    fn solve_complex(&self, b: &mut DVector<Complex<f64>>) -> bool {
        match self {
            Self::Dense { complex, .. } => complex.solve_mut(b),
            Self::Diagonal { complex, .. } => {
                if complex.iter().any(|d| d.re == 0.0 && d.im == 0.0) {
                    return false;
                }
                for (bi, d) in b.iter_mut().zip(complex) {
                    *bi /= *d;
                }
                true
            }
        }
    }
}

/// SciPy's `RadauDenseOutput(t_old, t, y_old, Q)` with `Q = Z.T @ P`: the collocation
/// polynomial `y_old + Q · (x, x², x³)`, `x = (t - t_old) / (t_new - t_old)`.
struct RadauDense {
    t_old: f64,
    t: f64,
    y_old: Vec<f64>,
    q: Vec<[f64; 3]>,
}

impl RadauDense {
    fn at(&self, s: f64) -> Vec<f64> {
        let x = (s - self.t_old) / (self.t - self.t_old);
        let p = [x, x * x, x * x * x];
        self.q
            .iter()
            .zip(&self.y_old)
            .map(|(q, y)| (q[0] * p[0] + q[1] * p[1] + q[2] * p[2]) + y)
            .collect()
    }
}

/// Materialize `(mu_real / h) I - jac` without first building a zero-filled
/// identity matrix and then overwriting every entry during subtraction.
fn dense_real_newton_matrix(jac: &DMatrix<f64>, mu_real: f64, h: f64) -> DMatrix<f64> {
    let shift = mu_real / h;
    DMatrix::<f64>::from_fn(jac.nrows(), jac.ncols(), |row, col| {
        if row == col {
            shift - jac[(row, col)]
        } else {
            0.0 - jac[(row, col)]
        }
    })
}

/// Configuration for the Radau solver (mirrors `BdfSolverConfig`).
#[derive(Debug, Clone)]
pub struct RadauSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
}

/// SciPy's `norm`: the RMS of `values / scale`, the scale repeating per stage.
fn scaled_rms<'a>(values: impl Iterator<Item = &'a f64>, scale: &[f64], count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for (i, &v) in values.enumerate() {
        let scaled = v / scale[i % scale.len()];
        s += scaled * scaled;
    }
    (s / count as f64).sqrt()
}

pub(crate) fn diagonal_jacobian_entries(jac: &DMatrix<f64>) -> Option<Vec<f64>> {
    let n = jac.nrows();
    if jac.ncols() != n {
        return None;
    }

    let mut diagonal = Vec::with_capacity(n);
    for row in 0..n {
        for col in 0..n {
            let value = jac[(row, col)];
            if row == col {
                diagonal.push(value);
            } else if value != 0.0 {
                return None;
            }
        }
    }
    Some(diagonal)
}

/// Radau IIA solver state.
pub struct RadauSolver {
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_bound: f64,
    direction: f64,
    /// SciPy's `h_abs`: the step size the next step starts from.
    h_abs: f64,
    /// The previous step's entry `h_abs` and error norm, feeding the predictive
    /// controller. SciPy discards them for a step whose entry size had to be clamped.
    h_abs_old: Option<f64>,
    error_norm_old: Option<f64>,
    max_step: f64,
    rtol: f64,
    atol: Vec<f64>,
    mode: RuntimeMode,
    state: OdeSolverState,
    tableau: Tableau,

    nfev: usize,
    njev: usize,
    nlu: usize,

    f: Vec<f64>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,

    jac: DMatrix<f64>,
    /// SciPy's `jac_factor`, carried by `num_jac` from one Jacobian to the next.
    jac_factor: Option<Vec<f64>>,
    /// `jac`'s diagonal when `jac` is exactly diagonal, found once per Jacobian.
    jac_diagonal: Option<Vec<f64>>,
    /// SciPy's `current_jac`: `jac` was taken at the current point.
    current_jac: bool,
    factors: Option<NewtonFactors>,
    sol: Option<RadauDense>,
}

impl RadauSolver {
    pub fn new<F>(
        fun: &mut F,
        config: RadauSolverConfig<'_>,
    ) -> Result<Self, crate::IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();
        let _ = validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol.clone(),
            n,
            config.mode,
        )?;
        if config.max_step.is_finite() || config.max_step.is_nan() {
            validate_max_step(config.max_step)?;
        }
        if let Some(first) = config.first_step {
            validate_first_step(first, config.t0, config.t_bound)?;
        }

        let direction = if config.t_bound >= config.t0 {
            1.0
        } else {
            -1.0
        };
        let atol_vec = match &config.atol {
            ToleranceValue::Scalar(v) => vec![*v; n],
            ToleranceValue::Vector(v) => v.clone(),
        };

        // SciPy's order: f0 = fun(t0, y0), then select_initial_step(..., order = 3), whose
        // probe evaluation counts toward nfev.
        let y0 = config.y0.to_vec();
        let f0 = fun(config.t0, &y0);
        validate_rhs_shape(f0.len(), n)?;
        if config.mode == RuntimeMode::Hardened && !f0.iter().all(|value| value.is_finite()) {
            return Err(crate::IntegrateValidationError::NonFiniteF0);
        }
        let mut nfev = 1;

        let h_abs = match config.first_step {
            Some(h) => h,
            None => {
                let request = InitialStepRequest {
                    t0: config.t0,
                    y0: config.y0,
                    t_bound: config.t_bound,
                    max_step: config.max_step,
                    f0: &f0,
                    direction,
                    order: 3.0,
                    rtol: config.rtol,
                    atol: config.atol.clone(),
                    mode: config.mode,
                };
                let mut counted = |t: f64, y: &[f64]| {
                    nfev += 1;
                    fun(t, y)
                };
                select_initial_step(&mut counted, &request)?
            }
        };

        // SciPy's Radau.__init__ takes the first Jacobian here (njev = 1, its evaluations
        // not counted in nfev) and marks it current.
        let mut jac_factor = None;
        let jac = num_jac(fun, config.t0, &y0, &f0, &atol_vec, &mut jac_factor);
        let jac_diagonal = diagonal_jacobian_entries(&jac);

        Ok(Self {
            n,
            t: config.t0,
            y: y0,
            t_bound: config.t_bound,
            direction,
            h_abs,
            h_abs_old: None,
            error_norm_old: None,
            max_step: config.max_step,
            rtol: config.rtol,
            atol: atol_vec,
            mode: config.mode,
            state: OdeSolverState::Running,
            tableau: Tableau::scipy(),
            nfev,
            njev: 1,
            nlu: 0,
            f: f0,
            t_old: None,
            y_old: None,
            jac,
            jac_factor,
            jac_diagonal,
            current_jac: true,
            factors: None,
            sol: None,
        })
    }

    pub fn t(&self) -> f64 {
        self.t
    }
    pub fn y(&self) -> &[f64] {
        &self.y
    }
    pub fn f(&self) -> &[f64] {
        &self.f
    }
    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }
    pub fn y_old(&self) -> Option<&[f64]> {
        self.y_old.as_deref()
    }
    pub fn nfev(&self) -> usize {
        self.nfev
    }
    pub fn njev(&self) -> usize {
        self.njev
    }
    pub fn nlu(&self) -> usize {
        self.nlu
    }
    pub fn state(&self) -> OdeSolverState {
        self.state
    }
    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

    /// SciPy's dense output for the last accepted step at `t`; `None` before the first.
    pub fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        self.sol.as_ref().map(|sol| sol.at(t))
    }

    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }
        if self.n == 0 || self.t == self.t_bound {
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.sol = None;
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }
        self.radau_step(fun)
    }

    /// A fresh Jacobian at `(t, y)` with `f = fun(t, y)` (SciPy's `jac_wrapped`).
    fn refresh_jacobian<F>(&mut self, fun: &mut F, t: f64, y: &[f64], f: &[f64])
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        self.jac = num_jac(fun, t, y, f, &self.atol, &mut self.jac_factor);
        self.njev += 1;
        self.jac_diagonal = diagonal_jacobian_entries(&self.jac);
        self.current_jac = true;
    }

    /// SciPy's `Radau._step_impl`.
    #[allow(clippy::needless_range_loop)]
    fn radau_step<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let t = self.t;
        let y = self.y.clone();
        let f = self.f.clone();
        let newton_tol = (10.0 * f64::EPSILON / self.rtol).max(0.03_f64.min(self.rtol.powf(0.5)));

        let spacing = if self.direction > 0.0 {
            t.next_up() - t
        } else {
            t - t.next_down()
        };
        let min_step = 10.0 * spacing.abs();

        // SciPy discards the two-step history when the entry step size had to be clamped.
        let (mut h_abs, h_abs_old, error_norm_old) = if self.h_abs > self.max_step {
            (self.max_step, None, None)
        } else if self.h_abs < min_step {
            (min_step, None, None)
        } else {
            (self.h_abs, self.h_abs_old, self.error_norm_old)
        };

        let mut rejected = false;
        let (t_new, y_new, z, n_iter, rate, error_norm, safety) = loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let mut h = h_abs * self.direction;
            let mut t_new = t + h;
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
            }
            h = t_new - t;
            h_abs = h.abs();

            // Newton start from the previous step's collocation polynomial.
            let z0: [Vec<f64>; 3] = match &self.sol {
                None => [vec![0.0; n], vec![0.0; n], vec![0.0; n]],
                Some(sol) => std::array::from_fn(|i| {
                    sol.at(t + h * self.tableau.c[i])
                        .iter()
                        .zip(&y)
                        .map(|(s, yj)| s - yj)
                        .collect()
                }),
            };
            let scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + y[j].abs() * self.rtol)
                .collect();

            let mut solved = None;
            loop {
                if self.factors.is_none() {
                    let rescan = RADAU_FORCE_DIAGONAL_RESCAN.load(Ordering::Relaxed);
                    let rescanned;
                    let diagonal = if rescan {
                        rescanned = diagonal_jacobian_entries(&self.jac);
                        rescanned.as_deref()
                    } else {
                        self.jac_diagonal.as_deref()
                    };
                    if diagonal.is_some() {
                        RADAU_DIAG_NEWTON_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                    self.factors = Some(NewtonFactors::build(
                        &self.jac,
                        diagonal,
                        self.tableau.mu_real,
                        h,
                    ));
                    self.nlu += 2;
                }
                let (converged, n_iter, z, rate) =
                    self.solve_collocation(fun, t, &y, h, &z0, &scale, newton_tol);
                if converged {
                    solved = Some((n_iter, z, rate));
                    break;
                }
                if self.current_jac {
                    break;
                }
                self.refresh_jacobian(fun, t, &y, &f);
                self.factors = None;
            }
            let Some((n_iter, z, rate)) = solved else {
                h_abs *= 0.5;
                self.factors = None;
                continue;
            };

            let y_new: Vec<f64> = (0..n).map(|j| y[j] + z[2][j]).collect();
            let e = self.tableau.e;
            let ze: Vec<f64> = (0..n)
                .map(|j| (z[0][j] * e[0] + z[1][j] * e[1] + z[2][j] * e[2]) / h)
                .collect();
            let factors = self.factors.as_ref().expect("factors built above");
            let mut error = DVector::from_iterator(n, (0..n).map(|j| f[j] + ze[j]));
            let mut error_ok = factors.solve_real(&mut error);
            let scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + y[j].abs().max(y_new[j].abs()) * self.rtol)
                .collect();
            let mut error_norm = if error_ok {
                scaled_rms(error.iter(), &scale, n)
            } else {
                f64::INFINITY
            };
            let safety = newton_safety(n_iter);
            if rejected && error_norm > 1.0 && error_ok {
                let y_err: Vec<f64> = (0..n).map(|j| y[j] + error[j]).collect();
                let f_err = fun(t, &y_err);
                self.nfev += 1;
                let factors = self.factors.as_ref().expect("factors built above");
                error = DVector::from_iterator(n, (0..n).map(|j| f_err[j] + ze[j]));
                error_ok = factors.solve_real(&mut error);
                error_norm = if error_ok {
                    scaled_rms(error.iter(), &scale, n)
                } else {
                    f64::INFINITY
                };
            }

            if error_norm > 1.0 {
                let factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old);
                h_abs *= MIN_FACTOR.max(safety * factor);
                self.factors = None;
                rejected = true;
            } else {
                break (t_new, y_new, z, n_iter, rate, error_norm, safety);
            }
        };

        let recompute_jac = n_iter > 2 && rate.is_some_and(|r| r > JAC_REFRESH_RATE);
        let mut factor =
            MAX_FACTOR.min(safety * predict_factor(h_abs, h_abs_old, error_norm, error_norm_old));
        if !recompute_jac && factor < 1.2 {
            factor = 1.0;
        } else {
            self.factors = None;
        }

        let f_new = fun(t_new, &y_new);
        self.nfev += 1;
        if recompute_jac {
            self.refresh_jacobian(fun, t_new, &y_new, &f_new);
        } else {
            self.current_jac = false;
        }

        self.h_abs_old = Some(self.h_abs);
        self.error_norm_old = Some(error_norm);
        self.h_abs = h_abs * factor;

        let p = self.tableau.p;
        let q: Vec<[f64; 3]> = (0..n)
            .map(|j| {
                std::array::from_fn(|k| z[0][j] * p[0][k] + z[1][j] * p[1][k] + z[2][j] * p[2][k])
            })
            .collect();
        self.sol = Some(RadauDense {
            t_old: t,
            t: t_new,
            y_old: y.clone(),
            q,
        });
        self.t_old = Some(t);
        self.y_old = Some(y);
        self.t = t_new;
        self.y = y_new;
        self.f = f_new;

        let state = if self.direction * (self.t - self.t_bound) >= 0.0 {
            self.state = OdeSolverState::Finished;
            OdeSolverState::Finished
        } else {
            OdeSolverState::Running
        };
        Ok(StepOutcome {
            message: None,
            state,
        })
    }

    /// SciPy's `solve_collocation_system`: simplified Newton in `W = T⁻¹Z` from the start
    /// `z0`. Returns `(converged, iterations, Z, last contraction rate)`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn solve_collocation<F>(
        &mut self,
        fun: &mut F,
        t: f64,
        y: &[f64],
        h: f64,
        z0: &[Vec<f64>; 3],
        scale: &[f64],
        tol: f64,
    ) -> (bool, usize, [Vec<f64>; 3], Option<f64>)
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let factors = self.factors.as_ref().expect("factors built before Newton");
        let m_real = self.tableau.mu_real / h;
        let m_complex = Complex::new(MU_COMPLEX.re / h, MU_COMPLEX.im / h);
        let ch = [
            h * self.tableau.c[0],
            h * self.tableau.c[1],
            h * self.tableau.c[2],
        ];

        let mut w: [Vec<f64>; 3] = std::array::from_fn(|p| {
            (0..n)
                .map(|j| {
                    RADAU_TI[p][0] * z0[0][j]
                        + RADAU_TI[p][1] * z0[1][j]
                        + RADAU_TI[p][2] * z0[2][j]
                })
                .collect()
        });
        let mut z = z0.clone();

        let force_alloc = RADAU_FORCE_PER_ITER_ALLOC.load(Ordering::Relaxed);
        let mut yi = vec![0.0; n];
        let mut stage: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut f_real = DVector::<f64>::zeros(n);
        let mut f_complex = DVector::<Complex<f64>>::zeros(n);

        let mut dw_norm_old: Option<f64> = None;
        let mut rate: Option<f64> = None;
        let mut converged = false;
        let mut iterations = 0;
        for k in 0..NEWTON_MAXITER {
            iterations = k + 1;
            if force_alloc {
                yi = vec![0.0; n];
                f_real = DVector::<f64>::zeros(n);
                f_complex = DVector::<Complex<f64>>::zeros(n);
            }
            let mut finite = true;
            for i in 0..3 {
                for j in 0..n {
                    yi[j] = y[j] + z[i][j];
                }
                let fi = fun(t + ch[i], &yi);
                self.nfev += 1;
                finite &= fi.iter().all(|v| v.is_finite());
                stage[i] = fi;
            }
            if !finite {
                break;
            }

            for j in 0..n {
                let (s0, s1, s2) = (stage[0][j], stage[1][j], stage[2][j]);
                let fr = s0 * RADAU_TI[0][0] + s1 * RADAU_TI[0][1] + s2 * RADAU_TI[0][2];
                f_real[j] = fr - m_real * w[0][j];
                let fc = Complex::new(
                    s0 * RADAU_TI[1][0] + s1 * RADAU_TI[1][1] + s2 * RADAU_TI[1][2],
                    s0 * RADAU_TI[2][0] + s1 * RADAU_TI[2][1] + s2 * RADAU_TI[2][2],
                );
                f_complex[j] = fc - m_complex * Complex::new(w[1][j], w[2][j]);
            }
            if !factors.solve_real(&mut f_real) || !factors.solve_complex(&mut f_complex) {
                break;
            }

            let dw = [
                f_real.iter().copied().collect::<Vec<f64>>(),
                f_complex.iter().map(|c| c.re).collect(),
                f_complex.iter().map(|c| c.im).collect(),
            ];
            let dw_norm = scaled_rms(dw.iter().flatten(), scale, 3 * n);
            if let Some(old) = dw_norm_old {
                rate = Some(dw_norm / old);
            }
            if let Some(r) = rate
                && (r >= 1.0 || r.powi((NEWTON_MAXITER - k) as i32) / (1.0 - r) * dw_norm > tol)
            {
                break;
            }

            for p in 0..3 {
                for j in 0..n {
                    w[p][j] += dw[p][j];
                }
            }
            for p in 0..3 {
                for j in 0..n {
                    z[p][j] =
                        RADAU_T[p][0] * w[0][j] + RADAU_T[p][1] * w[1][j] + RADAU_T[p][2] * w[2][j];
                }
            }

            if dw_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dw_norm < tol) {
                // status: SciPy's contraction test on the Newton increments
                converged = true;
                break;
            }
            dw_norm_old = Some(dw_norm);
        }
        (converged, iterations, z, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_real_newton_matrix_matches_identity_shift_expression() {
        let jac = DMatrix::<f64>::from_row_slice(
            3,
            3,
            &[1.25, -0.0, -3.5, 2.0, -4.0, 0.75, -1.0, 5.0, 6.25],
        );
        let mu_real = 3.637_834_252_744_496;
        let h = 0.0375;
        let original = DMatrix::<f64>::identity(3, 3) * (mu_real / h) - &jac;
        let candidate = dense_real_newton_matrix(&jac, mu_real, h);

        for (index, (&expected, &actual)) in original.iter().zip(candidate.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "shifted Newton matrix changed at flat index {index}"
            );
        }
    }

    // SciPy's module constants (pinned SciPy 1.17.1: `6 ** 0.5` and the expressions in
    // `_ivp/radau.py`), to the bit.
    #[test]
    fn tableau_constants_are_scipys_doubles() {
        let tab = Tableau::scipy();
        assert_eq!(tab.mu_real, 3.637_834_252_744_496);
        assert_eq!(
            tab.c,
            [0.155_051_025_721_682_22, 0.644_948_974_278_317_8, 1.0]
        );
        assert_eq!(
            tab.e,
            [
                -10.048_809_399_827_414,
                1.382_142_733_160_748,
                -0.333_333_333_333_333_3
            ]
        );
        assert_eq!(
            tab.p,
            [
                [
                    10.048_809_399_827_414,
                    -25.629_591_447_076_64,
                    15.580_782_047_249_224
                ],
                [
                    -1.382_142_733_160_748,
                    10.296_258_113_743_303,
                    -8.914_115_380_582_556
                ],
                [
                    0.333_333_333_333_333_3,
                    -2.666_666_666_666_666_5,
                    3.333_333_333_333_333_5
                ],
            ]
        );
    }

    // The exact-diagonal factors must solve like the dense LU of the same diagonal
    // matrices (both are one division per component), and a changed diagonal must be
    // visible to the comparison (must-miss arm).
    #[test]
    fn diagonal_newton_factors_solve_like_the_dense_lu() {
        let diagonal = [-1.5, 0.25, -1000.0, 3.0];
        let jac = DMatrix::from_diagonal(&DVector::from_row_slice(&diagonal));
        let (mu_real, h) = (Tableau::scipy().mu_real, 0.013);
        let fast = NewtonFactors::build(&jac, Some(&diagonal), mu_real, h);
        let dense = NewtonFactors::build(&jac, None, mu_real, h);
        let rhs_real = DVector::from_row_slice(&[1.0, -2.5, 7.0, 0.125]);
        let rhs_complex = DVector::from_iterator(
            4,
            [(1.0, 2.0), (-0.5, 0.0), (3.0, -4.0), (0.0, 1e-3)]
                .into_iter()
                .map(|(re, im)| Complex::new(re, im)),
        );
        let (mut fast_real, mut dense_real) = (rhs_real.clone(), rhs_real.clone());
        assert!(fast.solve_real(&mut fast_real) && dense.solve_real(&mut dense_real));
        assert_eq!(fast_real, dense_real);
        let (mut fast_complex, mut dense_complex) = (rhs_complex.clone(), rhs_complex.clone());
        assert!(fast.solve_complex(&mut fast_complex) && dense.solve_complex(&mut dense_complex));
        assert_eq!(fast_complex, dense_complex);

        let other = NewtonFactors::build(&jac, Some(&[-1.5, 0.25, -999.0, 3.0]), mu_real, h);
        let mut other_real = rhs_real.clone();
        assert!(other.solve_real(&mut other_real));
        assert_ne!(
            other_real, dense_real,
            "the comparison cannot see a changed diagonal entry"
        );
    }

    #[test]
    fn radau_first_step_hardened_rejects_non_finite_f0() {
        let mut fun = |_t: f64, _y: &[f64]| vec![f64::INFINITY];
        let config = RadauSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.1,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(1e-6),
            mode: RuntimeMode::Hardened,
        };

        assert!(matches!(
            RadauSolver::new(&mut fun, config),
            Err(crate::IntegrateValidationError::NonFiniteF0)
        ));
    }

    // A step that lands one ulp short of t_bound leaves a final step (1.1e-16) below
    // min_step = 10 ulp(t). SciPy takes it; the solver used to fail with StepSizeTooSmall
    // there (observed on Van der Pol mu = 100, rtol 1e-4). y' = 0 makes the first step's
    // error exactly 0, so it is accepted wherever it lands.
    #[test]
    fn radau_takes_a_final_step_shorter_than_min_step() {
        let mut zero = |_t: f64, _y: &[f64]| vec![0.0];
        let config = RadauSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(1.0 - f64::EPSILON / 2.0),
            mode: RuntimeMode::Strict,
        };
        let mut solver = RadauSolver::new(&mut zero, config).expect("Radau init");
        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut zero).expect("Radau step");
            steps += 1;
        }
        assert_eq!(solver.state(), OdeSolverState::Finished);
        assert_eq!(solver.t(), 1.0);
        assert_eq!(steps, 2, "the second step is the 1.1e-16 remainder");
    }

    // SciPy's Radau selects its first step with the ORDER-3 exponent and counts the probe:
    // scipy.integrate.Radau(lambda t, y: -y, 0, [1.0], 1.0, rtol=1e-6, atol=1e-8) has
    // h_abs = 0.01002490679314321, nfev = 2 and njev = 1 (pinned SciPy 1.17.1); the
    // Jacobian's evaluations (one column here) are not counted. The BDF selector used
    // before gave (0.01 / d)^(1/2) = 1.0e-4.
    #[test]
    fn radau_initial_step_uses_the_order_three_selector_like_scipy() {
        let mut calls = 0;
        let mut decay = |_t: f64, y: &[f64]| {
            calls += 1;
            vec![-y[0]]
        };
        let config = RadauSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
        };
        let solver = RadauSolver::new(&mut decay, config).expect("Radau init");
        let scipy_h_abs = 0.010_024_906_793_143_21;
        assert!(
            (solver.h_abs - scipy_h_abs).abs() <= 1e-14 * scipy_h_abs,
            "h_abs {} vs SciPy {scipy_h_abs}",
            solver.h_abs
        );
        assert_eq!(solver.nfev(), 2);
        assert_eq!(solver.njev(), 1);
        assert_eq!(calls, 3);
    }

    /// `predict_factor` against hand-evaluated SciPy values, including the branch
    /// a naive port drops.
    ///
    /// THE NEGATIVE CASE is the third assertion: with a history where the step
    /// grew and the error fell, the raw multiplier exceeds 1 and the unclamped
    /// prediction would be `2.0 * 16^0.25 * 1^-0.25 = 4.0`. SciPy takes
    /// `min(1, multiplier)`, so the answer must be 1.0.
    #[test]
    fn predict_factor_matches_scipy_two_step_rule_including_the_growth_clamp() {
        // No history: one-step rule, factor = ‖e‖^-1/4 exactly.
        let one_step = predict_factor(0.1, None, 16.0, None);
        assert!(
            (one_step - 0.5).abs() < 1e-15,
            "one-step rule: got {one_step}"
        );
        assert!(
            (predict_factor(0.1, Some(0.05), 16.0, None) - 0.5).abs() < 1e-15,
            "a missing error history must also fall back to the one-step rule"
        );

        // Two-step, shrinking multiplier: h halved, error unchanged.
        let damped = predict_factor(0.05, Some(0.1), 1.0, Some(1.0));
        assert!((damped - 0.5).abs() < 1e-15, "two-step rule: got {damped}");

        // NEGATIVE CASE: multiplier = 0.1/0.05 * (16/1)^0.25 = 4 > 1, clamped to 1.
        let clamped = predict_factor(0.1, Some(0.05), 1.0, Some(16.0));
        assert!(
            (clamped - 1.0).abs() < 1e-15,
            "min(1, multiplier) clamp is missing: got {clamped}, unclamped would be 4.0"
        );

        // Zero error norm degrades to the one-step rule (SciPy's `error_norm == 0`
        // guard) and is capped by the caller, not here.
        assert!(predict_factor(0.1, Some(0.05), 0.0, Some(1.0)).is_infinite());

        // SciPy's safety is 0.9 exactly for a single Newton iteration and decreases
        // from there.
        assert!((newton_safety(1) - 0.9).abs() < 1e-15);
        assert!(newton_safety(6) < newton_safety(2));
        assert!(newton_safety(NEWTON_MAXITER) > 0.0);
    }

    /// Every test that writes a `RADAU_FORCE_*` toggle or reads a global hit counter
    /// takes this, so concurrent tests cannot leak a toggle into another's arm.
    static TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// n components each decaying at 1 + 10 j and coupled through 1e-3 × their mean:
    /// a dense Jacobian, rates spanning three decades.
    fn run_coupled_decay(n: usize) -> (Vec<f64>, usize, usize, usize, usize) {
        let decay: Vec<f64> = (0..n).map(|j| 1.0 + 10.0 * j as f64).collect();
        let y0: Vec<f64> = (0..n).map(|j| 1.0 + 0.25 * ((j % 7) as f64)).collect();
        let mut fun = |_t: f64, y: &[f64]| {
            let mean = y.iter().sum::<f64>() / n as f64;
            (0..n).map(|j| -decay[j] * y[j] + 1e-3 * mean).collect()
        };
        let config = RadauSolverConfig {
            t0: 0.0,
            y0: &y0,
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: Some(1e-4),
            mode: RuntimeMode::Strict,
        };
        let mut solver = RadauSolver::new(&mut fun, config).expect("construct Radau solver");
        let mut steps = 0usize;
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("dense Radau step");
            steps += 1;
            assert!(steps < 100_000, "Radau failed to reach the bound");
        }
        (
            solver.y().to_vec(),
            steps,
            solver.nfev(),
            solver.njev(),
            solver.nlu(),
        )
    }

    /// SciPy's counts on the dense fixture, measured with pinned SciPy 1.17.1:
    /// `Radau(fun, 0, y0, 1, rtol=1e-8, atol=1e-10, first_step=1e-4)` stepped to the end,
    /// with the same right-hand side (Python's sequential `sum` for the mean). n = 48:
    /// 592 steps, nfev 4151, njev 2, nlu 60. n = 8: 433 steps, nfev 3038, njev 2, nlu 42.
    /// The shipped elementary controller this port replaced took 558 steps at n = 48 and
    /// its Newton started from zero, so none of these counts matched.
    #[test]
    fn dense_radau_step_sequence_matches_scipy() {
        let _guard = TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (n, steps, nfev, njev, nlu, y0_end) in [
            (48, 592, 4151, 2, 60, 0.367_892_689_064_116_65),
            (8, 433, 3038, 2, 42, 0.367_944_253_934_249_45),
        ] {
            let (y, got_steps, got_nfev, got_njev, got_nlu) = run_coupled_decay(n);
            assert_eq!(
                (got_steps, got_nfev, got_njev, got_nlu),
                (steps, nfev, njev, nlu),
                "n = {n}: (steps, nfev, njev, nlu) differ from SciPy's"
            );
            assert!(
                (y[0] - y0_end).abs() <= 1e-12,
                "n = {n}: y[0] = {} vs SciPy {y0_end}",
                y[0]
            );
        }
    }

    /// Both A/B toggles leave the trajectory bit-identical: the per-iteration scratch
    /// allocation and the diagonal rescan. The fixture's Jacobian is exactly diagonal so
    /// the rescan arm has something to rescan (execution proof: the diagonal-hit counter
    /// moves).
    #[test]
    fn scratch_and_rescan_toggles_leave_the_trajectory_bit_identical() {
        let _guard = TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let run = || {
            let mut fun = |t: f64, y: &[f64]| vec![-50.0 * y[0] + t.cos(), -0.5 * y[1]];
            let config = RadauSolverConfig {
                t0: 0.0,
                y0: &[1.0, 2.0],
                t_bound: 3.0,
                rtol: 1e-7,
                atol: ToleranceValue::Scalar(1e-9),
                max_step: f64::INFINITY,
                first_step: None,
                mode: RuntimeMode::Strict,
            };
            let mut solver = RadauSolver::new(&mut fun, config).expect("Radau init");
            while solver.state() == OdeSolverState::Running {
                solver.step_with(&mut fun).expect("Radau step");
            }
            (
                solver.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                solver.nfev(),
                solver.nlu(),
            )
        };
        let hits_before = RADAU_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        let base = run();
        assert!(RADAU_DIAG_NEWTON_HITS.load(Ordering::Relaxed) > hits_before);

        RADAU_FORCE_PER_ITER_ALLOC.store(true, Ordering::Relaxed);
        let alloc = run();
        RADAU_FORCE_PER_ITER_ALLOC.store(false, Ordering::Relaxed);
        RADAU_FORCE_DIAGONAL_RESCAN.store(true, Ordering::Relaxed);
        let rescan = run();
        RADAU_FORCE_DIAGONAL_RESCAN.store(false, Ordering::Relaxed);

        assert_eq!(alloc, base);
        assert_eq!(rescan, base);
    }
}
