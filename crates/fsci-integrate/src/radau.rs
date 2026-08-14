#![forbid(unsafe_code)]

//! Radau IIA implicit Runge-Kutta solver for stiff ODEs (3-stage, order 5),
//! matching `scipy.integrate.solve_ivp(method='Radau')`.
//!
//! Implemented as the mathematically-equivalent real `3n×3n` simplified-Newton
//! collocation: solve `Z_i = h Σ_j A_ij f(t + c_j h, y + Z_j)` for the stage
//! corrections `Z`, with the Newton matrix `I_{3n} − h (A ⊗ J)` and a lazily
//! refreshed finite-difference Jacobian `J`. The embedded order-3 error estimate
//! reuses scipy's `(MU_REAL/h) I − J` factor and the `E` coefficients. This yields
//! the same collocation solution as scipy's complex-eigenvalue formulation (only
//! the inner linear algebra differs), so results match scipy to Newton tolerance.

use crate::bdf::select_initial_step_bdf;
use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{
    ToleranceValue, validate_first_step, validate_max_step, validate_rhs_shape, validate_tol,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{Complex, DMatrix, DVector};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const NEWTON_MAXITER: usize = 6;

/// When `true`, the simplified-Newton corrector re-allocates its `yi`/`rhs`/`dz_scaled`
/// scratch (and zero-fills `fstage`) on EVERY iteration — the ORIG behaviour, kept only
/// for the same-binary A/B. When `false` (default), those buffers are hoisted above the
/// Newton loop and reused (every entry is overwritten before use, so the trajectory is
/// bit-identical). Stiff correctors run thousands of times per integration and small-n
/// stiff systems are malloc-bound, so eliminating the per-iteration allocations is a
/// monotone win. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_FORCE_PER_ITER_ALLOC: AtomicBool = AtomicBool::new(false);
/// When `true`, rediscover diagonal Jacobians by scanning the unchanged dense
/// matrix on every step. This preserves the original path for same-binary A/B;
/// production caches the structural certificate when the Jacobian is refreshed.
/// `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_FORCE_DIAGONAL_RESCAN: AtomicBool = AtomicBool::new(false);
/// When `true`, restore the original dense-Radau policy that changes the
/// accepted step size on every controller update and rebuilds both Newton
/// factors on the next step. The default retains a dense LU pair when the
/// controller deliberately holds the step size. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_FORCE_DENSE_LU_REBUILD: AtomicBool = AtomicBool::new(false);
/// Count of Radau Newton factorizations that took the exact-diagonal stage and
/// error-solve path. This is execution proof for live-incumbent harnesses: a
/// diagonal fixture reporting zero hits did not exercise the structural lever.
/// Incremented once per attempted factorization, before the Newton iterations.
/// `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_DIAG_NEWTON_HITS: AtomicUsize = AtomicUsize::new(0);
/// Count of accepted held-size dense steps whose real/complex LU pair was
/// consumed by the following step. This is execution proof for the same-ELF
/// completion harness. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_DENSE_LU_REUSE_HITS: AtomicUsize = AtomicUsize::new(0);
/// Opt in to scipy's two-step predictive step-size controller (Hairer & Wanner II,
/// Sec. IV.8) with its iteration-damped safety factor and post-acceptance Jacobian
/// refresh, in place of the shipped elementary rule `min(MAX, 0.9·‖e‖^-1/4)`.
///
/// **DEFAULT OFF — the lever was MEASURED and REJECTED (2026-08-14, RosePelican).**
/// The hypothesis was that the elementary controller's optimistic growth fails the
/// `factor < 1.2` hold test too often, discarding the dense LU pair and driving the
/// counted-factorization gap that `docs/perf_ledger_cc.md`'s dense-Radau n=128 row
/// attributes 16x of its 0.4188x loss to. Same-process A/B on the harness's dense
/// fixture family (n=48, rates 1+10i, 1e-3 mean coupling, rtol 1e-8): the two arms
/// produced **identical** factorization counts (`nlu=60` both) and the predictive
/// arm took MORE steps (592 vs 558). The premise was wrong — LU reuse was already
/// near-perfect (30 factorization events across ~570 steps).
///
/// Retained, not deleted, because it is validated-correct and is the natural
/// starting point for anyone re-attacking the controller; wiring it on is a
/// measured loss on this fixture family. `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_ENABLE_PREDICTIVE_CONTROLLER: AtomicBool = AtomicBool::new(false);
/// Count of accepted steps whose predicted factor was held at 1 by the controller,
/// i.e. the steps that make LU reuse possible at all. Execution proof: a candidate
/// arm reporting zero holds never exercised the predictive controller.
/// `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_HELD_STEP_HITS: AtomicUsize = AtomicUsize::new(0);
/// Count of accepted steps that triggered a post-acceptance Jacobian refresh
/// (`n_iter > 2 && rate > 1e-3`). `#[doc(hidden)]`.
#[doc(hidden)]
pub static RADAU_POST_STEP_JAC_REFRESH_HITS: AtomicUsize = AtomicUsize::new(0);
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 8.0;
const ERR_EXP: f64 = -0.25; // embedded estimator is order 3 → 1/(3+1).
/// Contraction rate above which scipy refreshes the Jacobian after an accepted
/// step (`scipy/integrate/_ivp/radau.py`, `recompute_jac`).
const JAC_REFRESH_RATE: f64 = 1e-3;

/// scipy's `predict_factor`: the two-step step-size prediction of Hairer &
/// Wanner II, Sec. IV.8, falling back to the one-step rule when no previous step
/// is on record.
///
/// The `min(1, multiplier)` clamp is the load-bearing part. It caps growth by the
/// ratio the PREVIOUS step actually achieved, so a solver that is already at its
/// comfortable step size predicts a factor near 1 instead of the elementary
/// rule's optimistic `0.9·‖e‖^-1/4`. Predicting near 1 is what lets the caller
/// hold the step size, and holding the step size is what lets the dense
/// real/complex LU pair survive into the next step instead of being rebuilt.
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

/// scipy's iteration-damped safety factor: a step that needed many Newton
/// iterations is trusted less.
fn newton_safety(n_iter: usize) -> f64 {
    let maxiter = NEWTON_MAXITER as f64;
    0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + n_iter as f64)
}

// scipy's Radau IIA eigen-transform constants (`scipy/integrate/_ivp/radau.py`).
// `MU_REAL` (see `RadauSolver::new`) and `MU_COMPLEX` are the eigenvalues of the
// inverse collocation matrix A⁻¹; `T`/`TI = T⁻¹` are the real similarity transform
// that block-diagonalises A. They let the dense Newton matrix `I_{3n} − h(A⊗J)` be
// solved as one real `n×n` factor `(MU_REAL/h)I − J` plus one complex `n×n` factor
// `(MU_COMPLEX/h)I − J`, instead of a full `3n×3n` LU (~5× less work for large n).
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

type DenseRealLu = nalgebra::linalg::LU<f64, nalgebra::Dyn, nalgebra::Dyn>;
type DenseComplexLu = nalgebra::linalg::LU<Complex<f64>, nalgebra::Dyn, nalgebra::Dyn>;

struct DenseLuPair {
    /// Signed step used to build both factors. The next step may consume the
    /// pair only when its untruncated requested step has the same bits.
    h: f64,
    real: DenseRealLu,
    complex: DenseComplexLu,
}

/// Solve the Radau dense Newton system `(I_{3n} − h(A⊗J)) dz = rhs` via scipy's
/// eigen-decoupling: transform `rhs` by `TI`, solve the real block with `lu_real`
/// = `(MU_REAL/h)I − J` and the complex block with `lu_complex` = `(MU_COMPLEX/h)
/// I − J`, then transform back by `T`. Mathematically identical to a full `3n×3n`
/// LU solve (verified byte-close), at ~one real + one complex `n×n` solve.
#[allow(clippy::too_many_arguments)]
fn solve_collocation_decoupled(
    rhs: &DVector<f64>,
    mu_real: f64,
    h: f64,
    lu_real: &nalgebra::linalg::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
    lu_complex: &nalgebra::linalg::LU<Complex<f64>, nalgebra::Dyn, nalgebra::Dyn>,
    n: usize,
) -> Option<Vec<f64>> {
    let mut g0 = DVector::<f64>::zeros(n);
    let mut gc = DVector::<Complex<f64>>::zeros(n);
    for i in 0..n {
        let (r0, r1, r2) = (rhs[i], rhs[n + i], rhs[2 * n + i]);
        g0[i] = RADAU_TI[0][0] * r0 + RADAU_TI[0][1] * r1 + RADAU_TI[0][2] * r2;
        let re = RADAU_TI[1][0] * r0 + RADAU_TI[1][1] * r1 + RADAU_TI[1][2] * r2;
        let im = RADAU_TI[2][0] * r0 + RADAU_TI[2][1] * r1 + RADAU_TI[2][2] * r2;
        gc[i] = Complex::new(re, im);
    }
    let w0 = lu_real.solve(&g0)?;
    let wc = lu_complex.solve(&gc)?;
    let sr = mu_real / h;
    let sc = MU_COMPLEX / Complex::new(h, 0.0);
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        let dw0 = sr * w0[i];
        let dwc = sc * wc[i];
        for p in 0..3 {
            out[p * n + i] = RADAU_T[p][0] * dw0 + RADAU_T[p][1] * dwc.re + RADAU_T[p][2] * dwc.im;
        }
    }
    Some(out)
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

fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f64).sqrt()
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

fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for pivot_col in 0..3 {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col][pivot_col].abs();
        for (row, values) in matrix.iter().enumerate().skip(pivot_col + 1) {
            let candidate_abs = values[pivot_col].abs();
            if candidate_abs > pivot_abs {
                pivot_row = row;
                pivot_abs = candidate_abs;
            }
        }
        if pivot_abs == 0.0 || !pivot_abs.is_finite() {
            return None;
        }
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = matrix[pivot_col][pivot_col];
        let pivot_values = matrix[pivot_col];
        let pivot_rhs = rhs[pivot_col];
        for (row, row_values) in matrix.iter_mut().enumerate().skip(pivot_col + 1) {
            let factor = row_values[pivot_col] / pivot;
            row_values[pivot_col] = 0.0;
            for (col, value) in row_values.iter_mut().enumerate().skip(pivot_col + 1) {
                *value -= factor * pivot_values[col];
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }

    let mut out = [0.0; 3];
    for row in (0..3).rev() {
        let mut value = rhs[row];
        for (col, &out_col) in out.iter().enumerate().skip(row + 1) {
            value -= matrix[row][col] * out_col;
        }
        out[row] = value / matrix[row][row];
    }
    Some(out)
}

fn solve_collocation_diagonal(
    diagonal: &[f64],
    h: f64,
    tableau_a: &[[f64; 3]; 3],
    rhs: &DVector<f64>,
) -> Option<Vec<f64>> {
    let n = diagonal.len();
    let mut out = vec![0.0; 3 * n];
    for (j, &lambda) in diagonal.iter().enumerate() {
        let mut block = [[0.0; 3]; 3];
        for i in 0..3 {
            for l in 0..3 {
                block[i][l] = -h * tableau_a[i][l] * lambda;
                if i == l {
                    block[i][l] += 1.0;
                }
            }
        }
        let solved = solve_3x3(block, [rhs[j], rhs[n + j], rhs[2 * n + j]])?;
        out[j] = solved[0];
        out[n + j] = solved[1];
        out[2 * n + j] = solved[2];
    }
    Some(out)
}

fn solve_real_diagonal(
    diagonal: &[f64],
    h: f64,
    mu_real: f64,
    rhs: &DVector<f64>,
) -> Option<Vec<f64>> {
    let shift = mu_real / h;
    let mut out = Vec::with_capacity(diagonal.len());
    for (j, &lambda) in diagonal.iter().enumerate() {
        let denom = shift - lambda;
        if denom == 0.0 || !denom.is_finite() {
            return None;
        }
        out.push(rhs[j] / denom);
    }
    Some(out)
}

/// Radau IIA solver state.
pub struct RadauSolver {
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_bound: f64,
    direction: f64,
    h: f64,
    max_step: f64,
    rtol: f64,
    atol: Vec<f64>,
    mode: RuntimeMode,
    state: OdeSolverState,

    // Radau IIA tableau (3-stage, order 5).
    c: [f64; 3],
    a: [[f64; 3]; 3],
    e: [f64; 3],
    mu_real: f64,

    nfev: usize,
    njev: usize,
    nlu: usize,

    f: Vec<f64>,
    f_old: Option<Vec<f64>>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,

    // Lazy Jacobian / factorization caches.
    jac: Option<DMatrix<f64>>,
    /// Exact diagonal entries derived alongside `jac`; `None` means the cached
    /// Jacobian is structurally dense (or no Jacobian has been built yet).
    jac_diagonal: Option<Vec<f64>>,
    /// Dense Newton factors retained only across an accepted held-size step.
    dense_lu: Option<DenseLuPair>,
    /// Previous step's entry step size and error norm, feeding scipy's two-step
    /// predictive controller. `None` until a step has been accepted, or whenever
    /// the entry step size was clamped (scipy discards the history in that case).
    h_abs_old: Option<f64>,
    error_norm_old: Option<f64>,
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

        let h_mag = match config.first_step {
            Some(h) => h,
            None => select_initial_step_bdf(
                fun,
                config.t0,
                config.y0,
                direction,
                config.rtol,
                &atol_vec,
                config.mode,
            )?
            .min(config.max_step),
        };
        let h = h_mag * direction;

        let y0 = config.y0.to_vec();
        let f0 = fun(config.t0, &y0);
        validate_rhs_shape(f0.len(), n)?;
        if config.mode == RuntimeMode::Hardened && !f0.iter().all(|value| value.is_finite()) {
            return Err(crate::IntegrateValidationError::NonFiniteF0);
        }

        let s6 = 6.0_f64.sqrt();
        let c = [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0];
        let a = [
            [
                (88.0 - 7.0 * s6) / 360.0,
                (296.0 - 169.0 * s6) / 1800.0,
                (-2.0 + 3.0 * s6) / 225.0,
            ],
            [
                (296.0 + 169.0 * s6) / 1800.0,
                (88.0 + 7.0 * s6) / 360.0,
                (-2.0 - 3.0 * s6) / 225.0,
            ],
            [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
        ];
        let e = [
            (-13.0 - 7.0 * s6) / 3.0,
            (-13.0 + 7.0 * s6) / 3.0,
            -1.0 / 3.0,
        ];
        let mu_real = 3.0 + 3.0_f64.powf(2.0 / 3.0) - 3.0_f64.powf(1.0 / 3.0);

        Ok(Self {
            n,
            t: config.t0,
            y: y0,
            t_bound: config.t_bound,
            direction,
            h,
            max_step: config.max_step,
            rtol: config.rtol,
            atol: atol_vec,
            mode: config.mode,
            state: OdeSolverState::Running,
            c,
            a,
            e,
            mu_real,
            nfev: 1,
            njev: 0,
            nlu: 0,
            f: f0,
            f_old: None,
            t_old: None,
            y_old: None,
            jac: None,
            jac_diagonal: None,
            dense_lu: None,
            h_abs_old: None,
            error_norm_old: None,
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
    pub fn f_old(&self) -> Option<&[f64]> {
        self.f_old.as_deref()
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
            self.dense_lu = None;
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }
        self.radau_step(fun)
    }

    fn compute_jacobian<F>(&mut self, fun: &mut F, t: f64, y: &[f64], f0: &[f64]) -> DMatrix<f64>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let eps = f64::EPSILON.sqrt();
        let mut jac = DMatrix::<f64>::zeros(self.n, self.n);
        let mut yp = y.to_vec();
        for col in 0..self.n {
            let perturb = eps * y[col].abs().max(1.0);
            yp[col] += perturb;
            let fp = fun(t, &yp);
            self.nfev += 1;
            for row in 0..self.n {
                jac[(row, col)] = (fp[row] - f0[row]) / perturb;
            }
            yp[col] = y[col];
        }
        self.njev += 1;
        jac
    }

    #[allow(clippy::needless_range_loop)]
    fn radau_step<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let newton_tol = (10.0 * f64::EPSILON / self.rtol).max(0.03_f64.min(self.rtol.sqrt()));

        let spacing = if self.direction > 0.0 {
            self.t.next_up() - self.t
        } else {
            self.t - self.t.next_down()
        };
        let min_step = 10.0 * spacing.abs();

        // scipy discards the two-step history whenever the entry step size had to
        // be clamped, because the previous (h, ‖e‖) pair no longer describes the
        // step actually about to be taken.
        let entry_h_abs = self.h.abs();
        let clamped = entry_h_abs > self.max_step || entry_h_abs < min_step;
        let (h_abs_old, error_norm_old) = if clamped {
            (None, None)
        } else {
            (self.h_abs_old, self.error_norm_old)
        };
        let mut h_abs = entry_h_abs.min(self.max_step).max(min_step);
        let mut rejected = false;
        let predictive_controller = RADAU_ENABLE_PREDICTIVE_CONTROLLER.load(Ordering::Relaxed);
        let force_dense_rebuild = RADAU_FORCE_DENSE_LU_REBUILD.load(Ordering::Relaxed);
        let mut retained_dense_lu = if force_dense_rebuild {
            self.dense_lu = None;
            None
        } else {
            self.dense_lu.take()
        };

        loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let requested_h = h_abs * self.direction;
            let mut h = requested_h;
            let mut t_new = self.t + h;
            let reached_bound = self.direction * (t_new - self.t_bound) > 0.0;
            if reached_bound {
                t_new = self.t_bound;
            }
            h = t_new - self.t;
            h_abs = h.abs();
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }

            // Ensure a Jacobian (reused across steps; recomputed only on Newton
            // failure). Dense factors can survive exactly one accepted held-size
            // step; every other transition drops them before this point.
            let jac_fresh = self.jac.is_none();
            if jac_fresh {
                retained_dense_lu = None;
                let f_cur = self.f.clone();
                let y_cur = self.y.clone();
                let jac = self.compute_jacobian(fun, self.t, &y_cur, &f_cur);
                self.jac_diagonal = diagonal_jacobian_entries(&jac);
                self.jac = Some(jac);
            }
            // M_3n = I_{3n} − h (A ⊗ J); M_real = (mu_real/h) I − J.
            // Exactly diagonal Jacobians split M_3n into n independent 3x3
            // systems and M_real into scalar solves, avoiding dense assembly
            // and LU while preserving the same simplified-Newton equations.
            let mut lu_real = None;
            let mut lu_complex = None;
            let rescanned_diagonal;
            let diagonal_jac = {
                let jac = self.jac.as_ref().expect("jacobian present");
                let force_rescan = RADAU_FORCE_DIAGONAL_RESCAN.load(Ordering::Relaxed);
                rescanned_diagonal = if force_rescan {
                    diagonal_jacobian_entries(jac)
                } else {
                    None
                };
                let diagonal = if force_rescan {
                    rescanned_diagonal.as_deref()
                } else {
                    self.jac_diagonal.as_deref()
                };
                if diagonal.is_none() {
                    if let Some(pair) = retained_dense_lu
                        .take()
                        .filter(|pair| !reached_bound && pair.h.to_bits() == requested_h.to_bits())
                    {
                        lu_real = Some(pair.real);
                        lu_complex = Some(pair.complex);
                        RADAU_DENSE_LU_REUSE_HITS.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // Dense Jacobian: factor the eigen-decoupled real and complex
                        // n×n blocks `(MU_REAL/h)I − J` and `(MU_COMPLEX/h)I − J`
                        // rather than a full 3n×3n LU. The real factor is also the
                        // one the embedded error estimate reuses.
                        let m_real = DMatrix::<f64>::identity(n, n) * (self.mu_real / h) - jac;
                        let m_complex = DMatrix::<Complex<f64>>::from_fn(n, n, |r, col| {
                            let mut val = -Complex::new(jac[(r, col)], 0.0);
                            if r == col {
                                val += MU_COMPLEX / Complex::new(h, 0.0);
                            }
                            val
                        });
                        lu_real = Some(m_real.lu());
                        lu_complex = Some(m_complex.lu());
                        self.nlu += 2;
                    }
                }
                diagonal
            };
            if diagonal_jac.is_some() {
                RADAU_DIAG_NEWTON_HITS.fetch_add(1, Ordering::Relaxed);
                retained_dense_lu = None;
                self.nlu += 2;
            }

            // Simplified Newton on the stage corrections Z (3 × n), initial guess 0.
            let mut z = vec![vec![0.0; n]; 3];
            let scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + self.rtol * self.y[j].abs())
                .collect();
            let mut converged = false;
            let mut dz_norm_old: Option<f64> = None;
            let mut bad = false;
            // Newton iterations actually spent, and the last observed contraction
            // rate. SciPy's Radau feeds both into the step controller: `n_iter`
            // damps the safety factor and, with `rate`, decides whether the
            // Jacobian is refreshed after an accepted step.
            let mut n_iter = NEWTON_MAXITER;
            let mut last_rate: Option<f64> = None;
            // Newton-corrector scratch hoisted out of the k-loop and reused: every entry
            // of `yi`/`rhs`/`dz_scaled` is overwritten before it is read each iteration, so
            // reuse is bit-identical while dropping ~(3·yi + rhs + dz_scaled + 3·fstage-init)
            // allocations per Newton iteration. `RADAU_FORCE_PER_ITER_ALLOC` restores the
            // per-iteration allocation for the same-binary A/B.
            let force_alloc = RADAU_FORCE_PER_ITER_ALLOC.load(Ordering::Relaxed);
            let mut yi = vec![0.0; n];
            let mut rhs = DVector::<f64>::zeros(3 * n);
            let mut dz_scaled = vec![0.0; 3 * n];
            for k in 0..NEWTON_MAXITER {
                if force_alloc {
                    yi = vec![0.0; n];
                    rhs = DVector::<f64>::zeros(3 * n);
                    dz_scaled = vec![0.0; 3 * n];
                }
                // Stage derivatives F_i = f(t + c_i h, y + Z_i). Each `fstage[i]` is replaced
                // wholesale by the `fun` return below, so start empty (the zero-fill is dead).
                let mut fstage: [Vec<f64>; 3] = if force_alloc {
                    [vec![0.0; n], vec![0.0; n], vec![0.0; n]]
                } else {
                    [Vec::new(), Vec::new(), Vec::new()]
                };
                let mut finite = true;
                for i in 0..3 {
                    for (j, yij) in yi.iter_mut().enumerate() {
                        *yij = self.y[j] + z[i][j];
                    }
                    let fi = fun(t_new - h + self.c[i] * h, &yi);
                    self.nfev += 1;
                    if !fi.iter().all(|v| v.is_finite()) {
                        finite = false;
                    }
                    fstage[i] = fi;
                }
                if !finite {
                    bad = true;
                    break;
                }
                // Residual G_i = Z_i − h Σ_j A_ij F_j  (we solve M·ΔZ = −G).
                for i in 0..3 {
                    for j in 0..n {
                        let mut acc = z[i][j];
                        for l in 0..3 {
                            acc -= h * self.a[i][l] * fstage[l][j];
                        }
                        rhs[i * n + j] = -acc;
                    }
                }
                let dz = if let Some(diagonal) = diagonal_jac {
                    solve_collocation_diagonal(diagonal, h, &self.a, &rhs)
                } else if let (Some(lr), Some(lc)) = (lu_real.as_ref(), lu_complex.as_ref()) {
                    solve_collocation_decoupled(&rhs, self.mu_real, h, lr, lc, n)
                } else {
                    None
                };
                let Some(dz) = dz else {
                    bad = true;
                    break;
                };
                for i in 0..3 {
                    for j in 0..n {
                        let d = dz[i * n + j];
                        z[i][j] += d;
                        dz_scaled[i * n + j] = d / scale[j];
                    }
                }
                let dz_norm = rms_norm(&dz_scaled);
                let rate = dz_norm_old.map(|old| if old > 0.0 { dz_norm / old } else { 0.0 });
                last_rate = rate;
                n_iter = k + 1;
                if let Some(r) = rate
                    && (r >= 1.0
                        || r.powi((NEWTON_MAXITER - k) as i32) / (1.0 - r) * dz_norm > newton_tol)
                {
                    break;
                }
                if dz_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dz_norm < newton_tol) {
                    converged = true;
                    break;
                }
                dz_norm_old = Some(dz_norm);
            }

            if bad || !converged {
                if jac_fresh {
                    h_abs *= 0.5;
                    rejected = true;
                    continue;
                }
                // Stale Jacobian — refresh and retry at the same h.
                self.jac = None;
                self.jac_diagonal = None;
                continue;
            }

            // Solution and embedded error estimate.
            let y_new: Vec<f64> = (0..n).map(|j| self.y[j] + z[2][j]).collect();
            let ze: Vec<f64> = (0..n)
                .map(|j| (self.e[0] * z[0][j] + self.e[1] * z[1][j] + self.e[2] * z[2][j]) / h)
                .collect();
            let err_scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + self.rtol * self.y[j].abs().max(y_new[j].abs()))
                .collect();
            let mut err_rhs = DVector::<f64>::from_iterator(n, (0..n).map(|j| self.f[j] + ze[j]));
            let mut error = if let Some(diagonal) = diagonal_jac {
                solve_real_diagonal(diagonal, h, self.mu_real, &err_rhs)
            } else {
                lu_real
                    .as_ref()
                    .and_then(|lu| lu.solve(&err_rhs))
                    .map(|v| (0..n).map(|j| v[j]).collect::<Vec<_>>())
            }
            .unwrap_or_else(|| vec![f64::NAN; n]);
            let mut error_norm =
                rms_norm(&(0..n).map(|j| error[j] / err_scale[j]).collect::<Vec<_>>());

            // Stabilised estimate after a rejection (scipy): re-solve with f(t, y+error).
            if rejected && error_norm > 1.0 {
                let yp: Vec<f64> = (0..n).map(|j| self.y[j] + error[j]).collect();
                let fp = fun(self.t, &yp);
                self.nfev += 1;
                err_rhs = DVector::<f64>::from_iterator(n, (0..n).map(|j| fp[j] + ze[j]));
                let corrected_error = if let Some(diagonal) = diagonal_jac {
                    solve_real_diagonal(diagonal, h, self.mu_real, &err_rhs)
                } else {
                    lu_real
                        .as_ref()
                        .and_then(|lu| lu.solve(&err_rhs))
                        .map(|v| (0..n).map(|j| v[j]).collect::<Vec<_>>())
                };
                if let Some(v) = corrected_error {
                    error = v;
                    error_norm =
                        rms_norm(&(0..n).map(|j| error[j] / err_scale[j]).collect::<Vec<_>>());
                }
            }

            if error_norm.is_nan() || error_norm > 1.0 {
                let factor = if error_norm.is_nan() {
                    0.5
                } else if !predictive_controller {
                    MIN_FACTOR.max(0.9 * error_norm.powf(ERR_EXP))
                } else {
                    MIN_FACTOR.max(
                        newton_safety(n_iter)
                            * predict_factor(h_abs, h_abs_old, error_norm, error_norm_old),
                    )
                };
                h_abs *= factor;
                rejected = true;
                continue;
            }

            // Accept.
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = t_new;
            self.y = y_new.clone();
            self.f = fun(t_new, &y_new);
            self.nfev += 1;

            let clamped_error_norm = error_norm.max(1e-10);
            let mut factor = if !predictive_controller {
                MAX_FACTOR.min(0.9 * clamped_error_norm.powf(ERR_EXP))
            } else {
                MAX_FACTOR.min(
                    newton_safety(n_iter)
                        * predict_factor(h_abs, h_abs_old, clamped_error_norm, error_norm_old),
                )
            };
            if rejected {
                factor = factor.min(1.0);
            }
            // scipy refreshes the Jacobian after an accepted step whose Newton
            // iteration was slow, and only then allows the step size to move. The
            // two halves are one policy: holding the step size with a stale, badly
            // contracting Jacobian buys a cheap step now and a Newton failure —
            // and a full re-factorization — on the next one.
            let recompute_jac = predictive_controller
                && n_iter > 2
                && last_rate.is_some_and(|rate| rate > JAC_REFRESH_RATE);
            if recompute_jac {
                RADAU_POST_STEP_JAC_REFRESH_HITS.fetch_add(1, Ordering::Relaxed);
            }
            let finishes_interval = reached_bound || t_new == self.t_bound;
            if !force_dense_rebuild
                && !recompute_jac
                && diagonal_jac.is_none()
                && !finishes_interval
                && factor < 1.2
            {
                factor = 1.0;
                RADAU_HELD_STEP_HITS.fetch_add(1, Ordering::Relaxed);
                self.dense_lu = match (lu_real.take(), lu_complex.take()) {
                    (Some(real), Some(complex)) => Some(DenseLuPair { h, real, complex }),
                    _ => None,
                };
            } else {
                self.dense_lu = None;
            }
            if recompute_jac {
                // Drop the cached Jacobian so the next step rebuilds it at
                // (t_new, y_new, f_new) — scipy computes it eagerly here; the
                // resulting Jacobian is the same one, evaluated at the same point.
                self.jac = None;
                self.jac_diagonal = None;
            }
            self.h_abs_old = Some(entry_h_abs);
            self.error_norm_old = Some(error_norm);
            self.h = (h_abs * factor) * self.direction;

            let state = if reached_bound {
                self.state = OdeSolverState::Finished;
                OdeSolverState::Finished
            } else {
                OdeSolverState::Running
            };
            return Ok(StepOutcome {
                message: None,
                state,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoupled_collocation_solve_matches_full_3n_lu() {
        // The eigen-decoupled solve must equal a direct full 3n×3n LU of the Radau
        // Newton matrix `I_{3n} − h(A⊗J)` on random dense Jacobians, to roundoff.
        let s6 = 6.0_f64.sqrt();
        let a = [
            [
                (88.0 - 7.0 * s6) / 360.0,
                (296.0 - 169.0 * s6) / 1800.0,
                (-2.0 + 3.0 * s6) / 225.0,
            ],
            [
                (296.0 + 169.0 * s6) / 1800.0,
                (88.0 + 7.0 * s6) / 360.0,
                (-2.0 - 3.0 * s6) / 225.0,
            ],
            [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
        ];
        let mu_real = 3.0 + 3.0_f64.powf(2.0 / 3.0) - 3.0_f64.powf(1.0 / 3.0);
        let mut seed = 0x1234_5678u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut worst = 0.0_f64;
        for n in [1usize, 2, 5, 9] {
            let h = 0.037;
            let jac = DMatrix::<f64>::from_fn(n, n, |_, _| rng());
            let rhs = DVector::<f64>::from_fn(3 * n, |_, _| rng());

            // Oracle: full 3n×3n LU of I − h(A⊗J).
            let mut m = DMatrix::<f64>::zeros(3 * n, 3 * n);
            for bi in 0..3 {
                for bj in 0..3 {
                    let coef = h * a[bi][bj];
                    for r in 0..n {
                        for col in 0..n {
                            let mut val = -coef * jac[(r, col)];
                            if bi == bj && r == col {
                                val += 1.0;
                            }
                            m[(bi * n + r, bj * n + col)] = val;
                        }
                    }
                }
            }
            let z_full = m.lu().solve(&rhs).expect("full 3n solve");

            // Decoupled path.
            let m_real = DMatrix::<f64>::identity(n, n) * (mu_real / h) - &jac;
            let m_complex = DMatrix::<Complex<f64>>::from_fn(n, n, |r, col| {
                let mut val = -Complex::new(jac[(r, col)], 0.0);
                if r == col {
                    val += MU_COMPLEX / Complex::new(h, 0.0);
                }
                val
            });
            let lu_real = m_real.lu();
            let lu_complex = m_complex.lu();
            let z_dec =
                solve_collocation_decoupled(&rhs, mu_real, h, &lu_real, &lu_complex, n).unwrap();

            for i in 0..3 * n {
                worst = worst.max((z_full[i] - z_dec[i]).abs());
            }
        }
        assert!(
            worst < 1e-10,
            "decoupled vs full-3n worst abs diff = {worst:.3e}"
        );
    }

    #[test]
    fn diagonal_collocation_solve_matches_dense_block_solve() {
        let s6 = 6.0_f64.sqrt();
        let tableau_a = [
            [
                (88.0 - 7.0 * s6) / 360.0,
                (296.0 - 169.0 * s6) / 1800.0,
                (-2.0 + 3.0 * s6) / 225.0,
            ],
            [
                (296.0 + 169.0 * s6) / 1800.0,
                (88.0 + 7.0 * s6) / 360.0,
                (-2.0 - 3.0 * s6) / 225.0,
            ],
            [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
        ];
        let diagonal = [-1.25, -32.0, -800.0];
        let h = 0.0025;
        let n = diagonal.len();
        let rhs = DVector::<f64>::from_vec(vec![0.25, -0.5, 1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0]);

        let diagonal_solution =
            solve_collocation_diagonal(&diagonal, h, &tableau_a, &rhs).expect("diagonal solve");

        let mut jac = DMatrix::<f64>::zeros(n, n);
        for (idx, &value) in diagonal.iter().enumerate() {
            jac[(idx, idx)] = value;
        }
        let mut dense = DMatrix::<f64>::zeros(3 * n, 3 * n);
        for bi in 0..3 {
            for bj in 0..3 {
                let coef = h * tableau_a[bi][bj];
                for row in 0..n {
                    for col in 0..n {
                        let mut value = -coef * jac[(row, col)];
                        if bi == bj && row == col {
                            value += 1.0;
                        }
                        dense[(bi * n + row, bj * n + col)] = value;
                    }
                }
            }
        }
        let dense_solution = dense.lu().solve(&rhs).expect("dense solve");

        for (diagonal_value, dense_value) in diagonal_solution.iter().zip(dense_solution.iter()) {
            assert!(
                (diagonal_value - dense_value).abs() <= 1e-12,
                "diagonal={diagonal_value}, dense={dense_value}"
            );
        }
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

    /// Every test that writes a `RADAU_FORCE_*` toggle or reads a global hit
    /// counter takes this. Without it, `cargo test` runs them concurrently in one
    /// process: one test's toggle leaks into another's arm (false drift) and one
    /// test's counter increments satisfy another's `> before` assertion (masked
    /// drift). Both failure modes print green.
    static TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn dense_radau_reuses_lu_pair_across_held_steps() {
        let _guard = TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        RADAU_FORCE_DENSE_LU_REBUILD.store(false, Ordering::Relaxed);
        RADAU_ENABLE_PREDICTIVE_CONTROLLER.store(false, Ordering::Relaxed);
        let hits_before = RADAU_DENSE_LU_REUSE_HITS.load(Ordering::Relaxed);
        let n = 8;
        let y0 = vec![1.0; n];
        let mut fun = |_t: f64, y: &[f64]| {
            let mean = y.iter().sum::<f64>() / n as f64;
            y.iter().map(|&value| -value + 0.01 * mean).collect()
        };
        let config = RadauSolverConfig {
            t0: 0.0,
            y0: &y0,
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: Some(1e-3),
            mode: RuntimeMode::Strict,
        };
        let mut solver = RadauSolver::new(&mut fun, config).expect("construct Radau solver");
        let mut steps = 0usize;
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("dense Radau step");
            steps += 1;
            assert!(steps < 10_000, "Radau failed to reach the bound");
        }

        let expected = (-0.99_f64).exp();
        let worst = solver
            .y()
            .iter()
            .map(|&value| (value - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst <= 2e-8, "dense Radau analytic error = {worst:.3e}");
        assert!(
            RADAU_DENSE_LU_REUSE_HITS.load(Ordering::Relaxed) > hits_before,
            "dense completion did not consume a retained LU pair"
        );
        assert!(
            solver.nlu() < 2 * steps,
            "factor count {} did not fall below two per step for {steps} steps",
            solver.nlu()
        );
        assert!(
            solver.dense_lu.is_none(),
            "a factor pair survived the terminal boundary"
        );
    }

    /// `predict_factor` against hand-evaluated scipy values, including the branch
    /// a naive port drops.
    ///
    /// THE NEGATIVE CASE is the third assertion: with a history where the step
    /// grew and the error fell, the raw multiplier exceeds 1 and the unclamped
    /// prediction would be `2.0 * 16^0.25 * 1^-0.25 = 4.0`. scipy takes
    /// `min(1, multiplier)`, so the answer must be 1.0. An implementation that
    /// forgets the clamp — the single most likely way to get this wrong, and the
    /// one that reproduces the elementary controller's over-optimistic growth —
    /// fails here and nowhere else.
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
        // multiplier = 0.05/0.1 * (1/1)^0.25 = 0.5; factor = 0.5 * 1^-0.25 = 0.5.
        let damped = predict_factor(0.05, Some(0.1), 1.0, Some(1.0));
        assert!((damped - 0.5).abs() < 1e-15, "two-step rule: got {damped}");

        // NEGATIVE CASE: multiplier = 0.1/0.05 * (16/1)^0.25 = 2*2 = 4 > 1.
        // Unclamped this predicts 4.0; scipy clamps the multiplier to 1.
        let clamped = predict_factor(0.1, Some(0.05), 1.0, Some(16.0));
        assert!(
            (clamped - 1.0).abs() < 1e-15,
            "min(1, multiplier) clamp is missing: got {clamped}, unclamped would be 4.0"
        );

        // Zero error norm degrades to the one-step rule (scipy's `error_norm == 0`
        // guard) and is capped by the caller, not here.
        assert!(predict_factor(0.1, Some(0.05), 0.0, Some(1.0)).is_infinite());

        // scipy's safety is 0.9 exactly for a single Newton iteration
        // (0.9·(2·6+1)/(2·6+1)) and decreases from there, so a step that needed
        // more iterations is trusted less. It is never 0.9 at n_iter = 0 — that
        // would be 0.975, above scipy's headline value.
        assert!((newton_safety(1) - 0.9).abs() < 1e-15);
        assert!(newton_safety(6) < newton_safety(2));
        assert!(newton_safety(NEWTON_MAXITER) > 0.0);
    }

    /// Dense-Radau factorization economy, anchored to the LIVE incumbent.
    ///
    /// WHY THIS EXISTS. `docs/perf_ledger_cc.md`'s 2026-07-28 dense-Radau n=128 row
    /// (`0.4188x`, a decided 2.39x loss) attributes ~16x of the gap to
    /// "1,178 counted factorizations per solve versus SciPy's 74", and its retry
    /// predicate forbids re-attacking dense Radau until that count comes down. This
    /// test pins the count so a regression back to per-step re-factorization —
    /// which is what a broken `dense_lu` retention or an over-eager step-size
    /// controller produces — fails here instead of silently costing 16x.
    ///
    /// THE BOUND IS AN INCUMBENT NUMBER, measured live on 2026-08-14 against
    /// scipy 1.17.1 on this exact fixture family (`scipy.integrate.solve_ivp`,
    /// `method="Radau"`, rtol 1e-8, atol 1e-10, t∈[0,1]): n=48 → 604 steps,
    /// nlu=64, njev=2; n=128 → 680 steps, nlu=74, njev=2. It is a regression
    /// bound, not a perf claim: a timing claim would need the incumbent running
    /// in the same invocation.
    ///
    /// It also carries the REFUTATION of the predictive-controller lever as a
    /// live A/B, so the negative stays reproducible rather than becoming folklore.
    #[test]
    fn dense_radau_factorization_count_tracks_the_scipy_incumbent() {
        let _guard = TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        RADAU_FORCE_DENSE_LU_REBUILD.store(false, Ordering::Relaxed);

        // Dense Jacobian by construction: every component couples to the mean of
        // all components, so no diagonal or banded fast path can fire and the
        // dense real/complex LU pair is the thing being counted.
        // Same family as the live-incumbent harness's `dense` fixture
        // (`perf_bdf_vs_scipy`): decay rates 1 + 10i spanning three decades, a
        // 1e-3 all-pairs mean coupling, and its staggered initial state. Smaller
        // n only to keep the unit test fast; the step-size behaviour under test
        // is set by the rate spread, not by n.
        let n = 48;
        let decay: Vec<f64> = (0..n).map(|j| 1.0 + 10.0 * j as f64).collect();
        let y0: Vec<f64> = (0..n).map(|j| 1.0 + 0.25 * ((j % 7) as f64)).collect();
        let run = |predictive: bool| {
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
            RADAU_ENABLE_PREDICTIVE_CONTROLLER.store(predictive, Ordering::Relaxed);
            let held_before = RADAU_HELD_STEP_HITS.load(Ordering::Relaxed);
            let mut solver = RadauSolver::new(&mut fun, config).expect("construct Radau solver");
            let mut steps = 0usize;
            while solver.state() == OdeSolverState::Running {
                solver.step_with(&mut fun).expect("dense Radau step");
                steps += 1;
                assert!(steps < 100_000, "Radau failed to reach the bound");
            }
            let held = RADAU_HELD_STEP_HITS.load(Ordering::Relaxed) - held_before;
            (solver.y().to_vec(), solver.nlu(), steps, held)
        };

        // Live-measured scipy 1.17.1 profile on this fixture at n=48.
        const SCIPY_NLU_N48: usize = 64;
        const SCIPY_STEPS_N48: usize = 604;

        let (y_ship, nlu_ship, steps_ship, held_ship) = run(false);
        let (y_pred, nlu_pred, steps_pred, held_pred) = run(true);
        RADAU_ENABLE_PREDICTIVE_CONTROLLER.store(false, Ordering::Relaxed);

        // Execution proof: if the shipped arm never holds a step, the LU pair is
        // never retained and the counts below are measuring a different code path
        // from the one this test claims to guard.
        assert!(
            held_ship > 0,
            "shipped arm held zero steps — the dense LU retention never engaged"
        );
        assert!(held_pred > 0, "predictive arm held zero steps");

        // THE GUARD. The pre-reuse behaviour was ~2 factorizations per step; that
        // is the regression this bound catches. 2x the incumbent's count leaves
        // room for a legitimately different step sequence while still failing an
        // order-of-magnitude regression by a wide margin.
        assert!(
            nlu_ship <= 2 * SCIPY_NLU_N48,
            "dense Radau factorization economy regressed: nlu={nlu_ship} over \
             {steps_ship} steps, against live scipy nlu={SCIPY_NLU_N48} over \
             {SCIPY_STEPS_N48} steps (bound {})",
            2 * SCIPY_NLU_N48
        );
        assert!(
            steps_ship <= 2 * SCIPY_STEPS_N48,
            "dense Radau step count regressed: {steps_ship} vs live scipy \
             {SCIPY_STEPS_N48}"
        );

        // Both arms are controlled to the same rtol/atol, so they must agree to
        // roughly that tolerance even though their step sequences differ.
        let worst = y_ship
            .iter()
            .zip(&y_pred)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= 1e-7,
            "controller arms disagree by {worst:.3e} — the predictive path is not \
             tolerance-preserving (shipped steps={steps_ship}, predictive \
             steps={steps_pred})"
        );

        // THE REFUTATION, kept executable. The predictive controller was adopted
        // on the hypothesis that it would cut factorizations; measured, it does
        // not. If a future change ever makes it win here, this assertion fires and
        // whoever sees it should re-measure the lever rather than trust the note
        // on RADAU_ENABLE_PREDICTIVE_CONTROLLER.
        assert!(
            nlu_pred >= nlu_ship,
            "the predictive controller now BEATS the shipped one \
             (pred nlu={nlu_pred} < shipped nlu={nlu_ship}) — the 2026-08-14 \
             rejection no longer holds; re-measure before trusting either note"
        );
    }
}
