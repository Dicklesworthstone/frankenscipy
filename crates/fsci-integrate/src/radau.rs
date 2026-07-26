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
use std::sync::atomic::{AtomicBool, Ordering};

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
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 8.0;
const ERR_EXP: f64 = -0.25; // embedded estimator is order 3 → 1/(3+1).

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

        let mut h_abs = self.h.abs().min(self.max_step).max(min_step);
        let mut rejected = false;

        loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let mut h = h_abs * self.direction;
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
            // failure). The two factorizations depend on h, which changes every
            // step, so they are rebuilt each attempt.
            let jac_fresh = self.jac.is_none();
            if jac_fresh {
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
                    // Dense Jacobian: factor the eigen-decoupled real and complex
                    // n×n blocks `(MU_REAL/h)I − J` and `(MU_COMPLEX/h)I − J` rather
                    // than a full 3n×3n LU (see `solve_collocation_decoupled`). The
                    // real factor is also the one the error estimate reuses.
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
                }
                diagonal
            };
            self.nlu += 2;

            // Simplified Newton on the stage corrections Z (3 × n), initial guess 0.
            let mut z = vec![vec![0.0; n]; 3];
            let scale: Vec<f64> = (0..n)
                .map(|j| self.atol[j] + self.rtol * self.y[j].abs())
                .collect();
            let mut converged = false;
            let mut dz_norm_old: Option<f64> = None;
            let mut bad = false;
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
                let dz = if let Some(diagonal) = diagonal_jac.as_deref() {
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
            let mut error = if let Some(diagonal) = diagonal_jac.as_deref() {
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
                let corrected_error = if let Some(diagonal) = diagonal_jac.as_deref() {
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
                } else {
                    MIN_FACTOR.max(0.9 * error_norm.powf(ERR_EXP))
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

            let mut factor = MAX_FACTOR.min(0.9 * error_norm.max(1e-10).powf(ERR_EXP));
            if rejected {
                factor = factor.min(1.0);
            }
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
}
