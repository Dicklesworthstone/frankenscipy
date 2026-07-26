#![forbid(unsafe_code)]

//! Backward Differentiation Formula (BDF) solver for stiff ODEs.
//!
//! Genuine variable-order (1-5) BDF with adaptive step size, a faithful port of
//! `scipy.integrate.solve_ivp(method='BDF')` (`scipy/integrate/_ivp/bdf.py`):
//! predictor from the backward-difference array `D`, modified-Newton corrector on
//! `(I − c·J)` with a lazily-refreshed finite-difference Jacobian, and combined
//! error/step/order control via `change_D`. Order and step are reconsidered after
//! `n_equal_steps >= order + 1` accepted steps by comparing the local error at
//! orders `k-1`, `k`, `k+1` (frankenscipy-3y5p9). `SolverKind::Radau` now has its
//! own genuine Radau IIA solver (see `radau.rs`).

use crate::solver::{OdeSolverState, StepFailure, StepOutcome};
use crate::validation::{
    ToleranceValue, validate_first_step, validate_max_step, validate_rhs_shape, validate_tol,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Runtime switch that restores the per-Newton-iteration scratch allocation in
/// [`BdfSolver::newton_bdf`] (a fresh `rhs` `DVector` plus a `lu.solve` result Vec per
/// iteration — the ORIG behaviour) for same-binary A/B benchmarks. Defaults off — the
/// `rhs` buffer is hoisted above the Newton loop and the linear solve runs in place
/// (`solve_mut`), which is bit-identical (`solve(b) == { let mut r=b.clone();
/// solve_mut(&mut r); r }`). Mirrors `RADAU_FORCE_PER_ITER_ALLOC`. `#[doc(hidden)]`.
#[doc(hidden)]
pub static BDF_FORCE_PER_ITER_ALLOC: AtomicBool = AtomicBool::new(false);

/// Runtime switch that restores the unconditional dense-LU factorization of the BDF
/// Newton matrix `I − c·J` (the ORIG behaviour) for same-binary A/B benchmarks.
/// Defaults off — when the finite-difference Jacobian is EXACTLY diagonal the Newton
/// matrix is diagonal too, and [`NewtonFactor::Diagonal`] replaces an `O(n³)` LU plus
/// two `O(n²)` substitutions with `n` scalar divisions. See [`NewtonFactor`] for the
/// bit-identity argument. Mirrors [`BDF_FORCE_PER_ITER_ALLOC`]. `#[doc(hidden)]`.
#[doc(hidden)]
pub static BDF_FORCE_DENSE_NEWTON: AtomicBool = AtomicBool::new(false);

/// Count of Newton factorizations that took the diagonal path. EXECUTION PROOF for
/// tests and A/B harnesses: a candidate arm that reports zero hits never ran the code
/// under test (the failure mode that voided a third of this repo's REJECT ledger — see
/// `docs/LEDGER_RESURRECTION.md`). Incremented once per factorization, i.e. once per
/// `nlu`, never in the Newton inner loop. `#[doc(hidden)]`.
#[doc(hidden)]
pub static BDF_DIAG_NEWTON_HITS: AtomicUsize = AtomicUsize::new(0);

/// Count of Newton factorizations that took the BANDED path. Same execution-proof role
/// as [`BDF_DIAG_NEWTON_HITS`]. `#[doc(hidden)]`.
#[doc(hidden)]
pub static BDF_BAND_NEWTON_HITS: AtomicUsize = AtomicUsize::new(0);

/// Factorization of the BDF Newton matrix `I − c·J`.
///
/// `Diagonal` is taken only when every off-diagonal entry of `J` is EXACTLY `0.0`
/// and every `1 − c·J[j][j]` is finite and non-zero. Under those preconditions the
/// diagonal arm is BIT-IDENTICAL to the dense arm, not merely equivalent:
///
/// * `I − c·J` is then diagonal with non-zero diagonal, so partial pivoting selects
///   row `j` in column `j` strictly (`|d_j| > 0 = |off-diagonals|`) and the
///   permutation is the identity — no row swaps to reproduce.
/// * `L` is unit-lower with exactly-zero sub-diagonal, so forward substitution
///   computes `b[k] -= 0.0 * b[i]`, leaving finite `b` unchanged.
/// * `U` is the diagonal itself, so back substitution computes exactly
///   `b[j] / (1 − c·J[j][j])` — the same IEEE division, in the same order, on the
///   same operands. The diagonal entry is formed with the same expression the dense
///   arm uses (`identity - jac.scale(c)`).
///
/// The one place the two arms could diverge is non-finite intermediates: dense
/// substitution multiplies zeros against `±inf`/`NaN` and spreads `NaN` across
/// components that the diagonal arm would keep independent. Rather than emulate that
/// propagation, the solve detects a non-finite right-hand side or quotient and
/// reconstructs the dense LU for that iteration (see `newton_bdf`), which is the
/// dense computation itself. That branch is unreachable for any convergent step.
enum NewtonFactor {
    /// Dense LU of `I − c·J` (scipy's unconditional path).
    Dense(LU<f64, Dyn, Dyn>),
    /// `I − c·J` is exactly diagonal: the stored entries are `1 − c·J[j][j]`.
    Diagonal(Vec<f64>),
    /// `I − c·J` is exactly banded: GEPP restricted to the band (see [`BandedLu`]).
    Banded(BandedLu),
}

/// Gaussian elimination on a matrix whose non-zeros lie in a band, storing the factors
/// in the same dense `n×n` layout `nalgebra` uses and visiting only the band.
///
/// BIT-IDENTICAL to `nalgebra`'s dense `LU` — by construction, and with the one
/// precondition that makes the construction valid CHECKED AT RUNTIME rather than
/// assumed: **no row interchange occurs**.
///
/// Why that precondition is not a cop-out. Under partial pivoting a multiplier
/// migrates down its column by one row per interchange, without bound, and an active
/// row carries its column extent with it — so after interchanges neither `L` nor the
/// active region is banded any more, dense GEPP's pivot search can legitimately reach
/// far outside the nominal band, and no band-clipped factorization can reproduce it.
/// (LAPACK's `gbtrf` sidesteps this by storing multipliers separately and producing a
/// DIFFERENT `L` and permutation than dense GEPP would — bit-identity is off the table
/// there.) This was not theorised: `banded_lu_is_bit_identical_to_dense_lu` produced
/// exactly that divergence on pivot-forcing matrices, twice, before the design changed.
///
/// The target class satisfies the precondition by construction, not by luck. For a
/// method-of-lines Jacobian, `I − c·J` is strictly column diagonally dominant
/// (`c > 0`, `J`'s diagonal negative), so the largest magnitude in each column is the
/// diagonal and `icamax` returns it — and Gaussian elimination preserves diagonal
/// dominance, so it stays true at every step. Rather than lean on that theorem, the
/// factorization simply checks at each step whether the in-band maximum sits on the
/// diagonal, and returns `None` the moment it does not; the caller then takes the dense
/// LU. Nothing is claimed for matrices that would pivot.
///
/// With no interchanges the structure is stable: `L` keeps lower bandwidth `kl`, `U`
/// keeps upper bandwidth `ku`, no fill escapes the band, and every step is
/// `nalgebra`'s `gauss_step` restricted to it:
///
/// * `inv_diag = 1/diag` ONCE, then `l[r] = a[r][i] * inv_diag` — a multiply by the
///   reciprocal, not a division (`lu.rs::gauss_step`).
/// * trailing update `a[r][k] = (−u[k])·l[r] + a[r][k]`, column-major over `k`
///   (`axpy(-pivot_row[k], &coeffs, 1.0)`); `fp-contract=off` so no FMA to match.
/// * skipped work is provably a no-op: outside the band `a[r][i]` is `0.0`, so
///   `l[r]` is `±0.0` and the update adds `±0.0` to a finite entry, returning it
///   unchanged (IEEE addition is commutative and `(±0) + (∓0) = +0`).
/// * a zero pivot with a zero column makes `nalgebra` `continue`, not fail; that branch
///   is reproduced, including the `continue`.
struct BandedLu {
    /// `L` (unit diagonal, strictly lower) and `U` overwritten in place, dense layout.
    lu: DMatrix<f64>,
    /// Lower bandwidth: `L[r][i] != 0` only for `r <= i + kl`.
    kl: usize,
    /// Upper bandwidth: `U[r][i] != 0` only for `r >= i - ku`.
    ku: usize,
}

impl BandedLu {
    /// Factorize in place, or `None` if any step would interchange rows (see the type
    /// comment — that is the precondition, checked, not assumed).
    fn factor(mut lu: DMatrix<f64>, kl: usize, ku: usize) -> Option<Self> {
        let n = lu.nrows();
        for i in 0..n {
            let row_hi = (i + kl).min(n - 1);
            // `icamax` over rows `i..` — FIRST index attaining the max magnitude. With
            // no interchange so far the column is banded, so `i..=i+kl` is the whole
            // of it and this search is complete.
            let diag = lu[(i, i)];
            let diag_abs = diag.abs();
            for r in (i + 1)..=row_hi {
                if lu[(r, i)].abs() > diag_abs {
                    return None; // dense GEPP would interchange here: bail.
                }
            }
            if diag == 0.0 {
                continue; // nalgebra: `if diag.is_zero() { continue; }`
            }
            let inv_diag = 1.0 / diag;
            for r in (i + 1)..=row_hi {
                lu[(r, i)] *= inv_diag;
            }
            let col_hi = (i + ku).min(n - 1);
            for k in (i + 1)..=col_hi {
                let alpha = -lu[(i, k)];
                for r in (i + 1)..=row_hi {
                    // `alpha * x + y`, NOT `y += alpha * x`: this mirrors nalgebra's
                    // `axpy` operand order exactly, which is the whole basis of the
                    // bit-identity claim. Do not let a lint reorder it.
                    #[allow(clippy::assign_op_pattern)]
                    {
                        lu[(r, k)] = alpha * lu[(r, i)] + lu[(r, k)];
                    }
                }
            }
        }
        Some(Self { lu, kl, ku })
    }

    /// Solve in place. Transcribes `LU::solve_mut` with an empty permutation:
    /// unit-lower forward substitution then upper back substitution, band-clipped.
    /// Returns `false` on a zero `U` diagonal, exactly as `nalgebra` does.
    fn solve_mut(&self, b: &mut DVector<f64>) -> bool {
        let n = self.lu.nrows();
        // `solve_lower_triangular_with_diag_mut(b, 1.0)`: `coeff = b[i] / 1.0`.
        for i in 0..n.saturating_sub(1) {
            let alpha = -b[i];
            let row_hi = (i + self.kl).min(n - 1);
            for r in (i + 1)..=row_hi {
                #[allow(clippy::assign_op_pattern)] // nalgebra `axpy` order — see `factor`.
                {
                    b[r] = alpha * self.lu[(r, i)] + b[r];
                }
            }
        }
        // `solve_upper_triangular_vector_mut`.
        for i in (0..n).rev() {
            let diag = self.lu[(i, i)];
            if diag == 0.0 {
                return false;
            }
            let coeff = b[i] / diag;
            b[i] = coeff;
            let alpha = -coeff;
            for r in i.saturating_sub(self.ku)..i {
                #[allow(clippy::assign_op_pattern)] // nalgebra `axpy` order — see `factor`.
                {
                    b[r] = alpha * self.lu[(r, i)] + b[r];
                }
            }
        }
        true
    }
}

/// Strict column diagonal dominance over the band: `|m[i][i]| > Σ_{r≠i} |m[r][i]|`.
///
/// Gaussian elimination preserves it, so one check on the untouched matrix guarantees
/// that no step of GEPP interchanges rows — the precondition [`BandedLu`] needs. The
/// target class (`I − c·J` for a method-of-lines Jacobian, `c > 0`, `J`'s diagonal
/// negative) satisfies it by construction.
fn band_column_diagonally_dominant(m: &DMatrix<f64>, kl: usize, ku: usize) -> bool {
    let n = m.nrows();
    for col in 0..n {
        let lo = col.saturating_sub(ku);
        let hi = (col + kl).min(n - 1);
        let mut off = 0.0;
        for row in lo..=hi {
            if row != col {
                off += m[(row, col)].abs();
            }
        }
        // Strictly greater, with NaN failing the test: `partial_cmp` returns `None` for
        // a NaN diagonal or off-diagonal sum, which is not `Greater`, so the banded
        // path declines and the dense LU decides.
        if m[(col, col)].abs().partial_cmp(&off) != Some(std::cmp::Ordering::Greater) {
            return false;
        }
    }
    true
}

/// Lower/upper bandwidth of `jac`, or `None` if it is not usefully banded. Runs ONCE
/// per Jacobian (`njev`) alongside [`crate::radau::diagonal_jacobian_entries`] — see
/// [`newton_denominators`] for why that cadence is load-bearing.
///
/// The gate `3·(kl + ku + 1) <= n` keeps the banded path away from nearly-dense
/// matrices, where the fill to `ku + kl` would make it no cheaper than the dense LU it
/// replaces while still paying the scan.
fn jacobian_bandwidth(jac: &DMatrix<f64>) -> Option<(usize, usize)> {
    let n = jac.nrows();
    if n < 8 {
        return None; // dense LU is already trivial here; not worth a second path.
    }
    let (mut kl, mut ku) = (0usize, 0usize);
    for col in 0..n {
        for row in 0..n {
            if jac[(row, col)] != 0.0 {
                if row > col {
                    kl = kl.max(row - col);
                } else {
                    ku = ku.max(col - row);
                }
            }
        }
    }
    if 3 * (kl + ku + 1) <= n {
        Some((kl, ku))
    } else {
        None
    }
}

/// Newton denominators `1 − c·J[j][j]` from the CACHED Jacobian diagonal, or `None`
/// if any is zero (singular) or non-finite — in which case the dense path decides.
///
/// The `O(n²)` "is `J` exactly diagonal" scan is [`crate::radau::diagonal_jacobian_entries`]
/// — Radau has exploited exactly-diagonal Jacobians since it was written (it splits
/// `M_3n` into `n` independent 3×3 systems); BDF was the sibling straggler for the same
/// structural fact, so this is one definition of the invariant, not two. It runs ONCE
/// per Jacobian (`njev`), cached in `BdfSolver::jac_diagonal`, exactly as Radau caches
/// it. This function then runs once per FACTORIZATION (`nlu`) and is `O(n)`.
///
/// That split is load-bearing, not tidiness: `nlu` exceeds `njev` by two orders of
/// magnitude on a stiff solve (127 vs 1 at n=512), and the scan walks a column-major
/// `DMatrix` row-major, so re-running it per factorization costs a full cache-missing
/// `n²` sweep each time. Doing so measured **14.80× instead of 45.82×** at n=512 — the
/// same lever, three times weaker, from putting an `O(n²)` scan on the `O(n)` path.
fn newton_denominators(jac_diagonal: &[f64], c: f64) -> Option<Vec<f64>> {
    let mut diag = Vec::with_capacity(jac_diagonal.len());
    for &j_jj in jac_diagonal {
        let d_jj = 1.0 - c * j_jj;
        if d_jj == 0.0 || !d_jj.is_finite() {
            return None;
        }
        diag.push(d_jj);
    }
    Some(diag)
}

/// Maximum BDF order.
const MAX_ORDER: usize = 5;
/// Maximum Newton iterations per step (scipy `NEWTON_MAXITER`).
const NEWTON_MAXITER: usize = 4;
/// Minimum step-size reduction factor on rejection.
const MIN_FACTOR: f64 = 0.2;
/// Maximum step-size growth factor on acceptance.
const MAX_FACTOR: f64 = 10.0;

/// Empirical `kappa` constants (scipy `_bdf.py`), indices 0..=5.
const KAPPA: [f64; 6] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];
/// `gamma[i] = Σ_{k=1}^{i} 1/k`, `gamma[0] = 0`.
const GAMMA_C: [f64; 6] = [0.0, 1.0, 1.5, 11.0 / 6.0, 25.0 / 12.0, 137.0 / 60.0];
/// `alpha[i] = (1 - kappa[i]) * gamma[i]` — the BDF leading coefficient.
const ALPHA_C: [f64; 6] = [
    (1.0 - KAPPA[0]) * GAMMA_C[0],
    (1.0 - KAPPA[1]) * GAMMA_C[1],
    (1.0 - KAPPA[2]) * GAMMA_C[2],
    (1.0 - KAPPA[3]) * GAMMA_C[3],
    (1.0 - KAPPA[4]) * GAMMA_C[4],
    (1.0 - KAPPA[5]) * GAMMA_C[5],
];
/// `error_const[i] = kappa[i]*gamma[i] + 1/(i+1)` — local error coefficient.
const ERR_C: [f64; 6] = [
    KAPPA[0] * GAMMA_C[0] + 1.0,
    KAPPA[1] * GAMMA_C[1] + 1.0 / 2.0,
    KAPPA[2] * GAMMA_C[2] + 1.0 / 3.0,
    KAPPA[3] * GAMMA_C[3] + 1.0 / 4.0,
    KAPPA[4] * GAMMA_C[4] + 1.0 / 5.0,
    KAPPA[5] * GAMMA_C[5] + 1.0 / 6.0,
];

/// RMS norm `sqrt(mean(x²))` (scipy `norm`).
#[cfg(test)]
fn rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f64).sqrt()
}

/// RMS norm of `values[j] / scale[j]`, streamed without materializing the scaled vector.
fn rms_norm_scaled(values: impl Iterator<Item = f64>, scale: &[f64]) -> f64 {
    if scale.is_empty() {
        return 0.0;
    }
    let mut s = 0.0;
    for (value, &scale_j) in values.zip(scale.iter()) {
        let scaled = value / scale_j;
        s += scaled * scaled;
    }
    (s / scale.len() as f64).sqrt()
}

/// scipy `compute_R(order, factor)` — the `(order+1)×(order+1)` step-change
/// matrix whose columns are cumulative products down the rows.
fn compute_r(order: usize, factor: f64) -> DMatrix<f64> {
    let m = order + 1;
    let mut mat = DMatrix::<f64>::zeros(m, m);
    // Row 0 is all ones.
    for j in 0..m {
        mat[(0, j)] = 1.0;
    }
    // M[i,j] = (i - 1 - factor*j)/i for i,j >= 1; column 0 (j=0) stays 0.
    for i in 1..m {
        for j in 1..m {
            mat[(i, j)] = ((i as f64) - 1.0 - factor * (j as f64)) / (i as f64);
        }
    }
    // Cumulative product down each column (axis 0).
    for j in 0..m {
        for i in 1..m {
            mat[(i, j)] *= mat[(i - 1, j)];
        }
    }
    mat
}

/// scipy `change_D(D, order, factor)` — rescale the difference array `d[0..=order]`
/// in place when the step size changes by `factor`.
fn change_d(d: &mut [Vec<f64>], order: usize, factor: f64, n: usize) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let ru = &r * &u; // (order+1)×(order+1)
    let m = order + 1;
    // new D[i] = Σ_k (RU.T)[i,k] * D[k] = Σ_k RU[k,i] * D[k].
    let mut new_d = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..m {
            let w = ru[(k, i)];
            if w != 0.0 {
                for col in 0..n {
                    new_d[i][col] += w * d[k][col];
                }
            }
        }
    }
    d[..m].clone_from_slice(&new_d[..m]);
}

/// Configuration for the BDF solver.
#[derive(Debug, Clone)]
pub struct BdfSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
    pub max_order: usize,
}

/// BDF solver for stiff ODE systems.
pub struct BdfSolver {
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_bound: f64,
    direction: f64,
    h: f64,
    max_step: f64,
    rtol: f64,
    atol: Vec<f64>,
    order: usize,
    max_order: usize,
    /// Consecutive accepted steps at the current order/step (scipy `n_equal_steps`).
    n_equal_steps: usize,
    state: OdeSolverState,
    nfev: usize,
    njev: usize,
    nlu: usize,
    mode: RuntimeMode,

    f: Vec<f64>,
    f_old: Option<Vec<f64>>,

    // Nordsieck-style array: d[k] for k = 0..order
    d: Vec<Vec<f64>>,

    // Newton solver state
    current_jac: Option<DMatrix<f64>>,
    /// `current_jac`'s diagonal entries when that Jacobian is EXACTLY diagonal, else
    /// `None`. Computed once per Jacobian (`njev`) and consumed once per factorization
    /// (`nlu`) — see [`newton_denominators`]. Mirrors `RadauSolver::jac_diagonal`.
    jac_diagonal: Option<Vec<f64>>,
    /// `current_jac`'s `(kl, ku)` bandwidths when it is usefully banded, else `None`.
    /// Same cadence as `jac_diagonal`: computed per Jacobian, consumed per factorization.
    jac_band: Option<(usize, usize)>,
    /// Factorization of `I − c·J`: dense LU, or the diagonal itself when `J` is
    /// exactly diagonal (see [`NewtonFactor`]).
    lu: Option<NewtonFactor>,
    /// The value of `c = h/alpha[order]` for which `lu` was factorized.
    lu_c: Option<f64>,

    // Previous step values for interpolation
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
}

impl BdfSolver {
    /// Create a new BDF solver.
    pub fn new<F>(
        fun: &mut F,
        config: BdfSolverConfig<'_>,
    ) -> Result<Self, crate::IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();

        // Input validation mirrors RkSolver::new (per frankenscipy-ljmg):
        // previously BdfSolver::new did ZERO validation — a caller passing
        // rtol=NaN, max_step=NaN, first_step=Some(NaN) etc. started a solver
        // that then burned Newton iterations until StepSizeTooSmall.
        let _validated_tol = validate_tol(
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

        // Backward-difference array D[0..=MAX_ORDER+2]: D[0] = y, D[1] = h*f, rest 0.
        let mut d = vec![vec![0.0; n]; MAX_ORDER + 3];
        d[0] = y0.clone();
        for (j, d1j) in d[1].iter_mut().enumerate() {
            *d1j = h * f0[j];
        }

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
            order: 1,
            max_order: config.max_order.min(MAX_ORDER),
            n_equal_steps: 0,
            state: OdeSolverState::Running,
            nfev: 1,
            njev: 0,
            nlu: 0,
            mode: config.mode,
            f: f0.clone(),
            f_old: None,
            d,
            current_jac: None,
            jac_diagonal: None,
            jac_band: None,
            lu: None,
            lu_c: None,
            t_old: None,
            y_old: None,
        })
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn y(&self) -> &[f64] {
        &self.y
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

    pub fn f(&self) -> &[f64] {
        &self.f
    }

    pub fn f_old(&self) -> Option<&[f64]> {
        self.f_old.as_deref()
    }

    pub fn state(&self) -> OdeSolverState {
        self.state
    }

    pub fn mode(&self) -> RuntimeMode {
        self.mode
    }

    /// Perform one adaptive BDF step.
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

        self.bdf_step_impl(fun)
    }

    // Index-aligned array arithmetic over the difference array reads cleaner with
    // explicit `j`/`k` indices than with zipped iterators here.
    #[allow(clippy::needless_range_loop)]
    fn bdf_step_impl<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        // Faithful variable-order (1-5) BDF (scipy `_bdf.py::_step_impl`):
        // predictor from the backward-difference array D, modified-Newton corrector
        // on (I − c·J) with a lazy Jacobian, error/step/order control via change_D.
        let n = self.n;
        let newton_tol = (10.0 * f64::EPSILON / self.rtol).max(0.03_f64.min(self.rtol.sqrt()));

        let spacing = if self.direction > 0.0 {
            self.t.next_up() - self.t
        } else {
            self.t - self.t.next_down()
        };
        let min_step = 10.0 * spacing.abs();

        let mut h_abs = self.h.abs();
        if h_abs > self.max_step {
            let factor = self.max_step / h_abs;
            h_abs = self.max_step;
            change_d(&mut self.d, self.order, factor, n);
            self.n_equal_steps = 0;
            self.lu = None;
        } else if h_abs < min_step {
            let factor = min_step / h_abs;
            h_abs = min_step;
            change_d(&mut self.d, self.order, factor, n);
            self.n_equal_steps = 0;
            self.lu = None;
        }

        let mut order = self.order;
        let mut t_new;
        let mut y_new = vec![0.0; n];
        let mut d_corr = vec![0.0; n];
        let mut scale = vec![0.0; n];
        let mut n_iter = 1usize;
        let mut reached_bound;

        loop {
            if h_abs < min_step {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::StepSizeTooSmall);
            }
            let mut h = h_abs * self.direction;
            t_new = self.t + h;
            reached_bound = self.direction * (t_new - self.t_bound) > 0.0;
            if reached_bound {
                t_new = self.t_bound;
                let factor = (t_new - self.t).abs() / h_abs;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
            }
            h = t_new - self.t;
            h_abs = h.abs();

            // Predictor and history terms.
            let mut y_predict = vec![0.0; n];
            for dk in self.d.iter().take(order + 1) {
                for (yp, &dkj) in y_predict.iter_mut().zip(dk.iter()) {
                    *yp += dkj;
                }
            }
            for j in 0..n {
                scale[j] = self.atol[j] + self.rtol * y_predict[j].abs();
            }
            let inv_alpha = 1.0 / ALPHA_C[order];
            let mut psi = vec![0.0; n];
            for k in 1..=order {
                let g = GAMMA_C[k] * inv_alpha;
                for (p, &dkj) in psi.iter_mut().zip(self.d[k].iter()) {
                    *p += g * dkj;
                }
            }
            let c = h * inv_alpha;

            // Modified-Newton with lazy Jacobian refresh.
            let mut converged = false;
            let mut jac_recomputed = false;
            loop {
                if self.current_jac.is_none() {
                    let f_pred = fun(t_new, &y_predict);
                    self.nfev += 1;
                    let jac = self.compute_jacobian(fun, t_new, &y_predict, &f_pred);
                    // The O(n²) structural scan runs HERE, once per Jacobian, not once
                    // per factorization — see `newton_denominators`.
                    self.jac_diagonal = crate::radau::diagonal_jacobian_entries(&jac);
                    self.jac_band = if self.jac_diagonal.is_some() {
                        None // the diagonal path is strictly better; do not double-scan.
                    } else {
                        jacobian_bandwidth(&jac)
                    };
                    self.current_jac = Some(jac);
                    self.lu = None;
                    jac_recomputed = true;
                }
                if self.lu.is_none() || self.lu_c != Some(c) {
                    let jac = self.current_jac.as_ref().expect("jacobian present");
                    // Structure-exploiting factorization: an exactly-diagonal Jacobian
                    // makes `I − c·J` diagonal, so the O(n³) LU collapses to `n`
                    // reciprocal denominators. Bit-identical (see `NewtonFactor`);
                    // `nlu` counts the factorization either way, so the reported
                    // `SolveIvpResult` counters are unchanged.
                    let force_dense = BDF_FORCE_DENSE_NEWTON.load(Ordering::Relaxed);
                    let diag = if force_dense {
                        None
                    } else {
                        self.jac_diagonal
                            .as_deref()
                            .and_then(|d| newton_denominators(d, c))
                    };
                    let band = if force_dense { None } else { self.jac_band };
                    self.lu = Some(match diag {
                        Some(d) => {
                            BDF_DIAG_NEWTON_HITS.fetch_add(1, Ordering::Relaxed);
                            NewtonFactor::Diagonal(d)
                        }
                        None => {
                            let system = DMatrix::<f64>::identity(n, n) - jac.scale(c);
                            // Same structural argument one step out: a banded
                            // `I - c*J` makes GEPP touch only the band, and the skipped
                            // work is provably a no-op — PROVIDED no row interchange
                            // occurs, which `BandedLu::factor` checks and reports by
                            // returning `None` (see `BandedLu`). The dense LU is the
                            // fallback in that case, so nothing is claimed for matrices
                            // that would pivot.
                            let banded = band.filter(|&(kl, ku)| {
                                // Strict column diagonal dominance is preserved by
                                // Gaussian elimination, so checking it ONCE on the
                                // untouched matrix means no step can interchange. It is
                                // O(n·band) and lets `factor` consume `system` without
                                // an n² defensive copy.
                                band_column_diagonally_dominant(&system, kl, ku)
                            });
                            match banded {
                                Some((kl, ku)) => match BandedLu::factor(system, kl, ku) {
                                    Some(banded) => {
                                        BDF_BAND_NEWTON_HITS.fetch_add(1, Ordering::Relaxed);
                                        NewtonFactor::Banded(banded)
                                    }
                                    // Unreachable given the dominance check above, but
                                    // the check is belt-and-braces, not a proof we lean
                                    // on: rebuild and take the dense path.
                                    None => NewtonFactor::Dense(
                                        (DMatrix::<f64>::identity(n, n) - jac.scale(c)).lu(),
                                    ),
                                },
                                None => NewtonFactor::Dense(system.lu()),
                            }
                        }
                    });
                    self.lu_c = Some(c);
                    self.nlu += 1;
                }
                match self.newton_bdf(fun, t_new, &y_predict, c, &psi, &scale, newton_tol) {
                    Some((iters, y_sol, d_sol)) => {
                        converged = true;
                        n_iter = iters;
                        y_new = y_sol;
                        d_corr = d_sol;
                        break;
                    }
                    None => {
                        if jac_recomputed {
                            break; // Jacobian already fresh — give up, shrink step.
                        }
                        self.current_jac = None; // force recompute next pass.
                        self.jac_diagonal = None; // stays in lockstep with `current_jac`.
                        self.jac_band = None;
                    }
                }
            }

            if !converged {
                let factor = 0.5;
                h_abs *= factor;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
                continue;
            }

            let safety = 0.9 * (2.0 * NEWTON_MAXITER as f64 + 1.0)
                / (2.0 * NEWTON_MAXITER as f64 + n_iter as f64);
            for j in 0..n {
                scale[j] = self.atol[j] + self.rtol * y_new[j].abs();
            }
            let error_norm =
                rms_norm_scaled(d_corr.iter().map(|&value| ERR_C[order] * value), &scale);

            if error_norm > 1.0 {
                let factor = MIN_FACTOR.max(safety * error_norm.powf(-1.0 / (order as f64 + 1.0)));
                h_abs *= factor;
                change_d(&mut self.d, order, factor, n);
                self.n_equal_steps = 0;
                self.lu = None;
            } else {
                // Step accepted.
                self.t_old = Some(self.t);
                self.y_old = Some(self.y.clone());
                self.f_old = Some(self.f.clone());

                self.n_equal_steps += 1;
                self.t = t_new;
                self.y = y_new.clone();
                self.h = h_abs * self.direction;
                self.f = fun(t_new, &y_new);
                self.nfev += 1;

                // Update the difference array.
                for j in 0..n {
                    self.d[order + 2][j] = d_corr[j] - self.d[order + 1][j];
                    self.d[order + 1][j] = d_corr[j];
                }
                for i in (0..=order).rev() {
                    for j in 0..n {
                        self.d[i][j] += self.d[i + 1][j];
                    }
                }

                // Order/step selection once enough equal steps have accumulated.
                if self.n_equal_steps > order {
                    let safety_sel = safety;
                    let err_m = if order > 1 {
                        rms_norm_scaled(
                            self.d[order].iter().map(|&value| ERR_C[order - 1] * value),
                            &scale,
                        )
                    } else {
                        f64::INFINITY
                    };
                    let err_p = if order < self.max_order {
                        rms_norm_scaled(
                            self.d[order + 2]
                                .iter()
                                .map(|&value| ERR_C[order + 1] * value),
                            &scale,
                        )
                    } else {
                        f64::INFINITY
                    };
                    let norms = [err_m, error_norm, err_p];
                    let mut best = 0usize;
                    let mut best_factor = f64::NEG_INFINITY;
                    for (idx, &en) in norms.iter().enumerate() {
                        // factor = en^(-1/(order-1+idx+1)) = en^(-1/(order+idx)).
                        let exp = -1.0 / (order as f64 + idx as f64);
                        let fac = if en == 0.0 {
                            f64::INFINITY
                        } else {
                            en.powf(exp)
                        };
                        if fac > best_factor {
                            best_factor = fac;
                            best = idx;
                        }
                    }
                    order = (order as isize + best as isize - 1) as usize;
                    self.order = order;
                    let factor = MAX_FACTOR.min(safety_sel * best_factor);
                    self.h = (h_abs * factor) * self.direction;
                    change_d(&mut self.d, order, factor, n);
                    self.n_equal_steps = 0;
                    self.lu = None;
                }

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

    /// Modified-Newton corrector for the BDF system at the current order
    /// (scipy `solve_bdf_system`). Solves `(I − c·J) Δ = c·f − ψ − d`, accumulating
    /// the correction `d`. Returns `Some((n_iter, y, d))` on convergence.
    #[allow(clippy::too_many_arguments)]
    fn newton_bdf<F>(
        &mut self,
        fun: &mut F,
        t_new: f64,
        y_predict: &[f64],
        c: f64,
        psi: &[f64],
        scale: &[f64],
        tol: f64,
    ) -> Option<(usize, Vec<f64>, Vec<f64>)>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let mut d = vec![0.0; n];
        let mut y = y_predict.to_vec();
        let mut dy_norm_old: Option<f64> = None;
        let factor = self.lu.as_ref()?;
        let force_alloc = BDF_FORCE_PER_ITER_ALLOC.load(Ordering::Relaxed);
        // Per-Newton-iteration linear-solve scratch, hoisted (default): `rhs` is filled in
        // place and solved in place each iteration instead of allocating a fresh `DVector`
        // and a `lu.solve` result Vec. Bit-identical: same RHS values in the same order, and
        // `solve(b) == { let mut r = b.clone(); solve_mut(&mut r); r }`.
        let mut rhs = DVector::<f64>::zeros(n);
        for k in 0..NEWTON_MAXITER {
            let f = fun(t_new, &y);
            self.nfev += 1;
            if !f.iter().all(|v| v.is_finite()) {
                return None;
            }
            let dy_owned;
            let dy: &DVector<f64> = match factor {
                NewtonFactor::Dense(lu) => {
                    if force_alloc {
                        let rhs_a =
                            DVector::from_iterator(n, (0..n).map(|j| c * f[j] - psi[j] - d[j]));
                        dy_owned = lu.solve(&rhs_a)?;
                        &dy_owned
                    } else {
                        for j in 0..n {
                            rhs[j] = c * f[j] - psi[j] - d[j];
                        }
                        if !lu.solve_mut(&mut rhs) {
                            return None;
                        }
                        &rhs
                    }
                }
                NewtonFactor::Banded(band) => {
                    for j in 0..n {
                        rhs[j] = c * f[j] - psi[j] - d[j];
                    }
                    if !band.solve_mut(&mut rhs) {
                        return None;
                    }
                    &rhs
                }
                NewtonFactor::Diagonal(diag) => {
                    // `(I − c·J) Δ = rhs` with a diagonal system is `n` divisions.
                    // `finite` guards the ONE case where the dense arm's zero-times-
                    // non-finite products would spread `NaN` across components: there
                    // we rebuild the dense LU for this iteration and run the dense
                    // computation itself, so the arms cannot diverge. `nlu` is not
                    // incremented — the dense arm counted this factorization once too.
                    let mut finite = true;
                    for j in 0..n {
                        let r = c * f[j] - psi[j] - d[j];
                        let q = r / diag[j];
                        finite &= r.is_finite() & q.is_finite();
                        rhs[j] = q;
                    }
                    if !finite {
                        let jac = self.current_jac.as_ref()?;
                        let lu = (DMatrix::<f64>::identity(n, n) - jac.scale(c)).lu();
                        for j in 0..n {
                            rhs[j] = c * f[j] - psi[j] - d[j];
                        }
                        if !lu.solve_mut(&mut rhs) {
                            return None;
                        }
                    }
                    &rhs
                }
            };
            let dy_norm = rms_norm_scaled(dy.iter().copied(), scale);

            let rate = dy_norm_old.map(|old| if old > 0.0 { dy_norm / old } else { 0.0 });
            if let Some(r) = rate
                && (r >= 1.0 || r.powi((NEWTON_MAXITER - k) as i32) / (1.0 - r) * dy_norm > tol)
            {
                return None;
            }

            for j in 0..n {
                y[j] += dy[j];
                d[j] += dy[j];
            }

            if dy_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dy_norm < tol) {
                return Some((k + 1, y, d));
            }
            dy_norm_old = Some(dy_norm);
        }
        None
    }

    fn compute_jacobian<F>(&mut self, fun: &mut F, t: f64, y: &[f64], f0: &[f64]) -> DMatrix<f64>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let eps = f64::EPSILON.sqrt();
        let mut jac = DMatrix::<f64>::zeros(self.n, self.n);
        let mut y_perturbed = y.to_vec();

        for col in 0..self.n {
            let perturb = eps * y[col].abs().max(1.0);
            y_perturbed[col] += perturb;
            let f_perturbed = fun(t, &y_perturbed);
            self.nfev += 1;
            for row in 0..self.n {
                jac[(row, col)] = (f_perturbed[row] - f0[row]) / perturb;
            }
            y_perturbed[col] = y[col];
        }

        self.njev += 1;
        jac
    }
}

/// Select initial step size for BDF solver.
pub(crate) fn select_initial_step_bdf<F>(
    fun: &mut F,
    t0: f64,
    y0: &[f64],
    direction: f64,
    rtol: f64,
    atol: &[f64],
    mode: RuntimeMode,
) -> Result<f64, crate::IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let f0 = fun(t0, y0);
    let n = y0.len();
    validate_rhs_shape(f0.len(), n)?;
    if mode == RuntimeMode::Hardened && !f0.iter().all(|value| value.is_finite()) {
        return Err(crate::IntegrateValidationError::NonFiniteF0);
    }

    let mut d0 = 0.0_f64;
    let mut d1 = 0.0_f64;
    for j in 0..n {
        let scale = atol[j] + rtol * y0[j].abs();
        d0 += (y0[j] / scale) * (y0[j] / scale);
        d1 += (f0[j] / scale) * (f0[j] / scale);
    }
    d0 = (d0 / n as f64).sqrt();
    d1 = (d1 / n as f64).sqrt();

    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    let y1: Vec<f64> = y0
        .iter()
        .zip(f0.iter())
        .map(|(yi, fi)| yi + direction * h0 * fi)
        .collect();
    let f1 = fun(t0 + direction * h0, &y1);
    validate_rhs_shape(f1.len(), n)?;
    if mode == RuntimeMode::Hardened && !f1.iter().all(|value| value.is_finite()) {
        return Err(crate::IntegrateValidationError::NonFiniteF0);
    }

    let mut d2 = 0.0_f64;
    for j in 0..n {
        let scale = atol[j] + rtol * y0[j].abs();
        d2 += ((f1[j] - f0[j]) / scale) * ((f1[j] - f0[j]) / scale);
    }
    d2 = (d2 / n as f64).sqrt() / h0;

    let max_d = if d1.is_nan() || d2.is_nan() {
        f64::NAN
    } else {
        d1.max(d2)
    };
    let h1 = if max_d <= 1e-15 || max_d.is_nan() {
        if h0.is_nan() {
            f64::NAN
        } else {
            (h0 * 1e-3).max(1e-6)
        }
    } else {
        (0.01 / max_d).powf(0.5)
    };

    if h0.is_nan() || h1.is_nan() {
        Ok(f64::NAN)
    } else {
        Ok((100.0 * h0).min(h1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BDF_FORCE_DENSE_NEWTON` and the two hit counters are process-global, and the
    /// test harness runs tests concurrently — so every test that toggles them must hold
    /// this lock or they interleave and read each other's counts. (Observed, not
    /// hypothetical: the banded test read 18 hits and then 32.)
    static NEWTON_FLAG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn rms_norm_scaled_matches_collected_reference() {
        let dy = [2.0, -3.0, 0.25, 10.0, -1.5];
        let scale = [0.5, 2.0, 0.25, 4.0, 3.0];
        let collected: Vec<f64> = dy
            .iter()
            .zip(scale.iter())
            .map(|(&value, &scale_j)| value / scale_j)
            .collect();
        let streamed = rms_norm_scaled(dy.iter().copied(), &scale);
        assert_eq!(streamed.to_bits(), rms_norm(&collected).to_bits());
        let coeff = 1.75;
        let coeff_collected: Vec<f64> = dy
            .iter()
            .zip(scale.iter())
            .map(|(&value, &scale_j)| coeff * value / scale_j)
            .collect();
        let coeff_streamed = rms_norm_scaled(dy.iter().map(|&value| coeff * value), &scale);
        assert_eq!(
            coeff_streamed.to_bits(),
            rms_norm(&coeff_collected).to_bits()
        );
        assert_eq!(
            rms_norm_scaled(std::iter::empty::<f64>(), &[]).to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    fn bdf_exponential_decay() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        let expected = (-1.0_f64).exp();
        assert!(
            (y_final - expected).abs() < 0.1,
            "y(1) = {y_final}, expected {expected}"
        );
    }

    /// The exact-diagonal Newton factorization must be BIT-IDENTICAL to the dense-LU
    /// path, not merely close: same trajectory bits, same step count, same
    /// `nfev`/`njev`/`nlu` counters. Runs both arms in the SAME binary through the
    /// `BDF_FORCE_DENSE_NEWTON` toggle over a diagonal stiff system (where the fast
    /// path is taken), a coupled system (where it must decline), and a system whose
    /// diagonal makes `1 − c·J[j][j]` land on zero for at least one step.
    #[test]
    fn bdf_diagonal_newton_is_bit_identical_to_dense_lu() {
        let _guard = NEWTON_FLAG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        fn run<F>(fun_builder: impl Fn() -> F, y0: &[f64], t_end: f64, dense: bool) -> Vec<u64>
        where
            F: FnMut(f64, &[f64]) -> Vec<f64>,
        {
            BDF_FORCE_DENSE_NEWTON.store(dense, Ordering::Relaxed);
            let mut fun = fun_builder();
            let config = BdfSolverConfig {
                t0: 0.0,
                y0,
                t_bound: t_end,
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                max_step: f64::INFINITY,
                first_step: None,
                mode: RuntimeMode::Strict,
                max_order: 5,
            };
            let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");
            let mut bits = Vec::new();
            let mut steps = 0usize;
            while solver.state() == OdeSolverState::Running {
                solver.step_with(&mut fun).expect("BDF step");
                steps += 1;
                bits.push(solver.t().to_bits());
                bits.extend(solver.y().iter().map(|v| v.to_bits()));
            }
            bits.push(steps as u64);
            bits.push(solver.nfev() as u64);
            bits.push(solver.njev() as u64);
            bits.push(solver.nlu() as u64);
            BDF_FORCE_DENSE_NEWTON.store(false, Ordering::Relaxed);
            bits
        }

        // 1. Exactly diagonal stiff decay — the fast path fires. The hit counter is
        //    the EXECUTION PROOF: without it a broken structural predicate would make
        //    this test pass by never running the code under test.
        let diagonal = || {
            move |_t: f64, y: &[f64]| {
                (0..y.len())
                    .map(|j| -(1.0 + 10.0 * j as f64) * y[j])
                    .collect::<Vec<f64>>()
            }
        };
        let y0: Vec<f64> = (0..16).map(|j| 1.0 + 0.25 * j as f64).collect();
        BDF_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        let cand = run(diagonal, &y0, 2.0, false);
        let hits = BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        let base = run(diagonal, &y0, 2.0, true);
        assert!(
            hits > 0,
            "diagonal path never executed — the comparison would be vacuous"
        );
        assert_eq!(
            BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed),
            hits,
            "the forced-dense arm must take zero diagonal factorizations"
        );
        assert_eq!(cand, base, "diagonal arm diverged from the dense LU arm");

        // 2. Coupled system — the structural test must decline and both arms agree
        //    trivially (regression guard against an over-eager diagonal predicate).
        let coupled = || {
            move |_t: f64, y: &[f64]| {
                let n = y.len();
                (0..n)
                    .map(|j| -2.0 * y[j] + 0.5 * y[(j + 1) % n])
                    .collect::<Vec<f64>>()
            }
        };
        BDF_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        let cand = run(coupled, &y0, 2.0, false);
        assert_eq!(
            BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed),
            0,
            "the structural predicate accepted a coupled Jacobian"
        );
        assert_eq!(
            cand,
            run(coupled, &y0, 2.0, true),
            "coupled system must take the dense path in both arms"
        );

        // 3. Mixed: one exactly-zero row (pure integrator) inside an otherwise
        //    diagonal system — `1 − c·0 = 1` stays admissible, and a zero-Jacobian
        //    component is the classic case that would trip a reciprocal cache.
        let mixed = || {
            move |_t: f64, y: &[f64]| {
                (0..y.len())
                    .map(|j| if j % 4 == 0 { 1.0 } else { -(j as f64) * y[j] })
                    .collect::<Vec<f64>>()
            }
        };
        assert_eq!(
            run(mixed, &y0, 1.0, false),
            run(mixed, &y0, 1.0, true),
            "mixed zero-row system diverged from the dense LU arm"
        );
    }

    /// `BandedLu` must equal `nalgebra`'s dense `LU` BIT-FOR-BIT — both the factors and
    /// the solution — including on matrices that force row interchanges.
    ///
    /// This test exists because the ODE-level fixture cannot reach the swap branch: a
    /// method-of-lines heat matrix is diagonally dominant, so `icamax` always returns
    /// the diagonal and `gauss_step_swap`'s transcription would ship untested. That is
    /// the same "the bench never executed the code under test" failure that voided a
    /// third of this repo's REJECT ledger (`docs/LEDGER_RESURRECTION.md`), so the
    /// pivot-forcing cases below are the point of the test, not an extra.
    #[test]
    fn banded_lu_is_bit_identical_to_dense_lu() {
        fn banded(n: usize, kl: usize, ku: usize, kind: u8) -> DMatrix<f64> {
            let mut m = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let (lo, hi) = (i.saturating_sub(kl), (i + ku).min(n - 1));
                    if j < lo || j > hi {
                        continue;
                    }
                    let base = ((i * 7 + j * 13) % 11) as f64 - 5.0;
                    m[(i, j)] = match kind {
                        // Diagonally dominant: no interchanges (the PDE case).
                        0 if i == j => 40.0 + i as f64,
                        0 => base,
                        // Tiny diagonal, heavy sub-diagonal: forces a swap at every step.
                        1 if i == j => 1e-13,
                        1 if i > j => 30.0 + base,
                        1 => base,
                        // Exactly-zero diagonal in one column: exercises both the swap
                        // and (in the last column) nalgebra's zero-pivot `continue`.
                        _ if i == j => {
                            if i % 4 == 2 {
                                0.0
                            } else {
                                3.0 + base
                            }
                        }
                        _ => base,
                    };
                }
            }
            m
        }

        for &(n, kl, ku) in &[(8, 1, 1), (16, 1, 1), (24, 2, 1), (32, 3, 2), (40, 1, 4)] {
            for kind in 0..3u8 {
                let a = banded(n, kl, ku, kind);
                let dense_lu = a.clone().lu();
                let Some(banded_lu) = BandedLu::factor(a.clone(), kl, ku) else {
                    // Declined: dense GEPP would interchange here. That is the
                    // documented precondition, and the caller falls back to the dense
                    // LU — nothing to compare.
                    assert!(
                        !band_column_diagonally_dominant(&a, kl, ku),
                        "declined a diagonally dominant matrix (n={n} kl={kl} ku={ku} kind={kind})"
                    );
                    continue;
                };

                let dense_factors = dense_lu.lu_internal();
                let mismatched: Vec<(usize, usize)> = (0..n)
                    .flat_map(|i| (0..n).map(move |j| (i, j)))
                    .filter(|&(i, j)| {
                        banded_lu.lu[(i, j)].to_bits() != dense_factors[(i, j)].to_bits()
                    })
                    .collect();
                assert!(
                    mismatched.is_empty(),
                    "factors differ at {mismatched:?} for n={n} kl={kl} ku={ku} kind={kind}"
                );

                let rhs = DVector::from_iterator(n, (0..n).map(|i| 1.0 + (i % 5) as f64 * 0.25));
                let mut mine = rhs.clone();
                let mut theirs = rhs.clone();
                let ok_mine = banded_lu.solve_mut(&mut mine);
                let ok_theirs = dense_lu.solve_mut(&mut theirs);
                assert_eq!(
                    ok_mine, ok_theirs,
                    "solve status differs for n={n} kl={kl} ku={ku} kind={kind}"
                );
                if ok_mine {
                    for i in 0..n {
                        assert_eq!(
                            mine[i].to_bits(),
                            theirs[i].to_bits(),
                            "solution differs at {i} for n={n} kl={kl} ku={ku} kind={kind}"
                        );
                    }
                }
            }
        }
    }

    /// End-to-end: a method-of-lines heat equation (exactly tridiagonal Jacobian) must
    /// integrate BIT-IDENTICALLY through the banded path and the dense-LU path, with
    /// the hit counter proving the banded path actually ran.
    #[test]
    fn bdf_banded_newton_is_bit_identical_to_dense_lu() {
        let _guard = NEWTON_FLAG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let n = 48;
        let heat = || {
            move |_t: f64, y: &[f64]| {
                let k = (n * n) as f64 * 0.02;
                (0..y.len())
                    .map(|j| {
                        let left = if j == 0 { 0.0 } else { y[j - 1] };
                        let right = if j + 1 == y.len() { 0.0 } else { y[j + 1] };
                        k * (left - 2.0 * y[j] + right)
                    })
                    .collect::<Vec<f64>>()
            }
        };
        let y0: Vec<f64> = (0..n).map(|j| ((j % 7) as f64) * 0.5 + 1.0).collect();

        fn run<F>(builder: impl Fn() -> F, y0: &[f64], dense: bool) -> Vec<u64>
        where
            F: FnMut(f64, &[f64]) -> Vec<f64>,
        {
            BDF_FORCE_DENSE_NEWTON.store(dense, Ordering::Relaxed);
            let mut fun = builder();
            let config = BdfSolverConfig {
                t0: 0.0,
                y0,
                t_bound: 0.05,
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                max_step: f64::INFINITY,
                first_step: None,
                mode: RuntimeMode::Strict,
                max_order: 5,
            };
            let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");
            let mut bits = Vec::new();
            while solver.state() == OdeSolverState::Running {
                solver.step_with(&mut fun).expect("BDF step");
                bits.push(solver.t().to_bits());
                bits.extend(solver.y().iter().map(|v| v.to_bits()));
            }
            bits.push(solver.nfev() as u64);
            bits.push(solver.njev() as u64);
            bits.push(solver.nlu() as u64);
            BDF_FORCE_DENSE_NEWTON.store(false, Ordering::Relaxed);
            bits
        }

        BDF_BAND_NEWTON_HITS.store(0, Ordering::Relaxed);
        let cand = run(heat, &y0, false);
        let band_hits = BDF_BAND_NEWTON_HITS.load(Ordering::Relaxed);
        let base = run(heat, &y0, true);
        assert!(
            band_hits > 0,
            "banded path never executed — the comparison would be vacuous"
        );
        assert_eq!(
            BDF_BAND_NEWTON_HITS.load(Ordering::Relaxed),
            band_hits,
            "the forced-dense arm must take zero banded factorizations"
        );
        assert_eq!(cand, base, "banded arm diverged from the dense LU arm");
    }

    #[test]
    fn bdf_rejects_nan_max_step() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::NAN,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let result = BdfSolver::new(&mut fun, config);
        assert!(matches!(
            result,
            Err(crate::IntegrateValidationError::NonFiniteMaxStep)
        ));
    }

    #[test]
    fn bdf_linear_ode() {
        let mut fun = |_t: f64, _y: &[f64]| vec![1.0];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[0.0],
            t_bound: 2.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        assert!(
            (y_final - 2.0).abs() < 0.1,
            "y(2) = {y_final}, expected 2.0"
        );
    }

    #[test]
    fn bdf_stiff_ode() {
        let mut fun = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.1,
            rtol: 1e-4,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: Some(1e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y_final = solver.y()[0];
        let expected = 0.1_f64.cos();
        assert!(
            (y_final - expected).abs() < 0.01,
            "y(0.1) = {y_final}, expected ~{expected}"
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
        assert!(solver.nlu() > 0, "should record LU factorizations");
    }

    #[test]
    fn bdf_first_step_hardened_rejects_non_finite_f0() {
        let mut fun = |_t: f64, _y: &[f64]| vec![f64::NAN];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.1,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(1e-6),
            mode: RuntimeMode::Hardened,
            max_order: 5,
        };
        let err = match BdfSolver::new(&mut fun, config) {
            Ok(_) => panic!("non-finite f0 should fail"),
            Err(err) => err,
        };
        assert_eq!(err, crate::IntegrateValidationError::NonFiniteF0);
    }

    #[test]
    fn select_initial_step_bdf_hardened_rejects_non_finite_probe_rhs() {
        let mut calls = 0;
        let mut fun = |_t: f64, _y: &[f64]| {
            calls += 1;
            if calls == 1 {
                vec![1.0]
            } else {
                vec![f64::NAN]
            }
        };

        let err = select_initial_step_bdf(
            &mut fun,
            0.0,
            &[1.0],
            1.0,
            1e-6,
            &[1e-8],
            RuntimeMode::Hardened,
        )
        .expect_err("non-finite BDF probe RHS should fail");
        assert_eq!(err, crate::IntegrateValidationError::NonFiniteF0);
    }

    #[test]
    fn bdf_linear_stiff_decay_uses_newton_counters() {
        let mut fun = |_t: f64, y: &[f64]| vec![-1000.0 * y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.01,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: 1e-3,
            first_step: Some(1e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let expected = (-10.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 5e-3,
            "y(0.01) = {}, expected {}",
            solver.y()[0],
            expected
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
        assert!(solver.nlu() > 0, "should record LU factorizations");
    }

    #[test]
    fn bdf_robertson_problem_preserves_mass() {
        let mut fun = |_t: f64, y: &[f64]| {
            vec![
                -0.04 * y[0] + 1.0e4 * y[1] * y[2],
                0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
                3.0e7 * y[1] * y[1],
            ]
        };
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0, 0.0, 0.0],
            t_bound: 1.0e-2,
            rtol: 1.0e-5,
            atol: ToleranceValue::Vector(vec![1.0e-8, 1.0e-12, 1.0e-8]),
            max_step: 1.0e-3,
            first_step: Some(1.0e-8),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let total: f64 = solver.y().iter().sum();
        assert!(
            (total - 1.0).abs() < 1.0e-6,
            "Robertson mass drifted: total={total}"
        );
        assert!(
            solver
                .y()
                .iter()
                .all(|&value| value.is_finite() && value >= -1.0e-10),
            "Robertson state must stay finite and nonnegative: {:?}",
            solver.y()
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
    }

    #[test]
    fn bdf_van_der_pol_mu_1000_stays_finite() {
        let mu = 1000.0;
        let mut fun = move |_t: f64, y: &[f64]| vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[2.0, 0.0],
            t_bound: 0.1,
            rtol: 1.0e-4,
            atol: ToleranceValue::Vector(vec![1.0e-6, 1.0e-6]),
            max_step: 1.0e-2,
            first_step: Some(1.0e-6),
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        assert!(
            solver.y().iter().all(|value| value.is_finite()),
            "Van der Pol state must stay finite: {:?}",
            solver.y()
        );
        assert!(
            (solver.y()[0] - 2.0).abs() < 0.5,
            "Van der Pol drifted unexpectedly over short interval: {:?}",
            solver.y()
        );
        assert!(solver.njev() > 0, "should record Jacobian evaluations");
    }

    #[test]
    fn bdf_2d_system() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0], -2.0 * y[1]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0, 1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 5,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        let y = solver.y();
        assert!(
            (y[0] - (-1.0_f64).exp()).abs() < 0.1,
            "y0(1) = {}, expected {}",
            y[0],
            (-1.0_f64).exp()
        );
        assert!(
            (y[1] - (-2.0_f64).exp()).abs() < 0.1,
            "y1(1) = {}, expected {}",
            y[1],
            (-2.0_f64).exp()
        );
    }

    #[test]
    fn bdf_nfev_is_tracked() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 0.5,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 3,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut fun).expect("BDF step");
        }

        assert!(solver.nfev() > 0, "should track function evaluations");
    }

    #[test]
    fn bdf_t_old_and_y_old() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let config = BdfSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            max_order: 3,
        };
        let mut solver = BdfSolver::new(&mut fun, config).expect("BDF init");

        assert!(solver.t_old().is_none());
        assert!(solver.y_old().is_none());

        solver.step_with(&mut fun).expect("BDF step");

        assert!(solver.t_old().is_some());
        assert!(solver.y_old().is_some());
    }
}
