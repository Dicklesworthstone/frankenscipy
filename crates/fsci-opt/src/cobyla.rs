#![forbid(unsafe_code)]
//! `scipy.optimize.minimize(method='COBYLA')` and `scipy.optimize.fmin_cobyla`: Powell's
//! Constrained Optimization BY Linear Approximation as modernised in PRIMA (frankenscipy-1ksfv.3).
//!
//! SciPy 1.17.1 runs COBYLA as `scipy._lib.pyprima`, a pure-Python translation of Zaikun Zhang's
//! modern-Fortran PRIMA. This module transcribes that Python statement by statement:
//! `cobyla.cobyla` (input processing, [`cobyla_core`]), `cobylb` (the trust-region loop with
//! `getcpen` and `fcratio`), `trustregion` (`trstlp`, `trstlp_sub`, `trrad`), `geometry`
//! (`setdrop_tr`, `geostep`), `update` (`updatexfc`, `findpole`, `updatepole`), `initialize`
//! (`initxfc`, `initfilt`), `common.selectx` (the filter), `redrho`, `redrat`, `checkbreak_con`,
//! the moderated extreme barrier of `common.evaluate`, and the `linalg` / `powalg` helpers
//! (`isminor`, `planerot`, `lsqr`, `qradd_Rdiag`, `qrexc_Rdiag`). SciPy's `_minimize_cobyla` and
//! the `pyprima.minimize` front end (constraint conversion, bounds as linear constraints, fixed
//! variables, projection of `x0`) are in [`minimize_cobyla`].
//!
//! The Python leaves its inner products, matrix products, sums, norms, inverses and least
//! squares to NumPy, and COBYLA's iterates follow every rounding of them. The `npx` module
//! therefore evaluates each one in the order NumPy 2.4 + OpenBLAS 0.3 use on x86-64 for these
//! sizes and memory layouts (fused multiply-add accumulation in BLAS kernel order, NumPy's
//! pairwise summation, LAPACK's LU inverse, `dgelsd` least squares), pinned against NumPy bit
//! for bit. Two parts are not reproduced bit for bit, and either can make the iterates drift
//! apart from SciPy's by rounding: `np.linalg.lstsq` with more than 25 active constraints
//! (LAPACK's divide-and-conquer SVD), where `lsqr` falls back to PRIMA's own reference
//! algorithm, and the projection of an infeasible `x0` onto linear constraints that are not
//! all equalities without bounds, which SciPy does with its own SLSQP and this module with
//! fsci's.

use crate::npx::{self, Mat};
use crate::types::{
    Bound, Constraint, ConstraintOrigin, ConstraintType, ConvergenceStatus, LinearConstraint,
    MinimizeCallback, MinimizeOptions, NewConstraint, NonlinearConstraint, OptError,
    OptimizeMethod, OptimizeResult, OptimizeTraceEntry,
};
use fsci_runtime::RuntimeMode;

// ══════════════════════════════════════════════════════════════════════
// Constants (pyprima/common/consts.py)
// ══════════════════════════════════════════════════════════════════════

const REALMAX: f64 = f64::MAX;
const REALMIN: f64 = f64::MIN_POSITIVE;
const EPS: f64 = f64::EPSILON;
const FUNCMAX: f64 = 1.0e30;
/// The value COBYLA's moderated extreme barrier gives a NaN or huge constraint value, and hence
/// the `maxcv` SciPy reports for a constraint that evaluates to NaN.
pub const CONSTRMAX: f64 = FUNCMAX;
const BOUNDMAX: f64 = REALMAX / 4.0;
const RHOBEG_DEFAULT: f64 = 1.0;
const RHOEND_DEFAULT: f64 = 1.0e-6;
const CWEIGHT_DEFAULT: f64 = 1.0e8;
const ETA1_DEFAULT: f64 = 0.1;
const GAMMA1_DEFAULT: f64 = 0.5;
const GAMMA2_DEFAULT: f64 = 2.0;
const MIN_MAXFILT: usize = 200;
const MAXFILT_DEFAULT: usize = 2000;
/// `min(PRIMA_MAX_HIST_MEM_MB * 10**6, iinfo(int32).max)` bytes.
const MAXHISTMEM: usize = 300_000_000;

/// SciPy's `_minimize_cobyla` defaults.
const SCIPY_RHOBEG: f64 = 1.0;
const SCIPY_TOL: f64 = 1.0e-4;
const SCIPY_MAXITER: usize = 1000;
/// `fmin_cobyla`'s `catol`.
const FMIN_CATOL: f64 = 2.0e-4;

const INFEASIBLE_MESSAGE: &str = "Did not converge to a solution satisfying the constraints. See `maxcv` for the magnitude of the violation.";

// ══════════════════════════════════════════════════════════════════════
// Exit flags (pyprima/common/infos.py, message.py)
// ══════════════════════════════════════════════════════════════════════

/// PRIMA's exit flag (`result.info`, SciPy's `status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CobylaInfo {
    /// 0: the trust-region radius reached `rhoend`.
    SmallTrRadius,
    /// 1: a feasible point with `f <= f_target` was found.
    FtargetAchieved,
    /// 3: `maxfun` evaluations were spent.
    MaxfunReached,
    /// 20: `10 * maxfun` trust-region iterations were spent.
    MaxtrReached,
    /// -1: NaN or Inf occurred in x.
    NanInfX,
    /// -2: the objective or a constraint returned NaN/+Inf past the barrier.
    NanInfF,
    /// 7: the inverse of the simplex could not be kept accurate.
    DamagingRounding,
    /// 30: the callback asked to stop.
    CallbackTerminate,
}

impl CobylaInfo {
    /// The integer SciPy reports as `status`.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::SmallTrRadius => 0,
            Self::FtargetAchieved => 1,
            Self::MaxfunReached => 3,
            Self::MaxtrReached => 20,
            Self::NanInfX => -1,
            Self::NanInfF => -2,
            Self::DamagingRounding => 7,
            Self::CallbackTerminate => 30,
        }
    }

    /// `get_info_string('COBYLA', info)`.
    #[must_use]
    pub fn message(self) -> String {
        let reason = match self {
            Self::FtargetAchieved => "the target function value is achieved.",
            Self::MaxfunReached => "the objective function has been evaluated MAXFUN times.",
            Self::MaxtrReached => "the maximal number of trust region iterations has been reached.",
            Self::SmallTrRadius => "the trust region radius reaches its lower bound.",
            Self::NanInfX => "NaN or Inf occurs in x.",
            Self::NanInfF => "the objective function returns NaN/+Inf.",
            Self::DamagingRounding => "rounding errors are becoming damaging.",
            Self::CallbackTerminate => "the callback function requested termination",
        };
        format!("Return from COBYLA because {reason}")
    }
}

// ══════════════════════════════════════════════════════════════════════
// Python and NumPy scalar semantics
// ══════════════════════════════════════════════════════════════════════

/// Python's builtin `max(a, b)`: `a` unless `b > a` (so a NaN `b` is ignored, a NaN `a` kept).
fn py_max2(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's builtin `min(a, b)`.
fn py_min2(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Python's builtin `max(iterable)`.
fn py_max(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |acc, &x| py_max2(acc, x))
}

/// Python's builtin `min(iterable)`.
fn py_min(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |acc, &x| py_min2(acc, x))
}

/// `np.maximum(a, b)`: NaN-propagating; on a tie (`0.0` vs `-0.0`) NumPy returns `b`.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// `np.minimum(a, b)`.
fn np_minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}

/// `np.max(v)`: NaN-propagating; a later equal element wins (as NumPy's reduction does for
/// signed zeros).
fn np_max(v: &[f64]) -> f64 {
    let mut acc = v[0];
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        if x >= acc {
            acc = x;
        }
    }
    acc
}

/// `np.min(v)`.
fn np_min(v: &[f64]) -> f64 {
    let mut acc = v[0];
    for &x in v {
        if x.is_nan() {
            return f64::NAN;
        }
        if x <= acc {
            acc = x;
        }
    }
    acc
}

/// `np.max(np.append(0, v))`.
fn np_max0(v: &[f64]) -> f64 {
    let mut all = Vec::with_capacity(v.len() + 1);
    all.push(0.0);
    all.extend_from_slice(v);
    np_max(&all)
}

/// `np.argmax(v)`: the first NaN, else the first maximum.
fn np_argmax(v: &[f64]) -> usize {
    if let Some(i) = v.iter().position(|x| x.is_nan()) {
        return i;
    }
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `np.argmin(v)`.
fn np_argmin(v: &[f64]) -> usize {
    if let Some(i) = v.iter().position(|x| x.is_nan()) {
        return i;
    }
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[best] {
            best = i;
        }
    }
    best
}

/// `np.ma.array(v, mask=mask).argmin()`: masked entries read as `+inf`.
fn masked_argmin(v: &[f64], mask: impl Fn(usize) -> bool) -> usize {
    let filled: Vec<f64> = (0..v.len())
        .map(|i| if mask(i) { f64::INFINITY } else { v[i] })
        .collect();
    np_argmin(&filled)
}

/// `np.ma.array(v, mask=mask).argmax()`: masked entries read as `-inf`.
fn masked_argmax(v: &[f64], mask: impl Fn(usize) -> bool) -> usize {
    let filled: Vec<f64> = (0..v.len())
        .map(|i| if mask(i) { f64::NEG_INFINITY } else { v[i] })
        .collect();
    np_argmax(&filled)
}

/// `np.sign(x)`.
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// A Python index into a sequence of length `len`, negative values counting from the end.
fn py_index(i: isize, len: usize) -> usize {
    if i < 0 {
        // A negative index wraps (Python semantics); `len` > 0 whenever this is reached.
        len.wrapping_sub(i.unsigned_abs())
    } else {
        i.unsigned_abs()
    }
}

// ══════════════════════════════════════════════════════════════════════
// pyprima/common/linalg.py and powalg.py
// ══════════════════════════════════════════════════════════════════════

/// `isminor(x, ref)`: is `x` negligible against `ref` (Powell's 0.1 sensitivity)?
fn isminor(x: f64, reference: f64) -> bool {
    let sensitivity = 0.1;
    let refa = reference.abs() + sensitivity * x.abs();
    let refb = reference.abs() + 2.0 * sensitivity * x.abs();
    reference.abs() >= refa || refa >= refb
}

/// `planerot(x)`: the Givens rotation `[[c, s], [-s, c]]` zeroing `x[1]`, returned as
/// `(c, s, g10)` with `g10` the lower-left entry (`+0.0` where PRIMA writes the integer `-0`).
fn planerot(x0: f64, x1: f64) -> (f64, f64, f64) {
    if x0.is_nan() || x1.is_nan() {
        (1.0, 0.0, 0.0)
    } else if x0.is_infinite() && x1.is_infinite() {
        let r = 1.0 / 2.0_f64.sqrt();
        let s = r * np_sign(x1);
        (r * np_sign(x0), s, -s)
    } else if x0.abs() <= 0.0 && x1.abs() <= 0.0 {
        (1.0, 0.0, 0.0)
    } else if x1.abs() <= EPS * x0.abs() {
        (np_sign(x0), 0.0, 0.0)
    } else if x0.abs() <= EPS * x1.abs() {
        let s = np_sign(x1);
        (0.0, s, -s)
    } else {
        let lo = REALMIN.sqrt();
        let hi = (REALMAX / 2.1).sqrt();
        let (c, s) = if lo < x0.abs() && x0.abs() < hi && lo < x1.abs() && x1.abs() < hi {
            let r = npx::norm(&[x0, x1]);
            (x0 / r, x1 / r)
        } else if x0.abs() > x1.abs() {
            let t = x1 / x0;
            let u = py_max2(py_max2(1.0, t.abs()), (1.0 + t * t).sqrt()) * np_sign(x0);
            (1.0 / u, t / u)
        } else {
            let t = x0 / x1;
            let u = py_max2(py_max2(1.0, t.abs()), (1.0 + t * t).sqrt()) * np_sign(x1);
            (t / u, 1.0 / u)
        };
        (c, s, -s)
    }
}

/// `qradd_Rdiag(c, Q, Rdiag, n)`: update the QR factorization when column `c` joins the active
/// set; returns the new `n` (unchanged when `c` is in the range of the current columns).
fn qradd_rdiag(c: &[f64], c_unit: bool, q: &mut Mat, rdiag: &mut [f64], n: usize) -> usize {
    let m = q.cols;
    let cq0 = npx::vec_mat(c, c_unit, q, m);
    let cabs: Vec<f64> = c.iter().map(|v| v.abs()).collect();
    let qabs = q.map(f64::abs);
    let cqa = npx::vec_mat(&cabs, true, &qabs, m);
    let mut cq: Vec<f64> = cq0
        .iter()
        .zip(&cqa)
        .map(|(&cqi, &cqai)| if isminor(cqi, cqai) { 0.0 } else { cqi })
        .collect();
    let mut k = m as isize - 2;
    while k >= n as isize {
        let ku = k.unsigned_abs();
        if cq[ku + 1].abs() > 0.0 {
            let g = planerot(cq[ku], cq[ku + 1]);
            npx::rotate_cols(q, ku, ku + 1, ku, ku + 1, g);
            cq[ku] = cq[ku].hypot(cq[ku + 1]);
        }
        k -= 1;
    }
    let mut n_new = n;
    if n_new < m && cq[n_new].abs() > EPS * EPS && !isminor(cq[n_new], cqa[n_new]) {
        n_new += 1;
    }
    if n_new >= 1 && n_new - 1 < m {
        rdiag[n_new - 1] = cq[n_new - 1];
    }
    n_new
}

/// `qrexc_Rdiag(A, Q, Rdiag, i)`: reorder columns `[i, i+1, .., n-1]` of `A` (shape `m x n`) to
/// `[i+1, .., n-1, i]` in the QR factorization.
fn qrexc_rdiag(a: &Mat, q: &mut Mat, rdiag: &mut [f64], i: usize) {
    let n = a.cols;
    let unit = q.cols == 1 && a.cols == 1;
    for k in i..n.saturating_sub(1) {
        let v = npx::dot(&q.col(k), &a.col(k + 1), unit);
        let g = planerot(rdiag[k + 1], v);
        npx::rotate_cols(q, k + 1, k, k, k + 1, g);
    }
    for k in i..n.saturating_sub(1) {
        rdiag[k] = npx::dot(&q.col(k), &a.col(k + 1), unit);
    }
    rdiag[n - 1] = npx::dot(&q.col(n - 1), &a.col(i), unit);
}

/// `lsqr(A, b, Q, Rdiag)` with `Q = z[:, :nact]`: SciPy's pyprima solves this least-squares
/// problem by `np.linalg.lstsq` (LAPACK `dgelsd`), reproduced by [`npx::lstsq`] for up to 25
/// columns. Beyond that (and for data LAPACK would rescale) this is PRIMA's own reference
/// `lsqr` (Powell's back substitution on the QR factors `trstlp` maintains), the same solution
/// up to rounding.
fn lsqr(a: &Mat, b: &[f64], z: &Mat, rdiag: &[f64]) -> Vec<f64> {
    if let Some(x) = npx::lstsq(a, b) {
        return x;
    }
    let (m, n) = (a.rows, a.cols);
    let rank = m.min(n);
    let mut x = vec![0.0_f64; n];
    let mut y = b.to_vec();
    let q_unit = z.cols == 1;
    for i in (0..rank).rev() {
        let qcol = z.col(i);
        let yq = npx::dot(&y, &qcol, q_unit);
        let yabs: Vec<f64> = y.iter().map(|v| v.abs()).collect();
        let qabs: Vec<f64> = qcol.iter().map(|v| v.abs()).collect();
        let yqa = npx::dot(&yabs, &qabs, true);
        if isminor(yq, yqa) {
            x[i] = 0.0;
        } else {
            x[i] = yq / rdiag[i];
            for (r, yr) in y.iter_mut().enumerate() {
                *yr -= x[i] * a.at(r, i);
            }
        }
    }
    x
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/trustregion.py
// ══════════════════════════════════════════════════════════════════════

/// `trstlp(A, b, delta, g)`: the trust-region step. Stage 1 minimizes the largest violation of
/// the linearized constraints `A^T d <= b` within `||d|| <= delta`; stage 2 then minimizes
/// `g^T d` without increasing that violation.
fn trstlp(a: &Mat, b: &[f64], delta: f64, g: &[f64]) -> Vec<f64> {
    let (n, m) = (a.rows, a.cols);
    let mut a_aug = Mat::zeros(n, m + 1);
    for i in 0..n {
        for j in 0..m {
            a_aug.set(i, j, a.at(i, j));
        }
        a_aug.set(i, m, g[i]);
    }
    let mut b_aug = b.to_vec();
    b_aug.push(0.0);
    for i in 0..=m {
        let col: Vec<f64> = a_aug.col(i).iter().map(|v| v.abs()).collect();
        let maxval = py_max(&col);
        if maxval > 1.0e12 {
            let modscal = py_max2(2.0 * REALMIN, 1.0 / maxval);
            for r in 0..n {
                a_aug.set(r, i, a_aug.at(r, i) * modscal);
            }
            b_aug[i] *= modscal;
        }
    }
    let mut iact = vec![0_usize; m + 1];
    let mut vmultc = vec![0.0_f64; m + 1];
    let stage1 = trstlp_sub(
        1,
        iact[..m].to_vec(),
        0,
        &a_aug,
        m,
        &b_aug[..m],
        delta,
        vec![0.0; n],
        vmultc[..m].to_vec(),
        Mat::zeros(n, n),
    );
    iact[..m].copy_from_slice(&stage1.iact);
    vmultc[..m].copy_from_slice(&stage1.vmultc);
    let stage2 = trstlp_sub(
        2,
        iact,
        stage1.nact,
        &a_aug,
        m + 1,
        &b_aug,
        delta,
        stage1.d,
        vmultc,
        stage1.z,
    );
    stage2.d
}

struct TrstlpState {
    iact: Vec<usize>,
    nact: usize,
    d: Vec<f64>,
    vmultc: Vec<f64>,
    z: Mat,
}

/// `trstlp_sub`: one stage of `trstlp`. `a` is `A_aug` of which the first `mcon` columns are
/// this stage's constraints (its row stride stays `A_aug`'s), `b` their right-hand sides.
fn trstlp_sub(
    stage: u8,
    iact_in: Vec<usize>,
    nact_in: usize,
    a: &Mat,
    mcon: usize,
    b: &[f64],
    delta: f64,
    d_in: Vec<f64>,
    vmultc_in: Vec<f64>,
    z_in: Mat,
) -> TrstlpState {
    let nv = a.rows;
    let acol_unit = a.cols == 1;
    let zcol_unit = nv == 1;
    let acol = |k: usize| a.col(k);
    let mut zdasav = vec![0.0_f64; nv];
    let mut vmultd = vec![0.0_f64; vmultc_in.len()];
    let mut zdota = vec![0.0_f64; nv];

    let mut iact;
    let mut nact;
    let mut d;
    let mut vmultc: Vec<f64>;
    let mut z;
    let mut cviol;
    let mut icon: isize;
    let num_constraints;
    let mut sdirn = vec![0.0_f64; nv];
    if stage == 1 {
        iact = (0..mcon).collect::<Vec<usize>>();
        nact = 0;
        d = vec![0.0_f64; nv];
        let neg_b: Vec<f64> = b.iter().map(|v| -v).collect();
        cviol = np_max0(&neg_b);
        vmultc = b.iter().map(|&bi| cviol + bi).collect();
        z = Mat::eye(nv);
        if mcon == 0 || cviol <= 0.0 {
            return TrstlpState {
                iact,
                nact,
                d,
                vmultc,
                z,
            };
        }
        if b.iter().all(|v| v.is_nan()) {
            return TrstlpState {
                iact,
                nact,
                d,
                vmultc,
                z,
            };
        }
        // np.nanargmax(-b): the first maximum, NaN ignored.
        let mut best: Option<usize> = None;
        for (i, &v) in neg_b.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if best.is_none_or(|bi| v > neg_b[bi]) {
                best = Some(i);
            }
        }
        icon = best.map_or(0, |i| i as isize);
        num_constraints = mcon;
    } else {
        iact = iact_in;
        nact = nact_in;
        d = d_in;
        vmultc = vmultc_in;
        z = z_in;
        if npx::dot(&d, &d, true) >= delta * delta {
            return TrstlpState {
                iact,
                nact,
                d,
                vmultc,
                z,
            };
        }
        iact[mcon - 1] = mcon - 1;
        vmultc[mcon - 1] = 0.0;
        num_constraints = mcon - 1;
        icon = (mcon - 1) as isize;
        let ad = npx::vec_mat(&d, true, a, num_constraints);
        let diff: Vec<f64> = (0..num_constraints).map(|i| ad[i] - b[i]).collect();
        cviol = np_max0(&diff);
    }
    for k in 0..nact {
        zdota[k] = npx::dot(&z.col(k), &acol(iact[k]), zcol_unit && acol_unit);
    }

    let mut optold = REALMAX;
    let mut nactold = nact;
    let mut nfail = 0;
    let maxiter = 10000.min(100 * num_constraints.max(nv));
    for _ in 0..maxiter {
        let optnew = if stage == 1 {
            cviol
        } else {
            npx::dot(&d, &acol(mcon - 1), acol_unit)
        };
        if optnew < optold || nact > nactold {
            nactold = nact;
            nfail = 0;
        } else {
            nfail += 1;
        }
        optold = np_minimum(optold, optnew);
        if nfail == 3 {
            break;
        }

        if icon >= nact as isize {
            let ic = icon.unsigned_abs();
            zdasav[..nact].copy_from_slice(&zdota[..nact]);
            let nactsav = nact;
            nact = qradd_rdiag(&acol(iact[ic]), acol_unit, &mut z, &mut zdota, nact);
            if nact == nactsav + 1 {
                if nact != ic + 1 {
                    vmultc[ic] = vmultc[nact - 1];
                    vmultc[nact - 1] = 0.0;
                    iact.swap(ic, nact - 1);
                } else {
                    vmultc[nact - 1] = 0.0;
                }
            } else {
                let sub = a.select_cols(&iact[..nact]);
                let vd = lsqr(&sub, &acol(iact[ic]), &z, &zdasav[..nact]);
                vmultd[..nact].copy_from_slice(&vd);
                if !(0..nact).any(|i| vmultd[i] > 0.0 && iact[i] <= num_constraints) {
                    break;
                }
                for v in vmultd.iter_mut().take(mcon).skip(nact) {
                    *v = -1.0;
                }
                let fracmult: Vec<f64> = (0..nact)
                    .map(|i| {
                        if vmultd[i] > 0.0 && iact[i] <= num_constraints {
                            vmultc[i] / vmultd[i]
                        } else {
                            REALMAX
                        }
                    })
                    .collect();
                let frac = py_min(&fracmult);
                for i in 0..nact {
                    vmultc[i] = np_maximum(0.0, vmultc[i] - frac * vmultd[i]);
                }
                if zdota[nact - 1].is_nan() || zdota[nact - 1].abs() <= EPS * EPS {
                    break;
                }
                vmultc[ic] = 0.0;
                vmultc[nact - 1] = frac;
                iact.swap(ic, nact - 1);
            }

            if stage == 2 && iact[nact - 1] != mcon - 1 {
                if nact <= 1 {
                    break;
                }
                let sub = a.select_cols(&iact[..nact]);
                qrexc_rdiag(&sub, &mut z, &mut zdota[..nact], nact - 2);
                iact.swap(nact - 2, nact - 1);
                vmultc.swap(nact - 2, nact - 1);
            }

            if zdota[nact - 1].is_nan() || zdota[nact - 1].abs() <= EPS * EPS {
                break;
            }

            let zc = z.col(nact - 1);
            if stage == 1 {
                let s =
                    (npx::dot(&sdirn, &acol(iact[nact - 1]), acol_unit) + 1.0) / zdota[nact - 1];
                for (sd, &zv) in sdirn.iter_mut().zip(&zc) {
                    *sd -= s * zv;
                }
            } else {
                let s = -1.0 / zdota[nact - 1];
                sdirn = zc.iter().map(|&zv| s * zv).collect();
            }
        } else {
            let ic = icon.unsigned_abs();
            let sub = a.select_cols(&iact[..nact]);
            qrexc_rdiag(&sub, &mut z, &mut zdota[..nact], ic);
            iact[ic..nact].rotate_left(1);
            vmultc[ic..nact].rotate_left(1);
            nact -= 1;
            if nact > 0 && (zdota[nact - 1].is_nan() || zdota[nact - 1].abs() <= EPS * EPS) {
                break;
            }
            if stage == 1 {
                let zc = z.col(nact);
                let s = npx::dot(&sdirn, &zc, zcol_unit);
                for (sd, &zv) in sdirn.iter_mut().zip(&zc) {
                    *sd -= s * zv;
                }
            } else {
                let k = py_index(nact as isize - 1, nv);
                let s = -1.0 / zdota[k];
                sdirn = z.col(k).iter().map(|&zv| s * zv).collect();
            }
        }

        let dd = delta * delta - npx::dot(&d, &d, true);
        let ss = npx::dot(&sdirn, &sdirn, true);
        let sd = npx::dot(&sdirn, &d, true);
        if dd <= 0.0 || ss <= EPS * delta * delta || sd.is_nan() {
            break;
        }
        let sqrtd = py_max2(
            py_max2((ss * dd + sd * sd).sqrt(), sd.abs()),
            (ss * dd).sqrt(),
        );
        let mut step = if sd > 0.0 {
            dd / (sqrtd + sd)
        } else {
            (sqrtd - sd) / ss
        };
        if step <= 0.0 || !step.is_finite() {
            break;
        }
        if stage == 1 {
            if isminor(cviol, step) {
                break;
            }
            step = py_min2(step, cviol);
        }

        let dnew: Vec<f64> = d
            .iter()
            .zip(&sdirn)
            .map(|(&di, &si)| di + step * si)
            .collect();
        // `A[:, iact[..]]` is a Fortran-ordered copy, so `x @ A[:, iact]` runs `dgemv_t`.
        if stage == 1 {
            let v = npx::mat_vec(&a.select_cols_t(&iact[..nact]), &dnew, true);
            let diff: Vec<f64> = (0..nact).map(|i| v[i] - b[iact[i]]).collect();
            cviol = np_max0(&diff);
        }
        let vd = lsqr(&a.select_cols(&iact[..nact]), &dnew, &z, &zdota[..nact]);
        for i in 0..nact {
            vmultd[i] = -vd[i];
        }
        if stage == 2 {
            let k = py_index(nact as isize - 1, vmultd.len());
            vmultd[k] = py_max2(0.0, vmultd[k]);
        }
        let full_t = a.select_cols_t(&iact[..mcon]);
        let av = npx::mat_vec(&full_t, &dnew, true);
        let mut cvshift: Vec<f64> = (0..mcon).map(|i| cviol - (av[i] - b[iact[i]])).collect();
        let dabs: Vec<f64> = dnew.iter().map(|v| v.abs()).collect();
        let avs = npx::mat_vec(&full_t.map(f64::abs), &dabs, true);
        let cvsabs: Vec<f64> = (0..mcon)
            .map(|i| (avs[i] + b[iact[i]].abs()) + cviol)
            .collect();
        for i in 0..mcon {
            if isminor(cvshift[i], cvsabs[i]) {
                cvshift[i] = 0.0;
            }
        }
        vmultd[nact..mcon].copy_from_slice(&cvshift[nact..mcon]);

        let mut fracmult = Vec::with_capacity(mcon + 1);
        fracmult.push(1.0);
        for i in 0..vmultd.len() {
            fracmult.push(if vmultd[i] < 0.0 {
                vmultc[i] / (vmultc[i] - vmultd[i])
            } else {
                REALMAX
            });
        }
        icon = np_argmin(&fracmult) as isize - 1;
        let frac = py_min(&fracmult);

        let dold = d.clone();
        d = d
            .iter()
            .zip(&dnew)
            .map(|(&di, &dn)| (1.0 - frac) * di + frac * dn)
            .collect();
        vmultc = vmultc
            .iter()
            .zip(&vmultd)
            .map(|(&vc, &vd)| np_maximum(0.0, (1.0 - frac) * vc + frac * vd))
            .collect();
        let dsum: Vec<f64> = d.iter().map(|v| v.abs()).collect();
        let vsum: Vec<f64> = vmultc.iter().map(|v| v.abs()).collect();
        if !(npx::sum(&dsum).is_finite() && npx::sum(&vsum).is_finite()) {
            d = dold;
            break;
        }
        if stage == 1 {
            let v = npx::vec_mat(&d, true, a, mcon);
            let diff: Vec<f64> = (0..mcon).map(|i| v[i] - b[i]).collect();
            cviol = np_max0(&diff);
        }
        if icon < 0 || icon >= mcon as isize {
            break;
        }
    }
    TrstlpState {
        iact,
        nact,
        d,
        vmultc,
        z,
    }
}

/// `trrad`: update the trust-region radius from the reduction ratio.
fn trrad(
    delta_in: f64,
    dnorm: f64,
    eta1: f64,
    eta2: f64,
    gamma1: f64,
    gamma2: f64,
    ratio: f64,
) -> f64 {
    if ratio <= eta1 {
        gamma1 * dnorm
    } else if ratio <= eta2 {
        py_max2(gamma1 * delta_in, dnorm)
    } else {
        py_max2(gamma1 * delta_in, gamma2 * dnorm)
    }
}

// ══════════════════════════════════════════════════════════════════════
// common/ratio.py, redrho.py, checkbreak.py, evaluate.py
// ══════════════════════════════════════════════════════════════════════

/// `redrat(ared, pred, rshrink)`.
fn redrat(ared: f64, pred: f64, rshrink: f64) -> f64 {
    if ared.is_nan() {
        -REALMAX
    } else if pred.is_nan() || pred <= 0.0 {
        if ared > 0.0 { rshrink / 2.0 } else { -REALMAX }
    } else if pred == f64::INFINITY && ared == f64::INFINITY {
        1.0
    } else if pred == f64::INFINITY && ared == f64::NEG_INFINITY {
        -REALMAX
    } else {
        ared / pred
    }
}

/// `redrho(rho, rhoend)`.
fn redrho(rho_in: f64, rhoend: f64) -> f64 {
    let rho_ratio = rho_in / rhoend;
    if rho_ratio > 250.0 {
        0.1 * rho_in
    } else if rho_ratio <= 16.0 {
        rhoend
    } else {
        rho_ratio.sqrt() * rhoend
    }
}

/// `checkbreak_con`. PRIMA asserts that `x` holds no NaN and that `f`, `cstrv` are neither NaN
/// nor `+inf` (SciPy raises an `AssertionError` otherwise); those cases report
/// `NanInfX` / `NanInfF` here.
fn checkbreak_con(
    maxfun: usize,
    nf: usize,
    cstrv: f64,
    ctol: f64,
    f: f64,
    ftarget: f64,
    x: &[f64],
) -> Option<CobylaInfo> {
    let mut info = None;
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        info = Some(CobylaInfo::NanInfX);
    }
    if f.is_nan() || f == f64::INFINITY || cstrv.is_nan() || cstrv == f64::INFINITY {
        info = Some(CobylaInfo::NanInfF);
    }
    if cstrv <= ctol && f <= ftarget {
        info = Some(CobylaInfo::FtargetAchieved);
    }
    if nf >= maxfun {
        info = Some(CobylaInfo::MaxfunReached);
    }
    info
}

fn moderatex(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                0.0
            } else {
                v.clamp(-REALMAX, REALMAX)
            }
        })
        .collect()
}

fn moderatef(f: f64) -> f64 {
    if f.is_nan() {
        FUNCMAX
    } else {
        f.clamp(-REALMAX, FUNCMAX)
    }
}

fn moderatec(c: f64) -> f64 {
    if c.is_nan() {
        CONSTRMAX
    } else {
        c.clamp(-CONSTRMAX, CONSTRMAX)
    }
}

/// The objective and the nonlinear constraints (PRIMA sign: feasible when `<= 0`) at `x`.
type Calcfc<'c> = dyn FnMut(&[f64]) -> Result<(f64, Vec<f64>), OptError> + 'c;

/// `evaluate(calcfc, x, m_nlcon, amat, bvec)` with the moderated extreme barrier.
fn evaluate(
    calcfc: &mut Calcfc<'_>,
    x: &[f64],
    m_nlcon: usize,
    amat: Option<&Mat>,
    bvec: &[f64],
) -> Result<(f64, Vec<f64>), OptError> {
    let m_lcon = bvec.len();
    let mut constr = vec![0.0_f64; m_lcon + m_nlcon];
    if let Some(amat) = amat {
        let v = npx::mat_vec(amat, x, true);
        for i in 0..m_lcon {
            constr[i] = v[i] - bvec[i];
        }
    }
    if x.iter().any(|v| v.is_nan()) {
        let f = npx::sum(x);
        for c in constr.iter_mut().skip(m_lcon) {
            *c = f;
        }
        return Ok((f, constr));
    }
    let (f, nl) = calcfc(&moderatex(x))?;
    if nl.len() != m_nlcon {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "COBYLA: the constraints returned {} values, {m_nlcon} at x0",
                nl.len()
            ),
        });
    }
    for (c, v) in constr.iter_mut().skip(m_lcon).zip(nl) {
        *c = moderatec(v);
    }
    Ok((moderatef(f), constr))
}

// ══════════════════════════════════════════════════════════════════════
// common/selectx.py: the filter
// ══════════════════════════════════════════════════════════════════════

/// `isbetter(f1, c1, f2, c2, ctol)`.
fn isbetter(f1: f64, c1: f64, f2: f64, c2: f64, ctol: f64) -> bool {
    let bad = |f: f64, c: f64| f.is_nan() || c.is_nan() || f == f64::INFINITY || c == f64::INFINITY;
    let mut is_better = bad(f1, c1) && !bad(f2, c2);
    is_better = is_better || (f1 < f2 && c1 <= c2);
    is_better = is_better || (f1 <= f2 && c1 < c2);
    let cref = 10.0 * py_max2(EPS, py_min2(ctol, 1.0e-2 * CONSTRMAX));
    is_better || (f1 < REALMAX && c1 <= ctol && (c2 > py_max2(ctol, cref) || c2.is_nan()))
}

/// `xfilt`, `ffilt`, `cfilt`, `confilt` (only the first `nfilt` entries; Python preallocates).
struct Filter {
    x: Vec<Vec<f64>>,
    f: Vec<f64>,
    c: Vec<f64>,
    con: Vec<Vec<f64>>,
    maxfilt: usize,
}

/// `savefilt`: add `x` to the filter unless a member is better, dropping the members `x` beats.
fn savefilt(
    filt: &mut Filter,
    cstrv: f64,
    ctol: f64,
    cweight: f64,
    f: f64,
    x: &[f64],
    constr: &[f64],
) {
    let nfilt = filt.f.len();
    if (0..nfilt).any(|i| isbetter(filt.f[i], filt.c[i], f, cstrv, ctol))
        || (0..nfilt).any(|i| filt.f[i] <= f && filt.c[i] <= cstrv)
    {
        return;
    }
    let mut keep: Vec<bool> = (0..nfilt)
        .map(|i| !isbetter(f, cstrv, filt.f[i], filt.c[i], ctol))
        .collect();
    if keep.iter().filter(|&&k| k).count() == filt.maxfilt {
        let shifted: Vec<f64> = filt.c.iter().map(|&c| np_maximum(c - ctol, 0.0)).collect();
        let phi: Vec<f64> = if cweight <= 0.0 {
            filt.f.clone()
        } else if cweight == f64::INFINITY {
            shifted.clone()
        } else {
            (0..nfilt)
                .map(|i| {
                    let mut p = np_maximum(filt.f[i], -REALMAX);
                    if p.is_nan() {
                        p = -REALMAX;
                    }
                    p + cweight * shifted[i]
                })
                .collect()
        };
        let phimax = py_max(&phi);
        let sel: Vec<f64> = (0..nfilt)
            .filter(|&i| phi[i] >= phimax)
            .map(|i| shifted[i])
            .collect();
        let cref = py_max(&sel);
        let sel: Vec<f64> = (0..nfilt)
            .filter(|&i| shifted[i] >= cref)
            .map(|i| filt.f[i])
            .collect();
        let fref = py_max(&sel);
        let mut kworst = masked_argmax(&filt.c, |i| filt.f[i] > fref);
        if kworst >= keep.len() {
            kworst = 0;
        }
        keep[kworst] = false;
    }
    let mut k = 0;
    filt.x.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let mut k = 0;
    filt.f.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let mut k = 0;
    filt.c.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    let mut k = 0;
    filt.con.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    filt.x.push(x.to_vec());
    filt.f.push(f);
    filt.c.push(cstrv);
    filt.con.push(constr.to_vec());
}

/// `selectx(fhist, chist, cweight, ctol)`: the index of the point to return.
fn selectx(fhist: &[f64], chist: &[f64], cweight: f64, ctol: f64) -> usize {
    let nhist = fhist.len();
    let any = |fr: f64, cr: f64| (0..nhist).any(|i| fhist[i] < fr && chist[i] < cr);
    let (fref, cref) = if any(FUNCMAX, CONSTRMAX) {
        (FUNCMAX, CONSTRMAX)
    } else if any(REALMAX, CONSTRMAX) {
        (REALMAX, CONSTRMAX)
    } else if any(FUNCMAX, REALMAX) {
        (FUNCMAX, REALMAX)
    } else {
        (REALMAX, REALMAX)
    };
    if !any(fref, cref) {
        return nhist - 1;
    }
    let shifted: Vec<f64> = chist.iter().map(|&c| np_maximum(c - ctol, 0.0)).collect();
    let sel: Vec<f64> = (0..nhist)
        .filter(|&i| fhist[i] < fref)
        .map(|i| shifted[i])
        .collect();
    let cmin = np_min(&sel);
    let cref = np_maximum(EPS, 2.0 * cmin);
    let phi: Vec<f64> = if cweight <= 0.0 {
        fhist.to_vec()
    } else if cweight == f64::INFINITY {
        shifted.clone()
    } else {
        (0..nhist)
            .map(|i| np_maximum(fhist[i], -REALMAX) + cweight * shifted[i])
            .collect()
    };
    let sel: Vec<f64> = (0..nhist)
        .filter(|&i| fhist[i] < fref && shifted[i] <= cref)
        .map(|i| phi[i])
        .collect();
    let phimin = np_min(&sel);
    let sel: Vec<f64> = (0..nhist)
        .filter(|&i| fhist[i] < fref && phi[i] <= phimin)
        .map(|i| shifted[i])
        .collect();
    let cref = np_min(&sel);
    let sel: Vec<f64> = (0..nhist)
        .filter(|&i| shifted[i] <= cref)
        .map(|i| fhist[i])
        .collect();
    let fref = np_min(&sel);
    masked_argmin(chist, |i| fhist[i] > fref)
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/geometry.py
// ══════════════════════════════════════════════════════════════════════

/// `sum(A**2, axis=0)` of the first `ncols` columns of `a`.
fn colsq_sums(a: &Mat, ncols: usize) -> Vec<f64> {
    let sq = a.first_cols(ncols).map(|v| v * v);
    npx::sum_axis0(&sq)
}

/// `setdrop_tr`: the vertex the trust-region point replaces (`None`: keep the simplex).
fn setdrop_tr(
    ximproved: bool,
    d: &[f64],
    delta: f64,
    rho: f64,
    sim: &Mat,
    simi: &Mat,
) -> Option<usize> {
    let n = sim.rows;
    let mut distsq = vec![0.0_f64; n + 1];
    if ximproved {
        let mut diff = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = sim.at(i, j) - d[i];
                diff.set(i, j, v * v);
            }
        }
        distsq[..n].copy_from_slice(&npx::sum_axis0(&diff));
        let dd: Vec<f64> = d.iter().map(|v| v * v).collect();
        distsq[n] = npx::sum(&dd);
    } else {
        distsq[..n].copy_from_slice(&colsq_sums(sim, n));
        distsq[n] = 0.0;
    }
    let scale = np_maximum(rho, delta / 10.0);
    let denom = scale * scale;
    let weight: Vec<f64> = distsq.iter().map(|&v| np_maximum(1.0, v / denom)).collect();
    let simid = npx::mat_vec(simi, d, true);
    let mut vlag = simid.clone();
    vlag.push(1.0 - npx::sum(&simid));
    let mut score: Vec<f64> = weight
        .iter()
        .zip(&vlag)
        .map(|(&w, &v)| w * v.abs())
        .collect();
    if !ximproved {
        score[n] = -1.0;
    }
    for s in &mut score {
        if s.is_nan() {
            *s = -1.0;
        }
    }
    let mut jdrop = None;
    if score.iter().any(|&s| s > 0.0) {
        jdrop = Some(np_argmax(&score));
    }
    if ximproved && jdrop.is_none() {
        jdrop = Some(np_argmax(&distsq));
    }
    jdrop
}

/// The linear models' gradients: `g` for the objective, the columns of `A` for the constraints.
fn linear_models(
    amat: Option<&Mat>,
    m_lcon: usize,
    conmat: &Mat,
    fval: &[f64],
    simi: &Mat,
) -> (Vec<f64>, Mat) {
    let n = simi.rows;
    let m = conmat.rows;
    let fdiff: Vec<f64> = (0..n).map(|j| fval[j] - fval[n]).collect();
    let g = npx::vec_mat(&fdiff, true, simi, n);
    let mut a = Mat::zeros(n, m);
    if let Some(amat) = amat {
        for i in 0..n {
            for j in 0..m_lcon {
                a.set(i, j, amat.at(j, i));
            }
        }
    }
    let m_nl = m - m_lcon;
    let mut cdiff = Mat::zeros(m_nl, n);
    for r in 0..m_nl {
        for c in 0..n {
            cdiff.set(r, c, conmat.at(m_lcon + r, c) - conmat.at(m_lcon + r, n));
        }
    }
    let prod = npx::mat_mat(&cdiff, simi, n);
    for r in 0..m_nl {
        for c in 0..n {
            a.set(c, m_lcon + r, prod.at(r, c));
        }
    }
    (g, a)
}

/// `geostep`: the geometry step, of length `delbar`, perpendicular to the face opposite vertex
/// `jdrop`, signed by the merit function's linear model.
fn geostep(
    jdrop: usize,
    amat: Option<&Mat>,
    m_lcon: usize,
    conmat: &Mat,
    cpen: f64,
    delbar: f64,
    fval: &[f64],
    simi: &Mat,
) -> Vec<f64> {
    let n = simi.rows;
    let row = simi.row(jdrop).to_vec();
    let nrm = npx::norm(&row);
    let mut d: Vec<f64> = row.iter().map(|&v| delbar * (v / nrm)).collect();
    let (g, a) = linear_models(amat, m_lcon, conmat, fval, simi);
    let ad = npx::vec_mat(&d, true, &a, a.cols);
    let cp: Vec<f64> = (0..a.cols).map(|i| conmat.at(i, n) + ad[i]).collect();
    let cn: Vec<f64> = (0..a.cols).map(|i| conmat.at(i, n) - ad[i]).collect();
    let cvpd = np_max0(&cp);
    let cvnd = np_max0(&cn);
    let dg = npx::dot(&d, &g, true);
    if -dg + cpen * cvnd < dg + cpen * cvpd {
        for v in &mut d {
            *v = -*v;
        }
    }
    d
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/update.py
// ══════════════════════════════════════════════════════════════════════

struct Simplex {
    sim: Mat,
    simi: Mat,
    fval: Vec<f64>,
    cval: Vec<f64>,
    conmat: Mat,
}

/// `np.max(abs(simi @ sim[:, :n] - eye(n)))`.
fn inverse_error(simi: &Mat, sim: &Mat) -> f64 {
    let n = sim.rows;
    let prod = npx::mat_mat(simi, sim, n);
    let mut errs = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let e = if i == j { 1.0 } else { 0.0 };
            errs.push((prod.at(i, j) - e).abs());
        }
    }
    np_max(&errs)
}

/// The `erri` test of `updatexfc` / `updatepole`: recompute `simi` from scratch when it has
/// drifted; returns the final error.
fn refresh_inverse(sp: &mut Simplex) -> f64 {
    let n = sp.sim.rows;
    let mut erri = inverse_error(&sp.simi, &sp.sim);
    if erri > 0.1 || erri.is_nan() {
        let simi_test = npx::inv(&sp.sim.first_cols(n));
        let erri_test = inverse_error(&simi_test, &sp.sim);
        if erri_test < erri || (erri.is_nan() && !erri_test.is_nan()) {
            sp.simi = simi_test;
            erri = erri_test;
        }
    }
    erri
}

/// `findpole`: the best vertex for the merit function `fval + cpen * cval`.
fn findpole(cpen: f64, cval: &[f64], fval: &[f64]) -> usize {
    let n = fval.len() - 1;
    let phi: Vec<f64> = fval.iter().zip(cval).map(|(&f, &c)| f + cpen * c).collect();
    let phimin = py_min(&phi);
    let mut jopt = n;
    if phimin < phi[jopt] || (0..=n).any(|i| cval[i] < cval[n] && phi[i] <= phi[n]) {
        jopt = masked_argmin(cval, |i| phi[i] > phimin);
    }
    jopt
}

/// `updatepole`: move the best vertex to the pole position `sim[:, n]`.
fn updatepole(cpen: f64, sp: &mut Simplex) -> Option<CobylaInfo> {
    let n = sp.sim.rows;
    let jopt = findpole(cpen, &sp.cval, &sp.fval);
    let sim_old = sp.sim.clone();
    let simi_old = sp.simi.clone();
    if jopt < n {
        for i in 0..n {
            sp.sim.set(i, n, sp.sim.at(i, n) + sp.sim.at(i, jopt));
        }
        let sim_jopt = sp.sim.col(jopt);
        for i in 0..n {
            sp.sim.set(i, jopt, 0.0);
        }
        for i in 0..n {
            for j in 0..n {
                sp.sim.set(i, j, sp.sim.at(i, j) - sim_jopt[i]);
            }
        }
        let colsum = npx::sum_axis0(&sp.simi);
        for (j, &s) in colsum.iter().enumerate() {
            sp.simi.set(jopt, j, -s);
        }
    }
    let erri = refresh_inverse(sp);
    if erri <= 1.0 {
        if jopt < n {
            sp.fval.swap(jopt, n);
            sp.conmat.swap_cols(jopt, n);
            sp.cval.swap(jopt, n);
        }
        None
    } else {
        sp.sim = sim_old;
        sp.simi = simi_old;
        Some(CobylaInfo::DamagingRounding)
    }
}

/// `updatexfc`: replace vertex `jdrop` by `sim[:, n] + d` with values `f`, `constr`, `cstrv`.
fn updatexfc(
    jdrop: Option<usize>,
    constr: &[f64],
    cpen: f64,
    cstrv: f64,
    d: &[f64],
    f: f64,
    sp: &mut Simplex,
) -> Option<CobylaInfo> {
    let jdrop = jdrop?;
    let n = sp.sim.rows;
    if jdrop < n {
        sp.sim.set_col(jdrop, d);
        let row = sp.simi.row(jdrop).to_vec();
        let denom = npx::dot(&row, d, true);
        let simi_jdrop: Vec<f64> = row.iter().map(|&v| v / denom).collect();
        let sd = npx::mat_vec(&sp.simi, d, true);
        for i in 0..n {
            for j in 0..n {
                sp.simi.set(i, j, sp.simi.at(i, j) - sd[i] * simi_jdrop[j]);
            }
        }
        sp.simi.row_mut(jdrop).copy_from_slice(&simi_jdrop);
    } else {
        for i in 0..n {
            sp.sim.set(i, n, sp.sim.at(i, n) + d[i]);
        }
        for i in 0..n {
            for j in 0..n {
                sp.sim.set(i, j, sp.sim.at(i, j) - d[i]);
            }
        }
        let simid = npx::mat_vec(&sp.simi, d, true);
        let sum_simi = npx::sum_axis0(&sp.simi);
        // Python's builtin `sum(simid)`: left to right from 0.
        let mut s = 0.0_f64;
        for &v in &simid {
            s += v;
        }
        let w: Vec<f64> = sum_simi.iter().map(|&v| v / (1.0 - s)).collect();
        for i in 0..n {
            for j in 0..n {
                sp.simi.set(i, j, sp.simi.at(i, j) + simid[i] * w[j]);
            }
        }
    }
    // PRIMA's "restore" on damaging rounding aliases the updated arrays; the caller stops either
    // way, so only the in-place updates above are kept.
    let simi_inplace = sp.simi.clone();
    let erri = refresh_inverse(sp);
    if erri <= 1.0 {
        sp.fval[jdrop] = f;
        sp.conmat.set_col(jdrop, constr);
        sp.cval[jdrop] = cstrv;
        updatepole(cpen, sp)
    } else {
        sp.simi = simi_inplace;
        Some(CobylaInfo::DamagingRounding)
    }
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/initialize.py
// ══════════════════════════════════════════════════════════════════════

/// `initxfc`: evaluate the initial simplex `x0 + rhobeg * e_j`, keeping the best vertex at the
/// pole. Returns the simplex, which vertices were evaluated, `nf` and an early exit flag.
fn initxfc(
    calcfc: &mut Calcfc<'_>,
    maxfun: usize,
    constr0: &[f64],
    amat: Option<&Mat>,
    bvec: &[f64],
    ctol: f64,
    f0: f64,
    ftarget: f64,
    rhobeg: f64,
    x0: &[f64],
) -> Result<(Simplex, Vec<bool>, usize, Option<CobylaInfo>), OptError> {
    let m = constr0.len();
    let m_lcon = bvec.len();
    let m_nlcon = m - m_lcon;
    let n = x0.len();
    let mut sim = Mat::zeros(n, n + 1);
    for i in 0..n {
        sim.set(i, i, rhobeg);
        sim.set(i, n, x0[i]);
    }
    let mut simi = Mat::zeros(n, n);
    for i in 0..n {
        simi.set(i, i, 1.0 / rhobeg);
    }
    let mut evaluated = vec![false; n + 1];
    let mut fval = vec![REALMAX; n + 1];
    let mut cval = vec![REALMAX; n + 1];
    let mut conmat = Mat::filled(m, n + 1, REALMAX);
    let mut info = None;
    for k in 0..=n {
        let mut x = sim.col(n);
        let j;
        let f;
        let constr;
        if k == 0 {
            j = n;
            f = f0;
            constr = constr0.to_vec();
        } else {
            j = k - 1;
            x[j] += rhobeg;
            (f, constr) = evaluate(calcfc, &x, m_nlcon, amat, bvec)?;
        }
        let cstrv = np_max0(&constr);
        evaluated[j] = true;
        fval[j] = f;
        conmat.set_col(j, &constr);
        cval[j] = cstrv;
        if let Some(sub) = checkbreak_con(maxfun, k, cstrv, ctol, f, ftarget, &x) {
            info = Some(sub);
            break;
        }
        if j < n && fval[j] < fval[n] {
            fval.swap(j, n);
            cval.swap(j, n);
            conmat.swap_cols(j, n);
            sim.set_col(n, &x);
            for c in 0..=j {
                sim.set(j, c, -rhobeg);
            }
        }
    }
    let nf = evaluated.iter().filter(|&&e| e).count();
    if evaluated.iter().all(|&e| e) {
        simi = npx::inv(&sim.first_cols(n));
    }
    Ok((
        Simplex {
            sim,
            simi,
            fval,
            cval,
            conmat,
        },
        evaluated,
        nf,
        info,
    ))
}

/// `initfilt`: seed the filter with the evaluated vertices.
fn initfilt(sp: &Simplex, ctol: f64, cweight: f64, evaluated: &[bool], filt: &mut Filter) {
    let n = sp.sim.rows;
    for i in 0..=n {
        if evaluated[i] {
            let x: Vec<f64> = if i < n {
                (0..n).map(|r| sp.sim.at(r, i) + sp.sim.at(r, n)).collect()
            } else {
                sp.sim.col(n)
            };
            savefilt(
                filt,
                sp.cval[i],
                ctol,
                cweight,
                sp.fval[i],
                &x,
                &sp.conmat.col(i),
            );
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/cobylb.py
// ══════════════════════════════════════════════════════════════════════

/// `fcratio`: the ratio between the typical changes of F and of the constraints.
fn fcratio(conmat: &Mat, fval: &[f64]) -> f64 {
    let m = conmat.rows;
    let neg_rows: Vec<Vec<f64>> = (0..m)
        .map(|i| conmat.row(i).iter().map(|v| -v).collect())
        .collect();
    let cmin: Vec<f64> = neg_rows.iter().map(|r| np_min(r)).collect();
    let cmax: Vec<f64> = neg_rows.iter().map(|r| np_max(r)).collect();
    let fmin = py_min(fval);
    let fmax = py_max(fval);
    if (0..m).any(|i| cmin[i] < 0.5 * cmax[i]) && fmin < fmax {
        let mut vals = vec![f64::INFINITY];
        for i in 0..m {
            if cmin[i] < 0.5 * cmax[i] {
                vals.push(np_maximum(cmax[i], 0.0) - cmin[i]);
            }
        }
        let denom = np_min(&vals);
        (fmax - fmin) / denom
    } else {
        0.0
    }
}

/// The trust-region step and its predicted reductions for the current simplex.
fn trust_step(
    amat: Option<&Mat>,
    m_lcon: usize,
    sp: &Simplex,
    delta: f64,
) -> (Vec<f64>, Vec<f64>, Mat) {
    let n = sp.sim.rows;
    let (g, a) = linear_models(amat, m_lcon, &sp.conmat, &sp.fval, &sp.simi);
    let b: Vec<f64> = (0..sp.conmat.rows).map(|i| -sp.conmat.at(i, n)).collect();
    let d = trstlp(&a, &b, delta, &g);
    (d, g, a)
}

/// `preref = -inprod(d, g)` and `prerec = cval[n] - max(0, conmat[:, n] + d @ A)`.
fn predicted(d: &[f64], g: &[f64], a: &Mat, sp: &Simplex) -> (f64, f64) {
    let n = sp.sim.rows;
    let preref = -npx::dot(d, g, true);
    let ad = npx::vec_mat(d, true, a, a.cols);
    let lin: Vec<f64> = (0..a.cols).map(|i| sp.conmat.at(i, n) + ad[i]).collect();
    let prerec = sp.cval[n] - np_max0(&lin);
    (preref, prerec)
}

/// `getcpen`: raise the penalty until the predicted merit reduction is positive.
fn getcpen(amat: Option<&Mat>, m_lcon: usize, sp_in: &Simplex, cpen_in: f64, delta: f64) -> f64 {
    let mut sp = Simplex {
        sim: sp_in.sim.clone(),
        simi: sp_in.simi.clone(),
        fval: sp_in.fval.clone(),
        cval: sp_in.cval.clone(),
        conmat: sp_in.conmat.clone(),
    };
    let n = sp.sim.rows;
    let mut cpen = cpen_in;
    for _ in 0..=n {
        if updatepole(cpen, &mut sp) == Some(CobylaInfo::DamagingRounding) {
            break;
        }
        let (d, g, a) = trust_step(amat, m_lcon, &sp, delta);
        let (preref, prerec) = predicted(&d, &g, &a, &sp);
        if !(prerec > 0.0 && preref < 0.0) {
            break;
        }
        cpen = py_max2(cpen, py_min2(-2.0 * preref / prerec, REALMAX));
        if findpole(cpen, &sp.cval, &sp.fval) == n {
            break;
        }
    }
    cpen
}

/// What `cobylb` returns.
#[derive(Debug, Clone)]
struct CoreOutput {
    x: Vec<f64>,
    f: f64,
    cstrv: f64,
    nf: usize,
    nit: usize,
    info: CobylaInfo,
}

/// The tuning constants `cobylb` reads.
struct Params {
    maxfilt: usize,
    maxfun: usize,
    ctol: f64,
    cweight: f64,
    eta1: f64,
    eta2: f64,
    ftarget: f64,
    gamma1: f64,
    gamma2: f64,
    rhobeg: f64,
    rhoend: f64,
}

/// Evaluate `x` unless it is within `1e-4 * rhoend` of a vertex (then reuse that vertex).
fn evaluate_trial(
    calcfc: &mut Calcfc<'_>,
    x: &[f64],
    sp: &Simplex,
    p: &Params,
    m_nlcon: usize,
    amat: Option<&Mat>,
    bvec: &[f64],
    nf: &mut usize,
    filt: &mut Filter,
) -> Result<(f64, Vec<f64>, f64), OptError> {
    let n = sp.sim.rows;
    let mut distsq = vec![0.0_f64; n + 1];
    let dx: Vec<f64> = (0..n)
        .map(|i| {
            let v = x[i] - sp.sim.at(i, n);
            v * v
        })
        .collect();
    distsq[n] = npx::sum(&dx);
    let mut diff = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = x[i] - (sp.sim.at(i, n) + sp.sim.at(i, j));
            diff.set(i, j, v * v);
        }
    }
    distsq[..n].copy_from_slice(&npx::sum_axis0(&diff));
    let j = np_argmin(&distsq);
    let lim = 1.0e-4 * p.rhoend;
    if distsq[j] <= lim * lim {
        return Ok((sp.fval[j], sp.conmat.col(j), sp.cval[j]));
    }
    let (f, constr) = evaluate(calcfc, x, m_nlcon, amat, bvec)?;
    let cstrv = np_max0(&constr);
    *nf += 1;
    savefilt(filt, cstrv, p.ctol, p.cweight, f, x, &constr);
    Ok((f, constr, cstrv))
}

/// `cobylb`: the COBYLA iterations.
fn cobylb(
    calcfc: &mut Calcfc<'_>,
    p: &Params,
    amat: Option<&Mat>,
    bvec: &[f64],
    constr0: &[f64],
    f0: f64,
    x0: &[f64],
    callback: Option<MinimizeCallback>,
) -> Result<CoreOutput, OptError> {
    let m_lcon = bvec.len();
    let m_nlcon = constr0.len() - m_lcon;
    let n = x0.len();
    let cpenmin = EPS;

    let (mut sp, evaluated, mut nf, subinfo) = initxfc(
        calcfc, p.maxfun, constr0, amat, bvec, p.ctol, f0, p.ftarget, p.rhobeg, x0,
    )?;
    let mut filt = Filter {
        x: Vec::new(),
        f: Vec::new(),
        c: Vec::new(),
        con: Vec::new(),
        maxfilt: p.maxfilt.max(1).min(p.maxfun),
    };
    initfilt(&sp, p.ctol, p.cweight, &evaluated, &mut filt);

    if let Some(info) = subinfo {
        let kopt = selectx(&filt.f, &filt.c, p.cweight, p.ctol);
        return Ok(CoreOutput {
            x: filt.x[kopt].clone(),
            f: filt.f[kopt],
            cstrv: filt.c[kopt],
            nf,
            nit: 0,
            info,
        });
    }

    let mut rho = p.rhobeg;
    let mut delta = p.rhobeg;
    let mut cpen = np_maximum(cpenmin, np_minimum(1.0e3, fcratio(&sp.conmat, &sp.fval)));
    let mut shortd = false;
    let mut ratio = -1.0_f64;
    let mut jdrop_tr: Option<usize> = Some(0);
    let gamma3 = np_maximum(1.0, np_minimum(0.75 * p.gamma2, 1.5));
    let maxtr = 10 * p.maxfun;
    let mut info = CobylaInfo::MaxtrReached;
    let mut d = vec![0.0_f64; n];
    let mut nit = 0;

    for tr in 0..maxtr {
        nit = tr + 1;
        cpen = getcpen(amat, m_lcon, &sp, cpen, delta);

        if updatepole(cpen, &mut sp) == Some(CobylaInfo::DamagingRounding) {
            info = CobylaInfo::DamagingRounding;
            break;
        }

        let lim = 4.0 * (delta * delta);
        let adequate_geo = colsq_sums(&sp.sim, n).iter().all(|&v| v <= lim);

        let (d_tr, g, a) = trust_step(amat, m_lcon, &sp, delta);
        d = d_tr;
        let dnorm = py_min2(delta, npx::norm(&d));
        shortd = dnorm <= 0.1 * rho;
        let (preref, prerec) = predicted(&d, &g, &a, &sp);
        let prerem = preref + cpen * prerec;
        let trfail = !(prerem > 1.0e-6 * py_min2(cpen, 1.0) * rho);

        if shortd || trfail {
            delta *= 0.1;
            if delta <= gamma3 * rho {
                delta = rho;
            }
        } else {
            let x: Vec<f64> = (0..n).map(|i| sp.sim.at(i, n) + d[i]).collect();
            let (f, constr, cstrv) =
                evaluate_trial(calcfc, &x, &sp, p, m_nlcon, amat, bvec, &mut nf, &mut filt)?;
            let actrem = (sp.fval[n] + cpen * sp.cval[n]) - (f + cpen * cstrv);
            ratio = redrat(actrem, prerem, p.eta1);
            delta = trrad(delta, dnorm, p.eta1, p.eta2, p.gamma1, p.gamma2, ratio);
            if delta <= gamma3 * rho {
                delta = rho;
            }
            let ximproved = actrem > 0.0;
            jdrop_tr = setdrop_tr(ximproved, &d, delta, rho, &sp.sim, &sp.simi);
            if updatexfc(jdrop_tr, &constr, cpen, cstrv, &d, f, &mut sp)
                == Some(CobylaInfo::DamagingRounding)
            {
                info = CobylaInfo::DamagingRounding;
                break;
            }
            if let Some(sub) = checkbreak_con(p.maxfun, nf, cstrv, p.ctol, f, p.ftarget, &x) {
                info = sub;
                break;
            }
        }

        let bad_trstep = shortd || trfail || ratio <= 0.0 || jdrop_tr.is_none();
        let improve_geo = bad_trstep && !adequate_geo;
        let reduce_rho = bad_trstep && adequate_geo && py_max2(delta, dnorm) <= rho;

        let lim = 4.0 * (delta * delta);
        if improve_geo && !colsq_sums(&sp.sim, n).iter().all(|&v| v <= lim) {
            let jdrop_geo = np_argmax(&colsq_sums(&sp.sim, n));
            let delbar = delta / 2.0;
            d = geostep(
                jdrop_geo, amat, m_lcon, &sp.conmat, cpen, delbar, &sp.fval, &sp.simi,
            );
            let x: Vec<f64> = (0..n).map(|i| sp.sim.at(i, n) + d[i]).collect();
            let (f, constr, cstrv) =
                evaluate_trial(calcfc, &x, &sp, p, m_nlcon, amat, bvec, &mut nf, &mut filt)?;
            if updatexfc(Some(jdrop_geo), &constr, cpen, cstrv, &d, f, &mut sp)
                == Some(CobylaInfo::DamagingRounding)
            {
                info = CobylaInfo::DamagingRounding;
                break;
            }
            if let Some(sub) = checkbreak_con(p.maxfun, nf, cstrv, p.ctol, f, p.ftarget, &x) {
                info = sub;
                break;
            }
        }

        if reduce_rho {
            if rho <= p.rhoend {
                info = CobylaInfo::SmallTrRadius;
                break;
            }
            delta = py_max2(0.5 * rho, redrho(rho, p.rhoend));
            rho = redrho(rho, p.rhoend);
            cpen = np_maximum(cpenmin, np_minimum(cpen, fcratio(&sp.conmat, &sp.fval)));
            if updatepole(cpen, &mut sp) == Some(CobylaInfo::DamagingRounding) {
                info = CobylaInfo::DamagingRounding;
                break;
            }
        }

        if let Some(cb) = callback
            && cb(&sp.sim.col(n))
        {
            info = CobylaInfo::CallbackTerminate;
            break;
        }
    }

    let x: Vec<f64> = (0..n).map(|i| sp.sim.at(i, n) + d[i]).collect();
    let step: Vec<f64> = (0..n).map(|i| x[i] - sp.sim.at(i, n)).collect();
    if info == CobylaInfo::SmallTrRadius
        && shortd
        && npx::norm(&step) > 1.0e-3 * p.rhoend
        && nf < p.maxfun
    {
        let (f, constr) = evaluate(calcfc, &x, m_nlcon, amat, bvec)?;
        let cstrv = np_max0(&constr);
        nf += 1;
        savefilt(&mut filt, cstrv, p.ctol, p.cweight, f, &x, &constr);
    }

    let kopt = selectx(&filt.f, &filt.c, py_max2(cpen, p.cweight), p.ctol);
    Ok(CoreOutput {
        x: filt.x[kopt].clone(),
        f: filt.f[kopt],
        cstrv: filt.c[kopt],
        nf,
        nit,
        info,
    })
}

// ══════════════════════════════════════════════════════════════════════
// cobyla/cobyla.py and common/preproc.py
// ══════════════════════════════════════════════════════════════════════

/// The problem as `pyprima.cobyla.cobyla` receives it.
struct CoreProblem {
    x0: Vec<f64>,
    aineq: Option<Mat>,
    bineq: Vec<f64>,
    aeq: Option<Mat>,
    beq: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    f0: f64,
    nlconstr0: Vec<f64>,
}

/// The user-facing settings, before `preproc`.
struct CoreSettings {
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    ctol: f64,
    ftarget: f64,
    callback: Option<MinimizeCallback>,
}

/// `get_lincon`: bounds and linear constraints as `amat @ x <= bvec`:
/// `[-I[ixl]; I[ixu]; -Aeq; Aeq; Aineq]`.
fn get_lincon(prob: &CoreProblem) -> (Option<Mat>, Vec<f64>) {
    let n = prob.x0.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut bvec = Vec::new();
    for i in 0..n {
        if prob.xl[i] > -BOUNDMAX {
            rows.push((0..n).map(|j| if i == j { -1.0 } else { -0.0 }).collect());
            bvec.push(-prob.xl[i]);
        }
    }
    for i in 0..n {
        if prob.xu[i] < BOUNDMAX {
            rows.push((0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect());
            bvec.push(prob.xu[i]);
        }
    }
    if let Some(aeq) = &prob.aeq {
        for r in 0..aeq.rows {
            rows.push(aeq.row(r).iter().map(|v| -v).collect());
            bvec.push(-prob.beq[r]);
        }
        for r in 0..aeq.rows {
            rows.push(aeq.row(r).to_vec());
            bvec.push(prob.beq[r]);
        }
    }
    if let Some(aineq) = &prob.aineq {
        for r in 0..aineq.rows {
            rows.push(aineq.row(r).to_vec());
            bvec.push(prob.bineq[r]);
        }
    }
    if rows.is_empty() {
        return (None, Vec::new());
    }
    let mut amat = Mat::zeros(rows.len(), n);
    for (r, row) in rows.iter().enumerate() {
        amat.row_mut(r).copy_from_slice(row);
    }
    (Some(amat), bvec)
}

/// Report one of `preproc`'s silent revisions: SciPy warns and proceeds (Strict records it in
/// the optimize trace); Hardened refuses the input.
fn revise(mode: RuntimeMode, detail: String) -> Result<(), OptError> {
    if mode == RuntimeMode::Hardened {
        return Err(OptError::InvalidArgument { detail });
    }
    crate::minimize::push_trace(OptimizeTraceEntry {
        ts_unix_ms: crate::minimize::now_unix_ms(),
        event: String::from("cobyla_input_revised"),
        method: OptimizeMethod::Cobyla,
        iter_num: 0,
        f_val: None,
        grad_norm: None,
        step_size: None,
        mode,
        reason: Some(detail),
        final_x: None,
        final_f: None,
        total_nfev: 0,
        fixture_id: None,
        seed: None,
    });
    Ok(())
}

/// `pyprima.cobyla.cobyla(calcfc, m_nlcon, x, Aineq, bineq, Aeq, beq, xl, xu, f0, nlconstr0,
/// rhobeg, rhoend, ftarget, ctol, maxfun)` with PRIMA's other defaults.
fn cobyla_core(
    calcfc: &mut Calcfc<'_>,
    mut prob: CoreProblem,
    s: &CoreSettings,
    mode: RuntimeMode,
) -> Result<CoreOutput, OptError> {
    let n = prob.x0.len();
    let m_nlcon = prob.nlconstr0.len();
    for v in &mut prob.xl {
        if v.is_nan() || *v < -BOUNDMAX {
            *v = -BOUNDMAX;
        }
    }
    for v in &mut prob.xu {
        if v.is_nan() || *v > BOUNDMAX {
            *v = BOUNDMAX;
        }
    }
    let (amat, bvec) = get_lincon(&prob);
    let m_lcon = bvec.len();
    let mmm = m_lcon + m_nlcon;

    let mut constr = vec![0.0_f64; mmm];
    let mut x = prob.x0.clone();
    let f;
    if prob.x0.iter().all(|v| v.is_finite()) {
        f = moderatef(prob.f0);
        if let Some(amat) = &amat {
            let v = npx::mat_vec(amat, &x, true);
            for i in 0..m_lcon {
                constr[i] = moderatec(v[i] - bvec[i]);
            }
        }
        for (c, &v) in constr.iter_mut().skip(m_lcon).zip(&prob.nlconstr0) {
            *c = moderatec(v);
        }
    } else {
        x = moderatex(&x);
        let (fe, ce) = evaluate(calcfc, &x, m_nlcon, amat.as_ref(), &bvec)?;
        f = fe;
        constr = ce;
        for c in constr.iter_mut().take(m_lcon) {
            *c = moderatec(*c);
        }
    }

    let eta1 = ETA1_DEFAULT;
    let eta2 = (eta1 + 2.0) / 3.0;

    // preproc
    let mut maxfun = s.maxfun;
    if maxfun < n + 2 {
        maxfun = n + 2;
        revise(
            mode,
            format!(
                "COBYLA: Invalid MAXFUN; it should be at least num_vars + 2; it is set to {maxfun}"
            ),
        )?;
    }
    let mut ftarget = s.ftarget;
    if ftarget.is_nan() {
        ftarget = -REALMAX;
        revise(
            mode,
            String::from(
                "COBYLA: Invalid FTARGET; it should be a real number; it is set to -REALMAX",
            ),
        )?;
    }
    let mut maxfilt = MAXFILT_DEFAULT.max(MIN_MAXFILT);
    let unit_memo = (mmm + n + 2) * 8;
    if maxfilt as f64 > MAXHISTMEM as f64 / unit_memo as f64 {
        maxfilt = (MAXHISTMEM as f64 / unit_memo as f64) as usize;
    }
    maxfilt = maxfun.min(MIN_MAXFILT.max(maxfilt));
    let mut rhobeg = s.rhobeg;
    let mut rhoend = s.rhoend;
    if (rhobeg - rhoend).abs() < 1.0e2 * EPS * py_max2(rhobeg.abs(), 1.0) {
        rhoend = rhobeg;
    }
    if rhobeg <= 0.0 || rhobeg.is_nan() || rhobeg.is_infinite() {
        rhobeg = if rhoend.is_finite() && rhoend > 0.0 {
            py_max2(10.0 * rhoend, RHOBEG_DEFAULT)
        } else {
            RHOBEG_DEFAULT
        };
        revise(
            mode,
            format!(
                "COBYLA: Invalid RHOBEG; it should be a positive number; it is set to {rhobeg}"
            ),
        )?;
    }
    if rhoend <= 0.0 || rhobeg < rhoend || rhoend.is_nan() || rhoend.is_infinite() {
        rhoend = py_max2(EPS, py_min2(0.1 * rhobeg, RHOEND_DEFAULT));
        revise(
            mode,
            format!(
                "COBYLA: Invalid RHOEND; it should be a positive number and RHOEND <= RHOBEG; it is set to {rhoend}"
            ),
        )?;
    }
    let mut ctol = s.ctol;
    if ctol.is_nan() || ctol < 0.0 {
        ctol = EPS.sqrt();
        if mmm > 0 {
            revise(
                mode,
                format!(
                    "COBYLA: Invalid CTOL; it should be a nonnegative number; it is set to {ctol}"
                ),
            )?;
        }
    }

    let params = Params {
        maxfilt,
        maxfun,
        ctol,
        cweight: CWEIGHT_DEFAULT,
        eta1,
        eta2,
        ftarget,
        gamma1: GAMMA1_DEFAULT,
        gamma2: GAMMA2_DEFAULT,
        rhobeg,
        rhoend,
    };
    cobylb(
        calcfc,
        &params,
        amat.as_ref(),
        &bvec,
        &constr,
        f,
        &x,
        s.callback,
    )
}

// ══════════════════════════════════════════════════════════════════════
// SciPy front end: _minimize_cobyla, pyprima.minimize, fmin_cobyla
// ══════════════════════════════════════════════════════════════════════

/// One source of nonlinear constraints, in `pyprima.process_nl_constraints` order.
enum NlPiece<'a> {
    /// A constraint dict: `NonlinearConstraint(fun, 0, 0)` for `eq`, `(fun, 0, inf)` for `ineq`.
    Dict(&'a Constraint<'a>),
    /// A `NonlinearConstraint(fun, lb, ub)`.
    Object(&'a NonlinearConstraint),
    /// An `fmin_cobyla` callable `c(x) >= 0`.
    Scalar(&'a dyn Fn(&[f64]) -> f64),
}

/// The nonlinear constraint values in PRIMA's sign (`lb - c(x)`, `c(x) - ub`; feasible `<= 0`).
fn eval_nonlinear(pieces: &[NlPiece<'_>], x: &[f64]) -> Result<Vec<f64>, OptError> {
    let mut out = Vec::new();
    for piece in pieces {
        match piece {
            NlPiece::Dict(c) => {
                let v = (c.fun)(x);
                out.extend(v.iter().map(|&vi| 0.0 - vi));
                if c.kind == ConstraintType::Eq {
                    out.extend(v.iter().map(|&vi| vi - 0.0));
                }
            }
            NlPiece::Object(nlc) => {
                let v = (nlc.fun)(x);
                for (bound, len) in [("lower", nlc.lb.len()), ("upper", nlc.ub.len())] {
                    if v.len() != len {
                        return Err(OptError::InvalidArgument {
                            detail: format!(
                                "The number of elements in the constraint function's output does not match the number of elements in the {bound} bound."
                            ),
                        });
                    }
                }
                for (&lb, &vi) in nlc.lb.iter().zip(&v) {
                    if lb > f64::NEG_INFINITY {
                        out.push(lb - vi);
                    }
                }
                for (&ub, &vi) in nlc.ub.iter().zip(&v) {
                    if ub < f64::INFINITY {
                        out.push(vi - ub);
                    }
                }
            }
            NlPiece::Scalar(c) => out.push(0.0 - c(x)),
        }
    }
    Ok(out)
}

/// The linear constraints of a problem, combined in order (`combine_multiple_linear_constraints`).
struct Linear {
    a: Mat,
    lb: Vec<f64>,
    ub: Vec<f64>,
}

fn combine_linear(cons: &[&LinearConstraint], n: usize) -> Result<Option<Linear>, OptError> {
    if cons.is_empty() {
        return Ok(None);
    }
    let rows: usize = cons.iter().map(|c| c.a.len()).sum();
    let mut a = Mat::zeros(rows, n);
    let mut lb = Vec::with_capacity(rows);
    let mut ub = Vec::with_capacity(rows);
    let mut r = 0;
    for c in cons {
        for (i, row) in c.a.iter().enumerate() {
            if row.len() != n {
                return Err(OptError::InvalidArgument {
                    detail: format!(
                        "LinearConstraint row {i} has {} columns; x0 has {n} entries",
                        row.len()
                    ),
                });
            }
            a.row_mut(r).copy_from_slice(row);
            r += 1;
        }
        lb.extend_from_slice(&c.lb);
        ub.extend_from_slice(&c.ub);
    }
    Ok(Some(Linear { a, lb, ub }))
}

/// `separate_LC_into_eq_and_ineq`: `Aeq x = beq` for rows with `ub <= lb + 2 eps`,
/// `Aineq x <= bineq` (`-A x <= -lb`, then `A x <= ub`) for the finite sides of the rest.
fn separate_linear(lin: &Linear) -> (Option<Mat>, Vec<f64>, Option<Mat>, Vec<f64>) {
    let m = lin.lb.len();
    let n = lin.a.cols;
    let is_eq: Vec<bool> = (0..m).map(|i| lin.ub[i] <= lin.lb[i] + 2.0 * EPS).collect();
    let mut eq_rows = Vec::new();
    let mut beq = Vec::new();
    for i in 0..m {
        if is_eq[i] {
            eq_rows.push(lin.a.row(i).to_vec());
            beq.push((lin.lb[i] + lin.ub[i]) / 2.0);
        }
    }
    let mut in_rows = Vec::new();
    let mut bineq = Vec::new();
    for i in 0..m {
        if !is_eq[i] && lin.lb[i] > f64::NEG_INFINITY {
            in_rows.push(lin.a.row(i).iter().map(|v| -v).collect::<Vec<f64>>());
            bineq.push(-lin.lb[i]);
        }
    }
    for i in 0..m {
        if !is_eq[i] && lin.ub[i] < f64::INFINITY {
            in_rows.push(lin.a.row(i).to_vec());
            bineq.push(lin.ub[i]);
        }
    }
    let to_mat = |rows: &[Vec<f64>]| -> Option<Mat> {
        if rows.is_empty() {
            return None;
        }
        let mut mat = Mat::zeros(rows.len(), n);
        for (r, row) in rows.iter().enumerate() {
            mat.row_mut(r).copy_from_slice(row);
        }
        Some(mat)
    };
    (to_mat(&eq_rows), beq, to_mat(&in_rows), bineq)
}

/// `np.nanmax((a, b), axis=0)` / `np.nanmin` for one entry (a later equal entry wins).
fn nanmax2(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a > b {
        a
    } else {
        b
    }
}

fn nanmin2(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a < b {
        a
    } else {
        b
    }
}

/// The minimum-norm least-squares solution of `A xi = r` for a matrix of full row rank, by
/// Householder QR of `A^T`: the fallback for the systems [`npx::lstsq`] does not reproduce
/// (more than 25 equality rows, or data LAPACK would rescale).
fn min_norm_solve(a: &Mat, r: &[f64]) -> Vec<f64> {
    let (m, n) = (a.rows, a.cols);
    // Columns of qt are the columns of A^T, i.e. the rows of A.
    let mut qt: Vec<Vec<f64>> = (0..m).map(|i| a.row(i).to_vec()).collect();
    let mut vs: Vec<Vec<f64>> = Vec::new();
    let mut rdiag = vec![0.0_f64; m];
    let mut rmat = vec![vec![0.0_f64; m]; m];
    let k_max = m.min(n);
    for k in 0..k_max {
        let col = &qt[k];
        let norm: f64 = col[k..].iter().map(|v| v * v).sum::<f64>().sqrt();
        let alpha = if col[k] >= 0.0 { -norm } else { norm };
        let mut v = vec![0.0_f64; n];
        v[k..].copy_from_slice(&col[k..]);
        v[k] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        for j in k..m {
            if vnorm2 > 0.0 {
                let dotv: f64 = (k..n).map(|i| v[i] * qt[j][i]).sum();
                let s = 2.0 * dotv / vnorm2;
                for i in k..n {
                    qt[j][i] -= s * v[i];
                }
            }
            rmat[k][j] = qt[j][k];
        }
        rdiag[k] = rmat[k][k];
        vs.push(v);
    }
    // Solve R^T y = r (R upper triangular m x m), dropping negligible pivots.
    let rmax = rdiag.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = rmax * EPS * (m.max(n) as f64);
    let mut y = vec![0.0_f64; n];
    for k in 0..k_max {
        let s: f64 = (0..k).map(|j| rmat[j][k] * y[j]).sum();
        y[k] = if rdiag[k].abs() > tol {
            (r[k] - s) / rdiag[k]
        } else {
            0.0
        };
    }
    // xi = Q y: apply the reflections in reverse.
    for (k, v) in vs.iter().enumerate().rev() {
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 > 0.0 {
            let dotv: f64 = (k..n).map(|i| v[i] * y[i]).sum();
            let s = 2.0 * dotv / vnorm2;
            for i in k..n {
                y[i] -= s * v[i];
            }
        }
    }
    y
}

/// `_project(x0, lb, ub, {"linear": linear, "nonlinear": None})`: move `x0` onto the bounds and
/// linear constraints (used only without nonlinear constraints).
fn project_x0(
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    linear: Option<&Linear>,
) -> Result<Vec<f64>, OptError> {
    let n = x0.len();
    let clip = |x: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| nanmin2(nanmax2(x[i], lb[i]), ub[i]))
            .collect()
    };
    let Some(lin) = linear else {
        return Ok(clip(x0));
    };
    let max_con = 1.0e20;
    let m = lin.lb.len();
    let all_eq = (0..m).all(|i| (lin.ub[i] - lin.lb[i]).abs() <= EPS);
    let lb_max = lb.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ub_min = ub.iter().copied().fold(f64::INFINITY, f64::min);
    if all_eq && lb_max <= -max_con && ub_min >= max_con {
        let b: Vec<f64> = (0..m).map(|i| (lin.lb[i] + lin.ub[i]) / 2.0).collect();
        let ax = npx::mat_vec(&lin.a, x0, true);
        let rhs: Vec<f64> = (0..m).map(|i| b[i] - ax[i]).collect();
        let xi = npx::lstsq(&lin.a, &rhs).unwrap_or_else(|| min_norm_solve(&lin.a, &rhs));
        let shifted: Vec<f64> = (0..n).map(|i| x0[i] + xi[i]).collect();
        return Ok(clip(&shifted));
    }
    let ax = npx::mat_vec(&lin.a, x0, true);
    let violated = (0..m).any(|i| {
        if lin.lb[i] == lin.ub[i] {
            ax[i] != lin.lb[i]
        } else {
            ax[i] > lin.ub[i] || lin.lb[i] > ax[i]
        }
    }) || (0..n).any(|i| x0[i] > ub[i] || lb[i] > x0[i]);
    if !violated {
        return Ok(x0.to_vec());
    }
    // SciPy runs SLSQP on min ||x - x0||^2 / 2 (gradient x - x0) under the bounds and the linear
    // constraints. In y = x - x0 the gradient is y itself, which a plain `fn` can supply.
    let ineq_rows: Vec<usize> = (0..m).filter(|&i| lin.lb[i] != lin.ub[i]).collect();
    let eq_rows: Vec<usize> = (0..m).filter(|&i| lin.lb[i] == lin.ub[i]).collect();
    let x0v = x0.to_vec();
    let rows_of =
        |idx: &[usize]| -> Vec<Vec<f64>> { idx.iter().map(|&i| lin.a.row(i).to_vec()).collect() };
    let ineq_a = rows_of(&ineq_rows);
    let eq_a = rows_of(&eq_rows);
    let mut constraints: Vec<Constraint<'_>> = Vec::new();
    let shifted_rows = |a: &[Vec<f64>], y: &[f64]| -> Vec<f64> {
        a.iter()
            .map(|row| {
                row.iter()
                    .zip(y.iter().zip(&x0v))
                    .map(|(&r, (&yi, &xi))| r * (yi + xi))
                    .sum()
            })
            .collect()
    };
    if !eq_rows.is_empty() {
        let lbs: Vec<f64> = eq_rows.iter().map(|&i| lin.lb[i]).collect();
        let a = eq_a.clone();
        let f = shifted_rows;
        constraints.push(
            Constraint::eq(move |y: &[f64]| {
                f(&a, y).iter().zip(&lbs).map(|(v, l)| v - l).collect()
            })
            .with_jac({
                let a = eq_a.clone();
                move |_: &[f64]| a.clone()
            }),
        );
    }
    if !ineq_rows.is_empty() {
        let lo: Vec<usize> = (0..ineq_rows.len())
            .filter(|&k| lin.lb[ineq_rows[k]] > f64::NEG_INFINITY)
            .collect();
        let hi: Vec<usize> = (0..ineq_rows.len())
            .filter(|&k| lin.ub[ineq_rows[k]] < f64::INFINITY)
            .collect();
        if !lo.is_empty() || !hi.is_empty() {
            let lbs: Vec<f64> = ineq_rows.iter().map(|&i| lin.lb[i]).collect();
            let ubs: Vec<f64> = ineq_rows.iter().map(|&i| lin.ub[i]).collect();
            let a = ineq_a.clone();
            let f = shifted_rows;
            let (lo_c, hi_c) = (lo.clone(), hi.clone());
            let jac_rows: Vec<Vec<f64>> = lo
                .iter()
                .map(|&k| ineq_a[k].clone())
                .chain(hi.iter().map(|&k| ineq_a[k].iter().map(|v| -v).collect()))
                .collect();
            constraints.push(
                Constraint::ineq(move |y: &[f64]| {
                    let v = f(&a, y);
                    lo_c.iter()
                        .map(|&k| v[k] - lbs[k])
                        .chain(hi_c.iter().map(|&k| ubs[k] - v[k]))
                        .collect()
                })
                .with_jac(move |_: &[f64]| jac_rows.clone()),
            );
        }
    }
    let bounds: Vec<Bound> = (0..n)
        .map(|i| {
            let lo = if lb[i].is_finite() {
                Some(lb[i] - x0[i])
            } else {
                None
            };
            let hi = if ub[i].is_finite() {
                Some(ub[i] - x0[i])
            } else {
                None
            };
            (lo, hi)
        })
        .collect();
    fn half_sq(y: &[f64]) -> f64 {
        npx::dot(y, y, true) / 2.0
    }
    fn identity_grad(y: &[f64]) -> Vec<f64> {
        y.to_vec()
    }
    let opts = MinimizeOptions {
        method: Some(OptimizeMethod::Slsqp),
        gradient: Some(identity_grad),
        bounds: Some(&bounds),
        constraints: &constraints,
        ..MinimizeOptions::default()
    };
    let res = crate::minimize::slsqp(&half_sq, &vec![0.0; n], opts)?;
    Ok((0..n).map(|i| x0[i] + res.x[i]).collect())
}

/// `get_arrays_tol(lb, ub)`.
fn arrays_tol(lb: &[f64], ub: &[f64]) -> f64 {
    let size = lb.len().max(ub.len()) as f64;
    let weight = |v: &[f64]| {
        v.iter()
            .filter(|x| x.is_finite())
            .fold(1.0_f64, |acc, x| acc.max(x.abs()))
    };
    10.0 * EPS * size.max(1.0) * weight(lb).max(weight(ub))
}

/// A COBYLA problem as SciPy's front end has it after option handling.
struct FrontEnd<'a> {
    lb: Vec<f64>,
    ub: Vec<f64>,
    linear: Vec<&'a LinearConstraint>,
    nonlinear: Vec<NlPiece<'a>>,
    settings: CoreSettings,
    mode: RuntimeMode,
}

/// `pyprima.minimize(fun, x0, method='cobyla', bounds, constraints, options)` followed by
/// `_minimize_cobyla`'s result handling.
fn run_front_end<F>(fun: &F, x0: &[f64], fe: &FrontEnd<'_>) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64 + ?Sized,
{
    let n = x0.len();
    let linear = combine_linear(&fe.linear, n)?;
    let has_constraints = linear.is_some() || !fe.nonlinear.is_empty();

    // Variables fixed by their bounds are removed (pyprima wraps only the objective for them).
    let tol = arrays_tol(&fe.lb, &fe.ub);
    let fixed: Vec<bool> = (0..n)
        .map(|i| fe.lb[i] <= fe.ub[i] && (fe.lb[i] - fe.ub[i]).abs() < tol)
        .collect();
    let any_fixed = fixed.iter().any(|&b| b);
    if any_fixed && fixed.iter().all(|&b| b) {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "COBYLA: every variable is fixed by its bounds (SciPy's COBYLA fails on this problem too)",
            ),
        });
    }
    if any_fixed && has_constraints {
        return Err(OptError::InvalidArgument {
            detail: String::from(
                "COBYLA: variables fixed by their bounds cannot be combined with constraints (SciPy 1.17.1 evaluates the constraints on the reduced x and fails)",
            ),
        });
    }
    let fixed_values: Vec<f64> = (0..n)
        .map(|i| (0.5 * (fe.lb[i] + fe.ub[i])).clamp(fe.lb[i], fe.ub[i]))
        .collect();
    let free: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
    let expand = |xr: &[f64]| -> Vec<f64> {
        if !any_fixed {
            return xr.to_vec();
        }
        let mut full = vec![0.0_f64; n];
        for i in 0..n {
            if fixed[i] {
                full[i] = fixed_values[i];
            }
        }
        for (k, &i) in free.iter().enumerate() {
            full[i] = xr[k];
        }
        full
    };
    let mut x0r: Vec<f64> = free.iter().map(|&i| x0[i]).collect();
    let lbr: Vec<f64> = free.iter().map(|&i| fe.lb[i]).collect();
    let ubr: Vec<f64> = free.iter().map(|&i| fe.ub[i]).collect();

    if fe.nonlinear.is_empty() {
        x0r = project_x0(&x0r, &lbr, &ubr, linear.as_ref())?;
    }
    let (aeq, beq, aineq, bineq) = match &linear {
        Some(lin) => separate_linear(lin),
        None => (None, Vec::new(), None, Vec::new()),
    };

    let mut calcfc = |x: &[f64]| -> Result<(f64, Vec<f64>), OptError> {
        let full = expand(x);
        let f = fun(&full);
        let c = eval_nonlinear(&fe.nonlinear, x)?;
        Ok((f, c))
    };
    let (f0, nlconstr0) = calcfc(&x0r)?;
    let prob = CoreProblem {
        x0: x0r,
        aineq,
        bineq,
        aeq,
        beq,
        xl: lbr,
        xu: ubr,
        f0,
        nlconstr0,
    };
    let out = cobyla_core(&mut calcfc, prob, &fe.settings, fe.mode)?;

    // `_minimize_cobyla` compares against its own `ctol` (the `catol` given), not the one
    // `preproc` may have revised.
    let (success, status, message) = if out.cstrv > fe.settings.ctol {
        (
            false,
            ConvergenceStatus::Infeasible,
            String::from(INFEASIBLE_MESSAGE),
        )
    } else {
        let status = match out.info {
            CobylaInfo::SmallTrRadius | CobylaInfo::FtargetAchieved => ConvergenceStatus::Success,
            CobylaInfo::MaxfunReached => ConvergenceStatus::MaxEvaluations,
            CobylaInfo::MaxtrReached => ConvergenceStatus::MaxIterations,
            CobylaInfo::NanInfX | CobylaInfo::NanInfF => ConvergenceStatus::NanEncountered,
            CobylaInfo::DamagingRounding => ConvergenceStatus::PrecisionLoss,
            CobylaInfo::CallbackTerminate => ConvergenceStatus::CallbackStop,
        };
        (
            status == ConvergenceStatus::Success,
            status,
            out.info.message(),
        )
    };
    Ok(OptimizeResult {
        x: expand(&out.x),
        fun: Some(out.f),
        success,
        status,
        message,
        nfev: out.nf,
        njev: 0,
        nhev: 0,
        nit: out.nit,
        jac: None,
        hess_inv: None,
        maxcv: Some(out.cstrv),
    })
}

/// `scipy.optimize.minimize(fun, x0, method='COBYLA', bounds=..., constraints=..., tol=...,
/// options={'rhobeg', 'maxiter', 'catol', 'f_target'})`.
///
/// SciPy's option mapping: `options.method_options.rhobeg` (default 1.0) is the initial
/// trust-region radius, `tol` is `rhoend` (default 1e-4), `maxiter` is the evaluation budget
/// `maxfun` (default 1000), `catol` the absolute constraint tolerance (default √ε) and
/// `f_target` the stopping target (default −∞). `maxfev` and `gradient_eps` are options SciPy's
/// COBYLA does not take: Strict records them in the optimize trace and ignores them, Hardened
/// refuses them. `options.constraints` are honoured as SciPy honours them: a constraint dict is
/// `NonlinearConstraint(fun, 0, 0)` (`eq`) or `(fun, 0, inf)` (`ineq`), a
/// [`Constraint::from_linear`] conversion is the original `LinearConstraint` (handled linearly,
/// with `x0` projected onto it when there are no nonlinear constraints), a
/// [`Constraint::from_nonlinear`] one the original `NonlinearConstraint`; `options.bounds` are
/// linear constraints too, with variables fixed by them removed. `success` is SciPy's: the
/// returned point violates the constraints by at most `catol` and the trust region reached
/// `rhoend` (or `f_target` was met). `nit` counts trust-region iterations; `status` maps PRIMA's
/// exit flag ([`CobylaInfo`]) onto [`ConvergenceStatus`], with `Infeasible` for SciPy's
/// "Did not converge to a solution satisfying the constraints".
pub fn minimize_cobyla<F>(
    fun: &F,
    x0: &[f64],
    options: MinimizeOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
{
    crate::minimize::check_cobyla_options(options)?;
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must be a finite 1-D vector with at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    crate::minimize::validate_bounds_for_x0(x0, options.bounds)?;
    let n = x0.len();
    let (lb, ub): (Vec<f64>, Vec<f64>) = match options.bounds {
        Some(bounds) => bounds
            .iter()
            .map(|&(lo, hi)| (lo.unwrap_or(f64::NEG_INFINITY), hi.unwrap_or(f64::INFINITY)))
            .unzip(),
        None => (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]),
    };

    let mut linear = Vec::new();
    let mut nonlinear = Vec::new();
    for c in options.constraints {
        match c.origin {
            Some(ConstraintOrigin { piece, .. }) if piece > 0 => {}
            Some(ConstraintOrigin {
                object: NewConstraint::Linear(lc),
                ..
            }) => linear.push(lc),
            Some(ConstraintOrigin {
                object: NewConstraint::Nonlinear(nlc),
                ..
            }) => nonlinear.push(NlPiece::Object(nlc)),
            None => nonlinear.push(NlPiece::Dict(c)),
        }
    }

    let mo = options.method_options;
    let maxfun = options.maxiter.unwrap_or(SCIPY_MAXITER);
    let fe = FrontEnd {
        lb,
        ub,
        linear,
        nonlinear,
        settings: CoreSettings {
            rhobeg: mo.rhobeg.unwrap_or(SCIPY_RHOBEG),
            rhoend: options.tol.unwrap_or(SCIPY_TOL),
            maxfun,
            ctol: mo.catol.unwrap_or(EPS.sqrt()),
            ftarget: mo.f_target.unwrap_or(f64::NEG_INFINITY),
            callback: options.callback,
        },
        mode: options.mode,
    };
    let result = run_front_end(fun, x0, &fe)?;
    crate::minimize::log_completion(OptimizeMethod::Cobyla, options, result.nit, &result);
    Ok(result)
}

/// The arguments of `scipy.optimize.fmin_cobyla` after `func`, `x0` and `cons`, with SciPy's
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FminCobylaOptions {
    /// Initial change to the variables (default 1.0).
    pub rhobeg: f64,
    /// Final trust-region radius (default 1e-4).
    pub rhoend: f64,
    /// Maximum number of function evaluations (default 1000).
    pub maxfun: usize,
    /// Absolute tolerance on the constraint violation (default 2e-4).
    pub catol: f64,
}

impl Default for FminCobylaOptions {
    fn default() -> Self {
        Self {
            rhobeg: SCIPY_RHOBEG,
            rhoend: SCIPY_TOL,
            maxfun: SCIPY_MAXITER,
            catol: FMIN_CATOL,
        }
    }
}

/// `scipy.optimize.fmin_cobyla(func, x0, cons, rhobeg, rhoend, maxfun, catol)` returning the
/// whole result: minimize `func` subject to `c(x) >= 0` for every `c` in `cons`, each wrapped as
/// SciPy wraps it (`NonlinearConstraint(c, 0, inf)`), by Powell's COBYLA (see
/// [`minimize_cobyla`]).
pub fn cobyla<F, G>(
    func: F,
    x0: &[f64],
    cons: &[G],
    options: FminCobylaOptions,
) -> Result<OptimizeResult, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: String::from("x0 must be a finite 1-D vector with at least one element"),
        });
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: String::from("x0 must not contain NaN or Inf"),
        });
    }
    if options.maxfun == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxfun must be >= 1"),
        });
    }
    if !options.rhoend.is_finite() || options.rhoend <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("rhoend must be finite and > 0"),
        });
    }
    let n = x0.len();
    let nonlinear: Vec<NlPiece<'_>> = cons
        .iter()
        .map(|c| NlPiece::Scalar(c as &dyn Fn(&[f64]) -> f64))
        .collect();
    let fe = FrontEnd {
        lb: vec![f64::NEG_INFINITY; n],
        ub: vec![f64::INFINITY; n],
        linear: Vec::new(),
        nonlinear,
        settings: CoreSettings {
            rhobeg: options.rhobeg,
            rhoend: options.rhoend,
            maxfun: options.maxfun,
            ctol: options.catol,
            ftarget: f64::NEG_INFINITY,
            callback: None,
        },
        mode: RuntimeMode::Strict,
    };
    run_front_end(&func, x0, &fe)
}

/// `scipy.optimize.fmin_cobyla(func, x0, cons, ...)`: the minimiser `x` of [`cobyla`].
pub fn fmin_cobyla<F, G>(
    func: F,
    x0: &[f64],
    cons: &[G],
    options: FminCobylaOptions,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    cobyla(func, x0, cons, options).map(|r| r.x)
}

#[cfg(test)]
mod tests {
    //! Pinned values are SciPy 1.17.1's (`scipy.optimize.minimize(method='COBYLA')` /
    //! `fmin_cobyla`, NumPy 2.4 + OpenBLAS on x86-64), the objective and constraints written with
    //! the same floating-point operations in the same order as here.
    use super::*;
    use crate::types::MinimizeMethodOptions;

    fn sq(t: f64) -> f64 {
        t * t
    }

    fn quad21(x: &[f64]) -> f64 {
        sq(x[0] - 2.0) + sq(x[1] - 1.0)
    }

    fn cobyla_opts<'a>(constraints: &'a [Constraint<'a>]) -> MinimizeOptions<'a> {
        MinimizeOptions {
            method: Some(OptimizeMethod::Cobyla),
            constraints,
            ..MinimizeOptions::default()
        }
    }

    fn assert_bits(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.to_bits(), e.to_bits(), "{actual:?} vs {expected:?}");
        }
    }

    const SMALL_TR: &str =
        "Return from COBYLA because the trust region radius reaches its lower bound.";
    const MAXFUN_MSG: &str =
        "Return from COBYLA because the objective function has been evaluated MAXFUN times.";

    #[test]
    fn inequality_quadratic_matches_scipy() {
        let cons = [Constraint::ineq(|x: &[f64]| vec![1.0 - x[0] - x[1]])];
        let r = crate::minimize(quad21, &[0.0, 0.0], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&r.x, &[1.0, 0.0]);
        assert_eq!(r.fun, Some(2.0));
        assert_eq!(r.nfev, 24);
        assert!(r.success, "{r:?}");
        assert_eq!(r.status, ConvergenceStatus::Success);
        assert_eq!(r.message, SMALL_TR);
        assert_eq!(r.maxcv, Some(0.0));
        assert!(r.nit > 0);
    }

    #[test]
    fn f_target_stops_early_with_success() {
        let cons = [Constraint::ineq(|x: &[f64]| vec![1.0 - x[0] - x[1]])];
        let mut opts = cobyla_opts(&cons);
        opts.method_options = MinimizeMethodOptions {
            f_target: Some(2.5),
            ..MinimizeMethodOptions::default()
        };
        let r = crate::minimize(quad21, &[0.0, 0.0], opts).expect("cobyla");
        assert_bits(&r.x, &[1.0, 0.0]);
        assert_eq!(r.nfev, 2);
        assert!(r.success);
        assert_eq!(r.status, ConvergenceStatus::Success);
        assert_eq!(r.message, CobylaInfo::FtargetAchieved.message());
    }

    #[test]
    fn one_variable_with_and_without_constraint() {
        let cons = [Constraint::ineq(|x: &[f64]| vec![2.0 - x[0]])];
        let f = |x: &[f64]| sq(x[0] - 3.0);
        let r = crate::minimize(f, &[0.0], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&r.x, &[2.0]);
        assert_eq!(r.fun, Some(1.0));
        assert_eq!(r.nfev, 5);
        assert!(r.success);

        let mut opts = cobyla_opts(&[]);
        opts.method_options.rhobeg = Some(0.5);
        let r = crate::minimize(f, &[0.0], opts).expect("cobyla");
        assert_bits(&r.x, &[3.0]);
        assert_eq!(r.nfev, 21);
        assert!(r.success);
        assert_eq!(r.maxcv, Some(0.0));
    }

    #[test]
    fn bounds_project_an_outside_x0() {
        let bounds: [Bound; 2] = [(Some(0.0), Some(2.0)), (Some(-1.0), Some(2.0))];
        let mut opts = cobyla_opts(&[]);
        opts.bounds = Some(&bounds);
        let r = crate::minimize(
            |x: &[f64]| sq(x[0] + 1.0) + sq(x[1] - 3.0),
            &[5.0, -4.0],
            opts,
        )
        .expect("cobyla");
        assert_bits(&r.x, &[-2.220_446_049_250_313e-16, 2.0]);
        assert_eq!(r.fun, Some(1.999_999_999_999_999_6));
        assert_eq!(r.nfev, 10);
        assert_eq!(r.maxcv, Some(2.220_446_049_250_313e-16));
        assert!(r.success);
    }

    #[test]
    fn variables_fixed_by_bounds_are_removed() {
        let bounds: [Bound; 3] = [(None, None), (Some(0.5), Some(0.5)), (None, None)];
        let mut opts = cobyla_opts(&[]);
        opts.bounds = Some(&bounds);
        let f = |x: &[f64]| sq(x[0] - 1.0) + sq(x[1] - 2.0) + sq(x[2] + 1.0);
        let r = crate::minimize(f, &[0.0, 0.0, 0.0], opts).expect("cobyla");
        assert_bits(
            &r.x,
            &[0.999_966_018_944_180_1, 0.5, -1.000_033_493_388_307_3],
        );
        assert_eq!(r.nfev, 38);
        assert!(r.success);

        // Every variable fixed, or fixed variables with constraints: SciPy 1.17.1 fails on both.
        let all_fixed: [Bound; 2] = [(Some(1.0), Some(1.0)), (Some(2.0), Some(2.0))];
        let mut opts = cobyla_opts(&[]);
        opts.bounds = Some(&all_fixed);
        let err = crate::minimize(quad21, &[1.0, 2.0], opts).expect_err("all fixed");
        assert!(matches!(err, OptError::InvalidArgument { .. }), "{err:?}");
        let cons = [Constraint::ineq(|x: &[f64]| vec![1.0 - x[0]])];
        let mut opts = cobyla_opts(&cons);
        opts.bounds = Some(&bounds);
        let err = crate::minimize(f, &[0.0, 0.0, 0.0], opts).expect_err("fixed + constraints");
        assert!(matches!(err, OptError::InvalidArgument { .. }), "{err:?}");
    }

    #[test]
    fn maxiter_exhaustion_is_not_success() {
        let rosen = |x: &[f64]| sq(1.0 - x[0]) + 100.0 * sq(x[1] - x[0] * x[0]);
        let cons = [Constraint::ineq(|x: &[f64]| {
            vec![2.0 - x[0] * x[0] - x[1] * x[1]]
        })];
        let mut opts = cobyla_opts(&cons);
        opts.maxiter = Some(5);
        let r = crate::minimize(rosen, &[-1.2, 1.0], opts).expect("cobyla");
        assert_bits(&r.x, &[-0.700_013_650_319_275_3, 0.996_305_391_367_939_9]);
        assert_eq!(r.fun, Some(28.522_626_217_281_648));
        assert_eq!(r.nfev, 5);
        assert!(!r.success);
        assert_eq!(r.status, ConvergenceStatus::MaxEvaluations);
        assert_eq!(r.message, MAXFUN_MSG);
    }

    #[test]
    fn maxiter_below_n_plus_2_is_revised_in_strict_and_refused_in_hardened() {
        let mut opts = cobyla_opts(&[]);
        opts.maxiter = Some(2);
        let r = crate::minimize(quad21, &[0.0, 0.0], opts).expect("strict revises");
        assert_bits(&r.x, &[1.948_683_298_050_513_8, 1.316_227_766_016_838]);
        assert_eq!(r.nfev, 4);
        assert_eq!(r.status, ConvergenceStatus::MaxEvaluations);
        opts.mode = RuntimeMode::Hardened;
        let err = crate::minimize(quad21, &[0.0, 0.0], opts).expect_err("hardened refuses");
        assert!(matches!(err, OptError::InvalidArgument { .. }), "{err:?}");
    }

    #[test]
    fn invalid_rhobeg_is_revised_in_strict_and_refused_in_hardened() {
        let cons = [Constraint::ineq(|x: &[f64]| vec![1.0 - x[0] - x[1]])];
        let mut opts = cobyla_opts(&cons);
        opts.method_options.rhobeg = Some(-1.0);
        let r = crate::minimize(quad21, &[0.0, 0.0], opts).expect("strict revises");
        // preproc resets RHOBEG to max(10 * rhoend, 1) = 1: the default run.
        assert_bits(&r.x, &[1.0, 0.0]);
        assert_eq!(r.nfev, 24);
        assert!(crate::get_optimize_traces().iter().any(|t| {
            t.event == "cobyla_input_revised"
                && t.reason.as_deref().is_some_and(|s| s.contains("RHOBEG"))
        }));
        opts.mode = RuntimeMode::Hardened;
        let err = crate::minimize(quad21, &[0.0, 0.0], opts).expect_err("hardened refuses");
        assert!(matches!(err, OptError::InvalidArgument { .. }), "{err:?}");
    }

    #[test]
    fn unknown_option_is_refused_in_hardened() {
        let mut opts = cobyla_opts(&[]);
        opts.method_options.gtol = Some(1e-6);
        assert!(crate::minimize(quad21, &[0.0, 0.0], opts).is_ok());
        opts.mode = RuntimeMode::Hardened;
        assert!(crate::minimize(quad21, &[0.0, 0.0], opts).is_err());

        // SciPy's COBYLA has no `maxfev`: Strict ignores it (the run is the default one),
        // Hardened refuses it.
        let mut opts = cobyla_opts(&[]);
        opts.maxfev = Some(3);
        let default = crate::minimize(quad21, &[0.0, 0.0], cobyla_opts(&[])).expect("cobyla");
        let r = crate::minimize(quad21, &[0.0, 0.0], opts).expect("strict ignores maxfev");
        assert_eq!(r.nfev, default.nfev);
        assert!(r.nfev > 3);
        assert_bits(&r.x, &default.x);
        opts.mode = RuntimeMode::Hardened;
        assert!(crate::minimize(quad21, &[0.0, 0.0], opts).is_err());
    }

    #[test]
    fn nan_constraint_is_maximally_violated() {
        let cons = [Constraint::ineq(|_: &[f64]| vec![f64::NAN])];
        let f = |x: &[f64]| sq(x[0] - 1.0) + sq(x[1] - 2.0);
        let r = crate::minimize(f, &[0.0, 0.0], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&r.x, &[1.040_355_150_187_472_4, 1.993_048_515_427_859]);
        assert_eq!(r.nfev, 29);
        assert_eq!(r.maxcv, Some(CONSTRMAX));
        assert!(!r.success);
        assert_eq!(r.status, ConvergenceStatus::Infeasible);
        assert_eq!(r.message, INFEASIBLE_MESSAGE);
        // The same problem with a satisfiable constraint succeeds.
        let ok = [Constraint::ineq(|x: &[f64]| vec![2.0 - x[0]])];
        assert!(
            crate::minimize(f, &[0.0, 0.0], cobyla_opts(&ok))
                .expect("cobyla")
                .success
        );
    }

    #[test]
    fn contradictory_constraints_report_infeasible() {
        let cons = [
            Constraint::ineq(|x: &[f64]| vec![x[0] - 1.0]),
            Constraint::ineq(|x: &[f64]| vec![-x[0]]),
        ];
        let r =
            crate::minimize(|x: &[f64]| x[0] * x[0], &[0.5], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&r.x, &[0.5]);
        assert_eq!(r.nfev, 4);
        assert_eq!(r.maxcv, Some(0.5));
        assert!(!r.success);
        assert_eq!(r.status, ConvergenceStatus::Infeasible);
    }

    #[test]
    fn equality_constraint_matches_scipy() {
        let cons = [Constraint::eq(|x: &[f64]| {
            vec![x[0] + 2.0 * x[1] + 3.0 * x[2] - 6.0]
        })];
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let r = crate::minimize(f, &[1.0, 1.0, 1.0], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(
            &r.x,
            &[
                0.428_562_270_975_701_65,
                0.857_174_162_211_063_2,
                1.285_696_468_200_724_1,
            ],
        );
        assert_eq!(r.fun, Some(2.571_428_572_809_904_4));
        assert_eq!(r.nfev, 48);
        assert!(r.success);
    }

    #[test]
    fn hock_schittkowski_71_matches_scipy() {
        // Up to four active constraints: the Lagrange multipliers come from `np.linalg.lstsq`
        // with three and four columns (LAPACK `dgelsd`'s full small-matrix path).
        let cons = [
            Constraint::ineq(|x: &[f64]| vec![x[0] * x[1] * x[2] * x[3] - 25.0]),
            Constraint::eq(|x: &[f64]| {
                vec![x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0]
            }),
        ];
        let bounds: [Bound; 4] = [(Some(1.0), Some(5.0)); 4];
        let mut opts = cobyla_opts(&cons);
        opts.bounds = Some(&bounds);
        let f = |x: &[f64]| x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        let r = crate::minimize(f, &[1.0, 5.0, 5.0, 1.0], opts).expect("cobyla");
        assert_bits(
            &r.x,
            &[
                1.0,
                4.743_568_561_243_491,
                3.820_406_522_231_309_5,
                1.379_511_257_359_476_3,
            ],
        );
        assert_eq!(r.fun, Some(17.014_017_814_990_332));
        assert_eq!(r.nfev, 82);
        assert_eq!(r.maxcv, Some(4.933_085_051_561_648e-10));
        assert!(r.success);
    }

    fn disc(x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1]]
    }

    fn two_outputs(x: &[f64]) -> Vec<f64> {
        vec![x[0], x[1]]
    }

    #[test]
    fn linear_and_nonlinear_constraint_objects() {
        // One range row and one one-sided row: `from_linear` splits them into two constraints,
        // and COBYLA must see the original LinearConstraint exactly once.
        let lc = LinearConstraint::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.5, f64::NEG_INFINITY],
            vec![0.8, 0.3],
        )
        .expect("linear constraint");
        let cons = Constraint::from_linear(&lc);
        let f = |x: &[f64]| sq(x[0] - 1.0) + sq(x[1] - 1.0);
        let r = crate::minimize(f, &[0.0, 0.0], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&r.x, &[0.8, 0.300_000_000_000_000_04]);
        assert_eq!(r.nfev, 12);
        assert_eq!(r.maxcv, Some(5.551_115_123_125_783e-17));
        assert!(r.success);

        let nlc = NonlinearConstraint::new(disc, vec![f64::NEG_INFINITY], vec![1.0])
            .expect("nonlinear constraint");
        let cons = Constraint::from_nonlinear(&nlc);
        let r = crate::minimize(|x: &[f64]| -(x[0] * x[1]), &[0.5, 0.5], cobyla_opts(&cons))
            .expect("cobyla");
        assert_bits(&r.x, &[0.707_000_517_625_467_9, 0.707_212_984_654_440_3]);
        assert_eq!(r.nfev, 37);
        assert!(r.success);
    }

    #[test]
    fn nonlinear_constraint_output_length_is_checked() {
        let nlc = NonlinearConstraint::new(two_outputs, vec![0.0], vec![1.0])
            .expect("nonlinear constraint");
        let cons = Constraint::from_nonlinear(&nlc);
        let err =
            crate::minimize(quad21, &[0.0, 0.0], cobyla_opts(&cons)).expect_err("length mismatch");
        assert!(matches!(err, OptError::InvalidArgument { .. }), "{err:?}");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(matches!(
            minimize_cobyla(&quad21, &[], cobyla_opts(&[])),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            minimize_cobyla(&quad21, &[f64::NAN, 0.0], cobyla_opts(&[])),
            Err(OptError::NonFiniteInput { .. })
        ));
        let bad: [Bound; 2] = [(Some(1.0), Some(0.0)), (None, None)];
        let mut opts = cobyla_opts(&[]);
        opts.bounds = Some(&bad);
        assert!(minimize_cobyla(&quad21, &[0.5, 0.0], opts).is_err());
        let short: [Bound; 1] = [(None, None)];
        opts.bounds = Some(&short);
        assert!(minimize_cobyla(&quad21, &[0.5, 0.0], opts).is_err());
        let lc = LinearConstraint::new(vec![vec![1.0, 1.0, 1.0]], vec![0.0], vec![1.0])
            .expect("linear constraint");
        let cons = Constraint::from_linear(&lc);
        assert!(matches!(
            minimize_cobyla(&quad21, &[0.0, 0.0], cobyla_opts(&cons)),
            Err(OptError::InvalidArgument { .. })
        ));
    }

    fn stop_now(_: &[f64]) -> bool {
        true
    }

    #[test]
    fn callback_can_stop_the_iteration() {
        let cons = [Constraint::ineq(|x: &[f64]| vec![1.0 - x[0] - x[1]])];
        let mut opts = cobyla_opts(&cons);
        opts.callback = Some(stop_now);
        let r = crate::minimize(quad21, &[0.0, 0.0], opts).expect("cobyla");
        assert!(!r.success);
        assert_eq!(r.status, ConvergenceStatus::CallbackStop);
        assert_eq!(r.nit, 1);
        assert_eq!(r.message, CobylaInfo::CallbackTerminate.message());
    }

    fn ten_d(x: &[f64]) -> f64 {
        let mut s = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            s += (i + 1) as f64 * sq(xi - 0.5 * (i % 3) as f64);
        }
        s + x[0] * x[9]
    }

    #[test]
    fn ten_variables_five_constraints_is_deterministic() {
        let cons: Vec<Constraint<'_>> = (0..5)
            .map(|k| {
                Constraint::ineq(move |x: &[f64]| {
                    let mut s = 0.0;
                    for j in 0..3 {
                        s += sq(x[(k + j) % 10]);
                    }
                    vec![4.0 - s - x[k]]
                })
            })
            .collect();
        let a = crate::minimize(ten_d, &[0.0; 10], cobyla_opts(&cons)).expect("cobyla");
        let b = crate::minimize(ten_d, &[0.0; 10], cobyla_opts(&cons)).expect("cobyla");
        assert_bits(&a.x, &b.x);
        assert_eq!(a.fun.map(f64::to_bits), b.fun.map(f64::to_bits));
        assert_eq!((a.nfev, a.nit, a.status), (b.nfev, b.nit, b.status));
        assert!(a.success, "{a:?}");
        assert!(a.maxcv.expect("maxcv") <= f64::EPSILON.sqrt());
    }

    #[test]
    fn fmin_cobyla_documentation_example() {
        let cons = [
            |x: &[f64]| 1.0 - (x[0] * x[0] + x[1] * x[1]),
            |x: &[f64]| x[1],
        ];
        let opts = FminCobylaOptions {
            rhoend: 1e-7,
            ..FminCobylaOptions::default()
        };
        let x =
            fmin_cobyla(|x: &[f64]| x[0] * x[1], &[0.0, 0.1], &cons, opts).expect("fmin_cobyla");
        assert_bits(&x, &[-0.707_141_950_356_017, 0.707_142_454_810_020_6]);
    }

    #[test]
    fn cobyla_wrapper_semantics() {
        let sum_ge_1 = [|x: &[f64]| x[0] + x[1] - 1.0];
        let opts = FminCobylaOptions {
            rhobeg: 0.5,
            ..FminCobylaOptions::default()
        };
        let r = cobyla(|x: &[f64]| x[0] + x[1], &[2.0, 2.0], &sum_ge_1, opts).expect("cobyla");
        assert_bits(&r.x, &[0.5, 0.5]);
        assert!(r.success);

        let none: [fn(&[f64]) -> f64; 0] = [];
        let starved = cobyla(
            |x: &[f64]| sq(x[0] - 3.0),
            &[0.0],
            &none,
            FminCobylaOptions {
                maxfun: 3,
                ..FminCobylaOptions::default()
            },
        )
        .expect("runs");
        assert!(!starved.success);
        assert_eq!(starved.status, ConvergenceStatus::MaxEvaluations);
        assert_eq!(starved.nfev, 3);

        assert!(matches!(
            cobyla(|_: &[f64]| 0.0, &[], &none, FminCobylaOptions::default()),
            Err(OptError::InvalidArgument { .. })
        ));
        assert!(matches!(
            cobyla(
                |_: &[f64]| 0.0,
                &[f64::INFINITY],
                &none,
                FminCobylaOptions::default()
            ),
            Err(OptError::NonFiniteInput { .. })
        ));
        let zero = FminCobylaOptions {
            maxfun: 0,
            ..FminCobylaOptions::default()
        };
        assert!(cobyla(|_: &[f64]| 0.0, &[1.0], &none, zero).is_err());
        let bad_rhoend = FminCobylaOptions {
            rhoend: f64::NAN,
            ..FminCobylaOptions::default()
        };
        assert!(cobyla(|_: &[f64]| 0.0, &[1.0], &none, bad_rhoend).is_err());
    }

    #[test]
    fn info_codes_and_messages() {
        assert_eq!(CobylaInfo::SmallTrRadius.code(), 0);
        assert_eq!(CobylaInfo::FtargetAchieved.code(), 1);
        assert_eq!(CobylaInfo::MaxfunReached.code(), 3);
        assert_eq!(CobylaInfo::MaxtrReached.code(), 20);
        assert_eq!(CobylaInfo::DamagingRounding.code(), 7);
        assert_eq!(CobylaInfo::CallbackTerminate.code(), 30);
        assert_eq!(CobylaInfo::SmallTrRadius.message(), SMALL_TR);
        assert_eq!(CobylaInfo::MaxfunReached.message(), MAXFUN_MSG);
    }

    #[test]
    fn python_reductions_follow_numpy_semantics() {
        assert_eq!(np_argmin(&[2.0, f64::NAN, 1.0]), 1);
        assert_eq!(np_argmax(&[1.0, 3.0, 3.0]), 1);
        assert!(np_maximum(f64::NAN, 1.0).is_nan());
        assert_eq!(py_max2(1.0, f64::NAN), 1.0);
        assert!(isminor(1e-20, 1.0));
        assert!(!isminor(0.5, 1.0));
    }
}
