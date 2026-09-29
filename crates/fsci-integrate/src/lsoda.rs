#![forbid(unsafe_code)]
//! ODEPACK's LSODA (A. C. Hindmarsh; automatic Adams/BDF switching after L. R. Petzold, SIAM J.
//! Sci. Stat. Comput. 4 (1983) 136-148) exactly as SciPy 1.17.1 runs it.
//!
//! SciPy 1.17.1 no longer builds the Fortran: `scipy.integrate.odeint`, `ode(...).set_integrator
//! ("lsoda")` and `solve_ivp(method="LSODA")` all call `scipy/integrate/src/lsoda.c`, a C
//! translation of DLSODA and the routines under it. This module ports that C file routine by
//! routine, keeping its operations and their order so the step sequence is SciPy's:
//!
//! | SciPy `lsoda.c` (ODEPACK)                   | here                                  |
//! |---------------------------------------------|---------------------------------------|
//! | `lsoda` (DLSODA), blocks A-G                | `Lsoda::new`, `Lsoda::call`, `Lsoda::initialize` |
//! | `stoda` (DSTODA)                            | `Lsoda::stoda`                        |
//! | `stoda_first_call_init` (jstart = 0)        | `Lsoda::first_call_init`              |
//! | label 150 (`stoda_reset`)                   | `Lsoda::reset_el`                     |
//! | labels 170-178 (`stoda_adjust_step_size`)   | `Lsoda::adjust_step_size`             |
//! | label 200 (`stoda_get_predicted_values`)    | `Lsoda::predict` (and `retract`)      |
//! | labels 220-410 (`stoda_corrector_loop`)     | `Lsoda::corrector`                    |
//! | labels 430-445 (`stoda_handle_corrector_failure`) | `Lsoda::corrector_failure`      |
//! | labels 470-478 / 480-486 (method switch)    | `Lsoda::consider_bdf` / `consider_adams` |
//! | `cfode` (DCFODE)                            | `cfode`                               |
//! | `prja` (DPRJA), full matrix                 | `Lsoda::prja`                         |
//! | `dgetrf_`, and `solsy` with `dgetrs_`       | `lu_factor`, `lu_solve`               |
//! | `intdy` (DINTDY), k = 0                     | `Lsoda::intdy`                        |
//! | `ewset` (DEWSET)                            | `Lsoda::ewset`                        |
//! | `vmnorm` (DVMNORM), `fnorm` (DFNORM)        | `vmnorm`, `fnorm`                     |
//!
//! Work arrays are separate vectors instead of offsets into `rwork`/`iwork`; the only place the
//! C shares storage (Adams history columns 6-12 overlap the BDF matrix `wm`) is never read before
//! it is written on either side of a method switch, so the split does not change a result.
//!
//! What is deliberately NOT ported, and why:
//! - Banded Jacobians (`jt` = 4, 5; `prja` miter 4/5, `bnorm`, `dgbtrf`/`dgbtrs`).
//! - `itask` 2, 3 and 4 (only `odeint`'s 1 and `solve_ivp`'s 5 have callers).
//! - `hmin`: SciPy 1.17.1's `lsoda()` reads `rwork[6]` into a local variable and never stores it
//!   in the common block, so the stepper always runs with `hmin = 0` whatever `min_step` or
//!   `odeint(hmin=...)` says (measured: `solve_ivp(..., method="LSODA", min_step=0.5)` takes the
//!   identical 727 steps as `min_step=0` on van der Pol mu = 1000). The `hmin` field is that 0.
//! - The `nhnil` ("t + h = t") warning counter, `ixpr` printing and the workspace-size checks,
//!   which print or size arrays but never change a step.
//!
//! LAPACK: SciPy factors `P = I - h*el0*J` with `dgetrf`/`dgetrs` from its bundled OpenBLAS.
//! `lu_factor` follows OpenBLAS's unblocked `getf2` (the path `dgetrf` takes at these sizes):
//! left-looking, dot-then-subtract updates, first-maximum pivot, reciprocal scaling; `lu_solve`
//! is `dlaswp` and the two `trsv` sweeps. It uses no fused multiply-add. For n = 1 and on Adams
//! no LAPACK arithmetic is involved at all. Measured against SciPy on the pinned host
//! (`diff_integrate_lsoda`), the BDF rows with n = 2, 3 and 8 are bit-identical too, with SciPy's
//! OpenBLAS on its default Haswell kernel (FMA) and under `OPENBLAS_CORETYPE` Nehalem, Sandybridge
//! and Prescott alike; a BLAS kernel that rounded these small updates differently would move last
//! bits, and through them possibly a step decision, on SciPy's side.

use crate::solver::StepFailure;

/// A user Jacobian for LSODA (`jt = 1`): `jac(t, y)[i][j] = d f_i / d y_j`, `n` rows of `n`.
pub type JacFn = fn(f64, &[f64]) -> Vec<Vec<f64>>;

/// Stability-region step bounds for the Adams methods of order 1..12 (`sm1` in `stoda`).
const SM1: [f64; 12] = [
    0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.20, 0.15, 0.10, 0.075, 0.050, 0.025,
];
/// Maximum orders of the Adams (1) and BDF (2) families (`mord`).
const MORD: [usize; 2] = [12, 5];
/// Columns of the Nordsieck history: order up to 12, plus the saved-correction column.
const YH_COLUMNS: usize = 13;

/// The two LSODA method families (`meth` 1 and 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsodaMethod {
    /// Implicit Adams, orders 1-12, functional (fixed-point) iteration: the nonstiff method.
    Adams,
    /// Backward differentiation formulas, orders 1-5, chord (Newton) iteration: the stiff method.
    Bdf,
}

impl LsodaMethod {
    fn from_meth(meth: usize) -> Option<Self> {
        match meth {
            1 => Some(Self::Adams),
            2 => Some(Self::Bdf),
            _ => None,
        }
    }
}

/// What one [`Lsoda::call`] is asked to do (`itask`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Task {
    /// `itask = 1`: step past `tout` and interpolate there (`odeint`).
    Interpolate,
    /// `itask = 5`: take one step without passing `tcrit` (`solve_ivp`).
    OneStepToTcrit,
}

/// A failed [`Lsoda::call`], by ODEPACK's `istate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LsodaFailure {
    /// istate = -1: `mxstep` steps on this call before reaching `tout`.
    ExcessWork,
    /// istate = -2: the tolerances are too small for the machine precision at the current t.
    ExcessAccuracy,
    /// istate = -3: illegal input.
    IllegalInput(&'static str),
    /// istate = -4: repeated error test failures.
    ErrorTestFailures,
    /// istate = -5: repeated corrector convergence failures.
    ConvergenceFailures,
    /// istate = -6: an error weight became zero (a pure-relative tolerance on a component that
    /// reached 0).
    ZeroErrorWeight,
    /// The right-hand side or the Jacobian returned the wrong shape (SciPy raises there).
    Callback(StepFailure),
}

impl LsodaFailure {
    /// SciPy's message for this `istate` (`scipy.integrate._ode.lsoda.messages`); for illegal
    /// input, which SciPy reports as "Illegal input detected (internal error).", the check that
    /// failed.
    pub(crate) fn message(&self) -> &'static str {
        match self {
            Self::ExcessWork => "Excess work done on this call (perhaps wrong Dfun type).",
            Self::ExcessAccuracy => "Excess accuracy requested (tolerances too small).",
            Self::IllegalInput(detail) => detail,
            Self::ErrorTestFailures => "Repeated error test failures (internal error).",
            Self::ConvergenceFailures => {
                "Repeated convergence failures (perhaps bad Jacobian or tolerances)."
            }
            Self::ZeroErrorWeight => "Error weight became zero during problem.",
            Self::Callback(StepFailure::RuntimeError(message)) => message,
            Self::Callback(_) => "The right-hand side or Jacobian callback failed.",
        }
    }

    pub(crate) fn into_step_failure(self) -> StepFailure {
        match self {
            Self::Callback(failure) => failure,
            Self::ConvergenceFailures => StepFailure::ConvergenceFailure,
            other => StepFailure::RuntimeError(other.message()),
        }
    }
}

const RHS_WRONG_SHAPE: &str = "The right-hand side returned an array of the wrong shape.";
const JAC_WRONG_SHAPE: &str = "The Jacobian returned an array of the wrong shape.";

/// Evaluate the right-hand side into `out` (the C `f(neq, t, y, out)` callback).
fn eval<F>(f: &mut F, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), LsodaFailure>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let value = f(t, y);
    if value.len() != out.len() {
        return Err(LsodaFailure::Callback(StepFailure::RuntimeError(
            RHS_WRONG_SHAPE,
        )));
    }
    out.copy_from_slice(&value);
    Ok(())
}

/// DVMNORM: `max_i |v_i| * w_i` (`fmax`, so a NaN component is skipped exactly as in C).
fn vmnorm(v: &[f64], w: &[f64]) -> f64 {
    let mut vm = 0.0_f64;
    for (vi, wi) in v.iter().zip(w) {
        vm = vm.max(vi.abs() * wi);
    }
    vm
}

/// DFNORM: the matrix norm consistent with [`vmnorm`], `max_i w_i sum_j |a_ij| / w_j`, for the
/// column-major `n x n` matrix `a`.
fn fnorm(n: usize, a: &[f64], w: &[f64]) -> f64 {
    let mut an = 0.0_f64;
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[i + j * n].abs() / w[j];
        }
        an = an.max(sum * w[i]);
    }
    an
}

/// DCFODE: the method coefficients `elco` (13 x 12, column `nq - 1` for order `nq`) and test
/// constants `tesco` (3 x 12) of the Adams (`meth = 1`) or BDF (`meth = 2`) family, both
/// column-major as in the C.
fn cfode(meth: usize, elco: &mut [f64; 156], tesco: &mut [f64; 36]) {
    let mut pc = [0.0_f64; 12];
    if meth == 1 {
        elco[0] = 1.0;
        elco[1] = 1.0;
        tesco[0] = 0.0;
        tesco[1] = 2.0;
        tesco[3] = 1.0;
        tesco[35] = 0.0;
        pc[0] = 1.0;
        let mut rqfac = 1.0_f64;
        for nq in 1..12_usize {
            // Form the coefficients of p(x)*(x + nq - 1) (C loop index nq is order - 1).
            let rq1fac = rqfac;
            rqfac /= (nq + 1) as f64;
            let fnq = nq as f64;
            pc[nq] = 0.0;
            for ib in 0..nq {
                pc[nq - ib] = pc[nq - ib - 1] + fnq * pc[nq - ib];
            }
            pc[0] *= fnq;
            // Integrals over [-1, 0] of p(x) and x*p(x).
            let mut pint = pc[0];
            let mut xpin = pc[0] * 0.5;
            let mut tsign = 1.0;
            for i in 1..=nq {
                tsign = -tsign;
                pint += tsign * pc[i] / (i + 1) as f64;
                xpin += tsign * pc[i] / (i + 2) as f64;
            }
            elco[nq * 13] = pint * rq1fac;
            elco[1 + nq * 13] = 1.0;
            for i in 1..=nq {
                elco[(i + 1) + nq * 13] = rq1fac * pc[i] / (i + 1) as f64;
            }
            let agamq = rqfac * xpin;
            let ragq = 1.0 / agamq;
            tesco[1 + 3 * nq] = ragq;
            if nq < 11 {
                tesco[3 * (nq + 1)] = ragq * rqfac / (nq + 2) as f64;
            }
            tesco[2 + 3 * (nq - 1)] = ragq;
        }
        return;
    }
    // BDF: p(x) = (x+1)(x+2)...(x+nq), el = p / (nq! (1 + 1/2 + ... + 1/nq)).
    pc[0] = 1.0;
    let mut rq1fac = 1.0_f64;
    for nq in 0..5_usize {
        let fnq1 = (nq + 1) as f64;
        pc[nq + 1] = 0.0;
        for ib in 0..=nq {
            pc[nq - ib + 1] = pc[nq - ib] + fnq1 * pc[nq - ib + 1];
        }
        pc[0] *= fnq1;
        for i in 0..=nq + 1 {
            elco[i + nq * 13] = pc[i] / pc[1];
        }
        elco[1 + nq * 13] = 1.0;
        tesco[3 * nq] = rq1fac;
        tesco[1 + 3 * nq] = (nq + 2) as f64 / elco[nq * 13];
        tesco[2 + 3 * nq] = (nq + 3) as f64 / elco[nq * 13];
        rq1fac /= fnq1;
    }
}

/// `dgetrf` on the column-major `n x n` matrix `a` as OpenBLAS's unblocked `getf2` computes it:
/// for each column, apply the earlier interchanges, finish the U part with dot products
/// (`a_ij -= sum_k l_ik u_kj`), update the rest of the column the same way, pick the first
/// largest magnitude as pivot, swap rows `j` and `p` over columns `0..=j`, and scale the
/// multipliers by the pivot's reciprocal. Returns `false` on an exactly zero pivot (LAPACK's
/// `info > 0`, which LSODA treats as a singular iteration matrix).
fn lu_factor(a: &mut [f64], ipvt: &mut [usize], n: usize) -> bool {
    let mut nonsingular = true;
    for j in 0..n {
        for i in 0..j {
            let p = ipvt[i];
            if p != i {
                a.swap(i + j * n, p + j * n);
            }
        }
        for i in 1..j {
            let mut dot = 0.0;
            for k in 0..i {
                dot += a[i + k * n] * a[k + j * n];
            }
            a[i + j * n] -= dot;
        }
        if j > 0 {
            for i in j..n {
                let mut dot = 0.0;
                for k in 0..j {
                    dot += a[i + k * n] * a[k + j * n];
                }
                a[i + j * n] -= dot;
            }
        }
        let mut jp = j;
        let mut big = a[j + j * n].abs();
        for i in j + 1..n {
            let v = a[i + j * n].abs();
            if v > big {
                big = v;
                jp = i;
            }
        }
        ipvt[j] = jp;
        let pivot = a[jp + j * n];
        if pivot == 0.0 {
            nonsingular = false;
            continue;
        }
        if jp != j {
            for k in 0..=j {
                a.swap(j + k * n, jp + k * n);
            }
        }
        if pivot.abs() >= f64::MIN_POSITIVE {
            let r = 1.0 / pivot;
            for i in j + 1..n {
                a[i + j * n] *= r;
            }
        } else {
            for i in j + 1..n {
                a[i + j * n] /= pivot;
            }
        }
    }
    nonsingular
}

/// `dgetrs('N')` with one right-hand side: the row interchanges, then unit-lower forward and
/// upper backward substitution, each column applied as an `axpy` (`b_i += (-b_j) * a_ij`), with
/// a true division by each diagonal entry.
fn lu_solve(a: &[f64], ipvt: &[usize], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let p = ipvt[i];
        if p != i {
            b.swap(i, p);
        }
    }
    for j in 0..n {
        let alpha = -b[j];
        for i in j + 1..n {
            b[i] += alpha * a[i + j * n];
        }
    }
    for j in (0..n).rev() {
        b[j] /= a[j + j * n];
        let alpha = -b[j];
        for i in 0..j {
            b[i] += alpha * a[i + j * n];
        }
    }
}

/// How one pass of the corrector ended (`stoda_corrector_loop`'s exits).
enum Corrector {
    /// Converged after `m` extra iterations with last correction norm `del` (label 450).
    Converged { m: usize, del: f64 },
    /// Diverged or too slow with a stale Jacobian: re-evaluate it and retry (label 220).
    Retry,
    /// Failed with a current Jacobian, or the iteration matrix was singular (label 430).
    Failed,
}

/// The inputs of one LSODA problem, fixed for its lifetime.
pub(crate) struct LsodaSetup {
    /// `rtol`, one value (`itol` 1, 2) or one per component (3, 4).
    pub rtol: Vec<f64>,
    /// `atol`, one value (`itol` 1, 3) or one per component (2, 4).
    pub atol: Vec<f64>,
    /// `h0` (`rwork[4]`): the signed first step, or 0 to let LSODA choose.
    pub h0: f64,
    /// `hmax` (`rwork[5]`): the largest |h|, or 0 for no bound.
    pub hmax: f64,
    /// `mxstep` (`iwork[5]`): steps allowed per call.
    pub mxstep: usize,
    /// `jac` for `jt = 1`; `None` is `jt = 2`, a finite-difference Jacobian.
    pub jac: Option<JacFn>,
}

/// The state of one LSODA integration: SciPy's `lsoda_common_struct_t` (the Fortran common
/// blocks LS0001 and LSA001) plus the work arrays that live in `rwork`/`iwork`.
pub(crate) struct Lsoda {
    n: usize,
    rtol: Vec<f64>,
    atol: Vec<f64>,
    jac: Option<JacFn>,
    h0: f64,
    hmxi: f64,
    /// Always 0: SciPy 1.17.1 never stores `rwork[6]` in the common block (module docs).
    hmin: f64,
    mxstep: usize,
    mxordn: usize,
    mxords: usize,
    jtyp: usize,
    /// istate > 1: the first call has initialized the problem.
    started: bool,

    // ls0001 / lsa001, doubles.
    conit: f64,
    crate_: f64,
    el: [f64; 13],
    elco: [f64; 156],
    hold: f64,
    rmax: f64,
    tesco: [f64; 36],
    ccmax: f64,
    el0: f64,
    h: f64,
    hu: f64,
    rc: f64,
    tn: f64,
    uround: f64,
    tsw: f64,
    pdest: f64,
    pdlast: f64,
    ratio: f64,
    cm1: [f64; 12],
    cm2: [f64; 5],
    pdnorm: f64,

    // ls0001 / lsa001, integers.
    nslast: usize,
    ialth: i32,
    ipup: usize,
    lmax: usize,
    nqnyh: usize,
    nslp: usize,
    /// `ierpj == 0`: the last `prja` produced a nonsingular iteration matrix.
    lu_ok: bool,
    /// `jcur == 1`: the Jacobian in `wm` was evaluated on this step.
    jcur: bool,
    jstart: i32,
    kflag: i32,
    l: usize,
    meth: usize,
    miter: usize,
    maxord: usize,
    maxcor: usize,
    msbp: usize,
    mxncf: usize,
    nq: usize,
    nst: usize,
    nfe: usize,
    nje: usize,
    nqu: usize,
    icount: i32,
    irflag: bool,
    mused: usize,

    // Work arrays (rwork/iwork regions).
    /// Nordsieck history `yh`, column-major `n x YH_COLUMNS`: column j holds h^j y^(j) / j!.
    yh: Vec<f64>,
    /// Inverted error weights `1 / (rtol |y| + atol)`.
    ewt: Vec<f64>,
    savf: Vec<f64>,
    acor: Vec<f64>,
    /// The LU factors of `P = I - h el0 J`, column-major (`wm[2..]`).
    wm: Vec<f64>,
    ipvt: Vec<usize>,
    /// `stoda`'s `y` argument: predicted and corrected values, scratch between calls.
    y: Vec<f64>,
}

impl Lsoda {
    /// An LSODA problem of `n` equations that has not taken its first call (istate = 1).
    pub(crate) fn new(n: usize, setup: LsodaSetup) -> Self {
        let jtyp = if setup.jac.is_some() { 1 } else { 2 };
        Self {
            n,
            rtol: setup.rtol,
            atol: setup.atol,
            jac: setup.jac,
            h0: setup.h0,
            hmxi: if setup.hmax > 0.0 {
                1.0 / setup.hmax
            } else {
                0.0
            },
            hmin: 0.0,
            mxstep: setup.mxstep,
            mxordn: MORD[0],
            mxords: MORD[1],
            jtyp,
            started: false,
            conit: 0.0,
            crate_: 0.0,
            el: [0.0; 13],
            elco: [0.0; 156],
            hold: 0.0,
            rmax: 0.0,
            tesco: [0.0; 36],
            ccmax: 0.0,
            el0: 0.0,
            h: 0.0,
            hu: 0.0,
            rc: 0.0,
            tn: 0.0,
            uround: f64::EPSILON,
            tsw: 0.0,
            pdest: 0.0,
            pdlast: 0.0,
            ratio: 0.0,
            cm1: [0.0; 12],
            cm2: [0.0; 5],
            pdnorm: 0.0,
            nslast: 0,
            ialth: 0,
            ipup: 0,
            lmax: 0,
            nqnyh: 0,
            nslp: 0,
            lu_ok: true,
            jcur: false,
            jstart: 0,
            kflag: 0,
            l: 0,
            meth: 1,
            miter: 0,
            maxord: 0,
            maxcor: 0,
            msbp: 0,
            mxncf: 0,
            nq: 0,
            nst: 0,
            nfe: 0,
            nje: 0,
            nqu: 0,
            icount: 0,
            irflag: false,
            mused: 0,
            yh: vec![0.0; n * YH_COLUMNS],
            ewt: vec![0.0; n],
            savf: vec![0.0; n],
            acor: vec![0.0; n],
            wm: vec![0.0; n * n],
            ipvt: vec![0; n],
            y: vec![0.0; n],
        }
    }

    fn rtol_at(&self, i: usize) -> f64 {
        if self.rtol.len() == 1 {
            self.rtol[0]
        } else {
            self.rtol[i]
        }
    }

    fn atol_at(&self, i: usize) -> f64 {
        if self.atol.len() == 1 {
            self.atol[0]
        } else {
            self.atol[i]
        }
    }

    /// NST (`iwork[10]`): accepted steps.
    pub(crate) fn nst(&self) -> usize {
        self.nst
    }

    /// NFE (`iwork[11]`): right-hand-side evaluations, those of finite-difference Jacobians
    /// included.
    pub(crate) fn nfe(&self) -> usize {
        self.nfe
    }

    /// NJE (`iwork[12]`): Jacobian evaluations, each followed by one LU factorization.
    pub(crate) fn nje(&self) -> usize {
        self.nje
    }

    /// NQU (`iwork[13]`): the order of the last accepted step.
    pub(crate) fn nqu(&self) -> usize {
        self.nqu
    }

    /// NQCUR (`iwork[14]`): the order to be attempted next.
    pub(crate) fn nq(&self) -> usize {
        self.nq
    }

    /// HU (`rwork[10]`): the size of the last accepted step.
    pub(crate) fn hu(&self) -> f64 {
        self.hu
    }

    /// HCUR (`rwork[11]`): the step size to be attempted next.
    pub(crate) fn h(&self) -> f64 {
        self.h
    }

    /// TSW (`rwork[14]`): the time of the last method switch (t0 if none).
    pub(crate) fn tsw(&self) -> f64 {
        self.tsw
    }

    /// MUSED (`iwork[18]`): the method of the last accepted step.
    pub(crate) fn method_used(&self) -> Option<LsodaMethod> {
        LsodaMethod::from_meth(self.mused)
    }

    /// MCUR (`iwork[19]`): the method to be attempted next.
    pub(crate) fn method_current(&self) -> Option<LsodaMethod> {
        LsodaMethod::from_meth(self.meth)
    }

    /// Column `j` of the Nordsieck history (`rwork[20 + j n ..]`).
    pub(crate) fn yh_column(&self, j: usize) -> &[f64] {
        &self.yh[j * self.n..(j + 1) * self.n]
    }

    /// DEWSET into `ewt` from the current solution `yh[:, 0]` (not yet inverted).
    fn ewset(&mut self) {
        for i in 0..self.n {
            self.ewt[i] = self.rtol_at(i) * self.yh[i].abs() + self.atol_at(i);
        }
    }

    /// Invert the error weights; `false` if one is not positive (the caller's error exit).
    fn invert_ewt(&mut self) -> bool {
        for w in &mut self.ewt {
            if *w <= 0.0 {
                return false;
            }
            *w = 1.0 / *w;
        }
        true
    }

    /// The error exits' `y = yh[:, 0]`, `t = tn`.
    fn current_state(&self, y: &mut [f64], t: &mut f64) {
        y.copy_from_slice(&self.yh[..self.n]);
        *t = self.tn;
    }

    /// One call of DLSODA (`istate` 1 on the first call, 2 after), advancing `(t, y)` per
    /// `task`: [`Task::Interpolate`] returns `y(tout)` from the history once the integrator has
    /// passed `tout`; [`Task::OneStepToTcrit`] takes one step, never past `tcrit`, and returns
    /// the step's end (exactly `tcrit` once within roundoff of it).
    ///
    /// On an error return `(t, y)` is the last point reached, as ODEPACK leaves it.
    pub(crate) fn call<F>(
        &mut self,
        f: &mut F,
        y: &mut [f64],
        t: &mut f64,
        tout: f64,
        task: Task,
        tcrit: f64,
    ) -> Result<(), LsodaFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        // Block E's bookkeeping (mxstep, ewset) is skipped once, right after initialization.
        let mut refresh_weights = if self.started {
            // Block D: stop tests before taking a step.
            self.nslast = self.nst;
            match task {
                Task::Interpolate => {
                    if !((self.tn - tout) * self.h < 0.0) {
                        if !self.intdy(tout, y) {
                            return Err(LsodaFailure::IllegalInput("tout outside the last step"));
                        }
                        *t = tout;
                        return Ok(());
                    }
                }
                Task::OneStepToTcrit => {
                    if (self.tn - tcrit) * self.h > 0.0 {
                        return Err(LsodaFailure::IllegalInput("tcrit behind tcur"));
                    }
                    let hmx = self.tn.abs() + self.h.abs();
                    if (self.tn - tcrit).abs() <= 100.0 * self.uround * hmx {
                        y.copy_from_slice(&self.yh[..n]);
                        *t = tcrit;
                        return Ok(());
                    }
                    let tnext = self.tn + self.h * (1.0 + 4.0 * self.uround);
                    if !((tnext - tcrit) * self.h <= 0.0) {
                        self.h = (tcrit - self.tn) * (1.0 - 4.0 * self.uround);
                        if self.jstart >= 0 {
                            self.jstart = -2;
                        }
                    }
                }
            }
            true
        } else {
            self.initialize(f, y, *t, tout, task, tcrit)?;
            false
        };

        // Block E: the stepping loop.
        loop {
            if refresh_weights {
                if self.nst - self.nslast >= self.mxstep {
                    self.current_state(y, t);
                    return Err(LsodaFailure::ExcessWork);
                }
                self.ewset();
                if !self.invert_ewt() {
                    self.current_state(y, t);
                    return Err(LsodaFailure::ZeroErrorWeight);
                }
            }
            refresh_weights = true;
            let tolsf = self.uround * vmnorm(&self.yh[..n], &self.ewt);
            if tolsf > 0.01 {
                if self.nst == 0 {
                    return Err(LsodaFailure::IllegalInput(
                        "tolerances too small for the machine precision at t0",
                    ));
                }
                self.current_state(y, t);
                return Err(LsodaFailure::ExcessAccuracy);
            }
            self.stoda(f)?;
            match self.kflag {
                -1 => {
                    self.current_state(y, t);
                    return Err(LsodaFailure::ErrorTestFailures);
                }
                -2 => {
                    self.current_state(y, t);
                    return Err(LsodaFailure::ConvergenceFailures);
                }
                _ => {}
            }
            // Block F: a successful step; complete a method switch on the next call.
            self.started = true;
            if self.meth != self.mused {
                self.tsw = self.tn;
                self.maxord = if self.meth == 2 {
                    self.mxords
                } else {
                    self.mxordn
                };
                self.jstart = -1;
            }
            match task {
                Task::Interpolate => {
                    if (self.tn - tout) * self.h < 0.0 {
                        continue;
                    }
                    if !self.intdy(tout, y) {
                        return Err(LsodaFailure::IllegalInput("tout outside the last step"));
                    }
                    *t = tout;
                    return Ok(());
                }
                Task::OneStepToTcrit => {
                    let hmx = self.tn.abs() + self.h.abs();
                    let ihit = (self.tn - tcrit).abs() <= 100.0 * self.uround * hmx;
                    y.copy_from_slice(&self.yh[..n]);
                    *t = if ihit { tcrit } else { self.tn };
                    return Ok(());
                }
            }
        }
    }

    /// Blocks A-C for istate = 1: check the inputs, evaluate f(t0, y0), load the history and
    /// the (inverted) error weights, and choose the first step.
    fn initialize<F>(
        &mut self,
        f: &mut F,
        y: &[f64],
        t: f64,
        tout: f64,
        task: Task,
        tcrit: f64,
    ) -> Result<(), LsodaFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        if tout == t {
            return Err(LsodaFailure::IllegalInput("tout = t on the first call"));
        }
        if n == 0 {
            return Err(LsodaFailure::IllegalInput("neq < 1"));
        }
        let mut h0 = self.h0;
        if (tout - t) * h0 < 0.0 {
            return Err(LsodaFailure::IllegalInput("h0 points away from tout"));
        }
        if (0..n).any(|i| self.rtol_at(i) < 0.0 || self.atol_at(i) < 0.0) {
            return Err(LsodaFailure::IllegalInput("negative rtol or atol"));
        }
        self.meth = 1;
        self.uround = f64::EPSILON;
        self.tn = t;
        self.tsw = t;
        self.maxord = self.mxordn;
        if task == Task::OneStepToTcrit {
            if (tcrit - tout) * (tout - t) < 0.0 {
                return Err(LsodaFailure::IllegalInput("tcrit before tout"));
            }
            if h0 != 0.0 && (t + h0 - tcrit) * h0 > 0.0 {
                h0 = tcrit - t;
            }
        }
        self.jstart = 0;
        self.nst = 0;
        self.nje = 0;
        self.nslast = 0;
        self.hu = 0.0;
        self.nqu = 0;
        self.mused = 0;
        self.miter = 0;
        self.ccmax = 0.3;
        self.maxcor = 3;
        self.msbp = 20;
        self.mxncf = 10;

        eval(f, t, y, &mut self.yh[n..2 * n])?;
        self.nfe = 1;
        self.yh[..n].copy_from_slice(y);
        self.nq = 1;
        self.h = 1.0;
        self.ewset();
        if !self.invert_ewt() {
            return Err(LsodaFailure::IllegalInput(
                "an error weight is not positive",
            ));
        }

        if h0 == 0.0 {
            // h0^-2 = 1/(tol w0^2) + tol |f0|^2, tol = max rtol (or max atol_i/|y_i| when rtol
            // is 0) clamped to [100 uround, 1e-3], w0 = max(|t|, |tout|).
            let tdist = (tout - t).abs();
            let w0 = t.abs().max(tout.abs());
            if tdist < 2.0 * self.uround * w0 {
                return Err(LsodaFailure::IllegalInput("tout too close to t to start"));
            }
            let mut tol = self.rtol[0];
            if self.rtol.len() > 1 {
                for &r in &self.rtol {
                    tol = tol.max(r);
                }
            }
            if tol <= 0.0 {
                for i in 0..n {
                    let atoli = self.atol_at(i);
                    if y[i].abs() != 0.0 {
                        tol = tol.max(atoli / y[i].abs());
                    }
                }
            }
            tol = tol.max(100.0 * self.uround);
            tol = tol.min(0.001);
            let mut sum = vmnorm(&self.yh[n..2 * n], &self.ewt);
            sum = 1.0 / (tol * w0 * w0) + tol * sum * sum;
            h0 = 1.0 / sum.sqrt();
            h0 = h0.min(tdist);
            h0 = h0.copysign(tout - t);
        }
        let rh = h0.abs() * self.hmxi;
        if rh > 1.0 {
            h0 /= rh;
        }
        self.h = h0;
        for v in &mut self.yh[n..2 * n] {
            *v *= h0;
        }
        Ok(())
    }

    /// DINTDY with k = 0: the history polynomial `sum_j yh[:, j] ((t - tn)/h)^j` at `t`, which
    /// must lie in the last step `[tn - hu, tn]` (up to roundoff); `false` otherwise.
    fn intdy(&self, t: f64, dky: &mut [f64]) -> bool {
        let n = self.n;
        let tp = self.tn - self.hu - 100.0 * self.uround * (self.tn + self.hu);
        if (t - tp) * (t - self.tn) > 0.0 {
            return false;
        }
        let s = (t - self.tn) / self.h;
        // c = 1 throughout for k = 0, and 1.0 * x is x.
        dky.copy_from_slice(&self.yh[(self.l - 1) * n..self.l * n]);
        for jb in 1..=self.nq {
            let col = (self.nq - jb) * n;
            for i in 0..n {
                dky[i] = self.yh[col + i] + s * dky[i];
            }
        }
        true
    }

    /// DSTODA: one step of the current method, with its error test, order and step selection
    /// and the Adams <-> BDF switching test. `kflag` reports the outcome (0, or -1 / -2 after
    /// repeated error-test / convergence failures).
    fn stoda<F>(&mut self, f: &mut F) -> Result<(), LsodaFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let told = self.tn;
        let mut ncf = 0_usize;
        let mut rh: f64;
        self.kflag = 0;
        self.lu_ok = true;
        self.jcur = false;

        match self.jstart {
            0 => {
                self.first_call_init();
                self.reset_el();
            }
            -1 => {
                // Label 100: new parameters, possibly a new method.
                self.ipup = self.miter;
                self.lmax = self.maxord + 1;
                if self.ialth == 1 {
                    self.ialth = 2;
                }
                if self.meth != self.mused {
                    cfode(self.meth, &mut self.elco, &mut self.tesco);
                    self.ialth = self.l as i32;
                    self.reset_el();
                }
                if self.h != self.hold {
                    rh = self.h / self.hold;
                    self.h = self.hold;
                    rh = rh.max(self.hmin / self.h.abs());
                    self.adjust_step_size(rh);
                }
            }
            // Only h changed (tcrit clamped it).
            -2 if self.h != self.hold => {
                rh = self.h / self.hold;
                self.h = self.hold;
                self.adjust_step_size(rh);
            }
            _ => {}
        }

        self.predict();
        let mut pnorm = vmnorm(&self.yh[..n], &self.ewt);
        let reset_rmax;
        loop {
            let (m, del) = match self.corrector(f, pnorm)? {
                Corrector::Converged { m, del } => (m, del),
                Corrector::Retry => continue,
                Corrector::Failed => {
                    if !self.corrector_failure(&mut ncf, told) {
                        self.hold = self.h;
                        self.jstart = 1;
                        return Ok(());
                    }
                    self.predict();
                    pnorm = vmnorm(&self.yh[..n], &self.ewt);
                    continue;
                }
            };

            // Label 450: the local error test.
            self.jcur = false;
            let dsm = if m == 0 {
                del / self.tesco[1 + (self.nq - 1) * 3]
            } else {
                vmnorm(&self.acor, &self.ewt) / self.tesco[1 + (self.nq - 1) * 3]
            };

            let mut rhup;
            if dsm > 1.0 {
                // Label 500: the error test failed; retract and retry smaller (and, after
                // three failures, restart at order 1 from a fresh derivative).
                self.kflag -= 1;
                self.tn = told;
                self.retract();
                self.rmax = 2.0;
                if self.h.abs() <= self.hmin * 1.00001 {
                    self.kflag = -1;
                    self.hold = self.h;
                    self.jstart = 1;
                    return Ok(());
                }
                if self.kflag <= -3 {
                    // Label 640.
                    if self.kflag == -10 {
                        self.kflag = -1;
                        self.hold = self.h;
                        self.jstart = 1;
                        return Ok(());
                    }
                    rh = 0.1;
                    rh = (self.hmin / self.h.abs()).max(rh);
                    self.h *= rh;
                    self.y.copy_from_slice(&self.yh[..n]);
                    eval(f, self.tn, &self.y, &mut self.savf)?;
                    self.nfe += 1;
                    for i in 0..n {
                        self.yh[i + n] = self.h * self.savf[i];
                    }
                    self.ipup = self.miter;
                    self.ialth = 5;
                    if self.nq != 1 {
                        self.nq = 1;
                        self.l = 2;
                        self.reset_el();
                    }
                    self.predict();
                    pnorm = vmnorm(&self.yh[..n], &self.ewt);
                    continue;
                }
                rhup = 0.0;
            } else {
                // The step is accepted: update the history.
                self.kflag = 0;
                self.nst += 1;
                self.hu = self.h;
                self.nqu = self.nq;
                self.mused = self.meth;
                for j in 0..self.l {
                    let elj = self.el[j];
                    for i in 0..n {
                        self.yh[i + j * n] += elj * self.acor[i];
                    }
                }
                self.icount -= 1;
                if self.icount < 0 {
                    let switched = if self.meth == 1 {
                        self.consider_bdf(dsm, pnorm)
                    } else {
                        self.consider_adams(dsm, pnorm)
                    };
                    if switched {
                        reset_rmax = true;
                        break;
                    }
                }
                // Label 488: no switch; the usual step and order selection every ialth steps.
                self.ialth -= 1;
                if self.ialth == 0 {
                    rhup = 0.0;
                    if self.l != self.lmax {
                        let last = (self.lmax - 1) * n;
                        for i in 0..n {
                            self.savf[i] = self.acor[i] - self.yh[i + last];
                        }
                        let dup = vmnorm(&self.savf, &self.ewt) / self.tesco[2 + (self.nq - 1) * 3];
                        let exup = 1.0 / (self.l as f64 + 1.0);
                        rhup = 1.0 / (1.4 * dup.powf(exup) + 0.0000014);
                    }
                } else {
                    // Save the correction for a possible order increase on the next step.
                    if self.ialth <= 1 && self.l != self.lmax {
                        let last = (self.lmax - 1) * n;
                        self.yh[last..last + n].copy_from_slice(&self.acor);
                    }
                    reset_rmax = false;
                    break;
                }
            }

            // Label 540: step size factors for the same order (rhsm), one lower (rhdn) and one
            // higher (rhup); the largest wins.
            let exsm = 1.0 / self.l as f64;
            let mut rhsm = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
            let mut rhdn = 0.0;
            // Read only on Adams (the 620 test), where label 550 has just set it.
            let mut pdh = 0.0;
            if self.nq != 1 {
                let top = (self.l - 1) * n;
                let ddn = vmnorm(&self.yh[top..top + n], &self.ewt) / self.tesco[(self.nq - 1) * 3];
                let exdn = 1.0 / self.nq as f64;
                rhdn = 1.0 / (1.3 * ddn.powf(exdn) + 0.0000013);
            }
            if self.meth == 1 {
                pdh = (self.h.abs() * self.pdlast).max(0.000001);
                if self.l < self.lmax {
                    rhup = rhup.min(SM1[self.l - 1] / pdh);
                }
                rhsm = rhsm.min(SM1[self.nq - 1] / pdh);
                if self.nq > 1 {
                    rhdn = rhdn.min(SM1[self.nq - 2] / pdh);
                }
                self.pdest = 0.0;
            }
            let newq;
            if rhsm >= rhup {
                if rhsm >= rhdn {
                    newq = self.nq;
                    rh = rhsm;
                } else {
                    newq = self.nq - 1;
                    rh = rhdn;
                    if self.kflag < 0 && rh > 1.0 {
                        rh = 1.0;
                    }
                }
            } else if rhup > rhdn {
                // Label 590: raise the order, seeding the new column from the correction.
                newq = self.l;
                rh = rhup;
                if rh < 1.1 {
                    self.ialth = 3;
                    reset_rmax = false;
                    break;
                }
                let r = self.el[self.l - 1] / self.l as f64;
                for i in 0..n {
                    self.yh[i + newq * n] = self.acor[i] * r;
                }
                self.nq = newq;
                self.l = self.nq + 1;
                self.reset_el();
                rh = rh.max(self.hmin / self.h.abs());
                self.adjust_step_size(rh);
                reset_rmax = true;
                break;
            } else {
                newq = self.nq - 1;
                rh = rhdn;
                if self.kflag < 0 && rh > 1.0 {
                    rh = 1.0;
                }
            }
            // Label 620: a change under 10% is not worth making, unless an Adams step is being
            // held down by its stability region.
            let bypass = self.meth == 1 && rh * pdh * 1.00001 >= SM1[newq - 1];
            if !bypass && self.kflag == 0 && rh < 1.1 {
                self.ialth = 3;
                reset_rmax = false;
                break;
            }
            if self.kflag <= -2 {
                rh = rh.min(0.2);
            }
            if newq != self.nq {
                self.nq = newq;
                self.l = self.nq + 1;
                self.reset_el();
            }
            rh = rh.max(self.hmin / self.h.abs());
            self.adjust_step_size(rh);
            if self.kflag == 0 {
                reset_rmax = true;
                break;
            }
            // An error-test failure: redo the step at the new size.
            self.predict();
            pnorm = vmnorm(&self.yh[..n], &self.ewt);
        }

        if reset_rmax {
            self.rmax = 10.0;
        }
        // Label 700.
        let r = 1.0 / self.tesco[1 + (self.nqu - 1) * 3];
        for a in &mut self.acor {
            *a *= r;
        }
        self.hold = self.h;
        self.jstart = 1;
        Ok(())
    }

    /// Labels 470-478: on Adams, test whether BDF could have taken a step at least `ratio` (5)
    /// times longer; if so switch (to order min(nq, mxords)) and return `true`.
    fn consider_bdf(&mut self, dsm: f64, pnorm: f64) -> bool {
        let n = self.n;
        let clean = dsm > 100.0 * pnorm * self.uround && self.pdest != 0.0;
        // Above order 5 the problem is taken to be nonstiff; with roundoff-polluted estimates,
        // switch only if the last step was held down by stability (irflag), doubling h.
        if self.nq > 5 || (!clean && !self.irflag) {
            return false;
        }
        let (rh2, nqm2) = if clean {
            let exsm = 1.0 / self.l as f64;
            let mut rh1 = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
            let mut rh1it = 2.0 * rh1;
            let pdh = self.pdlast * self.h.abs();
            if pdh * rh1 > 0.00001 {
                rh1it = SM1[self.nq - 1] / pdh;
            }
            rh1 = rh1.min(rh1it);
            let (rh2, nqm2) = if self.nq > self.mxords {
                let lm2 = self.mxords + 1;
                let exm2 = 1.0 / lm2 as f64;
                let dm2 =
                    vmnorm(&self.yh[lm2 * n..(lm2 + 1) * n], &self.ewt) / self.cm2[self.mxords - 1];
                (1.0 / (1.2 * dm2.powf(exm2) + 0.0000012), self.mxords)
            } else {
                let dm2 = dsm * (self.cm1[self.nq - 1] / self.cm2[self.nq - 1]);
                (1.0 / (1.2 * dm2.powf(exsm) + 0.0000012), self.nq)
            };
            // Written as C tests it (switch when rh2 >= ratio*rh1), so a NaN does not switch.
            if !(rh2 >= self.ratio * rh1) {
                return false;
            }
            (rh2, nqm2)
        } else {
            (2.0, self.nq.min(self.mxords))
        };
        // Label 478.
        self.icount = 20;
        self.meth = 2;
        self.miter = self.jtyp;
        self.pdlast = 0.0;
        self.nq = nqm2;
        self.l = self.nq + 1;
        let rh = rh2.max(self.hmin / self.h.abs());
        self.adjust_step_size(rh);
        true
    }

    /// Labels 480-486: on BDF, test whether Adams could take a step at least as long (5/ratio)
    /// without its error estimate sinking into roundoff; if so switch and return `true`.
    fn consider_adams(&mut self, dsm: f64, pnorm: f64) -> bool {
        let n = self.n;
        let exsm = 1.0 / self.l as f64;
        let (mut rh1, nqm1, exm1, mut dm1);
        if self.mxordn >= self.nq {
            dm1 = dsm * (self.cm2[self.nq - 1] / self.cm1[self.nq - 1]);
            rh1 = 1.0 / (1.2 * dm1.powf(exsm) + 0.0000012);
            nqm1 = self.nq;
            exm1 = exsm;
        } else {
            nqm1 = self.mxordn;
            let lm1 = self.mxordn + 1;
            exm1 = 1.0 / lm1 as f64;
            dm1 = vmnorm(&self.yh[lm1 * n..(lm1 + 1) * n], &self.ewt) / self.cm1[self.mxordn - 1];
            rh1 = 1.0 / (1.2 * dm1.powf(exm1) + 0.0000012);
        }
        let mut rh1it = 2.0 * rh1;
        let pdh = self.pdnorm * self.h.abs();
        if pdh * rh1 > 0.00001 {
            rh1it = SM1[nqm1 - 1] / pdh;
        }
        rh1 = rh1.min(rh1it);
        let rh2 = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
        // Both tests as C writes them (switch when they hold), so a NaN does not switch.
        if !(rh1 * self.ratio >= 5.0 * rh2) {
            return false;
        }
        // C: dm1 = pow(fmax(0.001, rh1), exm1) * dm1 (one product, commutative).
        dm1 *= rh1.max(0.001).powf(exm1);
        if !(dm1 > 1000.0 * self.uround * pnorm) {
            return false;
        }
        self.icount = 20;
        self.meth = 1;
        self.miter = 0;
        self.pdlast = 0.0;
        self.nq = nqm1;
        self.l = self.nq + 1;
        let rh = rh1.max(self.hmin / self.h.abs());
        self.adjust_step_size(rh);
        true
    }

    /// `stoda_first_call_init`: the jstart = 0 set-up (Adams, order 1) and both families'
    /// coefficients, BDF's first so `cm2` is read before the Adams `cfode` overwrites them.
    fn first_call_init(&mut self) {
        self.lmax = self.maxord + 1;
        self.nq = 1;
        self.l = 2;
        self.ialth = 2;
        self.rmax = 10000.0;
        self.rc = 0.0;
        self.el0 = 1.0;
        self.crate_ = 0.7;
        self.hold = self.h;
        self.nslp = 0;
        self.ipup = self.miter;
        self.icount = 20;
        self.irflag = false;
        self.pdest = 0.0;
        self.pdlast = 0.0;
        self.ratio = 5.0;
        cfode(2, &mut self.elco, &mut self.tesco);
        for i in 0..5 {
            self.cm2[i] = self.tesco[1 + i * 3] * self.elco[i + 1 + i * 13];
        }
        cfode(1, &mut self.elco, &mut self.tesco);
        for i in 0..12 {
            self.cm1[i] = self.tesco[1 + i * 3] * self.elco[i + 1 + i * 13];
        }
    }

    /// Label 150 (`stoda_reset`): load `el` for order `nq` and the constants that follow it.
    fn reset_el(&mut self) {
        for i in 0..self.l {
            self.el[i] = self.elco[i + (self.nq - 1) * 13];
        }
        self.nqnyh = self.nq * self.n;
        self.rc = self.rc * self.el[0] / self.el0;
        self.el0 = self.el[0];
        self.conit = 0.5 / (self.nq + 2) as f64;
    }

    /// Labels 175-178 (`stoda_adjust_step_size`): bound the ratio `rh` by `rmax` and hmax (and,
    /// on Adams, by the stability region, setting irflag), then rescale the history to the new h.
    fn adjust_step_size(&mut self, rh: f64) {
        let mut rh = rh.min(self.rmax);
        rh /= 1.0_f64.max(self.h.abs() * self.hmxi * rh);
        if self.meth != 2 {
            self.irflag = false;
            let pdh = (self.h.abs() * self.pdlast).max(0.000001);
            if rh * pdh * 1.00001 >= SM1[self.nq - 1] {
                rh = SM1[self.nq - 1] / pdh;
                self.irflag = true;
            }
        }
        let n = self.n;
        let mut r = 1.0;
        for j in 1..self.l {
            r *= rh;
            for v in &mut self.yh[j * n..(j + 1) * n] {
                *v *= r;
            }
        }
        self.h *= rh;
        self.rc *= rh;
        self.ialth = self.l as i32;
    }

    /// Label 200: flag a Jacobian update when h*el0 moved by more than ccmax or msbp steps
    /// passed, advance tn, and predict by multiplying the history by the Pascal triangle.
    fn predict(&mut self) {
        if (self.rc - 1.0).abs() > self.ccmax {
            self.ipup = self.miter;
        }
        if self.nst >= self.nslp + self.msbp {
            self.ipup = self.miter;
        }
        self.tn += self.h;
        let n = self.n;
        let mut i1 = self.nqnyh;
        for _ in 0..self.nq {
            i1 -= n;
            for i in i1..self.nqnyh {
                self.yh[i] += self.yh[i + n];
            }
        }
    }

    /// Undo [`Lsoda::predict`]'s Pascal-triangle multiply (labels 430-445, 500-515).
    fn retract(&mut self) {
        let n = self.n;
        let mut i1 = self.nqnyh;
        for _ in 0..self.nq {
            i1 -= n;
            for i in i1..self.nqnyh {
                self.yh[i] -= self.yh[i + n];
            }
        }
    }

    /// Labels 220-410: the corrector. Functional iteration on Adams (miter = 0), the chord
    /// method with `P = I - h el0 J` on BDF, at most `maxcor` (3) iterations, accumulating the
    /// correction in `acor` and, on convergence, the Lipschitz estimate `pdest`/`pdlast`.
    fn corrector<F>(&mut self, f: &mut F, pnorm: f64) -> Result<Corrector, LsodaFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        let mut m = 0_usize;
        let mut rate = 0.0_f64;
        let mut del;
        let mut delp = 0.0_f64;
        self.y.copy_from_slice(&self.yh[..n]);
        eval(f, self.tn, &self.y, &mut self.savf)?;
        self.nfe += 1;
        if self.ipup > 0 {
            self.prja(f)?;
            self.ipup = 0;
            self.rc = 1.0;
            self.nslp = self.nst;
            self.crate_ = 0.7;
            if !self.lu_ok {
                return Ok(Corrector::Failed);
            }
        }
        self.acor.fill(0.0);
        loop {
            if self.miter == 0 {
                for i in 0..n {
                    self.savf[i] = self.h * self.savf[i] - self.yh[i + n];
                    self.y[i] = self.savf[i] - self.acor[i];
                }
                del = vmnorm(&self.y, &self.ewt);
                for i in 0..n {
                    self.y[i] = self.yh[i] + self.el[0] * self.savf[i];
                    self.acor[i] = self.savf[i];
                }
            } else {
                for i in 0..n {
                    self.y[i] = self.h * self.savf[i] - (self.yh[i + n] + self.acor[i]);
                }
                lu_solve(&self.wm, &self.ipvt, n, &mut self.y);
                del = vmnorm(&self.y, &self.ewt);
                for i in 0..n {
                    self.acor[i] += self.y[i];
                    self.y[i] = self.yh[i] + self.el[0] * self.acor[i];
                }
            }
            // A correction at roundoff level has converged without a new rate estimate; else
            // Adams takes at least two iterations so the Lipschitz estimate exists.
            if del <= 100.0 * pnorm * self.uround {
                break;
            }
            if m != 0 || self.meth != 1 {
                if m != 0 {
                    let mut rm = 1024.0;
                    if del <= 1024.0 * delp {
                        rm = del / delp;
                    }
                    rate = rate.max(rm);
                    self.crate_ = (0.2 * self.crate_).max(rm);
                }
                let dcon = del * 1.0_f64.min(1.5 * self.crate_)
                    / (self.tesco[1 + (self.nq - 1) * 3] * self.conit);
                if dcon <= 1.0 {
                    self.pdest = self.pdest.max(rate / (self.h * self.el[0]).abs());
                    if self.pdest != 0.0 {
                        self.pdlast = self.pdest;
                    }
                    break;
                }
            }
            m += 1;
            if m == self.maxcor || (m >= 2 && del > 2.0 * delp) {
                if self.miter != 0 && !self.jcur {
                    self.ipup = self.miter;
                    return Ok(Corrector::Retry);
                }
                return Ok(Corrector::Failed);
            }
            delp = del;
            eval(f, self.tn, &self.y, &mut self.savf)?;
            self.nfe += 1;
        }
        self.jcur = false;
        Ok(Corrector::Converged { m, del })
    }

    /// Labels 430-445: after a corrector failure retract, and either give up (kflag = -2 at
    /// hmin or after mxncf failures) or retry at a quarter of the step with a fresh Jacobian.
    fn corrector_failure(&mut self, ncf: &mut usize, told: f64) -> bool {
        *ncf += 1;
        self.rmax = 2.0;
        self.tn = told;
        self.retract();
        if self.h.abs() <= self.hmin * 1.00001 || *ncf == self.mxncf {
            self.kflag = -2;
            return false;
        }
        self.ipup = self.miter;
        let rh = 0.25_f64.max(self.hmin / self.h.abs());
        self.adjust_step_size(rh);
        true
    }

    /// DPRJA (full matrix): `J` from the user (miter = 1) or by forward differences (miter = 2,
    /// `n` evaluations with increments `max(sqrt(uround)|y_j|, r0/ewt_j)`), `pdnorm = ||J||`,
    /// then `P = I - h el0 J` factored in place.
    fn prja<F>(&mut self, f: &mut F) -> Result<(), LsodaFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = self.n;
        self.nje += 1;
        self.lu_ok = true;
        self.jcur = true;
        let hl0 = self.h * self.el0;
        match (self.miter, self.jac) {
            (1, Some(jac)) => {
                let rows = jac(self.tn, &self.y);
                if rows.len() != n || rows.iter().any(|row| row.len() != n) {
                    return Err(LsodaFailure::Callback(StepFailure::RuntimeError(
                        JAC_WRONG_SHAPE,
                    )));
                }
                let con = -hl0;
                for (i, row) in rows.iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        self.wm[i + j * n] = value * con;
                    }
                }
            }
            _ => {
                let fac = vmnorm(&self.savf, &self.ewt);
                let mut r0 = 1000.0 * self.h.abs() * self.uround * n as f64 * fac;
                if r0 == 0.0 {
                    r0 = 1.0;
                }
                let srur = self.uround.sqrt();
                for j in 0..n {
                    let yj = self.y[j];
                    let r = (srur * yj.abs()).max(r0 / self.ewt[j]);
                    self.y[j] += r;
                    let fac = -hl0 / r;
                    eval(f, self.tn, &self.y, &mut self.acor)?;
                    for i in 0..n {
                        self.wm[i + j * n] = (self.acor[i] - self.savf[i]) * fac;
                    }
                    self.y[j] = yj;
                }
                self.nfe += n;
            }
        }
        self.pdnorm = fnorm(n, &self.wm, &self.ewt) / hl0.abs();
        for i in 0..n {
            self.wm[i + i * n] += 1.0;
        }
        self.lu_ok = lu_factor(&mut self.wm, &mut self.ipvt, n);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DCFODE against the closed forms it tabulates: BDF order 1 is backward Euler
    /// (el = [1, 1]), BDF order 2 has l(x) = (x+1)(x+2)/3 (el = [2/3, 1, 1/3]), Adams order 1
    /// is el = [1, 1] with error constant 1/2 (tesco(2,1) = 2), Adams order 2 (trapezoidal)
    /// el = [1/2, 1, 1/2].
    #[test]
    fn cfode_reproduces_the_low_order_formulas() {
        let mut elco = [0.0; 156];
        let mut tesco = [0.0; 36];
        cfode(2, &mut elco, &mut tesco);
        assert_eq!(&elco[0..2], &[1.0, 1.0]);
        assert_eq!(elco[13], 2.0 / 3.0);
        assert_eq!(elco[14], 1.0);
        assert_eq!(elco[15], 1.0 / 3.0);
        cfode(1, &mut elco, &mut tesco);
        assert_eq!(&elco[0..2], &[1.0, 1.0]);
        assert_eq!(tesco[1], 2.0);
        assert_eq!(&elco[13..16], &[0.5, 1.0, 0.5]);
        // Negative arm: the BDF and Adams order-2 leading coefficients must differ, so a cfode
        // that ignored `meth` could not pass both halves.
        assert_ne!(2.0 / 3.0, elco[13]);
    }

    /// The LU port against a hand-checked 3x3 with a row interchange at every column, and a
    /// singular matrix (zero pivot reported, not divided by).
    #[test]
    fn lu_factor_and_solve_with_pivoting() {
        // Column-major A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]; A x = b with x = [1, -2, 3].
        let mut a = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
        let mut ipvt = vec![0; 3];
        assert!(lu_factor(&mut a, &mut ipvt, 3));
        assert_eq!(ipvt, vec![2, 2, 2]);
        let mut b = vec![6.0, 12.0, 21.0];
        lu_solve(&a, &ipvt, 3, &mut b);
        for (got, want) in b.iter().zip([1.0, -2.0, 3.0]) {
            assert!((got - want).abs() < 1e-13, "{b:?}");
        }
        let mut singular = vec![1.0, 2.0, 2.0, 4.0];
        let mut ipvt = vec![0; 2];
        assert!(!lu_factor(&mut singular, &mut ipvt, 2));
    }

    /// vmnorm/fnorm follow C's fmax: a NaN component does not poison the norm.
    #[test]
    fn norms_follow_fmax() {
        assert_eq!(vmnorm(&[1.0, f64::NAN, -3.0], &[1.0, 1.0, 2.0]), 6.0);
        // [[1, -2], [3, 4]] with w = [1, 2]: rows 1*(1/1 + 2/2) = 2 and 2*(3/1 + 4/2) = 10.
        assert_eq!(fnorm(2, &[1.0, 3.0, -2.0, 4.0], &[1.0, 2.0]), 10.0);
    }
}
