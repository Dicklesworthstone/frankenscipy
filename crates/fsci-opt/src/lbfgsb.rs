//! SciPy 1.17.1's L-BFGS-B: a transcription of `scipy/optimize/__lbfgsb.c` (SciPy's C translation
//! of Nocedal & Morales' L-BFGS-B 3.0, 2011) driven the way `_minimize_lbfgsb` drives `setulb`:
//! the generalized Cauchy point, the direct primal subspace minimization with the 3.0
//! backtracking to the box, the compact limited-memory representation, and the MINPACK-2
//! `dcsrch` line search of the C file (see [`Minpack::C`]).
//!
//! The reverse-communication loop is unrolled into direct calls of [`LbfgsbObjective`], keeping
//! every counter and branch of `mainlb` and every check `_minimize_lbfgsb` makes between calls.
//! BLAS and LAPACK calls are the reference routines' operation order (`ddot` in order, `dpotrf`
//! as OpenBLAS's unblocked `potf2`, `dtrtrs` as reference `dtrsm`); `dnrm2` is OpenBLAS's x87
//! extended-precision sum of squares, emulated in double-double. std-only.

use crate::bfgs::{Dcsrch, Minpack, Task};

/// `np.finfo(np.float64).eps`, the `epsmach` of `__lbfgsb.c`.
const EPSMACH: f64 = 2.220_446_049_250_313e-16;

/// The objective as `_minimize_lbfgsb` sees it through SciPy's `ScalarFunction`.
pub(crate) trait LbfgsbObjective {
    type Error;
    /// `sf.fun_and_grad(x)`.
    fn fun_and_grad(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), Self::Error>;
    /// `sf.nfev`: function evaluations so far, finite-difference ones included.
    fn nfev(&self) -> usize;
    /// `_call_callback_maybe_halt` at a new iterate; `true` halts.
    fn callback(&mut self, x: &[f64], f: f64) -> bool;
}

/// `_minimize_lbfgsb`'s options after its own conversions.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LbfgsbParams {
    /// `maxcor`.
    pub m: usize,
    /// `ftol / eps`.
    pub factr: f64,
    /// `gtol`.
    pub pgtol: f64,
    pub maxfun: usize,
    pub maxiter: usize,
    pub maxls: usize,
}

/// `setulb`'s final task, as `_minimize_lbfgsb` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LbfgsbStop {
    /// `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`.
    ProjectedGradient,
    /// `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`.
    RelativeReduction,
    /// `STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT`.
    MaxFun,
    /// `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`.
    MaxIter,
    /// `STOP: CALLBACK REQUESTED HALT`.
    Callback,
    /// `ABNORMAL: ` — the line search failed with no correction pairs left to discard.
    Abnormal,
}

impl LbfgsbStop {
    /// `status_messages[task[0]] + ": " + task_messages[task[1]]`.
    pub(crate) const fn message(self) -> &'static str {
        match self {
            Self::ProjectedGradient => "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            Self::RelativeReduction => "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
            Self::MaxFun => "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
            Self::MaxIter => "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            Self::Callback => "STOP: CALLBACK REQUESTED HALT",
            Self::Abnormal => "ABNORMAL: ",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LbfgsbOutcome {
    pub x: Vec<f64>,
    pub fun: f64,
    pub jac: Vec<f64>,
    pub nit: usize,
    pub stop: LbfgsbStop,
    /// `_minimize_lbfgsb`'s warnflag: 0 converged, 1 maxfun / maxiter, 2 otherwise.
    pub warnflag: u8,
}

/// `ddot` with unit strides, in order.
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (a, b) in x.iter().zip(y) {
        sum += b * a;
    }
    sum
}

/// OpenBLAS's x86-64 `dnrm2` accumulates the sum of squares and takes the root in x87 extended
/// precision, then rounds once. Emulated with a double-double sum (error-free products and sums)
/// and one Newton correction of the root, which agrees with the extended result after rounding.
fn dnrm2(x: &[f64]) -> f64 {
    let (mut hi, mut lo) = (0.0_f64, 0.0_f64);
    for &v in x {
        let p = v * v;
        let p_err = v.mul_add(v, -p);
        let s = hi + p;
        let bb = s - hi;
        let s_err = (hi - (s - bb)) + (p - bb);
        hi = s;
        lo += s_err + p_err;
    }
    let sum = hi + lo;
    let lo = lo - (sum - hi);
    if sum <= 0.0 || !sum.is_finite() {
        return sum.sqrt();
    }
    let root = sum.sqrt();
    let residual = (-root).mul_add(root, sum) + lo;
    root + residual / (2.0 * root)
}

/// `y += a·x`.
fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// OpenBLAS `potf2_U` on the `n × n` upper triangle of the column-major `a` (leading dimension
/// `lda`): `a(j,j) −= dot`, root, then row `j` −= `A(0..j, k)ᵀ A(0..j, j)` and scaled by
/// `1/a(j,j)`. Returns LAPACK's `info` (0, or the 1-based order of the failing minor).
fn dpotrf_upper(a: &mut [f64], lda: usize, n: usize) -> i32 {
    for j in 0..n {
        let col_j = j * lda;
        let ajj = a[col_j + j] - ddot(&a[col_j..col_j + j], &a[col_j..col_j + j]);
        if ajj <= 0.0 {
            a[col_j + j] = ajj;
            return i32::try_from(j + 1).unwrap_or(i32::MAX);
        }
        let ajj = ajj.sqrt();
        a[col_j + j] = ajj;
        for k in j + 1..n {
            let col_k = k * lda;
            let temp = ddot(&a[col_k..col_k + j], &a[col_j..col_j + j]);
            a[col_k + j] -= temp;
        }
        let scale = 1.0 / ajj;
        for k in j + 1..n {
            a[k * lda + j] *= scale;
        }
    }
    0
}

/// `dtrtrs('U', trans, 'N', n, nrhs, A, lda, B, ldb)`: the singularity check, then each
/// right-hand side solved as OpenBLAS's `trsv` does (`transpose`: forward substitution with
/// `Aᵀ`, subtracting one finished dot product per entry, `trsv_L.c`; else back substitution
/// with `A` in axpy form, `trsv_U.c`).
fn dtrtrs_upper(
    transpose: bool,
    a: &[f64],
    lda: usize,
    n: usize,
    nrhs: usize,
    b: &mut [f64],
    ldb: usize,
) -> i32 {
    for i in 0..n {
        if a[i + i * lda] == 0.0 {
            return i32::try_from(i + 1).unwrap_or(i32::MAX);
        }
    }
    for j in 0..nrhs {
        let bj = j * ldb;
        if transpose {
            for i in 0..n {
                if i > 0 {
                    let dot = ddot(&a[i * lda..i * lda + i], &b[bj..bj + i]);
                    b[bj + i] -= dot;
                }
                b[bj + i] /= a[i + i * lda];
            }
        } else {
            for k in (0..n).rev() {
                b[bj + k] /= a[k + k * lda];
                let bk = -b[bj + k];
                for i in 0..k {
                    b[bj + i] += bk * a[i + k * lda];
                }
            }
        }
    }
    0
}

/// `active`: project `x` into the box and classify the variables. Returns `(cnstnd, boxed)`.
fn active(l: &[f64], u: &[f64], nbd: &[u8], x: &mut [f64], iwhere: &mut [i32]) -> (bool, bool) {
    let n = x.len();
    let mut cnstnd = false;
    let mut boxed = true;
    for i in 0..n {
        // Strict inner tests, as in C: a -0.0 on a 0.0 bound keeps its sign.
        if nbd[i] > 0 {
            if nbd[i] <= 2 && x[i] <= l[i] {
                if x[i] < l[i] {
                    x[i] = l[i];
                }
            } else if nbd[i] >= 2 && x[i] >= u[i] && x[i] > u[i] {
                x[i] = u[i];
            }
        }
    }
    for i in 0..n {
        if nbd[i] != 2 {
            boxed = false;
        }
        if nbd[i] == 0 {
            iwhere[i] = -1;
        } else {
            cnstnd = true;
            iwhere[i] = if nbd[i] == 2 && u[i] - l[i] <= 0.0 {
                3
            } else {
                0
            };
        }
    }
    (cnstnd, boxed)
}

/// `projgr`: the infinity norm of the projected gradient (NaN as soon as a component is NaN).
fn projgr(l: &[f64], u: &[f64], nbd: &[u8], x: &[f64], g: &[f64]) -> f64 {
    let mut sbgnrm = 0.0_f64;
    for i in 0..x.len() {
        let mut gi = g[i];
        if gi.is_nan() {
            return gi;
        }
        if nbd[i] != 0 {
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    gi = (x[i] - u[i]).max(gi);
                }
            } else if nbd[i] <= 2 {
                gi = (x[i] - l[i]).min(gi);
            }
        }
        sbgnrm = sbgnrm.max(gi.abs());
    }
    sbgnrm
}

/// The limited-memory BFGS matrix: `ws`, `wy` are `n × m` (column `k` at `k·n`), `sy`, `ss`,
/// `wt` are `m × m`, `wn`, `wn1` (`snd`) are `2m × 2m`, all column-major as in the C file.
struct Memory {
    n: usize,
    m: usize,
    ws: Vec<f64>,
    wy: Vec<f64>,
    sy: Vec<f64>,
    ss: Vec<f64>,
    wt: Vec<f64>,
    wn: Vec<f64>,
    wn1: Vec<f64>,
    theta: f64,
    col: usize,
    head: usize,
    itail: usize,
    iupdat: usize,
    updatd: bool,
}

impl Memory {
    fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m,
            ws: vec![0.0; m * n],
            wy: vec![0.0; m * n],
            sy: vec![0.0; m * m],
            ss: vec![0.0; m * m],
            wt: vec![0.0; m * m],
            wn: vec![0.0; 4 * m * m],
            wn1: vec![0.0; 4 * m * m],
            theta: 1.0,
            col: 0,
            head: 0,
            itail: 0,
            iupdat: 0,
            updatd: false,
        }
    }

    /// Discard the corrections (`col = head = 0`, `theta = 1`, `iupdat = 0`, not updated).
    fn reset(&mut self) {
        self.col = 0;
        self.head = 0;
        self.theta = 1.0;
        self.iupdat = 0;
        self.updatd = false;
    }

    /// `bmv`: `p = M·v` for the `2col × 2col` middle matrix, via the `wt` factor.
    fn bmv(&self, v: &[f64], p: &mut [f64]) -> i32 {
        let (m, col, sy) = (self.m, self.col, &self.sy);
        if col == 0 {
            return 0;
        }
        p[col] = v[col];
        for i in 1..col {
            let i2 = col + i;
            let mut ssum = 0.0;
            for k in 0..i {
                ssum += sy[i + m * k] * v[k] / sy[k + m * k];
            }
            p[i2] = v[i2] + ssum;
        }
        let info = dtrtrs_upper(true, &self.wt, m, col, 1, &mut p[col..2 * col], col);
        if info != 0 {
            return info;
        }
        for i in 0..col {
            p[i] = v[i] / sy[i + m * i].sqrt();
        }
        let info = dtrtrs_upper(false, &self.wt, m, col, 1, &mut p[col..2 * col], col);
        if info != 0 {
            return info;
        }
        for i in 0..col {
            p[i] = -p[i] / sy[i + m * i].sqrt();
        }
        for i in 0..col {
            let mut ssum = 0.0;
            for k in i + 1..col {
                ssum += sy[k + m * i] * p[col + k] / sy[i + m * i];
            }
            p[i] += ssum;
        }
        0
    }

    /// `formt`: `T = θ·SᵀS + L·D⁻¹·Lᵀ` into `wt`, factored `J·Jᵀ` by `dpotrf`.
    fn formt(&mut self) -> i32 {
        let (m, col, theta) = (self.m, self.col, self.theta);
        for j in 0..col {
            self.wt[m * j] = theta * self.ss[m * j];
        }
        for i in 1..col {
            for j in i..col {
                let k1 = i.min(j);
                let mut ddum = 0.0;
                for k in 0..k1 {
                    ddum += (self.sy[i + m * k] * self.sy[j + m * k]) / self.sy[k + m * k];
                }
                self.wt[i + m * j] = ddum + theta * self.ss[i + m * j];
            }
        }
        if dpotrf_upper(&mut self.wt, m, col) != 0 {
            -3
        } else {
            0
        }
    }

    /// `matupd`: store the new pair `(d, r)` and update `sy`, `ss`, `theta`.
    fn matupd(&mut self, d: &[f64], r: &[f64], rr: f64, dr: f64, stp: f64, dtd: f64) {
        let (n, m) = (self.n, self.m);
        if self.iupdat <= m {
            self.col = self.iupdat;
            self.itail = (self.head + self.iupdat - 1) % m;
        } else {
            self.itail = (self.itail + 1) % m;
            self.head = (self.head + 1) % m;
        }
        let itail = self.itail;
        self.ws[itail * n..itail * n + n].copy_from_slice(d);
        self.wy[itail * n..itail * n + n].copy_from_slice(r);
        self.theta = rr / dr;
        let col = self.col;
        if self.iupdat > m {
            // Shift the old information out: column j of the upper `ss` and the lower `sy`
            // moves up and left by one.
            for j in 1..col {
                let src = 1 + m * j;
                self.ss.copy_within(src..src + j, m * (j - 1));
                let src = j + m * j;
                self.sy
                    .copy_within(src..src + col - j, (j - 1) + m * (j - 1));
            }
        }
        let mut pointr = self.head;
        for j in 0..col - 1 {
            self.sy[col - 1 + m * j] = ddot(d, &self.wy[pointr * n..pointr * n + n]);
            self.ss[j + m * (col - 1)] = ddot(&self.ws[pointr * n..pointr * n + n], d);
            pointr = (pointr + 1) % m;
        }
        self.ss[(col - 1) + m * (col - 1)] = if stp == 1.0 { dtd } else { stp * stp * dtd };
        self.sy[(col - 1) + m * (col - 1)] = dr;
    }

    /// `formk`: the `2col × 2col` matrix of the subspace problem, factored into `wn`.
    #[allow(clippy::too_many_lines)]
    fn formk(
        &mut self,
        nsub: usize,
        ind: &[usize],
        nenter: usize,
        ileave: usize,
        indx2: &[usize],
    ) -> i32 {
        let (n, m, col, head, theta) = (self.n, self.m, self.col, self.head, self.theta);
        let m2 = 2 * m;
        let (ws, wy) = (&self.ws, &self.wy);
        let wn1 = &mut self.wn1;
        // The new rows and columns are formed first when the matrix was just updated; the
        // entering/leaving corrections below then cover the first `upcl` of them.
        let upcl = if self.updatd {
            if self.iupdat > m {
                // Shift the old part of `wn1` up and left by one row and column.
                for jy in 0..m - 1 {
                    let js = m + jy;
                    let len = m - (jy + 1);
                    let src = (jy + 1) + m2 * (jy + 1);
                    wn1.copy_within(src..src + len, jy + m2 * jy);
                    let src = (js + 1) + m2 * (js + 1);
                    wn1.copy_within(src..src + len, js + m2 * js);
                    let src = (m + 1) + m2 * (jy + 1);
                    wn1.copy_within(src..src + m - 1, m + m2 * jy);
                }
            }
            let (pbegin, pend, dbegin, dend) = (0, nsub, nsub, n);
            let iy = col - 1;
            let is = m + col - 1;
            let mut ipntr = head + col - 1;
            if ipntr >= m {
                ipntr -= m;
            }
            let mut jpntr = head;
            for jy in 0..col {
                let js = m + jy;
                let (mut temp1, mut temp2, mut temp3) = (0.0, 0.0, 0.0);
                for &k1 in &ind[pbegin..pend] {
                    temp1 += wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                for &k1 in &ind[dbegin..dend] {
                    temp2 += ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
                    temp3 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                wn1[iy + m2 * jy] = temp1;
                wn1[is + m2 * js] = temp2;
                wn1[is + m2 * jy] = temp3;
                jpntr = (jpntr + 1) % m;
            }
            let jy = col - 1;
            let mut jpntr = head + col - 1;
            if jpntr >= m {
                jpntr -= m;
            }
            let mut ipntr = head;
            for i in 0..col {
                let is = m + i;
                let mut temp3 = 0.0;
                for &k1 in &ind[pbegin..pend] {
                    temp3 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                ipntr = (ipntr + 1) % m;
                wn1[is + m2 * jy] = temp3;
            }
            col - 1
        } else {
            col
        };
        let mut ipntr = head;
        for iy in 0..upcl {
            let is = m + iy;
            let mut jpntr = head;
            for jy in 0..=iy {
                let js = m + jy;
                let (mut temp1, mut temp2, mut temp3, mut temp4) = (0.0, 0.0, 0.0, 0.0);
                for &k1 in &indx2[..nenter] {
                    temp1 += wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                    temp2 += ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
                }
                for &k1 in &indx2[ileave..n] {
                    temp3 += wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                    temp4 += ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
                }
                wn1[iy + m2 * jy] = wn1[iy + m2 * jy] + temp1 - temp3;
                wn1[is + m2 * js] = wn1[is + m2 * js] - temp2 + temp4;
                jpntr = (jpntr + 1) % m;
            }
            ipntr = (ipntr + 1) % m;
        }
        let mut ipntr = head;
        for is in m..m + upcl {
            let mut jpntr = head;
            for jy in 0..upcl {
                let (mut temp1, mut temp3) = (0.0, 0.0);
                for &k1 in &indx2[..nenter] {
                    temp1 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                for &k1 in &indx2[ileave..n] {
                    temp3 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                if is <= jy + m {
                    wn1[is + m2 * jy] = wn1[is + m2 * jy] + temp1 - temp3;
                } else {
                    wn1[is + m2 * jy] = wn1[is + m2 * jy] - temp1 + temp3;
                }
                jpntr = (jpntr + 1) % m;
            }
            ipntr = (ipntr + 1) % m;
        }
        let wn = &mut self.wn;
        for iy in 0..col {
            let is = col + iy;
            let is1 = m + iy;
            for jy in 0..=iy {
                let js = col + jy;
                let js1 = m + jy;
                wn[jy + m2 * iy] = wn1[iy + m2 * jy] / theta;
                wn[js + m2 * is] = wn1[is1 + m2 * js1] * theta;
            }
            for jy in 0..iy {
                wn[jy + m2 * is] = -wn1[is1 + m2 * jy];
            }
            for jy in iy..col {
                wn[jy + m2 * is] = wn1[is1 + m2 * jy];
            }
            wn[iy + m2 * iy] += self.sy[iy + m * iy];
        }
        if dpotrf_upper(wn, m2, col) != 0 {
            return -1;
        }
        let col2 = 2 * col;
        {
            let (a, b) = wn.split_at_mut(m2 * col);
            dtrtrs_upper(true, a, m2, col, col, b, m2);
        }
        for is in col..col2 {
            for js in is..col2 {
                let dot = ddot(&wn[m2 * is..m2 * is + col], &wn[m2 * js..m2 * js + col]);
                wn[is + m2 * js] += dot;
            }
        }
        if dpotrf_upper(&mut wn[col + m2 * col..], m2, col) != 0 {
            return -2;
        }
        0
    }
}

/// `hpsolb`: heap-sort step over `t[0..=n]` (1-based heap arithmetic as in the C file). With
/// `iheap == 0` the array is first arranged into a heap; then the least element is moved to
/// `t[n]` and the rest re-heaped.
fn hpsolb(n: usize, t: &mut [f64], iorder: &mut [usize], iheap: usize) {
    if iheap == 0 {
        for k in 2..=n + 1 {
            let ddum = t[k - 1];
            let indxin = iorder[k - 1];
            let mut i = k;
            while i > 1 {
                let j = i / 2;
                if ddum < t[j - 1] {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
        }
    }
    if n > 0 {
        let mut i = 1;
        let out = t[0];
        let indxout = iorder[0];
        let ddum = t[n];
        let indxin = iorder[n];
        loop {
            let mut j = i + i;
            if j <= n {
                if t[j] < t[j - 1] {
                    j += 1;
                }
                if t[j - 1] < ddum {
                    t[i - 1] = t[j - 1];
                    iorder[i - 1] = iorder[j - 1];
                    i = j;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        t[i - 1] = ddum;
        iorder[i - 1] = indxin;
        t[n] = out;
        iorder[n] = indxout;
    }
}

/// The `8m` scratch vector `wa` of `mainlb`, as the four `2m` pieces `cauchy` names `p`, `c`,
/// `wbp`, `v`; `cmprlb` and `subsm` reuse `p`.
struct Scratch {
    p: Vec<f64>,
    c: Vec<f64>,
    wbp: Vec<f64>,
    v: Vec<f64>,
}

/// `cauchy`: the generalized Cauchy point `xcp` along the projected steepest descent path.
/// Returns `info`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn cauchy(
    mem: &Memory,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[u8],
    g: &[f64],
    iorder: &mut [usize],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    wa: &mut Scratch,
    sbgnrm: f64,
) -> i32 {
    let (n, m, col, head, theta) = (mem.n, mem.m, mem.col, mem.head, mem.theta);
    let (ws, wy) = (&mem.ws, &mem.wy);
    if sbgnrm <= 0.0 {
        xcp.copy_from_slice(x);
        return 0;
    }
    let mut bnded = true;
    let mut nfree = n;
    let mut nbreak: isize = -1;
    let mut ibkmin = 0usize;
    let mut bkmin = 0.0;
    let col2 = 2 * col;
    let (mut f1, mut tl, mut tu) = (0.0, 0.0, 0.0);
    for pi in &mut wa.p[..col2] {
        *pi = 0.0;
    }
    for i in 0..n {
        let neggi = -g[i];
        if iwhere[i] != 3 && iwhere[i] != -1 {
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.0;
            let xupper = nbd[i] >= 2 && tu <= 0.0;
            iwhere[i] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i] = -3;
            }
        }
        let mut pointr = head;
        if iwhere[i] != 0 && iwhere[i] != -1 {
            d[i] = 0.0;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            for j in 0..col {
                wa.p[j] += wy[i + n * pointr] * neggi;
                wa.p[col + j] += ws[i + n * pointr] * neggi;
                pointr = (pointr + 1) % m;
            }
            if nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.0 {
                nbreak += 1;
                let nb = nbreak as usize;
                iorder[nb] = i;
                t[nb] = tl / (-neggi);
                if nbreak == 0 || t[nb] < bkmin {
                    bkmin = t[nb];
                    ibkmin = nb;
                }
            } else if nbd[i] >= 2 && neggi > 0.0 {
                nbreak += 1;
                let nb = nbreak as usize;
                iorder[nb] = i;
                t[nb] = tu / neggi;
                if nbreak == 0 || t[nb] < bkmin {
                    bkmin = t[nb];
                    ibkmin = nb;
                }
            } else {
                nfree -= 1;
                iorder[nfree] = i;
                if neggi.abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }
    if theta != 1.0 {
        for pi in &mut wa.p[col..col2] {
            *pi *= theta;
        }
    }
    xcp.copy_from_slice(x);
    if nbreak == -1 && nfree == n {
        return 0;
    }
    for ci in &mut wa.c[..col2] {
        *ci = 0.0;
    }
    let mut f2 = -theta * f1;
    let f2_org = f2;
    if col > 0 {
        let info = mem.bmv(&wa.p, &mut wa.v);
        if info != 0 {
            return info;
        }
        f2 -= ddot(&wa.v[..col2], &wa.p[..col2]);
    }
    let mut dtm = -f1 / f2;
    let mut tsum = 0.0;
    let mut skip_final_step = false;
    if nbreak != -1 {
        let mut nleft = nbreak;
        let mut iter = 0usize;
        let mut tj = 0.0;
        loop {
            let tj0 = tj;
            let ibp;
            if iter == 0 {
                tj = bkmin;
                ibp = iorder[ibkmin];
            } else {
                if iter == 1 && ibkmin as isize != nbreak {
                    let nb = nbreak as usize;
                    t[ibkmin] = t[nb];
                    iorder[ibkmin] = iorder[nb];
                }
                let nl = nleft as usize;
                hpsolb(nl, t, iorder, iter - 1);
                tj = t[nl];
                ibp = iorder[nl];
            }
            let dt = tj - tj0;
            if dtm < dt {
                break;
            }
            tsum += dt;
            nleft -= 1;
            iter += 1;
            let mut dibp = d[ibp];
            d[ibp] = 0.0;
            let zibp;
            if dibp > 0.0 {
                zibp = u[ibp] - x[ibp];
                xcp[ibp] = u[ibp];
                iwhere[ibp] = 2;
            } else {
                zibp = l[ibp] - x[ibp];
                xcp[ibp] = l[ibp];
                iwhere[ibp] = 1;
            }
            // `nbreak == n` can never hold (nbreak is at most n - 1): SciPy's C translation of
            // the Fortran `nleft .eq. 0 .and. nbreak .eq. n` never takes this exit.
            if nleft == -1 && nbreak == n as isize {
                dtm = dt;
                skip_final_step = true;
                break;
            }
            let dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if col > 0 {
                daxpy(dt, &wa.p[..col2], &mut wa.c[..col2]);
                let mut pointr = head;
                for j in 0..col {
                    wa.wbp[j] = wy[ibp + n * pointr];
                    wa.wbp[col + j] = theta * ws[ibp + n * pointr];
                    pointr = (pointr + 1) % m;
                }
                let info = mem.bmv(&wa.wbp, &mut wa.v);
                if info != 0 {
                    return info;
                }
                let wmc = ddot(&wa.c[..col2], &wa.v[..col2]);
                let wmp = ddot(&wa.p[..col2], &wa.v[..col2]);
                let wmw = ddot(&wa.wbp[..col2], &wa.v[..col2]);
                dibp = -dibp;
                daxpy(dibp, &wa.wbp[..col2], &mut wa.p[..col2]);
                dibp = -dibp;
                f1 += dibp * wmc;
                f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw;
            }
            f2 = (EPSMACH * f2_org).max(f2);
            if nleft >= 0 {
                dtm = -f1 / f2;
            } else if bnded {
                dtm = 0.0;
                break;
            } else {
                dtm = -f1 / f2;
                break;
            }
        }
    }
    if !skip_final_step {
        if dtm <= 0.0 {
            dtm = 0.0;
        }
        tsum += dtm;
        daxpy(tsum, d, xcp);
    }
    if col > 0 {
        daxpy(dtm, &wa.p[..col2], &mut wa.c[..col2]);
    }
    0
}

/// `freev`: the free / active partition at the Cauchy point and the variables entering and
/// leaving the free set. Returns `wrk` (whether `formk` must rebuild its matrix).
#[allow(clippy::too_many_arguments)]
fn freev(
    n: usize,
    nfree: &mut usize,
    index: &mut [usize],
    nenter: &mut usize,
    ileave: &mut usize,
    indx2: &mut [usize],
    iwhere: &[i32],
    updatd: bool,
    cnstnd: bool,
    iter: usize,
) -> bool {
    *nenter = 0;
    *ileave = n;
    if iter > 0 && cnstnd {
        for i in 0..*nfree {
            let k = index[i];
            if iwhere[k] > 0 {
                *ileave -= 1;
                indx2[*ileave] = k;
            }
        }
        for i in *nfree..n {
            let k = index[i];
            if iwhere[k] <= 0 {
                indx2[*nenter] = k;
                *nenter += 1;
            }
        }
    }
    let wrk = *ileave < n || *nenter > 0 || updatd;
    *nfree = 0;
    let mut iact = n;
    for i in 0..n {
        if iwhere[i] <= 0 {
            index[*nfree] = i;
            *nfree += 1;
        } else {
            iact -= 1;
            index[iact] = i;
        }
    }
    wrk
}

/// `cmprlb`: the reduced gradient `r = −Z'(B(xcp − x) + g)` of the subspace problem.
#[allow(clippy::too_many_arguments)]
fn cmprlb(
    mem: &Memory,
    x: &[f64],
    g: &[f64],
    z: &[f64],
    r: &mut [f64],
    wa: &mut Scratch,
    index: &[usize],
    nfree: usize,
    cnstnd: bool,
) -> i32 {
    let (n, m, col, head, theta) = (mem.n, mem.m, mem.col, mem.head, mem.theta);
    if !cnstnd && col > 0 {
        for i in 0..n {
            r[i] = -g[i];
        }
        return 0;
    }
    for i in 0..nfree {
        let k = index[i];
        r[i] = -theta * (z[k] - x[k]) - g[k];
    }
    if mem.bmv(&wa.c, &mut wa.p) != 0 {
        return -8;
    }
    let mut pointr = head;
    for j in 0..col {
        let a1 = wa.p[j];
        let a2 = theta * wa.p[col + j];
        for i in 0..nfree {
            let k = index[i];
            r[i] = r[i] + mem.wy[k + n * pointr] * a1 + mem.ws[k + n * pointr] * a2;
        }
        pointr = (pointr + 1) % m;
    }
    0
}

/// `subsm`: the direct primal subspace minimization over the free variables, with the 3.0
/// projection and its backtrack to the box when the projected step is not a descent direction.
/// `d` holds the reduced gradient on entry; `x` is the Cauchy point on entry and the subspace
/// minimizer on exit. Returns `info`.
#[allow(clippy::too_many_arguments)]
fn subsm(
    mem: &Memory,
    nsub: usize,
    ind: &[usize],
    l: &[f64],
    u: &[f64],
    nbd: &[u8],
    x: &mut [f64],
    d: &mut [f64],
    xp: &mut [f64],
    xx: &[f64],
    gg: &[f64],
    wv: &mut [f64],
) -> i32 {
    let (n, m, col, head, theta) = (mem.n, mem.m, mem.col, mem.head, mem.theta);
    let (ws, wy) = (&mem.ws, &mem.wy);
    if nsub == 0 {
        return 0;
    }
    let mut pointr = head;
    for i in 0..col {
        let (mut temp1, mut temp2) = (0.0, 0.0);
        for j in 0..nsub {
            let k = ind[j];
            temp1 += wy[k + n * pointr] * d[j];
            temp2 += ws[k + n * pointr] * d[j];
        }
        wv[i] = temp1;
        wv[col + i] = theta * temp2;
        pointr = (pointr + 1) % m;
    }
    let m2 = 2 * m;
    let col2 = 2 * col;
    let info = dtrtrs_upper(true, &mem.wn, m2, col2, 1, wv, m2);
    if info != 0 {
        return info;
    }
    for w in &mut wv[..col] {
        *w = -*w;
    }
    let info = dtrtrs_upper(false, &mem.wn, m2, col2, 1, wv, m2);
    if info != 0 {
        return info;
    }
    let mut pointr = head;
    for jy in 0..col {
        let js = col + jy;
        for i in 0..nsub {
            let k = ind[i];
            d[i] = d[i] + (wy[k + n * pointr] * wv[jy] / theta) + (ws[k + n * pointr] * wv[js]);
        }
        pointr = (pointr + 1) % m;
    }
    let temp = 1.0 / theta;
    for di in &mut d[..nsub] {
        *di *= temp;
    }
    let mut iword = false;
    xp.copy_from_slice(x);
    for i in 0..nsub {
        let k = ind[i];
        let dk = d[i];
        let xk = x[k];
        match nbd[k] {
            0 => x[k] = xk + dk,
            1 => {
                x[k] = l[k].max(xk + dk);
                if x[k] == l[k] {
                    iword = true;
                }
            }
            2 => {
                let xk = l[k].max(xk + dk);
                x[k] = u[k].min(xk);
                if x[k] == l[k] || x[k] == u[k] {
                    iword = true;
                }
            }
            _ => {
                x[k] = u[k].min(xk + dk);
                if x[k] == u[k] {
                    iword = true;
                }
            }
        }
    }
    if !iword {
        return 0;
    }
    let mut dd_p = 0.0;
    for i in 0..n {
        dd_p += (x[i] - xx[i]) * gg[i];
    }
    if dd_p > 0.0 {
        x.copy_from_slice(xp);
    } else {
        return 0;
    }
    let mut alpha = 1.0;
    let mut temp1 = 1.0;
    let mut ibd = 0;
    for i in 0..nsub {
        let k = ind[i];
        let dk = d[i];
        if nbd[k] != 0 {
            if dk < 0.0 && nbd[k] <= 2 {
                let temp2 = l[k] - x[k];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k] >= 2 {
                let temp2 = u[k] - x[k];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd = i;
            }
        }
    }
    if alpha < 1.0 {
        let dk = d[ibd];
        let k = ind[ibd];
        if dk > 0.0 {
            x[k] = u[k];
            d[ibd] = 0.0;
        } else if dk < 0.0 {
            x[k] = l[k];
            d[ibd] = 0.0;
        }
    }
    for i in 0..nsub {
        let k = ind[i];
        x[k] += alpha * d[i];
    }
    0
}

/// `_minimize_lbfgsb`'s bound encoding: `nbd` 0 free, 1 lower only, 2 both, 3 upper only, with
/// an absent bound stored as 0.0 (the Python driver's `zeros` arrays).
pub(crate) fn encode_bounds(lower: &[f64], upper: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
    let n = lower.len();
    let mut l = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut nbd = vec![0u8; n];
    for i in 0..n {
        let has_l = !lower[i].is_infinite();
        let has_u = !upper[i].is_infinite();
        if has_l {
            l[i] = lower[i];
        }
        if has_u {
            u[i] = upper[i];
        }
        nbd[i] = match (has_l, has_u) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
    }
    (l, u, nbd)
}

/// `_minimize_lbfgsb` from the (already clipped) `x0` with the encoded bounds. `m ≥ 1`,
/// `maxls ≥ 1`, `factr ≥ 0` and `l ≤ u` where both are present are the caller's to check.
#[allow(clippy::too_many_lines)]
pub(crate) fn minimize_lbfgsb<O: LbfgsbObjective>(
    obj: &mut O,
    x0: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[u8],
    params: &LbfgsbParams,
) -> Result<LbfgsbOutcome, O::Error> {
    let n = x0.len();
    let m = params.m;
    let pgtol = params.pgtol;
    let mut x = x0.to_vec();
    let mut mem = Memory::new(n, m);
    let mut z = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut xp = vec![0.0; n];
    let mut wa = Scratch {
        p: vec![0.0; 2 * m],
        c: vec![0.0; 2 * m],
        wbp: vec![0.0; 2 * m],
        v: vec![0.0; 2 * m],
    };
    let mut index = vec![0usize; n];
    let mut iwhere = vec![0i32; n];
    let mut indx2 = vec![0usize; n];

    let tol = params.factr * EPSMACH;
    let mut nfree = n;
    let mut nenter = 0usize;
    let mut ileave = 0usize;
    let mut iter = 0usize;
    let mut n_iterations = 0usize;
    // `lnsrlb`'s `dcsrch(ftol = 1e-3, gtol = 0.9, xtol = 0.1, stpmin = 0, stpmax = stpmx)`.
    let mut search = Dcsrch::new(Minpack::C, 1e-3, 0.9, 0.1, 0.0, 0.0);

    let (cnstnd, boxed) = active(l, u, nbd, &mut x, &mut iwhere);
    let (mut f, mut g) = obj.fun_and_grad(&x)?;
    let mut sbgnrm = projgr(l, u, nbd, &x, &g);

    let stop = 'mainlb: {
        if sbgnrm <= pgtol {
            break 'mainlb LbfgsbStop::ProjectedGradient;
        }
        // LINE222: every `continue` is the C file's `goto LINE222`.
        loop {
            let wrk = if !cnstnd && mem.col > 0 {
                z.copy_from_slice(&x);
                mem.updatd
            } else {
                let info = cauchy(
                    &mem,
                    &x,
                    l,
                    u,
                    nbd,
                    &g,
                    &mut indx2,
                    &mut iwhere,
                    &mut t,
                    &mut d,
                    &mut z,
                    &mut wa,
                    sbgnrm,
                );
                if info != 0 {
                    mem.reset();
                    continue;
                }
                freev(
                    n,
                    &mut nfree,
                    &mut index,
                    &mut nenter,
                    &mut ileave,
                    &mut indx2,
                    &iwhere,
                    mem.updatd,
                    cnstnd,
                    iter,
                )
            };
            // LINE333
            if nfree != 0 && mem.col != 0 {
                let mut info = 0;
                if wrk {
                    info = mem.formk(nfree, &index, nenter, ileave, &indx2);
                }
                if info != 0 {
                    mem.reset();
                    continue;
                }
                info = cmprlb(&mem, &x, &g, &z, &mut r, &mut wa, &index, nfree, cnstnd);
                if info == 0 {
                    info = subsm(
                        &mem, nfree, &index, l, u, nbd, &mut z, &mut r, &mut xp, &x, &g, &mut wa.p,
                    );
                }
                if info != 0 {
                    mem.reset();
                    continue;
                }
            }
            // LINE555
            for i in 0..n {
                d[i] = z[i] - x[i];
            }
            // LINE666: `lnsrlb`, entered afresh.
            let dnorm = dnrm2(&d);
            let dtd = dnorm * dnorm;
            let mut stpmx = 1.0e10;
            if cnstnd {
                if iter == 0 {
                    stpmx = 1.0;
                } else {
                    for i in 0..n {
                        let a1 = d[i];
                        if nbd[i] != 0 {
                            if a1 < 0.0 && nbd[i] <= 2 {
                                let a2 = l[i] - x[i];
                                if a2 >= 0.0 {
                                    stpmx = 0.0;
                                } else if a1 * stpmx < a2 {
                                    stpmx = a2 / a1;
                                }
                            } else if a1 > 0.0 && nbd[i] >= 2 {
                                let a2 = u[i] - x[i];
                                if a2 <= 0.0 {
                                    stpmx = 0.0;
                                } else if a1 * stpmx > a2 {
                                    stpmx = a2 / a1;
                                }
                            }
                        }
                    }
                }
            }
            let mut stp = if iter == 0 && !boxed {
                (1.0 / dnorm).min(stpmx)
            } else {
                1.0
            };
            t.copy_from_slice(&x);
            r.copy_from_slice(&g);
            let fold = f;
            let mut ifun = 0usize;
            let mut gd;
            let mut gdold = 0.0;
            search.set_stpmax(stpmx);
            let mut ls_task = Task::Start;
            // LINE556, until the search converges (`false`) or fails (`true`): a non-descent
            // direction, or `maxls` trial points without an acceptable one.
            let failed = loop {
                gd = ddot(&g, &d);
                if ifun == 0 {
                    gdold = gd;
                    if gd >= 0.0 {
                        break true;
                    }
                }
                let (next, task) = search.iterate(stp, f, gd, ls_task);
                stp = next;
                ls_task = task;
                if ls_task == Task::Convergence || ls_task == Task::Warning {
                    break false;
                }
                ifun += 1;
                let iback = ifun - 1;
                if stp == 1.0 {
                    x.copy_from_slice(&z);
                } else {
                    for i in 0..n {
                        x[i] = stp * d[i] + t[i];
                        if nbd[i] == 1 || nbd[i] == 2 {
                            x[i] = x[i].max(l[i]);
                        }
                        if nbd[i] == 2 || nbd[i] == 3 {
                            x[i] = x[i].min(u[i]);
                        }
                    }
                }
                // `mainlb` gives up on the search before evaluating its `maxls`-th trial point.
                if iback >= params.maxls {
                    break true;
                }
                (f, g) = obj.fun_and_grad(&x)?;
            };
            if failed {
                x.copy_from_slice(&t);
                g.copy_from_slice(&r);
                f = fold;
                if mem.col == 0 {
                    break 'mainlb LbfgsbStop::Abnormal;
                }
                // Discard the corrections and retry from the saved point (`task = RESTART`).
                mem.reset();
                continue;
            }
            // NEW_X: `mainlb` returns the iterate, `_minimize_lbfgsb` counts and checks it.
            iter += 1;
            sbgnrm = projgr(l, u, nbd, &x, &g);
            n_iterations += 1;
            let mut halt = None;
            if obj.callback(&x, f) {
                halt = Some(LbfgsbStop::Callback);
            }
            if n_iterations >= params.maxiter {
                halt = Some(LbfgsbStop::MaxIter);
            } else if obj.nfev() > params.maxfun {
                halt = Some(LbfgsbStop::MaxFun);
            }
            if let Some(stop) = halt {
                break 'mainlb stop;
            }
            // LINE777
            if sbgnrm <= pgtol {
                break 'mainlb LbfgsbStop::ProjectedGradient;
            }
            let ddum = fold.abs().max(f.abs()).max(1.0);
            if fold - f <= tol * ddum {
                break 'mainlb LbfgsbStop::RelativeReduction;
            }
            for i in 0..n {
                r[i] = g[i] - r[i];
            }
            let rnorm = dnrm2(&r);
            let rr = rnorm * rnorm;
            let (dr, ddum) = if stp == 1.0 {
                (gd - gdold, -gdold)
            } else {
                for di in &mut d {
                    *di *= stp;
                }
                ((gd - gdold) * stp, -gdold * stp)
            };
            if dr <= EPSMACH * ddum {
                mem.updatd = false;
                continue;
            }
            mem.updatd = true;
            mem.iupdat += 1;
            mem.matupd(&d, &r, rr, dr, stp, dtd);
            if mem.formt() != 0 {
                mem.reset();
            }
        }
    };

    let warnflag = match stop {
        LbfgsbStop::ProjectedGradient | LbfgsbStop::RelativeReduction => 0,
        _ if obj.nfev() > params.maxfun || n_iterations >= params.maxiter => 1,
        _ => 2,
    };
    Ok(LbfgsbOutcome {
        x,
        fun: f,
        jac: g,
        nit: n_iterations,
        stop,
        warnflag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(x: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            s += 100.0 * a * a + b * b;
        }
        s
    }

    fn rosen_der(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut g = vec![0.0; n];
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        for i in 1..n - 1 {
            g[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
                - 400.0 * x[i] * (x[i + 1] - x[i] * x[i])
                - 2.0 * (1.0 - x[i]);
        }
        g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
        g
    }

    struct Rosen {
        nfev: usize,
    }

    impl LbfgsbObjective for Rosen {
        type Error = ();
        fn fun_and_grad(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), ()> {
            self.nfev += 1;
            Ok((rosen(x), rosen_der(x)))
        }
        fn nfev(&self) -> usize {
            self.nfev
        }
        fn callback(&mut self, _x: &[f64], _f: f64) -> bool {
            false
        }
    }

    /// `maxcor` is not an fsci option, so the driver is checked at SciPy's `maxcor=3` directly:
    /// `minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='L-BFGS-B', jac=rosen_der,
    /// options={'maxcor': 3})` in SciPy 1.17.1, bit-identical on the Prescott, Nehalem,
    /// Sandybridge, Haswell and Zen kernels. With the default 10 corrections the same start takes
    /// 24 iterations, so a memory that ignored `m` fails this.
    #[test]
    fn three_corrections_take_scipys_path() {
        let (l, u, nbd) = encode_bounds(&[f64::NEG_INFINITY; 5], &[f64::INFINITY; 5]);
        let params = LbfgsbParams {
            m: 3,
            factr: 2.220_446_049_250_313e-9 / f64::EPSILON,
            pgtol: 1e-5,
            maxfun: 15_000,
            maxiter: 15_000,
            maxls: 20,
        };
        let mut obj = Rosen { nfev: 0 };
        let out = minimize_lbfgsb(&mut obj, &[1.3, 0.7, 0.8, 1.9, 1.2], &l, &u, &nbd, &params)
            .expect("rosen");
        let want: [f64; 5] = [
            0.999_999_937_483_500_9,
            0.999_999_971_209_320_2,
            1.000_000_355_435_985_5,
            1.000_001_386_628_8,
            1.000_003_148_009_862,
        ];
        assert_eq!(
            out.x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "x = {:?}",
            out.x
        );
        assert_eq!((out.nit, obj.nfev), (21, 26));
        assert_eq!(out.stop, LbfgsbStop::RelativeReduction);
        assert_eq!(out.warnflag, 0);
    }

    /// OpenBLAS's `dnrm2` (what `__lbfgsb.c` calls for `dnorm` and `rr`) is correctly rounded on
    /// this vector — 3.3401142779246746 under both the Haswell and the Nehalem kernel — where the
    /// root of an f64 running sum of squares is one ulp low.
    #[test]
    fn dnrm2_rounds_like_openblas() {
        let x = [
            -1.660_566_212_357_912_8,
            0.764_599_334_433_535_6,
            2.686_253_654_742_033_7,
            0.462_617_691_704_991_8,
            -0.619_917_152_095_319_1,
        ];
        assert_eq!(dnrm2(&x), 3.340_114_277_924_674_6);
        let naive = x.iter().fold(0.0, |s, v| s + v * v).sqrt();
        assert_eq!(naive, 3.340_114_277_924_674);
    }
}
