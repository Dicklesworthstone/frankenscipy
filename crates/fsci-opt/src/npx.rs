#![forbid(unsafe_code)]
//! NumPy's floating-point evaluation order for the array operations SciPy's pure-Python
//! solvers lean on (frankenscipy-1ksfv.3).
//!
//! `scipy._lib.pyprima` (SciPy's COBYLA) writes its linear algebra as `np.dot`, `@`, `np.sum`,
//! `np.linalg.norm`, `np.linalg.inv` and `np.linalg.lstsq`, and COBYLA's iterates follow every
//! rounding of them. Each function here evaluates its operation in the order NumPy 2.4 with
//! OpenBLAS 0.3 (x86-64 Haswell/SkylakeX kernels, reference LAPACK) does for the shapes and
//! memory layouts the caller states: fused multiply-adds in BLAS kernel order, NumPy's pairwise
//! summation, LAPACK's LU inverse and SVD least squares (`dgelsd`), down to the x87
//! extended-precision `dnrm2`. Every order was pinned against NumPy bit for bit on random data;
//! all of them are ordinary floating-point evaluations of the same mathematics.

/// A dense row-major matrix: a C-ordered NumPy array.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

impl Mat {
    pub(crate) fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Self::filled(rows, cols, 0.0)
    }

    pub(crate) fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub(crate) fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub(crate) fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub(crate) fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub(crate) fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub(crate) fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.at(i, j)).collect()
    }

    pub(crate) fn set_col(&mut self, j: usize, v: &[f64]) {
        for (i, &x) in v.iter().enumerate() {
            self.set(i, j, x);
        }
    }

    pub(crate) fn swap_cols(&mut self, a: usize, b: usize) {
        for i in 0..self.rows {
            self.data.swap(i * self.cols + a, i * self.cols + b);
        }
    }

    /// The selected columns as a fresh C-ordered matrix.
    pub(crate) fn select_cols(&self, idx: &[usize]) -> Self {
        let mut out = Self::zeros(self.rows, idx.len());
        for i in 0..self.rows {
            for (k, &j) in idx.iter().enumerate() {
                out.set(i, k, self.at(i, j));
            }
        }
        out
    }

    /// `A[:, idx]` as NumPy lays it out: Fortran-ordered, so the selected columns are contiguous.
    /// Returned transposed (row `k` is column `idx[k]`), the form [`mat_vec`] takes for
    /// `x @ A[:, idx]`.
    pub(crate) fn select_cols_t(&self, idx: &[usize]) -> Self {
        let mut out = Self::zeros(idx.len(), self.rows);
        for (k, &j) in idx.iter().enumerate() {
            for i in 0..self.rows {
                out.set(k, i, self.at(i, j));
            }
        }
        out
    }

    /// `A[:, :ncols]` copied.
    pub(crate) fn first_cols(&self, ncols: usize) -> Self {
        let idx: Vec<usize> = (0..ncols).collect();
        self.select_cols(&idx)
    }

    pub(crate) fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Level 1: ddot, np.dot, norms, sums
// ══════════════════════════════════════════════════════════════════════

/// OpenBLAS `ddot` (`kernel/x86_64/ddot.c`). `unit` is true when both vectors have unit stride
/// in NumPy's view of them; otherwise the strided kernel runs.
pub(crate) fn ddot(x: &[f64], y: &[f64], unit: bool) -> f64 {
    let n = x.len();
    if unit {
        let n1 = n & !15;
        let mut dot = 0.0_f64;
        let mut i = 0;
        if n1 > 0 {
            // ddot_microk_skylakex-2.c: four 512-bit, then four 256-bit FMA accumulators.
            let n32 = n1 & !31;
            let mut acc5 = [[0.0_f64; 8]; 4];
            while i < n32 {
                for (k, lanes) in acc5.iter_mut().enumerate() {
                    for (l, lane) in lanes.iter_mut().enumerate() {
                        let j = i + 8 * k + l;
                        *lane = x[j].mul_add(y[j], *lane);
                    }
                }
                i += 32;
            }
            let mut acc = [[0.0_f64; 4]; 4];
            for k in 0..4 {
                for l in 0..4 {
                    acc[k][l] = acc5[k][l] + acc5[k][l + 4];
                }
            }
            while i < n1 {
                for (k, lanes) in acc.iter_mut().enumerate() {
                    for (l, lane) in lanes.iter_mut().enumerate() {
                        let j = i + 4 * k + l;
                        *lane = x[j].mul_add(y[j], *lane);
                    }
                }
                i += 16;
            }
            let a: Vec<f64> = (0..4)
                .map(|l| ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l])
                .collect();
            dot = (a[0] + a[2]) + (a[1] + a[3]);
        }
        while i < n {
            dot = y[i].mul_add(x[i], dot);
            i += 1;
        }
        dot
    } else {
        let mut t1 = 0.0_f64;
        let mut t2 = 0.0_f64;
        let n1 = n & !3;
        let mut i = 0;
        while i < n1 {
            let m1 = y[i] * x[i];
            let m2 = y[i + 1] * x[i + 1];
            let m3 = y[i + 2] * x[i + 2];
            let m4 = y[i + 3] * x[i + 3];
            t1 += m1 + m3;
            t2 += m2 + m4;
            i += 4;
        }
        while i < n {
            t1 = y[i].mul_add(x[i], t1);
            i += 1;
        }
        t1 + t2
    }
}

/// `np.dot(x, y)` for 1-D arrays: a one-element pair is a plain product, longer ones `ddot`.
pub(crate) fn dot(x: &[f64], y: &[f64], unit: bool) -> f64 {
    match x.len() {
        0 => 0.0,
        1 => y[0] * x[0],
        _ => ddot(x, y, unit),
    }
}

/// `x @ y` for 1-D arrays (NumPy's matmul `DOUBLE_dot`: `0.0 + ddot`).
pub(crate) fn mm_dot(x: &[f64], y: &[f64], unit: bool) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    0.0 + ddot(x, y, unit)
}

/// `np.linalg.norm(v)` for a 1-D array: `sqrt(v.dot(v))`.
pub(crate) fn norm(v: &[f64]) -> f64 {
    dot(v, v, true).sqrt()
}

/// NumPy's pairwise summation (`np.sum` of a contiguous 1-D array).
pub(crate) fn sum(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 8 {
        let mut res = 0.0_f64;
        for &x in v {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0_f64; 8];
        r.copy_from_slice(&v[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, rk) in r.iter_mut().enumerate() {
                *rk += v[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += v[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        sum(&v[..n2]) + sum(&v[n2..])
    }
}

/// `np.sum(A, axis=0)` for a C-ordered `A`: row by row from zero, or pairwise when `A` is a
/// single (contiguous) column.
pub(crate) fn sum_axis0(a: &Mat) -> Vec<f64> {
    if a.cols == 1 {
        return vec![sum(&a.col(0))];
    }
    let mut out = vec![0.0_f64; a.cols];
    for i in 0..a.rows {
        for (j, o) in out.iter_mut().enumerate() {
            *o += a.at(i, j);
        }
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// Level 2/3: dgemv_t, dgemv_n, dgemm as NumPy's matmul reaches them
// ══════════════════════════════════════════════════════════════════════

/// OpenBLAS `dgemv_t` (`dgemv_t_4.c` + the Haswell 4x4 microkernel): output `j` is the inner
/// product of the contiguous vector `cols.row(j)` with `x`, the kernel chosen by `j`'s place
/// among the `cols.rows` outputs.
fn gemv_t(cols: &Mat, x: &[f64]) -> Vec<f64> {
    let m = x.len();
    let ncols = cols.rows;
    let m3 = m & 3;
    let nb = if m >= 4 { m - m3 } else { 0 };
    let n1 = ncols >> 2;
    (0..ncols)
        .map(|j| {
            let row = cols.row(j);
            let mut y = 0.0_f64;
            if nb > 0 {
                let part = if j < 4 * n1 {
                    kernel_t_4x4(&row[..nb], &x[..nb])
                } else if ncols & 2 != 0 && j < 4 * n1 + 2 {
                    kernel_t_4x2(&row[..nb], &x[..nb])
                } else {
                    kernel_t_4x1(&row[..nb], &x[..nb])
                };
                y += part;
            }
            let r = &row[nb..];
            let t = &x[nb..];
            match m3 {
                1 => r[0].mul_add(t[0], y),
                2 => y + r[0].mul_add(t[0], r[1] * t[1]),
                3 => y + r[2].mul_add(t[2], r[0].mul_add(t[0], r[1] * t[1])),
                _ => y,
            }
        })
        .collect()
}

/// AVX2 4x4 kernel: four FMA lanes over four-element chunks, then `(l0+l2) + (l1+l3)`.
fn kernel_t_4x4(a: &[f64], x: &[f64]) -> f64 {
    let mut l = [0.0_f64; 4];
    for i in (0..x.len()).step_by(4) {
        for (k, lane) in l.iter_mut().enumerate() {
            *lane = a[i + k].mul_add(x[i + k], *lane);
        }
    }
    (l[0] + l[2]) + (l[1] + l[3])
}

/// SSE2 4x2 kernel: two lanes of separately rounded multiply-adds, then a horizontal add.
fn kernel_t_4x2(a: &[f64], x: &[f64]) -> f64 {
    let mut l = [0.0_f64; 2];
    for i in (0..x.len()).step_by(2) {
        l[0] += a[i] * x[i];
        l[1] += a[i + 1] * x[i + 1];
    }
    l[0] + l[1]
}

/// SSE2 4x1 kernel: two 2-lane accumulators, summed lane-wise, then horizontally.
fn kernel_t_4x1(a: &[f64], x: &[f64]) -> f64 {
    let mut l10 = [0.0_f64; 2];
    let mut l9 = [0.0_f64; 2];
    for i in (0..x.len()).step_by(4) {
        l10[0] += a[i] * x[i];
        l10[1] += a[i + 1] * x[i + 1];
        l9[0] += a[i + 2] * x[i + 2];
        l9[1] += a[i + 3] * x[i + 3];
    }
    (l10[0] + l9[0]) + (l10[1] + l9[1])
}

/// OpenBLAS `dgemv_n` (`dgemv_n_4.c` + the SkylakeX microkernel), as NumPy reaches it for
/// `x @ B` with C-ordered `B` (first `ncols` columns, row stride `b.cols`): output `i` is
/// `sum_j x[j] B[j, i]`, accumulated four rows of `B` at a time.
fn gemv_n(b: &Mat, ncols: usize, x: &[f64], x_unit: bool) -> Vec<f64> {
    let n = x.len();
    let m = ncols;
    let lda = b.cols;
    let m3 = m & 3;
    let mb = m - m3;
    let mut y = vec![0.0_f64; m];
    let at = |j: usize, i: usize| b.at(j, i);
    let four = |j: usize, i: usize| {
        at(j + 3, i).mul_add(
            x[j + 3],
            at(j + 2, i).mul_add(x[j + 2], at(j, i).mul_add(x[j], at(j + 1, i) * x[j + 1])),
        )
    };
    if mb > 0 {
        let n1 = n >> 2;
        let n2 = n & 3;
        let mut j = 0;
        for _ in 0..n1 {
            for (i, yi) in y.iter_mut().enumerate().take(mb) {
                *yi += four(j, i);
            }
            j += 4;
        }
        if x_unit {
            if n2 & 2 != 0 {
                for (i, yi) in y.iter_mut().enumerate().take(mb) {
                    *yi += at(j, i).mul_add(x[j], at(j + 1, i) * x[j + 1]);
                }
                j += 2;
            }
            if n2 & 1 != 0 {
                for (i, yi) in y.iter_mut().enumerate().take(mb) {
                    *yi += at(j, i) * x[j];
                }
            }
        } else {
            for _ in 0..n2 {
                for (i, yi) in y.iter_mut().enumerate().take(mb) {
                    *yi += at(j, i) * x[j];
                }
                j += 1;
            }
        }
    }
    for (i, yi) in y.iter_mut().enumerate().skip(mb) {
        let mut t = 0.0_f64;
        let mut j = 0;
        if lda == m3 && x_unit && m3 >= 2 {
            while j < (n & !3) {
                t += at(j, i).mul_add(x[j], at(j + 1, i) * x[j + 1]);
                t += at(j + 2, i).mul_add(x[j + 2], at(j + 3, i) * x[j + 3]);
                j += 4;
            }
        } else if lda == m3 && x_unit && m3 == 1 {
            while j < (n & !3) {
                t += four(j, i);
                j += 4;
            }
        }
        while j < n {
            t = at(j, i).mul_add(x[j], t);
            j += 1;
        }
        *yi += t;
    }
    y
}

/// `x @ B` for 1-D `x` (unit stride or not) and C-ordered `B` restricted to its first `ncols`
/// columns (row stride `b.cols`).
pub(crate) fn vec_mat(x: &[f64], x_unit: bool, b: &Mat, ncols: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || ncols == 0 {
        return vec![0.0; ncols];
    }
    if ncols == 1 {
        return vec![mm_dot(x, &b.col(0), x_unit && b.cols == 1)];
    }
    if n == 1 {
        return (0..ncols).map(|j| 0.0 + x[0] * b.at(0, j)).collect();
    }
    gemv_n(b, ncols, x, x_unit)
}

/// `A @ x` for C-ordered `A` and 1-D `x`; equally `x @ F` for a Fortran-ordered `F = A^T`.
pub(crate) fn mat_vec(a: &Mat, x: &[f64], x_unit: bool) -> Vec<f64> {
    let n = x.len();
    if a.rows == 0 || n == 0 {
        return vec![0.0; a.rows];
    }
    if a.rows == 1 {
        return vec![mm_dot(a.row(0), x, x_unit)];
    }
    if n == 1 {
        return (0..a.rows).map(|i| 0.0 + a.at(i, 0) * x[0]).collect();
    }
    gemv_t(a, x)
}

/// `A @ B` for C- or Fortran-ordered `A` and C-ordered `B` (first `ncols` columns): `dgemm`
/// accumulates each entry by fused multiply-adds in `k` order from zero.
pub(crate) fn mat_mat(a: &Mat, b: &Mat, ncols: usize) -> Mat {
    let (m, k, p) = (a.rows, a.cols, ncols);
    let mut out = Mat::zeros(m, p);
    if m == 0 || k == 0 || p == 0 {
        return out;
    }
    if m == 1 && p == 1 {
        out.set(0, 0, mm_dot(a.row(0), &b.col(0), b.cols == 1));
        return out;
    }
    if k == 1 {
        for i in 0..m {
            for j in 0..p {
                out.set(i, j, 0.0 + a.at(i, 0) * b.at(0, j));
            }
        }
        return out;
    }
    if m == 1 {
        let row = vec_mat(a.row(0), true, b, p);
        out.row_mut(0).copy_from_slice(&row);
        return out;
    }
    if p == 1 {
        let col = mat_vec(a, &b.col(0), b.cols == 1);
        out.set_col(0, &col);
        return out;
    }
    for i in 0..m {
        for j in 0..p {
            let mut acc = 0.0_f64;
            for kk in 0..k {
                acc = a.at(i, kk).mul_add(b.at(kk, j), acc);
            }
            out.set(i, j, acc);
        }
    }
    out
}

/// Each entry of `Q[:, [a, b]] @ G.T` (two-term `dgemm` from zero) for `G = [[c, s], [g10, c]]`,
/// written back into columns `ka`, `kb` of `q`.
pub(crate) fn rotate_cols(
    q: &mut Mat,
    a: usize,
    b: usize,
    ka: usize,
    kb: usize,
    g: (f64, f64, f64),
) {
    let (c, s, g10) = g;
    for i in 0..q.rows {
        let qa = q.at(i, a);
        let qb = q.at(i, b);
        q.set(i, ka, qb.mul_add(s, qa.mul_add(c, 0.0)));
        q.set(i, kb, qb.mul_add(c, qa.mul_add(g10, 0.0)));
    }
}

// ══════════════════════════════════════════════════════════════════════
// LAPACK: np.linalg.inv (getrf/getrs) and np.linalg.lstsq (dgelsd)
// ══════════════════════════════════════════════════════════════════════

/// `np.linalg.inv(A)`: `getrf` (partial pivoting, the column scaled by the pivot's reciprocal)
/// and `getrs` on the identity (unit-lower forward substitution, upper back substitution
/// multiplying by each pivot's stored reciprocal). Pinned bit for bit on COBYLA's initial
/// simplex (lower triangular, entries `0`, `±rhobeg`); for other matrices it is the same
/// algorithm. A singular `A` gives a NaN matrix (NumPy raises `LinAlgError`).
pub(crate) fn inv(a: &Mat) -> Mat {
    let n = a.rows;
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if lu.at(i, k).abs() > lu.at(p, k).abs() {
                p = i;
            }
        }
        if p != k {
            for j in 0..n {
                let t = lu.at(k, j);
                lu.set(k, j, lu.at(p, j));
                lu.set(p, j, t);
            }
            perm.swap(k, p);
        }
        let pivot = lu.at(k, k);
        if pivot == 0.0 || pivot.is_nan() {
            return Mat::filled(n, n, f64::NAN);
        }
        let r = 1.0 / pivot;
        for i in k + 1..n {
            lu.set(i, k, lu.at(i, k) * r);
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let v = (-lu.at(i, k)).mul_add(lu.at(k, j), lu.at(i, j));
                lu.set(i, j, v);
            }
        }
    }
    let mut out = Mat::zeros(n, n);
    for col in 0..n {
        let mut x: Vec<f64> = perm
            .iter()
            .map(|&r| if r == col { 1.0 } else { 0.0 })
            .collect();
        for i in 0..n {
            let mut s = x[i];
            for k in 0..i {
                s = (-lu.at(i, k)).mul_add(x[k], s);
            }
            x[i] = s;
        }
        let mut y = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s = (-lu.at(i, k)).mul_add(y[k], s);
            }
            y[i] = s * (1.0 / lu.at(i, i));
        }
        out.set_col(col, &y);
    }
    out
}

/// An x87 extended-precision number (64-bit significand, round to nearest even), enough of it
/// for OpenBLAS's x86-64 `dnrm2` (`nrm2.S`), which squares and sums in the x87 stack.
#[derive(Debug, Clone, Copy)]
struct X87 {
    /// The significand: 0, or normalized with bit 63 set. The value is `mant * 2^exp`.
    mant: u64,
    exp: i32,
}

impl X87 {
    const ZERO: Self = Self { mant: 0, exp: 0 };

    /// `|x|` exactly (the sign never matters for a sum of squares).
    fn from_abs_f64(x: f64) -> Self {
        let bits = x.abs().to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1_u64 << 52) - 1);
        if biased == 0 && frac == 0 {
            return Self::ZERO;
        }
        let (m, e) = if biased == 0 {
            (frac, -1074)
        } else {
            (frac | (1_u64 << 52), biased - 1075)
        };
        Self::round(u128::from(m), e, false)
    }

    /// Round the exact value `(m + sticky) * 2^e` to a 64-bit significand.
    fn round(m: u128, e: i32, sticky: bool) -> Self {
        if m == 0 {
            return Self::ZERO;
        }
        let bits = 128 - m.leading_zeros() as i32;
        if bits <= 64 {
            let shift = 64 - bits;
            return Self {
                mant: (m << shift) as u64,
                exp: e - shift,
            };
        }
        let drop = (bits - 64) as u32;
        let kept = m >> drop;
        let rem = m & ((1_u128 << drop) - 1);
        let half = 1_u128 << (drop - 1);
        // A set sticky bit means the exact value lies strictly above `rem`.
        let up = rem > half || (rem == half && (sticky || kept & 1 == 1));
        let mut mant = kept;
        let mut exp = e + drop as i32;
        if up {
            mant += 1;
            if mant >> 64 != 0 {
                mant >>= 1;
                exp += 1;
            }
        }
        Self {
            mant: mant as u64,
            exp,
        }
    }

    fn mul(self, other: Self) -> Self {
        if self.mant == 0 || other.mant == 0 {
            return Self::ZERO;
        }
        let m = u128::from(self.mant) * u128::from(other.mant);
        Self::round(m, self.exp + other.exp, false)
    }

    /// Sum of two non-negative values.
    fn add(self, other: Self) -> Self {
        if self.mant == 0 {
            return other;
        }
        if other.mant == 0 {
            return self;
        }
        let (hi, lo) = if self.exp >= other.exp {
            (self, other)
        } else {
            (other, self)
        };
        // Align on hi.exp - 62 so both significands fit in a u128 with guard bits.
        let base = hi.exp - 62;
        let hi_m = u128::from(hi.mant) << 62;
        let shift = base - lo.exp;
        let (lo_m, sticky) = if shift <= 0 {
            (u128::from(lo.mant) << (-shift), false)
        } else if shift >= 128 {
            (0, true)
        } else {
            let s = shift as u32;
            let v = u128::from(lo.mant);
            (v >> s, v & ((1_u128 << s) - 1) != 0)
        };
        Self::round(hi_m + lo_m, base, sticky)
    }

    /// Correctly rounded square root.
    fn sqrt(self) -> Self {
        if self.mant == 0 {
            return Self::ZERO;
        }
        // value = mant * 2^exp; take M = mant << s with s making (exp - s) even and M of
        // 127 or 128 bits, so isqrt(M) has 64 bits.
        let mut s = 64;
        if (self.exp - s).rem_euclid(2) != 0 {
            s -= 1;
        }
        let big = u128::from(self.mant) << s;
        let r = big.isqrt();
        let rem = big - r * r;
        let e = (self.exp - s) / 2;
        // sqrt(big) = r + f with f in [0, 1) and never exactly 1/2; round up when f > 1/2,
        // i.e. when big > r^2 + r.
        Self::round(if rem > r { r + 1 } else { r }, e, false)
    }

    /// Store as a double (round to 53 bits, nearest even).
    fn to_f64(self) -> f64 {
        if self.mant == 0 {
            return 0.0;
        }
        let drop = 11_u32;
        let kept = self.mant >> drop;
        let rem = self.mant & ((1_u64 << drop) - 1);
        let half = 1_u64 << (drop - 1);
        let mut m = kept;
        let mut exp = self.exp + drop as i32;
        if rem > half || (rem == half && kept & 1 == 1) {
            m += 1;
            if m >> 53 != 0 {
                m >>= 1;
                exp += 1;
            }
        }
        ldexp(m as f64, exp)
    }
}

/// `x * 2^e` without intermediate overflow or underflow for representable results.
fn ldexp(x: f64, e: i32) -> f64 {
    let mut v = x;
    let mut e = e;
    while e > 1000 {
        v *= 2.0_f64.powi(1000);
        e -= 1000;
    }
    while e < -1000 {
        v *= 2.0_f64.powi(-1000);
        e += 1000;
    }
    v * 2.0_f64.powi(e)
}

/// OpenBLAS x86-64 `dnrm2` (`nrm2.S`): squares accumulated in four x87 registers (eight at a
/// time, the tail in the last), combined as `a + ((b + d) + c)`, `fsqrt`, stored as a double.
fn dnrm2(x: &[f64]) -> f64 {
    let sq = |v: f64| {
        let a = X87::from_abs_f64(v);
        a.mul(a)
    };
    let mut acc = [X87::ZERO; 4];
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        for blk in [0, 4] {
            acc[0] = acc[0].add(sq(x[i + blk + 3]));
            acc[1] = acc[1].add(sq(x[i + blk + 2]));
            acc[2] = acc[2].add(sq(x[i + blk + 1]));
            acc[3] = acc[3].add(sq(x[i + blk]));
        }
        i += 8;
    }
    while i < n {
        acc[3] = acc[3].add(sq(x[i]));
        i += 1;
    }
    let total = acc[0].add(acc[1].add(acc[3]).add(acc[2]));
    total.sqrt().to_f64()
}

/// `dlamch('E')`: half the machine epsilon.
const LAPACK_EPS: f64 = f64::EPSILON / 2.0;
/// `dlamch('S')`.
const LAPACK_SFMIN: f64 = f64::MIN_POSITIVE;

/// `dlapy2(x, y)`: `sqrt(x^2 + y^2)` without destructive overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    let (xa, ya) = (x.abs(), y.abs());
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        let t = z / w;
        w * (1.0 + t * t).sqrt()
    }
}

/// `dlarfg(n, alpha, x)`: the Householder reflector annihilating `x`; returns `(beta, tau)` and
/// overwrites `x` with the reflector's tail `v`.
fn dlarfg(alpha: f64, x: &mut [f64]) -> Option<(f64, f64)> {
    if x.is_empty() {
        return Some((alpha, 0.0));
    }
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return Some((alpha, 0.0));
    }
    let beta = -dlapy2(alpha, xnorm).copysign(alpha);
    if beta.abs() < LAPACK_SFMIN / LAPACK_EPS {
        // LAPACK rescales here; not reproduced.
        return None;
    }
    let tau = (beta - alpha) / beta;
    let s = 1.0 / (alpha - beta);
    for v in x.iter_mut() {
        *v *= s;
    }
    Some((beta, tau))
}

/// `dlarf('Left', m, n, v, incv, tau, C)` for the columns `cols` of `C` (`v[0] == 1`, any
/// stride): trailing zeros of `v` and zero columns of `C` trimmed (`iladlc`), `w = C^T v` by
/// `dgemv_t`, then `C -= tau v w^T` by `dger` (an FMA `daxpy` per column).
fn dlarf_left(v: &[f64], tau: f64, cols: &mut [&mut [f64]]) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = v.len();
    while lastv > 0 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    let mut lastc = cols.len();
    while lastc > 0 && cols[lastc - 1][..lastv].iter().all(|&t| t == 0.0) {
        lastc -= 1;
    }
    if lastv == 0 || lastc == 0 {
        return;
    }
    let mut ct = Mat::zeros(lastc, lastv);
    for (j, col) in cols.iter().take(lastc).enumerate() {
        ct.row_mut(j).copy_from_slice(&col[..lastv]);
    }
    let w = gemv_t(&ct, &v[..lastv]);
    for (j, col) in cols.iter_mut().take(lastc).enumerate() {
        let y0 = (-tau) * w[j];
        for i in 0..lastv {
            col[i] = y0.mul_add(v[i], col[i]);
        }
    }
}

/// `dlarf('Right', m, n, v, lda, tau, C, lda)` for the columns `cols` of `C` (one per entry of
/// `v`, `v[0] == 1`, `v` a matrix row): trailing zeros of `v` and zero rows of `C` trimmed
/// (`iladlr`), `w = C v` by `dgemv_n` (strided `x`, leading dimension `lda`), then
/// `C -= tau w v^T` by `dger`.
fn dlarf_right(v: &[f64], tau: f64, cols: &mut [&mut [f64]], lda: usize) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = v.len();
    while lastv > 0 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let m = cols.first().map_or(0, |c| c.len());
    let lastc = (0..m)
        .rev()
        .find(|&i| cols[..lastv].iter().any(|c| c[i] != 0.0))
        .map_or(0, |i| i + 1);
    if lastc == 0 {
        return;
    }
    // `gemv_n` reads column `j` of `C` as row `j` of a matrix whose row stride is `lda`.
    let mut c = Mat::zeros(lastv, lda.max(lastc));
    for (j, col) in cols.iter().take(lastv).enumerate() {
        c.row_mut(j)[..lastc].copy_from_slice(&col[..lastc]);
    }
    let w = gemv_n(&c, lastc, &v[..lastv], false);
    for (j, col) in cols.iter_mut().take(lastv).enumerate() {
        let y0 = (-tau) * v[j];
        for i in 0..lastc {
            col[i] = y0.mul_add(w[i], col[i]);
        }
    }
}

/// `dlasv2(f, g, h)`: the SVD of the 2x2 upper triangular `[[f, g], [0, h]]`; returns
/// `(ssmin, ssmax, snr, csr, snl, csl)`.
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (ssmin, ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut out = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if ga > fa {
            pmax = 2;
            if fa / ga < LAPACK_EPS {
                gasmal = false;
                let smin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                out = (smin, ga, 1.0, ft / gt, ht / gt, 1.0);
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            let smin = ha / a;
            let smax = fa * a;
            if mm == 0.0 {
                if l == 0.0 {
                    t = 2.0_f64.copysign(ft) * 1.0_f64.copysign(gt);
                } else {
                    t = gt / d.copysign(ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            let crt_v = 2.0 / l;
            let srt_v = t / l;
            let clt_v = (crt_v + srt_v * m) / a;
            let slt_v = (ht / ft) * srt_v / a;
            out = (smin, smax, clt_v, crt_v, slt_v, srt_v);
        }
        (ssmin, ssmax, clt, crt, slt, srt) = out;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let sg = |v: f64| 1.0_f64.copysign(v);
    let tsign = match pmax {
        1 => sg(csr) * sg(csl) * sg(f),
        2 => sg(snr) * sg(csl) * sg(g),
        _ => sg(snr) * sg(snl) * sg(h),
    };
    (
        ssmin.abs().copysign(tsign * sg(f) * sg(h)),
        ssmax.abs().copysign(tsign),
        snr,
        csr,
        snl,
        csl,
    )
}

/// OpenBLAS `drot` (strided kernel) on one pair.
fn drot_pair(x: f64, y: f64, c: f64, s: f64) -> (f64, f64) {
    (c.mul_add(x, s * y), c.mul_add(y, -(s * x)))
}

/// `dlartg(f, g)` (LAPACK 3.10+, `la_xlartg`): the plane rotation `(c, s, r)` with
/// `[c s; -s c] [f; g] = [r; 0]`.
fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    const SAFMIN: f64 = f64::MIN_POSITIVE;
    const SAFMAX: f64 = 1.0 / f64::MIN_POSITIVE;
    let rtmin = SAFMIN.sqrt();
    let rtmax = (SAFMAX / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, 1.0_f64.copysign(g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let r = d.copysign(f);
        (f1 / d, g / r, r)
    } else {
        let u = SAFMAX.min(SAFMIN.max(f1).max(g1));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let r = d.copysign(f);
        (fs.abs() / d, gs / r, r * u)
    }
}

/// `dlas2(f, g, h)`: the smaller singular value of `[[f, g], [0, h]]` (the larger is unused).
fn dlas2(f: f64, g: f64, h: f64) -> f64 {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        0.0
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let q = ga / fhmx;
        let au = q * q;
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        fhmn * c
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            (fhmn * fhmx) / ga
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p1 = as_ * au;
            let p2 = at * au;
            let c = 1.0 / ((1.0 + p1 * p1).sqrt() + (1.0 + p2 * p2).sqrt());
            let ssmin = (fhmn * c) * au;
            ssmin + ssmin
        }
    }
}

/// `dlasr('L', 'V', 'F' | 'B', ...)`: the plane rotations `(c[j], s[j])` applied to rows
/// `j, j + 1` of `rows`, forward or backward in `j`.
fn dlasr_left(forward: bool, c: &[f64], s: &[f64], rows: &mut [Vec<f64>]) {
    let k = rows.len().saturating_sub(1);
    for step in 0..k {
        let j = if forward { step } else { k - 1 - step };
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            let (lo, hi) = rows.split_at_mut(j + 1);
            let (a, b) = (&mut lo[j], &mut hi[0]);
            for i in 0..a.len() {
                let temp = b[i];
                b[i] = ct * temp - st * a[i];
                a[i] = st * temp + ct * a[i];
            }
        }
    }
}

/// `dbdsqr('U', n, ncvt = n, nru = 0, ncc = 1, d, e, VT, C)`: the SVD of the upper bidiagonal
/// `(d, e)` by implicit zero-shift and shifted QR sweeps, rotating the rows of `vt` and the
/// entries of `c`; singular values made positive and sorted decreasing. `None` if the
/// iteration does not converge (LAPACK's `info > 0`).
fn dbdsqr(d: &mut [f64], e: &mut [f64], vt: &mut [Vec<f64>], c: &mut [Vec<f64>]) -> Option<()> {
    let n = d.len();
    if n > 1 {
        let eps = LAPACK_EPS;
        let unfl = LAPACK_SFMIN;
        let tolmul = 10.0_f64.max(100.0_f64.min(eps.powf(-0.125)));
        let tol = tolmul * eps;
        let mut smax = 0.0_f64;
        for &v in d.iter() {
            smax = smax.max(v.abs());
        }
        for &v in e.iter() {
            smax = smax.max(v.abs());
        }
        let mut sminoa = d[0].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            for i in 1..n {
                mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == 0.0 {
                    break;
                }
            }
        }
        sminoa /= (n as f64).sqrt();
        let nf = n as f64;
        let thresh = (tol * sminoa).max(6.0 * (nf * (nf * unfl)));
        let maxitdivn = 6 * n;
        let mut iterdivn = 0;
        let mut iter = 0_usize;
        let mut iter_started = false;
        let (mut oldll, mut oldm) = (0_usize, 0_usize);
        let mut have_old = false;
        let mut idir = 0;
        let mut smin;
        // 1-based indices, as in LAPACK: d(i) is d[i - 1], e(i) is e[i - 1].
        let mut m = n;
        let mut cs_r = vec![0.0_f64; n];
        let mut sn_r = vec![0.0_f64; n];
        let mut cs_l = vec![0.0_f64; n];
        let mut sn_l = vec![0.0_f64; n];
        'outer: loop {
            if m <= 1 {
                break;
            }
            // `iter` starts at -1 in LAPACK; `iter_started` tracks whether it reached 0.
            if iter_started && iter >= n {
                iter -= n;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    return None;
                }
            }
            smax = d[m - 1].abs();
            let mut split = None;
            for lll in 1..m {
                let ll = m - lll;
                let abss = d[ll - 1].abs();
                let abse = e[ll - 1].abs();
                if abse <= thresh {
                    split = Some(ll);
                    break;
                }
                smax = smax.max(abss).max(abse);
            }
            let mut ll = match split {
                Some(ll) => {
                    e[ll - 1] = 0.0;
                    if ll == m - 1 {
                        m -= 1;
                        continue 'outer;
                    }
                    ll
                }
                None => 0,
            };
            ll += 1;
            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, sinl, cosl) = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
                d[m - 2] = sigmx;
                e[m - 2] = 0.0;
                d[m - 1] = sigmn;
                rotate_rows(vt, m - 2, cosr, sinr);
                rotate_rows(c, m - 2, cosl, sinl);
                m -= 2;
                continue 'outer;
            }
            if !have_old || ll > oldm || m < oldll {
                idir = if d[ll - 1].abs() >= d[m - 1].abs() {
                    1
                } else {
                    2
                };
            }
            if idir == 1 {
                if e[m - 2].abs() <= tol.abs() * d[m - 1].abs() {
                    e[m - 2] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[ll - 1].abs();
                smin = mu;
                for lll in ll..m {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = 0.0;
                        continue 'outer;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = smin.min(mu);
                }
            } else {
                if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs() {
                    e[ll - 1] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[m - 1].abs();
                smin = mu;
                for lll in (ll..m).rev() {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = 0.0;
                        continue 'outer;
                    }
                    mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = smin.min(mu);
                }
            }
            oldll = ll;
            oldm = m;
            have_old = true;
            let mut shift;
            if nf * tol * (smin / smax) <= eps.max(0.01 * tol) {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = d[ll - 1].abs();
                    shift = dlas2(d[m - 2], e[m - 2], d[m - 1]);
                } else {
                    sll = d[m - 1].abs();
                    shift = dlas2(d[ll - 1], e[ll - 1], d[ll]);
                }
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }
            if iter_started {
                iter += m - ll;
            } else {
                iter = m - ll - 1;
                iter_started = true;
            }
            let nrot = m - ll;
            if shift == 0.0 {
                if idir == 1 {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..m {
                        let (c1, sn, r) = dlartg(d[i - 1] * cs, e[i - 1]);
                        cs = c1;
                        if i > ll {
                            e[i - 2] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        let k = i - ll;
                        (cs_r[k], sn_r[k], cs_l[k], sn_l[k]) = (cs, sn, oldcs, oldsn);
                    }
                    let h = d[m - 1] * cs;
                    d[m - 1] = h * oldcs;
                    e[m - 2] = h * oldsn;
                    dlasr_left(true, &cs_r[..nrot], &sn_r[..nrot], &mut vt[ll - 1..m]);
                    dlasr_left(true, &cs_l[..nrot], &sn_l[..nrot], &mut c[ll - 1..m]);
                    if e[m - 2].abs() <= thresh {
                        e[m - 2] = 0.0;
                    }
                } else {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in (ll + 1..=m).rev() {
                        let (c1, sn, r) = dlartg(d[i - 1] * cs, e[i - 2]);
                        cs = c1;
                        if i < m {
                            e[i - 1] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i - 2] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        let k = i - ll - 1;
                        (cs_r[k], sn_r[k], cs_l[k], sn_l[k]) = (cs, -sn, oldcs, -oldsn);
                    }
                    let h = d[ll - 1] * cs;
                    d[ll - 1] = h * oldcs;
                    e[ll - 1] = h * oldsn;
                    dlasr_left(false, &cs_l[..nrot], &sn_l[..nrot], &mut vt[ll - 1..m]);
                    dlasr_left(false, &cs_r[..nrot], &sn_r[..nrot], &mut c[ll - 1..m]);
                    if e[ll - 1].abs() <= thresh {
                        e[ll - 1] = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f =
                    (d[ll - 1].abs() - shift) * (1.0_f64.copysign(d[ll - 1]) + shift / d[ll - 1]);
                let mut g = e[ll - 1];
                for i in ll..m {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        e[i - 2] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i - 1];
                    g = sinr * d[i];
                    d[i] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 1] + sinl * d[i];
                    d[i] = cosl * d[i] - sinl * e[i - 1];
                    if i < m - 1 {
                        g = sinl * e[i];
                        e[i] *= cosl;
                    }
                    let k = i - ll;
                    (cs_r[k], sn_r[k], cs_l[k], sn_l[k]) = (cosr, sinr, cosl, sinl);
                }
                e[m - 2] = f;
                dlasr_left(true, &cs_r[..nrot], &sn_r[..nrot], &mut vt[ll - 1..m]);
                dlasr_left(true, &cs_l[..nrot], &sn_l[..nrot], &mut c[ll - 1..m]);
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = 0.0;
                }
            } else {
                let mut f =
                    (d[m - 1].abs() - shift) * (1.0_f64.copysign(d[m - 1]) + shift / d[m - 1]);
                let mut g = e[m - 2];
                for i in (ll + 1..=m).rev() {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 2];
                    e[i - 2] = cosr * e[i - 2] - sinr * d[i - 1];
                    g = sinr * d[i - 2];
                    d[i - 2] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 2] + sinl * d[i - 2];
                    d[i - 2] = cosl * d[i - 2] - sinl * e[i - 2];
                    if i > ll + 1 {
                        g = sinl * e[i - 3];
                        e[i - 3] *= cosl;
                    }
                    let k = i - ll - 1;
                    (cs_r[k], sn_r[k], cs_l[k], sn_l[k]) = (cosr, -sinr, cosl, -sinl);
                }
                e[ll - 1] = f;
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = 0.0;
                }
                dlasr_left(false, &cs_l[..nrot], &sn_l[..nrot], &mut vt[ll - 1..m]);
                dlasr_left(false, &cs_r[..nrot], &sn_r[..nrot], &mut c[ll - 1..m]);
            }
        }
    }
    // Make the singular values positive (`dscal` by -1 on the rows of VT).
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for v in &mut vt[i] {
                *v *= -1.0;
            }
        }
    }
    // Sort decreasing: the smallest of the leading part moves to the end.
    for i in 1..n {
        let tgt = n - i;
        let mut isub = 0;
        let mut smin = d[0];
        for j in 1..=tgt {
            if d[j] <= smin {
                isub = j;
                smin = d[j];
            }
        }
        if isub != tgt {
            d[isub] = d[tgt];
            d[tgt] = smin;
            vt.swap(isub, tgt);
            c.swap(isub, tgt);
        }
    }
    Some(())
}

/// `drot` on rows `k` and `k + 1` (every column).
fn rotate_rows(rows: &mut [Vec<f64>], k: usize, c: f64, s: f64) {
    let (lo, hi) = rows.split_at_mut(k + 1);
    for (x, y) in lo[k].iter_mut().zip(hi[0].iter_mut()) {
        (*x, *y) = drot_pair(*x, *y, c, s);
    }
}

/// `dlalsd('U', smlsiz, n, 1, d, e, b, ldb, rcond)` for `n <= smlsiz = 25`: scale by the largest
/// entry, `dlasdq` (`dbdsqr`, then sorted increasing), the `rcond` truncation, `x = V b` by
/// `dgemm` (an FMA chain in `k` order per entry), unscale.
fn dlalsd(d: &[f64], e: &[f64], b: &[f64], rcond: f64) -> Option<Vec<f64>> {
    let n = d.len();
    if n == 1 {
        if d[0] == 0.0 {
            return Some(vec![0.0]);
        }
        return Some(vec![b[0] * (1.0 / d[0])]);
    }
    let mut orgnrm = d[n - 1].abs();
    for i in 0..n - 1 {
        orgnrm = orgnrm.max(d[i].abs()).max(e[i].abs());
    }
    if orgnrm == 0.0 {
        return Some(vec![0.0; n]);
    }
    let mul = 1.0 / orgnrm;
    let mut d: Vec<f64> = d.iter().map(|v| v * mul).collect();
    let mut e: Vec<f64> = e.iter().map(|v| v * mul).collect();
    let mut vt: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut c: Vec<Vec<f64>> = b.iter().map(|&v| vec![v]).collect();
    // dlasdq: dbdsqr, then an increasing selection sort.
    dbdsqr(&mut d, &mut e, &mut vt, &mut c)?;
    for i in 0..n {
        let mut isub = i;
        let mut smin = d[i];
        for j in i + 1..n {
            if d[j] < smin {
                isub = j;
                smin = d[j];
            }
        }
        if isub != i {
            d[isub] = d[i];
            d[i] = smin;
            vt.swap(isub, i);
            c.swap(isub, i);
        }
    }
    let mut imax = 0;
    for i in 1..n {
        if d[i].abs() > d[imax].abs() {
            imax = i;
        }
    }
    let tol = rcond * d[imax].abs();
    let bb: Vec<f64> = (0..n)
        .map(|i| {
            if d[i] <= tol {
                0.0
            } else {
                c[i][0] * (1.0 / d[i])
            }
        })
        .collect();
    let x = (0..n)
        .map(|i| {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc = vt[k][i].mul_add(bb[k], acc);
            }
            acc * mul
        })
        .collect();
    Some(x)
}

/// `dlalsd('L', ...)`: the lower bidiagonal `(d, e)` rotated to upper (`dlartg`, the rotations
/// applied to `b` by unit-stride `drot`), then solved as `dlalsd('U', ...)`.
fn dlalsd_lower(d: &[f64], e: &[f64], b: &[f64], rcond: f64) -> Option<Vec<f64>> {
    let n = d.len();
    if n == 1 {
        return dlalsd(d, e, b, rcond);
    }
    let mut d = d.to_vec();
    let mut e = e.to_vec();
    let mut b = b.to_vec();
    for i in 0..n - 1 {
        let (cs, sn, r) = dlartg(d[i], e[i]);
        d[i] = r;
        e[i] = sn * d[i + 1];
        d[i + 1] *= cs;
        (b[i], b[i + 1]) = drot_pair(b[i], b[i + 1], cs, sn);
    }
    dlalsd(&d, &e, &b, rcond)
}

/// `np.linalg.lstsq(A, b, rcond=None)[0]` for an `m x n` `A` with `min(m, n) <= 25`: LAPACK
/// `dgelsd` as OpenBLAS builds it, for every path it takes there. With `m >= n`: Householder QR
/// (`dgeqr2`, `dorm2r` on `b`) when `m >= 1.6 n`, then bidiagonalization (`dgebd2`, `dorm2r`
/// with its left reflectors), `dlalsd`'s direct small-matrix SVD solve and the right
/// reflectors (`dorml2`). With `m < n`: LQ (`dgelq2`) and the same solve on `L` when
/// `n >= 1.6 m`, else a lower bidiagonalization of `A` itself; the minimum-norm solution either
/// way. `None` for what it does not reproduce (`min(m, n) > 25`, where `dlalsd` divides and
/// conquers; non-finite data; LAPACK's rescaling of tiny or huge data; a non-converging SVD),
/// where the caller uses its own solver.
pub(crate) fn lstsq(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let (m, n) = (a.rows, a.cols);
    if n == 0 || m == 0 || m.min(n) > 25 || b.len() != m {
        return None;
    }
    if a.data.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    // dgelsd's scaling of A and b into [smlnum, bignum] is not reproduced.
    let smlnum = LAPACK_SFMIN / LAPACK_EPS;
    let bignum = 1.0 / smlnum;
    let anrm = a.data.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let bnrm = b.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if anrm == 0.0 {
        return Some(vec![0.0; n]);
    }
    if anrm < smlnum || anrm > bignum || (bnrm > 0.0 && bnrm < smlnum) || bnrm > bignum {
        return None;
    }
    // NumPy's Fortran copy of A has leading dimension m; rcond = eps * max(m, n).
    let lda = m;
    let rcond = f64::EPSILON * (m.max(n) as f64);
    // ilaenv(6, 'DGELSD'): int(real(min(m, n)) * 1.6e0), exact for min(m, n) <= 25.
    let mnthr = (16 * m.min(n)) / 10;
    // Column-major working copy: cols[j] is column j of A.
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| a.col(j)).collect();
    if m < n {
        return lstsq_under(cols, b, lda, rcond, n >= mnthr);
    }
    let mut bv = b.to_vec();
    let mm = if m >= mnthr {
        // dgeqr2.
        let mut tau = vec![0.0_f64; n];
        for i in 0..n {
            let (head, tail) = cols.split_at_mut(i + 1);
            let col = &mut head[i];
            let (beta, t) = {
                let (diag, below) = col[i..m].split_at_mut(1);
                dlarfg(diag[0], below)?
            };
            col[i] = 1.0;
            let mut sub: Vec<&mut [f64]> = tail.iter_mut().map(|c| &mut c[i..m]).collect();
            dlarf_left(&col[i..m], t, &mut sub);
            col[i] = beta;
            tau[i] = t;
        }
        // dorm2r('L', 'T') on b.
        for i in 0..n {
            let mut v = vec![1.0];
            v.extend_from_slice(&cols[i][i + 1..m]);
            dlarf_left(&v, tau[i], &mut [&mut bv[i..m]]);
        }
        // dlaset: zero below R.
        for (j, col) in cols.iter_mut().enumerate() {
            for v in &mut col[j + 1..m] {
                *v = 0.0;
            }
        }
        n
    } else {
        m
    };
    solve_upper(&mut cols, mm, &mut bv, lda, rcond)
}

/// `dgelsd` from its `dgebrd` on: `dgebd2` on the leading `mm x n` block of `cols` (`mm >= n`,
/// upper bidiagonal), `dormbr('Q', 'L', 'T')` on `bv[..mm]`, `dlalsd('U')`, and
/// `dormbr('P', 'L', 'N')`. Returns the `n` solution entries.
fn solve_upper(
    cols: &mut [Vec<f64>],
    mm: usize,
    bv: &mut [f64],
    lda: usize,
    rcond: f64,
) -> Option<Vec<f64>> {
    let n = cols.len();
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n - 1];
    let mut tauq = vec![0.0_f64; n];
    let mut taup = vec![0.0_f64; n];
    for i in 0..n {
        {
            let (head, tail) = cols.split_at_mut(i + 1);
            let col = &mut head[i];
            let (beta, t) = {
                let (diag, below) = col[i..mm].split_at_mut(1);
                dlarfg(diag[0], below)?
            };
            d[i] = beta;
            tauq[i] = t;
            col[i] = 1.0;
            let mut sub: Vec<&mut [f64]> = tail.iter_mut().map(|c| &mut c[i..mm]).collect();
            dlarf_left(&col[i..mm], t, &mut sub);
            col[i] = beta;
        }
        if i + 1 < n {
            let mut row: Vec<f64> = (i + 2..n).map(|j| cols[j][i]).collect();
            let (beta, t) = dlarfg(cols[i + 1][i], &mut row)?;
            for (k, j) in (i + 2..n).enumerate() {
                cols[j][i] = row[k];
            }
            e[i] = beta;
            taup[i] = t;
            let mut v = vec![1.0];
            v.extend_from_slice(&row);
            let mut sub: Vec<&mut [f64]> = cols[i + 1..n]
                .iter_mut()
                .map(|c| &mut c[i + 1..mm])
                .collect();
            dlarf_right(&v, t, &mut sub, lda);
            cols[i + 1][i] = beta;
        }
    }
    // dormbr('Q', 'L', 'T') = dorm2r with the left reflectors.
    for i in 0..n {
        let mut v = vec![1.0];
        v.extend_from_slice(&cols[i][i + 1..mm]);
        dlarf_left(&v, tauq[i], &mut [&mut bv[i..mm]]);
    }
    let mut x = dlalsd(&d, &e, &bv[..n], rcond)?;
    // dormbr('P', 'L', 'N') = dorml2('L', 'T') with the right reflectors, last to first.
    for i in (0..n.saturating_sub(1)).rev() {
        let mut v = vec![1.0];
        v.extend((i + 2..n).map(|j| cols[j][i]));
        dlarf_left(&v, taup[i], &mut [&mut x[i + 1..n]]);
    }
    Some(x)
}

/// `dgelsd` for `m < n` (`cols`: the `n` columns of length `m`): the minimum-norm solution.
/// With `lq` (`n >= 1.6 m`), `dgelq2`, the square solve on `L`, and `dorml2('L', 'T')` with
/// the LQ reflectors; otherwise `dgebd2`'s lower bidiagonalization of `A`, `dormbr('Q')` on
/// `b[1..]`, `dlalsd('L')`, and `dorml2('L', 'T')` with the right reflectors.
fn lstsq_under(
    mut cols: Vec<Vec<f64>>,
    b: &[f64],
    lda: usize,
    rcond: f64,
    lq: bool,
) -> Option<Vec<f64>> {
    let n = cols.len();
    let m = b.len();
    let mut bv = vec![0.0_f64; n];
    bv[..m].copy_from_slice(b);
    // The row reflector of row i: entries i + 1.. of row i are its tail.
    let row_reflector = |cols: &mut [Vec<f64>], i: usize| -> Option<(f64, f64, Vec<f64>)> {
        let mut row: Vec<f64> = (i + 1..n).map(|j| cols[j][i]).collect();
        let (beta, t) = dlarfg(cols[i][i], &mut row)?;
        for (k, j) in (i + 1..n).enumerate() {
            cols[j][i] = row[k];
        }
        cols[i][i] = beta;
        let mut v = vec![1.0];
        v.extend_from_slice(&row);
        Some((beta, t, v))
    };
    let mut taup = vec![0.0_f64; m];
    if lq {
        // dgelq2.
        for i in 0..m {
            let (_, t, v) = row_reflector(&mut cols, i)?;
            taup[i] = t;
            if i + 1 < m {
                let mut sub: Vec<&mut [f64]> =
                    cols[i..n].iter_mut().map(|c| &mut c[i + 1..m]).collect();
                dlarf_right(&v, t, &mut sub, lda);
            }
        }
        // L (lower triangular, m x m) into a work array with leading dimension m.
        let mut l: Vec<Vec<f64>> = (0..m)
            .map(|j| {
                (0..m)
                    .map(|i| if i >= j { cols[j][i] } else { 0.0 })
                    .collect()
            })
            .collect();
        let x = solve_upper(&mut l, m, &mut bv[..m], m, rcond)?;
        bv[..m].copy_from_slice(&x);
        for v in &mut bv[m..] {
            *v = 0.0;
        }
    } else {
        // dgebd2 with m < n: lower bidiagonal.
        let mut d = vec![0.0_f64; m];
        let mut e = vec![0.0_f64; m - 1];
        let mut tauq = vec![0.0_f64; m];
        for i in 0..m {
            let (beta, t, v) = row_reflector(&mut cols, i)?;
            d[i] = beta;
            taup[i] = t;
            if i + 1 < m {
                let mut sub: Vec<&mut [f64]> =
                    cols[i..n].iter_mut().map(|c| &mut c[i + 1..m]).collect();
                dlarf_right(&v, t, &mut sub, lda);
                let (beta, t) = {
                    let (diag, below) = cols[i][i + 1..m].split_at_mut(1);
                    dlarfg(diag[0], below)?
                };
                e[i] = beta;
                tauq[i] = t;
                let mut v = vec![1.0];
                v.extend_from_slice(&cols[i][i + 2..m]);
                let mut sub: Vec<&mut [f64]> = cols[i + 1..n]
                    .iter_mut()
                    .map(|c| &mut c[i + 1..m])
                    .collect();
                dlarf_left(&v, t, &mut sub);
                cols[i][i + 1] = beta;
            }
        }
        // dormbr('Q', 'L', 'T') with nq = m < k = n: dorm2r on rows 2..m.
        for i in 0..m - 1 {
            let mut v = vec![1.0];
            v.extend_from_slice(&cols[i][i + 2..m]);
            dlarf_left(&v, tauq[i], &mut [&mut bv[i + 1..m]]);
        }
        let x = dlalsd_lower(&d, &e, &bv[..m], rcond)?;
        bv[..m].copy_from_slice(&x);
        for v in &mut bv[m..] {
            *v = 0.0;
        }
    }
    // dorml2('L', 'T', n, 1, m) with the row reflectors, last to first.
    for i in (0..m).rev() {
        let mut v = vec![1.0];
        v.extend((i + 1..n).map(|j| cols[j][i]));
        dlarf_left(&v, taup[i], &mut [&mut bv[i..n]]);
    }
    Some(bv)
}

#[cfg(test)]
mod tests {
    //! Expected values are NumPy 2.4.3's (`np.dot`, `np.linalg.lstsq(A, b, rcond=None)[0]`,
    //! OpenBLAS 0.3.31 SkylakeX kernels), compared bit for bit.
    use super::*;

    fn mat(rows: &[&[f64]]) -> Mat {
        let mut m = Mat::zeros(rows.len(), rows[0].len());
        for (i, r) in rows.iter().enumerate() {
            m.row_mut(i).copy_from_slice(r);
        }
        m
    }

    fn assert_bits(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.to_bits(), e.to_bits(), "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn ddot_follows_the_blas_kernel_order() {
        let x: Vec<f64> = (0..37).map(|i| ((i * 7) % 13 - 6) as f64 / 7.0).collect();
        let y: Vec<f64> = (0..37).map(|i| ((i * 5) % 11 - 5) as f64 / 3.0).collect();
        // A left-to-right sum of products gives -12.095238095238093 here.
        assert_eq!(
            dot(&x, &y, true).to_bits(),
            (-12.095_238_095_238_095_f64).to_bits()
        );
    }

    #[test]
    fn lstsq_matches_dgelsd() {
        // m >= 1.6 n: QR first.
        let a = mat(&[
            &[1.5, -0.3, 2.0],
            &[0.7, 1.1, -0.4],
            &[-2.2, 0.9, 0.6],
            &[0.3, 0.25, 1.7],
            &[1.0, -1.9, 0.8],
        ]);
        let x = lstsq(&a, &[0.4, -1.3, 2.1, 0.6, -0.9]).expect("supported");
        assert_bits(
            &x,
            &[
                -0.830_982_459_379_769_3,
                0.058_695_252_834_130_35,
                0.649_140_545_737_84,
            ],
        );
        // Square: bidiagonalization of A itself.
        let a = mat(&[
            &[2.0, -1.0, 0.5, 0.0],
            &[0.3, 1.7, -0.6, 1.2],
            &[-0.8, 0.4, 3.1, -0.2],
            &[1.1, 0.9, 0.7, -2.4],
        ]);
        let x = lstsq(&a, &[1.0, -0.5, 0.25, 2.0]).expect("supported");
        assert_bits(
            &x,
            &[
                0.488_249_089_366_797_95,
                0.058_629_808_891_771_94,
                0.164_263_260_316_351_6,
                -0.539_656_204_780_200_5,
            ],
        );
        // Rank deficient (third column twice the first): the minimum-norm solution.
        let a = mat(&[
            &[1.0, 2.0, 2.0],
            &[0.5, -1.0, 1.0],
            &[2.0, 0.3, 4.0],
            &[-1.0, 0.7, -2.0],
        ]);
        let x = lstsq(&a, &[0.3, 1.1, -0.7, 0.2]).expect("supported");
        assert_bits(
            &x,
            &[
                -0.020_580_282_545_951_67,
                -0.076_332_978_885_006_88,
                -0.041_160_565_091_903_41,
            ],
        );
    }

    #[test]
    fn lstsq_matches_dgelsd_underdetermined() {
        // n >= 1.6 m: LQ first.
        let a = mat(&[&[1.5, -0.3, 2.0, 0.7, 1.1], &[-0.4, -2.2, 0.9, 0.6, 0.3]]);
        let x = lstsq(&a, &[0.4, -1.3]).expect("supported");
        assert_bits(
            &x,
            &[
                0.308_925_210_773_398_85,
                0.540_510_543_840_177_6,
                0.033_264_167_028_583_376,
                -0.063_644_489_789_352_98,
                0.069_807_414_352_123_16,
            ],
        );
        // n < 1.6 m: lower bidiagonalization of A itself.
        let a = mat(&[
            &[2.0, -1.0, 0.5, 0.0, 1.3],
            &[0.3, 1.7, -0.6, 1.2, -0.9],
            &[-0.8, 0.4, 3.1, -0.2, 0.5],
            &[1.1, 0.9, 0.7, -2.4, 0.6],
        ]);
        let x = lstsq(&a, &[1.0, -0.5, 0.25, 2.0]).expect("supported");
        assert_bits(
            &x,
            &[
                0.411_775_821_922_350_1,
                0.138_242_356_454_385_68,
                0.103_409_677_935_728_35,
                -0.512_026_502_081_026_2,
                0.202_296_825_878_324_43,
            ],
        );
    }

    #[test]
    fn lstsq_declines_what_it_does_not_reproduce() {
        assert!(lstsq(&Mat::zeros(30, 26), &[0.0; 30]).is_none());
        assert!(lstsq(&Mat::zeros(26, 30), &[0.0; 26]).is_none());
        assert!(lstsq(&mat(&[&[f64::NAN], &[1.0]]), &[1.0, 1.0]).is_none());
        assert!(lstsq(&mat(&[&[1e-310], &[1e-310]]), &[1.0, 1.0]).is_none());
        assert_bits(
            &lstsq(&Mat::zeros(3, 2), &[1.0, 2.0, 3.0]).expect("zero A"),
            &[0.0, 0.0],
        );
    }
}
