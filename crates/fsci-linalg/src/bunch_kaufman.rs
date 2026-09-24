//! Bunch–Kaufman symmetric indefinite factorization, `A = U·D·Uᵀ` or `A = L·D·Lᵀ` with `D`
//! block diagonal (1×1 and 2×2 blocks): LAPACK's `dsytf2`, `dsytrs` and `dsytri`, the routines
//! behind SciPy's `solve` / `inv` for a symmetric matrix that is not positive definite and for
//! `assume_a='sym'`, and behind `scipy.linalg.ldl`.
//!
//! Each routine is a line-for-line transcription of the reference LAPACK source, with its BLAS
//! calls (`idamax`, `dswap`, `dsyr`, `dscal`, `dger`, `dgemv`, `dsymv`, `ddot`) inlined in the
//! reference BLAS operation order. The pivots are therefore LAPACK's, and the factor agrees
//! with OpenBLAS's to the rounding of its BLAS kernels, which differ from each other by
//! OpenBLAS kernel (FMA or not, lane-parallel dot products). The indices below are LAPACK's
//! 1-based ones so the transcription can be checked against the source; `at(i, j)` is
//! column-major storage.
//!
//! Measured against SciPy 1.17.1's own `dsytf2` / `dsytrs` / `dsytri` on 60 symmetric
//! matrices (n = 1..64; random, zero-diagonal, KKT, cond up to 1e12, singular), both triangles,
//! under OpenBLAS's Prescott, Nehalem, Sandybridge, Haswell and Zen kernels
//! (frankenscipy-7tb8d.15):
//! - the pivots and the singular flag agree on all 120 factorizations;
//! - the factor is bit-identical on every non-FMA kernel (120/120), and on Haswell / Zen
//!   wherever `dsyr`'s fused multiply-add does not round differently (97/120);
//! - solves are bit-identical up to n = 4, and a few ulps apart above it, where OpenBLAS's
//!   `dgemv` splits the dot product into lanes;
//! - on the ill-conditioned cases the solution is within 2.2e-15 (relative) of SciPy's
//!   `solve`, where SciPy's own LU (`assume_a='gen'`) is 1.5e-12 to 3.3e-6 away.
//!
//! `dsytrf` factors the whole matrix with `dsytf2` when `n` is at most its block size (64), and
//! with the blocked `dlasyf` above it: the same pivot rule with delayed updates (the pivots
//! agreed on every case measured up to n = 300), so the result differs from this one only by
//! rounding. That rounding is as large as SciPy's own spread across kernels there (2e-8 to
//! 1.3e-7 relative at cond 1e10).

/// Which triangle of the symmetric matrix is read and factored (LAPACK's `UPLO`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Triangle {
    /// `A = U·D·Uᵀ`, read from the upper triangle (SciPy's `solve` / `inv` default).
    Upper,
    /// `A = L·D·Lᵀ`, read from the lower triangle (`scipy.linalg.ldl`'s default).
    Lower,
}

/// A `dsytf2` factorization: the factor in LAPACK's packed form plus its pivots.
#[derive(Clone, Debug)]
pub(crate) struct BunchKaufman {
    n: usize,
    triangle: Triangle,
    /// Column-major `n × n`; the `triangle` half holds `D` and the multipliers, as `dsytf2`
    /// leaves them. The other half is untouched input.
    a: Vec<f64>,
    /// LAPACK's `IPIV`: `ipiv[k-1] = p > 0` is a 1×1 block with rows `k` and `p` interchanged;
    /// `ipiv[k-1] = ipiv[k∓1-1] = -p` is a 2×2 block with row `k-1` (upper) / `k+1` (lower)
    /// and `p` interchanged.
    ipiv: Vec<isize>,
    /// LAPACK's `INFO`: 0, or the 1-based index of the first exactly zero `D(k,k)`, which
    /// makes `D` (and `A`) singular.
    info: usize,
}

/// The reference `IDAMAX` over the `len` values `value(1..=len)`: the 1-based index of the
/// first entry of largest magnitude.
fn idamax(len: usize, value: impl Fn(usize) -> f64) -> usize {
    let mut imax = 1;
    let mut dmax = value(1).abs();
    for t in 2..=len {
        let v = value(t).abs();
        if v > dmax {
            imax = t;
            dmax = v;
        }
    }
    imax
}

impl BunchKaufman {
    /// `dsytf2(uplo, n, a)` on the square matrix `rows`, reading only `triangle`.
    pub(crate) fn factor(rows: &[Vec<f64>], triangle: Triangle) -> Self {
        let n = rows.len();
        let mut a = vec![0.0; n * n];
        for (i, row) in rows.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                a[i + j * n] = value;
            }
        }
        let mut ipiv = vec![0_isize; n];
        let info = match triangle {
            Triangle::Upper => sytf2_upper(&mut a, n, &mut ipiv),
            Triangle::Lower => sytf2_lower(&mut a, n, &mut ipiv),
        };
        Self {
            n,
            triangle,
            a,
            ipiv,
            info,
        }
    }

    /// `D` has an exactly zero diagonal entry (`dsytrf` INFO > 0): SciPy reports the matrix
    /// singular.
    pub(crate) fn is_singular(&self) -> bool {
        self.info > 0
    }

    /// LAPACK's `IPIV` (see the field).
    pub(crate) fn ipiv(&self) -> &[isize] {
        &self.ipiv
    }

    /// Entry `(i, j)` (0-based) of the factored storage: `D` and the multipliers in the
    /// factored triangle.
    pub(crate) fn packed(&self, i: usize, j: usize) -> f64 {
        self.a[i + j * self.n]
    }

    /// `dsytrs(uplo, n, 1, a, ipiv, b)`: `x` with `A·x = b`. Callers check
    /// [`Self::is_singular`] first, as `dsytrf`'s INFO > 0 stops SciPy before it solves.
    pub(crate) fn solve(&self, b: &[f64]) -> Vec<f64> {
        let mut x = b.to_vec();
        match self.triangle {
            Triangle::Upper => sytrs_upper(&self.a, self.n, &self.ipiv, &mut x),
            Triangle::Lower => sytrs_lower(&self.a, self.n, &self.ipiv, &mut x),
        }
        x
    }

    /// `dsytri(uplo, n, a, ipiv)` followed by SciPy's mirror of the computed triangle: the
    /// inverse as rows, exactly symmetric. Callers check [`Self::is_singular`] first.
    pub(crate) fn inverse(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut a = self.a.clone();
        match self.triangle {
            Triangle::Upper => sytri_upper(&mut a, n, &self.ipiv),
            Triangle::Lower => sytri_lower(&mut a, n, &self.ipiv),
        }
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let (r, c) = match self.triangle {
                            Triangle::Upper => (i.min(j), i.max(j)),
                            Triangle::Lower => (i.max(j), i.min(j)),
                        };
                        a[r + c * n]
                    })
                    .collect()
            })
            .collect()
    }
}

/// `(1 + √17) / 8`, the Bunch–Kaufman growth bound `dsytf2` pivots against.
fn bunch_kaufman_alpha() -> f64 {
    (1.0 + 17.0_f64.sqrt()) / 8.0
}

/// `dsytf2('U')`. Returns INFO.
fn sytf2_upper(a: &mut [f64], n: usize, ipiv: &mut [isize]) -> usize {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    let alpha = bunch_kaufman_alpha();
    let mut info = 0;
    let mut k = n;
    while k >= 1 {
        let mut kstep = 1;
        let absakk = a[at(k, k)].abs();
        let (imax, colmax) = if k > 1 {
            let imax = idamax(k - 1, |t| a[at(t, k)]);
            (imax, a[at(imax, k)].abs())
        } else {
            (0, 0.0)
        };
        let kp;
        if absakk.max(colmax) == 0.0 || absakk.is_nan() {
            if info == 0 {
                info = k;
            }
            kp = k;
        } else {
            if absakk >= alpha * colmax {
                kp = k;
            } else {
                let mut jmax = imax + idamax(k - imax, |t| a[at(imax, imax + t)]);
                let mut rowmax = a[at(imax, jmax)].abs();
                if imax > 1 {
                    jmax = idamax(imax - 1, |t| a[at(t, imax)]);
                    rowmax = rowmax.max(a[at(jmax, imax)].abs());
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a[at(imax, imax)].abs() >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + 1 - kstep;
            if kp != kk {
                for i in 1..kp {
                    a.swap(at(i, kk), at(i, kp));
                }
                for t in 1..(kk - kp) {
                    a.swap(at(kp + t, kk), at(kp, kp + t));
                }
                a.swap(at(kk, kk), at(kp, kp));
                if kstep == 2 {
                    a.swap(at(k - 1, k), at(kp, k));
                }
            }
            if kstep == 1 {
                // DSYR('U', k-1, -r1, A(1,k)) then DSCAL(k-1, r1, A(1,k)). Column k is
                // read-only while columns 1..k-1 are updated, so it is split off.
                let r1 = 1.0 / a[at(k, k)];
                let (left, right) = a.split_at_mut((k - 1) * n);
                let x = &mut right[..k - 1];
                for j in 1..k {
                    let xj = x[j - 1];
                    if xj != 0.0 {
                        let temp = -r1 * xj;
                        let col = &mut left[(j - 1) * n..(j - 1) * n + j];
                        for (aij, &xi) in col.iter_mut().zip(&x[..j]) {
                            *aij += xi * temp;
                        }
                    }
                }
                for xi in x.iter_mut() {
                    *xi *= r1;
                }
            } else if k > 2 {
                let mut d12 = a[at(k - 1, k)];
                let d22 = a[at(k - 1, k - 1)] / d12;
                let d11 = a[at(k, k)] / d12;
                let t = 1.0 / (d11 * d22 - 1.0);
                d12 = t / d12;
                // Columns k-1 and k are read-only while columns 1..k-2 are updated; the
                // multipliers are written back after each column's update, as LAPACK does.
                let (left, right) = a.split_at_mut((k - 2) * n);
                let (xkm1, xk) = right.split_at_mut(n);
                for j in (1..=k - 2).rev() {
                    let wkm1 = d12 * (d11 * xkm1[j - 1] - xk[j - 1]);
                    let wk = d12 * (d22 * xk[j - 1] - xkm1[j - 1]);
                    let col = &mut left[(j - 1) * n..(j - 1) * n + j];
                    for ((aij, &uik), &uikm1) in col.iter_mut().zip(&xk[..j]).zip(&xkm1[..j]) {
                        *aij = *aij - uik * wk - uikm1 * wkm1;
                    }
                    xk[j - 1] = wk;
                    xkm1[j - 1] = wkm1;
                }
            }
        }
        let kp = kp as isize;
        if kstep == 1 {
            ipiv[k - 1] = kp;
        } else {
            ipiv[k - 1] = -kp;
            ipiv[k - 2] = -kp;
        }
        k -= kstep;
    }
    info
}

/// `dsytf2('L')`. Returns INFO.
fn sytf2_lower(a: &mut [f64], n: usize, ipiv: &mut [isize]) -> usize {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    let alpha = bunch_kaufman_alpha();
    let mut info = 0;
    let mut k = 1;
    while k <= n {
        let mut kstep = 1;
        let absakk = a[at(k, k)].abs();
        let (imax, colmax) = if k < n {
            let imax = k + idamax(n - k, |t| a[at(k + t, k)]);
            (imax, a[at(imax, k)].abs())
        } else {
            (0, 0.0)
        };
        let kp;
        if absakk.max(colmax) == 0.0 || absakk.is_nan() {
            if info == 0 {
                info = k;
            }
            kp = k;
        } else {
            if absakk >= alpha * colmax {
                kp = k;
            } else {
                let mut jmax = k - 1 + idamax(imax - k, |t| a[at(imax, k - 1 + t)]);
                let mut rowmax = a[at(imax, jmax)].abs();
                if imax < n {
                    jmax = imax + idamax(n - imax, |t| a[at(imax + t, imax)]);
                    rowmax = rowmax.max(a[at(jmax, imax)].abs());
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a[at(imax, imax)].abs() >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                for t in 1..=(n - kp) {
                    a.swap(at(kp + t, kk), at(kp + t, kp));
                }
                for t in 1..(kp - kk) {
                    a.swap(at(kk + t, kk), at(kp, kk + t));
                }
                a.swap(at(kk, kk), at(kp, kp));
                if kstep == 2 {
                    a.swap(at(k + 1, k), at(kp, k));
                }
            }
            if kstep == 1 {
                if k < n {
                    // DSYR('L', n-k, -d11, A(k+1,k), A(k+1,k+1)) then DSCAL(n-k, d11, A(k+1,k)).
                    // Column k is read-only while columns k+1..n are updated.
                    let d11 = 1.0 / a[at(k, k)];
                    let (left, right) = a.split_at_mut(k * n);
                    let x = &mut left[(k - 1) * n..];
                    for j in (k + 1)..=n {
                        let xj = x[j - 1];
                        if xj != 0.0 {
                            let temp = -d11 * xj;
                            let col = &mut right[(j - k - 1) * n + (j - 1)..(j - k) * n];
                            for (aij, &xi) in col.iter_mut().zip(&x[j - 1..]) {
                                *aij += xi * temp;
                            }
                        }
                    }
                    for xi in &mut x[k..] {
                        *xi *= d11;
                    }
                }
            } else if k + 1 < n {
                let mut d21 = a[at(k + 1, k)];
                let d11 = a[at(k + 1, k + 1)] / d21;
                let d22 = a[at(k, k)] / d21;
                let t = 1.0 / (d11 * d22 - 1.0);
                d21 = t / d21;
                // Columns k and k+1 are read-only while columns k+2..n are updated; the
                // multipliers are written back after each column's update, as LAPACK does.
                let (left, right) = a.split_at_mut((k + 1) * n);
                let (xk, xkp1) = left[(k - 1) * n..].split_at_mut(n);
                for j in (k + 2)..=n {
                    let wk = d21 * (d11 * xk[j - 1] - xkp1[j - 1]);
                    let wkp1 = d21 * (d22 * xkp1[j - 1] - xk[j - 1]);
                    let col = &mut right[(j - k - 2) * n + (j - 1)..(j - k - 1) * n];
                    for ((aij, &lik), &likp1) in
                        col.iter_mut().zip(&xk[j - 1..]).zip(&xkp1[j - 1..])
                    {
                        *aij = *aij - lik * wk - likp1 * wkp1;
                    }
                    xk[j - 1] = wk;
                    xkp1[j - 1] = wkp1;
                }
            }
        }
        let kp = kp as isize;
        if kstep == 1 {
            ipiv[k - 1] = kp;
        } else {
            ipiv[k - 1] = -kp;
            ipiv[k] = -kp;
        }
        k += kstep;
    }
    info
}

/// The row a 1-based `IPIV` entry names.
fn pivot_row(p: isize) -> usize {
    p.unsigned_abs()
}

/// `dsytrs('U')` for one right-hand side.
fn sytrs_upper(a: &[f64], n: usize, ipiv: &[isize], b: &mut [f64]) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    // Solve U·D·y = b, U = P(n)·U(n)·…·P(1)·U(1): k from n down.
    let mut k = n;
    while k >= 1 {
        if ipiv[k - 1] > 0 {
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            // DGER(k-1, 1, -1, A(1,k), b(k), b).
            let temp = -b[k - 1];
            for i in 1..k {
                b[i - 1] += a[at(i, k)] * temp;
            }
            b[k - 1] *= 1.0 / a[at(k, k)];
            k -= 1;
        } else {
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k - 1 {
                b.swap(k - 2, kp - 1);
            }
            let temp = -b[k - 1];
            for i in 1..(k - 1) {
                b[i - 1] += a[at(i, k)] * temp;
            }
            let temp = -b[k - 2];
            for i in 1..(k - 1) {
                b[i - 1] += a[at(i, k - 1)] * temp;
            }
            let akm1k = a[at(k - 1, k)];
            let akm1 = a[at(k - 1, k - 1)] / akm1k;
            let ak = a[at(k, k)] / akm1k;
            let denom = akm1 * ak - 1.0;
            let bkm1 = b[k - 2] / akm1k;
            let bk = b[k - 1] / akm1k;
            b[k - 2] = (ak * bkm1 - bk) / denom;
            b[k - 1] = (akm1 * bk - bkm1) / denom;
            k -= 2;
        }
    }
    // Solve Uᵀ·x = y: k from 1 up. DGEMV('T', k-1, 1, -1, b, A(1,k), 1, b(k)).
    let dot_above = |b: &[f64], col: usize, len: usize| {
        let mut temp = 0.0;
        for i in 1..=len {
            temp += b[i - 1] * a[at(i, col)];
        }
        temp
    };
    let mut k = 1;
    while k <= n {
        if ipiv[k - 1] > 0 {
            b[k - 1] -= dot_above(b, k, k - 1);
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            k += 1;
        } else {
            b[k - 1] -= dot_above(b, k, k - 1);
            b[k] -= dot_above(b, k + 1, k - 1);
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            k += 2;
        }
    }
}

/// `dsytrs('L')` for one right-hand side.
fn sytrs_lower(a: &[f64], n: usize, ipiv: &[isize], b: &mut [f64]) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    // Solve L·D·y = b: k from 1 up.
    let mut k = 1;
    while k <= n {
        if ipiv[k - 1] > 0 {
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            let temp = -b[k - 1];
            for i in (k + 1)..=n {
                b[i - 1] += a[at(i, k)] * temp;
            }
            b[k - 1] *= 1.0 / a[at(k, k)];
            k += 1;
        } else {
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k + 1 {
                b.swap(k, kp - 1);
            }
            if k + 1 < n {
                let temp = -b[k - 1];
                for i in (k + 2)..=n {
                    b[i - 1] += a[at(i, k)] * temp;
                }
                let temp = -b[k];
                for i in (k + 2)..=n {
                    b[i - 1] += a[at(i, k + 1)] * temp;
                }
            }
            let akm1k = a[at(k + 1, k)];
            let akm1 = a[at(k, k)] / akm1k;
            let ak = a[at(k + 1, k + 1)] / akm1k;
            let denom = akm1 * ak - 1.0;
            let bkm1 = b[k - 1] / akm1k;
            let bk = b[k] / akm1k;
            b[k - 1] = (ak * bkm1 - bk) / denom;
            b[k] = (akm1 * bk - bkm1) / denom;
            k += 2;
        }
    }
    // Solve Lᵀ·x = y: k from n down. DGEMV('T', n-k, 1, -1, b(k+1), A(k+1,k), 1, b(k)).
    let dot_below = |b: &[f64], col: usize, from: usize| {
        let mut temp = 0.0;
        for i in from..=n {
            temp += b[i - 1] * a[at(i, col)];
        }
        temp
    };
    let mut k = n;
    while k >= 1 {
        if ipiv[k - 1] > 0 {
            if k < n {
                b[k - 1] -= dot_below(b, k, k + 1);
            }
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            k -= 1;
        } else {
            if k < n {
                b[k - 1] -= dot_below(b, k, k + 1);
                b[k - 2] -= dot_below(b, k - 1, k + 1);
            }
            let kp = pivot_row(ipiv[k - 1]);
            if kp != k {
                b.swap(k - 1, kp - 1);
            }
            k -= 2;
        }
    }
}

/// Reference `DDOT` of `len` pairs `(x(t), y(t))`: sequential, left to right.
fn ddot(len: usize, x: impl Fn(usize) -> f64, y: impl Fn(usize) -> f64) -> f64 {
    let mut temp = 0.0;
    for t in 1..=len {
        temp += x(t) * y(t);
    }
    temp
}

/// `dsytri('U')`: overwrites the upper triangle of the factor with that of `A⁻¹`.
fn sytri_upper(a: &mut [f64], n: usize, ipiv: &[isize]) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    let mut work = vec![0.0; n];
    // y(1..m) := -A(1..m,1..m)·x with A symmetric in its upper triangle (reference DSYMV
    // 'U', alpha = -1, beta = 0), into column `col` of `a`.
    let symv_upper = |a: &mut [f64], m: usize, x: &[f64], col: usize| {
        for i in 1..=m {
            a[at(i, col)] = 0.0;
        }
        for j in 1..=m {
            let temp1 = -x[j - 1];
            let mut temp2 = 0.0;
            for i in 1..j {
                a[at(i, col)] += temp1 * a[at(i, j)];
                temp2 += a[at(i, j)] * x[i - 1];
            }
            a[at(j, col)] = a[at(j, col)] + temp1 * a[at(j, j)] + -temp2;
        }
    };
    let mut k = 1;
    while k <= n {
        let kstep = if ipiv[k - 1] > 0 {
            a[at(k, k)] = 1.0 / a[at(k, k)];
            if k > 1 {
                for i in 1..k {
                    work[i - 1] = a[at(i, k)];
                }
                symv_upper(a, k - 1, &work, k);
                let dot = ddot(k - 1, |t| work[t - 1], |t| a[at(t, k)]);
                a[at(k, k)] -= dot;
            }
            1
        } else {
            let t = a[at(k, k + 1)].abs();
            let ak = a[at(k, k)] / t;
            let akp1 = a[at(k + 1, k + 1)] / t;
            let akkp1 = a[at(k, k + 1)] / t;
            let d = t * (ak * akp1 - 1.0);
            a[at(k, k)] = akp1 / d;
            a[at(k + 1, k + 1)] = ak / d;
            a[at(k, k + 1)] = -akkp1 / d;
            if k > 1 {
                for i in 1..k {
                    work[i - 1] = a[at(i, k)];
                }
                symv_upper(a, k - 1, &work, k);
                let dot = ddot(k - 1, |t| work[t - 1], |t| a[at(t, k)]);
                a[at(k, k)] -= dot;
                let dot = ddot(k - 1, |t| a[at(t, k)], |t| a[at(t, k + 1)]);
                a[at(k, k + 1)] -= dot;
                for i in 1..k {
                    work[i - 1] = a[at(i, k + 1)];
                }
                symv_upper(a, k - 1, &work, k + 1);
                let dot = ddot(k - 1, |t| work[t - 1], |t| a[at(t, k + 1)]);
                a[at(k + 1, k + 1)] -= dot;
            }
            2
        };
        let kp = pivot_row(ipiv[k - 1]);
        if kp != k {
            for i in 1..kp {
                a.swap(at(i, k), at(i, kp));
            }
            for t in 1..(k - kp) {
                a.swap(at(kp + t, k), at(kp, kp + t));
            }
            a.swap(at(k, k), at(kp, kp));
            if kstep == 2 {
                a.swap(at(k, k + 1), at(kp, k + 1));
            }
        }
        k += kstep;
    }
}

/// `dsytri('L')`: overwrites the lower triangle of the factor with that of `A⁻¹`.
fn sytri_lower(a: &mut [f64], n: usize, ipiv: &[isize]) {
    let at = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    let mut work = vec![0.0; n];
    // y := -A(k+1..n, k+1..n)·x with that block symmetric in its lower triangle (reference
    // DSYMV 'L', alpha = -1, beta = 0), into rows k+1..n of column `col`.
    let symv_lower = |a: &mut [f64], k: usize, x: &[f64], col: usize| {
        let m = n - k;
        for i in 1..=m {
            a[at(k + i, col)] = 0.0;
        }
        for j in 1..=m {
            let temp1 = -x[j - 1];
            let mut temp2 = 0.0;
            a[at(k + j, col)] += temp1 * a[at(k + j, k + j)];
            for i in (j + 1)..=m {
                a[at(k + i, col)] += temp1 * a[at(k + i, k + j)];
                temp2 += a[at(k + i, k + j)] * x[i - 1];
            }
            a[at(k + j, col)] += -temp2;
        }
    };
    let mut k = n;
    while k >= 1 {
        let kstep = if ipiv[k - 1] > 0 {
            a[at(k, k)] = 1.0 / a[at(k, k)];
            if k < n {
                for i in 1..=(n - k) {
                    work[i - 1] = a[at(k + i, k)];
                }
                symv_lower(a, k, &work, k);
                let dot = ddot(n - k, |t| work[t - 1], |t| a[at(k + t, k)]);
                a[at(k, k)] -= dot;
            }
            1
        } else {
            let t = a[at(k, k - 1)].abs();
            let ak = a[at(k - 1, k - 1)] / t;
            let akp1 = a[at(k, k)] / t;
            let akkp1 = a[at(k, k - 1)] / t;
            let d = t * (ak * akp1 - 1.0);
            a[at(k - 1, k - 1)] = akp1 / d;
            a[at(k, k)] = ak / d;
            a[at(k, k - 1)] = -akkp1 / d;
            if k < n {
                for i in 1..=(n - k) {
                    work[i - 1] = a[at(k + i, k)];
                }
                symv_lower(a, k, &work, k);
                let dot = ddot(n - k, |t| work[t - 1], |t| a[at(k + t, k)]);
                a[at(k, k)] -= dot;
                let dot = ddot(n - k, |t| a[at(k + t, k)], |t| a[at(k + t, k - 1)]);
                a[at(k, k - 1)] -= dot;
                for i in 1..=(n - k) {
                    work[i - 1] = a[at(k + i, k - 1)];
                }
                symv_lower(a, k, &work, k - 1);
                let dot = ddot(n - k, |t| work[t - 1], |t| a[at(k + t, k - 1)]);
                a[at(k - 1, k - 1)] -= dot;
            }
            2
        };
        let kp = pivot_row(ipiv[k - 1]);
        if kp != k {
            for t in 1..=(n - kp) {
                a.swap(at(kp + t, k), at(kp + t, kp));
            }
            for t in 1..(kp - k) {
                a.swap(at(k + t, k), at(kp, k + t));
            }
            a.swap(at(k, k), at(kp, kp));
            if kstep == 2 {
                a.swap(at(k, k - 1), at(kp, k - 1));
            }
        }
        k -= kstep;
    }
}

#[cfg(test)]
mod tests {
    use super::{BunchKaufman, Triangle};

    // Reference values: SciPy 1.17.1's `lapack.dsytf2` / `dsytrs` / `dsytri` (the inverse with
    // its computed triangle mirrored). Values marked kernel-invariant were identical under
    // OpenBLAS's Prescott, Nehalem, Sandybridge, Haswell and Zen kernels.

    fn zero_diagonal() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ]
    }

    fn saddle() -> Vec<Vec<f64>> {
        vec![
            vec![4.0, 1.0, 0.0, 2.0],
            vec![1.0, 0.0, 3.0, 0.0],
            vec![0.0, 3.0, -1.0, 1.0],
            vec![2.0, 0.0, 1.0, 0.0],
        ]
    }

    /// `Pᵀ·diag(-1, 1, 1, -1)·P` with `P` unit upper triangular and integer: det = 1, so the
    /// inverse is the integer matrix `P⁻¹·D·P⁻ᵀ`; cond ≈ 1.3e9.
    fn unimodular() -> Vec<Vec<f64>> {
        vec![
            vec![-1.0, -9.0, 2.0, -5.0],
            vec![-9.0, -80.0, 30.0, -48.0],
            vec![2.0, 30.0, 141.0, -8.0],
            vec![-5.0, -48.0, -8.0, 307.0],
        ]
    }

    fn assert_bits(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "entry {i}: {a:e} vs {e:e}");
        }
    }

    #[test]
    fn pivots_and_singularity_follow_lapack() {
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        for (a, upper, lower) in [
            (zero_diagonal(), vec![1, -2, -2], vec![-3, -3, 3]),
            (saddle(), vec![1, -2, -2, 1], vec![1, -3, -3, 4]),
            (unimodular(), vec![1, 2, 3, 4], vec![2, 3, 4, 4]),
            (singular, vec![1, 2], vec![2, 2]),
        ] {
            for (triangle, expected) in [(Triangle::Upper, upper), (Triangle::Lower, lower)] {
                assert_eq!(BunchKaufman::factor(&a, triangle).ipiv, expected);
            }
        }
        // A zero diagonal needs 2×2 blocks: a factorization without them divides by zero.
        assert!(!BunchKaufman::factor(&zero_diagonal(), Triangle::Upper).is_singular());
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        for (triangle, info) in [(Triangle::Upper, 1), (Triangle::Lower, 2)] {
            let factor = BunchKaufman::factor(&singular, triangle);
            assert!(factor.is_singular());
            assert_eq!(factor.info, info);
        }
    }

    #[test]
    fn two_by_two_blocks_solve_and_invert_like_lapack() {
        // Kernel-invariant.
        let upper = BunchKaufman::factor(&zero_diagonal(), Triangle::Upper);
        let lower = BunchKaufman::factor(&zero_diagonal(), Triangle::Lower);
        assert_bits(
            &upper.solve(&[1.0, 2.0, 3.0]),
            &[1.0, 0.333_333_333_333_333_37, 0.333_333_333_333_333_3],
        );
        assert_bits(
            &lower.solve(&[1.0, 2.0, 3.0]),
            &[1.0, 0.333_333_333_333_333_3, 0.333_333_333_333_333_37],
        );
        let inverse = [
            -0.75,
            0.5,
            0.25,
            0.5,
            -0.333_333_333_333_333_3,
            0.166_666_666_666_666_66,
            0.25,
            0.166_666_666_666_666_66,
            -0.083_333_333_333_333_33,
        ];
        for factor in [&upper, &lower] {
            assert_bits(&factor.inverse().concat(), &inverse);
        }
        // Kernel-invariant, both triangles; the tiny entries are rounding residue of exact
        // zeros, which the transcription must reproduce bit for bit.
        let saddle_inverse = [
            -5.551_115_123_125_783e-17,
            -0.200_000_000_000_000_04,
            1.387_778_780_781_445_7e-17,
            0.600_000_000_000_000_1,
            -0.200_000_000_000_000_04,
            -4.163_336_342_344_337e-17,
            0.4,
            0.400_000_000_000_000_1,
            1.387_778_780_781_445_7e-17,
            0.4,
            -1.040_834_085_586_084_3e-17,
            -0.200_000_000_000_000_04,
            0.600_000_000_000_000_1,
            0.400_000_000_000_000_1,
            -0.200_000_000_000_000_04,
            -1.400_000_000_000_000_1,
        ];
        for triangle in [Triangle::Upper, Triangle::Lower] {
            let factor = BunchKaufman::factor(&saddle(), triangle);
            assert_bits(&factor.inverse().concat(), &saddle_inverse);
            // SciPy's Prescott kernel; the others round x[1] one ulp up (0.2000000000000002).
            assert_bits(
                &factor.solve(&[1.0, 0.0, -1.0, 2.0]),
                &[
                    1.200_000_000_000_000_2,
                    0.200_000_000_000_000_18,
                    -0.400_000_000_000_000_1,
                    -2.000_000_000_000_000_4,
                ],
            );
        }
    }

    #[test]
    fn ill_conditioned_factor_matches_lapack_not_lu() {
        let b = [1.0, -2.0, 3.0, -4.0];
        // SciPy's Prescott kernel; Nehalem through Zen differ by one ulp in x[1..=3].
        assert_bits(
            &BunchKaufman::factor(&unimodular(), Triangle::Upper).solve(&b),
            &[
                -5_030_928.001_527_862,
                547_597.000_166_301_7,
                -45_007.000_013_668_34,
                2_508.000_000_761_657,
            ],
        );
        // SciPy's Nehalem, Sandybridge, Haswell and Zen kernels (Prescott differs in the last
        // entry by one ulp). The exact inverse is the integer matrix; LU lands elsewhere
        // (x[0] = -5030927.999174224 from SciPy's `assume_a='gen'`).
        assert_bits(
            &BunchKaufman::factor(&unimodular(), Triangle::Upper)
                .inverse()
                .concat(),
            &[
                -4_035_964.001_225_698,
                439_299.000_133_412_4,
                -36_106.000_010_965_18,
                2_012.000_000_611_036_5,
                439_299.000_133_412_4,
                -47_816.000_014_521_42,
                3_930.000_001_193_517,
                -219.000_000_066_508_1,
                -36_106.000_010_965_18,
                3_930.000_001_193_517,
                -323.000_000_098_095_2,
                18.000_000_005_466_37,
                2_012.000_000_611_036_5,
                -219.000_000_066_508_1,
                18.000_000_005_466_37,
                -1.000_000_000_304_487_8,
            ],
        );
    }
}
