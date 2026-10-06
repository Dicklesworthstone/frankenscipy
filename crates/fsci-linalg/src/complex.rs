#![forbid(unsafe_code)]

//! Complex dense linear algebra: `scipy.linalg` on `complex128` matrices (frankenscipy-1ksfv.17).
//!
//! The crate root works on `&[Vec<f64>]` throughout; this module is its complex counterpart, with
//! matrices given row-major as `&[Vec<C64>]` exactly like the real API. Real input converts with
//! [`to_complex`]. Two entry points take REAL input and return what SciPy returns for it, a real
//! or a complex matrix ([`MaybeComplexMatrix`]): [`logm_real`] and [`sqrtm_real`], whose results
//! are complex when the matrix has a negative real eigenvalue (`logm(diag(-1, 1)) = diag(iπ, 0)`),
//! where the real `crate::logm` / `crate::sqrtm` can only return NaN.
//!
//! The kernels follow the LAPACK routines SciPy calls, statement by statement where it decides
//! a convention a caller can observe:
//! - [`lu_factor`], [`lu`], [`solve`], [`inv`], [`det`]: `zgetf2` partial pivoting (the pivot is
//!   the FIRST maximum of `|re| + |im|`, LAPACK's `izamax`), `zgetrs` for `trans` 0/1/2, and the
//!   `zgecon`/`zlacn2` 1-norm condition estimate behind SciPy's ill-conditioning warning.
//! - [`cholesky`]: `zpotf2` (upper by default, `A = Uᴴ·U`).
//! - [`qr`]: `zgeqr2` + `zung2r` Householder reflectors from `zlarfg`, so `R` has LAPACK's REAL
//!   diagonal and signs.
//! - [`hessenberg`], [`schur`], [`eig`], [`eigvals`]: `zgebal` balancing, `zgehd2`, `zunghr`, the
//!   complex single-shift QR iteration `zlahqr` (Ahues–Kressner deflation, exceptional shifts every
//!   10 iterations), `ztrevc3` back substitution and `zgeev`'s normalization of each eigenvector to
//!   unit 2-norm with its largest component real. For `n < 75`, where SciPy's `zhseqr` also runs
//!   `zlahqr`, the eigenvalues come out in SciPy's order.
//! - [`eigh`], [`eigvalsh`] (Hermitian) and [`svd`], [`svdvals`]: nalgebra's complex Householder
//!   tridiagonalization / bidiagonalization with implicit-shift QR; eigenvalues ascending and
//!   singular values descending as SciPy returns them. Eigen- and singular vectors are unique only
//!   up to a unit-modulus factor per vector (LAPACK does not normalize their phase either).
//! - [`lstsq`] and [`pinv`]: SciPy's SVD-based definitions (`gelsd` cutoff `cond·σ₁`, `pinv`'s
//!   `atol + rtol·σ₁`).
//! - [`expm`]: scaling and squaring with Padé degrees 3, 5, 7, 9 and 13 (Higham 2005).
//! - [`sqrtm`]: Schur method with the Björck–Hammarling recurrence; [`logm`]: SciPy's
//!   `_logm` — complex Schur form, then Al-Mohy & Higham's inverse scaling and squaring
//!   (`_logm_triu`, `_inverse_squaring_helper`) with SciPy's diagonal and superdiagonal
//!   recomputation.

use crate::{
    DecompOptions, LinalgError, LinalgWarning, MatrixAssumption, NormKind, SolveOptions,
    hardened_dimension_check,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{Complex, DMatrix};

/// A double-precision complex number (`numpy.complex128`).
pub type C64 = Complex<f64>;

/// A row-major complex matrix, the complex analogue of the crate's `Vec<Vec<f64>>`.
pub type ComplexMatrix = Vec<Vec<C64>>;

const CZERO: C64 = Complex::new(0.0, 0.0);
const CONE: C64 = Complex::new(1.0, 0.0);

/// SciPy's `numpy.finfo(float64).eps`.
const EPS: f64 = f64::EPSILON;

// ══════════════════════════════════════════════════════════════════════
// Result types
// ══════════════════════════════════════════════════════════════════════

/// `scipy.linalg.solve` on complex input.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexSolveResult {
    pub x: Vec<C64>,
    /// LAPACK `zgecon`'s 1-norm reciprocal condition estimate of `a` (1.0 for triangular and
    /// diagonal solves, which SciPy does not estimate either).
    pub rcond: f64,
    /// SciPy's `LinAlgWarning`: `rcond < eps`.
    pub warning: Option<LinalgWarning>,
}

/// `scipy.linalg.lu_factor` on complex input: the packed `LU` factors and 0-based pivots
/// (`piv[i]` is the row interchanged with row `i`).
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexLuFactor {
    lu: DMatrix<C64>,
    piv: Vec<usize>,
}

impl ComplexLuFactor {
    /// The packed factors: `U` on and above the diagonal, the unit-lower `L` below it.
    #[must_use]
    pub fn lu(&self) -> ComplexMatrix {
        rows_of(&self.lu)
    }

    /// The 0-based pivot indices, SciPy's `piv`.
    #[must_use]
    pub fn piv(&self) -> &[usize] {
        &self.piv
    }

    /// The matrix order `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.lu.nrows()
    }
}

/// `scipy.linalg.lu(a)` on complex input: `A = P·L·U`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexLuResult {
    pub p: Vec<Vec<f64>>,
    pub l: ComplexMatrix,
    pub u: ComplexMatrix,
}

/// `scipy.linalg.qr` on complex input.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexQrResult {
    pub q: ComplexMatrix,
    pub r: ComplexMatrix,
}

/// `scipy.linalg.svd` on complex input: `A = U·diag(s)·Vᴴ`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexSvdResult {
    pub u: ComplexMatrix,
    /// Singular values in descending order.
    pub s: Vec<f64>,
    pub vh: ComplexMatrix,
}

/// `scipy.linalg.eig` on complex input: right eigenvectors as COLUMNS (row-major storage).
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexEigResult {
    pub eigenvalues: Vec<C64>,
    pub eigenvectors: ComplexMatrix,
}

/// `scipy.linalg.eigh` on a Hermitian matrix: real ascending eigenvalues and orthonormal
/// eigenvectors as columns.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexEighResult {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: ComplexMatrix,
}

/// `scipy.linalg.schur(a, output='complex')`: `A = Z·T·Zᴴ` with `T` upper triangular.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexSchurResult {
    pub t: ComplexMatrix,
    pub z: ComplexMatrix,
}

/// `scipy.linalg.hessenberg(a, calc_q=True)`: `A = Q·H·Qᴴ`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexHessenbergResult {
    pub h: ComplexMatrix,
    pub q: ComplexMatrix,
}

/// `scipy.linalg.lstsq` on complex input.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexLstsqResult {
    pub x: Vec<C64>,
    /// `‖b − A·x‖²` when `rank == n < m` (SciPy's one-element `residues`); `None` otherwise
    /// (SciPy's empty array).
    pub residues: Option<f64>,
    pub rank: usize,
    /// Singular values of `a`, descending.
    pub s: Vec<f64>,
}

/// What SciPy returns from a matrix function of a REAL matrix: a real array, or a complex one
/// when the result has a non-negligible imaginary part.
#[derive(Debug, Clone, PartialEq)]
pub enum MaybeComplexMatrix {
    Real(Vec<Vec<f64>>),
    Complex(ComplexMatrix),
}

impl MaybeComplexMatrix {
    /// The result as a complex matrix (a real result gains zero imaginary parts).
    #[must_use]
    pub fn into_complex(self) -> ComplexMatrix {
        match self {
            Self::Real(m) => to_complex(&m),
            Self::Complex(m) => m,
        }
    }

    /// Whether SciPy's result dtype is complex.
    #[must_use]
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Conversions and small helpers
// ══════════════════════════════════════════════════════════════════════

/// Embed a real matrix in the complex field (`a.astype(complex)`).
#[must_use]
pub fn to_complex(a: &[Vec<f64>]) -> ComplexMatrix {
    a.iter()
        .map(|row| row.iter().map(|&v| Complex::new(v, 0.0)).collect())
        .collect()
}

/// The conjugate transpose `Aᴴ`.
#[must_use]
pub fn conj_transpose(a: &[Vec<C64>]) -> ComplexMatrix {
    let rows = a.len();
    let cols = a.first().map_or(0, Vec::len);
    (0..cols)
        .map(|j| (0..rows).map(|i| a[i][j].conj()).collect())
        .collect()
}

/// The matrix product `A·B`.
///
/// # Errors
/// [`LinalgError::RaggedMatrix`] for ragged input and [`LinalgError::IncompatibleShapes`] when
/// the inner dimensions differ.
pub fn matmul(a: &[Vec<C64>], b: &[Vec<C64>]) -> Result<ComplexMatrix, LinalgError> {
    let (m, k) = shape(a)?;
    let (k2, n) = shape(b)?;
    if k != k2 {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (m, k),
            b_len: k2,
        });
    }
    Ok(rows_of(&(to_dm(a, m, k) * to_dm(b, k2, n))))
}

/// LAPACK `CABS1`: `|re| + |im|`.
#[inline]
fn cabs1(z: C64) -> f64 {
    z.re.abs() + z.im.abs()
}

#[inline]
fn is_finite(z: C64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

/// LAPACK `zladiv`: complex division without the overflow of the textbook formula (Smith).
fn zladiv(a: C64, b: C64) -> C64 {
    if b.re.abs() >= b.im.abs() {
        let r = b.im / b.re;
        let d = b.re + b.im * r;
        Complex::new((a.re + a.im * r) / d, (a.im - a.re * r) / d)
    } else {
        let r = b.re / b.im;
        let d = b.im + b.re * r;
        Complex::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }
}

/// LAPACK `dlapy3`: `√(x² + y² + z²)` without destructive underflow or overflow.
fn dlapy3(x: f64, y: f64, z: f64) -> f64 {
    let (xa, ya, za) = (x.abs(), y.abs(), z.abs());
    let w = xa.max(ya).max(za);
    if w == 0.0 {
        xa + ya + za
    } else {
        w * ((xa / w).powi(2) + (ya / w).powi(2) + (za / w).powi(2)).sqrt()
    }
}

/// BLAS `dznrm2`: the Euclidean norm, scaled so it neither overflows nor underflows.
fn dznrm2(x: &[C64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for z in x {
        for part in [z.re, z.im] {
            if part != 0.0 {
                let a = part.abs();
                if scale < a {
                    ssq = 1.0 + ssq * (scale / a).powi(2);
                    scale = a;
                } else {
                    ssq += (a / scale).powi(2);
                }
            }
        }
    }
    scale * ssq.sqrt()
}

fn shape(a: &[Vec<C64>]) -> Result<(usize, usize), LinalgError> {
    if a.is_empty() {
        return Ok((0, 0));
    }
    let cols = a[0].len();
    if a.iter().any(|row| row.len() != cols) {
        return Err(LinalgError::RaggedMatrix);
    }
    Ok((a.len(), cols))
}

fn check_finite(a: &[Vec<C64>], mode: RuntimeMode, check: bool) -> Result<(), LinalgError> {
    if (check || mode == RuntimeMode::Hardened) && a.iter().flatten().any(|&z| !is_finite(z)) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

fn reject_nan(a: &[Vec<C64>]) -> Result<(), LinalgError> {
    if a.iter().flatten().any(|z| z.re.is_nan() || z.im.is_nan()) {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(())
}

/// Validate a square matrix the way every square-input routine here does; returns `n`.
fn square(a: &[Vec<C64>], options: DecompOptions) -> Result<usize, LinalgError> {
    let (rows, cols) = shape(a)?;
    if rows != cols {
        return Err(LinalgError::ExpectedSquareMatrix);
    }
    hardened_dimension_check(options.mode, rows, cols)?;
    check_finite(a, options.mode, options.check_finite)?;
    Ok(rows)
}

fn to_dm(a: &[Vec<C64>], rows: usize, cols: usize) -> DMatrix<C64> {
    DMatrix::from_fn(rows, cols, |i, j| a[i][j])
}

fn rows_of(m: &DMatrix<C64>) -> ComplexMatrix {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

fn identity(n: usize) -> DMatrix<C64> {
    DMatrix::from_fn(n, n, |i, j| if i == j { CONE } else { CZERO })
}

/// The matrix 1-norm, the maximum column sum of moduli (LAPACK `zlange('1')`).
fn one_norm(m: &DMatrix<C64>) -> f64 {
    (0..m.ncols())
        .map(|j| (0..m.nrows()).map(|i| m[(i, j)].norm()).sum::<f64>())
        .fold(0.0_f64, |acc, v| {
            if v.is_nan() || acc.is_nan() {
                f64::NAN
            } else {
                acc.max(v)
            }
        })
}

fn is_upper_triangular(m: &DMatrix<C64>) -> bool {
    (0..m.ncols()).all(|j| (j + 1..m.nrows()).all(|i| m[(i, j)] == CZERO))
}

// ══════════════════════════════════════════════════════════════════════
// Householder reflectors (zlarfg / zlarf)
// ══════════════════════════════════════════════════════════════════════

/// LAPACK `zlarfg`: the reflector `H = I − τ·v·vᴴ` (`v = [1; x']`) with `Hᴴ·[α; x] = [β; 0]`
/// and `β` REAL. Overwrites `x` with `x'` and returns `(β, τ)`.
fn zlarfg(alpha: C64, x: &mut [C64]) -> (C64, C64) {
    let mut xnorm = dznrm2(x);
    let (mut alphr, mut alphi) = (alpha.re, alpha.im);
    if xnorm == 0.0 && alphi == 0.0 {
        return (alpha, CZERO);
    }
    let mut beta = -dlapy3(alphr, alphi, xnorm).copysign(alphr);
    // dlamch('S') / dlamch('E') with LAPACK's eps = 2^-53.
    let safmin = f64::MIN_POSITIVE / (0.5 * EPS);
    let rsafmn = 1.0 / safmin;
    let mut knt = 0;
    let mut alpha = alpha;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            for v in x.iter_mut() {
                *v *= rsafmn;
            }
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = dznrm2(x);
        alpha = Complex::new(alphr, alphi);
        beta = -dlapy3(alphr, alphi, xnorm).copysign(alphr);
    }
    let tau = Complex::new((beta - alphr) / beta, -alphi / beta);
    let scal = zladiv(CONE, alpha - beta);
    for v in x.iter_mut() {
        *v = scal * *v;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    (Complex::new(beta, 0.0), tau)
}

/// LAPACK `zlarf('Left')`: `C := (I − τ·v·vᴴ)·C` on rows `row0..row0+v.len()` and the given
/// columns.
fn apply_left(
    c: &mut DMatrix<C64>,
    v: &[C64],
    row0: usize,
    cols: std::ops::Range<usize>,
    tau: C64,
) {
    if tau == CZERO {
        return;
    }
    for j in cols {
        let mut s = CZERO;
        for (k, &vk) in v.iter().enumerate() {
            s += vk.conj() * c[(row0 + k, j)];
        }
        let ts = tau * s;
        for (k, &vk) in v.iter().enumerate() {
            c[(row0 + k, j)] -= vk * ts;
        }
    }
}

/// LAPACK `zlarf('Right')`: `C := C·(I − τ·v·vᴴ)` on the given rows and columns
/// `col0..col0+v.len()`.
fn apply_right(
    c: &mut DMatrix<C64>,
    v: &[C64],
    rows: std::ops::Range<usize>,
    col0: usize,
    tau: C64,
) {
    if tau == CZERO {
        return;
    }
    for i in rows {
        let mut s = CZERO;
        for (k, &vk) in v.iter().enumerate() {
            s += c[(i, col0 + k)] * vk;
        }
        let ts = tau * s;
        for (k, &vk) in v.iter().enumerate() {
            c[(i, col0 + k)] -= ts * vk.conj();
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// LU (zgetf2 / zgetrs / zgecon)
// ══════════════════════════════════════════════════════════════════════

/// LAPACK `zgetf2` on an m×n matrix. Returns the 0-based pivots and the first zero pivot.
fn getrf(a: &mut DMatrix<C64>) -> (Vec<usize>, Option<usize>) {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut piv = vec![0usize; k];
    let mut info = None;
    let sfmin = f64::MIN_POSITIVE;
    for j in 0..k {
        let mut p = j;
        let mut best = cabs1(a[(j, j)]);
        for i in j + 1..m {
            let v = cabs1(a[(i, j)]);
            if v > best {
                best = v;
                p = i;
            }
        }
        piv[j] = p;
        if a[(p, j)] != CZERO {
            if p != j {
                a.swap_rows(p, j);
            }
            if j + 1 < m {
                let d = a[(j, j)];
                if d.norm() >= sfmin {
                    let r = zladiv(CONE, d);
                    for i in j + 1..m {
                        a[(i, j)] = r * a[(i, j)];
                    }
                } else {
                    for i in j + 1..m {
                        a[(i, j)] = zladiv(a[(i, j)], d);
                    }
                }
            }
        } else if info.is_none() {
            info = Some(j);
        }
        if j + 1 < k {
            for jj in j + 1..n {
                let u = a[(j, jj)];
                if u != CZERO {
                    for i in j + 1..m {
                        let l = a[(i, j)];
                        a[(i, jj)] -= l * u;
                    }
                }
            }
        }
    }
    (piv, info)
}

/// LAPACK `zgetrs` for one right-hand side: `trans` 0 solves `A·x = b`, 1 `Aᵀ·x = b`,
/// 2 `Aᴴ·x = b`.
fn getrs(lu: &DMatrix<C64>, piv: &[usize], b: &mut [C64], trans: u8) {
    let n = lu.nrows();
    let cj = |z: C64| if trans == 2 { z.conj() } else { z };
    if trans == 0 {
        for (j, &p) in piv.iter().enumerate() {
            if p != j {
                b.swap(j, p);
            }
        }
        for j in 0..n {
            let bj = b[j];
            if bj != CZERO {
                for i in j + 1..n {
                    b[i] -= bj * lu[(i, j)];
                }
            }
        }
        for j in (0..n).rev() {
            if b[j] != CZERO {
                b[j] = zladiv(b[j], lu[(j, j)]);
                let bj = b[j];
                for i in 0..j {
                    b[i] -= bj * lu[(i, j)];
                }
            }
        }
    } else {
        // Uᵀ (or Uᴴ) is lower triangular: forward substitution.
        for j in 0..n {
            let mut temp = b[j];
            for i in 0..j {
                temp -= cj(lu[(i, j)]) * b[i];
            }
            b[j] = zladiv(temp, cj(lu[(j, j)]));
        }
        // Lᵀ (or Lᴴ) is unit upper triangular: back substitution.
        for j in (0..n).rev() {
            let mut temp = b[j];
            for i in j + 1..n {
                temp -= cj(lu[(i, j)]) * b[i];
            }
            b[j] = temp;
        }
        for (j, &p) in piv.iter().enumerate().rev() {
            if p != j {
                b.swap(j, p);
            }
        }
    }
}

/// `inv(U)·inv(L)·x` (`trans == 0`) or `inv(L)ᴴ·inv(U)ᴴ·x` (`trans == 2`) without the row
/// permutation, which does not change the 1-norm `zgecon` estimates.
fn lu_apply_inverse_unpermuted(lu: &DMatrix<C64>, x: &mut [C64], trans: u8) {
    let n = lu.nrows();
    let identity_piv: Vec<usize> = (0..n).collect();
    getrs(lu, &identity_piv, x, trans);
}

/// LAPACK `zlacn2`: estimate `‖B‖₁` from products with `B` (`apply(x, false)`) and `Bᴴ`
/// (`apply(x, true)`).
fn zlacn2(n: usize, mut apply: impl FnMut(&mut [C64], bool)) -> f64 {
    const ITMAX: usize = 5;
    let safmin = f64::MIN_POSITIVE;
    let sum_abs = |x: &[C64]| x.iter().map(|z| z.norm()).sum::<f64>();
    let izmax1 = |x: &[C64]| {
        let mut best = 0usize;
        let mut bv = -1.0_f64;
        for (i, z) in x.iter().enumerate() {
            let v = z.norm();
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best
    };
    let unit_phase = |x: &mut [C64]| {
        for z in x.iter_mut() {
            let a = z.norm();
            *z = if a > safmin {
                Complex::new(z.re / a, z.im / a)
            } else {
                CONE
            };
        }
    };
    if n == 0 {
        return 0.0;
    }
    let mut x = vec![Complex::new(1.0 / n as f64, 0.0); n];
    apply(&mut x, false);
    if n == 1 {
        return x[0].norm();
    }
    let mut est = sum_abs(&x);
    unit_phase(&mut x);
    apply(&mut x, true);
    let mut j = izmax1(&x);
    let mut iter = 2;
    loop {
        x.iter_mut().for_each(|z| *z = CZERO);
        x[j] = CONE;
        apply(&mut x, false);
        let estold = est;
        est = sum_abs(&x);
        if est <= estold {
            break;
        }
        unit_phase(&mut x);
        apply(&mut x, true);
        let jlast = j;
        j = izmax1(&x);
        if x[jlast].norm() != x[j].norm() && iter < ITMAX {
            iter += 1;
            continue;
        }
        break;
    }
    let mut altsgn = 1.0;
    for (i, z) in x.iter_mut().enumerate() {
        *z = Complex::new(altsgn * (1.0 + i as f64 / (n - 1) as f64), 0.0);
        altsgn = -altsgn;
    }
    apply(&mut x, false);
    let temp = 2.0 * (sum_abs(&x) / (3 * n) as f64);
    if temp > est { temp } else { est }
}

/// LAPACK `zgecon('1')` from LU factors and `‖A‖₁`.
fn lu_rcond(lu: &DMatrix<C64>, anorm: f64) -> f64 {
    let n = lu.nrows();
    if n == 0 {
        return 1.0;
    }
    if anorm == 0.0 || anorm.is_nan() {
        return 0.0;
    }
    let ainvnm = zlacn2(n, |x, herm| {
        lu_apply_inverse_unpermuted(lu, x, if herm { 2 } else { 0 })
    });
    if ainvnm == 0.0 || !ainvnm.is_finite() {
        return 0.0;
    }
    (1.0 / ainvnm) / anorm
}

fn ill_conditioned(rcond: f64) -> Option<LinalgWarning> {
    (rcond < EPS).then_some(LinalgWarning::IllConditioned {
        reciprocal_condition: rcond,
    })
}

/// `scipy.linalg.lu_factor` on complex input.
///
/// # Errors
/// Shape and finiteness errors; a singular matrix is NOT an error (SciPy only warns), the zero
/// pivot is left in `U` and [`lu_solve`] then divides by it.
pub fn lu_factor(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexLuFactor, LinalgError> {
    let n = square(a, options)?;
    let mut lu = to_dm(a, n, n);
    let (piv, _info) = getrf(&mut lu);
    Ok(ComplexLuFactor { lu, piv })
}

/// `scipy.linalg.lu_solve((lu, piv), b, trans)`: `trans` 0 solves `A·x = b`, 1 `Aᵀ·x = b` and
/// 2 `Aᴴ·x = b`.
///
/// # Errors
/// [`LinalgError::IncompatibleShapes`] when `b` has the wrong length,
/// [`LinalgError::InvalidArgument`] for `trans > 2`.
pub fn lu_solve(factor: &ComplexLuFactor, b: &[C64], trans: u8) -> Result<Vec<C64>, LinalgError> {
    let n = factor.n();
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    if trans > 2 {
        return Err(LinalgError::InvalidArgument {
            detail: "trans must be 0, 1 or 2".to_string(),
        });
    }
    let mut x = b.to_vec();
    getrs(&factor.lu, &factor.piv, &mut x, trans);
    Ok(x)
}

/// `scipy.linalg.lu(a)` on complex input (`permute_l=False`): `A = P·L·U` for an m×n `a`, with
/// `L` m×k unit lower trapezoidal and `U` k×n upper trapezoidal, `k = min(m, n)`.
///
/// # Errors
/// Ragged input and non-finite entries (under `check_finite`).
pub fn lu(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexLuResult, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    let k = m.min(n);
    let mut f = to_dm(a, m, n);
    let (piv, _info) = getrf(&mut f);
    // perm[i] = original row now in position i.
    let mut perm: Vec<usize> = (0..m).collect();
    for (j, &p) in piv.iter().enumerate() {
        perm.swap(j, p);
    }
    let mut p = vec![vec![0.0; m]; m];
    for (i, &orig) in perm.iter().enumerate() {
        p[orig][i] = 1.0;
    }
    let l: ComplexMatrix = (0..m)
        .map(|i| {
            (0..k)
                .map(|j| match i.cmp(&j) {
                    std::cmp::Ordering::Greater => f[(i, j)],
                    std::cmp::Ordering::Equal => CONE,
                    std::cmp::Ordering::Less => CZERO,
                })
                .collect()
        })
        .collect();
    let u: ComplexMatrix = (0..k)
        .map(|i| {
            (0..n)
                .map(|j| if j >= i { f[(i, j)] } else { CZERO })
                .collect()
        })
        .collect();
    Ok(ComplexLuResult { p, l, u })
}

/// `scipy.linalg.det` on complex input: the product of `U`'s diagonal with the sign of the
/// row permutation.
///
/// # Errors
/// Shape and finiteness errors.
pub fn det(a: &[Vec<C64>], options: DecompOptions) -> Result<C64, LinalgError> {
    let n = square(a, options)?;
    if n == 0 {
        return Ok(CONE);
    }
    let mut lu = to_dm(a, n, n);
    let (piv, _info) = getrf(&mut lu);
    let mut d = CONE;
    for (j, &p) in piv.iter().enumerate() {
        d *= lu[(j, j)];
        if p != j {
            d = -d;
        }
    }
    Ok(d)
}

/// `scipy.linalg.solve(a, b)` on complex input.
///
/// `assume_a` selects the factorization as in SciPy: `PositiveDefinite` uses Cholesky
/// (`zpotrf`), `UpperTriangular`/`LowerTriangular` and `Diagonal` solve directly, and every
/// other assumption (general, symmetric, Hermitian) uses LU with partial pivoting. `transposed`
/// solves `Aᵀ·x = b`. A zero pivot is [`LinalgError::SingularMatrix`]; `rcond < eps` sets the
/// ill-conditioning warning SciPy emits.
///
/// # Errors
/// Shape, finiteness, singularity and (Cholesky) non-positive-definiteness errors.
pub fn solve(
    a: &[Vec<C64>],
    b: &[C64],
    options: SolveOptions,
) -> Result<ComplexSolveResult, LinalgError> {
    let decomp = DecompOptions {
        mode: options.mode,
        check_finite: options.check_finite,
    };
    let n = square(a, decomp)?;
    if b.len() != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: b.len(),
        });
    }
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && b.iter().any(|&z| !is_finite(z))
    {
        return Err(LinalgError::NonFiniteInput);
    }
    if options.transposed {
        return Err(LinalgError::NotSupported {
            detail: "scipy.linalg.solve can currently not solve a^T x = b or a^H x = b for \
                     complex matrices."
                .to_string(),
        });
    }
    if n == 0 {
        return Ok(ComplexSolveResult {
            x: Vec::new(),
            rcond: 1.0,
            warning: None,
        });
    }
    let am = to_dm(a, n, n);
    match options.assume_a {
        Some(MatrixAssumption::Diagonal) => {
            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                if am[(i, i)] == CZERO {
                    return Err(LinalgError::SingularMatrix);
                }
                x.push(zladiv(b[i], am[(i, i)]));
            }
            Ok(ComplexSolveResult {
                x,
                rcond: 1.0,
                warning: None,
            })
        }
        Some(MatrixAssumption::UpperTriangular | MatrixAssumption::LowerTriangular) => {
            let upper = matches!(options.assume_a, Some(MatrixAssumption::UpperTriangular));
            if (0..n).any(|i| am[(i, i)] == CZERO) {
                return Err(LinalgError::SingularMatrix);
            }
            let x = triangular_solve(&am, b, upper, false);
            Ok(ComplexSolveResult {
                x,
                rcond: 1.0,
                warning: None,
            })
        }
        Some(MatrixAssumption::PositiveDefinite) => {
            let u = potrf_upper(&am)?;
            // A = Uᴴ·U: solve Uᴴ·y = b, then U·x = y.
            let mut y = vec![CZERO; n];
            for j in 0..n {
                let mut t = b[j];
                for i in 0..j {
                    t -= u[(i, j)].conj() * y[i];
                }
                y[j] = zladiv(t, u[(j, j)].conj());
            }
            for j in (0..n).rev() {
                let mut t = y[j];
                for i in j + 1..n {
                    t -= u[(j, i)] * y[i];
                }
                y[j] = zladiv(t, u[(j, j)]);
            }
            let rcond = chol_rcond(&u, one_norm(&am));
            Ok(ComplexSolveResult {
                x: y,
                rcond,
                warning: ill_conditioned(rcond),
            })
        }
        _ => {
            let mut lu = am.clone();
            let (piv, info) = getrf(&mut lu);
            if info.is_some() {
                return Err(LinalgError::SingularMatrix);
            }
            let mut x = b.to_vec();
            getrs(&lu, &piv, &mut x, 0);
            let rcond = lu_rcond(&lu, one_norm(&am));
            Ok(ComplexSolveResult {
                x,
                rcond,
                warning: ill_conditioned(rcond),
            })
        }
    }
}

/// `scipy.linalg.solve(a, B)` for a matrix right-hand side (one solve per column of `B`).
///
/// # Errors
/// As [`solve`]; `B` must have `n` rows.
pub fn solve_matrix(
    a: &[Vec<C64>],
    b: &[Vec<C64>],
    options: SolveOptions,
) -> Result<ComplexMatrix, LinalgError> {
    let (rows, cols) = shape(b)?;
    let n = a.len();
    if rows != n {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (n, n),
            b_len: rows,
        });
    }
    let mut out = vec![vec![CZERO; cols]; rows];
    for j in 0..cols {
        let col: Vec<C64> = (0..rows).map(|i| b[i][j]).collect();
        let x = solve(a, &col, options)?.x;
        for i in 0..rows {
            out[i][j] = x[i];
        }
    }
    Ok(out)
}

/// Triangular solve `T·x = b` (`upper`), or `Tᵀ·x = b` when `transpose` (the triangle flips).
fn triangular_solve(t: &DMatrix<C64>, b: &[C64], upper: bool, transpose: bool) -> Vec<C64> {
    let n = t.nrows();
    let at = |i: usize, j: usize| if transpose { t[(j, i)] } else { t[(i, j)] };
    let mut x = b.to_vec();
    if upper {
        for j in (0..n).rev() {
            if x[j] != CZERO {
                x[j] = zladiv(x[j], at(j, j));
                let xj = x[j];
                for i in 0..j {
                    x[i] -= xj * at(i, j);
                }
            }
        }
    } else {
        for j in 0..n {
            if x[j] != CZERO {
                x[j] = zladiv(x[j], at(j, j));
                let xj = x[j];
                for i in j + 1..n {
                    x[i] -= xj * at(i, j);
                }
            }
        }
    }
    x
}

/// `scipy.linalg.inv` on complex input.
///
/// # Errors
/// [`LinalgError::SingularMatrix`] for a zero pivot, plus shape and finiteness errors.
pub fn inv(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexMatrix, LinalgError> {
    let n = square(a, options)?;
    let mut lu = to_dm(a, n, n);
    let (piv, info) = getrf(&mut lu);
    if info.is_some() {
        return Err(LinalgError::SingularMatrix);
    }
    let mut out = vec![vec![CZERO; n]; n];
    for j in 0..n {
        let mut e = vec![CZERO; n];
        e[j] = CONE;
        getrs(&lu, &piv, &mut e, 0);
        for i in 0..n {
            out[i][j] = e[i];
        }
    }
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════
// Cholesky (zpotf2)
// ══════════════════════════════════════════════════════════════════════

/// LAPACK `zpotf2('U')`: `A = Uᴴ·U` from the upper triangle. A non-positive pivot `j` is
/// SciPy's "j+1-th leading minor of the array is not positive definite".
fn potrf_upper(a: &DMatrix<C64>) -> Result<DMatrix<C64>, LinalgError> {
    let n = a.nrows();
    let mut u = DMatrix::from_element(n, n, CZERO);
    for j in 0..n {
        let mut ajj = a[(j, j)].re;
        for i in 0..j {
            ajj -= u[(i, j)].norm_sqr();
        }
        if ajj <= 0.0 || ajj.is_nan() {
            return Err(LinalgError::InvalidArgument {
                detail: format!(
                    "{}-th leading minor of the array is not positive definite",
                    j + 1
                ),
            });
        }
        let ajj = ajj.sqrt();
        u[(j, j)] = Complex::new(ajj, 0.0);
        for k in j + 1..n {
            let mut s = a[(j, k)];
            for i in 0..j {
                s -= u[(i, j)].conj() * u[(i, k)];
            }
            u[(j, k)] = s * (1.0 / ajj);
        }
    }
    Ok(u)
}

/// `zpocon`: the 1-norm reciprocal condition estimate from the Cholesky factor.
fn chol_rcond(u: &DMatrix<C64>, anorm: f64) -> f64 {
    let n = u.nrows();
    if n == 0 {
        return 1.0;
    }
    if anorm == 0.0 || anorm.is_nan() {
        return 0.0;
    }
    let ainvnm = zlacn2(n, |x, _herm| {
        // A⁻¹ = U⁻¹·U⁻ᴴ is Hermitian, so both products are the same solve.
        let mut y = x.to_vec();
        for j in 0..n {
            let mut t = y[j];
            for i in 0..j {
                t -= u[(i, j)].conj() * y[i];
            }
            y[j] = zladiv(t, u[(j, j)]);
        }
        for j in (0..n).rev() {
            let mut t = y[j];
            for i in j + 1..n {
                t -= u[(j, i)] * y[i];
            }
            y[j] = zladiv(t, u[(j, j)]);
        }
        x.copy_from_slice(&y);
    });
    if ainvnm == 0.0 || !ainvnm.is_finite() {
        return 0.0;
    }
    (1.0 / ainvnm) / anorm
}

/// `scipy.linalg.cholesky(a, lower)` on a Hermitian positive-definite matrix: `A = Uᴴ·U`
/// (`lower = false`, SciPy's default) or `A = L·Lᴴ`. Only the referenced triangle is read.
///
/// # Errors
/// [`LinalgError::InvalidArgument`] naming the leading minor that is not positive definite,
/// plus shape and finiteness errors.
pub fn cholesky(
    a: &[Vec<C64>],
    lower: bool,
    options: DecompOptions,
) -> Result<ComplexMatrix, LinalgError> {
    let n = square(a, options)?;
    let am = if lower {
        // The lower triangle of A is the upper triangle of Aᴴ; chol(Aᴴ) = Uᴴ·U gives L = Uᴴ.
        DMatrix::from_fn(n, n, |i, j| a[j][i].conj())
    } else {
        to_dm(a, n, n)
    };
    let u = potrf_upper(&am)?;
    Ok(if lower {
        rows_of(&u.adjoint())
    } else {
        rows_of(&u)
    })
}

// ══════════════════════════════════════════════════════════════════════
// QR (zgeqr2 / zung2r)
// ══════════════════════════════════════════════════════════════════════

/// LAPACK `zgeqr2`: `R` in the upper triangle, the reflectors below it; returns `τ`.
fn geqr2(a: &mut DMatrix<C64>) -> Vec<C64> {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut tau = vec![CZERO; k];
    for i in 0..k {
        let mut x: Vec<C64> = (i + 1..m).map(|r| a[(r, i)]).collect();
        let (beta, t) = zlarfg(a[(i, i)], &mut x);
        for (off, r) in (i + 1..m).enumerate() {
            a[(r, i)] = x[off];
        }
        a[(i, i)] = beta;
        tau[i] = t;
        if i + 1 < n {
            let v: Vec<C64> = std::iter::once(CONE).chain(x.iter().copied()).collect();
            apply_left(a, &v, i, i + 1..n, t.conj());
        }
    }
    tau
}

/// LAPACK `zung2r`: the first `ncols` columns of `Q = H(0)·H(1)···H(k−1)` from the reflectors
/// stored below the diagonal of `a` (m rows).
fn ung2r(a: &DMatrix<C64>, tau: &[C64], ncols: usize) -> DMatrix<C64> {
    let m = a.nrows();
    let k = tau.len();
    let mut q = DMatrix::from_element(m, ncols, CZERO);
    for j in 0..k.min(ncols) {
        for i in j + 1..m {
            q[(i, j)] = a[(i, j)];
        }
    }
    for j in k..ncols {
        q[(j, j)] = CONE;
    }
    for i in (0..k).rev() {
        if i + 1 < ncols {
            q[(i, i)] = CONE;
            let v: Vec<C64> = (i..m).map(|r| q[(r, i)]).collect();
            apply_left(&mut q, &v, i, i + 1..ncols, tau[i]);
        }
        for r in i + 1..m {
            q[(r, i)] = -tau[i] * q[(r, i)];
        }
        q[(i, i)] = CONE - tau[i];
        for r in 0..i {
            q[(r, i)] = CZERO;
        }
    }
    q
}

/// `scipy.linalg.qr(a, mode='full' | 'economic')` on complex input.
///
/// `economic = false`: `Q` m×m, `R` m×n. `economic = true`: `Q` m×k, `R` k×n, `k = min(m, n)`.
/// `R`'s diagonal is real, as LAPACK's `zgeqrf` leaves it.
///
/// # Errors
/// Ragged input and non-finite entries (under `check_finite`).
pub fn qr(
    a: &[Vec<C64>],
    economic: bool,
    options: DecompOptions,
) -> Result<ComplexQrResult, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    let k = m.min(n);
    let mut f = to_dm(a, m, n);
    let tau = geqr2(&mut f);
    let qcols = if economic { k } else { m };
    let q = ung2r(&f, &tau, qcols);
    let rrows = if economic { k } else { m };
    let r: ComplexMatrix = (0..rrows)
        .map(|i| {
            (0..n)
                .map(|j| if j >= i { f[(i, j)] } else { CZERO })
                .collect()
        })
        .collect();
    Ok(ComplexQrResult { q: rows_of(&q), r })
}

/// Extend the orthonormal columns of `u` (m×k) to an m×m unitary matrix: the trailing
/// `m − k` columns of the full `Q` of `u`'s QR factorization span the complement of `u`.
fn complete_unitary(u: &DMatrix<C64>) -> DMatrix<C64> {
    let (m, k) = u.shape();
    if k >= m {
        return u.clone();
    }
    let mut f = u.clone();
    let tau = geqr2(&mut f);
    let q = ung2r(&f, &tau, m);
    DMatrix::from_fn(m, m, |i, j| if j < k { u[(i, j)] } else { q[(i, j)] })
}

// ══════════════════════════════════════════════════════════════════════
// Balancing, Hessenberg reduction, Schur form (zgebal / zgehd2 / zunghr / zlahqr)
// ══════════════════════════════════════════════════════════════════════

/// LAPACK `zgebal`: returns `(ilo, ihi, scale)` (0-based, inclusive) where `scale[i]` is the
/// interchanged index for `i` outside `ilo..=ihi` and the scaling factor inside.
fn gebal(a: &mut DMatrix<C64>, permute: bool, scale: bool) -> (usize, usize, Vec<f64>) {
    let n = a.nrows();
    let mut sc = vec![1.0_f64; n];
    if n == 0 {
        return (0, 0, sc);
    }
    let mut k = 0usize;
    let mut l = n - 1;
    let nonzero = |z: C64| z.re != 0.0 || z.im != 0.0;
    if permute {
        // Rows isolating an eigenvalue, pushed down.
        let mut noconv = true;
        while noconv {
            noconv = false;
            let mut i = l as isize;
            while i >= 0 {
                let iu = i as usize;
                let canswap = (0..=l).all(|j| iu == j || !nonzero(a[(iu, j)]));
                if canswap {
                    sc[l] = iu as f64;
                    if iu != l {
                        a.swap_columns(iu, l);
                        for c in k..n {
                            let t = a[(iu, c)];
                            a[(iu, c)] = a[(l, c)];
                            a[(l, c)] = t;
                        }
                    }
                    noconv = true;
                    if l == 0 {
                        return (0, 0, sc);
                    }
                    l -= 1;
                }
                i -= 1;
            }
        }
        // Columns isolating an eigenvalue, pushed left.
        let mut noconv = true;
        while noconv {
            noconv = false;
            let mut j = k;
            while j <= l {
                let canswap = (k..=l).all(|i| i == j || !nonzero(a[(i, j)]));
                if canswap {
                    sc[k] = j as f64;
                    if j != k {
                        for r in 0..=l {
                            let t = a[(r, j)];
                            a[(r, j)] = a[(r, k)];
                            a[(r, k)] = t;
                        }
                        for c in k..n {
                            let t = a[(j, c)];
                            a[(j, c)] = a[(k, c)];
                            a[(k, c)] = t;
                        }
                    }
                    noconv = true;
                    k += 1;
                }
                j += 1;
            }
        }
    }
    for s in sc.iter_mut().take(l + 1).skip(k) {
        *s = 1.0;
    }
    if !scale {
        return (k, l, sc);
    }
    let sclfac = 2.0_f64;
    let factor = 0.95_f64;
    let sfmin1 = f64::MIN_POSITIVE / EPS;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * sclfac;
    let sfmax2 = 1.0 / sfmin2;
    let mut noconv = true;
    while noconv {
        noconv = false;
        for i in k..=l {
            let col: Vec<C64> = (k..=l).map(|r| a[(r, i)]).collect();
            let row: Vec<C64> = (k..=l).map(|c| a[(i, c)]).collect();
            let mut c = dznrm2(&col);
            let mut r = dznrm2(&row);
            let mut ca = (0..=l)
                .map(|r| a[(r, i)])
                .fold((CZERO, -1.0_f64), |(bz, bv), z| {
                    if cabs1(z) > bv {
                        (z, cabs1(z))
                    } else {
                        (bz, bv)
                    }
                })
                .0
                .norm();
            let mut ra = (k..n)
                .map(|c| a[(i, c)])
                .fold((CZERO, -1.0_f64), |(bz, bv), z| {
                    if cabs1(z) > bv {
                        (z, cabs1(z))
                    } else {
                        (bz, bv)
                    }
                })
                .0
                .norm();
            if c == 0.0 || r == 0.0 {
                continue;
            }
            if (c + ca + r + ra).is_nan() {
                return (k, l, sc);
            }
            let mut g = r / sclfac;
            let mut f = 1.0_f64;
            let s = c + r;
            while c < g && f.max(c).max(ca) < sfmax2 && r.min(g).min(ra) > sfmin2 {
                f *= sclfac;
                c *= sclfac;
                ca *= sclfac;
                r /= sclfac;
                g /= sclfac;
                ra /= sclfac;
            }
            g = c / sclfac;
            while g >= r && r.max(ra) < sfmax2 && f.min(c).min(g).min(ca) > sfmin2 {
                f /= sclfac;
                c /= sclfac;
                g /= sclfac;
                ca /= sclfac;
                r *= sclfac;
                ra *= sclfac;
            }
            if c + r >= factor * s {
                continue;
            }
            if f < 1.0 && sc[i] < 1.0 && f * sc[i] <= sfmin1 {
                continue;
            }
            if f > 1.0 && sc[i] > 1.0 && sc[i] >= sfmax1 / f {
                continue;
            }
            let g = 1.0 / f;
            sc[i] *= f;
            noconv = true;
            for c in k..n {
                a[(i, c)] *= g;
            }
            for r in 0..=l {
                a[(r, i)] *= f;
            }
        }
    }
    (k, l, sc)
}

/// LAPACK `zgebak` for right vectors: undo `gebal` on the rows of `v`.
fn gebak(v: &mut DMatrix<C64>, ilo: usize, ihi: usize, sc: &[f64], scaled: bool, permuted: bool) {
    let n = v.nrows();
    if n == 0 {
        return;
    }
    // LAPACK skips the scaling when ilo == ihi: `gebal` then returned early from the
    // permutation search and scale[ilo] holds a permutation index, not a scale factor.
    if scaled && ilo != ihi {
        for i in ilo..=ihi {
            let s = sc[i];
            for c in 0..v.ncols() {
                v[(i, c)] *= s;
            }
        }
    }
    if permuted {
        for ii in 0..n {
            let mut i = ii;
            if i >= ilo && i <= ihi {
                continue;
            }
            if i < ilo {
                i = ilo - 1 - ii;
            }
            let k = sc[i] as usize;
            if k != i {
                v.swap_rows(i, k);
            }
        }
    }
}

/// LAPACK `zgehd2` on rows/columns `ilo..=ihi`: returns `τ` (length `n − 1`).
fn gehd2(a: &mut DMatrix<C64>, ilo: usize, ihi: usize) -> Vec<C64> {
    let n = a.nrows();
    let mut tau = vec![CZERO; n.saturating_sub(1)];
    for i in ilo..ihi {
        let mut x: Vec<C64> = (i + 2..=ihi).map(|r| a[(r, i)]).collect();
        let (beta, t) = zlarfg(a[(i + 1, i)], &mut x);
        tau[i] = t;
        for (off, r) in (i + 2..=ihi).enumerate() {
            a[(r, i)] = x[off];
        }
        let v: Vec<C64> = std::iter::once(CONE).chain(x.iter().copied()).collect();
        apply_right(a, &v, 0..ihi + 1, i + 1, t);
        apply_left(a, &v, i + 1, i + 1..n, t.conj());
        a[(i + 1, i)] = beta;
    }
    tau
}

/// LAPACK `zunghr`: the unitary `Q` of `gehd2`.
fn unghr(a: &DMatrix<C64>, tau: &[C64], ilo: usize, ihi: usize) -> DMatrix<C64> {
    let n = a.nrows();
    let mut q = identity(n);
    let nh = ihi.saturating_sub(ilo);
    if nh == 0 {
        return q;
    }
    let mut b = DMatrix::from_element(nh, nh, CZERO);
    for c in 0..nh {
        for r in c + 1..nh {
            b[(r, c)] = a[(ilo + 1 + r, ilo + c)];
        }
    }
    let qb = ung2r(&b, &tau[ilo..ihi], nh);
    for r in 0..nh {
        for c in 0..nh {
            q[(ilo + 1 + r, ilo + 1 + c)] = qb[(r, c)];
        }
    }
    q
}

/// LAPACK `zlahqr`: the complex single-shift QR algorithm on the Hessenberg rows/columns
/// `ilo..=ihi` of `h`, accumulating into rows `iloz..=ihiz` of `z`. Returns the index of the
/// eigenvalue that failed to converge.
#[allow(clippy::too_many_lines)]
fn lahqr(
    wantt: bool,
    wantz: bool,
    h: &mut DMatrix<C64>,
    ilo: usize,
    ihi: usize,
    w: &mut [C64],
    z: &mut DMatrix<C64>,
    iloz: usize,
    ihiz: usize,
) -> Result<(), usize> {
    const DAT1: f64 = 0.75;
    const KEXSH: usize = 10;
    let n = h.nrows();
    if n == 0 {
        return Ok(());
    }
    if ilo == ihi {
        w[ilo] = h[(ilo, ilo)];
        return Ok(());
    }
    if ihi >= ilo + 3 {
        for j in ilo..=ihi - 3 {
            h[(j + 2, j)] = CZERO;
            h[(j + 3, j)] = CZERO;
        }
    }
    if ilo + 2 <= ihi {
        h[(ihi, ihi - 2)] = CZERO;
    }
    let (jlo, jhi) = if wantt { (0, n - 1) } else { (ilo, ihi) };
    for i in ilo + 1..=ihi {
        let hv = h[(i, i - 1)];
        if hv.im != 0.0 {
            let sc = hv / cabs1(hv);
            let sc = sc.conj() / sc.norm();
            h[(i, i - 1)] = Complex::new(hv.norm(), 0.0);
            for c in i..=jhi {
                h[(i, c)] = sc * h[(i, c)];
            }
            for r in jlo..=jhi.min(i + 1) {
                h[(r, i)] = sc.conj() * h[(r, i)];
            }
            if wantz {
                for r in iloz..=ihiz {
                    z[(r, i)] = sc.conj() * z[(r, i)];
                }
            }
        }
    }
    let nh = ihi - ilo + 1;
    let safmin = f64::MIN_POSITIVE;
    let ulp = EPS;
    let smlnum = safmin * (nh as f64 / ulp);
    let (mut i1, mut i2) = (0usize, n - 1);
    let itmax = 30 * nh.max(10);
    let mut kdefl = 0usize;
    let mut i = ihi as isize;
    while i >= ilo as isize {
        let iu = i as usize;
        let mut l = ilo;
        let mut converged = false;
        for _its in 0..=itmax {
            // A single small subdiagonal element.
            let mut k = iu;
            while k > l {
                if cabs1(h[(k, k - 1)]) <= smlnum {
                    break;
                }
                let mut tst = cabs1(h[(k - 1, k - 1)]) + cabs1(h[(k, k)]);
                if tst == 0.0 {
                    if k >= ilo + 2 {
                        tst += h[(k - 1, k - 2)].re.abs();
                    }
                    if k < ihi {
                        tst += h[(k + 1, k)].re.abs();
                    }
                }
                if h[(k, k - 1)].re.abs() <= ulp * tst {
                    let ab = cabs1(h[(k, k - 1)]).max(cabs1(h[(k - 1, k)]));
                    let ba = cabs1(h[(k, k - 1)]).min(cabs1(h[(k - 1, k)]));
                    let diff = h[(k - 1, k - 1)] - h[(k, k)];
                    let aa = cabs1(h[(k, k)]).max(cabs1(diff));
                    let bb = cabs1(h[(k, k)]).min(cabs1(diff));
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > ilo {
                h[(l, l - 1)] = CZERO;
            }
            if l >= iu {
                converged = true;
                break;
            }
            kdefl += 1;
            if !wantt {
                i1 = l;
                i2 = iu;
            }
            let t: C64 = if kdefl.is_multiple_of(2 * KEXSH) {
                let s = DAT1 * h[(iu, iu - 1)].re.abs();
                h[(iu, iu)] + s
            } else if kdefl.is_multiple_of(KEXSH) {
                let s = DAT1 * h[(l + 1, l)].re.abs();
                h[(l, l)] + s
            } else {
                // Wilkinson's shift.
                let mut t = h[(iu, iu)];
                let u = h[(iu - 1, iu)].sqrt() * h[(iu, iu - 1)].sqrt();
                let s = cabs1(u);
                if s != 0.0 {
                    let x = (h[(iu - 1, iu - 1)] - t) * 0.5;
                    let sx = cabs1(x);
                    let s = s.max(cabs1(x));
                    let xs = x / s;
                    let us = u / s;
                    let mut y = (xs * xs + us * us).sqrt() * s;
                    if sx > 0.0 {
                        let xn = x / sx;
                        if xn.re * y.re + xn.im * y.im < 0.0 {
                            y = -y;
                        }
                    }
                    t -= u * zladiv(u, x + y);
                }
                t
            };
            // Two consecutive small subdiagonal elements.
            let mut m = iu - 1;
            let mut v = [CZERO; 2];
            let mut found = false;
            while m > l {
                let h11 = h[(m, m)];
                let h22 = h[(m + 1, m + 1)];
                let mut h11s = h11 - t;
                let mut h21 = h[(m + 1, m)].re;
                let s = cabs1(h11s) + h21.abs();
                h11s /= s;
                h21 /= s;
                v = [h11s, Complex::new(h21, 0.0)];
                let h10 = h[(m, m - 1)].re;
                if h10.abs() * h21.abs() <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22))) {
                    found = true;
                    break;
                }
                m -= 1;
            }
            if !found {
                m = l;
                let h11 = h[(l, l)];
                let mut h11s = h11 - t;
                let mut h21 = h[(l + 1, l)].re;
                let s = cabs1(h11s) + h21.abs();
                h11s /= s;
                h21 /= s;
                v = [h11s, Complex::new(h21, 0.0)];
            }
            // Single-shift QR step.
            for k in m..iu {
                if k > m {
                    v = [h[(k, k - 1)], h[(k + 1, k - 1)]];
                }
                let mut x1 = [v[1]];
                let (beta, t1) = zlarfg(v[0], &mut x1);
                v = [beta, x1[0]];
                if k > m {
                    h[(k, k - 1)] = v[0];
                    h[(k + 1, k - 1)] = CZERO;
                }
                let v2 = v[1];
                let t2 = (t1 * v2).re;
                for j in k..=i2 {
                    let sum = t1.conj() * h[(k, j)] + h[(k + 1, j)] * t2;
                    h[(k, j)] -= sum;
                    h[(k + 1, j)] -= sum * v2;
                }
                for j in i1..=(k + 2).min(iu) {
                    let sum = t1 * h[(j, k)] + h[(j, k + 1)] * t2;
                    h[(j, k)] -= sum;
                    h[(j, k + 1)] -= sum * v2.conj();
                }
                if wantz {
                    for j in iloz..=ihiz {
                        let sum = t1 * z[(j, k)] + z[(j, k + 1)] * t2;
                        z[(j, k)] -= sum;
                        z[(j, k + 1)] -= sum * v2.conj();
                    }
                }
                if k == m && m > l {
                    let temp = CONE - t1;
                    let temp = temp / temp.norm();
                    h[(m + 1, m)] *= temp.conj();
                    if m + 2 <= iu {
                        h[(m + 2, m + 1)] *= temp;
                    }
                    for j in m..=iu {
                        if j != m + 1 {
                            if i2 > j {
                                for c in j + 1..=i2 {
                                    h[(j, c)] = temp * h[(j, c)];
                                }
                            }
                            for r in i1..j {
                                h[(r, j)] = temp.conj() * h[(r, j)];
                            }
                            if wantz {
                                for r in iloz..=ihiz {
                                    z[(r, j)] = temp.conj() * z[(r, j)];
                                }
                            }
                        }
                    }
                }
            }
            // Ensure H(i, i−1) is real.
            let temp = h[(iu, iu - 1)];
            if temp.im != 0.0 {
                let rtemp = temp.norm();
                h[(iu, iu - 1)] = Complex::new(rtemp, 0.0);
                let temp = temp / rtemp;
                if i2 > iu {
                    for c in iu + 1..=i2 {
                        h[(iu, c)] = temp.conj() * h[(iu, c)];
                    }
                }
                for r in i1..iu {
                    h[(r, iu)] = temp * h[(r, iu)];
                }
                if wantz {
                    for r in iloz..=ihiz {
                        z[(r, iu)] = temp * z[(r, iu)];
                    }
                }
            }
        }
        if !converged {
            return Err(iu);
        }
        w[iu] = h[(iu, iu)];
        kdefl = 0;
        i = l as isize - 1;
    }
    Ok(())
}

/// LAPACK `zhseqr` (small-matrix path): eigenvalues of the Hessenberg `h`, which becomes the
/// Schur form `T` when `wantt`, with the Schur vectors accumulated into `z` when `wantz`.
fn hseqr(
    wantt: bool,
    wantz: bool,
    h: &mut DMatrix<C64>,
    ilo: usize,
    ihi: usize,
    z: &mut DMatrix<C64>,
) -> Result<Vec<C64>, LinalgError> {
    let n = h.nrows();
    let mut w = vec![CZERO; n];
    if n == 0 {
        return Ok(w);
    }
    for i in 0..ilo {
        w[i] = h[(i, i)];
    }
    for i in ihi + 1..n {
        w[i] = h[(i, i)];
    }
    if ilo == ihi {
        w[ilo] = h[(ilo, ilo)];
        return Ok(w);
    }
    lahqr(wantt, wantz, h, ilo, ihi, &mut w, z, ilo, ihi).map_err(|i| {
        LinalgError::ConvergenceFailure {
            detail: format!(
                "the QR algorithm failed to compute all the eigenvalues ({} did not converge)",
                i + 1
            ),
        }
    })?;
    if wantt && n > 2 {
        for j in 0..n {
            for i in j + 2..n {
                h[(i, j)] = CZERO;
            }
        }
    }
    Ok(w)
}

/// Zero everything below the first subdiagonal (the Householder vectors `gehd2` stored there).
fn clear_below_subdiagonal(h: &mut DMatrix<C64>) {
    let n = h.nrows();
    for j in 0..n {
        for i in j + 2..n {
            h[(i, j)] = CZERO;
        }
    }
}

/// `scipy.linalg.hessenberg(a, calc_q=True)` on complex input (`gehrd` without balancing).
///
/// # Errors
/// Shape and finiteness errors.
pub fn hessenberg(
    a: &[Vec<C64>],
    options: DecompOptions,
) -> Result<ComplexHessenbergResult, LinalgError> {
    let n = square(a, options)?;
    if n <= 2 {
        return Ok(ComplexHessenbergResult {
            h: a.to_vec(),
            q: rows_of(&identity(n)),
        });
    }
    let mut h = to_dm(a, n, n);
    let tau = gehd2(&mut h, 0, n - 1);
    let q = unghr(&h, &tau, 0, n - 1);
    clear_below_subdiagonal(&mut h);
    Ok(ComplexHessenbergResult {
        h: rows_of(&h),
        q: rows_of(&q),
    })
}

/// The complex Schur form of `a` as `zgees` computes it (permutation balancing only).
fn schur_dm(a: &DMatrix<C64>) -> Result<(DMatrix<C64>, DMatrix<C64>), LinalgError> {
    let n = a.nrows();
    let mut h = a.clone();
    let (ilo, ihi, sc) = gebal(&mut h, true, false);
    let tau = gehd2(&mut h, ilo, ihi);
    let mut z = unghr(&h, &tau, ilo, ihi);
    clear_below_subdiagonal(&mut h);
    if n > 0 {
        hseqr(true, true, &mut h, ilo, ihi, &mut z)?;
    }
    gebak(&mut z, ilo, ihi, &sc, false, true);
    Ok((h, z))
}

/// `scipy.linalg.schur(a, output='complex')`: `A = Z·T·Zᴴ`, `T` upper triangular with the
/// eigenvalues on its diagonal. Real input converts with [`to_complex`] (SciPy does the same
/// for `output='complex'`).
///
/// # Errors
/// Shape and finiteness errors, and [`LinalgError::ConvergenceFailure`] when the QR iteration
/// does not converge.
pub fn schur(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexSchurResult, LinalgError> {
    let n = square(a, options)?;
    reject_nan(a)?;
    let (t, z) = schur_dm(&to_dm(a, n, n))?;
    Ok(ComplexSchurResult {
        t: rows_of(&t),
        z: rows_of(&z),
    })
}

/// LAPACK `ztrevc3('R', 'B')`: right eigenvectors of the upper-triangular `t`, back-transformed
/// by the columns of `vr` (overwritten), each scaled so its largest `|re| + |im|` is 1.
fn trevc_right(t: &DMatrix<C64>, vr: &mut DMatrix<C64>) {
    let n = t.nrows();
    let ulp = EPS;
    let smlnum = f64::MIN_POSITIVE * (n as f64 / ulp);
    let mut tt = t.clone();
    for ki in (0..n).rev() {
        let smin = (ulp * cabs1(t[(ki, ki)])).max(smlnum);
        let mut x: Vec<C64> = (0..ki).map(|k| -t[(k, ki)]).collect();
        for k in 0..ki {
            tt[(k, k)] = t[(k, k)] - t[(ki, ki)];
            if cabs1(tt[(k, k)]) < smin {
                tt[(k, k)] = Complex::new(smin, 0.0);
            }
        }
        for j in (0..ki).rev() {
            if x[j] != CZERO {
                x[j] = zladiv(x[j], tt[(j, j)]);
                let xj = x[j];
                for i in 0..j {
                    x[i] -= xj * tt[(i, j)];
                }
            }
        }
        for r in 0..n {
            let mut s = vr[(r, ki)];
            for (k, &xk) in x.iter().enumerate() {
                s += vr[(r, k)] * xk;
            }
            vr[(r, ki)] = s;
        }
        let remax = (0..n).map(|r| cabs1(vr[(r, ki)])).fold(0.0_f64, f64::max);
        if remax > 0.0 {
            let inv = 1.0 / remax;
            for r in 0..n {
                vr[(r, ki)] *= inv;
            }
        }
        for k in 0..ki {
            tt[(k, k)] = t[(k, k)];
        }
    }
}

/// `scipy.linalg.eig(a)` on complex input (`zgeev`): eigenvalues and right eigenvectors, each
/// of unit 2-norm with its largest-modulus component real and positive.
///
/// # Errors
/// Shape and finiteness errors, and [`LinalgError::ConvergenceFailure`] when the QR iteration
/// does not converge.
pub fn eig(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexEigResult, LinalgError> {
    let n = square(a, options)?;
    reject_nan(a)?;
    if n == 0 {
        return Ok(ComplexEigResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }
    let mut h = to_dm(a, n, n);
    let (ilo, ihi, sc) = gebal(&mut h, true, true);
    let tau = gehd2(&mut h, ilo, ihi);
    let mut vr = unghr(&h, &tau, ilo, ihi);
    clear_below_subdiagonal(&mut h);
    let w = hseqr(true, true, &mut h, ilo, ihi, &mut vr)?;
    trevc_right(&h, &mut vr);
    gebak(&mut vr, ilo, ihi, &sc, true, true);
    for c in 0..n {
        let col: Vec<C64> = (0..n).map(|r| vr[(r, c)]).collect();
        let scl = 1.0 / dznrm2(&col);
        for r in 0..n {
            vr[(r, c)] *= scl;
        }
        let mut k = 0usize;
        let mut best = -1.0_f64;
        for r in 0..n {
            let mag = vr[(r, c)].re.powi(2) + vr[(r, c)].im.powi(2);
            if mag > best {
                best = mag;
                k = r;
            }
        }
        let tmp = vr[(k, c)].conj() / best.sqrt();
        for r in 0..n {
            vr[(r, c)] = tmp * vr[(r, c)];
        }
        vr[(k, c)] = Complex::new(vr[(k, c)].re, 0.0);
    }
    Ok(ComplexEigResult {
        eigenvalues: w,
        eigenvectors: rows_of(&vr),
    })
}

/// `scipy.linalg.eigvals(a)` on complex input (`zgeev` with `jobvr='N'`).
///
/// # Errors
/// As [`eig`].
pub fn eigvals(a: &[Vec<C64>], options: DecompOptions) -> Result<Vec<C64>, LinalgError> {
    let n = square(a, options)?;
    reject_nan(a)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut h = to_dm(a, n, n);
    let (ilo, ihi, _sc) = gebal(&mut h, true, true);
    let _tau = gehd2(&mut h, ilo, ihi);
    clear_below_subdiagonal(&mut h);
    let mut z = DMatrix::from_element(0, 0, CZERO);
    hseqr(false, false, &mut h, ilo, ihi, &mut z)
}

// ══════════════════════════════════════════════════════════════════════
// Hermitian eigenproblem and SVD
// ══════════════════════════════════════════════════════════════════════

/// The Hermitian matrix defined by `a`'s lower triangle (SciPy's `eigh(lower=True)` default),
/// with the diagonal's imaginary parts ignored as LAPACK's `zheevr` ignores them.
fn hermitian_from_lower(a: &[Vec<C64>], n: usize) -> DMatrix<C64> {
    DMatrix::from_fn(n, n, |i, j| match i.cmp(&j) {
        std::cmp::Ordering::Equal => Complex::new(a[i][i].re, 0.0),
        std::cmp::Ordering::Greater => a[i][j],
        std::cmp::Ordering::Less => a[j][i].conj(),
    })
}

fn hermitian_eigen(
    a: &[Vec<C64>],
    options: DecompOptions,
) -> Result<(Vec<f64>, DMatrix<C64>), LinalgError> {
    let n = square(a, options)?;
    reject_nan(a)?;
    if n == 0 {
        return Ok((Vec::new(), DMatrix::from_element(0, 0, CZERO)));
    }
    let m = hermitian_from_lower(a, n);
    let eig = nalgebra::linalg::SymmetricEigen::try_new(m, EPS, 0).ok_or_else(|| {
        LinalgError::ConvergenceFailure {
            detail: "the Hermitian eigenvalue algorithm failed to converge".to_string(),
        }
    })?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    let vals = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let vecs = DMatrix::from_fn(n, n, |r, c| eig.eigenvectors[(r, order[c])]);
    Ok((vals, vecs))
}

/// `scipy.linalg.eigh(a)` on a Hermitian matrix (its lower triangle is read): ascending real
/// eigenvalues and orthonormal eigenvectors as columns.
///
/// # Errors
/// Shape and finiteness errors, and [`LinalgError::ConvergenceFailure`].
pub fn eigh(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexEighResult, LinalgError> {
    let (vals, vecs) = hermitian_eigen(a, options)?;
    Ok(ComplexEighResult {
        eigenvalues: vals,
        eigenvectors: rows_of(&vecs),
    })
}

/// `scipy.linalg.eigvalsh(a)` on a Hermitian matrix: ascending real eigenvalues.
///
/// # Errors
/// As [`eigh`].
pub fn eigvalsh(a: &[Vec<C64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    Ok(hermitian_eigen(a, options)?.0)
}

/// Thin SVD `A = U·diag(s)·Vᴴ` with `s` descending (U m×k, V n×k).
fn thin_svd(a: &DMatrix<C64>) -> Result<(DMatrix<C64>, Vec<f64>, DMatrix<C64>), LinalgError> {
    let (m, n) = a.shape();
    let k = m.min(n);
    if k == 0 {
        return Ok((
            DMatrix::from_element(m, 0, CZERO),
            Vec::new(),
            DMatrix::from_element(n, 0, CZERO),
        ));
    }
    let svd = nalgebra::linalg::SVD::try_new(a.clone(), true, true, EPS, 0).ok_or_else(|| {
        LinalgError::ConvergenceFailure {
            detail: "SVD did not converge".to_string(),
        }
    })?;
    let u = svd.u.ok_or_else(|| LinalgError::ConvergenceFailure {
        detail: "SVD did not produce U".to_string(),
    })?;
    let v_t = svd.v_t.ok_or_else(|| LinalgError::ConvergenceFailure {
        detail: "SVD did not produce Vᴴ".to_string(),
    })?;
    let s: Vec<f64> = svd.singular_values.iter().copied().collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| s[j].total_cmp(&s[i]));
    let s_sorted = order.iter().map(|&i| s[i]).collect();
    let u_sorted = DMatrix::from_fn(m, k, |r, c| u[(r, order[c])]);
    let v_sorted = DMatrix::from_fn(n, k, |r, c| v_t[(order[c], r)].conj());
    Ok((u_sorted, s_sorted, v_sorted))
}

/// `scipy.linalg.svd(a, full_matrices)` on complex input: `A = U·diag(s)·Vᴴ` with `s`
/// descending. `full_matrices = true` (SciPy's default) returns `U` m×m and `Vᴴ` n×n, the
/// columns beyond `min(m, n)` completing the unitary bases.
///
/// # Errors
/// [`LinalgError::NonFiniteInput`] for a NaN entry (SciPy raises for NaN even without
/// `check_finite`), plus shape errors and [`LinalgError::ConvergenceFailure`].
pub fn svd(
    a: &[Vec<C64>],
    full_matrices: bool,
    options: DecompOptions,
) -> Result<ComplexSvdResult, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    reject_nan(a)?;
    let (u, s, v) = thin_svd(&to_dm(a, m, n))?;
    let (u, v) = if full_matrices {
        (
            complete_unitary_or_identity(&u, m),
            complete_unitary_or_identity(&v, n),
        )
    } else {
        (u, v)
    };
    Ok(ComplexSvdResult {
        u: rows_of(&u),
        s,
        vh: rows_of(&v.adjoint()),
    })
}

fn complete_unitary_or_identity(u: &DMatrix<C64>, m: usize) -> DMatrix<C64> {
    if u.ncols() == 0 {
        identity(m)
    } else {
        complete_unitary(u)
    }
}

/// `scipy.linalg.svdvals(a)` on complex input: singular values, descending.
///
/// # Errors
/// As [`svd`].
pub fn svdvals(a: &[Vec<C64>], options: DecompOptions) -> Result<Vec<f64>, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    reject_nan(a)?;
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let svd =
        nalgebra::linalg::SVD::try_new(to_dm(a, m, n), false, false, EPS, 0).ok_or_else(|| {
            LinalgError::ConvergenceFailure {
                detail: "SVD did not converge".to_string(),
            }
        })?;
    let mut s: Vec<f64> = svd.singular_values.iter().copied().collect();
    s.sort_by(|x, y| y.total_cmp(x));
    Ok(s)
}

/// `scipy.linalg.lstsq(a, b, cond)` on complex input (SciPy's default `gelsd` semantics):
/// the minimum-norm least-squares solution, treating singular values `≤ cond·σ₁` as zero
/// (`cond` defaults to machine epsilon).
///
/// # Errors
/// Shape errors (`b` must have `m` entries), finiteness errors and SVD non-convergence.
pub fn lstsq(
    a: &[Vec<C64>],
    b: &[C64],
    cond: Option<f64>,
    options: DecompOptions,
) -> Result<ComplexLstsqResult, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    if b.len() != m {
        return Err(LinalgError::IncompatibleShapes {
            a_shape: (m, n),
            b_len: b.len(),
        });
    }
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && b.iter().any(|&z| !is_finite(z))
    {
        return Err(LinalgError::NonFiniteInput);
    }
    let am = to_dm(a, m, n);
    let (u, s, v) = thin_svd(&am)?;
    let rcond = cond.filter(|c| *c >= 0.0).unwrap_or(EPS);
    let thresh = rcond * s.first().copied().unwrap_or(0.0);
    let rank = s.iter().filter(|&&sv| sv > thresh).count();
    let mut x = vec![CZERO; n];
    for (c, &sc) in s.iter().enumerate().take(rank) {
        let mut coef = CZERO;
        for r in 0..m {
            coef += u[(r, c)].conj() * b[r];
        }
        let coef = coef / sc;
        for (r, xr) in x.iter_mut().enumerate() {
            *xr += v[(r, c)] * coef;
        }
    }
    let residues = (rank == n && m > n).then(|| {
        (0..m)
            .map(|r| {
                let mut ax = CZERO;
                for c in 0..n {
                    ax += am[(r, c)] * x[c];
                }
                (b[r] - ax).norm_sqr()
            })
            .sum()
    });
    Ok(ComplexLstsqResult {
        x,
        residues,
        rank,
        s,
    })
}

/// `scipy.linalg.pinv(a, atol, rtol)` on complex input: singular values above
/// `atol + rtol·σ₁` are inverted (`atol` defaults to 0, `rtol` to `max(m, n)·eps`).
///
/// # Errors
/// [`LinalgError::InvalidPinvThreshold`] for a negative tolerance, plus the errors of [`svd`].
pub fn pinv(
    a: &[Vec<C64>],
    atol: Option<f64>,
    rtol: Option<f64>,
    options: DecompOptions,
) -> Result<ComplexMatrix, LinalgError> {
    let (m, n) = shape(a)?;
    hardened_dimension_check(options.mode, m, n)?;
    check_finite(a, options.mode, options.check_finite)?;
    reject_nan(a)?;
    let atol = atol.unwrap_or(0.0);
    let rtol = rtol.unwrap_or(m.max(n) as f64 * EPS);
    if atol < 0.0 || rtol < 0.0 {
        return Err(LinalgError::InvalidPinvThreshold);
    }
    let (u, s, v) = thin_svd(&to_dm(a, m, n))?;
    let maxs = s.first().copied().unwrap_or(0.0);
    let val = atol + maxs * rtol;
    let rank = s.iter().filter(|&&sv| sv > val).count();
    // B = (U[:, :rank] / s[:rank] @ Vh[:rank]).conj().T = V[:, :rank] · diag(1/s) · U[:, :rank]ᴴ
    let mut out = vec![vec![CZERO; m]; n];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            let mut acc = CZERO;
            for (c, &sc) in s.iter().enumerate().take(rank) {
                acc += (u[(j, c)] / sc).conj() * v[(i, c)];
            }
            *slot = acc;
        }
    }
    Ok(out)
}

/// `scipy.linalg.norm(a, ord)` for a complex matrix: Frobenius, spectral (largest singular
/// value), 1-norm (maximum column sum of moduli) or ∞-norm (maximum row sum).
///
/// # Errors
/// Ragged input and non-finite entries (under `check_finite`).
pub fn norm(a: &[Vec<C64>], kind: NormKind, options: DecompOptions) -> Result<f64, LinalgError> {
    let (m, n) = shape(a)?;
    check_finite(a, options.mode, options.check_finite)?;
    if m == 0 || n == 0 {
        return Ok(0.0);
    }
    Ok(match kind {
        NormKind::Fro => {
            let flat: Vec<C64> = a.iter().flatten().copied().collect();
            dznrm2(&flat)
        }
        NormKind::Spectral => svdvals(a, options)?.first().copied().unwrap_or(0.0),
        NormKind::One => one_norm(&to_dm(a, m, n)),
        NormKind::Inf => a
            .iter()
            .map(|row| row.iter().map(|z| z.norm()).sum::<f64>())
            .fold(0.0_f64, f64::max),
    })
}

// ══════════════════════════════════════════════════════════════════════
// Matrix functions: expm, sqrtm, logm
// ══════════════════════════════════════════════════════════════════════

/// Solve `A·X = B` for a matrix right-hand side through one LU factorization.
fn lu_solve_matrix(a: &DMatrix<C64>, b: &DMatrix<C64>) -> Option<DMatrix<C64>> {
    let n = a.nrows();
    let mut lu = a.clone();
    let (piv, info) = getrf(&mut lu);
    if info.is_some() {
        return None;
    }
    let mut out = b.clone();
    for c in 0..b.ncols() {
        let mut col: Vec<C64> = (0..n).map(|r| b[(r, c)]).collect();
        getrs(&lu, &piv, &mut col, 0);
        for r in 0..n {
            out[(r, c)] = col[r];
        }
    }
    Some(out)
}

const PADE3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
const PADE5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
const PADE7: [f64; 8] = [
    17_297_280.0,
    8_648_640.0,
    1_995_840.0,
    277_200.0,
    25_200.0,
    1512.0,
    56.0,
    1.0,
];
const PADE9: [f64; 10] = [
    17_643_225_600.0,
    8_821_612_800.0,
    2_075_673_600.0,
    302_702_400.0,
    30_270_240.0,
    2_162_160.0,
    110_880.0,
    3960.0,
    90.0,
    1.0,
];
const PADE13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];
const THETA: [f64; 5] = [
    1.495_585_217_958_292e-2,
    2.539_398_330_063_23e-1,
    9.504_178_996_162_932e-1,
    2.097_847_961_257_068,
    5.371_920_351_148_152,
];

fn expm_dm(a: &DMatrix<C64>) -> Result<DMatrix<C64>, LinalgError> {
    let n = a.nrows();
    let ident = identity(n);
    let norm1 = one_norm(a);
    if norm1 == 0.0 {
        return Ok(ident);
    }
    if !norm1.is_finite() {
        return Ok(DMatrix::from_element(
            n,
            n,
            Complex::new(f64::NAN, f64::NAN),
        ));
    }
    let low: Option<&[f64]> = if norm1 <= THETA[0] {
        Some(&PADE3)
    } else if norm1 <= THETA[1] {
        Some(&PADE5)
    } else if norm1 <= THETA[2] {
        Some(&PADE7)
    } else if norm1 <= THETA[3] {
        Some(&PADE9)
    } else {
        None
    };
    let singular = || LinalgError::ConvergenceFailure {
        detail: "expm: the Padé denominator is singular".to_string(),
    };
    if let Some(b) = low {
        let a2 = a * a;
        let mut pe = &ident * Complex::new(b[0], 0.0) + &a2 * Complex::new(b[2], 0.0);
        let mut po = &ident * Complex::new(b[1], 0.0) + &a2 * Complex::new(b[3], 0.0);
        let mut cur = a2.clone();
        for j in 2..=(b.len() - 1) / 2 {
            cur = &cur * &a2;
            pe += &cur * Complex::new(b[2 * j], 0.0);
            po += &cur * Complex::new(b[2 * j + 1], 0.0);
        }
        let w = a * po;
        return lu_solve_matrix(&(&pe - &w), &(&pe + &w)).ok_or_else(singular);
    }
    let s = ((norm1 / THETA[4]).log2().ceil()).max(0.0) as i32;
    let scaled = a * Complex::new(2.0_f64.powi(-s), 0.0);
    let c = |k: usize| Complex::new(PADE13[k], 0.0);
    let a2 = &scaled * &scaled;
    let a4 = &a2 * &a2;
    let a6 = &a2 * &a4;
    let u_inner = &a6 * c(13) + &a4 * c(11) + &a2 * c(9);
    let u_poly = &a6 * u_inner + &a6 * c(7) + &a4 * c(5) + &a2 * c(3) + &ident * c(1);
    let u = &scaled * u_poly;
    let v_inner = &a6 * c(12) + &a4 * c(10) + &a2 * c(8);
    let v = &a6 * v_inner + &a6 * c(6) + &a4 * c(4) + &a2 * c(2) + &ident * c(0);
    let mut r = lu_solve_matrix(&(&v - &u), &(&v + &u)).ok_or_else(singular)?;
    for _ in 0..s {
        r = &r * &r;
    }
    Ok(r)
}

/// `scipy.linalg.expm` on complex input: scaling and squaring with Padé degrees 3–13
/// (Higham 2005), choosing the degree from the 1-norm.
///
/// # Errors
/// Shape and finiteness errors.
pub fn expm(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexMatrix, LinalgError> {
    let n = square(a, options)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    Ok(rows_of(&expm_dm(&to_dm(a, n, n))?))
}

/// The principal square root of an upper-triangular matrix (Björck–Hammarling recurrence, the
/// scalar form of SciPy's `_sqrtm_triu`). Where a superdiagonal entry has a zero denominator
/// `r_ii + r_jj` (a singular block with no square root that is a function of `T`), the entry is
/// the IEEE quotient `num / 0` SciPy produces (`sqrtm([[0, 1], [0, 0]])[0, 1] = inf + nan·i`),
/// or 0 when the numerator is 0 too.
fn sqrtm_triu(t: &DMatrix<C64>) -> DMatrix<C64> {
    let n = t.nrows();
    let mut r = DMatrix::from_element(n, n, CZERO);
    for i in 0..n {
        r[(i, i)] = t[(i, i)].sqrt();
    }
    for j in 0..n {
        for i in (0..j).rev() {
            let mut s = CZERO;
            for k in i + 1..j {
                s += r[(i, k)] * r[(k, j)];
            }
            let denom = r[(i, i)] + r[(j, j)];
            let num = t[(i, j)] - s;
            r[(i, j)] = if denom != CZERO {
                num / denom
            } else if num == CZERO {
                CZERO
            } else {
                Complex::new(num.re / 0.0, num.im / 0.0)
            };
        }
    }
    r
}

/// `Z·M·Zᴴ`.
fn similarity(z: &DMatrix<C64>, m: &DMatrix<C64>) -> DMatrix<C64> {
    z * m * z.adjoint()
}

/// `scipy.linalg.sqrtm` on complex input: the principal square root by the Schur method.
///
/// A matrix with no square root that is a function of it (a singular non-diagonalizable block)
/// gets SciPy's non-finite entries: `sqrtm([[0, 1], [0, 0]]) = [[0, inf + nan·i], [0, 0]]`
/// (SciPy also warns "Matrix is singular"). An upper-triangular input skips the Schur form,
/// as SciPy's does, so those entries do not spread through a similarity transform.
///
/// # Errors
/// Shape and finiteness errors and Schur non-convergence.
pub fn sqrtm(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexMatrix, LinalgError> {
    let n = square(a, options)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let am = to_dm(a, n, n);
    if is_upper_triangular(&am) {
        return Ok(rows_of(&sqrtm_triu(&am)));
    }
    let (t, z) = schur_dm(&am)?;
    Ok(rows_of(&similarity(&z, &sqrtm_triu(&t))))
}

/// Gauss–Legendre nodes and weights on `[-1, 1]` (`scipy.special.roots_legendre(m)`), nodes
/// ascending.
fn gauss_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; m];
    let mut weights = vec![0.0; m];
    for i in 0..m {
        // Newton on P_m from Tricomi's initial guess, largest node first.
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0_f64, x);
            for k in 2..=m {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if m == 0 {
                1.0
            } else if m == 1 {
                x
            } else {
                p1
            };
            let pm1 = if m == 1 { 1.0 } else { p0 };
            dp = m as f64 * (x * p - pm1) / (x * x - 1.0);
            let dx = p / dp;
            x -= dx;
            if dx.abs() <= 1e-16 * x.abs().max(1.0) {
                break;
            }
        }
        nodes[m - 1 - i] = x;
        weights[m - 1 - i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// SciPy `_unwindk`: the unwinding number `⌈(Im z − π) / 2π⌉`.
fn unwindk(z: C64) -> f64 {
    ((z.im - std::f64::consts::PI) / (2.0 * std::f64::consts::PI)).ceil()
}

/// SciPy `_briggs_helper_function`: `a^(1/2^k) − 1` with less cancellation (Al-Mohy 2012).
fn briggs_helper(a: C64, k: u32) -> C64 {
    match k {
        0 => a - CONE,
        1 => a.sqrt() - CONE,
        _ => {
            let mut a = a;
            let mut k_hat = k;
            if a.arg() >= std::f64::consts::FRAC_PI_2 {
                a = a.sqrt();
                k_hat = k - 1;
            }
            let z0 = a - CONE;
            a = a.sqrt();
            let mut r = CONE + a;
            for _ in 1..k_hat {
                a = a.sqrt();
                r *= CONE + a;
            }
            z0 / r
        }
    }
}

/// SciPy `_fractional_power_superdiag_entry` (Higham & Lin 2011, Eq. 5.6).
fn fractional_power_superdiag_entry(l1: C64, l2: C64, t12: C64, p: f64) -> C64 {
    if l1 == l2 {
        t12 * p * l1.powf(p - 1.0)
    } else if (l2 - l1).norm() > (l1 + l2).norm() / 2.0 {
        t12 * (l2.powf(p) - l1.powf(p)) / (l2 - l1)
    } else {
        let z = (l2 - l1) / (l2 + l1);
        let log_l1 = l1.ln();
        let log_l2 = l2.ln();
        let arctanh_z = z.atanh();
        let tmp_a = t12 * ((log_l2 + log_l1) * (p / 2.0)).exp();
        let tmp_u = unwindk(log_l2 - log_l1);
        let tmp_b = if tmp_u != 0.0 {
            (arctanh_z + Complex::new(0.0, std::f64::consts::PI * tmp_u)) * p
        } else {
            arctanh_z * p
        };
        let tmp_c = tmp_b.sinh() * 2.0 / (l2 - l1);
        tmp_a * tmp_c
    }
}

/// SciPy `_logm_superdiag_entry` (Higham 2008, Eq. 11.28, modified).
fn logm_superdiag_entry(l1: C64, l2: C64, t12: C64) -> C64 {
    if l1 == l2 {
        t12 / l1
    } else if (l2 - l1).norm() > (l1 + l2).norm() / 2.0 {
        t12 * (l2.ln() - l1.ln()) / (l2 - l1)
    } else {
        let z = (l2 - l1) / (l2 + l1);
        let u = unwindk(l2.ln() - l1.ln());
        if u != 0.0 {
            t12 * 2.0 * (z.atanh() + Complex::new(0.0, std::f64::consts::PI * u)) / (l2 - l1)
        } else {
            t12 * 2.0 * z.atanh() / (l2 - l1)
        }
    }
}

/// `‖(T − I)^p‖₁^(1/p)`, computed exactly (SciPy estimates it with `onenormest`, which is exact
/// for n ≤ 2 and a lower bound otherwise).
fn m1_power_norm_root(t: &DMatrix<C64>, p: i32) -> f64 {
    let n = t.nrows();
    let tm1 = t - identity(n);
    let mut acc = tm1.clone();
    for _ in 1..p {
        acc = &acc * &tm1;
    }
    one_norm(&acc).powf(1.0 / f64::from(p))
}

fn has_principal_branch(t0: &DMatrix<C64>) -> bool {
    (0..t0.nrows()).all(|i| t0[(i, i)].re > 0.0 || t0[(i, i)].im != 0.0)
}

/// SciPy `_inverse_squaring_helper`: square roots of `T0` until a Padé degree `m` is accurate;
/// returns `(R = T0^(1/2^s) − I, s, m)`.
fn inverse_squaring_helper(
    t0: &DMatrix<C64>,
    theta: &[f64; 17],
) -> Result<(DMatrix<C64>, u32, usize), LinalgError> {
    let n = t0.nrows();
    let mut t = t0.clone();
    let mut tmp_diag: Vec<C64> = (0..n).map(|i| t0[(i, i)]).collect();
    if tmp_diag.contains(&CZERO) {
        return Err(LinalgError::SingularMatrix);
    }
    let mut s0 = 0u32;
    while tmp_diag
        .iter()
        .map(|&d| (d - CONE).norm())
        .fold(0.0_f64, f64::max)
        > theta[7]
    {
        tmp_diag.iter_mut().for_each(|d| *d = d.sqrt());
        s0 += 1;
    }
    for _ in 0..s0 {
        t = sqrtm_triu(&t);
    }
    let mut s = s0;
    let mut k = 0;
    let d2 = m1_power_norm_root(&t, 2);
    let mut d3 = m1_power_norm_root(&t, 3);
    let a2 = d2.max(d3);
    let mut m = None;
    for i in [1usize, 2] {
        if a2 <= theta[i] {
            m = Some(i);
            break;
        }
    }
    while m.is_none() {
        if s > s0 {
            d3 = m1_power_norm_root(&t, 3);
        }
        let d4 = m1_power_norm_root(&t, 4);
        let a3 = d3.max(d4);
        if a3 <= theta[7] {
            let j1 = (3..=7).find(|&i| a3 <= theta[i]).unwrap_or(7);
            if j1 <= 6 {
                m = Some(j1);
                break;
            } else if a3 / 2.0 <= theta[5] && k < 2 {
                k += 1;
                t = sqrtm_triu(&t);
                s += 1;
                continue;
            }
        }
        let d5 = m1_power_norm_root(&t, 5);
        let a4 = d4.max(d5);
        let eta = a3.min(a4);
        for i in [6usize, 7] {
            if eta <= theta[i] {
                m = Some(i);
                break;
            }
        }
        if m.is_some() {
            break;
        }
        t = sqrtm_triu(&t);
        s += 1;
    }
    let mut r = &t - identity(n);
    if has_principal_branch(t0) {
        for j in 0..n {
            r[(j, j)] = briggs_helper(t0[(j, j)], s);
        }
        let p = 2.0_f64.powi(-(s as i32));
        for j in 0..n.saturating_sub(1) {
            r[(j, j + 1)] =
                fractional_power_superdiag_entry(t0[(j, j)], t0[(j + 1, j + 1)], t0[(j, j + 1)], p);
        }
    }
    Ok((r, s, m.unwrap_or(7)))
}

/// SciPy `_logm_triu`: the logarithm of an upper-triangular matrix by inverse scaling and
/// squaring (Al-Mohy & Higham 2012).
fn logm_triu(t0: &DMatrix<C64>) -> Result<DMatrix<C64>, LinalgError> {
    const THETA_LOG: [f64; 17] = [
        0.0, 1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2, 1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1, 4.39e-1,
        5.03e-1, 5.60e-1, 6.09e-1, 6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1,
    ];
    let n = t0.nrows();
    let (r, s, m) = inverse_squaring_helper(t0, &THETA_LOG)?;
    let (nodes, weights) = gauss_legendre(m);
    let ident = identity(n);
    let mut u = DMatrix::from_element(n, n, CZERO);
    for (&w, &x) in weights.iter().zip(nodes.iter()) {
        let alpha = 0.5 * w;
        let beta = 0.5 + 0.5 * x;
        // solve_triangular(I + beta·R, alpha·R): upper-triangular solve per column.
        let lhs = &ident + &r * Complex::new(beta, 0.0);
        for c in 0..n {
            let rhs: Vec<C64> = (0..n).map(|i| r[(i, c)] * alpha).collect();
            let x = triangular_solve(&lhs, &rhs, true, false);
            for i in 0..n {
                u[(i, c)] += x[i];
            }
        }
    }
    u *= Complex::new(2.0_f64.powi(s as i32), 0.0);
    if has_principal_branch(t0) {
        for i in 0..n {
            u[(i, i)] = t0[(i, i)].ln();
        }
        for i in 0..n.saturating_sub(1) {
            u[(i, i + 1)] = logm_superdiag_entry(t0[(i, i)], t0[(i + 1, i + 1)], t0[(i, i + 1)]);
        }
    }
    Ok(u)
}

/// SciPy `_logm_force_nonsingular_triangular_matrix`: an exactly zero diagonal entry becomes
/// `1e-20` (SciPy warns `LogmExactlySingularWarning`).
fn force_nonsingular(t: &mut DMatrix<C64>) {
    for i in 0..t.nrows() {
        if t[(i, i)] == CZERO {
            t[(i, i)] = Complex::new(1e-20, 0.0);
        }
    }
}

/// `scipy.linalg.logm` on complex input: the principal logarithm by SciPy's algorithm — the
/// complex Schur form (skipped for an upper-triangular input), then inverse scaling and
/// squaring on the triangular factor.
///
/// An exactly singular matrix has no logarithm; as SciPy does, its zero eigenvalues are moved to
/// `1e-20` and a (large) result is returned.
///
/// # Errors
/// Shape and finiteness errors and Schur non-convergence.
pub fn logm(a: &[Vec<C64>], options: DecompOptions) -> Result<ComplexMatrix, LinalgError> {
    let n = square(a, options)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let am = to_dm(a, n, n);
    if is_upper_triangular(&am) {
        let mut t = am;
        force_nonsingular(&mut t);
        return Ok(rows_of(&logm_triu(&t)?));
    }
    let (mut t, z) = schur_dm(&am)?;
    force_nonsingular(&mut t);
    let u = logm_triu(&t)?;
    Ok(rows_of(&similarity(&z, &u)))
}

/// The complex triangular Schur pair of a REAL matrix the way SciPy's real-input `logm` and
/// `sqrtm` build it: the real Schur form, converted with `rsf2csf` when it has 2×2 blocks.
fn real_schur_to_complex(
    a: &[Vec<f64>],
    options: DecompOptions,
) -> Result<(DMatrix<C64>, DMatrix<C64>), LinalgError> {
    let n = a.len();
    let real = crate::schur(a, options)?;
    let triangular = (0..n).all(|j| (j + 1..n).all(|i| real.t[i][j] == 0.0));
    if triangular {
        let t = DMatrix::from_fn(n, n, |i, j| Complex::new(real.t[i][j], 0.0));
        let z = DMatrix::from_fn(n, n, |i, j| Complex::new(real.z[i][j], 0.0));
        return Ok((t, z));
    }
    let (tc, zc) = crate::rsf2csf(&real.t, &real.z)?;
    let t = DMatrix::from_fn(n, n, |i, j| Complex::new(tc[i][j].0, tc[i][j].1));
    let z = DMatrix::from_fn(n, n, |i, j| Complex::new(zc[i][j].0, zc[i][j].1));
    Ok((t, z))
}

/// SciPy `_maybe_real` for a real input: the real part when every imaginary part is within
/// `1e6·eps` of zero, the complex matrix otherwise.
fn maybe_real(m: ComplexMatrix) -> MaybeComplexMatrix {
    let tol = EPS * 1e6;
    if m.iter().flatten().all(|z| z.im.abs() <= tol) {
        MaybeComplexMatrix::Real(m.iter().map(|r| r.iter().map(|z| z.re).collect()).collect())
    } else {
        MaybeComplexMatrix::Complex(m)
    }
}

fn real_square(a: &[Vec<f64>], options: DecompOptions) -> Result<usize, LinalgError> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) {
        return if a.iter().any(|r| r.len() != a[0].len()) {
            Err(LinalgError::RaggedMatrix)
        } else {
            Err(LinalgError::ExpectedSquareMatrix)
        };
    }
    hardened_dimension_check(options.mode, n, n)?;
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && a.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(LinalgError::NonFiniteInput);
    }
    Ok(n)
}

/// `scipy.linalg.logm` on REAL input, with SciPy's result type: real when the logarithm is
/// real to `1e6·eps`, complex otherwise — e.g. `logm(diag(-1, 1)) = diag(iπ, 0)`, where the real
/// [`crate::logm`] cannot represent the answer.
///
/// # Errors
/// Shape and finiteness errors and Schur non-convergence.
pub fn logm_real(
    a: &[Vec<f64>],
    options: DecompOptions,
) -> Result<MaybeComplexMatrix, LinalgError> {
    let n = real_square(a, options)?;
    if n == 0 {
        return Ok(MaybeComplexMatrix::Real(Vec::new()));
    }
    let upper = (0..n).all(|j| (j + 1..n).all(|i| a[i][j] == 0.0));
    let u = if upper {
        let mut t = DMatrix::from_fn(n, n, |i, j| Complex::new(a[i][j], 0.0));
        force_nonsingular(&mut t);
        logm_triu(&t)?
    } else {
        let (mut t, z) = real_schur_to_complex(a, options)?;
        force_nonsingular(&mut t);
        similarity(&z, &logm_triu(&t)?)
    };
    Ok(maybe_real(rows_of(&u)))
}

/// `scipy.linalg.sqrtm` on REAL input, with SciPy's result type: complex exactly when the real
/// Schur form has a negative real eigenvalue (`sqrtm(diag(-4, 1)) = diag(2i, 1)`), real
/// otherwise. A singular block without a primary square root gets SciPy's non-finite entries
/// (see [`sqrtm`]); an upper-triangular input skips the Schur form as SciPy's does.
///
/// # Errors
/// Shape and finiteness errors and Schur non-convergence.
pub fn sqrtm_real(
    a: &[Vec<f64>],
    options: DecompOptions,
) -> Result<MaybeComplexMatrix, LinalgError> {
    let n = real_square(a, options)?;
    if n == 0 {
        return Ok(MaybeComplexMatrix::Real(Vec::new()));
    }
    let upper = (0..n).all(|j| (j + 1..n).all(|i| a[i][j] == 0.0));
    let (t, z) = if upper {
        (
            DMatrix::from_fn(n, n, |i, j| Complex::new(a[i][j], 0.0)),
            None,
        )
    } else {
        let (t, z) = real_schur_to_complex(a, options)?;
        (t, Some(z))
    };
    let negative_real = (0..n).any(|i| t[(i, i)].im == 0.0 && t[(i, i)].re < 0.0);
    let r = sqrtm_triu(&t);
    let x = rows_of(&match z {
        Some(z) => similarity(&z, &r),
        None => r,
    });
    Ok(if negative_real {
        MaybeComplexMatrix::Complex(x)
    } else {
        MaybeComplexMatrix::Real(
            x.iter()
                .map(|row| row.iter().map(|z| z.re).collect())
                .collect(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        Complex::new(re, im)
    }

    fn opts() -> DecompOptions {
        DecompOptions::default()
    }

    fn max_abs_diff(a: &[Vec<C64>], b: &[Vec<C64>]) -> f64 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    fn sample() -> ComplexMatrix {
        vec![
            vec![c(4.0, 1.0), c(-2.0, 0.5), c(1.0, -1.0), c(0.3, 0.0)],
            vec![c(1.0, 2.0), c(3.0, -1.0), c(0.0, 0.5), c(-1.0, 1.0)],
            vec![c(-0.5, 0.0), c(2.0, 2.0), c(5.0, 0.0), c(1.5, -0.5)],
            vec![c(0.2, -0.3), c(-1.0, 0.0), c(0.7, 0.7), c(2.0, 3.0)],
        ]
    }

    #[test]
    fn solve_matches_scipy_doc_example() {
        // scipy: solve([[1+1j, 2], [3, 4-1j]], [1, 1j]) = [-1.3-0.9j, 0.7+1.1j]
        let a = vec![
            vec![c(1.0, 1.0), c(2.0, 0.0)],
            vec![c(3.0, 0.0), c(4.0, -1.0)],
        ];
        let r = solve(&a, &[c(1.0, 0.0), c(0.0, 1.0)], SolveOptions::default()).unwrap();
        assert!((r.x[0] - c(-1.3, -0.9)).norm() < 1e-14, "{:?}", r.x);
        assert!((r.x[1] - c(0.7, 1.1)).norm() < 1e-14, "{:?}", r.x);
        assert!(r.warning.is_none());
    }

    #[test]
    fn solve_transposed_and_positive_definite() {
        let a = sample();
        let b = vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, -1.0), c(-1.0, 0.5)];
        // SciPy raises NotImplementedError for transposed complex solves.
        let opts_t = SolveOptions {
            transposed: true,
            ..SolveOptions::default()
        };
        assert!(matches!(
            solve(&a, &b, opts_t),
            Err(LinalgError::NotSupported { .. })
        ));
        // Hermitian positive definite: A = Mᴴ M + I
        let mh = conj_transpose(&a);
        let mut hpd = matmul(&mh, &a).unwrap();
        for (i, row) in hpd.iter_mut().enumerate() {
            row[i] += CONE;
        }
        let pd = SolveOptions {
            assume_a: Some(MatrixAssumption::PositiveDefinite),
            ..SolveOptions::default()
        };
        let x_pd = solve(&hpd, &b, pd).unwrap().x;
        let x_gen = solve(&hpd, &b, SolveOptions::default()).unwrap().x;
        for (p, g) in x_pd.iter().zip(&x_gen) {
            assert!((p - g).norm() < 1e-12);
        }
    }

    #[test]
    fn solve_singular_and_shape_errors() {
        let a = vec![
            vec![c(1.0, 1.0), c(2.0, 2.0)],
            vec![c(1.0, 1.0), c(2.0, 2.0)],
        ];
        assert_eq!(
            solve(&a, &[CONE, CONE], SolveOptions::default()).unwrap_err(),
            LinalgError::SingularMatrix
        );
        assert!(matches!(
            solve(&a, &[CONE], SolveOptions::default()),
            Err(LinalgError::IncompatibleShapes { .. })
        ));
        let nan = vec![vec![c(f64::NAN, 0.0)]];
        assert_eq!(
            solve(&nan, &[CONE], SolveOptions::default()).unwrap_err(),
            LinalgError::NonFiniteInput
        );
        let empty: ComplexMatrix = Vec::new();
        assert!(
            solve(&empty, &[], SolveOptions::default())
                .unwrap()
                .x
                .is_empty()
        );
    }

    #[test]
    fn ill_conditioned_solve_warns() {
        let a = vec![
            vec![c(1.0, 0.0), c(1.0, 0.0)],
            vec![c(1.0, 0.0), c(1.0 + 1e-17, 1e-17)],
        ];
        // 1e-17 rounds away in (1 + 1e-17), leaving an exactly singular real part; the
        // imaginary 1e-17 keeps the pivot nonzero.
        let r = solve(&a, &[CONE, CZERO], SolveOptions::default()).unwrap();
        assert!(r.warning.is_some(), "rcond {}", r.rcond);
    }

    #[test]
    fn lu_reconstructs_and_matches_lu_factor() {
        let a = sample();
        let r = lu(&a, opts()).unwrap();
        let p = to_complex(&r.p);
        let plu = matmul(&matmul(&p, &r.l).unwrap(), &r.u).unwrap();
        assert!(max_abs_diff(&plu, &a) < 1e-13);
        let f = lu_factor(&a, opts()).unwrap();
        let b = vec![c(1.0, 2.0), c(0.0, 0.0), c(-1.0, 0.5), c(3.0, 0.0)];
        for trans in 0..=2u8 {
            let x = lu_solve(&f, &b, trans).unwrap();
            for i in 0..4 {
                let s: C64 = (0..4)
                    .map(|k| match trans {
                        0 => a[i][k] * x[k],
                        1 => a[k][i] * x[k],
                        _ => a[k][i].conj() * x[k],
                    })
                    .sum();
                assert!((s - b[i]).norm() < 1e-12, "trans {trans}");
            }
        }
        assert!(lu_solve(&f, &b, 3).is_err());
        // Rectangular LU.
        let wide: ComplexMatrix = a.iter().take(3).cloned().collect();
        let r = lu(&wide, opts()).unwrap();
        let p = to_complex(&r.p);
        let plu = matmul(&matmul(&p, &r.l).unwrap(), &r.u).unwrap();
        assert!(max_abs_diff(&plu, &wide) < 1e-13);
    }

    #[test]
    fn det_and_inv() {
        let a = vec![
            vec![c(1.0, 1.0), c(2.0, 0.0)],
            vec![c(3.0, 0.0), c(4.0, -1.0)],
        ];
        // (1+i)(4-i) - 6 = 4 - i + 4i + 1 - 6 = -1 + 3i
        let d = det(&a, opts()).unwrap();
        assert!((d - c(-1.0, 3.0)).norm() < 1e-14);
        let ai = inv(&a, opts()).unwrap();
        let prod = matmul(&a, &ai).unwrap();
        assert!(max_abs_diff(&prod, &rows_of(&identity(2))) < 1e-14);
        let sing = vec![vec![CONE, CONE], vec![CONE, CONE]];
        assert_eq!(inv(&sing, opts()).unwrap_err(), LinalgError::SingularMatrix);
        assert_eq!(det(&sing, opts()).unwrap(), CZERO);
    }

    #[test]
    fn cholesky_upper_and_lower() {
        let a = sample();
        let mut hpd = matmul(&conj_transpose(&a), &a).unwrap();
        for (i, row) in hpd.iter_mut().enumerate() {
            row[i] += CONE;
        }
        let u = cholesky(&hpd, false, opts()).unwrap();
        let back = matmul(&conj_transpose(&u), &u).unwrap();
        assert!(max_abs_diff(&back, &hpd) < 1e-12);
        let l = cholesky(&hpd, true, opts()).unwrap();
        let back = matmul(&l, &conj_transpose(&l)).unwrap();
        assert!(max_abs_diff(&back, &hpd) < 1e-12);
        let not_pd = vec![
            vec![c(1.0, 0.0), c(2.0, 0.0)],
            vec![c(2.0, 0.0), c(1.0, 0.0)],
        ];
        let err = cholesky(&not_pd, false, opts()).unwrap_err();
        assert!(
            matches!(err, LinalgError::InvalidArgument { ref detail } if detail.starts_with("2-th"))
        );
    }

    #[test]
    fn qr_full_and_economic() {
        let a: ComplexMatrix = sample()
            .into_iter()
            .take(4)
            .map(|r| r[..3].to_vec())
            .collect();
        for economic in [false, true] {
            let r = qr(&a, economic, opts()).unwrap();
            let back = matmul(&r.q, &r.r).unwrap();
            assert!(max_abs_diff(&back, &a) < 1e-13);
            let qhq = matmul(&conj_transpose(&r.q), &r.q).unwrap();
            assert!(max_abs_diff(&qhq, &rows_of(&identity(r.q[0].len()))) < 1e-13);
            for i in 0..r.r.len().min(3) {
                assert_eq!(r.r[i][i].im, 0.0, "R diagonal is real as in zgeqrf");
            }
        }
        let wide = conj_transpose(&a);
        let r = qr(&wide, false, opts()).unwrap();
        assert!(max_abs_diff(&matmul(&r.q, &r.r).unwrap(), &wide) < 1e-13);
    }

    #[test]
    fn schur_and_eig_reconstruct() {
        let a = sample();
        let s = schur(&a, opts()).unwrap();
        let back = matmul(&matmul(&s.z, &s.t).unwrap(), &conj_transpose(&s.z)).unwrap();
        assert!(max_abs_diff(&back, &a) < 1e-12);
        for j in 0..4 {
            for i in j + 1..4 {
                assert_eq!(s.t[i][j], CZERO);
            }
        }
        let e = eig(&a, opts()).unwrap();
        for k in 0..4 {
            let v: Vec<C64> = (0..4).map(|r| e.eigenvectors[r][k]).collect();
            let nrm: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-14);
            for i in 0..4 {
                let av: C64 = (0..4).map(|j| a[i][j] * v[j]).sum();
                assert!((av - e.eigenvalues[k] * v[i]).norm() < 1e-12);
            }
            let big = v.iter().map(|z| z.norm()).fold(0.0, f64::max);
            let kmax = v.iter().position(|z| z.norm() == big).unwrap();
            assert_eq!(v[kmax].im, 0.0);
        }
        let w = eigvals(&a, opts()).unwrap();
        for (x, y) in w.iter().zip(&e.eigenvalues) {
            assert!((x - y).norm() < 1e-12);
        }
    }

    #[test]
    fn eig_of_rotation_and_triangular() {
        // Real rotation: eigenvalues ±i.
        let a = to_complex(&[vec![0.0, -1.0], vec![1.0, 0.0]]);
        let w = eigvals(&a, opts()).unwrap();
        let mut ims: Vec<f64> = w.iter().map(|z| z.im).collect();
        ims.sort_by(f64::total_cmp);
        assert!((ims[0] + 1.0).abs() < 1e-15 && (ims[1] - 1.0).abs() < 1e-15);
        // Triangular input: balancing isolates everything.
        let t = vec![
            vec![c(1.0, 1.0), c(2.0, 0.0), c(3.0, 0.0)],
            vec![CZERO, c(2.0, -1.0), c(1.0, 1.0)],
            vec![CZERO, CZERO, c(-1.0, 0.0)],
        ];
        let e = eig(&t, opts()).unwrap();
        let mut got: Vec<C64> = e.eigenvalues.clone();
        got.sort_by(|x, y| x.re.total_cmp(&y.re));
        assert_eq!(got, vec![c(-1.0, 0.0), c(1.0, 1.0), c(2.0, -1.0)]);
    }

    #[test]
    fn eigh_is_ascending_and_orthonormal() {
        let a = sample();
        let mut h = matmul(&conj_transpose(&a), &a).unwrap();
        h[0][1] += c(0.0, 1.0);
        h[1][0] -= c(0.0, 1.0);
        let r = eigh(&h, opts()).unwrap();
        assert!(r.eigenvalues.windows(2).all(|w| w[0] <= w[1]));
        let v = &r.eigenvectors;
        let vhv = matmul(&conj_transpose(v), v).unwrap();
        assert!(max_abs_diff(&vhv, &rows_of(&identity(4))) < 1e-12);
        for k in 0..4 {
            for i in 0..4 {
                let hv: C64 = (0..4).map(|j| h[i][j] * v[j][k]).sum();
                assert!((hv - v[i][k] * r.eigenvalues[k]).norm() < 1e-11);
            }
        }
        assert_eq!(eigvalsh(&h, opts()).unwrap(), r.eigenvalues);
    }

    #[test]
    fn svd_full_and_thin() {
        let a: ComplexMatrix = sample().into_iter().map(|r| r[..3].to_vec()).collect();
        let r = svd(&a, true, opts()).unwrap();
        assert_eq!(r.u.len(), 4);
        assert_eq!(r.u[0].len(), 4);
        assert_eq!(r.vh.len(), 3);
        assert!(r.s.windows(2).all(|w| w[0] >= w[1]));
        let uhu = matmul(&conj_transpose(&r.u), &r.u).unwrap();
        assert!(max_abs_diff(&uhu, &rows_of(&identity(4))) < 1e-12);
        let mut sig = vec![vec![CZERO; 3]; 4];
        for i in 0..3 {
            sig[i][i] = c(r.s[i], 0.0);
        }
        let back = matmul(&matmul(&r.u, &sig).unwrap(), &r.vh).unwrap();
        assert!(max_abs_diff(&back, &a) < 1e-12);
        let thin = svd(&a, false, opts()).unwrap();
        assert_eq!(thin.u[0].len(), 3);
        let vals = svdvals(&a, opts()).unwrap();
        for (x, y) in vals.iter().zip(&r.s) {
            assert!((x - y).abs() < 1e-12);
        }
        assert_eq!(
            svd(
                &[vec![c(f64::NAN, 0.0)]],
                true,
                DecompOptions {
                    check_finite: false,
                    ..opts()
                }
            )
            .unwrap_err(),
            LinalgError::NonFiniteInput
        );
    }

    #[test]
    fn lstsq_and_pinv() {
        let a: ComplexMatrix = sample().into_iter().map(|r| r[..2].to_vec()).collect();
        let b = vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0), c(-1.0, -1.0)];
        let r = lstsq(&a, &b, None, opts()).unwrap();
        assert_eq!(r.rank, 2);
        // Normal equations Aᴴ(Ax − b) = 0.
        for j in 0..2 {
            let g: C64 = (0..4)
                .map(|i| {
                    let ax: C64 = (0..2).map(|k| a[i][k] * r.x[k]).sum();
                    a[i][j].conj() * (ax - b[i])
                })
                .sum();
            assert!(g.norm() < 1e-12);
        }
        assert!(r.residues.unwrap() > 0.0);
        let p = pinv(&a, None, None, opts()).unwrap();
        let x_pinv: Vec<C64> = (0..2)
            .map(|i| (0..4).map(|k| p[i][k] * b[k]).sum())
            .collect();
        for (x, y) in x_pinv.iter().zip(&r.x) {
            assert!((x - y).norm() < 1e-12);
        }
        assert_eq!(
            pinv(&a, Some(-1.0), None, opts()).unwrap_err(),
            LinalgError::InvalidPinvThreshold
        );
    }

    #[test]
    fn expm_of_rotation_generator() {
        // expm([[0, -θ], [θ, 0]]·i) etc.: check expm(iθ·I) = e^{iθ}·I and a 2×2 rotation.
        let theta = 0.7;
        let a = to_complex(&[vec![0.0, -theta], vec![theta, 0.0]]);
        let e = expm(&a, opts()).unwrap();
        assert!((e[0][0] - c(theta.cos(), 0.0)).norm() < 1e-15);
        assert!((e[1][0] - c(theta.sin(), 0.0)).norm() < 1e-15);
        let big = vec![vec![c(0.0, 10.0), CZERO], vec![CZERO, c(1.0, 3.0)]];
        let e = expm(&big, opts()).unwrap();
        assert!((e[0][0] - c(0.0, 10.0).exp()).norm() < 1e-13);
        assert!((e[1][1] - c(1.0, 3.0).exp()).norm() < 1e-13);
        // expm(logm(A)) = A.
        let a = sample();
        let l = logm(&a, opts()).unwrap();
        let back = expm(&l, opts()).unwrap();
        assert!(max_abs_diff(&back, &a) < 1e-11);
    }

    #[test]
    fn sqrtm_squares_back() {
        let a = sample();
        let r = sqrtm(&a, opts()).unwrap();
        let back = matmul(&r, &r).unwrap();
        assert!(max_abs_diff(&back, &a) < 1e-12);
        // SciPy: sqrtm([[0, 1], [0, 0]]) = [[0, inf+nanj], [0, 0]].
        let nilpotent = to_complex(&[vec![0.0, 1.0], vec![0.0, 0.0]]);
        let r = sqrtm(&nilpotent, opts()).unwrap();
        assert_eq!(r[0][1].re, f64::INFINITY);
        assert!(r[0][1].im.is_nan());
        assert_eq!((r[0][0], r[1][0], r[1][1]), (CZERO, CZERO, CZERO));
        let MaybeComplexMatrix::Real(rr) =
            sqrtm_real(&[vec![0.0, 1.0], vec![0.0, 0.0]], opts()).unwrap()
        else {
            panic!("expected a real result")
        };
        assert_eq!(rr, vec![vec![0.0, f64::INFINITY], vec![0.0, 0.0]]);
    }

    #[test]
    fn real_input_logm_and_sqrtm_go_complex_only_when_needed() {
        // logm(diag(-1, 1)) = diag(iπ, 0)
        let l = logm_real(&[vec![-1.0, 0.0], vec![0.0, 1.0]], opts()).unwrap();
        let MaybeComplexMatrix::Complex(m) = l else {
            panic!("expected complex")
        };
        assert!((m[0][0] - c(0.0, std::f64::consts::PI)).norm() < 1e-15);
        assert!(m[1][1].norm() < 1e-15);
        // sqrtm(diag(-4, 1)) = diag(2i, 1)
        let s = sqrtm_real(&[vec![-4.0, 0.0], vec![0.0, 1.0]], opts()).unwrap();
        let MaybeComplexMatrix::Complex(m) = s else {
            panic!("expected complex")
        };
        assert!((m[0][0] - c(0.0, 2.0)).norm() < 1e-15);
        // A positive-definite real matrix stays real.
        let spd = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        assert!(!logm_real(&spd, opts()).unwrap().is_complex());
        assert!(!sqrtm_real(&spd, opts()).unwrap().is_complex());
        // A rotation (complex eigenvalues, no negative real one) has a real logarithm.
        let rot = vec![
            vec![0.6_f64.cos(), -0.6_f64.sin()],
            vec![0.6_f64.sin(), 0.6_f64.cos()],
        ];
        let MaybeComplexMatrix::Real(lr) = logm_real(&rot, opts()).unwrap() else {
            panic!("expected real")
        };
        assert!((lr[1][0] - 0.6).abs() < 1e-14 && lr[0][0].abs() < 1e-14);
    }

    #[test]
    fn gauss_legendre_matches_known_rules() {
        let (x, w) = gauss_legendre(2);
        assert!((x[0] + 1.0 / 3.0_f64.sqrt()).abs() < 1e-15);
        assert!((w[0] - 1.0).abs() < 1e-15);
        let (x, w) = gauss_legendre(5);
        assert!(x.windows(2).all(|p| p[0] < p[1]));
        assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
        // ∫ x^8 over [-1, 1] = 2/9, exact for 5 nodes.
        let q: f64 = x.iter().zip(&w).map(|(xi, wi)| wi * xi.powi(8)).sum();
        assert!((q - 2.0 / 9.0).abs() < 1e-15);
    }

    #[test]
    fn hessenberg_reconstructs() {
        let a = sample();
        let h = hessenberg(&a, opts()).unwrap();
        let back = matmul(&matmul(&h.q, &h.h).unwrap(), &conj_transpose(&h.q)).unwrap();
        assert!(max_abs_diff(&back, &a) < 1e-12);
        for j in 0..4 {
            for i in j + 2..4 {
                assert_eq!(h.h[i][j], CZERO);
            }
        }
    }

    #[test]
    fn norms() {
        let a = vec![vec![c(3.0, 4.0), CZERO], vec![CZERO, c(0.0, 1.0)]];
        assert!((norm(&a, NormKind::Fro, opts()).unwrap() - 26.0_f64.sqrt()).abs() < 1e-15);
        assert!((norm(&a, NormKind::One, opts()).unwrap() - 5.0).abs() < 1e-15);
        assert!((norm(&a, NormKind::Inf, opts()).unwrap() - 5.0).abs() < 1e-15);
        assert!((norm(&a, NormKind::Spectral, opts()).unwrap() - 5.0).abs() < 1e-14);
    }
}
