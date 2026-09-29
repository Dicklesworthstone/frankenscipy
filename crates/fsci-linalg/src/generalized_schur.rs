//! Real generalized Schur (QZ) decomposition and its reordering -- frankenscipy-szq1n.5.
//!
//! `qz` used to form `A·B⁻¹` explicitly and take its real Schur form. That fails outright for
//! a singular `B` (infinite generalized eigenvalues, the routine case in descriptor systems and
//! the main reason to call QZ instead of `eig(solve(B, A))`) and loses accuracy in proportion
//! to κ(B). `ordqz` then "reordered" by permuting rows and columns, which does not preserve the
//! (quasi-)triangular form. This module is the Moler–Stewart algorithm LAPACK uses.
//!
//! * [`real_qz`]: Hessenberg–triangular reduction (Golub & Van Loan Alg. 7.7.1), then Francis
//!   double-shift QZ sweeps (Alg. 7.7.2) with LAPACK `dhgeqz`'s deflation: a negligible
//!   subdiagonal of `H` splits the pencil; a negligible diagonal entry of `T` is either split
//!   off the top of the active block or chased to the bottom, where it deflates an infinite
//!   eigenvalue; an exceptional shift every tenth sweep without a deflation. 2×2 blocks with
//!   real eigenvalues are split, and those with a complex pair are standardized so their `T`
//!   block is diagonal with positive entries (LAPACK's convention), as is every 1×1 `T` entry.
//! * [`reorder_selected_first`]: `dtgsen`'s strategy -- the selection is decided ONCE on the
//!   QZ output, then each selected block moves up past the unselected ones by adjacent block
//!   swaps. A swap computes a real basis of the moving block's deflating subspace from its
//!   eigenvector (real and imaginary parts for a complex pair), builds `Z` from it and `Q` from
//!   the QR of `T·Z`, and is accepted only if the decoupled entries are negligible.
//!
//! Convention throughout: `Qᵀ·A·Z = S` (quasi-upper-triangular), `Qᵀ·B·Z = T` (upper
//! triangular), with `Q` and `Z` orthogonal.

type Mat = Vec<Vec<f64>>;

fn identity(n: usize) -> Mat {
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// LAPACK `dlartg`: `(c, s, r)` with `[c s; -s c]·[f; g] = [r; 0]`.
fn givens(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, 1.0, g);
    }
    let r = f.hypot(g);
    let r = if f < 0.0 { -r } else { r };
    (f / r, g / r, r)
}

/// Rows `(i, j)` ← `[c s; -s c]·[row_i; row_j]` over `cols`.
fn rot_rows(m: &mut Mat, i: usize, j: usize, c: f64, s: f64, cols: std::ops::Range<usize>) {
    for k in cols {
        let (x, y) = (m[i][k], m[j][k]);
        m[i][k] = c * x + s * y;
        m[j][k] = c * y - s * x;
    }
}

/// Columns `(i, j)` ← `(c·col_i + s·col_j, c·col_j − s·col_i)` over `rows`.
fn rot_cols(m: &mut Mat, i: usize, j: usize, c: f64, s: f64, rows: std::ops::Range<usize>) {
    for k in rows {
        let (x, y) = (m[k][i], m[k][j]);
        m[k][i] = c * x + s * y;
        m[k][j] = c * y - s * x;
    }
}

/// Householder vector `v` with `(I − 2vvᵀ/vᵀv)·x = α·e₁`; `None` when `x` is already there.
fn householder_first(x: &[f64]) -> Option<Vec<f64>> {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 || x[1..].iter().all(|&v| v == 0.0) {
        return None;
    }
    let alpha = if x[0] > 0.0 { -norm } else { norm };
    let mut v = x.to_vec();
    v[0] -= alpha;
    Some(v)
}

/// Householder vector `v` with `(I − 2vvᵀ/vᵀv)·x = α·e_last`.
fn householder_last(x: &[f64]) -> Option<Vec<f64>> {
    let last = x.len() - 1;
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 || x[..last].iter().all(|&v| v == 0.0) {
        return None;
    }
    let alpha = if x[last] > 0.0 { -norm } else { norm };
    let mut v = x.to_vec();
    v[last] -= alpha;
    Some(v)
}

/// `m[r0..r0+len, cols]` ← `P·m[r0..r0+len, cols]` for the reflector `P` of `v`.
fn reflect_rows(m: &mut Mat, r0: usize, v: &[f64], cols: std::ops::Range<usize>) {
    let vtv: f64 = v.iter().map(|x| x * x).sum();
    for k in cols {
        let dot: f64 = (0..v.len()).map(|i| v[i] * m[r0 + i][k]).sum();
        let f = 2.0 * dot / vtv;
        for i in 0..v.len() {
            m[r0 + i][k] -= f * v[i];
        }
    }
}

/// `m[rows, c0..c0+len]` ← `m[rows, c0..c0+len]·P` for the reflector `P` of `v`.
fn reflect_cols(m: &mut Mat, c0: usize, v: &[f64], rows: std::ops::Range<usize>) {
    let vtv: f64 = v.iter().map(|x| x * x).sum();
    for k in rows {
        let dot: f64 = (0..v.len()).map(|i| v[i] * m[k][c0 + i]).sum();
        let f = 2.0 * dot / vtv;
        for i in 0..v.len() {
            m[k][c0 + i] -= f * v[i];
        }
    }
}

/// Hessenberg–triangular reduction: `(H, T, Q, Z)` with `Qᵀ·A·Z = H` upper Hessenberg and
/// `Qᵀ·B·Z = T` upper triangular.
fn hessenberg_triangular(a: &[Vec<f64>], b: &[Vec<f64>]) -> (Mat, Mat, Mat, Mat) {
    let n = a.len();
    let (mut h, mut t, mut q, mut z) = (a.to_vec(), b.to_vec(), identity(n), identity(n));
    // QR of B by Householder reflections.
    for k in 0..n.saturating_sub(1) {
        let x: Vec<f64> = (k..n).map(|i| t[i][k]).collect();
        if let Some(v) = householder_first(&x) {
            reflect_rows(&mut t, k, &v, k..n);
            reflect_rows(&mut h, k, &v, 0..n);
            reflect_cols(&mut q, k, &v, 0..n);
        }
        for row in t.iter_mut().skip(k + 1) {
            row[k] = 0.0;
        }
    }
    // Reduce H to Hessenberg form by Givens rotations, restoring T's triangularity each time.
    for j in 0..n.saturating_sub(2) {
        for i in (j + 2..n).rev() {
            let (c, s, r) = givens(h[i - 1][j], h[i][j]);
            rot_rows(&mut h, i - 1, i, c, s, j..n);
            h[i - 1][j] = r;
            h[i][j] = 0.0;
            rot_rows(&mut t, i - 1, i, c, s, i - 1..n);
            rot_cols(&mut q, i - 1, i, c, s, 0..n);
            let (c, s, r) = givens(t[i][i], t[i][i - 1]);
            rot_cols(&mut t, i, i - 1, c, s, 0..i);
            t[i][i] = r;
            t[i][i - 1] = 0.0;
            rot_cols(&mut h, i, i - 1, c, s, 0..n);
            rot_cols(&mut z, i, i - 1, c, s, 0..n);
        }
    }
    (h, t, q, z)
}

/// Inverse of the 3×3 upper-triangular block `T[l-2..=l, l-2..=l]`.
fn inverse_trailing_3x3(t: &Mat, l: usize) -> [[f64; 3]; 3] {
    let a = |i: usize, j: usize| t[l - 2 + i][l - 2 + j];
    let mut inv = [[0.0; 3]; 3];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0 / a(i, i);
    }
    inv[0][1] = -a(0, 1) * inv[1][1] / a(0, 0);
    inv[1][2] = -a(1, 2) * inv[2][2] / a(1, 1);
    inv[0][2] = -(a(0, 1) * inv[1][2] + a(0, 2) * inv[2][2]) / a(0, 0);
    inv
}

/// Mutable views of the four factors, so the sweeps take one argument for them.
struct Factors<'m> {
    h: &'m mut Mat,
    t: &'m mut Mat,
    q: &'m mut Mat,
    z: &'m mut Mat,
}

/// One Francis double-shift QZ sweep on the active block `[f, l]` (`l − f ≥ 2`, `T`'s diagonal
/// nonzero there).
fn double_shift_sweep(fx: &mut Factors<'_>, f: usize, l: usize, exceptional: bool) {
    let (h, t, q, z) = (&mut *fx.h, &mut *fx.t, &mut *fx.q, &mut *fx.z);
    let n = h.len();
    // Shifts: the trailing 2×2 of M = H·T⁻¹, as trace and determinant.
    let tinv = inverse_trailing_3x3(t, l);
    let mut m2 = [[0.0; 2]; 2];
    for (r, &row) in [l - 1, l].iter().enumerate() {
        for (c, col) in [1usize, 2].into_iter().enumerate() {
            m2[r][c] = (0..3).map(|k| h[row][l - 2 + k] * tinv[k][col]).sum();
        }
    }
    let (mut trace, mut det) = (
        m2[0][0] + m2[1][1],
        m2[0][0] * m2[1][1] - m2[0][1] * m2[1][0],
    );
    if exceptional {
        // Ad hoc shift after repeated non-convergence, as dhgeqz does.
        let w = (h[l][l - 1] / t[l - 1][l - 1]).abs() + (h[l - 1][l - 2] / t[l - 2][l - 2]).abs();
        trace = 1.5 * w;
        det = w * w;
    }
    // First column of (M² − trace·M + det·I)·e₁ from the leading entries of M.
    let m00 = h[f][f] / t[f][f];
    let m10 = h[f + 1][f] / t[f][f];
    let m01 = (h[f][f + 1] - m00 * t[f][f + 1]) / t[f + 1][f + 1];
    let m11 = (h[f + 1][f + 1] - m10 * t[f][f + 1]) / t[f + 1][f + 1];
    let m21 = h[f + 2][f + 1] / t[f + 1][f + 1];
    let mut x = [
        m00 * m00 + m01 * m10 - trace * m00 + det,
        m10 * (m00 + m11 - trace),
        m21 * m10,
    ];
    for k in f..l - 1 {
        if k > f {
            x = [h[k][k - 1], h[k + 1][k - 1], h[k + 2][k - 1]];
        }
        if let Some(v) = householder_first(&x) {
            let c0 = if k > f { k - 1 } else { f };
            reflect_rows(h, k, &v, c0..n);
            reflect_rows(t, k, &v, k..n);
            reflect_cols(q, k, &v, 0..n);
            if k > f {
                h[k + 1][k - 1] = 0.0;
                h[k + 2][k - 1] = 0.0;
            }
        }
        let rmax = (k + 4).min(l + 1);
        // Right reflector on columns k..k+2 zeroing T[k+2][k] and T[k+2][k+1].
        let row: Vec<f64> = (k..k + 3).map(|j| t[k + 2][j]).collect();
        if let Some(v) = householder_last(&row) {
            reflect_cols(h, k, &v, 0..rmax);
            reflect_cols(t, k, &v, 0..k + 3);
            reflect_cols(z, k, &v, 0..n);
            t[k + 2][k] = 0.0;
            t[k + 2][k + 1] = 0.0;
        }
        // Right rotation on columns (k, k+1) zeroing T[k+1][k].
        let (c, s, r) = givens(t[k + 1][k + 1], t[k + 1][k]);
        rot_cols(h, k + 1, k, c, s, 0..rmax);
        rot_cols(t, k + 1, k, c, s, 0..k + 1);
        t[k + 1][k + 1] = r;
        t[k + 1][k] = 0.0;
        rot_cols(z, k + 1, k, c, s, 0..n);
    }
    // Last step: rows (l-1, l) zeroing H[l][l-2], then columns (l-1, l) zeroing T[l][l-1].
    let (c, s, r) = givens(h[l - 1][l - 2], h[l][l - 2]);
    rot_rows(h, l - 1, l, c, s, l - 2..n);
    h[l - 1][l - 2] = r;
    h[l][l - 2] = 0.0;
    rot_rows(t, l - 1, l, c, s, l - 1..n);
    rot_cols(q, l - 1, l, c, s, 0..n);
    let (c, s, r) = givens(t[l][l], t[l][l - 1]);
    rot_cols(h, l, l - 1, c, s, 0..l + 1);
    rot_cols(t, l, l - 1, c, s, 0..l);
    t[l][l] = r;
    t[l][l - 1] = 0.0;
    rot_cols(z, l, l - 1, c, s, 0..n);
}

/// Standardize the 2×2 block at `(j, j+1)`: split it into two 1×1 blocks when its generalized
/// eigenvalues are real (returns `true`), otherwise make its `T` block diagonal.
fn standardize_2x2(fx: &mut Factors<'_>, j: usize) -> bool {
    let (h, t, q, z) = (&mut *fx.h, &mut *fx.t, &mut *fx.q, &mut *fx.z);
    let n = h.len();
    let (a11, a12, a21, a22) = (h[j][j], h[j][j + 1], h[j + 1][j], h[j + 1][j + 1]);
    let (b11, b12, b22) = (t[j][j], t[j][j + 1], t[j + 1][j + 1]);
    // det(A − λB) = (b11·b22)·λ² − (a11·b22 + a22·b11 − a21·b12)·λ + det(A).
    let qa = b11 * b22;
    let qb = -(a11 * b22 + a22 * b11 - a21 * b12);
    let qc = a11 * a22 - a12 * a21;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc >= 0.0 && qa != 0.0 {
        // Real λ: the first column of Z becomes the null vector of A − λB (larger root first
        // for accuracy); a row rotation then restores T's triangularity and H[j+1][j] → 0.
        let sq = disc.sqrt();
        let root = if qb >= 0.0 {
            (-qb - sq) / (2.0 * qa)
        } else {
            (-qb + sq) / (2.0 * qa)
        };
        let lambda = if root.is_finite() {
            root
        } else {
            qc / (qa * root)
        };
        let (m11, m12) = (a11 - lambda * b11, a12 - lambda * b12);
        let (m21, m22) = (a21, a22 - lambda * b22);
        let (u0, u1) = if m11.hypot(m12) >= m21.hypot(m22) {
            (m12, -m11)
        } else {
            (m22, -m21)
        };
        if u0 == 0.0 && u1 == 0.0 {
            return false;
        }
        let (c, s, _) = givens(u0, u1);
        rot_cols(h, j, j + 1, c, s, 0..n);
        rot_cols(t, j, j + 1, c, s, 0..n);
        rot_cols(z, j, j + 1, c, s, 0..n);
        let (c, s, r) = givens(t[j][j], t[j + 1][j]);
        rot_rows(h, j, j + 1, c, s, 0..n);
        rot_rows(t, j, j + 1, c, s, 0..n);
        t[j][j] = r;
        t[j + 1][j] = 0.0;
        rot_cols(q, j, j + 1, c, s, 0..n);
        h[j + 1][j] = 0.0;
        return true;
    }
    // Complex pair: 2×2 SVD of T's block. The column rotation diagonalizes BᵀB (so B·R has
    // orthogonal columns); the row rotation then makes it upper triangular, hence diagonal.
    let (p, pq, r) = (b11 * b11, b11 * b12, b12 * b12 + b22 * b22);
    let theta = 0.5 * (2.0 * pq).atan2(p - r);
    let (c, s) = (theta.cos(), theta.sin());
    rot_cols(h, j, j + 1, c, s, 0..n);
    rot_cols(t, j, j + 1, c, s, 0..n);
    rot_cols(z, j, j + 1, c, s, 0..n);
    let (c, s, r2) = givens(t[j][j], t[j + 1][j]);
    rot_rows(h, j, j + 1, c, s, 0..n);
    rot_rows(t, j, j + 1, c, s, 0..n);
    t[j][j] = r2;
    t[j + 1][j] = 0.0;
    t[j][j + 1] = 0.0;
    rot_cols(q, j, j + 1, c, s, 0..n);
    false
}

/// `T[il][il] = 0`: clear `H[il][il-1]` by a rotation of columns `(il-1, il)`; `T` stays
/// triangular because its `(il, il)` entry is zero.
fn clear_bottom(fx: &mut Factors<'_>, il: usize) {
    let n = fx.h.len();
    let (c, s, r) = givens(fx.h[il][il], fx.h[il][il - 1]);
    fx.h[il][il] = r;
    fx.h[il][il - 1] = 0.0;
    rot_cols(fx.h, il, il - 1, c, s, 0..il);
    rot_cols(fx.t, il, il - 1, c, s, 0..il);
    rot_cols(fx.z, il, il - 1, c, s, 0..n);
}

/// Real generalized Schur decomposition `(S, T, Q, Z)` of the square pencil `(a, b)`.
///
/// # Errors
/// When the QZ iteration has not deflated after `30·max(n, 10)` sweeps.
pub(crate) fn real_qz(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<(Mat, Mat, Mat, Mat), String> {
    let n = a.len();
    let (mut h, mut t, mut q, mut z) = hessenberg_triangular(a, b);
    if n == 0 {
        return Ok((h, t, q, z));
    }
    let ulp = f64::EPSILON;
    let safmin = f64::MIN_POSITIVE;
    let frobenius = |m: &Mat| m.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let atol = safmin.max(ulp * frobenius(&h));
    let btol = safmin.max(ulp * frobenius(&t));
    let mut fx = Factors {
        h: &mut h,
        t: &mut t,
        q: &mut q,
        z: &mut z,
    };
    let negligible = |h: &Mat, j: usize| {
        h[j][j - 1].abs() <= atol.max(ulp * (h[j][j].abs() + h[j - 1][j - 1].abs()))
    };
    let mut ilast = n;
    let mut since_deflation = 0usize;
    let max_sweeps = 30 * n.max(10);
    let mut sweeps = 0usize;
    while ilast > 0 {
        let il = ilast - 1;
        if il == 0 {
            ilast -= 1;
            continue;
        }
        if negligible(fx.h, il) {
            fx.h[il][il - 1] = 0.0;
            ilast -= 1;
            since_deflation = 0;
            continue;
        }
        if fx.t[il][il].abs() <= btol {
            fx.t[il][il] = 0.0;
            clear_bottom(&mut fx, il);
            ilast -= 1;
            since_deflation = 0;
            continue;
        }
        // Find the active block [first, il]; handle zero diagonal entries of T inside it.
        let mut first = None;
        let mut j = il - 1;
        loop {
            let split = j == 0 || negligible(fx.h, j);
            if split && j > 0 {
                fx.h[j][j - 1] = 0.0;
            }
            if fx.t[j][j].abs() < btol {
                fx.t[j][j] = 0.0;
                if split {
                    // Split 1×1 infinite eigenvalues off the top of the block.
                    let mut jch = j;
                    let mut resolved = false;
                    while jch < il {
                        let (c, s, r) = givens(fx.h[jch][jch], fx.h[jch + 1][jch]);
                        fx.h[jch][jch] = r;
                        fx.h[jch + 1][jch] = 0.0;
                        rot_rows(fx.h, jch, jch + 1, c, s, jch + 1..n);
                        rot_rows(fx.t, jch, jch + 1, c, s, jch + 1..n);
                        rot_cols(fx.q, jch, jch + 1, c, s, 0..n);
                        if fx.t[jch + 1][jch + 1].abs() >= btol {
                            if jch + 1 < il {
                                first = Some(jch + 1);
                            }
                            resolved = true;
                            break;
                        }
                        fx.t[jch + 1][jch + 1] = 0.0;
                        jch += 1;
                    }
                    if !resolved {
                        clear_bottom(&mut fx, il);
                    }
                    break;
                }
                // Chase the zero on T's diagonal down to T[il][il], then deflate there.
                for jch in j..il {
                    let (c, s, r) = givens(fx.t[jch][jch + 1], fx.t[jch + 1][jch + 1]);
                    fx.t[jch][jch + 1] = r;
                    fx.t[jch + 1][jch + 1] = 0.0;
                    rot_rows(fx.t, jch, jch + 1, c, s, jch + 2..n);
                    rot_rows(fx.h, jch, jch + 1, c, s, jch.saturating_sub(1)..n);
                    rot_cols(fx.q, jch, jch + 1, c, s, 0..n);
                    if jch > 0 {
                        let (c, s, r) = givens(fx.h[jch + 1][jch], fx.h[jch + 1][jch - 1]);
                        fx.h[jch + 1][jch] = r;
                        fx.h[jch + 1][jch - 1] = 0.0;
                        rot_cols(fx.h, jch, jch - 1, c, s, 0..jch + 1);
                        rot_cols(fx.t, jch, jch - 1, c, s, 0..jch);
                        rot_cols(fx.z, jch, jch - 1, c, s, 0..n);
                    }
                }
                clear_bottom(&mut fx, il);
                break;
            } else if split {
                first = Some(j);
                break;
            }
            j -= 1;
        }
        let Some(f) = first else {
            // A deflation-preparing move was made; re-examine from the bottom.
            continue;
        };
        if il - f == 1 {
            if !standardize_2x2(&mut fx, f) {
                ilast -= 2;
            }
            since_deflation = 0;
            continue;
        }
        sweeps += 1;
        since_deflation += 1;
        if sweeps > max_sweeps {
            return Err(format!(
                "QZ iteration did not converge after {max_sweeps} sweeps"
            ));
        }
        double_shift_sweep(&mut fx, f, il, since_deflation.is_multiple_of(10));
    }
    make_t_diagonal_nonnegative(&mut fx);
    Ok((h, t, q, z))
}

/// LAPACK's convention: every diagonal entry of `T` non-negative (a column of H, T and Z is
/// negated where it is not; for a complex pair `T`'s block is diagonal, so this keeps it so).
fn make_t_diagonal_nonnegative(fx: &mut Factors<'_>) {
    let n = fx.h.len();
    for k in 0..n {
        if fx.t[k][k] < 0.0 {
            for r in 0..n {
                fx.h[r][k] = -fx.h[r][k];
                fx.t[r][k] = -fx.t[r][k];
                fx.z[r][k] = -fx.z[r][k];
            }
        }
    }
}

/// `(start, size)` of each diagonal block of a quasi-triangular `S`.
pub(crate) fn diagonal_blocks(s: &Mat) -> Vec<(usize, usize)> {
    let n = s.len();
    let mut out = Vec::new();
    let mut j = 0;
    while j < n {
        let size = if j + 1 < n && s[j + 1][j] != 0.0 {
            2
        } else {
            1
        };
        out.push((j, size));
        j += size;
    }
    out
}

/// A block's generalized eigenvalue as `(alpha_re, alpha_im, beta)` (the first of a complex
/// pair; `beta` is the block's `T` scale, 0 for an infinite eigenvalue).
pub(crate) fn block_eigenvalue(s: &Mat, t: &Mat, start: usize, size: usize) -> (f64, f64, f64) {
    if size == 1 {
        return (s[start][start], 0.0, t[start][start]);
    }
    // T block is diagonal and positive: eigenvalues of diag(1/t)·S_block.
    let (b1, b2) = (t[start][start], t[start + 1][start + 1]);
    let (m11, m12) = (s[start][start] / b1, s[start][start + 1] / b1);
    let (m21, m22) = (s[start + 1][start] / b2, s[start + 1][start + 1] / b2);
    let half_trace = 0.5 * (m11 + m22);
    let det = m11 * m22 - m12 * m21;
    let im = (det - half_trace * half_trace).max(0.0).sqrt();
    (half_trace, im, 1.0)
}

type Cx = (f64, f64);
fn cmul(a: Cx, b: Cx) -> Cx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: Cx, b: Cx) -> Cx {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Null vector of a rank-deficient m×m complex matrix: Gaussian elimination with full
/// pivoting for m−1 steps, the remaining column taken as the free variable.
fn null_vector(mut a: Vec<Vec<Cx>>) -> Vec<Cx> {
    let m = a.len();
    let mut cols: Vec<usize> = (0..m).collect();
    for step in 0..m - 1 {
        let (mut pr, mut pc, mut best) = (step, step, -1.0);
        for (r, row) in a.iter().enumerate().skip(step) {
            for (ci, &col) in cols.iter().enumerate().skip(step) {
                let v = row[col].0.hypot(row[col].1);
                if v > best {
                    best = v;
                    pr = r;
                    pc = ci;
                }
            }
        }
        a.swap(step, pr);
        cols.swap(step, pc);
        let pivot = a[step][cols[step]];
        if pivot == (0.0, 0.0) {
            continue;
        }
        for r in step + 1..m {
            let factor = cdiv(a[r][cols[step]], pivot);
            for &col in &cols[step..] {
                let t = cmul(factor, a[step][col]);
                a[r][col] = (a[r][col].0 - t.0, a[r][col].1 - t.1);
            }
        }
    }
    let mut x = vec![(0.0, 0.0); m];
    x[cols[m - 1]] = (1.0, 0.0);
    for step in (0..m - 1).rev() {
        let mut acc = (0.0, 0.0);
        for &col in &cols[step + 1..] {
            let t = cmul(a[step][col], x[col]);
            acc = (acc.0 + t.0, acc.1 + t.1);
        }
        let pivot = a[step][cols[step]];
        x[cols[step]] = if pivot == (0.0, 0.0) {
            (0.0, 0.0)
        } else {
            cdiv((-acc.0, -acc.1), pivot)
        };
    }
    x
}

/// Swap the adjacent diagonal blocks `[k, k+p)` and `[k+p, k+p+q)` (`p, q ∈ {1, 2}`) so the
/// second block's eigenvalues come first.
fn swap_adjacent_blocks(fx: &mut Factors<'_>, k: usize, p: usize, q: usize) -> Result<(), String> {
    let n = fx.h.len();
    let m = p + q;
    let block = |x: &Mat| -> Mat { (0..m).map(|i| x[k + i][k..k + m].to_vec()).collect() };
    let (s_sub, t_sub) = (block(fx.h), block(fx.t));
    // Deflating subspace of the second block, from its eigenvector.
    let (alpha_re, alpha_im, beta) = block_eigenvalue(fx.h, fx.t, k + p, q);
    let (alpha, beta) = if beta == 0.0 {
        ((1.0, 0.0), (0.0, 0.0))
    } else {
        ((alpha_re / beta, alpha_im / beta), (1.0, 0.0))
    };
    let pencil: Vec<Vec<Cx>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    let x = cmul(beta, (s_sub[i][j], 0.0));
                    let y = cmul(alpha, (t_sub[i][j], 0.0));
                    (x.0 - y.0, x.1 - y.1)
                })
                .collect()
        })
        .collect();
    let eigenvector = null_vector(pencil);
    // Orthogonal W whose leading q columns span {Re v, Im v}: Householder QR of the basis.
    let mut basis: Mat = (0..m)
        .map(|i| {
            if q == 1 {
                vec![eigenvector[i].0]
            } else {
                vec![eigenvector[i].0, eigenvector[i].1]
            }
        })
        .collect();
    let mut w = identity(m);
    for c in 0..q {
        let x: Vec<f64> = (c..m).map(|i| basis[i][c]).collect();
        if let Some(v) = householder_first(&x) {
            reflect_rows(&mut basis, c, &v, 0..q);
            reflect_cols(&mut w, c, &v, 0..m);
        }
    }
    // V from the QR of T·W, so Vᵀ·T·W is upper triangular.
    let mut tw: Mat = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| (0..m).map(|l| t_sub[i][l] * w[l][j]).sum())
                .collect()
        })
        .collect();
    let mut v_acc = identity(m);
    for c in 0..m.saturating_sub(1) {
        let x: Vec<f64> = (c..m).map(|i| tw[i][c]).collect();
        if let Some(v) = householder_first(&x) {
            reflect_rows(&mut tw, c, &v, 0..m);
            reflect_cols(&mut v_acc, c, &v, 0..m);
        }
    }
    let left = |x: &mut Mat| {
        for col in 0..n {
            let old: Vec<f64> = (0..m).map(|i| x[k + i][col]).collect();
            for i in 0..m {
                x[k + i][col] = (0..m).map(|l| v_acc[l][i] * old[l]).sum();
            }
        }
    };
    let right = |x: &mut Mat, r: &Mat| {
        for row in x.iter_mut() {
            let old: Vec<f64> = row[k..k + m].to_vec();
            for j in 0..m {
                row[k + j] = (0..m).map(|l| old[l] * r[l][j]).sum();
            }
        }
    };
    left(fx.h);
    left(fx.t);
    right(fx.h, &w);
    right(fx.t, &w);
    right(fx.q, &v_acc);
    right(fx.z, &w);
    // Accept the swap only if what it must decouple is negligible (dtgex2 checks the same).
    let scale: f64 = s_sub
        .iter()
        .chain(&t_sub)
        .flatten()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    // `f64::max` dropped a NaN, so a swap that turned the blocks into NaN read as residual 0 and
    // was accepted. dtgex2's stability tests are `.LE.` comparisons, which a NaN fails
    // (INFO = 1, SciPy's "Reordering of (A, B) failed"), so a NaN residual is refused too.
    let nan_max = |acc: f64, v: f64| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.max(v)
        }
    };
    let mut residual: f64 = 0.0;
    for i in 0..m {
        for j in 0..i {
            residual = nan_max(residual, fx.t[k + i][k + j].abs());
            if i >= q && j < q {
                residual = nan_max(residual, fx.h[k + i][k + j].abs());
            }
        }
    }
    if residual.is_nan() || residual > 1e3 * f64::EPSILON * scale.max(f64::MIN_POSITIVE) {
        return Err(format!(
            "ordqz: swapping the blocks at {k} is too ill-conditioned (residual {residual:e})"
        ));
    }
    for i in 0..m {
        for j in 0..i {
            fx.t[k + i][k + j] = 0.0;
            if i >= q && j < q {
                fx.h[k + i][k + j] = 0.0;
            }
        }
    }
    if q == 2 {
        standardize_2x2(fx, k);
    }
    if p == 2 {
        standardize_2x2(fx, k + q);
    }
    Ok(())
}

/// Reorder a real generalized Schur form so every block `select` accepts comes first, in the
/// original relative order (LAPACK `dtgsen`). `select` receives the block's eigenvalue
/// `(alpha_re, alpha_im, beta)` as computed on the QZ output, ONCE per block.
///
/// # Errors
/// When a block swap is too ill-conditioned to perform stably.
pub(crate) fn reorder_selected_first(
    s: &mut Mat,
    t: &mut Mat,
    q: &mut Mat,
    z: &mut Mat,
    select: impl Fn(f64, f64, f64) -> bool,
) -> Result<(), String> {
    let mut order: Vec<(usize, bool)> = diagonal_blocks(s)
        .into_iter()
        .map(|(start, size)| {
            let (re, im, beta) = block_eigenvalue(s, t, start, size);
            (size, select(re, im, beta))
        })
        .collect();
    let mut fx = Factors { h: s, t, q, z };
    let mut next_slot = 0;
    for i in 0..order.len() {
        if !order[i].1 {
            continue;
        }
        let mut pos = i;
        while pos > next_slot {
            let start: usize = order[..pos - 1].iter().map(|b| b.0).sum();
            swap_adjacent_blocks(&mut fx, start, order[pos - 1].0, order[pos].0)?;
            order.swap(pos - 1, pos);
            pos -= 1;
        }
        next_slot += 1;
    }
    make_t_diagonal_nonnegative(&mut fx);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &Mat, b: &Mat) -> Mat {
        let n = a.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                    .collect()
            })
            .collect()
    }

    fn transpose(a: &Mat) -> Mat {
        let n = a.len();
        (0..n).map(|i| (0..n).map(|j| a[j][i]).collect()).collect()
    }

    fn rel_err(x: &Mat, y: &Mat) -> f64 {
        let diff: f64 = x
            .iter()
            .flatten()
            .zip(y.iter().flatten())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let norm: f64 = y.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        diff / norm.max(1e-300)
    }

    /// Reconstruction, orthogonality and structure of a generalized Schur form.
    fn assert_generalized_schur(a: &Mat, b: &Mat, s: &Mat, t: &Mat, q: &Mat, z: &Mat) {
        let n = a.len();
        let ra = rel_err(&matmul(&matmul(q, s), &transpose(z)), a);
        let rb = rel_err(&matmul(&matmul(q, t), &transpose(z)), b);
        let oq = rel_err(&matmul(&transpose(q), q), &identity(n));
        let oz = rel_err(&matmul(&transpose(z), z), &identity(n));
        assert!(
            ra < 1e-13 && rb < 1e-13 && oq < 1e-13 && oz < 1e-13,
            "ra {ra:e} rb {rb:e} oq {oq:e} oz {oz:e}"
        );
        for i in 0..n {
            for j in 0..i {
                assert_eq!(t[i][j], 0.0, "T[{i}][{j}]");
                if i > j + 1 {
                    assert_eq!(s[i][j], 0.0, "S[{i}][{j}]");
                }
            }
        }
        for i in 1..n.saturating_sub(1) {
            assert!(
                s[i][i - 1] == 0.0 || s[i + 1][i] == 0.0,
                "two consecutive subdiagonals at {i}"
            );
        }
        for i in 0..n {
            assert!(t[i][i] >= 0.0, "T diagonal must be non-negative");
        }
    }

    fn pencils() -> Vec<(Mat, Mat)> {
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        (0..120)
            .map(|trial| {
                let n = 1 + trial % 10;
                let a: Mat = (0..n).map(|_| (0..n).map(|_| rnd()).collect()).collect();
                let mut b: Mat = (0..n).map(|_| (0..n).map(|_| rnd()).collect()).collect();
                if trial % 3 == 1 && n > 1 {
                    // rank-deficient B: an infinite eigenvalue
                    for row in &mut b {
                        row[n - 1] = row[0];
                    }
                }
                (a, b)
            })
            .collect()
    }

    // frankenscipy-szq1n.5: singular B is the case QZ exists for. SciPy 1.17.1:
    // qz([[1,2],[3,4]], [[1,0],[0,0]]) succeeds with diag(BB) = [0.8, 0], i.e. the finite
    // eigenvalue -0.5 and one infinite eigenvalue. The A·B⁻¹ implementation returned an error.
    #[test]
    fn singular_b_gives_an_infinite_eigenvalue_like_scipy() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let (s, t, q, z) = real_qz(&a, &b).expect("qz");
        assert_generalized_schur(&a, &b, &s, &t, &q, &z);
        let mut betas = [t[0][0], t[1][1]];
        betas.sort_by(f64::total_cmp);
        assert!(
            betas[0].abs() < 1e-15 && (betas[1] - 0.8).abs() < 1e-14,
            "{t:?}"
        );
        let finite = if t[0][0] > 0.5 { 0 } else { 1 };
        assert!((s[finite][finite] / t[finite][finite] + 0.5).abs() < 1e-14);
    }

    #[test]
    fn qz_is_a_generalized_schur_form_including_singular_b() {
        for (a, b) in pencils() {
            let (s, t, q, z) = real_qz(&a, &b).expect("qz converges");
            assert_generalized_schur(&a, &b, &s, &t, &q, &z);
        }
    }

    #[test]
    fn reordering_keeps_the_form_and_puts_the_selection_first() {
        let mut reordered = 0;
        for (a, b) in pencils() {
            let (mut s, mut t, mut q, mut z) = real_qz(&a, &b).expect("qz");
            let before: Vec<(f64, f64, f64)> = diagonal_blocks(&s)
                .into_iter()
                .map(|(j, sz)| block_eigenvalue(&s, &t, j, sz))
                .collect();
            let lhp = |re: f64, _im: f64, beta: f64| beta != 0.0 && re / beta < 0.0;
            reorder_selected_first(&mut s, &mut t, &mut q, &mut z, lhp).expect("reorder");
            assert_generalized_schur(&a, &b, &s, &t, &q, &z);
            let flags: Vec<bool> = diagonal_blocks(&s)
                .into_iter()
                .filter(|&(j, _)| t[j][j] > 1e-13)
                .map(|(j, sz)| {
                    let (re, im, beta) = block_eigenvalue(&s, &t, j, sz);
                    lhp(re, im, beta)
                })
                .collect();
            assert!(
                flags.windows(2).all(|w| w[0] || !w[1]),
                "a selected block follows an unselected one: {flags:?}"
            );
            let selected_before = before.iter().filter(|e| lhp(e.0, e.1, e.2)).count();
            if selected_before > 0 && selected_before < before.len() {
                reordered += 1;
            }
        }
        assert!(
            reordered > 20,
            "the fixtures must exercise real reorders ({reordered})"
        );
    }
}
