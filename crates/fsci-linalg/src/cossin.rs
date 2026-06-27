//! Building blocks for the cosine-sine (CS) decomposition (`scipy.linalg.cossin`).
//!
//! The balanced-square factor path follows LAPACK's `dorcsd`, which chains
//! `dorbdb` (simultaneous bidiagonalization of the 2×2 partitioned orthogonal
//! matrix) and `dbbcsd` (bidiagonal CS values via an implicit QR sweep).
//! `cossin_angles` covers the general `(p, q)` CS middle factor.
//!
//! The LAPACK ports are intentionally direct: the Householder and rotation
//! primitives keep the reference conventions so the sign-gauge behavior can be
//! checked against SciPy's LAPACK-backed implementation.
#![allow(dead_code)]
// This module transcribes LAPACK reference kernels (dorbdb/dbbcsd and their
// Householder/rotation primitives) line-for-line. Index-based loops, explicit
// `x = x op y` updates, and the multi-factor result tuples mirror the Fortran
// for verifiability against the reference; the corresponding clippy style lints
// are intentionally allowed here.
#![allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::type_complexity,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains
)]

/// Generate an elementary Householder reflector with a **non-negative** beta,
/// faithfully matching LAPACK `DLARFGP`.
///
/// On input the reflector acts on the vector `[alpha; tail]` (length `n =
/// tail.len() + 1`). On output `tail` is overwritten with `v[1..]` (the implicit
/// `v[0] = 1`), and `(beta, tau)` are returned such that
/// `H = I − tau·v·vᵀ` satisfies `H·[alpha; tail] = [beta; 0]` with `beta ≥ 0`.
///
/// The extreme-underflow rescaling loop of the reference (`KNT`) is omitted: it
/// only triggers for subnormal `beta`, which never arises for the orthogonal
/// inputs of the CS decomposition.
pub(crate) fn dlarfgp(alpha_in: f64, tail: &mut [f64]) -> (f64, f64) {
    // n == 1 (empty tail): H is trivial; pick the sign so beta = |alpha| ≥ 0.
    if tail.is_empty() {
        return (alpha_in.abs(), if alpha_in >= 0.0 { 0.0 } else { 2.0 });
    }

    let eps = f64::EPSILON;
    let xnorm = tail.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut alpha = alpha_in;

    if xnorm <= eps * alpha.abs() {
        // H = [±1, 0; 0, I], sign chosen so beta = |alpha| ≥ 0.
        if alpha >= 0.0 {
            return (alpha, 0.0);
        }
        for v in tail.iter_mut() {
            *v = 0.0;
        }
        return (-alpha, 2.0);
    }

    // General case. beta = SIGN(hypot(alpha, xnorm), alpha) (Fortran SIGN: +0 → +).
    let mut beta = if alpha >= 0.0 {
        alpha.hypot(xnorm)
    } else {
        -alpha.hypot(xnorm)
    };
    let savealpha = alpha;
    alpha += beta;
    let tau;
    if beta < 0.0 {
        beta = -beta;
        tau = -alpha / beta;
    } else {
        alpha = xnorm * (xnorm / alpha);
        tau = alpha / beta;
        alpha = -alpha;
    }

    let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
    if tau.abs() <= smlnum {
        // Denormal tau loses accuracy; flush per the reference.
        if savealpha >= 0.0 {
            return (beta, 0.0);
        }
        for v in tail.iter_mut() {
            *v = 0.0;
        }
        return (-savealpha, 2.0);
    }

    // Scale the reflector tail v[1..] = x[1..] / alpha.
    for v in tail.iter_mut() {
        *v /= alpha;
    }
    (beta, tau)
}

/// Apply a Householder reflector `H = I − tau·v·vᵀ` (with `v[0] = 1`, the rest in
/// `v_tail`) to the columns of a row-major `rows × cols` matrix `c` from the
/// **left** (`H·C`), matching the `'L'` side of LAPACK `DLARF`/`DLARF1F`.
///
/// `v` has length `rows`: `v[0] = 1` implicitly, `v_tail = v[1..]`. `tau == 0`
/// is a no-op.
pub(crate) fn dlarf_left(c: &mut [Vec<f64>], v_tail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    let rows = c.len();
    if rows == 0 {
        return;
    }
    let cols = c[0].len();
    // v[0] = 1, v[1..] = v_tail.
    let v = |i: usize| if i == 0 { 1.0 } else { v_tail[i - 1] };
    // For each column j: w = vᵀ·C[:,j]; C[:,j] -= tau·w·v.
    for j in 0..cols {
        let mut w = 0.0;
        for i in 0..rows {
            w += v(i) * c[i][j];
        }
        let tw = tau * w;
        for i in 0..rows {
            c[i][j] -= tw * v(i);
        }
    }
}

/// Apply a Householder reflector to the rows of a row-major `rows × cols` matrix
/// `c` from the **right** (`C·H`), matching the `'R'` side of LAPACK `DLARF`.
///
/// `v` has length `cols`: `v[0] = 1` implicitly, `v_tail = v[1..]`.
pub(crate) fn dlarf_right(c: &mut [Vec<f64>], v_tail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    let rows = c.len();
    if rows == 0 {
        return;
    }
    let cols = c[0].len();
    let v = |j: usize| if j == 0 { 1.0 } else { v_tail[j - 1] };
    // For each row i: w = C[i,:]·v; C[i,:] -= tau·w·v.
    for row in c.iter_mut().take(rows) {
        let mut w = 0.0;
        for j in 0..cols {
            w += row[j] * v(j);
        }
        let tw = tau * w;
        for j in 0..cols {
            row[j] -= tw * v(j);
        }
    }
}

/// Compute the CS-decomposition angle vector `theta` from the upper-left block
/// `x11` (`p × q`) of an `m × m` orthogonal matrix, matching the `theta` returned
/// by `scipy.linalg.cossin(..., separate=True)`.
///
/// The cosines of the angles are the `r = min(p, q, m-p, m-q)` smallest singular
/// values of `x11` (the larger `min(p,q) − r` singular values are ≈ 1 and form
/// the identity blocks). With `n11 = min(p,q) − r` and the singular values `σ`
/// in descending order, `theta[i] = arccos(σ[n11 + i])`, ascending.
pub(crate) fn cs_angles(
    x11: &[Vec<f64>],
    p: usize,
    q: usize,
    m: usize,
) -> Result<Vec<f64>, crate::LinalgError> {
    let r = p.min(q).min(m - p).min(m - q);
    let n11 = p.min(q) - r;
    let res = crate::svd(x11, crate::DecompOptions::default())?;
    let theta = (0..r)
        .map(|i| res.s[n11 + i].clamp(-1.0, 1.0).acos())
        .collect();
    Ok(theta)
}

/// Cosine-sine decomposition angles and middle factor of an `m × m` orthogonal
/// matrix `x` partitioned with a `p × q` upper-left block, matching the
/// `compute_u=False, compute_vh=False` mode of `scipy.linalg.cossin`.
///
/// Returns `(theta, cs)`: `theta` is the vector of CS angles (radians), and `cs`
/// is the `m × m` cosine-sine middle factor (default `swap_sign=False`
/// convention). The orthogonal factors `u`/`vh` are not yet produced (their
/// LAPACK-identical signs require the `dorbdb`/`dbbcsd` port — see bead
/// frankenscipy-5tmu1); this function provides the angle/structure side, which
/// is independent of that sign convention.
///
/// `x` must be `m × m` orthogonal with `0 < p < m`, `0 < q < m`.
pub fn cossin_angles(
    x: &[Vec<f64>],
    p: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), crate::LinalgError> {
    let m = validate_partitioned_square(x, p, q, "cossin_angles")?;
    let x11: Vec<Vec<f64>> = x[..p].iter().map(|row| row[..q].to_vec()).collect();
    let theta = cs_angles(&x11, p, q, m)?;
    let cs = build_cs_matrix(&theta, p, q, m);
    Ok((theta, cs))
}

fn validate_partitioned_square(
    x: &[Vec<f64>],
    p: usize,
    q: usize,
    caller: &str,
) -> Result<usize, crate::LinalgError> {
    let m = x.len();
    if m == 0 || x.iter().any(|row| row.len() != m) {
        return Err(crate::LinalgError::InvalidArgument {
            detail: format!("{caller}: x must be a non-empty square matrix"),
        });
    }
    if p == 0 || q == 0 || p >= m || q >= m {
        return Err(crate::LinalgError::InvalidArgument {
            detail: format!("{caller}: p and q must satisfy 0 < p,q < m"),
        });
    }
    Ok(m)
}

/// Assemble the `m × m` cosine-sine middle factor `CS` from the angle vector
/// `theta`, matching the construction in scipy's `_cossin` (default
/// `swap_sign=False`: the `-S`/`-I` blocks sit in the upper-right).
///
/// `p`, `q` are the row/column counts of the upper-left block `X11`; `r =
/// min(p, q, m-p, m-q)` is the number of nontrivial angles (`theta.len() == r`).
/// The identity-block ranks are `n11 = min(p,q)-r`, `n12 = min(p,m-q)-r`,
/// `n21 = min(m-p,q)-r`, `n22 = min(m-p,m-q)-r`.
pub(crate) fn build_cs_matrix(theta: &[f64], p: usize, q: usize, m: usize) -> Vec<Vec<f64>> {
    let r = p.min(q).min(m - p).min(m - q);
    let n11 = p.min(q) - r;
    let n12 = p.min(m - q) - r;
    let n21 = (m - p).min(q) - r;
    let n22 = (m - p).min(m - q) - r;
    let mut cs = vec![vec![0.0_f64; m]; m];

    // Leading identity block.
    for i in 0..n11 {
        cs[i][i] = 1.0;
    }
    // Upper-right -I block.
    {
        let xs = n11 + r;
        let ys = n11 + n21 + n22 + 2 * r;
        for i in 0..n12 {
            cs[xs + i][ys + i] = -1.0;
        }
    }
    // Lower-left I block.
    {
        let xs = p + n22 + r;
        let ys = n11 + r;
        for i in 0..n21 {
            cs[xs + i][ys + i] = 1.0;
        }
    }
    // Middle I block.
    for i in 0..n22 {
        cs[p + i][q + i] = 1.0;
    }
    // Cosine diagonals (upper-left and lower-right).
    for (i, &t) in theta.iter().enumerate().take(r) {
        cs[n11 + i][n11 + i] = t.cos();
        cs[p + n22 + i][n11 + r + n21 + n22 + i] = t.cos();
    }
    // Sine blocks: -S (upper-right) and +S (lower-left).
    for (i, &t) in theta.iter().enumerate().take(r) {
        cs[n11 + i][n11 + n21 + n22 + r + i] = -t.sin();
        cs[p + n22 + i][n11 + i] = t.sin();
    }
    cs
}

// Apply H = I − tau·v·vᵀ (v[r0]=1, `vtail` at rows r0+1..) to columns `c0..c1`
// of the row-major matrix `c`, from the left.
fn apply_left_block(c: &mut [Vec<f64>], r0: usize, c0: usize, c1: usize, vtail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    for col in c0..c1 {
        let mut w = c[r0][col];
        for (k, &vt) in vtail.iter().enumerate() {
            w += vt * c[r0 + 1 + k][col];
        }
        let tw = tau * w;
        c[r0][col] -= tw;
        for (k, &vt) in vtail.iter().enumerate() {
            c[r0 + 1 + k][col] -= tw * vt;
        }
    }
}

// Apply C·H (H = I − tau·w·wᵀ, w[c0]=1, `wtail` at cols c0+1..) to rows `r0..r1`
// of the row-major matrix `c`, from the right.
fn apply_right_block(c: &mut [Vec<f64>], r0: usize, r1: usize, c0: usize, wtail: &[f64], tau: f64) {
    if tau == 0.0 {
        return;
    }
    for row in c.iter_mut().take(r1).skip(r0) {
        let mut w = row[c0];
        for (k, &wt) in wtail.iter().enumerate() {
            w += wt * row[c0 + 1 + k];
        }
        let tw = tau * w;
        row[c0] -= tw;
        for (k, &wt) in wtail.iter().enumerate() {
            row[c0 + 1 + k] -= tw * wt;
        }
    }
}

/// Reflector scalars and angles from [`dorbdb_balanced`].
pub(crate) struct OrbdbResult {
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub taup1: Vec<f64>,
    pub taup2: Vec<f64>,
    pub tauq1: Vec<f64>,
    pub tauq2: Vec<f64>,
}

/// Simultaneous bidiagonalization (LAPACK `DORBDB`, `TRANS='N'`, default signs)
/// of the four `q×q` blocks of a `2q×2q` partitioned orthogonal matrix (the
/// balanced square case `p = q = m/2`). On return the blocks hold the packed
/// Householder reflector tails; the angles and reflector scalars are returned.
pub(crate) fn dorbdb_balanced(
    x11: &mut [Vec<f64>],
    x12: &mut [Vec<f64>],
    x21: &mut [Vec<f64>],
    x22: &mut [Vec<f64>],
    q: usize,
) -> OrbdbResult {
    let (p, mp) = (q, q); // P = M-P = q
    let mut theta = vec![0.0f64; q];
    let mut phi = vec![0.0f64; q.saturating_sub(1)];
    let mut taup1 = vec![0.0; q];
    let mut taup2 = vec![0.0; q];
    let mut tauq1 = vec![0.0; q.saturating_sub(1)];
    let mut tauq2 = vec![0.0; q];

    for i in 0..q {
        // Step 1: column update mixing X11/X12 (and X21/X22) by phi[i-1].
        if i > 0 {
            let (c, s) = (phi[i - 1].cos(), phi[i - 1].sin());
            for r in i..p {
                x11[r][i] = c * x11[r][i] - s * x12[r][i - 1];
            }
            for r in i..mp {
                x21[r][i] = c * x21[r][i] - s * x22[r][i - 1];
            }
        }
        // Step 2: theta[i] = atan2(||X21(i:,i)||, ||X11(i:,i)||).
        let n21 = (i..mp).map(|r| x21[r][i] * x21[r][i]).sum::<f64>().sqrt();
        let n11 = (i..p).map(|r| x11[r][i] * x11[r][i]).sum::<f64>().sqrt();
        theta[i] = n21.atan2(n11);
        // Step 3: column reflectors P1, P2.
        {
            let mut tail: Vec<f64> = (i + 1..p).map(|r| x11[r][i]).collect();
            let (beta, tau) = dlarfgp(x11[i][i], &mut tail);
            x11[i][i] = beta;
            for (k, r) in (i + 1..p).enumerate() {
                x11[r][i] = tail[k];
            }
            taup1[i] = tau;
        }
        {
            let mut tail: Vec<f64> = (i + 1..mp).map(|r| x21[r][i]).collect();
            let (beta, tau) = dlarfgp(x21[i][i], &mut tail);
            x21[i][i] = beta;
            for (k, r) in (i + 1..mp).enumerate() {
                x21[r][i] = tail[k];
            }
            taup2[i] = tau;
        }
        // Step 4: apply P1/P2 from the left.
        let v1: Vec<f64> = (i + 1..p).map(|r| x11[r][i]).collect();
        if i + 1 < q {
            apply_left_block(x11, i, i + 1, q, &v1, taup1[i]);
        }
        apply_left_block(x12, i, i, q, &v1, taup1[i]);
        let v2: Vec<f64> = (i + 1..mp).map(|r| x21[r][i]).collect();
        if i + 1 < q {
            apply_left_block(x21, i, i + 1, q, &v2, taup2[i]);
        }
        apply_left_block(x22, i, i, q, &v2, taup2[i]);
        // Step 5: row update mixing X11/X21 (and X12/X22) by theta[i].
        let (st, ct) = (theta[i].sin(), theta[i].cos());
        if i + 1 < q {
            for col in i + 1..q {
                x11[i][col] = -st * x11[i][col] + ct * x21[i][col];
            }
        }
        for col in i..q {
            x12[i][col] = -st * x12[i][col] + ct * x22[i][col];
        }
        // Step 6: phi[i] = atan2(||X11(i,i+1:)||, ||X12(i,i:)||).
        if i + 1 < q {
            let n11r = (i + 1..q)
                .map(|c| x11[i][c] * x11[i][c])
                .sum::<f64>()
                .sqrt();
            let n12r = (i..q).map(|c| x12[i][c] * x12[i][c]).sum::<f64>().sqrt();
            phi[i] = n11r.atan2(n12r);
        }
        // Step 7: row reflectors Q1, Q2.
        if i + 1 < q {
            let mut tail: Vec<f64> = (i + 2..q).map(|c| x11[i][c]).collect();
            let (beta, tau) = dlarfgp(x11[i][i + 1], &mut tail);
            x11[i][i + 1] = beta;
            for (k, c) in (i + 2..q).enumerate() {
                x11[i][c] = tail[k];
            }
            tauq1[i] = tau;
        }
        {
            let mut tail: Vec<f64> = (i + 1..q).map(|c| x12[i][c]).collect();
            let (beta, tau) = dlarfgp(x12[i][i], &mut tail);
            x12[i][i] = beta;
            for (k, c) in (i + 1..q).enumerate() {
                x12[i][c] = tail[k];
            }
            tauq2[i] = tau;
        }
        // Step 8: apply Q1/Q2 from the right to the trailing rows.
        if i + 1 < q {
            let w1: Vec<f64> = (i + 2..q).map(|c| x11[i][c]).collect();
            apply_right_block(x11, i + 1, p, i + 1, &w1, tauq1[i]);
            apply_right_block(x21, i + 1, mp, i + 1, &w1, tauq1[i]);
        }
        let w2: Vec<f64> = (i + 1..q).map(|c| x12[i][c]).collect();
        if i + 1 < p {
            apply_right_block(x12, i + 1, p, i, &w2, tauq2[i]);
        }
        if i + 1 < mp {
            apply_right_block(x22, i + 1, mp, i, &w2, tauq2[i]);
        }
    }
    OrbdbResult {
        theta,
        phi,
        taup1,
        taup2,
        tauq1,
        tauq2,
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    a
}

// Accumulate the orthogonal factor from column Householder reflectors stored in
// `src` (reflector i: v[i]=1, tail = src[i+1..n][i]).
pub(crate) fn build_from_col_reflectors(src: &[Vec<f64>], tau: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut a = identity(n);
    for i in (0..tau.len()).rev() {
        let tail: Vec<f64> = (i + 1..n).map(|r| src[r][i]).collect();
        apply_left_block(&mut a, i, 0, n, &tail, tau[i]);
    }
    a
}

// Accumulate the orthogonal factor from row Householder reflectors stored in
// `src` (reflector i: w[i+col_off]=1, tail = src[i][i+col_off+1..n]).
pub(crate) fn build_from_row_reflectors(
    src: &[Vec<f64>],
    tau: &[f64],
    n: usize,
    col_off: usize,
) -> Vec<Vec<f64>> {
    let mut a = identity(n);
    for i in (0..tau.len()).rev() {
        let cs = i + col_off;
        let tail: Vec<f64> = (cs + 1..n).map(|c| src[i][c]).collect();
        apply_left_block(&mut a, cs, 0, n, &tail, tau[i]);
    }
    a
}

/// Build the `2q×2q` bidiagonal-block matrix `BigB` from the CS angles, matching
/// LAPACK's `DBBCSD` input layout (balanced square, `M-P-Q = 0`).
pub(crate) fn build_bigb(theta: &[f64], phi: &[f64], q: usize) -> Vec<Vec<f64>> {
    let m = 2 * q;
    let mut b = vec![vec![0.0; m]; m];
    let c: Vec<f64> = theta.iter().map(|t| t.cos()).collect();
    let s: Vec<f64> = theta.iter().map(|t| t.sin()).collect();
    // This matches the bidiagonalization produced by [`dorbdb_balanced`]:
    // B11 upper-bidiagonal, B22 lower-bidiagonal (LAPACK DBBCSD convention),
    // B12 lower-bidiagonal but sign-negated, and B21 upper-bidiagonal positive.
    for i in 0..q {
        let cp_prev = if i == 0 { 1.0 } else { phi[i - 1].cos() };
        let cp_cur = if i + 1 < q { phi[i].cos() } else { 1.0 };
        b[i][i] = c[i] * cp_prev; // B11 D
        b[q + i][i] = s[i] * cp_prev; // B21 D (positive)
        b[i][q + i] = -s[i] * cp_cur; // B12 D (negated)
        b[q + i][q + i] = c[i] * cp_cur; // B22 D
    }
    for i in 0..q.saturating_sub(1) {
        let sp = phi[i].sin();
        b[i][i + 1] = -s[i] * sp; // B11 superdiagonal
        b[i + 1][q + i] = -c[i + 1] * sp; // B12 subdiagonal (negated)
        b[q + i][i + 1] = c[i] * sp; // B21 superdiagonal (positive, upper)
        b[q + i + 1][q + i] = -s[i + 1] * sp; // B22 subdiagonal
    }
    b
}

// Rotation primitives for the (pending) DBBCSD CS bidiagonal-SVD sweep.
// Verified building blocks; will be consumed by the implicit-QR diagonalizer
// that produces scipy-exact U/V signs (bead frankenscipy-5tmu1).

/// Singular values of the 2×2 upper-triangular `[[f, g], [0, h]]`, matching
/// LAPACK `DLAS2`. Returns `(ssmin, ssmax)`.
#[allow(dead_code)]
pub(crate) fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt()
        };
        return (0.0, ssmax);
    }
    if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx).powi(2);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
            let mut ssmin = (fhmn * c) * au;
            ssmin += ssmin;
            (ssmin, ga / (c + c))
        }
    }
}

/// Plane rotation with non-negative `r`, matching LAPACK `DLARTGP`. Returns
/// `(cs, sn, r)` with `cs·f + sn·g = r ≥ 0` and `-sn·f + cs·g = 0`. (The
/// over/underflow scaling of the reference is unneeded for the O(1) CS values
/// this is used on.)
#[allow(dead_code)]
pub(crate) fn dlartgp(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        (if f >= 0.0 { 1.0 } else { -1.0 }, 0.0, f.abs())
    } else if f == 0.0 {
        (0.0, if g >= 0.0 { 1.0 } else { -1.0 }, g.abs())
    } else {
        let r = f.hypot(g);
        (f / r, g / r, r)
    }
}

/// Generate the rotation `(cs, sn)` for a CS-decomposition implicit-QR shift,
/// matching LAPACK `DLARTGS(x, y, sigma)`.
#[allow(dead_code)]
pub(crate) fn dlartgs(x: f64, y: f64, sigma: f64) -> (f64, f64) {
    let thresh = f64::EPSILON;
    let (z, w);
    if (sigma == 0.0 && x.abs() < thresh) || (x.abs() == sigma && y == 0.0) {
        z = 0.0;
        w = 0.0;
    } else if sigma == 0.0 {
        if x >= 0.0 {
            z = x;
            w = y;
        } else {
            z = -x;
            w = -y;
        }
    } else if x.abs() < thresh {
        z = -sigma * sigma;
        w = 0.0;
    } else {
        let s = if x >= 0.0 { 1.0 } else { -1.0 };
        z = s * (x.abs() - sigma) * (s + sigma / x);
        w = s * y;
    }
    // LAPACK: CALL DLARTGP(W, Z, SN, CS, R) — note the swapped outputs.
    let (sn, cs, _r) = dlartgp(w, z);
    (cs, sn)
}

/// Apply a forward sequence of variable-pivot plane rotations to a row-major
/// matrix `a`, matching LAPACK `DLASR` with `PIVOT='V'`, `DIRECT='F'`. Rotation
/// `k` (cosine `c[k]`, sine `s[k]`) acts in the `(k, k+1)` plane: on rows for
/// `side='L'` (`A := P·A`), on columns for `side='R'` (`A := A·Pᵀ`), applied
/// for `k = 0, 1, …` in order.
#[allow(dead_code)]
pub(crate) fn dlasr_var_fwd(side: char, c: &[f64], s: &[f64], a: &mut [Vec<f64>]) {
    let rows = a.len();
    if rows == 0 {
        return;
    }
    let cols = a[0].len();
    if side == 'L' {
        for j in 0..rows.saturating_sub(1) {
            let (ct, st) = (c[j], s[j]);
            if ct == 1.0 && st == 0.0 {
                continue;
            }
            for i in 0..cols {
                let temp = a[j + 1][i];
                a[j + 1][i] = ct * temp - st * a[j][i];
                a[j][i] = st * temp + ct * a[j][i];
            }
        }
    } else {
        // side == 'R'
        for j in 0..cols.saturating_sub(1) {
            let (ct, st) = (c[j], s[j]);
            if ct == 1.0 && st == 0.0 {
                continue;
            }
            for row in a.iter_mut().take(rows) {
                let temp = row[j + 1];
                row[j + 1] = ct * temp - st * row[j];
                row[j] = st * temp + ct * row[j];
            }
        }
    }
}

fn mm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, k, p) = (a.len(), b.len(), b[0].len());
    let mut c = vec![vec![0.0; p]; n];
    for i in 0..n {
        for l in 0..k {
            let ail = a[i][l];
            for j in 0..p {
                c[i][j] += ail * b[l][j];
            }
        }
    }
    c
}

// Apply a single Givens rotation (cs, sn) in plane (j, j+1) to columns of `a`
// (right side: a := a·G where G rotates cols j,j+1).
fn rot_cols(a: &mut [Vec<f64>], j: usize, cs: f64, sn: f64) {
    for row in a.iter_mut() {
        let t = row[j + 1];
        row[j + 1] = cs * t - sn * row[j];
        row[j] = sn * t + cs * row[j];
    }
}
// Apply a single Givens rotation (cs, sn) in plane (j, j+1) to rows of `a`
// (left side: a := G·a where G rotates rows j,j+1).
fn rot_rows(a: &mut [Vec<f64>], j: usize, cs: f64, sn: f64) {
    let n = a[0].len();
    for c in 0..n {
        let t = a[j + 1][c];
        a[j + 1][c] = cs * t - sn * a[j][c];
        a[j][c] = sn * t + cs * a[j][c];
    }
}

/// Result of [`dbbcsd_balanced`].
pub(crate) struct BbcsdResult {
    pub theta: Vec<f64>,
    pub u1: Vec<Vec<f64>>,
    pub u2: Vec<Vec<f64>>,
    pub v1t: Vec<Vec<f64>>,
    pub v2t: Vec<Vec<f64>>,
}

/// CS bidiagonal-SVD by implicit QR (LAPACK `DBBCSD`, balanced square `p=q=m/2`,
/// generic non-degenerate spectrum). Diagonalizes the bidiagonal-block form
/// defined by the `dorbdb` angles `theta`/`phi` into the clean CS form,
/// accumulating the Givens rotations into `u1`,`u2` (column-rotated) and
/// `v1t`,`v2t` (row-rotated, i.e. `V1ᵀ`/`V2ᵀ`), all `q×q`, identity-initialized.
///
/// This is the COLMAJOR path of the reference, restricted to the case where no
/// mid-sweep deflation (`RESTART`) is triggered — which holds for spectra with
/// distinct angles bounded away from `0` and `π/2` (random orthogonal inputs).
pub(crate) fn dbbcsd_balanced(theta_in: &[f64], phi_in: &[f64], q: usize) -> BbcsdResult {
    let thresh = (f64::EPSILON).powf(0.875); // matches dbbcsd TOLMUL-ish gate loosely
    let unfl = f64::MIN_POSITIVE;
    let pi2 = std::f64::consts::FRAC_PI_2;
    let mut theta = theta_in.to_vec();
    let mut phi = phi_in.to_vec(); // length q-1
    let mut u1 = identity(q);
    let mut u2 = identity(q);
    let mut v1t = identity(q);
    let mut v2t = identity(q);

    // Bidiagonal D/E arrays (0-indexed). D length q, E length q-1.
    let mut b11d = vec![0.0; q];
    let mut b11e = vec![0.0; q.saturating_sub(1)];
    let mut b12d = vec![0.0; q];
    let mut b12e = vec![0.0; q.saturating_sub(1)];
    let mut b21d = vec![0.0; q];
    let mut b21e = vec![0.0; q.saturating_sub(1)];
    let mut b22d = vec![0.0; q];
    let mut b22e = vec![0.0; q.saturating_sub(1)];

    // Initial IMIN/IMAX (0-indexed inclusive). imax = last index with nonzero phi chain.
    let mut imax = q - 1;
    while imax > 0 && phi[imax - 1] == 0.0 {
        imax -= 1;
    }
    let mut imin = imax.saturating_sub(1);
    while imin > 0 && phi[imin - 1] != 0.0 {
        imin -= 1;
    }

    let maxit = 6 * q * q;
    let mut iter = 0usize;

    while imax > 0 {
        // (Re)build the bidiagonal blocks over [imin, imax].
        b11d[imin] = theta[imin].cos();
        b21d[imin] = -theta[imin].sin();
        for i in imin..imax {
            b11e[i] = -theta[i].sin() * phi[i].sin();
            b11d[i + 1] = theta[i + 1].cos() * phi[i].cos();
            b12d[i] = theta[i].sin() * phi[i].cos();
            b12e[i] = theta[i + 1].cos() * phi[i].sin();
            b21e[i] = -theta[i].cos() * phi[i].sin();
            b21d[i + 1] = -theta[i + 1].sin() * phi[i].cos();
            b22d[i] = theta[i].cos() * phi[i].cos();
            b22e[i] = -theta[i + 1].sin() * phi[i].sin();
        }
        b12d[imax] = theta[imax].sin();
        b22d[imax] = theta[imax].cos();

        if iter > maxit {
            break;
        }
        iter += imax - imin;

        // Compute shift mu/nu.
        let mut thmax = theta[imin];
        let mut thmin = theta[imin];
        for i in imin + 1..=imax {
            thmax = thmax.max(theta[i]);
            thmin = thmin.min(theta[i]);
        }
        let (mu, nu);
        if thmax > pi2 - thresh {
            mu = 0.0;
            nu = 1.0;
        } else if thmin < thresh {
            mu = 1.0;
            nu = 0.0;
        } else {
            let (s11, _) = dlas2(b11d[imax - 1], b11e[imax - 1], b11d[imax]);
            let (s21, _) = dlas2(b21d[imax - 1], b21e[imax - 1], b21d[imax]);
            if s11 <= s21 {
                let m = s11;
                if m < thresh {
                    mu = 0.0;
                    nu = 1.0;
                } else {
                    mu = m;
                    nu = (1.0 - m * m).sqrt();
                }
            } else {
                let nn = s21;
                if nn < thresh {
                    mu = 1.0;
                    nu = 0.0;
                } else {
                    nu = nn;
                    mu = (1.0 - nn * nn).sqrt();
                }
            }
        }

        // Rotation arrays for this sweep (indexed by global position).
        let mut v1tcs = vec![0.0; q];
        let mut v1tsn = vec![0.0; q];
        let mut u1cs = vec![0.0; q];
        let mut u1sn = vec![0.0; q];
        let mut u2cs = vec![0.0; q];
        let mut u2sn = vec![0.0; q];
        let mut v2tcs = vec![0.0; q];
        let mut v2tsn = vec![0.0; q];

        // --- Initial V1T rotation at imin ---
        if mu <= nu {
            let (c, s) = dlartgs(b11d[imin], b11e[imin], mu);
            v1tcs[imin] = c;
            v1tsn[imin] = s;
        } else {
            let (c, s) = dlartgs(b21d[imin], b21e[imin], nu);
            v1tcs[imin] = c;
            v1tsn[imin] = s;
        }
        let (c, s) = (v1tcs[imin], v1tsn[imin]);
        let mut temp = c * b11d[imin] + s * b11e[imin];
        b11e[imin] = c * b11e[imin] - s * b11d[imin];
        b11d[imin] = temp;
        let mut b11bulge = s * b11d[imin + 1];
        b11d[imin + 1] = c * b11d[imin + 1];
        temp = c * b21d[imin] + s * b21e[imin];
        b21e[imin] = c * b21e[imin] - s * b21d[imin];
        b21d[imin] = temp;
        let mut b21bulge = s * b21d[imin + 1];
        b21d[imin + 1] = c * b21d[imin + 1];
        theta[imin] = (b21d[imin] * b21d[imin] + b21bulge * b21bulge)
            .sqrt()
            .atan2((b11d[imin] * b11d[imin] + b11bulge * b11bulge).sqrt());

        // --- U1 rotation at imin ---
        if b11d[imin] * b11d[imin] + b11bulge * b11bulge
            > (thresh * b11d[imin].abs().max(b11d[imin + 1].abs()).max(unfl)).powi(2)
        {
            let (sn, cs, _r) = dlartgp(b11bulge, b11d[imin]);
            u1sn[imin] = sn;
            u1cs[imin] = cs;
        } else if mu <= nu {
            let (cc, ss) = dlartgs(b11e[imin], b11d[imin + 1], mu);
            u1cs[imin] = cc;
            u1sn[imin] = ss;
        } else {
            let (cc, ss) = dlartgs(b12d[imin], b12e[imin], nu);
            u1cs[imin] = cc;
            u1sn[imin] = ss;
        }
        // --- U2 rotation at imin ---
        if b21d[imin] * b21d[imin] + b21bulge * b21bulge
            > (thresh * b21d[imin].abs().max(b21d[imin + 1].abs()).max(unfl)).powi(2)
        {
            let (sn, cs, _r) = dlartgp(b21bulge, b21d[imin]);
            u2sn[imin] = sn;
            u2cs[imin] = cs;
        } else if nu < mu {
            let (cc, ss) = dlartgs(b21e[imin], b21d[imin + 1], nu);
            u2cs[imin] = cc;
            u2sn[imin] = ss;
        } else {
            let (cc, ss) = dlartgs(b22d[imin], b22e[imin], mu);
            u2cs[imin] = cc;
            u2sn[imin] = ss;
        }
        u2cs[imin] = -u2cs[imin];
        u2sn[imin] = -u2sn[imin];

        // Apply U1 to B11,B12 and U2 to B21,B22 at imin.
        let (u1c, u1s) = (u1cs[imin], u1sn[imin]);
        temp = u1c * b11e[imin] + u1s * b11d[imin + 1];
        b11d[imin + 1] = u1c * b11d[imin + 1] - u1s * b11e[imin];
        b11e[imin] = temp;
        if imax > imin + 1 {
            b11bulge = u1s * b11e[imin + 1];
            b11e[imin + 1] = u1c * b11e[imin + 1];
        }
        temp = u1c * b12d[imin] + u1s * b12e[imin];
        b12e[imin] = u1c * b12e[imin] - u1s * b12d[imin];
        b12d[imin] = temp;
        let mut b12bulge = u1s * b12d[imin + 1];
        b12d[imin + 1] = u1c * b12d[imin + 1];
        let (u2c, u2s) = (u2cs[imin], u2sn[imin]);
        temp = u2c * b21e[imin] + u2s * b21d[imin + 1];
        b21d[imin + 1] = u2c * b21d[imin + 1] - u2s * b21e[imin];
        b21e[imin] = temp;
        if imax > imin + 1 {
            b21bulge = u2s * b21e[imin + 1];
            b21e[imin + 1] = u2c * b21e[imin + 1];
        }
        temp = u2c * b22d[imin] + u2s * b22e[imin];
        b22e[imin] = u2c * b22e[imin] - u2s * b22d[imin];
        b22d[imin] = temp;
        let mut b22bulge = u2s * b22d[imin + 1];
        b22d[imin + 1] = u2c * b22d[imin + 1];

        // --- Inner loop i = imin+1 .. imax-1 ---
        for i in (imin + 1)..imax {
            let x1 = theta[i - 1].sin() * b11e[i - 1] + theta[i - 1].cos() * b21e[i - 1];
            let x2 = theta[i - 1].sin() * b11bulge + theta[i - 1].cos() * b21bulge;
            let y1 = theta[i - 1].sin() * b12d[i - 1] + theta[i - 1].cos() * b22d[i - 1];
            let y2 = theta[i - 1].sin() * b12bulge + theta[i - 1].cos() * b22bulge;
            phi[i - 1] = (x1 * x1 + x2 * x2).sqrt().atan2((y1 * y1 + y2 * y2).sqrt());

            // V1T rotation at i with LAPACK dbbcsd RESTART branches, then negate.
            let restart11 = b11e[i - 1] * b11e[i - 1] + b11bulge * b11bulge
                <= (thresh * b11d[i - 1].abs().max(b11d[i].abs()).max(unfl)).powi(2);
            let restart21 = b21e[i - 1] * b21e[i - 1] + b21bulge * b21bulge
                <= (thresh * b21d[i - 1].abs().max(b21d[i].abs()).max(unfl)).powi(2);
            if !restart11 && !restart21 {
                let (sn, cs, _r) = dlartgp(x2, x1);
                v1tsn[i] = sn;
                v1tcs[i] = cs;
            } else if !restart11 && restart21 {
                let (sn, cs, _r) = dlartgp(b11bulge, b11e[i - 1]);
                v1tsn[i] = sn;
                v1tcs[i] = cs;
            } else if restart11 && !restart21 {
                let (sn, cs, _r) = dlartgp(b21bulge, b21e[i - 1]);
                v1tsn[i] = sn;
                v1tcs[i] = cs;
            } else if mu <= nu {
                let (cc, ss) = dlartgs(b11d[i], b11e[i], mu);
                v1tcs[i] = cc;
                v1tsn[i] = ss;
            } else {
                let (cc, ss) = dlartgs(b21d[i], b21e[i], nu);
                v1tcs[i] = cc;
                v1tsn[i] = ss;
            }
            v1tcs[i] = -v1tcs[i];
            v1tsn[i] = -v1tsn[i];
            // V2T rotation at i-1 with RESTART branches (not negated).
            let restart12 = b12d[i - 1] * b12d[i - 1] + b12bulge * b12bulge
                <= (thresh * b12e[i - 1].abs().max(b12d[i].abs()).max(unfl)).powi(2);
            let restart22 = b22d[i - 1] * b22d[i - 1] + b22bulge * b22bulge
                <= (thresh * b22e[i - 1].abs().max(b22d[i].abs()).max(unfl)).powi(2);
            if !restart12 && !restart22 {
                let (sn2, cs2, _r2) = dlartgp(y2, y1);
                v2tsn[i - 1] = sn2;
                v2tcs[i - 1] = cs2;
            } else if !restart12 && restart22 {
                let (sn2, cs2, _r2) = dlartgp(b12bulge, b12d[i - 1]);
                v2tsn[i - 1] = sn2;
                v2tcs[i - 1] = cs2;
            } else if restart12 && !restart22 {
                let (sn2, cs2, _r2) = dlartgp(b22bulge, b22d[i - 1]);
                v2tsn[i - 1] = sn2;
                v2tcs[i - 1] = cs2;
            } else if nu < mu {
                let (cc, ss) = dlartgs(b12e[i - 1], b12d[i], nu);
                v2tcs[i - 1] = cc;
                v2tsn[i - 1] = ss;
            } else {
                let (cc, ss) = dlartgs(b22e[i - 1], b22d[i], mu);
                v2tcs[i - 1] = cc;
                v2tsn[i - 1] = ss;
            }

            let (c, s) = (v1tcs[i], v1tsn[i]);
            temp = c * b11d[i] + s * b11e[i];
            b11e[i] = c * b11e[i] - s * b11d[i];
            b11d[i] = temp;
            b11bulge = s * b11d[i + 1];
            b11d[i + 1] = c * b11d[i + 1];
            temp = c * b21d[i] + s * b21e[i];
            b21e[i] = c * b21e[i] - s * b21d[i];
            b21d[i] = temp;
            b21bulge = s * b21d[i + 1];
            b21d[i + 1] = c * b21d[i + 1];
            let (c2, s2) = (v2tcs[i - 1], v2tsn[i - 1]);
            temp = c2 * b12e[i - 1] + s2 * b12d[i];
            b12d[i] = c2 * b12d[i] - s2 * b12e[i - 1];
            b12e[i - 1] = temp;
            b12bulge = s2 * b12e[i];
            b12e[i] = c2 * b12e[i];
            temp = c2 * b22e[i - 1] + s2 * b22d[i];
            b22d[i] = c2 * b22d[i] - s2 * b22e[i - 1];
            b22e[i - 1] = temp;
            b22bulge = s2 * b22e[i];
            b22e[i] = c2 * b22e[i];

            let x1b = phi[i - 1].cos() * b11d[i] + phi[i - 1].sin() * b12e[i - 1];
            let x2b = phi[i - 1].cos() * b11bulge + phi[i - 1].sin() * b12bulge;
            let y1b = phi[i - 1].cos() * b21d[i] + phi[i - 1].sin() * b22e[i - 1];
            let y2b = phi[i - 1].cos() * b21bulge + phi[i - 1].sin() * b22bulge;
            theta[i] = (y1b * y1b + y2b * y2b)
                .sqrt()
                .atan2((x1b * x1b + x2b * x2b).sqrt());

            // U1 rotation at i with LAPACK dbbcsd RESTART branches.
            let urestart11 = b11d[i] * b11d[i] + b11bulge * b11bulge
                <= (thresh * b11e[i].abs().max(b11d[i + 1].abs()).max(unfl)).powi(2);
            let urestart12 = b12e[i - 1] * b12e[i - 1] + b12bulge * b12bulge
                <= (thresh * b12d[i].abs().max(b12e[i].abs()).max(unfl)).powi(2);
            if !urestart11 && !urestart12 {
                let (sn, cs, _r) = dlartgp(x2b, x1b);
                u1sn[i] = sn;
                u1cs[i] = cs;
            } else if !urestart11 && urestart12 {
                let (sn, cs, _r) = dlartgp(b11bulge, b11d[i]);
                u1sn[i] = sn;
                u1cs[i] = cs;
            } else if urestart11 && !urestart12 {
                let (sn, cs, _r) = dlartgp(b12bulge, b12e[i - 1]);
                u1sn[i] = sn;
                u1cs[i] = cs;
            } else if mu <= nu {
                let (cc, ss) = dlartgs(b11e[i], b11d[i + 1], mu);
                u1cs[i] = cc;
                u1sn[i] = ss;
            } else {
                let (cc, ss) = dlartgs(b12d[i], b12e[i], nu);
                u1cs[i] = cc;
                u1sn[i] = ss;
            }
            // U2 rotation at i with RESTART branches, then negate.
            let urestart21 = b21d[i] * b21d[i] + b21bulge * b21bulge
                <= (thresh * b21e[i].abs().max(b21d[i + 1].abs()).max(unfl)).powi(2);
            let urestart22 = b22e[i - 1] * b22e[i - 1] + b22bulge * b22bulge
                <= (thresh * b22d[i].abs().max(b22e[i].abs()).max(unfl)).powi(2);
            if !urestart21 && !urestart22 {
                let (sn2, cs2, _r2) = dlartgp(y2b, y1b);
                u2sn[i] = sn2;
                u2cs[i] = cs2;
            } else if !urestart21 && urestart22 {
                let (sn2, cs2, _r2) = dlartgp(b21bulge, b21d[i]);
                u2sn[i] = sn2;
                u2cs[i] = cs2;
            } else if urestart21 && !urestart22 {
                let (sn2, cs2, _r2) = dlartgp(b22bulge, b22e[i - 1]);
                u2sn[i] = sn2;
                u2cs[i] = cs2;
            } else if nu < mu {
                let (cc, ss) = dlartgs(b21e[i], b21d[i + 1], nu);
                u2cs[i] = cc;
                u2sn[i] = ss;
            } else {
                let (cc, ss) = dlartgs(b22d[i], b22e[i], mu);
                u2cs[i] = cc;
                u2sn[i] = ss;
            }
            u2cs[i] = -u2cs[i];
            u2sn[i] = -u2sn[i];

            let (u1c, u1s) = (u1cs[i], u1sn[i]);
            temp = u1c * b11e[i] + u1s * b11d[i + 1];
            b11d[i + 1] = u1c * b11d[i + 1] - u1s * b11e[i];
            b11e[i] = temp;
            if i < imax - 1 {
                b11bulge = u1s * b11e[i + 1];
                b11e[i + 1] = u1c * b11e[i + 1];
            }
            let (u2c, u2s) = (u2cs[i], u2sn[i]);
            temp = u2c * b21e[i] + u2s * b21d[i + 1];
            b21d[i + 1] = u2c * b21d[i + 1] - u2s * b21e[i];
            b21e[i] = temp;
            if i < imax - 1 {
                b21bulge = u2s * b21e[i + 1];
                b21e[i + 1] = u2c * b21e[i + 1];
            }
            temp = u1c * b12d[i] + u1s * b12e[i];
            b12e[i] = u1c * b12e[i] - u1s * b12d[i];
            b12d[i] = temp;
            b12bulge = u1s * b12d[i + 1];
            b12d[i + 1] = u1c * b12d[i + 1];
            temp = u2c * b22d[i] + u2s * b22e[i];
            b22e[i] = u2c * b22e[i] - u2s * b22d[i];
            b22d[i] = temp;
            b22bulge = u2s * b22d[i + 1];
            b22d[i + 1] = u2c * b22d[i + 1];
        }

        // --- Final V2T rotation at imax-1 ---
        let x1 = theta[imax - 1].sin() * b11e[imax - 1] + theta[imax - 1].cos() * b21e[imax - 1];
        let y1 = theta[imax - 1].sin() * b12d[imax - 1] + theta[imax - 1].cos() * b22d[imax - 1];
        let y2 = theta[imax - 1].sin() * b12bulge + theta[imax - 1].cos() * b22bulge;
        phi[imax - 1] = x1.abs().atan2((y1 * y1 + y2 * y2).sqrt());
        // Post-loop V2T at imax-1 with LAPACK dbbcsd RESTART branches.
        let prestart12 = b12d[imax - 1] * b12d[imax - 1] + b12bulge * b12bulge
            <= (thresh * b12e[imax - 1].abs().max(unfl)).powi(2);
        let prestart22 = b22d[imax - 1] * b22d[imax - 1] + b22bulge * b22bulge
            <= (thresh * b22e[imax - 1].abs().max(unfl)).powi(2);
        if !prestart12 && !prestart22 {
            let (sn, cs, _r) = dlartgp(y2, y1);
            v2tsn[imax - 1] = sn;
            v2tcs[imax - 1] = cs;
        } else if !prestart12 && prestart22 {
            let (sn, cs, _r) = dlartgp(b12bulge, b12d[imax - 1]);
            v2tsn[imax - 1] = sn;
            v2tcs[imax - 1] = cs;
        } else if prestart12 && !prestart22 {
            let (sn, cs, _r) = dlartgp(b22bulge, b22d[imax - 1]);
            v2tsn[imax - 1] = sn;
            v2tcs[imax - 1] = cs;
        } else if nu < mu {
            let (cc, ss) = dlartgs(b12e[imax - 1], b12d[imax], nu);
            v2tcs[imax - 1] = cc;
            v2tsn[imax - 1] = ss;
        } else {
            let (cc, ss) = dlartgs(b22e[imax - 1], b22d[imax], mu);
            v2tcs[imax - 1] = cc;
            v2tsn[imax - 1] = ss;
        }
        let (c2, s2) = (v2tcs[imax - 1], v2tsn[imax - 1]);
        temp = c2 * b12e[imax - 1] + s2 * b12d[imax];
        b12d[imax] = c2 * b12d[imax] - s2 * b12e[imax - 1];
        b12e[imax - 1] = temp;
        temp = c2 * b22e[imax - 1] + s2 * b22d[imax];
        b22d[imax] = c2 * b22d[imax] - s2 * b22e[imax - 1];
        b22e[imax - 1] = temp;

        // --- Apply accumulated rotations to U1,U2 (columns) and V1T,V2T (rows) ---
        for j in imin..imax {
            rot_cols(&mut u1, j, u1cs[j], u1sn[j]);
            rot_cols(&mut u2, j, u2cs[j], u2sn[j]);
            rot_rows(&mut v1t, j, v1tcs[j], v1tsn[j]);
            rot_rows(&mut v2t, j, v2tcs[j], v2tsn[j]);
        }

        // --- Sign cleanup ---
        if b11e[imax - 1] + b21e[imax - 1] > 0.0 {
            b11d[imax] = -b11d[imax];
            b21d[imax] = -b21d[imax];
            // LAPACK negates ROW imax of V1T (DSCAL on V1T(IMAX,1), stride LDV1T),
            // mirroring the V2T row-negation below — NOT a column scale.
            for c in 0..q {
                v1t[imax][c] = -v1t[imax][c];
            }
        }
        let x1f = phi[imax - 1].cos() * b11d[imax] + phi[imax - 1].sin() * b12e[imax - 1];
        let y1f = phi[imax - 1].cos() * b21d[imax] + phi[imax - 1].sin() * b22e[imax - 1];
        theta[imax] = y1f.abs().atan2(x1f.abs());
        if b11d[imax] + b12e[imax - 1] < 0.0 {
            b12d[imax] = -b12d[imax];
            for r in u1.iter_mut() {
                r[imax] = -r[imax];
            }
        }
        if b21d[imax] + b22e[imax - 1] > 0.0 {
            b22d[imax] = -b22d[imax];
            for r in u2.iter_mut() {
                r[imax] = -r[imax];
            }
        }
        if b12d[imax] + b22d[imax] < 0.0 {
            for c in 0..q {
                v2t[imax][c] = -v2t[imax][c];
            }
        }

        // --- theta/phi cleanup ---
        for t in theta.iter_mut().take(imax + 1).skip(imin) {
            if *t < thresh {
                *t = 0.0;
            } else if *t > pi2 - thresh {
                *t = pi2;
            }
        }
        for p in phi.iter_mut().take(imax).skip(imin) {
            if *p < thresh {
                *p = 0.0;
            } else if *p > pi2 - thresh {
                *p = pi2;
            }
        }

        // --- Update imin/imax ---
        if imax > 0 {
            while imax > 0 && phi[imax - 1] == 0.0 {
                imax -= 1;
            }
        }
        if imin > imax.saturating_sub(1) {
            imin = imax.saturating_sub(1);
        }
        if imin > 0 {
            while imin > 0 && phi[imin - 1] != 0.0 {
                imin -= 1;
            }
        }
    }

    // --- Sort theta ascending, swapping U1/U2 columns and V1T/V2T rows ---
    for i in 0..q {
        let mut mini = i;
        let mut tmin = theta[i];
        for j in i + 1..q {
            if theta[j] < tmin {
                mini = j;
                tmin = theta[j];
            }
        }
        if mini != i {
            theta.swap(mini, i);
            for r in u1.iter_mut() {
                r.swap(i, mini);
            }
            for r in u2.iter_mut() {
                r.swap(i, mini);
            }
            v1t.swap(i, mini);
            v2t.swap(i, mini);
        }
    }

    BbcsdResult {
        theta,
        u1,
        u2,
        v1t,
        v2t,
    }
}

/// Balanced-square (`p = q = m/2`) cosine-sine decomposition factors via
/// `dorbdb` (band-to-bidiagonal reduction) + `dbbcsd` (implicit-QR CS sweep),
/// the exact LAPACK `dorcsd` pipeline. Returns `(u, vh, theta)` where `u`/`vh`
/// are the `2q×2q` orthogonal CS factors and `theta` are the CS angles (sorted
/// ascending, aligned with the `u`/`vh` columns).
///
/// `u`/`vh` are the sign-gauge factors emitted by the `dbbcsd` sweep. They
/// match SciPy's reference-LAPACK signs for pinned low-order golden fixtures;
/// larger cases are valid CSD factors, with sign choices determined by
/// near-zero cleanup roundoff in the LAPACK-style sweep.
pub(crate) fn cossin_factors_balanced(
    x: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>), crate::LinalgError> {
    let q = x.len() / 2;
    let mut x11: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
    let mut x12: Vec<Vec<f64>> = x[..q].iter().map(|r| r[q..].to_vec()).collect();
    let mut x21: Vec<Vec<f64>> = x[q..].iter().map(|r| r[..q].to_vec()).collect();
    let mut x22: Vec<Vec<f64>> = x[q..].iter().map(|r| r[q..].to_vec()).collect();
    let res = dorbdb_balanced(&mut x11, &mut x12, &mut x21, &mut x22, q);
    let p1 = build_from_col_reflectors(&x11, &res.taup1, q);
    let p2 = build_from_col_reflectors(&x21, &res.taup2, q);
    let q1 = build_from_row_reflectors(&x11, &res.tauq1, q, 1);
    let q2 = build_from_row_reflectors(&x12, &res.tauq2, q, 0);
    // Diagonalize the bidiagonal-block form (defined by the dorbdb angles
    // theta/phi) via the LAPACK `dbbcsd` implicit-QR sweep. This yields the CS
    // singular-vector blocks with LAPACK's exact per-vector sign convention —
    // the SVD-of-B11 alternative reconstructs a valid CSD but with SVD-derived
    // signs that do not match scipy. `dbbcsd` accumulates U1b,U2b (q×q column
    // rotations from identity) and V1ᵀb,V2ᵀb (row rotations); recover the
    // right-vector blocks Vb1 = (V1ᵀb)ᵀ, Vb2 = (V2ᵀb)ᵀ.
    let bb = dbbcsd_balanced(&res.theta, &res.phi, q);
    let ub1 = bb.u1;
    let ub2 = bb.u2;
    let vb1: Vec<Vec<f64>> = (0..q)
        .map(|i| (0..q).map(|k| bb.v1t[k][i]).collect())
        .collect();
    let vb2: Vec<Vec<f64>> = (0..q)
        .map(|i| (0..q).map(|k| bb.v2t[k][i]).collect())
        .collect();
    let u1 = mm(&p1, &ub1);
    let u2 = mm(&p2, &ub2);
    let v1 = mm(&q1, &vb1);
    let v2 = mm(&q2, &vb2);
    let m = 2 * q;
    let mut u = vec![vec![0.0; m]; m];
    let mut vh = vec![vec![0.0; m]; m];
    for i in 0..q {
        for j in 0..q {
            u[i][j] = u1[i][j];
            u[q + i][q + j] = u2[i][j];
            // vh = Vᵀ ; V = diag(V1,V2)
            vh[j][i] = v1[i][j];
            vh[q + j][q + i] = v2[i][j];
        }
    }
    Ok((u, vh, bb.theta))
}

/// Cosine-sine decomposition of a `2q×2q` orthogonal matrix `x` partitioned into
/// equal `q×q` blocks (the balanced square case `p = q = m/2`), returning
/// `(u, cs, vh)` with `x = u · cs · vh`.
///
/// `cs` is the cosine-sine middle factor, and `u = diag(U1, U2)` /
/// `vh = diag(V1ᵀ, V2ᵀ)` are the orthogonal CS factors. The `cs` factor matches
/// SciPy's balanced-square construction; `u`/`vh` use the LAPACK-style sign
/// gauge produced by the `dorbdb` + `dbbcsd` sweep.
pub fn cossin(
    x: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>), crate::LinalgError> {
    let m = x.len();
    let q = m / 2;
    if m == 0 || m % 2 != 0 || x.iter().any(|r| r.len() != m) {
        return Err(crate::LinalgError::InvalidArgument {
            detail: "cossin: x must be a non-empty 2q×2q (even-order square) matrix".to_string(),
        });
    }
    let (u, vh, theta) = cossin_factors_balanced(x)?;
    let cs = build_cs_matrix(&theta, q, q, m);
    Ok((u, cs, vh))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Byte-for-byte parity against `scipy.linalg.cossin(x, 2, 2)` (SciPy 1.17.1,
    // reference LAPACK). `x` is `scipy.linalg.qr` of a fixed integer matrix; the
    // expected `u`/`cs`/`vh` are captured directly from SciPy. This pins the exact
    // LAPACK `dorbdb`+`dbbcsd` sign convention, not merely a valid CSD.
    #[test]
    fn cossin_matches_scipy_golden_q2() {
        let x = vec![
            vec![
                -5.34522483824848571e-01,
                1.83040856991163514e-01,
                -7.74944674321111249e-01,
                -2.83271090475186726e-01,
            ],
            vec![
                -2.67261241912424397e-01,
                -8.69444070708026651e-01,
                1.23948035531193074e-01,
                -3.96579526665261417e-01,
            ],
            vec![
                8.01783725737273079e-01,
                -1.67787452241899893e-01,
                -4.75313771037009780e-01,
                -3.21040569205211623e-01,
            ],
            vec![
                -0.0,
                -4.27095332979381459e-01,
                -3.97711522704436826e-01,
                8.12043792695535283e-01,
            ],
        ];
        let expect_u = vec![
            vec![3.20020240285562863e-02, -9.99487804056695461e-01, 0.0, 0.0],
            vec![-9.99487804056695572e-01, -3.20020240285562307e-02, 0.0, 0.0],
            vec![0.0, 0.0, 1.42181254653794664e-01, 9.89840639105645570e-01],
            vec![0.0, 0.0, -9.89840639105645570e-01, 1.42181254653794664e-01],
        ];
        let expect_cs = vec![
            vec![9.09880781320930754e-01, 0.0, -4.14869815463613290e-01, -0.0],
            vec![0.0, 5.64532344815665055e-01, -0.0, -8.25410947139015749e-01],
            vec![4.14869815463613290e-01, 0.0, 9.09880781320930754e-01, 0.0],
            vec![0.0, 8.25410947139015749e-01, 0.0, 5.64532344815665055e-01],
        ];
        let expect_vh = vec![
            vec![2.74781658817301766e-01, 9.61506651031396764e-01, 0.0, 0.0],
            vec![9.61506651031396764e-01, -2.74781658817301766e-01, 0.0, 0.0],
            vec![0.0, 0.0, 3.58387962673269223e-01, -9.33572743931025184e-01],
            vec![0.0, 0.0, -9.33572743931025295e-01, -3.58387962673269223e-01],
        ];
        let (u, cs, vh) = cossin(&x).unwrap();
        for (lbl, got, want) in [
            ("u", &u, &expect_u),
            ("cs", &cs, &expect_cs),
            ("vh", &vh, &expect_vh),
        ] {
            let mut err = 0.0f64;
            for i in 0..4 {
                for j in 0..4 {
                    err = err.max((got[i][j] - want[i][j]).abs());
                }
            }
            assert!(err < 1e-12, "{lbl} differs from scipy by {err:.3e}");
        }
    }

    #[test]
    fn cossin_angles_rejects_invalid_shapes_and_partitions() {
        let x = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        for (p, q) in [(0, 1), (1, 0), (2, 1), (1, 2)] {
            let err = cossin_angles(&x, p, q).unwrap_err();
            assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));
        }

        let empty: Vec<Vec<f64>> = Vec::new();
        let err = cossin_angles(&empty, 1, 1).unwrap_err();
        assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));

        let ragged = vec![vec![1.0, 0.0], vec![0.0]];
        let err = cossin_angles(&ragged, 1, 1).unwrap_err();
        assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));
    }

    #[test]
    fn cossin_rejects_invalid_shapes() {
        let empty: Vec<Vec<f64>> = Vec::new();
        let err = cossin(&empty).unwrap_err();
        assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));

        let ragged = vec![vec![1.0, 0.0], vec![0.0]];
        let err = cossin(&ragged).unwrap_err();
        assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));

        let odd = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let err = cossin(&odd).unwrap_err();
        assert!(matches!(err, crate::LinalgError::InvalidArgument { .. }));
    }

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, k, p) = (a.len(), b.len(), b[0].len());
        let mut c = vec![vec![0.0; p]; n];
        for i in 0..n {
            for l in 0..k {
                let ail = a[i][l];
                for j in 0..p {
                    c[i][j] += ail * b[l][j];
                }
            }
        }
        c
    }

    fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, m) = (a.len(), a[0].len());
        let mut t = vec![vec![0.0; n]; m];
        for i in 0..n {
            for j in 0..m {
                t[j][i] = a[i][j];
            }
        }
        t
    }

    fn orthogonal(m: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut a = vec![vec![0.0; m]; m];
        for row in a.iter_mut() {
            for v in row.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
            }
        }
        crate::qr(&a, crate::DecompOptions::default()).unwrap().q
    }

    #[test]
    fn dorbdb_reconstructs_balanced() {
        for &q in &[2usize, 3, 4, 5, 6] {
            for seed in 0..4u64 {
                let m = 2 * q;
                let x = orthogonal(m, seed + q as u64 * 100);
                let mut x11: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
                let mut x12: Vec<Vec<f64>> = x[..q].iter().map(|r| r[q..].to_vec()).collect();
                let mut x21: Vec<Vec<f64>> = x[q..].iter().map(|r| r[..q].to_vec()).collect();
                let mut x22: Vec<Vec<f64>> = x[q..].iter().map(|r| r[q..].to_vec()).collect();

                let res = dorbdb_balanced(&mut x11, &mut x12, &mut x21, &mut x22, q);
                let p1 = build_from_col_reflectors(&x11, &res.taup1, q);
                let p2 = build_from_col_reflectors(&x21, &res.taup2, q);
                let q1 = build_from_row_reflectors(&x11, &res.tauq1, q, 1);
                let q2 = build_from_row_reflectors(&x12, &res.tauq2, q, 0);
                let bigb = build_bigb(&res.theta, &res.phi, q);

                // diag(P1,P2) · BigB · diag(Q1,Q2)ᵀ
                let mut big_p = vec![vec![0.0; m]; m];
                let mut big_q = vec![vec![0.0; m]; m];
                for i in 0..q {
                    for j in 0..q {
                        big_p[i][j] = p1[i][j];
                        big_p[q + i][q + j] = p2[i][j];
                        big_q[i][j] = q1[i][j];
                        big_q[q + i][q + j] = q2[i][j];
                    }
                }
                let bqt = transpose(&big_q);
                let recon = matmul(&matmul(&big_p, &bigb), &bqt);
                let mut maxerr = 0.0f64;
                for i in 0..m {
                    for j in 0..m {
                        maxerr = maxerr.max((recon[i][j] - x[i][j]).abs());
                    }
                }
                assert!(
                    maxerr < 1e-10,
                    "q={q} seed={seed}: reconstruction error {maxerr:.3e}"
                );
            }
        }
    }

    // Independent 2×2 SVD singular values of [[f,g],[0,h]].
    fn ref_sv2(f: f64, g: f64, h: f64) -> (f64, f64) {
        // eigenvalues of MᵀM = [[f², fg], [fg, g²+h²]]
        let a = f * f;
        let b = f * g;
        let d = g * g + h * h;
        let tr = a + d;
        let det = a * d - b * b;
        let disc = ((tr * tr - 4.0 * det).max(0.0)).sqrt();
        let l1 = (tr + disc) / 2.0;
        let l2 = (tr - disc) / 2.0;
        (l2.max(0.0).sqrt(), l1.max(0.0).sqrt())
    }

    #[test]
    fn dlas2_matches_reference() {
        let cases = [
            (3.0, 1.0, 2.0),
            (0.0, 5.0, 0.0),
            (-2.0, 0.3, 4.0),
            (1e-8, 2.0, 1e-9),
            (7.0, -0.0, 0.5),
            (0.1, 0.2, 0.3),
        ];
        for &(f, g, h) in &cases {
            let (smn, smx) = dlas2(f, g, h);
            let (rmn, rmx) = ref_sv2(f, g, h);
            assert!(
                (smn - rmn).abs() < 1e-12 * (1.0 + rmn),
                "ssmin {smn} vs {rmn}"
            );
            assert!(
                (smx - rmx).abs() < 1e-12 * (1.0 + rmx),
                "ssmax {smx} vs {rmx}"
            );
        }
    }

    #[test]
    fn dlartgp_rotation_property() {
        for &(f, g) in &[
            (3.0, 4.0),
            (-1.0, 2.0),
            (5.0, 0.0),
            (0.0, -3.0),
            (-2.0, -2.0),
        ] {
            let (cs, sn, r) = dlartgp(f, g);
            assert!(r >= 0.0, "r must be ≥ 0");
            assert!((cs * cs + sn * sn - 1.0).abs() < 1e-14);
            assert!((cs * f + sn * g - r).abs() < 1e-12);
            assert!((-sn * f + cs * g).abs() < 1e-12);
        }
    }

    #[test]
    fn dlasr_matches_explicit_rotations() {
        // side='R': A·Pᵀ where P = P(n-2)...P(0), each a (k,k+1) column rotation.
        let n = 5usize;
        let a0 = orthogonal(n, 7);
        let c = vec![0.6, 0.8, -0.5, 0.96];
        let s = vec![0.8, -0.6, 0.866_025_403_784, 0.28];
        // Reference (right): for each k in 0..n-1, rotate columns k,k+1.
        let mut aref = a0.clone();
        for k in 0..n - 1 {
            let (ct, st) = (c[k], s[k]);
            for row in aref.iter_mut() {
                let t = row[k + 1];
                row[k + 1] = ct * t - st * row[k];
                row[k] = st * t + ct * row[k];
            }
        }
        let mut ar = a0.clone();
        dlasr_var_fwd('R', &c, &s, &mut ar);
        let mut maxd = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                maxd = maxd.max((ar[i][j] - aref[i][j]).abs());
            }
        }
        assert!(maxd < 1e-14, "DLASR 'R' mismatch {maxd:.2e}");
        // side='L' on rows, reference.
        let mut lref = a0.clone();
        for k in 0..n - 1 {
            let (ct, st) = (c[k], s[k]);
            for i in 0..n {
                let t = lref[k + 1][i];
                lref[k + 1][i] = ct * t - st * lref[k][i];
                lref[k][i] = st * t + ct * lref[k][i];
            }
        }
        let mut al = a0.clone();
        dlasr_var_fwd('L', &c, &s, &mut al);
        let mut maxl = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                maxl = maxl.max((al[i][j] - lref[i][j]).abs());
            }
        }
        assert!(maxl < 1e-14, "DLASR 'L' mismatch {maxl:.2e}");
    }

    #[test]
    fn dlartgs_orthonormal() {
        for &(x, y, sig) in &[
            (2.0, 1.0, 0.5),
            (-3.0, 2.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.4, 0.7, 0.3),
        ] {
            let (cs, sn) = dlartgs(x, y, sig);
            assert!(
                (cs * cs + sn * sn - 1.0).abs() < 1e-13,
                "not orthonormal for ({x},{y},{sig})"
            );
        }
    }

    #[test]
    fn dbbcsd_sweep_converges_theta() {
        // The implicit-QR sweep diagonalizes the bidiagonal-block form to the
        // correct CS angles (same as the singular values of X11 via cs_angles).
        for &q in &[2usize, 3, 4, 5, 6] {
            for seed in 0..4u64 {
                let m = 2 * q;
                let x = orthogonal(m, seed + q as u64 * 17);
                let mut x11: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
                let mut x12: Vec<Vec<f64>> = x[..q].iter().map(|r| r[q..].to_vec()).collect();
                let mut x21: Vec<Vec<f64>> = x[q..].iter().map(|r| r[..q].to_vec()).collect();
                let mut x22: Vec<Vec<f64>> = x[q..].iter().map(|r| r[q..].to_vec()).collect();
                let res = dorbdb_balanced(&mut x11, &mut x12, &mut x21, &mut x22, q);
                let bb = dbbcsd_balanced(&res.theta, &res.phi, q);
                // Reference: arccos of singular values of X11 (ascending).
                let x11ref: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
                let mut want = cs_angles(&x11ref, q, q, m).unwrap();
                want.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut got = bb.theta.clone();
                got.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut maxerr = 0.0f64;
                for i in 0..q {
                    maxerr = maxerr.max((got[i] - want[i]).abs());
                }
                assert!(
                    maxerr < 1e-9,
                    "q={q} seed={seed}: dbbcsd theta error {maxerr:.3e}"
                );
            }
        }
    }

    #[test]
    fn cossin_public_valid_decomposition() {
        for &q in &[2usize, 3, 4, 5, 6] {
            for seed in 0..4u64 {
                let m = 2 * q;
                let x = orthogonal(m, seed + q as u64 * 23);
                let (u, cs, vh) = cossin(&x).unwrap();
                // x == u · cs · vh
                let recon = matmul(&matmul(&u, &cs), &vh);
                let mut rerr = 0.0f64;
                for i in 0..m {
                    for j in 0..m {
                        rerr = rerr.max((recon[i][j] - x[i][j]).abs());
                    }
                }
                assert!(rerr < 1e-9, "q={q} seed={seed}: recon err {rerr:.3e}");
                // cs == build_cs_matrix(cs_angles(x11))
                let x11: Vec<Vec<f64>> = x[..q].iter().map(|r| r[..q].to_vec()).collect();
                let cs_ref = build_cs_matrix(&cs_angles(&x11, q, q, m).unwrap(), q, q, m);
                for i in 0..m {
                    for j in 0..m {
                        assert!((cs[i][j] - cs_ref[i][j]).abs() < 1e-12);
                    }
                }
                // u, vh orthogonal
                for a in [&u, &vh] {
                    let g = matmul(&transpose(a), a);
                    let mut oe = 0.0f64;
                    for i in 0..m {
                        for j in 0..m {
                            oe = oe.max((g[i][j] - if i == j { 1.0 } else { 0.0 }).abs());
                        }
                    }
                    assert!(oe < 1e-9, "q={q} seed={seed}: not orthogonal {oe:.3e}");
                }
            }
        }
    }

    #[test]
    fn cossin_factors_valid_csd() {
        for &q in &[2usize, 3, 4, 5, 6] {
            for seed in 0..4u64 {
                let m = 2 * q;
                let x = orthogonal(m, seed + q as u64 * 37);
                let (u, vh, theta) = cossin_factors_balanced(&x).unwrap();
                let cs = build_cs_matrix(&theta, q, q, m);
                // u · cs · vh == x
                let recon = matmul(&matmul(&u, &cs), &vh);
                let mut rerr = 0.0f64;
                for i in 0..m {
                    for j in 0..m {
                        rerr = rerr.max((recon[i][j] - x[i][j]).abs());
                    }
                }
                assert!(rerr < 1e-9, "q={q} seed={seed}: u·cs·vh err {rerr:.3e}");
                // u, vh orthogonal
                for (lbl, a) in [("u", &u), ("vh", &vh)] {
                    let ata = matmul(&transpose(a), a);
                    let mut oerr = 0.0f64;
                    for i in 0..m {
                        for j in 0..m {
                            oerr = oerr.max((ata[i][j] - if i == j { 1.0 } else { 0.0 }).abs());
                        }
                    }
                    assert!(
                        oerr < 1e-9,
                        "q={q} seed={seed}: {lbl} not orthogonal {oerr:.3e}"
                    );
                }
            }
        }
    }

    // Reconstruct H = I - tau v vᵀ and apply to x; check H·x = beta·e1, beta≥0.
    fn check_reflector(x: &[f64]) {
        let n = x.len();
        let alpha = x[0];
        let mut tail: Vec<f64> = x[1..].to_vec();
        let (beta, tau) = dlarfgp(alpha, &mut tail);
        assert!(beta >= 0.0, "beta must be non-negative, got {beta}");

        // v = [1, tail...]; H·x via the application routine.
        let mut col: Vec<Vec<f64>> = x.iter().map(|&xi| vec![xi]).collect();
        dlarf_left(&mut col, &tail, tau);
        let hx: Vec<f64> = col.iter().map(|r| r[0]).collect();
        assert!(
            (hx[0] - beta).abs() < 1e-12,
            "H·x[0] = {} vs beta {beta}",
            hx[0]
        );
        for (i, &v) in hx.iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-12, "H·x[{i}] = {v}, expected 0");
        }
        // beta = ±‖x‖.
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((beta - nrm).abs() < 1e-12, "beta {beta} vs ‖x‖ {nrm}");
        let _ = n;
    }

    #[test]
    fn dlarfgp_reflects_to_positive_beta() {
        check_reflector(&[3.0, 4.0]);
        check_reflector(&[-3.0, 4.0]);
        check_reflector(&[3.0, -4.0, 12.0]);
        check_reflector(&[-1.0, -2.0, -2.0]);
        check_reflector(&[0.0, 3.0, 4.0]);
        check_reflector(&[5.0]); // n=1
        check_reflector(&[-5.0]); // n=1, negative
        check_reflector(&[2.0, 1e-20]); // tail negligible vs alpha
    }

    fn assert_cs(theta: &[f64], p: usize, q: usize, m: usize, want_flat: &[f64]) {
        let cs = build_cs_matrix(theta, p, q, m);
        for i in 0..m {
            for j in 0..m {
                let w = want_flat[i * m + j];
                assert!(
                    (cs[i][j] - w).abs() < 1e-6,
                    "CS[{i}][{j}] = {} vs {w}",
                    cs[i][j]
                );
            }
        }
    }

    #[test]
    fn cossin_angles_matches_scipy_full() {
        // Full 6x6 orthogonal X, p=q=3; scipy.linalg.cossin oracle.
        let x = vec![
            vec![
                -0.87391881375,
                0.167562492975,
                -0.235103506178,
                0.350213063322,
                -0.006202049134,
                0.173860300728,
            ],
            vec![
                0.000460285786,
                -0.628191216821,
                -0.001453845601,
                0.54997211923,
                -0.208189382876,
                -0.509471607163,
            ],
            vec![
                -0.261214975137,
                0.006389522704,
                -0.246757897299,
                -0.471708053971,
                0.418748888032,
                -0.687733474157,
            ],
            vec![
                -0.141882309423,
                -0.492064853719,
                0.399089339788,
                0.00027656113,
                0.687774023242,
                0.324709262791,
            ],
            vec![
                0.023462349781,
                -0.528172950388,
                -0.67836107675,
                -0.326551827636,
                -0.150226127071,
                0.362084405606,
            ],
            vec![
                0.383863726197,
                0.236987962287,
                -0.514187432999,
                0.495718486188,
                0.534478332503,
                0.026319645199,
            ],
        ];
        let (theta, cs) = cossin_angles(&x, 3, 3).unwrap();
        let want_theta = [0.186263718828, 0.906992013555, 1.410204824056];
        for (g, w) in theta.iter().zip(want_theta.iter()) {
            assert!((g - w).abs() < 1e-9, "theta {g} vs {w}");
        }
        let want_cs = [
            0.982703009127,
            0.0,
            0.0,
            -0.185188541364,
            0.0,
            0.0,
            0.0,
            0.616117785876,
            0.0,
            0.0,
            -0.787654031874,
            0.0,
            0.0,
            0.0,
            0.159902126352,
            0.0,
            0.0,
            -0.987132873522,
            0.185188541364,
            0.0,
            0.0,
            0.982703009127,
            0.0,
            0.0,
            0.0,
            0.787654031874,
            0.0,
            0.0,
            0.616117785876,
            0.0,
            0.0,
            0.0,
            0.987132873522,
            0.0,
            0.0,
            0.159902126352,
        ];
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    (cs[r][c] - want_cs[r * 6 + c]).abs() < 1e-9,
                    "cs[{r}][{c}] = {} vs {}",
                    cs[r][c],
                    want_cs[r * 6 + c]
                );
            }
        }
    }

    #[test]
    fn cs_angles_match_scipy() {
        // Case A: m=6, p=q=3, r=3, n11=0.
        let a_x11 = vec![
            vec![-0.5299718695, -0.2604647036, -0.4936740867],
            vec![-0.105673699, -0.0378100654, 0.1484074059],
            vec![0.3840964564, -0.4523877513, -0.6696005085],
        ];
        let a_theta = cs_angles(&a_x11, 3, 3, 6).unwrap();
        let a_want = [0.159776054, 0.8461927778, 1.45411065];
        for (g, w) in a_theta.iter().zip(a_want.iter()) {
            assert!((g - w).abs() < 1e-8, "A theta {g} vs {w}");
        }

        // Case B: m=8, p=q=5, r=3, n11=2 (identity block present).
        let b_x11 = vec![
            vec![
                -0.597715282,
                -0.2284842244,
                6.461e-06,
                -0.6729100837,
                0.2199323067,
            ],
            vec![
                -0.1438553634,
                -0.4268619363,
                -0.5315047497,
                -0.0442671386,
                -0.2683036699,
            ],
            vec![
                -0.2517464206,
                0.5361883302,
                -0.0607574542,
                -0.0670250601,
                -0.7047191734,
            ],
            vec![
                -0.2152718386,
                -0.016432015,
                0.7485130952,
                -0.1263420574,
                -0.2090926243,
            ],
            vec![
                0.0634716335,
                -0.1290424626,
                0.0822154112,
                -0.0422565627,
                -0.1356398819,
            ],
        ];
        let b_theta = cs_angles(&b_x11, 5, 5, 8).unwrap();
        let b_want = [0.4701951603, 1.0653266212, 1.4673215];
        for (g, w) in b_theta.iter().zip(b_want.iter()) {
            assert!((g - w).abs() < 1e-7, "B theta {g} vs {w}");
        }
    }

    #[test]
    fn build_cs_matrix_matches_scipy() {
        // scipy.linalg.cossin CS oracles (separate=False, swap_sign=False).
        assert_cs(
            &[0.27701, 0.340936, 1.327589, 1.483862],
            4,
            4,
            8,
            &[
                0.961877, 0.0, 0.0, 0.0, -0.273481, 0.0, 0.0, 0.0, 0.0, 0.942442, 0.0, 0.0, 0.0,
                -0.334369, 0.0, 0.0, 0.0, 0.0, 0.240817, 0.0, 0.0, 0.0, -0.970571, 0.0, 0.0, 0.0,
                0.0, 0.086825, 0.0, 0.0, 0.0, -0.996224, 0.273481, 0.0, 0.0, 0.0, 0.961877, 0.0,
                0.0, 0.0, 0.0, 0.334369, 0.0, 0.0, 0.0, 0.942442, 0.0, 0.0, 0.0, 0.0, 0.970571,
                0.0, 0.0, 0.0, 0.240817, 0.0, 0.0, 0.0, 0.0, 0.996224, 0.0, 0.0, 0.0, 0.086825,
            ],
        );
        // Unbalanced p<q.
        assert_cs(
            &[0.042939, 0.54894, 0.872127],
            3,
            5,
            8,
            &[
                0.999078, 0.0, 0.0, 0.0, 0.0, -0.042926, 0.0, 0.0, 0.0, 0.853078, 0.0, 0.0, 0.0,
                0.0, -0.521784, 0.0, 0.0, 0.0, 0.643199, 0.0, 0.0, 0.0, 0.0, -0.765699, 0.042926,
                0.0, 0.0, 0.0, 0.0, 0.999078, 0.0, 0.0, 0.0, 0.521784, 0.0, 0.0, 0.0, 0.0,
                0.853078, 0.0, 0.0, 0.0, 0.765699, 0.0, 0.0, 0.0, 0.0, 0.643199, 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
        // Smaller unbalanced p=2,q=3,m=6.
        assert_cs(
            &[0.193469, 0.956576],
            2,
            3,
            6,
            &[
                0.981343, 0.0, 0.0, 0.0, -0.192264, 0.0, 0.0, 0.576321, 0.0, 0.0, 0.0, -0.817223,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.192264, 0.0, 0.0, 0.0, 0.981343, 0.0, 0.0,
                0.817223, 0.0, 0.0, 0.0, 0.576321, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        );
    }

    #[test]
    fn dlarf_left_right_match_explicit_h() {
        // Build H explicitly from (v, tau) and compare to dlarf application.
        let x = [2.0, -1.0, 3.0];
        let mut tail = x[1..].to_vec();
        let (_, tau) = dlarfgp(x[0], &mut tail);
        let v = [1.0, tail[0], tail[1]];
        // Explicit H = I - tau v vᵀ.
        let mut h = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = (if i == j { 1.0 } else { 0.0 }) - tau * v[i] * v[j];
            }
        }
        // Random matrix C.
        let c0 = vec![
            vec![1.0, 2.0, -1.0],
            vec![0.5, -3.0, 2.0],
            vec![4.0, 1.0, 0.0],
        ];
        // Left: H·C.
        let mut left = c0.clone();
        dlarf_left(&mut left, &tail, tau);
        for i in 0..3 {
            for j in 0..3 {
                let expect: f64 = (0..3).map(|k| h[i][k] * c0[k][j]).sum();
                assert!((left[i][j] - expect).abs() < 1e-12, "left[{i}][{j}]");
            }
        }
        // Right: C·H.
        let mut right = c0.clone();
        dlarf_right(&mut right, &tail, tau);
        for i in 0..3 {
            for j in 0..3 {
                let expect: f64 = (0..3).map(|k| c0[i][k] * h[k][j]).sum();
                assert!((right[i][j] - expect).abs() < 1e-12, "right[{i}][{j}]");
            }
        }
    }
}
