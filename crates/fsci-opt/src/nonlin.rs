//! Reusable Jacobian approximations for nonlinear systems
//! -- `scipy.optimize.nonlin`.
//!
//! `root.rs` already has `broyden1`, `broyden2` and `newton_krylov` as FUNCTIONS. What
//! it does not have is the Jacobian approximation as an object the caller owns, which
//! is what `scipy.optimize` exposes as `BroydenFirst`, `BroydenSecond` and
//! `InverseJacobian`, and the difference is not cosmetic.
//!
//! # Why the object is not the same as the function
//!
//! `root.rs`'s `broyden1` stores a DENSE `n x n` inverse Jacobian and updates it in
//! place. That is O(n^2) memory and O(n^2) work per step, which puts a ceiling on the
//! problem size well below where these methods are actually wanted -- the reason to
//! reach for a Broyden method rather than a Newton method is precisely that you cannot
//! afford to form a Jacobian.
//!
//! This module stores the same approximation as `alpha*I + sum_i c_i d_i^T`: O(n*m)
//! memory for `m` accumulated steps, and `m` capped by a rank-reduction policy. The
//! secant condition is identical; only the representation changes.
//!
//! # Own linear algebra
//!
//! Applying the Jacobian (as opposed to its inverse) needs the Sherman-Morrison-Woodbury
//! identity, whose only dense step is an `m x m` solve with `m` the retained rank --
//! small by construction. That solve is a partial-pivoting LU written here in safe Rust.
//! Nothing in this module links a BLAS or LAPACK.

#![allow(clippy::needless_range_loop)]

/// A Jacobian approximation that can be applied and inverted
/// -- `scipy.optimize.nonlin.Jacobian`.
///
/// The four operations are deliberately separate rather than derived from a stored
/// matrix: for a low-rank representation, applying the inverse is CHEAP (a few dot
/// products) while applying the Jacobian itself needs a solve. A caller that only ever
/// takes Newton steps pays only for the cheap direction.
pub trait Jacobian {
    /// `w = J^-1 v` without mutating -- the Newton step direction, the operation a
    /// solver actually wants.
    fn solve_ref(&self, v: &[f64]) -> Vec<f64>;

    /// `w = J^-1 v`, permitted to REPAIR a degenerated representation first.
    ///
    /// Split from `solve_ref` because the two callers want different things. A solver
    /// taking a step wants the recovery; a wrapper like `InverseJacobian` holds only
    /// `&self` and cannot have it. Defaulting to the non-mutating path keeps
    /// implementations that never degenerate free of the distinction.
    fn solve(&mut self, v: &[f64]) -> Vec<f64> {
        self.solve_ref(v)
    }

    /// `w = J v`.
    fn matvec(&self, v: &[f64]) -> Vec<f64>;

    /// `w = J^-T v`.
    fn rsolve(&self, v: &[f64]) -> Vec<f64>;

    /// `w = J^T v`.
    fn rmatvec(&self, v: &[f64]) -> Vec<f64>;

    /// Absorb the step ending at `x` with residual `f`.
    fn update(&mut self, x: &[f64], f: &[f64]);

    /// Prepare for a solve started at `x0` with residual `f0`.
    fn setup(&mut self, x0: &[f64], f0: &[f64]);

    /// Problem dimension.
    fn dimension(&self) -> usize;
}

/// Swaps a Jacobian's forward and inverse operations
/// -- `scipy.optimize.nonlin.InverseJacobian`.
///
/// A preconditioner wants `J^-1` where a matrix-free solver wants `J`; this presents
/// one as the other without copying or re-deriving anything. `solve` and `matvec` trade
/// places, as do `rsolve` and `rmatvec`.
#[derive(Debug, Clone)]
pub struct InverseJacobian<J> {
    /// The wrapped approximation.
    pub jacobian: J,
}

impl<J: Jacobian> InverseJacobian<J> {
    /// Wrap `jacobian`, presenting its inverse.
    pub fn new(jacobian: J) -> Self {
        Self { jacobian }
    }
}

impl<J: Jacobian> Jacobian for InverseJacobian<J> {
    fn solve_ref(&self, v: &[f64]) -> Vec<f64> {
        self.jacobian.matvec(v)
    }
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        // Applying the inverse-of-the-inverse is the original's solve. Only the
        // non-mutating path is reachable here, so a degenerate representation
        // propagates rather than being repaired -- see `Jacobian::solve`.
        self.jacobian.solve_ref(v)
    }
    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        self.jacobian.rmatvec(v)
    }
    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        self.jacobian.rsolve(v)
    }
    fn update(&mut self, x: &[f64], f: &[f64]) {
        self.jacobian.update(x, f);
    }
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.jacobian.setup(x0, f0);
    }
    fn dimension(&self) -> usize {
        self.jacobian.dimension()
    }
}

/// How the accumulated rank is kept bounded -- `reduction_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMethod {
    /// Drop every stored vector once the cap is exceeded. SciPy's default.
    Restart,
    /// Drop the oldest stored vectors until the cap is met.
    Simple,
    /// Keep only the most significant SVD components -- the "Broyden Rank Reduction
    /// Inverse" of Van der Rotten's thesis.
    ///
    /// The other two policies choose by AGE, which is a proxy for relevance and not a
    /// good one: the oldest stored direction may still carry the largest part of the
    /// operator. This one chooses by singular value, so what is discarded is the part
    /// of the update that contributes least, measured rather than guessed.
    ///
    /// `to_retain` defaults to `max_rank - 2`, as in SciPy.
    Svd {
        /// How many components survive a reduction.
        to_retain: Option<usize>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// LowRankMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// `alpha*I + sum_i c_i d_i^T`, held as its factors
/// -- `scipy.optimize.nonlin.LowRankMatrix`.
///
/// # Why it collapses
///
/// Once more than `n` vectors have accumulated the representation is no longer saving
/// anything -- `n` rank-1 terms cost as much as the dense matrix they describe, and the
/// repeated dot products cost MORE. Past that point the matrix is formed explicitly and
/// the factors are dropped. SciPy does the same, and the threshold is exactly `n`.
#[derive(Debug, Clone)]
pub struct LowRankMatrix {
    alpha: f64,
    cs: Vec<Vec<f64>>,
    ds: Vec<Vec<f64>>,
    n: usize,
    /// Set once the representation has been collapsed to a dense `n x n`, row-major.
    collapsed: Option<Vec<f64>>,
}

impl LowRankMatrix {
    /// A pure multiple of the identity, with no rank-1 terms yet.
    #[must_use]
    pub fn new(alpha: f64, n: usize) -> Self {
        Self {
            alpha,
            cs: Vec::new(),
            ds: Vec::new(),
            n,
            collapsed: None,
        }
    }

    /// Dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Number of stored rank-1 terms; zero once collapsed.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.cs.len()
    }

    /// Whether the factored form has been abandoned for a dense one.
    #[must_use]
    pub fn is_collapsed(&self) -> bool {
        self.collapsed.is_some()
    }

    fn low_rank_matvec(v: &[f64], alpha: f64, cs: &[Vec<f64>], ds: &[Vec<f64>]) -> Vec<f64> {
        let mut w: Vec<f64> = v.iter().map(|x| alpha * x).collect();
        for (c, d) in cs.iter().zip(ds) {
            let a: f64 = d.iter().zip(v).map(|(x, y)| x * y).sum();
            for (wi, ci) in w.iter_mut().zip(c) {
                *wi += a * ci;
            }
        }
        w
    }

    /// `w = M^-1 v` via Sherman-Morrison-Woodbury.
    ///
    /// `(alpha*I + C D^T)^-1 = I/alpha - C (alpha*I + D^T C)^-1 D^T / alpha`
    ///
    /// The only dense work is the `m x m` solve, with `m` the number of stored vectors.
    /// That is what makes the inverse affordable at large `n`: the cost is set by the
    /// retained rank, not by the dimension.
    fn low_rank_solve(v: &[f64], alpha: f64, cs: &[Vec<f64>], ds: &[Vec<f64>]) -> Vec<f64> {
        if cs.is_empty() {
            return v.iter().map(|x| x / alpha).collect();
        }
        let m = cs.len();
        // A = alpha*I + D^T C
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            a[i * m + i] = alpha;
            for j in 0..m {
                let dot: f64 = ds[i].iter().zip(&cs[j]).map(|(x, y)| x * y).sum();
                a[i * m + j] += dot;
            }
        }
        let mut q: Vec<f64> = ds
            .iter()
            .map(|d| d.iter().zip(v).map(|(x, y)| x * y).sum::<f64>() / alpha)
            .collect();
        // A singular inner system means the representation itself is degenerate. Rather
        // than inventing an answer, propagate non-finite values: `BroydenFirst::solve`
        // watches for exactly that and rebuilds, which is the recovery SciPy performs.
        q = match lu_solve_in_place(&mut a, &q, m) {
            Some(sol) => sol,
            None => vec![f64::NAN; m],
        };
        let mut w: Vec<f64> = v.iter().map(|x| x / alpha).collect();
        for (c, qc) in cs.iter().zip(&q) {
            for (wi, ci) in w.iter_mut().zip(c) {
                *wi -= qc * ci;
            }
        }
        w
    }

    fn dense_matvec(dense: &[f64], n: usize, v: &[f64], transpose: bool) -> Vec<f64> {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let e = if transpose { dense[j * n + i] } else { dense[i * n + j] };
                        e * v[j]
                    })
                    .sum()
            })
            .collect()
    }

    /// `w = M v`.
    #[must_use]
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        match &self.collapsed {
            Some(dense) => Self::dense_matvec(dense, self.n, v, false),
            None => Self::low_rank_matvec(v, self.alpha, &self.cs, &self.ds),
        }
    }

    /// `w = M^T v`.
    ///
    /// The transpose of `alpha*I + sum c_i d_i^T` is `alpha*I + sum d_i c_i^T`, so the
    /// same kernel serves with the factor lists exchanged.
    #[must_use]
    pub fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        match &self.collapsed {
            Some(dense) => Self::dense_matvec(dense, self.n, v, true),
            None => Self::low_rank_matvec(v, self.alpha, &self.ds, &self.cs),
        }
    }

    /// `w = M^-1 v`.
    #[must_use]
    pub fn solve(&self, v: &[f64]) -> Vec<f64> {
        match &self.collapsed {
            Some(dense) => {
                let mut a = dense.clone();
                lu_solve_in_place(&mut a, v, self.n).unwrap_or_else(|| vec![f64::NAN; self.n])
            }
            None => Self::low_rank_solve(v, self.alpha, &self.cs, &self.ds),
        }
    }

    /// `w = M^-T v`.
    #[must_use]
    pub fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        match &self.collapsed {
            Some(dense) => {
                let mut a = vec![0.0; self.n * self.n];
                for i in 0..self.n {
                    for j in 0..self.n {
                        a[i * self.n + j] = dense[j * self.n + i];
                    }
                }
                lu_solve_in_place(&mut a, v, self.n).unwrap_or_else(|| vec![f64::NAN; self.n])
            }
            None => Self::low_rank_solve(v, self.alpha, &self.ds, &self.cs),
        }
    }

    /// Append the rank-1 term `c d^T`.
    ///
    /// Collapses once the stored count would exceed `n`, past which the factored form
    /// costs more than the matrix it represents.
    pub fn append(&mut self, c: Vec<f64>, d: Vec<f64>) {
        if let Some(dense) = &mut self.collapsed {
            for i in 0..self.n {
                for j in 0..self.n {
                    dense[i * self.n + j] += c[i] * d[j];
                }
            }
            return;
        }
        self.cs.push(c);
        self.ds.push(d);
        if self.cs.len() > self.n {
            self.collapse();
        }
    }

    /// Form the dense matrix, row-major.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f64> {
        if let Some(dense) = &self.collapsed {
            return dense.clone();
        }
        let mut m = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            m[i * self.n + i] = self.alpha;
        }
        for (c, d) in self.cs.iter().zip(&self.ds) {
            for i in 0..self.n {
                for j in 0..self.n {
                    m[i * self.n + j] += c[i] * d[j];
                }
            }
        }
        m
    }

    /// Abandon the factored form for an explicit matrix.
    pub fn collapse(&mut self) {
        if self.collapsed.is_some() {
            return;
        }
        self.collapsed = Some(self.to_dense());
        self.cs.clear();
        self.ds.clear();
    }

    /// Drop ALL stored vectors once `rank` is exceeded.
    ///
    /// Blunt, and SciPy's default: it discards the accumulated curvature entirely and
    /// restarts from `alpha*I`. The next update restores the most recent secant
    /// condition, so the method stays well defined -- it simply forgets.
    pub fn restart_reduce(&mut self, rank: usize) {
        if self.collapsed.is_some() || rank == 0 {
            return;
        }
        if self.cs.len() > rank {
            self.cs.clear();
            self.ds.clear();
        }
    }

    /// Drop the OLDEST stored vectors until at most `rank` remain.
    pub fn simple_reduce(&mut self, rank: usize) {
        if self.collapsed.is_some() || rank == 0 {
            return;
        }
        while self.cs.len() > rank {
            self.cs.remove(0);
            self.ds.remove(0);
        }
    }

    /// Retain only the `to_retain` most significant SVD components
    /// -- `LowRankMatrix.svd_reduce`, the Van der Rotten limited-memory Broyden update.
    ///
    /// # The decomposition is of an m-sized problem, not an n-sized one
    ///
    /// Naively this asks for the SVD of an `n x m` matrix, which for large `n` is
    /// exactly the cost the low-rank representation exists to avoid. It is not needed.
    /// Take the economic QR `D = Q R`; then
    ///
    /// ```text
    ///     C D^T = (C R^T) Q^T
    /// ```
    ///
    /// so the operator is unchanged by replacing `C` with `C R^T` and `D` with `Q`.
    /// The remaining SVD is of `C R^T`, and since `Q` has orthonormal columns, the
    /// singular values wanted are those of an `m x m` problem. Every dense step is
    /// sized by the retained rank.
    ///
    /// # Why the rotation needs no explicit inverse
    ///
    /// SciPy computes `C <- C inv(W^H)` and `D <- D W`. Written with a ONE-SIDED Jacobi
    /// SVD the inverse disappears: the method returns the already-rotated `C V = U S`
    /// directly, and `V` is orthogonal so `inv(V^T) = V`. Both updates become a right
    /// multiplication by `V`, and no inverse of a possibly ill-conditioned factor is
    /// ever formed.
    ///
    /// The columns are then ordered by singular value and truncated, so what is dropped
    /// is the smallest part of the operator in the Frobenius norm.
    pub fn svd_reduce(&mut self, max_rank: usize, to_retain: Option<usize>) {
        if self.collapsed.is_some() || self.cs.is_empty() {
            return;
        }
        // SciPy's bookkeeping: p caps at the dimension, q defaults to p - 2 and is
        // clamped below p so a reduction always removes at least one component.
        let mut p = max_rank;
        let mut q = to_retain.unwrap_or_else(|| p.saturating_sub(2));
        p = p.min(self.cs[0].len());
        q = q.min(p.saturating_sub(1));
        let m = self.cs.len();
        if m < p {
            // Below the cap there is nothing to do; reducing here would discard
            // information the caller still has room for.
            return;
        }
        if q == 0 {
            self.cs.clear();
            self.ds.clear();
            return;
        }

        let (qm, r) = economic_qr(&self.ds);
        // C1 = C R^T, i.e. column j of C1 is sum_k C[:, k] * R[j][k].
        let n = self.cs[0].len();
        let mut c1: Vec<Vec<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let mut col = vec![0.0; n];
            for k in j..m {
                let rjk = r[j][k];
                if rjk != 0.0 {
                    for (ci, ck) in col.iter_mut().zip(&self.cs[k]) {
                        *ci += rjk * ck;
                    }
                }
            }
            c1.push(col);
        }

        let v = one_sided_jacobi(&mut c1);
        // D2 = Q V.
        let mut d2: Vec<Vec<f64>> = Vec::with_capacity(m);
        for j in 0..m {
            let mut col = vec![0.0; n];
            for k in 0..m {
                // `v[j]` is COLUMN j of V, so `v[j][k]` is `V[k][j]` -- the entry that
                // multiplies Q's k-th column when forming `D2 = Q V`. Indexing this the
                // other way round yields `Q V^T`, which is orthogonal, plausible, and
                // silently changes the operator.
                let vkj = v[j][k];
                if vkj != 0.0 {
                    for (di, qk) in col.iter_mut().zip(&qm[k]) {
                        *di += vkj * qk;
                    }
                }
            }
            d2.push(col);
        }

        c1.truncate(q);
        d2.truncate(q);
        self.cs = c1;
        self.ds = d2;
    }
}

/// Economic QR of the matrix whose COLUMNS are `cols`, by modified Gram-Schmidt with
/// one reorthogonalisation pass.
///
/// Returns `(q_cols, r)` with `r[i][j]` the upper-triangular factor, satisfying
/// `A = Q R`. Reorthogonalising once is the standard "twice is enough" remedy: plain
/// modified Gram-Schmidt loses orthogonality in proportion to the condition number, and
/// Broyden directions become nearly dependent exactly when a reduction is due, which is
/// the worst case for it.
///
/// A column that is annihilated by the projection leaves a zero on the diagonal and a
/// zero `Q` column. That is deliberate: `A = Q R` still holds, so the caller's
/// `C R^T Q^T = C A^T` identity survives a rank-deficient input rather than the routine
/// having to reject it.
fn economic_qr(cols: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = cols.len();
    let n = if m == 0 { 0 } else { cols[0].len() };
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut r = vec![vec![0.0; m]; m];

    for j in 0..m {
        let mut v = cols[j].clone();
        for _pass in 0..2 {
            for i in 0..j {
                let proj: f64 = q[i].iter().zip(&v).map(|(a, b)| a * b).sum();
                if proj != 0.0 {
                    r[i][j] += proj;
                    for (vi, qi) in v.iter_mut().zip(&q[i]) {
                        *vi -= proj * qi;
                    }
                }
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j][j] = nrm;
        if nrm > 0.0 && nrm.is_finite() {
            for vi in v.iter_mut() {
                *vi /= nrm;
            }
            q.push(v);
        } else {
            q.push(vec![0.0; n]);
        }
    }
    (q, r)
}

/// One-sided Jacobi SVD of the matrix whose COLUMNS are `a`, rotating `a` IN PLACE.
///
/// On return `a` holds `A V` -- that is, `u_i * s_i` in each column, ordered by
/// descending singular value -- and the returned value holds the columns of the
/// orthogonal `V`, so that `A = U S V^T`.
///
/// One-sided Jacobi is chosen over a bidiagonalisation here for two reasons: it is a
/// few dozen lines of safe Rust with no BLAS behind it, and it computes small singular
/// values to high RELATIVE accuracy, which is the property that matters when the whole
/// purpose is to decide which components are small enough to discard.
fn one_sided_jacobi(a: &mut [Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut v: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut col = vec![0.0; m];
            col[i] = 1.0;
            col
        })
        .collect();
    if m < 2 {
        sort_by_column_norm(a, &mut v);
        return v;
    }

    // Enough sweeps for convergence in practice; Jacobi is quadratically convergent and
    // m is the retained rank, not the dimension. The loop exits early once no rotation
    // exceeds the threshold, so the bound only caps a pathological case.
    let max_sweeps = 30;
    for _sweep in 0..max_sweeps {
        let mut rotated = false;
        for pi in 0..(m - 1) {
            for qi in (pi + 1)..m {
                let alpha: f64 = a[pi].iter().map(|x| x * x).sum();
                let beta: f64 = a[qi].iter().map(|x| x * x).sum();
                let gamma: f64 = a[pi].iter().zip(&a[qi]).map(|(x, y)| x * y).sum();
                if gamma == 0.0 || !gamma.is_finite() {
                    continue;
                }
                // Relative threshold: two columns are orthogonal enough when their
                // inner product is negligible against their own magnitudes. An
                // absolute test would never fire on a large-scale problem and always
                // fire on a small one.
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for k in 0..a[pi].len() {
                    let (x, y) = (a[pi][k], a[qi][k]);
                    a[pi][k] = c * x - sn * y;
                    a[qi][k] = sn * x + c * y;
                }
                for k in 0..m {
                    let (x, y) = (v[pi][k], v[qi][k]);
                    v[pi][k] = c * x - sn * y;
                    v[qi][k] = sn * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    sort_by_column_norm(a, &mut v);
    v
}

/// Order both column sets by descending column norm of `a` -- descending singular
/// value. Jacobi produces no particular order, and the truncation that follows is only
/// meaningful once the smallest components are last.
fn sort_by_column_norm(a: &mut [Vec<f64>], v: &mut [Vec<f64>]) {
    let mut order: Vec<usize> = (0..a.len()).collect();
    let norms: Vec<f64> = a
        .iter()
        .map(|col| col.iter().map(|x| x * x).sum::<f64>())
        .collect();
    order.sort_by(|&i, &j| {
        norms[j]
            .partial_cmp(&norms[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let a_sorted: Vec<Vec<f64>> = order.iter().map(|&i| a[i].clone()).collect();
    let v_sorted: Vec<Vec<f64>> = order.iter().map(|&i| v[i].clone()).collect();
    for (slot, col) in a.iter_mut().zip(a_sorted) {
        *slot = col;
    }
    for (slot, col) in v.iter_mut().zip(v_sorted) {
        *slot = col;
    }
}

/// Partial-pivoting LU solve of a row-major `n x n` system, destroying `a`.
///
/// Returns `None` when the pivot is not finite or vanishes, which is the caller's
/// signal that the representation has degenerated. Own arithmetic, no BLAS.
fn lu_solve_in_place(a: &mut [f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut x = b.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut piv = k;
        let mut best = a[perm[k] * n + k].abs();
        for i in (k + 1)..n {
            let v = a[perm[i] * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if !best.is_finite() || best == 0.0 {
            return None;
        }
        perm.swap(k, piv);
        let pk = perm[k];
        for i in (k + 1)..n {
            let pi = perm[i];
            let factor = a[pi * n + k] / a[pk * n + k];
            a[pi * n + k] = factor;
            for j in (k + 1)..n {
                a[pi * n + j] -= factor * a[pk * n + j];
            }
        }
    }
    // Forward substitution on the permuted right-hand side.
    let mut y = vec![0.0; n];
    for i in 0..n {
        let pi = perm[i];
        let mut s = x[pi];
        for j in 0..i {
            s -= a[pi * n + j] * y[j];
        }
        y[i] = s;
    }
    // Back substitution.
    for i in (0..n).rev() {
        let pi = perm[i];
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= a[pi * n + j] * x[j];
        }
        let d = a[pi * n + i];
        if !d.is_finite() || d == 0.0 {
            return None;
        }
        x[i] = s / d;
    }
    Some(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Broyden
// ─────────────────────────────────────────────────────────────────────────────

/// Which Broyden update the approximation carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroydenVariant {
    /// Broyden's "good" method -- `scipy.optimize.BroydenFirst`.
    ///
    /// The rank-1 update is chosen to satisfy the secant condition while making the
    /// smallest change to the JACOBIAN in the Frobenius norm.
    First,
    /// Broyden's "bad" method -- `scipy.optimize.BroydenSecond`.
    ///
    /// Minimises the change to the INVERSE Jacobian instead. The names are historical
    /// and the second is not uniformly worse; it is cheaper per step because it needs
    /// no transpose apply.
    Second,
}

/// Limited-memory Broyden approximation of a Jacobian
/// -- `scipy.optimize.BroydenFirst` / `BroydenSecond`.
///
/// The stored object `Gm` approximates the INVERSE Jacobian, which is why `solve` is
/// the cheap direction: it is a matvec against the low-rank form, while `matvec` needs
/// the Sherman-Morrison-Woodbury solve.
///
/// # `alpha`
///
/// The initial Jacobian guess is `-1/alpha`. Left unset, it is auto-scaled on setup to
/// `0.5 * max(||x0||, 1) / ||f0||` -- the same heuristic SciPy uses, and one worth
/// keeping: a fixed initial scale carries the units of nothing in particular, so the
/// first several steps are spent rediscovering the problem's scale instead of its
/// curvature.
#[derive(Debug, Clone)]
pub struct BroydenJacobian {
    variant: BroydenVariant,
    gm: LowRankMatrix,
    alpha: Option<f64>,
    max_rank: Option<usize>,
    reduction: ReductionMethod,
    last_x: Vec<f64>,
    last_f: Vec<f64>,
    n: usize,
}

impl BroydenJacobian {
    /// `alpha = None` auto-scales on setup; `max_rank = None` means no rank reduction,
    /// matching SciPy's defaults.
    #[must_use]
    pub fn new(
        variant: BroydenVariant,
        alpha: Option<f64>,
        reduction: ReductionMethod,
        max_rank: Option<usize>,
    ) -> Self {
        Self {
            variant,
            gm: LowRankMatrix::new(-1.0, 0),
            alpha,
            max_rank,
            reduction,
            last_x: Vec::new(),
            last_f: Vec::new(),
            n: 0,
        }
    }

    /// Broyden's good method with SciPy's defaults.
    #[must_use]
    pub fn first() -> Self {
        Self::new(BroydenVariant::First, None, ReductionMethod::Restart, None)
    }

    /// Broyden's bad method with SciPy's defaults.
    #[must_use]
    pub fn second() -> Self {
        Self::new(BroydenVariant::Second, None, ReductionMethod::Restart, None)
    }

    /// The current inverse-Jacobian approximation.
    #[must_use]
    pub fn inverse_matrix(&self) -> &LowRankMatrix {
        &self.gm
    }

    /// Retained rank of the low-rank representation.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.gm.rank()
    }

    fn reduce(&mut self) {
        // Reduce BEFORE appending, so the update that follows re-establishes the most
        // recent secant condition against the reduced matrix. Reducing afterwards would
        // be free to throw away the very term just added.
        if let Some(cap) = self.max_rank {
            let target = cap.saturating_sub(1);
            match self.reduction {
                ReductionMethod::Restart => self.gm.restart_reduce(target),
                ReductionMethod::Simple => self.gm.simple_reduce(target),
                ReductionMethod::Svd { to_retain } => self.gm.svd_reduce(target, to_retain),
            }
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl Jacobian for BroydenJacobian {
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        self.last_x = x0.to_vec();
        self.last_f = f0.to_vec();
        if self.alpha.is_none() {
            let normf0 = norm(f0);
            self.alpha = Some(if normf0 != 0.0 {
                0.5 * norm(x0).max(1.0) / normf0
            } else {
                1.0
            });
        }
        let alpha = self.alpha.unwrap_or(1.0);
        self.gm = LowRankMatrix::new(-alpha, self.n);
    }

    fn solve_ref(&self, v: &[f64]) -> Vec<f64> {
        self.gm.matvec(v)
    }

    fn solve(&mut self, v: &[f64]) -> Vec<f64> {
        let r = self.gm.matvec(v);
        if r.iter().all(|x| x.is_finite()) {
            return r;
        }
        // Singular: rebuild from the initial guess and retry once. Returning the
        // non-finite vector would put NaN into the caller's iterate, where it is far
        // harder to attribute than a lost update.
        let (x, f) = (self.last_x.clone(), self.last_f.clone());
        self.setup(&x, &f);
        self.gm.matvec(v)
    }

    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        self.gm.solve(v)
    }

    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        self.gm.rmatvec(v)
    }

    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        self.gm.rsolve(v)
    }

    fn update(&mut self, x: &[f64], f: &[f64]) {
        if self.n == 0 || x.len() != self.n || f.len() != self.n {
            return;
        }
        let dx: Vec<f64> = x.iter().zip(&self.last_x).map(|(a, b)| a - b).collect();
        let df: Vec<f64> = f.iter().zip(&self.last_f).map(|(a, b)| a - b).collect();

        self.reduce();

        let gm_df = self.gm.matvec(&df);
        let c: Vec<f64> = dx.iter().zip(&gm_df).map(|(a, b)| a - b).collect();

        let d = match self.variant {
            BroydenVariant::First => {
                // v = Gm^T dx, d = v / (df . v). The transpose apply is what makes the
                // good method more expensive per step than the bad one.
                let v = self.gm.rmatvec(&dx);
                let denom: f64 = df.iter().zip(&v).map(|(a, b)| a * b).sum();
                if denom == 0.0 || !denom.is_finite() {
                    self.last_x = x.to_vec();
                    self.last_f = f.to_vec();
                    return;
                }
                v.iter().map(|vi| vi / denom).collect()
            }
            BroydenVariant::Second => {
                let df_norm2: f64 = df.iter().map(|a| a * a).sum();
                if df_norm2 == 0.0 || !df_norm2.is_finite() {
                    self.last_x = x.to_vec();
                    self.last_f = f.to_vec();
                    return;
                }
                df.iter().map(|v| v / df_norm2).collect()
            }
        };

        self.gm.append(c, d);
        self.last_x = x.to_vec();
        self.last_f = f.to_vec();
    }

    fn dimension(&self) -> usize {
        self.n
    }
}

/// Low-rank Jacobian representations and the Broyden updates built on them.
///
/// The load-bearing property throughout is the SECANT CONDITION: after absorbing a
/// step, `Gm df = dx` holds to rounding rather than approximately, because `Gm`
/// approximates the INVERSE Jacobian and both Broyden updates are constructed to
/// enforce exactly that. Testing against it states what the code must be.
// ─────────────────────────────────────────────────────────────────────────────
// nonlin_solve — the driver that turns a Jacobian approximation into a solver
// ─────────────────────────────────────────────────────────────────────────────

/// What `nonlin_solve` actually needs from a Jacobian approximation.
///
/// Narrower than [`Jacobian`] on purpose. The solver never forms a transpose, and
/// `KrylovJacobian` cannot provide one, so requiring the full trait would exclude the
/// matrix-free method from its own driver. Each type implements this explicitly rather
/// than through a blanket impl over `Jacobian`, because a blanket impl plus a manual one
/// for `KrylovJacobian` is a coherence conflict -- rustc will not reason negatively
/// about the missing `Jacobian` impl even though one does not exist.
pub trait NonlinJacobian {
    /// Prepare at `x0` with residual `f0`.
    fn setup_at(&mut self, x0: &[f64], f0: &[f64]);
    /// Move to `x` with residual `f`.
    fn absorb(&mut self, x: &[f64], f: &[f64]);
    /// Approximate Newton direction: solve `J dx = rhs` to relative tolerance `rtol`.
    ///
    /// `rtol` is honoured by the iterative methods and ignored by the direct ones,
    /// which have no inner tolerance to honour.
    fn newton_direction(&mut self, rhs: &[f64], rtol: f64) -> Vec<f64>;
}

macro_rules! impl_nonlin_jacobian {
    ($ty:ty) => {
        impl NonlinJacobian for $ty {
            fn setup_at(&mut self, x0: &[f64], f0: &[f64]) {
                <Self as Jacobian>::setup(self, x0, f0)
            }
            fn absorb(&mut self, x: &[f64], f: &[f64]) {
                <Self as Jacobian>::update(self, x, f)
            }
            fn newton_direction(&mut self, rhs: &[f64], _rtol: f64) -> Vec<f64> {
                <Self as Jacobian>::solve(self, rhs)
            }
        }
    };
}

impl_nonlin_jacobian!(BroydenJacobian);
impl_nonlin_jacobian!(AndersonJacobian);
impl_nonlin_jacobian!(DiagBroydenJacobian);
impl_nonlin_jacobian!(LinearMixingJacobian);
impl_nonlin_jacobian!(ExcitingMixingJacobian);

impl<F> NonlinJacobian for KrylovJacobian<F>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    fn setup_at(&mut self, x0: &[f64], f0: &[f64]) {
        KrylovJacobian::setup(self, x0, f0)
    }
    fn absorb(&mut self, x: &[f64], f: &[f64]) {
        KrylovJacobian::update(self, x, f)
    }
    /// The one implementation that genuinely uses `rtol`: it is the inner Krylov
    /// tolerance, and the whole point of the forcing sequence below is to choose it.
    fn newton_direction(&mut self, rhs: &[f64], rtol: f64) -> Vec<f64> {
        KrylovJacobian::solve(self, rhs, rtol, InnerMethod::Lgmres)
    }
}

/// Which line search `nonlin_solve` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineSearch {
    /// Take the full step. Fast when it works, divergent when it does not.
    None,
    /// Armijo backtracking with quadratic then cubic interpolation. SciPy's default.
    #[default]
    Armijo,
}

/// Stopping rule -- `scipy.optimize.nonlin.TerminationCondition`.
///
/// Terminates when the residual is small in BOTH an absolute and a relative sense, and
/// the step is small in both senses too. Defaults leave the relative and step-size
/// bounds at infinity so that, out of the box, only `f_tol` binds -- the same shape
/// SciPy ships, where the extra conditions are opt-in rather than silently active.
#[derive(Debug, Clone, Copy)]
pub struct NonlinOptions {
    /// Absolute residual tolerance. Default `eps^(1/3)`, as in SciPy -- deliberately
    /// loose, because the residual of a finite-difference method cannot be driven to
    /// `eps` and asking for it only buys wasted iterations.
    pub f_tol: f64,
    /// Residual tolerance relative to the initial residual. Default infinite.
    pub f_rtol: f64,
    /// Absolute step tolerance. Default infinite.
    pub x_tol: f64,
    /// Step tolerance relative to `||x||`. Default infinite.
    pub x_rtol: f64,
    /// Maximum outer iterations. `None` uses SciPy's `100 * (n + 1)`.
    pub maxiter: Option<usize>,
    /// Line search.
    pub line_search: LineSearch,
}

impl Default for NonlinOptions {
    fn default() -> Self {
        Self {
            f_tol: f64::EPSILON.cbrt(),
            f_rtol: f64::INFINITY,
            x_tol: f64::INFINITY,
            x_rtol: f64::INFINITY,
            maxiter: None,
            line_search: LineSearch::Armijo,
        }
    }
}

/// Outcome of [`nonlin_solve`].
#[derive(Debug, Clone)]
pub struct NonlinResult {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Residual there.
    pub fun: Vec<f64>,
    /// Whether the termination condition was met, as opposed to the iteration cap.
    pub success: bool,
    /// Outer iterations performed.
    pub iterations: usize,
    /// Residual evaluations consumed.
    pub function_calls: usize,
    /// Human-readable status.
    pub message: String,
}

/// Max-norm. SciPy's `tol_norm` default, and NOT the 2-norm.
///
/// The distinction is not cosmetic at large `n`: the 2-norm of a residual grows like
/// `sqrt(n)` for a fixed per-component error, so a 2-norm tolerance silently tightens as
/// the problem grows while a max-norm one means the same thing at every size.
fn max_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

/// Armijo backtracking with quadratic then cubic interpolation
/// -- `scipy.optimize._linesearch.scalar_search_armijo`.
///
/// Returns `None` when no step above `amin` satisfies the sufficient-decrease
/// condition; the caller then takes the full step, which is what SciPy does and is a
/// deliberate gamble rather than a failure.
fn scalar_search_armijo(
    mut phi: impl FnMut(f64) -> f64,
    phi0: f64,
    derphi0: f64,
    c1: f64,
    amin: f64,
) -> Option<(f64, f64)> {
    let alpha0 = 1.0;
    let phi_a0 = phi(alpha0);
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0 {
        return Some((alpha0, phi_a0));
    }

    // Minimiser of the quadratic through phi(0), phi'(0), phi(alpha0).
    let denom = phi_a0 - phi0 - derphi0 * alpha0;
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let mut alpha1 = -derphi0 * alpha0 * alpha0 / 2.0 / denom;
    let mut phi_a1 = phi(alpha1);
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0 {
        return Some((alpha1, phi_a1));
    }

    let mut alpha0 = alpha0;
    let mut phi_a0 = phi_a0;
    while alpha1 > amin {
        let factor = alpha0 * alpha0 * alpha1 * alpha1 * (alpha1 - alpha0);
        if factor == 0.0 || !factor.is_finite() {
            return None;
        }
        let d0 = phi_a1 - phi0 - derphi0 * alpha1;
        let d1 = phi_a0 - phi0 - derphi0 * alpha0;
        let a = (alpha0 * alpha0 * d0 - alpha1 * alpha1 * d1) / factor;
        let b = (-alpha0 * alpha0 * alpha0 * d0 + alpha1 * alpha1 * alpha1 * d1) / factor;
        if a == 0.0 || !a.is_finite() || !b.is_finite() {
            return None;
        }
        let mut alpha2 = (-b + (b * b - 3.0 * a * derphi0).abs().sqrt()) / (3.0 * a);
        let phi_a2 = phi(alpha2);
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0 {
            return Some((alpha2, phi_a2));
        }
        // Guard against a cubic step that barely moves, or moves the wrong way: halve
        // instead. Without it the loop can stall short of `amin` and never terminate.
        if (alpha1 - alpha2) > alpha1 / 2.0 || (1.0 - alpha2 / alpha1) < 0.96 {
            alpha2 = alpha1 / 2.0;
        }
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        // SciPy carries the phi value computed at the PRE-adjustment alpha2 here, even
        // when the halving above replaced it. Reproduced deliberately: re-evaluating
        // would be tidier but would cost an extra residual call per backtrack and would
        // put this on a different iterate sequence than the incumbent.
        phi_a1 = phi_a2;
    }
    None
}

/// Solve `F(x) = 0` with a Jacobian approximation -- `scipy.optimize.nonlin_solve`.
///
/// This is the driver every one of SciPy's `broyden1`, `anderson`, `linearmixing`,
/// `diagbroyden`, `excitingmixing` and `newton_krylov` wrappers is built from, and
/// having it means those are one line each rather than six near-copies of a Newton loop.
///
/// # The forcing sequence is the part worth porting carefully
///
/// For an inexact method the inner solve tolerance controls the whole cost profile:
/// solve too tightly and every outer step overpays, too loosely and the outer iteration
/// stops converging. SciPy uses the Eisenstat-Walker choice with safeguarding,
///
/// ```text
///     eta_A = gamma * ||F_new||^2 / ||F_old||^2
///     eta   = min(eta_max, eta_A)                        if gamma*eta^2 < 0.1
///     eta   = min(eta_max, max(eta_A, gamma*eta^2))      otherwise
/// ```
///
/// so the inner tolerance tightens quadratically as the outer residual falls, and the
/// safeguard stops a single good step from collapsing it prematurely. `root.rs`'s
/// existing `newton_krylov` instead uses a fixed `clamp(0.1*||F||, 1e-10, 0.1)`, which
/// has no memory of the previous step at all.
pub fn nonlin_solve<F, J>(
    func: F,
    x0: &[f64],
    jacobian: &mut J,
    options: NonlinOptions,
) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: NonlinJacobian,
{
    let n = x0.len();
    let mut calls = 0usize;
    let mut x = x0.to_vec();
    let mut fx = func(&x);
    calls += 1;
    let mut fx_norm = max_norm(&fx);
    let f0_norm = fx_norm;
    let mut dx = vec![f64::INFINITY; n];

    jacobian.setup_at(&x, &fx);
    let maxiter = options.maxiter.unwrap_or(100 * (n + 1));

    // Eisenstat-Walker parameters, SciPy's values.
    let gamma = 0.9;
    let eta_max = 0.9999;
    let eta_threshold = 0.1;
    let mut eta = 1e-3;

    let mut iterations = 0usize;
    let mut success = false;

    for _ in 0..maxiter {
        // The condition is checked BEFORE the step, so a starting point that already
        // satisfies it costs no iterations -- and `dx` starts at infinity so the step
        // conditions cannot be met vacuously on the first pass.
        if fx_norm == 0.0
            || (fx_norm <= options.f_tol
                && fx_norm / options.f_rtol <= f0_norm
                && max_norm(&dx) <= options.x_tol
                && max_norm(&dx) / options.x_rtol <= max_norm(&x))
        {
            success = true;
            break;
        }
        iterations += 1;

        let tol = eta.min(eta * fx_norm);
        let step = jacobian.newton_direction(&fx, tol);
        dx = step.iter().map(|v| -v).collect();

        let fx_norm_new;
        match options.line_search {
            LineSearch::None => {
                for (xi, di) in x.iter_mut().zip(&dx) {
                    *xi += di;
                }
                fx = func(&x);
                calls += 1;
                fx_norm_new = max_norm(&fx);
            }
            LineSearch::Armijo => {
                // phi(s) = ||F(x + s dx)||^2, so phi'(0) = -2||F||^2 in exact
                // arithmetic; SciPy passes -phi0 rather than -2*phi0, and this matches
                // it rather than the derivative, because the Armijo constant is
                // calibrated against that choice.
                let phi0: f64 = fx.iter().map(|v| v * v).sum();
                let base_x = x.clone();
                let mut best: Option<(f64, Vec<f64>)> = None;
                let mut inner_calls = 0usize;
                // Scoped so the closure's mutable borrows of `best` and `inner_calls`
                // are definitively released before either is read below.
                let found = {
                    let mut phi = |s: f64| -> f64 {
                        let xt: Vec<f64> =
                            base_x.iter().zip(&dx).map(|(a, b)| a + s * b).collect();
                        let v = func(&xt);
                        inner_calls += 1;
                        let p: f64 = v.iter().map(|q| q * q).sum();
                        best = Some((s, v));
                        p
                    };
                    scalar_search_armijo(&mut phi, phi0, -phi0, 1e-4, 1e-2)
                };
                calls += inner_calls;
                let s = found.map(|(s, _)| s).unwrap_or(1.0);
                for (xi, (bx, di)) in x.iter_mut().zip(base_x.iter().zip(&dx)) {
                    *xi = bx + s * di;
                }
                // Reuse the residual only when it belongs to the step actually taken.
                match &best {
                    Some((bs, bf)) if *bs == s => fx = bf.clone(),
                    _ => {
                        fx = func(&x);
                        calls += 1;
                    }
                }
                fx_norm_new = max_norm(&fx);
            }
        }

        jacobian.absorb(&x, &fx);

        let eta_a = if fx_norm > 0.0 {
            gamma * (fx_norm_new * fx_norm_new) / (fx_norm * fx_norm)
        } else {
            eta_max
        };
        eta = if gamma * eta * eta < eta_threshold {
            eta_max.min(eta_a)
        } else {
            eta_max.min(eta_a.max(gamma * eta * eta))
        };
        fx_norm = fx_norm_new;
    }

    NonlinResult {
        message: if success {
            "A solution was found at the specified tolerance.".to_string()
        } else {
            "The maximum number of iterations allowed has been reached.".to_string()
        },
        x,
        fun: fx,
        success,
        iterations,
        function_calls: calls,
    }
}

/// `scipy.optimize.anderson`.
pub fn anderson<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = AndersonJacobian::new(None, None, None);
    nonlin_solve(func, x0, &mut j, options)
}

/// `scipy.optimize.linearmixing`.
pub fn linear_mixing<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = LinearMixingJacobian::new(None);
    nonlin_solve(func, x0, &mut j, options)
}

/// `scipy.optimize.diagbroyden`.
pub fn diag_broyden<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = DiagBroydenJacobian::new(None);
    nonlin_solve(func, x0, &mut j, options)
}

/// `scipy.optimize.excitingmixing`.
pub fn exciting_mixing<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = ExcitingMixingJacobian::new(None, None);
    nonlin_solve(func, x0, &mut j, options)
}

/// `scipy.optimize.broyden1`, on the LOW-RANK representation.
///
/// Distinct from `root::broyden1`, which stores a dense `n x n` inverse Jacobian. Same
/// method, same iterates; O(n*m) memory instead of O(n^2).
pub fn broyden1_lowrank<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = BroydenJacobian::first();
    nonlin_solve(func, x0, &mut j, options)
}

/// `scipy.optimize.broyden2`, on the low-rank representation.
pub fn broyden2_lowrank<F>(func: F, x0: &[f64], options: NonlinOptions) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut j = BroydenJacobian::second();
    nonlin_solve(func, x0, &mut j, options)
}

// ─────────────────────────────────────────────────────────────────────────────
// root(method=...) dispatch for the nonlin family
// ─────────────────────────────────────────────────────────────────────────────

/// The multivariate root methods `scipy.optimize.root` reaches by name.
///
/// Our existing `RootMethod` in `types.rs` covers the SCALAR bracketing and Newton
/// methods only. These are the vector-valued ones, and they are a separate axis rather
/// than more variants of the same enum: they take a vector residual, they are driven by
/// [`nonlin_solve`], and none of them brackets anything.
///
/// `hybr` and `lm` are absent deliberately -- they are MINPACK ports rather than members
/// of this family, and pretending they belong here by name while dispatching elsewhere
/// would misrepresent what the caller gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NonlinMethod {
    /// Broyden's good method on the low-rank representation. SciPy's `broyden1`.
    #[default]
    Broyden1,
    /// Broyden's bad method. SciPy's `broyden2`.
    Broyden2,
    /// Anderson mixing. SciPy's `anderson`.
    Anderson,
    /// Scalar Jacobian. SciPy's `linearmixing`.
    LinearMixing,
    /// Diagonal Broyden. SciPy's `diagbroyden`.
    DiagBroyden,
    /// Adaptive diagonal steps. SciPy's `excitingmixing`.
    ExcitingMixing,
    /// Matrix-free Newton-Krylov. SciPy's `krylov`.
    Krylov,
}

impl NonlinMethod {
    /// Parse SciPy's spelling, case-insensitively.
    ///
    /// Accepts the names `scipy.optimize.root` accepts for this family and nothing
    /// else. Returning `None` for `hybr` and `lm` is deliberate: they are real SciPy
    /// methods that this dispatch does not provide, and silently substituting a
    /// different method would be worse than refusing the name.
    #[must_use]
    pub fn from_scipy_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "broyden1" => Some(Self::Broyden1),
            "broyden2" => Some(Self::Broyden2),
            "anderson" => Some(Self::Anderson),
            "linearmixing" => Some(Self::LinearMixing),
            "diagbroyden" => Some(Self::DiagBroyden),
            "excitingmixing" => Some(Self::ExcitingMixing),
            "krylov" => Some(Self::Krylov),
            _ => None,
        }
    }

    /// SciPy's spelling of this method.
    #[must_use]
    pub fn scipy_name(self) -> &'static str {
        match self {
            Self::Broyden1 => "broyden1",
            Self::Broyden2 => "broyden2",
            Self::Anderson => "anderson",
            Self::LinearMixing => "linearmixing",
            Self::DiagBroyden => "diagbroyden",
            Self::ExcitingMixing => "excitingmixing",
            Self::Krylov => "krylov",
        }
    }
}

/// Solve `F(x) = 0` by name -- the `scipy.optimize.root` entry point for this family.
///
/// Every method shares [`nonlin_solve`], so they differ only in the Jacobian
/// approximation handed to it. That is the point of having built the driver: adding a
/// method here costs one match arm rather than another Newton loop.
///
/// The `Krylov` arm has to construct its Jacobian around a BORROW of `func`, because a
/// `KrylovJacobian` owns its residual function while the others do not. That is why this
/// takes `func` by reference and the arm passes `&func` rather than moving it.
pub fn root_nonlin<F>(
    func: F,
    x0: &[f64],
    method: NonlinMethod,
    options: NonlinOptions,
) -> NonlinResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    match method {
        NonlinMethod::Broyden1 => {
            let mut j = BroydenJacobian::first();
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::Broyden2 => {
            let mut j = BroydenJacobian::second();
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::Anderson => {
            let mut j = AndersonJacobian::new(None, None, None);
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::LinearMixing => {
            let mut j = LinearMixingJacobian::new(None);
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::DiagBroyden => {
            let mut j = DiagBroydenJacobian::new(None);
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::ExcitingMixing => {
            let mut j = ExcitingMixingJacobian::new(None, None);
            nonlin_solve(&func, x0, &mut j, options)
        }
        NonlinMethod::Krylov => {
            let mut j = KrylovJacobian::with_defaults(&func);
            nonlin_solve(&func, x0, &mut j, options)
        }
    }
}

#[cfg(test)]
mod nonlin_tests {
    use super::{
        BroydenJacobian, BroydenVariant, InverseJacobian, Jacobian, LowRankMatrix,
        ReductionMethod, economic_qr, lu_solve_in_place, one_sided_jacobi,
    };

    const N: usize = 5;

    /// Deterministic pseudo-random vectors: a fixed multiplicative recurrence, so the
    /// test is reproducible without a dependency and without a table of literals.
    fn pseudo(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn with_rank(alpha: f64, rank: usize) -> LowRankMatrix {
        let mut m = LowRankMatrix::new(alpha, N);
        for k in 0..rank {
            m.append(pseudo(k as u64 + 1, N), pseudo(k as u64 + 101, N));
        }
        m
    }

    fn dense_solve(m: &LowRankMatrix, v: &[f64]) -> Vec<f64> {
        let mut a = m.to_dense();
        lu_solve_in_place(&mut a, v, N).expect("reference system is nonsingular")
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// MUST-HIT: the Sherman-Morrison-Woodbury solve agrees with a dense solve of the
    /// matrix it claims to represent, at every rank from 0 up.
    ///
    /// MUST-MISS: at nonzero rank it must NOT agree with `v/alpha`, which is what the
    /// answer would be if the rank-1 terms were silently dropped. Without that arm, a
    /// `solve` that ignored `cs` and `ds` entirely would pass the rank-0 case and the
    /// suite would call it correct.
    #[test]
    fn the_smw_solve_agrees_with_a_dense_solve_at_every_rank() {
        let alpha = -0.7;
        let v = pseudo(999, N);
        for rank in 0..4 {
            let m = with_rank(alpha, rank);
            let got = m.solve(&v);
            let want = dense_solve(&m, &v);
            assert!(
                max_diff(&got, &want) < 1e-9,
                "rank {rank}: SMW solve disagrees with the dense solve by {}",
                max_diff(&got, &want)
            );
            // Round trip through the operator it inverts.
            assert!(
                max_diff(&m.matvec(&got), &v) < 1e-9,
                "rank {rank}: M (M^-1 v) is not v"
            );

            let ignoring_terms: Vec<f64> = v.iter().map(|x| x / alpha).collect();
            if rank == 0 {
                assert!(max_diff(&got, &ignoring_terms) < 1e-12);
            } else {
                assert!(
                    max_diff(&got, &ignoring_terms) > 1e-3,
                    "rank {rank}: the solve matched v/alpha, so the rank-1 terms were \
                     never applied and the agreement above is vacuous"
                );
            }
        }
    }

    /// The transpose operations are a separate code path -- the factor lists are
    /// exchanged rather than the matrix being formed -- so they are checked against an
    /// explicit dense transpose rather than against each other.
    #[test]
    fn the_transpose_operations_match_an_explicit_transpose() {
        let m = with_rank(-0.7, 3);
        let v = pseudo(1234, N);
        let dense = m.to_dense();

        let want_rmatvec: Vec<f64> = (0..N)
            .map(|i| (0..N).map(|j| dense[j * N + i] * v[j]).sum())
            .collect();
        assert!(
            max_diff(&m.rmatvec(&v), &want_rmatvec) < 1e-9,
            "rmatvec is not M^T v"
        );

        let mut at = vec![0.0; N * N];
        for i in 0..N {
            for j in 0..N {
                at[i * N + j] = dense[j * N + i];
            }
        }
        let want_rsolve = lu_solve_in_place(&mut at, &v, N).expect("nonsingular");
        assert!(
            max_diff(&m.rsolve(&v), &want_rsolve) < 1e-9,
            "rsolve is not M^-T v"
        );
        // MUST-MISS: for this non-symmetric matrix the transpose ops differ from the
        // forward ones, so the checks above are not passing on symmetry by accident.
        assert!(
            max_diff(&m.rmatvec(&v), &m.matvec(&v)) > 1e-3,
            "rmatvec and matvec agree; the matrix is symmetric and the test is weak"
        );
    }

    /// Past `n` stored terms the factored form costs more than the matrix it describes,
    /// so it is abandoned. The operator must be UNCHANGED across that switch -- a
    /// representation change that moved the answer would be a bug, not an optimisation.
    #[test]
    fn collapsing_past_n_terms_preserves_the_operator() {
        let alpha = -0.7;
        let v = pseudo(555, N);

        let at_n = with_rank(alpha, N);
        // MUST-MISS: at exactly n terms it has NOT collapsed, so the assertion below
        // is detecting the threshold rather than reporting a matrix that always was.
        assert!(!at_n.is_collapsed(), "collapsed at n terms, one too early");
        assert_eq!(at_n.rank(), N);

        let past_n = with_rank(alpha, N + 1);
        assert!(past_n.is_collapsed(), "did not collapse past n terms");
        assert_eq!(past_n.rank(), 0, "collapsed but kept the factors as well");

        // The same n+1 terms in a representation with no collapse threshold.
        let mut reference = LowRankMatrix::new(alpha, N);
        for k in 0..(N + 1) {
            reference.append(pseudo(k as u64 + 1, N), pseudo(k as u64 + 101, N));
            if reference.is_collapsed() {
                break;
            }
        }
        assert!(
            max_diff(&past_n.matvec(&v), &reference.matvec(&v)) < 1e-9,
            "the operator moved when the representation collapsed"
        );
        assert!(
            max_diff(&past_n.solve(&v), &dense_solve(&past_n, &v)) < 1e-9,
            "the collapsed solve disagrees with a dense solve"
        );
    }

    fn residual(x: &[f64]) -> Vec<f64> {
        vec![
            x[0] * x[0] + x[1] - 3.0,
            x[0] + x[1] * x[1] * x[1] - 5.0,
            x[2] * x[2] - x[0] - 1.0,
            x[3] - x[0].sin(),
            x[4] * x[4] + x[3] - 2.0,
        ]
    }

    /// Both Broyden updates enforce `Gm df = dx` exactly. Driven over several steps of
    /// a genuinely nonlinear system so the condition is re-established each time
    /// against an accumulating representation, not just once from the identity.
    ///
    /// MUST-MISS: the condition must fail for a step never absorbed, or a `matvec` that
    /// simply returned its argument would pass.
    #[test]
    fn both_broyden_updates_satisfy_the_secant_condition() {
        for variant in [BroydenVariant::First, BroydenVariant::Second] {
            let mut x = vec![1.0, 1.2, 0.9, 0.4, 1.1];
            let mut f = residual(&x);
            let mut j =
                BroydenJacobian::new(variant, None, ReductionMethod::Restart, None);
            j.setup(&x, &f);

            for step in 0..6 {
                let dir = j.solve_ref(&f);
                let next: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a - b).collect();
                let next_f = residual(&next);
                let dx: Vec<f64> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
                let df: Vec<f64> = next_f.iter().zip(&f).map(|(a, b)| a - b).collect();

                j.update(&next, &next_f);
                x = next;
                f = next_f;

                let got = j.solve_ref(&df);
                assert!(
                    max_diff(&got, &dx) < 1e-8,
                    "{variant:?} step {step}: secant condition off by {}",
                    max_diff(&got, &dx)
                );

                // MUST-MISS.
                let unrelated = pseudo(step as u64 + 31, N);
                assert!(
                    max_diff(&j.solve_ref(&unrelated), &unrelated) > 1e-6,
                    "{variant:?} step {step}: the approximation acts as the identity, \
                     so the secant check above is vacuous"
                );
            }
        }
    }

    /// The two reduction policies differ in WHAT they keep, not just how much, so
    /// checking the retained count alone would not distinguish them.
    #[test]
    fn the_reduction_policies_keep_different_vectors() {
        let mut restart = LowRankMatrix::new(-1.0, 20);
        let mut simple = LowRankMatrix::new(-1.0, 20);
        for k in 0..5u64 {
            let c = vec![k as f64; N];
            let d = pseudo(k + 7, N);
            restart.append(c.clone(), d.clone());
            simple.append(c, d);
        }

        // MUST-MISS: below the cap neither drops anything.
        restart.restart_reduce(9);
        simple.simple_reduce(9);
        assert_eq!(restart.rank(), 5, "restart dropped below its cap");
        assert_eq!(simple.rank(), 5, "simple dropped below its cap");

        restart.restart_reduce(3);
        simple.simple_reduce(3);
        assert_eq!(restart.rank(), 0, "restart kept vectors; it drops all of them");
        assert_eq!(simple.rank(), 3, "simple did not reduce to its cap");

        // WHICH three it kept is the actual claim, and a count cannot check it. Build
        // the operator that keeps only the three NEWEST and require agreement; then
        // build the one that keeps the three OLDEST and require disagreement, so the
        // test distinguishes the two policies rather than merely counting.
        let probe = pseudo(88, N);

        let mut newest = LowRankMatrix::new(-1.0, 20);
        for k in 2..5u64 {
            newest.append(vec![k as f64; N], pseudo(k + 7, N));
        }
        assert!(
            max_diff(&simple.matvec(&probe), &newest.matvec(&probe)) < 1e-12,
            "simple_reduce did not retain the three newest vectors"
        );

        let mut oldest = LowRankMatrix::new(-1.0, 20);
        for k in 0..3u64 {
            oldest.append(vec![k as f64; N], pseudo(k + 7, N));
        }
        assert!(
            max_diff(&simple.matvec(&probe), &oldest.matvec(&probe)) > 1e-3,
            "keeping the oldest three is indistinguishable here, so the check above \
             does not establish which end was dropped"
        );

        // Restart dropped everything, so it is now the bare multiple of the identity.
        let bare: Vec<f64> = probe.iter().map(|x| -x).collect();
        assert!(
            max_diff(&restart.matvec(&probe), &bare) < 1e-12,
            "restart_reduce left rank-1 terms behind"
        );
    }

    /// `InverseJacobian` presents solve as matvec and back. Checked against the
    /// wrapped object directly rather than against a second computation of the same
    /// thing, so a wrapper that forwarded without swapping would fail.
    #[test]
    fn inverse_jacobian_swaps_the_two_directions() {
        let x = vec![1.0, 1.2, 0.9, 0.4, 1.1];
        let f = residual(&x);
        let mut inner =
            BroydenJacobian::new(BroydenVariant::First, Some(0.5), ReductionMethod::Restart, None);
        inner.setup(&x, &f);
        let next: Vec<f64> = x.iter().map(|a| a + 0.05).collect();
        inner.update(&next, &residual(&next));

        let v = pseudo(4242, N);
        let want_solve = inner.matvec(&v);
        let want_matvec = inner.solve_ref(&v);
        // MUST-MISS: the two directions genuinely differ here, so a wrapper that did
        // nothing would be caught.
        assert!(
            max_diff(&want_solve, &want_matvec) > 1e-6,
            "the two directions agree; the swap cannot be observed"
        );

        let wrapped = InverseJacobian::new(inner);
        assert!(
            max_diff(&wrapped.solve_ref(&v), &want_solve) < 1e-12,
            "InverseJacobian::solve is not the inner matvec"
        );
        assert!(
            max_diff(&wrapped.matvec(&v), &want_matvec) < 1e-12,
            "InverseJacobian::matvec is not the inner solve"
        );
    }

    /// The auto-scale is `0.5 * max(||x0||, 1) / ||f0||`, applied only when `alpha` was
    /// left unset. A fixed initial scale carries the units of nothing in particular.
    #[test]
    fn alpha_is_auto_scaled_only_when_unset() {
        let x = vec![3.0, 4.0, 0.0, 0.0, 0.0]; // ||x|| = 5
        let f = vec![0.0, 0.0, 2.0, 0.0, 0.0]; // ||f|| = 2
        let mut auto = BroydenJacobian::first();
        auto.setup(&x, &f);
        // Gm starts at -alpha * I, so Gm e0 = -alpha e0 with alpha = 0.5*5/2 = 1.25.
        let e0 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(
            (auto.solve_ref(&e0)[0] + 1.25).abs() < 1e-12,
            "auto-scaled alpha is not 0.5*max(||x||,1)/||f||, got {}",
            -auto.solve_ref(&e0)[0]
        );

        // MUST-MISS: an explicit alpha is left alone.
        let mut fixed = BroydenJacobian::new(
            BroydenVariant::First,
            Some(0.25),
            ReductionMethod::Restart,
            None,
        );
        fixed.setup(&x, &f);
        assert!(
            (fixed.solve_ref(&e0)[0] + 0.25).abs() < 1e-12,
            "an explicitly supplied alpha was overwritten by the heuristic"
        );
    }

    /// The LU reports singularity rather than returning a fabricated answer, and it
    /// pivots -- a zero leading entry is not a failure.
    #[test]
    fn the_lu_pivots_and_reports_singularity() {
        // Zero pivot in position (0,0): solvable only with pivoting.
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        let got = lu_solve_in_place(&mut a, &[2.0, 3.0], 2).expect("pivoting should succeed");
        assert!(
            (got[0] - 3.0).abs() < 1e-12 && (got[1] - 2.0).abs() < 1e-12,
            "pivoted solve returned {got:?}, expected [3, 2]"
        );

        // MUST-HIT the singularity guard: a rank-1 matrix has no inverse.
        let mut singular = vec![1.0, 2.0, 2.0, 4.0];
        assert!(
            lu_solve_in_place(&mut singular, &[1.0, 1.0], 2).is_none(),
            "a singular system produced an answer"
        );
    }

    /// Build a matrix whose OLDEST terms are the large ones and whose newest are
    /// negligible. This is the case the SVD policy exists for, and the one where
    /// choosing by age gets it exactly backwards.
    fn skewed_terms() -> LowRankMatrix {
        let mut m = LowRankMatrix::new(-1.0, 6);
        for k in 0..4u64 {
            let scale = if k < 2 { 1.0 } else { 1e-6 };
            let c: Vec<f64> = pseudo(k + 1, 6).iter().map(|x| x * scale).collect();
            m.append(c, pseudo(k + 101, 6));
        }
        m
    }

    fn operator_error(reduced: &LowRankMatrix, reference: &[f64]) -> f64 {
        max_diff(&reduced.to_dense(), reference)
    }

    /// The whole justification for the SVD policy in one comparison: it keeps what is
    /// LARGE, while restart and simple keep what is RECENT, and recency is a proxy for
    /// relevance that can be arbitrarily wrong.
    ///
    /// MUST-HIT: reducing 4 terms to 2 by singular value costs ~1e-7 here.
    /// MUST-MISS: the same reduction by age costs ~0.93 -- six orders of magnitude
    /// worse -- so the first number cannot be explained by the operator being easy to
    /// approximate, or by the reduction having quietly done nothing.
    #[test]
    fn svd_reduce_keeps_the_largest_components_not_the_newest() {
        let reference = skewed_terms().to_dense();

        let mut by_svd = skewed_terms();
        by_svd.svd_reduce(4, Some(2));
        assert_eq!(by_svd.rank(), 2, "svd_reduce did not retain exactly 2 components");
        let svd_err = operator_error(&by_svd, &reference);
        assert!(
            svd_err < 1e-5,
            "svd_reduce lost {svd_err}; it should have kept the two dominant terms"
        );

        let mut by_age = skewed_terms();
        by_age.simple_reduce(2);
        let age_err = operator_error(&by_age, &reference);
        assert!(
            age_err > 0.1,
            "keeping the newest two cost only {age_err}; the test data does not \
             actually distinguish the policies"
        );
        assert!(
            svd_err < age_err / 1000.0,
            "svd_reduce ({svd_err}) is not decisively better than age-based \
             ({age_err}); the ranking by singular value is not doing its job"
        );
    }

    /// Below the cap there is nothing to reduce, and reducing anyway would throw away
    /// information the caller still has room for.
    #[test]
    fn svd_reduce_does_nothing_below_the_cap() {
        let mut below = skewed_terms();
        let before = below.to_dense();
        // 4 stored terms, cap of 5: `m < p`, so this must be a no-op.
        below.svd_reduce(5, Some(2));
        assert_eq!(below.rank(), 4, "reduced below the cap");
        assert!(
            max_diff(&below.to_dense(), &before) < 1e-12,
            "a no-op reduction changed the operator"
        );

        // MUST-MISS: at the cap it does reduce, so the check above is detecting the
        // threshold rather than a routine that never fires.
        let mut at = skewed_terms();
        at.svd_reduce(4, Some(2));
        assert_eq!(at.rank(), 2, "reduction did not fire at the cap");
    }

    /// `to_retain` defaults to `max_rank - 2`, as in SciPy.
    #[test]
    fn svd_reduce_defaults_to_retain_two_fewer() {
        let mut m = skewed_terms();
        m.svd_reduce(4, None);
        assert_eq!(m.rank(), 2, "default to_retain is not max_rank - 2");
    }

    /// `A = Q R` with `Q` orthonormal, including for a RANK-DEFICIENT input, where the
    /// routine leaves a zero column and a zero pivot rather than failing. The caller's
    /// operator identity depends on the factorisation holding in that case too.
    #[test]
    fn economic_qr_factors_its_input_including_rank_deficient_ones() {
        let full: Vec<Vec<f64>> = (0..3).map(|k| pseudo(k + 11, 6)).collect();
        let dup = vec![pseudo(11, 6), pseudo(12, 6), pseudo(11, 6)];

        for (label, cols) in [("full rank", full), ("rank deficient", dup)] {
            let (q, r) = economic_qr(&cols);
            let m = cols.len();
            for j in 0..m {
                let rebuilt: Vec<f64> = (0..6)
                    .map(|i| (0..m).map(|k| q[k][i] * r[k][j]).sum())
                    .collect();
                assert!(
                    max_diff(&rebuilt, &cols[j]) < 1e-10,
                    "{label}: column {j} does not satisfy A = Q R"
                );
            }
            // Nonzero Q columns are orthonormal; the deficient one is exactly zero.
            for i in 0..m {
                let nrm: f64 = q[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!(
                    nrm < 1e-12 || (nrm - 1.0).abs() < 1e-10,
                    "{label}: Q column {i} has norm {nrm}, neither unit nor zero"
                );
                for j in (i + 1)..m {
                    let ip: f64 = q[i].iter().zip(&q[j]).map(|(a, b)| a * b).sum();
                    assert!(
                        ip.abs() < 1e-10,
                        "{label}: Q columns {i} and {j} are not orthogonal ({ip})"
                    );
                }
            }
        }
    }

    /// The Jacobi SVD is checked against INVARIANTS rather than a reference
    /// implementation, which this crate does not have: the output columns are mutually
    /// orthogonal, `V` is orthogonal, the rotation reproduces `A V`, and the singular
    /// values satisfy two exact identities -- their squares sum to the Frobenius norm
    /// and their product is `det(A^T A)`. Together those pin the spectrum without
    /// anything to compare against.
    #[test]
    fn the_jacobi_svd_is_an_orthogonal_rotation_with_the_right_spectrum() {
        let original: Vec<Vec<f64>> = (0..3).map(|k| pseudo(k + 21, 7)).collect();
        let mut a = original.clone();
        let v = one_sided_jacobi(&mut a);
        let m = 3;

        // V orthogonal.
        for i in 0..m {
            for j in 0..m {
                let ip: f64 = v[i].iter().zip(&v[j]).map(|(x, y)| x * y).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((ip - want).abs() < 1e-10, "V is not orthogonal at ({i}, {j})");
            }
        }
        // The rotated matrix is A V: column j is sum_k A[:,k] V[k][j], and V[k][j] is
        // `v[j][k]` because `v[j]` is column j.
        for j in 0..m {
            let want: Vec<f64> = (0..7)
                .map(|i| (0..m).map(|k| original[k][i] * v[j][k]).sum())
                .collect();
            assert!(
                max_diff(&a[j], &want) < 1e-10,
                "rotated column {j} is not (A V) column {j}"
            );
        }
        // Output columns are mutually orthogonal -- that is what "one-sided Jacobi has
        // converged" means.
        for i in 0..m {
            for j in (i + 1)..m {
                let ip: f64 = a[i].iter().zip(&a[j]).map(|(x, y)| x * y).sum();
                let scale = norm_of(&a[i]) * norm_of(&a[j]);
                assert!(
                    ip.abs() < 1e-9 * scale.max(1.0),
                    "columns {i} and {j} are not orthogonal after convergence ({ip})"
                );
            }
        }

        let svals: Vec<f64> = a.iter().map(|c| norm_of(c)).collect();
        // Sorted descending, without which the truncation is meaningless.
        for w in svals.windows(2) {
            assert!(w[0] >= w[1], "singular values are not sorted descending: {svals:?}");
        }
        // Sum of squares = squared Frobenius norm of the input.
        let frob2: f64 = original.iter().flat_map(|c| c.iter()).map(|x| x * x).sum();
        let sum_sq: f64 = svals.iter().map(|s| s * s).sum();
        assert!(
            (sum_sq - frob2).abs() < 1e-9 * frob2,
            "sum of squared singular values {sum_sq} != Frobenius norm squared {frob2}"
        );
        // Product of squares = det(A^T A), computed independently by LU.
        let mut gram = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                gram[i * m + j] = original[i].iter().zip(&original[j]).map(|(x, y)| x * y).sum();
            }
        }
        let det = det_via_lu(&mut gram, m);
        let prod_sq: f64 = svals.iter().map(|s| s * s).product();
        assert!(
            (prod_sq - det).abs() < 1e-6 * det.abs().max(1.0),
            "product of squared singular values {prod_sq} != det(A^T A) {det}"
        );
    }

    fn norm_of(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Determinant by the same elimination the solver uses, for an independent check on
    /// the singular values. Sign tracking matters: an even number of row swaps must not
    /// flip it.
    fn det_via_lu(a: &mut [f64], n: usize) -> f64 {
        let mut det = 1.0;
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut piv = k;
            let mut best = a[perm[k] * n + k].abs();
            for i in (k + 1)..n {
                let v = a[perm[i] * n + k].abs();
                if v > best {
                    best = v;
                    piv = i;
                }
            }
            if best == 0.0 {
                return 0.0;
            }
            if piv != k {
                perm.swap(k, piv);
                det = -det;
            }
            let pk = perm[k];
            det *= a[pk * n + k];
            for i in (k + 1)..n {
                let pi = perm[i];
                let factor = a[pi * n + k] / a[pk * n + k];
                for j in (k + 1)..n {
                    a[pi * n + j] -= factor * a[pk * n + j];
                }
            }
        }
        det
    }

    /// A Broyden run driven with the SVD policy still satisfies the secant condition and
    /// stays within its rank cap. The reduction runs BEFORE each append, so the cap is
    /// what bounds memory across the whole run.
    #[test]
    fn the_svd_policy_bounds_rank_without_breaking_the_secant_condition() {
        let mut x = vec![1.0, 1.2, 0.9, 0.4, 1.1];
        let mut f = residual(&x);
        let mut j = BroydenJacobian::new(
            BroydenVariant::First,
            None,
            // `to_retain` is explicit here: with a cap of 3 SciPy's own default
            // arithmetic gives `q = (3 - 1) - 2 = 0`, which clears the matrix and would
            // quietly turn this into a test of the restart policy.
            ReductionMethod::Svd { to_retain: Some(2) },
            Some(4),
        );
        j.setup(&x, &f);

        for step in 0..8 {
            let dir = j.solve_ref(&f);
            let next: Vec<f64> = x.iter().zip(&dir).map(|(a, b)| a - b).collect();
            let next_f = residual(&next);
            let df: Vec<f64> = next_f.iter().zip(&f).map(|(a, b)| a - b).collect();
            let dx: Vec<f64> = next.iter().zip(&x).map(|(a, b)| a - b).collect();

            j.update(&next, &next_f);
            x = next;
            f = next_f;

            assert!(
                j.rank() <= 3,
                "step {step}: rank {} exceeded the cap of 3",
                j.rank()
            );
            assert!(
                max_diff(&j.solve_ref(&df), &dx) < 1e-8,
                "step {step}: the most recent secant condition was not restored after \
                 reduction"
            );
        }
    }

    // ── nonlin_solve driver and dispatch ────────────────────────────────────

    use super::{
        LineSearch, NonlinMethod, NonlinOptions, NonlinResult, nonlin_solve, root_nonlin,
    };

    /// A well-behaved nonlinear system with a known root, used to check that the
    /// DRIVER converges rather than that any particular Jacobian is clever.
    /// `x_i^2 = i + 1` with a weak coupling term, root near `sqrt(i + 1)`.
    fn solvable(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|i| x[i] * x[i] - (i as f64 + 1.0) + 0.05 * x[(i + 1) % n])
            .collect()
    }

    /// Every dispatchable method must actually solve a solvable system. This is the
    /// test that would catch a Jacobian wired into the driver backwards -- each one
    /// converges on its own or the arm is broken, regardless of how the others do.
    #[test]
    fn every_dispatched_method_solves_a_solvable_system() {
        let x0 = vec![1.0, 1.0, 1.0, 1.0];
        let opts = NonlinOptions {
            f_tol: 1e-8,
            maxiter: Some(400),
            ..NonlinOptions::default()
        };
        for method in [
            NonlinMethod::Broyden1,
            NonlinMethod::Broyden2,
            NonlinMethod::Anderson,
            NonlinMethod::LinearMixing,
            NonlinMethod::DiagBroyden,
            NonlinMethod::ExcitingMixing,
            NonlinMethod::Krylov,
        ] {
            let r: NonlinResult = root_nonlin(solvable, &x0, method, opts);
            let resid = r.fun.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
            assert!(
                resid < 1e-6,
                "{} did not solve the system: residual {resid}, success={}, iters={}",
                method.scipy_name(),
                r.success,
                r.iterations
            );
            assert!(
                r.function_calls > 0,
                "{} reported zero function calls",
                method.scipy_name()
            );
        }
    }

    /// A starting point that already satisfies the tolerance must cost ZERO iterations,
    /// and one that does not must cost at least one. The condition is checked before the
    /// step, and `dx` starts at infinity so the step-size clauses cannot pass vacuously
    /// on the first pass -- a mistake that would make every solve terminate immediately.
    #[test]
    fn an_already_converged_start_costs_no_iterations() {
        let opts = NonlinOptions {
            f_tol: 1e-8,
            ..NonlinOptions::default()
        };
        let zero = |_: &[f64]| vec![0.0, 0.0];
        let mut j = LinearMixingJacobian::new(Some(0.5));
        let r = nonlin_solve(zero, &[1.0, 2.0], &mut j, opts);
        assert_eq!(r.iterations, 0, "a converged start still iterated");
        assert!(r.success, "a converged start was not reported as success");

        // MUST-MISS: a genuine problem does iterate, so the check above is about the
        // starting point and not about the driver refusing to run at all.
        let mut j2 = DiagBroydenJacobian::new(Some(0.5));
        let r2 = nonlin_solve(solvable, &[1.0, 1.0, 1.0, 1.0], &mut j2, opts);
        assert!(r2.iterations > 0, "the driver never iterated on a real problem");
    }

    /// The iteration cap is honoured and reported as a failure rather than as a
    /// success with a bad answer -- the distinction a caller acts on.
    #[test]
    fn hitting_the_iteration_cap_reports_failure() {
        let opts = NonlinOptions {
            f_tol: 1e-14,
            maxiter: Some(2),
            ..NonlinOptions::default()
        };
        let mut j = LinearMixingJacobian::new(Some(0.01));
        let r = nonlin_solve(solvable, &[5.0, 5.0, 5.0, 5.0], &mut j, opts);
        assert!(!r.success, "hitting the cap was reported as success");
        assert_eq!(r.iterations, 2, "the cap was not honoured");
        assert!(
            r.message.contains("maximum number of iterations"),
            "unhelpful message: {}",
            r.message
        );
    }

    /// The Armijo search must cost extra residual evaluations relative to taking the
    /// full step, and must not change the answer on a problem both settings solve. If
    /// the counts matched, the line search would not be running at all.
    #[test]
    fn the_line_search_costs_evaluations_and_is_actually_running() {
        let x0 = vec![3.0, 3.0, 3.0, 3.0];
        let base = NonlinOptions {
            f_tol: 1e-9,
            maxiter: Some(200),
            ..NonlinOptions::default()
        };

        let mut j1 = BroydenJacobian::first();
        let with = nonlin_solve(
            solvable,
            &x0,
            &mut j1,
            NonlinOptions {
                line_search: LineSearch::Armijo,
                ..base
            },
        );
        let mut j2 = BroydenJacobian::first();
        let without = nonlin_solve(
            solvable,
            &x0,
            &mut j2,
            NonlinOptions {
                line_search: LineSearch::None,
                ..base
            },
        );

        assert!(with.success && without.success, "both settings should solve this");
        assert!(
            with.function_calls != without.function_calls,
            "the two line-search settings consumed identical evaluations ({}); the \
             search is not running",
            with.function_calls
        );
        // Same root, whichever path got there.
        let d = max_diff(&with.x, &without.x);
        assert!(d < 1e-5, "the two settings converged to different points ({d})");
    }

    /// Name parsing accepts exactly SciPy's spellings for this family and refuses the
    /// ones it does not implement, rather than silently substituting a method.
    #[test]
    fn scipy_names_round_trip_and_unsupported_ones_are_refused() {
        for m in [
            NonlinMethod::Broyden1,
            NonlinMethod::Broyden2,
            NonlinMethod::Anderson,
            NonlinMethod::LinearMixing,
            NonlinMethod::DiagBroyden,
            NonlinMethod::ExcitingMixing,
            NonlinMethod::Krylov,
        ] {
            assert_eq!(
                NonlinMethod::from_scipy_name(m.scipy_name()),
                Some(m),
                "{} does not round-trip",
                m.scipy_name()
            );
            assert_eq!(
                NonlinMethod::from_scipy_name(&m.scipy_name().to_uppercase()),
                Some(m),
                "{} is not matched case-insensitively",
                m.scipy_name()
            );
        }
        // MUST-MISS: real SciPy methods this dispatch does NOT provide are refused
        // rather than mapped to something else.
        for absent in ["hybr", "lm", "df-sane", "", "broyden3"] {
            assert_eq!(
                NonlinMethod::from_scipy_name(absent),
                None,
                "{absent} was accepted but is not provided here"
            );
        }
    }

    // ── Mixing and Anderson Jacobians ───────────────────────────────────────

    use super::{
        AndersonJacobian, DiagBroydenJacobian, ExcitingMixingJacobian, LinearMixingJacobian,
    };

    /// Build the dense operator a `Jacobian` represents, column by column, by applying
    /// it to the basis. Used to check the transpose operations against an EXPLICIT
    /// transpose rather than against another formula.
    fn dense_of(n: usize, apply: impl Fn(&[f64]) -> Vec<f64>) -> Vec<Vec<f64>> {
        (0..n)
            .map(|j| {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                apply(&e)
            })
            .collect() // column j
    }

    fn transposed(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = cols.len();
        (0..n).map(|j| (0..n).map(|i| cols[i][j]).collect()).collect()
    }

    fn max_diff_cols(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| max_diff(x, y))
            .fold(0.0, f64::max)
    }

    /// Linear mixing holds a FIXED Jacobian, so `update` must change nothing at all.
    /// Paired with DiagBroyden on the identical steps, which must change, so this is a
    /// statement about linear mixing rather than about the steps being uninformative.
    #[test]
    fn linear_mixing_never_updates_but_diag_broyden_does() {
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let f0 = vec![0.5, -0.25, 0.75, -1.0];
        let probe = pseudo(404, 4);

        let mut lm = LinearMixingJacobian::new(Some(0.4));
        lm.setup(&x0, &f0);
        let before = lm.solve_ref(&probe);

        let mut db = DiagBroydenJacobian::new(Some(0.4));
        db.setup(&x0, &f0);
        let db_before = db.solve_ref(&probe);
        // Both start as the same scalar operator, which is what makes the pairing fair.
        assert!(
            max_diff(&before, &db_before) < 1e-12,
            "the two methods do not start from the same operator"
        );

        let x1 = vec![1.3, 2.1, 2.6, 4.2];
        let f1 = vec![0.2, -0.4, 0.5, -0.6];
        lm.update(&x1, &f1);
        db.update(&x1, &f1);

        assert!(
            lm.solve_ref(&probe)
                .iter()
                .zip(&before)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "linear mixing changed its operator on update"
        );
        // MUST-MISS.
        assert!(
            max_diff(&db.solve_ref(&probe), &db_before) > 1e-6,
            "DiagBroyden also ignored the step, so the check above says nothing about \
             linear mixing specifically"
        );
    }

    /// DiagBroyden satisfies the secant condition ONLY in the coordinate the step
    /// touched -- exactly, to rounding -- and structurally cannot satisfy it elsewhere,
    /// because a diagonal operator applied to a single-coordinate step is zero in every
    /// other slot. That is the method's defining limitation rather than a defect, and
    /// the test states both halves so neither can be mistaken for the other.
    #[test]
    fn diag_broyden_matches_the_secant_condition_only_where_the_step_moved() {
        let n = 5;
        let k = 2;
        let alpha = 0.4;
        let x0 = vec![0.0; n];
        let f0 = vec![0.0; n];
        let mut db = DiagBroydenJacobian::new(Some(alpha));
        db.setup(&x0, &f0);

        let mut x1 = vec![0.0; n];
        x1[k] = 0.75;
        let df = pseudo(5, n);
        // f0 is zero, so df is the new residual outright.
        db.update(&x1, &df);

        let jdx = db.matvec(&x1);
        assert!(
            (jdx[k] - df[k]).abs() < 1e-12,
            "the moved coordinate does not satisfy the secant condition: {} vs {}",
            jdx[k],
            df[k]
        );
        for i in 0..n {
            if i != k {
                assert_eq!(jdx[i], 0.0, "a diagonal operator responded off-coordinate");
                assert!(
                    df[i].abs() > 1e-3,
                    "the test data has a near-zero residual at {i}, so the limitation \
                     below is not actually exercised"
                );
            }
        }
        // The full-vector condition therefore FAILS, and by a wide margin.
        assert!(
            max_diff(&jdx, &df) > 0.1,
            "the full secant condition appears to hold, which a diagonal approximation \
             cannot do here -- the test data must be degenerate"
        );
    }

    /// Exciting mixing grows the step where the residual keeps its sign, resets it
    /// where the sign flips, and saturates at `alphamax`. All three are exact.
    #[test]
    fn exciting_mixing_grows_resets_and_saturates() {
        let alpha = 0.3;
        let x = vec![0.0; 4];
        let last_f = vec![1.0, -1.0, 2.0, -0.5];
        let mut em = ExcitingMixingJacobian::new(Some(alpha), Some(1.0));
        em.setup(&x, &last_f);
        assert_eq!(em.beta(), &[0.3, 0.3, 0.3, 0.3]);

        // Signs: keep, keep, flip, flip.
        let f = vec![0.5, -2.0, -1.0, 0.25];
        em.update(&x, &f);
        let b = em.beta();
        for (i, want) in [0.6, 0.6, 0.3, 0.3].into_iter().enumerate() {
            assert!(
                (b[i] - want).abs() < 1e-12,
                "beta[{i}] is {} but should be {want}",
                b[i]
            );
        }

        // Saturation: repeated sign-keeping must stop at alphamax and not run away.
        let mut sat = ExcitingMixingJacobian::new(Some(alpha), Some(1.0));
        let one = vec![1.0];
        let origin = [0.0];
        sat.setup(&origin, &one);
        for _ in 0..10 {
            sat.update(&origin, &one);
        }
        assert!(
            (sat.beta()[0] - 1.0).abs() < 1e-12,
            "beta did not saturate at alphamax, reached {}",
            sat.beta()[0]
        );
        // MUST-MISS: a larger cap is actually reached, so the clamp above is the clamp
        // and not an accident of the growth rate.
        let mut wide = ExcitingMixingJacobian::new(Some(alpha), Some(5.0));
        wide.setup(&origin, &one);
        for _ in 0..10 {
            wide.update(&origin, &one);
        }
        assert!(
            wide.beta()[0] > 1.5,
            "with alphamax = 5 the step should have grown past 1.5, got {}",
            wide.beta()[0]
        );
    }

    /// Anderson keeps two SEPARATELY written formulas -- one for the inverse, one for
    /// the forward product, with different matrices -- and they must be actual inverses
    /// of each other. Neither is derivable from the other by inspection, so this is the
    /// check that catches a slip in either.
    #[test]
    fn anderson_forward_and_inverse_products_are_inverses() {
        let n = 6;
        let mut a = AndersonJacobian::new(Some(0.35), Some(0.01), Some(5));
        let x0 = vec![0.0; n];
        let f0 = vec![0.0; n];
        a.setup(&x0, &f0);
        // Three history pairs, built by feeding cumulative points.
        let mut x = x0.clone();
        let mut f = f0.clone();
        for k in 0..3u64 {
            let dx = pseudo(k + 1, n);
            let df = pseudo(k + 51, n);
            for i in 0..n {
                x[i] += dx[i];
                f[i] += df[i];
            }
            a.update(&x, &f);
        }
        assert_eq!(a.history_len(), 3);

        let jinv = dense_of(n, |v| a.solve_ref(v));
        let jfwd = dense_of(n, |v| a.matvec(v));
        for i in 0..n {
            for j in 0..n {
                let prod: f64 = (0..n).map(|k| jfwd[k][i] * jinv[j][k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod - want).abs() < 1e-8,
                    "J * J^-1 is not the identity at ({i}, {j}): {prod}"
                );
            }
        }
    }

    /// The transpose operations are DERIVED here rather than ported -- SciPy's Anderson
    /// does not define them -- so they are checked against an explicit dense transpose
    /// of the very operators they claim to transpose, and the operator is confirmed
    /// non-symmetric so the check cannot pass trivially.
    #[test]
    fn anderson_transposes_match_an_explicit_dense_transpose() {
        let n = 6;
        let mut a = AndersonJacobian::new(Some(0.35), Some(0.01), Some(5));
        let zeros = vec![0.0; n];
        a.setup(&zeros, &zeros);
        let mut x = vec![0.0; n];
        let mut f = vec![0.0; n];
        for k in 0..3u64 {
            let dx = pseudo(k + 1, n);
            let df = pseudo(k + 51, n);
            for i in 0..n {
                x[i] += dx[i];
                f[i] += df[i];
            }
            a.update(&x, &f);
        }

        let jinv = dense_of(n, |v| a.solve_ref(v));
        let jfwd = dense_of(n, |v| a.matvec(v));
        let rs = dense_of(n, |v| a.rsolve(v));
        let rm = dense_of(n, |v| a.rmatvec(v));

        assert!(
            max_diff_cols(&rs, &transposed(&jinv)) < 1e-9,
            "rsolve is not the transpose of the inverse product"
        );
        assert!(
            max_diff_cols(&rm, &transposed(&jfwd)) < 1e-9,
            "rmatvec is not the transpose of the forward product"
        );
        // MUST-MISS: the operator is genuinely non-symmetric, so the two checks above
        // are not satisfied by any operator that ignored the transpose entirely.
        assert!(
            max_diff_cols(&jinv, &transposed(&jinv)) > 0.1,
            "the Anderson operator is symmetric on this history; the transpose tests \
             cannot distinguish a correct implementation from a no-op"
        );
    }

    /// The history is capped at `m`, and `m = 0` disables the method entirely -- it
    /// degenerates to linear mixing, which is what SciPy does and is worth pinning
    /// because it is the one setting that silently changes the algorithm.
    #[test]
    fn anderson_history_is_capped_and_zero_m_degenerates_to_linear_mixing() {
        let n = 4;
        let alpha = 0.35;
        let mut a = AndersonJacobian::new(Some(alpha), Some(0.01), Some(2));
        let zeros = vec![0.0; n];
        a.setup(&zeros, &zeros);
        let mut x = vec![0.0; n];
        let mut f = vec![0.0; n];
        for k in 0..5u64 {
            for i in 0..n {
                x[i] += pseudo(k + 1, n)[i];
                f[i] += pseudo(k + 71, n)[i];
            }
            a.update(&x, &f);
            assert!(a.history_len() <= 2, "history exceeded m = 2");
        }
        assert_eq!(a.history_len(), 2, "history never filled to m");

        let mut zero = AndersonJacobian::new(Some(alpha), Some(0.01), Some(0));
        let zeros = vec![0.0; n];
        zero.setup(&zeros, &zeros);
        zero.update(&x, &f);
        assert_eq!(zero.history_len(), 0, "m = 0 still accumulated history");
        let probe = pseudo(808, n);
        let want: Vec<f64> = probe.iter().map(|v| -alpha * v).collect();
        assert!(
            max_diff(&zero.solve_ref(&probe), &want) < 1e-12,
            "with m = 0 the step is not plain linear mixing"
        );
    }

    /// All four new strategies drive through the trait, alongside the two already
    /// there. A trait with six implementors is only worth having if calling through it
    /// actually works for each.
    #[test]
    fn every_strategy_drives_through_the_trait() {
        let n = 4;
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let f0 = vec![0.5, -0.25, 0.75, -1.0];
        let x1 = vec![1.3, 2.1, 2.6, 4.2];
        let f1 = vec![0.2, -0.4, 0.5, -0.6];

        let mut lm = LinearMixingJacobian::new(Some(0.4));
        let mut db = DiagBroydenJacobian::new(Some(0.4));
        let mut em = ExcitingMixingJacobian::new(Some(0.4), None);
        let mut an = AndersonJacobian::new(Some(0.4), None, None);
        let mut br = BroydenJacobian::first();
        let strategies: [&mut dyn Jacobian; 5] =
            [&mut lm, &mut db, &mut em, &mut an, &mut br];

        for (i, s) in strategies.into_iter().enumerate() {
            s.setup(&x0, &f0);
            assert_eq!(s.dimension(), n, "strategy {i} reported the wrong dimension");
            s.update(&x1, &f1);
            let probe = pseudo(909, n);
            let step = s.solve_ref(&probe);
            assert_eq!(step.len(), n, "strategy {i} returned the wrong length");
            assert!(
                step.iter().all(|v| v.is_finite()),
                "strategy {i} produced a non-finite step"
            );
            // Every one of these is a descent-direction approximation of -J^-1, so a
            // nonzero residual must produce a nonzero step.
            assert!(
                norm_of(&step) > 1e-12,
                "strategy {i} returned an all-zero step for a nonzero residual"
            );
        }
    }

    // ── KrylovJacobian ──────────────────────────────────────────────────────

    use super::{InnerMethod, KrylovJacobian};

    /// `F_i = x_i^2 + 0.1 x_{i+1}`, whose Jacobian is known in closed form, so the
    /// finite-difference operator can be checked against the truth rather than against
    /// another approximation.
    fn quad_residual(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n).map(|i| x[i] * x[i] + 0.1 * x[(i + 1) % n]).collect()
    }

    fn quad_jacobian_times(x: &[f64], v: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|i| 2.0 * x[i] * v[i] + 0.1 * v[(i + 1) % n])
            .collect()
    }

    /// MUST-HIT: the directional derivative agrees with the analytic Jacobian.
    /// MUST-MISS: it does NOT agree with a perturbed Jacobian, so the agreement is
    /// evidence about the operator and not about the tolerance being loose.
    #[test]
    fn the_finite_difference_operator_matches_the_analytic_jacobian() {
        let x: Vec<f64> = (0..8).map(|i| 1.0 + 0.1 * i as f64).collect();
        let f = quad_residual(&x);
        let mut j = KrylovJacobian::with_defaults(quad_residual);
        j.setup(&x, &f);

        let v = pseudo(77, 8);
        let got = j.matvec(&v);
        let want = quad_jacobian_times(&x, &v);
        let scale = want.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            max_diff(&got, &want) < 1e-6 * scale,
            "FD operator differs from the analytic Jacobian by {}",
            max_diff(&got, &want)
        );

        let wrong: Vec<f64> = want.iter().map(|v| v * 1.01).collect();
        assert!(
            max_diff(&got, &wrong) > 1e-4 * scale,
            "the FD operator matches a 1%-perturbed Jacobian equally well; the check \
             above cannot distinguish a correct operator from a wrong one"
        );
    }

    /// The differencing scale is SciPy's, and this recovers it exactly rather than
    /// trusting the code that computed it.
    ///
    /// For `F(x) = x .* x` the difference quotient is `2 x_i v_i + h v_i^2` with
    /// `h = omega / ||v||`, so the step actually taken can be solved for from the
    /// output and compared against `rdiff * max(1, ||x||_inf) / max(1, ||f||_inf)`.
    /// A test that recomputed the formula and compared it to itself would prove nothing.
    #[test]
    fn the_differencing_scale_is_the_scipy_expression() {
        let square = |x: &[f64]| -> Vec<f64> { x.iter().map(|v| v * v).collect() };
        // ||x||_inf = 3.0, ||f||_inf = 9.0, both above 1, so neither guard is active
        // and a formula that dropped either term would give a different answer.
        let x = vec![3.0, 1.0, -2.0, 0.5];
        let f = square(&x);
        // rdiff is set LARGE on purpose. The quantity being recovered is the
        // second-order remainder of the difference quotient, and at the default
        // sqrt(eps) it sits about 500x BELOW the cancellation noise of the quotient
        // itself -- the recovered value comes back with the wrong sign. The formula is
        // linear in rdiff, so measuring it at a step where it is resolvable pins it
        // exactly; the default value is pinned separately below, by bit-identity.
        let rdiff = 1e-2;
        let mut j = KrylovJacobian::new(square, Some(rdiff), 20, 10);
        j.setup(&x, &f);

        let v = vec![1.0, -1.0, 0.5, 2.0];
        let got = j.matvec(&v);
        let vnorm = norm_of(&v);
        let want_h = rdiff * 3.0_f64.max(1.0) / 9.0_f64.max(1.0) / vnorm;

        // Recover h from the component with the largest v_i^2, where the recovery is
        // best conditioned -- this is a difference of nearly equal numbers.
        let idx = (0..4).max_by(|&a, &b| v[a].abs().partial_cmp(&v[b].abs()).unwrap()).unwrap();
        let recovered = (got[idx] - 2.0 * x[idx] * v[idx]) / (v[idx] * v[idx]);
        assert!(
            (recovered - want_h).abs() < 1e-8 * want_h,
            "recovered step {recovered} does not match SciPy's {want_h}"
        );

        // MUST-MISS: the scale root.rs's newton_krylov uses is materially different, so
        // the check above is not passing on any plausible formula.
        let other_h = rdiff * (1.0 + norm_of(&x)) / vnorm;
        assert!(
            (recovered - other_h).abs() > 1e-2 * other_h,
            "SciPy's scale and the rdiff*(1+||x||_2)/||v|| scale are indistinguishable \
             on this input; the test does not pin the formula"
        );

        // The DEFAULT rdiff is sqrt(eps). Checked by bit-identity against an explicit
        // sqrt(eps) rather than by recovering the step, which is exactly the
        // measurement that fails at that magnitude.
        let mut defaulted = KrylovJacobian::with_defaults(square);
        defaulted.setup(&x, &f);
        let mut explicit = KrylovJacobian::new(square, Some(f64::EPSILON.sqrt()), 20, 10);
        explicit.setup(&x, &f);
        let a = defaulted.matvec(&v);
        let b = explicit.matvec(&v);
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "the default rdiff is not sqrt(eps)"
        );
        // MUST-MISS: a different rdiff gives a different answer, so bit-identity above
        // is evidence rather than a property of the operator being insensitive.
        let mut other = KrylovJacobian::new(square, Some(1e-3), 20, 10);
        other.setup(&x, &f);
        let c = other.matvec(&v);
        assert!(
            a.iter().zip(&c).any(|(p, q)| p.to_bits() != q.to_bits()),
            "changing rdiff did not change the product; the bit-identity check is vacuous"
        );
    }

    /// The inner solver actually solves. Driven on a LINEAR residual, where the
    /// finite-difference operator is exact, so any error is the Krylov method's.
    #[test]
    fn the_inner_solver_solves_the_newton_system() {
        // F(x) = A x with A strictly diagonally dominant, hence nonsingular.
        let n = 12;
        let amul = |x: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    4.0 * x[i] - x[(i + 1) % n] - 0.5 * x[(i + n - 1) % n]
                })
                .collect()
        };
        let x0 = vec![0.0; n];
        let f0 = amul(&x0);
        for method in [InnerMethod::Gmres, InnerMethod::Lgmres] {
            let mut j = KrylovJacobian::new(amul, None, 30, 10);
            j.setup(&x0, &f0);
            let rhs = pseudo(303, n);
            let dx = j.solve(&rhs, 1e-10, method);
            let back = amul(&dx);
            assert!(
                max_diff(&back, &rhs) < 1e-6,
                "{method:?}: J dx does not reproduce the right-hand side, off by {}",
                max_diff(&back, &rhs)
            );
            // MUST-MISS: a nonzero right-hand side must not be answered with zero.
            assert!(
                norm_of(&dx) > 1e-6,
                "{method:?}: returned an all-but-zero step for a nonzero rhs"
            );
        }
    }

    /// The augmentation is carried across Newton steps and capped at `outer_k`.
    /// Plain GMRES must carry none, or the two modes are the same code.
    #[test]
    fn the_augmentation_is_carried_and_capped() {
        let n = 10;
        let mut x: Vec<f64> = (0..n).map(|i| 1.0 + 0.05 * i as f64).collect();
        let mut f = quad_residual(&x);
        let mut j = KrylovJacobian::new(quad_residual, None, 8, 3);
        j.setup(&x, &f);

        for step in 0..6 {
            let rhs: Vec<f64> = f.iter().map(|v| -v).collect();
            let dx = j.solve(&rhs, 1e-6, InnerMethod::Lgmres);
            for (xi, di) in x.iter_mut().zip(&dx) {
                *xi += di;
            }
            f = quad_residual(&x);
            j.update(&x, &f);
            assert!(
                j.augmentation_rank() <= 3,
                "step {step}: augmentation rank {} exceeded outer_k = 3",
                j.augmentation_rank()
            );
        }
        assert_eq!(
            j.augmentation_rank(),
            3,
            "the augmentation never filled up; it is not being carried across steps"
        );

        // MUST-MISS: the plain GMRES mode accumulates nothing.
        let mut g = KrylovJacobian::new(quad_residual, None, 8, 3);
        let x2: Vec<f64> = (0..n).map(|i| 1.0 + 0.05 * i as f64).collect();
        let f2 = quad_residual(&x2);
        g.setup(&x2, &f2);
        for _ in 0..6 {
            let rhs: Vec<f64> = f2.iter().map(|v| -v).collect();
            g.solve(&rhs, 1e-6, InnerMethod::Gmres);
        }
        assert_eq!(
            g.augmentation_rank(),
            0,
            "the plain GMRES mode accumulated augmentation directions"
        );
    }

    /// Every product costs exactly one residual evaluation, and a zero direction costs
    /// none. That count is the real cost model for a matrix-free method, so it is worth
    /// pinning rather than inferring.
    #[test]
    fn products_cost_one_evaluation_and_zero_directions_cost_none() {
        let x = vec![1.0, 2.0, 3.0];
        let f = quad_residual(&x);
        let mut j = KrylovJacobian::with_defaults(quad_residual);
        j.setup(&x, &f);
        assert_eq!(j.function_evaluations(), 0, "setup should evaluate nothing");

        j.matvec(&[1.0, 0.0, 0.0]);
        assert_eq!(j.function_evaluations(), 1);
        j.matvec(&[0.0, 1.0, 1.0]);
        assert_eq!(j.function_evaluations(), 2);

        // MUST-MISS the counter: a zero direction short-circuits.
        let z = j.matvec(&[0.0, 0.0, 0.0]);
        assert_eq!(
            j.function_evaluations(),
            2,
            "a zero direction consumed an evaluation"
        );
        assert!(z.iter().all(|v| *v == 0.0), "a zero direction gave a nonzero product");
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Simple mixing Jacobians — scipy.optimize LinearMixing / DiagBroyden / ExcitingMixing
// ─────────────────────────────────────────────────────────────────────────────

/// Shared auto-scaling for the mixing methods.
///
/// `GenericBroyden.setup`'s heuristic: `alpha = 0.5 * max(||x0||, 1) / ||f0||`, or 1
/// when the residual already vanishes. Every method below inherits it, so it lives here
/// rather than being written out four times with four chances to diverge.
fn auto_alpha(x0: &[f64], f0: &[f64]) -> f64 {
    let nf = norm(f0);
    if nf != 0.0 {
        0.5 * norm(x0).max(1.0) / nf
    } else {
        1.0
    }
}

/// Scalar Jacobian approximation `J = -1/alpha * I` -- `scipy.optimize.LinearMixing`.
///
/// The Jacobian never changes; `update` is genuinely a no-op, which is not an oversight
/// but the definition of the method. It is the crudest member of the family and is here
/// because it is the right baseline: an adaptive method that fails to beat plain linear
/// mixing on a problem is not earning its bookkeeping there.
#[derive(Debug, Clone)]
pub struct LinearMixingJacobian {
    alpha: Option<f64>,
    n: usize,
}

impl LinearMixingJacobian {
    /// `alpha = None` auto-scales on setup.
    #[must_use]
    pub fn new(alpha: Option<f64>) -> Self {
        Self { alpha, n: 0 }
    }
    fn a(&self) -> f64 {
        self.alpha.unwrap_or(1.0)
    }
}

impl Jacobian for LinearMixingJacobian {
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        if self.alpha.is_none() {
            self.alpha = Some(auto_alpha(x0, f0));
        }
    }
    fn solve_ref(&self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| -x * self.a()).collect()
    }
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|x| -x / self.a()).collect()
    }
    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        self.solve_ref(v)
    }
    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        self.matvec(v)
    }
    /// Deliberately empty: the whole point of linear mixing is a fixed Jacobian.
    fn update(&mut self, _x: &[f64], _f: &[f64]) {}
    fn dimension(&self) -> usize {
        self.n
    }
}

/// Diagonal Jacobian approximation `J = -diag(d)` -- `scipy.optimize.DiagBroyden`.
///
/// Broyden's rank-1 update restricted to the diagonal: each entry absorbs only the part
/// of the secant condition its own coordinate explains. That buys O(n) memory and O(n)
/// work per step, and buys nothing at all on a problem whose coupling is off the
/// diagonal -- the honest limit of the method rather than a defect of it.
#[derive(Debug, Clone)]
pub struct DiagBroydenJacobian {
    alpha: Option<f64>,
    d: Vec<f64>,
    last_x: Vec<f64>,
    last_f: Vec<f64>,
    n: usize,
}

impl DiagBroydenJacobian {
    /// `alpha = None` auto-scales on setup.
    #[must_use]
    pub fn new(alpha: Option<f64>) -> Self {
        Self {
            alpha,
            d: Vec::new(),
            last_x: Vec::new(),
            last_f: Vec::new(),
            n: 0,
        }
    }

    /// The current diagonal.
    #[must_use]
    pub fn diagonal(&self) -> &[f64] {
        &self.d
    }
}

impl Jacobian for DiagBroydenJacobian {
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        if self.alpha.is_none() {
            self.alpha = Some(auto_alpha(x0, f0));
        }
        self.d = vec![1.0 / self.alpha.unwrap_or(1.0); self.n];
        self.last_x = x0.to_vec();
        self.last_f = f0.to_vec();
    }
    fn solve_ref(&self, v: &[f64]) -> Vec<f64> {
        v.iter().zip(&self.d).map(|(x, d)| -x / d).collect()
    }
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().zip(&self.d).map(|(x, d)| -x * d).collect()
    }
    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        self.solve_ref(v)
    }
    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        self.matvec(v)
    }
    fn update(&mut self, x: &[f64], f: &[f64]) {
        if x.len() != self.n || f.len() != self.n {
            return;
        }
        let dx: Vec<f64> = x.iter().zip(&self.last_x).map(|(a, b)| a - b).collect();
        let df: Vec<f64> = f.iter().zip(&self.last_f).map(|(a, b)| a - b).collect();
        let dx_norm2: f64 = dx.iter().map(|v| v * v).sum();
        // A zero step carries no information and would divide by zero.
        if dx_norm2 > 0.0 && dx_norm2.is_finite() {
            for i in 0..self.n {
                self.d[i] -= (df[i] + self.d[i] * dx[i]) * dx[i] / dx_norm2;
            }
        }
        self.last_x = x.to_vec();
        self.last_f = f.to_vec();
    }
    fn dimension(&self) -> usize {
        self.n
    }
}

/// Diagonal Jacobian with a per-coordinate adaptive step
/// -- `scipy.optimize.ExcitingMixing`.
///
/// A coordinate whose residual KEEPS ITS SIGN is being approached steadily, so its step
/// grows by `alpha`; one whose residual flips sign has overshot, so its step is reset to
/// `alpha` outright. That asymmetry -- grow slowly, reset hard -- is the whole method,
/// and it is a heuristic rather than an approximation of anything: no secant condition
/// is involved, which is why it has no convergence theory and is judged only by results.
#[derive(Debug, Clone)]
pub struct ExcitingMixingJacobian {
    alpha: Option<f64>,
    alphamax: f64,
    beta: Vec<f64>,
    last_f: Vec<f64>,
    n: usize,
}

impl ExcitingMixingJacobian {
    /// `alpha = None` auto-scales on setup; `alphamax` defaults to 1.0.
    #[must_use]
    pub fn new(alpha: Option<f64>, alphamax: Option<f64>) -> Self {
        Self {
            alpha,
            alphamax: alphamax.unwrap_or(1.0),
            beta: Vec::new(),
            last_f: Vec::new(),
            n: 0,
        }
    }

    /// The current per-coordinate steps.
    #[must_use]
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }
}

impl Jacobian for ExcitingMixingJacobian {
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        if self.alpha.is_none() {
            self.alpha = Some(auto_alpha(x0, f0));
        }
        self.beta = vec![self.alpha.unwrap_or(1.0); self.n];
        self.last_f = f0.to_vec();
    }
    fn solve_ref(&self, v: &[f64]) -> Vec<f64> {
        v.iter().zip(&self.beta).map(|(x, b)| -x * b).collect()
    }
    fn matvec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().zip(&self.beta).map(|(x, b)| -x / b).collect()
    }
    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        self.solve_ref(v)
    }
    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        self.matvec(v)
    }
    fn update(&mut self, _x: &[f64], f: &[f64]) {
        if f.len() != self.n {
            return;
        }
        let alpha = self.alpha.unwrap_or(1.0);
        for i in 0..self.n {
            // Compared against the PREVIOUS residual, which is why `last_f` is
            // refreshed only after the whole sweep.
            if f[i] * self.last_f[i] > 0.0 {
                self.beta[i] += alpha;
            } else {
                self.beta[i] = alpha;
            }
            self.beta[i] = self.beta[i].clamp(0.0, self.alphamax);
        }
        self.last_f = f.to_vec();
    }
    fn dimension(&self) -> usize {
        self.n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Anderson mixing
// ─────────────────────────────────────────────────────────────────────────────

/// Anderson mixing -- `scipy.optimize.Anderson`.
///
/// Rather than maintaining a Jacobian, this keeps the last `m` steps and residual
/// changes and, at each solve, picks the combination of them that best cancels the
/// current residual in a least-squares sense. It is Anderson acceleration in the shape
/// of a Jacobian object, and it is often the strongest of these methods where the
/// Jacobian is badly conditioned but the iterates lie near a low-dimensional manifold.
///
/// # `w0` regularises a problem that is rank-deficient exactly when it matters
///
/// The normal-equations matrix is `a[i][j] = (1 + w0^2 delta_ij) <df_i, df_j>`. Stored
/// residual differences become nearly parallel as the iteration converges -- that is
/// what converging means -- so `a` approaches singular precisely when the method is
/// working. The `w0^2` ridge on the diagonal is what keeps the solve meaningful there.
/// When it fails anyway the history is DISCARDED and the step falls back to plain
/// linear mixing, which is SciPy's behaviour and the right one: a stale history is worse
/// than no history.
#[derive(Debug, Clone)]
pub struct AndersonJacobian {
    alpha: Option<f64>,
    w0: f64,
    m: usize,
    dxs: Vec<Vec<f64>>,
    dfs: Vec<Vec<f64>>,
    /// Normal-equations matrix, row-major, rebuilt on every update.
    a: Vec<f64>,
    last_x: Vec<f64>,
    last_f: Vec<f64>,
    n: usize,
}

impl AndersonJacobian {
    /// SciPy's defaults are `w0 = 0.01` and `m = 5`; `alpha = None` auto-scales.
    #[must_use]
    pub fn new(alpha: Option<f64>, w0: Option<f64>, m: Option<usize>) -> Self {
        Self {
            alpha,
            w0: w0.unwrap_or(0.01),
            m: m.unwrap_or(5),
            dxs: Vec::new(),
            dfs: Vec::new(),
            a: Vec::new(),
            last_x: Vec::new(),
            last_f: Vec::new(),
            n: 0,
        }
    }

    /// Number of retained history pairs.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.dxs.len()
    }

    fn alpha_v(&self) -> f64 {
        self.alpha.unwrap_or(1.0)
    }

    /// `b[i][j] = <df_i, dx_j> - delta_ij <df_i, df_i> w0^2 alpha`, the matrix the
    /// FORWARD product uses. Unlike `a` it is NOT symmetric, which is why the transpose
    /// operation below solves with its transpose instead of reusing the same matrix.
    fn b_matrix(&self) -> Vec<f64> {
        let k = self.dxs.len();
        let mut b = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut v: f64 = self.dfs[i]
                    .iter()
                    .zip(&self.dxs[j])
                    .map(|(p, q)| p * q)
                    .sum();
                if i == j && self.w0 != 0.0 {
                    let dd: f64 = self.dfs[i].iter().map(|p| p * p).sum();
                    v -= dd * self.w0 * self.w0 * self.alpha_v();
                }
                b[i * k + j] = v;
            }
        }
        b
    }

    fn dots_with(vecs: &[Vec<f64>], f: &[f64]) -> Vec<f64> {
        vecs.iter()
            .map(|v| v.iter().zip(f).map(|(p, q)| p * q).sum())
            .collect()
    }
}

impl Jacobian for AndersonJacobian {
    fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        if self.alpha.is_none() {
            self.alpha = Some(auto_alpha(x0, f0));
        }
        self.dxs.clear();
        self.dfs.clear();
        self.a.clear();
        self.last_x = x0.to_vec();
        self.last_f = f0.to_vec();
    }

    fn solve_ref(&self, f: &[f64]) -> Vec<f64> {
        let alpha = self.alpha_v();
        let mut dx: Vec<f64> = f.iter().map(|v| -alpha * v).collect();
        let k = self.dxs.len();
        if k == 0 {
            return dx;
        }
        let rhs = Self::dots_with(&self.dfs, f);
        let mut a = self.a.clone();
        // A singular history falls back to plain linear mixing rather than inventing a
        // step; the `&mut` path additionally clears it.
        let Some(gamma) = lu_solve_in_place(&mut a, &rhs, k) else {
            return dx;
        };
        for m in 0..k {
            let g = gamma[m];
            for i in 0..self.n {
                dx[i] += g * (self.dxs[m][i] + alpha * self.dfs[m][i]);
            }
        }
        dx
    }

    /// As `solve_ref`, but discards a history that has gone singular so the next step
    /// starts clean -- SciPy resets the same way.
    fn solve(&mut self, f: &[f64]) -> Vec<f64> {
        let k = self.dxs.len();
        if k > 0 {
            let rhs = Self::dots_with(&self.dfs, f);
            let mut a = self.a.clone();
            if lu_solve_in_place(&mut a, &rhs, k).is_none() {
                self.dxs.clear();
                self.dfs.clear();
                self.a.clear();
            }
        }
        self.solve_ref(f)
    }

    fn matvec(&self, f: &[f64]) -> Vec<f64> {
        let alpha = self.alpha_v();
        let mut dx: Vec<f64> = f.iter().map(|v| -v / alpha).collect();
        let k = self.dxs.len();
        if k == 0 {
            return dx;
        }
        let rhs = Self::dots_with(&self.dfs, f);
        let mut b = self.b_matrix();
        let Some(gamma) = lu_solve_in_place(&mut b, &rhs, k) else {
            return dx;
        };
        for m in 0..k {
            let g = gamma[m];
            for i in 0..self.n {
                dx[i] += g * (self.dfs[m][i] + self.dxs[m][i] / alpha);
            }
        }
        dx
    }

    /// `(J^-1)^T v`.
    ///
    /// SciPy does not define this, so it is DERIVED rather than ported, and the tests
    /// check it against an explicit dense transpose instead of against a formula.
    /// Writing the inverse as `-alpha I + U a^-1 D^T` with `U = [dx_m + alpha df_m]` and
    /// `D = [df_m]`, its transpose is `-alpha I + D a^-T U^T`; `a` is SYMMETRIC, so the
    /// same matrix serves both directions.
    fn rsolve(&self, v: &[f64]) -> Vec<f64> {
        let alpha = self.alpha_v();
        let mut out: Vec<f64> = v.iter().map(|x| -alpha * x).collect();
        let k = self.dxs.len();
        if k == 0 {
            return out;
        }
        let u: Vec<Vec<f64>> = (0..k)
            .map(|m| {
                self.dxs[m]
                    .iter()
                    .zip(&self.dfs[m])
                    .map(|(a, b)| a + alpha * b)
                    .collect()
            })
            .collect();
        let rhs = Self::dots_with(&u, v);
        let mut a = self.a.clone();
        let Some(gamma) = lu_solve_in_place(&mut a, &rhs, k) else {
            return out;
        };
        for m in 0..k {
            let g = gamma[m];
            for i in 0..self.n {
                out[i] += g * self.dfs[m][i];
            }
        }
        out
    }

    /// `J^T v`, derived the same way from `-I/alpha + V b^-1 D^T` with
    /// `V = [df_m + dx_m/alpha]`. Here `b` is NOT symmetric, so this solves against its
    /// transpose rather than reusing the forward matrix -- the one place the two
    /// directions genuinely differ.
    fn rmatvec(&self, v: &[f64]) -> Vec<f64> {
        let alpha = self.alpha_v();
        let mut out: Vec<f64> = v.iter().map(|x| -x / alpha).collect();
        let k = self.dxs.len();
        if k == 0 {
            return out;
        }
        let vmat: Vec<Vec<f64>> = (0..k)
            .map(|m| {
                self.dfs[m]
                    .iter()
                    .zip(&self.dxs[m])
                    .map(|(a, b)| a + b / alpha)
                    .collect()
            })
            .collect();
        let rhs = Self::dots_with(&vmat, v);
        let b = self.b_matrix();
        let mut bt = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                bt[i * k + j] = b[j * k + i];
            }
        }
        let Some(gamma) = lu_solve_in_place(&mut bt, &rhs, k) else {
            return out;
        };
        for m in 0..k {
            let g = gamma[m];
            for i in 0..self.n {
                out[i] += g * self.dfs[m][i];
            }
        }
        out
    }

    fn update(&mut self, x: &[f64], f: &[f64]) {
        if x.len() != self.n || f.len() != self.n || self.m == 0 {
            return;
        }
        let dx: Vec<f64> = x.iter().zip(&self.last_x).map(|(a, b)| a - b).collect();
        let df: Vec<f64> = f.iter().zip(&self.last_f).map(|(a, b)| a - b).collect();
        self.last_x = x.to_vec();
        self.last_f = f.to_vec();

        self.dxs.push(dx);
        self.dfs.push(df);
        while self.dxs.len() > self.m {
            self.dxs.remove(0);
            self.dfs.remove(0);
        }

        let k = self.dxs.len();
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let wd = if i == j { self.w0 * self.w0 } else { 0.0 };
                let d: f64 = self.dfs[i]
                    .iter()
                    .zip(&self.dfs[j])
                    .map(|(p, q)| p * q)
                    .sum();
                a[i * k + j] = (1.0 + wd) * d;
            }
        }
        self.a = a;
    }

    fn dimension(&self) -> usize {
        self.n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KrylovJacobian
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix-free Jacobian applied by finite differences -- `scipy.optimize.KrylovJacobian`.
///
/// The Jacobian is never formed. `J v` is estimated as a directional derivative of the
/// residual, and the Newton system is solved by an inner Krylov method that only ever
/// asks for products. Memory is O(n * inner_maxiter) regardless of how dense the true
/// Jacobian is, which is what makes this the method of choice for discretised PDEs.
///
/// # The differencing scale is the whole numerical difficulty
///
/// A directional derivative `(F(x + h v) - F(x)) / h` is a fight between truncation
/// error, which falls with `h`, and cancellation, which grows as `h` shrinks and the two
/// residuals agree to more digits. SciPy picks
///
/// ```text
///     omega = rdiff * max(1, ||x||_inf) / max(1, ||f||_inf),   h = omega / ||v||
/// ```
///
/// with `rdiff = sqrt(eps)`, and this implementation uses the same expression. The two
/// guards matter and are easy to drop: dividing by `max(1, ||f||_inf)` shrinks the step
/// when the residual is large, where the difference would otherwise be swamped, and
/// scaling by `1/||v||` makes the step invariant to the length of the direction the
/// Krylov method happens to hand over -- without it the estimate degrades as the basis
/// vectors change scale.
///
/// `root.rs`'s existing `newton_krylov` uses `rdiff * (1 + ||x||_2) / ||v||`, which drops
/// the residual term and uses a different norm. This type exists partly to carry SciPy's
/// actual choice.
///
/// # Why this does not implement the `Jacobian` trait
///
/// The trait requires `rmatvec` and `rsolve`. A finite-difference operator has NO
/// transpose: `J^T v` is not a directional derivative of `F` in any direction, and
/// producing one would need a second, different approximation the caller did not ask
/// for. Implementing them to satisfy a signature -- by returning the forward result, or
/// zeros, or by panicking at runtime -- would each be worse than not offering them.
/// SciPy's `KrylovJacobian` likewise defines only `matvec` and `solve`.
pub struct KrylovJacobian<F> {
    func: F,
    x0: Vec<f64>,
    f0: Vec<f64>,
    rdiff: f64,
    omega: f64,
    inner_maxiter: usize,
    outer_k: usize,
    /// Augmentation directions carried across outer Newton steps -- LGMRES's `outer_v`.
    outer_v: Vec<Vec<f64>>,
    /// Residual evaluations consumed, the honest cost measure for a matrix-free method.
    nfev: usize,
    n: usize,
}

/// Which inner Krylov method solves the Newton system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerMethod {
    /// Restarted GMRES with no augmentation.
    Gmres,
    /// GMRES augmented with directions carried over from previous Newton steps.
    ///
    /// The augmentation is the point: a restarted Krylov method throws away everything
    /// it learned at each restart, and across a Newton iteration it throws away
    /// everything it learned about a Jacobian that has barely changed. Keeping a few
    /// directions recovers the components a restart would otherwise have to rediscover.
    Lgmres,
}

impl<F> KrylovJacobian<F>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    /// `rdiff` defaults to `sqrt(f64::EPSILON)`, `inner_maxiter` to 20 and `outer_k` to
    /// 10, matching SciPy.
    pub fn new(func: F, rdiff: Option<f64>, inner_maxiter: usize, outer_k: usize) -> Self {
        Self {
            func,
            x0: Vec::new(),
            f0: Vec::new(),
            rdiff: rdiff.unwrap_or_else(|| f64::EPSILON.sqrt()),
            omega: 0.0,
            inner_maxiter,
            outer_k,
            outer_v: Vec::new(),
            nfev: 0,
            n: 0,
        }
    }

    /// SciPy's defaults throughout.
    pub fn with_defaults(func: F) -> Self {
        Self::new(func, None, 20, 10)
    }

    /// Residual evaluations consumed so far.
    ///
    /// For a matrix-free method this, not wall time, is the cost that matters: every
    /// Krylov product is one call to `F`, and on the problems this method is for that
    /// call dominates everything else by orders of magnitude.
    pub fn function_evaluations(&self) -> usize {
        self.nfev
    }

    /// Number of augmentation directions currently carried.
    pub fn augmentation_rank(&self) -> usize {
        self.outer_v.len()
    }

    fn update_diff_step(&mut self) {
        let mx = self.x0.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let mf = self.f0.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        self.omega = self.rdiff * mx.max(1.0) / mf.max(1.0);
    }

    /// Prepare for a solve at `x0` with residual `f0`.
    pub fn setup(&mut self, x0: &[f64], f0: &[f64]) {
        self.n = x0.len();
        self.x0 = x0.to_vec();
        self.f0 = f0.to_vec();
        self.outer_v.clear();
        self.update_diff_step();
    }

    /// Move to a new point. The augmentation directions are DELIBERATELY kept.
    ///
    /// SciPy carries `outer_v` across nonlinear steps but explicitly does not carry the
    /// matching `A v` products, because the Jacobian may have moved. This does the same:
    /// the directions are reused, their images are recomputed against the current
    /// Jacobian. Reusing stale products would be the cheap-looking mistake -- it saves
    /// the evaluations that make the augmentation correct.
    pub fn update(&mut self, x: &[f64], f: &[f64]) {
        self.x0 = x.to_vec();
        self.f0 = f.to_vec();
        self.update_diff_step();
    }

    /// `J v`, by a directional derivative of the residual.
    ///
    /// Costs exactly one residual evaluation. A zero direction short-circuits: the
    /// derivative is zero and the scaling would divide by zero.
    pub fn matvec(&mut self, v: &[f64]) -> Vec<f64> {
        let nv = norm(v);
        if nv == 0.0 {
            return vec![0.0; self.n];
        }
        let sc = self.omega / nv;
        let xp: Vec<f64> = self.x0.iter().zip(v).map(|(a, b)| a + sc * b).collect();
        let fp = (self.func)(&xp);
        self.nfev += 1;
        fp.iter()
            .zip(&self.f0)
            .map(|(a, b)| (a - b) / sc)
            .collect()
    }

    /// Solve `J dx = rhs` with the inner Krylov method, to relative tolerance `rtol`.
    ///
    /// Returns the step and leaves the augmentation updated when running LGMRES.
    pub fn solve(&mut self, rhs: &[f64], rtol: f64, method: InnerMethod) -> Vec<f64> {
        let augment = match method {
            InnerMethod::Gmres => Vec::new(),
            InnerMethod::Lgmres => self.outer_v.clone(),
        };
        let dx = self.gcr_solve(rhs, rtol, &augment);
        if method == InnerMethod::Lgmres && self.outer_k > 0 {
            let nd = norm(&dx);
            if nd > 0.0 && nd.is_finite() {
                self.outer_v.push(dx.iter().map(|v| v / nd).collect());
                while self.outer_v.len() > self.outer_k {
                    self.outer_v.remove(0);
                }
            }
        }
        dx
    }

    /// Generalised conjugate residual, optionally seeded with augmentation directions.
    ///
    /// GCR rather than an Arnoldi GMRES because augmentation drops straight in: a
    /// direction is a direction, whether it came from the current Krylov sequence or
    /// from a previous Newton step, so the augmented subspace needs no special basis
    /// bookkeeping. Over the same subspace GCR minimises the same residual GMRES does.
    ///
    /// The paired update is what keeps it honest: whenever `A z` is orthogonalised
    /// against an earlier image, the SAME coefficient is applied to `z`, so the stored
    /// pair always satisfies `q_j = A z_j` exactly and the residual bookkeeping stays
    /// consistent with the operator.
    fn gcr_solve(&mut self, rhs: &[f64], rtol: f64, augment: &[Vec<f64>]) -> Vec<f64> {
        let n = rhs.len();
        let mut x = vec![0.0; n];
        let mut r = rhs.to_vec();
        let rhs_norm = norm(rhs);
        if rhs_norm == 0.0 {
            return x;
        }
        let target = rtol * rhs_norm;

        let mut zs: Vec<Vec<f64>> = Vec::new();
        let mut qs: Vec<Vec<f64>> = Vec::new();

        for j in 0..self.inner_maxiter {
            // Seed with an augmentation direction while any remain, then fall back to
            // the current residual -- the ordinary Krylov choice.
            let mut z = if j < augment.len() {
                augment[j].clone()
            } else {
                r.clone()
            };
            let mut w = self.matvec(&z);

            // Modified Gram-Schmidt against the previous images, applying every
            // coefficient to `z` as well so the pair stays exact.
            for i in 0..qs.len() {
                let beta: f64 = qs[i].iter().zip(&w).map(|(a, b)| a * b).sum();
                if beta != 0.0 {
                    for (wi, qi) in w.iter_mut().zip(&qs[i]) {
                        *wi -= beta * qi;
                    }
                    for (zi, zzi) in z.iter_mut().zip(&zs[i]) {
                        *zi -= beta * zzi;
                    }
                }
            }
            let nw = norm(&w);
            if nw == 0.0 || !nw.is_finite() {
                // The direction added nothing; with an augmentation seed that just means
                // it was already in the space, so continue rather than give up.
                if j < augment.len() {
                    continue;
                }
                break;
            }
            for wi in w.iter_mut() {
                *wi /= nw;
            }
            for zi in z.iter_mut() {
                *zi /= nw;
            }

            let alpha: f64 = w.iter().zip(&r).map(|(a, b)| a * b).sum();
            for (xi, zi) in x.iter_mut().zip(&z) {
                *xi += alpha * zi;
            }
            for (ri, wi) in r.iter_mut().zip(&w) {
                *ri -= alpha * wi;
            }
            zs.push(z);
            qs.push(w);

            if norm(&r) <= target {
                break;
            }
        }
        x
    }
}
