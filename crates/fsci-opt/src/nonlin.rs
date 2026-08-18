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
}
