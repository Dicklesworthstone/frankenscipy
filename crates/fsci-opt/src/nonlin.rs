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
