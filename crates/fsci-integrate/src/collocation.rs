#![forbid(unsafe_code)]

//! The collocation algorithm behind [`crate::solve_bvp`]: SciPy's `scipy.integrate._bvp`
//! (Kierzenka & Shampine, "A BVP Solver Based on Residual Control and the MATLAB PSE", ACM TOMS
//! 27(3), 2001), transcribed function by function.
//!
//! The solution is a C¹ cubic spline whose derivative matches the ODE right-hand side at the
//! mesh nodes and at the interval midpoints (4th-order Lobatto IIIA collocation). The
//! collocation equations plus the boundary conditions are solved by a damped Newton method on a
//! sparse Jacobian (affine-invariant criterion |J⁻¹r|², Armijo backtracking, the Jacobian kept
//! across full steps); then the RMS relative residual of the spline on each interval is
//! estimated by 5-point Lobatto quadrature and 1 node (tol < r < 100·tol) or 2 nodes
//! (r ≥ 100·tol) are inserted until every interval meets `tol` or `max_nodes` would be exceeded.
//!
//! Arrays are node-major (`y[j·n + i]` is component `i` at node `j`), which is SciPy's
//! `ravel(order='F')` of its `(n, m)` arrays, so the Newton vector has SciPy's layout. The
//! sparse LU is supplied by the caller through [`LinearSolver`].

/// A factorized square matrix.
pub(crate) trait LinearSolver {
    /// `A⁻¹·b`, or `None` when the factorization cannot solve (singular).
    fn solve(&self, b: &[f64]) -> Option<Vec<f64>>;
}

/// Factorizes the `size×size` matrix given by `(row, col, value)` triplets; `None` = singular.
pub(crate) type Factorize<'a> =
    &'a dyn Fn(usize, &[(usize, usize, f64)]) -> Option<Box<dyn LinearSolver>>;

/// `f(x, y, p)` at one point.
pub(crate) type Rhs<'a> = &'a dyn Fn(f64, &[f64], &[f64]) -> Vec<f64>;
/// `bc(ya, yb, p)`: n + k residuals.
pub(crate) type Boundary<'a> = &'a dyn Fn(&[f64], &[f64], &[f64]) -> Vec<f64>;
/// `∂f/∂y` (n×n, row-major) and `∂f/∂p` (n×k, row-major) at one point.
pub(crate) type FunJac<'a> = &'a dyn Fn(f64, &[f64], &[f64]) -> (Vec<f64>, Vec<f64>);
/// `(∂bc/∂ya, ∂bc/∂yb, ∂bc/∂p)`: (n+k)×n, (n+k)×n and (n+k)×k, row-major.
pub(crate) type BcJacobian = (Vec<f64>, Vec<f64>, Vec<f64>);
pub(crate) type BcJac<'a> = &'a dyn Fn(&[f64], &[f64], &[f64]) -> BcJacobian;

/// The singular term `S·y/(x − a)`: `b = I − S⁺S` (imposes `S·y(a) = 0`) and `d = (I − S)⁺`
/// (corrects `y'(a)`), all n×n row-major.
pub(crate) struct Singular {
    pub s: Vec<f64>,
    pub b: Vec<f64>,
    pub d: Vec<f64>,
}

pub(crate) struct Problem<'a> {
    pub n: usize,
    pub k: usize,
    pub a: f64,
    pub fun: Rhs<'a>,
    pub bc: Boundary<'a>,
    pub fun_jac: Option<FunJac<'a>>,
    pub bc_jac: Option<BcJac<'a>>,
    pub singular: Option<Singular>,
}

pub(crate) struct Outcome {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yp: Vec<f64>,
    pub p: Vec<f64>,
    pub rms_residuals: Vec<f64>,
    pub niter: usize,
    pub status: usize,
}

/// SciPy's `TERMINATION_MESSAGES`.
pub(crate) fn termination_message(status: usize) -> &'static str {
    match status {
        0 => "The algorithm converged to the desired accuracy.",
        1 => "The maximum number of mesh nodes is exceeded.",
        2 => "A singular Jacobian encountered when solving the collocation system.",
        _ => "The solver was unable to satisfy boundary conditions tolerance on iteration 10.",
    }
}

const SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

fn matvec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * v[j]).sum())
        .collect()
}

impl Problem<'_> {
    /// f(x, y, p) including the singular term; `x == a` takes `D·f`.
    fn f(&self, x: f64, y: &[f64], p: &[f64]) -> Result<Vec<f64>, String> {
        let mut f = (self.fun)(x, y, p);
        if f.len() != self.n {
            return Err(format!(
                "`fun` returned {} values, expected {}",
                f.len(),
                self.n
            ));
        }
        if let Some(sing) = &self.singular {
            if x == self.a {
                f = matvec(&sing.d, &f, self.n);
            } else {
                let sy = matvec(&sing.s, y, self.n);
                for (fi, syi) in f.iter_mut().zip(sy) {
                    *fi += syi / (x - self.a);
                }
            }
        }
        Ok(f)
    }

    fn bc(&self, ya: &[f64], yb: &[f64], p: &[f64]) -> Result<Vec<f64>, String> {
        let r = (self.bc)(ya, yb, p);
        if r.len() != self.n + self.k {
            return Err(format!(
                "`bc` returned {} values, expected {}",
                r.len(),
                self.n + self.k
            ));
        }
        Ok(r)
    }

    /// `∂f/∂y` (n×n) and `∂f/∂p` (n×k) at one point, forward differences as
    /// `estimate_fun_jac` when no analytic Jacobian is given.
    fn f_jac(
        &self,
        x: f64,
        y: &[f64],
        p: &[f64],
        f0: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let (n, k) = (self.n, self.k);
        if let Some(jac) = self.fun_jac {
            let (mut dfdy, dfdp) = jac(x, y, p);
            if dfdy.len() != n * n || dfdp.len() != n * k {
                return Err(format!(
                    "`fun_jac` must return {n}x{n} and {n}x{k} matrices"
                ));
            }
            if let Some(sing) = &self.singular {
                if x == self.a {
                    let mut out = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            out[i * n + j] =
                                (0..n).map(|l| sing.d[i * n + l] * dfdy[l * n + j]).sum();
                        }
                    }
                    dfdy = out;
                } else {
                    for (v, s) in dfdy.iter_mut().zip(&sing.s) {
                        *v += s / (x - self.a);
                    }
                }
            }
            return Ok((dfdy, dfdp));
        }
        let mut dfdy = vec![0.0; n * n];
        let mut y_new = y.to_vec();
        for j in 0..n {
            let h = SQRT_EPS * (1.0 + y[j].abs());
            y_new[j] = y[j] + h;
            let hj = y_new[j] - y[j];
            let f_new = self.f(x, &y_new, p)?;
            for i in 0..n {
                dfdy[i * n + j] = (f_new[i] - f0[i]) / hj;
            }
            y_new[j] = y[j];
        }
        let mut dfdp = vec![0.0; n * k];
        let mut p_new = p.to_vec();
        for l in 0..k {
            let h = SQRT_EPS * (1.0 + p[l].abs());
            p_new[l] = p[l] + h;
            let hl = p_new[l] - p[l];
            let f_new = self.f(x, y, &p_new)?;
            for i in 0..n {
                dfdp[i * k + l] = (f_new[i] - f0[i]) / hl;
            }
            p_new[l] = p[l];
        }
        Ok((dfdy, dfdp))
    }

    /// `estimate_bc_jac`: `(∂bc/∂ya, ∂bc/∂yb, ∂bc/∂p)`, row-major with n+k rows.
    fn bc_jacobian(
        &self,
        ya: &[f64],
        yb: &[f64],
        p: &[f64],
        bc0: &[f64],
    ) -> Result<BcJacobian, String> {
        let (n, k) = (self.n, self.k);
        let rows = n + k;
        if let Some(jac) = self.bc_jac {
            let (dya, dyb, dp) = jac(ya, yb, p);
            if dya.len() != rows * n || dyb.len() != rows * n || dp.len() != rows * k {
                return Err(format!(
                    "`bc_jac` must return {rows}x{n}, {rows}x{n} and {rows}x{k} matrices"
                ));
            }
            return Ok((dya, dyb, dp));
        }
        let column = |which: usize, j: usize| -> Result<Vec<f64>, String> {
            let (mut ya2, mut yb2, mut p2) = (ya.to_vec(), yb.to_vec(), p.to_vec());
            let target = match which {
                0 => &mut ya2[j],
                1 => &mut yb2[j],
                _ => &mut p2[j],
            };
            let base = *target;
            *target = base + SQRT_EPS * (1.0 + base.abs());
            let h = *target - base;
            let r = self.bc(&ya2, &yb2, &p2)?;
            Ok(r.iter().zip(bc0).map(|(a, b)| (a - b) / h).collect())
        };
        let mut dya = vec![0.0; rows * n];
        let mut dyb = vec![0.0; rows * n];
        let mut dp = vec![0.0; rows * k];
        for j in 0..n {
            let ca = column(0, j)?;
            let cb = column(1, j)?;
            for b in 0..rows {
                dya[b * n + j] = ca[b];
                dyb[b * n + j] = cb[b];
            }
        }
        for l in 0..k {
            let cp = column(2, l)?;
            for b in 0..rows {
                dp[b * k + l] = cp[b];
            }
        }
        Ok((dya, dyb, dp))
    }
}

/// `collocation_fun`: residuals, midpoint values, and f at nodes and midpoints.
struct Collocation {
    col_res: Vec<f64>,
    y_middle: Vec<f64>,
    f: Vec<f64>,
    f_middle: Vec<f64>,
}

fn collocation(
    prob: &Problem<'_>,
    y: &[f64],
    p: &[f64],
    x: &[f64],
    h: &[f64],
) -> Result<Collocation, String> {
    let n = prob.n;
    let m = x.len();
    let mut f = Vec::with_capacity(n * m);
    for j in 0..m {
        f.extend(prob.f(x[j], &y[j * n..(j + 1) * n], p)?);
    }
    let mut y_middle = vec![0.0; n * (m - 1)];
    let mut f_middle = Vec::with_capacity(n * (m - 1));
    let mut col_res = vec![0.0; n * (m - 1)];
    for q in 0..m - 1 {
        for i in 0..n {
            y_middle[q * n + i] = 0.5 * (y[(q + 1) * n + i] + y[q * n + i])
                - 0.125 * h[q] * (f[(q + 1) * n + i] - f[q * n + i]);
        }
        f_middle.extend(prob.f(x[q] + 0.5 * h[q], &y_middle[q * n..(q + 1) * n], p)?);
    }
    for q in 0..m - 1 {
        for i in 0..n {
            col_res[q * n + i] = y[(q + 1) * n + i]
                - y[q * n + i]
                - h[q] / 6.0 * (f[q * n + i] + f[(q + 1) * n + i] + 4.0 * f_middle[q * n + i]);
        }
    }
    Ok(Collocation {
        col_res,
        y_middle,
        f,
        f_middle,
    })
}

/// `construct_global_jac` as `(row, col, value)` triplets of the (n·m + k)² system.
#[allow(clippy::too_many_arguments)]
fn global_jacobian(
    prob: &Problem<'_>,
    x: &[f64],
    h: &[f64],
    y: &[f64],
    p: &[f64],
    col: &Collocation,
    bc0: &[f64],
) -> Result<Vec<(usize, usize, f64)>, String> {
    let (n, k) = (prob.n, prob.k);
    let m = x.len();
    let mut node_jac = Vec::with_capacity(m);
    for j in 0..m {
        node_jac.push(prob.f_jac(x[j], &y[j * n..(j + 1) * n], p, &col.f[j * n..(j + 1) * n])?);
    }
    let mut triplets =
        Vec::with_capacity(2 * (m - 1) * n * n + (n + k) * (2 * n + k) + (m - 1) * n * k);
    for q in 0..m - 1 {
        let xm = x[q] + 0.5 * h[q];
        let (am, dpm) = prob.f_jac(
            xm,
            &col.y_middle[q * n..(q + 1) * n],
            p,
            &col.f_middle[q * n..(q + 1) * n],
        )?;
        let (a0, dp0) = &node_jac[q];
        let (a1, dp1) = &node_jac[q + 1];
        let hq = h[q];
        for i in 0..n {
            for j in 0..n {
                let t0: f64 = (0..n).map(|l| am[i * n + l] * a0[l * n + j]).sum();
                let t1: f64 = (0..n).map(|l| am[i * n + l] * a1[l * n + j]).sum();
                let delta = if i == j { 1.0 } else { 0.0 };
                let mut v0 = -delta;
                v0 -= hq / 6.0 * (a0[i * n + j] + 2.0 * am[i * n + j]);
                v0 -= hq * hq / 12.0 * t0;
                let mut v1 = delta;
                v1 -= hq / 6.0 * (a1[i * n + j] + 2.0 * am[i * n + j]);
                v1 += hq * hq / 12.0 * t1;
                triplets.push((q * n + i, q * n + j, v0));
                triplets.push((q * n + i, (q + 1) * n + j, v1));
            }
        }
        for i in 0..n {
            for l in 0..k {
                let t: f64 = (0..n)
                    .map(|r| am[i * n + r] * (dp0[r * k + l] - dp1[r * k + l]))
                    .sum();
                let dpm_il = dpm[i * k + l] + 0.125 * hq * t;
                let v = -hq / 6.0 * (dp0[i * k + l] + dp1[i * k + l] + 4.0 * dpm_il);
                triplets.push((q * n + i, m * n + l, v));
            }
        }
    }
    let (dya, dyb, dp) = prob.bc_jacobian(&y[..n], &y[(m - 1) * n..m * n], p, bc0)?;
    let base = (m - 1) * n;
    for b in 0..n + k {
        for j in 0..n {
            triplets.push((base + b, j, dya[b * n + j]));
            triplets.push((base + b, (m - 1) * n + j, dyb[b * n + j]));
        }
        for l in 0..k {
            triplets.push((base + b, m * n + l, dp[b * k + l]));
        }
    }
    Ok(triplets)
}

fn stack_residual(col_res: &[f64], bc_res: &[f64]) -> Vec<f64> {
    col_res.iter().chain(bc_res).copied().collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `solve_newton`: returns `(y, p, singular)`.
#[allow(clippy::too_many_arguments)]
fn solve_newton(
    prob: &Problem<'_>,
    x: &[f64],
    h: &[f64],
    mut y: Vec<f64>,
    mut p: Vec<f64>,
    bvp_tol: f64,
    bc_tol: f64,
    factorize: Factorize<'_>,
) -> Result<(Vec<f64>, Vec<f64>, bool), String> {
    let (n, k) = (prob.n, prob.k);
    let m = x.len();
    let size = n * m + k;
    let tol_r: Vec<f64> = h.iter().map(|hq| 2.0 / 3.0 * hq * 5e-2 * bvp_tol).collect();
    const MAX_NJEV: usize = 4;
    const MAX_ITER: usize = 8;
    const SIGMA: f64 = 0.2;
    const TAU: f64 = 0.5;
    const N_TRIAL: usize = 4;

    let mut col = collocation(prob, &y, &p, x, h)?;
    let mut bc_res = prob.bc(&y[..n], &y[(m - 1) * n..], &p)?;
    let mut res = stack_residual(&col.col_res, &bc_res);

    let mut njev = 0;
    let mut singular = false;
    let mut recompute_jac = true;
    let mut lu: Option<Box<dyn LinearSolver>> = None;
    let mut step = vec![0.0; size];
    let mut cost = 0.0;
    for _ in 0..MAX_ITER {
        if recompute_jac {
            let triplets = global_jacobian(prob, x, h, &y, &p, &col, &bc_res)?;
            njev += 1;
            lu = factorize(size, &triplets);
            let Some(solver) = lu.as_ref() else {
                singular = true;
                break;
            };
            let Some(s) = solver.solve(&res) else {
                singular = true;
                break;
            };
            step = s;
            cost = dot(&step, &step);
        }
        let solver = lu.as_ref().expect("factorized above");

        let mut alpha = 1.0;
        let mut y_new = vec![0.0; n * m];
        let mut p_new = vec![0.0; k];
        let mut step_new = Vec::new();
        let mut cost_new = 0.0;
        for trial in 0..=N_TRIAL {
            for (yn, (yi, si)) in y_new.iter_mut().zip(y.iter().zip(&step[..n * m])) {
                *yn = yi - alpha * si;
            }
            if let Some(sing) = &prob.singular {
                let y0 = matvec(&sing.b, &y_new[..n], n);
                y_new[..n].copy_from_slice(&y0);
            }
            for (pn, (pi, si)) in p_new.iter_mut().zip(p.iter().zip(&step[n * m..])) {
                *pn = pi - alpha * si;
            }
            col = collocation(prob, &y_new, &p_new, x, h)?;
            bc_res = prob.bc(&y_new[..n], &y_new[(m - 1) * n..], &p_new)?;
            res = stack_residual(&col.col_res, &bc_res);
            let Some(s) = solver.solve(&res) else {
                return Ok((y_new, p_new, true));
            };
            step_new = s;
            cost_new = dot(&step_new, &step_new);
            if cost_new < (1.0 - 2.0 * alpha * SIGMA) * cost {
                break;
            }
            if trial < N_TRIAL {
                alpha *= TAU;
            }
        }
        y = y_new;
        p = p_new;
        if njev == MAX_NJEV {
            break;
        }
        let col_ok = col
            .col_res
            .iter()
            .zip(&col.f_middle)
            .enumerate()
            .all(|(idx, (r, fm))| r.abs() < tol_r[idx / n] * (1.0 + fm.abs()));
        if col_ok && bc_res.iter().all(|r| r.abs() < bc_tol) {
            break;
        }
        if alpha == 1.0 {
            step = step_new;
            cost = cost_new;
            recompute_jac = false;
        } else {
            recompute_jac = true;
        }
    }
    Ok((y, p, singular))
}

/// The C¹ cubic on the mesh (`create_spline`): coefficients `c[q][0..4][i]` of
/// `c0·d³ + c1·d² + c2·d + c3`, `d = x − x_q`.
#[derive(Debug, Clone)]
pub(crate) struct Spline {
    pub x: Vec<f64>,
    pub n: usize,
    /// `coef[(q·4 + power)·n + i]`, power 0 = cubic term.
    coef: Vec<f64>,
}

impl Spline {
    pub(crate) fn new(x: &[f64], y: &[f64], yp: &[f64], n: usize) -> Self {
        let m = x.len();
        let mut coef = vec![0.0; (m - 1) * 4 * n];
        for q in 0..m - 1 {
            let h = x[q + 1] - x[q];
            for i in 0..n {
                let slope = (y[(q + 1) * n + i] - y[q * n + i]) / h;
                let t = (yp[q * n + i] + yp[(q + 1) * n + i] - 2.0 * slope) / h;
                coef[(q * 4) * n + i] = t / h;
                coef[(q * 4 + 1) * n + i] = (slope - yp[q * n + i]) / h - t;
                coef[(q * 4 + 2) * n + i] = yp[q * n + i];
                coef[(q * 4 + 3) * n + i] = y[q * n + i];
            }
        }
        Self {
            x: x.to_vec(),
            n,
            coef,
        }
    }

    /// PPoly's interval: the last breakpoint ≤ xq, clipped to the first/last interval
    /// (extrapolation).
    fn interval(&self, xq: f64) -> usize {
        let last = self.x.len() - 2;
        let pos = self.x.partition_point(|&b| b <= xq);
        pos.saturating_sub(1).min(last)
    }

    /// Value (`order` 0) or first derivative (`order` 1) at `xq`, summed as SciPy's
    /// `evaluate_poly1` does (ascending powers).
    pub(crate) fn eval_in(&self, q: usize, xq: f64, order: usize, out: &mut [f64]) {
        let s = xq - self.x[q];
        let n = self.n;
        for (i, o) in out.iter_mut().enumerate().take(n) {
            let c = |power: usize| self.coef[(q * 4 + power) * n + i];
            *o = if order == 0 {
                let mut res = c(3);
                let mut z = s;
                res += c(2) * z;
                z *= s;
                res += c(1) * z;
                z *= s;
                res + c(0) * z
            } else {
                let mut res = c(2);
                let mut z = s;
                res += c(1) * z * 2.0;
                z *= s;
                res + c(0) * z * 3.0
            };
        }
    }

    pub(crate) fn eval(&self, xq: f64, order: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        self.eval_in(self.interval(xq), xq, order, &mut out);
        out
    }
}

/// `estimate_rms_residuals`.
fn rms_residuals(
    prob: &Problem<'_>,
    sol: &Spline,
    x: &[f64],
    h: &[f64],
    p: &[f64],
    col: &Collocation,
) -> Result<Vec<f64>, String> {
    let n = prob.n;
    let m = x.len();
    let mut out = Vec::with_capacity(m - 1);
    let mut buf = vec![0.0; n];
    let mut dbuf = vec![0.0; n];
    for q in 0..m - 1 {
        let x_middle = x[q] + 0.5 * h[q];
        let s = 0.5 * h[q] * (3.0_f64 / 7.0).sqrt();
        let mut sums = [0.0_f64; 2];
        for (slot, xs) in [x_middle + s, x_middle - s].into_iter().enumerate() {
            let iq = sol.interval(xs);
            sol.eval_in(iq, xs, 0, &mut buf);
            sol.eval_in(iq, xs, 1, &mut dbuf);
            let f = prob.f(xs, &buf, p)?;
            sums[slot] = dbuf
                .iter()
                .zip(&f)
                .map(|(yp, fv)| {
                    let r = (yp - fv) / (1.0 + fv.abs());
                    r * r
                })
                .sum();
        }
        let r_middle: f64 = (0..n)
            .map(|i| {
                let r = 1.5 * col.col_res[q * n + i] / h[q] / (1.0 + col.f_middle[q * n + i].abs());
                r * r
            })
            .sum();
        out.push((0.5 * (32.0 / 45.0 * r_middle + 49.0 / 90.0 * (sums[0] + sums[1]))).sqrt());
    }
    Ok(out)
}

/// `modify_mesh`: one node in each interval of `insert_1`, two in each of `insert_2`.
fn modify_mesh(x: &[f64], insert_1: &[usize], insert_2: &[usize]) -> Vec<f64> {
    let mut out = x.to_vec();
    for &i in insert_1 {
        out.push(0.5 * (x[i] + x[i + 1]));
    }
    for &i in insert_2 {
        out.push((2.0 * x[i] + x[i + 1]) / 3.0);
    }
    for &i in insert_2 {
        out.push((x[i] + 2.0 * x[i + 1]) / 3.0);
    }
    out.sort_by(f64::total_cmp);
    out
}

/// The `solve_bvp` main loop. `y` is node-major (n per node), `tol` already floored at
/// 100·eps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    prob: &Problem<'_>,
    mut x: Vec<f64>,
    mut y: Vec<f64>,
    mut p: Vec<f64>,
    tol: f64,
    max_nodes: usize,
    bc_tol: f64,
    factorize: Factorize<'_>,
) -> Result<(Outcome, Spline), String> {
    const MAX_ITERATION: usize = 10;
    let n = prob.n;
    if let Some(sing) = &prob.singular {
        let y0 = matvec(&sing.b, &y[..n], n);
        y[..n].copy_from_slice(&y0);
    }
    let mut h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let mut iteration = 0;
    loop {
        let m = x.len();
        let (y_new, p_new, singular) = solve_newton(prob, &x, &h, y, p, tol, bc_tol, factorize)?;
        y = y_new;
        p = p_new;
        iteration += 1;

        let col = collocation(prob, &y, &p, &x, &h)?;
        let bc_res = prob.bc(&y[..n], &y[(m - 1) * n..], &p)?;
        // numpy's `np.max(abs(bc_res))`: one NaN residual makes it NaN, so `max_bc_res <= bc_tol`
        // below is false, as in SciPy. `f64::max` drops a NaN, which read a NaN boundary residual
        // as 0 and reported status 0 on an all-NaN solution.
        let max_bc_res = bc_res.iter().fold(0.0_f64, |acc, r| {
            if acc.is_nan() || r.is_nan() {
                f64::NAN
            } else {
                acc.max(r.abs())
            }
        });
        let sol = Spline::new(&x, &y, &col.f, n);
        let rms = rms_residuals(prob, &sol, &x, &h, &p, &col)?;

        let finish =
            |status: usize, x: Vec<f64>, y: Vec<f64>, p: Vec<f64>, rms: Vec<f64>| Outcome {
                x,
                y,
                yp: col.f.clone(),
                p,
                rms_residuals: rms,
                niter: iteration,
                status,
            };
        if singular {
            return Ok((finish(2, x, y, p, rms), sol));
        }
        let insert_1: Vec<usize> = (0..m - 1)
            .filter(|&q| rms[q] > tol && rms[q] < 100.0 * tol)
            .collect();
        let insert_2: Vec<usize> = (0..m - 1).filter(|&q| rms[q] >= 100.0 * tol).collect();
        let nodes_added = insert_1.len() + 2 * insert_2.len();
        if m + nodes_added > max_nodes {
            return Ok((finish(1, x, y, p, rms), sol));
        }
        if nodes_added > 0 {
            x = modify_mesh(&x, &insert_1, &insert_2);
            h = x.windows(2).map(|w| w[1] - w[0]).collect();
            y = x.iter().flat_map(|&xq| sol.eval(xq, 0)).collect();
        } else if max_bc_res <= bc_tol {
            return Ok((finish(0, x, y, p, rms), sol));
        } else if iteration >= MAX_ITERATION {
            return Ok((finish(3, x, y, p, rms), sol));
        }
    }
}
