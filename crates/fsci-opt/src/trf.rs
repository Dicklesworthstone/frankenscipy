#![forbid(unsafe_code)]

//! The Trust Region Reflective least-squares algorithm behind `least_squares(method = Trf)`:
//! SciPy's `scipy/optimize/_lsq/trf.py` with the helpers of `_lsq/common.py` and the robust
//! losses of `_lsq/least_squares.py` (Branch, Coleman & Li, SIAM J. Sci. Comput. 21(1), 1999),
//! transcribed function by function with the dense (`tr_solver = 'exact'`) subproblem solver:
//! Moré's algorithm on one SVD per outer iteration.
//!
//! Bounds are handled by the Coleman–Li scaling `v`, a step that is reflected off the first
//! bound it hits, and a choice among the scaled trust-region step, its reflection and the
//! scaled gradient, as in SciPy. Without bounds the plain trust-region step is taken.

const EPS: f64 = f64::EPSILON;

/// SciPy's `loss` names; `f_scale` is applied by [`Loss::rho`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossKind {
    Linear,
    SoftL1,
    Huber,
    Cauchy,
    Arctan,
}

pub(crate) struct Loss {
    pub kind: LossKind,
    pub f_scale: f64,
}

impl Loss {
    /// `construct_loss_function(...)(f)`: `rho[k][i]` for k = 0, 1, 2 (value, first and second
    /// derivative in z = (f/f_scale)²), scaled back by `f_scale`.
    /// `(ρ(z), ρ'(z), ρ''(z))` of the unscaled loss (SciPy's `huber`, `soft_l1`, ...).
    fn raw(&self, z: f64) -> (f64, f64, f64) {
        match self.kind {
            LossKind::Linear => (z, 1.0, 0.0),
            LossKind::Huber => {
                if z <= 1.0 {
                    (z, 1.0, 0.0)
                } else {
                    (2.0 * z.sqrt() - 1.0, z.powf(-0.5), -0.5 * z.powf(-1.5))
                }
            }
            LossKind::SoftL1 => {
                let t = 1.0 + z;
                (2.0 * (t.sqrt() - 1.0), t.powf(-0.5), -0.5 * t.powf(-1.5))
            }
            LossKind::Cauchy => {
                let t = 1.0 + z;
                (z.ln_1p(), 1.0 / t, -1.0 / (t * t))
            }
            LossKind::Arctan => {
                let t = 1.0 + z * z;
                (z.atan(), 1.0 / t, -2.0 * z / (t * t))
            }
        }
    }

    fn rho(&self, f: &[f64]) -> [Vec<f64>; 3] {
        let m = f.len();
        let fs2 = self.f_scale * self.f_scale;
        let mut rho = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
        for (i, &fi) in f.iter().enumerate() {
            let (r0, r1, r2) = self.raw((fi / self.f_scale).powi(2));
            rho[0][i] = r0 * fs2;
            rho[1][i] = r1;
            rho[2][i] = r2 / fs2;
        }
        rho
    }

    /// `loss_function(f, cost_only=True)`: ½·f_scale²·Σρ(z), in SciPy's order.
    fn cost(&self, f: &[f64]) -> f64 {
        let raw: f64 = f
            .iter()
            .map(|fi| self.raw((fi / self.f_scale).powi(2)).0)
            .sum();
        0.5 * (self.f_scale * self.f_scale) * raw
    }
}

/// `x_scale`: fixed positive scales, or SciPy's `'jac'` (inverse column norms of J, never
/// decreasing across iterations).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum XScale {
    Fixed(Vec<f64>),
    Jac,
}

/// What `trf` returns (SciPy's `OptimizeResult` fields).
pub(crate) struct TrfOutcome {
    pub x: Vec<f64>,
    pub cost: f64,
    pub fun: Vec<f64>,
    pub jac: Vec<Vec<f64>>,
    pub grad: Vec<f64>,
    pub optimality: f64,
    pub active_mask: Vec<i8>,
    pub nfev: usize,
    pub njev: usize,
    /// 0 max_nfev, 1 gtol, 2 ftol, 3 xtol, 4 ftol and xtol.
    pub status: i32,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// numpy's `norm(a, ord=inf)`: one NaN entry makes it NaN. `f64::max` would drop the NaN, and
/// `g_norm < gtol` would then report convergence on a NaN gradient (frankenscipy-qdb8s).
fn norm_inf(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v.abs())
        }
    })
}

/// The matrix the dense subproblem factors held a NaN or an infinity. SciPy's `trf` passes it to
/// `scipy.linalg.svd`, whose `check_finite` raises `ValueError: array must not contain infs or
/// NaNs` at exactly this point, after the gtol and max_nfev exits have been tested.
pub(crate) struct NonFiniteJacobian;

fn all_finite(m: &[Vec<f64>]) -> bool {
    m.iter().flatten().all(|v| v.is_finite())
}

fn mat_vec(j: &[Vec<f64>], s: &[f64]) -> Vec<f64> {
    j.iter().map(|row| dot(row, s)).collect()
}

/// `Jᵀ·f`.
fn compute_grad(j: &[Vec<f64>], f: &[f64]) -> Vec<f64> {
    let n = j.first().map_or(0, Vec::len);
    let mut g = vec![0.0; n];
    for (row, &fi) in j.iter().zip(f) {
        for (gk, &jk) in g.iter_mut().zip(row) {
            *gk += jk * fi;
        }
    }
    g
}

/// `compute_jac_scale`: `(scale, scale_inv)` with `scale_inv` the column norms of J.
fn compute_jac_scale(j: &[Vec<f64>], scale_inv_old: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let n = j.first().map_or(0, Vec::len);
    let mut scale_inv = vec![0.0; n];
    for row in j {
        for (s, &v) in scale_inv.iter_mut().zip(row) {
            *s += v * v;
        }
    }
    for s in &mut scale_inv {
        *s = s.sqrt();
    }
    match scale_inv_old {
        None => {
            for s in &mut scale_inv {
                if *s == 0.0 {
                    *s = 1.0;
                }
            }
        }
        Some(old) => {
            for (s, &o) in scale_inv.iter_mut().zip(old) {
                *s = s.max(o);
            }
        }
    }
    (scale_inv.iter().map(|s| 1.0 / s).collect(), scale_inv)
}

/// `scale_for_robust_loss_function`: rows of J and entries of f scaled in place.
fn scale_for_robust_loss(j: &mut [Vec<f64>], f: &mut [f64], rho: &[Vec<f64>; 3]) {
    for (i, (row, fi)) in j.iter_mut().zip(f.iter_mut()).enumerate() {
        let mut js = rho[1][i] + 2.0 * rho[2][i] * *fi * *fi;
        if js < EPS {
            js = EPS;
        }
        let js = js.sqrt();
        *fi *= rho[1][i] / js;
        for v in row.iter_mut() {
            *v *= js;
        }
    }
}

/// Thin SVD `J = U·diag(s)·Vᵀ` (singular values descending) by one-sided Jacobi. Returns
/// `(U` as k columns of length m, `s`, `V` as k columns of length n`)`, k = min(m, n).
#[allow(clippy::type_complexity)]
fn svd_thin(j: &[Vec<f64>], n: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = j.len();
    let transposed = m < n;
    // Columns of the matrix whose columns are orthogonalized: J (m ≥ n) or Jᵀ (m < n).
    let (rows, cols) = if transposed { (n, m) } else { (m, n) };
    let mut a: Vec<Vec<f64>> = (0..cols)
        .map(|c| {
            (0..rows)
                .map(|r| if transposed { j[c][r] } else { j[r][c] })
                .collect()
        })
        .collect();
    let mut v: Vec<Vec<f64>> = (0..cols)
        .map(|c| (0..cols).map(|r| if r == c { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..cols {
            for q in p + 1..cols {
                let alpha = dot(&a[p], &a[p]);
                let beta = dot(&a[q], &a[q]);
                let gamma = dot(&a[p], &a[q]);
                if gamma.abs() <= EPS * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for k in 0..rows {
                    let (ap, aq) = (a[p][k], a[q][k]);
                    a[p][k] = c * ap - s * aq;
                    a[q][k] = s * ap + c * aq;
                }
                for k in 0..cols {
                    let (vp, vq) = (v[p][k], v[q][k]);
                    v[p][k] = c * vp - s * vq;
                    v[q][k] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<usize> = (0..cols).collect();
    let sv: Vec<f64> = a.iter().map(|col| norm(col)).collect();
    order.sort_by(|&x, &y| sv[y].total_cmp(&sv[x]));
    let k = m.min(n);
    let mut left = Vec::with_capacity(k);
    let mut s = Vec::with_capacity(k);
    let mut right = Vec::with_capacity(k);
    for &idx in order.iter().take(k) {
        let sigma = sv[idx];
        let unit: Vec<f64> = if sigma > 0.0 {
            a[idx].iter().map(|x| x / sigma).collect()
        } else {
            vec![0.0; rows]
        };
        s.push(sigma);
        if transposed {
            // Jᵀ = Ũ·S·Ṽᵀ ⇒ J = Ṽ·S·Ũᵀ.
            left.push(v[idx].clone());
            right.push(unit);
        } else {
            left.push(unit);
            right.push(v[idx].clone());
        }
    }
    (left, s, right)
}

/// `intersect_trust_region`: the two t with ‖x + t·s‖ = Δ, ascending.
fn intersect_trust_region(x: &[f64], s: &[f64], delta: f64) -> (f64, f64) {
    let a = dot(s, s);
    let b = dot(x, s);
    let c = dot(x, x) - delta * delta;
    let d = (b * b - a * c).sqrt();
    let q = -(b + d.copysign(b));
    let t1 = q / a;
    let t2 = c / q;
    if t1 < t2 { (t1, t2) } else { (t2, t1) }
}

/// `solve_lsq_trust_region` (Moré's method on the SVD): returns `(p, alpha)`.
#[allow(clippy::too_many_arguments)]
fn solve_lsq_trust_region(
    n: usize,
    m: usize,
    uf: &[f64],
    s: &[f64],
    v: &[Vec<f64>],
    delta: f64,
    initial_alpha: f64,
) -> (Vec<f64>, f64) {
    const RTOL: f64 = 0.01;
    const MAX_ITER: usize = 10;
    let suf: Vec<f64> = s.iter().zip(uf).map(|(a, b)| a * b).collect();
    let v_dot = |w: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; n];
        for (col, &wk) in v.iter().zip(w) {
            for (o, &c) in out.iter_mut().zip(col) {
                *o += c * wk;
            }
        }
        out
    };
    let phi_and_derivative = |alpha: f64| -> (f64, f64) {
        let denom: Vec<f64> = s.iter().map(|si| si * si + alpha).collect();
        let p_norm = norm(
            &suf.iter()
                .zip(&denom)
                .map(|(a, b)| a / b)
                .collect::<Vec<_>>(),
        );
        let phi = p_norm - delta;
        let phi_prime = -suf
            .iter()
            .zip(&denom)
            .map(|(a, d)| a * a / (d * d * d))
            .sum::<f64>()
            / p_norm;
        (phi, phi_prime)
    };

    let full_rank = if m >= n {
        let threshold = EPS * m as f64 * s[0];
        s[s.len() - 1] > threshold
    } else {
        false
    };
    if full_rank {
        let p: Vec<f64> = v_dot(&uf.iter().zip(s).map(|(u, si)| u / si).collect::<Vec<_>>())
            .into_iter()
            .map(|x| -x)
            .collect();
        if norm(&p) <= delta {
            return (p, 0.0);
        }
    }
    let mut alpha_upper = norm(&suf) / delta;
    let mut alpha_lower = if full_rank {
        let (phi, phi_prime) = phi_and_derivative(0.0);
        -phi / phi_prime
    } else {
        0.0
    };
    let mut alpha = if !full_rank && initial_alpha == 0.0 {
        (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt())
    } else {
        initial_alpha
    };
    for _ in 0..MAX_ITER {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = (0.001 * alpha_upper).max((alpha_lower * alpha_upper).sqrt());
        }
        let (phi, phi_prime) = phi_and_derivative(alpha);
        if phi < 0.0 {
            alpha_upper = alpha;
        }
        let ratio = phi / phi_prime;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + delta) * ratio / delta;
        if phi.abs() < RTOL * delta {
            break;
        }
    }
    let w: Vec<f64> = suf
        .iter()
        .zip(s)
        .map(|(a, si)| a / (si * si + alpha))
        .collect();
    let mut p: Vec<f64> = v_dot(&w).into_iter().map(|x| -x).collect();
    let scale = delta / norm(&p);
    for x in &mut p {
        *x *= scale;
    }
    (p, alpha)
}

/// `update_tr_radius`: `(Δ_new, ratio)`.
fn update_tr_radius(
    delta: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    step_norm: f64,
    bound_hit: bool,
) -> (f64, f64) {
    let ratio = if predicted_reduction > 0.0 {
        actual_reduction / predicted_reduction
    } else if predicted_reduction == actual_reduction && actual_reduction == 0.0 {
        1.0
    } else {
        0.0
    };
    let delta = if ratio < 0.25 {
        0.25 * step_norm
    } else if ratio > 0.75 && bound_hit {
        delta * 2.0
    } else {
        delta
    };
    (delta, ratio)
}

/// `build_quadratic_1d`: coefficients of q(t) = a·t² + b·t (+ c with `s0`).
fn build_quadratic_1d(
    j: &[Vec<f64>],
    g: &[f64],
    s: &[f64],
    diag: Option<&[f64]>,
    s0: Option<&[f64]>,
) -> (f64, f64, f64) {
    let v = mat_vec(j, s);
    let mut a = dot(&v, &v);
    if let Some(d) = diag {
        a += s
            .iter()
            .zip(d)
            .zip(s)
            .map(|((x, y), z)| x * y * z)
            .sum::<f64>();
    }
    a *= 0.5;
    let mut b = dot(g, s);
    let mut c = 0.0;
    if let Some(s0) = s0 {
        let u = mat_vec(j, s0);
        b += dot(&u, &v);
        c = 0.5 * dot(&u, &u) + dot(g, s0);
        if let Some(d) = diag {
            b += s0
                .iter()
                .zip(d)
                .zip(s)
                .map(|((x, y), z)| x * y * z)
                .sum::<f64>();
            c += 0.5
                * s0.iter()
                    .zip(d)
                    .zip(s0)
                    .map(|((x, y), z)| x * y * z)
                    .sum::<f64>();
        }
    }
    (a, b, c)
}

/// `minimize_quadratic_1d` on `[lb, ub]`: `(t, q(t))`.
fn minimize_quadratic_1d(a: f64, b: f64, lb: f64, ub: f64, c: f64) -> (f64, f64) {
    let mut t = vec![lb, ub];
    if a != 0.0 {
        let extremum = -0.5 * b / a;
        if lb < extremum && extremum < ub {
            t.push(extremum);
        }
    }
    let mut best = (t[0], t[0] * (a * t[0] + b) + c);
    for &ti in &t[1..] {
        let y = ti * (a * ti + b) + c;
        if y < best.1 {
            best = (ti, y);
        }
    }
    best
}

/// `evaluate_quadratic`: ½·‖J·s‖² + ½·sᵀ·diag·s + gᵀ·s.
fn evaluate_quadratic(j: &[Vec<f64>], g: &[f64], s: &[f64], diag: Option<&[f64]>) -> f64 {
    let js = mat_vec(j, s);
    let mut q = dot(&js, &js);
    if let Some(d) = diag {
        q += s
            .iter()
            .zip(d)
            .zip(s)
            .map(|((x, y), z)| x * y * z)
            .sum::<f64>();
    }
    0.5 * q + dot(s, g)
}

fn in_bounds(x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    x.iter()
        .zip(lb.iter().zip(ub))
        .all(|(&xi, (&l, &u))| xi >= l && xi <= u)
}

/// `step_size_to_bound`: `(min_step, hits)` with hits ∈ {−1, 0, 1} per coordinate.
fn step_size_to_bound(x: &[f64], s: &[f64], lb: &[f64], ub: &[f64]) -> (f64, Vec<i8>) {
    let steps: Vec<f64> = x
        .iter()
        .zip(s)
        .zip(lb.iter().zip(ub))
        .map(|((&xi, &si), (&l, &u))| {
            if si == 0.0 {
                f64::INFINITY
            } else {
                ((l - xi) / si).max((u - xi) / si)
            }
        })
        .collect();
    let min_step = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let hits = steps
        .iter()
        .zip(s)
        .map(|(&st, &si)| {
            if st == min_step {
                if si > 0.0 {
                    1
                } else if si < 0.0 {
                    -1
                } else {
                    0
                }
            } else {
                0
            }
        })
        .collect();
    (min_step, hits)
}

/// `find_active_constraints`: −1 at an active lower bound, 1 at an active upper bound.
pub(crate) fn find_active_constraints(x: &[f64], lb: &[f64], ub: &[f64], rtol: f64) -> Vec<i8> {
    x.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&xi, (&l, &u))| {
            if rtol == 0.0 {
                return if xi >= u {
                    1
                } else if xi <= l {
                    -1
                } else {
                    0
                };
            }
            let lower_dist = xi - l;
            let upper_dist = u - xi;
            let lower_threshold = rtol * l.abs().max(1.0);
            let upper_threshold = rtol * u.abs().max(1.0);
            let mut active = 0;
            if l.is_finite() && lower_dist <= upper_dist.min(lower_threshold) {
                active = -1;
            }
            if u.is_finite() && upper_dist <= lower_dist.min(upper_threshold) {
                active = 1;
            }
            active
        })
        .collect()
}

/// `make_strictly_feasible`.
pub(crate) fn make_strictly_feasible(x: &[f64], lb: &[f64], ub: &[f64], rstep: f64) -> Vec<f64> {
    let active = find_active_constraints(x, lb, ub, rstep);
    x.iter()
        .zip(&active)
        .zip(lb.iter().zip(ub))
        .map(|((&xi, &a), (&l, &u))| {
            let mut v = xi;
            if a == -1 {
                v = if rstep == 0.0 {
                    next_toward(l, u)
                } else {
                    l + rstep * l.abs().max(1.0)
                };
            } else if a == 1 {
                v = if rstep == 0.0 {
                    next_toward(u, l)
                } else {
                    u - rstep * u.abs().max(1.0)
                };
            }
            if v < l || v > u { 0.5 * (l + u) } else { v }
        })
        .collect()
}

/// `np.nextafter(from, to)` for finite `from`.
fn next_toward(from: f64, to: f64) -> f64 {
    if from == to || from.is_nan() || to.is_nan() {
        return from;
    }
    if from == 0.0 {
        return f64::from_bits(1).copysign(to);
    }
    let bits = from.to_bits();
    let up = (to > from) == (from > 0.0);
    f64::from_bits(if up { bits + 1 } else { bits - 1 })
}

/// `CL_scaling_vector`: Coleman–Li `(v, dv)`.
fn cl_scaling_vector(x: &[f64], g: &[f64], lb: &[f64], ub: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut v = vec![1.0; n];
    let mut dv = vec![0.0; n];
    for i in 0..n {
        if g[i] < 0.0 && ub[i].is_finite() {
            v[i] = ub[i] - x[i];
            dv[i] = -1.0;
        }
        if g[i] > 0.0 && lb[i].is_finite() {
            v[i] = x[i] - lb[i];
            dv[i] = 1.0;
        }
    }
    (v, dv)
}

/// `check_termination`.
fn check_termination(
    df: f64,
    f: f64,
    dx_norm: f64,
    x_norm: f64,
    ratio: f64,
    ftol: f64,
    xtol: f64,
) -> Option<i32> {
    let ftol_satisfied = df < ftol * f && ratio > 0.25;
    let xtol_satisfied = dx_norm < xtol * (xtol + x_norm);
    match (ftol_satisfied, xtol_satisfied) {
        (true, true) => Some(4),
        (true, false) => Some(2),
        (false, true) => Some(3),
        (false, false) => None,
    }
}

/// `select_step`: the best of the trust-region step (clipped at the bound and reflected) and
/// the scaled-gradient step. Returns `(step, step_h, predicted_reduction)`.
#[allow(clippy::too_many_arguments)]
fn select_step(
    x: &[f64],
    j_h: &[Vec<f64>],
    diag_h: &[f64],
    g_h: &[f64],
    mut p: Vec<f64>,
    mut p_h: Vec<f64>,
    d: &[f64],
    delta: f64,
    lb: &[f64],
    ub: &[f64],
    theta: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let x_plus_p: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    if in_bounds(&x_plus_p, lb, ub) {
        let p_value = evaluate_quadratic(j_h, g_h, &p_h, Some(diag_h));
        return (p, p_h, -p_value);
    }
    let (p_stride, hits) = step_size_to_bound(x, &p, lb, ub);
    let mut r_h = p_h.clone();
    for (r, &h) in r_h.iter_mut().zip(&hits) {
        if h != 0 {
            *r = -*r;
        }
    }
    let mut r: Vec<f64> = d.iter().zip(&r_h).map(|(a, b)| a * b).collect();
    for v in &mut p {
        *v *= p_stride;
    }
    for v in &mut p_h {
        *v *= p_stride;
    }
    let x_on_bound: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
    let (_, to_tr) = intersect_trust_region(&p_h, &r_h, delta);
    let (to_bound, _) = step_size_to_bound(&x_on_bound, &r, lb, ub);
    let r_stride = to_bound.min(to_tr);
    let (r_stride_l, r_stride_u) = if r_stride > 0.0 {
        let l = (1.0 - theta) * p_stride / r_stride;
        let u = if r_stride == to_bound {
            theta * to_bound
        } else {
            to_tr
        };
        (l, u)
    } else {
        (0.0, -1.0)
    };
    let r_value = if r_stride_l <= r_stride_u {
        let (a, b, c) = build_quadratic_1d(j_h, g_h, &r_h, Some(diag_h), Some(&p_h));
        let (stride, value) = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
        for (rh, &ph) in r_h.iter_mut().zip(&p_h) {
            *rh = *rh * stride + ph;
        }
        r = r_h.iter().zip(d).map(|(a, b)| a * b).collect();
        value
    } else {
        f64::INFINITY
    };
    for v in &mut p {
        *v *= theta;
    }
    for v in &mut p_h {
        *v *= theta;
    }
    let p_value = evaluate_quadratic(j_h, g_h, &p_h, Some(diag_h));
    let mut ag_h: Vec<f64> = g_h.iter().map(|v| -v).collect();
    let mut ag: Vec<f64> = d.iter().zip(&ag_h).map(|(a, b)| a * b).collect();
    let to_tr = delta / norm(&ag_h);
    let (to_bound, _) = step_size_to_bound(x, &ag, lb, ub);
    let ag_stride = if to_bound < to_tr {
        theta * to_bound
    } else {
        to_tr
    };
    let (a, b, _) = build_quadratic_1d(j_h, g_h, &ag_h, Some(diag_h), None);
    let (ag_stride, ag_value) = minimize_quadratic_1d(a, b, 0.0, ag_stride, 0.0);
    for v in &mut ag_h {
        *v *= ag_stride;
    }
    for v in &mut ag {
        *v *= ag_stride;
    }
    if p_value < r_value && p_value < ag_value {
        (p, p_h, -p_value)
    } else if r_value < p_value && r_value < ag_value {
        (r, r_h, -r_value)
    } else {
        (ag, ag_h, -ag_value)
    }
}

/// Residuals at `x`.
pub(crate) type ResidualFn<'a, E> = &'a mut dyn FnMut(&[f64]) -> Result<Vec<f64>, E>;
/// Jacobian at `x` (rows = residuals), given the residuals `f` just evaluated there.
pub(crate) type JacobianFn<'a, E> = &'a mut dyn FnMut(&[f64], &[f64]) -> Result<Vec<Vec<f64>>, E>;

/// `trf`: `fun(x)` evaluates the residuals, `jac(x, f)` the Jacobian (rows = residuals) at the
/// point where `f` was just evaluated. `x0` must already be strictly feasible (SciPy's caller
/// applies `make_strictly_feasible`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn trf<E>(
    fun: ResidualFn<'_, E>,
    jac: JacobianFn<'_, E>,
    x0: &[f64],
    f0: Vec<f64>,
    j0: Vec<Vec<f64>>,
    lb: &[f64],
    ub: &[f64],
    ftol: f64,
    xtol: f64,
    gtol: f64,
    max_nfev: Option<usize>,
    x_scale: &XScale,
    loss: Option<&Loss>,
) -> Result<TrfOutcome, E>
where
    E: From<NonFiniteJacobian>,
{
    let bounded =
        lb.iter().any(|v| *v != f64::NEG_INFINITY) || ub.iter().any(|v| *v != f64::INFINITY);
    let n = x0.len();
    let m = f0.len();
    let mut x = x0.to_vec();
    let mut f = f0;
    let mut f_true = f.clone();
    let mut nfev = 1;
    let mut j = j0;
    let mut njev = 1;
    let mut cost = match loss {
        Some(l) => {
            let rho = l.rho(&f);
            let c = 0.5 * rho[0].iter().sum::<f64>();
            scale_for_robust_loss(&mut j, &mut f, &rho);
            c
        }
        None => 0.5 * dot(&f, &f),
    };
    let mut g = compute_grad(&j, &f);
    let jac_scale = matches!(x_scale, XScale::Jac);
    let (mut scale, mut scale_inv) = match x_scale {
        XScale::Jac => compute_jac_scale(&j, None),
        XScale::Fixed(s) => (s.clone(), s.iter().map(|v| 1.0 / v).collect()),
    };
    let mut delta = if bounded {
        let (mut v, dv) = cl_scaling_vector(&x, &g, lb, ub);
        for i in 0..n {
            if dv[i] != 0.0 {
                v[i] *= scale_inv[i];
            }
        }
        norm(
            &x0.iter()
                .zip(&scale_inv)
                .zip(&v)
                .map(|((a, b), c)| a * b / c.sqrt())
                .collect::<Vec<_>>(),
        )
    } else {
        norm(
            &x0.iter()
                .zip(&scale_inv)
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        )
    };
    if delta == 0.0 {
        delta = 1.0;
    }
    let max_nfev = max_nfev.unwrap_or(n * 100);
    let mut alpha = 0.0;
    let mut termination_status: Option<i32> = None;
    let mut g_norm;

    loop {
        let (mut v, dv) = if bounded {
            cl_scaling_vector(&x, &g, lb, ub)
        } else {
            (vec![1.0; n], vec![0.0; n])
        };
        g_norm = norm_inf(&g.iter().zip(&v).map(|(a, b)| a * b).collect::<Vec<_>>());
        if g_norm < gtol {
            termination_status = Some(1);
        }
        if termination_status.is_some() || nfev == max_nfev {
            break;
        }

        // Scaled variables: d = v^½·scale (bounds) or d = scale.
        for i in 0..n {
            if dv[i] != 0.0 {
                v[i] *= scale_inv[i];
            }
        }
        let d: Vec<f64> = if bounded {
            v.iter().zip(&scale).map(|(a, b)| a.sqrt() * b).collect()
        } else {
            scale.clone()
        };
        let diag_h: Vec<f64> = if bounded {
            g.iter()
                .zip(&dv)
                .zip(&scale)
                .map(|((a, b), c)| a * b * c)
                .collect()
        } else {
            vec![0.0; n]
        };
        let g_h: Vec<f64> = d.iter().zip(&g).map(|(a, b)| a * b).collect();
        let j_h: Vec<Vec<f64>> = j
            .iter()
            .map(|row| row.iter().zip(&d).map(|(a, b)| a * b).collect())
            .collect();
        let (s, vv, uf) = if bounded {
            let mut aug = j_h.clone();
            for (i, dh) in diag_h.iter().enumerate() {
                let mut row = vec![0.0; n];
                row[i] = dh.sqrt();
                aug.push(row);
            }
            if !all_finite(&aug) {
                return Err(NonFiniteJacobian.into());
            }
            let (u, s, vv) = svd_thin(&aug, n);
            let mut f_aug = f.clone();
            f_aug.extend(std::iter::repeat_n(0.0, n));
            let uf: Vec<f64> = u.iter().map(|col| dot(col, &f_aug)).collect();
            (s, vv, uf)
        } else {
            if !all_finite(&j_h) {
                return Err(NonFiniteJacobian.into());
            }
            let (u, s, vv) = svd_thin(&j_h, n);
            let uf: Vec<f64> = u.iter().map(|col| dot(col, &f)).collect();
            (s, vv, uf)
        };
        let theta = if bounded {
            0.995_f64.max(1.0 - g_norm)
        } else {
            0.0
        };

        let mut actual_reduction = -1.0;
        let mut x_new = x.clone();
        let mut f_new = f.clone();
        let mut cost_new = cost;
        while actual_reduction <= 0.0 && nfev < max_nfev {
            let (p_h, new_alpha) = solve_lsq_trust_region(n, m, &uf, &s, &vv, delta, alpha);
            alpha = new_alpha;
            let (step, step_h, predicted_reduction) = if bounded {
                let p: Vec<f64> = d.iter().zip(&p_h).map(|(a, b)| a * b).collect();
                select_step(&x, &j_h, &diag_h, &g_h, p, p_h, &d, delta, lb, ub, theta)
            } else {
                let predicted = -evaluate_quadratic(&j_h, &g_h, &p_h, None);
                let step: Vec<f64> = d.iter().zip(&p_h).map(|(a, b)| a * b).collect();
                (step, p_h, predicted)
            };
            x_new = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            if bounded {
                x_new = make_strictly_feasible(&x_new, lb, ub, 0.0);
            }
            f_new = fun(&x_new)?;
            nfev += 1;
            let step_h_norm = norm(&step_h);
            if f_new.iter().any(|v| !v.is_finite()) {
                delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = match loss {
                Some(l) => l.cost(&f_new),
                None => 0.5 * dot(&f_new, &f_new),
            };
            actual_reduction = cost - cost_new;
            let (delta_new, ratio) = update_tr_radius(
                delta,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                step_h_norm > 0.95 * delta,
            );
            let step_norm = norm(&step);
            termination_status = check_termination(
                actual_reduction,
                cost,
                step_norm,
                norm(&x),
                ratio,
                ftol,
                xtol,
            );
            if termination_status.is_some() {
                break;
            }
            alpha *= delta / delta_new;
            delta = delta_new;
        }

        if actual_reduction > 0.0 {
            x = x_new;
            f = f_new;
            f_true = f.clone();
            cost = cost_new;
            j = jac(&x, &f)?;
            njev += 1;
            if let Some(l) = loss {
                let rho = l.rho(&f);
                scale_for_robust_loss(&mut j, &mut f, &rho);
            }
            g = compute_grad(&j, &f);
            if jac_scale {
                let (sc, sci) = compute_jac_scale(&j, Some(&scale_inv));
                scale = sc;
                scale_inv = sci;
            }
        }
    }

    let active_mask = if bounded {
        find_active_constraints(&x, lb, ub, xtol)
    } else {
        vec![0; n]
    };
    Ok(TrfOutcome {
        x,
        cost,
        fun: f_true,
        jac: j,
        grad: g,
        optimality: g_norm,
        active_mask,
        nfev,
        njev,
        status: termination_status.unwrap_or(0),
    })
}
