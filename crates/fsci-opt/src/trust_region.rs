//! Trust-region Newton minimization: SciPy's `_minimize_trust_region` driver with its three
//! subproblem solvers — CG-Steihaug (`trust-ncg`), dogleg, and the nearly-exact iterative
//! solver of Moré & Sorensen / Conn, Gould & Toint (`trust-exact`) — transcribed from
//! `scipy/optimize/_trustregion{,_ncg,_dogleg,_exact}.py` (SciPy 1.17.1).
//!
//! std-only: derivatives are reached through [`TrustObjective`]; matrices are dense row-major.

/// What the driver needs from the objective. Every value is requested at most once per iterate.
pub(crate) trait TrustObjective {
    type Error;
    fn fun(&mut self, x: &[f64]) -> Result<f64, Self::Error>;
    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, Self::Error>;
    /// Whether the caller supplied a Hessian (SciPy's `ScalarFunction` then evaluates it at
    /// `x0` up front, which the driver mirrors so evaluation counts agree).
    fn has_hess(&self) -> bool;
    /// Dense Hessian, row-major `n × n`.
    fn hess(&mut self, x: &[f64]) -> Result<Vec<f64>, Self::Error>;
    /// Whether the caller supplied a Hessian-vector product (otherwise products use `hess`).
    fn has_hessp(&self) -> bool;
    fn hessp(&mut self, x: &[f64], p: &[f64]) -> Result<Vec<f64>, Self::Error>;
    /// Called after every iteration with the current iterate; `true` stops the solve.
    fn callback(&mut self, x: &[f64], fun: f64) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Subproblem {
    CgSteihaug,
    Dogleg,
    /// `maxiter` bounds the λ iterations per subproblem (SciPy `subproblem_maxiter`, default 25).
    Exact {
        maxiter: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrustParams {
    pub initial_trust_radius: f64,
    pub max_trust_radius: f64,
    pub eta: f64,
    pub gtol: f64,
    pub maxiter: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TrustOutcome {
    pub x: Vec<f64>,
    pub fun: f64,
    pub jac: Vec<f64>,
    pub nit: usize,
    /// SciPy `warnflag`: 0 success, 1 maxiter, 2 no predicted improvement, 3 linalg error.
    pub status: u8,
    pub stopped_by_callback: bool,
}

pub(crate) fn status_message(status: u8) -> &'static str {
    match status {
        0 => "Optimization terminated successfully.",
        1 => "Maximum number of iterations has been exceeded.",
        2 => "A bad approximation caused failure to predict improvement.",
        _ => "A linalg error occurred, such as a non-psd Hessian.",
    }
}

enum SolveError<E> {
    /// SciPy `np.linalg.LinAlgError` (dogleg's Cholesky of an indefinite Hessian).
    LinAlg,
    Objective(E),
}

impl<E> From<E> for SolveError<E> {
    fn from(err: E) -> Self {
        Self::Objective(err)
    }
}

/// Quantities `IterativeSubproblem.__init__` derives from the Hessian, plus its λ state.
struct ExactState {
    gershgorin_lb: f64,
    gershgorin_ub: f64,
    hess_inf: f64,
    hess_fro: f64,
    close_to_zero: f64,
    previous_tr_radius: f64,
    lambda_lb: f64,
}

/// SciPy's `BaseQuadraticSubproblem`: the quadratic model at one iterate, whose values are
/// evaluated on demand and then kept.
struct Model {
    x: Vec<f64>,
    f: Option<f64>,
    g: Option<Vec<f64>>,
    h: Option<Vec<f64>>,
    g_mag: Option<f64>,
    cauchy_point: Option<Vec<f64>>,
    newton_point: Option<Vec<f64>>,
    exact: Option<ExactState>,
}

impl Model {
    fn new<O: TrustObjective>(
        x: Vec<f64>,
        subproblem: Subproblem,
        obj: &mut O,
    ) -> Result<Self, O::Error> {
        let mut model = Self {
            x,
            f: None,
            g: None,
            h: None,
            g_mag: None,
            cauchy_point: None,
            newton_point: None,
            exact: None,
        };
        // `IterativeSubproblem.__init__` reads the Hessian at once (its Gershgorin bounds and
        // norms), so trust-exact evaluates it at every proposed point, accepted or not.
        if matches!(subproblem, Subproblem::Exact { .. }) {
            let n = model.x.len();
            let h = model.hess(obj)?.to_vec();
            let (gershgorin_lb, gershgorin_ub) = gershgorin_bounds(&h, n);
            let hess_inf = (0..n)
                .map(|i| h[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f64>())
                .fold(f64::NEG_INFINITY, f64::max);
            let hess_fro = h.iter().map(|v| v * v).sum::<f64>().sqrt();
            model.exact = Some(ExactState {
                gershgorin_lb,
                gershgorin_ub,
                hess_inf,
                hess_fro,
                close_to_zero: n as f64 * f64::EPSILON * hess_inf,
                previous_tr_radius: -1.0,
                lambda_lb: 0.0,
            });
        }
        Ok(model)
    }

    fn fun<O: TrustObjective>(&mut self, obj: &mut O) -> Result<f64, O::Error> {
        if let Some(f) = self.f {
            return Ok(f);
        }
        let f = obj.fun(&self.x)?;
        self.f = Some(f);
        Ok(f)
    }

    fn jac<O: TrustObjective>(&mut self, obj: &mut O) -> Result<&[f64], O::Error> {
        if self.g.is_none() {
            self.g = Some(obj.grad(&self.x)?);
        }
        Ok(self.g.as_deref().unwrap_or_default())
    }

    fn hess<O: TrustObjective>(&mut self, obj: &mut O) -> Result<&[f64], O::Error> {
        if self.h.is_none() {
            self.h = Some(obj.hess(&self.x)?);
        }
        Ok(self.h.as_deref().unwrap_or_default())
    }

    fn jac_mag<O: TrustObjective>(&mut self, obj: &mut O) -> Result<f64, O::Error> {
        if let Some(mag) = self.g_mag {
            return Ok(mag);
        }
        let mag = norm2(self.jac(obj)?);
        self.g_mag = Some(mag);
        Ok(mag)
    }

    fn hessp<O: TrustObjective>(&mut self, obj: &mut O, p: &[f64]) -> Result<Vec<f64>, O::Error> {
        if obj.has_hessp() {
            return obj.hessp(&self.x, p);
        }
        let n = p.len();
        let h = self.hess(obj)?;
        Ok(mat_vec(h, n, p))
    }

    /// The model value `f + gᵀp + ½ pᵀHp` (SciPy `BaseQuadraticSubproblem.__call__`).
    fn value<O: TrustObjective>(&mut self, obj: &mut O, p: &[f64]) -> Result<f64, O::Error> {
        let f = self.fun(obj)?;
        let gp = dot(self.jac(obj)?, p);
        let hp = self.hessp(obj, p)?;
        Ok(f + gp + 0.5 * dot(p, &hp))
    }

    fn solve<O: TrustObjective>(
        &mut self,
        obj: &mut O,
        subproblem: Subproblem,
        trust_radius: f64,
    ) -> Result<(Vec<f64>, bool), SolveError<O::Error>> {
        match subproblem {
            Subproblem::CgSteihaug => Ok(self.solve_cg_steihaug(obj, trust_radius)?),
            Subproblem::Dogleg => self.solve_dogleg(obj, trust_radius),
            Subproblem::Exact { maxiter } => Ok(self.solve_exact(obj, trust_radius, maxiter)?),
        }
    }

    /// Nocedal & Wright algorithm 7.2 (SciPy `CGSteihaugSubproblem.solve`).
    fn solve_cg_steihaug<O: TrustObjective>(
        &mut self,
        obj: &mut O,
        trust_radius: f64,
    ) -> Result<(Vec<f64>, bool), O::Error> {
        let jac_mag = self.jac_mag(obj)?;
        let n = self.x.len();
        let tolerance = 0.5_f64.min(jac_mag.sqrt()) * jac_mag;
        if jac_mag < tolerance {
            return Ok((vec![0.0; n], false));
        }
        let mut z = vec![0.0; n];
        let mut r = self.jac(obj)?.to_vec();
        let mut d: Vec<f64> = r.iter().map(|v| -v).collect();
        loop {
            let bd = self.hessp(obj, &d)?;
            let dbd = dot(&d, &bd);
            if dbd <= 0.0 {
                let (ta, tb) = boundaries_intersections(&z, &d, trust_radius);
                let pa = axpy(&z, ta, &d);
                let pb = axpy(&z, tb, &d);
                let p_boundary = if self.value(obj, &pa)? < self.value(obj, &pb)? {
                    pa
                } else {
                    pb
                };
                return Ok((p_boundary, true));
            }
            let r_squared = dot(&r, &r);
            let alpha = r_squared / dbd;
            let z_next = axpy(&z, alpha, &d);
            if norm2(&z_next) >= trust_radius {
                let (_, tb) = boundaries_intersections(&z, &d, trust_radius);
                return Ok((axpy(&z, tb, &d), true));
            }
            let r_next = axpy(&r, alpha, &bd);
            let r_next_squared = dot(&r_next, &r_next);
            if r_next_squared.sqrt() < tolerance {
                return Ok((z_next, false));
            }
            let beta_next = r_next_squared / r_squared;
            d = r_next
                .iter()
                .zip(&d)
                .map(|(rn, di)| -rn + beta_next * di)
                .collect();
            z = z_next;
            r = r_next;
        }
    }

    /// Nocedal & Wright §4.1 (SciPy `DoglegSubproblem.solve`); the Hessian must be positive
    /// definite, and an indefinite one is SciPy's `LinAlgError`.
    fn solve_dogleg<O: TrustObjective>(
        &mut self,
        obj: &mut O,
        trust_radius: f64,
    ) -> Result<(Vec<f64>, bool), SolveError<O::Error>> {
        let n = self.x.len();
        if self.newton_point.is_none() {
            let g = self.jac(obj)?.to_vec();
            let h = self.hess(obj)?;
            let (u, info) = cholesky_upper(h, n);
            if info != 0 {
                return Err(SolveError::LinAlg);
            }
            let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
            self.newton_point = Some(cho_solve_upper(&u, n, &neg_g));
        }
        let p_best = self.newton_point.clone().unwrap_or_default();
        if norm2(&p_best) < trust_radius {
            return Ok((p_best, false));
        }
        if self.cauchy_point.is_none() {
            let g = self.jac(obj)?.to_vec();
            let bg = self.hessp(obj, &g)?;
            let scale = -(dot(&g, &g) / dot(&g, &bg));
            self.cauchy_point = Some(g.iter().map(|v| scale * v).collect());
        }
        let p_u = self.cauchy_point.clone().unwrap_or_default();
        let p_u_norm = norm2(&p_u);
        if p_u_norm >= trust_radius {
            let scale = trust_radius / p_u_norm;
            return Ok((p_u.iter().map(|v| v * scale).collect(), true));
        }
        let direction: Vec<f64> = p_best.iter().zip(&p_u).map(|(b, u)| b - u).collect();
        let (_, tb) = boundaries_intersections(&p_u, &direction, trust_radius);
        Ok((axpy(&p_u, tb, &direction), true))
    }

    /// SciPy `IterativeSubproblem._initial_values`.
    fn exact_initial_values(
        &self,
        tr_radius: f64,
        jac_mag: f64,
        h: &[f64],
        n: usize,
    ) -> (f64, f64, f64) {
        let Some(state) = self.exact.as_ref() else {
            return (0.0, 0.0, 0.0);
        };
        let lambda_ub = 0.0_f64.max(
            jac_mag / tr_radius
                + (-state.gershgorin_lb)
                    .min(state.hess_fro)
                    .min(state.hess_inf),
        );
        let min_diag = (0..n).map(|i| h[i * n + i]).fold(f64::INFINITY, f64::min);
        let mut lambda_lb = 0.0_f64
            .max(-min_diag)
            .max(jac_mag / tr_radius - state.gershgorin_ub.min(state.hess_fro).min(state.hess_inf));
        if tr_radius < state.previous_tr_radius {
            lambda_lb = state.lambda_lb.max(lambda_lb);
        }
        let lambda_initial = if lambda_lb == 0.0 {
            0.0
        } else {
            (lambda_lb * lambda_ub)
                .sqrt()
                .max(lambda_lb + EXACT_UPDATE_COEFF * (lambda_ub - lambda_lb))
        };
        (lambda_initial, lambda_lb, lambda_ub)
    }

    /// SciPy `IterativeSubproblem.solve`, including its bookkeeping: after a successful trial
    /// factorization of `H + λ_new·I` in the "inside the boundary" branch, SciPy moves to λ_new
    /// but keeps the previous factor `U` for the next pass (it assigns the new factor to `c`).
    /// That is reproduced, because it decides the iteration path.
    fn solve_exact<O: TrustObjective>(
        &mut self,
        obj: &mut O,
        tr_radius: f64,
        maxiter: usize,
    ) -> Result<(Vec<f64>, bool), O::Error> {
        let n = self.x.len();
        let jac_mag = self.jac_mag(obj)?;
        let jac = self.jac(obj)?.to_vec();
        let hess = self.hess(obj)?.to_vec();
        let (mut lambda_current, mut lambda_lb, mut lambda_ub) =
            self.exact_initial_values(tr_radius, jac_mag, &hess, n);
        let close_to_zero = self.exact.as_ref().map_or(0.0, |s| s.close_to_zero);
        let neg_jac: Vec<f64> = jac.iter().map(|v| -v).collect();

        let mut hits_boundary = true;
        let mut already_factorized = false;
        let mut h_matrix = shifted(&hess, n, lambda_current);
        let mut u = vec![0.0; n * n];
        let mut info = 0_usize;
        let mut p = vec![0.0; n];
        let mut niter = 0_usize;
        while niter < maxiter {
            if already_factorized {
                already_factorized = false;
            } else {
                h_matrix = shifted(&hess, n, lambda_current);
                (u, info) = cholesky_upper(&h_matrix, n);
            }
            niter += 1;

            if info == 0 && jac_mag > close_to_zero {
                p = cho_solve_upper(&u, n, &neg_jac);
                let p_norm = norm2(&p);
                if p_norm <= tr_radius && lambda_current == 0.0 {
                    hits_boundary = false;
                    break;
                }
                let w = solve_upper_transposed(&u, n, &p);
                let w_norm = norm2(&w);
                let delta_lambda = (p_norm / w_norm).powi(2) * (p_norm - tr_radius) / tr_radius;
                let lambda_new = lambda_current + delta_lambda;

                if p_norm < tr_radius {
                    let (s_min, z_min) = estimate_smallest_singular_value(&u, n);
                    let (ta, tb) = boundaries_intersections(&p, &z_min, tr_radius);
                    // Python's `min([ta, tb], key=abs)` keeps the first on a tie.
                    let step_len = if tb.abs() < ta.abs() { tb } else { ta };
                    let quadratic_term = dot(&p, &mat_vec(&h_matrix, n, &p));
                    let relative_error = (step_len * step_len * s_min * s_min)
                        / (quadratic_term + lambda_current * tr_radius * tr_radius);
                    if relative_error <= EXACT_K_HARD {
                        for (pi, zi) in p.iter_mut().zip(&z_min) {
                            *pi += step_len * zi;
                        }
                        break;
                    }
                    lambda_ub = lambda_current;
                    lambda_lb = lambda_lb.max(lambda_current - s_min * s_min);
                    h_matrix = shifted(&hess, n, lambda_new);
                    let (_, trial_info) = cholesky_upper(&h_matrix, n);
                    info = trial_info;
                    if info == 0 {
                        lambda_current = lambda_new;
                        already_factorized = true;
                    } else {
                        lambda_lb = lambda_lb.max(lambda_new);
                        lambda_current = (lambda_lb * lambda_ub)
                            .abs()
                            .sqrt()
                            .max(lambda_lb + EXACT_UPDATE_COEFF * (lambda_ub - lambda_lb));
                    }
                } else {
                    let relative_error = (p_norm - tr_radius).abs() / tr_radius;
                    if relative_error <= EXACT_K_EASY {
                        break;
                    }
                    lambda_lb = lambda_current;
                    lambda_current = lambda_new;
                }
            } else if info == 0 && jac_mag <= close_to_zero {
                if lambda_current == 0.0 {
                    p = vec![0.0; n];
                    hits_boundary = false;
                    break;
                }
                let (s_min, z_min) = estimate_smallest_singular_value(&u, n);
                let step_len = tr_radius;
                p = z_min.iter().map(|z| step_len * z).collect();
                if step_len * step_len * s_min * s_min
                    <= EXACT_K_HARD * lambda_current * tr_radius * tr_radius
                {
                    break;
                }
                lambda_ub = lambda_current;
                lambda_lb = lambda_lb.max(lambda_current - s_min * s_min);
                lambda_current = (lambda_lb * lambda_ub)
                    .sqrt()
                    .max(lambda_lb + EXACT_UPDATE_COEFF * (lambda_ub - lambda_lb));
            } else {
                let (delta, v) = singular_leading_submatrix(&h_matrix, &u, n, info);
                let v_norm = norm2(&v);
                lambda_lb = lambda_lb.max(lambda_current + delta / (v_norm * v_norm));
                lambda_current = (lambda_lb * lambda_ub)
                    .abs()
                    .sqrt()
                    .max(lambda_lb + EXACT_UPDATE_COEFF * (lambda_ub - lambda_lb));
            }
        }

        if let Some(state) = self.exact.as_mut() {
            state.lambda_lb = lambda_lb;
            state.previous_tr_radius = tr_radius;
        }
        Ok((p, hits_boundary))
    }
}

const EXACT_UPDATE_COEFF: f64 = 0.01;
const EXACT_K_EASY: f64 = 0.1;
const EXACT_K_HARD: f64 = 0.2;

/// SciPy `_minimize_trust_region`. Parameters are validated by the caller.
pub(crate) fn minimize<O: TrustObjective>(
    obj: &mut O,
    x0: &[f64],
    subproblem: Subproblem,
    params: TrustParams,
) -> Result<TrustOutcome, O::Error> {
    let mut trust_radius = params.initial_trust_radius;
    let mut x = x0.to_vec();
    let mut m = Model::new(x.clone(), subproblem, obj)?;
    if obj.has_hess() {
        m.hess(obj)?;
    }
    let mut k = 0_usize;
    let mut warnflag = 0_u8;
    let mut stopped_by_callback = false;

    // `while m.jac_mag >= gtol` — a NaN gradient norm ends the loop, as in SciPy.
    while m.jac_mag(obj)? >= params.gtol {
        let (p, hits_boundary) = match m.solve(obj, subproblem, trust_radius) {
            Ok(step) => step,
            Err(SolveError::LinAlg) => {
                warnflag = 3;
                break;
            }
            Err(SolveError::Objective(err)) => return Err(err),
        };
        let predicted_value = m.value(obj, &p)?;
        let x_proposed: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + pi).collect();
        let mut m_proposed = Model::new(x_proposed.clone(), subproblem, obj)?;

        let actual_reduction = m.fun(obj)? - m_proposed.fun(obj)?;
        let predicted_reduction = m.fun(obj)? - predicted_value;
        if predicted_reduction <= 0.0 {
            warnflag = 2;
            break;
        }
        let rho = actual_reduction / predicted_reduction;

        if rho < 0.25 {
            trust_radius *= 0.25;
        } else if rho > 0.75 && hits_boundary {
            trust_radius = (2.0 * trust_radius).min(params.max_trust_radius);
        }
        if rho > params.eta {
            x = x_proposed;
            m = m_proposed;
        }
        k += 1;

        let current_fun = m.fun(obj)?;
        if obj.callback(&x, current_fun) {
            stopped_by_callback = true;
            break;
        }
        if m.jac_mag(obj)? < params.gtol {
            warnflag = 0;
            break;
        }
        if k >= params.maxiter {
            warnflag = 1;
            break;
        }
    }

    let fun = m.fun(obj)?;
    let jac = m.jac(obj)?.to_vec();
    Ok(TrustOutcome {
        x,
        fun,
        jac,
        nit: k,
        status: warnflag,
        stopped_by_callback,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn axpy(z: &[f64], t: f64, d: &[f64]) -> Vec<f64> {
    z.iter().zip(d).map(|(zi, di)| zi + t * di).collect()
}

fn mat_vec(a: &[f64], n: usize, p: &[f64]) -> Vec<f64> {
    (0..n).map(|i| dot(&a[i * n..(i + 1) * n], p)).collect()
}

fn shifted(h: &[f64], n: usize, lambda: f64) -> Vec<f64> {
    let mut out = h.to_vec();
    for i in 0..n {
        out[i * n + i] += lambda;
    }
    out
}

/// The two roots `t` of `‖z + t·d‖ = Δ`, low to high (SciPy `get_boundaries_intersections`).
fn boundaries_intersections(z: &[f64], d: &[f64], trust_radius: f64) -> (f64, f64) {
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - trust_radius * trust_radius;
    let sqrt_discriminant = (b * b - 4.0 * a * c).sqrt();
    let aux = b + sqrt_discriminant.copysign(b);
    let ta = -aux / (2.0 * a);
    let tb = -2.0 * c / aux;
    if tb < ta { (tb, ta) } else { (ta, tb) }
}

/// LAPACK `potrf` (upper): `A = UᵀU`. On failure `info = k` (1-based) is the first column whose
/// pivot is not positive, and `U`'s leading `k-1` columns — including column `k-1` above the
/// diagonal — are complete, which is what [`singular_leading_submatrix`] reads.
fn cholesky_upper(a: &[f64], n: usize) -> (Vec<f64>, usize) {
    let mut u = vec![0.0; n * n];
    for j in 0..n {
        let mut ajj = a[j * n + j];
        for k in 0..j {
            ajj -= u[k * n + j] * u[k * n + j];
        }
        // `!(ajj > 0)` also catches NaN, as `potrf`'s `AJJ.LE.ZERO .OR. DISNAN(AJJ)` does.
        if ajj.is_nan() || ajj <= 0.0 {
            return (u, j + 1);
        }
        let ujj = ajj.sqrt();
        u[j * n + j] = ujj;
        for c in j + 1..n {
            let mut value = a[j * n + c];
            for k in 0..j {
                value -= u[k * n + j] * u[k * n + c];
            }
            u[j * n + c] = value / ujj;
        }
    }
    (u, 0)
}

/// Solve `Uᵀ w = b` for upper-triangular `U`.
fn solve_upper_transposed(u: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut value = b[i];
        for k in 0..i {
            value -= u[k * n + i] * w[k];
        }
        w[i] = value / u[i * n + i];
    }
    w
}

/// Solve `U v = b` for the leading `m × m` block of upper-triangular `U` (row stride `n`).
fn solve_upper(u: &[f64], n: usize, m: usize, b: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; m];
    for i in (0..m).rev() {
        let mut value = b[i];
        for k in i + 1..m {
            value -= u[i * n + k] * v[k];
        }
        v[i] = value / u[i * n + i];
    }
    v
}

/// `cho_solve((U, False), b)`: solve `UᵀU x = b`.
fn cho_solve_upper(u: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let y = solve_upper_transposed(u, n, b);
    solve_upper(u, n, n, &y)
}

/// SciPy `gershgorin_bounds`: bounds on the eigenvalues of `H` from Gershgorin discs.
fn gershgorin_bounds(h: &[f64], n: usize) -> (f64, f64) {
    let mut lb = f64::INFINITY;
    let mut ub = f64::NEG_INFINITY;
    for i in 0..n {
        let row = &h[i * n..(i + 1) * n];
        let diag = row[i];
        let row_sum: f64 = row.iter().map(|v| v.abs()).sum();
        lb = lb.min(diag + diag.abs() - row_sum);
        ub = ub.max(diag - diag.abs() + row_sum);
    }
    (lb, ub)
}

/// SciPy `estimate_smallest_singular_value` (Cline, Moler, Stewart & Wilkinson): an estimate
/// `s_min` of the smallest singular value of upper-triangular `U` and a unit `z_min` with
/// `‖U z_min‖ ≈ s_min`.
fn estimate_smallest_singular_value(u: &[f64], n: usize) -> (f64, Vec<f64>) {
    let mut p = vec![0.0; n];
    let mut w = vec![0.0; n];
    for k in 0..n {
        let ukk = u[k * n + k];
        let wp = (1.0 - p[k]) / ukk;
        let wm = (-1.0 - p[k]) / ukk;
        let row = &u[k * n + k + 1..(k + 1) * n];
        let pp: Vec<f64> = p[k + 1..]
            .iter()
            .zip(row)
            .map(|(pi, ui)| pi + ui * wp)
            .collect();
        let pm: Vec<f64> = p[k + 1..]
            .iter()
            .zip(row)
            .map(|(pi, ui)| pi + ui * wm)
            .collect();
        let l1 = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>();
        if wp.abs() + l1(&pp) >= wm.abs() + l1(&pm) {
            w[k] = wp;
            p[k + 1..].copy_from_slice(&pp);
        } else {
            w[k] = wm;
            p[k + 1..].copy_from_slice(&pm);
        }
    }
    let v = solve_upper(u, n, n, &w);
    let v_norm = norm2(&v);
    let w_norm = norm2(&w);
    (w_norm / v_norm, v.iter().map(|x| x / v_norm).collect())
}

/// SciPy `singular_leading_submatrix`: for `potrf` failing at column `k` (1-based), the shift
/// `δ` that makes the leading `k × k` block singular and a vector `v` in its null space.
fn singular_leading_submatrix(a: &[f64], u: &[f64], n: usize, k: usize) -> (f64, Vec<f64>) {
    let col = k - 1;
    let delta = (0..col)
        .map(|i| u[i * n + col] * u[i * n + col])
        .sum::<f64>()
        - a[col * n + col];
    let mut v = vec![0.0; n];
    v[col] = 1.0;
    if k != 1 {
        let rhs: Vec<f64> = (0..col).map(|i| -u[i * n + col]).collect();
        let head = solve_upper(u, n, col, &rhs);
        v[..col].copy_from_slice(&head);
    }
    (delta, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadratic `½xᵀAx − bᵀx` with fixed `A`, counting evaluations.
    struct Quadratic {
        a: Vec<f64>,
        b: Vec<f64>,
        n: usize,
        with_hessp: bool,
        hessp_calls: usize,
    }

    impl TrustObjective for Quadratic {
        type Error = ();
        fn fun(&mut self, x: &[f64]) -> Result<f64, ()> {
            Ok(0.5 * dot(x, &mat_vec(&self.a, self.n, x)) - dot(&self.b, x))
        }
        fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, ()> {
            Ok(mat_vec(&self.a, self.n, x)
                .iter()
                .zip(&self.b)
                .map(|(ax, b)| ax - b)
                .collect())
        }
        fn has_hess(&self) -> bool {
            true
        }
        fn hess(&mut self, _x: &[f64]) -> Result<Vec<f64>, ()> {
            Ok(self.a.clone())
        }
        fn has_hessp(&self) -> bool {
            self.with_hessp
        }
        fn hessp(&mut self, _x: &[f64], p: &[f64]) -> Result<Vec<f64>, ()> {
            self.hessp_calls += 1;
            Ok(mat_vec(&self.a, self.n, p))
        }
        fn callback(&mut self, _x: &[f64], _fun: f64) -> bool {
            false
        }
    }

    fn quadratic(a: Vec<f64>, b: Vec<f64>) -> Quadratic {
        let n = b.len();
        Quadratic {
            a,
            b,
            n,
            with_hessp: false,
            hessp_calls: 0,
        }
    }

    #[test]
    fn steihaug_stops_on_the_boundary_and_on_negative_curvature() {
        // Positive definite, Newton step (10, 10) far outside Δ = 1: CG leaves the region on its
        // first step and must return the boundary point along −g.
        let mut obj = quadratic(vec![1.0, 0.0, 0.0, 1.0], vec![10.0, 10.0]);
        let mut model = Model::new(vec![0.0, 0.0], Subproblem::CgSteihaug, &mut obj).unwrap();
        let (p, hits) = model.solve_cg_steihaug(&mut obj, 1.0).unwrap();
        assert!(hits);
        assert!((norm2(&p) - 1.0).abs() < 1e-14, "|p| = {}", norm2(&p));
        assert!((p[0] - p[1]).abs() < 1e-15 && p[0] > 0.0, "p = {p:?}");

        // Indefinite: the first direction −g = (−1, 0) has curvature −1 < 0, so CG must stop on
        // the boundary at once — the lower-model point of the two intersections, p = (−2, 0)
        // (model −4) rather than (2, 0) (model 0).
        let mut obj = quadratic(vec![-1.0, 0.0, 0.0, 1.0], vec![-1.0, 0.0]);
        let mut model = Model::new(vec![0.0, 0.0], Subproblem::CgSteihaug, &mut obj).unwrap();
        let (p, hits) = model.solve_cg_steihaug(&mut obj, 2.0).unwrap();
        assert!(hits);
        assert_eq!(p, vec![-2.0, 0.0]);

        // Interior, truncated: A = diag(2, 4), b = (1, 1). The forcing tolerance is
        // min(0.5, √‖g‖)·‖g‖ = 0.707 and one CG step leaves a residual of 0.471, so SciPy stops
        // there at (1/3, 1/3) — not at the Newton point (0.5, 0.25).
        let mut obj = quadratic(vec![2.0, 0.0, 0.0, 4.0], vec![1.0, 1.0]);
        let mut model = Model::new(vec![0.0, 0.0], Subproblem::CgSteihaug, &mut obj).unwrap();
        let (p, hits) = model.solve_cg_steihaug(&mut obj, 10.0).unwrap();
        assert!(!hits);
        assert!(
            (p[0] - 1.0 / 3.0).abs() < 1e-15 && (p[1] - 1.0 / 3.0).abs() < 1e-15,
            "p = {p:?}"
        );
        // With a small gradient the tolerance (√‖g‖·‖g‖) is tight and CG runs to the Newton point.
        let mut obj = quadratic(vec![2.0, 0.0, 0.0, 4.0], vec![1e-4, 1e-4]);
        let mut model = Model::new(vec![0.0, 0.0], Subproblem::CgSteihaug, &mut obj).unwrap();
        let (p, hits) = model.solve_cg_steihaug(&mut obj, 10.0).unwrap();
        assert!(!hits);
        assert!(
            (p[0] - 0.5e-4).abs() < 1e-18 && (p[1] - 0.25e-4).abs() < 1e-18,
            "p = {p:?}"
        );
    }

    #[test]
    fn dogleg_refuses_an_indefinite_hessian() {
        let mut obj = quadratic(vec![-1.0, 0.0, 0.0, 1.0], vec![-1.0, 0.0]);
        let mut model = Model::new(vec![0.0, 0.0], Subproblem::Dogleg, &mut obj).unwrap();
        assert!(matches!(
            model.solve_dogleg(&mut obj, 1.0),
            Err(SolveError::LinAlg)
        ));
        // The same problem through the driver ends with SciPy's status 3 at iteration 0.
        let params = TrustParams {
            initial_trust_radius: 1.0,
            max_trust_radius: 1000.0,
            eta: 0.15,
            gtol: 1e-4,
            maxiter: 100,
        };
        let out = minimize(&mut obj, &[0.0, 0.0], Subproblem::Dogleg, params).unwrap();
        assert_eq!((out.status, out.nit), (3, 0));
    }

    #[test]
    fn exact_subproblem_matches_the_eigen_solution_on_the_boundary() {
        // H = diag(−2, 1, 3) (indefinite), g = (1, 1, 1), Δ = 1. The global minimizer of the
        // model on ‖p‖ ≤ Δ is p(λ) = −(H + λI)⁻¹g with ‖p(λ)‖ = Δ, λ > 2 (Moré–Sorensen).
        let h = vec![-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0];
        let mut obj = quadratic(h, vec![-1.0, -1.0, -1.0]);
        let mut model =
            Model::new(vec![0.0; 3], Subproblem::Exact { maxiter: 25 }, &mut obj).unwrap();
        let (p, hits) = model.solve_exact(&mut obj, 1.0, 25).unwrap();
        assert!(hits);
        // Bisect the secular equation for the reference λ*.
        let norm_at = |lambda: f64| {
            ((1.0 / (lambda - 2.0)).powi(2)
                + (1.0 / (lambda + 1.0)).powi(2)
                + (1.0 / (lambda + 3.0)).powi(2))
            .sqrt()
        };
        let (mut lo, mut hi) = (2.0 + 1e-12, 100.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if norm_at(mid) > 1.0 {
                lo = mid
            } else {
                hi = mid
            }
        }
        let lambda = 0.5 * (lo + hi);
        let reference = [
            -1.0 / (lambda - 2.0),
            -1.0 / (lambda + 1.0),
            -1.0 / (lambda + 3.0),
        ];
        // SciPy's stopping rule accepts ‖p‖ within k_easy = 10% of Δ (either side), so the step
        // is compared to the reference to that slack, and the model value it reaches must be
        // at least 90% of the optimum's decrease.
        let model_at = |q: &[f64]| {
            dot(&[1.0, 1.0, 1.0], q) + 0.5 * (-2.0 * q[0] * q[0] + q[1] * q[1] + 3.0 * q[2] * q[2])
        };
        let (ours, best) = (model_at(&p), model_at(&reference));
        assert!(
            (norm2(&p) - 1.0).abs() <= EXACT_K_EASY,
            "|p| = {}",
            norm2(&p)
        );
        assert!(ours <= 0.9 * best, "model {ours} vs optimum {best}");
        let direction_error: f64 = p
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            direction_error < 0.1,
            "p = {p:?}, reference = {reference:?}"
        );
    }

    #[test]
    fn potrf_failure_reports_the_column_and_its_complete_leading_part() {
        // Leading 1×1 block is fine, the 2×2 block is singular + negative: info = 2.
        let a = vec![4.0, 2.0, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0];
        let (u, info) = cholesky_upper(&a, 3);
        assert_eq!(info, 2);
        assert_eq!((u[0], u[1]), (2.0, 1.0));
        let (delta, v) = singular_leading_submatrix(&a, &u, 3, info);
        // δ = u₀₁² − a₁₁ = 1 − 0.5; (A + δ e₁e₁ᵀ) v = 0 on the leading block.
        assert_eq!(delta, 0.5);
        assert_eq!(v, vec![-0.5, 1.0, 0.0]);
        let r0 = 4.0 * v[0] + 2.0 * v[1];
        let r1 = 2.0 * v[0] + (0.5 + delta) * v[1];
        assert_eq!((r0, r1), (0.0, 0.0));
    }
}
