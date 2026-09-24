//! SLSQP — Kraft's sequential least-squares quadratic programming (ACM TOMS 733, 1994;
//! DFVLR-FB 88-28, 1988), ported from SciPy 1.17's C translation (`scipy/optimize/__slsqp.c`
//! and `__nnls.c`, BSD).
//!
//! The QP subproblem at each iterate is solved as a least-squares problem with linear equality
//! and inequality constraints (LSEI): the equality block by an RQ factorization, the rest as an
//! LSI problem reduced to a least-distance problem (LDP) whose dual is solved by Lawson–Hanson
//! NNLS. The Lagrangian Hessian is a BFGS approximation held as `L·D·Lᵀ` in packed storage and
//! updated by Fletcher–Powell rank-one modifications; the step length comes from an L1
//! exact-penalty line search. Inconsistent linearizations are handled by Kraft's augmented
//! problem with one extra variable. The LAPACK/BLAS kernels the C code calls (Householder QR/RQ
//! and their appliers, `dlarfgp`, `dlartgp`, triangular solves, packed products) are written
//! out here in safe Rust, column-major, with the same argument conventions.
//!
//! Exit modes follow SciPy: 0 success, 2 more equality constraints than variables, 3 LSQ
//! iteration limit, 4 inconsistent inequality constraints, 5/6 singular E/C in the LSQ
//! subproblem, 7 rank-deficient equality-constrained subproblem, 8 positive directional
//! derivative in the line search, 9 iteration limit.

/// LAPACK `dlamch('P')` and the value SciPy's C code hardcodes as `epsmach`.
const EPSMACH: f64 = 2.220_446_049_250_313e-16;
/// LAPACK `dlamch('S') / dlamch('E')`, the rescaling threshold in `dlarfg`/`dlarfgp`.
const SMLNUM: f64 = f64::MIN_POSITIVE / (f64::EPSILON * 0.5);

/// SciPy's `exit_modes` messages.
pub(crate) fn exit_message(mode: i32) -> &'static str {
    match mode {
        -1 => "Gradient evaluation required (g & a)",
        0 => "Optimization terminated successfully",
        1 => "Function evaluation required (f & c)",
        2 => "More equality constraints than independent variables",
        3 => "More than 3*n iterations in LSQ subproblem",
        4 => "Inequality constraints incompatible",
        5 => "Singular matrix E in LSQ subproblem",
        6 => "Singular matrix C in LSQ subproblem",
        7 => "Rank-deficient equality constraint subproblem HFTI",
        8 => "Positive directional derivative for linesearch",
        _ => "Iteration limit reached",
    }
}

/// The objective, the constraints and their derivatives, as SLSQP asks for them.
pub(crate) trait SlsqpProblem {
    type Error;
    /// f(x); writes the `m` constraint values into `d`, equalities first (`d_i = 0` wanted),
    /// then inequalities (`d_i >= 0` wanted).
    fn eval_fc(&mut self, x: &[f64], d: &mut [f64]) -> Result<f64, Self::Error>;
    /// ∇f(x) into `g` and the constraint Jacobian into `c` (column-major, leading dimension
    /// `max(m, 1)`). `fx` is f at this same `x`.
    fn eval_gc(
        &mut self,
        x: &[f64],
        fx: f64,
        g: &mut [f64],
        c: &mut [f64],
    ) -> Result<(), Self::Error>;
    /// Called once per major iteration; `true` stops the solver.
    fn callback(&mut self, x: &[f64], fx: f64) -> bool;
}

pub(crate) struct SlsqpOutcome {
    pub mode: i32,
    pub iter: usize,
    pub fx: f64,
    pub grad: Vec<f64>,
    pub stopped_by_callback: bool,
}

/// Clamp `x` into the finite bounds (NaN marks an absent bound), as SciPy's `slsqp` wrapper
/// does before every function or gradient evaluation.
fn clamp_to_bounds(x: &mut [f64], xl: &[f64], xu: &[f64]) {
    for ((xi, &lo), &hi) in x.iter_mut().zip(xl).zip(xu) {
        if !lo.is_nan() && *xi < lo {
            *xi = lo;
        } else if !hi.is_nan() && *xi > hi {
            *xi = hi;
        }
    }
}

/// Run SLSQP from `x` (updated in place) with bounds `xl`/`xu` (NaN = unbounded), `m`
/// constraints of which the first `meq` are equalities, accuracy `acc` (SciPy `ftol`) and at
/// most `maxiter` major iterations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<P: SlsqpProblem>(
    problem: &mut P,
    x: &mut [f64],
    xl: &[f64],
    xu: &[f64],
    m: usize,
    meq: usize,
    acc: f64,
    maxiter: usize,
) -> Result<SlsqpOutcome, P::Error> {
    let n = x.len();
    let lda = m.max(1);
    let n1 = n + 1;
    let n2 = n1 * n / 2;

    clamp_to_bounds(x, xl, xu);
    let mut d = vec![0.0; m];
    let mut fx = problem.eval_fc(x, &mut d)?;
    let mut g = vec![0.0; n];
    let mut c = vec![0.0; lda * n];
    problem.eval_gc(x, fx, &mut g, &mut c)?;

    let mut bfgs = vec![0.0; n2];
    let mut x0 = vec![0.0; n];
    let mut mu = vec![0.0; m];
    let mut s = vec![0.0; n1];
    let mut u = vec![0.0; n1];
    let mut v = vec![0.0; n1];
    let mut mult = vec![0.0; m + 2 * n + 2];

    let acc = acc.abs();
    let tol = 10.0 * acc;
    let mut iter = 0_usize;
    let mut reset = 0_usize;
    let mut f0 = 0.0_f64;
    let mut h4: f64;
    // SciPy's C body re-initializes `badlin` on every reverse-communication entry, so it only
    // guards the convergence tests reached before the next function or gradient evaluation.
    let mut badlin = false;

    let finish = |mode: i32, iter: usize, fx: f64, g: Vec<f64>, cb: bool| SlsqpOutcome {
        mode,
        iter,
        fx,
        grad: g,
        stopped_by_callback: cb,
    };

    let penalty_term = |d: &[f64], j: usize| -> f64 {
        let eq = if j < meq { d[j] } else { 0.0 };
        (-d[j]).max(eq)
    };

    'reset: loop {
        reset += 1;
        if reset > 5 {
            // Relaxed convergence after repeated positive directional derivatives.
            let h3: f64 = (0..m).map(|j| penalty_term(&d, j)).sum();
            let mode = if ((fx - f0).abs() < tol || nrm2(&s[..n]) < tol)
                && h3 < tol
                && !badlin
                && !fx.is_nan()
            {
                0
            } else {
                8
            };
            return Ok(finish(mode, iter, fx, g, false));
        }
        bfgs.fill(0.0);
        let mut j = 0;
        for i in 0..n {
            bfgs[j] = 1.0;
            j += n - i;
        }

        loop {
            if iter >= maxiter {
                return Ok(finish(9, iter, fx, g, false));
            }
            iter += 1;

            // Search direction as solution of the QP subproblem.
            for i in 0..n {
                u[i] = -x[i] + xl[i];
                v[i] = -x[i] + xu[i];
            }
            h4 = 1.0;
            let mut mode = lsq(
                m, meq, n, None, &bfgs, &g, &c, &d, &mut u, &mut v, &mut s, &mut mult,
            );
            badlin = false;
            if mode == 6 && n == meq {
                mode = 4;
            }
            if mode == 4 {
                // Inconsistent linearization: Kraft's augmented problem.
                badlin = true;
                s[..n].fill(0.0);
                let mut rho = 100.0;
                let mut inconsistent = 0;
                loop {
                    mode = lsq(
                        m,
                        meq,
                        n,
                        Some(rho),
                        &bfgs,
                        &g,
                        &c,
                        &d,
                        &mut u,
                        &mut v,
                        &mut s,
                        &mut mult,
                    );
                    h4 = 1.0 - s[n];
                    if mode == 4 {
                        rho *= 10.0;
                        inconsistent += 1;
                        if inconsistent > 5 {
                            return Ok(finish(4, iter, fx, g, false));
                        }
                        continue;
                    } else if mode != 1 {
                        return Ok(finish(mode, iter, fx, g, false));
                    }
                    break;
                }
            } else if mode != 1 {
                return Ok(finish(mode, iter, fx, g, false));
            }

            // Multipliers for the L1 test: v = ∇f − Cᵀ·mult.
            for i in 0..n {
                let col = &c[i * lda..i * lda + m];
                v[i] = g[i] - dot(col, &mult[..m]);
            }
            f0 = fx;
            x0.copy_from_slice(x);
            let gs = dot(&g, &s[..n]);
            let mut h1 = gs.abs();
            let mut h2 = 0.0;
            for j in 0..m {
                h2 += penalty_term(&d, j);
                let h3 = mult[j].abs();
                mu[j] = h3.max((mu[j] + h3) / 2.0);
                h1 += h3 * d[j].abs();
            }
            if h1 < acc && h2 < acc && !badlin && !fx.is_nan() {
                return Ok(finish(0, iter, fx, g, false));
            }
            let h1: f64 = (0..m).map(|j| mu[j] * penalty_term(&d, j)).sum();
            let t0 = fx + h1;
            let mut h3 = gs - h1 * h4;
            if h3 >= 0.0 {
                continue 'reset;
            }

            // Inexact line search on the L1 penalty function.
            let mut line = 0;
            let mut alpha = 1.0;
            let mut first_eval = true;
            loop {
                line += 1;
                h3 *= alpha;
                for si in &mut s[..n] {
                    *si *= alpha;
                }
                for i in 0..n {
                    x[i] = x0[i] + s[i];
                }
                clamp_to_bounds(x, xl, xu);
                fx = problem.eval_fc(x, &mut d)?;
                badlin = false;
                if first_eval {
                    first_eval = false;
                    if problem.callback(x, fx) {
                        return Ok(finish(1, iter, fx, g, true));
                    }
                }
                let t = fx + (0..m).map(|j| mu[j] * penalty_term(&d, j)).sum::<f64>();
                let h1 = t - t0;
                if h1 <= h3 / 10.0 || line > 10 {
                    break;
                }
                alpha = (h3 / (2.0 * (h3 - h1))).max(0.1);
            }

            let h3: f64 = (0..m).map(|j| penalty_term(&d, j)).sum();
            if ((fx - f0).abs() < acc || nrm2(&s[..n]) < acc) && h3 < acc && !badlin && !fx.is_nan()
            {
                return Ok(finish(0, iter, fx, g, false));
            }

            problem.eval_gc(x, fx, &mut g, &mut c)?;
            badlin = false;

            // BFGS update of the L·D·Lᵀ factors of the Lagrangian Hessian.
            for i in 0..n {
                let col = &c[i * lda..i * lda + m];
                u[i] = g[i] - dot(col, &mult[..m]) - v[i];
            }
            v[..n].copy_from_slice(&s[..n]);
            tpmv_lower_unit_trans(n, &bfgs, &mut v[..n]);
            let mut j = 0;
            for i in 0..n {
                v[i] *= bfgs[j];
                j += n - i;
            }
            tpmv_lower_unit(n, &bfgs, &mut v[..n]);

            let mut h1 = dot(&s[..n], &u[..n]);
            let h2 = dot(&s[..n], &v[..n]);
            let h3 = 0.2 * h2;
            if h1 < h3 {
                h4 = (h2 - h3) / (h2 - h1);
                h1 = h3;
                let one_minus = 1.0 - h4;
                for i in 0..n {
                    u[i] = h4 * u[i] + one_minus * v[i];
                }
            }
            if h1 == 0.0 || h2 == 0.0 {
                continue 'reset;
            }
            ldl_update(n, &mut bfgs, &mut u[..n], 1.0 / h1, &mut v[..n]);
            ldl_update(n, &mut bfgs, &mut v[..n], -1.0 / h2, &mut u[..n]);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QP subproblem: LSQ → LSEI → LSI → LDP → NNLS
// ──────────────────────────────────────────────────────────────────────────────

/// Minimize |A·x − b| s.t. E·x = f, G·x ≥ h, xl ≤ x ≤ xu, where A, b come from the packed
/// `L·D·Lᵀ` factor `lf` and the gradient (A = √D·Lᵀ, b = −(L√D)⁻¹·g). With `augment =
/// Some(weight)` the problem gets Kraft's extra variable for inconsistent linearizations.
/// `x`, `xl`, `xu` hold n+1 entries; multipliers go to `y` (equalities, then inequalities;
/// bound multipliers NaN).
#[allow(clippy::too_many_arguments)]
fn lsq(
    m: usize,
    meq: usize,
    n_orig: usize,
    augment: Option<f64>,
    lf: &[f64],
    gradx: &[f64],
    c: &[f64],
    d: &[f64],
    xl: &mut [f64],
    xu: &mut [f64],
    x: &mut [f64],
    y: &mut [f64],
) -> i32 {
    let mineq = m - meq;
    let mut n = n_orig;
    let ld = if augment.is_some() { n + 1 } else { n };
    if augment.is_some() {
        x[n] = 1.0;
        xl[n] = 0.0;
        xu[n] = 1.0;
    }

    // A = √D·Lᵀ (upper triangular) and b = −D^{-1/2}·L⁻¹·g.
    let mut wa = vec![0.0; ld * ld];
    let mut wb = vec![0.0; ld];
    let mut cursor = 0;
    for j in 0..n {
        let diag = lf[cursor].sqrt();
        cursor += 1;
        wa[j + j * ld] = diag;
        for i in (j + 1)..n {
            wa[j + i * ld] = lf[cursor] * diag;
            cursor += 1;
        }
    }
    wb[..n].copy_from_slice(&gradx[..n]);
    tpsv_lower_unit(n, lf, &mut wb[..n]);
    cursor = 0;
    for i in 0..n {
        wb[i] /= -lf[cursor].sqrt();
        cursor += n - i;
    }
    if let Some(weight) = augment {
        wa[ld * ld - 1] = weight;
        n += 1;
    }

    // Equality constraints E·x = f.
    let lde = meq.max(1);
    let mut we = vec![0.0; lde * n];
    let mut wf = vec![0.0; meq];
    if meq > 0 {
        for j in 0..n - 1 {
            for i in 0..meq {
                we[i + j * meq] = c[i + j * m];
            }
        }
        for i in 0..meq {
            we[i + (n - 1) * meq] = if augment.is_some() {
                -d[i]
            } else {
                c[i + (n - 1) * m]
            };
            wf[i] = -d[i];
        }
    }

    // Inequality constraints G·x ≥ h: linearized constraints, then the finite bounds as ±I.
    let mut wh = Vec::with_capacity(mineq + 2 * n);
    for i in 0..mineq {
        wh.push(-d[meq + i]);
    }
    for &lo in &xl[..n] {
        if !lo.is_nan() {
            wh.push(lo);
        }
    }
    for &hi in &xu[..n] {
        if !hi.is_nan() {
            wh.push(-hi);
        }
    }
    let mg = wh.len();
    let ldg = mg.max(1);
    let mut wg = vec![0.0; ldg * n];
    for j in 0..n_orig {
        for i in 0..mineq {
            wg[i + j * mg] = c[meq + i + j * m];
        }
    }
    if augment.is_some() {
        for i in 0..mineq {
            wg[i + n_orig * mg] = (-d[meq + i]).max(0.0);
        }
    }
    let mut nrow = mineq;
    for (i, &lo) in xl[..n].iter().enumerate() {
        if !lo.is_nan() {
            wg[nrow + i * mg] = 1.0;
            nrow += 1;
        }
    }
    for (i, &hi) in xu[..n].iter().enumerate() {
        if !hi.is_nan() {
            wg[nrow + i * mg] = -1.0;
            nrow += 1;
        }
    }

    let (mode, gmults, emults) = lsei(
        ld, meq, mg, n, &mut wa, &mut wb, &mut we, &mut wf, &mut wg, &mut wh, x,
    );
    if mode == 1 {
        y[..meq].copy_from_slice(&emults);
        y[meq..meq + mineq].copy_from_slice(&gmults[..mineq]);
        for yi in &mut y[m..m + 2 * n] {
            *yi = f64::NAN;
        }
    }
    clamp_to_bounds(&mut x[..n], &xl[..n], &xu[..n]);
    mode
}

/// Least squares with equality and inequality constraints:
/// min |A·x − b| s.t. E·x = f, G·x ≥ h. A is `ma×n` (lda `ma`), E is `me×n` (lde
/// `max(me,1)`), G is `mg×n` (ldg `max(mg,1)`). Returns `(mode, inequality multipliers,
/// equality multipliers)`.
#[allow(clippy::too_many_arguments)]
fn lsei(
    ma: usize,
    me: usize,
    mg: usize,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    e: &mut [f64],
    f: &mut [f64],
    g: &mut [f64],
    h: &mut [f64],
    x: &mut [f64],
) -> (i32, Vec<f64>, Vec<f64>) {
    x[..n].fill(0.0);
    let mut gmults = vec![0.0; mg];
    let mut emults = vec![0.0; me];
    if me > n {
        return (2, gmults, emults);
    }
    let nvars = n - me;
    let lde = me.max(1);
    let ldg = mg.max(1);

    // RQ factorization of E; apply Qᵀ to A and G from the right.
    let tau = gerq2(me, n, e, lde);
    ormr2_right_trans(ma, n, me, e, lde, &tau, a, ma);
    ormr2_right_trans(mg, n, me, e, lde, &tau, g, ldg);

    // A NaN diagonal counts as singular, as the C test `!(|e_ii| >= eps)` does.
    for i in 0..me {
        let diag = e[i + (nvars + i) * lde].abs();
        if diag.is_nan() || diag < EPSMACH {
            return (6, gmults, emults);
        }
    }
    // Solve E·x = f for the trailing me components (R sits at the right of E).
    x[nvars..n].copy_from_slice(&f[..me]);
    trsv_upper(me, &e[nvars * lde..], lde, &mut x[nvars..n]);

    let mut mode = 1;
    let mut xnorm;
    if me < n {
        // wb = b − A1·xe
        let mut wb = b[..ma].to_vec();
        for (jj, &xe) in x[nvars..n].iter().enumerate() {
            let col = &a[(nvars + jj) * ma..(nvars + jj) * ma + ma];
            for (w, &aij) in wb.iter_mut().zip(col) {
                *w -= aij * xe;
            }
        }
        let mut a2 = a[..ma * nvars].to_vec();
        let mut g2 = vec![0.0; mg * nvars];
        for j in 0..nvars {
            g2[j * mg..j * mg + mg].copy_from_slice(&g[j * ldg..j * ldg + mg]);
        }

        if mg == 0 {
            // No inequality constraints: a rank-revealing least-squares solve (dgelsy).
            let wb_orig = wb.clone();
            let (sol, krank) = gelsy(ma, nvars, &mut a2, &mut wb, EPSMACH.sqrt());
            x[..nvars].copy_from_slice(&sol);
            let mut resid = vec![0.0; ma];
            for (i, r) in resid.iter_mut().enumerate() {
                let mut acc = -wb_orig[i];
                for j in 0..nvars {
                    acc += a[i + j * ma] * x[j];
                }
                *r = acc;
            }
            // xnorm is the LSQ residual norm; SLSQP itself does not read it.
            let _ = nrm2(&resid);
            if krank < nvars {
                return (7, gmults, emults);
            }
        } else {
            // h −= G1·xe, then the inequality-constrained problem in the remaining variables.
            for (jj, &xe) in x[nvars..n].iter().enumerate() {
                let col = &g[(nvars + jj) * ldg..(nvars + jj) * ldg + mg];
                for (hi, &gij) in h.iter_mut().zip(col) {
                    *hi -= gij * xe;
                }
            }
            let (lsi_mode, lsi_xnorm, mults) =
                lsi(ma, mg, nvars, &mut a2, &mut wb, &mut g2, h, &mut x[..nvars]);
            mode = lsi_mode;
            xnorm = lsi_xnorm;
            gmults.copy_from_slice(&mults);
            if me == 0 {
                return (mode, gmults, emults);
            }
            let t = nrm2(&x[nvars..n]);
            xnorm = xnorm.hypot(t);
            let _ = xnorm;
            if mode != 1 {
                return (mode, gmults, emults);
            }
        }
    }

    // Back to the original basis: residuals, multipliers, x = Qᵀ·x.
    for i in 0..ma {
        let mut acc = -b[i];
        for j in 0..n {
            acc += a[i + j * ma] * x[j];
        }
        b[i] = acc;
    }
    for i in 0..me {
        let col = &a[(nvars + i) * ma..(nvars + i) * ma + ma];
        let mut acc = dot(col, &b[..ma]);
        if mg > 0 {
            let gcol = &g[(nvars + i) * ldg..(nvars + i) * ldg + mg];
            acc -= dot(gcol, &gmults);
        }
        f[i] = acc;
    }
    ormr2_left_trans(n, me, e, lde, &tau, &mut x[..n]);
    emults.copy_from_slice(&f[..me]);
    trsv_upper_trans(me, &e[(n - me) * lde..], lde, &mut emults);
    (mode, gmults, emults)
}

/// Inequality-constrained least squares: min |A·x − b| s.t. G·x ≥ h, reduced to an LDP.
/// Returns `(mode, residual norm, multipliers)`.
#[allow(clippy::too_many_arguments)]
fn lsi(
    ma: usize,
    mg: usize,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    g: &mut [f64],
    h: &mut [f64],
    x: &mut [f64],
) -> (i32, f64, Vec<f64>) {
    let k = ma.min(n);
    let tau = geqr2(ma, n, a, ma);
    orm2r_left_trans(ma, k, a, ma, &tau, b);
    for i in 0..k {
        let diag = a[i + i * ma].abs();
        if diag.is_nan() || diag < EPSMACH {
            return (5, 0.0, vec![0.0; mg]);
        }
    }
    // G := G·R⁻¹ and h := h − G·b.
    trsm_right_upper(mg, n, a, ma, g, mg);
    for (i, hi) in h.iter_mut().enumerate().take(mg) {
        let mut acc = 0.0;
        for j in 0..n {
            acc += g[i + j * mg] * b[j];
        }
        *hi -= acc;
    }
    let (mode, mut xnorm, mults) = ldp(mg, n, g, h, x);
    if mode != 1 {
        return (mode, xnorm, mults);
    }
    for (xi, &bi) in x.iter_mut().zip(b.iter()).take(n) {
        *xi += bi;
    }
    trsv_upper(n, a, ma, x);
    if ma > n {
        xnorm = xnorm.hypot(nrm2(&b[n..ma]));
    }
    (mode, xnorm, mults)
}

/// Least distance programming: min ½|x|² s.t. G·x ≥ h (G `m×n`, column-major, ld `m`), via
/// the NNLS dual [Gᵀ; hᵀ]·y ≈ [0; 1]. Returns `(mode, |x|, multipliers)`.
fn ldp(m: usize, n: usize, g: &[f64], h: &[f64], x: &mut [f64]) -> (i32, f64, Vec<f64>) {
    if n == 0 {
        return (2, 0.0, vec![0.0; m]);
    }
    x[..n].fill(0.0);
    if m == 0 {
        return (1, 0.0, Vec::new());
    }
    let rows = n + 1;
    let mut a = vec![0.0; rows * m];
    for j in 0..m {
        for i in 0..n {
            a[i + j * rows] = g[j + i * m];
        }
        a[n + j * rows] = h[j];
    }
    let mut b = vec![0.0; rows];
    b[n] = 1.0;
    let (y, rnorm, info) = nnls_lh(rows, m, &mut a, &mut b, 3 * m);
    if info != 1 {
        return (info, 0.0, vec![0.0; m]);
    }
    if rnorm <= 0.0 {
        return (4, 0.0, vec![0.0; m]);
    }
    let fac = 1.0 - dot(&h[..m], &y);
    let rounded = (1.0 + fac) - 1.0;
    if rounded.is_nan() || rounded <= 0.0 {
        return (4, 0.0, vec![0.0; m]);
    }
    let fac = 1.0 / fac;
    for (i, xi) in x.iter_mut().enumerate().take(n) {
        let mut acc = 0.0;
        for j in 0..m {
            acc += g[j + i * m] * y[j];
        }
        *xi = fac * acc;
    }
    let xnorm = nrm2(&x[..n]);
    let mults = y.iter().map(|&yi| fac * yi).collect();
    (1, xnorm, mults)
}

/// Lawson–Hanson NNLS, min |A·x − b| s.t. x ≥ 0, on a column-major `m×n` A (overwritten)
/// and b (overwritten), exactly as SciPy's `__nnls`. Returns `(x, rnorm, info)` with info 1
/// on success, 2 bad dimensions, 3 iteration limit.
pub(crate) fn nnls_lh(
    m: usize,
    n: usize,
    a: &mut [f64],
    b: &mut [f64],
    maxiter: usize,
) -> (Vec<f64>, f64, i32) {
    let mut x = vec![0.0; n];
    if m == 0 || n == 0 {
        return (x, 0.0, 2);
    }
    let mut info = 1;
    let mut w = vec![0.0; n];
    let mut zz = vec![0.0; m];
    let mut indices: Vec<usize> = (0..n).collect();
    let mut indz = 0_usize;
    let mut iteration = 0_usize;
    let mut jj = 0_usize;
    let mut tau;
    let mut pivot;

    'outer: while indz < m.min(n) {
        for &j in &indices[indz..n] {
            w[j] = dot(&a[indz + j * m..(j + 1) * m], &b[indz..m]);
        }

        // Select the next column to move from Z to P.
        let (iz, j) = loop {
            let mut wmax = 0.0;
            let mut izmax = 0;
            for (k, &jk) in indices.iter().enumerate().take(n).skip(indz) {
                if w[jk] > wmax {
                    wmax = w[jk];
                    izmax = k;
                }
            }
            if wmax <= 0.0 {
                break 'outer;
            }
            let iz = izmax;
            let j = indices[iz];
            let len = m - indz;
            let (beta, t) = dlarfgp(len, a[indz + j * m], a, indz + 1 + j * m, 1);
            pivot = beta;
            tau = t;
            let unorm = if indz > 0 {
                nrm2(&a[j * m..j * m + indz])
            } else {
                0.0
            };
            let spacing = if unorm > 0.0 {
                next_up(unorm) - unorm
            } else {
                0.0
            };
            if pivot.abs() > 100.0 * spacing {
                zz.copy_from_slice(&b[..m]);
                let pivot2 = a[indz + j * m];
                a[indz + j * m] = 1.0;
                let v: Vec<f64> = a[indz + j * m..(j + 1) * m].to_vec();
                larf_left_vec(&v, tau, &mut zz[indz..m]);
                let ztest = zz[indz] / pivot;
                if ztest > 0.0 {
                    // Keep the reflector vector in a (a[indz + j*m] is 1 for now).
                    break (iz, j);
                }
                a[indz + j * m] = pivot2;
            }
            w[j] = 0.0;
        };

        b[..m].copy_from_slice(&zz);
        indices[iz] = indices[indz];
        indices[indz] = j;
        indz += 1;
        if indz < n {
            let v: Vec<f64> = a[indz - 1 + j * m..(j + 1) * m].to_vec();
            for k in indz..n {
                let col = indices[k];
                larf_left_vec(&v, tau, &mut a[indz - 1 + col * m..(col + 1) * m]);
            }
        }
        a[indz - 1 + j * m] = pivot;
        for i in indz..m {
            a[j * m + i] = 0.0;
        }
        w[j] = 0.0;

        solve_permuted_triangular(m, a, &indices, indz, &mut zz, &mut jj);

        loop {
            iteration += 1;
            if iteration >= maxiter {
                info = 3;
                break 'outer;
            }
            let mut alpha = 2.0;
            for ip in 0..indz {
                let k = indices[ip];
                if zz[ip] <= 0.0 {
                    let t = -x[k] / (zz[ip] - x[k]);
                    if alpha > t {
                        alpha = t;
                        jj = ip;
                    }
                }
            }
            if alpha == 2.0 {
                break;
            }
            for ip in 0..indz {
                let k = indices[ip];
                x[k] += alpha * (zz[ip] - x[k]);
            }

            // Move coefficients from P back to Z, restoring the triangular form by Givens.
            let mut i = indices[jj];
            loop {
                x[i] = 0.0;
                if jj != indz - 1 {
                    jj += 1;
                    for jcol in jj..indz {
                        let ii = indices[jcol];
                        indices[jcol - 1] = ii;
                        let (cc, ss, r) = dlartgp(a[jcol - 1 + ii * m], a[jcol + ii * m]);
                        a[jcol - 1 + ii * m] = r;
                        a[jcol + ii * m] = 0.0;
                        for k in 0..n {
                            if k != ii {
                                let tmp = a[jcol - 1 + k * m];
                                a[jcol - 1 + k * m] = cc * tmp + ss * a[jcol + k * m];
                                a[jcol + k * m] = -ss * tmp + cc * a[jcol + k * m];
                            }
                        }
                        let tmp = b[jcol - 1];
                        b[jcol - 1] = cc * tmp + ss * b[jcol];
                        b[jcol] = -ss * tmp + cc * b[jcol];
                    }
                }
                indz -= 1;
                indices[indz] = i;

                // Any remaining nonpositive coefficient in P is round-off: move it out too.
                // (An empty P set is feasible, as in the Fortran original.)
                let mut infeasible = None;
                for (pos, &idx) in indices.iter().enumerate().take(indz) {
                    if x[idx] <= 0.0 {
                        infeasible = Some(pos);
                        break;
                    }
                }
                match infeasible {
                    Some(pos) => {
                        jj = pos;
                        i = indices[pos];
                    }
                    None => break,
                }
            }

            zz.copy_from_slice(&b[..m]);
            solve_permuted_triangular(m, a, &indices, indz, &mut zz, &mut jj);
        }

        for k in 0..indz {
            x[indices[k]] = zz[k];
        }
    }

    let rnorm = if indz < m { nrm2(&b[indz..m]) } else { 0.0 };
    (x, rnorm, info)
}

/// Back substitution on the permuted upper-triangular P block of the NNLS matrix, into `zz`.
/// `jj` carries the last column index used, as the C loop leaves it.
fn solve_permuted_triangular(
    m: usize,
    a: &[f64],
    indices: &[usize],
    indz: usize,
    zz: &mut [f64],
    jj: &mut usize,
) {
    for k in 0..indz {
        let ip = indz - 1 - k;
        if k != 0 {
            let z_next = zz[ip + 1];
            for i in 0..=ip {
                zz[i] -= a[i + *jj * m] * z_next;
            }
        }
        *jj = indices[ip];
        zz[ip] /= a[ip + *jj * m];
    }
}

/// Fletcher–Powell composite-t update of the packed `L·D·Lᵀ` factors by `sigma·z·zᵀ`.
fn ldl_update(n: usize, a: &mut [f64], z: &mut [f64], sigma: f64, w: &mut [f64]) {
    if sigma == 0.0 {
        return;
    }
    let mut ij = 0_usize;
    let mut t = 1.0 / sigma;
    if sigma <= 0.0 {
        w[..n].copy_from_slice(&z[..n]);
        for i in 0..n {
            let v = w[i];
            t += v * v / a[ij];
            for j in (i + 1)..n {
                ij += 1;
                w[j] -= v * a[ij];
            }
            ij += 1;
        }
        if t >= 0.0 {
            t = EPSMACH / sigma;
        }
        for i in 0..n {
            let j = n - i - 1;
            ij -= i + 1;
            let u = w[j];
            w[j] = t;
            t -= u * u / a[ij];
        }
    }
    for i in 0..n {
        let v = z[i];
        let delta = v / a[ij];
        let tp = if sigma < 0.0 { w[i] } else { t + delta * v };
        let alpha = tp / t;
        a[ij] *= alpha;
        if i == n - 1 {
            return;
        }
        let beta = delta / tp;
        if alpha <= 4.0 {
            for j in (i + 1)..n {
                ij += 1;
                z[j] -= v * a[ij];
                a[ij] += beta * z[j];
            }
        } else {
            let gamma = t / tp;
            for j in (i + 1)..n {
                ij += 1;
                let u = a[ij];
                a[ij] = gamma * u + beta * z[j];
                z[j] -= v * u;
            }
        }
        ij += 1;
        t = tp;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BLAS/LAPACK kernels (column-major, reference-algorithm order)
// ──────────────────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scaled two-norm (reference BLAS `dnrm2`).
fn nrm2(x: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &xi in x {
        if xi != 0.0 {
            let ax = xi.abs();
            if scale < ax {
                let r = scale / ax;
                ssq = 1.0 + ssq * r * r;
                scale = ax;
            } else {
                let r = ax / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

fn nrm2_strided(buf: &[f64], start: usize, inc: usize, len: usize) -> f64 {
    let v: Vec<f64> = (0..len).map(|k| buf[start + k * inc]).collect();
    nrm2(&v)
}

fn next_up(x: f64) -> f64 {
    // x is positive and finite here (NNLS column norm).
    f64::from_bits(x.to_bits() + 1)
}

/// `dlarfg`: reflector H with H·[alpha; x] = [beta; 0]. The `len − 1` entries of x live in
/// `buf[xstart + k·incx]` and are overwritten with v(2:len). Returns `(beta, tau)`.
fn dlarfg(len: usize, alpha: f64, buf: &mut [f64], xstart: usize, incx: usize) -> (f64, f64) {
    if len <= 1 {
        return (alpha, 0.0);
    }
    let mut xnorm = nrm2_strided(buf, xstart, incx, len - 1);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let mut alpha = alpha;
    let mut beta = -alpha.hypot(xnorm).copysign(alpha);
    let mut knt = 0;
    if beta.abs() < SMLNUM {
        let rsafmn = 1.0 / SMLNUM;
        loop {
            knt += 1;
            for k in 0..len - 1 {
                buf[xstart + k * incx] *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < SMLNUM && knt < 20) {
                break;
            }
        }
        xnorm = nrm2_strided(buf, xstart, incx, len - 1);
        beta = -alpha.hypot(xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let scal = 1.0 / (alpha - beta);
    for k in 0..len - 1 {
        buf[xstart + k * incx] *= scal;
    }
    for _ in 0..knt {
        beta *= SMLNUM;
    }
    (beta, tau)
}

/// `dlarfgp`: as `dlarfg` but with beta ≥ 0. Returns `(beta, tau)`.
fn dlarfgp(len: usize, alpha: f64, buf: &mut [f64], xstart: usize, incx: usize) -> (f64, f64) {
    if len == 0 {
        return (alpha, 0.0);
    }
    let mut xnorm = nrm2_strided(buf, xstart, incx, len - 1);
    if xnorm <= EPSMACH * alpha.abs() {
        if alpha >= 0.0 {
            return (alpha, 0.0);
        }
        for k in 0..len - 1 {
            buf[xstart + k * incx] = 0.0;
        }
        return (-alpha, 2.0);
    }
    let mut alpha = alpha;
    let mut beta = alpha.hypot(xnorm).copysign(alpha);
    let mut knt = 0;
    if beta.abs() < SMLNUM {
        let bignum = 1.0 / SMLNUM;
        loop {
            knt += 1;
            for k in 0..len - 1 {
                buf[xstart + k * incx] *= bignum;
            }
            beta *= bignum;
            alpha *= bignum;
            if !(beta.abs() < SMLNUM && knt < 20) {
                break;
            }
        }
        xnorm = nrm2_strided(buf, xstart, incx, len - 1);
        beta = alpha.hypot(xnorm).copysign(alpha);
    }
    let savealpha = alpha;
    alpha += beta;
    let mut tau;
    if beta < 0.0 {
        beta = -beta;
        tau = -alpha / beta;
    } else {
        alpha = xnorm * (xnorm / alpha);
        tau = alpha / beta;
        alpha = -alpha;
    }
    if tau.abs() <= SMLNUM {
        if savealpha >= 0.0 {
            tau = 0.0;
        } else {
            tau = 2.0;
            for k in 0..len - 1 {
                buf[xstart + k * incx] = 0.0;
            }
            beta = -savealpha;
        }
    } else {
        let scal = 1.0 / alpha;
        for k in 0..len - 1 {
            buf[xstart + k * incx] *= scal;
        }
    }
    for _ in 0..knt {
        beta *= SMLNUM;
    }
    (beta, tau)
}

/// `dlartgp`: plane rotation with r ≥ 0. Returns `(cs, sn, r)`.
fn dlartgp(f: f64, g: f64) -> (f64, f64, f64) {
    if g == 0.0 {
        return (1.0_f64.copysign(f), 0.0, f.abs());
    }
    if f == 0.0 {
        return (0.0, 1.0_f64.copysign(g), g.abs());
    }
    let r = f.hypot(g);
    (f / r, g / r, r)
}

/// Apply H = I − tau·v·vᵀ from the left to the vector `c` (len(v) == len(c)).
fn larf_left_vec(v: &[f64], tau: f64, c: &mut [f64]) {
    if tau == 0.0 {
        return;
    }
    let w = dot(v, c);
    for (ci, &vi) in c.iter_mut().zip(v) {
        *ci -= tau * vi * w;
    }
}

/// Unblocked Householder QR (`dgeqr2`) of the `m×n` matrix `a` (leading dimension `lda`).
fn geqr2(m: usize, n: usize, a: &mut [f64], lda: usize) -> Vec<f64> {
    let k = m.min(n);
    let mut tau = vec![0.0; k];
    for i in 0..k {
        let xstart = (i + 1).min(m - 1) + i * lda;
        let (beta, t) = dlarfg(m - i, a[i + i * lda], a, xstart, 1);
        a[i + i * lda] = beta;
        tau[i] = t;
        if i + 1 < n {
            let aii = a[i + i * lda];
            a[i + i * lda] = 1.0;
            let v: Vec<f64> = a[i + i * lda..i * lda + m].to_vec();
            for col in (i + 1)..n {
                larf_left_vec(&v, t, &mut a[i + col * lda..col * lda + m]);
            }
            a[i + i * lda] = aii;
        }
    }
    tau
}

/// `dorm2r('L', 'T')` for one right-hand side: c := Qᵀ·c with Q from `geqr2`.
fn orm2r_left_trans(m: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64], c: &mut [f64]) {
    for i in 0..k {
        let aii = a[i + i * lda];
        a[i + i * lda] = 1.0;
        let v: Vec<f64> = a[i + i * lda..i * lda + m].to_vec();
        larf_left_vec(&v, tau[i], &mut c[i..m]);
        a[i + i * lda] = aii;
    }
}

/// Unblocked RQ factorization (`dgerq2`) of the `m×n` matrix `a`, m ≤ n. R ends up in the
/// last m columns; the reflector vectors are stored in the rows to its left.
fn gerq2(m: usize, n: usize, a: &mut [f64], lda: usize) -> Vec<f64> {
    let k = m.min(n);
    let mut tau = vec![0.0; k];
    for i in (1..=k).rev() {
        let row = m - k + i - 1;
        let col = n - k + i - 1;
        let (beta, t) = dlarfg(n - k + i, a[row + col * lda], a, row, lda);
        a[row + col * lda] = beta;
        tau[i - 1] = t;
        // Apply H(i) to A(0..row, 0..=col) from the right.
        let aii = a[row + col * lda];
        a[row + col * lda] = 1.0;
        let v: Vec<f64> = (0..=col).map(|jc| a[row + jc * lda]).collect();
        larf_right(row, col + 1, &v, t, a, lda);
        a[row + col * lda] = aii;
    }
    tau
}

/// Apply H = I − tau·v·vᵀ from the right to the `m×len(v)` block at the top-left of `c`.
fn larf_right(m: usize, ncols: usize, v: &[f64], tau: f64, c: &mut [f64], ldc: usize) {
    if tau == 0.0 || m == 0 {
        return;
    }
    let mut w = vec![0.0; m];
    for (j, &vj) in v.iter().enumerate().take(ncols) {
        for (i, wi) in w.iter_mut().enumerate() {
            *wi += c[i + j * ldc] * vj;
        }
    }
    for (j, &vj) in v.iter().enumerate().take(ncols) {
        let scale = tau * vj;
        for (i, &wi) in w.iter().enumerate() {
            c[i + j * ldc] -= wi * scale;
        }
    }
}

/// `dormr2('R', 'T')`: C := C·Qᵀ for the `m×n` matrix C, Q = H(1)⋯H(k) from `gerq2` on the
/// `k×n` matrix `a`.
#[allow(clippy::too_many_arguments)]
fn ormr2_right_trans(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 {
        return;
    }
    for i in (1..=k).rev() {
        let ni = n - k + i;
        let pos = (i - 1) + (ni - 1) * lda;
        let aii = a[pos];
        a[pos] = 1.0;
        let v: Vec<f64> = (0..ni).map(|jc| a[(i - 1) + jc * lda]).collect();
        larf_right(m, ni, &v, tau[i - 1], c, ldc);
        a[pos] = aii;
    }
}

/// `dormr2('L', 'T')` for one right-hand side: x := Qᵀ·x (len n) with Q from `gerq2`.
fn ormr2_left_trans(n: usize, k: usize, a: &mut [f64], lda: usize, tau: &[f64], x: &mut [f64]) {
    for i in 1..=k {
        let mi = n - k + i;
        let pos = (i - 1) + (mi - 1) * lda;
        let aii = a[pos];
        a[pos] = 1.0;
        let v: Vec<f64> = (0..mi).map(|jc| a[(i - 1) + jc * lda]).collect();
        larf_left_vec(&v, tau[i - 1], &mut x[..mi]);
        a[pos] = aii;
    }
}

/// Solve U·x = b in place, U upper triangular `n×n` at the start of `a` (ld `lda`).
fn trsv_upper(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    for j in (0..n).rev() {
        if x[j] != 0.0 {
            x[j] /= a[j + j * lda];
            let temp = x[j];
            for i in (0..j).rev() {
                x[i] -= temp * a[i + j * lda];
            }
        }
    }
}

/// Solve Uᵀ·x = b in place.
fn trsv_upper_trans(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    for j in 0..n {
        let mut temp = x[j];
        for i in 0..j {
            temp -= a[i + j * lda] * x[i];
        }
        x[j] = temp / a[j + j * lda];
    }
}

/// `dtrsm('R', 'U', 'N', 'N')`: B := B·U⁻¹ for the `m×n` matrix B.
fn trsm_right_upper(m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    for j in 0..n {
        for k in 0..j {
            let akj = a[k + j * lda];
            if akj != 0.0 {
                for i in 0..m {
                    b[i + j * ldb] -= akj * b[i + k * ldb];
                }
            }
        }
        let temp = 1.0 / a[j + j * lda];
        for i in 0..m {
            b[i + j * ldb] *= temp;
        }
    }
}

/// Index of the diagonal entry (j, j) in packed lower-triangular column-major storage;
/// entry (i, j), i ≥ j, sits at `packed_col_start(n, j) + (i − j)`.
fn packed_col_start(n: usize, j: usize) -> usize {
    j * n - j * j.saturating_sub(1) / 2
}

/// x := L·x, L unit lower triangular in packed storage (stored diagonal ignored).
fn tpmv_lower_unit(n: usize, ap: &[f64], x: &mut [f64]) {
    for j in (0..n).rev() {
        if x[j] != 0.0 {
            let temp = x[j];
            let kk = packed_col_start(n, j);
            for i in ((j + 1)..n).rev() {
                x[i] += temp * ap[kk + (i - j)];
            }
        }
    }
}

/// x := Lᵀ·x, L unit lower triangular in packed storage.
fn tpmv_lower_unit_trans(n: usize, ap: &[f64], x: &mut [f64]) {
    for j in 0..n {
        let kk = packed_col_start(n, j);
        let mut temp = x[j];
        for i in (j + 1)..n {
            temp += ap[kk + (i - j)] * x[i];
        }
        x[j] = temp;
    }
}

/// Solve L·x = b in place, L unit lower triangular in packed storage.
fn tpsv_lower_unit(n: usize, ap: &[f64], x: &mut [f64]) {
    for j in 0..n {
        if x[j] != 0.0 {
            let temp = x[j];
            let kk = packed_col_start(n, j);
            for i in (j + 1)..n {
                x[i] -= temp * ap[kk + (i - j)];
            }
        }
    }
}

/// Rank-revealing least squares (`dgelsy` role): Householder QR with column pivoting,
/// rank = number of leading pivots with |R_kk| > rcond·|R_00|. Returns `(x, rank)`; x is the
/// basic solution on the leading `rank` pivoted columns.
fn gelsy(m: usize, n: usize, a: &mut [f64], b: &mut [f64], rcond: f64) -> (Vec<f64>, usize) {
    let k = m.min(n);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut norms: Vec<f64> = (0..n).map(|j| nrm2(&a[j * m..j * m + m])).collect();
    let mut tau = vec![0.0; k];
    for i in 0..k {
        // Pivot the remaining column of largest norm into position i.
        let mut best = i;
        for j in (i + 1)..n {
            if norms[j] > norms[best] {
                best = j;
            }
        }
        if best != i {
            for r in 0..m {
                a.swap(r + i * m, r + best * m);
            }
            perm.swap(i, best);
            norms.swap(i, best);
        }
        let xstart = (i + 1).min(m - 1) + i * m;
        let (beta, t) = dlarfg(m - i, a[i + i * m], a, xstart, 1);
        a[i + i * m] = beta;
        tau[i] = t;
        let aii = a[i + i * m];
        a[i + i * m] = 1.0;
        let v: Vec<f64> = a[i + i * m..i * m + m].to_vec();
        for col in (i + 1)..n {
            larf_left_vec(&v, t, &mut a[i + col * m..col * m + m]);
        }
        larf_left_vec(&v, t, &mut b[i..m]);
        a[i + i * m] = aii;
        for (col, norm) in norms.iter_mut().enumerate().skip(i + 1) {
            *norm = nrm2(&a[i + 1 + col * m..col * m + m]);
        }
    }
    let r00 = if k > 0 { a[0].abs() } else { 0.0 };
    let mut rank = 0;
    while rank < k && a[rank + rank * m].abs() > rcond * r00 {
        rank += 1;
    }
    let mut y = b[..rank].to_vec();
    trsv_upper(rank, a, m, &mut y);
    let mut x = vec![0.0; n];
    for (pos, &yi) in y.iter().enumerate() {
        x[perm[pos]] = yi;
    }
    (x, rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    // min |A·x − b|, x ≥ 0 with A = [[1,0],[0,1],[1,1]], b = [2,−1,1]. Unconstrained least
    // squares gives (2, −1); the active bound y = 0 leaves min (x−2)² + (x−1)², so x = (1.5, 0)
    // with residual √1.5.
    #[test]
    fn nnls_moves_the_negative_coefficient_to_its_bound() {
        let mut a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]; // column-major 3×2
        let mut b = vec![2.0, -1.0, 1.0];
        let (x, rnorm, info) = nnls_lh(3, 2, &mut a, &mut b, 6);
        assert_eq!(info, 1);
        assert!((x[0] - 1.5).abs() < 1e-14 && x[1] == 0.0, "x = {x:?}");
        assert!((rnorm - 1.5_f64.sqrt()).abs() < 1e-14, "rnorm = {rnorm}");
    }

    // The packed L·D·Lᵀ factors after `ldl_update(σ, z)` must reproduce I + σ·z·zᵀ, and the
    // negative update must take them back to I.
    #[test]
    fn ldl_update_is_the_rank_one_modification() {
        let n = 3;
        let mut packed = vec![0.0; n * (n + 1) / 2];
        for j in 0..n {
            packed[packed_col_start(n, j)] = 1.0;
        }
        let z = [1.0, 2.0, 3.0];
        let dense = |ap: &[f64]| {
            let mut l = [[0.0; 3]; 3];
            let mut d = [0.0; 3];
            for j in 0..n {
                d[j] = ap[packed_col_start(n, j)];
                l[j][j] = 1.0;
                for i in (j + 1)..n {
                    l[i][j] = ap[packed_col_start(n, j) + (i - j)];
                }
            }
            let mut m = [[0.0; 3]; 3];
            for i in 0..n {
                for k in 0..n {
                    m[i][k] = (0..n).map(|j| l[i][j] * d[j] * l[k][j]).sum();
                }
            }
            m
        };
        let mut zz = z;
        let mut w = [0.0; 3];
        ldl_update(n, &mut packed, &mut zz, 2.0, &mut w);
        let m = dense(&packed);
        for i in 0..n {
            for k in 0..n {
                let want = f64::from(u8::from(i == k)) + 2.0 * z[i] * z[k];
                assert!(
                    (m[i][k] - want).abs() < 1e-12,
                    "({i},{k}): {} vs {want}",
                    m[i][k]
                );
            }
        }
        let mut zz = z;
        ldl_update(n, &mut packed, &mut zz, -2.0, &mut w);
        let m = dense(&packed);
        for i in 0..n {
            for k in 0..n {
                let want = f64::from(u8::from(i == k));
                assert!(
                    (m[i][k] - want).abs() < 1e-12,
                    "({i},{k}): {} vs {want}",
                    m[i][k]
                );
            }
        }
    }

    // LDP: min ½|x|² s.t. x₀ + x₁ ≥ 2 is x = (1, 1); adding x₀ ≤ 0 (−x₀ ≥ 0) moves it to
    // (0, 2); adding x₀ ≥ 1 as well makes the constraints incompatible (mode 4).
    #[test]
    fn ldp_solves_the_least_distance_problem_and_detects_incompatibility() {
        let mut x = [0.0; 2];
        let (mode, _, _) = ldp(1, 2, &[1.0, 1.0], &[2.0], &mut x);
        assert_eq!(mode, 1);
        assert!(
            (x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14,
            "{x:?}"
        );

        // G rows: [1,1], [−1,0] (column-major 2×2).
        let (mode, _, _) = ldp(2, 2, &[1.0, -1.0, 1.0, 0.0], &[2.0, 0.0], &mut x);
        assert_eq!(mode, 1);
        assert!(x[0].abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14, "{x:?}");

        // G rows: [1,0] ≥ 1 and [−1,0] ≥ 0 cannot both hold.
        let (mode, _, _) = ldp(2, 2, &[1.0, -1.0, 0.0, 0.0], &[1.0, 0.0], &mut x);
        assert_eq!(mode, 4);
    }
}
