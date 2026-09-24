//! SciPy's line searches, BFGS and CG, transcribed from `scipy/optimize/_dcsrch.py` (MINPACK-2
//! `dcsrch` / `dcstep`), `_linesearch.py` (`line_search_wolfe1`, `scalar_search_wolfe2` with
//! its `extra_condition`, `_zoom`, `_cubicmin`, `_quadmin`) and `_optimize.py`
//! (`_line_search_wolfe12`, `_minimize_bfgs`, `_minimize_cg`) of SciPy 1.17.1, including their
//! Python `min`/`max`/NaN semantics. std-only: the objective is reached through
//! [`LineObjective`].

/// Python's `max(a, b)`: keeps `a` unless `b > a` (so a NaN `b` never wins).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: keeps `a` unless `b < a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

fn py_max3(a: f64, b: f64, c: f64) -> f64 {
    py_max(py_max(a, b), c)
}

/// `np.sign`: 0 at 0, NaN at NaN.
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// `np.clip(x, lo, hi)`, which propagates NaN.
fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    Start,
    Fg,
    Convergence,
    Warning,
    Error,
}

/// MINPACK-2 `dcsrch` state (SciPy `DCSRCH`).
struct Dcsrch {
    ftol: f64,
    gtol: f64,
    xtol: f64,
    stpmin: f64,
    stpmax: f64,
    brackt: bool,
    stage: u8,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl Dcsrch {
    fn new(ftol: f64, gtol: f64, xtol: f64, stpmin: f64, stpmax: f64) -> Self {
        Self {
            ftol,
            gtol,
            xtol,
            stpmin,
            stpmax,
            brackt: false,
            stage: 1,
            ginit: 0.0,
            gtest: 0.0,
            gx: 0.0,
            gy: 0.0,
            finit: 0.0,
            fx: 0.0,
            fy: 0.0,
            stx: 0.0,
            sty: 0.0,
            stmin: 0.0,
            stmax: 0.0,
            width: 0.0,
            width1: 0.0,
        }
    }

    fn iterate(&mut self, mut stp: f64, f: f64, g: f64, task: Task) -> (f64, Task) {
        const P5: f64 = 0.5;
        const P66: f64 = 0.66;
        const XTRAPL: f64 = 1.1;
        const XTRAPU: f64 = 4.0;

        if task == Task::Start {
            let error = stp < self.stpmin
                || stp > self.stpmax
                || g >= 0.0
                || self.ftol < 0.0
                || self.gtol < 0.0
                || self.xtol < 0.0
                || self.stpmin < 0.0
                || self.stpmax < self.stpmin;
            if error {
                return (stp, Task::Error);
            }
            self.brackt = false;
            self.stage = 1;
            self.finit = f;
            self.ginit = g;
            self.gtest = self.ftol * self.ginit;
            self.width = self.stpmax - self.stpmin;
            self.width1 = self.width / P5;
            self.stx = 0.0;
            self.fx = self.finit;
            self.gx = self.ginit;
            self.sty = 0.0;
            self.fy = self.finit;
            self.gy = self.ginit;
            self.stmin = 0.0;
            self.stmax = stp + XTRAPU * stp;
            return (stp, Task::Fg);
        }

        let ftest = self.finit + stp * self.gtest;
        if self.stage == 1 && f <= ftest && g >= 0.0 {
            self.stage = 2;
        }
        let mut out = Task::Fg;
        if self.brackt && (stp <= self.stmin || stp >= self.stmax) {
            out = Task::Warning;
        }
        if self.brackt && self.stmax - self.stmin <= self.xtol * self.stmax {
            out = Task::Warning;
        }
        if stp == self.stpmax && f <= ftest && g <= self.gtest {
            out = Task::Warning;
        }
        if stp == self.stpmin && (f > ftest || g >= self.gtest) {
            out = Task::Warning;
        }
        if f <= ftest && g.abs() <= self.gtol * -self.ginit {
            out = Task::Convergence;
        }
        if out != Task::Fg {
            return (stp, out);
        }

        if self.stage == 1 && f <= self.fx && f > ftest {
            let fm = f - stp * self.gtest;
            let fxm = self.fx - self.stx * self.gtest;
            let fym = self.fy - self.sty * self.gtest;
            let gm = g - self.gtest;
            let gxm = self.gx - self.gtest;
            let gym = self.gy - self.gtest;
            let s = dcstep(
                self.stx,
                fxm,
                gxm,
                self.sty,
                fym,
                gym,
                stp,
                fm,
                gm,
                self.brackt,
                self.stmin,
                self.stmax,
            );
            self.stx = s.stx;
            self.sty = s.sty;
            stp = s.stp;
            self.brackt = s.brackt;
            self.fx = s.fx + self.stx * self.gtest;
            self.fy = s.fy + self.sty * self.gtest;
            self.gx = s.dx + self.gtest;
            self.gy = s.dy + self.gtest;
        } else {
            let s = dcstep(
                self.stx,
                self.fx,
                self.gx,
                self.sty,
                self.fy,
                self.gy,
                stp,
                f,
                g,
                self.brackt,
                self.stmin,
                self.stmax,
            );
            self.stx = s.stx;
            self.fx = s.fx;
            self.gx = s.dx;
            self.sty = s.sty;
            self.fy = s.fy;
            self.gy = s.dy;
            stp = s.stp;
            self.brackt = s.brackt;
        }

        if self.brackt {
            if (self.sty - self.stx).abs() >= P66 * self.width1 {
                stp = self.stx + P5 * (self.sty - self.stx);
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        if self.brackt {
            self.stmin = py_min(self.stx, self.sty);
            self.stmax = py_max(self.stx, self.sty);
        } else {
            self.stmin = stp + XTRAPL * (stp - self.stx);
            self.stmax = stp + XTRAPU * (stp - self.stx);
        }
        stp = np_clip(stp, self.stpmin, self.stpmax);
        if self.brackt && (stp <= self.stmin || stp >= self.stmax)
            || (self.brackt && self.stmax - self.stmin <= self.xtol * self.stmax)
        {
            stp = self.stx;
        }
        (stp, Task::Fg)
    }
}

struct Step {
    stx: f64,
    fx: f64,
    dx: f64,
    sty: f64,
    fy: f64,
    dy: f64,
    stp: f64,
    brackt: bool,
}

/// MINPACK-2 `dcstep` (SciPy `dcstep`): the safeguarded cubic/quadratic step.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    mut stx: f64,
    mut fx: f64,
    mut dx: f64,
    mut sty: f64,
    mut fy: f64,
    mut dy: f64,
    stp: f64,
    fp: f64,
    dp: f64,
    mut brackt: bool,
    stpmin: f64,
    stpmax: f64,
) -> Step {
    let sgnd = np_sign(dp) * np_sign(dx);
    let stpf;
    if fp > fx {
        let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        let s = py_max3(theta.abs(), dx.abs(), dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
        if stp < stx {
            gamma = -gamma;
        }
        let p = (gamma - dx) + theta;
        let q = ((gamma - dx) + gamma) + dp;
        let r = p / q;
        let stpc = stx + r * (stp - stx);
        let stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
        stpf = if (stpc - stx).abs() <= (stpq - stx).abs() {
            stpc
        } else {
            stpc + (stpq - stpc) / 2.0
        };
        brackt = true;
    } else if sgnd < 0.0 {
        let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        let s = py_max3(theta.abs(), dx.abs(), dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
        if stp > stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + dx;
        let r = p / q;
        let stpc = stp + r * (stx - stp);
        let stpq = stp + (dp / (dp - dx)) * (stx - stp);
        stpf = if (stpc - stp).abs() > (stpq - stp).abs() {
            stpc
        } else {
            stpq
        };
        brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        let s = py_max3(theta.abs(), dx.abs(), dp.abs());
        let mut gamma = s * py_max(0.0, (theta / s).powi(2) - (dx / s) * (dp / s)).sqrt();
        if stp > stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = (gamma + (dx - dp)) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 {
            stp + r * (stx - stp)
        } else if stp > stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if brackt {
            let mut f = if (stpc - stp).abs() < (stpq - stp).abs() {
                stpc
            } else {
                stpq
            };
            f = if stp > stx {
                py_min(stp + 0.66 * (sty - stp), f)
            } else {
                py_max(stp + 0.66 * (sty - stp), f)
            };
            stpf = f;
        } else {
            let f = if (stpc - stp).abs() > (stpq - stp).abs() {
                stpc
            } else {
                stpq
            };
            stpf = np_clip(f, stpmin, stpmax);
        }
    } else if brackt {
        let theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
        let s = py_max3(theta.abs(), dy.abs(), dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (dy / s) * (dp / s)).sqrt();
        if stp > sty {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + dy;
        let r = p / q;
        stpf = stp + r * (sty - stp);
    } else if stp > stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }

    if fp > fx {
        sty = stp;
        fy = fp;
        dy = dp;
    } else {
        if sgnd < 0.0 {
            sty = stx;
            fy = fx;
            dy = dx;
        }
        stx = stp;
        fx = fp;
        dx = dp;
    }
    Step {
        stx,
        fx,
        dx,
        sty,
        fy,
        dy,
        stp: stpf,
        brackt,
    }
}

/// The objective as the line searches and BFGS see it. Evaluation counting and caching are the
/// implementor's business (SciPy's `ScalarFunction` caches the last point).
pub(crate) trait LineObjective {
    type Error;
    fn fun(&mut self, x: &[f64]) -> Result<f64, Self::Error>;
    fn grad(&mut self, x: &[f64]) -> Result<Vec<f64>, Self::Error>;
    /// Called by [`minimize_bfgs`] after each iteration with the new iterate; `true` stops it
    /// (SciPy's `_call_callback_maybe_halt`).
    fn callback(&mut self, _x: &[f64], _fun: f64) -> bool {
        false
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn point(xk: &[f64], s: f64, pk: &[f64]) -> Vec<f64> {
    xk.iter().zip(pk).map(|(x, p)| x + s * p).collect()
}

pub(crate) struct LineSearch {
    pub alpha: f64,
    pub fval: f64,
    /// `old_fval` as SciPy hands it back (φ(0)).
    pub old_fval: f64,
    /// Gradient at the accepted point, if the search evaluated it.
    pub grad: Option<Vec<f64>>,
}

/// SciPy `line_search_wolfe1` → `scalar_search_wolfe1` → `DCSRCH`. `Ok(None)` is SciPy's
/// `stp = None` (the search failed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn line_search_wolfe1<O: LineObjective>(
    obj: &mut O,
    xk: &[f64],
    pk: &[f64],
    gfk: &[f64],
    old_fval: f64,
    old_old_fval: Option<f64>,
    c1: f64,
    c2: f64,
    amax: f64,
    amin: f64,
    xtol: f64,
) -> Result<Option<LineSearch>, O::Error> {
    let derphi0 = dot(gfk, pk);
    let phi0 = old_fval;
    let mut alpha1 = match old_old_fval {
        Some(old_phi0) if derphi0 != 0.0 => {
            let a = py_min(1.0, 1.01 * 2.0 * (phi0 - old_phi0) / derphi0);
            if a < 0.0 { 1.0 } else { a }
        }
        _ => 1.0,
    };
    let mut search = Dcsrch::new(c1, c2, xtol, amin, amax);
    let mut phi1 = phi0;
    let mut derphi1 = derphi0;
    let mut gval = gfk.to_vec();
    let mut task = Task::Start;
    let mut converged = None;
    for _ in 0..100 {
        let (stp, next) = search.iterate(alpha1, phi1, derphi1, task);
        task = next;
        if !stp.is_finite() {
            break;
        }
        if task == Task::Fg {
            alpha1 = stp;
            let x = point(xk, stp, pk);
            phi1 = obj.fun(&x)?;
            gval = obj.grad(&x)?;
            derphi1 = dot(&gval, pk);
        } else {
            if task == Task::Convergence {
                converged = Some(stp);
            }
            break;
        }
    }
    Ok(converged.map(|alpha| LineSearch {
        alpha,
        fval: phi1,
        old_fval: phi0,
        grad: Some(gval),
    }))
}

/// SciPy `_cubicmin`: minimizer of the cubic through (a, fa, fpa), (b, fb), (c, fc), or None on
/// any floating-point exception or a non-finite result.
fn cubicmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64, c: f64, fc: f64) -> Option<f64> {
    let cc = fpa;
    let db = b - a;
    let dc = c - a;
    let denom = (db * dc).powi(2) * (db - dc);
    let r0 = fb - fa - cc * db;
    let r1 = fc - fa - cc * dc;
    let mut aa = dc * dc * r0 + (-(db * db)) * r1;
    let mut bb = (-(dc * dc * dc)) * r0 + db * db * db * r1;
    if denom == 0.0 || !denom.is_finite() || !aa.is_finite() || !bb.is_finite() {
        return None;
    }
    aa /= denom;
    bb /= denom;
    let radical = bb * bb - 3.0 * aa * cc;
    if radical < 0.0 || aa == 0.0 || !radical.is_finite() {
        return None;
    }
    let xmin = a + (-bb + radical.sqrt()) / (3.0 * aa);
    xmin.is_finite().then_some(xmin)
}

/// SciPy `_quadmin`.
fn quadmin(a: f64, fa: f64, fpa: f64, b: f64, fb: f64) -> Option<f64> {
    let d = fa;
    let c = fpa;
    let db = b - a;
    if db == 0.0 {
        return None;
    }
    let bb = (fb - d - c * db) / (db * db);
    if bb == 0.0 || !bb.is_finite() {
        return None;
    }
    let xmin = a - c / (2.0 * bb);
    xmin.is_finite().then_some(xmin)
}

/// SciPy's `extra_condition(alpha, x, phi, gval)`, a further test an accepted step must pass
/// (CG's sufficient-descent check).
pub(crate) type ExtraCondition<'e> = &'e mut dyn FnMut(f64, &[f64], f64, &[f64]) -> bool;

struct Phi<'a, O: LineObjective> {
    obj: &'a mut O,
    xk: &'a [f64],
    pk: &'a [f64],
    gval: Option<Vec<f64>>,
    /// The step `gval` was evaluated at (SciPy's `gval_alpha`).
    gval_alpha: Option<f64>,
}

impl<O: LineObjective> Phi<'_, O> {
    fn phi(&mut self, alpha: f64) -> Result<f64, O::Error> {
        let x = point(self.xk, alpha, self.pk);
        self.obj.fun(&x)
    }
    fn derphi(&mut self, alpha: f64) -> Result<f64, O::Error> {
        let x = point(self.xk, alpha, self.pk);
        let g = self.obj.grad(&x)?;
        let d = dot(&g, self.pk);
        self.gval = Some(g);
        self.gval_alpha = Some(alpha);
        Ok(d)
    }
    /// SciPy's `extra_condition2`: the caller's condition with the gradient at `alpha`
    /// (evaluated only if the last one was taken elsewhere); true when there is none.
    fn extra(
        &mut self,
        alpha: f64,
        phi: f64,
        extra: &mut Option<ExtraCondition<'_>>,
    ) -> Result<bool, O::Error> {
        let Some(condition) = extra.as_deref_mut() else {
            return Ok(true);
        };
        if self.gval_alpha != Some(alpha) {
            self.derphi(alpha)?;
        }
        let x = point(self.xk, alpha, self.pk);
        Ok(condition(
            alpha,
            &x,
            phi,
            self.gval.as_deref().unwrap_or_default(),
        ))
    }
}

/// SciPy `_zoom`.
#[allow(clippy::too_many_arguments)]
fn zoom<O: LineObjective>(
    f: &mut Phi<'_, O>,
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut phi_hi: f64,
    mut derphi_lo: f64,
    phi0: f64,
    derphi0: f64,
    c1: f64,
    c2: f64,
    extra: &mut Option<ExtraCondition<'_>>,
) -> Result<Option<(f64, f64)>, O::Error> {
    const MAXITER: usize = 10;
    const DELTA1: f64 = 0.2;
    const DELTA2: f64 = 0.1;
    let mut i = 0;
    let mut phi_rec = phi0;
    let mut a_rec = 0.0;
    loop {
        let dalpha = a_hi - a_lo;
        let (a, b) = if dalpha < 0.0 {
            (a_hi, a_lo)
        } else {
            (a_lo, a_hi)
        };
        let mut a_j = None;
        let mut cchk = 0.0;
        if i > 0 {
            cchk = DELTA1 * dalpha;
            a_j = cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec);
        }
        let a_j = match a_j {
            Some(v) if i != 0 && !(v > b - cchk) && !(v < a + cchk) => v,
            _ => {
                let qchk = DELTA2 * dalpha;
                match quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi) {
                    Some(v) if !(v > b - qchk) && !(v < a + qchk) => v,
                    _ => a_lo + 0.5 * dalpha,
                }
            }
        };
        let phi_aj = f.phi(a_j)?;
        if phi_aj > phi0 + c1 * a_j * derphi0 || phi_aj >= phi_lo {
            phi_rec = phi_hi;
            a_rec = a_hi;
            a_hi = a_j;
            phi_hi = phi_aj;
        } else {
            let derphi_aj = f.derphi(a_j)?;
            if derphi_aj.abs() <= -c2 * derphi0 && f.extra(a_j, phi_aj, extra)? {
                return Ok(Some((a_j, phi_aj)));
            }
            if derphi_aj * (a_hi - a_lo) >= 0.0 {
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_lo;
                phi_hi = phi_lo;
            } else {
                phi_rec = phi_lo;
                a_rec = a_lo;
            }
            a_lo = a_j;
            phi_lo = phi_aj;
            derphi_lo = derphi_aj;
        }
        i += 1;
        if i > MAXITER {
            return Ok(None);
        }
    }
}

/// SciPy `line_search_wolfe2` → `scalar_search_wolfe2`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn line_search_wolfe2<O: LineObjective>(
    obj: &mut O,
    xk: &[f64],
    pk: &[f64],
    gfk: &[f64],
    old_fval: f64,
    old_old_fval: Option<f64>,
    c1: f64,
    c2: f64,
    amax: Option<f64>,
    maxiter: usize,
    mut extra: Option<ExtraCondition<'_>>,
) -> Result<Option<LineSearch>, O::Error> {
    let derphi0 = dot(gfk, pk);
    let phi0 = old_fval;
    let mut f = Phi {
        obj,
        xk,
        pk,
        gval: None,
        gval_alpha: None,
    };
    let mut alpha0 = 0.0;
    let mut alpha1 = match old_old_fval {
        Some(old_phi0) if derphi0 != 0.0 => py_min(1.0, 1.01 * 2.0 * (phi0 - old_phi0) / derphi0),
        _ => 1.0,
    };
    if alpha1 < 0.0 {
        alpha1 = 1.0;
    }
    if let Some(amax) = amax {
        alpha1 = py_min(alpha1, amax);
    }
    let mut phi_a1 = f.phi(alpha1)?;
    let mut phi_a0 = phi0;
    let mut derphi_a0 = derphi0;
    for i in 0..maxiter {
        if alpha1 == 0.0 || amax.is_some_and(|m| alpha0 > m) {
            return Ok(None);
        }
        let found = if phi_a1 > phi0 + c1 * alpha1 * derphi0 || (phi_a1 >= phi_a0 && i > 0) {
            zoom(
                &mut f, alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi0, derphi0, c1, c2,
                &mut extra,
            )?
        } else {
            let derphi_a1 = f.derphi(alpha1)?;
            // A strong-Wolfe step the extra condition rejects falls through, as in SciPy.
            if derphi_a1.abs() <= -c2 * derphi0 && f.extra(alpha1, phi_a1, &mut extra)? {
                Some((alpha1, phi_a1))
            } else if derphi_a1 >= 0.0 {
                zoom(
                    &mut f, alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi0, derphi0, c1, c2,
                    &mut extra,
                )?
            } else {
                let mut alpha2 = 2.0 * alpha1;
                if let Some(amax) = amax {
                    alpha2 = py_min(alpha2, amax);
                }
                alpha0 = alpha1;
                alpha1 = alpha2;
                phi_a0 = phi_a1;
                phi_a1 = f.phi(alpha1)?;
                derphi_a0 = derphi_a1;
                continue;
            }
        };
        // A zoom that fails is SciPy's `alpha_star = None`.
        return Ok(found.map(|(alpha, fval)| LineSearch {
            alpha,
            fval,
            old_fval: phi0,
            grad: f.gval.take(),
        }));
    }
    // SciPy's for-else: out of iterations, it still returns the last trial step, without a
    // gradient ("The line search algorithm did not converge").
    Ok(Some(LineSearch {
        alpha: alpha1,
        fval: phi_a1,
        old_fval: phi0,
        grad: None,
    }))
}

/// SciPy `_line_search_wolfe12`: dcsrch, then the Wolfe-2 search if dcsrch fails or its step
/// fails `extra`. `Ok(None)` is SciPy's `_LineSearchError`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn line_search_wolfe12<O: LineObjective>(
    obj: &mut O,
    xk: &[f64],
    pk: &[f64],
    gfk: &[f64],
    old_fval: f64,
    old_old_fval: Option<f64>,
    c1: f64,
    c2: f64,
    amin: f64,
    amax: f64,
    mut extra: Option<ExtraCondition<'_>>,
) -> Result<Option<LineSearch>, O::Error> {
    if let Some(found) = line_search_wolfe1(
        obj,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        c1,
        c2,
        amax,
        amin,
        1e-14,
    )? {
        let accepted = match (extra.as_deref_mut(), &found.grad) {
            (Some(condition), Some(grad)) => {
                condition(found.alpha, &point(xk, found.alpha, pk), found.fval, grad)
            }
            _ => true,
        };
        if accepted {
            return Ok(Some(found));
        }
    }
    line_search_wolfe2(
        obj,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        c1,
        c2,
        Some(amax),
        10,
        extra,
    )
}

pub(crate) struct BfgsParams {
    pub gtol: f64,
    /// `norm=np.inf` (SciPy's default) when true, else the 2-norm.
    pub norm_inf: bool,
    pub maxiter: usize,
    pub xrtol: f64,
    pub c1: f64,
    pub c2: f64,
}

pub(crate) struct BfgsOutcome {
    pub x: Vec<f64>,
    pub fun: f64,
    pub jac: Vec<f64>,
    /// Row-major n × n inverse-Hessian approximation.
    pub hess_inv: Vec<f64>,
    pub nit: usize,
    /// SciPy warnflag: 0 success, 1 maxiter, 2 precision loss, 3 NaN.
    pub status: u8,
    /// The callback asked to stop (SciPy's `StopIteration`, status 99 in `minimize`).
    pub stopped_by_callback: bool,
}

/// SciPy's `_status_message` for a BFGS warnflag.
pub(crate) const fn status_message(status: u8) -> &'static str {
    match status {
        0 => "Optimization terminated successfully.",
        1 => "Maximum number of iterations has been exceeded.",
        2 => "Desired error not necessarily achieved due to precision loss.",
        _ => "NaN result encountered.",
    }
}

fn vecnorm(v: &[f64], inf: bool) -> f64 {
    if inf {
        // `np.amax(np.abs(v))` propagates NaN.
        if v.iter().any(|x| x.is_nan()) {
            f64::NAN
        } else {
            v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
        }
    } else {
        dot(v, v).sqrt()
    }
}

/// SciPy `_minimize_bfgs` (without `hess_inv0`).
pub(crate) fn minimize_bfgs<O: LineObjective>(
    obj: &mut O,
    x0: &[f64],
    params: &BfgsParams,
) -> Result<BfgsOutcome, O::Error> {
    let n = x0.len();
    let mut old_fval = obj.fun(x0)?;
    let mut gfk = obj.grad(x0)?;
    let mut k = 0;
    let mut hk = vec![0.0; n * n];
    for i in 0..n {
        hk[i * n + i] = 1.0;
    }
    let mut old_old_fval = Some(old_fval + dot(&gfk, &gfk).sqrt() / 2.0);
    let mut xk = x0.to_vec();
    let mut warnflag = 0_u8;
    let mut stopped_by_callback = false;
    let mut gnorm = vecnorm(&gfk, params.norm_inf);
    while gnorm > params.gtol && k < params.maxiter {
        let pk: Vec<f64> = (0..n)
            .map(|i| -dot(&hk[i * n..(i + 1) * n], &gfk))
            .collect();
        let Some(ls) = line_search_wolfe12(
            obj,
            &xk,
            &pk,
            &gfk,
            old_fval,
            old_old_fval,
            params.c1,
            params.c2,
            1e-100,
            1e100,
            None,
        )?
        else {
            warnflag = 2;
            break;
        };
        old_old_fval = Some(ls.old_fval);
        old_fval = ls.fval;
        let alpha_k = ls.alpha;
        let sk: Vec<f64> = pk.iter().map(|p| alpha_k * p).collect();
        let xkp1: Vec<f64> = xk.iter().zip(&sk).map(|(x, s)| x + s).collect();
        xk = xkp1;
        let gfkp1 = match ls.grad {
            Some(g) => g,
            None => obj.grad(&xk)?,
        };
        let yk: Vec<f64> = gfkp1.iter().zip(&gfk).map(|(a, b)| a - b).collect();
        gfk = gfkp1;
        k += 1;
        if obj.callback(&xk, old_fval) {
            stopped_by_callback = true;
            break;
        }
        gnorm = vecnorm(&gfk, params.norm_inf);
        if gnorm <= params.gtol {
            break;
        }
        if alpha_k * vecnorm(&pk, false) <= params.xrtol * (params.xrtol + vecnorm(&xk, false)) {
            break;
        }
        if !old_fval.is_finite() {
            warnflag = 2;
            break;
        }
        let rhok_inv = dot(&yk, &sk);
        let rhok = if rhok_inv == 0.0 {
            1000.0
        } else {
            1.0 / rhok_inv
        };
        // Hk ← (I − ρ s yᵀ) Hk (I − ρ y sᵀ) + ρ s sᵀ, as numpy evaluates it:
        // A1 = I − s[:,None]*y[None,:]*ρ, A2 = I − y[:,None]*s[None,:]*ρ, then A1 @ (Hk @ A2).
        // Kept in this form, not the O(n²) rank-two expansion: with a finite-difference
        // gradient the iteration amplifies rounding in Hk, and the expansion rounds differently
        // enough that `minimize(rosen, [-1.2, 1])` ended in precision loss where SciPy converges.
        let mut a1 = vec![0.0; n * n];
        let mut a2 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let id = if i == j { 1.0 } else { 0.0 };
                a1[i * n + j] = id - sk[i] * yk[j] * rhok;
                a2[i * n + j] = id - yk[i] * sk[j] * rhok;
            }
        }
        let hk_a2 = matmul(&hk, &a2, n);
        let prod = matmul(&a1, &hk_a2, n);
        for i in 0..n {
            for j in 0..n {
                hk[i * n + j] = prod[i * n + j] + rhok * sk[i] * sk[j];
            }
        }
    }
    let fval = old_fval;
    if warnflag != 2 {
        if k >= params.maxiter {
            warnflag = 1;
        } else if gnorm.is_nan() || fval.is_nan() || xk.iter().any(|v| v.is_nan()) {
            warnflag = 3;
        }
    }
    Ok(BfgsOutcome {
        x: xk,
        fun: fval,
        jac: gfk,
        hess_inv: hk,
        nit: k,
        status: warnflag,
        stopped_by_callback,
    })
}

pub(crate) struct CgParams {
    pub gtol: f64,
    /// `norm=np.inf` (SciPy's default) when true, else the 2-norm.
    pub norm_inf: bool,
    pub maxiter: usize,
    pub c1: f64,
    pub c2: f64,
}

pub(crate) struct CgOutcome {
    pub x: Vec<f64>,
    pub fun: f64,
    pub jac: Vec<f64>,
    pub nit: usize,
    /// SciPy warnflag, as for BFGS.
    pub status: u8,
    pub stopped_by_callback: bool,
}

/// One Polak–Ribière+ update from `xk` along `pk` (SciPy's `polak_ribiere_powell_step`).
struct PrStep {
    alpha: f64,
    x: Vec<f64>,
    p: Vec<f64>,
    g: Vec<f64>,
    gnorm: f64,
}

fn polak_ribiere_powell_step(
    alpha: f64,
    gfkp1: Vec<f64>,
    xk: &[f64],
    pk: &[f64],
    gfk: &[f64],
    deltak: f64,
    norm_inf: bool,
) -> PrStep {
    let x = point(xk, alpha, pk);
    let yk: Vec<f64> = gfkp1.iter().zip(gfk).map(|(a, b)| a - b).collect();
    // Python's `max(0, v)`: 0 unless v > 0, so a NaN ratio restarts along -g.
    let beta_k = py_max(0.0, dot(&yk, &gfkp1) / deltak);
    let p = gfkp1.iter().zip(pk).map(|(g, p)| -g + beta_k * p).collect();
    let gnorm = vecnorm(&gfkp1, norm_inf);
    PrStep {
        alpha,
        x,
        p,
        g: gfkp1,
        gnorm,
    }
}

/// SciPy `_minimize_cg`: Polak–Ribière+ on `_line_search_wolfe12` with the sufficient-descent
/// condition of Gilbert & Nocedal (σ₃ = 0.01) as the line search's extra condition.
pub(crate) fn minimize_cg<O: LineObjective>(
    obj: &mut O,
    x0: &[f64],
    params: &CgParams,
) -> Result<CgOutcome, O::Error> {
    const SIGMA_3: f64 = 0.01;
    let mut old_fval = obj.fun(x0)?;
    let mut gfk = obj.grad(x0)?;
    let mut k = 0;
    let mut xk = x0.to_vec();
    // Sets the initial step guess to dx ~ 1.
    let mut old_old_fval = Some(old_fval + dot(&gfk, &gfk).sqrt() / 2.0);
    let mut warnflag = 0_u8;
    let mut stopped_by_callback = false;
    let mut pk: Vec<f64> = gfk.iter().map(|g| -g).collect();
    let mut gnorm = vecnorm(&gfk, params.norm_inf);
    while gnorm > params.gtol && k < params.maxiter {
        let deltak = dot(&gfk, &gfk);
        let mut cached: Option<PrStep> = None;
        let found = {
            let mut descent = |alpha: f64, _x: &[f64], _f: f64, gfkp1: &[f64]| {
                let step = polak_ribiere_powell_step(
                    alpha,
                    gfkp1.to_vec(),
                    &xk,
                    &pk,
                    &gfk,
                    deltak,
                    params.norm_inf,
                );
                // Accept a step that converges, or one whose next direction descends enough.
                let accept = step.gnorm <= params.gtol
                    || dot(&step.p, &step.g) <= -SIGMA_3 * dot(&step.g, &step.g);
                cached = Some(step);
                accept
            };
            line_search_wolfe12(
                obj,
                &xk,
                &pk,
                &gfk,
                old_fval,
                old_old_fval,
                params.c1,
                params.c2,
                1e-100,
                1e100,
                Some(&mut descent),
            )?
        };
        let Some(ls) = found else {
            warnflag = 2;
            break;
        };
        old_old_fval = Some(ls.old_fval);
        old_fval = ls.fval;
        // Reuse the step the extra condition computed if it is the accepted one.
        let step = match cached {
            Some(step) if step.alpha == ls.alpha => step,
            _ => {
                let gfkp1 = match ls.grad {
                    Some(g) => g,
                    None => obj.grad(&point(&xk, ls.alpha, &pk))?,
                };
                polak_ribiere_powell_step(ls.alpha, gfkp1, &xk, &pk, &gfk, deltak, params.norm_inf)
            }
        };
        xk = step.x;
        pk = step.p;
        gfk = step.g;
        gnorm = step.gnorm;
        k += 1;
        if obj.callback(&xk, old_fval) {
            stopped_by_callback = true;
            break;
        }
    }
    let fval = old_fval;
    if warnflag != 2 {
        if k >= params.maxiter {
            warnflag = 1;
        } else if gnorm.is_nan() || fval.is_nan() || xk.iter().any(|v| v.is_nan()) {
            warnflag = 3;
        }
    }
    Ok(CgOutcome {
        x: xk,
        fun: fval,
        jac: gfk,
        nit: k,
        status: warnflag,
        stopped_by_callback,
    })
}

/// Row-major n × n product, accumulated over k in order for each entry (the loop order only
/// keeps the inner loop unit-stride).
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for (c_row, a_row) in c.chunks_exact_mut(n).zip(a.chunks_exact(n)) {
        for (&aik, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (cij, &bkj) in c_row.iter_mut().zip(b_row) {
                *cij += aik * bkj;
            }
        }
    }
    c
}
