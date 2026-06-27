//! Gridded spherical spline fitting — `RectSphereBivariateSpline`
//! (FITPACK `spgrid`/`fpspgr`), default pole options.
//!
//! Faithful safe-Rust port of Dierckx's FITPACK gridded-spherical spline for the
//! default case used by `scipy.interpolate.RectSphereBivariateSpline(u, v, r, s)`
//! (pole_continuity=False, pole_values=None ⇒ `iopt=[0,0,0]`,
//! `ider=[-1,0,-1,0]`): the pole values are free parameters optimised by the
//! routine. Chain: fpgrsp (the dense gridded tensor solve with periodic v via
//! the cyclic solvers) → fpopsp (free-pole-value Newton step) → fpspgr (knot
//! determination + smoothing p-iteration). 1-based indexing mirrors the Fortran.

use crate::InterpError;
use crate::fitpack_cyclic::fpbacp;
use crate::surfit::{fpback, fpbspl, fpdisc, fpgivs, fprati, fprota};

const PI: f64 = std::f64::consts::PI;

/// FITPACK `fpknot`: insert one knot into the interval of maximal residual.
/// Mutates `t`, `fpint`, `nrdata`; returns the updated `(n, nrint)`.
#[allow(clippy::too_many_arguments)]
fn fpknot(
    x: &[f64],
    t: &mut [f64],
    mut n: usize,
    fpint: &mut [f64],
    nrdata: &mut [i64],
    mut nrint: usize,
    istart: usize,
) -> (usize, usize) {
    let k = (n - nrint - 1) / 2;
    let mut fpmax = 0.0f64;
    let mut jbegin = istart;
    let mut iserr = true;
    let mut number = 0usize;
    let mut maxpt = 0i64;
    let mut maxbeg = 0usize;
    for j in 1..=nrint {
        let jpoint = nrdata[j];
        if !(fpmax >= fpint[j] || jpoint == 0) {
            iserr = false;
            fpmax = fpint[j];
            number = j;
            maxpt = jpoint;
            maxbeg = jbegin;
        }
        jbegin = jbegin + jpoint as usize + 1;
    }
    if !iserr {
        let ihalf = maxpt / 2 + 1;
        let nrx = maxbeg + ihalf as usize;
        let next = number + 1;
        if next <= nrint {
            // jj must descend (nrint..next) so right-shifts don't clobber.
            for j in next..=nrint {
                let jj = next + nrint - j;
                fpint[jj + 1] = fpint[jj];
                nrdata[jj + 1] = nrdata[jj];
                let jk = jj + k;
                t[jk + 1] = t[jk];
            }
        }
        nrdata[number] = ihalf - 1;
        nrdata[next] = maxpt - ihalf;
        let am = maxpt as f64;
        let an = nrdata[number] as f64;
        fpint[number] = fpmax * an / am;
        let an2 = nrdata[next] as f64;
        fpint[next] = fpmax * an2 / am;
        let jk = next + k;
        t[jk] = x[nrx];
    }
    n += 1;
    nrint += 1;
    (n, nrint)
}

/// FITPACK `fpsysy`: solve the symmetric system `a·g = g` in place (`a` is
/// `n×n`, 1-based; LDL^T-style elimination). Result returned in `g`.
fn fpsysy(a: &mut [Vec<f64>], n: usize, g: &mut [f64]) {
    g[1] /= a[1][1];
    if n == 1 {
        return;
    }
    for k in 2..=n {
        a[k][1] /= a[1][1];
    }
    for i in 2..=n {
        for k in i..=n {
            let mut fac = a[k][i];
            for j in 1..=(i - 1) {
                fac -= a[j][j] * a[k][j] * a[i][j];
            }
            a[k][i] = fac;
            if k > i {
                a[k][i] = fac / a[i][i];
            }
        }
    }
    for i in 2..=n {
        let mut fac = g[i];
        for j in 1..=(i - 1) {
            fac -= g[j] * a[j][j] * a[i][j];
        }
        g[i] = fac / a[i][i];
    }
    let mut i = n;
    for _j in 2..=n {
        let i1 = i;
        i -= 1;
        let mut fac = g[i];
        for k in i1..=n {
            fac -= g[k] * a[k][i];
        }
        g[i] = fac;
    }
}

/// Scratch buffers reused across fpgrsp invocations within one fit.
struct GrspWork {
    spu: Vec<[f64; 5]>,
    spv: Vec<[f64; 5]>,
    nru: Vec<usize>,
    nrv: Vec<usize>,
    bu: Vec<Vec<f64>>,
    bv: Vec<Vec<f64>>,
    au: Vec<Vec<f64>>,
    av1: Vec<Vec<f64>>,
    av2: Vec<Vec<f64>>,
    q: Vec<f64>,
    ifsu: bool,
    ifsv: bool,
    ifbu: bool,
    ifbv: bool,
}

/// FITPACK `fpgrsp` (default pole case, iop0=iop1=0). Builds and solves the
/// gridded least-squares/penalised system for the current knots/penalty `p` and
/// pole values `dr[1]`,`dr[4]`, writing the coefficient tensor into `c` and
/// returning `(sq, fp)`. With `iback=true` only `sq` is needed (returns early
/// before back-substitution), used by fpopsp's gradient probes.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn fpgrsp(
    w: &mut GrspWork,
    iback: bool,
    u: &[f64],
    mu: usize,
    v: &[f64],
    mv: usize,
    r: &[f64],
    dr: &[f64],
    tu: &[f64],
    nu: usize,
    tv: &[f64],
    nv: usize,
    p: f64,
    c: &mut [f64],
    fpu: &mut [f64],
    fpv: &mut [f64],
) -> (f64, f64) {
    let nu4 = nu - 4;
    let nu7 = nu - 7;
    let nv4 = nv - 4;
    let nv7 = nv - 7;
    let nv8 = nv - 8;
    let nv11 = nv as i64 - 11;
    let nuu = nu4 - 2; // iop0=iop1=0
    let pinv = if p > 0.0 { 1.0 / p } else { 0.0 };
    let mut h = [0.0f64; 6];

    // (1) u/v B-spline bases.
    if !w.ifsu {
        let mut l = 4usize;
        let mut l1 = 5usize;
        let mut number = 0usize;
        for it in 1..=mu {
            let arg = u[it];
            while !(arg < tu[l1] || l == nu4) {
                l = l1;
                l1 = l + 1;
                number += 1;
            }
            fpbspl(tu, 3, arg, l, &mut h);
            for i in 1..=4 {
                w.spu[it][i] = h[i];
            }
            w.nru[it] = number;
        }
        w.ifsu = true;
    }
    if !w.ifsv {
        let mut l = 4usize;
        let mut l1 = 5usize;
        let mut number = 0usize;
        for it in 1..=mv {
            let arg = v[it];
            while !(arg < tv[l1] || l == nv4) {
                l = l1;
                l1 = l + 1;
                number += 1;
            }
            fpbspl(tv, 3, arg, l, &mut h);
            for i in 1..=4 {
                w.spv[it][i] = h[i];
            }
            w.nrv[it] = number;
        }
        w.ifsv = true;
    }
    // (3) discontinuity jumps for the penalty.
    if p > 0.0 {
        if !w.ifbu && nu - 8 != 0 {
            fpdisc(tu, nu, 5, &mut w.bu);
            w.ifbu = true;
        }
        if !w.ifbv && nv8 != 0 {
            fpdisc(tv, nv, 5, &mut w.bv);
            w.ifbv = true;
        }
    }
    // (4) pole rows (only the constant pole values; iop0=iop1=0).
    let dr01 = dr[1];
    let dr11 = dr[4];
    let mvv = mv; // iop0=iop1=0
    // a0/a1 are constant rows = dr01 / dr11 over the mv columns.

    // (5) STAGE 1 — u direction.
    let ncof_stage = nuu * nv7;
    let mut sq = 0.0f64;
    let lq = mvv * nuu;
    for i in 1..=lq {
        w.q[i] = 0.0;
    }
    for i in 1..=nuu {
        for j in 1..=5 {
            w.au[i][j] = 0.0;
        }
    }
    let mut right = vec![0.0f64; mvv + 2];
    let mut l = 0usize; // index into r (row-major mu×mv)
    let mut nrold = 0usize;
    let mut n1 = nrold + 1;
    for it in 1..=mu {
        let number = w.nru[it];
        loop {
            for j in 1..=mvv {
                right[j] = 0.0;
            }
            let (i0, i1);
            if nrold != number {
                if p <= 0.0 {
                    // skip to advance
                    nrold = n1;
                    n1 += 1;
                    continue;
                }
                for j in 1..=5 {
                    h[j] = w.bu[n1][j] * pinv;
                }
                i0 = 1;
                i1 = 5;
            } else {
                for j in 1..=4 {
                    h[j] = w.spu[it][j];
                }
                for j in 1..=mv {
                    l += 1;
                    right[j] = r[l];
                }
                i0 = 1;
                i1 = 4;
            }
            // subtract pole rows: j0 vs iop0=0 -> only j0==1; j1 vs iop1=0 -> only j1==1.
            let mut i0m = i0;
            let mut i1m = i1;
            // first-pole (a0): active only when n1==1 (the j0 loop runs for j0<=1).
            let mut j0 = n1;
            while j0 == 1 {
                // j0-1 (=0) > iop0(=0)? no -> apply
                let fac0 = h[i0m];
                for j in 1..=mv {
                    right[j] -= fac0 * dr01;
                }
                j0 += 1;
                i0m += 1;
                break;
            }
            // last-pole (a1): the Fortran loops j1 = nu7-number up to 1, decrementing
            // i1 each step. With iop1=0 only j1==1 hits a valid a1 row (=dr11); for
            // j1<1 the (out-of-range) a1 row is zero so it contributes nothing but
            // still shrinks the Givens band by decrementing i1.
            let mut j1 = nu7 as i64 - number as i64;
            while j1 <= 1 {
                if j1 == 1 {
                    let fac1 = h[i1m];
                    for j in 1..=mv {
                        right[j] -= fac1 * dr11;
                    }
                }
                j1 += 1;
                if i1m == 0 {
                    break;
                }
                i1m -= 1;
            }
            // Givens-reduce h[i0m..i1m] into au + rotate right into q.
            let mut irot = nrold as i64 - 1; // iop0=0 -> nrold-iop0-1
            if irot < 0 {
                irot = 0;
            }
            if i0m <= i1m {
                for i in i0m..=i1m {
                    irot += 1;
                    let piv = h[i];
                    if piv == 0.0 {
                        continue;
                    }
                    let ir = irot as usize;
                    let (co, si) = fpgivs(piv, &mut w.au[ir][1]);
                    let mut iq = (ir - 1) * mvv;
                    for j in 1..=mvv {
                        iq += 1;
                        let mut qv = w.q[iq];
                        fprota(co, si, &mut right[j], &mut qv);
                        w.q[iq] = qv;
                    }
                    if i == i1m {
                        continue;
                    }
                    let mut i2 = 1usize;
                    for j in (i + 1)..=i1m {
                        i2 += 1;
                        let mut hj = h[j];
                        fprota(co, si, &mut hj, &mut w.au[ir][i2]);
                        h[j] = hj;
                    }
                }
            }
            for j in 1..=mvv {
                sq += right[j] * right[j];
            }
            if nrold == number {
                break;
            }
            nrold = n1;
            n1 += 1;
        }
    }
    let _ = ncof_stage;

    // (6) STAGE 2 — v direction (periodic).
    let ncof = nuu * nv7;
    for i in 1..=ncof {
        c[i] = 0.0;
    }
    for i in 1..=nv4 {
        w.av1[i][5] = 0.0;
        for j in 1..=4 {
            w.av1[i][j] = 0.0;
            w.av2[i][j] = 0.0;
        }
    }
    let mut jper = false;
    let mut nrold2: i64 = 0;
    let mut h1 = [0.0f64; 6];
    let mut h2 = [0.0f64; 5];
    for it in 1..=mv {
        let number = w.nrv[it] as i64;
        loop {
            let mut rightv = vec![0.0f64; nuu + 2];
            let do_disc;
            if nrold2 != number {
                if p <= 0.0 {
                    nrold2 += 1;
                    continue;
                }
                let n1b = (nrold2 + 1) as usize;
                for j in 1..=5 {
                    h[j] = w.bv[n1b][j] * pinv;
                }
                // mv==mvv (iop=0) -> right stays 0
                do_disc = true;
            } else {
                h[5] = 0.0;
                for j in 1..=4 {
                    h[j] = w.spv[it][j];
                }
                let mut lk = it;
                for j in 1..=nuu {
                    rightv[j] = w.q[lk];
                    lk += mvv;
                }
                do_disc = false;
            }
            let _ = do_disc;
            if nrold2 < nv11 {
                // non-periodic interior row
                let mut irot = nrold2;
                for i in 1..=5 {
                    irot += 1;
                    let piv = h[i];
                    if piv == 0.0 {
                        continue;
                    }
                    let ir = irot as usize;
                    let (co, si) = fpgivs(piv, &mut w.av1[ir][1]);
                    let mut ic = ir;
                    for j in 1..=nuu {
                        let mut cv = c[ic];
                        fprota(co, si, &mut rightv[j], &mut cv);
                        c[ic] = cv;
                        ic += nv7;
                    }
                    if i == 5 {
                        continue;
                    }
                    let mut i2 = 1usize;
                    for j in (i + 1)..=5 {
                        i2 += 1;
                        let mut hj = h[j];
                        fprota(co, si, &mut hj, &mut w.av1[ir][i2]);
                        h[j] = hj;
                    }
                }
                for i in 1..=nuu {
                    sq += rightv[i] * rightv[i];
                }
            } else {
                // periodic wrap rows
                if !jper {
                    // copy the cyclic tail of av1 into av2.
                    let mut jk = (nv11 + 1) as i64;
                    for i in 1..=4 {
                        let mut ik = jk;
                        for j in 1..=5 {
                            if ik <= 0 {
                                break;
                            }
                            w.av2[ik as usize][i] = w.av1[ik as usize][j];
                            ik -= 1;
                        }
                        jk += 1;
                    }
                    jper = true;
                }
                for i in 1..=4 {
                    h1[i] = 0.0;
                    h2[i] = 0.0;
                }
                h1[5] = 0.0;
                let mut j = nrold2 - nv11;
                for i in 1..=5 {
                    j += 1;
                    let mut l0 = j;
                    loop {
                        let l1b = l0 - 4;
                        if l1b <= 0 {
                            h2[l0 as usize] += h[i];
                            break;
                        }
                        if l1b <= nv11 {
                            h1[l1b as usize] = h[i];
                            break;
                        }
                        l0 = l1b - nv11;
                    }
                }
                if nv11 > 0 {
                    let mut broke670 = false;
                    for jj in 1..=(nv11 as usize) {
                        let piv = h1[1];
                        let i2 = (nv11 - jj as i64).min(4).max(0) as usize;
                        if piv != 0.0 {
                            let (co, si) = fpgivs(piv, &mut w.av1[jj][1]);
                            let mut ic = jj;
                            for i in 1..=nuu {
                                let mut cv = c[ic];
                                fprota(co, si, &mut rightv[i], &mut cv);
                                c[ic] = cv;
                                ic += nv7;
                            }
                            for i in 1..=4 {
                                let mut h2i = h2[i];
                                fprota(co, si, &mut h2i, &mut w.av2[jj][i]);
                                h2[i] = h2i;
                            }
                            if i2 == 0 {
                                broke670 = true;
                                break;
                            }
                            for i in 1..=i2 {
                                let mut h1i1 = h1[i + 1];
                                fprota(co, si, &mut h1i1, &mut w.av1[jj][i + 1]);
                                h1[i + 1] = h1i1;
                            }
                        }
                        for i in 1..=i2 {
                            h1[i] = h1[i + 1];
                        }
                        h1[i2 + 1] = 0.0;
                    }
                    let _ = broke670;
                }
                for j in 1..=4 {
                    let ij = nv11 + j as i64;
                    if ij <= 0 {
                        continue;
                    }
                    let piv = h2[j];
                    if piv == 0.0 {
                        continue;
                    }
                    let iju = ij as usize;
                    let (co, si) = fpgivs(piv, &mut w.av2[iju][j]);
                    let mut ic = iju;
                    for i in 1..=nuu {
                        let mut cv = c[ic];
                        fprota(co, si, &mut rightv[i], &mut cv);
                        c[ic] = cv;
                        ic += nv7;
                    }
                    if j == 4 {
                        continue;
                    }
                    for i in (j + 1)..=4 {
                        let mut h2i = h2[i];
                        fprota(co, si, &mut h2i, &mut w.av2[iju][i]);
                        h2[i] = h2i;
                    }
                }
                for i in 1..=nuu {
                    sq += rightv[i] * rightv[i];
                }
            }
            if nrold2 == number {
                break;
            }
            nrold2 += 1;
        }
    }
    if iback {
        return (sq, 0.0);
    }
    // back-substitution.
    if nuu != 0 {
        let mut k = 1usize;
        for _i in 1..=nuu {
            // fpbacp on c[k..] of length nv7, bandwidth 4.
            let mut sub = vec![0.0f64; nv7 + 1];
            for t in 1..=nv7 {
                sub[t] = c[k + t - 1];
            }
            let mut out = vec![0.0f64; nv7 + 1];
            fpbacp(&w.av1, &w.av2, &sub, nv7, 4, &mut out);
            for t in 1..=nv7 {
                c[k + t - 1] = out[t];
            }
            k += nv7;
        }
        // u back-sub via au.
        for j in 1..=nv7 {
            let mut rr = vec![0.0f64; nuu + 1];
            let mut lk = j;
            for i in 1..=nuu {
                rr[i] = c[lk];
                lk += nv7;
            }
            let mut out = vec![0.0f64; nuu + 1];
            fpback(&w.au, &rr, nuu, 5, &mut out);
            let mut lk2 = j;
            for i in 1..=nuu {
                c[lk2] = out[i];
                lk2 += nv7;
            }
        }
    }
    // (8) assemble full coefficient tensor (nu4×nv4) into a temp, copy to c.
    let ncoff = nu4 * nv4;
    let mut q = vec![0.0f64; ncoff + 2];
    let mut jdn = ncoff;
    for l in 1..=nv4 {
        q[l] = dr01;
        q[jdn] = dr11;
        jdn -= 1;
    }
    let mut i = nv4;
    let mut j = 0usize;
    // iop0=0 -> no c0 block
    if nuu != 0 {
        for _l in 1..=nuu {
            let mut ii = i;
            for _kk in 1..=nv7 {
                i += 1;
                j += 1;
                q[i] = c[j];
            }
            for _kk in 1..=3 {
                ii += 1;
                i += 1;
                q[i] = q[ii];
            }
        }
    }
    // iop1=0 -> no c1 block
    for ic in 1..=ncoff {
        c[ic] = q[ic];
    }
    // (9) fp, fpu, fpv.
    let mut fp = 0.0f64;
    for i in 1..=nu {
        fpu[i] = 0.0;
    }
    for i in 1..=nv {
        fpv[i] = 0.0;
    }
    let mut ir = 0usize;
    let mut nroldu = 0usize;
    for i1 in 1..=mu {
        let numu = w.nru[i1];
        let numu1 = numu + 1;
        let mut nroldv = 0usize;
        for i2 in 1..=mv {
            let numv = w.nrv[i2];
            let numv1 = numv + 1;
            ir += 1;
            let mut term = 0.0f64;
            let mut k1 = numu * nv4 + numv;
            for l1 in 1..=4 {
                let mut k2 = k1;
                let fac = w.spu[i1][l1];
                for l2 in 1..=4 {
                    k2 += 1;
                    term += fac * w.spv[i2][l2] * c[k2];
                }
                k1 += nv4;
            }
            term = (r[ir] - term).powi(2);
            fp += term;
            fpu[numu1] += term;
            fpv[numv1] += term;
            let fac = term * 0.5;
            if numv != nroldv {
                fpv[numv1] -= fac;
                fpv[numv] += fac;
            }
            nroldv = numv;
            if numu != nroldu {
                fpu[numu1] -= fac;
                fpu[numu] += fac;
            }
        }
        nroldu = numu;
    }
    (sq, fp)
}

/// FITPACK `fpopsp` (default: free pole values dr[1], dr[4]). Optimises the two
/// pole values with one finite-difference Newton step, then re-evaluates;
/// returns `(sq, fp)` and leaves `c` holding the final coefficients.
#[allow(clippy::too_many_arguments)]
fn fpopsp(
    w: &mut GrspWork,
    u: &[f64],
    mu: usize,
    v: &[f64],
    mv: usize,
    r: &[f64],
    dr: &mut [f64],
    tu: &[f64],
    nu: usize,
    tv: &[f64],
    nv: usize,
    p: f64,
    step: &[f64],
    c: &mut [f64],
    fpu: &mut [f64],
    fpv: &mut [f64],
) -> (f64, f64) {
    w.ifsu = false; // reduction depends on knots/p; recompute each fpopsp call set
    let (mut sq, mut fp) = fpgrsp(
        w, false, u, mu, v, mv, r, dr, tu, nu, tv, nv, p, c, fpu, fpv,
    );
    if sq <= 0.0 {
        return (sq, fp);
    }
    if step[1] <= 0.0 && step[2] <= 0.0 {
        return (sq, fp);
    }
    // free DOFs: nr=[1,4], deltas=[step1,step2].
    let nr = [0usize, 1, 4];
    let delta = [0.0f64, step[1], step[2]];
    let number = 2usize;
    let mut drr = [0.0f64; 7];
    for i in 1..=6 {
        drr[i] = dr[i];
    }
    let mut amat = vec![vec![0.0f64; 7]; 7];
    let mut g = [0.0f64; 7];
    let mut sumv = [0.0f64; 7];
    let mut singular = false;
    for i in 1..=number {
        let li = nr[i];
        let s1 = delta[i];
        drr[li] = dr[li] + s1;
        let (sp, _) = {
            let mut ct = c.to_vec();
            fpgrsp(
                w, true, u, mu, v, mv, r, &drr, tu, nu, tv, nv, p, &mut ct, fpu, fpv,
            )
        };
        sumv[i] = sp;
        drr[li] = dr[li] - s1;
        let (sqq, _) = {
            let mut ct = c.to_vec();
            fpgrsp(
                w, true, u, mu, v, mv, r, &drr, tu, nu, tv, nv, p, &mut ct, fpu, fpv,
            )
        };
        drr[li] = dr[li];
        amat[i][i] = (sumv[i] + sqq - sq - sq) / (s1 * s1);
        if amat[i][i] <= 0.0 {
            singular = true;
            break;
        }
        g[i] = (sqq - sumv[i]) / (s1 + s1);
    }
    if !singular {
        if number >= 2 {
            for i in 2..=number {
                let l1 = nr[i];
                let s1 = delta[i];
                drr[l1] = dr[l1] + s1;
                for jj in 1..=(i - 1) {
                    let l2 = nr[jj];
                    let s2 = delta[jj];
                    drr[l2] = dr[l2] + s2;
                    let (sqq, _) = {
                        let mut ct = c.to_vec();
                        fpgrsp(
                            w, true, u, mu, v, mv, r, &drr, tu, nu, tv, nv, p, &mut ct, fpu, fpv,
                        )
                    };
                    amat[i][jj] = (sq + sqq - sumv[i] - sumv[jj]) / (s1 * s2);
                    drr[l2] = dr[l2];
                }
                drr[l1] = dr[l1];
            }
        }
        fpsysy(&mut amat, number, &mut g);
        for i in 1..=number {
            let li = nr[i];
            dr[li] += g[i];
        }
    }
    let res = fpgrsp(
        w, false, u, mu, v, mv, r, dr, tu, nu, tv, nv, p, c, fpu, fpv,
    );
    sq = res.0;
    fp = res.1;
    (sq, fp)
}

/// FITPACK `fpspgr` (default case): knot determination + smoothing p-iteration.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fpspgr(
    u: &[f64],
    mu: usize,
    v: &[f64],
    mv: usize,
    r: &[f64],
    s: f64,
    nuest: usize,
    nvest: usize,
    tol: f64,
    maxit: usize,
) -> (usize, Vec<f64>, usize, Vec<f64>, Vec<f64>, f64, i32) {
    let con1 = 0.1f64;
    let con9 = 0.9f64;
    let con4 = 0.04f64;
    let acc = tol * s;
    let per = PI + PI;
    let vb = v[1];
    let ve = vb + per;
    let numax = mu + 6;
    let nvmax = mv + 7;
    let nue = numax.min(nuest);
    let nve = nvmax.min(nvest);

    let mut tu = vec![0.0f64; nuest + 2];
    let mut tv = vec![0.0f64; nvest + 2];
    let ncoff = (nuest - 4) * (nvest - 4);
    let mut c = vec![0.0f64; ncoff + 2];
    let mut fpintu = vec![0.0f64; nuest + 2];
    let mut fpintv = vec![0.0f64; nvest + 2];
    let mut nrdatu = vec![0i64; nuest + 2];
    let mut nrdatv = vec![0i64; nvest + 2];
    let mut dr = [0.0f64; 7];
    let mut step = [0.0f64; 3];

    // GrspWork scratch.
    let nv11_cap = nvest;
    let mut w = GrspWork {
        spu: vec![[0.0; 5]; mu + 2],
        spv: vec![[0.0; 5]; mv + 2],
        nru: vec![0; mu + 2],
        nrv: vec![0; mv + 2],
        bu: vec![vec![0.0; 6]; nuest + 2],
        bv: vec![vec![0.0; 6]; nvest + 2],
        au: vec![vec![0.0; 6]; nuest + 2],
        av1: vec![vec![0.0; 7]; nv11_cap + 2],
        av2: vec![vec![0.0; 5]; nv11_cap + 2],
        q: vec![0.0; (nuest) * (mv + nvest) + 2],
        ifsu: false,
        ifsv: false,
        ifbu: false,
        ifbv: false,
    };

    let mut ier: i32;
    let mut p = -1.0f64;
    let mut fp = 0.0f64;
    let mut fp0 = 0.0f64;
    let mut fpold = 0.0f64;
    let mut reducu = 0.0f64;
    let mut reducv = 0.0f64;
    let mut lastdi = 0i32;
    let mut nplusu = 0usize;
    let mut nplusv = 0usize;
    let mut lastu0 = usize::MAX;
    let mut lastu1 = usize::MAX;
    let mumin = 3usize; // ider(1)>=0? no(-1); so mumin stays 4-1=3 (id0=-1 -> mumin-1). default: 4-1-1=2? recompute below

    // mumin: 4; ider(1)>=0 -> -1 (id0=-1, NOT >=0, so no). Actually ider=[-1,0,-1,0]:
    //   ider(1)=-1 (not >=0) -> no decrement; ider(3)=-1 -> no decrement;
    //   iopt(2)=0 -> no; iopt(3)=0 -> no. So mumin stays 4.
    let mumin = {
        let _ = mumin;
        4usize
    };

    // initial: ier=-2 path (iopt=0, s>0 start with minimal knots).
    ier = -2;
    let mut nu = 8usize;
    let mut nv = 8usize;
    nrdatu[1] = (mu as i64) - 2;
    nrdatv[1] = (mv as i64) - 1;
    // idd handled implicitly (id0=id1=-1 free; step computed below).

    let mpm = mu + mv;
    let mut go_part2 = false;
    let mut finished = false;
    for _iter in 1..=mpm {
        let nrintu = nu - 7;
        let nrintv = nv - 7;
        // boundary knots.
        let mut iidx = nu;
        for j in 1..=4 {
            tu[j] = 0.0;
            tu[iidx] = PI;
            iidx -= 1;
        }
        let mut l1 = 4usize;
        let mut l2 = l1;
        let mut l3 = nv - 3;
        let mut l4 = l3;
        tv[l2] = vb;
        tv[l3] = ve;
        for _j in 1..=3 {
            l1 += 1;
            l2 -= 1;
            l3 += 1;
            l4 -= 1;
            tv[l2] = tv[l4] - per;
            tv[l3] = tv[l1] + per;
        }
        // step sizes from pole-adjacent residual ranges.
        let mut ktu = nrdatu[1] as usize + 2;
        if ktu < mumin {
            ktu = mumin;
        }
        if ktu != lastu0 {
            let mut rmin = r[1];
            let mut rmax = r[1];
            let lcnt = mv * ktu;
            for i in 1..=lcnt {
                if r[i] < rmin {
                    rmin = r[i];
                }
                if r[i] > rmax {
                    rmax = r[i];
                }
            }
            step[1] = rmax - rmin;
            lastu0 = ktu;
        }
        let mut ktu1 = nrdatu[nrintu] as usize + 2;
        if ktu1 < mumin {
            ktu1 = mumin;
        }
        if ktu1 != lastu1 {
            let mut rmin = r[mu * mv];
            let mut rmax = r[mu * mv];
            let lcnt = mv * ktu1;
            let mut jr = mu * mv;
            for _i in 1..=lcnt {
                if r[jr] < rmin {
                    rmin = r[jr];
                }
                if r[jr] > rmax {
                    rmax = r[jr];
                }
                jr -= 1;
            }
            step[2] = rmax - rmin;
            lastu1 = ktu1;
        }
        let res = fpopsp(
            &mut w,
            u,
            mu,
            v,
            mv,
            r,
            &mut dr,
            &tu,
            nu,
            &tv,
            nv,
            p,
            &step,
            &mut c,
            &mut fpintu,
            &mut fpintv,
        );
        fp = res.1;
        if step[1] < 0.0 {
            step[1] = -step[1];
        }
        if step[2] < 0.0 {
            step[2] = -step[2];
        }
        if ier == -2 {
            fp0 = fp;
        }
        let fpms = fp - s;
        if fpms.abs() < acc {
            finished = true;
            break;
        }
        if fpms < 0.0 {
            go_part2 = true;
            break;
        }
        if nu == numax && nv == nvmax {
            ier = -1;
            fp = 0.0;
            finished = true;
            break;
        }
        if nu == nue && nv == nve {
            ier = 1;
            finished = true;
            break;
        }
        ier = 0;
        // Direction decision (FITPACK labels 150-230). The very first knot
        // (lastdi==0) is ALWAYS added in v; otherwise record the reduction in
        // the last direction, recompute the adaptive knot counts, and compare.
        let mut nplu = 1usize;
        let mut nplv = 3usize;
        let mut go_add_u;
        let mut go_add_v;
        if lastdi == 0 {
            nplv = 3;
            fpold = fp;
            go_add_u = false;
            go_add_v = true;
        } else {
            if lastdi < 0 {
                reducu = fpold - fp;
            } else {
                reducv = fpold - fp;
            }
            fpold = fp;
            if nu != 8 {
                let mut npl1 = nplusu * 2;
                let rn = nplusu as f64;
                if reducu > acc {
                    npl1 = (rn * fpms / reducu) as usize;
                }
                nplu = (nplusu * 2).min(npl1.max((nplusu / 2).max(1)));
            }
            if nv != 8 {
                let mut npl1 = nplusv * 2;
                let rn = nplusv as f64;
                if reducv > acc {
                    npl1 = (rn * fpms / reducv) as usize;
                }
                nplv = (nplusv * 2).min(npl1.max((nplusv / 2).max(1)));
            }
            if nplu < nplv {
                go_add_u = true;
                go_add_v = false;
            } else if nplu == nplv {
                if lastdi < 0 {
                    go_add_u = false;
                    go_add_v = true;
                } else {
                    go_add_u = true;
                    go_add_v = false;
                }
            } else {
                go_add_u = false;
                go_add_v = true;
            }
        }
        // 210/230 mutual fallback when a direction is already maxed.
        if go_add_u && nu == nue {
            go_add_u = false;
            go_add_v = true;
        }
        if go_add_v && nv == nve {
            go_add_v = false;
            go_add_u = true;
        }
        if go_add_u {
            lastdi = -1;
            nplusu = nplu;
            w.ifsu = false;
            let mut nri = nrintu;
            for _l in 1..=nplusu {
                let (nn, ni) = fpknot(u, &mut tu, nu, &mut fpintu, &mut nrdatu, nri, 1);
                nu = nn;
                nri = ni;
                if nu == nue {
                    break;
                }
            }
        } else if go_add_v {
            lastdi = 1;
            nplusv = nplv;
            w.ifsv = false;
            let mut nri = nrintv;
            for _l in 1..=nplusv {
                let (nn, ni) = fpknot(v, &mut tv, nv, &mut fpintv, &mut nrdatv, nri, 1);
                nv = nn;
                nri = ni;
                if nv == nve {
                    break;
                }
            }
        }
    }

    if finished {
        let nc = (nu - 4) * (nv - 4);
        return (
            nu,
            tu[1..=nu].to_vec(),
            nv,
            tv[1..=nv].to_vec(),
            c[1..=nc].to_vec(),
            fp,
            ier,
        );
    }
    let _ = go_part2;
    if ier == -2 {
        let nc = (nu - 4) * (nv - 4);
        return (
            nu,
            tu[1..=nu].to_vec(),
            nv,
            tv[1..=nv].to_vec(),
            c[1..=nc].to_vec(),
            fp,
            ier,
        );
    }

    // ── smoothing-spline p-iteration ──
    let mut p1 = 0.0f64;
    let mut f1 = fp0 - s;
    let mut p3 = -1.0f64;
    let mut f3 = fp - s;
    p = 1.0;
    let mut drr = dr;
    let mut ich1 = 0i32;
    let mut ich3 = 0i32;
    for iter in 1..=maxit {
        let res = fpopsp(
            &mut w,
            u,
            mu,
            v,
            mv,
            r,
            &mut drr,
            &tu,
            nu,
            &tv,
            nv,
            p,
            &step,
            &mut c,
            &mut fpintu,
            &mut fpintv,
        );
        fp = res.1;
        let fpms = fp - s;
        if fpms.abs() < acc {
            break;
        }
        if iter == maxit {
            ier = 3;
            break;
        }
        let p2 = p;
        let f2 = fpms;
        if ich3 == 0 {
            if (f2 - f3) <= acc {
                p3 = p2;
                f3 = f2;
                p *= con4;
                if p <= p1 {
                    p = p1 * con9 + p2 * con1;
                }
                continue;
            }
            if f2 < 0.0 {
                ich3 = 1;
            }
        }
        if ich1 == 0 {
            if (f1 - f2) <= acc {
                p1 = p2;
                f1 = f2;
                p /= con4;
                if p3 >= 0.0 && p >= p3 {
                    p = p2 * con1 + p3 * con9;
                }
                continue;
            }
            if f2 > 0.0 {
                ich1 = 1;
            }
        }
        if f2 >= f1 || f2 <= f3 {
            ier = 2;
            break;
        }
        p = fprati(&mut p1, &mut f1, p2, f2, &mut p3, &mut f3);
    }
    let nc = (nu - 4) * (nv - 4);
    (
        nu,
        tu[1..=nu].to_vec(),
        nv,
        tv[1..=nv].to_vec(),
        c[1..=nc].to_vec(),
        fp,
        ier,
    )
}

/// Smooth bicubic spline approximation of data on a latitude/longitude grid,
/// matching `scipy.interpolate.RectSphereBivariateSpline(u, v, r, s)` with the
/// default pole options (`pole_continuity=False`, `pole_values=None`).
///
/// `u` are strictly-increasing colatitudes in `(0, π)`, `v` strictly-increasing
/// longitudes with `v[0] >= -π` and `v[-1] <= v[0] + 2π`; `r` is the row-major
/// `len(u) × len(v)` data grid. Returns the tck tuple `(tu, tv, c)` (bicubic).
#[allow(clippy::type_complexity)]
pub fn rect_sphere_bivariate_spline(
    u: &[f64],
    v: &[f64],
    r: &[f64],
    s: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), InterpError> {
    let mu = u.len();
    let mv = v.len();
    if r.len() != mu * mv {
        return Err(InterpError::InvalidArgument {
            detail: "r must be len(u)*len(v) (row-major)".to_string(),
        });
    }
    if mu < 2 || mv < 2 {
        return Err(InterpError::InvalidArgument {
            detail: "need at least 2 points per axis".to_string(),
        });
    }
    if !(u[0] > 0.0 && u[mu - 1] < PI) {
        return Err(InterpError::InvalidArgument {
            detail: "u must lie strictly in (0, pi)".to_string(),
        });
    }
    if !(v[0] >= -PI && v[mv - 1] <= v[0] + 2.0 * PI) {
        return Err(InterpError::InvalidArgument {
            detail: "v must satisfy v[0] >= -pi and v[-1] <= v[0] + 2pi".to_string(),
        });
    }
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "s must be >= 0".to_string(),
        });
    }
    // 1-based copies.
    let mut uv = vec![0.0f64; mu + 2];
    let mut vv = vec![0.0f64; mv + 2];
    let mut rv = vec![0.0f64; mu * mv + 2];
    for i in 1..=mu {
        uv[i] = u[i - 1];
    }
    for i in 1..=mv {
        vv[i] = v[i - 1];
    }
    for i in 1..=(mu * mv) {
        rv[i] = r[i - 1];
    }
    let nuest = (mu + 6).max(8);
    let nvest = (mv + 7).max(11);
    let (nu, tu, nv, tv, c, _fp, ier) = fpspgr(&uv, mu, &vv, mv, &rv, s, nuest, nvest, 1e-3, 20);
    if ier > 0 {
        return Err(InterpError::InvalidArgument {
            detail: format!("spgrid fit failed (ier={ier})"),
        });
    }
    let _ = (nu, nv);
    Ok((tu, tv, c))
}

#[cfg(test)]
mod tests {
    use super::rect_sphere_bivariate_spline;
    use crate::bisplev;
    use std::f64::consts::PI;

    #[test]
    fn rect_sphere_bivariate_spline_matches_scipy() {
        let (mu, mv) = (9usize, 12usize);
        let u: Vec<f64> = (0..mu).map(|i| PI * (i as f64 + 0.5) / mu as f64).collect();
        let v: Vec<f64> = (0..mv).map(|j| 2.0 * PI * j as f64 / mv as f64).collect();
        let mut r = vec![0.0f64; mu * mv];
        for i in 0..mu {
            for j in 0..mv {
                r[i * mv + j] = 1.0 + 0.3 * u[i].cos() + 0.2 * v[j].sin() * u[i].sin();
            }
        }
        let (tu, tv, c) = rect_sphere_bivariate_spline(&u, &v, &r, 0.5).unwrap();
        assert_eq!(tu.len(), 8);
        assert_eq!(tv.len(), 11);
        let ev = bisplev(&[0.5, 1.5, 2.5], &[0.5, 3.0, 5.0], &(tu, tv, c, 3, 3)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        // scipy.interpolate.RectSphereBivariateSpline oracle.
        let want = [
            1.279724666318102,
            1.268738166169769,
            1.232139647810211,
            1.049969333880451,
            1.0294869655171937,
            0.9612555611146137,
            0.7779973079033633,
            0.7646552474341715,
            0.7202098245566996,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }
}
