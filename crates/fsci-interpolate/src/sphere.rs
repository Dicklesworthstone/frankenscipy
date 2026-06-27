//! Smooth bicubic spherical spline fitting — `SmoothSphereBivariateSpline`
//! (FITPACK `sphere`/`fpsphe`).
//!
//! Faithful safe-Rust port of Dierckx's FITPACK `sphere`/`fpsphe` (and `fprpsp`
//! for the spherical→standard B-spline coefficient repack), reusing the shared
//! FITPACK helpers from [`crate::surfit`]. Given data `(theta_i, phi_i, r_i)` on
//! the sphere (`0<=theta<=pi`, `0<=phi<=2pi`), determines a smooth bicubic
//! spline `s(theta,phi)` with the pole/periodicity continuity conditions, whose
//! weighted residual sum of squares is `<= s`. Matches
//! `scipy.interpolate.SmoothSphereBivariateSpline`. 1-based indexing mirrors the
//! Fortran line-for-line.

use crate::InterpError;
use crate::surfit::{fpback, fpbspl, fpdisc, fpgivs, fporde, fprank, fprati, fprota};

const PI: f64 = std::f64::consts::PI;

struct SphereResult {
    nt: usize,
    np: usize,
    tt: Vec<f64>,
    tp: Vec<f64>,
    c: Vec<f64>,
    fp: f64,
    ier: i32,
}

/// FITPACK `fpsphe` for `iopt >= 0` (automatic knots).
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn fpsphe(
    iopt: i32,
    m: usize,
    teta: &[f64], // 1-based, len m+1
    phi: &[f64],
    r: &[f64],
    w: &[f64],
    s: f64,
    ntest: usize,
    npest: usize,
    eta: f64,
    tol: f64,
    maxit: usize,
    // For iopt < 0: user-supplied full knot vectors (1-based, len nt0+1 / np0+1).
    provided: Option<(usize, Vec<f64>, usize, Vec<f64>)>,
) -> SphereResult {
    let con1 = 0.1f64;
    let con9 = 0.9f64;
    let con4 = 0.04f64;
    let half = 0.5f64;
    let ten = 10.0f64;
    let pi2 = PI + PI;
    let eps = eta.sqrt();
    let acc = tol * s;

    // Dimensions / band bounds (from the sphere driver, using the upper bounds).
    let ntt_max = ntest - 7;
    let npp_max = npest - 7;
    let ncc = 6 + npp_max * (ntt_max - 1);
    let ncof_max = 6 + 3 * npp_max;
    let mut ib1 = 4 * npp_max;
    if ncof_max > ib1 {
        ib1 = ncof_max;
    }
    let mut ib3 = ib1 + 3;
    if ncof_max > ib3 {
        ib3 = ncof_max;
    }
    let nc = (ntest - 4) * (npest - 4);
    let nrint_max = ntt_max + npp_max + 2;
    let nreg_max = ntt_max * npp_max + 2;

    // Working storage (1-based).
    let mut tt = vec![0.0f64; ntest + 2];
    let mut tp = vec![0.0f64; npest + 2];
    let mut c = vec![0.0f64; nc + 2];
    let mut f = vec![0.0f64; ncc + 2];
    let mut ff = vec![0.0f64; nc + 2];
    let mut row = vec![0.0f64; npest + 2];
    let mut coco = vec![0.0f64; npest + 2];
    let mut cosi = vec![0.0f64; npest + 2];
    let mut a = vec![vec![0.0f64; ib1 + 2]; ncc + 2];
    let mut q = vec![vec![0.0f64; ib3 + 2]; ncc + 2];
    let mut bt = vec![vec![0.0f64; 6]; ntest + 2];
    let mut bp = vec![vec![0.0f64; 6]; npest + 2];
    let mut spt = vec![vec![0.0f64; 5]; m + 2];
    let mut spp = vec![vec![0.0f64; 5]; m + 2];
    let mut h = vec![0.0f64; ib3 + 2];
    let mut fpint = vec![0.0f64; nrint_max];
    let mut coord = vec![0.0f64; nrint_max];
    let mut index = vec![0usize; nreg_max + m + 2];
    let mut nummer = vec![0usize; m + 2];
    let mut ht = [0.0f64; 6];
    let mut hp = [0.0f64; 6];

    let mut nt;
    let mut np;
    let mut ier: i32 = 0;
    let mut fp = 0.0f64;
    let mut sup;
    let mut fpms;

    if iopt < 0 {
        // Least-squares spline for a user-given set of knots.
        let (pnt, ptt, pnp, ptp) = provided.expect("iopt<0 requires provided knots");
        nt = pnt;
        np = pnp;
        for i in 1..=nt {
            tt[i] = ptt[i];
        }
        for i in 1..=np {
            tp[i] = ptp[i];
        }
        sup = 0.0; // unused for iopt<0
        fpms = 0.0;
    } else {
        // ── least-squares constant approximation (the s=inf / p=0 limit) ──
        sup = 0.0;
        let mut d1 = 0.0f64;
        let mut d2 = 0.0f64;
        let mut c1 = 0.0f64;
        let mut cn = 0.0f64;
        let fac1 = PI * (1.0 + half);
        let fac2 = (1.0 + 1.0) / PI.powi(3);
        let mut aa = 0.0f64;
        for i in 1..=m {
            let wi = w[i];
            let mut ri = r[i] * wi;
            let arg = teta[i];
            let mut fnv = fac2 * arg * arg * (fac1 - arg);
            let mut f1 = (1.0 - fnv) * wi;
            fnv *= wi;
            if fnv != 0.0 {
                let (co, si) = fpgivs(fnv, &mut d1);
                fprota(co, si, &mut f1, &mut aa);
                fprota(co, si, &mut ri, &mut cn);
            }
            if f1 != 0.0 {
                let (co, si) = fpgivs(f1, &mut d2);
                fprota(co, si, &mut ri, &mut c1);
            }
            sup += ri * ri;
        }
        if d2 != 0.0 {
            c1 /= d2;
        }
        if d1 != 0.0 {
            cn = (cn - aa * c1) / d1;
        }
        nt = 8;
        np = 8;
        for i in 1..=4 {
            c[i] = c1;
            c[i + 4] = c1;
            c[i + 8] = cn;
            c[i + 12] = cn;
            tt[i] = 0.0;
            tt[i + 4] = PI;
            tp[i] = 0.0;
            tp[i + 4] = pi2;
        }
        fp = sup;
        fpms = sup - s;
        if fpms < acc {
            // least-squares constant accepted.
            return finish(nt, np, &tt, &tp, &c, fp, -2);
        }
        // label 60: start with a minimal interior-knot configuration.
        if npest < 11 || ntest < 9 {
            return finish(nt, np, &tt, &tp, &c, fp, 1);
        }
        np = 11;
        tp[5] = PI * half;
        tp[6] = PI;
        tp[7] = tp[5] + PI;
        nt = 9;
        tt[5] = tp[5];
    }

    let mut ncof = 0usize;
    let mut ncoff;
    let mut iband1 = 0usize;
    let mut iband;
    let mut npp = 0usize;
    let mut ntt;
    let mut np4 = 0usize;
    let mut nt4;
    let mut nrr;
    let mut nrint = 0usize;
    let mut nreg = 0usize;
    let mut rank = 0usize;
    let mut do_part2 = false;
    let mut do_finish = false;

    // ── part 1: knot determination ──
    'outer: for _iter in 1..=m {
        // periodic phi knots.
        let mut l1 = 4usize;
        let mut l2 = l1;
        let mut l3 = np - 3;
        let mut l4 = l3;
        tp[l2] = 0.0;
        tp[l3] = pi2;
        for _i in 1..=3 {
            l1 += 1;
            l2 -= 1;
            l3 += 1;
            l4 -= 1;
            tp[l2] = tp[l4] - pi2;
            tp[l3] = tp[l1] + pi2;
        }
        // pole teta knots.
        let mut l = nt;
        for i in 1..=4 {
            tt[i] = 0.0;
            tt[l] = PI;
            l -= 1;
        }
        ntt = nt - 7;
        npp = np - 7;
        nrr = npp / 2;
        nrint = ntt + npp;
        nreg = ntt * npp;
        fporde(
            teta,
            phi,
            m,
            3,
            3,
            &tt,
            nt,
            &tp,
            np,
            &mut nummer,
            &mut index,
            nreg,
        );
        // coco/cosi: cos/sin projected onto the periodic phi basis.
        for i in 1..=npp {
            coco[i] = 0.0;
            cosi[i] = 0.0;
            for j in 1..=npp {
                a[i][j] = 0.0;
            }
        }
        for i in 1..=npp {
            let l2i = i + 3;
            let arg = tp[l2i];
            fpbspl(&tp, 3, arg, l2i, &mut hp);
            for j in 1..=npp {
                row[j] = 0.0;
            }
            let mut ll = i;
            for j in 1..=3 {
                if ll > npp {
                    ll = 1;
                }
                row[ll] += hp[j];
                ll += 1;
            }
            let mut facc = arg.cos();
            let mut facs = arg.sin();
            for j in 1..=npp {
                let piv = row[j];
                if piv == 0.0 {
                    continue;
                }
                let (co, si) = fpgivs(piv, &mut a[j][1]);
                fprota(co, si, &mut facc, &mut coco[j]);
                fprota(co, si, &mut facs, &mut cosi[j]);
                if j == npp {
                    break;
                }
                let mut i2 = 1usize;
                for l in (j + 1)..=npp {
                    i2 += 1;
                    let mut rl = row[l];
                    fprota(co, si, &mut rl, &mut a[j][i2]);
                    row[l] = rl;
                }
            }
        }
        let coco_in = coco.clone();
        fpback(&a, &coco_in, npp, npp, &mut coco);
        let cosi_in = cosi.clone();
        fpback(&a, &cosi_in, npp, npp, &mut cosi);
        nt4 = nt - 4;
        np4 = np - 4;
        ncoff = nt4 * np4;
        ncof = 6 + npp * (ntt - 1);
        iband = 4 * npp;
        if ntt == 4 {
            iband = 3 * (npp + 1);
        }
        if ntt < 4 {
            iband = ncof;
        }
        iband1 = iband - 1;
        for i in 1..=ncof {
            f[i] = 0.0;
            for j in 1..=iband {
                a[i][j] = 0.0;
            }
        }
        fp = 0.0;
        for num in 1..=nreg {
            let num1 = num - 1;
            let lt0 = num1 / npp;
            let l1p = lt0 + 4;
            let lp = num1 - lt0 * npp + 1;
            let l2p = lp + 3;
            let lt = lt0 + 1;
            let mut jrot = 0usize;
            if lt > 2 {
                jrot = 3 + (lt - 3) * npp;
            }
            let mut inp = index[num];
            while inp != 0 {
                let wi = w[inp];
                let mut ri = r[inp] * wi;
                fpbspl(&tt, 3, teta[inp], l1p, &mut ht);
                fpbspl(&tp, 3, phi[inp], l2p, &mut hp);
                for i in 1..=4 {
                    spp[inp][i] = hp[i];
                    spt[inp][i] = ht[i];
                }
                for i in 1..=iband {
                    h[i] = 0.0;
                }
                for i in 1..=npp {
                    row[i] = 0.0;
                }
                let mut ll = lp;
                for i in 1..=4 {
                    if ll > npp {
                        ll = 1;
                    }
                    row[ll] += hp[i];
                    ll += 1;
                }
                let mut facc = 0.0f64;
                let mut facs = 0.0f64;
                if !(lt > 2 && lt < ntt - 1) {
                    for i in 1..=npp {
                        facc += row[i] * coco[i];
                        facs += row[i] * cosi[i];
                    }
                }
                let mut j1 = 0usize;
                for j in 1..=4 {
                    let jlt = j + lt;
                    let htj = ht[j];
                    if jlt > 2 && jlt <= nt4 {
                        if jlt == 3 || jlt == nt4 {
                            // pole-adjacent row.
                            if jlt == 3 {
                                h[1] += htj;
                                h[2] = facc * htj;
                                h[3] = facs * htj;
                                j1 = 3;
                            } else {
                                h[j1 + 1] = facc * htj;
                                h[j1 + 2] = facs * htj;
                                h[j1 + 3] = htj;
                                j1 += 2;
                            }
                        } else {
                            for i in 1..=npp {
                                j1 += 1;
                                h[j1] = row[i] * htj;
                            }
                        }
                    } else {
                        j1 += 1;
                        h[j1] += htj;
                    }
                }
                for i in 1..=iband {
                    h[i] *= wi;
                }
                let mut irot = jrot;
                for i in 1..=iband {
                    irot += 1;
                    let piv = h[i];
                    if piv == 0.0 {
                        continue;
                    }
                    let (co, si) = fpgivs(piv, &mut a[irot][1]);
                    fprota(co, si, &mut ri, &mut f[irot]);
                    if i == iband {
                        break;
                    }
                    let mut i2 = 1usize;
                    for j in (i + 1)..=iband {
                        i2 += 1;
                        let mut hj = h[j];
                        fprota(co, si, &mut hj, &mut a[irot][i2]);
                        h[j] = hj;
                    }
                }
                fp += ri * ri;
                inp = nummer[inp];
            }
        }
        // solve.
        let mut dmax = 0.0f64;
        for i in 1..=ncof {
            if a[i][1] > dmax {
                dmax = a[i][1];
            }
        }
        let sigma = eps * dmax;
        let mut deficient = false;
        for i in 1..=ncof {
            if a[i][1] <= sigma {
                deficient = true;
                break;
            }
        }
        if !deficient {
            fpback(&a, &f, ncof, iband, &mut c);
            rank = ncof;
        } else {
            let mut qff = f.clone();
            for i in 1..=ncof {
                for j in 1..=iband {
                    q[i][j] = a[i][j];
                }
            }
            let mut cc = vec![0.0f64; ncof + 2];
            let (sq, rk) = fprank(&mut q, &mut qff, ncof, iband, sigma, &mut cc);
            for i in 1..=ncof {
                c[i] = cc[i];
            }
            rank = rk;
            fp += sq;
        }
        // repack spherical coeffs -> standard bicubic.
        let mut ftmp = ff.clone();
        fprpsp(nt, np, &coco, &cosi, &mut c, &mut ftmp, ncoff);
        if iopt < 0 {
            do_finish = true;
            if fp <= 0.0 {
                ier = -1;
                fp = 0.0;
            }
            break;
        }
        fpms = fp - s;
        if fpms.abs() <= acc {
            do_finish = true;
            if fp <= 0.0 {
                ier = -1;
                fp = 0.0;
            }
            break;
        }
        if fpms < 0.0 {
            do_part2 = true;
            break;
        }
        if ncof > m {
            return finish(nt, np, &tt, &tp, &c, fp, 4);
        }
        // knot search.
        for i in 1..=nrint {
            fpint[i] = 0.0;
            coord[i] = 0.0;
        }
        for num in 1..=nreg {
            let num1 = num - 1;
            let lt0 = num1 / npp;
            let l1i = lt0 + 1;
            let lp = num1 - lt0 * npp;
            let l2i = lp + 1 + ntt;
            let jrot = lt0 * np4 + lp;
            let mut inp = index[num];
            while inp != 0 {
                let mut store = 0.0f64;
                let mut i1 = jrot;
                for i in 1..=4 {
                    let hti = spt[inp][i];
                    let mut j1 = i1;
                    for j in 1..=4 {
                        j1 += 1;
                        store += hti * spp[inp][j] * c[j1];
                    }
                    i1 += np4;
                }
                store = (w[inp] * (r[inp] - store)).powi(2);
                fpint[l1i] += store;
                coord[l1i] += store * teta[inp];
                fpint[l2i] += store;
                coord[l2i] += store * phi[inp];
                inp = nummer[inp];
            }
        }
        let mut l1s = 1usize;
        let mut l2s = nrint;
        if ntest < nt + 1 {
            l1s = ntt + 1;
        }
        if npest < np + 2 {
            l2s = ntt;
        }
        if l1s > l2s {
            return finish(nt, np, &tt, &tp, &c, fp, 1);
        }
        // pick interval with maximal fpint; add a knot.
        loop {
            let mut fpmax = 0.0f64;
            let mut lsel = 0usize;
            for i in l1s..=l2s {
                if fpmax < fpint[i] {
                    lsel = i;
                    fpmax = fpint[i];
                }
            }
            if lsel == 0 {
                return finish(nt, np, &tt, &tp, &c, fp, 5);
            }
            let mut arg = coord[lsel] / fpint[lsel];
            if lsel <= ntt {
                // teta-direction insertion.
                let l4i = lsel + 4;
                fpint[lsel] = 0.0;
                let fac1 = tt[l4i] - arg;
                let fac2 = arg - tt[l4i - 1];
                if fac1 > ten * fac2 || fac2 > ten * fac1 {
                    continue;
                }
                let mut j = nt;
                for _i in l4i..=nt {
                    tt[j + 1] = tt[j];
                    j -= 1;
                }
                tt[l4i] = arg;
                nt += 1;
                continue 'outer;
            } else {
                // phi-direction insertion (two symmetric knots).
                let mut l4i = lsel + 4 - ntt;
                if arg >= PI {
                    arg -= PI;
                    l4i -= nrr;
                }
                fpint[lsel] = 0.0;
                let fac1 = tp[l4i] - arg;
                let fac2 = arg - tp[l4i - 1];
                if fac1 > ten * fac2 || fac2 > ten * fac1 {
                    continue;
                }
                let ll = nrr + 4;
                let mut j = ll;
                for _i in l4i..=ll {
                    tp[j + 1] = tp[j];
                    j -= 1;
                }
                tp[l4i] = arg;
                np += 2;
                nrr += 1;
                for i in 5..=ll {
                    let jj = i + nrr;
                    tp[jj] = tp[i] + PI;
                }
                continue 'outer;
            }
        }
    }

    if do_finish {
        if ncof != rank {
            ier = -(rank as i32);
        }
        return finish(nt, np, &tt, &tp, &c, fp, ier);
    }
    let _ = do_part2;

    // ── part 2: smoothing-spline p iteration ──
    fpdisc(&tt, nt, 5, &mut bt);
    fpdisc(&tp, np, 5, &mut bp);
    let mut p1 = 0.0f64;
    let mut f1 = sup - s;
    let mut p3 = -1.0f64;
    let mut f3 = fpms;
    let mut p = 0.0f64;
    for i in 1..=ncof {
        p += a[i][1];
    }
    p = ncof as f64 / p;
    ntt = nt - 7;
    nt4 = nt - 4;
    let iband = 4 * npp;
    let iband_eff = if ntt == 4 {
        3 * (npp + 1)
    } else if ntt < 4 {
        ncof
    } else {
        iband
    };
    iband1 = iband_eff - 1;
    let mut iband4 = iband_eff + 3;
    if ntt <= 4 {
        iband4 = ncof;
    }
    let iband3 = iband4 - 1;
    let mut ich1 = 0i32;
    let mut ich3 = 0i32;
    ncoff = nt4 * np4;

    for iter in 1..=maxit {
        let pinv = 1.0 / p;
        for i in 1..=ncof {
            ff[i] = f[i];
            for j in 1..=iband4 {
                q[i][j] = 0.0;
            }
            for j in 1..=iband_eff {
                q[i][j] = a[i][j];
            }
        }
        let nt6 = nt - 6;
        // phi-direction discontinuity rows.
        for i in 5..=np4 {
            let ii = i - 4;
            for l in 1..=npp {
                row[l] = 0.0;
            }
            let mut ll = ii;
            for l in 1..=5 {
                if ll > npp {
                    ll = 1;
                }
                row[ll] += bp[ii][l];
                ll += 1;
            }
            let mut facc = 0.0f64;
            let mut facs = 0.0f64;
            for l in 1..=npp {
                facc += row[l] * coco[l];
                facs += row[l] * cosi[l];
            }
            for j in 1..=nt6 {
                for l in 1..=iband_eff {
                    h[l] = 0.0;
                }
                let mut jrot = 4 + (j as i64 - 2) * npp as i64;
                if j > 1 && j < nt6 {
                    for l in 1..=npp {
                        h[l] = row[l];
                    }
                } else {
                    h[1] = facc;
                    h[2] = facs;
                    if j == 1 {
                        jrot = 2;
                    }
                }
                for l in 1..=iband_eff {
                    h[l] *= pinv;
                }
                let mut ri = 0.0f64;
                let mut irot = jrot;
                while irot <= ncof as i64 {
                    let piv = h[1];
                    let i2 = (iband1 as i64).min(ncof as i64 - irot).max(0) as usize;
                    if piv != 0.0 {
                        let (co, si) = fpgivs(piv, &mut q[irot as usize][1]);
                        fprota(co, si, &mut ri, &mut ff[irot as usize]);
                        if i2 == 0 {
                            break;
                        }
                        for l in 1..=i2 {
                            let mut hl1 = h[l + 1];
                            fprota(co, si, &mut hl1, &mut q[irot as usize][l + 1]);
                            h[l + 1] = hl1;
                        }
                    } else if i2 == 0 {
                        break;
                    }
                    for l in 1..=i2 {
                        h[l] = h[l + 1];
                    }
                    h[i2 + 1] = 0.0;
                    irot += 1;
                }
            }
        }
        // teta-direction discontinuity rows.
        for i in 5..=nt4 {
            let ii = i - 4;
            for j in 1..=npp {
                for l in 1..=iband4 {
                    h[l] = 0.0;
                }
                let mut j1 = 1i64;
                for l in 1..=5 {
                    let il = ii + l;
                    let mut ij = npp as i64;
                    if il == 3 || il == nt4 {
                        j1 = j1 + 3 - j as i64;
                        let mut j2 = j1 - 2;
                        ij = 0;
                        if il == 3 {
                            j1 = 1;
                            j2 = 2;
                            ij = j as i64 + 2;
                        }
                        h[j2 as usize] = bt[ii][l] * coco[j];
                        h[(j2 + 1) as usize] = bt[ii][l] * cosi[j];
                    }
                    h[j1 as usize] += bt[ii][l];
                    j1 += ij;
                }
                for l in 1..=iband4 {
                    h[l] *= pinv;
                }
                let mut ri = 0.0f64;
                let mut jrot = 1i64;
                if ii > 2 {
                    jrot = 3 + j as i64 + (ii as i64 - 3) * npp as i64;
                }
                let mut irot = jrot;
                while irot <= ncof as i64 {
                    let piv = h[1];
                    let i2 = (iband3 as i64).min(ncof as i64 - irot).max(0) as usize;
                    if piv != 0.0 {
                        let (co, si) = fpgivs(piv, &mut q[irot as usize][1]);
                        fprota(co, si, &mut ri, &mut ff[irot as usize]);
                        if i2 == 0 {
                            break;
                        }
                        for l in 1..=i2 {
                            let mut hl1 = h[l + 1];
                            fprota(co, si, &mut hl1, &mut q[irot as usize][l + 1]);
                            h[l + 1] = hl1;
                        }
                    } else if i2 == 0 {
                        break;
                    }
                    for l in 1..=i2 {
                        h[l] = h[l + 1];
                    }
                    h[i2 + 1] = 0.0;
                    irot += 1;
                }
            }
        }
        // solve.
        let mut dmax = 0.0f64;
        for i in 1..=ncof {
            if q[i][1] > dmax {
                dmax = q[i][1];
            }
        }
        let sigma = eps * dmax;
        let mut deficient = false;
        for i in 1..=ncof {
            if q[i][1] <= sigma {
                deficient = true;
                break;
            }
        }
        if !deficient {
            fpback(&q, &ff, ncof, iband4, &mut c);
            rank = ncof;
        } else {
            let mut qff = ff.clone();
            let mut cc = vec![0.0f64; ncof + 2];
            let (_sq, rk) = fprank(&mut q, &mut qff, ncof, iband4, sigma, &mut cc);
            for i in 1..=ncof {
                c[i] = cc[i];
            }
            rank = rk;
        }
        let mut ftmp = ff.clone();
        fprpsp(nt, np, &coco, &cosi, &mut c, &mut ftmp, ncoff);
        // compute f(p).
        fp = 0.0;
        for num in 1..=nreg {
            let num1 = num - 1;
            let lt0 = num1 / npp;
            let lp = num1 - lt0 * npp;
            let jrot = lt0 * np4 + lp;
            let mut inp = index[num];
            while inp != 0 {
                let mut store = 0.0f64;
                let mut i1 = jrot;
                for i in 1..=4 {
                    let hti = spt[inp][i];
                    let mut j1 = i1;
                    for j in 1..=4 {
                        j1 += 1;
                        store += hti * spp[inp][j] * c[j1];
                    }
                    i1 += np4;
                }
                fp += (w[inp] * (r[inp] - store)).powi(2);
                inp = nummer[inp];
            }
        }
        fpms = fp - s;
        if fpms.abs() <= acc {
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

    if ncof != rank {
        ier = -(rank as i32);
    }
    finish(nt, np, &tt, &tp, &c, fp, ier)
}

/// FITPACK `fprpsp`: repack the reduced spherical spline coefficients into the
/// standard bicubic B-spline representation.
#[allow(clippy::needless_range_loop)]
fn fprpsp(
    nt: usize,
    np: usize,
    co: &[f64],
    si: &[f64],
    c: &mut [f64],
    fbuf: &mut [f64],
    ncoff: usize,
) {
    let nt4 = nt - 4;
    let np4 = np - 4;
    let npp = np4 - 3;
    let ncof = 6 + npp * (nt4 - 4);
    let mut c1 = c[1];
    let cn = c[ncof];
    let mut j = ncoff;
    for i in 1..=np4 {
        fbuf[i] = c1;
        fbuf[j] = cn;
        j -= 1;
    }
    let mut i = np4;
    let mut j = 1usize;
    for l in 3..=nt4 {
        let mut ii = i;
        if l == 3 || l == nt4 {
            if l == nt4 {
                c1 = cn;
            }
            let c2 = c[j + 1];
            let c3 = c[j + 2];
            j += 2;
            for k in 1..=npp {
                i += 1;
                fbuf[i] = c1 + c2 * co[k] + c3 * si[k];
            }
        } else {
            for _k in 1..=npp {
                i += 1;
                j += 1;
                fbuf[i] = c[j];
            }
        }
        for _k in 1..=3 {
            ii += 1;
            i += 1;
            fbuf[i] = fbuf[ii];
        }
    }
    for idx in 1..=ncoff {
        c[idx] = fbuf[idx];
    }
}

fn finish(
    nt: usize,
    np: usize,
    tt: &[f64],
    tp: &[f64],
    c: &[f64],
    fp: f64,
    ier: i32,
) -> SphereResult {
    let ncoff = (nt - 4) * (np - 4);
    SphereResult {
        nt,
        np,
        tt: tt[1..=nt].to_vec(),
        tp: tp[1..=np].to_vec(),
        c: c[1..=ncoff].to_vec(),
        fp,
        ier,
    }
}

/// Smooth bicubic spherical spline approximation of scattered data on the
/// sphere, matching `scipy.interpolate.SmoothSphereBivariateSpline(theta, phi,
/// r, w, s, eps)`.
///
/// Returns the tck tuple `(tt, tp, c)` (bicubic; degrees implicitly `(3, 3)`)
/// consumable by [`crate::bisplev`].
#[allow(clippy::type_complexity)]
pub fn smooth_sphere_bivariate_spline(
    theta: &[f64],
    phi: &[f64],
    r: &[f64],
    w: Option<&[f64]>,
    s: f64,
    eps: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), InterpError> {
    let m = theta.len();
    if phi.len() != m || r.len() != m {
        return Err(InterpError::InvalidArgument {
            detail: "theta, phi, r must have equal length".to_string(),
        });
    }
    if m < 2 {
        return Err(InterpError::InvalidArgument {
            detail: "m >= 2 required".to_string(),
        });
    }
    if !(0.0..1.0).contains(&eps) || eps <= 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "eps must be in (0, 1)".to_string(),
        });
    }
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "s must be >= 0".to_string(),
        });
    }
    for i in 0..m {
        if !(0.0..=PI).contains(&theta[i]) {
            return Err(InterpError::InvalidArgument {
                detail: "theta must be in [0, pi]".to_string(),
            });
        }
        if !(0.0..=(2.0 * PI)).contains(&phi[i]) {
            return Err(InterpError::InvalidArgument {
                detail: "phi must be in [0, 2pi]".to_string(),
            });
        }
    }
    let ntest = 8 + ((m / 2) as f64).sqrt() as usize;
    let npest = 8 + ((m / 2) as f64).sqrt() as usize;
    let ntest = ntest.max(8);
    let npest = npest.max(8);

    // 1-based copies.
    let mut tv = vec![0.0f64; m + 2];
    let mut pv = vec![0.0f64; m + 2];
    let mut rv = vec![0.0f64; m + 2];
    let mut wv = vec![0.0f64; m + 2];
    for i in 1..=m {
        tv[i] = theta[i - 1];
        pv[i] = phi[i - 1];
        rv[i] = r[i - 1];
        wv[i] = w.map_or(1.0, |ww| ww[i - 1]);
    }
    let res = fpsphe(
        0, m, &tv, &pv, &rv, &wv, s, ntest, npest, eps, 1e-3, 20, None,
    );
    if res.ier > 0 {
        return Err(InterpError::InvalidArgument {
            detail: format!("sphere fit failed (ier={})", res.ier),
        });
    }
    Ok((res.tt, res.tp, res.c))
}

/// Weighted least-squares bicubic spherical spline with user-given interior
/// knots, matching `scipy.interpolate.LSQSphereBivariateSpline(theta, phi, r,
/// tt, tp, w, eps)` (FITPACK `sphere` with `iopt=-1`). `tt_interior` are the
/// interior theta knots in `(0, pi)`; `tp_interior` the interior phi knots in
/// `(0, 2pi)`. Returns the tck tuple `(tt, tp, c)` for [`crate::bisplev`].
#[allow(clippy::type_complexity)]
pub fn lsq_sphere_bivariate_spline(
    theta: &[f64],
    phi: &[f64],
    r: &[f64],
    tt_interior: &[f64],
    tp_interior: &[f64],
    w: Option<&[f64]>,
    eps: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), InterpError> {
    let m = theta.len();
    if phi.len() != m || r.len() != m {
        return Err(InterpError::InvalidArgument {
            detail: "theta, phi, r must have equal length".to_string(),
        });
    }
    for &t in tt_interior {
        if !(0.0..PI).contains(&t) {
            return Err(InterpError::InvalidArgument {
                detail: "interior theta knots must be in (0, pi)".to_string(),
            });
        }
    }
    for &p in tp_interior {
        if !(0.0..(2.0 * PI)).contains(&p) {
            return Err(InterpError::InvalidArgument {
                detail: "interior phi knots must be in (0, 2pi)".to_string(),
            });
        }
    }
    // Full knot vectors: 4 boundary copies at 0/pi (theta) and 0/2pi (phi).
    let nt = 8 + tt_interior.len();
    let np = 8 + tp_interior.len();
    let ntest = nt.max(8);
    let npest = np.max(8);
    let mut ptt = vec![0.0f64; ntest + 2];
    let mut ptp = vec![0.0f64; npest + 2];
    for i in 1..=4 {
        ptt[i] = 0.0;
        ptt[nt - 4 + i] = PI;
        ptp[i] = 0.0;
        ptp[np - 4 + i] = 2.0 * PI;
    }
    for (j, &t) in tt_interior.iter().enumerate() {
        ptt[4 + 1 + j] = t;
    }
    for (j, &p) in tp_interior.iter().enumerate() {
        ptp[4 + 1 + j] = p;
    }

    let mut tv = vec![0.0f64; m + 2];
    let mut pv = vec![0.0f64; m + 2];
    let mut rv = vec![0.0f64; m + 2];
    let mut wv = vec![0.0f64; m + 2];
    for i in 1..=m {
        tv[i] = theta[i - 1];
        pv[i] = phi[i - 1];
        rv[i] = r[i - 1];
        wv[i] = w.map_or(1.0, |ww| ww[i - 1]);
    }
    let res = fpsphe(
        -1,
        m,
        &tv,
        &pv,
        &rv,
        &wv,
        0.0,
        ntest,
        npest,
        eps,
        1e-3,
        20,
        Some((nt, ptt, np, ptp)),
    );
    if res.ier > 0 {
        return Err(InterpError::InvalidArgument {
            detail: format!("sphere LSQ fit failed (ier={})", res.ier),
        });
    }
    Ok((res.tt, res.tp, res.c))
}

#[cfg(test)]
mod tests {
    use super::smooth_sphere_bivariate_spline;
    use crate::bisplev;
    use std::f64::consts::PI;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (nt, nph) = (9usize, 12usize);
        let (mut th, mut ph, mut r) = (vec![], vec![], vec![]);
        for i in 0..nt {
            for j in 0..nph {
                let t = PI * (i as f64 + 0.5) / nt as f64;
                let p = 2.0 * PI * j as f64 / nph as f64;
                th.push(t);
                ph.push(p);
                r.push(1.0 + 0.3 * t.cos() + 0.2 * p.sin() * t.sin());
            }
        }
        (th, ph, r)
    }

    #[test]
    fn smooth_sphere_bivariate_spline_matches_scipy() {
        let (th, ph, r) = data();
        let (tt, tp, c) = smooth_sphere_bivariate_spline(&th, &ph, &r, None, 0.5, 1e-16).unwrap();
        // s=0.5 -> interior knots in both directions (knot-finding + part-2 smoothing).
        assert_eq!(tt.len(), 9);
        assert_eq!(tp.len(), 11);
        let ev = bisplev(&[0.5, 1.5, 2.5], &[0.5, 3.0, 5.0], &(tt, tp, c, 3, 3)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        // scipy.interpolate.SmoothSphereBivariateSpline oracle.
        let want = [
            1.2786497110392325,
            1.267583220945373,
            1.2307182379746604,
            1.0495145430050359,
            1.029024980256254,
            0.960769609727319,
            0.7803966668337577,
            0.766976490927095,
            0.7222708478490245,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn lsq_sphere_bivariate_spline_matches_scipy() {
        let (nt, nph) = (11usize, 14usize);
        let (mut th, mut ph, mut r) = (vec![], vec![], vec![]);
        for i in 0..nt {
            for j in 0..nph {
                let t = PI * (i as f64 + 0.5) / nt as f64;
                let p = 2.0 * PI * j as f64 / nph as f64;
                th.push(t);
                ph.push(p);
                r.push(1.0 + 0.3 * t.cos() + 0.2 * p.sin() * t.sin());
            }
        }
        let tt = [PI / 2.0];
        let tp = [PI / 2.0, PI, 3.0 * PI / 2.0];
        let (ttf, tpf, c) =
            super::lsq_sphere_bivariate_spline(&th, &ph, &r, &tt, &tp, None, 1e-16).unwrap();
        assert_eq!(ttf.len(), 9);
        assert_eq!(tpf.len(), 11);
        let ev = bisplev(&[0.5, 1.5, 2.5], &[0.5, 3.0, 5.0], &(ttf, tpf, c, 3, 3)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        // scipy.interpolate.LSQSphereBivariateSpline oracle.
        let want = [
            1.3079668420357589,
            1.2761503091421216,
            1.1701622376431176,
            1.114257725617918,
            1.0479486099139708,
            0.8270579390240861,
            0.8173569197622835,
            0.7777814343618882,
            0.6459465361268361,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn smooth_sphere_constant_limit() {
        // Large s -> the pole-constant spline (no interior knots).
        let (th, ph, r) = data();
        let (tt, tp, _c) = smooth_sphere_bivariate_spline(&th, &ph, &r, None, 5.0, 1e-16).unwrap();
        assert_eq!(tt.len(), 8);
        assert_eq!(tp.len(), 8);
    }
}
