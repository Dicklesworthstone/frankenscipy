//! Bivariate spline surface fitting — `bisplrep` (FITPACK `surfit`).
//!
//! Faithful safe-Rust port of Dierckx's FITPACK `surfit`/`fpsurf` and the
//! helpers it calls (`fpbspl`, `fpgivs`, `fprota`, `fpback`, `fpdisc`,
//! `fporde`, `fprank`, `fprati`). Given scattered data `(x_i, y_i, z_i)` with
//! weights `w_i`, it determines a smooth bivariate B-spline of degrees `kx`,
//! `ky` whose weighted residual sum of squares is `<= s`. Matches
//! `scipy.interpolate.bisplrep`. Internal arrays use 1-based indexing (index 0
//! unused) so the index arithmetic mirrors the Fortran line-for-line.

use crate::InterpError;

/// Evaluate the `k+1` non-zero B-splines of degree `k` at `t(l) <= x < t(l+1)`
/// (de Boor–Cox). `t` and `h` are 1-based; `h[1..=k+1]` is written.
pub(crate) fn fpbspl(t: &[f64], k: usize, x: f64, l: usize, h: &mut [f64]) {
    let mut hh = [0.0f64; 20];
    h[1] = 1.0;
    for j in 1..=k {
        for i in 1..=j {
            hh[i] = h[i];
        }
        h[1] = 0.0;
        for i in 1..=j {
            let li = l + i;
            let lj = li - j;
            if t[li] != t[lj] {
                let f = hh[i] / (t[li] - t[lj]);
                h[i] += f * (t[li] - x);
                h[i + 1] = f * (x - t[lj]);
            } else {
                h[i + 1] = 0.0;
            }
        }
    }
}

/// Givens transformation parameters; updates `ww` in place, returns `(cos, sin)`.
pub(crate) fn fpgivs(piv: f64, ww: &mut f64) -> (f64, f64) {
    let store = piv.abs();
    let dd = if store >= *ww {
        store * (1.0 + (*ww / piv).powi(2)).sqrt()
    } else {
        *ww * (1.0 + (piv / *ww).powi(2)).sqrt()
    };
    let cos = *ww / dd;
    let sin = piv / dd;
    *ww = dd;
    (cos, sin)
}

/// Apply a Givens rotation to `a` and `b` in place.
#[inline]
pub(crate) fn fprota(cos: f64, sin: f64, a: &mut f64, b: &mut f64) {
    let stor1 = *a;
    let stor2 = *b;
    *b = cos * stor2 + sin * stor1;
    *a = cos * stor1 - sin * stor2;
}

/// Back-substitution for `a*c = z`, `a` an `n×n` upper-triangular band matrix of
/// bandwidth `k` (band storage, 1-based `a[i][1..=k]`). Writes `c[1..=n]`.
pub(crate) fn fpback(a: &[Vec<f64>], z: &[f64], n: usize, k: usize, c: &mut [f64]) {
    let k1 = k - 1;
    c[n] = z[n] / a[n][1];
    if n == 1 {
        return;
    }
    let mut i = n - 1;
    for j in 2..=n {
        let mut store = z[i];
        let i1 = if j <= k1 { j - 1 } else { k1 };
        let mut m = i;
        for l in 1..=i1 {
            m += 1;
            store -= c[m] * a[i][l + 1];
        }
        c[i] = store / a[i][1];
        if i == 1 {
            break;
        }
        i -= 1;
    }
}

/// Discontinuity jumps of the `k`-th derivative of the degree-`k` B-splines at
/// the interior knots. `t` 1-based; writes `b[1..=nk1-k1][1..=k2]`.
pub(crate) fn fpdisc(t: &[f64], n: usize, k2: usize, b: &mut [Vec<f64>]) {
    let k1 = k2 - 1;
    let k = k1 - 1;
    let nk1 = n - k1;
    let nrint = nk1 - k;
    let an = nrint as f64;
    let fac = an / (t[nk1 + 1] - t[k1]);
    let mut h = [0.0f64; 13];
    for l in k2..=nk1 {
        let lmk = l - k1;
        for j in 1..=k1 {
            let ik = j + k1;
            let lj = l + j;
            let lk = lj - k2;
            h[j] = t[l] - t[lk];
            h[ik] = t[l] - t[lj];
        }
        let mut lp = lmk;
        for j in 1..=k2 {
            let mut jk = j;
            let mut prod = h[j];
            for _i in 1..=k {
                jk += 1;
                prod = prod * h[jk] * fac;
            }
            let lk = lp + k1;
            b[lmk][j] = (t[lk] - t[lp]) / prod;
            lp += 1;
        }
    }
}

/// Sort data points into the panels they belong to. `nummer`/`index` 1-based.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fporde(
    x: &[f64],
    y: &[f64],
    m: usize,
    kx: usize,
    ky: usize,
    tx: &[f64],
    nx: usize,
    ty: &[f64],
    ny: usize,
    nummer: &mut [usize],
    index: &mut [usize],
    nreg: usize,
) {
    let kx1 = kx + 1;
    let ky1 = ky + 1;
    let nk1x = nx - kx1;
    let nk1y = ny - ky1;
    let nyy = nk1y - ky;
    for i in 1..=nreg {
        index[i] = 0;
    }
    for im in 1..=m {
        let xi = x[im];
        let yi = y[im];
        let mut l = kx1;
        let mut l1 = l + 1;
        while !(xi < tx[l1] || l == nk1x) {
            l = l1;
            l1 = l + 1;
        }
        let mut k = ky1;
        let mut k1 = k + 1;
        while !(yi < ty[k1] || k == nk1y) {
            k = k1;
            k1 = k + 1;
        }
        let num = (l - kx1) * nyy + k - ky;
        nummer[im] = index[num];
        index[num] = im;
    }
}

/// Rational interpolation root of `r(p)=(u p+v)/(p+w)=0`; updates `p1,f1,p3,f3`.
pub(crate) fn fprati(
    p1: &mut f64,
    f1: &mut f64,
    p2: f64,
    f2: f64,
    p3: &mut f64,
    f3: &mut f64,
) -> f64 {
    let p = if *p3 <= 0.0 {
        (*p1 * (*f1 - *f3) * f2 - p2 * (f2 - *f3) * *f1) / ((*f1 - f2) * *f3)
    } else {
        let h1 = *f1 * (f2 - *f3);
        let h2 = f2 * (*f3 - *f1);
        let h3 = *f3 * (*f1 - f2);
        -(*p1 * p2 * h3 + p2 * *p3 * h1 + *p3 * *p1 * h2) / (*p1 * h1 + p2 * h2 + *p3 * h3)
    };
    if f2 < 0.0 {
        *p3 = p2;
        *f3 = f2;
    } else {
        *p1 = p2;
        *f1 = f2;
    }
    p
}

/// Minimum-norm least-squares solution of a rank-deficient banded triangular
/// system (FITPACK `fprank`). `a` is band storage `a[i][1..=m]` (1-based),
/// `f[1..=n]` the rhs; writes `c[1..=n]`, returns `(sq, rank)`.
pub(crate) fn fprank(
    a: &mut [Vec<f64>],
    f: &mut [f64],
    n: usize,
    m: usize,
    tol: f64,
    c: &mut [f64],
) -> (f64, usize) {
    let m1 = m - 1;
    let mut nl = 0usize;
    let mut sq = 0.0f64;
    let mut h = vec![0.0f64; m + 2];
    for i in 1..=n {
        if a[i][1] > tol {
            continue;
        }
        nl += 1;
        if i == n {
            continue;
        }
        let mut yi = f[i];
        for j in 1..=m1 {
            h[j] = a[i][j + 1];
        }
        h[m] = 0.0;
        let i1 = i + 1;
        for ii in i1..=n {
            let i2 = (n - ii).min(m1);
            let piv = h[1];
            if piv != 0.0 {
                let (cos, sin) = fpgivs(piv, &mut a[ii][1]);
                fprota(cos, sin, &mut yi, &mut f[ii]);
                if i2 == 0 {
                    sq += yi * yi;
                    break;
                }
                for j in 1..=i2 {
                    let j1 = j + 1;
                    let mut hj1 = h[j1];
                    fprota(cos, sin, &mut hj1, &mut a[ii][j1]);
                    h[j] = hj1;
                }
            } else {
                if i2 == 0 {
                    sq += yi * yi;
                    break;
                }
                for j in 1..=i2 {
                    h[j] = h[j + 1];
                }
            }
            h[i2 + 1] = 0.0;
            if ii == n {
                sq += yi * yi;
            }
        }
        if i1 > n {
            sq += yi * yi;
        }
    }
    let rank = n - nl;
    // aa: rank×m band; ff: rank rhs.
    let mut aa = vec![vec![0.0f64; m + 2]; rank + 2];
    let mut ff = vec![0.0f64; n + 2];
    let mut ii = 0usize;
    for i in 1..=n {
        if a[i][1] <= tol {
            continue;
        }
        ii += 1;
        ff[ii] = f[i];
        aa[ii][1] = a[i][1];
        let mut jj = ii;
        let mut kk = 1usize;
        let mut j = i;
        let j1 = (j - 1).min(m1);
        if j1 == 0 {
            continue;
        }
        for k in 1..=j1 {
            j -= 1;
            if a[j][1] <= tol {
                continue;
            }
            kk += 1;
            jj -= 1;
            aa[jj][kk] = a[j][k + 1];
        }
    }
    // Form columns of a with a zero diagonal, rotate them into aa.
    ii = 0;
    let mut i = 1;
    while i <= n {
        ii += 1;
        if a[i][1] > tol {
            i += 1;
            continue;
        }
        ii -= 1;
        if ii == 0 {
            i += 1;
            continue;
        }
        let mut jj = 1usize;
        let mut j = i;
        let j1 = (j - 1).min(m1);
        for k in 1..=j1 {
            j -= 1;
            if a[j][1] <= tol {
                continue;
            }
            h[jj] = a[j][k + 1];
            jj += 1;
        }
        for kk in jj..=m {
            h[kk] = 0.0;
        }
        let mut jjr = ii;
        let mut broke = false;
        for _i1 in 1..=ii {
            let j1 = (jjr - 1).min(m1);
            let piv = h[1];
            if piv == 0.0 {
                if j1 == 0 {
                    broke = true;
                    break;
                }
                for j2 in 1..=j1 {
                    h[j2] = h[j2 + 1];
                }
            } else {
                let (cos, sin) = fpgivs(piv, &mut aa[jjr][1]);
                if j1 == 0 {
                    broke = true;
                    break;
                }
                let mut kk = jjr;
                for j2 in 1..=j1 {
                    let j3 = j2 + 1;
                    kk -= 1;
                    let mut hj3 = h[j3];
                    fprota(cos, sin, &mut hj3, &mut aa[kk][j3]);
                    h[j2] = hj3;
                }
            }
            jjr -= 1;
            h[j1 + 1] = 0.0;
        }
        let _ = broke;
        i += 1;
    }
    // Solve (aa)(f1)=ff.
    ff[rank] = ff[rank] / aa[rank][1];
    if rank > 1 {
        let mut i = rank - 1;
        for j in 2..=rank {
            let mut store = ff[i];
            let i1 = (j - 1).min(m1);
            let mut k = i;
            for _ii in 1..=i1 {
                k += 1;
                store -= ff[k] * aa[i][_ii + 1];
            }
            ff[i] = store / aa[i][1];
            if i == 1 {
                break;
            }
            i -= 1;
        }
    }
    // Solve (aa)'(f2)=f1.
    ff[1] = ff[1] / aa[1][1];
    if rank > 1 {
        for j in 2..=rank {
            let mut store = ff[j];
            let i1 = (j - 1).min(m1);
            let mut k = j;
            for _ii in 1..=i1 {
                k -= 1;
                store -= ff[k] * aa[k][_ii + 1];
            }
            ff[j] = store / aa[j][1];
        }
    }
    // Premultiply f2 by the transpose of a.
    let mut k = 0usize;
    for i in 1..=n {
        let mut store = 0.0f64;
        if a[i][1] > tol {
            k += 1;
        }
        let j1 = i.min(m);
        let mut kk = k;
        let mut ij = i + 1;
        for j in 1..=j1 {
            ij -= 1;
            if a[ij][1] <= tol {
                continue;
            }
            store += a[ij][j] * ff[kk];
            kk -= 1;
        }
        c[i] = store;
    }
    // Contribution of zeroing the small diagonal elements to the residual.
    let mut stor3 = 0.0f64;
    for i in 1..=n {
        if a[i][1] > tol {
            continue;
        }
        let mut store = f[i];
        let i1 = (n - i).min(m1);
        for j in 1..=i1 {
            let ij = i + j;
            store -= c[ij] * a[i][j + 1];
        }
        let stor1 = a[i][1] * c[i];
        stor3 += stor1 * (stor1 - store - store);
    }
    sq += stor3;
    (sq, rank)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
struct FpsurfResult {
    nx: usize,
    ny: usize,
    tx: Vec<f64>,
    ty: Vec<f64>,
    c: Vec<f64>,
    fp: f64,
    ier: i32,
}

/// Core FITPACK `fpsurf` for `iopt >= 0` (automatic knots). `x`/`y` are owned
/// (they may be interchanged internally). Returns trimmed knot/coef vectors.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn fpsurf(
    iopt: i32,
    m: usize,
    mut x: Vec<f64>,
    mut y: Vec<f64>,
    z: &[f64],
    w: &[f64],
    xb: f64,
    xe: f64,
    yb: f64,
    ye: f64,
    kxx: usize,
    kyy: usize,
    s: f64,
    nxest: usize,
    nyest: usize,
    eta: f64,
    tol: f64,
    maxit: usize,
    // For iopt < 0: the user-supplied full knot vectors (1-based, len nx0+1/ny0+1).
    provided: Option<(usize, Vec<f64>, usize, Vec<f64>)>,
) -> FpsurfResult {
    let con1 = 0.1f64;
    let con9 = 0.9f64;
    let con4 = 0.04f64;
    let ten = 10.0f64;

    let nmax = nxest.max(nyest);
    let km1 = kxx.max(kyy) + 1;
    let km2 = km1 + 1;
    let kx1_0 = kxx + 1;
    let ky1_0 = kyy + 1;
    let nxk0 = nxest - kx1_0;
    let nyk0 = nyest - ky1_0;
    let ncest = nxk0 * nyk0;
    // Band widths (min-bandwidth orientation), per surfit.
    let mut ib1 = kxx * nyk0 + ky1_0;
    let jb1 = kyy * nxk0 + kx1_0;
    let mut ib3 = kx1_0 * nyk0 + 1;
    if ib1 > jb1 {
        ib1 = jb1;
        ib3 = ky1_0 * nxk0 + 1;
    }
    let acol = ib1 + 2;
    let qcol = ib3 + 2;

    // Working storage (1-based).
    let mut tx = vec![0.0f64; nmax + 2];
    let mut ty = vec![0.0f64; nmax + 2];
    let mut c = vec![0.0f64; ncest + 2];
    let mut f = vec![0.0f64; ncest + 2];
    let mut ff = vec![0.0f64; ncest + 2];
    let nrint_max = (nxest + nyest) + 2;
    let mut fpint = vec![0.0f64; nrint_max];
    let mut coord = vec![0.0f64; nrint_max];
    let mut a = vec![vec![0.0f64; acol]; ncest + 2];
    let mut q = vec![vec![0.0f64; qcol]; ncest + 2];
    let mut bx = vec![vec![0.0f64; km2 + 1]; nmax + 2];
    let mut by = vec![vec![0.0f64; km2 + 1]; nmax + 2];
    let mut spx = vec![vec![0.0f64; km1 + 1]; m + 2];
    let mut spy = vec![vec![0.0f64; km1 + 1]; m + 2];
    let mut hh = vec![0.0f64; qcol + 1];
    let mut hx = [0.0f64; 7];
    let mut hy = [0.0f64; 7];
    let mut nummer = vec![0usize; m + 2];
    let mut index = vec![0usize; nrint_max + m + 2];

    let mut ichang: i32 = -1;
    let mut x0 = xb;
    let mut x1 = xe;
    let mut y0 = yb;
    let mut y1 = ye;
    let mut kx = kxx;
    let mut ky = kyy;
    let mut kx1 = kx + 1;
    let mut ky1 = ky + 1;
    let mut nxe = nxest;
    let mut nye = nyest;
    let eps = eta.sqrt();
    let acc = tol * s;

    let mut nx;
    let mut ny;
    let mut ier: i32;
    if iopt < 0 {
        // Least-squares spline for a user-given set of knots.
        let (pnx, ptx, pny, pty) = provided.expect("iopt<0 requires provided knots");
        nx = pnx;
        ny = pny;
        for i in 1..=nx {
            tx[i] = ptx[i];
        }
        for i in 1..=ny {
            ty[i] = pty[i];
        }
        ier = 0;
    } else {
        nx = 2 * kx1;
        ny = 2 * ky1;
        ier = -2;
    }
    let mut fp0 = 0.0f64;
    let mut fp = 0.0f64;
    let mut fpms;
    let mut nk1x = 0usize;
    let mut nk1y = 0usize;
    let mut ncof = 0usize;
    let mut nxx;
    let mut nyy = 0usize;
    let mut iband1 = 0usize;
    let mut rank = 0usize;

    // ── part 1: knot determination ──
    let mut do_finish = false;
    let mut do_part2 = false;
    'outer: for _iter in 1..=m {
        // additional knots for the b-spline representation.
        let mut l = nx;
        for i in 1..=kx1 {
            tx[i] = x0;
            tx[l] = x1;
            l -= 1;
        }
        l = ny;
        for i in 1..=ky1 {
            ty[i] = y0;
            ty[l] = y1;
            l -= 1;
        }
        let _ = (kx1, ky1);
        nxx = nx - 2 * kx1 + 1;
        nyy = ny - 2 * ky1 + 1;
        let mut nrint = nxx + nyy;
        let mut nreg = nxx * nyy;
        iband1 = kx * (ny - ky1) + ky;
        l = ky * (nx - kx1) + kx;
        if iband1 > l {
            iband1 = l;
            ichang = -ichang;
            for i in 1..=m {
                std::mem::swap(&mut x[i], &mut y[i]);
            }
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
            let n = nx.min(ny);
            for i in 1..=n {
                let st = tx[i];
                tx[i] = ty[i];
                ty[i] = st;
            }
            let n1 = n + 1;
            match nx.cmp(&ny) {
                std::cmp::Ordering::Less => {
                    for i in n1..=ny {
                        tx[i] = ty[i];
                    }
                }
                std::cmp::Ordering::Greater => {
                    for i in n1..=nx {
                        ty[i] = tx[i];
                    }
                }
                std::cmp::Ordering::Equal => {}
            }
            std::mem::swap(&mut nx, &mut ny);
            std::mem::swap(&mut nxe, &mut nye);
            std::mem::swap(&mut kx, &mut ky);
            kx1 = kx + 1;
            ky1 = ky + 1;
            nxx = nx - 2 * kx1 + 1;
            nyy = ny - 2 * ky1 + 1;
            let _ = (nxx, nrint, nreg);
            nrint = nxx + nyy;
            nreg = nxx * nyy;
        }
        let iband = iband1 + 1;
        fporde(
            &x,
            &y,
            m,
            kx,
            ky,
            &tx,
            nx,
            &ty,
            ny,
            &mut nummer,
            &mut index,
            nreg,
        );
        nk1x = nx - kx1;
        nk1y = ny - ky1;
        ncof = nk1x * nk1y;
        for i in 1..=ncof {
            f[i] = 0.0;
            for j in 1..=iband {
                a[i][j] = 0.0;
            }
        }
        fp = 0.0;
        for num in 1..=nreg {
            let num1 = num - 1;
            let lx = num1 / nyy;
            let l1 = lx + kx1;
            let ly = num1 - lx * nyy;
            let l2 = ly + ky1;
            let jrot = lx * nk1y + ly;
            let mut inp = index[num];
            while inp != 0 {
                let wi = w[inp];
                let mut zi = z[inp] * wi;
                fpbspl(&tx, kx, x[inp], l1, &mut hx);
                fpbspl(&ty, ky, y[inp], l2, &mut hy);
                for i in 1..=kx1 {
                    spx[inp][i] = hx[i];
                }
                for i in 1..=ky1 {
                    spy[inp][i] = hy[i];
                }
                for i in 1..=iband {
                    hh[i] = 0.0;
                }
                let mut i1 = 0usize;
                for i in 1..=kx1 {
                    let hxi = hx[i];
                    let mut j1 = i1;
                    for j in 1..=ky1 {
                        j1 += 1;
                        hh[j1] = hxi * hy[j] * wi;
                    }
                    i1 += nk1y;
                }
                let mut irot = jrot;
                for i in 1..=iband {
                    irot += 1;
                    let piv = hh[i];
                    if piv == 0.0 {
                        continue;
                    }
                    let (cos, sin) = fpgivs(piv, &mut a[irot][1]);
                    fprota(cos, sin, &mut zi, &mut f[irot]);
                    if i == iband {
                        break;
                    }
                    let mut i2 = 1usize;
                    let i3 = i + 1;
                    for j in i3..=iband {
                        i2 += 1;
                        let mut hj = hh[j];
                        fprota(cos, sin, &mut hj, &mut a[irot][i2]);
                        hh[j] = hj;
                    }
                }
                fp += zi * zi;
                inp = nummer[inp];
            }
        }
        let mut dmax = 0.0f64;
        for i in 1..=ncof {
            if a[i][1] > dmax {
                dmax = a[i][1];
            }
        }
        let sigma = eps * dmax;
        let mut rank_deficient = false;
        for i in 1..=ncof {
            if a[i][1] <= sigma {
                rank_deficient = true;
                break;
            }
        }
        if !rank_deficient {
            fpback(&a, &f, ncof, iband, &mut c);
            rank = ncof;
            for i in 1..=ncof {
                q[i][1] = a[i][1] / dmax;
            }
        } else {
            for i in 1..=ncof {
                ff[i] = f[i];
                for j in 1..=iband {
                    q[i][j] = a[i][j];
                }
            }
            let mut cc = vec![0.0f64; ncof + 2];
            let mut qff = ff.clone();
            let (sq, rk) = fprank(&mut q, &mut qff, ncof, iband, sigma, &mut cc);
            for i in 1..=ncof {
                c[i] = cc[i];
            }
            rank = rk;
            for i in 1..=ncof {
                q[i][1] /= dmax;
            }
            fp += sq;
        }
        if ier == -2 {
            fp0 = fp;
        }
        if iopt < 0 {
            do_finish = true;
            break;
        }
        fpms = fp - s;
        if fpms.abs() <= acc {
            if fp <= 0.0 {
                ier = -1;
                fp = 0.0;
            }
            do_finish = true;
            break;
        }
        if fpms < 0.0 {
            do_part2 = true;
            break;
        }
        if ncof > m {
            ier = 4;
            return finish(
                ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty, &mut x,
                &mut y, m, fp, ier,
            );
        }
        ier = 0;
        // search where to add a knot.
        for i in 1..=nrint {
            fpint[i] = 0.0;
            coord[i] = 0.0;
        }
        for num in 1..=nreg {
            let num1 = num - 1;
            let lx = num1 / nyy;
            let l1 = lx + 1;
            let ly = num1 - lx * nyy;
            let l2 = ly + 1 + nxx;
            let jrot = lx * nk1y + ly;
            let mut inp = index[num];
            while inp != 0 {
                let mut store = 0.0f64;
                let mut i1 = jrot;
                for i in 1..=kx1 {
                    let hxi = spx[inp][i];
                    let mut j1 = i1;
                    for j in 1..=ky1 {
                        j1 += 1;
                        store += hxi * spy[inp][j] * c[j1];
                    }
                    i1 += nk1y;
                }
                store = (w[inp] * (z[inp] - store)).powi(2);
                fpint[l1] += store;
                coord[l1] += store * x[inp];
                fpint[l2] += store;
                coord[l2] += store * y[inp];
                inp = nummer[inp];
            }
        }
        // pick the interval with maximal fpint where a knot can still be added.
        let mut added = false;
        loop {
            let mut l = 0usize;
            let mut fpmax = 0.0f64;
            let mut l1 = 1usize;
            let mut l2 = nrint;
            if nx == nxe {
                l1 = nxx + 1;
            }
            if ny == nye {
                l2 = nxx;
            }
            if l1 > l2 {
                ier = 1;
                return finish(
                    ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty,
                    &mut x, &mut y, m, fp, ier,
                );
            }
            for i in l1..=l2 {
                if fpmax < fpint[i] {
                    l = i;
                    fpmax = fpint[i];
                }
            }
            if l == 0 {
                ier = 5;
                return finish(
                    ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty,
                    &mut x, &mut y, m, fp, ier,
                );
            }
            let arg = coord[l] / fpint[l];
            if l > nxx {
                let jxy = l + ky1 - nxx;
                fpint[l] = 0.0;
                let fac1 = ty[jxy] - arg;
                let fac2 = arg - ty[jxy - 1];
                if fac1 > ten * fac2 || fac2 > ten * fac1 {
                    continue;
                }
                let mut j = ny;
                for _i in jxy..=ny {
                    ty[j + 1] = ty[j];
                    j -= 1;
                }
                ty[jxy] = arg;
                ny += 1;
                added = true;
                break;
            } else {
                let jxy = l + kx1;
                fpint[l] = 0.0;
                let fac1 = tx[jxy] - arg;
                let fac2 = arg - tx[jxy - 1];
                if fac1 > ten * fac2 || fac2 > ten * fac1 {
                    continue;
                }
                let mut j = nx;
                for _i in jxy..=nx {
                    tx[j + 1] = tx[j];
                    j -= 1;
                }
                tx[jxy] = arg;
                nx += 1;
                added = true;
                break;
            }
        }
        let _ = added;
        continue 'outer;
    }

    // 820: accept the current least-squares spline (iopt<0 or |fp-s|<=acc).
    if do_finish {
        if ncof != rank {
            ier = -(rank as i32);
        }
        return finish(
            ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty, &mut x,
            &mut y, m, fp, ier,
        );
    }
    let _ = do_part2;
    // 430: if the least-squares polynomial already overshoots, return it.
    if ier == -2 {
        return finish(
            ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty, &mut x,
            &mut y, m, fp, ier,
        );
    }

    // ── part 2: smoothing spline ──
    let kx2 = kx1 + 1;
    if nk1x != kx1 {
        fpdisc(&tx, nx, kx2, &mut bx);
    }
    let ky2 = ky1 + 1;
    if nk1y != ky1 {
        fpdisc(&ty, ny, ky2, &mut by);
    }
    let mut p1 = 0.0f64;
    let mut f1 = fp0 - s;
    let mut p3 = -1.0f64;
    let mut f3 = fp - s;
    let mut p = 0.0f64;
    for i in 1..=ncof {
        p += a[i][1];
    }
    let rn = ncof as f64;
    p = rn / p;
    let iband3 = kx1 * nk1y;
    let iband4 = iband3 + 1;
    let mut ich1 = 0i32;
    let mut ich3 = 0i32;
    let iband = iband1 + 1;

    for iter in 1..=maxit {
        let pinv = 1.0 / p;
        for i in 1..=ncof {
            ff[i] = f[i];
            for j in 1..=iband {
                q[i][j] = a[i][j];
            }
            for j in (iband + 1)..=iband4 {
                q[i][j] = 0.0;
            }
        }
        if nk1y != ky1 {
            for i in ky2..=nk1y {
                let ii = i - ky1;
                for j in 1..=nk1x {
                    for l in 1..=iband {
                        hh[l] = 0.0;
                    }
                    for l in 1..=ky2 {
                        hh[l] = by[ii][l] * pinv;
                    }
                    let mut zi = 0.0f64;
                    let jrot = (j - 1) * nk1y + ii;
                    let mut skip_row = false;
                    for irot in jrot..=ncof {
                        let piv = hh[1];
                        let i2 = iband1.min(ncof - irot);
                        if piv == 0.0 {
                            if i2 == 0 {
                                skip_row = true;
                                break;
                            }
                        } else {
                            let (cos, sin) = fpgivs(piv, &mut q[irot][1]);
                            fprota(cos, sin, &mut zi, &mut ff[irot]);
                            if i2 == 0 {
                                skip_row = true;
                                break;
                            }
                            for l in 1..=i2 {
                                let l1 = l + 1;
                                let mut hl1 = hh[l1];
                                fprota(cos, sin, &mut hl1, &mut q[irot][l1]);
                                hh[l1] = hl1;
                            }
                        }
                        for l in 1..=i2 {
                            hh[l] = hh[l + 1];
                        }
                        hh[i2 + 1] = 0.0;
                    }
                    let _ = skip_row;
                }
            }
        }
        if nk1x != kx1 {
            for i in kx2..=nk1x {
                let ii = i - kx1;
                for j in 1..=nk1y {
                    for l in 1..=iband4 {
                        hh[l] = 0.0;
                    }
                    let mut j1 = 1usize;
                    for l in 1..=kx2 {
                        hh[j1] = bx[ii][l] * pinv;
                        j1 += nk1y;
                    }
                    let mut zi = 0.0f64;
                    let jrot = (i - kx2) * nk1y + j;
                    let mut skip_row = false;
                    for irot in jrot..=ncof {
                        let piv = hh[1];
                        let i2 = iband3.min(ncof - irot);
                        if piv == 0.0 {
                            if i2 == 0 {
                                skip_row = true;
                                break;
                            }
                        } else {
                            let (cos, sin) = fpgivs(piv, &mut q[irot][1]);
                            fprota(cos, sin, &mut zi, &mut ff[irot]);
                            if i2 == 0 {
                                skip_row = true;
                                break;
                            }
                            for l in 1..=i2 {
                                let l1 = l + 1;
                                let mut hl1 = hh[l1];
                                fprota(cos, sin, &mut hl1, &mut q[irot][l1]);
                                hh[l1] = hl1;
                            }
                        }
                        for l in 1..=i2 {
                            hh[l] = hh[l + 1];
                        }
                        hh[i2 + 1] = 0.0;
                    }
                    let _ = skip_row;
                }
            }
        }
        let mut dmax = 0.0f64;
        for i in 1..=ncof {
            if q[i][1] > dmax {
                dmax = q[i][1];
            }
        }
        let sigma = eps * dmax;
        let mut rank_deficient = false;
        for i in 1..=ncof {
            if q[i][1] <= sigma {
                rank_deficient = true;
                break;
            }
        }
        if !rank_deficient {
            fpback(&q, &ff, ncof, iband4, &mut c);
            rank = ncof;
        } else {
            let mut cc = vec![0.0f64; ncof + 2];
            let mut qff = ff.clone();
            let (_sq, rk) = fprank(&mut q, &mut qff, ncof, iband4, sigma, &mut cc);
            for i in 1..=ncof {
                c[i] = cc[i];
            }
            rank = rk;
        }
        for i in 1..=ncof {
            q[i][1] /= dmax;
        }
        // compute f(p).
        fp = 0.0;
        for num in 1..=(nxx_of(nx, kx1) * nyy) {
            let num1 = num - 1;
            let lx = num1 / nyy;
            let ly = num1 - lx * nyy;
            let jrot = lx * nk1y + ly;
            let mut inp = index[num];
            while inp != 0 {
                let mut store = 0.0f64;
                let mut i1 = jrot;
                for i in 1..=kx1 {
                    let hxi = spx[inp][i];
                    let mut j1 = i1;
                    for j in 1..=ky1 {
                        j1 += 1;
                        store += hxi * spy[inp][j] * c[j1];
                    }
                    i1 += nk1y;
                }
                fp += (w[inp] * (z[inp] - store)).powi(2);
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
    finish(
        ichang, nx, ny, nk1x, nk1y, ncof, rank, &mut c, &mut f, &mut tx, &mut ty, &mut x, &mut y,
        m, fp, ier,
    )
}

#[inline]
fn nxx_of(nx: usize, kx1: usize) -> usize {
    nx - 2 * kx1 + 1
}

/// FITPACK end-stage: undo the x/y interchange (if any) and trim outputs.
#[allow(clippy::too_many_arguments)]
fn finish(
    ichang: i32,
    mut nx: usize,
    mut ny: usize,
    nk1x: usize,
    nk1y: usize,
    ncof: usize,
    _rank: usize,
    c: &mut [f64],
    f: &mut [f64],
    tx: &mut [f64],
    ty: &mut [f64],
    x: &mut [f64],
    y: &mut [f64],
    m: usize,
    fp: f64,
    ier: i32,
) -> FpsurfResult {
    if ichang >= 0 {
        // interchange x and y once more.
        let mut l1 = 1usize;
        for i in 1..=nk1x {
            let mut l2 = i;
            for _j in 1..=nk1y {
                f[l2] = c[l1];
                l1 += 1;
                l2 += nk1x;
            }
        }
        for i in 1..=ncof {
            c[i] = f[i];
        }
        // swap full x/y arrays
        for i in 1..=m {
            let st = x[i];
            x[i] = y[i];
            y[i] = st;
        }
        let n = nx.min(ny);
        for i in 1..=n {
            let st = tx[i];
            tx[i] = ty[i];
            ty[i] = st;
        }
        let n1 = n + 1;
        match nx.cmp(&ny) {
            std::cmp::Ordering::Less => {
                for i in n1..=ny {
                    tx[i] = ty[i];
                }
            }
            std::cmp::Ordering::Greater => {
                for i in n1..=nx {
                    ty[i] = tx[i];
                }
            }
            std::cmp::Ordering::Equal => {}
        }
        std::mem::swap(&mut nx, &mut ny);
    }
    let tx_out: Vec<f64> = tx[1..=nx].to_vec();
    let ty_out: Vec<f64> = ty[1..=ny].to_vec();
    let c_out: Vec<f64> = c[1..=ncof].to_vec();
    FpsurfResult {
        nx,
        ny,
        tx: tx_out,
        ty: ty_out,
        c: c_out,
        fp,
        ier,
    }
}

/// Bivariate spline surface fit, matching `scipy.interpolate.bisplrep(x, y, z,
/// kx=kx, ky=ky, s=s)` with the default automatic-knot mode (`task=0`).
///
/// Returns the tck tuple `(tx, ty, c, kx, ky)` consumable by [`crate::bisplev`].
/// Uses scipy's defaults: unit weights, domain bounds from the data extents,
/// `eps = 1e-16`, and `nxest = max(int(kx + sqrt(m/2)), 2 kx + 3)` (similarly
/// for `nyest`).
#[allow(clippy::type_complexity)]
pub fn bisplrep(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    kx: usize,
    ky: usize,
    s: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize), InterpError> {
    let m = x.len();
    if y.len() != m || z.len() != m {
        return Err(InterpError::InvalidArgument {
            detail: "x, y, z must have equal length".to_string(),
        });
    }
    if !(1..=5).contains(&kx) || !(1..=5).contains(&ky) {
        return Err(InterpError::InvalidArgument {
            detail: format!("kx,ky={kx},{ky} unsupported (1<=k<=5)"),
        });
    }
    if m < (kx + 1) * (ky + 1) {
        return Err(InterpError::InvalidArgument {
            detail: "m >= (kx+1)(ky+1) must hold".to_string(),
        });
    }
    if s < 0.0 {
        return Err(InterpError::InvalidArgument {
            detail: "s must be >= 0".to_string(),
        });
    }
    let xb = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xe = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let yb = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let ye = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let nxest = ((kx as f64 + (m as f64 / 2.0).sqrt()) as usize).max(2 * kx + 3);
    let nyest = ((ky as f64 + (m as f64 / 2.0).sqrt()) as usize).max(2 * ky + 3);

    // 1-based copies for the FITPACK core.
    let mut xv = vec![0.0f64; m + 2];
    let mut yv = vec![0.0f64; m + 2];
    let mut zv = vec![0.0f64; m + 2];
    let mut wv = vec![0.0f64; m + 2];
    for i in 1..=m {
        xv[i] = x[i - 1];
        yv[i] = y[i - 1];
        zv[i] = z[i - 1];
        wv[i] = 1.0;
    }

    let res = fpsurf(
        0, m, xv, yv, &zv, &wv, xb, xe, yb, ye, kx, ky, s, nxest, nyest, 1e-16, 1e-3, 20, None,
    );
    if res.ier > 0 {
        return Err(InterpError::InvalidArgument {
            detail: format!("surfit failed (ier={})", res.ier),
        });
    }
    Ok((res.tx, res.ty, res.c, kx, ky))
}

/// Weighted least-squares bivariate spline with user-given interior knots,
/// matching `scipy.interpolate.LSQBivariateSpline` (FITPACK `surfit` with
/// `task=-1`). `tx_interior`/`ty_interior` are the interior knots only; the
/// boundary knots (multiplicity `kx+1` at the data extents) are added
/// internally. Returns the tck tuple `(tx, ty, c, kx, ky)` for [`crate::bisplev`].
#[allow(clippy::type_complexity)]
pub fn lsq_bivariate_spline(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    tx_interior: &[f64],
    ty_interior: &[f64],
    kx: usize,
    ky: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize), InterpError> {
    let m = x.len();
    if y.len() != m || z.len() != m {
        return Err(InterpError::InvalidArgument {
            detail: "x, y, z must have equal length".to_string(),
        });
    }
    if !(1..=5).contains(&kx) || !(1..=5).contains(&ky) {
        return Err(InterpError::InvalidArgument {
            detail: format!("kx,ky={kx},{ky} unsupported (1<=k<=5)"),
        });
    }
    let xb = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xe = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let yb = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let ye = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let kx1 = kx + 1;
    let ky1 = ky + 1;
    let nx = 2 * kx1 + tx_interior.len();
    let ny = 2 * ky1 + ty_interior.len();
    // Build 1-based full knot vectors with the boundary multiplicities.
    let mut ptx = vec![0.0f64; nx + 1];
    let mut pty = vec![0.0f64; ny + 1];
    for i in 1..=kx1 {
        ptx[i] = xb;
        ptx[nx - kx1 + i] = xe;
    }
    for (j, &t) in tx_interior.iter().enumerate() {
        ptx[kx1 + 1 + j] = t;
    }
    for i in 1..=ky1 {
        pty[i] = yb;
        pty[ny - ky1 + i] = ye;
    }
    for (j, &t) in ty_interior.iter().enumerate() {
        pty[ky1 + 1 + j] = t;
    }
    // Validate strict interior-knot ordering inside the domain.
    for i in kx1..nx - kx1 {
        if ptx[i + 1] <= ptx[i] {
            return Err(InterpError::InvalidArgument {
                detail: "x knots must be strictly increasing within (xb, xe)".to_string(),
            });
        }
    }
    for i in ky1..ny - ky1 {
        if pty[i + 1] <= pty[i] {
            return Err(InterpError::InvalidArgument {
                detail: "y knots must be strictly increasing within (yb, ye)".to_string(),
            });
        }
    }

    let mut xv = vec![0.0f64; m + 2];
    let mut yv = vec![0.0f64; m + 2];
    let mut zv = vec![0.0f64; m + 2];
    let mut wv = vec![0.0f64; m + 2];
    for i in 1..=m {
        xv[i] = x[i - 1];
        yv[i] = y[i - 1];
        zv[i] = z[i - 1];
        wv[i] = 1.0;
    }
    // nxest/nyest must bound the supplied knot counts.
    let nxest = nx.max(2 * kx + 3);
    let nyest = ny.max(2 * ky + 3);
    let res = fpsurf(
        -1,
        m,
        xv,
        yv,
        &zv,
        &wv,
        xb,
        xe,
        yb,
        ye,
        kx,
        ky,
        0.0,
        nxest,
        nyest,
        1e-16,
        1e-3,
        20,
        Some((nx, ptx, ny, pty)),
    );
    if res.ier > 0 {
        return Err(InterpError::InvalidArgument {
            detail: format!("surfit (task=-1) failed (ier={})", res.ier),
        });
    }
    Ok((res.tx, res.ty, res.c, kx, ky))
}

/// Evaluate the partial derivative of order `(nux, nuy)` of a bivariate spline
/// on the grid `x × y`, matching `scipy.interpolate.bisplev(x, y, tck, nux, nuy)`
/// (FITPACK `parder`). `0 <= nux < kx`, `0 <= nuy < ky`.
///
/// The derivative of a bivariate spline is itself a (lower-degree) bivariate
/// spline; we form its B-spline coefficients via the de Boor difference
/// recurrence and evaluate with [`crate::bisplev`].
#[allow(clippy::type_complexity)]
pub fn bisplev_derivative(
    x: &[f64],
    y: &[f64],
    tck: &(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize),
    nux: usize,
    nuy: usize,
) -> Result<Vec<Vec<f64>>, InterpError> {
    let (tx0, ty0, c0, kx, ky) = tck;
    let (kx, ky) = (*kx, *ky);
    if nux >= kx || nuy >= ky {
        return Err(InterpError::InvalidArgument {
            detail: format!(
                "derivative orders must satisfy 0<=nux<kx, 0<=nuy<ky (got {nux},{nuy})"
            ),
        });
    }
    let nx = tx0.len();
    let ny = ty0.len();
    let kx1 = kx + 1;
    let ky1 = ky + 1;
    let nkx1 = nx - kx1;
    let nky1 = ny - ky1;
    let nc = nkx1 * nky1;
    if c0.len() != nc {
        return Err(InterpError::InvalidArgument {
            detail: "coefficient array length does not match knot vectors".to_string(),
        });
    }
    // 1-based copies.
    let mut tx = vec![0.0f64; nx + 1];
    let mut ty = vec![0.0f64; ny + 1];
    for i in 1..=nx {
        tx[i] = tx0[i - 1];
    }
    for i in 1..=ny {
        ty[i] = ty0[i - 1];
    }
    let mut wrk = vec![0.0f64; nc + 2];
    for i in 1..=nc {
        wrk[i] = c0[i - 1];
    }
    let mut nxx = nkx1;
    let mut nyy = nky1;
    let mut kkx = kx;
    let mut kky = ky;
    // x-direction differencing.
    if nux > 0 {
        let mut lx = 1usize;
        for _j in 1..=nux {
            let ak = kkx as f64;
            nxx -= 1;
            let mut l1 = lx;
            let mut m0 = 1usize;
            for _i in 1..=nxx {
                l1 += 1;
                let l2 = l1 + kkx;
                let fac = tx[l2] - tx[l1];
                if fac > 0.0 {
                    for _m in 1..=nyy {
                        let m1 = m0 + nyy;
                        wrk[m0] = (wrk[m1] - wrk[m0]) * ak / fac;
                        m0 += 1;
                    }
                }
            }
            lx += 1;
            kkx -= 1;
        }
    }
    // y-direction differencing.
    if nuy > 0 {
        let mut ly = 1usize;
        for _j in 1..=nuy {
            let ak = kky as f64;
            nyy -= 1;
            let mut l1 = ly;
            for i in 1..=nyy {
                l1 += 1;
                let l2 = l1 + kky;
                let fac = ty[l2] - ty[l1];
                if fac > 0.0 {
                    let mut m0 = i;
                    for _m in 1..=nxx {
                        let m1 = m0 + 1;
                        wrk[m0] = (wrk[m1] - wrk[m0]) * ak / fac;
                        m0 += nky1;
                    }
                }
            }
            ly += 1;
            kky -= 1;
        }
        // compact the coefficients from stride nky1 to stride nyy.
        let mut m0 = nyy;
        let mut m1 = nky1;
        for _m in 2..=nxx {
            for _i in 1..=nyy {
                m0 += 1;
                m1 += 1;
                wrk[m0] = wrk[m1];
            }
            m1 += nuy;
        }
    }
    // Build the (reduced) derivative spline and evaluate it.
    let txd: Vec<f64> = tx[(nux + 1)..=(nx - nux)].to_vec();
    let tyd: Vec<f64> = ty[(nuy + 1)..=(ny - nuy)].to_vec();
    let cd: Vec<f64> = wrk[1..=(nxx * nyy)].to_vec();
    crate::bisplev(x, y, &(txd, tyd, cd, kkx, kky))
}

#[cfg(test)]
mod tests {
    use super::{bisplev_derivative, bisplrep, lsq_bivariate_spline};
    use crate::bisplev;

    #[test]
    fn lsq_bivariate_spline_matches_scipy() {
        let (x, y, z) = make_data(11, 4.0, 3.0);
        let tx = [0.35, 0.65];
        let ty = [0.4, 0.75];
        let (txf, tyf, c, kx, ky) = lsq_bivariate_spline(&x, &y, &z, &tx, &ty, 3, 3).unwrap();
        // boundary(4)+interior(2) per axis -> 10 knots each.
        assert_eq!(txf.len(), 10);
        assert_eq!(tyf.len(), 10);
        let ev = bisplev(&[0.3, 0.65], &[0.25, 0.7], &(txf, tyf, c, kx, ky)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        // scipy.interpolate.LSQBivariateSpline oracle.
        let want = [
            0.6825051319867358,
            -0.4725558036296167,
            0.37799170888496864,
            -0.2617155056951865,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    fn make_data(n: usize, fx: f64, fy: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
        for i in 0..n {
            for j in 0..n {
                let xi = i as f64 / (n - 1) as f64;
                let yj = j as f64 / (n - 1) as f64;
                x.push(xi);
                y.push(yj);
                z.push((xi * fx).sin() * (yj * fy).cos());
            }
        }
        (x, y, z)
    }

    #[test]
    fn bisplrep_matches_scipy_polynomial() {
        // Large s -> least-squares bicubic polynomial (no interior knots).
        let (x, y, z) = make_data(9, 4.0, 3.0);
        let (tx, ty, c, kx, ky) = bisplrep(&x, &y, &z, 3, 3, 0.2).unwrap();
        assert_eq!(tx.len(), 8);
        assert_eq!(ty.len(), 8);
        let ev = bisplev(&[0.3, 0.65], &[0.25, 0.7], &(tx, ty, c, kx, ky)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        // scipy.interpolate.bisplev oracle.
        let want = [
            0.6857950628031665,
            -0.4703992921795626,
            0.368667201093501,
            -0.25287553068018775,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn bisplev_derivative_matches_scipy() {
        let (x, y, z) = make_data(11, 4.0, 3.0);
        let tck = bisplrep(&x, &y, &z, 3, 3, 0.05).unwrap();
        let xe = [0.3, 0.65];
        let ye = [0.25, 0.7];
        // scipy.interpolate.bisplev(xe, ye, tck, dx, dy) oracles.
        let cases: [((usize, usize), [f64; 4]); 4] = [
            (
                (1, 0),
                [
                    0.7974601971394006,
                    -0.5384285552518469,
                    -2.275513914343819,
                    1.5535999993784064,
                ],
            ),
            (
                (0, 1),
                [
                    -1.9562629281796118,
                    -2.4238517238365818,
                    -1.0659886054383279,
                    -1.3183170341165176,
                ],
            ),
            (
                (1, 1),
                [
                    -2.410409438743434,
                    -2.8636666378869515,
                    6.630816594290233,
                    8.105481076423876,
                ],
            ),
            (
                (2, 0),
                [
                    -10.30580102423366,
                    7.080537822045849,
                    -5.32053684518824,
                    3.6468391345560938,
                ],
            ),
        ];
        for ((dx, dy), want) in cases {
            let ev = bisplev_derivative(&xe, &ye, &tck, dx, dy).unwrap();
            let flat: Vec<f64> = ev.into_iter().flatten().collect();
            for (a, b) in flat.iter().zip(want.iter()) {
                assert!(
                    (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                    "D{dx}{dy}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn bisplrep_matches_scipy_interior_knots() {
        // Smaller s -> the adaptive knot-insertion + smoothing-spline iteration.
        let (x, y, z) = make_data(9, 4.0, 3.0);
        let (tx, ty, c, kx, ky) = bisplrep(&x, &y, &z, 3, 3, 0.02).unwrap();
        assert!(tx.len() > 8 || ty.len() > 8, "expected interior knots");
        let ev = bisplev(&[0.3, 0.65], &[0.25, 0.7], &(tx, ty, c, kx, ky)).unwrap();
        let flat: Vec<f64> = ev.into_iter().flatten().collect();
        let want = [
            0.6815038958882568,
            -0.46765153429222844,
            0.37148713219926105,
            -0.25468121386409487,
        ];
        for (a, b) in flat.iter().zip(want.iter()) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }
}
