//! QUADPACK's adaptive integrators, as `scipy.integrate.quad` runs them -- frankenscipy-1ksfv.8.
//!
//! * [`qagse`]: finite `[a, b]`. 21-point Gauss–Kronrod panels, the panel with the largest error
//!   estimate bisected first, `limit` = maximum number of SUBINTERVALS, and Wynn's epsilon
//!   algorithm ([`qelg`]) extrapolating the sequence of sums, which is what makes endpoint
//!   singularities such as `x^-0.9` converge.
//! * [`qagie`]: `[bound, ∞)`, `(-∞, bound]` or `(-∞, ∞)` mapped onto `t ∈ (0, 1]` by
//!   `x = bound + sign·(1 − t)/t`, with 15-point Kronrod panels on `t`.
//! * [`qagpe`]: finite `[a, b]` with user breakpoints (`points=`) as initial panel edges.
//!
//! Port of Piessens, de Doncker-Kapenga, Überhuber & Kahaner (1983), netlib `dqagse`, `dqagie`,
//! `dqagpe`, `dqk21`, `dqk15i`, `dqelg`, `dqpsrt`, statement by statement. The interval lists
//! are 1-based internally (index 0 unused) so the control flow can be checked against the
//! Fortran line by line; [`Qags`] reports them 0-based.
//!
//! `fsci_integrate::quad` used to be a recursive GK15 bisection that treated `limit` as a
//! recursion DEPTH (up to 2^limit panels), bisected every unconverged panel, had no
//! extrapolation, and truncated infinite ranges at `t = 1 − 1e-10`.

/// The outcome of one QUADPACK integration, with SciPy's `infodict` lists (0-based).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qags {
    pub result: f64,
    pub abserr: f64,
    pub neval: usize,
    /// QUADPACK's `ier`: 0 success, 1 `limit` reached, 2 roundoff, 3 bad integrand behaviour,
    /// 4 extrapolation roundoff, 5 divergent or slowly convergent, 6 invalid input.
    pub ier: u8,
    /// Number of subintervals produced.
    pub last: usize,
    pub alist: Vec<f64>,
    pub blist: Vec<f64>,
    pub rlist: Vec<f64>,
    pub elist: Vec<f64>,
    /// Subinterval indices (0-based) in decreasing order of error estimate, as far as
    /// QUADPACK keeps them sorted.
    pub iord: Vec<usize>,
}

/// One Kronrod panel: `(result, abserr, resabs, resasc)` as `dqk21`/`dqk15i` return them.
#[derive(Debug, Clone, Copy)]
struct Panel {
    result: f64,
    abserr: f64,
    /// ∫|f| on the panel.
    resabs: f64,
    /// ∫|f − mean(f)| on the panel.
    resasc: f64,
}

const EPMACH: f64 = f64::EPSILON;
const UFLOW: f64 = f64::MIN_POSITIVE;
const OFLOW: f64 = f64::MAX;

/// The "invalid input" test every driver starts with.
fn invalid_tolerances(epsabs: f64, epsrel: f64) -> bool {
    epsabs <= 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28)
}

/// QUADPACK's error rescaling shared by `dqk21` and `dqk15i`.
fn kronrod_error(resk: f64, resg: f64, hlgth: f64, resabs: f64, resasc: f64) -> f64 {
    let mut abserr = ((resk - resg) * hlgth).abs();
    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * (200.0 * abserr / resasc).powf(1.5).min(1.0);
    }
    if resabs > UFLOW / (50.0 * EPMACH) {
        abserr = (EPMACH * 50.0 * resabs).max(abserr);
    }
    abserr
}

const XGK21: [f64; 11] = [
    0.995_657_163_025_808_1,
    0.973_906_528_517_171_7,
    0.930_157_491_355_708_2,
    0.865_063_366_688_984_5,
    0.780_817_726_586_416_9,
    0.679_409_568_299_024_4,
    0.562_757_134_668_604_7,
    0.433_395_394_129_247_2,
    0.294_392_862_701_460_2,
    0.148_874_338_981_631_22,
    0.0,
];
const WGK21: [f64; 11] = [
    0.011_694_638_867_371_874,
    0.032_558_162_307_964_725,
    0.054_755_896_574_351_995,
    0.075_039_674_810_919_96,
    0.093_125_454_583_697_6,
    0.109_387_158_802_297_64,
    0.123_491_976_262_065_84,
    0.134_709_217_311_473_34,
    0.142_775_938_577_060_09,
    0.147_739_104_901_338_49,
    0.149_445_554_002_916_9,
];
const WG10: [f64; 5] = [
    0.066_671_344_308_688_14,
    0.149_451_349_150_580_6,
    0.219_086_362_515_982_04,
    0.269_266_719_309_996_35,
    0.295_524_224_714_752_87,
];

/// `dqk21`: the 21-point Kronrod rule on `[a, b]` with its embedded 10-point Gauss rule.
fn qk21(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64) -> Panel {
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let dhlgth = hlgth.abs();
    let mut fv1 = [0.0; 10];
    let mut fv2 = [0.0; 10];
    let mut resg = 0.0;
    let fc = f(centr);
    let mut resk = WGK21[10] * fc;
    let mut resabs = resk.abs();
    for j in 1..=5 {
        let jtw = 2 * j; // 1-based Fortran index into xgk/wgk
        let absc = hlgth * XGK21[jtw - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtw - 1] = fval1;
        fv2[jtw - 1] = fval2;
        let fsum = fval1 + fval2;
        resg += WG10[j - 1] * fsum;
        resk += WGK21[jtw - 1] * fsum;
        resabs += WGK21[jtw - 1] * (fval1.abs() + fval2.abs());
    }
    for j in 1..=5 {
        let jtwm1 = 2 * j - 1;
        let absc = hlgth * XGK21[jtwm1 - 1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtwm1 - 1] = fval1;
        fv2[jtwm1 - 1] = fval2;
        let fsum = fval1 + fval2;
        resk += WGK21[jtwm1 - 1] * fsum;
        resabs += WGK21[jtwm1 - 1] * (fval1.abs() + fval2.abs());
    }
    let reskh = resk * 0.5;
    let mut resasc = WGK21[10] * (fc - reskh).abs();
    for j in 0..10 {
        resasc += WGK21[j] * ((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs());
    }
    let result = resk * hlgth;
    let resabs = resabs * dhlgth;
    let resasc = resasc * dhlgth;
    Panel {
        result,
        abserr: kronrod_error(resk, resg, hlgth, resabs, resasc),
        resabs,
        resasc,
    }
}

const XGK15: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_5,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_48,
    0.0,
];
const WGK15: [f64; 8] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_19,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_89,
    0.209_482_141_084_727_82,
];
/// `dqk15i`'s Gauss weights, zero at the Kronrod-only nodes.
const WG15I: [f64; 8] = [
    0.0,
    0.129_484_966_168_869_7,
    0.0,
    0.279_705_391_489_276_64,
    0.0,
    0.381_830_050_505_118_9,
    0.0,
    0.417_959_183_673_469_4,
];

/// `dqk15i`: the 15-point Kronrod rule on `t ∈ [a, b] ⊂ (0, 1]` for the transformed integrand
/// `f(bound + dinf·(1 − t)/t)/t²`, summing the mirrored point too when `inf == 2`.
fn qk15i(f: &mut impl FnMut(f64) -> f64, boun: f64, inf: i32, a: f64, b: f64) -> Panel {
    let dinf = f64::from(inf.min(1));
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let mut transformed = |t: f64| {
        let x = boun + dinf * (1.0 - t) / t;
        let mut value = f(x);
        if inf == 2 {
            value += f(-x);
        }
        (value / t) / t
    };
    let mut fv1 = [0.0; 7];
    let mut fv2 = [0.0; 7];
    let fc = transformed(centr);
    let mut resg = WG15I[7] * fc;
    let mut resk = WGK15[7] * fc;
    let mut resabs = resk.abs();
    for j in 0..7 {
        let absc = hlgth * XGK15[j];
        let fval1 = transformed(centr - absc);
        let fval2 = transformed(centr + absc);
        fv1[j] = fval1;
        fv2[j] = fval2;
        let fsum = fval1 + fval2;
        resg += WG15I[j] * fsum;
        resk += WGK15[j] * fsum;
        resabs += WGK15[j] * (fval1.abs() + fval2.abs());
    }
    let reskh = resk * 0.5;
    let mut resasc = WGK15[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc += WGK15[j] * ((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs());
    }
    let result = resk * hlgth;
    let resasc = resasc * hlgth;
    let resabs = resabs * hlgth;
    Panel {
        result,
        abserr: kronrod_error(resk, resg, hlgth, resabs, resasc),
        resabs,
        resasc,
    }
}

/// `dqpsrt`: keep `iord[nrmax..]` sorted by decreasing error after a bisection, and return the
/// next interval to bisect in `maxerr`/`ermax`. 1-based `elist`/`iord`.
fn qpsrt(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    ermax: &mut f64,
    elist: &[f64],
    iord: &mut [usize],
    nrmax: &mut usize,
) {
    if last <= 2 {
        iord[1] = 1;
        iord[2] = 2;
    } else {
        let errmax = elist[*maxerr];
        if *nrmax != 1 {
            let ido = *nrmax - 1;
            for _ in 1..=ido {
                let isucc = iord[*nrmax - 1];
                if errmax <= elist[isucc] {
                    break;
                }
                iord[*nrmax] = isucc;
                *nrmax -= 1;
            }
        }
        let jupbn = if last > limit / 2 + 2 {
            limit + 3 - last
        } else {
            last
        };
        let errmin = elist[last];
        let jbnd = jupbn - 1;
        let ibeg = *nrmax + 1;
        let mut inserted = false;
        if ibeg <= jbnd {
            for i in ibeg..=jbnd {
                let isucc = iord[i];
                if errmax >= elist[isucc] {
                    iord[i - 1] = *maxerr;
                    let mut k = jbnd;
                    let mut placed = false;
                    for _ in i..=jbnd {
                        let isucc = iord[k];
                        if errmin < elist[isucc] {
                            iord[k + 1] = last;
                            placed = true;
                            break;
                        }
                        iord[k + 1] = isucc;
                        k -= 1;
                    }
                    if !placed {
                        iord[i] = last;
                    }
                    inserted = true;
                    break;
                }
                iord[i - 1] = isucc;
            }
        }
        if !inserted {
            iord[jbnd] = *maxerr;
            iord[jupbn] = last;
        }
    }
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
}

/// Size of the epsilon table (`rlist2(52)` plus the two slots `dqelg` writes past `n`).
const EPSTAB_LEN: usize = 55;

/// `dqelg`: Wynn's epsilon algorithm on the 1-based table `epstab[1..=n]` of partial sums.
/// Returns the extrapolated limit in `result` with an error estimate in `abserr`; `res3la`
/// (1-based) holds the last three results for that estimate. `n` is updated as the table is
/// shortened.
fn qelg(
    n: &mut usize,
    epstab: &mut [f64; EPSTAB_LEN],
    result: &mut f64,
    abserr: &mut f64,
    res3la: &mut [f64; 4],
    nres: &mut usize,
) {
    *nres += 1;
    *abserr = OFLOW;
    *result = epstab[*n];
    if *n >= 3 {
        qelg_table(n, epstab, result, abserr, res3la, *nres);
    }
    // Label 100, reached on every path.
    *abserr = abserr.max(5.0 * EPMACH * result.abs());
}

/// The body of `dqelg` past its `n < 3` exit: extend the epsilon table by one diagonal, keep
/// the best estimate, shorten the table, and estimate the error from the last three results.
fn qelg_table(
    n: &mut usize,
    epstab: &mut [f64; EPSTAB_LEN],
    result: &mut f64,
    abserr: &mut f64,
    res3la: &mut [f64; 4],
    nres: usize,
) {
    const LIMEXP: usize = 50;
    epstab[*n + 2] = epstab[*n];
    let newelm = (*n - 1) / 2;
    epstab[*n] = OFLOW;
    let num = *n;
    let mut k1 = *n;
    for i in 1..=newelm {
        let k2 = k1 - 1;
        let k3 = k1 - 2;
        let mut res = epstab[k1 + 2];
        let e0 = epstab[k3];
        let e1 = epstab[k2];
        let e2 = res;
        let e1abs = e1.abs();
        let delta2 = e2 - e1;
        let err2 = delta2.abs();
        let tol2 = e2.abs().max(e1abs) * EPMACH;
        let delta3 = e1 - e0;
        let err3 = delta3.abs();
        let tol3 = e1abs.max(e0.abs()) * EPMACH;
        if !(err2 > tol2 || err3 > tol3) {
            // e0, e1 and e2 are equal to within machine accuracy: convergence.
            *result = res;
            *abserr = err2 + err3;
            return;
        }
        let e3 = epstab[k1];
        epstab[k1] = e1;
        let delta1 = e1 - e3;
        let err1 = delta1.abs();
        let tol1 = e1abs.max(e3.abs()) * EPMACH;
        if err1 <= tol1 || err2 <= tol2 || err3 <= tol3 {
            *n = i + i - 1;
            break;
        }
        let ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
        let epsinf = (ss * e1).abs();
        // Fortran `IF (EPSINF.GT.0.1D-03) GO TO 30`: a NaN epsinf also stops the table here.
        if epsinf.is_nan() || epsinf <= 1.0e-4 {
            *n = i + i - 1;
            break;
        }
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 -= 2;
        let error = err2 + (res - e2).abs() + err3;
        if error > *abserr {
            continue;
        }
        *abserr = error;
        *result = res;
    }
    if *n == LIMEXP {
        *n = 2 * (LIMEXP / 2) - 1;
    }
    let mut ib = if num.is_multiple_of(2) { 2 } else { 1 };
    let ie = newelm + 1;
    for _ in 1..=ie {
        let ib2 = ib + 2;
        epstab[ib] = epstab[ib2];
        ib = ib2;
    }
    if num != *n {
        let shift = num - *n;
        for i in 1..=*n {
            epstab[i] = epstab[i + shift];
        }
    }
    if nres < 4 {
        res3la[nres] = *result;
        *abserr = OFLOW;
    } else {
        *abserr =
            (*result - res3la[3]).abs() + (*result - res3la[2]).abs() + (*result - res3la[1]).abs();
        res3la[1] = res3la[2];
        res3la[2] = res3la[3];
        res3la[3] = *result;
    }
}

/// 1-based interval lists of length `limit + 1`.
struct Lists {
    alist: Vec<f64>,
    blist: Vec<f64>,
    rlist: Vec<f64>,
    elist: Vec<f64>,
    iord: Vec<usize>,
}

impl Lists {
    fn new(limit: usize) -> Self {
        Self {
            alist: vec![0.0; limit + 2],
            blist: vec![0.0; limit + 2],
            rlist: vec![0.0; limit + 2],
            elist: vec![0.0; limit + 2],
            iord: vec![0; limit + 2],
        }
    }

    fn into_output(self, result: f64, abserr: f64, neval: usize, ier: u8, last: usize) -> Qags {
        let range = 1..=last;
        Qags {
            result,
            abserr,
            neval,
            ier,
            last,
            alist: self.alist[range.clone()].to_vec(),
            blist: self.blist[range.clone()].to_vec(),
            rlist: self.rlist[range.clone()].to_vec(),
            elist: self.elist[range.clone()].to_vec(),
            iord: self.iord[range]
                .iter()
                .map(|&i| i.saturating_sub(1))
                .collect(),
        }
    }
}

/// The adaptive loop shared by `dqagse` and `dqagie` (they differ only in the panel rule, the
/// interval, and the evaluation count per panel).
fn qags_driver(
    rule: &mut impl FnMut(f64, f64) -> Panel,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    evals_per_panel: usize,
) -> Qags {
    let mut lists = Lists::new(limit);
    lists.alist[1] = a;
    lists.blist[1] = b;
    if invalid_tolerances(epsabs, epsrel) {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let mut ier: u8 = 0;
    let mut ierro = 0;
    let first = rule(a, b);
    let mut panels = 1;
    let mut result = first.result;
    let mut abserr = first.abserr;
    // dqk21(f,a,b,result,abserr,defabs,resabs): defabs = ∫|f|, resabs = ∫|f − mean|.
    let defabs = first.resabs;
    let resabs = first.resasc;
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    let mut last = 1;
    lists.rlist[1] = result;
    lists.elist[1] = abserr;
    lists.iord[1] = 1;
    if abserr <= 100.0 * EPMACH * defabs && abserr > errbnd {
        ier = 2;
    }
    if limit == 1 {
        ier = 1;
    }
    if ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0.0 {
        return lists.into_output(result, abserr, evals_per_panel, ier, last);
    }

    let mut rlist2 = [0.0; EPSTAB_LEN];
    let mut res3la = [0.0; 4];
    rlist2[1] = result;
    let mut errmax = abserr;
    let mut maxerr = 1;
    let mut area = result;
    let mut errsum = abserr;
    abserr = OFLOW;
    let mut nrmax = 1;
    let mut nres = 0;
    let mut numrl2 = 2;
    let mut ktmin = 0;
    let mut extrap = false;
    let mut noext = false;
    let (mut iroff1, mut iroff2, mut iroff3) = (0, 0, 0);
    let ksgn = if dres >= (1.0 - 50.0 * EPMACH) * defabs {
        1
    } else {
        -1
    };
    let (mut small, mut erlarg, mut ertest, mut correc) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut reseps = 0.0;
    let mut abseps = 0.0;
    let mut sum_up = false; // `go to 115`: the result is the plain sum of the panels

    for current in 2..=limit {
        last = current;
        let a1 = lists.alist[maxerr];
        let b1 = 0.5 * (lists.alist[maxerr] + lists.blist[maxerr]);
        let a2 = b1;
        let b2 = lists.blist[maxerr];
        let erlast = errmax;
        let p1 = rule(a1, b1);
        let p2 = rule(a2, b2);
        panels += 2;
        let (area1, error1, defab1) = (p1.result, p1.abserr, p1.resasc);
        let (area2, error2, defab2) = (p2.result, p2.abserr, p2.resasc);
        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - lists.rlist[maxerr];
        if !(defab1 == error1 || defab2 == error2) {
            if !((lists.rlist[maxerr] - area12).abs() > 1.0e-5 * area12.abs()
                || erro12 < 0.99 * errmax)
            {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if last > 10 && erro12 > errmax {
                iroff3 += 1;
            }
        }
        lists.rlist[maxerr] = area1;
        lists.rlist[last] = area2;
        errbnd = epsabs.max(epsrel * area.abs());
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierro = 3;
        }
        if last == limit {
            ier = 1;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }
        if error2 > error1 {
            lists.alist[maxerr] = a2;
            lists.alist[last] = a1;
            lists.blist[last] = b1;
            lists.rlist[maxerr] = area2;
            lists.rlist[last] = area1;
            lists.elist[maxerr] = error2;
            lists.elist[last] = error1;
        } else {
            lists.alist[last] = a2;
            lists.blist[maxerr] = b1;
            lists.blist[last] = b2;
            lists.elist[maxerr] = error1;
            lists.elist[last] = error2;
        }
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &lists.elist,
            &mut lists.iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            sum_up = true;
            break;
        }
        if ier != 0 {
            break;
        }
        if last == 2 {
            small = (b - a).abs() * 0.375;
            erlarg = errsum;
            ertest = errbnd;
            rlist2[2] = area;
            continue;
        }
        if noext {
            continue;
        }
        erlarg -= erlast;
        if (b1 - a1).abs() > small {
            erlarg += erro12;
        }
        if !extrap {
            // Test whether the interval to be bisected next is the smallest interval.
            if (lists.blist[maxerr] - lists.alist[maxerr]).abs() > small {
                continue;
            }
            extrap = true;
            nrmax = 2;
        }
        if ierro != 3 && erlarg > ertest {
            // Bisect the large intervals first, before extrapolating again.
            let id = nrmax;
            let jupbnd = if last > 2 + limit / 2 {
                limit + 3 - last
            } else {
                last
            };
            let mut large_left = false;
            for _ in id..=jupbnd {
                maxerr = lists.iord[nrmax];
                errmax = lists.elist[maxerr];
                if (lists.blist[maxerr] - lists.alist[maxerr]).abs() > small {
                    large_left = true;
                    break;
                }
                nrmax += 1;
            }
            if large_left {
                continue;
            }
        }
        // Perform extrapolation.
        numrl2 += 1;
        rlist2[numrl2] = area;
        qelg(
            &mut numrl2,
            &mut rlist2,
            &mut reseps,
            &mut abseps,
            &mut res3la,
            &mut nres,
        );
        ktmin += 1;
        if ktmin > 5 && abserr < 1.0e-3 * errsum {
            ier = 5;
        }
        if abseps < abserr {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            correc = erlarg;
            ertest = epsabs.max(epsrel * reseps.abs());
            if abserr <= ertest {
                break;
            }
        }
        if numrl2 == 1 {
            noext = true;
        }
        if ier == 5 {
            break;
        }
        maxerr = lists.iord[1];
        errmax = lists.elist[maxerr];
        nrmax = 1;
        extrap = false;
        small *= 0.5;
        erlarg = errsum;
    }

    let neval = evals_per_panel * panels;
    let (result, abserr, ier) = finish(
        sum_up, &lists, last, result, abserr, area, errsum, ier, ierro, correc, ksgn, defabs,
    );
    lists.into_output(result, abserr, neval, ier, last)
}

/// Labels 100–130 of `dqagse`/`dqagie` (170–210 of `dqagpe`): choose between the extrapolated
/// result and the plain sum of the panels, test for divergence, then map `ier`.
#[allow(clippy::too_many_arguments)]
fn finish(
    sum_up: bool,
    lists: &Lists,
    last: usize,
    mut result: f64,
    mut abserr: f64,
    area: f64,
    errsum: f64,
    mut ier: u8,
    ierro: u8,
    correc: f64,
    ksgn: i32,
    defabs: f64,
) -> (f64, f64, u8) {
    #[derive(PartialEq)]
    enum Tail {
        Sum,
        Check,
        Done,
    }
    let mut tail = if sum_up { Tail::Sum } else { Tail::Check };
    if tail == Tail::Check {
        if abserr == OFLOW {
            tail = Tail::Sum;
        } else if ier + ierro != 0 {
            if ierro == 3 {
                abserr += correc;
            }
            if ier == 0 {
                ier = 3;
            }
            if result != 0.0 && area != 0.0 {
                if abserr / result.abs() > errsum / area.abs() {
                    tail = Tail::Sum;
                }
            } else if abserr > errsum {
                tail = Tail::Sum;
            } else if area == 0.0 {
                tail = Tail::Done;
            }
        }
        if tail == Tail::Check {
            // Label 110: test on divergence.
            if !(ksgn == -1 && result.abs().max(area.abs()) <= defabs * 0.01)
                && (0.01 > result / area || result / area > 100.0 || errsum > area.abs())
            {
                ier = 6;
            }
            tail = Tail::Done;
        }
    }
    if tail == Tail::Sum {
        result = lists.rlist[1..=last].iter().sum();
        abserr = errsum;
    }
    if ier > 2 {
        ier -= 1;
    }
    (result, abserr, ier)
}

/// `dqagse`: `∫_a^b f` for finite `a`, `b`.
pub(crate) fn qagse(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Qags {
    qags_driver(
        &mut |lo, hi| qk21(f, lo, hi),
        a,
        b,
        epsabs,
        epsrel,
        limit,
        21,
    )
}

/// `dqagie`: `inf = 1` integrates over `[bound, ∞)`, `inf = -1` over `(-∞, bound]`, `inf = 2`
/// over `(-∞, ∞)` (`bound` ignored).
pub(crate) fn qagie(
    f: &mut impl FnMut(f64) -> f64,
    bound: f64,
    inf: i32,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Qags {
    let boun = if inf == 2 { 0.0 } else { bound };
    let evals = if inf == 2 { 30 } else { 15 };
    qags_driver(
        &mut |lo, hi| qk15i(f, boun, inf, lo, hi),
        0.0,
        1.0,
        epsabs,
        epsrel,
        limit,
        evals,
    )
}

/// `dqagpe`: `∫_a^b f` with the interior breakpoints `points` (sorted, distinct, strictly inside
/// `(min(a, b), max(a, b))`) as the initial panel edges.
pub(crate) fn qagpe(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    points: &[f64],
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Qags {
    let npts = points.len();
    let npts2 = npts + 2;
    let mut lists = Lists::new(limit.max(npts2));
    let mut level = vec![0_usize; limit.max(npts2) + 2];
    lists.alist[1] = a;
    lists.blist[1] = b;
    if limit <= npts || invalid_tolerances(epsabs, epsrel) {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let sign = if a > b { -1.0 } else { 1.0 };
    let mut pts = Vec::with_capacity(npts2);
    pts.push(a.min(b));
    pts.extend_from_slice(points);
    pts.push(a.max(b));
    pts.sort_by(f64::total_cmp);
    if pts[0] != a.min(b) || pts[npts2 - 1] != a.max(b) {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let nint = npts + 1;
    let mut ier: u8 = 0;
    let mut result = 0.0;
    let mut abserr = 0.0;
    let mut resabs = 0.0;
    let mut ndin = vec![false; nint + 1];
    let mut a1 = pts[0];
    for i in 1..=nint {
        let b1 = pts[i];
        let panel = qk21(f, a1, b1);
        abserr += panel.abserr;
        result += panel.result;
        ndin[i] = panel.abserr == panel.resasc && panel.abserr != 0.0;
        resabs += panel.resabs;
        level[i] = 0;
        lists.elist[i] = panel.abserr;
        lists.alist[i] = a1;
        lists.blist[i] = b1;
        lists.rlist[i] = panel.result;
        lists.iord[i] = i;
        a1 = b1;
    }
    let mut errsum = 0.0;
    for i in 1..=nint {
        if ndin[i] {
            lists.elist[i] = abserr;
        }
        errsum += lists.elist[i];
    }
    let mut last = nint;
    let mut neval = 21 * nint;
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    if abserr <= 100.0 * EPMACH * resabs && abserr > errbnd {
        ier = 2;
    }
    if nint > 1 {
        for i in 1..=npts {
            let jlow = i + 1;
            let mut ind1 = lists.iord[i];
            let mut k = i;
            for j in jlow..=nint {
                let ind2 = lists.iord[j];
                if lists.elist[ind1] > lists.elist[ind2] {
                    continue;
                }
                ind1 = ind2;
                k = j;
            }
            if ind1 != lists.iord[i] {
                lists.iord[k] = lists.iord[i];
                lists.iord[i] = ind1;
            }
        }
        if limit < npts2 {
            ier = 1;
        }
    }
    if ier != 0 || abserr <= errbnd {
        let ier = if ier > 2 { ier - 1 } else { ier };
        return lists.into_output(result * sign, abserr, neval, ier, last);
    }

    let mut rlist2 = [0.0; EPSTAB_LEN];
    let mut res3la = [0.0; 4];
    rlist2[1] = result;
    let mut maxerr = lists.iord[1];
    let mut errmax = lists.elist[maxerr];
    let mut area = result;
    let mut nrmax = 1;
    let mut nres = 0;
    let mut numrl2 = 1;
    let mut ktmin = 0;
    let mut extrap = false;
    let mut noext = false;
    let mut erlarg = errsum;
    let mut ertest = errbnd;
    let mut levmax = 1;
    let (mut iroff1, mut iroff2, mut iroff3) = (0, 0, 0);
    let mut ierro = 0;
    abserr = OFLOW;
    let ksgn = if dres >= (1.0 - 50.0 * EPMACH) * resabs {
        1
    } else {
        -1
    };
    let mut correc = 0.0;
    let mut reseps = 0.0;
    let mut abseps = 0.0;
    let mut sum_up = false;

    for current in npts2..=limit {
        last = current;
        let levcur = level[maxerr] + 1;
        let a1 = lists.alist[maxerr];
        let b1 = 0.5 * (lists.alist[maxerr] + lists.blist[maxerr]);
        let a2 = b1;
        let b2 = lists.blist[maxerr];
        let erlast = errmax;
        let p1 = qk21(f, a1, b1);
        let p2 = qk21(f, a2, b2);
        neval += 42;
        let (area1, error1, defab1) = (p1.result, p1.abserr, p1.resasc);
        let (area2, error2, defab2) = (p2.result, p2.abserr, p2.resasc);
        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - lists.rlist[maxerr];
        if !(defab1 == error1 || defab2 == error2) {
            if !((lists.rlist[maxerr] - area12).abs() > 1.0e-5 * area12.abs()
                || erro12 < 0.99 * errmax)
            {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if last > 10 && erro12 > errmax {
                iroff3 += 1;
            }
        }
        level[maxerr] = levcur;
        level[last] = levcur;
        lists.rlist[maxerr] = area1;
        lists.rlist[last] = area2;
        errbnd = epsabs.max(epsrel * area.abs());
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierro = 3;
        }
        if last == limit {
            ier = 1;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }
        if error2 > error1 {
            lists.alist[maxerr] = a2;
            lists.alist[last] = a1;
            lists.blist[last] = b1;
            lists.rlist[maxerr] = area2;
            lists.rlist[last] = area1;
            lists.elist[maxerr] = error2;
            lists.elist[last] = error1;
        } else {
            lists.alist[last] = a2;
            lists.blist[maxerr] = b1;
            lists.blist[last] = b2;
            lists.elist[maxerr] = error1;
            lists.elist[last] = error2;
        }
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &lists.elist,
            &mut lists.iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            sum_up = true;
            break;
        }
        if ier != 0 {
            break;
        }
        if noext {
            continue;
        }
        erlarg -= erlast;
        if levcur < levmax {
            erlarg += erro12;
        }
        if !extrap {
            // Test whether the interval to be bisected next is the smallest interval.
            if level[maxerr] < levmax {
                continue;
            }
            extrap = true;
            nrmax = 2;
        }
        let mut skip_extrapolation = false;
        if ierro != 3 && erlarg > ertest {
            let id = nrmax;
            let jupbnd = if last > 2 + limit / 2 {
                limit + 3 - last
            } else {
                last
            };
            let mut large_left = false;
            for _ in id..=jupbnd {
                maxerr = lists.iord[nrmax];
                errmax = lists.elist[maxerr];
                if level[maxerr] < levmax {
                    large_left = true;
                    break;
                }
                nrmax += 1;
            }
            if large_left {
                continue;
            }
        }
        numrl2 += 1;
        rlist2[numrl2] = area;
        if numrl2 <= 2 {
            skip_extrapolation = true;
        }
        if !skip_extrapolation {
            qelg(
                &mut numrl2,
                &mut rlist2,
                &mut reseps,
                &mut abseps,
                &mut res3la,
                &mut nres,
            );
            ktmin += 1;
            if ktmin > 5 && abserr < 1.0e-3 * errsum {
                ier = 5;
            }
            if abseps < abserr {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = epsabs.max(epsrel * reseps.abs());
                if abserr < ertest {
                    break;
                }
            }
            if numrl2 == 1 {
                noext = true;
            }
            if ier >= 5 {
                break;
            }
        }
        maxerr = lists.iord[1];
        errmax = lists.elist[maxerr];
        nrmax = 1;
        extrap = false;
        levmax += 1;
        erlarg = errsum;
    }

    let (result, abserr, ier) = finish(
        sum_up, &lists, last, result, abserr, area, errsum, ier, ierro, correc, ksgn, resabs,
    );
    lists.into_output(result * sign, abserr, neval, ier, last)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Weighted integrands: `quad(..., weight=...)`. QAWCE (Cauchy principal value), QAWSE
// (algebraic–logarithmic endpoint weights), QAWOE / QAWFE (cos/sin weights, finite / [a, ∞)).
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// `cos(kπ/24)`, k = 1..11: the Clenshaw–Curtis abscissae of the modified 25-point rules
/// (`x` in `dqc25c`/`dqc25s`/`dqc25f`), 1-based via index `k - 1`.
const CC_X: [f64; 11] = [
    0.991_444_861_373_810_4,
    0.965_925_826_289_068_3,
    0.923_879_532_511_286_7,
    0.866_025_403_784_438_6,
    0.793_353_340_291_235_2,
    std::f64::consts::FRAC_1_SQRT_2,
    0.608_761_429_008_720_7,
    0.5,
    0.382_683_432_365_089_8,
    0.258_819_045_102_520_74,
    0.130_526_192_220_051_6,
];

/// `dqk15w`'s own (16-digit) Kronrod data.
const K15W_X: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_3,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
const K15W_WGK: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_54,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
const K15W_WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// `dqk15w`: the 15-point Kronrod rule for `f(x)·w(x)` on `[a, b]`.
fn qk15w(f: &mut impl FnMut(f64) -> f64, w: &dyn Fn(f64) -> f64, a: f64, b: f64) -> Panel {
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let dhlgth = hlgth.abs();
    let mut fv1 = [0.0; 7];
    let mut fv2 = [0.0; 7];
    let fc = f(centr) * w(centr);
    let mut resg = K15W_WG[3] * fc;
    let mut resk = K15W_WGK[7] * fc;
    let mut resabs = resk.abs();
    for j in 1..=3 {
        let jtw = j * 2;
        let absc = hlgth * K15W_X[jtw - 1];
        let (absc1, absc2) = (centr - absc, centr + absc);
        let fval1 = f(absc1) * w(absc1);
        let fval2 = f(absc2) * w(absc2);
        fv1[jtw - 1] = fval1;
        fv2[jtw - 1] = fval2;
        let fsum = fval1 + fval2;
        resg += K15W_WG[j - 1] * fsum;
        resk += K15W_WGK[jtw - 1] * fsum;
        resabs += K15W_WGK[jtw - 1] * (fval1.abs() + fval2.abs());
    }
    for j in 1..=4 {
        let jtwm1 = j * 2 - 1;
        let absc = hlgth * K15W_X[jtwm1 - 1];
        let (absc1, absc2) = (centr - absc, centr + absc);
        let fval1 = f(absc1) * w(absc1);
        let fval2 = f(absc2) * w(absc2);
        fv1[jtwm1 - 1] = fval1;
        fv2[jtwm1 - 1] = fval2;
        let fsum = fval1 + fval2;
        resk += K15W_WGK[jtwm1 - 1] * fsum;
        resabs += K15W_WGK[jtwm1 - 1] * (fval1.abs() + fval2.abs());
    }
    let reskh = resk * 0.5;
    let mut resasc = K15W_WGK[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc += K15W_WGK[j] * ((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs());
    }
    let result = resk * hlgth;
    let resabs = resabs * dhlgth;
    let resasc = resasc * dhlgth;
    Panel {
        result,
        abserr: kronrod_error(resk, resg, hlgth, resabs, resasc),
        resabs,
        resasc,
    }
}

/// `dqcheb`: Chebyshev coefficients of degree 12 and 24 from the 25 function values `fval`
/// (1-based, overwritten) at the abscissae `CC_X`. `cheb12[1..=13]`, `cheb24[1..=25]`.
fn qcheb(fval: &mut [f64; 26], cheb12: &mut [f64; 14], cheb24: &mut [f64; 26]) {
    let x = |i: usize| CC_X[i - 1];
    let mut v = [0.0; 13];
    for i in 1..=12 {
        let j = 26 - i;
        v[i] = fval[i] - fval[j];
        fval[i] += fval[j];
    }
    let mut alam1 = v[1] - v[9];
    let mut alam2 = x(6) * (v[3] - v[7] - v[11]);
    cheb12[4] = alam1 + alam2;
    cheb12[10] = alam1 - alam2;
    alam1 = v[2] - v[8] - v[10];
    alam2 = v[4] - v[6] - v[12];
    let mut alam = x(3) * alam1 + x(9) * alam2;
    cheb24[4] = cheb12[4] + alam;
    cheb24[22] = cheb12[4] - alam;
    alam = x(9) * alam1 - x(3) * alam2;
    cheb24[10] = cheb12[10] + alam;
    cheb24[16] = cheb12[10] - alam;
    let part1 = x(4) * v[5];
    let part2 = x(8) * v[9];
    let part3 = x(6) * v[7];
    alam1 = v[1] + part1 + part2;
    alam2 = x(2) * v[3] + part3 + x(10) * v[11];
    cheb12[2] = alam1 + alam2;
    cheb12[12] = alam1 - alam2;
    alam = x(1) * v[2] + x(3) * v[4] + x(5) * v[6] + x(7) * v[8] + x(9) * v[10] + x(11) * v[12];
    cheb24[2] = cheb12[2] + alam;
    cheb24[24] = cheb12[2] - alam;
    alam = x(11) * v[2] - x(9) * v[4] + x(7) * v[6] - x(5) * v[8] + x(3) * v[10] - x(1) * v[12];
    cheb24[12] = cheb12[12] + alam;
    cheb24[14] = cheb12[12] - alam;
    alam1 = v[1] - part1 + part2;
    alam2 = x(10) * v[3] - part3 + x(2) * v[11];
    cheb12[6] = alam1 + alam2;
    cheb12[8] = alam1 - alam2;
    alam = x(5) * v[2] - x(9) * v[4] - x(1) * v[6] - x(11) * v[8] + x(3) * v[10] + x(7) * v[12];
    cheb24[6] = cheb12[6] + alam;
    cheb24[20] = cheb12[6] - alam;
    alam = x(7) * v[2] - x(3) * v[4] - x(11) * v[6] + x(1) * v[8] - x(9) * v[10] - x(5) * v[12];
    cheb24[8] = cheb12[8] + alam;
    cheb24[18] = cheb12[8] - alam;
    for i in 1..=6 {
        let j = 14 - i;
        v[i] = fval[i] - fval[j];
        fval[i] += fval[j];
    }
    alam1 = v[1] + x(8) * v[5];
    alam2 = x(4) * v[3];
    cheb12[3] = alam1 + alam2;
    cheb12[11] = alam1 - alam2;
    cheb12[7] = v[1] - v[5];
    alam = x(2) * v[2] + x(6) * v[4] + x(10) * v[6];
    cheb24[3] = cheb12[3] + alam;
    cheb24[23] = cheb12[3] - alam;
    alam = x(6) * (v[2] - v[4] - v[6]);
    cheb24[7] = cheb12[7] + alam;
    cheb24[19] = cheb12[7] - alam;
    alam = x(10) * v[2] - x(6) * v[4] + x(2) * v[6];
    cheb24[11] = cheb12[11] + alam;
    cheb24[15] = cheb12[11] - alam;
    for i in 1..=3 {
        let j = 8 - i;
        v[i] = fval[i] - fval[j];
        fval[i] += fval[j];
    }
    cheb12[5] = v[1] + x(8) * v[3];
    cheb12[9] = fval[1] - x(8) * fval[3];
    alam = x(4) * v[2];
    cheb24[5] = cheb12[5] + alam;
    cheb24[21] = cheb12[5] - alam;
    alam = x(8) * fval[2] - fval[4];
    cheb24[9] = cheb12[9] + alam;
    cheb24[17] = cheb12[9] - alam;
    cheb12[1] = fval[1] + fval[3];
    alam = fval[2] + fval[4];
    cheb24[1] = cheb12[1] + alam;
    cheb24[25] = cheb12[1] - alam;
    cheb12[13] = v[1] - v[3];
    cheb24[13] = cheb12[13];
    alam = 1.0 / 6.0;
    for value in &mut cheb12[2..=12] {
        *value *= alam;
    }
    alam *= 0.5;
    cheb12[1] *= alam;
    cheb12[13] *= alam;
    for value in &mut cheb24[2..=24] {
        *value *= alam;
    }
    cheb24[1] *= 0.5 * alam;
    cheb24[25] *= 0.5 * alam;
}

/// Sample `g` at the 25 Clenshaw–Curtis points of `[centr − hlgth, centr + hlgth]` into the
/// 1-based `fval`, halving the two endpoint values, as `dqc25c`/`dqc25f` do.
fn cc_samples(g: &mut impl FnMut(f64) -> f64, centr: f64, hlgth: f64) -> [f64; 26] {
    let mut fval = [0.0; 26];
    fval[1] = 0.5 * g(hlgth + centr);
    fval[13] = g(centr);
    fval[25] = 0.5 * g(centr - hlgth);
    for i in 2..=12 {
        let u = hlgth * CC_X[i - 2];
        let isym = 26 - i;
        fval[i] = g(u + centr);
        fval[isym] = g(centr - u);
    }
    fval
}

/// `dqc25c`: `∫ f(x)/(x − c)` on `[a, b]` by the modified 25-point Clenshaw–Curtis rule when
/// `c` is near the interval, else the 15-point Kronrod rule (then `krul` is decremented, and
/// restored if `resasc == abserr`). Returns `(result, abserr, neval)`.
fn qc25c(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    c: f64,
    krul: &mut i32,
) -> (f64, f64, usize) {
    let cc = (2.0 * c - b - a) / (b - a);
    if cc.abs() >= 1.1 {
        *krul -= 1;
        let panel = qk15w(f, &|x| 1.0 / (x - c), a, b);
        if panel.resasc == panel.abserr {
            *krul += 1;
        }
        return (panel.result, panel.abserr, 15);
    }
    let hlgth = 0.5 * (b - a);
    let centr = 0.5 * (b + a);
    let mut fval = cc_samples(f, centr, hlgth);
    let mut cheb12 = [0.0; 14];
    let mut cheb24 = [0.0; 26];
    qcheb(&mut fval, &mut cheb12, &mut cheb24);
    let mut amom0 = ((1.0 - cc) / (1.0 + cc)).abs().ln();
    let mut amom1 = 2.0 + cc * amom0;
    let mut res12 = cheb12[1] * amom0 + cheb12[2] * amom1;
    let mut res24 = cheb24[1] * amom0 + cheb24[2] * amom1;
    for k in 3_usize..=25 {
        let mut amom2 = 2.0 * cc * amom1 - amom0;
        let ak22 = ((k - 2) * (k - 2)) as f64;
        if k.is_multiple_of(2) {
            amom2 -= 4.0 / (ak22 - 1.0);
        }
        if k <= 13 {
            res12 += cheb12[k] * amom2;
        }
        res24 += cheb24[k] * amom2;
        amom0 = amom1;
        amom1 = amom2;
    }
    (res24, (res24 - res12).abs(), 25)
}

/// `dqawce`: the Cauchy principal value of `∫_a^b f(x)/(x − c) dx`, `c` strictly inside.
pub(crate) fn qawce(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    c: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Qags {
    let mut lists = Lists::new(limit);
    lists.alist[1] = a;
    lists.blist[1] = b;
    if c == a || c == b || invalid_tolerances(epsabs, epsrel) {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let (aa, bb) = if a <= b { (a, b) } else { (b, a) };
    let mut ier = 0;
    let mut krule = 1;
    let (mut result, mut abserr, mut neval) = qc25c(f, aa, bb, c, &mut krule);
    let mut last = 1;
    lists.rlist[1] = result;
    lists.elist[1] = abserr;
    lists.iord[1] = 1;
    let mut errbnd = epsabs.max(epsrel * result.abs());
    if limit == 1 {
        ier = 1;
    }
    if !(abserr < (0.01 * result.abs()).min(errbnd) || ier == 1) {
        lists.alist[1] = aa;
        lists.blist[1] = bb;
        let mut errmax = abserr;
        let mut maxerr = 1;
        let mut area = result;
        let mut errsum = abserr;
        let mut nrmax = 1;
        let (mut iroff1, mut iroff2) = (0, 0);
        for current in 2..=limit {
            last = current;
            let a1 = lists.alist[maxerr];
            let mut b1 = 0.5 * (lists.alist[maxerr] + lists.blist[maxerr]);
            let b2 = lists.blist[maxerr];
            if c <= b1 && c > a1 {
                b1 = 0.5 * (c + b2);
            }
            if c > b1 && c < b2 {
                b1 = 0.5 * (a1 + c);
            }
            let a2 = b1;
            krule = 2;
            let (area1, error1, nev1) = qc25c(f, a1, b1, c, &mut krule);
            let (area2, error2, nev2) = qc25c(f, a2, b2, c, &mut krule);
            neval += nev1 + nev2;
            let area12 = area1 + area2;
            let erro12 = error1 + error2;
            errsum = errsum + erro12 - errmax;
            area = area + area12 - lists.rlist[maxerr];
            if (lists.rlist[maxerr] - area12).abs() < 1.0e-5 * area12.abs()
                && erro12 >= 0.99 * errmax
                && krule == 0
            {
                iroff1 += 1;
            }
            if last > 10 && erro12 > errmax && krule == 0 {
                iroff2 += 1;
            }
            lists.rlist[maxerr] = area1;
            lists.rlist[last] = area2;
            errbnd = epsabs.max(epsrel * area.abs());
            if errsum > errbnd {
                if iroff1 >= 6 && iroff2 > 20 {
                    ier = 2;
                }
                if last == limit {
                    ier = 1;
                }
                if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
                    ier = 3;
                }
            }
            if error2 > error1 {
                lists.alist[maxerr] = a2;
                lists.alist[last] = a1;
                lists.blist[last] = b1;
                lists.rlist[maxerr] = area2;
                lists.rlist[last] = area1;
                lists.elist[maxerr] = error2;
                lists.elist[last] = error1;
            } else {
                lists.alist[last] = a2;
                lists.blist[maxerr] = b1;
                lists.blist[last] = b2;
                lists.elist[maxerr] = error1;
                lists.elist[last] = error2;
            }
            qpsrt(
                limit,
                last,
                &mut maxerr,
                &mut errmax,
                &lists.elist,
                &mut lists.iord,
                &mut nrmax,
            );
            if ier != 0 || errsum <= errbnd {
                break;
            }
        }
        result = lists.rlist[1..=last].iter().sum();
        abserr = errsum;
    }
    if aa == b {
        result = -result;
    }
    lists.into_output(result, abserr, neval, ier, last)
}

/// `dqmomo`'s modified Chebyshev moments for the algebraic–logarithmic weights (1-based).
struct AlgMoments {
    ri: [f64; 26],
    rj: [f64; 26],
    rg: [f64; 26],
    rh: [f64; 26],
}

/// `dqmomo`: the moments of `(1+x)^alfa`, `(1−x)^beta` and, for `integr` 2–4, their products
/// with `log((1+x)/2)` / `log((1−x)/2)` against the Chebyshev polynomials on `[-1, 1]`.
fn qmomo(alfa: f64, beta: f64, integr: u8) -> AlgMoments {
    let mut m = AlgMoments {
        ri: [0.0; 26],
        rj: [0.0; 26],
        rg: [0.0; 26],
        rh: [0.0; 26],
    };
    let alfp1 = alfa + 1.0;
    let betp1 = beta + 1.0;
    let alfp2 = alfa + 2.0;
    let betp2 = beta + 2.0;
    let ralf = 2.0_f64.powf(alfp1);
    let rbet = 2.0_f64.powf(betp1);
    m.ri[1] = ralf / alfp1;
    m.rj[1] = rbet / betp1;
    m.ri[2] = m.ri[1] * alfa / alfp2;
    m.rj[2] = m.rj[1] * beta / betp2;
    let mut an = 2.0;
    let mut anm1 = 1.0;
    for i in 3..=25 {
        m.ri[i] = -(ralf + an * (an - alfp2) * m.ri[i - 1]) / (anm1 * (an + alfp1));
        m.rj[i] = -(rbet + an * (an - betp2) * m.rj[i - 1]) / (anm1 * (an + betp1));
        anm1 = an;
        an += 1.0;
    }
    if integr != 1 {
        if integr != 3 {
            m.rg[1] = -m.ri[1] / alfp1;
            m.rg[2] = -(ralf + ralf) / (alfp2 * alfp2) - m.rg[1];
            let (mut an, mut anm1) = (2.0, 1.0);
            let mut im1 = 2;
            for i in 3..=25 {
                m.rg[i] = -(an * (an - alfp2) * m.rg[im1] - an * m.ri[im1] + anm1 * m.ri[i])
                    / (anm1 * (an + alfp1));
                anm1 = an;
                an += 1.0;
                im1 = i;
            }
        }
        if integr != 2 {
            m.rh[1] = -m.rj[1] / betp1;
            m.rh[2] = -(rbet + rbet) / (betp2 * betp2) - m.rh[1];
            let (mut an, mut anm1) = (2.0, 1.0);
            let mut im1 = 2;
            for i in 3..=25 {
                m.rh[i] = -(an * (an - betp2) * m.rh[im1] - an * m.rj[im1] + anm1 * m.rj[i])
                    / (anm1 * (an + betp1));
                anm1 = an;
                an += 1.0;
                im1 = i;
            }
            for i in (2..=25).step_by(2) {
                m.rh[i] = -m.rh[i];
            }
        }
    }
    for i in (2..=25).step_by(2) {
        m.rj[i] = -m.rj[i];
    }
    m
}

/// `dqwgts`: `(x−a)^alfa (b−x)^beta` times `log(x−a)`, `log(b−x)` or both for `integr` 2, 3, 4.
fn qwgts(x: f64, a: f64, b: f64, alfa: f64, beta: f64, integr: u8) -> f64 {
    let xma = x - a;
    let bmx = b - x;
    let w = xma.powf(alfa) * bmx.powf(beta);
    match integr {
        2 => w * xma.ln(),
        3 => w * bmx.ln(),
        4 => w * xma.ln() * bmx.ln(),
        _ => w,
    }
}

/// `res12`/`res24` of a modified Clenshaw–Curtis rule: `Σ cheb·moment` over 13 / 25 terms.
fn cc_moment_sums(cheb12: &[f64; 14], cheb24: &[f64; 26], moment: &[f64; 26]) -> (f64, f64) {
    let mut res12 = 0.0;
    let mut res24 = 0.0;
    for i in 1..=13 {
        res12 += cheb12[i] * moment[i];
        res24 += cheb24[i] * moment[i];
    }
    for i in 14..=25 {
        res24 += cheb24[i] * moment[i];
    }
    (res12, res24)
}

/// `dqc25s`: `∫_{bl}^{br} f·w` for the algebraic–logarithmic weight on `[a, b]`; the modified
/// Clenshaw–Curtis rule on a subinterval touching a singular endpoint, 15-point Kronrod
/// elsewhere. Returns `(result, abserr, resasc, nev)`.
#[allow(clippy::too_many_arguments)]
fn qc25s(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    bl: f64,
    br: f64,
    alfa: f64,
    beta: f64,
    moments: &AlgMoments,
    integr: u8,
) -> (f64, f64, f64, usize) {
    let left_singular = bl == a && (alfa != 0.0 || integr == 2 || integr == 4);
    let right_singular = br == b && (beta != 0.0 || integr == 3 || integr == 4);
    if !left_singular && !right_singular {
        let panel = qk15w(f, &|x| qwgts(x, a, b, alfa, beta, integr), bl, br);
        return (panel.result, panel.abserr, panel.resasc, 15);
    }
    let hlgth = 0.5 * (br - bl);
    let centr = 0.5 * (br + bl);
    let mut result = 0.0;
    let mut abserr = 0.0;
    let res12;
    let res24;
    let factor;
    if left_singular {
        // Label 10: the weight (x−a)^alfa is in the moments; (b−x)^beta stays in f.
        let fix = b - centr;
        let mut fval = [0.0; 26];
        fval[1] = 0.5 * f(hlgth + centr) * (fix - hlgth).powf(beta);
        fval[13] = f(centr) * fix.powf(beta);
        fval[25] = 0.5 * f(centr - hlgth) * (fix + hlgth).powf(beta);
        for i in 2..=12 {
            let u = hlgth * CC_X[i - 2];
            let isym = 26 - i;
            fval[i] = f(u + centr) * (fix - u).powf(beta);
            fval[isym] = f(centr - u) * (fix + u).powf(beta);
        }
        factor = hlgth.powf(alfa + 1.0);
        if integr > 2 {
            // Label 70: log(b−x) is also carried by f.
            fval[1] *= (fix - hlgth).ln();
            fval[13] *= fix.ln();
            fval[25] *= (fix + hlgth).ln();
            for i in 2..=12 {
                let u = hlgth * CC_X[i - 2];
                let isym = 26 - i;
                fval[i] *= (fix - u).ln();
                fval[isym] *= (fix + u).ln();
            }
        }
        let mut cheb12 = [0.0; 14];
        let mut cheb24 = [0.0; 26];
        qcheb(&mut fval, &mut cheb12, &mut cheb24);
        let (r12, r24) = cc_moment_sums(&cheb12, &cheb24, &moments.ri);
        if integr == 1 || integr == 3 {
            res12 = r12;
            res24 = r24;
        } else {
            let dc = (br - bl).ln();
            result = r24 * dc;
            abserr = ((r24 - r12) * dc).abs();
            let mut s12 = 0.0;
            let mut s24 = 0.0;
            for i in 1..=13 {
                s12 += cheb12[i] * moments.rg[i];
                // netlib's loop 50 reads `res24 = res12+cheb24(i)*rg(i)`, a typo for `res24+`;
                // SciPy's QUADPACK has the corrected statement (measured: the typo is off by
                // ~3e-15 and changes abserr on alg-loga cases), and so does this port.
                s24 += cheb24[i] * moments.rg[i];
            }
            for i in 14..=25 {
                s24 += cheb24[i] * moments.rg[i];
            }
            res12 = s12;
            res24 = s24;
        }
    } else {
        // Label 140: the weight (b−x)^beta is in the moments; (x−a)^alfa stays in f.
        let fix = centr - a;
        let mut fval = [0.0; 26];
        fval[1] = 0.5 * f(hlgth + centr) * (fix + hlgth).powf(alfa);
        fval[13] = f(centr) * fix.powf(alfa);
        fval[25] = 0.5 * f(centr - hlgth) * (fix - hlgth).powf(alfa);
        for i in 2..=12 {
            let u = hlgth * CC_X[i - 2];
            let isym = 26 - i;
            fval[i] = f(u + centr) * (fix + u).powf(alfa);
            fval[isym] = f(centr - u) * (fix - u).powf(alfa);
        }
        factor = hlgth.powf(beta + 1.0);
        if integr == 2 || integr == 4 {
            // Label 200: log(x−a) is also carried by f.
            fval[1] *= (hlgth + fix).ln();
            fval[13] *= fix.ln();
            fval[25] *= (fix - hlgth).ln();
            for i in 2..=12 {
                let u = hlgth * CC_X[i - 2];
                let isym = 26 - i;
                fval[i] *= (u + fix).ln();
                fval[isym] *= (fix - u).ln();
            }
        }
        let mut cheb12 = [0.0; 14];
        let mut cheb24 = [0.0; 26];
        qcheb(&mut fval, &mut cheb12, &mut cheb24);
        let (r12, r24) = cc_moment_sums(&cheb12, &cheb24, &moments.rj);
        if integr == 1 || integr == 2 {
            res12 = r12;
            res24 = r24;
        } else {
            let dc = (br - bl).ln();
            result = r24 * dc;
            abserr = ((r24 - r12) * dc).abs();
            let (s12, s24) = cc_moment_sums(&cheb12, &cheb24, &moments.rh);
            res12 = s12;
            res24 = s24;
        }
    }
    result = (result + res24) * factor;
    abserr = (abserr + (res24 - res12).abs()) * factor;
    (result, abserr, 0.0, 25)
}

/// `dqawse`: `∫_a^b f(x)·(x−a)^alfa·(b−x)^beta·v(x)` with `v = 1`, `log(x−a)`, `log(b−x)` or
/// their product for `integr` 1–4 (SciPy's `weight='alg' | 'alg-loga' | 'alg-logb' |
/// 'alg-log'`, `wvar = (alfa, beta)`), `alfa, beta > -1`, `a < b`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qawse(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    alfa: f64,
    beta: f64,
    integr: u8,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Qags {
    let mut lists = Lists::new(limit.max(2));
    if b <= a
        || (epsabs == 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28))
        || alfa <= -1.0
        || beta <= -1.0
        || !(1..=4).contains(&integr)
        || limit < 2
    {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let moments = qmomo(alfa, beta, integr);
    let centre = 0.5 * (b + a);
    let (area1, error1, _, nev1) = qc25s(f, a, b, a, centre, alfa, beta, &moments, integr);
    let (area2, error2, _, nev2) = qc25s(f, a, b, centre, b, alfa, beta, &moments, integr);
    let mut neval = nev1 + nev2;
    let mut last = 2;
    let mut result = area1 + area2;
    let mut abserr = error1 + error2;
    let errbnd = epsabs.max(epsrel * result.abs());
    if error2 > error1 {
        lists.alist[1] = centre;
        lists.alist[2] = a;
        lists.blist[1] = b;
        lists.blist[2] = centre;
        lists.rlist[1] = area2;
        lists.rlist[2] = area1;
        lists.elist[1] = error2;
        lists.elist[2] = error1;
    } else {
        lists.alist[1] = a;
        lists.alist[2] = centre;
        lists.blist[1] = centre;
        lists.blist[2] = b;
        lists.rlist[1] = area1;
        lists.rlist[2] = area2;
        lists.elist[1] = error1;
        lists.elist[2] = error2;
    }
    lists.iord[1] = 1;
    lists.iord[2] = 2;
    let mut ier = 0;
    if limit == 2 {
        ier = 1;
    }
    if abserr <= errbnd || ier == 1 {
        return lists.into_output(result, abserr, neval, ier, last);
    }
    let mut errmax = lists.elist[1];
    let mut maxerr = 1;
    let mut nrmax = 1;
    let mut area = result;
    let mut errsum = abserr;
    let (mut iroff1, mut iroff2) = (0, 0);
    for current in 3..=limit {
        last = current;
        let a1 = lists.alist[maxerr];
        let b1 = 0.5 * (lists.alist[maxerr] + lists.blist[maxerr]);
        let a2 = b1;
        let b2 = lists.blist[maxerr];
        let (area1, error1, resas1, nev1) = qc25s(f, a, b, a1, b1, alfa, beta, &moments, integr);
        let (area2, error2, resas2, nev2) = qc25s(f, a, b, a2, b2, alfa, beta, &moments, integr);
        neval += nev1 + nev2;
        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - lists.rlist[maxerr];
        if !(a == a1 || b == b2 || resas1 == error1 || resas2 == error2) {
            if (lists.rlist[maxerr] - area12).abs() < 1.0e-5 * area12.abs()
                && erro12 >= 0.99 * errmax
            {
                iroff1 += 1;
            }
            if last > 10 && erro12 > errmax {
                iroff2 += 1;
            }
        }
        lists.rlist[maxerr] = area1;
        lists.rlist[last] = area2;
        let errbnd = epsabs.max(epsrel * area.abs());
        if errsum > errbnd {
            if last == limit {
                ier = 1;
            }
            if iroff1 >= 6 || iroff2 >= 20 {
                ier = 2;
            }
            if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
                ier = 3;
            }
        }
        if error2 > error1 {
            lists.alist[maxerr] = a2;
            lists.alist[last] = a1;
            lists.blist[last] = b1;
            lists.rlist[maxerr] = area2;
            lists.rlist[last] = area1;
            lists.elist[maxerr] = error2;
            lists.elist[last] = error1;
        } else {
            lists.alist[last] = a2;
            lists.blist[maxerr] = b1;
            lists.blist[last] = b2;
            lists.elist[maxerr] = error1;
            lists.elist[last] = error2;
        }
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &lists.elist,
            &mut lists.iord,
            &mut nrmax,
        );
        if ier != 0 || errsum <= errbnd {
            break;
        }
    }
    result = lists.rlist[1..=last].iter().sum();
    abserr = errsum;
    lists.into_output(result, abserr, neval, ier, last)
}

/// LINPACK `dgtsl`: solve the tridiagonal system with sub-diagonal `c[2..=n]`, diagonal
/// `d[1..=n]`, super-diagonal `e[1..n]` (all 1-based, overwritten) and right-hand side
/// `b[off+1..=off+n]` in place, by Gaussian elimination with partial pivoting. Returns
/// LINPACK's `info` (0, or the index of a zero pivot).
fn dgtsl(
    n: usize,
    c: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    b: &mut [f64],
    off: usize,
) -> usize {
    let bi = |k: usize| off + k;
    c[1] = d[1];
    let nm1 = n - 1;
    if nm1 >= 1 {
        d[1] = e[1];
        e[1] = 0.0;
        e[n] = 0.0;
        for k in 1..=nm1 {
            let kp1 = k + 1;
            if c[kp1].abs() >= c[k].abs() {
                c.swap(kp1, k);
                d.swap(kp1, k);
                e.swap(kp1, k);
                b.swap(bi(kp1), bi(k));
            }
            if c[k] == 0.0 {
                return k;
            }
            let t = -c[kp1] / c[k];
            c[kp1] = d[kp1] + t * d[k];
            d[kp1] = e[kp1] + t * e[k];
            e[kp1] = 0.0;
            b[bi(kp1)] += t * b[bi(k)];
        }
    }
    if c[n] == 0.0 {
        return n;
    }
    b[bi(n)] /= c[n];
    if n > 1 {
        b[bi(nm1)] = (b[bi(nm1)] - d[nm1] * b[bi(n)]) / c[nm1];
        if n >= 3 {
            let nm2 = n - 2;
            for kb in 1..=nm2 {
                let k = nm2 - kb + 1;
                b[bi(k)] = (b[bi(k)] - d[k] * b[bi(k + 1)] - e[k] * b[bi(k + 2)]) / c[k];
            }
        }
    }
    0
}

/// The Chebyshev moments `dqc25f` computes for the cos/sin weight, kept across calls (and
/// across `dqawfe`'s cycles) as QUADPACK keeps `momcom`, `chebmo` and `dqc25f`'s saved `m`.
pub(crate) struct FourierMoments {
    momcom: usize,
    /// `chebmo[m][k]`, both 1-based: `maxp1` rows of 25 moments.
    chebmo: Vec<[f64; 26]>,
    m: usize,
}

impl FourierMoments {
    pub(crate) fn new(maxp1: usize) -> Self {
        Self {
            momcom: 0,
            chebmo: vec![[0.0; 26]; maxp1 + 1],
            m: 0,
        }
    }
}

/// `dqc25f`: `∫_a^b f(x)·cos(omega·x)` (`integr` 1) or `·sin(omega·x)` (2) by the modified
/// Clenshaw–Curtis rule when `|omega·(b−a)/2| > 2`, 15-point Kronrod otherwise. `nrmom` is the
/// bisection level of the interval (its moments are reused by level). Returns
/// `(result, abserr, neval, resabs, resasc)`.
#[allow(clippy::too_many_arguments)]
fn qc25f(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    omega: f64,
    integr: u8,
    nrmom: usize,
    maxp1: usize,
    ksave: u8,
    moments: &mut FourierMoments,
) -> (f64, f64, usize, f64, f64) {
    let centr = 0.5 * (b + a);
    let hlgth = 0.5 * (b - a);
    let parint = omega * hlgth;
    if parint.abs() <= 2.0 {
        let panel = qk15w(
            f,
            &|x| {
                if integr == 1 {
                    (omega * x).cos()
                } else {
                    (omega * x).sin()
                }
            },
            a,
            b,
        );
        return (panel.result, panel.abserr, 15, panel.resabs, panel.resasc);
    }
    let conc = hlgth * (centr * omega).cos();
    let cons = hlgth * (centr * omega).sin();
    let resasc = OFLOW;
    if !(nrmom < moments.momcom || ksave == 1) {
        // Compute the moments of cos and sin for this level (labels 10–110).
        let m = moments.momcom + 1;
        moments.m = m;
        let par2 = parint * parint;
        let par22 = par2 + 2.0;
        let sinpar = parint.sin();
        let cospar = parint.cos();
        let mut v = [0.0; 29];
        let mut d = [0.0; 26];
        let mut d1 = [0.0; 26];
        let mut d2 = [0.0; 26];
        const NOEQU: usize = 25;
        const NOEQ1: usize = NOEQU - 1;
        v[1] = 2.0 * sinpar / parint;
        v[2] = (8.0 * cospar + (par2 + par2 - 8.0) * sinpar / parint) / par2;
        v[3] = (32.0 * (par2 - 12.0) * cospar
            + (2.0 * ((par2 - 80.0) * par2 + 192.0) * sinpar) / parint)
            / (par2 * par2);
        let ac = 8.0 * cospar;
        let as_ = 24.0 * parint * sinpar;
        if parint.abs() > 24.0 {
            let mut an = 4.0;
            for i in 4..=13 {
                let an2 = an * an;
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] - ac) + as_
                    - par2 * (an + 1.0) * (an + 2.0) * v[i - 2])
                    / (par2 * (an - 1.0) * (an - 2.0));
                an += 2.0;
            }
        } else {
            let mut an = 6.0;
            for k in 1..=NOEQ1 {
                let an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 3] = as_ - (an2 - 4.0) * ac;
                an += 2.0;
            }
            let an2 = an * an;
            d[NOEQU] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
            v[NOEQU + 3] = as_ - (an2 - 4.0) * ac;
            v[4] -= 56.0 * par2 * v[3];
            let ass = parint * sinpar;
            let asap = (((((210.0 * par2 - 1.0) * cospar - (105.0 * par2 - 63.0) * ass) / an2
                - (1.0 - 15.0 * par2) * cospar
                + 15.0 * ass)
                / an2
                - cospar
                + 3.0 * ass)
                / an2
                - cospar)
                / an2;
            v[NOEQU + 3] -= 2.0 * asap * par2 * (an - 1.0) * (an - 2.0);
            dgtsl(NOEQU, &mut d1, &mut d, &mut d2, &mut v, 3);
        }
        for j in 1..=13 {
            moments.chebmo[m][2 * j - 1] = v[j];
        }
        v[1] = 2.0 * (sinpar - parint * cospar) / par2;
        v[2] = (18.0 - 48.0 / par2) * sinpar / par2 + (-2.0 + 48.0 / par2) * cospar / parint;
        let ac = -24.0 * parint * cospar;
        let as_ = -8.0 * sinpar;
        if parint.abs() > 24.0 {
            let mut an = 3.0;
            for i in 3..=12 {
                let an2 = an * an;
                v[i] = ((an2 - 4.0) * (2.0 * (par22 - an2 - an2) * v[i - 1] + as_) + ac
                    - par2 * (an + 1.0) * (an + 2.0) * v[i - 2])
                    / (par2 * (an - 1.0) * (an - 2.0));
                an += 2.0;
            }
        } else {
            let mut an = 5.0;
            for k in 1..=NOEQ1 {
                let an2 = an * an;
                d[k] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
                d2[k] = (an - 1.0) * (an - 2.0) * par2;
                d1[k + 1] = (an + 3.0) * (an + 4.0) * par2;
                v[k + 2] = ac + (an2 - 4.0) * as_;
                an += 2.0;
            }
            let an2 = an * an;
            d[NOEQU] = -2.0 * (an2 - 4.0) * (par22 - an2 - an2);
            v[NOEQU + 2] = ac + (an2 - 4.0) * as_;
            v[3] -= 42.0 * par2 * v[2];
            let ass = parint * cospar;
            let asap = (((((105.0 * par2 - 63.0) * ass + (210.0 * par2 - 1.0) * sinpar) / an2
                + (15.0 * par2 - 1.0) * sinpar
                - 15.0 * ass)
                / an2
                - 3.0 * ass
                - sinpar)
                / an2
                - sinpar)
                / an2;
            v[NOEQU + 2] -= 2.0 * asap * par2 * (an - 1.0) * (an - 2.0);
            dgtsl(NOEQU, &mut d1, &mut d, &mut d2, &mut v, 2);
        }
        for j in 1..=12 {
            moments.chebmo[m][2 * j] = v[j];
        }
    }
    // Label 120.
    if nrmom < moments.momcom {
        moments.m = nrmom + 1;
    }
    if moments.momcom < maxp1 - 1 && nrmom >= moments.momcom {
        moments.momcom += 1;
    }
    let m = moments.m;
    let mut fval = cc_samples(f, centr, hlgth);
    let mut cheb12 = [0.0; 14];
    let mut cheb24 = [0.0; 26];
    qcheb(&mut fval, &mut cheb12, &mut cheb24);
    let mo = &moments.chebmo[m];
    // Fortran runs K = 11, 9, …, 1 (then 23, 21, …, 1) and leaves K = −1 behind; the ranges
    // below take the same terms in the same order without that final decrement.
    let mut resc12 = cheb12[13] * mo[13];
    let mut ress12 = 0.0;
    for k in (1..=11).rev().step_by(2) {
        resc12 += cheb12[k] * mo[k];
        ress12 += cheb12[k + 1] * mo[k + 1];
    }
    let mut resc24 = cheb24[25] * mo[25];
    let mut ress24 = 0.0;
    let mut resabs = cheb24[25].abs();
    for k in (1..=23).rev().step_by(2) {
        resc24 += cheb24[k] * mo[k];
        ress24 += cheb24[k + 1] * mo[k + 1];
        resabs += cheb24[k].abs() + cheb24[k + 1].abs();
    }
    let estc = (resc24 - resc12).abs();
    let ests = (ress24 - ress12).abs();
    let resabs = resabs * hlgth.abs();
    let (result, abserr) = if integr == 2 {
        (
            conc * ress24 + cons * resc24,
            (conc * ests).abs() + (cons * estc).abs(),
        )
    } else {
        (
            conc * resc24 - cons * ress24,
            (conc * estc).abs() + (cons * ests).abs(),
        )
    };
    (result, abserr, 25, resabs, resasc)
}

/// `dqawoe`: `∫_a^b f(x)·cos(omega·x)` (`integr` 1) or `·sin(omega·x)` (2). `icall > 1` reuses
/// the moments already in `moments` (as `dqawfe`'s cycles do).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qawoe(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    omega: f64,
    integr: u8,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
    icall: usize,
    maxp1: usize,
    moments: &mut FourierMoments,
) -> Qags {
    let mut lists = Lists::new(limit);
    let mut nnlog = vec![0_usize; limit + 2];
    lists.alist[1] = a;
    lists.blist[1] = b;
    if (integr != 1 && integr != 2) || invalid_tolerances(epsabs, epsrel) || icall < 1 || maxp1 < 1
    {
        return lists.into_output(0.0, 0.0, 0, 6, 0);
    }
    let domega = omega.abs();
    let mut nrmom = 0;
    if icall <= 1 {
        moments.momcom = 0;
    }
    let (first, first_err, mut neval, defabs, _) =
        qc25f(f, a, b, domega, integr, nrmom, maxp1, 0, moments);
    let mut result = first;
    let mut abserr = first_err;
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    lists.rlist[1] = result;
    lists.elist[1] = abserr;
    lists.iord[1] = 1;
    let mut ier: u8 = 0;
    // `dqawoe` sets `last` only in its bisection loop, so a first-panel exit reports 0
    // subintervals (SciPy's infodict["last"] == 0), unlike `dqagse`.
    let mut last = 0;
    if abserr <= 100.0 * EPMACH * defabs && abserr > errbnd {
        ier = 2;
    }
    if limit == 1 {
        ier = 1;
    }
    if ier != 0 || abserr <= errbnd {
        if integr == 2 && omega < 0.0 {
            result = -result;
        }
        return lists.into_output(result, abserr, neval, ier, last);
    }
    let mut errmax = abserr;
    let mut maxerr = 1;
    let mut area = result;
    let mut errsum = abserr;
    abserr = OFLOW;
    let mut nrmax = 1;
    let mut extrap = false;
    let mut noext = false;
    let mut ierro: u8 = 0;
    let (mut iroff1, mut iroff2, mut iroff3) = (0, 0, 0);
    let mut ktmin = 0;
    let mut small = (b - a).abs() * 0.75;
    let mut nres = 0;
    let mut numrl2 = 0;
    let mut extall = false;
    let mut rlist2 = [0.0; EPSTAB_LEN];
    let mut res3la = [0.0; 4];
    if 0.5 * (b - a).abs() * domega <= 2.0 {
        numrl2 = 1;
        extall = true;
        rlist2[1] = result;
    }
    if 0.25 * (b - a).abs() * domega <= 2.0 {
        extall = true;
    }
    let ksgn = if dres >= (1.0 - 50.0 * EPMACH) * defabs {
        1
    } else {
        -1
    };
    let (mut erlarg, mut ertest, mut correc) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut reseps, mut abseps) = (0.0, 0.0);
    let mut sum_up = false;

    'bisect: for current in 2..=limit {
        last = current;
        nrmom = nnlog[maxerr] + 1;
        let a1 = lists.alist[maxerr];
        let b1 = 0.5 * (lists.alist[maxerr] + lists.blist[maxerr]);
        let a2 = b1;
        let b2 = lists.blist[maxerr];
        let erlast = errmax;
        let (area1, error1, nev1, _, defab1) =
            qc25f(f, a1, b1, domega, integr, nrmom, maxp1, 0, moments);
        let (area2, error2, nev2, _, defab2) =
            qc25f(f, a2, b2, domega, integr, nrmom, maxp1, 1, moments);
        neval += nev1 + nev2;
        let area12 = area1 + area2;
        let erro12 = error1 + error2;
        errsum = errsum + erro12 - errmax;
        area = area + area12 - lists.rlist[maxerr];
        if !(defab1 == error1 || defab2 == error2) {
            if !((lists.rlist[maxerr] - area12).abs() > 1.0e-5 * area12.abs()
                || erro12 < 0.99 * errmax)
            {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if last > 10 && erro12 > errmax {
                iroff3 += 1;
            }
        }
        lists.rlist[maxerr] = area1;
        lists.rlist[last] = area2;
        nnlog[maxerr] = nrmom;
        nnlog[last] = nrmom;
        errbnd = epsabs.max(epsrel * area.abs());
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierro = 3;
        }
        if last == limit {
            ier = 1;
        }
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }
        if error2 > error1 {
            lists.alist[maxerr] = a2;
            lists.alist[last] = a1;
            lists.blist[last] = b1;
            lists.rlist[maxerr] = area2;
            lists.rlist[last] = area1;
            lists.elist[maxerr] = error2;
            lists.elist[last] = error1;
        } else {
            lists.alist[last] = a2;
            lists.blist[maxerr] = b1;
            lists.blist[last] = b2;
            lists.elist[maxerr] = error1;
            lists.elist[last] = error2;
        }
        qpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &lists.elist,
            &mut lists.iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            sum_up = true;
            break;
        }
        if ier != 0 {
            break;
        }
        if last == 2 && extall {
            // Label 120 then 130.
            small *= 0.5;
            numrl2 += 1;
            rlist2[numrl2] = area;
            ertest = errbnd;
            erlarg = errsum;
            continue;
        }
        if noext {
            continue;
        }
        let mut goto70 = false;
        if extall {
            erlarg -= erlast;
            if (b1 - a1).abs() > small {
                erlarg += erro12;
            }
            goto70 = extrap;
        }
        if !goto70 {
            // Label 50.
            let width = (lists.blist[maxerr] - lists.alist[maxerr]).abs();
            if width > small {
                continue;
            }
            if !extall {
                small *= 0.5;
                if 0.25 * width * domega > 2.0 {
                    continue;
                }
                extall = true;
                // Label 130.
                ertest = errbnd;
                erlarg = errsum;
                continue;
            }
            // Label 60.
            extrap = true;
            nrmax = 2;
        }
        // Label 70.
        if ierro != 3 && erlarg > ertest {
            let jupbnd = if last > limit / 2 + 2 {
                limit + 3 - last
            } else {
                last
            };
            let id = nrmax;
            for _ in id..=jupbnd {
                maxerr = lists.iord[nrmax];
                errmax = lists.elist[maxerr];
                if (lists.blist[maxerr] - lists.alist[maxerr]).abs() > small {
                    continue 'bisect;
                }
                nrmax += 1;
            }
        }
        // Label 90.
        numrl2 += 1;
        rlist2[numrl2] = area;
        if numrl2 >= 3 {
            qelg(
                &mut numrl2,
                &mut rlist2,
                &mut reseps,
                &mut abseps,
                &mut res3la,
                &mut nres,
            );
            ktmin += 1;
            if ktmin > 5 && abserr < 1.0e-3 * errsum {
                ier = 5;
            }
            if abseps < abserr {
                ktmin = 0;
                abserr = abseps;
                result = reseps;
                correc = erlarg;
                ertest = epsabs.max(epsrel * reseps.abs());
                if abserr <= ertest {
                    break;
                }
            }
            // Label 100.
            if numrl2 == 1 {
                noext = true;
            }
            if ier == 5 {
                break;
            }
        }
        // Label 110.
        maxerr = lists.iord[1];
        errmax = lists.elist[maxerr];
        nrmax = 1;
        extrap = false;
        small *= 0.5;
        erlarg = errsum;
    }

    // Labels 150–190: `dqawoe` also sums the panels when no extrapolation was ever made
    // (`nres == 0`) and flags divergence on `errsum >= |area|`.
    #[derive(PartialEq)]
    enum Tail {
        Sum,
        Check,
        Done,
    }
    let mut tail = if sum_up || abserr == OFLOW || nres == 0 {
        Tail::Sum
    } else {
        Tail::Check
    };
    if tail == Tail::Check && ier + ierro != 0 {
        if ierro == 3 {
            abserr += correc;
        }
        if ier == 0 {
            ier = 3;
        }
        if result != 0.0 && area != 0.0 {
            if abserr / result.abs() > errsum / area.abs() {
                tail = Tail::Sum;
            }
        } else if abserr > errsum {
            tail = Tail::Sum;
        } else if area == 0.0 {
            tail = Tail::Done;
        }
    }
    if tail == Tail::Check {
        if !(ksgn == -1 && result.abs().max(area.abs()) <= defabs * 0.01)
            && (0.01 > result / area || result / area > 100.0 || errsum >= area.abs())
        {
            ier = 6;
        }
        tail = Tail::Done;
    }
    if tail == Tail::Sum {
        result = lists.rlist[1..=last].iter().sum();
        abserr = errsum;
    }
    if ier > 2 {
        ier -= 1;
    }
    if integr == 2 && omega < 0.0 {
        result = -result;
    }
    lists.into_output(result, abserr, neval, ier, last)
}

/// The outcome of `dqawfe`: the integral over `[a, ∞)` and the per-cycle results.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qawfe {
    pub result: f64,
    pub abserr: f64,
    pub neval: usize,
    /// QUADPACK's `ier` for the whole integral: 0, 1 (`limlst` cycles), 4 (extrapolation
    /// table does not converge), 6 (invalid input), 7 (a cycle failed; see `ierlst`).
    pub ier: u8,
    pub lst: usize,
    pub rslst: Vec<f64>,
    pub erlst: Vec<f64>,
    pub ierlst: Vec<u8>,
}

/// `dqawfe`: `∫_a^∞ f(x)·cos(omega·x)` or `·sin(omega·x)`, integrated over successive cycles
/// of length `(2⌊|omega|⌋ + 1)·π/|omega|` with `dqawoe` and the sum extrapolated by `dqelg`.
/// Only `epsabs` is used, as in QUADPACK and SciPy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qawfe(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    omega: f64,
    integr: u8,
    epsabs: f64,
    limlst: usize,
    limit: usize,
    maxp1: usize,
) -> Qawfe {
    let mut out = Qawfe {
        result: 0.0,
        abserr: 0.0,
        neval: 0,
        ier: 0,
        lst: 0,
        rslst: Vec::new(),
        erlst: Vec::new(),
        ierlst: Vec::new(),
    };
    if (integr != 1 && integr != 2) || epsabs <= 0.0 || limlst < 3 {
        out.ier = 6;
        return out;
    }
    if omega == 0.0 {
        if integr == 1 {
            // netlib QUADPACK integrates from 0 here (`dqagie(f, 0.0d+00, 1, ...)`), not from
            // `a`, and SciPy 1.17.1 still does: quad(exp(-x), 1, inf, weight='cos', wvar=0)
            // returns 1.0 where the integral is e^-1. This port integrates from `a`: a known
            // wrong answer is not a compatibility contract.
            let q = qagie(f, a, 1, epsabs, 0.0, limit);
            out.result = q.result;
            out.abserr = q.abserr;
            out.neval = q.neval;
            out.ier = q.ier;
            out.rslst.push(q.result);
            out.erlst.push(q.abserr);
            out.ierlst.push(q.ier);
        } else {
            out.rslst.push(0.0);
            out.erlst.push(0.0);
            out.ierlst.push(0);
        }
        out.lst = 1;
        return out;
    }
    const P: f64 = 0.9;
    let l = omega.abs().trunc();
    let dl = 2.0 * l + 1.0;
    let cycle = dl * std::f64::consts::PI / omega.abs();
    let mut ier: u8 = 0;
    let mut ktmin = 0;
    let mut numrl2 = 0;
    let mut nres = 0;
    let mut c1 = a;
    let mut c2 = cycle + a;
    let p1 = 1.0 - P;
    let mut eps = epsabs;
    if epsabs > UFLOW / p1 {
        eps = epsabs * p1;
    }
    let ep = eps;
    let mut fact = 1.0;
    let mut correc = 0.0_f64;
    let mut abserr = 0.0;
    let mut result = 0.0;
    let mut errsum = 0.0;
    let mut psum = [0.0; EPSTAB_LEN];
    let mut res3la = [0.0; 4];
    let mut ll = 0;
    let mut drl = 0.0;
    let mut moments = FourierMoments::new(maxp1);
    let (mut reseps, mut abseps) = (0.0, 0.0);
    // `go to 80`: return the partial sum instead of the extrapolated value. Every other exit
    // from the loop is `go to 60`.
    let mut use_sum = false;
    for lst in 1..=limlst {
        out.lst = lst;
        let epsa = eps * fact;
        let cyc = qawoe(
            f,
            c1,
            c2,
            omega,
            integr,
            epsa,
            0.0,
            limit,
            lst,
            maxp1,
            &mut moments,
        );
        out.rslst.push(cyc.result);
        out.erlst.push(cyc.abserr);
        out.ierlst.push(cyc.ier);
        out.neval += cyc.neval;
        fact *= P;
        errsum += cyc.abserr;
        drl = 50.0 * cyc.result.abs();
        if errsum + drl <= epsabs && lst >= 6 {
            use_sum = true;
            break;
        }
        correc = correc.max(cyc.abserr);
        if cyc.ier != 0 {
            eps = ep.max(correc * p1);
            ier = 7;
        }
        if ier == 7 && errsum + drl <= correc * 10.0 && lst > 5 {
            use_sum = true;
            break;
        }
        numrl2 += 1;
        if lst > 1 {
            psum[numrl2] = psum[ll] + cyc.result;
            if lst != 2 {
                if lst == limlst {
                    ier = 1;
                }
                qelg(
                    &mut numrl2,
                    &mut psum,
                    &mut reseps,
                    &mut abseps,
                    &mut res3la,
                    &mut nres,
                );
                ktmin += 1;
                if ktmin >= 15 && abserr <= 1.0e-3 * (errsum + drl) {
                    ier = 4;
                }
                if !(abseps > abserr && lst != 3) {
                    abserr = abseps;
                    result = reseps;
                    ktmin = 0;
                    if abserr + 10.0 * correc <= epsabs
                        || (abserr <= epsabs && 10.0 * correc >= epsabs)
                    {
                        break;
                    }
                }
                if ier != 0 && ier != 7 {
                    break;
                }
            }
        } else {
            psum[1] = cyc.result;
        }
        ll = numrl2;
        c1 = c2;
        c2 += cycle;
    }
    if !use_sum {
        // Label 60: keep the extrapolated `result` (return here) unless the plain partial sum
        // is the better estimate (fall through to label 80).
        abserr += 10.0 * correc;
        let psum_last = psum[numrl2];
        let keep_extrapolated = if ier == 0 {
            true
        } else if result != 0.0 && psum_last != 0.0 {
            abserr / result.abs() <= (errsum + drl) / psum_last.abs()
        } else if abserr > errsum {
            false
        } else if psum_last == 0.0 {
            return Qawfe {
                result,
                abserr,
                ier,
                ..out
            };
        } else {
            // Label 70 reached with `result == 0`: its ratio test divides by zero and fails.
            abserr / result.abs() <= (errsum + drl) / psum_last.abs()
        };
        if keep_extrapolated {
            if ier != 0 && ier != 7 {
                abserr += drl;
            }
            return Qawfe {
                result,
                abserr,
                ier,
                ..out
            };
        }
    }
    // Label 80.
    Qawfe {
        result: psum[numrl2],
        abserr: errsum + drl,
        ier,
        ..out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wynn's epsilon algorithm on the partial sums of 1 − 1/2 + 1/3 − … (→ ln 2), fed the way
    /// the drivers feed it: one new sum per call.
    #[test]
    fn epsilon_algorithm_accelerates_the_alternating_harmonic_series() {
        let mut table = [0.0; EPSTAB_LEN];
        let mut res3la = [0.0; 4];
        let (mut n, mut nres) = (0, 0);
        let (mut result, mut abserr) = (0.0, 0.0);
        let mut partial = 0.0;
        for k in 1..=12 {
            partial += if k % 2 == 1 { 1.0 } else { -1.0 } / f64::from(k);
            n += 1;
            table[n] = partial;
            qelg(
                &mut n,
                &mut table,
                &mut result,
                &mut abserr,
                &mut res3la,
                &mut nres,
            );
        }
        // Twelve partial sums are only good to ~4e-2; the extrapolated limit is good to 1.0e-9.
        assert!((result - 2.0_f64.ln()).abs() < 1e-8, "{result}");
        assert!((partial - 2.0_f64.ln()).abs() > 1e-2);
    }

    /// GK21 integrates polynomials of degree ≤ 31 exactly; its Gauss-10 part degree ≤ 19.
    #[test]
    fn gk21_panel_is_exact_on_high_degree_polynomials() {
        for degree in [0_i32, 7, 19, 31] {
            let mut f = |x: f64| x.powi(degree);
            let panel = qk21(&mut f, -0.3, 1.7);
            let exact =
                (1.7_f64.powi(degree + 1) - (-0.3_f64).powi(degree + 1)) / f64::from(degree + 1);
            assert!(
                (panel.result - exact).abs() <= 8.0 * f64::EPSILON * exact.abs().max(1.0),
                "degree {degree}: {} vs {exact}",
                panel.result
            );
        }
    }
}
