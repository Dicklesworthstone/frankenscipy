//! The regularized incomplete beta ratio `I_x(a, b)` and its complement `1 − I_x(a, b)`:
//! a port of `bratio` from ACM TOMS 708 (A. R. Didonato and A. H. Morris Jr., "Significant
//! digit computation of the incomplete beta function ratios", ACM Trans. Math. Softw. 18(3),
//! 360-373, 1992) as SciPy 1.17.1 ships it in `scipy/special/cdflib.c` (BSD-3-Clause,
//! Copyright (C) 2024 SciPy developers), with every helper it calls (frankenscipy-5pnba).
//!
//! It replaced a Numerical Recipes continued fraction whose 200-term cap silently truncated
//! near the mean once a shape parameter was large: `stdtr(1e15, 2)` was off by 8e-4 and
//! `betainc(2.5, 1e20, 5e-20)` came out as −4.9e281. `bratio` instead picks one of seven
//! expansions by region (power series, a continued fraction in `λ = (a+b)·y − b`, Temme's
//! asymptotic expansions for one or both parameters large) and returns both `w = I_x(a, b)`
//! and `w1 = 1 − I_x(a, b)` to full relative precision, so a caller that wants the upper tail
//! reads `w1` instead of subtracting.
//!
//! The module is std-only and private. Its helpers are cdflib's own (`betaln`, `gamln`,
//! `psi`, …): fsci's `betaln_scalar`, `gammaln` and `digamma` compute the same functions by
//! different formulas, so reusing them would make this a different algorithm with different
//! rounding. Being self-contained is also what lets the port be checked bit-for-bit against
//! the compiled `cdflib.c`.
//!
//! Departures from the C. One changes numbers: `λ = a − (a + b)·x` is formed without the
//! rounding of `(a + b)·x` (see [`beta_lambda`]), which is what limits TOMS 708 when both
//! parameters are large and `x` is near the mean. Everything else is bit-for-bit cdflib; with
//! plain-arithmetic `λ` the port matched the compiled `cdflib.c` on all 60,483 rows of a
//! regime sweep. The others are fail-closed and not reachable from finite, valid input: every
//! open-ended loop has a term cap, and a result that would have come from a truncated loop,
//! or from an input the C reports through `ierr`, is `NaN` rather than TOMS's `0`.

use std::f64::consts::PI;

/// Term cap for the loops cdflib runs until convergence. With finite, valid input they stop
/// long before this (the largest count seen over the frankenscipy-5pnba sweep was far below
/// it); the cap only turns a loop that cannot converge into a `NaN`.
const MAX_TERMS: u32 = 100_000;

/// `(w, w1) = (I_x(a, b), 1 − I_x(a, b))` for `a, b ≥ 0`, `x ∈ [0, 1]` and `y = 1 − x` passed
/// in separately, so a caller that knows `1 − x` in closed form keeps its digits.
///
/// Inputs cdflib rejects (a negative shape, `a = b = 0`, `x` or `y` outside `[0, 1]`,
/// `|x + y − 1| > 3·10⁻¹⁵`, `x = a = 0`, `y = b = 0`) and NaN or infinite inputs give
/// `(NaN, NaN)`.
pub(crate) fn bratio(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    const FAIL: (f64, f64) = (f64::NAN, f64::NAN);
    // cdflib: eps = max(spmpar[0], 1e-15).
    let eps = f64::EPSILON.max(1e-15);

    if !(a.is_finite() && b.is_finite() && x.is_finite() && y.is_finite()) {
        return FAIL;
    }
    if a < 0.0 || b < 0.0 || (a == 0.0 && b == 0.0) {
        return FAIL;
    }
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        return FAIL;
    }
    if (((x + y) - 0.5) - 0.5).abs() > 3.0 * eps {
        return FAIL;
    }
    if x == 0.0 {
        return if a == 0.0 { FAIL } else { (0.0, 1.0) };
    }
    if y == 0.0 {
        return if b == 0.0 { FAIL } else { (1.0, 0.0) };
    }
    if a == 0.0 {
        return (1.0, 0.0);
    }
    if b == 0.0 {
        return (0.0, 1.0);
    }
    if a.max(b) < 1e-3 * eps {
        return (b / (a + b), a / (a + b));
    }

    // `ind`: the expansions below run on (a0, b0, x0, y0), which is (a, b, x, y) or its
    // reflection (b, a, y, x); a reflected run computes I_y(b, a) = 1 − I_x(a, b).
    let (mut a0, mut b0, mut x0, mut y0) = (a, b, x, y);
    let mut ind = false;
    let orient = |ind: bool, w: f64, w1: f64| if ind { (w1, w) } else { (w, w1) };
    let from_w = |ind: bool, w: f64| orient(ind, w, 0.5 + (0.5 - w));
    let from_w1 = |ind: bool, w1: f64| orient(ind, 0.5 + (0.5 - w1), w1);
    // cdflib's label 140: I_y0(b0, a0) by BUP up to b0 + 20 and BGRAT from there.
    let bup_then_bgrat = |a0: f64, b0: f64, x0: f64, y0: f64| {
        let n = 20;
        let w1 = bup(b0, a0, y0, x0, n, eps);
        bgrat(b0 + f64::from(n), a0, y0, x0, w1, 15.0 * eps)
    };

    if a0.min(b0) <= 1.0 {
        if !(x <= 0.5) {
            ind = true;
            (a0, b0, x0, y0) = (b, a, y, x);
        }
        if b0 < eps.min(eps * a0) {
            return from_w(ind, fpser(a0, b0, x0, eps));
        }
        if a0 < eps.min(eps * b0) && b0 * x0 <= 1.0 {
            return from_w1(ind, apser(a0, b0, x0, eps));
        }
        if a0.max(b0) <= 1.0 {
            if x0.powf(a0) <= 0.9 || a0 >= 0.2_f64.min(b0) {
                return from_w(ind, bpser(a0, b0, x0, eps));
            }
            if x0 >= 0.3 {
                return from_w1(ind, bpser(b0, a0, y0, eps));
            }
            return from_w1(ind, bup_then_bgrat(a0, b0, x0, y0));
        }
        // cdflib's label 20: max(a0, b0) > 1.
        if b0 <= 1.0 {
            return from_w(ind, bpser(a0, b0, x0, eps));
        }
        if x0 >= 0.3 {
            return from_w1(ind, bpser(b0, a0, y0, eps));
        }
        if x0 >= 0.1 {
            if b0 > 15.0 {
                return from_w1(ind, bgrat(b0, a0, y0, x0, 0.0, 15.0 * eps));
            }
            return from_w1(ind, bup_then_bgrat(a0, b0, x0, y0));
        }
        if (x0 * b0).powf(a0) <= 0.7 {
            return from_w(ind, bpser(a0, b0, x0, eps));
        }
        if b0 > 15.0 {
            return from_w1(ind, bgrat(b0, a0, y0, x0, 0.0, 15.0 * eps));
        }
        return from_w1(ind, bup_then_bgrat(a0, b0, x0, y0));
    }

    // min(a, b) > 1. Reflect so that λ = a0 − (a0 + b0)·x0 ≥ 0: x0 is at or below the mean.
    let mut lambda = beta_lambda(a, b, x, y);
    if lambda < 0.0 {
        ind = true;
        (a0, b0, x0, y0) = (b, a, y, x);
        lambda = lambda.abs();
    }
    // Not TOMS 708: Boost's ibeta_imp, which SciPy's betainc runs, relates integer shapes with
    // b < 40 to the binomial law at this point, I_x(a, b) = P(Bin(a + b − 1, x) ≥ a), a sum of
    // at most b positive terms. It is exact, and it is 2.6x cheaper than the two log-Beta
    // prefactors of bup + bpser below (frankenscipy-z3pk9).
    if b0 < 40.0
        && a0.fract() == 0.0
        && b0.fract() == 0.0
        && a0 < f64::from(i32::MAX - 100)
        && y0 != 1.0
    {
        let k = a0 - 1.0;
        return from_w(ind, binomial_ccdf(b0 + k, k, x0, y0));
    }
    if b0 < 40.0 && b0 * x0 <= 0.7 {
        return from_w(ind, bpser(a0, b0, x0, eps));
    }
    if b0 < 40.0 {
        // I_x0(a0, b0) = [I_x0(a0, b0 − n) −…] by BUP down to b0 − n ∈ (0, 1], then BPSER or
        // (BUP up in a0 and) BGRAT.
        let mut n = b0 as i32;
        b0 -= f64::from(n);
        if b0 == 0.0 {
            n -= 1;
            b0 = 1.0;
        }
        let mut w = bup(b0, a0, y0, x0, n, eps);
        if x0 <= 0.7 {
            w += bpser(a0, b0, x0, eps);
            return from_w(ind, w);
        }
        if a0 <= 15.0 {
            let n = 20;
            w += bup(a0, b0, x0, y0, n, eps);
            a0 += f64::from(n);
        }
        return from_w(ind, bgrat(a0, b0, x0, y0, w, 15.0 * eps));
    }
    // cdflib tests b0 when a0 > b0 and a0 otherwise: the smaller of the two.
    let smaller = a0.min(b0);
    if smaller <= 100.0 || lambda > 0.03 * smaller {
        from_w(ind, bfrac(a0, b0, x0, y0, lambda, 15.0 * eps))
    } else {
        from_w(ind, basym(a0, b0, lambda, 100.0 * eps))
    }
}

/// TOMS 708's `λ`: `(a + b)·y − b` when `a > b` and `a − (a + b)·x` otherwise, the
/// displacement of `x` from the mean in units of `1/(a + b)`.
///
/// cdflib forms it in plain arithmetic, whose rounding of `(a + b)·x` leaves an absolute
/// error of `ε·a` in `λ`. Near the mean `λ` is only about `√(ab/(a+b))`, so that is a relative
/// error of about `ε·√(a+b)` in `λ` and, through `BASYM` and `BRCOMP`'s `e^(−λ²/…)`, in
/// `I_x` itself: against mpmath at the exact double `x`, `a = b = 1e10` at 1 sd missed by 1e-11,
/// `1e15` by 1.5e-10 and `1e20` by 2e-7 (frankenscipy-5pnba). Here `a + b` is carried as an
/// exact two-sum and the product as an exact `fma` two-product, so the only roundings are
/// those of `λ`'s own size, and the same rows miss by 1e-14.
/// Boost's `binomial_ccdf(n, k, x, y)` (`special_functions/beta.hpp`): P(X > k) for
/// X ~ Binomial(n, x), with `y = 1 − x` passed exactly. The terms are summed from `x^n` down to
/// `i = k + 1`, each from the last by `C(n, i) / C(n, i + 1) = (i + 1) / (n − i)`. When `x^n`
/// underflows, the sum starts just above the mode and runs outwards. Boost takes the binomial
/// coefficient from an exact factorial table; [`binomial_coefficient`] here is exact up to its
/// final rounding.
///
/// Not Boost: when that starting term is not a product of normal numbers, the sum is formed by
/// [`binomial_ccdf_log_anchored`] instead (frankenscipy-xzrpr). Boost then added the terms one
/// by one as `x^i·y^(n−i)·C(n, i)`, and a factor that had already gone subnormal (or to 0) took
/// the result's digits with it: `bdtr(21, 64, 1 − 4.43e-8)` came out 4.59e-300 for 2.4754e-300,
/// and the results of `bdtr`, `bdtrc`, `nbdtr` and `nbdtrc` below about 1e-290 that take this
/// route were 0 or wrong in their leading digits (SciPy's `incbet` holds 1e-13 there).
fn binomial_ccdf(n: f64, k: f64, x: f64, y: f64) -> f64 {
    let mut result = x.powf(n);
    if result > f64::MIN_POSITIVE {
        let mut term = result;
        let mut i = n - 1.0;
        while i > k {
            term *= ((i + 1.0) * y) / ((n - i) * x);
            result += term;
            i -= 1.0;
        }
        return result;
    }
    let mut start = (n * x).trunc();
    if start <= k + 1.0 {
        start = (k + 2.0).trunc();
    }
    let x_pow = x.powf(start);
    let y_pow = y.powf(n - start);
    let xy_pow = x_pow * y_pow;
    result = xy_pow * binomial_coefficient(n, start);
    let normal = |v: f64| (f64::MIN_POSITIVE..=f64::MAX).contains(&v);
    if !(normal(x_pow) && normal(y_pow) && normal(xy_pow) && normal(result)) {
        return binomial_ccdf_log_anchored(n, k, x, y);
    }
    let start_term = result;
    let mut term = result;
    let mut i = start - 1.0;
    while i > k {
        term *= ((i + 1.0) * y) / ((n - i) * x);
        result += term;
        i -= 1.0;
    }
    term = start_term;
    let mut i = start + 1.0;
    while i <= n {
        term *= (n - i + 1.0) * x / (i * y);
        result += term;
        i += 1.0;
    }
    result
}

/// [`binomial_ccdf`]'s sum when its terms leave the normal range (frankenscipy-xzrpr).
///
/// [`crate::bratio::bratio`] only takes the binomial route with `x` at or below the mean
/// (`λ ≥ 0`), so `n·x < k + 1` and the largest term of the tail `i = k + 1, …, n` is its first
/// one, `t = C(n, k+1)·x^(k+1)·y^(n−k−1)`. The others enter relative to it through the ratio
/// recurrence, in the normal range: each ratio is below 1, so the relative terms fall
/// monotonically and the loop stops once they no longer move the sum.
///
/// `t` itself is formed with its binary exponent carried apart: `x = m_x·2^e_x` and
/// `y = m_y·2^e_y` with `m ∈ [½, 1)`, so `x^i = (m_x^i)·2^(i·e_x)` where `m_x^i` stays normal
/// for `i < 1000`. The mantissas multiply to a normal number and the exponents add exactly;
/// [`scale_by_pow2`] then rounds the result once, into the subnormal range if it is that small.
/// Every step is a correctly rounded `powf`, product or sum, so the result carries a few ulps.
/// Beyond that range (`k + 1` or `n − k − 1` of 1000 or more) `t` is `exp` of its log, whose
/// rounding is `|ln t|·ε`: about 1e-13 relative at the bottom of the normal range, what
/// SciPy's `incbet` holds there by the same `exp`.
fn binomial_ccdf_log_anchored(n: f64, k: f64, x: f64, y: f64) -> f64 {
    let i0 = k + 1.0;
    let j0 = n - i0;
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut i = i0 + 1.0;
    while i <= n {
        term *= (n - i + 1.0) * x / (i * y);
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
        i += 1.0;
    }
    let c = binomial_coefficient(n, i0);
    if i0 < 1000.0 && j0 < 1000.0 && c.is_finite() {
        let (mx, ex) = split_exponent(x);
        let (my, ey) = split_exponent(y);
        let (px, epx) = split_exponent(mx.powf(i0));
        let (py, epy) = split_exponent(my.powf(j0));
        // i0, j0 < 1000 and |e| ≤ 1074, so the products are exact integers.
        let exponent = (ex as f64 * i0) as i64 + (ey as f64 * j0) as i64 + epx + epy;
        return scale_by_pow2(c * px * py * sum, exponent);
    }
    let ln_first = ln_binomial_coefficient(n, i0) + i0 * x.ln() + j0 * y.ln();
    (ln_first + sum.ln()).exp()
}

/// `x = m·2^e` with `m ∈ [½, 1)`, for finite `x > 0` (C's `frexp`).
fn split_exponent(x: f64) -> (f64, i64) {
    // A subnormal x is first scaled into the normal range by 2^54.
    let (x, bias) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, 54)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1022;
    let m = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1022_u64 << 52));
    (m, e - bias)
}

/// `2^e` for `−1074 ≤ e ≤ 1023`, exactly (a subnormal below −1022).
fn pow2(e: i64) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1_u64 << (e + 1074))
    }
}

/// `m·2^e` for finite `m > 0`, rounded once (C's `ldexp`). A result in the normal range is
/// exact; a subnormal one is the single rounding of an exact normal number times an exact
/// power of two.
fn scale_by_pow2(m: f64, e: i64) -> f64 {
    let (mm, me) = split_exponent(m);
    let t = me + e; // the result is mm·2^t with mm ∈ [½, 1)
    if t > 1024 {
        return f64::INFINITY;
    }
    if t == 1024 {
        return (2.0 * mm) * pow2(1023);
    }
    if t >= -1021 {
        return mm * pow2(t);
    }
    let shift = -1021 - t; // mm·2^−1021 is normal; the multiply by 2^−shift rounds once
    if shift > 1074 {
        return 0.0;
    }
    (mm * pow2(-1021)) * pow2(-shift)
}

/// `ln C(n, k)` for integers `0 ≤ k ≤ n`: the log of [`binomial_coefficient`] while that is
/// finite, and otherwise the sum of the logs of its factors `(n − i)/(i + 1)`,
/// `i < min(k, n − k)`, which is short wherever [`binomial_ccdf`] is used.
fn ln_binomial_coefficient(n: f64, k: f64) -> f64 {
    let c = binomial_coefficient(n, k);
    if c.is_finite() {
        return c.ln();
    }
    let m = k.min(n - k);
    let mut sum = 0.0;
    let mut i = 0.0;
    while i < m {
        sum += ((n - i) / (i + 1.0)).ln();
        i += 1.0;
    }
    sum
}

/// C(n, k) for integers 0 ≤ k ≤ n (as f64). The recurrence C(n, i + 1) = C(n, i)·(n − i)/(i + 1)
/// runs exactly in u128 while the product fits, so the result is rounded once; past that it
/// continues in f64. In [`binomial_ccdf`] min(k, n − k) < 40, so the loop is short. A log-Γ
/// form (`gamma::comb` beyond 20 terms) was 4.7e-14 off at C(77, 37), which the
/// (39, 39, 1e-5) test row caught.
fn binomial_coefficient(n: f64, k: f64) -> f64 {
    let m = k.min(n - k) as u64;
    let n = n as u64;
    let mut exact: u128 = 1;
    let mut i = 0_u64;
    while i < m {
        let Some(product) = exact.checked_mul(u128::from(n - i)) else {
            break;
        };
        exact = product / u128::from(i + 1);
        i += 1;
    }
    let mut c = exact as f64;
    while i < m {
        c = c * (n - i) as f64 / (i + 1) as f64;
        i += 1;
    }
    c
}

fn beta_lambda(a: f64, b: f64, x: f64, y: f64) -> f64 {
    // a + b = s + es exactly (Knuth's two-sum).
    let s = a + b;
    let bv = s - a;
    let es = (a - (s - bv)) + (b - bv);
    if a > b {
        // s·y = p + ep exactly.
        let p = s * y;
        let ep = s.mul_add(y, -p);
        ((p - b) + ep) + es * y
    } else {
        let p = s * x;
        let ep = s.mul_add(x, -p);
        ((a - p) - ep) - es * x
    }
}

/// `ln(Γ(b) / Γ(a + b))` for `b ≥ 8`.
fn algdiv(a: f64, b: f64) -> f64 {
    const C: [f64; 6] = [
        0.833333333333333e-01,
        -0.277777777760991e-02,
        0.793650666825390e-03,
        -0.595202931351870e-03,
        0.837308034031215e-03,
        -0.165322962780713e-02,
    ];
    let (h, c, x, d);
    if a > b {
        h = b / a;
        c = 1.0 / (1.0 + h);
        x = h / (1.0 + h);
        d = a + (b - 0.5);
    } else {
        h = a / b;
        c = h / (1.0 + h);
        x = 1.0 / (1.0 + h);
        d = b + (a - 0.5);
    }
    // s_n = (1 − xⁿ)/(1 − x).
    let x2 = x * x;
    let s3 = 1.0 + (x + x2);
    let s5 = 1.0 + (x + x2 * s3);
    let s7 = 1.0 + (x + x2 * s5);
    let s9 = 1.0 + (x + x2 * s7);
    let s11 = 1.0 + (x + x2 * s9);
    // w = del(b) − del(a + b).
    let rb = 1.0 / b;
    let t = rb * rb;
    let mut w =
        (((((C[5] * s11) * t + C[4] * s9) * t + C[3] * s7) * t + C[2] * s5) * t + C[1] * s3) * t
            + C[0];
    w *= c / b;
    let u = d * alnrel(a / b);
    let v = a * (b.ln() - 1.0);
    if u > v { (w - v) - u } else { (w - u) - v }
}

/// `ln(1 + a)`.
fn alnrel(a: f64) -> f64 {
    const P: [f64; 3] = [
        -0.129418923021993e+01,
        0.405303492862024e+00,
        -0.178874546012214e-01,
    ];
    const Q: [f64; 3] = [
        -0.162752256355323e+01,
        0.747811014037616e+00,
        -0.845104217945565e-01,
    ];
    if a.abs() > 0.375 {
        return (1.0 + a).ln();
    }
    let t = a / (a + 2.0);
    let t2 = t * t;
    let mut w = ((P[2] * t2 + P[1]) * t2 + P[0]) * t2 + 1.0;
    w /= ((Q[2] * t2 + Q[1]) * t2 + Q[0]) * t2 + 1.0;
    2.0 * t * w
}

/// `I_{1−x}(b, a)` for `a ≤ min(ε, ε·b)`, `b·x ≤ 1` and `x ≤ 1/2`.
fn apser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    let g = 0.577215664901532860606512090082;
    let bx = b * x;
    let mut t = x - bx;
    let c = if b * eps > 0.02 {
        bx.ln() + g + t
    } else {
        x.ln() + psi(b) + g + t
    };
    let tol = 5.0 * eps * c.abs();
    let mut j = 1.0;
    let mut s = 0.0;
    let mut terms = 0;
    loop {
        j += 1.0;
        t *= x - bx / j;
        let aj = t / j;
        s += aj;
        if !(aj.abs() > tol) {
            break;
        }
        terms += 1;
        if terms >= MAX_TERMS {
            return f64::NAN;
        }
    }
    -a * (c + s)
}

/// Asymptotic expansion of `I_x(a, b)` for `a, b ≥ 15`, with `λ = (a + b)·y − b ≥ 0`.
fn basym(a: f64, b: f64, lambda: f64, eps: f64) -> f64 {
    // The a0, b0, c, d arrays hold num + 1 = 21 terms; num must be even.
    const NUM: usize = 20;
    let mut a0 = [0.0_f64; NUM + 1];
    let mut b0 = [0.0_f64; NUM + 1];
    let mut c = [0.0_f64; NUM + 1];
    let mut d = [0.0_f64; NUM + 1];
    let e0 = 2.0 / PI.sqrt();
    let e1 = 2.0_f64.powf(-3.0 / 2.0);

    let (h, r0, r1, w0);
    if a < b {
        h = a / b;
        r0 = 1.0 / (1.0 + h);
        r1 = (b - a) / b;
        w0 = 1.0 / (a * (1.0 + h)).sqrt();
    } else {
        h = b / a;
        r0 = 1.0 / (1.0 + h);
        r1 = (b - a) / a;
        w0 = 1.0 / (b * (1.0 + h)).sqrt();
    }
    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let t = (-f).exp();
    if t == 0.0 {
        return 0.0;
    }
    let z0 = f.sqrt();
    let z = 0.5 * (z0 / e1);
    let z2 = f + f;

    a0[0] = (2.0 / 3.0) * r1;
    c[0] = -0.5 * a0[0];
    d[0] = -c[0];
    let mut j0 = (0.5 / e0) * erfc1(true, z0);
    let mut j1 = e1;
    let mut ssum = j0 + d[0] * w0 * j1;

    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;

    for n in (2..=NUM).step_by(2) {
        let nf = n as f64;
        hn *= h2;
        a0[n - 1] = 2.0 * r0 * (1.0 + h * hn) / (nf + 2.0);
        s += hn;
        a0[n] = 2.0 * r1 * s / (nf + 3.0);

        for i in n..=n + 1 {
            let r = -0.5 * (i as f64 + 1.0);
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for j in 1..i {
                let imj = i - j;
                dsum += d[imj - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }
        j0 = e1 * znm1 + (nf - 1.0) * j0;
        j1 = e1 * zn + nf * j1;
        znm1 *= z2;
        zn *= z2;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[n] * w * j1;
        ssum += t0 + t1;
        if (t0.abs() + t1.abs()) <= eps * ssum {
            break;
        }
    }
    let u = (-bcorr(a, b)).exp();
    e0 * t * u * ssum
}

/// `del(a0) + del(b0) − del(a0 + b0)`, where `ln Γ(a) = (a − ½)·ln a − a + ½·ln(2π) + del(a)`,
/// for `a0, b0 ≥ 8`.
fn bcorr(a0: f64, b0: f64) -> f64 {
    const C: [f64; 6] = [
        0.833333333333333e-01,
        -0.277777777760991e-02,
        0.793650666825390e-03,
        -0.595202931351870e-03,
        0.837308034031215e-03,
        -0.165322962780713e-02,
    ];
    let a = a0.min(b0);
    let b = a0.max(b0);
    let h = a / b;
    let c = h / (1.0 + h);
    let x = 1.0 / (1.0 + h);
    let x2 = x * x;
    // s_n = (1 − xⁿ)/(1 − x).
    let s3 = 1.0 + (x + x2);
    let s5 = 1.0 + (x + x2 * s3);
    let s7 = 1.0 + (x + x2 * s5);
    let s9 = 1.0 + (x + x2 * s7);
    let s11 = 1.0 + (x + x2 * s9);
    // w = del(b) − del(a + b).
    let rb = 1.0 / b;
    let mut t = rb * rb;
    let mut w =
        (((((C[5] * s11) * t + C[4] * s9) * t + C[3] * s7) * t + C[2] * s5) * t + C[1] * s3) * t
            + C[0];
    w *= c / b;
    // del(a) + w.
    let ra = 1.0 / a;
    t = ra * ra;
    ((((((C[5]) * t + C[4]) * t + C[3]) * t + C[2]) * t + C[1]) * t + C[0]) / a + w
}

/// `ln B(a0, b0)`.
pub(crate) fn betaln(a0: f64, b0: f64) -> f64 {
    let e = 0.918938533204673;
    let mut a = a0.min(b0);
    let mut b = a0.max(b0);

    if a >= 8.0 {
        let w = bcorr(a, b);
        let h = a / b;
        let c = h / (1.0 + h);
        let u = -(a - 0.5) * c.ln();
        let v = b * alnrel(h);
        return if u > v {
            (((-0.5 * b.ln() + e) + w) - v) - u
        } else {
            (((-0.5 * b.ln() + e) + w) - u) - v
        };
    }
    if a < 1.0 {
        return if b > 8.0 {
            gamln(a) + algdiv(a, b)
        } else {
            gamln(a) + (gamln(b) - gamln(a + b))
        };
    }
    // 1 ≤ a < 8.
    let mut w = 0.0;
    if a <= 2.0 {
        if b <= 2.0 {
            return gamln(a) + gamln(b) - gsumln(a, b);
        }
        if b >= 8.0 {
            return gamln(a) + algdiv(a, b);
        }
    } else {
        // Reduce a to (1, 2] by the recurrence.
        let n = (a - 1.0) as i32;
        w = 1.0;
        if b <= 1000.0 {
            for _ in 0..n {
                a -= 1.0;
                let h = a / b;
                w *= h / (1.0 + h);
            }
            w = w.ln();
            if b >= 8.0 {
                return w + gamln(a) + algdiv(a, b);
            }
        } else {
            for _ in 0..n {
                a -= 1.0;
                w *= a / (1.0 + (a / b));
            }
            return (w.ln() - f64::from(n) * b.ln()) + (gamln(a) + algdiv(a, b));
        }
    }
    // Reduce b to (1, 2] by the recurrence.
    let n = (b - 1.0) as i32;
    let mut z = 1.0;
    for _ in 0..n {
        b -= 1.0;
        z *= b / (a + b);
    }
    w + z.ln() + (gamln(a) + gamln(b) - gsumln(a, b))
}

/// Continued fraction for `I_x(a, b)` when `a, b > 1`, with `λ = (a + b)·y − b`.
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64) -> f64 {
    let result = brcomp(a, b, x, y);
    if result == 0.0 {
        return 0.0;
    }
    result * bfrac_cf(a, b, x, y, lambda, eps)
}

/// The continued fraction of [`bfrac`], `I_x(a, b)` divided by [`brcomp`]'s `x^a·y^b / B(a, b)`.
fn bfrac_cf(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64) -> f64 {
    let c = 1.0 + lambda;
    let c0 = b / a;
    let c1 = 1.0 + (1.0 / a);
    let yp1 = y + 1.0;
    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;
    let mut terms = 0;
    loop {
        n += 1.0;
        let mut t = n / a;
        let w = n * (b - n) * x;
        let mut e = a / s;
        let alpha = (p * (p + c0) * e * e) * (w * x);
        e = (1.0 + t) / (c1 + t + t);
        let beta = n + (w / s) + e * (c + n * yp1);
        p = 1.0 + t;
        s += 2.0;
        // Update an, bn, anp1 and bnp1.
        t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;
        let r0 = r;
        r = anp1 / bnp1;
        if !((r - r0).abs() > eps * r) {
            break;
        }
        terms += 1;
        if terms >= MAX_TERMS {
            return f64::NAN;
        }
        // Rescale an, bn, anp1 and bnp1.
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    r
}

/// Asymptotic expansion for `I_x(a, b)` when `a` is larger than `b`, added to `w`. Assumes
/// `a ≥ 15` and `b ≤ 1`. cdflib also reports a status, which `bratio` ignores; a failed
/// expansion returns `w` unchanged, as there.
fn bgrat(a: f64, b: f64, x: f64, y: f64, w: f64, eps: f64) -> f64 {
    let bm1 = (b - 0.5) - 0.5;
    let nu = a + bm1 * 0.5;
    let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
    let z = -nu * lnx;

    if b * z == 0.0 {
        return w;
    }
    // r = e^(−z)·z^b / Γ(b).
    let mut r = b * (1.0 + gam1(b)) * (b * z.ln()).exp();
    r *= (a * lnx).exp() * (0.5 * bm1 * lnx).exp();
    let mut u = algdiv(b, a) + b * nu.ln();
    u = r * (-u).exp();
    if u == 0.0 {
        return w;
    }
    let (_, q) = grat1(b, z, r, eps);
    match bgrat_sum(b, z, lnx, nu, q / r, w / u, eps) {
        Some(ssum) => w + u * ssum,
        None => w,
    }
}

/// [`bgrat`]'s expansion `Σ dₙ·Jₙ`, from `J₀ = Q(b, z)/r` and with `l = w/u` in the stopping
/// test; `None` where the sum turns non-positive and cdflib gives up on the expansion.
/// `stdtrit` sums it too, with its own prefactor (frankenscipy-eiqnk).
pub(crate) fn bgrat_sum(
    b: f64,
    z: f64,
    lnx: f64,
    nu: f64,
    j0: f64,
    l: f64,
    eps: f64,
) -> Option<f64> {
    let mut c = [0.0_f64; 30];
    let mut d = [0.0_f64; 30];
    let bm1 = (b - 0.5) - 0.5;
    let rnu = 1.0 / nu;
    let v = 0.25 * (rnu * rnu);
    let t2 = 0.25 * lnx * lnx;
    let mut j = j0;
    let mut ssum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;

    for n in 1..=30_usize {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn *= 1.0 / (n2 * (n2 + 1.0));
        c[n - 1] = cn;
        let mut s = 0.0;
        if n > 1 {
            let mut coef = b - n as f64;
            for i in 1..n {
                s += coef * c[i - 1] * d[n - i - 1];
                coef += b;
            }
        }
        d[n - 1] = bm1 * cn + s / n as f64;
        let dj = d[n - 1] * j;
        ssum += dj;
        if ssum <= 0.0 {
            return None;
        }
        if !(dj.abs() > eps * (ssum + l)) {
            break;
        }
    }
    Some(ssum)
}

/// `ln I_x(a, b)` deep in the lower tail, where `I` underflows, by TOMS 708's own expansion for
/// the region in log form: `BGRAT` when `a ≥ 15` and `b ≤ 1`, and `ln BRCOMP + ln BFRAC`'s
/// continued fraction when `a, b > 1` and `x` is below the mean (frankenscipy-5pnba). `None`
/// outside those regions, or where the expansion does not apply.
///
/// These are the expansions `bratio` itself picks in the far tail. A Numerical Recipes
/// continued fraction in `x` loses to cancellation there once `a` is large and `x` near 1:
/// `1 − (a + b)·x/(a + 1)` is the difference of two numbers within `y` of 1. At the Student t
/// tail `df = 1e10`, `t = −40` that cost 4e-10 of `I`.
pub(crate) fn ln_bratio_lower_tail(a: f64, b: f64, x: f64, y: f64) -> Option<f64> {
    let eps = f64::EPSILON.max(1e-15);
    if !(a.is_finite() && b.is_finite() && x > 0.0 && y > 0.0) {
        return None;
    }
    if a >= 15.0 && b <= 1.0 {
        let bm1 = (b - 0.5) - 0.5;
        let nu = a + bm1 * 0.5;
        let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
        let z = -nu * lnx;
        // The far tail has z = −ν·ln x large; grat1's continued fraction then gives
        // Q(b, z) = r·an0, so J₀ = Q/r = an0 without forming the underflowing r. At b = 1/2,
        // grat1 takes Q = erfc(√z) instead, and Q/r = e^z·erfc(√z)·√π/√z.
        if !(z >= 1.1) {
            return None;
        }
        let j0 = if b == 0.5 {
            erfc1(true, z.sqrt()) * (PI / z).sqrt()
        } else {
            grat1_cf(b, z, 15.0 * eps)?
        };
        // u = r·Γ(a + b)/(Γ(a)·ν^b) as bgrat forms it, with r = z^b·e^(−z)/Γ(b), in logs.
        let ln_r = (b * (1.0 + gam1(b))).ln() + b * z.ln() - z;
        let ln_u = ln_r - (algdiv(b, a) + b * nu.ln());
        let ssum = bgrat_sum(b, z, lnx, nu, j0, 0.0, 15.0 * eps)?;
        return Some(ln_u + ssum.ln());
    }
    if a > 1.0 && b > 1.0 {
        let lambda = beta_lambda(a, b, x, y);
        if !(lambda >= 0.0) {
            return None;
        }
        let r = bfrac_cf(a, b, x, y, lambda, 15.0 * eps);
        if !(r > 0.0 && r.is_finite()) {
            return None;
        }
        return Some(ln_brcomp(a, b, x, y) + r.ln());
    }
    None
}

/// Power series for `I_x(a, b)` when `b ≤ 1` or `b·x ≤ 0.7`.
fn bpser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    // The factor x^a / (a·B(a, b)).
    let a0 = a.min(b);
    let result = if a0 < 1.0 {
        let mut b0 = a.max(b);
        if b0 <= 1.0 {
            let mut result = x.powf(a);
            if result == 0.0 {
                return 0.0;
            }
            let apb = a + b;
            let z = if apb > 1.0 {
                let u = a + b - 1.0;
                (1.0 + gam1(u)) / apb
            } else {
                1.0 + gam1(apb)
            };
            let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / z;
            result *= c * (b / apb);
            result
        } else if b0 < 8.0 {
            let mut u = gamln1(a0);
            let m = (b0 - 1.0) as i32;
            if m > 0 {
                let mut c = 1.0;
                for _ in 0..m {
                    b0 -= 1.0;
                    c *= b0 / (a0 + b0);
                }
                u += c.ln();
            }
            let z = a * x.ln() - u;
            b0 -= 1.0;
            let apb = a0 + b0;
            let t = if apb > 1.0 {
                let u = a0 + b0 - 1.0;
                (1.0 + gam1(u)) / apb
            } else {
                1.0 + gam1(apb)
            };
            z.exp() * (a0 / a) * (1.0 + gam1(b0)) / t
        } else {
            let u = gamln1(a0) + algdiv(a0, b0);
            let z = a * x.ln() - u;
            (a0 / a) * z.exp()
        }
    } else {
        let z = a * x.ln() - betaln(a, b);
        z.exp() / a
    };
    if result == 0.0 || a <= 0.1 * eps {
        return result;
    }
    // The series.
    let mut ssum = 0.0;
    let mut n = 0.0;
    let mut c = 1.0;
    let tol = eps / a;
    let mut terms = 0;
    loop {
        n += 1.0;
        c *= (0.5 + (0.5 - b / n)) * x;
        let w = c / (a + n);
        ssum += w;
        if !(w.abs() > tol) {
            break;
        }
        terms += 1;
        if terms >= MAX_TERMS {
            return f64::NAN;
        }
    }
    result * (1.0 + a * ssum)
}

/// `e^mu · x^a·y^b / B(a, b)`.
fn brcmp1(mu: i32, a: f64, b: f64, x: f64, y: f64) -> f64 {
    let r2pi = 1.0 / (2.0 * PI).sqrt();
    let a0 = a.min(b);
    if a0 >= 8.0 {
        let (x0, y0);
        if a > b {
            let h = b / a;
            x0 = 1.0 / (1.0 + h);
            y0 = h / (1.0 + h);
        } else {
            let h = a / b;
            x0 = h / (1.0 + h);
            y0 = 1.0 / (1.0 + h);
        }
        let lambda = beta_lambda(a, b, x, y);
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 {
            e - (x / x0).ln()
        } else {
            rlog1(e)
        };
        e = lambda / b;
        let v = if e.abs() > 0.6 {
            e - (y / y0).ln()
        } else {
            rlog1(e)
        };
        let z = esum(mu, -(a * u + b * v));
        return r2pi * (b * x0).sqrt() * z * (-bcorr(a, b)).exp();
    }
    let (lnx, lny) = if x <= 0.375 {
        (x.ln(), alnrel(-x))
    } else if y > 0.375 {
        (x.ln(), y.ln())
    } else {
        (alnrel(-y), y.ln())
    };
    let mut z = a * lnx + b * lny;

    if a0 >= 1.0 {
        z -= betaln(a, b);
        return esum(mu, z);
    }
    let mut b0 = a.max(b);
    if b0 >= 8.0 {
        let u = gamln1(a0) + algdiv(a0, b0);
        return a0 * esum(mu, z - u);
    }
    if b0 > 1.0 {
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 0..n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            let u = a0 + b0 - 1.0;
            (1.0 + gam1(u)) / apb
        } else {
            1.0 + gam1(apb)
        };
        return a0 * esum(mu, z) * (1.0 + gam1(b0)) / t;
    }
    // a0 < 1 and b0 ≤ 1. TOMS 708 scales by e^mu here too; SciPy's C writes exp(z), which is
    // the same because bratio never reaches this branch with mu ≠ 0 (bup sets mu = 708 only
    // for a ≥ 1 and a + b ≥ 1.1·(a + 1), so b > 1).
    let t = esum(mu, z);
    if t == 0.0 {
        return 0.0;
    }
    let apb = a + b;
    let zz = if apb > 1.0 {
        let u = a + b - 1.0;
        (1.0 + gam1(u)) / apb
    } else {
        1.0 + gam1(apb)
    };
    let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / zz;
    t * (a0 * c) / (1.0 + a0 / b0)
}

/// `x^a·y^b / B(a, b)`.
fn brcomp(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let r2pi = 1.0 / (2.0 * PI).sqrt();
    if x == 0.0 || y == 0.0 {
        return 0.0;
    }
    let a0 = a.min(b);
    if a0 >= 8.0 {
        let (x0, y0);
        if a > b {
            let h = b / a;
            x0 = 1.0 / (1.0 + h);
            y0 = h / (1.0 + h);
        } else {
            let h = a / b;
            x0 = h / (1.0 + h);
            y0 = 1.0 / (1.0 + h);
        }
        let lambda = beta_lambda(a, b, x, y);
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 {
            e - (x / x0).ln()
        } else {
            rlog1(e)
        };
        e = lambda / b;
        let v = if e.abs() > 0.6 {
            e - (y / y0).ln()
        } else {
            rlog1(e)
        };
        let z = (-(a * u + b * v)).exp();
        return r2pi * (b * x0).sqrt() * z * (-bcorr(a, b)).exp();
    }
    let (lnx, lny) = if x <= 0.375 {
        (x.ln(), alnrel(-x))
    } else if y > 0.375 {
        (x.ln(), y.ln())
    } else {
        (alnrel(-y), y.ln())
    };
    let mut z = a * lnx + b * lny;
    if a0 >= 1.0 {
        z -= betaln(a, b);
        return z.exp();
    }
    let mut b0 = a.max(b);
    if b0 >= 8.0 {
        let u = gamln1(a0) + algdiv(a0, b0);
        return a0 * (z - u).exp();
    }
    if b0 > 1.0 {
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 0..n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            let u = a0 + b0 - 1.0;
            (1.0 + gam1(u)) / apb
        } else {
            1.0 + gam1(apb)
        };
        return a0 * z.exp() * (1.0 + gam1(b0)) / t;
    }
    let t = z.exp();
    if t == 0.0 {
        return 0.0;
    }
    let apb = a + b;
    let zz = if apb > 1.0 {
        let u = a + b - 1.0;
        (1.0 + gam1(u)) / apb
    } else {
        1.0 + gam1(apb)
    };
    let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / zz;
    t * (a0 * c) / (1.0 + a0 / b0)
}

/// `ln(x^a·y^b / B(a, b))`: [`brcomp`] in log form, branch for branch, so it stays finite
/// where `brcomp` underflows. It is the front factor of the log incomplete beta in the deep
/// tail (frankenscipy-5pnba), where the log-space `a·ln x + b·ln y − ln B(a, b)` it replaces
/// cancels to `ε·a·ln a` once the parameters are large.
pub(crate) fn ln_brcomp(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let ln_r2pi = -0.5 * (2.0 * PI).ln();
    if x == 0.0 || y == 0.0 {
        return f64::NEG_INFINITY;
    }
    let a0 = a.min(b);
    if a0 >= 8.0 {
        let (x0, y0);
        if a > b {
            let h = b / a;
            x0 = 1.0 / (1.0 + h);
            y0 = h / (1.0 + h);
        } else {
            let h = a / b;
            x0 = h / (1.0 + h);
            y0 = 1.0 / (1.0 + h);
        }
        let lambda = beta_lambda(a, b, x, y);
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 {
            e - (x / x0).ln()
        } else {
            rlog1(e)
        };
        e = lambda / b;
        let v = if e.abs() > 0.6 {
            e - (y / y0).ln()
        } else {
            rlog1(e)
        };
        return ln_r2pi + 0.5 * (b * x0).ln() - (a * u + b * v) - bcorr(a, b);
    }
    let (lnx, lny) = if x <= 0.375 {
        (x.ln(), alnrel(-x))
    } else if y > 0.375 {
        (x.ln(), y.ln())
    } else {
        (alnrel(-y), y.ln())
    };
    let mut z = a * lnx + b * lny;
    if a0 >= 1.0 {
        return z - betaln(a, b);
    }
    let mut b0 = a.max(b);
    if b0 >= 8.0 {
        let u = gamln1(a0) + algdiv(a0, b0);
        return a0.ln() + (z - u);
    }
    if b0 > 1.0 {
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 0..n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u += c.ln();
        }
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            let u = a0 + b0 - 1.0;
            (1.0 + gam1(u)) / apb
        } else {
            1.0 + gam1(apb)
        };
        return a0.ln() + z + ((1.0 + gam1(b0)) / t).ln();
    }
    let apb = a + b;
    let zz = if apb > 1.0 {
        let u = a + b - 1.0;
        (1.0 + gam1(u)) / apb
    } else {
        1.0 + gam1(apb)
    };
    let c = (1.0 + gam1(a)) * (1.0 + gam1(b)) / zz;
    z + ((a0 * c) / (1.0 + a0 / b0)).ln()
}

/// `I_x(a, b) − I_x(a + n, b)` for a positive integer `n`.
fn bup(a: f64, b: f64, x: f64, y: f64, n: i32, eps: f64) -> f64 {
    let apb = a + b;
    let ap1 = a + 1.0;
    let mut d = 1.0;
    let mut mu = 0;
    // The scaling factor e^(−mu), and e^mu·(x^a·y^b / B(a, b)) / a.
    if !(n == 1 || a < 1.0 || apb < 1.1 * ap1) {
        mu = 708;
        d = (-708.0_f64).exp();
    }
    let result = brcmp1(mu, a, b, x, y) / a;
    if n == 1 || result == 0.0 {
        return result;
    }
    let nm1 = n - 1;
    let mut w = d;
    let term = |d: f64, i: i32| d * (((apb + f64::from(i)) / (ap1 + f64::from(i))) * x);

    // k is the index of the largest term.
    let k = if b <= 1.0 {
        0
    } else if y > 1e-4 {
        let r = (b - 1.0) * x / y - a;
        if r < 1.0 {
            0
        } else if r < f64::from(nm1) {
            r as i32
        } else {
            nm1
        }
    } else {
        nm1
    };
    // The increasing terms of the series.
    for i in 0..k {
        d = term(d, i);
        w += d;
    }
    if k == nm1 {
        return result * w;
    }
    // The remaining terms.
    for i in k..nm1 {
        d = term(d, i);
        w += d;
        if d <= eps * w {
            break;
        }
    }
    result * w
}

/// The real error function.
fn erf(x: f64) -> f64 {
    let c = 0.564189583547756;
    const A: [f64; 5] = [
        0.771058495001320e-04,
        -0.133733772997339e-02,
        0.323076579225834e-01,
        0.479137145607681e-01,
        0.128379167095513e+00,
    ];
    const B: [f64; 3] = [
        0.301048631703895e-02,
        0.538971687740286e-01,
        0.375795757275549e+00,
    ];
    const P: [f64; 8] = [
        -1.36864857382717e-07,
        5.64195517478974e-01,
        7.21175825088309e+00,
        4.31622272220567e+01,
        1.52989285046940e+02,
        3.39320816734344e+02,
        4.51918953711873e+02,
        3.00459261020162e+02,
    ];
    const Q: [f64; 8] = [
        1.00000000000000e+00,
        1.27827273196294e+01,
        7.70001529352295e+01,
        2.77585444743988e+02,
        6.38980264465631e+02,
        9.31354094850610e+02,
        7.90950925327898e+02,
        3.00459260956983e+02,
    ];
    const R: [f64; 5] = [
        2.10144126479064e+00,
        2.62370141675169e+01,
        2.13688200555087e+01,
        4.65807828718470e+00,
        2.82094791773523e-01,
    ];
    const S: [f64; 4] = [
        9.41537750555460e+01,
        1.87114811799590e+02,
        9.90191814623914e+01,
        1.80124575948747e+01,
    ];
    let ax = x.abs();
    if ax <= 0.5 {
        let t = x * x;
        let top = (((A[0] * t + A[1]) * t + A[2]) * t + A[3]) * t + A[4] + 1.0;
        let bot = ((B[0] * t + B[1]) * t + B[2]) * t + 1.0;
        return x * (top / bot);
    }
    if ax <= 4.0 {
        let top = ((((((P[0] * ax + P[1]) * ax + P[2]) * ax + P[3]) * ax + P[4]) * ax + P[5]) * ax
            + P[6])
            * ax
            + P[7];
        let bot = ((((((Q[0] * ax + Q[1]) * ax + Q[2]) * ax + Q[3]) * ax + Q[4]) * ax + Q[5]) * ax
            + Q[6])
            * ax
            + Q[7];
        let t = 0.5 + (0.5 - (-x * x).exp() * (top / bot));
        return if x < 0.0 { -t } else { t };
    }
    if ax < 5.8 {
        let rx = 1.0 / x;
        let t = rx * rx;
        let top = (((R[0] * t + R[1]) * t + R[2]) * t + R[3]) * t + R[4];
        let bot = (((S[0] * t + S[1]) * t + S[2]) * t + S[3]) * t + 1.0;
        let t = 0.5 + (0.5 - (-x * x).exp() * (c - top / (x * x * bot)) / ax);
        return if x < 0.0 { -t } else { t };
    }
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// `erfc(x)`, or `e^(x²)·erfc(x)` when `scaled`.
pub(crate) fn erfc1(scaled: bool, x: f64) -> f64 {
    let c = 0.564189583547756;
    const A: [f64; 5] = [
        0.771058495001320e-04,
        -0.133733772997339e-02,
        0.323076579225834e-01,
        0.479137145607681e-01,
        0.128379167095513e+00,
    ];
    const B: [f64; 3] = [
        0.301048631703895e-02,
        0.538971687740286e-01,
        0.375795757275549e+00,
    ];
    const P: [f64; 8] = [
        -1.36864857382717e-07,
        5.64195517478974e-01,
        7.21175825088309e+00,
        4.31622272220567e+01,
        1.52989285046940e+02,
        3.39320816734344e+02,
        4.51918953711873e+02,
        3.00459261020162e+02,
    ];
    const Q: [f64; 8] = [
        1.00000000000000e+00,
        1.27827273196294e+01,
        7.70001529352295e+01,
        2.77585444743988e+02,
        6.38980264465631e+02,
        9.31354094850610e+02,
        7.90950925327898e+02,
        3.00459260956983e+02,
    ];
    const R: [f64; 5] = [
        2.10144126479064e+00,
        2.62370141675169e+01,
        2.13688200555087e+01,
        4.65807828718470e+00,
        2.82094791773523e-01,
    ];
    const S: [f64; 4] = [
        9.41537750555460e+01,
        1.87114811799590e+02,
        9.90191814623914e+01,
        1.80124575948747e+01,
    ];

    if x <= -5.6 {
        return if scaled { 2.0 * (x * x).exp() } else { 2.0 };
    }
    // sqrt(ln(f64::MAX)) ≈ 26.64.
    if !scaled && x > 26.64 {
        return 0.0;
    }
    let ax = x.abs();
    let mut result;
    if ax <= 0.5 {
        let t = x * x;
        let top = (((A[0] * t + A[1]) * t + A[2]) * t + A[3]) * t + A[4] + 1.0;
        let bot = ((B[0] * t + B[1]) * t + B[2]) * t + 1.0;
        result = 0.5 + (0.5 - x * (top / bot));
        return if scaled { result * t.exp() } else { result };
    } else if ax <= 4.0 {
        let top = ((((((P[0] * ax + P[1]) * ax + P[2]) * ax + P[3]) * ax + P[4]) * ax + P[5]) * ax
            + P[6])
            * ax
            + P[7];
        let bot = ((((((Q[0] * ax + Q[1]) * ax + Q[2]) * ax + Q[3]) * ax + Q[4]) * ax + Q[5]) * ax
            + Q[6])
            * ax
            + Q[7];
        result = top / bot;
    } else {
        let rx = 1.0 / x;
        let t = rx * rx;
        let top = (((R[0] * t + R[1]) * t + R[2]) * t + R[3]) * t + R[4];
        let bot = (((S[0] * t + S[1]) * t + S[2]) * t + S[3]) * t + 1.0;
        result = (c - t * (top / bot)) / ax;
    }
    if scaled {
        if x < 0.0 {
            2.0 * (x * x).exp() - result
        } else {
            result
        }
    } else {
        result *= (-(x * x)).exp();
        if x < 0.0 { 2.0 - result } else { result }
    }
}

/// `e^(mu + x)`, formed so that neither factor overflows when the other is large.
fn esum(mu: i32, x: f64) -> f64 {
    let m = f64::from(mu);
    if x > 0.0 {
        if mu > 0 || m + x < 0.0 {
            m.exp() * x.exp()
        } else {
            (m + x).exp()
        }
    } else if mu < 0 || m + x > 0.0 {
        m.exp() * x.exp()
    } else {
        (m + x).exp()
    }
}

/// `I_x(a, b)` for `b < min(ε, ε·a)` and `x ≤ 1/2`.
fn fpser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    let mut result = 1.0;
    if !(a <= 1e-3 * eps) {
        let t = a * x.ln();
        if t < -708.0 {
            return 0.0;
        }
        result = t.exp();
    }
    // 1/B(a, b) = b.
    result *= b / a;
    let tol = eps / a;
    let mut an = a + 1.0;
    let mut t = x;
    let mut s = t / an;
    let mut terms = 0;
    loop {
        an += 1.0;
        t *= x;
        let c = t / an;
        s += c;
        if !(c.abs() > tol) {
            break;
        }
        terms += 1;
        if terms >= MAX_TERMS {
            return f64::NAN;
        }
    }
    result * (1.0 + a * s)
}

/// `1/Γ(a + 1) − 1` for `−0.5 ≤ a ≤ 1.5`.
fn gam1(a: f64) -> f64 {
    const P: [f64; 7] = [
        0.577215664901533e+00,
        -0.409078193005776e+00,
        -0.230975380857675e+00,
        0.597275330452234e-01,
        0.766968181649490e-02,
        -0.514889771323592e-02,
        0.589597428611429e-03,
    ];
    const Q: [f64; 5] = [
        0.100000000000000e+01,
        0.427569613095214e+00,
        0.158451672430138e+00,
        0.261132021441447e-01,
        0.423244297896961e-02,
    ];
    const R: [f64; 9] = [
        -0.422784335098468e+00,
        -0.771330383816272e+00,
        -0.244757765222226e+00,
        0.118378989872749e+00,
        0.930357293360349e-03,
        -0.118290993445146e-01,
        0.223047661158249e-02,
        0.266505979058923e-03,
        -0.132674909766242e-03,
    ];
    const S: [f64; 2] = [0.273076135303957e+00, 0.559398236957378e-01];

    let d = a - 0.5;
    let t = if d > 0.0 { d - 0.5 } else { a };
    if t == 0.0 {
        return 0.0;
    }
    if t < 0.0 {
        let top = (((((((R[8] * t + R[7]) * t + R[6]) * t + R[5]) * t + R[4]) * t + R[3]) * t
            + R[2])
            * t
            + R[1])
            * t
            + R[0];
        let bot = (S[1] * t + S[0]) * t + 1.0;
        let w = top / bot;
        return if d > 0.0 {
            t * w / a
        } else {
            a * ((w + 0.5) + 0.5)
        };
    }
    let top = (((((P[6] * t + P[5]) * t + P[4]) * t + P[3]) * t + P[2]) * t + P[1]) * t + P[0];
    let bot = (((Q[4] * t + Q[3]) * t + Q[2]) * t + Q[1]) * t + 1.0;
    let w = top / bot;
    if d > 0.0 {
        (t / a) * ((w - 0.5) - 0.5)
    } else {
        a * w
    }
}

/// `ln Γ(a)` for `a > 0`.
fn gamln(a: f64) -> f64 {
    let d = 0.418938533204673;
    const C: [f64; 6] = [
        0.833333333333333e-01,
        -0.277777777760991e-02,
        0.793650666825390e-03,
        -0.595202931351870e-03,
        0.837308034031215e-03,
        -0.165322962780713e-02,
    ];
    if a <= 0.8 {
        return gamln1(a) - a.ln();
    }
    if a <= 2.25 {
        let t = (a - 0.5) - 0.5;
        return gamln1(t);
    }
    if a < 10.0 {
        let n = (a - 1.25) as i32;
        let mut t = a;
        let mut w = 1.0;
        for _ in 0..n {
            t -= 1.0;
            w *= t;
        }
        return gamln1(t - 1.0) + w.ln();
    }
    let ra = 1.0 / a;
    let t = ra * ra;
    let w = (((((C[5] * t + C[4]) * t + C[3]) * t + C[2]) * t + C[1]) * t + C[0]) / a;
    (d + w) + (a - 0.5) * (a.ln() - 1.0)
}

/// `ln Γ(1 + a)` for `−0.2 ≤ a ≤ 1.25`.
fn gamln1(a: f64) -> f64 {
    const P: [f64; 7] = [
        0.577215664901533e+00,
        0.844203922187225e+00,
        -0.168860593646662e+00,
        -0.780427615533591e+00,
        -0.402055799310489e+00,
        -0.673562214325671e-01,
        -0.271935708322958e-02,
    ];
    const Q: [f64; 6] = [
        0.288743195473681e+01,
        0.312755088914843e+01,
        0.156875193295039e+01,
        0.361951990101499e+00,
        0.325038868253937e-01,
        0.667465618796164e-03,
    ];
    const R: [f64; 6] = [
        0.422784335098467e+00,
        0.848044614534529e+00,
        0.565221050691933e+00,
        0.156513060486551e+00,
        0.170502484022650e-01,
        0.497958207639485e-03,
    ];
    const S: [f64; 5] = [
        0.124313399877507e+01,
        0.548042109832463e+00,
        0.101552187439830e+00,
        0.713309612391000e-02,
        0.116165475989616e-03,
    ];
    if a < 0.6 {
        let top = (((((P[6] * a + P[5]) * a + P[4]) * a + P[3]) * a + P[2]) * a + P[1]) * a + P[0];
        let bot = (((((Q[5] * a + Q[4]) * a + Q[3]) * a + Q[2]) * a + Q[1]) * a + Q[0]) * a + 1.0;
        let w = top / bot;
        return -a * w;
    }
    let x = (a - 0.5) - 0.5;
    let top = ((((R[5] * x + R[4]) * x + R[3]) * x + R[2]) * x + R[1]) * x + R[0];
    let bot = ((((S[4] * x + S[3]) * x + S[2]) * x + S[1]) * x + S[0]) * x + 1.0;
    let w = top / bot;
    x * w
}

/// The incomplete gamma ratios `(P(a, x), Q(a, x))` for `a ≤ 1`, given
/// `r = e^(−x)·x^a / Γ(a)`.
fn grat1(a: f64, x: f64, r: f64, eps: f64) -> (f64, f64) {
    if a * x == 0.0 {
        return if x > a { (1.0, 0.0) } else { (0.0, 1.0) };
    }
    if a == 0.5 {
        if x < 0.25 {
            let p = erf(x.sqrt());
            return (p, 0.5 + (0.5 - p));
        }
        let q = erfc1(false, x.sqrt());
        return (0.5 + (0.5 - q), q);
    }
    if x < 1.1 {
        // Taylor series for P(a, x) / x^a.
        let mut an = 3.0;
        let mut c = x;
        let mut ssum = x / (a + 3.0);
        let tol = 0.1 * eps / (a + 1.0);
        let mut terms = 0;
        loop {
            an += 1.0;
            c *= -(x / an);
            let t = c / (a + an);
            ssum += t;
            if !(t.abs() > tol) {
                break;
            }
            terms += 1;
            if terms >= MAX_TERMS {
                return (f64::NAN, f64::NAN);
            }
        }
        let j = a * x * ((ssum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
        let z = a * x.ln();
        let h = gam1(a);
        let g = 1.0 + h;

        if (x >= 0.25 && a >= x / 2.59) || (x < 0.25 && z <= -0.13394) {
            let w = z.exp();
            let p = w * g * (0.5 + (0.5 - j));
            let q = 0.5 + (0.5 - p);
            return (p, q);
        }
        let l = rexp(z);
        let w = 0.5 + (0.5 + l);
        let q = (w * j - l) * g - h;
        if q < 0.0 {
            return (1.0, 0.0);
        }
        let p = 0.5 + (0.5 - q);
        return (p, q);
    }
    // Continued fraction.
    let Some(an0) = grat1_cf(a, x, eps) else {
        return (f64::NAN, f64::NAN);
    };
    let q = r * an0;
    let p = 0.5 + (0.5 - q);
    (p, q)
}

/// [`grat1`]'s continued fraction for `x ≥ 1.1`: `Q(a, x) / r` with `r = e^(−x)·x^a / Γ(a)`.
/// `None` if it has not converged within [`MAX_TERMS`].
fn grat1_cf(a: f64, x: f64, eps: f64) -> Option<f64> {
    let mut a2nm1 = 1.0;
    let mut a2n = 1.0;
    let mut b2nm1 = x;
    let mut b2n = x + (1.0 - a);
    let mut c = 1.0;
    let mut terms = 0;
    loop {
        a2nm1 = x * a2n + c * a2nm1;
        b2nm1 = x * b2n + c * b2nm1;
        let am0 = a2nm1 / b2nm1;
        c += 1.0;
        let cma = c - a;
        a2n = a2nm1 + cma * a2n;
        b2n = b2nm1 + cma * b2n;
        let an0 = a2n / b2n;
        if !((an0 - am0).abs() >= eps * an0) {
            return Some(an0);
        }
        terms += 1;
        if terms >= MAX_TERMS {
            return None;
        }
    }
}

/// `ln Γ(a + b)` for `1 ≤ a ≤ 2` and `1 ≤ b ≤ 2`.
fn gsumln(a: f64, b: f64) -> f64 {
    let x = a + b - 2.0;
    if x <= 0.25 {
        return gamln1(1.0 + x);
    }
    if x <= 1.25 {
        return gamln1(x) + alnrel(x);
    }
    gamln1(x - 1.0) + (x * (1.0 + x)).ln()
}

/// The digamma function (Cody, Strecok and Thacher, Math. Comp. 27, 123-127, 1973, as
/// modified by A. H. Morris). `0` where it cannot be computed.
fn psi(xx: f64) -> f64 {
    const P1: [f64; 7] = [
        0.895385022981970e-02,
        0.477762828042627e+01,
        0.142441585084029e+03,
        0.118645200713425e+04,
        0.363351846806499e+04,
        0.413810161269013e+04,
        0.130560269827897e+04,
    ];
    const Q1: [f64; 6] = [
        0.448452573429826e+02,
        0.520752771467162e+03,
        0.221000799247830e+04,
        0.364127349079381e+04,
        0.190831076596300e+04,
        0.691091682714533e-05,
    ];
    const P2: [f64; 4] = [
        -0.212940445131011e+01,
        -0.701677227766759e+01,
        -0.448616543918019e+01,
        -0.648157123766197e+00,
    ];
    const Q2: [f64; 4] = [
        0.322703493791143e+02,
        0.892920700481861e+02,
        0.546117738103215e+02,
        0.777788548522962e+01,
    ];
    let dx0 = 1.461632144968362341262659542325721325;
    let xmax1 = 4503599627370496.0;
    let xsmall = 1e-9;
    let mut x = xx;
    let mut aug = 0.0;

    if x < 0.5 {
        if x.abs() <= xsmall {
            if x == 0.0 {
                return 0.0;
            }
            aug = -1.0 / x;
        } else {
            // Reflection: aug = −π·cot(π·x), from the reduced argument.
            let mut w = -x;
            let mut sgn = PI / 4.0;
            if w <= 0.0 {
                w = -w;
                sgn = -sgn;
            }
            if w >= xmax1 {
                return 0.0;
            }
            w -= w.trunc();
            let nq = (w * 4.0) as i32;
            w = 4.0 * (w - 0.25 * f64::from(nq));
            if nq % 2 == 1 {
                w = 1.0 - w;
            }
            let z = (PI / 4.0) * w;
            if (nq / 2) % 2 == 1 {
                sgn = -sgn;
            }
            if ((nq + 1) / 2) % 2 == 1 {
                aug = sgn * (z.tan() * 4.0);
            } else {
                if z == 0.0 {
                    return 0.0;
                }
                aug = sgn * (4.0 / z.tan());
            }
        }
        x = 1.0 - x;
    }
    if x <= 3.0 {
        let mut den = x;
        let mut upper = P1[0] * x;
        for i in 0..5 {
            den = (den + Q1[i]) * x;
            upper = (upper + P1[i + 1]) * x;
        }
        den = (upper + P1[6]) / (den + Q1[5]);
        let xmx0 = x - dx0;
        return (den * xmx0) + aug;
    }
    if x < xmax1 {
        let w = 1.0 / (x * x);
        let mut den = w;
        let mut upper = P2[0] * w;
        for i in 0..3 {
            den = (den + Q2[i]) * w;
            upper = (upper + P2[i + 1]) * w;
        }
        aug += upper / (den + Q2[3]) - 0.5 / x;
    }
    aug + x.ln()
}

/// `e^x − 1`.
fn rexp(x: f64) -> f64 {
    const P: [f64; 2] = [0.914041914819518e-09, 0.238082361044469e-01];
    const Q: [f64; 4] = [
        -0.499999999085958e+00,
        0.107141568980644e+00,
        -0.119041179760821e-01,
        0.595130811860248e-03,
    ];
    if x.abs() <= 0.15 {
        return x
            * (((P[1] * x + P[0]) * x + 1.0)
                / ((((Q[3] * x + Q[2]) * x + Q[1]) * x + Q[0]) * x + 1.0));
    }
    let w = x.exp();
    if x > 0.0 {
        w * (0.5 + (0.5 - 1.0 / w))
    } else {
        (w - 0.5) - 0.5
    }
}

/// `x − ln(1 + x)`.
fn rlog1(x: f64) -> f64 {
    let a = 0.566749439387324e-01;
    let b = 0.456512608815524e-01;
    let p0 = 0.333333333333333e+00;
    let p1 = -0.224696413112536e+00;
    let p2 = 0.620886815375787e-02;
    let q1 = -0.127408923933623e+01;
    let q2 = 0.354508718369557e+00;

    if !(-0.39..=0.57).contains(&x) {
        return x - ((x + 0.5) + 0.5).ln();
    }
    let (h, w1) = if (-0.18..=0.18).contains(&x) {
        (x, 0.0)
    } else if x < -0.18 {
        let h = (x + 0.3) / 0.7;
        (h, a - h * 0.3)
    } else {
        // 0.18 < x ≤ 0.57.
        let h = 0.75 * x - 0.25;
        (h, b + h / 3.0)
    };
    let r = h / (h + 2.0);
    let t = r * r;
    let w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.0);
    2.0 * t * (1.0 / (1.0 - r) - r * w) + w1
}
