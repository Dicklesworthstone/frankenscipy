//! Metamorphic tests for `fsci-special`.
//!
//! Recurrence Γ(x+1) = x Γ(x), gamma/factorial agreement, beta
//! symmetry, ellipk(0) = π/2, gammaln consistency, Bessel symmetry.
//!
//! Run with: `cargo test -p fsci-special --test metamorphic_tests`

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialResult, SpecialTensor, agm, ai, arctanh, bei, ber, bernoulli, beta, betaln, bi,
    boxcox_transform_scalar, chdtr, chdtrc, comb, digamma_scalar, ellipe, ellipeinc, ellipj,
    ellipk, ellipkinc, ellipkm1, elliprc, elliprd, elliprf, elliprg, entr, erf_scalar, exp1, expi,
    expit, expn_scalar, factorial, factorial2, fresnel, gamma, gammainc, gammaincc, gammaln,
    hyp1f1, hyp2f1, i0, i0_scalar, inv_boxcox_scalar, j0, jn, jv, lambertw_scalar, logit,
    orthopoly::{
        eval_chebyt, eval_chebyu, eval_gegenbauer, eval_genlaguerre, eval_hermite,
        eval_hermitenorm, eval_jacobi, eval_laguerre, eval_legendre, eval_sh_legendre, lpmv,
        roots_chebyt, roots_legendre, sph_harm,
    },
    perm, rgamma, shichi, sici, struve, xlog1py, xlogy, zeta_scalar,
};

const ATOL: f64 = 1e-10;
const RTOL: f64 = 1e-9;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

fn real(value: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(value)
}

fn unwrap_real(result: SpecialResult) -> f64 {
    match result {
        Ok(SpecialTensor::RealScalar(v)) => v,
        Ok(_) | Err(_) => f64::NAN,
    }
}

fn gamma_real(x: f64) -> f64 {
    unwrap_real(gamma(&real(x), RuntimeMode::Strict))
}

fn gammainc_real(a: f64, x: f64) -> f64 {
    unwrap_real(gammainc(&real(a), &real(x), RuntimeMode::Strict))
}

fn gammaincc_real(a: f64, x: f64) -> f64 {
    unwrap_real(gammaincc(&real(a), &real(x), RuntimeMode::Strict))
}

fn hyp2f1_real(a: f64, b: f64, c: f64, z: f64) -> f64 {
    unwrap_real(hyp2f1(
        &real(a),
        &real(b),
        &real(c),
        &real(z),
        RuntimeMode::Strict,
    ))
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — Γ(x + 1) = x · Γ(x) for various non-integer x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_recurrence() {
    for x in [0.3_f64, 1.7, 2.4, 3.5, 4.2, 5.8, 7.1] {
        let g_x = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let g_x1 = unwrap_real(gamma(&real(x + 1.0), RuntimeMode::Strict));
        assert!(
            close(g_x1, x * g_x),
            "MR1 recurrence at x={x}: Γ(x+1)={g_x1} vs x·Γ(x)={}",
            x * g_x
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — Γ(n+1) = n! for non-negative integers.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_matches_factorial() {
    for n in [0u64, 1, 2, 3, 5, 8, 15, 20] {
        let g = unwrap_real(gamma(&real((n + 1) as f64), RuntimeMode::Strict));
        let f = factorial(n);
        let rel_err = ((g - f) / f.abs().max(1.0)).abs();
        assert!(
            rel_err < 1e-9,
            "MR2 Γ({}) = {g}, expected n! = {f}, rel_err={rel_err:e}",
            n + 1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — beta(a, b) = beta(b, a) (symmetry).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_beta_symmetric() {
    let pairs: [(f64, f64); 5] = [(1.5, 2.5), (3.0, 0.7), (5.5, 4.5), (0.1, 0.9), (10.0, 2.0)];
    for (a, b) in pairs {
        let bab = unwrap_real(beta(&real(a), &real(b), RuntimeMode::Strict));
        let bba = unwrap_real(beta(&real(b), &real(a), RuntimeMode::Strict));
        assert!(
            close(bab, bba),
            "MR3 beta symmetry: B({a},{b})={bab} vs B({b},{a})={bba}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — ellipk(0) = π/2 and ellipk(m) ≥ π/2 for m ∈ [0, 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipk_endpoint_and_monotonicity() {
    let k0 = unwrap_real(ellipk(&real(0.0), RuntimeMode::Strict));
    assert!(close(k0, PI / 2.0), "MR4 ellipk(0)={k0}, expected π/2");
    let mut prev = k0;
    for k in 1..50 {
        let m = k as f64 / 50.0;
        let v = unwrap_real(ellipk(&real(m), RuntimeMode::Strict));
        assert!(
            v >= prev - 1e-12,
            "MR4 ellipk not monotone at m={m}: {v} < prev={prev}"
        );
        assert!(v >= PI / 2.0 - 1e-12, "MR4 ellipk({m}) below π/2: {v}");
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — gammaln consistency: gammaln(x) ≈ ln(|gamma(x)|) for x > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammaln_matches_log_gamma() {
    for x in [0.5_f64, 1.0, 2.0, 3.5, 7.0, 12.5, 50.0] {
        let g = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let lg = unwrap_real(gammaln(&real(x), RuntimeMode::Strict));
        let direct = g.abs().ln();
        let rel_err = (lg - direct).abs() / direct.abs().max(1.0);
        assert!(
            rel_err < 1e-9,
            "MR5 gammaln at x={x}: lg={lg}, ln|Γ|={direct}, rel={rel_err:e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — Bessel j0(0) = 1 (zeroth-order Bessel at the origin).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bessel_j0_at_zero_is_one() {
    let v = unwrap_real(j0(&real(0.0), RuntimeMode::Strict));
    assert!(close(v, 1.0), "MR6 j0(0) = {v}, expected 1");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — j_n is even for even n, odd for odd n.
//   j_n(-x) = (-1)^n j_n(x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jn_parity() {
    for n in 0..6_i64 {
        for x in [0.5_f64, 1.5, 3.7, 5.0] {
            let pos = unwrap_real(jn(&real(n as f64), &real(x), RuntimeMode::Strict));
            let neg = unwrap_real(jn(&real(n as f64), &real(-x), RuntimeMode::Strict));
            let expected = if n % 2 == 0 { pos } else { -pos };
            assert!(
                close(neg, expected),
                "MR7 j_{n}({x}) parity: j_n(-x)={neg}, expected {expected}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — Modified Bessel i0(0) = 1 and i0(x) ≥ 1 for x ∈ R (i0 is even
//        and convex with minimum 1 at 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_i0_minimum_at_zero() {
    let v0 = unwrap_real(i0(&real(0.0), RuntimeMode::Strict));
    assert!(close(v0, 1.0), "MR8 i0(0) = {v0}, expected 1");
    for x in [0.1_f64, -0.5, 1.0, -2.5, 5.0, -3.7] {
        let v = i0_scalar(x);
        assert!(
            v >= 1.0 - 1e-12,
            "MR8 i0({x}) = {v} < 1 (i0 has minimum 1 at zero)"
        );
        // i0 is even.
        let v_neg = i0_scalar(-x);
        assert!(
            close(v, v_neg),
            "MR8 i0 not even at x={x}: i0(x)={v}, i0(-x)={v_neg}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — Multiplication formula: Γ(x) · Γ(1-x) · sin(πx) = π
// (reflection formula).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gamma_reflection_formula() {
    for x in [0.3_f64, 0.4, 0.7, 0.9] {
        let g_x = unwrap_real(gamma(&real(x), RuntimeMode::Strict));
        let g_1mx = unwrap_real(gamma(&real(1.0 - x), RuntimeMode::Strict));
        let product = g_x * g_1mx * (PI * x).sin();
        assert!(
            close(product, PI),
            "MR9 reflection at x={x}: Γ(x)·Γ(1-x)·sin(πx) = {product}, expected π"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Orthogonal polynomial relations
// ─────────────────────────────────────────────────────────────────────

// MR10 — Legendre P_n(1) = 1 and P_n(-1) = (-1)^n.
#[test]
fn mr_legendre_endpoints() {
    for n in 0..=10_u32 {
        let p_at_1 = eval_legendre(n, 1.0);
        assert!(close(p_at_1, 1.0), "MR10 P_{n}(1) = {p_at_1}, expected 1");
        let p_at_neg1 = eval_legendre(n, -1.0);
        let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            close(p_at_neg1, expected),
            "MR10 P_{n}(-1) = {p_at_neg1}, expected {expected}"
        );
    }
}

// MR11 — Chebyshev T_n(cos θ) = cos(n θ).
#[test]
fn mr_chebyt_cos_relation() {
    for n in 0..=8_u32 {
        for k in 0..=20 {
            let theta = k as f64 * PI / 20.0;
            let x = theta.cos();
            let t = eval_chebyt(n, x);
            let expected = (n as f64 * theta).cos();
            assert!(
                close(t, expected),
                "MR11 T_{n}(cos {theta}) = {t}, expected cos({n}θ) = {expected}"
            );
        }
    }
}

// MR12 — Chebyshev U_n(cos θ) sin θ = sin((n+1) θ).
#[test]
fn mr_chebyu_sin_relation() {
    for n in 0..=8_u32 {
        for k in 1..=19 {
            let theta = k as f64 * PI / 20.0;
            let x = theta.cos();
            let u = eval_chebyu(n, x);
            let lhs = u * theta.sin();
            let expected = ((n + 1) as f64 * theta).sin();
            assert!(
                close(lhs, expected),
                "MR12 U_{n}(cos θ) sin θ = {lhs}, expected sin({}θ) = {expected}",
                n + 1
            );
        }
    }
}

// MR13 — Hermite recurrence H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x).
#[test]
fn mr_hermite_three_term_recurrence() {
    for x in [-2.0_f64, -0.5, 0.0, 0.7, 1.5, 3.0] {
        let mut prev = eval_hermite(0, x);
        let mut curr = eval_hermite(1, x);
        for n in 1..=10_u32 {
            let next_expected = 2.0 * x * curr - 2.0 * (n as f64) * prev;
            let next = eval_hermite(n + 1, x);
            assert!(
                close(next, next_expected),
                "MR13 Hermite recurrence at n={n}, x={x}: H_{}(x) = {next}, recurrence = {next_expected}",
                n + 1
            );
            prev = curr;
            curr = next;
        }
    }
}

// MR14 — Probabilist's Hermite recurrence: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x).
#[test]
fn mr_hermitenorm_three_term_recurrence() {
    for x in [-2.0_f64, -0.5, 0.0, 0.7, 1.5, 3.0] {
        let mut prev = eval_hermitenorm(0, x);
        let mut curr = eval_hermitenorm(1, x);
        for n in 1..=10_u32 {
            let next_expected = x * curr - (n as f64) * prev;
            let next = eval_hermitenorm(n + 1, x);
            assert!(
                close(next, next_expected),
                "MR14 He recurrence at n={n}, x={x}: He_{}(x) = {next}, recurrence = {next_expected}",
                n + 1
            );
            prev = curr;
            curr = next;
        }
    }
}

// MR15 — Laguerre L_n(0) = 1 for all n.
#[test]
fn mr_laguerre_at_zero_is_one() {
    for n in 0..=12_u32 {
        let l = eval_laguerre(n, 0.0);
        assert!(close(l, 1.0), "MR15 L_{n}(0) = {l}, expected 1");
    }
}

// MR16 — roots of Legendre P_n give P_n(root_i) ≈ 0.
#[test]
fn mr_roots_legendre_evaluate_to_zero() {
    for n in 2..=10_usize {
        let (xs, weights) = roots_legendre(n);
        assert_eq!(xs.len(), n);
        assert_eq!(weights.len(), n);
        for &x in &xs {
            let v = eval_legendre(n as u32, x);
            assert!(v.abs() < 1e-10, "MR16 P_{n}(root={x}) = {v}, expected 0");
            // Roots must be in (-1, 1).
            assert!(x.abs() < 1.0, "MR16 root {x} outside open interval");
        }
        // Weights must be positive and sum to 2 (Gauss-Legendre).
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-10,
            "MR16 Legendre weights sum to {sum}, expected 2"
        );
        for &w in &weights {
            assert!(w > 0.0, "MR16 negative Legendre weight: {w}");
        }
    }
}

// MR17 — roots of Chebyshev T_n give T_n(root_i) ≈ 0; weights sum to π.
#[test]
fn mr_roots_chebyt_evaluate_to_zero() {
    for n in 2..=10_usize {
        let (xs, weights) = roots_chebyt(n);
        assert_eq!(xs.len(), n);
        for &x in &xs {
            let v = eval_chebyt(n as u32, x);
            assert!(v.abs() < 1e-10, "MR17 T_{n}(root={x}) = {v}, expected 0");
            assert!(x.abs() < 1.0, "MR17 root {x} outside (-1, 1)");
        }
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - PI).abs() < 1e-10,
            "MR17 Chebyshev T weights sum to {sum}, expected π"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — Gauss-Legendre n-point quadrature integrates polynomials of
// degree ≤ 2n − 1 exactly on [−1, 1].
//
// We test against ∫_{-1}^{1} x^k dx, which equals 2/(k+1) for even k
// and 0 for odd k. Required tolerance is fairly tight because the
// Gauss-Legendre weights and nodes can be evaluated to machine
// precision.
// ─────────────────────────────────────────────────────────────────────

fn analytic_x_k(k: usize) -> f64 {
    if k % 2 == 1 {
        0.0
    } else {
        2.0 / (k as f64 + 1.0)
    }
}

#[test]
fn mr_gauss_legendre_exactness_on_polynomials() {
    for n in 2..=8_usize {
        let (xs, weights) = roots_legendre(n);
        let max_degree = 2 * n - 1;
        for k in 0..=max_degree {
            let approx: f64 = xs
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = analytic_x_k(k);
            let bound = 1e-10 + 1e-9 * exact.abs().max(1.0);
            assert!(
                (approx - exact).abs() <= bound,
                "MR18 GL-{n} on x^{k}: got {approx:.16e}, exact {exact:.16e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Gauss-Chebyshev T n-point quadrature integrates
//   ∫_{-1}^{1} f(x) / sqrt(1 − x²) dx
// exactly for f a polynomial of degree ≤ 2n − 1. The simplest closed
// form: ∫_{-1}^{1} x^{2k} / sqrt(1 − x²) dx = π · (2k)! / (4^k (k!)²)
// (and zero for odd powers).
// ─────────────────────────────────────────────────────────────────────

fn central_binomial_coefficient(k: usize) -> f64 {
    // C(2k, k) = (2k)! / (k!)²
    let mut c = 1.0_f64;
    for i in 0..k {
        c *= (k + i + 1) as f64;
        c /= (i + 1) as f64;
    }
    c
}

fn analytic_gc_x_k(k: usize) -> f64 {
    if k % 2 == 1 {
        0.0
    } else {
        let half = k / 2;
        PI * central_binomial_coefficient(half) / (4.0_f64.powi(half as i32))
    }
}

#[test]
fn mr_gauss_chebyt_exactness_on_polynomials() {
    for n in 2..=8_usize {
        let (xs, weights) = roots_chebyt(n);
        let max_degree = 2 * n - 1;
        for k in 0..=max_degree {
            let approx: f64 = xs
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = analytic_gc_x_k(k);
            let bound = 1e-10 + 1e-9 * exact.abs().max(1.0);
            assert!(
                (approx - exact).abs() <= bound,
                "MR19 GCT-{n} on x^{k}: got {approx:.16e}, exact {exact:.16e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Boundary values of gammainc:
//   gammainc(a, 0) = 0 and gammainc(a, ∞) = 1.
// gammaincc is the complement, so gammaincc(a, 0) = 1, gammaincc(a, ∞) = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammainc_endpoints() {
    for a in [0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        let p_at_0 = unwrap_real(gammainc(&real(a), &real(0.0), RuntimeMode::Strict));
        assert!(
            p_at_0.abs() < 1e-12,
            "MR20 gammainc({a}, 0) = {p_at_0}, expected 0"
        );
        // For x = 100 we should be very close to 1 for all a ≤ 10.
        let p_at_inf = unwrap_real(gammainc(&real(a), &real(100.0), RuntimeMode::Strict));
        assert!(
            (p_at_inf - 1.0).abs() < 1e-9,
            "MR20 gammainc({a}, large) = {p_at_inf}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — gammainc(a, x) + gammaincc(a, x) = 1 for any (a > 0, x ≥ 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammainc_complement() {
    for a in [0.7_f64, 1.5, 3.0, 7.5] {
        for x in [0.1_f64, 0.5, 1.0, 3.0, 7.0, 15.0] {
            let p = unwrap_real(gammainc(&real(a), &real(x), RuntimeMode::Strict));
            let q = unwrap_real(gammaincc(&real(a), &real(x), RuntimeMode::Strict));
            let sum = p + q;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "MR21 gammainc({a}, {x}) + gammaincc({a}, {x}) = {sum}, expected 1"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — erf and gammainc relationship: for x ≥ 0,
//   erf(x) = gammainc(1/2, x²).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_erf_equals_gammainc_half() {
    for x in [0.1_f64, 0.5, 1.0, 1.5, 2.5, 4.0] {
        let lhs = erf_scalar(x);
        let rhs = unwrap_real(gammainc(&real(0.5), &real(x * x), RuntimeMode::Strict));
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "MR22 erf({x}) = {lhs}, gammainc(1/2, {}²) = {rhs}",
            x
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — Integer-order Bessel J via jv(n, x) agrees with j0/j1/jn
// for n=0 and n=1. Tests cross-routine consistency between the
// integer-special-case implementations and the general-order jv path.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jv_matches_jn_for_integer_orders() {
    for x in [0.5_f64, 1.5, 3.0, 7.5] {
        for n in 0..=4_i64 {
            let from_jn = unwrap_real(jn(&real(n as f64), &real(x), RuntimeMode::Strict));
            let from_jv = unwrap_real(jv(&real(n as f64), &real(x), RuntimeMode::Strict));
            assert!(
                (from_jn - from_jv).abs() < 1e-10,
                "MR23 jn({n}, {x}) = {from_jn}, jv({n}, {x}) = {from_jv}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — Bessel J recurrence: J_{n+1}(x) = (2n/x) J_n(x) − J_{n-1}(x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bessel_jn_recurrence() {
    for x in [0.5_f64, 1.5, 3.0, 7.5, 12.0] {
        let mut prev = unwrap_real(jn(&real(0.0), &real(x), RuntimeMode::Strict));
        let mut curr = unwrap_real(jn(&real(1.0), &real(x), RuntimeMode::Strict));
        for n in 1..=8_i64 {
            let next_expected = (2.0 * n as f64 / x) * curr - prev;
            let next = unwrap_real(jn(&real((n + 1) as f64), &real(x), RuntimeMode::Strict));
            assert!(
                (next - next_expected).abs() < 1e-7 * next.abs().max(1.0),
                "MR24 J recurrence at n={n}, x={x}: J_{}={next}, recurrence={next_expected}",
                n + 1
            );
            prev = curr;
            curr = next;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — ellipe boundary and monotonicity:
//   ellipe(0) = π/2, ellipe(1) = 1, monotone decreasing on [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipe_endpoint_and_monotonicity() {
    let e0 = unwrap_real(ellipe(&real(0.0), RuntimeMode::Strict));
    assert!(
        (e0 - PI / 2.0).abs() < 1e-12,
        "MR25 ellipe(0) = {e0}, expected π/2"
    );
    let e1 = unwrap_real(ellipe(&real(1.0), RuntimeMode::Strict));
    assert!(
        (e1 - 1.0).abs() < 1e-12,
        "MR25 ellipe(1) = {e1}, expected 1"
    );
    let mut prev = e0;
    for k in 1..50 {
        let m = k as f64 / 50.0;
        let v = unwrap_real(ellipe(&real(m), RuntimeMode::Strict));
        assert!(
            v <= prev + 1e-12,
            "MR25 ellipe not monotone-decreasing at m={m}: {v} > prev={prev}"
        );
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — hyp1f1(a, a, z) = exp(z): the confluent hypergeometric with
// numerator parameter equal to denominator parameter degenerates to
// the exponential.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hyp1f1_degenerate_exp() {
    for a in [0.5_f64, 1.0, 2.5, 5.0] {
        for z in [-2.0_f64, -0.5, 0.5, 1.0, 2.0] {
            let lhs = unwrap_real(hyp1f1(&real(a), &real(a), &real(z), RuntimeMode::Strict));
            let rhs = z.exp();
            let rel_err = (lhs - rhs).abs() / rhs.abs().max(1.0);
            assert!(
                rel_err < 1e-9,
                "MR26 hyp1f1({a}, {a}, {z}) = {lhs}, expected e^{z} = {rhs}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — hyp2f1(a, b; c; 0) = 1: every Gauss hypergeometric reduces
// to 1 at z = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hyp2f1_at_zero_is_one() {
    for &(a, b, c) in &[
        (1.0_f64, 1.0_f64, 1.0_f64),
        (0.5, 1.5, 2.5),
        (2.0, 3.0, 4.0),
        (-1.5, 0.7, 3.2),
    ] {
        let v = unwrap_real(hyp2f1(
            &real(a),
            &real(b),
            &real(c),
            &real(0.0),
            RuntimeMode::Strict,
        ));
        assert!(
            (v - 1.0).abs() < 1e-12,
            "MR27 hyp2f1({a}, {b}; {c}; 0) = {v}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — Airy function values at the origin:
//   ai(0) = 1 / (3^(2/3) · Γ(2/3)) ≈ 0.355028053887817...
//   bi(0) = 1 / (3^(1/6) · Γ(2/3)) ≈ 0.614926627446001...
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_airy_at_zero() {
    let ai0 = unwrap_real(ai(&real(0.0), RuntimeMode::Strict));
    let bi0 = unwrap_real(bi(&real(0.0), RuntimeMode::Strict));
    let expected_ai0 = 0.355_028_053_887_817_2;
    let expected_bi0 = 0.614_926_627_446_001;
    assert!(
        (ai0 - expected_ai0).abs() < 1e-9,
        "MR28 ai(0) = {ai0}, expected {expected_ai0}"
    );
    assert!(
        (bi0 - expected_bi0).abs() < 1e-9,
        "MR28 bi(0) = {bi0}, expected {expected_bi0}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — Incomplete elliptic integrals at φ = 0 are 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipinc_at_zero_phi() {
    for &m in &[0.0_f64, 0.25, 0.5, 0.75] {
        let f = unwrap_real(ellipkinc(&real(0.0), &real(m), RuntimeMode::Strict));
        assert!(f.abs() < 1e-12, "MR29 ellipkinc(0, {m}) = {f}, expected 0");
        let e = unwrap_real(ellipeinc(&real(0.0), &real(m), RuntimeMode::Strict));
        assert!(e.abs() < 1e-12, "MR29 ellipeinc(0, {m}) = {e}, expected 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — ellipkinc(π/2, m) = ellipk(m): the incomplete integral
// becomes the complete one at the upper limit π/2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipkinc_at_half_pi_matches_complete() {
    for &m in &[0.0_f64, 0.1, 0.3, 0.5, 0.7, 0.9] {
        let inc = unwrap_real(ellipkinc(&real(PI / 2.0), &real(m), RuntimeMode::Strict));
        let comp = unwrap_real(ellipk(&real(m), RuntimeMode::Strict));
        assert!(
            (inc - comp).abs() < 1e-9,
            "MR30 ellipkinc(π/2, {m}) = {inc} vs ellipk({m}) = {comp}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — expit∘logit = id on (0, 1), and logit∘expit = id on R.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_expit_logit_round_trip() {
    for &p in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
        let l = unwrap_real(logit(&real(p), RuntimeMode::Strict));
        let back = unwrap_real(expit(&real(l), RuntimeMode::Strict));
        assert!((back - p).abs() < 1e-12, "MR31 expit(logit({p})) = {back}");
    }
    for &x in &[-3.5_f64, -1.0, 0.0, 1.0, 3.5] {
        let e = unwrap_real(expit(&real(x), RuntimeMode::Strict));
        let back = unwrap_real(logit(&real(e), RuntimeMode::Strict));
        assert!((back - x).abs() < 1e-9, "MR31 logit(expit({x})) = {back}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — xlogy(0, y) = 0 for any finite y > 0; xlogy(x, 1) = 0;
// xlog1py(0, y) = 0 for finite y; xlog1py(x, 0) = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_xlogy_xlog1py_boundary_zeros() {
    for &y in &[0.5_f64, 1.0, 2.0, 100.0] {
        let v = unwrap_real(xlogy(&real(0.0), &real(y), RuntimeMode::Strict));
        assert!(v.abs() < 1e-15, "MR32 xlogy(0, {y}) = {v}, expected 0");
    }
    for &x in &[1.0_f64, 5.0, 100.0] {
        let v = unwrap_real(xlogy(&real(x), &real(1.0), RuntimeMode::Strict));
        assert!(v.abs() < 1e-15, "MR32 xlogy({x}, 1) = {v}, expected 0");
    }
    for &y in &[-0.5_f64, 0.0, 1.0, 5.0] {
        let v = unwrap_real(xlog1py(&real(0.0), &real(y), RuntimeMode::Strict));
        assert!(v.abs() < 1e-15, "MR32 xlog1py(0, {y}) = {v}, expected 0");
    }
    for &x in &[1.0_f64, 5.0, 100.0] {
        let v = unwrap_real(xlog1py(&real(x), &real(0.0), RuntimeMode::Strict));
        assert!(v.abs() < 1e-15, "MR32 xlog1py({x}, 0) = {v}, expected 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — Arithmetic-Geometric Mean (AGM) is symmetric in its arguments
// and AGM(a, a) = a.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_agm_symmetric_and_diagonal() {
    for &(a, b) in &[(1.0_f64, 4.0), (0.5, 2.5), (2.0, 7.0), (10.0, 1.0)] {
        let ab = agm(a, b);
        let ba = agm(b, a);
        assert!(
            (ab - ba).abs() < 1e-12,
            "MR33 agm({a}, {b}) = {ab}, agm({b}, {a}) = {ba}"
        );
    }
    for &a in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        let v = agm(a, a);
        assert!((v - a).abs() < 1e-12, "MR33 agm({a}, {a}) = {v}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — betaln(a, b) = lnΓ(a) + lnΓ(b) - lnΓ(a + b).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_betaln_via_gammaln() {
    for &(a, b) in &[
        (0.5_f64, 0.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (5.5, 4.2),
        (10.0, 0.7),
    ] {
        let bln = unwrap_real(betaln(&real(a), &real(b), RuntimeMode::Strict));
        let gln_a = unwrap_real(gammaln(&real(a), RuntimeMode::Strict));
        let gln_b = unwrap_real(gammaln(&real(b), RuntimeMode::Strict));
        let gln_ab = unwrap_real(gammaln(&real(a + b), RuntimeMode::Strict));
        let expected = gln_a + gln_b - gln_ab;
        assert!(
            (bln - expected).abs() < 1e-9,
            "MR34 betaln({a}, {b}) = {bln} vs gammaln combo = {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — arctanh(0) = 0, arctanh(tanh(x)) = x for |x| < 17.
// (arctanh saturates near ±1 due to log; tanh(17) is ≈ 1, so any x
// with |x| ≤ 17 is well within numerical range.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_arctanh_round_trip() {
    assert!(arctanh(0.0).abs() < 1e-15, "MR35 arctanh(0) != 0");
    for &x in &[-2.5_f64, -1.0, -0.3, 0.5, 1.5, 3.0] {
        let back = arctanh(x.tanh());
        assert!((back - x).abs() < 1e-9, "MR35 arctanh(tanh({x})) = {back}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — beta(a, b) = exp(betaln(a, b)) for moderate a, b. (Cross-check
// of the two flavours of the beta function.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_beta_exp_betaln() {
    for &(a, b) in &[
        (0.5_f64, 0.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (4.0, 4.0),
        (1.5, 2.5),
    ] {
        let b_val = unwrap_real(beta(&real(a), &real(b), RuntimeMode::Strict));
        let bln = unwrap_real(betaln(&real(a), &real(b), RuntimeMode::Strict));
        let expected = bln.exp();
        let rel = (b_val - expected).abs() / expected.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "MR36 beta({a}, {b}) = {b_val} vs exp(betaln) = {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — ζ(2) = π²/6 and ζ(4) = π⁴/90 (Basel problem and follow-ups).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_zeta_basel_values() {
    let z2 = zeta_scalar(2.0);
    let expected2 = PI * PI / 6.0;
    assert!(
        (z2 - expected2).abs() < 1e-10,
        "MR37 ζ(2) = {z2}, expected π²/6 = {expected2}"
    );
    let z4 = zeta_scalar(4.0);
    let expected4 = PI.powi(4) / 90.0;
    assert!(
        (z4 - expected4).abs() < 1e-12,
        "MR37 ζ(4) = {z4}, expected π⁴/90 = {expected4}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — Lambert W on values where the implementation returns finite
// output satisfies W(x)·exp(W(x)) = x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lambertw_defining_identity() {
    assert_eq!(
        lambertw_scalar(f64::INFINITY),
        f64::INFINITY,
        "MR38 W(+inf) should be +inf"
    );

    for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0, 100.0] {
        let w = lambertw_scalar(x);
        assert!(w.is_finite(), "MR38 W({x}) returned non-finite: {w}");
        let recovered = w * w.exp();
        assert!(
            (recovered - x).abs() < 1e-6 * x.abs().max(1.0),
            "MR38 W({x}) · exp(W({x})) = {recovered}, expected {x}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — boxcox_transform(x, λ=1) = x - 1 (the λ=1 specialisation).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_boxcox_lambda_one_specializes() {
    for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        let v = boxcox_transform_scalar(x, 1.0);
        assert!(
            (v - (x - 1.0)).abs() < 1e-12,
            "MR39 boxcox(x={x}, λ=1) = {v} vs x - 1 = {}",
            x - 1.0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — boxcox / inv_boxcox round-trip for x > 0 and any λ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_boxcox_inv_roundtrip() {
    for &x in &[0.5_f64, 1.0, 2.0, 5.0, 10.0] {
        for &lam in &[-1.0_f64, 0.0, 0.5, 1.0, 2.5] {
            let y = boxcox_transform_scalar(x, lam);
            let back = inv_boxcox_scalar(y, lam);
            assert!(
                (back - x).abs() < 1e-9 * x.abs().max(1.0),
                "MR40 inv_boxcox(boxcox({x}, λ={lam})) = {back}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — digamma_scalar(1) = -γ (the Euler-Mascheroni constant).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_digamma_at_one_is_neg_gamma() {
    let v = digamma_scalar(1.0);
    let euler_gamma = 0.577_215_664_901_532_9_f64;
    assert!(
        (v + euler_gamma).abs() < 1e-9,
        "MR41 ψ(1) = {v}, expected -γ = {}",
        -euler_gamma
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — entr(1) = 0 (entropy of x = 1 is -1·ln(1) = 0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_entr_at_one_is_zero() {
    let v = unwrap_real(entr(&real(1.0), RuntimeMode::Strict));
    assert!(v.abs() < 1e-12, "MR42 entr(1) = {v}, expected 0");
    // entr(0) = 0 by SciPy convention (lim x·log(x) as x → 0+).
    let v0 = unwrap_real(entr(&real(0.0), RuntimeMode::Strict));
    assert!(v0.abs() < 1e-12, "MR42 entr(0) = {v0}, expected 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — Jacobi P_0^(α, β)(x) = 1 for any x and any (α, β).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jacobi_zero_order_is_one() {
    for &(a, b) in &[(0.0_f64, 0.0), (0.5, 1.5), (-0.3, 0.7), (2.0, 3.0)] {
        for &x in &[-0.9_f64, -0.5, 0.0, 0.3, 0.7, 0.95] {
            let v = eval_jacobi(0, a, b, x);
            assert!(
                (v - 1.0).abs() < 1e-12,
                "MR43 jacobi(0, {a}, {b}, {x}) = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — Gegenbauer C_0^α(x) = 1 for any x and α.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gegenbauer_zero_order_is_one() {
    for &alpha in &[0.5_f64, 1.0, 1.5, 2.5] {
        for &x in &[-0.9_f64, -0.5, 0.0, 0.3, 0.7, 0.95] {
            let v = eval_gegenbauer(0, alpha, x);
            assert!(
                (v - 1.0).abs() < 1e-12,
                "MR44 gegenbauer(0, {alpha}, {x}) = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — Generalized Laguerre L_0^α(x) = 1 for any x, α.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_genlaguerre_zero_order_is_one() {
    for &alpha in &[0.0_f64, 0.5, 1.0, 2.5] {
        for &x in &[0.0_f64, 0.5, 1.0, 3.0, 10.0] {
            let v = eval_genlaguerre(0, alpha, x);
            assert!(
                (v - 1.0).abs() < 1e-12,
                "MR45 genlaguerre(0, {alpha}, {x}) = {v}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — Shifted Legendre P*_n(t) = P_n(2t - 1) for t ∈ [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sh_legendre_matches_shifted_legendre() {
    for n in [0u32, 1, 2, 3, 4, 5] {
        for &t in &[0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let lhs = eval_sh_legendre(n, t);
            let rhs = eval_legendre(n, 2.0 * t - 1.0);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "MR46 sh_legendre(n={n}, t={t}) = {lhs} vs legendre(2t-1) = {rhs}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — Associated Legendre lpmv with m = 0 reduces to standard
// Legendre P_l(x): lpmv(0, l, x) = eval_legendre(l, x).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_lpmv_zero_m_matches_legendre() {
    for l in 0u32..=5 {
        for &x in &[-0.9_f64, -0.5, 0.0, 0.3, 0.7, 0.95] {
            let lhs = lpmv(0, l, x);
            let rhs = eval_legendre(l, x);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "MR47 lpmv(0, {l}, {x}) = {lhs} vs legendre({l}, {x}) = {rhs}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — Spherical harmonics Y_l^0 are real (imaginary part vanishes)
// for any θ, φ.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sph_harm_m0_is_real() {
    for l in 0u32..=4 {
        for &theta in &[0.0_f64, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI] {
            for &phi in &[0.0_f64, PI / 3.0, PI, 1.5 * PI] {
                let y = sph_harm(0, l, theta, phi);
                assert!(
                    y.im.abs() < 1e-9,
                    "MR48 imag(Y_{l}^0(θ={theta}, φ={phi})) = {}",
                    y.im
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — comb(n, 0) = 1 and comb(n, n) = 1 for any n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_comb_endpoints_one() {
    for n in 0u64..=15 {
        assert!(
            (comb(n, 0) - 1.0).abs() < 1e-12,
            "MR49 comb({n}, 0) = {}",
            comb(n, 0)
        );
        assert!(
            (comb(n, n) - 1.0).abs() < 1e-12,
            "MR49 comb({n}, {n}) = {}",
            comb(n, n)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — comb is symmetric: comb(n, k) = comb(n, n - k).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_comb_symmetric_in_k() {
    for n in 0u64..=12 {
        for k in 0..=n {
            let lhs = comb(n, k);
            let rhs = comb(n, n - k);
            assert!(
                (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
                "MR50 comb({n}, {k}) = {lhs} vs comb({n}, {}) = {rhs}",
                n - k
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — perm(n, 0) = 1 and perm(n, 1) = n.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_perm_basics() {
    for n in 1u64..=15 {
        assert!(
            (perm(n, 0) - 1.0).abs() < 1e-12,
            "MR51 perm({n}, 0) = {}",
            perm(n, 0)
        );
        assert!(
            (perm(n, 1) - n as f64).abs() < 1e-12,
            "MR51 perm({n}, 1) = {} vs {n}",
            perm(n, 1)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — factorial2 of small odd integers matches the textbook double
// factorial formula: 5!! = 15, 7!! = 105, 9!! = 945.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_factorial2_known_values() {
    let cases: &[(i64, f64)] = &[
        (0, 1.0),
        (1, 1.0),
        (2, 2.0),
        (3, 3.0),
        (4, 8.0),
        (5, 15.0),
        (6, 48.0),
        (7, 105.0),
        (8, 384.0),
        (9, 945.0),
    ];
    for &(n, expected) in cases {
        let v = factorial2(n);
        assert!(
            (v - expected).abs() < 1e-9 * expected.max(1.0),
            "MR52 factorial2({n}) = {v}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — rgamma(1) = 1 and rgamma(2) = 1 (1/Γ at positive integers ≤ 2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rgamma_at_small_integers() {
    let r1 = unwrap_real(rgamma(&real(1.0), RuntimeMode::Strict));
    let r2 = unwrap_real(rgamma(&real(2.0), RuntimeMode::Strict));
    assert!(
        (r1 - 1.0).abs() < 1e-12,
        "MR53 rgamma(1) = {r1}, expected 1"
    );
    assert!(
        (r2 - 1.0).abs() < 1e-12,
        "MR53 rgamma(2) = {r2}, expected 1"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — Chi-squared CDF + complement: chdtr(v, x) + chdtrc(v, x) = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_chdtr_complement_sums_to_one() {
    for &v in &[1.0_f64, 2.0, 5.0, 10.0] {
        for &x in &[0.5_f64, 1.0, 3.0, 5.0, 10.0, 20.0] {
            let cdf = chdtr(v, x);
            let sf = chdtrc(v, x);
            assert!(
                (cdf + sf - 1.0).abs() < 1e-9,
                "MR54 chdtr({v}, {x}) + chdtrc = {} expected 1",
                cdf + sf
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — fresnel(0) = (0, 0): the Fresnel integrals vanish at the origin.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fresnel_at_zero() {
    let (s, c) = fresnel(0.0);
    assert!(s.abs() < 1e-12, "MR55 S(0) = {s}, expected 0");
    assert!(c.abs() < 1e-12, "MR55 C(0) = {c}, expected 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — sici(0) = (0, -∞): Si(0) = 0, Ci(0) = -∞.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sici_at_zero() {
    let (si, ci) = sici(0.0);
    assert!(si.abs() < 1e-12, "MR56 Si(0) = {si}, expected 0");
    assert!(
        ci.is_infinite() && ci < 0.0,
        "MR56 Ci(0) = {ci}, expected -∞"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR57 — shichi(0) = (0, -∞): Shi(0) = 0, Chi(0) = -∞.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_shichi_at_zero() {
    let (shi, chi) = shichi(0.0);
    assert!(shi.abs() < 1e-12, "MR57 Shi(0) = {shi}, expected 0");
    assert!(
        chi.is_infinite() && chi < 0.0,
        "MR57 Chi(0) = {chi}, expected -∞"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR58 — Kelvin functions at zero: ber(0) = 1, bei(0) = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kelvin_at_zero() {
    let b = ber(0.0);
    let bi_v = bei(0.0);
    assert!((b - 1.0).abs() < 1e-12, "MR58 ber(0) = {b}, expected 1");
    assert!(bi_v.abs() < 1e-12, "MR58 bei(0) = {bi_v}, expected 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR59 — struve(v, 0) = 0 for v > -1. Test on small non-negative orders.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_struve_at_zero() {
    for &v in &[0.0_f64, 0.5, 1.0, 1.5, 2.0] {
        let s = struve(v, 0.0);
        assert!(s.abs() < 1e-9, "MR59 struve({v}, 0) = {s}, expected 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR60 — Bernoulli numbers: B_0 = 1, B_2 = 1/6, B_4 = -1/30.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bernoulli_known_values() {
    let cases: &[(u32, f64)] = &[(0, 1.0), (2, 1.0 / 6.0), (4, -1.0 / 30.0), (6, 1.0 / 42.0)];
    for &(n, expected) in cases {
        let v = bernoulli(n);
        assert!(
            (v - expected).abs() < 1e-9,
            "MR60 bernoulli({n}) = {v}, expected {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR61 — ellipj(0, m) = (sn=0, cn=1, dn=1, am=0).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipj_at_u_zero() {
    for &m in &[0.0_f64, 0.25, 0.5, 0.75, 0.9] {
        let (sn, cn, dn, am) = ellipj(0.0, m);
        assert!(sn.abs() < 1e-12, "MR61 sn(0, {m}) = {sn}");
        assert!((cn - 1.0).abs() < 1e-12, "MR61 cn(0, {m}) = {cn}");
        assert!((dn - 1.0).abs() < 1e-12, "MR61 dn(0, {m}) = {dn}");
        assert!(am.abs() < 1e-12, "MR61 am(0, {m}) = {am}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR62 — ellipkm1(p) = ellipk(1 - p) for small p > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipkm1_matches_ellipk_complement() {
    for &p in &[0.01_f64, 0.1, 0.3, 0.5] {
        let lhs = unwrap_real(ellipkm1(&real(p), RuntimeMode::Strict));
        let rhs = unwrap_real(ellipk(&real(1.0 - p), RuntimeMode::Strict));
        assert!(
            (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
            "MR62 ellipkm1({p}) = {lhs} vs ellipk({}) = {rhs}",
            1.0 - p
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR63 — exp1(x) = E_1(x) = ∫_x^∞ exp(-t)/t dt is positive for x > 0
// and decreasing in x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_exp1_positive_decreasing() {
    let xs = [0.5_f64, 1.0, 2.0, 3.0, 5.0, 10.0];
    let mut prev = f64::INFINITY;
    for &x in &xs {
        let v = unwrap_real(exp1(&real(x), RuntimeMode::Strict));
        assert!(v > 0.0, "MR63 exp1({x}) = {v} ≤ 0");
        assert!(
            v <= prev + 1e-12,
            "MR63 exp1({x}) = {v} > previous = {prev}"
        );
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR64 — expi(x) is increasing in x for x > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_expi_increasing_positive() {
    let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    let mut prev = f64::NEG_INFINITY;
    for &x in &xs {
        let v = unwrap_real(expi(&real(x), RuntimeMode::Strict));
        assert!(v >= prev - 1e-9, "MR64 expi({x}) = {v} < previous = {prev}");
        prev = v;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR65 — expn_scalar(1, x) equals exp1(x) for positive x.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_expn_n1_matches_exp1() {
    for &x in &[0.5_f64, 1.0, 2.0, 3.0, 5.0] {
        let v_n1 = expn_scalar(1, x);
        let v_e1 = unwrap_real(exp1(&real(x), RuntimeMode::Strict));
        assert!(
            (v_n1 - v_e1).abs() < 1e-9 * v_e1.abs().max(1.0),
            "MR65 expn_scalar(1, {x}) = {v_n1} vs exp1({x}) = {v_e1}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR66 — Pythagorean identity for Jacobi elliptic functions:
//   sn² + cn² = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ellipj_pythagorean_identity() {
    for &u in &[0.0_f64, 0.5, 1.0, 1.5, 2.0] {
        for &m in &[0.0_f64, 0.25, 0.5, 0.75, 0.9] {
            let (sn, cn, _dn, _am) = ellipj(u, m);
            let s = sn * sn + cn * cn;
            assert!(
                (s - 1.0).abs() < 1e-9,
                "MR66 sn² + cn² = {s} for u = {u}, m = {m}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR67 — Carlson elliptic homogeneity (scaling) laws.
//
// All four Carlson symmetric integrals are homogeneous functions:
//
//   RC(λx, λy)        = λ^(-1/2) · RC(x, y)
//   RF(λx, λy, λz)    = λ^(-1/2) · RF(x, y, z)
//   RD(λx, λy, λz)    = λ^(-3/2) · RD(x, y, z)
//   RG(λx, λy, λz)    = λ^( 1/2) · RG(x, y, z)
//
// These exercise the duplication+convergence path at multiple input
// scales. A normalization or transcription bug anywhere in the
// duplication formula or the closing series would break the identity.
// Resolves [frankenscipy-ekfyi].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_elliprc_homogeneity() {
    let bases: &[(f64, f64)] = &[(1.0, 2.0), (0.5, 4.0), (3.0, 7.5), (0.1, 0.9)];
    let lambdas: &[f64] = &[0.25_f64, 1.0, 4.0, 16.0];
    for &(x, y) in bases {
        let base = elliprc(x, y);
        for &lam in lambdas {
            let scaled = elliprc(lam * x, lam * y);
            let predicted = base * lam.powf(-0.5);
            assert!(
                close(scaled, predicted),
                "MR67 RC homogeneity broken: RC({} ,{}) = {}, RC(λ·,λ·) = {}, predicted {} (λ={})",
                x,
                y,
                base,
                scaled,
                predicted,
                lam
            );
        }
    }
}

#[test]
fn mr_elliprf_homogeneity() {
    let bases: &[(f64, f64, f64)] = &[
        (1.0, 2.0, 3.0),
        (0.5, 1.5, 4.0),
        (0.0, 1.0, 1.0),
        (0.1, 0.5, 2.0),
    ];
    let lambdas: &[f64] = &[0.25_f64, 1.0, 4.0, 16.0];
    for &(x, y, z) in bases {
        let base = elliprf(x, y, z);
        for &lam in lambdas {
            let scaled = elliprf(lam * x, lam * y, lam * z);
            let predicted = base * lam.powf(-0.5);
            assert!(
                close(scaled, predicted),
                "MR67 RF homogeneity broken at ({}, {}, {}) λ={}: scaled {} vs predicted {}",
                x,
                y,
                z,
                lam,
                scaled,
                predicted
            );
        }
    }
}

#[test]
fn mr_elliprd_homogeneity() {
    let bases: &[(f64, f64, f64)] = &[(1.0, 2.0, 3.0), (0.5, 1.5, 4.0), (0.1, 0.5, 2.0)];
    let lambdas: &[f64] = &[0.25_f64, 1.0, 4.0, 16.0];
    for &(x, y, z) in bases {
        let base = elliprd(x, y, z);
        for &lam in lambdas {
            let scaled = elliprd(lam * x, lam * y, lam * z);
            let predicted = base * lam.powf(-1.5);
            assert!(
                close(scaled, predicted),
                "MR67 RD homogeneity broken at ({}, {}, {}) λ={}: scaled {} vs predicted {}",
                x,
                y,
                z,
                lam,
                scaled,
                predicted
            );
        }
    }
}

#[test]
fn mr_elliprg_homogeneity() {
    let bases: &[(f64, f64, f64)] = &[
        (1.0, 2.0, 3.0),
        (0.5, 1.5, 4.0),
        (0.0, 1.0, 1.0),
        (0.1, 0.5, 2.0),
    ];
    let lambdas: &[f64] = &[0.25_f64, 1.0, 4.0, 16.0];
    for &(x, y, z) in bases {
        let base = elliprg(x, y, z);
        for &lam in lambdas {
            let scaled = elliprg(lam * x, lam * y, lam * z);
            let predicted = base * lam.powf(0.5);
            assert!(
                close(scaled, predicted),
                "MR67 RG homogeneity broken at ({}, {}, {}) λ={}: scaled {} vs predicted {}",
                x,
                y,
                z,
                lam,
                scaled,
                predicted
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR68 — Pfaff transformation for Gauss hypergeometric 2F1:
//
//   2F1(a,b;c;z) = (1-z)^(-a) · 2F1(a,c-b;c;z/(z-1))
//
// This relates the branch-sensitive negative-z dispatch path to an
// ordinary positive argument inside the unit disk.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hyp2f1_pfaff_transformation() {
    let cases: &[(f64, f64, f64, f64)] = &[
        (0.5, 1.0, 2.5, -0.2),
        (1.25, 0.75, 2.75, -0.75),
        (2.0, 1.5, 4.0, -2.0),
    ];
    for &(a, b, c, z) in cases {
        let lhs = hyp2f1_real(a, b, c, z);
        let z_new = z / (z - 1.0);
        let rhs = (1.0 - z).powf(-a) * hyp2f1_real(a, c - b, c, z_new);
        assert!(
            close(lhs, rhs),
            "MR68 Pfaff identity failed for 2F1({a}, {b}; {c}; {z}): lhs={lhs}, rhs={rhs}, z_new={z_new}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR69 — Euler transformation for Gauss hypergeometric 2F1:
//
//   2F1(a,b;c;z) = (1-z)^(c-a-b) · 2F1(c-a,c-b;c;z)
//
// This catches parameter-transposition and exponent bugs without a
// SciPy oracle.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hyp2f1_euler_transformation() {
    let cases: &[(f64, f64, f64, f64)] = &[
        (0.5, 1.25, 2.75, 0.2),
        (1.0, 0.75, 2.5, 0.4),
        (1.5, 0.5, 3.2, 0.6),
    ];
    for &(a, b, c, z) in cases {
        let lhs = hyp2f1_real(a, b, c, z);
        let rhs = (1.0 - z).powf(c - a - b) * hyp2f1_real(c - a, c - b, c, z);
        assert!(
            close(lhs, rhs),
            "MR69 Euler identity failed for 2F1({a}, {b}; {c}; {z}): lhs={lhs}, rhs={rhs}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR70 — Lower regularized gamma adjacent-shape recurrence:
//
//   P(a+1,x) = P(a,x) - x^a · exp(-x) / Γ(a+1)
//
// This cross-checks the recurrence bridge between the lower incomplete
// gamma and gamma normalization paths without a SciPy oracle.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammainc_adjacent_shape_recurrence() {
    let cases: &[(f64, f64)] = &[
        (0.25, 0.1),
        (0.5, 0.75),
        (1.25, 0.6),
        (2.5, 3.0),
        (5.0, 8.0),
    ];
    for &(a, x) in cases {
        let p_a = gammainc_real(a, x);
        let p_next = gammainc_real(a + 1.0, x);
        let recurrence_term = x.powf(a) * (-x).exp() / gamma_real(a + 1.0);
        let rhs = p_a - recurrence_term;
        assert!(
            close(p_next, rhs),
            "MR70 P(a+1,x) recurrence failed for a={a}, x={x}: p_next={p_next}, rhs={rhs}, term={recurrence_term}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR71 — Upper regularized gamma adjacent-shape recurrence:
//
//   Q(a+1,x) = Q(a,x) + x^a · exp(-x) / Γ(a+1)
//
// This exercises the complement implementation on the same recurrence
// term as MR70 with the opposite sign.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gammaincc_adjacent_shape_recurrence() {
    let cases: &[(f64, f64)] = &[
        (0.25, 0.1),
        (0.5, 0.75),
        (1.25, 0.6),
        (2.5, 3.0),
        (5.0, 8.0),
    ];
    for &(a, x) in cases {
        let q_a = gammaincc_real(a, x);
        let q_next = gammaincc_real(a + 1.0, x);
        let recurrence_term = x.powf(a) * (-x).exp() / gamma_real(a + 1.0);
        let rhs = q_a + recurrence_term;
        assert!(
            close(q_next, rhs),
            "MR71 Q(a+1,x) recurrence failed for a={a}, x={x}: q_next={q_next}, rhs={rhs}, term={recurrence_term}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Golden parity — hyp2f1 real analytic continuation for z > 1
// (Euler-terminating family). frankenscipy-nwvrw.
//
// For real z > 1, scipy.special.hyp2f1 returns a finite value exactly when
// the Euler transformation 2F1(a,b;c;z) = (1-z)^{c-a-b} 2F1(c-a,c-b;c;z)
// terminates with a real prefactor: c-a a nonpositive integer with b
// integer, or c-b a nonpositive integer with a integer. Every other real
// z > 1 lies on the [1, ∞) branch cut and returns +inf. Golden values were
// produced by scipy 1.17.1 (sp.hyp2f1). Asserted at < 1e-10 abs.
// ─────────────────────────────────────────────────────────────────────

#[test]
#[allow(clippy::excessive_precision)] // golden constants copied verbatim from scipy 1.17.1
fn golden_hyp2f1_real_continuation_z_gt_one() {
    // (a, b, c, z, scipy_value)
    let finite: &[(f64, f64, f64, f64, f64)] = &[
        // New Euler-terminating cases (c-a or c-b nonpos int, other integer).
        (1.0, 1.5, 0.5, 1.2, 55.00000000000003),
        (1.0, 1.5, 0.5, 2.0, 3.0),
        (1.0, 1.5, 0.5, 5.0, 0.375),
        (1.5, 1.0, 0.5, 1.5, 10.0),
        (1.5, 2.0, 0.5, 2.0, -7.0),
        (1.5, 3.0, 0.5, 3.0, 1.0),
        (2.0, 1.5, 0.5, 2.0, -7.0),
        (2.0, 2.0, 1.0, 1.5, -20.0),
        (2.0, 3.0, 1.0, 2.0, 5.0),
        (1.0, 3.0, 2.0, 1.5, 1.0),
        (3.0, 1.0, 2.0, 4.0, -0.1111111111111111),
        // Regression — the old c==a / c==b linear-fractional identity (n=0).
        (2.0, 1.0, 2.0, 1.5, -2.0),
        (1.0, 2.0, 2.0, 3.0, -0.5),
        (0.5, 1.0, 0.5, 2.0, -1.0),
        (1.0, 1.0, 1.0, 2.0, -1.0),
        // Negative-z Pfaff path sanity (unchanged).
        (0.5, 0.5, 1.5, -2.0, 0.8104969894767537),
    ];
    for &(a, b, c, z, expected) in finite {
        let got = hyp2f1_real(a, b, c, z);
        assert!(
            (got - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
            "hyp2f1({a}, {b}; {c}; {z}) = {got}, scipy = {expected}"
        );
    }

    // Branch-cut cases that must stay +inf (no spurious finite continuation).
    let branch_cut: &[(f64, f64, f64, f64)] = &[
        (0.5, 0.5, 1.5, 2.0),
        (0.3, 0.7, 1.1, 2.0),
        (0.25, 0.75, 1.25, 3.0),
        (2.3, 1.3, 0.7, 2.0),
        // c-b nonpos int but a NOT integer -> still on the cut.
        (2.5, 2.0, 1.0, 1.2),
    ];
    for &(a, b, c, z) in branch_cut {
        let got = hyp2f1_real(a, b, c, z);
        assert!(
            got.is_infinite() && got > 0.0,
            "hyp2f1({a}, {b}; {c}; {z}) = {got}, expected +inf (branch cut)"
        );
    }
}
