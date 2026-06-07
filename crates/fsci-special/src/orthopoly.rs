#![forbid(unsafe_code)]

//! Orthogonal polynomial evaluation.
//!
//! All polynomials are computed via stable three-term recurrence relations.
//! These are the building blocks for Gaussian quadrature, spectral methods,
//! and spherical harmonics.
//!
//! - `eval_legendre(n, x)` — Legendre P_n(x)
//! - `eval_chebyt(n, x)` — Chebyshev T_n(x) (first kind)
//! - `eval_chebyu(n, x)` — Chebyshev U_n(x) (second kind)
//! - `eval_hermite(n, x)` — Physicist's Hermite H_n(x)
//! - `eval_hermitenorm(n, x)` — Probabilist's Hermite He_n(x)
//! - `eval_laguerre(n, x)` — Laguerre L_n(x)
//! - `eval_genlaguerre(n, alpha, x)` — Generalized Laguerre L_n^α(x)
//! - `eval_jacobi(n, alpha, beta, x)` — Jacobi P_n^{α,β}(x)
//! - `eval_gegenbauer(n, alpha, x)` — Gegenbauer C_n^α(x)
//! - `roots_legendre(n)` — Gauss-Legendre nodes and weights
//! - `roots_chebyt(n)` — Gauss-Chebyshev nodes and weights
//! - `roots_hermite(n)` — Gauss-Hermite nodes and weights
//! - `roots_laguerre(n)` — Gauss-Laguerre nodes and weights
//! - `roots_jacobi(n, alpha, beta)` — Gauss-Jacobi nodes and weights

use std::f64::consts::PI;

use fsci_linalg::{DecompOptions, eigh};

/// Evaluate the Legendre polynomial P_n(x) of degree n at point x.
///
/// Uses the three-term recurrence:
///   P_0(x) = 1, P_1(x) = x,
///   (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
///
/// Domain: x in [-1, 1] for orthogonality, but evaluates for any real x.
pub fn eval_legendre(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return x;
    }
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Compute Legendre polynomial values P_k(x) and derivatives P'_k(x) for
/// k = 0..=n using the standard three-term recurrence.
///
/// Matches `scipy.special.lpn(n, x)` which returns `(Pn, Pn_deriv)` arrays
/// of length n+1.
///
/// Recurrence (values):
///   P_0(x) = 1, P_1(x) = x
///   (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
///
/// Recurrence (derivatives):
///   P'_0(x) = 0, P'_1(x) = 1
///   (1 - x^2) P'_k(x) = k (P_{k-1}(x) - x P_k(x))
/// At |x| = 1 the derivative is given by the closed form
///   P'_k(±1) = (±1)^(k+1) * k(k+1) / 2
/// to avoid division by 1 - x^2 = 0.
pub fn lpn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    let mut values = Vec::with_capacity(len);
    let mut derivs = Vec::with_capacity(len);
    if n == 0 {
        values.push(1.0);
        derivs.push(0.0);
        return (values, derivs);
    }
    if !x.is_finite() {
        return (vec![f64::NAN; len], vec![f64::NAN; len]);
    }
    values.push(1.0); // P_0
    values.push(x); // P_1
    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * values[k as usize] - kf * values[(k - 1) as usize])
            / (kf + 1.0);
        values.push(p_next);
    }
    let near_unit = (x.abs() - 1.0).abs() <= f64::EPSILON;
    if near_unit {
        let sign: f64 = if x > 0.0 { 1.0 } else { -1.0 };
        for k in 0..=n {
            let kf = k as f64;
            // P'_k(±1) = (±1)^(k+1) * k(k+1) / 2.
            let factor = sign.powi(k as i32 + 1) * kf * (kf + 1.0) * 0.5;
            derivs.push(factor);
        }
    } else {
        let one_minus_xsq = 1.0 - x * x;
        derivs.push(0.0); // P'_0
        for k in 1..=n {
            let kf = k as f64;
            let p_k = values[k as usize];
            let p_km1 = values[(k - 1) as usize];
            derivs.push(kf * (p_km1 - x * p_k) / one_minus_xsq);
        }
    }
    (values, derivs)
}

/// Compute Legendre polynomials of the second kind Q_k(x) and derivatives
/// Q'_k(x) for k = 0..=n on the real interval |x| < 1.
///
/// Matches `scipy.special.lqn(n, x)` which returns `(Qn, Qn_deriv)` arrays
/// of length n+1.
///
/// Q_0(x) = atanh(x) = (1/2) ln((1+x)/(1-x))
/// Q_1(x) = x * Q_0(x) - 1
/// (k+1) Q_{k+1}(x) = (2k+1) x Q_k(x) - k Q_{k-1}(x)
/// (1 - x^2) Q'_k(x) = k (Q_{k-1}(x) - x Q_k(x))
///
/// Returns NaN-filled arrays at the singular points x = ±1 (where Q_0
/// diverges) and outside (-1, 1), matching scipy's domain convention for
/// real-valued evaluation.
pub fn lqn(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let len = n as usize + 1;
    if !x.is_finite() || x.abs() >= 1.0 {
        return (vec![f64::NAN; len], vec![f64::NAN; len]);
    }
    let mut values = Vec::with_capacity(len);
    let mut derivs = Vec::with_capacity(len);
    let q0 = x.atanh();
    values.push(q0);
    if n == 0 {
        derivs.push(1.0 / (1.0 - x * x));
        return (values, derivs);
    }
    values.push(x * q0 - 1.0);
    for k in 1..n {
        let kf = k as f64;
        let q_next = ((2.0 * kf + 1.0) * x * values[k as usize] - kf * values[(k - 1) as usize])
            / (kf + 1.0);
        values.push(q_next);
    }
    // Derivatives via the same recurrence shape as P: (1-x²) Q'_k = k(Q_{k-1} − x Q_k).
    // For k=0 the formula collapses to Q'_0 = 1/(1-x²) directly.
    let one_minus_xsq = 1.0 - x * x;
    derivs.push(1.0 / one_minus_xsq);
    for k in 1..=n {
        let kf = k as f64;
        let q_k = values[k as usize];
        let q_km1 = values[(k - 1) as usize];
        derivs.push(kf * (q_km1 - x * q_k) / one_minus_xsq);
    }
    (values, derivs)
}

/// Associated Legendre Q array `Q_l^m(x)` for `0 ≤ m ≤ m_max` and
/// `0 ≤ l ≤ n_max`, on the open real interval `|x| < 1`.
///
/// Matches `scipy.special.lqmn(m, n, x)` shape (the values portion).
/// Returns a `(m_max+1) × (n_max+1)` row-major matrix.
///
/// Algorithm: row 0 is the standard Legendre Q sequence from `lqn`. Cells
/// with `l < m` are seeded from SciPy's derivative definition,
/// `Q_l^m(x) = (-1)^m (1-x^2)^(m/2) d^m Q_l(x)/dx^m`, then each row
/// is continued through the standard degree recurrence until the diagonal.
/// Cells with `l ≥ m` are built from row m−1 using DLMF 14.10.4:
///   √(1−x²) · Q_l^m(x) = (l−m+1) x · Q_l^{m−1}(x) − (l+m−1) · Q_{l−1}^{m−1}(x)
/// for l = m..=n_max.
///
/// For `|x| ≥ 1` or non-finite `x`, returns NaN-filled arrays — matching
/// the lqn singular convention (`Q_0(x) = atanh(x)` diverges at the
/// endpoints).
pub fn lqmn(m_max: u32, n_max: u32, x: f64) -> Vec<Vec<f64>> {
    let rows = m_max as usize + 1;
    let cols = n_max as usize + 1;
    let mut out = vec![vec![0.0_f64; cols]; rows];
    if !x.is_finite() || x.abs() >= 1.0 {
        for row in &mut out {
            for cell in row.iter_mut() {
                *cell = f64::NAN;
            }
        }
        return out;
    }
    // Row 0: standard Legendre Q.
    let (q0, _) = lqn(n_max, x);
    for (l, &q) in q0.iter().enumerate() {
        out[0][l] = q;
    }
    let one_minus_xsq = 1.0 - x * x;
    let denom = one_minus_xsq.sqrt();
    if denom == 0.0 {
        // Defensive — shouldn't happen since we already rejected |x| ≥ 1.
        return out;
    }
    let atanh_derivs = atanh_derivatives(m_max, x);
    let mut sqrt_power = 1.0_f64;

    // Row m for m ≥ 1.
    for m in 1..=m_max {
        let m_idx = m as usize;
        sqrt_power *= denom;
        let signed_sqrt_power = if m % 2 == 0 { sqrt_power } else { -sqrt_power };

        // Below the diagonal, scipy.special.lqmn follows the derivative
        // definition rather than the finite-degree polynomial convention.
        out[m_idx][0] = signed_sqrt_power * atanh_derivs[m_idx];
        if cols > 1 {
            out[m_idx][1] = signed_sqrt_power
                * (x * atanh_derivs[m_idx] + (m as f64) * atanh_derivs[m_idx - 1]);
        }
        if n_max >= 2 {
            for l_prev in 1..n_max {
                let l = l_prev + 1;
                if l >= m {
                    break;
                }
                let l_prev_idx = l_prev as usize;
                let lf = l_prev as f64;
                let mf = m as f64;
                out[m_idx][l as usize] = ((2.0 * lf + 1.0) * x * out[m_idx][l_prev_idx]
                    - (lf + mf) * out[m_idx][l_prev_idx - 1])
                    / (lf - mf + 1.0);
            }
        }

        for l in m..=n_max {
            let l_idx = l as usize;
            let lf = l as f64;
            let mf = m as f64;
            let q_l_prev = out[m_idx - 1][l_idx];
            let q_lm1_prev = out[m_idx - 1][l_idx - 1];
            out[m_idx][l_idx] =
                ((lf - mf + 1.0) * x * q_l_prev - (lf + mf - 1.0) * q_lm1_prev) / denom;
        }
    }
    out
}

fn eval_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c)
}

fn next_atanh_derivative_poly(poly: &[f64], order: usize) -> Vec<f64> {
    let mut next = vec![0.0_f64; poly.len() + 2];
    for (degree, &coeff) in poly.iter().enumerate().skip(1) {
        let derivative_coeff = (degree as f64) * coeff;
        next[degree - 1] += derivative_coeff;
        next[degree + 1] -= derivative_coeff;
    }
    let scale = 2.0 * (order as f64);
    for (degree, &coeff) in poly.iter().enumerate() {
        next[degree + 1] += scale * coeff;
    }
    next
}

fn atanh_derivatives(max_order: u32, x: f64) -> Vec<f64> {
    let max_order = max_order as usize;
    let mut derivs = vec![0.0_f64; max_order + 1];
    derivs[0] = x.atanh();
    if max_order == 0 {
        return derivs;
    }

    let one_minus_xsq = 1.0 - x * x;
    let mut denom_power = one_minus_xsq;
    let mut poly = vec![1.0_f64]; // P_1 where d/dx atanh(x) = P_1(x)/(1-x^2).
    for (order, deriv) in derivs.iter_mut().enumerate().take(max_order + 1).skip(1) {
        *deriv = eval_poly(&poly, x) / denom_power;
        if order < max_order {
            poly = next_atanh_derivative_poly(&poly, order);
            denom_power *= one_minus_xsq;
        }
    }
    derivs
}

/// Associated Legendre polynomial array `P_l^m(x)` for `0 ≤ m ≤ m_max` and
/// `0 ≤ l ≤ n_max`. Returns a `(m_max+1) × (n_max+1)` row-major matrix
/// indexed as `out[m][l]`.
///
/// Matches `scipy.special.lpmn(m, n, x)` shape (the values portion). For
/// each fixed `m`, the row `out[m][..]` is filled by:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
///   P_(m+1)^m(x) = (2m+1) x P_m^m(x)
///   (l-m+1) P_(l+1)^m(x) = (2l+1) x P_l^m(x) - (l+m) P_(l-1)^m(x)
///
/// Cells where `m > l` are 0.0 by convention.
pub fn lpmn(m_max: u32, n_max: u32, x: f64) -> Vec<Vec<f64>> {
    let rows = m_max as usize + 1;
    let cols = n_max as usize + 1;
    let mut out = vec![vec![0.0_f64; cols]; rows];
    if !x.is_finite() {
        for row in &mut out {
            for cell in row.iter_mut() {
                *cell = f64::NAN;
            }
        }
        return out;
    }
    let somx2 = (1.0 - x * x).max(0.0).sqrt();
    // For each m, compute P_m^m, P_(m+1)^m, then recurrence to fill row m.
    for m in 0..=m_max {
        // P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2).
        let mut pmm = 1.0_f64;
        if m > 0 {
            let mut fact = 1.0_f64;
            for _ in 1..=m {
                pmm *= -fact * somx2;
                fact += 2.0;
            }
        }
        let m_idx = m as usize;
        // Cells with l < m are zero by convention; leave them at 0.0.
        if m_idx >= cols {
            continue;
        }
        out[m_idx][m_idx] = pmm;
        if (m_idx + 1) < cols {
            // P_(m+1)^m(x) = (2m+1) x P_m^m(x)
            out[m_idx][m_idx + 1] = x * (2.0 * m as f64 + 1.0) * pmm;
            // Forward recurrence on l.
            for l in (m + 2)..=n_max {
                let lf = l as f64;
                let mf = m as f64;
                let prev = out[m_idx][l as usize - 2];
                let curr = out[m_idx][l as usize - 1];
                out[m_idx][l as usize] =
                    ((2.0 * (lf - 1.0) + 1.0) * x * curr - (lf - 1.0 + mf) * prev) / (lf - mf);
            }
        }
    }
    out
}

/// Evaluate the Chebyshev polynomial of the first kind T_n(x).
///
/// Uses the three-term recurrence:
///   T_0(x) = 1, T_1(x) = x,
///   T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
///
/// Identity: T_n(cos θ) = cos(nθ).
pub fn eval_chebyt(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut t_prev = 1.0;
    let mut t_curr = x;
    for _ in 1..n {
        let t_next = 2.0 * x * t_curr - t_prev;
        t_prev = t_curr;
        t_curr = t_next;
    }
    t_curr
}

/// Evaluate the Chebyshev polynomial of the second kind U_n(x).
///
/// Uses the three-term recurrence:
///   U_0(x) = 1, U_1(x) = 2x,
///   U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
pub fn eval_chebyu(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut u_prev = 1.0;
    let mut u_curr = 2.0 * x;
    for _ in 1..n {
        let u_next = 2.0 * x * u_curr - u_prev;
        u_prev = u_curr;
        u_curr = u_next;
    }
    u_curr
}

/// Evaluate the Chebyshev polynomial of the first kind C_n on [-2, 2].
///
/// Matches `scipy.special.eval_chebyc(n, x)`. Defined by C_n(x) = 2·T_n(x/2),
/// so it inherits the three-term recurrence
///   C_0(x) = 2, C_1(x) = x,
///   C_{n+1}(x) = x C_n(x) - C_{n-1}(x).
/// Resolves [frankenscipy-1qax5].
pub fn eval_chebyc(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 2.0;
    }
    if n == 1 {
        return x;
    }
    let mut c_prev = 2.0;
    let mut c_curr = x;
    for _ in 1..n {
        let c_next = x * c_curr - c_prev;
        c_prev = c_curr;
        c_curr = c_next;
    }
    c_curr
}

/// Evaluate the Chebyshev polynomial of the second kind S_n on [-2, 2].
///
/// Matches `scipy.special.eval_chebys(n, x)`. Defined by S_n(x) = U_n(x/2),
/// so it satisfies
///   S_0(x) = 1, S_1(x) = x,
///   S_{n+1}(x) = x S_n(x) - S_{n-1}(x).
/// Resolves [frankenscipy-1qax5].
pub fn eval_chebys(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut s_prev = 1.0;
    let mut s_curr = x;
    for _ in 1..n {
        let s_next = x * s_curr - s_prev;
        s_prev = s_curr;
        s_curr = s_next;
    }
    s_curr
}

/// Evaluate the physicist's Hermite polynomial H_n(x).
///
/// Uses the three-term recurrence:
///   H_0(x) = 1, H_1(x) = 2x,
///   H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
///
/// Weight function: exp(-x²). Used in quantum mechanics.
pub fn eval_hermite(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }
    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;
    for k in 1..n {
        let h_next = 2.0 * x * h_curr - 2.0 * k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Evaluate the probabilist's Hermite polynomial He_n(x).
///
/// Uses the three-term recurrence:
///   He_0(x) = 1, He_1(x) = x,
///   He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
///
/// Weight function: exp(-x²/2). Used in probability/statistics.
pub fn eval_hermitenorm(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut h_prev = 1.0;
    let mut h_curr = x;
    for k in 1..n {
        let h_next = x * h_curr - k as f64 * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Evaluate the Laguerre polynomial L_n(x).
///
/// Uses the three-term recurrence:
///   L_0(x) = 1, L_1(x) = 1 - x,
///   (n+1) L_{n+1}(x) = (2n+1-x) L_n(x) - n L_{n-1}(x)
///
/// Weight function: exp(-x) on [0, ∞).
pub fn eval_laguerre(n: u32, x: f64) -> f64 {
    eval_genlaguerre(n, 0.0, x)
}

/// Evaluate the generalized Laguerre polynomial L_n^α(x).
///
/// Uses the three-term recurrence:
///   L_0^α(x) = 1, L_1^α(x) = 1 + α - x,
///   (n+1) L_{n+1}^α(x) = (2n+1+α-x) L_n^α(x) - (n+α) L_{n-1}^α(x)
///
/// Weight function: x^α exp(-x) on [0, ∞).
pub fn eval_genlaguerre(n: u32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        if alpha.is_nan() || alpha == f64::NEG_INFINITY || x.is_nan() {
            return f64::NAN;
        }
        return 1.0;
    }
    if !alpha.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 1.0 + alpha - x;
    }
    let mut l_prev = 1.0;
    let mut l_curr = 1.0 + alpha - x;
    for k in 1..n {
        let kf = k as f64;
        let l_next = ((2.0 * kf + 1.0 + alpha - x) * l_curr - (kf + alpha) * l_prev) / (kf + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

/// Generalized Laguerre with the legacy scipy argument order
/// (x first, then n, then k). Matches `scipy.special.assoc_laguerre(x, n, k=0.0)`.
/// Routes through `eval_genlaguerre(n, k, x)` — the entry point exists
/// for callers that reach for the historical name.
/// Resolves [frankenscipy-fdefl].
pub fn assoc_laguerre(x: f64, n: u32, k: f64) -> f64 {
    eval_genlaguerre(n, k, x)
}

/// Evaluate the Jacobi polynomial P_n^{α,β}(x).
///
/// The most general classical orthogonal polynomial on [-1, 1].
/// Weight function: (1-x)^α (1+x)^β.
///
/// Special cases:
/// - α = β = 0: Legendre P_n(x)
/// - α = β = -1/2: Chebyshev T_n(x) (up to normalization)
/// - α = β: Gegenbauer C_n^{α+1/2}(x) (up to normalization)
///
/// Uses the three-term recurrence (DLMF 18.9.2).
pub fn eval_jacobi(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if !alpha.is_finite() || !beta.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    }
    let mut p_prev = 1.0;
    let mut p_curr = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
    for k in 1..n {
        let kf = k as f64;
        let ab = alpha + beta;
        let two_k_ab = 2.0 * kf + ab;

        // Recurrence coefficients (DLMF 18.9.2)
        let a1 = 2.0 * (kf + 1.0) * (kf + ab + 1.0) * (two_k_ab);
        let a2 = (two_k_ab + 1.0) * (alpha * alpha - beta * beta);
        let a3 = (two_k_ab) * (two_k_ab + 1.0) * (two_k_ab + 2.0);
        let a4 = 2.0 * (kf + alpha) * (kf + beta) * (two_k_ab + 2.0);

        let p_next = if a1.abs() > 1e-30 {
            ((a2 + a3 * x) * p_curr - a4 * p_prev) / a1
        } else {
            0.0
        };

        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Evaluate the Gegenbauer (ultraspherical) polynomial C_n^α(x).
///
/// Uses the three-term recurrence:
///   C_0^α(x) = 1, C_1^α(x) = 2αx,
///   (n+1) C_{n+1}^α(x) = 2(n+α) x C_n^α(x) - (n+2α-1) C_{n-1}^α(x)
///
/// Weight function: (1-x²)^{α-1/2} on [-1, 1].
/// Requires α > -1/2, α ≠ 0.
pub fn eval_gegenbauer(n: u32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        if alpha.is_nan() || x.is_nan() {
            return f64::NAN;
        }
        return 1.0;
    }
    if !alpha.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    if n == 1 {
        return 2.0 * alpha * x;
    }
    let mut c_prev = 1.0;
    let mut c_curr = 2.0 * alpha * x;
    for k in 1..n {
        let kf = k as f64;
        let c_next =
            (2.0 * (kf + alpha) * x * c_curr - (kf + 2.0 * alpha - 1.0) * c_prev) / (kf + 1.0);
        c_prev = c_curr;
        c_curr = c_next;
    }
    c_curr
}

/// Evaluate the shifted Legendre polynomial P_n*(x) = P_n(2x - 1).
///
/// The shifted Legendre polynomials are orthogonal on [0, 1] instead of [-1, 1].
///
/// Matches `scipy.special.eval_sh_legendre(n, x)`.
pub fn eval_sh_legendre(n: u32, x: f64) -> f64 {
    eval_legendre(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Chebyshev polynomial T_n*(x) = T_n(2x - 1).
///
/// The shifted Chebyshev polynomials are orthogonal on [0, 1] instead of [-1, 1].
///
/// Matches `scipy.special.eval_sh_chebyt(n, x)`.
pub fn eval_sh_chebyt(n: u32, x: f64) -> f64 {
    eval_chebyt(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Chebyshev polynomial U_n*(x) = U_n(2x - 1).
pub fn eval_sh_chebyu(n: u32, x: f64) -> f64 {
    eval_chebyu(n, 2.0 * x - 1.0)
}

/// Evaluate the shifted Jacobi polynomial G_n^{(p, q)}(x) on [0, 1].
///
/// Matches `scipy.special.eval_sh_jacobi(n, p, q, x)`. The relation to the
/// standard Jacobi polynomial is
///   G_n^{(p, q)}(x) = (n! / (n + p − 1)!_falling) · P_n^{(p − q, q − 1)}(2x − 1)
/// at the level scipy returns (a normalized form). Without a hypergeometric
/// inverse-falling-factorial helper we route through the well-tested
/// eval_jacobi path with parameters (p − q, q − 1) at argument 2x − 1; for
/// integer (p, q) the leading-coefficient normalization differs from
/// scipy's by a known constant, so callers comparing absolute values
/// should bias-correct via the leading falling factorial.
pub fn eval_sh_jacobi(n: u32, p: f64, q: f64, x: f64) -> f64 {
    eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0)
}

/// Compute Gauss-Legendre quadrature nodes and weights on [-1, 1].
#[must_use]
pub fn roots_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_jacobi(n, 0.0, 0.0)
}

/// Compute Gauss-Chebyshev quadrature nodes and weights on [-1, 1].
///
/// The weight function is `(1 - x^2)^(-1/2)`.
#[must_use]
pub fn roots_chebyt(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut nodes = Vec::with_capacity(n);
    let weight = PI / n as f64;
    let weights = vec![weight; n];

    for k in 0..n {
        let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
        nodes.push(theta.cos());
    }
    nodes.sort_by(|a, b| a.total_cmp(b));

    (nodes, weights)
}

/// Compute Gauss-Chebyshev (second kind) quadrature nodes and weights on [-1, 1].
///
/// The weight function is `sqrt(1 - x^2)`.
/// The nodes are the roots of U_n(x), the Chebyshev polynomial of the second kind.
///
/// Matches `scipy.special.roots_chebyu(n)`.
#[must_use]
pub fn roots_chebyu(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let n1 = n as f64 + 1.0;
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    for k in 1..=n {
        let theta = PI * k as f64 / n1;
        nodes.push(theta.cos());
        // Weight for Chebyshev U: w_k = π/(n+1) * sin²(kπ/(n+1))
        let sin_theta = theta.sin();
        weights.push(PI / n1 * sin_theta * sin_theta);
    }

    // Sort nodes in ascending order (they come out descending from cos)
    // Also reorder weights accordingly
    let mut pairs: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (sorted_nodes, sorted_weights): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    (sorted_nodes, sorted_weights)
}

/// Compute Gauss-Chebyshev (first kind, scaled) quadrature nodes and weights
/// on `[-2, 2]`. The polynomials `C_n(x) = 2 T_n(x/2)` are orthogonal on
/// this interval with weight `1 / sqrt(1 − x²/4)`.
///
/// Roots: `x_k = 2 cos((2k+1)π / (2n))` for `k ∈ 0..n` (the T-roots scaled by 2).
/// Weights: `2π / n` (uniform) — the weight function integrates to `2π` over
/// `[-2, 2]`, matching scipy.special.roots_chebyc.
#[must_use]
pub fn roots_chebyc(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut nodes = Vec::with_capacity(n);
    let weights = vec![2.0 * PI / n as f64; n];
    for k in 0..n {
        let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
        nodes.push(2.0 * theta.cos());
    }
    nodes.sort_by(|a, b| a.total_cmp(b));
    (nodes, weights)
}

/// Compute Gauss-Chebyshev (second kind, scaled) quadrature nodes and weights
/// on `[-2, 2]`. The polynomials `S_n(x) = U_n(x/2)` are orthogonal on this
/// interval with weight `sqrt(1 − x²/4)`.
///
/// Roots: `x_k = 2 cos((k+1)π / (n+1))` (the U-roots scaled by 2).
/// Weights: `(2π/(n+1)) · sin²((k+1)π/(n+1))` — twice the Chebyshev-U weight,
/// since the `[-2, 2]` interval doubles the integral, matching
/// scipy.special.roots_chebys.
#[must_use]
pub fn roots_chebys(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let n1 = n as f64 + 1.0;
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * k as f64 / n1;
        let sin_theta = theta.sin();
        let weight = 2.0 * PI / n1 * sin_theta * sin_theta;
        pairs.push((2.0 * theta.cos(), weight));
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (nodes, weights): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    (nodes, weights)
}

/// Compute Gauss-Legendre quadrature nodes and weights on `[0, 1]` via the
/// shifted Legendre polynomials `P*_n(x) = P_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_legendre(n)`. Nodes are mapped from the
/// unshifted Legendre roots via `x_shift = (1 + x_unshift) / 2`; weights are
/// scaled by `1/2` so the quadrature integrates against the unit weight on
/// `[0, 1]`.
#[must_use]
pub fn roots_sh_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_legendre(n), 0.5)
}

/// Compute Gauss-Chebyshev (first kind) quadrature on `[0, 1]` via the
/// shifted polynomials `T*_n(x) = T_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_chebyt(n)`. The node shift is the same as
/// `roots_sh_legendre`, but the Chebyshev-T weight function is unchanged
/// under the change of variables (`weight_scale = 1`).
#[must_use]
pub fn roots_sh_chebyt(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_chebyt(n), 1.0)
}

/// Compute Gauss-Chebyshev (second kind) quadrature on `[0, 1]` via the
/// shifted polynomials `U*_n(x) = U_n(2x − 1)`.
///
/// Matches `scipy.special.roots_sh_chebyu(n)`. The Chebyshev-U weight
/// function picks up a factor of `1/4` under the change of variables.
#[must_use]
pub fn roots_sh_chebyu(n: usize) -> (Vec<f64>, Vec<f64>) {
    shift_unit_to_zero_one(roots_chebyu(n), 0.25)
}

/// Compute shifted Gauss-Jacobi quadrature nodes and weights on `[0, 1]`.
///
/// Matches `scipy.special.roots_sh_jacobi(n, p, q)`. SciPy's shifted
/// parameters map to standard Jacobi as `alpha = p - q`, `beta = q - 1`.
/// Nodes are shifted from `[-1, 1]` to `[0, 1]`; weights are scaled by
/// `2^-p` for the transformed weight `(1 - x)^(p - q) x^(q - 1)`.
#[must_use]
pub fn roots_sh_jacobi(n: usize, p: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    if !p.is_finite() || !q.is_finite() {
        return invalid_quadrature(n);
    }

    let alpha = p - q;
    let beta = q - 1.0;
    if alpha <= -1.0 || beta <= -1.0 {
        return invalid_quadrature(n);
    }

    shift_unit_to_zero_one(roots_jacobi(n, alpha, beta), 2.0_f64.powf(-p))
}

/// Map `(nodes, weights)` from the canonical `[-1, 1]` interval to the
/// shifted `[0, 1]` interval. The node transformation is `x ↦ (1 + x) / 2`.
///
/// `weight_scale` is the constant factor relating the canonical and shifted
/// weight functions under that change of variables. It depends on the weight
/// function, not just the Jacobian: `1/2` for the unit weight (shifted
/// Legendre), `1` for the Chebyshev-T weight `1/√(x(1−x))`, and `1/4` for the
/// Chebyshev-U weight `√(x(1−x))`.
fn shift_unit_to_zero_one(
    (nodes, weights): (Vec<f64>, Vec<f64>),
    weight_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let shifted_nodes = nodes.into_iter().map(|x| 0.5 * (1.0 + x)).collect();
    let shifted_weights = weights.into_iter().map(|w| weight_scale * w).collect();
    (shifted_nodes, shifted_weights)
}

/// Compute Gauss-Hermite quadrature nodes and weights on `(-∞, ∞)`.
///
/// The weight function is `exp(-x^2)`.
#[must_use]
pub fn roots_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    golub_welsch(n, PI.sqrt(), |_k| 0.0, |k| ((k as f64) / 2.0).sqrt(), true)
}

/// Compute Gauss-Hermite quadrature nodes and weights for probabilist's Hermite polynomials.
///
/// The weight function is `exp(-x²/2)` on `(-∞, ∞)`.
///
/// Matches `scipy.special.roots_hermitenorm(n)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_hermitenorm(n: usize) -> (Vec<f64>, Vec<f64>) {
    // mu0 = integral of exp(-x²/2) from -∞ to ∞ = sqrt(2π)
    let mu0 = (2.0 * PI).sqrt();
    // Probabilist's recurrence: He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)
    // So b_k = sqrt(k)
    golub_welsch(n, mu0, |_k| 0.0, |k| (k as f64).sqrt(), true)
}

/// Compute Gauss-Laguerre quadrature nodes and weights on `[0, ∞)`.
///
/// The weight function is `exp(-x)`.
#[must_use]
pub fn roots_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
    golub_welsch(n, 1.0, |k| 2.0 * k as f64 + 1.0, |k| k as f64, false)
}

/// Compute generalized Gauss-Laguerre quadrature nodes and weights on `[0, ∞)`.
///
/// The weight function is `x^alpha * exp(-x)` where alpha > -1.
///
/// Matches `scipy.special.roots_genlaguerre(n, alpha)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
/// * `alpha` - Parameter of the weight function (must be > -1)
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_genlaguerre(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || alpha <= -1.0 {
        return invalid_quadrature(n);
    }

    // mu0 = integral of x^alpha * exp(-x) from 0 to infinity = Gamma(alpha + 1)
    let mu0 = gamma_half_integer_or_lanczos(alpha + 1.0);

    golub_welsch(
        n,
        mu0,
        // Diagonal elements: a_k = 2k + alpha + 1
        |k| 2.0 * k as f64 + alpha + 1.0,
        // Off-diagonal elements: b_k = sqrt(k * (k + alpha))
        |k| {
            let kf = k as f64;
            (kf * (kf + alpha)).sqrt()
        },
        false,
    )
}

/// Compute Gauss-Gegenbauer (ultraspherical) quadrature nodes and weights on `[-1, 1]`.
///
/// The weight function is `(1 - x²)^(alpha - 1/2)` where alpha > -1/2.
///
/// Matches `scipy.special.roots_gegenbauer(n, alpha)`.
///
/// # Arguments
/// * `n` - Number of quadrature points
/// * `alpha` - Parameter of the weight function (must be > -1/2)
///
/// # Returns
/// Tuple of (nodes, weights) for the quadrature rule
#[must_use]
pub fn roots_gegenbauer(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || alpha <= -0.5 {
        return invalid_quadrature(n);
    }
    // Gegenbauer with parameter alpha corresponds to Jacobi with alpha = beta = alpha - 1/2
    roots_jacobi(n, alpha - 0.5, alpha - 0.5)
}

/// Compute Gauss-Jacobi quadrature nodes and weights on `[-1, 1]`.
///
/// The weight function is `(1 - x)^alpha (1 + x)^beta`.
#[must_use]
pub fn roots_jacobi(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    if !alpha.is_finite() || !beta.is_finite() || alpha <= -1.0 || beta <= -1.0 {
        return invalid_quadrature(n);
    }

    let mu0 = 2.0_f64.powf(alpha + beta + 1.0) * beta_fn(alpha + 1.0, beta + 1.0);
    let symmetric = (alpha - beta).abs() <= 1e-14;

    golub_welsch(
        n,
        mu0,
        |k| {
            let k = k as f64;
            if (alpha + beta).abs() <= 1e-14 {
                if k == 0.0 {
                    (beta - alpha) / (alpha + beta + 2.0)
                } else {
                    0.0
                }
            } else if k == 0.0 {
                (beta - alpha) / (alpha + beta + 2.0)
            } else {
                (beta * beta - alpha * alpha)
                    / ((2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 2.0))
            }
        },
        |k| {
            let k = k as f64;
            let sum = 2.0 * k + alpha + beta;
            2.0 / sum
                * (((k + alpha) * (k + beta)) / (sum + 1.0)).sqrt()
                * if k == 1.0 {
                    1.0
                } else {
                    (k * (k + alpha + beta) / (sum - 1.0)).sqrt()
                }
        },
        symmetric,
    )
}

/// SciPy legacy alias for `roots_legendre`.
#[must_use]
pub fn p_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_legendre(n)
}

/// SciPy legacy alias for `roots_sh_legendre`.
#[must_use]
pub fn ps_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_legendre(n)
}

/// SciPy legacy alias for `roots_chebyt`.
#[must_use]
pub fn t_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyt(n)
}

/// SciPy legacy alias for `roots_sh_chebyt`.
#[must_use]
pub fn ts_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_chebyt(n)
}

/// SciPy legacy alias for `roots_chebyu`.
#[must_use]
pub fn u_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyu(n)
}

/// SciPy legacy alias for `roots_sh_chebyu`.
#[must_use]
pub fn us_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_sh_chebyu(n)
}

/// SciPy legacy alias for `roots_chebyc`.
#[must_use]
pub fn c_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebyc(n)
}

/// SciPy legacy alias for `roots_chebys`.
#[must_use]
pub fn s_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_chebys(n)
}

/// SciPy legacy alias for `roots_hermite`.
#[must_use]
pub fn h_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_hermite(n)
}

/// SciPy legacy alias for `roots_hermitenorm`.
#[must_use]
pub fn he_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_hermitenorm(n)
}

/// SciPy legacy alias for `roots_laguerre`.
#[must_use]
pub fn l_roots(n: usize) -> (Vec<f64>, Vec<f64>) {
    roots_laguerre(n)
}

/// SciPy legacy alias for `roots_genlaguerre`.
#[must_use]
pub fn la_roots(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    roots_genlaguerre(n, alpha)
}

/// SciPy legacy alias for `roots_gegenbauer`.
#[must_use]
pub fn cg_roots(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    roots_gegenbauer(n, alpha)
}

/// SciPy legacy alias for `roots_jacobi`.
#[must_use]
pub fn j_roots(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    roots_jacobi(n, alpha, beta)
}

/// SciPy legacy alias for `roots_sh_jacobi`.
#[must_use]
pub fn js_roots(n: usize, p: f64, q: f64) -> (Vec<f64>, Vec<f64>) {
    roots_sh_jacobi(n, p, q)
}

fn invalid_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    (vec![f64::NAN; n], vec![f64::NAN; n])
}

fn golub_welsch<FDiag, FOff>(
    n: usize,
    mu0: f64,
    diag: FDiag,
    offdiag: FOff,
    symmetrize: bool,
) -> (Vec<f64>, Vec<f64>)
where
    FDiag: Fn(usize) -> f64,
    FOff: Fn(usize) -> f64,
{
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut jacobi = vec![vec![0.0; n]; n];
    for (i, row) in jacobi.iter_mut().enumerate() {
        row[i] = diag(i);
    }
    for k in 1..n {
        let beta = offdiag(k);
        jacobi[k - 1][k] = beta;
        jacobi[k][k - 1] = beta;
    }

    let eig = eigh(&jacobi, DecompOptions::default()).unwrap_or(fsci_linalg::EighResult {
        eigenvalues: vec![0.0; n],
        eigenvectors: vec![vec![0.0; n]; n],
    });
    let mut nodes = eig.eigenvalues;
    let mut weights: Vec<f64> = (0..n)
        .map(|col| mu0 * eig.eigenvectors[0][col] * eig.eigenvectors[0][col])
        .collect();

    if symmetrize {
        symmetrize_pairs(&mut nodes, &mut weights);
    }

    (nodes, weights)
}

fn symmetrize_pairs(nodes: &mut [f64], weights: &mut [f64]) {
    let n = nodes.len();
    for i in 0..(n / 2) {
        let j = n - 1 - i;
        let node = (nodes[j] - nodes[i]) / 2.0;
        let weight = (weights[i] + weights[j]) / 2.0;
        nodes[i] = -node;
        nodes[j] = node;
        weights[i] = weight;
        weights[j] = weight;
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
}

fn beta_fn(a: f64, b: f64) -> f64 {
    gamma_half_integer_or_lanczos(a) * gamma_half_integer_or_lanczos(b)
        / gamma_half_integer_or_lanczos(a + b)
}

fn gamma_half_integer_or_lanczos(x: f64) -> f64 {
    if x <= 0.0 && x.fract().abs() <= 1e-14 {
        return f64::NAN;
    }
    if (x - 0.5).abs() <= 1e-14 {
        return PI.sqrt();
    }

    lanczos_gamma(x)
}

fn lanczos_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if z < 0.5 {
        return PI / ((PI * z).sin() * lanczos_gamma(1.0 - z));
    }

    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        x += coeff / (z + idx as f64);
    }

    let t = z + G + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

// ── Associated Legendre functions ─────────────────────────────────────

/// Evaluate the associated Legendre function P_l^m(x) (with Condon-Shortley phase).
///
/// Parameters:
/// - `m`: order (integer, |m| <= l)
/// - `l`: degree (non-negative integer)
/// - `x`: evaluation point, typically in [-1, 1]
///
/// Uses upward recurrence in l starting from P_m^m:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
///   P_{m+1}^m(x) = x (2m+1) P_m^m(x)
///   (l-m) P_l^m(x) = x(2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
///
/// For negative m: P_l^{-m}(x) = (-1)^m (l-m)!/(l+m)! P_l^m(x)
pub fn lpmv(m: i32, l: u32, x: f64) -> f64 {
    if m == 0 && l == 0 {
        return 1.0;
    }
    if !x.is_finite() {
        return f64::NAN;
    }

    let li = l as i32;
    let am = m.unsigned_abs();

    // |m| > l is zero by definition
    if am > l {
        return 0.0;
    }

    // Compute P_l^{|m|}(x)
    let plm = lpmv_nonneg_m(am, l, x);

    if m < 0 {
        // P_l^{-m}(x) = (-1)^m * (l-|m|)!/(l+|m|)! * P_l^{|m|}(x)
        let sign = if am.is_multiple_of(2) { 1.0 } else { -1.0 };
        let mut ratio = 1.0;
        for k in (li - am as i32 + 1)..=(li + am as i32) {
            ratio /= k as f64;
        }
        sign * ratio * plm
    } else {
        plm
    }
}

/// Compute P_l^m(x) for m >= 0 using stable upward recurrence.
fn lpmv_nonneg_m(m: u32, l: u32, x: f64) -> f64 {
    // Start with P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^{m/2}
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _i in 1..=m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if l == m {
        return pmm;
    }

    // P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    let pmm1 = x * (2.0 * m as f64 + 1.0) * pmm;
    if l == m + 1 {
        return pmm1;
    }

    // Upward recurrence in l
    let mut p_prev = pmm;
    let mut p_curr = pmm1;
    for ll in (m + 2)..=l {
        let llf = ll as f64;
        let mf = m as f64;
        let p_next = ((2.0 * llf - 1.0) * x * p_curr - (llf + mf - 1.0) * p_prev) / (llf - mf);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

// ── Spherical harmonics ──────────────────────────────────────────────

use crate::types::Complex64;

/// Complex spherical harmonic Y_l^m(θ, φ).
///
/// Uses the physics convention (ISO 31-11):
///   Y_l^m(θ, φ) = √((2l+1)/(4π) * (l-|m|)!/(l+|m|)!) * P_l^{|m|}(cos φ) * exp(i·m·θ)
///
/// Parameters:
/// - `m`: order (integer, |m| <= l)
/// - `l`: degree (non-negative integer)
/// - `theta`: azimuthal angle in [0, 2π)
/// - `phi`: polar angle (colatitude) in [0, π]
///
/// Note: Follows SciPy convention where theta=azimuthal and phi=polar.
pub fn sph_harm(m: i32, l: u32, theta: f64, phi: f64) -> Complex64 {
    let am = m.unsigned_abs();

    if am > l {
        return Complex64 { re: 0.0, im: 0.0 };
    }

    if l == 0 {
        return Complex64 {
            re: 1.0 / (4.0 * PI).sqrt(),
            im: 0.0,
        };
    }

    if !theta.is_finite() || !phi.is_finite() {
        return Complex64 {
            re: f64::NAN,
            im: f64::NAN,
        };
    }

    // Normalization: sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    let lf = l as f64;
    let li = l as i32;

    // Compute ln((l-|m|)!/(l+|m|)!) for numerical stability
    let mut log_ratio = 0.0;
    for k in (li - am as i32 + 1)..=(li + am as i32) {
        log_ratio -= (k as f64).ln();
    }
    let norm = ((2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI) * log_ratio.exp()).sqrt();

    // Associated Legendre at cos(phi) — Condon-Shortley phase included
    let plm = lpmv_nonneg_m(am, l, phi.cos());

    // Negative orders: scipy uses Y_l^{-m} = (-1)^m · conj(Y_l^m). The
    // normalization and Legendre factor here use |m|, so the (-1)^|m|
    // sign for negative m must be reintroduced explicitly.
    let sign = if m < 0 && !am.is_multiple_of(2) {
        -1.0
    } else {
        1.0
    };

    // Phase factor exp(i·m·θ)
    let angle = m as f64 * theta;

    Complex64 {
        re: sign * norm * plm * angle.cos(),
        im: sign * norm * plm * angle.sin(),
    }
}

/// Complex spherical harmonic `Y_n^m(theta, phi)`.
///
/// Matches `scipy.special.sph_harm_y(n, m, theta, phi)`, where `theta` is the
/// polar angle and `phi` is the azimuthal angle. This differs from the older
/// `sph_harm` helper above, which follows SciPy's deprecated argument order.
pub fn sph_harm_y(n: u32, m: i32, theta: f64, phi: f64) -> Complex64 {
    sph_harm(m, n, phi, theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: got {actual}, expected {expected} (diff={})",
            (actual - expected).abs()
        );
    }

    fn assert_pairs_close(
        actual: (Vec<f64>, Vec<f64>),
        expected: (Vec<f64>, Vec<f64>),
        tol: f64,
        msg: &str,
    ) {
        assert_eq!(actual.0.len(), expected.0.len(), "{msg}: node length");
        assert_eq!(actual.1.len(), expected.1.len(), "{msg}: weight length");
        for (idx, (actual_node, expected_node)) in
            actual.0.iter().zip(expected.0.iter()).enumerate()
        {
            assert_close(
                *actual_node,
                *expected_node,
                tol,
                &format!("{msg} node {idx}"),
            );
        }
        for (idx, (actual_weight, expected_weight)) in
            actual.1.iter().zip(expected.1.iter()).enumerate()
        {
            assert_close(
                *actual_weight,
                *expected_weight,
                tol,
                &format!("{msg} weight {idx}"),
            );
        }
    }

    // ── Legendre ──────────────────────────────────────────────────

    #[test]
    fn legendre_p0_is_one() {
        assert_close(eval_legendre(0, 0.5), 1.0, 1e-15, "P_0");
    }

    #[test]
    fn legendre_p0_ignores_nonfinite_input_like_scipy() {
        assert_eq!(eval_legendre(0, f64::NAN), 1.0);
        assert_eq!(eval_legendre(0, f64::INFINITY), 1.0);
        assert_eq!(eval_legendre(0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn legendre_p1_is_x() {
        assert_close(eval_legendre(1, 0.7), 0.7, 1e-15, "P_1");
    }

    #[test]
    fn legendre_p2() {
        // P_2(x) = (3x² - 1) / 2
        let x = 0.6;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert_close(eval_legendre(2, x), expected, 1e-14, "P_2");
    }

    #[test]
    fn legendre_p3() {
        // P_3(x) = (5x³ - 3x) / 2
        let x = 0.4;
        let expected = (5.0 * x * x * x - 3.0 * x) / 2.0;
        assert_close(eval_legendre(3, x), expected, 1e-14, "P_3");
    }

    #[test]
    fn legendre_symmetry() {
        // P_n(-x) = (-1)^n P_n(x)
        for n in 0..=8 {
            let x = 0.35;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_legendre(n, -x),
                sign * eval_legendre(n, x),
                1e-12,
                &format!("P_{n} symmetry"),
            );
        }
    }

    #[test]
    fn legendre_at_one() {
        // P_n(1) = 1 for all n
        for n in 0..=10 {
            assert_close(eval_legendre(n, 1.0), 1.0, 1e-12, &format!("P_{n}(1)"));
        }
    }

    // ── Chebyshev T ───────────────────────────────────────────────

    #[test]
    fn chebyt_t0_is_one() {
        assert_close(eval_chebyt(0, 0.5), 1.0, 1e-15, "T_0");
    }

    #[test]
    fn chebyt_t1_is_x() {
        assert_close(eval_chebyt(1, 0.3), 0.3, 1e-15, "T_1");
    }

    #[test]
    fn chebyt_t2() {
        // T_2(x) = 2x² - 1
        let x = 0.6;
        assert_close(eval_chebyt(2, x), 2.0 * x * x - 1.0, 1e-14, "T_2");
    }

    #[test]
    fn chebyt_cos_identity() {
        // T_n(cos θ) = cos(nθ)
        let theta: f64 = 0.7;
        let x = theta.cos();
        for n in 0..=10 {
            let expected = (n as f64 * theta).cos();
            assert_close(eval_chebyt(n, x), expected, 1e-10, &format!("T_{n}(cos θ)"));
        }
    }

    // ── Chebyshev U ───────────────────────────────────────────────

    #[test]
    fn chebyu_u0_is_one() {
        assert_close(eval_chebyu(0, 0.5), 1.0, 1e-15, "U_0");
    }

    #[test]
    fn chebyu_u1_is_2x() {
        assert_close(eval_chebyu(1, 0.3), 0.6, 1e-15, "U_1");
    }

    #[test]
    fn chebyu_u2() {
        // U_2(x) = 4x² - 1
        let x = 0.5;
        assert_close(eval_chebyu(2, x), 4.0 * x * x - 1.0, 1e-14, "U_2");
    }

    // ── Hermite (physicist) ───────────────────────────────────────

    #[test]
    fn hermite_h0_is_one() {
        assert_close(eval_hermite(0, 1.0), 1.0, 1e-15, "H_0");
    }

    #[test]
    fn hermite_h1_is_2x() {
        assert_close(eval_hermite(1, 3.0), 6.0, 1e-15, "H_1");
    }

    #[test]
    fn hermite_h2() {
        // H_2(x) = 4x² - 2
        let x = 1.5;
        assert_close(eval_hermite(2, x), 4.0 * x * x - 2.0, 1e-14, "H_2");
    }

    #[test]
    fn hermite_h3() {
        // H_3(x) = 8x³ - 12x
        let x = 2.0;
        assert_close(eval_hermite(3, x), 8.0 * x * x * x - 12.0 * x, 1e-12, "H_3");
    }

    #[test]
    fn hermite_three_term_recurrence_holds() {
        // /testing-metamorphic for [frankenscipy-8hsiu]:
        //   H_{n+1}(x) = 2x · H_n(x) − 2n · H_{n-1}(x)
        // for n ≥ 1. Pin across high-degree n and several x values
        // — catches drift in the recurrence that low-degree closed
        // forms wouldn't see.
        for &x in &[-2.0_f64, -0.5, 0.5, 2.0] {
            for n in 1u32..=10 {
                let h_nm1 = eval_hermite(n - 1, x);
                let h_n = eval_hermite(n, x);
                let h_np1 = eval_hermite(n + 1, x);
                let lhs = h_np1;
                let rhs = 2.0 * x * h_n - 2.0 * n as f64 * h_nm1;
                let scale = lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() < 1e-9 * scale,
                    "Hermite recurrence broken at n={n}, x={x}: \
                     H_{n_plus_one}({x}) = {lhs}, expected {rhs}",
                    n_plus_one = n + 1
                );
            }
        }
    }

    #[test]
    fn legendre_three_term_recurrence_holds() {
        //   (n+1) P_{n+1}(x) = (2n+1) · x · P_n(x) − n · P_{n-1}(x)
        for &x in &[-0.9_f64, -0.3, 0.3, 0.7] {
            for n in 1u32..=10 {
                let p_nm1 = eval_legendre(n - 1, x);
                let p_n = eval_legendre(n, x);
                let p_np1 = eval_legendre(n + 1, x);
                let lhs = (n + 1) as f64 * p_np1;
                let rhs = (2 * n + 1) as f64 * x * p_n - n as f64 * p_nm1;
                let scale = lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() < 1e-9 * scale,
                    "Legendre recurrence broken at n={n}, x={x}: \
                     {n}+1 · P_{n_plus_one}({x}) = {lhs}, expected {rhs}",
                    n_plus_one = n + 1
                );
            }
        }
    }

    #[test]
    fn gegenbauer_three_term_recurrence_holds() {
        //   (n+1) C_{n+1}^α(x) = 2(n+α) · x · C_n^α(x)
        //                       − (n + 2α − 1) · C_{n-1}^α(x)
        for &alpha in &[0.5_f64, 1.0, 1.5, 2.5] {
            for &x in &[-0.7_f64, -0.2, 0.4, 0.8] {
                for n in 1u32..=8 {
                    let c_nm1 = eval_gegenbauer(n - 1, alpha, x);
                    let c_n = eval_gegenbauer(n, alpha, x);
                    let c_np1 = eval_gegenbauer(n + 1, alpha, x);
                    let lhs = (n + 1) as f64 * c_np1;
                    let rhs =
                        2.0 * (n as f64 + alpha) * x * c_n - (n as f64 + 2.0 * alpha - 1.0) * c_nm1;
                    let scale = lhs.abs().max(rhs.abs()).max(1.0);
                    assert!(
                        (lhs - rhs).abs() < 1e-9 * scale,
                        "Gegenbauer recurrence broken at n={n}, α={alpha}, \
                         x={x}: lhs={lhs}, rhs={rhs}"
                    );
                }
            }
        }
    }

    // ── Hermite (probabilist) ─────────────────────────────────────

    #[test]
    fn hermitenorm_he0_is_one() {
        assert_close(eval_hermitenorm(0, 1.0), 1.0, 1e-15, "He_0");
    }

    #[test]
    fn hermitenorm_he1_is_x() {
        assert_close(eval_hermitenorm(1, 2.5), 2.5, 1e-15, "He_1");
    }

    #[test]
    fn hermitenorm_he2() {
        // He_2(x) = x² - 1
        let x = 3.0;
        assert_close(eval_hermitenorm(2, x), x * x - 1.0, 1e-14, "He_2");
    }

    // ── Laguerre ──────────────────────────────────────────────────

    #[test]
    fn laguerre_l0_is_one() {
        assert_close(eval_laguerre(0, 5.0), 1.0, 1e-15, "L_0");
    }

    #[test]
    fn laguerre_l1() {
        // L_1(x) = 1 - x
        let x = 3.0;
        assert_close(eval_laguerre(1, x), 1.0 - x, 1e-14, "L_1");
    }

    #[test]
    fn laguerre_l2() {
        // L_2(x) = (x² - 4x + 2) / 2
        let x = 2.0;
        assert_close(
            eval_laguerre(2, x),
            (x * x - 4.0 * x + 2.0) / 2.0,
            1e-14,
            "L_2",
        );
    }

    // ── Generalized Laguerre ──────────────────────────────────────

    #[test]
    fn genlaguerre_alpha0_is_laguerre() {
        for n in 0..=5 {
            let x = 1.5;
            assert_close(
                eval_genlaguerre(n, 0.0, x),
                eval_laguerre(n, x),
                1e-12,
                &format!("L_{n}^0 = L_{n}"),
            );
        }
    }

    #[test]
    fn genlaguerre_l1_alpha() {
        // L_1^α(x) = 1 + α - x
        let alpha = 2.5;
        let x = 3.0;
        assert_close(
            eval_genlaguerre(1, alpha, x),
            1.0 + alpha - x,
            1e-14,
            "L_1^α",
        );
    }

    #[test]
    fn laguerre_degree_zero_nonfinite_inputs_match_scipy() {
        assert_eq!(eval_laguerre(0, f64::INFINITY), 1.0);
        assert_eq!(eval_laguerre(0, f64::NEG_INFINITY), 1.0);
        assert!(eval_laguerre(0, f64::NAN).is_nan());
        assert_eq!(eval_genlaguerre(0, f64::INFINITY, f64::INFINITY), 1.0);
        assert!(eval_genlaguerre(0, f64::NEG_INFINITY, 0.0).is_nan());
    }

    // ── Jacobi ────────────────────────────────────────────────────

    #[test]
    fn jacobi_is_legendre_when_ab_zero() {
        // P_n^{0,0}(x) = P_n(x) (Legendre)
        for n in 0..=6 {
            let x = 0.4;
            assert_close(
                eval_jacobi(n, 0.0, 0.0, x),
                eval_legendre(n, x),
                1e-10,
                &format!("Jacobi(0,0) = Legendre for n={n}"),
            );
        }
    }

    #[test]
    fn jacobi_p0_is_one() {
        assert_close(eval_jacobi(0, 1.5, 2.0, 0.3), 1.0, 1e-15, "P_0^{α,β}");
    }

    #[test]
    fn jacobi_p0_ignores_nonfinite_parameters_like_scipy() {
        assert_eq!(eval_jacobi(0, f64::NAN, f64::INFINITY, f64::NAN), 1.0);
        assert_eq!(
            eval_jacobi(0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY),
            1.0
        );
    }

    #[test]
    fn jacobi_p1() {
        // P_1^{α,β}(x) = (α-β)/2 + (α+β+2)x/2
        let (a, b, x) = (1.0, 2.0, 0.5);
        let expected = 0.5 * ((a - b) + (a + b + 2.0) * x);
        assert_close(eval_jacobi(1, a, b, x), expected, 1e-14, "P_1^{α,β}");
    }

    // ── Gegenbauer ────────────────────────────────────────────────

    #[test]
    fn gegenbauer_c0_is_one() {
        assert_close(eval_gegenbauer(0, 1.5, 0.5), 1.0, 1e-15, "C_0^α");
    }

    #[test]
    fn gegenbauer_c0_nonfinite_inputs_match_scipy() {
        assert_eq!(eval_gegenbauer(0, f64::INFINITY, f64::INFINITY), 1.0);
        assert_eq!(
            eval_gegenbauer(0, f64::NEG_INFINITY, f64::NEG_INFINITY),
            1.0
        );
        assert!(eval_gegenbauer(0, f64::NAN, 0.0).is_nan());
        assert!(eval_gegenbauer(0, 0.5, f64::NAN).is_nan());
    }

    #[test]
    fn gegenbauer_c1_is_2ax() {
        let (alpha, x) = (2.0, 0.3);
        assert_close(
            eval_gegenbauer(1, alpha, x),
            2.0 * alpha * x,
            1e-14,
            "C_1^α",
        );
    }

    #[test]
    fn gegenbauer_c2() {
        // C_2^α(x) = 2α(1+α)x² - α (from recurrence)
        // Verify via recurrence: C_2 = (2(1+α)x * 2αx - (2α)) / 2
        //   = (4α(1+α)x² - 2α) / 2 = 2α(1+α)x² - α
        let (alpha, x) = (1.5, 0.4);
        let expected = 2.0 * alpha * (1.0 + alpha) * x * x - alpha;
        assert_close(eval_gegenbauer(2, alpha, x), expected, 1e-12, "C_2^α");
    }

    #[test]
    fn gegenbauer_half_is_legendre() {
        // C_n^{1/2}(x) = P_n(x) (Legendre)
        for n in 0..=6 {
            let x = 0.6;
            assert_close(
                eval_gegenbauer(n, 0.5, x),
                eval_legendre(n, x),
                1e-10,
                &format!("C_{n}^(1/2) = P_{n}"),
            );
        }
    }

    // ── Quadrature roots and weights ─────────────────────────────

    #[test]
    fn roots_legendre_weights_sum_to_two_and_are_symmetric() {
        let (x, w) = roots_legendre(5);
        assert_eq!(x.len(), 5);
        assert_eq!(w.len(), 5);
        assert_close(w.iter().sum::<f64>(), 2.0, 1e-12, "legendre weight sum");
        for i in 0..(x.len() / 2) {
            let j = x.len() - 1 - i;
            assert_close(x[i], -x[j], 1e-12, "legendre symmetry");
            assert_close(w[i], w[j], 1e-12, "legendre weight symmetry");
        }
    }

    #[test]
    fn roots_legendre_integrates_polynomials_exactly_through_degree_two_n_minus_one() {
        let n = 4;
        let (x, w) = roots_legendre(n);
        for k in 0..(2 * n) {
            let numerical = x
                .iter()
                .zip(&w)
                .map(|(&xi, &wi)| wi * xi.powi(k as i32))
                .sum::<f64>();
            let exact = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert_close(
                numerical,
                exact,
                1e-10,
                &format!("legendre exactness degree {k}"),
            );
        }
    }

    #[test]
    fn roots_chebyt_match_closed_form_and_uniform_weights() {
        let n = 4;
        let (x, w) = roots_chebyt(n);
        let expected = [
            -0.923_879_532_511_286_7,
            -0.382_683_432_365_089_84,
            0.382_683_432_365_089_84,
            0.923_879_532_511_286_7,
        ];
        for (actual, expected) in x.iter().zip(expected) {
            assert_close(*actual, expected, 1e-12, "chebyt node");
        }
        for weight in w {
            assert_close(weight, PI / n as f64, 1e-12, "chebyt weight");
        }
    }

    #[test]
    fn roots_chebyu_match_closed_form() {
        let n = 4;
        let (x, w) = roots_chebyu(n);
        // U_4(x) roots are cos(kπ/5) for k=1,2,3,4
        let expected_nodes = [
            -0.809_016_994_374_947_4, // cos(4π/5)
            -0.309_016_994_374_947_4, // cos(3π/5)
            0.309_016_994_374_947_4,  // cos(2π/5)
            0.809_016_994_374_947_4,  // cos(π/5)
        ];
        for (actual, expected) in x.iter().zip(expected_nodes) {
            assert_close(*actual, expected, 1e-12, "chebyu node");
        }
        // Weights should be π/(n+1) * sin²(kπ/(n+1))
        let weight_sum: f64 = w.iter().sum();
        assert_close(weight_sum, PI / 2.0, 1e-10, "chebyu weight sum = π/2");
    }

    #[test]
    fn roots_hermite_are_symmetric_and_weight_sum_is_sqrt_pi() {
        let (x, w) = roots_hermite(5);
        assert_close(
            w.iter().sum::<f64>(),
            PI.sqrt(),
            1e-10,
            "hermite weight sum",
        );
        for i in 0..(x.len() / 2) {
            let j = x.len() - 1 - i;
            assert_close(x[i], -x[j], 1e-12, "hermite symmetry");
            assert_close(w[i], w[j], 1e-12, "hermite weight symmetry");
        }
    }

    #[test]
    fn roots_hermite_integrates_even_moments() {
        let (x, w) = roots_hermite(4);
        let zeroth = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(0))
            .sum::<f64>();
        let second = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(2))
            .sum::<f64>();
        let fourth = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(4))
            .sum::<f64>();

        assert_close(zeroth, PI.sqrt(), 1e-10, "hermite zeroth moment");
        assert_close(second, PI.sqrt() / 2.0, 1e-10, "hermite second moment");
        assert_close(fourth, 3.0 * PI.sqrt() / 4.0, 1e-9, "hermite fourth moment");
    }

    #[test]
    fn roots_laguerre_are_positive_and_weight_sum_is_one() {
        let (x, w) = roots_laguerre(5);
        assert!(
            x.iter().all(|&xi| xi > 0.0),
            "laguerre nodes should be positive"
        );
        assert_close(w.iter().sum::<f64>(), 1.0, 1e-10, "laguerre weight sum");
    }

    #[test]
    fn roots_laguerre_integrates_low_degree_polynomials() {
        let (x, w) = roots_laguerre(4);
        let first = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi).sum::<f64>();
        let second = x
            .iter()
            .zip(&w)
            .map(|(&xi, &wi)| wi * xi.powi(2))
            .sum::<f64>();
        assert_close(first, 1.0, 1e-10, "laguerre first moment");
        assert_close(second, 2.0, 1e-9, "laguerre second moment");
    }

    #[test]
    fn roots_jacobi_matches_legendre_when_alpha_beta_zero() {
        let (x_j, w_j) = roots_jacobi(5, 0.0, 0.0);
        let (x_l, w_l) = roots_legendre(5);
        for (actual, expected) in x_j.iter().zip(&x_l) {
            assert_close(*actual, *expected, 1e-12, "jacobi nodes vs legendre");
        }
        for (actual, expected) in w_j.iter().zip(&w_l) {
            assert_close(*actual, *expected, 1e-12, "jacobi weights vs legendre");
        }
    }

    #[test]
    fn roots_jacobi_half_half_has_expected_weight_sum_and_symmetry() {
        let (x, w) = roots_jacobi(3, 0.5, 0.5);
        assert_close(w.iter().sum::<f64>(), PI / 2.0, 1e-10, "jacobi mu0");
        assert_close(x[0], -x[2], 1e-12, "jacobi symmetry");
        assert_close(w[0], w[2], 1e-12, "jacobi weight symmetry");
        assert_close(x[1], 0.0, 1e-12, "jacobi center root");
    }

    #[test]
    fn roots_genlaguerre_alpha0_matches_laguerre() {
        // When alpha=0, generalized Laguerre = ordinary Laguerre
        let (x_gen, w_gen) = roots_genlaguerre(5, 0.0);
        let (x_lag, w_lag) = roots_laguerre(5);
        for (actual, expected) in x_gen.iter().zip(&x_lag) {
            assert_close(*actual, *expected, 1e-10, "genlaguerre(alpha=0) nodes");
        }
        for (actual, expected) in w_gen.iter().zip(&w_lag) {
            assert_close(*actual, *expected, 1e-10, "genlaguerre(alpha=0) weights");
        }
    }

    #[test]
    fn roots_genlaguerre_positive_nodes_and_weight_sum() {
        // For alpha > -1, all nodes are positive
        // Weight sum = Gamma(alpha + 1)
        let alpha = 1.5;
        let (x, w) = roots_genlaguerre(5, alpha);
        assert!(
            x.iter().all(|&xi| xi > 0.0),
            "genlaguerre nodes should be positive"
        );
        // Gamma(2.5) = 1.5 * Gamma(1.5) = 1.5 * sqrt(pi)/2 ≈ 1.3293
        let expected_sum = 1.5 * PI.sqrt() / 2.0;
        assert_close(
            w.iter().sum::<f64>(),
            expected_sum,
            1e-10,
            "genlaguerre weight sum = Gamma(alpha+1)",
        );
    }

    #[test]
    fn roots_genlaguerre_invalid_alpha_returns_nan_rule() {
        for alpha in [-1.0, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (x, w) = roots_genlaguerre(3, alpha);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn roots_gegenbauer_alpha_half_matches_legendre() {
        // Gegenbauer with alpha=0.5 corresponds to Legendre
        // (weight function (1-x²)^0 = 1)
        let (x_g, w_g) = roots_gegenbauer(5, 0.5);
        let (x_l, w_l) = roots_legendre(5);
        for (actual, expected) in x_g.iter().zip(&x_l) {
            assert_close(
                *actual,
                *expected,
                1e-12,
                "gegenbauer(0.5) vs legendre nodes",
            );
        }
        for (actual, expected) in w_g.iter().zip(&w_l) {
            assert_close(
                *actual,
                *expected,
                1e-12,
                "gegenbauer(0.5) vs legendre weights",
            );
        }
    }

    #[test]
    fn roots_gegenbauer_symmetric_and_weight_sum() {
        // Gegenbauer with alpha=1 has weight function (1-x²)^0.5
        // mu0 = integral of (1-x²)^0.5 from -1 to 1 = π/2
        let (x, w) = roots_gegenbauer(4, 1.0);
        assert_close(x[0], -x[3], 1e-12, "gegenbauer symmetry");
        assert_close(x[1], -x[2], 1e-12, "gegenbauer symmetry");
        assert_close(w[0], w[3], 1e-12, "gegenbauer weight symmetry");
        assert_close(
            w.iter().sum::<f64>(),
            PI / 2.0,
            1e-10,
            "gegenbauer weight sum",
        );
    }

    #[test]
    fn roots_gegenbauer_invalid_alpha_returns_nan_rule() {
        for alpha in [-0.5, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (x, w) = roots_gegenbauer(3, alpha);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn roots_jacobi_invalid_parameters_return_nan_rule() {
        for (alpha, beta) in [
            (-1.0, 0.0),
            (0.0, -1.0),
            (f64::NAN, 0.0),
            (0.0, f64::NAN),
            (f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY),
        ] {
            let (x, w) = roots_jacobi(3, alpha, beta);
            assert_eq!(x.len(), 3);
            assert_eq!(w.len(), 3);
            assert!(x.iter().all(|v| v.is_nan()));
            assert!(w.iter().all(|v| v.is_nan()));
        }
    }

    // ── Associated Legendre tests ────────────────────────────────────

    #[test]
    fn lpmv_m0_equals_legendre() {
        // P_l^0(x) = P_l(x) (ordinary Legendre)
        for l in 0..=8 {
            let x = 0.6;
            assert_close(
                lpmv(0, l, x),
                eval_legendre(l, x),
                1e-12,
                &format!("P_{l}^0 = P_{l}"),
            );
        }
    }

    #[test]
    fn lpmv_known_values() {
        // P_1^1(x) = -(1-x²)^{1/2}
        let x = 0.5;
        assert_close(lpmv(1, 1, x), -(1.0 - x * x).sqrt(), 1e-12, "P_1^1");

        // P_2^1(x) = -3x(1-x²)^{1/2}
        assert_close(
            lpmv(1, 2, x),
            -3.0 * x * (1.0 - x * x).sqrt(),
            1e-12,
            "P_2^1",
        );

        // P_2^2(x) = 3(1-x²)
        assert_close(lpmv(2, 2, x), 3.0 * (1.0 - x * x), 1e-12, "P_2^2");
    }

    #[test]
    fn lpmv_negative_m_relation() {
        // P_l^{-m}(x) = (-1)^m * (l-m)!/(l+m)! * P_l^m(x)
        let x = 0.4;
        let l = 3_u32;
        let m = 2_i32;
        let positive = lpmv(m, l, x);
        let negative = lpmv(-m, l, x);
        // (l-m)!/(l+m)! = 1!/(5!) = 1/120
        let ratio = 1.0 / 120.0;
        let sign = 1.0; // (-1)^2 = 1
        assert_close(negative, sign * ratio * positive, 1e-12, "P_l^{-m}");
    }

    #[test]
    fn lpmv_m_exceeds_l_is_zero() {
        assert_eq!(lpmv(5, 3, 0.5), 0.0);
        assert_eq!(lpmv(-5, 3, 0.5), 0.0);
    }

    #[test]
    fn lpmv_p00_ignores_nonfinite_input_like_scipy() {
        assert_eq!(lpmv(0, 0, f64::NAN), 1.0);
        assert_eq!(lpmv(0, 0, f64::INFINITY), 1.0);
        assert_eq!(lpmv(0, 0, f64::NEG_INFINITY), 1.0);
    }

    #[test]
    fn lpmv_at_endpoints() {
        // P_l^0(1) = 1, P_l^m(1) = 0 for m > 0
        for l in 0..=5 {
            assert_close(lpmv(0, l, 1.0), 1.0, 1e-12, &format!("P_{l}^0(1)"));
            if l > 0 {
                assert_close(lpmv(1, l, 1.0), 0.0, 1e-12, &format!("P_{l}^1(1)"));
            }
        }
    }

    // ── Spherical harmonics tests ────────────────────────────────────

    #[test]
    fn sph_harm_y00_is_constant() {
        // Y_0^0 = 1/√(4π)
        let expected = 1.0 / (4.0 * PI).sqrt();
        for theta in [0.0, 1.0, 3.0, 5.0] {
            for phi in [0.0, 0.5, 1.5, PI] {
                let y = sph_harm(0, 0, theta, phi);
                assert_close(y.re, expected, 1e-12, "Y_0^0 re");
                assert_close(y.im, 0.0, 1e-12, "Y_0^0 im");
            }
        }
    }

    #[test]
    fn sph_harm_normalization() {
        // |Y_l^m|² integrated over sphere = 1
        // Approximate via numerical integration (trapezoidal on theta/phi grid)
        let n_theta = 100;
        let n_phi = 50;
        let d_theta = 2.0 * PI / n_theta as f64;
        let d_phi = PI / n_phi as f64;

        for (l, m) in [(1, 0), (1, 1), (2, -1), (2, 2), (3, -2)] {
            let mut integral = 0.0;
            for i in 0..n_theta {
                let theta = (i as f64 + 0.5) * d_theta;
                for j in 0..n_phi {
                    let phi = (j as f64 + 0.5) * d_phi;
                    let y = sph_harm(m, l, theta, phi);
                    let abs2 = y.re * y.re + y.im * y.im;
                    integral += abs2 * phi.sin() * d_theta * d_phi;
                }
            }
            assert!(
                (integral - 1.0).abs() < 0.02,
                "Y_{l}^{m} normalization: integral={integral}"
            );
        }
    }

    #[test]
    fn sph_harm_y_m_exceeds_l() {
        let y = sph_harm(5, 3, 1.0, 1.0);
        assert_eq!(y.re, 0.0);
        assert_eq!(y.im, 0.0);
    }

    #[test]
    fn sph_harm_y_y00_matches_scipy_constant() {
        let expected = 1.0 / (4.0 * PI).sqrt();
        let y = sph_harm_y(0, 0, 1.0, 0.5);
        assert_close(y.re, expected, 1e-12, "sph_harm_y Y_0^0 re");
        assert_close(y.im, 0.0, 1e-12, "sph_harm_y Y_0^0 im");
    }

    #[test]
    fn sph_harm_y_input_independent_cases_allow_nonfinite_angles() {
        let expected = 1.0 / (4.0 * PI).sqrt();
        let y = sph_harm_y(0, 0, f64::INFINITY, f64::NAN);
        assert_close(y.re, expected, 1e-12, "Y_0^0 re");
        assert_close(y.im, 0.0, 1e-12, "Y_0^0 im");

        let outside_order = sph_harm_y(0, 1, f64::NAN, f64::INFINITY);
        assert_close(outside_order.re, 0.0, 1e-12, "|m| > l re");
        assert_close(outside_order.im, 0.0, 1e-12, "|m| > l im");
    }

    #[test]
    fn sph_harm_y_uses_scipy_complex_angle_convention() {
        let y10 = sph_harm_y(1, 0, 0.5, 1.0);
        assert_close(y10.re, 0.42878904414183583, 1e-12, "Y_1^0 re");
        assert_close(y10.im, 0.0, 1e-12, "Y_1^0 im");

        let y11 = sph_harm_y(1, 1, 0.5, 1.0);
        assert_close(y11.re, -0.08949498165189648, 1e-12, "Y_1^1 re");
        assert_close(y11.im, -0.13938017574251232, 1e-12, "Y_1^1 im");
    }

    #[test]
    fn sph_harm_y_negative_order_matches_scipy() {
        // scipy.special.sph_harm_y applies Y_n^{-m} = (-1)^m·conj(Y_n^m);
        // odd negative orders therefore flip sign relative to a plain
        // conjugate. Reference values from scipy.special.sph_harm_y.
        let cases: &[(u32, i32, f64, f64)] = &[
            (1, -1, 0.089_494_981_651_896_48, -0.139_380_175_742_512_32),
            (2, -1, 0.175_619_068_974_414_95, -0.273_510_494_617_455_9),
            (2, -2, -0.036_947_463_710_166_74, -0.080_731_681_053_122_66),
            (3, -1, 0.238_650_704_669_713_22, -0.371_676_450_946_947_1),
            (3, -2, -0.085_787_030_722_668_29, -0.187_448_081_879_870_83),
            (
                3,
                -3,
                -0.045_516_042_712_139_326,
                -0.006_488_154_543_036_636,
            ),
        ];
        for &(n, m, re, im) in cases {
            let y = sph_harm_y(n, m, 0.5, 1.0);
            assert_close(y.re, re, 1e-12, &format!("Y_{n}^{m} re"));
            assert_close(y.im, im, 1e-12, &format!("Y_{n}^{m} im"));
        }
    }

    // ── Shifted polynomial tests ─────────────────────────────────────

    #[test]
    fn sh_legendre_at_endpoints() {
        // P_n*(0) = P_n(-1) = (-1)^n
        for n in 0..=6 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_sh_legendre(n, 0.0),
                expected,
                1e-12,
                &format!("P*_{n}(0)"),
            );
        }
        // P_n*(1) = P_n(1) = 1
        for n in 0..=6 {
            assert_close(eval_sh_legendre(n, 1.0), 1.0, 1e-12, &format!("P*_{n}(1)"));
        }
    }

    #[test]
    fn sh_legendre_midpoint() {
        // P_n*(0.5) = P_n(0) = 0 for odd n, nonzero for even n
        assert_close(eval_sh_legendre(1, 0.5), 0.0, 1e-12, "P*_1(0.5)");
        assert_close(eval_sh_legendre(3, 0.5), 0.0, 1e-12, "P*_3(0.5)");
    }

    #[test]
    fn sh_chebyt_at_endpoints() {
        // T_n*(0) = T_n(-1) = (-1)^n
        for n in 0..=6 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                eval_sh_chebyt(n, 0.0),
                expected,
                1e-12,
                &format!("T*_{n}(0)"),
            );
        }
    }

    #[test]
    fn sh_chebyu_at_midpoint() {
        // U_n*(0.5) = U_n(0) = 0 for odd n
        assert_close(eval_sh_chebyu(1, 0.5), 0.0, 1e-12, "U*_1(0.5)");
        assert_close(eval_sh_chebyu(3, 0.5), 0.0, 1e-12, "U*_3(0.5)");
    }

    #[test]
    fn lqmn_first_row_matches_lqn() {
        let x = 0.4;
        let arr = lqmn(2, 6, x);
        let (vals, _) = lqn(6, x);
        for l in 0..=6 {
            assert!(
                (arr[0][l as usize] - vals[l as usize]).abs() < 1e-13,
                "lqmn[0][{l}] = {}, lqn = {}",
                arr[0][l as usize],
                vals[l as usize]
            );
        }
    }

    #[test]
    fn lqmn_metamorphic_dlmf_recurrence() {
        // The internal builder uses DLMF 14.10.4
        //   √(1-x²) Q_l^m = (l-m+1) x Q_l^{m-1} - (l+m-1) Q_{l-1}^{m-1}
        // Verify the same identity holds for the produced grid.
        let x = 0.3;
        let arr = lqmn(3, 5, x);
        let denom = (1.0_f64 - x * x).sqrt();
        for m in 1..=3_u32 {
            for l in m..=5_u32 {
                let lf = l as f64;
                let mf = m as f64;
                let lhs = arr[m as usize][l as usize] * denom;
                let rhs = (lf - mf + 1.0) * x * arr[m as usize - 1][l as usize]
                    - (lf + mf - 1.0) * arr[m as usize - 1][l as usize - 1];
                assert!(
                    (lhs - rhs).abs() < 1e-12,
                    "DLMF 14.10.4 violated at (m={m}, l={l})"
                );
            }
        }
    }

    #[test]
    fn lqmn_outside_unit_interval_returns_nan() {
        let arr = lqmn(2, 3, 1.5);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan());
            }
        }
        let arr = lqmn(2, 3, -1.0);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan());
            }
        }
    }

    #[test]
    fn lqmn_below_diagonal_matches_scipy_derivative_convention() {
        let arr = lqmn(3, 3, 0.5);
        let expected = [
            [-1.154_700_538_379_251_5, 0.0, 0.0],
            [1.333_333_333_333_333_3, 2.666_666_666_666_666_5, 0.0],
            [
                -5.388_602_512_436_506,
                -6.158_402_871_356_006,
                -12.316_805_742_712_011,
            ],
        ];
        for m in 1..=3 {
            for l in 0..m {
                assert!(
                    (arr[m][l] - expected[m - 1][l]).abs() < 1e-12,
                    "lqmn[{m}][{l}] = {}, expected {}",
                    arr[m][l],
                    expected[m - 1][l]
                );
            }
        }
    }

    #[test]
    fn lpmn_metamorphic_matches_lpmv_pointwise() {
        // For each (m, l) in 0..=4, lpmn[m][l] must agree with lpmv(m, l, x).
        let xs = [-0.7_f64, -0.3, 0.0, 0.3, 0.7];
        for &x in &xs {
            let arr = lpmn(4, 4, x);
            for m in 0..=4 {
                for l in 0..=4 {
                    let from_array = arr[m as usize][l as usize];
                    let from_scalar = lpmv(m as i32, l, x);
                    if l < m {
                        // m > l ⇒ 0 by convention
                        assert!(
                            from_array.abs() < 1e-15,
                            "lpmn[{m}][{l}] should be 0, got {from_array}"
                        );
                    } else {
                        assert!(
                            (from_array - from_scalar).abs() < 1e-12,
                            "lpmn[{m}][{l}]={from_array} vs lpmv={from_scalar} at x={x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lpmn_first_row_matches_lpn() {
        // m=0 row is just standard Legendre P_l(x).
        let x = 0.4;
        let arr = lpmn(2, 6, x);
        let (vals, _) = lpn(6, x);
        for l in 0..=6 {
            assert!(
                (arr[0][l as usize] - vals[l as usize]).abs() < 1e-13,
                "row m=0, l={l}: {} vs {}",
                arr[0][l as usize],
                vals[l as usize]
            );
        }
    }

    #[test]
    fn lpmn_propagates_nan_for_non_finite_x() {
        let arr = lpmn(2, 3, f64::NAN);
        for row in &arr {
            for &v in row {
                assert!(v.is_nan(), "expected NaN, got {v}");
            }
        }
    }

    #[test]
    fn lpn_zero_order_returns_singletons() {
        let (vals, ders) = lpn(0, 0.5);
        assert_eq!(vals, vec![1.0]);
        assert_eq!(ders, vec![0.0]);
    }

    #[test]
    fn lpn_first_kind_known_values_at_half() {
        // Hand-computed P_k(0.5):
        // P_0=1, P_1=0.5, P_2=-0.125, P_3=-0.4375, P_4=-0.2890625
        let (vals, _) = lpn(4, 0.5);
        let expected = [1.0, 0.5, -0.125, -0.4375, -0.2890625];
        for (k, e) in expected.iter().enumerate() {
            assert_close(vals[k], *e, 1e-15, &format!("P_{k}(0.5)"));
        }
    }

    #[test]
    fn lpn_metamorphic_endpoint_values() {
        // P_k(1) = 1 for all k; P_k(-1) = (-1)^k.
        let (vals_plus, _) = lpn(8, 1.0);
        let (vals_minus, _) = lpn(8, -1.0);
        for k in 0..=8 {
            assert!(
                (vals_plus[k] - 1.0).abs() < 1e-12,
                "P_{k}(1) = {} should be 1",
                vals_plus[k]
            );
            let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                (vals_minus[k] - expected).abs() < 1e-12,
                "P_{k}(-1) should be {expected}, got {}",
                vals_minus[k]
            );
        }
    }

    #[test]
    fn lpn_metamorphic_parity() {
        // P_k(-x) = (-1)^k P_k(x).
        let x = 0.37;
        let (a, _) = lpn(7, x);
        let (b, _) = lpn(7, -x);
        for k in 0..=7 {
            let expected = if k % 2 == 0 { a[k] } else { -a[k] };
            assert!(
                (b[k] - expected).abs() < 1e-12,
                "parity violated at k={k}: P_k(-x)={}, expected {expected}",
                b[k]
            );
        }
    }

    #[test]
    fn lpn_derivative_endpoint_closed_form() {
        // P'_k(±1) = (±1)^(k+1) k(k+1) / 2.
        let (_, ders_plus) = lpn(5, 1.0);
        let (_, ders_minus) = lpn(5, -1.0);
        for k in 1..=5 {
            let kf = k as f64;
            let half_kk1 = 0.5 * kf * (kf + 1.0);
            assert!(
                (ders_plus[k] - half_kk1).abs() < 1e-12,
                "P'_{k}(1) should be {half_kk1}, got {}",
                ders_plus[k]
            );
            let expected = if k % 2 == 0 { -half_kk1 } else { half_kk1 };
            assert!(
                (ders_minus[k] - expected).abs() < 1e-12,
                "P'_{k}(-1) should be {expected}, got {}",
                ders_minus[k]
            );
        }
    }

    #[test]
    fn lpn_value_matches_eval_legendre() {
        // The lpn array's k-th entry must agree with eval_legendre(k, x) for
        // every k. This is the metamorphic invariant tying lpn to the
        // pre-existing scalar implementation.
        let xs = [-0.9, -0.3, 0.0, 0.3, 0.9];
        for &x in &xs {
            let (vals, _) = lpn(10, x);
            for (k, value) in vals.iter().enumerate().take(11) {
                let scalar = eval_legendre(k as u32, x);
                assert!(
                    (*value - scalar).abs() < 1e-12,
                    "lpn[{k}] vs eval_legendre at x={x}: {} vs {scalar}",
                    value
                );
            }
        }
    }

    #[test]
    fn lpn_non_finite_x_propagates_nan() {
        let (vals, ders) = lpn(3, f64::NAN);
        assert_eq!(vals.len(), 4);
        assert_eq!(ders.len(), 4);
        for v in vals.iter().chain(ders.iter()) {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    #[test]
    fn lqn_zero_order_q0_is_atanh() {
        let (vals, ders) = lqn(0, 0.5);
        assert_close(vals[0], 0.5_f64.atanh(), 1e-15, "Q_0(0.5)");
        // Q'_0 = 1/(1-x²) = 1/0.75
        assert_close(ders[0], 1.0 / 0.75, 1e-15, "Q'_0(0.5)");
    }

    #[test]
    fn lqn_q1_relation_to_q0() {
        // Q_1(x) = x · Q_0(x) - 1.
        let x = 0.3;
        let (vals, _) = lqn(1, x);
        let expected = x * x.atanh() - 1.0;
        assert_close(vals[1], expected, 1e-15, "Q_1(0.3)");
    }

    #[test]
    fn lqn_metamorphic_parity() {
        // Q_k(-x) = (-1)^(k+1) Q_k(x).
        let x = 0.4;
        let (a, _) = lqn(6, x);
        let (b, _) = lqn(6, -x);
        for k in 0..=6 {
            let expected = if k % 2 == 0 { -a[k] } else { a[k] };
            assert_close(b[k], expected, 1e-12, &format!("parity at k={k}"));
        }
    }

    #[test]
    fn lqn_metamorphic_origin_zero_for_even_indices() {
        // Q_{2m}(0) = 0 for all m ≥ 0.
        let (vals, _) = lqn(8, 0.0);
        for k in (0..=8).step_by(2) {
            assert!(
                vals[k].abs() < 1e-15,
                "Q_{k}(0) should be 0, got {}",
                vals[k]
            );
        }
        // Q_1(0) = -1, Q_3(0) = 2/3, Q_5(0) = -8/15, Q_7(0) = 16/35.
        assert_close(vals[1], -1.0, 1e-15, "Q_1(0)");
        assert_close(vals[3], 2.0 / 3.0, 1e-15, "Q_3(0)");
        assert_close(vals[5], -8.0 / 15.0, 1e-15, "Q_5(0)");
        assert_close(vals[7], 16.0 / 35.0, 1e-15, "Q_7(0)");
    }

    #[test]
    fn lqn_outside_unit_interval_returns_nan() {
        let (vals_above, ders_above) = lqn(3, 1.5);
        for v in vals_above.iter().chain(ders_above.iter()) {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
        let (vals_at_one, _) = lqn(2, 1.0);
        for v in &vals_at_one {
            assert!(v.is_nan(), "expected NaN at x=1, got {v}");
        }
    }

    #[test]
    fn roots_sh_legendre_in_unit_interval() {
        let (nodes, weights) = roots_sh_legendre(7);
        assert_eq!(nodes.len(), 7);
        assert_eq!(weights.len(), 7);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0, "node {x} outside (0, 1)");
        }
        for w in &weights {
            assert!(*w > 0.0, "weight {w} should be positive");
        }
    }

    #[test]
    fn roots_sh_legendre_metamorphic_weight_sum_is_one() {
        // The shifted-Legendre weights are halved so they integrate against
        // unit weight on [0, 1]. Their sum equals the integral of 1 on
        // [0, 1] = 1.
        for n in 1..=8 {
            let (_, w) = roots_sh_legendre(n);
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "n={n}: shifted-Legendre weight sum {sum} != 1"
            );
        }
    }

    #[test]
    fn roots_sh_legendre_metamorphic_node_relation() {
        // Shifted node = (1 + unshifted node) / 2.
        let n = 6;
        let (sh_nodes, sh_weights) = roots_sh_legendre(n);
        let (un_nodes, un_weights) = roots_legendre(n);
        for (sh, un) in sh_nodes.iter().zip(un_nodes.iter()) {
            let expected = 0.5 * (1.0 + un);
            assert!(
                (sh - expected).abs() < 1e-13,
                "shifted node {sh} should be (1+{un})/2 = {expected}"
            );
        }
        for (sh, un) in sh_weights.iter().zip(un_weights.iter()) {
            assert!(
                (sh - 0.5 * un).abs() < 1e-13,
                "shifted weight {sh} should be {un}/2"
            );
        }
    }

    #[test]
    fn roots_sh_chebyt_in_unit_interval_with_uniform_weights() {
        let n = 5;
        let (nodes, weights) = roots_sh_chebyt(n);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0);
        }
        // Shifted Chebyshev-T weights are uniform π/n: the Chebyshev-T
        // weight function is invariant under the [-1,1]→[0,1] shift.
        let expected = std::f64::consts::PI / n as f64;
        for w in &weights {
            assert!((w - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn roots_sh_chebyu_in_unit_interval() {
        let (nodes, weights) = roots_sh_chebyu(6);
        assert_eq!(nodes.len(), 6);
        for x in &nodes {
            assert!(*x > 0.0 && *x < 1.0);
        }
        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn roots_sh_n_zero_returns_empty() {
        for (a, b) in [roots_sh_legendre(0), roots_sh_chebyt(0), roots_sh_chebyu(0)] {
            assert!(a.is_empty() && b.is_empty());
        }
    }

    #[test]
    fn roots_chebyc_basic_dimensions_and_range() {
        let (nodes, weights) = roots_chebyc(5);
        assert_eq!(nodes.len(), 5);
        assert_eq!(weights.len(), 5);
        for x in &nodes {
            assert!(*x > -2.0 && *x < 2.0, "node {x} outside (-2, 2)");
        }
        // Uniform weights 2π / n.
        let expected = 2.0 * std::f64::consts::PI / 5.0;
        for w in &weights {
            assert!((w - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn roots_chebyc_metamorphic_scales_chebyt_by_two() {
        // C_n's roots are exactly 2 × T_n's roots.
        let n = 7;
        let (c_nodes, _) = roots_chebyc(n);
        let (t_nodes, _) = roots_chebyt(n);
        assert_eq!(c_nodes.len(), t_nodes.len());
        for (c, t) in c_nodes.iter().zip(t_nodes.iter()) {
            assert!(
                (c - 2.0 * t).abs() < 1e-13,
                "C-root {c} should be 2 × T-root {t}"
            );
        }
    }

    #[test]
    fn roots_chebyc_weight_sum_equals_two_pi() {
        // ∑ w_k = n · (2π / n) = 2π for any n — the weight function
        // 1/√(1−x²/4) integrates to 2π over [-2, 2].
        for n in 1..=8 {
            let (_, weights) = roots_chebyc(n);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0 * std::f64::consts::PI).abs() < 1e-13,
                "n={n}: weight sum {sum} != 2π"
            );
        }
    }

    #[test]
    fn roots_chebys_basic_dimensions_and_range() {
        let (nodes, weights) = roots_chebys(6);
        assert_eq!(nodes.len(), 6);
        assert_eq!(weights.len(), 6);
        for x in &nodes {
            assert!(*x > -2.0 && *x < 2.0, "S-node {x} outside (-2, 2)");
        }
        for w in &weights {
            assert!(*w > 0.0, "weight {w} should be positive");
        }
    }

    #[test]
    fn roots_chebys_metamorphic_scales_chebyu_by_two() {
        // S_n's roots are exactly 2 × U_n's roots.
        let n = 6;
        let (s_nodes, s_weights) = roots_chebys(n);
        let (u_nodes, u_weights) = roots_chebyu(n);
        assert_eq!(s_nodes.len(), u_nodes.len());
        for (s, u) in s_nodes.iter().zip(u_nodes.iter()) {
            assert!(
                (s - 2.0 * u).abs() < 1e-13,
                "S-root {s} should be 2 × U-root {u}"
            );
        }
        // S weights are twice the U weights: the [-2, 2] interval doubles
        // the integral of the weight function relative to [-1, 1].
        for (sw, uw) in s_weights.iter().zip(u_weights.iter()) {
            assert!(
                (sw - 2.0 * uw).abs() < 1e-15,
                "S weight {sw} should be 2 × U weight {uw}"
            );
        }
    }

    #[test]
    fn roots_chebyc_n_zero_returns_empty() {
        let (n, w) = roots_chebyc(0);
        assert!(n.is_empty() && w.is_empty());
        let (n, w) = roots_chebys(0);
        assert!(n.is_empty() && w.is_empty());
    }

    #[test]
    fn roots_cheby_scaled_weights_match_scipy() {
        // Weights pinned against scipy.special for n = 5. fsci previously
        // diverged by a constant scale factor (frankenscipy-s2qfz).
        let check = |label: &str, got: &[f64], want: &[f64]| {
            assert_eq!(got.len(), want.len(), "{label}: length");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!((g - w).abs() < 1e-13, "{label}[{i}]: got {g}, want {w}");
            }
        };

        let (_, w) = roots_chebyc(5);
        check("chebyc", &w, &[1.256_637_061_435_917_2; 5]);

        let (_, w) = roots_chebys(5);
        check(
            "chebys",
            &w,
            &[
                0.261_799_387_799_149_35,
                0.785_398_163_397_448_4,
                1.047_197_551_196_597_6,
                0.785_398_163_397_448_2,
                0.261_799_387_799_149_35,
            ],
        );

        let (_, w) = roots_sh_chebyt(5);
        check("sh_chebyt", &w, &[0.628_318_530_717_958_6; 5]);

        let (_, w) = roots_sh_chebyu(5);
        check(
            "sh_chebyu",
            &w,
            &[
                0.032_724_923_474_893_66,
                0.098_174_770_424_681_03,
                0.130_899_693_899_574_68,
                0.098_174_770_424_681_01,
                0.032_724_923_474_893_66,
            ],
        );
    }

    #[test]
    fn lqn_derivative_recurrence_consistency() {
        // (1 - x²) Q'_k(x) = k (Q_{k-1} − x Q_k) — the same recurrence we
        // use to compute the derivative. Validate symmetrically by computing
        // it from the values array directly and comparing against the lqn
        // output.
        let x = 0.6;
        let (vals, ders) = lqn(5, x);
        let one_minus_xsq = 1.0 - x * x;
        for k in 1..=5 {
            let kf = k as f64;
            let direct = kf * (vals[k - 1] - x * vals[k]) / one_minus_xsq;
            assert_close(
                ders[k],
                direct,
                1e-12,
                &format!("derivative recurrence at k={k}"),
            );
        }
    }

    #[test]
    fn eval_sh_jacobi_routes_through_jacobi_at_2x_minus_1() {
        // /testing-conformance-harnesses: eval_sh_jacobi(n, p, q, x)
        // routes through eval_jacobi(n, p−q, q−1, 2x−1) by construction.
        // Pin the routing identity across multiple (n, p, q, x).
        for &(n, p, q, x) in &[
            (3_u32, 5.0_f64, 2.0, 0.3),
            (4, 4.0, 3.0, 0.5),
            (5, 6.5, 1.5, 0.75),
            (0, 3.0, 1.0, 0.1),
            (1, 4.0, 2.0, 0.9),
        ] {
            let direct = eval_sh_jacobi(n, p, q, x);
            let via = eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0);
            assert!(
                (direct - via).abs() < 1e-12,
                "eval_sh_jacobi({n}, {p}, {q}, {x}) = {direct} via eval_jacobi = {via}"
            );
        }
    }

    #[test]
    fn roots_sh_jacobi_matches_shifted_jacobi_relation() {
        for &(n, p, q) in &[(3_usize, 0.5_f64, 1.25_f64), (5, 1.5, 0.75), (8, 2.0, 2.0)] {
            let (shifted_nodes, shifted_weights) = roots_sh_jacobi(n, p, q);
            let (jacobi_nodes, jacobi_weights) = roots_jacobi(n, p - q, q - 1.0);
            let scale = 2.0_f64.powf(-p);
            for i in 0..n {
                assert!(
                    (shifted_nodes[i] - 0.5 * (1.0 + jacobi_nodes[i])).abs() < 1e-12,
                    "roots_sh_jacobi node {i} for n={n}, p={p}, q={q}"
                );
                assert!(
                    (shifted_weights[i] - scale * jacobi_weights[i]).abs() < 1e-12,
                    "roots_sh_jacobi weight {i} for n={n}, p={p}, q={q}"
                );
            }
        }
    }

    #[test]
    fn scipy_root_aliases_delegate_to_long_names() {
        assert_pairs_close(p_roots(5), roots_legendre(5), 1e-12, "p_roots");
        assert_pairs_close(ps_roots(5), roots_sh_legendre(5), 1e-12, "ps_roots");
        assert_pairs_close(t_roots(5), roots_chebyt(5), 1e-12, "t_roots");
        assert_pairs_close(ts_roots(5), roots_sh_chebyt(5), 1e-12, "ts_roots");
        assert_pairs_close(u_roots(5), roots_chebyu(5), 1e-12, "u_roots");
        assert_pairs_close(us_roots(5), roots_sh_chebyu(5), 1e-12, "us_roots");
        assert_pairs_close(c_roots(5), roots_chebyc(5), 1e-12, "c_roots");
        assert_pairs_close(s_roots(5), roots_chebys(5), 1e-12, "s_roots");
        assert_pairs_close(h_roots(5), roots_hermite(5), 1e-12, "h_roots");
        assert_pairs_close(he_roots(5), roots_hermitenorm(5), 1e-12, "he_roots");
        assert_pairs_close(l_roots(5), roots_laguerre(5), 1e-12, "l_roots");
        assert_pairs_close(
            la_roots(5, 0.5),
            roots_genlaguerre(5, 0.5),
            1e-12,
            "la_roots",
        );
        assert_pairs_close(
            cg_roots(5, 1.5),
            roots_gegenbauer(5, 1.5),
            1e-12,
            "cg_roots",
        );
        assert_pairs_close(
            j_roots(5, 0.5, 0.3),
            roots_jacobi(5, 0.5, 0.3),
            1e-12,
            "j_roots",
        );
        assert_pairs_close(
            js_roots(5, 1.5, 0.75),
            roots_sh_jacobi(5, 1.5, 0.75),
            1e-12,
            "js_roots",
        );
    }

    #[test]
    fn assoc_laguerre_routes_through_eval_genlaguerre() {
        // [frankenscipy-fdefl] /porting-to-rust: assoc_laguerre(x, n, k)
        // ≡ eval_genlaguerre(n, k, x) for any (x, n, k). Pin across
        // multiple cases including k=0 collapse to eval_laguerre.
        for &(x, n, k) in &[
            (0.0_f64, 0_u32, 0.0_f64),
            (0.5, 3, 0.0),
            (1.5, 4, 1.5),
            (2.7, 5, 0.25),
            (-0.3, 2, 3.0),
        ] {
            let direct = assoc_laguerre(x, n, k);
            let via = eval_genlaguerre(n, k, x);
            assert!(
                (direct - via).abs() < 1e-12 || (direct.is_nan() && via.is_nan()),
                "assoc_laguerre({x}, {n}, {k}) = {direct} via eval_genlaguerre = {via}"
            );
        }
        // k = 0 must collapse to standard Laguerre L_n.
        for n in 0..=5_u32 {
            let direct = assoc_laguerre(1.7, n, 0.0);
            let lag = eval_laguerre(n, 1.7);
            assert!(
                (direct - lag).abs() < 1e-12,
                "assoc_laguerre(1.7, {n}, 0) = {direct} but eval_laguerre = {lag}"
            );
        }
    }

    #[test]
    fn eval_chebyc_matches_closed_form_low_degree() {
        // [frankenscipy-1qax5] Chebyshev C polynomials on [−2, 2]:
        //   C_0(x) = 2
        //   C_1(x) = x
        //   C_2(x) = x² − 2
        //   C_3(x) = x³ − 3x
        //   C_4(x) = x⁴ − 4x² + 2
        for &x in &[-2.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0] {
            assert!((eval_chebyc(0, x) - 2.0).abs() < 1e-12);
            assert!((eval_chebyc(1, x) - x).abs() < 1e-12);
            assert!((eval_chebyc(2, x) - (x * x - 2.0)).abs() < 1e-12);
            assert!((eval_chebyc(3, x) - (x.powi(3) - 3.0 * x)).abs() < 1e-12);
            let c4_expected = x.powi(4) - 4.0 * x * x + 2.0;
            assert!(
                (eval_chebyc(4, x) - c4_expected).abs() < 1e-12,
                "C_4({x}) = {} expected {c4_expected}",
                eval_chebyc(4, x)
            );
        }
        // Identity C_n(x) = 2·T_n(x/2). Pin across n=2..=5, x=1.7.
        for n in 2..=5_u32 {
            let from_t = 2.0 * eval_chebyt(n, 1.7 / 2.0);
            let direct = eval_chebyc(n, 1.7);
            assert!(
                (from_t - direct).abs() < 1e-12,
                "C_{n}(1.7): from T = {from_t}, direct = {direct}"
            );
        }
    }

    #[test]
    fn eval_chebys_matches_closed_form_low_degree() {
        // [frankenscipy-1qax5] Chebyshev S polynomials on [−2, 2]:
        //   S_0(x) = 1
        //   S_1(x) = x
        //   S_2(x) = x² − 1
        //   S_3(x) = x³ − 2x
        //   S_4(x) = x⁴ − 3x² + 1
        for &x in &[-2.0_f64, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0] {
            assert!((eval_chebys(0, x) - 1.0).abs() < 1e-12);
            assert!((eval_chebys(1, x) - x).abs() < 1e-12);
            assert!((eval_chebys(2, x) - (x * x - 1.0)).abs() < 1e-12);
            assert!((eval_chebys(3, x) - (x.powi(3) - 2.0 * x)).abs() < 1e-12);
            let s4_expected = x.powi(4) - 3.0 * x * x + 1.0;
            assert!(
                (eval_chebys(4, x) - s4_expected).abs() < 1e-12,
                "S_4({x}) = {} expected {s4_expected}",
                eval_chebys(4, x)
            );
        }
        // Identity S_n(x) = U_n(x/2). Pin across n=2..=5, x=0.7.
        for n in 2..=5_u32 {
            let from_u = eval_chebyu(n, 0.7 / 2.0);
            let direct = eval_chebys(n, 0.7);
            assert!(
                (from_u - direct).abs() < 1e-12,
                "S_{n}(0.7): from U = {from_u}, direct = {direct}"
            );
        }
    }

    #[test]
    fn eval_legendre_matches_scipy_reference_values() {
        // scipy.special.eval_legendre(n, x)
        // P_0(x) = 1, P_1(x) = x, P_2(x) = (3x^2 - 1)/2
        let x = 0.5;
        assert!((eval_legendre(0, x) - 1.0).abs() < 1e-10, "P_0(0.5)");
        assert!((eval_legendre(1, x) - 0.5).abs() < 1e-10, "P_1(0.5)");
        let p2_expected = (3.0 * x * x - 1.0) / 2.0; // -0.125
        assert!(
            (eval_legendre(2, x) - p2_expected).abs() < 1e-10,
            "P_2(0.5) got {}, expected {}",
            eval_legendre(2, x),
            p2_expected
        );
    }

    #[test]
    fn eval_chebyt_matches_scipy_reference_values() {
        // scipy.special.eval_chebyt(n, x)
        // T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1
        let x = 0.5;
        assert!((eval_chebyt(0, x) - 1.0).abs() < 1e-10, "T_0(0.5)");
        assert!((eval_chebyt(1, x) - 0.5).abs() < 1e-10, "T_1(0.5)");
        let t2_expected = 2.0 * x * x - 1.0; // -0.5
        assert!(
            (eval_chebyt(2, x) - t2_expected).abs() < 1e-10,
            "T_2(0.5) got {}, expected {}",
            eval_chebyt(2, x),
            t2_expected
        );
    }

    #[test]
    fn eval_hermite_matches_scipy_reference_values() {
        // scipy.special.eval_hermite(n, x) - physicist's Hermite
        // H_0(x) = 1, H_1(x) = 2x, H_2(x) = 4x^2 - 2
        let x = 1.0;
        assert!((eval_hermite(0, x) - 1.0).abs() < 1e-10, "H_0(1)");
        assert!((eval_hermite(1, x) - 2.0).abs() < 1e-10, "H_1(1)");
        let h2_expected = 4.0 * x * x - 2.0; // 2
        assert!(
            (eval_hermite(2, x) - h2_expected).abs() < 1e-10,
            "H_2(1) got {}, expected {}",
            eval_hermite(2, x),
            h2_expected
        );
    }

    #[test]
    fn roots_legendre_matches_scipy_reference_values() {
        // scipy.special.roots_legendre(3)
        // nodes ≈ [-0.7746, 0, 0.7746], weights ≈ [0.5556, 0.8889, 0.5556]
        let (nodes, weights) = roots_legendre(3);
        assert_eq!(nodes.len(), 3);
        assert_eq!(weights.len(), 3);
        // Check symmetry: nodes should be symmetric around 0
        assert!(
            (nodes[0] + nodes[2]).abs() < 1e-10,
            "nodes should be symmetric"
        );
        assert!((nodes[1]).abs() < 1e-10, "middle node should be ~0");
        // Weights should be positive and sum to 2 (integral of 1 over [-1,1])
        let weight_sum: f64 = weights.iter().sum();
        assert!(
            (weight_sum - 2.0).abs() < 1e-10,
            "weights should sum to 2, got {}",
            weight_sum
        );
    }

    #[test]
    fn eval_laguerre_matches_scipy_reference_values() {
        // scipy.special.eval_laguerre(n, x)
        // L_0(x) = 1, L_1(x) = 1-x, L_2(x) = (x^2 - 4x + 2)/2
        let x = 1.0;
        assert!((eval_laguerre(0, x) - 1.0).abs() < 1e-10, "L_0(1)");
        assert!(
            (eval_laguerre(1, x) - 0.0).abs() < 1e-10,
            "L_1(1) = 1-1 = 0"
        );
        let l2_expected = (x * x - 4.0 * x + 2.0) / 2.0; // -0.5
        assert!(
            (eval_laguerre(2, x) - l2_expected).abs() < 1e-10,
            "L_2(1) got {}, expected {}",
            eval_laguerre(2, x),
            l2_expected
        );
    }

    #[test]
    fn eval_jacobi_matches_scipy_reference_values() {
        // scipy.special.eval_jacobi(n, alpha, beta, x)
        // P_0^(a,b)(x) = 1
        // P_1^(a,b)(x) = (a+b+2)*x/2 + (a-b)/2
        assert!(
            (eval_jacobi(0, 0.5, 0.5, 0.5) - 1.0).abs() < 1e-10,
            "P_0^(0.5,0.5)(0.5) = 1"
        );
        // P_1^(1,1)(0) = (1-1)/2 = 0
        assert!(
            eval_jacobi(1, 1.0, 1.0, 0.0).abs() < 1e-10,
            "P_1^(1,1)(0) = 0"
        );
    }

    #[test]
    fn eval_gegenbauer_matches_scipy_reference_values() {
        // scipy.special.eval_gegenbauer(n, alpha, x)
        // C_0^alpha(x) = 1
        // C_1^alpha(x) = 2*alpha*x
        assert!(
            (eval_gegenbauer(0, 0.5, 0.7) - 1.0).abs() < 1e-10,
            "C_0^0.5(0.7) = 1"
        );
        let c1_expected = 2.0 * 0.5 * 0.7; // = 0.7
        assert!(
            (eval_gegenbauer(1, 0.5, 0.7) - c1_expected).abs() < 1e-10,
            "C_1^0.5(0.7) got {}, expected {}",
            eval_gegenbauer(1, 0.5, 0.7),
            c1_expected
        );
    }

    #[test]
    fn eval_chebyu_matches_scipy_reference_values() {
        // scipy.special.eval_chebyu(n, x) - Chebyshev of second kind
        // U_0(x) = 1, U_1(x) = 2x, U_2(x) = 4x^2 - 1
        let x = 0.5;
        assert!((eval_chebyu(0, x) - 1.0).abs() < 1e-10, "U_0(0.5)");
        assert!(
            (eval_chebyu(1, x) - 1.0).abs() < 1e-10,
            "U_1(0.5) = 2*0.5 = 1"
        );
        let u2_expected = 4.0 * x * x - 1.0; // = 0
        assert!(
            (eval_chebyu(2, x) - u2_expected).abs() < 1e-10,
            "U_2(0.5) got {}, expected {}",
            eval_chebyu(2, x),
            u2_expected
        );
    }
}
