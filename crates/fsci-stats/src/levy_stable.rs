//! The standardized Lévy-stable law as `scipy.stats.levy_stable` computes it by default.
//!
//! A port of SciPy 1.17.1's `scipy/stats/_levy_stable/__init__.py` and of the `levyst.Nolan`
//! extension it calls (`c_src/levyst.c`, Nolan's `g`). SciPy is BSD-3-Clause: "Copyright (c)
//! 2001-2002 Enthought, Inc. 2003, SciPy Developers." frankenscipy-1ksfv.16
//!
//! Ported, in SciPy's evaluation order so the arithmetic is the incumbent's:
//! - the default `piecewise` pdf and cdf (Nolan 1997, "Numerical calculation of stable densities
//!   and distribution functions") on the S0 (Zolotarev M) law, with the S1 -> S0 shift in front,
//!   SciPy's closed forms (alpha = 2, the Lévy case, the alpha = 1/2 Fresnel form, Cauchy) and
//!   its rounding of x near zeta and of alpha near 1;
//! - `_rvs_Z1`, Chambers–Mallows–Stuck;
//! - `_fitstart_S1` / `_fitstart_S0`, McCulloch's (1986) quantile estimator.
//!
//! Not ported: the `dni` and `fft-simpson` methods, which SciPy runs only when
//! `pdf_default_method` / `cdf_default_method` are changed, and the MLE `fit`.

use std::f64::consts::{FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_PI_2, PI, SQRT_2};

use fsci_integrate::{QuadOptions, quad_full_output};
use fsci_runtime::RuntimeMode;
use fsci_special::SpecialTensor;

/// `_QUAD_EPS`, `levy_stable.quad_eps`: `epsrel` of every quadrature (with `epsabs = 0`) and the
/// `xtol` of the bisection for the integrand's peak.
const QUAD_EPS: f64 = 1.2e-14;
/// `levy_stable.piecewise_x_tol_near_zeta`.
const X_TOL_NEAR_ZETA: f64 = 0.005;
/// `levy_stable.piecewise_alpha_tol_near_one`.
const ALPHA_TOL_NEAR_ONE: f64 = 0.005;
/// `scipy.integrate.quad(..., limit=100)` in both piecewise integrals.
const QUAD_LIMIT: usize = 100;

/// Nolan's `g(theta)` in the two forms `levyst.c` evaluates, with the constants its
/// `nolan_precan` hoists out of the integrand.
enum NolanG {
    /// `g_alpha_ne_one`.
    AlphaNeOne {
        zeta_prefactor: f64,
        alpha_exp: f64,
        alpha_xi: f64,
        zeta_offset: f64,
    },
    /// `g_alpha_eq_one`.
    AlphaEqOne {
        two_beta_div_pi: f64,
        pi_div_two_beta: f64,
        x0_div_term: f64,
    },
}

/// `struct nolan_precanned`, as `nolan_precan(alpha, beta, x0)` fills it. The wheel ships
/// `levyst.c` only compiled; this form reproduced SciPy 1.17.1's `levyst.Nolan` bit for bit on
/// 1,800 `zeta`/`xi`/`c1`/`c2`/`c3` values and 15,120 `g(theta)` evaluations (alpha from 0.3
/// to 2 including 1, beta from -1 to 1 including 0, x0 on both sides of zeta).
struct Nolan {
    alpha: f64,
    zeta: f64,
    xi: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    g: NolanG,
}

impl Nolan {
    fn new(alpha: f64, beta: f64, x0: f64) -> Self {
        let zeta = -beta * (PI * alpha / 2.0).tan();
        if alpha != 1.0 {
            let xi = (-zeta).atan() / alpha;
            let (c1, c3) = if alpha < 1.0 {
                (0.5 - xi * FRAC_1_PI, FRAC_1_PI)
            } else {
                (1.0, -FRAC_1_PI)
            };
            Self {
                alpha,
                zeta,
                xi,
                c1,
                c2: alpha * FRAC_1_PI / (alpha - 1.0).abs() / (x0 - zeta),
                c3,
                g: NolanG::AlphaNeOne {
                    zeta_prefactor: (zeta.powf(2.0) + 1.0).powf(-1.0 / (2.0 * (alpha - 1.0))),
                    alpha_exp: alpha / (alpha - 1.0),
                    alpha_xi: (-zeta).atan(),
                    zeta_offset: x0 - zeta,
                },
            }
        } else {
            let two_beta_div_pi = beta * FRAC_2_PI;
            Self {
                alpha,
                zeta,
                xi: FRAC_PI_2,
                c1: 0.0,
                c2: 0.5 / beta.abs(),
                c3: FRAC_1_PI,
                g: NolanG::AlphaEqOne {
                    two_beta_div_pi,
                    pi_div_two_beta: FRAC_PI_2 / beta,
                    x0_div_term: x0 / two_beta_div_pi,
                },
            }
        }
    }

    /// `Nolan.g(theta)`. The ends of `[-xi, pi/2]` are answered by the guards, not by the
    /// formula, exactly as in `levyst.c`.
    fn g(&self, theta: f64) -> f64 {
        match self.g {
            NolanG::AlphaNeOne {
                zeta_prefactor,
                alpha_exp,
                alpha_xi,
                zeta_offset,
            } => {
                if theta == -self.xi {
                    return if self.alpha < 1.0 { 0.0 } else { f64::INFINITY };
                }
                if theta == FRAC_PI_2 {
                    return if self.alpha < 1.0 { f64::INFINITY } else { 0.0 };
                }
                let cos_theta = theta.cos();
                zeta_prefactor
                    * (cos_theta / (alpha_xi + self.alpha * theta).sin() * zeta_offset)
                        .powf(alpha_exp)
                    * (alpha_xi + (self.alpha - 1.0) * theta).cos()
                    / cos_theta
            }
            NolanG::AlphaEqOne {
                two_beta_div_pi,
                pi_div_two_beta,
                x0_div_term,
            } => {
                if theta == -self.xi {
                    return 0.0;
                }
                if theta == FRAC_PI_2 {
                    return f64::INFINITY;
                }
                (1.0 + theta * two_beta_div_pi)
                    * ((pi_div_two_beta + theta) * theta.tan() - x0_div_term).exp()
                    / theta.cos()
            }
        }
    }
}

/// `scipy.optimize.bisect` (`scipy/optimize/Zeros/bisect.c`) with its defaults `rtol = 4·eps`
/// and `maxiter = 100`. `None` where SciPy raises: a NaN function value (`_wrap_nan_raise`), no
/// sign change, or no convergence.
fn bisect(f: impl Fn(f64) -> f64, xa: f64, xb: f64, xtol: f64) -> Option<f64> {
    const RTOL: f64 = 4.0 * f64::EPSILON;
    const MAXITER: usize = 100;
    let mut xa = xa;
    let fa = f(xa);
    let fb = f(xb);
    if fa.is_nan() || fb.is_nan() || fa * fb > 0.0 {
        return None;
    }
    if fa == 0.0 {
        return Some(xa);
    }
    if fb == 0.0 {
        return Some(xb);
    }
    let mut dm = xb - xa;
    for _ in 0..MAXITER {
        dm *= 0.5;
        let xm = xa + dm;
        let fm = f(xm);
        if fm.is_nan() {
            return None;
        }
        // `fa` is never updated, as in the C: its sign is all the loop needs.
        if fm * fa >= 0.0 {
            xa = xm;
        }
        if fm == 0.0 || dm.abs() < xtol + RTOL * xm.abs() {
            return Some(xm);
        }
    }
    None
}

/// `scipy.special.gamma` on a real argument.
fn gamma(x: f64) -> f64 {
    match fsci_special::gamma(&SpecialTensor::RealScalar(x), RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => v,
        _ => f64::NAN,
    }
}

/// `scipy.special.ndtr` (cephes `ndtr`), which `_norm_cdf` calls.
fn ndtr(a: f64) -> f64 {
    let x = a * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < FRAC_1_SQRT_2 {
        0.5 + 0.5 * fsci_special::erf_scalar(x)
    } else {
        let y = 0.5 * fsci_special::erfc_scalar(z);
        if x > 0.0 { 1.0 - y } else { y }
    }
}

/// `np.isclose(v, np.pi / 2, rtol=1e-14, atol=1e-14)`: the integration range `[-xi, pi/2]` is
/// empty.
fn is_close_to_half_pi(v: f64) -> bool {
    (v - FRAC_PI_2).abs() <= 1e-14 + 1e-14 * FRAC_PI_2.abs()
}

/// `_nolan_round_x_near_zeta`: Nolan's STABLE sets x0 to zeta when it is within
/// `tol·alpha^(1/alpha)` of it.
fn round_x_near_zeta(x0: f64, alpha: f64, zeta: f64) -> f64 {
    if (x0 - zeta).abs() < X_TOL_NEAR_ZETA * alpha.powf(1.0 / alpha) {
        zeta
    } else {
        x0
    }
}

/// `_nolan_round_difficult_input`: alpha within 0.005 of 1 becomes 1 (beta is left alone), then
/// x0 is rounded to `zeta`, which the caller computed from the UNROUNDED alpha.
fn round_difficult_input(x0: f64, alpha: f64, zeta: f64) -> (f64, f64) {
    let alpha = if (alpha - 1.0).abs() < ALPHA_TOL_NEAR_ONE {
        1.0
    } else {
        alpha
    };
    (round_x_near_zeta(x0, alpha, zeta), alpha)
}

/// `_pdf_single_value_piecewise_Z1`: the S1 density at `x` is the S0 density at
/// `x - beta·tan(pi·alpha/2)`, except at alpha = 1 where the two coincide.
pub(crate) fn pdf_z1(x: f64, alpha: f64, beta: f64) -> f64 {
    let zeta = -beta * (PI * alpha / 2.0).tan();
    pdf_z0(if alpha == 1.0 { x } else { x + zeta }, alpha, beta)
}

/// `_pdf_single_value_piecewise_Z0`: the closed forms SciPy knows, else Nolan's integral.
pub(crate) fn pdf_z0(x0: f64, alpha: f64, beta: f64) -> f64 {
    let zeta = -beta * (PI * alpha / 2.0).tan();
    let (x0, alpha) = round_difficult_input(x0, alpha, zeta);
    if alpha == 2.0 {
        // Normal with scale sqrt(2): `_norm_pdf(x0 / sqrt(2)) / sqrt(2)`.
        let x = x0 / SQRT_2;
        (-(x * x) / 2.0).exp() / (2.0 * PI).sqrt() / SQRT_2
    } else if alpha == 0.5 && beta == 1.0 {
        // Lévy: S(1/2, 1, gamma, delta; S0) is S(1/2, 1, gamma, gamma + delta; S1).
        let x = x0 + 1.0;
        if x <= 0.0 {
            return 0.0;
        }
        1.0 / (2.0 * PI * x).sqrt() / x * (-1.0 / (2.0 * x)).exp()
    } else if alpha == 0.5 && beta == 0.0 && x0 != 0.0 {
        // Hopcraft, Jakeman & Tanner (1999), through the Fresnel integrals.
        let ax = x0.abs();
        let (s, c) = fsci_special::fresnel(1.0 / (2.0 * PI * ax).sqrt());
        let arg = 1.0 / (4.0 * ax);
        (arg.sin() * (0.5 - s) + arg.cos() * (0.5 - c)) / (2.0 * PI * ax.powf(3.0)).sqrt()
    } else if alpha == 1.0 && beta == 0.0 {
        // Cauchy.
        1.0 / (1.0 + x0 * x0) / PI
    } else {
        pdf_post_rounding_z0(x0, alpha, beta)
    }
}

/// `_pdf_single_value_piecewise_post_rounding_Z0`: Nolan's
/// `c2 · ∫_{-xi}^{pi/2} g e^{-g} dtheta`.
fn pdf_post_rounding_z0(x0: f64, alpha: f64, beta: f64) -> f64 {
    let nolan = Nolan::new(alpha, beta, x0);
    // Round again: zeta was recomputed in C and may differ in the last bit (scipy#18133).
    let x0 = round_x_near_zeta(x0, alpha, nolan.zeta);
    if x0 == nolan.zeta {
        return gamma(1.0 + 1.0 / alpha) * nolan.xi.cos()
            / PI
            / (1.0 + nolan.zeta * nolan.zeta).powf(1.0 / alpha / 2.0);
    } else if x0 < nolan.zeta {
        return pdf_post_rounding_z0(-x0, alpha, -beta);
    }
    // From here x0 > zeta when alpha != 1, and beta != 0 when alpha == 1.
    if is_close_to_half_pi(-nolan.xi) {
        return 0.0;
    }
    let lo = -nolan.xi;
    // The integrand can be very peaked: SciPy forces QUADPACK to split at theta = 0, at the peak
    // g = 1 and where g reaches 100, 10 and 5, to see the tail's descent.
    let Some(peak) = bisect(|t| nolan.g(t) - 1.0, lo, FRAC_PI_2, QUAD_EPS) else {
        return f64::NAN;
    };
    let mut points = vec![0.0, peak];
    for height in [100.0, 10.0, 5.0] {
        // `optimize.bisect`'s default xtol.
        let Some(t) = bisect(|t| nolan.g(t) - height, lo, FRAC_PI_2, 2e-12) else {
            return f64::NAN;
        };
        points.push(t);
    }
    let integrand = |theta: f64| {
        // Numerical trouble can make g negative or non-finite at the ends of the range.
        let g = nolan.g(theta);
        let g = if !g.is_finite() || g < 0.0 { 0.0 } else { g };
        g * (-g).exp()
    };
    let options = QuadOptions {
        epsabs: 0.0,
        epsrel: QUAD_EPS,
        limit: QUAD_LIMIT,
    };
    match quad_full_output(integrand, lo, FRAC_PI_2, &points, options) {
        Ok((result, _)) => nolan.c2 * result.integral,
        Err(_) => f64::NAN,
    }
}

/// `_cdf_single_value_piecewise_Z1`.
pub(crate) fn cdf_z1(x: f64, alpha: f64, beta: f64) -> f64 {
    let zeta = -beta * (PI * alpha / 2.0).tan();
    cdf_z0(if alpha == 1.0 { x } else { x + zeta }, alpha, beta)
}

/// `_cdf_single_value_piecewise_Z0`.
pub(crate) fn cdf_z0(x0: f64, alpha: f64, beta: f64) -> f64 {
    let zeta = -beta * (PI * alpha / 2.0).tan();
    let (x0, alpha) = round_difficult_input(x0, alpha, zeta);
    if alpha == 2.0 {
        ndtr(x0 / SQRT_2)
    } else if alpha == 0.5 && beta == 1.0 {
        let x = x0 + 1.0;
        if x <= 0.0 {
            return 0.0;
        }
        fsci_special::erfc_scalar((0.5 / x).sqrt())
    } else if alpha == 1.0 && beta == 0.0 {
        0.5 + x0.atan() / PI
    } else {
        cdf_post_rounding_z0(x0, alpha, beta)
    }
}

/// `_cdf_single_value_piecewise_post_rounding_Z0`: Nolan's
/// `c1 + c3 · ∫_{-xi}^{pi/2} e^{-g} dtheta`.
fn cdf_post_rounding_z0(x0: f64, alpha: f64, beta: f64) -> f64 {
    let nolan = Nolan::new(alpha, beta, x0);
    let x0 = round_x_near_zeta(x0, alpha, nolan.zeta);
    if (alpha == 1.0 && beta < 0.0) || x0 < nolan.zeta {
        // Nolan's paper has F(x) = 1 - F(x, alpha, -beta) here, a typo; SciPy reflects x too.
        return 1.0 - cdf_post_rounding_z0(-x0, alpha, -beta);
    } else if x0 == nolan.zeta {
        return 0.5 - nolan.xi / PI;
    }
    // From here x0 > zeta when alpha != 1, and beta > 0 when alpha == 1.
    if is_close_to_half_pi(-nolan.xi) {
        return nolan.c1;
    }
    let integrand = |theta: f64| (-nolan.g(theta)).exp();
    let (left, right) = (-nolan.xi, FRAC_PI_2);
    // SciPy shrinks [left, right] with an L-BFGS-B minimisation when the integrand is nonzero at
    // the end where it should vanish (left for alpha > 1, right otherwise). It never is: SciPy
    // evaluates it at exactly -xi and pi/2, where `g`'s guards return +inf and e^-inf = 0, so
    // that branch is unreachable and not ported.
    //
    // SciPy passes `points=[left, right]`, which leaves no interior breakpoint, so its QUADPACK
    // runs QAGPE on the whole range; `quad_full_output` runs QAGSE for an empty interior. The two
    // returned identical bits on all 1,500 cdfs of a random (alpha, beta, x) sweep against
    // SciPy 1.17.1.
    let options = QuadOptions {
        epsabs: 0.0,
        epsrel: QUAD_EPS,
        limit: QUAD_LIMIT,
    };
    match quad_full_output(integrand, left, right, &[left, right], options) {
        Ok((result, _)) => nolan.c1 + nolan.c3 * result.integral,
        Err(_) => f64::NAN,
    }
}

/// `_rvs_Z1` for one draw (S1, loc 0, scale 1), Chambers–Mallows–Stuck as Nolan writes it:
/// `th` uniform on `[-pi/2, pi/2)`, `w` standard exponential.
pub(crate) fn rvs_z1(alpha: f64, beta: f64, th: f64, w: f64) -> f64 {
    let a_th = alpha * th;
    let b_th = beta * th;
    let cos_th = th.cos();
    let tan_th = th.tan();
    if alpha == 1.0 {
        2.0 / PI
            * ((PI / 2.0 + b_th) * tan_th
                - beta * ((PI / 2.0 * w * cos_th) / (PI / 2.0 + b_th)).ln())
    } else if beta == 0.0 {
        w / (cos_th / a_th.tan() + th.sin())
            * ((a_th.cos() + a_th.sin() * tan_th) / w).powf(1.0 / alpha)
    } else {
        let val0 = beta * (PI * alpha / 2.0).tan();
        let th0 = val0.atan() / alpha;
        let val3 = w / (cos_th / (alpha * (th0 + th)).tan() + th.sin());
        val3 * ((a_th.cos() + a_th.sin() * tan_th - val0 * (a_th.sin() - a_th.cos() * tan_th)) / w)
            .powf(1.0 / alpha)
    }
}

// McCulloch (1986), "Simple consistent estimators of stable distribution parameters", as SciPy
// tabulates it. The tables are SciPy's literals in SciPy's row order.

/// Tables III and IV are indexed by `nu_alpha` (rows) ...
const NU_ALPHA_RANGE: [f64; 15] = [
    2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0,
];
/// ... and `nu_beta` (columns).
const NU_BETA_RANGE: [f64; 7] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];

/// Table III, `alpha = psi_1(nu_alpha, nu_beta)`.
const ALPHA_TABLE: [[f64; 7]; 15] = [
    [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
    [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
    [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
    [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
    [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
    [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
    [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
    [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
    [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
    [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
    [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
    [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
    [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
    [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
    [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513],
];

/// Table IV, `beta = psi_2(nu_alpha, nu_beta)`.
const BETA_TABLE: [[f64; 7]; 15] = [
    [0.0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
    [0.0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
    [0.0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
    [0.0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
    [0.0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
    [0.0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
    [0.0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
    [0.0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
    [0.0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
    [0.0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
    [0.0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
    [0.0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
    [0.0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
    [0.0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
    [0.0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274],
];

/// Tables V and VII are indexed by `alpha` (rows) and `beta` (columns). SciPy lists alpha from
/// 2 down to 0.5 and reverses the list; this is the reversed, ascending grid.
const ALPHA_RANGE: [f64; 16] = [
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
];
const BETA_RANGE: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Table V, `nu_c = psi_3(alpha, beta)`, rows for alpha = 2, 1.9, ..., 0.5 as SciPy lists them.
const NU_C_TABLE: [[f64; 5]; 16] = [
    [1.908, 1.908, 1.908, 1.908, 1.908],
    [1.914, 1.915, 1.916, 1.918, 1.921],
    [1.921, 1.922, 1.927, 1.936, 1.947],
    [1.927, 1.930, 1.943, 1.961, 1.987],
    [1.933, 1.940, 1.962, 1.997, 2.043],
    [1.939, 1.952, 1.988, 2.045, 2.116],
    [1.946, 1.967, 2.022, 2.106, 2.211],
    [1.955, 1.984, 2.067, 2.188, 2.333],
    [1.965, 2.007, 2.125, 2.294, 2.491],
    [1.980, 2.040, 2.205, 2.435, 2.696],
    [2.000, 2.085, 2.311, 2.624, 2.973],
    [2.040, 2.149, 2.461, 2.886, 3.356],
    [2.098, 2.244, 2.676, 3.265, 3.912],
    [2.189, 2.392, 3.004, 3.844, 4.775],
    [2.337, 2.634, 3.542, 4.808, 6.247],
    [2.588, 3.073, 4.534, 6.636, 9.144],
];

/// Table VII, `nu_zeta = psi_5(alpha, beta)`, rows for alpha = 2, 1.9, ..., 0.5.
const NU_ZETA_TABLE: [[f64; 5]; 16] = [
    [0.0, 0.000, 0.000, 0.000, 0.000],
    [0.0, -0.017, -0.032, -0.049, -0.064],
    [0.0, -0.030, -0.061, -0.092, -0.123],
    [0.0, -0.043, -0.088, -0.132, -0.179],
    [0.0, -0.056, -0.111, -0.170, -0.232],
    [0.0, -0.066, -0.134, -0.206, -0.283],
    [0.0, -0.075, -0.154, -0.241, -0.335],
    [0.0, -0.084, -0.173, -0.276, -0.390],
    [0.0, -0.090, -0.192, -0.310, -0.447],
    [0.0, -0.095, -0.208, -0.346, -0.508],
    [0.0, -0.098, -0.223, -0.380, -0.576],
    [0.0, -0.099, -0.237, -0.424, -0.652],
    [0.0, -0.096, -0.250, -0.469, -0.742],
    [0.0, -0.089, -0.262, -0.520, -0.853],
    [0.0, -0.078, -0.272, -0.581, -0.997],
    [0.0, -0.061, -0.279, -0.659, -1.198],
];

/// FITPACK's degree-1 basis on the knots `[t0, t0, t1, ..., t_{m-1}, t_{m-1}]` that
/// `RectBivariateSpline(..., kx=1, ky=1, s=0)` places on a grid: `fpbisp` clamps the argument to
/// `[t0, t_{m-1}]` and finds `t_i <= arg < t_{i+1}` (the last interval closed), and `fpbspl`
/// weighs the two ends. Returns `i` and the weights of `t_i` and `t_{i+1}`.
fn linear_basis(t: &[f64], x: f64) -> (usize, [f64; 2]) {
    let last = t.len() - 2;
    let arg = x.clamp(t[0], t[last + 1]);
    let mut i = 0;
    while !(arg < t[i + 1] || i == last) {
        i += 1;
    }
    let f = 1.0 / (t[i + 1] - t[i]);
    (i, [f * (t[i + 1] - arg), f * (arg - t[i])])
}

/// `RectBivariateSpline(xs, ys, z, kx=1, ky=1, s=0)(x, y)[0, 0]`, summed in `fpbisp`'s order.
/// A linear interpolating spline's coefficients are the grid values themselves (SciPy's
/// `get_coeffs()` equals each table exactly), so `z(i, j)` is the table entry at `(xs[i], ys[j])`.
fn fitpack_bilinear(
    xs: &[f64],
    ys: &[f64],
    z: impl Fn(usize, usize) -> f64,
    x: f64,
    y: f64,
) -> f64 {
    let (i, hx) = linear_basis(xs, x);
    let (j, hy) = linear_basis(ys, y);
    let mut sp = 0.0;
    sp += z(i, j) * hx[0] * hy[0];
    sp += z(i, j + 1) * hx[0] * hy[1];
    sp += z(i + 1, j) * hx[1] * hy[0];
    sp += z(i + 1, j + 1) * hx[1] * hy[1];
    sp
}

fn psi_1(nu_beta: f64, nu_alpha: f64) -> f64 {
    let z = |i: usize, j: usize| ALPHA_TABLE[j][i];
    fitpack_bilinear(&NU_BETA_RANGE, &NU_ALPHA_RANGE, z, nu_beta, nu_alpha)
}

fn psi_2(nu_beta: f64, nu_alpha: f64) -> f64 {
    let z = |i: usize, j: usize| BETA_TABLE[j][i];
    fitpack_bilinear(&NU_BETA_RANGE, &NU_ALPHA_RANGE, z, nu_beta, nu_alpha)
}

fn phi_3(beta: f64, alpha: f64) -> f64 {
    let z = |i: usize, j: usize| NU_C_TABLE[ALPHA_RANGE.len() - 1 - j][i];
    fitpack_bilinear(&BETA_RANGE, &ALPHA_RANGE, z, beta, alpha)
}

fn phi_5(beta: f64, alpha: f64) -> f64 {
    let z = |i: usize, j: usize| NU_ZETA_TABLE[ALPHA_RANGE.len() - 1 - j][i];
    fitpack_bilinear(&BETA_RANGE, &ALPHA_RANGE, z, beta, alpha)
}

/// `numpy.percentile(sorted, q)` with numpy's default `method="linear"`: the virtual index
/// `(n - 1)·(q / 100)` and numpy's `_lerp`, which interpolates down from the upper neighbour once
/// the weight reaches 1/2. `sorted` is ascending, non-empty and NaN-free.
fn numpy_percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let (prev, next, t) = if virtual_index >= (n - 1) as f64 {
        // numpy points both neighbours at the last element (index -1) and weighs by
        // `virtual_index - (-1)`.
        (n - 1, n - 1, virtual_index + 1.0)
    } else {
        let floor = virtual_index.floor();
        // `floor` is a non-negative integer below n - 1.
        let prev = floor as usize;
        (prev, prev + 1, virtual_index - floor)
    };
    let (a, b) = (sorted[prev], sorted[next]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `np.sign`: 0 at ±0 and NaN at NaN, unlike `f64::signum`.
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

/// `_fitstart_S1(data)`: McCulloch's quantile estimates `(alpha, beta, delta, gamma)` of the S1
/// law, i.e. `(alpha, beta, loc, scale)`. All NaN for empty data (where `np.percentile` raises).
pub(crate) fn fitstart_s1(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    // np.percentile returns NaN when the data holds one.
    let [p05, p50, p95, p25, p75] = if data.iter().any(|v| v.is_nan()) {
        [f64::NAN; 5]
    } else {
        let mut sorted = data.to_vec();
        sorted.sort_by(f64::total_cmp);
        [5.0, 50.0, 95.0, 25.0, 75.0].map(|q| numpy_percentile(&sorted, q))
    };

    let nu_alpha = (p95 - p05) / (p75 - p25);
    let nu_beta = (p95 + p05 - 2.0 * p50) / (p95 - p05);

    let (alpha, beta) = if nu_alpha >= 2.439 {
        let psi_1_1 = if nu_beta > 0.0 {
            psi_1(nu_beta, nu_alpha)
        } else {
            psi_1(-nu_beta, nu_alpha)
        };
        let psi_2_1 = if nu_beta > 0.0 {
            psi_2(nu_beta, nu_alpha)
        } else {
            -psi_2(-nu_beta, nu_alpha)
        };
        (psi_1_1.clamp(f64::EPSILON, 2.0), psi_2_1.clamp(-1.0, 1.0))
    } else {
        (2.0, np_sign(nu_beta))
    };
    let phi_3_1 = if beta > 0.0 {
        phi_3(beta, alpha)
    } else {
        phi_3(-beta, alpha)
    };
    let phi_5_1 = if beta > 0.0 {
        phi_5(beta, alpha)
    } else {
        -phi_5(-beta, alpha)
    };
    let c = (p75 - p25) / phi_3_1;
    let zeta = p50 + c * phi_5_1;
    let delta = if alpha == 1.0 {
        zeta
    } else {
        zeta - beta * c * (PI * alpha / 2.0).tan()
    };
    (alpha, beta, delta, c)
}

/// `_fitstart_S0(data)`: [`fitstart_s1`] with the location moved to S0; only delta changes.
pub(crate) fn fitstart_s0(data: &[f64]) -> (f64, f64, f64, f64) {
    let (alpha, beta, delta1, gamma) = fitstart_s1(data);
    let delta0 = if alpha == 1.0 {
        delta1 + 2.0 * beta * gamma * gamma.ln() / PI
    } else {
        delta1 + beta * gamma * (PI * alpha / 2.0).tan()
    };
    (alpha, beta, delta0, gamma)
}
