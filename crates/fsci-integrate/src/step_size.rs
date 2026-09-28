#![forbid(unsafe_code)]

use fsci_runtime::RuntimeMode;

use crate::validation::validate_rhs_shape;
use crate::{IntegrateValidationError, ToleranceValue};

/// Type alias for the right-hand side function used in step size selection.
pub type StepRhsFn = dyn FnMut(f64, &[f64]) -> Vec<f64>;

/// `x^(1/k)` for integer `k >= 1`, computed with IEEE-754 `+ - * /` only.
///
/// Step-size control needs `err^(-1/(order+1))` and `(0.01/d)^(1/(order+1))`.
/// `f64::powf` goes to the platform libm, whose last bits differ between
/// glibc, macOS and wasm, so the same problem took different step sequences on
/// different platforms. Consumers that promise bit-identical output (FrankenManim's
/// certified StreamLines) cannot allow that. This root is a pure function of its
/// inputs on every IEEE-754 platform. It splits `x = m * 2^e` exactly, takes
/// `y = m * 2^(e mod k)` in `[1, 2^k)`, and runs Newton's iteration for `y^(1/k)`
/// from above (the start `1 + (y-1)/k` bounds the root by Bernoulli's
/// inequality, so iterates decrease monotonically), stopping when an iterate no
/// longer decreases. The result is within a few ulps of the exact root; it is
/// not claimed to be correctly rounded, only deterministic.
///
/// NaN and negative inputs return NaN; `+0`, `+inf` and `k == 1` return `x`.
pub(crate) fn kth_root(x: f64, k: u32) -> f64 {
    if x.is_nan() || x < 0.0 || k == 0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() || k == 1 {
        return x;
    }
    const MANTISSA: u64 = 0x000f_ffff_ffff_ffff;
    // Normalize subnormals exactly: multiply by 2^64.
    let (v, bias) = if (x.to_bits() >> 52) & 0x7ff == 0 {
        (x * pow2(64), 64_i64)
    } else {
        (x, 0)
    };
    let e = ((v.to_bits() >> 52) & 0x7ff) as i64 - 1023 - bias;
    let m = f64::from_bits((v.to_bits() & MANTISSA) | (1023_u64 << 52));
    let k_i = i64::from(k);
    let (q, r) = (e.div_euclid(k_i), e.rem_euclid(k_i));
    let y = m * pow2(r); // exact: y in [1, 2^k)
    let k_f = f64::from(k);
    let mut z = 1.0 + (y - 1.0) / k_f;
    for _ in 0..200 {
        let mut z_km1 = 1.0;
        for _ in 1..k {
            z_km1 *= z;
        }
        let next = ((k_f - 1.0) * z + y / z_km1) / k_f;
        if !(next < z) {
            break;
        }
        z = next;
    }
    scale_pow2(z, q)
}

/// `2^n` for `-1022 <= n <= 1023`, built exactly from its bits.
fn pow2(n: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&n));
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// `z * 2^n` by exact power-of-two multiplications (no `powi`/`ldexp`).
fn scale_pow2(mut z: f64, mut n: i64) -> f64 {
    while n > 1000 {
        z *= pow2(1000);
        n -= 1000;
    }
    while n < -1000 {
        z *= pow2(-1000);
        n += 1000;
    }
    z * pow2(n)
}

/// Request parameters for the initial step size heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialStepRequest<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub max_step: f64,
    pub f0: &'a [f64],
    pub direction: f64,
    pub order: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub mode: RuntimeMode,
}

/// Empirically select a good initial step size.
///
/// Implements the algorithm from Hairer, Norsett & Wanner,
/// "Solving Ordinary Differential Equations I: Nonstiff Problems", Sec. II.4.
///
/// # Contract (P2C-001-D2)
/// - Matches SciPy's `select_initial_step` from `_ivp/common.py`.
/// - Returns `Ok(h_abs)` with h_abs > 0 on success.
/// - Returns `Ok(f64::INFINITY)` for empty systems (n=0).
/// - Returns `Ok(0.0)` for zero-length intervals.
/// - In Hardened mode, validates that all inputs are finite.
pub fn select_initial_step<F>(
    fun: &mut F,
    request: &InitialStepRequest<'_>,
) -> Result<f64, IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64> + ?Sized,
{
    let n = request.y0.len();
    validate_rhs_shape(request.f0.len(), n)?;

    // Empty system: infinite step is fine.
    if n == 0 {
        return Ok(f64::INFINITY);
    }

    // Hardened mode: validate inputs are finite.
    if request.mode == RuntimeMode::Hardened {
        if !request.y0.iter().all(|v| v.is_finite()) {
            return Err(IntegrateValidationError::NonFiniteY0);
        }
        if !request.f0.iter().all(|v| v.is_finite()) {
            return Err(IntegrateValidationError::NonFiniteF0);
        }
    }

    let interval_length = (request.t_bound - request.t0).abs();
    if interval_length == 0.0 {
        return Ok(0.0);
    }

    // Compute scale = atol + |y0| * rtol and d0 = norm(y0 / scale).
    // This keeps the same formula as before, but avoids building temporary
    // scaled vectors solely for RMS computation.
    let mut scale = Vec::with_capacity(n);
    let mut scaled_y0_sum_sq = 0.0_f64;
    match &request.atol {
        ToleranceValue::Scalar(atol) => {
            for &y in request.y0 {
                let s = atol + y.abs() * request.rtol;
                scale.push(s);
                let scaled = y / s;
                scaled_y0_sum_sq += scaled * scaled;
            }
        }
        ToleranceValue::Vector(atol_vec) => {
            for (&y, &a) in request.y0.iter().zip(atol_vec.iter()) {
                let s = a + y.abs() * request.rtol;
                scale.push(s);
                let scaled = y / s;
                scaled_y0_sum_sq += scaled * scaled;
            }
        }
    }
    let d0 = if scale.is_empty() {
        0.0
    } else {
        (scaled_y0_sum_sq / scale.len() as f64).sqrt()
    };

    // d1 = norm(f0 / scale)
    let mut scaled_f0_sum_sq = 0.0_f64;
    let mut scaled_f0_len = 0usize;
    for (&f, &s) in request.f0.iter().zip(scale.iter()) {
        let scaled = f / s;
        scaled_f0_sum_sq += scaled * scaled;
        scaled_f0_len += 1;
    }
    let d1 = if scaled_f0_len == 0 {
        0.0
    } else {
        (scaled_f0_sum_sq / scaled_f0_len as f64).sqrt()
    };

    // Initial guess h0
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let h0 = h0.min(interval_length);

    // Euler step: y1 = y0 + h0 * direction * f0
    let y1: Vec<f64> = request
        .y0
        .iter()
        .zip(request.f0.iter())
        .map(|(y, f)| y + h0 * request.direction * f)
        .collect();

    // Evaluate f1 = fun(t0 + h0 * direction, y1)
    let f1 = fun(request.t0 + h0 * request.direction, &y1);
    validate_rhs_shape(f1.len(), n)?;
    if request.mode == RuntimeMode::Hardened && !f1.iter().all(|v| v.is_finite()) {
        return Err(IntegrateValidationError::NonFiniteF0);
    }

    // d2 = norm((f1 - f0) / scale) / h0
    let mut diff_scaled_sum_sq = 0.0_f64;
    let mut diff_scaled_len = 0usize;
    for ((&f1v, &f0v), &s) in f1.iter().zip(request.f0.iter()).zip(scale.iter()) {
        let diff = (f1v - f0v) / s;
        diff_scaled_sum_sq += diff * diff;
        diff_scaled_len += 1;
    }
    let d2 = if diff_scaled_len == 0 {
        0.0
    } else {
        (diff_scaled_sum_sq / diff_scaled_len as f64).sqrt() / h0
    };

    // Compute h1
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        if h0.is_nan() {
            f64::NAN
        } else {
            (1e-6_f64).max(h0 * 1e-3)
        }
    } else {
        let max_d = if d1.is_nan() || d2.is_nan() {
            f64::NAN
        } else {
            d1.max(d2)
        };
        // Integer orders (every RK tableau, BDF's first step) take the
        // deterministic root; only a caller-supplied non-integral order falls
        // back to the platform `powf`.
        let root = request.order + 1.0;
        if root.fract() == 0.0 && (1.0..=64.0).contains(&root) {
            kth_root(0.01 / max_d, root as u32)
        } else {
            (0.01 / max_d).powf(1.0 / root)
        }
    };

    // Return min(100 * h0, h1, interval_length, max_step)
    let min_h1 = if h0.is_nan() || h1.is_nan() {
        f64::NAN
    } else {
        (100.0 * h0).min(h1)
    };
    let min_interval = if min_h1.is_nan() || interval_length.is_nan() {
        f64::NAN
    } else {
        min_h1.min(interval_length)
    };
    let final_h = if min_interval.is_nan() || request.max_step.is_nan() {
        f64::NAN
    } else {
        min_interval.min(request.max_step)
    };
    Ok(final_h)
}

/// `num_jac`'s thresholds `(eps^0.875, eps^0.75, eps^0.25, eps^0.5)` without the platform
/// `powf`. `f64::EPSILON` is `2^-52`, so the last three are the powers of two `2^-39`,
/// `2^-13` and `2^-26`, which IEEE `sqrt` and a product produce exactly. `eps^0.875` is
/// `sqrt(eps * eps^0.75) = sqrt(2^-91)`, correctly rounded by `sqrt`. The results are the
/// same on every platform and equal to the exact values a correctly rounded `pow` returns.
fn eps_powers() -> (f64, f64, f64, f64) {
    let sqrt_eps = f64::EPSILON.sqrt();
    let quarter = sqrt_eps.sqrt();
    let three_quarters = sqrt_eps * quarter;
    let seven_eighths = (f64::EPSILON * three_quarters).sqrt();
    (seven_eighths, three_quarters, quarter, sqrt_eps)
}

/// Finite-difference Jacobian of `fun` at `(t, y)`, SciPy's `_ivp.common.num_jac` (dense
/// path), for the implicit solvers.
///
/// `f` is `fun(t, y)`; `threshold` is the solver's `atol` (per component). `factor` is the
/// per-column relative step SciPy carries between calls as `jac_factor`: `None` starts at
/// `sqrt(eps)`, and it is updated in place. Each column steps `h_j = (y_j + factor_j ·
/// y_scale_j) - y_j` in the direction `f_j` points, with `y_scale_j = max(threshold_j, |y_j|)`.
/// A column whose largest difference is below `eps^0.875` of the function's magnitude is
/// retried with a 10x larger factor and the retry is kept if its relative difference is
/// larger. The factor then grows 10x if the difference is below `eps^0.75`, shrinks 10x if
/// above `eps^0.25`, and never falls below `1000·eps`. The evaluations are NOT counted in
/// `nfev` (SciPy's `fun_vectorized` bypasses its counter).
// Column-indexed arithmetic over several parallel per-column arrays, as in SciPy.
#[allow(clippy::needless_range_loop)]
pub(crate) fn num_jac<F>(
    fun: &mut F,
    t: f64,
    y: &[f64],
    f: &[f64],
    threshold: &[f64],
    factor: &mut Option<Vec<f64>>,
) -> nalgebra::DMatrix<f64>
where
    F: FnMut(f64, &[f64]) -> Vec<f64> + ?Sized,
{
    let n = y.len();
    if n == 0 {
        return nalgebra::DMatrix::zeros(0, 0);
    }
    let (diff_reject, diff_small, diff_big, sqrt_eps) = eps_powers();
    let min_factor = 1e3 * f64::EPSILON;

    let mut fac = factor.take().unwrap_or_else(|| vec![sqrt_eps; n]);
    let mut y_scale = vec![0.0; n];
    let mut h = vec![0.0; n];
    for j in 0..n {
        let sign = if f[j] >= 0.0 { 1.0 } else { -1.0 };
        y_scale[j] = sign * threshold[j].max(y[j].abs());
        h[j] = (y[j] + fac[j] * y_scale[j]) - y[j];
        while h[j] == 0.0 {
            fac[j] *= 10.0;
            h[j] = (y[j] + fac[j] * y_scale[j]) - y[j];
        }
    }

    // One column: fun at y + h_j e_j, its difference from f, and the row of the largest
    // |difference| (the first such row, as numpy's argmax).
    let column = |fun: &mut F, j: usize, hj: f64| {
        let mut yj = y.to_vec();
        yj[j] += hj;
        let f_new = fun(t, &yj);
        let diff: Vec<f64> = f_new.iter().zip(f).map(|(a, b)| a - b).collect();
        let mut max_ind = 0;
        for i in 1..n {
            if diff[i].abs() > diff[max_ind].abs() {
                max_ind = i;
            }
        }
        let max_diff = diff[max_ind].abs();
        let scale = f[max_ind].abs().max(f_new[max_ind].abs());
        (diff, max_diff, scale)
    };

    let mut diffs = Vec::with_capacity(n);
    let mut max_diff = vec![0.0; n];
    let mut scale = vec![0.0; n];
    for j in 0..n {
        let (d, md, s) = column(fun, j, h[j]);
        diffs.push(d);
        max_diff[j] = md;
        scale[j] = s;
    }
    for j in 0..n {
        if max_diff[j] < diff_reject * scale[j] {
            let new_factor = 10.0 * fac[j];
            let h_new = (y[j] + new_factor * y_scale[j]) - y[j];
            let (d, md, s) = column(fun, j, h_new);
            if max_diff[j] * s < md * scale[j] {
                fac[j] = new_factor;
                h[j] = h_new;
                diffs[j] = d;
                scale[j] = s;
                max_diff[j] = md;
            }
        }
    }

    let jac = nalgebra::DMatrix::from_fn(n, n, |i, j| diffs[j][i] / h[j]);
    for j in 0..n {
        if max_diff[j] < diff_small * scale[j] {
            fac[j] *= 10.0;
        } else if max_diff[j] > diff_big * scale[j] {
            fac[j] *= 0.1;
        }
        fac[j] = fac[j].max(min_factor);
    }
    *factor = Some(fac);
    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eps_powers_are_the_exact_values() {
        let (seven_eighths, three_quarters, quarter, sqrt_eps) = eps_powers();
        assert_eq!(sqrt_eps.to_bits(), 2.0_f64.powi(-26).to_bits());
        assert_eq!(quarter.to_bits(), 2.0_f64.powi(-13).to_bits());
        assert_eq!(three_quarters.to_bits(), 2.0_f64.powi(-39).to_bits());
        // 2^-45.5 = sqrt(2) * 2^-46; the power-of-two scaling is exact, so the correctly
        // rounded value is the correctly rounded SQRT_2 scaled.
        assert_eq!(
            seven_eighths.to_bits(),
            (std::f64::consts::SQRT_2 * 2.0_f64.powi(-46)).to_bits()
        );
    }

    #[test]
    fn eps_powers_match_this_hosts_powf() {
        // Not an oracle (powf is the thing being replaced), a same-host check that the
        // replacement did not move num_jac's thresholds where the platform powf is exact.
        let (seven_eighths, three_quarters, quarter, sqrt_eps) = eps_powers();
        assert_eq!(seven_eighths, f64::EPSILON.powf(0.875));
        assert_eq!(three_quarters, f64::EPSILON.powf(0.75));
        assert_eq!(quarter, f64::EPSILON.powf(0.25));
        assert_eq!(sqrt_eps, f64::EPSILON.powf(0.5));
    }

    #[test]
    fn select_initial_step_empty_system() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![], &request).unwrap();
        assert!(h.is_infinite());
    }

    #[test]
    fn select_initial_step_zero_interval() {
        let request = InitialStepRequest {
            t0: 1.0,
            y0: &[1.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[0.5],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![0.5], &request).unwrap();
        assert_eq!(h, 0.0);
    }

    #[test]
    fn select_initial_step_rejects_wrong_size_f0() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[1.0, 2.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[0.5],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let err = select_initial_step(&mut |_t, _y| vec![0.5, 0.25], &request)
            .expect_err("wrong-size f0");
        assert_eq!(
            err,
            IntegrateValidationError::RhsWrongShape {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn select_initial_step_rejects_wrong_size_probe_rhs() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[1.0, 2.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[0.5, 0.25],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let err =
            select_initial_step(&mut |_t, _y| vec![0.5], &request).expect_err("wrong-size f1");
        assert_eq!(
            err,
            IntegrateValidationError::RhsWrongShape {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn select_initial_step_exponential_decay() {
        // y' = -0.5*y, y(0) = 1.0
        let y0 = [1.0];
        let f0 = [-0.5]; // fun(0, [1.0])
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 10.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0, // RK45 error_estimator_order
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h > 0.0, "step must be positive, got {h}");
        assert!(h <= 10.0, "step must not exceed interval, got {h}");
    }

    #[test]
    fn select_initial_step_respects_max_step() {
        let y0 = [1.0];
        let f0 = [-0.5];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 100.0,
            max_step: 0.001,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h <= 0.001, "step must respect max_step, got {h}");
    }

    #[test]
    fn select_initial_step_backward_integration() {
        let y0 = [1.0];
        let f0 = [-0.5];
        let request = InitialStepRequest {
            t0: 10.0,
            y0: &y0,
            t_bound: 0.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: -1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0]], &request).unwrap();
        assert!(h > 0.0, "h_abs must be positive");
        assert!(h <= 10.0, "h_abs must not exceed interval");
    }

    #[test]
    fn select_initial_step_vector_atol() {
        let y0 = [1.0, 100.0];
        let f0 = [-0.5, -50.0];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 10.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Vector(vec![1e-6, 1e-4]),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, y| vec![-0.5 * y[0], -0.5 * y[1]], &request).unwrap();
        assert!(h > 0.0);
    }

    #[test]
    fn select_initial_step_small_derivatives() {
        // When d0 < 1e-5 and d1 < 1e-5, h0 = 1e-6
        let y0 = [1e-8];
        let f0 = [1e-8];
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &y0,
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &f0,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let h = select_initial_step(&mut |_t, _y| vec![1e-8], &request).unwrap();
        assert!(h > 0.0);
    }

    #[test]
    fn select_initial_step_hardened_rejects_non_finite_y0() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[f64::NAN],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[1.0],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Hardened,
        };

        let err = select_initial_step(&mut |_t, _y| vec![], &request)
            .expect_err("non-finite y0 must fail in Hardened mode");
        assert_eq!(err, IntegrateValidationError::NonFiniteY0);
    }

    #[test]
    fn select_initial_step_hardened_rejects_non_finite_f0() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[f64::INFINITY],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Hardened,
        };

        let err = select_initial_step(&mut |_t, _y| vec![], &request)
            .expect_err("non-finite f0 must fail in Hardened mode");
        assert_eq!(err, IntegrateValidationError::NonFiniteF0);
    }

    #[test]
    fn select_initial_step_hardened_rejects_non_finite_probe_rhs() {
        let request = InitialStepRequest {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &[1.0],
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Hardened,
        };

        let err = select_initial_step(&mut |_t, _y| vec![f64::NAN], &request)
            .expect_err("non-finite probe rhs must fail in Hardened mode");
        assert_eq!(err, IntegrateValidationError::NonFiniteF0);
    }

    fn ulps(a: f64, b: f64) -> u64 {
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    fn kth_root_special_values() {
        for k in 1..=9 {
            assert_eq!(kth_root(0.0, k).to_bits(), 0.0_f64.to_bits());
            assert_eq!(kth_root(f64::INFINITY, k), f64::INFINITY);
            assert!(kth_root(f64::NAN, k).is_nan());
            assert!(kth_root(-1.0, k).is_nan());
        }
        assert!(kth_root(2.0, 0).is_nan());
        assert_eq!(kth_root(7.25, 1), 7.25);
    }

    #[test]
    fn kth_root_is_exact_on_perfect_powers() {
        assert_eq!(kth_root(32.0, 5), 2.0);
        assert_eq!(kth_root(1.0, 7), 1.0);
        assert_eq!(kth_root(1.0 / 1024.0, 10), 0.5);
        assert_eq!(kth_root(81.0, 4), 3.0);
        assert_eq!(kth_root(1e-15, 5), 1e-3);
        assert_eq!(kth_root(9.0, 2), 3.0);
    }

    #[test]
    fn kth_root_satisfies_its_definition_across_magnitudes() {
        // Deterministic sweep: every decade from 1e-300 to 1e300, plus a
        // subnormal and f64::MAX, at each order step control uses. The oracle
        // is the definition, root^k == x. `powf(x, 1/k)` is not a valid oracle
        // at extreme magnitudes: 1/k is itself rounded, and ln|x| amplifies that
        // error (about 1e-14 relative at 1e-300 for k = 3, measured against a
        // 200-bit reference, where this root was within 2e-19).
        let mut inputs: Vec<f64> = (-300..=300).map(|e| 1.2345 * 10f64.powi(e)).collect();
        inputs.push(f64::MIN_POSITIVE / 1024.0);
        inputs.push(f64::MAX);
        for k in 2..=9 {
            for &x in &inputs {
                let got = kth_root(x, k);
                // Compare scaled copies so root^k cannot overflow: with
                // got = g * 2^m (g in [1, 2)), check g^k against x * 2^(-m*k).
                let m = ((got.to_bits() >> 52) & 0x7ff) as i64 - 1023;
                let g = scale_pow2(got, -m);
                let mut power = 1.0;
                for _ in 0..k {
                    power *= g;
                }
                let target = scale_pow2(x, -m * i64::from(k));
                let rel = (power / target - 1.0).abs();
                assert!(
                    rel <= f64::from(k + 2) * f64::EPSILON,
                    "k={k} x={x:e}: root {got:e}, root^k/x - 1 = {rel:e}"
                );
            }
        }
    }

    #[test]
    fn kth_root_matches_powf_where_powf_is_accurate() {
        // Near unit magnitude the rounding of 1/k costs powf under an ulp, so
        // the two must agree closely there (step control lives in this range).
        for e in -10..=10 {
            for k in 2..=9 {
                let x = 1.2345 * 10f64.powi(e);
                let got = kth_root(x, k);
                let want = x.powf(1.0 / f64::from(k));
                assert!(
                    ulps(got, want) <= 4,
                    "k={k} x={x:e}: {got:e} vs powf {want:e}"
                );
            }
        }
    }

    #[test]
    fn kth_root_bits_are_pinned() {
        // Uses only + - * / on exactly decomposed inputs, so these bits hold on
        // every IEEE-754 platform. A change here changes every step sequence.
        // Recorded from the first build of this function (x86-64 Linux). The
        // correctly rounded sqrt(0.5) is ...F3BD, one ulp above: this root is
        // deterministic, not correctly rounded.
        let pinned: [(f64, u32, u64); 4] = [
            (0.123_456_789, 5, 0x3FE5_0F4B_0556_11F7),
            (3.0e-7, 5, 0x3FA9_644F_C6A9_FC93),
            (123_456.0, 8, 0x4011_516B_CBD7_A68F),
            (0.5, 2, 0x3FE6_A09E_667F_3BCC),
        ];
        let observed: Vec<u64> = pinned
            .iter()
            .map(|&(x, k, _)| kth_root(x, k).to_bits())
            .collect();
        let expected: Vec<u64> = pinned.iter().map(|&(_, _, bits)| bits).collect();
        assert_eq!(observed, expected, "observed bits: {observed:#X?}");
    }
}
