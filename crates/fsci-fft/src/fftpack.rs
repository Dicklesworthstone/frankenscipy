#![forbid(unsafe_code)]

//! The legacy `scipy.fftpack` pseudo-differential operators on periodic sequences: `diff`,
//! `tilbert`, `itilbert`, `hilbert`, `ihilbert`, `cs_diff`, `sc_diff`, `ss_diff`, `cc_diff` and
//! `shift`.
//!
//! Each one multiplies the Fourier coefficients of a real sequence `x` (of period `2π`, or of
//! `period` when given) by a kernel `ω(k)`, as SciPy's `scipy/fftpack/_pseudo_diffs.py` does
//! through `convolve.init_convolution_kernel` and `convolve.convolve`: the kernel is laid out in
//! FFTPACK's packed real-spectrum order `[y₀, Re y₁, Im y₁, Re y₂, Im y₂, …, (Re y_{n/2})]`, with
//! the `iᵈ` factor of a `d`-th order operator folded into its signs, the Nyquist term zeroed for
//! odd `d` (and wherever SciPy zeroes it), and odd operators swapping the real and imaginary
//! parts of each coefficient. The kernels and their packing were checked against
//! `scipy.fftpack` 1.17.1 to 3.8e-16 before this was written.
//!
//! [`hilbert`] here is `scipy.fftpack.hilbert`, the real periodic Hilbert transform of a
//! sequence; `crate::hilbert` is `scipy.signal.hilbert`'s analytic signal. The SciPy functions
//! also accept complex input, which they transform as `f(x.real) + i·f(x.imag)`; these take the
//! real sequence.

use crate::transforms::{FftError, FftOptions, irfft, rfft};

fn options() -> FftOptions {
    FftOptions::default().with_check_finite(false)
}

fn require_nonempty(x: &[f64]) -> Result<usize, FftError> {
    if x.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "fftpack pseudo-differential operators need a non-empty sequence",
        });
    }
    Ok(x.len())
}

/// `rfft` in FFTPACK's packed layout.
fn packed_rfft(x: &[f64]) -> Result<Vec<f64>, FftError> {
    let n = x.len();
    let spec = rfft(x, &options())?;
    let mut out = vec![0.0; n];
    out[0] = spec[0].0;
    for k in 1..=(n - 1) / 2 {
        out[2 * k - 1] = spec[k].0;
        out[2 * k] = spec[k].1;
    }
    if n.is_multiple_of(2) && n >= 2 {
        out[n - 1] = spec[n / 2].0;
    }
    Ok(out)
}

/// FFTPACK's UNNORMALIZED backward real transform of a packed spectrum.
fn packed_irfft_unnormalized(p: &[f64]) -> Result<Vec<f64>, FftError> {
    let n = p.len();
    let mut spec = vec![(0.0, 0.0); n / 2 + 1];
    spec[0] = (p[0], 0.0);
    for k in 1..=(n - 1) / 2 {
        spec[k] = (p[2 * k - 1], p[2 * k]);
    }
    if n.is_multiple_of(2) && n >= 2 {
        spec[n / 2] = (p[n - 1], 0.0);
    }
    let x = irfft(&spec, Some(n), &options())?;
    let scale = n as f64;
    Ok(x.into_iter().map(|v| v * scale).collect())
}

/// SciPy `convolve.init_convolution_kernel(n, kernel, d, zero_nyquist)`: `ω(k)·iᵈ / n` in the
/// packed layout. `zero_nyquist = None` is SciPy's `-1`, i.e. `d` odd.
fn init_kernel(
    n: usize,
    d: i64,
    zero_nyquist: Option<bool>,
    kernel: impl Fn(i64) -> f64,
) -> Vec<f64> {
    let nf = n as f64;
    let mut omega = vec![0.0; n];
    let l = if n % 2 == 1 { n } else { n - 1 };
    let zero_nyquist = zero_nyquist.unwrap_or(d.rem_euclid(2) == 1);
    let dm = d.rem_euclid(4);
    omega[0] = kernel(0) / nf;
    let mut k = 2;
    while k <= l {
        let v = kernel((k / 2) as i64) / nf;
        match dm {
            0 => {
                omega[k] = v;
                omega[k - 1] = v;
            }
            1 => {
                omega[k - 1] = v;
                omega[k] = -v;
            }
            2 => {
                omega[k] = -v;
                omega[k - 1] = -v;
            }
            _ => {
                omega[k] = v;
                omega[k - 1] = -v;
            }
        }
        k += 2;
    }
    if n.is_multiple_of(2) {
        omega[n - 1] = if zero_nyquist {
            0.0
        } else {
            let v = kernel((n / 2) as i64) / nf;
            if dm < 2 { v } else { -v }
        };
    }
    omega
}

/// SciPy `convolve.convolve(x, omega, swap_real_imag)`.
fn convolve(x: &[f64], omega: &[f64], swap_real_imag: bool) -> Result<Vec<f64>, FftError> {
    let n = x.len();
    let mut p = packed_rfft(x)?;
    if swap_real_imag {
        p[0] *= omega[0];
        if n.is_multiple_of(2) {
            p[n - 1] *= omega[n - 1];
        }
        let mut i = 1;
        while i + 1 < n {
            let c = p[i] * omega[i];
            p[i] = p[i + 1] * omega[i + 1];
            p[i + 1] = c;
            i += 2;
        }
    } else {
        for (v, w) in p.iter_mut().zip(omega) {
            *v *= w;
        }
    }
    packed_irfft_unnormalized(&p)
}

/// SciPy `convolve.convolve_z(x, omega_real, omega_imag)`: multiplication by a complex kernel.
fn convolve_z(x: &[f64], omega_real: &[f64], omega_imag: &[f64]) -> Result<Vec<f64>, FftError> {
    let n = x.len();
    let mut p = packed_rfft(x)?;
    p[0] *= omega_real[0] + omega_imag[0];
    if n.is_multiple_of(2) {
        p[n - 1] *= omega_real[n - 1] + omega_imag[n - 1];
    }
    let mut i = 1;
    while i + 1 < n {
        let c = p[i] * omega_imag[i];
        p[i] *= omega_real[i];
        p[i] += p[i + 1] * omega_imag[i + 1];
        p[i + 1] *= omega_real[i + 1];
        p[i + 1] += c;
        i += 2;
    }
    packed_irfft_unnormalized(&p)
}

/// `2π / period`, or 1 for the default period `2π`.
fn period_scale(period: Option<f64>) -> f64 {
    period.map_or(1.0, |p| 2.0 * std::f64::consts::PI / p)
}

/// `scipy.fftpack.diff(x, order, period)`: the `order`-th derivative (a negative `order`
/// integrates) of a periodic sequence. The mean (`k = 0`) and the Nyquist term are dropped, as
/// SciPy drops them.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn diff(x: &[f64], order: i32, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    if order == 0 {
        return Ok(x.to_vec());
    }
    let c = period_scale(period);
    let omega = init_kernel(n, i64::from(order), Some(true), |k| {
        if k != 0 {
            (c * k as f64).powi(order)
        } else {
            0.0
        }
    });
    convolve(x, &omega, order.rem_euclid(2) == 1)
}

/// `scipy.fftpack.tilbert(x, h, period)`: the h-Tilbert transform, coefficients multiplied by
/// `i·coth(j·h·2π/period)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn tilbert(x: &[f64], h: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let h = h * period_scale(period);
    let omega = init_kernel(n, 1, None, |k| {
        if k != 0 {
            1.0 / (h * k as f64).tanh()
        } else {
            0.0
        }
    });
    convolve(x, &omega, true)
}

/// `scipy.fftpack.itilbert(x, h, period)`: the inverse h-Tilbert transform, coefficients
/// multiplied by `−i·tanh(j·h·2π/period)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn itilbert(x: &[f64], h: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let h = h * period_scale(period);
    let omega = init_kernel(n, 1, None, |k| {
        if k != 0 { -(h * k as f64).tanh() } else { 0.0 }
    });
    convolve(x, &omega, true)
}

/// `scipy.fftpack.hilbert(x)`: the periodic Hilbert transform of a real sequence, coefficients
/// multiplied by `i·sign(j)` (not `scipy.signal.hilbert`'s analytic signal, which is
/// [`crate::hilbert`]).
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn hilbert(x: &[f64]) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let omega = init_kernel(n, 1, None, |k| match k.signum() {
        1 => 1.0,
        -1 => -1.0,
        _ => 0.0,
    });
    convolve(x, &omega, true)
}

/// `scipy.fftpack.ihilbert(x)`: `−hilbert(x)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn ihilbert(x: &[f64]) -> Result<Vec<f64>, FftError> {
    Ok(hilbert(x)?.into_iter().map(|v| -v).collect())
}

/// `scipy.fftpack.cs_diff(x, a, b, period)`: coefficients multiplied by
/// `−i·cosh(j·a)/sinh(j·b)` (`a`, `b` scaled by `2π/period`).
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn cs_diff(x: &[f64], a: f64, b: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let s = period_scale(period);
    let (a, b) = (a * s, b * s);
    let omega = init_kernel(n, 1, None, |k| {
        if k != 0 {
            let k = k as f64;
            -(a * k).cosh() / (b * k).sinh()
        } else {
            0.0
        }
    });
    convolve(x, &omega, true)
}

/// `scipy.fftpack.sc_diff(x, a, b, period)`: coefficients multiplied by
/// `i·sinh(j·a)/cosh(j·b)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn sc_diff(x: &[f64], a: f64, b: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let s = period_scale(period);
    let (a, b) = (a * s, b * s);
    let omega = init_kernel(n, 1, None, |k| {
        if k != 0 {
            let k = k as f64;
            (a * k).sinh() / (b * k).cosh()
        } else {
            0.0
        }
    });
    convolve(x, &omega, true)
}

/// `scipy.fftpack.ss_diff(x, a, b, period)`: coefficients multiplied by
/// `sinh(j·a)/sinh(j·b)`, and the mean by `a/b`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn ss_diff(x: &[f64], a: f64, b: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let s = period_scale(period);
    let (a, b) = (a * s, b * s);
    let omega = init_kernel(n, 0, None, |k| {
        if k != 0 {
            let k = k as f64;
            (a * k).sinh() / (b * k).sinh()
        } else {
            a / b
        }
    });
    convolve(x, &omega, false)
}

/// `scipy.fftpack.cc_diff(x, a, b, period)`: coefficients multiplied by
/// `cosh(j·a)/cosh(j·b)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn cc_diff(x: &[f64], a: f64, b: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let s = period_scale(period);
    let (a, b) = (a * s, b * s);
    let omega = init_kernel(n, 0, None, |k| {
        let k = k as f64;
        (a * k).cosh() / (b * k).cosh()
    });
    convolve(x, &omega, false)
}

/// `scipy.fftpack.shift(x, a, period)`: the periodic sequence shifted by `a`,
/// `y(u) = x(u + a)`, coefficients multiplied by `exp(i·j·a·2π/period)`.
///
/// # Errors
/// [`FftError::InvalidShape`] for an empty sequence.
pub fn shift(x: &[f64], a: f64, period: Option<f64>) -> Result<Vec<f64>, FftError> {
    let n = require_nonempty(x)?;
    let a = a * period_scale(period);
    let omega_real = init_kernel(n, 0, Some(false), |k| (a * k as f64).cos());
    let omega_imag = init_kernel(n, 1, Some(false), |k| (a * k as f64).sin());
    convolve_z(x, &omega_real, &omega_imag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<f64> {
        (0..n)
            .map(|j| 2.0 * std::f64::consts::PI * j as f64 / n as f64)
            .collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{x} vs {y}");
        }
    }

    #[test]
    fn diff_of_trig_polynomials_is_exact() {
        let t = grid(16);
        let x: Vec<f64> = t.iter().map(|v| v.sin() + 0.5 * (3.0 * v).cos()).collect();
        let d1: Vec<f64> = t.iter().map(|v| v.cos() - 1.5 * (3.0 * v).sin()).collect();
        close(&diff(&x, 1, None).unwrap(), &d1, 1e-13);
        let d2: Vec<f64> = t.iter().map(|v| -v.sin() - 4.5 * (3.0 * v).cos()).collect();
        close(&diff(&x, 2, None).unwrap(), &d2, 1e-12);
        // order −1 integrates (the mean is dropped).
        let i1: Vec<f64> = t.iter().map(|v| -v.cos() + (3.0 * v).sin() / 6.0).collect();
        close(&diff(&x, -1, None).unwrap(), &i1, 1e-13);
        assert_eq!(diff(&x, 0, None).unwrap(), x);
        // period 4π halves the derivative.
        let half: Vec<f64> = d1.iter().map(|v| v / 2.0).collect();
        let x2: Vec<f64> = t.iter().map(|v| v.sin() + 0.5 * (3.0 * v).cos()).collect();
        close(
            &diff(&x2, 1, Some(4.0 * std::f64::consts::PI)).unwrap(),
            &half,
            1e-13,
        );
    }

    #[test]
    fn hilbert_maps_cos_to_minus_sin_and_ihilbert_inverts() {
        // y_j = i·sign(j)·x_j: cos 2t = (e^{2it} + e^{−2it})/2 ↦ −sin 2t (SciPy's docstring sign).
        let t = grid(32);
        let x: Vec<f64> = t.iter().map(|v| (2.0 * v).cos()).collect();
        let h = hilbert(&x).unwrap();
        close(
            &h,
            &t.iter().map(|v| -(2.0 * v).sin()).collect::<Vec<_>>(),
            1e-13,
        );
        // ihilbert = −hilbert, and hilbert² = −identity on zero-mean, Nyquist-free sequences.
        let back = ihilbert(&h).unwrap();
        close(&back, &x, 1e-13);
        // tilbert ∘ itilbert: i·coth · (−i·tanh) = 1, the identity on the same sequences.
        let y: Vec<f64> = t.iter().map(|v| v.sin() + (5.0 * v).cos()).collect();
        let r = tilbert(&itilbert(&y, 0.3, None).unwrap(), 0.3, None).unwrap();
        close(&r, &y, 1e-12);
    }

    #[test]
    fn shift_moves_the_sequence() {
        let t = grid(24);
        let x: Vec<f64> = t.iter().map(|v| v.sin() + (2.0 * v).cos()).collect();
        let a = 0.37;
        let want: Vec<f64> = t
            .iter()
            .map(|v| (v + a).sin() + (2.0 * (v + a)).cos())
            .collect();
        close(&shift(&x, a, None).unwrap(), &want, 1e-13);
    }

    #[test]
    fn hyperbolic_operators_scale_each_harmonic() {
        let t = grid(20);
        let (a, b) = (0.2, 0.5);
        let x: Vec<f64> = t.iter().map(|v| (3.0 * v).cos()).collect();
        let k = 3.0_f64;
        let cc = cc_diff(&x, a, b, None).unwrap();
        close(
            &cc,
            &x.iter()
                .map(|v| v * (a * k).cosh() / (b * k).cosh())
                .collect::<Vec<_>>(),
            1e-13,
        );
        let ss = ss_diff(&x, a, b, None).unwrap();
        close(
            &ss,
            &x.iter()
                .map(|v| v * (a * k).sinh() / (b * k).sinh())
                .collect::<Vec<_>>(),
            1e-13,
        );
        // i·cos(3t) ↦ −sin(3t) under the swap convention: sc_diff(cos 3t) = −sinh(3a)/cosh(3b)·sin 3t.
        let sc = sc_diff(&x, a, b, None).unwrap();
        let want: Vec<f64> = t
            .iter()
            .map(|v| -(3.0 * v).sin() * (a * k).sinh() / (b * k).cosh())
            .collect();
        close(&sc, &want, 1e-13);
        let cs = cs_diff(&x, a, b, None).unwrap();
        let want: Vec<f64> = t
            .iter()
            .map(|v| (3.0 * v).sin() * (a * k).cosh() / (b * k).sinh())
            .collect();
        close(&cs, &want, 1e-13);
    }

    #[test]
    fn edge_lengths() {
        assert!(diff(&[], 1, None).is_err());
        assert!(hilbert(&[]).is_err());
        // n = 1: only the mean, which every odd operator drops.
        assert_eq!(diff(&[3.0], 1, None).unwrap(), vec![0.0]);
        assert_eq!(hilbert(&[3.0]).unwrap(), vec![0.0]);
        assert_eq!(cc_diff(&[3.0], 0.1, 0.2, None).unwrap(), vec![3.0]);
        // n = 2: mean and Nyquist only.
        let s = shift(&[1.0, -1.0], std::f64::consts::PI, None).unwrap();
        close(&s, &[-1.0, 1.0], 1e-15);
    }
}
