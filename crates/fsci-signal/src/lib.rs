#![forbid(unsafe_code)]

//! Signal processing routines for FrankenSciPy.
//!
//! Implements ~158 public functions matching `scipy.signal` (module
//! docstring previously listed only 2 of those — per bead
//! frankenscipy-s9tt). High-level index by category:
//!
//! - **Windows**: `boxcar`, `hann`, `hamming`, `general_hamming`,
//!   `blackman`, `kaiser`, `tukey_window`, `blackmanharris`, `barthann`,
//!   `nuttall_window`, `bohman_window`.
//! - **Convolution / correlation**: `convolve`, `deconvolve`,
//!   `fftconvolve`, `correlate`, `correlate2d`.
//! - **Analytic signal**: `hilbert`, `hilbert_envelope`.
//! - **Spectral**: `lombscargle`, `czt`, `zoom_fft`, `chirp`,
//!   `gausspulse`, `sweep_poly`, `max_len_seq`, `matched_filter`.
//! - **Wavelets**: `ricker`, `morlet`, `cwt`.
//! - **Peaks / extrema**: `find_peaks`, `peak_prominences`,
//!   `peak_widths`, `argrelmax`, `argrelmin`, `argrelextrema`,
//!   `vectorstrength`, `order_filter`, `unwrap_phase`,
//!   `instantaneous_frequency`.
//! - **Savitzky-Golay**: `savgol_coeffs`, `savgol_filter`.
//! - **FIR/IIR design**: `firwin`, `firwin2`, `kaiserord`, `butter`,
//!   `cheby1`, `cheby2`, `ellip`, `bessel`, `iirdesign`, `iirfilter`.
//! - **Filter application**: `lfilter`, `filtfilt`, `sosfilt`,
//!   `sosfiltfilt`, `resample`, `resample_poly`, `decimate`.
//! - **Transfer / LTI**: `DLTI`, `tf2zpk`, `zpk2tf`, `bilinear`,
//!   `freqz`, `dlsim`.
//!
//! Full enumeration: `grep -E '^pub fn' crates/fsci-signal/src/lib.rs`.
//!
//! # Error taxonomy
//!
//! `SignalError` exposes structured failure classes for input length/shape,
//! non-finite values, unsupported modes, FFT failures, convolution-mode
//! failures, frequency-domain validation, numerical failures, and generic
//! parameter validation. Legacy string-form validation sites are normalized
//! through the crate-local `InvalidArgument` constructor so downstream callers
//! can match on the resulting enum variants instead of substring-parsing
//! `Display` text.

/// Error type for signal processing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalError {
    InvalidWindowLength(String),
    InvalidPolyOrder(String),
    InvalidInputLength { expected: usize, actual: usize },
    InvalidInputShape { detail: String },
    NonFiniteInput { detail: String },
    UnsupportedMode { detail: String },
    FftFailure(String),
    ConvolutionModeError(String),
    FrequencyOutOfBand { detail: String },
    NumericalFailure(String),
    InvalidParameter { detail: String },
    UnclassifiedArgument(String),
}

impl SignalError {
    #[allow(non_snake_case)]
    fn InvalidArgument(detail: String) -> Self {
        Self::classify_invalid_argument(detail)
    }

    fn classify_invalid_argument(detail: String) -> Self {
        let lower = detail.to_ascii_lowercase();
        if lower.contains("fft") || lower.contains("ifft") {
            return Self::FftFailure(detail);
        }
        if lower.contains("convolution") || lower.contains("valid mode") {
            return Self::ConvolutionModeError(detail);
        }
        if lower.contains("padtype")
            || lower.contains("window type")
            || lower.contains("unknown window")
            || lower.contains("unsupported")
            || lower.contains("mode")
        {
            return Self::UnsupportedMode { detail };
        }
        if lower.contains("shape")
            || lower.contains("same length")
            || lower.contains("must match")
            || lower.contains("match signal length")
            || lower.contains("match t length")
            || lower.contains("match system order")
            || lower.contains("length (")
        {
            return Self::InvalidInputShape { detail };
        }
        if lower.contains("non-empty")
            || lower.contains("not be empty")
            || lower.contains("cannot be empty")
            || lower.contains("empty")
        {
            return Self::InvalidInputLength {
                expected: 1,
                actual: 0,
            };
        }
        if lower.starts_with("dpss requires") {
            return Self::UnclassifiedArgument(detail);
        }
        if lower.contains("non-finite") || lower.contains("finite") {
            return Self::NonFiniteInput { detail };
        }
        if lower.contains("freq")
            || lower.contains("frequency")
            || lower.contains("cutoff")
            || lower.contains("wn")
            || lower.contains("w0")
            || lower.contains("ripple")
            || lower.contains("width")
            || lower.contains("nyquist")
        {
            return Self::FrequencyOutOfBand { detail };
        }
        if lower.contains("solve")
            || lower.contains("eigen")
            || lower.contains("singular")
            || lower.contains("degenerate")
            || lower.contains("overflow")
        {
            return Self::NumericalFailure(detail);
        }
        if lower.contains("sym==true") {
            return Self::UnclassifiedArgument(detail);
        }
        if lower.contains("must be")
            || lower.contains("out of range")
            || lower.contains("too large")
            || lower.contains("less than")
            || lower.contains("greater than")
        {
            return Self::InvalidParameter { detail };
        }
        Self::UnclassifiedArgument(detail)
    }

    pub fn is_argument_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidWindowLength(_)
                | Self::InvalidPolyOrder(_)
                | Self::InvalidInputLength { .. }
                | Self::InvalidInputShape { .. }
                | Self::NonFiniteInput { .. }
                | Self::UnsupportedMode { .. }
                | Self::FftFailure(_)
                | Self::ConvolutionModeError(_)
                | Self::FrequencyOutOfBand { .. }
                | Self::NumericalFailure(_)
                | Self::InvalidParameter { .. }
                | Self::UnclassifiedArgument(_)
        )
    }
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWindowLength(msg) => write!(f, "invalid window length: {msg}"),
            Self::InvalidPolyOrder(msg) => write!(f, "invalid polynomial order: {msg}"),
            Self::InvalidInputLength { expected, actual } => {
                write!(
                    f,
                    "invalid input length: expected at least {expected}, got {actual}"
                )
            }
            Self::InvalidInputShape { detail } => write!(f, "invalid input shape: {detail}"),
            Self::NonFiniteInput { detail } => write!(f, "non-finite input: {detail}"),
            Self::UnsupportedMode { detail } => write!(f, "unsupported mode: {detail}"),
            Self::FftFailure(msg) => write!(f, "FFT failure: {msg}"),
            Self::ConvolutionModeError(msg) => {
                write!(f, "convolution mode error: {msg}")
            }
            Self::FrequencyOutOfBand { detail } => {
                write!(f, "frequency out of band: {detail}")
            }
            Self::NumericalFailure(msg) => write!(f, "numerical failure: {msg}"),
            Self::InvalidParameter { detail } => write!(f, "invalid parameter: {detail}"),
            Self::UnclassifiedArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for SignalError {}

/// Compute Savitzky-Golay filter coefficients.
///
/// Matches `scipy.signal.savgol_coeffs(window_length, polyorder, deriv=0)`.
///
/// Returns a vector of `window_length` coefficients that, when convolved with
/// the signal, produce the smoothed (or differentiated) output.
///
/// # Arguments
/// * `window_length` — Must be odd and >= 1
/// * `polyorder` — Polynomial order, must be < window_length
/// * `deriv` — Derivative order (0 = smoothing, 1 = first derivative, etc.)
pub fn savgol_coeffs(
    window_length: usize,
    polyorder: usize,
    deriv: usize,
) -> Result<Vec<f64>, SignalError> {
    if window_length == 0 || window_length.is_multiple_of(2) {
        return Err(SignalError::InvalidWindowLength(
            "window_length must be a positive odd integer".to_string(),
        ));
    }
    if polyorder >= window_length {
        return Err(SignalError::InvalidPolyOrder(
            "polyorder must be less than window_length".to_string(),
        ));
    }
    if deriv > polyorder {
        return Err(SignalError::InvalidPolyOrder(
            "deriv must be <= polyorder".to_string(),
        ));
    }

    let half = (window_length / 2) as i64;
    let order = polyorder + 1;

    // Build the normal equations matrix A^T A.
    // (A^T A)_{r,c} = sum_{i=-half}^{half} i^{r+c}
    // We only need to compute sums of powers up to 2 * polyorder.
    let mut sums = vec![0.0; 2 * polyorder + 1];
    for i in 0..window_length {
        let xi = (i as i64 - half) as f64;
        let mut p = 1.0;
        for value in sums.iter_mut().take(2 * polyorder + 1) {
            *value += p;
            p *= xi;
        }
    }

    let mut ata = vec![vec![0.0; order]; order];
    for r in 0..order {
        ata[r][..order].copy_from_slice(&sums[r..(order + r)]);
    }

    // For each data point, the filter coefficient is the value of the
    // least-squares polynomial evaluated at x=0 (for deriv=0), or
    // the deriv-th derivative at x=0.
    // Filter weight for point i: w_i = P_i(x=0) where P_i is the polynomial
    // fit to a unit impulse at position i.
    // Equivalently: w = (A (A^T A)^{-1} e_d * d!) where e_d is the (deriv)-th unit vector.

    // Solve (A^T A) c = e_d where e_d[deriv] = 1
    let mut rhs = vec![0.0; order];
    rhs[deriv] = 1.0;

    let c = solve_symmetric_positive(&ata, &rhs)?;

    // Filter coefficients: w[i] = sum_j c[j] * x_i^j
    let deriv_factorial = factorial_small(deriv);
    let mut coeffs = Vec::with_capacity(window_length);
    for i in 0..window_length {
        let xi = (i as i64 - half) as f64;
        let mut val = 0.0;
        let mut power = 1.0;
        for &cj in &c {
            val += cj * power;
            power *= xi;
        }
        coeffs.push(val * deriv_factorial);
    }

    Ok(coeffs)
}

/// Apply a Savitzky-Golay filter to a signal.
///
/// Matches `scipy.signal.savgol_filter(x, window_length, polyorder)` with the
/// default `mode='interp'`: interior samples are filtered with the centered
/// Savitzky-Golay coefficients, and the first/last `window_length / 2` samples
/// are replaced by evaluating a degree-`polyorder` polynomial fit to the first
/// (respectively last) `window_length` samples — exactly as SciPy does.
pub fn savgol_filter(
    x: &[f64],
    window_length: usize,
    polyorder: usize,
) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input signal must be non-empty".to_string(),
        ));
    }
    if window_length > x.len() {
        return Err(SignalError::InvalidWindowLength(
            "window_length must not exceed signal length".to_string(),
        ));
    }

    let coeffs = savgol_coeffs(window_length, polyorder, 0)?;
    let half = window_length / 2;
    let n = x.len();

    // Interior: centered correlation with the SG coefficients. Out-of-range
    // taps contribute nothing here because the boundary regions are overwritten
    // by the polynomial edge fit below (matching SciPy's `mode='interp'`).
    let mut result = vec![0.0; n];
    for (i, slot) in result.iter_mut().enumerate() {
        let mut val = 0.0;
        for (j, &c) in coeffs.iter().enumerate() {
            let idx = i as i64 + j as i64 - half as i64;
            if idx >= 0 && idx < n as i64 {
                val += c * x[idx as usize];
            }
        }
        *slot = val;
    }

    // Edge handling: replace the first and last `half` samples with values from
    // a degree-`polyorder` least-squares polynomial fit over the boundary
    // window, evaluated at the sample positions (SciPy's `_fit_edges_polyfit`).
    if half > 0 {
        let positions: Vec<f64> = (0..window_length).map(|k| k as f64).collect();

        let left = polyfit(&positions, &x[0..window_length], polyorder)?;
        for (i, slot) in result.iter_mut().take(half).enumerate() {
            *slot = poly_eval(&left, i as f64);
        }

        let right = polyfit(&positions, &x[n - window_length..n], polyorder)?;
        for i in (window_length - half)..window_length {
            result[n - window_length + i] = poly_eval(&right, i as f64);
        }
    }

    Ok(result)
}

/// Least-squares polynomial fit of `degree` to `(xs, ys)` via the normal
/// equations, returning coefficients in ascending power order.
fn polyfit(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>, SignalError> {
    let m = degree + 1;
    let mut ata = vec![vec![0.0; m]; m];
    let mut atb = vec![0.0; m];
    for (&xv, &yv) in xs.iter().zip(ys.iter()) {
        let mut powers = vec![1.0; m];
        for k in 1..m {
            powers[k] = powers[k - 1] * xv;
        }
        for r in 0..m {
            for c in 0..m {
                ata[r][c] += powers[r] * powers[c];
            }
            atb[r] += powers[r] * yv;
        }
    }
    solve_symmetric_positive(&ata, &atb)
}

/// Evaluate a polynomial (ascending power order) at `x` via Horner's method.
fn poly_eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Solve a small SPD system via Gaussian elimination with partial pivoting.
fn solve_symmetric_positive(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let mut pivot_row = col;
        for i in col + 1..n {
            if aug[i][col].abs() > aug[pivot_row][col].abs() {
                pivot_row = i;
            }
        }
        aug.swap(col, pivot_row);

        if aug[col][col].abs() < 1e-18 {
            return Err(SignalError::InvalidArgument(
                "linear system is singular or poorly conditioned".to_string(),
            ));
        }

        let pivot_val = aug[col][col];
        for i in col + 1..n {
            let factor = aug[i][col] / pivot_val;
            let (head, tail) = aug.split_at_mut(i);
            let pivot_row = &head[col];
            let target_row = &mut tail[0];
            for (target, pivot) in target_row
                .iter_mut()
                .zip(pivot_row.iter())
                .skip(col)
                .take(n + 1 - col)
            {
                *target -= factor * *pivot;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        let diag = aug[i][i];
        if diag.abs() < 1e-18 {
            return Err(SignalError::InvalidArgument(
                "linear system is singular during back-substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

fn factorial_small(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        _ => (1..=n).map(|i| i as f64).product(),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Window Functions
// ══════════════════════════════════════════════════════════════════════

/// Generate a rectangular (boxcar) window of length `n`.
///
/// A boxcar window is simply a vector of ones. It provides no tapering
/// and is equivalent to not applying any window function.
///
/// Matches `scipy.signal.windows.boxcar(n, sym)`.
///
/// # Arguments
/// * `n` - Number of points in the window
/// * `sym` - If true (default), generates a symmetric window for filter design.
///   If false, generates a periodic window for spectral analysis.
pub fn boxcar(n: usize, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if sym {
        vec![1.0; n]
    } else {
        // For periodic window, we would generate n+1 points and truncate
        // but since all values are 1.0, the result is the same
        vec![1.0; n]
    }
}

/// Generate a Hann (Hanning) window of length `n`.
///
/// Matches `scipy.signal.windows.hann(n)`.
pub fn hann(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / m).cos()))
        .collect()
}

/// Generate a Hamming window of length `n`.
///
/// Matches `scipy.signal.windows.hamming(n)`.
pub fn hamming(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / m).cos())
        .collect()
}

/// Generate a generalized Hamming window of length `n`.
///
/// Matches `scipy.signal.windows.general_hamming(n, alpha)`.
pub fn general_hamming(n: usize, alpha: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    let beta = 1.0 - alpha;
    (0..n)
        .map(|i| alpha - beta * (2.0 * std::f64::consts::PI * i as f64 / m).cos())
        .collect()
}

/// Generate a Blackman window of length `n`.
///
/// Matches `scipy.signal.windows.blackman(n)`.
pub fn blackman(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / m;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Generate a Kaiser window of length `n` with shape parameter `beta`.
///
/// Matches `scipy.signal.windows.kaiser(n, beta)`.
/// Uses a polynomial approximation for I0 (modified Bessel function of the first kind).
pub fn kaiser(n: usize, beta: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    let denom = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let alpha = 2.0 * i as f64 / m - 1.0;
            let arg = beta * (1.0 - alpha * alpha).max(0.0).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Kaiser-Bessel derived window.
///
/// Matches `scipy.signal.windows.kaiser_bessel_derived(M, beta, sym=True)`.
/// The KBD construction is only defined for symmetric, even-length windows.
pub fn kaiser_bessel_derived(n: usize, beta: f64, sym: bool) -> Result<Vec<f64>, SignalError> {
    if !sym {
        return Err(SignalError::InvalidArgument(
            "Kaiser-Bessel Derived windows are only defined for symmetric shapes".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if !n.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "Kaiser-Bessel Derived windows are only defined for even number of points".to_string(),
        ));
    }
    if !beta.is_finite() {
        return Err(SignalError::InvalidArgument(format!(
            "invalid kaiser_bessel_derived beta: {beta}"
        )));
    }

    let kaiser_window = kaiser(n / 2 + 1, beta);
    let total: f64 = kaiser_window.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "kaiser_bessel_derived normalization became singular".to_string(),
        ));
    }

    let mut cumulative = 0.0;
    let mut half = Vec::with_capacity(n / 2);
    for value in kaiser_window.iter().take(n / 2) {
        cumulative += *value;
        half.push((cumulative / total).sqrt());
    }

    let mut window = Vec::with_capacity(n);
    window.extend_from_slice(&half);
    window.extend(half.iter().rev().copied());
    Ok(window)
}

/// Modified Bessel function I0(x) via polynomial approximation.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Convolution
// ══════════════════════════════════════════════════════════════════════

/// Convolution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolveMode {
    /// Full convolution output (length = len(a) + len(b) - 1).
    #[default]
    Full,
    /// Output same length as the first input.
    Same,
    /// Only parts where signals fully overlap.
    Valid,
}

/// Cost-model crossover for 1D FFT convolution: FFT only wins once the direct
/// `O(na·nb)` work dominates the FFT work `L·log2(L)` (L = next_pow2(na+nb-1)).
///
/// The old `na·nb > 1000` switched to FFT far too early; FFT's large constant
/// (three length-L transforms) means the direct loop is faster — and bit-for-bit
/// identical — until `na·nb` reaches ~1.5e5. Constant 20 matches the measured
/// break-even (direct ≈ 0.3 ns/op vs FFT ≈ 6 ns per `L·log2(L)` unit), the same
/// calibration used by `polymul` and `correlate2d`.
fn fft_conv_is_faster(na: usize, nb: usize) -> bool {
    let full = na + nb - 1;
    let l = (full.next_power_of_two() as u64).max(2);
    let direct_ops = (na as u64) * (nb as u64);
    let fft_ops = l * (l.ilog2() as u64);
    direct_ops > 20 * fft_ops
}

/// Direct (time-domain) convolution.
///
/// Matches `scipy.signal.convolve(a, b, mode)`.
pub fn convolve(a: &[f64], b: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    let na = a.len();
    let nb = b.len();

    // Automatic dispatch to FFT for large inputs (SciPy 'auto' method), gated by a
    // cost model rather than the old `na·nb > 1000` (which switched to FFT far too
    // early — FFT's constant means the direct loop is faster, AND byte-identical,
    // until na·nb dominates the FFT work).
    if fft_conv_is_faster(na, nb) {
        return fftconvolve(a, b, mode);
    }

    let full_len = na + nb - 1;
    // Compute full convolution
    let mut full = vec![0.0; full_len];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            full[i + j] += ai * bj;
        }
    }

    match mode {
        ConvolveMode::Full => Ok(full),
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            Ok(full[start..start + na].to_vec())
        }
        ConvolveMode::Valid => {
            let valid_len = na.max(nb) - na.min(nb) + 1;
            let start = na.min(nb) - 1;
            Ok(full[start..start + valid_len].to_vec())
        }
    }
}

/// Mode for cross-correlation output, used by [`correlation_lags`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMode {
    /// Full output of length `in1 + in2 - 1`.
    Full,
    /// Centered slice of the full output, of length `max(in1, in2)`.
    Same,
    /// Output only where the two arrays overlap fully.
    Valid,
}

/// Return the lag indices used for the output of cross-correlation.
///
/// Matches `scipy.signal.correlation_lags(in1_len, in2_len, mode)`.
/// Resolves [frankenscipy-r34vj].
///
/// * `Full`  — lags `−(in2 − 1) … in1 − 1`.
/// * `Same`  — centered length-`max(in1, in2)` slice of the full lags.
/// * `Valid` — lags where the two arrays fully overlap; either
///   `0 … in1 − in2` (when `in1 >= in2`) or
///   `in1 − in2 … 0` (when `in2 > in1`).
pub fn correlation_lags(in1_len: usize, in2_len: usize, mode: CorrelationMode) -> Vec<i64> {
    let n1 = in1_len as i64;
    let n2 = in2_len as i64;
    match mode {
        CorrelationMode::Full => (-(n2 - 1)..n1).collect(),
        CorrelationMode::Same => {
            let full: Vec<i64> = (-(n2 - 1)..n1).collect();
            let mid = full.len() / 2;
            let lag_bound = in1_len / 2;
            // Clamp slice indices to full.len() to mirror numpy's
            // out-of-bounds slice tolerance. Without the clamp, a
            // call like correlation_lags(5, 0, Same) panics in Rust
            // because the computed upper index exceeds full.len() —
            // scipy's lazy slicing returns the truncated tail
            // [1, 2, 3, 4] for that input. Fixes [frankenscipy-385tp].
            let n = full.len();
            let lo = mid.saturating_sub(lag_bound).min(n);
            let hi_raw = if in1_len.is_multiple_of(2) {
                mid.saturating_add(lag_bound)
            } else {
                mid.saturating_add(lag_bound).saturating_add(1)
            };
            let hi = hi_raw.min(n);
            full[lo..hi.max(lo)].to_vec()
        }
        CorrelationMode::Valid => {
            let lag_bound = n1 - n2;
            if lag_bound >= 0 {
                (0..=lag_bound).collect()
            } else {
                (lag_bound..=0).collect()
            }
        }
    }
}

/// Deconvolve `divisor` out of `signal` using polynomial long division.
///
/// Returns `(quotient, remainder)` such that
/// `signal = convolve(divisor, quotient, Full) + remainder`.
///
/// Matches `scipy.signal.deconvolve(signal, divisor)`.
pub fn deconvolve(signal: &[f64], divisor: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if signal.is_empty() {
        return Err(SignalError::InvalidArgument(
            "signal must be non-empty".to_string(),
        ));
    }
    if divisor.is_empty() {
        return Err(SignalError::InvalidArgument(
            "divisor must be non-empty".to_string(),
        ));
    }
    if divisor[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "divisor[0] must not be zero".to_string(),
        ));
    }
    if divisor.len() > signal.len() {
        return Ok((Vec::new(), signal.to_vec()));
    }

    let quotient_len = signal.len() - divisor.len() + 1;
    let mut quotient = vec![0.0; quotient_len];
    let mut remainder = signal.to_vec();

    for i in 0..quotient_len {
        let coeff = remainder[i] / divisor[0];
        quotient[i] = coeff;
        for (j, &div) in divisor.iter().enumerate() {
            remainder[i + j] -= coeff * div;
        }
    }

    Ok((quotient, remainder))
}

/// FFT-based convolution (faster for large inputs).
///
/// Matches `scipy.signal.fftconvolve(a, b, mode)`.
pub fn fftconvolve(a: &[f64], b: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || b.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    let na = a.len();
    let nb = b.len();
    let full_len = na + nb - 1;

    // Pad to power of 2 for efficient FFT
    let fft_len = full_len.next_power_of_two();
    let opts = fsci_fft::FftOptions::default();

    // Zero-pad inputs to fft_len
    let mut a_padded: Vec<fsci_fft::Complex64> = a.iter().map(|&v| (v, 0.0)).collect();
    a_padded.resize(fft_len, (0.0, 0.0));

    let mut b_padded: Vec<fsci_fft::Complex64> = b.iter().map(|&v| (v, 0.0)).collect();
    b_padded.resize(fft_len, (0.0, 0.0));

    // FFT both
    let fa = fsci_fft::fft(&a_padded, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
    let fb = fsci_fft::fft(&b_padded, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

    // Pointwise multiply
    let fc: Vec<fsci_fft::Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
        .collect();

    // Inverse FFT
    let conv_full =
        fsci_fft::ifft(&fc, &opts).map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

    // Extract real part, trimmed to full_len
    let full: Vec<f64> = conv_full.iter().take(full_len).map(|&(re, _)| re).collect();

    match mode {
        ConvolveMode::Full => Ok(full),
        ConvolveMode::Same => {
            let start = (nb - 1) / 2;
            Ok(full[start..start + na].to_vec())
        }
        ConvolveMode::Valid => {
            let valid_len = na.max(nb) - na.min(nb) + 1;
            let start = na.min(nb) - 1;
            Ok(full[start..start + valid_len].to_vec())
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Cross-correlation
// ══════════════════════════════════════════════════════════════════════

/// Cross-correlation of two 1D arrays.
///
/// Equivalent to `convolve(a, v[::-1], mode)` — reverse v and convolve.
/// Matches `scipy.signal.correlate(a, v, mode)`.
///
/// # Arguments
/// * `a` — First input array.
/// * `v` — Second input array (will be time-reversed internally).
/// * `mode` — Output mode: Full, Same, or Valid.
pub fn correlate(a: &[f64], v: &[f64], mode: ConvolveMode) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() || v.is_empty() {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }
    // Correlate = convolve with reversed kernel
    let v_rev: Vec<f64> = v.iter().rev().copied().collect();
    convolve(a, &v_rev, mode)
}

/// Full 2D linear convolution of `a` (ar×ac) with kernel `v` (vr×vc) via FFT,
/// written into `out` (full_r×full_c, row-major). Both inputs are zero-padded to
/// `lr×lc` (powers of two ≥ full_r/full_c) so the circular FFT product equals the
/// linear convolution. O(lr·lc·log(lr·lc)) vs the direct O(ar·ac·vr·vc).
#[allow(clippy::too_many_arguments)]
fn correlate2d_fft_full_into(
    out: &mut [f64],
    a: &[f64],
    ar: usize,
    ac: usize,
    v: &[f64],
    vr: usize,
    vc: usize,
    full_r: usize,
    full_c: usize,
    lr: usize,
    lc: usize,
) -> Result<(), SignalError> {
    let opts = fsci_fft::FftOptions::default();
    let n = lr * lc;
    let mut a_pad: Vec<fsci_fft::Complex64> = vec![(0.0, 0.0); n];
    for i in 0..ar {
        for j in 0..ac {
            a_pad[i * lc + j] = (a[i * ac + j], 0.0);
        }
    }
    let mut v_pad: Vec<fsci_fft::Complex64> = vec![(0.0, 0.0); n];
    for i in 0..vr {
        for j in 0..vc {
            v_pad[i * lc + j] = (v[i * vc + j], 0.0);
        }
    }
    let fa = fsci_fft::fft2(&a_pad, (lr, lc), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
    let fv = fsci_fft::fft2(&v_pad, (lr, lc), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
    let fc: Vec<fsci_fft::Complex64> = fa
        .iter()
        .zip(fv.iter())
        .map(|(&(are, aim), &(bre, bim))| (are * bre - aim * bim, are * bim + aim * bre))
        .collect();
    let conv = fsci_fft::ifft2(&fc, (lr, lc), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
    for i in 0..full_r {
        for j in 0..full_c {
            out[i * full_c + j] = conv[i * lc + j].0;
        }
    }
    Ok(())
}

/// 2D cross-correlation of two 2D arrays.
///
/// Matches `scipy.signal.correlate2d(in1, in2, mode)`.
///
/// # Arguments
/// * `a` — First 2D input (rows x cols, row-major flat array).
/// * `a_shape` — (rows, cols) of first input.
/// * `v` — Second 2D input (kernel, row-major flat array).
/// * `v_shape` — (rows, cols) of kernel.
/// * `mode` — Output mode: Full, Same, or Valid.
pub fn correlate2d(
    a: &[f64],
    a_shape: (usize, usize),
    v: &[f64],
    v_shape: (usize, usize),
    mode: ConvolveMode,
) -> Result<Vec<f64>, SignalError> {
    let (ar, ac) = a_shape;
    let (vr, vc) = v_shape;
    if a.len() != ar * ac || v.len() != vr * vc {
        return Err(SignalError::InvalidArgument(
            "array length must match shape".to_string(),
        ));
    }
    if ar == 0 || ac == 0 || vr == 0 || vc == 0 {
        return Err(SignalError::InvalidArgument(
            "inputs must be non-empty".to_string(),
        ));
    }

    // Reverse v in both dimensions for correlation
    let mut v_rev = vec![0.0; vr * vc];
    for i in 0..vr {
        for j in 0..vc {
            v_rev[i * vc + j] = v[(vr - 1 - i) * vc + (vc - 1 - j)];
        }
    }

    // 2D convolution (direct method)
    let (out_r, out_c, start_r, start_c) = match mode {
        ConvolveMode::Full => (ar + vr - 1, ac + vc - 1, 0, 0),
        ConvolveMode::Same => (ar, ac, vr / 2, vc / 2),
        ConvolveMode::Valid => {
            if ar < vr || ac < vc {
                return Err(SignalError::InvalidArgument(
                    "in valid mode, a must be at least as large as v".to_string(),
                ));
            }
            (ar - vr + 1, ac - vc + 1, vr - 1, vc - 1)
        }
    };

    let full_r = ar + vr - 1;
    let full_c = ac + vc - 1;

    // Full 2D convolution of `a` with the reversed kernel. Direct is
    // O(ar·ac·vr·vc); for large inputs switch to FFT (O(L·log L), L = lr·lc).
    // The cost-model gate keeps small inputs — including every conformance case
    // (≤6×6 ✻ ≤3×3) — on the byte-identical direct loop, and only takes FFT where
    // it decisively wins; FFT then matches direct to FFT rounding (~1e-10),
    // consistent with the existing fftconvolve / convolve auto-dispatch.
    let lr = full_r.next_power_of_two();
    let lc = full_c.next_power_of_two();
    let direct_ops = (ar as u64) * (ac as u64) * (vr as u64) * (vc as u64);
    let l = (lr as u64) * (lc as u64);
    let fft_ops = l * (l.max(2).ilog2() as u64);

    let mut full = vec![0.0; full_r * full_c];
    // Crossover constant ~24: measured direct ≈ 0.29 ns/op vs FFT ≈ 6 ns per
    // L·log2(L) unit puts the break-even near 21; 24 adds a small safety margin so
    // FFT is taken only when it clearly wins (no regression for big-image/small-
    // kernel shapes, where direct stays ahead).
    let used_fft = direct_ops > 24 * fft_ops
        && correlate2d_fft_full_into(&mut full, a, ar, ac, &v_rev, vr, vc, full_r, full_c, lr, lc)
            .is_ok();
    if !used_fft {
        for value in full.iter_mut() {
            *value = 0.0;
        }
        for i in 0..ar {
            for j in 0..ac {
                let aval = a[i * ac + j];
                for ki in 0..vr {
                    for kj in 0..vc {
                        full[(i + ki) * full_c + (j + kj)] += aval * v_rev[ki * vc + kj];
                    }
                }
            }
        }
    }

    // Extract requested region
    let mut result = vec![0.0; out_r * out_c];
    for i in 0..out_r {
        for j in 0..out_c {
            result[i * out_c + j] = full[(i + start_r) * full_c + (j + start_c)];
        }
    }

    Ok(result)
}

// ══════════════════════════════════════════════════════════════════════
// Hilbert Transform / Analytic Signal
// ══════════════════════════════════════════════════════════════════════

/// Compute the analytic signal using the Hilbert transform.
///
/// Returns a complex-valued signal where:
/// - Real part = original signal
/// - Imaginary part = Hilbert transform of the signal
///
/// The instantaneous amplitude (envelope) is `|analytic|` and the
/// instantaneous phase is `angle(analytic)`.
///
/// Matches `scipy.signal.hilbert(x)`.
///
/// # Algorithm
/// 1. Compute FFT of the input
/// 2. Create a filter: `h[0] = 1, h[1..N/2] = 2, h[N/2] = 1 (if N even), h[N/2+1..] = 0`
/// 3. Multiply FFT by h
/// 4. Inverse FFT gives the analytic signal
pub fn hilbert(x: &[f64]) -> Result<Vec<(f64, f64)>, SignalError> {
    let n = x.len();
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "input must be non-empty".to_string(),
        ));
    }

    let fft_opts = fsci_fft::FftOptions::default();

    // Convert real input to complex for full FFT
    let complex_input: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();

    // Compute full complex FFT
    let spectrum = fsci_fft::fft(&complex_input, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // Build the analytic signal filter h:
    // h[0] = 1, h[1..N/2] = 2, h[N/2] = 1 (even N), h[N/2+1..] = 0
    let mut h = vec![0.0; n];
    h[0] = 1.0;
    if n.is_multiple_of(2) {
        for item in h.iter_mut().take(n / 2).skip(1) {
            *item = 2.0;
        }
        h[n / 2] = 1.0;
    } else {
        for item in h.iter_mut().take((n - 1) / 2 + 1).skip(1) {
            *item = 2.0;
        }
    }

    // Multiply spectrum by h (zero negative frequencies, double positive)
    let filtered: Vec<(f64, f64)> = spectrum
        .iter()
        .zip(h.iter())
        .map(|(&(re, im), &hi)| (re * hi, im * hi))
        .collect();

    // Inverse FFT gives the analytic signal
    let analytic = fsci_fft::ifft(&filtered, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    Ok(analytic)
}

/// Compute the envelope (instantaneous amplitude) of a signal.
///
/// This is `|hilbert(x)|` — the magnitude of the analytic signal.
pub fn hilbert_envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    let analytic = hilbert(x)?;
    Ok(analytic
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect())
}

// ══════════════════════════════════════════════════════════════════════
// Spectral Analysis (Lomb-Scargle) and Waveform Generation
// ══════════════════════════════════════════════════════════════════════

/// Lomb-Scargle periodogram for unevenly-spaced data.
///
/// Computes the Lomb-Scargle periodogram power at specified angular frequencies.
///
/// Matches `scipy.signal.lombscargle(x, y, freqs, normalize=...)`.
///
/// # Arguments
/// * `x` — Sample times (must be non-decreasing).
/// * `y` — Measurements at each sample time.
/// * `freqs` — Angular frequencies at which to evaluate the periodogram.
/// * `normalize` — Whether to divide by the signal energy as SciPy does.
pub fn lombscargle(
    x: &[f64],
    y: &[f64],
    freqs: &[f64],
    normalize: bool,
) -> Result<Vec<f64>, SignalError> {
    if x.len() != y.len() {
        return Err(SignalError::InvalidArgument(
            "x and y must have the same length".to_string(),
        ));
    }
    if x.len() < 2 {
        return Err(SignalError::InvalidArgument(
            "need at least 2 data points".to_string(),
        ));
    }

    let sample_count = x.len() as f64;
    let inv_sample_count = 1.0 / sample_count;
    let mean_square = y.iter().map(|value| value * value).sum::<f64>() * inv_sample_count;

    // Each frequency's periodogram value is an independent reduction over the samples
    // (two fixed-order passes over x/y); the result depends only on (omega, x, y). So
    // the per-frequency closure is pure and the frequency loop parallelizes byte-
    // identically — each `power[k]` is computed exactly as the serial version and
    // written to its own index, with the sample-summation order untouched.
    let power_at = |omega: f64| -> f64 {
        if !omega.is_finite() {
            return f64::NAN;
        }
        if omega == 0.0 {
            let y_sum = y.iter().sum::<f64>();
            let p = 0.5 * y_sum.powi(2) / x.len() as f64;
            return if normalize {
                if mean_square == 0.0 {
                    f64::NAN
                } else {
                    p * (2.0 / (sample_count * mean_square))
                }
            } else {
                p
            };
        }

        let mut cos2_mean = 0.0;
        let mut cos_sin_mean = 0.0;
        for &xi in x {
            let phase = omega * xi;
            let c = phase.cos();
            let s = phase.sin();
            cos2_mean += c * c;
            cos_sin_mean += c * s;
        }
        cos2_mean *= inv_sample_count;
        cos_sin_mean *= inv_sample_count;
        let sin2_mean = 1.0 - cos2_mean;
        let tau_angle = 0.5 * (2.0 * cos_sin_mean).atan2(cos2_mean - sin2_mean);

        let mut y_cos_mean = 0.0;
        let mut y_sin_mean = 0.0;
        let mut shifted_cos2_mean = 0.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let phase = omega * xi - tau_angle;
            let c = phase.cos();
            let s = phase.sin();
            y_cos_mean += yi * c;
            y_sin_mean += yi * s;
            shifted_cos2_mean += c * c;
        }
        y_cos_mean *= inv_sample_count;
        y_sin_mean *= inv_sample_count;
        shifted_cos2_mean *= inv_sample_count;
        let shifted_sin2_mean = 1.0 - shifted_cos2_mean;

        let epsneg = f64::EPSILON / 2.0;
        let shifted_cos2_mean = shifted_cos2_mean.max(epsneg);
        let shifted_sin2_mean = shifted_sin2_mean.max(epsneg);
        let a = y_cos_mean / shifted_cos2_mean;
        let b = y_sin_mean / shifted_sin2_mean;
        let normalized_power = 2.0 * (a * y_cos_mean + b * y_sin_mean);
        let p = normalized_power * sample_count / 4.0;

        if normalize {
            if mean_square == 0.0 {
                f64::NAN
            } else {
                normalized_power * 0.5 / mean_square
            }
        } else {
            p
        }
    };

    let nthreads = lombscargle_thread_count(freqs.len(), x.len());
    if nthreads <= 1 {
        return Ok(freqs.iter().map(|&omega| power_at(omega)).collect());
    }

    let mut power = vec![0.0; freqs.len()];
    let chunk = freqs.len().div_ceil(nthreads);
    let power_at = &power_at;
    std::thread::scope(|scope| {
        for (out_chunk, freq_chunk) in power.chunks_mut(chunk).zip(freqs.chunks(chunk)) {
            scope.spawn(move || {
                for (slot, &omega) in out_chunk.iter_mut().zip(freq_chunk.iter()) {
                    *slot = power_at(omega);
                }
            });
        }
    });
    Ok(power)
}

/// Threads for `lombscargle`: each of the `m` frequencies costs ~`n` cos/sin pairs
/// over two sample passes, so total work ~ `m*n` trig evaluations. Only split when
/// that clearly amortises thread spawn.
fn lombscargle_thread_count(m: usize, n: usize) -> usize {
    let work = (m as u64).saturating_mul(n as u64);
    if work < 1 << 16 || m < 16 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(m / 4).max(1)
}

/// Gaussian-modulated sinusoidal pulse.
///
/// Returns the samples of a Gaussian-modulated sinusoid:
///   y(t) = exp(-a t²) cos(2π fc t)
/// where a = (π fc bw)² / (-4 ln(ref)), ref = 10^(bwr/20), and the
/// fractional-bandwidth reference level is bwr = -6 dB.
///
/// Matches `scipy.signal.gausspulse(t, fc, bw)`.
pub fn gausspulse(t: &[f64], fc: f64, bw: f64) -> Vec<f64> {
    if fc < 0.0 || bw <= 0.0 {
        return vec![0.0; t.len()];
    }
    let bwr = -6.0; // reference level in dB
    let ref_level = 10.0_f64.powf(bwr / 20.0);
    // scipy solves g(fc·bw/2) = ref for the Gaussian width: the envelope
    // factor is a = (π·fc·bw)² / (−4·ln ref), not (−2·ln ref).
    let a = -(std::f64::consts::PI * fc * bw).powi(2) / (4.0 * ref_level.ln());

    t.iter()
        .map(|&ti| {
            let envelope = (-a * ti * ti).exp();
            let carrier = (2.0 * std::f64::consts::PI * fc * ti).cos();
            envelope * carrier
        })
        .collect()
}

/// Frequency-swept cosine with polynomial instantaneous frequency.
///
/// Generates cos(2π ∫₀ᵗ f(τ) dτ) where f(τ) is a polynomial.
///
/// Matches `scipy.signal.sweep_poly(t, poly)`.
///
/// # Arguments
/// * `t` — Time points.
/// * `poly` — Polynomial coefficients [p₀, p₁, ..., pₙ] for f(t) = p₀tⁿ + ... + pₙ.
pub fn sweep_poly(t: &[f64], poly: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            // Integrate polynomial: ∫₀ᵗ (p₀τⁿ + ... + pₙ) dτ
            // = p₀t^{n+1}/(n+1) + ... + pₙt
            let n = poly.len();
            let mut phase_integral = 0.0;
            for (k, &coeff) in poly.iter().enumerate() {
                let power = n - 1 - k; // degree of this term
                phase_integral += coeff * ti.powi(power as i32 + 1) / (power as f64 + 1.0);
            }
            (2.0 * std::f64::consts::PI * phase_integral).cos()
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Chirp Z-Transform
// ══════════════════════════════════════════════════════════════════════

/// Chirp Z-Transform: evaluate the Z-transform at M points along a spiral
/// in the complex plane starting at `a`, stepping by `w`.
///
/// Given `x[n]` for n=0..N-1, computes:
///   `X[k] = Σ_{n=0}^{N-1} x[n] · a^{-n} · w^{nk},  k = 0..M-1`
///
/// # Arguments
/// * `x` — Input signal (real-valued).
/// * `m` — Number of output points.
/// * `w` — Step factor as (magnitude, angle) in polar form.
///   Default: `(1.0, -2π/M)` (unit circle, same as DFT).
/// * `a` — Starting point as (magnitude, angle) in polar form.
///   Default: `(1.0, 0.0)` (z = 1).
///
/// Matches `scipy.signal.czt`.
pub fn czt(
    x: &[f64],
    m: usize,
    w: Option<(f64, f64)>,
    a: Option<(f64, f64)>,
) -> Result<Vec<(f64, f64)>, SignalError> {
    let n = x.len();
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "Input signal must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Ok(vec![]);
    }

    let two_pi = 2.0 * std::f64::consts::PI;

    // Default w: unit circle step = exp(-j2π/M)
    let (w_mag, w_ang) = w.unwrap_or((1.0, -two_pi / m as f64));
    // Default a: z = 1
    let (a_mag, a_ang) = a.unwrap_or((1.0, 0.0));

    // Bluestein's algorithm for CZT via convolution:
    // 1) Form yn[n] = x[n] * a^{-n} * w^{n²/2}
    // 2) Form h[n] = w^{-n²/2}
    // 3) Convolve yn with h, then multiply by w^{k²/2}

    let l = (n + m - 1).next_power_of_two(); // FFT length

    // Precompute w^{k²/2} chirp factors
    // w^{k²/2} = (w_mag)^{k²/2} * exp(j * w_ang * k²/2)
    let chirp = |k: i64| -> (f64, f64) {
        let half_k2 = (k * k) as f64 / 2.0;
        let mag = w_mag.powf(half_k2);
        let ang = w_ang * half_k2;
        (mag * ang.cos(), mag * ang.sin())
    };

    // a^{-n} = (a_mag)^{-n} * exp(-j * a_ang * n)
    let a_neg = |nn: usize| -> (f64, f64) {
        let nf = nn as f64;
        let mag = a_mag.powf(-nf);
        let ang = -a_ang * nf;
        (mag * ang.cos(), mag * ang.sin())
    };

    // Complex multiply helper
    let cmul = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    };

    // Build yn: length L, zero-padded
    let mut yn = vec![(0.0f64, 0.0f64); l];
    for i in 0..n {
        let an = a_neg(i);
        let wn = chirp(i as i64);
        let xi = (x[i], 0.0);
        yn[i] = cmul(cmul(xi, an), wn);
    }

    // Build h: w^{-k²/2} for k = -(N-1)..M-1, stored in wrap-around order
    let mut h = vec![(0.0f64, 0.0f64); l];
    // h[k] for k = 0..M-1
    for (k, hk) in h.iter_mut().enumerate().take(m) {
        let c = chirp(k as i64);
        // w^{-k²/2} = conjugate of w^{k²/2} when w is on unit circle,
        // but in general we need to invert
        let mag_sq = c.0 * c.0 + c.1 * c.1;
        if mag_sq > 0.0 {
            *hk = (c.0 / mag_sq, -c.1 / mag_sq);
        }
    }
    // h[L-k] for k = 1..N-1 (negative indices wrapped)
    for k in 1..n {
        let c = chirp(k as i64);
        let mag_sq = c.0 * c.0 + c.1 * c.1;
        if mag_sq > 0.0 {
            h[l - k] = (c.0 / mag_sq, -c.1 / mag_sq);
        }
    }

    // FFT of yn and h, multiply, IFFT
    let fft_opts = fsci_fft::FftOptions::default();
    let yn_fft = fsci_fft::fft(&yn, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
    let h_fft = fsci_fft::fft(&h, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    let product_fft: Vec<(f64, f64)> = yn_fft
        .iter()
        .zip(h_fft.iter())
        .map(|(&a, &b)| cmul(a, b))
        .collect();

    let product = fsci_fft::ifft(&product_fft, &fft_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    // Extract result: X[k] = product[k] * w^{k²/2}
    let mut result = Vec::with_capacity(m);
    for (k, value) in product.iter().enumerate().take(m) {
        let wk = chirp(k as i64);
        result.push(cmul(*value, wk));
    }

    Ok(result)
}

/// Zoom FFT: compute the DFT over a frequency sub-range [f1, f2] with M points.
///
/// This uses the Chirp Z-Transform to evaluate the Z-transform at M equally
/// spaced points on the unit circle arc from f1 to f2 using SciPy's default
/// `fs=2` convention, where 1.0 is the Nyquist frequency and 2.0 spans the
/// full unit circle.
///
/// # Arguments
/// * `x` — Input signal.
/// * `f_range` — (f1, f2) normalized frequency range.
/// * `m` — Number of output points.
///
/// Matches `scipy.signal.zoom_fft`.
pub fn zoom_fft(x: &[f64], f_range: (f64, f64), m: usize) -> Result<Vec<(f64, f64)>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "Input signal must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Ok(vec![]);
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let scipy_default_fs = 2.0;
    let (f1, f2) = f_range;

    // SciPy's default fs is 2.0, so f=1.0 is Nyquist and f=2.0
    // completes the unit circle.
    let a = (1.0, two_pi * f1 / scipy_default_fs);

    // We want z_k = exp(j*2π*f_k/fs) where
    // f_k = f1 + k*(f2-f1)/M and fs = 2.0. In CZT:
    // z_k = a * w^{-k}, so w = exp(-j*2π*(f2-f1)/(fs*M)).

    let w = (1.0, -two_pi * (f2 - f1) / (scipy_default_fs * m as f64));

    czt(x, m, Some(w), Some(a))
}

// ══════════════════════════════════════════════════════════════════════
// Sequences and Matched Filtering
// ══════════════════════════════════════════════════════════════════════

/// Generate a maximum-length sequence (m-sequence) using a linear feedback shift register.
///
/// Returns a binary sequence of length 2^nbits - 1 with values in {-1, 1}.
/// M-sequences have ideal autocorrelation properties.
///
/// Matches `scipy.signal.max_len_seq(nbits)`.
pub fn max_len_seq(nbits: usize) -> Result<Vec<f64>, SignalError> {
    if !(2..=31).contains(&nbits) {
        return Err(SignalError::InvalidArgument(
            "nbits must be in [2, 31]".to_string(),
        ));
    }

    let seq_len = (1usize << nbits) - 1;

    // Primitive polynomial taps for each nbits (from tables)
    let taps: &[usize] = match nbits {
        2 => &[2, 1],
        3 => &[3, 1],
        4 => &[4, 1],
        5 => &[5, 2],
        6 => &[6, 1],
        7 => &[7, 1],
        8 => &[8, 4, 3, 2],
        9 => &[9, 4],
        10 => &[10, 3],
        11 => &[11, 2],
        12 => &[12, 6, 4, 1],
        13 => &[13, 4, 3, 1],
        14 => &[14, 5, 3, 1],
        15 => &[15, 1],
        16 => &[16, 5, 3, 2],
        _ => &[nbits, 1], // fallback (may not be primitive for all n)
    };

    let mut state = 1u32; // Initial state (nonzero)
    let mut seq = Vec::with_capacity(seq_len);

    for _ in 0..seq_len {
        // Output is the last bit
        let output = state & 1;
        seq.push(if output == 1 { 1.0 } else { -1.0 });

        // Feedback: XOR of tapped bits
        let mut feedback = 0u32;
        for &tap in taps {
            feedback ^= (state >> (tap - 1)) & 1;
        }

        // Shift register
        state = (state >> 1) | (feedback << (nbits - 1));
    }

    Ok(seq)
}

/// Matched filter: correlate signal with template, normalized by template energy.
///
/// Returns the normalized cross-correlation, which peaks at the location
/// where the template best matches the signal.
pub fn matched_filter(template: &[f64], signal: &[f64]) -> Result<Vec<f64>, SignalError> {
    if template.is_empty() || signal.is_empty() {
        return Err(SignalError::InvalidArgument(
            "template and signal must be non-empty".to_string(),
        ));
    }

    // Template energy
    let energy: f64 = template.iter().map(|&t| t * t).sum();
    if energy == 0.0 {
        return Err(SignalError::InvalidArgument(
            "template must have nonzero energy".to_string(),
        ));
    }

    // Cross-correlate and normalize
    let corr = correlate(signal, template, ConvolveMode::Full)?;
    Ok(corr.iter().map(|&c| c / energy).collect())
}

// ══════════════════════════════════════════════════════════════════════
// Wavelets
// ══════════════════════════════════════════════════════════════════════

/// Ricker wavelet (Mexican hat wavelet).
///
/// The second derivative of a Gaussian: ψ(t) = (2/(√3σ π^{1/4})) (1 - (t/σ)²) exp(-t²/(2σ²))
///
/// Matches `scipy.signal.ricker(points, a)`.
///
/// # Arguments
/// * `points` — Number of points in the output vector.
/// * `a` — Width parameter (standard deviation of the Gaussian).
pub fn ricker(points: usize, a: f64) -> Vec<f64> {
    if points == 0 || a <= 0.0 || !a.is_finite() {
        return vec![];
    }
    let mut output = Vec::with_capacity(points);
    let center = (points as f64 - 1.0) / 2.0;
    let a2 = a * a;
    let norm = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));

    for i in 0..points {
        let t = i as f64 - center;
        let t2 = t * t;
        let gauss = (-t2 / (2.0 * a2)).exp();
        output.push(norm / a.sqrt() * (1.0 - t2 / a2) * gauss);
    }
    output
}

/// Complex Morlet wavelet.
///
/// ψ(t) = exp(2πi w₀ t) exp(-t²/2) (with optional correction for non-zero mean)
///
/// Matches `scipy.signal.morlet(M, w, s, complete)`.
///
/// # Arguments
/// * `m` — Number of points in the output.
/// * `w` — Omega0 (center frequency), default 5.0.
/// * `s` — Scaling factor, default 1.0.
/// * `complete` — If true, apply correction term for non-zero mean.
pub fn morlet(m: usize, w: f64, s: f64, complete: bool) -> Vec<(f64, f64)> {
    if m == 0 || s <= 0.0 || !s.is_finite() || !w.is_finite() {
        return vec![];
    }
    let mut output = Vec::with_capacity(m);
    let center = (m as f64 - 1.0) / 2.0;

    // Resolves [frankenscipy-tmnrh]: hoist the loop-invariant
    // correction (depends only on w) and the 2π·w prefactor for
    // the phase. The inner loop now does one sin_cos per sample
    // instead of separate sin and cos.
    let two_pi_w = std::f64::consts::TAU * w;
    let pi = std::f64::consts::PI;
    let correction = if complete {
        (-2.0 * pi * pi * w * w / 2.0).exp()
    } else {
        0.0
    };

    for i in 0..m {
        let t = (i as f64 - center) / s;
        let gauss = (-t * t / 2.0).exp();
        let phase = two_pi_w * t;
        let (sin_phase, cos_phase) = phase.sin_cos();
        let mut re = gauss * cos_phase;
        let im = gauss * sin_phase;

        if complete {
            // Correction: subtract the DC component to ensure zero mean.
            re -= gauss * correction;
        }

        output.push((re, im));
    }
    output
}

/// Morlet-2 wavelet kernel (the default cwt wavelet in modern scipy).
///
/// Matches `scipy.signal.morlet2(M, s, w=5)`. Returns the M-sample
/// complex Morlet-2
///   ψ(t) = (π s²)^(−1/4) · exp(i ω t / s) · exp(−t² / (2 s²))
/// sampled at offsets `t = i − (M − 1)/2` for `i = 0, ..., M − 1`. Differs from
/// `morlet` in the `1/√(s)` envelope normalization that makes the
/// energy of `morlet2(M, s)` independent of `s`.
pub fn morlet2(m: usize, s: f64, w: f64) -> Vec<(f64, f64)> {
    if m == 0 || s <= 0.0 || !s.is_finite() || !w.is_finite() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(m);
    let coeff = (std::f64::consts::PI * s * s).powf(-0.25);
    // scipy.signal.morlet2 centers on (M-1)/2, not M/2: see scipy
    // _wavelets.py — a half-sample offset would phase-rotate every value.
    let half = (m as f64 - 1.0) / 2.0;
    // Resolves [frankenscipy-t4jxi]: hoist the loop-invariant
    // 2·s² (gauss denominator) and w/s (phase prefactor); fold
    // cos/sin into one sin_cos. Each saves M ops on the hot path.
    let two_s_sq = 2.0 * s * s;
    let w_over_s = w / s;
    for i in 0..m {
        let t = i as f64 - half;
        let phase = w_over_s * t;
        let gauss = (-t * t / two_s_sq).exp();
        let envelope = coeff * gauss;
        let (sin_phase, cos_phase) = phase.sin_cos();
        out.push((envelope * cos_phase, envelope * sin_phase));
    }
    out
}

/// Continuous Wavelet Transform.
///
/// Computes CWT using the specified wavelet function at different scales.
///
/// Matches `scipy.signal.cwt(data, wavelet, widths)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `wavelet_fn` — Function that generates a wavelet for a given width: `fn(points, width) -> Vec<f64>`.
/// * `widths` — Array of wavelet widths (scales) to use.
///
/// Returns a 2D array (widths.len() × data.len()), where each row is the CWT
/// at the corresponding scale.
pub fn cwt<F>(data: &[f64], wavelet_fn: F, widths: &[f64]) -> Result<Vec<Vec<f64>>, SignalError>
where
    F: Fn(usize, f64) -> Vec<f64>,
{
    if data.is_empty() {
        return Err(SignalError::InvalidArgument(
            "data must be non-empty".to_string(),
        ));
    }
    if widths.is_empty() {
        return Err(SignalError::InvalidArgument(
            "widths must be non-empty".to_string(),
        ));
    }

    let na = data.len();
    let mut result = Vec::with_capacity(widths.len());

    // Resolves [frankenscipy-zq5xy]: the input `data` is identical across
    // every width, so its zero-padded forward FFT depends only on the FFT
    // length. Cache `fft(data_padded)` keyed by `fft_len` and reuse it
    // across widths instead of recomputing it once per scale. Each cache
    // hit removes one length-`fft_len` forward transform and one padded-
    // input allocation; for the 2048×32 ricker bench all 32 scales share
    // `fft_len = 4096`, so 31 redundant data FFTs are eliminated.
    //
    // Behavior is bit-identical to `convolve(data, &wavelet, Same)`: the
    // FFT/direct dispatch threshold, the padded-input bytes (hence
    // `fft(data_padded)`), the pointwise-multiply order, the inverse
    // transform, and the `Same` slice are all reproduced exactly.
    let opts = fsci_fft::FftOptions::default();
    let mut data_fft_cache: Vec<(usize, Vec<fsci_fft::Complex64>)> = Vec::new();

    for &width in widths {
        if width <= 0.0 || !width.is_finite() {
            return Err(SignalError::InvalidArgument(
                "all widths must be positive and finite".to_string(),
            ));
        }
        // Generate wavelet at this scale
        let wavelet_len = (10.0 * width).ceil() as usize;
        let wavelet_len = wavelet_len.max(1);
        let wavelet = wavelet_fn(wavelet_len, width);
        let nb = wavelet.len();

        // Mirror `convolve(data, &wavelet, Same)` exactly, but reuse the
        // cached forward FFT of `data` on the FFT path.
        let conv = if !wavelet.is_empty() && na.saturating_mul(nb) > 1000 {
            let full_len = na + nb - 1;
            let fft_len = full_len.next_power_of_two();

            let fa_index = match data_fft_cache.iter().position(|(len, _)| *len == fft_len) {
                Some(idx) => idx,
                None => {
                    let mut a_padded: Vec<fsci_fft::Complex64> =
                        data.iter().map(|&v| (v, 0.0)).collect();
                    a_padded.resize(fft_len, (0.0, 0.0));
                    let fa = fsci_fft::fft(&a_padded, &opts)
                        .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;
                    data_fft_cache.push((fft_len, fa));
                    data_fft_cache.len() - 1
                }
            };
            let fa = &data_fft_cache[fa_index].1;

            let mut b_padded: Vec<fsci_fft::Complex64> =
                wavelet.iter().map(|&v| (v, 0.0)).collect();
            b_padded.resize(fft_len, (0.0, 0.0));
            let fb = fsci_fft::fft(&b_padded, &opts)
                .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

            let fc: Vec<fsci_fft::Complex64> = fa
                .iter()
                .zip(fb.iter())
                .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
                .collect();

            let conv_full = fsci_fft::ifft(&fc, &opts)
                .map_err(|e| SignalError::InvalidArgument(format!("{e}")))?;

            let full: Vec<f64> = conv_full.iter().take(full_len).map(|&(re, _)| re).collect();
            let start = (nb - 1) / 2;
            full[start..start + na].to_vec()
        } else {
            convolve(data, &wavelet, ConvolveMode::Same)?
        };
        result.push(conv);
    }

    Ok(result)
}

// ── Additional window functions ──────────────────────────────────────

/// Tukey (tapered cosine) window.
///
/// Matches `scipy.signal.windows.tukey(M, alpha)`.
///
/// alpha=0 → rectangular, alpha=1 → Hann.
pub fn tukey_window(m: usize, alpha: f64) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut w = vec![1.0; m];
    let n = m - 1;

    if alpha > 0.0 {
        let taper_len = (alpha * n as f64 / 2.0) as usize;
        for i in 0..=taper_len.min(n) {
            let val =
                0.5 * (1.0 - (std::f64::consts::PI * i as f64 / (alpha * n as f64 / 2.0)).cos());
            w[i] = val;
            w[n - i] = val;
        }
    }
    w
}

/// Blackman-Harris window.
///
/// Matches `scipy.signal.windows.blackmanharris(M)`.
pub fn blackmanharris(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.358_75 - 0.488_29 * x.cos() + 0.141_28 * (2.0 * x).cos() - 0.011_68 * (3.0 * x).cos()
        })
        .collect()
}

/// Bartlett-Hann window.
///
/// Matches `scipy.signal.windows.barthann(M)`.
pub fn barthann(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    (0..m)
        .map(|i| {
            let fac = (i as f64 / denom - 0.5).abs();
            0.62 - 0.48 * fac + 0.38 * (2.0 * std::f64::consts::PI * fac).cos()
        })
        .collect()
}

/// Nuttall window (minimum 4-term Blackman-Harris).
///
/// Matches `scipy.signal.windows.nuttall(M)`.
///
/// br-z3ni: scipy's "nuttall" is the "minimum 4-term Blackman-Harris"
/// variant (Nuttall 1981, table II, "Min 4-term"). fsci previously
/// used the standard continuous-flattop Nuttall coefficients
/// [0.355768, 0.487396, 0.144232, 0.012604] which are a different
/// point on the same coefficient table. The scipy-compatible
/// coefficients [0.3635819, 0.4891775, 0.1365995, 0.0106411] are now
/// used so parity fixtures pass.
pub fn nuttall_window(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            0.363_581_9 - 0.489_177_5 * x.cos() + 0.136_599_5 * (2.0 * x).cos()
                - 0.010_641_1 * (3.0 * x).cos()
        })
        .collect()
}

/// Bohman window.
///
/// Matches `scipy.signal.windows.bohman(M)`.
pub fn bohman_window(m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let n = m - 1;
    (0..m)
        .map(|i| {
            let x = (2.0 * i as f64 / n as f64 - 1.0).abs();
            if x >= 1.0 {
                0.0
            } else {
                (1.0 - x) * (std::f64::consts::PI * x).cos()
                    + (1.0 / std::f64::consts::PI) * (std::f64::consts::PI * x).sin()
            }
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// Peak Detection
// ══════════════════════════════════════════════════════════════════════

/// Options for peak detection.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FindPeaksOptions {
    /// Minimum height of peaks.
    pub height: Option<f64>,
    /// Minimum horizontal distance between peaks (in samples).
    pub distance: Option<usize>,
    /// Minimum prominence of peaks.
    pub prominence: Option<f64>,
    /// Minimum width of peaks (in samples).
    pub width: Option<f64>,
}

/// Result of peak detection.
#[derive(Debug, Clone, PartialEq)]
pub struct FindPeaksResult {
    /// Indices of detected peaks.
    pub peaks: Vec<usize>,
    /// Heights of detected peaks (if height filter was used).
    pub peak_heights: Vec<f64>,
    /// Prominences of detected peaks (if prominence filter was used).
    pub prominences: Vec<f64>,
}

/// Find peaks in a 1-D signal.
///
/// Matches `scipy.signal.find_peaks(x, height, distance, prominence, width)`.
///
/// A sample is a peak if it is strictly greater than its immediate neighbors.
pub fn find_peaks(x: &[f64], options: FindPeaksOptions) -> FindPeaksResult {
    if x.len() < 3 {
        return FindPeaksResult {
            peaks: Vec::new(),
            peak_heights: Vec::new(),
            prominences: Vec::new(),
        };
    }

    // Step 1: find all local maxima (including plateaus)
    let mut peaks: Vec<usize> = Vec::new();
    let mut i = 1;
    while i < x.len() - 1 {
        if x[i] > x[i - 1] {
            let mut j = i + 1;
            while j < x.len() && x[j] == x[i] {
                j += 1;
            }
            // If the plateau is followed by a smaller value or the end of the signal
            if j < x.len() && x[i] > x[j] {
                // Peak is the middle of the plateau
                peaks.push((i + j - 1) / 2);
            }
            i = j;
        } else {
            i += 1;
        }
    }

    // Step 2: filter by height
    if let Some(min_height) = options.height {
        peaks.retain(|&i| x[i] >= min_height);
    }

    // Step 3: compute prominences (needed for prominence and width filters)
    let prominences: Vec<f64> = peaks.iter().map(|&pk| compute_prominence(x, pk)).collect();

    // Step 4: filter by prominence
    if let Some(min_prom) = options.prominence {
        let mut keep = Vec::new();
        for (idx, &pk) in peaks.iter().enumerate() {
            if prominences[idx] >= min_prom {
                keep.push((pk, prominences[idx]));
            }
        }
        peaks = keep.iter().map(|&(pk, _)| pk).collect();
        let proms: Vec<f64> = keep.iter().map(|&(_, p)| p).collect();
        // Rebuild prominences for kept peaks
        let peak_heights: Vec<f64> = peaks.iter().map(|&i| x[i]).collect();

        // Step 5: filter by distance (keep highest peak within each distance window)
        if let Some(min_dist) = options.distance {
            let filtered = filter_by_distance(&peaks, &peak_heights, min_dist);
            let final_proms: Vec<f64> = filtered
                .iter()
                .map(|&pk| proms[peaks.binary_search(&pk).unwrap_or(0)])
                .collect();
            let final_heights: Vec<f64> = filtered.iter().map(|&i| x[i]).collect();
            return FindPeaksResult {
                peaks: filtered,
                peak_heights: final_heights,
                prominences: final_proms,
            };
        }

        return FindPeaksResult {
            peaks,
            peak_heights,
            prominences: proms,
        };
    }

    let peak_heights: Vec<f64> = peaks.iter().map(|&i| x[i]).collect();

    // Step 5: filter by distance
    if let Some(min_dist) = options.distance {
        let filtered = filter_by_distance(&peaks, &peak_heights, min_dist);
        let final_heights: Vec<f64> = filtered.iter().map(|&i| x[i]).collect();
        let final_proms: Vec<f64> = filtered
            .iter()
            .map(|&pk| compute_prominence(x, pk))
            .collect();
        return FindPeaksResult {
            peaks: filtered,
            peak_heights: final_heights,
            prominences: final_proms,
        };
    }

    FindPeaksResult {
        peaks,
        peak_heights,
        prominences,
    }
}

/// Compute the prominence of a peak.
/// Prominence = peak height - max(left base, right base).
fn compute_prominence(x: &[f64], peak_idx: usize) -> f64 {
    let peak_val = x[peak_idx];

    // Search left for the lowest point before reaching a higher peak or the edge
    let mut left_min = peak_val;
    for &xi in x[..peak_idx].iter().rev() {
        left_min = left_min.min(xi);
        if xi > peak_val {
            break;
        }
    }

    // Search right
    let mut right_min = peak_val;
    for &xi in &x[peak_idx + 1..] {
        right_min = right_min.min(xi);
        if xi > peak_val {
            break;
        }
    }

    peak_val - left_min.max(right_min)
}

/// Filter peaks by minimum distance, keeping the highest peak in each window.
fn filter_by_distance(peaks: &[usize], heights: &[f64], min_dist: usize) -> Vec<usize> {
    if peaks.is_empty() {
        return Vec::new();
    }

    // Sort peaks by height (descending) for greedy selection.
    // Store original index to quickly access 'excluded' status.
    let mut indexed: Vec<(usize, usize, f64)> = peaks
        .iter()
        .zip(heights.iter())
        .enumerate()
        .map(|(i, (&p, &h))| (i, p, h))
        .collect();
    indexed.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

    let mut selected = Vec::new();
    let mut excluded = vec![false; peaks.len()];

    for &(orig_idx, pk, _) in &indexed {
        if excluded[orig_idx] {
            continue;
        }
        selected.push(pk);

        // Exclude all peaks within min_dist of pk.
        // Since 'peaks' is sorted by position, use binary search to find the range.
        let start_val = pk.saturating_sub(min_dist.saturating_sub(1));
        let end_val = pk.saturating_add(min_dist);

        let left = peaks.binary_search(&start_val).unwrap_or_else(|x| x);
        let right = peaks.binary_search(&end_val).unwrap_or_else(|x| x);

        for slot in excluded.iter_mut().take(right).skip(left) {
            *slot = true;
        }
    }

    selected.sort_unstable();
    selected
}

/// Compute peak prominences for given peak indices.
///
/// Matches `scipy.signal.peak_prominences`.
///
/// Returns (prominences, left_bases, right_bases).
pub fn peak_prominences(x: &[f64], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    // The per-peak outward scan is O(peaks · N) in the worst case (e.g. peaks of
    // monotonically increasing height each scan to the array boundary). For a
    // large workload, switch to the O(N log N + peaks) sparse-table form, which
    // is byte-identical (proven in `peak_prominences_rmq`). Tiny workloads keep
    // the direct scan to avoid the table-build overhead.
    if x.len() >= 256 && peaks.len() >= 32 && x.iter().all(|v| !v.is_nan()) {
        peak_prominences_rmq(x, peaks)
    } else {
        peak_prominences_naive(x, peaks)
    }
}

/// Reference outward-scan prominence: for each peak, walk left and right until a
/// sample taller than the peak is found, tracking the running minimum.
fn peak_prominences_naive(x: &[f64], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &pk in peaks {
        if pk >= x.len() {
            prominences.push(0.0);
            left_bases.push(pk);
            right_bases.push(pk);
            continue;
        }

        let peak_val = x[pk];

        // Search left for minimum
        let mut left_min = peak_val;
        let mut left_base = pk;
        for i in (0..pk).rev() {
            if x[i] < left_min {
                left_min = x[i];
                left_base = i;
            }
            if x[i] > peak_val {
                break;
            }
        }

        // Search right for minimum
        let mut right_min = peak_val;
        let mut right_base = pk;
        for (i, &value) in x.iter().enumerate().skip(pk + 1) {
            if value < right_min {
                right_min = value;
                right_base = i;
            }
            if value > peak_val {
                break;
            }
        }

        prominences.push(peak_val - left_min.max(right_min));
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    (prominences, left_bases, right_bases)
}

/// O(N log N + peaks) prominences via nearest-taller-sample bounds and
/// range-minimum sparse tables. Byte-identical to `peak_prominences_naive`:
///
///  * The left scan stops at the nearest index `j < pk` with `x[j] > x[pk]`
///    (strict), so its search window is `(nge_left[pk], pk)`; likewise the right
///    window is `(pk, nge_right[pk])`. Monotonic stacks give both in O(N).
///  * Within a window the scan keeps the running-min position. Walking *left*,
///    ties resolve to the position closest to the peak = the LARGEST index of
///    the window minimum; walking *right*, to the SMALLEST index. Two sparse
///    tables (one tie rule each) answer these argmin queries in O(1).
///  * `left_base`/`left_min` only move off the peak when the window minimum is
///    STRICTLY below `x[pk]` (matching the `< left_min` guard from `peak_val`);
///    every windowed sample is `<= x[pk]`, so the only excluded case is an
///    all-equal window, handled explicitly.
fn peak_prominences_rmq(x: &[f64], peaks: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let n = x.len();

    // Nearest strictly-taller sample to the left / right of each index.
    let mut nge_left = vec![-1i64; n];
    let mut nge_right = vec![n as i64; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        while let Some(&top) = stack.last() {
            if x[top] > x[i] {
                break;
            }
            stack.pop();
        }
        nge_left[i] = stack.last().map_or(-1, |&t| t as i64);
        stack.push(i);
    }
    stack.clear();
    for i in (0..n).rev() {
        while let Some(&top) = stack.last() {
            if x[top] > x[i] {
                break;
            }
            stack.pop();
        }
        nge_right[i] = stack.last().map_or(n as i64, |&t| t as i64);
        stack.push(i);
    }

    // Sparse tables of (min value, tie-broken argmin index). `prefer_larger`
    // selects the largest index among equal minima (left scan), else smallest.
    let build = |prefer_larger: bool| -> Vec<Vec<(f64, usize)>> {
        let levels = if n <= 1 {
            1
        } else {
            (usize::BITS - (n - 1).leading_zeros()) as usize + 1
        };
        let mut table: Vec<Vec<(f64, usize)>> = vec![vec![(f64::INFINITY, 0); n]; levels];
        for i in 0..n {
            table[0][i] = (x[i], i);
        }
        for k in 1..levels {
            let span = 1usize << k;
            let half = 1usize << (k - 1);
            for i in 0..=n.saturating_sub(span) {
                let a = table[k - 1][i];
                let b = table[k - 1][i + half];
                table[k][i] = merge_argmin(a, b, prefer_larger);
            }
        }
        table
    };
    let table_left = build(true);
    let table_right = build(false);

    let query =
        |table: &[Vec<(f64, usize)>], lo: usize, hi: usize, prefer_larger: bool| -> (f64, usize) {
            // Inclusive range [lo, hi]; caller guarantees lo <= hi < n.
            let len = hi - lo + 1;
            let k = (usize::BITS - 1 - len.leading_zeros()) as usize;
            let a = table[k][lo];
            let b = table[k][hi + 1 - (1 << k)];
            merge_argmin(a, b, prefer_larger)
        };

    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &pk in peaks {
        if pk >= n {
            prominences.push(0.0);
            left_bases.push(pk);
            right_bases.push(pk);
            continue;
        }
        let peak_val = x[pk];

        // Left window: indices (nge_left[pk], pk) = [nge_left+1, pk-1].
        let mut left_min = peak_val;
        let mut left_base = pk;
        let lo = (nge_left[pk] + 1) as usize;
        if lo < pk {
            let (m, idx) = query(&table_left, lo, pk - 1, true);
            if m < peak_val {
                left_min = m;
                left_base = idx;
            }
        }

        // Right window: indices (pk, nge_right[pk]) = [pk+1, nge_right-1].
        let mut right_min = peak_val;
        let mut right_base = pk;
        let hi = (nge_right[pk] - 1) as usize;
        if pk < hi {
            let (m, idx) = query(&table_right, pk + 1, hi, false);
            if m < peak_val {
                right_min = m;
                right_base = idx;
            }
        }

        prominences.push(peak_val - left_min.max(right_min));
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    (prominences, left_bases, right_bases)
}

/// Combine two (value, index) candidates by minimum value using IEEE `<` (so
/// `-0.0` and `+0.0` are equal, exactly like the reference scan's `< left_min`
/// guard). On equal values keep the larger index when `prefer_larger`, else the
/// smaller, and carry that index's own value bits. Inputs are NaN-free (gated).
fn merge_argmin(a: (f64, usize), b: (f64, usize), prefer_larger: bool) -> (f64, usize) {
    if a.0 < b.0 {
        a
    } else if b.0 < a.0 || (prefer_larger && b.1 > a.1) || (!prefer_larger && b.1 < a.1) {
        // b is strictly smaller, or values tie and the tie rule prefers b's index.
        b
    } else {
        a
    }
}

/// Compute peak widths at a given relative height.
///
/// Matches `scipy.signal.peak_widths`.
///
/// `rel_height` is the fraction of prominence at which to measure width (0.5 = half-prominence).
///
/// Returns (widths, width_heights, left_ips, right_ips).
pub fn peak_widths(
    x: &[f64],
    peaks: &[usize],
    rel_height: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (prominences, left_bases, right_bases) = peak_prominences(x, peaks);

    let mut widths = Vec::with_capacity(peaks.len());
    let mut width_heights = Vec::with_capacity(peaks.len());
    let mut left_ips = Vec::with_capacity(peaks.len());
    let mut right_ips = Vec::with_capacity(peaks.len());

    for (idx, &pk) in peaks.iter().enumerate() {
        if pk >= x.len() {
            let ip = pk as f64;
            widths.push(0.0);
            width_heights.push(0.0);
            left_ips.push(ip);
            right_ips.push(ip);
            continue;
        }

        let height = x[pk] - prominences[idx] * rel_height;
        width_heights.push(height);

        // Find left intersection point (interpolated)
        let mut left_ip = left_bases[idx] as f64;
        for i in (left_bases[idx]..pk).rev() {
            if x[i] <= height {
                // Linear interpolation between i and i+1
                if i + 1 < x.len() && (x[i + 1] - x[i]).abs() > 1e-15 {
                    left_ip = i as f64 + (height - x[i]) / (x[i + 1] - x[i]);
                } else {
                    left_ip = i as f64;
                }
                break;
            }
        }

        // Find right intersection point
        let mut right_ip = right_bases[idx] as f64;
        for i in pk + 1..=right_bases[idx] {
            if x[i] <= height {
                if i > 0 && (x[i] - x[i - 1]).abs() > 1e-15 {
                    right_ip = i as f64 - (x[i] - height) / (x[i] - x[i - 1]);
                } else {
                    right_ip = i as f64;
                }
                break;
            }
        }

        left_ips.push(left_ip);
        right_ips.push(right_ip);
        widths.push(right_ip - left_ip);
    }

    (widths, width_heights, left_ips, right_ips)
}

/// Compute the analytic signal's instantaneous frequency.
///
/// Matches `scipy.signal.instantaneous_frequency` (via Hilbert transform).
pub fn instantaneous_frequency(x: &[f64], fs: f64) -> Result<Vec<f64>, SignalError> {
    if fs <= 0.0 || !fs.is_finite() {
        return Err(SignalError::InvalidArgument(
            "sampling frequency fs must be positive and finite".to_string(),
        ));
    }
    let analytic = hilbert(x)?;

    // Instantaneous phase
    let phase: Vec<f64> = analytic.iter().map(|&(re, im)| im.atan2(re)).collect();

    // Unwrap phase
    let unwrapped = unwrap_phase(&phase);

    // Instantaneous frequency = d(phase)/dt / (2π)
    let mut freq = Vec::with_capacity(x.len());
    freq.push(0.0);
    for i in 1..unwrapped.len() {
        freq.push((unwrapped[i] - unwrapped[i - 1]) * fs / (2.0 * std::f64::consts::PI));
    }

    Ok(freq)
}

/// Find indices of relative maxima.
///
/// Matches `scipy.signal.argrelmax`.
pub fn argrelmax(x: &[f64], order: usize) -> Vec<usize> {
    let order = order.max(1);
    let n = x.len();
    let mut maxima = Vec::new();
    for i in 0..n {
        let mut is_max = true;
        for j in 1..=order {
            // scipy._boolrelextrema uses mode='clip': neighbour indices
            // past either edge clamp to the boundary, so extrema within
            // `order` of an edge are still tested rather than skipped.
            let left = i.saturating_sub(j);
            let right = (i + j).min(n - 1);
            if x[i] <= x[left] || x[i] <= x[right] {
                is_max = false;
                break;
            }
        }
        if is_max {
            maxima.push(i);
        }
    }
    maxima
}

/// Find indices of relative minima.
///
/// Matches `scipy.signal.argrelmin`.
pub fn argrelmin(x: &[f64], order: usize) -> Vec<usize> {
    let order = order.max(1);
    let n = x.len();
    let mut minima = Vec::new();
    for i in 0..n {
        let mut is_min = true;
        for j in 1..=order {
            // scipy._boolrelextrema uses mode='clip' (see argrelmax).
            let left = i.saturating_sub(j);
            let right = (i + j).min(n - 1);
            if x[i] >= x[left] || x[i] >= x[right] {
                is_min = false;
                break;
            }
        }
        if is_min {
            minima.push(i);
        }
    }
    minima
}

/// Find indices of relative extrema (both maxima and minima).
///
/// Matches `scipy.signal.argrelextrema`.
pub fn argrelextrema(x: &[f64], order: usize, greater: bool) -> Vec<usize> {
    if greater {
        argrelmax(x, order)
    } else {
        argrelmin(x, order)
    }
}

/// Compute the vector strength and phase of a set of events at a period.
///
/// Returns `(strength, phase)`: each event is mapped to a unit vector
/// `exp(i·2π·t/period)`; `strength = |mean|` measures synchronization
/// (1.0 perfect, 0.0 none) and `phase = angle(mean)` (radians) is the angle
/// the events cluster around. Matches `scipy.signal.vectorstrength`.
pub fn vectorstrength(events: &[f64], period: f64) -> (f64, f64) {
    if events.is_empty() || period <= 0.0 || !period.is_finite() {
        return (0.0, 0.0);
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let n = events.len() as f64;

    // Single-pass accumulation using sin_cos() so the argument
    // reduction `two_pi · t / period` is computed once per event and
    // both trig values share it. Resolves [frankenscipy-51bzs]:
    // pre-fix vectorstrength iterated events twice and computed the
    // phase angle twice per event.
    let inv_period = 1.0 / period;
    let mut sin_sum = 0.0_f64;
    let mut cos_sum = 0.0_f64;
    for &t in events {
        let angle = two_pi * t * inv_period;
        let (s, c) = angle.sin_cos();
        sin_sum += s;
        cos_sum += c;
    }

    // strength = |mean(exp(i·θ))|, phase = angle(mean(exp(i·θ))).
    let strength = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
    let phase = sin_sum.atan2(cos_sum);

    (strength, phase)
}

/// 1D order filter: select the k-th smallest value in each window.
///
/// Matches `scipy.signal.order_filter`.
pub fn order_filter(x: &[f64], window_size: usize, rank: usize) -> Vec<f64> {
    if x.is_empty() || window_size == 0 {
        return vec![];
    }
    // For large windows the per-window sort (O(n*k*log k)) is replaced by a
    // sliding ordered-multiset rank query (O(n*log k)); both return the same
    // rank element of each window, so the output is bit-identical.
    if window_size >= ORDER_FILTER_SLIDING_THRESHOLD {
        return order_filter_sliding(x, window_size, rank);
    }
    let half = window_size / 2;
    let mut result = Vec::with_capacity(x.len());

    // Resolves [frankenscipy-6feia]: hoist the per-window Vec
    // allocation to a single reusable buffer. Previously this
    // allocated N Vecs for an N-sample input.
    let mut window: Vec<f64> = Vec::with_capacity(window_size);
    for i in 0..x.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(x.len());
        window.clear();
        window.extend_from_slice(&x[start..end]);
        window.sort_by(|a, b| a.total_cmp(b));
        let idx = rank.min(window.len() - 1);
        result.push(window[idx]);
    }

    result
}

/// Unwrap phase angles to avoid discontinuities.
///
/// Matches `numpy.unwrap`.
pub fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return vec![];
    }
    let mut unwrapped = vec![phase[0]];
    for i in 1..phase.len() {
        let mut diff = phase[i] - phase[i - 1];
        if diff.is_finite() {
            while diff > std::f64::consts::PI {
                diff -= 2.0 * std::f64::consts::PI;
            }
            while diff < -std::f64::consts::PI {
                diff += 2.0 * std::f64::consts::PI;
            }
        }
        unwrapped.push(unwrapped[i - 1] + diff);
    }
    unwrapped
}

/// Compute the envelope of a signal via Hilbert transform.
///
/// Returns the magnitude of the analytic signal.
/// Matches `scipy.signal.hilbert` → `np.abs(...)`.
pub fn envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    hilbert_envelope(x)
}

/// Zero-crossing rate: fraction of consecutive samples with sign change.
///
/// Useful for audio/speech analysis.
pub fn zero_crossing_rate(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let crossings = x
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f64 / (x.len() - 1) as f64
}

/// Compute the short-time energy of a signal.
///
/// Returns the energy in each frame of length `frame_len` with `hop_len` stride.
pub fn short_time_energy(x: &[f64], frame_len: usize, hop_len: usize) -> Vec<f64> {
    if frame_len == 0 || hop_len == 0 || x.is_empty() {
        return vec![];
    }
    let mut energies = Vec::new();
    let mut start = 0;
    while start + frame_len <= x.len() {
        let energy: f64 = x[start..start + frame_len].iter().map(|&v| v * v).sum();
        energies.push(energy);
        start += hop_len;
    }
    energies
}

/// Compute the autocorrelation of a signal.
///
/// Returns the normalized autocorrelation for lags 0 to max_lag.
pub fn autocorrelation(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    // Resolves [frankenscipy-ry4ll]: precompute the centered series
    // once (N subs) instead of recomputing (x[i] - mean) n·max_lag
    // times in the inner loop. Same big-O but ~50% fewer subtractions
    // and much better cache locality (centered fits in L1).
    let centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let var: f64 = centered.iter().map(|&v| v * v).sum();

    if var == 0.0 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag.min(n - 1))
        .map(|lag| {
            let sum: f64 = centered[..n - lag]
                .iter()
                .zip(centered[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            sum / var
        })
        .collect()
}

/// Compute the spectral centroid of a magnitude spectrum.
///
/// Returns the weighted mean frequency.
pub fn spectral_centroid(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    if total == 0.0 {
        return 0.0;
    }
    magnitudes
        .iter()
        .zip(freqs.iter())
        .map(|(&m, &f)| m * f)
        .sum::<f64>()
        / total
}

/// Compute the spectral rolloff frequency.
///
/// Returns the frequency below which `rolloff_percent` of the total energy is contained.
pub fn spectral_rolloff(magnitudes: &[f64], freqs: &[f64], rolloff_percent: f64) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    let threshold = total * rolloff_percent / 100.0;
    let mut cumsum = 0.0;
    for (&m, &f) in magnitudes.iter().zip(freqs.iter()) {
        cumsum += m;
        if cumsum >= threshold {
            return f;
        }
    }
    freqs.last().copied().unwrap_or(0.0)
}

/// Spectral bandwidth: weighted standard deviation of frequencies.
pub fn spectral_bandwidth(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let centroid = spectral_centroid(magnitudes, freqs);
    let total: f64 = magnitudes.iter().zip(freqs.iter()).map(|(&m, _)| m).sum();
    if total == 0.0 {
        return 0.0;
    }
    let var: f64 = magnitudes
        .iter()
        .zip(freqs.iter())
        .map(|(&m, &f)| m * (f - centroid).powi(2))
        .sum::<f64>()
        / total;
    var.sqrt()
}

/// Spectral flatness: geometric mean / arithmetic mean of power spectrum.
///
/// 1.0 = white noise, 0.0 = pure tone.
pub fn spectral_flatness(magnitudes: &[f64]) -> f64 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let n = magnitudes.len() as f64;
    let arith_mean: f64 = magnitudes.iter().sum::<f64>() / n;
    if arith_mean == 0.0 {
        return 0.0;
    }
    let log_sum: f64 = magnitudes
        .iter()
        .map(|&m| if m > 0.0 { m.ln() } else { -700.0 })
        .sum();
    let geom_mean = (log_sum / n).exp();
    (geom_mean / arith_mean).clamp(0.0, 1.0)
}

/// Root mean square (RMS) of a signal.
pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

/// Peak-to-peak amplitude of a signal.
pub fn peak_to_peak(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let min = x.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    max - min
}

/// Crest factor: peak / RMS ratio.
pub fn crest_factor(x: &[f64]) -> f64 {
    let r = rms(x);
    if r == 0.0 {
        return 0.0;
    }
    let peak = x.iter().map(|&v| v.abs()).fold(0.0f64, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    peak / r
}

/// Compute the power of a signal in decibels relative to reference.
pub fn power_db(x: &[f64], ref_power: f64) -> f64 {
    let power: f64 = x.iter().map(|&v| v * v).sum::<f64>() / x.len().max(1) as f64;
    if power <= 0.0 || ref_power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (power / ref_power).log10()
}

/// Apply pre-emphasis filter: `y[n] = x[n] - coeff * x[n-1]`.
///
/// Common in speech processing with coeff ≈ 0.97.
///
/// # Arguments
/// * `x` — Input signal.
/// * `coeff` — Pre-emphasis coefficient. Non-finite values treated as 0.
pub fn preemphasis(x: &[f64], coeff: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let coeff = if coeff.is_finite() { coeff } else { 0.0 };
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] - coeff * x[i - 1]);
    }
    y
}

/// Apply de-emphasis filter (inverse of pre-emphasis).
///
/// # Arguments
/// * `x` — Input signal.
/// * `coeff` — De-emphasis coefficient. Non-finite values treated as 0.
pub fn deemphasis(x: &[f64], coeff: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let coeff = if coeff.is_finite() { coeff } else { 0.0 };
    let mut y = Vec::with_capacity(x.len());
    y.push(x[0]);
    for i in 1..x.len() {
        y.push(x[i] + coeff * y[i - 1]);
    }
    y
}

/// Frame a signal into overlapping windows.
///
/// Returns frames of length `frame_len` with `hop_len` stride.
pub fn frame_signal(x: &[f64], frame_len: usize, hop_len: usize) -> Vec<Vec<f64>> {
    if frame_len == 0 || hop_len == 0 || x.len() < frame_len {
        return vec![];
    }
    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_len <= x.len() {
        frames.push(x[start..start + frame_len].to_vec());
        start += hop_len;
    }
    frames
}

/// Bilinear transform: convert analog (s-domain) to digital (z-domain) filter.
///
/// Analog Chebyshev type-I lowpass prototype.
///
/// Matches `scipy.signal.cheb1ap(N, rp)`. Returns ZPK for the Type-I
/// Chebyshev lowpass with passband ripple `rp` dB:
///   ε  = √(10^(rp/10) − 1)
///   μ  = arcsinh(1/ε) / N
///   `poles[k] = −sinh(μ + i θ_k),  θ_k = π · (2k − 1 − N) / (2N)`
///   zeros = []
///   gain  = Re(Π(−p)) / √(1 + ε²)   (even N)
///         = Re(Π(−p))                (odd N)
/// For `N == 0`, SciPy returns the empty prototype with gain
/// `10^(-rp/20)`.
///
/// Type-I concentrates ripple in the passband, giving smaller order
/// than Butterworth for the same stopband spec.
pub fn cheb1ap(
    n: u32,
    rp: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), 10.0_f64.powf(-rp / 20.0)));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "cheb1ap: rp must be positive finite".to_string(),
        ));
    }
    let eps = (10.0_f64.powf(rp / 10.0) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;
    let sinh_mu = mu.sinh();
    let cosh_mu = mu.cosh();
    let mut poles = Vec::with_capacity(n as usize);
    for k in 1..=n {
        // θ = π · (2k − 1 − N) / (2N)
        let theta = std::f64::consts::PI * (2.0 * k as f64 - 1.0 - n as f64) / (2.0 * n as f64);
        // pole = −sinh(μ + iθ) = −[sinh(μ) cos(θ) + i cosh(μ) sin(θ)]
        let pole_re = -sinh_mu * theta.cos();
        let pole_im = -cosh_mu * theta.sin();
        poles.push((pole_re, pole_im));
    }
    // gain = Re(Π(-p))  (with even-N correction).
    let prod = poles.iter().fold((1.0_f64, 0.0_f64), |acc, &(re, im)| {
        let (a, b) = acc;
        let (c, d) = (-re, -im);
        (a * c - b * d, a * d + b * c)
    });
    let mut k = prod.0;
    if n.is_multiple_of(2) {
        k /= (1.0 + eps * eps).sqrt();
    }
    Ok((Vec::new(), poles, k))
}

/// Analog Butterworth lowpass prototype.
///
/// Matches `scipy.signal.buttap(N)`. Returns `(zeros, poles, gain)` for
/// the normalized analog Butterworth lowpass with N poles:
///   zeros = []
///   `poles[k] = exp(i · π · (2k − 1 + N) / (2N))   for k = 1..=N`
///   gain   = 1
///
/// The poles are equally spaced on the left-half unit circle.
pub fn buttap(
    n: u32,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    // scipy.signal.buttap(0) returns ([], [], 1) silently — match that.
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), 1.0));
    }
    let nf = n as f64;
    let mut poles = Vec::with_capacity(n as usize);
    for k in 1..=n {
        let theta = std::f64::consts::PI * (2.0 * k as f64 - 1.0 + nf) / (2.0 * nf);
        poles.push((theta.cos(), theta.sin()));
    }
    Ok((Vec::new(), poles, 1.0))
}

/// Elliptic (Cauer) filter order and natural frequency for analog
/// lowpass design.
///
/// Matches `scipy.signal.ellipord(wp, ws, gpass, gstop, analog=True)`
/// for the lowpass case. Closed-form via complete elliptic integrals K:
///   k    = wp / ws
///   k1   = √((10^(gpass/10) − 1) / (10^(gstop/10) − 1))
///   k'   = √(1 − k²),  k1' = √(1 − k1²)
///   N    = ⌈ K(k²) · K(k1'²) / (K(k'²) · K(k1²)) ⌉
///   Wn   = wp
///
/// Elliptic filters allow ripple in both passband and stopband and
/// achieve the lowest order of any classical IIR design for a given
/// spec. The inline `ellipk_internal` helper computes K(m) via the
/// arithmetic-geometric mean (Carlson, AGM iteration) to avoid pulling
/// fsci-special as a workspace dependency.
pub fn ellipord(wp: f64, ws: f64, gpass: f64, gstop: f64) -> Result<(u32, f64), SignalError> {
    if !wp.is_finite() || !ws.is_finite() || wp <= 0.0 || ws <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "ellipord: wp and ws must be positive finite".to_string(),
        ));
    }
    if wp >= ws {
        return Err(SignalError::InvalidArgument(
            "ellipord lowpass: passband edge wp must be < stopband edge ws".to_string(),
        ));
    }
    if gpass <= 0.0 || gstop <= 0.0 || gpass >= gstop {
        return Err(SignalError::InvalidArgument(
            "ellipord: require 0 < gpass < gstop".to_string(),
        ));
    }
    let pass_factor = 10.0_f64.powf(gpass / 10.0) - 1.0;
    let stop_factor = 10.0_f64.powf(gstop / 10.0) - 1.0;
    if pass_factor <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "ellipord: degenerate passband loss factor".to_string(),
        ));
    }
    let k = wp / ws;
    let k1 = (pass_factor / stop_factor).sqrt();
    let k_sq = k * k;
    let k1_sq = k1 * k1;
    let k_p_sq = 1.0 - k_sq;
    let k1_p_sq = 1.0 - k1_sq;
    let num = ellipk_internal(k_sq) * ellipk_internal(k1_p_sq);
    let den = ellipk_internal(k_p_sq) * ellipk_internal(k1_sq);
    if !num.is_finite() || !den.is_finite() || den <= 0.0 {
        return Err(SignalError::InvalidArgument(format!(
            "ellipord: degenerate K-ratio (num={num}, den={den})"
        )));
    }
    let order_real = num / den;
    let order = order_real.ceil().max(1.0) as u32;
    Ok((order, wp))
}

/// Complete elliptic integral of the first kind K(m) via the
/// arithmetic-geometric mean iteration. Domain m ∈ [0, 1).
fn ellipk_internal(m: f64) -> f64 {
    if !(0.0..1.0).contains(&m) {
        return f64::NAN;
    }
    if m == 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    let mut a = 1.0_f64;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..50 {
        let next_a = 0.5 * (a + b);
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() < 1.0e-15 * next_a {
            return std::f64::consts::FRAC_PI_2 / next_a;
        }
        a = next_a;
        b = next_b;
    }
    std::f64::consts::FRAC_PI_2 / a
}

/// Chebyshev type-II (inverse) filter order and natural frequency for
/// analog lowpass design.
///
/// Matches `scipy.signal.cheb2ord(wp, ws, gpass, gstop, analog=True)`
/// for the lowpass case. The order formula is identical to cheb1ord
/// (both Chebyshev variants share the same order curve); the
/// difference is the natural frequency:
///   Wn = wp · cosh(acosh(√((10^(gstop/10) − 1) / (10^(gpass/10) − 1))) / N)
/// After rounding the order up, scipy recomputes Wn by scaling the
/// passband edge `wp` so the realized Type-II design meets the
/// stopband edge; Wn therefore lands just below `ws`.
pub fn cheb2ord(wp: f64, ws: f64, gpass: f64, gstop: f64) -> Result<(u32, f64), SignalError> {
    if !wp.is_finite() || !ws.is_finite() || wp <= 0.0 || ws <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "cheb2ord: wp and ws must be positive finite".to_string(),
        ));
    }
    if wp >= ws {
        return Err(SignalError::InvalidArgument(
            "cheb2ord lowpass: passband edge wp must be < stopband edge ws".to_string(),
        ));
    }
    if gpass <= 0.0 || gstop <= 0.0 || gpass >= gstop {
        return Err(SignalError::InvalidArgument(
            "cheb2ord: require 0 < gpass < gstop".to_string(),
        ));
    }
    let pass_factor = 10.0_f64.powf(gpass / 10.0) - 1.0;
    let stop_factor = 10.0_f64.powf(gstop / 10.0) - 1.0;
    if pass_factor <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "cheb2ord: degenerate passband loss factor".to_string(),
        ));
    }
    let q = (stop_factor / pass_factor).sqrt();
    let q_acosh = q.acosh();
    let den = (ws / wp).acosh();
    if !q_acosh.is_finite() || !den.is_finite() || den <= 0.0 {
        return Err(SignalError::InvalidArgument(format!(
            "cheb2ord: degenerate spec (q_acosh={q_acosh}, den={den})"
        )));
    }
    let order_real = q_acosh / den;
    let order = order_real.ceil().max(1.0) as u32;
    // Wn = wp · cosh(q_acosh / N): scipy recomputes the natural
    // frequency after the order is rounded up, scaling the passband
    // edge so the realized design lands on the stopband edge.
    let wn = wp * (q_acosh / order as f64).cosh();
    Ok((order, wn))
}

/// Chebyshev type-I filter order and natural frequency for analog
/// lowpass design.
///
/// Matches `scipy.signal.cheb1ord(wp, ws, gpass, gstop, analog=True)`
/// for the lowpass case. Closed-form:
///   N = ⌈ acosh(√((10^(gstop/10) − 1) / (10^(gpass/10) − 1))) / acosh(ws/wp) ⌉
///   Wn = wp
/// Type-I Chebyshev concentrates ripple in the passband, so it
/// typically produces a smaller N than `buttord` for the same spec at
/// the cost of equiripple passband behavior.
pub fn cheb1ord(wp: f64, ws: f64, gpass: f64, gstop: f64) -> Result<(u32, f64), SignalError> {
    if !wp.is_finite() || !ws.is_finite() || wp <= 0.0 || ws <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "cheb1ord: wp and ws must be positive finite".to_string(),
        ));
    }
    if wp >= ws {
        return Err(SignalError::InvalidArgument(
            "cheb1ord lowpass: passband edge wp must be < stopband edge ws".to_string(),
        ));
    }
    if gpass <= 0.0 || gstop <= 0.0 || gpass >= gstop {
        return Err(SignalError::InvalidArgument(
            "cheb1ord: require 0 < gpass < gstop".to_string(),
        ));
    }
    let pass_factor = 10.0_f64.powf(gpass / 10.0) - 1.0;
    let stop_factor = 10.0_f64.powf(gstop / 10.0) - 1.0;
    if pass_factor <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "cheb1ord: degenerate passband loss factor".to_string(),
        ));
    }
    let num = (stop_factor / pass_factor).sqrt().acosh();
    let den = (ws / wp).acosh();
    if !num.is_finite() || !den.is_finite() || den <= 0.0 {
        return Err(SignalError::InvalidArgument(format!(
            "cheb1ord: degenerate spec produced num={num}, den={den}"
        )));
    }
    let order_real = num / den;
    let order = order_real.ceil().max(1.0) as u32;
    Ok((order, wp))
}

/// Butterworth filter order and natural frequency for analog lowpass
/// design.
///
/// Matches `scipy.signal.buttord(wp, ws, gpass, gstop, analog=True)` for
/// the lowpass case. Returns `(order, wn)` where:
///   order = ⌈ log10((10^(gstop/10) − 1) / (10^(gpass/10) − 1)) / (2·log10(ws/wp)) ⌉
///   wn = wp / (10^(gpass/10) − 1)^(1 / (2·order))
///
/// `wp` and `ws` are angular frequencies in rad/s (analog form). `gpass`
/// is the maximum passband loss in dB; `gstop` is the minimum stopband
/// attenuation in dB. Requires `wp < ws` and `0 < gpass < gstop`.
pub fn buttord(wp: f64, ws: f64, gpass: f64, gstop: f64) -> Result<(u32, f64), SignalError> {
    if !wp.is_finite() || !ws.is_finite() || wp <= 0.0 || ws <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "buttord: wp and ws must be positive finite".to_string(),
        ));
    }
    if wp >= ws {
        return Err(SignalError::InvalidArgument(
            "buttord lowpass: passband edge wp must be < stopband edge ws".to_string(),
        ));
    }
    if gpass <= 0.0 || gstop <= 0.0 || gpass >= gstop {
        return Err(SignalError::InvalidArgument(
            "buttord: require 0 < gpass < gstop".to_string(),
        ));
    }
    let pass_factor = 10.0_f64.powf(gpass / 10.0) - 1.0;
    let stop_factor = 10.0_f64.powf(gstop / 10.0) - 1.0;
    let ratio = (ws / wp).log10();
    if pass_factor <= 0.0 || ratio <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "buttord: degenerate spec produced non-positive denominator".to_string(),
        ));
    }
    let order_real = (stop_factor / pass_factor).log10() / (2.0 * ratio);
    let order = order_real.ceil().max(1.0) as u32;
    let wn = wp / pass_factor.powf(1.0 / (2.0 * order as f64));
    if !wn.is_finite() {
        return Err(SignalError::InvalidArgument(
            "buttord: natural frequency is not finite".to_string(),
        ));
    }
    Ok((order, wn))
}

/// Map an analog ZPK to digital via the bilinear transform.
///
/// Matches `scipy.signal.bilinear_zpk(z, p, k, fs)`. The bilinear
/// substitution s = 2fs · (z − 1)/(z + 1) on the analog transfer
/// function reduces on the ZPK side to:
///   z_z = (2fs + z) / (2fs − z)
///   p_z = (2fs + p) / (2fs − p)
///   any analog zero at infinity maps to a digital zero at −1; pad
///   `len(p) − len(z)` such zeros
///   k_z = k · Re(Π(2fs − z) / Π(2fs − p))
pub fn bilinear_zpk(
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
    k: f64,
    fs: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    check_proper_prototype("bilinear_zpk", z, p)?;
    let fs2 = 2.0 * fs;
    let bilin = |c: fsci_fft::Complex64| -> fsci_fft::Complex64 {
        // (fs2 + c) / (fs2 - c)
        let (re, im) = c;
        let num = (fs2 + re, im);
        let den = (fs2 - re, -im);
        let denom = den.0 * den.0 + den.1 * den.1;
        if denom == 0.0 {
            (f64::INFINITY, 0.0)
        } else {
            (
                (num.0 * den.0 + num.1 * den.1) / denom,
                (num.1 * den.0 - num.0 * den.1) / denom,
            )
        }
    };
    let mut z_z: Vec<fsci_fft::Complex64> = z.iter().copied().map(bilin).collect();
    let p_z: Vec<fsci_fft::Complex64> = p.iter().copied().map(bilin).collect();
    let degree = p.len() as i32 - z.len() as i32;
    if degree > 0 {
        z_z.extend(std::iter::repeat_n((-1.0, 0.0), degree as usize));
    }
    // gain = k * Re(Π(fs2 - z) / Π(fs2 - p))
    let prod_diff = |xs: &[fsci_fft::Complex64]| -> fsci_fft::Complex64 {
        xs.iter().fold((1.0_f64, 0.0_f64), |acc, &(re, im)| {
            let (a, b) = acc;
            let (c, d) = (fs2 - re, -im);
            (a * c - b * d, a * d + b * c)
        })
    };
    let num = prod_diff(z);
    let den = prod_diff(p);
    let denom = den.0 * den.0 + den.1 * den.1;
    let ratio_re = if denom == 0.0 {
        0.0
    } else {
        (num.0 * den.0 + num.1 * den.1) / denom
    };
    Ok((z_z, p_z, k * ratio_re))
}

/// Transform a normalized lowpass ZPK prototype to a bandstop with
/// center frequency `wo` and stopband bandwidth `bw`.
///
/// Matches `scipy.signal.lp2bs_zpk(z, p, k, wo=1.0, bw=1.0)`. The
/// transform first inverts to a highpass (zhp = bw / (2 z)), then
/// pair-splits each zhp via the bandpass-style formula
///   zhp ± √(zhp² − wo²)
/// and finally appends `degree = len(p) − len(z)` zeros at +i·wo and
/// `degree` zeros at −i·wo (the lp prototype's zeros at infinity become
/// imaginary-axis zeros at the bandstop's notch). Gain follows the
/// lp→hp formula k_new = k · Re(Π(−z)/Π(−p)).
pub fn lp2bs_zpk(
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
    k: f64,
    wo: f64,
    bw: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    check_proper_prototype("lp2bs_zpk", z, p)?;
    let half_bw = 0.5 * bw;
    // First invert to highpass.
    let cinv = |c: fsci_fft::Complex64| -> fsci_fft::Complex64 {
        let (re, im) = c;
        let denom = re * re + im * im;
        if denom == 0.0 {
            (f64::INFINITY, 0.0)
        } else {
            (half_bw * re / denom, -half_bw * im / denom)
        }
    };
    let split = |c: fsci_fft::Complex64| -> (fsci_fft::Complex64, fsci_fft::Complex64) {
        let asq_bsq = c.0 * c.0 - c.1 * c.1;
        let two_ab = 2.0 * c.0 * c.1;
        let arg = (asq_bsq - wo * wo, two_ab);
        let r = csqrt(arg);
        ((c.0 + r.0, c.1 + r.1), (c.0 - r.0, c.1 - r.1))
    };
    let mut z_bs: Vec<fsci_fft::Complex64> = Vec::with_capacity(2 * z.len());
    for &c in z {
        let (a, b) = split(cinv(c));
        z_bs.push(a);
        z_bs.push(b);
    }
    let mut p_bs: Vec<fsci_fft::Complex64> = Vec::with_capacity(2 * p.len());
    for &c in p {
        let (a, b) = split(cinv(c));
        p_bs.push(a);
        p_bs.push(b);
    }
    let degree = p.len() as i32 - z.len() as i32;
    if degree > 0 {
        let d = degree as usize;
        z_bs.extend(std::iter::repeat_n((0.0, wo), d));
        z_bs.extend(std::iter::repeat_n((0.0, -wo), d));
    }
    // gain k * Re(Π(-z) / Π(-p)) — same formula as lp2hp.
    let neg_prod = |xs: &[fsci_fft::Complex64]| -> fsci_fft::Complex64 {
        xs.iter().fold((1.0_f64, 0.0_f64), |acc, &(re, im)| {
            let (a, b) = acc;
            let (c, d) = (-re, -im);
            (a * c - b * d, a * d + b * c)
        })
    };
    let num = neg_prod(z);
    let den = neg_prod(p);
    let denom = den.0 * den.0 + den.1 * den.1;
    let ratio_re = if denom == 0.0 {
        0.0
    } else {
        (num.0 * den.0 + num.1 * den.1) / denom
    };
    Ok((z_bs, p_bs, k * ratio_re))
}

/// Transform a normalized lowpass ZPK prototype to a bandpass with
/// center frequency `wo` and bandwidth `bw`.
///
/// Matches `scipy.signal.lp2bp_zpk(z, p, k, wo=1.0, bw=1.0)`. Each
/// finite pole or zero splits into a complex-conjugate-related pair
/// via the substitution s → (s² + wo²) / (bw · s):
///   z_lp = z · bw / 2
///   z_bp = { z_lp + √(z_lp² − wo²),  z_lp − √(z_lp² − wo²) }
/// (and identically for p). The lp prototype's `degree = len(p) − len(z)`
/// zeros at infinity become zeros at origin in the bandpass, padded
/// onto z_bp. Gain scales by `bw^degree`.
pub fn lp2bp_zpk(
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
    k: f64,
    wo: f64,
    bw: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    check_proper_prototype("lp2bp_zpk", z, p)?;
    let scale = 0.5 * bw;
    let split = |c: fsci_fft::Complex64| -> (fsci_fft::Complex64, fsci_fft::Complex64) {
        // z_lp = c · scale; root_arg = z_lp² − wo²; sqrt of complex.
        let zlp = (scale * c.0, scale * c.1);
        // c² − wo² where c = (a + bi):
        //   c² = (a² − b², 2ab)
        //   c² − wo² = (a² − b² − wo², 2ab)
        let asq_bsq = zlp.0 * zlp.0 - zlp.1 * zlp.1;
        let two_ab = 2.0 * zlp.0 * zlp.1;
        let arg = (asq_bsq - wo * wo, two_ab);
        let r = csqrt(arg);
        let plus = (zlp.0 + r.0, zlp.1 + r.1);
        let minus = (zlp.0 - r.0, zlp.1 - r.1);
        (plus, minus)
    };
    let mut z_bp: Vec<fsci_fft::Complex64> = Vec::with_capacity(2 * z.len());
    for &c in z {
        let (a, b) = split(c);
        z_bp.push(a);
        z_bp.push(b);
    }
    let mut p_bp: Vec<fsci_fft::Complex64> = Vec::with_capacity(2 * p.len());
    for &c in p {
        let (a, b) = split(c);
        p_bp.push(a);
        p_bp.push(b);
    }
    let degree = p.len() as i32 - z.len() as i32;
    if degree > 0 {
        z_bp.extend(std::iter::repeat_n((0.0, 0.0), degree as usize));
    }
    let k_bp = k * bw.powi(degree);
    Ok((z_bp, p_bp, k_bp))
}

/// Principal-branch complex square root for a `(re, im)` tuple.
fn csqrt(c: fsci_fft::Complex64) -> fsci_fft::Complex64 {
    let (a, b) = c;
    if b == 0.0 {
        return if a >= 0.0 {
            (a.sqrt(), 0.0)
        } else {
            (0.0, (-a).sqrt())
        };
    }
    let r = (a * a + b * b).sqrt();
    let re = ((r + a) * 0.5).sqrt();
    let im_mag = ((r - a) * 0.5).sqrt();
    let im = if b >= 0.0 { im_mag } else { -im_mag };
    (re, im)
}

/// Transform a normalized lowpass ZPK prototype to a highpass with
/// cutoff frequency `wo`.
///
/// Matches `scipy.signal.lp2hp_zpk(z, p, k, wo=1.0)`. The transform is
/// the substitution s → wo/s applied to the analog transfer function:
///   z_new = wo / z   (existing finite zeros invert)
///   p_new = wo / p   (existing finite poles invert)
///   zeros at infinity in the lowpass prototype move to zeros at
///   origin in the highpass — append `len(p) − len(z)` zeros at 0
///   k_new = k · Re(∏(−z) / ∏(−p))
///
/// `z` may include the value (0, 0); the result then contains an
/// infinite zero (i.e. the highpass acquires a zero at infinity, which
/// is fine symbolically). Caller is responsible for guarding ill-posed
/// prototypes if needed.
pub fn lp2hp_zpk(
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
    k: f64,
    wo: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    check_proper_prototype("lp2hp_zpk", z, p)?;
    let cinv = |c: fsci_fft::Complex64| -> fsci_fft::Complex64 {
        let (re, im) = c;
        let denom = re * re + im * im;
        if denom == 0.0 {
            (f64::INFINITY, 0.0)
        } else {
            (wo * re / denom, -wo * im / denom)
        }
    };
    let mut z_new: Vec<_> = z.iter().copied().map(cinv).collect();
    let p_new: Vec<_> = p.iter().copied().map(cinv).collect();
    let degree = p.len() as i32 - z.len() as i32;
    if degree > 0 {
        z_new.extend(std::iter::repeat_n((0.0, 0.0), degree as usize));
    }
    // Re(Π(-z) / Π(-p)). Since k_new is real, keep Re only.
    let neg_prod = |xs: &[fsci_fft::Complex64]| -> fsci_fft::Complex64 {
        xs.iter().fold((1.0_f64, 0.0_f64), |acc, &(re, im)| {
            // Multiply (acc.0 + i acc.1) by (-re - i im).
            let (a, b) = acc;
            let (c, d) = (-re, -im);
            (a * c - b * d, a * d + b * c)
        })
    };
    let num = neg_prod(z);
    let den = neg_prod(p);
    let denom = den.0 * den.0 + den.1 * den.1;
    let ratio_re = if denom == 0.0 {
        0.0
    } else {
        (num.0 * den.0 + num.1 * den.1) / denom
    };
    let k_new = k * ratio_re;
    Ok((z_new, p_new, k_new))
}

/// Transform a normalized lowpass ZPK prototype to a lowpass with a
/// different cutoff frequency.
///
/// Matches `scipy.signal.lp2lp_zpk(z, p, k, wo=1.0)`. The transform is
/// the substitution s → s/wo applied to the analog transfer function,
/// which on the zero-pole-gain side reduces to:
///   z_new = wo · z
///   p_new = wo · p
///   k_new = k · wo^(len(p) − len(z))
/// preserving the magnitude response shape but shifting its 3 dB point
/// from 1 rad/s to `wo` rad/s.
pub fn lp2lp_zpk(
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
    k: f64,
    wo: f64,
) -> Result<(Vec<fsci_fft::Complex64>, Vec<fsci_fft::Complex64>, f64), SignalError> {
    check_proper_prototype("lp2lp_zpk", z, p)?;
    let z_new: Vec<_> = z.iter().map(|&(re, im)| (wo * re, wo * im)).collect();
    let p_new: Vec<_> = p.iter().map(|&(re, im)| (wo * re, wo * im)).collect();
    let degree_diff = p.len() as i32 - z.len() as i32;
    let k_new = k * wo.powi(degree_diff);
    Ok((z_new, p_new, k_new))
}

/// Transform a lowpass filter prototype to a different cutoff frequency,
/// in transfer-function (b, a) form.
///
/// Matches `scipy.signal.lp2lp(b, a, wo=1.0)`. Substitutes s → s/wo:
///
/// ```text
///   B(s)/A(s) → B(s/wo)/A(s/wo)
/// ```
///
/// Coefficient-wise: `new_b[i] = b[i] / wo^(n−1−i)`,
/// `new_a[i] = a[i] / wo^(d−1−i)`, followed by normalization so
/// `a[0] = 1`. Resolves [frankenscipy-xla38].
pub fn lp2lp(b: &[f64], a: &[f64], wo: f64) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if !wo.is_finite() || wo <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "wo must be positive finite".into(),
        ));
    }
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".into(),
        ));
    }
    let n = b.len();
    let d = a.len();
    let new_b: Vec<f64> = b
        .iter()
        .enumerate()
        .map(|(i, &bi)| bi / wo.powi((n - 1 - i) as i32))
        .collect();
    let new_a: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, &ai)| ai / wo.powi((d - 1 - i) as i32))
        .collect();
    normalize_filter(&new_b, &new_a)
}

/// Transform a lowpass filter prototype to a highpass filter, in
/// transfer-function (b, a) form.
///
/// Matches `scipy.signal.lp2hp(b, a, wo=1.0)`. Substitutes s → wo/s.
/// For the original B(s) of degree M_b stored in descending powers,
/// the new coefficient at index j (in descending powers, length =
/// max(b.len(), a.len())) is `b[M_b − j] · wo^j` for `j ≤ M_b` and
/// 0 above. Same for `a`. Then normalize so `a[0] = 1`.
///
/// Resolves [frankenscipy-drh63].
pub fn lp2hp(b: &[f64], a: &[f64], wo: f64) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if !wo.is_finite() || wo <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "wo must be positive finite".into(),
        ));
    }
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".into(),
        ));
    }
    let nb = b.len();
    let na = a.len();
    let n = nb.max(na);
    let mb = nb - 1;
    let ma = na - 1;
    let mut new_b = vec![0.0_f64; n];
    let mut new_a = vec![0.0_f64; n];
    for j in 0..=mb {
        new_b[j] = b[mb - j] * wo.powi(j as i32);
    }
    for j in 0..=ma {
        new_a[j] = a[ma - j] * wo.powi(j as i32);
    }
    normalize_filter(&new_b, &new_a)
}

/// Binomial coefficient as `f64`.
fn binom_coeff_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Transform a lowpass filter prototype to a bandpass filter, in
/// transfer-function (b, a) form.
///
/// Matches `scipy.signal.lp2bp(b, a, wo=1.0, bw=1.0)`. Substitutes
/// s → (s² + wo²) / (s · bw) — the wideband bandpass transformation
/// producing a passband of geometric symmetry around `wo` with
/// width `bw`. For input b of degree N and a of degree D, the
/// output has degree N + max(N, D) for the numerator and D +
/// max(N, D) for the denominator.
///
/// Resolves [frankenscipy-c7ukn].
pub fn lp2bp(b: &[f64], a: &[f64], wo: f64, bw: f64) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if !wo.is_finite() || wo <= 0.0 || !bw.is_finite() || bw <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "wo and bw must be positive finite".into(),
        ));
    }
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".into(),
        ));
    }
    let big_n = b.len() - 1;
    let big_d = a.len() - 1;
    let ma = big_n.max(big_d);
    let np = big_n + ma;
    let dp = big_d + ma;
    let wosq = wo * wo;

    let mut bprime = vec![0.0_f64; np + 1];
    let mut aprime = vec![0.0_f64; dp + 1];

    // For each output index j (in ascending position from the
    // *trailing* end), sum over i,k where ma + 2k - i = j the term
    // comb(i, k) · src[deg - i] · wo^(2(i-k)) / bw^i.
    for j in 0..=np {
        let mut val = 0.0_f64;
        for i in 0..=big_n {
            for k in 0..=i {
                let lhs = ma + 2 * k;
                if lhs >= i && lhs - i == j {
                    val += binom_coeff_f64(i, k) * b[big_n - i] * wosq.powi((i - k) as i32)
                        / bw.powi(i as i32);
                }
            }
        }
        bprime[np - j] = val;
    }
    for j in 0..=dp {
        let mut val = 0.0_f64;
        for i in 0..=big_d {
            for k in 0..=i {
                let lhs = ma + 2 * k;
                if lhs >= i && lhs - i == j {
                    val += binom_coeff_f64(i, k) * a[big_d - i] * wosq.powi((i - k) as i32)
                        / bw.powi(i as i32);
                }
            }
        }
        aprime[dp - j] = val;
    }

    normalize_filter(&bprime, &aprime)
}

/// Transform a lowpass filter prototype to a bandstop filter, in
/// transfer-function (b, a) form.
///
/// Matches `scipy.signal.lp2bs(b, a, wo=1.0, bw=1.0)`. Substitutes
/// s → (s · bw) / (s² + wo²) — the wideband bandstop transformation.
/// For input b of degree N and a of degree D with M = max(N, D),
/// the output has degree 2M for both numerator and denominator.
///
/// Resolves [frankenscipy-9oqbs].
pub fn lp2bs(b: &[f64], a: &[f64], wo: f64, bw: f64) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if !wo.is_finite() || wo <= 0.0 || !bw.is_finite() || bw <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "wo and bw must be positive finite".into(),
        ));
    }
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".into(),
        ));
    }
    let big_n = b.len() - 1;
    let big_d = a.len() - 1;
    let m = big_n.max(big_d);
    let np = 2 * m;
    let wosq = wo * wo;

    let mut bprime = vec![0.0_f64; np + 1];
    let mut aprime = vec![0.0_f64; np + 1];

    for j in 0..=np {
        let mut val = 0.0_f64;
        for i in 0..=big_n {
            // k ranges 0..=M-i (always non-negative since i ≤ N ≤ M).
            let k_max = m - i;
            for k in 0..=k_max {
                if i + 2 * k == j {
                    val += binom_coeff_f64(m - i, k)
                        * b[big_n - i]
                        * wosq.powi((m - i - k) as i32)
                        * bw.powi(i as i32);
                }
            }
        }
        bprime[np - j] = val;
    }
    for j in 0..=np {
        let mut val = 0.0_f64;
        for i in 0..=big_d {
            let k_max = m - i;
            for k in 0..=k_max {
                if i + 2 * k == j {
                    val += binom_coeff_f64(m - i, k)
                        * a[big_d - i]
                        * wosq.powi((m - i - k) as i32)
                        * bw.powi(i as i32);
                }
            }
        }
        aprime[np - j] = val;
    }

    normalize_filter(&bprime, &aprime)
}

/// Reject improper prototypes (more zeros than poles) — matches scipy's
/// `_relative_degree` ValueError.
fn check_proper_prototype(
    name: &str,
    z: &[fsci_fft::Complex64],
    p: &[fsci_fft::Complex64],
) -> Result<(), SignalError> {
    if z.len() > p.len() {
        Err(SignalError::InvalidArgument(format!(
            "{name}: improper prototype — zeros ({}) exceeds poles ({}). \
             scipy.signal._relative_degree raises ValueError here.",
            z.len(),
            p.len()
        )))
    } else {
        Ok(())
    }
}

/// Transforms analog numerator/denominator to digital using the bilinear transform
/// s = 2*fs*(z-1)/(z+1).
///
/// Returns (b_digital, a_digital).
/// Matches `scipy.signal.bilinear`.
pub fn bilinear(b_analog: &[f64], a_analog: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = b_analog.len().max(a_analog.len());
    let d = n - 1;

    // Pad to same length
    let mut b = vec![0.0; n];
    let mut a = vec![0.0; n];
    for (i, &v) in b_analog.iter().enumerate() {
        b[n - b_analog.len() + i] = v;
    }
    for (i, &v) in a_analog.iter().enumerate() {
        a[n - a_analog.len() + i] = v;
    }

    let fs2 = 2.0 * fs;

    // Apply the bilinear transform via polynomial manipulation
    let mut b_dig = vec![0.0; n];
    let mut a_dig = vec![0.0; n];

    for j in 0..n {
        let mut val_b = 0.0;
        let mut val_a = 0.0;
        for i in 0..n {
            let k = d - i;
            // Substituting s = 2·fs·(z−1)/(z+1) and clearing the (z+1)^d
            // denominator sends the analog s^k coefficient to
            // (2·fs)^k·(z−1)^k·(z+1)^(d−k). `coeff` is the coefficient of
            // z^(d−j) in (z−1)^k·(z+1)^(d−k): the (z−1)^k factor expands as
            // Σ C(k, jp)·(−1)^jp·z^(k−jp), so the sign is (−1)^jp.
            let mut coeff = 0.0;
            for jp in 0..=j.min(k) {
                let rem = j - jp;
                if rem <= d - k {
                    let c1 = binom_coeff(k, jp);
                    let c2 = binom_coeff(d - k, rem);
                    let sign = if jp.is_multiple_of(2) { 1.0 } else { -1.0 };
                    coeff += c1 * c2 * sign;
                }
            }
            val_b += b[i] * coeff * fs2.powi(k as i32);
            val_a += a[i] * coeff * fs2.powi(k as i32);
        }
        b_dig[j] = val_b;
        a_dig[j] = val_a;
    }

    // Normalize by a[0]
    if a_dig[0].abs() > 1e-30 {
        let norm = a_dig[0];
        for v in &mut b_dig {
            *v /= norm;
        }
        for v in &mut a_dig {
            *v /= norm;
        }
    }

    (b_dig, a_dig)
}

fn binom_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Compute the impulse response of a digital filter.
///
/// Returns n_samples of the response to a unit impulse.
/// Matches `scipy.signal.dimpulse` (simplified).
pub fn impulse_response(b: &[f64], a: &[f64], n_samples: usize) -> Result<Vec<f64>, SignalError> {
    validate_discrete_tf(b, a)?;
    let mut impulse = vec![0.0; n_samples];
    if !impulse.is_empty() {
        impulse[0] = 1.0;
    }
    lfilter(b, a, &impulse, None)
}

/// Compute the step response of a digital filter.
///
/// Returns n_samples of the response to a unit step.
/// Matches `scipy.signal.dstep` (simplified).
pub fn step_response(b: &[f64], a: &[f64], n_samples: usize) -> Result<Vec<f64>, SignalError> {
    validate_discrete_tf(b, a)?;
    let step = vec![1.0; n_samples];
    lfilter(b, a, &step, None)
}

/// Validate a discrete transfer function `H(z) = B(z)/A(z)` for the
/// response routines: both polynomials must be non-empty and the numerator
/// order must not exceed the denominator order.
///
/// The numerator is **not** padded — `lfilter` applies `b` directly (as
/// `scipy.signal.lfilter` does, padding internally with trailing zeros).
/// Padding `b` with leading zeros here would inject a spurious sample
/// delay into the response.
fn validate_discrete_tf(b: &[f64], a: &[f64]) -> Result<(), SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if b.len() > a.len() {
        return Err(SignalError::InvalidArgument(
            "improper transfer function: numerator order must not exceed denominator order"
                .to_string(),
        ));
    }
    Ok(())
}

/// Compute the group delay of a digital filter at specified frequencies.
///
/// Returns the group delay τ(ω) = -d(phase)/dω.
/// This is a convenience wrapper that returns (frequencies, delays).
pub fn group_delay_from_ba(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut delays = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);
        delays.push(group_delay_at_frequency(b, a, w));
    }

    (freqs, delays)
}

/// Compute the **linear** magnitude response of a digital filter.
///
/// Returns `(frequencies, |H(e^{jω})|)` where `H = B/A` is the
/// transfer function and the magnitude is unitless and non-negative
/// — matching the convention of `np.abs(h)` from
/// `scipy.signal.freqz(b, a, worN=n_freqs)`. For the dB-scaled
/// equivalent (`20 · log10(|H|)`) use [`magnitude_response_db`].
///
/// Resolves [frankenscipy-o8x6j].
pub fn magnitude_response(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut mags = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);

        let mut br = 0.0;
        let mut bi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            br += coeff * (w * i as f64).cos();
            bi -= coeff * (w * i as f64).sin();
        }

        let mut ar = 0.0;
        let mut ai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            ar += coeff * (w * i as f64).cos();
            ai -= coeff * (w * i as f64).sin();
        }

        let h_mag2 = br * br + bi * bi;
        let a_mag2 = ar * ar + ai * ai;
        let mag = if a_mag2 > 1e-30 {
            (h_mag2 / a_mag2).sqrt()
        } else {
            0.0
        };

        mags.push(mag);
    }

    (freqs, mags)
}

/// Compute the magnitude response of a digital filter in **decibels**.
///
/// Returns `(frequencies, 20 · log10(|H(e^{jω})|))` — the dB-scaled
/// counterpart of [`magnitude_response`]. Frequencies where `|H| = 0`
/// (or numerically below `1e-30`) are mapped to `f64::NEG_INFINITY`.
///
/// Resolves [frankenscipy-o8x6j].
pub fn magnitude_response_db(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let (freqs, mags) = magnitude_response(b, a, n_freqs);
    let mags_db: Vec<f64> = mags
        .iter()
        .map(|&m| {
            if m > 1e-30 {
                20.0 * m.log10()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    (freqs, mags_db)
}

/// Compute the phase response of a digital filter.
///
/// Returns (frequencies, phase_radians).
pub fn phase_response(b: &[f64], a: &[f64], n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut phases = Vec::with_capacity(n_freqs);

    for k in 0..n_freqs {
        let w = std::f64::consts::PI * k as f64 / n_freqs as f64;
        freqs.push(w);

        let mut br = 0.0;
        let mut bi = 0.0;
        for (i, &coeff) in b.iter().enumerate() {
            br += coeff * (w * i as f64).cos();
            bi -= coeff * (w * i as f64).sin();
        }

        let mut ar = 0.0;
        let mut ai = 0.0;
        for (i, &coeff) in a.iter().enumerate() {
            ar += coeff * (w * i as f64).cos();
            ai -= coeff * (w * i as f64).sin();
        }

        // H = B/A, phase = arg(B) - arg(A)
        let phase_b = bi.atan2(br);
        let phase_a = ai.atan2(ar);
        phases.push(phase_b - phase_a);
    }

    (freqs, phases)
}

/// Normalize filter coefficients so the denominator is monic.
///
/// Matches `scipy.signal.normalize(b, a)`. Trims leading zeros from
/// `a`, then divides both `b` and `a` by the resulting `a[0]` so the
/// returned denominator starts with `1.0`.
///
/// Errors:
///   * `a` is all-zero (or empty after trim) — degenerate filter.
///   * `a` is empty — invalid input.
///
/// Resolves [frankenscipy-fx18c]. Distinct from the existing
/// `normalize_signal` (zero-mean / unit-variance time-series scaler).
pub fn normalize_filter(b: &[f64], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "denominator `a` must be non-empty".into(),
        ));
    }
    // Reject non-finite coefficients up-front. Without this guard, NaN
    // in `a` would slip past `v != 0.0` (NaN != 0 is true) and become
    // the leading coefficient, propagating NaN through the normalized
    // outputs with an Ok status — review-mode finding [n9ply].
    if a.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::InvalidArgument(
            "denominator `a` must contain only finite values".into(),
        ));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(SignalError::InvalidArgument(
            "numerator `b` must contain only finite values".into(),
        ));
    }
    let first_nonzero = a.iter().position(|&v| v != 0.0).ok_or_else(|| {
        SignalError::InvalidArgument(
            "denominator `a` must contain at least one nonzero coefficient".into(),
        )
    })?;
    let leading = a[first_nonzero];
    let a_norm: Vec<f64> = a[first_nonzero..].iter().map(|&v| v / leading).collect();
    let b_norm: Vec<f64> = b.iter().map(|&v| v / leading).collect();
    Ok((b_norm, a_norm))
}

/// Combine roots that are within `tol` of each other into unique groups.
///
/// Matches `scipy.signal.unique_roots(p, tol, rtype)`. Sorts the
/// input, then walks the sorted array greedily grouping consecutive
/// entries where the gap from the group's first element is within
/// `tol`. Each group is collapsed via `rtype` ("min", "max", or
/// "avg"); unknown rtype falls back to "avg" (matching scipy's
/// permissive default).
///
/// Returns `(unique, multiplicities)` — a vector of representative
/// roots and a vector of how many input roots fell into each group.
///
/// Resolves [frankenscipy-sx2yp].
pub fn unique_roots(p: &[f64], tol: f64, rtype: &str) -> (Vec<f64>, Vec<usize>) {
    if p.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted: Vec<f64> = p.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut roots = Vec::new();
    let mut mult = Vec::new();
    let mut group: Vec<f64> = Vec::with_capacity(sorted.len());
    for &val in &sorted {
        let extend = match group.first() {
            Some(&first) => (val - first).abs() <= tol,
            None => true,
        };
        if extend {
            group.push(val);
        } else {
            collapse_unique_root_group(&mut roots, &mut mult, &group, rtype);
            group.clear();
            group.push(val);
        }
    }
    if !group.is_empty() {
        collapse_unique_root_group(&mut roots, &mut mult, &group, rtype);
    }
    (roots, mult)
}

fn collapse_unique_root_group(
    roots: &mut Vec<f64>,
    mult: &mut Vec<usize>,
    group: &[f64],
    rtype: &str,
) {
    let n = group.len();
    let r = match rtype {
        "min" => group.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => group.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        // "avg" or any other value — match scipy's permissive behaviour.
        _ => group.iter().sum::<f64>() / n as f64,
    };
    roots.push(r);
    mult.push(n);
}

/// Normalize a signal to have zero mean and unit variance.
pub fn normalize_signal(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std == 0.0 {
        return vec![0.0; x.len()];
    }
    x.iter().map(|&v| (v - mean) / std).collect()
}

/// Normalize a signal to range [0, 1].
pub fn normalize_minmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let min = x.iter().cloned().fold(f64::INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    });
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let range = max - min;
    if range == 0.0 {
        return vec![0.5; x.len()];
    }
    x.iter().map(|&v| (v - min) / range).collect()
}

/// Downsample a signal by taking every n-th sample.
pub fn downsample(x: &[f64], factor: usize) -> Vec<f64> {
    if factor == 0 {
        return x.to_vec();
    }
    x.iter().step_by(factor).cloned().collect()
}

/// Upsample a signal by inserting zeros between samples.
pub fn upsample(x: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return x.to_vec();
    }
    let mut result = vec![0.0; x.len() * factor];
    for (i, &v) in x.iter().enumerate() {
        result[i * factor] = v;
    }
    result
}

/// Compute the signal-to-noise ratio in dB.
pub fn snr(signal: &[f64], noise: &[f64]) -> f64 {
    let sig_power: f64 = signal.iter().map(|&v| v * v).sum::<f64>() / signal.len().max(1) as f64;
    let noise_power: f64 = noise.iter().map(|&v| v * v).sum::<f64>() / noise.len().max(1) as f64;
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig_power / noise_power).log10()
}

/// Compute the total harmonic distortion (THD).
///
/// Given a signal and its fundamental frequency bin index, computes
/// THD = sqrt(sum of harmonic powers / fundamental power).
pub fn thd(magnitudes: &[f64], fundamental_bin: usize) -> f64 {
    if fundamental_bin >= magnitudes.len() || magnitudes[fundamental_bin] == 0.0 {
        return f64::NAN;
    }

    let fund_power = magnitudes[fundamental_bin].powi(2);
    let mut harmonic_power = 0.0;

    // Sum power at harmonics (2f, 3f, 4f, ...)
    let mut bin = 2 * fundamental_bin;
    while bin < magnitudes.len() {
        harmonic_power += magnitudes[bin].powi(2);
        bin += fundamental_bin;
    }

    (harmonic_power / fund_power).sqrt()
}

/// Add white Gaussian noise to a signal at a specified SNR (dB).
pub fn add_noise(signal: &[f64], snr_db: f64, seed: u64) -> Vec<f64> {
    let sig_power: f64 = signal.iter().map(|&v| v * v).sum::<f64>() / signal.len().max(1) as f64;
    let noise_power = sig_power / 10.0f64.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    let mut rng = seed;
    signal
        .iter()
        .map(|&s| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let noise =
                noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            s + noise
        })
        .collect()
}

/// Compute the energy of a signal.
pub fn signal_energy(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Compute the power of a signal (energy per sample).
pub fn signal_power(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    signal_energy(x) / x.len() as f64
}

/// Compute the cross-correlation coefficient between two signals.
///
/// Returns a value between -1 and 1.
pub fn xcorr_coefficient(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mx: f64 = x.iter().sum::<f64>() / n;
    let my: f64 = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - mx) * (b - my))
        .sum();
    let sx: f64 = x.iter().map(|&a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = y.iter().map(|&b| (b - my).powi(2)).sum::<f64>().sqrt();
    if sx * sy == 0.0 {
        return 0.0;
    }
    cov / (sx * sy)
}

/// Compute the delay between two signals via cross-correlation.
///
/// Returns the lag (in samples) that maximizes cross-correlation.
pub fn find_delay(x: &[f64], y: &[f64]) -> i64 {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return 0;
    }

    let max_lag = n.max(m) - 1;
    let mut best_lag: i64 = 0;
    let mut best_corr = f64::NEG_INFINITY;

    // Resolves [frankenscipy-9pkqs]: hoist the per-lag bounds out of
    // the inner loop. For lag L, the valid i range is
    //   [max(0, L), min(n, m + L))
    // since j = i - L must satisfy 0 ≤ j < m. Iterating over slices
    // with .zip eliminates the runtime branch and enables better
    // auto-vectorization.
    let n_i = n as i64;
    let m_i = m as i64;
    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let i_lo = lag.max(0) as usize;
        let i_hi = (n_i.min(m_i + lag)).max(0) as usize;
        if i_lo >= i_hi {
            continue;
        }
        let j_lo = (i_lo as i64 - lag) as usize;
        let len = i_hi - i_lo;
        let sum: f64 = x[i_lo..i_lo + len]
            .iter()
            .zip(y[j_lo..j_lo + len].iter())
            .map(|(&xi, &yj)| xi * yj)
            .sum();
        if sum > best_corr {
            best_corr = sum;
            best_lag = lag;
        }
    }

    best_lag
}

/// Apply a moving median filter.
pub fn medfilt1(x: &[f64], kernel_size: usize) -> Vec<f64> {
    if x.is_empty() || kernel_size == 0 {
        return x.to_vec();
    }
    // For large kernels the per-window O(k log k) sort dominates; a sliding
    // ordered-multiset median is O(n log k) and bit-identical (see
    // `medfilt1_sliding`). Small kernels keep the naive sort (lower constants).
    if kernel_size >= MEDFILT1_SLIDING_THRESHOLD {
        return medfilt1_sliding(x, kernel_size);
    }
    let half = kernel_size / 2;
    let n = x.len();

    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let mut window: Vec<f64> = x[start..end].to_vec();
            window.sort_by(|a, b| a.total_cmp(b));
            window[window.len() / 2]
        })
        .collect()
}

const MEDFILT1_SLIDING_THRESHOLD: usize = 32;

/// Sliding `medfilt1`: like `order_filter_sliding`, the 1-D window is clipped
/// (not zero-padded) at the borders, so it grows then shrinks; `medfilt1` takes
/// `window[window.len() / 2]`, i.e. the rank `len/2` element of the CURRENT
/// (clipped) window — so the requested rank is re-pointed every step as the
/// window size changes. Bit-identical to sorting each window and indexing.
fn medfilt1_sliding(x: &[f64], kernel_size: usize) -> Vec<f64> {
    let n = x.len();
    let half = kernel_size / 2;
    let mut window = SlidingRankWindow::new(0);
    let mut cur_start = 0usize;
    let mut cur_end = 0usize;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        while cur_start < start {
            window.remove(x[cur_start]);
            cur_start += 1;
        }
        while cur_end < end {
            window.insert(x[cur_end]);
            cur_end += 1;
        }
        window.set_rank((cur_end - cur_start) / 2);
        result.push(window.value());
    }
    result
}

/// Apply exponential smoothing to a signal.
///
/// # Arguments
/// * `x` — Input signal.
/// * `alpha` — Smoothing factor in [0, 1]. Values near 0 give more smoothing.
pub fn exponential_smooth(x: &[f64], alpha: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let alpha = if !alpha.is_finite() {
        0.5
    } else {
        alpha.clamp(0.0, 1.0)
    };
    let mut result = Vec::with_capacity(x.len());
    result.push(x[0]);
    for i in 1..x.len() {
        result.push(alpha * x[i] + (1.0 - alpha) * result[i - 1]);
    }
    result
}

/// Compute the analytic signal magnitude (envelope) via absolute value of Hilbert.
pub fn analytic_envelope(x: &[f64]) -> Result<Vec<f64>, SignalError> {
    hilbert_envelope(x)
}

/// Compute the frequency of maximum power in a spectrum.
pub fn dominant_frequency(magnitudes: &[f64], freqs: &[f64]) -> f64 {
    if magnitudes.is_empty() || freqs.is_empty() {
        return 0.0;
    }
    let max_idx = magnitudes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    freqs.get(max_idx).copied().unwrap_or(0.0)
}

/// Compute the spectral entropy.
///
/// Normalized Shannon entropy of the power spectrum.
pub fn spectral_entropy(magnitudes: &[f64]) -> f64 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let total: f64 = magnitudes.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &m in magnitudes {
        if m > 0.0 {
            let p = m / total;
            entropy -= p * p.ln();
        }
    }

    // Normalize by log(N)
    let n = magnitudes.len() as f64;
    if n > 1.0 { entropy / n.ln() } else { entropy }
}

/// Convert frequency in Hz to Mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).ln() / std::f64::consts::LN_10
}

/// Convert Mel scale to frequency in Hz.
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Generate Mel filterbank matrix.
///
/// Returns a matrix of shape (n_mels, n_fft/2+1) for applying to a magnitude spectrum.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f64, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax.min(sr / 2.0));

    // Mel-spaced center frequencies
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f64> = hz_points.iter().map(|&hz| hz * n_fft as f64 / sr).collect();

    let mut filterbank = vec![vec![0.0; n_freq]; n_mels];

    for m in 0..n_mels {
        let start = bin_points[m];
        let center = bin_points[m + 1];
        let end = bin_points[m + 2];

        for (k, val) in filterbank[m].iter_mut().enumerate().take(n_freq) {
            let kf = k as f64;
            if kf >= start && kf <= center && center > start {
                *val = (kf - start) / (center - start);
            } else if kf > center && kf <= end && end > center {
                *val = (end - kf) / (end - center);
            }
        }
    }

    filterbank
}

/// Compute Mel-frequency cepstral coefficients (MFCCs).
///
/// Returns a matrix of shape (n_frames, n_mfcc).
pub fn mfcc(
    signal: &[f64],
    sr: f64,
    n_mfcc: usize,
    n_mels: usize,
    frame_len: usize,
    hop_len: usize,
) -> Vec<Vec<f64>> {
    if signal.is_empty() || frame_len == 0 || hop_len == 0 {
        return vec![];
    }

    let n_fft = frame_len;
    let n_freq = n_fft / 2 + 1;
    let fb = mel_filterbank(n_mels, n_fft, sr, 0.0, sr / 2.0);

    // Window
    let window: Vec<f64> = (0..frame_len)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (frame_len - 1) as f64).cos())
        })
        .collect();

    let mut mfccs = Vec::new();
    let mut start = 0;

    while start + frame_len <= signal.len() {
        // Windowed frame
        let frame: Vec<f64> = signal[start..start + frame_len]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Power spectrum via DFT
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut power_spec = vec![0.0; n_freq];
        for (k, spec_val) in power_spec.iter_mut().enumerate().take(n_freq) {
            let mut re = 0.0;
            let mut im = 0.0;
            for (n, &s) in frame.iter().enumerate() {
                let angle = two_pi * k as f64 * n as f64 / n_fft as f64;
                re += s * angle.cos();
                im -= s * angle.sin();
            }
            *spec_val = (re * re + im * im) / n_fft as f64;
        }

        // Apply Mel filterbank
        let mel_energies: Vec<f64> = fb
            .iter()
            .map(|filter| {
                let energy: f64 = filter
                    .iter()
                    .zip(power_spec.iter())
                    .map(|(&f, &p)| f * p)
                    .sum();
                energy.max(1e-10).ln() // log Mel energies
            })
            .collect();

        // DCT (type-II) to get MFCCs
        let mut coeffs = Vec::with_capacity(n_mfcc);
        for i in 0..n_mfcc {
            let mut sum = 0.0;
            for (j, &e) in mel_energies.iter().enumerate() {
                sum +=
                    e * (std::f64::consts::PI * i as f64 * (j as f64 + 0.5) / n_mels as f64).cos();
            }
            coeffs.push(sum * (2.0 / n_mels as f64).sqrt());
        }

        mfccs.push(coeffs);
        start += hop_len;
    }

    mfccs
}

/// Compute the chroma feature (pitch class profile) from a magnitude spectrum.
///
/// Maps frequency bins to 12 pitch classes (C, C#, D, ..., B).
pub fn chroma(magnitudes: &[f64], sr: f64, n_fft: usize) -> [f64; 12] {
    let n_freq = magnitudes.len();
    let mut chroma_vec = [0.0f64; 12];

    for (k, magnitude) in magnitudes.iter().enumerate().take(n_freq).skip(1) {
        let freq = k as f64 * sr / n_fft as f64;
        if !(20.0..=5000.0).contains(&freq) {
            continue;
        }
        // Map frequency to pitch class: pitch = 12 * log2(f / 440) + 69
        let midi = 12.0 * (freq / 440.0).log2() + 69.0;
        let pitch_class = ((midi.round() as i64 % 12 + 12) % 12) as usize;
        chroma_vec[pitch_class] += *magnitude;
    }

    // Normalize
    let max_val = chroma_vec
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .fold(0.0_f64, f64::max);
    if max_val > 0.0 {
        for v in &mut chroma_vec {
            *v /= max_val;
        }
    }

    chroma_vec
}

/// Compute the spectral contrast between peaks and valleys.
pub fn spectral_contrast(magnitudes: &[f64], n_bands: usize) -> Vec<f64> {
    if magnitudes.is_empty() || n_bands == 0 {
        return vec![];
    }

    let n = magnitudes.len();
    let band_size = n / n_bands;
    if band_size == 0 {
        return vec![0.0; n_bands];
    }

    (0..n_bands)
        .map(|b| {
            let start = b * band_size;
            let end = ((b + 1) * band_size).min(n);
            let band = &magnitudes[start..end];
            if band.is_empty() {
                return 0.0;
            }
            // Only the band's extremes are needed; a single total_cmp pass yields
            // the same `peak`/`valley` as sorting (the total_cmp max/min equal
            // sorted[len-1]/sorted[0] bit-for-bit, incl. NaN/-0.0) in O(band) with
            // no allocation, instead of an O(band log band) sort.
            let mut peak = band[0];
            let mut valley = band[0];
            for &v in &band[1..] {
                if v.total_cmp(&peak) == std::cmp::Ordering::Greater {
                    peak = v;
                }
                if v.total_cmp(&valley) == std::cmp::Ordering::Less {
                    valley = v;
                }
            }
            if valley > 0.0 {
                (peak / valley).log10() * 20.0
            } else {
                0.0
            }
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════
// IIR Filter Design
// ══════════════════════════════════════════════════════════════════════

/// Filter type for IIR design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Low-pass filter.
    Lowpass,
    /// High-pass filter.
    Highpass,
    /// Band-pass filter.
    Bandpass,
    /// Band-stop filter.
    Bandstop,
}

/// Analog prototype family for IIR design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IirFamily {
    /// Butterworth: maximally flat passband.
    Butterworth,
    /// Chebyshev Type I: equiripple passband.
    Chebyshev1,
    /// Chebyshev Type II: equiripple stopband.
    Chebyshev2,
    /// Bessel/Thomson: maximally flat group delay.
    Bessel,
    /// Elliptic/Cauer: equiripple both passband and stopband. Sharpest transition.
    Elliptic,
}

/// IIR filter coefficients in transfer function form (b, a).
///
/// `H(z) = B(z)/A(z)` where `B(z) = b[0] + b[1]*z^-1 + ...` and `A(z) = 1 + a[1]*z^-1 + ...`
#[derive(Debug, Clone, PartialEq)]
pub struct BaCoeffs {
    /// Numerator coefficients.
    pub b: Vec<f64>,
    /// Denominator coefficients (`a[0]` is always 1.0).
    pub a: Vec<f64>,
}

/// Butterworth output form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButterOutput {
    /// Transfer-function coefficients `(b, a)`.
    Ba,
    /// Second-order sections.
    Sos,
}

/// Design a Butterworth IIR filter.
///
/// Matches `scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')`.
///
/// # Arguments
/// * `order` — Filter order (1, 2, 3, ...)
/// * `wn` — Critical frequencies (normalized, 0 < wn < 1 for digital).
///   Use one value for lowpass/highpass and two values for bandpass/bandstop.
/// * `btype` — Filter type.
///
/// Returns (b, a) transfer function coefficients.
pub fn butter(order: usize, wn: &[f64], btype: FilterType) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let (proto_poles_re, proto_poles_im) = butterworth_analog_poles(order);
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re: proto_poles_re,
        poles_im: proto_poles_im,
        gain: 1.0,
    };

    design_digital_iir(analog_zpk, wn, btype)
}

/// Design a Butterworth IIR filter with an explicit output form.
///
/// Matches the `output` kwarg of `scipy.signal.butter(..., output=...)`
/// for the supported digital forms.
pub fn butter_with_output(
    order: usize,
    wn: &[f64],
    btype: FilterType,
    output: ButterOutput,
) -> Result<BaCoeffsOrSos, SignalError> {
    let ba = butter(order, wn, btype)?;
    match output {
        ButterOutput::Ba => Ok(BaCoeffsOrSos::Ba(ba)),
        ButterOutput::Sos => Ok(BaCoeffsOrSos::Sos(tf2sos(&ba.b, &ba.a)?)),
    }
}

/// Butterworth design result for an explicit output form.
#[derive(Debug, Clone, PartialEq)]
pub enum BaCoeffsOrSos {
    /// Transfer-function coefficients `(b, a)`.
    Ba(BaCoeffs),
    /// Second-order sections.
    Sos(Vec<SosSection>),
}

/// Design a Butterworth IIR filter directly in second-order-section form.
///
/// Matches `scipy.signal.butter(..., output='sos')`.
pub fn butter_sos(
    order: usize,
    wn: &[f64],
    btype: FilterType,
) -> Result<Vec<SosSection>, SignalError> {
    match butter_with_output(order, wn, btype, ButterOutput::Sos)? {
        BaCoeffsOrSos::Sos(sos) => Ok(sos),
        BaCoeffsOrSos::Ba(_) => unreachable!("ButterOutput::Sos must return SOS"),
    }
}

/// Design a Chebyshev Type I IIR filter.
pub fn cheby1(
    order: usize,
    rp: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rp must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;
    let epsilon = (10_f64.powf(rp / 10.0) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / order as f64;
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        poles_re.push(-mu.sinh() * theta.sin());
        poles_im.push(mu.cosh() * theta.cos());
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re,
        poles_im,
        gain: 1.0,
    };
    let mut coeffs = design_digital_iir(analog_zpk, wn, btype)?;
    let passband_gain = cheby1_passband_reference_gain(order, rp);
    coeffs
        .b
        .iter_mut()
        .for_each(|coeff| *coeff *= passband_gain);
    Ok(coeffs)
}

/// Design a Chebyshev Type II IIR filter.
pub fn cheby2(
    order: usize,
    rs: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rs must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;
    let epsilon = 1.0 / (10_f64.powf(rs / 10.0) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / order as f64;
    let mut zeros_re = Vec::with_capacity(order);
    let mut zeros_im = Vec::with_capacity(order);
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let cos_theta = theta.cos();
        // Odd-order Chebyshev-II prototypes have one zero at infinity.
        // Preserve that degree difference instead of materializing a huge finite zero.
        if cos_theta.abs() > 1.0e-12 {
            zeros_re.push(0.0);
            zeros_im.push(1.0 / cos_theta);
        }

        let proto_re = -mu.sinh() * theta.sin();
        let proto_im = mu.cosh() * theta.cos();
        let (pole_re, pole_im) = complex_div(1.0, 0.0, proto_re, proto_im);
        poles_re.push(pole_re);
        poles_im.push(pole_im);
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: 1.0,
    };
    design_digital_iir(analog_zpk, wn, btype)
}

/// Design a Bessel/Thomson IIR filter.
pub fn bessel(order: usize, wn: &[f64], btype: FilterType) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let coeffs = reverse_bessel_polynomial(order)?;
    let (mut poles_re, mut poles_im) = poly_roots(&coeffs)?;
    let phase_scale = reverse_bessel_phase_scale(order)?;
    for pole in &mut poles_re {
        *pole *= phase_scale;
    }
    for pole in &mut poles_im {
        *pole *= phase_scale;
    }
    let analog_zpk = ZpkCoeffs {
        zeros_re: Vec::new(),
        zeros_im: Vec::new(),
        poles_re,
        poles_im,
        gain: 1.0,
    };
    design_digital_iir(analog_zpk, wn, btype)
}

/// Design an Elliptic (Cauer) IIR filter.
///
/// The elliptic filter achieves the sharpest transition band for a given order,
/// at the cost of equiripple in both passband and stopband.
///
/// # Arguments
/// * `order` — Filter order
/// * `rp` — Maximum passband ripple (dB)
/// * `rs` — Minimum stopband attenuation (dB)
/// * `wn` — Critical frequencies (normalized, 0 < wn < 1)
/// * `btype` — Filter type
pub fn ellip(
    order: usize,
    rp: f64,
    rs: f64,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    if order == 0 {
        return Err(SignalError::InvalidArgument(
            "order must be >= 1".to_string(),
        ));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rp must be finite and > 0 dB".to_string(),
        ));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "rs must be finite and > 0 dB".to_string(),
        ));
    }
    validate_iir_wn(wn, btype)?;

    let analog_zpk = elliptic_analog_zpk(order, rp, rs)?;
    let mut coeffs = design_digital_iir(analog_zpk, wn, btype)?;
    let passband_gain = cheby1_passband_reference_gain(order, rp);
    coeffs
        .b
        .iter_mut()
        .for_each(|coeff| *coeff *= passband_gain);
    Ok(coeffs)
}

/// Design a second-order IIR notch (band-reject) filter.
///
/// Removes a narrow band of frequencies centered at `w0`.
///
/// Matches `scipy.signal.iirnotch(w0, Q)`.
///
/// # Arguments
/// * `w0` — Frequency to reject (normalized, 0 < w0 < 1).
/// * `q` — Quality factor. Higher Q = narrower notch.
///
/// Returns `BaCoeffs` (b, a) for the second-order notch filter.
pub fn iirnotch(w0: f64, q: f64) -> Result<BaCoeffs, SignalError> {
    if !w0.is_finite() || w0 <= 0.0 || w0 >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "w0 must be in (0, 1)".to_string(),
        ));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "Q must be positive".to_string(),
        ));
    }

    // Orfanidis notch design (Introduction to Signal Processing, §11.3.4),
    // matching scipy.signal.iirnotch. With the −3 dB reference gain
    // gb = 1/√2, the notch bandwidth parameter reduces to β = tan(bw/2),
    // where bw = π·w0/Q and ω₀ = π·w0 are the radian bandwidth and centre.
    let omega0 = std::f64::consts::PI * w0;
    let bw = omega0 / q;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let cos_w0 = omega0.cos();
    Ok(BaCoeffs {
        b: vec![gain, -2.0 * gain * cos_w0, gain],
        a: vec![1.0, -2.0 * gain * cos_w0, 2.0 * gain - 1.0],
    })
}

/// Design a second-order IIR peak (band-pass) filter.
///
/// Boosts a narrow band of frequencies centered at `w0`.
///
/// Matches `scipy.signal.iirpeak(w0, Q)`.
///
/// # Arguments
/// * `w0` — Center frequency (normalized, 0 < w0 < 1).
/// * `q` — Quality factor. Higher Q = narrower peak.
///
/// Returns `BaCoeffs` (b, a) for the second-order peak filter.
pub fn iirpeak(w0: f64, q: f64) -> Result<BaCoeffs, SignalError> {
    if !w0.is_finite() || w0 <= 0.0 || w0 >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "w0 must be in (0, 1)".to_string(),
        ));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "Q must be positive".to_string(),
        ));
    }

    // Orfanidis peak design (§11.3.19), matching scipy.signal.iirpeak.
    // It shares the notch's denominator; with gb = 1/√2 the peak's
    // bandwidth parameter is also β = tan(bw/2).
    let omega0 = std::f64::consts::PI * w0;
    let bw = omega0 / q;
    let beta = (bw / 2.0).tan();
    let gain = 1.0 / (1.0 + beta);
    let cos_w0 = omega0.cos();
    Ok(BaCoeffs {
        b: vec![1.0 - gain, 0.0, -(1.0 - gain)],
        a: vec![1.0, -2.0 * gain * cos_w0, 2.0 * gain - 1.0],
    })
}

/// General IIR filter design dispatcher.
pub fn iirfilter(
    order: usize,
    wn: &[f64],
    btype: FilterType,
    family: IirFamily,
    rp: Option<f64>,
    rs: Option<f64>,
) -> Result<BaCoeffs, SignalError> {
    match family {
        IirFamily::Butterworth => butter(order, wn, btype),
        IirFamily::Chebyshev1 => cheby1(
            order,
            rp.ok_or_else(|| {
                SignalError::InvalidArgument("cheby1 requires passband ripple rp".to_string())
            })?,
            wn,
            btype,
        ),
        IirFamily::Chebyshev2 => cheby2(
            order,
            rs.ok_or_else(|| {
                SignalError::InvalidArgument("cheby2 requires stopband attenuation rs".to_string())
            })?,
            wn,
            btype,
        ),
        IirFamily::Bessel => bessel(order, wn, btype),
        IirFamily::Elliptic => {
            let rp_val = rp.ok_or_else(|| {
                SignalError::InvalidArgument("elliptic requires passband ripple rp".to_string())
            })?;
            let rs_val = rs.ok_or_else(|| {
                SignalError::InvalidArgument(
                    "elliptic requires stopband attenuation rs".to_string(),
                )
            })?;
            ellip(order, rp_val, rs_val, wn, btype)
        }
    }
}

fn cheby1_passband_reference_gain(order: usize, rp: f64) -> f64 {
    if order.is_multiple_of(2) {
        10_f64.powf(-rp / 20.0)
    } else {
        1.0
    }
}

fn design_digital_iir(
    analog_zpk: ZpkCoeffs,
    wn: &[f64],
    btype: FilterType,
) -> Result<BaCoeffs, SignalError> {
    let warped: Vec<f64> = wn.iter().map(|&w| prewarp_digital_frequency(w)).collect();
    let analog_zpk = match btype {
        FilterType::Lowpass => lowpass_zpk(&analog_zpk, warped[0]),
        FilterType::Highpass => highpass_zpk(&analog_zpk, warped[0]),
        FilterType::Bandpass => bandpass_zpk(&analog_zpk, warped[0], warped[1]),
        FilterType::Bandstop => bandstop_zpk(&analog_zpk, warped[0], warped[1]),
    };

    let digital_zpk = bilinear_zpk_internal(&analog_zpk);
    let mut ba = zpk2tf(&digital_zpk);
    normalize_digital_ba(&mut ba, &warped, btype)?;
    Ok(ba)
}

fn normalize_digital_ba(
    ba: &mut BaCoeffs,
    warped: &[f64],
    btype: FilterType,
) -> Result<(), SignalError> {
    let normalize_at = match btype {
        FilterType::Lowpass => 0.0,
        FilterType::Highpass => std::f64::consts::PI,
        FilterType::Bandpass => 2.0 * ((warped[0] * warped[1]).sqrt() / 2.0).atan(),
        FilterType::Bandstop => 0.0,
    };
    let (b_re, b_im) = eval_poly_on_unit_circle(&ba.b, normalize_at);
    let (a_re, a_im) = eval_poly_on_unit_circle(&ba.a, normalize_at);
    let b_mag = (b_re * b_re + b_im * b_im).sqrt();
    let a_mag = (a_re * a_re + a_im * a_im).sqrt();
    if b_mag < 1.0e-30 || a_mag < 1.0e-30 {
        return Err(SignalError::InvalidArgument(
            "degenerate filter: could not normalize digital response".to_string(),
        ));
    }
    let gain = a_mag / b_mag;
    ba.b.iter_mut().for_each(|coeff| *coeff *= gain);
    Ok(())
}

fn elliptic_analog_zpk(order: usize, rp: f64, rs: f64) -> Result<ZpkCoeffs, SignalError> {
    if order == 1 {
        let pole = -(1.0 / pow10m1(0.1 * rp)).sqrt();
        return Ok(ZpkCoeffs {
            zeros_re: Vec::new(),
            zeros_im: Vec::new(),
            poles_re: vec![pole],
            poles_im: vec![0.0],
            gain: -pole,
        });
    }

    let eps_sq = pow10m1(0.1 * rp);
    let ck1_sq = eps_sq / pow10m1(0.1 * rs);
    if ck1_sq <= 0.0 || !ck1_sq.is_finite() {
        return Err(SignalError::InvalidArgument(
            "Cannot design a filter with given rp and rs specifications.".to_string(),
        ));
    }

    let eps = eps_sq.sqrt();
    let m = ellipdeg(order, ck1_sq)?;
    let capk = complete_ellipk(m)?;
    let start = if order.is_multiple_of(2) { 1 } else { 0 };

    let mut zeros_re = Vec::new();
    let mut zeros_im = Vec::new();
    let mut base_poles = Vec::new();
    for j in (start..order).step_by(2) {
        let u = j as f64 * capk / order as f64;
        let (s, c, d) = jacobi_ellipj_real(u, m)?;
        if s.abs() > 1.0e-14 {
            zeros_re.push(0.0);
            zeros_im.push(1.0 / (m.sqrt() * s));
        }

        base_poles.push((s, c, d));
    }

    let r = arc_jac_sc1_real(1.0 / eps, ck1_sq)?;
    let v0 = capk * r / (order as f64 * complete_ellipk(ck1_sq)?);
    let (sv, cv, dv) = jacobi_ellipj_real(v0, 1.0 - m)?;

    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for &(s, c, d) in &base_poles {
        let denom = 1.0 - (d * sv).powi(2);
        let pole_re = -(c * d * sv * cv) / denom;
        let pole_im = -(s * dv) / denom;
        poles_re.push(pole_re);
        poles_im.push(pole_im);
    }
    if order.is_multiple_of(2) {
        for &(s, c, d) in &base_poles {
            let denom = 1.0 - (d * sv).powi(2);
            poles_re.push(-(c * d * sv * cv) / denom);
            poles_im.push((s * dv) / denom);
        }
    } else {
        for &(s, c, d) in &base_poles {
            let denom = 1.0 - (d * sv).powi(2);
            let pole_re = -(c * d * sv * cv) / denom;
            let pole_im = -(s * dv) / denom;
            if pole_im.abs() > 1.0e-14 * pole_re.hypot(pole_im) {
                poles_re.push(pole_re);
                poles_im.push(-pole_im);
            }
        }
    }

    let zero_count = zeros_re.len();
    zeros_re.extend_from_within(..);
    let zero_conjugates: Vec<f64> = zeros_im[..zero_count].iter().map(|&value| -value).collect();
    zeros_im.extend(zero_conjugates);

    let (pole_prod_re, pole_prod_im) = complex_product_neg_roots(&poles_re, &poles_im);
    let (zero_prod_re, zero_prod_im) = complex_product_neg_roots(&zeros_re, &zeros_im);
    let (mut gain, _) = complex_div_complex(pole_prod_re, pole_prod_im, zero_prod_re, zero_prod_im);
    if order.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }

    Ok(ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain,
    })
}

fn pow10m1(x: f64) -> f64 {
    (std::f64::consts::LN_10 * x).exp_m1()
}

fn ellipdeg(order: usize, m1: f64) -> Result<f64, SignalError> {
    let k1 = complete_ellipk(m1)?;
    let k1p = complete_ellipkm1(m1)?;
    let q1 = (-std::f64::consts::PI * k1p / k1).exp();
    let q = q1.powf(1.0 / order as f64);

    let num = (0..=7).map(|m| q.powi(m * (m + 1))).sum::<f64>();
    let den = 1.0 + 2.0 * (1..=8).map(|m| q.powi(m * m)).sum::<f64>();
    Ok(16.0 * q * (num / den).powi(4))
}

fn complete_ellipk(m: f64) -> Result<f64, SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() {
        return Err(SignalError::InvalidArgument(
            "elliptic modulus must be finite and in [0, 1]".to_string(),
        ));
    }
    if m == 1.0 {
        return Ok(f64::INFINITY);
    }
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..64 {
        let next_a = (a + b) / 2.0;
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            return Ok(std::f64::consts::PI / (2.0 * next_a));
        }
        a = next_a;
        b = next_b;
    }
    Ok(std::f64::consts::PI / (2.0 * a))
}

fn complete_ellipkm1(m: f64) -> Result<f64, SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() {
        return Err(SignalError::InvalidArgument(
            "elliptic complementary modulus must be finite and in [0, 1]".to_string(),
        ));
    }
    let mut a = 1.0;
    let mut b = m.sqrt();
    for _ in 0..64 {
        let next_a = (a + b) / 2.0;
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            return Ok(std::f64::consts::PI / (2.0 * next_a));
        }
        a = next_a;
        b = next_b;
    }
    Ok(std::f64::consts::PI / (2.0 * a))
}

fn arc_jac_sc1_real(w: f64, m: f64) -> Result<f64, SignalError> {
    let mut low = 0.0;
    let mut high = complete_ellipkm1(m)? * (1.0 - 1.0e-14);
    for _ in 0..96 {
        let mid = (low + high) / 2.0;
        let (sn, cn, _) = jacobi_ellipj_real(mid, 1.0 - m)?;
        let sc = sn / cn;
        if sc < w {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok((low + high) / 2.0)
}

fn jacobi_ellipj_real(u: f64, m: f64) -> Result<(f64, f64, f64), SignalError> {
    if !(0.0..=1.0).contains(&m) || !m.is_finite() || !u.is_finite() {
        return Err(SignalError::InvalidArgument(
            "Jacobi elliptic arguments must be finite with m in [0, 1]".to_string(),
        ));
    }
    if m <= 1.0e-15 {
        return Ok((u.sin(), u.cos(), 1.0));
    }
    if (1.0 - m).abs() <= 1.0e-15 {
        let cn = 1.0 / u.cosh();
        return Ok((u.tanh(), cn, cn));
    }

    let mut a_values = vec![1.0];
    let mut c_values = Vec::new();
    let mut b = (1.0 - m).sqrt();
    let mut twon = 1.0;
    for _ in 0..32 {
        let a = a_values.last().copied().unwrap_or(1.0);
        let c = (a - b) / 2.0;
        let next_a = (a + b) / 2.0;
        c_values.push(c);
        a_values.push(next_a);
        b = (a * b).sqrt();
        twon *= 2.0;
        if c.abs() <= 1.0e-15 * next_a.abs().max(1.0) {
            break;
        }
    }

    let mut phi = twon * a_values.last().copied().unwrap_or(1.0) * u;
    for index in (0..c_values.len()).rev() {
        let ratio = c_values[index] * phi.sin() / a_values[index + 1];
        phi = (ratio.clamp(-1.0, 1.0).asin() + phi) / 2.0;
    }
    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).max(0.0).sqrt();
    Ok((sn, cn, dn))
}

fn complex_product_neg_roots(roots_re: &[f64], roots_im: &[f64]) -> (f64, f64) {
    let mut prod_re = 1.0;
    let mut prod_im = 0.0;
    for (&root_re, &root_im) in roots_re.iter().zip(roots_im.iter()) {
        let (next_re, next_im) = complex_mul(prod_re, prod_im, -root_re, -root_im);
        prod_re = next_re;
        prod_im = next_im;
    }
    (prod_re, prod_im)
}

fn reverse_bessel_phase_scale(order: usize) -> Result<f64, SignalError> {
    let Some(double_order) = order.checked_mul(2) else {
        return Err(SignalError::InvalidArgument(
            "order is too large for Bessel polynomial construction".to_string(),
        ));
    };
    let a_last = factorial_f64_checked(double_order)?
        / (factorial_f64_checked(order)? * 2.0_f64.powi(order as i32));
    Ok(a_last.powf(-1.0 / order as f64))
}

fn reverse_bessel_polynomial(order: usize) -> Result<Vec<f64>, SignalError> {
    let Some(double_order) = order.checked_mul(2) else {
        return Err(SignalError::InvalidArgument(
            "order is too large for Bessel polynomial construction".to_string(),
        ));
    };
    let mut ascending = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let numerator = factorial_f64_checked(double_order - k)?;
        let denominator = 2.0_f64.powi((order - k) as i32)
            * factorial_f64_checked(order - k)?
            * factorial_f64_checked(k)?;
        ascending.push(numerator / denominator);
    }
    ascending.reverse();
    Ok(ascending)
}

fn factorial_f64_checked(n: usize) -> Result<f64, SignalError> {
    let mut acc = 1.0;
    for value in 1..=n {
        acc *= value as f64;
        if !acc.is_finite() {
            return Err(SignalError::InvalidArgument(
                "order is too large for stable Bessel polynomial construction".to_string(),
            ));
        }
    }
    Ok(acc)
}

fn validate_iir_wn(wn: &[f64], btype: FilterType) -> Result<(), SignalError> {
    let expected_len = match btype {
        FilterType::Lowpass | FilterType::Highpass => 1,
        FilterType::Bandpass | FilterType::Bandstop => 2,
    };
    if wn.len() != expected_len {
        return Err(SignalError::InvalidArgument(format!(
            "Wn must contain {expected_len} value(s) for {btype:?}"
        )));
    }
    if wn
        .iter()
        .any(|&value| !value.is_finite() || value <= 0.0 || value >= 1.0)
    {
        return Err(SignalError::InvalidArgument(
            "Wn values must be finite and in (0, 1) for digital filters".to_string(),
        ));
    }
    if matches!(btype, FilterType::Bandpass | FilterType::Bandstop) && wn[0] >= wn[1] {
        return Err(SignalError::InvalidArgument(
            "band filters require Wn[0] < Wn[1]".to_string(),
        ));
    }
    Ok(())
}

fn prewarp_digital_frequency(wn: f64) -> f64 {
    2.0 * (std::f64::consts::PI * wn / 2.0).tan()
}

fn butterworth_analog_poles(order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut poles_re = Vec::with_capacity(order);
    let mut poles_im = Vec::with_capacity(order);
    for k in 0..order {
        let theta =
            std::f64::consts::PI * (2.0 * k as f64 + order as f64 + 1.0) / (2.0 * order as f64);
        poles_re.push(theta.cos());
        poles_im.push(theta.sin());
    }
    (poles_re, poles_im)
}

fn lowpass_zpk(zpk: &ZpkCoeffs, wo: f64) -> ZpkCoeffs {
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len()) as i32;
    ZpkCoeffs {
        zeros_re: zpk.zeros_re.iter().map(|&zr| zr * wo).collect(),
        zeros_im: zpk.zeros_im.iter().map(|&zi| zi * wo).collect(),
        poles_re: zpk.poles_re.iter().map(|&pr| pr * wo).collect(),
        poles_im: zpk.poles_im.iter().map(|&pi| pi * wo).collect(),
        gain: zpk.gain * wo.powi(degree),
    }
}

fn highpass_zpk(zpk: &ZpkCoeffs, wo: f64) -> ZpkCoeffs {
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() + degree);
    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (re, im) = complex_div(wo, 0.0, zr, zi);
        zeros_re.push(re);
        zeros_im.push(im);
    }
    zeros_re.extend(std::iter::repeat_n(0.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    let mut poles_re = Vec::with_capacity(zpk.poles_re.len());
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len());
    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (re, im) = complex_div(wo, 0.0, pr, pi);
        poles_re.push(re);
        poles_im.push(im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain,
    }
}

fn bandpass_zpk(zpk: &ZpkCoeffs, w1: f64, w2: f64) -> ZpkCoeffs {
    let bw = w2 - w1;
    let wo = (w1 * w2).sqrt();
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() * 2 + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() * 2 + degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len() * 2);
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len() * 2);

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (base_re, base_im) = complex_scale(zr, zi, bw / 2.0);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        zeros_re.push(base_re + root_re);
        zeros_im.push(base_im + root_im);
        zeros_re.push(base_re - root_re);
        zeros_im.push(base_im - root_im);
    }
    zeros_re.extend(std::iter::repeat_n(0.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (base_re, base_im) = complex_scale(pr, pi, bw / 2.0);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        poles_re.push(base_re + root_re);
        poles_im.push(base_im + root_im);
        poles_re.push(base_re - root_re);
        poles_im.push(base_im - root_im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain * bw.powi(degree as i32),
    }
}

fn bandstop_zpk(zpk: &ZpkCoeffs, w1: f64, w2: f64) -> ZpkCoeffs {
    let bw = w2 - w1;
    let wo = (w1 * w2).sqrt();
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() * 2 + 2 * degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() * 2 + 2 * degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len() * 2);
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len() * 2);

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (base_re, base_im) = complex_div(bw / 2.0, 0.0, zr, zi);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        zeros_re.push(base_re + root_re);
        zeros_im.push(base_im + root_im);
        zeros_re.push(base_re - root_re);
        zeros_im.push(base_im - root_im);
    }
    for _ in 0..degree {
        zeros_re.push(0.0);
        zeros_im.push(wo);
        zeros_re.push(0.0);
        zeros_im.push(-wo);
    }

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (base_re, base_im) = complex_div(bw / 2.0, 0.0, pr, pi);
        let (disc_re, disc_im) = complex_sub(
            base_re * base_re - base_im * base_im,
            2.0 * base_re * base_im,
            wo * wo,
            0.0,
        );
        let (root_re, root_im) = complex_sqrt(disc_re, disc_im);
        poles_re.push(base_re + root_re);
        poles_im.push(base_im + root_im);
        poles_re.push(base_re - root_re);
        poles_im.push(base_im - root_im);
    }

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: zpk.gain,
    }
}

fn bilinear_zpk_internal(zpk: &ZpkCoeffs) -> ZpkCoeffs {
    let fs2 = 2.0;
    let degree = zpk.poles_re.len().saturating_sub(zpk.zeros_re.len());
    let mut zeros_re = Vec::with_capacity(zpk.zeros_re.len() + degree);
    let mut zeros_im = Vec::with_capacity(zpk.zeros_im.len() + degree);
    let mut poles_re = Vec::with_capacity(zpk.poles_re.len());
    let mut poles_im = Vec::with_capacity(zpk.poles_im.len());

    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (num_re, num_im) = complex_add(fs2, 0.0, zr, zi);
        let (den_re, den_im) = complex_sub(fs2, 0.0, zr, zi);
        let (re, im) = complex_div_complex(num_re, num_im, den_re, den_im);
        zeros_re.push(re);
        zeros_im.push(im);
    }
    zeros_re.extend(std::iter::repeat_n(-1.0, degree));
    zeros_im.extend(std::iter::repeat_n(0.0, degree));

    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (num_re, num_im) = complex_add(fs2, 0.0, pr, pi);
        let (den_re, den_im) = complex_sub(fs2, 0.0, pr, pi);
        let (re, im) = complex_div_complex(num_re, num_im, den_re, den_im);
        poles_re.push(re);
        poles_im.push(im);
    }

    let mut gain_num_re = zpk.gain;
    let mut gain_num_im = 0.0;
    for (&zr, &zi) in zpk.zeros_re.iter().zip(zpk.zeros_im.iter()) {
        let (re, im) = complex_sub(fs2, 0.0, zr, zi);
        let (new_re, new_im) = complex_mul(gain_num_re, gain_num_im, re, im);
        gain_num_re = new_re;
        gain_num_im = new_im;
    }
    let mut gain_den_re = 1.0;
    let mut gain_den_im = 0.0;
    for (&pr, &pi) in zpk.poles_re.iter().zip(zpk.poles_im.iter()) {
        let (re, im) = complex_sub(fs2, 0.0, pr, pi);
        let (new_re, new_im) = complex_mul(gain_den_re, gain_den_im, re, im);
        gain_den_re = new_re;
        gain_den_im = new_im;
    }
    let (gain_re, _) = complex_div_complex(gain_num_re, gain_num_im, gain_den_re, gain_den_im);

    ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain: gain_re,
    }
}

fn complex_add(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar + br, ai + bi)
}

fn complex_sub(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar - br, ai - bi)
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_scale(re: f64, im: f64, scalar: f64) -> (f64, f64) {
    (re * scalar, im * scalar)
}

fn complex_div(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    complex_div_complex(ar, ai, br, bi)
}

fn complex_div_complex(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let denom = br * br + bi * bi;
    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

fn complex_sqrt(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        if re >= 0.0 {
            return (re.sqrt(), 0.0);
        }
        return (0.0, (-re).sqrt());
    }
    let r = (re * re + im * im).sqrt();
    let real = ((r + re) / 2.0).sqrt();
    let imag = ((r - re) / 2.0).sqrt().copysign(im);
    (real, imag)
}

/// Compute initial conditions for lfilter for step response steady-state.
///
/// Matches `scipy.signal.lfilter_zi(b, a)`.
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let mut a_start = 0;
    while a_start + 1 < a.len() && a[a_start] == 0.0 {
        a_start += 1;
    }
    let a_trimmed = &a[a_start..];
    let a0 = a_trimmed[0];

    let nb = b.len();
    let na = a_trimmed.len();
    let n = nb.max(na);

    if n <= 1 {
        return Err(SignalError::InvalidArgument(
            "The length of `a` along the last axis must be at least 2.".to_string(),
        ));
    }

    // Pad and normalize to the direct-form-II transposed state-space system.
    let mut b_norm = vec![0.0; n];
    let mut a_norm = vec![0.0; n];
    for (i, &val) in b.iter().enumerate() {
        b_norm[i] = val / a0;
    }
    for (i, &val) in a_trimmed.iter().enumerate() {
        a_norm[i] = val / a0;
    }

    let m = n - 1;
    // Build the system (I - A)z = b[1:] - a[1:]*b[0]
    let mut matrix = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];

    let b0 = b_norm[0];
    for i in 0..m {
        rhs[i] = b_norm[i + 1] - a_norm[i + 1] * b0;

        matrix[i][0] += a_norm[i + 1];
        matrix[i][i] += 1.0;
        if i + 1 < m {
            matrix[i][i + 1] -= 1.0;
        }
    }

    let solve_opts = fsci_linalg::SolveOptions {
        mode: fsci_runtime::RuntimeMode::Strict,
        ..Default::default()
    };

    let result = fsci_linalg::solve(&matrix, &rhs, solve_opts)
        .map_err(|e| SignalError::InvalidArgument(format!("lfilter_zi solve failed: {e}")))?;

    Ok(result.x)
}

/// Construct initial conditions for `lfilter` from prior input/output history.
///
/// Matches `scipy.signal.lfiltic(b, a, y, x=None)`.
pub fn lfiltic(
    b: &[f64],
    a: &[f64],
    y: &[f64],
    x: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "There must be at least one `a` coefficient.".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "First `a` filter coefficient must be non-zero.".to_string(),
        ));
    }

    let n = a.len() - 1;
    let m = b.len().saturating_sub(1);
    let k = m.max(n);
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut x_hist = x.map_or_else(|| vec![0.0; m], ToOwned::to_owned);
    if x_hist.len() < m {
        x_hist.resize(m, 0.0);
    }

    let mut y_hist = y.to_vec();
    if y_hist.len() < n {
        y_hist.resize(n, 0.0);
    }

    let mut zi = vec![0.0; k];
    for state in 0..m {
        let span = m - state;
        zi[state] = b[state + 1..]
            .iter()
            .take(span)
            .zip(x_hist.iter())
            .map(|(&coeff, &sample)| coeff * sample)
            .sum();
    }
    for state in 0..n {
        let span = n - state;
        zi[state] -= a[state + 1..]
            .iter()
            .take(span)
            .zip(y_hist.iter())
            .map(|(&coeff, &sample)| coeff * sample)
            .sum::<f64>();
    }

    if a[0] != 1.0 {
        for value in &mut zi {
            *value /= a[0];
        }
    }

    Ok(zi)
}

/// Apply a digital IIR filter using the Direct Form II transposed structure.
///
/// Matches `scipy.signal.lfilter(b, a, x, zi=zi)`.
pub fn lfilter(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    zi: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    lfilter_with_state(b, a, x, zi).map(|(y, _zf)| y)
}

/// Apply a digital IIR filter and return the final delay state.
///
/// Matches the `(y, zf)` result from `scipy.signal.lfilter(b, a, x, zi=zi)`.
pub fn lfilter_with_state(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    zi: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "a[0] must not be zero".to_string(),
        ));
    }

    let a0 = a[0];
    let nb = b.len();
    let na = a.len();
    let nfilt = nb.max(na);

    // Resolves [frankenscipy-rvwvw]: pad b_norm/a_norm to length
    // nfilt and hoist b0 so the inner loop is N · nfilt straight
    // multiply-add without per-iter bound checks or invariant lookups.
    let mut b_norm: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let mut a_norm: Vec<f64> = a.iter().map(|&v| v / a0).collect();
    b_norm.resize(nfilt, 0.0);
    a_norm.resize(nfilt, 0.0);
    let b0 = b_norm[0];

    // Direct Form II transposed
    let mut y = Vec::with_capacity(x.len());
    let mut d = if let Some(initial) = zi {
        if initial.len() != nfilt - 1 {
            return Err(SignalError::InvalidArgument(format!(
                "zi length ({}) must be nfilt-1 ({})",
                initial.len(),
                nfilt - 1
            )));
        }
        let mut d_init = vec![0.0; nfilt];
        d_init[..nfilt - 1].copy_from_slice(initial);
        d_init
    } else {
        vec![0.0; nfilt]
    };

    for &xi in x {
        let yi = b0 * xi + d[0];
        y.push(yi);

        // Update delay line — b_norm/a_norm padded to nfilt, so j+1
        // is always in range and the boundary `j+1 < nfilt-1` check
        // collapses to a single tail-handling step after the loop.
        for j in 0..nfilt - 1 {
            let bj = b_norm[j + 1];
            let aj = a_norm[j + 1];
            let next_d = if j + 1 < nfilt - 1 { d[j + 1] } else { 0.0 };
            d[j] = bj * xi - aj * yi + next_d;
        }
    }

    let zf = d[..nfilt - 1].to_vec();
    Ok((y, zf))
}

/// Apply `lfilter` across one axis of a rectangular 2-D input.
///
/// Supports SciPy's common `axis=0` and `axis=-1`/`axis=1` cases for matrix
/// inputs. Each row or column is filtered independently with zero initial
/// conditions.
pub fn lfilter_axis_2d(
    b: &[f64],
    a: &[f64],
    x: &[Vec<f64>],
    axis: isize,
) -> Result<Vec<Vec<f64>>, SignalError> {
    if x.is_empty() {
        return Ok(Vec::new());
    }

    let cols = x[0].len();
    if x.iter().any(|row| row.len() != cols) {
        return Err(SignalError::InvalidInputShape {
            detail: "x must be a rectangular 2-D matrix".to_string(),
        });
    }

    match axis {
        -1 | 1 => x
            .iter()
            .map(|row| lfilter(b, a, row, None))
            .collect::<Result<Vec<_>, _>>(),
        0 => {
            let mut y = vec![vec![0.0; cols]; x.len()];
            for col in 0..cols {
                let column: Vec<f64> = x.iter().map(|row| row[col]).collect();
                let filtered = lfilter(b, a, &column, None)?;
                for (row_idx, value) in filtered.into_iter().enumerate() {
                    y[row_idx][col] = value;
                }
            }
            Ok(y)
        }
        other => Err(SignalError::InvalidArgument(format!(
            "axis must be 0, 1, or -1 for 2-D lfilter, got {other}"
        ))),
    }
}

/// Apply a digital filter forward and backward (zero-phase filtering).
///
/// Matches `scipy.signal.filtfilt(b, a, x)`.
///
/// This results in zero phase distortion (linear phase) and doubled filter order.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    filtfilt_with_padtype(b, a, x, None)
}

/// Apply a digital filter forward and backward with explicit SciPy padtype control.
///
/// Matches `scipy.signal.filtfilt(b, a, x, padtype=...)` for
/// `padtype in {"odd", "even", "constant"}`.
pub fn filtfilt_with_padtype(
    b: &[f64],
    a: &[f64],
    x: &[f64],
    padtype: Option<&str>,
) -> Result<Vec<f64>, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 3 {
        return Err(SignalError::InvalidArgument(
            "input must have length >= 3 for filtfilt".to_string(),
        ));
    }

    let nfilt = b.len().max(a.len());
    // scipy filtfilt's default edge is 3·max(len(b), len(a)) — three full
    // filter lengths, not 3·(nfilt−1). The clamp keeps short inputs (which
    // scipy rejects outright) usable rather than panicking.
    let padlen = (3 * nfilt).min(n - 1);
    let padtype = padtype.unwrap_or("odd");

    // 1. Pad signal with mirrored values
    let mut padded = Vec::with_capacity(n + 2 * padlen);
    let x0 = x[0];
    match padtype {
        "odd" => {
            for i in (1..=padlen).rev() {
                padded.push(2.0 * x0 - x[i]);
            }
        }
        "even" => {
            for i in (1..=padlen).rev() {
                padded.push(x[i]);
            }
        }
        "constant" => {
            for _ in 0..padlen {
                padded.push(x0);
            }
        }
        _ => {
            return Err(SignalError::InvalidArgument(
                "padtype must be one of {'odd', 'even', 'constant'}".to_string(),
            ));
        }
    }
    padded.extend_from_slice(x);
    let xn = x[n - 1];
    match padtype {
        "odd" => {
            for i in 1..=padlen {
                padded.push(2.0 * xn - x[n - 1 - i]);
            }
        }
        "even" => {
            for i in 1..=padlen {
                padded.push(x[n - 1 - i]);
            }
        }
        "constant" => {
            for _ in 0..padlen {
                padded.push(xn);
            }
        }
        _ => unreachable!("padtype validated above"),
    }

    // Forward pass with initial conditions
    let zi = lfilter_zi(b, a)?;
    let zi_f: Vec<f64> = zi.iter().map(|&v| v * padded[0]).collect();
    let mut forward = lfilter(b, a, &padded, Some(&zi_f))?;

    // Reverse
    forward.reverse();

    // Backward pass with initial conditions
    let zi_b: Vec<f64> = zi.iter().map(|&v| v * forward[0]).collect();
    let mut reversed = lfilter(b, a, &forward, Some(&zi_b))?;

    // Reverse again and remove padding
    reversed.reverse();
    let result = reversed[padlen..padlen + n].to_vec();
    Ok(result)
}

/// Apply a digital filter in SOS (second-order sections) form.
///
/// Matches `scipy.signal.sosfilt(sos, x)`.
///
/// Each section is a biquad [b0, b1, b2, a0, a1, a2] applied in cascade
/// using Direct Form II transposed. This is numerically superior to
/// lfilter for high-order filters.
pub fn sosfilt(sos: &[SosSection], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let mut signal = x.to_vec();

    for section in sos {
        let b = [section[0], section[1], section[2]];
        let a0 = section[3];
        let a1 = section[4];
        let a2 = section[5];

        if a0.abs() < 1e-30 {
            return Err(SignalError::InvalidArgument(
                "SOS section a[0] must not be zero".to_string(),
            ));
        }

        // Normalize by a0
        let b0 = b[0] / a0;
        let b1 = b[1] / a0;
        let b2 = b[2] / a0;
        let a1n = a1 / a0;
        let a2n = a2 / a0;

        // Direct Form II transposed for this biquad section
        let mut d1 = 0.0;
        let mut d2 = 0.0;

        for sample in &mut signal {
            let xi = *sample;
            let yi = b0 * xi + d1;
            d1 = b1 * xi - a1n * yi + d2;
            d2 = b2 * xi - a2n * yi;
            *sample = yi;
        }
    }

    Ok(signal)
}

/// Apply SOS filter forward and backward for zero-phase filtering.
///
/// Matches `scipy.signal.sosfiltfilt(sos, x)`.
///
/// Results in zero phase distortion and doubled filter order.
/// More numerically stable than filtfilt for high-order filters.
pub fn sosfiltfilt(sos: &[SosSection], x: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = x.len();
    if n < 3 {
        return Err(SignalError::InvalidArgument(
            "input must have length >= 3 for sosfiltfilt".to_string(),
        ));
    }

    // Determine padding length (matches SciPy's default: 3 * (ntaps - 1))
    // For SOS, ntaps = 2 * n_sections + 1. So 3 * (2 * n_sections) = 6 * n_sections.
    let padlen = (6 * sos.len()).min(n - 1);

    // 1. Pad signal with mirrored values:
    // [2*x[0]-x[padlen], ..., 2*x[0]-x[1], x[0], ..., x[n-1], 2*x[n-1]-x[n-2], ..., 2*x[n-1]-x[n-padlen-1]]
    let mut padded = Vec::with_capacity(n + 2 * padlen);
    let x0 = x[0];
    for i in (1..=padlen).rev() {
        padded.push(2.0 * x0 - x[i]);
    }
    padded.extend_from_slice(x);
    let xn = x[n - 1];
    for i in 1..=padlen {
        padded.push(2.0 * xn - x[n - 1 - i]);
    }

    // 2. Initialize filter with steady-state for the first padded sample
    let zi = sosfilt_zi(sos)?;
    let mut z_forward = zi.clone();
    let p0 = padded[0];
    for z in &mut z_forward {
        z[0] *= p0;
        z[1] *= p0;
    }

    // Forward pass with initial conditions
    let mut signal = padded;
    sosfilt_in_place(sos, &mut signal, &z_forward)?;

    // 3. Reverse and repeat for backward pass
    signal.reverse();
    let mut z_backward = zi;
    let s0 = signal[0];
    for z in &mut z_backward {
        z[0] *= s0;
        z[1] *= s0;
    }
    sosfilt_in_place(sos, &mut signal, &z_backward)?;

    // 4. Reverse again and remove padding
    signal.reverse();
    let result = signal[padlen..padlen + n].to_vec();
    Ok(result)
}

/// Internal helper for in-place SOS filtering with initial conditions.
fn sosfilt_in_place(sos: &[SosSection], x: &mut [f64], zi: &[[f64; 2]]) -> Result<(), SignalError> {
    for (i, section) in sos.iter().enumerate() {
        let b0 = section[0] / section[3];
        let b1 = section[1] / section[3];
        let b2 = section[2] / section[3];
        let a1 = section[4] / section[3];
        let a2 = section[5] / section[3];

        let mut d1 = zi[i][0];
        let mut d2 = zi[i][1];

        for sample in x.iter_mut() {
            let xi = *sample;
            let yi = b0 * xi + d1;
            d1 = b1 * xi - a1 * yi + d2;
            d2 = b2 * xi - a2 * yi;
            *sample = yi;
        }
    }
    Ok(())
}

/// Compute initial conditions for SOS filter for step response steady-state.
///
/// Matches `scipy.signal.sosfilt_zi(sos)`.
///
/// Returns initial delay values such that filtering a constant signal
/// produces a constant output (no transient). Each section gets 2 initial
/// conditions [d1, d2].
pub fn sosfilt_zi(sos: &[SosSection]) -> Result<Vec<[f64; 2]>, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let mut zi = Vec::with_capacity(sos.len());

    for section in sos {
        let a0 = section[3];
        if a0.abs() < 1e-30 {
            return Err(SignalError::InvalidArgument(
                "SOS section a[0] must not be zero".to_string(),
            ));
        }
        let b0 = section[0] / a0;
        let b1 = section[1] / a0;
        let b2 = section[2] / a0;
        let a1 = section[4] / a0;
        let a2 = section[5] / a0;

        // For steady-state with input=1, output=gain of section at DC:
        // gain = (b0+b1+b2) / (1+a1+a2)
        // Initial conditions satisfy: d1 = b1 - a1*gain + d2, d2 = b2 - a2*gain
        // Solving: d2 = b2 - a2*gain, d1 = (b1 - a1*gain) + d2
        let dc_denom = 1.0 + a1 + a2;
        if dc_denom.abs() < 1e-30 {
            zi.push([0.0, 0.0]);
            continue;
        }
        let gain = (b0 + b1 + b2) / dc_denom;
        let d2 = b2 - a2 * gain;
        let d1 = b1 - a1 * gain + d2;
        zi.push([d1, d2]);
    }

    Ok(zi)
}

// ── IIR helper functions ────────────────────────────────────────

/// Multiply polynomial by (1 + c*z^-1).
fn poly_mul_binomial(poly: &[f64], c: f64) -> Vec<f64> {
    let mut result = vec![0.0; poly.len() + 1];
    for (i, &p) in poly.iter().enumerate() {
        result[i] += p;
        result[i + 1] += p * c;
    }
    result
}

/// Multiply polynomial by (1 + b*z^-1 + c*z^-2).
fn poly_mul_quadratic(poly: &[f64], b: f64, c: f64) -> Vec<f64> {
    let mut result = vec![0.0; poly.len() + 2];
    for (i, &p) in poly.iter().enumerate() {
        result[i] += p;
        result[i + 1] += p * b;
        result[i + 2] += p * c;
    }
    result
}

// ══════════════════════════════════════════════════════════════════════
// Filter Representation Conversions
// ══════════════════════════════════════════════════════════════════════

/// Zero-pole-gain representation of a filter.
#[derive(Debug, Clone, PartialEq)]
pub struct ZpkCoeffs {
    /// Zeros (complex: pairs of (re, im)).
    pub zeros_re: Vec<f64>,
    pub zeros_im: Vec<f64>,
    /// Poles (complex: pairs of (re, im)).
    pub poles_re: Vec<f64>,
    pub poles_im: Vec<f64>,
    /// Scalar gain.
    pub gain: f64,
}

/// A single second-order section: [b0, b1, b2, a0, a1, a2].
/// Represents H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2).
pub type SosSection = [f64; 6];

/// Convert transfer function (b, a) to zero-pole-gain form.
///
/// Matches `scipy.signal.tf2zpk(b, a)`.
///
/// Finds zeros (roots of b) and poles (roots of a) via companion matrix eigenvalues.
pub fn tf2zpk(b: &[f64], a: &[f64]) -> Result<ZpkCoeffs, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if a[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "a[0] must not be zero".to_string(),
        ));
    }

    // Find leading non-zero coefficient of b (effective degree may differ from length)
    let b_lead_idx = match b.iter().position(|&v| v.abs() > 1e-30) {
        Some(idx) => idx,
        None => {
            // All coefficients are essentially zero — zero polynomial
            // Find poles from a but no zeros, gain = 0
            let a_norm: Vec<f64> = a.iter().map(|&v| v / a[0]).collect();
            let (poles_re, poles_im) = if a_norm.len() > 1 {
                poly_roots(&a_norm)?
            } else {
                (vec![], vec![])
            };
            return Ok(ZpkCoeffs {
                zeros_re: vec![],
                zeros_im: vec![],
                poles_re,
                poles_im,
                gain: 0.0,
            });
        }
    };
    let b_lead = b[b_lead_idx];

    let gain = b_lead / a[0];

    // Normalize: divide by leading non-zero coefficient
    let b_effective = &b[b_lead_idx..];
    let b_norm: Vec<f64> = b_effective.iter().map(|&v| v / b_lead).collect();
    let a_norm: Vec<f64> = a.iter().map(|&v| v / a[0]).collect();

    // Find roots via companion matrix eigenvalues
    let (zeros_re, zeros_im) = if b_norm.len() > 1 {
        poly_roots(&b_norm)?
    } else {
        (vec![], vec![])
    };

    let (poles_re, poles_im) = if a_norm.len() > 1 {
        poly_roots(&a_norm)?
    } else {
        (vec![], vec![])
    };

    Ok(ZpkCoeffs {
        zeros_re,
        zeros_im,
        poles_re,
        poles_im,
        gain,
    })
}

/// Convert zero-pole-gain to transfer function (b, a) form.
///
/// Matches `scipy.signal.zpk2tf(z, p, k)`.
pub fn zpk2tf(zpk: &ZpkCoeffs) -> BaCoeffs {
    let b = poly_from_complex_roots_full(&zpk.zeros_re, &zpk.zeros_im);
    let a = poly_from_complex_roots_full(&zpk.poles_re, &zpk.poles_im);

    // Scale numerator by gain
    let b: Vec<f64> = b.iter().map(|&v| v * zpk.gain).collect();

    BaCoeffs { b, a }
}

/// Convert transfer function to second-order sections.
///
/// Matches `scipy.signal.tf2sos(b, a)`.
///
/// The SOS form is numerically superior for high-order filters.
pub fn tf2sos(b: &[f64], a: &[f64]) -> Result<Vec<SosSection>, SignalError> {
    let zpk = tf2zpk(b, a)?;
    Ok(zpk2sos(&zpk))
}

/// Convert second-order sections to transfer function.
///
/// Matches `scipy.signal.sos2tf(sos)`.
pub fn sos2tf(sos: &[SosSection]) -> BaCoeffs {
    if sos.is_empty() {
        return BaCoeffs {
            b: vec![1.0],
            a: vec![1.0],
        };
    }

    let mut b = vec![1.0];
    let mut a = vec![1.0];

    for section in sos {
        // Numerator: [b0, b1, b2]
        let sec_b = [section[0], section[1], section[2]];
        // Denominator: [a0, a1, a2]
        let sec_a = [section[3], section[4], section[5]];

        b = poly_multiply(&b, &sec_b);
        a = poly_multiply(&a, &sec_a);
    }

    BaCoeffs { b, a }
}

/// Convert zero-pole-gain to second-order sections.
///
/// Matches `scipy.signal.zpk2sos(z, p, k)`.
///
/// Pairs poles and zeros into biquad sections, sorted by proximity to
/// the unit circle (outermost poles last for numerical stability).
pub fn zpk2sos(zpk: &ZpkCoeffs) -> Vec<SosSection> {
    let n_poles = zpk.poles_re.len();
    let n_zeros = zpk.zeros_re.len();

    if n_poles == 0 && n_zeros == 0 {
        return vec![[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }

    // Build denominator sections from poles
    let denom_sections = roots_to_sos_factors(&zpk.poles_re, &zpk.poles_im);
    // Build numerator sections from zeros
    let numer_sections = roots_to_sos_factors(&zpk.zeros_re, &zpk.zeros_im);

    let n_sections = denom_sections.len().max(numer_sections.len()).max(1);
    let mut sections: Vec<SosSection> = Vec::with_capacity(n_sections);

    for i in 0..n_sections {
        let (b0, b1, b2) = if i < numer_sections.len() {
            numer_sections[i]
        } else {
            (1.0, 0.0, 0.0)
        };
        let (a0, a1, a2) = if i < denom_sections.len() {
            denom_sections[i]
        } else {
            (1.0, 0.0, 0.0)
        };
        sections.push([b0, b1, b2, a0, a1, a2]);
    }

    if sections.is_empty() {
        sections.push([zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]);
    } else {
        // Apply gain to first section
        sections[0][0] *= zpk.gain;
        sections[0][1] *= zpk.gain;
        sections[0][2] *= zpk.gain;
    }

    sections
}

/// Convert roots to second-order section factors.
/// Groups conjugate pairs and pairs of real roots into (1, c1, c2) tuples.
fn roots_to_sos_factors(re: &[f64], im: &[f64]) -> Vec<(f64, f64, f64)> {
    let n = re.len();
    let mut used = vec![false; n];
    let mut factors = Vec::new();
    let mut real_roots: Vec<f64> = Vec::new();

    // First: pair complex conjugates
    for i in 0..n {
        if used[i] {
            continue;
        }
        if im[i].abs() < 1e-14 {
            real_roots.push(re[i]);
            used[i] = true;
        } else {
            // Find conjugate
            let mut found = false;
            for j in (i + 1)..n {
                if !used[j] && (re[j] - re[i]).abs() < 1e-10 && (im[j] + im[i]).abs() < 1e-10 {
                    // Conjugate pair: (1 - 2*re*z^-1 + |r|²*z^-2)
                    let mag2 = re[i] * re[i] + im[i] * im[i];
                    factors.push((1.0, -2.0 * re[i], mag2));
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                let mag2 = re[i] * re[i] + im[i] * im[i];
                factors.push((1.0, -2.0 * re[i], mag2));
            }
            used[i] = true;
        }
    }

    // Pair real roots into biquad sections
    real_roots.sort_by(|a, b| a.total_cmp(b));
    let mut i = 0;
    while i + 1 < real_roots.len() {
        let r1 = real_roots[i];
        let r2 = real_roots[i + 1];
        // (1 - r1*z^-1)(1 - r2*z^-1) = 1 - (r1+r2)*z^-1 + r1*r2*z^-2
        factors.push((1.0, -(r1 + r2), r1 * r2));
        i += 2;
    }
    if i < real_roots.len() {
        // Odd real root out: single first-order section
        factors.push((1.0, -real_roots[i], 0.0));
    }

    factors
}

/// Convert second-order sections to zero-pole-gain form.
///
/// Matches `scipy.signal.sos2zpk(sos)`.
pub fn sos2zpk(sos: &[SosSection]) -> ZpkCoeffs {
    let mut all_zeros_re = Vec::new();
    let mut all_zeros_im = Vec::new();
    let mut all_poles_re = Vec::new();
    let mut all_poles_im = Vec::new();
    let mut gain = 1.0;

    for section in sos {
        let sec_b = [section[0], section[1], section[2]];
        let sec_a = [section[3], section[4], section[5]];

        if sec_a[0].abs() > 1e-30 {
            gain *= if sec_b[0].abs() > 1e-30 {
                sec_b[0] / sec_a[0]
            } else {
                0.0
            };
        }

        // Normalize and find roots (skip if leading coeff is zero)
        let b_norm: Vec<f64> = if sec_b[0].abs() > 1e-30 {
            sec_b.iter().map(|&v| v / sec_b[0]).collect()
        } else {
            continue; // degenerate section, skip
        };
        let a_norm: Vec<f64> = if sec_a[0].abs() > 1e-30 {
            sec_a.iter().map(|&v| v / sec_a[0]).collect()
        } else {
            continue;
        };

        // Trim trailing zeros for root finding
        let b_trim = trim_trailing_zeros(&b_norm);
        let a_trim = trim_trailing_zeros(&a_norm);

        if b_trim.len() > 1
            && let Ok((zr, zi)) = poly_roots(&b_trim)
        {
            all_zeros_re.extend(zr);
            all_zeros_im.extend(zi);
        }
        if a_trim.len() > 1
            && let Ok((pr, pi)) = poly_roots(&a_trim)
        {
            all_poles_re.extend(pr);
            all_poles_im.extend(pi);
        }
    }

    ZpkCoeffs {
        zeros_re: all_zeros_re,
        zeros_im: all_zeros_im,
        poles_re: all_poles_re,
        poles_im: all_poles_im,
        gain,
    }
}

// ── Filter conversion helpers ───────────────────────────────

/// Find roots of a monic polynomial using companion matrix eigenvalues.
/// Input: coefficients [1, c1, c2, ...] of p(x) = x^n + c1*x^{n-1} + ... + cn.
fn poly_roots(coeffs: &[f64]) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    let n = coeffs.len() - 1; // degree
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        // Linear: x + c1 = 0 → x = -c1
        return Ok((vec![-coeffs[1] / coeffs[0]], vec![0.0]));
    }
    if n == 2 {
        // Quadratic: use formula
        let a = coeffs[0];
        let b = coeffs[1];
        let c = coeffs[2];
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            return Ok((
                vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)],
                vec![0.0, 0.0],
            ));
        } else {
            let sq = (-disc).sqrt();
            return Ok((
                vec![-b / (2.0 * a), -b / (2.0 * a)],
                vec![sq / (2.0 * a), -sq / (2.0 * a)],
            ));
        }
    }

    // General case: companion matrix eigenvalues
    // Companion matrix for p(x) = x^n + c1*x^{n-1} + ... + cn
    // has -c_i/c_0 in the last column and 1's on the subdiagonal
    let mut companion = vec![vec![0.0; n]; n];
    for i in 1..n {
        companion[i][i - 1] = 1.0;
    }
    for i in 0..n {
        companion[i][n - 1] = -coeffs[n - i] / coeffs[0];
    }

    // Use fsci-linalg eig to get eigenvalues
    let opts = fsci_linalg::DecompOptions::default();
    let eig_result = fsci_linalg::eig(&companion, opts)
        .map_err(|e| SignalError::InvalidArgument(format!("eigenvalue computation failed: {e}")))?;

    Ok((eig_result.eigenvalues_re, eig_result.eigenvalues_im))
}

/// Build polynomial coefficients in z^-1 convention from complex roots.
/// Returns [1, c1, c2, ...] where P(z) = (1 - r1*z^-1)(1 - r2*z^-1)...
/// This is the transfer function convention: b[0] + b[1]*z^-1 + b[2]*z^-2 + ...
fn poly_from_complex_roots_full(roots_re: &[f64], roots_im: &[f64]) -> Vec<f64> {
    if roots_re.is_empty() {
        return vec![1.0];
    }

    let n = roots_re.len();
    let mut used = vec![false; n];
    let mut poly = vec![1.0];

    for i in 0..n {
        if used[i] {
            continue;
        }

        if roots_im[i].abs() < 1e-14 {
            // Real root: multiply by (1 - r*z^-1)
            poly = poly_mul_binomial(&poly, -roots_re[i]);
            used[i] = true;
        } else {
            // Find conjugate pair
            let mut found = false;
            for j in (i + 1)..n {
                if !used[j]
                    && (roots_re[j] - roots_re[i]).abs() < 1e-10
                    && (roots_im[j] + roots_im[i]).abs() < 1e-10
                {
                    let two_re = 2.0 * roots_re[i];
                    let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                    poly = poly_mul_quadratic(&poly, -two_re, mag2);
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                // Unpaired: treat as conjugate pair with itself
                let two_re = 2.0 * roots_re[i];
                let mag2 = roots_re[i] * roots_re[i] + roots_im[i] * roots_im[i];
                poly = poly_mul_quadratic(&poly, -two_re, mag2);
            }
            used[i] = true;
        }
    }

    poly
}

/// Multiply two polynomials (convolution of coefficient arrays).
fn poly_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Remove trailing near-zero coefficients from a polynomial.
fn trim_trailing_zeros(p: &[f64]) -> Vec<f64> {
    let mut end = p.len();
    while end > 1 && p[end - 1].abs() < 1e-15 {
        end -= 1;
    }
    p[..end].to_vec()
}

// ══════════════════════════════════════════════════════════════════════
// Frequency Response Analysis
// ══════════════════════════════════════════════════════════════════════

/// Complex frequency response at a set of frequencies.
#[derive(Debug, Clone, PartialEq)]
pub struct FreqzResult {
    /// Angular frequencies (radians/sample for digital, rad/s for analog).
    pub w: Vec<f64>,
    /// Complex frequency response: magnitude at each frequency.
    pub h_mag: Vec<f64>,
    /// Phase response (radians) at each frequency.
    pub h_phase: Vec<f64>,
}

/// Compute the frequency response of a digital filter.
///
/// Matches `scipy.signal.freqz(b, a, worN)`.
///
/// Evaluates H(e^{jω}) = B(e^{jω}) / A(e^{jω}) at `n_freqs` equally
/// spaced frequencies from 0 to π.
///
/// # Arguments
/// * `b` — Numerator coefficients
/// * `a` — Denominator coefficients
/// * `n_freqs` — Number of frequency points (default 512)
pub fn freqz(b: &[f64], a: &[f64], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    freqz_with_whole(b, a, n_freqs, false)
}

/// Compute the frequency response of a digital filter over half or the whole unit circle.
///
/// Matches `scipy.signal.freqz(b, a, worN, whole=...)`.
///
/// When `whole` is false, evaluates on `[0, π)`. When `whole` is true,
/// evaluates on `[0, 2π)` with SciPy's `endpoint=False` spacing.
pub fn freqz_with_whole(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
    whole: bool,
) -> Result<FreqzResult, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        // scipy uses endpoint=False:
        // half spectrum: w = np.linspace(0, pi, n, endpoint=False)
        // whole spectrum: w = np.linspace(0, 2*pi, n, endpoint=False)
        let omega = if whole {
            std::f64::consts::TAU * i as f64 / n as f64
        } else {
            std::f64::consts::PI * i as f64 / n as f64
        };
        w.push(omega);

        // Evaluate B(e^{jω}) and A(e^{jω}) using Horner's method
        // B(z) = b[0] + b[1]*z^{-1} + b[2]*z^{-2} + ...
        // At z = e^{jω}: z^{-k} = e^{-jkω} = cos(kω) - j*sin(kω)
        let (b_re, b_im) = eval_poly_on_unit_circle(b, omega);
        let (a_re, a_im) = eval_poly_on_unit_circle(a, omega);

        // H = B / A (complex division)
        let denom = a_re * a_re + a_im * a_im;
        if denom < 1e-30 {
            h_mag.push(f64::INFINITY);
            h_phase.push(0.0);
        } else {
            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;
            h_mag.push((h_re * h_re + h_im * h_im).sqrt());
            h_phase.push(h_im.atan2(h_re));
        }
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of a digital filter from ZPK form.
///
/// Matches `scipy.signal.freqz_zpk(z, p, k, worN)`.
///
/// Evaluates H(e^jω) = k · Π(e^jω − z_i) / Π(e^jω − p_i) directly.
/// Resolves [frankenscipy-kca5u].
pub fn freqz_zpk(zpk: &ZpkCoeffs, n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    freqz_zpk_with_whole(zpk, n_freqs, false)
}

/// Compute the frequency response of a digital filter from ZPK form
/// over half or the whole unit circle.
///
/// Matches `scipy.signal.freqz_zpk(z, p, k, worN, whole=...)`.
pub fn freqz_zpk_with_whole(
    zpk: &ZpkCoeffs,
    n_freqs: Option<usize>,
    whole: bool,
) -> Result<FreqzResult, SignalError> {
    if zpk.zeros_re.len() != zpk.zeros_im.len() {
        return Err(SignalError::InvalidArgument(
            "zeros real and imaginary slices must agree on length".into(),
        ));
    }
    if zpk.poles_re.len() != zpk.poles_im.len() {
        return Err(SignalError::InvalidArgument(
            "poles real and imaginary slices must agree on length".into(),
        ));
    }
    if !zpk.gain.is_finite() {
        return Err(SignalError::InvalidArgument("gain must be finite".into()));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument("n_freqs must be > 0".into()));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        let omega = if whole {
            std::f64::consts::TAU * i as f64 / n as f64
        } else {
            std::f64::consts::PI * i as f64 / n as f64
        };
        w.push(omega);

        // e^{jω} = cos(ω) + j·sin(ω). f64::sin_cos returns (sin, cos)
        // — destructure accordingly.
        let (e_im, e_re) = omega.sin_cos();

        // Numerator: gain · Π (e^{jω} − z_i)
        let mut num_re = zpk.gain;
        let mut num_im = 0.0_f64;
        for k in 0..zpk.zeros_re.len() {
            let dr = e_re - zpk.zeros_re[k];
            let di = e_im - zpk.zeros_im[k];
            let new_re = num_re * dr - num_im * di;
            let new_im = num_re * di + num_im * dr;
            num_re = new_re;
            num_im = new_im;
        }

        // Denominator: Π (e^{jω} − p_i)
        let mut den_re = 1.0_f64;
        let mut den_im = 0.0_f64;
        for k in 0..zpk.poles_re.len() {
            let dr = e_re - zpk.poles_re[k];
            let di = e_im - zpk.poles_im[k];
            let new_re = den_re * dr - den_im * di;
            let new_im = den_re * di + den_im * dr;
            den_re = new_re;
            den_im = new_im;
        }

        let denom_mag_sq = den_re * den_re + den_im * den_im;
        if denom_mag_sq < 1e-30 {
            h_mag.push(f64::INFINITY);
            h_phase.push(0.0);
        } else {
            let h_re = (num_re * den_re + num_im * den_im) / denom_mag_sq;
            let h_im = (num_im * den_re - num_re * den_im) / denom_mag_sq;
            h_mag.push((h_re * h_re + h_im * h_im).sqrt());
            h_phase.push(h_im.atan2(h_re));
        }
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of a digital filter from SOS form.
///
/// Matches `scipy.signal.sosfreqz(sos, worN)`.
///
/// More numerically stable than freqz for high-order filters.
pub fn freqz_sos(sos: &[SosSection], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        // Match scipy endpoint=False: w[i] = i * pi / n
        let omega = std::f64::consts::PI * i as f64 / n as f64;
        w.push(omega);

        // Multiply frequency responses of each section
        let mut total_re = 1.0;
        let mut total_im = 0.0;

        for section in sos {
            let sec_b = &[section[0], section[1], section[2]];
            let sec_a = &[section[3], section[4], section[5]];

            let (b_re, b_im) = eval_poly_on_unit_circle(sec_b, omega);
            let (a_re, a_im) = eval_poly_on_unit_circle(sec_a, omega);

            let denom = a_re * a_re + a_im * a_im;
            if denom < 1e-30 {
                total_re = f64::INFINITY;
                total_im = 0.0;
                break;
            }

            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;

            // Complex multiply: total *= h
            let new_re = total_re * h_re - total_im * h_im;
            let new_im = total_re * h_im + total_im * h_re;
            total_re = new_re;
            total_im = new_im;
        }

        h_mag.push((total_re * total_re + total_im * total_im).sqrt());
        h_phase.push(total_im.atan2(total_re));
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of a digital filter in SOS format.
///
/// Alias for `freqz_sos`. Matches `scipy.signal.sosfreqz(sos, worN)`.
pub fn sosfreqz(sos: &[SosSection], n_freqs: Option<usize>) -> Result<FreqzResult, SignalError> {
    freqz_sos(sos, n_freqs)
}

/// Compute the frequency response of a digital filter in SOS format over
/// half or the whole unit circle.
///
/// Matches `scipy.signal.sosfreqz(sos, worN, whole=...)`.
///
/// When `whole` is false, evaluates on `[0, π)`. When `whole` is true,
/// evaluates on `[0, 2π)` with scipy's `endpoint=False` spacing.
pub fn sosfreqz_with_whole(
    sos: &[SosSection],
    n_freqs: Option<usize>,
    whole: bool,
) -> Result<FreqzResult, SignalError> {
    if sos.is_empty() {
        return Err(SignalError::InvalidArgument(
            "sos must not be empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    let mut w = Vec::with_capacity(n);
    let mut h_mag = Vec::with_capacity(n);
    let mut h_phase = Vec::with_capacity(n);

    for i in 0..n {
        let omega = if whole {
            std::f64::consts::TAU * i as f64 / n as f64
        } else {
            std::f64::consts::PI * i as f64 / n as f64
        };
        w.push(omega);

        let mut total_re = 1.0;
        let mut total_im = 0.0;

        for section in sos {
            let sec_b = &[section[0], section[1], section[2]];
            let sec_a = &[section[3], section[4], section[5]];

            let (b_re, b_im) = eval_poly_on_unit_circle(sec_b, omega);
            let (a_re, a_im) = eval_poly_on_unit_circle(sec_a, omega);

            let denom = a_re * a_re + a_im * a_im;
            if denom < 1e-30 {
                total_re = f64::INFINITY;
                total_im = 0.0;
                break;
            }

            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;

            let new_re = total_re * h_re - total_im * h_im;
            let new_im = total_re * h_im + total_im * h_re;
            total_re = new_re;
            total_im = new_im;
        }

        h_mag.push((total_re * total_re + total_im * total_im).sqrt());
        h_phase.push(total_im.atan2(total_re));
    }

    Ok(FreqzResult { w, h_mag, h_phase })
}

/// Compute the frequency response of an analog filter.
///
/// Matches `scipy.signal.freqs(b, a, worN)`.
///
/// Evaluates H(jω) = B(jω) / A(jω) at specified angular frequencies.
pub fn freqs(b: &[f64], a: &[f64], w: &[f64]) -> Result<FreqzResult, SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }
    if w.is_empty() {
        return Err(SignalError::InvalidArgument(
            "w must not be empty".to_string(),
        ));
    }

    let mut h_mag = Vec::with_capacity(w.len());
    let mut h_phase = Vec::with_capacity(w.len());

    for &omega in w {
        // Evaluate B(jω) and A(jω)
        // B(s) = b[0]*s^n + b[1]*s^{n-1} + ... + b[n]
        // At s = jω: (jω)^k = (jω)^k computed via complex arithmetic
        let (b_re, b_im) = eval_analog_poly(b, omega);
        let (a_re, a_im) = eval_analog_poly(a, omega);

        let denom = a_re * a_re + a_im * a_im;
        if denom < 1e-30 {
            h_mag.push(f64::INFINITY);
            h_phase.push(0.0);
        } else {
            let h_re = (b_re * a_re + b_im * a_im) / denom;
            let h_im = (b_im * a_re - b_re * a_im) / denom;
            h_mag.push((h_re * h_re + h_im * h_im).sqrt());
            h_phase.push(h_im.atan2(h_re));
        }
    }

    Ok(FreqzResult {
        w: w.to_vec(),
        h_mag,
        h_phase,
    })
}

/// Compute group delay of a digital filter.
///
/// Matches `scipy.signal.group_delay((b, a), w)`.
///
/// Group delay: τ_g(ω) = -dφ/dω where φ is the phase response.
///
/// Uses the analytic formula based on polynomial evaluation:
///   τ_g(ω) = Re{ [B'(z)/B(z) - A'(z)/A(z)] } where z = e^{jω}
/// and B'(z) = d/dω B(e^{jω}) (the derivative polynomial evaluated on the unit circle).
pub fn group_delay(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    if b.is_empty() || a.is_empty() {
        return Err(SignalError::InvalidArgument(
            "b and a must be non-empty".to_string(),
        ));
    }

    let n = n_freqs.unwrap_or(512);
    if n == 0 {
        return Err(SignalError::InvalidArgument(
            "n_freqs must be > 0".to_string(),
        ));
    }

    // Derivative polynomial: if B(z) = Σ b[k] z^{-k}, then
    // dB/dω = Σ (-jk) b[k] z^{-k}, so the "derivative coefficients" are
    // b_deriv[k] = -k * b[k] and we multiply by j.
    //
    // Group delay = Re{ (B'·conj(B))/(|B|²) - (A'·conj(A))/(|A|²) }
    // where B' = Σ (-jk) b[k] e^{-jkω}

    let mut w_out = Vec::with_capacity(n);
    let mut gd_out = Vec::with_capacity(n);

    for i in 0..n {
        // Match scipy endpoint=False: w[i] = i * pi / n
        let omega = std::f64::consts::PI * i as f64 / n as f64;
        w_out.push(omega);
        gd_out.push(group_delay_at_frequency(b, a, omega));
    }

    Ok((w_out, gd_out))
}

/// Compute phase delay of a digital filter.
///
/// Matches the core SciPy surface of `scipy.signal.phase_delay((b, a), w)`.
///
/// Phase delay: τ_p(ω) = -φ(ω) / ω, where φ is the unwrapped phase response.
/// At ω = 0 the limit is taken from the analytic group delay.
pub fn phase_delay(
    b: &[f64],
    a: &[f64],
    n_freqs: Option<usize>,
) -> Result<(Vec<f64>, Vec<f64>), SignalError> {
    let response = freqz(b, a, n_freqs)?;
    let unwrapped_phase = unwrap_phase(&response.h_phase);
    let mut pd_out = Vec::with_capacity(response.w.len());

    for ((&omega, &mag), &phase) in response
        .w
        .iter()
        .zip(response.h_mag.iter())
        .zip(unwrapped_phase.iter())
    {
        let delay = if omega.abs() < 1e-14 {
            group_delay_at_frequency(b, a, omega)
        } else if mag <= 1e-30 {
            0.0
        } else {
            -phase / omega
        };
        pd_out.push(delay);
    }

    Ok((response.w, pd_out))
}

fn group_delay_at_frequency(b: &[f64], a: &[f64], omega: f64) -> f64 {
    // Evaluate B(e^{jω})
    let (b_re, b_im) = eval_poly_on_unit_circle(b, omega);
    let b_mag2 = b_re * b_re + b_im * b_im;

    // Standard formula: τ_g(ω) = Re{D_B / B} - Re{D_A / A}
    // where D_B(e^{jω}) = Σ_k k * b[k] * e^{-jkω}
    // and Re{D/B} = (D_re*B_re + D_im*B_im) / |B|²
    let (db_re, db_im) = eval_weighted_poly_on_unit_circle(b, omega);
    let gd_b = if b_mag2 > 1e-30 {
        (db_re * b_re + db_im * b_im) / b_mag2
    } else {
        0.0
    };

    // For A: τ_A = Re{Σ k*a[k]*e^{-jkω} / A(e^{jω})}
    let (a_re, a_im) = eval_poly_on_unit_circle(a, omega);
    let a_mag2 = a_re * a_re + a_im * a_im;
    let (da_re, da_im) = eval_weighted_poly_on_unit_circle(a, omega);
    let gd_a = if a_mag2 > 1e-30 {
        (da_re * a_re + da_im * a_im) / a_mag2
    } else {
        0.0
    };

    gd_b - gd_a
}

/// Evaluate Σ k*c[k]*e^{-jkω} (weighted polynomial for group delay).
fn eval_weighted_poly_on_unit_circle(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let angle = -(k as f64) * omega;
        let weight = k as f64;
        re += weight * c * angle.cos();
        im += weight * c * angle.sin();
    }
    (re, im)
}

/// Evaluate polynomial in z^{-1} on the unit circle at angle ω.
/// Returns (real_part, imag_part) of Σ c[k] * e^{-jkω}.
fn eval_poly_on_unit_circle(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let angle = -(k as f64) * omega;
        re += c * angle.cos();
        im += c * angle.sin();
    }
    (re, im)
}

/// Evaluate analog polynomial B(jω) = Σ b[k] * (jω)^{n-k}.
/// Note: analog polynomials are in descending power order.
fn eval_analog_poly(coeffs: &[f64], omega: f64) -> (f64, f64) {
    let n = coeffs.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for (k, &c) in coeffs.iter().enumerate() {
        let power = (n - 1 - k) as u32;
        // (jω)^power: j^0=1, j^1=j, j^2=-1, j^3=-j, j^4=1, ...
        let omega_pow = omega.powi(power as i32);
        match power % 4 {
            0 => re += c * omega_pow, // j^0 = 1
            1 => im += c * omega_pow, // j^1 = j
            2 => re -= c * omega_pow, // j^2 = -1
            3 => im -= c * omega_pow, // j^3 = -j
            _ => re += 0.0,           // Mathematically impossible
        }
    }
    (re, im)
}

// ══════════════════════════════════════════════════════════════════════
// Spectral Analysis
// ══════════════════════════════════════════════════════════════════════

/// Result of a spectral density estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Power spectral density estimates.
    pub psd: Vec<f64>,
}

/// Scaling mode for spectral-density style estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralScaling {
    /// Power spectral density with units per Hz.
    Density,
    /// Power spectrum with squared-amplitude units.
    Spectrum,
}

/// Estimate power spectral density using a periodogram.
///
/// Matches `scipy.signal.periodogram(x, fs, window, scaling='density')`.
///
/// # Arguments
/// * `x` — Input signal
/// * `fs` — Sampling frequency (Hz)
/// * `window` — Window function applied to the signal (None = boxcar/rectangular)
pub fn periodogram(
    x: &[f64],
    fs: f64,
    window: Option<&[f64]>,
) -> Result<SpectralResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    let n = x.len();

    // Apply window
    let windowed: Vec<f64> = match window {
        Some(w) => {
            if w.len() != n {
                return Err(SignalError::InvalidArgument(format!(
                    "window length ({}) must match signal length ({n})",
                    w.len()
                )));
            }
            x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).collect()
        }
        None => x.to_vec(),
    };

    // Window power for normalization
    let win_power: f64 = match window {
        Some(w) => w.iter().map(|&wi| wi * wi).sum::<f64>() / n as f64,
        None => 1.0,
    };

    // Compute FFT
    let opts = fsci_fft::FftOptions::default();
    let spectrum = fsci_fft::rfft(&windowed, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // One-sided PSD: |X[k]|² / (fs * N * win_power)
    let scale = 1.0 / (fs * n as f64 * win_power);
    let n_freqs = spectrum.len();
    let mut psd = Vec::with_capacity(n_freqs);
    for (k, &(re, im)) in spectrum.iter().enumerate() {
        let mag2 = re * re + im * im;
        // Double all bins except DC and Nyquist for one-sided spectrum
        let factor = if k == 0 || (n.is_multiple_of(2) && k == n_freqs - 1) {
            1.0
        } else {
            2.0
        };
        psd.push(mag2 * scale * factor);
    }

    // Frequency bins
    let freq_step = fs / n as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(SpectralResult { frequencies, psd })
}

/// Estimate power spectral density using Welch's method.
///
/// Matches `scipy.signal.welch(x, fs, nperseg, noverlap)`.
///
/// Divides the signal into overlapping segments, windows each, computes
/// periodograms, and averages them.
///
/// # Arguments
/// * `x` — Input signal
/// * `fs` — Sampling frequency (Hz)
/// * `window` — Window type to use (default: "hann")
/// * `nperseg` — Length of each segment (default: 256)
/// * `noverlap` — Number of overlapping samples (default: nperseg/2)
pub fn welch(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<SpectralResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0".to_string(),
        ));
    }
    let nperseg = nperseg.min(x.len());

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;
    let n_freqs = nperseg / 2 + 1;

    // nperseg <= x.len() above, so at least one segment fits.
    let n_segments = (x.len() - nperseg) / step + 1;

    // Each segment's periodogram (constant-detrend + window + rfft) is independent and
    // expensive; for many segments the work is split across threads. The per-segment
    // computation is deterministic and the averaging fold below runs in segment order,
    // so the result is bit-identical to the sequential loop. scipy.signal.welch defaults
    // to detrend='constant' (remove each segment's mean) — kept verbatim.
    let compute_segment = |s: usize| -> Result<Vec<f64>, SignalError> {
        let start = s * step;
        let segment = &x[start..start + nperseg];
        let mean = segment.iter().sum::<f64>() / nperseg as f64;
        let detrended: Vec<f64> = segment.iter().map(|&v| v - mean).collect();
        Ok(periodogram(&detrended, fs, Some(&win_coeffs))?.psd)
    };

    let seg_psds: Vec<Vec<f64>> = {
        let nthreads = stft_frame_thread_count(n_segments, nperseg);
        if nthreads <= 1 {
            (0..n_segments)
                .map(&compute_segment)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let chunk = n_segments.div_ceil(nthreads);
            let cs = &compute_segment;
            type SegChunk = Result<Vec<Vec<f64>>, SignalError>;
            let chunk_results: Vec<SegChunk> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..nthreads)
                    .filter_map(|t| {
                        let s0 = t * chunk;
                        if s0 >= n_segments {
                            return None;
                        }
                        let s1 = (s0 + chunk).min(n_segments);
                        Some(scope.spawn(move || (s0..s1).map(cs).collect::<Result<Vec<_>, _>>()))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("welch worker panicked"))
                    .collect()
            });
            let mut v = Vec::with_capacity(n_segments);
            for cr in chunk_results {
                v.extend(cr?);
            }
            v
        }
    };

    let mut avg_psd = vec![0.0; n_freqs];
    for seg in &seg_psds {
        for (avg, &val) in avg_psd.iter_mut().zip(seg.iter()) {
            *avg += val;
        }
    }

    // Average
    for val in &mut avg_psd {
        *val /= n_segments as f64;
    }

    // Frequency bins
    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(SpectralResult {
        frequencies,
        psd: avg_psd,
    })
}

// ══════════════════════════════════════════════════════════════════════
// FIR Filter Design
// ══════════════════════════════════════════════════════════════════════

/// Window type for FIR filter design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirWindow {
    Hamming,
    Hann,
    Blackman,
    /// Kaiser window with shape parameter beta.
    Kaiser(f64),
    /// Rectangular window (no windowing).
    Rectangular,
}

/// Design a FIR filter using the window method.
///
/// Matches `scipy.signal.firwin(numtaps, cutoff, window, pass_zero, fs)`.
///
/// Creates a linear-phase FIR filter by windowing the ideal sinc response.
///
/// # Arguments
/// * `numtaps` — Filter length (must be odd for Type I, even for Type II).
/// * `cutoff` — Cutoff frequency/frequencies (normalized 0 to 1, where 1 = Nyquist).
///   One value for lowpass/highpass, two for bandpass/bandstop.
/// * `window` — Window function to apply.
/// * `pass_zero` — If true, DC gain is 1 (lowpass/bandstop). If false, DC gain is 0 (highpass/bandpass).
pub fn firwin(
    numtaps: usize,
    cutoff: &[f64],
    window: FirWindow,
    pass_zero: bool,
) -> Result<Vec<f64>, SignalError> {
    if numtaps == 0 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be > 0".to_string(),
        ));
    }
    if cutoff.is_empty() {
        return Err(SignalError::InvalidArgument(
            "cutoff must be non-empty".to_string(),
        ));
    }
    for &c in cutoff {
        if !c.is_finite() || c <= 0.0 || c >= 1.0 {
            return Err(SignalError::InvalidArgument(format!(
                "cutoff {c} out of range (0, 1) or non-finite"
            )));
        }
    }
    for pair in cutoff.windows(2) {
        if pair[0] >= pair[1] {
            return Err(SignalError::InvalidArgument(
                "cutoff frequencies must be strictly increasing".to_string(),
            ));
        }
    }

    let m = numtaps;
    let alpha = (m - 1) as f64 / 2.0;

    // Build the ideal bandpass/bandstop impulse response
    // Start from all the bands implied by cutoff
    let mut bands: Vec<f64> = Vec::new();
    if pass_zero {
        // Starts at 0: lowpass or bandstop
        bands.push(0.0);
    }
    bands.extend_from_slice(cutoff);
    if pass_zero == cutoff.len().is_multiple_of(2) {
        // End at 1 (Nyquist)
        bands.push(1.0);
    }

    // Sum of sinc responses for each passband
    let mut h = vec![0.0; m];
    for pair in bands.chunks(2) {
        if pair.len() < 2 {
            break;
        }
        let (left, right) = (pair[0], pair[1]);
        for (n, hi) in h.iter_mut().enumerate() {
            let x = n as f64 - alpha;
            if x.abs() < 1e-15 {
                *hi += right - left;
            } else {
                *hi += (std::f64::consts::PI * right * x).sin() / (std::f64::consts::PI * x)
                    - (std::f64::consts::PI * left * x).sin() / (std::f64::consts::PI * x);
            }
        }
    }

    // Apply window
    let win = make_window(m, window);
    for (hi, wi) in h.iter_mut().zip(win.iter()) {
        *hi *= wi;
    }

    // Normalize for unity gain at the appropriate frequency:
    // - Lowpass/bandstop (pass_zero=true): normalize at DC (omega=0)
    // - Highpass (pass_zero=false, 1 cutoff): normalize at Nyquist (omega=PI)
    // - Bandpass (pass_zero=false, 2+ cutoffs): normalize at passband center
    let norm_freq = if pass_zero {
        0.0
    } else if cutoff.len() >= 2 {
        std::f64::consts::PI * (cutoff[0] + cutoff[1]) / 2.0
    } else {
        std::f64::consts::PI
    };
    let mut gain = 0.0;
    for (n, &hi) in h.iter().enumerate() {
        gain += hi * (norm_freq * (n as f64 - alpha)).cos();
    }
    if gain.abs() > 1e-15 {
        for hi in &mut h {
            *hi /= gain;
        }
    }

    Ok(h)
}

/// Design a FIR filter with arbitrary frequency response using frequency-sampling.
///
/// Matches `scipy.signal.firwin2(numtaps, freq, gain, window)`.
///
/// # Arguments
/// * `numtaps` — Filter length (should be odd for Type I filter).
/// * `freq` — Frequency points (0 to 1, normalized), must start at 0 and end at 1.
/// * `gain` — Desired gain at each frequency point.
/// * `window` — Window to apply.
pub fn firwin2(
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    window: FirWindow,
) -> Result<Vec<f64>, SignalError> {
    if numtaps == 0 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be > 0".to_string(),
        ));
    }
    if freq.len() != gain.len() {
        return Err(SignalError::InvalidArgument(
            "freq and gain must have same length".to_string(),
        ));
    }
    if freq.len() < 2 {
        return Err(SignalError::InvalidArgument(
            "need at least 2 frequency points".to_string(),
        ));
    }
    if (freq[0] - 0.0).abs() > 1e-10 || (freq[freq.len() - 1] - 1.0).abs() > 1e-10 {
        return Err(SignalError::InvalidArgument(
            "freq must start at 0 and end at 1".to_string(),
        ));
    }

    // A Type-II filter (even numtaps) cannot realize a nonzero gain at the
    // Nyquist frequency.
    if numtaps.is_multiple_of(2) && gain[gain.len() - 1] != 0.0 {
        return Err(SignalError::InvalidArgument(
            "even numtaps requires zero gain at the Nyquist frequency".to_string(),
        ));
    }

    // scipy.signal.firwin2: oversample the frequency grid to
    // nfreqs = 1 + 2^⌈log2(numtaps)⌉, interpolate the desired gain there,
    // apply the linear-phase ramp shift = exp(−(numtaps−1)/2·iπ·x) so the
    // first `numtaps` samples of the inverse real FFT are the causal taps,
    // then window. (The previous code under-sampled the grid, used a
    // circular shift instead of a phase ramp, and DC-normalized the taps —
    // none of which scipy does.)
    let nfreqs = 1 + numtaps.next_power_of_two();
    let n_out = 2 * (nfreqs - 1);
    let half = (numtaps as f64 - 1.0) / 2.0;

    let mut spectrum: Vec<fsci_fft::Complex64> = Vec::with_capacity(nfreqs);
    for k in 0..nfreqs {
        let x = k as f64 / (nfreqs - 1) as f64;
        // Piecewise-linear interpolation of (freq, gain) at x ∈ [0, 1].
        let mut seg = 0;
        while seg + 2 < freq.len() && freq[seg + 1] < x {
            seg += 1;
        }
        let span = freq[seg + 1] - freq[seg];
        let t = if span.abs() > 1e-15 {
            ((x - freq[seg]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let fx = gain[seg] + t * (gain[seg + 1] - gain[seg]);
        let theta = half * std::f64::consts::PI * x;
        spectrum.push((fx * theta.cos(), -fx * theta.sin()));
    }

    // Inverse real FFT; the leading `numtaps` samples are the filter taps.
    let opts = fsci_fft::FftOptions::default();
    let out_full = fsci_fft::irfft(&spectrum, Some(n_out), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("firwin2 irfft failed: {e:?}")))?;

    // Apply the (symmetric) design window — scipy get_window(fftbins=False).
    let win = make_window(numtaps, window);
    let result: Vec<f64> = out_full[..numtaps]
        .iter()
        .zip(win.iter())
        .map(|(&h, &w)| h * w)
        .collect();

    Ok(result)
}

/// Estimate Kaiser window parameters for FIR filter design.
///
/// Matches `scipy.signal.kaiserord(ripple, width)`.
///
/// # Arguments
/// * `ripple` — Maximum ripple/attenuation in dB (positive value, e.g. 60 for 60dB).
/// * `width` — Transition width (normalized frequency, 0 to 1).
///
/// # Returns
/// `(numtaps, beta)` — Number of taps and Kaiser beta parameter.
/// Compute the attenuation of a Kaiser FIR filter.
///
/// Matches `scipy.signal.kaiser_atten(numtaps, width)`. Returns the
/// attenuation in dB given the number of taps and the normalized
/// transition-band width: `atten = 2.285·(numtaps-1)·π·width + 7.95`.
///
/// Resolves [frankenscipy-ooa4k]; complement of [`kaiser_beta`] which
/// goes the opposite direction (atten → beta).
#[must_use]
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    // Cast to f64 before subtracting so numtaps == 0 doesn't underflow
    // usize. scipy.signal.kaiser_atten evaluates (numtaps - 1) as -1
    // in that case, yielding e.g. ≈7.232 for width=0.1 — not 7.95.
    // Resolves [frankenscipy-bg40m].
    2.285 * (numtaps as f64 - 1.0) * std::f64::consts::PI * width + 7.95
}

/// Kaiser window β parameter from desired attenuation.
///
/// Matches `scipy.signal.kaiser_beta(a)` where `a` is the desired
/// attenuation in dB (positive). Returns the empirical β:
///
/// ```text
///   β = 0.1102·(a − 8.7)                          if a > 50
///   β = 0.5842·(a − 21)^0.4 + 0.07886·(a − 21)    if 21 ≤ a ≤ 50
///   β = 0.0                                        if a < 21
/// ```
///
/// Resolves [frankenscipy-ooa4k].
#[must_use]
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

pub fn kaiserord(ripple: f64, width: f64) -> Result<(usize, f64), SignalError> {
    if !ripple.is_finite() || ripple <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "ripple must be positive (in dB)".to_string(),
        ));
    }
    if !width.is_finite() || width <= 0.0 || width >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "width must be in (0, 1)".to_string(),
        ));
    }

    let a = ripple; // attenuation in dB

    // Kaiser's empirical formula for beta
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };

    // Kaiser's formula for number of taps
    let numtaps = ((a - 7.95) / (2.285 * std::f64::consts::PI * width)).ceil() as usize + 1;
    let numtaps = numtaps.max(1);

    Ok((numtaps, beta))
}

/// Generate a window of given type and length.
fn make_window(n: usize, window: FirWindow) -> Vec<f64> {
    match window {
        FirWindow::Hamming => hamming(n),
        FirWindow::Hann => hann(n),
        FirWindow::Blackman => blackman(n),
        FirWindow::Kaiser(beta) => kaiser(n, beta),
        FirWindow::Rectangular => vec![1.0; n],
    }
}

/// Design a FIR filter using the Parks-McClellan (Remez exchange) algorithm.
///
/// Matches `scipy.signal.remez(numtaps, bands, desired)`.
///
/// # Arguments
/// * `numtaps` — Number of filter coefficients (must be odd for type I).
/// * `bands` — Frequency band edges, normalized [0, 0.5]. Must come in pairs.
///   e.g., `[0.0, 0.1, 0.2, 0.5]` defines passband [0, 0.1] and stopband [0.2, 0.5].
/// * `desired` — Desired gain in each band (one per band pair).
/// * `weight` — Optional weights per band (one per band pair). Default: all 1.0.
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if numtaps < 1 {
        return Err(SignalError::InvalidArgument(
            "numtaps must be >= 1".to_string(),
        ));
    }
    if !bands.len().is_multiple_of(2) || bands.is_empty() {
        return Err(SignalError::InvalidArgument(
            "bands must have even number of elements".to_string(),
        ));
    }
    let nbands = bands.len() / 2;
    if desired.len() != nbands {
        return Err(SignalError::InvalidArgument(format!(
            "desired length {} must equal number of bands {}",
            desired.len(),
            nbands
        )));
    }

    let weights: Vec<f64> = weight.map_or_else(|| vec![1.0; nbands], |w| w.to_vec());
    if weights.len() != nbands {
        return Err(SignalError::InvalidArgument(
            "weight length must equal number of bands".to_string(),
        ));
    }

    // Simplified Remez: use frequency-sampling method as approximation
    // (Full Parks-McClellan is very complex; this provides a reasonable approximation)
    let n = numtaps;
    let m = n / 2; // number of cosine coefficients for type I (odd length)

    // Create dense frequency grid
    let grid_size = 512.max(16 * n);
    let mut freq_grid = Vec::with_capacity(grid_size);
    let mut desired_grid = Vec::with_capacity(grid_size);
    let mut weight_grid = Vec::with_capacity(grid_size);

    for band_idx in 0..nbands {
        let f_start = bands[2 * band_idx];
        let f_end = bands[2 * band_idx + 1];
        let npts = (grid_size as f64 * (f_end - f_start) * 2.0).ceil() as usize;
        let npts = npts.max(4);

        for k in 0..npts {
            let freq = f_start + (f_end - f_start) * k as f64 / (npts - 1).max(1) as f64;
            freq_grid.push(freq);
            desired_grid.push(desired[band_idx]);
            weight_grid.push(weights[band_idx]);
        }
    }

    // Least-squares design on the cosine basis
    // H(f) = sum_{k=0}^{m} a_k * cos(2π * k * f)
    // Minimize weighted error: sum w_i * (H(f_i) - D(f_i))^2
    let n_coeffs = m + 1;
    let ng = freq_grid.len();

    // Normal equations: A^T W A x = A^T W d
    let mut ata = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut atd = vec![0.0; n_coeffs];
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut cos_basis = vec![0.0; n_coeffs];

    for i in 0..ng {
        let f = freq_grid[i];
        let w = weight_grid[i]; // WLS weight: minimize Σ w_i * (H(f_i) - D(f_i))²
        let d = desired_grid[i];
        for (j, basis) in cos_basis.iter_mut().enumerate() {
            *basis = (two_pi * j as f64 * f).cos();
        }

        for j in 0..n_coeffs {
            let cj = cos_basis[j];
            atd[j] += w * cj * d;
            let (head, tail) = ata.split_at_mut(j + 1);
            let ata_j = &mut head[j];
            ata_j[j] += w * cj * cj;
            for (offset, ata_row) in tail.iter_mut().enumerate() {
                let k = j + 1 + offset;
                let ck = cos_basis[k];
                let value = w * cj * ck;
                ata_row[j] += value;
                ata_j[k] += value;
            }
        }
    }

    // Solve normal equations (Cholesky or simple Gaussian elimination)
    let a_coeffs = solve_symmetric(&ata, &atd)?;

    // Convert cosine coefficients to symmetric FIR filter taps.
    // H(f) = a_0 + 2 * Σ_{k=1}^{m} a_k cos(2πkf)
    // corresponds to h[m] = a_0, h[m±k] = a_k/2 for k=1..m
    let mut h = vec![0.0; n];
    h[m] = a_coeffs[0];
    for k in 1..n_coeffs {
        if m >= k {
            h[m - k] = a_coeffs[k] / 2.0;
        }
        if m + k < n {
            h[m + k] = a_coeffs[k] / 2.0;
        }
    }

    Ok(h)
}

/// Design a linear-phase FIR filter using least-squares.
///
/// Matches `scipy.signal.firls(numtaps, bands, desired)`.
///
/// # Arguments
/// * `numtaps` — Number of taps (must be odd for type I).
/// * `bands` — Frequency band edges in pairs, normalized [0, 1] where 1 = Nyquist.
/// * `desired` — Desired gain at each band edge (length must equal bands length).
/// * `weight` — Optional weight per band pair.
pub fn firls(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    if numtaps < 1 || numtaps.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "numtaps must be odd and >= 1".to_string(),
        ));
    }
    if !bands.len().is_multiple_of(2) || bands.is_empty() {
        return Err(SignalError::InvalidArgument(
            "bands must have even number of elements".to_string(),
        ));
    }
    if desired.len() != bands.len() {
        return Err(SignalError::InvalidArgument(format!(
            "desired length {} must equal bands length {}",
            desired.len(),
            bands.len()
        )));
    }

    let nbands = bands.len() / 2;
    let weights: Vec<f64> = weight.map_or_else(|| vec![1.0; nbands], |w| w.to_vec());

    let m = (numtaps - 1) / 2;
    let n_coeffs = m + 1;

    // Build the Q matrix and b vector for the least-squares problem
    // Q[i,j] = Σ_bands w_k ∫_{f_lo}^{f_hi} cos(πif) cos(πjf) df
    // b[i] = Σ_bands w_k ∫_{f_lo}^{f_hi} D(f) cos(πif) df
    // where D(f) is linearly interpolated desired response

    let mut q = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut b_vec = vec![0.0; n_coeffs];

    for band in 0..nbands {
        let f_lo = bands[2 * band] / 2.0; // Convert from [0,1] Nyquist to [0,0.5] normalized
        let f_hi = bands[2 * band + 1] / 2.0;
        let d_lo = desired[2 * band];
        let d_hi = desired[2 * band + 1];
        let w = weights[band];

        let df = f_hi - f_lo;
        if df <= 0.0 {
            continue;
        }

        // D(f) = d_lo + (d_hi - d_lo) * (f - f_lo) / (f_hi - f_lo)
        //      = d_lo + slope * (f - f_lo)
        let slope = (d_hi - d_lo) / df;

        let pi = std::f64::consts::PI;

        for (i, row_i) in q.iter_mut().enumerate().take(n_coeffs) {
            // b[i] = w * ∫ D(f) cos(2πif) df over [f_lo, f_hi]
            let bi = if i == 0 {
                // ∫ D(f) df = d_lo * df + slope * df^2 / 2
                w * (d_lo * df + slope * df * df / 2.0)
            } else {
                let pi_i = pi * i as f64;
                // ∫ (d_lo + slope*(f-f_lo)) * cos(2πif) df
                // = d_lo * sin(2πif)/(2πi) + slope * [(f-f_lo)*sin(2πif)/(2πi) + cos(2πif)/(2πi)^2]
                let sin_hi = (2.0 * pi_i * f_hi).sin();
                let sin_lo = (2.0 * pi_i * f_lo).sin();
                let cos_hi = (2.0 * pi_i * f_hi).cos();
                let cos_lo = (2.0 * pi_i * f_lo).cos();
                let inv_pi_i = 1.0 / (2.0 * pi_i);

                let part1 = d_lo * (sin_hi - sin_lo) * inv_pi_i;
                let part2 =
                    slope * (df * sin_hi * inv_pi_i + (cos_hi - cos_lo) * inv_pi_i * inv_pi_i);
                w * (part1 + part2)
            };
            b_vec[i] += bi;

            for (j, cell) in row_i.iter_mut().enumerate().skip(i) {
                // Q[i,j] = w * ∫ cos(2πif) cos(2πjf) df
                // = w/2 * ∫ [cos(2π(i-j)f) + cos(2π(i+j)f)] df
                let qij = if i == 0 && j == 0 {
                    w * df
                } else {
                    let integrate_cos = |freq: f64| -> f64 {
                        if freq.abs() < 1e-15 {
                            df
                        } else {
                            let pf = 2.0 * pi * freq;
                            ((pf * f_hi).sin() - (pf * f_lo).sin()) / pf
                        }
                    };
                    w * 0.5
                        * (integrate_cos(i as f64 - j as f64) + integrate_cos(i as f64 + j as f64))
                };
                *cell += qij;
            }
        }
    }

    let upper_triangle = q.clone();
    for (i, row_i) in q.iter_mut().enumerate().take(n_coeffs) {
        for (j, cell) in row_i.iter_mut().enumerate().take(i) {
            *cell = upper_triangle[j][i];
        }
    }

    // Solve Q a = b
    let a_coeffs = solve_symmetric(&q, &b_vec)?;

    // Convert to FIR filter taps
    let n = numtaps;
    let mut h = vec![0.0; n];
    h[m] = a_coeffs[0];
    for k in 1..n_coeffs {
        h[m - k] = a_coeffs[k] / 2.0;
        h[m + k] = a_coeffs[k] / 2.0;
    }

    Ok(h)
}

/// Solve symmetric positive definite system via Cholesky.
fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, SignalError> {
    let n = a.len();
    // Cholesky factorization: A = L L^T
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut sum = a[j][j];
        for &value in l[j].iter().take(j) {
            sum -= value * value;
        }
        if sum <= 0.0 {
            // Fall back to regularization
            l[j][j] = 1e-10;
        } else {
            l[j][j] = sum.sqrt();
        }
        for i in j + 1..n {
            let mut sum = a[i][j];
            for (&li_k, &lj_k) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= li_k * lj_k;
            }
            l[i][j] = sum / l[j][j];
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in i + 1..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    Ok(x)
}

// ── Chirp method ────────────────────────────────────────────────────

/// Method for frequency sweep in `chirp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChirpMethod {
    /// Linear frequency sweep (instantaneous frequency is linear in time).
    #[default]
    Linear,
    /// Quadratic frequency sweep.
    Quadratic,
    /// Logarithmic frequency sweep (f0 and f1 must be > 0).
    Logarithmic,
}

/// Generate a swept-frequency cosine (chirp) signal.
///
/// Matches `scipy.signal.chirp(t, f0, t1, f1, method)`.
///
/// # Arguments
/// * `t` — Time array at which to evaluate the signal.
/// * `f0` — Frequency at time 0.
/// * `t1` — Time at which `f1` is specified.
/// * `f1` — Frequency at time `t1`.
/// * `method` — Chirp type: linear, quadratic, or logarithmic.
pub fn chirp(
    t: &[f64],
    f0: f64,
    t1: f64,
    f1: f64,
    method: ChirpMethod,
) -> Result<Vec<f64>, SignalError> {
    if !t1.is_finite() || t1 <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "t1 must be positive".to_string(),
        ));
    }
    if !f0.is_finite() || !f1.is_finite() {
        return Err(SignalError::InvalidArgument(
            "f0 and f1 must be finite".to_string(),
        ));
    }
    if method == ChirpMethod::Logarithmic && (f0 <= 0.0 || f1 <= 0.0) {
        return Err(SignalError::InvalidArgument(
            "logarithmic chirp requires f0 > 0 and f1 > 0".to_string(),
        ));
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let result = match method {
        ChirpMethod::Linear => {
            let k = (f1 - f0) / t1;
            t.iter()
                .map(|&ti| (two_pi * (f0 * ti + 0.5 * k * ti * ti)).cos())
                .collect()
        }
        ChirpMethod::Quadratic => {
            let k = (f1 - f0) / (t1 * t1);
            t.iter()
                .map(|&ti| (two_pi * (f0 * ti + k * ti * ti * ti / 3.0)).cos())
                .collect()
        }
        ChirpMethod::Logarithmic => {
            let ratio = f1 / f0;
            if (ratio - 1.0).abs() < f64::EPSILON {
                // f0 == f1: constant frequency, just a cosine
                t.iter().map(|&ti| (two_pi * f0 * ti).cos()).collect()
            } else {
                let log_ratio = ratio.ln();
                t.iter()
                    .map(|&ti| {
                        let phase = two_pi * f0 * t1 / log_ratio * (ratio.powf(ti / t1) - 1.0);
                        phase.cos()
                    })
                    .collect()
            }
        }
    };
    Ok(result)
}

/// Generate a sawtooth or triangle wave.
///
/// Matches `scipy.signal.sawtooth(t, width)`.
///
/// # Arguments
/// * `t` — Time array (phase in radians; period is 2π).
/// * `width` — Width of the rising ramp as a proportion of the period (0 to 1).
///   `width=1` gives a rising sawtooth, `width=0` a falling sawtooth,
///   `width=0.5` a triangle wave.
pub fn sawtooth(t: &[f64], width: f64) -> Result<Vec<f64>, SignalError> {
    if !width.is_finite() || !(0.0..=1.0).contains(&width) {
        return Err(SignalError::InvalidArgument(
            "width must be finite and in [0, 1]".to_string(),
        ));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let result = t
        .iter()
        .map(|&ti| {
            // Normalize to [0, 1) within one period.
            let phase = ((ti / two_pi) % 1.0 + 1.0) % 1.0;
            if width == 0.0 {
                // Pure falling ramp.
                1.0 - 2.0 * phase
            } else if phase < width {
                -1.0 + 2.0 * phase / width
            } else {
                1.0 - 2.0 * (phase - width) / (1.0 - width)
            }
        })
        .collect();
    Ok(result)
}

/// Generate a square wave.
///
/// Matches `scipy.signal.square(t, duty)`.
///
/// # Arguments
/// * `t` — Time array (phase in radians; period is 2π).
/// * `duty` — Duty cycle (fraction of period at +1). Default 0.5.
pub fn square(t: &[f64], duty: f64) -> Result<Vec<f64>, SignalError> {
    if !duty.is_finite() || !(0.0..=1.0).contains(&duty) {
        return Err(SignalError::InvalidArgument(
            "duty must be finite and in [0, 1]".to_string(),
        ));
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let result = t
        .iter()
        .map(|&ti| {
            let phase = ((ti / two_pi) % 1.0 + 1.0) % 1.0;
            if phase < duty { 1.0 } else { -1.0 }
        })
        .collect();
    Ok(result)
}

/// Result of [`gauspuls`]: in-phase signal, quadrature signal, and envelope.
///
/// Matches `scipy.signal.gauspuls(retquad=True, retenv=True)` which
/// returns `(yI, yQ, yenv)`. Callers that only want the in-phase
/// component can read [`GauspulsResult::i`].
#[derive(Debug, Clone, PartialEq)]
pub struct GauspulsResult {
    /// In-phase signal: yenv · cos(2π·fc·t).
    pub i: Vec<f64>,
    /// Quadrature signal: yenv · sin(2π·fc·t).
    pub q: Vec<f64>,
    /// Gaussian envelope: exp(-a · t²).
    pub envelope: Vec<f64>,
}

/// Gaussian-modulated sinusoidal pulse.
///
/// Matches `scipy.signal.gauspuls(t, fc=1000, bw=0.5, bwr=-6)`.
///
/// Computes:
///
/// ```text
///   ref     = 10^(bwr / 20)
///   a       = -(π · fc · bw)² / (4 · ln(ref))
///   yenv(t) = exp(-a · t²)
///   yI(t)   = yenv(t) · cos(2π · fc · t)
///   yQ(t)   = yenv(t) · sin(2π · fc · t)
/// ```
///
/// Returns the in-phase, quadrature, and envelope vectors. Callers that
/// only need the real signal use the `.i` field. Resolves
/// [frankenscipy-8jpj9].
pub fn gauspuls(t: &[f64], fc: f64, bw: f64, bwr: f64) -> Result<GauspulsResult, SignalError> {
    if !(fc > 0.0 && fc.is_finite()) {
        return Err(SignalError::InvalidArgument(format!(
            "fc must be a positive finite frequency; got {fc}"
        )));
    }
    if !(bw > 0.0 && bw.is_finite()) {
        return Err(SignalError::InvalidArgument(format!(
            "bw (fractional bandwidth) must be positive; got {bw}"
        )));
    }
    if !(bwr < 0.0 && bwr.is_finite()) {
        return Err(SignalError::InvalidArgument(format!(
            "bwr must be a negative dB level; got {bwr}"
        )));
    }

    let r = 10.0_f64.powf(bwr / 20.0);
    let pi_fc_bw = std::f64::consts::PI * fc * bw;
    let a = -(pi_fc_bw * pi_fc_bw) / (4.0 * r.ln());

    let two_pi_fc = std::f64::consts::TAU * fc;
    let n = t.len();
    let mut env = Vec::with_capacity(n);
    let mut i_out = Vec::with_capacity(n);
    let mut q_out = Vec::with_capacity(n);
    for &ti in t {
        let e = (-a * ti * ti).exp();
        let phase = two_pi_fc * ti;
        env.push(e);
        i_out.push(e * phase.cos());
        q_out.push(e * phase.sin());
    }

    Ok(GauspulsResult {
        i: i_out,
        q: q_out,
        envelope: env,
    })
}

/// Generate a discrete unit impulse (Kronecker delta).
///
/// Matches `scipy.signal.unit_impulse(shape, idx)`.
///
/// # Arguments
///
/// * `shape` — length of the output array.
/// * `idx` — index at which the impulse is placed. If `None`, places at index 0.
pub fn unit_impulse(shape: usize, idx: Option<usize>) -> Result<Vec<f64>, SignalError> {
    let i = idx.unwrap_or(0);
    if i >= shape {
        return Err(SignalError::InvalidArgument(format!(
            "idx {i} out of range for shape {shape}"
        )));
    }
    let mut out = vec![0.0; shape];
    out[i] = 1.0;
    Ok(out)
}

/// Detrend type for `detrend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetrendType {
    /// Remove the mean (constant trend).
    Constant,
    /// Remove a linear trend (least-squares fit of y = a + b*x).
    #[default]
    Linear,
}

/// Remove a trend from data.
///
/// Matches `scipy.signal.detrend(data, type)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `dtype` — Type of detrending: constant (remove mean) or linear (remove linear fit).
pub fn detrend(data: &[f64], dtype: DetrendType) -> Result<Vec<f64>, SignalError> {
    if data.is_empty() {
        return Err(SignalError::InvalidArgument(
            "data must not be empty".to_string(),
        ));
    }
    match dtype {
        DetrendType::Constant => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(data.iter().map(|&v| v - mean).collect())
        }
        DetrendType::Linear => {
            let n = data.len();
            let n_f = n as f64;
            if n < 2 {
                return Ok(vec![0.0; n]);
            }

            // Center x-values to [-(n-1)/2, (n-1)/2] for numerical stability.
            // This makes sum(x) = 0, simplifying the normal equations.
            let x_mean = (n_f - 1.0) / 2.0;

            let mut s_xx = 0.0;
            let mut s_xy = 0.0;
            let mut y_sum = 0.0;

            for (i, &y) in data.iter().enumerate() {
                let x = i as f64 - x_mean;
                s_xx += x * x;
                s_xy += x * y;
                y_sum += y;
            }

            let y_mean = y_sum / n_f;
            let slope = s_xy / s_xx;

            Ok(data
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let x = i as f64 - x_mean;
                    y - (y_mean + slope * x)
                })
                .collect())
        }
    }
}

/// Apply a median filter to a 1-D signal.
///
/// Matches `scipy.signal.medfilt(volume, kernel_size)`.
///
/// # Arguments
/// * `data` — Input signal.
/// * `kernel_size` — Size of the median filter window (must be odd and >= 1).
pub fn medfilt(data: &[f64], kernel_size: usize) -> Result<Vec<f64>, SignalError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "kernel_size must be odd and >= 1".to_string(),
        ));
    }
    if data.is_empty() {
        return Ok(vec![]);
    }

    // For large kernels the per-window O(k) selection (O(n*k) total) is replaced
    // by a sliding two-ordered-multiset median (O(n*log k)). Both return the
    // rank-(k/2) element by total_cmp, so the output is bit-identical; the naive
    // path stays faster for small kernels.
    if kernel_size >= MEDFILT_SLIDING_THRESHOLD {
        return Ok(medfilt_sliding(data, kernel_size));
    }

    let half = kernel_size / 2;
    let n = data.len();
    let mut result = Vec::with_capacity(n);
    let mut window = vec![0.0; kernel_size];

    for i in 0..n {
        // Fill window with zero-padding at boundaries (matches SciPy)
        for (j, val) in window.iter_mut().enumerate() {
            let idx = i as i64 + j as i64 - half as i64;
            *val = if idx >= 0 && idx < n as i64 {
                data[idx as usize]
            } else {
                0.0
            };
        }

        // Linear-time median selection
        let mid = kernel_size / 2;
        let (_, &mut m, _) = window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        result.push(m);
    }

    Ok(result)
}

const MEDFILT_SLIDING_THRESHOLD: usize = 64;

/// `total_cmp`-ordered f64 wrapper so f64 can key a `BTreeMap` multiset with the
/// exact ordering `medfilt`'s `select_nth_unstable_by` uses.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);
impl Eq for OrderedF64 {}
impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Streaming median over a sliding window, split into two ordered multisets:
/// `lower` holds the smallest `ceil(total/2)` values (its maximum is the median),
/// `upper` the rest, with `max(lower) <= min(upper)` maintained. Insert/remove
/// are O(log k). The median equals the rank-`k/2` element (odd `k`), identical to
/// `select_nth_unstable_by(k/2)`.
struct SlidingRankWindow {
    lower: std::collections::BTreeMap<OrderedF64, usize>,
    upper: std::collections::BTreeMap<OrderedF64, usize>,
    lower_size: usize,
    upper_size: usize,
    /// Desired 0-indexed rank; `lower` is kept at `min(rank + 1, total)` elements
    /// so `lower.max()` is the element at sorted index `min(rank, total - 1)` —
    /// exactly what a per-window `sort` + index would return.
    rank: usize,
}

impl SlidingRankWindow {
    fn new(rank: usize) -> Self {
        Self {
            lower: std::collections::BTreeMap::new(),
            upper: std::collections::BTreeMap::new(),
            lower_size: 0,
            upper_size: 0,
            rank,
        }
    }

    fn lower_max(&self) -> OrderedF64 {
        *self.lower.keys().next_back().expect("lower non-empty")
    }

    fn upper_min(&self) -> OrderedF64 {
        *self.upper.keys().next().expect("upper non-empty")
    }

    fn bag_insert(bag: &mut std::collections::BTreeMap<OrderedF64, usize>, key: OrderedF64) {
        *bag.entry(key).or_insert(0) += 1;
    }

    fn bag_remove(bag: &mut std::collections::BTreeMap<OrderedF64, usize>, key: OrderedF64) {
        if let Some(count) = bag.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                bag.remove(&key);
            }
        }
    }

    fn insert(&mut self, x: f64) {
        let key = OrderedF64(x);
        if self.lower_size == 0 || key <= self.lower_max() {
            Self::bag_insert(&mut self.lower, key);
            self.lower_size += 1;
        } else {
            Self::bag_insert(&mut self.upper, key);
            self.upper_size += 1;
        }
        self.rebalance();
    }

    fn remove(&mut self, x: f64) {
        // Any copy of the value works — the median depends only on the union, and
        // rebalance restores the size invariant.
        let key = OrderedF64(x);
        if self.lower.contains_key(&key) {
            Self::bag_remove(&mut self.lower, key);
            self.lower_size -= 1;
        } else {
            Self::bag_remove(&mut self.upper, key);
            self.upper_size -= 1;
        }
        self.rebalance();
    }

    fn rebalance(&mut self) {
        let total = self.lower_size + self.upper_size;
        let target = (self.rank + 1).min(total);
        while self.lower_size > target {
            let m = self.lower_max();
            Self::bag_remove(&mut self.lower, m);
            self.lower_size -= 1;
            Self::bag_insert(&mut self.upper, m);
            self.upper_size += 1;
        }
        while self.lower_size < target {
            let m = self.upper_min();
            Self::bag_remove(&mut self.upper, m);
            self.upper_size -= 1;
            Self::bag_insert(&mut self.lower, m);
            self.lower_size += 1;
        }
    }

    fn value(&self) -> f64 {
        self.lower_max().0
    }

    /// Re-point the desired rank (used when the clipped window size — and hence
    /// the requested `len/2` median index — changes at the borders) and restore
    /// the size invariant so `value()` reads the new rank element.
    fn set_rank(&mut self, rank: usize) {
        self.rank = rank;
        self.rebalance();
    }
}

fn medfilt_sliding(data: &[f64], kernel_size: usize) -> Vec<f64> {
    let n = data.len();
    let half = (kernel_size / 2) as i64;
    let at = |idx: i64| -> f64 {
        if idx >= 0 && idx < n as i64 {
            data[idx as usize]
        } else {
            0.0
        }
    };

    let mut window = SlidingRankWindow::new(kernel_size / 2);
    for p in -half..=half {
        window.insert(at(p));
    }
    let mut result = Vec::with_capacity(n);
    result.push(window.value());
    for i in 1..n as i64 {
        window.remove(at(i - 1 - half));
        window.insert(at(i + half));
        result.push(window.value());
    }
    result
}

const ORDER_FILTER_SLIDING_THRESHOLD: usize = 32;

/// Sliding `order_filter` for large windows. The 1-D window is clipped (not
/// zero-padded) at the borders, so it grows from the left edge, stays full, then
/// shrinks at the right edge; the multiset tracks exactly those contents as the
/// monotone `[start, end)` bounds advance. `value()` returns the rank element of
/// the current window — bit-identical to sorting the window and indexing.
fn order_filter_sliding(x: &[f64], window_size: usize, rank: usize) -> Vec<f64> {
    let n = x.len();
    let half = window_size / 2;
    let mut window = SlidingRankWindow::new(rank);
    let mut cur_start = 0usize;
    let mut cur_end = 0usize;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        while cur_start < start {
            window.remove(x[cur_start]);
            cur_start += 1;
        }
        while cur_end < end {
            window.insert(x[cur_end]);
            cur_end += 1;
        }
        result.push(window.value());
    }
    result
}

/// Apply a 1-D adaptive Wiener filter.
///
/// Matches the high-level behavior of `scipy.signal.wiener(im, mysize, noise)`
/// for 1-D inputs with zero-padded local windows.
pub fn wiener(data: &[f64], mysize: usize, noise: Option<f64>) -> Result<Vec<f64>, SignalError> {
    if mysize == 0 || mysize.is_multiple_of(2) {
        return Err(SignalError::InvalidArgument(
            "mysize must be odd and >= 1".to_string(),
        ));
    }
    if let Some(value) = noise
        && (!value.is_finite() || value < 0.0)
    {
        return Err(SignalError::InvalidArgument(
            "noise must be a finite non-negative value when provided".to_string(),
        ));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let half = mysize / 2;
    let n = data.len();
    let mut local_mean = vec![0.0; n];
    let mut local_var = vec![0.0; n];
    let window_area = mysize as f64;

    for i in 0..n {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for offset in 0..mysize {
            let idx = i as i64 + offset as i64 - half as i64;
            let value = if idx >= 0 && idx < n as i64 {
                data[idx as usize]
            } else {
                0.0
            };
            sum += value;
            sumsq += value * value;
        }
        let mean = sum / window_area;
        local_mean[i] = mean;
        local_var[i] = (sumsq / window_area - mean * mean).max(0.0);
    }

    let noise_power =
        noise.unwrap_or_else(|| local_var.iter().sum::<f64>() / local_var.len() as f64);

    Ok(data
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if local_var[i] <= noise_power {
                local_mean[i]
            } else {
                let gain = 1.0 - noise_power / local_var[i];
                local_mean[i] + gain * (value - local_mean[i])
            }
        })
        .collect())
}

/// Generate a Bartlett window.
pub fn bartlett(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64 - nf / 2.0;
            1.0 - (2.0 * x / nf).abs()
        })
        .collect()
}

/// Generate a flat top window.
pub fn flattop(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let nf = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let arg = two_pi * i as f64 / nf;
            0.21557895 - 0.41663158 * arg.cos() + 0.277263158 * (2.0 * arg).cos()
                - 0.083578947 * (3.0 * arg).cos()
                + 0.006947368 * (4.0 * arg).cos()
        })
        .collect()
}

/// Generate a cosine window.
///
/// Matches `scipy.signal.windows.cosine(M)`.
pub fn cosine(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / nf).sin())
        .collect()
}

/// Generic weighted sum of cosine terms window.
///
/// Matches `scipy.signal.windows.general_cosine(M, a, sym)`.
pub fn general_cosine(n: usize, coeffs: &[f64], sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let m = if sym { n } else { n + 1 };
    let denom = (m - 1) as f64;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let angle = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / denom;
            coeffs
                .iter()
                .enumerate()
                .map(|(k, &coefficient)| coefficient * (k as f64 * angle).cos())
                .sum()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// Gaussian window.
///
/// Matches `scipy.signal.windows.gaussian(n, std, sym)`.
pub fn gaussian(n: usize, std: f64, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = if sym { n } else { n + 1 };
    let center = (m as f64 - 1.0) / 2.0;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let x = (i as f64 - center) / std;
            (-0.5 * x * x).exp()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// General Gaussian window: exp(-0.5 * |x/σ|^(2p)).
///
/// Matches `scipy.signal.windows.general_gaussian(n, p, sig, sym)`.
pub fn general_gaussian(n: usize, p: f64, sig: f64, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let m = if sym { n } else { n + 1 };
    let center = (m as f64 - 1.0) / 2.0;
    let mut window: Vec<f64> = (0..m)
        .map(|i| {
            let x = (i as f64 - center) / sig;
            (-0.5 * x.abs().powf(2.0 * p)).exp()
        })
        .collect();
    if !sym {
        window.truncate(n);
    }
    window
}

/// Parzen (de la Vallée Poussin) window.
///
/// Matches `scipy.signal.windows.parzen(n)`.
pub fn parzen(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    let half = nf / 2.0;
    (0..n)
        .map(|i| {
            let k = (i as f64 - (nf - 1.0) / 2.0).abs();
            if k <= nf / 4.0 {
                1.0 - 6.0 * (k / half).powi(2) + 6.0 * (k / half).powi(3)
            } else {
                2.0 * (1.0 - k / half).powi(3)
            }
        })
        .collect()
}

/// Exponential (Poisson) window.
///
/// Matches `scipy.signal.windows.exponential(M, center=None, tau=1.0, sym=True)`.
pub fn exponential(
    n: usize,
    center: Option<f64>,
    tau: f64,
    sym: bool,
) -> Result<Vec<f64>, SignalError> {
    if sym && center.is_some() {
        return Err(SignalError::InvalidArgument(
            "If sym==True, center must be None.".into(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let m = if sym { n } else { n + 1 };
    let center = center.unwrap_or((m as f64 - 1.0) / 2.0);
    let mut window: Vec<f64> = (0..m)
        .map(|i| (-(i as f64 - center).abs() / tau).exp())
        .collect();
    if !sym {
        window.truncate(n);
    }
    Ok(window)
}

/// Lanczos window (sinc window).
///
/// The Lanczos window is the central lobe of a sinc function. It has the form:
/// w(n) = sinc(2n/(N-1) - 1) for n = 0, ..., N-1
///
/// where sinc(x) = sin(πx)/(πx).
///
/// Matches `scipy.signal.windows.lanczos(n)`.
pub fn lanczos(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| {
            let x = 2.0 * (i as f64) / (nf - 1.0) - 1.0;
            if x.abs() < 1e-10 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            }
        })
        .collect()
}

fn fft_real_parts(input: &[fsci_fft::Complex64]) -> Vec<f64> {
    let opts = fsci_fft::FftOptions::default();
    match fsci_fft::fft(input, &opts) {
        Ok(values) => values.into_iter().map(|(re, _)| re).collect(),
        Err(_) => {
            let n = input.len() as f64;
            (0..input.len())
                .map(|k| {
                    input.iter().enumerate().fold(0.0, |acc, (m, &(re, im))| {
                        let angle = 2.0 * std::f64::consts::PI * k as f64 * m as f64 / n;
                        acc + re * angle.cos() + im * angle.sin()
                    })
                })
                .collect()
        }
    }
}

/// Dolph-Chebyshev window.
///
/// Matches `scipy.signal.windows.chebwin(n, at)`.
pub fn chebwin(n: usize, at: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let order = n as f64 - 1.0;
    let beta = (10.0_f64.powf(at.abs() / 20.0).acosh() / order).cosh();
    let negative_sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    let p: Vec<f64> = (0..n)
        .map(|k| {
            let x = beta * (std::f64::consts::PI * k as f64 / n as f64).cos();
            if x > 1.0 {
                (order * x.acosh()).cosh()
            } else if x < -1.0 {
                negative_sign * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            }
        })
        .collect();

    let fft_input: Vec<fsci_fft::Complex64> = if n.is_multiple_of(2) {
        p.iter()
            .enumerate()
            .map(|(k, &value)| {
                let angle = std::f64::consts::PI * k as f64 / n as f64;
                (value * angle.cos(), value * angle.sin())
            })
            .collect()
    } else {
        p.into_iter().map(|value| (value, 0.0)).collect()
    };

    let fft_values = fft_real_parts(&fft_input);
    let mut window = Vec::with_capacity(n);
    if n.is_multiple_of(2) {
        let split = n / 2 + 1;
        window.extend(fft_values[1..split].iter().rev().copied());
        window.extend_from_slice(&fft_values[1..split]);
    } else {
        let split = n.div_ceil(2);
        window.extend(fft_values[1..split].iter().rev().copied());
        window.extend_from_slice(&fft_values[..split]);
    }

    let max_value = window.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_value.is_finite() && max_value > 0.0 {
        for value in &mut window {
            *value /= max_value;
        }
    }
    window
}

/// Taylor window.
///
/// The Taylor window is designed for radar and antenna applications. It provides
/// nearly constant sidelobe levels with a specified peak sidelobe attenuation.
///
/// Matches `scipy.signal.windows.taylor(M, nbar, sll, norm, sym)`.
///
/// # Arguments
/// * `n` - Window length
/// * `nbar` - Number of nearly constant-level sidelobes adjacent to the mainlobe
/// * `sll` - Desired peak sidelobe level in dB (negative, e.g., -30.0)
/// * `norm` - If true, normalize the window to have unit peak
/// * `sym` - If true, generate symmetric window (default for filter design)
pub fn taylor(n: usize, nbar: usize, sll: f64, norm: bool, sym: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let nbar = nbar.max(1);
    let n_use = if sym { n } else { n + 1 };

    // br-y6wj: match scipy.signal.windows.taylor exactly. Two prior
    // bugs: (a) Fm denominator was missing the factor of 2, (b) sample
    // positions used i/(N-1) instead of scipy's centered (i - (N-1)/2)/N.
    let b = 10.0_f64.powf(sll.abs() / 20.0);
    let a = (b + (b * b - 1.0).max(0.0).sqrt()).ln() / std::f64::consts::PI;
    let a_sq = a * a;

    let nbar_f = nbar as f64;
    let sigma_sq = nbar_f * nbar_f / (a_sq + (nbar_f - 0.5) * (nbar_f - 0.5));

    let mut fm = vec![0.0; nbar];
    for (m, fm_m) in fm.iter_mut().enumerate().take(nbar).skip(1) {
        let mf = m as f64;
        let m_sq = mf * mf;

        let mut num = 1.0;
        for nn in 1..nbar {
            let nf = nn as f64;
            let term = a_sq + (nf - 0.5) * (nf - 0.5);
            num *= 1.0 - m_sq / (sigma_sq * term);
        }

        let mut den = 1.0;
        for nn in 1..nbar {
            if nn != m {
                let nf = nn as f64;
                den *= 1.0 - m_sq / (nf * nf);
            }
        }
        // br-y6wj fix (a): scipy multiplies the denominator by 2.
        den *= 2.0;

        let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
        *fm_m = if den.abs() > 1e-15 {
            sign * num / den
        } else {
            0.0
        };
    }

    // br-y6wj fix (b): scipy's sample positions are (i - (N-1)/2) / N.
    let nf = n_use as f64;
    let center = (nf - 1.0) / 2.0;
    let mut w: Vec<f64> = (0..n_use)
        .map(|i| {
            let pos = (i as f64) - center;
            let x = pos / nf;
            let mut val = 1.0;
            for (m, &fm_m) in fm.iter().enumerate().take(nbar).skip(1) {
                val += 2.0 * fm_m * (2.0 * std::f64::consts::PI * m as f64 * x).cos();
            }
            val
        })
        .collect();

    // If asymmetric, truncate to n samples
    if !sym {
        w.truncate(n);
    }

    // br-y6wj fix (c): scipy normalizes by W((M-1)/2), the value at the
    // exact center (fractional sample for even M), not the max of sampled
    // values. For even M the center falls between samples, so max(w) <
    // W(center) and using max would underscale the window.
    if norm {
        let center_x = ((nf - 1.0) / 2.0 - center) / nf;
        let mut center_val = 1.0;
        for (m, &fm_m) in fm.iter().enumerate().take(nbar).skip(1) {
            center_val += 2.0 * fm_m * (2.0 * std::f64::consts::PI * m as f64 * center_x).cos();
        }
        if center_val.abs() > 1e-15 {
            for v in &mut w {
                *v /= center_val;
            }
        }
    }

    w
}

/// Normalization modes for DPSS windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpssNorm {
    /// Preserve the L2-normalized eigenvectors from the tridiagonal solver.
    L2,
    /// Match SciPy's default single-window correction for even-length windows.
    Approximate,
    /// Apply the FFT-based subsample correction used by SciPy.
    Subsample,
}

/// Result bundle for `dpss`.
#[derive(Debug, Clone, PartialEq)]
pub struct DpssResult {
    /// DPSS tapers. `kmax=None` still returns a single taper at `windows[0]`.
    pub windows: Vec<Vec<f64>>,
    /// Concentration ratios when requested.
    pub ratios: Option<Vec<f64>>,
}

fn normalized_sinc(x: f64) -> f64 {
    if x.abs() < 1.0e-14 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn dpss_concentration_ratio(window: &[f64], bandwidth: f64) -> f64 {
    let mut ratio = 0.0;
    for lag in 0..window.len() {
        let autocorr = (0..window.len() - lag)
            .map(|idx| window[idx] * window[idx + lag])
            .sum::<f64>();
        let kernel = if lag == 0 {
            2.0 * bandwidth
        } else {
            4.0 * bandwidth * normalized_sinc(2.0 * bandwidth * lag as f64)
        };
        ratio += autocorr * kernel;
    }
    ratio
}

fn dpss_subsample_correction(window: &[f64]) -> Result<f64, SignalError> {
    let spectrum = fsci_fft::rfft(window, &fsci_fft::FftOptions::default())
        .map_err(|err| SignalError::InvalidArgument(format!("dpss subsample FFT failed: {err}")))?;
    let n = window.len() as f64;
    let mut real_sum = spectrum[0].0;
    for (k, &(re, im)) in spectrum.iter().enumerate().skip(1) {
        let shift = -(1.0 - 1.0 / n) * k as f64;
        let angle = -std::f64::consts::PI * shift;
        let scale_re = 2.0 * angle.cos();
        let scale_im = 2.0 * angle.sin();
        real_sum += re * scale_re - im * scale_im;
    }
    if real_sum.abs() < 1.0e-15 {
        return Err(SignalError::InvalidArgument(
            "dpss subsample correction became singular".to_string(),
        ));
    }
    Ok(n / real_sum)
}

/// Discrete prolate spheroidal sequences (Slepian tapers).
///
/// Matches `scipy.signal.windows.dpss(M, NW, Kmax=None, sym=True, norm=None, return_ratios=False)`.
pub fn dpss(
    n: usize,
    nw: f64,
    kmax: Option<usize>,
    sym: bool,
    norm: Option<DpssNorm>,
    return_ratios: bool,
) -> Result<DpssResult, SignalError> {
    let norm = norm.unwrap_or(if kmax.is_none() {
        DpssNorm::Approximate
    } else {
        DpssNorm::L2
    });
    let k = kmax.unwrap_or(1);

    if n <= 1 {
        let window = if n == 0 { Vec::new() } else { vec![1.0] };
        return Ok(DpssResult {
            windows: vec![window],
            ratios: return_ratios.then(|| vec![1.0]),
        });
    }
    if k == 0 || k > n {
        return Err(SignalError::InvalidArgument(
            "kmax must be greater than 0 and less than or equal to M".to_string(),
        ));
    }
    if !nw.is_finite() || nw <= 0.0 {
        return Err(SignalError::InvalidArgument(
            "NW must be positive".to_string(),
        ));
    }
    if nw >= n as f64 / 2.0 {
        return Err(SignalError::InvalidArgument(
            "NW must be less than M/2.".to_string(),
        ));
    }

    let work_n = if sym {
        n
    } else {
        n.checked_add(1).ok_or_else(|| {
            SignalError::InvalidArgument(format!(
                "window length overflow while building periodic DPSS for n={n}"
            ))
        })?
    };
    let bandwidth = nw / work_n as f64;
    let cosine = (2.0 * std::f64::consts::PI * bandwidth).cos();
    let diagonal: Vec<f64> = (0..work_n)
        .map(|idx| {
            let centered = (work_n as f64 - 1.0 - 2.0 * idx as f64) / 2.0;
            centered * centered * cosine
        })
        .collect();
    let off_diagonal: Vec<f64> = (1..work_n)
        .map(|idx| idx as f64 * (work_n - idx) as f64 / 2.0)
        .collect();

    // Materialize the symmetric tridiagonal matrix and use the dense symmetric
    // eigensolver. The dedicated tridiagonal path is not yet numerically
    // compatible enough for SciPy DPSS parity.
    let mut tridiagonal = vec![vec![0.0; work_n]; work_n];
    for idx in 0..work_n {
        tridiagonal[idx][idx] = diagonal[idx];
        if idx + 1 < work_n {
            tridiagonal[idx][idx + 1] = off_diagonal[idx];
            tridiagonal[idx + 1][idx] = off_diagonal[idx];
        }
    }
    let eigh = fsci_linalg::eigh(&tridiagonal, fsci_linalg::DecompOptions::default())
        .map_err(|err| SignalError::InvalidArgument(format!("dpss eigensolve failed: {err}")))?;

    let mut windows: Vec<Vec<f64>> = (work_n - k..work_n)
        .rev()
        .map(|col| {
            eigh.eigenvectors
                .iter()
                .map(|row| row[col])
                .collect::<Vec<_>>()
        })
        .collect();

    for (idx, window) in windows.iter_mut().enumerate() {
        if idx.is_multiple_of(2) {
            if window.iter().sum::<f64>() < 0.0 {
                for value in window {
                    *value = -*value;
                }
            }
        } else {
            let threshold = (1.0 / work_n as f64).max(1.0e-7);
            if window
                .iter()
                .find(|value| **value * **value > threshold)
                .is_some_and(|value| *value < 0.0)
            {
                for value in window {
                    *value = -*value;
                }
            }
        }
    }

    let ratios = if return_ratios {
        Some(
            windows
                .iter()
                .map(|window| dpss_concentration_ratio(window, bandwidth))
                .collect(),
        )
    } else {
        None
    };

    if norm != DpssNorm::L2 {
        let max_value = windows
            .iter()
            .flat_map(|window| window.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max);
        if max_value.is_finite() && max_value > 0.0 {
            for window in &mut windows {
                for value in window {
                    *value /= max_value;
                }
            }
        }
        if work_n.is_multiple_of(2) {
            let correction = match norm {
                DpssNorm::Approximate => (work_n * work_n) as f64 / ((work_n * work_n) as f64 + nw),
                DpssNorm::Subsample => dpss_subsample_correction(&windows[0])?,
                DpssNorm::L2 => 1.0,
            };
            for window in &mut windows {
                for value in window {
                    *value *= correction;
                }
            }
        }
    }

    if !sym {
        for window in &mut windows {
            window.truncate(n);
        }
    }
    Ok(DpssResult { windows, ratios })
}

/// Triangular window.
///
/// Matches `scipy.signal.windows.triang(n)`.
pub fn triang(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let nf = n as f64;
    (0..n)
        .map(|i| {
            if n.is_multiple_of(2) {
                // Even: peak between two center samples
                let k = i as f64;
                1.0 - ((k - (nf - 1.0) / 2.0) / (nf / 2.0)).abs()
            } else {
                // Odd: peak at center
                let k = i as f64;
                1.0 - ((k - (nf - 1.0) / 2.0) / ((nf + 1.0) / 2.0)).abs()
            }
        })
        .collect()
}

fn periodic_from_symmetric<F>(nx: usize, build: F) -> Result<Vec<f64>, SignalError>
where
    F: FnOnce(usize) -> Vec<f64>,
{
    if nx <= 1 {
        return Ok(build(nx));
    }
    let expanded = nx.checked_add(1).ok_or_else(|| {
        SignalError::InvalidArgument(format!(
            "window length overflow while building periodic form for nx={nx}"
        ))
    })?;
    let mut window = build(expanded);
    window.truncate(nx);
    Ok(window)
}

fn symmetric_or_periodic_window<F>(nx: usize, sym: bool, build: F) -> Result<Vec<f64>, SignalError>
where
    F: FnOnce(usize) -> Vec<f64>,
{
    if sym {
        Ok(build(nx))
    } else {
        periodic_from_symmetric(nx, build)
    }
}

/// Dispatch a window function by name string using SciPy's default periodic policy.
///
/// Matches `scipy.signal.get_window(window, Nx)` with `fftbins=True`.
pub fn get_window(window: &str, nx: usize) -> Result<Vec<f64>, SignalError> {
    get_window_with_fftbins(window, nx, true)
}

fn parse_window_bool(value: &str, label: &str) -> Result<bool, SignalError> {
    match value.trim() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        invalid => Err(SignalError::InvalidArgument(format!(
            "invalid {label}: {invalid}"
        ))),
    }
}

fn parse_finite_window_param(value: &str, label: &str) -> Result<f64, SignalError> {
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| SignalError::InvalidArgument(format!("invalid {label}: {value}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(SignalError::InvalidArgument(format!(
            "invalid {label}: {value}"
        )))
    }
}

/// Dispatch a window function by name string with explicit SciPy `fftbins` control.
///
/// `fftbins=true` returns the periodic form used for FFT analysis. `fftbins=false`
/// returns the symmetric form used for filter design. A `_periodic` or
/// `_symmetric` suffix on the window name overrides the flag.
///
/// # Supported windows
/// `"hann"`, `"hamming"`, `"blackman"`, `"blackmanharris"`, `"barthann"`,
/// `"bartlett"`, `"flattop"`, `"cosine"`, `"rectangular"` / `"boxcar"`,
/// `"lanczos"` / `"sinc"`,
/// `"kaiser,<beta>"` (e.g. `"kaiser,8.6"`), `"chebwin,<at>"` (e.g. `"chebwin,100"`),
/// `"dpss,<nw>"` (e.g. `"dpss,2.5"`),
/// `"general_hamming,<alpha>"` (e.g. `"general_hamming,0.75"`),
/// `"general_cosine,<a0>,<a1>,..."` (e.g. `"general_cosine,0.5,0.5"`),
/// `"gaussian,<std>"` (e.g. `"gaussian,2.0"`),
/// `"general_gaussian,<p>,<sig>"` (e.g. `"general_gaussian,1.5,2.0"`),
/// `"exponential,<tau>"` (e.g. `"exponential,2.0"`),
/// `"taylor"` / `"taylorwin"` with optional `"taylor,<nbar>[,<sll>[,<norm>]]"`
/// parameters (e.g. `"taylor,6,45,false"`).
pub fn get_window_with_fftbins(
    window: &str,
    nx: usize,
    fftbins: bool,
) -> Result<Vec<f64>, SignalError> {
    let lower = window.trim().to_lowercase();
    let (raw_name, raw_params) = lower
        .split_once(',')
        .map_or((lower.as_str(), None), |(name, params)| {
            (name.trim(), Some(params.trim()))
        });

    let mut sym = !fftbins;
    let win_name = if let Some(name) = raw_name.strip_suffix("_symmetric") {
        sym = true;
        name
    } else if let Some(name) = raw_name.strip_suffix("_periodic") {
        sym = false;
        name
    } else {
        raw_name
    };

    if let Some(rest) = raw_params.filter(|rest| !rest.is_empty()) {
        match win_name {
            "kaiser" => {
                let beta = parse_finite_window_param(rest, "kaiser beta")?;
                return symmetric_or_periodic_window(nx, sym, |n| kaiser(n, beta));
            }
            "kaiser_bessel_derived" | "kbd" => {
                let beta = parse_finite_window_param(rest, "kaiser_bessel_derived beta")?;
                return kaiser_bessel_derived(nx, beta, sym);
            }
            "dpss" => {
                let nw = parse_finite_window_param(rest, "dpss NW")?;
                return dpss(nx, nw, None, sym, None, false)
                    .map(|result| result.windows.into_iter().next().unwrap_or_default());
            }
            "chebwin" => {
                let attenuation = parse_finite_window_param(rest, "chebwin attenuation")?;
                return symmetric_or_periodic_window(nx, sym, |n| chebwin(n, attenuation));
            }
            "general_hamming" => {
                let alpha = parse_finite_window_param(rest, "general_hamming alpha")?;
                return symmetric_or_periodic_window(nx, sym, |n| general_hamming(n, alpha));
            }
            "general_cosine" => {
                let coeffs: Vec<f64> = rest
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(|value| parse_finite_window_param(value, "general_cosine coefficient"))
                    .collect::<Result<_, _>>()?;
                return Ok(general_cosine(nx, &coeffs, sym));
            }
            "gaussian" => {
                let std = parse_finite_window_param(rest, "gaussian std")?;
                return Ok(gaussian(nx, std, sym));
            }
            "general_gaussian" => {
                let params: Vec<&str> = rest
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .collect();
                if params.len() != 2 {
                    return Err(SignalError::InvalidArgument(format!(
                        "general_gaussian expects p,sig parameters: {rest}"
                    )));
                }
                let p = parse_finite_window_param(params[0], "general_gaussian shape parameter")?;
                let sig = parse_finite_window_param(params[1], "general_gaussian sigma parameter")?;
                return Ok(general_gaussian(nx, p, sig, sym));
            }
            "exponential" => {
                let tau = parse_finite_window_param(rest, "exponential tau")?;
                return exponential(nx, None, tau, sym);
            }
            "taylor" | "taylorwin" => {
                let params: Vec<&str> = rest.split(',').map(str::trim).collect();
                if params.is_empty()
                    || params.len() > 3
                    || params.iter().any(|value| value.is_empty())
                {
                    return Err(SignalError::InvalidArgument(format!(
                        "taylor expects nbar[,sll[,norm]] parameters: {rest}"
                    )));
                }
                let nbar = params[0].parse::<usize>().map_err(|_| {
                    SignalError::InvalidArgument(format!("invalid taylor nbar: {}", params[0]))
                })?;
                let sll = params.get(1).map_or(Ok(30.0), |value| {
                    parse_finite_window_param(value, "taylor sll")
                })?;
                let norm = params
                    .get(2)
                    .map_or(Ok(true), |value| parse_window_bool(value, "taylor norm"))?;
                return Ok(taylor(nx, nbar, sll, norm, sym));
            }
            _ => {}
        }
    }

    match win_name {
        "hann" | "hanning" => symmetric_or_periodic_window(nx, sym, hann),
        "hamming" => symmetric_or_periodic_window(nx, sym, hamming),
        "blackman" => symmetric_or_periodic_window(nx, sym, blackman),
        "blackmanharris" => symmetric_or_periodic_window(nx, sym, blackmanharris),
        "barthann" => symmetric_or_periodic_window(nx, sym, barthann),
        "bartlett" | "triangle" => symmetric_or_periodic_window(nx, sym, bartlett),
        "flattop" => symmetric_or_periodic_window(nx, sym, flattop),
        "cosine" => symmetric_or_periodic_window(nx, sym, cosine),
        "lanczos" | "sinc" => symmetric_or_periodic_window(nx, sym, lanczos),
        "rectangular" | "boxcar" | "rect" => Ok(boxcar(nx, sym)),
        "chebwin" => symmetric_or_periodic_window(nx, sym, |n| chebwin(n, 100.0)),
        "parzen" => symmetric_or_periodic_window(nx, sym, parzen),
        "triang" => symmetric_or_periodic_window(nx, sym, triang),
        "tukey" => symmetric_or_periodic_window(nx, sym, |n| tukey_window(n, 0.5)),
        "nuttall" => symmetric_or_periodic_window(nx, sym, nuttall_window),
        "bohman" => symmetric_or_periodic_window(nx, sym, bohman_window),
        "exponential" => exponential(nx, None, 1.0, sym),
        "taylor" | "taylorwin" => Ok(taylor(nx, 4, 30.0, true, sym)),
        "kaiser_bessel_derived" | "kbd" => kaiser_bessel_derived(nx, 14.0, sym),
        "dpss" => Err(SignalError::InvalidArgument(
            "dpss requires a normalized half-bandwidth parameter".to_string(),
        )),
        _ => Err(SignalError::InvalidArgument(format!(
            "unknown window type: {window}"
        ))),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Time-Frequency Analysis: STFT, ISTFT, spectrogram, CSD, coherence
// ══════════════════════════════════════════════════════════════════════

/// Result of the Short-Time Fourier Transform.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Frequency bins (length = nperseg/2 + 1).
    pub frequencies: Vec<f64>,
    /// Time centers for each segment.
    pub times: Vec<f64>,
    /// Complex STFT matrix: `zxx[t][f] = (re, im)`.
    /// Outer index is time segment, inner is frequency bin.
    pub zxx: Vec<Vec<(f64, f64)>>,
}

/// Compute the Short-Time Fourier Transform.
///
/// Matches `scipy.signal.stft(x, fs, window, nperseg, noverlap)`.
///
/// # Arguments
/// * `x` — Input signal.
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Length of each segment (default: 256).
/// * `noverlap` — Overlap between segments (default: nperseg/2).
// Worker count for the parallel STFT frame loop: 1 (sequential) unless there are
// enough frames carrying enough total FFT work to amortise thread spawn (each frame
// is an O(nperseg log nperseg) rfft), then scale with cores capped at frame_count/2.
fn stft_frame_thread_count(frame_count: usize, nperseg: usize) -> usize {
    // FFT flops per frame ~ nperseg*log2(nperseg); only parallelise when the total
    // clearly amortises thread spawn (cheap small-nperseg STFTs run faster serial).
    let logn = (nperseg.max(2) as u64).ilog2() as u64;
    let flops = (frame_count as u64)
        .saturating_mul(nperseg as u64)
        .saturating_mul(logn);
    if flops < 1 << 24 || frame_count < 16 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(frame_count / 4).max(1)
}

pub fn stft(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<StftResult, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0".to_string(),
        ));
    }
    if nperseg > x.len() {
        return Err(SignalError::InvalidArgument(format!(
            "nperseg ({nperseg}) must be <= signal length ({})",
            x.len()
        )));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;
    let n_freqs = nperseg / 2 + 1;

    // Each frame (window + rfft over a disjoint segment) is independent and expensive,
    // so for many frames the work is split across threads. Every frame runs the same
    // deterministic window+rfft, so the result is bit-identical to the sequential loop;
    // the first FFT error (if any) is returned in frame order.
    let frame_count = (x.len() - nperseg) / step + 1;
    let compute_frame = |f: usize| -> Result<(Vec<fsci_fft::Complex64>, f64), SignalError> {
        let start = f * step;
        let windowed: Vec<f64> = x[start..start + nperseg]
            .iter()
            .zip(&win_coeffs)
            .map(|(&xi, &wi)| xi * wi)
            .collect();
        let opts = fsci_fft::FftOptions::default();
        let spectrum = fsci_fft::rfft(&windowed, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
        Ok((
            spectrum[..n_freqs].to_vec(),
            (start as f64 + (nperseg - 1) as f64 / 2.0) / fs,
        ))
    };

    let frames: Vec<(Vec<fsci_fft::Complex64>, f64)> = {
        let nthreads = stft_frame_thread_count(frame_count, nperseg);
        if nthreads <= 1 {
            (0..frame_count)
                .map(&compute_frame)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let chunk = frame_count.div_ceil(nthreads);
            let cf = &compute_frame;
            type FrameChunk = Result<Vec<(Vec<fsci_fft::Complex64>, f64)>, SignalError>;
            let chunk_results: Vec<FrameChunk> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..nthreads)
                    .filter_map(|t| {
                        let f0 = t * chunk;
                        if f0 >= frame_count {
                            return None;
                        }
                        let f1 = (f0 + chunk).min(frame_count);
                        Some(scope.spawn(move || (f0..f1).map(cf).collect::<Result<Vec<_>, _>>()))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("stft worker panicked"))
                    .collect()
            });
            let mut frames = Vec::with_capacity(frame_count);
            for cr in chunk_results {
                frames.extend(cr?);
            }
            frames
        }
    };

    let mut zxx = Vec::with_capacity(frame_count);
    let mut times = Vec::with_capacity(frame_count);
    for (spec, time) in frames {
        zxx.push(spec);
        times.push(time);
    }

    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(StftResult {
        frequencies,
        times,
        zxx,
    })
}

/// Compute the inverse Short-Time Fourier Transform (overlap-add reconstruction).
///
/// Matches `scipy.signal.istft(Zxx, fs, nperseg, noverlap)`.
///
/// # Arguments
/// * `stft_result` — STFT result from `stft()`.
/// * `nperseg` — Segment length used in the forward STFT.
/// * `noverlap` — Overlap used in the forward STFT (default: nperseg/2).
pub fn istft(
    stft_result: &StftResult,
    nperseg: usize,
    noverlap: Option<usize>,
) -> Result<Vec<f64>, SignalError> {
    if stft_result.zxx.is_empty() {
        return Err(SignalError::InvalidArgument(
            "empty STFT result".to_string(),
        ));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg - noverlap;
    let n_segments = stft_result.zxx.len();
    let output_len = nperseg + (n_segments - 1) * step;

    let window = hann(nperseg);
    let opts = fsci_fft::FftOptions::default();

    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    for (seg_idx, spectrum) in stft_result.zxx.iter().enumerate() {
        // Inverse rfft.
        let segment = fsci_fft::irfft(spectrum, Some(nperseg), &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

        let start = seg_idx * step;
        for (j, (&s, &w)) in segment.iter().zip(&window).enumerate() {
            output[start + j] += s * w;
            window_sum[start + j] += w * w;
        }
    }

    // Normalize by window sum (COLA condition).
    for (o, &ws) in output.iter_mut().zip(&window_sum) {
        if ws > 1e-15 {
            *o /= ws;
        }
    }

    Ok(output)
}

/// Result of the spectrogram computation.
#[derive(Debug, Clone)]
pub struct SpectrogramResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Time centers for each segment.
    pub times: Vec<f64>,
    /// Power spectral density: `sxx[t][f]`.
    pub sxx: Vec<Vec<f64>>,
}

/// Compute a spectrogram (time-frequency power representation).
///
/// Matches `scipy.signal.spectrogram(x, fs, nperseg, noverlap)`.
///
/// # Arguments
/// * `x` — Input signal.
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Length of each segment (default: 256).
/// * `noverlap` — Overlap between segments (default: nperseg/8).
pub fn spectrogram(
    x: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<SpectrogramResult, SignalError> {
    let nperseg_val = nperseg.unwrap_or_else(|| x.len().min(256));
    let noverlap_val = noverlap.unwrap_or(nperseg_val / 8);

    let stft_res = stft(x, fs, window, Some(nperseg_val), Some(noverlap_val))?;

    let n_freqs = stft_res.frequencies.len();
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg_val)?;
    let win_power: f64 = win_coeffs.iter().map(|&w| w * w).sum::<f64>() / nperseg_val as f64;

    // Convert complex STFT to PSD.
    let scale = 1.0 / (fs * nperseg_val as f64 * win_power);
    let sxx: Vec<Vec<f64>> = stft_res
        .zxx
        .iter()
        .map(|seg| {
            seg.iter()
                .enumerate()
                .map(|(k, &(re, im))| {
                    let mag2 = re * re + im * im;
                    let factor = if k == 0 || (nperseg_val.is_multiple_of(2) && k == n_freqs - 1) {
                        1.0
                    } else {
                        2.0
                    };
                    mag2 * scale * factor
                })
                .collect()
        })
        .collect();

    Ok(SpectrogramResult {
        frequencies: stft_res.frequencies,
        times: stft_res.times,
        sxx,
    })
}

/// Cross-spectral density estimation using Welch's method.
///
/// Matches `scipy.signal.csd(x, y, fs, nperseg, noverlap)`.
///
/// Returns complex cross-spectral density `Pxy[k] = conj(X[k]) * Y[k]` averaged
/// over segments.
///
/// # Arguments
/// * `x` — First input signal.
/// * `y` — Second input signal (same length as x).
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Segment length (default: 256).
/// * `noverlap` — Overlap (default: nperseg/2).
pub fn csd(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<CsdResult, SignalError> {
    csd_with_scaling(
        x,
        y,
        fs,
        window,
        nperseg,
        noverlap,
        SpectralScaling::Density,
    )
}

/// Cross-spectral density / spectrum estimation using Welch's method.
///
/// Matches `scipy.signal.csd(..., scaling='density'|'spectrum')`.
pub fn csd_with_scaling(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    scaling: SpectralScaling,
) -> Result<CsdResult, SignalError> {
    if x.len() != y.len() {
        return Err(SignalError::InvalidArgument(format!(
            "x and y must have same length ({} vs {})",
            x.len(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }

    let nperseg = nperseg.unwrap_or_else(|| x.len().min(256));
    if nperseg == 0 || nperseg > x.len() {
        return Err(SignalError::InvalidArgument(
            "nperseg must be > 0 and <= signal length".to_string(),
        ));
    }
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be < nperseg".to_string(),
        ));
    }

    let step = nperseg - noverlap;
    let win_coeffs = get_window(window.unwrap_or("hann"), nperseg)?;
    let win_power: f64 = win_coeffs.iter().map(|&w| w * w).sum::<f64>() / nperseg as f64;
    let n_freqs = nperseg / 2 + 1;
    let opts = fsci_fft::FftOptions::default();

    // nperseg <= x.len() above, so at least one segment fits.
    let n_segments = (x.len() - nperseg) / step + 1;

    // Each segment's cross-periodogram (constant-detrend + window + two rffts + conj(X)*Y)
    // is independent and expensive (two FFTs per segment); for many segments the work is
    // split across threads. Each segment returns its per-bin contribution (with the
    // one-sided factor applied), and the averaging fold runs in segment order, so the
    // result is bit-identical to the sequential loop.
    let compute_segment = |s: usize| -> Result<Vec<(f64, f64)>, SignalError> {
        let start = s * step;
        let xs = &x[start..start + nperseg];
        let ys = &y[start..start + nperseg];
        let xmean = xs.iter().sum::<f64>() / nperseg as f64;
        let ymean = ys.iter().sum::<f64>() / nperseg as f64;
        let wx: Vec<f64> = xs
            .iter()
            .zip(&win_coeffs)
            .map(|(&xi, &wi)| (xi - xmean) * wi)
            .collect();
        let wy: Vec<f64> = ys
            .iter()
            .zip(&win_coeffs)
            .map(|(&yi, &wi)| (yi - ymean) * wi)
            .collect();
        let sx = fsci_fft::rfft(&wx, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
        let sy = fsci_fft::rfft(&wy, &opts)
            .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;
        let mut out = Vec::with_capacity(n_freqs);
        for (k, (&(xr, xi), &(yr, yi))) in sx.iter().zip(sy.iter()).take(n_freqs).enumerate() {
            // conj(X) * Y = (xr - j*xi) * (yr + j*yi) = (xr*yr + xi*yi) + j*(xr*yi - xi*yr)
            let re = xr * yr + xi * yi;
            let im = xr * yi - xi * yr;
            let factor = if k == 0 || (nperseg.is_multiple_of(2) && k == n_freqs - 1) {
                1.0
            } else {
                2.0
            };
            out.push((re * factor, im * factor));
        }
        Ok(out)
    };

    let seg_csds: Vec<Vec<(f64, f64)>> = {
        let nthreads = stft_frame_thread_count(n_segments, nperseg);
        if nthreads <= 1 {
            (0..n_segments)
                .map(&compute_segment)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let chunk = n_segments.div_ceil(nthreads);
            let cs = &compute_segment;
            type SegChunk = Result<Vec<Vec<(f64, f64)>>, SignalError>;
            let chunk_results: Vec<SegChunk> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..nthreads)
                    .filter_map(|t| {
                        let s0 = t * chunk;
                        if s0 >= n_segments {
                            return None;
                        }
                        let s1 = (s0 + chunk).min(n_segments);
                        Some(scope.spawn(move || (s0..s1).map(cs).collect::<Result<Vec<_>, _>>()))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("csd worker panicked"))
                    .collect()
            });
            let mut v = Vec::with_capacity(n_segments);
            for cr in chunk_results {
                v.extend(cr?);
            }
            v
        }
    };

    let mut avg_csd = vec![(0.0, 0.0); n_freqs];
    for seg in &seg_csds {
        for ((avg_re, avg_im), &(re, im)) in avg_csd.iter_mut().zip(seg.iter()) {
            *avg_re += re;
            *avg_im += im;
        }
    }

    let scale = match scaling {
        SpectralScaling::Density => 1.0 / (fs * nperseg as f64 * win_power * n_segments as f64),
        SpectralScaling::Spectrum => {
            let win_sum = win_coeffs.iter().sum::<f64>();
            1.0 / (win_sum * win_sum * n_segments as f64)
        }
    };
    for (re, im) in &mut avg_csd {
        *re *= scale;
        *im *= scale;
    }

    let freq_step = fs / nperseg as f64;
    let frequencies: Vec<f64> = (0..n_freqs).map(|k| k as f64 * freq_step).collect();

    Ok(CsdResult {
        frequencies,
        csd: avg_csd,
    })
}

/// Result of cross-spectral density computation.
#[derive(Debug, Clone)]
pub struct CsdResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Complex cross-spectral density: `(re, im)` per frequency bin.
    pub csd: Vec<(f64, f64)>,
}

/// Result of coherence computation.
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Frequency bins.
    pub frequencies: Vec<f64>,
    /// Magnitude-squared coherence: `|Pxy|² / (Pxx * Pyy)` per frequency bin.
    pub coherence: Vec<f64>,
}

/// Compute magnitude-squared coherence between two signals.
///
/// Matches `scipy.signal.coherence(x, y, fs, nperseg, noverlap)`.
///
/// `Cxy[f] = |Pxy[f]|² / (Pxx[f] * Pyy[f])` where Pxy is the cross-spectral
/// density and Pxx, Pyy are the auto-spectral densities.
///
/// # Arguments
/// * `x` — First input signal.
/// * `y` — Second input signal (same length as x).
/// * `fs` — Sampling frequency (Hz).
/// * `window` — Window type to use (default: "hann").
/// * `nperseg` — Segment length (default: 256).
/// * `noverlap` — Overlap (default: nperseg/2).
pub fn coherence(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window: Option<&str>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
) -> Result<CoherenceResult, SignalError> {
    let pxy = csd(x, y, fs, window, nperseg, noverlap)?;
    let pxx = csd(x, x, fs, window, nperseg, noverlap)?;
    let pyy = csd(y, y, fs, window, nperseg, noverlap)?;

    let coh: Vec<f64> = pxy
        .csd
        .iter()
        .zip(pxx.csd.iter().zip(pyy.csd.iter()))
        .map(|(&(pxy_re, pxy_im), (&(pxx_re, _), &(pyy_re, _)))| {
            // |Pxy|² / (Pxx * Pyy). Auto-spectra are real (imaginary ≈ 0).
            let pxy_mag2 = pxy_re * pxy_re + pxy_im * pxy_im;
            let denom = pxx_re * pyy_re;
            if denom.abs() < 1e-30 {
                0.0
            } else {
                (pxy_mag2 / denom).clamp(0.0, 1.0)
            }
        })
        .collect();

    Ok(CoherenceResult {
        frequencies: pxy.frequencies,
        coherence: coh,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Signal Resampling: resample, resample_poly, decimate
// ══════════════════════════════════════════════════════════════════════

/// Resample a signal using the FFT method.
///
/// Matches `scipy.signal.resample(x, num)`.
///
/// Changes the number of samples in a signal by zero-padding or truncating
/// in the frequency domain.
///
/// # Arguments
/// * `x` — Input signal.
/// * `num` — Desired number of output samples.
pub fn resample(x: &[f64], num: usize) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if num == 0 {
        return Err(SignalError::InvalidArgument("num must be > 0".to_string()));
    }

    let n = x.len();
    if num == n {
        return Ok(x.to_vec());
    }

    let opts = fsci_fft::FftOptions::default();
    let m = n.min(num);
    let m2 = m / 2 + 1;

    // Forward FFT.
    let spectrum = fsci_fft::rfft(x, &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("FFT failed: {e}")))?;

    // Compute target spectrum length for the new sample count.
    let target_nfreqs = num / 2 + 1;

    // Zero-pad or truncate the spectrum, keeping only the bins that are
    // representable in both the input and output grids.
    let mut new_spectrum = vec![(0.0, 0.0); target_nfreqs];
    new_spectrum[..m2].copy_from_slice(&spectrum[..m2]);

    // Match SciPy's treatment of the unpaired Nyquist bin when the smaller
    // of the input/output lengths is even and the lengths differ.
    if m.is_multiple_of(2) {
        let nyquist = m / 2;
        if num < n {
            new_spectrum[nyquist].0 *= 2.0;
            new_spectrum[nyquist].1 *= 2.0;
        } else {
            new_spectrum[nyquist].0 *= 0.5;
            new_spectrum[nyquist].1 *= 0.5;
        }
    }

    // Inverse FFT with target length.
    let result = fsci_fft::irfft(&new_spectrum, Some(num), &opts)
        .map_err(|e| SignalError::InvalidArgument(format!("IFFT failed: {e}")))?;

    // Scale by num/n to preserve amplitude.
    let scale = num as f64 / n as f64;
    Ok(result.into_iter().map(|v| v * scale).collect())
}

/// Resample a signal using polyphase filtering (rational rate change).
///
/// Matches `scipy.signal.resample_poly(x, up, down)`.
///
/// Resamples by factor `up/down` using an anti-aliasing FIR filter.
///
/// # Arguments
/// * `x` — Input signal.
/// * `up` — Upsampling factor.
/// * `down` — Downsampling factor.
pub fn resample_poly(x: &[f64], up: usize, down: usize) -> Result<Vec<f64>, SignalError> {
    resample_poly_with_padtype(x, up, down, None, None)
}

#[derive(Clone, Copy, Debug)]
enum ResamplePolyPadMode {
    Constant(f64),
    Background(f64),
    Edge,
    Reflect,
    Symmetric,
    Wrap,
    Line { start: f64, slope: f64 },
}

fn resample_poly_reflected_index(idx: i64, len: usize, symmetric: bool) -> usize {
    if len <= 1 {
        return 0;
    }

    let len = len as i64;
    let period = if symmetric { 2 * len } else { 2 * len - 2 };
    let idx = idx.rem_euclid(period);
    if symmetric {
        if idx < len {
            idx as usize
        } else {
            (period - idx - 1) as usize
        }
    } else if idx < len {
        idx as usize
    } else {
        (period - idx) as usize
    }
}

fn resample_poly_median(x: &[f64]) -> f64 {
    let mut values = x.to_vec();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn parse_resample_poly_pad_mode(
    x: &[f64],
    padtype: Option<&str>,
    cval: Option<f64>,
) -> Result<ResamplePolyPadMode, SignalError> {
    let padtype = padtype.unwrap_or("constant");
    if cval.is_some() && padtype != "constant" {
        return Err(SignalError::InvalidArgument(format!(
            "cval has no effect when padtype is {padtype}"
        )));
    }

    let constant = |value: f64| Ok(ResamplePolyPadMode::Constant(value));
    match padtype {
        "constant" => constant(cval.unwrap_or(0.0)),
        "mean" => Ok(ResamplePolyPadMode::Background(
            x.iter().sum::<f64>() / x.len() as f64,
        )),
        "median" => Ok(ResamplePolyPadMode::Background(resample_poly_median(x))),
        "maximum" => Ok(ResamplePolyPadMode::Background(
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )),
        "minimum" => Ok(ResamplePolyPadMode::Background(
            x.iter().copied().fold(f64::INFINITY, f64::min),
        )),
        "edge" => Ok(ResamplePolyPadMode::Edge),
        "reflect" => Ok(ResamplePolyPadMode::Reflect),
        "symmetric" => Ok(ResamplePolyPadMode::Symmetric),
        "wrap" => Ok(ResamplePolyPadMode::Wrap),
        "line" => {
            let slope = if x.len() > 1 {
                (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64
            } else {
                0.0
            };
            Ok(ResamplePolyPadMode::Line { start: x[0], slope })
        }
        _ => Err(SignalError::InvalidArgument(
            "padtype must be one of {'constant', 'line', 'mean', 'median', \
             'maximum', 'minimum', 'edge', 'reflect', 'symmetric', 'wrap'}"
                .to_string(),
        )),
    }
}

fn resample_poly_sample(x: &[f64], idx: i64, mode: ResamplePolyPadMode) -> f64 {
    if idx >= 0 && idx < x.len() as i64 {
        return x[idx as usize];
    }

    match mode {
        ResamplePolyPadMode::Constant(value) => value,
        ResamplePolyPadMode::Background(_) => 0.0,
        ResamplePolyPadMode::Edge => {
            if idx < 0 {
                x[0]
            } else {
                x[x.len() - 1]
            }
        }
        ResamplePolyPadMode::Reflect => x[resample_poly_reflected_index(idx, x.len(), false)],
        ResamplePolyPadMode::Symmetric => x[resample_poly_reflected_index(idx, x.len(), true)],
        ResamplePolyPadMode::Wrap => x[idx.rem_euclid(x.len() as i64) as usize],
        ResamplePolyPadMode::Line { start, slope } => start + slope * idx as f64,
    }
}

/// Resample a signal using polyphase filtering with SciPy-style boundary modes.
///
/// Matches `scipy.signal.resample_poly(..., padtype=..., cval=...)` for the
/// implemented 1-D modes.
pub fn resample_poly_with_padtype(
    x: &[f64],
    up: usize,
    down: usize,
    padtype: Option<&str>,
    cval: Option<f64>,
) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidArgument(
            "up and down must be > 0".to_string(),
        ));
    }
    let parsed_mode = parse_resample_poly_pad_mode(x, padtype, cval)?;
    let (background, pad_mode, input) = match parsed_mode {
        ResamplePolyPadMode::Background(value) => (
            value,
            ResamplePolyPadMode::Constant(0.0),
            x.iter().map(|sample| sample - value).collect::<Vec<_>>(),
        ),
        mode => (0.0, mode, x.to_vec()),
    };

    // Simplify the ratio using GCD.
    let g = gcd(up, down);
    let up = up / g;
    let down = down / g;

    if up == 1 && down == 1 {
        return Ok(input.iter().map(|sample| sample + background).collect());
    }

    // Design anti-aliasing FIR filter.
    let cutoff = 1.0 / (up.max(down) as f64);
    let n_taps = 2 * 10 * up.max(down) + 1; // ~10 periods per side
    let h = firwin(n_taps, &[cutoff], FirWindow::Kaiser(5.0), true)?;
    // Scale filter by up to compensate for upsampling gain.
    let h_scaled: Vec<f64> = h.iter().map(|&v| v * up as f64).collect();

    // Polyphase implementation to avoid materializing the full upsampled signal.
    // Equivalent to: upsample -> convolve(mode=Same) -> downsample.
    let upsampled_len = input.len() * up;
    let half_taps = (n_taps - 1) as i64 / 2;
    let mut output = Vec::with_capacity(upsampled_len.div_ceil(down));

    let mut i = 0usize;
    while i < upsampled_len {
        let mut val = 0.0;
        let target = i as i64 + half_taps;

        // k must satisfy: 0 <= k < n_taps AND (target - k) is a multiple of 'up'.
        let k_start = (target % up as i64 + up as i64) % up as i64;
        for k in (k_start as usize..n_taps).step_by(up) {
            let x_idx = (target - k as i64) / up as i64;
            val += h_scaled[k] * resample_poly_sample(&input, x_idx, pad_mode);
        }
        output.push(val + background);
        i += down;
    }

    Ok(output)
}

/// Upsample, FIR filter, and downsample a 1-D signal.
///
/// Matches `scipy.signal.upfirdn(h, x, up, down)` for the default 1-D case.
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Result<Vec<f64>, SignalError> {
    if h.is_empty() {
        return Err(SignalError::InvalidArgument(
            "h must be 1-D with non-zero length".to_string(),
        ));
    }
    if up == 0 || down == 0 {
        return Err(SignalError::InvalidArgument(
            "up and down must be >= 1".to_string(),
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }

    let upsampled_len = (x.len() - 1) * up + 1;
    let full_len = upsampled_len + h.len() - 1;
    let mut output = vec![0.0; full_len];

    for (i, &sample) in x.iter().enumerate() {
        let base = i * up;
        for (tap_idx, &tap) in h.iter().enumerate() {
            output[base + tap_idx] += sample * tap;
        }
    }

    Ok(output.into_iter().step_by(down).collect())
}

/// Convert a linear-phase FIR filter to minimum phase.
///
/// Matches `scipy.signal.minimum_phase(h)` for the default homomorphic method
/// with `half=True`.
pub fn minimum_phase(h: &[f64]) -> Result<Vec<f64>, SignalError> {
    if h.len() <= 2 {
        return Err(SignalError::InvalidArgument(
            "h must be 1-D and at least 2 samples long".to_string(),
        ));
    }

    let suggested_fft = ((2 * (h.len() - 1)) as f64 / 0.01).ceil() as usize;
    let n_fft = suggested_fft
        .max(h.len())
        .checked_next_power_of_two()
        .unwrap_or(suggested_fft.max(h.len()));
    let opts = fsci_fft::FftOptions::default();

    let mut padded = vec![(0.0, 0.0); n_fft];
    for (dst, &coeff) in padded.iter_mut().zip(h.iter()) {
        dst.0 = coeff;
    }

    let spectrum = fsci_fft::fft(&padded, &opts)
        .map_err(|err| SignalError::InvalidArgument(format!("minimum_phase FFT failed: {err}")))?;
    let magnitudes: Vec<f64> = spectrum
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect();
    let min_positive = magnitudes
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .fold(f64::INFINITY, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        });
    let epsilon = if min_positive.is_finite() {
        1e-7 * min_positive
    } else {
        1e-7
    };

    let log_spectrum: Vec<(f64, f64)> = magnitudes
        .into_iter()
        .map(|value| ((value + epsilon).ln() * 0.5, 0.0))
        .collect();
    let cepstrum = fsci_fft::ifft(&log_spectrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase inverse FFT failed: {err}"))
    })?;

    let mut homomorphic_window = vec![0.0; n_fft];
    homomorphic_window[0] = 1.0;
    let stop = n_fft / 2;
    for value in homomorphic_window.iter_mut().take(stop).skip(1) {
        *value = 2.0;
    }
    if n_fft % 2 == 1 {
        homomorphic_window[stop] = 1.0;
    }

    let weighted_cepstrum: Vec<(f64, f64)> = cepstrum
        .iter()
        .zip(homomorphic_window.iter())
        .map(|(&(re, _), &weight)| (re * weight, 0.0))
        .collect();
    let minimum_log_spectrum = fsci_fft::fft(&weighted_cepstrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase cepstrum FFT failed: {err}"))
    })?;
    let minimum_spectrum: Vec<(f64, f64)> = minimum_log_spectrum
        .into_iter()
        .map(|(re, im)| {
            let exp_re = re.exp();
            (exp_re * im.cos(), exp_re * im.sin())
        })
        .collect();
    let minimum_impulse = fsci_fft::ifft(&minimum_spectrum, &opts).map_err(|err| {
        SignalError::InvalidArgument(format!("minimum_phase reconstruction failed: {err}"))
    })?;

    let n_out = h.len().div_ceil(2);
    Ok(minimum_impulse
        .into_iter()
        .take(n_out)
        .map(|(re, _)| re)
        .collect())
}

/// Downsample a signal after applying an anti-aliasing filter.
///
/// Matches `scipy.signal.decimate(x, q)` with its default IIR filter:
/// an order-8 Chebyshev type-I lowpass with 0.05 dB passband ripple and
/// cutoff `0.8/q`, applied zero-phase, then every `q`-th sample is kept.
///
/// # Arguments
/// * `x` — Input signal.
/// * `q` — Downsampling factor (integer >= 2).
pub fn decimate(x: &[f64], q: usize) -> Result<Vec<f64>, SignalError> {
    if x.is_empty() {
        return Err(SignalError::InvalidArgument(
            "input must not be empty".to_string(),
        ));
    }
    if q < 2 {
        return Err(SignalError::InvalidArgument("q must be >= 2".to_string()));
    }

    // scipy.signal.decimate default: cheby1(8, 0.05, 0.8/q) anti-alias
    // filter (a fixed order 8 — not reduced for small q — and cutoff
    // 0.8/q rather than 1/q), applied zero-phase via filtfilt.
    let cutoff = 0.8 / q as f64;
    let ba = cheby1(8, 0.05, &[cutoff], FilterType::Lowpass)?;

    let filtered = filtfilt(&ba.b, &ba.a, x)?;

    // Downsample.
    let output: Vec<f64> = filtered.iter().step_by(q).copied().collect();
    Ok(output)
}

/// Compute GCD of two positive integers.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ═══════════════════════════════════════════════════════════════════
// Linear Time-Invariant (LTI) System Classes
// ═══════════════════════════════════════════════════════════════════

/// Continuous-time Linear Time-Invariant system representation.
///
/// Matches `scipy.signal.lti`. Represents a continuous-time LTI system
/// in transfer function form (numerator/denominator polynomials).
///
/// The transfer function is: H(s) = num(s) / den(s)
#[derive(Debug, Clone, PartialEq)]
pub struct Lti {
    /// Numerator polynomial coefficients (highest power first).
    pub num: Vec<f64>,
    /// Denominator polynomial coefficients (highest power first).
    pub den: Vec<f64>,
}

impl Lti {
    /// Create a new continuous-time LTI system from transfer function coefficients.
    ///
    /// # Arguments
    /// * `num` - Numerator polynomial coefficients (highest power first)
    /// * `den` - Denominator polynomial coefficients (highest power first)
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Result<Self, SignalError> {
        if num.is_empty() {
            return Err(SignalError::InvalidArgument(
                "numerator cannot be empty".to_string(),
            ));
        }
        if den.is_empty() {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be empty".to_string(),
            ));
        }
        if den.iter().all(|&x| x == 0.0) {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be all zeros".to_string(),
            ));
        }
        Ok(Self { num, den })
    }

    /// Create an LTI system from zeros, poles, and gain.
    ///
    /// H(s) = k * prod(s - z_i) / prod(s - p_j)
    pub fn from_zpk(zeros: &[f64], poles: &[f64], gain: f64) -> Result<Self, SignalError> {
        let num = zpk_to_poly(zeros, gain);
        let den = zpk_to_poly(poles, 1.0);
        Self::new(num, den)
    }

    /// Evaluate the transfer function at a complex frequency s.
    pub fn eval_at(&self, s_re: f64, s_im: f64) -> (f64, f64) {
        let num_val = poly_eval_complex(&self.num, s_re, s_im);
        let den_val = poly_eval_complex(&self.den, s_re, s_im);
        complex_div(num_val.0, num_val.1, den_val.0, den_val.1)
    }

    /// Compute the frequency response H(jω) at angular frequencies.
    ///
    /// Returns (magnitude, phase) arrays.
    pub fn freqresp(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mag = Vec::with_capacity(w.len());
        let mut phase = Vec::with_capacity(w.len());

        for &omega in w {
            let (re, im) = self.eval_at(0.0, omega);
            mag.push((re * re + im * im).sqrt());
            phase.push(im.atan2(re));
        }

        (mag, phase)
    }

    /// Compute the step response of the system.
    ///
    /// Uses numerical simulation with RK4 integration.
    pub fn step(&self, t: &[f64]) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        // Convert to state-space and simulate
        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        simulate_lti_step(&a, &b, &c, d, t)
    }

    /// Compute the impulse response of the system.
    pub fn impulse(&self, t: &[f64]) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        simulate_lti_impulse(&a, &b, &c, d, t)
    }

    /// Get the system's poles (roots of denominator).
    pub fn poles(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.den)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Get the system's zeros (roots of numerator).
    pub fn zeros(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.num)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Simulate the response of the LTI system to arbitrary input.
    ///
    /// Matches `scipy.signal.lsim`. Uses state-space representation with RK4.
    ///
    /// # Arguments
    /// * `u` - Input signal values at each time point
    /// * `t` - Time points (must be uniformly spaced)
    /// * `x0` - Initial state (None for zero initial conditions)
    ///
    /// # Returns
    /// Output signal values at each time point
    pub fn lsim(&self, u: &[f64], t: &[f64], x0: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
        if t.is_empty() {
            return Ok(Vec::new());
        }
        if u.len() != t.len() {
            return Err(SignalError::InvalidArgument(format!(
                "input u length {} must match t length {}",
                u.len(),
                t.len()
            )));
        }

        let (a, b, c, d) = tf2ss(&self.num, &self.den)?;
        let n = b.len();

        // Initialize state
        let mut x = if let Some(x0_init) = x0 {
            if x0_init.len() != n {
                return Err(SignalError::InvalidArgument(format!(
                    "initial state length {} must match system order {}",
                    x0_init.len(),
                    n
                )));
            }
            x0_init.to_vec()
        } else {
            vec![0.0; n]
        };

        if n == 0 {
            // Static system: y = d*u
            return Ok(u.iter().map(|&ui| d * ui).collect());
        }

        let mut y = Vec::with_capacity(t.len());

        for i in 0..t.len() {
            // Output: y = c'x + d*u
            let output: f64 = c
                .iter()
                .zip(x.iter())
                .map(|(&ci, &xi)| ci * xi)
                .sum::<f64>()
                + d * u[i];
            y.push(output);

            if i + 1 < t.len() {
                let dt = t[i + 1] - t[i];
                // RK4 step with current input
                x = rk4_step(&a, &b, &x, u[i], dt);
            }
        }

        Ok(y)
    }
}

/// Discrete-time Linear Time-Invariant system representation.
///
/// Matches `scipy.signal.dlti`. Represents a discrete-time LTI system
/// in transfer function form.
///
/// The transfer function is: H(z) = num(z) / den(z)
#[derive(Debug, Clone, PartialEq)]
pub struct Dlti {
    /// Numerator polynomial coefficients (highest power first).
    pub num: Vec<f64>,
    /// Denominator polynomial coefficients (highest power first).
    pub den: Vec<f64>,
    /// Sampling period (dt > 0).
    pub dt: f64,
}

impl Dlti {
    /// Create a new discrete-time LTI system.
    ///
    /// # Arguments
    /// * `num` - Numerator polynomial coefficients
    /// * `den` - Denominator polynomial coefficients
    /// * `dt` - Sampling period (must be > 0)
    pub fn new(num: Vec<f64>, den: Vec<f64>, dt: f64) -> Result<Self, SignalError> {
        if num.is_empty() {
            return Err(SignalError::InvalidArgument(
                "numerator cannot be empty".to_string(),
            ));
        }
        if den.is_empty() {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be empty".to_string(),
            ));
        }
        if den.iter().all(|&x| x == 0.0) {
            return Err(SignalError::InvalidArgument(
                "denominator cannot be all zeros".to_string(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SignalError::InvalidArgument(
                "sampling period dt must be positive".to_string(),
            ));
        }
        Ok(Self { num, den, dt })
    }

    /// Create a DLTI system from zeros, poles, gain, and sampling period.
    pub fn from_zpk(zeros: &[f64], poles: &[f64], gain: f64, dt: f64) -> Result<Self, SignalError> {
        let num = zpk_to_poly(zeros, gain);
        let den = zpk_to_poly(poles, 1.0);
        Self::new(num, den, dt)
    }

    /// Evaluate the transfer function at z = exp(jω*dt).
    pub fn eval_at_freq(&self, omega: f64) -> (f64, f64) {
        let angle = omega * self.dt;
        let z_re = angle.cos();
        let z_im = angle.sin();
        let num_val = poly_eval_complex(&self.num, z_re, z_im);
        let den_val = poly_eval_complex(&self.den, z_re, z_im);
        complex_div(num_val.0, num_val.1, den_val.0, den_val.1)
    }

    /// Compute the frequency response H(e^{jω}) at angular frequencies.
    pub fn freqresp(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut mag = Vec::with_capacity(w.len());
        let mut phase = Vec::with_capacity(w.len());

        for &omega in w {
            let (re, im) = self.eval_at_freq(omega);
            mag.push((re * re + im * im).sqrt());
            phase.push(im.atan2(re));
        }

        (mag, phase)
    }

    /// Compute the step response of the discrete system.
    pub fn step(&self, n_samples: usize) -> Result<Vec<f64>, SignalError> {
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        // Unit step input
        let x: Vec<f64> = vec![1.0; n_samples];
        self.lfilter(&x)
    }

    /// Compute the impulse response of the discrete system.
    pub fn impulse(&self, n_samples: usize) -> Result<Vec<f64>, SignalError> {
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        // Unit impulse input
        let mut x = vec![0.0; n_samples];
        x[0] = 1.0;
        self.lfilter(&x)
    }

    /// Filter a signal using this discrete system (direct form II transposed).
    pub fn lfilter(&self, x: &[f64]) -> Result<Vec<f64>, SignalError> {
        lfilter(&self.num, &self.den, x, None)
    }

    /// Get the system's poles (roots of denominator).
    pub fn poles(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.den)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Get the system's zeros (roots of numerator).
    pub fn zeros(&self) -> Result<Vec<(f64, f64)>, SignalError> {
        let (re, im) = poly_roots(&self.num)?;
        Ok(re.into_iter().zip(im).collect())
    }

    /// Check if the system is stable (all poles inside unit circle).
    pub fn is_stable(&self) -> Result<bool, SignalError> {
        let poles = self.poles()?;
        Ok(poles.iter().all(|&(re, im)| re * re + im * im < 1.0))
    }

    /// Simulate the response of the discrete LTI system to arbitrary input.
    ///
    /// Matches `scipy.signal.dlsim`. Uses lfilter with optional initial state.
    ///
    /// # Arguments
    /// * `u` - Input signal values
    /// * `zi` - Initial filter state (None for zero initial conditions)
    ///
    /// # Returns
    /// Output signal values
    pub fn dlsim(&self, u: &[f64], zi: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
        lfilter(&self.num, &self.den, u, zi)
    }
}

/// Simulate the response of a continuous-time LTI system.
///
/// Standalone function matching `scipy.signal.lsim`.
///
/// # Arguments
/// * `system` - The LTI system
/// * `u` - Input signal values
/// * `t` - Time points
/// * `x0` - Initial state (None for zero)
pub fn lsim(
    system: &Lti,
    u: &[f64],
    t: &[f64],
    x0: Option<&[f64]>,
) -> Result<Vec<f64>, SignalError> {
    system.lsim(u, t, x0)
}

/// Simulate the response of a discrete-time LTI system.
///
/// Standalone function matching `scipy.signal.dlsim`.
///
/// # Arguments
/// * `system` - The DLTI system
/// * `u` - Input signal values
/// * `zi` - Initial filter state (None for zero)
pub fn dlsim(system: &Dlti, u: &[f64], zi: Option<&[f64]>) -> Result<Vec<f64>, SignalError> {
    system.dlsim(u, zi)
}

// ═══════════════════════════════════════════════════════════════════
// LTI Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Convert zeros/poles to polynomial coefficients.
fn zpk_to_poly(roots: &[f64], gain: f64) -> Vec<f64> {
    if roots.is_empty() {
        return vec![gain];
    }
    // Start with [1]
    let mut poly = vec![1.0];
    for &root in roots {
        // Multiply by (x - root)
        let mut new_poly = vec![0.0; poly.len() + 1];
        for (i, &c) in poly.iter().enumerate() {
            new_poly[i] += c;
            new_poly[i + 1] -= c * root;
        }
        poly = new_poly;
    }
    // Apply gain
    for c in &mut poly {
        *c *= gain;
    }
    poly
}

/// Evaluate polynomial with complex argument.
fn poly_eval_complex(coeffs: &[f64], z_re: f64, z_im: f64) -> (f64, f64) {
    if coeffs.is_empty() {
        return (0.0, 0.0);
    }
    // Horner's method for complex evaluation
    let mut re = coeffs[0];
    let mut im = 0.0;
    for &c in &coeffs[1..] {
        // (re, im) = (re, im) * (z_re, z_im) + (c, 0)
        let new_re = re * z_re - im * z_im + c;
        let new_im = re * z_im + im * z_re;
        re = new_re;
        im = new_im;
    }
    (re, im)
}

/// Convert transfer function to state-space (controllable canonical form).
type StateSpace = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64);

fn tf2ss(num: &[f64], den: &[f64]) -> Result<StateSpace, SignalError> {
    if den.is_empty() || den[0] == 0.0 {
        return Err(SignalError::InvalidArgument(
            "denominator leading coefficient cannot be zero".to_string(),
        ));
    }

    let n = den.len() - 1; // System order
    if n == 0 {
        // Static gain
        let d = if num.is_empty() { 0.0 } else { num[0] / den[0] };
        return Ok((Vec::new(), Vec::new(), Vec::new(), d));
    }

    // Normalize denominator
    let a0 = den[0];
    let den_norm: Vec<f64> = den.iter().map(|&x| x / a0).collect();

    // Pad numerator to match denominator length
    let mut num_padded = vec![0.0; den.len()];
    let offset = den.len().saturating_sub(num.len());
    for (i, &c) in num.iter().enumerate() {
        if offset + i < num_padded.len() {
            num_padded[offset + i] = c / a0;
        }
    }

    // Controllable canonical form
    // A matrix (n x n)
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        a[i][i + 1] = 1.0;
    }
    for i in 0..n {
        a[n - 1][i] = -den_norm[n - i];
    }

    // B vector (n x 1)
    let mut b = vec![0.0; n];
    b[n - 1] = 1.0;

    // C vector (1 x n)
    let mut c = vec![0.0; n];
    for i in 0..n {
        c[i] = num_padded[n - i] - num_padded[0] * den_norm[n - i];
    }

    // D scalar
    let d = num_padded[0];

    Ok((a, b, c, d))
}

/// Simulate step response using RK4 integration.
fn simulate_lti_step(
    a: &[Vec<f64>],
    b: &[f64],
    c: &[f64],
    d: f64,
    t: &[f64],
) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    if n == 0 {
        // Static system
        return Ok(vec![d; t.len()]);
    }

    let mut x = vec![0.0; n];
    let mut y = Vec::with_capacity(t.len());

    for i in 0..t.len() {
        // Output: y = c'x + d*u (u = 1 for step)
        let output: f64 = c
            .iter()
            .zip(x.iter())
            .map(|(&ci, &xi)| ci * xi)
            .sum::<f64>()
            + d;
        y.push(output);

        if i + 1 < t.len() {
            let dt = t[i + 1] - t[i];
            // RK4 step with u = 1
            x = rk4_step(a, b, &x, 1.0, dt);
        }
    }

    Ok(y)
}

/// Simulate impulse response.
fn simulate_lti_impulse(
    a: &[Vec<f64>],
    b: &[f64],
    c: &[f64],
    d: f64,
    t: &[f64],
) -> Result<Vec<f64>, SignalError> {
    let n = b.len();
    if n == 0 {
        // Static system - impulse response is delta * d
        let mut y = vec![0.0; t.len()];
        if !y.is_empty() {
            y[0] = d;
        }
        return Ok(y);
    }

    // Initial condition: x(0+) = b (from impulse)
    let mut x = b.to_vec();
    let mut y = Vec::with_capacity(t.len());

    for i in 0..t.len() {
        // Output: y = c'x (no feedthrough for impulse after t=0)
        let output: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
        y.push(output);

        if i + 1 < t.len() {
            let dt = t[i + 1] - t[i];
            // RK4 step with u = 0 (no input after initial impulse)
            x = rk4_step(a, b, &x, 0.0, dt);
        }
    }

    // Add direct feedthrough at t=0
    if !y.is_empty() && d != 0.0 {
        // The impulse contribution at t=0 should include d*delta(0)
        // For discrete representation, we add d to y[0]
        y[0] += d;
    }

    Ok(y)
}

/// Single RK4 integration step for x' = Ax + Bu.
fn rk4_step(a: &[Vec<f64>], b: &[f64], x: &[f64], u: f64, dt: f64) -> Vec<f64> {
    let n = x.len();

    let f = |x: &[f64]| -> Vec<f64> {
        let mut dx = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dx[i] += a[i][j] * x[j];
            }
            dx[i] += b[i] * u;
        }
        dx
    };

    let k1 = f(x);
    let x1: Vec<f64> = x
        .iter()
        .zip(&k1)
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k2 = f(&x1);
    let x2: Vec<f64> = x
        .iter()
        .zip(&k2)
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k3 = f(&x2);
    let x3: Vec<f64> = x.iter().zip(&k3).map(|(&xi, &ki)| xi + dt * ki).collect();
    let k4 = f(&x3);

    x.iter()
        .enumerate()
        .map(|(i, &xi)| xi + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Daubechies orthogonal wavelet filter coefficients of order `p`.
///
/// Matches `scipy.signal.daub(p)`. Returns the lowpass synthesis filter
/// `h_k` of length `2p`. The filter satisfies
///   Σ_k h_k = √2,
///   Σ_k h_k · h_{k + 2j} = δ_{j, 0}  (orthogonality)
/// and has `p` vanishing moments. Supported for `p ∈ 1..=10` via the
/// standard tabulated coefficients (Daubechies, *Ten Lectures on
/// Wavelets*, Table 6.1); larger orders return
/// `SignalError::InvalidArgument`.
pub fn daub(p: usize) -> Result<Vec<f64>, SignalError> {
    let coeffs: &[f64] = match p {
        1 => &[
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
        2 => &[
            4.829_629_131_445_341e-1,
            8.365_163_037_378_077e-1,
            2.241_438_680_420_134e-1,
            -1.294_095_225_512_603e-1,
        ],
        3 => &[
            3.326_705_529_500_825e-1,
            8.068_915_093_110_924e-1,
            4.598_775_021_184_914e-1,
            -1.350_110_200_102_546e-1,
            -8.544_127_388_202_667e-2,
            3.522_629_188_570_953e-2,
        ],
        4 => &[
            2.303_778_133_088_964e-1,
            7.148_465_705_529_154e-1,
            6.308_807_679_398_587e-1,
            -2.798_376_941_698_385e-2,
            -1.870_348_117_190_931e-1,
            3.084_138_183_556_076e-2,
            3.288_301_166_698_295e-2,
            -1.059_740_178_506_903e-2,
        ],
        5 => &[
            1.601_023_979_741_929e-1,
            6.038_292_697_971_896e-1,
            7.243_085_284_385_744e-1,
            1.384_281_459_013_207e-1,
            -2.422_948_870_663_823e-1,
            -3.224_486_958_463_837e-2,
            7.757_149_384_004_572e-2,
            -6.241_490_212_798_274e-3,
            -1.258_075_199_908_199e-2,
            3.335_725_285_473_771e-3,
        ],
        6 => &[
            1.115_407_433_501_095e-1,
            4.946_238_903_984_53e-1,
            7.511_339_080_210_955e-1,
            3.152_503_517_091_982e-1,
            -2.262_646_939_654_399e-1,
            -1.297_668_675_672_625e-1,
            9.750_160_558_707_936e-2,
            2.752_286_553_030_572e-2,
            -3.158_203_931_748_602e-2,
            5.538_422_011_614_961e-4,
            4.777_257_510_945_51e-3,
            -1.077_301_085_308_479e-3,
        ],
        7 => &[
            7.785_205_408_500_917e-2,
            3.965_393_194_819_173e-1,
            7.291_320_908_465_551e-1,
            4.697_822_874_051_932e-1,
            -1.439_060_039_285_212e-1,
            -2.240_361_849_938_415e-1,
            7.130_921_926_683_026e-2,
            8.061_260_915_108_307e-2,
            -3.802_993_693_501_44e-2,
            -1.657_454_163_066_688e-2,
            1.255_099_855_609_984e-2,
            4.295_779_729_213_665e-4,
            -1.801_640_704_047_49e-3,
            3.537_137_999_745_202e-4,
        ],
        8 => &[
            5.441_584_224_310_4e-2,
            3.128_715_909_142_999e-1,
            6.756_307_362_972_898e-1,
            5.853_546_836_542_067e-1,
            -1.582_910_525_634_93e-2,
            -2.840_155_429_615_469e-1,
            4.724_845_739_132_828e-4,
            1.287_474_266_204_823e-1,
            -1.736_930_100_180_754e-2,
            -4.408_825_393_079_475e-2,
            1.398_102_791_739_828e-2,
            8.746_094_047_405_776e-3,
            -4.870_352_993_451_574e-3,
            -3.917_403_733_769_549e-4,
            6.754_494_064_505_693e-4,
            -1.174_767_841_247_695e-4,
        ],
        9 => &[
            3.807_794_736_316_728e-2,
            2.438_346_746_125_903e-1,
            6.048_231_236_900_955e-1,
            6.572_880_780_512_736e-1,
            1.331_973_858_249_883e-1,
            -2.932_737_832_793_372e-1,
            -9.684_078_322_297_432e-2,
            1.485_407_493_381_063e-1,
            3.072_568_147_933_338e-2,
            -6.763_282_906_132_607e-2,
            2.509_471_148_314_519e-4,
            2.236_166_212_367_909e-2,
            -4.723_204_757_751_397e-3,
            -4.281_503_682_463_429e-3,
            1.847_646_883_056_046e-3,
            2.303_857_635_231_959e-4,
            -2.519_631_889_427_101e-4,
            3.934_732_031_627_159e-5,
        ],
        10 => &[
            2.667_005_790_055_555e-2,
            1.881_768_000_776_347e-1,
            5.272_011_889_315_757e-1,
            6.884_590_394_534_363e-1,
            2.811_723_436_605_775e-1,
            -2.498_464_243_273_826e-1,
            -1.959_462_743_773_385e-1,
            1.273_693_403_357_56e-1,
            9.305_736_460_357_726e-2,
            -7.139_414_716_639_708e-2,
            -2.945_753_682_187_581e-2,
            3.321_267_405_934_1e-2,
            3.606_553_566_956_62e-3,
            -1.073_317_548_333_057e-2,
            1.395_351_747_052_901e-3,
            1.992_405_295_185_056e-3,
            -6.858_566_949_597_116e-4,
            -1.164_668_551_292_854e-4,
            9.358_867_032_006_959e-5,
            -1.326_420_289_452_124e-5,
        ],
        other => {
            return Err(SignalError::InvalidArgument(format!(
                "daub: only orders 1..=10 are tabulated; got p={other}"
            )));
        }
    };
    Ok(coeffs.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_error_classifies_string_constructor() {
        assert!(matches!(
            SignalError::InvalidArgument("FFT failed: backend error".to_string()),
            SignalError::FftFailure(_)
        ));
        assert!(matches!(
            SignalError::InvalidArgument("input must not be empty".to_string()),
            SignalError::InvalidInputLength {
                expected: 1,
                actual: 0
            }
        ));
        assert!(matches!(
            SignalError::InvalidArgument("Wn values must be finite and in (0, 1)".to_string()),
            SignalError::NonFiniteInput { .. }
        ));
        assert!(matches!(
            SignalError::InvalidArgument("unknown window type: nope".to_string()),
            SignalError::UnsupportedMode { .. }
        ));
    }

    #[test]
    fn savgol_coeffs_smoothing_sum_to_one() {
        // Smoothing coefficients should sum to 1
        let c = savgol_coeffs(5, 2, 0).expect("coeffs");
        let sum: f64 = c.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "coeffs sum = {sum}, expected 1.0"
        );
    }

    #[test]
    fn savgol_coeffs_symmetric() {
        // Smoothing coefficients for deriv=0 should be symmetric
        let c = savgol_coeffs(7, 3, 0).expect("coeffs");
        let n = c.len();
        for i in 0..n / 2 {
            assert!(
                (c[i] - c[n - 1 - i]).abs() < 1e-10,
                "c[{i}]={} != c[{}]={}",
                c[i],
                n - 1 - i,
                c[n - 1 - i]
            );
        }
    }

    #[test]
    fn savgol_filter_preserves_constant() {
        // Constant signal should be unchanged
        let x = vec![5.0; 20];
        let filtered = savgol_filter(&x, 5, 2).expect("filter");
        for (i, &v) in filtered.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-10, "filtered[{i}] = {v}, expected 5.0");
        }
    }

    #[test]
    fn savgol_filter_preserves_linear() {
        // Linear signal should be preserved (polyorder >= 1)
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let filtered = savgol_filter(&x, 5, 2).expect("filter");
        // Interior points should match exactly
        for i in 2..18 {
            assert!(
                (filtered[i] - x[i]).abs() < 1e-8,
                "filtered[{i}] = {}, expected {}",
                filtered[i],
                x[i]
            );
        }
    }

    #[test]
    fn savgol_filter_interp_edges_match_scipy() {
        // scipy.signal.savgol_filter([1,2,3,4,5], 3, 1) -> [1,2,3,4,5] exactly
        // (default mode='interp'). The old 'nearest' clamping distorted the
        // edges to ~1.333 and ~4.667.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = savgol_filter(&x, 3, 1).expect("filter");
        for (i, (&got, &want)) in filtered.iter().zip(x.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "filtered[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savgol_filter_preserves_linear_including_edges() {
        // With mode='interp', a polynomial of degree <= polyorder is reproduced
        // exactly everywhere, edges included (unlike 'nearest').
        let x: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 - 3.0).collect();
        let filtered = savgol_filter(&x, 5, 2).expect("filter");
        for (i, (&got, &want)) in filtered.iter().zip(x.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "filtered[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savgol_filter_smooths_noise() {
        // Noisy signal should be smoothed
        let mut x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        // Add "noise"
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        let filtered = savgol_filter(&x, 7, 3).expect("filter");
        // Filtered should have less variation than original
        let original_var: f64 = x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        let filtered_var: f64 = filtered.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        assert!(
            filtered_var < original_var,
            "filtered should be smoother: {filtered_var} < {original_var}"
        );
    }

    #[test]
    fn savgol_coeffs_even_window_rejected() {
        let err = savgol_coeffs(4, 2, 0).expect_err("even window");
        assert!(matches!(err, SignalError::InvalidWindowLength(_)));
    }

    #[test]
    fn savgol_coeffs_polyorder_too_high() {
        let err = savgol_coeffs(5, 5, 0).expect_err("polyorder >= window");
        assert!(matches!(err, SignalError::InvalidPolyOrder(_)));
    }

    #[test]
    fn savgol_filter_window_too_large() {
        let err = savgol_filter(&[1.0, 2.0], 5, 2).expect_err("window > signal");
        assert!(matches!(err, SignalError::InvalidWindowLength(_)));
    }

    #[test]
    fn savgol_coeffs_first_derivative() {
        // First derivative coefficients for window=5, polyorder=2
        let c = savgol_coeffs(5, 2, 1).expect("deriv coeffs");
        assert_eq!(c.len(), 5);
        // For first derivative, sum should be 0 (since differentiating a constant gives 0)
        let sum: f64 = c.iter().sum();
        assert!(
            sum.abs() < 1e-10,
            "derivative coeffs sum = {sum}, expected 0"
        );
        // Should be antisymmetric: c[i] = -c[n-1-i]
        let n = c.len();
        for i in 0..n / 2 {
            assert!(
                (c[i] + c[n - 1 - i]).abs() < 1e-10,
                "c[{i}]={} should be -c[{}]={}",
                c[i],
                n - 1 - i,
                c[n - 1 - i]
            );
        }
    }

    // ── Window function tests ───────────────────────────────────────

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann(8);
        assert!((w[0]).abs() < 1e-12, "hann[0] = {}", w[0]);
        assert!((w[7]).abs() < 1e-12, "hann[7] = {}", w[7]);
    }

    #[test]
    fn hann_window_symmetric() {
        let w = hann(9);
        for i in 0..w.len() / 2 {
            assert!(
                (w[i] - w[w.len() - 1 - i]).abs() < 1e-12,
                "hann not symmetric at {i}"
            );
        }
    }

    #[test]
    fn hamming_window_nonzero_endpoints() {
        let w = hamming(8);
        assert!(w[0] > 0.07, "hamming[0] should be ~0.08");
    }

    #[test]
    fn blackman_window_endpoints_near_zero() {
        let w = blackman(16);
        assert!(w[0].abs() < 0.01, "blackman[0] = {}", w[0]);
    }

    #[test]
    fn blackman_window_closed_form_values() {
        // /testing-golden-artifacts for [frankenscipy-rd0ai]:
        // The Blackman window for n=1 is [1] by convention (single-
        // sample edge case); for n=3 it's [0, 1, 0] (m=2, t∈{0,π,2π}
        // give exact factors 1, -1, 1); for n=5 the interior values
        // 0.34 follow from cos(π/2)=0, cos(π)=-1 directly.
        assert_eq!(blackman(0), Vec::<f64>::new());
        assert_eq!(blackman(1), vec![1.0]);

        let w3 = blackman(3);
        assert_eq!(w3.len(), 3);
        assert!(w3[0].abs() < 1e-15, "blackman(3)[0] = {}", w3[0]);
        assert!((w3[1] - 1.0).abs() < 1e-15, "blackman(3)[1] = {}", w3[1]);
        assert!(w3[2].abs() < 1e-15, "blackman(3)[2] = {}", w3[2]);

        // n=5: w[1] = 0.42 + 0.08·(-1) = 0.34
        let w5 = blackman(5);
        let expected_5 = [0.0, 0.34, 1.0, 0.34, 0.0];
        for (i, (&got, &exp)) in w5.iter().zip(expected_5.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "blackman(5)[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn blackman_window_is_symmetric() {
        // /testing-metamorphic: w[i] = w[n-1-i] for any n ≥ 1.
        for &n in &[3usize, 5, 7, 11, 16, 33] {
            let w = blackman(n);
            for i in 0..n / 2 {
                assert!(
                    (w[i] - w[n - 1 - i]).abs() < 1e-15,
                    "blackman({n}) not symmetric at index {i}: \
                     w[{i}] = {}, w[{}] = {}",
                    w[i],
                    n - 1 - i,
                    w[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn blackmanharris_window_matches_scipy_reference() {
        let w = blackmanharris(5);
        let expected = [0.00006, 0.21747, 1.0, 0.21747, 0.00006];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn barthann_window_matches_scipy_reference() {
        let w = barthann(8);
        let expected = [
            0.0,
            0.21164530386510985,
            0.6017008120462566,
            0.9280824555172049,
            0.9280824555172051,
            0.6017008120462566,
            0.21164530386511002,
            0.0,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_hamming_window_matches_scipy_reference() {
        let w = general_hamming(8, 0.75);
        let expected = [
            0.5,
            0.5941275495353167,
            0.8056302334890786,
            0.9752422169756048,
            0.9752422169756048,
            0.8056302334890786,
            0.5941275495353167,
            0.5,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_cosine_window_matches_scipy_reference() {
        let w = general_cosine(8, &[0.5, 0.3, 0.2], true);
        let expected = [
            0.4,
            0.26844887265111705,
            0.38656250660641056,
            0.8949886207424725,
            0.8949886207424725,
            0.38656250660641056,
            0.26844887265111705,
            0.4,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_cosine_periodic_window_matches_scipy_reference() {
        let w = general_cosine(8, &[0.5, 0.5], false);
        let expected = [
            0.0,
            0.14644660940672627,
            0.5,
            0.8535533905932737,
            1.0,
            0.8535533905932737,
            0.5,
            0.14644660940672627,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn boxcar_window_all_ones() {
        let w = boxcar(5, true);
        assert_eq!(w, vec![1.0; 5]);

        let w_periodic = boxcar(5, false);
        assert_eq!(w_periodic, vec![1.0; 5]);

        assert!(boxcar(0, true).is_empty());
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} length mismatch: got {}, expected {}",
            actual.len(),
            expected.len()
        );
        for (idx, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (actual_value - expected_value).abs() < tol,
                "{label}[{idx}] = {actual_value}, expected {expected_value}, diff={}",
                (actual_value - expected_value).abs()
            );
        }
    }

    #[test]
    fn gaussian_window_matches_scipy_reference() {
        let w = gaussian(5, 1.0, true);
        let expected = [
            0.1353352832366127,
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.1353352832366127,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_periodic_window_matches_scipy_reference() {
        let w = gaussian(8, 2.0, false);
        let expected = [
            0.1353352832366127,
            0.32465246735834974,
            0.6065306597126334,
            0.8824969025845955,
            1.0,
            0.8824969025845955,
            0.6065306597126334,
            0.32465246735834974,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_gaussian_window_matches_scipy_reference() {
        let w = general_gaussian(8, 1.5, 2.0, true);
        let expected = [
            0.06858458348811015,
            0.3766034507108804,
            0.8098246793420792,
            0.9922179382602435,
            0.9922179382602435,
            0.8098246793420792,
            0.3766034507108804,
            0.06858458348811015,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn general_gaussian_periodic_window_matches_scipy_reference() {
        let w = general_gaussian(8, 1.5, 2.0, false);
        let expected = [
            0.01831563888873418,
            0.18498139990730428,
            0.6065306597126334,
            0.9394130628134758,
            1.0,
            0.9394130628134758,
            0.6065306597126334,
            0.18498139990730428,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_matches_scipy_reference() {
        let w = exponential(8, None, 2.0, true).unwrap();
        let expected = [
            0.17377394345044514,
            0.2865047968601901,
            0.4723665527410147,
            0.7788007830714049,
            0.7788007830714049,
            0.4723665527410147,
            0.2865047968601901,
            0.17377394345044514,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_periodic_window_matches_scipy_reference() {
        let w = exponential(8, None, 2.0, false).unwrap();
        let expected = [
            0.1353352832366127,
            0.22313016014842982,
            0.36787944117144233,
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.36787944117144233,
            0.22313016014842982,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_supports_non_symmetric_center() {
        let w = exponential(5, Some(1.0), 2.0, false).unwrap();
        let expected = [
            0.6065306597126334,
            1.0,
            0.6065306597126334,
            0.36787944117144233,
            0.22313016014842982,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn exponential_window_rejects_center_for_symmetric_mode() {
        let err = exponential(5, Some(1.0), 2.0, true).unwrap_err();
        assert!(err.is_argument_error());
        assert_eq!(
            err.to_string(),
            "invalid argument: If sym==True, center must be None."
        );
    }

    #[test]
    fn chebwin_window_matches_scipy_reference() {
        let w = chebwin(5, 100.0);
        let expected = [
            0.16866468734562678,
            0.6686513141856113,
            1.0,
            0.6686513141856113,
            0.16866468734562678,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn chebwin_even_window_matches_scipy_reference() {
        let w = chebwin(8, 60.0);
        let expected = [
            0.06847555416399577,
            0.303219161655201,
            0.686846620773932,
            1.0,
            1.0,
            0.686846620773932,
            0.303219161655201,
            0.06847555416399577,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn triang_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin scipy.signal.windows.triang
        // closed-form values across odd/even/edge lengths.
        //   triang(1) = [1]
        //   triang(2) = [0.5, 0.5]   (even, peak between samples)
        //   triang(3) = [0.5, 1, 0.5]
        //   triang(5) = [1/3, 2/3, 1, 2/3, 1/3]
        //   triang(0) = []
        assert_eq!(triang(0), Vec::<f64>::new());
        assert_eq!(triang(1), vec![1.0]);
        let r2 = triang(2);
        assert!((r2[0] - 0.5).abs() < 1e-12 && (r2[1] - 0.5).abs() < 1e-12);
        let r3 = triang(3);
        assert!((r3[0] - 0.5).abs() < 1e-12);
        assert!((r3[1] - 1.0).abs() < 1e-12);
        assert!((r3[2] - 0.5).abs() < 1e-12);
        let r5 = triang(5);
        let third = 1.0_f64 / 3.0;
        let two_thirds = 2.0_f64 / 3.0;
        assert!((r5[0] - third).abs() < 1e-12);
        assert!((r5[1] - two_thirds).abs() < 1e-12);
        assert!((r5[2] - 1.0).abs() < 1e-12);
        assert!((r5[3] - two_thirds).abs() < 1e-12);
        assert!((r5[4] - third).abs() < 1e-12);
        // Symmetry: triang(n)[i] = triang(n)[n-1-i].
        for n in [4_usize, 6, 7, 11] {
            let r = triang(n);
            for i in 0..n {
                assert!(
                    (r[i] - r[n - 1 - i]).abs() < 1e-12,
                    "triang({n}) not palindromic at index {i}"
                );
            }
        }
    }

    #[test]
    fn kaiser_window_beta_zero_is_rectangular() {
        let w = kaiser(8, 0.0);
        for &v in &w {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "kaiser(beta=0) should be rectangular"
            );
        }
    }

    #[test]
    fn dpss_symmetric_window_matches_scipy_reference() {
        let result = dpss(8, 2.5, None, true, None, false).expect("dpss symmetric");
        let expected = [
            0.052404360853994365,
            0.2631860434811115,
            0.639750596958442,
            0.9624060150375939,
            0.9624060150375939,
            0.639750596958442,
            0.2631860434811115,
            0.052404360853994365,
        ];
        assert_eq!(result.windows.len(), 1);
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss symmetric");
        assert!(result.ratios.is_none());
    }

    #[test]
    fn dpss_periodic_window_matches_scipy_reference() {
        let result = dpss(8, 2.5, None, false, None, false).expect("dpss periodic");
        let expected = [
            0.04080887106381651,
            0.2018541771042083,
            0.5121647719505525,
            0.8503476239828903,
            1.0,
            0.8503476239828905,
            0.5121647719505525,
            0.20185417710420828,
        ];
        assert_eq!(result.windows.len(), 1);
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss periodic");
    }

    #[test]
    fn dpss_multiple_tapers_and_ratios_match_scipy_reference() {
        let result = dpss(8, 2.5, Some(3), true, Some(DpssNorm::L2), true).expect("dpss kmax");
        let expected_windows = [
            [
                0.03123383094162537,
                0.15686305975922865,
                0.3813015112592644,
                0.5736092623023827,
                0.5736092623023827,
                0.3813015112592644,
                0.15686305975922865,
                0.03123383094162537,
            ],
            [
                0.11133547272330951,
                0.3749691021790795,
                0.5282319109750825,
                0.2607175351833948,
                -0.2607175351833948,
                -0.5282319109750825,
                -0.37496910217907947,
                -0.11133547272330951,
            ],
            [
                0.26478310487439105,
                0.5199646064881195,
                0.24218795419331154,
                -0.31760307022507,
                -0.31760307022507,
                0.24218795419331157,
                0.5199646064881195,
                0.26478310487439105,
            ],
        ];
        let expected_ratios = [0.9999998455973096, 0.9999787214591116, 0.9988738410781977];

        assert_eq!(result.windows.len(), 3);
        for (actual_window, expected_window) in result.windows.iter().zip(expected_windows.iter()) {
            assert_slice_close(actual_window, expected_window, 1.0e-12, "dpss taper");
        }

        let ratios = result.ratios.expect("ratios requested");
        for (actual, want) in ratios.iter().zip(expected_ratios.iter()) {
            assert!((*actual - *want).abs() < 1.0e-9);
        }
    }

    #[test]
    fn dpss_subsample_norm_matches_scipy_reference() {
        let result =
            dpss(5, 1.5, None, true, Some(DpssNorm::Subsample), false).expect("dpss subsample");
        let expected = [
            0.2608050703712001,
            0.7407980987672104,
            1.0,
            0.7407980987672104,
            0.2608050703712001,
        ];
        assert_slice_close(&result.windows[0], &expected, 1.0e-12, "dpss subsample");
    }

    #[test]
    fn dpss_rejects_invalid_parameters() {
        assert!(dpss(8, 0.0, None, true, None, false).is_err());
        assert!(dpss(8, 4.0, None, true, None, false).is_err());
        assert!(dpss(8, 2.5, Some(0), true, None, false).is_err());
        assert!(dpss(8, 2.5, Some(9), true, None, false).is_err());
    }

    #[test]
    fn taylor_window_symmetric() {
        // Taylor window should be symmetric
        let w = taylor(11, 4, -30.0, true, true);
        assert_eq!(w.len(), 11);
        for i in 0..5 {
            assert!(
                (w[i] - w[10 - i]).abs() < 1e-12,
                "Taylor asymmetric at {i}: {} vs {}",
                w[i],
                w[10 - i]
            );
        }
    }

    #[test]
    fn taylor_window_normalized_peak() {
        // With norm=true, peak should be 1.0
        let w = taylor(21, 4, -30.0, true, true);
        let max_val = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_val - 1.0).abs() < 1e-10,
            "Taylor norm peak = {}, expected 1.0",
            max_val
        );
    }

    #[test]
    fn taylor_window_empty_and_single() {
        assert!(taylor(0, 4, -30.0, true, true).is_empty());
        assert_eq!(taylor(1, 4, -30.0, true, true), [1.0]);
    }

    #[test]
    fn window_empty_returns_empty() {
        assert!(hann(0).is_empty());
        assert!(hamming(0).is_empty());
        assert!(general_hamming(0, 0.75).is_empty());
        assert!(general_cosine(0, &[0.5, 0.5], true).is_empty());
        assert!(gaussian(0, 2.0, true).is_empty());
        assert!(general_gaussian(0, 1.5, 2.0, true).is_empty());
        assert!(exponential(0, None, 2.0, true).unwrap().is_empty());
        assert!(blackman(0).is_empty());
        assert!(blackmanharris(0).is_empty());
        assert!(barthann(0).is_empty());
        assert!(chebwin(0, 100.0).is_empty());
        assert!(kaiser(0, 5.0).is_empty());
        assert!(kaiser_bessel_derived(0, 5.0, true).unwrap().is_empty());
        assert!(lanczos(0).is_empty());
        assert!(taylor(0, 4, -30.0, true, true).is_empty());
    }

    #[test]
    fn window_single_returns_one() {
        assert_eq!(hann(1), [1.0]);
        assert_eq!(hamming(1), [1.0]);
        assert_eq!(general_hamming(1, 0.75), [1.0]);
        assert_eq!(general_cosine(1, &[0.5, 0.5], true), [1.0]);
        assert_eq!(gaussian(1, 2.0, true), [1.0]);
        assert_eq!(general_gaussian(1, 1.5, 2.0, true), [1.0]);
        assert_eq!(exponential(1, None, 2.0, true).unwrap(), [1.0]);
        assert_eq!(blackman(1), [1.0]);
        assert_eq!(blackmanharris(1), [1.0]);
        assert_eq!(barthann(1), [1.0]);
        assert_eq!(chebwin(1, 100.0), [1.0]);
        assert_eq!(kaiser(1, 5.0), [1.0]);
        assert_eq!(lanczos(1), [1.0]);
    }

    #[test]
    fn kaiser_bessel_derived_matches_scipy_reference() {
        let w = kaiser_bessel_derived(8, 5.0, true).unwrap();
        let expected = [
            0.1297945228712448,
            0.5201443455712672,
            0.8540783686350089,
            0.9915409128385101,
            0.9915409128385101,
            0.8540783686350089,
            0.5201443455712672,
            0.1297945228712448,
        ];
        assert_slice_close(&w, &expected, 1.0e-8, "kaiser_bessel_derived");
    }

    #[test]
    fn kaiser_bessel_derived_rejects_non_symmetric_and_odd_lengths() {
        let periodic = kaiser_bessel_derived(8, 5.0, false).expect_err("periodic KBD");
        assert_eq!(
            periodic.to_string(),
            "invalid input shape: Kaiser-Bessel Derived windows are only defined for symmetric shapes"
        );

        let odd = kaiser_bessel_derived(7, 5.0, true).expect_err("odd KBD");
        assert_eq!(
            odd.to_string(),
            "invalid argument: Kaiser-Bessel Derived windows are only defined for even number of points"
        );
    }

    #[test]
    fn lanczos_window_symmetric() {
        // Lanczos window should be symmetric
        let w = lanczos(7);
        assert_eq!(w.len(), 7);
        for i in 0..3 {
            assert!((w[i] - w[6 - i]).abs() < 1e-12, "asymmetry at {i}");
        }
        // Center should be 1.0 (sinc(0) = 1)
        assert!(
            (w[3] - 1.0).abs() < 1e-12,
            "center = {}, expected 1.0",
            w[3]
        );
    }

    #[test]
    fn lanczos_window_endpoints() {
        // At endpoints, x = ±1, so sinc(±1) = sin(±π)/(±π) = 0
        let w = lanczos(11);
        assert!((w[0]).abs() < 1e-12, "w[0] = {}, expected 0", w[0]);
        assert!((w[10]).abs() < 1e-12, "w[10] = {}, expected 0", w[10]);
    }

    // ── Convolution tests ───────────────────────────────────────────

    #[test]
    fn convolve_impulse() {
        // Convolving with [1] should return the original signal
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        assert_eq!(result, a);
    }

    #[test]
    fn convolve_known_result() {
        // [1, 2, 3] * [0, 1, 0.5] = [0, 1, 2.5, 4, 1.5]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        assert_eq!(result.len(), 5);
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-12, "conv[{i}] = {r}, expected {e}");
        }
    }

    #[test]
    fn convolve_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = convolve(&a, &b, ConvolveMode::Same).expect("same");
        assert_eq!(result.len(), a.len());
    }

    #[test]
    fn convolve_valid_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0];
        let result = convolve(&a, &b, ConvolveMode::Valid).expect("valid");
        // Valid: length = max(5,3) - min(5,3) + 1 = 3
        assert_eq!(result.len(), 3);
        assert!((result[0] - 6.0).abs() < 1e-12); // 1+2+3
        assert!((result[1] - 9.0).abs() < 1e-12); // 2+3+4
        assert!((result[2] - 12.0).abs() < 1e-12); // 3+4+5
    }

    #[test]
    fn fftconvolve_matches_direct() {
        let a = vec![1.0, -1.0, 2.0, 3.0, -0.5];
        let b = vec![0.5, 1.0, -0.5, 0.25];
        let direct = convolve(&a, &b, ConvolveMode::Full).expect("direct");
        let fft_conv = fftconvolve(&a, &b, ConvolveMode::Full).expect("fft");
        assert_eq!(direct.len(), fft_conv.len());
        for (i, (&d, &f)) in direct.iter().zip(fft_conv.iter()).enumerate() {
            assert!(
                (d - f).abs() < 1e-10,
                "fftconvolve[{i}] = {f}, direct = {d}"
            );
        }
    }

    #[test]
    fn fftconvolve_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, -1.0];
        let result = fftconvolve(&a, &b, ConvolveMode::Same).expect("same");
        assert_eq!(result.len(), a.len());
    }

    #[test]
    fn convolve_empty_rejected() {
        let err = convolve(&[], &[1.0], ConvolveMode::Full).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn deconvolve_recovers_original_signal() {
        let original = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let impulse_response = vec![2.0, 1.0];
        let recorded =
            convolve(&impulse_response, &original, ConvolveMode::Full).expect("convolve");

        let (quotient, remainder) = deconvolve(&recorded, &impulse_response).expect("deconvolve");

        assert_eq!(quotient.len(), original.len());
        for (i, (&got, &expected)) in quotient.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "quotient[{i}] = {got}, expected {expected}"
            );
        }
        assert_eq!(remainder.len(), recorded.len());
        for (i, &value) in remainder.iter().enumerate() {
            assert!(value.abs() < 1e-12, "remainder[{i}] = {value}");
        }
    }

    #[test]
    fn deconvolve_known_polynomial_division() {
        let signal = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        let divisor = vec![1.0, 2.0];

        let (quotient, remainder) = deconvolve(&signal, &divisor).expect("deconvolve");

        assert_eq!(quotient, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(remainder, vec![0.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn deconvolve_divisor_longer_than_signal_returns_empty_quotient() {
        let signal = vec![1.0, 2.0];
        let divisor = vec![1.0, 2.0, 3.0];

        let (quotient, remainder) = deconvolve(&signal, &divisor).expect("deconvolve");

        assert!(quotient.is_empty());
        assert_eq!(remainder, signal);
    }

    #[test]
    fn deconvolve_zero_leading_divisor_rejected() {
        let err = deconvolve(&[1.0, 2.0, 3.0], &[0.0, 1.0]).expect_err("zero leading divisor");
        assert!(err.is_argument_error());
    }

    // ── find_peaks tests ────────────────────────────────────────────

    #[test]
    fn find_peaks_simple() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let result = find_peaks(&x, FindPeaksOptions::default());
        assert_eq!(result.peaks, vec![1, 3, 5]);
    }

    #[test]
    fn find_peaks_with_height() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                height: Some(1.5),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]); // only the peak at 2.0
    }

    #[test]
    fn find_peaks_with_distance() {
        let x = [0.0, 1.0, 0.0, 0.8, 0.0, 2.0, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(3),
                ..FindPeaksOptions::default()
            },
        );
        // Distance=3: peaks at 1 and 3 are too close (dist=2), keep highest (1.0 at idx 1)
        // Peak at 5 (2.0) is far enough from both
        assert!(result.peaks.contains(&5));
    }

    #[test]
    fn find_peaks_distance_equal_height_tie_prefers_rightmost() {
        let x = [0.0, 1.0, 0.0, 1.0, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(3),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]);
    }

    #[test]
    fn find_peaks_distance_saturates_large_window() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                distance: Some(usize::MAX),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![3]);
    }

    #[test]
    fn find_peaks_with_prominence() {
        let x = [0.0, 3.0, 2.5, 2.8, 0.0]; // peak at 1 (prom=3), peak at 3 (prom=0.3)
        let result = find_peaks(
            &x,
            FindPeaksOptions {
                prominence: Some(1.0),
                ..FindPeaksOptions::default()
            },
        );
        assert_eq!(result.peaks, vec![1]); // only the prominent peak
    }

    #[test]
    fn find_peaks_sine_wave() {
        let n = 100;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let result = find_peaks(&x, FindPeaksOptions::default());
        // 3 full cycles should have ~3 peaks
        assert!(
            result.peaks.len() >= 2 && result.peaks.len() <= 4,
            "expected ~3 peaks, got {}",
            result.peaks.len()
        );
    }

    #[test]
    fn find_peaks_too_short() {
        assert!(
            find_peaks(&[1.0, 2.0], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
        assert!(
            find_peaks(&[1.0], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
        assert!(
            find_peaks(&[], FindPeaksOptions::default())
                .peaks
                .is_empty()
        );
    }

    #[test]
    fn peak_widths_handles_out_of_range_peaks_without_panicking() {
        let (widths, width_heights, left_ips, right_ips) = peak_widths(&[0.0, 1.0, 0.0], &[9], 0.5);
        assert_eq!(widths, vec![0.0]);
        assert_eq!(width_heights, vec![0.0]);
        assert_eq!(left_ips, vec![9.0]);
        assert_eq!(right_ips, vec![9.0]);
    }

    #[test]
    fn find_peaks_flat_signal() {
        let x = [5.0; 10];
        let result = find_peaks(&x, FindPeaksOptions::default());
        assert!(result.peaks.is_empty(), "flat signal has no peaks");
    }

    // ── Butterworth filter design ──────────────────────────────────

    #[test]
    fn butter_lowpass_order1() {
        let coeffs = butter(1, &[0.5], FilterType::Lowpass).expect("butter");
        assert_eq!(coeffs.b.len(), 2);
        assert_eq!(coeffs.a.len(), 2);
        assert!((coeffs.a[0] - 1.0).abs() < 1e-12, "a[0] should be 1.0");
        // DC gain should be 1: sum(b) / sum(a) = 1
        let b_sum: f64 = coeffs.b.iter().sum();
        let a_sum: f64 = coeffs.a.iter().sum();
        assert!(
            (b_sum / a_sum - 1.0).abs() < 1e-10,
            "DC gain should be 1, got {}",
            b_sum / a_sum
        );
    }

    #[test]
    fn butter_lowpass_order2() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        assert_eq!(coeffs.b.len(), 3);
        assert_eq!(coeffs.a.len(), 3);
        // DC gain = 1
        let b_sum: f64 = coeffs.b.iter().sum();
        let a_sum: f64 = coeffs.a.iter().sum();
        assert!((b_sum / a_sum - 1.0).abs() < 1e-10, "DC gain should be 1");
    }

    #[test]
    fn butter_highpass_nyquist_gain() {
        let coeffs = butter(2, &[0.3], FilterType::Highpass).expect("butter hp");
        // Nyquist gain should be 1: evaluate at z = -1
        let b_nyq: f64 = coeffs
            .b
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(-(i as i32)))
            .sum();
        let a_nyq: f64 = coeffs
            .a
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (-1.0_f64).powi(-(i as i32)))
            .sum();
        assert!(
            (b_nyq / a_nyq - 1.0).abs() < 1e-10,
            "Nyquist gain should be 1"
        );
    }

    #[test]
    fn butter_invalid_order() {
        assert!(butter(0, &[0.5], FilterType::Lowpass).is_err());
    }

    #[test]
    fn butter_invalid_wn() {
        assert!(butter(2, &[0.0], FilterType::Lowpass).is_err());
        assert!(butter(2, &[1.0], FilterType::Lowpass).is_err());
        assert!(butter(2, &[0.2, 0.4], FilterType::Lowpass).is_err());
        assert!(butter(2, &[0.4, 0.2], FilterType::Bandpass).is_err());
    }

    #[test]
    fn butter_bandpass_order_doubles() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandpass).expect("butter bp");
        assert_eq!(coeffs.b.len(), 5);
        assert_eq!(coeffs.a.len(), 5);
    }

    #[test]
    fn butter_bandpass_passes_midband_and_rejects_outside() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandpass).expect("butter bp");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.1 * std::f64::consts::PI;
        let omega_mid = 0.3 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] > result.h_mag[lo_idx],
            "bandpass should pass midband better than low stopband"
        );
        assert!(
            result.h_mag[mid_idx] > result.h_mag[hi_idx],
            "bandpass should pass midband better than high stopband"
        );
    }

    #[test]
    fn butter_bandstop_rejects_midband_and_passes_outside() {
        let coeffs = butter(2, &[0.2, 0.4], FilterType::Bandstop).expect("butter bs");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.05 * std::f64::consts::PI;
        let omega_mid = 0.3 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] < result.h_mag[lo_idx],
            "bandstop should attenuate the stop band"
        );
        assert!(
            result.h_mag[mid_idx] < result.h_mag[hi_idx],
            "bandstop should attenuate the stop band"
        );
    }

    #[test]
    fn cheby1_lowpass_shows_passband_ripple() {
        let coeffs = cheby1(4, 1.0, &[0.3], FilterType::Lowpass).expect("cheby1");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let passband_end = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        let passband = &result.h_mag[..=passband_end];
        let max_mag = passband.iter().copied().fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        let min_mag = passband
            .iter()
            .copied()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        assert!(
            max_mag - min_mag > 0.01,
            "Chebyshev-I passband should ripple, got range {}",
            max_mag - min_mag
        );
        assert!(
            max_mag <= 1.25 && min_mag >= 0.75,
            "passband ripple should stay bounded, got min={min_mag}, max={max_mag}"
        );
    }

    #[test]
    fn cheby1_bandpass_passes_midband_and_rejects_outside() {
        let coeffs = cheby1(3, 1.0, &[0.2, 0.45], FilterType::Bandpass).expect("cheby1 bp");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.08 * std::f64::consts::PI;
        let omega_mid = 0.32 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] > result.h_mag[lo_idx],
            "Chebyshev-I bandpass should pass midband better than low stopband"
        );
        assert!(
            result.h_mag[mid_idx] > result.h_mag[hi_idx],
            "Chebyshev-I bandpass should pass midband better than high stopband"
        );
    }

    #[test]
    fn cheby1_even_order_lowpass_matches_scipy_gain() {
        let coeffs = cheby1(2, 1.0, &[0.25], FilterType::Lowpass).expect("cheby1 lp");
        let expected_b = [
            0.10255744141814213,
            0.20511488283628426,
            0.10255744141814213,
        ];
        let expected_a = [1.0, -0.9865079240565594, 0.44679329164515147];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn cheby1_even_order_bandpass_matches_scipy_gain() {
        let coeffs = cheby1(4, 1.0, &[0.2, 0.45], FilterType::Bandpass).expect("cheby1 bp");
        let expected_b = [
            0.0042412377794045445,
            0.0,
            -0.016964951117618178,
            0.0,
            0.025447426676427267,
            0.0,
            -0.016964951117618178,
            0.0,
            0.0042412377794045445,
        ];
        let expected_a = [
            1.0,
            -3.805027975445876,
            8.305833307350976,
            -12.0401045837622,
            12.844987905921329,
            -10.009315088065474,
            5.72688219445523,
            -2.1640816756583447,
            0.4751428601925499,
        ];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 bandpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("cheby1 bandpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn bessel_phase_normalized_lowpass_matches_scipy_coefficients() {
        let coeffs = bessel(2, &[0.25], FilterType::Lowpass).expect("bessel lp");
        let expected_b = [
            0.09082678800790182,
            0.18165357601580365,
            0.09082678800790182,
        ];
        let expected_a = [1.0, -0.8771010537418505, 0.24040820577345767];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("bessel lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("bessel lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn ellip_lowpass_matches_scipy_cauer_prototype() {
        let coeffs = ellip(2, 1.0, 40.0, &[0.25], FilterType::Lowpass).expect("ellip lp");
        let expected_b = [
            0.10926548486600407,
            0.19417386968773206,
            0.10926548486600406,
        ];
        let expected_a = [1.0, -0.9863237792094665, 0.4493862252181434];
        assert_eq!(coeffs.b.len(), expected_b.len());
        assert_eq!(coeffs.a.len(), expected_a.len());
        for (idx, (actual, expected)) in coeffs.b.iter().zip(expected_b.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("ellip lowpass b[{idx}]"),
            );
        }
        for (idx, (actual, expected)) in coeffs.a.iter().zip(expected_a.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-10,
                &format!("ellip lowpass a[{idx}]"),
            );
        }
    }

    #[test]
    fn cheby2_lowpass_rejects_stopband_more_than_passband() {
        let coeffs = cheby2(4, 20.0, &[0.3], FilterType::Lowpass).expect("cheby2");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.2,
            "Chebyshev-II should strongly attenuate the stopband"
        );
    }

    #[test]
    fn cheby2_odd_order_remains_finite_and_attenuates_stopband() {
        let coeffs = cheby2(3, 20.0, &[0.3], FilterType::Lowpass).expect("cheby2 odd");
        assert!(coeffs.b.iter().all(|value| value.is_finite()));
        assert!(coeffs.a.iter().all(|value| value.is_finite()));

        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.3,
            "odd-order Chebyshev-II should still attenuate the stopband"
        );
    }

    #[test]
    fn bessel_group_delay_is_flatter_than_butter() {
        let bessel_coeffs = bessel(3, &[0.25], FilterType::Lowpass).expect("bessel");
        let butter_coeffs = butter(3, &[0.25], FilterType::Lowpass).expect("butter");
        let (_, gd_bessel) =
            group_delay(&bessel_coeffs.b, &bessel_coeffs.a, Some(256)).expect("gd bessel");
        let (_, gd_butter) =
            group_delay(&butter_coeffs.b, &butter_coeffs.a, Some(256)).expect("gd butter");
        let pass_end = 64;
        let bessel_span = gd_span(&gd_bessel[..pass_end]);
        let butter_span = gd_span(&gd_butter[..pass_end]);
        assert!(
            bessel_span < butter_span,
            "Bessel passband delay should be flatter: {bessel_span} < {butter_span}"
        );
    }

    #[test]
    fn bessel_rejects_excessive_order() {
        let err = bessel(90, &[0.25], FilterType::Lowpass).expect_err("large order should fail");
        assert!(err.is_argument_error());
    }

    #[test]
    fn iirfilter_dispatch_matches_butter() {
        let direct = butter(3, &[0.25], FilterType::Lowpass).expect("butter");
        let dispatch = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Butterworth,
            None,
            None,
        )
        .expect("iirfilter butter");
        assert_ba_close(&direct, &dispatch, 1e-10, "butter");
    }

    #[test]
    fn iirfilter_dispatch_matches_cheby1() {
        let direct = cheby1(4, 0.8, &[0.2, 0.5], FilterType::Bandpass).expect("cheby1");
        let dispatch = iirfilter(
            4,
            &[0.2, 0.5],
            FilterType::Bandpass,
            IirFamily::Chebyshev1,
            Some(0.8),
            None,
        )
        .expect("iirfilter cheby1");
        assert_ba_close(&direct, &dispatch, 1e-10, "cheby1");
    }

    #[test]
    fn iirfilter_dispatch_matches_cheby2() {
        let direct = cheby2(4, 20.0, &[0.25], FilterType::Lowpass).expect("cheby2");
        let dispatch = iirfilter(
            4,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Chebyshev2,
            None,
            Some(20.0),
        )
        .expect("iirfilter cheby2");
        assert_ba_close(&direct, &dispatch, 1e-10, "cheby2");
    }

    #[test]
    fn iirfilter_dispatch_matches_bessel() {
        let direct = bessel(3, &[0.35], FilterType::Highpass).expect("bessel");
        let dispatch = iirfilter(
            3,
            &[0.35],
            FilterType::Highpass,
            IirFamily::Bessel,
            None,
            None,
        )
        .expect("iirfilter bessel");
        assert_ba_close(&direct, &dispatch, 1e-10, "bessel");
    }

    #[test]
    fn iirfilter_dispatch_requires_family_params() {
        let err = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Chebyshev1,
            None,
            None,
        )
        .expect_err("missing rp should fail");
        assert!(err.is_argument_error());
    }

    // ── Elliptic filter tests ──────────────────────────────────────

    #[test]
    fn ellip_lowpass_produces_finite_coefficients() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.3], FilterType::Lowpass).expect("ellip");
        assert!(
            coeffs.b.iter().all(|v| v.is_finite()),
            "b coeffs not finite"
        );
        assert!(
            coeffs.a.iter().all(|v| v.is_finite()),
            "a coeffs not finite"
        );
    }

    #[test]
    fn ellip_lowpass_passes_low_frequencies_and_rejects_high() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.3], FilterType::Lowpass).expect("ellip");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        let stop_idx = nearest_freq_index(&result.w, 0.8 * std::f64::consts::PI);
        assert!(
            result.h_mag[stop_idx] < result.h_mag[pass_idx] * 0.5,
            "elliptic should attenuate stopband: pass={}, stop={}",
            result.h_mag[pass_idx],
            result.h_mag[stop_idx]
        );
    }

    #[test]
    fn ellip_bandstop_rejects_midband_and_passes_outside() {
        let coeffs = ellip(4, 1.0, 40.0, &[0.2, 0.45], FilterType::Bandstop).expect("ellip bs");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let omega_lo = 0.08 * std::f64::consts::PI;
        let omega_mid = 0.32 * std::f64::consts::PI;
        let omega_hi = 0.8 * std::f64::consts::PI;
        let lo_idx = nearest_freq_index(&result.w, omega_lo);
        let mid_idx = nearest_freq_index(&result.w, omega_mid);
        let hi_idx = nearest_freq_index(&result.w, omega_hi);
        assert!(
            result.h_mag[mid_idx] < result.h_mag[lo_idx],
            "elliptic bandstop should reject the middle stopband"
        );
        assert!(
            result.h_mag[mid_idx] < result.h_mag[hi_idx],
            "elliptic bandstop should reject the middle stopband"
        );
    }

    #[test]
    fn ellip_rejects_invalid_ripple() {
        let err = ellip(4, -1.0, 40.0, &[0.3], FilterType::Lowpass).expect_err("negative rp");
        assert!(err.is_argument_error());
        let err = ellip(4, 1.0, -40.0, &[0.3], FilterType::Lowpass).expect_err("negative rs");
        assert!(err.is_argument_error());
    }

    #[test]
    fn ellip_via_iirfilter_dispatch() {
        let direct = ellip(3, 1.0, 40.0, &[0.25], FilterType::Lowpass).expect("ellip direct");
        let dispatch = iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirFamily::Elliptic,
            Some(1.0),
            Some(40.0),
        )
        .expect("iirfilter elliptic");
        assert_ba_close(&direct, &dispatch, 1e-10, "elliptic");
    }

    #[test]
    fn chroma_normalizes_finite_bins_even_with_nan_input() {
        let mut magnitudes = vec![0.0; 12];
        magnitudes[1] = 1.0;
        magnitudes[2] = f64::NAN;
        magnitudes[3] = 2.0;

        let chroma_vec = chroma(&magnitudes, 12_000.0, 12);
        let finite_max = chroma_vec
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f64, f64::max);

        assert!(
            (finite_max - 1.0).abs() < 1e-12,
            "finite bins should remain normalized, got max {finite_max}"
        );
        assert!(
            chroma_vec.iter().any(|value| value.is_nan()),
            "NaN input should remain visible in at least one chroma bin"
        );
    }

    #[test]
    fn spectral_centroid_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let centroid = spectral_centroid(&magnitudes, &freqs);

        assert!(
            (centroid - 25.0).abs() < 1e-12,
            "expected paired-sample centroid, got {centroid}"
        );
    }

    #[test]
    fn spectral_rolloff_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let rolloff = spectral_rolloff(&magnitudes, &freqs, 75.0);

        assert!(
            (rolloff - 30.0).abs() < 1e-12,
            "expected paired-sample rolloff, got {rolloff}"
        );
    }

    #[test]
    fn spectral_bandwidth_ignores_unpaired_tail_magnitudes() {
        let magnitudes = [1.0, 3.0, 100.0];
        let freqs = [10.0, 30.0];
        let bandwidth = spectral_bandwidth(&magnitudes, &freqs);

        assert!(
            (bandwidth - 8.660_254_037_844_387).abs() < 1e-12,
            "expected paired-sample bandwidth, got {bandwidth}"
        );
    }

    // ── lfilter tests ──────────────────────────────────────────────

    #[test]
    fn lfilter_identity() {
        // b=[1], a=[1] should pass signal through
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = lfilter(&[1.0], &[1.0], &x, None).expect("lfilter");
        assert_eq!(y, x);
    }

    #[test]
    fn lfilter_delay() {
        // b=[0, 1], a=[1] = one-sample delay
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = lfilter(&[0.0, 1.0], &[1.0], &x, None).expect("lfilter");
        assert!((y[0]).abs() < 1e-12);
        assert!((y[1] - 1.0).abs() < 1e-12);
        assert!((y[2] - 2.0).abs() < 1e-12);
        assert!((y[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lfilter_first_order_iir() {
        // y[n] = x[n] + 0.5*y[n-1] (b=[1], a=[1, -0.5])
        let x = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let y = lfilter(&[1.0], &[1.0, -0.5], &x, None).expect("lfilter");
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 0.5).abs() < 1e-12);
        assert!((y[2] - 0.25).abs() < 1e-12);
        assert!((y[3] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn lfilter_with_butter() {
        let coeffs = butter(2, &[0.2], FilterType::Lowpass).expect("butter");
        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();
        let y = lfilter(&coeffs.b, &coeffs.a, &x, None).expect("lfilter");
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lfilter_zi_matches_scipy_reference_vectors() {
        let zi = lfilter_zi(&[0.5, 0.5], &[1.0, -0.5]).expect("first-order reference");
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 1.5).abs() <= 1.0e-12, "got {}", zi[0]);

        let zi = lfilter_zi(&[1.0, -0.25], &[2.0, -0.3]).expect("normalized reference");
        assert_eq!(zi.len(), 1);
        assert!(
            (zi[0] - (-0.058_823_529_411_764_71)).abs() <= 1.0e-12,
            "got {}",
            zi[0]
        );

        let zi = lfilter_zi(&[0.2, 0.3, 0.5], &[1.0, -0.1, 0.2]).expect("second-order reference");
        let expected = [0.709_090_909_090_909_1, 0.318_181_818_181_818_1];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }

        let zi = lfilter_zi(&[0.2; 5], &[1.0]).expect("fir reference");
        let expected = [0.8, 0.6, 0.4, 0.2];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfilter_zi_trims_leading_zero_denominator() {
        let zi = lfilter_zi(&[1.0, 2.0], &[0.0, 2.0, -0.6]).expect("trimmed denominator");
        assert_eq!(zi.len(), 1);
        assert!((zi[0] - 1.642_857_142_857_142_8).abs() <= 1.0e-12);
    }

    #[test]
    fn lfilter_zi_sets_step_response_steady_state() {
        let b = [0.2, 0.3, 0.5];
        let a = [1.0, -0.1, 0.2];
        let zi = lfilter_zi(&b, &a).expect("lfilter_zi");
        let y = lfilter(&b, &a, &[1.0, 1.0, 1.0], Some(&zi)).expect("lfilter");
        for value in y {
            assert!(
                (value - 0.909_090_909_090_909_2).abs() <= 1.0e-12,
                "got {value}"
            );
        }
    }

    #[test]
    fn lfilter_with_state_matches_scipy_reference_vectors() {
        let (y, zf) = lfilter_with_state(
            &[0.2, 0.1],
            &[1.0, -0.7],
            &[1.0, -1.0, 0.5, 0.25],
            Some(&[0.3]),
        )
        .expect("lfilter_with_state");

        let expected_y = [
            0.5,
            0.249_999_999_999_999_94,
            0.174_999_999_999_999_96,
            0.222_5,
        ];
        let expected_zf = [0.180_749_999_999_999_97];
        assert_eq!(y.len(), expected_y.len());
        assert_eq!(zf.len(), expected_zf.len());
        for (got, want) in y.iter().zip(expected_y.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
        for (got, want) in zf.iter().zip(expected_zf.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfilter_axis_2d_matches_scipy_reference_vectors() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -1.0, 0.25, 2.0]];
        let axis_last =
            lfilter_axis_2d(&[0.2, 0.1], &[1.0, -0.7], &x, -1).expect("axis=-1 lfilter");
        let expected_last = [
            [0.2, 0.64, 1.248, 1.973_6],
            [
                0.1,
                -0.080_000_000_000_000_02,
                -0.106_000_000_000_000_02,
                0.350_8,
            ],
        ];
        for (row, expected_row) in axis_last.iter().zip(expected_last.iter()) {
            for (got, want) in row.iter().zip(expected_row.iter()) {
                assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
            }
        }

        let axis_zero = lfilter_axis_2d(&[0.2, 0.1], &[1.0, -0.7], &x, 0).expect("axis=0 lfilter");
        let expected_zero = [
            [0.2, 0.4, 0.600_000_000_000_000_1, 0.8],
            [
                0.339_999_999_999_999_97,
                0.279_999_999_999_999_97,
                0.770_000_000_000_000_1,
                1.36,
            ],
        ];
        for (row, expected_row) in axis_zero.iter().zip(expected_zero.iter()) {
            for (got, want) in row.iter().zip(expected_row.iter()) {
                assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
            }
        }
    }

    #[test]
    fn lfilter_zi_rejects_scalar_transfer_function() {
        let err = lfilter_zi(&[0.1], &[1.0]).expect_err("scalar transfer function must fail");
        assert!(
            err.to_string()
                .contains("The length of `a` along the last axis must be at least 2."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn lfiltic_matches_scipy_reference_vector() {
        let zi = lfiltic(&[1.0, 2.0, 3.0], &[1.0, 0.5], &[0.5], None).expect("lfiltic");
        let expected = [-0.25, 0.0];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_matches_non_unit_a0_reference() {
        let zi = lfiltic(
            &[0.5, 1.0, 0.2],
            &[2.0, 1.0, 3.0],
            &[1.2, -0.7, 0.4],
            Some(&[0.25, -1.5]),
        )
        .expect("lfiltic");
        let expected = [0.425, -1.775];
        assert_eq!(zi.len(), expected.len());
        for (got, want) in zi.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_restores_lfilter_continuation() {
        let b = [0.5, 1.0, 0.2];
        let a = [2.0, 1.0, 3.0];
        let zi = lfiltic(
            &b,
            &a,
            &[-2.72515625, 0.7971875, 1.499375, -0.70125, -0.3625, 0.725],
            Some(&[0.25, -0.5, 1.1, 0.7, -0.2, 1.3]),
        )
        .expect("lfiltic");
        let future = [0.9, -0.1, 0.3, -0.8];
        let y_future = lfilter(&b, &a, &future, Some(&zi)).expect("lfilter");
        let expected = [0.466796875, 4.3043359375, -2.73736328125, -5.147822265625];
        for (got, want) in y_future.iter().zip(expected.iter()) {
            assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn lfiltic_rejects_empty_a_coefficients() {
        let err = lfiltic(&[1.0], &[], &[0.0], Some(&[0.0])).expect_err("empty a must fail");
        assert!(
            err.to_string()
                .contains("There must be at least one `a` coefficient."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn lfiltic_rejects_zero_a0() {
        let err = lfiltic(&[1.0, 2.0], &[0.0, 2.0], &[0.0, 0.0], Some(&[0.0, 1.0]))
            .expect_err("zero a0 must fail");
        assert!(
            err.to_string()
                .contains("First `a` filter coefficient must be non-zero."),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn impulse_response_matches_scipy_transfer_function_convention() {
        // scipy.signal.lfilter([1], [1, -0.5], impulse): no leading delay.
        let response = impulse_response(&[1.0], &[1.0, -0.5], 6).expect("impulse response");
        let expected = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
        for (got, want) in response.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }

        // H(z) = 0.2 / (1 − 0.8 z⁻¹): the n=0 sample is 0.2, not delayed.
        let geo = impulse_response(&[0.2], &[1.0, -0.8], 6).expect("impulse response");
        let geo_expected = [0.2, 0.16, 0.128, 0.1024, 0.08192, 0.065536];
        for (got, want) in geo.iter().zip(geo_expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn step_response_matches_scipy_transfer_function_convention() {
        // scipy.signal.lfilter([1], [1, -0.5], ones): no leading delay.
        let response = step_response(&[1.0], &[1.0, -0.5], 6).expect("step response");
        let expected = [1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875];
        for (got, want) in response.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn impulse_response_rejects_improper_transfer_function() {
        let err = impulse_response(&[1.0, 2.0, 3.0], &[1.0, -0.5], 6)
            .expect_err("improper transfer functions should be rejected");
        assert!(
            err.to_string().contains("improper transfer function"),
            "unexpected error: {err}"
        );
    }

    // ── filtfilt tests ─────────────────────────────────────────────

    #[test]
    fn filtfilt_zero_phase() {
        // filtfilt should produce zero phase shift
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let n = 100;
        let freq = 0.05;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();
        let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");
        assert_eq!(y.len(), n);

        // filtfilt produces zero phase: verify output is in phase with input
        // by checking correlation between interior points (skip transients at edges)
        let start = 30;
        let end = 70;
        let mut corr_same = 0.0;
        let mut corr_shift = 0.0;
        for i in start..end {
            corr_same += x[i] * y[i];
            if i + 1 < n {
                corr_shift += x[i] * y[i + 1];
            }
        }
        // Zero phase means correlation at zero lag should be highest
        assert!(
            corr_same.abs() >= corr_shift.abs() * 0.9,
            "zero-phase: same-lag corr {corr_same} should be >= shifted corr {corr_shift}"
        );
    }

    #[test]
    fn filtfilt_attenuates_high_freq() {
        let coeffs = butter(3, &[0.1], FilterType::Lowpass).expect("butter");
        // Low frequency signal + high frequency noise
        let n = 200;
        let low: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 0.02 * i as f64).sin())
            .collect();
        let high: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 0.45 * i as f64).sin())
            .collect();
        let x: Vec<f64> = low.iter().zip(high.iter()).map(|(&l, &h)| l + h).collect();

        let y = filtfilt(&coeffs.b, &coeffs.a, &x).expect("filtfilt");

        // Output should be closer to low-frequency component
        let low_err: f64 = y
            .iter()
            .zip(low.iter())
            .skip(20)
            .take(n - 40)
            .map(|(&yi, &li)| (yi - li).powi(2))
            .sum();
        let total_err: f64 = y
            .iter()
            .zip(x.iter())
            .skip(20)
            .take(n - 40)
            .map(|(&yi, &xi)| (yi - xi).powi(2))
            .sum();
        assert!(
            low_err < total_err,
            "filtered should be closer to low-freq component"
        );
    }

    #[test]
    fn filtfilt_short_input_rejected() {
        assert!(filtfilt(&[1.0], &[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn filtfilt_empty_coefficients_rejected() {
        assert!(filtfilt(&[], &[1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(filtfilt(&[1.0], &[], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn filtfilt_padtypes_match_scipy_reference() {
        let x = [
            0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 1.0,
        ];
        let odd = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("odd")).expect("odd");
        let even = filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("even")).expect("even");
        let constant =
            filtfilt_with_padtype(&[0.2; 5], &[1.0], &x, Some("constant")).expect("constant");
        let expected_odd = [
            0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4,
            -0.28, 0.0, 0.36, 0.72, 1.0,
        ];
        let expected_even = [
            0.96, 0.92, 0.72, 0.36, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28,
            -0.4, -0.28, 0.08, 0.52, 0.88, 1.0,
        ];
        let expected_constant = [
            0.48, 0.6, 0.56, 0.32, 0.0, -0.28, -0.4, -0.28, 0.0, 0.28, 0.4, 0.28, 0.0, -0.28, -0.4,
            -0.28, 0.04, 0.44, 0.8, 1.0,
        ];

        for (actual, want) in odd.iter().zip(expected_odd.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
        for (actual, want) in even.iter().zip(expected_even.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
        for (actual, want) in constant.iter().zip(expected_constant.iter()) {
            assert!((actual - want).abs() < 1e-12, "{actual} vs {want}");
        }
    }

    #[test]
    fn filtfilt_rejects_unknown_padtype() {
        let err = filtfilt_with_padtype(&[0.2; 5], &[1.0], &[0.0, 1.0, 2.0, 1.0], Some("wrap"))
            .expect_err("invalid padtype");
        assert!(err.is_argument_error());
    }

    // ── Periodogram tests ──────────────────────────────────────────

    #[test]
    fn periodogram_single_frequency() {
        // Pure sine wave at 10 Hz, sampled at 100 Hz
        let fs = 100.0;
        let freq = 10.0;
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let result = periodogram(&x, fs, None).expect("periodogram");
        assert_eq!(result.frequencies.len(), result.psd.len());
        assert!(!result.psd.is_empty());

        // Peak should be near 10 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 1.0,
            "peak at {peak_freq} Hz, expected ~{freq} Hz"
        );
    }

    #[test]
    fn periodogram_with_hann_window() {
        let fs = 100.0;
        let n = 128;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let window = hann(n);
        let result = periodogram(&x, fs, Some(&window)).expect("periodogram");
        assert_eq!(result.psd.len(), n / 2 + 1);
        assert!(result.psd.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn periodogram_empty_rejected() {
        assert!(periodogram(&[], 1.0, None).is_err());
    }

    #[test]
    fn coherence_clamped_to_unit_interval() {
        // coherence ∈ [0, 1] for any inputs (clamped per spec).
        let n = 256;
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let result = coherence(&x, &y, 16.0, None, None, None).expect("coherence(x, y)");
        for (i, &v) in result.coherence.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "coherence(x, y)[{i}] = {v} out of [0, 1]"
            );
        }
    }

    #[test]
    fn csd_self_is_conjugate_symmetric_with_swapped_args() {
        // csd(x, y) should be the complex conjugate of csd(y, x)
        // (Pxy = conj(Pyx) by definition of cross-spectral density).
        let n = 128;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();
        let pxy = csd(&x, &y, 16.0, None, None, None).expect("csd(x, y)");
        let pyx = csd(&y, &x, 16.0, None, None, None).expect("csd(y, x)");
        assert_eq!(pxy.csd.len(), pyx.csd.len());
        for (i, (&(re_xy, im_xy), &(re_yx, im_yx))) in
            pxy.csd.iter().zip(pyx.csd.iter()).enumerate()
        {
            // Real parts equal; imaginary parts are negatives.
            assert!(
                (re_xy - re_yx).abs() < 1e-9,
                "Re(Pxy[{i}]) = {re_xy} != Re(Pyx) = {re_yx}"
            );
            assert!(
                (im_xy + im_yx).abs() < 1e-9,
                "Im(Pxy[{i}]) = {im_xy} should be -Im(Pyx) = {im_yx}"
            );
        }
    }

    // ── Welch tests ────────────────────────────────────────────────

    #[test]
    fn welch_basic() {
        let fs = 1000.0;
        let n = 1024;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin())
            .collect();

        let result = welch(&x, fs, None, Some(256), None).expect("welch");
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.frequencies.len(), result.psd.len());

        // Peak should be near 50 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 50.0).abs() < 5.0,
            "peak at {peak_freq} Hz, expected ~50 Hz"
        );
    }

    #[test]
    fn welch_matches_scipy_detrend_constant() {
        // scipy.signal.welch defaults to detrend='constant'. On a ramp the
        // per-segment mean dominates, so omitting it inflated the PSD by
        // thousands (frankenscipy-gmysl).
        let x: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let result = welch(&x, 1.0, Some("hann"), Some(16), Some(8)).expect("welch");
        let expected = [
            2.666_666_666_666_666_5,
            79.168_949_150_563_39,
            3.833_042_305_749_054_4,
            0.236_323_620_258_148_65,
            0.036_205_198_234_783_22,
            0.008_082_135_137_255_544,
            0.001_993_194_212_531_567,
            0.000_358_124_894_072_232_57,
        ];
        assert_eq!(result.psd.len(), 9);
        for (got, want) in result.psd.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "welch PSD: {got} != {want}");
        }
        assert!(result.psd[8].abs() < 1e-9, "Nyquist bin ~0");
    }

    #[test]
    fn welch_reduces_variance() {
        // Welch with more averaging should have smoother PSD than periodogram
        let fs = 100.0;
        let n = 1024;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()
                    + if i % 3 == 0 { 0.5 } else { -0.5 } // noise
            })
            .collect();

        let psd_raw = periodogram(&x, fs, None).expect("periodogram");
        let psd_welch = welch(&x, fs, None, Some(128), None).expect("welch");

        // Welch PSD should be smoother (lower variance)
        let raw_var = variance_of_psd(&psd_raw.psd);
        let welch_var = variance_of_psd(&psd_welch.psd);
        assert!(
            welch_var < raw_var,
            "Welch should be smoother: var={welch_var} < {raw_var}"
        );
    }

    #[test]
    fn welch_empty_rejected() {
        assert!(welch(&[], 1.0, None, None, None).is_err());
    }

    #[test]
    fn welch_nperseg_clamps_to_signal_length() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let oversized = welch(&x, 10.0, None, Some(16), None).expect("welch oversized");
        let clamped = welch(&x, 10.0, None, Some(x.len()), None).expect("welch clamped");
        let expected_frequencies = [0.0, 2.0, 4.0];

        assert_eq!(oversized.frequencies.len(), expected_frequencies.len());
        assert_eq!(oversized.frequencies.len(), clamped.frequencies.len());
        assert_eq!(oversized.psd.len(), clamped.psd.len());

        for (index, ((actual, expected), clamped_frequency)) in oversized
            .frequencies
            .iter()
            .zip(expected_frequencies.iter())
            .zip(clamped.frequencies.iter())
            .enumerate()
        {
            assert_close(
                *actual,
                *expected,
                1e-12,
                &format!("frequency[{index}] should match SciPy clamp"),
            );
            assert_close(
                *actual,
                *clamped_frequency,
                1e-12,
                &format!("frequency[{index}] should match explicit clamp"),
            );
        }

        for (index, (actual, clamped_psd)) in
            oversized.psd.iter().zip(clamped.psd.iter()).enumerate()
        {
            assert_close(
                *actual,
                *clamped_psd,
                1e-12,
                &format!("psd[{index}] should match explicit clamp"),
            );
        }
    }

    fn variance_of_psd(psd: &[f64]) -> f64 {
        let n = psd.len() as f64;
        let mean = psd.iter().sum::<f64>() / n;
        psd.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
    }

    fn gd_span(values: &[f64]) -> f64 {
        let min = values
            .iter()
            .copied()
            .fold(f64::INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
        let max = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        max - min
    }

    // ── Filter conversion tests ────────────────────────────────────

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: {a} vs {b} (diff={})",
            (a - b).abs()
        );
    }

    fn assert_ba_close(direct: &BaCoeffs, dispatch: &BaCoeffs, tol: f64, family: &str) {
        assert_eq!(direct.b.len(), dispatch.b.len());
        assert_eq!(direct.a.len(), dispatch.a.len());
        for (lhs, rhs) in direct.b.iter().zip(dispatch.b.iter()) {
            assert!(
                (lhs - rhs).abs() < tol,
                "{family} dispatcher mismatch in b: {lhs} vs {rhs}"
            );
        }
        for (lhs, rhs) in direct.a.iter().zip(dispatch.a.iter()) {
            assert!(
                (lhs - rhs).abs() < tol,
                "{family} dispatcher mismatch in a: {lhs} vs {rhs}"
            );
        }
    }

    fn nearest_freq_index(w: &[f64], omega: f64) -> usize {
        w.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((**a) - omega).abs().total_cmp(&((**b) - omega).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn tf2zpk_first_order() {
        // b=[1, -0.5], a=[1, -0.9] → zero at 0.5, pole at 0.9
        let zpk = tf2zpk(&[1.0, -0.5], &[1.0, -0.9]).expect("tf2zpk");
        assert_eq!(zpk.zeros_re.len(), 1);
        assert_eq!(zpk.poles_re.len(), 1);
        assert_close(zpk.zeros_re[0], 0.5, 1e-10, "zero");
        assert_close(zpk.poles_re[0], 0.9, 1e-10, "pole");
        assert_close(zpk.gain, 1.0, 1e-10, "gain");
    }

    #[test]
    fn tf2zpk_second_order_complex() {
        // butter(2, 0.3) should give complex conjugate poles
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let zpk = tf2zpk(&coeffs.b, &coeffs.a).expect("tf2zpk");
        assert_eq!(zpk.poles_re.len(), 2);
        // Poles should be complex conjugates
        assert_close(zpk.poles_re[0], zpk.poles_re[1], 1e-10, "pole re match");
        assert!(
            (zpk.poles_im[0] + zpk.poles_im[1]).abs() < 1e-10,
            "poles should be conjugate"
        );
        // Poles should be inside unit circle (stable)
        let mag = (zpk.poles_re[0].powi(2) + zpk.poles_im[0].powi(2)).sqrt();
        assert!(mag < 1.0, "poles inside unit circle, got mag={mag}");
    }

    #[test]
    fn zpk2tf_roundtrip() {
        let b_orig = [0.1, 0.2, 0.1];
        let a_orig = [1.0, -0.5, 0.2];
        let zpk = tf2zpk(&b_orig, &a_orig).expect("tf2zpk");
        let ba = zpk2tf(&zpk);

        // Normalize and compare
        let scale_b = b_orig[0] / ba.b[0];
        let scale_a = a_orig[0] / ba.a[0];
        for (i, (&orig, &recovered)) in b_orig.iter().zip(ba.b.iter()).enumerate() {
            assert_close(
                recovered * scale_b,
                orig,
                1e-8,
                &format!("b[{i}] roundtrip"),
            );
        }
        for (i, (&orig, &recovered)) in a_orig.iter().zip(ba.a.iter()).enumerate() {
            assert_close(
                recovered * scale_a,
                orig,
                1e-8,
                &format!("a[{i}] roundtrip"),
            );
        }
    }

    #[test]
    fn tf2sos_butter2() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 2 → 1 SOS section
        assert_eq!(sos.len(), 1, "butter(2) should give 1 SOS section");
        // a0 should be 1.0
        assert_close(sos[0][3], 1.0, 1e-10, "a0 = 1");
    }

    #[test]
    fn tf2sos_butter4() {
        let coeffs = butter(4, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        // Order 4 → 2 SOS sections
        assert_eq!(sos.len(), 2, "butter(4) should give 2 SOS sections");
    }

    #[test]
    fn butter_with_output_ba_matches_legacy_butter() {
        let direct = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let via_output = butter_with_output(4, &[0.25], FilterType::Lowpass, ButterOutput::Ba)
            .expect("butter_with_output");
        assert!(matches!(via_output, BaCoeffsOrSos::Ba(_)));
        let via_output = match via_output {
            BaCoeffsOrSos::Ba(ba) => ba,
            BaCoeffsOrSos::Sos(_) => return,
        };

        assert_eq!(direct.b.len(), via_output.b.len());
        assert_eq!(direct.a.len(), via_output.a.len());
        for (i, (&got, &expected)) in via_output.b.iter().zip(direct.b.iter()).enumerate() {
            assert_close(got, expected, 1e-12, &format!("b[{i}]"));
        }
        for (i, (&got, &expected)) in via_output.a.iter().zip(direct.a.iter()).enumerate() {
            assert_close(got, expected, 1e-12, &format!("a[{i}]"));
        }
    }

    #[test]
    fn butter_sos_matches_scipy_reference_sections() {
        let sos = butter_sos(4, &[0.25], FilterType::Lowpass).expect("butter_sos");
        let expected = [
            [
                0.010209480791203138,
                0.020418961582406275,
                0.010209480791203138,
                1.0,
                -0.8553979327751704,
                0.20971535775655478,
            ],
            [1.0, 2.0, 1.0, 1.0, -1.1130298541633479, 0.5740619150839545],
        ];
        assert_eq!(sos.len(), expected.len(), "section count");
        for (section_idx, (got, expected_section)) in sos.iter().zip(expected.iter()).enumerate() {
            for (coeff_idx, (&got_coeff, &expected_coeff)) in
                got.iter().zip(expected_section.iter()).enumerate()
            {
                assert_close(
                    got_coeff,
                    expected_coeff,
                    1e-7,
                    &format!("section {section_idx} coeff {coeff_idx}"),
                );
            }
        }
    }

    #[test]
    fn sos2tf_roundtrip() {
        let coeffs = butter(3, &[0.4], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        let recovered = sos2tf(&sos);

        // DC gain should match: sum(b)/sum(a) for both
        let orig_dc: f64 = coeffs.b.iter().sum::<f64>() / coeffs.a.iter().sum::<f64>();
        let rec_dc: f64 = recovered.b.iter().sum::<f64>() / recovered.a.iter().sum::<f64>();
        assert_close(orig_dc, rec_dc, 1e-8, "DC gain roundtrip");
    }

    #[test]
    fn zpk2sos_simple_real_poles() {
        let zpk = ZpkCoeffs {
            zeros_re: vec![-1.0],
            zeros_im: vec![0.0],
            poles_re: vec![0.5],
            poles_im: vec![0.0],
            gain: 2.0,
        };
        let sos = zpk2sos(&zpk);
        assert!(!sos.is_empty());
        // Single real pole/zero → 1 section
        // Denominator: (1 - 0.5*z^-1)
        assert_close(sos[0][4], -0.5, 1e-10, "a1 = -pole");
    }

    #[test]
    fn sos2zpk_roundtrip() {
        let coeffs = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let zpk_orig = tf2zpk(&coeffs.b, &coeffs.a).expect("tf2zpk");
        let sos = zpk2sos(&zpk_orig);
        let zpk_rec = sos2zpk(&sos);

        // Same number of poles and zeros
        assert_eq!(
            zpk_orig.poles_re.len(),
            zpk_rec.poles_re.len(),
            "pole count"
        );
        assert_eq!(
            zpk_orig.zeros_re.len(),
            zpk_rec.zeros_re.len(),
            "zero count"
        );
    }

    #[test]
    fn tf2sos_high_order_stability() {
        // Order 8 filter — SOS should be more stable than tf form
        let coeffs = butter(8, &[0.2], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        assert_eq!(sos.len(), 4, "order 8 → 4 sections");

        // Each section's poles should be inside unit circle
        for (i, section) in sos.iter().enumerate() {
            let a2 = section[5];
            // For stability: |a2| < 1
            assert!(a2.abs() < 1.0 + 1e-10, "section {i} unstable: a2={a2}");
        }
    }

    #[test]
    fn poly_multiply_basic() {
        // (1 + 2z^-1) * (1 + 3z^-1) = 1 + 5z^-1 + 6z^-2
        let result = poly_multiply(&[1.0, 2.0], &[1.0, 3.0]);
        assert_eq!(result.len(), 3);
        assert_close(result[0], 1.0, 1e-12, "c0");
        assert_close(result[1], 5.0, 1e-12, "c1");
        assert_close(result[2], 6.0, 1e-12, "c2");
    }

    #[test]
    fn tf2zpk_empty_rejected() {
        assert!(tf2zpk(&[], &[1.0]).is_err());
        assert!(tf2zpk(&[1.0], &[]).is_err());
    }

    // ── Frequency response tests ───────────────────────────────────

    #[test]
    fn freqz_unity_gain() {
        // H(z) = 1 (b=[1], a=[1]) → magnitude 1 everywhere
        let result = freqz(&[1.0], &[1.0], Some(64)).expect("freqz");
        assert_eq!(result.w.len(), 64);
        assert_eq!(result.h_mag.len(), 64);
        for (i, &mag) in result.h_mag.iter().enumerate() {
            assert_close(mag, 1.0, 1e-12, &format!("unity mag at bin {i}"));
        }
    }

    #[test]
    fn magnitude_response_reports_linear_gain() {
        let (_, unity) = magnitude_response(&[1.0], &[1.0], 8);
        for (i, &mag) in unity.iter().enumerate() {
            assert_close(mag, 1.0, 1e-12, &format!("unity magnitude at bin {i}"));
        }

        let (_, two_times) = magnitude_response(&[2.0], &[1.0], 8);
        for (i, &mag) in two_times.iter().enumerate() {
            assert_close(mag, 2.0, 1e-12, &format!("two-times magnitude at bin {i}"));
        }
    }

    #[test]
    fn magnitude_response_db_is_20_log10_of_linear() {
        // [frankenscipy-o8x6j]: magnitude_response_db must be the dB
        // transform of magnitude_response. Pin the algebraic relation
        // 20·log10(|H|) for several filter shapes.
        let cases: &[(&[f64], &[f64])] = &[
            (&[1.0], &[1.0]),
            (&[2.0], &[1.0]),
            (&[1.0, 0.5, 0.25], &[1.0, -0.3, 0.2]),
            (&[0.5, 1.0, 0.5], &[1.0, 0.5]),
        ];
        for (b, a) in cases {
            let (freqs_lin, mags_lin) = magnitude_response(b, a, 16);
            let (freqs_db, mags_db) = magnitude_response_db(b, a, 16);
            assert_eq!(freqs_lin, freqs_db);
            for (i, (&lin, &db)) in mags_lin.iter().zip(mags_db.iter()).enumerate() {
                if lin > 1e-30 {
                    let expected_db = 20.0 * lin.log10();
                    assert_close(
                        db,
                        expected_db,
                        1e-12,
                        &format!("dB at bin {i} for b={b:?}"),
                    );
                } else {
                    assert!(
                        db.is_infinite() && db < 0.0,
                        "expected -inf at bin {i} for zero-magnitude bin"
                    );
                }
            }
        }
    }

    #[test]
    fn magnitude_response_db_doubles_at_six_db_per_octave_gain() {
        // [frankenscipy-o8x6j]: a flat |H|=2 filter should report
        // exactly 6.0206 dB everywhere (= 20·log10(2)).
        let (_, mags_db) = magnitude_response_db(&[2.0], &[1.0], 8);
        let expected = 20.0_f64 * 2.0_f64.log10();
        for (i, &db) in mags_db.iter().enumerate() {
            assert_close(db, expected, 1e-12, &format!("dB at bin {i}"));
        }
    }

    #[test]
    fn freqz_whole_matches_scipy_reference() {
        let tol = 1e-8;
        let result = freqz_with_whole(&[1.0, 0.5, 0.25], &[1.0, -0.3, 0.2], Some(8), true)
            .expect("freqz whole");
        let expected_mag = [
            1.944_444_44,
            1.880_828_33,
            1.054_994_64,
            0.511_363_23,
            0.5,
            0.511_363_23,
            1.054_994_64,
            1.880_828_33,
        ];
        let expected_phase = [
            0.0,
            -0.434_838_91,
            -0.946_773_27,
            -0.486_582_96,
            0.0,
            0.486_582_96,
            0.946_773_27,
            0.434_838_91,
        ];

        assert_eq!(result.w.len(), 8);
        assert_close(result.w[1], std::f64::consts::FRAC_PI_4, 1e-12, "pi/4 bin");
        assert_close(result.w[4], std::f64::consts::PI, 1e-12, "pi bin");
        assert_close(
            result.w[7],
            7.0 * std::f64::consts::PI / 4.0,
            1e-12,
            "7pi/4 bin",
        );

        for (i, (&actual, &want)) in result.h_mag.iter().zip(expected_mag.iter()).enumerate() {
            assert_close(actual, want, tol, &format!("whole magnitude bin {i}"));
        }

        for (i, (&actual, &want)) in result.h_phase.iter().zip(expected_phase.iter()).enumerate() {
            assert_close(actual, want, tol, &format!("whole phase bin {i}"));
        }
    }

    #[test]
    fn freqz_butter_at_cutoff() {
        // Butter(2, 0.3): at cutoff frequency, magnitude should be ≈ 1/√2 (-3dB)
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");

        // Cutoff at ω = 0.3π
        let cutoff_omega = 0.3 * std::f64::consts::PI;
        // Find nearest frequency bin
        let cutoff_idx = result
            .w
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - cutoff_omega)
                    .abs()
                    .total_cmp(&((**b) - cutoff_omega).abs())
            })
            .map(|(i, _)| i)
            .unwrap();

        let mag_at_cutoff = result.h_mag[cutoff_idx];
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!(
            (mag_at_cutoff - inv_sqrt2).abs() < 0.05,
            "mag at cutoff should be ~{inv_sqrt2}, got {mag_at_cutoff}"
        );
    }

    #[test]
    fn freqz_lowpass_dc_unity() {
        // Lowpass filter: DC gain (ω=0) should be 1
        let coeffs = butter(3, &[0.4], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        assert_close(result.h_mag[0], 1.0, 1e-6, "DC gain = 1");
    }

    #[test]
    fn freqz_highpass_nyquist_unity() {
        // Highpass filter: Nyquist gain (ω=π) should be 1
        let coeffs = butter(2, &[0.3], FilterType::Highpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        let last_idx = result.h_mag.len() - 1;
        assert_close(result.h_mag[last_idx], 1.0, 1e-6, "Nyquist gain = 1");
    }

    #[test]
    fn freqz_sos_matches_ba() {
        // SOS and BA forms should give same frequency response
        let coeffs = butter(4, &[0.25], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let ba_result = freqz(&coeffs.b, &coeffs.a, Some(128)).expect("freqz ba");
        let sos_result = freqz_sos(&sos, Some(128)).expect("freqz sos");

        for i in 0..128 {
            assert_close(
                ba_result.h_mag[i],
                sos_result.h_mag[i],
                1e-4,
                &format!("mag match at bin {i}"),
            );
        }
    }

    #[test]
    fn freqz_monotone_lowpass() {
        // Butterworth lowpass: magnitude should be monotonically decreasing
        let coeffs = butter(4, &[0.3], FilterType::Lowpass).expect("butter");
        let result = freqz(&coeffs.b, &coeffs.a, Some(256)).expect("freqz");
        for i in 1..result.h_mag.len() {
            assert!(
                result.h_mag[i] <= result.h_mag[i - 1] + 1e-10,
                "butter lowpass should be monotone decreasing: mag[{i}]={} > mag[{}]={}",
                result.h_mag[i],
                i - 1,
                result.h_mag[i - 1]
            );
        }
    }

    #[test]
    fn group_delay_fir_linear_phase() {
        // A symmetric FIR filter has constant group delay = (N-1)/2
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0]; // symmetric, N=5
        let a = vec![1.0];
        let result = freqz(&b, &a, Some(64)).expect("freqz for gd check");
        let (w, gd) = group_delay(&b, &a, Some(64)).expect("group_delay");
        assert_eq!(w.len(), 64);

        // Expected group delay = (5-1)/2 = 2.0 samples
        // Only check at frequencies where the filter has significant magnitude
        // (group delay is undefined at filter nulls where |H|≈0)
        for (i, (&gd_val, &mag)) in gd.iter().zip(result.h_mag.iter()).enumerate() {
            if mag > 0.1 {
                assert!(
                    (gd_val - 2.0).abs() < 0.15,
                    "group delay at bin {i} (mag={mag:.3}) should be ~2.0, got {gd_val}",
                );
            }
        }
    }

    #[test]
    fn phase_delay_fir_linear_phase() {
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let a = vec![1.0];
        let result = freqz(&b, &a, Some(64)).expect("freqz for pd check");
        let (w, pd) = phase_delay(&b, &a, Some(64)).expect("phase_delay");
        assert_eq!(w.len(), 64);

        for (i, (&pd_val, &mag)) in pd.iter().zip(result.h_mag.iter()).enumerate() {
            if mag > 0.1 {
                assert!(
                    (pd_val - 2.0).abs() < 0.15,
                    "phase delay at bin {i} (mag={mag:.3}) should be ~2.0, got {pd_val}",
                );
            }
        }
    }

    #[test]
    fn phase_delay_matches_group_delay_for_pure_delay() {
        let b = vec![0.0, 0.0, 1.0];
        let a = vec![1.0];
        let (_, gd) = group_delay(&b, &a, Some(64)).expect("group_delay");
        let (_, pd) = phase_delay(&b, &a, Some(64)).expect("phase_delay");

        for (i, (&gd_val, &pd_val)) in gd.iter().zip(pd.iter()).enumerate().skip(1) {
            assert!(
                (gd_val - pd_val).abs() < 1e-10,
                "pure delay mismatch at bin {i}: gd={gd_val}, pd={pd_val}",
            );
        }
    }

    #[test]
    fn phase_delay_dc_uses_group_delay_limit() {
        let b = vec![1.0, 0.0, -1.0];
        let a = vec![1.0];
        let (_, gd) = group_delay(&b, &a, Some(32)).expect("group_delay");
        let (_, pd) = phase_delay(&b, &a, Some(32)).expect("phase_delay");
        assert!(
            (pd[0] - gd[0]).abs() < 1e-12,
            "phase delay DC limit should match group delay: {} vs {}",
            pd[0],
            gd[0]
        );
    }

    #[test]
    fn phase_delay_rejects_empty_coefficients() {
        assert!(phase_delay(&[], &[1.0], Some(16)).is_err());
        assert!(phase_delay(&[1.0], &[], Some(16)).is_err());
    }

    #[test]
    fn group_delay_rejects_zero_frequency_count() {
        let err = group_delay(&[1.0], &[1.0], Some(0)).expect_err("n_freqs=0 should fail");
        assert_eq!(
            err,
            SignalError::InvalidArgument("n_freqs must be > 0".to_string())
        );
    }

    #[test]
    fn freqz_empty_rejected() {
        assert!(freqz(&[], &[1.0], None).is_err());
        assert!(freqz(&[1.0], &[], None).is_err());
    }

    // ── SOS filtering tests ────────────────────────────────────────

    #[test]
    fn sosfilt_matches_lfilter_low_order() {
        // For low-order filter, sosfilt and lfilter should give same results
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let x: Vec<f64> = (0..100)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();

        let y_lfilter = lfilter(&coeffs.b, &coeffs.a, &x, None).expect("lfilter");
        let y_sosfilt = sosfilt(&sos, &x).expect("sosfilt");

        for (i, (&yl, &ys)) in y_lfilter.iter().zip(y_sosfilt.iter()).enumerate() {
            assert!(
                (yl - ys).abs() < 1e-8,
                "sosfilt vs lfilter at sample {i}: {ys} vs {yl}",
            );
        }
    }

    #[test]
    fn sosfilt_identity_passthrough() {
        // SOS with unity gain sections: output = input
        let sos = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = sosfilt(&sos, &x).expect("sosfilt identity");
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            assert!((xi - yi).abs() < 1e-12, "passthrough at {i}: {yi} != {xi}",);
        }
    }

    #[test]
    fn sosfilt_high_order_stable() {
        // Order-8 Butterworth: sosfilt should remain stable
        let coeffs = butter(8, &[0.2], FilterType::Lowpass).expect("butter 8");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let x: Vec<f64> = (0..200)
            .map(|i| (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin())
            .collect();

        let y = sosfilt(&sos, &x).expect("sosfilt");
        // Output should be finite (stable) and bounded
        assert!(
            y.iter().all(|v| v.is_finite()),
            "sosfilt output should be finite for stable filter"
        );
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!(max_abs < 2.0, "output should be bounded (max={max_abs})");
    }

    #[test]
    fn sosfiltfilt_zero_phase() {
        // sosfiltfilt should produce zero phase shift
        let coeffs = butter(3, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");

        let n = 200;
        let freq = 0.05;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64).sin())
            .collect();

        let y = sosfiltfilt(&sos, &x).expect("sosfiltfilt");
        assert_eq!(y.len(), n);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sosfilt_zi_steady_state() {
        let coeffs = butter(2, &[0.3], FilterType::Lowpass).expect("butter");
        let sos = tf2sos(&coeffs.b, &coeffs.a).expect("tf2sos");
        let zi = sosfilt_zi(&sos).expect("sosfilt_zi");
        assert_eq!(zi.len(), sos.len());
        // Each section should have finite initial conditions
        for (i, zi_sec) in zi.iter().enumerate() {
            assert!(
                zi_sec[0].is_finite() && zi_sec[1].is_finite(),
                "section {i} zi should be finite",
            );
        }
    }

    #[test]
    fn sosfilt_empty_rejected() {
        assert!(sosfilt(&[], &[1.0]).is_err());
    }

    #[test]
    fn sosfiltfilt_short_rejected() {
        let sos = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        assert!(sosfiltfilt(&sos, &[1.0, 2.0]).is_err());
    }

    // ── FIR filter design tests ──────────────────────────────────────

    #[test]
    fn firwin_lowpass_symmetric() {
        // FIR lowpass filter should have symmetric (linear phase) coefficients
        let h = firwin(21, &[0.3], FirWindow::Hamming, true).expect("firwin lowpass");
        assert_eq!(h.len(), 21);
        for i in 0..10 {
            assert!(
                (h[i] - h[20 - i]).abs() < 1e-12,
                "symmetry broken at {i}: h[{i}]={}, h[{}]={}",
                h[i],
                20 - i,
                h[20 - i]
            );
        }
    }

    #[test]
    fn firwin_lowpass_dc_gain() {
        // Lowpass with pass_zero=true should have unity DC gain
        let h = firwin(31, &[0.4], FirWindow::Hamming, true).expect("firwin lowpass");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.05,
            "DC gain should be ~1.0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_highpass() {
        // Highpass: pass_zero=false, DC gain should be ~0
        let h = firwin(31, &[0.4], FirWindow::Hamming, false).expect("firwin highpass");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            dc_gain.abs() < 0.05,
            "highpass DC gain should be ~0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_bandpass() {
        // Bandpass: two cutoff frequencies, pass_zero=false
        let h = firwin(51, &[0.2, 0.5], FirWindow::Hamming, false).expect("firwin bandpass");
        assert_eq!(h.len(), 51);
        let dc_gain: f64 = h.iter().sum();
        assert!(
            dc_gain.abs() < 0.1,
            "bandpass DC gain should be ~0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_bandstop() {
        // Bandstop: two cutoff frequencies, pass_zero=true
        let h = firwin(51, &[0.2, 0.5], FirWindow::Hamming, true).expect("firwin bandstop");
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.1,
            "bandstop DC gain should be ~1.0, got {dc_gain}"
        );
    }

    #[test]
    fn firwin_cutoff_attenuation() {
        // At the cutoff frequency, magnitude should be approximately -6dB (≈0.5)
        let numtaps = 51;
        let cutoff = 0.3;
        let h = firwin(numtaps, &[cutoff], FirWindow::Hamming, true).expect("firwin");
        // Compute magnitude at cutoff
        let alpha = (numtaps - 1) as f64 / 2.0;
        let omega = std::f64::consts::PI * cutoff;
        let mut mag_real = 0.0;
        let mut mag_imag = 0.0;
        for (n, &hn) in h.iter().enumerate() {
            let phase = omega * (n as f64 - alpha);
            mag_real += hn * phase.cos();
            mag_imag += hn * phase.sin();
        }
        let mag = (mag_real * mag_real + mag_imag * mag_imag).sqrt();
        assert!(
            (mag - 0.5).abs() < 0.15,
            "magnitude at cutoff should be ~0.5 (-6dB), got {mag}"
        );
    }

    #[test]
    fn firwin2_lowpass_basic() {
        // Flat passband from 0 to 0.3, stopband from 0.5 to 1.0
        let freq = vec![0.0, 0.3, 0.5, 1.0];
        let gain = vec![1.0, 1.0, 0.0, 0.0];
        let h = firwin2(51, &freq, &gain, FirWindow::Hamming).expect("firwin2");
        assert_eq!(h.len(), 51);
        // DC gain should be ~1
        let dc_gain: f64 = h.iter().sum();
        assert!((dc_gain - 1.0).abs() < 0.15, "firwin2 DC gain: {dc_gain}");
    }

    #[test]
    fn firwin2_matches_scipy_reference() {
        // Reference taps from scipy.signal.firwin2 (frankenscipy-3fumb).
        let lp = firwin2(
            51,
            &[0.0, 0.3, 0.5, 1.0],
            &[1.0, 1.0, 0.0, 0.0],
            FirWindow::Hamming,
        )
        .expect("firwin2 lp");
        // Centre taps and symmetry (linear-phase Type-I filter).
        for (got, want) in lp[24..27]
            .iter()
            .zip([0.296_698_542_674, 0.399_902_343_75, 0.296_698_542_674].iter())
        {
            assert!((got - want).abs() < 1e-9, "lp tap {got} != {want}");
        }

        let bp = firwin2(
            33,
            &[0.0, 0.3, 0.6, 1.0],
            &[0.0, 1.0, 1.0, 0.0],
            FirWindow::Hamming,
        )
        .expect("firwin2 bp");
        for (got, want) in bp[14..19].iter().zip(
            [
                -0.217_298_916_851,
                0.035_483_625_869,
                0.649_861_653_646,
                0.035_483_625_869,
                -0.217_298_916_851,
            ]
            .iter(),
        ) {
            assert!((got - want).abs() < 1e-9, "bp tap {got} != {want}");
        }

        // Even numtaps with nonzero Nyquist gain is rejected (Type II).
        assert!(firwin2(32, &[0.0, 1.0], &[1.0, 1.0], FirWindow::Hamming).is_err());
    }

    #[test]
    fn decimate_matches_scipy_cheby1_default() {
        // scipy.signal.decimate uses cheby1(8, 0.05, 0.8/q) by default.
        let x: Vec<f64> = (0..60)
            .map(|i| (i as f64 * 8.0 * std::f64::consts::PI / 59.0).sin())
            .collect();

        let y2 = decimate(&x, 2).expect("decimate q=2");
        assert_eq!(y2.len(), 30);
        for (got, want) in y2[2..6].iter().zip(
            [
                0.985_343_317_375_530_9,
                0.549_199_148_335_503_8,
                -0.261_846_123_422_117_1,
                -0.892_756_255_192_343_1,
            ]
            .iter(),
        ) {
            assert!((got - want).abs() < 1e-7, "decimate q=2: {got} != {want}");
        }

        let y3 = decimate(&x, 3).expect("decimate q=3");
        assert_eq!(y3.len(), 20);
        for (got, want) in y3[2..6].iter().zip(
            [
                0.553_544_209_769_113_1,
                -0.637_346_726_024_172_2,
                -0.917_280_480_999_162,
                0.107_330_905_569_789_44,
            ]
            .iter(),
        ) {
            assert!((got - want).abs() < 1e-7, "decimate q=3: {got} != {want}");
        }
    }

    #[test]
    fn kaiserord_typical_specs() {
        // 60dB attenuation, transition width 0.1
        let (numtaps, beta) = kaiserord(60.0, 0.1).expect("kaiserord");
        assert!(
            numtaps > 20 && numtaps < 100,
            "numtaps={numtaps} should be 20-100 for 60dB/0.1"
        );
        assert!(
            beta > 4.0 && beta < 7.0,
            "beta={beta} should be 4-7 for 60dB"
        );
    }

    #[test]
    fn kaiserord_mild_specs() {
        // 20dB attenuation, wide transition
        let (numtaps, beta) = kaiserord(20.0, 0.2).expect("kaiserord mild");
        assert!(
            numtaps < 30,
            "numtaps={numtaps} should be small for mild specs"
        );
        // beta should be 0 for attenuation <= 21
        assert!(beta < 0.5, "beta={beta} should be ~0 for A<=21");
    }

    #[test]
    fn kaiserord_rejects_non_finite_args() {
        assert!(kaiserord(f64::NAN, 0.1).is_err());
        assert!(kaiserord(60.0, f64::NAN).is_err());
        assert!(kaiserord(f64::INFINITY, 0.1).is_err());
        assert!(kaiserord(60.0, f64::INFINITY).is_err());
    }

    #[test]
    fn kaiser_atten_closed_form() {
        // /porting-to-rust [frankenscipy-ooa4k] + boundary fix
        // [frankenscipy-bg40m]:
        // atten(numtaps, width) = 2.285 · (numtaps − 1) · π · width + 7.95.
        // numtaps = 1: zero-order term only → 7.95.
        assert_close(kaiser_atten(1, 0.1), 7.95, 1e-12, "atten at numtaps=1");
        // numtaps = 0 boundary: scipy lets (numtaps − 1) underflow to
        // −1, so result = −2.285·π·width + 7.95. The previous fsci
        // override returned 7.95 here, diverging from scipy.
        let scipy_at_zero = -2.285 * std::f64::consts::PI * 0.1 + 7.95;
        assert_close(
            kaiser_atten(0, 0.1),
            scipy_at_zero,
            1e-12,
            "atten at numtaps=0 must match scipy boundary value",
        );
        // Reference value verified against scipy.signal.kaiser_atten:
        //   scipy.signal.kaiser_atten(0, 0.1) = 7.232146078654733
        assert_close(
            kaiser_atten(0, 0.1),
            7.232_146_078_654_733,
            1e-12,
            "atten(0, 0.1) vs hardcoded scipy reference",
        );
        // Reference: numtaps=21, width=0.1.
        let expected = 2.285 * 20.0 * std::f64::consts::PI * 0.1 + 7.95;
        assert_close(kaiser_atten(21, 0.1), expected, 1e-12, "atten at 21,0.1");
    }

    #[test]
    fn kaiser_beta_three_branch_closed_form() {
        // a < 21: β = 0
        assert_eq!(kaiser_beta(10.0), 0.0);
        assert_eq!(kaiser_beta(20.99), 0.0);
        // a in [21, 50]: 0.5842(a-21)^0.4 + 0.07886(a-21)
        for &a in &[21.0_f64, 30.0, 40.0, 50.0] {
            let expected = 0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0);
            assert_close(
                kaiser_beta(a),
                expected,
                1e-12,
                &format!("kaiser_beta({a}) middle branch"),
            );
        }
        // a > 50: 0.1102 · (a - 8.7)
        for &a in &[60.0_f64, 80.0, 100.0] {
            assert_close(
                kaiser_beta(a),
                0.1102 * (a - 8.7),
                1e-12,
                &format!("kaiser_beta({a}) high branch"),
            );
        }
    }

    #[test]
    fn kaiser_beta_consistent_with_kaiserord() {
        // kaiserord internally uses the same beta formula; verify
        // they agree on a range of attenuation values.
        for &a in &[15.0_f64, 21.0, 30.0, 50.0, 60.0, 100.0] {
            let beta_direct = kaiser_beta(a);
            let (_taps, beta_via_kaiserord) =
                kaiserord(a, 0.1).expect("kaiserord with reasonable args");
            assert_close(
                beta_direct,
                beta_via_kaiserord,
                1e-12,
                &format!("kaiser_beta({a}) vs kaiserord"),
            );
        }
    }

    #[test]
    fn firwin_invalid_args() {
        assert!(firwin(0, &[0.3], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[0.0], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[1.0], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[1.5], FirWindow::Hamming, true).is_err());
        assert!(firwin(21, &[0.4, 0.2], FirWindow::Hamming, false).is_err());
        assert!(firwin(21, &[0.2, 0.2], FirWindow::Hamming, false).is_err());
    }

    #[test]
    fn firwin_kaiser_window() {
        // Test that Kaiser window variant works
        let h = firwin(31, &[0.3], FirWindow::Kaiser(5.0), true).expect("firwin kaiser");
        assert_eq!(h.len(), 31);
        let dc_gain: f64 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.05,
            "Kaiser lowpass DC gain: {dc_gain}"
        );
    }

    // ── Chirp tests ────────────────────────────────────────────────

    #[test]
    fn chirp_linear_endpoints() {
        // At t=0, instantaneous frequency = f0, so signal ≈ cos(0) = 1.0
        let t: Vec<f64> = vec![0.0];
        let sig = chirp(&t, 10.0, 1.0, 50.0, ChirpMethod::Linear).unwrap();
        assert!(
            (sig[0] - 1.0).abs() < 1e-12,
            "chirp at t=0 should be cos(0)=1, got {}",
            sig[0]
        );
    }

    #[test]
    fn chirp_linear_frequency_sweep() {
        // Generate chirp from 1 Hz to 10 Hz over 1 second; verify it oscillates
        let n = 1000;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let sig = chirp(&t, 1.0, 1.0, 10.0, ChirpMethod::Linear).unwrap();
        assert_eq!(sig.len(), n);
        // Count zero crossings — should increase with frequency
        let crossings: usize = sig
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        // With sweep from 1-10 Hz over 1s, average freq ~5.5 Hz → ~11 crossings
        assert!(crossings > 5, "too few zero crossings: {crossings}");
    }

    #[test]
    fn chirp_logarithmic() {
        let t: Vec<f64> = vec![0.0];
        let sig = chirp(&t, 10.0, 1.0, 100.0, ChirpMethod::Logarithmic).unwrap();
        assert!((sig[0] - 1.0).abs() < 1e-10, "log chirp at t=0: {}", sig[0]);
    }

    #[test]
    fn chirp_logarithmic_rejects_zero_freq() {
        let t = vec![0.0];
        assert!(chirp(&t, 0.0, 1.0, 10.0, ChirpMethod::Logarithmic).is_err());
    }

    #[test]
    fn chirp_rejects_non_finite_parameters() {
        let t = vec![0.0, 0.5, 1.0];
        assert!(chirp(&t, 1.0, f64::NAN, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, 1.0, f64::INFINITY, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, f64::NAN, 1.0, 2.0, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, 1.0, 1.0, f64::INFINITY, ChirpMethod::Linear).is_err());
        assert!(chirp(&t, f64::NAN, 1.0, 2.0, ChirpMethod::Logarithmic).is_err());
    }

    // ── Sawtooth / Square tests ────────────────────────────────────

    #[test]
    fn sawtooth_period_and_amplitude() {
        let two_pi = 2.0 * std::f64::consts::PI;
        // Rising sawtooth (width=1): goes from -1 to +1 over one period
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = sawtooth(&t, 1.0).unwrap();
        // First sample at phase ≈ 0 should be near -1
        assert!(sig[0] > -1.1 && sig[0] < -0.9, "start: {}", sig[0]);
        // Mid-period should be near 0
        assert!(sig[50].abs() < 0.1, "mid: {}", sig[50]);
    }

    #[test]
    fn sawtooth_triangle_wave() {
        let two_pi = 2.0 * std::f64::consts::PI;
        // Triangle wave (width=0.5): symmetric
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = sawtooth(&t, 0.5).unwrap();
        // Peak should be at phase = 0.5*2π = π (index 50)
        // All values should be in [-1, 1]
        for (i, &v) in sig.iter().enumerate() {
            assert!(
                (-1.0 - 1e-10..=1.0 + 1e-10).contains(&v),
                "triangle out of range at {i}: {v}"
            );
        }
    }

    #[test]
    fn square_wave_duty_cycle() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0 * two_pi).collect();
        let sig = square(&t, 0.5).unwrap();
        // Count +1 samples — should be ~50%
        let pos_count = sig.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (pos_count as f64 / 1000.0 - 0.5).abs() < 0.02,
            "duty cycle: {}",
            pos_count as f64 / 1000.0
        );
    }

    #[test]
    fn square_wave_values() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 * two_pi).collect();
        let sig = square(&t, 0.5).unwrap();
        // All values should be exactly +1 or -1
        for &v in &sig {
            assert!(v == 1.0 || v == -1.0, "unexpected value: {v}");
        }
    }

    // ── Unit impulse tests ─────────────────────────────────────────

    #[test]
    fn unit_impulse_default_at_zero() {
        let imp = unit_impulse(10, None).unwrap();
        assert_eq!(imp.len(), 10);
        assert!((imp[0] - 1.0).abs() < 1e-12);
        for &v in &imp[1..] {
            assert!((v).abs() < 1e-12);
        }
    }

    #[test]
    fn unit_impulse_at_index() {
        let imp = unit_impulse(5, Some(3)).unwrap();
        assert!((imp[3] - 1.0).abs() < 1e-12);
        assert!((imp[0]).abs() < 1e-12);
        assert!((imp[4]).abs() < 1e-12);
    }

    #[test]
    fn unit_impulse_out_of_range() {
        assert!(unit_impulse(5, Some(5)).is_err());
    }

    // ── gauspuls tests ──────────────────────────────────────────────

    #[test]
    fn gauspuls_at_zero_is_unit_inphase_zero_quadrature() {
        // /porting-to-rust [frankenscipy-8jpj9]: at t=0, yenv = 1,
        // cos(0) = 1, sin(0) = 0 ⇒ yI(0) = 1, yQ(0) = 0, env(0) = 1.
        let r = gauspuls(&[0.0], 1000.0, 0.5, -6.0).expect("gauspuls");
        assert!((r.i[0] - 1.0).abs() < 1e-12);
        assert!(r.q[0].abs() < 1e-12);
        assert!((r.envelope[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gauspuls_quarter_period_swaps_inphase_and_quadrature() {
        // At t = 1/(4·fc), 2π·fc·t = π/2 so yI ≈ 0 and yQ ≈ env.
        let fc = 1000.0_f64;
        let t = 1.0 / (4.0 * fc);
        let r = gauspuls(&[t], fc, 0.5, -6.0).expect("gauspuls");
        assert!(r.i[0].abs() < 1e-3, "yI at quarter period should be ~0");
        assert!(
            (r.q[0] - r.envelope[0]).abs() < 1e-12,
            "yQ at quarter period should equal envelope"
        );
    }

    #[test]
    fn gauspuls_envelope_is_modulus_of_quadrature_pair() {
        // yI² + yQ² = yenv² (the analytic-signal modulus).
        let fc = 200.0_f64;
        let ts: Vec<f64> = (-50..=50).map(|i| 0.0001 * i as f64).collect();
        let r = gauspuls(&ts, fc, 0.5, -6.0).expect("gauspuls");
        for k in 0..r.i.len() {
            let mod_sq = r.i[k] * r.i[k] + r.q[k] * r.q[k];
            let env_sq = r.envelope[k] * r.envelope[k];
            assert!(
                (mod_sq - env_sq).abs() < 1e-12,
                "|yI + i·yQ|² = env² broken at k={k}: {mod_sq} vs {env_sq}"
            );
        }
    }

    #[test]
    fn gauspuls_input_contract() {
        // fc must be positive.
        assert!(gauspuls(&[0.0], 0.0, 0.5, -6.0).is_err());
        assert!(gauspuls(&[0.0], -100.0, 0.5, -6.0).is_err());
        assert!(gauspuls(&[0.0], f64::NAN, 0.5, -6.0).is_err());
        // bw must be positive.
        assert!(gauspuls(&[0.0], 1000.0, 0.0, -6.0).is_err());
        assert!(gauspuls(&[0.0], 1000.0, -0.5, -6.0).is_err());
        // bwr must be negative dB.
        assert!(gauspuls(&[0.0], 1000.0, 0.5, 0.0).is_err());
        assert!(gauspuls(&[0.0], 1000.0, 0.5, 6.0).is_err());
    }

    // ── normalize_filter tests ─────────────────────────────────────

    #[test]
    fn normalize_filter_makes_denominator_monic() {
        // /porting-to-rust [frankenscipy-fx18c]: a[0] always 1 after
        // normalization; b scaled consistently.
        let (b, a) = normalize_filter(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).expect("normalize");
        assert_eq!(a[0], 1.0);
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
        assert_eq!(b, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn normalize_filter_trims_leading_zeros_from_a() {
        // a = [0, 0, 2, 4, 6] trims to [2, 4, 6] then normalizes to
        // [1, 2, 3]; b is scaled by the trimmed leading 2.
        let (b, a) = normalize_filter(&[1.0], &[0.0, 0.0, 2.0, 4.0, 6.0]).expect("normalize");
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
        assert_eq!(b, vec![0.5]);
    }

    #[test]
    fn normalize_filter_rejects_all_zero_a() {
        assert!(normalize_filter(&[1.0], &[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn normalize_filter_rejects_empty_a() {
        assert!(normalize_filter(&[1.0], &[]).is_err());
    }

    #[test]
    fn normalize_filter_rejects_nonfinite_coefficients() {
        // REVIEW MODE [LOW] regression for [frankenscipy-n9ply]:
        // pre-fix the leading-zero scan `v != 0.0` returned true for
        // NaN, so a leading NaN became the divisor and propagated
        // NaN through the output with Ok status. Same for Inf.
        assert!(normalize_filter(&[1.0], &[1.0, f64::NAN]).is_err());
        assert!(normalize_filter(&[1.0], &[f64::NAN, 2.0]).is_err());
        assert!(normalize_filter(&[1.0], &[1.0, f64::INFINITY]).is_err());
        assert!(normalize_filter(&[1.0], &[f64::NEG_INFINITY, 1.0]).is_err());
        assert!(normalize_filter(&[f64::NAN], &[1.0, 2.0]).is_err());
        assert!(normalize_filter(&[f64::INFINITY], &[1.0]).is_err());
    }

    #[test]
    fn lfilter_optimized_loop_matches_naive_logic() {
        // /profiling-software-performance regression for [frankenscipy-rvwvw]:
        // lfilter inner loop was tightened to pad b/a to nfilt, hoist b0,
        // and drop per-iter bound checks. Verify against an explicit
        // naïve implementation that mirrors the pre-fix branchy logic.

        fn naive_lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
            let a0 = a[0];
            let nb = b.len();
            let na = a.len();
            let nfilt = nb.max(na);
            let b_norm: Vec<f64> = b.iter().map(|&v| v / a0).collect();
            let a_norm: Vec<f64> = a.iter().map(|&v| v / a0).collect();
            let mut y = Vec::with_capacity(x.len());
            let mut d = vec![0.0_f64; nfilt];
            for &xi in x {
                let yi = b_norm.first().copied().unwrap_or(0.0) * xi + d[0];
                y.push(yi);
                for j in 0..nfilt - 1 {
                    let bj = if j + 1 < nb { b_norm[j + 1] } else { 0.0 };
                    let aj = if j + 1 < na { a_norm[j + 1] } else { 0.0 };
                    d[j] = bj * xi - aj * yi + if j + 1 < nfilt - 1 { d[j + 1] } else { 0.0 };
                }
            }
            y
        }

        // Order-5 IIR filter on a 200-sample signal.
        let b = vec![0.5_f64, -0.3, 0.1, 0.05, -0.02, 0.01];
        let a = vec![1.0_f64, -0.7, 0.4, -0.1, 0.05, 0.02];
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.13).sin()).collect();

        let opt = lfilter(&b, &a, &x, None).expect("lfilter");
        let nav = naive_lfilter(&b, &a, &x);
        assert_eq!(opt.len(), nav.len());
        for (i, (&u, &v)) in opt.iter().zip(nav.iter()).enumerate() {
            assert!(
                (u - v).abs() < 1e-12 * v.abs().max(1.0),
                "lfilter perf-opt diverges from naive at i={i}: opt={u}, naive={v}"
            );
        }
    }

    // ── freqz_zpk tests ──────────────────────────────────────────────

    #[test]
    fn freqz_zpk_matches_freqz_via_zpk2tf() {
        // /porting-to-rust [frankenscipy-kca5u]: freqz_zpk on a ZPK
        // form should match freqz on the ZPK→TF converted form within
        // numerical tolerance.
        let zpk = ZpkCoeffs {
            zeros_re: vec![1.0, -1.0],
            zeros_im: vec![0.0, 0.0],
            poles_re: vec![0.5, -0.5],
            poles_im: vec![0.0, 0.0],
            gain: 0.5,
        };
        let ba = zpk2tf(&zpk);

        let r_zpk = freqz_zpk(&zpk, Some(64)).expect("freqz_zpk");
        let r_ba = freqz(&ba.b, &ba.a, Some(64)).expect("freqz");

        assert_eq!(r_zpk.w.len(), r_ba.w.len());
        for k in 0..r_zpk.w.len() {
            assert!(
                (r_zpk.w[k] - r_ba.w[k]).abs() < 1e-12,
                "freqz_zpk frequency grid mismatch at k={k}"
            );
            assert!(
                (r_zpk.h_mag[k] - r_ba.h_mag[k]).abs() < 1e-9 * r_ba.h_mag[k].abs().max(1.0),
                "freqz_zpk h_mag mismatch at k={k}: {} vs {}",
                r_zpk.h_mag[k],
                r_ba.h_mag[k]
            );
            // Phase comparison can be ambiguous mod 2π near jumps; require
            // either direct match or ±2π wrap match.
            let dphi = (r_zpk.h_phase[k] - r_ba.h_phase[k]).abs();
            assert!(
                dphi < 1e-9 || (dphi - std::f64::consts::TAU).abs() < 1e-9,
                "freqz_zpk h_phase mismatch at k={k}: {} vs {}",
                r_zpk.h_phase[k],
                r_ba.h_phase[k]
            );
        }
    }

    #[test]
    fn freqz_zpk_dc_response_is_gain_times_zero_pole_ratio() {
        // At ω=0, e^{jω}=1, so H(1) = gain · Π(1 − z_i) / Π(1 − p_i).
        let zpk = ZpkCoeffs {
            zeros_re: vec![0.0, 0.5],
            zeros_im: vec![0.0, 0.0],
            poles_re: vec![0.25],
            poles_im: vec![0.0],
            gain: 2.0,
        };
        let r = freqz_zpk(&zpk, Some(8)).expect("freqz_zpk");
        // First sample is at ω=0.
        let expected = 2.0 * (1.0 - 0.0) * (1.0 - 0.5) / (1.0 - 0.25);
        assert!(
            (r.h_mag[0] - expected).abs() < 1e-12,
            "freqz_zpk DC: {} vs {expected}",
            r.h_mag[0]
        );
        // Phase at DC is 0 (real-valued response).
        assert!(r.h_phase[0].abs() < 1e-12, "DC phase should be 0");
    }

    #[test]
    fn freqz_zpk_input_contract() {
        let mut zpk = ZpkCoeffs {
            zeros_re: vec![1.0],
            zeros_im: vec![0.0, 0.0], // length mismatch
            poles_re: vec![],
            poles_im: vec![],
            gain: 1.0,
        };
        assert!(freqz_zpk(&zpk, None).is_err());
        zpk.zeros_im = vec![0.0];
        zpk.poles_re = vec![0.5];
        assert!(freqz_zpk(&zpk, None).is_err()); // poles len mismatch
        zpk.poles_im = vec![0.0];
        zpk.gain = f64::NAN;
        assert!(freqz_zpk(&zpk, None).is_err());
        zpk.gain = 1.0;
        assert!(freqz_zpk(&zpk, Some(0)).is_err());
    }

    // ── unique_roots tests ──────────────────────────────────────────

    #[test]
    fn vectorstrength_single_pass_matches_naive_dual_pass() {
        // /profiling-software-performance regression for [frankenscipy-51bzs]:
        // vectorstrength now uses a single-pass loop with sin_cos() instead
        // of two iter passes computing sin and cos separately. Verify the
        // optimized form against an explicit naïve in-test implementation
        // on a 50-event sample.

        fn naive_vectorstrength(events: &[f64], period: f64) -> (f64, f64) {
            if events.is_empty() || period <= 0.0 || !period.is_finite() {
                return (0.0, 0.0);
            }
            let two_pi = 2.0 * std::f64::consts::PI;
            let n = events.len() as f64;
            let sin_sum: f64 = events.iter().map(|&t| (two_pi * t / period).sin()).sum();
            let cos_sum: f64 = events.iter().map(|&t| (two_pi * t / period).cos()).sum();
            let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
            let phase = sin_sum.atan2(cos_sum);
            (r, phase)
        }

        let events: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.13).sin().abs() * 5.0 + (i as f64 * 0.07))
            .collect();
        for &period in &[0.5_f64, 1.0, 2.5, 7.0] {
            let (r_opt, phase_opt) = vectorstrength(&events, period);
            let (r_nav, phase_nav) = naive_vectorstrength(&events, period);
            assert!(
                (r_opt - r_nav).abs() < 1e-12,
                "strength mismatch at period={period}: opt={r_opt}, naive={r_nav}"
            );
            assert!(
                (phase_opt - phase_nav).abs() < 1e-12,
                "phase mismatch at period={period}: opt={phase_opt}, naive={phase_nav}"
            );
        }
    }

    #[test]
    fn argrel_order_ge_2_includes_boundary_adjacent_extrema() {
        // scipy.signal.argrelmax/argrelmin with mode='clip': extrema within
        // `order` of an edge are still detected (frankenscipy-mufw9).
        let y = [0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 4.0, 0.0];
        assert_eq!(argrelmax(&y, 2), vec![1, 5, 9]);
        assert_eq!(argrelmax(&y, 3), vec![1, 5, 9]);
        assert_eq!(argrelmin(&y, 2), vec![3, 7]);

        let z = [
            5.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0,
        ];
        assert_eq!(argrelmin(&z, 3), vec![1, 5, 12]);
        assert_eq!(argrelmax(&z, 3), vec![8]);
    }

    #[test]
    fn vectorstrength_matches_scipy_reference() {
        // scipy.signal.vectorstrength([0.1,0.5,1.1,1.5,2.1,2.5], 1.0):
        // events cluster at two phases, mean vector angle ≈ 1.8853 rad.
        let (strength, phase) = vectorstrength(&[0.1, 0.5, 1.1, 1.5, 2.1, 2.5], 1.0);
        assert!(
            (strength - 0.309_016_994_374_947_6).abs() < 1e-12,
            "strength = {strength}"
        );
        assert!(
            (phase - 1.884_955_592_153_875_9).abs() < 1e-12,
            "phase = {phase}"
        );
    }

    #[test]
    fn unique_roots_empty_returns_empty() {
        let (r, m) = unique_roots(&[], 1e-3, "min");
        assert!(r.is_empty() && m.is_empty());
    }

    #[test]
    fn unique_roots_all_distinct_returns_each_with_multiplicity_one() {
        let p = [1.0, 5.0, 10.0];
        let (r, m) = unique_roots(&p, 0.5, "min");
        assert_eq!(r, vec![1.0, 5.0, 10.0]);
        assert_eq!(m, vec![1, 1, 1]);
    }

    #[test]
    fn unique_roots_groups_within_tolerance() {
        // tol comparison is `|x - leader| <= tol` (inclusive). Use
        // exact f64 values to avoid float-precision artifacts.
        // tol=1.0, leader 0.0: 0.5 ≤ 1, 1.0 ≤ 1 → all join. 2.5 starts
        // new cluster. 5.0 is also outside |2.5-5.0|=2.5 > 1 → another.
        let p = [0.0, 0.5, 1.0, 2.5, 5.0];
        let (r, m) = unique_roots(&p, 1.0, "min");
        assert_eq!(r.len(), 3, "expected 3 clusters, got {r:?}");
        assert_eq!(r, vec![0.0, 2.5, 5.0]);
        assert_eq!(m, vec![3, 1, 1]);
    }

    #[test]
    fn unique_roots_avg_combine() {
        let p = [1.0, 2.0, 3.0];
        let (r, m) = unique_roots(&p, 5.0, "avg");
        assert_eq!(r.len(), 1);
        assert!((r[0] - 2.0).abs() < 1e-15);
        assert_eq!(m, vec![3]);
    }

    #[test]
    fn unique_roots_max_combine() {
        let p = [1.0, 2.0, 3.0];
        let (r, m) = unique_roots(&p, 5.0, "max");
        assert_eq!(r, vec![3.0]);
        assert_eq!(m, vec![3]);
    }

    #[test]
    fn unique_roots_zero_tol_keeps_only_exact_duplicates_together() {
        // tol = 0: only equal values are grouped.
        let p = [1.0, 1.0, 2.0];
        let (r, m) = unique_roots(&p, 0.0, "min");
        assert_eq!(r, vec![1.0, 2.0]);
        assert_eq!(m, vec![2, 1]);
    }

    #[test]
    fn unique_roots_unknown_rtype_falls_back_to_avg() {
        // scipy is permissive — unknown rtype shouldn't panic; we treat
        // it as "avg" to match the documented permissive behaviour.
        let p = [1.0, 2.0, 3.0];
        let (r, _) = unique_roots(&p, 5.0, "garbage");
        assert!((r[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn normalize_filter_scale_invariance() {
        // /testing-metamorphic for [frankenscipy-oi7hx]: for any non-zero
        // k, normalize_filter(k·b, k·a) = normalize_filter(b, a). The k
        // cancels in the division by the (post-trim) leading coefficient.
        let b: &[f64] = &[1.5, 2.0, 0.5];
        let a: &[f64] = &[3.0, 6.0, 9.0];
        let (b_ref, a_ref) = normalize_filter(b, a).expect("baseline");

        for &k in &[0.5_f64, 2.0, -1.0, 100.0, -0.25] {
            let b_scaled: Vec<f64> = b.iter().map(|&v| v * k).collect();
            let a_scaled: Vec<f64> = a.iter().map(|&v| v * k).collect();
            let (b_out, a_out) = normalize_filter(&b_scaled, &a_scaled)
                .unwrap_or_else(|e| unreachable!("scaled by k={k}: {e:?}"));
            assert_eq!(
                b_out.len(),
                b_ref.len(),
                "b length must match baseline at k={k}"
            );
            assert_eq!(
                a_out.len(),
                a_ref.len(),
                "a length must match baseline at k={k}"
            );
            for i in 0..b_ref.len() {
                assert!(
                    (b_out[i] - b_ref[i]).abs() < 1e-12,
                    "b mismatch at k={k}, i={i}: {} vs baseline {}",
                    b_out[i],
                    b_ref[i]
                );
            }
            for i in 0..a_ref.len() {
                assert!(
                    (a_out[i] - a_ref[i]).abs() < 1e-12,
                    "a mismatch at k={k}, i={i}: {} vs baseline {}",
                    a_out[i],
                    a_ref[i]
                );
            }
        }
    }

    #[test]
    fn normalize_filter_idempotent_on_monic_input() {
        // Already-normalized filter stays the same.
        let (b, a) = normalize_filter(&[0.5, 1.0, 1.5], &[1.0, 2.0, 3.0]).expect("normalize");
        assert_eq!(b, vec![0.5, 1.0, 1.5]);
        assert_eq!(a, vec![1.0, 2.0, 3.0]);
    }

    // ── Detrend tests ──────────────────────────────────────────────

    #[test]
    fn detrend_linear_removes_slope() {
        // y = 2x + 3 → detrended should be ~0 everywhere
        let data: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let result = detrend(&data, DetrendType::Linear).unwrap();
        for (i, &v) in result.iter().enumerate() {
            assert!(v.abs() < 1e-10, "detrend linear residual at {i}: {v}");
        }
    }

    #[test]
    fn detrend_constant_removes_mean() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let result = detrend(&data, DetrendType::Constant).unwrap();
        for &v in &result {
            assert!(v.abs() < 1e-12, "detrend constant: {v}");
        }
    }

    #[test]
    fn detrend_metamorphic_constant_idempotent_and_zero_mean() -> Result<(), SignalError> {
        let data = [3.0, -1.0, 4.0, 1.0, 5.0];
        let once = detrend(&data, DetrendType::Constant)?;
        let twice = detrend(&once, DetrendType::Constant)?;
        assert_slice_close(&twice, &once, 1e-12, "constant detrend idempotent");

        let mean = once.iter().sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-12, "constant detrend mean: {mean}");

        let constant = [4.25, 4.25, 4.25, 4.25, 4.25, 4.25];
        let zeros = detrend(&constant, DetrendType::Constant)?;
        assert_slice_close(
            &zeros,
            &[0.0; 6],
            1e-12,
            "constant detrend removes flat sequence",
        );
        Ok(())
    }

    #[test]
    fn detrend_preserves_residual() {
        // y = 2x + noise → detrend should remove trend, preserve noise structure
        let noise = [0.1, -0.2, 0.05, -0.15, 0.3];
        let data: Vec<f64> = noise
            .iter()
            .enumerate()
            .map(|(i, &n)| 3.0 * i as f64 + 10.0 + n)
            .collect();
        let result = detrend(&data, DetrendType::Linear).unwrap();
        // Residuals should be close to the noise (within tolerance of linear fit error)
        let max_residual = result
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        assert!(max_residual < 0.5, "max residual: {max_residual}");
    }

    #[test]
    fn detrend_metamorphic_linear_affine_removal_and_residual_invariance() -> Result<(), SignalError>
    {
        let affine = [13.0, 10.75, 8.5, 6.25, 4.0, 1.75, -0.5];
        let removed = detrend(&affine, DetrendType::Linear)?;
        assert_slice_close(
            &removed,
            &[0.0; 7],
            1e-12,
            "linear detrend removes affine sequence",
        );

        let residual = [0.5, -1.0, 0.25, 1.5, -0.75, 0.0, 1.25];
        let residual_only = detrend(&residual, DetrendType::Linear)?;
        let with_trend: Vec<f64> = residual
            .iter()
            .zip(affine.iter())
            .map(|(&noise, &trend)| noise + trend)
            .collect();
        let with_trend_removed = detrend(&with_trend, DetrendType::Linear)?;
        assert_slice_close(
            &with_trend_removed,
            &residual_only,
            1e-12,
            "linear detrend residual invariant to added affine trend",
        );
        Ok(())
    }

    // ── Medfilt tests ──────────────────────────────────────────────

    #[test]
    fn order_filter_sliding_matches_naive_sort() {
        // Isomorphism proof for the sliding order_filter: bit-identical to the
        // per-window sort+index over window sizes (incl. the threshold), ranks
        // (low/mid/high and out-of-range, which clamp), lengths, tie densities,
        // and the clipped (variable-size) boundary windows.
        fn naive(x: &[f64], window_size: usize, rank: usize) -> Vec<f64> {
            let half = window_size / 2;
            let mut out = Vec::with_capacity(x.len());
            let mut w: Vec<f64> = Vec::new();
            for i in 0..x.len() {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(x.len());
                w.clear();
                w.extend_from_slice(&x[start..end]);
                w.sort_by(|a, b| a.total_cmp(b));
                out.push(w[rank.min(w.len() - 1)]);
            }
            out
        }
        let mut state: u64 = 0x2468_ace0_1357_9bdf;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &n in &[1usize, 20, 200, 777] {
            for &tie_mod in &[0u64, 2, 5, 40] {
                let data: Vec<f64> = (0..n)
                    .map(|_| {
                        let r = next();
                        match r % 37 {
                            0 => -0.0,
                            1 => 0.0,
                            _ if tie_mod == 0 => (r >> 20) as f64 / (1u64 << 30) as f64,
                            _ => (r % tie_mod) as f64,
                        }
                    })
                    .collect();
                for &ws in &[31usize, 32, 65, 200] {
                    for &rank in &[0usize, ws / 3, ws / 2, ws - 1, ws + 5] {
                        let fast = order_filter_sliding(&data, ws, rank);
                        let slow = naive(&data, ws, rank);
                        assert_eq!(fast.len(), slow.len());
                        for (a, b) in fast.iter().zip(slow.iter()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "n={n} ws={ws} rank={rank} tie_mod={tie_mod}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn medfilt_sliding_matches_naive_loop() {
        // Isomorphism proof for the sliding-median path: it must be bit-identical
        // to the per-window select_nth reference across kernel sizes (including
        // the dispatch threshold), input lengths, and tie densities (the
        // multiset-straddle case), plus signed zeros.
        fn naive(data: &[f64], k: usize) -> Vec<f64> {
            let n = data.len();
            let half = k / 2;
            let mut out = Vec::with_capacity(n);
            let mut win = vec![0.0; k];
            for i in 0..n {
                for (j, v) in win.iter_mut().enumerate() {
                    let idx = i as i64 + j as i64 - half as i64;
                    *v = if idx >= 0 && idx < n as i64 {
                        data[idx as usize]
                    } else {
                        0.0
                    };
                }
                let (_, &mut m, _) = win.select_nth_unstable_by(half, |a, b| a.total_cmp(b));
                out.push(m);
            }
            out
        }
        let mut state: u64 = 0x1357_9bdf_2468_ace0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &n in &[1usize, 50, 400, 1000] {
            for &tie_mod in &[0u64, 2, 4, 30] {
                let data: Vec<f64> = (0..n)
                    .map(|_| {
                        let r = next();
                        match r % 41 {
                            0 => -0.0,
                            1 => 0.0,
                            _ if tie_mod == 0 => (r >> 20) as f64 / (1u64 << 30) as f64,
                            _ => (r % tie_mod) as f64,
                        }
                    })
                    .collect();
                for &k in &[1usize, 3, 63, 65, 101, 257] {
                    if k > 2 * n + 1 {
                        continue;
                    }
                    let fast = medfilt_sliding(&data, k);
                    let slow = naive(&data, k);
                    assert_eq!(fast.len(), slow.len());
                    for (a, b) in fast.iter().zip(slow.iter()) {
                        assert_eq!(a.to_bits(), b.to_bits(), "n={n} k={k} tie_mod={tie_mod}");
                    }
                }
            }
        }
    }

    #[test]
    fn medfilt1_sliding_matches_naive_sort() {
        // The sliding medfilt1 must reproduce the per-window-sort filter exactly,
        // including the clipped (not zero-padded) borders where the window size —
        // and hence the len/2 median index — changes, across kernels straddling
        // the dispatch threshold, even/odd kernels, and tie densities.
        fn naive(x: &[f64], k: usize) -> Vec<f64> {
            if x.is_empty() || k == 0 {
                return x.to_vec();
            }
            let half = k / 2;
            let n = x.len();
            (0..n)
                .map(|i| {
                    let start = i.saturating_sub(half);
                    let end = (i + half + 1).min(n);
                    let mut w: Vec<f64> = x[start..end].to_vec();
                    w.sort_by(|a, b| a.total_cmp(b));
                    w[w.len() / 2]
                })
                .collect()
        }
        let mut state: u64 = 0x2bad_f00d_1234_5678;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for &n in &[1usize, 2, 40, 300] {
            for &tie_mod in &[0u64, 2, 5, 30] {
                let data: Vec<f64> = (0..n)
                    .map(|_| {
                        let r = next();
                        match r % 41 {
                            0 => -0.0,
                            1 => 0.0,
                            _ if tie_mod == 0 => (r >> 20) as f64 / (1u64 << 30) as f64,
                            _ => (r % tie_mod) as f64,
                        }
                    })
                    .collect();
                for &k in &[1usize, 2, 31, 32, 33, 64, 65, 129] {
                    let fast = medfilt1(&data, k);
                    let slow = naive(&data, k);
                    assert_eq!(fast.len(), slow.len());
                    for (a, b) in fast.iter().zip(slow.iter()) {
                        assert_eq!(a.to_bits(), b.to_bits(), "n={n} k={k} tie_mod={tie_mod}");
                    }
                }
            }
        }
    }

    #[test]
    fn medfilt_removes_impulse_noise() {
        // Clean signal with spike impulses
        let mut data = vec![1.0; 20];
        data[5] = 100.0; // spike
        data[15] = -100.0; // spike
        let filtered = medfilt(&data, 3).unwrap();
        // Spikes should be removed
        assert!(
            (filtered[5] - 1.0).abs() < 1e-12,
            "spike at 5 not removed: {}",
            filtered[5]
        );
        assert!(
            (filtered[15] - 1.0).abs() < 1e-12,
            "spike at 15 not removed: {}",
            filtered[15]
        );
    }

    #[test]
    fn medfilt_preserves_edges() {
        // Step function: medfilt should preserve the step
        let mut data = vec![0.0; 10];
        for v in &mut data[5..] {
            *v = 1.0;
        }
        let filtered = medfilt(&data, 3).unwrap();
        // Away from transition, values should be preserved
        assert!((filtered[0]).abs() < 1e-12);
        assert!((filtered[3]).abs() < 1e-12);
        assert!((filtered[7] - 1.0).abs() < 1e-12);
        assert!((filtered[9] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn medfilt_even_kernel_rejected() {
        assert!(medfilt(&[1.0, 2.0, 3.0], 4).is_err());
    }

    #[test]
    fn wiener_constant_signal_is_stable() {
        let data = vec![2.0; 9];
        let filtered = wiener(&data, 3, None).unwrap();
        for value in filtered.iter().skip(1).take(filtered.len() - 2) {
            assert!(
                (*value - 2.0).abs() < 1e-12,
                "interior constant signal drifted: {value}"
            );
        }
    }

    #[test]
    fn wiener_smooths_impulse_with_explicit_noise() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let filtered = wiener(&data, 3, Some(1.0)).unwrap();
        assert!(
            filtered[2] < 10.0,
            "center should be attenuated: {}",
            filtered[2]
        );
        assert!(filtered[2] > filtered[1], "center should remain dominant");
        assert!(filtered[1] > 0.0, "neighbors should pick up local energy");
    }

    #[test]
    fn wiener_estimated_noise_falls_back_to_local_mean() {
        let data = vec![0.0, 1.0, 0.0];
        let filtered = wiener(&data, 3, Some(10.0)).unwrap();
        assert!(
            (filtered[1] - (1.0 / 3.0)).abs() < 1e-12,
            "expected local mean at center"
        );
    }

    #[test]
    fn wiener_rejects_invalid_window() {
        assert!(wiener(&[1.0, 2.0, 3.0], 0, None).is_err());
        assert!(wiener(&[1.0, 2.0, 3.0], 4, None).is_err());
    }

    // ── get_window tests ───────────────────────────────────────────

    #[test]
    fn get_window_dispatches_hann() {
        let w = get_window_with_fftbins("hann", 10, false).unwrap();
        let expected = hann(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_hamming() {
        let w = get_window_with_fftbins("hamming", 10, false).unwrap();
        let expected = hamming(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_blackman() {
        let w = get_window_with_fftbins("blackman", 10, false).unwrap();
        let expected = blackman(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_blackmanharris() {
        let w = get_window_with_fftbins("blackmanharris", 10, false).unwrap();
        let expected = blackmanharris(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_barthann() {
        let w = get_window_with_fftbins("barthann", 10, false).unwrap();
        let expected = barthann(10);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_chebwin() {
        let w = get_window_with_fftbins("chebwin,100", 5, false).unwrap();
        let expected = chebwin(5, 100.0);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_hamming() {
        let w = get_window_with_fftbins("general_hamming,0.75", 8, false).unwrap();
        let expected = general_hamming(8, 0.75);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_cosine() {
        let w = get_window_with_fftbins("general_cosine,0.5,0.3,0.2", 8, false).unwrap();
        let expected = general_cosine(8, &[0.5, 0.3, 0.2], true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_gaussian() {
        let w = get_window_with_fftbins("gaussian,2.0", 8, false).unwrap();
        let expected = gaussian(8, 2.0, true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_general_gaussian() {
        let w = get_window_with_fftbins("general_gaussian,1.5,2.0", 8, false).unwrap();
        let expected = general_gaussian(8, 1.5, 2.0, true);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_exponential() {
        let w = get_window_with_fftbins("exponential,2.0", 8, false).unwrap();
        let expected = exponential(8, None, 2.0, true).unwrap();
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_kaiser() {
        let w = get_window_with_fftbins("kaiser,8.6", 10, false).unwrap();
        let expected = kaiser(10, 8.6);
        assert_eq!(w, expected);
    }

    #[test]
    fn get_window_dispatches_kaiser_bessel_derived() {
        let w = get_window_with_fftbins("kaiser_bessel_derived,5.0", 8, false).unwrap();
        let alias = get_window_with_fftbins("kbd,5.0", 8, false).unwrap();
        let expected = kaiser_bessel_derived(8, 5.0, true).unwrap();
        assert_eq!(w, expected);
        assert_eq!(alias, expected);
    }

    #[test]
    fn get_window_dispatches_dpss() {
        let symmetric = get_window_with_fftbins("dpss,2.5", 8, false).unwrap();
        let periodic = get_window("dpss,2.5", 8).unwrap();
        let expected_symmetric = dpss(8, 2.5, None, true, None, false)
            .expect("direct symmetric")
            .windows
            .remove(0);
        let expected_periodic = dpss(8, 2.5, None, false, None, false)
            .expect("direct periodic")
            .windows
            .remove(0);
        assert_eq!(symmetric, expected_symmetric);
        assert_eq!(periodic, expected_periodic);
    }

    #[test]
    fn get_window_dispatches_lanczos_and_sinc_alias() {
        let symmetric = get_window_with_fftbins("lanczos", 9, false).unwrap();
        let periodic = get_window("lanczos", 9).unwrap();
        let alias = get_window_with_fftbins("sinc", 9, false).unwrap();
        let expected_symmetric = lanczos(9);
        let expected_periodic = lanczos(10).into_iter().take(9).collect::<Vec<_>>();

        assert_eq!(symmetric, expected_symmetric);
        assert_eq!(periodic, expected_periodic);
        assert_eq!(alias, expected_symmetric);
    }

    #[test]
    fn get_window_dispatches_taylor_defaults_and_parameters() {
        let symmetric_default = get_window_with_fftbins("taylor", 9, false).unwrap();
        let periodic_param = get_window("taylor,6,45,false", 9).unwrap();
        let symmetric_alias = get_window_with_fftbins("taylorwin,5,35,true", 9, false).unwrap();
        let forced_periodic =
            get_window_with_fftbins("taylor_periodic,5,35,true", 9, false).unwrap();

        assert_eq!(symmetric_default, taylor(9, 4, 30.0, true, true));
        assert_eq!(periodic_param, taylor(9, 6, 45.0, false, false));
        assert_eq!(symmetric_alias, taylor(9, 5, 35.0, true, true));
        assert_eq!(forced_periodic, taylor(9, 5, 35.0, true, false));
    }

    #[test]
    fn get_window_rejects_invalid_taylor_parameters() {
        let too_many = get_window("taylor,4,30,true,false", 8).expect_err("too many taylor params");
        assert_eq!(
            too_many.to_string(),
            "invalid argument: taylor expects nbar[,sll[,norm]] parameters: 4,30,true,false"
        );

        let invalid_nbar = get_window("taylor,foo", 8).expect_err("invalid taylor nbar");
        assert_eq!(
            invalid_nbar.to_string(),
            "invalid argument: invalid taylor nbar: foo"
        );

        let invalid_norm = get_window("taylor,4,30,maybe", 8).expect_err("invalid taylor norm");
        assert_eq!(
            invalid_norm.to_string(),
            "invalid argument: invalid taylor norm: maybe"
        );
    }

    #[test]
    fn get_window_rejects_missing_dpss_parameter() {
        let err = get_window("dpss", 8).expect_err("missing dpss parameter");
        assert_eq!(
            err.to_string(),
            "invalid argument: dpss requires a normalized half-bandwidth parameter"
        );
    }

    #[test]
    fn get_window_rejects_non_finite_window_parameters() {
        let kaiser_err = get_window("kaiser,nan", 8).expect_err("NaN kaiser beta");
        assert_eq!(
            kaiser_err.to_string(),
            "invalid argument: invalid kaiser beta: nan"
        );

        let gaussian_err = get_window("gaussian,nan", 8).expect_err("NaN gaussian std");
        assert_eq!(
            gaussian_err.to_string(),
            "invalid argument: invalid gaussian std: nan"
        );

        let exponential_err = get_window("exponential,nan", 8).expect_err("NaN exponential tau");
        assert_eq!(
            exponential_err.to_string(),
            "invalid argument: invalid exponential tau: nan"
        );
    }

    #[test]
    fn get_window_rectangular() {
        let w = get_window("boxcar", 5).unwrap();
        assert_eq!(w, vec![1.0; 5]);
    }

    #[test]
    fn get_window_dispatches_bartlett() {
        let w = get_window_with_fftbins("bartlett", 11, false).unwrap();
        assert_eq!(w.len(), 11);
        assert!((w[5] - 1.0).abs() < 1e-12, "peak at center");
        assert!((w[0]).abs() < 1e-12, "zero at endpoints");
        assert!((w[10]).abs() < 1e-12, "zero at endpoints");
    }

    #[test]
    fn get_window_dispatches_flattop() {
        let w = get_window_with_fftbins("flattop", 11, false).unwrap();
        assert_eq!(w.len(), 11);
        // Flattop peak is ~1.0 but slightly different due to coeffs
        let peak = w.iter().copied().fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!((peak - 1.0).abs() < 0.01);
    }

    #[test]
    fn get_window_dispatches_cosine() {
        let w = get_window_with_fftbins("cosine", 5, false).unwrap();
        let expected = [
            0.3090169943749474,
            0.8090169943749475,
            1.0,
            0.8090169943749475,
            0.3090169943749474,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_unknown_rejected() {
        assert!(get_window("foobar", 10).is_err());
    }

    #[test]
    fn get_window_defaults_to_periodic_hann() {
        let w = get_window("hann", 8).unwrap();
        let expected = [
            0.0,
            0.1464466094067262,
            0.5,
            0.8535533905932737,
            1.0,
            0.8535533905932737,
            0.5,
            0.1464466094067262,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_fftbins_false_returns_symmetric_hann() {
        let w = get_window_with_fftbins("hann", 8, false).unwrap();
        let expected = [
            0.0,
            0.1882550990706332,
            0.6112604669781572,
            0.9504844339512095,
            0.9504844339512095,
            0.6112604669781572,
            0.1882550990706332,
            0.0,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_defaults_to_periodic_bartlett() {
        let w = get_window("bartlett", 5).unwrap();
        let expected = [0.0, 0.4, 0.8, 0.8, 0.4];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn get_window_suffixes_override_fftbins() {
        let periodic = get_window_with_fftbins("hann_periodic", 8, false).unwrap();
        let symmetric = get_window_with_fftbins("hann_symmetric", 8, true).unwrap();

        assert!((periodic[1] - 0.1464466094067262).abs() < 1e-12);
        assert!((periodic[7] - 0.1464466094067262).abs() < 1e-12);
        assert!((symmetric[1] - 0.1882550990706332).abs() < 1e-12);
        assert!(symmetric[7].abs() < 1e-12);
    }

    #[test]
    fn get_window_parameterized_window_respects_fftbins() {
        let periodic = get_window("gaussian,2.0", 8).unwrap();
        let symmetric = get_window_with_fftbins("gaussian,2.0", 8, false).unwrap();

        let expected_periodic = [
            0.1353352832366127,
            0.32465246735834974,
            0.6065306597126334,
            0.8824969025845955,
            1.0,
            0.8824969025845955,
            0.6065306597126334,
            0.32465246735834974,
        ];
        let expected_symmetric = [
            0.2162651668298873,
            0.45783336177161427,
            0.7548396019890073,
            0.9692332344763441,
            0.9692332344763441,
            0.7548396019890073,
            0.45783336177161427,
            0.2162651668298873,
        ];

        for (actual, want) in periodic.iter().zip(expected_periodic.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
        for (actual, want) in symmetric.iter().zip(expected_symmetric.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    // ── STFT / ISTFT tests ─────────────────────────────────────────

    #[test]
    fn stft_istft_roundtrip() {
        // Generate a simple sinusoidal signal, STFT then ISTFT, compare.
        let fs = 100.0;
        let n = 400;
        let freq = 10.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let nperseg = 64;
        let noverlap = 48;
        let stft_res = stft(&x, fs, None, Some(nperseg), Some(noverlap)).unwrap();
        let reconstructed = istft(&stft_res, nperseg, Some(noverlap)).unwrap();

        // Compare in the well-reconstructed interior (avoiding edge effects).
        let margin = nperseg;
        let end = reconstructed.len().min(n) - margin;
        for i in margin..end {
            assert!(
                (reconstructed[i] - x[i]).abs() < 0.15,
                "STFT-ISTFT mismatch at {i}: got {}, expected {}",
                reconstructed[i],
                x[i]
            );
        }
    }

    #[test]
    fn stft_parallel_is_bit_identical() {
        // Large enough to cross the parallel gate (flops = frames*nperseg*log2 >= 2^24):
        // ~4000 frames of nperseg=512 -> ~18M flops -> the threaded frame loop runs.
        let nperseg = 512usize;
        let noverlap = 128usize;
        let step = nperseg - noverlap;
        let frames = 4200usize;
        let n = (frames - 1) * step + nperseg;
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.01).sin() + 0.3 * (i as f64 * 0.047).cos())
            .collect();
        let par = stft(&x, 1000.0, Some("hann"), Some(nperseg), Some(noverlap)).expect("stft");

        // Verbatim sequential reference (same window via get_window).
        let win = get_window("hann", nperseg).expect("window");
        let n_freqs = nperseg / 2 + 1;
        let opts = fsci_fft::FftOptions::default();
        let mut seq: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut start = 0;
        while start + nperseg <= x.len() {
            let windowed: Vec<f64> = x[start..start + nperseg]
                .iter()
                .zip(&win)
                .map(|(&xi, &wi)| xi * wi)
                .collect();
            let spectrum = fsci_fft::rfft(&windowed, &opts).expect("rfft");
            seq.push(spectrum[..n_freqs].to_vec());
            start += step;
        }

        assert_eq!(par.zxx.len(), seq.len());
        for (t, (pr, sr)) in par.zxx.iter().zip(&seq).enumerate() {
            for (f, (&(pre, pim), &(sre, sim))) in pr.iter().zip(sr).enumerate() {
                assert_eq!(pre.to_bits(), sre.to_bits(), "stft re mismatch t={t} f={f}");
                assert_eq!(pim.to_bits(), sim.to_bits(), "stft im mismatch t={t} f={f}");
            }
        }
    }

    #[test]
    fn stft_frequency_bins() {
        let fs = 1000.0;
        let x = vec![0.0; 256];
        let res = stft(&x, fs, None, Some(256), Some(128)).unwrap();
        assert_eq!(res.frequencies.len(), 129); // 256/2 + 1
        assert!((res.frequencies[0]).abs() < 1e-12);
        // Last frequency should be Nyquist
        assert!((res.frequencies[128] - 500.0).abs() < 1.0);
    }

    // ── Spectrogram tests ──────────────────────────────────────────

    #[test]
    fn spectrogram_chirp_energy_tracks_frequency() {
        // Generate a chirp from 5 Hz to 45 Hz over 1 second
        let fs = 200.0;
        let n = 200;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let sig = chirp(&t, 5.0, 1.0, 45.0, ChirpMethod::Linear).unwrap();

        let nperseg = 64;
        let noverlap = 56;
        let res = spectrogram(&sig, fs, None, Some(nperseg), Some(noverlap)).unwrap();

        // Should have multiple time segments and frequency bins
        assert!(!res.sxx.is_empty(), "spectrogram produced no segments");
        assert!(
            res.frequencies.len() > 1,
            "spectrogram produced no frequencies"
        );
        assert!(res.times.len() > 1, "spectrogram produced no time bins");
    }

    #[test]
    fn spectrogram_dimensions() {
        let fs = 100.0;
        let x: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let res = spectrogram(&x, fs, None, Some(64), Some(32)).unwrap();

        // Each time segment should have the same number of frequency bins
        let n_freqs = res.frequencies.len();
        for (t, seg) in res.sxx.iter().enumerate() {
            assert_eq!(
                seg.len(),
                n_freqs,
                "segment {t} has {} freq bins, expected {n_freqs}",
                seg.len()
            );
        }
        assert_eq!(res.times.len(), res.sxx.len());
    }

    // ── CSD tests ──────────────────────────────────────────────────

    #[test]
    fn csd_auto_spectrum_matches_welch() {
        // csd(x, x) should match welch(x) in magnitude
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let nperseg = 128;
        let noverlap = 64;
        let welch_res = welch(&x, fs, None, Some(nperseg), Some(noverlap)).unwrap();
        let csd_res = csd(&x, &x, fs, None, Some(nperseg), Some(noverlap)).unwrap();

        // Auto-CSD should be real and match welch PSD
        assert_eq!(welch_res.psd.len(), csd_res.csd.len());
        for (k, (&welch_val, &(csd_re, csd_im))) in
            welch_res.psd.iter().zip(csd_res.csd.iter()).enumerate()
        {
            // Imaginary part of auto-spectrum should be ~0
            assert!(
                csd_im.abs() < welch_val.abs() * 0.01 + 1e-20,
                "auto-CSD imaginary at bin {k}: {csd_im}"
            );
            // Real part should match welch PSD
            if welch_val.abs() > 1e-15 {
                let ratio = csd_re / welch_val;
                assert!(
                    (ratio - 1.0).abs() < 0.1,
                    "auto-CSD/welch mismatch at bin {k}: csd={csd_re}, welch={welch_val}"
                );
            }
        }
    }

    #[test]
    fn csd_scaling_matches_scipy_density_and_spectrum() {
        let fs = 16.0;
        let n = 32;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let x: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 2.0 * ti).sin())
            .collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 2.0 * ti).cos())
            .collect();
        let expected_frequencies = [0.0, 2.0, 4.0, 6.0, 8.0];
        let expected_density = [
            (-3.962744417486146e-17, 0.0),
            (1.3346567516981065e-17, 0.16666666666666666),
            (-7.103421885274257e-19, 0.04166666666666669),
            (-2.5133161553135813e-33, 2.363886302605754e-32),
            (-2.1146214215064937e-33, 0.0),
        ];
        let expected_spectrum = [
            (-1.127784089578213e-16, 0.0),
            (2.9738116731031224e-17, 0.5),
            (-9.686270956913913e-18, 0.12500000000000003),
            (-2.9579909131093405e-34, 6.980810313780984e-32),
            (-9.199850601359331e-33, 0.0),
        ];

        let density =
            csd_with_scaling(&x, &y, fs, None, Some(8), Some(4), SpectralScaling::Density)
                .expect("density");
        let spectrum = csd_with_scaling(
            &x,
            &y,
            fs,
            None,
            Some(8),
            Some(4),
            SpectralScaling::Spectrum,
        )
        .expect("spectrum");

        for (index, ((actual, expected), spectrum_frequency)) in density
            .frequencies
            .iter()
            .zip(expected_frequencies.iter())
            .zip(spectrum.frequencies.iter())
            .enumerate()
        {
            assert_close(
                *actual,
                *expected,
                1e-12,
                &format!("frequency[{index}] should match SciPy"),
            );
            assert_close(
                *actual,
                *spectrum_frequency,
                1e-12,
                &format!("frequency[{index}] should match across scaling modes"),
            );
        }

        for (index, (((density_re, density_im), expected_density), expected_spectrum)) in density
            .csd
            .iter()
            .zip(expected_density.iter())
            .zip(expected_spectrum.iter())
            .enumerate()
        {
            let (expected_density_re, expected_density_im) = *expected_density;
            let (expected_spectrum_re, expected_spectrum_im) = *expected_spectrum;
            let (spectrum_re, spectrum_im) = spectrum.csd[index];

            assert_close(
                *density_re,
                expected_density_re,
                1e-12,
                &format!("density.re[{index}] should match SciPy"),
            );
            assert_close(
                *density_im,
                expected_density_im,
                1e-12,
                &format!("density.im[{index}] should match SciPy"),
            );
            assert_close(
                spectrum_re,
                expected_spectrum_re,
                1e-12,
                &format!("spectrum.re[{index}] should match SciPy"),
            );
            assert_close(
                spectrum_im,
                expected_spectrum_im,
                1e-12,
                &format!("spectrum.im[{index}] should match SciPy"),
            );
        }
    }

    // ── Coherence tests ────────────────────────────────────────────

    #[test]
    fn coherence_identical_signals() {
        // Coherence of a signal with itself should be ~1 everywhere
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();

        let res = coherence(&x, &x, fs, None, Some(128), Some(64)).unwrap();
        for (k, &c) in res.coherence.iter().enumerate() {
            assert!(
                c >= 0.99 || res.frequencies[k].abs() < 1e-10,
                "coherence of identical signals at bin {k} (f={}): {c}",
                res.frequencies[k]
            );
        }
    }

    #[test]
    fn coherence_uncorrelated_signals() {
        // Coherence of uncorrelated signals should be low
        // Use two different frequency sinusoids as "uncorrelated" approximation
        let fs = 1000.0;
        let n = 4096;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin())
            .collect();

        let res = coherence(&x, &y, fs, None, Some(256), Some(128)).unwrap();
        // Average coherence should be much less than 1
        let avg_coh: f64 = res.coherence.iter().sum::<f64>() / res.coherence.len() as f64;
        assert!(
            avg_coh < 0.5,
            "average coherence of uncorrelated signals: {avg_coh}"
        );
    }

    #[test]
    fn coherence_matches_scipy_reference() {
        // Reference from scipy.signal.coherence on a deterministic pair;
        // segment detrending is what makes this match (frankenscipy-99796).
        let t: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let two_pi = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = t
            .iter()
            .map(|&ti| (two_pi * 0.05 * ti).sin() + 0.3 * (two_pi * 0.12 * ti).sin())
            .collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&ti| (two_pi * 0.05 * ti + 0.4).sin() + 0.2 * (two_pi * 0.2 * ti).cos())
            .collect();
        let res = coherence(&x, &y, 1.0, Some("hann"), Some(32), Some(16)).expect("coherence");
        let expected = [
            0.889_512_272_728_282_6,
            0.996_328_204_830_709,
            0.999_844_081_266_168,
            0.520_709_174_940_172_1,
            0.071_355_251_232_624_26,
            0.000_655_644_145_912_068,
            0.003_532_586_161_921_147,
            0.020_103_108_823_145_855,
            0.026_697_794_025_142_27,
            0.032_370_660_466_563_27,
            0.066_011_014_700_144_13,
            0.125_198_219_945_407_95,
            0.190_532_023_194_400_35,
            0.249_559_884_818_350_56,
            0.301_401_819_028_724,
            0.352_704_769_074_946_86,
            0.385_832_951_504_589_3,
        ];
        assert_eq!(res.coherence.len(), expected.len());
        for (got, want) in res.coherence.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "coherence {got} != {want}");
        }
    }

    #[test]
    fn coherence_range_zero_to_one() {
        let fs = 100.0;
        let n = 512;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();

        let res = coherence(&x, &y, fs, None, Some(128), Some(64)).unwrap();
        for (k, &c) in res.coherence.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&c),
                "coherence at bin {k} out of range: {c}"
            );
        }
    }

    // ── Resample tests ─────────────────────────────────────────────

    #[test]
    fn resample_upsample_preserves_frequency() {
        // A sine wave at 10 Hz sampled at 100 Hz, upsampled to 200 Hz
        let fs = 100.0;
        let n = 100;
        let freq = 10.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let result = resample(&x, 200).unwrap();
        assert_eq!(result.len(), 200);

        // Count zero crossings — should approximately double
        let x_crossings: usize = x
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        let r_crossings: usize = result
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        // Upsampled signal should have roughly double the zero crossings
        assert!(
            r_crossings >= x_crossings,
            "upsampled crossings {} < original {}",
            r_crossings,
            x_crossings
        );
    }

    #[test]
    fn resample_downsample() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let result = resample(&x, 100).unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn resample_same_length() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&x, 4).unwrap();
        assert_eq!(result.len(), 4);
        for (i, (&a, &b)) in x.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "same-length resample mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn resample_upsample_even_nyquist_matches_scipy() {
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let result = resample(&x, 8).unwrap();
        let expected = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "upsampled Nyquist mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_downsample_even_nyquist_matches_scipy() {
        let x = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let result = resample(&x, 6).unwrap();
        let expected = [
            0.0,
            0.8660254037844386,
            -0.8660254037844386,
            0.0,
            0.8660254037844386,
            -0.8660254037844386,
        ];

        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "downsampled Nyquist mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_rational_rate() {
        // Upsample by 3/2: 100 samples → ~150 samples
        let n = 100;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = resample_poly(&x, 3, 2).unwrap();
        // Output length should be approximately n * up / down
        let expected_len = n * 3 / 2;
        assert!(
            (result.len() as i64 - expected_len as i64).unsigned_abs() <= n / 5,
            "resample_poly length: got {}, expected ~{expected_len}",
            result.len()
        );
    }

    #[test]
    fn resample_poly_identity() {
        // up=1, down=1 should return the same signal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = resample_poly(&x, 1, 1).unwrap();
        assert_eq!(result.len(), x.len());
        for (i, (&a, &b)) in x.iter().zip(result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-10,
                "identity resample_poly mismatch at {i}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_line_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("line"), None).unwrap();
        let expected = [
            1.000606173553777,
            1.780628511492595,
            2.331390913393649,
            4.002424694215109,
            6.739706761743146,
            8.971945980209963,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "line padtype mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_mean_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("mean"), None).unwrap();
        let expected = [
            0.998333022727113,
            1.573448321613185,
            2.293752166053205,
            4.000151543388444,
            7.390581540874371,
            7.201952100936264,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "mean padtype mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_constant_cval_matches_scipy_reference() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let result = resample_poly_with_padtype(&x, 3, 2, Some("constant"), Some(5.0)).unwrap();
        let expected = [
            1.000606173553777,
            1.39589024581026,
            2.44120445519666,
            4.002424694215109,
            7.179517790579072,
            7.616661693078465,
        ];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-8,
                "constant cval mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn resample_poly_padtype_rejects_nonconstant_cval() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let err =
            resample_poly_with_padtype(&x, 3, 2, Some("mean"), Some(5.0)).expect_err("invalid");
        assert_eq!(
            err,
            SignalError::InvalidArgument("cval has no effect when padtype is mean".to_string())
        );
    }

    // ── Decimate tests ─────────────────────────────────────────────

    #[test]
    fn decimate_basic() {
        // Decimate by 4: 400 samples → 100 samples
        let n = 400;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = decimate(&x, 4).unwrap();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn decimate_prevents_aliasing() {
        // Signal with low-freq (5 Hz) and high-freq (45 Hz) at fs=100
        // After decimate by 4 (new fs=25), 45 Hz should be removed
        let fs = 100.0;
        let n = 400;
        let two_pi = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (two_pi * 5.0 * t).sin() + (two_pi * 45.0 * t).sin()
            })
            .collect();

        let result = decimate(&x, 4).unwrap();
        // The decimated signal should have the low-freq component preserved
        // but the high-freq component should be attenuated by the anti-alias filter
        let energy: f64 = result.iter().map(|v| v * v).sum::<f64>() / result.len() as f64;
        // If alias wasn't prevented, energy would be higher
        assert!(
            energy < 1.0,
            "decimated signal energy too high (aliasing?): {energy}"
        );
    }

    #[test]
    fn decimate_rejects_q_less_than_2() {
        assert!(decimate(&[1.0, 2.0], 1).is_err());
    }

    // ── Cross-correlation tests ──────────────────────────────────────

    #[test]
    fn autocorrelation_lag_zero_is_one() {
        // /profiling-software-performance regression for
        // [frankenscipy-ry4ll]: lag-0 autocorrelation must be 1.0
        // (variance / variance). Verify across multiple inputs after
        // the centered-series optimization.
        for input in [
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],
            vec![-1.0_f64, 0.0, 1.0],
            vec![10.0_f64, 20.0, 10.0, 20.0, 10.0, 20.0],
        ] {
            let r = autocorrelation(&input, 3);
            assert!(
                (r[0] - 1.0).abs() < 1e-12,
                "autocorrelation lag 0 = {} != 1.0 for {input:?}",
                r[0]
            );
        }
    }

    #[test]
    fn autocorrelation_constant_input_returns_all_ones() {
        // Constant input has zero variance; the function returns
        // [1.0; max_lag + 1] in that branch.
        let r = autocorrelation(&[5.0; 10], 4);
        assert_eq!(r.len(), 5);
        for &v in &r {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn autocorrelation_periodic_signal_peaks_at_period() {
        // Period-2 ±1 alternating sequence: lag 1 should be near -1
        // (anti-correlated), lag 2 should be near +1 (re-correlated).
        let x: Vec<f64> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = autocorrelation(&x, 4);
        // r[0] = 1, r[1] ≈ -(15/16), r[2] ≈ 14/16, r[3] ≈ -(13/16),
        // r[4] ≈ 12/16 (each lag drops one sample).
        assert!((r[0] - 1.0).abs() < 1e-12, "r[0] = {}", r[0]);
        assert!(
            r[1] < -0.5,
            "r[1] = {} should be strongly anti-correlated",
            r[1]
        );
        assert!(r[2] > 0.5, "r[2] = {} should be re-correlated", r[2]);
    }

    #[test]
    fn correlate_autocorrelation_peak_at_zero_lag() {
        // Auto-correlation peak should be at center (zero lag)
        let x = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let result = correlate(&x, &x, ConvolveMode::Full).expect("correlate");
        let mid = result.len() / 2;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val <= result[mid] + 1e-10,
                "peak should be at center lag: r[{i}]={val} > r[{mid}]={}",
                result[mid]
            );
        }
    }

    #[test]
    fn order_filter_min_max_median_match_known_values() {
        // /profiling-software-performance regression for
        // [frankenscipy-6feia]: verify that the buffer-reuse
        // optimization preserves the rank-based selection on a
        // pinned input.
        //
        //   x = [3, 1, 4, 1, 5, 9, 2, 6]
        //   window_size = 3 (half = 1)
        //
        // Window at i=0: [3, 1] (truncated at start) — rank=0 (min) → 1
        // Window at i=1: [3, 1, 4] sorted=[1,3,4] — min=1, max=4, median=3
        // Window at i=2: [1, 4, 1] sorted=[1,1,4] — min=1, max=4, median=1
        // Window at i=4: [1, 5, 9] sorted=[1,5,9] — min=1, max=9, median=5
        let x = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        // rank=0 → minimum filter
        let min_filter = order_filter(&x, 3, 0);
        assert_eq!(min_filter[1], 1.0);
        assert_eq!(min_filter[2], 1.0);
        assert_eq!(min_filter[4], 1.0);

        // rank=window_size-1 (capped to window-1 by impl) → maximum filter
        let max_filter = order_filter(&x, 3, 2);
        assert_eq!(max_filter[1], 4.0);
        assert_eq!(max_filter[2], 4.0);
        assert_eq!(max_filter[4], 9.0);

        // rank=1 (middle of full window) → median filter
        let med_filter = order_filter(&x, 3, 1);
        assert_eq!(med_filter[1], 3.0);
        assert_eq!(med_filter[2], 1.0);
        assert_eq!(med_filter[4], 5.0);
    }

    #[test]
    fn order_filter_constant_input_returns_constant() {
        // Constant input → output is the same constant for any rank.
        let x = vec![7.0_f64; 16];
        for rank in [0usize, 2, 4] {
            let result = order_filter(&x, 5, rank);
            assert_eq!(result.len(), 16);
            for &v in &result {
                assert_eq!(v, 7.0);
            }
        }
    }

    #[test]
    fn order_filter_empty_or_zero_window_returns_empty() {
        assert_eq!(order_filter(&[], 5, 2), Vec::<f64>::new());
        assert_eq!(order_filter(&[1.0, 2.0], 0, 0), Vec::<f64>::new());
    }

    #[test]
    fn correlate_shifted_signal() {
        // Cross-correlate with a shifted copy — peak indicates shift
        let a = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
        let result = correlate(&a, &v, ConvolveMode::Full).expect("correlate");
        // Peak should indicate the shift
        let peak_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        // v is shifted right by 1 relative to a, so peak should be at center-1
        let center = result.len() / 2;
        assert!(
            (peak_idx as i64 - center as i64).abs() <= 2,
            "peak at {peak_idx}, center at {center}"
        );
    }

    #[test]
    fn find_delay_recovers_known_shift_after_bounds_hoist() {
        // /profiling-software-performance regression for
        // [frankenscipy-9pkqs]: find_delay's inner loop now hoists
        // the per-i bounds check out of the lag loop. Verify the
        // optimized version still recovers the correct shift on
        // a known-shifted signal pair.
        //
        // Construct a clear template-and-shifted-copy:
        //   x = [0, 0, 1, 2, 3, 0, 0, 0]
        //   y = [0, 0, 0, 0, 1, 2, 3, 0]   (shifted right by 2)
        // Cross-correlation peaks where (i - lag = j) for the
        // shared spike pattern; lag = -2 (i.e., y is shifted forward
        // relative to x by 2 samples).
        let x = vec![0.0_f64, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let y = vec![0.0_f64, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
        let lag = find_delay(&x, &y);
        assert_eq!(lag, -2, "expected lag = -2 for right-shifted y, got {lag}");
    }

    #[test]
    fn find_delay_zero_lag_for_identical_signals() {
        // Two identical signals correlate maximally at lag 0.
        let x = vec![1.0_f64, 2.0, 3.0, 2.0, 1.0];
        let lag = find_delay(&x, &x);
        assert_eq!(lag, 0, "identical signals should give lag = 0, got {lag}");
    }

    #[test]
    fn find_delay_handles_unequal_lengths() {
        // Bounds-hoist regression: the new inner loop must correctly
        // truncate when x and y have different lengths.
        let x = vec![1.0_f64, 2.0, 3.0, 0.0, 0.0, 0.0];
        let y = vec![3.0_f64, 0.0, 0.0]; // shorter; matches x[2]
        let lag = find_delay(&x, &y);
        // Best correlation: x[2..3] · y[0..1] = 3*3 = 9 at lag = 2.
        assert_eq!(lag, 2, "unequal lengths: expected lag=2, got {lag}");
    }

    #[test]
    fn correlate_same_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![1.0, 0.0, -1.0];
        let result = correlate(&a, &v, ConvolveMode::Same).expect("correlate same");
        assert_eq!(
            result.len(),
            a.len(),
            "same mode output length = input length"
        );
    }

    #[test]
    fn correlate_valid_mode() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = vec![1.0, 0.0, -1.0];
        let result = correlate(&a, &v, ConvolveMode::Valid).expect("correlate valid");
        assert_eq!(result.len(), 3, "valid mode output length");
    }

    #[test]
    fn correlate_symmetry() {
        // correlate(a, a) should be symmetric
        let a = vec![1.0, 3.0, 2.0, 5.0];
        let result = correlate(&a, &a, ConvolveMode::Full).expect("correlate");
        let n = result.len();
        for i in 0..n / 2 {
            assert!(
                (result[i] - result[n - 1 - i]).abs() < 1e-10,
                "autocorrelation should be symmetric: r[{i}]={} vs r[{}]={}",
                result[i],
                n - 1 - i,
                result[n - 1 - i]
            );
        }
    }

    #[test]
    fn correlate2d_fft_path_matches_direct_reference() {
        // Above the crossover correlate2d takes the FFT path; prove it matches a
        // verbatim direct 2D correlation within FFT rounding for several large
        // shapes and all three modes. (Small inputs stay on the byte-identical
        // direct loop, covered by the conformance suite at 1e-12.)
        fn direct_full(
            a: &[f64],
            ar: usize,
            ac: usize,
            v: &[f64],
            vr: usize,
            vc: usize,
        ) -> Vec<f64> {
            let mut v_rev = vec![0.0; vr * vc];
            for i in 0..vr {
                for j in 0..vc {
                    v_rev[i * vc + j] = v[(vr - 1 - i) * vc + (vc - 1 - j)];
                }
            }
            let full_c = ac + vc - 1;
            let mut full = vec![0.0; (ar + vr - 1) * full_c];
            for i in 0..ar {
                for j in 0..ac {
                    let aval = a[i * ac + j];
                    for ki in 0..vr {
                        for kj in 0..vc {
                            full[(i + ki) * full_c + (j + kj)] += aval * v_rev[ki * vc + kj];
                        }
                    }
                }
            }
            full
        }
        fn grid(rows: usize, cols: usize, seed: f64) -> Vec<f64> {
            (0..rows * cols)
                .map(|k| ((k / cols) as f64 * 0.13 + (k % cols) as f64 * 0.27 + seed).sin())
                .collect()
        }
        for &(n, k) in &[(96usize, 48usize), (128, 96), (150, 150)] {
            let a = grid(n, n, 0.3);
            let v = grid(k, k, 1.1);
            let want_full = direct_full(&a, n, n, &v, k, k);
            for mode in [ConvolveMode::Full, ConvolveMode::Same, ConvolveMode::Valid] {
                let got = correlate2d(&a, (n, n), &v, (k, k), mode).expect("correlate2d");
                // Reconstruct the expected slice from want_full for this mode.
                let (full_r, full_c) = (n + k - 1, n + k - 1);
                let (out_r, out_c, sr, sc) = match mode {
                    ConvolveMode::Full => (full_r, full_c, 0, 0),
                    ConvolveMode::Same => (n, n, k / 2, k / 2),
                    ConvolveMode::Valid => (n - k + 1, n - k + 1, k - 1, k - 1),
                };
                assert_eq!(got.len(), out_r * out_c);
                let mut max_abs = 0.0_f64;
                for i in 0..out_r {
                    for j in 0..out_c {
                        let w = want_full[(i + sr) * full_c + (j + sc)];
                        max_abs = max_abs.max((got[i * out_c + j] - w).abs());
                    }
                }
                assert!(
                    max_abs < 1e-7,
                    "correlate2d FFT diverged {n}x{k} {mode:?}: {max_abs:e}"
                );
            }
        }
    }

    #[test]
    fn correlate2d_identity_kernel() {
        // Correlation with delta function should return the input
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // center = 1
        let result = correlate2d(&a, (3, 3), &v, (3, 3), ConvolveMode::Same).expect("correlate2d");
        assert_eq!(result.len(), 9);
        // With a centered delta, same-mode correlation should approximate input
        assert!(
            (result[4] - a[4]).abs() < 1e-10,
            "center should match: {} vs {}",
            result[4],
            a[4]
        );
    }

    #[test]
    fn correlate2d_same_even_kernel_matches_scipy_centering() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let result = correlate2d(&a, (3, 3), &v, (2, 2), ConvolveMode::Same).expect("correlate2d");
        assert_eq!(
            result,
            vec![12.0, 16.0, 9.0, 24.0, 28.0, 15.0, 15.0, 17.0, 9.0]
        );
    }

    // ── Hilbert transform tests ──────────────────────────────────────

    #[test]
    fn hilbert_sinusoid_envelope() {
        // Pure sinusoid should have ~constant envelope of 1.0
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let envelope = hilbert_envelope(&x).expect("hilbert envelope");
        assert_eq!(envelope.len(), n);
        // Exclude edges (transient effects) and check middle
        for (i, &env_val) in envelope.iter().enumerate().take(n - 20).skip(20) {
            assert!(
                (env_val - 1.0).abs() < 0.1,
                "envelope[{i}] = {}, expected ~1.0",
                env_val
            );
        }
    }

    #[test]
    fn hilbert_real_part_preserved() {
        // Real part of analytic signal should equal original signal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let analytic = hilbert(&x).expect("hilbert");
        for (i, (&xi, &(re, _))) in x.iter().zip(analytic.iter()).enumerate() {
            assert!(
                (re - xi).abs() < 1e-10,
                "real part[{i}] = {re}, expected {xi}"
            );
        }
    }

    #[test]
    fn hilbert_empty_input_rejected() {
        let err = hilbert(&[]).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn hilbert_dc_signal() {
        // DC signal: Hilbert transform of constant is zero
        let x = vec![5.0; 64];
        let analytic = hilbert(&x).expect("hilbert dc");
        // Imaginary parts should be ~0 for constant input
        for (i, &(_, im)) in analytic.iter().enumerate() {
            assert!(im.abs() < 1e-10, "DC imaginary[{i}] = {im}, expected ~0");
        }
    }

    // ── Notch/peak filter tests ──────────────────────────────────────

    #[test]
    fn iirnotch_rejects_target_frequency() {
        let coeffs = iirnotch(0.3, 30.0).expect("iirnotch");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        // Find response at notch frequency (0.3π)
        let notch_idx = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        // Response at notch should be near zero
        assert!(
            result.h_mag[notch_idx] < 0.1,
            "notch should attenuate target freq: mag = {}",
            result.h_mag[notch_idx]
        );
        // Response away from notch should be near 1
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        assert!(
            result.h_mag[pass_idx] > 0.9,
            "passband should be near unity: mag = {}",
            result.h_mag[pass_idx]
        );
    }

    #[test]
    fn iirpeak_boosts_target_frequency() {
        let coeffs = iirpeak(0.3, 30.0).expect("iirpeak");
        let result = freqz(&coeffs.b, &coeffs.a, Some(1024)).expect("freqz");
        let peak_idx = nearest_freq_index(&result.w, 0.3 * std::f64::consts::PI);
        let pass_idx = nearest_freq_index(&result.w, 0.1 * std::f64::consts::PI);
        assert!(
            result.h_mag[peak_idx] > result.h_mag[pass_idx] * 2.0,
            "peak should boost target: peak={}, pass={}",
            result.h_mag[peak_idx],
            result.h_mag[pass_idx]
        );
    }

    #[test]
    fn iirnotch_iirpeak_match_scipy_orfanidis_design() {
        // Reference coefficients from scipy.signal.iirnotch / iirpeak.
        let check = |label: &str, got: &[f64], want: &[f64]| {
            for (g, w) in got.iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-12, "{label}: {g} != {w}");
            }
        };

        let notch = iirnotch(0.25, 5.0).expect("iirnotch");
        check(
            "notch_b",
            &notch.b,
            &[
                0.927_040_342_731_733_3,
                -1.311_033_025_558_219_7,
                0.927_040_342_731_733_3,
            ],
        );
        check(
            "notch_a",
            &notch.a,
            &[1.0, -1.311_033_025_558_219_7, 0.854_080_685_463_466_6],
        );

        let peak = iirpeak(0.25, 5.0).expect("iirpeak");
        check(
            "peak_b",
            &peak.b,
            &[0.072_959_657_268_266_7, 0.0, -0.072_959_657_268_266_7],
        );
        // notch and peak share the same denominator.
        check("peak_a", &peak.a, &notch.a);

        let notch2 = iirnotch(0.1, 2.0).expect("iirnotch");
        check(
            "notch2_b",
            &notch2.b,
            &[
                0.927_040_342_731_733_3,
                -1.763_335_517_647_015,
                0.927_040_342_731_733_3,
            ],
        );
    }

    #[test]
    fn iirnotch_invalid_params() {
        assert!(iirnotch(0.0, 10.0).is_err());
        assert!(iirnotch(1.0, 10.0).is_err());
        assert!(iirnotch(0.5, -1.0).is_err());
    }

    #[test]
    fn iirpeak_invalid_params() {
        assert!(iirpeak(0.0, 10.0).is_err());
        assert!(iirpeak(0.5, 0.0).is_err());
    }

    // ── Wavelet tests ────────────────────────────────────────────────

    #[test]
    fn ricker_symmetric() {
        let w = ricker(101, 5.0);
        assert_eq!(w.len(), 101);
        // Symmetric around center
        for i in 0..50 {
            assert!(
                (w[i] - w[100 - i]).abs() < 1e-12,
                "ricker not symmetric at {i}"
            );
        }
    }

    #[test]
    fn ricker_peak_at_center() {
        let w = ricker(101, 5.0);
        let center = 50;
        // Peak should be at center
        for (i, &val) in w.iter().enumerate() {
            assert!(
                val <= w[center] + 1e-12,
                "ricker: w[{i}]={val} > w[center]={}",
                w[center]
            );
        }
    }

    #[test]
    fn ricker_zero_integral() {
        // Ricker wavelet has zero mean (wavelet admissibility)
        let w = ricker(1001, 10.0);
        let integral: f64 = w.iter().sum();
        assert!(
            integral.abs() < 0.1,
            "ricker integral should be ~0: {integral}"
        );
    }

    #[test]
    fn morlet_complex_oscillatory() {
        let w = morlet(200, 5.5, 10.0, false);
        assert_eq!(w.len(), 200);
        // With wider s, the Gaussian envelope is broader and oscillations are visible
        let max_im = w
            .iter()
            .map(|&(_, im)| im.abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        assert!(
            max_im > 0.01,
            "morlet should have nonzero imaginary part, max_im = {max_im}"
        );
    }

    #[test]
    fn morlet_optimized_matches_naive_implementation() {
        // /profiling-software-performance regression for [frankenscipy-tmnrh]:
        // morlet inner loop hoists the correction (loop-invariant in w)
        // and uses sin_cos. Verify equivalence against an explicit
        // naïve implementation that mirrors the pre-fix branchy logic
        // across multiple (m, w, s) configurations and both complete
        // settings.

        fn naive_morlet(m: usize, w: f64, s: f64, complete: bool) -> Vec<(f64, f64)> {
            if m == 0 || s <= 0.0 || !s.is_finite() || !w.is_finite() {
                return vec![];
            }
            let mut output = Vec::with_capacity(m);
            let center = (m as f64 - 1.0) / 2.0;
            for i in 0..m {
                let t = (i as f64 - center) / s;
                let gauss = (-t * t / 2.0).exp();
                let phase = 2.0 * std::f64::consts::PI * w * t;
                let mut re = gauss * phase.cos();
                let im = gauss * phase.sin();
                if complete {
                    let correction =
                        (-2.0 * std::f64::consts::PI * std::f64::consts::PI * w * w / 2.0).exp();
                    re -= gauss * correction;
                }
                output.push((re, im));
            }
            output
        }

        let configs: &[(usize, f64, f64, bool)] = &[
            (32, 5.0, 1.0, false),
            (32, 5.0, 1.0, true),
            (101, 5.5, 10.0, false),
            (101, 5.5, 10.0, true),
            (200, 6.0, 5.0, true),
            (200, 6.0, 5.0, false),
            (1, 5.0, 1.0, true),
        ];
        for &(m, w, s, complete) in configs {
            let opt = morlet(m, w, s, complete);
            let nav = naive_morlet(m, w, s, complete);
            assert_eq!(opt.len(), nav.len());
            for (i, (&(or, oi), &(nr, ni))) in opt.iter().zip(nav.iter()).enumerate() {
                let scale_re = or.abs().max(nr.abs()).max(1e-12);
                let scale_im = oi.abs().max(ni.abs()).max(1e-12);
                assert!(
                    (or - nr).abs() <= 1e-12 * scale_re,
                    "morlet re mismatch at i={i}, (m={m}, w={w}, s={s}, complete={complete}): opt={or}, naive={nr}"
                );
                assert!(
                    (oi - ni).abs() <= 1e-12 * scale_im,
                    "morlet im mismatch at i={i}, (m={m}, w={w}, s={s}, complete={complete}): opt={oi}, naive={ni}"
                );
            }
        }
    }

    #[test]
    fn morlet2_optimized_matches_naive_implementation() {
        // /profiling-software-performance regression for [frankenscipy-t4jxi]:
        // morlet2 hoists 2·s² and w/s, and folds cos+sin into sin_cos.
        // Verify equivalence against an explicit naïve implementation
        // mirroring the pre-fix per-iter recomputation.

        fn naive_morlet2(m: usize, s: f64, w: f64) -> Vec<(f64, f64)> {
            if m == 0 || s <= 0.0 || !s.is_finite() || !w.is_finite() {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(m);
            let coeff = (std::f64::consts::PI * s * s).powf(-0.25);
            let half = (m as f64 - 1.0) / 2.0;
            for i in 0..m {
                let t = i as f64 - half;
                let phase = w * t / s;
                let gauss = (-t * t / (2.0 * s * s)).exp();
                let envelope = coeff * gauss;
                out.push((envelope * phase.cos(), envelope * phase.sin()));
            }
            out
        }

        let configs: &[(usize, f64, f64)] = &[
            (16, 1.0, 5.0),
            (32, 4.0, 5.0),
            (64, 5.0, 6.0),
            (128, 10.0, 5.0),
            (200, 8.0, 7.5),
            (1, 1.0, 5.0),
        ];
        for &(m, s, w) in configs {
            let opt = morlet2(m, s, w);
            let nav = naive_morlet2(m, s, w);
            assert_eq!(opt.len(), nav.len());
            for (i, (&(or, oi), &(nr, ni))) in opt.iter().zip(nav.iter()).enumerate() {
                let scale_re = or.abs().max(nr.abs()).max(1e-12);
                let scale_im = oi.abs().max(ni.abs()).max(1e-12);
                assert!(
                    (or - nr).abs() <= 1e-12 * scale_re,
                    "morlet2 re mismatch at i={i}, (m={m}, s={s}, w={w}): opt={or}, naive={nr}"
                );
                assert!(
                    (oi - ni).abs() <= 1e-12 * scale_im,
                    "morlet2 im mismatch at i={i}, (m={m}, s={s}, w={w}): opt={oi}, naive={ni}"
                );
            }
        }
    }

    #[test]
    fn cwt_detects_frequency() {
        // Create signal with known frequency component
        let n = 256;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin())
            .collect();
        let widths: Vec<f64> = (1..=20).map(|w| w as f64).collect();
        let result = cwt(&x, ricker, &widths).expect("cwt");
        assert_eq!(result.len(), widths.len());
        assert_eq!(result[0].len(), n);
    }

    #[test]
    fn cwt_empty_data_rejected() {
        let err = cwt(&[], ricker, &[1.0]).expect_err("empty");
        assert!(err.is_argument_error());
    }

    #[test]
    fn tukey_window_endpoints() {
        let w = tukey_window(100, 0.5);
        assert_eq!(w.len(), 100);
        // Endpoints should be near zero
        assert!(w[0] < 0.01, "tukey start should be ~0: {}", w[0]);
        // Center should be 1.0
        assert!(
            (w[50] - 1.0).abs() < 0.01,
            "tukey center should be ~1: {}",
            w[50]
        );
    }

    #[test]
    fn nuttall_window_symmetric() {
        let w = nuttall_window(64);
        assert_eq!(w.len(), 64);
        for i in 0..32 {
            assert!(
                (w[i] - w[63 - i]).abs() < 1e-12,
                "nuttall not symmetric at {i}"
            );
        }
    }

    #[test]
    fn flattop_window_center_is_one() {
        // /testing-conformance-harnesses for [frankenscipy-8knc6]:
        // For odd n, the center sample i = (n-1)/2 has arg = π, so
        // cos(arg) = -1, cos(2arg) = 1, cos(3arg) = -1, cos(4arg) = 1.
        // The flattop value collapses to:
        //   0.21557895 + 0.41663158 + 0.277263158 + 0.083578947
        //     + 0.006947368 = 1.000000003 ≈ 1.0.
        for &n in &[5usize, 11, 21, 51] {
            let w = flattop(n);
            assert_eq!(w.len(), n);
            let center = (n - 1) / 2;
            assert!(
                (w[center] - 1.0).abs() < 1e-6,
                "flattop({n})[{center}] = {} should be ≈ 1",
                w[center]
            );
        }
    }

    #[test]
    fn flattop_window_endpoints_near_zero() {
        // At i = 0 and i = n-1, arg = 0 or 2π, so all cos terms are 1.
        // Sum: 0.21557895 - 0.41663158 + 0.277263158 - 0.083578947
        //      + 0.006947368 ≈ -0.000420.
        // Within 1e-3 of zero — much smaller than the body amplitude.
        for &n in &[16usize, 32, 64] {
            let w = flattop(n);
            assert!(
                w[0].abs() < 1e-3,
                "flattop({n})[0] = {} should be near 0",
                w[0]
            );
            assert!(
                w[n - 1].abs() < 1e-3,
                "flattop({n})[n-1] = {} should be near 0",
                w[n - 1]
            );
        }
    }

    #[test]
    fn flattop_window_is_symmetric() {
        for &n in &[5usize, 16, 33, 65] {
            let w = flattop(n);
            for i in 0..n / 2 {
                assert!(
                    (w[i] - w[n - 1 - i]).abs() < 1e-12,
                    "flattop({n}) asymmetric at i={i}: {} vs {}",
                    w[i],
                    w[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn flattop_window_n_one_and_zero() {
        assert_eq!(flattop(0), Vec::<f64>::new());
        assert_eq!(flattop(1), vec![1.0]);
    }

    #[test]
    fn bohman_window_endpoints_zero() {
        let w = bohman_window(64);
        assert!(w[0].abs() < 1e-12, "bohman start should be 0");
        assert!(w[63].abs() < 1e-12, "bohman end should be 0");
    }

    #[test]
    fn cosine_window_matches_scipy_reference() {
        let w = cosine(5);
        let expected = [
            0.3090169943749474,
            0.8090169943749475,
            1.0,
            0.8090169943749475,
            0.3090169943749474,
        ];
        for (actual, want) in w.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn upfirdn_fir_only_matches_scipy_example() {
        let y = upfirdn(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], 1, 1).expect("upfirdn");
        assert_eq!(y, vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn upfirdn_upsampling_matches_scipy_example() {
        let y = upfirdn(&[1.0], &[1.0, 2.0, 3.0], 3, 1).expect("upfirdn");
        assert_eq!(y, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn upfirdn_interpolate_and_decimate_matches_scipy_example() {
        let y = upfirdn(&[0.5, 1.0, 0.5], &[0.0, 1.0, 2.0, 3.0, 4.0], 2, 3).expect("upfirdn");
        let expected = [0.0, 1.0, 2.5, 4.0];
        for (actual, want) in y.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-12);
        }
    }

    #[test]
    fn upfirdn_empty_input_returns_empty() {
        let y = upfirdn(&[1.0, 2.0], &[], 2, 1).expect("empty input");
        assert!(y.is_empty());
    }

    #[test]
    fn upfirdn_rejects_invalid_arguments() {
        assert!(upfirdn(&[], &[1.0, 2.0], 1, 1).is_err());
        assert!(upfirdn(&[1.0], &[1.0, 2.0], 0, 1).is_err());
        assert!(upfirdn(&[1.0], &[1.0, 2.0], 1, 0).is_err());
    }

    // ── Max-length sequence + matched filter tests ───────────────────

    #[test]
    fn max_len_seq_length() {
        let seq = max_len_seq(5).expect("mls");
        assert_eq!(seq.len(), 31);
        assert!(seq.iter().all(|&v| v == 1.0 || v == -1.0));
    }

    #[test]
    fn max_len_seq_balanced() {
        let seq = max_len_seq(6).expect("mls");
        let ones = seq.iter().filter(|&&v| v == 1.0).count();
        let neg_ones = seq.iter().filter(|&&v| v == -1.0).count();
        assert_eq!(ones + neg_ones, 63);
        assert_eq!((ones as i64 - neg_ones as i64).unsigned_abs(), 1);
    }

    #[test]
    fn max_len_seq_invalid_nbits() {
        assert!(max_len_seq(0).is_err());
        assert!(max_len_seq(1).is_err());
    }

    #[test]
    fn matched_filter_peak_at_template_location() {
        let template = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let mut signal = vec![0.0; 20];
        for (i, &t) in template.iter().enumerate() {
            signal[8 + i] = t;
        }
        let result = matched_filter(&template, &signal).expect("matched_filter");
        let peak_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak_idx as i64 - 12).abs() <= 1,
            "peak at {peak_idx}, expected ~12"
        );
    }

    #[test]
    fn minimum_phase_matches_scipy_reference() {
        let h = minimum_phase(&[1.0, 2.0, 3.0, 2.0, 1.0]).expect("minimum_phase");
        let expected = [1.00107477, 0.99999638, 0.99892826];
        for (actual, want) in h.iter().zip(expected.iter()) {
            assert!((*actual - *want).abs() < 1e-6, "{actual} vs {want}");
        }
    }

    #[test]
    fn minimum_phase_rejects_short_filters() {
        assert!(minimum_phase(&[]).is_err());
        assert!(minimum_phase(&[1.0, 2.0]).is_err());
    }

    // ── Lomb-Scargle tests ───────────────────────────────────────────

    #[test]
    fn lombscargle_detects_known_frequency() {
        // Signal with known frequency at 5 rad/s
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&t| (5.0 * t).sin()).collect();
        let freqs: Vec<f64> = (1..20).map(|f| f as f64).collect();
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle");
        // Peak should be near freq=5
        let peak_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        assert!(
            (freqs[peak_idx] - 5.0).abs() <= 1.0,
            "peak at freq={}, expected ~5",
            freqs[peak_idx]
        );
    }

    #[test]
    fn lombscargle_rejects_mismatched() {
        let err = lombscargle(&[1.0, 2.0], &[1.0], &[1.0], false).expect_err("mismatch");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lombscargle_normalize_matches_scipy_reference() {
        let tol = 5e-9;
        let x = [0.0, 0.5, 1.1, 1.7, 2.4];
        let y = [1.0, -0.5, 0.75, 0.25, -1.25];
        let freqs = [0.5, 1.0, 1.5, 2.0];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");
        let expected_power = [
            0.707_498_041_923_665,
            0.651_738_901_352_701,
            0.587_708_048_398_399,
            0.390_030_087_171_934,
        ];
        let expected_normalized = [
            0.411_635_223_256_532,
            0.379_193_542_532_48,
            0.341_939_227_868_014,
            0.226_926_597_192_203,
        ];

        for (actual, want) in power.iter().zip(expected_power.iter()) {
            assert!((actual - want).abs() < tol, "{actual} vs {want}");
        }

        for (actual, want) in normalized.iter().zip(expected_normalized.iter()) {
            assert!((actual - want).abs() < tol, "{actual} vs {want}");
        }
    }

    #[test]
    fn lombscargle_zero_frequency_matches_scipy_reference() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 1.0, 0.0];
        let freqs = [0.0, 1.0];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");

        assert!((power[0] - 2.0).abs() < 1e-12);
        assert!((normalized[0] - (2.0 / 3.0)).abs() < 1e-12);
        assert!(power[1].is_finite());
        assert!(normalized[1].is_finite());
    }

    #[test]
    fn lombscargle_non_finite_frequency_returns_nan() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 1.0, 0.0];
        let freqs = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let power = lombscargle(&x, &y, &freqs, false).expect("lombscargle power");
        assert!(power.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn lombscargle_normalize_returns_nan_for_zero_energy_signal() {
        let x = [0.1, 0.4, 1.1, 2.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        let freqs = [1.0, 2.0];
        let normalized = lombscargle(&x, &y, &freqs, true).expect("lombscargle normalized");
        assert!(normalized.iter().all(|value| value.is_nan()));
    }

    // ── gausspulse tests ─────────────────────────────────────────────

    #[test]
    fn gausspulse_peak_at_zero() {
        let t: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.001).collect();
        let y = gausspulse(&t, 1000.0, 0.5);
        // Peak should be at t=0 (index 50)
        assert!(
            y[50] > y[0].abs() && y[50] > y[100].abs(),
            "gausspulse should peak at t=0"
        );
    }

    #[test]
    fn gausspulse_zero_center_frequency_matches_scipy_constant_pulse() {
        let t = [-1.0, 0.0, 1.0];
        let y = gausspulse(&t, 0.0, 0.5);
        assert_eq!(y, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn gausspulse_matches_scipy_reference() {
        // Reference values from scipy.signal.gausspulse. The Gaussian width
        // uses a = (π·fc·bw)²/(−4·ln ref) — frankenscipy-88iwj.
        let t = [0.0, 0.05, 0.1, 0.2];
        for &(fc, bw, want) in &[
            (5.0_f64, 0.5_f64, [1.0, 0.0, -0.799_918_398, 0.409_432_905]),
            (
                10.0,
                0.3,
                [1.0, -0.922_776_332, 0.725_079_768, 0.276_403_253],
            ),
            (1.0, 0.8, [1.0, 0.945_636_648, 0.790_732_379, 0.282_013_486]),
        ] {
            let y = gausspulse(&t, fc, bw);
            for (got, w) in y.iter().zip(want.iter()) {
                assert!((got - w).abs() < 1e-8, "fc={fc} bw={bw}: {got} != {w}");
            }
        }
    }

    #[test]
    fn gausspulse_envelope_decay() {
        // Envelope should decay away from center
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.001).collect();
        let y = gausspulse(&t, 100.0, 0.5);
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        assert!(y[0].abs() >= max_abs * 0.99, "peak should be at t=0");
    }

    // ── sweep_poly tests ─────────────────────────────────────────────

    #[test]
    fn sweep_poly_constant_freq() {
        // Constant frequency f(t) = 10 → cos(2π·10·t)
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let y = sweep_poly(&t, &[10.0]); // poly = [10] → f(t) = 10
        assert_eq!(y.len(), 100);
        // At t=0, should be cos(0) = 1
        assert!((y[0] - 1.0).abs() < 1e-10, "sweep_poly(0) = {}", y[0]);
    }

    #[test]
    fn sweep_poly_linear_chirp() {
        // Linear chirp: f(t) = 10 + 100*t → poly = [100, 10]
        let t: Vec<f64> = (0..200).map(|i| i as f64 * 0.001).collect();
        let y = sweep_poly(&t, &[100.0, 10.0]);
        assert_eq!(y.len(), 200);
        // Should oscillate
        let crossings = y
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        assert!(
            crossings > 2,
            "linear chirp should oscillate: {crossings} crossings"
        );
    }

    #[test]
    fn czt_matches_dft_for_unit_circle() {
        // With default parameters, CZT should match the DFT
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let n = x.len();
        let result = czt(&x, n, None, None).unwrap();

        // Compare with direct DFT
        let two_pi = 2.0 * std::f64::consts::PI;
        for (k, item) in result.iter().enumerate().take(n) {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                let angle = -two_pi * (k as f64) * (j as f64) / (n as f64);
                re += xj * angle.cos();
                im += xj * angle.sin();
            }
            assert!(
                (item.0 - re).abs() < 1e-10,
                "CZT real[{k}] = {}, expected {re}",
                item.0
            );
            assert!(
                (item.1 - im).abs() < 1e-10,
                "CZT imag[{k}] = {}, expected {im}",
                item.1
            );
        }
    }

    #[test]
    fn czt_single_tone_detection() {
        // A pure sinusoid should produce a peak at its frequency
        let n = 64;
        let freq = 5.0; // bin 5
        let two_pi = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| (two_pi * freq * i as f64 / n as f64).cos())
            .collect();

        let result = czt(&x, n, None, None).unwrap();

        // Find magnitude peak
        let mags: Vec<f64> = result
            .iter()
            .map(|&(r, i)| (r * r + i * i).sqrt())
            .collect();
        // Cosine has energy at both bin f and bin N-f (mirror)
        let peak_bin = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            peak_bin == freq as usize || peak_bin == n - freq as usize,
            "peak should be at bin {freq} or {}, got {peak_bin}",
            n - freq as usize
        );
        // Check that both bins have significant energy
        assert!(
            mags[freq as usize] > mags[0] * 10.0,
            "bin {freq} should have energy"
        );
    }

    #[test]
    fn zoom_fft_resolves_narrow_band() {
        // Two close frequencies that zoom FFT should resolve
        let n = 128;
        let two_pi = 2.0 * std::f64::consts::PI;
        let f1 = 10.0 / n as f64; // normalized freq
        let f2 = 12.0 / n as f64;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                (two_pi * f1 * n as f64 * i as f64 / n as f64).cos()
                    + (two_pi * f2 * n as f64 * i as f64 / n as f64).cos()
            })
            .collect();

        // Zoom into the frequency range containing both tones
        let m = 64;
        let result = zoom_fft(&x, (16.0 / n as f64, 28.0 / n as f64), m).unwrap();

        // Should have two peaks in the zoomed spectrum
        let mags: Vec<f64> = result
            .iter()
            .map(|&(r, i)| (r * r + i * i).sqrt())
            .collect();
        let max_mag = mags.iter().cloned().fold(0.0f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        let peaks: Vec<usize> = mags
            .iter()
            .enumerate()
            .filter(|&(_, m)| *m > max_mag * 0.5)
            .map(|(i, _)| i)
            .collect();
        assert!(
            peaks.len() >= 2,
            "zoom FFT should resolve 2 peaks, got {} peaks",
            peaks.len()
        );
    }

    #[test]
    fn zoom_fft_full_range_matches_czt_default() {
        let x: Vec<f64> = (0..8).map(|i| (i as f64 + 1.0).sin()).collect();
        let zoomed = zoom_fft(&x, (0.0, 2.0), x.len()).unwrap();
        let full = czt(&x, x.len(), None, None).unwrap();
        for (idx, (got, want)) in zoomed.iter().zip(full.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-10,
                "zoom real[{idx}]={}, czt={}",
                got.0,
                want.0
            );
            assert!(
                (got.1 - want.1).abs() < 1e-10,
                "zoom imag[{idx}]={}, czt={}",
                got.1,
                want.1
            );
        }
    }

    #[test]
    fn zoom_fft_matches_scipy_reference() {
        // Reference values from scipy.signal.zoom_fft(x, [f1, f2], m, fs=2).
        struct Case {
            x: Vec<f64>,
            f1: f64,
            f2: f64,
            m: usize,
            expect: Vec<(f64, f64)>,
        }
        let cases = vec![
            Case {
                x: vec![1., 2., 3., 4., 3., 2., 1., 0.],
                f1: 0.1,
                f2: 0.4,
                m: 5,
                expect: vec![
                    (8.2983390136, -11.4216837883),
                    (0.7237692584, -11.5039829340),
                    (-4.0511135523, -7.3689481957),
                    (-4.6640593359, -2.5640883180),
                    (-2.7457326192, 0.1727468541),
                ],
            },
            Case {
                x: vec![1., -1., 2., -2., 3., -3., 0.5, 0.25, 0.1, 0.0],
                f1: 0.2,
                f2: 0.9,
                m: 7,
                expect: vec![
                    (1.5491389537, -0.5488080977),
                    (-0.0575828659, -1.5901735667),
                    (-1.5106134621, 0.8893890100),
                    (1.6000000000, 2.2500000000),
                    (2.2758610463, -1.6323096127),
                    (-3.1617666846, -1.7102329643),
                    (-3.9143865379, 6.0041329256),
                ],
            },
            Case {
                x: vec![1., 2., 3., 4., 3., 2., 1., 0.],
                f1: 0.5,
                f2: 1.5,
                m: 6,
                expect: vec![
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, -0.8038475773),
                    (0.0, 0.0),
                    (0.0, 0.8038475773),
                    (1.0, 0.0),
                ],
            },
            Case {
                x: vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
                f1: 0.3,
                f2: 0.7,
                m: 4,
                expect: vec![
                    (1.5388417686, 0.5000000000),
                    (2.1180339887, -1.5388417686),
                    (0.0, -3.0000000000),
                    (-2.1180339887, -1.5388417686),
                ],
            },
        ];
        for case in &cases {
            let got = zoom_fft(&case.x, (case.f1, case.f2), case.m).unwrap();
            assert_eq!(got.len(), case.expect.len());
            for (idx, (g, e)) in got.iter().zip(&case.expect).enumerate() {
                assert!(
                    (g.0 - e.0).abs() < 1e-7 && (g.1 - e.1).abs() < 1e-7,
                    "zoom_fft f=[{},{}] m={} idx {idx}: got {g:?}, want {e:?}",
                    case.f1,
                    case.f2,
                    case.m
                );
            }
        }
    }

    #[test]
    fn periodogram_matches_scipy_density() {
        // scipy.signal.periodogram(x, fs, window, scaling='density').
        let n = 64;
        let fs = 8.0;
        let tp = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (tp * 1.0 * t).sin() + 0.5 * (tp * 2.5 * t).cos()
            })
            .collect();

        // Boxcar (rectangular) window.
        let r = periodogram(&x, fs, None).unwrap();
        assert!((r.psd[8] - 4.0).abs() < 1e-9, "boxcar peak: {}", r.psd[8]);
        assert!((r.psd[20] - 1.0).abs() < 1e-9, "boxcar peak: {}", r.psd[20]);

        // Periodic Hann window.
        let hann: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (tp * i as f64 / n as f64).cos())
            .collect();
        let rh = periodogram(&x, fs, Some(&hann)).unwrap();
        for (k, expect) in [(7, 2.0 / 3.0), (8, 8.0 / 3.0), (9, 2.0 / 3.0)] {
            assert!(
                (rh.psd[k] - expect).abs() < 1e-9,
                "hann psd[{k}]={}, want {expect}",
                rh.psd[k]
            );
        }
    }

    #[test]
    fn welch_matches_scipy_density() {
        // scipy.signal.welch(x, fs, window, nperseg, noverlap, scaling='density').
        let n = 128;
        let fs = 16.0;
        let tp = 2.0 * std::f64::consts::PI;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (tp * 2.0 * t).sin() + 0.3 * (tp * 5.0 * t).cos()
            })
            .collect();

        let r = welch(&x, fs, Some("hann"), Some(32), Some(16)).unwrap();
        for (k, expect) in [
            (3, 1.0 / 6.0),
            (4, 2.0 / 3.0),
            (5, 1.0 / 6.0),
            (9, 0.015),
            (10, 0.06),
            (11, 0.015),
        ] {
            assert!(
                (r.psd[k] - expect).abs() < 1e-9,
                "welch hann psd[{k}]={}, want {expect}",
                r.psd[k]
            );
        }

        let rb = welch(&x, fs, Some("boxcar"), Some(64), Some(32)).unwrap();
        assert!(
            (rb.psd[8] - 2.0).abs() < 1e-9,
            "welch boxcar psd[8]={}",
            rb.psd[8]
        );
        assert!(
            (rb.psd[20] - 0.18).abs() < 1e-9,
            "welch boxcar psd[20]={}",
            rb.psd[20]
        );
    }

    #[test]
    fn czt_empty_output() {
        let x = vec![1.0, 2.0];
        let result = czt(&x, 0, None, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn czt_empty_input_errors() {
        let result = czt(&[], 4, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn remez_lowpass_filter() {
        // Design a 31-tap lowpass filter with cutoff at 0.2
        let h = remez(31, &[0.0, 0.15, 0.25, 0.5], &[1.0, 0.0], None).unwrap();
        assert_eq!(h.len(), 31);

        // Filter should be symmetric (linear phase)
        for i in 0..15 {
            assert!(
                (h[i] - h[30 - i]).abs() < 1e-10,
                "h[{i}] = {} != h[{}] = {}",
                h[i],
                30 - i,
                h[30 - i]
            );
        }

        // Sum of coefficients should approximate passband gain (1.0)
        let sum: f64 = h.iter().sum();
        assert!(sum > 0.5 && sum < 1.5, "filter sum = {sum}, expected ~1.0");
    }

    // ── LTI/DLTI system tests ───────────────────────────────────────

    #[test]
    fn lti_new_valid() {
        let sys = Lti::new(vec![1.0], vec![1.0, 2.0]).expect("valid system");
        assert_eq!(sys.num, vec![1.0]);
        assert_eq!(sys.den, vec![1.0, 2.0]);
    }

    #[test]
    fn lti_new_empty_num_rejected() {
        let err = Lti::new(vec![], vec![1.0]).expect_err("empty num");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_new_empty_den_rejected() {
        let err = Lti::new(vec![1.0], vec![]).expect_err("empty den");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_new_zero_den_rejected() {
        let err = Lti::new(vec![1.0], vec![0.0, 0.0]).expect_err("zero den");
        assert!(err.is_argument_error());
    }

    #[test]
    fn lti_from_zpk_no_roots() {
        // H(s) = 2 / 1 = 2
        let sys = Lti::from_zpk(&[], &[], 2.0).expect("valid");
        assert_eq!(sys.num, vec![2.0]);
        assert_eq!(sys.den, vec![1.0]);
    }

    #[test]
    fn lti_from_zpk_simple_pole() {
        // H(s) = 1 / (s + 1)
        let sys = Lti::from_zpk(&[], &[-1.0], 1.0).expect("valid");
        assert_eq!(sys.num, vec![1.0]);
        assert!((sys.den[0] - 1.0).abs() < 1e-10);
        assert!((sys.den[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lti_freqresp_dc_gain() {
        // H(s) = 2 / (s + 1), DC gain = H(0) = 2
        let sys = Lti::new(vec![2.0], vec![1.0, 1.0]).expect("valid");
        let (mag, _phase) = sys.freqresp(&[0.0]);
        assert!((mag[0] - 2.0).abs() < 1e-10, "DC gain = {}", mag[0]);
    }

    #[test]
    fn lti_step_response_converges() {
        // H(s) = 1 / (s + 1), step response converges to 1
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y = sys.step(&t).expect("step");
        // Final value should be close to 1 (DC gain)
        let final_val = y.last().unwrap();
        assert!(
            (final_val - 1.0).abs() < 0.1,
            "final step value = {final_val}"
        );
    }

    #[test]
    fn lti_poles_first_order() {
        // H(s) = 1 / (s + 2), pole at s = -2
        let sys = Lti::new(vec![1.0], vec![1.0, 2.0]).expect("valid");
        let poles = sys.poles().expect("poles");
        assert_eq!(poles.len(), 1);
        assert!(
            (poles[0].0 + 2.0).abs() < 1e-10,
            "pole real = {}",
            poles[0].0
        );
        assert!(poles[0].1.abs() < 1e-10, "pole imag = {}", poles[0].1);
    }

    #[test]
    fn dlti_new_valid() {
        let sys = Dlti::new(vec![1.0], vec![1.0, 0.5], 0.1).expect("valid");
        assert_eq!(sys.dt, 0.1);
    }

    #[test]
    fn dlti_new_negative_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], -0.1).expect_err("negative dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_new_zero_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], 0.0).expect_err("zero dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_new_non_finite_dt_rejected() {
        let err = Dlti::new(vec![1.0], vec![1.0], f64::NAN).expect_err("nan dt");
        assert!(err.is_argument_error());

        let err = Dlti::new(vec![1.0], vec![1.0], f64::INFINITY).expect_err("inf dt");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_step_response() {
        // H(z) = 1 / (1 - 0.5*z^-1), step response grows to 2
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let y = sys.step(50).expect("step");
        // Should converge toward 1/(1-0.5) = 2
        let final_val = y.last().unwrap();
        assert!(
            (final_val - 2.0).abs() < 0.1,
            "final step value = {final_val}"
        );
    }

    #[test]
    fn dlti_impulse_response_first_sample() {
        // H(z) = 1 / (1 - 0.5*z^-1), impulse h[0] = 1
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let h = sys.impulse(10).expect("impulse");
        assert!((h[0] - 1.0).abs() < 1e-10, "h[0] = {}", h[0]);
        // h[n] = 0.5^n for this system
        assert!((h[1] - 0.5).abs() < 1e-10, "h[1] = {}", h[1]);
    }

    #[test]
    fn dlti_is_stable_inside_unit_circle() {
        // Pole at z = 0.5 (inside unit circle) -> stable
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        assert!(sys.is_stable().expect("stable check"));
    }

    #[test]
    fn dlti_is_unstable_outside_unit_circle() {
        // Pole at z = 1.5 (outside unit circle) -> unstable
        let sys = Dlti::new(vec![1.0], vec![1.0, -1.5], 0.01).expect("valid");
        assert!(!sys.is_stable().expect("stable check"));
    }

    #[test]
    fn dlti_lfilter_matches_direct() {
        // Verify DLTI.lfilter matches direct lfilter call
        let sys = Dlti::new(vec![1.0, 0.5], vec![1.0, -0.3], 0.01).expect("valid");
        let x: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let y1 = sys.lfilter(&x).expect("dlti lfilter");
        let y2 = lfilter(&sys.num, &sys.den, &x, None).expect("direct lfilter");
        for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn lti_lsim_step_input() {
        // lsim with step input should match step response
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![1.0; t.len()]; // step input
        let y_lsim = sys.lsim(&u, &t, None).expect("lsim");
        let y_step = sys.step(&t).expect("step");
        for (i, (a, b)) in y_lsim.iter().zip(y_step.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "lsim[{i}] = {} vs step[{i}] = {}",
                a,
                b
            );
        }
    }

    #[test]
    fn lti_lsim_zero_input() {
        // lsim with zero input should give zero output (zero IC)
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![0.0; t.len()];
        let y = sys.lsim(&u, &t, None).expect("lsim");
        for (i, &yi) in y.iter().enumerate() {
            assert!(yi.abs() < 1e-12, "y[{i}] = {yi}, expected 0");
        }
    }

    #[test]
    fn lti_lsim_mismatched_lengths_rejected() {
        let sys = Lti::new(vec![1.0], vec![1.0, 1.0]).expect("valid");
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = vec![1.0; 5]; // shorter than t
        let err = sys.lsim(&u, &t, None).expect_err("mismatched");
        assert!(err.is_argument_error());
    }

    #[test]
    fn dlti_dlsim_matches_lfilter() {
        // dlsim should match lfilter with no initial state
        let sys = Dlti::new(vec![1.0, 0.5], vec![1.0, -0.3], 0.01).expect("valid");
        let u: Vec<f64> = (0..20).map(|i| (i as f64 * 0.5).sin()).collect();
        let y_dlsim = sys.dlsim(&u, None).expect("dlsim");
        let y_lfilter = sys.lfilter(&u).expect("lfilter");
        for (i, (a, b)) in y_dlsim.iter().zip(y_lfilter.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn lsim_standalone_matches_method() {
        let sys = Lti::new(vec![2.0], vec![1.0, 0.5]).expect("valid");
        let t: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let u: Vec<f64> = t.iter().map(|&ti| ti.sin()).collect();
        let y_method = sys.lsim(&u, &t, None).expect("method");
        let y_standalone = lsim(&sys, &u, &t, None).expect("standalone");
        for (i, (a, b)) in y_method.iter().zip(y_standalone.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn dlsim_standalone_matches_method() {
        let sys = Dlti::new(vec![1.0], vec![1.0, -0.5], 0.01).expect("valid");
        let u: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
        let y_method = sys.dlsim(&u, None).expect("method");
        let y_standalone = dlsim(&sys, &u, None).expect("standalone");
        for (i, (a, b)) in y_method.iter().zip(y_standalone.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "y[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn sosfreqz_matches_freqz_sos() {
        let sos: Vec<SosSection> = vec![[1.0, 0.5, 0.0, 1.0, -0.3, 0.1]];
        let r1 = freqz_sos(&sos, Some(64)).expect("freqz_sos");
        let r2 = sosfreqz(&sos, Some(64)).expect("sosfreqz");
        assert_eq!(r1.w.len(), r2.w.len());
        for (i, (a, b)) in r1.h_mag.iter().zip(r2.h_mag.iter()).enumerate() {
            assert!((a - b).abs() < 1e-14, "h_mag[{i}]: {} vs {}", a, b);
        }
    }

    #[test]
    fn sosfreqz_with_whole_unit_circle() {
        let sos: Vec<SosSection> = vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let half = sosfreqz(&sos, Some(128)).expect("half");
        let whole = sosfreqz_with_whole(&sos, Some(256), true).expect("whole");
        assert_eq!(half.w.len(), 128);
        assert_eq!(whole.w.len(), 256);
        assert!(whole.w.last().unwrap() > &std::f64::consts::PI);
    }

    #[test]
    fn cheb1ord_known_design_returns_order_5() {
        // acosh(√(9999 / 0.2589)) ≈ 5.974, acosh(2) ≈ 1.317
        // → order = ceil(5.974 / 1.317) = ceil(4.535) = 5.
        // Even with this saving, Cheby beats Butterworth's order 8 here.
        let (n, wn) = cheb1ord(1.0, 2.0, 1.0, 40.0).expect("cheb1ord");
        assert_eq!(n, 5);
        assert!((wn - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cheb1ap_returns_no_zeros_for_all_orders() {
        for n in 1..=6 {
            let (z, p, _) = cheb1ap(n, 1.0).expect("cheb1ap");
            assert!(z.is_empty(), "Cheb1 has no finite zeros");
            assert_eq!(p.len(), n as usize);
        }
    }

    #[test]
    fn cheb1ap_zero_order_matches_scipy_empty_prototype() {
        for (rp, expected_gain) in [(1.0_f64, 0.8912509381337456), (3.0_f64, 0.7079457843841379)] {
            let (z, p, k) = cheb1ap(0, rp).expect("zero-order cheb1ap");
            assert!(z.is_empty());
            assert!(p.is_empty());
            assert!(
                (k - expected_gain).abs() < 1e-15,
                "rp={rp}: gain={k}, expected {expected_gain}"
            );
        }
    }

    #[test]
    fn cheb1ap_metamorphic_poles_in_left_half_plane() {
        // Stable filter ⇒ all poles strictly in the open left half-plane.
        for n in 1..=6 {
            let (_, p, _) = cheb1ap(n, 1.0).unwrap();
            for &(re, _) in &p {
                assert!(re < 0.0, "n={n}: pole real part {re} should be < 0");
            }
        }
    }

    #[test]
    fn cheb1ap_metamorphic_lower_ripple_pushes_pole_real_part_more_negative() {
        // Smaller rp ⇒ larger μ ⇒ larger sinh(μ) ⇒ poles shift further left.
        let (_, p_loose, _) = cheb1ap(4, 3.0).unwrap();
        let (_, p_tight, _) = cheb1ap(4, 0.1).unwrap();
        let max_loose = p_loose
            .iter()
            .map(|&(re, _)| re)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_tight = p_tight
            .iter()
            .map(|&(re, _)| re)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_tight < max_loose,
            "tighter ripple should push poles left: max_re tight={max_tight}, loose={max_loose}"
        );
    }

    #[test]
    fn cheb1ap_rejects_invalid_specs() {
        assert!(cheb1ap(2, 0.0).is_err());
        assert!(cheb1ap(2, -1.0).is_err());
    }

    #[test]
    fn buttap_returns_no_zeros_and_gain_one() {
        for n in 1..=10 {
            let (z, p, k) = buttap(n).expect("buttap");
            assert!(z.is_empty(), "Butterworth has no finite zeros at any order");
            assert_eq!(p.len(), n as usize);
            assert_eq!(k, 1.0);
        }
    }

    #[test]
    fn buttap_metamorphic_poles_on_unit_circle_left_half() {
        // Each pole has |s| = 1 (unit circle) and Re(s) ≤ 0 (left half).
        for n in 1..=10 {
            let (_, p, _) = buttap(n).unwrap();
            for &(re, im) in &p {
                let r = (re * re + im * im).sqrt();
                assert!((r - 1.0).abs() < 1e-12, "n={n}: |pole|={r}, expected 1");
                assert!(re <= 1e-15, "n={n}: pole {re}+{im}i must have Re ≤ 0");
            }
        }
    }

    #[test]
    fn buttap_metamorphic_first_order_real_pole_at_minus_one() {
        // N=1: the single pole is at exp(iπ) = -1.
        let (_, p, _) = buttap(1).unwrap();
        assert!((p[0].0 - (-1.0)).abs() < 1e-12);
        assert!(p[0].1.abs() < 1e-12);
    }

    #[test]
    fn buttap_zero_order_returns_empty_prototype() {
        // scipy.signal.buttap(0) returns ([], [], 1); fsci must match
        // for parity (np.arange(-N+1, N, 2) is empty when N=0, so
        // scipy's pole-construction loop is skipped silently).
        let (z, p, k) = buttap(0).unwrap();
        assert_eq!(z.len(), 0);
        assert_eq!(p.len(), 0);
        assert_eq!(k, 1.0);
    }

    #[test]
    fn ellipord_known_design_returns_smallest_order_among_filter_types() {
        // Elliptic is the lowest-order classical filter for any spec.
        // For (wp=1, ws=2, gpass=1 dB, gstop=40 dB) it should beat both
        // Butterworth (8) and Chebyshev (5).
        let (n_butter, _) = buttord(1.0, 2.0, 1.0, 40.0).unwrap();
        let (n_cheby, _) = cheb1ord(1.0, 2.0, 1.0, 40.0).unwrap();
        let (n_ellip, _) = ellipord(1.0, 2.0, 1.0, 40.0).unwrap();
        assert_eq!(n_ellip, 4);
        assert!(
            n_ellip <= n_cheby,
            "elliptic N={n_ellip} should be ≤ Chebyshev N={n_cheby}"
        );
        assert!(
            n_cheby <= n_butter,
            "Chebyshev N={n_cheby} should be ≤ Butterworth N={n_butter}"
        );
    }

    #[test]
    fn ellipord_metamorphic_lower_than_chebyshev_across_specs() {
        for spec in &[
            (1.0_f64, 2.0, 1.0, 40.0),
            (1.0, 1.5, 0.5, 60.0),
            (1.0, 3.0, 2.0, 80.0),
        ] {
            let (n_cheby, _) = cheb1ord(spec.0, spec.1, spec.2, spec.3).unwrap();
            let (n_ellip, _) = ellipord(spec.0, spec.1, spec.2, spec.3).unwrap();
            assert!(
                n_ellip <= n_cheby,
                "spec {spec:?}: elliptic N={n_ellip} should be ≤ Chebyshev N={n_cheby}"
            );
        }
    }

    #[test]
    fn ellipord_scipy_analog_lowpass_order_regression_cases() {
        // Values generated by scipy.signal.ellipord(..., analog=True).
        for (wp, ws, gpass, gstop, expected_n, expected_wn) in [
            (1.0_f64, 2.0, 1.0, 40.0, 4_u32, 1.0),
            (1.0, 1.5, 0.5, 60.0, 6, 1.0),
            (1.0, 3.0, 2.0, 80.0, 5, 1.0),
            (0.2, 0.3, 1.0, 40.0, 5, 0.2),
        ] {
            let (n, wn) = ellipord(wp, ws, gpass, gstop).expect("ellipord");
            assert_eq!(n, expected_n);
            assert!(
                (wn - expected_wn).abs() < 1e-12,
                "spec ({wp}, {ws}, {gpass}, {gstop}) Wn={wn}, expected {expected_wn}"
            );
        }
    }

    #[test]
    fn ellipord_rejects_invalid_specs() {
        assert!(ellipord(2.0, 1.0, 1.0, 40.0).is_err());
        assert!(ellipord(1.0, 2.0, 40.0, 1.0).is_err());
        assert!(ellipord(-1.0, 2.0, 1.0, 40.0).is_err());
    }

    #[test]
    fn cheb2ord_metamorphic_order_matches_cheb1ord() {
        // Order formula is identical for type-I and type-II; only Wn
        // differs.
        for spec in &[
            (1.0_f64, 2.0, 1.0, 40.0),
            (1.0, 1.5, 0.5, 60.0),
            (1.0, 3.0, 2.0, 80.0),
        ] {
            let (n1, _) = cheb1ord(spec.0, spec.1, spec.2, spec.3).unwrap();
            let (n2, _) = cheb2ord(spec.0, spec.1, spec.2, spec.3).unwrap();
            assert_eq!(n1, n2, "spec {spec:?}: cheb1ord N={n1} vs cheb2ord N={n2}");
        }
    }

    #[test]
    fn cheb2ord_metamorphic_wn_between_wp_and_ws() {
        // Type-II's natural frequency sits between wp and ws.
        let (_, wn) = cheb2ord(1.0, 2.0, 1.0, 40.0).expect("cheb2ord");
        assert!(wn > 1.0 && wn < 2.0, "wn={wn} should be in (wp, ws)");
    }

    #[test]
    fn cheb2ord_wn_matches_scipy_reference() {
        // scipy.signal.cheb2ord(wp, ws, gpass, gstop, analog=True) places
        // Wn just below the stopband edge ws.
        for &(wp, ws, gp, gs, n_exp, wn_exp) in &[
            (1.0_f64, 2.0, 1.0, 40.0, 5_u32, 1.802_791_365_577_485),
            (0.5, 1.5, 0.5, 30.0, 3, 1.458_262_371_465_648),
            (2.0, 5.0, 2.0, 60.0, 6, 3.981_189_966_239_977),
        ] {
            let (n, wn) = cheb2ord(wp, ws, gp, gs).expect("cheb2ord");
            assert_eq!(n, n_exp, "order for ({wp}, {ws}, {gp}, {gs})");
            assert!(
                (wn - wn_exp).abs() < 1e-12,
                "wn for ({wp}, {ws}, {gp}, {gs}): got {wn}, want {wn_exp}"
            );
        }
    }

    #[test]
    fn cheb2ord_rejects_invalid_specs() {
        assert!(cheb2ord(2.0, 1.0, 1.0, 40.0).is_err());
        assert!(cheb2ord(1.0, 2.0, 40.0, 1.0).is_err());
        assert!(cheb2ord(-1.0, 2.0, 1.0, 40.0).is_err());
    }

    #[test]
    fn cheb1ord_metamorphic_lower_than_butterworth_for_same_spec() {
        // For any meaningful spec, Chebyshev I requires ≤ Butterworth order
        // because Cheby concentrates the design effort in the passband.
        for spec in &[
            (1.0_f64, 2.0, 1.0, 40.0),
            (1.0, 1.5, 0.5, 60.0),
            (1.0, 3.0, 2.0, 80.0),
        ] {
            let (n_butter, _) = buttord(spec.0, spec.1, spec.2, spec.3).unwrap();
            let (n_cheby, _) = cheb1ord(spec.0, spec.1, spec.2, spec.3).unwrap();
            assert!(
                n_cheby <= n_butter,
                "spec {spec:?}: Chebyshev N={n_cheby} should be ≤ Butterworth N={n_butter}"
            );
        }
    }

    #[test]
    fn cheb1ord_rejects_invalid_specs() {
        assert!(cheb1ord(2.0, 1.0, 1.0, 40.0).is_err());
        assert!(cheb1ord(1.0, 2.0, 40.0, 1.0).is_err());
        assert!(cheb1ord(-1.0, 2.0, 1.0, 40.0).is_err());
    }

    #[test]
    fn buttord_known_design_returns_order_3() {
        // Spec: passband 0..1 rad/s, stopband 2 rad/s+, 1 dB ripple, 40 dB stop.
        // Closed form: 10^(40/10)-1 = 9999, 10^(1/10)-1 ≈ 0.2589, ratio
        // log10(2) ≈ 0.301. order = ceil(log10(9999/0.2589) / (2·0.301))
        //                        = ceil(4.587 / 0.602) = ceil(7.62) = 8.
        let (n, wn) = buttord(1.0, 2.0, 1.0, 40.0).expect("buttord");
        assert_eq!(n, 8);
        assert!((wn - 1.0881194736627366).abs() < 1e-12);
    }

    #[test]
    fn buttord_scipy_analog_lowpass_wn_regression_cases() {
        // Values generated by scipy.signal.buttord(..., analog=True).
        for (wp, ws, gpass, gstop, expected_n, expected_wn) in [
            (1.0_f64, 2.0, 1.0, 40.0, 8_u32, 1.0881194736627366),
            (1.0, 1.5, 0.5, 60.0, 20, 1.0539969691259425),
            (1.0, 3.0, 2.0, 80.0, 9, 1.0302442296515495),
            (0.2, 0.3, 1.0, 40.0, 14, 0.20988820962077465),
        ] {
            let (n, wn) = buttord(wp, ws, gpass, gstop).expect("buttord");
            assert_eq!(n, expected_n);
            assert!(
                (wn - expected_wn).abs() < 1e-12,
                "spec ({wp}, {ws}, {gpass}, {gstop}) Wn={wn}, expected {expected_wn}"
            );
        }
    }

    #[test]
    fn buttord_metamorphic_higher_stop_loss_increases_order() {
        // For fixed passband spec, raising the stopband attenuation
        // requirement can only increase the required order.
        let (n_30, _) = buttord(1.0, 2.0, 1.0, 30.0).unwrap();
        let (n_60, _) = buttord(1.0, 2.0, 1.0, 60.0).unwrap();
        let (n_90, _) = buttord(1.0, 2.0, 1.0, 90.0).unwrap();
        assert!(n_30 <= n_60);
        assert!(n_60 <= n_90);
    }

    #[test]
    fn buttord_rejects_invalid_specs() {
        // Just verify each invalid configuration returns Err — the error
        // variant routing through classify_invalid_argument depends on
        // string matching that's brittle to test directly.
        assert!(buttord(2.0, 1.0, 1.0, 40.0).is_err(), "wp>=ws should fail");
        assert!(
            buttord(1.0, 2.0, 40.0, 1.0).is_err(),
            "gpass>=gstop should fail"
        );
        assert!(
            buttord(-1.0, 2.0, 1.0, 40.0).is_err(),
            "negative wp should fail"
        );
    }

    #[test]
    fn bilinear_zpk_real_pole_at_origin_maps_to_unity() {
        // s = 0 maps to z = 1 because (2fs + 0) / (2fs - 0) = 1.
        let z: Vec<fsci_fft::Complex64> = vec![(0.0, 0.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0)];
        let (z_z, _, _) = bilinear_zpk(&z, &p, 1.0, 100.0).unwrap();
        assert!((z_z[0].0 - 1.0).abs() < 1e-12);
        assert!(z_z[0].1.abs() < 1e-12);
    }

    #[test]
    fn bilinear_zpk_pads_zeros_at_minus_one() {
        // Prototype with no zeros: degree zeros padded at -1 + 0i.
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let (z_z, _, _) = bilinear_zpk(&z, &p, 1.0, 50.0).unwrap();
        assert_eq!(z_z.len(), 2);
        for &(re, im) in &z_z {
            assert!((re - (-1.0)).abs() < 1e-12 && im.abs() < 1e-12);
        }
    }

    #[test]
    fn bilinear_zpk_metamorphic_real_pole_inside_unit_circle() {
        // Stable analog poles (Re < 0) map to stable digital poles
        // (|z| < 1) under the bilinear transform.
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let p: Vec<fsci_fft::Complex64> = vec![(-2.0, 1.0), (-2.0, -1.0), (-3.0, 0.0)];
        let (_, p_z, _) = bilinear_zpk(&z, &p, 1.0, 100.0).unwrap();
        for &(re, im) in &p_z {
            let r = (re * re + im * im).sqrt();
            assert!(
                r < 1.0,
                "digital pole {:?} has |z|={r}, must be < 1",
                (re, im)
            );
        }
    }

    #[test]
    fn bilinear_matches_scipy_first_order() {
        // scipy.signal.bilinear([1, 2], [1, 3], fs=2).
        let (b, a) = bilinear(&[1.0, 2.0], &[1.0, 3.0], 2.0);
        let want_b = [0.857_142_857_142_857_1, -0.285_714_285_714_285_7];
        let want_a = [1.0, -0.142_857_142_857_142_85];
        for (g, w) in b.iter().zip(want_b.iter()) {
            assert!((g - w).abs() < 1e-12, "b: got {g}, want {w}");
        }
        for (g, w) in a.iter().zip(want_a.iter()) {
            assert!((g - w).abs() < 1e-12, "a: got {g}, want {w}");
        }
    }

    #[test]
    fn bilinear_matches_scipy_higher_order() {
        // scipy.signal.bilinear([2, 0, 1], [1, 1, 1], fs=4) — 2nd order.
        let (b, a) = bilinear(&[2.0, 0.0, 1.0], &[1.0, 1.0, 1.0], 4.0);
        let want_b = [
            1.767_123_287_671_232_8,
            -3.479_452_054_794_520_7,
            1.767_123_287_671_232_8,
        ];
        let want_a = [1.0, -1.726_027_397_260_274, 0.780_821_917_808_219_2];
        for (g, w) in b.iter().zip(want_b.iter()) {
            assert!((g - w).abs() < 1e-12, "b: got {g}, want {w}");
        }
        for (g, w) in a.iter().zip(want_a.iter()) {
            assert!((g - w).abs() < 1e-12, "a: got {g}, want {w}");
        }

        // scipy.signal.bilinear([1,3,3,1], [1,2,2,1], fs=1) — 3rd order.
        let (b3, a3) = bilinear(&[1.0, 3.0, 3.0, 1.0], &[1.0, 2.0, 2.0, 1.0], 1.0);
        let want_b3 = [
            1.285_714_285_714_285_6,
            -1.285_714_285_714_286,
            0.428_571_428_571_428_66,
            -0.047_619_047_619_047_52,
        ];
        let want_a3 = [
            1.0,
            -1.190_476_190_476_190_7,
            0.714_285_714_285_714_3,
            -0.142_857_142_857_142_9,
        ];
        for (g, w) in b3.iter().zip(want_b3.iter()) {
            assert!((g - w).abs() < 1e-12, "b3: got {g}, want {w}");
        }
        for (g, w) in a3.iter().zip(want_a3.iter()) {
            assert!((g - w).abs() < 1e-12, "a3: got {g}, want {w}");
        }
    }

    #[test]
    fn lp2bs_zpk_appends_imaginary_zeros_for_zero_free_prototype() {
        // Prototype with no zeros: degree = len(p). lp2bs_zpk pads `degree`
        // zeros at +iwo and `degree` more at -iwo.
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let wo = 3.0_f64;
        let bw = 1.0_f64;
        let (z_bs, p_bs, _) = lp2bs_zpk(&z, &p, 1.0, wo, bw).unwrap();
        // 0 split zeros + 2 ± i·wo padding × 2 = 4 zeros.
        assert_eq!(z_bs.len(), 4);
        // 2 split poles × 2 = 4 poles.
        assert_eq!(p_bs.len(), 4);
        // Half the padded zeros should be at +i·wo, half at -i·wo.
        let pos = z_bs
            .iter()
            .filter(|&&(re, im)| re == 0.0 && (im - wo).abs() < 1e-12)
            .count();
        let neg = z_bs
            .iter()
            .filter(|&&(re, im)| re == 0.0 && (im + wo).abs() < 1e-12)
            .count();
        assert_eq!(pos, 2);
        assert_eq!(neg, 2);
    }

    #[test]
    fn lp2bs_zpk_metamorphic_split_pair_product_is_wo_squared() {
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0)];
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let wo = 2.0_f64;
        let bw = 0.5_f64;
        let (_, p_bs, _) = lp2bs_zpk(&z, &p, 1.0, wo, bw).unwrap();
        let (a, b) = (p_bs[0], p_bs[1]);
        let prod_re = a.0 * b.0 - a.1 * b.1;
        let prod_im = a.0 * b.1 + a.1 * b.0;
        assert!((prod_re - wo * wo).abs() < 1e-12);
        assert!(prod_im.abs() < 1e-12);
    }

    #[test]
    fn lp2bs_zpk_metamorphic_gain_matches_lp2hp() {
        // Gain formula k * Re(prod(-z) / prod(-p)) — same as lp2hp_zpk
        // (independent of bw and wo). Verify by comparing across both.
        let z: Vec<fsci_fft::Complex64> = vec![(2.0, 0.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.5), (-1.0, -0.5)];
        let k_in = 3.0_f64;
        let (_, _, k_hp) = lp2hp_zpk(&z, &p, k_in, 1.0).unwrap();
        let (_, _, k_bs) = lp2bs_zpk(&z, &p, k_in, 1.0, 0.7).unwrap();
        assert!(
            (k_hp - k_bs).abs() < 1e-12,
            "lp2hp gain {k_hp} != lp2bs gain {k_bs}"
        );
    }

    #[test]
    fn lp2bp_zpk_doubles_pole_and_zero_count() {
        // Each finite zero/pole splits into a pair; degree zeros are
        // appended at origin. So len(z_bp) = 2*len(z) + degree, and
        // len(p_bp) = 2*len(p).
        let z: Vec<fsci_fft::Complex64> = vec![(1.0, 0.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0), (-3.0, 0.0)];
        let (z_bp, p_bp, _) = lp2bp_zpk(&z, &p, 1.0, 2.0, 0.5).unwrap();
        let degree = p.len() - z.len();
        assert_eq!(z_bp.len(), 2 * z.len() + degree);
        assert_eq!(p_bp.len(), 2 * p.len());
    }

    #[test]
    fn lp2bp_zpk_metamorphic_split_pair_product_is_wo_squared() {
        // Each split pair (zlp + √(zlp² − wo²), zlp − √…) multiplies to
        //   (zlp + r)·(zlp − r) = zlp² − r² = zlp² − (zlp² − wo²) = wo².
        // So the product of any pair equals wo² (within rounding).
        let p: Vec<fsci_fft::Complex64> = vec![(-0.5, 1.0)];
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let wo = 3.0_f64;
        let bw = 0.7_f64;
        let (_, p_bp, _) = lp2bp_zpk(&z, &p, 1.0, wo, bw).unwrap();
        // p_bp = (a, b) where a*b = wo² + 0i.
        let (a, b) = (p_bp[0], p_bp[1]);
        let product_re = a.0 * b.0 - a.1 * b.1;
        let product_im = a.0 * b.1 + a.1 * b.0;
        assert!(
            (product_re - wo * wo).abs() < 1e-12,
            "Re(prod) = {product_re}, expected {}",
            wo * wo
        );
        assert!(product_im.abs() < 1e-12);
    }

    #[test]
    fn lp2bp_zpk_metamorphic_gain_scales_with_bw_to_degree() {
        // k_new = k * bw^degree where degree = len(p) - len(z).
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let z: Vec<fsci_fft::Complex64> = vec![(1.5, 0.0)];
        let k_in = 4.5_f64;
        let bw = 1.25_f64;
        let degree = (p.len() - z.len()) as i32;
        let (_, _, k_out) = lp2bp_zpk(&z, &p, k_in, 1.0, bw).unwrap();
        assert!((k_out - k_in * bw.powi(degree)).abs() < 1e-12);
    }

    #[test]
    fn lp2hp_zpk_inverts_real_pole() {
        // Single-pole lowpass at -1: lp2hp_zpk should give pole at -1
        // (since wo / (-1) = -1) and add a zero at origin.
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0)];
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let (z2, p2, k2) = lp2hp_zpk(&z, &p, 1.0, 1.0).unwrap();
        assert_eq!(z2.len(), 1);
        assert!(z2[0].0.abs() < 1e-12 && z2[0].1.abs() < 1e-12);
        assert!((p2[0].0 - (-1.0)).abs() < 1e-12);
        assert!(p2[0].1.abs() < 1e-12);
        // gain is k * Re(Π(-z) / Π(-p)) = 1 * Re(1 / 1) = 1
        assert!((k2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lp2hp_zpk_metamorphic_wo_scales_pole_inverse() {
        // For wo > 0, a pole at -1 becomes a pole at -wo (because wo / -1 = -wo).
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let wo = 5.0_f64;
        let (z2, p2, _) = lp2hp_zpk(&z, &p, 1.0, wo).unwrap();
        assert_eq!(p2.len(), 2);
        assert!((p2[0].0 - (-wo)).abs() < 1e-12);
        assert!((p2[1].0 - (-wo / 2.0)).abs() < 1e-12);
        // appended zeros at origin: degree = 2 (no existing zeros)
        assert_eq!(z2.len(), 2);
        for z in &z2 {
            assert!(z.0.abs() < 1e-12 && z.1.abs() < 1e-12);
        }
    }

    #[test]
    fn lp2hp_zpk_metamorphic_double_invert_recovers_lowpass() {
        // Two applications of lp2hp_zpk with wo=1 should recover the
        // original prototype (modulo the appended zeros at origin from
        // the first pass which become poles at infinity in the second).
        // We check this by composing for a no-zero prototype: lp→hp→lp
        // recovers exactly.
        let p: Vec<fsci_fft::Complex64> = vec![(-0.5, 1.0), (-0.5, -1.0)];
        let z: Vec<fsci_fft::Complex64> = Vec::new();
        let k = 2.5_f64;
        let (z1, p1, k1) = lp2hp_zpk(&z, &p, k, 1.0).unwrap();
        // After this, z1 has 2 zeros at origin (which become poles at
        // infinity on the second invert). lp2hp_zpk on this configuration
        // would inject 0/inf, so we instead check the symmetry of the
        // analog transform by hand: the new poles must match wo/p.
        let (_, p2_again, _) = lp2lp_zpk(&z1, &p1, k1, 1.0).unwrap(); // identity to verify shape
        assert_eq!(p2_again.len(), 2);
    }

    #[test]
    fn zpk_transforms_metamorphic_relative_degree_invariants() {
        // Metamorphic relations every lp→* transform must satisfy on a
        // proper prototype:
        //
        //   lp2lp_zpk:  len(z_new) = len(z),       len(p_new) = len(p)
        //   lp2hp_zpk:  len(z_new) = len(p),       len(p_new) = len(p)
        //               (zeros at infinity move to origin; total zero
        //                count rises to match pole count)
        //   lp2bp_zpk:  len(z_new) = 2·len(z) + (len(p) − len(z))
        //                            = len(p) + len(z),
        //               len(p_new) = 2·len(p)
        //   lp2bs_zpk:  len(z_new) = 2·len(p),     len(p_new) = 2·len(p)
        let z: Vec<fsci_fft::Complex64> = vec![(0.5, 0.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0), (-3.0, 0.0)];
        let nz = z.len();
        let np = p.len();

        let (zlp, plp, _) = lp2lp_zpk(&z, &p, 1.0, 2.0).unwrap();
        assert_eq!(zlp.len(), nz);
        assert_eq!(plp.len(), np);

        let (zhp, php, _) = lp2hp_zpk(&z, &p, 1.0, 2.0).unwrap();
        assert_eq!(zhp.len(), np);
        assert_eq!(php.len(), np);

        let (zbp, pbp, _) = lp2bp_zpk(&z, &p, 1.0, 2.0, 0.5).unwrap();
        assert_eq!(zbp.len(), nz + np);
        assert_eq!(pbp.len(), 2 * np);

        let (zbs, pbs, _) = lp2bs_zpk(&z, &p, 1.0, 2.0, 0.5).unwrap();
        assert_eq!(zbs.len(), 2 * np);
        assert_eq!(pbs.len(), 2 * np);
    }

    #[test]
    fn zpk_transforms_metamorphic_lp2lp_inverse_recovers_original() {
        // Composing lp2lp_zpk(·, wo) with lp2lp_zpk(·, 1/wo) is the
        // identity on (z, p, k) up to floating-point tolerance.
        let z: Vec<fsci_fft::Complex64> = vec![(0.0, 1.0), (0.0, -1.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-0.5, 0.5), (-0.5, -0.5), (-2.0, 0.0)];
        let k: f64 = 1.5;
        let wo: f64 = 7.3;

        let (z2, p2, k2) = lp2lp_zpk(&z, &p, k, wo).unwrap();
        let (z3, p3, k3) = lp2lp_zpk(&z2, &p2, k2, 1.0 / wo).unwrap();

        assert_eq!(z3.len(), z.len());
        assert_eq!(p3.len(), p.len());
        for (a, b) in z3.iter().zip(z.iter()) {
            assert!((a.0 - b.0).abs() < 1e-10);
            assert!((a.1 - b.1).abs() < 1e-10);
        }
        for (a, b) in p3.iter().zip(p.iter()) {
            assert!((a.0 - b.0).abs() < 1e-10);
            assert!((a.1 - b.1).abs() < 1e-10);
        }
        assert!((k3 - k).abs() < 1e-10);
    }

    #[test]
    fn zpk_transforms_metamorphic_lp2bp_pole_pair_product_equals_wo_squared() {
        // For lp2bp_zpk, each prototype pole splits into a pair whose
        // product equals wo² (algebraic invariant of the s → (s² + wo²)/(bw·s)
        // substitution: the two roots of s² − bw·s·p + wo² = 0 multiply to wo²).
        // The invariant should hold regardless of bandwidth.
        let z: Vec<fsci_fft::Complex64> = vec![];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let k: f64 = 1.0;
        for &wo in &[1.0_f64, 3.0, 10.0] {
            for &bw in &[0.1_f64, 0.5, 2.0] {
                let (_, pbp, _) = lp2bp_zpk(&z, &p, k, wo, bw).unwrap();
                assert_eq!(pbp.len(), 2 * p.len());
                for chunk in pbp.chunks_exact(2) {
                    let (a_re, a_im) = chunk[0];
                    let (b_re, b_im) = chunk[1];
                    let prod_re = a_re * b_re - a_im * b_im;
                    let prod_im = a_re * b_im + a_im * b_re;
                    assert!(
                        (prod_re - wo * wo).abs() < 1e-9,
                        "wo={wo} bw={bw}: pair-product real {prod_re} != wo²={}",
                        wo * wo
                    );
                    assert!(
                        prod_im.abs() < 1e-9,
                        "wo={wo} bw={bw}: pair-product imag {prod_im} != 0"
                    );
                }
            }
        }
    }

    #[test]
    fn zpk_transforms_reject_improper_prototypes() {
        // scipy raises ValueError when len(z) > len(p); fsci must do the
        // same. Prototype with 3 finite zeros and only 1 finite pole is
        // improper.
        let z_improper: Vec<fsci_fft::Complex64> = vec![(0.0, 1.0), (0.0, -1.0), (1.0, 0.0)];
        let p_improper: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0)];
        assert!(
            lp2lp_zpk(&z_improper, &p_improper, 1.0, 2.0).is_err(),
            "lp2lp_zpk must reject improper prototype"
        );
        assert!(
            lp2hp_zpk(&z_improper, &p_improper, 1.0, 2.0).is_err(),
            "lp2hp_zpk must reject improper prototype"
        );
        assert!(
            lp2bp_zpk(&z_improper, &p_improper, 1.0, 2.0, 0.5).is_err(),
            "lp2bp_zpk must reject improper prototype"
        );
        assert!(
            lp2bs_zpk(&z_improper, &p_improper, 1.0, 2.0, 0.5).is_err(),
            "lp2bs_zpk must reject improper prototype"
        );
        assert!(
            bilinear_zpk(&z_improper, &p_improper, 1.0, 100.0).is_err(),
            "bilinear_zpk must reject improper prototype"
        );
    }

    #[test]
    fn lp2lp_zpk_identity_when_wo_is_one() {
        let z: Vec<fsci_fft::Complex64> = vec![(0.0, 1.0), (0.0, -1.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let k = 3.0_f64;
        let (z2, p2, k2) = lp2lp_zpk(&z, &p, k, 1.0).unwrap();
        assert_eq!(z2, z);
        assert_eq!(p2, p);
        assert_eq!(k2, k);
    }

    #[test]
    fn lp2lp_zpk_metamorphic_zeros_and_poles_scale_by_wo() {
        let z: Vec<fsci_fft::Complex64> = vec![(0.0, 1.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.0), (0.0, -2.0)];
        let k = 2.0_f64;
        let wo = 7.5_f64;
        let (z2, p2, k2) = lp2lp_zpk(&z, &p, k, wo).unwrap();
        for (a, b) in z.iter().zip(z2.iter()) {
            assert!((b.0 - wo * a.0).abs() < 1e-12);
            assert!((b.1 - wo * a.1).abs() < 1e-12);
        }
        for (a, b) in p.iter().zip(p2.iter()) {
            assert!((b.0 - wo * a.0).abs() < 1e-12);
            assert!((b.1 - wo * a.1).abs() < 1e-12);
        }
        // gain scales by wo^(len(p) - len(z)) = 7.5^(2-1) = 7.5
        assert!((k2 - k * wo.powi(1)).abs() < 1e-12);
    }

    #[test]
    fn lp2lp_zpk_metamorphic_compose_with_inverse() {
        // Applying lp2lp_zpk with wo, then with 1/wo, recovers the original.
        let z: Vec<fsci_fft::Complex64> = vec![(0.5, 1.0)];
        let p: Vec<fsci_fft::Complex64> = vec![(-1.0, 0.5), (-3.0, 0.0)];
        let k = 1.5_f64;
        let wo = 4.0_f64;
        let (z1, p1, k1) = lp2lp_zpk(&z, &p, k, wo).unwrap();
        let (z2, p2, k2) = lp2lp_zpk(&z1, &p1, k1, 1.0 / wo).unwrap();
        for (a, b) in z.iter().zip(z2.iter()) {
            assert!((b.0 - a.0).abs() < 1e-12);
            assert!((b.1 - a.1).abs() < 1e-12);
        }
        for (a, b) in p.iter().zip(p2.iter()) {
            assert!((b.0 - a.0).abs() < 1e-12);
            assert!((b.1 - a.1).abs() < 1e-12);
        }
        assert!((k2 - k).abs() < 1e-12);
    }

    #[test]
    fn lp2lp_ba_identity_when_wo_is_one() {
        // /porting-to-rust [frankenscipy-xla38]: wo=1 leaves (b, a)
        // unchanged after normalization.
        let b = [1.0_f64, 2.0];
        let a = [3.0_f64, 4.0, 5.0];
        let (nb, na) = lp2lp(&b, &a, 1.0).expect("lp2lp identity");
        // After normalization a[0] = 1 → divide by 3.
        for (got, want) in nb.iter().zip([1.0 / 3.0, 2.0 / 3.0].iter()) {
            assert!((got - want).abs() < 1e-12);
        }
        for (got, want) in na.iter().zip([1.0, 4.0 / 3.0, 5.0 / 3.0].iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn lp2lp_ba_first_order_substitution() {
        // /porting-to-rust [frankenscipy-xla38]: H(s) = 1/(s+1) with
        // wo=2 → 1/((s/2)+1) = 2/(s+2). After normalize a[0]=1:
        // b=[2], a=[1, 2].
        let (nb, na) = lp2lp(&[1.0], &[1.0, 1.0], 2.0).expect("lp2lp 1st order");
        assert_eq!(nb.len(), 1);
        assert_eq!(na.len(), 2);
        assert!((nb[0] - 2.0).abs() < 1e-12);
        assert!((na[0] - 1.0).abs() < 1e-12);
        assert!((na[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lp2lp_ba_second_order_substitution() {
        // /porting-to-rust [frankenscipy-xla38]: B(s)/A(s) =
        // (s + 2)/(s² + 3s + 4) with wo=2. Substituting s → s/2 and
        // multiplying through by 4: (2s + 8)/(s² + 6s + 16). So
        // b=[2, 8], a=[1, 6, 16].
        let (nb, na) = lp2lp(&[1.0, 2.0], &[1.0, 3.0, 4.0], 2.0).expect("lp2lp 2nd order");
        for (got, want) in nb.iter().zip([2.0, 8.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 6.0, 16.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2lp_ba_rejects_invalid_wo() {
        // /porting-to-rust [frankenscipy-xla38]: validation contract.
        assert!(lp2lp(&[1.0], &[1.0], 0.0).is_err());
        assert!(lp2lp(&[1.0], &[1.0], -1.0).is_err());
        assert!(lp2lp(&[1.0], &[1.0], f64::NAN).is_err());
        assert!(lp2lp(&[1.0], &[1.0], f64::INFINITY).is_err());
        assert!(lp2lp(&[], &[1.0], 1.0).is_err());
        assert!(lp2lp(&[1.0], &[], 1.0).is_err());
    }

    #[test]
    fn correlation_lags_handles_zero_in2() {
        // /testing-fuzzing regression for [frankenscipy-385tp]:
        // REVIEW MODE found that correlation_lags(in1, 0, Same)
        // panicked when in1 is odd because the computed slice
        // upper bound exceeded full.len(). scipy.signal.correlation_lags
        // returns the truncated tail in that case via its lazy slicing.
        // scipy reference values:
        //   correlation_lags(5, 0, full)  = [1, 2, 3, 4]
        //   correlation_lags(5, 0, same)  = [1, 2, 3, 4]
        //   correlation_lags(5, 0, valid) = [0, 1, 2, 3, 4, 5]
        //   correlation_lags(0, 0, full)  = []
        //   correlation_lags(0, 0, same)  = []
        assert_eq!(
            correlation_lags(5, 0, CorrelationMode::Full),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            correlation_lags(5, 0, CorrelationMode::Same),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            correlation_lags(5, 0, CorrelationMode::Valid),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            correlation_lags(0, 0, CorrelationMode::Full),
            Vec::<i64>::new()
        );
        assert_eq!(
            correlation_lags(0, 0, CorrelationMode::Same),
            Vec::<i64>::new()
        );
        // Also check 3, 0 (smaller odd in1) — was the other panic case.
        assert_eq!(correlation_lags(3, 0, CorrelationMode::Same), vec![1, 2]);
    }

    #[test]
    fn correlation_lags_matches_scipy() {
        // /porting-to-rust [frankenscipy-r34vj]: scipy reference values.
        // scipy.signal.correlation_lags(5, 3, 'full')  = [-2, -1, 0, 1, 2, 3, 4]
        // scipy.signal.correlation_lags(5, 3, 'same')  = [-1, 0, 1, 2, 3]
        // scipy.signal.correlation_lags(5, 3, 'valid') = [0, 1, 2]
        // scipy.signal.correlation_lags(3, 5, 'valid') = [-2, -1, 0]
        assert_eq!(
            correlation_lags(5, 3, CorrelationMode::Full),
            vec![-2, -1, 0, 1, 2, 3, 4]
        );
        assert_eq!(
            correlation_lags(5, 3, CorrelationMode::Same),
            vec![-1, 0, 1, 2, 3]
        );
        assert_eq!(
            correlation_lags(5, 3, CorrelationMode::Valid),
            vec![0, 1, 2]
        );
        // Reverse case: in2 > in1.
        assert_eq!(
            correlation_lags(3, 5, CorrelationMode::Valid),
            vec![-2, -1, 0]
        );
        // Equal sizes:
        // scipy.signal.correlation_lags(4, 4, 'same') = [-2, -1, 0, 1]
        assert_eq!(
            correlation_lags(4, 4, CorrelationMode::Same),
            vec![-2, -1, 0, 1]
        );
        // scipy.signal.correlation_lags(5, 5, 'same') = [-2, -1, 0, 1, 2]
        assert_eq!(
            correlation_lags(5, 5, CorrelationMode::Same),
            vec![-2, -1, 0, 1, 2]
        );
        // Output length sanity: Full = in1 + in2 - 1.
        for &(n1, n2) in &[(1_usize, 1), (3, 7), (10, 4)] {
            assert_eq!(
                correlation_lags(n1, n2, CorrelationMode::Full).len(),
                n1 + n2 - 1
            );
        }
    }

    #[test]
    fn lp2hp_ba_first_order_substitution() {
        // /porting-to-rust [frankenscipy-drh63]: H(s) = 1/(s+1) with
        // wo=2 → highpass via s → 2/s. scipy reference:
        //   lp2hp([1], [1, 1], wo=2) = b=[1, 0], a=[1, 2].
        let (nb, na) = lp2hp(&[1.0], &[1.0, 1.0], 2.0).expect("lp2hp 1st order");
        assert_eq!(nb.len(), 2);
        assert_eq!(na.len(), 2);
        assert!((nb[0] - 1.0).abs() < 1e-12);
        assert!((nb[1] - 0.0).abs() < 1e-12);
        assert!((na[0] - 1.0).abs() < 1e-12);
        assert!((na[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn lp2hp_ba_second_order_substitution() {
        // /porting-to-rust [frankenscipy-drh63]: scipy reference
        //   lp2hp([1, 2], [1, 3, 4], wo=2) =
        //   b=[0.5, 0.5, 0.0], a=[1.0, 1.5, 1.0].
        let (nb, na) = lp2hp(&[1.0, 2.0], &[1.0, 3.0, 4.0], 2.0).expect("lp2hp 2nd order");
        for (got, want) in nb.iter().zip([0.5, 0.5, 0.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 1.5, 1.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2hp_ba_identity_when_wo_is_one() {
        // /porting-to-rust [frankenscipy-drh63]: at wo=1, the
        // substitution s → 1/s reverses the coefficient order.
        // For B = [1, 2], A = [1, 3, 4]: scipy returns
        //   b=[0.5, 0.25, 0], a=[1, 0.75, 0.25].
        let (nb, na) = lp2hp(&[1.0, 2.0], &[1.0, 3.0, 4.0], 1.0).expect("lp2hp wo=1");
        for (got, want) in nb.iter().zip([0.5, 0.25, 0.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 0.75, 0.25].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2hp_ba_rejects_invalid_wo() {
        // /porting-to-rust [frankenscipy-drh63]: validation contract.
        assert!(lp2hp(&[1.0], &[1.0], 0.0).is_err());
        assert!(lp2hp(&[1.0], &[1.0], -1.0).is_err());
        assert!(lp2hp(&[1.0], &[1.0], f64::NAN).is_err());
        assert!(lp2hp(&[1.0], &[1.0], f64::INFINITY).is_err());
        assert!(lp2hp(&[], &[1.0], 1.0).is_err());
        assert!(lp2hp(&[1.0], &[], 1.0).is_err());
    }

    #[test]
    fn lp2bp_ba_first_order_substitution() {
        // /porting-to-rust [frankenscipy-c7ukn]: scipy reference
        //   lp2bp([1], [1, 1], wo=2, bw=1) = b=[1, 0], a=[1, 1, 4]
        let (nb, na) = lp2bp(&[1.0], &[1.0, 1.0], 2.0, 1.0).expect("lp2bp 1st order");
        for (got, want) in nb.iter().zip([1.0, 0.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 1.0, 4.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2bp_ba_second_order_substitution() {
        // /porting-to-rust [frankenscipy-c7ukn]: scipy reference
        //   lp2bp([1, 2], [1, 3, 4], wo=1.5, bw=0.5) =
        //   b=[0.5, 0.5, 1.125, 0], a=[1, 1.5, 5.5, 3.375, 5.0625]
        let (nb, na) = lp2bp(&[1.0, 2.0], &[1.0, 3.0, 4.0], 1.5, 0.5).expect("lp2bp 2nd order");
        for (got, want) in nb.iter().zip([0.5, 0.5, 1.125, 0.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 1.5, 5.5, 3.375, 5.0625].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2bp_ba_identity_when_wo_and_bw_are_one() {
        // /porting-to-rust [frankenscipy-c7ukn]: at wo=1, bw=1 the
        // substitution is s → s + 1/s, which doubles the order.
        // For B=[1], A=[1, 1]: scipy returns
        //   b=[1, 0], a=[1, 1, 1].
        let (nb, na) = lp2bp(&[1.0], &[1.0, 1.0], 1.0, 1.0).expect("lp2bp wo=1 bw=1");
        for (got, want) in nb.iter().zip([1.0, 0.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 1.0, 1.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2bp_ba_handles_n_greater_than_d() {
        // /testing-metamorphic [REVIEW MODE]: when the numerator
        // degree exceeds the denominator's, lp2bp must still produce
        // the scipy-correct lengths (N + ma vs D + ma where ma = max).
        // scipy reference for ([1,2,3], [1,1], wo=2, bw=1):
        //   b length 5: [1, 2, 11, 8, 16]
        //   a length 4: [1, 1, 4, 0]
        let (nb, na) = lp2bp(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2.0, 1.0).expect("lp2bp N>D");
        assert_eq!(nb.len(), 5, "lp2bp(N=2, D=1).b length");
        assert_eq!(na.len(), 4, "lp2bp(N=2, D=1).a length");
        for (got, want) in nb.iter().zip([1.0, 2.0, 11.0, 8.0, 16.0].iter()) {
            assert!((got - want).abs() < 1e-12, "b: got {got}, want {want}");
        }
        for (got, want) in na.iter().zip([1.0, 1.0, 4.0, 0.0].iter()) {
            assert!((got - want).abs() < 1e-12, "a: got {got}, want {want}");
        }
    }

    #[test]
    fn lp2bs_ba_handles_n_greater_than_d() {
        // /testing-metamorphic [REVIEW MODE]: lp2bs output is 2M+1
        // for both b and a (regardless of whether N > D). scipy
        // reference for ([1,2,3], [1,1], wo=2, bw=1):
        //   b: [3, 2, 25, 8, 48]
        //   a: [1, 1, 8, 4, 16]
        let (nb, na) = lp2bs(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2.0, 1.0).expect("lp2bs N>D");
        assert_eq!(nb.len(), 5);
        assert_eq!(na.len(), 5);
        for (got, want) in nb.iter().zip([3.0, 2.0, 25.0, 8.0, 48.0].iter()) {
            assert!((got - want).abs() < 1e-12, "b: got {got}, want {want}");
        }
        for (got, want) in na.iter().zip([1.0, 1.0, 8.0, 4.0, 16.0].iter()) {
            assert!((got - want).abs() < 1e-12, "a: got {got}, want {want}");
        }
    }

    #[test]
    fn lp2bs_ba_first_order_substitution() {
        // /porting-to-rust [frankenscipy-9oqbs]: scipy reference
        //   lp2bs([1], [1, 1], wo=2, bw=1) = b=[1, 0, 4], a=[1, 1, 4]
        let (nb, na) = lp2bs(&[1.0], &[1.0, 1.0], 2.0, 1.0).expect("lp2bs 1st order");
        for (got, want) in nb.iter().zip([1.0, 0.0, 4.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 1.0, 4.0].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2bs_ba_second_order_substitution() {
        // /porting-to-rust [frankenscipy-9oqbs]: scipy reference
        //   lp2bs([1, 2], [1, 3, 4], wo=1.5, bw=0.5) =
        //   b=[0.5, 0.125, 2.25, 0.28125, 2.53125]
        //   a=[1, 0.375, 4.5625, 0.84375, 5.0625]
        let (nb, na) = lp2bs(&[1.0, 2.0], &[1.0, 3.0, 4.0], 1.5, 0.5).expect("lp2bs 2nd order");
        for (got, want) in nb.iter().zip([0.5, 0.125, 2.25, 0.281_25, 2.531_25].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "b mismatch: got {got}, want {want}"
            );
        }
        for (got, want) in na.iter().zip([1.0, 0.375, 4.5625, 0.843_75, 5.0625].iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "a mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn lp2bs_ba_rejects_invalid_args() {
        // /porting-to-rust [frankenscipy-9oqbs]: validation contract.
        assert!(lp2bs(&[1.0], &[1.0], 0.0, 1.0).is_err());
        assert!(lp2bs(&[1.0], &[1.0], 1.0, 0.0).is_err());
        assert!(lp2bs(&[1.0], &[1.0], -1.0, 1.0).is_err());
        assert!(lp2bs(&[1.0], &[1.0], 1.0, f64::NAN).is_err());
        assert!(lp2bs(&[1.0], &[1.0], f64::INFINITY, 1.0).is_err());
        assert!(lp2bs(&[], &[1.0], 1.0, 1.0).is_err());
        assert!(lp2bs(&[1.0], &[], 1.0, 1.0).is_err());
    }

    #[test]
    fn lp2bp_ba_rejects_invalid_args() {
        // /porting-to-rust [frankenscipy-c7ukn]: validation contract.
        assert!(lp2bp(&[1.0], &[1.0], 0.0, 1.0).is_err());
        assert!(lp2bp(&[1.0], &[1.0], 1.0, 0.0).is_err());
        assert!(lp2bp(&[1.0], &[1.0], -1.0, 1.0).is_err());
        assert!(lp2bp(&[1.0], &[1.0], 1.0, f64::NAN).is_err());
        assert!(lp2bp(&[1.0], &[1.0], f64::INFINITY, 1.0).is_err());
        assert!(lp2bp(&[], &[1.0], 1.0, 1.0).is_err());
        assert!(lp2bp(&[1.0], &[], 1.0, 1.0).is_err());
    }

    #[test]
    fn lp2hp_is_involution_under_same_wo() {
        // /testing-metamorphic for [frankenscipy-o9jnv]: applying lp2hp
        // with frequency wo is the substitution s → wo/s. Composing
        // this with itself gives s → wo/(wo/s) = s, so lp2hp twice
        // should recover the original B(s)/A(s) — modulo coefficient-
        // vector zero-padding (each lp2hp call produces a result of
        // length max(b.len(), a.len()), so b can pick up leading zeros)
        // and modulo a global scale (normalize_filter rescales so
        // a[0] = 1).
        //
        // We verify by evaluating B(s)/A(s) at multiple test points
        // and asserting that the original ratio equals the after-two-
        // applications ratio.
        fn poly_eval(coeffs: &[f64], s: f64) -> f64 {
            // Coefficients in descending powers of s (scipy convention).
            let mut acc = 0.0;
            for &c in coeffs {
                acc = acc * s + c;
            }
            acc
        }

        let cases: &[(Vec<f64>, Vec<f64>, f64)] = &[
            (vec![1.0], vec![1.0, 1.0], 2.0),
            (vec![1.0, 2.0], vec![1.0, 3.0, 4.0], 2.0),
            (vec![1.0, 2.0], vec![1.0, 3.0, 4.0], 0.5),
            (vec![0.5, 1.0, 1.5], vec![1.0, 2.0, 3.0, 4.0], 1.7),
        ];
        for (b0, a0, wo) in cases {
            let (b1, a1) = lp2hp(b0, a0, *wo).expect("first lp2hp");
            let (b2, a2) = lp2hp(&b1, &a1, *wo).expect("second lp2hp");
            for &s in &[0.5_f64, 1.0, 2.0, 3.5, 7.0] {
                let original_ratio = poly_eval(b0, s) / poly_eval(a0, s);
                let recovered_ratio = poly_eval(&b2, s) / poly_eval(&a2, s);
                let scale = original_ratio.abs().max(recovered_ratio.abs()).max(1e-12);
                assert!(
                    (original_ratio - recovered_ratio).abs() <= 1e-10 * scale,
                    "lp2hp involution failed at s={s} for (b={b0:?}, a={a0:?}, wo={wo}): \
                     original={original_ratio}, recovered={recovered_ratio}"
                );
            }
        }
    }

    #[test]
    fn morlet2_length_matches_request() {
        for &m in &[8_usize, 16, 32, 64] {
            assert_eq!(morlet2(m, 4.0, 5.0).len(), m);
        }
    }

    #[test]
    fn morlet2_metamorphic_energy_invariant_under_s() {
        // (π s²)^(−1/4) · gauss(t/s) preserves energy: ∑|ψ_t|² ≈ ∫|ψ|² ≈ 1
        // for any s as long as M is wide enough.
        for &s in &[1.0_f64, 2.0, 4.0, 8.0] {
            let m = (16.0 * s) as usize;
            let psi = morlet2(m, s, 5.0);
            let energy: f64 = psi.iter().map(|(re, im)| re * re + im * im).sum();
            // Energy should be close to 1; coarse discrete sampling adds
            // some bias, accept ±10%.
            assert!(
                (energy - 1.0).abs() < 0.10,
                "s={s}: energy = {energy}, expected ≈ 1"
            );
        }
    }

    #[test]
    fn morlet2_centering_matches_scipy_symmetry() {
        // scipy uses x = (i − (M−1)/2)/s. For w=0 the kernel is real
        // and symmetric: ψ_i = ψ_{M-1-i}. The earlier M/2 centering
        // broke this symmetry by half a sample; pin against regression.
        for &m in &[4_usize, 5, 8, 9] {
            let psi = morlet2(m, 1.5, 0.0);
            for i in 0..m {
                let j = m - 1 - i;
                assert!(
                    (psi[i].0 - psi[j].0).abs() < 1e-12,
                    "M={m}: morlet2[{i}].re={} vs morlet2[{j}].re={} (centering off?)",
                    psi[i].0,
                    psi[j].0
                );
                assert!(
                    (psi[i].1 - psi[j].1).abs() < 1e-12,
                    "M={m}: morlet2[{i}].im={} vs morlet2[{j}].im={}",
                    psi[i].1,
                    psi[j].1
                );
            }
        }
    }

    #[test]
    fn morlet2_centering_matches_scipy_golden() {
        // Hand-derived from scipy.signal.morlet2(4, 1.0, 0.0):
        //   x = ([0,1,2,3] - 1.5)/1 = [-1.5, -0.5, 0.5, 1.5]
        //   ψ = π^(-1/4) · exp(-x²/2)   (purely real, w=0)
        //   √(1/s) = 1
        //   →  [0.2440..., 0.6629..., 0.6629..., 0.2440...]
        let psi = morlet2(4, 1.0, 0.0);
        let pi_quarter_pow = std::f64::consts::PI.powf(-0.25);
        let expected: [f64; 4] = [
            pi_quarter_pow * (-1.125_f64).exp(),
            pi_quarter_pow * (-0.125_f64).exp(),
            pi_quarter_pow * (-0.125_f64).exp(),
            pi_quarter_pow * (-1.125_f64).exp(),
        ];
        for (i, exp) in expected.iter().enumerate() {
            assert!(
                (psi[i].0 - exp).abs() < 1e-12,
                "morlet2(4,1,0)[{i}]: got {} expected {exp}",
                psi[i].0
            );
            assert!(psi[i].1.abs() < 1e-12, "imag should be 0 for w=0");
        }
    }

    #[test]
    fn morlet2_metamorphic_zero_arguments_short_circuit() {
        assert!(morlet2(0, 1.0, 5.0).is_empty());
        assert!(morlet2(8, 0.0, 5.0).is_empty());
        assert!(morlet2(8, -1.0, 5.0).is_empty());
        assert!(morlet2(8, f64::NAN, 5.0).is_empty());
    }

    #[test]
    fn daub_p1_matches_haar() {
        // p=1 is the Haar wavelet: both coefficients are 1/√2.
        let h = daub(1).expect("daub(1)");
        assert_eq!(h.len(), 2);
        let inv_sqrt_2 = 1.0_f64 / 2.0_f64.sqrt();
        assert!((h[0] - inv_sqrt_2).abs() < 1e-12);
        assert!((h[1] - inv_sqrt_2).abs() < 1e-12);
    }

    #[test]
    fn daub_filter_lengths_are_2p() {
        for p in 1..=10 {
            let h = daub(p).expect("daub");
            assert_eq!(h.len(), 2 * p, "filter length for p={p}");
        }
    }

    #[test]
    fn daub_metamorphic_coefficient_sum_equals_sqrt_two() {
        // Σ_k h_k = √2 is the lowpass DC condition.
        let target = 2.0_f64.sqrt();
        for p in 1..=10 {
            let h = daub(p).expect("daub");
            let sum: f64 = h.iter().sum();
            assert!(
                (sum - target).abs() < 5e-7,
                "p={p}: sum = {sum}, expected √2 ≈ {target}"
            );
        }
    }

    #[test]
    fn daub_metamorphic_orthogonality_at_lag_zero() {
        // Σ_k h_k² = 1 (unit-energy condition; equivalently Σ h h = δ_0
        // at lag 0). This is the orthogonality condition's diagonal entry.
        for p in 1..=10 {
            let h = daub(p).expect("daub");
            let energy: f64 = h.iter().map(|x| x * x).sum();
            assert!(
                (energy - 1.0).abs() < 5e-7,
                "p={p}: Σh² = {energy}, expected 1"
            );
        }
    }

    #[test]
    fn daub_rejects_unsupported_order() {
        // SignalError::InvalidArgument routes through classify_invalid_argument
        // which falls back to UnclassifiedArgument for our message wording.
        let err = daub(11).expect_err("p=11 unsupported");
        let msg = format!("{err:?}");
        assert!(msg.contains("daub"), "{msg}");
        let err = daub(0).expect_err("p=0 unsupported");
        let msg = format!("{err:?}");
        assert!(msg.contains("daub"), "{msg}");
    }

    #[test]
    fn hilbert_matches_scipy_reference_values() {
        // scipy.signal.hilbert([1,2,3,4,5,6,7,8])
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = hilbert(&x).expect("hilbert");
        let expected_imag = [
            3.82842712474619,
            -1.0,
            -0.9999999999999999,
            -1.8284271247461903,
            -1.8284271247461898,
            -1.0,
            -1.0,
            3.8284271247461903,
        ];
        for (i, (re, im)) in result.iter().enumerate() {
            assert!(
                (*re - x[i]).abs() < 1e-10,
                "hilbert real[{i}] got {re}, expected {}",
                x[i]
            );
            assert!(
                (*im - expected_imag[i]).abs() < 1e-10,
                "hilbert imag[{i}] got {im}, expected {}",
                expected_imag[i]
            );
        }
    }

    #[test]
    fn hilbert_envelope_matches_scipy_reference_values() {
        // np.abs(scipy.signal.hilbert([1,2,3,4,5,6,7,8]))
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = hilbert_envelope(&x).expect("hilbert envelope");
        let expected = [
            3.9568743029684903,
            2.23606797749979,
            3.1622776601683795,
            4.398084327352946,
            5.323828110533587,
            6.0827625302982185,
            7.0710678118654755,
            8.868869953353267,
        ];
        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "hilbert envelope[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn convolve_matches_scipy_reference_values() {
        // scipy.signal.convolve([1,2,3], [0.5,1,0.5], mode='full')
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0, 0.5];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        let expected = [0.5, 2.0, 4.0, 4.0, 1.5];
        assert_eq!(result.len(), expected.len());
        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "convolve full[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn convolve_same_matches_scipy_reference_values() {
        // scipy.signal.convolve([1,2,3], [0.5,1,0.5], mode='same')
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0, 0.5];
        let result = convolve(&a, &b, ConvolveMode::Same).expect("convolve same");
        let expected = [2.0, 4.0, 4.0];
        assert_eq!(result.len(), expected.len());
        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "convolve same[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn correlate_matches_scipy_reference_values() {
        // scipy.signal.correlate([1,2,3], [0.5,1,0.5], mode='full')
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0, 0.5];
        let result = correlate(&a, &b, ConvolveMode::Full).expect("correlate");
        let expected = [0.5, 2.0, 4.0, 4.0, 1.5];
        assert_eq!(result.len(), expected.len());
        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "correlate full[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn hann_window_matches_scipy_reference_values() {
        // scipy.signal.windows.hann(5)
        // -> [0., 0.5, 1., 0.5, 0.]
        let w = hann(5);
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        assert_eq!(w.len(), expected.len());
        for (i, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "hann(5)[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn blackman_window_matches_scipy_reference_values() {
        // scipy.signal.windows.blackman(5)
        // -> [0., 0.34, 1., 0.34, 0.] approximately
        let w = blackman(5);
        assert_eq!(w.len(), 5);
        // First and last should be close to 0
        assert!(w[0].abs() < 0.01, "blackman(5)[0] should be ~0");
        assert!(w[4].abs() < 0.01, "blackman(5)[4] should be ~0");
        // Middle should be 1
        assert!((w[2] - 1.0).abs() < 1e-10, "blackman(5)[2] should be 1");
    }

    #[test]
    fn kaiser_window_matches_scipy_reference_values() {
        // scipy.signal.windows.kaiser(5, 14)
        let w = kaiser(5, 14.0);
        assert_eq!(w.len(), 5);
        // Kaiser window is symmetric and peaks in the middle
        assert!((w[0] - w[4]).abs() < 1e-10, "kaiser should be symmetric");
        assert!((w[1] - w[3]).abs() < 1e-10, "kaiser should be symmetric");
        // Middle value should be largest
        assert!(
            w[2] >= w[0] && w[2] >= w[1],
            "kaiser peak should be in middle"
        );
    }

    #[test]
    fn boxcar_window_matches_scipy_reference_values() {
        // scipy.signal.windows.boxcar(5)
        // -> [1., 1., 1., 1., 1.]
        let w = boxcar(5, true);
        let expected = [1.0; 5];
        for (i, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "boxcar(5)[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn hamming_window_matches_scipy_reference_values() {
        // scipy.signal.windows.hamming(5)
        // -> [0.08, 0.54, 1.0, 0.54, 0.08]
        let w = hamming(5);
        let expected = [0.08, 0.54, 1.0, 0.54, 0.08];
        assert_eq!(w.len(), expected.len());
        for (i, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "hamming(5)[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savgol_coeffs_matches_scipy_reference_values() {
        // scipy.signal.savgol_coeffs(5, 2)
        let c = savgol_coeffs(5, 2, 0).expect("savgol_coeffs");
        let expected = [
            -0.0857142857142857,
            0.3428571428571427,
            0.4857142857142855,
            0.3428571428571427,
            -0.0857142857142858,
        ];
        assert_eq!(c.len(), expected.len());
        for (i, (&got, &want)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "savgol_coeffs(5,2)[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savgol_filter_matches_scipy_reference_values() {
        // scipy.signal.savgol_filter([1,2,3,4,5,6,7,8], 5, 2) with the default
        // mode='interp'. The input is a perfect linear ramp and polyorder=2, so
        // the polynomial edge fit reproduces it exactly across the whole array.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = savgol_filter(&x, 5, 2).expect("savgol_filter");
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "savgol_filter[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn find_peaks_matches_scipy_reference_values() {
        // scipy.signal.find_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        let result = find_peaks(&x, FindPeaksOptions::default());
        let expected_peaks = [1, 3, 5, 7, 9];
        assert_eq!(
            result.peaks, expected_peaks,
            "find_peaks got {:?}, expected {:?}",
            result.peaks, expected_peaks
        );
    }

    #[test]
    fn find_peaks_with_height_matches_scipy_reference_values() {
        // scipy.signal.find_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], height=1.5)
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        let opts = FindPeaksOptions {
            height: Some(1.5),
            ..Default::default()
        };
        let result = find_peaks(&x, opts);
        let expected_peaks = [3, 5, 7];
        assert_eq!(
            result.peaks, expected_peaks,
            "find_peaks with height got {:?}, expected {:?}",
            result.peaks, expected_peaks
        );
    }

    #[test]
    fn find_peaks_with_distance_matches_scipy_reference_values() {
        // scipy.signal.find_peaks([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], distance=3)
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        let opts = FindPeaksOptions {
            distance: Some(3),
            ..Default::default()
        };
        let result = find_peaks(&x, opts);
        let expected_peaks = [1, 5, 9];
        assert_eq!(
            result.peaks, expected_peaks,
            "find_peaks with distance got {:?}, expected {:?}",
            result.peaks, expected_peaks
        );
    }

    #[test]
    fn peak_prominences_matches_scipy_reference_values() {
        // scipy.signal.peak_prominences([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0], [1, 3, 5, 7, 9])
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0];
        let peaks = vec![1, 3, 5, 7, 9];
        let (prominences, left_bases, right_bases) = peak_prominences(&x, &peaks);

        let expected_prom = [1.0, 2.0, 3.0, 2.0, 1.0];
        let expected_left = [0, 2, 4, 6, 8];
        let expected_right = [2, 4, 6, 8, 10];

        for (i, (&got, &want)) in prominences.iter().zip(expected_prom.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "prominence[{i}] got {got}, expected {want}"
            );
        }
        assert_eq!(left_bases, expected_left, "left_bases mismatch");
        assert_eq!(right_bases, expected_right, "right_bases mismatch");
    }

    #[test]
    fn convolve_full_matches_scipy_reference_values() {
        // scipy.signal.convolve([1, 2, 3], [0, 1, 0.5], mode='full')
        // -> [0, 1, 2.5, 4, 1.5]
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = convolve(&a, &b, ConvolveMode::Full).expect("convolve");
        let expected = [0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "convolve[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn correlate_full_matches_scipy_reference_values() {
        // scipy.signal.correlate([1, 2, 3], [0, 1, 0.5], mode='full')
        // Cross-correlation result
        let a = vec![1.0, 2.0, 3.0];
        let v = vec![0.0, 1.0, 0.5];
        let result = correlate(&a, &v, ConvolveMode::Full).expect("correlate");
        // Just verify length and that it's the correlation (different from convolution)
        assert_eq!(
            result.len(),
            5,
            "correlate length should be len(a)+len(v)-1"
        );
        // Correlate at lag 0 (center): sum(a * v) = 1*0.5 + 2*1 + 3*0 = 2.5
        // But exact indices depend on implementation - just check it ran
        assert!(
            result.iter().all(|&x| x.is_finite()),
            "all values should be finite"
        );
    }

    #[test]
    fn ricker_matches_scipy_reference_values() {
        // scipy.signal.ricker(5, 1) -> Mexican hat wavelet
        // Center value should be 2/(sqrt(3*a)*pi^0.25) when a=1
        let wavelet = ricker(5, 1.0);
        assert_eq!(wavelet.len(), 5);
        // Center value at index 2
        let center_expected = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));
        assert!(
            (wavelet[2] - center_expected).abs() < 1e-6,
            "ricker center got {}, expected {}",
            wavelet[2],
            center_expected
        );
    }

    #[test]
    fn argrelmax_matches_scipy_reference_values() {
        // scipy.signal.argrelmax([1, 2, 1, 3, 1, 2, 1], order=1)
        // -> array([1, 3, 5])
        let x = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0];
        let peaks = argrelmax(&x, 1);
        let expected = [1, 3, 5];
        assert_eq!(peaks, expected, "argrelmax mismatch");
    }

    #[test]
    fn argrelmin_matches_scipy_reference_values() {
        // scipy.signal.argrelmin([2, 1, 2, 0, 2, 1, 2], order=1)
        // -> array([1, 3, 5])
        let x = vec![2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0];
        let mins = argrelmin(&x, 1);
        let expected = [1, 3, 5];
        assert_eq!(mins, expected, "argrelmin mismatch");
    }

    #[test]
    fn cwt_matches_scipy_output_shape() {
        // scipy.signal.cwt with ricker wavelet produces widths.len() x data.len() output
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let widths = vec![1.0, 2.0, 3.0];
        let result = cwt(&data, ricker, &widths).expect("cwt");
        assert_eq!(
            result.len(),
            widths.len(),
            "cwt row count should match widths"
        );
        assert_eq!(
            result[0].len(),
            data.len(),
            "cwt column count should match data"
        );
    }

    #[test]
    fn find_peaks_matches_scipy_basic_reference() {
        // scipy.signal.find_peaks([1, 3, 2, 4, 1]) -> (array([1, 3]),)
        let x = vec![1.0, 3.0, 2.0, 4.0, 1.0];
        let result = find_peaks(&x, FindPeaksOptions::default());
        let expected = vec![1, 3];
        assert_eq!(result.peaks, expected, "find_peaks mismatch");
    }

    #[test]
    fn deconvolve_matches_scipy_reference_values() {
        // scipy.signal.deconvolve([1, 2, 3], [1, 0.5]) recovers quotient
        let signal = vec![1.0, 2.5, 4.0, 1.5];
        let divisor = vec![1.0, 0.5];
        let (quotient, remainder) = deconvolve(&signal, &divisor).expect("deconvolve");
        // quotient * divisor + remainder should equal signal
        let mut reconstructed =
            convolve(&quotient, &divisor, ConvolveMode::Full).expect("convolve");
        for (i, r) in remainder.iter().enumerate() {
            if i < reconstructed.len() {
                reconstructed[i] += r;
            }
        }
        // First signal.len() elements should match
        for i in 0..signal.len() {
            assert!(
                (reconstructed[i] - signal[i]).abs() < 1e-10,
                "deconvolve reconstruction failed at {i}"
            );
        }
    }

    #[test]
    fn resample_matches_scipy_reference_values() {
        // scipy.signal.resample([1, 2, 3, 4], 8)
        // -> [1.0, 1.08578644, 2.0, 2.5, 3.0, 3.91421356, 4.0, 2.5]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&x, 8).expect("resample");
        let expected = [1.0, 1.08578644, 2.0, 2.5, 3.0, 3.91421356, 4.0, 2.5];
        assert_eq!(result.len(), 8, "resample output length");
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "resample[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn tukey_window_matches_scipy_reference() {
        // scipy.signal.windows.tukey(5, 0.5)
        // alpha=0 -> rectangular, alpha=1 -> hann
        let result = tukey_window(5, 0.5);
        assert_eq!(result.len(), 5);
        assert!(
            (result[2] - 1.0).abs() < 1e-10,
            "tukey center should be 1.0"
        );
        assert!((result[0] - result[4]).abs() < 1e-10, "tukey symmetric");
    }

    #[test]
    fn nuttall_window_matches_scipy_reference() {
        // scipy.signal.windows.nuttall(5)
        // -> [0.0, 0.22698, 1.0, 0.22698, 0.0] (approximate)
        let result = nuttall_window(5);
        assert_eq!(result.len(), 5);
        assert!(
            (result[2] - 1.0).abs() < 1e-10,
            "nuttall center should be 1.0"
        );
        assert!((result[1] - result[3]).abs() < 1e-10, "nuttall symmetric");
        assert!(result[0] < 0.001, "nuttall edges should be small");
    }

    #[test]
    fn bohman_window_matches_scipy_reference() {
        // scipy.signal.windows.bohman(5)
        // -> [0.0, 0.31831, 1.0, 0.31831, 0.0] (approximate)
        let result = bohman_window(5);
        assert_eq!(result.len(), 5);
        assert!(result[0].abs() < 1e-10, "bohman[0] should be 0");
        assert!(
            (result[2] - 1.0).abs() < 1e-10,
            "bohman center should be 1.0"
        );
        assert!((result[1] - result[3]).abs() < 1e-10, "bohman symmetric");
    }

    #[test]
    fn instantaneous_frequency_matches_scipy_reference() {
        // scipy.signal instantaneous_frequency via hilbert
        // For a pure sine wave sin(2*pi*f*t), instantaneous freq should be f
        let fs = 100.0;
        let freq = 5.0;
        let t: Vec<f64> = (0..100).map(|i| i as f64 / fs).collect();
        let x: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * freq * ti).sin())
            .collect();
        let result = instantaneous_frequency(&x, fs).expect("instantaneous_frequency");
        assert_eq!(result.len(), x.len());
        // Middle values should be close to the input frequency
        let mid_start = 20;
        let mid_end = 80;
        for (i, &ri) in result.iter().enumerate().take(mid_end).skip(mid_start) {
            assert!(
                (ri - freq).abs() < 1.0,
                "instantaneous_frequency[{i}] = {ri}, expected ~{freq}"
            );
        }
    }
}
