use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, RuntimeMode, casp_now_unix_ms};

use crate::plan::{
    PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy, lookup_shared_plan, store_shared_plan,
};
use crate::{Normalization, TransformKind};

/// Internal complex representation used by the initial FFT surface.
pub type Complex64 = (f64, f64);

/// Backends that can serve FFT requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Naive O(n²) DFT — reference implementation.
    NaiveDft,
    /// Cooley-Tukey FFT — O(n log n) for power-of-2, mixed-radix + Bluestein for general n.
    #[default]
    CooleyTukey,
}

pub trait FftBackend {
    fn kind(&self) -> BackendKind;
    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64>;
    /// Perform transform in-place on the `data` slice.
    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool);
}

#[derive(Debug, Default)]
pub struct NaiveDftBackend;

impl FftBackend for NaiveDftBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::NaiveDft
    }

    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let mut output = input.to_vec();
        self.transform_1d_inplace(&mut output, inverse);
        output
    }

    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let input_copy = data.to_vec();
        let sign = if inverse { 1.0 } else { -1.0 };
        for (k, out) in data.iter_mut().enumerate() {
            let mut acc = (0.0, 0.0);
            for (t, &value) in input_copy.iter().enumerate() {
                let angle = sign * 2.0 * PI * (k as f64) * (t as f64) / (n as f64);
                let twiddle = (angle.cos(), angle.sin());
                acc = complex_add(acc, complex_mul(value, twiddle));
            }
            *out = acc;
        }
    }
}

static NAIVE_BACKEND: NaiveDftBackend = NaiveDftBackend;

/// Cooley-Tukey FFT backend — O(n log n) for power-of-2 sizes.
/// Falls back to Bluestein's algorithm for arbitrary sizes.
#[derive(Debug, Default)]
pub struct CooleyTukeyBackend;

impl FftBackend for CooleyTukeyBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CooleyTukey
    }

    fn transform_1d_unscaled(&self, input: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let mut output = input.to_vec();
        self.transform_1d_inplace(&mut output, inverse);
        output
    }

    fn transform_1d_inplace(&self, data: &mut [Complex64], inverse: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }
        if n.is_power_of_two() {
            cooley_tukey_radix2_inplace(data, inverse);
        } else {
            // For non-power-of-2: use Bluestein's algorithm (chirp-z transform)
            let result = bluestein_fft(data, inverse);
            data.copy_from_slice(&result);
        }
    }
}

static COOLEY_TUKEY_BACKEND: CooleyTukeyBackend = CooleyTukeyBackend;

/// Create a new shared audit ledger for synchronous FFT operations.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

type TwiddleKey = (usize, bool);
type TwiddleTable = Arc<[Complex64]>;
static TWIDDLE_CACHE: OnceLock<RwLock<HashMap<TwiddleKey, TwiddleTable>>> = OnceLock::new();

fn get_twiddle_cache() -> &'static RwLock<HashMap<TwiddleKey, TwiddleTable>> {
    TWIDDLE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn get_or_compute_twiddles(n: usize, inverse: bool) -> TwiddleTable {
    let cache = get_twiddle_cache();
    let key = (n, inverse);

    if let Some(table) = cache.read().ok().and_then(|guard| guard.get(&key).cloned()) {
        return table;
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut table = Vec::with_capacity(n);
    for k in 0..n {
        let angle = sign * 2.0 * PI * k as f64 / n as f64;
        table.push((angle.cos(), angle.sin()));
    }

    let table = Arc::<[Complex64]>::from(table);

    if let Ok(mut guard) = cache.write() {
        guard.insert(key, Arc::clone(&table));
    }

    table
}

type BluesteinKey = (usize, bool);
#[derive(Clone)]
struct BluesteinPlan {
    chirp: Vec<Complex64>,
    b_fft: Vec<Complex64>,
    m: usize,
}
static BLUESTEIN_CACHE: OnceLock<RwLock<HashMap<BluesteinKey, BluesteinPlan>>> = OnceLock::new();

fn get_bluestein_cache() -> &'static RwLock<HashMap<BluesteinKey, BluesteinPlan>> {
    BLUESTEIN_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn get_or_compute_bluestein_plan(n: usize, inverse: bool) -> BluesteinPlan {
    let cache = get_bluestein_cache();
    let key = (n, inverse);

    if let Some(plan) = cache.read().ok().and_then(|guard| guard.get(&key).cloned()) {
        return plan;
    }

    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };

    let mut chirp = Vec::with_capacity(n);
    for k in 0..n {
        let angle = sign * PI * (k as f64).powi(2) / (n as f64);
        chirp.push((angle.cos(), angle.sin()));
    }

    let mut b = vec![(0.0, 0.0); m];
    b[0] = complex_conj(chirp[0]);
    for k in 1..n {
        b[k] = complex_conj(chirp[k]);
        b[m - k] = complex_conj(chirp[k]);
    }
    cooley_tukey_radix2_inplace(&mut b, false);

    let plan = BluesteinPlan { chirp, b_fft: b, m };

    if let Ok(mut guard) = cache.write() {
        guard.insert(key, plan.clone());
    }

    plan
}

/// Radix-2 Cooley-Tukey FFT for power-of-2 lengths.
/// Iterative (bottom-up) implementation with cached twiddle factors.
fn cooley_tukey_radix2_inplace(data: &mut [Complex64], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let twiddles = get_or_compute_twiddles(n, inverse);
    cooley_tukey_radix2_inplace_with_twiddles(data, &twiddles);
}

fn cooley_tukey_radix2_inplace_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= n);
    // Bit-reversal permutation
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }

    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let stride = n / stage_len;

        let mut base = 0;
        while base < n {
            for k in 0..half {
                let even_idx = base + k;
                let odd_idx = base + k + half;
                let twiddle = twiddles[k * stride];
                let odd_val = complex_mul(data[odd_idx], twiddle);
                let even_val = data[even_idx];
                data[even_idx] = complex_add(even_val, odd_val);
                data[odd_idx] = complex_sub(even_val, odd_val);
            }
            base += stage_len;
        }
        stage_len *= 2;
    }
}

fn cooley_tukey_radix2_inplace_with_plan(
    data: &mut [Complex64],
    twiddles: &[Complex64],
    bit_reverse_swaps: &[(usize, usize)],
) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= n);
    for &(i, j) in bit_reverse_swaps {
        data.swap(i, j);
    }

    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let stride = n / stage_len;

        let mut base = 0;
        while base < n {
            for k in 0..half {
                let even_idx = base + k;
                let odd_idx = base + k + half;
                let twiddle = twiddles[k * stride];
                let odd_val = complex_mul(data[odd_idx], twiddle);
                let even_val = data[even_idx];
                data[even_idx] = complex_add(even_val, odd_val);
                data[odd_idx] = complex_sub(even_val, odd_val);
            }
            base += stage_len;
        }
        stage_len *= 2;
    }
}

fn bit_reverse_swaps(n: usize) -> Vec<(usize, usize)> {
    let log_n = n.trailing_zeros() as usize;
    let mut swaps = Vec::with_capacity(n / 2);
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            swaps.push((i, j));
        }
    }
    swaps
}

/// Bluestein's algorithm for arbitrary-length FFT.
/// Converts an n-point DFT into a circular convolution of length m (power of 2 >= 2n-1),
/// then uses radix-2 FFT on the padded data.
/// Uses cached chirp and pre-FFT'd b sequence for repeated transforms of the same size.
fn bluestein_fft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    let plan = get_or_compute_bluestein_plan(n, inverse);
    let m = plan.m;

    let mut a = vec![(0.0, 0.0); m];
    for (slot, (&sample, &chirp)) in a.iter_mut().zip(input.iter().zip(&plan.chirp)) {
        *slot = complex_mul(sample, chirp);
    }

    cooley_tukey_radix2_inplace(&mut a, false);
    for (value, &b_fft) in a.iter_mut().zip(&plan.b_fft) {
        *value = complex_mul(*value, b_fft);
    }
    cooley_tukey_radix2_inplace(&mut a, true);

    let inv_m = 1.0 / m as f64;
    a.iter()
        .zip(&plan.chirp)
        .map(|(&value, &chirp)| complex_mul(chirp, complex_scale(value, inv_m)))
        .collect()
}

/// Reverse the lower `bits` bits of `x`.
/// Specialized real FFT: pack N real values into N/2 complex, FFT, then unpack.
/// Returns N/2 + 1 complex values (the non-redundant half of the spectrum).
fn real_fft_specialized(input: &[f64], backend: &dyn FftBackend) -> Vec<Complex64> {
    let n = input.len();
    let half = n / 2;

    // Pack: z[k] = x[2k] + i*x[2k+1]
    let mut packed = Vec::with_capacity(half);
    for k in 0..half {
        packed.push((input[2 * k], input[2 * k + 1]));
    }

    // FFT of half-length complex sequence
    let z = backend.transform_1d_unscaled(&packed, false);
    let twiddles = get_or_compute_twiddles(n, false);

    // Unpack: X[k] = (Z[k] + conj(Z[N/2-k]))/2 - i*exp(-2πik/N)*(Z[k] - conj(Z[N/2-k]))/2
    let mut result = Vec::with_capacity(half + 1);
    for k in 0..=half {
        let zk = if k < half { z[k] } else { z[0] };
        let zn_k = if k == 0 || k == half {
            z[0]
        } else {
            z[half - k]
        };
        let zn_k_conj = complex_conj(zn_k);

        let even = (0.5 * (zk.0 + zn_k_conj.0), 0.5 * (zk.1 + zn_k_conj.1));
        let odd = (0.5 * (zk.0 - zn_k_conj.0), 0.5 * (zk.1 - zn_k_conj.1));

        // Twiddle: exp(-2πik/N)
        let twiddle = twiddles[k];
        let odd_tw = complex_mul(odd, twiddle);

        // Multiply by -i: (a, b) -> (b, -a)
        let odd_final = (odd_tw.1, -odd_tw.0);

        result.push(complex_add(even, odd_final));
    }

    result
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - bits)
}

fn complex_sub(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

/// Worker control policy for transform execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerPolicy {
    /// Let backend/runtime pick an execution width.
    #[default]
    Auto,
    /// Require an exact worker count.
    Exact(usize),
    /// Upper-bound worker count.
    Max(usize),
}

/// Common options shared by FFT transform entrypoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftOptions {
    pub mode: RuntimeMode,
    pub normalization: Normalization,
    pub workers: WorkerPolicy,
    pub backend: BackendKind,
    pub check_finite: bool,
    pub overwrite_input: bool,
}

impl Default for FftOptions {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Strict,
            normalization: Normalization::Backward,
            workers: WorkerPolicy::Auto,
            backend: BackendKind::default(),
            check_finite: false,
            overwrite_input: false,
        }
    }
}

impl FftOptions {
    #[must_use]
    pub fn with_mode(mut self, mode: RuntimeMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    #[must_use]
    pub fn with_workers(mut self, workers: WorkerPolicy) -> Self {
        self.workers = workers;
        self
    }

    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub fn with_check_finite(mut self, check_finite: bool) -> Self {
        self.check_finite = check_finite;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FftError {
    InvalidShape { detail: &'static str },
    InvalidAxes { detail: &'static str },
    InvalidWorkers { requested: usize },
    LengthMismatch { expected: usize, actual: usize },
    NonPositiveSampleSpacing,
    NonFiniteInput,
}

impl Display for FftError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape { detail } => write!(f, "invalid shape: {detail}"),
            Self::InvalidAxes { detail } => write!(f, "invalid axes: {detail}"),
            Self::InvalidWorkers { requested } => write!(f, "invalid worker count: {requested}"),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {expected}, got {actual}")
            }
            Self::NonPositiveSampleSpacing => {
                write!(f, "sample spacing must be finite and greater than zero")
            }
            Self::NonFiniteInput => write!(f, "non-finite input rejected by policy"),
        }
    }
}

impl std::error::Error for FftError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformTrace {
    pub operation_id: String,
    pub kind: TransformKind,
    pub direction: &'static str,
    pub n: usize,
    pub backend: BackendKind,
    pub plan_cache_hit: bool,
    pub mode: RuntimeMode,
    pub timing_ns: u128,
}

impl TransformTrace {
    #[must_use]
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"operation_id\":\"{}\",\"operation\":\"{}\",\"direction\":\"{}\",\"n\":{},\"backend\":\"{}\",\"plan_cache_hit\":{},\"mode\":\"{}\",\"timing_ns\":{}}}",
            self.operation_id,
            transform_kind_name(self.kind),
            self.direction,
            self.n,
            backend_kind_name(self.backend),
            self.plan_cache_hit,
            runtime_mode_name(self.mode),
            self.timing_ns,
        )
    }
}

static TRACE_LOG: OnceLock<Mutex<Vec<TransformTrace>>> = OnceLock::new();
static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

fn trace_log() -> &'static Mutex<Vec<TransformTrace>> {
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

fn next_operation_id() -> String {
    let next = OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("fft-op-{next:016x}")
}

/// Acquire the trace-log guard, recovering from a poisoned mutex so
/// trace entries are preserved across panics.
/// Resolves [frankenscipy-wtxpg].
fn lock_trace_log_or_recover() -> std::sync::MutexGuard<'static, Vec<TransformTrace>> {
    let log = trace_log();
    match log.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            log.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn record_trace(trace: TransformTrace) {
    lock_trace_log_or_recover().push(trace);
}

#[must_use]
pub fn take_transform_traces() -> Vec<TransformTrace> {
    let mut log = lock_trace_log_or_recover();
    let mut out = Vec::with_capacity(log.len());
    std::mem::swap(&mut *log, &mut out);
    out
}

/// 1D forward complex FFT.
pub fn fft(input: &[Complex64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    fft_impl(input, options, None)
}

/// 1D forward complex FFT with audit logging.
pub fn fft_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    fft_impl(input, options, Some(audit_ledger))
}

/// 1D inverse complex FFT.
pub fn ifft(input: &[Complex64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    ifft_impl(input, options, None)
}

/// 1D inverse complex FFT with audit logging.
pub fn ifft_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ifft_impl(input, options, Some(audit_ledger))
}

/// 1D real-input FFT.
pub fn rfft(input: &[f64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    rfft_impl(input, options, None)
}

/// 1D real-input FFT with audit logging.
pub fn rfft_with_audit(
    input: &[f64],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    rfft_impl(input, options, Some(audit_ledger))
}

/// 1D inverse real FFT.
pub fn irfft(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    irfft_impl(input, output_len, options, None)
}

/// 1D inverse real FFT with audit logging.
pub fn irfft_with_audit(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    irfft_impl(input, output_len, options, Some(audit_ledger))
}

fn fft_impl(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Fft, input, options, false, audit_ledger)
}

fn ifft_impl(
    input: &[Complex64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_1d(TransformKind::Ifft, input, options, true, audit_ledger)
}

fn rfft_impl(
    input: &[f64],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        TransformKind::Rfft,
        vec![input.len()],
        vec![0],
        options.normalization,
        true,
    );
    let plan_cache_hit = touch_plan_cache(&key, input.len());

    let started = Instant::now();
    let mut output = real_fft_unscaled(input, backend);
    apply_normalization(&mut output, options.normalization, input.len(), false);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind: TransformKind::Rfft,
        direction: "forward",
        n: input.len(),
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn irfft_impl(
    input: &[Complex64],
    output_len: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let n = output_len.unwrap_or_else(|| {
        if input.len() == 1 {
            1
        } else {
            input.len().saturating_sub(1).saturating_mul(2)
        }
    });
    if n == 0 {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "invalid_output_len",
            "rejected: output_len cannot be zero",
        );
        return Err(FftError::InvalidShape {
            detail: "output_len cannot be zero",
        });
    }

    let expected_len = n / 2 + 1;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected spectrum length {expected_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        TransformKind::Irfft,
        vec![n],
        vec![0],
        options.normalization,
        true,
    );
    let plan_cache_hit = touch_plan_cache(&key, n);

    let started = Instant::now();
    let mut output = real_ifft_unscaled(input, n, backend);
    apply_real_normalization(&mut output, options.normalization, n, true);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind: TransformKind::Irfft,
        direction: "inverse",
        n,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

/// 2D forward complex FFT via row/column decomposition.
pub fn fft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    fft2_impl(input, &dims, options, None)
}

/// 2D forward complex FFT with audit logging.
pub fn fft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    fft2_impl(input, &dims, options, Some(audit_ledger))
}

/// 2D inverse complex FFT via row/column decomposition.
pub fn ifft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ifft2_impl(input, &dims, options, None)
}

/// 2D inverse complex FFT with audit logging.
pub fn ifft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ifft2_impl(input, &dims, options, Some(audit_ledger))
}

/// N-dimensional forward complex FFT.
pub fn fftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    fftn_impl(input, shape, options, None)
}

/// N-dimensional forward complex FFT with audit logging.
pub fn fftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    fftn_impl(input, shape, options, Some(audit_ledger))
}

/// N-dimensional inverse complex FFT.
///
/// Matches `scipy.fft.ifftn(x, s)`.
pub fn ifftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    ifftn_impl(input, shape, options, None)
}

/// N-dimensional inverse complex FFT with audit logging.
pub fn ifftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ifftn_impl(input, shape, options, Some(audit_ledger))
}

/// N-dimensional real-input FFT.
///
/// Matches `scipy.fft.rfftn(x, s)`.
pub fn rfftn(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, None)
}

/// N-dimensional real-input FFT with audit logging.
pub fn rfftn_with_audit(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, Some(audit_ledger))
}

fn fft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Fft2,
        input,
        shape,
        options,
        false,
        audit_ledger,
    )
}

fn ifft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Ifft2,
        input,
        shape,
        options,
        true,
        audit_ledger,
    )
}

fn fftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Fftn,
        input,
        shape,
        options,
        false,
        audit_ledger,
    )
}

fn ifftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_complex_nd(
        TransformKind::Ifftn,
        input,
        shape,
        options,
        true,
        audit_ledger,
    )
}

fn rfftn_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    run_real_nd_forward(TransformKind::Rfftn, input, shape, options, audit_ledger)
}

/// Discrete Cosine Transform (Type II).
///
/// Matches `scipy.fft.dct(x, type=2)`.
/// DCT-II: `X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(π*(2n+1)*k / (2N))`
///
/// Computed via FFT of a reordered and mirrored sequence.
pub fn dct(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // Compute DCT-II via FFT of a mirrored 2N-length sequence
    let mut extended = vec![(0.0, 0.0); 2 * n];
    for i in 0..n {
        extended[i] = (input[i], 0.0);
    }
    for i in 0..n {
        extended[2 * n - 1 - i] = (input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    // Extract DCT coefficients: X[k] = Re(spectrum[k] * exp(-i*π*k/(2N)))
    let mut result = Vec::with_capacity(n);
    for (k, &sk) in spectrum.iter().enumerate().take(n) {
        let angle = -PI * k as f64 / (2.0 * n as f64);
        let twiddle = (angle.cos(), angle.sin());
        let val = complex_mul(sk, twiddle);
        result.push(val.0); // take real part
    }
    // br-yjas slice 2 (DCT-II): scipy normalization. Ortho scales all
    // entries by 1/sqrt(2N) plus an extra 1/sqrt(2) on the FIRST entry.
    // Forward applies a uniform 1/(2N).
    let nf = n as f64;
    match options.normalization {
        Normalization::Backward => {}
        Normalization::Ortho => {
            let s = 1.0 / (2.0 * nf).sqrt();
            for v in result.iter_mut() {
                *v *= s;
            }
            if let Some(first) = result.first_mut() {
                *first *= 1.0 / 2.0_f64.sqrt();
            }
        }
        Normalization::Forward => {
            let s = 1.0 / (2.0 * nf);
            for v in result.iter_mut() {
                *v *= s;
            }
        }
    }
    Ok(result)
}

/// Inverse Discrete Cosine Transform (Type III, the inverse of DCT-II).
///
/// Matches `scipy.fft.idct(x, type=2)`.
pub fn idct(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    let nf = n as f64;

    // Pre-scale the input to invert whichever forward normalization
    // dct() applied. Resolves [frankenscipy-v0vm5]: idct previously
    // hardcoded 1/(2N) regardless of options.normalization, so
    // idct(dct(x, ortho), ortho) returned x/(2N·N·...) instead of x.
    //
    //   Backward (default):  dct returned 2·sum  → no pre-scale needed.
    //   Ortho:               dct returned 2·sum / √(2N) (with extra
    //                         /√2 on [0]) → pre-scale input by √(2N)
    //                         (and ×√2 on [0]) to recover 2·sum.
    //   Forward:             dct returned (2·sum)/(2N) = sum/N → pre-
    //                         scale input by 2N.
    //
    // After pre-scaling, the existing 1/(2N) tail-scale recovers x.
    let scaled_input: Vec<f64> = match options.normalization {
        Normalization::Backward => input.to_vec(),
        Normalization::Ortho => {
            let s = (2.0 * nf).sqrt();
            let mut v: Vec<f64> = input.iter().map(|x| x * s).collect();
            if let Some(first) = v.first_mut() {
                *first *= 2.0_f64.sqrt();
            }
            v
        }
        Normalization::Forward => input.iter().map(|x| x * (2.0 * nf)).collect(),
    };

    // DCT-III: x[n] = X[0]/(2N) + (1/N) * sum_{k=1}^{N-1} X[k] * cos(π*k*(2n+1)/(2N))
    // Compute via inverse of the DCT-II process
    let backend = resolve_backend(options.backend);

    // Prepare: multiply by twiddle and create Hermitian-symmetric spectrum
    let mut spectrum = vec![(0.0, 0.0); 2 * n];
    for k in 0..n {
        let angle = PI * k as f64 / (2.0 * nf);
        let twiddle = (angle.cos(), angle.sin());
        spectrum[k] = complex_mul((scaled_input[k], 0.0), twiddle);
    }
    // Hermitian symmetry for real output
    for k in 1..n {
        spectrum[2 * n - k] = complex_conj(spectrum[k]);
    }

    let time_domain = backend.transform_1d_unscaled(&spectrum, true);

    // Extract: take first N values, scale by 1/(2N)
    let scale = 1.0 / (2.0 * nf);
    let result: Vec<f64> = time_domain.iter().take(n).map(|v| v.0 * scale).collect();

    Ok(result)
}

/// Discrete Cosine Transform Type I.
///
/// DCT-I: `X[k] = x[0] + (-1)^k * x[N-1] + 2 * Σ_{n=1}^{N-2} x[n] * cos(πnk/(N-1))`
///
/// Matches `scipy.fft.dct(x, type=1)`.
pub fn dct_i(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    if n == 1 {
        return Ok(vec![input[0]]);
    }

    let compute_backward = |src: &[f64]| -> Vec<f64> {
        let m = 2 * n - 2;
        let mut extended = vec![(0.0, 0.0); m];
        for i in 0..n {
            extended[i] = (src[i], 0.0);
        }
        for i in 1..n - 1 {
            extended[m - i] = (src[i], 0.0);
        }
        let backend = resolve_backend(options.backend);
        let spectrum = backend.transform_1d_unscaled(&extended, false);
        spectrum.into_iter().take(n).map(|v| v.0).collect()
    };

    let nm1 = (n - 1) as f64;
    match options.normalization {
        Normalization::Backward => Ok(compute_backward(input)),
        Normalization::Forward => {
            let mut result = compute_backward(input);
            let s = 1.0 / (2.0 * nm1);
            for v in result.iter_mut() {
                *v *= s;
            }
            Ok(result)
        }
        Normalization::Ortho => {
            // br-mhnr: scipy DCT-I ortho. Pre-scale x[0] and x[N-1] by
            // sqrt(2), compute backward DCT-I, post-scale result[0] and
            // result[N-1] by 1/sqrt(2), then scale all by 1/sqrt(2*(N-1)).
            let sqrt2 = 2.0_f64.sqrt();
            let mut adjusted = input.to_vec();
            adjusted[0] *= sqrt2;
            adjusted[n - 1] *= sqrt2;
            let mut result = compute_backward(&adjusted);
            result[0] /= sqrt2;
            result[n - 1] /= sqrt2;
            let s = 1.0 / (2.0 * nm1).sqrt();
            for v in result.iter_mut() {
                *v *= s;
            }
            Ok(result)
        }
    }
}

/// Discrete Cosine Transform Type III.
///
/// DCT-III: `x[n] = X[0]/2 + Σ_{k=1}^{N-1} X[k] * cos(πk(2n+1)/(2N))`
///
/// This is the inverse of DCT-II (up to scaling).
/// Matches `scipy.fft.dct(x, type=3)`.
pub fn dct_iii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    let nf = n as f64;

    // br-mhnr: DCT-III normalization. fsci's `idct` always returns the
    // 1/(2N)-scaled inverse, so dct_iii_backward = 2N * idct(x). For
    // forward we want dct_iii_backward / (2N) = idct(x). For ortho we
    // pre-scale input[0] by sqrt(2), compute backward DCT-III, then scale
    // all entries by 1/sqrt(2N) — this matches scipy's
    //   y[k] = sqrt(1/N)*x[0] + sqrt(2/N)*sum_{n>=1} x[n]*cos(...)
    let backward_opts = options.clone().with_normalization(Normalization::Backward);
    match options.normalization {
        Normalization::Backward => {
            let idct_result = idct(input, &backward_opts)?;
            Ok(idct_result.into_iter().map(|v| v * 2.0 * nf).collect())
        }
        Normalization::Forward => idct(input, &backward_opts),
        Normalization::Ortho => {
            let mut adjusted = input.to_vec();
            adjusted[0] *= 2.0_f64.sqrt();
            let idct_result = idct(&adjusted, &backward_opts)?;
            // dct_iii_backward(adjusted) * (1/sqrt(2N)) = (2N * idct) / sqrt(2N)
            //                                          = idct * sqrt(2N)
            let scale = (2.0 * nf).sqrt();
            Ok(idct_result.into_iter().map(|v| v * scale).collect())
        }
    }
}

/// Discrete Cosine Transform Type IV.
///
/// DCT-IV: `X[k] = 2 * Σ_{n=0}^{N-1} x[n] * cos(π(2n+1)(2k+1)/(4N))`
///
/// Matches `scipy.fft.dct(x, type=4)`.
pub fn dct_iv(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DCT-IV via FFT of length 8N
    let m = 8 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // Real part of bin 2k+1
        result.push(spectrum[2 * k + 1].0);
    }
    // br-yjas: scipy normalization. DCT-IV is orthonormal up to a
    // 1/sqrt(2N) factor; "ortho" applies that factor and "forward"
    // applies 1/(2N).
    let scale = match options.normalization {
        Normalization::Backward => 1.0,
        Normalization::Ortho => 1.0 / (2.0 * n as f64).sqrt(),
        Normalization::Forward => 1.0 / (2.0 * n as f64),
    };
    if (scale - 1.0).abs() > f64::EPSILON {
        for v in result.iter_mut() {
            *v *= scale;
        }
    }
    Ok(result)
}

/// Discrete Sine Transform Type I.
///
/// DST-I: `X[k] = 2 * Σ_{n=0}^{N-1} x[n] * sin(π(n+1)(k+1)/(N+1))`
///
/// Matches `scipy.fft.dst(x, type=1)`.
pub fn dst_i(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-I via FFT of length 2N+2
    let m = 2 * n + 2;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[i + 1] = (input[i], 0.0);
        extended[m - i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for val in spectrum.iter().take(n + 1).skip(1) {
        result.push(-val.1); // -Im part corresponds to 2 * sum
    }
    // br-0wg1: DST-I normalization is uniform — ortho divides by
    // sqrt(2*(N+1)), forward by 2*(N+1).
    let nf = (n + 1) as f64;
    let scale = match options.normalization {
        Normalization::Backward => 1.0,
        Normalization::Ortho => 1.0 / (2.0 * nf).sqrt(),
        Normalization::Forward => 1.0 / (2.0 * nf),
    };
    if (scale - 1.0).abs() > f64::EPSILON {
        for v in result.iter_mut() {
            *v *= scale;
        }
    }
    Ok(result)
}

/// Discrete Sine Transform Type II.
///
/// DST-II: `X[k] = Σ_{n=0}^{N-1} x[n] * sin(π(2n+1)(k+1)/(2N))`
///
/// Matches `scipy.fft.dst(x, type=2)`.
pub fn dst_ii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-II via FFT of length 4N.
    // X[k] = sum_{n=0}^{N-1} x[n] * sin(pi*(2n+1)*(k+1)/(2N))
    let m = 4 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // Imaginary part of bin k+1 corresponds to -2 * sum_{n=0}^{N-1} x[n] * sin(...)
        // Scale by -1.0 to get 2 * sum.
        result.push(-spectrum[k + 1].1);
    }
    // br-yjas slice 2 (DST-II): scipy normalization. Ortho scales all
    // entries by 1/sqrt(2N) plus an extra 1/sqrt(2) on the LAST entry
    // (scipy DST-II convention; mirror of DCT-II's first-entry adj).
    // Forward applies a uniform 1/(2N).
    let nf = n as f64;
    match options.normalization {
        Normalization::Backward => {}
        Normalization::Ortho => {
            let s = 1.0 / (2.0 * nf).sqrt();
            for v in result.iter_mut() {
                *v *= s;
            }
            if let Some(last) = result.last_mut() {
                *last *= 1.0 / 2.0_f64.sqrt();
            }
        }
        Normalization::Forward => {
            let s = 1.0 / (2.0 * nf);
            for v in result.iter_mut() {
                *v *= s;
            }
        }
    }
    Ok(result)
}

/// Discrete Sine Transform Type III.
///
/// DST-III: `x[n] = (-1)^n * X[N-1]/2 + Σ_{k=0}^{N-2} X[k] * sin(π(k+1)(2n+1)/(2N))`
///
/// This is the inverse of DST-II (up to scaling).
/// Matches `scipy.fft.dst(x, type=3)`.
pub fn dst_iii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    let nf = n as f64;

    // br-mhnr: DST-III normalization. Backward is the raw 2*sum result;
    // forward divides by 2N; ortho pre-scales x[N-1] by sqrt(2) (the lone
    // unpaired term in scipy's ortho formula) then applies 1/sqrt(2N).
    let compute_backward = |src: &[f64]| -> Vec<f64> {
        let m = 4 * n;
        let mut extended = vec![(0.0, 0.0); m];
        for k in 0..n - 1 {
            extended[k + 1] = (0.0, -src[k]);
            extended[m - k - 1] = (0.0, src[k]);
        }
        let last_val = src[n - 1] * 0.5;
        extended[n] = (0.0, -last_val);
        extended[m - n] = (0.0, last_val);

        let backend = resolve_backend(options.backend);
        let time_domain = backend.transform_1d_unscaled(&extended, true);

        (0..n).map(|i| time_domain[2 * i + 1].0).collect()
    };

    match options.normalization {
        Normalization::Backward => Ok(compute_backward(input)),
        Normalization::Forward => {
            let mut result = compute_backward(input);
            let s = 1.0 / (2.0 * nf);
            for v in result.iter_mut() {
                *v *= s;
            }
            Ok(result)
        }
        Normalization::Ortho => {
            let mut adjusted = input.to_vec();
            adjusted[n - 1] *= 2.0_f64.sqrt();
            let mut result = compute_backward(&adjusted);
            let s = 1.0 / (2.0 * nf).sqrt();
            for v in result.iter_mut() {
                *v *= s;
            }
            Ok(result)
        }
    }
}

/// Discrete Sine Transform Type IV.
///
/// DST-IV: `X[k] = 2 * Σ_{n=0}^{N-1} x[n] * sin(π(2n+1)(2k+1)/(4N))`
///
/// Matches `scipy.fft.dst(x, type=4)`.
pub fn dst_iv(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();
    // DST-IV via FFT of length 8N
    let m = 8 * n;
    let mut extended = vec![(0.0, 0.0); m];
    for i in 0..n {
        extended[2 * i + 1] = (input[i], 0.0);
        extended[m - 2 * i - 1] = (-input[i], 0.0);
    }

    let backend = resolve_backend(options.backend);
    let spectrum = backend.transform_1d_unscaled(&extended, false);

    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        // -Imaginary part of bin 2k+1
        result.push(-spectrum[2 * k + 1].1);
    }
    // br-yjas: scipy normalization for DST-IV (mirror of DCT-IV).
    let scale = match options.normalization {
        Normalization::Backward => 1.0,
        Normalization::Ortho => 1.0 / (2.0 * n as f64).sqrt(),
        Normalization::Forward => 1.0 / (2.0 * n as f64),
    };
    if (scale - 1.0).abs() > f64::EPSILON {
        for v in result.iter_mut() {
            *v *= scale;
        }
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// N-Dimensional DCT/DST
// ═══════════════════════════════════════════════════════════════════════════

/// N-dimensional Discrete Cosine Transform.
///
/// Matches `scipy.fft.dctn`. Applies the DCT-II transform along each axis.
///
/// # Arguments
/// * `input` - Input array (flattened, row-major order)
/// * `shape` - Shape of the N-D array
/// * `options` - FFT options (normalization, etc.)
///
/// # Returns
/// DCT coefficients (flattened, same shape as input)
pub fn dctn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply DCT along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in (0..ndim).rev() {
        data = apply_dct_along_axis(&data, shape, axis, options, false)?;
    }

    Ok(data)
}

/// Inverse N-dimensional Discrete Cosine Transform.
///
/// Matches `scipy.fft.idctn`. Applies the inverse DCT-II (DCT-III) along each axis.
pub fn idctn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply IDCT along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in 0..ndim {
        data = apply_dct_along_axis(&data, shape, axis, options, true)?;
    }

    Ok(data)
}

/// N-dimensional Discrete Sine Transform.
///
/// Matches `scipy.fft.dstn`. Applies the DST-II transform along each axis.
pub fn dstn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply DST along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in (0..ndim).rev() {
        data = apply_dst_along_axis(&data, shape, axis, options, false)?;
    }

    Ok(data)
}

/// Inverse N-dimensional Discrete Sine Transform.
///
/// Matches `scipy.fft.idstn`. Applies the inverse DST-II (DST-III) along each axis.
pub fn idstn(input: &[f64], shape: &[usize], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total: usize = shape.iter().product();
    if total != input.len() {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;

    // Apply IDST along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in 0..ndim {
        data = apply_dst_along_axis(&data, shape, axis, options, true)?;
    }

    Ok(data)
}

/// Apply 1-D DCT along a specific axis of an N-D array.
fn apply_dct_along_axis(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<f64>, FftError> {
    let ndim = shape.len();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(data.to_vec());
    }

    // Calculate strides for the input array
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let outer_size = total / axis_len;
    let mut result = vec![0.0; total];

    // For each "fiber" along the axis
    let mut temp = vec![0.0; axis_len];
    let mut temp_out = vec![0.0; axis_len];

    for outer_idx in 0..outer_size {
        // Convert outer_idx to multi-index (excluding axis)
        let mut multi_idx = vec![0usize; ndim];
        let mut remaining = outer_idx;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            multi_idx[d] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Extract the fiber
        for (i, value) in temp.iter_mut().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            *value = data[flat_idx];
        }

        // Apply DCT or IDCT
        let transformed = if inverse {
            idct(&temp, options)?
        } else {
            dct(&temp, options)?
        };
        temp_out.copy_from_slice(&transformed);

        // Store back
        for (i, &value) in temp_out.iter().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            result[flat_idx] = value;
        }
    }

    Ok(result)
}

/// Apply 1-D DST along a specific axis of an N-D array.
fn apply_dst_along_axis(
    data: &[f64],
    shape: &[usize],
    axis: usize,
    options: &FftOptions,
    inverse: bool,
) -> Result<Vec<f64>, FftError> {
    let ndim = shape.len();
    let axis_len = shape[axis];

    if axis_len == 0 {
        return Ok(data.to_vec());
    }

    // Calculate strides for the input array
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let outer_size = total / axis_len;
    let mut result = vec![0.0; total];

    // For each "fiber" along the axis
    let mut temp = vec![0.0; axis_len];
    let mut temp_out = vec![0.0; axis_len];

    for outer_idx in 0..outer_size {
        // Convert outer_idx to multi-index (excluding axis)
        let mut multi_idx = vec![0usize; ndim];
        let mut remaining = outer_idx;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            multi_idx[d] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Extract the fiber
        for (i, value) in temp.iter_mut().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            *value = data[flat_idx];
        }

        // Apply DST or IDST (DST-II forward, DST-III inverse)
        let mut transformed = if inverse {
            dst_iii(&temp, options)?
        } else {
            dst_ii(&temp, options)?
        };
        // dst_iii in Backward mode returns the unnormalized DST-III
        // (matching scipy.fft.dst(type=3)); idstn must apply the
        // 1/(2N) per-axis factor that scipy.fft.idst injects so that
        // idstn(dstn(x)) = x. The Forward and Ortho variants already
        // include the correct scaling inside dst_ii/dst_iii.
        if inverse && options.normalization == Normalization::Backward {
            let scale = 1.0 / (2.0 * axis_len as f64);
            for v in transformed.iter_mut() {
                *v *= scale;
            }
        }
        temp_out.copy_from_slice(&transformed);

        // Store back
        for (i, &value) in temp_out.iter().enumerate().take(axis_len) {
            multi_idx[axis] = i;
            let flat_idx: usize = multi_idx
                .iter()
                .zip(strides.iter())
                .map(|(m, s)| m * s)
                .sum();
            result[flat_idx] = value;
        }
    }

    Ok(result)
}

/// Compute the analytic signal using the Hilbert transform.
///
/// Matches `scipy.signal.hilbert(x)`. Returns the analytic signal
/// whose real part is the original signal and imaginary part is the
/// Hilbert transform.
pub fn hilbert(input: &[f64], options: &FftOptions) -> Result<Vec<Complex64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // FFT the real input
    let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
    let backend = resolve_backend(options.backend);
    let mut spectrum = backend.transform_1d_unscaled(&complex_input, false);

    // Create step function h[k]:
    // Even N: h[0]=1, h[1..N/2]=2, h[N/2]=1, h[N/2+1..N]=0
    // Odd N:  h[0]=1, h[1..(N+1)/2]=2, h[(N+1)/2..N]=0
    if n > 1 {
        if n.is_multiple_of(2) {
            // Even: bins 1..N/2 doubled, Nyquist (N/2) unchanged, rest zeroed
            for sk in spectrum.iter_mut().take(n / 2).skip(1) {
                *sk = complex_scale(*sk, 2.0);
            }
            for sk in spectrum.iter_mut().skip(n / 2 + 1) {
                *sk = (0.0, 0.0);
            }
        } else {
            // Odd: bins 1..(N+1)/2 doubled (no Nyquist), rest zeroed
            let half_pos = n.div_ceil(2);
            for sk in spectrum.iter_mut().take(half_pos).skip(1) {
                *sk = complex_scale(*sk, 2.0);
            }
            for sk in spectrum.iter_mut().skip(half_pos) {
                *sk = (0.0, 0.0);
            }
        }
    }

    // IFFT to get analytic signal
    let mut analytic = backend.transform_1d_unscaled(&spectrum, true);
    // Apply 1/N normalization (Backward mode)
    let inv_n = 1.0 / n as f64;
    for v in &mut analytic {
        *v = complex_scale(*v, inv_n);
    }

    Ok(analytic)
}

/// Compute optimal offset for Fast Hankel Transform.
///
/// Matches `scipy.fft.fhtoffset(dln, mu, initial=0.0, bias=0.0)`.
///
/// Returns the optimal offset parameter for the FHT to minimize ringing.
///
/// # Arguments
/// * `dln` - Log-spacing of the input array (∆ln r)
/// * `mu` - Order of the Hankel transform (J_μ Bessel function order)
/// * `initial` - Initial guess for the offset (default 0.0)
/// * `bias` - Bias parameter (default 0.0)
pub fn fhtoffset(dln: f64, mu: f64, initial: f64, bias: f64) -> f64 {
    // Low-ringing offset (Hamilton 2000): the offset nearest `initial` for
    // which the FHT kernel U_μ has unit modulus at the Nyquist frequency,
    // i.e. arg(U_μ) is a multiple of π there. With the kernel
    //   U_μ(x) = 2^x · Γ((μ+1+x)/2) / Γ((μ+1−x)/2),
    // the condition reduces to a fractional-part adjustment of the
    // log-gamma phases. Faithful port of scipy.fft.fhtoffset.
    use std::f64::consts::{LN_2, PI};
    let (lnkr, q) = (initial, bias);
    let xp = (mu + 1.0 + q) / 2.0;
    let xm = (mu + 1.0 - q) / 2.0;
    let y = PI / (2.0 * dln);
    let zp = complex_ln_gamma((xp, y));
    let zm = complex_ln_gamma((xm, y));
    let arg = (LN_2 - lnkr) / dln + (zp.1 + zm.1) / PI;
    lnkr + (arg - arg.round()) * dln
}

/// Fast Hankel Transform.
///
/// Matches `scipy.fft.fht(a, dln, mu, offset=0.0, bias=0.0)`.
///
/// Compute the discrete Hankel transform of a logarithmically spaced
/// periodic sequence.
///
/// # Arguments
/// * `input` - Input array (logarithmically spaced)
/// * `dln` - Uniform logarithmic spacing ∆ln r
/// * `mu` - Order of the Hankel transform (Bessel function order J_μ)
/// * `offset` - Offset parameter (use `fhtoffset` to compute optimal value)
/// * `bias` - Power-law bias exponent
/// * `options` - FFT options
///
/// # Returns
/// The transformed array
pub fn fht(
    input: &[f64],
    dln: f64,
    mu: f64,
    offset: f64,
    bias: f64,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // Bias the input: a_q(r) = a(r) · (r/r_c)^{−q}.
    let mut a = input.to_vec();
    if bias != 0.0 {
        let j_c = (n as f64 - 1.0) / 2.0;
        for (j, v) in a.iter_mut().enumerate() {
            *v *= (-bias * (j as f64 - j_c) * dln).exp();
        }
    }

    let u = fhtcoeff(n, dln, mu, offset, bias, false);
    let mut out = fhtq(&a, &u, false, options)?;

    // Un-bias the output: A(k) = A_q(k) · (k/k_c)^{−q} · (k_c·r_c)^{−q}.
    if bias != 0.0 {
        let j_c = (n as f64 - 1.0) / 2.0;
        for (j, v) in out.iter_mut().enumerate() {
            *v *= (-bias * ((j as f64 - j_c) * dln + offset)).exp();
        }
    }

    Ok(out)
}

/// Inverse Fast Hankel Transform.
///
/// Matches `scipy.fft.ifht(A, dln, mu, offset=0.0, bias=0.0)`.
///
/// Compute the inverse discrete Hankel transform.
///
/// # Arguments
/// * `input` - Input array in Hankel space
/// * `dln` - Uniform logarithmic spacing ∆ln k
/// * `mu` - Order of the Hankel transform
/// * `offset` - Offset parameter (use `fhtoffset` to compute optimal value)
/// * `bias` - Power-law bias exponent
/// * `options` - FFT options
///
/// # Returns
/// The inverse transformed array
pub fn ifht(
    input: &[f64],
    dln: f64,
    mu: f64,
    offset: f64,
    bias: f64,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;

    let n = input.len();

    // Bias the input array.
    let mut big_a = input.to_vec();
    if bias != 0.0 {
        let j_c = (n as f64 - 1.0) / 2.0;
        for (j, v) in big_a.iter_mut().enumerate() {
            *v *= (bias * ((j as f64 - j_c) * dln + offset)).exp();
        }
    }

    // The inverse transform divides by conj(u) instead of multiplying.
    let u = fhtcoeff(n, dln, mu, offset, bias, true);
    let mut out = fhtq(&big_a, &u, true, options)?;

    // Un-bias the output: a(r) = a_q(r) · (r/r_c)^{q}.
    if bias != 0.0 {
        let j_c = (n as f64 - 1.0) / 2.0;
        for (j, v) in out.iter_mut().enumerate() {
            *v *= (bias * (j as f64 - j_c) * dln).exp();
        }
    }

    Ok(out)
}

/// Coefficient array for the fast Hankel transform (FFTLog, Hamilton 2000).
///
/// Returns `u` of length `n/2 + 1` (the `rfft` spectrum length) with
///
///   u_m = exp[ lnΓ(x₊ + i·yₘ) − lnΓ(x₋ + i·yₘ) + q·ln2
///              + i·2·(ln2 − offset)·yₘ ],
///
/// where `x± = (μ + 1 ± q)/2`, `q = bias`, and `yₘ = π·m/(n·dln)`. The
/// Nyquist coefficient is forced real for even `n`. Faithful port of
/// scipy's `fhtcoeff`.
fn fhtcoeff(n: usize, dln: f64, mu: f64, offset: f64, bias: f64, inverse: bool) -> Vec<Complex64> {
    use std::f64::consts::{LN_2, PI};
    let (lnkr, q) = (offset, bias);
    let xp = (mu + 1.0 + q) / 2.0;
    let xm = (mu + 1.0 - q) / 2.0;
    let len = n / 2 + 1;
    let mut u = Vec::with_capacity(len);
    for m in 0..len {
        let y = PI * m as f64 / (n as f64 * dln);
        let lp = complex_ln_gamma((xp, y));
        let lm = complex_ln_gamma((xm, y));
        // exp() is 2πi-periodic in its imaginary part, so any principal-
        // branch ambiguity in the log-gamma phases cancels here.
        let re = lp.0 - lm.0 + LN_2 * q;
        let im = lp.1 + lm.1 + 2.0 * (LN_2 - lnkr) * y;
        u.push(complex_exp((re, im)));
    }
    // For even-length transforms the Nyquist coefficient must be real.
    if n.is_multiple_of(2)
        && let Some(last) = u.last_mut()
    {
        last.1 = 0.0;
    }
    // u_0 = 2^q · Γ(x₊)/Γ(x₋); if the log form overflowed at a Γ pole,
    // recover whether the true value is 0 (x₋ pole) or ∞ (x₊ pole).
    if !u[0].0.is_finite() || !u[0].1.is_finite() {
        let is_pole = |x: f64| x <= 0.0 && (x - x.round()).abs() < 1e-9;
        u[0] = if is_pole(xm) {
            (0.0, 0.0)
        } else {
            (f64::INFINITY, 0.0)
        };
    }
    // A singular forward (resp. inverse) transform has u_0 = ∞ (resp. 0);
    // scipy substitutes the opposite extreme so the rest still transforms.
    if !inverse && u[0].0.is_infinite() {
        u[0] = (0.0, 0.0);
    } else if inverse && u[0] == (0.0, 0.0) {
        u[0] = (f64::INFINITY, 0.0);
    }
    u
}

/// Biased fast Hankel transform core: real FFT, kernel multiply, inverse
/// real FFT, then reverse. Matches scipy's `_fhtq`.
fn fhtq(
    a: &[f64],
    u: &[Complex64],
    inverse: bool,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let n = a.len();
    let mut spectrum = rfft(a, options)?;
    for (s, &uc) in spectrum.iter_mut().zip(u.iter()) {
        *s = if inverse {
            complex_div(*s, complex_conj(uc))
        } else {
            complex_mul(*s, uc)
        };
    }
    let mut out = irfft(&spectrum, Some(n), options)?;
    out.reverse();
    Ok(out)
}

/// Complex division `lhs / rhs`.
fn complex_div(lhs: Complex64, rhs: Complex64) -> Complex64 {
    let denom = rhs.0 * rhs.0 + rhs.1 * rhs.1;
    (
        (lhs.0 * rhs.0 + lhs.1 * rhs.1) / denom,
        (lhs.1 * rhs.0 - lhs.0 * rhs.1) / denom,
    )
}

/// Complex exponential `exp(z)`.
fn complex_exp(z: Complex64) -> Complex64 {
    let r = z.0.exp();
    (r * z.1.cos(), r * z.1.sin())
}

/// Principal-branch complex logarithm `ln(z)`.
fn complex_ln(z: Complex64) -> Complex64 {
    (0.5 * (z.0 * z.0 + z.1 * z.1).ln(), z.1.atan2(z.0))
}

/// Complex sine `sin(z)`.
fn complex_sin(z: Complex64) -> Complex64 {
    (z.0.sin() * z.1.cosh(), z.0.cos() * z.1.sinh())
}

/// Complex log-gamma `lnΓ(z)` via the Lanczos `g = 7` approximation, with
/// the reflection formula for `Re(z) < 1/2`. The real part is the exact
/// `ln|Γ|`; the imaginary part is correct modulo `2π`, which is all the
/// FFTLog kernel and offset depend on.
fn complex_ln_gamma(z: Complex64) -> Complex64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
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
    if z.0 < 0.5 {
        // Reflection: lnΓ(z) = ln(π) − ln(sin(πz)) − lnΓ(1 − z).
        let pi = std::f64::consts::PI;
        let sin_piz = complex_sin((pi * z.0, pi * z.1));
        let ln_sin = complex_ln(sin_piz);
        let refl = complex_ln_gamma((1.0 - z.0, -z.1));
        (pi.ln() - ln_sin.0 - refl.0, -ln_sin.1 - refl.1)
    } else {
        let zs = (z.0 - 1.0, z.1);
        let mut sum: Complex64 = (C[0], 0.0);
        for (i, &c) in C.iter().enumerate().skip(1) {
            sum = complex_add(sum, complex_div((c, 0.0), (zs.0 + i as f64, zs.1)));
        }
        let t = (zs.0 + G + 0.5, zs.1);
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        // lnΓ(z) = ½·ln(2π) + (z − ½)·ln(t) − t + ln(sum).
        let term2 = complex_mul((zs.0 + 0.5, zs.1), complex_ln(t));
        complex_add(
            complex_add((half_ln_2pi, 0.0), term2),
            complex_add((-t.0, -t.1), complex_ln(sum)),
        )
    }
}

fn run_complex_1d(
    kind: TransformKind,
    input: &[Complex64],
    options: &FftOptions,
    inverse: bool,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let key = PlanKey::new(
        kind,
        vec![input.len()],
        vec![0],
        options.normalization,
        false,
    );
    let plan_cache_hit = touch_plan_cache(&key, input.len());

    let started = Instant::now();
    let mut output = backend.transform_1d_unscaled(input, inverse);
    apply_normalization(&mut output, options.normalization, input.len(), inverse);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: if inverse { "inverse" } else { "forward" },
        n: input.len(),
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn run_complex_nd(
    kind: TransformKind,
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    inverse: bool,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = complex_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {expected_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, false);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut output = transform_nd_unscaled(backend, input, shape, inverse);
    apply_normalization(&mut output, options.normalization, expected_len, inverse);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: if inverse { "inverse" } else { "forward" },
        n: expected_len,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn run_real_nd_forward(
    kind: TransformKind,
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    if input.len() != expected_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {expected_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: expected_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let last_len = *shape.last().ok_or(FftError::InvalidShape {
        detail: "empty shape",
    })?;
    let reduced_last = last_len / 2 + 1;
    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, true);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut output = Vec::with_capacity(expected_len / last_len * reduced_last);
    for lane in input.chunks_exact(last_len) {
        output.extend(real_fft_unscaled(lane, backend));
    }

    let mut complex_shape = shape.to_vec();
    if let Some(last) = complex_shape.last_mut() {
        *last = reduced_last;
    } else {
        return Err(FftError::InvalidShape {
            detail: "empty shape",
        });
    }
    if complex_shape.len() > 1 {
        let max_axis_len = complex_shape.iter().max().copied().unwrap_or(0);
        let mut scratch = vec![(0.0, 0.0); max_axis_len];
        for axis in 0..complex_shape.len() - 1 {
            apply_axis_transform(
                backend,
                &mut output,
                &complex_shape,
                axis,
                false,
                &mut scratch,
            );
        }
    }
    apply_normalization(&mut output, options.normalization, expected_len, false);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: "forward",
        n: expected_len,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn run_real_nd_inverse(
    kind: TransformKind,
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_shape_fingerprint(input, shape);
    validate_shape_with_audit(shape, &fingerprint, audit_ledger)?;
    let expected_len = checked_product(shape).ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "shape_product_overflow",
            "rejected: nd shape product overflow",
        );
        FftError::InvalidShape {
            detail: "nd shape product overflow",
        }
    })?;
    let last_len = *shape.last().ok_or_else(|| {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "empty_shape",
            "rejected: nd shape cannot be empty",
        );
        FftError::InvalidShape {
            detail: "empty shape",
        }
    })?;
    let reduced_last = last_len / 2 + 1;
    let complex_len = shape[..shape.len() - 1]
        .iter()
        .try_fold(reduced_last, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            record_fail_closed(
                audit_ledger,
                &fingerprint,
                "shape_product_overflow",
                "rejected: nd shape product overflow",
            );
            FftError::InvalidShape {
                detail: "nd shape product overflow",
            }
        })?;
    if input.len() != complex_len {
        record_fail_closed(
            audit_ledger,
            &fingerprint,
            "length_mismatch",
            format!(
                "rejected: expected input length {complex_len}, got {}",
                input.len()
            ),
        );
        return Err(FftError::LengthMismatch {
            expected: complex_len,
            actual: input.len(),
        });
    }

    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;

    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, true);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    let mut complex_data = input.to_vec();
    let mut complex_shape = shape.to_vec();
    if let Some(last) = complex_shape.last_mut() {
        *last = reduced_last;
    } else {
        return Err(FftError::InvalidShape {
            detail: "empty shape",
        });
    }
    if complex_shape.len() > 1 {
        let max_axis_len = complex_shape.iter().max().copied().unwrap_or(0);
        let mut scratch = vec![(0.0, 0.0); max_axis_len];
        for axis in 0..complex_shape.len() - 1 {
            apply_axis_transform(
                backend,
                &mut complex_data,
                &complex_shape,
                axis,
                true,
                &mut scratch,
            );
        }
    }

    let mut output = Vec::with_capacity(expected_len);
    for lane in complex_data.chunks_exact(reduced_last) {
        output.extend(real_ifft_unscaled(lane, last_len, backend));
    }
    apply_real_normalization(&mut output, options.normalization, expected_len, true);

    record_trace(TransformTrace {
        operation_id: next_operation_id(),
        kind,
        direction: "inverse",
        n: expected_len,
        backend: backend.kind(),
        plan_cache_hit,
        mode: options.mode,
        timing_ns: started.elapsed().as_nanos(),
    });

    Ok(output)
}

fn transform_nd_unscaled(
    backend: &dyn FftBackend,
    input: &[Complex64],
    shape: &[usize],
    inverse: bool,
) -> Vec<Complex64> {
    let mut data = input.to_vec();
    let max_axis_len = shape.iter().max().copied().unwrap_or(0);
    let mut scratch = vec![(0.0, 0.0); max_axis_len];
    for axis in 0..shape.len() {
        apply_axis_transform(backend, &mut data, shape, axis, inverse, &mut scratch);
    }
    data
}

/// Threads to use for one parallel axis pass. `work` is the element count touched.
/// Each parallel phase spawns real OS threads, so splitting only pays off on large
/// passes; below ~1M elements the strided-copy / transform work is cheaper done
/// sequentially (spinning up 64 threads for microseconds of work regressed small
/// nd FFTs). We also keep >=64K elements per thread so thread count tracks work.
fn nd_axis_thread_count(lanes: usize, axis_len: usize) -> usize {
    const MIN_TOTAL_WORK: usize = 1 << 20; // 1,048,576 elements
    const MIN_WORK_PER_THREAD: usize = 1 << 16; // 65,536 elements
    let work = lanes.saturating_mul(axis_len);
    if lanes < 64 || work < MIN_TOTAL_WORK {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let by_work = (work / MIN_WORK_PER_THREAD).max(1);
    cores.min(by_work).min(lanes).max(1)
}

/// Axis-0 transform via transpose. Axis 0 has `repeats == 1`, so its `stride`
/// independent lanes are strided through the whole array and cannot be split into
/// disjoint contiguous slices in place. We transpose those lanes into a contiguous
/// scratch buffer (one lane per `axis_len` run), transform each lane, and transpose
/// back. Every phase is parallel over disjoint output regions:
///   * gather+transform: thread owns offsets [o0, o0+k) -> contiguous transposed lanes
///   * scatter: thread owns indices [i0, i0+m) -> contiguous `data` rows of `stride`
///
/// Byte-identical to the sequential strided walk: each lane's 1D FFT sees the same
/// inputs in the same order, and gather/scatter are pure permutations.
fn apply_axis0_transpose_transform(
    data: &mut [Complex64],
    axis_len: usize,
    stride: usize,
    nthreads: usize,
    twiddles: &TwiddleTable,
    bit_reverse_swaps: &[(usize, usize)],
) {
    let mut transposed = vec![(0.0, 0.0); axis_len * stride];

    // Parallel gather + transform fused: lane `offset` is transposed[offset*axis_len..]
    let offsets_per_thread = stride.div_ceil(nthreads);
    let gather_chunk = offsets_per_thread * axis_len;
    {
        let data_ref: &[Complex64] = data;
        std::thread::scope(|scope| {
            for (chunk_idx, chunk) in transposed.chunks_mut(gather_chunk).enumerate() {
                let o_start = chunk_idx * offsets_per_thread;
                scope.spawn(move || {
                    for (off_local, lane) in chunk.chunks_mut(axis_len).enumerate() {
                        let offset = o_start + off_local;
                        for (index, slot) in lane.iter_mut().enumerate() {
                            *slot = data_ref[index * stride + offset];
                        }
                        cooley_tukey_radix2_inplace_with_plan(lane, twiddles, bit_reverse_swaps);
                    }
                });
            }
        });
    }

    // Parallel scatter: row `index` is data[index*stride..(index+1)*stride]
    let indices_per_thread = axis_len.div_ceil(nthreads);
    let scatter_chunk = indices_per_thread * stride;
    {
        let transposed_ref: &[Complex64] = &transposed;
        std::thread::scope(|scope| {
            for (chunk_idx, chunk) in data.chunks_mut(scatter_chunk).enumerate() {
                let i_start = chunk_idx * indices_per_thread;
                scope.spawn(move || {
                    for (idx_local, row) in chunk.chunks_mut(stride).enumerate() {
                        let index = i_start + idx_local;
                        for (offset, slot) in row.iter_mut().enumerate() {
                            *slot = transposed_ref[offset * axis_len + index];
                        }
                    }
                });
            }
        });
    }
}

fn apply_axis_transform(
    backend: &dyn FftBackend,
    data: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    inverse: bool,
    scratch: &mut [Complex64],
) {
    let axis_len = shape[axis];
    let stride = shape[axis + 1..].iter().product::<usize>().max(1);
    let repeats = shape[..axis].iter().product::<usize>().max(1);
    let block = axis_len * stride;
    let cooley_tukey_axis_plan =
        if backend.kind() == BackendKind::CooleyTukey && axis_len.is_power_of_two() {
            Some((
                get_or_compute_twiddles(axis_len, inverse),
                bit_reverse_swaps(axis_len),
            ))
        } else {
            None
        };

    // Axis 0 (repeats == 1) can only be parallelized via the transpose path, which
    // costs two extra array passes; only take it when it will actually run threaded,
    // otherwise the direct strided loop below is cheaper.
    if axis == 0
        && repeats == 1
        && let Some((twiddles, bit_reverse_swaps)) = &cooley_tukey_axis_plan
    {
        let nthreads = nd_axis_thread_count(stride, axis_len);
        if nthreads > 1 {
            apply_axis0_transpose_transform(
                data,
                axis_len,
                stride,
                nthreads,
                twiddles,
                bit_reverse_swaps,
            );
            return;
        }
    }

    // Axis > 0: each outer block is a disjoint contiguous `block`-element slice whose
    // lanes are independent, so split whole blocks across threads (each with its own
    // scratch) and transform in place. Byte-identical to the sequential (outer, offset)
    // walk because each lane's read/transform/write set is unchanged.
    let nthreads = nd_axis_thread_count(repeats.saturating_mul(stride), axis_len);
    if repeats >= 2 && nthreads > 1 {
        let backend_kind = backend.kind();
        let plan = &cooley_tukey_axis_plan;
        let blocks_per_thread = repeats.div_ceil(nthreads);
        let chunk_len = blocks_per_thread * block;
        std::thread::scope(|scope| {
            for chunk in data.chunks_mut(chunk_len) {
                scope.spawn(move || {
                    let backend = resolve_backend(backend_kind);
                    let mut local = vec![(0.0, 0.0); axis_len];
                    for block_slice in chunk.chunks_mut(block) {
                        for offset in 0..stride {
                            for (index, slot) in local.iter_mut().enumerate() {
                                *slot = block_slice[index * stride + offset];
                            }
                            if let Some((twiddles, bit_reverse_swaps)) = plan {
                                cooley_tukey_radix2_inplace_with_plan(
                                    &mut local,
                                    twiddles,
                                    bit_reverse_swaps,
                                );
                            } else {
                                backend.transform_1d_inplace(&mut local, inverse);
                            }
                            for (index, &value) in local.iter().enumerate() {
                                block_slice[index * stride + offset] = value;
                            }
                        }
                    }
                });
            }
        });
        return;
    }

    let axis_scratch = &mut scratch[..axis_len];
    for outer in 0..repeats {
        let outer_base = outer * block;
        for offset in 0..stride {
            for (index, slot) in axis_scratch.iter_mut().enumerate() {
                *slot = data[outer_base + index * stride + offset];
            }
            if let Some((twiddles, bit_reverse_swaps)) = &cooley_tukey_axis_plan {
                cooley_tukey_radix2_inplace_with_plan(axis_scratch, twiddles, bit_reverse_swaps);
            } else {
                backend.transform_1d_inplace(axis_scratch, inverse);
            }
            for (index, &value) in axis_scratch.iter().enumerate() {
                data[outer_base + index * stride + offset] = value;
            }
        }
    }
}

fn rebuild_hermitian(half: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut full = vec![(0.0, 0.0); n];
    full[0] = half[0];

    for (k, &value) in half.iter().enumerate().skip(1) {
        full[k] = value;
    }
    for k in half.len()..n {
        let mirror = n - k;
        full[k] = complex_conj(full[mirror]);
    }

    if n.is_multiple_of(2) && n / 2 < full.len() {
        full[n / 2].1 = 0.0;
    }
    full
}

fn real_fft_unscaled(input: &[f64], backend: &dyn FftBackend) -> Vec<Complex64> {
    if input.len().is_power_of_two() && input.len() >= 4 {
        real_fft_specialized(input, backend)
    } else {
        let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
        let full = backend.transform_1d_unscaled(&complex_input, false);
        full.into_iter().take(input.len() / 2 + 1).collect()
    }
}

fn real_ifft_unscaled(input: &[Complex64], n: usize, backend: &dyn FftBackend) -> Vec<f64> {
    let reconstructed = rebuild_hermitian(input, n);
    backend
        .transform_1d_unscaled(&reconstructed, true)
        .into_iter()
        .map(|(re, _)| re)
        .collect()
}

fn resolve_backend(kind: BackendKind) -> &'static dyn FftBackend {
    match kind {
        BackendKind::NaiveDft => &NAIVE_BACKEND,
        BackendKind::CooleyTukey => &COOLEY_TUKEY_BACKEND,
    }
}

fn touch_plan_cache(key: &PlanKey, n: usize) -> bool {
    if lookup_shared_plan(key).is_some() {
        return true;
    }

    let _ = store_shared_plan(PlanMetadata {
        key: key.clone(),
        fingerprint: PlanFingerprint {
            radix_path: factorize_radix_path(n),
            estimated_flops: estimate_fft_flops(n),
            scratch_bytes: n.saturating_mul(std::mem::size_of::<Complex64>()),
        },
        generated_by: PlanningStrategy::EstimateOnly,
    });
    false
}

fn estimate_fft_flops(n: usize) -> u64 {
    if n <= 1 {
        return 0;
    }
    let stages = usize::BITS - (n - 1).leading_zeros();
    (n as u64).saturating_mul(stages as u64).saturating_mul(5)
}

fn factorize_radix_path(mut n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![1];
    }

    let mut factors = Vec::new();
    let mut p = 2usize;
    while p * p <= n {
        while n.is_multiple_of(p) {
            factors.push(p);
            n /= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

fn validate_shape(shape: &[usize]) -> Result<(), FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "nd shape cannot be empty",
        });
    }
    if shape.contains(&0) {
        return Err(FftError::InvalidShape {
            detail: "nd shape dimensions must be greater than zero",
        });
    }
    Ok(())
}

fn checked_product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &next| acc.checked_mul(next))
}

fn validate_workers(policy: WorkerPolicy) -> Result<(), FftError> {
    match policy {
        WorkerPolicy::Auto => Ok(()),
        WorkerPolicy::Exact(0) | WorkerPolicy::Max(0) => {
            Err(FftError::InvalidWorkers { requested: 0 })
        }
        WorkerPolicy::Exact(_) | WorkerPolicy::Max(_) => Ok(()),
    }
}

fn validate_finite_complex(input: &[Complex64], options: &FftOptions) -> Result<(), FftError> {
    let should_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if should_check
        && input
            .iter()
            .any(|&(re, im)| !re.is_finite() || !im.is_finite())
    {
        return Err(FftError::NonFiniteInput);
    }
    Ok(())
}

fn validate_finite_real(input: &[f64], options: &FftOptions) -> Result<(), FftError> {
    let should_check = options.check_finite || options.mode == RuntimeMode::Hardened;
    if should_check && input.iter().any(|value| !value.is_finite()) {
        return Err(FftError::NonFiniteInput);
    }
    Ok(())
}

fn ensure_non_empty(len: usize) -> Result<(), FftError> {
    if len == 0 {
        return Err(FftError::InvalidShape {
            detail: "input length must be greater than zero",
        });
    }
    Ok(())
}

fn record_audit_event(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    action: AuditAction,
    outcome: impl Into<String>,
) {
    let Some(audit_ledger) = audit_ledger else {
        return;
    };
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        action,
        outcome,
    );
    // Resolves [frankenscipy-be4cw] (deferred from kt4od): the previous
    // `if let Ok(mut guard) = lock()` pattern silently dropped events
    // on poisoned mutexes, breaking the fail-closed audit contract.
    let mut guard = match audit_ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            audit_ledger.clear_poison();
            poisoned.into_inner()
        }
    };
    guard.record(event);
}

fn record_mode_decision(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    mode: RuntimeMode,
    outcome: impl Into<String>,
) {
    record_audit_event(
        audit_ledger,
        input_bytes,
        AuditAction::ModeDecision { mode },
        outcome,
    );
}

fn record_fail_closed(
    audit_ledger: Option<&SyncSharedAuditLedger>,
    input_bytes: &[u8],
    reason: &str,
    outcome: impl Into<String>,
) {
    record_audit_event(
        audit_ledger,
        input_bytes,
        AuditAction::FailClosed {
            reason: reason.to_owned(),
        },
        outcome,
    );
}

fn ensure_non_empty_with_audit(
    len: usize,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if len == 0 {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "empty_input",
            "rejected: input length must be greater than zero",
        );
    }
    ensure_non_empty(len)
}

fn validate_workers_with_audit(
    policy: WorkerPolicy,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if matches!(policy, WorkerPolicy::Exact(0) | WorkerPolicy::Max(0)) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "invalid_workers",
            "rejected: worker count must be greater than zero",
        );
    }
    validate_workers(policy)
}

fn validate_finite_complex_with_audit(
    input: &[Complex64],
    options: &FftOptions,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if options.mode == RuntimeMode::Hardened && !options.check_finite {
        record_mode_decision(
            audit_ledger,
            fingerprint,
            RuntimeMode::Hardened,
            "promoted finite-check policy for complex FFT input",
        );
    }
    // Short-circuit on the cheap policy flag BEFORE the O(n) finiteness scan:
    // the scan's result only ever feeds `record_fail_closed`, which is gated on
    // `check_finite || Hardened`. On the default path (check_finite=false,
    // Strict) the scan can record nothing, so running it is a wasted full-array
    // read pass (~2% of a large fft()). Behavior is unchanged — the record still
    // fires exactly when policy demands a check and the input is non-finite.
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && input
            .iter()
            .any(|&(re, im)| !re.is_finite() || !im.is_finite())
    {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "non_finite_input",
            "rejected: complex input contains NaN or Inf",
        );
    }
    validate_finite_complex(input, options)
}

fn validate_finite_real_with_audit(
    input: &[f64],
    options: &FftOptions,
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if options.mode == RuntimeMode::Hardened && !options.check_finite {
        record_mode_decision(
            audit_ledger,
            fingerprint,
            RuntimeMode::Hardened,
            "promoted finite-check policy for real FFT input",
        );
    }
    // See validate_finite_complex_with_audit: gate the O(n) scan on the policy
    // flag so the default (check_finite=false) path skips a wasted full pass.
    if (options.check_finite || options.mode == RuntimeMode::Hardened)
        && input.iter().any(|value| !value.is_finite())
    {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "non_finite_input",
            "rejected: real input contains NaN or Inf",
        );
    }
    validate_finite_real(input, options)
}

fn validate_shape_with_audit(
    shape: &[usize],
    fingerprint: &[u8],
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<(), FftError> {
    if shape.is_empty() {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "empty_shape",
            "rejected: nd shape cannot be empty",
        );
    } else if shape.contains(&0) {
        record_fail_closed(
            audit_ledger,
            fingerprint,
            "zero_dimension",
            "rejected: nd shape dimensions must be greater than zero",
        );
    }
    validate_shape(shape)
}

fn real_fingerprint(input: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len().min(64) * std::mem::size_of::<f64>());
    for &value in input.iter().take(64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn complex_fingerprint(input: &[Complex64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len().min(32) * 2 * std::mem::size_of::<f64>());
    for &(re, im) in input.iter().take(32) {
        bytes.extend_from_slice(&re.to_le_bytes());
        bytes.extend_from_slice(&im.to_le_bytes());
    }
    bytes
}

fn append_shape_fingerprint(bytes: &mut Vec<u8>, shape: &[usize]) {
    for &dim in shape.iter().take(16) {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
}

fn real_shape_fingerprint(input: &[f64], shape: &[usize]) -> Vec<u8> {
    let mut bytes = real_fingerprint(input);
    append_shape_fingerprint(&mut bytes, shape);
    bytes
}

fn complex_shape_fingerprint(input: &[Complex64], shape: &[usize]) -> Vec<u8> {
    let mut bytes = complex_fingerprint(input);
    append_shape_fingerprint(&mut bytes, shape);
    bytes
}

fn normalization_scale(normalization: Normalization, n: usize, inverse: bool) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let n_as_f64 = n as f64;
    match normalization {
        Normalization::Backward => {
            if inverse {
                1.0 / n_as_f64
            } else {
                1.0
            }
        }
        Normalization::Forward => {
            if inverse {
                1.0
            } else {
                1.0 / n_as_f64
            }
        }
        Normalization::Ortho => 1.0 / n_as_f64.sqrt(),
    }
}

fn apply_normalization(
    data: &mut [Complex64],
    normalization: Normalization,
    n: usize,
    inverse: bool,
) {
    let scale = normalization_scale(normalization, n, inverse);
    if (scale - 1.0).abs() <= f64::EPSILON {
        return;
    }
    for value in data.iter_mut() {
        *value = complex_scale(*value, scale);
    }
}

fn apply_real_normalization(
    data: &mut [f64],
    normalization: Normalization,
    n: usize,
    inverse: bool,
) {
    let scale = normalization_scale(normalization, n, inverse);
    if (scale - 1.0).abs() <= f64::EPSILON {
        return;
    }
    for value in data.iter_mut() {
        *value *= scale;
    }
}

fn complex_add(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

fn complex_mul(lhs: Complex64, rhs: Complex64) -> Complex64 {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn complex_scale(value: Complex64, scale: f64) -> Complex64 {
    (value.0 * scale, value.1 * scale)
}

fn complex_conj(value: Complex64) -> Complex64 {
    (value.0, -value.1)
}

fn transform_kind_name(kind: TransformKind) -> &'static str {
    match kind {
        TransformKind::Fft => "fft",
        TransformKind::Ifft => "ifft",
        TransformKind::Rfft => "rfft",
        TransformKind::Irfft => "irfft",
        TransformKind::Fft2 => "fft2",
        TransformKind::Ifft2 => "ifft2",
        TransformKind::Fftn => "fftn",
        TransformKind::Ifftn => "ifftn",
        TransformKind::Rfftn => "rfftn",
        TransformKind::Irfftn => "irfftn",
    }
}

fn backend_kind_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::NaiveDft => "naive_dft",
        BackendKind::CooleyTukey => "cooley_tukey",
    }
}

fn runtime_mode_name(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::Strict => "Strict",
        RuntimeMode::Hardened => "Hardened",
    }
}

// ══════════════════════════════════════════════════════════════════════
// Multi-D Real FFT, next_fast_len, hfft
// ══════════════════════════════════════════════════════════════════════

/// 2D real-input FFT.
///
/// Matches `scipy.fft.rfft2(x, s)`.
///
/// Computes rfft along the last axis, then fft along the first axis.
pub fn rfft2(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    rfft2_impl(input, &dims, options, None)
}

/// 2D real-input FFT with audit logging.
pub fn rfft2_with_audit(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    rfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// 2D inverse real FFT.
///
/// Matches `scipy.fft.irfft2(x, s)`.
pub fn irfft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    irfft2_impl(input, &dims, options, None)
}

/// 2D inverse real FFT with audit logging.
pub fn irfft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    irfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// N-dimensional inverse real FFT.
///
/// Matches `scipy.fft.irfftn(x, s)`.
pub fn irfftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, None)
}

/// N-dimensional inverse real FFT with audit logging.
pub fn irfftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, Some(audit_ledger))
}

/// Find the next fast length for FFT computation.
///
/// Matches `scipy.fft.next_fast_len(target)`.
///
/// Returns the smallest integer >= `target` that is a product of
/// small prime factors (2, 3, 5), since FFT is most efficient for these sizes.
pub fn next_fast_len(target: usize) -> usize {
    if target <= 1 {
        return target;
    }
    let mut n = target;
    loop {
        if is_fast_len(n) {
            return n;
        }
        n += 1;
    }
}

/// Find the previous fast FFT length.
///
/// Matches `scipy.fft.prev_fast_len(target)`.
///
/// Returns the largest integer ≤ `target` that is a product of small
/// prime factors {2, 3, 5}. Used to downsize an FFT to a length cheaper
/// than `target`. The empty-product `1` is always 5-smooth, so the
/// search terminates at 1 in the worst case. Input `0` returns `0`.
#[must_use]
pub fn prev_fast_len(target: usize) -> usize {
    if target == 0 {
        return 0;
    }
    let mut n = target;
    loop {
        if is_fast_len(n) {
            return n;
        }
        n -= 1;
    }
}

/// Check if n is composed only of factors 2, 3, 5 (Hamming numbers / regular numbers).
#[allow(clippy::manual_is_multiple_of)]
fn is_fast_len(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    while n % 2 == 0 {
        n /= 2;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    n == 1
}

/// Hermitian FFT (FFT of a signal with Hermitian symmetry in the frequency domain).
///
/// Matches `scipy.fft.hfft(x, n)`.
///
/// Takes a half-spectrum (like rfft output) and produces a real-valued full signal.
/// This is essentially irfft scaled differently — hfft(x, n) = n * irfft(conj(x), n).
pub fn hfft(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    hfft_impl(input, n, options, None)
}

/// Hermitian FFT with audit logging.
pub fn hfft_with_audit(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    hfft_impl(input, n, options, Some(audit_ledger))
}

/// Inverse of the Hermitian FFT (hfft).
///
/// Matches `scipy.fft.ihfft(x, n)`.
///
/// Takes a real signal and returns the Hermitian-symmetric spectrum.
/// This is the inverse of `hfft`: `ihfft(hfft(x)) ≈ x`.
pub fn ihfft(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    ihfft_impl(input, n, options, None)
}

/// Inverse Hermitian FFT with audit logging.
pub fn ihfft_with_audit(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ihfft_impl(input, n, options, Some(audit_ledger))
}

/// 2D Hermitian FFT.
///
/// Matches `scipy.fft.hfft2(x, s)` for flattened row-major input.
pub fn hfft2(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    hfft2_impl(input, &dims, options, None)
}

/// 2D Hermitian FFT with audit logging.
pub fn hfft2_with_audit(
    input: &[Complex64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    let dims = [shape.0, shape.1];
    hfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// 2D inverse Hermitian FFT.
///
/// Matches `scipy.fft.ihfft2(x, s)` for flattened row-major input.
pub fn ihfft2(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ihfft2_impl(input, &dims, options, None)
}

/// 2D inverse Hermitian FFT with audit logging.
pub fn ihfft2_with_audit(
    input: &[f64],
    shape: (usize, usize),
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    let dims = [shape.0, shape.1];
    ihfft2_impl(input, &dims, options, Some(audit_ledger))
}

/// N-dimensional Hermitian FFT.
///
/// Matches `scipy.fft.hfftn(x, s)` for flattened row-major input.
pub fn hfftn(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    hfftn_impl(input, shape, options, None)
}

/// N-dimensional Hermitian FFT with audit logging.
pub fn hfftn_with_audit(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<f64>, FftError> {
    hfftn_impl(input, shape, options, Some(audit_ledger))
}

/// N-dimensional inverse Hermitian FFT.
///
/// Matches `scipy.fft.ihfftn(x, s)` for flattened row-major input.
pub fn ihfftn(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    ihfftn_impl(input, shape, options, None)
}

/// N-dimensional inverse Hermitian FFT with audit logging.
pub fn ihfftn_with_audit(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: &SyncSharedAuditLedger,
) -> Result<Vec<Complex64>, FftError> {
    ihfftn_impl(input, shape, options, Some(audit_ledger))
}

fn rfft2_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    rfftn_impl(input, shape, options, audit_ledger)
}

fn irfft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    irfftn_impl(input, shape, options, audit_ledger)
}

fn irfftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    run_real_nd_inverse(TransformKind::Irfftn, input, shape, options, audit_ledger)
}

fn hfft_impl(
    input: &[Complex64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let fingerprint = complex_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    let out_len = n.unwrap_or_else(|| {
        if input.len() == 1 {
            1
        } else {
            input.len().saturating_sub(1).saturating_mul(2)
        }
    });
    let conjugated: Vec<Complex64> = input.iter().copied().map(complex_conj).collect();
    let mut result = irfft_impl(&conjugated, Some(out_len), options, audit_ledger)?;
    let scale = out_len as f64;
    for v in &mut result {
        *v *= scale;
    }
    Ok(result)
}

fn ihfft_impl(
    input: &[f64],
    n: Option<usize>,
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let fingerprint = real_fingerprint(input);
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_finite_real_with_audit(input, options, &fingerprint, audit_ledger)?;

    let in_len = n.unwrap_or(input.len());
    let mut padded = vec![0.0; in_len];
    let copy_len = input.len().min(in_len);
    padded[..copy_len].copy_from_slice(&input[..copy_len]);

    let mut result = rfft_impl(&padded, options, audit_ledger)?;
    let scale = 1.0 / in_len as f64;
    for c in &mut result {
        c.0 *= scale;
        c.1 *= -scale;
    }

    Ok(result)
}

fn hfft2_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    hfftn_impl(input, shape, options, audit_ledger)
}

fn ihfft2_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    ihfftn_impl(input, shape, options, audit_ledger)
}

fn hfftn_impl(
    input: &[Complex64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<f64>, FftError> {
    let conjugated: Vec<Complex64> = input.iter().copied().map(complex_conj).collect();
    let mut result = irfftn_impl(&conjugated, shape, options, audit_ledger)?;
    let scale = result.len() as f64;
    for value in &mut result {
        *value *= scale;
    }
    Ok(result)
}

fn ihfftn_impl(
    input: &[f64],
    shape: &[usize],
    options: &FftOptions,
    audit_ledger: Option<&SyncSharedAuditLedger>,
) -> Result<Vec<Complex64>, FftError> {
    let mut result = rfftn_impl(input, shape, options, audit_ledger)?;
    let scale = 1.0 / input.len() as f64;
    for value in &mut result {
        value.0 *= scale;
        value.1 *= -scale;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::{AuditAction, RuntimeMode};

    use super::{
        FftError, FftOptions, TransformKind, WorkerPolicy, dct, dct_iv, dctn, dst_ii, dst_iii,
        estimate_fft_flops, fft, fft_with_audit, fft2, fftn, hfft, hfft2, hfftn, idct, idctn, ifft,
        ifft2, ifftn, ihfft2, ihfftn, irfft, irfft2, irfftn, is_fast_len, next_fast_len,
        prev_fast_len, rfft, rfft_with_audit, rfft2, rfftn, sync_audit_ledger,
        take_transform_traces,
    };
    use crate::Normalization;
    use crate::plan::{clear_shared_plan_cache, shared_cache_test_lock};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!((actual - expected).abs() <= tol, "{actual} !~= {expected}");
    }

    fn assert_close_complex(actual: (f64, f64), expected: (f64, f64), tol: f64) {
        assert_close(actual.0, expected.0, tol);
        assert_close(actual.1, expected.1, tol);
    }

    #[test]
    fn options_default_to_strict_backward_mode() {
        let opts = FftOptions::default();
        assert_eq!(opts.mode, RuntimeMode::Strict);
        assert_eq!(opts.normalization, Normalization::Backward);
    }

    #[test]
    fn exact_zero_workers_is_rejected() {
        let opts = FftOptions::default().with_workers(WorkerPolicy::Exact(0));
        let err = fft(&[(0.0, 0.0)], &opts).expect_err("zero workers should be rejected");
        assert_eq!(err, FftError::InvalidWorkers { requested: 0 });
    }

    #[test]
    fn fft_ifft_roundtrip_identity() {
        let input = vec![(1.0, 0.0), (2.0, -1.0), (0.5, 0.25), (-3.0, 2.0)];
        let opts = FftOptions::default();
        let spectrum = fft(&input, &opts).expect("fft should succeed");
        let recovered = ifft(&spectrum, &opts).expect("ifft should succeed");
        for (&lhs, &rhs) in recovered.iter().zip(&input) {
            assert_close_complex(lhs, rhs, 1e-9);
        }
    }

    #[test]
    fn rfft_length_matches_scipy_convention() {
        let output =
            rfft(&[1.0, 2.0, 3.0, 4.0, 5.0], &FftOptions::default()).expect("rfft should succeed");
        assert_eq!(output.len(), 5 / 2 + 1);
    }

    #[test]
    fn fft2_ifft2_roundtrip_identity() {
        let input = vec![(1.0, 0.0), (2.0, 1.0), (3.0, -1.0), (4.0, 0.5)];
        let spectrum = fft2(&input, (2, 2), &FftOptions::default()).expect("fft2 should succeed");
        let recovered =
            ifft2(&spectrum, (2, 2), &FftOptions::default()).expect("ifft2 should succeed");
        for (&lhs, &rhs) in recovered.iter().zip(&input) {
            assert_close_complex(lhs, rhs, 1e-9);
        }
    }

    #[test]
    fn irfft_respects_requested_length() {
        let input = vec![(15.0, 0.0), (-2.5, 3.4409548), (-2.5, 0.81229924)];
        let signal = irfft(&input, Some(5), &FftOptions::default()).expect("irfft should succeed");
        assert_eq!(signal.len(), 5);
    }

    #[test]
    fn irfft_length_1() {
        let input = vec![(5.0, 0.0)];
        let signal = irfft(&input, Some(1), &FftOptions::default()).expect("irfft len 1");
        assert_eq!(signal.len(), 1);
        assert_close(signal[0], 5.0, 1e-12);

        // Default length for input length 1 should also be 1
        let signal_default =
            irfft(&input, None, &FftOptions::default()).expect("irfft default len 1");
        assert_eq!(signal_default.len(), 1);
        assert_close(signal_default[0], 5.0, 1e-12);
    }

    #[test]
    fn hardened_mode_rejects_non_finite_input() {
        let opts = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = rfft(&[1.0, f64::NAN], &opts).expect_err("hardened mode should reject NaN");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    #[test]
    fn hardened_audit_records_mode_decision_and_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let opts = FftOptions::default().with_mode(RuntimeMode::Hardened);
        let err = rfft_with_audit(&[1.0, f64::NAN], &opts, &audit_ledger)
            .expect_err("hardened mode should reject NaN");
        assert_eq!(err, FftError::NonFiniteInput);

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert!(ledger.entries().iter().any(|entry| matches!(
            entry.action,
            AuditAction::ModeDecision {
                mode: RuntimeMode::Hardened
            }
        )));
        assert!(ledger.entries().iter().any(|entry| matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "non_finite_input"
        )));
    }

    #[test]
    fn fft_with_audit_records_empty_input_fail_closed() {
        let audit_ledger = sync_audit_ledger();
        let err = fft_with_audit(&[], &FftOptions::default(), &audit_ledger)
            .expect_err("empty input should be rejected");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "input length must be greater than zero",
            }
        );

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert!(ledger.entries().iter().any(|entry| matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "empty_input"
        )));
    }

    #[test]
    fn fft_with_audit_records_event_after_poison_recovery() {
        // /modes-of-reasoning-project-analysis regression for
        // frankenscipy-be4cw (deferred from kt4od): the silent-drop
        // pattern at transforms.rs:2359 has been replaced with explicit
        // poison recovery. After deliberately poisoning the ledger,
        // fft_with_audit's failure path must still record an event.
        let audit_ledger = sync_audit_ledger();
        // Poison the ledger by panicking while holding the guard.
        let poisoned_thread = {
            let l = audit_ledger.clone();
            std::thread::spawn(move || {
                let _g = l.lock().expect("acquire");
                std::panic::resume_unwind(Box::new("poison the FFT audit ledger on purpose"));
            })
            .join()
        };
        assert!(poisoned_thread.is_err(), "thread should have panicked");
        assert!(
            audit_ledger.lock().is_err(),
            "ledger must be poisoned after panic"
        );

        // Trigger an audit-emitting failure path post-poison.
        let err = fft_with_audit(&[], &FftOptions::default(), &audit_ledger)
            .expect_err("empty input should still be rejected");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "input length must be greater than zero",
            }
        );

        // The fail_closed event must be present despite the prior poison.
        let ledger = audit_ledger.lock().expect("ledger should recover");
        assert!(
            ledger.entries().iter().any(|entry| matches!(
                &entry.action,
                AuditAction::FailClosed { reason } if reason == "empty_input"
            )),
            "fft_with_audit must record post-poison; entries: {:?}",
            ledger.entries()
        );
    }

    #[test]
    fn fftn_validates_shape_product_against_input() {
        let err = fftn(&[(0.0, 0.0); 5], &[2, 3], &FftOptions::default())
            .expect_err("shape/input mismatch should fail");
        assert_eq!(
            err,
            FftError::LengthMismatch {
                expected: 6,
                actual: 5,
            }
        );
    }

    #[test]
    fn rfft_metamorphic_matches_first_half_of_fft() {
        // /testing-metamorphic: rfft(x) on real x must equal the first
        // ⌊N/2⌋+1 bins of fft(x as complex). Pin across multiple N.
        for n in [4_usize, 8, 16, 32, 13] {
            let real_input: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let complex_input: Vec<(f64, f64)> = real_input.iter().map(|&v| (v, 0.0)).collect();
            let opts = FftOptions::default();
            let r = rfft(&real_input, &opts).expect("rfft");
            let c = fft(&complex_input, &opts).expect("fft");
            assert_eq!(r.len(), n / 2 + 1, "rfft length mismatch at N={n}");
            for k in 0..r.len() {
                assert!(
                    (r[k].0 - c[k].0).abs() < 1e-10 && (r[k].1 - c[k].1).abs() < 1e-10,
                    "N={n}, k={k}: rfft = ({}, {}), fft = ({}, {})",
                    r[k].0,
                    r[k].1,
                    c[k].0,
                    c[k].1
                );
            }
        }
    }

    #[test]
    fn idct_dct_metamorphic_roundtrip_under_backward() {
        // /testing-metamorphic: under default (backward) normalization,
        // dct returns 2·sum and idct's 1/(2N) inverse cancels exactly,
        // so idct(dct(x)) = x.
        for n in [4_usize, 8, 16, 11, 13] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64).sin() * 0.5 + 0.25).collect();
            let opts = FftOptions::default();
            let spectrum = dct(&x, &opts).expect("dct");
            let recovered = idct(&spectrum, &opts).expect("idct");
            assert_eq!(recovered.len(), n);
            for (i, (&got, &want)) in recovered.iter().zip(x.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "N={n}, idct(dct(x))[{i}] = {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn dct_iv_self_inverse_under_ortho() {
        // /testing-conformance-harnesses + /testing-metamorphic:
        // DCT-IV with ortho normalization is its own inverse:
        //   dct_iv(dct_iv(x, ortho), ortho) = x.
        // Pin across N ∈ {4, 8, 11, 13, 16} within 1e-9.
        for n in [4_usize, 8, 16, 11, 13] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64).cos() * 0.5 + 0.7).collect();
            let opts = FftOptions::default().with_normalization(super::Normalization::Ortho);
            let once = dct_iv(&x, &opts).expect("dct_iv first");
            let twice = dct_iv(&once, &opts).expect("dct_iv second");
            for (i, (&got, &want)) in twice.iter().zip(x.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "N={n}: dct_iv(dct_iv(x, ortho), ortho)[{i}] = {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn dst_iii_dst_ii_metamorphic_roundtrip_per_normalization() {
        // /testing-metamorphic: dst_iii is the inverse of dst_ii up to
        // a normalization-dependent scale factor:
        //   Backward (default): dst_iii(dst_ii(x)) = 2N · x
        //   Ortho:               dst_iii(dst_ii(x)) =  x  (isometric)
        //   Forward:             dst_iii(dst_ii(x)) =  x · (1/(2N))? — scipy
        //                        forward divides dst by 2N so the inverse
        //                        without rescaling gives x · (raw 2N) =
        //                        … actually forward roundtrip = x.
        // The backward 2N factor matches scipy.fft.idst(dst(x)) = 2N·x.
        for n in [4_usize, 8, 16, 11, 13] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin() * 0.7).collect();
            // Backward: roundtrip multiplies by 2N.
            let opts_back = FftOptions::default();
            let spectrum = dst_ii(&x, &opts_back).expect("dst_ii");
            let recovered = dst_iii(&spectrum, &opts_back).expect("dst_iii");
            for (i, (&got, &want)) in recovered.iter().zip(x.iter()).enumerate() {
                let expected = want * 2.0 * n as f64;
                assert!(
                    (got - expected).abs() < 1e-9,
                    "Backward N={n}: dst_iii(dst_ii(x))[{i}] = {got}, expected 2N·x = {expected}"
                );
            }
            // Ortho: isometric — roundtrip recovers x exactly.
            let opts_ortho = FftOptions::default().with_normalization(super::Normalization::Ortho);
            let spectrum_ortho = dst_ii(&x, &opts_ortho).expect("dst_ii");
            let recovered_ortho = dst_iii(&spectrum_ortho, &opts_ortho).expect("dst_iii");
            for (i, (&got, &want)) in recovered_ortho.iter().zip(x.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "Ortho N={n}: dst_iii(dst_ii(x))[{i}] = {got}, expected {want}"
                );
            }
            // Forward: dst applies 1/(2N) and dst_iii applies the
            // raw 2-sum (no inverse rescale), so the roundtrip
            // collapses to x/(2N) — symmetric with the Backward case
            // (×2N): Backward·Forward = 1.
            let opts_fwd = FftOptions::default().with_normalization(super::Normalization::Forward);
            let spectrum_fwd = dst_ii(&x, &opts_fwd).expect("dst_ii");
            let recovered_fwd = dst_iii(&spectrum_fwd, &opts_fwd).expect("dst_iii");
            let two_n = 2.0 * n as f64;
            for (i, (&got, &want)) in recovered_fwd.iter().zip(x.iter()).enumerate() {
                let expected = want / two_n;
                assert!(
                    (got - expected).abs() < 1e-9,
                    "Forward N={n}: dst_iii(dst_ii(x))[{i}] = {got}, expected x/(2N) = {expected}"
                );
            }
        }
    }

    #[test]
    fn idct_dct_metamorphic_roundtrip_under_ortho_and_forward() {
        // /testing-metamorphic: idct(dct(x)) under ortho and forward
        // normalizations. After [frankenscipy-v0vm5] fix idct now
        // pre-scales the input to invert whichever normalization dct
        // applied, so the roundtrip is the identity in all three
        // modes within f64 precision.
        for n in [4_usize, 8, 16, 11, 13] {
            let x: Vec<f64> = (0..n).map(|i| (i as f64).sin() * 0.5 + 0.25).collect();
            for norm in [super::Normalization::Ortho, super::Normalization::Forward] {
                let opts = FftOptions::default().with_normalization(norm);
                let spectrum = dct(&x, &opts).expect("dct");
                let recovered = idct(&spectrum, &opts).expect("idct");
                assert_eq!(recovered.len(), n);
                for (i, (&got, &want)) in recovered.iter().zip(x.iter()).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-9,
                        "norm={norm:?} N={n}: idct(dct(x))[{i}] = {got}, expected {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn dst_ii_short_input_matches_closed_form() {
        // /testing-conformance-harnesses: scipy.fft.dst([1, 2], type=2)
        // closed-form derivation:
        //   y[0] = 2·[sin(π/4) + 2·sin(3π/4)] = 2·(3/√2) = 3√2
        //   y[1] = 2·[sin(π/2) + 2·sin(3π/2)] = 2·(1 − 2) = −2
        let input = [1.0_f64, 2.0];
        let opts = FftOptions::default();
        let y = dst_ii(&input, &opts).expect("dst-ii");
        let expected = [3.0 * 2.0_f64.sqrt(), -2.0];
        assert_eq!(y.len(), expected.len());
        for (i, (got, want)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "dst_ii[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn dct_ii_impulse_matches_closed_form() {
        // /testing-conformance-harnesses: DCT-II of an impulse [1, 0, 0, 0]
        // has the closed-form y[k] = 2·cos(π·k / 8) (since x[0] = 1, n=4).
        // Pins scipy.fft.dct(x, type=2, norm=None) reference output.
        let input = [1.0_f64, 0.0, 0.0, 0.0];
        let opts = FftOptions::default();
        let y = dct(&input, &opts).expect("dct");
        let pi = std::f64::consts::PI;
        let expected: [f64; 4] = [
            2.0 * (0.0_f64).cos(),        // 2.0
            2.0 * (pi / 8.0).cos(),       // ~1.8478
            2.0 * (pi / 4.0).cos(),       // √2
            2.0 * (3.0 * pi / 8.0).cos(), // ~0.7654
        ];
        for (i, (got, want)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "dct(impulse)[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn fft_metamorphic_parseval_ortho_normalization() {
        // /testing-metamorphic: with ortho normalization, the FFT
        // preserves energy: Σ|x[n]|² = Σ|X[k]|². Reference-free
        // identity that catches any normalization drift in either
        // the forward path or the apply_normalization helper.
        let input: Vec<(f64, f64)> = (0..16)
            .map(|i| ((i as f64).sin(), (0.5 * i as f64).cos()))
            .collect();
        let opts = FftOptions::default().with_normalization(Normalization::Ortho);
        let spectrum = fft(&input, &opts).expect("fft");
        let energy_x: f64 = input.iter().map(|(re, im)| re * re + im * im).sum();
        let energy_xf: f64 = spectrum.iter().map(|(re, im)| re * re + im * im).sum();
        assert!(
            (energy_x - energy_xf).abs() < 1e-12,
            "Parseval (ortho): Σ|x|² = {energy_x}, Σ|X|² = {energy_xf}"
        );
    }

    #[test]
    fn fft_impulse_produces_constant_spectrum() {
        // DFT of impulse [1,0,0,0] = [1,1,1,1]
        let input = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let spectrum = fft(&input, &FftOptions::default()).expect("fft impulse");
        for &val in &spectrum {
            assert_close_complex(val, (1.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft_constant_produces_impulse_spectrum() {
        // DFT of constant [1,1,1,1] = [4,0,0,0]
        let input = vec![(1.0, 0.0); 4];
        let spectrum = fft(&input, &FftOptions::default()).expect("fft constant");
        assert_close_complex(spectrum[0], (4.0, 0.0), 1e-9);
        for &val in &spectrum[1..] {
            assert_close_complex(val, (0.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft_length_1_is_identity() {
        let input = vec![(3.5, -1.2)];
        let result = fft(&input, &FftOptions::default()).expect("fft len 1");
        assert_close_complex(result[0], input[0], 1e-12);
    }

    #[test]
    fn fft_length_2_manual_check() {
        // DFT of [a, b] = [a+b, a-b]
        let input = vec![(3.0, 0.0), (1.0, 0.0)];
        let result = fft(&input, &FftOptions::default()).expect("fft len 2");
        assert_close_complex(result[0], (4.0, 0.0), 1e-9);
        assert_close_complex(result[1], (2.0, 0.0), 1e-9);
    }

    #[test]
    fn fft_non_power_of_2_roundtrip() {
        let input: Vec<_> = (0..7).map(|i| (i as f64, 0.0)).collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft n=7");
        let recovered = ifft(&spectrum, &FftOptions::default()).expect("ifft n=7");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft_power_of_2_roundtrip() {
        let input: Vec<_> = (0..16)
            .map(|i| ((i as f64).sin(), (i as f64).cos()))
            .collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft n=16");
        let recovered = ifft(&spectrum, &FftOptions::default()).expect("ifft n=16");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft_known_sine_wave() {
        // sin(2π·k/N) for k=1 has DFT peak at bin 1 and N-1
        let n = 8;
        let input: Vec<_> = (0..n)
            .map(|k| {
                (
                    (2.0 * std::f64::consts::PI * k as f64 / n as f64).sin(),
                    0.0,
                )
            })
            .collect();
        let spectrum = fft(&input, &FftOptions::default()).expect("fft sine");
        // bin 0 should be ~0, bin 1 should have large imaginary part
        assert_close(spectrum[0].0, 0.0, 1e-9);
        assert!(spectrum[1].0.abs() < 1e-9);
        assert!(spectrum[1].1.abs() > 1.0); // non-trivial imaginary at bin 1
    }

    #[test]
    fn rfft_roundtrip_even_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let opts = FftOptions::default();
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(6), &opts).expect("irfft");
        for (a, b) in recovered.iter().zip(&input) {
            assert_close(*a, *b, 1e-9);
        }
    }

    #[test]
    fn rfft_roundtrip_odd_length() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let opts = FftOptions::default();
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(5), &opts).expect("irfft");
        for (a, b) in recovered.iter().zip(&input) {
            assert_close(*a, *b, 1e-9);
        }
    }

    #[test]
    fn rfft_hermitian_symmetry() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = rfft(&input, &FftOptions::default()).expect("rfft");
        // For real input, X[0] is real-valued
        assert_close(spectrum[0].1, 0.0, 1e-9);
        // X[N/2] (last element for even N) is real-valued
        assert_close(spectrum[spectrum.len() - 1].1, 0.0, 1e-9);
    }

    #[test]
    fn rfft_real_sine_wave() {
        let n = 16;
        let input: Vec<f64> = (0..n)
            .map(|k| (2.0 * std::f64::consts::PI * k as f64 / n as f64).sin())
            .collect();
        let spectrum = rfft(&input, &FftOptions::default()).expect("rfft sine");
        assert_eq!(spectrum.len(), n / 2 + 1);
        // DC component should be ~0
        assert_close(spectrum[0].0, 0.0, 1e-9);
    }

    #[test]
    fn fft2_impulse_2d() {
        // 2D impulse at (0,0): all spectral bins should be 1
        let input = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        let spectrum = fft2(&input, (2, 2), &FftOptions::default()).expect("fft2 impulse");
        for &val in &spectrum {
            assert_close_complex(val, (1.0, 0.0), 1e-9);
        }
    }

    #[test]
    fn fft2_non_square() {
        let input: Vec<_> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let opts = FftOptions::default();
        let spectrum = fft2(&input, (2, 3), &opts).expect("fft2 2x3");
        let recovered = ifft2(&spectrum, (2, 3), &opts).expect("ifft2 2x3");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn fft2_separable_product() {
        // f(x,y) = a(x) * b(y) => F(u,v) = A(u) * B(v)
        let a = vec![(1.0, 0.0), (2.0, 0.0)];
        let b = vec![(3.0, 0.0), (-1.0, 0.0)];
        let opts = FftOptions::default();
        let mut input = Vec::with_capacity(4);
        for ax in &a {
            for bx in &b {
                input.push((ax.0 * bx.0, 0.0));
            }
        }
        let result_2d = fft2(&input, (2, 2), &opts).expect("fft2");
        let fa = fft(&a, &opts).expect("fft a");
        let fb = fft(&b, &opts).expect("fft b");
        for i in 0..2 {
            for j in 0..2 {
                let expected = complex_mul_test(fa[i], fb[j]);
                assert_close_complex(result_2d[i * 2 + j], expected, 1e-9);
            }
        }
    }

    #[test]
    fn normalization_forward_scales_forward_pass() {
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let opts_backward = FftOptions::default();
        let opts_forward = FftOptions::default().with_normalization(Normalization::Forward);
        let backward = fft(&input, &opts_backward).expect("backward");
        let forward = fft(&input, &opts_forward).expect("forward");
        let n = input.len() as f64;
        for (&bv, &fv) in backward.iter().zip(&forward) {
            assert_close_complex(fv, (bv.0 / n, bv.1 / n), 1e-9);
        }
    }

    #[test]
    fn normalization_ortho_is_symmetric() {
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let opts = FftOptions::default().with_normalization(Normalization::Ortho);
        let spectrum = fft(&input, &opts).expect("fft ortho");
        let recovered = ifft(&spectrum, &opts).expect("ifft ortho");
        for (&a, &b) in recovered.iter().zip(&input) {
            assert_close_complex(a, b, 1e-9);
        }
    }

    #[test]
    fn check_finite_rejects_nan_complex() {
        let opts = FftOptions::default().with_check_finite(true);
        let err = fft(&[(1.0, f64::NAN)], &opts).expect_err("NaN should be rejected");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    #[test]
    fn check_finite_rejects_inf_real() {
        let opts = FftOptions::default().with_check_finite(true);
        let err = rfft(&[1.0, f64::INFINITY], &opts).expect_err("Inf should be rejected");
        assert_eq!(err, FftError::NonFiniteInput);
    }

    fn complex_mul_test(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[test]
    fn repeated_calls_emit_plan_cache_hits_in_trace() {
        // Hold the shared-plan-cache test lock so this test does not race
        // with the plan.rs::tests::shared_cache_* suite under cargo test's
        // default parallelism (frankenscipy-lw3rl).
        let _g = shared_cache_test_lock();
        clear_shared_plan_cache();
        let _ = take_transform_traces();

        // Use n=137 (prime, large, unlikely to collide in parallel tests)
        let input = (0..137usize)
            .map(|i| (i as f64, (i % 3) as f64 - 1.0))
            .collect::<Vec<_>>();
        let opts = FftOptions::default();
        let _ = fft(&input, &opts).expect("first fft should succeed");
        let _ = fft(&input, &opts).expect("second fft should succeed");

        let mut fft_traces = take_transform_traces()
            .into_iter()
            .filter(|trace| trace.kind == TransformKind::Fft && trace.n == 137)
            .collect::<Vec<_>>();
        fft_traces.sort_by(|lhs, rhs| lhs.operation_id.cmp(&rhs.operation_id));

        assert!(
            fft_traces.len() >= 2,
            "Expected at least 2 traces for n=137, got {}",
            fft_traces.len()
        );
        let last_two = &fft_traces[fft_traces.len() - 2..];
        assert!(
            !last_two[0].plan_cache_hit,
            "First call should be a cache miss"
        );
        assert!(
            last_two[1].plan_cache_hit,
            "Second call should be a cache hit"
        );
        assert!(last_two[0].to_json_line().contains("\"operation_id\""));
    }

    #[test]
    fn plan_cache_flop_estimate_tracks_fft_complexity() {
        assert_eq!(estimate_fft_flops(0), 0);
        assert_eq!(estimate_fft_flops(1), 0);
        assert_eq!(estimate_fft_flops(1024), 1024 * 10 * 5);
        assert!(estimate_fft_flops(1024) < 1024 * 1024);
    }

    // ── rfft2 / irfft2 tests ───────────────────────────────────────

    #[test]
    fn rfft2_irfft2_roundtrip() {
        let rows = 4;
        let cols = 6;
        let input: Vec<f64> = (0..(rows * cols)).map(|i| (i as f64 * 0.3).sin()).collect();
        let opts = FftOptions::default();
        let spectrum = rfft2(&input, (rows, cols), &opts).expect("rfft2");
        let recovered = irfft2(&spectrum, (rows, cols), &opts).expect("irfft2");
        assert_eq!(recovered.len(), input.len());
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "rfft2 roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rfft2_output_shape() {
        let rows = 4;
        let cols = 8;
        let input = vec![0.0; rows * cols];
        let opts = FftOptions::default();
        let spectrum = rfft2(&input, (rows, cols), &opts).expect("rfft2");
        // Output should have rows * (cols/2 + 1) elements
        assert_eq!(spectrum.len(), rows * (cols / 2 + 1));
    }

    #[test]
    fn rfftn_irfftn_roundtrip() {
        let shape = [2, 3, 4];
        let input: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|i| (i as f64 * 0.2).cos())
            .collect();
        let opts = FftOptions::default();
        let spectrum = rfftn(&input, &shape, &opts).expect("rfftn");
        let recovered = irfftn(&spectrum, &shape, &opts).expect("irfftn");
        assert_eq!(recovered.len(), input.len());
        for (i, (&a, &b)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "rfftn roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rfftn_output_shape() {
        let shape = [2, 3, 8];
        let input = vec![0.0; shape.iter().product()];
        let opts = FftOptions::default();
        let spectrum = rfftn(&input, &shape, &opts).expect("rfftn");
        assert_eq!(spectrum.len(), shape[0] * shape[1] * (shape[2] / 2 + 1));
    }

    // ── next_fast_len tests ────────────────────────────────────────

    #[test]
    fn next_fast_len_powers_of_2() {
        assert_eq!(next_fast_len(1), 1);
        assert_eq!(next_fast_len(2), 2);
        assert_eq!(next_fast_len(4), 4);
        assert_eq!(next_fast_len(8), 8);
    }

    #[test]
    fn next_fast_len_non_powers() {
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(next_fast_len(11), 12); // 12 = 2^2 * 3
        assert_eq!(next_fast_len(13), 15); // 15 = 3 * 5
        assert_eq!(next_fast_len(17), 18); // 18 = 2 * 3^2
    }

    #[test]
    fn next_fast_len_already_fast() {
        // 30 = 2 * 3 * 5
        assert_eq!(next_fast_len(30), 30);
        // 100 = 2^2 * 5^2
        assert_eq!(next_fast_len(100), 100);
    }

    // ── dctn / idctn tests ─────────────────────────────────────────

    #[test]
    fn dctn_idctn_roundtrip_2d_under_all_normalizations() {
        // /mock-code-finder regression for [frankenscipy-ilbpb]:
        // dctn and idctn ship as scipy.fft parity surfaces but had
        // ZERO direct tests. Verify the n-D roundtrip identity
        // idctn(dctn(x, shape), shape) ≈ x on a 4×4 input across all
        // 3 normalizations.
        let x: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5 + 1.0).collect();
        let shape = [4usize, 4];
        for norm in [
            Normalization::Backward,
            Normalization::Forward,
            Normalization::Ortho,
        ] {
            let opts = FftOptions {
                normalization: norm,
                ..FftOptions::default()
            };
            let forward = dctn(&x, &shape, &opts).expect("dctn");
            let recovered = idctn(&forward, &shape, &opts).expect("idctn");
            assert_eq!(recovered.len(), x.len());
            for (i, (&got, &orig)) in recovered.iter().zip(x.iter()).enumerate() {
                assert!(
                    (got - orig).abs() < 1e-9 + 1e-9 * orig.abs(),
                    "dctn↔idctn[{i}] = {got}, expected {orig} (norm={norm:?})"
                );
            }
        }
    }

    #[test]
    fn dctn_1d_shape_matches_dct_directly() {
        // For a 1-D shape, dctn must reduce to the 1-D dct exactly.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let opts = FftOptions::default();
        let from_dctn = dctn(&x, &[8usize], &opts).expect("dctn 1d");
        let from_dct = dct(&x, &opts).expect("dct");
        assert_eq!(from_dctn.len(), from_dct.len());
        for (i, (&a, &b)) in from_dctn.iter().zip(from_dct.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "dctn(x, [8])[{i}] = {a} != dct(x)[{i}] = {b}"
            );
        }
    }

    #[test]
    fn dctn_rejects_shape_length_mismatch() {
        // shape product != input length → error.
        let x = vec![1.0; 12];
        let opts = FftOptions::default();
        assert!(dctn(&x, &[3usize, 5], &opts).is_err()); // 15 != 12
        assert!(dctn(&x, &[2usize, 4], &opts).is_err()); // 8  != 12

        // empty shape → error.
        assert!(dctn(&x, &[], &opts).is_err());
    }

    // ── prev_fast_len tests ────────────────────────────────────────

    #[test]
    fn prev_fast_len_already_fast_returns_self() {
        // Each {2,3,5}-smooth value is its own predecessor.
        assert_eq!(prev_fast_len(1), 1);
        assert_eq!(prev_fast_len(2), 2);
        assert_eq!(prev_fast_len(8), 8);
        assert_eq!(prev_fast_len(30), 30); // 2·3·5
        assert_eq!(prev_fast_len(100), 100); // 2²·5²
    }

    #[test]
    fn prev_fast_len_non_smooth() {
        assert_eq!(prev_fast_len(7), 6); // 6 = 2·3, 7 prime
        assert_eq!(prev_fast_len(11), 10); // 10 = 2·5
        assert_eq!(prev_fast_len(13), 12); // 12 = 2²·3
        assert_eq!(prev_fast_len(17), 16); // 16 = 2⁴
        assert_eq!(prev_fast_len(31), 30); // 30 = 2·3·5
    }

    #[test]
    fn prev_fast_len_zero_input() {
        assert_eq!(prev_fast_len(0), 0);
    }

    #[test]
    fn prev_next_fast_len_bracket_target() {
        // For any target ≥ 1, prev_fast_len(t) ≤ t ≤ next_fast_len(t).
        for t in 1usize..=128 {
            let p = prev_fast_len(t);
            let n = next_fast_len(t);
            assert!(p <= t, "prev_fast_len({t}) = {p} should be ≤ {t}");
            assert!(t <= n, "next_fast_len({t}) = {n} should be ≥ {t}");
            assert!(is_fast_len(p), "prev_fast_len({t}) = {p} not 5-smooth");
            assert!(is_fast_len(n), "next_fast_len({t}) = {n} not 5-smooth");
        }
    }

    // ── hfft tests ─────────────────────────────────────────────────

    #[test]
    fn hfft_basic() {
        // hfft of a half-spectrum should produce real output
        let opts = FftOptions::default();
        let input = vec![(4.0, 0.0), (1.0, -1.0), (0.0, 0.0)]; // 3 elements → n=4
        let result = hfft(&input, Some(4), &opts).expect("hfft");
        assert_eq!(result.len(), 4);
        // All values should be real (no imaginary component issue)
        for &v in &result {
            assert!(v.is_finite(), "hfft produced non-finite value: {v}");
        }
    }

    #[test]
    fn hfft_rfft_inverse() {
        // hfft = n * irfft(conj(x), n), so hfft(conj(rfft(x)), n) / n = x.
        let opts = FftOptions::default();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let n = x.len();
        let spectrum = rfft(&x, &opts)
            .expect("rfft")
            .into_iter()
            .map(|(re, im)| (re, -im))
            .collect::<Vec<_>>();
        let recovered = hfft(&spectrum, Some(n), &opts).expect("hfft");
        for (i, (&a, &b)) in x.iter().zip(recovered.iter()).enumerate() {
            let b_scaled = b / n as f64;
            assert!(
                (a - b_scaled).abs() < 1e-9,
                "hfft roundtrip mismatch at {i}: {a} vs {b_scaled}"
            );
        }
    }

    #[test]
    fn hfft_empty_input_returns_invalid_shape() {
        let err = hfft(&[], None, &FftOptions::default()).expect_err("empty hfft must fail");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "input length must be greater than zero",
            }
        );
    }

    #[test]
    fn ihfft_basic() {
        use super::ihfft;
        // ihfft should produce a Hermitian-symmetric spectrum
        let opts = FftOptions::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = ihfft(&input, None, &opts).expect("ihfft");
        // For n=4 real input, rfft gives n/2+1 = 3 complex outputs
        assert_eq!(result.len(), 3);
        for &(re, im) in &result {
            assert!(
                re.is_finite() && im.is_finite(),
                "ihfft produced non-finite"
            );
        }
    }

    #[test]
    fn ihfft_hfft_roundtrip() {
        use super::ihfft;
        // ihfft(hfft(spectrum)) should recover the original spectrum (with scaling)
        let opts = FftOptions::default();
        let spectrum: Vec<(f64, f64)> = vec![(4.0, 0.0), (1.0, -1.0), (0.0, 0.0)];
        let n = 4; // output length for hfft
        let real_signal = hfft(&spectrum, Some(n), &opts).expect("hfft");
        let recovered = ihfft(&real_signal, None, &opts).expect("ihfft");
        // recovered should approximately equal spectrum
        for (i, (&(re1, im1), &(re2, im2))) in spectrum.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (re1 - re2).abs() < 1e-9,
                "re mismatch at {i}: {re1} vs {re2}"
            );
            assert!(
                (im1 - im2).abs() < 1e-9,
                "im mismatch at {i}: {im1} vs {im2}"
            );
        }
    }

    #[test]
    fn ihfftn_hfftn_roundtrip_3d() {
        let opts = FftOptions::default();
        let shape = [2usize, 3, 4];
        let signal: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|i| i as f64 - 5.0)
            .collect();
        let spectrum = ihfftn(&signal, &shape, &opts).expect("ihfftn");
        assert_eq!(spectrum.len(), shape[0] * shape[1] * (shape[2] / 2 + 1));

        let recovered = hfftn(&spectrum, &shape, &opts).expect("hfftn");
        assert_eq!(recovered.len(), signal.len());
        for (i, (&got, &want)) in recovered.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "hfftn(ihfftn(x))[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn ihfft2_hfft2_roundtrip_2d() {
        let opts = FftOptions::default();
        let shape = (3usize, 4usize);
        let signal: Vec<f64> = (0..shape.0 * shape.1)
            .map(|i| (i as f64 * 0.5) - 2.0)
            .collect();
        let spectrum = ihfft2(&signal, shape, &opts).expect("ihfft2");
        assert_eq!(spectrum.len(), shape.0 * (shape.1 / 2 + 1));

        let recovered = hfft2(&spectrum, shape, &opts).expect("hfft2");
        assert_eq!(recovered.len(), signal.len());
        for (i, (&got, &want)) in recovered.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "hfft2(ihfft2(x))[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn fftn_matches_scipy_reference_values() {
        // scipy.fft.fftn([[1, 2], [3, 4]])
        // -> [[10+0j, -2+0j], [-4+0j, 0+0j]]
        let opts = FftOptions::default();
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let shape = vec![2, 2];
        let result = fftn(&input, &shape, &opts).expect("fftn");
        // Check DC component: sum of all inputs = 10
        assert!(
            (result[0].0 - 10.0).abs() < 1e-10,
            "fftn[0] re got {}, expected 10",
            result[0].0
        );
        assert!(result[0].1.abs() < 1e-10, "fftn[0] im should be ~0");
    }

    #[test]
    fn ifftn_fftn_roundtrip_matches_scipy() {
        // scipy.fft.ifftn(scipy.fft.fftn(x)) == x
        let opts = FftOptions::default();
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let shape = vec![2, 2];
        let spectrum = fftn(&input, &shape, &opts).expect("fftn");
        let recovered = ifftn(&spectrum, &shape, &opts).expect("ifftn");
        for (i, (got, want)) in recovered.iter().zip(input.iter()).enumerate() {
            assert!(
                (got.0 - want.0).abs() < 1e-10,
                "ifftn(fftn(x))[{i}] re got {}, expected {}",
                got.0,
                want.0
            );
            assert!(
                (got.1 - want.1).abs() < 1e-10,
                "ifftn(fftn(x))[{i}] im got {}, expected {}",
                got.1,
                want.1
            );
        }
    }

    #[test]
    fn fft2_matches_scipy_reference_values() {
        // scipy.fft.fft2([[1, 2], [3, 4]]) == fftn for 2D
        // -> [[10+0j, -2+0j], [-4+0j, 0+0j]]
        let opts = FftOptions::default();
        let input = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        let shape = (2usize, 2usize);
        let result = fft2(&input, shape, &opts).expect("fft2");
        // DC component: 10
        assert!(
            (result[0].0 - 10.0).abs() < 1e-10,
            "fft2[0] re got {}, expected 10",
            result[0].0
        );
        // result[1] should be -2
        assert!(
            (result[1].0 - (-2.0)).abs() < 1e-10,
            "fft2[1] re got {}, expected -2",
            result[1].0
        );
    }

    #[test]
    fn rfft_matches_scipy_reference_values() {
        // scipy.fft.rfft([1, 2, 3, 4]) -> [10+0j, -2+2j, -2+0j]
        let opts = FftOptions::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = rfft(&input, &opts).expect("rfft");
        // Length should be n/2 + 1 = 3
        assert_eq!(result.len(), 3, "rfft output length");
        // DC component: sum = 10
        assert!(
            (result[0].0 - 10.0).abs() < 1e-10,
            "rfft[0] re got {}, expected 10",
            result[0].0
        );
        // rfft[1] = -2 + 2j
        assert!(
            (result[1].0 - (-2.0)).abs() < 1e-10,
            "rfft[1] re got {}, expected -2",
            result[1].0
        );
        assert!(
            (result[1].1 - 2.0).abs() < 1e-10,
            "rfft[1] im got {}, expected 2",
            result[1].1
        );
    }

    #[test]
    fn dct_matches_scipy_reference_values() {
        // scipy.fft.dct([1, 2, 3, 4], type=2)
        // -> [20, -6.30..., 0, -0.448...]
        let opts = FftOptions::default();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = dct(&input, &opts).expect("dct");
        assert_eq!(result.len(), 4);
        // First coefficient should be 2*sum = 20
        assert!(
            (result[0] - 20.0).abs() < 1e-6,
            "dct[0] got {}, expected 20",
            result[0]
        );
    }

    #[test]
    fn idct_dct_roundtrip_matches_scipy() {
        // scipy.fft.idct(scipy.fft.dct(x)) == x (with normalization)
        let opts = FftOptions::default().with_normalization(Normalization::Ortho);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = dct(&input, &opts).expect("dct");
        let recovered = idct(&spectrum, &opts).expect("idct");
        for (i, (&got, &want)) in recovered.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "idct(dct(x))[{i}] got {got}, expected {want}"
            );
        }
    }
}
