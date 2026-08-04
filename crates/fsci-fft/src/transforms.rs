use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

pub use fsci_runtime::SyncSharedAuditLedger;
use fsci_runtime::{AuditAction, AuditEvent, AuditLedger, RuntimeMode, casp_now_unix_ms};

use crate::plan::{
    PlanFingerprint, PlanKey, PlanMetadata, PlanningStrategy, store_shared_plan, touch_shared_plan,
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
            // Radix-2² (fused-radix-4) sweep: halves the number of full passes
            // over `data` vs the flat radix-2 kernel (the memory-bandwidth
            // bottleneck at large n) while staying bit-identical to it. The
            // top-level single transform fans each stage's disjoint groups across
            // cores (byte-identical; SciPy's 1-D FFT is single-threaded).
            let twiddles = get_or_compute_twiddles(n, inverse);
            cooley_tukey_radix4_inplace_with_twiddles_par(data, &twiddles);
        } else {
            // Non-power-of-2: the hot {3,5}*2^k family uses an iterative
            // stage plan so each stage reuses one twiddle table across all
            // same-size subtransforms. Other shapes keep the general recursive
            // mixed-radix/Bluestein route.
            let mut scratch = vec![(0.0, 0.0); n];
            if !mixed_radix_iterative_odd_power_tail(data, &mut scratch, inverse) {
                mixed_radix_fft(data, 0, 1, &mut scratch, n, inverse);
            }
            data.copy_from_slice(&scratch);
        }
    }
}

static COOLEY_TUKEY_BACKEND: CooleyTukeyBackend = CooleyTukeyBackend;

/// Runtime switch to force the Bluestein path for large-prime FFTs (bypassing the
/// Rader route) for same-binary A/B benchmarks. Defaults off. `#[doc(hidden)]`.
#[doc(hidden)]
pub static FFT_PRIME_FORCE_BLUESTEIN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

thread_local! {
    /// Per-thread twiddle cache: a lock-free fast path in front of the shared
    /// [`TWIDDLE_CACHE`]. In parallel transform loops (e.g. N-D DCT/DST/rfft
    /// fiber passes) every worker re-fetches the same `(n, inverse)` table per
    /// fiber; routing those hits through a thread-local map removes the global
    /// `RwLock` read on each call (64 threads otherwise ping-pong the lock's
    /// reader counter across 1M+ acquisitions). The cached `Arc<[Complex64]>` is
    /// shared with the global cache, so the returned twiddles are bit-identical.
    static LOCAL_TWIDDLE_CACHE: std::cell::RefCell<HashMap<TwiddleKey, TwiddleTable>> =
        std::cell::RefCell::new(HashMap::new());
}

fn get_or_compute_twiddles(n: usize, inverse: bool) -> TwiddleTable {
    let key = (n, inverse);

    // Lock-free per-thread fast path.
    if let Some(table) = LOCAL_TWIDDLE_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return table;
    }

    let cache = get_twiddle_cache();
    let table = if let Some(table) = cache.read().ok().and_then(|guard| guard.get(&key).cloned()) {
        table
    } else {
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
    };

    LOCAL_TWIDDLE_CACHE.with(|c| c.borrow_mut().insert(key, Arc::clone(&table)));
    table
}

#[derive(Debug)]
struct OddPowerTailPlan {
    factors: Box<[usize]>,
    tail: usize,
    leaf_bases: Box<[usize]>,
}

type OddPowerTailPlanRef = Arc<OddPowerTailPlan>;
static ODD_POWER_TAIL_PLAN_CACHE: OnceLock<RwLock<HashMap<usize, OddPowerTailPlanRef>>> =
    OnceLock::new();

fn get_odd_power_tail_plan_cache() -> &'static RwLock<HashMap<usize, OddPowerTailPlanRef>> {
    ODD_POWER_TAIL_PLAN_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

thread_local! {
    static LOCAL_ODD_POWER_TAIL_PLAN_CACHE: std::cell::RefCell<HashMap<usize, OddPowerTailPlanRef>> =
        std::cell::RefCell::new(HashMap::new());
}

fn get_or_compute_odd_power_tail_plan(n: usize) -> Option<OddPowerTailPlanRef> {
    if n <= 1 || n.is_power_of_two() {
        return None;
    }

    if let Some(plan) =
        LOCAL_ODD_POWER_TAIL_PLAN_CACHE.with(|cache| cache.borrow().get(&n).cloned())
    {
        return Some(plan);
    }

    let cache = get_odd_power_tail_plan_cache();
    if let Some(plan) = cache.read().ok().and_then(|guard| guard.get(&n).cloned()) {
        LOCAL_ODD_POWER_TAIL_PLAN_CACHE.with(|local| {
            local.borrow_mut().insert(n, Arc::clone(&plan));
        });
        return Some(plan);
    }

    let (factors, tail) = odd_power_tail_factorization(n)?;
    // Odd-heavy shapes (small power-of-two `tail`) thrash the iterative tail's
    // strided leaf gather: it runs `n/tail` leaves, each gathering `tail` samples
    // `n/tail` apart, so at large n nearly every read is a cache miss (n=101250=
    // 2·3⁴·5⁴ → tail=2 → 50625 stride-50625 gathers ≈ 13 ms). The recursive
    // `mixed_radix_fft` is far more cache-friendly and wins for medium n — BUT it
    // is single-threaded, so above ~500k the iterative path's cross-leaf/stage
    // parallelism overtakes it (n=600000 recursive 20 ms vs iterative 17;
    // n=781250 recursive 42 ms). So decline the iterative plan (→ recursive) only
    // for odd-heavy `tail ≤ 64` in the medium band [2¹⁶, 500 000]; larger n and
    // larger (more-power-of-two) tails keep the parallel iterative path. Measured
    // wins vs iterative: 100 000 7.2→1.5 ms, 200 000 9.7→3.1, 250 000 11.9→6.6,
    // 405 000 13→7.4, 202 500 11.7→3.7, 101 250 13→2.1; tail==1 (pure odd) and
    // n>500k (e.g. 810 000 iter 17 ms, beats numpy) stay on the iterative path.
    if (2..=64).contains(&tail) && (1 << 16..=500_000).contains(&n) {
        return None;
    }
    let odd_len = n / tail;
    let leaf_bases = (0..odd_len)
        .map(|leaf| mixed_radix_leaf_base(leaf, &factors))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let plan = Arc::new(OddPowerTailPlan {
        factors: factors.into_boxed_slice(),
        tail,
        leaf_bases,
    });

    if let Ok(mut guard) = cache.write() {
        guard.insert(n, Arc::clone(&plan));
    }
    LOCAL_ODD_POWER_TAIL_PLAN_CACHE.with(|local| {
        local.borrow_mut().insert(n, Arc::clone(&plan));
    });

    Some(plan)
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

type RaderKey = (usize, bool);
#[derive(Clone)]
struct RaderPlan {
    pow_g: Vec<usize>,     // g^j mod p — input gather permutation
    out_idx: Vec<usize>,   // g^{-q} mod p — output scatter permutation
    c_fft: Vec<Complex64>, // FFT of the (reversed) convolution kernel c̃
}
static RADER_CACHE: OnceLock<RwLock<HashMap<RaderKey, RaderPlan>>> = OnceLock::new();

fn get_rader_cache() -> &'static RwLock<HashMap<RaderKey, RaderPlan>> {
    RADER_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Smallest even 5-smooth number (2^a·3^b·5^c with a ≥ 1) ≥ `n`.
///
/// Bluestein converts a length-`n` DFT into a circular convolution of length
/// `m ≥ 2n-1`; any such `m` is valid. The previous choice `next_power_of_two`
/// is the smallest *power of two* ≥ 2n-1, but fsci's mixed-radix engine has fast
/// radix-2/3/4/5 butterflies, so an even 5-smooth `m` — always ≤ next_pow2 and
/// computed on the optimized `{3,5}·2^k` iterative path — does strictly fewer
/// points (up to ~1.9× when 2n-1 sits just above a power of two). `next_fast_len`
/// is the wrong helper: it admits factor 7 (e.g. 3·7³), which hits a slow
/// large-prime stage. Even length (a ≥ 1) keeps it off the pure-odd recursive path.
fn next_regular_even(n: usize) -> usize {
    if n <= 2 {
        return 2;
    }
    let mut best = usize::MAX;
    let mut p5 = 1usize;
    while p5 < n.saturating_mul(2) {
        let mut p35 = p5;
        while p35 < n.saturating_mul(2) {
            let mut q = p35.saturating_mul(2);
            while q < n {
                q = q.saturating_mul(2);
            }
            best = best.min(q);
            if p35 > n {
                break;
            }
            p35 = p35.saturating_mul(3);
        }
        if p5 > n {
            break;
        }
        p5 = p5.saturating_mul(5);
    }
    best
}

/// Unscaled in-place transform used inside Bluestein for the convolution length
/// `m`. For a power-of-two `m` this stays on the original flat radix-2 kernel (no
/// parallel-fan-out overhead, which matters for the small `m` Bluestein hits);
/// for the even 5-smooth shapes it uses the mixed-radix backend. `m` is always
/// 5-smooth so the backend never recurses back into Bluestein.
#[inline]
fn bluestein_transform_inplace(data: &mut [Complex64], inverse: bool) {
    if data.len().is_power_of_two() {
        cooley_tukey_radix2_inplace(data, inverse);
    } else {
        COOLEY_TUKEY_BACKEND.transform_1d_inplace(data, inverse);
    }
}

fn get_or_compute_bluestein_plan(n: usize, inverse: bool) -> BluesteinPlan {
    let cache = get_bluestein_cache();
    let key = (n, inverse);

    if let Some(plan) = cache.read().ok().and_then(|guard| guard.get(&key).cloned()) {
        return plan;
    }

    // Bluestein needs any m >= 2n-1. The even 5-smooth length is <= next_pow2 and
    // does fewer points, but the mixed-radix transform has a higher per-point
    // constant + a scratch alloc, so it only wins when it's SUBSTANTIALLY smaller
    // (it is exactly when 2n-1 sits just above a power of two — m_pow2 nearly
    // doubles). When the two are close (2n-1 just below a pow2, or a small prime
    // factor), keep the pow2 length on the fast flat radix-2 path. Gate at <=60%.
    let m_pow2 = (2 * n - 1).next_power_of_two();
    let m_5smooth = next_regular_even(2 * n - 1);
    let m = if m_5smooth * 5 <= m_pow2 * 3 {
        m_5smooth
    } else {
        m_pow2
    };
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
    // m is now even 5-smooth (not necessarily a power of two), so route through
    // the general unscaled transform: radix-4 for pow2 m (bit-identical to the
    // old radix-2), mixed-radix for the 5-smooth shapes. Convention is unchanged
    // (unscaled forward + inverse), so the manual 1/m scaling below still holds.
    bluestein_transform_inplace(&mut b, false);

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

/// Radix-2² (split-stage radix-4) Cooley-Tukey FFT for power-of-2 lengths.
///
/// Algebraically fuses every pair of consecutive radix-2 stages into a single
/// radix-4 butterfly, halving the number of full passes over `data` (the
/// memory-bandwidth bottleneck at large `n`). All three twiddle factors are
/// fetched from the *same* full-length table at the *same indices* the two
/// radix-2 stages would use, and the adds/mults run in the same order, so the
/// result is **bit-identical** to [`cooley_tukey_radix2_inplace_with_twiddles`].
/// An odd `log2(n)` is handled with one leading radix-2 stage.
/// One fused radix-4 stage applied to `block`, which holds a whole number of
/// size-`4l` groups laid out contiguously (group `g` at `g*4l`). The twiddle
/// index pattern depends only on the in-group butterfly index `k` (not the group
/// base), so this is correct whether `block` is the full array or one parallel
/// sub-run of groups. Single source of truth for the butterfly → serial and
/// parallel kernels are byte-identical by construction.
#[inline]
fn radix4_stage_run(
    block: &mut [Complex64],
    l: usize,
    stride2: usize,
    stride4: usize,
    quarter: usize,
    twiddles: &[Complex64],
) {
    let mut base = 0;
    while base < block.len() {
        for k in 0..l {
            let a = base + k;
            let b = a + l;
            let c = b + l;
            let d = c + l;
            let wa = twiddles[k * stride2];
            let wb1 = twiddles[k * stride4];
            let wb2 = twiddles[k * stride4 + quarter];

            // Inner radix-2 stage (size 2l) on (a,b) and (c,d).
            let tb = complex_mul(block[b], wa);
            let a1 = complex_add(block[a], tb);
            let b1 = complex_sub(block[a], tb);
            let td = complex_mul(block[d], wa);
            let c1 = complex_add(block[c], td);
            let d1 = complex_sub(block[c], td);

            // Outer radix-2 stage (size 4l) on (a1,c1) and (b1,d1).
            let tc = complex_mul(c1, wb1);
            block[a] = complex_add(a1, tc);
            block[c] = complex_sub(a1, tc);
            let td2 = complex_mul(d1, wb2);
            block[b] = complex_add(b1, td2);
            block[d] = complex_sub(b1, td2);
        }
        base += 4 * l;
    }
}

/// Bit-reverse + leading-radix-2 (odd stage count) prologue shared by the serial
/// and parallel radix-4 sweeps. Returns the starting `l`.
#[inline]
fn radix4_prologue(data: &mut [Complex64], log_n: usize) -> usize {
    apply_bit_reverse_permutation_incremental(data);
    if log_n % 2 == 1 {
        let mut base = 0;
        let n = data.len();
        while base < n {
            let even = data[base];
            let odd = data[base + 1];
            data[base] = complex_add(even, odd);
            data[base + 1] = complex_sub(even, odd);
            base += 2;
        }
        2
    } else {
        1
    }
}

fn cooley_tukey_radix4_inplace_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= n);
    let log_n = n.trailing_zeros() as usize;
    let mut l = radix4_prologue(data, log_n);
    let quarter = n / 4;
    // Radix-4 fused stages: combine four size-`l` blocks into one size-`4l`.
    while l < n {
        let stride2 = n / (2 * l);
        let stride4 = n / (4 * l);
        radix4_stage_run(data, l, stride2, stride4, quarter, twiddles);
        l *= 4;
    }
}

/// Parallel single-FFT radix-4 sweep: within each stage the size-`4l` groups are
/// DISJOINT CONTIGUOUS blocks, so fan them across cores (chunks_mut, one barrier
/// per stage via thread::scope). BIT-IDENTICAL to the serial sweep — same
/// `radix4_stage_run` math, same per-group order; only which core owns a group
/// changes. Used ONLY by the top-level single pow2 transform (never nested under
/// the already-parallel non-pow2 leaf phase). SciPy's 1-D FFT is single-threaded.
fn cooley_tukey_radix4_inplace_with_twiddles_par(data: &mut [Complex64], twiddles: &[Complex64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    debug_assert!(twiddles.len() >= n);
    let log_n = n.trailing_zeros() as usize;
    let mut l = radix4_prologue(data, log_n);
    let quarter = n / 4;
    while l < n {
        let stride2 = n / (2 * l);
        let stride4 = n / (4 * l);
        let group_len = 4 * l;
        let ngroups = n / group_len;
        let nthreads = fft_radix4_par_threads(n, ngroups);
        if nthreads <= 1 {
            radix4_stage_run(data, l, stride2, stride4, quarter, twiddles);
        } else {
            let groups_per = ngroups.div_ceil(nthreads);
            std::thread::scope(|scope| {
                for chunk in data.chunks_mut(groups_per * group_len) {
                    scope.spawn(move || {
                        radix4_stage_run(chunk, l, stride2, stride4, quarter, twiddles);
                    });
                }
            });
        }
        l *= 4;
    }
}

fn apply_bit_reverse_permutation_incremental(data: &mut [Complex64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
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

    bluestein_transform_inplace(&mut a, false);
    for (value, &b_fft) in a.iter_mut().zip(&plan.b_fft) {
        *value = complex_mul(*value, b_fft);
    }
    bluestein_transform_inplace(&mut a, true);

    let inv_m = 1.0 / m as f64;
    a.iter()
        .zip(&plan.chirp)
        .map(|(&value, &chirp)| complex_mul(chirp, complex_scale(value, inv_m)))
        .collect()
}

/// `base^exp mod modulus` (modulus fits in u32-range FFT lengths, products in u64).
fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

/// Distinct prime factors of `n` (ascending).
fn distinct_prime_factors(mut n: usize) -> Vec<usize> {
    let mut f = Vec::new();
    let mut d = 2usize;
    while d * d <= n {
        if n.is_multiple_of(d) {
            f.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// Largest prime factor of `n` (n ≥ 2) by trial division.
fn largest_prime_factor(mut n: usize) -> usize {
    let mut largest = 1usize;
    let mut d = 2usize;
    while d * d <= n {
        while n.is_multiple_of(d) {
            largest = d;
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        largest = n;
    }
    largest
}

/// Smallest primitive root of a prime `p` (generator of the multiplicative group).
fn primitive_root(p: usize) -> usize {
    let phi = p - 1;
    let factors = distinct_prime_factors(phi);
    for g in 2..p {
        if factors
            .iter()
            .all(|&q| pow_mod(g as u64, (phi / q) as u64, p as u64) != 1)
        {
            return g;
        }
    }
    p - 1
}

/// Rader's algorithm: a prime-length-`p` DFT as a length-`(p-1)` cyclic
/// convolution. Cheaper than Bluestein (whose convolution length is ≥ 2p-1) when
/// `p-1` is itself FFT-friendly. Gated by the caller to primes where `p-1` has no
/// prime factor > `MIXED_RADIX_DIRECT_MAX_PRIME`, so the inner length-`(p-1)`
/// transforms never recurse back into Bluestein/Rader. Unscaled, matching the
/// engine convention.
fn get_or_compute_rader_plan(p: usize, inverse: bool) -> RaderPlan {
    let cache = get_rader_cache();
    let key = (p, inverse);
    if let Some(plan) = cache.read().ok().and_then(|guard| guard.get(&key).cloned()) {
        return plan;
    }

    let l = p - 1;
    let g = primitive_root(p);
    // pow_g[j] = g^j mod p (j = 0..l); enumerates every nonzero residue once.
    let mut pow_g = vec![0usize; l];
    let mut acc = 1usize;
    for slot in pow_g.iter_mut() {
        *slot = acc;
        acc = acc * g % p;
    }
    // output index g^{-q} mod p = pow_g[(l-q) mod l]
    let out_idx: Vec<usize> = (0..l).map(|q| pow_g[(l - q) % l]).collect();
    // c[m] = W^{g^m mod p}; W = exp(±2πi/p). c̃[m] = c[(l-m) mod l] turns the
    // cyclic correlation X[g^{-q}] = x0 + Σ_j a_j c[(j-q) mod l] into the cyclic
    // convolution (a ⊛ c̃)[q]. Store FFT(c̃) so each transform reuses it.
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut c_fft = vec![(0.0, 0.0); l];
    for m in 0..l {
        let e = pow_g[(l - m) % l];
        let ang = sign * 2.0 * PI * (e as f64) / (p as f64);
        c_fft[m] = (ang.cos(), ang.sin());
    }
    COOLEY_TUKEY_BACKEND.transform_1d_inplace(&mut c_fft, false);

    let plan = RaderPlan {
        pow_g,
        out_idx,
        c_fft,
    };
    if let Ok(mut guard) = cache.write() {
        guard.insert(key, plan.clone());
    }
    plan
}

fn rader_fft(input: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let p = input.len();
    let plan = get_or_compute_rader_plan(p, inverse);
    let l = p - 1;
    // a[j] = x[g^j]; cyclic conv via FFT (both length l, unscaled, /l on inverse).
    let mut a: Vec<Complex64> = plan.pow_g.iter().map(|&idx| input[idx]).collect();
    COOLEY_TUKEY_BACKEND.transform_1d_inplace(&mut a, false);
    for (av, &cv) in a.iter_mut().zip(&plan.c_fft) {
        *av = complex_mul(*av, cv);
    }
    COOLEY_TUKEY_BACKEND.transform_1d_inplace(&mut a, true);

    let x0 = input.iter().fold((0.0, 0.0), |acc, &v| complex_add(acc, v));
    let inv_l = 1.0 / l as f64;
    let mut out = vec![(0.0, 0.0); p];
    out[0] = x0;
    for (q, &idx) in plan.out_idx.iter().enumerate() {
        out[idx] = complex_add(input[0], complex_scale(a[q], inv_l));
    }
    out
}

/// Largest prime length still handled by an in-place O(p²) DFT before falling
/// back to Bluestein. Small odd-prime factors (3, 5, 7, …) bottom out here
/// cheaply; only genuinely large residual prime factors go through Bluestein.
const MIXED_RADIX_DIRECT_MAX_PRIME: usize = 61;

/// Smallest prime factor of `n` (n ≥ 2) by trial division; returns `n` if prime.
fn smallest_prime_factor(n: usize) -> usize {
    if n.is_multiple_of(2) {
        return 2;
    }
    let mut f = 3;
    while f * f <= n {
        if n.is_multiple_of(f) {
            return f;
        }
        f += 2;
    }
    n
}

fn smallest_odd_prime_factor(n: usize) -> Option<usize> {
    let odd_part = n >> n.trailing_zeros();
    (odd_part > 1).then(|| smallest_prime_factor(odd_part))
}

fn mixed_radix_split_factor(n: usize) -> usize {
    if let Some(p) = smallest_odd_prime_factor(n) {
        // A large odd prime is far cheaper transformed *whole* (Rader/Bluestein,
        // O(p log p)) than used as a radix-`p` combine — the general-`p` branch of
        // the combine is a direct O(p²) DFT across blocks. When the smallest odd
        // prime factor is large but `n` still has a power-of-two part, peel the
        // 2/4 first so the large prime is reached as a standalone prime length
        // (e.g. 502 = 2·251 → peel 2 → 251 via Rader, not a radix-251 O(251²)
        // combine; this compounds badly when nested, e.g. 2510 = 2·5·251).
        if p > MIXED_RADIX_DIRECT_MAX_PRIME && n.is_multiple_of(2) {
            if n.is_multiple_of(4) { 4 } else { 2 }
        } else {
            p
        }
    } else if n.is_multiple_of(4) {
        4
    } else {
        2
    }
}

/// Recursive mixed-radix Cooley-Tukey (decimation-in-time).
///
/// Reads the length-`n` input as `x[t] = src[base + t·stride]` and writes the
/// unscaled DFT (sign per `inverse`, like the radix-2 / Bluestein kernels) into
/// `out[0..n]`. For a composite `n = p·m` (`p` = smallest prime factor):
///   X[r + m·u] = Σ_{j<p} (W_n^{j·r} · X_j[r]) · W_p^{j·u}
/// where `X_j = DFT_m` of the stride-`p` decimated sub-sequence `x[j + p·s]`.
/// The `p` sub-FFTs land in disjoint length-`m` blocks of `out`; the combine
/// step twiddles and applies a length-`p` DFT across blocks (radix-2 hard-coded,
/// other small primes via a direct length-`p` DFT). Prime lengths bottom out in
/// a direct O(p²) DFT, or Bluestein when large.
fn mixed_radix_fft(
    src: &[Complex64],
    base: usize,
    stride: usize,
    out: &mut [Complex64],
    n: usize,
    inverse: bool,
) {
    if n.is_power_of_two() {
        for (t, slot) in out.iter_mut().enumerate().take(n) {
            *slot = src[base + t * stride];
        }
        if n <= 16 {
            mixed_radix_small_power_tail(&mut out[..n], inverse);
        } else {
            let twiddles = get_or_compute_twiddles(n, inverse);
            cooley_tukey_radix4_inplace_with_twiddles(&mut out[..n], &twiddles);
        }
        return;
    }

    // Peel odd factors first so smooth lengths bottom out in the optimized
    // radix-2² power tail instead of thousands of tiny strided odd-prime DFTs.
    let p = mixed_radix_split_factor(n);
    if p == n {
        if n > MIXED_RADIX_DIRECT_MAX_PRIME {
            // Large prime: gather the strided samples. Rader (a length-(n-1) cyclic
            // convolution) beats Bluestein (length ≥ 2n-1, ALWAYS >2× larger + a chirp
            // multiply) whenever the inner FFT(n-1) itself stays on fast kernels. fsci
            // now runs {2,3,5,7} on dedicated radices and 11/13 on a cheap O(p²) direct
            // DFT, so the Rader break-even sits at n-1 being {2,3,5,7,11,13}-smooth (was
            // 5-smooth, stale — it forced e.g. 1409→1408=2⁷·11 onto Bluestein even
            // though FFT(1408) is fast). Above 13 the direct-DFT combine (up to O(61²))
            // rivals Bluestein's pow2 FFT, so keep those on Bluestein.
            let gathered: Vec<Complex64> = (0..n).map(|t| src[base + t * stride]).collect();
            let spectrum = if largest_prime_factor(n - 1) <= 13
                && !FFT_PRIME_FORCE_BLUESTEIN.load(std::sync::atomic::Ordering::Relaxed)
            {
                rader_fft(&gathered, inverse)
            } else {
                bluestein_fft(&gathered, inverse)
            };
            out[..n].copy_from_slice(&spectrum);
        } else {
            let tw = get_or_compute_twiddles(n, inverse);
            for (k, slot) in out.iter_mut().enumerate().take(n) {
                let mut acc = (0.0, 0.0);
                for t in 0..n {
                    acc = complex_add(acc, complex_mul(src[base + t * stride], tw[(t * k) % n]));
                }
                *slot = acc;
            }
        }
        return;
    }

    let m = n / p;
    // p sub-FFTs of length m over the stride-p decimated sub-sequences, written
    // into disjoint contiguous blocks of `out`.
    for (j, block) in out.chunks_mut(m).enumerate().take(p) {
        mixed_radix_fft(src, base + j * stride, stride * p, block, m, inverse);
    }

    // Combine: twiddle each block then apply the length-p DFT across blocks.
    let twn = get_or_compute_twiddles(n, inverse);
    if p == 4 {
        // Radix-4 butterfly: halves the passes over the power-of-2 part versus
        // two radix-2 stages. z = DFT_4 of the twiddled blocks (W_4 = ∓i).
        for r in 0..m {
            let t0 = out[r];
            let t1 = complex_mul(out[m + r], twn[r % n]);
            let t2 = complex_mul(out[2 * m + r], twn[(2 * r) % n]);
            let t3 = complex_mul(out[3 * m + r], twn[(3 * r) % n]);
            let a02 = complex_add(t0, t2);
            let b02 = complex_sub(t0, t2);
            let a13 = complex_add(t1, t3);
            let b13 = complex_sub(t1, t3);
            // ∓i·b13: forward (-i)·(x,y)=(y,-x); inverse (+i)·(x,y)=(-y,x).
            let rot = if inverse {
                (-b13.1, b13.0)
            } else {
                (b13.1, -b13.0)
            };
            out[r] = complex_add(a02, a13);
            out[m + r] = complex_add(b02, rot);
            out[2 * m + r] = complex_sub(a02, a13);
            out[3 * m + r] = complex_sub(b02, rot);
        }
    } else if p == 2 {
        for r in 0..m {
            let a = out[r];
            let b = complex_mul(out[m + r], twn[r % n]);
            out[r] = complex_add(a, b);
            out[m + r] = complex_sub(a, b);
        }
    } else if p == 3 {
        // Radix-3 butterfly. c = cos(2π/3) = -1/2, s = sin(2π/3).
        const S3: f64 = 0.866_025_403_784_438_6;
        let s = if inverse { -S3 } else { S3 };
        for r in 0..m {
            let t0 = out[r];
            let t1 = complex_mul(out[m + r], twn[r % n]);
            let t2 = complex_mul(out[2 * m + r], twn[(2 * r) % n]);
            let psum = complex_add(t1, t2);
            let pdif = complex_sub(t1, t2);
            let a = (t0.0 - 0.5 * psum.0, t0.1 - 0.5 * psum.1);
            out[r] = complex_add(t0, psum);
            out[m + r] = (a.0 + s * pdif.1, a.1 - s * pdif.0);
            out[2 * m + r] = (a.0 - s * pdif.1, a.1 + s * pdif.0);
        }
    } else if p == 5 {
        // Radix-5 butterfly. c1=cos(2π/5), c2=cos(4π/5), s1=sin(2π/5), s2=sin(4π/5).
        const C1: f64 = 0.309_016_994_374_947_45;
        const C2: f64 = -0.809_016_994_374_947_4;
        const S1: f64 = 0.951_056_516_295_153_6;
        const S2: f64 = 0.587_785_252_292_473_1;
        let (s1, s2) = if inverse { (-S1, -S2) } else { (S1, S2) };
        for r in 0..m {
            let t0 = out[r];
            let t1 = complex_mul(out[m + r], twn[r % n]);
            let t2 = complex_mul(out[2 * m + r], twn[(2 * r) % n]);
            let t3 = complex_mul(out[3 * m + r], twn[(3 * r) % n]);
            let t4 = complex_mul(out[4 * m + r], twn[(4 * r) % n]);
            let t1p4 = complex_add(t1, t4);
            let t1m4 = complex_sub(t1, t4);
            let t2p3 = complex_add(t2, t3);
            let t2m3 = complex_sub(t2, t3);
            let a1 = (
                t0.0 + C1 * t1p4.0 + C2 * t2p3.0,
                t0.1 + C1 * t1p4.1 + C2 * t2p3.1,
            );
            let a2 = (
                t0.0 + C2 * t1p4.0 + C1 * t2p3.0,
                t0.1 + C2 * t1p4.1 + C1 * t2p3.1,
            );
            let b1 = (s1 * t1m4.0 + s2 * t2m3.0, s1 * t1m4.1 + s2 * t2m3.1);
            let b2 = (s2 * t1m4.0 - s1 * t2m3.0, s2 * t1m4.1 - s1 * t2m3.1);
            out[r] = (t0.0 + t1p4.0 + t2p3.0, t0.1 + t1p4.1 + t2p3.1);
            out[m + r] = (a1.0 + b1.1, a1.1 - b1.0);
            out[2 * m + r] = (a2.0 + b2.1, a2.1 - b2.0);
            out[3 * m + r] = (a2.0 - b2.1, a2.1 + b2.0);
            out[4 * m + r] = (a1.0 - b1.1, a1.1 + b1.0);
        }
    } else {
        let twp = get_or_compute_twiddles(p, inverse);
        let mut tmp = vec![(0.0, 0.0); p];
        for r in 0..m {
            for (j, slot) in tmp.iter_mut().enumerate() {
                *slot = complex_mul(out[j * m + r], twn[(j * r) % n]);
            }
            for u in 0..p {
                let mut acc = (0.0, 0.0);
                for (j, &t) in tmp.iter().enumerate() {
                    acc = complex_add(acc, complex_mul(t, twp[(j * u) % p]));
                }
                out[u * m + r] = acc;
            }
        }
    }
}

fn mixed_radix_iterative_odd_power_tail(
    src: &[Complex64],
    out: &mut [Complex64],
    inverse: bool,
) -> bool {
    let n = src.len();
    debug_assert_eq!(out.len(), n);
    if n <= 1 || n.is_power_of_two() {
        return false;
    }
    let Some(plan) = get_or_compute_odd_power_tail_plan(n) else {
        return false;
    };

    let tail = plan.tail;
    debug_assert_eq!(plan.leaf_bases.len(), n / tail);
    debug_assert_eq!(
        plan.factors.iter().product::<usize>(),
        plan.leaf_bases.len()
    );

    // Leaf phase: gather each strided 2^k tail once, then run the existing
    // power-of-two kernel on a contiguous block. This is the same DIT factor
    // order as the recursive route; only scheduling/twiddle reuse changes.
    // Each leaf writes a DISJOINT contiguous `out` block from read-only `src`, so
    // fan the leaves across cores — BIT-IDENTICAL (per-leaf FFT is deterministic;
    // only which core owns a leaf changes). SciPy's 1-D FFT is single-threaded.
    let nleaves = plan.leaf_bases.len();
    let leaf_threads = fft_iter_par_threads(n, nleaves);
    if leaf_threads <= 1 {
        for (leaf, &base) in plan.leaf_bases.iter().enumerate() {
            let block = &mut out[leaf * tail..(leaf + 1) * tail];
            leaf_tail_fft(src, base, nleaves, tail, block, inverse, None);
        }
    } else {
        let tail_tw = (tail > 16).then(|| get_or_compute_twiddles(tail, inverse));
        let tail_tw_ref: Option<&[Complex64]> = tail_tw.as_deref();
        let leaves_per = nleaves.div_ceil(leaf_threads);
        std::thread::scope(|scope| {
            for (out_chunk, bases_chunk) in out
                .chunks_mut(leaves_per * tail)
                .zip(plan.leaf_bases.chunks(leaves_per))
            {
                scope.spawn(move || {
                    for (li, &base) in bases_chunk.iter().enumerate() {
                        let block = &mut out_chunk[li * tail..(li + 1) * tail];
                        leaf_tail_fft(src, base, nleaves, tail, block, inverse, tail_tw_ref);
                    }
                });
            }
        });
    }

    // Combine from the innermost odd factor outwards. All groups at a stage
    // share the same twiddle table, avoiding recursive per-group cache lookups.
    // Groups within a stage are DISJOINT contiguous blocks → fan across cores,
    // BIT-IDENTICAL (each group's combine is independent and deterministic).
    let mut m = tail;
    for &p in plan.factors.iter().rev() {
        let stage_len = p * m;
        let twn = get_or_compute_twiddles(stage_len, inverse);
        let ngroups = n / stage_len;
        let stage_threads = fft_iter_par_threads(n, ngroups);
        if stage_threads <= 1 {
            for group in out.chunks_exact_mut(stage_len) {
                mixed_radix_combine_stage(group, p, m, &twn, inverse);
            }
        } else {
            let twn_ref: &[Complex64] = &twn;
            let groups_per = ngroups.div_ceil(stage_threads);
            std::thread::scope(|scope| {
                for out_chunk in out.chunks_mut(groups_per * stage_len) {
                    scope.spawn(move || {
                        for group in out_chunk.chunks_exact_mut(stage_len) {
                            mixed_radix_combine_stage(group, p, m, twn_ref, inverse);
                        }
                    });
                }
            });
        }
        m = stage_len;
    }
    true
}

/// One leaf of the iterative odd-power tail: gather the strided 2^k sub-sequence
/// at `base` (stride `nleaves`) into `block` and run the power-of-two kernel.
/// `tail_tw` is the precomputed `tail` twiddle table (for `tail > 16`); pass
/// `None` to look it up on demand (serial path).
#[inline]
fn leaf_tail_fft(
    src: &[Complex64],
    base: usize,
    nleaves: usize,
    tail: usize,
    block: &mut [Complex64],
    inverse: bool,
    tail_tw: Option<&[Complex64]>,
) {
    if tail <= 16 {
        mixed_radix_gather_small_power_tail(src, base, nleaves, block, inverse);
    } else {
        for (s, slot) in block.iter_mut().enumerate() {
            *slot = src[base + s * nleaves];
        }
        match tail_tw {
            Some(tw) => cooley_tukey_radix4_inplace_with_twiddles(block, tw),
            None => {
                let twiddles = get_or_compute_twiddles(tail, inverse);
                cooley_tukey_radix4_inplace_with_twiddles(block, &twiddles);
            }
        }
    }
}

/// Worker count for the iterative odd-power-tail phases: serial below 1<<16 work
/// or fewer than 2 independent blocks; otherwise `min(cores, nblocks)`.
fn fft_iter_par_threads(n: usize, nblocks: usize) -> usize {
    if n < (1 << 16) || nblocks < 2 {
        return 1;
    }
    dispatch_worker_limit().min(nblocks)
}

/// Worker count for the single-FFT radix-4 sweep. The sweep re-spawns workers
/// once PER STAGE (~log4(n) barriers), so the per-stage work must amortize the
/// spawn cost: gate at n ≥ 1<<20 and cap at 16 workers (a single 1-D FFT is
/// memory-bandwidth-bound — more cores oversubscribe without scaling, and fewer
/// spawns/stage keeps the overhead small). Below the gate the serial sweep is
/// already ≈ SciPy parity, so staying serial avoids a small-size regression.
fn fft_radix4_par_threads(n: usize, ngroups: usize) -> usize {
    if n < (1 << 20) || ngroups < 2 {
        return 1;
    }
    dispatch_worker_limit().min(ngroups).min(16)
}

fn mixed_radix_gather_small_power_tail(
    src: &[Complex64],
    base: usize,
    stride: usize,
    block: &mut [Complex64],
    inverse: bool,
) {
    match block.len() {
        0 => {}
        1 => block[0] = src[base],
        2 => {
            let mut tmp = [src[base], src[base + stride]];
            mixed_radix_small_power_tail(&mut tmp, inverse);
            block.copy_from_slice(&tmp);
        }
        4 => {
            let mut tmp = [
                src[base],
                src[base + stride],
                src[base + 2 * stride],
                src[base + 3 * stride],
            ];
            fft4_kernel(&mut tmp, inverse);
            block.copy_from_slice(&tmp);
        }
        8 => {
            let mut tmp = [
                src[base],
                src[base + stride],
                src[base + 2 * stride],
                src[base + 3 * stride],
                src[base + 4 * stride],
                src[base + 5 * stride],
                src[base + 6 * stride],
                src[base + 7 * stride],
            ];
            fft8_kernel(&mut tmp, inverse);
            block.copy_from_slice(&tmp);
        }
        16 => {
            let mut tmp = [
                src[base],
                src[base + stride],
                src[base + 2 * stride],
                src[base + 3 * stride],
                src[base + 4 * stride],
                src[base + 5 * stride],
                src[base + 6 * stride],
                src[base + 7 * stride],
                src[base + 8 * stride],
                src[base + 9 * stride],
                src[base + 10 * stride],
                src[base + 11 * stride],
                src[base + 12 * stride],
                src[base + 13 * stride],
                src[base + 14 * stride],
                src[base + 15 * stride],
            ];
            fft16_kernel(&mut tmp, inverse);
            block.copy_from_slice(&tmp);
        }
        _ => unreachable!("small power tail only supports powers of two up to 16"),
    }
}

fn odd_power_tail_factorization(mut n: usize) -> Option<(Vec<usize>, usize)> {
    let mut factors = Vec::new();
    while !n.is_power_of_two() {
        let odd_part = n >> n.trailing_zeros();
        if odd_part <= 1 {
            break;
        }
        let p = smallest_prime_factor(odd_part);
        if p != 3 && p != 5 && p != 7 && p != 11 && p != 13 && p != 17 {
            return None;
        }
        factors.push(p);
        n /= p;
    }
    (!factors.is_empty() && n.is_power_of_two()).then_some((factors, n))
}

fn mixed_radix_leaf_base(mut leaf: usize, factors: &[usize]) -> usize {
    let mut divisor = factors.iter().product::<usize>();
    let mut base = 0usize;
    let mut input_stride = 1usize;
    for &p in factors {
        divisor /= p;
        let digit = leaf / divisor;
        leaf %= divisor;
        base += digit * input_stride;
        input_stride *= p;
    }
    base
}

fn mixed_radix_combine_stage(
    out: &mut [Complex64],
    p: usize,
    m: usize,
    twn: &[Complex64],
    inverse: bool,
) {
    debug_assert_eq!(out.len(), p * m);
    if p == 3 {
        const S3: f64 = 0.866_025_403_784_438_6;
        let s = if inverse { -S3 } else { S3 };
        let (out0, tail) = out.split_at_mut(m);
        let (out1, out2) = tail.split_at_mut(m);
        for r in 0..m {
            let t0 = out0[r];
            let (x1r, x1i) = out1[r];
            let (w1r, w1i) = twn[r];
            let t1 = (x1r * w1r - x1i * w1i, x1r * w1i + x1i * w1r);
            let (x2r, x2i) = out2[r];
            let (w2r, w2i) = twn[2 * r];
            let t2 = (x2r * w2r - x2i * w2i, x2r * w2i + x2i * w2r);
            let psum = (t1.0 + t2.0, t1.1 + t2.1);
            let pdif = (t1.0 - t2.0, t1.1 - t2.1);
            let a = (t0.0 - 0.5 * psum.0, t0.1 - 0.5 * psum.1);
            out0[r] = (t0.0 + psum.0, t0.1 + psum.1);
            out1[r] = (a.0 + s * pdif.1, a.1 - s * pdif.0);
            out2[r] = (a.0 - s * pdif.1, a.1 + s * pdif.0);
        }
    } else if p == 5 {
        const C1: f64 = 0.309_016_994_374_947_45;
        const C2: f64 = -0.809_016_994_374_947_4;
        const S1: f64 = 0.951_056_516_295_153_6;
        const S2: f64 = 0.587_785_252_292_473_1;
        let (s1, s2) = if inverse { (-S1, -S2) } else { (S1, S2) };
        let (out0, tail) = out.split_at_mut(m);
        let (out1, tail) = tail.split_at_mut(m);
        let (out2, tail) = tail.split_at_mut(m);
        let (out3, out4) = tail.split_at_mut(m);
        for r in 0..m {
            let t0 = out0[r];
            let (x1r, x1i) = out1[r];
            let (w1r, w1i) = twn[r];
            let t1 = (x1r * w1r - x1i * w1i, x1r * w1i + x1i * w1r);
            let (x2r, x2i) = out2[r];
            let (w2r, w2i) = twn[2 * r];
            let t2 = (x2r * w2r - x2i * w2i, x2r * w2i + x2i * w2r);
            let (x3r, x3i) = out3[r];
            let (w3r, w3i) = twn[3 * r];
            let t3 = (x3r * w3r - x3i * w3i, x3r * w3i + x3i * w3r);
            let (x4r, x4i) = out4[r];
            let (w4r, w4i) = twn[4 * r];
            let t4 = (x4r * w4r - x4i * w4i, x4r * w4i + x4i * w4r);
            let t1p4 = (t1.0 + t4.0, t1.1 + t4.1);
            let t1m4 = (t1.0 - t4.0, t1.1 - t4.1);
            let t2p3 = (t2.0 + t3.0, t2.1 + t3.1);
            let t2m3 = (t2.0 - t3.0, t2.1 - t3.1);
            let a1 = (
                t0.0 + C1 * t1p4.0 + C2 * t2p3.0,
                t0.1 + C1 * t1p4.1 + C2 * t2p3.1,
            );
            let a2 = (
                t0.0 + C2 * t1p4.0 + C1 * t2p3.0,
                t0.1 + C2 * t1p4.1 + C1 * t2p3.1,
            );
            let b1 = (s1 * t1m4.0 + s2 * t2m3.0, s1 * t1m4.1 + s2 * t2m3.1);
            let b2 = (s2 * t1m4.0 - s1 * t2m3.0, s2 * t1m4.1 - s1 * t2m3.1);
            out0[r] = (t0.0 + t1p4.0 + t2p3.0, t0.1 + t1p4.1 + t2p3.1);
            out1[r] = (a1.0 + b1.1, a1.1 - b1.0);
            out2[r] = (a2.0 + b2.1, a2.1 - b2.0);
            out3[r] = (a2.0 - b2.1, a2.1 + b2.0);
            out4[r] = (a1.0 - b1.1, a1.1 + b1.0);
        }
    } else if p == 7 {
        // Radix-7 combine: three conjugate pairs (1,6)(2,5)(3,4). Cosine sums
        // give the six a-terms, sine differences the b-terms, mirroring the
        // radix-5 structure. cₖ = cos(2πk/7), sₖ = sin(2πk/7) for k=1,2,3;
        // outᵤ = aᵤ ∓ i·bᵤ with its conjugate partner out_{7−u} = aᵤ ± i·bᵤ.
        const C1: f64 = 0.623_489_801_858_733_5;
        const C2: f64 = -0.222_520_933_956_314_4;
        const C3: f64 = -0.900_968_867_902_419_1;
        const S1: f64 = 0.781_831_482_468_029_8;
        const S2: f64 = 0.974_927_912_181_823_6;
        const S3: f64 = 0.433_883_739_117_558_1;
        let (s1, s2, s3) = if inverse {
            (-S1, -S2, -S3)
        } else {
            (S1, S2, S3)
        };
        let (out0, tail) = out.split_at_mut(m);
        let (out1, tail) = tail.split_at_mut(m);
        let (out2, tail) = tail.split_at_mut(m);
        let (out3, tail) = tail.split_at_mut(m);
        let (out4, tail) = tail.split_at_mut(m);
        let (out5, out6) = tail.split_at_mut(m);
        for r in 0..m {
            let t0 = out0[r];
            let (x1r, x1i) = out1[r];
            let (w1r, w1i) = twn[r];
            let t1 = (x1r * w1r - x1i * w1i, x1r * w1i + x1i * w1r);
            let (x2r, x2i) = out2[r];
            let (w2r, w2i) = twn[2 * r];
            let t2 = (x2r * w2r - x2i * w2i, x2r * w2i + x2i * w2r);
            let (x3r, x3i) = out3[r];
            let (w3r, w3i) = twn[3 * r];
            let t3 = (x3r * w3r - x3i * w3i, x3r * w3i + x3i * w3r);
            let (x4r, x4i) = out4[r];
            let (w4r, w4i) = twn[4 * r];
            let t4 = (x4r * w4r - x4i * w4i, x4r * w4i + x4i * w4r);
            let (x5r, x5i) = out5[r];
            let (w5r, w5i) = twn[5 * r];
            let t5 = (x5r * w5r - x5i * w5i, x5r * w5i + x5i * w5r);
            let (x6r, x6i) = out6[r];
            let (w6r, w6i) = twn[6 * r];
            let t6 = (x6r * w6r - x6i * w6i, x6r * w6i + x6i * w6r);
            // conjugate-pair sums (p) and differences (m)
            let p1 = (t1.0 + t6.0, t1.1 + t6.1);
            let m1 = (t1.0 - t6.0, t1.1 - t6.1);
            let p2 = (t2.0 + t5.0, t2.1 + t5.1);
            let m2 = (t2.0 - t5.0, t2.1 - t5.1);
            let p3 = (t3.0 + t4.0, t3.1 + t4.1);
            let m3 = (t3.0 - t4.0, t3.1 - t4.1);
            let a1 = (
                t0.0 + C1 * p1.0 + C2 * p2.0 + C3 * p3.0,
                t0.1 + C1 * p1.1 + C2 * p2.1 + C3 * p3.1,
            );
            let a2 = (
                t0.0 + C2 * p1.0 + C3 * p2.0 + C1 * p3.0,
                t0.1 + C2 * p1.1 + C3 * p2.1 + C1 * p3.1,
            );
            let a3 = (
                t0.0 + C3 * p1.0 + C1 * p2.0 + C2 * p3.0,
                t0.1 + C3 * p1.1 + C1 * p2.1 + C2 * p3.1,
            );
            let b1 = (
                s1 * m1.0 + s2 * m2.0 + s3 * m3.0,
                s1 * m1.1 + s2 * m2.1 + s3 * m3.1,
            );
            let b2 = (
                s2 * m1.0 - s3 * m2.0 - s1 * m3.0,
                s2 * m1.1 - s3 * m2.1 - s1 * m3.1,
            );
            let b3 = (
                s3 * m1.0 - s1 * m2.0 + s2 * m3.0,
                s3 * m1.1 - s1 * m2.1 + s2 * m3.1,
            );
            out0[r] = (t0.0 + p1.0 + p2.0 + p3.0, t0.1 + p1.1 + p2.1 + p3.1);
            out1[r] = (a1.0 + b1.1, a1.1 - b1.0);
            out2[r] = (a2.0 + b2.1, a2.1 - b2.0);
            out3[r] = (a3.0 + b3.1, a3.1 - b3.0);
            out4[r] = (a3.0 - b3.1, a3.1 + b3.0);
            out5[r] = (a2.0 - b2.1, a2.1 + b2.0);
            out6[r] = (a1.0 - b1.1, a1.1 + b1.0);
        }
    } else if p == 11 {
        // Radix-11 combine: five conjugate pairs (1,10)(2,9)(3,8)(4,7)(5,6). Same
        // conjugate-symmetry structure as radix-5/7 — cosine sums build the five
        // a-terms, sine differences the b-terms, with the (k·u mod 11) cosine/sine
        // permutation per output. cₖ=cos(2πk/11), sₖ=sin(2πk/11) for k=1..5;
        // outᵤ = aᵤ ∓ i·bᵤ, out_{11−u} = aᵤ ± i·bᵤ.
        const C1: f64 = 0.841_253_532_831_181_2;
        const C2: f64 = 0.415_415_013_001_886_4;
        const C3: f64 = -0.142_314_838_273_285_1;
        const C4: f64 = -0.654_860_733_945_285_1;
        const C5: f64 = -0.959_492_973_614_497_4;
        const S1: f64 = 0.540_640_817_455_597_6;
        const S2: f64 = 0.909_631_995_354_518_4;
        const S3: f64 = 0.989_821_441_880_932_7;
        const S4: f64 = 0.755_749_574_354_258_3;
        const S5: f64 = 0.281_732_556_841_429_7;
        let (s1, s2, s3, s4, s5) = if inverse {
            (-S1, -S2, -S3, -S4, -S5)
        } else {
            (S1, S2, S3, S4, S5)
        };
        let (out0, tail) = out.split_at_mut(m);
        let (out1, tail) = tail.split_at_mut(m);
        let (out2, tail) = tail.split_at_mut(m);
        let (out3, tail) = tail.split_at_mut(m);
        let (out4, tail) = tail.split_at_mut(m);
        let (out5, tail) = tail.split_at_mut(m);
        let (out6, tail) = tail.split_at_mut(m);
        let (out7, tail) = tail.split_at_mut(m);
        let (out8, tail) = tail.split_at_mut(m);
        let (out9, out10) = tail.split_at_mut(m);
        for r in 0..m {
            let t0 = out0[r];
            let tw = |blk: &[Complex64], j: usize| -> (f64, f64) {
                let (xr, xi) = blk[r];
                let (wr, wi) = twn[j * r];
                (xr * wr - xi * wi, xr * wi + xi * wr)
            };
            let t1 = tw(out1, 1);
            let t2 = tw(out2, 2);
            let t3 = tw(out3, 3);
            let t4 = tw(out4, 4);
            let t5 = tw(out5, 5);
            let t6 = tw(out6, 6);
            let t7 = tw(out7, 7);
            let t8 = tw(out8, 8);
            let t9 = tw(out9, 9);
            let t10 = tw(out10, 10);
            let p1 = (t1.0 + t10.0, t1.1 + t10.1);
            let q1 = (t1.0 - t10.0, t1.1 - t10.1);
            let p2 = (t2.0 + t9.0, t2.1 + t9.1);
            let q2 = (t2.0 - t9.0, t2.1 - t9.1);
            let p3 = (t3.0 + t8.0, t3.1 + t8.1);
            let q3 = (t3.0 - t8.0, t3.1 - t8.1);
            let p4 = (t4.0 + t7.0, t4.1 + t7.1);
            let q4 = (t4.0 - t7.0, t4.1 - t7.1);
            let p5 = (t5.0 + t6.0, t5.1 + t6.1);
            let q5 = (t5.0 - t6.0, t5.1 - t6.1);
            // a_u = t0 + Σ cos(2π·k·u/11)·p_k ; permutation from (k·u mod 11).
            let acc = |ca: f64, cb: f64, cc: f64, cd: f64, ce: f64| -> (f64, f64) {
                (
                    t0.0 + ca * p1.0 + cb * p2.0 + cc * p3.0 + cd * p4.0 + ce * p5.0,
                    t0.1 + ca * p1.1 + cb * p2.1 + cc * p3.1 + cd * p4.1 + ce * p5.1,
                )
            };
            let a1 = acc(C1, C2, C3, C4, C5);
            let a2 = acc(C2, C4, C5, C3, C1);
            let a3 = acc(C3, C5, C2, C1, C4);
            let a4 = acc(C4, C3, C1, C5, C2);
            let a5 = acc(C5, C1, C4, C2, C3);
            // b_u = Σ sin(2π·k·u/11)·q_k (sign from the mod-11 reflection).
            let bcc = |sa: f64, sb: f64, sc: f64, sd: f64, se: f64| -> (f64, f64) {
                (
                    sa * q1.0 + sb * q2.0 + sc * q3.0 + sd * q4.0 + se * q5.0,
                    sa * q1.1 + sb * q2.1 + sc * q3.1 + sd * q4.1 + se * q5.1,
                )
            };
            let b1 = bcc(s1, s2, s3, s4, s5);
            let b2 = bcc(s2, s4, -s5, -s3, -s1);
            let b3 = bcc(s3, -s5, -s2, s1, s4);
            let b4 = bcc(s4, -s3, s1, s5, -s2);
            let b5 = bcc(s5, -s1, s4, -s2, s3);
            out0[r] = (
                t0.0 + p1.0 + p2.0 + p3.0 + p4.0 + p5.0,
                t0.1 + p1.1 + p2.1 + p3.1 + p4.1 + p5.1,
            );
            out1[r] = (a1.0 + b1.1, a1.1 - b1.0);
            out10[r] = (a1.0 - b1.1, a1.1 + b1.0);
            out2[r] = (a2.0 + b2.1, a2.1 - b2.0);
            out9[r] = (a2.0 - b2.1, a2.1 + b2.0);
            out3[r] = (a3.0 + b3.1, a3.1 - b3.0);
            out8[r] = (a3.0 - b3.1, a3.1 + b3.0);
            out4[r] = (a4.0 + b4.1, a4.1 - b4.0);
            out7[r] = (a4.0 - b4.1, a4.1 + b4.0);
            out5[r] = (a5.0 + b5.1, a5.1 - b5.0);
            out6[r] = (a5.0 - b5.1, a5.1 + b5.0);
        }
    } else if p == 13 {
        // Radix-13 combine: six conjugate pairs (1,12)(2,11)(3,10)(4,9)(5,8)(6,7).
        // cₖ=cos(2πk/13), sₖ=sin(2πk/13) for k=1..6; a-terms are cosine sums with
        // the (k·u mod 13) permutation, b-terms sine diffs with the reflection sign.
        const C1: f64 = 0.885_456_025_653_21;
        const C2: f64 = 0.568_064_746_731_155_9;
        const C3: f64 = 0.120_536_680_255_323_1;
        const C4: f64 = -0.354_604_887_042_535_6;
        const C5: f64 = -0.748_510_748_171_101_1;
        const C6: f64 = -0.970_941_817_426_052;
        const S1: f64 = 0.464_723_172_043_768_6;
        const S2: f64 = 0.822_983_865_893_656_4;
        const S3: f64 = 0.992_708_874_098_054;
        const S4: f64 = 0.935_016_242_685_414_8;
        const S5: f64 = 0.663_122_658_240_795_3;
        const S6: f64 = 0.239_315_664_287_557_8;
        let (s1, s2, s3, s4, s5, s6) = if inverse {
            (-S1, -S2, -S3, -S4, -S5, -S6)
        } else {
            (S1, S2, S3, S4, S5, S6)
        };
        let (out0, tail) = out.split_at_mut(m);
        let (out1, tail) = tail.split_at_mut(m);
        let (out2, tail) = tail.split_at_mut(m);
        let (out3, tail) = tail.split_at_mut(m);
        let (out4, tail) = tail.split_at_mut(m);
        let (out5, tail) = tail.split_at_mut(m);
        let (out6, tail) = tail.split_at_mut(m);
        let (out7, tail) = tail.split_at_mut(m);
        let (out8, tail) = tail.split_at_mut(m);
        let (out9, tail) = tail.split_at_mut(m);
        let (out10, tail) = tail.split_at_mut(m);
        let (out11, out12) = tail.split_at_mut(m);
        for r in 0..m {
            let t0 = out0[r];
            let tw = |blk: &[Complex64], j: usize| -> (f64, f64) {
                let (xr, xi) = blk[r];
                let (wr, wi) = twn[j * r];
                (xr * wr - xi * wi, xr * wi + xi * wr)
            };
            let t1 = tw(out1, 1);
            let t2 = tw(out2, 2);
            let t3 = tw(out3, 3);
            let t4 = tw(out4, 4);
            let t5 = tw(out5, 5);
            let t6 = tw(out6, 6);
            let t7 = tw(out7, 7);
            let t8 = tw(out8, 8);
            let t9 = tw(out9, 9);
            let t10 = tw(out10, 10);
            let t11 = tw(out11, 11);
            let t12 = tw(out12, 12);
            let p1 = (t1.0 + t12.0, t1.1 + t12.1);
            let q1 = (t1.0 - t12.0, t1.1 - t12.1);
            let p2 = (t2.0 + t11.0, t2.1 + t11.1);
            let q2 = (t2.0 - t11.0, t2.1 - t11.1);
            let p3 = (t3.0 + t10.0, t3.1 + t10.1);
            let q3 = (t3.0 - t10.0, t3.1 - t10.1);
            let p4 = (t4.0 + t9.0, t4.1 + t9.1);
            let q4 = (t4.0 - t9.0, t4.1 - t9.1);
            let p5 = (t5.0 + t8.0, t5.1 + t8.1);
            let q5 = (t5.0 - t8.0, t5.1 - t8.1);
            let p6 = (t6.0 + t7.0, t6.1 + t7.1);
            let q6 = (t6.0 - t7.0, t6.1 - t7.1);
            let acc = |c1: f64, c2: f64, c3: f64, c4: f64, c5: f64, c6: f64| -> (f64, f64) {
                (
                    t0.0 + c1 * p1.0 + c2 * p2.0 + c3 * p3.0 + c4 * p4.0 + c5 * p5.0 + c6 * p6.0,
                    t0.1 + c1 * p1.1 + c2 * p2.1 + c3 * p3.1 + c4 * p4.1 + c5 * p5.1 + c6 * p6.1,
                )
            };
            let a1 = acc(C1, C2, C3, C4, C5, C6);
            let a2 = acc(C2, C4, C6, C5, C3, C1);
            let a3 = acc(C3, C6, C4, C1, C2, C5);
            let a4 = acc(C4, C5, C1, C3, C6, C2);
            let a5 = acc(C5, C3, C2, C6, C1, C4);
            let a6 = acc(C6, C1, C5, C2, C4, C3);
            let bcc = |v1: f64, v2: f64, v3: f64, v4: f64, v5: f64, v6: f64| -> (f64, f64) {
                (
                    v1 * q1.0 + v2 * q2.0 + v3 * q3.0 + v4 * q4.0 + v5 * q5.0 + v6 * q6.0,
                    v1 * q1.1 + v2 * q2.1 + v3 * q3.1 + v4 * q4.1 + v5 * q5.1 + v6 * q6.1,
                )
            };
            let b1 = bcc(s1, s2, s3, s4, s5, s6);
            let b2 = bcc(s2, s4, s6, -s5, -s3, -s1);
            let b3 = bcc(s3, s6, -s4, -s1, s2, s5);
            let b4 = bcc(s4, -s5, -s1, s3, -s6, -s2);
            let b5 = bcc(s5, -s3, s2, -s6, -s1, s4);
            let b6 = bcc(s6, -s1, s5, -s2, s4, -s3);
            out0[r] = (
                t0.0 + p1.0 + p2.0 + p3.0 + p4.0 + p5.0 + p6.0,
                t0.1 + p1.1 + p2.1 + p3.1 + p4.1 + p5.1 + p6.1,
            );
            out1[r] = (a1.0 + b1.1, a1.1 - b1.0);
            out12[r] = (a1.0 - b1.1, a1.1 + b1.0);
            out2[r] = (a2.0 + b2.1, a2.1 - b2.0);
            out11[r] = (a2.0 - b2.1, a2.1 + b2.0);
            out3[r] = (a3.0 + b3.1, a3.1 - b3.0);
            out10[r] = (a3.0 - b3.1, a3.1 + b3.0);
            out4[r] = (a4.0 + b4.1, a4.1 - b4.0);
            out9[r] = (a4.0 - b4.1, a4.1 + b4.0);
            out5[r] = (a5.0 + b5.1, a5.1 - b5.0);
            out8[r] = (a5.0 - b5.1, a5.1 + b5.0);
            out6[r] = (a6.0 + b6.1, a6.1 - b6.0);
            out7[r] = (a6.0 - b6.1, a6.1 + b6.0);
        }
    } else if p == 17 {
        let twp = get_or_compute_twiddles(17, inverse);
        mixed_radix_combine_stage_conj_pairs::<17>(out, m, twn, &twp);
    } else {
        unreachable!("iterative odd-tail FFT supports only radix 3/5/7/11/13/17 stages")
    }
}

fn mixed_radix_combine_stage_conj_pairs<const P: usize>(
    out: &mut [Complex64],
    m: usize,
    twn: &[Complex64],
    twp: &[Complex64],
) {
    debug_assert_eq!(P % 2, 1);
    debug_assert!(P >= 3);
    debug_assert_eq!(out.len(), P * m);
    debug_assert!(twp.len() >= P);
    let half = P / 2;
    for r in 0..m {
        let mut t = [(0.0, 0.0); P];
        t[0] = out[r];
        for j in 1..P {
            let (xr, xi) = out[j * m + r];
            let (wr, wi) = twn[j * r];
            t[j] = (xr * wr - xi * wi, xr * wi + xi * wr);
        }

        let mut dc = t[0];
        for j in 1..=half {
            dc.0 += t[j].0 + t[P - j].0;
            dc.1 += t[j].1 + t[P - j].1;
        }
        out[r] = dc;

        for u in 1..=half {
            let mut a = t[0];
            let mut b = (0.0, 0.0);
            for j in 1..=half {
                let pair_sum = (t[j].0 + t[P - j].0, t[j].1 + t[P - j].1);
                let pair_diff = (t[j].0 - t[P - j].0, t[j].1 - t[P - j].1);
                let (c, s_signed) = twp[(j * u) % P];
                let s = -s_signed;
                a.0 += c * pair_sum.0;
                a.1 += c * pair_sum.1;
                b.0 += s * pair_diff.0;
                b.1 += s * pair_diff.1;
            }
            out[u * m + r] = (a.0 + b.1, a.1 - b.0);
            out[(P - u) * m + r] = (a.0 - b.1, a.1 + b.0);
        }
    }
}

fn mixed_radix_small_power_tail(data: &mut [Complex64], inverse: bool) {
    match data.len() {
        0 | 1 => {}
        2 => {
            let a = data[0];
            let b = data[1];
            data[0] = complex_add(a, b);
            data[1] = complex_sub(a, b);
        }
        4 => fft4_kernel(data, inverse),
        8 => fft8_kernel(data, inverse),
        16 => fft16_kernel(data, inverse),
        _ => unreachable!("small power tail only supports powers of two up to 16"),
    }
}

fn fft4_kernel(data: &mut [Complex64], inverse: bool) {
    debug_assert_eq!(data.len(), 4);
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let a02 = complex_add(x0, x2);
    let b02 = complex_sub(x0, x2);
    let a13 = complex_add(x1, x3);
    let b13 = complex_sub(x1, x3);
    let rot = if inverse {
        (-b13.1, b13.0)
    } else {
        (b13.1, -b13.0)
    };
    data[0] = complex_add(a02, a13);
    data[1] = complex_add(b02, rot);
    data[2] = complex_sub(a02, a13);
    data[3] = complex_sub(b02, rot);
}

fn fft8_kernel(data: &mut [Complex64], inverse: bool) {
    debug_assert_eq!(data.len(), 8);
    let mut even = [data[0], data[2], data[4], data[6]];
    let mut odd = [data[1], data[3], data[5], data[7]];
    fft4_kernel(&mut even, inverse);
    fft4_kernel(&mut odd, inverse);
    for k in 0..4 {
        let t = complex_mul(odd[k], small_twiddle_8(k, inverse));
        data[k] = complex_add(even[k], t);
        data[k + 4] = complex_sub(even[k], t);
    }
}

fn fft16_kernel(data: &mut [Complex64], inverse: bool) {
    debug_assert_eq!(data.len(), 16);
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
    ];
    fft8_kernel(&mut even, inverse);
    fft8_kernel(&mut odd, inverse);
    for k in 0..8 {
        let t = complex_mul(odd[k], small_twiddle_16(k, inverse));
        data[k] = complex_add(even[k], t);
        data[k + 8] = complex_sub(even[k], t);
    }
}

fn small_twiddle_8(k: usize, inverse: bool) -> Complex64 {
    debug_assert!(k < 4);
    const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let twiddle = match k {
        0 => (1.0, 0.0),
        1 => (S, -S),
        2 => (0.0, -1.0),
        3 => (-S, -S),
        _ => unreachable!(),
    };
    if inverse {
        (twiddle.0, -twiddle.1)
    } else {
        twiddle
    }
}

fn small_twiddle_16(k: usize, inverse: bool) -> Complex64 {
    debug_assert!(k < 8);
    const C1: f64 = 0.923_879_532_511_286_7;
    const S1: f64 = 0.382_683_432_365_089_8;
    const S: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const C3: f64 = 0.382_683_432_365_089_84;
    const S3: f64 = 0.923_879_532_511_286_7;
    let twiddle = match k {
        0 => (1.0, 0.0),
        1 => (C1, -S1),
        2 => (S, -S),
        3 => (C3, -S3),
        4 => (0.0, -1.0),
        5 => (-C3, -S3),
        6 => (-S, -S),
        7 => (-C1, -S1),
        _ => unreachable!(),
    };
    if inverse {
        (twiddle.0, -twiddle.1)
    } else {
        twiddle
    }
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

thread_local! {
    static DEFAULT_WORKERS: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    // Set by an individual transform invocation so low-level dispatch helpers
    // cannot bypass `FftOptions::workers` by consulting the host directly.
    static ACTIVE_WORKER_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Return the default FFT worker count for the current thread.
///
/// The value starts at one and is temporarily overridden by [`set_workers`].
/// Like SciPy's context-local setting, an override is not inherited by another
/// thread.
#[must_use]
pub fn get_workers() -> usize {
    DEFAULT_WORKERS.get()
}

/// Guard returned by [`set_workers`].
///
/// Dropping the guard restores the prior worker count. The `Rc` marker keeps
/// the guard on the thread whose thread-local setting it owns.
#[derive(Debug)]
pub struct WorkersGuard {
    previous: usize,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for WorkersGuard {
    fn drop(&mut self) {
        DEFAULT_WORKERS.set(self.previous);
    }
}

/// Temporarily set the default FFT worker count for the current thread.
///
/// Positive values are used directly. Negative values wrap from the available
/// CPU count following SciPy's convention: `-1` selects every available CPU
/// and `-N` selects one worker when `N` CPUs are available. Zero and values
/// below `-N` are rejected.
pub fn set_workers(workers: isize) -> Result<WorkersGuard, FftError> {
    let workers = normalize_workers(workers)?;
    let previous = DEFAULT_WORKERS.replace(workers);
    Ok(WorkersGuard {
        previous,
        _not_send: std::marker::PhantomData,
    })
}

fn normalize_workers(workers: isize) -> Result<usize, FftError> {
    if workers > 0 {
        return usize::try_from(workers)
            .map_err(|_| FftError::InvalidWorkers { requested: workers });
    }
    if workers == 0 {
        return Err(FftError::InvalidWorkers { requested: workers });
    }

    let cpu_count = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let distance_from_end = workers.unsigned_abs();
    if distance_from_end > cpu_count {
        return Err(FftError::InvalidWorkers { requested: workers });
    }
    Ok(cpu_count + 1 - distance_from_end)
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

#[derive(Debug)]
struct ActiveWorkerCapGuard {
    previous: usize,
}

impl Drop for ActiveWorkerCapGuard {
    fn drop(&mut self) {
        ACTIVE_WORKER_CAP.set(self.previous);
    }
}

fn worker_policy_cap(policy: WorkerPolicy, available: usize) -> usize {
    match policy {
        WorkerPolicy::Auto => available,
        WorkerPolicy::Exact(workers) | WorkerPolicy::Max(workers) => available.min(workers),
    }
}

fn install_worker_cap(policy: WorkerPolicy) -> ActiveWorkerCapGuard {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let previous = ACTIVE_WORKER_CAP.replace(worker_policy_cap(policy, available));
    ActiveWorkerCapGuard { previous }
}

fn dispatch_worker_limit() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(ACTIVE_WORKER_CAP.get())
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
            workers: WorkerPolicy::Exact(get_workers()),
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
    InvalidWorkers { requested: isize },
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
                write!(f, "sample spacing must be non-zero")
            }
            Self::NonFiniteInput => write!(f, "non-finite input rejected by policy"),
        }
    }
}

impl std::error::Error for FftError {}

/// Fast Walsh–Hadamard transform (natural / Hadamard ordering) in O(n·log n).
///
/// Computes `H_n · x`, where `H_n` is the n×n Hadamard matrix
/// (`H_n[i][j] = (−1)^popcount(i & j)`, the n-fold Kronecker power of `[[1,1],[1,−1]]`) and
/// `n` must be a power of two. The in-place butterfly evaluates it in O(n·log n) versus the
/// O(n²) explicit matrix–vector product one would otherwise form as
/// `scipy.linalg.hadamard(n) @ x`. Unnormalized: `fwht` is its own inverse up to scale —
/// `fwht(fwht(x)) == n·x`, so the inverse transform is `fwht(y)` divided by `n`. `options`
/// only governs the finite-input policy.
pub fn fwht(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(FftError::InvalidShape {
            detail: "fwht length must be a power of two",
        });
    }
    let mut x = input.to_vec();
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    Ok(x)
}

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

/// Worker count for a batched (across-rows) 1-D transform: each row is an O(ncols·log ncols) transform,
/// so fan out one core per row up to `available_parallelism`, gated off for tiny batches where the
/// thread-spawn floor would dominate.
fn batched_axis2d_threads(rows: usize, work: usize, policy: WorkerPolicy) -> usize {
    if rows < 2 || work < (1 << 14) {
        return 1;
    }
    worker_policy_cap(policy, dispatch_worker_limit()).min(rows)
}

/// Batched 1-D forward complex FFT along the last axis of a row-major `rows × ncols` array.
///
/// Equivalent to `scipy.fft.fft(x.reshape(rows, ncols), axis=-1)` — `rows` INDEPENDENT length-`ncols`
/// transforms — but parallel ACROSS rows (each row's 1-D FFT runs serially on its owning thread). This
/// beats both looping the 1-D [`fft`] and SciPy's default `workers=1` (serial over rows). Output is
/// row-major `rows × ncols`; row `r` is bit-identical to `fft(&input[r*ncols..(r+1)*ncols], options)`.
pub fn fft_axis2d(
    input: &[Complex64],
    rows: usize,
    ncols: usize,
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    if rows == 0 || ncols == 0 {
        return Err(FftError::InvalidShape {
            detail: "rows and ncols must be > 0",
        });
    }
    if input.len() != rows.saturating_mul(ncols) {
        return Err(FftError::LengthMismatch {
            expected: rows.saturating_mul(ncols),
            actual: input.len(),
        });
    }
    // Parallelism is ACROSS rows; keep each row's transform serial to avoid 64×64 oversubscription.
    let mut inner = options.clone();
    inner.workers = WorkerPolicy::Exact(1);
    let row_out = ncols;
    let mut out = vec![(0.0, 0.0); rows * row_out];
    let work = rows.saturating_mul(ncols);
    let nthreads = batched_axis2d_threads(rows, work, options.workers);
    if nthreads <= 1 {
        for r in 0..rows {
            let res = fft(&input[r * ncols..(r + 1) * ncols], &inner)?;
            out[r * row_out..(r + 1) * row_out].copy_from_slice(&res);
        }
        return Ok(out);
    }
    let rows_per = rows.div_ceil(nthreads);
    let inner = &inner;
    let first_err = std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows_per * row_out)
            .enumerate()
            .map(|(ti, chunk)| {
                let r0 = ti * rows_per;
                scope.spawn(move || -> Result<(), FftError> {
                    for (rr, slot) in chunk.chunks_mut(row_out).enumerate() {
                        let r = r0 + rr;
                        let res = fft(&input[r * ncols..(r + 1) * ncols], inner)?;
                        slot.copy_from_slice(&res);
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap().err())
            .next()
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Batched 1-D real-input FFT along the last axis of a row-major `rows × ncols` real array.
///
/// Equivalent to `scipy.fft.rfft(x.reshape(rows, ncols), axis=-1)` — `rows` INDEPENDENT length-`ncols`
/// real transforms — but parallel ACROSS rows. Output is row-major `rows × (ncols/2 + 1)` complex; row
/// `r` is bit-identical to `rfft(&input[r*ncols..(r+1)*ncols], options)`.
pub fn rfft_axis2d(
    input: &[f64],
    rows: usize,
    ncols: usize,
    options: &FftOptions,
) -> Result<Vec<Complex64>, FftError> {
    if rows == 0 || ncols == 0 {
        return Err(FftError::InvalidShape {
            detail: "rows and ncols must be > 0",
        });
    }
    if input.len() != rows.saturating_mul(ncols) {
        return Err(FftError::LengthMismatch {
            expected: rows.saturating_mul(ncols),
            actual: input.len(),
        });
    }
    let mut inner = options.clone();
    inner.workers = WorkerPolicy::Exact(1);
    let row_out = ncols / 2 + 1;
    let mut out = vec![(0.0, 0.0); rows * row_out];
    let work = rows.saturating_mul(ncols);
    let nthreads = batched_axis2d_threads(rows, work, options.workers);
    if nthreads <= 1 {
        for r in 0..rows {
            let res = rfft(&input[r * ncols..(r + 1) * ncols], &inner)?;
            out[r * row_out..(r + 1) * row_out].copy_from_slice(&res);
        }
        return Ok(out);
    }
    let rows_per = rows.div_ceil(nthreads);
    let inner = &inner;
    let first_err = std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows_per * row_out)
            .enumerate()
            .map(|(ti, chunk)| {
                let r0 = ti * rows_per;
                scope.spawn(move || -> Result<(), FftError> {
                    for (rr, slot) in chunk.chunks_mut(row_out).enumerate() {
                        let r = r0 + rr;
                        let res = rfft(&input[r * ncols..(r + 1) * ncols], inner)?;
                        slot.copy_from_slice(&res);
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap().err())
            .next()
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Run a real→real (length-preserving) per-row transform over every length-`ncols` row of a row-major
/// `rows × ncols` array, parallel ACROSS rows. Output row `r` is exactly `transform(row r)`. Shared by the
/// batched DCT/DST entrypoints.
fn batched_real_axis2d<F>(
    input: &[f64],
    rows: usize,
    ncols: usize,
    policy: WorkerPolicy,
    transform: F,
) -> Result<Vec<f64>, FftError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FftError> + Sync,
{
    if rows == 0 || ncols == 0 {
        return Err(FftError::InvalidShape {
            detail: "rows and ncols must be > 0",
        });
    }
    if input.len() != rows.saturating_mul(ncols) {
        return Err(FftError::LengthMismatch {
            expected: rows.saturating_mul(ncols),
            actual: input.len(),
        });
    }
    let row_out = ncols;
    let mut out = vec![0.0f64; rows * row_out];
    let work = rows.saturating_mul(ncols);
    let nthreads = batched_axis2d_threads(rows, work, policy);
    if nthreads <= 1 {
        for r in 0..rows {
            let res = transform(&input[r * ncols..(r + 1) * ncols])?;
            out[r * row_out..(r + 1) * row_out].copy_from_slice(&res);
        }
        return Ok(out);
    }
    let rows_per = rows.div_ceil(nthreads);
    let transform = &transform;
    let first_err = std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows_per * row_out)
            .enumerate()
            .map(|(ti, chunk)| {
                let r0 = ti * rows_per;
                scope.spawn(move || -> Result<(), FftError> {
                    for (rr, slot) in chunk.chunks_mut(row_out).enumerate() {
                        let r = r0 + rr;
                        let res = transform(&input[r * ncols..(r + 1) * ncols])?;
                        slot.copy_from_slice(&res);
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap().err())
            .next()
    });
    match first_err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Batched DCT-II along the last axis of a row-major `rows × ncols` real array.
///
/// Equivalent to `scipy.fft.dct(x.reshape(rows, ncols), type=2, axis=-1)` — `rows` INDEPENDENT
/// length-`ncols` DCTs — but parallel ACROSS rows (each row's DCT serial on its owning thread). Per-row /
/// per-block DCT is the core of nearly every image/audio compression pipeline. Row `r` is bit-identical to
/// `dct(&input[r*ncols..(r+1)*ncols], options)`.
pub fn dct_axis2d(
    input: &[f64],
    rows: usize,
    ncols: usize,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let mut inner = options.clone();
    inner.workers = WorkerPolicy::Exact(1);
    batched_real_axis2d(input, rows, ncols, options.workers, |row| dct(row, &inner))
}

/// Batched inverse DCT (DCT-III) along the last axis of a row-major `rows × ncols` real array.
///
/// Equivalent to `scipy.fft.idct(x.reshape(rows, ncols), type=2, axis=-1)`, parallel ACROSS rows. Row `r`
/// is bit-identical to `idct(&input[r*ncols..(r+1)*ncols], options)`.
pub fn idct_axis2d(
    input: &[f64],
    rows: usize,
    ncols: usize,
    options: &FftOptions,
) -> Result<Vec<f64>, FftError> {
    let mut inner = options.clone();
    inner.workers = WorkerPolicy::Exact(1);
    batched_real_axis2d(input, rows, ncols, options.workers, |row| idct(row, &inner))
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
    let _worker_cap = install_worker_cap(options.workers);

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
    let fingerprint = audit_ledger.map_or_else(Vec::new, |_| complex_fingerprint(input));
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;
    let _worker_cap = install_worker_cap(options.workers);

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
/// Cached DCT-II extract twiddles exp(-iπk/(2N)), k=0..N-1, keyed by N. The factor is identical
/// across calls of the same length, so computing cos/sin per coefficient on every call (the bulk
/// of DCT time) is wasteful — scipy caches the plan likewise. Bit-identical to the inline form.
static DCT2_TWIDDLE_CACHE: OnceLock<RwLock<HashMap<usize, TwiddleTable>>> = OnceLock::new();
fn get_dct2_cache() -> &'static RwLock<HashMap<usize, TwiddleTable>> {
    DCT2_TWIDDLE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}
thread_local! {
    static LOCAL_DCT2_CACHE: std::cell::RefCell<HashMap<usize, TwiddleTable>> =
        std::cell::RefCell::new(HashMap::new());
}
static DCT4_TWIDDLE_CACHE: OnceLock<RwLock<HashMap<usize, TwiddleTable>>> = OnceLock::new();
fn get_dct4_cache() -> &'static RwLock<HashMap<usize, TwiddleTable>> {
    DCT4_TWIDDLE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}
thread_local! {
    static LOCAL_DCT4_CACHE: std::cell::RefCell<HashMap<usize, TwiddleTable>> =
        std::cell::RefCell::new(HashMap::new());
}
/// Cached DCT-IV extract twiddles exp(-iπ(2k+1)/(4N)), k=0..N-1, keyed by N. Bit-identical to
/// the inline cos/sin form; avoids recomputing trig per coefficient on every dct_iv call.
fn get_or_compute_dct4_twiddles(n: usize) -> TwiddleTable {
    if let Some(t) = LOCAL_DCT4_CACHE.with(|c| c.borrow().get(&n).cloned()) {
        return t;
    }
    let cache = get_dct4_cache();
    let table = if let Some(t) = cache.read().ok().and_then(|g| g.get(&n).cloned()) {
        t
    } else {
        let mut table = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -PI * (2 * k + 1) as f64 / (4.0 * n as f64);
            table.push((angle.cos(), angle.sin()));
        }
        let table: TwiddleTable = Arc::from(table);
        if let Ok(mut g) = cache.write() {
            g.insert(n, Arc::clone(&table));
        }
        table
    };
    LOCAL_DCT4_CACHE.with(|c| c.borrow_mut().insert(n, Arc::clone(&table)));
    table
}

static DCT4_SPLIT_TWIDDLE_CACHE: OnceLock<RwLock<HashMap<usize, TwiddleTable>>> = OnceLock::new();
fn get_dct4_split_cache() -> &'static RwLock<HashMap<usize, TwiddleTable>> {
    DCT4_SPLIT_TWIDDLE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}
thread_local! {
    static LOCAL_DCT4_SPLIT_CACHE: std::cell::RefCell<HashMap<usize, TwiddleTable>> =
        std::cell::RefCell::new(HashMap::new());
}
/// Cached split twiddle exp(-iπk/N), k=0..N-1, keyed by N. Forms the odd-output
/// sub-sequence u'[k] = u[k]·exp(-iπk/N) in the Type-IV core (see dct4_core_fft).
fn get_or_compute_dct4_split_twiddles(n: usize) -> TwiddleTable {
    if let Some(t) = LOCAL_DCT4_SPLIT_CACHE.with(|c| c.borrow().get(&n).cloned()) {
        return t;
    }
    let cache = get_dct4_split_cache();
    let table = if let Some(t) = cache.read().ok().and_then(|g| g.get(&n).cloned()) {
        t
    } else {
        let mut table = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -PI * k as f64 / n as f64;
            table.push((angle.cos(), angle.sin()));
        }
        let table: TwiddleTable = Arc::from(table);
        if let Ok(mut g) = cache.write() {
            g.insert(n, Arc::clone(&table));
        }
        table
    };
    LOCAL_DCT4_SPLIT_CACHE.with(|c| c.borrow_mut().insert(n, Arc::clone(&table)));
    table
}

fn get_or_compute_dct2_twiddles(n: usize) -> TwiddleTable {
    if let Some(t) = LOCAL_DCT2_CACHE.with(|c| c.borrow().get(&n).cloned()) {
        return t;
    }
    let cache = get_dct2_cache();
    let table = if let Some(t) = cache.read().ok().and_then(|g| g.get(&n).cloned()) {
        t
    } else {
        let mut table = Vec::with_capacity(n);
        for k in 0..n {
            let angle = -PI * k as f64 / (2.0 * n as f64);
            table.push((angle.cos(), angle.sin()));
        }
        let table: TwiddleTable = Arc::from(table);
        if let Ok(mut g) = cache.write() {
            g.insert(n, Arc::clone(&table));
        }
        table
    };
    LOCAL_DCT2_CACHE.with(|c| c.borrow_mut().insert(n, Arc::clone(&table)));
    table
}

pub fn dct(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    ensure_non_empty(input.len())?;
    validate_finite_real(input, options)?;
    let _worker_cap = install_worker_cap(options.workers);

    let n = input.len();

    // DCT-II via a single N-point FFT (Makhoul): the even-indexed samples go
    // forward and the odd-indexed samples come back in reverse, so the FFT of
    // that permutation is the DCT after a twiddle. This is ~2x cheaper than the
    // naive 2N-point complex FFT of the mirror-symmetric extension (which both
    // doubles the transform length and feeds it redundant data).
    //   v[j] = x[2j]          (2j < N)
    //   v[j] = x[2(N-j)-1]    (2j ≥ N)
    //   X[k] = 2·Re(exp(-iπk/2N) · V[k]),  V = FFT_N(v)
    let mut v = vec![0.0f64; n];
    for (j, slot) in v.iter_mut().enumerate() {
        let src = if 2 * j < n { 2 * j } else { 2 * (n - j) - 1 };
        *slot = input[src];
    }

    let backend = resolve_backend(options.backend);
    // `v` is real, so for even N use the real-input FFT (an N/2-point complex
    // transform) and recover the upper half by Hermitian symmetry
    // V[N-k] = conj(V[k]); odd N falls back to a full N-point complex FFT.
    // Extract DCT coefficients: X[k] = 2·Re(V[k] · exp(-iπk/(2N))).
    let mut result = Vec::with_capacity(n);
    let dct_tw = get_or_compute_dct2_twiddles(n);
    // Only the REAL part of V[k]·e^{-iπk/2N} is needed: 2·(vₖ.re·tw.re − vₖ.im·tw.im).
    // Computing it directly skips the wasted imaginary half of a full complex_mul.
    let extract = |k: usize, vk: Complex64| {
        let tw = dct_tw[k];
        2.0 * (vk.0 * tw.0 - vk.1 * tw.1)
    };
    if n.is_multiple_of(2) {
        let half = real_fft_specialized(&v, backend); // V[0..=N/2]
        // Split the k ≤ N/2 (Vₖ = half[k]) and k > N/2 (Vₖ = conj(half[N−k]))
        // ranges into two branch-free loops. The conjugate's real part folds the
        // sign into a `+`: 2·(half[N−k].re·tw.re + half[N−k].im·tw.im). Bit-identical.
        for (k, &vk) in half.iter().enumerate().take(n / 2 + 1) {
            result.push(extract(k, vk));
        }
        for k in (n / 2 + 1)..n {
            let (vr, vi) = half[n - k];
            let tw = dct_tw[k];
            result.push(2.0 * (vr * tw.0 + vi * tw.1));
        }
    } else {
        let complex_v: Vec<Complex64> = v.iter().map(|&x| (x, 0.0)).collect();
        let spectrum = backend.transform_1d_unscaled(&complex_v, false);
        for (k, &vk) in spectrum.iter().enumerate().take(n) {
            result.push(extract(k, vk));
        }
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
    // Compute via the inverse Makhoul reorder: an N/2-point real inverse FFT
    // instead of the naive 2N-point complex inverse FFT.
    let backend = resolve_backend(options.backend);

    if n.is_multiple_of(2) {
        // Recover the length-(N/2+1) half-spectrum V (the FFT of the reordered
        // real sequence v) from the DCT coefficients X = scaled_input:
        //   e^{-iπk/2N}·V[k] = (X[k] - i·X[N-k])/2  ⇒  V[k] = e^{+iπk/2N}(X[k]-iX[N-k])/2,
        //   V[0] = X[0]/2.
        let m = n / 2;
        // Twiddle e^{+iπk/2N} = conj(dct2_tw[k]); reuse the cached DCT-II table
        // instead of recomputing N/2 cos/sin on every call. Bit-identical (cos is
        // even, sin odd → conj of the stored (cos(-θ),sin(-θ)) == (cos θ,sin θ),
        // verified to_bits across 5.6e5 k/N). Lifts idct and its dct_iii/dst_iii
        // callers (was ~6-8ms of stray transcendentals at N=2^20).
        let idct_tw = get_or_compute_dct2_twiddles(n);
        let mut half = Vec::with_capacity(m + 1);
        half.push((0.5 * scaled_input[0], 0.0));
        for k in 1..=m {
            let xk = scaled_input[k];
            let xnk = scaled_input[n - k];
            let twiddle = complex_conj(idct_tw[k]);
            half.push(complex_mul((0.5 * xk, -0.5 * xnk), twiddle));
        }
        // v = N·v_true (unscaled real inverse FFT); un-interleave the forward
        // reorder and divide by N to recover x.
        let v = real_ifft_unscaled(&half, n, backend);
        let inv_n = 1.0 / nf;
        let mut result = vec![0.0; n];
        for (j, &vj) in v.iter().enumerate() {
            let dst = if 2 * j < n { 2 * j } else { 2 * (n - j) - 1 };
            result[dst] = vj * inv_n;
        }
        return Ok(result);
    }

    // Odd N: fall back to the 2N-point complex inverse FFT of the Hermitian
    // spectrum (the real-FFT pack needs an even length).
    let mut spectrum = vec![(0.0, 0.0); 2 * n];
    for k in 0..n {
        let angle = PI * k as f64 / (2.0 * nf);
        let twiddle = (angle.cos(), angle.sin());
        spectrum[k] = complex_mul((scaled_input[k], 0.0), twiddle);
    }
    for k in 1..n {
        spectrum[2 * n - k] = complex_conj(spectrum[k]);
    }
    let time_domain = backend.transform_1d_unscaled(&spectrum, true);
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

    // DCT-I is the FFT of the real symmetric period-(2N-2) extension; since that
    // extension is real, use the real-FFT pack (an (N-1)-point complex FFT)
    // rather than a full (2N-2)-point complex FFT — half the FFT points.
    let compute_backward = |src: &[f64]| -> Vec<f64> {
        let m = 2 * n - 2;
        let mut extended = vec![0.0f64; m];
        extended[..n].copy_from_slice(&src[..n]);
        for i in 1..n - 1 {
            extended[m - i] = src[i];
        }
        let backend = resolve_backend(options.backend);
        let half = real_fft_specialized(&extended, backend); // bins 0..=(N-1)
        half.into_iter().take(n).map(|v| v.0).collect()
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

/// At/above this N the two N-point sub-FFTs of the Type-IV core run on separate
/// threads (each is heavy enough — ~N log N — to dwarf the spawn cost; scipy's
/// pocketfft is single-threaded here so the second core is free domination).
const DCT4_PARALLEL_GATE: usize = 1 << 16;

/// Shared core for the Type-IV cosine/sine transforms. Returns the first N bins
/// `U[k] = Σ_{n=0}^{N-1} u[n]·exp(-iπnk/N)`, `u[n] = x[n]·exp(-iπn/2N)`.
///
/// The old route ran ONE 2N-point complex FFT of the zero-padded `u`, which both
/// doubles the transform length and thrashes the cache. Instead split that 2N
/// transform by OUTPUT parity into two independent N-point FFTs (exact
/// Cooley-Tukey decimation in frequency):
///   U[2m]   = FFT_N(u)[m]
///   U[2m+1] = FFT_N(u')[m],   u'[n] = u[n]·exp(-iπn/N)
/// Two N-point FFTs are far cheaper than one 2N-point FFT at large N (less work
/// AND cache-resident) and the pair runs concurrently above DCT4_PARALLEL_GATE.
/// DCT-IV reads `2·Re(e^{-iπ(2k+1)/4N}·U[k])` and DST-IV `-2·Im(...)`.
fn dct4_core_fft(input: &[f64], options: &FftOptions) -> Vec<Complex64> {
    let n = input.len();
    let pre_tw = get_or_compute_dct2_twiddles(n); // exp(-iπn/2N)
    let split_tw = get_or_compute_dct4_split_twiddles(n); // exp(-iπn/N)
    // u[n] = x[n]·exp(-iπn/2N);  u'[n] = u[n]·exp(-iπn/N)
    let mut u = vec![(0.0, 0.0); n];
    let mut up = vec![(0.0, 0.0); n];
    for i in 0..n {
        let un = complex_mul((input[i], 0.0), pre_tw[i]);
        u[i] = un;
        up[i] = complex_mul(un, split_tw[i]);
    }

    let kind = options.backend;
    // A = FFT_N(u), B = FFT_N(u'); concurrent for large N (scipy is serial here).
    let (a, b) = if n >= DCT4_PARALLEL_GATE {
        let u_ref = &u;
        std::thread::scope(|s| {
            let h = s.spawn(move || resolve_backend(kind).transform_1d_unscaled(u_ref, false));
            let b = resolve_backend(kind).transform_1d_unscaled(&up, false);
            (h.join().expect("dct4 sub-FFT thread panicked"), b)
        })
    } else {
        let backend = resolve_backend(kind);
        (
            backend.transform_1d_unscaled(&u, false),
            backend.transform_1d_unscaled(&up, false),
        )
    };

    // Interleave the parity halves: U[2m] = A[m], U[2m+1] = B[m].
    let mut out = vec![(0.0, 0.0); n];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { a[k / 2] } else { b[k / 2] };
    }
    out
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
    // DCT-IV via a 2N-point complex FFT (pre/post half-sample rotation) instead
    // of the old 8N-point complex FFT. Factoring the kernel
    //   (2n+1)(2k+1)/4N = nk/N + n/2N + k/2N + 1/4N  gives
    //   X[k] = 2·Re{ e^{-iπ(2k+1)/4N} · U[k] },  U = FFT_{2N}(u),
    //   u[n] = x[n]·e^{-iπn/2N} (n<N), zero-padded to 2N.
    let spectrum = dct4_core_fft(input, options);
    let mut result = Vec::with_capacity(n);
    let post_tw = get_or_compute_dct4_twiddles(n);
    for (k, &uk) in spectrum.iter().enumerate().take(n) {
        result.push(2.0 * complex_mul(post_tw[k], uk).0); // 2·Re
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
    // DST-I is the FFT of the real antisymmetric period-(2N+2) extension; that
    // extension is real, so use the real-FFT pack (an (N+1)-point complex FFT)
    // instead of a full (2N+2)-point complex FFT — half the FFT points.
    let m = 2 * n + 2;
    let mut extended = vec![0.0f64; m];
    for i in 0..n {
        extended[i + 1] = input[i];
        extended[m - i - 1] = -input[i];
    }

    let backend = resolve_backend(options.backend);
    let half = real_fft_specialized(&extended, backend); // bins 0..=(N+1)

    let mut result = Vec::with_capacity(n);
    for val in half.iter().take(n + 1).skip(1) {
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
    // DST-II is a sign-modulated, index-reversed DCT-II:
    //   cos(π(N-j)(2n+1)/2N) = (-1)^n · sin(πj(2n+1)/2N)   (j = k+1)
    // ⇒ DST-II(x)[k] = DCT-II(y)[N-1-k] with y[n] = (-1)^n·x[n].
    // Route through the fast N/2-point real-FFT `dct` (Backward = the raw 2·sum
    // form) instead of the old 4N-point complex FFT, then reverse.
    let y: Vec<f64> = input
        .iter()
        .enumerate()
        .map(|(idx, &x)| if idx % 2 == 0 { x } else { -x })
        .collect();
    let mut backward = options.clone();
    backward.normalization = Normalization::Backward;
    let mut result = dct(&y, &backward)?;
    result.reverse();
    let nf = n as f64;
    // br-yjas slice 2 (DST-II): scipy normalization. Ortho scales all
    // entries by 1/sqrt(2N) plus an extra 1/sqrt(2) on the LAST entry
    // (scipy DST-II convention; mirror of DCT-II's first-entry adj).
    // Forward applies a uniform 1/(2N).
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
    //
    // DST-III inverts DST-II, so it is the transpose of the forward identity
    // (DST-II(x)[k] = DCT-II((-1)^n·x)[N-1-k]):
    //   DST-III(X)[n] = 2N·(-1)^n·idct(reverse(X))[n]   (Backward = 2N·x form).
    // Routes through the fast N/2-point real-FFT `idct` instead of the old
    // 4N-point complex inverse FFT.
    let compute_backward = |src: &[f64]| -> Result<Vec<f64>, FftError> {
        let mut rev = src.to_vec();
        rev.reverse();
        let mut bo = options.clone();
        bo.normalization = Normalization::Backward;
        let y = idct(&rev, &bo)?;
        let two_n = 2.0 * nf;
        Ok(y.iter()
            .enumerate()
            .map(|(idx, &val)| if idx % 2 == 0 { two_n } else { -two_n } * val)
            .collect())
    };

    match options.normalization {
        Normalization::Backward => compute_backward(input),
        Normalization::Forward => {
            let mut result = compute_backward(input)?;
            let s = 1.0 / (2.0 * nf);
            for v in result.iter_mut() {
                *v *= s;
            }
            Ok(result)
        }
        Normalization::Ortho => {
            let mut adjusted = input.to_vec();
            adjusted[n - 1] *= 2.0_f64.sqrt();
            let mut result = compute_backward(&adjusted)?;
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
    // DST-IV via the same 2N-point complex FFT as DCT-IV (pre/post rotation);
    // it reads -2·Im instead of 2·Re of e^{-iπ(2k+1)/4N}·U[k]. 8N → 2N.
    let spectrum = dct4_core_fft(input, options);
    let post_tw = get_or_compute_dct4_twiddles(n); // exp(-iπ(2k+1)/4N), cached (shared with dct_iv)
    let mut result = Vec::with_capacity(n);
    for (k, &uk) in spectrum.iter().enumerate().take(n) {
        result.push(-2.0 * complex_mul(post_tw[k], uk).1); // -2·Im
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

/// Discrete Sine Transform dispatcher, matching `scipy.fft.dst(x, type, norm)`.
///
/// Routes to the type-1..4 DST (`dst_i`/`dst_ii`/`dst_iii`/`dst_iv`); scipy's
/// default is `type=2`. `dst_type` must be one of 1, 2, 3, 4.
pub fn dst(input: &[f64], dst_type: u8, options: &FftOptions) -> Result<Vec<f64>, FftError> {
    match dst_type {
        1 => dst_i(input, options),
        2 => dst_ii(input, options),
        3 => dst_iii(input, options),
        4 => dst_iv(input, options),
        _ => Err(FftError::InvalidShape {
            detail: "dst type must be 1, 2, 3, or 4",
        }),
    }
}

/// Inverse Discrete Sine Transform, matching `scipy.fft.idst(x, type, norm)`.
///
/// The inverse of a type-`t` DST is the forward DST of the inverse type
/// (`{1:1, 2:3, 3:2, 4:4}`) with reciprocal normalization: `Ortho` stays
/// `Ortho`; `Forward` maps to a `Backward` forward-DST; the default `Backward`
/// additionally divides by `2N` (`2(N+1)` for type 1). `dst_type` must be one
/// of 1, 2, 3, 4.
pub fn idst(input: &[f64], dst_type: u8, options: &FftOptions) -> Result<Vec<f64>, FftError> {
    let inv_type: u8 = match dst_type {
        1 => 1,
        2 => 3,
        3 => 2,
        4 => 4,
        _ => {
            return Err(FftError::InvalidShape {
                detail: "idst type must be 1, 2, 3, or 4",
            });
        }
    };
    let n = input.len();
    match options.normalization {
        Normalization::Ortho => dst(input, inv_type, options),
        Normalization::Forward => {
            let opts = options.clone().with_normalization(Normalization::Backward);
            dst(input, inv_type, &opts)
        }
        Normalization::Backward => {
            let opts = options.clone().with_normalization(Normalization::Backward);
            let mut out = dst(input, inv_type, &opts)?;
            let scale = if dst_type == 1 {
                2.0 * (n as f64 + 1.0)
            } else {
                2.0 * n as f64
            };
            for v in out.iter_mut() {
                *v /= scale;
            }
            Ok(out)
        }
    }
}

// Forward DCT of a given type (1..4): dct_i / dct (II) / dct_iii / dct_iv.
fn dct_by_type(input: &[f64], dct_type: u8, options: &FftOptions) -> Result<Vec<f64>, FftError> {
    match dct_type {
        1 => dct_i(input, options),
        2 => dct(input, options),
        3 => dct_iii(input, options),
        4 => dct_iv(input, options),
        _ => Err(FftError::InvalidShape {
            detail: "dct type must be 1, 2, 3, or 4",
        }),
    }
}

// Inverse DCT of a given type, mirroring the idst routing: the inverse of a
// type-t DCT is the forward DCT of the inverse type ({1:1,2:3,3:2,4:4}) with
// reciprocal normalization (Ortho→Ortho, Forward→Backward, Backward→Backward
// then ÷ 2N, or ÷ 2(N-1) for type 1).
fn idct_dispatch(input: &[f64], dct_type: u8, options: &FftOptions) -> Result<Vec<f64>, FftError> {
    let inv_type: u8 = match dct_type {
        1 => 1,
        2 => 3,
        3 => 2,
        4 => 4,
        _ => {
            return Err(FftError::InvalidShape {
                detail: "idct type must be 1, 2, 3, or 4",
            });
        }
    };
    let n = input.len();
    match options.normalization {
        Normalization::Ortho => dct_by_type(input, inv_type, options),
        Normalization::Forward => {
            let opts = options.clone().with_normalization(Normalization::Backward);
            dct_by_type(input, inv_type, &opts)
        }
        Normalization::Backward => {
            let opts = options.clone().with_normalization(Normalization::Backward);
            let mut out = dct_by_type(input, inv_type, &opts)?;
            let scale = if dct_type == 1 {
                2.0 * (n as f64 - 1.0)
            } else {
                2.0 * n as f64
            };
            for v in out.iter_mut() {
                *v /= scale;
            }
            Ok(out)
        }
    }
}

/// Inverse DCT-I (the inverse of [`dct_i`]). Matches `scipy.fft.idct(x, type=1)`.
///
/// DCT-I is its own inverse up to scaling; the backward norm divides by
/// `2(N-1)`.
pub fn idct_i(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    idct_dispatch(input, 1, options)
}

/// Inverse DCT-III (the inverse of [`dct_iii`]). Matches `scipy.fft.idct(x, type=3)`.
///
/// The inverse of a type-III DCT is a type-II DCT; the backward norm divides by
/// `2N`.
pub fn idct_iii(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    idct_dispatch(input, 3, options)
}

/// Inverse DCT-IV (the inverse of [`dct_iv`]). Matches `scipy.fft.idct(x, type=4)`.
///
/// DCT-IV is its own inverse up to scaling; the backward norm divides by `2N`.
pub fn idct_iv(input: &[f64], options: &FftOptions) -> Result<Vec<f64>, FftError> {
    idct_dispatch(input, 4, options)
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
    let total = validate_real_nd_transform_len(input.len(), shape)?;
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;
    let _worker_cap = install_worker_cap(options.workers);

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
    let total = validate_real_nd_transform_len(input.len(), shape)?;
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;
    let _worker_cap = install_worker_cap(options.workers);

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
    let total = validate_real_nd_transform_len(input.len(), shape)?;
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;
    let _worker_cap = install_worker_cap(options.workers);

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
    let total = validate_real_nd_transform_len(input.len(), shape)?;
    if total == 0 {
        return Ok(Vec::new());
    }

    validate_finite_real(input, options)?;
    let _worker_cap = install_worker_cap(options.workers);

    // Apply IDST along each axis in sequence
    let mut data = input.to_vec();
    let ndim = shape.len();

    for axis in 0..ndim {
        data = apply_dst_along_axis(&data, shape, axis, options, true)?;
    }

    Ok(data)
}

fn validate_real_nd_transform_len(input_len: usize, shape: &[usize]) -> Result<usize, FftError> {
    if shape.is_empty() {
        return Err(FftError::InvalidShape {
            detail: "shape cannot be empty",
        });
    }
    let total = checked_product(shape).ok_or(FftError::InvalidShape {
        detail: "nd shape product overflow",
    })?;
    if total != input_len {
        return Err(FftError::LengthMismatch {
            expected: total,
            actual: input_len,
        });
    }
    Ok(total)
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
    let axis_stride = strides[axis];

    // The fiber for `outer_idx` occupies flat indices `base + i*axis_stride` for `i` in
    // 0..axis_len, where `base` is the flat offset of its non-axis coordinates. The
    // multi-index decomposition is a bijection, so distinct fibers touch disjoint flat
    // indices — every output index is written exactly once. `base` is recovered directly
    // (no per-element O(ndim) index sum), matching the original `Σ multi_idx[d]*strides[d]`
    // bit-for-bit since `multi_idx[axis]=i` contributes exactly `i*axis_stride`.
    let base_of = |outer_idx: usize| -> usize {
        let mut remaining = outer_idx;
        let mut base = 0usize;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            base += (remaining % dim_size) * strides[d];
            remaining /= dim_size;
        }
        base
    };

    // Each fiber is extracted from the immutable `data` (gathered via `base + i*axis_stride`),
    // transformed by a pure 1-D DCT/IDCT, and returned by move — independent work, so the
    // transformed fibers are computed in parallel (threads own disjoint `outer_idx` slots) and
    // scattered serially afterwards. Byte-identical to the sequential fiber loop: identical
    // per-fiber input, identical pure-DCT output, each disjoint output index written once.
    let fiber = |outer_idx: usize| -> Result<Vec<f64>, FftError> {
        let base = base_of(outer_idx);
        let mut temp = vec![0.0; axis_len];
        for (i, value) in temp.iter_mut().enumerate() {
            *value = data[base + i * axis_stride];
        }
        if inverse {
            idct(&temp, options)
        } else {
            dct(&temp, options)
        }
    };

    // Threads are re-spawned per axis pass, so cap by work-PER-THREAD, not just
    // `min(cores, outer_size)`: the DCT/DST fibers are cheap, so one thread per
    // core oversubscribes on small/medium grids — 64 threads for 256 fibers of
    // ~1µs work is spawn-overhead-bound (256² dctn was 5.1 ms ≈ 14× SciPy).
    // Require ~2¹⁶ element-transforms per thread; small grids stay serial and
    // skip the `available_parallelism` syscall entirely. Byte-identical: only the
    // thread COUNT changes (fibers are independent, each output index written once).
    let work = (outer_size as u64).saturating_mul(axis_len as u64);
    let max_by_work = (work / (1 << 16)) as usize;
    let nthreads = if max_by_work < 2 || outer_size < 2 {
        1
    } else {
        dispatch_worker_limit().min(outer_size).min(max_by_work)
    };

    let fibers: Vec<Result<Vec<f64>, FftError>> = if nthreads <= 1 {
        (0..outer_size).map(fiber).collect()
    } else {
        let mut out: Vec<Result<Vec<f64>, FftError>> =
            (0..outer_size).map(|_| Ok(Vec::new())).collect();
        let chunk = outer_size.div_ceil(nthreads);
        let fiber = &fiber;
        std::thread::scope(|scope| {
            for (t, slot) in out.chunks_mut(chunk).enumerate() {
                let base_outer = t * chunk;
                scope.spawn(move || {
                    for (off, o) in slot.iter_mut().enumerate() {
                        *o = fiber(base_outer + off);
                    }
                });
            }
        });
        out
    };

    // Serial scatter of the transformed fibers into `result` (disjoint flat indices).
    for (outer_idx, fib) in fibers.into_iter().enumerate() {
        let transformed = fib?;
        let base = base_of(outer_idx);
        for (i, &value) in transformed.iter().enumerate() {
            result[base + i * axis_stride] = value;
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
    let axis_stride = strides[axis];

    // Flat offset of fiber `outer_idx` (axis coord 0); fiber elements are at
    // `base + i*axis_stride`. Distinct fibers touch disjoint flat indices (the multi-index
    // decomposition is a bijection), matching the original `Σ multi_idx[d]*strides[d]`
    // bit-for-bit (`multi_idx[axis]=i` contributes exactly `i*axis_stride`).
    let base_of = |outer_idx: usize| -> usize {
        let mut remaining = outer_idx;
        let mut base = 0usize;
        for d in (0..ndim).rev() {
            if d == axis {
                continue;
            }
            let dim_size = shape[d];
            base += (remaining % dim_size) * strides[d];
            remaining /= dim_size;
        }
        base
    };

    // Each fiber is extracted from the immutable `data`, transformed by a pure 1-D DST-II
    // (forward) / DST-III + idstn rescale (inverse), and returned by move — independent work,
    // computed in parallel (threads own disjoint `outer_idx` slots) then scattered serially.
    // Byte-identical to the sequential fiber loop (same per-fiber input, same pure transform +
    // per-fiber scale, each disjoint output index written exactly once).
    let fiber = |outer_idx: usize| -> Result<Vec<f64>, FftError> {
        let base = base_of(outer_idx);
        let mut temp = vec![0.0; axis_len];
        for (i, value) in temp.iter_mut().enumerate() {
            *value = data[base + i * axis_stride];
        }
        let mut transformed = if inverse {
            dst_iii(&temp, options)?
        } else {
            dst_ii(&temp, options)?
        };
        // Convert dst_iii into idst (the inverse of dst type=2 that scipy.fft.idstn computes),
        // so idstn(dstn(x)) = x for every normalization (per-axis scale; see history below):
        //   backward: dst_iii is unnormalized DST-III → apply 1/(2N).
        //   forward:  dst_iii already /2N → multiply back by 2N (the forward dstn absorbed it).
        //   ortho:    dst_iii already applied 1/sqrt(2N), which idst also needs.
        if inverse {
            let scale = match options.normalization {
                Normalization::Backward => 1.0 / (2.0 * axis_len as f64),
                Normalization::Forward => 2.0 * axis_len as f64,
                Normalization::Ortho => 1.0,
            };
            if scale != 1.0 {
                for v in transformed.iter_mut() {
                    *v *= scale;
                }
            }
        }
        Ok(transformed)
    };

    // Threads are re-spawned per axis pass, so cap by work-PER-THREAD, not just
    // `min(cores, outer_size)`: the DCT/DST fibers are cheap, so one thread per
    // core oversubscribes on small/medium grids — 64 threads for 256 fibers of
    // ~1µs work is spawn-overhead-bound (256² dctn was 5.1 ms ≈ 14× SciPy).
    // Require ~2¹⁶ element-transforms per thread; small grids stay serial and
    // skip the `available_parallelism` syscall entirely. Byte-identical: only the
    // thread COUNT changes (fibers are independent, each output index written once).
    let work = (outer_size as u64).saturating_mul(axis_len as u64);
    let max_by_work = (work / (1 << 16)) as usize;
    let nthreads = if max_by_work < 2 || outer_size < 2 {
        1
    } else {
        dispatch_worker_limit().min(outer_size).min(max_by_work)
    };

    let fibers: Vec<Result<Vec<f64>, FftError>> = if nthreads <= 1 {
        (0..outer_size).map(fiber).collect()
    } else {
        let mut out: Vec<Result<Vec<f64>, FftError>> =
            (0..outer_size).map(|_| Ok(Vec::new())).collect();
        let chunk = outer_size.div_ceil(nthreads);
        let fiber = &fiber;
        std::thread::scope(|scope| {
            for (t, slot) in out.chunks_mut(chunk).enumerate() {
                let base_outer = t * chunk;
                scope.spawn(move || {
                    for (off, o) in slot.iter_mut().enumerate() {
                        *o = fiber(base_outer + off);
                    }
                });
            }
        });
        out
    };

    // Serial scatter of the transformed fibers into `result` (disjoint flat indices).
    for (outer_idx, fib) in fibers.into_iter().enumerate() {
        let transformed = fib?;
        let base = base_of(outer_idx);
        for (i, &value) in transformed.iter().enumerate() {
            result[base + i * axis_stride] = value;
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
    let _worker_cap = install_worker_cap(options.workers);

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
    let _worker_cap = install_worker_cap(options.workers);

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
/// When `true`, [`fht`]/[`ifht`] compute their FHT coefficient table serially (the ORIG behaviour);
/// default `false` fans the per-element log-gamma/exp fill across index-chunks. Byte-identical.
/// `#[doc(hidden)]` — internal, exposed only for the same-binary A/B benchmark.
#[doc(hidden)]
pub static FHTCOEFF_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn fhtcoeff(n: usize, dln: f64, mu: f64, offset: f64, bias: f64, inverse: bool) -> Vec<Complex64> {
    use std::f64::consts::{LN_2, PI};
    let (lnkr, q) = (offset, bias);
    let xp = (mu + 1.0 + q) / 2.0;
    let xm = (mu + 1.0 - q) / 2.0;
    let len = n / 2 + 1;

    // Each coefficient u[m] is a pure function of m: y = π·m/(n·dln), then two complex log-gammas
    // and a complex exp — an expensive transcendental per element, independent of every other. Fan
    // the fill across index-chunks; each u[m] is computed by the identical expression, written to a
    // disjoint slot in ascending order → BYTE-IDENTICAL to the serial fill. (The Nyquist/u[0]
    // fixups below run serially on the finished array.) Gated on the coefficient count.
    let coeff = |m: usize| -> Complex64 {
        let y = PI * m as f64 / (n as f64 * dln);
        let lp = complex_ln_gamma((xp, y));
        let lm = complex_ln_gamma((xm, y));
        // exp() is 2πi-periodic in its imaginary part, so any principal-
        // branch ambiguity in the log-gamma phases cancels here.
        let re = lp.0 - lm.0 + LN_2 * q;
        let im = lp.1 + lm.1 + 2.0 * (LN_2 - lnkr) * y;
        complex_exp((re, im))
    };
    let nthreads = if FHTCOEFF_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed) || len < 1024
    {
        1
    } else {
        dispatch_worker_limit().min(len)
    };
    let mut u: Vec<Complex64> = if nthreads <= 1 {
        (0..len).map(coeff).collect()
    } else {
        let mut u = vec![(0.0f64, 0.0f64); len];
        let chunk = len.div_ceil(nthreads);
        let coeff_ref = &coeff;
        std::thread::scope(|scope| {
            for (ci, block) in u.chunks_mut(chunk).enumerate() {
                let base = ci * chunk;
                scope.spawn(move || {
                    for (k, slot) in block.iter_mut().enumerate() {
                        *slot = coeff_ref(base + k);
                    }
                });
            }
        });
        u
    };
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
    let fingerprint = audit_ledger.map_or_else(Vec::new, |_| complex_fingerprint(input));
    ensure_non_empty_with_audit(input.len(), &fingerprint, audit_ledger)?;
    validate_workers_with_audit(options.workers, &fingerprint, audit_ledger)?;
    validate_finite_complex_with_audit(input, options, &fingerprint, audit_ledger)?;
    let _worker_cap = install_worker_cap(options.workers);

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
    let fingerprint =
        audit_ledger.map_or_else(Vec::new, |_| complex_shape_fingerprint(input, shape));
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
    let _worker_cap = install_worker_cap(options.workers);

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
    let _worker_cap = install_worker_cap(options.workers);

    let last_len = *shape.last().ok_or(FftError::InvalidShape {
        detail: "empty shape",
    })?;
    let reduced_last = last_len / 2 + 1;
    let backend = resolve_backend(options.backend);
    let axes = (0..shape.len()).collect::<Vec<_>>();
    let key = PlanKey::new(kind, shape.to_vec(), axes, options.normalization, true);
    let plan_cache_hit = touch_plan_cache(&key, expected_len);

    let started = Instant::now();
    // Real-FFT each contiguous lane along the last axis. The lanes are independent (each is a
    // separate real→complex transform writing its own `reduced_last` outputs), so they are
    // computed in parallel into disjoint contiguous output chunks — byte-identical to the
    // serial `chunks_exact`/`extend` (same per-lane transform, same lane order).
    let n_lanes = expected_len / last_len;
    let mut output = vec![(0.0f64, 0.0f64); n_lanes * reduced_last];
    let lane_work = (n_lanes as u64).saturating_mul(last_len as u64);
    let nthreads = if lane_work < (1 << 15) || n_lanes < 2 {
        1
    } else {
        dispatch_worker_limit().min(n_lanes)
    };
    if nthreads <= 1 {
        for (lane, slot) in input
            .chunks_exact(last_len)
            .zip(output.chunks_mut(reduced_last))
        {
            slot.copy_from_slice(&real_fft_unscaled(lane, backend));
        }
    } else {
        // `&dyn FftBackend` is not `Sync`, so each worker re-resolves the backend from its
        // `BackendKind` (Copy) — the same idiom `apply_axis_transform` uses.
        let backend_kind = options.backend;
        let chunk = n_lanes.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (in_block, out_block) in input
                .chunks(chunk * last_len)
                .zip(output.chunks_mut(chunk * reduced_last))
            {
                scope.spawn(move || {
                    let backend = resolve_backend(backend_kind);
                    for (lane, slot) in in_block
                        .chunks_exact(last_len)
                        .zip(out_block.chunks_mut(reduced_last))
                    {
                        slot.copy_from_slice(&real_fft_unscaled(lane, backend));
                    }
                });
            }
        });
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
    let _worker_cap = install_worker_cap(options.workers);

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
    let cores = dispatch_worker_limit();
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
                        cooley_tukey_radix4_inplace_with_twiddles(lane, twiddles);
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
    // Pow2 axes use the fused radix-2² (radix-4) per-line kernel — bit-identical to
    // the flat radix-2 (proven for the 1-D path, 6bbc73c6) but ~1.35-1.47x fewer
    // passes. It does its own bit-reverse, so the plan carries only the twiddles.
    let cooley_tukey_axis_plan =
        if backend.kind() == BackendKind::CooleyTukey && axis_len.is_power_of_two() {
            Some(get_or_compute_twiddles(axis_len, inverse))
        } else {
            None
        };

    // Axis 0 (repeats == 1) can only be parallelized via the transpose path, which
    // costs two extra array passes; only take it when it will actually run threaded,
    // otherwise the direct strided loop below is cheaper.
    if axis == 0
        && repeats == 1
        && let Some(twiddles) = &cooley_tukey_axis_plan
    {
        let nthreads = nd_axis_thread_count(stride, axis_len);
        if nthreads > 1 {
            apply_axis0_transpose_transform(data, axis_len, stride, nthreads, twiddles);
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
                            if let Some(twiddles) = plan {
                                cooley_tukey_radix4_inplace_with_twiddles(&mut local, twiddles);
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
            if let Some(twiddles) = &cooley_tukey_axis_plan {
                cooley_tukey_radix4_inplace_with_twiddles(axis_scratch, twiddles);
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
    // The pack-two-reals-into-one-complex trick (`real_fft_specialized`) only
    // needs an N/2-point complex FFT, which the backend computes for any size —
    // it is not specific to powers of two. So every EVEN N takes the half-size
    // path instead of transforming N points and discarding the redundant half;
    // for the common non-pow2 even lengths (1000, 22050, 44100, …) that roughly
    // halves the FFT work. Odd N keeps the full transform (no even pack exists).
    if input.len() >= 4 && input.len().is_multiple_of(2) {
        real_fft_specialized(input, backend)
    } else {
        let complex_input: Vec<Complex64> = input.iter().map(|&x| (x, 0.0)).collect();
        let full = backend.transform_1d_unscaled(&complex_input, false);
        full.into_iter().take(input.len() / 2 + 1).collect()
    }
}

fn real_ifft_unscaled(input: &[Complex64], n: usize, backend: &dyn FftBackend) -> Vec<f64> {
    // Even n: invert the real-FFT pack (mirror of `real_fft_specialized`) with a
    // single M = n/2-point complex inverse FFT instead of a full n-point one.
    // Recover the packed half-spectrum Z[k] = even + i·e^{+2πik/n}·odd, where
    // even = (X[k] + conj(X[M-k]))/2 and odd = (X[k] - conj(X[M-k]))/2; then the
    // M-point IFFT gives z, and x[2k] = 2·Re z[k], x[2k+1] = 2·Im z[k] (the ×2
    // matches the unscaled n-point convention this helper returns).
    if n >= 4 && n.is_multiple_of(2) {
        let m = n / 2;
        let twiddles = get_or_compute_twiddles(n, true); // e^{+2πik/n}
        let mut packed = Vec::with_capacity(m);
        for k in 0..m {
            // The packed even/odd split mixes the DC (index 0) and Nyquist
            // (index m) bins only at k == 0. scipy.fft.irfft reconstructs the
            // full Hermitian spectrum and returns the real part, so the
            // imaginary parts of the DC and Nyquist bins NEVER contribute to the
            // output. For a true Hermitian half-spectrum those imaginary parts
            // are already zero (this is a no-op, byte-identical round-trip), but
            // for non-Hermitian last-axis input — as produced by irfftn's
            // complex iFFT over the non-last axes, and thus hfftn/hfft2 — keeping
            // them diverges from scipy. Drop them here to match scipy exactly.
            let (xk, xmk) = if k == 0 {
                ((input[0].0, 0.0), (input[m].0, 0.0))
            } else {
                (input[k], complex_conj(input[m - k]))
            };
            let even = (0.5 * (xk.0 + xmk.0), 0.5 * (xk.1 + xmk.1));
            let odd = (0.5 * (xk.0 - xmk.0), 0.5 * (xk.1 - xmk.1));
            let odd_tw = complex_mul(odd, twiddles[k]);
            // multiply by i: (a, b) -> (-b, a)
            packed.push(complex_add(even, (-odd_tw.1, odd_tw.0)));
        }
        let z = backend.transform_1d_unscaled(&packed, true);
        let mut out = vec![0.0; n];
        for (k, &(re, im)) in z.iter().enumerate() {
            out[2 * k] = 2.0 * re;
            out[2 * k + 1] = 2.0 * im;
        }
        return out;
    }

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
    if touch_shared_plan(key) {
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
/// Returns the smallest integer >= `target` that scipy treats as fast for
/// complex FFTs: products of the small prime factors 2, 3, 5, 7, and 11.
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
/// prime factors {2, 3, 5, 7, 11}. Used to downsize an FFT to a length cheaper
/// than `target`. The empty-product `1` is always fast, so the
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

/// Check if n is composed only of scipy's complex FFT fast factors.
#[allow(clippy::manual_is_multiple_of)]
fn is_fast_len(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for factor in [2usize, 3, 5, 7, 11] {
        while n % factor == 0 {
            n /= factor;
        }
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
    // hfft is a FORWARD transform: scipy normalizes it like fft (backward → 1,
    // ortho → 1/√n, forward → 1/n). Run the inner irfft with BACKWARD
    // normalization (which yields R/n for the unscaled real transform R), then
    // apply the hfft scale — otherwise the inner irfft applies the inverse
    // normalization AND the outer `*out_len` double-counts it (ortho was off by
    // n, forward by n²).
    let backward = options.clone().with_normalization(Normalization::Backward);
    let mut result = irfft_impl(&conjugated, Some(out_len), &backward, audit_ledger)?;
    let nf = out_len as f64;
    let scale = match options.normalization {
        Normalization::Backward => nf,
        Normalization::Ortho => nf.sqrt(),
        Normalization::Forward => 1.0,
    };
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

    // ihfft is an INVERSE transform: scipy normalizes it like ifft (backward →
    // 1/n, ortho → 1/√n, forward → 1). Run the inner rfft with BACKWARD
    // normalization (unscaled spectrum S), then apply the ihfft scale —
    // otherwise the inner rfft applies the forward normalization AND the outer
    // `1/in_len` double-counts it (ortho was off by 1/n, forward by 1/n²).
    let backward = options.clone().with_normalization(Normalization::Backward);
    let mut result = rfft_impl(&padded, &backward, audit_ledger)?;
    let nf = in_len as f64;
    let scale = match options.normalization {
        Normalization::Backward => 1.0 / nf,
        Normalization::Ortho => 1.0 / nf.sqrt(),
        Normalization::Forward => 1.0,
    };
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
    // Forward transform: normalize like fftn (backward → 1, ortho → 1/√N,
    // forward → 1/N) where N = total real output size. Run the inner irfftn with
    // BACKWARD normalization and apply the hfft scale here, so the inner
    // normalization is not double-counted (see hfft_impl).
    let backward = options.clone().with_normalization(Normalization::Backward);
    let mut result = irfftn_impl(&conjugated, shape, &backward, audit_ledger)?;
    let nf = result.len() as f64;
    let scale = match options.normalization {
        Normalization::Backward => nf,
        Normalization::Ortho => nf.sqrt(),
        Normalization::Forward => 1.0,
    };
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
    // Inverse transform: normalize like ifftn (backward → 1/N, ortho → 1/√N,
    // forward → 1). Run the inner rfftn with BACKWARD normalization and apply
    // the ihfft scale here, so the inner normalization is not double-counted
    // (see ihfft_impl).
    let backward = options.clone().with_normalization(Normalization::Backward);
    let mut result = rfftn_impl(input, shape, &backward, audit_ledger)?;
    let nf = input.len() as f64;
    let scale = match options.normalization {
        Normalization::Backward => 1.0 / nf,
        Normalization::Ortho => 1.0 / nf.sqrt(),
        Normalization::Forward => 1.0,
    };
    for value in &mut result {
        value.0 *= scale;
        value.1 *= -scale;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use fsci_runtime::{AuditAction, AuditLedger, RuntimeMode};

    use super::{
        Complex64, FftError, FftOptions, TransformKind, WorkerPolicy, dct, dct_axis2d, dct_iv,
        dctn, dst, dst_ii, dst_iii, dstn, estimate_fft_flops, fft, fft_axis2d,
        fft_iter_par_threads, fft_radix4_par_threads, fft_with_audit, fft2, fft2_with_audit, fftn,
        fwht, get_workers, hfft, hfft2, hfftn, idct, idct_axis2d, idctn, idstn, ifft, ifft2, ifftn,
        ihfft, ihfft2, ihfftn, install_worker_cap, irfft, irfft_with_audit, irfft2, irfftn,
        is_fast_len, next_fast_len, prev_fast_len, rfft, rfft_axis2d, rfft_with_audit, rfft2,
        rfftn, set_workers, sync_audit_ledger, take_transform_traces, worker_policy_cap,
    };
    use super::{
        cooley_tukey_radix2_inplace, cooley_tukey_radix4_inplace_with_twiddles,
        get_or_compute_twiddles, rader_fft,
    };

    fn naive_fft_for_test(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        let mut out = vec![(0.0, 0.0); n];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = (0.0, 0.0);
            for (t, &value) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (k * t % n) as f64 / n as f64;
                let twiddle = (angle.cos(), angle.sin());
                acc = (
                    acc.0 + value.0 * twiddle.0 - value.1 * twiddle.1,
                    acc.1 + value.0 * twiddle.1 + value.1 * twiddle.0,
                );
            }
            *slot = acc;
        }
        out
    }

    #[test]
    fn mixed_radix_smooth_power_tail_matches_naive_dft() {
        for n in [12_usize, 20, 60, 100, 360] {
            let input: Vec<Complex64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    (
                        (0.19 * t).sin() + 0.07 * t.cos(),
                        (0.23 * t).cos() - 0.03 * t.sin(),
                    )
                })
                .collect();
            let got = fft(&input, &FftOptions::default()).expect("fft");
            let want = naive_fft_for_test(&input);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let err = ((g.0 - w.0).powi(2) + (g.1 - w.1).powi(2)).sqrt();
                assert!(err < 1.0e-9, "n={n} i={i} err={err} got={g:?} want={w:?}");
            }
        }
    }

    #[test]
    fn radix4_bit_identical_to_radix2() {
        // The fused radix-2² kernel must be BIT-IDENTICAL to the radix-2 sweep
        // (same twiddle indices, same op order) across even and odd log2(n).
        let mut seed = 0xD1B54A32D192ED03u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for &e in &[1usize, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 16] {
            let n = 1usize << e;
            let x: Vec<Complex64> = (0..n).map(|_| (rnd(), rnd())).collect();
            for inverse in [false, true] {
                let tw = get_or_compute_twiddles(n, inverse);
                let mut a = x.clone();
                let mut b = x.clone();
                cooley_tukey_radix2_inplace(&mut a, inverse);
                cooley_tukey_radix4_inplace_with_twiddles(&mut b, &tw);
                assert_eq!(a, b, "radix4 != radix2 e={e} inverse={inverse}");
            }
        }
    }

    #[test]
    fn hfft_ihfft_match_scipy_all_norms() {
        // Regression: hfft/ihfft passed the user's `norm` to the inner irfft/rfft
        // AND applied their own scaling, double-counting normalization — so
        // ortho was off by n (hfft) / 1/n (ihfft) and forward by n² / 1/n². Only
        // the default backward norm was correct. Reference values from
        // scipy.fft.hfft / scipy.fft.ihfft 1.17.1.
        let opt = |n| FftOptions::default().with_normalization(n);

        // hfft of a 3-point spectrum → default n = 2*(3-1) = 4 real outputs.
        let spec: Vec<Complex64> = vec![(2.0, 0.0), (1.0, -1.0), (0.5, 0.0)];
        let hfft_expected = [
            (Normalization::Backward, [4.5, -0.5, 0.5, 3.5]),
            (Normalization::Ortho, [2.25, -0.25, 0.25, 1.75]),
            (Normalization::Forward, [1.125, -0.125, 0.125, 0.875]),
        ];
        for (norm, expected) in hfft_expected {
            let got = hfft(&spec, None, &opt(norm)).expect("hfft");
            assert_eq!(got.len(), 4);
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-12, "hfft {norm:?}: got {g}, want {e}");
            }
        }

        // ihfft of a 4-point real signal → 3-point Hermitian half-spectrum.
        let sig = [1.0, 2.0, 3.0, 4.0];
        let ihfft_expected = [
            (
                Normalization::Backward,
                [(2.5, 0.0), (-0.5, -0.5), (-0.5, 0.0)],
            ),
            (
                Normalization::Ortho,
                [(5.0, 0.0), (-1.0, -1.0), (-1.0, 0.0)],
            ),
            (
                Normalization::Forward,
                [(10.0, 0.0), (-2.0, -2.0), (-2.0, 0.0)],
            ),
        ];
        for (norm, expected) in ihfft_expected {
            let got = ihfft(&sig, None, &opt(norm)).expect("ihfft");
            assert_eq!(got.len(), 3);
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!(
                    (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                    "ihfft {norm:?}: got {g:?}, want {e:?}"
                );
            }
        }

        // N-D inverse (ihfft2) is also clean across norms: the ortho/forward
        // result equals the backward result scaled by √N / N (the relationship
        // the double-count violated). N = 12 for a 3×4 grid.
        let sig2: Vec<f64> = (0..12)
            .map(|k| (0.5 * k as f64).cos() + 0.2 * k as f64 + 0.9)
            .collect();
        let back = ihfft2(&sig2, (3, 4), &opt(Normalization::Backward)).expect("ihfft2 b");
        let ortho = ihfft2(&sig2, (3, 4), &opt(Normalization::Ortho)).expect("ihfft2 o");
        let fwd = ihfft2(&sig2, (3, 4), &opt(Normalization::Forward)).expect("ihfft2 f");
        let n = 12.0_f64;
        for ((b, o), f) in back.iter().zip(ortho.iter()).zip(fwd.iter()) {
            assert!((o.0 - b.0 * n.sqrt()).abs() < 1e-10 && (o.1 - b.1 * n.sqrt()).abs() < 1e-10);
            assert!((f.0 - b.0 * n).abs() < 1e-10 && (f.1 - b.1 * n).abs() < 1e-10);
        }
    }

    #[test]
    fn irfftn_hfft2_match_scipy_non_hermitian() {
        // Regression (frankenscipy-w4vr9): irfftn applies a complex iFFT over the
        // non-last axes and an irfft over the last axis. The packed even-n irfft
        // mixed the DC and Nyquist bins at k==0, carrying their imaginary parts —
        // which scipy.fft.irfft always drops (it reconstructs the full Hermitian
        // spectrum and returns the real part). For a TRUE Hermitian half-spectrum
        // those imaginary parts are zero (round-trip stayed bit-clean), but the
        // non-Hermitian half-spectrum produced by the non-last-axis iFFT diverged
        // on every element. Reference values from scipy.fft.irfft2 / hfft2 1.17.1.
        let o = FftOptions::default();

        // irfft2 of an arbitrary (non-Hermitian) 3×3 half-spectrum → 3×4 real.
        let spec: Vec<Complex64> = (0..9)
            .map(|k| {
                let t = k as f64;
                ((0.4 * t).cos() + 0.5, -(0.3 * t).sin() * 0.6)
            })
            .collect();
        let irfft2_expected = [
            0.483_875_616_500_547_8,
            0.320_968_444_421_923_7,
            0.000_662_572_822_901_8,
            -0.097_185_287_433_564_1,
            0.462_071_204_481_245_2,
            -0.232_222_197_841_266_9,
            -0.018_774_427_774_383_7,
            0.151_765_591_068_411,
            0.438_760_353_356_441,
            0.075_733_138_080_953_6,
            -0.018_241_964_713_169_3,
            -0.067_413_042_970_040_0,
        ];
        let got = irfft2(&spec, (3, 4), &o).expect("irfft2");
        assert_eq!(got.len(), 12);
        for (g, e) in got.iter().zip(irfft2_expected.iter()) {
            assert!((g - e).abs() < 1e-12, "irfft2: got {g}, want {e}");
        }

        // hfft2 == irfftn(conj(x)) * N on the same (non-Hermitian) spectrum.
        let spec_h: Vec<Complex64> = (0..9)
            .map(|k| {
                let t = k as f64;
                ((0.4 * t).cos() + 0.5, (0.3 * t).sin() * 0.6)
            })
            .collect();
        let hfft2_expected = [
            5.806_507_398_006_573,
            3.851_621_333_063_085,
            0.007_950_873_874_821_5,
            -1.166_223_449_202_769_2,
            5.544_854_453_774_942,
            -2.786_666_374_095_203,
            -0.225_293_133_292_604_2,
            1.821_187_092_820_931_7,
            5.265_124_240_277_292,
            0.908_797_656_971_443_3,
            -0.218_903_576_558_031_6,
            -0.808_956_515_640_480_3,
        ];
        let got_h = hfft2(&spec_h, (3, 4), &o).expect("hfft2");
        assert_eq!(got_h.len(), 12);
        for (g, e) in got_h.iter().zip(hfft2_expected.iter()) {
            assert!((g - e).abs() < 1e-11, "hfft2: got {g}, want {e}");
        }
    }

    #[test]
    fn idstn_dstn_round_trip_all_norms() {
        // idstn(dstn(x)) == x for every normalization. Regression: idstn with
        // Forward normalization was wrong by 1/(2N) per axis (the inverse forgot
        // to undo dst_iii's forward 1/(2N) scaling), so the round trip drifted by
        // the product of (2*N_axis).
        let shape = [3usize, 4];
        let x: Vec<f64> = (0..12)
            .map(|k| (0.6 * k as f64).sin() + 0.2 * k as f64 - 0.04 * (k * k) as f64 + 1.1)
            .collect();
        for norm in [
            Normalization::Backward,
            Normalization::Ortho,
            Normalization::Forward,
        ] {
            let o = FftOptions {
                normalization: norm,
                ..Default::default()
            };
            let fwd = dstn(&x, &shape, &o).expect("dstn");
            let back = idstn(&fwd, &shape, &o).expect("idstn");
            for (a, b) in x.iter().zip(&back) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "idstn(dstn(x)) != x for {norm:?}: {a} vs {b}"
                );
            }
        }
    }
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
        assert_eq!(opts.workers, WorkerPolicy::Exact(1));
    }

    #[test]
    fn worker_context_is_thread_local_and_nestable() {
        assert_eq!(get_workers(), 1);
        {
            let _outer = set_workers(4).expect("positive worker count");
            assert_eq!(get_workers(), 4);
            assert_eq!(
                FftOptions::default().workers,
                WorkerPolicy::Exact(4),
                "default options capture the active context"
            );
            assert_eq!(
                std::thread::spawn(get_workers)
                    .join()
                    .expect("worker thread"),
                1,
                "another thread starts with SciPy's default"
            );
            {
                let _inner = set_workers(2).expect("nested worker count");
                assert_eq!(get_workers(), 2);
            }
            assert_eq!(get_workers(), 4);
        }
        assert_eq!(get_workers(), 1);
    }

    #[test]
    fn worker_policy_caps_all_1d_fft_spawn_counts() {
        // These helpers return the number of scoped workers that their callers
        // spawn. Exact(1) must therefore keep both parallel FFT routes serial.
        assert_eq!(worker_policy_cap(WorkerPolicy::Auto, 8), 8);
        assert_eq!(worker_policy_cap(WorkerPolicy::Exact(1), 8), 1);
        assert_eq!(worker_policy_cap(WorkerPolicy::Exact(16), 8), 8);
        assert_eq!(worker_policy_cap(WorkerPolicy::Max(3), 8), 3);

        let _serial = install_worker_cap(WorkerPolicy::Exact(1));
        assert_eq!(fft_iter_par_threads(1 << 16, 8), 1);
        assert_eq!(fft_radix4_par_threads(1 << 20, 8), 1);
    }

    #[test]
    fn worker_context_matches_scipy_validation_and_negative_wrapping() {
        assert_eq!(
            set_workers(0).expect_err("zero must be rejected"),
            FftError::InvalidWorkers { requested: 0 }
        );

        let cpu_count = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        {
            let _all = set_workers(-1).expect("-1 selects every available CPU");
            assert_eq!(get_workers(), cpu_count);
        }
        {
            let last_relative =
                -isize::try_from(cpu_count).expect("available CPU count fits isize");
            let _one = set_workers(last_relative).expect("-N selects one worker");
            assert_eq!(get_workers(), 1);
        }
        let below_range = -isize::try_from(cpu_count).expect("available CPU count fits isize") - 1;
        assert_eq!(
            set_workers(below_range).expect_err("values below -N must be rejected"),
            FftError::InvalidWorkers {
                requested: below_range
            }
        );
    }

    #[test]
    fn exact_zero_workers_is_rejected() {
        let opts = FftOptions::default().with_workers(WorkerPolicy::Exact(0));
        let err = fft(&[(0.0, 0.0)], &opts).expect_err("zero workers should be rejected");
        assert_eq!(err, FftError::InvalidWorkers { requested: 0 });
    }

    #[test]
    fn fft_rfft_match_numpy_small() {
        // numpy.fft.fft([1,2,3,4]) = [10, -2+2j, -2, -2-2j];
        // numpy.fft.rfft([1,2,3,4]) = [10, -2+2j, -2].
        let opts = FftOptions::default();
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&v| (v, 0.0)).collect();
        let f = fft(&x, &opts).expect("fft");
        let ef = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
        for (g, e) in f.iter().zip(&ef) {
            assert!(
                (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                "fft: {g:?} vs {e:?}"
            );
        }
        let r = rfft(&[1.0, 2.0, 3.0, 4.0], &opts).expect("rfft");
        let er = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)];
        assert_eq!(r.len(), 3, "rfft length");
        for (g, e) in r.iter().zip(&er) {
            assert!(
                (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                "rfft: {g:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn hfft_match_scipy() {
        // scipy.fft.hfft of a Hermitian half-spectrum -> real output.
        let input: Vec<Complex64> = vec![(1.0, 0.0), (2.0, 1.0), (3.0, 0.0)];
        let r = hfft(&input, Some(4), &FftOptions::default()).expect("hfft");
        for (g, ex) in r.iter().zip(&[8.0, 0.0, 0.0, -4.0]) {
            assert!((g - ex).abs() < 1e-12, "hfft: {g} vs {ex}");
        }
    }

    #[test]
    fn ifft2_match_scipy() {
        // scipy.fft.ifft2([[10,-2],[-4,0]]) = [[1,2],[3,4]] (inverse of fft2).
        let x: Vec<Complex64> = [10.0, -2.0, -4.0, 0.0].iter().map(|&r| (r, 0.0)).collect();
        let r = ifft2(&x, (2, 2), &FftOptions::default()).expect("ifft2");
        let expect = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        for (g, e) in r.iter().zip(&expect) {
            assert!(
                (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                "ifft2: {g:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn fft2_match_scipy() {
        // scipy.fft.fft2([[1,2],[3,4]]) = [[10,-2],[-4,0]] (row-major flat).
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&r| (r, 0.0)).collect();
        let r = fft2(&x, (2, 2), &FftOptions::default()).expect("fft2");
        let expect = [(10.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), (0.0, 0.0)];
        for (g, e) in r.iter().zip(&expect) {
            assert!(
                (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                "fft2: {g:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn dst_ii_match_scipy() {
        // scipy.fft.dst([1,2,3,4], type=2). Backward == norm=None; Ortho == 'ortho'.
        let x = [1.0, 2.0, 3.0, 4.0];
        let back = dst(&x, 2, &FftOptions::default()).expect("dst");
        let eb = [
            13.065_629_648_763_766,
            -5.656_854_249_492_38,
            5.411_961_001_461_97,
            -4.0,
        ];
        for (g, e) in back.iter().zip(&eb) {
            assert!((g - e).abs() < 1e-11, "dst backward: {g} vs {e}");
        }
        let ortho = dst(
            &x,
            2,
            &FftOptions::default().with_normalization(Normalization::Ortho),
        )
        .expect("dst ortho");
        let eo = [
            4.619_397_662_556_434,
            -2.0,
            1.913_417_161_825_449,
            -1.000_000_000_000_000_2,
        ];
        for (g, e) in ortho.iter().zip(&eo) {
            assert!((g - e).abs() < 1e-11, "dst ortho: {g} vs {e}");
        }
    }

    #[test]
    fn fft_normalization_conventions_match_scipy() {
        // scipy.fft.fft norm=backward/ortho/forward and ifft (1/n) for [1,2,3,4].
        let x: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&r| (r, 0.0)).collect();
        let chk = |got: &[Complex64], want: &[(f64, f64)], n: &str| {
            for (g, e) in got.iter().zip(want) {
                assert!(
                    (g.0 - e.0).abs() < 1e-12 && (g.1 - e.1).abs() < 1e-12,
                    "{n}: {g:?} vs {e:?}"
                );
            }
        };
        let back = fft(&x, &FftOptions::default()).unwrap();
        chk(
            &back,
            &[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            "fft backward",
        );
        let ortho = fft(
            &x,
            &FftOptions::default().with_normalization(Normalization::Ortho),
        )
        .unwrap();
        chk(
            &ortho,
            &[(5.0, 0.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0)],
            "fft ortho",
        );
        let fwd = fft(
            &x,
            &FftOptions::default().with_normalization(Normalization::Forward),
        )
        .unwrap();
        chk(
            &fwd,
            &[(2.5, 0.0), (-0.5, 0.5), (-0.5, 0.0), (-0.5, -0.5)],
            "fft forward",
        );
        let inv = ifft(&x, &FftOptions::default()).unwrap();
        chk(
            &inv,
            &[(2.5, 0.0), (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5)],
            "ifft",
        );
    }

    #[test]
    fn dct_ii_match_scipy() {
        // scipy.fft.dct([1,2,3,4], type=2). Backward == norm=None; Ortho == 'ortho'.
        let x = [1.0, 2.0, 3.0, 4.0];
        let back = dct(&x, &FftOptions::default()).expect("dct");
        let eb = [20.0, -6.308_644_059_797_899, 0.0, -0.448_341_529_167_965_1];
        for (g, e) in back.iter().zip(&eb) {
            assert!((g - e).abs() < 1e-12, "dct backward: {g} vs {e}");
        }
        let ortho = dct(
            &x,
            &FftOptions::default().with_normalization(Normalization::Ortho),
        )
        .expect("dct ortho");
        let eo = [
            5.000_000_000_000_001,
            -2.230_442_497_387_663_5,
            0.0,
            -0.158_512_667_781_107_06,
        ];
        for (g, e) in ortho.iter().zip(&eo) {
            assert!((g - e).abs() < 1e-12, "dct ortho: {g} vs {e}");
        }
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
    fn fft_with_audit_preserves_complex_input_fingerprint() {
        let input: [Complex64; 2] = [(1.25, -0.0), (-3.5, 2.0)];
        let mut input_bytes = Vec::new();
        for &(re, im) in &input {
            input_bytes.extend_from_slice(&re.to_le_bytes());
            input_bytes.extend_from_slice(&im.to_le_bytes());
        }
        let expected_fingerprint = AuditLedger::fingerprint_bytes(&input_bytes);
        let audit_ledger = sync_audit_ledger();
        let opts = FftOptions::default().with_workers(WorkerPolicy::Exact(0));

        let err = fft_with_audit(&input, &opts, &audit_ledger)
            .expect_err("zero workers should be rejected");
        assert_eq!(err, FftError::InvalidWorkers { requested: 0 });

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert_eq!(ledger.entries().len(), 1);
        let entry = &ledger.entries()[0];
        assert!(matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "invalid_workers"
        ));
        assert_eq!(entry.input_fingerprint, expected_fingerprint);
    }

    #[test]
    fn fft2_with_audit_preserves_complex_shape_fingerprint() {
        let input: [Complex64; 4] = [
            (1.25, -0.0),
            (f64::from_bits(0x7ff8_0000_0000_0042), 3.5),
            (-2.0, 4.25),
            (7.0, -8.5),
        ];
        let shape = (2usize, 2usize);
        let mut input_bytes = Vec::new();
        for &(re, im) in &input {
            input_bytes.extend_from_slice(&re.to_le_bytes());
            input_bytes.extend_from_slice(&im.to_le_bytes());
        }
        for dim in [shape.0, shape.1] {
            input_bytes.extend_from_slice(&dim.to_le_bytes());
        }
        let expected_fingerprint = AuditLedger::fingerprint_bytes(&input_bytes);
        let audit_ledger = sync_audit_ledger();
        let opts = FftOptions::default()
            .with_workers(WorkerPolicy::Exact(0))
            .with_check_finite(true);

        let err = fft2_with_audit(&input, shape, &opts, &audit_ledger)
            .expect_err("zero workers should be rejected before the finite-input check");
        assert_eq!(err, FftError::InvalidWorkers { requested: 0 });

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert_eq!(ledger.entries().len(), 1);
        let entry = &ledger.entries()[0];
        assert!(matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "invalid_workers"
        ));
        assert_eq!(entry.input_fingerprint, expected_fingerprint);
    }

    #[test]
    fn irfft_with_audit_preserves_complex_fingerprint_on_invalid_output_len() {
        let input: [Complex64; 2] = [(1.25, -0.0), (f64::from_bits(0x7ff8_0000_0000_0042), 3.5)];
        let mut input_bytes = Vec::new();
        for (re, im) in input {
            input_bytes.extend_from_slice(&re.to_le_bytes());
            input_bytes.extend_from_slice(&im.to_le_bytes());
        }
        let expected_fingerprint = AuditLedger::fingerprint_bytes(&input_bytes);
        let audit_ledger = sync_audit_ledger();

        let err = irfft_with_audit(&input, Some(0), &FftOptions::default(), &audit_ledger)
            .expect_err("zero output length should be rejected");
        assert_eq!(
            err,
            FftError::InvalidShape {
                detail: "output_len cannot be zero",
            }
        );

        let ledger = audit_ledger.lock().expect("audit ledger lock");
        assert_eq!(ledger.entries().len(), 1);
        let entry = &ledger.entries()[0];
        assert!(matches!(
            &entry.action,
            AuditAction::FailClosed { reason } if reason == "invalid_output_len"
        ));
        assert_eq!(entry.input_fingerprint, expected_fingerprint);
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
        // ⌊N/2⌋+1 bins of fft(x as complex). Pin across multiple N, including the
        // awkward prime-factor lengths that route the complex engine through Rader
        // (101), Bluestein-5smooth (1031), and the split-factor peel (1010 = 2·5·101,
        // 2510 = 2·5·251) — guards that rfft inherits those fast paths correctly.
        for n in [4_usize, 8, 16, 32, 13, 101, 1031, 1010, 2510] {
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
    fn mixed_radix_2k_large_prime_matches_naive() {
        // n = 2^k · large-prime lengths: the split-factor fix peels the 2/4 first
        // so the large prime is transformed whole (Rader/Bluestein) instead of via
        // an O(p²) radix-p combine. Verify the result still equals the naive DFT.
        let naive_dft = |x: &[Complex64]| -> Vec<Complex64> {
            let n = x.len();
            (0..n)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for (j, &(re, im)) in x.iter().enumerate() {
                        let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                        let (c, s) = (a.cos(), a.sin());
                        acc.0 += re * c - im * s;
                        acc.1 += re * s + im * c;
                    }
                    acc
                })
                .collect()
        };
        // 202=2·101, 502=2·251, 1010=2·5·101, 2510=2·5·251, 1030=2·5·103, 1016=2³·127
        for &n in &[202usize, 502, 1010, 2510, 1030, 1016] {
            let x: Vec<Complex64> = (0..n)
                .map(|i| ((i as f64 * 0.3).sin(), (i as f64 * 0.11).cos()))
                .collect();
            let got = fft(&x, &FftOptions::default()).expect("fft");
            let want = naive_dft(&x);
            let maxerr = got
                .iter()
                .zip(&want)
                .map(|(&(a, b), &(c, d))| ((a - c).powi(2) + (b - d).powi(2)).sqrt())
                .fold(0.0_f64, f64::max);
            assert!(maxerr < 1e-7, "n={n} fft maxerr {maxerr} vs naive DFT");
        }
    }

    #[test]
    fn mixed_radix_factor17_iterative_matches_naive() {
        let naive_dft = |x: &[Complex64]| -> Vec<Complex64> {
            let n = x.len();
            (0..n)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for (j, &(re, im)) in x.iter().enumerate() {
                        let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                        let (c, s) = (a.cos(), a.sin());
                        acc.0 += re * c - im * s;
                        acc.1 += re * s + im * c;
                    }
                    acc
                })
                .collect()
        };
        for &n in &[17usize, 34, 51, 68, 85, 289] {
            let x: Vec<Complex64> = (0..n)
                .map(|i| ((i as f64 * 0.3).sin(), (i as f64 * 0.11).cos()))
                .collect();
            let got = fft(&x, &FftOptions::default()).expect("fft");
            let want = naive_dft(&x);
            let maxerr = got
                .iter()
                .zip(&want)
                .map(|(&(a, b), &(c, d))| ((a - c).powi(2) + (b - d).powi(2)).sqrt())
                .fold(0.0_f64, f64::max);
            assert!(
                maxerr < 1e-7,
                "n={n} factor-17 fft maxerr {maxerr} vs naive DFT"
            );
        }
    }

    #[test]
    fn rader_thirteen_smooth_prime_minus_one_matches_naive() {
        let naive_dft = |x: &[Complex64]| -> Vec<Complex64> {
            let n = x.len();
            (0..n)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for (j, &(re, im)) in x.iter().enumerate() {
                        let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                        let (c, s) = (a.cos(), a.sin());
                        acc.0 += re * c - im * s;
                        acc.1 += re * s + im * c;
                    }
                    acc
                })
                .collect()
        };
        for &n in &[1093usize, 1409] {
            let x: Vec<Complex64> = (0..n)
                .map(|i| ((i as f64 * 0.19).sin(), (i as f64 * 0.13).cos()))
                .collect();
            let got = fft(&x, &FftOptions::default()).expect("fft");
            let want = naive_dft(&x);
            let maxerr = got
                .iter()
                .zip(&want)
                .map(|(&(a, b), &(c, d))| ((a - c).powi(2) + (b - d).powi(2)).sqrt())
                .fold(0.0_f64, f64::max);
            assert!(
                maxerr < 1e-7,
                "n={n} Rader FFT maxerr {maxerr} vs naive DFT"
            );
        }
    }

    #[test]
    fn rader_prime_dft_matches_naive() {
        let naive_dft = |x: &[Complex64], inverse: bool| -> Vec<Complex64> {
            let n = x.len();
            let sign = if inverse { 1.0 } else { -1.0 };
            (0..n)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for (j, &(re, im)) in x.iter().enumerate() {
                        let a = sign * 2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                        let (c, s) = (a.cos(), a.sin());
                        acc.0 += re * c - im * s;
                        acc.1 += re * s + im * c;
                    }
                    acc
                })
                .collect()
        };
        // primes p where p-1 is smooth (the gated case): 67,71,101,103,1031.
        for &p in &[67usize, 71, 101, 103, 1031] {
            let x: Vec<Complex64> = (0..p)
                .map(|i| ((i as f64 * 0.7 + 1.0).sin(), (i as f64 * 0.23).cos()))
                .collect();
            for &inv in &[false, true] {
                let got = rader_fft(&x, inv);
                let want = naive_dft(&x, inv);
                let maxerr = got
                    .iter()
                    .zip(&want)
                    .map(|(&(a, b), &(c, d))| ((a - c).powi(2) + (b - d).powi(2)).sqrt())
                    .fold(0.0_f64, f64::max);
                assert!(maxerr < 1e-7, "p={p} inv={inv} rader maxerr {maxerr}");
            }
        }
    }

    #[test]
    fn bluestein_5smooth_lengths_match_naive_dft() {
        // Lengths whose largest prime factor > 61 take the Bluestein route, and
        // for large-prime n the convolution length is now an even 5-smooth (e.g.
        // n=1031 → m=2160, not pow2 4096). Verify the spectrum still equals the
        // naive O(n²) DFT to FFT tolerance for both gate branches (n=1031 picks
        // 5-smooth m; n=1030 = 2·5·103 keeps pow2 m via the gate).
        let naive_dft = |x: &[Complex64]| -> Vec<Complex64> {
            let n = x.len();
            (0..n)
                .map(|k| {
                    let mut acc = (0.0, 0.0);
                    for (j, &(re, im)) in x.iter().enumerate() {
                        let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                        let (c, s) = (a.cos(), a.sin());
                        acc.0 += re * c - im * s;
                        acc.1 += re * s + im * c;
                    }
                    acc
                })
                .collect()
        };
        for &n in &[1031usize, 1030] {
            let x: Vec<Complex64> = (0..n)
                .map(|i| ((i as f64 * 0.3).sin(), (i as f64 * 0.11).cos()))
                .collect();
            let got = fft(&x, &FftOptions::default()).expect("fft");
            let want = naive_dft(&x);
            let maxerr = got
                .iter()
                .zip(&want)
                .map(|(&(a, b), &(c, d))| ((a - c).powi(2) + (b - d).powi(2)).sqrt())
                .fold(0.0_f64, f64::max);
            assert!(
                maxerr < 1e-7,
                "n={n} Bluestein fft maxerr {maxerr} vs naive DFT"
            );
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
        assert_eq!(next_fast_len(7), 7);
        assert_eq!(next_fast_len(11), 11);
        assert_eq!(next_fast_len(13), 14); // 14 = 2 * 7
        assert_eq!(next_fast_len(17), 18); // 18 = 2 * 3^2
        assert_eq!(next_fast_len(101), 105); // 105 = 3 * 5 * 7
    }

    #[test]
    fn next_fast_len_already_fast() {
        // 30 = 2 * 3 * 5
        assert_eq!(next_fast_len(30), 30);
        // 77 = 7 * 11
        assert_eq!(next_fast_len(77), 77);
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

    #[test]
    fn real_nd_transforms_reject_shape_product_overflow() {
        fn assert_overflow_error(result: Result<Vec<f64>, FftError>) {
            assert_eq!(
                result,
                Err(FftError::InvalidShape {
                    detail: "nd shape product overflow",
                })
            );
        }

        let x = [1.0_f64];
        let shape = [usize::MAX, 2usize];
        let opts = FftOptions::default();

        assert_overflow_error(dctn(&x, &shape, &opts));
        assert_overflow_error(idctn(&x, &shape, &opts));
        assert_overflow_error(dstn(&x, &shape, &opts));
        assert_overflow_error(idstn(&x, &shape, &opts));
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
        assert_eq!(prev_fast_len(7), 7); // 7 is fast for scipy's complex FFT path
        assert_eq!(prev_fast_len(11), 11); // 11 is fast for scipy's complex FFT path
        assert_eq!(prev_fast_len(13), 12); // 12 = 2²·3
        assert_eq!(prev_fast_len(17), 16); // 16 = 2⁴
        assert_eq!(prev_fast_len(31), 30); // 30 = 2·3·5
        assert_eq!(prev_fast_len(1025), 1024);
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
            assert!(is_fast_len(p), "prev_fast_len({t}) = {p} not scipy-fast");
            assert!(is_fast_len(n), "next_fast_len({t}) = {n} not scipy-fast");
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

    #[test]
    fn fwht_matches_naive_hadamard_and_is_involution() {
        let opts = FftOptions::default();
        // Smallest case: [a, b] -> [a+b, a-b].
        let two = fwht(&[3.0, 5.0], &opts).expect("fwht");
        assert_eq!(two, vec![8.0, -2.0]);

        // scipy.linalg.hadamard(4) @ [1, -2, 3, 4].
        let four = fwht(&[1.0, -2.0, 3.0, 4.0], &opts).expect("fwht");
        assert_eq!(four, vec![6.0, 2.0, -8.0, 4.0]);

        // Match the naive H_n·x (H[k][j] = (-1)^popcount(k&j)) for random input.
        let n = 64usize;
        let mut s: u64 = 0xdead_beef_1234_5678;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        let x: Vec<f64> = (0..n).map(|_| rng()).collect();
        let fast = fwht(&x, &opts).expect("fwht");
        for (k, &fk) in fast.iter().enumerate() {
            let naive: f64 = (0..n)
                .map(|j| {
                    if (k & j).count_ones() & 1 == 0 {
                        x[j]
                    } else {
                        -x[j]
                    }
                })
                .sum();
            assert!(
                (fk - naive).abs() < 1e-12,
                "fwht[{k}] {fk} vs naive {naive}"
            );
        }

        // Involution up to scale: fwht(fwht(x)) == n·x.
        let twice = fwht(&fast, &opts).expect("fwht");
        for (&t, &xi) in twice.iter().zip(&x) {
            assert!(
                (t - n as f64 * xi).abs() < 1e-10,
                "involution: {t} vs {}",
                n as f64 * xi
            );
        }

        // Non-power-of-two length is rejected.
        assert!(fwht(&[1.0, 2.0, 3.0], &opts).is_err());
    }

    #[test]
    fn fft_rfft_axis2d_match_per_row() {
        let opts = FftOptions::default();
        // A few (rows, ncols) including a non-power-of-two ncols to exercise the general path.
        for &(rows, ncols) in &[(5usize, 8usize), (7, 12), (33, 16)] {
            let cplx: Vec<Complex64> = (0..rows * ncols)
                .map(|i| ((i as f64).sin(), (i as f64 * 0.3).cos()))
                .collect();
            let real: Vec<f64> = (0..rows * ncols).map(|i| (i as f64 * 0.7).sin()).collect();

            let fa = fft_axis2d(&cplx, rows, ncols, &opts).unwrap();
            let ra = rfft_axis2d(&real, rows, ncols, &opts).unwrap();
            let rout = ncols / 2 + 1;
            for r in 0..rows {
                let frow = fft(&cplx[r * ncols..(r + 1) * ncols], &opts).unwrap();
                let rrow = rfft(&real[r * ncols..(r + 1) * ncols], &opts).unwrap();
                for c in 0..ncols {
                    assert_eq!(
                        fa[r * ncols + c].0.to_bits(),
                        frow[c].0.to_bits(),
                        "fft re {r},{c}"
                    );
                    assert_eq!(
                        fa[r * ncols + c].1.to_bits(),
                        frow[c].1.to_bits(),
                        "fft im {r},{c}"
                    );
                }
                for c in 0..rout {
                    assert_eq!(
                        ra[r * rout + c].0.to_bits(),
                        rrow[c].0.to_bits(),
                        "rfft re {r},{c}"
                    );
                    assert_eq!(
                        ra[r * rout + c].1.to_bits(),
                        rrow[c].1.to_bits(),
                        "rfft im {r},{c}"
                    );
                }
            }
        }
        // Shape validation.
        assert!(fft_axis2d(&[(1.0, 0.0); 6], 2, 4, &opts).is_err());
        assert!(rfft_axis2d(&[1.0; 6], 0, 4, &opts).is_err());
    }

    #[test]
    fn dct_idct_axis2d_match_per_row() {
        let opts = FftOptions::default();
        for &(rows, ncols) in &[(5usize, 8usize), (7, 12), (33, 16)] {
            let real: Vec<f64> = (0..rows * ncols).map(|i| (i as f64 * 0.7).sin()).collect();
            let da = dct_axis2d(&real, rows, ncols, &opts).unwrap();
            let ia = idct_axis2d(&real, rows, ncols, &opts).unwrap();
            for r in 0..rows {
                let drow = dct(&real[r * ncols..(r + 1) * ncols], &opts).unwrap();
                let irow = idct(&real[r * ncols..(r + 1) * ncols], &opts).unwrap();
                for c in 0..ncols {
                    assert_eq!(
                        da[r * ncols + c].to_bits(),
                        drow[c].to_bits(),
                        "dct {r},{c}"
                    );
                    assert_eq!(
                        ia[r * ncols + c].to_bits(),
                        irow[c].to_bits(),
                        "idct {r},{c}"
                    );
                }
            }
        }
        assert!(dct_axis2d(&[1.0; 6], 2, 4, &opts).is_err());
        assert!(idct_axis2d(&[1.0; 6], 0, 4, &opts).is_err());
    }
}
