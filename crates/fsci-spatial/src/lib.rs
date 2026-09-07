#![feature(portable_simd)]
#![forbid(unsafe_code)]

//! Spatial data structures and algorithms for FrankenSciPy.
//!
//! Matches `scipy.spatial` core types:
//! - `KDTree` — k-d tree for fast nearest-neighbor queries
//! - `cKDTree` — SciPy parity alias to `KDTree`
//! - `Rectangle` — hyperrectangle utility used by SciPy spatial search
//! - `HalfspaceIntersection` — N-D halfspace intersection
//! - `distance` — pairwise distance computations

use std::collections::BTreeMap;

use fsci_runtime::casp_now_unix_ms;
pub use fsci_runtime::{
    AuditAction, AuditEvent, AuditLedger, HARDENED_MAX_DIM, RuntimeMode, SyncSharedAuditLedger,
};

/// Create a new shared audit ledger for synchronous contexts.
#[must_use]
pub fn sync_audit_ledger() -> SyncSharedAuditLedger {
    AuditLedger::shared()
}

fn lock_or_recover(ledger: &SyncSharedAuditLedger) -> std::sync::MutexGuard<'_, AuditLedger> {
    match ledger.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            ledger.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Record a fail-closed audit event when Hardened mode rejects input.
pub fn record_fail_closed(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    reason: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::FailClosed {
            reason: reason.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

/// Record a bounded-recovery audit event when Hardened mode falls back.
pub fn record_bounded_recovery(
    ledger: &SyncSharedAuditLedger,
    input_bytes: &[u8],
    recovery_action: &str,
    outcome: &str,
) {
    let event = AuditEvent::new(
        casp_now_unix_ms(),
        AuditLedger::fingerprint_bytes(input_bytes),
        AuditAction::BoundedRecovery {
            recovery_action: recovery_action.to_string(),
        },
        outcome.to_string(),
    );
    lock_or_recover(ledger).record(event);
}

use fsci_linalg::{DecompOptions, svd};
use fsci_sparse::{CooMatrix, DokMatrix, Shape2D};

/// Error raised by Qhull-backed spatial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QhullError {
    message: String,
}

impl QhullError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for QhullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for QhullError {}

/// Error type for spatial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialError {
    EmptyData,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidArgument(String),
    Qhull(QhullError),
}

impl std::fmt::Display for SpatialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyData => write!(f, "empty data"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidArgument(msg) => write!(f, "{msg}"),
            Self::Qhull(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for SpatialError {}

impl From<QhullError> for SpatialError {
    fn from(value: QhullError) -> Self {
        Self::Qhull(value)
    }
}

fn sparse_error_to_spatial(err: impl std::fmt::Display) -> SpatialError {
    SpatialError::InvalidArgument(err.to_string())
}

// ══════════════════════════════════════════════════════════════════════
// Distance Functions
// ══════════════════════════════════════════════════════════════════════

/// Euclidean distance between two points.
pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    sqeuclidean(a, b).sqrt()
}

/// Squared Euclidean distance (avoids sqrt for comparisons). The reduction
/// `Σ (aᵢ-bᵢ)²` runs as two 8-wide SIMD accumulators (the scalar `.sum()` fold is
/// not auto-vectorized), with a scalar tail — the dominant inner kernel of
/// `cdist`/`pdist`/nearest-neighbour searches.
pub fn sqeuclidean(a: &[f64], b: &[f64]) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut acc0 = Simd::<f64, L>::splat(0.0);
    let mut acc1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= n {
        let d0 = Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L]);
        let d1 = Simd::<f64, L>::from_slice(&a[i + L..i + 2 * L])
            - Simd::from_slice(&b[i + L..i + 2 * L]);
        acc0 += d0 * d0;
        acc1 += d1 * d1;
        i += 2 * L;
    }
    if i + L <= n {
        let d = Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L]);
        acc0 += d * d;
        i += L;
    }
    let mut s = (acc0 + acc1).reduce_sum();
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

/// 8-wide `Σ a[i]·b[i]` over `min(a,b)` length (two accumulators + scalar tail). The
/// scalar `.map().sum()` fold is not auto-vectorized.
#[inline]
fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut acc0 = Simd::<f64, L>::splat(0.0);
    let mut acc1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= n {
        acc0 += Simd::<f64, L>::from_slice(&a[i..i + L]) * Simd::from_slice(&b[i..i + L]);
        acc1 += Simd::<f64, L>::from_slice(&a[i + L..i + 2 * L])
            * Simd::from_slice(&b[i + L..i + 2 * L]);
        i += 2 * L;
    }
    if i + L <= n {
        acc0 += Simd::<f64, L>::from_slice(&a[i..i + L]) * Simd::from_slice(&b[i..i + L]);
        i += L;
    }
    let mut s = (acc0 + acc1).reduce_sum();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Fused 8-wide `(Σ a·b, Σ a², Σ b²)` in ONE pass — byte-identical to
/// `(simd_dot(a,b), simd_sqsum(a), simd_sqsum(b))` when `a.len() == b.len()`: each of the six
/// accumulators receives the exact same SIMD-lane operation sequence, the same two-accumulator
/// layout, the same `(acc0+acc1).reduce_sum()` reduction and the same scalar tail as its standalone
/// helper. Reading `a`/`b` once each instead of twice halves the memory traffic. The caller only
/// takes this path for equal-length inputs (the helpers iterate different ranges otherwise).
#[inline]
fn fused_dot_sqsum(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut d0 = Simd::<f64, L>::splat(0.0);
    let mut d1 = Simd::<f64, L>::splat(0.0);
    let mut sa0 = Simd::<f64, L>::splat(0.0);
    let mut sa1 = Simd::<f64, L>::splat(0.0);
    let mut sb0 = Simd::<f64, L>::splat(0.0);
    let mut sb1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= n {
        let av0 = Simd::<f64, L>::from_slice(&a[i..i + L]);
        let bv0 = Simd::<f64, L>::from_slice(&b[i..i + L]);
        let av1 = Simd::<f64, L>::from_slice(&a[i + L..i + 2 * L]);
        let bv1 = Simd::<f64, L>::from_slice(&b[i + L..i + 2 * L]);
        d0 += av0 * bv0;
        d1 += av1 * bv1;
        sa0 += av0 * av0;
        sa1 += av1 * av1;
        sb0 += bv0 * bv0;
        sb1 += bv1 * bv1;
        i += 2 * L;
    }
    if i + L <= n {
        let av = Simd::<f64, L>::from_slice(&a[i..i + L]);
        let bv = Simd::<f64, L>::from_slice(&b[i..i + L]);
        d0 += av * bv;
        sa0 += av * av;
        sb0 += bv * bv;
        i += L;
    }
    let mut dot = (d0 + d1).reduce_sum();
    let mut sa = (sa0 + sa1).reduce_sum();
    let mut sb = (sb0 + sb1).reduce_sum();
    while i < n {
        dot += a[i] * b[i];
        sa += a[i] * a[i];
        sb += b[i] * b[i];
        i += 1;
    }
    (dot, sa, sb)
}

/// 8-wide `Σ x[i]²` (two accumulators + scalar tail).
#[inline]
fn simd_sqsum(x: &[f64]) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let mut acc0 = Simd::<f64, L>::splat(0.0);
    let mut acc1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= x.len() {
        let v0 = Simd::<f64, L>::from_slice(&x[i..i + L]);
        let v1 = Simd::<f64, L>::from_slice(&x[i + L..i + 2 * L]);
        acc0 += v0 * v0;
        acc1 += v1 * v1;
        i += 2 * L;
    }
    if i + L <= x.len() {
        let v = Simd::<f64, L>::from_slice(&x[i..i + L]);
        acc0 += v * v;
        i += L;
    }
    let mut s = (acc0 + acc1).reduce_sum();
    while i < x.len() {
        s += x[i] * x[i];
        i += 1;
    }
    s
}

#[inline]
fn sqeuclidean4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    let d3 = a[3] - b[3];
    d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
}

#[inline]
fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

#[inline]
fn sqsum4(x: &[f64; 4]) -> f64 {
    x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
}

#[inline]
fn chebyshev4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let mut max = 0.0_f64;
    let d0 = (a[0] - b[0]).abs();
    if d0.is_nan() {
        return f64::NAN;
    }
    max = max.max(d0);
    let d1 = (a[1] - b[1]).abs();
    if d1.is_nan() {
        return f64::NAN;
    }
    max = max.max(d1);
    let d2 = (a[2] - b[2]).abs();
    if d2.is_nan() {
        return f64::NAN;
    }
    max = max.max(d2);
    let d3 = (a[3] - b[3]).abs();
    if d3.is_nan() {
        return f64::NAN;
    }
    max.max(d3)
}

fn collect_dim4_points(x: &[Vec<f64>]) -> Vec<[f64; 4]> {
    x.iter()
        .map(|row| [row[0], row[1], row[2], row[3]])
        .collect()
}

/// Manhattan (L1) distance: `Σ |a[i]-b[i]|`, 8-wide.
pub fn cityblock(a: &[f64], b: &[f64]) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut acc0 = Simd::<f64, L>::splat(0.0);
    let mut acc1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= n {
        acc0 += (Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L])).abs();
        acc1 += (Simd::<f64, L>::from_slice(&a[i + L..i + 2 * L])
            - Simd::from_slice(&b[i + L..i + 2 * L]))
        .abs();
        i += 2 * L;
    }
    if i + L <= n {
        acc0 += (Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L])).abs();
        i += L;
    }
    let mut s = (acc0 + acc1).reduce_sum();
    while i < n {
        s += (a[i] - b[i]).abs();
        i += 1;
    }
    s
}

/// Chebyshev (L∞) distance.
pub fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    use std::simd::{Select, Simd, cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut vmax = Simd::<f64, L>::splat(0.0);
    let mut nan_mask = vmax.simd_ne(vmax);
    let mut i = 0usize;
    while i + L <= n {
        let d = (Simd::<f64, L>::from_slice(&a[i..i + L])
            - Simd::<f64, L>::from_slice(&b[i..i + L]))
        .abs();
        nan_mask |= d.simd_ne(d);
        vmax = vmax.simd_gt(d).select(vmax, d);
        i += L;
    }
    if nan_mask.any() {
        return f64::NAN;
    }
    let mut max = vmax.reduce_max();
    while i < n {
        let d = (a[i] - b[i]).abs();
        if d.is_nan() {
            return f64::NAN;
        }
        max = max.max(d);
        i += 1;
    }
    max
}

/// When `true`, [`cosine`] computes its dot product and two squared-norms as three separate SIMD
/// passes (the ORIG behaviour); default `false` fuses them into one pass. Byte-identical.
#[doc(hidden)]
pub static COSINE_FUSE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Cosine distance: 1 - cosine_similarity(a, b).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    // Fuse the dot product and the two squared-norms into ONE pass over (a, b) — byte-identical to
    // the three separate SIMD helpers (see `fused_dot_sqsum`) but reading each vector once instead
    // of twice. Only for equal-length inputs (the helpers cover different ranges otherwise).
    let (dot, sqsum_a, sqsum_b) =
        if COSINE_FUSE_DISABLE.load(std::sync::atomic::Ordering::Relaxed) || a.len() != b.len() {
            (simd_dot(a, b), simd_sqsum(a), simd_sqsum(b))
        } else {
            fused_dot_sqsum(a, b)
        };
    let norm_a = sqsum_a.sqrt();
    let norm_b = sqsum_b.sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        // scipy returns NaN when either vector has zero norm: the cosine
        // similarity 0/0 is undefined, so 1 - (0/0) propagates to NaN.
        return f64::NAN;
    }
    1.0 - dot / denom
}

/// Minkowski distance of order `p`.
/// `Σ |a-b|^p` for a small INTEGER exponent `p`. `|d|^p` is computed by repeated
/// multiplication (≈p mults) instead of a ≈80-cycle `powf`, 8-wide SIMD with two
/// accumulators + scalar tail. scipy uses `pow()` even for integer p, so this is a large
/// win for the common `p = 2, 3, 4, …` Minkowski case. Reassociated (~1e-14) vs the scalar
/// `powf` fold, within the distance tolerance.
#[inline]
fn minkowski_pow_sum(a: &[f64], b: &[f64], p: u32) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let mut acc0 = Simd::<f64, L>::splat(0.0);
    let mut acc1 = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + 2 * L <= n {
        let d0 = (Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L])).abs();
        let d1 = (Simd::<f64, L>::from_slice(&a[i + L..i + 2 * L])
            - Simd::from_slice(&b[i + L..i + 2 * L]))
        .abs();
        let (mut t0, mut t1) = (d0, d1);
        for _ in 1..p {
            t0 *= d0;
            t1 *= d1;
        }
        acc0 += t0;
        acc1 += t1;
        i += 2 * L;
    }
    if i + L <= n {
        let d = (Simd::<f64, L>::from_slice(&a[i..i + L]) - Simd::from_slice(&b[i..i + L])).abs();
        let mut t = d;
        for _ in 1..p {
            t *= d;
        }
        acc0 += t;
        i += L;
    }
    let mut s = (acc0 + acc1).reduce_sum();
    while i < n {
        let d = (a[i] - b[i]).abs();
        let mut t = d;
        for _ in 1..p {
            t *= d;
        }
        s += t;
        i += 1;
    }
    s
}

/// `(Σ |a-b|^p)^(1/p)` for a small integer `p` (see [`minkowski_pow_sum`]).
#[inline]
fn minkowski_int(a: &[f64], b: &[f64], p: u32) -> f64 {
    minkowski_pow_sum(a, b, p).powf(1.0 / p as f64)
}

pub fn minkowski(a: &[f64], b: &[f64], p: f64) -> f64 {
    if p <= 0.0 || p.is_nan() {
        return f64::NAN;
    }
    if p == f64::INFINITY {
        return chebyshev(a, b);
    }
    if p == 1.0 {
        return cityblock(a, b);
    }
    if p == 2.0 {
        return euclidean(a, b);
    }
    // Small integer exponents: repeated multiplication + SIMD instead of per-element powf.
    let pr = p.round();
    if pr == p && (3.0..=64.0).contains(&pr) {
        return minkowski_int(a, b, pr as u32);
    }
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Correlation distance: 1 - pearson_r(a, b).
///
/// Treats a and b as samples and computes 1 minus the Pearson correlation.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if a.len() < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let mut ssab = 0.0;
    let mut ssa = 0.0;
    let mut ssb = 0.0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        ssab += da * db;
        ssa += da * da;
        ssb += db * db;
    }
    let denom = (ssa * ssb).sqrt();
    if denom == 0.0 {
        // scipy returns NaN when either vector is constant (zero variance):
        // the Pearson correlation 0/0 is undefined, so 1 - (0/0) is NaN.
        return f64::NAN;
    }
    1.0 - ssab / denom
}

/// Hamming distance: fraction of elements that differ.
///
/// Matches `scipy.spatial.distance.hamming(u, v)`.
/// For real-valued vectors, counts positions where u_i != v_i.
pub fn hamming(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mismatches = a
        .iter()
        .zip(b.iter())
        .filter(|&(&ai, &bi)| ai != bi)
        .count();
    mismatches as f64 / a.len() as f64
}

/// Jaccard dissimilarity.
///
/// Matches `scipy.spatial.distance.jaccard(u, v)` (scipy ≥1.15
/// semantics: numeric input is converted to Boolean before
/// computation). The result is (c_TF + c_FT) / (c_TT + c_TF + c_FT)
/// where positions are classified by `(ai != 0.0, bi != 0.0)`.
/// If all positions are (False, False), returns 0.
///
/// frankenscipy-z747j: prior to this revision the impl counted
/// real-valued inequality between nonzero positions, which matched
/// scipy ≤1.14 but diverged from scipy ≥1.15 (per scipy 1.15
/// changelog: "Non-0/1 numeric input used to produce an ad hoc
/// result. Since 1.15.0, numeric input is converted to Boolean
/// before computation.")
pub fn jaccard(a: &[f64], b: &[f64]) -> f64 {
    let mut nonzero = 0usize;
    let mut unequal_nonzero = 0usize;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let a_b = ai != 0.0;
        let b_b = bi != 0.0;
        if a_b || b_b {
            nonzero += 1;
            if a_b != b_b {
                unequal_nonzero += 1;
            }
        }
    }
    if nonzero == 0 {
        return 0.0;
    }
    unequal_nonzero as f64 / nonzero as f64
}

/// Canberra distance.
///
/// Matches `scipy.spatial.distance.canberra(u, v)`.
/// d(u,v) = Σ|u_i - v_i| / (|u_i| + |v_i|)
pub fn canberra(a: &[f64], b: &[f64]) -> f64 {
    // Σ |ai-bi| / (|ai|+|bi|), term=0 when denom==0. The per-element divide makes this
    // compute-bound, so 8-wide SIMD wins ~2x. `simd_eq(0)` reproduces the denom==0⇒0 guard
    // via a lane mask (denom==0 ⇒ both 0 ⇒ 0/0=NaN, masked to 0 before the add).
    use std::simd::{Select, Simd, cmp::SimdPartialEq, num::SimdFloat};
    const L: usize = 8;
    let n = a.len().min(b.len());
    let zero = Simd::<f64, L>::splat(0.0);
    let mut acc = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + L <= n {
        let av = Simd::<f64, L>::from_slice(&a[i..i + L]);
        let bv = Simd::<f64, L>::from_slice(&b[i..i + L]);
        let denom = av.abs() + bv.abs();
        let term = (av - bv).abs() / denom;
        acc += denom.simd_eq(zero).select(zero, term);
        i += L;
    }
    let mut s = acc.reduce_sum();
    while i < n {
        let (ai, bi) = (a[i], b[i]);
        let denom = ai.abs() + bi.abs();
        if denom != 0.0 {
            s += (ai - bi).abs() / denom;
        }
        i += 1;
    }
    s
}

/// When `true`, [`braycurtis`] computes its numerator `Σ|a-b|` and denominator `Σ|a+b|` in two
/// separate passes (the ORIG behaviour); default `false` computes both in one pass. Byte-identical.
#[doc(hidden)]
pub static BRAYCURTIS_FUSE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Bray-Curtis dissimilarity.
///
/// Matches `scipy.spatial.distance.braycurtis(u, v)`.
/// d(u,v) = Σ|u_i - v_i| / Σ|u_i + v_i|
pub fn braycurtis(a: &[f64], b: &[f64]) -> f64 {
    // The numerator `Σ|a-b|` and denominator `Σ|a+b|` are two independent reductions over the same
    // (a, b) — accumulate both in ONE pass so each vector is read once instead of twice.
    // BYTE-IDENTICAL: each Σ stays a scalar left-to-right `+=` from 0.0 in the same index order as
    // the original `iter().map().sum()` (kept scalar so no SIMD reassociation).
    let (num, den) = if BRAYCURTIS_FUSE_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
        let num: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).abs())
            .sum();
        let den: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai + bi).abs())
            .sum();
        (num, den)
    } else {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            num += (ai - bi).abs();
            den += (ai + bi).abs();
        }
        (num, den)
    };
    if den == 0.0 { 0.0 } else { num / den }
}

fn relative_entropy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * (x / y).ln() }
}

/// Jensen-Shannon distance between two probability vectors.
///
/// Matches `scipy.spatial.distance.jensenshannon(p, q, base=None)` for
/// one-dimensional inputs. The input vectors are normalized to sum to 1.
pub fn jensenshannon(p: &[f64], q: &[f64], base: Option<f64>) -> f64 {
    if p.is_empty() || q.is_empty() || p.len() != q.len() {
        return f64::NAN;
    }
    let sum_p: f64 = p.iter().sum();
    let sum_q: f64 = q.iter().sum();
    if sum_p <= 0.0 || sum_q <= 0.0 || !sum_p.is_finite() || !sum_q.is_finite() {
        return f64::NAN;
    }

    // Keep scalar accumulation order while vectorizing the per-lane normalization
    // and logarithms. This preserves the old normalized-Vec result bits.
    use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialEq};
    const L: usize = 8;
    let n = p.len();
    let sp = Simd::<f64, L>::splat(sum_p);
    let sq = Simd::<f64, L>::splat(sum_q);
    let zero = Simd::<f64, L>::splat(0.0);
    let two = Simd::<f64, L>::splat(2.0);
    let mut js = 0.0;
    let mut i = 0;
    while i + L <= n {
        let left = Simd::<f64, L>::from_slice(&p[i..i + L]) / sp;
        let right = Simd::<f64, L>::from_slice(&q[i..i + L]) / sq;
        let mean = (left + right) / two;
        let tl = left.simd_eq(zero).select(zero, left * (left / mean).ln());
        let tr = right
            .simd_eq(zero)
            .select(zero, right * (right / mean).ln());
        for term in (tl + tr).to_array() {
            js += term;
        }
        i += L;
    }
    while i < n {
        let left = p[i] / sum_p;
        let right = q[i] / sum_q;
        let mean = (left + right) / 2.0;
        js += relative_entropy(left, mean) + relative_entropy(right, mean);
        i += 1;
    }

    let scaled = match base {
        Some(log_base) => js / log_base.ln(),
        None => js,
    };
    (scaled / 2.0).sqrt()
}

/// Distance metric identifiers for use with `pdist` and `cdist_metric`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    SqEuclidean,
    Cityblock,
    Chebyshev,
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    Canberra,
    Braycurtis,
}

/// Compute the distance between two points using the specified metric.
pub fn metric_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean(a, b),
        DistanceMetric::SqEuclidean => sqeuclidean(a, b),
        DistanceMetric::Cityblock => cityblock(a, b),
        DistanceMetric::Chebyshev => chebyshev(a, b),
        DistanceMetric::Cosine => cosine(a, b),
        DistanceMetric::Correlation => correlation(a, b),
        DistanceMetric::Hamming => hamming(a, b),
        DistanceMetric::Jaccard => jaccard(a, b),
        DistanceMetric::Canberra => canberra(a, b),
        DistanceMetric::Braycurtis => braycurtis(a, b),
    }
}

/// Compute pairwise distances between observations in condensed form.
///
/// Matches `scipy.spatial.distance.pdist(X, metric)`.
///
/// Returns a condensed distance vector of length n*(n-1)/2, containing
/// the upper-triangular entries in row order.
/// Condensed pairwise Minkowski (L^p) distances within `x`, matching
/// `scipy.spatial.distance.pdist(X, 'minkowski', p=p)`. Parallel via `pdist_fill` + tested scalar
/// [`minkowski`]. (Not in `DistanceMetric` — that enum derives `Eq` and p is f64.)
pub fn pdist_minkowski(x: &[Vec<f64>], p: f64) -> Result<Vec<f64>, SpatialError> {
    let n = x.len();
    if n == 0 {
        return Err(SpatialError::EmptyData);
    }
    if p.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(SpatialError::InvalidArgument(
            "minkowski p must be > 0".to_string(),
        ));
    }
    let dim = x[0].len();
    for row in x.iter() {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    let total = n * (n - 1) / 2;
    let nthreads = pdist_thread_count(n, dim);
    Ok(pdist_fill(n, total, nthreads, |i, j| {
        minkowski(&x[i], &x[j], p)
    }))
}

/// Condensed pairwise standardized-Euclidean distances within `x`, matching
/// `scipy.spatial.distance.pdist(X, 'seuclidean', V=v)`. Parallel via `pdist_fill` + tested scalar
/// [`seuclidean`].
pub fn pdist_seuclidean(x: &[Vec<f64>], v: &[f64]) -> Result<Vec<f64>, SpatialError> {
    let n = x.len();
    if n == 0 {
        return Err(SpatialError::EmptyData);
    }
    let dim = x[0].len();
    if v.len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: v.len(),
        });
    }
    for row in x.iter() {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    // Whitened-Euclidean fast path (see `cdist_seuclidean`): scale by 1/√vₖ once, then it's a
    // plain Euclidean pdist with no per-pair divide/`v≤0` mask. ~1e-15 agreement. dim==4 also
    // picks up pdist(Euclidean)'s dedicated dim-4 SoA kernel (measured ~7x); 5≤dim≤7 is a wash
    // (measured d6 regressed), so gate on the two proven win regions.
    if (dim == 4 || dim >= 8)
        && !SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
        && v.iter().all(|&vk| vk > 0.0)
    {
        let inv_sqrt: Vec<f64> = v.iter().map(|&vk| 1.0 / vk.sqrt()).collect();
        let scaled: Vec<Vec<f64>> = x
            .iter()
            .map(|r| r.iter().zip(&inv_sqrt).map(|(&xi, &s)| xi * s).collect())
            .collect();
        return pdist(&scaled, DistanceMetric::Euclidean);
    }
    let total = n * (n - 1) / 2;
    let nthreads = pdist_thread_count(n, dim);
    Ok(pdist_fill(n, total, nthreads, |i, j| {
        seuclidean(&x[i], &x[j], v)
    }))
}

pub fn pdist(x: &[Vec<f64>], metric: DistanceMetric) -> Result<Vec<f64>, SpatialError> {
    let n = x.len();
    if n < 2 {
        return Err(SpatialError::InvalidArgument(
            "pdist requires at least 2 observations".to_string(),
        ));
    }
    let dim = x[0].len();
    for row in x.iter() {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }

    let total = n * (n - 1) / 2;
    // The vectorized dim-4 Euclidean/Cosine serial kernel (SIMD across pairs) beats SciPy
    // ~1.5-2.2x and stays compute-bound up to n≈2048; below that crossover the parallel
    // path loses to its own thread-spawn cost (e.g. n=1024: serial 0.67ms vs spawning ~64
    // threads 2.8ms). Above it the O(n²) memory-bound work amortizes the spawn and the
    // parallel fill wins (n=4096: 13ms vs 43ms serial). Measured same-box, see nm8ex.
    let nthreads = if dim == 4
        && n <= 2048
        && matches!(
            metric,
            DistanceMetric::Euclidean
                | DistanceMetric::Cosine
                | DistanceMetric::SqEuclidean
                | DistanceMetric::Cityblock
                | DistanceMetric::Chebyshev
        ) {
        1
    } else {
        pdist_thread_count(n, dim)
    };

    // Cosine and Correlation recompute each vector's norm/mean+centered-norm — quantities
    // that depend on ONE vector — inside the O(n²) pair loop. Precompute them ONCE (O(n·d))
    // so each pair does only the cross term (a single reduction). Byte-identical: the
    // precomputed per-vector quantities use the same arithmetic/summation order as the
    // scalar `cosine`/`correlation` helpers, and the cross term is summed in the same order.
    let result = match metric {
        DistanceMetric::Euclidean if dim == 4 => {
            let points = collect_dim4_points(x);
            pdist_fill_euclidean4(&points, n, total, nthreads)
        }
        DistanceMetric::SqEuclidean if dim == 4 => {
            let points = collect_dim4_points(x);
            pdist_fill_dim4(&points, n, total, nthreads, fill_sqeuclidean4_rows)
        }
        DistanceMetric::Cityblock if dim == 4 => {
            let points = collect_dim4_points(x);
            pdist_fill_dim4(&points, n, total, nthreads, fill_cityblock4_rows)
        }
        DistanceMetric::Chebyshev if dim == 4 => {
            let points = collect_dim4_points(x);
            pdist_fill_dim4(&points, n, total, nthreads, fill_chebyshev4_rows)
        }
        DistanceMetric::Cosine if dim == 4 => {
            let points = collect_dim4_points(x);
            let norms: Vec<f64> = points.iter().map(|v| sqsum4(v).sqrt()).collect();
            pdist_fill_cosine4(&points, &norms, n, total, nthreads)
        }
        // Every width below 16 EXCEPT 4, which keeps its own hand-tuned kernel (that one
        // also carries a measured serial/parallel crossover this general path has not been
        // swept for).
        DistanceMetric::Euclidean
            if dim != 4
                && dim > 0
                && dim < PDIST_SOA_MAX_DIM
                && PDIST_SOA_EUCLIDEAN.load(std::sync::atomic::Ordering::Relaxed) =>
        {
            let mut flat = Vec::with_capacity(n * dim);
            for row in x {
                flat.extend_from_slice(row);
            }
            let soa = pdist_soa_columns(x, n, dim);
            pdist_fill_euclidean_soa(&soa, &flat, n, dim, total, nthreads)
        }
        DistanceMetric::Cosine => {
            let norms: Vec<f64> = x.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            pdist_fill(n, total, nthreads, |i, j| {
                let denom = norms[i] * norms[j];
                if denom == 0.0 {
                    f64::NAN
                } else {
                    1.0 - simd_dot(&x[i], &x[j]) / denom
                }
            })
        }
        DistanceMetric::Correlation if dim >= 2 => {
            let dn = dim as f64;
            // Per vector: centered values c = x − mean, and ssa = Σ c². Same fused k-order
            // as `correlation`, so the values are bit-identical.
            let prep: Vec<(Vec<f64>, f64)> = x
                .iter()
                .map(|v| {
                    let mean = v.iter().sum::<f64>() / dn;
                    let c: Vec<f64> = v.iter().map(|&xi| xi - mean).collect();
                    let ssa: f64 = c.iter().map(|&ci| ci * ci).sum();
                    (c, ssa)
                })
                .collect();
            pdist_fill(n, total, nthreads, |i, j| {
                let (ci, ssa) = &prep[i];
                let (cj, ssb) = &prep[j];
                let ssab: f64 = ci.iter().zip(cj.iter()).map(|(&p, &q)| p * q).sum();
                let denom = (ssa * ssb).sqrt();
                if denom == 0.0 {
                    f64::NAN
                } else {
                    1.0 - ssab / denom
                }
            })
        }
        // Jaccard over the condensed upper triangle: pack rows to bits ONCE, then each of
        // the n(n-1)/2 pairs is an O(d/64) popcount (byte-identical, any numeric input).
        // Same gate/lever as cdist; the packing amortizes even better here (more pairs/row).
        DistanceMetric::Jaccard if spatial_bool_popcount() && dim >= JACCARD_POPCOUNT_MIN_DIM => {
            let (bits, ones) = cdist_pack_bits(x, dim);
            pdist_fill(n, total, nthreads, |i, j| {
                jaccard_bits_pair(&bits[i], ones[i], &bits[j], ones[j])
            })
        }
        _ => {
            // AoS rows (`&[Vec<f64>]`) force the O(n²) inner loop to chase one heap
            // pointer per row; flatten into a single contiguous row-major buffer so
            // consecutive j-rows stream through cache (hardware prefetch) instead of
            // hitting scattered allocations. Byte-identical: `metric_distance` reads
            // the same values in the same order, just from a contiguous slice.
            let mut flat = Vec::with_capacity(n * dim);
            for row in x {
                flat.extend_from_slice(row);
            }
            pdist_fill(n, total, nthreads, |i, j| {
                metric_distance(
                    &flat[i * dim..i * dim + dim],
                    &flat[j * dim..j * dim + dim],
                    metric,
                )
            })
        }
    };
    Ok(result)
}

/// Transpose dim-4 AoS points into 4 contiguous coordinate columns (SoA). Lets the
/// all-pairs loop load `L` consecutive `j`-points per coordinate with one aligned SIMD
/// gather, so each output element of a SIMD chunk is a *different* pair — the dependent
/// per-pair `sqrt`/`div` then pipeline across lanes instead of stalling one at a time.
/// Largest `dim` for which `sqeuclidean`'s reduction is plainly ascending in `d`, so an
/// across-pairs kernel accumulating ascending is BIT-identical to it.
///
/// `sqeuclidean` runs two 8-wide accumulators and finishes with an ORDERED horizontal
/// reduce. Below 16 dimensions only `acc0` is ever written, so the reduce is
/// `0.0 + d₀² + … + d₇²` and any scalar tail continues that same ascending chain. At
/// `dim == 16` the second accumulator becomes live and `(acc0 + acc1)` pairs lane `k` with
/// lane `k + 8` BEFORE the reduce — a genuinely different summation order that an ascending
/// accumulation cannot reproduce. So this bound is a property of the reference reduction,
/// not a convenience: widening it would silently change results rather than speed them up.
const PDIST_SOA_MAX_DIM: usize = 16;

/// Route `pdist`'s Euclidean fill through the general across-pairs SoA kernel.
///
/// Bit-identical to the `metric_distance` scan it replaces, so this is a cost knob only.
pub static PDIST_SOA_EUCLIDEAN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Transpose points into flat column-major coordinates: `soa[d * n + i]` is coordinate `d`
/// of point `i`, so a SIMD load pulls the SAME coordinate from `L` CONSECUTIVE points.
fn pdist_soa_columns(x: &[Vec<f64>], n: usize, dim: usize) -> Vec<f64> {
    let mut soa = vec![0.0_f64; n * dim];
    for (i, row) in x.iter().enumerate() {
        for d in 0..dim {
            soa[d * n + i] = row[d];
        }
    }
    soa
}

/// SIMD-ACROSS-PAIRS Euclidean fill for rows `r0..r1`, for any `dim < PDIST_SOA_MAX_DIM`.
///
/// WHY. The generic path calls `euclidean` once per pair, and `sqeuclidean` inside it is
/// SIMD WITHIN one vector: it produces `dim` squared differences in lanes and then collapses
/// them with an ORDERED horizontal reduce, which is a chain of dependent adds, followed by a
/// dependent scalar `sqrt`. At `dim = 8` that is ~8 serial adds plus a `sqrt` for ONE pair,
/// and `pdist` at n=2000 has 2M pairs. The lanes are being spent on the one axis that has to
/// be summed away again.
///
/// Different PAIRS are independent, so lane `k` here holds pair `(i, start + j + k)` and the
/// `d` loop walks dimensions on the OUTSIDE. Nothing is ever reduced horizontally: the `d`
/// accumulation is `L` independent chains, and one `sqrt` covers `L` pairs. This is the same
/// shape the `dim == 4` kernel already uses; it was simply never written for other widths.
///
/// BIT-IDENTICAL, see `PDIST_SOA_MAX_DIM`: each lane accumulates `0.0 + d₀² + … ` ascending,
/// exactly the chain `sqeuclidean` produces below 16 dimensions, and `Simd::sqrt` is the same
/// correctly-rounded IEEE operation as `f64::sqrt`. The scalar tail defers to `euclidean`
/// itself rather than reimplementing it.
fn fill_euclidean_soa_rows(
    soa: &[f64],
    flat: &[f64],
    n: usize,
    dim: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::{Simd, StdFloat};
    const L: usize = 8;
    let mut pos = 0usize;
    for i in r0..r1 {
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let mut acc = Simd::<f64, L>::splat(0.0);
            for d in 0..dim {
                let column = &soa[d * n..d * n + n];
                let a = Simd::<f64, L>::splat(column[i]);
                let b = Simd::<f64, L>::from_slice(&column[s..s + L]);
                let diff = a - b;
                acc += diff * diff;
            }
            acc.sqrt().copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            let s = start + j;
            seg[pos + j] = euclidean(&flat[i * dim..i * dim + dim], &flat[s * dim..s * dim + dim]);
            j += 1;
        }
        pos += row;
    }
}

/// Threaded driver for `fill_euclidean_soa_rows`, splitting the condensed output into
/// disjoint row-aligned segments exactly as the `dim == 4` path does.
fn pdist_fill_euclidean_soa(
    soa: &[f64],
    flat: &[f64],
    n: usize,
    dim: usize,
    total: usize,
    nthreads: usize,
) -> Vec<f64> {
    let mut result = vec![0.0_f64; total];
    if nthreads <= 1 {
        fill_euclidean_soa_rows(soa, flat, n, dim, 0, n, &mut result);
        return result;
    }
    let bounds = pdist_row_bounds(n, nthreads);
    let offset = |r: usize| -> usize { r * (n - 1) - r * (r.saturating_sub(1)) / 2 };
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut result;
        let mut prev = 0usize;
        for w in 0..bounds.len() - 1 {
            let r0 = bounds[w];
            let r1 = bounds[w + 1];
            let take = offset(r1) - prev;
            prev = offset(r1);
            let (seg, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || fill_euclidean_soa_rows(soa, flat, n, dim, r0, r1, seg));
        }
    });
    result
}

fn dim4_soa(x: &[[f64; 4]]) -> [Vec<f64>; 4] {
    let n = x.len();
    let mut c = [
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
    ];
    for (idx, p) in x.iter().enumerate() {
        c[0][idx] = p[0];
        c[1][idx] = p[1];
        c[2][idx] = p[2];
        c[3][idx] = p[3];
    }
    c
}

/// SIMD-across-pairs fill of the Euclidean dim-4 condensed distances for rows `r0..r1`
/// into `seg` (exactly the pairs of those rows, contiguous, row r0 first). Lane k of a
/// chunk holds pair (i, start+j+k); per lane the squared sum d0²+d1²+d2²+d3² and its sqrt
/// run in the same left-to-right order as scalar `sqeuclidean4(..).sqrt()`, so output is
/// BIT-identical while the 8-wide `vsqrtpd` chunk pipelines the otherwise-serial sqrts.
fn fill_euclidean4_rows(
    x: &[[f64; 4]],
    c: &[Vec<f64>; 4],
    n: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::{Simd, StdFloat};
    const L: usize = 8;
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    let mut pos = 0usize;
    for i in r0..r1 {
        let a0 = Simd::<f64, L>::splat(c0[i]);
        let a1 = Simd::<f64, L>::splat(c1[i]);
        let a2 = Simd::<f64, L>::splat(c2[i]);
        let a3 = Simd::<f64, L>::splat(c3[i]);
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let d0 = a0 - Simd::<f64, L>::from_slice(&c0[s..s + L]);
            let d1 = a1 - Simd::<f64, L>::from_slice(&c1[s..s + L]);
            let d2 = a2 - Simd::<f64, L>::from_slice(&c2[s..s + L]);
            let d3 = a3 - Simd::<f64, L>::from_slice(&c3[s..s + L]);
            let sq = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            sq.sqrt().copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            seg[pos + j] = sqeuclidean4(&x[i], &x[start + j]).sqrt();
            j += 1;
        }
        pos += row;
    }
}

fn pdist_fill_euclidean4(x: &[[f64; 4]], n: usize, total: usize, nthreads: usize) -> Vec<f64> {
    let c = dim4_soa(x);
    let mut result = vec![0.0_f64; total];
    if nthreads <= 1 {
        fill_euclidean4_rows(x, &c, n, 0, n, &mut result);
        return result;
    }
    let bounds = pdist_row_bounds(n, nthreads);
    let offset = |r: usize| -> usize { r * (n - 1) - r * (r.saturating_sub(1)) / 2 };
    let c = &c;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut result;
        let mut prev = 0usize;
        for w in 0..bounds.len() - 1 {
            let r0 = bounds[w];
            let r1 = bounds[w + 1];
            let take = offset(r1) - prev;
            prev = offset(r1);
            let (seg, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || fill_euclidean4_rows(x, c, n, r0, r1, seg));
        }
    });
    result
}

/// SIMD-across-pairs fill of the Cosine dim-4 condensed distances for rows `r0..r1` into
/// `seg`. Lane k holds pair (i, start+j+k); per lane `1 - dot/(ni·nj)` and the
/// denom==0 ⇒ NaN guard run identically to the scalar form, so output is BIT-identical
/// while the 8-wide `vdivpd` chunk pipelines the dependent divisions (cosine bottleneck).
fn fill_cosine4_rows(
    x: &[[f64; 4]],
    c: &[Vec<f64>; 4],
    norms: &[f64],
    n: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::{Select, Simd, cmp::SimdPartialEq};
    const L: usize = 8;
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let nan = Simd::<f64, L>::splat(f64::NAN);
    let mut pos = 0usize;
    for i in r0..r1 {
        let a0 = Simd::<f64, L>::splat(c0[i]);
        let a1 = Simd::<f64, L>::splat(c1[i]);
        let a2 = Simd::<f64, L>::splat(c2[i]);
        let a3 = Simd::<f64, L>::splat(c3[i]);
        let ni = Simd::<f64, L>::splat(norms[i]);
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let b0 = Simd::<f64, L>::from_slice(&c0[s..s + L]);
            let b1 = Simd::<f64, L>::from_slice(&c1[s..s + L]);
            let b2 = Simd::<f64, L>::from_slice(&c2[s..s + L]);
            let b3 = Simd::<f64, L>::from_slice(&c3[s..s + L]);
            let nj = Simd::<f64, L>::from_slice(&norms[s..s + L]);
            let denom = ni * nj;
            let dot = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            let val = one - dot / denom;
            denom
                .simd_eq(zero)
                .select(nan, val)
                .copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            let s = start + j;
            let denom = norms[i] * norms[s];
            seg[pos + j] = if denom == 0.0 {
                f64::NAN
            } else {
                1.0 - dot4(&x[i], &x[s]) / denom
            };
            j += 1;
        }
        pos += row;
    }
}

fn pdist_fill_cosine4(
    x: &[[f64; 4]],
    norms: &[f64],
    n: usize,
    total: usize,
    nthreads: usize,
) -> Vec<f64> {
    let c = dim4_soa(x);
    let mut result = vec![0.0_f64; total];
    if nthreads <= 1 {
        fill_cosine4_rows(x, &c, norms, n, 0, n, &mut result);
        return result;
    }
    let bounds = pdist_row_bounds(n, nthreads);
    let offset = |r: usize| -> usize { r * (n - 1) - r * (r.saturating_sub(1)) / 2 };
    let c = &c;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut result;
        let mut prev = 0usize;
        for w in 0..bounds.len() - 1 {
            let r0 = bounds[w];
            let r1 = bounds[w + 1];
            let take = offset(r1) - prev;
            prev = offset(r1);
            let (seg, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || fill_cosine4_rows(x, c, norms, n, r0, r1, seg));
        }
    });
    result
}

/// SIMD-across-pairs fill of the SqEuclidean dim-4 condensed distances for rows `r0..r1`.
/// Identical to `fill_euclidean4_rows` minus the final `sqrt` — per lane the squared sum runs
/// in the same order as scalar `sqeuclidean` at d=4 (the `0.0+` accumulator start is a no-op),
/// so BIT-identical to the metric helper.
fn fill_sqeuclidean4_rows(
    x: &[[f64; 4]],
    c: &[Vec<f64>; 4],
    n: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::Simd;
    const L: usize = 8;
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    let mut pos = 0usize;
    for i in r0..r1 {
        let a0 = Simd::<f64, L>::splat(c0[i]);
        let a1 = Simd::<f64, L>::splat(c1[i]);
        let a2 = Simd::<f64, L>::splat(c2[i]);
        let a3 = Simd::<f64, L>::splat(c3[i]);
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let d0 = a0 - Simd::<f64, L>::from_slice(&c0[s..s + L]);
            let d1 = a1 - Simd::<f64, L>::from_slice(&c1[s..s + L]);
            let d2 = a2 - Simd::<f64, L>::from_slice(&c2[s..s + L]);
            let d3 = a3 - Simd::<f64, L>::from_slice(&c3[s..s + L]);
            (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            seg[pos + j] = sqeuclidean4(&x[i], &x[start + j]);
            j += 1;
        }
        pos += row;
    }
}

/// SIMD-across-pairs fill of the Cityblock (L1) dim-4 condensed distances for rows `r0..r1`.
/// Per lane `|d0|+|d1|+|d2|+|d3|` runs in the same left-to-right order as scalar `cityblock`
/// at d=4, so BIT-identical to the metric helper.
fn fill_cityblock4_rows(
    x: &[[f64; 4]],
    c: &[Vec<f64>; 4],
    n: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    let mut pos = 0usize;
    for i in r0..r1 {
        let a0 = Simd::<f64, L>::splat(c0[i]);
        let a1 = Simd::<f64, L>::splat(c1[i]);
        let a2 = Simd::<f64, L>::splat(c2[i]);
        let a3 = Simd::<f64, L>::splat(c3[i]);
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let d0 = (a0 - Simd::<f64, L>::from_slice(&c0[s..s + L])).abs();
            let d1 = (a1 - Simd::<f64, L>::from_slice(&c1[s..s + L])).abs();
            let d2 = (a2 - Simd::<f64, L>::from_slice(&c2[s..s + L])).abs();
            let d3 = (a3 - Simd::<f64, L>::from_slice(&c3[s..s + L])).abs();
            (d0 + d1 + d2 + d3).copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            seg[pos + j] = cityblock(&x[i], &x[start + j]);
            j += 1;
        }
        pos += row;
    }
}

/// SIMD-across-pairs fill of the Chebyshev (L∞) dim-4 condensed distances.
/// The NaN mask preserves the scalar helper's `fold(0.0, nan-propagating max)` contract.
fn fill_chebyshev4_rows(
    x: &[[f64; 4]],
    c: &[Vec<f64>; 4],
    n: usize,
    r0: usize,
    r1: usize,
    seg: &mut [f64],
) {
    use std::simd::{Select, Simd, cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
    const L: usize = 8;
    let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
    let nan = Simd::<f64, L>::splat(f64::NAN);
    let mut pos = 0usize;
    for i in r0..r1 {
        let a0 = Simd::<f64, L>::splat(c0[i]);
        let a1 = Simd::<f64, L>::splat(c1[i]);
        let a2 = Simd::<f64, L>::splat(c2[i]);
        let a3 = Simd::<f64, L>::splat(c3[i]);
        let row = n - 1 - i;
        let start = i + 1;
        let mut j = 0usize;
        while j + L <= row {
            let s = start + j;
            let d0 = (a0 - Simd::<f64, L>::from_slice(&c0[s..s + L])).abs();
            let d1 = (a1 - Simd::<f64, L>::from_slice(&c1[s..s + L])).abs();
            let d2 = (a2 - Simd::<f64, L>::from_slice(&c2[s..s + L])).abs();
            let d3 = (a3 - Simd::<f64, L>::from_slice(&c3[s..s + L])).abs();
            let any_nan = d0.simd_ne(d0) | d1.simd_ne(d1) | d2.simd_ne(d2) | d3.simd_ne(d3);
            let m01 = d0.simd_gt(d1).select(d0, d1);
            let m23 = d2.simd_gt(d3).select(d2, d3);
            let max = m01.simd_gt(m23).select(m01, m23);
            any_nan
                .select(nan, max)
                .copy_to_slice(&mut seg[pos + j..pos + j + L]);
            j += L;
        }
        while j < row {
            seg[pos + j] = chebyshev4(&x[i], &x[start + j]);
            j += 1;
        }
        pos += row;
    }
}

/// Parallel wrapper shared by the dim-4 SqEuclidean/Cityblock fast paths: identical row-block
/// split as `pdist_fill_euclidean4` (disjoint contiguous pair ranges, i<j order → bit-identical
/// regardless of worker count). `fill` is the per-row-block SIMD filler.
fn pdist_fill_dim4<F>(x: &[[f64; 4]], n: usize, total: usize, nthreads: usize, fill: F) -> Vec<f64>
where
    F: Fn(&[[f64; 4]], &[Vec<f64>; 4], usize, usize, usize, &mut [f64]) + Sync,
{
    let c = dim4_soa(x);
    let mut result = vec![0.0_f64; total];
    if nthreads <= 1 {
        fill(x, &c, n, 0, n, &mut result);
        return result;
    }
    let bounds = pdist_row_bounds(n, nthreads);
    let offset = |r: usize| -> usize { r * (n - 1) - r * (r.saturating_sub(1)) / 2 };
    let c = &c;
    let fill = &fill;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut result;
        let mut prev = 0usize;
        for w in 0..bounds.len() - 1 {
            let r0 = bounds[w];
            let r1 = bounds[w + 1];
            let take = offset(r1) - prev;
            prev = offset(r1);
            let (seg, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || fill(x, c, n, r0, r1, seg));
        }
    });
    result
}

/// Fill a condensed pairwise-distance vector by evaluating `pair(i, j)` for every i<j.
/// Row i writes a contiguous run of (n−1−i) entries at offset i·(n−1) − i·(i−1)/2; rows are
/// split across threads at pair-balanced boundaries and each fills its disjoint contiguous
/// slice — bit-identical to the sequential i<j order regardless of which core owns a row.
fn pdist_fill<F>(n: usize, total: usize, nthreads: usize, pair: F) -> Vec<f64>
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    if nthreads <= 1 {
        let mut result = Vec::with_capacity(total);
        for i in 0..n {
            for j in (i + 1)..n {
                result.push(pair(i, j));
            }
        }
        return result;
    }
    let mut result = vec![0.0_f64; total];
    let bounds = pdist_row_bounds(n, nthreads);
    let offset = |r: usize| -> usize { r * (n - 1) - r * (r.saturating_sub(1)) / 2 };
    let pair = &pair;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut result;
        let mut prev = 0usize;
        for w in 0..bounds.len() - 1 {
            let r0 = bounds[w];
            let r1 = bounds[w + 1];
            let take = offset(r1) - prev;
            prev = offset(r1);
            let (seg, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || {
                let mut local = 0usize;
                for i in r0..r1 {
                    for j in (i + 1)..n {
                        seg[local] = pair(i, j);
                        local += 1;
                    }
                }
            });
        }
    });
    result
}

/// Pair-balanced row boundaries for a parallel `pdist`: returns up to `nthreads+1`
/// monotonic row indices `[0, ..., n]` so each segment carries ~equal numbers of
/// condensed pairs (row `i` has `n-1-i` pairs, so early rows weigh more).
fn pdist_row_bounds(n: usize, nthreads: usize) -> Vec<usize> {
    let total = n * (n - 1) / 2;
    let mut bounds = Vec::with_capacity(nthreads + 1);
    bounds.push(0);
    let mut acc = 0usize;
    let mut t = 1usize;
    for i in 0..n {
        acc += n - 1 - i;
        while t < nthreads && acc >= t * total / nthreads {
            bounds.push(i + 1);
            t += 1;
        }
    }
    bounds.push(n);
    bounds.dedup();
    bounds
}

/// Convert a condensed distance vector to a square distance matrix.
///
/// Matches `scipy.spatial.distance.squareform(y)` where y is condensed.
///
/// Input: condensed vector of length n*(n-1)/2.
/// Output: symmetric n×n matrix with zeros on diagonal.
pub fn squareform_to_matrix(condensed: &[f64]) -> Result<Vec<Vec<f64>>, SpatialError> {
    // Solve n*(n-1)/2 = len for n
    let len = condensed.len();
    let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    if n * (n - 1) / 2 != len {
        return Err(SpatialError::InvalidArgument(format!(
            "condensed vector length {len} does not correspond to a valid square matrix"
        )));
    }

    // Build each row independently straight from `condensed` (diagonal 0, off-diagonal the pair
    // value) — no pre-zero pass and no cross-row mirror writes, so rows are disjoint and parallel.
    // matrix[i][j] is the same condensed value as the serial fill+mirror ⇒ byte-identical.
    let row_at = |i: usize, out: &mut Vec<f64>| {
        for j in 0..n {
            out.push(if i == j {
                0.0
            } else {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                condensed[a * n - a * (a + 1) / 2 + (b - a - 1)]
            });
        }
    };
    let nthreads = cdist_thread_count(n, n, 1);
    if nthreads <= 1 {
        let mut matrix = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            row_at(i, &mut row);
            matrix.push(row);
        }
        return Ok(matrix);
    }
    let cond = condensed;
    let chunk = n.div_ceil(nthreads);
    let matrix: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|r0| {
                let r1 = (r0 + chunk).min(n);
                scope.spawn(move || {
                    let mut part = Vec::with_capacity(r1 - r0);
                    for i in r0..r1 {
                        let mut row = Vec::with_capacity(n);
                        for j in 0..n {
                            row.push(if i == j {
                                0.0
                            } else {
                                let (a, b) = if i < j { (i, j) } else { (j, i) };
                                cond[a * n - a * (a + 1) / 2 + (b - a - 1)]
                            });
                        }
                        part.push(row);
                    }
                    part
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    Ok(matrix)
}

/// Convert a square distance matrix to condensed form.
///
/// Matches `scipy.spatial.distance.squareform(X)` where X is a matrix.
///
/// Input: symmetric n×n distance matrix.
/// Output: condensed vector of length n*(n-1)/2.
pub fn squareform_to_condensed(matrix: &[Vec<f64>]) -> Result<Vec<f64>, SpatialError> {
    let n = matrix.len();
    if n < 2 {
        return Err(SpatialError::InvalidArgument(
            "matrix must be at least 2×2".to_string(),
        ));
    }
    for row in matrix {
        if row.len() != n {
            return Err(SpatialError::InvalidArgument(format!(
                "matrix must be square: expected {} columns, got {}",
                n,
                row.len()
            )));
        }
    }
    for (i, row) in matrix.iter().enumerate() {
        if row[i] != 0.0 {
            return Err(SpatialError::InvalidArgument(
                "distance matrix diagonal must be zero".to_string(),
            ));
        }
        for j in (i + 1)..n {
            if row[j] != matrix[j][i] {
                return Err(SpatialError::InvalidArgument(
                    "distance matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for (i, row) in matrix.iter().enumerate() {
        for &val in &row[i + 1..] {
            condensed.push(val);
        }
    }
    Ok(condensed)
}

/// Validate that `matrix` is a square symmetric distance matrix with
/// zeros on the diagonal (within `tol`).
///
/// Matches `scipy.spatial.distance.is_valid_dm(D, tol)`. Returns `true`
/// for any conformant matrix; rejects ragged rows, asymmetry beyond
/// `tol`, and non-zero diagonal entries beyond `tol`. NaN entries fail
/// validation.
#[must_use]
pub fn is_valid_dm(matrix: &[Vec<f64>], tol: f64) -> bool {
    let n = matrix.len();
    if n == 0 {
        return false;
    }
    for row in matrix {
        if row.len() != n {
            return false;
        }
    }
    for (i, row) in matrix.iter().enumerate() {
        if row[i].is_nan() || row[i].abs() > tol {
            return false;
        }
        for j in (i + 1)..n {
            let upper = row[j];
            let lower = matrix[j][i];
            if upper.is_nan() || lower.is_nan() || (upper - lower).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Validate that `condensed` is a valid condensed distance vector,
/// i.e. its length equals n*(n-1)/2 for some integer n ≥ 1.
///
/// Matches `scipy.spatial.distance.is_valid_y(y)`. NaN entries fail.
#[must_use]
pub fn is_valid_y(condensed: &[f64]) -> bool {
    if condensed.iter().any(|v| v.is_nan()) {
        return false;
    }
    let len = condensed.len();
    if len == 0 {
        // Length 0 corresponds to n=1 (a single observation has no
        // pairwise distances). scipy treats this as valid.
        return true;
    }
    let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    n >= 1 && n * (n - 1) / 2 == len
}

/// Number of observations represented by a square distance matrix.
///
/// Matches `scipy.spatial.distance.num_obs_dm(D)`. Returns the matrix
/// side length. Caller is responsible for prior validation via
/// [`is_valid_dm`] if untrusted input.
#[must_use]
pub fn num_obs_dm(matrix: &[Vec<f64>]) -> usize {
    matrix.len()
}

/// Number of observations represented by a condensed distance vector.
///
/// Matches `scipy.spatial.distance.num_obs_y(y)`. Returns `N` such that
/// `y.len() = N*(N-1)/2`. Returns `0` on invalid input lengths to mirror
/// scipy's failing-with-zero behavior on malformed vectors.
#[must_use]
pub fn num_obs_y(condensed: &[f64]) -> usize {
    let len = condensed.len();
    if len == 0 {
        return 1;
    }
    let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    if n * (n - 1) / 2 == len { n } else { 0 }
}

/// Compute pairwise Euclidean distance matrix.
///
/// Matches `scipy.spatial.distance.cdist(XA, XB, 'euclidean')`.
/// Returns a matrix where `result[i][j] = distance(xa[i], xb[j])`.
pub fn cdist(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }

    cdist_metric(xa, xb, DistanceMetric::Euclidean)
}

/// Compute pairwise distance matrix with a specified metric.
///
/// Matches `scipy.spatial.distance.cdist(XA, XB, metric)`.
/// Cross-distance matrix under the Minkowski (L^p) metric, matching
/// `scipy.spatial.distance.cdist(XA, XB, 'minkowski', p=p)` — the full na×nb matrix (distinct
/// from row-wise [`minkowski_distance`]). Parallel over rows via `cdist_fill`; each pair uses the
/// tested scalar [`minkowski`]. SciPy's per-element pow makes its cdist-minkowski slow; fsci
/// parallelizes it. Not in `DistanceMetric` because that enum derives `Eq` (p is f64).
pub fn cdist_minkowski(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    p: f64,
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    if p.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(SpatialError::InvalidArgument(
            "minkowski p must be > 0".to_string(),
        ));
    }
    let dim = xa[0].len();
    for row in xa.iter().chain(xb.iter()) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    let na = xa.len();
    let nb = xb.len();
    let nthreads = cdist_thread_count(na, nb, dim);
    Ok(cdist_fill(na, nb, nthreads, |i, j| {
        minkowski(&xa[i], &xb[j], p)
    }))
}

/// Cross-distance matrix under the standardized-Euclidean metric, matching
/// `scipy.spatial.distance.cdist(XA, XB, 'seuclidean', V=v)` (each squared component divided by
/// `v[k]`). Parallel over rows via `cdist_fill`; per pair uses the tested scalar [`seuclidean`].
pub fn cdist_seuclidean(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    v: &[f64],
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if v.len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: v.len(),
        });
    }
    for row in xa.iter().chain(xb.iter()) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    let (na, nb) = (xa.len(), xb.len());
    let nthreads = cdist_thread_count(na, nb, dim);
    // Standardized-Euclidean is Euclidean in the whitened space `x/√vₖ`: the per-pair
    // `seuclidean` divides by `vₖ` AND re-checks `vₖ ≤ 0` per element on EVERY pair, even
    // though `v` is shared. When all `vₖ > 0`, scale both point sets by `1/√vₖ` ONCE and
    // route through the already-tuned Euclidean cdist (no per-pair divide/mask). Not
    // byte-identical (`x/√v` rounds first) but agrees to ~1e-15. A non-positive `vₖ` keeps
    // the exact per-pair NaN behavior below.
    if dim >= 8
        && !SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
        && v.iter().all(|&vk| vk > 0.0)
    {
        let inv_sqrt: Vec<f64> = v.iter().map(|&vk| 1.0 / vk.sqrt()).collect();
        let scale = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
            rows.iter()
                .map(|r| r.iter().zip(&inv_sqrt).map(|(&x, &s)| x * s).collect())
                .collect()
        };
        let sa = scale(xa);
        let sb = scale(xb);
        return cdist_metric(&sa, &sb, DistanceMetric::Euclidean);
    }
    // Small d: the per-pair `seuclidean(&xa[i],&xb[j],v)` pointer-chases the Vec<Vec>
    // and loses to SciPy's contiguous loop (same loss the Euclidean/Canberra SoA paths
    // flipped). For d<8 `seuclidean` is a scalar left-fold ⇒ the per-lane SoA reduction
    // is bit-identical. Bandwidth-bound ⇒ cap workers at 16.
    if dim < 8 {
        let b_soa = cdist_soa(xb, dim);
        return Ok(cdist_fill_rows(na, nthreads.min(16), |i| {
            cdist_row_seuclidean_soa(&xa[i], &b_soa, v, nb)
        }));
    }
    Ok(cdist_fill(na, nb, nthreads, |i, j| {
        seuclidean(&xa[i], &xb[j], v)
    }))
}

/// Same-binary A/B toggle: when `true`, `cdist_seuclidean` uses the per-pair divide path;
/// when `false` (default) it whitens to `x/√v` once and routes through Euclidean cdist.
/// CONTRACT: NOT byte-identical -- agrees to ~1e-15 relative. Standardized Euclidean is
/// Euclidean in the whitened space, so the fast arm scales both point sets by `1/sqrt(v_k)`
/// ONCE and routes through the tuned Euclidean kernel instead of dividing by `v_k` per
/// pair. The rounding differs because the scaling rounds FIRST: `(x_i - y_i)^2 / v` and
/// `(x_i*s - y_i*s)^2` are the same quantity in exact arithmetic and two different f64
/// values, with `1/sqrt(v_k)` carrying its own rounding into every coordinate.
///
/// The gate `v.iter().all(|&vk| vk > 0.0)` is part of the contract, not a performance
/// heuristic. A zero or negative `v_k` makes `1/sqrt(v_k)` infinite or NaN and would
/// poison every pair through the whitened path, whereas the per-pair arm produces scipy's
/// exact per-element NaN behaviour. Non-positive variance therefore keeps the slow arm on
/// both sides of the toggle.
pub static SPATIAL_SEUCLIDEAN_PRESCALE_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Same-binary A/B toggle: when `true`, the boolean Jaccard `pdist`/`cdist` paths use the
/// scalar/SoA per-dimension count; when `false` (default) they pack rows to `u64` words and
/// count via hardware popcount (O(d/64) per pair). The popcount path is byte-identical — the
/// integer (in)equality counts are exact either way — and that claim is gated by
/// `bool_popcount_ab_tests::jaccard_popcount_ab_is_byte_identical_above_the_dim_gate`.
///
/// JACCARD ONLY, and only above [`JACCARD_POPCOUNT_MIN_DIM`]. This doc used to say
/// "Hamming/Jaccard", but there is no packed Hamming path: both reads of this toggle
/// (the `pdist` and `cdist` dispatches) match on `DistanceMetric::Jaccard`, and Hamming
/// goes to the scalar/SoA count unconditionally. Below the dim gate Jaccard does too, so
/// an A/B run at small `d` compares the scalar path with itself (frankenscipy-0b982).
pub static SPATIAL_BOOL_POPCOUNT_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[inline]
fn spatial_bool_popcount() -> bool {
    !SPATIAL_BOOL_POPCOUNT_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Minimum coordinate count before the bit-packed popcount Jaccard cdist beats the
/// already-vectorized SoA count (below it the packing overhead dominates).
const JACCARD_POPCOUNT_MIN_DIM: usize = 256;

/// Pack each row's `dim` coordinates into `⌈dim/64⌉` `u64` words, bit k = `(value != 0.0)`.
/// Padding bits in the final word stay 0, so they contribute nothing to AND/OR/XOR popcounts.
/// Also returns each row's set-bit count (its number of nonzero coordinates).
fn cdist_pack_bits(x: &[Vec<f64>], dim: usize) -> (Vec<Vec<u64>>, Vec<u32>) {
    let words = dim.div_ceil(64);
    let mut bits = vec![vec![0u64; words]; x.len()];
    let mut ones = vec![0u32; x.len()];
    for (row, (packed, cnt)) in x.iter().zip(bits.iter_mut().zip(ones.iter_mut())) {
        for (k, &v) in row.iter().enumerate() {
            if v != 0.0 {
                packed[k >> 6] |= 1u64 << (k & 63);
            }
        }
        *cnt = packed.iter().map(|w| w.count_ones()).sum();
    }
    (bits, ones)
}

/// One Jaccard cdist row from packed bits: `(c_TF + c_FT) / (c_TT + c_TF + c_FT) =
/// popcount(a ⊕ b) / popcount(a ∨ b)`, returning 0 when the union is empty (matches scalar).
/// Jaccard dissimilarity of two packed rows: `popcount(a⊕b) / popcount(a∨b)`, where
/// `popcount(a∨b) = ones_a + ones_b − popcount(a∧b)`; 0 when the union is empty.
#[inline]
fn jaccard_bits_pair(a: &[u64], ones_a: u32, b: &[u64], ones_b: u32) -> f64 {
    let ntt: u32 = a
        .iter()
        .zip(b)
        .map(|(&aw, &bw)| (aw & bw).count_ones())
        .sum();
    let n_union = ones_a + ones_b - ntt;
    if n_union == 0 {
        0.0
    } else {
        (n_union - ntt) as f64 / n_union as f64
    }
}

fn cdist_row_jaccard_packed(
    a: &[u64],
    ones_a: u32,
    b_bits: &[Vec<u64>],
    ones_b: &[u32],
    nb: usize,
) -> Vec<f64> {
    (0..nb)
        .map(|j| jaccard_bits_pair(a, ones_a, &b_bits[j], ones_b[j]))
        .collect()
}

pub fn cdist_metric(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    metric: DistanceMetric,
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }
    for row in xa.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }
    for row in xb.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual: row.len(),
            });
        }
    }

    // Each output row (all distances from xa[i] to xb) is an independent reduction
    // over the same pure `metric_distance`, so rows can be computed on different cores
    // with a bit-identical result — only the owning thread changes. Split the rows of
    // xa across threads when the matrix is large enough to amortise spawn.
    let na = xa.len();
    let nb = xb.len();
    let nthreads = cdist_thread_count(na, nb, dim);

    // Cosine and Correlation recompute each vector's norm / mean+centered+ssa per pair —
    // quantities depending on ONE vector. Precompute them once for BOTH xa and xb (O((na+nb)·d))
    // so each of the na·nb pairs does only the cross term. Byte-identical: precomputed
    // quantities and the cross term use the same arithmetic/summation order as the scalar
    // `cosine`/`correlation` helpers (see `pdist`).
    let result: Vec<Vec<f64>> = match metric {
        // Dim-4 Euclidean/Cosine: vectorize a whole output row across xb columns (SoA) so the
        // dependent per-pair sqrt/divide pipeline across SIMD lanes. BIT-identical to the
        // generic per-pair arm at d=4 (see pdist nm8ex); flips a 2.4-2.6x SciPy loss.
        DistanceMetric::Euclidean if dim == 4 => {
            let xa4 = collect_dim4_points(xa);
            let b4 = collect_dim4_points(xb);
            let b_soa = dim4_soa(&b4);
            // The d=4 kernel is memory-bandwidth-bound (≈40 bytes traffic per 8-byte output,
            // low arithmetic intensity), so ~16 threads saturate bandwidth; beyond that extra
            // workers only contend (measured: cap16 1.9ms vs cap64 4.6ms vs SciPy 2.2ms).
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_euclidean4(&xa4[i], &b_soa, nb)
            })
        }
        // Generalize the d=4 SoA-across-pairs SIMD to any small d (d<8). For d<8
        // `sqeuclidean` is a scalar left-fold (its SIMD chunk needs d>=8), so the
        // per-lane left-fold reduction `Σ_k (ai[k]-b[k][lane])²` is BIT-identical
        // to the generic per-pair arm — this flips the common d=2/d=3 point-cloud
        // cdist loss, where the generic arm pointer-chases xa[i]/xb[j] through the
        // Vec<Vec> and loses to SciPy's tight contiguous loop at low d.
        // Bandwidth-bound like d=4 ⇒ cap workers at 16.
        DistanceMetric::Euclidean if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_euclidean_soa(&xa[i], &b_soa, nb)
            })
        }
        // Same SoA-across-pairs lever for SqEuclidean (drop the sqrt) and
        // Cityblock (sum of |Δ|) at small d. `sqeuclidean`/`cityblock` are both
        // scalar left-folds for d<8, so the per-lane reduction is BIT-identical
        // to the generic arm. Flips the d=3 cdist losses (8.4x / 4.6x slower).
        DistanceMetric::SqEuclidean if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_sqeuclidean_soa(&xa[i], &b_soa, nb)
            })
        }
        DistanceMetric::Cityblock if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_cityblock_soa(&xa[i], &b_soa, nb)
            })
        }
        // Chebyshev = max_k |Δ_k| with NaN-propagation. Mirrors the scalar
        // `chebyshev` helper (a separate per-lane NaN mask, since the running max
        // would otherwise overwrite a NaN). Bit-identical for d<8.
        DistanceMetric::Chebyshev if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_chebyshev_soa(&xa[i], &b_soa, nb)
            })
        }
        DistanceMetric::Cosine if dim == 4 => {
            let na_norm: Vec<f64> = xa.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            let nb_norm: Vec<f64> = xb.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            let xa4 = collect_dim4_points(xa);
            let b4 = collect_dim4_points(xb);
            let b_soa = dim4_soa(&b4);
            // Bandwidth-bound like Euclidean: cap workers at 16 to avoid contention.
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_cosine4(&xa4[i], na_norm[i], &b_soa, &nb_norm, nb)
            })
        }
        // Generalize the d=4 Cosine SoA to small d (d<8): same precomputed norms
        // + per-lane `1 - dot/(ni·nj)` with the denom==0⇒NaN select. `simd_dot`
        // is a scalar left-fold for d<8, so this is BIT-identical to the generic
        // `cosine` arm. Flips the d=3 cosine loss (3.5x slower).
        DistanceMetric::Cosine if dim < 8 => {
            let na_norm: Vec<f64> = xa.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            let nb_norm: Vec<f64> = xb.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_cosine_soa(&xa[i], na_norm[i], &b_soa, &nb_norm, nb)
            })
        }
        DistanceMetric::Cosine => {
            let na_norm: Vec<f64> = xa.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            let nb_norm: Vec<f64> = xb.iter().map(|v| simd_sqsum(v).sqrt()).collect();
            cdist_fill(na, nb, nthreads, |i, j| {
                let denom = na_norm[i] * nb_norm[j];
                if denom == 0.0 {
                    f64::NAN
                } else {
                    1.0 - simd_dot(&xa[i], &xb[j]) / denom
                }
            })
        }
        DistanceMetric::Correlation if dim >= 2 => {
            let dn = dim as f64;
            let prep = |v: &[f64]| -> (Vec<f64>, f64) {
                let mean = v.iter().sum::<f64>() / dn;
                let c: Vec<f64> = v.iter().map(|&xi| xi - mean).collect();
                let ss: f64 = c.iter().map(|&ci| ci * ci).sum();
                (c, ss)
            };
            let pa: Vec<(Vec<f64>, f64)> = xa.iter().map(|v| prep(v)).collect();
            let pb: Vec<(Vec<f64>, f64)> = xb.iter().map(|v| prep(v)).collect();
            // Small d: SoA-across-pairs the centered cross term (like Cosine) — the per-pair
            // dot otherwise pointer-chases pb[j]'s centered Vec. Transpose the centered xb
            // columns + collect ssb; per output row vectorize the dot over xb columns. The
            // per-lane reduction matches the scalar dot (`.sum()` left-fold) bit-for-bit, and
            // `denom=(ssa*ssb).sqrt()` is computed identically. Bandwidth-bound ⇒ cap at 16.
            if dim < 8 {
                let cb_soa: Vec<Vec<f64>> = (0..dim)
                    .map(|k| pb.iter().map(|(c, _)| c[k]).collect())
                    .collect();
                let ssb: Vec<f64> = pb.iter().map(|(_, ss)| *ss).collect();
                cdist_fill_rows(na, nthreads.min(16), |i| {
                    let (ci, ssa) = &pa[i];
                    cdist_row_correlation_soa(ci, *ssa, &cb_soa, &ssb, nb)
                })
            } else {
                cdist_fill(na, nb, nthreads, |i, j| {
                    let (ci, ssa) = &pa[i];
                    let (cj, ssb) = &pb[j];
                    let ssab: f64 = ci.iter().zip(cj.iter()).map(|(&p, &q)| p * q).sum();
                    let denom = (ssa * ssb).sqrt();
                    if denom == 0.0 {
                        f64::NAN
                    } else {
                        1.0 - ssab / denom
                    }
                })
            }
        }
        // Canberra / Bray-Curtis at small d hit the generic per-pair arm below,
        // which pointer-chases xa[i]/xb[j] through the Vec<Vec> and loses to SciPy's
        // contiguous C loop (same loss the Euclidean/Cityblock SoA paths flipped).
        // Both are scalar left-folds over dimensions for d<8 ⇒ the per-lane SoA
        // reduction is bit-identical. Bandwidth-bound ⇒ cap workers at 16.
        DistanceMetric::Canberra if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_canberra_soa(&xa[i], &b_soa, nb)
            })
        }
        DistanceMetric::Braycurtis if dim < 8 => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_braycurtis_soa(&xa[i], &b_soa, nb)
            })
        }
        // Hamming / Jaccard count (in)equalities per pair — the scalar helpers are
        // ALWAYS scalar filter/count (no per-dim SIMD), so the SoA-across-pairs count
        // is bit-identical at EVERY d (exact integer counters), and the win grows with
        // d (more comparisons to vectorize across pairs). Bandwidth-bound ⇒ cap at 16.
        DistanceMetric::Hamming => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_hamming_soa(&xa[i], &b_soa, nb)
            })
        }
        // Jaccard converts to boolean internally (bit = value≠0), so bit-packing is
        // byte-identical for ANY numeric input — no boolean check needed. Gated on `dim`:
        // at small d the already-vectorized SoA count wins; the O(d/64) popcount pulls
        // ahead once there are enough words to amortize the packing (crossover ~128-256).
        DistanceMetric::Jaccard if spatial_bool_popcount() && dim >= JACCARD_POPCOUNT_MIN_DIM => {
            let (xa_bits, ones_a) = cdist_pack_bits(xa, dim);
            let (xb_bits, ones_b) = cdist_pack_bits(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_jaccard_packed(&xa_bits[i], ones_a[i], &xb_bits, &ones_b, nb)
            })
        }
        DistanceMetric::Jaccard => {
            let b_soa = cdist_soa(xb, dim);
            cdist_fill_rows(na, nthreads.min(16), |i| {
                cdist_row_jaccard_soa(&xa[i], &b_soa, nb)
            })
        }
        _ => cdist_fill(na, nb, nthreads, |i, j| {
            metric_distance(&xa[i], &xb[j], metric)
        }),
    };
    Ok(result)
}

/// Fill a `na × nb` distance matrix by evaluating `pair(i, j)`. Rows of xa are split across
/// threads (each fills a contiguous row block) and reassembled in order — bit-identical to
/// the sequential row-major order regardless of which core owns a row.
fn cdist_fill<F>(na: usize, nb: usize, nthreads: usize, pair: F) -> Vec<Vec<f64>>
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    if nthreads <= 1 {
        return (0..na)
            .map(|i| (0..nb).map(|j| pair(i, j)).collect())
            .collect();
    }
    let chunk = na.div_ceil(nthreads);
    let pair = &pair;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= na {
                    return None;
                }
                let i1 = (i0 + chunk).min(na);
                Some(scope.spawn(move || {
                    (i0..i1)
                        .map(|i| (0..nb).map(|j| pair(i, j)).collect::<Vec<f64>>())
                        .collect::<Vec<Vec<f64>>>()
                }))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("cdist worker panicked"))
            .collect()
    })
}

/// Worker count for a parallel `cdist`: 1 (sequential) unless the distance matrix
/// carries enough total work (`na·nb·dim`) to amortise thread spawn, then scale with
/// cores, capped so each thread owns at least a couple of output rows.
fn cdist_thread_count(na: usize, nb: usize, dim: usize) -> usize {
    let work = (na as u64)
        .saturating_mul(nb as u64)
        .saturating_mul(dim.max(1) as u64);
    if work < 1 << 18 || na < 4 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(na / 2).max(1)
}

/// Fill a `na × nb` distance matrix one row at a time via `fill_row(i)`, splitting xa's
/// rows across threads (each thread owns a contiguous block, reassembled in order — so the
/// result is bit-identical to sequential row-major regardless of which core owns a row).
/// Lets the dim-4 fast paths vectorize a whole output row with one SIMD-across-columns pass.
fn cdist_fill_rows<F>(na: usize, nthreads: usize, fill_row: F) -> Vec<Vec<f64>>
where
    F: Fn(usize) -> Vec<f64> + Sync,
{
    if nthreads <= 1 {
        return (0..na).map(&fill_row).collect();
    }
    let chunk = na.div_ceil(nthreads);
    let fill_row = &fill_row;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= na {
                    return None;
                }
                let i1 = (i0 + chunk).min(na);
                Some(scope.spawn(move || (i0..i1).map(fill_row).collect::<Vec<Vec<f64>>>()))
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("cdist worker panicked"))
            .collect()
    })
}

/// One dim-4 Euclidean cdist output row: distances from `ai` to every xb point, xb stored
/// SoA in columns `b`. Lane k holds column j+k; per lane the squared sum and sqrt run in
/// the same order as scalar `sqeuclidean4(..).sqrt()` (== `euclidean` at d=4), so the row is
/// BIT-identical to the generic `metric_distance` arm while `vsqrtpd` pipelines the sqrts.
fn cdist_row_euclidean4(ai: &[f64; 4], b: &[Vec<f64>; 4], nb: usize) -> Vec<f64> {
    use std::simd::{Simd, StdFloat};
    const L: usize = 8;
    let (b0, b1, b2, b3) = (&b[0], &b[1], &b[2], &b[3]);
    let a0 = Simd::<f64, L>::splat(ai[0]);
    let a1 = Simd::<f64, L>::splat(ai[1]);
    let a2 = Simd::<f64, L>::splat(ai[2]);
    let a3 = Simd::<f64, L>::splat(ai[3]);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let d0 = a0 - Simd::<f64, L>::from_slice(&b0[j..j + L]);
        let d1 = a1 - Simd::<f64, L>::from_slice(&b1[j..j + L]);
        let d2 = a2 - Simd::<f64, L>::from_slice(&b2[j..j + L]);
        let d3 = a3 - Simd::<f64, L>::from_slice(&b3[j..j + L]);
        let sq = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        sq.sqrt().copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let bj = [b0[j], b1[j], b2[j], b3[j]];
        row[j] = sqeuclidean4(ai, &bj).sqrt();
        j += 1;
    }
    row
}

/// Structure-of-arrays layout of `x`: `out[k][j]` is coordinate `k` of point `j`,
/// so the SoA SIMD kernels read each coordinate column contiguously.
fn cdist_soa(x: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut soa = vec![vec![0.0_f64; n]; dim];
    for (j, row) in x.iter().enumerate() {
        for (k, col) in soa.iter_mut().enumerate() {
            col[j] = row[k];
        }
    }
    soa
}

/// One Euclidean cdist output row for small `d` (`d < 8`) via SoA-across-pairs
/// SIMD: lane `l` holds column `j+l` of xb, so the dependent per-pair `sqrt`
/// pipelines across lanes. The per-lane reduction `Σ_k (ai[k]-b[k][lane])²` is a
/// left-fold over the dimensions — BIT-identical to `sqeuclidean` (itself a
/// scalar left-fold for `d < 8`), hence to the generic per-pair arm.
fn cdist_row_euclidean_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Simd, StdFloat};
    const L: usize = 8;
    let d = ai.len();
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut sq = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let dk = Simd::<f64, L>::splat(ai[k]) - Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            sq += dk * dk;
        }
        sq.sqrt().copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            let dk = ai[k] - b[k][j];
            s += dk * dk;
        }
        row[j] = s.sqrt();
        j += 1;
    }
    row
}

/// SoA-across-pairs SqEuclidean cdist row (small `d`) — `cdist_row_euclidean_soa`
/// minus the final `sqrt`. Bit-identical to scalar `sqeuclidean` for `d < 8`.
fn cdist_row_sqeuclidean_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::Simd;
    const L: usize = 8;
    let d = ai.len();
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut sq = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let dk = Simd::<f64, L>::splat(ai[k]) - Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            sq += dk * dk;
        }
        sq.copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            let dk = ai[k] - b[k][j];
            s += dk * dk;
        }
        row[j] = s;
        j += 1;
    }
    row
}

/// SoA-across-pairs Cityblock (Manhattan) cdist row (small `d`): per-lane
/// `Σ_k |ai[k]-b[k][lane]|`, a left-fold of absolute differences — bit-identical
/// to scalar `cityblock` for `d < 8`.
fn cdist_row_cityblock_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Simd, num::SimdFloat};
    const L: usize = 8;
    let d = ai.len();
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut acc = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let dk = Simd::<f64, L>::splat(ai[k]) - Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            acc += dk.abs();
        }
        acc.copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            s += (ai[k] - b[k][j]).abs();
        }
        row[j] = s;
        j += 1;
    }
    row
}

/// SoA-across-pairs Chebyshev (L∞) cdist row (small `d`): per-lane running max of
/// `|ai[k]-b[k][lane]|` with a separate NaN mask (a NaN diff would otherwise be
/// overwritten by a later max), exactly mirroring the scalar `chebyshev` helper.
/// Bit-identical for `d < 8` (the helper's max starts from 0.0 = `reduce_max` of
/// the zeroed accumulator).
fn cdist_row_chebyshev_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq, cmp::SimdPartialOrd, num::SimdFloat};
    const L: usize = 8;
    let d = ai.len();
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut vmax = Simd::<f64, L>::splat(0.0);
        let mut nan = vmax.simd_ne(vmax); // all-false mask
        for k in 0..d {
            let dk =
                (Simd::<f64, L>::splat(ai[k]) - Simd::<f64, L>::from_slice(&b[k][j..j + L])).abs();
            nan |= dk.simd_ne(dk);
            vmax = vmax.simd_gt(dk).select(vmax, dk);
        }
        nan.select(Simd::<f64, L>::splat(f64::NAN), vmax)
            .copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            let dk = (ai[k] - b[k][j]).abs();
            if dk.is_nan() {
                s = f64::NAN;
                break;
            }
            s = s.max(dk);
        }
        row[j] = s;
        j += 1;
    }
    row
}

/// SoA-across-pairs Canberra cdist row (small `d`): per-lane
/// `Σ_k |ai[k]-b[k][lane]| / (|ai[k]|+|b[k][lane]|)` with the `denom==0 ⇒ term 0`
/// mask, exactly mirroring the scalar `canberra` helper (whose per-dim SIMD does
/// not engage for `d < 8`, so it left-folds `k=0..d` skipping denom==0 — and
/// `s + 0.0 == s`, so the masked add is bit-identical).
fn cdist_row_canberra_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq, num::SimdFloat};
    const L: usize = 8;
    let d = ai.len();
    let zero = Simd::<f64, L>::splat(0.0);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut acc = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let av = Simd::<f64, L>::splat(ai[k]);
            let bv = Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            let denom = av.abs() + bv.abs();
            let term = (av - bv).abs() / denom;
            acc += denom.simd_eq(zero).select(zero, term);
        }
        acc.copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            let denom = ai[k].abs() + b[k][j].abs();
            if denom != 0.0 {
                s += (ai[k] - b[k][j]).abs() / denom;
            }
        }
        row[j] = s;
        j += 1;
    }
    row
}

/// SoA-across-pairs Bray-Curtis cdist row (small `d`): per-lane
/// `Σ|ai[k]-b[k][lane]| / Σ|ai[k]+b[k][lane]|` with the `den==0 ⇒ 0` guard. The
/// numerator/denominator left-folds over `k=0..d` match the scalar `braycurtis`
/// helper's two `.sum()` passes (same accumulation order), so this is bit-identical
/// for `d < 8`.
fn cdist_row_braycurtis_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq, num::SimdFloat};
    const L: usize = 8;
    let d = ai.len();
    let zero = Simd::<f64, L>::splat(0.0);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut num = Simd::<f64, L>::splat(0.0);
        let mut den = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let av = Simd::<f64, L>::splat(ai[k]);
            let bv = Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            num += (av - bv).abs();
            den += (av + bv).abs();
        }
        let res = den.simd_eq(zero).select(zero, num / den);
        res.copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for k in 0..d {
            num += (ai[k] - b[k][j]).abs();
            den += (ai[k] + b[k][j]).abs();
        }
        row[j] = if den == 0.0 { 0.0 } else { num / den };
        j += 1;
    }
    row
}

/// SoA-across-pairs Hamming cdist row: per-lane `count_k(ai[k] != b[k][lane]) / d`.
/// The count is an exact integer (sum of 1.0s, `< 2^53`), so it equals the scalar
/// helper's `mismatches as f64`, and the final `/ d` matches — bit-identical for ANY
/// `d` (the scalar `hamming` is always a scalar filter/count, no per-dim SIMD to match,
/// so the across-pairs reduction is exact regardless of dimension). `simd_ne` treats
/// NaN as unequal, exactly like Rust `!=`.
fn cdist_row_hamming_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq};
    const L: usize = 8;
    let d = ai.len();
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let dv = Simd::<f64, L>::splat(d as f64);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut acc = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let av = Simd::<f64, L>::splat(ai[k]);
            let bv = Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            acc += av.simd_ne(bv).select(one, zero);
        }
        (acc / dv).copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut cnt = 0.0_f64;
        for k in 0..d {
            if ai[k] != b[k][j] {
                cnt += 1.0;
            }
        }
        row[j] = cnt / d as f64;
        j += 1;
    }
    row
}

/// SoA-across-pairs Jaccard cdist row: per-lane counts `nz = #(ai[k]≠0 ∨ b≠0)` and
/// `uneq = #(ai[k]≠0 ⊕ b≠0)`, result `uneq/nz` (0 if `nz==0`). `a_b ⊕ b_b ⇒ a_b ∨ b_b`,
/// so counting `uneq` unconditionally equals the scalar helper's guarded count; both
/// counters are exact integers, so this is bit-identical for ANY `d`.
///
/// `a_b = splat(ai[k]).ne(0)` is LANE-UNIFORM (ai[k] is a scalar), so we branch on
/// the scalar once per k instead of materializing the splat/OR/XOR per chunk: when
/// `ai[k]≠0` every lane is `(a_b|b_b)=1` (so `nz += 1`) and `(a_b^b_b)=!b_b`; when
/// `ai[k]=0` both reduce to `b_b`. Halves the per-element ALU at high `d` (the
/// compute-bound regime) while incrementing exactly the same lanes ⇒ still bit-identical.
fn cdist_row_jaccard_soa(ai: &[f64], b: &[Vec<f64>], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq};
    const L: usize = 8;
    let d = ai.len();
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut nz = Simd::<f64, L>::splat(0.0);
        let mut uneq = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let b_b = Simd::<f64, L>::from_slice(&b[k][j..j + L]).simd_ne(zero);
            if ai[k] != 0.0 {
                nz += one;
                uneq += b_b.select(zero, one);
            } else {
                let bm = b_b.select(one, zero);
                nz += bm;
                uneq += bm;
            }
        }
        nz.simd_eq(zero)
            .select(zero, uneq / nz)
            .copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let (mut nz, mut uneq) = (0.0_f64, 0.0_f64);
        for k in 0..d {
            let a_b = ai[k] != 0.0;
            let b_b = b[k][j] != 0.0;
            if a_b || b_b {
                nz += 1.0;
                if a_b != b_b {
                    uneq += 1.0;
                }
            }
        }
        row[j] = if nz == 0.0 { 0.0 } else { uneq / nz };
        j += 1;
    }
    row
}

/// SoA-across-pairs standardized-Euclidean cdist row (small `d`) with variance
/// vector `v`: per-lane `sqrt(Σ_k (ai[k]-b[k][lane])²/v[k])` with the `v[k]<=0 ⇒ NaN`
/// mask. Bit-identical to the scalar `seuclidean` for `d < 8` (whose per-dim SIMD
/// needs d≥8, so it left-folds `k=0..d`; `dk*dk == dk.powi(2)`, the `v<=0` branch
/// becomes a lane select on a uniformly-true/false mask since `v[k]` is a splat).
fn cdist_row_seuclidean_soa(ai: &[f64], b: &[Vec<f64>], v: &[f64], nb: usize) -> Vec<f64> {
    use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialOrd};
    const L: usize = 8;
    let d = ai.len();
    let zero = Simd::<f64, L>::splat(0.0);
    let nan_v = Simd::<f64, L>::splat(f64::NAN);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut acc = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            let dk = Simd::<f64, L>::splat(ai[k]) - Simd::<f64, L>::from_slice(&b[k][j..j + L]);
            let vv = Simd::<f64, L>::splat(v[k]);
            acc += vv.simd_le(zero).select(nan_v, (dk * dk) / vv);
        }
        acc.sqrt().copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let mut s = 0.0_f64;
        for k in 0..d {
            let vi = v[k];
            s += if vi <= 0.0 {
                f64::NAN
            } else {
                (ai[k] - b[k][j]).powi(2) / vi
            };
        }
        row[j] = s.sqrt();
        j += 1;
    }
    row
}

/// SoA-across-pairs Cosine cdist row (small `d`) with precomputed norms (`ni` for
/// `ai`, `nb_norm` for xb): per-lane `1 - dot/(ni·nj)` with the denom==0⇒NaN
/// guard, the dot a left-fold over dimensions. Bit-identical to the scalar
/// `cosine` (`simd_dot`/`simd_sqsum` are scalar left-folds for `d < 8`).
fn cdist_row_cosine_soa(
    ai: &[f64],
    ni: f64,
    b: &[Vec<f64>],
    nb_norm: &[f64],
    nb: usize,
) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq};
    const L: usize = 8;
    let d = ai.len();
    let niv = Simd::<f64, L>::splat(ni);
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let nan = Simd::<f64, L>::splat(f64::NAN);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut dot = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            dot += Simd::<f64, L>::splat(ai[k]) * Simd::<f64, L>::from_slice(&b[k][j..j + L]);
        }
        let njv = Simd::<f64, L>::from_slice(&nb_norm[j..j + L]);
        let denom = niv * njv;
        let val = one - dot / denom;
        denom
            .simd_eq(zero)
            .select(nan, val)
            .copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let denom = ni * nb_norm[j];
        row[j] = if denom == 0.0 {
            f64::NAN
        } else {
            let mut dot = 0.0_f64;
            for k in 0..d {
                dot += ai[k] * b[k][j];
            }
            1.0 - dot / denom
        };
        j += 1;
    }
    row
}

/// SoA-across-pairs Correlation cdist row (small `d`) with precomputed centered xa row
/// `ci` (+ its `ssa`) and SoA-transposed centered xb columns `cb` (+ per-column `ssb`):
/// per-lane `1 - ssab/sqrt(ssa·ssb)` with the `denom==0 ⇒ NaN` guard, the cross term
/// `ssab = Σ_k ci[k]·cb[k][lane]` a left-fold over dimensions. Bit-identical to the scalar
/// precompute arm (same centered vectors/ss, same dot order, `denom=(ssa·ssb).sqrt()`).
fn cdist_row_correlation_soa(
    ci: &[f64],
    ssa: f64,
    cb: &[Vec<f64>],
    ssb: &[f64],
    nb: usize,
) -> Vec<f64> {
    use std::simd::{Select, Simd, StdFloat, cmp::SimdPartialEq};
    const L: usize = 8;
    let d = ci.len();
    let ssav = Simd::<f64, L>::splat(ssa);
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let nan = Simd::<f64, L>::splat(f64::NAN);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let mut dot = Simd::<f64, L>::splat(0.0);
        for k in 0..d {
            dot += Simd::<f64, L>::splat(ci[k]) * Simd::<f64, L>::from_slice(&cb[k][j..j + L]);
        }
        let denom = (ssav * Simd::<f64, L>::from_slice(&ssb[j..j + L])).sqrt();
        let val = one - dot / denom;
        denom
            .simd_eq(zero)
            .select(nan, val)
            .copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let denom = (ssa * ssb[j]).sqrt();
        row[j] = if denom == 0.0 {
            f64::NAN
        } else {
            let mut dot = 0.0_f64;
            for k in 0..d {
                dot += ci[k] * cb[k][j];
            }
            1.0 - dot / denom
        };
        j += 1;
    }
    row
}

/// One dim-4 Cosine cdist output row with precomputed norms (`ni` for `ai`, `nb_norm` for
/// xb). Lane k holds column j+k; per lane `1 - dot/(ni·nj)` and the denom==0⇒NaN guard run
/// identically to the scalar Cosine arm, so BIT-identical while `vdivpd` pipelines the
/// dependent divisions.
fn cdist_row_cosine4(
    ai: &[f64; 4],
    ni: f64,
    b: &[Vec<f64>; 4],
    nb_norm: &[f64],
    nb: usize,
) -> Vec<f64> {
    use std::simd::{Select, Simd, cmp::SimdPartialEq};
    const L: usize = 8;
    let (b0, b1, b2, b3) = (&b[0], &b[1], &b[2], &b[3]);
    let a0 = Simd::<f64, L>::splat(ai[0]);
    let a1 = Simd::<f64, L>::splat(ai[1]);
    let a2 = Simd::<f64, L>::splat(ai[2]);
    let a3 = Simd::<f64, L>::splat(ai[3]);
    let niv = Simd::<f64, L>::splat(ni);
    let one = Simd::<f64, L>::splat(1.0);
    let zero = Simd::<f64, L>::splat(0.0);
    let nan = Simd::<f64, L>::splat(f64::NAN);
    let mut row = vec![0.0_f64; nb];
    let mut j = 0usize;
    while j + L <= nb {
        let bb0 = Simd::<f64, L>::from_slice(&b0[j..j + L]);
        let bb1 = Simd::<f64, L>::from_slice(&b1[j..j + L]);
        let bb2 = Simd::<f64, L>::from_slice(&b2[j..j + L]);
        let bb3 = Simd::<f64, L>::from_slice(&b3[j..j + L]);
        let njv = Simd::<f64, L>::from_slice(&nb_norm[j..j + L]);
        let denom = niv * njv;
        let dot = a0 * bb0 + a1 * bb1 + a2 * bb2 + a3 * bb3;
        let val = one - dot / denom;
        denom
            .simd_eq(zero)
            .select(nan, val)
            .copy_to_slice(&mut row[j..j + L]);
        j += L;
    }
    while j < nb {
        let denom = ni * nb_norm[j];
        row[j] = if denom == 0.0 {
            f64::NAN
        } else {
            let bj = [b0[j], b1[j], b2[j], b3[j]];
            1.0 - dot4(ai, &bj) / denom
        };
        j += 1;
    }
    row
}

/// Worker count for a parallel `pdist`. The n·(n-1)/2 condensed pairs carry `pairs·dim`
/// element-ops. Stay serial unless that clearly exceeds the thread-spawn cost: spawning ~64
/// threads costs ~2.4ms on this contended 64-core box, so small pdists were up to 13x slower
/// than SciPy purely from over-spawn (only the Euclidean/Cosine dim-4 path had a serial
/// guard; cityblock/sqeuclidean/chebyshev fell straight to `cdist_thread_count`'s 1<<18 gate
/// and parallelized trivially-small work). Cap workers at 16 — the per-pair kernels are
/// low-arithmetic-intensity / memory-bandwidth-bound, so ~16 saturate bandwidth and more only
/// contend (same finding as the cdist dim-4 cap, nm8ex). Byte-identical: thread count never
/// changes values (`pdist_fill` fills disjoint contiguous pair ranges in i<j order).
fn pdist_thread_count(n: usize, dim: usize) -> usize {
    if n < 8 {
        return 1;
    }
    let pairs = (n as u64) * (n as u64 - 1) / 2;
    let work = pairs.saturating_mul(dim.max(1) as u64);
    if work < (1 << 20) {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(16).min(n / 4).max(1)
}

/// Compute the full Euclidean distance matrix between two point sets.
///
/// Matches `scipy.spatial.distance_matrix(x, y)`.
///
/// This is an explicit convenience wrapper around Euclidean `cdist`.
pub fn distance_matrix(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SpatialError> {
    cdist_metric(x, y, DistanceMetric::Euclidean)
}

fn rectangle_norm(components: &[f64], p: f64) -> Result<f64, SpatialError> {
    if components.iter().any(|value| !value.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "rectangle distances require finite components".to_owned(),
        ));
    }
    if p == f64::INFINITY {
        return Ok(components
            .iter()
            .copied()
            .fold(0.0_f64, |a: f64, b: f64| a.max(b.abs())));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "rectangle distances require p > 0 or infinity".to_owned(),
        ));
    }
    Ok(components
        .iter()
        .map(|value| value.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p))
}

// ══════════════════════════════════════════════════════════════════════
// Rectangle
// ══════════════════════════════════════════════════════════════════════

/// Hyperrectangle utility matching `scipy.spatial.Rectangle`.
///
/// Represents a Cartesian product of intervals with normalized bounds:
/// `mins[i] <= maxes[i]` for every axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub maxes: Vec<f64>,
    pub mins: Vec<f64>,
    pub m: usize,
}

impl Rectangle {
    /// Construct a hyperrectangle from per-axis maxima and minima.
    pub fn new(maxes: &[f64], mins: &[f64]) -> Result<Self, SpatialError> {
        if maxes.is_empty() || mins.is_empty() {
            return Err(SpatialError::EmptyData);
        }
        if maxes.len() != mins.len() {
            return Err(SpatialError::DimensionMismatch {
                expected: maxes.len(),
                actual: mins.len(),
            });
        }
        if maxes
            .iter()
            .chain(mins.iter())
            .any(|value| !value.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "rectangle bounds must be finite".to_owned(),
            ));
        }

        let mut norm_maxes = Vec::with_capacity(maxes.len());
        let mut norm_mins = Vec::with_capacity(maxes.len());
        for (&max_v, &min_v) in maxes.iter().zip(mins.iter()) {
            norm_maxes.push(max_v.max(min_v));
            norm_mins.push(max_v.min(min_v));
        }

        Ok(Self {
            m: norm_maxes.len(),
            maxes: norm_maxes,
            mins: norm_mins,
        })
    }

    /// Total volume of the hyperrectangle.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.maxes
            .iter()
            .zip(self.mins.iter())
            .map(|(max_v, min_v)| max_v - min_v)
            .product()
    }

    /// Split the hyperrectangle along axis `d` at coordinate `split`.
    pub fn split(&self, d: usize, split: f64) -> Result<(Self, Self), SpatialError> {
        if d >= self.m {
            return Err(SpatialError::InvalidArgument(format!(
                "split axis {d} out of bounds for dimension {}",
                self.m
            )));
        }
        if !split.is_finite() {
            return Err(SpatialError::InvalidArgument(
                "split coordinate must be finite".to_owned(),
            ));
        }

        let mut lower_maxes = self.maxes.clone();
        lower_maxes[d] = split;
        let lower = Self::new(&lower_maxes, &self.mins)?;

        let mut upper_mins = self.mins.clone();
        upper_mins[d] = split;
        let upper = Self::new(&self.maxes, &upper_mins)?;

        Ok((lower, upper))
    }

    /// Minimum distance between the rectangle and a point under Minkowski `p`.
    pub fn min_distance_point(&self, x: &[f64], p: f64) -> Result<f64, SpatialError> {
        if x.len() != self.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: x.len(),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "point coordinates must be finite".to_owned(),
            ));
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(x.iter())
            .map(|((&min_v, &max_v), &coord)| {
                if coord < min_v {
                    min_v - coord
                } else if coord > max_v {
                    coord - max_v
                } else {
                    0.0
                }
            })
            .collect();
        rectangle_norm(&components, p)
    }

    /// Maximum distance between the rectangle and a point under Minkowski `p`.
    pub fn max_distance_point(&self, x: &[f64], p: f64) -> Result<f64, SpatialError> {
        if x.len() != self.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: x.len(),
            });
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "point coordinates must be finite".to_owned(),
            ));
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(x.iter())
            .map(|((&min_v, &max_v), &coord)| (max_v - coord).abs().max((coord - min_v).abs()))
            .collect();
        rectangle_norm(&components, p)
    }

    /// Minimum distance between two rectangles under Minkowski `p`.
    pub fn min_distance_rectangle(&self, other: &Self, p: f64) -> Result<f64, SpatialError> {
        if self.m != other.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: other.m,
            });
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(other.mins.iter().zip(other.maxes.iter()))
            .map(|((&self_min, &self_max), (&other_min, &other_max))| {
                0.0_f64.max((self_min - other_max).max(other_min - self_max))
            })
            .collect();
        rectangle_norm(&components, p)
    }

    /// Maximum distance between two rectangles under Minkowski `p`.
    pub fn max_distance_rectangle(&self, other: &Self, p: f64) -> Result<f64, SpatialError> {
        if self.m != other.m {
            return Err(SpatialError::DimensionMismatch {
                expected: self.m,
                actual: other.m,
            });
        }

        let components: Vec<f64> = self
            .mins
            .iter()
            .zip(self.maxes.iter())
            .zip(other.mins.iter().zip(other.maxes.iter()))
            .map(|((&self_min, &self_max), (&other_min, &other_max))| {
                (self_max - other_min)
                    .abs()
                    .max((other_max - self_min).abs())
            })
            .collect();
        rectangle_norm(&components, p)
    }
}

// ══════════════════════════════════════════════════════════════════════
// KDTree
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub struct SparseDistanceMatrixRecord {
    pub i: usize,
    pub j: usize,
    pub v: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SparseDistanceMatrixOutput {
    DokMatrix(DokMatrix),
    CooMatrix(CooMatrix),
    Dict(BTreeMap<(usize, usize), f64>),
    Ndarray(Vec<SparseDistanceMatrixRecord>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SparseDistanceMatrixOutputType {
    DokMatrix,
    CooMatrix,
    Dict,
    Ndarray,
}

impl SparseDistanceMatrixOutputType {
    fn parse(output_type: &str) -> Result<Self, SpatialError> {
        match output_type {
            "dok_matrix" => Ok(Self::DokMatrix),
            "coo_matrix" => Ok(Self::CooMatrix),
            "dict" => Ok(Self::Dict),
            "ndarray" => Ok(Self::Ndarray),
            _ => Err(SpatialError::InvalidArgument(
                "Invalid output type".to_string(),
            )),
        }
    }
}

fn sparse_distance_matrix_output_from_triplets(
    shape: Shape2D,
    triplets: &[(usize, usize, f64)],
    output_type: SparseDistanceMatrixOutputType,
) -> Result<SparseDistanceMatrixOutput, SpatialError> {
    let mut rows = Vec::with_capacity(triplets.len());
    let mut cols = Vec::with_capacity(triplets.len());
    let mut values = Vec::with_capacity(triplets.len());
    for &(row, col, value) in triplets {
        rows.push(row);
        cols.push(col);
        values.push(value);
    }

    match output_type {
        SparseDistanceMatrixOutputType::DokMatrix => {
            let matrix = DokMatrix::from_triplets(shape, values, rows, cols)
                .map_err(sparse_error_to_spatial)?;
            Ok(SparseDistanceMatrixOutput::DokMatrix(matrix))
        }
        SparseDistanceMatrixOutputType::CooMatrix => {
            let matrix = CooMatrix::from_triplets(shape, values, rows, cols, false)
                .map_err(sparse_error_to_spatial)?;
            Ok(SparseDistanceMatrixOutput::CooMatrix(matrix))
        }
        SparseDistanceMatrixOutputType::Dict => {
            let entries = triplets
                .iter()
                .map(|&(row, col, value)| ((row, col), value))
                .collect();
            Ok(SparseDistanceMatrixOutput::Dict(entries))
        }
        SparseDistanceMatrixOutputType::Ndarray => {
            let entries = triplets
                .iter()
                .map(|&(i, j, v)| SparseDistanceMatrixRecord { i, j, v })
                .collect();
            Ok(SparseDistanceMatrixOutput::Ndarray(entries))
        }
    }
}

/// A k-d tree for efficient nearest-neighbor search.
///
/// Matches `scipy.spatial.KDTree(data)`.
#[derive(Debug, Clone)]
pub struct KDTree {
    nodes: Vec<KDNode>,
    dim: usize,
    /// Node-ordered contiguous coordinate slab: node `i`'s point occupies
    /// `points[i*dim .. (i+1)*dim]`. The k-NN traversal reads coordinates from
    /// here instead of chasing each node's scattered heap `Vec<f64>` — for a
    /// large high-dimensional tree the per-node pointer chase (one cache miss per
    /// visited node, and backtracking visits many) dominated the query.
    points: Vec<f64>,
}

#[derive(Debug, Clone)]
struct KDNode {
    index: usize,
    left: Option<usize>,
    right: Option<usize>,
    split_dim: usize,
}

impl KDTree {
    /// Build a k-d tree from a set of points.
    ///
    /// Each point is a `dim`-dimensional vector.
    pub fn new(data: &[Vec<f64>]) -> Result<Self, SpatialError> {
        Self::new_with_mode(data, RuntimeMode::Strict, None)
    }

    /// Build a k-d tree under an explicit runtime policy with optional audit ledger.
    pub fn new_with_mode(
        data: &[Vec<f64>],
        mode: RuntimeMode,
        audit_ledger: Option<&SyncSharedAuditLedger>,
    ) -> Result<Self, SpatialError> {
        if matches!(mode, RuntimeMode::Hardened) {
            if data.len() > HARDENED_MAX_DIM
                || (!data.is_empty() && data[0].len() > HARDENED_MAX_DIM)
            {
                if let Some(ledger) = audit_ledger {
                    record_fail_closed(
                        ledger,
                        &data.len().to_le_bytes(),
                        "dimension exceeds hardened limit",
                        "rejected",
                    );
                }
                return Err(SpatialError::InvalidArgument(format!(
                    "data count or dimension exceeds hardened limit ({HARDENED_MAX_DIM})"
                )));
            }
        }
        if data.is_empty() {
            return Err(SpatialError::EmptyData);
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(SpatialError::InvalidArgument(
                "points must have at least 1 dimension".to_string(),
            ));
        }
        if let Some(actual) = data.iter().map(Vec::len).find(|&len| len != dim) {
            return Err(SpatialError::DimensionMismatch {
                expected: dim,
                actual,
            });
        }
        if data.iter().flatten().any(|value| !value.is_finite()) {
            if matches!(mode, RuntimeMode::Hardened) {
                if let Some(ledger) = audit_ledger {
                    record_fail_closed(
                        ledger,
                        b"non-finite-points",
                        "points must be finite in hardened mode",
                        "rejected",
                    );
                }
            }
            return Err(SpatialError::InvalidArgument(
                "points must be finite".to_string(),
            ));
        }

        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut nodes = Vec::with_capacity(data.len());
        // Flatten points into one contiguous row-major buffer up front: the
        // O(n log n) build partitions read `coords[idx*dim + split_dim]` instead
        // of chasing a scattered per-point `Vec<f64>` pointer on every compare.
        // The build then writes the selected points once into node order, which
        // is the sole coordinate storage for every query path.
        let coords: Vec<f64> = data.iter().flatten().copied().collect();
        let mut points = Vec::with_capacity(data.len() * dim);
        build_kdtree(&coords, &mut indices, 0, &mut nodes, &mut points, dim);

        Ok(Self { nodes, dim, points })
    }

    #[inline]
    fn node_point(&self, node_idx: usize) -> &[f64] {
        &self.points[node_idx * self.dim..node_idx * self.dim + self.dim]
    }

    /// Find the nearest neighbor to `query`.
    /// Returns `(index, distance)`.
    pub fn query(&self, query: &[f64]) -> Result<(usize, f64), SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Err(SpatialError::EmptyData);
        }

        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        nn_search(
            &self.nodes,
            &self.points,
            self.dim,
            0,
            query,
            &mut best_idx,
            &mut best_dist,
        );

        Ok((best_idx, best_dist.sqrt()))
    }

    /// Nearest-neighbour query for a batch of points — matches
    /// `scipy.spatial.cKDTree.query(X, k=1)`. Each query is an independent,
    /// read-only tree traversal (`nn_search`), so the batch is parallelized
    /// across query points; results are returned in input order and are
    /// bit-for-bit identical to calling [`KDTree::query`] on each point (same
    /// traversal, same `sqrt`). Every query is validated up front so the
    /// parallel section cannot fault. Tree traversal is latency-bound
    /// (pointer-chasing), so — unlike bandwidth-bound 1-D scans — the parallel
    /// fan-out scales.
    pub fn query_many(&self, queries: &[Vec<f64>]) -> Result<Vec<(usize, f64)>, SpatialError> {
        if self.nodes.is_empty() {
            return Err(SpatialError::EmptyData);
        }
        for q in queries {
            if q.len() != self.dim {
                return Err(SpatialError::DimensionMismatch {
                    expected: self.dim,
                    actual: q.len(),
                });
            }
            if q.iter().any(|value| !value.is_finite()) {
                return Err(SpatialError::InvalidArgument(
                    "query must be finite".to_string(),
                ));
            }
        }
        let nq = queries.len();
        let mut out = vec![(0usize, 0.0f64); nq];
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        // Single-NN backtracking grows fast with dimension: at higher dim each
        // query is compute-heavy, so let it use all cores (a flat 16-cap left d=8
        // ~4x slower than SciPy workers=-1); low-dim queries are cheap and would
        // over-spawn, so keep the 16 ceiling and ≥128 queries/thread there.
        let thread_ceiling = if self.dim >= 6 { cores } else { cores.min(16) };
        let nthreads = if nq >= 512 {
            thread_ceiling.min(nq / 128).max(1)
        } else {
            1
        };
        let nodes = &self.nodes;
        let points = &self.points;
        let dim = self.dim;
        let eval = |slot: &mut (usize, f64), q: &[f64]| {
            let mut bi = 0;
            let mut bd = f64::INFINITY;
            nn_search(nodes, points, dim, 0, q, &mut bi, &mut bd);
            *slot = (bi, bd.sqrt());
        };
        if nthreads <= 1 {
            for (slot, q) in out.iter_mut().zip(queries) {
                eval(slot, q);
            }
            return Ok(out);
        }
        let chunk = nq.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (qchunk, ochunk) in queries.chunks(chunk).zip(out.chunks_mut(chunk)) {
                s.spawn(move || {
                    for (slot, q) in ochunk.iter_mut().zip(qchunk) {
                        let mut bi = 0;
                        let mut bd = f64::INFINITY;
                        nn_search(nodes, points, dim, 0, q, &mut bi, &mut bd);
                        *slot = (bi, bd.sqrt());
                    }
                });
            }
        });
        Ok(out)
    }

    /// Find the k nearest neighbors to `query`.
    /// Returns vectors of (index, distance) sorted by distance.
    pub fn query_k(&self, query: &[f64], k: usize) -> Result<Vec<(usize, f64)>, SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(self.nodes.len());
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(k);

        // Simple approach: find k nearest via repeated queries with exclusion
        // For a proper implementation, use a max-heap bounded to k elements.
        knn_search(
            &self.nodes,
            &self.points,
            self.dim,
            0,
            query,
            k,
            &mut results,
        );

        // Sort by distance and convert from squared to actual distance
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        for r in &mut results {
            r.1 = r.1.sqrt();
        }

        Ok(results)
    }

    /// k-nearest-neighbour query for a batch of points — matches
    /// `scipy.spatial.cKDTree.query(X, k)`. Each query runs the same independent,
    /// read-only `knn_search` + total_cmp sort + sqrt as [`KDTree::query_k`], so
    /// the batch is parallelized across query points and the per-query result is
    /// bit-for-bit identical (same neighbours, same order, same distance bits).
    /// Results are returned in input order; all queries are validated up front.
    pub fn query_k_many(
        &self,
        queries: &[Vec<f64>],
        k: usize,
    ) -> Result<Vec<Vec<(usize, f64)>>, SpatialError> {
        for q in queries {
            if q.len() != self.dim {
                return Err(SpatialError::DimensionMismatch {
                    expected: self.dim,
                    actual: q.len(),
                });
            }
            if q.iter().any(|value| !value.is_finite()) {
                return Err(SpatialError::InvalidArgument(
                    "query must be finite".to_string(),
                ));
            }
        }
        let nq = queries.len();
        let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nq];
        if k == 0 {
            return Ok(out);
        }
        let kk = k.min(self.nodes.len());
        let nodes = &self.nodes;
        let points = &self.points;
        let dim = self.dim;
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        // k-NN per-query cost grows fast with dimension (more backtracking), so
        // scale threads with the *work* (queries × dim × k), not a flat cap. A
        // fixed 16-thread ceiling left high-dimensional batches (e.g. d=8) ~3.5x
        // slower than SciPy's `workers=-1`; here we let compute-heavy batches use
        // all cores while keeping ≥32 queries/thread so cheap low-d batches don't
        // over-spawn.
        let per_query_work = self.dim.saturating_mul(kk.max(1));
        let thread_ceiling = if per_query_work >= 24 {
            cores
        } else {
            cores.min(16)
        };
        let nthreads = if nq >= 128 {
            thread_ceiling.min(nq / 32).max(1)
        } else {
            1
        };
        if nthreads <= 1 {
            for (slot, q) in out.iter_mut().zip(queries) {
                let mut results: Vec<(usize, f64)> = Vec::with_capacity(kk);
                knn_search(nodes, points, dim, 0, q, kk, &mut results);
                results.sort_by(|a, b| a.1.total_cmp(&b.1));
                for r in &mut results {
                    r.1 = r.1.sqrt();
                }
                *slot = results;
            }
            return Ok(out);
        }
        let chunk = nq.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (qchunk, ochunk) in queries.chunks(chunk).zip(out.chunks_mut(chunk)) {
                s.spawn(move || {
                    for (slot, q) in ochunk.iter_mut().zip(qchunk) {
                        let mut results: Vec<(usize, f64)> = Vec::with_capacity(kk);
                        knn_search(nodes, points, dim, 0, q, kk, &mut results);
                        results.sort_by(|a, b| a.1.total_cmp(&b.1));
                        for r in &mut results {
                            r.1 = r.1.sqrt();
                        }
                        *slot = results;
                    }
                });
            }
        });
        Ok(out)
    }

    /// Number of points in the tree.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Find all points within distance `r` of a query point.
    ///
    /// Matches `scipy.spatial.KDTree.query_ball_point(x, r)`.
    ///
    /// Returns indices of all points within Euclidean distance `r`.
    pub fn query_ball_point(&self, query: &[f64], r: f64) -> Result<Vec<usize>, SpatialError> {
        if query.len() != self.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(
                "query must be finite".to_string(),
            ));
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let r_sq = r * r;
        let mut results = Vec::new();
        ball_search(
            &self.nodes,
            &self.points,
            self.dim,
            0,
            query,
            r_sq,
            &mut results,
        );
        results.sort_unstable();
        Ok(results)
    }

    /// Radius (ball) query for a batch of points — matches
    /// `scipy.spatial.cKDTree.query_ball_point(X, r)`. Each query runs the same
    /// independent, read-only `ball_search` + `sort_unstable` as
    /// [`KDTree::query_ball_point`], parallelized across query points; the
    /// per-query index list is bit-for-bit identical (same members, same sorted
    /// order) and returned in input order. All queries and `r` are validated up
    /// front so the parallel section cannot fault.
    pub fn query_ball_point_many(
        &self,
        queries: &[Vec<f64>],
        r: f64,
    ) -> Result<Vec<Vec<usize>>, SpatialError> {
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        for q in queries {
            if q.len() != self.dim {
                return Err(SpatialError::DimensionMismatch {
                    expected: self.dim,
                    actual: q.len(),
                });
            }
            if q.iter().any(|value| !value.is_finite()) {
                return Err(SpatialError::InvalidArgument(
                    "query must be finite".to_string(),
                ));
            }
        }
        let nq = queries.len();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); nq];
        if self.nodes.is_empty() {
            return Ok(out);
        }
        let r_sq = r * r;
        let nodes = &self.nodes;
        let points = &self.points;
        let dim = self.dim;
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        // Each radius query can touch many nodes and sort its hit list, so it is
        // compute-heavy — a low gate amortizes spawn well.
        let nthreads = if nq >= 64 {
            cores.min(16).min(nq / 16).max(1)
        } else {
            1
        };
        if nthreads <= 1 {
            for (slot, q) in out.iter_mut().zip(queries) {
                ball_search(nodes, points, dim, 0, q, r_sq, slot);
                slot.sort_unstable();
            }
            return Ok(out);
        }
        let chunk = nq.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (qchunk, ochunk) in queries.chunks(chunk).zip(out.chunks_mut(chunk)) {
                s.spawn(move || {
                    for (slot, q) in ochunk.iter_mut().zip(qchunk) {
                        ball_search(nodes, points, dim, 0, q, r_sq, slot);
                        slot.sort_unstable();
                    }
                });
            }
        });
        Ok(out)
    }

    /// Find all cross-tree neighbors within distance `r`.
    ///
    /// Matches `scipy.spatial.KDTree.query_ball_tree(other, r)`.
    ///
    /// Returns one neighbor list per point in `self`, where each inner vector
    /// contains indices from `other` within Euclidean distance `r`.
    pub fn query_ball_tree(&self, other: &KDTree, r: f64) -> Result<Vec<Vec<usize>>, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let n = self.nodes.len();
        // Each point's neighbour list is an independent query against `other` and is stored
        // at its own `node.index`, so the queries parallelize bit-identically.
        let nthreads = cdist_thread_count(n, other.nodes.len(), self.dim);
        if nthreads <= 1 {
            let mut results = vec![Vec::new(); n];
            for (node_idx, node) in self.nodes.iter().enumerate() {
                results[node.index] = other.query_ball_point(self.node_point(node_idx), r)?;
            }
            return Ok(results);
        }
        let chunk = n.div_ceil(nthreads);
        let points = &self.points;
        let dim = self.dim;
        type Computed = Result<Vec<(usize, Vec<usize>)>, SpatialError>;
        let computed: Vec<Computed> = std::thread::scope(|scope| {
            let handles: Vec<_> = self
                .nodes
                .chunks(chunk)
                .enumerate()
                .map(|(chunk_idx, chunk_nodes)| {
                    let first_node = chunk_idx * chunk;
                    scope.spawn(move || {
                        chunk_nodes
                            .iter()
                            .enumerate()
                            .map(|(offset, nd)| {
                                other
                                    .query_ball_point(
                                        &points[(first_node + offset) * dim
                                            ..(first_node + offset + 1) * dim],
                                        r,
                                    )
                                    .map(|res| (nd.index, res))
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("query_ball_tree worker panicked"))
                .collect()
        });
        let mut results = vec![Vec::new(); n];
        for chunk_res in computed {
            for (idx, res) in chunk_res? {
                results[idx] = res;
            }
        }
        Ok(results)
    }

    /// Find all unique in-tree point pairs within distance `r`.
    ///
    /// Matches `scipy.spatial.KDTree.query_pairs(r)`.
    ///
    /// Returns lexicographically sorted `(i, j)` pairs with `i < j`.
    pub fn query_pairs(&self, r: f64) -> Result<Vec<(usize, usize)>, SpatialError> {
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.len() < 2 {
            return Ok(Vec::new());
        }

        // Each anchor's pairs are an independent query; the final sort makes the collected
        // set order-independent, so gathering per-thread pair lists is bit-identical.
        let n = self.nodes.len();
        let nthreads = cdist_thread_count(n, n, self.dim);
        let mut pairs = if nthreads <= 1 {
            let mut pairs = Vec::new();
            for (node_idx, node) in self.nodes.iter().enumerate() {
                for neighbor_index in self.query_ball_point(self.node_point(node_idx), r)? {
                    if neighbor_index > node.index {
                        pairs.push((node.index, neighbor_index));
                    }
                }
            }
            pairs
        } else {
            let tree: &KDTree = self;
            let chunk = n.div_ceil(nthreads);
            let points = &self.points;
            let dim = self.dim;
            type Computed = Result<Vec<(usize, usize)>, SpatialError>;
            let computed: Vec<Computed> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .nodes
                    .chunks(chunk)
                    .enumerate()
                    .map(|(chunk_idx, chunk_nodes)| {
                        let first_node = chunk_idx * chunk;
                        scope.spawn(move || {
                            let mut local = Vec::new();
                            for (offset, node) in chunk_nodes.iter().enumerate() {
                                let point = &points
                                    [(first_node + offset) * dim..(first_node + offset + 1) * dim];
                                for neighbor_index in tree.query_ball_point(point, r)? {
                                    if neighbor_index > node.index {
                                        local.push((node.index, neighbor_index));
                                    }
                                }
                            }
                            Ok(local)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("query_pairs worker panicked"))
                    .collect()
            });
            let mut pairs = Vec::new();
            for chunk_res in computed {
                pairs.extend(chunk_res?);
            }
            pairs
        };
        pairs.sort_unstable();
        Ok(pairs)
    }

    /// Count pairs of points within distance `r` between this tree and another.
    ///
    /// Matches `scipy.spatial.KDTree.count_neighbors(other, r)`.
    pub fn count_neighbors(&self, other: &KDTree, r: f64) -> Result<usize, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !r.is_finite() || r < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "radius must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return Ok(0);
        }
        let r_sq = r * r;
        // Each query into `self` is an independent tree descent, and the total is an exact
        // integer sum (associative/commutative), so splitting `other`'s points across
        // threads and summing the partials is bit-identical to the serial accumulation.
        let m = other.nodes.len();
        let nthreads = cdist_thread_count(m, self.nodes.len(), self.dim);
        if nthreads <= 1 {
            let mut count = 0;
            for node_idx in 0..other.nodes.len() {
                count += ball_search_count(
                    &self.nodes,
                    &self.points,
                    self.dim,
                    0,
                    other.node_point(node_idx),
                    r_sq,
                );
            }
            return Ok(count);
        }
        let self_nodes = self.nodes.as_slice();
        let self_points = self.points.as_slice();
        let other_points = other.points.as_slice();
        let dim = self.dim;
        let chunk = m.div_ceil(nthreads);
        let count: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = other
                .nodes
                .chunks(chunk)
                .enumerate()
                .map(|(chunk_idx, chunk_nodes)| {
                    let first_node = chunk_idx * chunk;
                    scope.spawn(move || {
                        chunk_nodes
                            .iter()
                            .enumerate()
                            .map(|(offset, _)| {
                                ball_search_count(
                                    self_nodes,
                                    self_points,
                                    dim,
                                    0,
                                    &other_points[(first_node + offset) * dim
                                        ..(first_node + offset + 1) * dim],
                                    r_sq,
                                )
                            })
                            .sum::<usize>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("count_neighbors worker panicked"))
                .sum()
        });
        Ok(count)
    }

    /// Compute a sparse cross-distance matrix for pairs within `max_distance`.
    ///
    /// Matches the default `scipy.spatial.KDTree.sparse_distance_matrix`
    /// container semantics by returning a DOK-style sparse matrix.
    pub fn sparse_distance_matrix(
        &self,
        other: &KDTree,
        max_distance: f64,
    ) -> Result<DokMatrix, SpatialError> {
        let output =
            self.sparse_distance_matrix_with_output_type(other, max_distance, "dok_matrix")?;
        match output {
            SparseDistanceMatrixOutput::DokMatrix(matrix) => Ok(matrix),
            _ => unreachable!("dok_matrix output type must produce a DokMatrix"),
        }
    }

    /// Compute a sparse cross-distance matrix using SciPy's `output_type` modes.
    ///
    /// Supported output types are `"dok_matrix"`, `"coo_matrix"`, `"dict"`,
    /// and `"ndarray"`. Distances use the Euclidean metric.
    pub fn sparse_distance_matrix_with_output_type(
        &self,
        other: &KDTree,
        max_distance: f64,
        output_type: &str,
    ) -> Result<SparseDistanceMatrixOutput, SpatialError> {
        let parsed_output_type = SparseDistanceMatrixOutputType::parse(output_type)?;
        let triplets = self.sparse_distance_matrix_triplets(other, max_distance)?;
        sparse_distance_matrix_output_from_triplets(
            Shape2D::new(self.size(), other.size()),
            &triplets,
            parsed_output_type,
        )
    }

    fn sparse_distance_matrix_triplets(
        &self,
        other: &KDTree,
        max_distance: f64,
    ) -> Result<Vec<(usize, usize, f64)>, SpatialError> {
        if self.dim != other.dim {
            return Err(SpatialError::DimensionMismatch {
                expected: self.dim,
                actual: other.dim,
            });
        }
        if !max_distance.is_finite() || max_distance < 0.0 {
            return Err(SpatialError::InvalidArgument(
                "max_distance must be finite and nonnegative".to_string(),
            ));
        }
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let mut other_points = vec![None; other.nodes.len()];
        for (node_idx, node) in other.nodes.iter().enumerate() {
            other_points[node.index] = Some(other.node_point(node_idx));
        }

        // Each `self` node's neighbour query against `other` is independent and the
        // entries are sorted by (row, col) at the end (keys are unique — each (i,j)
        // pair occurs once), so the final order is independent of collection order.
        // Parallelize the outer query loop across `self`'s points (mirrors the
        // already-parallel `query_ball_tree` / `count_neighbors`); bit-identical.
        type SparseTriplet = (usize, usize, f64);
        type SparseTripletResult = Result<Vec<SparseTriplet>, SpatialError>;
        let op = &other_points;
        let collect_chunk = |first_node: usize, chunk_nodes: &[KDNode]| -> SparseTripletResult {
            let mut local = Vec::new();
            for (offset, node) in chunk_nodes.iter().enumerate() {
                let point = self.node_point(first_node + offset);
                for other_index in other.query_ball_point(point, max_distance)? {
                    let Some(other_point) = op[other_index] else {
                        return Err(SpatialError::InvalidArgument(
                            "kdtree internal point index mapping was inconsistent".to_string(),
                        ));
                    };
                    local.push((
                        node.index,
                        other_index,
                        sqeuclidean(point, other_point).sqrt(),
                    ));
                }
            }
            Ok(local)
        };
        let n = self.nodes.len();
        let nthreads = cdist_thread_count(n, other.nodes.len(), self.dim);
        let mut entries = if nthreads <= 1 {
            collect_chunk(0, &self.nodes)?
        } else {
            let chunk = n.div_ceil(nthreads);
            let collect_chunk = &collect_chunk;
            let parts: Vec<SparseTripletResult> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .nodes
                    .chunks(chunk)
                    .enumerate()
                    .map(|(chunk_idx, cn)| {
                        scope.spawn(move || collect_chunk(chunk_idx * chunk, cn))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("sparse_distance_matrix worker panicked"))
                    .collect()
            });
            let mut all = Vec::new();
            for part in parts {
                all.extend(part?);
            }
            all
        };
        entries.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        Ok(entries)
    }
}

/// SciPy parity alias for `KDTree`.
///
/// SciPy exposes both `KDTree` and `cKDTree`. FrankenSciPy uses one Rust
/// implementation for both surfaces.
#[allow(non_camel_case_types)]
pub type cKDTree = KDTree;

fn ball_search_count(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    r_sq: f64,
) -> usize {
    let node = &nodes[node_idx];
    let point = &points[node_idx * dim..node_idx * dim + dim];
    let dist_sq = sqeuclidean(query, point);

    let mut count = if dist_sq <= r_sq { 1 } else { 0 };

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        count += ball_search_count(nodes, points, dim, near_idx, query, r_sq);
    }

    if diff * diff <= r_sq
        && let Some(far_idx) = far
    {
        count += ball_search_count(nodes, points, dim, far_idx, query, r_sq);
    }

    count
}

fn build_kdtree(
    coords: &[f64],
    indices: &mut [usize],
    depth: usize,
    nodes: &mut Vec<KDNode>,
    points: &mut Vec<f64>,
    dim: usize,
) -> Option<usize> {
    if indices.is_empty() {
        return None;
    }

    let split_dim = depth % dim;

    let median = indices.len() / 2;

    // Partition by the split dimension to find the median in O(N) time
    indices.select_nth_unstable_by(median, |&a, &b| {
        coords[a * dim + split_dim].total_cmp(&coords[b * dim + split_dim])
    });

    let node_idx = nodes.len();
    let point_idx = indices[median];

    nodes.push(KDNode {
        index: point_idx,
        left: None,
        right: None,
        split_dim,
    });
    points.extend_from_slice(&coords[point_idx * dim..][..dim]);

    let (left_indices, right_part) = indices.split_at_mut(median);
    let right_indices = &mut right_part[1..]; // skip median

    let left = build_kdtree(coords, left_indices, depth + 1, nodes, points, dim);
    let right = build_kdtree(coords, right_indices, depth + 1, nodes, points, dim);

    nodes[node_idx].left = left;
    nodes[node_idx].right = right;

    Some(node_idx)
}

fn nn_search(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    best_idx: &mut usize,
    best_dist_sq: &mut f64,
) {
    let node = &nodes[node_idx];
    // Contiguous node-ordered slab instead of the scattered per-node heap Vec.
    let point = &points[node_idx * dim..node_idx * dim + dim];
    let dist_sq = sqeuclidean(query, point);

    if dist_sq < *best_dist_sq {
        *best_dist_sq = dist_sq;
        *best_idx = node.index;
    }

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        nn_search(nodes, points, dim, near_idx, query, best_idx, best_dist_sq);
    }

    // Check if the other subtree could contain a closer point
    if diff * diff < *best_dist_sq
        && let Some(far_idx) = far
    {
        nn_search(nodes, points, dim, far_idx, query, best_idx, best_dist_sq);
    }
}

fn knn_search(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    k: usize,
    results: &mut Vec<(usize, f64)>,
) {
    let node = &nodes[node_idx];
    // Read this node's coordinates from the contiguous node-ordered slab rather
    // than the scattered per-node heap `Vec` (cache-friendly on deep backtracks).
    let point = &points[node_idx * dim..node_idx * dim + dim];
    let dist_sq = sqeuclidean(query, point);

    // Insert if we have room or this is closer than the worst
    let is_better = results
        .last()
        .is_none_or(|&(_, worst_dist)| dist_sq < worst_dist);
    if results.len() < k || is_better {
        let pos = results
            .binary_search_by(|probe| probe.1.total_cmp(&dist_sq))
            .unwrap_or_else(|e| e);
        if results.len() < k {
            results.insert(pos, (node.index, dist_sq));
        } else if pos < k {
            results.pop();
            results.insert(pos, (node.index, dist_sq));
        }
    }

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        knn_search(nodes, points, dim, near_idx, query, k, results);
    }

    let worst_dist = if results.len() < k {
        f64::INFINITY
    } else {
        results[k - 1].1
    };

    if diff * diff < worst_dist
        && let Some(far_idx) = far
    {
        knn_search(nodes, points, dim, far_idx, query, k, results);
    }
}

fn ball_search(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    r_sq: f64,
    results: &mut Vec<usize>,
) {
    let node = &nodes[node_idx];
    let point = &points[node_idx * dim..node_idx * dim + dim];
    let dist_sq = sqeuclidean(query, point);

    if dist_sq <= r_sq {
        results.push(node.index);
    }

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        ball_search(nodes, points, dim, near_idx, query, r_sq, results);
    }

    if diff * diff <= r_sq
        && let Some(far_idx) = far
    {
        ball_search(nodes, points, dim, far_idx, query, r_sq, results);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Convex Hull (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D convex hull computation.
///
/// Matches `scipy.spatial.ConvexHull(points)` for 2D point sets.
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Indices of input points that form the convex hull (in CCW order).
    pub vertices: Vec<usize>,
    /// Hull edges as pairs of vertex indices (into the original points array).
    pub simplices: Vec<(usize, usize)>,
    /// Area enclosed by the convex hull.
    pub area: f64,
    /// Perimeter of the convex hull (called "volume" in SciPy for consistency
    /// with higher dimensions, but for 2D it's the perimeter).
    pub perimeter: f64,
}

impl ConvexHull {
    /// Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
    ///
    /// Time complexity: O(n log n).
    ///
    /// # Arguments
    /// * `points` — Slice of (x, y) coordinate pairs.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        if points.len() < 3 {
            return Err(qhull_error(
                "QH6214 qhull input error: not enough points to construct initial simplex",
            ));
        }

        let n = points.len();

        // Sort points by x, then by y (lexicographic)
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            points[a]
                .0
                .total_cmp(&points[b].0)
                .then(points[a].1.total_cmp(&points[b].1))
        });

        // Build lower hull
        let mut lower: Vec<usize> = Vec::new();
        for &idx in &sorted_indices {
            while lower.len() >= 2
                && cross(
                    points[lower[lower.len() - 2]],
                    points[lower[lower.len() - 1]],
                    points[idx],
                ) <= 0.0
            {
                lower.pop();
            }
            lower.push(idx);
        }

        // Build upper hull
        let mut upper: Vec<usize> = Vec::new();
        for &idx in sorted_indices.iter().rev() {
            while upper.len() >= 2
                && cross(
                    points[upper[upper.len() - 2]],
                    points[upper[upper.len() - 1]],
                    points[idx],
                ) <= 0.0
            {
                upper.pop();
            }
            upper.push(idx);
        }

        // Remove last point of each half because it's repeated
        lower.pop();
        upper.pop();

        let mut vertices: Vec<usize> = lower;
        vertices.extend(upper);

        // Handle degenerate cases (all collinear)
        if vertices.len() < 3 {
            return Err(qhull_error(
                "QH6154 Qhull precision error: initial simplex is flat",
            ));
        }

        // Compute simplices (edges)
        let nv = vertices.len();
        let simplices: Vec<(usize, usize)> = (0..nv)
            .map(|i| (vertices[i], vertices[(i + 1) % nv]))
            .collect();

        // Compute area using the shoelace formula
        let mut area = 0.0;
        for i in 0..nv {
            let j = (i + 1) % nv;
            let pi = points[vertices[i]];
            let pj = points[vertices[j]];
            area += pi.0 * pj.1 - pj.0 * pi.1;
        }
        area = area.abs() / 2.0;

        // Compute perimeter
        let mut perimeter = 0.0;
        for &(a, b) in &simplices {
            let dx = points[b].0 - points[a].0;
            let dy = points[b].1 - points[a].1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        Ok(Self {
            vertices,
            simplices,
            area,
            perimeter,
        })
    }
}

// ══════════════════════════════════════════════════════════════════════
// Halfspace Intersection
// ══════════════════════════════════════════════════════════════════════

type Point2 = (f64, f64);
type Halfspace2 = [f64; 3];
type Equation2 = [f64; 3];
type NdVertices = Vec<Vec<f64>>;
type NdFacets = Vec<Vec<usize>>;
type NdVertexCandidate = (Vec<f64>, Vec<usize>);

/// Halfspace intersection result for bounded N-D halfspaces.
///
/// Mirrors the core public attributes of
/// `scipy.spatial.HalfspaceIntersection(halfspaces, interior_point)`. The 2D
/// constructor retains the richer dual-hull metadata surface; `from_nd`
/// generalizes bounded intersections to higher dimensions with vector-backed
/// fields while leaving `dual_area`, `dual_volume`, and `dual_equations`
/// populated only for the 2D fast path.
#[derive(Debug, Clone)]
pub struct HalfspaceIntersection {
    /// Input halfspaces in SciPy row format `[a_0, ..., a_{n-1}, b]`.
    pub halfspaces: Vec<Vec<f64>>,
    /// Feasible point that must be strictly inside every halfspace.
    pub interior_point: Vec<f64>,
    /// Primal intersection vertices.
    pub intersections: Vec<Vec<f64>>,
    /// Qhull-style dual points `-A / (A * interior_point + b)`.
    pub dual_points: Vec<Vec<f64>>,
    /// Indices of dual points forming each dual convex-hull facet.
    pub dual_facets: Vec<Vec<usize>>,
    /// Indices of halfspaces that form the dual convex-hull vertices.
    pub dual_vertices: Vec<usize>,
    /// Dual hull facet equations. Populated only for the 2D fast path.
    pub dual_equations: Vec<Vec<f64>>,
    /// Perimeter of the 2D dual convex hull, matching SciPy's `dual_area`.
    /// `NaN` for higher-dimensional bounded intersections.
    pub dual_area: f64,
    /// Area of the 2D dual convex hull, matching SciPy's `dual_volume`.
    /// `NaN` for higher-dimensional bounded intersections.
    pub dual_volume: f64,
    /// Spatial dimension.
    pub ndim: usize,
    /// Number of input inequalities.
    pub nineq: usize,
    /// Whether the primal feasible region is bounded.
    pub is_bounded: bool,
}

impl HalfspaceIntersection {
    /// Compute the intersection of 2D halfspaces.
    pub fn new(halfspaces: &[Halfspace2], interior_point: Point2) -> Result<Self, SpatialError> {
        validate_halfspace_intersection_inputs(halfspaces, interior_point)?;

        let dual_points = halfspace_dual_points(halfspaces, interior_point);
        let dual_hull = ConvexHull::new(&dual_points).map_err(|err| match err {
            SpatialError::Qhull(qhull) => SpatialError::Qhull(qhull),
            other => qhull_error(format!(
                "Qhull failed to construct halfspace dual hull: {other}"
            )),
        })?;

        let dual_facets = dual_hull
            .simplices
            .iter()
            .map(|&(a, b)| vec![a, b])
            .collect::<Vec<_>>();
        let dual_equations = dual_hull
            .simplices
            .iter()
            .map(|&(a, b)| dual_edge_equation(dual_points[a], dual_points[b]))
            .collect::<Vec<_>>();
        let intersections = dual_equations
            .iter()
            .map(|equation| intersection_from_dual_equation(*equation, interior_point))
            .collect::<Vec<_>>();

        Ok(Self {
            halfspaces: halfspaces.iter().map(|row| row.to_vec()).collect(),
            interior_point: vec![interior_point.0, interior_point.1],
            intersections: intersections.into_iter().map(|(x, y)| vec![x, y]).collect(),
            dual_points: dual_points.into_iter().map(|(x, y)| vec![x, y]).collect(),
            dual_facets,
            dual_vertices: dual_hull.vertices,
            dual_equations: dual_equations.into_iter().map(|eq| eq.to_vec()).collect(),
            dual_area: dual_hull.perimeter,
            dual_volume: dual_hull.area,
            ndim: 2,
            nineq: halfspaces.len(),
            is_bounded: halfspace_region_is_bounded(halfspaces),
        })
    }

    /// Construct from SciPy-shaped rows.
    pub fn from_nd(halfspaces: &[Vec<f64>], interior_point: &[f64]) -> Result<Self, SpatialError> {
        validate_halfspace_intersection_inputs_nd(halfspaces, interior_point)?;
        let ndim = interior_point.len();
        if ndim == 2 {
            let rows = halfspaces
                .iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect::<Vec<_>>();
            return Self::new(&rows, (interior_point[0], interior_point[1]));
        }

        let is_bounded = halfspace_region_is_bounded_nd(halfspaces, ndim);
        let dual_points = halfspace_dual_points_nd(halfspaces, interior_point);
        let (intersections, dual_facets) = enumerate_halfspace_vertices_nd(halfspaces, ndim)?;
        let dual_vertices = collect_dual_vertices(&dual_facets, halfspaces.len());

        Ok(Self {
            halfspaces: halfspaces.to_vec(),
            interior_point: interior_point.to_vec(),
            intersections,
            dual_points,
            dual_facets,
            dual_vertices,
            dual_equations: Vec::new(),
            dual_area: f64::NAN,
            dual_volume: f64::NAN,
            ndim,
            nineq: halfspaces.len(),
            is_bounded,
        })
    }

    /// Recompute the intersection after appending halfspaces.
    ///
    /// The current pure-Rust implementation does not retain a live Qhull handle;
    /// `restart` is accepted for SciPy surface parity and recomputation is always
    /// deterministic from the complete halfspace set.
    pub fn add_halfspaces(
        &mut self,
        halfspaces: &[Vec<f64>],
        _restart: bool,
    ) -> Result<(), SpatialError> {
        let mut combined = self.halfspaces.clone();
        combined.extend_from_slice(halfspaces);
        *self = Self::from_nd(&combined, &self.interior_point)?;
        Ok(())
    }

    /// No-op parity method for SciPy's incremental object lifecycle.
    pub fn close(&mut self) {}
}

// ══════════════════════════════════════════════════════════════════════
// Delaunay Triangulation (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D Delaunay triangulation.
///
/// Matches the core surface of `scipy.spatial.Delaunay(points)` for 2D point
/// sets: the original points, the triangle simplices, and simplex lookup.
#[derive(Debug, Clone)]
pub struct Delaunay {
    /// Input points in the original order.
    pub points: Vec<(f64, f64)>,
    /// Triangle simplices as triples of indices into `points`.
    pub simplices: Vec<(usize, usize, usize)>,
}

impl Delaunay {
    /// Compute a 2D Delaunay triangulation with a Bowyer-Watson sweep.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        let n = points.len();
        if n < 3 {
            return Err(qhull_error(
                "QH6214 qhull input error: not enough points to construct initial simplex",
            ));
        }
        if points
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "delaunay triangulation requires finite points".to_string(),
            ));
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let dx = (max_x - min_x).max(1e-10);
        let dy = (max_y - min_y).max(1e-10);

        // SUPER-TRIANGLE MARGIN, ESCALATED UNTIL THE RESULT IS PROVABLY COMPLETE.
        //
        // Bowyer-Watson starts from a super-triangle and deletes the triangles still touching its
        // vertices at the end. If the super-triangle is too close, a genuine near-boundary
        // triangle gets attached to a super-vertex and is deleted with it, leaving a HOLE inside
        // the convex hull. `LinearNDInterpolator` then returns NaN for points in that hole, which
        // is indistinguishable from "outside the hull" (frankenscipy-tkpwa).
        //
        // The margin used to be a fixed 10.0, and a fixed value CANNOT be right: the deficit was
        // measured to grow with the point count.
        //
        //     margin   N=400   N=800   N=2000
        //         10       1       4        8      <- triangles missing
        //         30       1       2        6
        //        100       0       0        4
        //        300       0       0        1
        //       1000       0       0        0
        //
        // So this escalates instead, and checks the result rather than trusting the constant.
        // Euler's identity gives the exact triangle count for a complete triangulation of points
        // in general position: `2N - 2 - h`, with `h` the number of convex-hull vertices. The
        // first margin whose output satisfies it is accepted. Escalation is capped, and the last
        // attempt is returned either way so a pathological input degrades to the old behaviour
        // rather than failing outright.
        //
        // Starting at 1000 rather than 10 because the sweep above shows small margins essentially
        // always leave holes; starting low would just pay for a discarded attempt every call.
        let hull_vertices = ConvexHull::new(points).ok().map(|h| h.vertices.len());
        let mut simplices = Vec::new();
        for &margin in &[1000.0_f64, 1.0e5, 1.0e7] {
            let mut all_points = points.to_vec();
            all_points.push((min_x - margin * dx, min_y - margin * dy));
            all_points.push((max_x + margin * dx, min_y - margin * dy));
            all_points.push(((min_x + max_x) / 2.0, max_y + margin * dy));

            simplices = if n >= DELAUNAY_CIRCLE_GRID_THRESHOLD {
                delaunay_triangulate_circle_grid(&all_points, n, min_x, min_y, dx, dy)
            } else {
                delaunay_triangulate_linear(&all_points, n)
            };

            // Degenerate inputs (all-collinear, duplicate-heavy) have no `2N - 2 - h` to meet;
            // when the hull is unavailable or the count cannot apply, accept the first result.
            match hull_vertices {
                Some(h) if n >= 3 && 2 * n >= 2 + h => {
                    if simplices.len() >= 2 * n - 2 - h {
                        break;
                    }
                }
                _ => break,
            }
        }
        if simplices.is_empty() {
            return Err(qhull_error(
                "QH6154 Qhull precision error: initial simplex is flat",
            ));
        }

        Ok(Self {
            points: points.to_vec(),
            simplices,
        })
    }

    /// Find the simplex containing a query point.
    ///
    /// Returns the simplex index and its barycentric coordinates when the point
    /// lies in or on a triangle, or `None` if it falls outside the triangulation.
    pub fn find_simplex(&self, query: (f64, f64)) -> Option<(usize, f64, f64, f64)> {
        for (idx, &(a, b, c)) in self.simplices.iter().enumerate() {
            let (l1, l2, l3) =
                barycentric_2d(self.points[a], self.points[b], self.points[c], query);
            if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 {
                return Some((idx, l1, l2, l3));
            }
        }
        None
    }

    /// Locate the containing simplex for a BATCH of query points — matches
    /// `scipy.spatial.Delaunay.find_simplex(X)` (with fsci's barycentric return).
    /// Bit-for-bit identical to calling [`Delaunay::find_simplex`] per point: it
    /// returns the same lowest-index containing simplex and identical barycentric
    /// coordinates. Two amortized accelerations over the per-point linear scan:
    /// (1) each triangle's padded axis-aligned bbox is precomputed ONCE for the
    /// whole batch and used as a cheap reject before the barycentric test — the
    /// pad (`1e-8·extent`) safely dominates the `1e-10` barycentric tolerance, so
    /// a triangle is skipped only when it cannot contain the point; (2) the
    /// independent per-point scans are parallelized across the batch.
    pub fn find_simplex_many(&self, queries: &[(f64, f64)]) -> Vec<Option<(usize, f64, f64, f64)>> {
        let ns = self.simplices.len();
        let bboxes: Vec<(f64, f64, f64, f64)> = self
            .simplices
            .iter()
            .map(|&(a, b, c)| {
                let (pa, pb, pc) = (self.points[a], self.points[b], self.points[c]);
                let minx = pa.0.min(pb.0).min(pc.0);
                let maxx = pa.0.max(pb.0).max(pc.0);
                let miny = pa.1.min(pb.1).min(pc.1);
                let maxy = pa.1.max(pb.1).max(pc.1);
                let pad = (maxx - minx).max(maxy - miny) * 1e-8 + 1e-12;
                (minx - pad, maxx + pad, miny - pad, maxy + pad)
            })
            .collect();
        let nq = queries.len();
        let mut out: Vec<Option<(usize, f64, f64, f64)>> = vec![None; nq];
        let points = &self.points;
        let simplices = &self.simplices;
        let bb = &bboxes;

        // Uniform grid over the points' bbox. Each triangle is binned (in ASCENDING
        // index order, so every cell list stays sorted) into every cell its padded
        // bbox overlaps; a query then scans only its own cell's candidate list and
        // returns the first (= lowest-index) containing triangle. That cell list is a
        // superset of every triangle whose padded bbox contains the query point, so
        // the result is bit-for-bit identical to the O(num_simplices) bbox linear
        // scan. Degenerate / small inputs use g=1 (one cell = the full scan).
        let (mut gminx, mut gminy, mut gmaxx, mut gmaxy) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for &(x, y) in points {
            gminx = gminx.min(x);
            gmaxx = gmaxx.max(x);
            gminy = gminy.min(y);
            gmaxy = gmaxy.max(y);
        }
        let degenerate = !matches!(gmaxx.partial_cmp(&gminx), Some(std::cmp::Ordering::Greater))
            || !matches!(gmaxy.partial_cmp(&gminy), Some(std::cmp::Ordering::Greater))
            || !gminx.is_finite();
        let g: usize = if ns >= 64 && !degenerate {
            ((ns as f64).sqrt().ceil()).clamp(1.0, 1024.0) as usize
        } else {
            1
        };
        let inv_cw = if g > 1 {
            g as f64 / (gmaxx - gminx)
        } else {
            0.0
        };
        let inv_ch = if g > 1 {
            g as f64 / (gmaxy - gminy)
        } else {
            0.0
        };
        let cell_x = move |x: f64| -> usize {
            (((x - gminx) * inv_cw) as isize).clamp(0, g as isize - 1) as usize
        };
        let cell_y = move |y: f64| -> usize {
            (((y - gminy) * inv_ch) as isize).clamp(0, g as isize - 1) as usize
        };
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); g * g];
        for (idx, &(lx, hx, ly, hy)) in bb.iter().enumerate().take(ns) {
            for cy in cell_y(ly)..=cell_y(hy) {
                let row = cy * g;
                for cx in cell_x(lx)..=cell_x(hx) {
                    cells[row + cx].push(idx as u32);
                }
            }
        }
        let cells = &cells;
        let eval = move |q: (f64, f64)| -> Option<(usize, f64, f64, f64)> {
            for &idx in &cells[cell_y(q.1) * g + cell_x(q.0)] {
                let idx = idx as usize;
                let (lx, hx, ly, hy) = bb[idx];
                if q.0 < lx || q.0 > hx || q.1 < ly || q.1 > hy {
                    continue;
                }
                let (a, b, c) = simplices[idx];
                let (l1, l2, l3) = barycentric_2d(points[a], points[b], points[c], q);
                if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10 {
                    return Some((idx, l1, l2, l3));
                }
            }
            None
        };
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let nthreads = if nq >= 64 {
            cores.min(16).min(nq / 16).max(1)
        } else {
            1
        };
        if nthreads <= 1 {
            for (slot, &q) in out.iter_mut().zip(queries) {
                *slot = eval(q);
            }
            return out;
        }
        let chunk = nq.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (qchunk, ochunk) in queries.chunks(chunk).zip(out.chunks_mut(chunk)) {
                s.spawn(move || {
                    for (slot, &q) in ochunk.iter_mut().zip(qchunk) {
                        *slot = eval(q);
                    }
                });
            }
        });
        out
    }
}

const DELAUNAY_CIRCLE_GRID_THRESHOLD: usize = 4096;

fn delaunay_triangulate_linear(all_points: &[(f64, f64)], n: usize) -> Vec<(usize, usize, usize)> {
    let mut triangles = vec![(n, n + 1, n + 2)];
    // Circumcircles kept parallel to `triangles` so the bad scan tests dist²<r²
    // instead of a per-pair in-circle determinant. frankenscipy-9l5oo.
    let mut circ: Vec<(f64, f64, f64)> = vec![circumcircle_of(
        all_points[n],
        all_points[n + 1],
        all_points[n + 2],
    )];
    // Per-point scratch hoisted out of the insertion loop and cleared each pass
    // (both are fully rebuilt every point -> byte-identical), saving 2n Vec
    // allocations. frankenscipy-8d2z2.
    let mut bad: Vec<usize> = Vec::new();
    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for p_idx in 0..n {
        let point = all_points[p_idx];
        bad.clear();
        for (t_idx, &(cx, cy, r2)) in circ.iter().enumerate() {
            let ddx = point.0 - cx;
            let ddy = point.1 - cy;
            if ddx * ddx + ddy * ddy < r2 {
                bad.push(t_idx);
            }
        }

        boundary.clear();
        delaunay_collect_boundary(&triangles, &bad, &mut boundary);

        bad.sort_unstable();
        for &idx in bad.iter().rev() {
            triangles.swap_remove(idx);
            circ.swap_remove(idx);
        }
        for &(e0, e1) in &boundary {
            triangles.push((p_idx, e0, e1));
            circ.push(circumcircle_of(
                all_points[p_idx],
                all_points[e0],
                all_points[e1],
            ));
        }
    }

    triangles
        .into_iter()
        .filter(|&(a, b, c)| a < n && b < n && c < n)
        .collect()
}

fn delaunay_triangulate_circle_grid(
    all_points: &[(f64, f64)],
    n: usize,
    min_x: f64,
    min_y: f64,
    dx: f64,
    dy: f64,
) -> Vec<(usize, usize, usize)> {
    let mut triangles = vec![(n, n + 1, n + 2)];
    let mut circ: Vec<(f64, f64, f64)> = vec![circumcircle_of(
        all_points[n],
        all_points[n + 1],
        all_points[n + 2],
    )];
    let mut active = vec![true];
    let mut grid = DelaunayCircleGrid::new(n, min_x, min_y, dx, dy);
    grid.insert_circle(circ[0], 0);

    let mut bad: Vec<usize> = Vec::new();
    let mut boundary: Vec<(usize, usize)> = Vec::new();
    for p_idx in 0..n {
        let point = all_points[p_idx];
        bad.clear();
        grid.bad_triangles(point, &circ, &active, &mut bad);
        if bad.is_empty() {
            delaunay_scan_active_bad_triangles(point, &circ, &active, &mut bad);
        }
        bad.sort_unstable();
        bad.dedup();

        boundary.clear();
        delaunay_collect_boundary(&triangles, &bad, &mut boundary);

        for &idx in &bad {
            active[idx] = false;
        }
        for &(e0, e1) in &boundary {
            let triangle_idx = triangles.len();
            let circle = circumcircle_of(all_points[p_idx], all_points[e0], all_points[e1]);
            triangles.push((p_idx, e0, e1));
            circ.push(circle);
            active.push(true);
            grid.insert_circle(circle, triangle_idx);
        }
    }

    triangles
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (a, b, c))| {
            (active[idx] && a < n && b < n && c < n).then_some((a, b, c))
        })
        .collect()
}

fn delaunay_scan_active_bad_triangles(
    point: (f64, f64),
    circ: &[(f64, f64, f64)],
    active: &[bool],
    bad: &mut Vec<usize>,
) {
    for (t_idx, &(cx, cy, r2)) in circ.iter().enumerate() {
        if !active[t_idx] {
            continue;
        }
        let ddx = point.0 - cx;
        let ddy = point.1 - cy;
        if ddx * ddx + ddy * ddy < r2 {
            bad.push(t_idx);
        }
    }
}

fn delaunay_collect_boundary(
    triangles: &[(usize, usize, usize)],
    bad: &[usize],
    boundary: &mut Vec<(usize, usize)>,
) {
    for &t_idx in bad {
        let (a, b, c) = triangles[t_idx];
        for &(e0, e1) in &[(a, b), (b, c), (c, a)] {
            if !bad.iter().any(|&other_idx| {
                other_idx != t_idx
                    && triangle_has_edge(
                        triangles[other_idx].0,
                        triangles[other_idx].1,
                        triangles[other_idx].2,
                        e0,
                        e1,
                    )
            }) {
                boundary.push((e0, e1));
            }
        }
    }
}

struct DelaunayCircleGrid {
    min_x: f64,
    min_y: f64,
    inv_dx: f64,
    inv_dy: f64,
    dim: usize,
    cells: Vec<Vec<usize>>,
}

impl DelaunayCircleGrid {
    fn new(n: usize, min_x: f64, min_y: f64, dx: f64, dy: f64) -> Self {
        let dim = ((n as f64).sqrt() as usize).clamp(16, 128);
        Self {
            min_x,
            min_y,
            inv_dx: dim as f64 / dx.max(1e-10),
            inv_dy: dim as f64 / dy.max(1e-10),
            dim,
            cells: vec![Vec::new(); dim * dim],
        }
    }

    fn insert_circle(&mut self, circle: (f64, f64, f64), triangle_idx: usize) {
        let (cx, cy, r2) = circle;
        if !cx.is_finite() || !cy.is_finite() || !r2.is_finite() || r2 < 0.0 {
            return;
        }
        let r = r2.sqrt();
        let x0 = self.cell_x(cx - r);
        let x1 = self.cell_x(cx + r);
        let y0 = self.cell_y(cy - r);
        let y1 = self.cell_y(cy + r);
        for y in y0..=y1 {
            let row = y * self.dim;
            for x in x0..=x1 {
                self.cells[row + x].push(triangle_idx);
            }
        }
    }

    fn bad_triangles(
        &self,
        point: (f64, f64),
        circ: &[(f64, f64, f64)],
        active: &[bool],
        bad: &mut Vec<usize>,
    ) {
        let cell = self.point_cell(point);
        for &t_idx in &self.cells[cell] {
            if !active[t_idx] {
                continue;
            }
            let (cx, cy, r2) = circ[t_idx];
            let ddx = point.0 - cx;
            let ddy = point.1 - cy;
            if ddx * ddx + ddy * ddy < r2 {
                bad.push(t_idx);
            }
        }
    }

    fn point_cell(&self, point: (f64, f64)) -> usize {
        self.cell_y(point.1) * self.dim + self.cell_x(point.0)
    }

    fn cell_x(&self, x: f64) -> usize {
        clamp_delaunay_grid_cell((x - self.min_x) * self.inv_dx, self.dim)
    }

    fn cell_y(&self, y: f64) -> usize {
        clamp_delaunay_grid_cell((y - self.min_y) * self.inv_dy, self.dim)
    }
}

fn clamp_delaunay_grid_cell(scaled: f64, dim: usize) -> usize {
    if scaled <= 0.0 {
        0
    } else if scaled >= dim as f64 {
        dim - 1
    } else {
        scaled as usize
    }
}

/// Find the simplices containing the given points, matching
/// `scipy.spatial.tsearch(tri, xi)`.
///
/// This is the functional form of [`Delaunay::find_simplex`]: for each query
/// point it returns the index of the containing simplex, or `-1` for points
/// that fall outside the triangulation (scipy's sentinel). As with
/// [`Delaunay::find_simplex`], the simplex indices refer to this crate's own
/// Bowyer-Watson triangulation ordering rather than qhull's.
#[must_use]
pub fn tsearch(tri: &Delaunay, xi: &[(f64, f64)]) -> Vec<i64> {
    // Route through the grid-accelerated batch locator instead of an
    // O(nq·num_simplices) per-point linear scan. `find_simplex_many` is
    // documented bit-for-bit identical to calling `find_simplex` per point
    // (same lowest-index containing simplex), so the result is unchanged while
    // the cost drops from O(nq·S) to roughly O(nq) for well-distributed inputs.
    tri.find_simplex_many(xi)
        .into_iter()
        .map(|hit| match hit {
            Some((idx, _, _, _)) => idx as i64,
            None => -1,
        })
        .collect()
}

/// Cross product of vectors OA and OB where O, A, B are 2D points.
/// Positive = counter-clockwise, negative = clockwise, zero = collinear.
fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

#[allow(dead_code)]
fn point_in_circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let (ax, ay) = (a.0 - d.0, a.1 - d.1);
    let (bx, by) = (b.0 - d.0, b.1 - d.1);
    let (cx, cy) = (c.0 - d.0, c.1 - d.1);
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);
    let orient = cross(a, b, c);
    if orient > 0.0 { det > 0.0 } else { det < 0.0 }
}

/// Circumcircle (center_x, center_y, radius²) of a non-degenerate triangle. Precomputed
/// once per triangle so the Bowyer-Watson bad-triangle scan tests `dist² < r²` (~5 flops)
/// instead of recomputing the full in-circle determinant + orientation per (point,
/// triangle) pair (~20 flops). For non-degenerate points this is the same in/out verdict
/// as `point_in_circumcircle`; cocircular boundary cases agree (det=0 ⇔ dist²=r², both
/// excluded by the strict `<`). frankenscipy-9l5oo.
fn circumcircle_of(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> (f64, f64, f64) {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (cx, cy) = c;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
    (ux, uy, r2)
}

fn triangle_has_edge(a: usize, b: usize, c: usize, e0: usize, e1: usize) -> bool {
    [(a, b), (b, c), (c, a)]
        .iter()
        .any(|&(x, y)| (x == e0 && y == e1) || (x == e1 && y == e0))
}

fn barycentric_2d(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> (f64, f64, f64) {
    let (v0x, v0y) = (b.0 - a.0, b.1 - a.1);
    let (v1x, v1y) = (c.0 - a.0, c.1 - a.1);
    let (v2x, v2y) = (p.0 - a.0, p.1 - a.1);
    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-30 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let l2 = (d11 * d20 - d01 * d21) / denom;
    let l3 = (d00 * d21 - d01 * d20) / denom;
    (1.0 - l2 - l3, l2, l3)
}

fn qhull_error(message: impl Into<String>) -> SpatialError {
    SpatialError::Qhull(QhullError::new(message))
}

fn validate_halfspace_intersection_inputs(
    halfspaces: &[Halfspace2],
    interior_point: Point2,
) -> Result<(), SpatialError> {
    if halfspaces.len() < 3 {
        return Err(qhull_error(format!(
            "QH6214 qhull input error: not enough halfspaces({}) to construct initial simplex (need 3)",
            halfspaces.len()
        )));
    }
    if !interior_point.0.is_finite() || !interior_point.1.is_finite() {
        return Err(SpatialError::InvalidArgument(
            "interior_point must contain finite coordinates".to_string(),
        ));
    }

    for (idx, &[a, b, offset]) in halfspaces.iter().enumerate() {
        if !a.is_finite() || !b.is_finite() || !offset.is_finite() {
            return Err(SpatialError::InvalidArgument(format!(
                "halfspace row {idx} contains a non-finite coefficient"
            )));
        }
        let normal_norm = a.hypot(b);
        if normal_norm <= f64::EPSILON {
            return Err(qhull_error(format!(
                "QH6154 Qhull precision error: halfspace row {idx} has zero normal"
            )));
        }
        let signed_distance = a * interior_point.0 + b * interior_point.1 + offset;
        let clear_inside_tol = 1e-12 * normal_norm.max(1.0);
        if signed_distance >= -clear_inside_tol {
            return Err(qhull_error(
                "QH6023 qhull input error: feasible point is not clearly inside halfspace",
            ));
        }
    }

    Ok(())
}

fn validate_halfspace_intersection_inputs_nd(
    halfspaces: &[Vec<f64>],
    interior_point: &[f64],
) -> Result<(), SpatialError> {
    let ndim = interior_point.len();
    if ndim == 0 {
        return Err(SpatialError::InvalidArgument(
            "interior_point must not be empty".to_string(),
        ));
    }
    if halfspaces.len() < ndim + 1 {
        return Err(qhull_error(format!(
            "QH6214 qhull input error: not enough halfspaces({}) to construct initial simplex (need {})",
            halfspaces.len(),
            ndim + 1
        )));
    }
    if interior_point.iter().any(|value| !value.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "interior_point must contain finite coordinates".to_string(),
        ));
    }

    for (idx, row) in halfspaces.iter().enumerate() {
        if row.len() != ndim + 1 {
            return Err(SpatialError::DimensionMismatch {
                expected: ndim + 1,
                actual: row.len(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(SpatialError::InvalidArgument(format!(
                "halfspace row {idx} contains a non-finite coefficient"
            )));
        }
        let normal_norm = row[..ndim]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if normal_norm <= f64::EPSILON {
            return Err(qhull_error(format!(
                "QH6154 Qhull precision error: halfspace row {idx} has zero normal"
            )));
        }
        let signed_distance = row[..ndim]
            .iter()
            .zip(interior_point.iter())
            .map(|(a, x)| a * x)
            .sum::<f64>()
            + row[ndim];
        let clear_inside_tol = 1e-12 * normal_norm.max(1.0);
        if signed_distance >= -clear_inside_tol {
            return Err(qhull_error(
                "QH6023 qhull input error: feasible point is not clearly inside halfspace",
            ));
        }
    }

    Ok(())
}

fn halfspace_dual_points(halfspaces: &[Halfspace2], interior_point: Point2) -> Vec<Point2> {
    halfspaces
        .iter()
        .map(|&[a, b, offset]| {
            let distance = a * interior_point.0 + b * interior_point.1 + offset;
            (-a / distance, -b / distance)
        })
        .collect()
}

fn halfspace_dual_points_nd(halfspaces: &[Vec<f64>], interior_point: &[f64]) -> Vec<Vec<f64>> {
    let ndim = interior_point.len();
    halfspaces
        .iter()
        .map(|row| {
            let distance = row[..ndim]
                .iter()
                .zip(interior_point.iter())
                .map(|(a, x)| a * x)
                .sum::<f64>()
                + row[ndim];
            row[..ndim].iter().map(|value| -value / distance).collect()
        })
        .collect()
}

fn dual_edge_equation(lhs: Point2, rhs: Point2) -> Equation2 {
    let dx = rhs.0 - lhs.0;
    let dy = rhs.1 - lhs.1;
    let length = dx.hypot(dy);
    if length <= f64::EPSILON {
        return [f64::NAN, f64::NAN, f64::NAN];
    }

    let mut normal_x = -dy / length;
    let mut normal_y = dx / length;
    let mut offset = -(normal_x * lhs.0 + normal_y * lhs.1);
    if offset > 0.0 {
        normal_x = -normal_x;
        normal_y = -normal_y;
        offset = -offset;
    }
    [normal_x, normal_y, offset]
}

fn intersection_from_dual_equation(equation: Equation2, interior_point: Point2) -> Point2 {
    let denominator = -equation[2];
    (
        interior_point.0 + equation[0] / denominator,
        interior_point.1 + equation[1] / denominator,
    )
}

fn halfspace_region_is_bounded(halfspaces: &[Halfspace2]) -> bool {
    let mut angles = halfspaces
        .iter()
        .map(|&[a, b, _]| {
            let mut angle = b.atan2(a);
            if angle < 0.0 {
                angle += std::f64::consts::TAU;
            }
            angle
        })
        .collect::<Vec<_>>();
    angles.sort_by(f64::total_cmp);
    angles.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() < 1e-12);
    if angles.len() < 3 {
        return false;
    }

    for idx in 0..angles.len() {
        let next = if idx + 1 == angles.len() {
            angles[0] + std::f64::consts::TAU
        } else {
            angles[idx + 1]
        };
        if next - angles[idx] >= std::f64::consts::PI - 1e-12 {
            return false;
        }
    }
    true
}

fn halfspace_region_is_bounded_nd(halfspaces: &[Vec<f64>], ndim: usize) -> bool {
    let normals = halfspaces
        .iter()
        .map(|row| row[..ndim].to_vec())
        .collect::<Vec<_>>();

    // The region is bounded iff some (ndim+1)-subset of normals has the origin
    // strictly inside its positive cone. Test each (ndim+1)-subset LAZILY in
    // lexicographic order, short-circuiting on the first witness — identical to
    // `combinations_recursive(..).into_iter().any(..)` but WITHOUT first
    // materializing all C(m, ndim+1) subsets (≈ m/(ndim+1)× more than the vertex
    // enumeration, e.g. 8.2M tiny Vecs for m=120, ndim=3). For bounded regions a
    // witness is found almost immediately; unbounded regions still scan all
    // subsets but no longer pay the O(C(m,ndim+1)) allocation up front.
    combinations_any(normals.len(), ndim + 1, |combo| {
        let mut matrix = vec![vec![0.0; ndim + 1]; ndim + 1];
        let mut rhs = vec![0.0; ndim + 1];
        rhs[ndim] = 1.0;

        for (col, &normal_idx) in combo.iter().enumerate() {
            for row in 0..ndim {
                matrix[row][col] = normals[normal_idx][row];
            }
            matrix[ndim][col] = 1.0;
        }

        solve_linear_system(&matrix, &rhs, 1e-10)
            .is_some_and(|weights| weights.iter().all(|value| *value > 1e-9))
    })
}

/// Evaluate `pred` on each k-subset of `0..n` in the SAME lexicographic order as
/// [`combinations_recursive`], short-circuiting (true) on the first subset that
/// satisfies it. Equivalent to
/// `combinations_recursive(n, k, ..).into_iter().any(pred)` but streams the
/// subsets instead of materializing all C(n, k) of them first.
fn combinations_any(n: usize, k: usize, mut pred: impl FnMut(&[usize]) -> bool) -> bool {
    fn rec(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        pred: &mut dyn FnMut(&[usize]) -> bool,
    ) -> bool {
        if current.len() == k {
            return pred(current);
        }
        if start >= n {
            return false;
        }
        for next in start..=n - (k - current.len()) {
            current.push(next);
            if rec(n, k, next + 1, current, pred) {
                return true;
            }
            current.pop();
        }
        false
    }
    let mut current = Vec::with_capacity(k);
    rec(n, k, 0, &mut current, &mut pred)
}

fn enumerate_halfspace_vertices_nd(
    halfspaces: &[Vec<f64>],
    ndim: usize,
) -> Result<(NdVertices, NdFacets), SpatialError> {
    let mut combos = Vec::new();
    combinations_recursive(halfspaces.len(), ndim, 0, &mut Vec::new(), &mut combos);

    let mut vertices = Vec::<Vec<f64>>::new();
    let mut facets = Vec::<Vec<usize>>::new();
    for (solution, combo) in halfspace_vertex_candidates_nd(halfspaces, ndim, &combos) {
        if let Some(existing_idx) = vertices
            .iter()
            .position(|vertex| points_approx_eq(vertex, &solution, 1e-8))
        {
            merge_sorted_unique(&mut facets[existing_idx], &combo);
        } else {
            vertices.push(solution);
            facets.push(combo);
        }
    }

    if vertices.is_empty() {
        return Err(qhull_error(
            "QH6154 Qhull precision error: initial simplex is flat",
        ));
    }

    Ok((vertices, facets))
}

fn halfspace_vertex_candidates_nd(
    halfspaces: &[Vec<f64>],
    ndim: usize,
    combos: &[Vec<usize>],
) -> Vec<NdVertexCandidate> {
    let worker_count = halfspace_vertex_enum_thread_count(combos.len(), halfspaces.len(), ndim);
    halfspace_vertex_candidates_nd_with_workers(halfspaces, ndim, combos, worker_count)
}

fn halfspace_vertex_candidates_nd_with_workers(
    halfspaces: &[Vec<f64>],
    ndim: usize,
    combos: &[Vec<usize>],
    worker_count: usize,
) -> Vec<NdVertexCandidate> {
    if worker_count <= 1 {
        return combos
            .iter()
            .filter_map(|combo| halfspace_vertex_candidate_nd(halfspaces, ndim, combo))
            .collect();
    }

    let worker_count = worker_count.min(combos.len()).max(1);
    let chunk_len = combos.len().div_ceil(worker_count);
    let parts: Vec<Vec<NdVertexCandidate>> = std::thread::scope(|scope| {
        let handles: Vec<_> = combos
            .chunks(chunk_len)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .filter_map(|combo| halfspace_vertex_candidate_nd(halfspaces, ndim, combo))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("halfspace vertex enumeration worker panicked")
            })
            .collect()
    });
    parts.into_iter().flatten().collect()
}

fn halfspace_vertex_enum_thread_count(
    combo_count: usize,
    halfspace_count: usize,
    ndim: usize,
) -> usize {
    let work = (combo_count as u64)
        .saturating_mul(halfspace_count as u64)
        .saturating_mul(ndim.max(1) as u64);
    if combo_count < 4096 || work < 1 << 20 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    cores.min(combo_count / 1024).clamp(1, 16)
}

fn halfspace_vertex_candidate_nd(
    halfspaces: &[Vec<f64>],
    ndim: usize,
    combo: &[usize],
) -> Option<NdVertexCandidate> {
    let mut matrix = vec![vec![0.0; ndim]; ndim];
    let mut rhs = vec![0.0; ndim];
    for (row_idx, &halfspace_idx) in combo.iter().enumerate() {
        matrix[row_idx].copy_from_slice(&halfspaces[halfspace_idx][..ndim]);
        rhs[row_idx] = -halfspaces[halfspace_idx][ndim];
    }

    let solution = solve_linear_system(&matrix, &rhs, 1e-10)?;
    if !point_satisfies_halfspaces(&solution, halfspaces, 1e-8) {
        return None;
    }

    Some((solution, combo.to_vec()))
}

fn collect_dual_vertices(facets: &[Vec<usize>], total_halfspaces: usize) -> Vec<usize> {
    let mut present = vec![false; total_halfspaces];
    for facet in facets {
        for &index in facet {
            present[index] = true;
        }
    }
    present
        .into_iter()
        .enumerate()
        .filter_map(|(index, used)| used.then_some(index))
        .collect()
}

fn point_satisfies_halfspaces(point: &[f64], halfspaces: &[Vec<f64>], tol: f64) -> bool {
    let ndim = point.len();
    halfspaces.iter().all(|row| {
        row[..ndim]
            .iter()
            .zip(point.iter())
            .map(|(a, x)| a * x)
            .sum::<f64>()
            + row[ndim]
            <= tol
    })
}

fn points_approx_eq(lhs: &[f64], rhs: &[f64], tol: f64) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| (left - right).abs() <= tol)
}

fn merge_sorted_unique(target: &mut Vec<usize>, incoming: &[usize]) {
    for &value in incoming {
        if !target.contains(&value) {
            target.push(value);
        }
    }
    target.sort_unstable();
}

fn combinations_recursive(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        output.push(current.clone());
        return;
    }
    if start >= n {
        return;
    }
    for next in start..=n - (k - current.len()) {
        current.push(next);
        combinations_recursive(n, k, next + 1, current, output);
        current.pop();
    }
}

fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64], tol: f64) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut augmented = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, rhs_value)| {
            let mut combined = row.clone();
            combined.push(*rhs_value);
            combined
        })
        .collect::<Vec<_>>();

    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_value = augmented[pivot][pivot].abs();
        for (row_idx, row) in augmented.iter().enumerate().skip(pivot + 1) {
            let candidate = row[pivot].abs();
            if candidate > best_value {
                best_value = candidate;
                best_row = row_idx;
            }
        }
        if best_value <= tol {
            return None;
        }
        if best_row != pivot {
            augmented.swap(best_row, pivot);
        }

        let pivot_value = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot) {
            *value /= pivot_value;
        }

        let pivot_row = augmented[pivot].clone();
        for (row_idx, row) in augmented.iter_mut().enumerate() {
            if row_idx == pivot {
                continue;
            }
            let factor = row[pivot];
            if factor.abs() <= tol {
                continue;
            }
            for (value, pivot_value) in row.iter_mut().zip(pivot_row.iter()).skip(pivot) {
                *value -= factor * *pivot_value;
            }
        }
    }

    Some(augmented.into_iter().map(|row| row[n]).collect())
}

// ══════════════════════════════════════════════════════════════════════
// Voronoi Diagram (2D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 2D Voronoi diagram construction.
///
/// Matches the core SciPy surface for `scipy.spatial.Voronoi(points)`:
/// input points, Voronoi vertices, ridge connectivity, and point regions.
#[derive(Debug, Clone)]
pub struct Voronoi {
    /// Input points in the original order.
    pub points: Vec<(f64, f64)>,
    /// Voronoi vertices (circumcenters of Delaunay simplices).
    pub vertices: Vec<(f64, f64)>,
    /// Input-point pairs whose dual Delaunay edge defines each Voronoi ridge.
    pub ridge_points: Vec<(usize, usize)>,
    /// Voronoi vertex pairs for each ridge. `-1` denotes an unbounded ray.
    pub ridge_vertices: Vec<(isize, isize)>,
    /// Region vertex indices for each Voronoi region. `-1` denotes infinity.
    pub regions: Vec<Vec<isize>>,
    /// Region index for each input point.
    pub point_region: Vec<usize>,
}

impl Voronoi {
    /// Compute a 2D Voronoi diagram via the dual of the Delaunay triangulation.
    pub fn new(points: &[(f64, f64)]) -> Result<Self, SpatialError> {
        let delaunay = Delaunay::new(points)?;
        let n = points.len();

        let mut vertices = Vec::with_capacity(delaunay.simplices.len());
        for &(a, b, c) in &delaunay.simplices {
            vertices.push(circumcenter_2d(points[a], points[b], points[c])?);
        }

        let mut edge_to_triangles: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        let mut point_to_triangles = vec![Vec::new(); n];
        for (tri_idx, &(a, b, c)) in delaunay.simplices.iter().enumerate() {
            point_to_triangles[a].push(tri_idx);
            point_to_triangles[b].push(tri_idx);
            point_to_triangles[c].push(tri_idx);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let edge = canonical_edge(u, v);
                edge_to_triangles.entry(edge).or_default().push(tri_idx);
            }
        }

        let mut ridge_points = Vec::with_capacity(edge_to_triangles.len());
        let mut ridge_vertices = Vec::with_capacity(edge_to_triangles.len());
        let mut point_is_unbounded = vec![false; n];

        for (&(u, v), triangles) in &edge_to_triangles {
            ridge_points.push((u, v));
            match triangles.as_slice() {
                [tri_idx] => {
                    point_is_unbounded[u] = true;
                    point_is_unbounded[v] = true;
                    ridge_vertices.push((-1, *tri_idx as isize));
                }
                [left, right] => {
                    ridge_vertices.push((*left as isize, *right as isize));
                }
                _ => {
                    return Err(qhull_error(
                        "Qhull topology error: Voronoi construction encountered a non-manifold Delaunay edge",
                    ));
                }
            }
        }

        let mut regions = vec![Vec::new()];
        let mut point_region = vec![0usize; n];
        for point_idx in 0..n {
            let point = points[point_idx];
            // Each point's incident-triangle list is read exactly once here and
            // never reused, so MOVE it out instead of cloning. Byte-identical.
            let mut incident = std::mem::take(&mut point_to_triangles[point_idx]);
            incident.sort_by(|&lhs, &rhs| {
                let a = vertices[lhs];
                let b = vertices[rhs];
                let angle_a = (a.1 - point.1).atan2(a.0 - point.0);
                let angle_b = (b.1 - point.1).atan2(b.0 - point.0);
                angle_a.total_cmp(&angle_b)
            });

            let mut region: Vec<isize> = incident.into_iter().map(|idx| idx as isize).collect();
            if point_is_unbounded[point_idx] {
                region.insert(0, -1);
            }

            point_region[point_idx] = regions.len();
            regions.push(region);
        }

        Ok(Self {
            points: points.to_vec(),
            vertices,
            ridge_points,
            ridge_vertices,
            regions,
            point_region,
        })
    }
}

fn canonical_edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn circumcenter_2d(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
) -> Result<(f64, f64), SpatialError> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-30 {
        return Err(qhull_error(
            "QH6154 Qhull precision error: initial simplex is flat",
        ));
    }

    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
    let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
    Ok((ux, uy))
}

// ══════════════════════════════════════════════════════════════════════
// Spherical Voronoi Diagram (3D)
// ══════════════════════════════════════════════════════════════════════

/// Result of a 3D spherical Voronoi diagram for points on a common sphere.
///
/// Matches the core SciPy surface for `scipy.spatial.SphericalVoronoi`:
/// input points, Voronoi vertices, and per-point vertex regions.
#[derive(Debug, Clone)]
pub struct SphericalVoronoi {
    /// Input points in original order.
    pub points: Vec<[f64; 3]>,
    /// Voronoi vertices on the sphere.
    pub vertices: Vec<[f64; 3]>,
    /// For each input point, the ordered vertex indices of its Voronoi region.
    pub regions: Vec<Vec<usize>>,
    /// Sphere center.
    pub center: [f64; 3],
    /// Sphere radius.
    pub radius: f64,
}

/// A candidate Voronoi face: its generating point triplet `(i, j, k)` and the
/// projected vertex on the sphere.
type SvCandidateFace = ((usize, usize, usize), [f64; 3]);

/// Triangular facets of the 3-D convex hull of `points`, returned as index
/// triples. For points on a sphere every point is a hull vertex and the hull
/// facets are exactly the spherical Delaunay triangles, so this replaces the
/// O(n⁴) all-triplets gift-wrapping face scan with an O(n²) incremental hull
/// (Clarkson–Shor style insertion). Orientation within each triple is
/// unspecified — callers recompute the outward normal — so only the facet *set*
/// matters. Returns `None` if the points are degenerate (fewer than four
/// affinely independent, e.g. all coplanar on a great circle).
fn convex_hull_3d_facets(points: &[[f64; 3]], _center: [f64; 3]) -> Option<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 4 {
        return None;
    }
    // For large point counts the all-faces visibility scan below is O(n) per
    // insertion → O(n²) overall, which loses to SciPy's O(n log n) Qhull. Route
    // those through the conflict-graph hull (visible region found by adjacency
    // walk, O(n log n) expected). It inserts in the SAME index order with the
    // SAME centroid orientation and SAME visibility test, so the facet *set* is
    // identical (property-tested); it bails to `None` on any geometric
    // inconsistency, falling through to the robust O(n²) path here.
    if n >= 512
        && let Some(f) = convex_hull_3d_facets_fast(points)
    {
        return Some(f);
    }
    let tol = 1e-12;
    let face_normal = |f: &[usize; 3]| {
        cross3(
            sub3(points[f[1]], points[f[0]]),
            sub3(points[f[2]], points[f[0]]),
        )
    };

    // Seed an affinely independent tetrahedron: distinct, non-collinear,
    // non-coplanar points. Any missing => degenerate input.
    let p0 = 0usize;
    let p1 = (1..n).find(|&i| norm3(sub3(points[i], points[p0])) > tol)?;
    let p2 = (0..n).find(|&i| {
        i != p0
            && i != p1
            && norm3(cross3(
                sub3(points[p1], points[p0]),
                sub3(points[i], points[p0]),
            )) > tol
    })?;
    let base_normal = cross3(sub3(points[p1], points[p0]), sub3(points[p2], points[p0]));
    let p3 = (0..n).find(|&i| {
        i != p0 && i != p1 && i != p2 && dot3(base_normal, sub3(points[i], points[p0])).abs() > tol
    })?;

    // Orient faces against the seed-tetrahedron centroid, NOT the sphere centre.
    // The centroid is strictly inside the seed tetra and stays inside every
    // intermediate hull (incremental insertion only ever expands the hull), so
    // "normal points away from the centroid" is the true outward direction at
    // every step. The sphere centre is wrong here: early partial hulls of a few
    // on-sphere points need not enclose the sphere centre, which silently flips
    // face windings, corrupts the horizon twin-edge test, and blows the facet
    // count up (or drops faces) — the bug this replaces.
    let centroid = scale3(
        add3(add3(points[p0], points[p1]), add3(points[p2], points[p3])),
        0.25,
    );
    // Outward-oriented facet from three indices: flip so the normal points away
    // from the hull centroid, making the visibility test unambiguous.
    let make_face = |a: usize, b: usize, c: usize| -> [usize; 3] {
        let normal = cross3(sub3(points[b], points[a]), sub3(points[c], points[a]));
        if dot3(normal, sub3(points[a], centroid)) >= 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        }
    };

    let mut faces: Vec<[usize; 3]> = vec![
        make_face(p0, p1, p2),
        make_face(p0, p1, p3),
        make_face(p0, p2, p3),
        make_face(p1, p2, p3),
    ];
    let mut face_normals: Vec<[f64; 3]> = faces.iter().map(face_normal).collect();
    let mut in_hull = vec![false; n];
    for &s in &[p0, p1, p2, p3] {
        in_hull[s] = true;
    }

    // Reused scratch buffers — the insertion loop is O(n) iterations and each
    // touches O(faces) work, so allocating these fresh per point dominated the
    // wall time (gap-grows-with-n alloc tell). Clear-and-refill instead.
    let mut visible_faces: Vec<usize> = Vec::new();
    let mut visible_marks: Vec<usize> = Vec::new();
    let mut visible_stamp = 0usize;
    let mut visible_edges: Vec<(usize, usize)> = Vec::new();
    let mut sorted_visible_edges: Vec<(usize, usize)> = Vec::new();
    let mut horizon: Vec<(usize, usize)> = Vec::new();

    for p in 0..n {
        if in_hull[p] {
            continue;
        }
        let pp = points[p];
        visible_faces.clear();
        visible_stamp += 1;
        if visible_marks.len() < faces.len() {
            visible_marks.resize(faces.len(), 0);
        }
        for (fi, (f, &normal)) in faces.iter().zip(&face_normals).enumerate() {
            if dot3(normal, sub3(pp, points[f[0]])) > tol {
                visible_faces.push(fi);
                visible_marks[fi] = visible_stamp;
            }
        }
        if visible_faces.is_empty() {
            continue; // strictly inside the current hull (shouldn't happen on a sphere)
        }
        // Directed edges of all visible faces; a directed edge whose twin is not
        // itself in a visible face lies on the horizon. Keep the original scan
        // order for deterministic face emission, but use a sorted scratch list
        // for cache-local membership checks instead of hashing every edge.
        visible_edges.clear();
        for &fi in &visible_faces {
            let f = faces[fi];
            visible_edges.push((f[0], f[1]));
            visible_edges.push((f[1], f[2]));
            visible_edges.push((f[2], f[0]));
        }
        sorted_visible_edges.clear();
        sorted_visible_edges.extend_from_slice(&visible_edges);
        sorted_visible_edges.sort_unstable();
        horizon.clear();
        for &(u, v) in &visible_edges {
            if sorted_visible_edges.binary_search(&(v, u)).is_err() {
                horizon.push((u, v));
            }
        }
        // Drop the visible faces by compacting the survivors in place (no new
        // allocation), then cone the horizon to the new apex.
        let mut w = 0;
        for r in 0..faces.len() {
            if visible_marks[r] != visible_stamp {
                faces[w] = faces[r];
                face_normals[w] = face_normals[r];
                w += 1;
            }
        }
        faces.truncate(w);
        face_normals.truncate(w);
        for &(u, v) in &horizon {
            let face = make_face(u, v, p);
            face_normals.push(face_normal(&face));
            faces.push(face);
        }
        in_hull[p] = true;
    }

    Some(faces)
}

/// Multiplicative (FxHash-style) hasher for the conflict-graph hull's directed
/// edge → face map. Edge keys are single `u64`s, so std's byte-wise SipHash is
/// pure overhead here; this mixes one word per key and measurably outpaces it on
/// the millions of edge insert/lookup/remove ops a large hull performs.
#[derive(Default)]
struct EdgeHasher(u64);
impl std::hash::Hasher for EdgeHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}
type EdgeMap = std::collections::HashMap<u64, usize, std::hash::BuildHasherDefault<EdgeHasher>>;
#[inline]
fn edge_key(u: usize, v: usize) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// Append a centroid-oriented facet `{a,b,c}` to the conflict-graph hull,
/// registering its three CCW directed edges for adjacency. Returns the new
/// (stable) face id. Mirrors the `make_face` + `face_normal` of the O(n²) path
/// exactly so both produce the same facet set.
#[allow(clippy::too_many_arguments)]
fn hull_add_face(
    points: &[[f64; 3]],
    centroid: [f64; 3],
    fv: &mut Vec<[usize; 3]>,
    fnorm: &mut Vec<[f64; 3]>,
    falive: &mut Vec<bool>,
    fpts: &mut Vec<Vec<usize>>,
    edge_face: &mut EdgeMap,
    a: usize,
    b: usize,
    c: usize,
) -> usize {
    let nrm0 = cross3(sub3(points[b], points[a]), sub3(points[c], points[a]));
    let tri = if dot3(nrm0, sub3(points[a], centroid)) >= 0.0 {
        [a, b, c]
    } else {
        [a, c, b]
    };
    let nrm = cross3(
        sub3(points[tri[1]], points[tri[0]]),
        sub3(points[tri[2]], points[tri[0]]),
    );
    let id = fv.len();
    fv.push(tri);
    fnorm.push(nrm);
    falive.push(true);
    fpts.push(Vec::new());
    edge_face.insert(edge_key(tri[0], tri[1]), id);
    edge_face.insert(edge_key(tri[1], tri[2]), id);
    edge_face.insert(edge_key(tri[2], tri[0]), id);
    id
}

/// O(n log n)-expected 3-D convex hull via Clarkson–Shor conflict-graph
/// incremental insertion, used by `convex_hull_3d_facets` for large `n`.
///
/// The visible-face region of an external point is connected on a convex
/// polytope, so rather than testing every live face on each insertion (the
/// O(n²) cost), we keep a *seed* visible face per uninserted point and discover
/// the whole visible region by an adjacency walk (DFS over edge→face twins).
/// Points displaced when their seed face is deleted are redistributed onto the
/// freshly coned faces. Insertion order, orientation and the visibility test
/// match the O(n²) path, so the emitted facet *set* is identical. Returns
/// `None` on degenerate input or any geometric inconsistency (caller falls back
/// to the robust O(n²) hull).
fn convex_hull_3d_facets_fast(points: &[[f64; 3]]) -> Option<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 4 {
        return None;
    }
    let tol = 1e-12;

    // Seed tetrahedron — identical selection to the O(n²) path.
    let p0 = 0usize;
    let p1 = (1..n).find(|&i| norm3(sub3(points[i], points[p0])) > tol)?;
    let p2 = (0..n).find(|&i| {
        i != p0
            && i != p1
            && norm3(cross3(
                sub3(points[p1], points[p0]),
                sub3(points[i], points[p0]),
            )) > tol
    })?;
    let base_normal = cross3(sub3(points[p1], points[p0]), sub3(points[p2], points[p0]));
    let p3 = (0..n).find(|&i| {
        i != p0 && i != p1 && i != p2 && dot3(base_normal, sub3(points[i], points[p0])).abs() > tol
    })?;
    let centroid = scale3(
        add3(add3(points[p0], points[p1]), add3(points[p2], points[p3])),
        0.25,
    );

    // Stable face storage (faces are never reindexed; dead faces keep their slot).
    let mut fv: Vec<[usize; 3]> = Vec::with_capacity(2 * n);
    let mut fnorm: Vec<[f64; 3]> = Vec::with_capacity(2 * n);
    let mut falive: Vec<bool> = Vec::with_capacity(2 * n);
    let mut fpts: Vec<Vec<usize>> = Vec::with_capacity(2 * n);
    let mut edge_face: EdgeMap =
        EdgeMap::with_capacity_and_hasher(6 * n, std::hash::BuildHasherDefault::default());

    let f_ids = [
        hull_add_face(
            points,
            centroid,
            &mut fv,
            &mut fnorm,
            &mut falive,
            &mut fpts,
            &mut edge_face,
            p0,
            p1,
            p2,
        ),
        hull_add_face(
            points,
            centroid,
            &mut fv,
            &mut fnorm,
            &mut falive,
            &mut fpts,
            &mut edge_face,
            p0,
            p1,
            p3,
        ),
        hull_add_face(
            points,
            centroid,
            &mut fv,
            &mut fnorm,
            &mut falive,
            &mut fpts,
            &mut edge_face,
            p0,
            p2,
            p3,
        ),
        hull_add_face(
            points,
            centroid,
            &mut fv,
            &mut fnorm,
            &mut falive,
            &mut fpts,
            &mut edge_face,
            p1,
            p2,
            p3,
        ),
    ];

    let mut in_hull = vec![false; n];
    for &s in &[p0, p1, p2, p3] {
        in_hull[s] = true;
    }
    // Each uninserted point sees ≥1 tetra face (it lies outside the inscribed
    // tetra). Assign a seed; a point seeing none is degenerate → bail.
    let mut point_seed: Vec<usize> = vec![usize::MAX; n];
    for q in 0..n {
        if in_hull[q] {
            continue;
        }
        let qq = points[q];
        for &fid in &f_ids {
            if dot3(fnorm[fid], sub3(qq, points[fv[fid][0]])) > tol {
                point_seed[q] = fid;
                fpts[fid].push(q);
                break;
            }
        }
        if point_seed[q] == usize::MAX {
            return None;
        }
    }

    // Clarkson–Shor is O(n log n) *expected* only for a random insertion order;
    // a spatially-coherent order (e.g. a sphere sampled along a spiral) drives
    // it toward O(n²) by growing long conflict lists. Insert in a deterministic
    // shuffle (fixed-seed Fisher–Yates) to recover the expected bound. The hull
    // facet *set* of points in general position is order-independent, so this
    // does not change the result (property-tested against the index-order path).
    let mut order: Vec<usize> = (0..n).filter(|&i| !in_hull[i]).collect();
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    for i in (1..order.len()).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        order.swap(i, j);
    }

    // Reused scratch.
    let mut vstamp: Vec<u32> = Vec::new();
    let mut cur_stamp: u32 = 0;
    let mut visible: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut horizon: Vec<(usize, usize)> = Vec::new();
    let mut candidates: Vec<usize> = Vec::new();

    for &p in &order {
        if in_hull[p] {
            continue;
        }
        let seed = point_seed[p];
        if seed == usize::MAX || !falive[seed] {
            return None; // seed invariant violated → fall back to robust path
        }
        let pp = points[p];

        // Discover the connected visible region by adjacency walk from the seed.
        cur_stamp += 1;
        if vstamp.len() < fv.len() {
            vstamp.resize(fv.len(), 0);
        }
        visible.clear();
        stack.clear();
        stack.push(seed);
        vstamp[seed] = cur_stamp;
        while let Some(f) = stack.pop() {
            visible.push(f);
            let tri = fv[f];
            for &(u, v) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if let Some(&nb) = edge_face.get(&edge_key(v, u))
                    && falive[nb]
                    && vstamp[nb] != cur_stamp
                    && dot3(fnorm[nb], sub3(pp, points[fv[nb][0]])) > tol
                {
                    vstamp[nb] = cur_stamp;
                    stack.push(nb);
                }
            }
        }

        // Horizon = directed edges of visible faces whose twin face is not
        // visible. Also gather the uninserted points stranded on visible faces.
        horizon.clear();
        candidates.clear();
        for &f in &visible {
            let tri = fv[f];
            for &(u, v) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let twin_visible = match edge_face.get(&edge_key(v, u)) {
                    Some(&nb) => falive[nb] && vstamp[nb] == cur_stamp,
                    None => false,
                };
                if !twin_visible {
                    horizon.push((u, v));
                }
            }
            for &q in &fpts[f] {
                if q != p && !in_hull[q] {
                    candidates.push(q);
                }
            }
        }

        // Retire visible faces (drop only the edge entries they still own).
        for &f in &visible {
            falive[f] = false;
            let tri = fv[f];
            for &(u, v) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let k = edge_key(u, v);
                if edge_face.get(&k) == Some(&f) {
                    edge_face.remove(&k);
                }
            }
            fpts[f].clear();
        }

        // Cone the horizon to apex p.
        let new_start = fv.len();
        for &(u, v) in &horizon {
            hull_add_face(
                points,
                centroid,
                &mut fv,
                &mut fnorm,
                &mut falive,
                &mut fpts,
                &mut edge_face,
                u,
                v,
                p,
            );
        }
        let new_end = fv.len();
        in_hull[p] = true;
        point_seed[p] = usize::MAX;

        // Redistribute stranded points onto a face they can still see (new faces
        // first; rare fallback to any live face keeps every seed valid).
        for &q in &candidates {
            let qq = points[q];
            let mut assigned = usize::MAX;
            for nf in new_start..new_end {
                if dot3(fnorm[nf], sub3(qq, points[fv[nf][0]])) > tol {
                    assigned = nf;
                    break;
                }
            }
            if assigned == usize::MAX {
                for nf in 0..fv.len() {
                    if falive[nf] && dot3(fnorm[nf], sub3(qq, points[fv[nf][0]])) > tol {
                        assigned = nf;
                        break;
                    }
                }
            }
            if assigned == usize::MAX {
                return None;
            }
            point_seed[q] = assigned;
            fpts[assigned].push(q);
        }
    }

    let mut out: Vec<[usize; 3]> = Vec::new();
    for (id, tri) in fv.iter().enumerate() {
        if falive[id] {
            out.push(*tri);
        }
    }
    Some(out)
}

impl SphericalVoronoi {
    /// Construct a spherical Voronoi diagram for 3D points on a common sphere.
    pub fn new(points: &[[f64; 3]], center: [f64; 3], radius: f64) -> Result<Self, SpatialError> {
        if points.len() < 4 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires at least 4 points".to_string(),
            ));
        }
        if points.iter().flatten().any(|v| !v.is_finite()) || !center.iter().all(|v| v.is_finite())
        {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi points and center must be finite".to_string(),
            ));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires a positive finite radius".to_string(),
            ));
        }

        let tol = 1e-8;
        for (i, &point) in points.iter().enumerate() {
            let dist = norm3(sub3(point, center));
            if (dist - radius).abs() > tol * radius.max(1.0) {
                return Err(SpatialError::InvalidArgument(format!(
                    "point {i} is not on the specified sphere"
                )));
            }
        }
        let radius_tol = tol * radius.max(1.0);
        let radius_tol_sq = radius_tol * radius_tol;
        let n = points.len();

        // Reject near-duplicate generators (any pair within `radius_tol`). The
        // former O(n²) all-pairs scan is replaced for large n by a uniform spatial
        // hash with cell width `radius_tol`: two points closer than `radius_tol`
        // differ by < one cell on every axis, so probing the 27-cell neighbourhood
        // finds every such pair — identical accept/reject to the all-pairs scan
        // (the result is only Ok/Err, no data depends on which pair triggers it).
        // This mirrors the Voronoi-vertex dedup grid below; the small/mid-n branch
        // keeps the direct scan where the hash cannot pay for itself (measured
        // crossover: the cheap 8-flop pair test beats the hash-map probe until
        // n ≈ 2k, so gate well past it to avoid any small-n regression).
        if n >= 2048 {
            let cell_key = |p: [f64; 3]| -> (i64, i64, i64) {
                (
                    (p[0] / radius_tol).floor() as i64,
                    (p[1] / radius_tol).floor() as i64,
                    (p[2] / radius_tol).floor() as i64,
                )
            };
            let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
                std::collections::HashMap::with_capacity(n * 2);
            for (idx, &p) in points.iter().enumerate() {
                let key = cell_key(p);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if grid.get(&(key.0 + dx, key.1 + dy, key.2 + dz)).is_some_and(
                                |cands| {
                                    cands.iter().any(|&c| {
                                        norm3_squared(sub3(points[c], p)) <= radius_tol_sq
                                    })
                                },
                            ) {
                                return Err(SpatialError::InvalidArgument(
                                    "spherical voronoi requires distinct points".to_string(),
                                ));
                            }
                        }
                    }
                }
                grid.entry(key).or_default().push(idx);
            }
        } else {
            for i in 0..n {
                for j in (i + 1)..n {
                    if norm3_squared(sub3(points[i], points[j])) <= radius_tol_sq {
                        return Err(SpatialError::InvalidArgument(
                            "spherical voronoi requires distinct points".to_string(),
                        ));
                    }
                }
            }
        }

        // Face detection used to be an O(n³)-triplet × O(n)-validation gift-wrap
        // (overall O(n⁴)). The accepted faces are exactly the 3-D convex-hull
        // facets of the on-sphere generators, so compute the hull in O(n²) via
        // incremental insertion and project each facet's outward normal onto the
        // sphere — the same Voronoi vertex the gift-wrap produced, just found far
        // faster. Orientation within each triple is unspecified (the normal is
        // recomputed and flipped here), and the downstream region build is
        // order-independent, so the diagram is structurally identical.
        let facets = convex_hull_3d_facets(points, center).ok_or_else(|| {
            SpatialError::InvalidArgument(
                "spherical voronoi requires non-coplanar generators on the sphere".to_string(),
            )
        })?;
        let accepted: Vec<SvCandidateFace> = facets
            .iter()
            .map(|&[i, j, k]| {
                let mut normal = cross3(sub3(points[j], points[i]), sub3(points[k], points[i]));
                if dot3(normal, sub3(points[i], center)) < 0.0 {
                    normal = scale3(normal, -1.0);
                }
                let unit = scale3(normal, 1.0 / norm3(normal));
                let vertex = add3(center, scale3(unit, radius));
                ((i, j, k), vertex)
            })
            .collect();

        let mut vertices = Vec::with_capacity(accepted.len());
        let mut face_indices = Vec::with_capacity(accepted.len());
        if accepted.len() > 128 {
            let cell_width = radius_tol;
            let cell_key = |vertex: [f64; 3]| -> (i64, i64, i64) {
                (
                    (vertex[0] / cell_width).floor() as i64,
                    (vertex[1] / cell_width).floor() as i64,
                    (vertex[2] / cell_width).floor() as i64,
                )
            };
            let mut duplicate_grid: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
                std::collections::HashMap::with_capacity(accepted.len() * 2);
            for ((i, j, k), vertex) in accepted {
                let key = cell_key(vertex);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if duplicate_grid
                                .get(&(key.0 + dx, key.1 + dy, key.2 + dz))
                                .is_some_and(|candidates| {
                                    candidates.iter().any(|&idx| {
                                        norm3_squared(sub3(vertices[idx], vertex)) <= radius_tol_sq
                                    })
                                })
                            {
                                return Err(SpatialError::InvalidArgument(
                                    "spherical voronoi requires non-coplanar generators on the sphere"
                                        .to_string(),
                                ));
                            }
                        }
                    }
                }
                let vertex_idx = vertices.len();
                duplicate_grid.entry(key).or_default().push(vertex_idx);
                vertices.push(vertex);
                face_indices.push((i, j, k));
            }
        } else {
            for ((i, j, k), vertex) in accepted {
                if vertices
                    .iter()
                    .any(|&existing| norm3_squared(sub3(existing, vertex)) <= radius_tol_sq)
                {
                    return Err(SpatialError::InvalidArgument(
                        "spherical voronoi requires non-coplanar generators on the sphere"
                            .to_string(),
                    ));
                }
                vertices.push(vertex);
                face_indices.push((i, j, k));
            }
        }

        if vertices.len() < 4 {
            return Err(SpatialError::InvalidArgument(
                "spherical voronoi requires non-degenerate 3D point configuration".to_string(),
            ));
        }

        let mut regions = vec![Vec::new(); n];
        for (vertex_idx, &(i, j, k)) in face_indices.iter().enumerate() {
            regions[i].push(vertex_idx);
            regions[j].push(vertex_idx);
            regions[k].push(vertex_idx);
        }

        for (point_idx, region) in regions.iter_mut().enumerate() {
            if region.len() < 3 {
                return Err(SpatialError::InvalidArgument(format!(
                    "point {point_idx} does not have a valid spherical Voronoi region"
                )));
            }
            sort_spherical_region(region, points[point_idx], &vertices, center);
        }

        Ok(Self {
            points: points.to_vec(),
            vertices,
            regions,
            center,
            radius,
        })
    }
}

fn sort_spherical_region(
    region: &mut [usize],
    point: [f64; 3],
    vertices: &[[f64; 3]],
    center: [f64; 3],
) {
    let point_dir = normalize3(sub3(point, center));
    let first = normalize3(project_to_tangent(
        sub3(vertices[region[0]], center),
        point_dir,
    ));
    let ortho = cross3(point_dir, first);
    region.sort_by(|&lhs, &rhs| {
        let lhs_dir = normalize3(project_to_tangent(sub3(vertices[lhs], center), point_dir));
        let rhs_dir = normalize3(project_to_tangent(sub3(vertices[rhs], center), point_dir));
        let lhs_angle = dot3(lhs_dir, ortho).atan2(dot3(lhs_dir, first));
        let rhs_angle = dot3(rhs_dir, ortho).atan2(dot3(rhs_dir, first));
        lhs_angle.total_cmp(&rhs_angle)
    });
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3_squared(a: [f64; 3]) -> f64 {
    dot3(a, a)
}

fn norm3(a: [f64; 3]) -> f64 {
    norm3_squared(a).sqrt()
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = norm3(a);
    if n <= 1e-15 { a } else { scale3(a, 1.0 / n) }
}

fn project_to_tangent(v: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    sub3(v, scale3(normal, dot3(v, normal)))
}

// ══════════════════════════════════════════════════════════════════════
// Procrustes Analysis
// ══════════════════════════════════════════════════════════════════════

/// Result of Procrustes analysis.
#[derive(Debug, Clone)]
pub struct ProcrustesResult {
    /// Standardized version of data1.
    pub mtx1: Vec<Vec<f64>>,
    /// Standardized and transformed version of data2.
    pub mtx2: Vec<Vec<f64>>,
    /// Residual disparity (sum of squared differences after alignment).
    pub disparity: f64,
}

/// Procrustes analysis: find optimal rotation/reflection + scaling to align two point sets.
///
/// Both inputs must have the same shape (n × d). The data are centered, scaled
/// to unit Frobenius norm, then the optimal orthogonal transformation is found via SVD.
///
/// Matches `scipy.spatial.procrustes(data1, data2)`.
pub fn procrustes(
    data1: &[Vec<f64>],
    data2: &[Vec<f64>],
) -> Result<ProcrustesResult, SpatialError> {
    let n = data1.len();
    if n == 0 || data2.len() != n {
        return Err(SpatialError::InvalidArgument(
            "inputs must be non-empty and have same number of rows".to_string(),
        ));
    }
    let d = data1[0].len();
    if d == 0 {
        return Err(SpatialError::InvalidArgument(
            "inputs must have at least one dimension".to_string(),
        ));
    }
    for (i, row) in data1.iter().enumerate() {
        if row.len() != d {
            return Err(SpatialError::InvalidArgument(format!(
                "data1 row {} has {} columns, expected {}",
                i,
                row.len(),
                d
            )));
        }
    }
    for (i, row) in data2.iter().enumerate() {
        if row.len() != d {
            return Err(SpatialError::InvalidArgument(format!(
                "data2 row {} has {} columns, expected {}",
                i,
                row.len(),
                d
            )));
        }
    }

    // Center both datasets
    let mut mtx1 = center_matrix(data1);
    let mut mtx2 = center_matrix(data2);

    // Scale to unit Frobenius norm
    let norm1 = frobenius_norm(&mtx1);
    let norm2 = frobenius_norm(&mtx2);
    if norm1 > 0.0 {
        scale_matrix(&mut mtx1, 1.0 / norm1);
    }
    if norm2 > 0.0 {
        scale_matrix(&mut mtx2, 1.0 / norm2);
    }

    // scipy.spatial.procrustes uses scipy.linalg.orthogonal_procrustes
    // under the hood, which solves the rotation + scale problem
    // mtx2 ≈ mtx1 @ R^T · s by:
    //   1. M = mtx1^T @ mtx2
    //   2. (U, Σ, V^T) = svd(M)
    //   3. R = U @ V^T  (orthogonal)
    //   4. s = trace(Σ) = Σ singular_values
    //   5. mtx2_aligned = mtx2 @ R^T · s
    //
    // Pre-fix fsci built M = mtx2^T @ mtx1 (the transpose) and skipped
    // the scale factor s, which left mtx2 under-scaled and rotated by
    // R instead of R^T. The rotation difference cancels because
    // svd(M_fsci) = svd(M^T) gives U_f = V, V^T_f = U^T, so
    // R_fsci = R^T already; the missing factor was just the scale.
    // (frankenscipy-u98xh)
    // M = mtx2^T @ mtx1. Hoist k outermost so the inner j-loop is a stride-1
    // SAXPY over contiguous rows (cache-friendly + autovectorizable) instead of
    // reading mtx2[k][i]/mtx1[k][j] column-strided. Bit-identical: each m[i][j]
    // still accumulates k in 0..n order. [frankenscipy-146ld]
    let mut m = vec![vec![0.0; d]; d];
    for k in 0..n {
        let r2 = &mtx2[k];
        let r1 = &mtx1[k];
        for i in 0..d {
            let v = r2[i];
            let mi = &mut m[i];
            for j in 0..d {
                mi[j] += v * r1[j];
            }
        }
    }

    let (rotation, scale) = orthogonal_procrustes_rotation_with_scale(&m)?;

    // Apply rotation and dilation: mtx2_aligned = mtx2 · R · scale. Hoist k so
    // the inner j-loop streams rotation[k][..] contiguously (was column-strided
    // rotation[k][j]); bit-identical (each entry accumulates k in 0..d order).
    let mut aligned = vec![vec![0.0; d]; n];
    for i in 0..n {
        let r2 = &mtx2[i];
        let ai = &mut aligned[i];
        for k in 0..d {
            let v = r2[k];
            let rk = &rotation[k];
            for j in 0..d {
                ai[j] += v * rk[j];
            }
        }
    }
    for row in aligned.iter_mut() {
        for v in row.iter_mut() {
            *v *= scale;
        }
    }

    // Compute disparity: sum of squared differences
    let mut disparity = 0.0;
    for i in 0..n {
        for j in 0..d {
            disparity += (mtx1[i][j] - aligned[i][j]).powi(2);
        }
    }

    Ok(ProcrustesResult {
        mtx1,
        mtx2: aligned,
        disparity,
    })
}

fn center_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = data.len();
    let d = data[0].len();
    let mut means = vec![0.0; d];
    for row in data {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }
    data.iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - means[j]).collect())
        .collect()
}

fn frobenius_norm(data: &[Vec<f64>]) -> f64 {
    data.iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * v)
        .sum::<f64>()
        .sqrt()
}

#[expect(
    dead_code,
    reason = "scale-free wrapper pairs with the scaled Procrustes helper for internal callers"
)]
fn orthogonal_procrustes_rotation(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SpatialError> {
    let (rotation, _scale) = orthogonal_procrustes_rotation_with_scale(m)?;
    Ok(rotation)
}

fn orthogonal_procrustes_rotation_with_scale(
    m: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, f64), SpatialError> {
    let d = m.len();
    if d == 0 {
        return Ok((Vec::new(), 0.0));
    }

    let svd_result = svd(m, DecompOptions::default())
        .map_err(|err| SpatialError::InvalidArgument(format!("procrustes SVD failed: {err}")))?;

    // For M = mtx2^T @ mtx1 and M = U S V^T, the minimizer of
    // ||mtx1 - mtx2 R||_F is the polar factor R = U V^T. The optimal
    // dilation factor for the spatial procrustes is s = Σ singular
    // values (sum, not Frobenius norm) — this gives mtx2_aligned the
    // correct trace alignment with mtx1. SciPy compatible.
    let mut rotation = vec![vec![0.0; d]; d];
    for (i, row) in rotation.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (0..d)
                .map(|k| svd_result.u[i][k] * svd_result.vt[k][j])
                .sum();
        }
    }
    let scale = svd_result.s.iter().sum::<f64>();
    Ok((rotation, scale))
}

fn scale_matrix(data: &mut [Vec<f64>], factor: f64) {
    for row in data.iter_mut() {
        for v in row.iter_mut() {
            *v *= factor;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Geometric SLERP
// ══════════════════════════════════════════════════════════════════════

/// Spherical linear interpolation (SLERP) between two N-dimensional unit vectors.
///
/// Interpolates along the great circle arc on the unit sphere.
///
/// Matches `scipy.spatial.geometric_slerp(start, end, t)`.
///
/// # Arguments
/// * `start` — Starting point on the unit sphere.
/// * `end` — Ending point on the unit sphere.
/// * `t` — Interpolation parameters in [0, 1]. Can be multiple values.
/// When `true`, force `geometric_slerp` onto its ORIG serial per-t loop. Byte-identical either way.
/// Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static SPATIAL_SLERP_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn geometric_slerp(
    start: &[f64],
    end: &[f64],
    t_values: &[f64],
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if start.len() != end.len() {
        return Err(SpatialError::DimensionMismatch {
            expected: start.len(),
            actual: end.len(),
        });
    }
    let d = start.len();

    // Compute angle between start and end
    let dot: f64 = start.iter().zip(end.iter()).map(|(&a, &b)| a * b).sum();
    let dot = dot.clamp(-1.0, 1.0);
    let omega = dot.acos();

    let mut results = Vec::with_capacity(t_values.len());

    if omega.abs() < 1e-10 {
        // Points are the same: all interpolations = start
        for _ in t_values {
            results.push(start.to_vec());
        }
        return Ok(results);
    }

    let sin_omega = omega.sin();

    // Each t's interpolated point is INDEPENDENT (its `a`/`b` come from `t` + the shared `omega`,
    // then `a·start + b·end`), so distribute the t-values across cores. BYTE-IDENTICAL to the serial
    // loop: identical per-t arithmetic, results collected in t order; only the owning core changes.
    // The per-point work (two `sin` + `d` FMAs) is compute-bound. scipy's geometric_slerp is serial.
    let point_at = |t: f64| -> Vec<f64> {
        let a = ((1.0 - t) * omega).sin() / sin_omega;
        let b = (t * omega).sin() / sin_omega;
        (0..d).map(|i| a * start[i] + b * end[i]).collect()
    };

    let n_t = t_values.len();
    let nthreads =
        // Both serial conditions fold into one arm (frankenscipy-iit9c). The
        // order is preserved deliberately: the toggle load stays FIRST so the
        // A/B control is still consulted on every call, and `||` short-circuits
        // exactly as the original nested form did.
        if SPATIAL_SLERP_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
            || n_t < 2
            || (n_t as u64).saturating_mul((d + 8) as u64) < (1 << 16)
        {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(n_t)
        };

    if nthreads <= 1 {
        for &t in t_values {
            results.push(point_at(t));
        }
        return Ok(results);
    }

    let chunk = n_t.div_ceil(nthreads);
    let point_at = &point_at;
    let parts: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        t_values
            .chunks(chunk)
            .map(|c| scope.spawn(move || c.iter().map(|&t| point_at(t)).collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("slerp chunk panicked"))
            .collect()
    });
    for part in parts {
        results.extend(part);
    }
    Ok(results)
}

// ══════════════════════════════════════════════════════════════════════
// Coordinate Transforms
// ══════════════════════════════════════════════════════════════════════

/// Convert spherical coordinates (r, θ, φ) to Cartesian (x, y, z).
///
/// θ = polar angle from z-axis (0 to π), φ = azimuthal angle in x-y plane (0 to 2π).
pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> (f64, f64, f64) {
    (
        r * theta.sin() * phi.cos(),
        r * theta.sin() * phi.sin(),
        r * theta.cos(),
    )
}

/// Convert Cartesian (x, y, z) to spherical (r, θ, φ).
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    let theta = if r > 0.0 {
        (z / r).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };
    let phi = y.atan2(x);
    (r, theta, phi)
}

/// Convert cylindrical coordinates (r, θ, z) to Cartesian (x, y, z).
pub fn cylindrical_to_cartesian(rho: f64, theta: f64, z: f64) -> (f64, f64, f64) {
    (rho * theta.cos(), rho * theta.sin(), z)
}

/// Convert Cartesian (x, y, z) to cylindrical (ρ, θ, z).
pub fn cartesian_to_cylindrical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let rho = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    (rho, theta, z)
}

// ══════════════════════════════════════════════════════════════════════
// Rotation
// ══════════════════════════════════════════════════════════════════════

/// 3D rotation matrix from axis-angle representation.
///
/// Rodrigues' rotation formula. `axis` must be a unit vector.
/// `angle` is in radians.
///
/// Matches `scipy.spatial.transform.Rotation.from_rotvec`.
pub fn rotation_matrix(axis: &[f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;
    let [x, y, z] = *axis;

    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Apply a 3x3 rotation matrix to a 3D point.
pub fn rotate_point(r: &[[f64; 3]; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    ]
}

/// Compute the angle between two vectors.
pub fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

/// Cross product of two 3D vectors.
pub fn cross_3d(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Normalize a vector to unit length.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Directed Hausdorff distance between two point sets.
///
/// Returns max_{a in A} min_{b in B} ||a - b||.
/// Matches `scipy.spatial.distance.directed_hausdorff`.
pub fn directed_hausdorff(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<f64, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let dim = xa[0].len();
    if dim == 0 {
        return Err(SpatialError::InvalidArgument(
            "points must have at least one dimension".to_string(),
        ));
    }
    if xb[0].len() != dim {
        return Err(SpatialError::DimensionMismatch {
            expected: dim,
            actual: xb[0].len(),
        });
    }
    for row in xa.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::InvalidArgument(
                "all points in xa must have the same dimension".to_string(),
            ));
        }
    }
    for row in xb.iter().skip(1) {
        if row.len() != dim {
            return Err(SpatialError::InvalidArgument(
                "all points in xb must have the same dimension".to_string(),
            ));
        }
    }
    if xa.iter().flatten().any(|v| !v.is_finite()) || xb.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SpatialError::InvalidArgument(
            "hausdorff distance requires finite points".to_string(),
        ));
    }

    // Taha & Hanbury (2015) early-break: directed Hausdorff is
    //   max_{a∈xa} min_{b∈xb} d(a,b).
    // Once an inner distance drops below the running max `cmax`, that `a`'s
    // minimum is already < cmax and so cannot raise the max — abandon its scan.
    // The returned scalar is byte-identical to the full O(N·M) double loop: the
    // achieving pair's squared distance is computed by the same `sqeuclidean`,
    // and `min`/`max` are value-only (order-independent), so pruning skipped
    // pairs cannot change the result. Iterating in a fixed deterministic
    // shuffled order makes the early break effective on adversarially-ordered
    // input without affecting the result.
    // Flatten to contiguous n*dim buffers: the shuffled inner scan does a random `xb[bi]` access
    // per pair; with Vec<Vec> that is a pointer-chase + cache miss, with a flat buffer it is one
    // contiguous slice. Parallelize the outer (a) loop with a per-thread local cmax; the result is
    // max_a min_b d, which is order- and partition-independent, so per-thread cmax only changes
    // WHICH non-achieving pairs are pruned — the achieving a never early-breaks (its min_b ≥ cmax),
    // so its min is computed by the same sqeuclidean → the scalar is byte-identical to the serial
    // single-cmax loop.
    let na = xa.len();
    let nb = xb.len();
    let xa_flat: Vec<f64> = xa.iter().flat_map(|p| p.iter().copied()).collect();
    let xb_flat: Vec<f64> = xb.iter().flat_map(|p| p.iter().copied()).collect();
    let order_a = hausdorff_scan_order(na, 0x9E37_79B9_7F4A_7C15);
    let order_b = hausdorff_scan_order(nb, 0xD1B5_4A32_D192_ED03);
    let nthreads = cdist_thread_count(na, nb, dim);

    let cmax = if nthreads <= 1 {
        hausdorff_taha_chunk(&order_a, &xa_flat, &xb_flat, &order_b, dim)
    } else {
        let (xf, bf, ob) = (&xa_flat, &xb_flat, &order_b);
        let chunk_size = na.div_ceil(nthreads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = order_a
                .chunks(chunk_size)
                .map(|chunk| scope.spawn(move || hausdorff_taha_chunk(chunk, xf, bf, ob, dim)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .fold(0.0_f64, f64::max)
        })
    };
    Ok(cmax.sqrt())
}

/// Taha & Hanbury directed-Hausdorff inner kernel over a chunk of `a`-indices (flat n*dim buffers).
/// Returns the chunk's local `max_a min_b d²`; callers max-reduce across chunks. The early break
/// (`d² < cmax`) prunes only `a`s that cannot raise the max, so the result is exact.
fn hausdorff_taha_chunk(
    chunk: &[usize],
    xa_flat: &[f64],
    xb_flat: &[f64],
    order_b: &[usize],
    dim: usize,
) -> f64 {
    let mut cmax = 0.0_f64;
    for &ai in chunk {
        let a = &xa_flat[ai * dim..ai * dim + dim];
        let mut cmin = f64::INFINITY;
        for &bi in order_b {
            let b = &xb_flat[bi * dim..bi * dim + dim];
            let d_sq = sqeuclidean(a, b);
            if d_sq < cmax {
                cmin = d_sq;
                break;
            }
            if d_sq < cmin {
                cmin = d_sq;
            }
        }
        if cmin > cmax {
            cmax = cmin;
        }
    }
    cmax
}

/// Deterministic Fisher–Yates permutation of `0..n` from a fixed seed. Only the
/// scan ORDER changes (not the set), so the Hausdorff result is unaffected; a
/// pseudo-random order keeps the early break effective regardless of how the
/// caller ordered the points.
fn hausdorff_scan_order(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut state = seed | 1;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        order.swap(i, j);
    }
    order
}

/// Hausdorff distance between two point sets (symmetric).
pub fn hausdorff_distance(xa: &[Vec<f64>], xb: &[Vec<f64>]) -> Result<f64, SpatialError> {
    let d1 = directed_hausdorff(xa, xb)?;
    let d2 = directed_hausdorff(xb, xa)?;
    Ok(d1.max(d2))
}

/// Mahalanobis distance between two vectors given an inverse covariance matrix.
///
/// d = sqrt((x-y)^T * VI * (x-y))
/// Matches `scipy.spatial.distance.mahalanobis`.
pub fn mahalanobis(x: &[f64], y: &[f64], vi: &[Vec<f64>]) -> f64 {
    let n = x.len();
    if n == 0 || y.len() != n || vi.len() != n {
        return f64::NAN;
    }
    for row in vi {
        if row.len() != n {
            return f64::NAN;
        }
    }

    let diff: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();

    // V⁻¹·diff is the O(n²) bottleneck: each component is a full dot product of a row of
    // `vi` with `diff`, so SIMD-vectorise it (compute-bound, unlike a single elementwise
    // reduction). The final quadratic form `diffᵀ·(V⁻¹·diff)` is one more dot.
    let mut vi_diff = vec![0.0; n];
    for (vd, row) in vi_diff.iter_mut().zip(vi.iter()) {
        *vd = simd_dot(row, &diff);
    }
    let result = simd_dot(&diff, &vi_diff);
    result.max(0.0).sqrt()
}

/// Pairwise Mahalanobis distance matrix between the rows of `xa` and `xb` given the inverse
/// covariance matrix `vi` (d×d, symmetric).
///
/// Matches `scipy.spatial.distance.cdist(XA, XB, 'mahalanobis', VI=vi)`.
///
/// The naive form evaluates the quadratic `(xᵢ−yⱼ)ᵀ·VI·(xᵢ−yⱼ)` for every one of the na·nb
/// pairs — O(na·nb·d²). Instead expand it:
///   (x−y)ᵀ VI (x−y) = xᵀVIx + yᵀVIy − 2·(VIx)ᵀy
/// and share `VIxᵢ` across all pairs. `U = XA·VI` and `W = XB·VI` (two blocked GEMMs,
/// O((na+nb)·d²)) give the per-row quadratic forms `qx[i]=xᵢ·Uᵢ`, `qy[j]=yⱼ·Wⱼ`; the cross
/// term is the single GEMM `C = U·XBᵀ` (O(na·nb·d)). Total O((na+nb)·d² + na·nb·d) — a ~d×
/// reduction over the per-pair form. Tolerance-parity with the direct quadratic (the cross
/// term reassociates), ~1e-12, far inside distance tolerance; clamped at 0 like
/// [`mahalanobis`] before the sqrt.
///
/// When `true`, the final `sqrt`-assembly of [`cdist_mahalanobis`]/[`pdist_mahalanobis`] runs serially
/// (the ORIG loop); default `false` splits its rows across cores writing disjoint output slices.
/// BYTE-IDENTICAL either way (rows reassembled in order). `#[doc(hidden)]` — internal A/B gate.
#[doc(hidden)]
pub static MAHALANOBIS_ASSEMBLY_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn cdist_mahalanobis(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    vi: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, SpatialError> {
    if xa.is_empty() || xb.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let d = xa[0].len();
    let dim_err = |actual: usize| SpatialError::DimensionMismatch {
        expected: d,
        actual,
    };
    if vi.len() != d {
        return Err(dim_err(vi.len()));
    }
    for row in vi {
        if row.len() != d {
            return Err(dim_err(row.len()));
        }
    }
    for row in xa.iter().chain(xb.iter()) {
        if row.len() != d {
            return Err(dim_err(row.len()));
        }
    }
    let na = xa.len();
    let nb = xb.len();

    // U[i] = VI·xᵢ (VI symmetric ⇒ XA·VI gives rows VI·xᵢ); likewise W[j] = VI·yⱼ.
    let map_err = |_| SpatialError::InvalidArgument("matmul shape error".to_string());
    let u = fsci_linalg::matmul(xa, vi).map_err(map_err)?;
    let w = fsci_linalg::matmul(xb, vi).map_err(map_err)?;
    let qx: Vec<f64> = xa
        .iter()
        .zip(u.iter())
        .map(|(x, ui)| simd_dot(x, ui))
        .collect();
    let qy: Vec<f64> = xb
        .iter()
        .zip(w.iter())
        .map(|(y, wj)| simd_dot(y, wj))
        .collect();

    // Cross term C[i][j] = Uᵢ·yⱼ = xᵢᵀ VI yⱼ, via the blocked GEMM U·XBᵀ.
    let mut xbt = vec![vec![0.0; nb]; d];
    for (j, row) in xb.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            xbt[k][j] = v;
        }
    }
    let cross = fsci_linalg::matmul(&u, &xbt).map_err(map_err)?;

    // Final assembly `out[i][j] = sqrt(max(qx[i] + qy[j] − 2·C[i][j], 0))` — O(na·nb) with a `sqrt`
    // per cell, a real serial cost once the GEMMs above are parallel. Each output ROW is independent,
    // so split them across cores (BYTE-IDENTICAL — `cdist_fill_rows` reassembles rows in order). Gate
    // on na·nb work (dim=1, the assembly has no d factor); serial under the toggle.
    let assemble = |i: usize| -> Vec<f64> {
        let qxi = qx[i];
        let ci = &cross[i];
        (0..nb)
            .map(|j| (qxi + qy[j] - 2.0 * ci[j]).max(0.0).sqrt())
            .collect()
    };
    // The assembly is a CHEAP per-cell op (one sqrt), so it needs a much higher work gate than the
    // dim-factored distance compute: measured 0.65x @1M cells (64 threads on a ~5ms op) but 2.6x @16M.
    // Gate na·nb ≥ 1<<22 and cap workers so each owns ≥64k cells.
    let cells = (na as u64).saturating_mul(nb as u64);
    let nthreads = if MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || cells < (1 << 22)
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(na)
            .min((cells / (1 << 16)) as usize)
            .max(1)
    };
    let out = if nthreads <= 1 {
        (0..na).map(assemble).collect()
    } else {
        cdist_fill_rows(na, nthreads, assemble)
    };
    Ok(out)
}

/// Condensed pairwise Mahalanobis distance matrix over the rows of `x` given the inverse
/// covariance matrix `vi` (d×d, symmetric).
///
/// Matches `scipy.spatial.distance.pdist(X, 'mahalanobis', VI=vi)`: the upper triangle in
/// row-major order, `d(0,1), d(0,2), …, d(0,n-1), d(1,2), …`.
///
/// Same GEMM-expansion as [`cdist_mahalanobis`] specialised to self-pairs:
/// `q[i] = xᵢᵀVIxᵢ` from `U = X·VI`, cross term `G = U·Xᵀ`, then
/// `d(i,j) = sqrt(q[i] + q[j] − 2·G[i][j])`. O(n²·d + n·d²) versus the naive
/// O(n²·d²) per-pair quadratic — a ~d× reduction.
pub fn pdist_mahalanobis(x: &[Vec<f64>], vi: &[Vec<f64>]) -> Result<Vec<f64>, SpatialError> {
    if x.is_empty() {
        return Err(SpatialError::EmptyData);
    }
    let d = x[0].len();
    let dim_err = |actual: usize| SpatialError::DimensionMismatch {
        expected: d,
        actual,
    };
    if vi.len() != d {
        return Err(dim_err(vi.len()));
    }
    for row in vi {
        if row.len() != d {
            return Err(dim_err(row.len()));
        }
    }
    for row in x {
        if row.len() != d {
            return Err(dim_err(row.len()));
        }
    }
    let n = x.len();

    let map_err = |_| SpatialError::InvalidArgument("matmul shape error".to_string());
    let u = fsci_linalg::matmul(x, vi).map_err(map_err)?; // U[i] = VI·xᵢ
    let q: Vec<f64> = x
        .iter()
        .zip(u.iter())
        .map(|(xi, ui)| simd_dot(xi, ui))
        .collect();

    // Cross term G[i][j] = Uᵢ·xⱼ = xᵢᵀ VI xⱼ via the blocked GEMM U·Xᵀ (full matrix; only
    // the strict upper triangle is read — the symmetric lower half is redundant work but
    // the blocked GEMM still dwarfs the per-pair quadratic).
    let mut xt = vec![vec![0.0; n]; d];
    for (i, row) in x.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            xt[k][i] = v;
        }
    }
    let g = fsci_linalg::matmul(&u, &xt).map_err(map_err)?;

    // Final condensed assembly `d(i,j) = sqrt(max(q[i]+q[j]−2·G[i][j], 0))` for j>i — O(n²) serial
    // with a sqrt per cell, a real cost once the GEMMs are parallel. Row i's (n−1−i) outputs occupy the
    // contiguous span `out[offset(i)..offset(i+1)]`, so split the ROWS across cores and let each thread
    // write its disjoint output slice (via split_at_mut). BYTE-IDENTICAL — same values at the same
    // condensed positions. Cheap per-cell → gate high (n²/2 ≥ 1<<22) + ≥64k-cells/worker cap.
    let total = n * n.saturating_sub(1) / 2;
    let cells = total as u64;
    let nthreads = if MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || cells < (1 << 22)
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 2)
            .min((cells / (1 << 16)) as usize)
            .max(1)
    };
    if nthreads <= 1 {
        let mut out = Vec::with_capacity(total);
        for i in 0..n {
            let qi = q[i];
            let gi = &g[i];
            for j in (i + 1)..n {
                out.push((qi + q[j] - 2.0 * gi[j]).max(0.0).sqrt());
            }
        }
        return Ok(out);
    }
    let offset = |i: usize| i * (2 * n - i - 1) / 2;
    let mut out = vec![0.0f64; total];
    let chunk_rows = n.div_ceil(nthreads);
    let (q_ref, g_ref) = (&q, &g);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = &mut out;
        let mut i0 = 0usize;
        while i0 < n {
            let i1 = (i0 + chunk_rows).min(n);
            let (head, tail) = rest.split_at_mut(offset(i1) - offset(i0));
            rest = tail;
            let base = offset(i0);
            scope.spawn(move || {
                for i in i0..i1 {
                    let qi = q_ref[i];
                    let gi = &g_ref[i];
                    let k0 = offset(i) - base;
                    for (slot, j) in head[k0..].iter_mut().zip((i + 1)..n) {
                        *slot = (qi + q_ref[j] - 2.0 * gi[j]).max(0.0).sqrt();
                    }
                }
            });
            i0 = i1;
        }
    });
    Ok(out)
}

/// Standardized Euclidean distance.
///
/// d = sqrt(sum((x-y)² / v)) where v is the per-component variance.
/// Matches `scipy.spatial.distance.seuclidean`.
pub fn seuclidean(x: &[f64], y: &[f64], v: &[f64]) -> f64 {
    if x.is_empty() || x.len() != y.len() || x.len() != v.len() {
        return f64::NAN;
    }
    // Σ (x-y)²/v with v<=0 ⇒ NaN. Compute-heavy (sub, square, divide, compare, select per
    // element), so 8-wide SIMD wins. `simd_le(0)` reproduces the `vi <= 0.0 ⇒ NaN` branch
    // via a lane mask; NaN propagates through the sum exactly as the scalar fold.
    // Σ (x-y)²/v with v<=0 ⇒ NaN. Compute-heavy (sub, square, divide, compare, select per
    // element), so 8-wide SIMD wins ~2x. `simd_le(0)` reproduces the `vi <= 0.0 ⇒ NaN` branch
    // via a lane mask; NaN propagates through the sum exactly as the scalar fold.
    use std::simd::{Select, Simd, cmp::SimdPartialOrd, num::SimdFloat};
    const L: usize = 8;
    let n = x.len();
    let nan_v = Simd::<f64, L>::splat(f64::NAN);
    let zero = Simd::<f64, L>::splat(0.0);
    let mut acc = Simd::<f64, L>::splat(0.0);
    let mut i = 0;
    while i + L <= n {
        let d = Simd::<f64, L>::from_slice(&x[i..i + L]) - Simd::from_slice(&y[i..i + L]);
        let vv = Simd::<f64, L>::from_slice(&v[i..i + L]);
        acc += vv.simd_le(zero).select(nan_v, (d * d) / vv);
        i += L;
    }
    let mut s = acc.reduce_sum();
    while i < n {
        let vi = v[i];
        s += if vi <= 0.0 {
            f64::NAN
        } else {
            (x[i] - y[i]).powi(2) / vi
        };
        i += 1;
    }
    s.sqrt()
}

/// Weighted Minkowski distance.
///
/// Matches `scipy.spatial.distance.wminkowski`.
pub fn wminkowski(x: &[f64], y: &[f64], p: f64, w: &[f64]) -> f64 {
    if x.is_empty() || x.len() != y.len() || x.len() != w.len() || p <= 0.0 {
        return f64::NAN;
    }
    x.iter()
        .zip(y.iter())
        .zip(w.iter())
        .map(|((&xi, &yi), &wi)| (wi * (xi - yi).abs()).powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// pth power of the L^p (Minkowski) distance between rows of two point arrays.
///
/// Matches `scipy.spatial.minkowski_distance_p(x, y, p)`. Each inner `Vec<f64>`
/// is one K-dimensional point; the reduction is over that last axis. For
/// efficiency the pth root is *not* taken — for `p == 1` or `p == ∞` this is
/// already the true L^p distance, otherwise it is `distance**p`.
///
/// `xa` and `xb` must have the same number of rows, or one of them may have a
/// single row which is broadcast against the other (mirroring numpy's
/// leading-axis broadcasting). Every compared row pair must share the same
/// length K. Branches on `p` exactly as scipy does (`∞ → max`, `1 → sum`,
/// `else → sum(|·|^p)`).
pub fn minkowski_distance_p(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    p: f64,
) -> Result<Vec<f64>, SpatialError> {
    minkowski_rowwise(xa, xb, p, false)
}

/// L^p (Minkowski) distance between rows of two point arrays.
///
/// Matches `scipy.spatial.minkowski_distance(x, y, p)`: the same row-wise
/// reduction as [`minkowski_distance_p`] but with the pth root applied for
/// finite `p > 1` (`p == 1` and `p == ∞` need no root).
pub fn minkowski_distance(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    p: f64,
) -> Result<Vec<f64>, SpatialError> {
    minkowski_rowwise(xa, xb, p, true)
}

fn minkowski_rowwise(
    xa: &[Vec<f64>],
    xb: &[Vec<f64>],
    p: f64,
    take_root: bool,
) -> Result<Vec<f64>, SpatialError> {
    let n = match (xa.len(), xb.len()) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        (a, b) => {
            return Err(SpatialError::InvalidArgument(format!(
                "minkowski_distance: incompatible row counts {a} and {b}"
            )));
        }
    };

    let row_p = |a: &[f64], b: &[f64]| -> Result<f64, SpatialError> {
        if a.len() != b.len() {
            return Err(SpatialError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        // Mirror scipy's exact three-way branch on p.
        let val = if p == f64::INFINITY {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| (bi - ai).abs())
                .fold(0.0_f64, f64::max)
        } else if p == 1.0 {
            a.iter()
                .zip(b.iter())
                .map(|(&ai, &bi)| (bi - ai).abs())
                .sum()
        } else {
            // Small integer exponents: repeated multiplication + SIMD instead of per-element
            // powf (scipy's minkowski_distance pays a pow() per element regardless).
            let pr = p.round();
            if pr == p && (2.0..=64.0).contains(&pr) {
                minkowski_pow_sum(a, b, pr as u32)
            } else {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| (bi - ai).abs().powf(p))
                    .sum()
            }
        };
        Ok(val)
    };

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = &xa[if xa.len() == 1 { 0 } else { i }];
        let b = &xb[if xb.len() == 1 { 0 } else { i }];
        let mut d = row_p(a, b)?;
        // scipy only takes the root for finite p != 1 (p == ∞ returns the max).
        if take_root && p != 1.0 && p != f64::INFINITY {
            d = d.powf(1.0 / p);
        }
        out.push(d);
    }
    Ok(out)
}

/// Yule dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.yule`.
pub fn yule(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize; // true-false
    let mut cft = 0usize; // false-true
    let mut ctt = 0usize; // true-true
    let mut cff = 0usize; // false-false

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            (false, false) => cff += 1,
        }
    }

    let r = ctf * cft;
    let s = ctt * cff;
    if r + s == 0 {
        0.0
    } else {
        2.0 * r as f64 / (r + s) as f64
    }
}

/// Dice dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.dice`.
pub fn dice(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    // scipy.spatial.distance.dice computes (ctf+cft)/(ctf+cft+2*ctt); for two
    // entirely-false vectors that is 0/0 = NaN (scipy emits a divide warning and
    // returns nan). The previous guard returned 0.0 there, an undocumented
    // divergence — drop it so the degenerate case matches scipy's NaN.
    let ntt = (ctf + cft) as f64;
    ntt / (ntt + 2.0 * ctt as f64)
}

/// Kulsinski dissimilarity for boolean vectors.
///
/// Implements the historical `scipy.spatial.distance.kulsinski` formula
/// `(ntf + nft - ntt + n) / (ntf + nft + n)`. NOTE: SciPy deprecated this metric
/// and **removed it in SciPy 1.15+** (gone from 1.17), so there is no current
/// SciPy oracle; retained here for backward compatibility.
pub fn kulsinski(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len() as f64;
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    let ntt = (ctf + cft) as f64;
    if ctt == 0 && ntt == 0.0 {
        0.0
    } else {
        (ntt + n - ctt as f64) / (ntt + n)
    }
}

/// Rogerstanimoto dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.rogerstanimoto`.
pub fn rogerstanimoto(u: &[bool], v: &[bool]) -> f64 {
    let mut ndiff = 0usize;
    let mut nsame = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        if a == b {
            nsame += 1;
        } else {
            ndiff += 1;
        }
    }

    let r = 2 * ndiff;
    (r as f64) / (nsame + r) as f64
}

/// Russell-Rao dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.russellrao`.
pub fn russellrao(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len();
    if n == 0 {
        return 0.0;
    }
    let ctt = u.iter().zip(v.iter()).filter(|&(a, b)| *a && *b).count();
    (n - ctt) as f64 / n as f64
}

/// Sokal-Michener dissimilarity for boolean vectors.
///
/// Equal to [`rogerstanimoto`] (`2(ntf+nft) / (ntt+nff + 2(ntf+nft))`). NOTE:
/// SciPy deprecated this metric and **removed it in SciPy 1.15+** (gone from
/// 1.17); retained here for backward compatibility.
pub fn sokalmichener(u: &[bool], v: &[bool]) -> f64 {
    rogerstanimoto(u, v) // Same formula
}

/// Sokal-Sneath dissimilarity for boolean vectors.
///
/// Matches `scipy.spatial.distance.sokalsneath`.
pub fn sokalsneath(u: &[bool], v: &[bool]) -> f64 {
    let mut ctf = 0usize;
    let mut cft = 0usize;
    let mut ctt = 0usize;

    for (&a, &b) in u.iter().zip(v.iter()) {
        match (a, b) {
            (true, true) => ctt += 1,
            (true, false) => ctf += 1,
            (false, true) => cft += 1,
            _ => {}
        }
    }

    // scipy.spatial.distance.sokalsneath raises ValueError ("not defined for
    // vectors that are entirely false") for the all-false case, which is 2R/(2R+ctt)
    // = 0/0. Our f64 signature can't raise; return NaN (the f64 "undefined" signal)
    // instead of the previous, silently-wrong 0.0.
    let r = (2 * (ctf + cft)) as f64;
    r / (r + ctt as f64)
}

/// Matching dissimilarity for boolean vectors.
///
/// Fraction of positions where the vectors disagree — identical to [`hamming`]
/// for boolean input. NOTE: `scipy.spatial.distance.matching` was an alias for
/// `hamming` that SciPy **removed in 1.15+** (gone from 1.17); use [`hamming`]
/// for current SciPy parity.
pub fn matching(u: &[bool], v: &[bool]) -> f64 {
    let n = u.len();
    if n == 0 {
        return 0.0;
    }
    let ndiff = u.iter().zip(v.iter()).filter(|&(a, b)| *a != *b).count();
    ndiff as f64 / n as f64
}

/// Compute all pairwise distances using a specified metric function.
///
/// Returns condensed distance vector.
pub fn pdist_func<F>(data: &[Vec<f64>], metric: F) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    let n = data.len();
    // Row `i` contributes the condensed block [(i,i+1)..(i,n-1)]; the blocks are INDEPENDENT and the
    // condensed vector is exactly block_0 ++ block_1 ++ … So split contiguous i-ranges across cores
    // and concatenate the per-chunk blocks in i-order: BYTE-IDENTICAL to the serial double loop
    // (identical per-pair `metric` calls, identical order). scipy's callable-metric pdist is serial.
    let total = n * (n.saturating_sub(1)) / 2;
    let nthreads = if SPATIAL_CDIST_FUNC_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || n < 3
        || (total as u64) < (1 << 14)
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n)
    };

    if nthreads <= 1 {
        let mut dists = Vec::with_capacity(total);
        for i in 0..n {
            for j in i + 1..n {
                dists.push(metric(&data[i], &data[j]));
            }
        }
        return dists;
    }

    let block_of =
        |i: usize| -> Vec<f64> { (i + 1..n).map(|j| metric(&data[i], &data[j])).collect() };
    let block_of = &block_of;
    let chunk = n.div_ceil(nthreads);
    let parts: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        (0..n)
            .step_by(chunk)
            .map(|i0| {
                scope.spawn(move || {
                    let i1 = (i0 + chunk).min(n);
                    (i0..i1).map(block_of).collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("pdist_func chunk panicked"))
            .collect()
    });
    let mut dists = Vec::with_capacity(total);
    for part in parts {
        for block in part {
            dists.extend(block);
        }
    }
    dists
}

/// When `true`, force `cdist_func` onto its ORIG serial row map. Byte-identical either way.
/// Benchmark knob for the same-binary A/B.
#[doc(hidden)]
pub static SPATIAL_CDIST_FUNC_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Compute pairwise distances between two sets using a metric function.
///
/// Returns an m×n matrix.
pub fn cdist_func<F>(xa: &[Vec<f64>], xb: &[Vec<f64>], metric: F) -> Vec<Vec<f64>>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    // Each `xa` row's distances to all of `xb` are INDEPENDENT, so fan contiguous row-chunks across
    // cores. BYTE-IDENTICAL to the serial map: identical per-pair `metric` calls, rows assembled in
    // order; only the owning core changes. Custom-metric cdist over a compute-bound `metric`
    // (euclidean/minkowski) is O(m·n·d) — worth splitting; scipy's callable-metric cdist is serial.
    let row = |a: &[f64]| -> Vec<f64> { xb.iter().map(|b| metric(a, b)).collect() };
    let m = xa.len();
    let nthreads = if SPATIAL_CDIST_FUNC_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        || m < 2
        || (m as u64).saturating_mul(xb.len() as u64) < (1 << 14)
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(m)
    };

    if nthreads <= 1 {
        return xa.iter().map(|a| row(a)).collect();
    }

    let chunk = m.div_ceil(nthreads);
    let row = &row;
    let parts: Vec<Vec<Vec<f64>>> = std::thread::scope(|scope| {
        xa.chunks(chunk)
            .map(|c| scope.spawn(move || c.iter().map(|a| row(a)).collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("cdist_func chunk panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(m);
    for part in parts {
        out.extend(part);
    }
    out
}

/// Compute the nearest neighbor for each point in a dataset.
///
/// Returns (indices, distances) of nearest neighbors.
/// For empty input, returns empty vectors.
/// For single-element input, returns `(vec![None], vec![])` conceptually,
/// but since each point needs at least one other point to have a neighbor,
/// the index is undefined and the distance is `INFINITY`.
pub fn nearest_neighbors(data: &[Vec<f64>]) -> (Vec<Option<usize>>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Fast path: a KD-tree locates each point's nearest neighbour in O(log n)
    // average instead of the O(n) inner scan, so all-points 1-NN drops from
    // O(n^2) to O(n log n). It is BYTE-IDENTICAL to the brute force below:
    //   * `sqeuclidean(a,b).sqrt() == euclidean(a,b)` (same map/sum/sqrt), and
    //     the tree stores each point verbatim keyed by its original index, so a
    //     matched pair yields identical distance bits;
    //   * `nn_search_lowest_index` visits every point whose split-plane bound
    //     `diff^2 <= best_dist_sq`; since best_dist_sq never drops below the true
    //     minimum, EVERY minimum-distance point is visited, and the
    //     `index < best_idx` tie-break reproduces the brute force's lowest-index
    //     pick (it keeps the first `d < min_dist`, i.e. smallest index on ties).
    // Any input the tree can't index (ragged rows, non-finite) falls through to
    // the brute force, so behaviour is unconditionally preserved.
    if let Ok(tree) = KDTree::new(data) {
        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for (i, point) in data.iter().enumerate() {
            let mut best_idx = usize::MAX;
            let mut best_dist_sq = f64::INFINITY;
            if !tree.nodes.is_empty() {
                nn_search_lowest_index(
                    &tree.nodes,
                    &tree.points,
                    tree.dim,
                    0,
                    point,
                    i,
                    &mut best_idx,
                    &mut best_dist_sq,
                );
            }
            if best_idx == usize::MAX {
                indices.push(None);
                distances.push(f64::INFINITY);
            } else {
                indices.push(Some(best_idx));
                distances.push(best_dist_sq.sqrt());
            }
        }
        return (indices, distances);
    }

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        let mut min_idx: Option<usize> = None;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean(&data[i], &data[j]);
            if d < min_dist {
                min_dist = d;
                min_idx = Some(j);
            }
        }
        indices.push(min_idx);
        distances.push(min_dist);
    }

    (indices, distances)
}

/// Nearest-neighbour KD-tree descent that excludes one index and breaks ties by
/// lowest index, reproducing the brute-force all-pairs scan bit-for-bit.
///
/// `diff * diff <= best_dist_sq` (rather than `<`) keeps equal-distance subtrees
/// in play so a lower-indexed point at the same distance is never missed.
/// The eight parameters are the RECURSION STATE of a KD-tree descent: the
/// immutable tree (`nodes`, `points`, `dim`), where we are (`node_idx`), what
/// we are looking for (`query`, and the exclusions/limits), and the running
/// best carried by `&mut`. Bundling them into a struct to satisfy the 7-argument
/// limit would put an indirection in the hot path of a tree walk that runs once
/// per visited node, to buy nothing a reader of this signature does not already
/// get from the names (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn nn_search_lowest_index(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    exclude: usize,
    best_idx: &mut usize,
    best_dist_sq: &mut f64,
) {
    let node = &nodes[node_idx];
    let point = &points[node_idx * dim..node_idx * dim + dim];
    if node.index != exclude {
        let dist_sq = sqeuclidean(query, point);
        if dist_sq < *best_dist_sq || (dist_sq == *best_dist_sq && node.index < *best_idx) {
            *best_dist_sq = dist_sq;
            *best_idx = node.index;
        }
    }

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        nn_search_lowest_index(
            nodes,
            points,
            dim,
            near_idx,
            query,
            exclude,
            best_idx,
            best_dist_sq,
        );
    }
    if diff * diff <= *best_dist_sq
        && let Some(far_idx) = far
    {
        nn_search_lowest_index(
            nodes,
            points,
            dim,
            far_idx,
            query,
            exclude,
            best_idx,
            best_dist_sq,
        );
    }
}

/// Compute the k nearest neighbors for each point.
///
/// Returns (indices, distances) where each inner vec has k elements.
pub fn k_nearest_neighbors(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data.len();
    let mut all_indices = Vec::with_capacity(n);
    let mut all_distances = Vec::with_capacity(n);

    // Fast path: a KD-tree finds each point's k nearest in O(log n + k) average
    // rather than scanning all m = n-1 distances, so all-points k-NN drops from
    // O(n^2) to O(n log n + n k). BYTE-IDENTICAL to the brute force below:
    //   * the bounded set is ordered by the SAME composite key (squared distance
    //     via total_cmp, then ascending index) the brute force sorts by, and a
    //     point's squared distance feeds the identical `sqrt` -> `euclidean`
    //     bits, so matched results carry identical distance bits;
    //   * `knn_search_composite` prunes a far subtree only when its split-plane
    //     bound `diff*diff` STRICTLY exceeds the current k-th squared distance, so
    //     every point that could enter the k smallest (including equal-distance
    //     points with a smaller index) is visited and the composite order then
    //     selects exactly the brute force's k.
    // Inputs the tree can't index (ragged rows / non-finite) fall through.
    if k > 0
        && let Ok(tree) = KDTree::new(data)
    {
        // Each point's k-NN descent reads the shared, immutable tree and writes only its
        // own output row, so the queries are independent. Running them across threads into
        // index-ordered slots is byte-identical to the serial loop (same per-point
        // `knn_search_composite` result, reassembled in the original 0..n order).
        let query = |i: usize| -> (Vec<usize>, Vec<f64>) {
            let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
            if !tree.nodes.is_empty() {
                knn_search_composite(
                    &tree.nodes,
                    &tree.points,
                    tree.dim,
                    0,
                    &data[i],
                    i,
                    k,
                    &mut best,
                );
            }
            (
                best.iter().map(|&(_, idx)| idx).collect(),
                best.iter().map(|&(ds, _)| ds.sqrt()).collect(),
            )
        };
        let work = (n as u64).saturating_mul((k as u64).max(1));
        let nthreads = if work < (1 << 14) || n < 64 {
            1
        } else {
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(n)
        };
        if nthreads <= 1 {
            for i in 0..n {
                let (idx, dst) = query(i);
                all_indices.push(idx);
                all_distances.push(dst);
            }
        } else {
            let mut idx_out: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut dst_out: Vec<Vec<f64>> = vec![Vec::new(); n];
            let chunk = n.div_ceil(nthreads);
            let query = &query;
            std::thread::scope(|scope| {
                for (t, (islot, dslot)) in idx_out
                    .chunks_mut(chunk)
                    .zip(dst_out.chunks_mut(chunk))
                    .enumerate()
                {
                    let base = t * chunk;
                    scope.spawn(move || {
                        for (off, (io, do_)) in islot.iter_mut().zip(dslot.iter_mut()).enumerate() {
                            let (idx, dst) = query(base + off);
                            *io = idx;
                            *do_ = dst;
                        }
                    });
                }
            });
            all_indices = idx_out;
            all_distances = dst_out;
        }
        return (all_indices, all_distances);
    }

    // Composite order (distance, then ascending index). Because the candidates
    // are produced in ascending `j`, a stable sort by distance breaks ties by
    // ascending index — exactly what this total order reproduces.
    let by_dist_then_idx =
        |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean(&data[i], &data[j])))
            .collect();

        // Only the k smallest are needed, so partition in O(m) and sort just
        // those instead of fully sorting all m = n-1 distances (O(m log m)).
        // Byte-identical to the full sort: the composite key is a strict total
        // order (indices are unique), so the selected-and-sorted prefix matches
        // the stable-by-distance prefix element-for-element.
        let k_actual = k.min(dists.len());
        if k_actual < dists.len() {
            dists.select_nth_unstable_by(k_actual, by_dist_then_idx);
        }
        dists[..k_actual].sort_by(by_dist_then_idx);

        all_indices.push(dists[..k_actual].iter().map(|&(idx, _)| idx).collect());
        all_distances.push(dists[..k_actual].iter().map(|&(_, d)| d).collect());
    }

    (all_indices, all_distances)
}

/// Bounded k-nearest KD-tree descent (excluding one index) whose result is the
/// `k` smallest points ordered by the composite (squared-distance, index) key —
/// the same set and order as the brute-force all-pairs select+sort.
///
/// `best` stays sorted ascending by that key, capped at `k`. The far subtree is
/// pruned only when `diff*diff` exceeds the current k-th squared distance, so
/// equal-distance lower-index points are never missed.
/// The eight parameters are the RECURSION STATE of a KD-tree descent: the
/// immutable tree (`nodes`, `points`, `dim`), where we are (`node_idx`), what
/// we are looking for (`query`, and the exclusions/limits), and the running
/// best carried by `&mut`. Bundling them into a struct to satisfy the 7-argument
/// limit would put an indirection in the hot path of a tree walk that runs once
/// per visited node, to buy nothing a reader of this signature does not already
/// get from the names (frankenscipy-iit9c).
#[allow(clippy::too_many_arguments)]
fn knn_search_composite(
    nodes: &[KDNode],
    points: &[f64],
    dim: usize,
    node_idx: usize,
    query: &[f64],
    exclude: usize,
    k: usize,
    best: &mut Vec<(f64, usize)>,
) {
    use std::cmp::Ordering;
    let node = &nodes[node_idx];
    let point = &points[node_idx * dim..node_idx * dim + dim];
    if node.index != exclude {
        let dist_sq = sqeuclidean(query, point);
        let should_insert = best.len() < k
            || best.last().is_none_or(|w| {
                dist_sq.total_cmp(&w.0).then(node.index.cmp(&w.1)) == Ordering::Less
            });
        if should_insert {
            let pos = best.partition_point(|p| {
                p.0.total_cmp(&dist_sq).then(p.1.cmp(&node.index)) == Ordering::Less
            });
            best.insert(pos, (dist_sq, node.index));
            if best.len() > k {
                best.pop();
            }
        }
    }

    let diff = query[node.split_dim] - point[node.split_dim];
    let (near, far) = if diff <= 0.0 {
        (node.left, node.right)
    } else {
        (node.right, node.left)
    };

    if let Some(near_idx) = near {
        knn_search_composite(nodes, points, dim, near_idx, query, exclude, k, best);
    }
    let worst = if best.len() < k {
        f64::INFINITY
    } else {
        best[best.len() - 1].0
    };
    if diff * diff <= worst
        && let Some(far_idx) = far
    {
        knn_search_composite(nodes, points, dim, far_idx, query, exclude, k, best);
    }
}

/// Compute the centroid of a set of points.
pub fn centroid(points: &[Vec<f64>]) -> Vec<f64> {
    if points.is_empty() {
        return vec![];
    }
    let d = points[0].len();
    for p in points {
        if p.len() != d {
            return vec![f64::NAN; d];
        }
    }
    let n = points.len() as f64;
    let mut center = vec![0.0; d];
    for p in points {
        for (j, &v) in p.iter().enumerate() {
            center[j] += v;
        }
    }
    for v in &mut center {
        *v /= n;
    }
    center
}

/// Compute the medoid (point minimizing sum of distances) of a set.
///
/// Returns `None` for an empty point set.
pub fn medoid(points: &[Vec<f64>]) -> Option<usize> {
    let n = points.len();
    if n == 0 {
        return None;
    }

    // Each point's total distance to all others is an independent O(n·d) reduction; the
    // argmin scan uses a strict `<` (first/lowest index wins ties). Compute the totals in
    // parallel into index order, then run the identical serial argmin — byte-identical
    // (each total still sums in the same j order, and the tie-break is unchanged).
    let dim = points[0].len();
    let total_at = |i: usize| -> f64 {
        (0..n)
            .filter(|&j| j != i)
            .map(|j| euclidean(&points[i], &points[j]))
            .sum()
    };
    let nthreads = cdist_thread_count(n, n, dim);
    let totals: Vec<f64> = if nthreads <= 1 {
        (0..n).map(&total_at).collect()
    } else {
        let chunk = n.div_ceil(nthreads);
        let total_at = &total_at;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..nthreads)
                .filter_map(|t| {
                    let i0 = t * chunk;
                    if i0 >= n {
                        return None;
                    }
                    let i1 = (i0 + chunk).min(n);
                    Some(scope.spawn(move || (i0..i1).map(total_at).collect::<Vec<f64>>()))
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("medoid worker panicked"))
                .collect()
        })
    };

    let mut best = 0;
    let mut best_total = f64::INFINITY;
    for (i, &total) in totals.iter().enumerate() {
        if total < best_total {
            best_total = total;
            best = i;
        }
    }
    Some(best)
}

/// Compute the diameter of a point set (maximum pairwise distance).
pub fn diameter(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    // Upper-triangle max-of-pairwise-distances. The combine rule (NaN if any operand is
    // NaN, else max) is commutative and associative, so a parallel row-split reduction
    // yields the identical scalar — NaN iff any pairwise distance is NaN, else the max.
    let dim = if n > 0 { points[0].len() } else { 0 };
    let row_max = |i: usize| -> f64 {
        let mut m = 0.0f64;
        for j in i + 1..n {
            let d = euclidean(&points[i], &points[j]);
            m = if m.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                m.max(d)
            };
        }
        m
    };
    let combine = |a: f64, b: f64| -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    };
    let nthreads = cdist_thread_count(n, n, dim);
    if nthreads <= 1 {
        return (0..n).map(row_max).fold(0.0f64, combine);
    }
    let chunk = n.div_ceil(nthreads);
    let row_max = &row_max;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let i0 = t * chunk;
                if i0 >= n {
                    return None;
                }
                let i1 = (i0 + chunk).min(n);
                Some(scope.spawn(move || (i0..i1).map(row_max).fold(0.0f64, combine)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("diameter worker panicked"))
            .fold(0.0f64, combine)
    })
}

/// Compute the spread (average distance from centroid) of a point set.
pub fn spread(points: &[Vec<f64>]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let center = centroid(points);
    let n = points.len() as f64;
    points.iter().map(|p| euclidean(p, &center)).sum::<f64>() / n
}

// ══════════════════════════════════════════════════════════════════════
// Rotation (scipy.spatial.transform.Rotation)
// ══════════════════════════════════════════════════════════════════════

/// 3D rotation represented internally as a unit quaternion.
///
/// Matches `scipy.spatial.transform.Rotation`.
///
/// Quaternion convention: `[x, y, z, w]` where `w` is the scalar component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
    quat: [f64; 4],
}

impl Rotation {
    /// Create the identity rotation.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            quat: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation from a unit quaternion `[x, y, z, w]`.
    ///
    /// The quaternion is normalized if not already unit.
    #[must_use]
    pub fn from_quat(quat: [f64; 4]) -> Self {
        let [x, y, z, w] = quat;
        let norm = (x * x + y * y + z * z + w * w).sqrt();
        if norm < 1e-15 {
            return Self::identity();
        }
        Self {
            quat: [x / norm, y / norm, z / norm, w / norm],
        }
    }

    /// Create a rotation from a 3x3 rotation matrix.
    #[must_use]
    pub fn from_matrix(matrix: [[f64; 3]; 3]) -> Self {
        let m = matrix;
        let trace = m[0][0] + m[1][1] + m[2][2];

        let (x, y, z, w) = if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            let w = 0.25 / s;
            let x = (m[2][1] - m[1][2]) * s;
            let y = (m[0][2] - m[2][0]) * s;
            let z = (m[1][0] - m[0][1]) * s;
            (x, y, z, w)
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
            let w = (m[2][1] - m[1][2]) / s;
            let x = 0.25 * s;
            let y = (m[0][1] + m[1][0]) / s;
            let z = (m[0][2] + m[2][0]) / s;
            (x, y, z, w)
        } else if m[1][1] > m[2][2] {
            let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
            let w = (m[0][2] - m[2][0]) / s;
            let x = (m[0][1] + m[1][0]) / s;
            let y = 0.25 * s;
            let z = (m[1][2] + m[2][1]) / s;
            (x, y, z, w)
        } else {
            let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
            let w = (m[1][0] - m[0][1]) / s;
            let x = (m[0][2] + m[2][0]) / s;
            let y = (m[1][2] + m[2][1]) / s;
            let z = 0.25 * s;
            (x, y, z, w)
        };

        Self::from_quat([x, y, z, w])
    }

    /// Create a rotation from a rotation vector (axis-angle representation).
    ///
    /// The rotation vector's direction is the axis and its magnitude is the angle in radians.
    #[must_use]
    pub fn from_rotvec(rotvec: [f64; 3]) -> Self {
        let [rx, ry, rz] = rotvec;
        let angle = (rx * rx + ry * ry + rz * rz).sqrt();

        if angle < 1e-15 {
            return Self::identity();
        }

        let half_angle = angle / 2.0;
        let s = half_angle.sin() / angle;

        Self::from_quat([rx * s, ry * s, rz * s, half_angle.cos()])
    }

    /// Create a rotation from Euler angles.
    ///
    /// `seq` is a 3-character string like "xyz", "zyx", "XYZ", etc.
    /// Lowercase = extrinsic (rotations about fixed axes), uppercase = intrinsic (rotations about body axes).
    /// `angles` are in radians.
    #[must_use]
    pub fn from_euler(seq: &str, angles: [f64; 3]) -> Self {
        if seq.len() != 3 {
            return Self::identity();
        }

        let intrinsic = seq.chars().next().is_some_and(|c| c.is_uppercase());
        let axes: Vec<char> = seq.to_lowercase().chars().collect();

        let mut result = Self::identity();
        let order: Vec<usize> = if intrinsic {
            vec![0, 1, 2]
        } else {
            vec![2, 1, 0]
        };

        for &i in &order {
            let idx = if intrinsic { i } else { 2 - i };
            let angle = angles[idx];
            let elem = Self::from_single_axis(axes[idx], angle);
            result = if intrinsic {
                result.multiply(&elem)
            } else {
                elem.multiply(&result)
            };
        }

        result
    }

    fn from_single_axis(axis: char, angle: f64) -> Self {
        let half = angle / 2.0;
        let (s, c) = (half.sin(), half.cos());
        match axis {
            'x' => Self::from_quat([s, 0.0, 0.0, c]),
            'y' => Self::from_quat([0.0, s, 0.0, c]),
            'z' => Self::from_quat([0.0, 0.0, s, c]),
            _ => Self::identity(),
        }
    }

    /// Return the quaternion representation `[x, y, z, w]`.
    #[must_use]
    pub fn as_quat(&self) -> [f64; 4] {
        self.quat
    }

    /// Return the 3x3 rotation matrix.
    #[must_use]
    pub fn as_matrix(&self) -> [[f64; 3]; 3] {
        let [x, y, z, w] = self.quat;

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }

    /// Return the rotation vector (axis * angle).
    #[must_use]
    pub fn as_rotvec(&self) -> [f64; 3] {
        let [x, y, z, w] = self.quat;
        let sin_half = (x * x + y * y + z * z).sqrt();

        if sin_half < 1e-15 {
            return [0.0, 0.0, 0.0];
        }

        let angle = 2.0 * sin_half.atan2(w.abs());
        let scale = if w < 0.0 {
            -angle / sin_half
        } else {
            angle / sin_half
        };

        [x * scale, y * scale, z * scale]
    }

    /// Return Euler angles for the given sequence.
    ///
    /// `seq` is a 3-character string like "xyz", "zyx", etc.
    #[must_use]
    pub fn as_euler(&self, seq: &str) -> [f64; 3] {
        use std::f64::consts::{FRAC_PI_2, PI, TAU};

        // Faithful port of `scipy.spatial.transform.Rotation.as_euler`
        // (Bernardes & Viollet 2022 quaternion method, via scipy's `_get_angles`).
        // Operates DIRECTLY on the quaternion `[x, y, z, w]` and reproduces scipy
        // across all 12 sequences (extrinsic lower-case / intrinsic upper-case),
        // including the gimbal-lock singular cases. The previous matrix-formula
        // implementation did NOT match scipy for asymmetric (Tait–Bryan) sequences.
        if seq.len() != 3 {
            return [0.0, 0.0, 0.0];
        }
        let intrinsic = seq.chars().next().is_some_and(char::is_uppercase);
        let extrinsic = !intrinsic;
        let lower: Vec<char> = seq.to_lowercase().chars().collect();
        let mut axes = [
            Self::axis_index(lower[0]),
            Self::axis_index(lower[1]),
            Self::axis_index(lower[2]),
        ];
        // The algorithm assumes extrinsic; intrinsic reverses the axis order.
        if intrinsic {
            axes.reverse();
        }
        let (i, j) = (axes[0], axes[1]);
        let mut k = axes[2];
        let symmetric = i == k;
        if symmetric {
            k = 3 - i - j;
        }
        // Levi-Civita sign of the permutation (i, j, k); the product is ±2.
        let sign =
            ((i as i32 - j as i32) * (j as i32 - k as i32) * (k as i32 - i as i32) / 2) as f64;

        let q = self.quat; // [x, y, z, w]
        let w = q[3];
        let (a, b, c, d) = if symmetric {
            (w, q[i], q[j], q[k] * sign)
        } else {
            (w - q[j], q[i] + q[k] * sign, q[j] + w, q[k] * sign - q[i])
        };

        let eps = 1e-7;
        let half_sum = b.atan2(a);
        let half_diff = d.atan2(c);
        let mut angles = [0.0f64; 3];
        angles[1] = 2.0 * c.hypot(d).atan2(a.hypot(b));

        let angle_first = if extrinsic { 0 } else { 2 };
        let angle_third = if extrinsic { 2 } else { 0 };

        let case1 = angles[1].abs() <= eps;
        let case2 = (angles[1] - PI).abs() <= eps;
        let case0 = !(case1 || case2);

        // The assignment ORDER mirrors scipy's `_get_angles` exactly: `angles[0]`
        // is set first, then `angle_first`, then `angle_third` — so a singular
        // (non-case0) fallback reads whatever the array holds at that moment.
        angles[0] = if case1 {
            2.0 * half_sum
        } else {
            2.0 * half_diff * if extrinsic { -1.0 } else { 1.0 }
        };
        angles[angle_first] = if case0 {
            half_sum - half_diff
        } else {
            angles[angle_first]
        };
        let mut a3 = if case0 {
            half_sum + half_diff
        } else {
            angles[angle_third]
        };
        if !symmetric {
            a3 *= sign;
            angles[1] -= FRAC_PI_2;
        }
        angles[angle_third] = a3;

        for ang in &mut angles {
            *ang = (*ang + PI).rem_euclid(TAU) - PI;
        }
        angles
    }

    fn axis_index(c: char) -> usize {
        match c {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            _ => 0,
        }
    }

    /// Apply this rotation to a 3D vector.
    #[must_use]
    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        let m = self.as_matrix();
        let [x, y, z] = vector;
        [
            m[0][0] * x + m[0][1] * y + m[0][2] * z,
            m[1][0] * x + m[1][1] * y + m[1][2] * z,
            m[2][0] * x + m[2][1] * y + m[2][2] * z,
        ]
    }

    /// Rotate many vectors (the realistic point-cloud workload, matching
    /// `scipy.spatial.transform.Rotation.apply` on an array of vectors).
    ///
    /// Byte-identical to mapping [`apply`](Self::apply), but the quaternion→matrix
    /// conversion is done ONCE rather than for every vector.
    ///
    /// For very large clouds (`n >= PAR_THRESHOLD`) the loop fans out across cores.
    /// Single-threaded access to the AoS `[f64; 3]` layout is *latency*-bound once
    /// the working set (input + output ≈ `n·48` bytes) spills L3 — it stalls near
    /// ~3 GB/s, far below DRAM bandwidth — because one thread can't issue enough
    /// outstanding loads to hide DRAM latency. Splitting the cloud into per-core
    /// chunks supplies that memory-level parallelism and saturates aggregate
    /// bandwidth: measured 5.4× at 1.4M and 8.3× at 5M vectors on a 64-core box,
    /// flipping a 6× loss vs SciPy's vectorized `Rotation.apply` into a win. Each
    /// chunk runs the identical per-vector arithmetic in input order, so the result
    /// is byte-identical to the serial path. The threshold sits just past the L3
    /// cliff (serial is ~3 ms at 1.3M, ~16 ms at 1.4M); below it the serial loop is
    /// faster than thread-spawn overhead, so small inputs keep the plain map.
    #[must_use]
    pub fn apply_many(&self, vectors: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let m = self.as_matrix();
        const PAR_THRESHOLD: usize = 1_400_000;
        let n = vectors.len();
        if n < PAR_THRESHOLD {
            return vectors
                .iter()
                .map(|&[x, y, z]| {
                    [
                        m[0][0] * x + m[0][1] * y + m[0][2] * z,
                        m[1][0] * x + m[1][1] * y + m[1][2] * z,
                        m[2][0] * x + m[2][1] * y + m[2][2] * z,
                    ]
                })
                .collect();
        }

        let mut out = vec![[0.0f64; 3]; n];
        let cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(64);
        let chunk = n.div_ceil(cores);
        std::thread::scope(|scope| {
            for (out_chunk, vec_chunk) in out.chunks_mut(chunk).zip(vectors.chunks(chunk)) {
                scope.spawn(move || {
                    for (o, &[x, y, z]) in out_chunk.iter_mut().zip(vec_chunk.iter()) {
                        *o = [
                            m[0][0] * x + m[0][1] * y + m[0][2] * z,
                            m[1][0] * x + m[1][1] * y + m[1][2] * z,
                            m[2][0] * x + m[2][1] * y + m[2][2] * z,
                        ];
                    }
                });
            }
        });
        out
    }

    /// Return the inverse rotation.
    #[must_use]
    pub fn inv(&self) -> Self {
        let [x, y, z, w] = self.quat;
        Self {
            quat: [-x, -y, -z, w],
        }
    }

    /// Compose this rotation with another (self * other).
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let [x1, y1, z1, w1] = self.quat;
        let [x2, y2, z2, w2] = other.quat;

        Self::from_quat([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])
    }

    /// Return the rotation angle in radians.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        let [x, y, z, w] = self.quat;
        let sin_half = (x * x + y * y + z * z).sqrt();
        2.0 * sin_half.atan2(w.abs())
    }

    /// Check if this is approximately the identity rotation.
    #[must_use]
    pub fn is_identity(&self, tol: f64) -> bool {
        self.magnitude() < tol
    }

    /// Create a random rotation (uniform over SO(3)).
    #[must_use]
    pub fn random() -> Self {
        use std::f64::consts::PI;

        fn simple_rand(seed: &mut u64) -> f64 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*seed >> 33) as f64 / (1u64 << 31) as f64
        }

        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(12345, |d| d.as_nanos() as u64);

        let u1 = simple_rand(&mut seed);
        let u2 = simple_rand(&mut seed);
        let u3 = simple_rand(&mut seed);

        let sqrt_u1 = u1.sqrt();
        let sqrt_1mu1 = (1.0 - u1).sqrt();

        Self::from_quat([
            sqrt_1mu1 * (2.0 * PI * u2).sin(),
            sqrt_1mu1 * (2.0 * PI * u2).cos(),
            sqrt_u1 * (2.0 * PI * u3).sin(),
            sqrt_u1 * (2.0 * PI * u3).cos(),
        ])
    }

    /// Spherical linear interpolation between two rotations.
    ///
    /// `t` in [0, 1]: 0 returns `self`, 1 returns `other`.
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let [x1, y1, z1, w1] = self.quat;
        let [mut x2, mut y2, mut z2, mut w2] = other.quat;

        let mut dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;

        if dot < 0.0 {
            x2 = -x2;
            y2 = -y2;
            z2 = -z2;
            w2 = -w2;
            dot = -dot;
        }

        if dot > 0.9995 {
            let x = x1 + t * (x2 - x1);
            let y = y1 + t * (y2 - y1);
            let z = z1 + t * (z2 - z1);
            let w = w1 + t * (w2 - w1);
            return Self::from_quat([x, y, z, w]);
        }

        let theta_0 = dot.clamp(-1.0, 1.0).acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Self::from_quat([
            s0 * x1 + s1 * x2,
            s0 * y1 + s1 * y2,
            s0 * z1 + s1 * z2,
            s0 * w1 + s1 * w2,
        ])
    }

    /// Mean rotation of a set of rotations.
    #[must_use]
    pub fn mean(rotations: &[Self]) -> Self {
        if rotations.is_empty() {
            return Self::identity();
        }
        if rotations.len() == 1 {
            return rotations[0];
        }

        let mut sum = [[0.0; 4]; 4];
        for r in rotations {
            let q = r.quat;
            for i in 0..4 {
                for j in 0..4 {
                    sum[i][j] += q[i] * q[j];
                }
            }
        }

        let mut eigenvec = [0.0, 0.0, 0.0, 1.0];
        for _ in 0..20 {
            let mut new_vec = [0.0; 4];
            for i in 0..4 {
                for j in 0..4 {
                    new_vec[i] += sum[i][j] * eigenvec[j];
                }
            }
            let norm =
                (new_vec[0].powi(2) + new_vec[1].powi(2) + new_vec[2].powi(2) + new_vec[3].powi(2))
                    .sqrt();
            if norm > 1e-15 {
                for v in &mut new_vec {
                    *v /= norm;
                }
            }
            eigenvec = new_vec;
        }

        Self::from_quat(eigenvec)
    }
}

impl std::ops::Mul for Rotation {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl Default for Rotation {
    fn default() -> Self {
        Self::identity()
    }
}

/// A proper rigid transform in 3-D (rotation + translation), matching
/// `scipy.spatial.transform.RigidTransform`.
///
/// A transform `T` maps a point `x` to `R·x + t`, where `R` is the rotation and
/// `t` the translation. Composition `a * b` (via [`RigidTransform::compose`])
/// applies `b` first then `a`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    rotation: Rotation,
    translation: [f64; 3],
}

/// Apply a per-point transform across cores, writing into a pre-allocated output
/// in input order — byte-identical to the equivalent serial `iter().map(op)`.
/// Used by the batch transform kernels once the working set spills L3 and
/// single-threaded AoS `[f64; 3]` access turns latency-bound (see the analysis on
/// [`Rotation::apply_many`]). Generic over the closure so it monomorphizes to the
/// same tight loop as the inline serial path; the chunks are disjoint so no
/// synchronization is needed.
fn par_point_map<F>(points: &[[f64; 3]], op: F) -> Vec<[f64; 3]>
where
    F: Fn(&[f64; 3]) -> [f64; 3] + Sync,
{
    let n = points.len();
    let mut out = vec![[0.0f64; 3]; n];
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(64);
    let chunk = n.div_ceil(cores);
    let op = &op;
    std::thread::scope(|scope| {
        for (out_chunk, pts_chunk) in out.chunks_mut(chunk).zip(points.chunks(chunk)) {
            scope.spawn(move || {
                for (o, p) in out_chunk.iter_mut().zip(pts_chunk.iter()) {
                    *o = op(p);
                }
            });
        }
    });
    out
}

/// Parallel index map `0..n → Vec<T>`, byte-identical to `(0..n).map(op).collect()`
/// (chunks run in input order and are concatenated in order). Serial below `gate`
/// so small batches avoid thread-spawn overhead; used by the batch `Rotation`
/// kernels, whose per-element work is cheap but whose SciPy peer is a slow
/// Python-object-wrapped `numpy` vectorization.
fn rotation_par_index_map<T, F>(n: usize, gate: usize, op: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if n == 0 {
        return Vec::new();
    }
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(64);
    let nthreads = if n < gate || cores <= 1 {
        1
    } else {
        cores.min(n)
    };
    if nthreads <= 1 {
        return (0..n).map(&op).collect();
    }
    let chunk = n.div_ceil(nthreads);
    let op = &op;
    let parts: Vec<Vec<T>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..nthreads)
            .filter_map(|t| {
                let s0 = t * chunk;
                if s0 >= n {
                    return None;
                }
                let s1 = (s0 + chunk).min(n);
                Some(scope.spawn(move || (s0..s1).map(op).collect::<Vec<T>>()))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rotation batch worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend(p);
    }
    out
}

/// Per-element parallel gate for the batch `Rotation` kernels. The per-item work
/// (a few dozen flops on an AoS `[f64; N]`) is cheap, so keep small batches serial;
/// the working set spills L3 and the win appears well before this threshold.
const ROTATION_BATCH_GATE: usize = 1 << 15;

/// Compose `a[i] * b[i]` for every `i` (elementwise rotation composition), matching
/// `scipy.spatial.transform.Rotation.__mul__` on two equal-length rotation stacks.
///
/// Byte-identical to `a.iter().zip(b).map(|(x, y)| x.multiply(y))`, fanned across
/// cores. SciPy composes via a Python-wrapped `numpy` quaternion product whose
/// per-element overhead dominates (~0.4 µs/compose), so this is a large win at scale.
#[must_use]
pub fn rotations_multiply_many(a: &[Rotation], b: &[Rotation]) -> Vec<Rotation> {
    assert_eq!(a.len(), b.len(), "rotations_multiply_many: length mismatch");
    rotation_par_index_map(a.len(), ROTATION_BATCH_GATE, |i| a[i].multiply(&b[i]))
}

/// Convert many 3×3 rotation matrices to [`Rotation`]s, matching
/// `scipy.spatial.transform.Rotation.from_matrix` on a `(N, 3, 3)` stack.
///
/// Byte-identical to mapping [`Rotation::from_matrix`], fanned across cores.
#[must_use]
pub fn rotations_from_matrix_many(matrices: &[[[f64; 3]; 3]]) -> Vec<Rotation> {
    rotation_par_index_map(matrices.len(), ROTATION_BATCH_GATE, |i| {
        Rotation::from_matrix(matrices[i])
    })
}

/// Euler angles (sequence `seq`) for many rotations, matching
/// `scipy.spatial.transform.Rotation.as_euler` on a rotation stack.
///
/// Byte-identical to mapping [`Rotation::as_euler`], fanned across cores.
#[must_use]
pub fn rotations_as_euler_many(rotations: &[Rotation], seq: &str) -> Vec<[f64; 3]> {
    rotation_par_index_map(rotations.len(), ROTATION_BATCH_GATE, |i| {
        rotations[i].as_euler(seq)
    })
}

/// Build many rotations from Euler angles (shared sequence `seq`), matching
/// `scipy.spatial.transform.Rotation.from_euler` on an `(N, 3)` angle array.
///
/// Byte-identical to mapping [`Rotation::from_euler`], fanned across cores. SciPy
/// composes each rotation from three elementary quaternions in a Python-wrapped
/// `numpy` loop, so this is a large win at scale.
#[must_use]
pub fn rotations_from_euler_many(seq: &str, angles: &[[f64; 3]]) -> Vec<Rotation> {
    rotation_par_index_map(angles.len(), ROTATION_BATCH_GATE, |i| {
        Rotation::from_euler(seq, angles[i])
    })
}

/// Rotation vectors (axis · angle) for many rotations, matching
/// `scipy.spatial.transform.Rotation.as_rotvec` on a rotation stack.
///
/// Byte-identical to mapping [`Rotation::as_rotvec`], fanned across cores.
#[must_use]
pub fn rotations_as_rotvec_many(rotations: &[Rotation]) -> Vec<[f64; 3]> {
    rotation_par_index_map(rotations.len(), ROTATION_BATCH_GATE, |i| {
        rotations[i].as_rotvec()
    })
}

/// Spherical-linear rotation interpolator over a sequence of key rotations at
/// strictly increasing key times, matching `scipy.spatial.transform.Slerp`.
///
/// Construction precomputes, for each key interval `[t_i, t_{i+1}]`, the relative
/// rotation vector `rotvec_i = (R_i⁻¹ · R_{i+1}).as_rotvec()`. Evaluation at a query
/// time `t ∈ [t_i, t_{i+1}]` returns `R_i · from_rotvec(rotvec_i · α)` with
/// `α = (t − t_i)/(t_{i+1} − t_i)` — the exact formulation scipy uses.
pub struct Slerp {
    times: Vec<f64>,
    timedelta: Vec<f64>,
    rotations: Vec<Rotation>, // key rotations 0..n-1 (left endpoints)
    rotvecs: Vec<[f64; 3]>,   // relative rotvec per interval
}

impl Slerp {
    /// Build a `Slerp` from `n` key `times` (strictly increasing) and `n` key
    /// `rotations`. Matches `scipy.spatial.transform.Slerp(times, rotations)`.
    pub fn new(times: &[f64], rotations: &[Rotation]) -> Result<Self, SpatialError> {
        if rotations.len() <= 1 {
            return Err(SpatialError::InvalidArgument(
                "`rotations` must be a sequence of at least 2 rotations.".to_string(),
            ));
        }
        if times.len() != rotations.len() {
            return Err(SpatialError::InvalidArgument(format!(
                "Expected number of rotations to be equal to number of timestamps given, got {} rotations and {} timestamps.",
                rotations.len(),
                times.len()
            )));
        }
        let timedelta: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        if timedelta.iter().any(|&d| d <= 0.0) {
            return Err(SpatialError::InvalidArgument(
                "Times must be in strictly increasing order.".to_string(),
            ));
        }
        // rotvec_i = as_rotvec(R_i⁻¹ · R_{i+1}) — the relative rotation over interval i.
        let rotvecs: Vec<[f64; 3]> = (0..rotations.len() - 1)
            .map(|i| rotations[i].inv().multiply(&rotations[i + 1]).as_rotvec())
            .collect();
        Ok(Self {
            times: times.to_vec(),
            timedelta,
            rotations: rotations[..rotations.len() - 1].to_vec(),
            rotvecs,
        })
    }

    /// Locate the interval index for query time `t` (scipy: `searchsorted(times, t,
    /// side='left') - 1`, with `t == times[0]` mapped to interval 0), or an error if
    /// `t` is outside `[times[0], times[last]]`.
    #[inline]
    fn interval(&self, t: f64) -> Result<usize, SpatialError> {
        let pp = self.times.partition_point(|&x| x < t);
        let ind = if t == self.times[0] {
            0isize
        } else {
            pp as isize - 1
        };
        if ind < 0 || ind > self.rotations.len() as isize - 1 {
            return Err(SpatialError::InvalidArgument(format!(
                "Interpolation times must be within the range [{}, {}], both inclusive.",
                self.times[0],
                self.times[self.times.len() - 1]
            )));
        }
        Ok(ind as usize)
    }

    #[inline]
    fn interp_valid(&self, t: f64, ind: usize) -> Rotation {
        let alpha = (t - self.times[ind]) / self.timedelta[ind];
        let rv = self.rotvecs[ind];
        self.rotations[ind].multiply(&Rotation::from_rotvec([
            rv[0] * alpha,
            rv[1] * alpha,
            rv[2] * alpha,
        ]))
    }

    /// Interpolate a single rotation at query time `t`.
    pub fn eval(&self, t: f64) -> Result<Rotation, SpatialError> {
        let ind = self.interval(t)?;
        Ok(self.interp_valid(t, ind))
    }

    /// Interpolate rotations at many query times, matching `Slerp.__call__` on an
    /// array of times. Errors if ANY time is out of range (as scipy does). The valid
    /// per-time interpolations run in parallel across cores, byte-identical to the
    /// serial `ts.iter().map(|t| self.eval(t))`.
    pub fn eval_many(&self, ts: &[f64]) -> Result<Vec<Rotation>, SpatialError> {
        // Validate range up front (scipy raises on any out-of-range time).
        for &t in ts {
            self.interval(t)?;
        }
        Ok(rotation_par_index_map(ts.len(), ROTATION_BATCH_GATE, |i| {
            let t = ts[i];
            // interval() cannot fail here (validated above); recompute the index.
            let ind = self.interval(t).unwrap_or(0);
            self.interp_valid(t, ind)
        }))
    }
}

/// Compose `a[i] * b[i]` for every `i` (elementwise rigid-transform composition),
/// matching `scipy.spatial.transform.RigidTransform.__mul__` on two equal-length
/// transform stacks.
///
/// Byte-identical to `a.iter().zip(b).map(|(x, y)| x.compose(y))`, fanned across
/// cores. Like the `Rotation` stack, SciPy composes via a Python-wrapped `numpy`
/// path whose per-element overhead dominates the ~40-flop work.
#[must_use]
pub fn rigid_transforms_compose_many(
    a: &[RigidTransform],
    b: &[RigidTransform],
) -> Vec<RigidTransform> {
    assert_eq!(
        a.len(),
        b.len(),
        "rigid_transforms_compose_many: length mismatch"
    );
    rotation_par_index_map(a.len(), ROTATION_BATCH_GATE, |i| a[i].compose(&b[i]))
}

impl RigidTransform {
    /// The identity transform (no rotation, no translation).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: Rotation::identity(),
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Build from a translation vector and a rotation (`scipy from_components`).
    #[must_use]
    pub fn from_components(translation: [f64; 3], rotation: Rotation) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Build a pure rotation (zero translation).
    #[must_use]
    pub fn from_rotation(rotation: Rotation) -> Self {
        Self {
            rotation,
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Build a pure translation (identity rotation).
    #[must_use]
    pub fn from_translation(translation: [f64; 3]) -> Self {
        Self {
            rotation: Rotation::identity(),
            translation,
        }
    }

    /// Build from a 4×4 homogeneous transform matrix `[[R | t], [0 0 0 1]]`.
    #[must_use]
    pub fn from_matrix(matrix: [[f64; 4]; 4]) -> Self {
        let r = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]],
        ];
        Self {
            rotation: Rotation::from_matrix(r),
            translation: [matrix[0][3], matrix[1][3], matrix[2][3]],
        }
    }

    /// The 4×4 homogeneous transform matrix.
    #[must_use]
    pub fn as_matrix(&self) -> [[f64; 4]; 4] {
        let r = self.rotation.as_matrix();
        let t = self.translation;
        [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The `(translation, rotation)` components.
    #[must_use]
    pub fn as_components(&self) -> ([f64; 3], Rotation) {
        (self.translation, self.rotation)
    }

    /// The rotation component.
    #[must_use]
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }

    /// The translation component.
    #[must_use]
    pub fn translation(&self) -> [f64; 3] {
        self.translation
    }

    /// Apply the transform to a point. With `inverse = true`, applies the
    /// inverse transform `Rᵀ·(x − t)`.
    #[must_use]
    pub fn apply(&self, point: [f64; 3], inverse: bool) -> [f64; 3] {
        if inverse {
            let shifted = [
                point[0] - self.translation[0],
                point[1] - self.translation[1],
                point[2] - self.translation[2],
            ];
            self.rotation.inv().apply(shifted)
        } else {
            let r = self.rotation.apply(point);
            [
                r[0] + self.translation[0],
                r[1] + self.translation[1],
                r[2] + self.translation[2],
            ]
        }
    }

    /// Apply the transform to many points (the realistic point-cloud workload).
    ///
    /// Byte-identical to mapping [`apply`](Self::apply), but the rotation matrix
    /// (and, for `inverse`, the inverse rotation) is built ONCE rather than per
    /// point.
    #[must_use]
    pub fn apply_many(&self, points: &[[f64; 3]], inverse: bool) -> Vec<[f64; 3]> {
        // Same AoS-latency-bound profile as `Rotation::apply_many`: serial above the
        // L3 cliff stalls far below DRAM bandwidth, so fan out across cores once the
        // working set is large. Byte-identical to the serial map (disjoint chunks,
        // input order, identical per-point arithmetic).
        const PAR_THRESHOLD: usize = 1_400_000;
        let t = self.translation;
        let parallel = points.len() >= PAR_THRESHOLD;
        if inverse {
            let m = self.rotation.inv().as_matrix();
            let op = move |&[x, y, z]: &[f64; 3]| {
                let (sx, sy, sz) = (x - t[0], y - t[1], z - t[2]);
                [
                    m[0][0] * sx + m[0][1] * sy + m[0][2] * sz,
                    m[1][0] * sx + m[1][1] * sy + m[1][2] * sz,
                    m[2][0] * sx + m[2][1] * sy + m[2][2] * sz,
                ]
            };
            if parallel {
                par_point_map(points, op)
            } else {
                points.iter().map(op).collect()
            }
        } else {
            let m = self.rotation.as_matrix();
            let op = move |&[x, y, z]: &[f64; 3]| {
                [
                    m[0][0] * x + m[0][1] * y + m[0][2] * z + t[0],
                    m[1][0] * x + m[1][1] * y + m[1][2] * z + t[1],
                    m[2][0] * x + m[2][1] * y + m[2][2] * z + t[2],
                ]
            };
            if parallel {
                par_point_map(points, op)
            } else {
                points.iter().map(op).collect()
            }
        }
    }

    /// Compose two transforms: `self.compose(other)` applies `other` first,
    /// then `self` (equivalent to scipy's `self * other`).
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let rotation = self.rotation.multiply(&other.rotation);
        // t = R_self · t_other + t_self.
        let rt = self.rotation.apply(other.translation);
        Self {
            rotation,
            translation: [
                rt[0] + self.translation[0],
                rt[1] + self.translation[1],
                rt[2] + self.translation[2],
            ],
        }
    }

    /// The inverse transform: `R⁻¹` with translation `−R⁻¹·t`.
    #[must_use]
    pub fn inv(&self) -> Self {
        let rotation = self.rotation.inv();
        let rt = rotation.apply(self.translation);
        Self {
            rotation,
            translation: [-rt[0], -rt[1], -rt[2]],
        }
    }
}

impl Default for RigidTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for RigidTransform {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

// ── RotationSpline helpers (faithful port of scipy._rotation_spline) ──────────

fn rs_skew(x: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]]
}

fn rs_matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn rs_matvec3(a: &[[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2],
        a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2],
        a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2],
    ]
}

fn rs_norm(x: [f64; 3]) -> f64 {
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
}

fn rs_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn rs_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Matrix transforming an angular rate to the rotation-vector derivative.
fn rs_angular_rate_to_rotvec_dot(rotvec: [f64; 3]) -> [[f64; 3]; 3] {
    let norm = rs_norm(rotvec);
    let k = if norm > 1e-4 {
        (1.0 - 0.5 * norm / (0.5 * norm).tan()) / (norm * norm)
    } else {
        1.0 / 12.0 + norm * norm / 720.0
    };
    let skew = rs_skew(rotvec);
    let skew2 = rs_matmul3(&skew, &skew);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            r[i][j] = id + 0.5 * skew[i][j] + k * skew2[i][j];
        }
    }
    r
}

// Matrix transforming a rotation-vector derivative to the angular rate.
fn rs_rotvec_dot_to_angular_rate(rotvec: [f64; 3]) -> [[f64; 3]; 3] {
    let norm = rs_norm(rotvec);
    let (k1, k2) = if norm > 1e-4 {
        (
            (1.0 - norm.cos()) / (norm * norm),
            (norm - norm.sin()) / (norm * norm * norm),
        )
    } else {
        (0.5 - norm * norm / 24.0, 1.0 / 6.0 - norm * norm / 120.0)
    };
    let skew = rs_skew(rotvec);
    let skew2 = rs_matmul3(&skew, &skew);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            r[i][j] = id - k1 * skew[i][j] + k2 * skew2[i][j];
        }
    }
    r
}

// Non-linear (quadratic-in-rotvec_dot) term of the angular acceleration.
fn rs_angular_accel_nonlinear(rotvec: [f64; 3], rotvec_dot: [f64; 3]) -> [f64; 3] {
    let norm = rs_norm(rotvec);
    let dp = rs_dot(rotvec, rotvec_dot);
    let cp = rs_cross(rotvec, rotvec_dot);
    let ccp = rs_cross(rotvec, cp);
    let dccp = rs_cross(rotvec_dot, cp);
    let (k1, k2, k3) = if norm > 1e-4 {
        let n = norm;
        (
            (-n * n.sin() - 2.0 * (n.cos() - 1.0)) / n.powi(4),
            (-2.0 * n + 3.0 * n.sin() - n * n.cos()) / n.powi(5),
            (n - n.sin()) / n.powi(3),
        )
    } else {
        let n2 = norm * norm;
        (
            1.0 / 12.0 - n2 / 180.0,
            -1.0 / 60.0 + n2 / 12604.0,
            1.0 / 6.0 - n2 / 120.0,
        )
    };
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = dp * (k1 * cp[i] + k2 * ccp[i]) + k3 * dccp[i];
    }
    out
}

fn rs_compute_angular_rate(rotvec: [f64; 3], rotvec_dot: [f64; 3]) -> [f64; 3] {
    rs_matvec3(&rs_rotvec_dot_to_angular_rate(rotvec), rotvec_dot)
}

fn rs_compute_angular_accel(
    rotvec: [f64; 3],
    rotvec_dot: [f64; 3],
    rotvec_dot_dot: [f64; 3],
) -> [f64; 3] {
    let lin = rs_compute_angular_rate(rotvec, rotvec_dot_dot);
    let nl = rs_angular_accel_nonlinear(rotvec, rotvec_dot);
    [lin[0] + nl[0], lin[1] + nl[1], lin[2] + nl[2]]
}

/// Interpolate rotations with continuous angular rate and acceleration,
/// matching `scipy.spatial.transform.RotationSpline`.
///
/// The rotation vector between consecutive orientations is a cubic function of
/// time with continuous angular rate and acceleration (a rotation analogue of
/// cubic-spline interpolation). Construct with [`RotationSpline::new`] then query
/// [`RotationSpline::evaluate`] (rotation), [`RotationSpline::angular_rate`], or
/// [`RotationSpline::angular_acceleration`].
#[derive(Debug, Clone)]
pub struct RotationSpline {
    times: Vec<f64>,
    rotations: Vec<Rotation>,
    // Per-segment cubic coefficients: coeff[seg][k] are the degree-(3-k) terms.
    coeff: Vec<[[f64; 3]; 4]>,
}

impl RotationSpline {
    const MAX_ITER: usize = 10;
    const TOL: f64 = 1e-9;

    /// Build a spline through `rotations` at the strictly-increasing `times`
    /// (at least 2 of each, equal counts).
    pub fn new(times: &[f64], rotations: &[Rotation]) -> Result<Self, String> {
        let n = rotations.len();
        if n < 2 {
            return Err("`rotations` must contain at least 2 rotations.".to_string());
        }
        if times.len() != n {
            return Err("Expected number of rotations to equal number of times.".to_string());
        }
        let dt: Vec<f64> = (0..n - 1).map(|i| times[i + 1] - times[i]).collect();
        if dt.iter().any(|&d| d <= 0.0) {
            return Err("Values in `times` must be strictly increasing.".to_string());
        }

        // rotvecs[i] = (R_i^{-1} ∘ R_{i+1}) as a rotation vector.
        let rotvecs: Vec<[f64; 3]> = (0..n - 1)
            .map(|i| rotations[i].inv().multiply(&rotations[i + 1]).as_rotvec())
            .collect();
        let mut angular_rates: Vec<[f64; 3]> = (0..n - 1)
            .map(|i| {
                [
                    rotvecs[i][0] / dt[i],
                    rotvecs[i][1] / dt[i],
                    rotvecs[i][2] / dt[i],
                ]
            })
            .collect();

        let rotvecs_dot: Vec<[f64; 3]> = if n == 2 {
            angular_rates.clone()
        } else {
            let (ar, rd) = Self::solve_for_angular_rates(&dt, &mut angular_rates, &rotvecs);
            angular_rates = ar;
            rd
        };

        let mut coeff: Vec<[[f64; 3]; 4]> = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let d = dt[i];
            let mut c = [[0.0; 3]; 4];
            for k in 0..3 {
                let rv = rotvecs[i][k];
                let ar = angular_rates[i][k];
                let rd = rotvecs_dot[i][k];
                c[0][k] = (-2.0 * rv + d * ar + d * rd) / d.powi(3);
                c[1][k] = (3.0 * rv - 2.0 * d * ar - d * rd) / d.powi(2);
                c[2][k] = ar;
                c[3][k] = 0.0;
            }
            coeff.push(c);
        }

        Ok(Self {
            times: times.to_vec(),
            rotations: rotations.to_vec(),
            coeff,
        })
    }

    fn solve_for_angular_rates(
        dt: &[f64],
        angular_rates: &mut [[f64; 3]],
        rotvecs: &[[f64; 3]],
    ) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let m = rotvecs.len(); // = n - 1
        let angular_rate_first = angular_rates[0];
        let a: Vec<[[f64; 3]; 3]> = rotvecs
            .iter()
            .map(|&rv| rs_angular_rate_to_rotvec_dot(rv))
            .collect();
        let a_inv: Vec<[[f64; 3]; 3]> = rotvecs
            .iter()
            .map(|&rv| rs_rotvec_dot_to_angular_rate(rv))
            .collect();

        let nb = m - 1; // number of diagonal blocks (= n - 2)
        let sz = 3 * nb;
        // Block-tridiagonal system matrix (dense; the system is well-conditioned).
        let d: Vec<f64> = (0..nb)
            .map(|i| 4.0 * (1.0 / dt[i] + 1.0 / dt[i + 1]))
            .collect();
        let mut mat = vec![vec![0.0; sz]; sz];
        for (i, &di) in d.iter().enumerate() {
            for k in 0..3 {
                mat[3 * i + k][3 * i + k] = di;
            }
        }
        for i in 0..nb - 1 {
            let f = 2.0 / dt[1 + i];
            for r in 0..3 {
                for c in 0..3 {
                    mat[3 * (i + 1) + r][3 * i + c] = f * a_inv[1 + i][r][c]; // sub-diagonal A
                    mat[3 * i + r][3 * (i + 1) + c] = f * a[1 + i][r][c]; // super-diagonal B
                }
            }
        }

        // Constant part of the right-hand side.
        let mut b0: Vec<[f64; 3]> = (0..nb)
            .map(|i| {
                let mut v = [0.0; 3];
                for k in 0..3 {
                    v[k] = 6.0
                        * (rotvecs[i][k] / (dt[i] * dt[i])
                            + rotvecs[i + 1][k] / (dt[i + 1] * dt[i + 1]));
                }
                v
            })
            .collect();
        let corr0 = rs_matvec3(&a_inv[0], angular_rate_first);
        for k in 0..3 {
            b0[0][k] -= 2.0 / dt[0] * corr0[k];
        }
        let corr_last = rs_matvec3(&a[m - 1], angular_rates[m - 1]);
        for k in 0..3 {
            b0[nb - 1][k] -= 2.0 / dt[m - 1] * corr_last[k];
        }

        for _ in 0..Self::MAX_ITER {
            // rotvecs_dot = A · angular_rates over all m entries.
            let rotvecs_dot: Vec<[f64; 3]> = (0..m)
                .map(|i| rs_matvec3(&a[i], angular_rates[i]))
                .collect();
            let mut rhs = vec![0.0; sz];
            for i in 0..nb {
                let db = rs_angular_accel_nonlinear(rotvecs[i], rotvecs_dot[i]);
                for k in 0..3 {
                    rhs[3 * i + k] = b0[i][k] - db[k];
                }
            }
            let sol = solve_linear_system(&mat, &rhs, 1e-12)
                .expect("RotationSpline banded system is solvable");
            let mut converged = true;
            for i in 0..nb {
                let new_i = [sol[3 * i], sol[3 * i + 1], sol[3 * i + 2]];
                for k in 0..3 {
                    let delta = (new_i[k] - angular_rates[i][k]).abs();
                    if delta >= Self::TOL * (1.0 + new_i[k].abs()) {
                        converged = false;
                    }
                }
                angular_rates[i] = new_i; // angular_rates[:-1] = angular_rates_new
            }
            if converged {
                break;
            }
        }

        let rotvecs_dot: Vec<[f64; 3]> = (0..m)
            .map(|i| rs_matvec3(&a[i], angular_rates[i]))
            .collect();
        // angular_rates = vstack(angular_rate_first, angular_rates[:-1]).
        let mut final_rates = Vec::with_capacity(m);
        final_rates.push(angular_rate_first);
        for rate in angular_rates.iter().take(m - 1) {
            final_rates.push(*rate);
        }
        (final_rates, rotvecs_dot)
    }

    // Segment index and local offset for a query time (matches scipy's PPoly +
    // searchsorted(times, t, 'right') - 1, clamped to a valid segment).
    fn locate(&self, t: f64) -> (usize, f64) {
        let n_seg = self.times.len() - 1;
        let ss = self.times.partition_point(|&x| x <= t);
        let idx = ss.saturating_sub(1).min(n_seg - 1);
        (idx, t - self.times[idx])
    }

    fn eval_rotvec(&self, idx: usize, dx: f64, order: usize) -> [f64; 3] {
        let c = &self.coeff[idx];
        let mut out = [0.0; 3];
        for k in 0..3 {
            out[k] = match order {
                0 => ((c[0][k] * dx + c[1][k]) * dx + c[2][k]) * dx + c[3][k],
                1 => (3.0 * c[0][k] * dx + 2.0 * c[1][k]) * dx + c[2][k],
                _ => 6.0 * c[0][k] * dx + 2.0 * c[1][k],
            };
        }
        out
    }

    /// Interpolated rotation at time `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Rotation {
        let (idx, dx) = self.locate(t);
        let rotvec = self.eval_rotvec(idx, dx, 0);
        self.rotations[idx].multiply(&Rotation::from_rotvec(rotvec))
    }

    /// Angular rate (rad/s) at time `t`.
    #[must_use]
    pub fn angular_rate(&self, t: f64) -> [f64; 3] {
        let (idx, dx) = self.locate(t);
        let rotvec = self.eval_rotvec(idx, dx, 0);
        let rotvec_dot = self.eval_rotvec(idx, dx, 1);
        rs_compute_angular_rate(rotvec, rotvec_dot)
    }

    /// Angular acceleration (rad/s²) at time `t`.
    #[must_use]
    pub fn angular_acceleration(&self, t: f64) -> [f64; 3] {
        let (idx, dx) = self.locate(t);
        let rotvec = self.eval_rotvec(idx, dx, 0);
        let rotvec_dot = self.eval_rotvec(idx, dx, 1);
        let rotvec_dot_dot = self.eval_rotvec(idx, dx, 2);
        rs_compute_angular_accel(rotvec, rotvec_dot, rotvec_dot_dot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `query_many` must be bit-for-bit identical to calling `query` per point
    /// (same traversal + sqrt), across dims and batch sizes that span the
    /// serial/parallel gate, including non-finite/wrong-dim error propagation.
    /// `query_k_many` must be bit-for-bit identical to calling `query_k` per
    /// point (same neighbours/order/distance bits), across dims, k, and batch
    /// sizes spanning the serial/parallel gate, plus error propagation.
    /// `query_ball_point_many` must be bit-for-bit identical to calling
    /// `query_ball_point` per point (same sorted index lists), across dims,
    /// radii, and batch sizes spanning the gate, plus error/empty handling.
    /// `find_simplex_many` must be bit-for-bit identical to calling
    /// `find_simplex` per point (same simplex index, same barycentric bits),
    /// across interior / exterior / on-vertex queries and a batch that crosses
    /// the serial/parallel gate.
    /// The (now parallel) `sparse_distance_matrix_triplets` must equal a
    /// brute-force all-pairs reference, bit-for-bit, at a size that triggers the
    /// parallel collection path (the final (row,col) sort makes the result
    /// independent of thread/collection order).
    #[test]
    fn sparse_distance_matrix_triplets_matches_brute_force() {
        let n = 1500usize;
        let mk = |seed: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    vec![
                        ((t * 0.0137 + seed).sin() * 0.5 + 0.5),
                        ((t * 0.0291 + seed).cos() * 0.5 + 0.5),
                    ]
                })
                .collect()
        };
        let pa = mk(0.0);
        let pb = mk(1.3);
        let ta = KDTree::new(&pa).expect("tree a");
        let tb = KDTree::new(&pb).expect("tree b");
        let r = 0.05;
        let mut got = ta
            .sparse_distance_matrix_triplets(&tb, r)
            .expect("triplets");
        let mut brute: Vec<(usize, usize, f64)> = Vec::new();
        let r_sq = r * r;
        for (i, a) in pa.iter().enumerate() {
            for (j, b) in pb.iter().enumerate() {
                let d_sq = sqeuclidean(a, b);
                if d_sq <= r_sq {
                    brute.push((i, j, d_sq.sqrt()));
                }
            }
        }
        brute.sort_by(|l, r| l.0.cmp(&r.0).then(l.1.cmp(&r.1)));
        got.sort_by(|l, r| l.0.cmp(&r.0).then(l.1.cmp(&r.1)));
        assert_eq!(got.len(), brute.len(), "nnz mismatch");
        for (g, b) in got.iter().zip(&brute) {
            assert_eq!(g.0, b.0);
            assert_eq!(g.1, b.1);
            assert_eq!(g.2.to_bits(), b.2.to_bits(), "dist bits ({},{})", g.0, g.1);
        }
    }

    #[test]
    fn delaunay_find_simplex_many_matches_per_point() {
        // Deterministic scattered points (mirror the perf bench distribution).
        let n = 400usize;
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let t = i as f64;
                ((t * 0.137).sin() * 0.5 + 0.5, (t * 0.071).cos() * 0.5 + 0.5)
            })
            .collect();
        let tri = Delaunay::new(&pts).expect("delaunay");
        let mut queries: Vec<(f64, f64)> = (0..2000)
            .map(|i| {
                let t = i as f64;
                ((t * 0.0191).fract(), (t * 0.0233).fract())
            })
            .collect();
        // include some exact input points (on-vertex / shared-edge stress)
        queries.extend(pts.iter().copied().take(50));
        let batch = tri.find_simplex_many(&queries);
        assert_eq!(batch.len(), queries.len());
        for (q, got) in queries.iter().zip(&batch) {
            let expect = tri.find_simplex(*q);
            match (got, expect) {
                (Some((gi, g1, g2, g3)), Some((ei, e1, e2, e3))) => {
                    assert_eq!(gi, &ei, "simplex idx q={q:?}");
                    assert_eq!(g1.to_bits(), e1.to_bits(), "l1 q={q:?}");
                    assert_eq!(g2.to_bits(), e2.to_bits(), "l2 q={q:?}");
                    assert_eq!(g3.to_bits(), e3.to_bits(), "l3 q={q:?}");
                }
                (None, None) => {}
                _ => assert_eq!(
                    got.is_some(),
                    expect.is_some(),
                    "mismatch presence q={q:?}: {got:?} vs {expect:?}"
                ),
            }
        }
    }

    #[test]
    fn kdtree_query_ball_point_many_matches_per_query() {
        for &d in &[2usize, 3] {
            for &n in &[40usize, 800] {
                let data: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|c| ((i * 19 + c * 7) as f64 * 0.5).sin())
                            .collect()
                    })
                    .collect();
                let tree = KDTree::new(&data).expect("build");
                let queries: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|c| ((i * 13 + c * 3) as f64 * 0.5 + 0.2).cos())
                            .collect()
                    })
                    .collect();
                for &r in &[0.1f64, 0.5, 1.5] {
                    let batch = tree.query_ball_point_many(&queries, r).expect("batch");
                    assert_eq!(batch.len(), queries.len());
                    for (q, brow) in queries.iter().zip(&batch) {
                        let erow = tree.query_ball_point(q, r).expect("single");
                        assert_eq!(brow, &erow, "d={d} n={n} r={r}");
                    }
                }
            }
        }
        let data: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let tree = KDTree::new(&data).expect("build");
        assert!(
            tree.query_ball_point_many(&[vec![1.0, 2.0, 3.0]], 1.0)
                .is_err()
        );
        assert!(tree.query_ball_point_many(&[vec![1.0, 2.0]], -1.0).is_err());
        assert!(
            tree.query_ball_point_many(&[vec![f64::NAN, 0.0]], 1.0)
                .is_err()
        );
    }

    #[test]
    fn kdtree_query_k_many_matches_per_query() {
        for &d in &[2usize, 3, 8] {
            for &n in &[60usize, 1200] {
                let data: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|c| ((i * 29 + c * 11) as f64 * 0.021).sin())
                            .collect()
                    })
                    .collect();
                let tree = KDTree::new(&data).expect("build");
                let queries: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|c| ((i * 23 + c * 5) as f64 * 0.017 + 0.3).cos())
                            .collect()
                    })
                    .collect();
                for &k in &[1usize, 5, 12] {
                    let batch = tree.query_k_many(&queries, k).expect("query_k_many");
                    assert_eq!(batch.len(), queries.len());
                    for (q, brow) in queries.iter().zip(&batch) {
                        let erow = tree.query_k(q, k).expect("query_k");
                        assert_eq!(brow.len(), erow.len(), "len d={d} n={n} k={k}");
                        for (&(bi, bd), &(ei, ed)) in brow.iter().zip(&erow) {
                            assert_eq!(bi, ei, "idx d={d} n={n} k={k}");
                            assert_eq!(bd.to_bits(), ed.to_bits(), "dist bits d={d} n={n} k={k}");
                        }
                    }
                }
            }
        }
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let tree = KDTree::new(&data).expect("build");
        assert!(tree.query_k_many(&[vec![1.0, 2.0, 3.0]], 3).is_err());
        assert!(tree.query_k_many(&[vec![f64::NAN, 0.0]], 3).is_err());
        assert_eq!(
            tree.query_k_many(&[vec![1.0, 2.0]], 0).unwrap(),
            vec![Vec::new()]
        );
    }

    #[test]
    fn kdtree_query_many_matches_per_query() {
        for &d in &[2usize, 3, 8] {
            for &n in &[50usize, 1500] {
                let data: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|k| ((i * 31 + k * 7) as f64 * 0.019).sin())
                            .collect()
                    })
                    .collect();
                let tree = KDTree::new(&data).expect("build");
                // batch crosses the 512 parallel gate at n=1500
                let queries: Vec<Vec<f64>> = (0..n)
                    .map(|i| {
                        (0..d)
                            .map(|k| ((i * 17 + k * 13) as f64 * 0.023 + 0.5).cos())
                            .collect()
                    })
                    .collect();
                let batch = tree.query_many(&queries).expect("query_many");
                assert_eq!(batch.len(), queries.len());
                for (q, &(bi, bd)) in queries.iter().zip(&batch) {
                    let (ei, ed) = tree.query(q).expect("query");
                    assert_eq!(bi, ei, "idx mismatch d={d} n={n}");
                    assert_eq!(bd.to_bits(), ed.to_bits(), "dist bits mismatch d={d} n={n}");
                }
            }
        }
        // Error propagation: wrong-dim and non-finite queries.
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let tree = KDTree::new(&data).expect("build");
        assert!(tree.query_many(&[vec![1.0, 2.0, 3.0]]).is_err());
        assert!(tree.query_many(&[vec![f64::NAN, 0.0]]).is_err());
    }

    #[test]
    fn rotation_spline_matches_scipy() {
        let quats = [
            [0.0, 0.0, 0.0, 1.0],
            [
                0.022260026714733816,
                0.43967973954090955,
                0.3604234056503559,
                0.8223631719059994,
            ],
            [
                0.0,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            [0.5, 0.5, 0.5, 0.5],
        ];
        let rots: Vec<Rotation> = quats.iter().map(|&q| Rotation::from_quat(q)).collect();
        let times = [0.0, 1.0, 2.5, 4.0];
        let sp = RotationSpline::new(&times, &rots).unwrap();

        let close3 = |a: [f64; 3], b: [f64; 3], msg: &str| {
            for k in 0..3 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-8,
                    "{msg}[{k}]: {} vs {}",
                    a[k],
                    b[k]
                );
            }
        };
        let close_quat = |q: [f64; 4], w: [f64; 4], msg: &str| {
            // Quaternion sign is arbitrary; compare up to global sign.
            let same = (0..4).all(|i| (q[i] - w[i]).abs() < 1e-8);
            let neg = (0..4).all(|i| (q[i] + w[i]).abs() < 1e-8);
            assert!(same || neg, "{msg}: {q:?} vs {w:?}");
        };

        // scipy oracle.
        close_quat(
            sp.evaluate(0.5).as_quat(),
            [0.0190532683, 0.2724705434, 0.1865647354, 0.9437109597],
            "q@0.5",
        );
        close3(
            sp.angular_rate(0.5),
            [0.0776760538, 1.1108034724, 0.7605840739],
            "rate@0.5",
        );
        close3(
            sp.angular_acceleration(0.5),
            [-0.2497892938, -0.6616823691, -0.0200775903],
            "acc@0.5",
        );
        close_quat(
            sp.evaluate(1.7).as_quat(),
            [-0.028057948, 0.2710122459, 0.5893995291, 0.7605085859],
            "q@1.7",
        );
        close3(
            sp.angular_rate(1.7),
            [-0.7664033976, -0.580715646, 0.3985478359],
            "rate@1.7",
        );
        close3(
            sp.angular_acceleration(3.0),
            [1.1515799765, 0.3738062021, -0.3029762586],
            "acc@3.0",
        );
        close_quat(
            sp.evaluate(3.0).as_quat(),
            [0.1429451934, 0.0900362009, 0.7085729963, 0.6851163865],
            "q@3.0",
        );
        // At a knot, returns the input rotation.
        close_quat(sp.evaluate(2.5).as_quat(), quats[2], "knot@2.5");
    }

    #[test]
    fn rigid_transform_apply_many_matches_apply() {
        // apply_many (matrix precomputed once) must be byte-identical to per-point apply.
        let r = Rotation::from_quat([
            0.022260026714733816,
            0.43967973954090955,
            0.3604234056503559,
            0.8223631719059994,
        ]);
        let tf = RigidTransform::from_components([1.0, 2.0, 3.0], r);
        let pts = [[0.5, -1.0, 2.0], [1.0, 0.0, 0.0], [-3.0, 2.5, -0.25]];
        for inverse in [false, true] {
            let batch = tf.apply_many(&pts, inverse);
            for (p, b) in pts.iter().zip(batch.iter()) {
                assert_eq!(
                    *b,
                    tf.apply(*p, inverse),
                    "apply_many({inverse}) != apply at {p:?}"
                );
            }
        }

        // Large input crosses the parallel threshold (1.4M); the multi-threaded
        // chunked path must stay byte-identical to per-point apply (both branches).
        let n = 1_500_000usize;
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15 - 0.5
        };
        let big: Vec<[f64; 3]> = (0..n).map(|_| [next(), next(), next()]).collect();
        for inverse in [false, true] {
            let big_batch = tf.apply_many(&big, inverse);
            assert_eq!(big_batch.len(), n);
            for &i in &[0usize, 1, 999_999, 1_400_001, n - 1] {
                assert_eq!(
                    big_batch[i],
                    tf.apply(big[i], inverse),
                    "parallel apply_many({inverse}) != apply at index {i}"
                );
            }
        }
    }

    #[test]
    fn rigid_transform_matches_scipy() {
        let close = |a: [f64; 3], b: [f64; 3], msg: &str| {
            for k in 0..3 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-9,
                    "{msg}[{k}]: {} vs {}",
                    a[k],
                    b[k]
                );
            }
        };
        let r = Rotation::from_quat([
            0.022260026714733816,
            0.43967973954090955,
            0.3604234056503559,
            0.8223631719059994,
        ]);
        let tf = RigidTransform::from_components([1.0, 2.0, 3.0], r);
        let pt = [0.5, -1.0, 2.0];
        close(
            tf.apply(pt, false),
            [3.2283978394802326, 2.1276474698876013, 3.5176380902050415],
            "apply",
        );
        close(
            tf.apply(pt, true),
            [-1.3067872211974727, -2.284538497461942, -1.8229621532355842],
            "apply_inv",
        );
        close(
            tf.inv().translation(),
            [0.5430220815747795, -1.965834706556691, -3.1369763986073207],
            "inv_t",
        );

        // 4x4 matrix round-trip.
        let m = tf.as_matrix();
        let tf_rt = RigidTransform::from_matrix(m);
        close(tf_rt.translation(), tf.translation(), "matrix_rt_t");
        close(
            tf_rt.apply(pt, false),
            tf.apply(pt, false),
            "matrix_rt_apply",
        );
        assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);

        // Composition: tf * tf2.
        let r2 = Rotation::from_quat([
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ]);
        let tf2 = RigidTransform::from_components([0.0, 0.0, 1.0], r2);
        let comp = tf * tf2;
        close(
            comp.apply(pt, false),
            [3.2845384974619423, 3.822962153235584, 4.306787221197473],
            "compose_apply",
        );
        close(
            comp.translation(),
            [1.7391989197401165, 2.2803300858899105, 3.6123724356957942],
            "compose_t",
        );

        // inv composed with self is identity (apply ~ identity).
        let back = tf.inv() * tf;
        close(back.apply(pt, false), pt, "inv_roundtrip");
    }

    #[test]
    fn dice_sokalsneath_all_false_match_scipy_nan() {
        // Regression (frankenscipy-wwsdx): the all-false 0/0 case was guarded to
        // 0.0, an undocumented divergence. scipy.spatial.distance.dice returns NaN
        // there and sokalsneath raises "not defined for vectors that are entirely
        // false" — so NaN is the f64 "undefined" signal for both. Normal inputs are
        // unchanged.
        let af = [false, false, false, false];
        assert!(
            dice(&af, &af).is_nan(),
            "dice(all-false) must be NaN like scipy"
        );
        assert!(
            sokalsneath(&af, &af).is_nan(),
            "sokalsneath(all-false) must be NaN (scipy raises)"
        );

        // Non-degenerate inputs still match scipy.spatial.distance 1.17.1 exactly.
        let u = [true, false, true, true, false, true, false, false];
        let v = [true, true, false, true, false, false, true, false];
        assert!((dice(&u, &v) - 0.5).abs() < 1e-12);
        assert!((sokalsneath(&u, &v) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn convex_hull_2d_match_scipy() {
        // Unit square + interior point. scipy: area(=perimeter)=4, volume(=2D
        // area)=1, 4 hull vertices (interior point excluded). fsci swaps the
        // area/perimeter names (documented): area=2D area, perimeter=scipy area.
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let h = ConvexHull::new(&pts).expect("hull");
        assert_eq!(h.vertices.len(), 4, "interior point excluded");
        assert!(
            (h.area - 1.0).abs() < 1e-12,
            "area (scipy volume): {}",
            h.area
        );
        assert!(
            (h.perimeter - 4.0).abs() < 1e-12,
            "perimeter (scipy area): {}",
            h.perimeter
        );
    }

    #[test]
    fn kdtree_query_match_scipy() {
        // scipy.spatial.KDTree query / query(k=2) / query_ball_point.
        let pts = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![0.0, 3.0],
        ];
        let tree = KDTree::new(&pts).expect("kdtree");
        let (idx, dist) = tree.query(&[0.6, 0.6]).expect("query");
        assert_eq!(idx, 1);
        assert!((dist - 0.565_685_424_949_238).abs() < 1e-12, "dist: {dist}");
        let knn = tree.query_k(&[0.6, 0.6], 2).expect("query_k");
        assert_eq!(knn[0].0, 1);
        assert_eq!(knn[1].0, 0);
        assert!((knn[0].1 - 0.565_685_424_949_238).abs() < 1e-12);
        assert!((knn[1].1 - 0.848_528_137_423_857).abs() < 1e-12);
        let mut ball = tree.query_ball_point(&[0.0, 0.0], 1.5).expect("ball");
        ball.sort();
        assert_eq!(ball, vec![0, 1]);
    }

    #[test]
    fn coordinate_transforms_roundtrip_and_analytic() {
        use std::f64::consts::FRAC_PI_2;
        // Spherical: (r, theta=polar angle from +z, phi=azimuth). All four
        // transforms were previously untested.
        let (r, theta, phi) = cartesian_to_spherical(1.0, 0.0, 0.0);
        assert!(
            (r - 1.0).abs() < 1e-12 && (theta - FRAC_PI_2).abs() < 1e-12 && phi.abs() < 1e-12,
            "c2s (1,0,0)"
        );
        // Round-trip spherical.
        let (x, y, z) = spherical_to_cartesian(2.0, 1.0, 0.5);
        let (r2, t2, p2) = cartesian_to_spherical(x, y, z);
        assert!(
            (r2 - 2.0).abs() < 1e-12 && (t2 - 1.0).abs() < 1e-12 && (p2 - 0.5).abs() < 1e-12,
            "spherical roundtrip"
        );
        // Cylindrical direct + round-trip.
        let (rho, th, zc) = cartesian_to_cylindrical(3.0, 4.0, 5.0);
        assert!(
            (rho - 5.0).abs() < 1e-12 && (zc - 5.0).abs() < 1e-12,
            "c2cyl (3,4,5)"
        );
        let (xx, yy, zz) = cylindrical_to_cartesian(rho, th, zc);
        assert!(
            (xx - 3.0).abs() < 1e-12 && (yy - 4.0).abs() < 1e-12 && (zz - 5.0).abs() < 1e-12,
            "cyl roundtrip"
        );
    }

    #[test]
    fn spatial_vector_helpers_match_analytic() {
        // Previously-untested spatial vector helpers vs analytic identities.
        assert!(
            (dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-12,
            "dot"
        );
        let u = normalize(&[3.0, 4.0]);
        assert!(
            (u[0] - 0.6).abs() < 1e-12 && (u[1] - 0.8).abs() < 1e-12,
            "normalize unit"
        );
        // angle between perpendicular vectors = pi/2.
        assert!(
            (angle_between(&[1.0, 0.0], &[0.0, 1.0]) - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "angle"
        );
        // matching = boolean Hamming = fraction differing; 2 of 4 differ. (scipy 1.17
        // removed matching as a deprecated hamming alias, so assert the identity.)
        let m = matching(&[true, false, true, true], &[true, true, false, true]);
        assert!((m - 0.5).abs() < 1e-12, "matching");
    }

    #[test]
    fn mahalanobis_match_scipy() {
        // scipy.spatial.distance.mahalanobis(u, v, VI): sqrt((u-v)^T VI (u-v)).
        let vi = vec![vec![2.0, 0.5], vec![0.5, 2.0]];
        let d = mahalanobis(&[1.0, 0.0], &[0.0, 1.0], &vi);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-12, "mahalanobis: {d}");
    }

    #[test]
    fn pdist_extra_metrics_match_scipy() {
        // scipy.spatial.distance.pdist cosine/canberra/braycurtis/correlation.
        let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 6.0, 8.0]];
        assert!(
            (pdist(&x, DistanceMetric::Cosine).unwrap()[0] - 0.007_416_666_029_069_652).abs()
                < 1e-12,
            "cosine"
        );
        assert!(
            (pdist(&x, DistanceMetric::Canberra).unwrap()[0] - 1.554_545_454_545_454_7).abs()
                < 1e-12,
            "canberra"
        );
        assert!(
            (pdist(&x, DistanceMetric::Braycurtis).unwrap()[0] - 0.5).abs() < 1e-12,
            "braycurtis"
        );
        assert!(
            pdist(&x, DistanceMetric::Correlation).unwrap()[0].abs() < 1e-12,
            "correlation ~0"
        );
    }

    #[test]
    fn cdist_pdist_match_scipy() {
        // scipy.spatial.distance.cdist (euclidean) and pdist (condensed), 1.17.1.
        let a = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let c = cdist(&a, &b).unwrap();
        let ec = [[1.0, 2.0], [1.0, std::f64::consts::SQRT_2]];
        for (gr, er) in c.iter().zip(&ec) {
            for (g, e) in gr.iter().zip(er) {
                assert!((g - e).abs() < 1e-12, "cdist: {g} vs {e}");
            }
        }
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let pe = pdist(&x, DistanceMetric::Euclidean).unwrap();
        for (g, e) in pe
            .iter()
            .zip(&[std::f64::consts::SQRT_2, 2.0, std::f64::consts::SQRT_2])
        {
            assert!((g - e).abs() < 1e-12, "pdist euclidean: {g} vs {e}");
        }
        assert_eq!(
            pdist(&x, DistanceMetric::Cityblock).unwrap(),
            vec![2.0, 2.0, 2.0]
        );
        assert_eq!(
            pdist(&x, DistanceMetric::Chebyshev).unwrap(),
            vec![1.0, 2.0, 1.0]
        );
    }

    // The test above uses 2-D points with integer coordinates, so every distance
    // lands on a round number and only the generic scalar path runs. But `pdist`
    // and `cdist` dispatch to entirely separate SIMD kernels when dim == 4 --
    // pdist_fill_euclidean4, fill_cityblock4_rows, fill_chebyshev4_rows,
    // fill_sqeuclidean4_rows, pdist_fill_cosine4 -- selected by an explicit
    // `if dim == 4` arm. None of those kernels is reached by any scipy-value
    // test.
    //
    // They are covered today only by pdist_dim4_fast_paths_match_metric_helpers,
    // cdist_dim4_fast_paths_match_metric_distance and friends, which compare the
    // fast path against fsci's OWN scalar helpers. That is self-consistency: if
    // the scalar helper and the SIMD kernel share a mistake, or the scalar
    // reference is itself wrong, every one of those tests still passes. This
    // pins the dim-4 kernels to scipy instead.
    //
    // Coordinates are deliberately non-round so no distance can come out right
    // by arithmetic coincidence.
    #[test]
    fn pdist_cdist_dim4_simd_kernels_match_scipy() {
        let x = vec![
            vec![1.5, -2.25, 0.75, 3.0],
            vec![-0.5, 4.0, 2.25, -1.125],
            vec![3.25, 0.5, -1.75, 2.5],
            vec![0.125, -3.5, 4.75, 1.0],
        ];
        assert_eq!(x[0].len(), 4, "this test exists for the dim == 4 kernels");

        // scipy.spatial.distance.pdist(X, metric=...) -- condensed order
        // (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
        let cases: [(DistanceMetric, [f64; 6], &str); 3] = [
            (
                DistanceMetric::Euclidean,
                [
                    7.894_816_337_319_064,
                    4.138_236_339_311_712,
                    4.842_842_656_952_629,
                    7.446_685_504_303_24,
                    8.210_130_936_836_514,
                    8.382_459_364_649_494,
                ],
                "euclidean",
            ),
            (
                DistanceMetric::Cityblock,
                [13.875, 7.5, 8.625, 14.875, 12.75, 15.125],
                "cityblock",
            ),
            (
                DistanceMetric::Chebyshev,
                [6.25, 2.75, 4.0, 4.0, 7.5, 6.5],
                "chebyshev",
            ),
        ];
        for (metric, expected, label) in cases {
            let got = pdist(&x, metric).unwrap_or_else(|e| panic!("pdist {label}: {e:?}"));
            assert_eq!(got.len(), 6, "pdist {label} condensed length");
            for (k, (g, e)) in got.iter().zip(&expected).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "pdist dim4 {label} [{k}]: {g} vs scipy {e}"
                );
            }
        }

        // scipy.spatial.distance.cdist(X[:2], X[2:], metric='euclidean')
        let a = vec![x[0].clone(), x[1].clone()];
        let b = vec![x[2].clone(), x[3].clone()];
        let c = cdist(&a, &b).expect("cdist dim4");
        let ec = [
            [4.138_236_339_311_712, 4.842_842_656_952_629],
            [7.446_685_504_303_24, 8.210_130_936_836_514],
        ];
        for (i, (gr, er)) in c.iter().zip(&ec).enumerate() {
            for (j, (g, e)) in gr.iter().zip(er).enumerate() {
                assert!(
                    (g - e).abs() < 1e-12,
                    "cdist dim4 euclidean [{i}][{j}]: {g} vs scipy {e}"
                );
            }
        }
    }

    #[test]
    fn boolean_metrics_match_scipy_1_17() {
        // Golden values from scipy.spatial.distance (1.17.1) for
        // u=[T,F,T,F], v=[T,T,F,F]: contingency ntt=ntf=nft=nff=1.
        let u = [true, false, true, false];
        let v = [true, true, false, false];
        assert!((yule(&u, &v) - 1.0).abs() < 1e-12, "yule");
        assert!((dice(&u, &v) - 0.5).abs() < 1e-12, "dice");
        assert!(
            (rogerstanimoto(&u, &v) - 0.666_666_666_666_666_6).abs() < 1e-12,
            "rogerstanimoto"
        );
        assert!((russellrao(&u, &v) - 0.75).abs() < 1e-12, "russellrao");
        assert!((sokalsneath(&u, &v) - 0.8).abs() < 1e-12, "sokalsneath");
        // sokalmichener == rogerstanimoto (removed from SciPy but kept here).
        assert_eq!(sokalmichener(&u, &v), rogerstanimoto(&u, &v));
        // yule's all-equal degenerate case is 0.0 (half_R == 0).
        assert_eq!(yule(&[true, true], &[true, true]), 0.0);
    }

    #[test]
    fn float_metrics_match_scipy_1_17() {
        // Golden values from scipy.spatial.distance (1.17.1) for
        // a=[1,2,3], b=[4,0,-1].
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, -1.0];
        let close = |got: f64, want: f64, name: &str| {
            assert!((got - want).abs() < 1e-12, "{name}: {got} != {want}");
        };
        close(cityblock(&a, &b), 9.0, "cityblock");
        close(chebyshev(&a, &b), 4.0, "chebyshev");
        close(minkowski(&a, &b, 2.0), 5.385_164_807_134_504, "euclidean");
        close(
            minkowski(&a, &b, 3.0),
            4.626_065_009_182_741,
            "minkowski p3",
        );
        close(cosine(&a, &b), 0.935_179_627_644_783_6, "cosine");
        close(correlation(&a, &b), 1.944_911_182_523_068, "correlation");
        close(canberra(&a, &b), 2.6, "canberra");
        close(braycurtis(&a, &b), 1.0, "braycurtis");
        // Jensen-Shannon over probability vectors (base=None ⇒ natural log).
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        close(
            jensenshannon(&p, &q, None),
            0.180_359_656_027_125_61,
            "jensenshannon",
        );
    }

    /// The multithreaded `pdist` must be BIT-IDENTICAL to the sequential condensed
    /// i<j push order; pair-balanced row boundaries must tile the output exactly.
    #[test]
    fn pdist_parallel_is_bit_identical() {
        let grid = |n: usize, dim: usize, seed: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|j| (i as f64 * 0.017 + j as f64 * 0.09 + seed).sin() - 0.1)
                        .collect()
                })
                .collect()
        };
        for &metric in &[
            DistanceMetric::Euclidean,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
        ] {
            // n=1100 (dim 2) => ~604k pairs * 2 = 1.21M >= the 2^20 pdist gate -> parallel.
            let x = grid(1100, 2, 0.5);
            let n = x.len();
            let got = pdist(&x, metric).expect("parallel pdist");
            let mut want = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    want.push(metric_distance(&x[i], &x[j], metric));
                }
            }
            assert_eq!(got.len(), want.len());
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "pdist mismatch at {k} {metric:?}");
            }
        }
    }

    /// The vectorized dim-4 Euclidean/Cosine fast paths must stay BIT-identical to the
    /// scalar metric helper above the serial gate (n>2048), where rows are split across
    /// worker threads — exercising the `r0>0` mid-triangle offset of the SIMD row-filler
    /// that the serial path does not reach.
    #[test]
    fn pdist_dim4_parallel_matches_metric_helpers() {
        let n = 2100;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64;
                vec![
                    (t * 0.11).sin(),
                    (t * 0.07).cos(),
                    t * 0.003,
                    (t * 0.17).sin() - 0.25,
                ]
            })
            .collect();
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::SqEuclidean,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
        ] {
            let got = pdist(&x, metric).expect("pdist dim4 parallel fast path");
            let mut want = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    want.push(metric_distance(&x[i], &x[j], metric));
                }
            }
            assert_eq!(got.len(), want.len());
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "dim4 parallel pdist mismatch at {k} {metric:?}"
                );
            }
        }
    }

    #[test]
    fn pdist_dim4_fast_paths_match_metric_helpers() {
        let x: Vec<Vec<f64>> = (0..32)
            .map(|i| {
                let t = i as f64;
                vec![
                    (t * 0.11).sin(),
                    (t * 0.07).cos(),
                    t * 0.003,
                    (t * 0.17).sin() - 0.25,
                ]
            })
            .collect();
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::SqEuclidean,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
        ] {
            let got = pdist(&x, metric).expect("pdist dim4 fast path");
            let mut want = Vec::with_capacity(x.len() * (x.len() - 1) / 2);
            for i in 0..x.len() {
                for j in (i + 1)..x.len() {
                    want.push(metric_distance(&x[i], &x[j], metric));
                }
            }
            assert_eq!(got.len(), want.len());
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "dim4 pdist mismatch at {k} {metric:?}"
                );
            }
        }
    }

    #[test]
    fn pdist_dim4_chebyshev_fast_path_preserves_nan_fold() {
        let mut x: Vec<Vec<f64>> = (0..32)
            .map(|i| {
                let t = i as f64;
                vec![
                    (t * 0.11).sin(),
                    (t * 0.07).cos(),
                    t * 0.003,
                    (t * 0.17).sin() - 0.25,
                ]
            })
            .collect();
        x[3][2] = f64::NAN;

        let got = pdist(&x, DistanceMetric::Chebyshev).expect("pdist dim4 chebyshev");
        let mut want = Vec::with_capacity(x.len() * (x.len() - 1) / 2);
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                want.push(metric_distance(&x[i], &x[j], DistanceMetric::Chebyshev));
            }
        }

        for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "nan fold mismatch at {k}");
        }
    }

    #[test]
    fn pdist_wide_chebyshev_matches_scalar_nan_fold() {
        fn scalar_chebyshev_ref(a: &[f64], b: &[f64]) -> f64 {
            a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).fold(
                0.0_f64,
                |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                },
            )
        }

        for &dim in &[16usize, 64] {
            let mut x: Vec<Vec<f64>> = (0..40)
                .map(|i| {
                    (0..dim)
                        .map(|k| ((i * 37 + k * 17) as f64 * 0.013).sin() - 0.25)
                        .collect()
                })
                .collect();
            x[7][dim / 2] = f64::NAN;

            let got = pdist(&x, DistanceMetric::Chebyshev).expect("wide chebyshev pdist");
            let mut want = Vec::with_capacity(x.len() * (x.len() - 1) / 2);
            for i in 0..x.len() {
                for j in (i + 1)..x.len() {
                    want.push(scalar_chebyshev_ref(&x[i], &x[j]));
                }
            }

            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "wide chebyshev mismatch at dim {dim}, pair {k}"
                );
            }
        }
    }

    /// The multithreaded `cdist_metric` must be BIT-IDENTICAL to the sequential
    /// row-by-row computation: each output row is an independent reduction over the
    /// same pure `metric_distance`, so only the owning thread changes. Uses a size
    /// above the parallel gate so the threaded path actually runs.
    #[test]
    fn cdist_metric_parallel_is_bit_identical() {
        let grid = |n: usize, dim: usize, seed: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|j| (i as f64 * 0.013 + j as f64 * 0.07 + seed).sin() + 0.5)
                        .collect()
                })
                .collect()
        };
        for &metric in &[
            DistanceMetric::Euclidean,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
        ] {
            // na=600, nb=600, dim=2 => work 720k >= the 2^18 gate -> parallel path.
            let xa = grid(600, 2, 0.3);
            let xb = grid(600, 2, 1.1);
            let got = cdist_metric(&xa, &xb, metric).expect("parallel cdist");
            let want: Vec<Vec<f64>> = xa
                .iter()
                .map(|a| xb.iter().map(|b| metric_distance(a, b, metric)).collect())
                .collect();
            assert_eq!(got.len(), want.len());
            for (i, (gr, wr)) in got.iter().zip(&want).enumerate() {
                for (j, (&g, &w)) in gr.iter().zip(wr).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "cdist mismatch at ({i},{j}) {metric:?}"
                    );
                }
            }
        }
    }

    /// The dim-4 Euclidean/Cosine cdist SoA fast paths (SIMD across xb columns) must stay
    /// BIT-identical to the generic per-pair `metric_distance` arm, at BOTH a serial size
    /// and a size above the parallel gate (where xa rows split across workers). Includes a
    /// zero-norm xb row so the Cosine denom==0⇒NaN select is exercised.
    #[test]
    fn cdist_dim4_fast_paths_match_metric_distance() {
        let mk = |n: usize, seed: f64, zero_row: bool| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    if zero_row && i == n / 2 {
                        return vec![0.0, 0.0, 0.0, 0.0];
                    }
                    let t = i as f64;
                    vec![
                        (t * 0.11 + seed).sin(),
                        (t * 0.07 + seed).cos(),
                        t * 0.003 - 0.2,
                        (t * 0.17 + seed).sin() - 0.25,
                    ]
                })
                .collect()
        };
        // (na, nb): (40,40) stays serial; (600,600) exceeds the 2^18 work gate -> parallel.
        for &(na, nb) in &[(40usize, 40usize), (600, 600)] {
            let xa = mk(na, 0.3, false);
            let xb = mk(nb, 1.1, true);
            for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
                let got = cdist_metric(&xa, &xb, metric).expect("dim4 cdist");
                let want: Vec<Vec<f64>> = xa
                    .iter()
                    .map(|a| xb.iter().map(|b| metric_distance(a, b, metric)).collect())
                    .collect();
                for (i, (gr, wr)) in got.iter().zip(&want).enumerate() {
                    for (j, (&g, &w)) in gr.iter().zip(wr).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "dim4 cdist mismatch at ({i},{j}) {metric:?} na={na} nb={nb}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cdist_euclidean_small_d_soa_matches_scalar_bitwise() {
        // The SoA-across-pairs Euclidean kernel (dim<8) must be BIT-identical to
        // the scalar per-pair `euclidean` for every small dimension — verifying
        // the lane left-fold reduction matches `sqeuclidean`'s scalar left-fold.
        let mut s: u64 = 0xABCD_1234_5678_9001;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for d in [1usize, 2, 3, 5, 6, 7] {
            // nb=41 = 5·8 + 1 exercises both the SIMD-chunk and the scalar tail;
            // (40,41) stays serial, (600,300) crosses the work gate -> parallel.
            for &(na, nb) in &[(40usize, 41usize), (600, 300)] {
                let xa: Vec<Vec<f64>> = (0..na)
                    .map(|_| (0..d).map(|_| rng() * 4.0 - 2.0).collect())
                    .collect();
                let xb: Vec<Vec<f64>> = (0..nb)
                    .map(|_| (0..d).map(|_| rng() * 4.0 - 2.0).collect())
                    .collect();
                for metric in [
                    DistanceMetric::Euclidean,
                    DistanceMetric::SqEuclidean,
                    DistanceMetric::Cityblock,
                    DistanceMetric::Chebyshev,
                    DistanceMetric::Cosine,
                    DistanceMetric::Canberra,
                    DistanceMetric::Braycurtis,
                    DistanceMetric::Correlation,
                    DistanceMetric::Hamming,
                    DistanceMetric::Jaccard,
                ] {
                    let got = cdist_metric(&xa, &xb, metric).expect("cdist");
                    for (i, gr) in got.iter().enumerate() {
                        for (j, &g) in gr.iter().enumerate() {
                            let want = metric_distance(&xa[i], &xb[j], metric);
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "soa cdist mismatch at ({i},{j}) {metric:?} d={d} na={na} nb={nb}"
                            );
                        }
                    }
                }
            }
        }
        // NaN-propagation: a NaN coordinate must make every distance involving it
        // NaN (matching the scalar helpers), for both the SIMD-chunk lanes and the
        // scalar tail. Chebyshev is the sharp case (its running max would drop a
        // NaN without the explicit mask).
        for d in [2usize, 3, 5, 7] {
            let mut xa: Vec<Vec<f64>> = (0..10).map(|_| (0..d).map(|_| rng()).collect()).collect();
            xa[4][d - 1] = f64::NAN;
            xa[3] = vec![0.0; d]; // zero norm -> cosine denom==0 -> NaN path
            let xb: Vec<Vec<f64>> = (0..20).map(|_| (0..d).map(|_| rng()).collect()).collect();
            for metric in [
                DistanceMetric::Euclidean,
                DistanceMetric::SqEuclidean,
                DistanceMetric::Cityblock,
                DistanceMetric::Chebyshev,
                DistanceMetric::Cosine,
            ] {
                let got = cdist_metric(&xa, &xb, metric).expect("cdist nan");
                for (i, gr) in got.iter().enumerate() {
                    for (j, &g) in gr.iter().enumerate() {
                        let want = metric_distance(&xa[i], &xb[j], metric);
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "nan cdist mismatch at ({i},{j}) {metric:?} d={d}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cdist_boolean_soa_matches_scalar_bitwise() {
        // Hamming/Jaccard SoA-across-pairs counts must be BIT-identical to the scalar
        // helpers on real binary data (where ties/zeros actually exercise the count
        // logic) across low AND high d — the integer counters are exact regardless of
        // dimension. nb=43 hits both the 8-lane chunk and the scalar tail.
        let mut s: u64 = 0x2468_ACE0_1357_9BDF;
        let mut bit = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) & 1) as f64
        };
        for &d in &[1usize, 3, 7, 8, 64, 200] {
            for &(na, nb) in &[(40usize, 43usize), (500, 300)] {
                let xa: Vec<Vec<f64>> = (0..na).map(|_| (0..d).map(|_| bit()).collect()).collect();
                let xb: Vec<Vec<f64>> = (0..nb).map(|_| (0..d).map(|_| bit()).collect()).collect();
                for metric in [DistanceMetric::Hamming, DistanceMetric::Jaccard] {
                    let got = cdist_metric(&xa, &xb, metric).expect("cdist");
                    for (i, gr) in got.iter().enumerate() {
                        for (j, &g) in gr.iter().enumerate() {
                            let want = metric_distance(&xa[i], &xb[j], metric);
                            assert_eq!(
                                g.to_bits(),
                                want.to_bits(),
                                "soa boolean cdist mismatch at ({i},{j}) {metric:?} d={d} na={na} nb={nb}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Isomorphism proof for the procrustes k-hoist [frankenscipy-146ld]: the
    /// cache-friendly k-outermost cross-covariance/matmul loops must be
    /// BIT-IDENTICAL to the naive column-strided versions (same per-element k
    /// accumulation order).
    #[test]
    fn procrustes_khoist_is_bit_identical() {
        let mk = |rows: usize, cols: usize, seed: u64| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let r = (seed
                                .wrapping_mul(i as u64 + 1)
                                .wrapping_add(j as u64 * 7 + 3)
                                % 1999) as f64
                                / 997.0;
                            r - 1.0
                        })
                        .collect()
                })
                .collect()
        };
        let (n, d) = (13usize, 4usize);
        let a = mk(n, d, 1);
        let b = mk(n, d, 2);
        // Naive M = a^T @ b (column-strided).
        let mut m_naive = vec![vec![0.0f64; d]; d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..n {
                    m_naive[i][j] += a[k][i] * b[k][j];
                }
            }
        }
        // k-hoisted M (mirrors the procrustes implementation).
        let mut m_hoist = vec![vec![0.0f64; d]; d];
        for k in 0..n {
            let (r2, r1) = (&a[k], &b[k]);
            for i in 0..d {
                let v = r2[i];
                let mi = &mut m_hoist[i];
                for j in 0..d {
                    mi[j] += v * r1[j];
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                assert_eq!(
                    m_hoist[i][j].to_bits(),
                    m_naive[i][j].to_bits(),
                    "procrustes M k-hoist not bit-identical at ({i},{j})"
                );
            }
        }
    }

    /// Wall-clock witness for the procrustes M k-hoist [frankenscipy-146ld].
    /// Point-cloud alignment shape: large n, small d. Run with
    /// `cargo test -p fsci-spatial procrustes_khoist_perf -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn procrustes_khoist_perf_vs_naive() {
        use std::time::Instant;
        let (n, d) = (200_000usize, 8usize);
        let mk = |seed: u64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..d)
                        .map(|j| {
                            ((seed
                                .wrapping_mul(i as u64 + 1)
                                .wrapping_add(j as u64 * 13 + 5))
                                % 4099) as f64
                                / 1024.0
                        })
                        .collect()
                })
                .collect()
        };
        let a = mk(1);
        let b = mk(2);
        let reps = 20;

        let t0 = Instant::now();
        let mut sink = 0.0f64;
        for _ in 0..reps {
            let mut m = vec![vec![0.0f64; d]; d];
            for i in 0..d {
                for j in 0..d {
                    for k in 0..n {
                        m[i][j] += a[k][i] * b[k][j];
                    }
                }
            }
            sink += m[d - 1][d - 1];
        }
        let naive = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..reps {
            let mut m = vec![vec![0.0f64; d]; d];
            for k in 0..n {
                let (r2, r1) = (&a[k], &b[k]);
                for i in 0..d {
                    let v = r2[i];
                    let mi = &mut m[i];
                    for j in 0..d {
                        mi[j] += v * r1[j];
                    }
                }
            }
            sink += m[d - 1][d - 1];
        }
        let hoist = t1.elapsed();

        let ratio = naive.as_secs_f64() / hoist.as_secs_f64();
        println!("procrustes M: naive={naive:?} hoist={hoist:?} speedup={ratio:.2}x sink={sink}");
        assert!(
            ratio > 1.0,
            "k-hoist should be at least as fast (got {ratio:.2}x)"
        );
    }

    fn point_set_contains(points: &[Vec<f64>], expected: &[f64]) -> bool {
        points
            .iter()
            .any(|point| points_approx_eq(point, expected, 1e-10))
    }

    #[test]
    fn mahalanobis_identity_inverse_equals_euclidean() {
        // /testing-metamorphic: with VI = I (identity inverse covariance),
        //   mahalanobis(x, y, I) = sqrt((x−y)·(x−y)) = euclidean(x, y)
        // Pin across multiple (x, y) pairs and dimensionalities.
        let cases: &[(Vec<f64>, Vec<f64>)] = &[
            (vec![0.0_f64, 0.0], vec![3.0, 4.0]),                 // 5
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),           // sqrt(27)
            (vec![1.0, 1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0, 0.0]), // 2
        ];
        for (x, y) in cases {
            let n = x.len();
            let mut id = vec![vec![0.0; n]; n];
            for (i, row) in id.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            let m = mahalanobis(x, y, &id);
            let e = euclidean(x, y);
            assert!(
                (m - e).abs() < 1e-12,
                "mahalanobis({x:?}, {y:?}, I) = {m}, expected euclidean = {e}"
            );
        }
    }

    #[test]
    fn euclidean_distance() {
        assert!((euclidean(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert!((euclidean(&[1.0], &[1.0])).abs() < 1e-12);
    }

    #[test]
    fn cityblock_distance() {
        assert!((cityblock(&[0.0, 0.0], &[3.0, 4.0]) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_distance() {
        assert!((chebyshev(&[0.0, 0.0], &[3.0, 4.0]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cdist_metamorphic_transpose_symmetry() {
        // /testing-metamorphic: cdist(X, Y)[i][j] = cdist(Y, X)[j][i]
        // for any metric (Euclidean is symmetric in its arguments).
        // Catches any regression that introduces direction-dependent
        // logic into the matrix-pair distance loop.
        let xa = vec![vec![0.0_f64, 0.0], vec![1.0, 0.0], vec![3.0, 4.0]];
        let xb = vec![vec![0.0_f64, 1.0], vec![1.0, 1.0]];
        let d_ab = cdist(&xa, &xb).unwrap();
        let d_ba = cdist(&xb, &xa).unwrap();
        for i in 0..xa.len() {
            for j in 0..xb.len() {
                assert!(
                    (d_ab[i][j] - d_ba[j][i]).abs() < 1e-12,
                    "cdist(X,Y)[{i}][{j}] = {} != cdist(Y,X)[{j}][{i}] = {}",
                    d_ab[i][j],
                    d_ba[j][i]
                );
            }
        }
    }

    #[test]
    fn cdist_basic() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = cdist(&xa, &xb).expect("cdist");
        assert!((d[0][0] - 1.0).abs() < 1e-12); // (0,0) to (0,1)
        assert!((d[0][1] - 2.0_f64.sqrt()).abs() < 1e-12); // (0,0) to (1,1)
        assert!((d[1][0] - 2.0_f64.sqrt()).abs() < 1e-12); // (1,0) to (0,1)
        assert!((d[1][1] - 1.0).abs() < 1e-12); // (1,0) to (1,1)
    }

    #[test]
    fn kdtree_nearest_neighbor() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, dist) = tree.query(&[0.6, 0.6]).expect("query");
        assert_eq!(idx, 4); // closest to (0.5, 0.5)
        assert!(dist < 0.2);
    }

    #[test]
    fn kdtree_node_slab_preserves_every_original_coordinate() {
        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0],
        ];
        let tree = KDTree::new(&data).expect("kdtree");

        assert_eq!(tree.points.len(), data.len() * data[0].len());
        for (node_idx, node) in tree.nodes.iter().enumerate() {
            assert_eq!(tree.node_point(node_idx), data[node.index].as_slice());
        }
        assert_eq!(tree.query(&[8.8, 9.8, 10.8]).expect("query").0, 3);
    }

    /// `KDTree::query`, `query_k` and `query_ball_point` against SciPy.
    ///
    /// The pre-existing KDTree tests are self-referential: `kdtree_k_nearest`
    /// asserts only that two distances are `< 0.6` (which a SQUARED distance of
    /// 0.25 would also satisfy, so the distance convention itself was
    /// unpinned), `kdtree_query_ball_point` sorts the result before comparing
    /// so the return order was unpinned, and
    /// `k_nearest_neighbors_kdtree_matches_brute_force_bitwise` cross-checks
    /// against fsci's own brute force rather than against SciPy. This pins the
    /// conventions to `scipy.spatial.KDTree` on scipy 1.17.1 / numpy 2.4.3.
    ///
    /// TIES: where two neighbours are equidistant, SciPy's index order is a
    /// heap artifact (it returns `[2, 1]` and `[3, 0]` below, not ascending
    /// index order). That is not a contract worth pinning, so tied groups are
    /// compared as SETS while the distances stay exact.
    /// [frankenscipy-9g0xv]
    #[test]
    fn kdtree_query_family_matches_scipy_golden() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
            vec![0.5, 0.5],
        ];
        let tree = KDTree::new(&data).expect("kdtree");

        // scipy: t.query([0.4, 0.4]) -> (0.14142135623730948, 6)
        // The distance is EUCLIDEAN, not squared: sqrt(0.02) = 0.1414...,
        // where the squared value would be 0.02.
        let (idx, dist) = tree.query(&[0.4, 0.4]).expect("query");
        assert_eq!(idx, 6, "query nearest index");
        assert!(
            (dist - 0.141_421_356_237_309_48).abs() < 1e-15,
            "query distance {dist} vs scipy 0.14142135623730948 (squared would be 0.02)"
        );

        // scipy: t.query([0.4, 0.4], k=3)
        //   -> dists [0.14142135623730948, 0.5656854249492381, 0.7211102550927979]
        //      idx   [6, 0, 2]
        // Ranks 0 and 1 are unambiguous. Rank 2 is NOT: points 1 = [1,0] and
        // 2 = [0,1] are both at sqrt(0.52) = 0.7211102550927978 from [0.4,0.4],
        // so which one fills the third slot is a tie-break. SciPy returns 2 and
        // fsci returns 1; both are correct, so only the distance is pinned and
        // membership is checked against the tied pair. (Writing this test as
        // `assert_eq!(idx, 2)` is what a naive port of SciPy's output would do,
        // and it would fail for no good reason.)
        let got = tree.query_k(&[0.4, 0.4], 3).expect("query_k");
        let want_d = [
            0.141_421_356_237_309_48,
            0.565_685_424_949_238_1,
            0.721_110_255_092_797_8,
        ];
        assert_eq!(got.len(), 3, "query_k length");
        for (rank, ((_, gd), wd)) in got.iter().zip(want_d.iter()).enumerate() {
            assert!(
                (gd - wd).abs() < 1e-15,
                "query_k rank {rank} distance {gd} vs scipy {wd}"
            );
        }
        assert_eq!(got[0].0, 6, "query_k rank 0 index");
        assert_eq!(got[1].0, 0, "query_k rank 1 index");
        assert!(
            got[2].0 == 1 || got[2].0 == 2,
            "query_k rank 2 index {} must be one of the tied pair {{1, 2}}",
            got[2].0
        );

        // scipy: t.query([1.0, 1.0], k=4)
        //   -> dists [0.0, 0.7071067811865476, 1.0, 1.0]
        //      idx   [3, 6, 2, 1]
        // The last two tie at distance 1.0; compare that pair as a set.
        let got = tree.query_k(&[1.0, 1.0], 4).expect("query_k");
        let want_d = [0.0, 0.707_106_781_186_547_6, 1.0, 1.0];
        for (rank, ((_, gd), wd)) in got.iter().zip(want_d.iter()).enumerate() {
            assert!(
                (gd - wd).abs() < 1e-15,
                "tie case rank {rank} distance {gd} vs scipy {wd}"
            );
        }
        assert_eq!(got[0].0, 3, "tie case nearest is the exact match");
        assert_eq!(got[1].0, 6, "tie case second is unambiguous");
        let mut tied: Vec<usize> = got[2..].iter().map(|&(i, _)| i).collect();
        tied.sort_unstable();
        assert_eq!(tied, vec![1, 2], "tie case: the distance-1.0 pair");

        // scipy query_ball_point returns indices SORTED for a single query
        // point, so the order is asserted directly rather than sorted first.
        //   t.query_ball_point([0, 0], 1.2) -> [0, 1, 2, 6]
        //   t.query_ball_point([1, 1], 1.5) -> [0, 1, 2, 3, 4, 6]
        //   t.query_ball_point([0, 0], 0.0) -> [0]   (radius 0 keeps an exact hit)
        assert_eq!(
            tree.query_ball_point(&[0.0, 0.0], 1.2).expect("ball"),
            vec![0, 1, 2, 6],
            "query_ball_point r=1.2 (scipy returns sorted indices)"
        );
        assert_eq!(
            tree.query_ball_point(&[1.0, 1.0], 1.5).expect("ball"),
            vec![0, 1, 2, 3, 4, 6],
            "query_ball_point r=1.5"
        );
        assert_eq!(
            tree.query_ball_point(&[0.0, 0.0], 0.0).expect("ball"),
            vec![0],
            "query_ball_point r=0 still returns an exact hit"
        );

        // 1-D tie case, scipy: KDTree([[0],[1],[2],[3],[4]]).query([1.5], k=4)
        //   -> dists [0.5, 0.5, 1.5, 1.5], idx [1, 2, 3, 0]
        // Both pairs tie, so both are set-compared. This is the case the old
        // kdtree_k_nearest hedged about in a comment instead of asserting.
        let line = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let line_tree = KDTree::new(&line).expect("kdtree");
        let got = line_tree.query_k(&[1.5], 4).expect("query_k");
        for (rank, wd) in [0.5, 0.5, 1.5, 1.5].iter().enumerate() {
            assert!(
                (got[rank].1 - wd).abs() < 1e-15,
                "1-D rank {rank} distance {} vs scipy {wd}",
                got[rank].1
            );
        }
        let mut near: Vec<usize> = got[..2].iter().map(|&(i, _)| i).collect();
        near.sort_unstable();
        assert_eq!(near, vec![1, 2], "1-D nearest tied pair");
        let mut far: Vec<usize> = got[2..].iter().map(|&(i, _)| i).collect();
        far.sort_unstable();
        assert_eq!(far, vec![0, 3], "1-D second tied pair");
    }

    #[test]
    fn kdtree_exact_match() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, dist) = tree.query(&[1.0, 2.0, 3.0]).expect("query");
        assert_eq!(idx, 0);
        assert!(dist < 1e-12);
    }

    #[test]
    fn kdtree_k_nearest() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let results = tree.query_k(&[1.5], 3).expect("query_k");
        assert_eq!(results.len(), 3);
        // Nearest to 1.5: 1.0 (dist=0.5), 2.0 (dist=0.5), 0.0 (dist=1.5) or 3.0 (dist=1.5)
        assert!(results[0].1 < 0.6);
        assert!(results[1].1 < 0.6);
    }

    #[test]
    fn kdtree_query_k_zero_returns_empty() {
        let data = vec![vec![0.0], vec![1.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let results = tree.query_k(&[0.0], 0).expect("query_k");
        assert!(results.is_empty());
    }

    #[test]
    fn kdtree_empty_rejected() {
        let err = KDTree::new(&[]).expect_err("empty");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn kdtree_rejects_mixed_input_dimensions() {
        let err = KDTree::new(&[vec![0.0, 1.0], vec![2.0]])
            .expect_err("mixed-dimension points should be rejected");
        assert!(matches!(
            err,
            SpatialError::DimensionMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn kdtree_rejects_non_finite_points() {
        let err = KDTree::new(&[vec![0.0, 1.0], vec![f64::NAN, 2.0]])
            .expect_err("non-finite points should be rejected");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let err = tree.query(&[1.0]).expect_err("dim mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_rejects_non_finite_query() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let err = tree.query(&[f64::NAN, 0.0]).expect_err("nan query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
        let err = tree
            .query_k(&[f64::INFINITY, 0.0], 1)
            .expect_err("inf query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
        let err = tree
            .query_ball_point(&[0.0, f64::NEG_INFINITY], 1.0)
            .expect_err("inf query");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let tree = KDTree::new(&data).expect("kdtree");
        let (idx, _) = tree.query(&[0.0, 0.0]).expect("query");
        assert_eq!(idx, 0);
    }

    #[test]
    fn ckdtree_alias_construction_and_query() {
        let data = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![0.0, 2.0]];
        let tree = cKDTree::new(&data).expect("ckdtree");
        let (idx, dist) = tree.query(&[0.2, 0.1]).expect("query");
        assert_eq!(idx, 0);
        assert!(dist < 0.25);
    }

    #[test]
    fn ckdtree_alias_supports_cross_tree_methods() {
        let left = cKDTree::new(&[vec![0.0, 0.0], vec![2.0, 0.0]]).expect("left");
        let right = cKDTree::new(&[vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]]).expect("right");

        let neighbors = left.query_ball_tree(&right, 0.75).expect("neighbors");
        assert_eq!(neighbors, vec![vec![0], vec![1]]);

        let count = left.count_neighbors(&right, 0.75).expect("count");
        assert_eq!(count, 2);
    }

    #[test]
    fn ckdtree_alias_supports_query_pairs() {
        let tree = cKDTree::new(&[
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![1.2, 0.0],
            vec![5.0, 5.0],
        ])
        .expect("tree");

        let pairs = tree.query_pairs(0.8).expect("pairs");
        assert_eq!(pairs, vec![(0, 1), (1, 2)]);
    }

    // ── New distance metrics ───────────────────────────────────

    #[test]
    fn cosine_distance_orthogonal() {
        // Orthogonal vectors: cosine distance = 1
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_identical() {
        // Same direction: cosine distance = 0
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0])).abs() < 1e-12);
    }

    #[test]
    fn cosine_distance_opposite() {
        // Opposite direction: cosine distance = 2
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn minkowski_p1_is_cityblock() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, 1.0) - cityblock(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn minkowski_p2_is_euclidean() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, 2.0) - euclidean(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn minkowski_pinf_is_chebyshev() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!((minkowski(&a, &b, f64::INFINITY) - chebyshev(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn minkowski_rejects_nonpositive_p_like_weighted_variant() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, 1.0];
        assert!(minkowski(&a, &b, 0.0).is_nan());
        assert!(minkowski(&a, &b, -1.0).is_nan());
    }

    #[test]
    fn correlation_distance_perfect() {
        // Perfectly correlated: distance = 0
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        assert!(correlation(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn correlation_distance_anticorrelated() {
        // Anti-correlated: distance = 2
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [8.0, 6.0, 4.0, 2.0];
        assert!((correlation(&a, &b) - 2.0).abs() < 1e-12);
    }

    // ── pdist and squareform ──────────────────────────────────

    #[test]
    fn pdist_euclidean_three_points() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let d = pdist(&x, DistanceMetric::Euclidean).expect("pdist");
        // 3 points -> 3 distances: (0,1), (0,2), (1,2)
        assert_eq!(d.len(), 3);
        assert!((d[0] - 1.0).abs() < 1e-12); // dist(0,1)
        assert!((d[1] - 1.0).abs() < 1e-12); // dist(0,2)
        assert!((d[2] - 2.0_f64.sqrt()).abs() < 1e-12); // dist(1,2)
    }

    #[test]
    fn pdist_cityblock() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let d = pdist(&x, DistanceMetric::Cityblock).expect("pdist");
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-12); // |1|+|1|
        assert!((d[1] - 2.0).abs() < 1e-12); // |2|+|0|
        assert!((d[2] - 2.0).abs() < 1e-12); // |1|+|1|
    }

    #[test]
    fn squareform_roundtrip() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let condensed = pdist(&x, DistanceMetric::Euclidean).expect("pdist");
        let matrix = squareform_to_matrix(&condensed).expect("to_matrix");
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        // Diagonal is zero
        assert!((matrix[0][0]).abs() < 1e-12);
        assert!((matrix[1][1]).abs() < 1e-12);
        assert!((matrix[2][2]).abs() < 1e-12);
        // Symmetric
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-12);
        // Roundtrip back to condensed
        let condensed2 = squareform_to_condensed(&matrix).expect("to_condensed");
        for (a, b) in condensed.iter().zip(condensed2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn squareform_invalid_length() {
        assert!(squareform_to_matrix(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn squareform_to_condensed_rejects_asymmetric_matrix() {
        let matrix = vec![vec![0.0, 1.0], vec![2.0, 0.0]];
        assert!(squareform_to_condensed(&matrix).is_err());
    }

    #[test]
    fn squareform_to_condensed_rejects_nonzero_diagonal() {
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 0.0]];
        assert!(squareform_to_condensed(&matrix).is_err());
    }

    #[test]
    fn is_valid_dm_accepts_symmetric_zero_diagonal() {
        let m = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ];
        assert!(is_valid_dm(&m, 0.0));
        assert_eq!(num_obs_dm(&m), 3);
    }

    #[test]
    fn is_valid_dm_rejects_asymmetric() {
        let m = vec![vec![0.0, 1.0], vec![2.0, 0.0]];
        assert!(!is_valid_dm(&m, 0.0));
        assert!(is_valid_dm(&m, 1.5));
    }

    #[test]
    fn is_valid_dm_rejects_nonzero_diagonal_and_ragged_and_nan() {
        assert!(!is_valid_dm(&[vec![1.0, 2.0], vec![2.0, 0.0]], 0.0));
        assert!(!is_valid_dm(&[vec![0.0, 1.0], vec![1.0]], 0.0));
        assert!(!is_valid_dm(
            &[vec![0.0, f64::NAN], vec![f64::NAN, 0.0]],
            0.0
        ));
        assert!(!is_valid_dm(&[], 0.0));
    }

    #[test]
    fn is_valid_dm_fused_nan_scan_matches_previous_behavior() {
        fn previous(matrix: &[Vec<f64>], tol: f64) -> bool {
            let n = matrix.len();
            if n == 0 {
                return false;
            }
            for row in matrix {
                if row.len() != n {
                    return false;
                }
                if row.iter().any(|value| value.is_nan()) {
                    return false;
                }
            }
            for (i, row) in matrix.iter().enumerate() {
                if row[i].abs() > tol {
                    return false;
                }
                for j in (i + 1)..n {
                    if (row[j] - matrix[j][i]).abs() > tol {
                        return false;
                    }
                }
            }
            true
        }

        let payload_nan = f64::from_bits(0x7ff8_0000_0000_0042);
        let matrices = [
            vec![],
            vec![vec![0.0]],
            vec![vec![]],
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![0.0, 1.0], vec![2.0, 0.0]],
            vec![vec![1.0, 2.0], vec![2.0, 0.0]],
            vec![vec![0.0, payload_nan], vec![payload_nan, 0.0]],
            vec![vec![0.0, payload_nan], vec![1.0, 0.0]],
            vec![vec![0.0, 1.0], vec![payload_nan, 0.0]],
            vec![vec![payload_nan, 1.0], vec![1.0, 0.0]],
            vec![vec![-0.0, 0.0], vec![-0.0, 0.0]],
            vec![vec![0.0, f64::INFINITY], vec![f64::INFINITY, 0.0]],
            vec![vec![0.0, f64::INFINITY], vec![f64::NEG_INFINITY, 0.0]],
            vec![vec![0.0, 1.0], vec![1.0]],
            vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0]],
        ];
        let tolerances = [0.0, 0.5, 1.5, f64::INFINITY, f64::NAN, -0.5];

        for (case, matrix) in matrices.iter().enumerate() {
            for &tol in &tolerances {
                assert_eq!(
                    is_valid_dm(matrix, tol),
                    previous(matrix, tol),
                    "case {case}, tol {tol:?}"
                );
            }
        }
    }

    #[test]
    fn is_valid_y_correct_lengths() {
        // n=1 has 0 pairs; n=2 → 1; n=3 → 3; n=4 → 6; n=5 → 10.
        assert!(is_valid_y(&[]));
        assert!(is_valid_y(&[1.0]));
        assert!(is_valid_y(&[1.0, 2.0, 3.0]));
        assert!(is_valid_y(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert!(is_valid_y(&[1.0; 10]));
    }

    #[test]
    fn is_valid_y_rejects_invalid_lengths_and_nan() {
        // 2 and 4 and 5 and 7 are not triangular numbers.
        assert!(!is_valid_y(&[1.0, 2.0]));
        assert!(!is_valid_y(&[1.0, 2.0, 3.0, 4.0]));
        assert!(!is_valid_y(&[1.0; 5]));
        assert!(!is_valid_y(&[1.0; 7]));
        assert!(!is_valid_y(&[f64::NAN, 1.0, 2.0]));
    }

    #[test]
    fn num_obs_y_inverse_of_pair_count() {
        // num_obs_y(condensed) gives back N.
        assert_eq!(num_obs_y(&[]), 1);
        assert_eq!(num_obs_y(&[1.0]), 2);
        assert_eq!(num_obs_y(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(num_obs_y(&[1.0; 6]), 4);
        assert_eq!(num_obs_y(&[1.0; 10]), 5);
        assert_eq!(num_obs_y(&[1.0, 2.0]), 0); // invalid
    }

    #[test]
    fn squareform_validators_roundtrip() {
        let m = vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 0.0, 4.0, 5.0],
            vec![2.0, 4.0, 0.0, 6.0],
            vec![3.0, 5.0, 6.0, 0.0],
        ];
        assert!(is_valid_dm(&m, 0.0));
        let y = squareform_to_condensed(&m).expect("to condensed");
        assert!(is_valid_y(&y));
        assert_eq!(num_obs_y(&y), num_obs_dm(&m));
        let m2 = squareform_to_matrix(&y).expect("to matrix");
        assert!(is_valid_dm(&m2, 0.0));
    }

    #[test]
    fn pdist_too_few_points() {
        assert!(pdist(&[vec![1.0]], DistanceMetric::Euclidean).is_err());
    }

    #[test]
    fn cdist_metric_cosine() {
        let xa = vec![vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let d = cdist_metric(&xa, &xb, DistanceMetric::Cosine).expect("cdist cosine");
        assert!((d[0][0] - 1.0).abs() < 1e-12); // orthogonal
        assert!((d[0][1]).abs() < 1e-12); // identical direction
    }

    #[test]
    fn distance_matrix_matches_cdist() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let dm = distance_matrix(&xa, &xb).expect("distance_matrix");
        let cd = cdist(&xa, &xb).expect("cdist");
        assert_eq!(dm.len(), cd.len());
        for (row_dm, row_cd) in dm.iter().zip(cd.iter()) {
            assert_eq!(row_dm.len(), row_cd.len());
            for (&a, &b) in row_dm.iter().zip(row_cd.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "distance_matrix mismatch: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn rectangle_normalizes_bounds_and_volume() {
        let rect = Rectangle::new(&[0.0, 1.0], &[1.0, 0.0]).expect("rectangle");
        assert_eq!(rect.mins, vec![0.0, 0.0]);
        assert_eq!(rect.maxes, vec![1.0, 1.0]);
        assert_eq!(rect.m, 2);
        assert!((rect.volume() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_split_matches_scipy_semantics() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        let (lower, upper) = rect.split(0, 0.5).expect("split");
        assert_eq!(lower.mins, vec![0.0, -1.0]);
        assert_eq!(lower.maxes, vec![0.5, 2.0]);
        assert_eq!(upper.mins, vec![0.5, -1.0]);
        assert_eq!(upper.maxes, vec![1.0, 2.0]);

        let (outside_lower, outside_upper) = rect.split(0, -1.0).expect("outside split");
        assert_eq!(outside_lower.mins, vec![-1.0, -1.0]);
        assert_eq!(outside_lower.maxes, vec![0.0, 2.0]);
        assert_eq!(outside_upper.mins, vec![-1.0, -1.0]);
        assert_eq!(outside_upper.maxes, vec![1.0, 2.0]);
    }

    #[test]
    fn rectangle_point_distances_match_scipy() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        assert!(rect.min_distance_point(&[0.5, 0.0], 2.0).unwrap().abs() < 1e-12);
        assert!(
            (rect.max_distance_point(&[0.5, 0.0], 2.0).unwrap() - 2.0615528128088303).abs() < 1e-12
        );
        assert!((rect.min_distance_point(&[3.0, 0.0], 1.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((rect.max_distance_point(&[3.0, 0.0], f64::INFINITY).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_rectangle_distances_match_scipy() {
        let rect = Rectangle::new(&[1.0, 2.0], &[0.0, -1.0]).expect("rectangle");
        let other = Rectangle::new(&[2.0, 1.0], &[1.5, -2.0]).expect("other");
        assert!((rect.min_distance_rectangle(&other, 2.0).unwrap() - 0.5).abs() < 1e-12);
        assert!(
            (rect.max_distance_rectangle(&other, 2.0).unwrap() - 4.47213595499958).abs() < 1e-12
        );

        let far = Rectangle::new(&[4.0, 4.0], &[2.0, 1.0]).expect("far");
        assert!((rect.min_distance_rectangle(&far, 1.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((rect.max_distance_rectangle(&far, f64::INFINITY).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rectangle_rejects_invalid_inputs() {
        let err = Rectangle::new(&[1.0, 2.0], &[0.0]).expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));

        let rect = Rectangle::new(&[1.0], &[0.0]).expect("rectangle");
        let err = rect.split(2, 0.5).expect_err("bad axis");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));

        let err = rect
            .min_distance_point(&[0.0], 0.0)
            .expect_err("invalid p should be rejected");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn cdist_metric_dimension_mismatch_in_later_row() {
        let xa = vec![vec![0.0, 0.0], vec![1.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let err = cdist_metric(&xa, &xb, DistanceMetric::Euclidean)
            .expect_err("later row dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_large_random() {
        // Verify KDTree gives same result as brute force for 100 points
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![(i as f64 * 0.73) % 10.0, (i as f64 * 1.37) % 10.0])
            .collect();
        let tree = KDTree::new(&data).expect("kdtree");
        let query = [5.0, 5.0];

        let (tree_idx, tree_dist) = tree.query(&query).expect("query");

        // Brute force
        let (brute_idx, brute_dist) = data
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean(p, &query)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();

        assert_eq!(tree_idx, brute_idx);
        assert!((tree_dist - brute_dist).abs() < 1e-10);
    }

    // ── New distance metric tests ──────────────────────────────────

    #[test]
    fn hamming_basic() {
        assert!((hamming(&[1.0, 0.0, 1.0], &[1.0, 1.0, 1.0]) - 1.0 / 3.0).abs() < 1e-12);
        assert!((hamming(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-12);
    }

    #[test]
    fn jaccard_basic() {
        // u=[1,0,1], v=[1,1,0]: TT=1, TF=1, FT=1 → J = 2/3
        assert!((jaccard(&[1.0, 0.0, 1.0], &[1.0, 1.0, 0.0]) - 2.0 / 3.0).abs() < 1e-12);
        // identical nonzero → J = 0
        assert!(jaccard(&[1.0, 1.0], &[1.0, 1.0]).abs() < 1e-12);
    }

    #[test]
    fn jaccard_all_zeros() {
        // Both zero → J = 0 (no nonzero elements)
        assert!(jaccard(&[0.0, 0.0], &[0.0, 0.0]).abs() < 1e-12);
    }

    #[test]
    fn jaccard_real_valued_matches_modern_scipy() {
        // [frankenscipy-z747j] scipy 1.15 changelog:
        // "Non-0/1 numeric input used to produce an ad hoc result.
        // Since 1.15.0, numeric input is converted to Boolean
        // before computation." This test pins the modern semantics.
        // (Supersedes [frankenscipy-ayl5s], which encoded the older
        // scipy ≤1.14 ad-hoc semantics.)
        //
        // jaccard([2.5], [3.7]) — both positions bool=True → c_TT=1,
        // c_TF=c_FT=0 → 0/1 = 0.0.
        assert!(jaccard(&[2.5_f64], &[3.7]).abs() < 1e-12);
        // jaccard([1,2,3], [1,5,7]) — all positions bool=True → all
        // c_TT, no TF/FT → 0.
        assert!(jaccard(&[1.0, 2.0, 3.0], &[1.0, 5.0, 7.0]).abs() < 1e-12);
        // jaccard([1,1,0], [1,0,1]) — pos0 TT, pos1 TF, pos2 FT
        // → 2/3.
        assert!((jaccard(&[1.0, 1.0, 0.0], &[1.0, 0.0, 1.0]) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn canberra_basic() {
        // |1-2|/(1+2) + |3-4|/(3+4) = 1/3 + 1/7
        let expected = 1.0 / 3.0 + 1.0 / 7.0;
        assert!((canberra(&[1.0, 3.0], &[2.0, 4.0]) - expected).abs() < 1e-12);
    }

    #[test]
    fn canberra_zeros() {
        // Both zero at a position → contribution is 0
        assert!((canberra(&[0.0, 1.0], &[0.0, 2.0]) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn braycurtis_basic() {
        // |1-2| + |3-4| = 2; |1+2| + |3+4| = 10 → 2/10 = 0.2
        assert!((braycurtis(&[1.0, 3.0], &[2.0, 4.0]) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn braycurtis_identical() {
        assert!(braycurtis(&[5.0, 10.0], &[5.0, 10.0]).abs() < 1e-12);
    }

    #[test]
    fn jensenshannon_matches_scipy_default_base() {
        let distance = jensenshannon(&[1.0, 0.0], &[0.5, 0.5], None);
        assert!((distance - 0.464501404022459).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_respects_base_kwarg() {
        let distance = jensenshannon(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], Some(2.0));
        assert!((distance - 1.0).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_normalizes_input_masses() {
        let normalized = jensenshannon(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], Some(2.0));
        let unnormalized = jensenshannon(&[2.0, 0.0, 0.0], &[0.0, 4.0, 0.0], Some(2.0));
        assert!((normalized - unnormalized).abs() < 1e-15);
    }

    #[test]
    fn jensenshannon_zero_mass_vectors_match_scipy_nan() {
        let distance = jensenshannon(&[0.0, 0.0], &[0.0, 0.0], None);
        assert!(distance.is_nan());
    }

    #[test]
    fn jensenshannon_simd_matches_old_normalized_vec_bits() {
        fn old_reference(p: &[f64], q: &[f64], base: Option<f64>) -> f64 {
            if p.is_empty() || q.is_empty() || p.len() != q.len() {
                return f64::NAN;
            }
            let sum_p: f64 = p.iter().sum();
            let sum_q: f64 = q.iter().sum();
            if sum_p <= 0.0 || sum_q <= 0.0 || !sum_p.is_finite() || !sum_q.is_finite() {
                return f64::NAN;
            }

            let normalized_p: Vec<f64> = p.iter().map(|value| value / sum_p).collect();
            let normalized_q: Vec<f64> = q.iter().map(|value| value / sum_q).collect();

            let js = normalized_p
                .iter()
                .zip(normalized_q.iter())
                .map(|(&left, &right)| {
                    let mean = (left + right) / 2.0;
                    relative_entropy(left, mean) + relative_entropy(right, mean)
                })
                .sum::<f64>();

            let scaled = match base {
                Some(log_base) => js / log_base.ln(),
                None => js,
            };
            (scaled / 2.0).sqrt()
        }

        for n in [1usize, 2, 3, 7, 8, 31, 256, 1025] {
            let p: Vec<f64> = (0..n)
                .map(|i| 0.1 + (i as f64 * 0.37).sin().abs())
                .collect();
            let q: Vec<f64> = (0..n)
                .map(|i| 0.1 + (i as f64 * 0.41 + 1.0).cos().abs())
                .collect();
            for base in [None, Some(2.0), Some(10.0)] {
                assert_eq!(
                    jensenshannon(&p, &q, base).to_bits(),
                    old_reference(&p, &q, base).to_bits(),
                    "n={n} base={base:?}"
                );
            }
        }
    }

    #[test]
    fn pdist_with_canberra() {
        let x = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let d = pdist(&x, DistanceMetric::Canberra).unwrap();
        assert_eq!(d.len(), 3);
        // All distances should be positive
        for &di in &d {
            assert!(di >= 0.0);
        }
    }

    // ── KDTree extension tests ─────────────────────────────────────

    #[test]
    fn kdtree_query_ball_point() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ];
        let tree = KDTree::new(&data).unwrap();

        // Radius 1.5 from origin should include points 0, 1, 2 but not 3
        let mut result = tree.query_ball_point(&[0.0, 0.0], 1.5).unwrap();
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);

        // Radius 0.5 from origin should include only point 0
        let result = tree.query_ball_point(&[0.0, 0.0], 0.5).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn kdtree_query_ball_point_empty() {
        let data = vec![vec![10.0, 10.0]];
        let tree = KDTree::new(&data).unwrap();
        let result = tree.query_ball_point(&[0.0, 0.0], 1.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn kdtree_query_ball_point_negative_radius_rejected() {
        let tree = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let err = tree
            .query_ball_point(&[0.0, 0.0], -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_count_neighbors() {
        let data1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let data2 = vec![vec![0.5, 0.0], vec![10.0, 10.0]];
        let tree1 = KDTree::new(&data1).unwrap();
        let tree2 = KDTree::new(&data2).unwrap();

        // Within radius 1.0: (0,0)↔(0.5,0) and (1,0)↔(0.5,0) = 2 pairs
        let count = tree1.count_neighbors(&tree2, 1.0).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn kdtree_count_neighbors_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .count_neighbors(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_count_neighbors_negative_radius_rejected() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .count_neighbors(&tree2, -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_query_ball_tree() {
        let data1 = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![5.0, 5.0]];
        let data2 = vec![vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]];
        let tree1 = KDTree::new(&data1).unwrap();
        let tree2 = KDTree::new(&data2).unwrap();

        let neighbors = tree1.query_ball_tree(&tree2, 0.75).unwrap();
        assert_eq!(neighbors.len(), data1.len());
        assert_eq!(neighbors[0], vec![0]);
        assert_eq!(neighbors[1], vec![1]);
        assert!(neighbors[2].is_empty());
    }

    #[test]
    fn kdtree_query_ball_tree_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .query_ball_tree(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_query_ball_tree_negative_radius_rejected() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .query_ball_tree(&tree2, -1.0)
            .expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_query_pairs_basic() {
        let tree = KDTree::new(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
            vec![5.0, 5.0],
        ])
        .unwrap();

        let pairs = tree.query_pairs(1.01).unwrap();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 3)]);
    }

    #[test]
    fn kdtree_query_pairs_negative_radius_rejected() {
        let tree = KDTree::new(&[vec![0.0, 0.0], vec![1.0, 0.0]]).unwrap();
        let err = tree.query_pairs(-1.0).expect_err("negative radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_basic() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]]).unwrap();

        let entries = tree1.sparse_distance_matrix(&tree2, 0.75).unwrap();
        assert_eq!(
            entries.entries(),
            &BTreeMap::from([((0, 0), 0.5), ((1, 1), 0.5)]),
            "default DOK output should preserve original point indices"
        );
        assert_eq!(entries.shape(), Shape2D::new(2, 3));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_empty() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![10.0, 10.0]]).unwrap();
        let entries = tree1.sparse_distance_matrix(&tree2, 1.0).unwrap();
        assert!(entries.entries().is_empty());
        assert_eq!(entries.shape(), Shape2D::new(1, 1));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_dimension_mismatch() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.0, 0.0, 0.0]]).unwrap();
        let err = tree1
            .sparse_distance_matrix(&tree2, 1.0)
            .expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn kdtree_sparse_distance_matrix_output_type_variants() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0], vec![2.5, 0.0], vec![9.0, 9.0]]).unwrap();

        let dict_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "dict")
            .unwrap();
        assert_eq!(
            dict_output,
            SparseDistanceMatrixOutput::Dict(BTreeMap::from([((0, 0), 0.5), ((1, 1), 0.5)]))
        );

        let ndarray_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "ndarray")
            .unwrap();
        assert_eq!(
            ndarray_output,
            SparseDistanceMatrixOutput::Ndarray(vec![
                SparseDistanceMatrixRecord { i: 0, j: 0, v: 0.5 },
                SparseDistanceMatrixRecord { i: 1, j: 1, v: 0.5 },
            ])
        );

        let coo_output = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 0.75, "coo_matrix")
            .unwrap();
        assert!(matches!(
            coo_output,
            SparseDistanceMatrixOutput::CooMatrix(_)
        ));
        if let SparseDistanceMatrixOutput::CooMatrix(matrix) = coo_output {
            assert_eq!(matrix.shape(), Shape2D::new(2, 3));
            assert_eq!(matrix.row_indices(), &[0, 1]);
            assert_eq!(matrix.col_indices(), &[0, 1]);
            assert_eq!(matrix.data(), &[0.5, 0.5]);
        }
    }

    #[test]
    fn kdtree_sparse_distance_matrix_rejects_invalid_output_type() {
        let tree1 = KDTree::new(&[vec![0.0, 0.0]]).unwrap();
        let tree2 = KDTree::new(&[vec![0.5, 0.0]]).unwrap();
        let err = tree1
            .sparse_distance_matrix_with_output_type(&tree2, 1.0, "bad")
            .expect_err("invalid output type");
        assert!(
            matches!(err, SpatialError::InvalidArgument(message) if message == "Invalid output type")
        );
    }

    // ── ConvexHull tests ─────────────────────────────────────────────

    #[test]
    fn convex_hull_square() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), 4, "square has 4 hull vertices");
        assert!((hull.area - 1.0).abs() < 1e-10, "area = {}", hull.area);
        assert!(
            (hull.perimeter - 4.0).abs() < 1e-10,
            "perimeter = {}",
            hull.perimeter
        );
    }

    #[test]
    fn convex_hull_triangle() {
        let points = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), 3, "triangle has 3 hull vertices");
        // Area of triangle with base 4, height 3 = 6
        assert!((hull.area - 6.0).abs() < 1e-10, "area = {}", hull.area);
    }

    #[test]
    fn convex_hull_interior_points_excluded() {
        // Square with interior point
        let points = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (1.0, 1.0), // interior
        ];
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(
            hull.vertices.len(),
            4,
            "interior point should be excluded: {:?}",
            hull.vertices
        );
        assert!(!hull.vertices.contains(&4), "point 4 is interior");
    }

    #[test]
    fn convex_hull_simplices_form_closed_polygon() {
        let points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let hull = ConvexHull::new(&points).expect("hull");
        // Each vertex appears exactly twice in simplices (as start and end of adjacent edges)
        let mut vertex_count = std::collections::HashMap::new();
        for &(a, b) in &hull.simplices {
            *vertex_count.entry(a).or_insert(0) += 1;
            *vertex_count.entry(b).or_insert(0) += 1;
        }
        for &v in &hull.vertices {
            assert_eq!(
                vertex_count.get(&v),
                Some(&2),
                "vertex {v} should appear exactly twice in simplices"
            );
        }
    }

    #[test]
    fn convex_hull_too_few_points() {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let err = ConvexHull::new(&points).expect_err("too few");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn convex_hull_collinear_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let err = ConvexHull::new(&points).expect_err("collinear");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn convex_hull_many_points() {
        // Generate points on a circle (all should be on hull) + center
        let n = 20;
        let mut points: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (theta.cos(), theta.sin())
            })
            .collect();
        points.push((0.0, 0.0)); // center point
        let hull = ConvexHull::new(&points).expect("hull");
        assert_eq!(hull.vertices.len(), n, "circle points all on hull");
        assert!(!hull.vertices.contains(&n), "center point excluded");
        // Area should approximate π
        assert!(
            (hull.area - std::f64::consts::PI).abs() < 0.1,
            "area ≈ π: {}",
            hull.area
        );
    }

    // ── HalfspaceIntersection tests ──────────────────────────────────

    #[test]
    fn qhull_error_variant_preserves_message() {
        let err = QhullError::new("QH6154 Qhull precision error");
        assert!(err.message().contains("QH6154"));
        let spatial: SpatialError = err.clone().into();
        assert_eq!(spatial.to_string(), err.to_string());
        assert!(matches!(spatial, SpatialError::Qhull(_)));
    }

    #[test]
    fn halfspace_intersection_square_matches_dual_surface() {
        let halfspaces = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ];
        let hs = HalfspaceIntersection::new(&halfspaces, (0.5, 0.5)).expect("halfspaces");

        assert_eq!(hs.ndim, 2);
        assert_eq!(hs.nineq, 4);
        assert!(hs.is_bounded);
        assert_eq!(hs.dual_vertices.len(), 4);
        assert_eq!(hs.dual_facets.len(), 4);
        assert!((hs.dual_points[0][0] + 2.0).abs() < 1e-10);
        assert!((hs.dual_points[1][1] + 2.0).abs() < 1e-10);
        assert!((hs.dual_volume - 8.0).abs() < 1e-10);
        assert!((hs.dual_area - 8.0 * 2.0_f64.sqrt()).abs() < 1e-10);

        for expected in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            assert!(
                point_set_contains(&hs.intersections, &expected),
                "missing intersection {expected:?}: {:?}",
                hs.intersections
            );
        }
    }

    #[test]
    fn halfspace_intersection_rejects_boundary_feasible_point_as_qhull_error() {
        let halfspaces = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
        ];
        let err = HalfspaceIntersection::new(&halfspaces, (0.0, 0.5))
            .expect_err("boundary feasible point");
        assert!(matches!(
            err,
            SpatialError::Qhull(ref qhull) if qhull.message().contains("QH6023")
        ));
    }

    #[test]
    fn halfspace_intersection_unbounded_region_is_marked() {
        let halfspaces = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0]];
        let hs = HalfspaceIntersection::new(&halfspaces, (0.5, 0.5)).expect("halfspaces");

        assert!(!hs.is_bounded);
        assert!(point_set_contains(&hs.intersections, &[0.0, 0.0]));
        assert!(point_set_contains(&hs.intersections, &[1.0, 0.0]));
        assert!(
            hs.intersections
                .iter()
                .any(|row| row.iter().any(|value| !value.is_finite())),
            "unbounded dual edge should surface a non-finite intersection"
        );
    }

    #[test]
    fn halfspace_intersection_from_nd_supports_bounded_3d_tetrahedron() {
        let halfspaces = vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 1.0, 1.0, -1.0],
        ];
        let hs = HalfspaceIntersection::from_nd(&halfspaces, &[0.2, 0.2, 0.2]).expect("3D bounded");
        assert_eq!(hs.ndim, 3);
        assert!(hs.is_bounded);
        assert_eq!(hs.dual_vertices, vec![0, 1, 2, 3]);
        for expected in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            assert!(
                point_set_contains(&hs.intersections, &expected),
                "missing 3D intersection {expected:?}: {:?}",
                hs.intersections
            );
        }
        assert!(hs.dual_area.is_nan());
        assert!(hs.dual_volume.is_nan());
        assert!(hs.dual_equations.is_empty());
    }

    #[test]
    fn halfspace_vertex_candidates_parallel_matches_serial_bits_and_order() {
        let halfspaces = vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 0.0, 0.0, -1.0],
            vec![0.0, 1.0, 0.0, -1.0],
            vec![0.0, 0.0, 1.0, -1.0],
            vec![1.0, 1.0, 1.0, -1.8],
            vec![-1.0, -1.0, 0.0, 0.1],
        ];
        let ndim = 3;
        let mut combos = Vec::new();
        combinations_recursive(halfspaces.len(), ndim, 0, &mut Vec::new(), &mut combos);

        let serial = halfspace_vertex_candidates_nd_with_workers(&halfspaces, ndim, &combos, 1);
        let parallel = halfspace_vertex_candidates_nd_with_workers(&halfspaces, ndim, &combos, 4);

        assert_eq!(serial.len(), parallel.len());
        for ((serial_point, serial_combo), (parallel_point, parallel_combo)) in
            serial.iter().zip(parallel.iter())
        {
            assert_eq!(serial_combo, parallel_combo);
            assert_eq!(serial_point.len(), parallel_point.len());
            for (&serial_coord, &parallel_coord) in serial_point.iter().zip(parallel_point.iter()) {
                assert_eq!(serial_coord.to_bits(), parallel_coord.to_bits());
            }
        }
    }

    #[test]
    #[ignore]
    fn halfspace_vertex_candidates_parallel_perf_probe() {
        fn next_u64(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        }
        fn unit(state: &mut u64) -> f64 {
            (next_u64(state) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn halfspaces(m: usize, ndim: usize, seed: u64) -> Vec<Vec<f64>> {
            let mut state = seed;
            let mut out = Vec::with_capacity(m);
            while out.len() < m {
                let a = (0..ndim)
                    .map(|_| unit(&mut state) * 2.0 - 1.0)
                    .collect::<Vec<_>>();
                let nrm = a.iter().map(|value| value * value).sum::<f64>().sqrt();
                if nrm < 1e-6 {
                    continue;
                }
                let mut row = a.iter().map(|value| value / nrm).collect::<Vec<_>>();
                row.push(-1.0);
                out.push(row);
            }
            out
        }
        fn digest(candidates: &[NdVertexCandidate]) -> u64 {
            let mut h = 1469598103934665603u64;
            for (point, combo) in candidates {
                for &coord in point {
                    h = (h ^ coord.to_bits()).wrapping_mul(1099511628211);
                }
                h = h.wrapping_mul(31);
                for &idx in combo {
                    h = (h ^ idx as u64).wrapping_mul(1099511628211);
                }
                h = h.wrapping_mul(31);
            }
            h
        }

        for &(m, ndim) in &[(120usize, 3usize), (60, 4)] {
            let halfspaces = halfspaces(m, ndim, 7);
            let mut combos = Vec::new();
            combinations_recursive(halfspaces.len(), ndim, 0, &mut Vec::new(), &mut combos);
            let workers = halfspace_vertex_enum_thread_count(combos.len(), halfspaces.len(), ndim);

            let serial = halfspace_vertex_candidates_nd_with_workers(&halfspaces, ndim, &combos, 1);
            let parallel = halfspace_vertex_candidates_nd(&halfspaces, ndim, &combos);
            let serial_digest = digest(&serial);
            let parallel_digest = digest(&parallel);
            assert_eq!(serial.len(), parallel.len());
            assert_eq!(serial_digest, parallel_digest);

            let reps = 3;
            let start = std::time::Instant::now();
            let mut serial_acc = 0u64;
            for _ in 0..reps {
                let candidates =
                    halfspace_vertex_candidates_nd_with_workers(&halfspaces, ndim, &combos, 1);
                serial_acc ^= digest(std::hint::black_box(&candidates));
            }
            let serial_ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;

            let start = std::time::Instant::now();
            let mut parallel_acc = 0u64;
            for _ in 0..reps {
                let candidates = halfspace_vertex_candidates_nd(&halfspaces, ndim, &combos);
                parallel_acc ^= digest(std::hint::black_box(&candidates));
            }
            let parallel_ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            assert_eq!(serial_acc, parallel_acc);

            println!(
                "halfspace_vertex_candidates m={m} ndim={ndim} combos={} workers={workers} serial_ms={serial_ms:.6} parallel_ms={parallel_ms:.6} speedup={:.6} digest=0x{serial_digest:016x}",
                combos.len(),
                serial_ms / parallel_ms
            );
        }
    }

    #[test]
    fn halfspace_intersection_from_nd_marks_unbounded_3d_region() {
        let halfspaces = vec![
            vec![-1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0],
            vec![1.0, 1.0, 0.0, -1.0],
        ];
        let hs =
            HalfspaceIntersection::from_nd(&halfspaces, &[0.2, 0.2, 0.2]).expect("3D unbounded");

        assert_eq!(hs.ndim, 3);
        assert!(!hs.is_bounded);
        assert_eq!(hs.dual_vertices, vec![0, 1, 2, 3]);
        for expected in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            assert!(
                point_set_contains(&hs.intersections, &expected),
                "missing 3D unbounded vertex {expected:?}: {:?}",
                hs.intersections
            );
        }
        assert_eq!(hs.intersections.len(), 3);
        assert!(hs.dual_area.is_nan());
        assert!(hs.dual_volume.is_nan());
        assert!(hs.dual_equations.is_empty());
    }

    #[test]
    fn halfspace_intersection_from_nd_rejects_invalid_higher_dimensional_inputs() {
        let too_few_halfspaces = vec![vec![-1.0, 0.0, 0.0, 0.0]; 4];
        let err = HalfspaceIntersection::from_nd(&too_few_halfspaces, &[0.25, 0.25, 0.25, 0.25])
            .expect_err("4D input needs at least 5 halfspaces");
        assert!(matches!(err, SpatialError::Qhull(_)));

        let bad_rows = vec![vec![-1.0, 0.0, 0.0], vec![0.0, -1.0], vec![1.0, 1.0, -1.0]];
        let err = HalfspaceIntersection::from_nd(&bad_rows, &[0.25, 0.25])
            .expect_err("row shape mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn halfspace_intersection_add_halfspaces_recomputes_region() {
        let mut hs = HalfspaceIntersection::new(
            &[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, -1.0]],
            (0.5, 0.5),
        )
        .expect("initial halfspaces");
        assert!(!hs.is_bounded);

        hs.add_halfspaces(&[vec![0.0, 1.0, -1.0]], false)
            .expect("append y <= 1");
        assert!(hs.is_bounded);
        assert_eq!(hs.nineq, 4);
        assert!(point_set_contains(&hs.intersections, &[1.0, 1.0]));
    }

    // ── Delaunay tests ──────────────────────────────────────────────

    #[test]
    fn delaunay_square_triangulates() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        assert_eq!(tri.points, points);
        assert_eq!(
            tri.simplices.len(),
            2,
            "square should split into 2 triangles"
        );
        for &(a, b, c) in &tri.simplices {
            assert!(a < points.len() && b < points.len() && c < points.len());
            assert_ne!(cross(points[a], points[b], points[c]), 0.0);
        }
    }

    #[test]
    fn delaunay_find_simplex_inside_triangle() {
        let points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        let (_, l1, l2, l3) = tri.find_simplex((0.25, 0.5)).expect("inside simplex");
        assert!(l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0);
        assert!(((l1 + l2 + l3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn delaunay_find_simplex_outside_returns_none() {
        let points = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        assert!(tri.find_simplex((3.0, 3.0)).is_none());
    }

    #[test]
    fn tsearch_matches_find_simplex_and_marks_outside() {
        // Unit square -> two triangles; mix of inside and outside query points.
        let points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let tri = Delaunay::new(&points).expect("triangulation");
        let xi = [(0.25, 0.25), (0.75, 0.75), (5.0, 5.0), (-1.0, 0.5)];
        let got = tsearch(&tri, &xi);
        assert_eq!(got.len(), xi.len());
        for (&point, &idx) in xi.iter().zip(&got) {
            // tsearch is the functional form of find_simplex: same index, -1 outside.
            match tri.find_simplex(point) {
                Some((expected, _, _, _)) => assert_eq!(idx, expected as i64),
                None => assert_eq!(idx, -1),
            }
            // When a simplex is reported, the point must actually lie in it.
            if idx >= 0 {
                let (a, b, c) = tri.simplices[idx as usize];
                let (l1, l2, l3) =
                    barycentric_2d(tri.points[a], tri.points[b], tri.points[c], point);
                assert!(l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10);
            }
        }
        // The two interior points sit in real triangles; the two far points are out.
        assert!(got[0] >= 0 && got[1] >= 0);
        assert_eq!(got[2], -1);
        assert_eq!(got[3], -1);
    }

    #[test]
    fn delaunay_collinear_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let err = Delaunay::new(&points).expect_err("collinear");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn delaunay_too_few_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 0.0)];
        let err = Delaunay::new(&points).expect_err("too few");
        assert!(matches!(err, SpatialError::Qhull(_)));
    }

    #[test]
    fn delaunay_non_finite_points_rejected() {
        let points = [(0.0, 0.0), (1.0, 0.0), (f64::NAN, f64::NAN)];
        let err = Delaunay::new(&points).expect_err("non-finite points");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    // ── Voronoi tests ───────────────────────────────────────────────

    #[test]
    fn voronoi_triangle_has_single_vertex_and_infinite_ridges() {
        let points = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)];
        let vor = Voronoi::new(&points).expect("voronoi");

        assert_eq!(vor.points, points);
        assert_eq!(vor.vertices.len(), 1);
        assert!((vor.vertices[0].0 - 1.0).abs() < 1e-10);
        assert!((vor.vertices[0].1 - 0.75).abs() < 1e-10);
        assert_eq!(vor.ridge_points.len(), 3);
        assert_eq!(vor.ridge_vertices.len(), 3);
        assert!(vor.ridge_vertices.iter().all(|&(a, b)| a == -1 && b == 0));
        assert_eq!(vor.point_region.len(), 3);
        for &region_idx in &vor.point_region {
            let region = &vor.regions[region_idx];
            assert!(region.contains(&-1));
            assert!(region.contains(&0));
        }
    }

    #[test]
    fn voronoi_center_point_gets_finite_region() {
        let points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)];
        let vor = Voronoi::new(&points).expect("voronoi");

        assert_eq!(vor.vertices.len(), 4);
        let center_region = &vor.regions[vor.point_region[4]];
        assert_eq!(center_region.len(), 4);
        assert!(center_region.iter().all(|&idx| idx >= 0));

        let expected = [(2.0, 1.0), (1.0, 0.0), (1.0, 2.0), (0.0, 1.0)];
        for &(ex, ey) in &expected {
            assert!(
                vor.vertices
                    .iter()
                    .any(|&(vx, vy)| (vx - ex).abs() < 1e-10 && (vy - ey).abs() < 1e-10),
                "missing Voronoi vertex ({ex}, {ey})"
            );
        }

        for &vertex_idx in center_region {
            let vertex = vor.vertices[vertex_idx as usize];
            assert!(
                expected
                    .iter()
                    .any(|&(ex, ey)| (vertex.0 - ex).abs() < 1e-10 && (vertex.1 - ey).abs() < 1e-10),
                "unexpected center-region vertex ({}, {})",
                vertex.0,
                vertex.1
            );
        }
    }

    // ── Spherical Voronoi tests ─────────────────────────────────────

    #[test]
    fn spherical_voronoi_tetrahedron_has_four_vertices() {
        let scale = 1.0 / 3.0_f64.sqrt();
        let points = [
            [scale, scale, scale],
            [scale, -scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
        ];
        let sv = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect("spherical voronoi");

        assert_eq!(sv.points, points);
        assert_eq!(sv.vertices.len(), 4);
        assert_eq!(sv.regions.len(), 4);
        for region in &sv.regions {
            assert_eq!(region.len(), 3);
            for &vertex_idx in region {
                assert!(vertex_idx < sv.vertices.len());
            }
        }
        for vertex in &sv.vertices {
            assert!((norm3(*vertex) - 1.0).abs() < 1e-10);
        }
    }

    // Deterministic pseudo-random unit-sphere points (LCG → Marsaglia method).
    fn lcg_sphere_points(n: usize, mut state: u64) -> Vec<[f64; 3]> {
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut pts = Vec::with_capacity(n);
        while pts.len() < n {
            let (a, b) = (next(), next());
            let s = a * a + b * b;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let f = 2.0 * (1.0 - s).sqrt();
            pts.push([a * f, b * f, 1.0 - 2.0 * s]);
        }
        pts
    }

    fn facet_set(facets: &[[usize; 3]]) -> Vec<[usize; 3]> {
        let mut v: Vec<[usize; 3]> = facets
            .iter()
            .map(|f| {
                let mut t = *f;
                t.sort_unstable();
                t
            })
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn convex_hull_3d_fast_matches_simple_facet_set() {
        // Below the n>=512 dispatch threshold `convex_hull_3d_facets` runs the
        // robust O(n²) path, so it is the oracle for the conflict-graph hull.
        // The algorithm is n-independent, so set-equality here certifies the
        // fast path that production uses at larger n.
        for &n in &[200usize, 350, 500] {
            let pts = lcg_sphere_points(n, 0x1234_5678_9abc_def0 ^ n as u64);
            let simple = convex_hull_3d_facets(&pts, [0.0, 0.0, 0.0]).expect("simple hull");
            let fast = convex_hull_3d_facets_fast(&pts).expect("fast hull");
            assert_eq!(
                facet_set(&fast),
                facet_set(&simple),
                "fast hull facet set differs from O(n^2) hull at n={n}"
            );
        }
    }

    #[test]
    fn convex_hull_3d_fast_is_complete_hull_large_n() {
        // The fast path is exercised here (n>=512). A triangulated convex hull
        // of n points in general position on a sphere has exactly 2n-4 facets
        // (Euler), and every facet must be outward-oriented w.r.t. the centroid.
        let n = 1000usize;
        let pts = lcg_sphere_points(n, 0xdead_beef_cafe_1234);
        let facets = convex_hull_3d_facets(&pts, [0.0, 0.0, 0.0]).expect("fast hull");
        assert_eq!(
            facets.len(),
            2 * n - 4,
            "hull is not a complete triangulation"
        );
        // Each undirected edge must be shared by exactly two facets (closed mesh).
        let mut edges: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for f in &facets {
            for &(a, b) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        assert!(
            edges.values().all(|&c| c == 2),
            "fast hull is not a closed 2-manifold"
        );
    }

    #[test]
    fn spherical_voronoi_rejects_wrong_radius() {
        let points = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 2.0).expect_err("radius");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_rejects_duplicate_points() {
        let points = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect_err("duplicate");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_duplicate_check_grid_path() {
        // n >= 2048 exercises the spatial-hash duplicate check (vs the small-n
        // all-pairs scan). A Fibonacci-sphere lattice gives well-separated
        // generators that must build; duplicating one must still be rejected,
        // proving the grid path agrees with the all-pairs accept/reject.
        let n = 2050usize;
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut points: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = golden * i as f64;
                [theta.cos() * r, y, theta.sin() * r]
            })
            .collect();
        assert!(points.len() >= 2048);
        assert!(SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).is_ok());
        points.push(points[0]);
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0)
            .expect_err("duplicate via grid path");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_rejects_coplanar_great_circle_points() {
        let points = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let err = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect_err("degenerate");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn spherical_voronoi_hull_euler_invariant_random() -> Result<(), SpatialError> {
        // The convex-hull face detector must produce a simplicial hull of the
        // on-sphere generators: V_voronoi == 2n - 4 (Euler, all faces tris),
        // every generator owns a region of >= 3 vertices, and every Voronoi
        // vertex sits on the unit sphere. This guards the incremental-hull
        // rewrite against the orientation-reference regression that blew the
        // facet count up (or dropped faces) for n > 4.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &n in &[8usize, 17, 40, 75] {
            let points: Vec<[f64; 3]> = (0..n)
                .map(|_| {
                    let z = 2.0 * next() - 1.0;
                    let theta = 2.0 * std::f64::consts::PI * next();
                    let r = (1.0 - z * z).sqrt();
                    [r * theta.cos(), r * theta.sin(), z]
                })
                .collect();
            let sv = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0)?;
            assert_eq!(sv.vertices.len(), 2 * n - 4, "n={n}: V != 2n-4");
            assert_eq!(sv.regions.len(), n, "n={n}: region count");
            for (pi, region) in sv.regions.iter().enumerate() {
                assert!(region.len() >= 3, "n={n}: region {pi} too small");
                for &vi in region {
                    assert!(vi < sv.vertices.len(), "n={n}: region {pi} bad index");
                }
            }
            for v in &sv.vertices {
                assert!((norm3(*v) - 1.0).abs() < 1e-9, "n={n}: vertex off sphere");
            }
        }
        Ok(())
    }

    // ── Procrustes tests ─────────────────────────────────────────────

    #[test]
    fn procrustes_identical() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = procrustes(&data, &data).expect("procrustes");
        assert!(
            result.disparity < 1e-10,
            "identical data should have 0 disparity: {}",
            result.disparity
        );
    }

    #[test]
    fn procrustes_rotated() {
        // data2 is data1 rotated 90 degrees
        let data1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let data2 = vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
        let result = procrustes(&data1, &data2).expect("procrustes");
        assert!(
            result.disparity < 0.01,
            "rotated data should align: disparity = {}",
            result.disparity
        );
    }

    #[test]
    fn procrustes_scaled() {
        // data2 is data1 scaled by 2 (should align after normalization)
        let data1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let data2: Vec<Vec<f64>> = data1
            .iter()
            .map(|p| p.iter().map(|&v| v * 2.0).collect())
            .collect();
        let result = procrustes(&data1, &data2).expect("procrustes");
        assert!(
            result.disparity < 1e-10,
            "scaled data should align: disparity = {}",
            result.disparity
        );
    }

    /// procrustes on noisy near-aligned 2-D data matches scipy
    /// (frankenscipy-u98xh). Pre-fix the rotation-only alignment
    /// (no dilation factor s = Σ singular values) drifted disparity
    /// by ~1.5e-6 abs and mtx2 by ~6.2e-4 L∞ on the 5-point fixture
    /// where data2 = data1 + small_noise.
    #[test]
    fn procrustes_noisy_near_aligned_matches_scipy() {
        let data1: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.5],
            vec![3.0, 1.5],
            vec![4.0, 3.0],
        ];
        let data2: Vec<Vec<f64>> = vec![
            vec![0.05, -0.02],
            vec![0.97, 0.04],
            vec![2.04, 0.55],
            vec![2.95, 1.47],
            vec![4.02, 3.04],
        ];
        let result = procrustes(&data1, &data2).expect("procrustes");
        // From scipy.spatial.procrustes(data1, data2):
        let expected_disparity = 0.0007857355802631467;
        let expected_mtx2 = [
            [-0.48406838, -0.25356692],
            [-0.2568692, -0.23979018],
            [0.00786914, -0.11508539],
            [0.23356811, 0.11102129],
            [0.49950032, 0.49742119],
        ];
        assert!(
            (result.disparity - expected_disparity).abs() < 1e-12,
            "disparity {} vs scipy {}",
            result.disparity,
            expected_disparity
        );
        for (i, row) in result.mtx2.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let exp = expected_mtx2[i][j];
                assert!(
                    (got - exp).abs() < 1e-7,
                    "mtx2[{i}][{j}] = {got} vs scipy {exp}"
                );
            }
        }
    }

    #[test]
    fn procrustes_rank_deficient_uses_svd_alignment() {
        // Per frankenscipy-l15l: M = mtx2^T @ mtx1 is rank-deficient here.
        // The old Newton-Schulz inverse-square-root path diverged or had to
        // return Err; the SVD path should align the one-dimensional subspaces.
        let data1 = vec![vec![1.0, 0.0], vec![-1.0, 0.0]]; // along x-axis
        let data2 = vec![vec![0.0, 1.0], vec![0.0, -1.0]]; // along y-axis
        let result = procrustes(&data1, &data2).expect("rank-deficient procrustes");
        assert!(
            result.disparity.is_finite() && result.disparity < 1e-10,
            "rank-deficient inputs should align via SVD: {}",
            result.disparity
        );
    }

    #[test]
    fn procrustes_metamorphic_translation_invariance() {
        // /testing-metamorphic for [frankenscipy-q9yi6]:
        // procrustes centers both inputs to remove the translation, so
        // procrustes(A, B).disparity must equal procrustes(A+t, B+s)
        // for any constant translation vectors t, s. This catches a
        // future bug that drops the centering step or applies it
        // incorrectly per-axis.
        let data1 = vec![
            vec![0.0_f64, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let data2 = vec![
            vec![0.5_f64, 0.3],
            vec![1.7, 0.6],
            vec![0.2, 1.4],
            vec![1.9, 2.1],
        ];
        let baseline = procrustes(&data1, &data2).expect("baseline");

        for &(t1x, t1y, t2x, t2y) in &[
            (5.0_f64, 0.0, 0.0, 0.0),
            (0.0, 0.0, -3.0, 7.0),
            (10.0, -10.0, 100.0, 100.0),
        ] {
            let shifted1: Vec<Vec<f64>> =
                data1.iter().map(|p| vec![p[0] + t1x, p[1] + t1y]).collect();
            let shifted2: Vec<Vec<f64>> =
                data2.iter().map(|p| vec![p[0] + t2x, p[1] + t2y]).collect();
            let shifted = procrustes(&shifted1, &shifted2).expect("shifted");
            assert!(
                (shifted.disparity - baseline.disparity).abs() < 1e-10,
                "translation t1=({t1x},{t1y}) t2=({t2x},{t2y}) changed \
                 disparity: baseline={} vs shifted={}",
                baseline.disparity,
                shifted.disparity
            );
        }
    }

    #[test]
    fn procrustes_metamorphic_disparity_is_symmetric() {
        // procrustes(A, B).disparity ≈ procrustes(B, A).disparity:
        // the optimal rotation R is orthogonal, so finding R that
        // minimizes ||A - BR||² is equivalent to finding R^T that
        // minimizes ||AR^T - B||². The disparity (Frobenius distance
        // after centering + scaling + rotation) must agree.
        let data1 = vec![
            vec![0.0_f64, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.5, 0.7],
        ];
        let data2 = vec![
            vec![0.0_f64, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![-0.7, 1.5],
        ];
        let ab = procrustes(&data1, &data2).expect("a->b");
        let ba = procrustes(&data2, &data1).expect("b->a");
        assert!(
            (ab.disparity - ba.disparity).abs() < 1e-10,
            "disparity asymmetry: A→B = {}, B→A = {}",
            ab.disparity,
            ba.disparity
        );
    }

    // ── Geometric SLERP tests ────────────────────────────────────────

    #[test]
    fn slerp_endpoints() {
        let start = vec![1.0, 0.0, 0.0];
        let end = vec![0.0, 1.0, 0.0];
        let result = geometric_slerp(&start, &end, &[0.0, 1.0]).expect("slerp");
        assert_eq!(result.len(), 2);
        // t=0 → start
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        // t=1 → end
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn slerp_midpoint_on_sphere() {
        let start = vec![1.0, 0.0];
        let end = vec![0.0, 1.0];
        let result = geometric_slerp(&start, &end, &[0.5]).expect("slerp");
        // Midpoint should be on unit circle
        let norm: f64 = result[0].iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "midpoint should be on unit sphere: norm = {norm}"
        );
        // Should be at 45 degrees
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (result[0][0] - expected).abs() < 1e-10,
            "slerp midpoint x: {}",
            result[0][0]
        );
    }

    #[test]
    fn slerp_dimension_mismatch() {
        let err = geometric_slerp(&[1.0, 0.0], &[1.0, 0.0, 0.0], &[0.5]).expect_err("dim");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    // ── Distance metric edge case tests ───────────────────────────

    #[test]
    fn cosine_zero_vector_returns_nan() {
        // scipy.spatial.distance.cosine returns NaN when either vector has
        // zero norm (1 - 0/0 is undefined), not 0.0.
        assert!(cosine(&[0.0, 0.0], &[1.0, 2.0]).is_nan());
        assert!(cosine(&[1.0, 2.0], &[0.0, 0.0]).is_nan());
        assert!(cosine(&[0.0, 0.0], &[0.0, 0.0]).is_nan());
    }

    #[test]
    fn correlation_constant_vector_returns_nan() {
        // scipy.spatial.distance.correlation returns NaN when either vector
        // is constant (zero variance), even if the other varies.
        assert!(correlation(&[2.0, 2.0, 2.0], &[5.0, 5.0, 5.0]).is_nan());
        assert!(correlation(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).is_nan());
        assert!(correlation(&[1.0, 2.0, 3.0], &[7.0, 7.0, 7.0]).is_nan());
    }

    #[test]
    fn euclidean_identical_points() {
        assert_eq!(euclidean(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
    }

    #[test]
    fn hamming_identical_vectors() {
        assert_eq!(hamming(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn hamming_completely_different() {
        assert!((hamming(&[1.0, 2.0], &[3.0, 4.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_single_dimension() {
        assert!((chebyshev(&[5.0], &[2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn minkowski_special_cases() {
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        // p=1 should equal cityblock
        assert!((minkowski(&a, &b, 1.0) - cityblock(&a, &b)).abs() < 1e-12);
        // p=2 should equal euclidean
        assert!((minkowski(&a, &b, 2.0) - euclidean(&a, &b)).abs() < 1e-12);
        // p=inf should equal chebyshev
        assert!((minkowski(&a, &b, f64::INFINITY) - chebyshev(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn canberra_single_zero_pair() {
        // When both elements are zero at same index, that term contributes 0
        assert_eq!(canberra(&[0.0], &[0.0]), 0.0);
    }

    // ── Hausdorff distance validation tests ─────────────────────────────

    #[test]
    fn directed_hausdorff_basic() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let d = directed_hausdorff(&xa, &xb).expect("valid input");
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn directed_hausdorff_returns_true_euclidean_after_sq_optimization() {
        // /profiling-software-performance regression for
        // [frankenscipy-ws4co]: the inner loop now compares squared
        // distances and only sqrt's the surviving max-of-mins. Verify
        // the output is still the *Euclidean* directed Hausdorff
        // distance, not the squared variant.
        //
        //   xa = {(0, 0), (3, 0)}
        //   xb = {(4, 0)}
        //   nearest distance (0,0) → (4,0) = 4
        //   nearest distance (3,0) → (4,0) = 1
        //   directed_hausdorff(xa, xb) = max(4, 1) = 4
        let xa = vec![vec![0.0_f64, 0.0], vec![3.0, 0.0]];
        let xb = vec![vec![4.0_f64, 0.0]];
        let d = directed_hausdorff(&xa, &xb).expect("valid input");
        assert!(
            (d - 4.0).abs() < 1e-12,
            "directed_hausdorff = {d}, expected 4 (true Euclidean, not 16 squared)"
        );
    }

    #[test]
    fn directed_hausdorff_symmetric_pair_pythagorean() {
        // Higher-dimension sanity: a 3-4-5 right triangle in 2D.
        //   xa = {(0, 0)}, xb = {(3, 4)} → distance = 5.
        let xa = vec![vec![0.0_f64, 0.0]];
        let xb = vec![vec![3.0_f64, 4.0]];
        let d = directed_hausdorff(&xa, &xb).expect("valid input");
        assert!(
            (d - 5.0).abs() < 1e-12,
            "directed_hausdorff = {d}, expected 5 (sqrt(9+16))"
        );
    }

    #[test]
    fn directed_hausdorff_empty_xa_rejected() {
        let err = directed_hausdorff(&[], &[vec![1.0]]).expect_err("empty xa");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn directed_hausdorff_empty_xb_rejected() {
        let err = directed_hausdorff(&[vec![1.0]], &[]).expect_err("empty xb");
        assert!(matches!(err, SpatialError::EmptyData));
    }

    #[test]
    fn directed_hausdorff_nan_rejected() {
        let xa = vec![vec![1.0, 2.0]];
        let xb = vec![vec![3.0, f64::NAN]];
        let err = directed_hausdorff(&xa, &xb).expect_err("NaN input");
        assert!(matches!(err, SpatialError::InvalidArgument(_)));
    }

    #[test]
    fn directed_hausdorff_dimension_mismatch_rejected() {
        let xa = vec![vec![1.0, 2.0]];
        let xb = vec![vec![3.0]];
        let err = directed_hausdorff(&xa, &xb).expect_err("dimension mismatch");
        assert!(matches!(err, SpatialError::DimensionMismatch { .. }));
    }

    #[test]
    fn hausdorff_distance_basic() {
        let xa = vec![vec![0.0, 0.0]];
        let xb = vec![vec![3.0, 4.0]];
        let d = hausdorff_distance(&xa, &xb).expect("valid input");
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn hausdorff_distance_symmetric() {
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 2.0]];
        let d1 = hausdorff_distance(&xa, &xb).expect("valid");
        let d2 = hausdorff_distance(&xb, &xa).expect("valid");
        assert!((d1 - d2).abs() < 1e-10);
    }

    // ── Medoid validation tests ─────────────────────────────────────────

    #[test]
    fn medoid_basic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.0]];
        // Point 2 (0.5, 0) minimizes total distance
        let idx = medoid(&points).expect("non-empty");
        assert_eq!(idx, 2);
    }

    #[test]
    fn medoid_empty_returns_none() {
        let points: Vec<Vec<f64>> = vec![];
        assert!(medoid(&points).is_none());
    }

    #[test]
    fn medoid_single_point() {
        let points = vec![vec![5.0, 5.0]];
        let idx = medoid(&points).expect("single point");
        assert_eq!(idx, 0);
    }

    // ── Nearest neighbors validation tests ──────────────────────────────

    #[test]
    fn nearest_neighbors_basic() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]];
        let (indices, distances) = nearest_neighbors(&data);
        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
        // Point 0's nearest is point 1
        assert_eq!(indices[0], Some(1));
        assert!((distances[0] - 1.0).abs() < 1e-10);
        // Point 1's nearest is point 0
        assert_eq!(indices[1], Some(0));
        // Point 2's nearest is point 1
        assert_eq!(indices[2], Some(1));
    }

    #[test]
    fn nearest_neighbors_empty() {
        let (indices, distances) = nearest_neighbors(&[]);
        assert!(indices.is_empty());
        assert!(distances.is_empty());
    }

    #[test]
    fn nearest_neighbors_single_point() {
        let data = vec![vec![5.0, 5.0]];
        let (indices, distances) = nearest_neighbors(&data);
        assert_eq!(indices.len(), 1);
        assert!(indices[0].is_none());
        assert_eq!(distances[0], f64::INFINITY);
    }

    #[test]
    fn nearest_neighbors_kdtree_matches_brute_force_bitwise() {
        // The KD-tree fast path must reproduce the O(n^2) brute force exactly,
        // including lowest-index tie-breaking and distance bits. Grid-snapped
        // coordinates create duplicate / equidistant points that exercise ties.
        fn brute(data: &[Vec<f64>]) -> (Vec<Option<usize>>, Vec<f64>) {
            let n = data.len();
            let mut idx = Vec::with_capacity(n);
            let mut dist = Vec::with_capacity(n);
            for i in 0..n {
                let mut md = f64::INFINITY;
                let mut mi: Option<usize> = None;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let d = euclidean(&data[i], &data[j]);
                    if d < md {
                        md = d;
                        mi = Some(j);
                    }
                }
                idx.push(mi);
                dist.push(md);
            }
            (idx, dist)
        }
        let mut state: u64 = 0xabcd_1234_5678_9f01;
        let mut next = |g: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (state >> 11) as f64 / (1u64 << 53) as f64;
            if g == 0 { v } else { (v * g as f64).floor() }
        };
        for &n in &[1usize, 2, 9, 64, 200] {
            for &dim in &[1usize, 2, 3] {
                for &grid in &[0u64, 4] {
                    let data: Vec<Vec<f64>> = (0..n)
                        .map(|_| (0..dim).map(|_| next(grid)).collect())
                        .collect();
                    let (gi, gd) = nearest_neighbors(&data);
                    let (wi, wd) = brute(&data);
                    assert_eq!(gi, wi, "index mismatch n={n} dim={dim} grid={grid}");
                    for (a, b) in gd.iter().zip(&wd) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "dist mismatch n={n} dim={dim} grid={grid}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn k_nearest_neighbors_kdtree_matches_brute_force_bitwise() {
        // The KD-tree fast path must reproduce the O(n^2) brute force exactly for
        // every k: same k indices in the same composite order and identical
        // distance bits, including tie-heavy grid-snapped data.
        fn brute(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
            let n = data.len();
            let cmp = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
            let mut idx = Vec::new();
            let mut dist = Vec::new();
            for i in 0..n {
                let mut d: Vec<(usize, f64)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, euclidean(&data[i], &data[j])))
                    .collect();
                let ka = k.min(d.len());
                if ka < d.len() {
                    d.select_nth_unstable_by(ka, cmp);
                }
                d[..ka].sort_by(cmp);
                idx.push(d[..ka].iter().map(|&(j, _)| j).collect());
                dist.push(d[..ka].iter().map(|&(_, v)| v).collect());
            }
            (idx, dist)
        }
        let mut state: u64 = 0x51ed_2718_2845_9045;
        let mut next = |g: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (state >> 11) as f64 / (1u64 << 53) as f64;
            if g == 0 { v } else { (v * g as f64).floor() }
        };
        for &n in &[1usize, 2, 9, 64, 150] {
            for &dim in &[1usize, 2, 3] {
                for &k in &[1usize, 3, 8, 1000] {
                    for &grid in &[0u64, 4] {
                        let data: Vec<Vec<f64>> = (0..n)
                            .map(|_| (0..dim).map(|_| next(grid)).collect())
                            .collect();
                        let (gi, gd) = k_nearest_neighbors(&data, k);
                        let (wi, wd) = brute(&data, k);
                        assert_eq!(gi, wi, "index mismatch n={n} dim={dim} k={k} grid={grid}");
                        for (a, b) in gd.iter().flatten().zip(wd.iter().flatten()) {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "dist mismatch n={n} dim={dim} k={k} grid={grid}"
                            );
                        }
                    }
                }
            }
        }
    }

    // ── Rotation tests ────────────────────────────────────────────────────

    #[test]
    fn rotation_identity() {
        let r = Rotation::identity();
        let v = [1.0, 2.0, 3.0];
        let rv = r.apply(v);
        assert!((rv[0] - v[0]).abs() < 1e-12);
        assert!((rv[1] - v[1]).abs() < 1e-12);
        assert!((rv[2] - v[2]).abs() < 1e-12);
    }

    #[test]
    fn rotation_from_quat_normalized() {
        let r = Rotation::from_quat([0.0, 0.0, 2.0, 2.0]);
        let q = r.as_quat();
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_90_deg_z_axis() {
        use std::f64::consts::FRAC_PI_2;
        let r = Rotation::from_rotvec([0.0, 0.0, FRAC_PI_2]);
        let v = [1.0, 0.0, 0.0];
        let rv = r.apply(v);
        assert!((rv[0]).abs() < 1e-12);
        assert!((rv[1] - 1.0).abs() < 1e-12);
        assert!((rv[2]).abs() < 1e-12);
    }

    #[test]
    fn rotation_apply_many_matches_apply() {
        // apply_many (matrix precomputed once) must be byte-identical to per-vector apply.
        let r = Rotation::from_quat([
            0.022260026714733816,
            0.43967973954090955,
            0.3604234056503559,
            0.8223631719059994,
        ]);
        let pts = [[0.5, -1.0, 2.0], [1.0, 0.0, 0.0], [-3.0, 2.5, -0.25]];
        let batch = r.apply_many(&pts);
        for (p, b) in pts.iter().zip(batch.iter()) {
            assert_eq!(*b, r.apply(*p), "apply_many != apply at {p:?}");
        }

        // Large input crosses the parallel threshold (1.4M); the multi-threaded
        // chunked path must stay byte-identical to per-vector apply.
        let n = 1_500_000usize;
        let mut s: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15 - 0.5
        };
        let big: Vec<[f64; 3]> = (0..n).map(|_| [next(), next(), next()]).collect();
        let big_batch = r.apply_many(&big);
        assert_eq!(big_batch.len(), n);
        // Spot-check a spread of indices against the scalar reference.
        for &i in &[0usize, 1, 999_999, 1_400_001, n - 1] {
            assert_eq!(
                big_batch[i],
                r.apply(big[i]),
                "parallel apply_many != apply at index {i}"
            );
        }
    }

    #[test]
    fn rotations_batch_many_match_scalar_loop_bit_for_bit() {
        // rotations_multiply_many / rotations_from_matrix_many must be byte-identical
        // to the scalar per-element loop. n crosses ROTATION_BATCH_GATE (1<<15).
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15 * 2.0 - 1.0
        };
        let mkq = |u: &mut dyn FnMut() -> f64| {
            let mut q = [u(), u(), u(), u()];
            let nn = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
            for v in &mut q {
                *v /= nn;
            }
            Rotation::from_quat(q)
        };
        let n = 40_000usize; // above the gate
        let a: Vec<Rotation> = (0..n).map(|_| mkq(&mut u)).collect();
        let b: Vec<Rotation> = (0..n).map(|_| mkq(&mut u)).collect();
        let mats: Vec<[[f64; 3]; 3]> = a.iter().map(Rotation::as_matrix).collect();

        let angs: Vec<[f64; 3]> = (0..n).map(|_| [u() * 3.0, u() * 3.0, u() * 3.0]).collect();
        let mul = rotations_multiply_many(&a, &b);
        let fm = rotations_from_matrix_many(&mats);
        let eu = rotations_as_euler_many(&a, "xyz");
        let ez = rotations_as_euler_many(&a, "ZYX");
        let fe = rotations_from_euler_many("xyz", &angs);
        let rv = rotations_as_rotvec_many(&a);
        assert_eq!(mul.len(), n);
        assert_eq!(fm.len(), n);
        for i in 0..n {
            assert_eq!(mul[i].as_quat(), a[i].multiply(&b[i]).as_quat());
            assert_eq!(fm[i].as_quat(), Rotation::from_matrix(mats[i]).as_quat());
            assert_eq!(eu[i], a[i].as_euler("xyz"));
            assert_eq!(ez[i], a[i].as_euler("ZYX"));
            assert_eq!(
                fe[i].as_quat(),
                Rotation::from_euler("xyz", angs[i]).as_quat()
            );
            assert_eq!(rv[i], a[i].as_rotvec());
            // from_matrix round-trips: recovered rotation reproduces the matrix.
            let rec = fm[i].as_matrix();
            for r in 0..3 {
                for c in 0..3 {
                    assert!((rec[r][c] - mats[i][r][c]).abs() < 1e-12);
                }
            }
        }
        assert!(rotations_multiply_many(&[], &[]).is_empty());
        // Below-gate stays serial and byte-identical.
        let small_a = &a[..8];
        let small_b = &b[..8];
        let sm = rotations_multiply_many(small_a, small_b);
        for i in 0..8 {
            assert_eq!(sm[i].as_quat(), small_a[i].multiply(&small_b[i]).as_quat());
        }
    }

    #[test]
    fn slerp_matches_scipy_and_batch_is_bit_identical() {
        // Reference values from scipy.spatial.transform.Slerp (v1.17.1).
        let mk = |q: [f64; 4]| {
            let n = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
            Rotation::from_quat([q[0] / n, q[1] / n, q[2] / n, q[3] / n])
        };
        let kt = [0.0, 1.0, 3.0, 6.0];
        let kr = [
            mk([0.0, 0.0, 0.0, 1.0]),
            mk([0.1, 0.2, 0.3, 0.9]),
            mk([0.5, -0.1, 0.2, 0.8]),
            mk([-0.2, 0.3, 0.1, 0.7]),
        ];
        let sl = Slerp::new(&kt, &kr).expect("slerp");
        let cases: [(f64, [f64; 4]); 6] = [
            (0.0, [0.0, 0.0, 0.0, 1.0]),
            (
                0.5,
                [
                    0.05231070742733861,
                    0.10462141485467721,
                    0.1569321222820158,
                    0.9806580741717876,
                ],
            ),
            (
                1.0,
                [
                    0.10259783520851541,
                    0.20519567041703082,
                    0.3077935056255462,
                    0.9233805168766387,
                ],
            ),
            (
                2.0,
                [
                    0.32082346922827804,
                    0.05295281370689373,
                    0.2667409875261423,
                    0.9072582579874562,
                ],
            ),
            (
                4.5,
                [
                    0.14813911965010984,
                    0.15436725350922376,
                    0.18663685531798327,
                    0.9588494337018318,
                ],
            ),
            (
                6.0,
                [
                    -0.2519763153394849,
                    0.37796447300922736,
                    0.12598815766974245,
                    0.8819171036881969,
                ],
            ),
        ];
        for (t, want) in cases {
            let q = sl.eval(t).expect("eval").as_quat();
            // Quaternion sign is arbitrary; compare up to sign.
            let d_pos: f64 = (0..4).map(|i| (q[i] - want[i]).abs()).fold(0.0, f64::max);
            let d_neg: f64 = (0..4).map(|i| (q[i] + want[i]).abs()).fold(0.0, f64::max);
            assert!(
                d_pos.min(d_neg) < 1e-12,
                "slerp t={t}: {q:?} vs scipy {want:?}"
            );
        }
        // Endpoints exactly recover the key rotations (continuity).
        assert!(
            sl.eval(3.0)
                .unwrap()
                .multiply(&kr[2].inv())
                .is_identity(1e-12)
        );

        // Batch is bit-identical to the serial eval loop (n crosses the gate).
        let mut s: u64 = 0xABCD_1234_5678_9EF1;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15
        };
        let ts: Vec<f64> = (0..40_000).map(|_| 6.0 * u()).collect();
        let batch = sl.eval_many(&ts).expect("eval_many");
        assert_eq!(batch.len(), ts.len());
        for (t, r) in ts.iter().zip(&batch) {
            assert_eq!(r.as_quat(), sl.eval(*t).unwrap().as_quat());
        }
        // Out-of-range times error (matches scipy).
        assert!(sl.eval(-0.1).is_err());
        assert!(sl.eval(6.1).is_err());
        assert!(sl.eval_many(&[1.0, 7.0]).is_err());
        assert!(Slerp::new(&[0.0], &[kr[0]]).is_err());
    }

    #[test]
    fn rigid_transforms_compose_many_matches_scalar_loop_bit_for_bit() {
        // rigid_transforms_compose_many must be byte-identical to the scalar
        // per-element compose loop. n crosses ROTATION_BATCH_GATE.
        let mut s: u64 = 0xD1B5_4A32_D192_ED03;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / 9.007_199_254_740_992e15 * 2.0 - 1.0
        };
        let mk = |u: &mut dyn FnMut() -> f64| {
            let mut q = [u(), u(), u(), u()];
            let nn = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
            for v in &mut q {
                *v /= nn;
            }
            RigidTransform::from_components([u(), u(), u()], Rotation::from_quat(q))
        };
        let n = 40_000usize;
        let a: Vec<RigidTransform> = (0..n).map(|_| mk(&mut u)).collect();
        let b: Vec<RigidTransform> = (0..n).map(|_| mk(&mut u)).collect();
        let c = rigid_transforms_compose_many(&a, &b);
        assert_eq!(c.len(), n);
        for i in 0..n {
            assert_eq!(c[i].as_matrix(), a[i].compose(&b[i]).as_matrix());
        }
        assert!(rigid_transforms_compose_many(&[], &[]).is_empty());
    }

    #[test]
    fn rotation_matrix_roundtrip() {
        use std::f64::consts::FRAC_PI_4;
        let r1 = Rotation::from_rotvec([FRAC_PI_4, 0.0, 0.0]);
        let m = r1.as_matrix();
        let r2 = Rotation::from_matrix(m);
        let v = [1.0, 1.0, 1.0];
        let rv1 = r1.apply(v);
        let rv2 = r2.apply(v);
        assert!((rv1[0] - rv2[0]).abs() < 1e-10);
        assert!((rv1[1] - rv2[1]).abs() < 1e-10);
        assert!((rv1[2] - rv2[2]).abs() < 1e-10);
    }

    #[test]
    fn rotation_as_euler_matches_scipy() {
        // Reference values from scipy.spatial.transform.Rotation (v1.17.1).
        // A generic (non-gimbal) rotation exercised across extrinsic + intrinsic,
        // asymmetric (Tait-Bryan) + symmetric (proper-Euler) sequences.
        let r = Rotation::from_matrix([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]]);
        let close = |g: [f64; 3], e: [f64; 3], seq: &str| {
            for a in 0..3 {
                assert!(
                    (g[a] - e[a]).abs() < 1e-12,
                    "as_euler({seq})[{a}] = {} vs scipy {}",
                    g[a],
                    e[a]
                );
            }
        };
        // scipy: Rotation.from_matrix(m).as_euler(seq)
        close(
            r.as_euler("xyz"),
            [0.817645045832702, -0.500654712404588, -1.147942400661956],
            "xyz",
        );
        close(
            r.as_euler("ZYX"),
            [-1.1479424006619559, -0.5006547124045881, 0.8176450458327023],
            "ZYX",
        );
        // Symmetric sequence from a known quaternion (scipy reference).
        let mut qa = [0.1, 0.2, 0.3, 0.9];
        let na = (qa.iter().map(|v| v * v).sum::<f64>()).sqrt();
        for v in &mut qa {
            *v /= na;
        }
        let ra = Rotation::from_quat(qa);
        close(
            ra.as_euler("zyx"),
            [0.627070662589018, 0.457944420467095, 0.070471344578796],
            "zyx",
        );
        close(
            ra.as_euler("ZXZ"),
            [1.4288992721907325, 0.46295472794035675, -0.7853981633974483],
            "ZXZ",
        );
    }

    #[test]
    fn rotation_inverse() {
        use std::f64::consts::FRAC_PI_3;
        let r = Rotation::from_rotvec([0.0, FRAC_PI_3, 0.0]);
        let r_inv = r.inv();
        let composed = r.multiply(&r_inv);
        assert!(composed.is_identity(1e-10));
    }

    #[test]
    fn rotation_multiply() {
        use std::f64::consts::FRAC_PI_2;
        let r1 = Rotation::from_rotvec([0.0, 0.0, FRAC_PI_2]);
        let r2 = Rotation::from_rotvec([0.0, 0.0, FRAC_PI_2]);
        let composed = r1 * r2;
        let v = [1.0, 0.0, 0.0];
        let rv = composed.apply(v);
        assert!((rv[0] + 1.0).abs() < 1e-10);
        assert!((rv[1]).abs() < 1e-10);
    }

    #[test]
    fn rotation_euler_roundtrip() {
        let angles = [0.1, 0.2, 0.3];
        let r = Rotation::from_euler("xyz", angles);
        let v = [1.0, 0.0, 0.0];
        let rv = r.apply(v);
        let m = r.as_matrix();
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        assert!((det - 1.0).abs() < 1e-10);
        assert!((rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_magnitude() {
        use std::f64::consts::PI;
        let r = Rotation::from_rotvec([0.0, PI / 3.0, 0.0]);
        assert!((r.magnitude() - PI / 3.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_slerp_endpoints() {
        let r1 = Rotation::from_rotvec([0.1, 0.0, 0.0]);
        let r2 = Rotation::from_rotvec([0.0, 0.5, 0.0]);
        let s0 = r1.slerp(&r2, 0.0);
        let s1 = r1.slerp(&r2, 1.0);

        let v = [1.0, 0.0, 0.0];
        let vs0 = s0.apply(v);
        let vr1 = r1.apply(v);
        assert!((vs0[0] - vr1[0]).abs() < 1e-10);
        assert!((vs0[1] - vr1[1]).abs() < 1e-10);

        let vs1 = s1.apply(v);
        let vr2 = r2.apply(v);
        assert!((vs1[0] - vr2[0]).abs() < 1e-10);
        assert!((vs1[1] - vr2[1]).abs() < 1e-10);
    }

    #[test]
    fn jensenshannon_metamorphic_symmetry_under_swap() {
        // JSD is symmetric: jensenshannon(p, q) == jensenshannon(q, p).
        let p = [0.5_f64, 0.3, 0.2];
        let q = [0.1_f64, 0.7, 0.2];
        let d_pq = jensenshannon(&p, &q, None);
        let d_qp = jensenshannon(&q, &p, None);
        assert!(
            (d_pq - d_qp).abs() < 1e-15,
            "symmetry: d(p,q)={d_pq}, d(q,p)={d_qp}"
        );
    }

    #[test]
    fn jensenshannon_metamorphic_disjoint_natural_log_bound() {
        // For maximally-disjoint distributions [1, 0] vs [0, 1], the JSD
        // in nats is ln(2), so the distance √JSD = √ln(2) ≈ 0.832.
        let d = jensenshannon(&[1.0, 0.0], &[0.0, 1.0], None);
        let bound = (2.0_f64.ln()).sqrt();
        assert!(
            (d - bound).abs() < 1e-12,
            "disjoint √ln(2): got {d}, expected {bound}"
        );
    }

    #[test]
    fn pdist_euclidean_matches_scipy_reference_values() {
        // scipy.spatial.distance.pdist([[0,0],[1,0],[0,1],[1,1]], 'euclidean')
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let result = pdist(&x, DistanceMetric::Euclidean).expect("pdist");
        let expected = [
            1.0,
            1.0,
            std::f64::consts::SQRT_2,
            std::f64::consts::SQRT_2,
            1.0,
            1.0,
        ];
        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - expected[i]).abs() < 1e-10,
                "pdist[{i}] got {val}, expected {}",
                expected[i]
            );
        }
    }

    #[test]
    fn cdist_euclidean_matches_scipy_reference_values() {
        // scipy.spatial.distance.cdist([[0,0],[1,0]], [[0,1],[1,1]], 'euclidean')
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let result = cdist(&xa, &xb).expect("cdist");
        let expected = [
            [1.0, std::f64::consts::SQRT_2],
            [std::f64::consts::SQRT_2, 1.0],
        ];
        for (i, row) in result.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (*val - expected[i][j]).abs() < 1e-10,
                    "cdist[{i}][{j}] got {val}, expected {}",
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn euclidean_matches_scipy_reference_values() {
        // scipy.spatial.distance.euclidean([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = euclidean(&a, &b);
        assert!(
            (result - 5.196152422706632).abs() < 1e-10,
            "euclidean got {result}, expected 5.196152422706632"
        );
    }

    #[test]
    fn cityblock_matches_scipy_reference_values() {
        // scipy.spatial.distance.cityblock([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = cityblock(&a, &b);
        assert!(
            (result - 9.0).abs() < 1e-10,
            "cityblock got {result}, expected 9.0"
        );
    }

    #[test]
    fn chebyshev_matches_scipy_reference_values() {
        // scipy.spatial.distance.chebyshev([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = chebyshev(&a, &b);
        assert!(
            (result - 3.0).abs() < 1e-10,
            "chebyshev got {result}, expected 3.0"
        );
    }

    #[test]
    fn minkowski_matches_scipy_reference_values() {
        // scipy.spatial.distance.minkowski([1,2,3], [4,5,6], 3)
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = minkowski(&a, &b, 3.0);
        assert!(
            (result - 4.3267487109222245).abs() < 1e-10,
            "minkowski got {result}, expected 4.3267487109222245"
        );
    }

    #[test]
    fn minkowski_distance_batched_matches_scipy() {
        // scipy.spatial.minkowski_distance_p / minkowski_distance over row arrays.
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![vec![4.0, 0.0, 3.0]];
        // mdp p2 = 13.0, md p2 = 3.605551275463989
        let mdp = minkowski_distance_p(&x, &y, 2.0).unwrap();
        assert!((mdp[0] - 13.0).abs() < 1e-12);
        let md = minkowski_distance(&x, &y, 2.0).unwrap();
        assert!((md[0] - 3.605551275463989).abs() < 1e-12);
        // p == 1 and p == inf: distance == _p (no root).
        assert!((minkowski_distance(&x, &y, 1.0).unwrap()[0] - 5.0).abs() < 1e-12);
        assert!((minkowski_distance(&x, &y, f64::INFINITY).unwrap()[0] - 3.0).abs() < 1e-12);
        assert!((minkowski_distance_p(&x, &y, f64::INFINITY).unwrap()[0] - 3.0).abs() < 1e-12);

        // 2D row-wise: scipy.spatial.minkowski_distance(X, Y, 2) -> [2.236.., 3.605..]
        let xx = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let yy = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let md2 = minkowski_distance(&xx, &yy, 2.0).unwrap();
        assert!((md2[0] - 2.23606797749979).abs() < 1e-12);
        assert!((md2[1] - 3.605551275463989).abs() < 1e-12);
        // mdp p3 -> [9.0, 35.0]
        let mdp3 = minkowski_distance_p(&xx, &yy, 3.0).unwrap();
        assert!((mdp3[0] - 9.0).abs() < 1e-12 && (mdp3[1] - 35.0).abs() < 1e-12);

        // Single-row broadcast against many rows.
        let one = vec![vec![0.0, 0.0]];
        let bc = minkowski_distance(&one, &xx, 2.0).unwrap();
        assert_eq!(bc.len(), 2);
        assert!((bc[0] - 2.23606797749979).abs() < 1e-12);
        assert!((bc[1] - 5.0).abs() < 1e-12);

        // Incompatible row counts are rejected.
        assert!(minkowski_distance(&xx, &one, 2.0).is_ok()); // 2 vs 1 broadcasts
        let three = vec![vec![0.0, 0.0]; 3];
        assert!(minkowski_distance(&xx, &three, 2.0).is_err()); // 2 vs 3
    }

    #[test]
    fn cosine_matches_scipy_reference_values() {
        // scipy.spatial.distance.cosine([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = cosine(&a, &b);
        assert!(
            (result - 0.025368153802923787).abs() < 1e-10,
            "cosine got {result}, expected 0.025368153802923787"
        );
    }

    #[test]
    fn braycurtis_matches_scipy_reference_values() {
        // scipy.spatial.distance.braycurtis([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = braycurtis(&a, &b);
        assert!(
            (result - 0.42857142857142855).abs() < 1e-10,
            "braycurtis got {result}, expected 0.42857142857142855"
        );
    }

    #[test]
    fn canberra_matches_scipy_reference_values() {
        // scipy.spatial.distance.canberra([1,2,3], [4,5,6])
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = canberra(&a, &b);
        assert!(
            (result - 1.3619047619047617).abs() < 1e-10,
            "canberra got {result}, expected 1.3619047619047617"
        );
    }

    #[test]
    fn cdist_mahalanobis_matches_per_pair() {
        // The GEMM-expansion batch path must agree with the direct per-pair
        // `mahalanobis` to within rounding, for every (i, j).
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        let d = 12usize;
        let na = 23usize;
        let nb = 17usize;
        let xa: Vec<Vec<f64>> = (0..na)
            .map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect())
            .collect();
        let xb: Vec<Vec<f64>> = (0..nb)
            .map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect())
            .collect();
        // SPD inverse-covariance VI = M·Mᵀ/d + I.
        let m: Vec<Vec<f64>> = (0..d)
            .map(|_| (0..d).map(|_| rng() - 0.5).collect())
            .collect();
        let mut vi = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d).map(|k| m[i][k] * m[j][k]).sum();
                vi[i][j] = dot / d as f64 + if i == j { 1.0 } else { 0.0 };
            }
        }
        let got = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist_mahalanobis");
        assert_eq!(got.len(), na);
        for (i, row) in got.iter().enumerate() {
            assert_eq!(row.len(), nb);
            for (j, &g) in row.iter().enumerate() {
                let want = mahalanobis(&xa[i], &xb[j], &vi);
                assert!(
                    (g - want).abs() < 1e-9,
                    "cdist_mahalanobis[{i}][{j}] = {g}, per-pair = {want}"
                );
            }
        }
    }

    #[test]
    fn cdist_mahalanobis_matches_scipy_reference() {
        // scipy.spatial.distance.cdist([[0,0],[1,1]], [[0,2],[2,0]], 'mahalanobis',
        //   VI=[[1,0],[0,1]]) == [[2, 2], [sqrt(2), sqrt(2)]]
        let xa = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let xb = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let vi = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let got = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist_mahalanobis");
        let want = [[2.0, 2.0], [2.0_f64.sqrt(), 2.0_f64.sqrt()]];
        for (i, row) in got.iter().enumerate() {
            for (j, &g) in row.iter().enumerate() {
                assert!(
                    (g - want[i][j]).abs() < 1e-10,
                    "cdist_mahalanobis[{i}][{j}] = {g}, expected {}",
                    want[i][j]
                );
            }
        }
    }

    #[test]
    fn cdist_mahalanobis_rejects_bad_dims() {
        let xa = vec![vec![0.0, 1.0]];
        let xb = vec![vec![0.0, 1.0]];
        let vi_bad = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]; // 2x3, not 2x2
        assert!(cdist_mahalanobis(&xa, &xb, &vi_bad).is_err());
    }

    #[test]
    fn pdist_mahalanobis_matches_per_pair_condensed() {
        // The condensed batch path must equal the direct per-pair `mahalanobis` in
        // scipy's upper-triangle row-major order.
        let mut s: u64 = 0xfeed_face_cafe_babe;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        let d = 9usize;
        let n = 19usize;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect())
            .collect();
        let m: Vec<Vec<f64>> = (0..d)
            .map(|_| (0..d).map(|_| rng() - 0.5).collect())
            .collect();
        let mut vi = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d).map(|k| m[i][k] * m[j][k]).sum();
                vi[i][j] = dot / d as f64 + if i == j { 1.0 } else { 0.0 };
            }
        }
        let got = pdist_mahalanobis(&x, &vi).expect("pdist_mahalanobis");
        assert_eq!(got.len(), n * (n - 1) / 2);
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let want = mahalanobis(&x[i], &x[j], &vi);
                assert!(
                    (got[k] - want).abs() < 1e-9,
                    "pdist_mahalanobis[{k}] (i={i},j={j}) = {}, per-pair = {want}",
                    got[k]
                );
                k += 1;
            }
        }
    }

    #[test]
    fn squareform_condensed_to_matrix_matches_scipy_reference_values() {
        // scipy.spatial.distance.squareform([1, 2, 3])
        // -> array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        let condensed = vec![1.0, 2.0, 3.0];
        let result = squareform_to_matrix(&condensed).expect("squareform should succeed");
        let expected = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]];
        for (i, row) in result.iter().enumerate() {
            for (j, &got) in row.iter().enumerate() {
                let want = expected[i][j];
                assert!(
                    (got - want).abs() < 1e-10,
                    "matrix[{i}][{j}] got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn squareform_matrix_to_condensed_matches_scipy_reference_values() {
        // scipy.spatial.distance.squareform([[0,1,2],[1,0,3],[2,3,0]])
        // -> array([1., 2., 3.])
        let matrix = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ];
        let result = squareform_to_condensed(&matrix).expect("squareform should succeed");
        let expected = [1.0, 2.0, 3.0];
        for (i, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "condensed[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn directed_hausdorff_matches_scipy_reference_values() {
        // scipy.spatial.distance.directed_hausdorff([[0,0],[1,0]], [[0,1],[1,1]])[0]
        // -> 1.0
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let result = directed_hausdorff(&xa, &xb).expect("directed_hausdorff should succeed");
        assert!(
            (result - 1.0).abs() < 1e-10,
            "directed_hausdorff got {result}, expected 1.0"
        );
    }

    #[test]
    fn hausdorff_distance_matches_scipy_reference_values() {
        // max of directed_hausdorff in both directions
        // scipy.spatial.distance.directed_hausdorff gives 1.0 both ways here
        let xa = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let xb = vec![vec![0.0, 1.0], vec![1.0, 1.0]];
        let result = hausdorff_distance(&xa, &xb).expect("hausdorff_distance should succeed");
        assert!(
            (result - 1.0).abs() < 1e-10,
            "hausdorff_distance got {result}, expected 1.0"
        );
    }

    #[test]
    fn mahalanobis_matches_scipy_reference_values() {
        // scipy.spatial.distance.mahalanobis([0, 2], [0, 1], [[1, 0], [0, 1]])
        // -> 1.0
        let x = [0.0, 2.0];
        let y = [0.0, 1.0];
        let vi = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = mahalanobis(&x, &y, &vi);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "mahalanobis got {result}, expected 1.0"
        );
    }

    #[test]
    fn seuclidean_matches_scipy_reference_values() {
        // scipy.spatial.distance.seuclidean([1, 2], [3, 4], [1, 1])
        // -> sqrt((1-3)^2/1 + (2-4)^2/1) = sqrt(4 + 4) = sqrt(8) = 2.8284271247461903
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        let v = [1.0, 1.0];
        let result = seuclidean(&x, &y, &v);
        assert!(
            (result - 2.8284271247461903).abs() < 1e-10,
            "seuclidean got {result}, expected 2.8284271247461903"
        );
    }

    #[test]
    fn sqeuclidean_matches_scipy_reference_values() {
        // scipy.spatial.distance.sqeuclidean([1, 2], [3, 4])
        // -> (1-3)^2 + (2-4)^2 = 4 + 4 = 8
        let result = sqeuclidean(&[1.0, 2.0], &[3.0, 4.0]);
        assert!(
            (result - 8.0).abs() < 1e-10,
            "sqeuclidean got {result}, expected 8.0"
        );
    }

    #[test]
    fn correlation_matches_scipy_reference_values() {
        // scipy.spatial.distance.correlation([1, 2, 3], [1, 2, 3])
        // -> 0.0 (identical vectors have zero correlation distance)
        let result = correlation(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(
            result.abs() < 1e-10,
            "correlation of identical vectors got {result}, expected 0.0"
        );

        // scipy.spatial.distance.correlation([1, 2, 3], [3, 2, 1])
        // -> 2.0 (perfectly anti-correlated)
        let result2 = correlation(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]);
        assert!(
            (result2 - 2.0).abs() < 1e-10,
            "correlation of anti-correlated vectors got {result2}, expected 2.0"
        );
    }

    #[test]
    fn hamming_matches_scipy_reference_values() {
        // scipy.spatial.distance.hamming([1, 0, 0], [0, 1, 0])
        // -> 2/3 = 0.6666... (2 out of 3 elements differ)
        let result = hamming(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        let expected = 2.0 / 3.0;
        assert!(
            (result - expected).abs() < 1e-10,
            "hamming got {result}, expected {expected}"
        );
    }

    #[test]
    fn jaccard_matches_scipy_reference_values() {
        // scipy.spatial.distance.jaccard([1, 0, 0], [1, 1, 0])
        // -> 1 - |{1,0,0} ∩ {1,1,0}| / |{1,0,0} ∪ {1,1,0}|
        // intersection at index 0 (both 1), union at indices 0,1 (any 1)
        // For boolean: 1 - 1/2 = 0.5
        let result = jaccard(&[1.0, 0.0, 0.0], &[1.0, 1.0, 0.0]);
        assert!(
            (result - 0.5).abs() < 1e-10,
            "jaccard got {result}, expected 0.5"
        );
    }

    #[test]
    fn kdtree_query_matches_scipy_reference_values() {
        // scipy.spatial.KDTree([[0,0], [1,1], [2,2]]).query([0.5, 0.5])
        // -> (dist=0.7071..., idx=0)  (closest to [0,0] or [1,1])
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let tree = KDTree::new(&points).expect("kdtree");
        let (idx, dist) = tree.query(&[0.5, 0.5]).expect("query");
        let expected_dist = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (dist - expected_dist).abs() < 1e-10,
            "kdtree dist got {dist}, expected {expected_dist}"
        );
        // Index should be 0 or 1 (both are equidistant from [0.5, 0.5])
        assert!(
            idx == 0 || idx == 1,
            "kdtree idx got {idx}, expected 0 or 1"
        );
    }

    #[test]
    fn convex_hull_area_matches_scipy_reference_values() {
        // scipy.spatial.ConvexHull([[0,0], [1,0], [1,1], [0,1], [0.5,0.5]]).volume
        // -> 1.0 (area of unit square)
        let points: Vec<(f64, f64)> =
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)];
        let hull = ConvexHull::new(&points).expect("convex_hull");
        // Hull should be the 4 corner vertices
        assert_eq!(hull.vertices.len(), 4, "hull should have 4 vertices");
        // Area should be 1.0
        assert!(
            (hull.area - 1.0).abs() < 1e-10,
            "hull area got {}, expected 1.0",
            hull.area
        );
    }

    #[test]
    fn delaunay_triangulation_matches_scipy_reference_values() {
        // scipy.spatial.Delaunay([[0,0], [1,0], [1,1], [0,1]]).simplices
        // -> 2 triangles covering the square
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tri = Delaunay::new(&points).expect("delaunay");
        // Should have exactly 2 triangles
        assert_eq!(tri.simplices.len(), 2, "should have 2 triangles");
    }

    #[test]
    fn delaunay_empty_circumcircle_property_random() {
        // Delaunay safety net (frankenscipy-9l5oo): the defining property is that no
        // input point lies strictly inside any triangle's circumcircle. This validates
        // the O(n²) Bowyer-Watson on a non-degenerate scattered point set AND is the
        // invariant any future O(n log n) point-location rewrite must preserve — the
        // existing test only covers 4 points / 2 triangles. Deterministic low-
        // discrepancy (golden-ratio) points avoid exact cocircular degeneracies.
        let n = 200usize;
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let t = i as f64;
                (
                    (t * 0.618_033_988_75).fract() * 100.0,
                    (t * 0.414_213_562_37).fract() * 100.0,
                )
            })
            .collect();
        let tri = Delaunay::new(&pts).expect("delaunay");
        assert!(!tri.simplices.is_empty(), "should produce triangles");
        for &(a, b, c) in &tri.simplices {
            let (pa, pb, pc) = (pts[a], pts[b], pts[c]);
            for (k, &p) in pts.iter().enumerate() {
                if k == a || k == b || k == c {
                    continue;
                }
                assert!(
                    !point_in_circumcircle(pa, pb, pc, p),
                    "Delaunay property violated: point {k} inside circumcircle of \
                     triangle ({a},{b},{c})"
                );
            }
        }
    }

    #[test]
    fn voronoi_vertices_matches_scipy_reference_values() {
        // scipy.spatial.Voronoi([[0,0], [1,0], [1,1], [0,1]]).vertices
        // -> [[0.5, 0.5]] (center of square)
        let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let vor = Voronoi::new(&points).expect("voronoi");
        // Should have 1 finite vertex at the center
        assert!(!vor.vertices.is_empty(), "should have at least 1 vertex");
        // Check that center vertex exists
        let has_center = vor
            .vertices
            .iter()
            .any(|&(x, y)| (x - 0.5).abs() < 1e-10 && (y - 0.5).abs() < 1e-10);
        assert!(has_center, "should have vertex at center (0.5, 0.5)");
    }

    #[test]
    fn wminkowski_matches_scipy_reference_values() {
        // scipy.spatial.distance.wminkowski([1, 2], [3, 4], p=2, w=[1, 1])
        // -> sqrt((1-3)^2 + (2-4)^2) = sqrt(8) = 2.8284...
        let result = wminkowski(&[1.0, 2.0], &[3.0, 4.0], 2.0, &[1.0, 1.0]);
        assert!(
            (result - 2.8284271247461903).abs() < 1e-10,
            "wminkowski got {result}, expected 2.8284271247461903"
        );
    }

    #[test]
    fn is_valid_dm_matches_scipy_reference_values() {
        // scipy.spatial.distance.is_valid_dm([[0, 1], [1, 0]]) -> True
        let matrix = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert!(
            is_valid_dm(&matrix, 1e-10),
            "symmetric distance matrix should be valid"
        );
    }

    #[test]
    fn num_obs_dm_matches_scipy_reference_values() {
        // scipy.spatial.distance.num_obs_dm([[0, 1, 2], [1, 0, 1], [2, 1, 0]]) -> 3
        let matrix = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let n = num_obs_dm(&matrix);
        assert_eq!(n, 3, "num_obs_dm should return 3");
    }
}

#[cfg(test)]
mod cdist_minkowski_tests {
    use super::*;
    #[test]
    fn cdist_minkowski_matches_scalar_minkowski() {
        let xa = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, -1.0, 0.5],
            vec![-2.0, 2.0, 1.0],
        ];
        let xb = vec![vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]];
        for &p in &[1.0_f64, 1.5, 2.0, 3.0] {
            let m = cdist_minkowski(&xa, &xb, p).unwrap();
            assert_eq!(m.len(), xa.len());
            for i in 0..xa.len() {
                for j in 0..xb.len() {
                    let expect = minkowski(&xa[i], &xb[j], p);
                    assert!(
                        (m[i][j] - expect).abs() <= 1e-12 * expect.abs().max(1e-12),
                        "p={p} i={i} j={j}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod cdist_seuclidean_tests {
    use super::*;
    #[test]
    fn cdist_seuclidean_matches_scalar() {
        let xa = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, -1.0, 0.5],
            vec![-2.0, 2.0, 1.0],
        ];
        let xb = vec![vec![1.0, 1.0, 1.0], vec![0.0, 0.0, 0.0]];
        let v = vec![0.5, 2.0, 1.5];
        let m = cdist_seuclidean(&xa, &xb, &v).unwrap();
        assert_eq!(m.len(), xa.len());
        for i in 0..xa.len() {
            for j in 0..xb.len() {
                assert!((m[i][j] - seuclidean(&xa[i], &xb[j], &v)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cdist_seuclidean_small_d_soa_matches_scalar_bitwise() {
        // The SoA-across-pairs seuclidean kernel (dim<8) must be BIT-identical to the
        // scalar per-pair `seuclidean` for every small dimension — including the v<=0⇒NaN
        // mask — across the SIMD-chunk lanes and the scalar tail.
        let mut s: u64 = 0x1357_9BDF_2468_ACE0;
        let mut rng = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for d in [1usize, 2, 3, 5, 6, 7] {
            // nb=41 = 5·8+1 exercises both the SIMD chunk and the scalar tail; (600,300)
            // crosses the work gate into the parallel fill.
            for &(na, nb) in &[(40usize, 41usize), (600, 300)] {
                let xa: Vec<Vec<f64>> = (0..na)
                    .map(|_| (0..d).map(|_| rng() * 4.0 - 2.0).collect())
                    .collect();
                let xb: Vec<Vec<f64>> = (0..nb)
                    .map(|_| (0..d).map(|_| rng() * 4.0 - 2.0).collect())
                    .collect();
                // All-positive variances: every finite distance must be bit-exact.
                let v: Vec<f64> = (0..d).map(|_| rng() * 3.0 + 0.1).collect();
                let got = cdist_seuclidean(&xa, &xb, &v).unwrap();
                for (i, gr) in got.iter().enumerate() {
                    for (j, &g) in gr.iter().enumerate() {
                        let want = seuclidean(&xa[i], &xb[j], &v);
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "soa seuclidean mismatch at ({i},{j}) d={d} na={na} nb={nb}"
                        );
                    }
                }
                // A zero/negative variance entry must make every distance NaN (the v<=0
                // mask), matching the scalar helper — verify the mask fires on both the
                // SIMD-chunk lanes and the scalar tail.
                if d >= 2 {
                    let mut vneg = v.clone();
                    vneg[d / 2] = 0.0;
                    let got_n = cdist_seuclidean(&xa, &xb, &vneg).unwrap();
                    for (i, gr) in got_n.iter().enumerate() {
                        for (j, &g) in gr.iter().enumerate() {
                            assert!(
                                g.is_nan() && seuclidean(&xa[i], &xb[j], &vneg).is_nan(),
                                "seuclidean v<=0 NaN mask failed at ({i},{j}) d={d}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod bool_popcount_ab_tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    /// Shared by every test in this crate that writes a process-global perf
    /// toggle. These statics are visible to all test threads at once, so two
    /// toggle-writing tests running concurrently can both invent drift (one
    /// flips the flag mid-way through the other's A arm) and mask it (both
    /// happen to read the same setting). One lock over all of them.
    static TOGGLE_LOCK: Mutex<()> = Mutex::new(());
    fn toggle_lock() -> MutexGuard<'static, ()> {
        TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The across-pairs Euclidean kernel must return EXACTLY the bits the per-pair
    /// `metric_distance` scan returns, for every width it claims.
    ///
    /// Bit equality is the right assertion and it is NOT free here: the reference
    /// `sqeuclidean` is itself SIMD, finishing with an ORDERED horizontal reduce, and the
    /// new kernel reproduces that chain only because below 16 dimensions just one
    /// accumulator is live. A tolerance would hide exactly the case this gate exists for.
    ///
    /// Sweeps `dim` right up to the boundary (15) and `n` so that the per-row remainder
    /// `n - 1 - i` takes every value mod the 8-wide lane count — the scalar tail is where an
    /// off-by-one in a chunked fill hides, and at some rows the tail is the whole row.
    #[test]
    fn pdist_soa_euclidean_matches_metric_distance_bits() {
        let _guard = toggle_lock();
        let coord = |i: usize, d: usize| -> f64 {
            let key = (i * 2_654_435_761usize).wrapping_add(d * 40_503) % 100_003;
            key as f64 / 100_003.0 - 0.5
        };
        for dim in [1usize, 2, 3, 5, 6, 7, 8, 9, 12, 15] {
            for n in [2usize, 3, 8, 9, 10, 16, 17, 23, 40] {
                let points: Vec<Vec<f64>> = (0..n)
                    .map(|i| (0..dim).map(|d| coord(i, d)).collect())
                    .collect();

                PDIST_SOA_EUCLIDEAN.store(false, Ordering::Relaxed);
                let reference = pdist(&points, DistanceMetric::Euclidean).expect("scalar scan");
                PDIST_SOA_EUCLIDEAN.store(true, Ordering::Relaxed);
                let candidate = pdist(&points, DistanceMetric::Euclidean).expect("soa kernel");

                assert_eq!(reference.len(), candidate.len(), "dim={dim} n={n}: length");
                for (idx, (want, got)) in reference.iter().zip(candidate.iter()).enumerate() {
                    assert_eq!(
                        want.to_bits(),
                        got.to_bits(),
                        "dim={dim} n={n}: condensed[{idx}] differs ({want} vs {got})"
                    );
                }
            }
        }
        PDIST_SOA_EUCLIDEAN.store(true, Ordering::Relaxed);
    }

    /// The `dim < 16` bound must be the REASON the kernel is exact, not a coincidence.
    ///
    /// At `dim == 16` the reference's second accumulator goes live and `(acc0 + acc1)` pairs
    /// lane `k` with lane `k + 8` before the ordered reduce, so ascending accumulation is a
    /// different summation order. This asserts the two orders genuinely disagree on real
    /// data — without it, `PDIST_SOA_MAX_DIM` could be widened by someone who tested only
    /// well-conditioned inputs, and the gate above would be an unexamined constant.
    #[test]
    fn pdist_soa_dim_bound_marks_a_real_change_in_summation_order() {
        let dim = PDIST_SOA_MAX_DIM; // 16
        // CONSTRUCTED, not sampled. Random same-scale coordinates round identically under
        // both orders, so they would show the bound to be arbitrary even when it is real.
        // Here seven tiny terms sit in lanes 0..8 and the only large term in lane 8, which
        // is exactly the lane pairing `(acc0 + acc1)` performs:
        //   ascending  0 + 8·2⁻⁵⁴ = 2⁻⁵¹, then + 1.0  ->  1.0 + 2⁻⁵¹   (2 ULP above one)
        //   reference  lane 0 becomes 2⁻⁵⁴ + 1.0 -> 1.0 (below half-ULP, absorbed), and
        //              every later tiny addend is then absorbed too  ->  1.0
        // Each partial sum above is exactly representable, so the disagreement is a
        // property of the ORDER and not of the particular values.
        let mut a = vec![0.0_f64; dim];
        for slot in a.iter_mut().take(8) {
            // Square is 2⁻⁵⁴, a QUARTER ULP of 1.0, so `1.0 + t` absorbs it. The square has
            // to land below the half-ULP 2⁻⁵³ for the pairing to be observable at all.
            *slot = 2.0f64.powi(-27);
        }
        a[8] = 1.0;
        let b = vec![0.0_f64; dim];

        let reference = sqeuclidean(&a, &b);
        let ascending = a
            .iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (&p, &q)| acc + (p - q) * (p - q));

        assert_ne!(
            reference.to_bits(),
            ascending.to_bits(),
            "at dim={dim} the two-accumulator reduce and an ascending fold agreed, so \
             PDIST_SOA_MAX_DIM no longer marks a real boundary and this test cannot \
             justify the gate ({reference:e} vs {ascending:e})"
        );

        // And one width below the bound the two orders MUST agree, or the gate is drawn in
        // the wrong place. Without this arm the test above would pass just as happily if
        // ascending accumulation were wrong everywhere.
        let below = PDIST_SOA_MAX_DIM - 1;
        let reference_below = sqeuclidean(&a[..below], &b[..below]);
        let ascending_below = a[..below]
            .iter()
            .zip(b[..below].iter())
            .fold(0.0_f64, |acc, (&p, &q)| acc + (p - q) * (p - q));
        assert_eq!(
            reference_below.to_bits(),
            ascending_below.to_bits(),
            "at dim={below}, below the bound, the orders disagreed \
             ({reference_below:e} vs {ascending_below:e})"
        );
    }

    /// `SPATIAL_BOOL_POPCOUNT_DISABLE` claims the bit-packed popcount path is
    /// "byte-identical" to the scalar per-dimension count. Before this test that
    /// claim was asserted NOWHERE: the toggle is driven only by
    /// `src/bin/perf_pdist_jaccard.rs` and `src/bin/perf_bool_popcount.rs`,
    /// which are timing harnesses. A perf bin proves the toggle changes speed
    /// and says nothing about whether the two arms agree (frankenscipy-0b982).
    #[test]
    fn jaccard_popcount_ab_is_byte_identical_above_the_dim_gate() {
        let _guard = toggle_lock();

        // The packed path is gated `dim >= JACCARD_POPCOUNT_MIN_DIM` (= 256) at
        // lib.rs:867 (pdist) and :1889 (cdist), so a test below that dim would
        // compare the scalar path against itself and pass vacuously. Every dim
        // here is above the gate, and deliberately NOT a multiple of 64, so the
        // final `u64` word is partial and its padding bits are exercised.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };

        for &dim in &[257usize, 300, 511, 1000] {
            let rows = |n: usize, rand: &mut dyn FnMut() -> f64| -> Vec<Vec<f64>> {
                (0..n)
                    .map(|_| {
                        (0..dim)
                            .map(|_| if rand() < 0.5 { 1.0 } else { 0.0 })
                            .collect()
                    })
                    .collect()
            };
            let xa = rows(9, &mut rand);
            let xb = rows(7, &mut rand);

            // An all-zero row on each side: the union is empty for that pair, the
            // one branch where the packed kernel has to agree with the scalar
            // convention by hand rather than by arithmetic (it returns 0.0).
            let mut xa = xa;
            let mut xb = xb;
            xa.push(vec![0.0; dim]);
            xb.push(vec![0.0; dim]);

            SPATIAL_BOOL_POPCOUNT_DISABLE.store(false, Ordering::Relaxed);
            let packed = cdist_metric(&xa, &xb, DistanceMetric::Jaccard).expect("packed");
            SPATIAL_BOOL_POPCOUNT_DISABLE.store(true, Ordering::Relaxed);
            let scalar = cdist_metric(&xa, &xb, DistanceMetric::Jaccard).expect("scalar");
            SPATIAL_BOOL_POPCOUNT_DISABLE.store(false, Ordering::Relaxed);

            let mut nonzero = 0usize;
            for i in 0..xa.len() {
                for j in 0..xb.len() {
                    assert_eq!(
                        packed[i][j].to_bits(),
                        scalar[i][j].to_bits(),
                        "dim={dim} pair ({i},{j}): packed {} vs scalar {}",
                        packed[i][j],
                        scalar[i][j]
                    );
                    if packed[i][j] != 0.0 {
                        nonzero += 1;
                    }
                }
            }
            // Non-degeneracy: an all-equal or all-zero matrix would satisfy the
            // bit comparison above while testing nothing.
            assert!(
                nonzero >= xa.len(),
                "dim={dim}: only {nonzero} nonzero distances — fixture is degenerate"
            );
        }
    }

    #[test]
    fn pdist_minkowski_seuclidean_match_scalar() {
        let x = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, -1.0, 0.5],
            vec![-2.0, 2.0, 1.0],
            vec![1.0, 0.0, -1.0],
        ];
        let v = vec![0.5, 2.0, 1.5];
        let m = pdist_minkowski(&x, 3.0).unwrap();
        let se = pdist_seuclidean(&x, &v).unwrap();
        let mut idx = 0;
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                assert!((m[idx] - minkowski(&x[i], &x[j], 3.0)).abs() < 1e-12);
                assert!((se[idx] - seuclidean(&x[i], &x[j], &v)).abs() < 1e-12);
                idx += 1;
            }
        }
    }
}

/// frankenscipy-5f06d — driver for `MAHALANOBIS_ASSEMBLY_FORCE_SERIAL`, which
/// had none. Documented BYTE-IDENTICAL: the parallel arm splits the final
/// `sqrt` assembly across rows writing DISJOINT output slices, so no float is
/// combined across threads and rows are reassembled in order.
#[cfg(test)]
mod toggle_ab_mahalanobis_assembly {
    use super::{MAHALANOBIS_ASSEMBLY_FORCE_SERIAL, cdist_mahalanobis};
    use std::sync::atomic::Ordering;

    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The assembly is gated on CELLS, `na * nb >= 1 << 22` -- not on the point
    /// dimension and not on either input length alone. 2048 x 2048 clears it
    /// exactly; anything smaller sends both settings down the same serial loop
    /// and the comparison becomes a path compared with itself.
    ///
    /// `d` is deliberately 2. The gate is met by the CELL COUNT, so dimension is
    /// free to be small -- and the distance compute underneath is O(na*nb*d), so
    /// a larger `d` would only make an unoptimized test binary slower without
    /// making the comparison stronger.
    #[test]
    fn mahalanobis_assembly_toggle_is_byte_identical() {
        let _g = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        const NA: usize = 2048;
        const NB: usize = 2048;
        const D: usize = 2;
        const _: () = assert!(NA * NB >= (1 << 22));
        assert!(
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                >= 2,
            "the parallel arm is vacuous on a host reporting < 2 usable cores"
        );

        // Detector must-hit/must-miss: several assertions below are bit-exact, so
        // "nothing differed" is the passing outcome and would look identical to a
        // comparison that cannot see a difference. `to_bits`, since -0.0 == 0.0.
        assert!(
            f64::from_bits(3.5f64.to_bits() ^ 1).to_bits() != 3.5f64.to_bits()
                && 0.0f64.to_bits() != (-0.0f64).to_bits(),
            "the bit comparison is blind; the exact assertion below would pass \
             vacuously"
        );

        let pt = |i: u64, j: u64| {
            let k = i
                .wrapping_mul(2_654_435_761)
                .wrapping_add(j.wrapping_mul(40_503));
            ((k % 100_003) as f64) / 1000.0
        };
        let xa: Vec<Vec<f64>> = (0..NA)
            .map(|i| (0..D).map(|j| pt(i as u64, j as u64)).collect())
            .collect();
        let xb: Vec<Vec<f64>> = (0..NB)
            .map(|i| (0..D).map(|j| pt(i as u64 + 7919, j as u64)).collect())
            .collect();
        // Identity inverse-covariance keeps the fixture well-conditioned; the
        // toggle gates the sqrt ASSEMBLY, not the covariance algebra.
        let vi: Vec<Vec<f64>> = (0..D)
            .map(|i| (0..D).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist serial");
        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let par = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist parallel");
        MAHALANOBIS_ASSEMBLY_FORCE_SERIAL.store(false, Ordering::Relaxed);

        assert_eq!(serial.len(), NA, "cdist returned the wrong row count");
        // A result that is all zeros -- or all NaN -- compares equal in both arms
        // while exercising nothing, so the fixture is checked for signal first.
        assert!(
            serial.iter().flatten().any(|v| v.is_finite() && *v > 0.0),
            "mahalanobis fixture is degenerate: no positive finite distance"
        );

        let mut first_diff = None;
        'outer: for (r, (rs, rp)) in serial.iter().zip(&par).enumerate() {
            assert_eq!(rs.len(), rp.len(), "row {r} length differs between arms");
            for (c, (x, y)) in rs.iter().zip(rp).enumerate() {
                if x.to_bits() != y.to_bits() {
                    first_diff = Some((r, c, *x, *y));
                    break 'outer;
                }
            }
        }
        assert!(
            first_diff.is_none(),
            "MAHALANOBIS_ASSEMBLY_FORCE_SERIAL is documented BYTE-IDENTICAL and \
             is not: first difference at {first_diff:?}. The parallel arm writes \
             disjoint row slices and combines no floats across threads, so a \
             difference here is a placement or bounds fault, not rounding"
        );

        // Checking only that the two arms AGREE cannot catch a fault they share --
        // e.g. both arms leaving the tail rows untouched. The last row is verified
        // against the closed form directly, since with an identity `vi` the
        // Mahalanobis distance IS the Euclidean one.
        let last = NA - 1;
        let expect: f64 = (0..D)
            .map(|j| (xa[last][j] - xb[NB - 1][j]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (par[last][NB - 1] - expect).abs() <= 1e-9 * expect.max(1.0),
            "the final cell disagrees with the closed form under an identity vi: \
             got {}, expected {expect}. Both arms agreeing on a wrong value is \
             exactly what an arms-only comparison cannot see",
            par[last][NB - 1]
        );
    }

    /// SciPy parity for `scipy.spatial` entry points with no differential coverage
    /// (frankenscipy-ivxx6): `minkowski_distance`, `minkowski_distance_p`, `tsearch`,
    /// `SphericalVoronoi`.
    ///
    /// TWO OF THESE HAVE IMPLEMENTATION-DEPENDENT OUTPUT and are deliberately NOT compared
    /// elementwise, because doing so would manufacture a divergence out of a legal difference:
    ///   * `tsearch` returns a SIMPLEX INDEX, and two libraries may build different but equally
    ///     valid Delaunay triangulations of the same points. What is canonical is whether a query
    ///     lies inside the hull, so that is what is pinned against SciPy; separately the returned
    ///     simplex is checked to actually CONTAIN its query, which is a property rather than a
    ///     convention and needs no oracle.
    ///   * `SphericalVoronoi` vertex and region ORDER is not canonical. Pinned instead on
    ///     order-independent invariants: vertex count, lexicographically sorted vertices, sorted
    ///     region sizes, and that every vertex lies on the sphere.
    ///
    /// `minkowski_distance*` has no such freedom and is compared elementwise, including p < 1
    /// (where it is not a metric, so an implementation assuming the triangle inequality can
    /// quietly special-case it) and p = infinity.
    ///
    /// Measured against SciPy 1.17.1: 17 groups, worst 1.110e-16.
    #[test]
    fn minkowski_tsearch_sphericalvoronoi_against_scipy_1_17_1() {
        use super::{
            Delaunay, SphericalVoronoi, minkowski_distance, minkowski_distance_p, tsearch,
        };

        fn close(name: &str, got: &[f64], want: &[f64]) {
            assert_eq!(got.len(), want.len(), "{name}: length");
            for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12,
                    "{name}[{i}]: got {a:.17e}, SciPy 1.17.1 gives {b:.17e}"
                );
            }
        }

        let xa: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, -1.0],
            vec![-3.0, 0.5, 2.0],
            vec![4.0, -4.0, 1.5],
        ];
        let xb: Vec<Vec<f64>> = vec![
            vec![1.0, -1.0, 2.0],
            vec![0.0, 0.0, 0.0],
            vec![2.5, 1.5, -2.0],
            vec![-1.0, 3.0, 0.0],
        ];

        for (p, want_d, want_dp) in [
            (
                0.5,
                vec![
                    11.65685424949238,
                    11.65685424949238,
                    28.571247279470295,
                    37.29012583965875,
                ],
                vec![
                    3.414213562373095,
                    3.414213562373095,
                    5.345207879911715,
                    6.106564159955969,
                ],
            ),
            (1.0, vec![4.0, 4.0, 10.5, 13.5], vec![4.0, 4.0, 10.5, 13.5]),
            (
                1.5,
                vec![
                    2.8567382778502783,
                    2.8567382778502783,
                    7.827290881260925,
                    9.982059680283934,
                ],
                vec![
                    4.82842712474619,
                    4.82842712474619,
                    21.89864333951443,
                    31.537716372038467,
                ],
            ),
            (
                2.0,
                vec![
                    2.449489742783178,
                    2.449489742783178,
                    6.87386354243376,
                    8.73212459828649,
                ],
                vec![6.0, 6.0, 47.25, 76.25],
            ),
            (
                3.0,
                vec![
                    2.154434690031884,
                    2.154434690031884,
                    6.139110878621298,
                    7.782554699694896,
                ],
                vec![10.0, 10.0, 231.375, 471.375],
            ),
        ] {
            close(
                &format!("minkowski_distance(p={p})"),
                &minkowski_distance(&xa, &xb, p).unwrap(),
                &want_d,
            );
            close(
                &format!("minkowski_distance_p(p={p})"),
                &minkowski_distance_p(&xa, &xb, p).unwrap(),
                &want_dp,
            );
        }
        close(
            "minkowski_distance(p=inf)",
            &minkowski_distance(&xa, &xb, f64::INFINITY).unwrap(),
            &[2.0, 2.0, 5.5, 7.0],
        );

        // ---- tsearch: inside/outside is the canonical part ------------------------------------
        let pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.35, 0.45),
            (0.7, 0.25),
            (0.25, 0.75),
        ];
        let tri = Delaunay::new(&pts).expect("delaunay");
        let queries: Vec<(f64, f64)> = vec![
            (0.5, 0.5),
            (0.2, 0.2),
            (0.85, 0.6),
            (0.1, 0.9),
            (0.5, 0.05),
            (-0.5, 0.5),
            (1.5, 0.5),
            (0.5, -0.4),
        ];
        let found = tsearch(&tri, &queries);
        let inside: Vec<f64> = found.iter().map(|&s| f64::from(s >= 0)).collect();
        close(
            "tsearch/inside",
            &inside,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );
        assert!(
            inside.contains(&1.0) && inside.contains(&0.0),
            "the tsearch fixture must contain queries BOTH inside and outside the hull"
        );

        // PROPERTY, not a convention: the reported simplex must contain its query point.
        for (q, &s) in queries.iter().zip(found.iter()) {
            if s < 0 {
                continue;
            }
            let (i0, i1, i2) = tri.simplices[s as usize];
            let ((ax, ay), (bx, by), (cx, cy)) = (pts[i0], pts[i1], pts[i2]);
            let det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            let l1 = ((bx - q.0) * (cy - q.1) - (cx - q.0) * (by - q.1)) / det;
            let l2 = ((cx - q.0) * (ay - q.1) - (ax - q.0) * (cy - q.1)) / det;
            let l3 = 1.0 - l1 - l2;
            assert!(
                l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9,
                "tsearch returned simplex {s} for {q:?}, which does not contain it"
            );
        }

        // ---- SphericalVoronoi: order-independent invariants -----------------------------------
        let sphere: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [
                0.577_350_269_189_625_8,
                0.577_350_269_189_625_8,
                0.577_350_269_189_625_8,
            ],
            [
                -0.577_350_269_189_625_8,
                -0.577_350_269_189_625_8,
                0.577_350_269_189_625_8,
            ],
        ];
        let sv = SphericalVoronoi::new(&sphere, [0.0, 0.0, 0.0], 1.0).expect("spherical voronoi");
        close(
            "SphericalVoronoi/n_vertices",
            &[sv.vertices.len() as f64],
            &[12.0],
        );

        let mut sorted = sv.vertices.clone();
        sorted.sort_by(|a, b| {
            a[0].total_cmp(&b[0])
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });
        let flat: Vec<f64> = sorted.iter().flat_map(|v| v.iter().copied()).collect();
        close(
            "SphericalVoronoi/vertices(sorted)",
            &flat,
            &[
                -0.6947465906068657,
                -0.6947465906068657,
                -0.18615678789738566,
                -0.6947465906068657,
                0.18615678789738566,
                0.6947465906068657,
                -0.5773502691896258,
                -0.5773502691896258,
                -0.5773502691896258,
                -0.5773502691896258,
                0.5773502691896258,
                -0.5773502691896258,
                -0.5773502691896258,
                0.5773502691896258,
                0.5773502691896258,
                -0.18615678789738566,
                0.6947465906068657,
                0.6947465906068657,
                0.18615678789738566,
                -0.6947465906068657,
                0.6947465906068657,
                0.5773502691896258,
                -0.5773502691896258,
                -0.5773502691896258,
                0.5773502691896258,
                -0.5773502691896258,
                0.5773502691896258,
                0.5773502691896258,
                0.5773502691896258,
                -0.5773502691896258,
                0.6947465906068657,
                -0.18615678789738566,
                0.6947465906068657,
                0.6947465906068657,
                0.6947465906068657,
                -0.18615678789738566,
            ],
        );

        let mut sizes: Vec<f64> = sv.regions.iter().map(|r| r.len() as f64).collect();
        sizes.sort_by(f64::total_cmp);
        close(
            "SphericalVoronoi/region_sizes(sorted)",
            &sizes,
            &[3.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0],
        );

        // PROPERTY: every Voronoi vertex lies on the sphere.
        let worst = sv
            .vertices
            .iter()
            .map(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= 1e-12,
            "SphericalVoronoi vertex off the sphere by {worst:.3e}"
        );
    }

    /// `Delaunay::new` produces a COMPLETE triangulation: exactly `2N - 2 - h` triangles, covering
    /// the whole convex hull (frankenscipy-tkpwa, fixed).
    ///
    /// THE INVARIANT IS ARITHMETIC, not a golden value. A Delaunay triangulation of N points in
    /// general position with `h` of them on the convex hull has EXACTLY `2N - 2 - h` triangles.
    ///
    /// THIS TEST PREVIOUSLY PINNED THE DEFECT and was rewritten when the fix landed, as its own
    /// closing message instructed. Before the fix the triangulation was missing triangles, leaving
    /// holes INSIDE the hull:
    ///
    /// | N | h | expected | actual BEFORE | deficit | uncovered hull area BEFORE |
    /// |---|---|---|---|---|---|
    /// | 400 | 20 | 778 | 777 | 1 | 7.738e-05 |
    /// | 800 | 19 | 1579 | 1575 | 4 | 7.431e-04 |
    /// | 2000 | 23 | 3975 | 3967 | 8 | 1.152e-03 |
    ///
    /// The consequence was silent: `LinearNDInterpolator::eval` returns NaN when no triangle
    /// contains the query, which is right for a point outside the hull and was exactly how a point
    /// in one of these holes got reported. Against live SciPy 1.17.1 on the same fixture, 4 of
    /// 2000 queries (N=800) came back NaN from this crate and finite from SciPy, every one of
    /// them 1.7e-4 to 9.4e-4 INSIDE the hull -- far too deep for the `-1e-10` barycentric
    /// tolerance in `find_simplex` to have been the cause, so widening that tolerance was never
    /// the fix.
    ///
    /// The cause was a FIXED super-triangle margin of 10.0 in `Delaunay::new`, and a fixed value
    /// cannot be right: the required margin was measured to grow with N (100 suffices at N=800,
    /// while N=2000 still lost 4 triangles and needed 1000). The margin now escalates and the
    /// result is checked against `2N - 2 - h`, so the constant is verified rather than trusted.
    ///
    /// N=2000 is included here deliberately: it is the size that survived every intermediate
    /// margin, so a regression that reintroduced a too-small constant would pass at 400 and 800.
    #[test]
    fn delaunay_triangulation_is_complete() {
        use super::{ConvexHull, Delaunay};

        fn lcg(state: &mut u64) -> f64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        }

        // Same generator and seed as the griddata fixture the defect was found on, so this test
        // and that probe describe the same point sets.
        for n in [400usize, 800, 2000] {
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let pts: Vec<(f64, f64)> = (0..n).map(|_| (lcg(&mut s), lcg(&mut s))).collect();
            let tri = Delaunay::new(&pts).expect("delaunay");
            let hull = ConvexHull::new(&pts).expect("hull");
            let expected = 2 * n - 2 - hull.vertices.len();
            assert_eq!(
                tri.simplices.len(),
                expected,
                "N={n}: a Delaunay triangulation of these points has exactly {expected} triangles \
                 (2N-2-h, h={}); a shortfall means holes inside the hull, and queries landing in \
                 one of them silently return NaN",
                hull.vertices.len()
            );

            // The area check is independent of the triangle count: sum the triangle areas and
            // compare against the hull area. A covering triangulation matches exactly. Both are
            // asserted because either alone can be met by a wrong triangulation -- a count by
            // triangles that overlap, an area by two errors that cancel.
            let area = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
                ((b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1)).abs() * 0.5
            };
            let tri_area: f64 = tri
                .simplices
                .iter()
                .map(|&(i, j, k)| area(pts[i], pts[j], pts[k]))
                .sum();
            let hv: Vec<(f64, f64)> = hull.vertices.iter().map(|&i| pts[i]).collect();
            let mut hull_area = 0.0;
            for i in 1..hv.len() - 1 {
                hull_area += area(hv[0], hv[i], hv[i + 1]);
            }
            let uncovered = hull_area - tri_area;
            assert!(
                uncovered.abs() <= 1e-12,
                "N={n}: triangulated area {tri_area:.15} differs from hull area \
                 {hull_area:.15} by {uncovered:.3e}; positive means the triangulation leaves a \
                 hole, negative means triangles overlap"
            );

            // And the user-visible symptom directly: every triangle's own centroid is strictly
            // inside that triangle, so `find_simplex` must locate one for each. This catches a
            // triangulation and a point-location routine that disagree with each other, which the
            // two area/count measures above cannot see.
            let mut centroid_not_found = 0usize;
            for &(i, j, k) in &tri.simplices {
                let c = (
                    (pts[i].0 + pts[j].0 + pts[k].0) / 3.0,
                    (pts[i].1 + pts[j].1 + pts[k].1) / 3.0,
                );
                if tri.find_simplex(c).is_none() {
                    centroid_not_found += 1;
                }
            }
            assert_eq!(
                centroid_not_found,
                0,
                "N={n}: {centroid_not_found} of {} triangle centroids were not located by \
                 find_simplex, so point location disagrees with the triangulation it searches",
                tri.simplices.len()
            );
        }
    }

    #[test]
    fn kdtree_mode_audit_and_dimension_cap() {
        let data = vec![vec![1.0, 2.0], vec![3.0, f64::NAN]];
        let ledger = crate::sync_audit_ledger();

        // Hardened mode rejects non-finite point coordinate
        let err = crate::KDTree::new_with_mode(&data, crate::RuntimeMode::Hardened, Some(&ledger));
        assert!(err.is_err());

        // Verify audit event
        let guard = ledger.lock().unwrap();
        assert_eq!(guard.entries().len(), 1);
        assert!(matches!(
            guard.entries()[0].action,
            crate::AuditAction::FailClosed { .. }
        ));
    }
}
