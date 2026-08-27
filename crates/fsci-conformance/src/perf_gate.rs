//! The performance decision gate — one implementation, shared by every bench.
//!
//! `docs/OPTIMIZATION_PROTOCOL.md` is the prose; this is the code. It exists because
//! the rule it replaces (`cv < 5%`) survived for weeks in hand-rolled harnesses and
//! rejected four candidates whose measured effects were 1.20×, 1.47×, 3.82× and
//! 17–19× against A/A nulls of 1.00–1.02 (three later kept — see
//! `docs/LEDGER_RESURRECTION.md`). A gate that each bench re-derives is a gate that
//! drifts, so: use these functions, do not re-write them.
//!
//! # The rule
//!
//! A speedup is DECIDED iff the candidate's 95% CI lower bound on the **median of
//! per-round ratios** exceeds `1 + 2·(null_edge − 1)`, where `null_edge` is the worse
//! side of the A/A null's own 95% CI. `cv` is reported as provenance and never enters
//! the decision.
//!
//! # Usage
//!
//! ```ignore
//! use fsci_conformance::perf_gate::{elf_sha256, paired, decide};
//!
//! println!("elf_sha256={}", elf_sha256().unwrap_or_else(|e| e));
//! // ... prove bit-identity and execution FIRST, abort on mismatch ...
//! let null = paired(rounds, min_of, || run(Arm::Base), || run(Arm::Base));
//! let cand = paired(rounds, min_of, || run(Arm::Base), || run(Arm::Cand));
//! let verdict = decide(&null, &cand);
//! println!("{null}\n{cand}\n{verdict}");
//! ```
//!
//! Both arms must live in the SAME binary and the SAME invocation, switched by a
//! `#[doc(hidden)] pub static … : AtomicBool`. Whole-binary A/Bs (ISA flags, LTO,
//! allocator) cannot use `paired` — see §6 of the protocol.

use std::fmt;
use std::time::Instant;

/// Fleet-wide default: the sample length below which preemption noise dominates.
pub const DEFAULT_MIN_SAMPLE_MS: f64 = 2.0;
/// Fleet-wide default: inner replicates per sample, keeping the minimum. This is the
/// dominant knob — at a 2 ms sample, `×1 → ×3` moved the measured floor 1.048 → 1.012.
pub const DEFAULT_MIN_OF: usize = 3;
/// Deterministic percentile-bootstrap resamples used for the median confidence interval.
pub const DEFAULT_BOOTSTRAP_RESAMPLES: usize = 10_000;

const BOOTSTRAP_SEED: u64 = 0x6a09_e667_f3bc_c909;

/// One paired arm-vs-arm measurement.
#[derive(Debug, Clone)]
pub struct Paired {
    /// Median wall time of arm A, seconds.
    pub p50_a: f64,
    /// Median wall time of arm B, seconds.
    pub p50_b: f64,
    /// Per-round ratios `a / b` (>1 means arm B is faster).
    pub ratios: Vec<f64>,
    /// Median of `ratios` — the statistic the gate decides on.
    pub ratio_p50: f64,
    /// Lower bound of the 95% percentile-bootstrap CI on the median of `ratios`.
    pub ratio_lo: f64,
    /// Upper bound of the 95% percentile-bootstrap CI on the median of `ratios`.
    pub ratio_hi: f64,
    /// Coefficient of variation of `ratios`. **Provenance only — never a verdict.**
    pub cv: f64,
}

impl fmt::Display for Paired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p50_a={:.6}ms p50_b={:.6}ms ratio_p50={:.6} ci95=[{:.6},{:.6}] cv={:.3}% n={}",
            self.p50_a * 1e3,
            self.p50_b * 1e3,
            self.ratio_p50,
            self.ratio_lo,
            self.ratio_hi,
            self.cv * 100.0,
            self.ratios.len()
        )
    }
}

/// The gate's output.
#[derive(Debug, Clone, Copy)]
pub struct Verdict {
    /// Worse side of the A/A null's 95% CI, expressed as a ratio ≥ 1.
    pub null_edge: f64,
    /// The bar the candidate's CI lower bound must clear: `1 + 2·(null_edge − 1)`.
    pub required: f64,
    /// Candidate CI lower bound actually observed.
    pub observed_lo: f64,
    /// Candidate median ratio.
    pub ratio_p50: f64,
    /// True iff `observed_lo > required`.
    pub decided: bool,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "gate: null_edge={:.6} required={:.6} cand_ci_lo={:.6} => {}\nverdict: {} ratio_p50={:.4}x",
            self.null_edge,
            self.required,
            self.observed_lo,
            if self.decided {
                "DECIDED"
            } else {
                "NOT DECIDED"
            },
            if self.decided { "WIN" } else { "IN-FLOOR" },
            self.ratio_p50
        )
    }
}

/// SHA-256 of this process's own executable, as lowercase hex.
///
/// Print it as line 1 of every bench. A hash computed by a shell step *next to* the
/// run proves nothing about which ELF executed, and rch compiles into an opaque
/// per-worker target dir you cannot predict. Returns the error text on failure so a
/// bench can print it unconditionally rather than silently omitting provenance.
///
/// Implemented without a hash dependency so this crate stays dependency-neutral: the
/// caller feeds the bytes to whatever digest it already links (`sha2`, `blake3`). If
/// you have neither, `sha256_hex` below is a small, correct FIPS-180-4 implementation.
pub fn elf_bytes() -> Result<Vec<u8>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    std::fs::read(&exe).map_err(|e| format!("read {}: {e}", exe.display()))
}

/// `elf_bytes()` hashed with the bundled SHA-256.
pub fn elf_sha256() -> Result<String, String> {
    Ok(sha256_hex(&elf_bytes()?))
}

/// FIPS 180-4 SHA-256, hex-encoded. Small enough to keep the gate dependency-free.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = &chunk[i * 4..i * 4 + 4];
            *word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    median_in_place(&mut xs)
}

fn median_in_place(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

/// Deterministic 95% percentile-bootstrap CI for the sample median.
///
/// The gate is about uncertainty in the median, not the 2.5th and 97.5th
/// percentiles of the raw observations. Those are different statistics: a raw
/// interval describes sample spread and does not narrow as evidence accumulates.
fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let mut generator = XorShift64(BOOTSTRAP_SEED);
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(DEFAULT_BOOTSTRAP_RESAMPLES);
    for _ in 0..DEFAULT_BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            sample.push(values[generator.next() as usize % values.len()]);
        }
        medians.push(median_in_place(&mut sample));
    }
    medians.sort_by(f64::total_cmp);

    let low = DEFAULT_BOOTSTRAP_RESAMPLES * 25 / 1_000;
    let high = DEFAULT_BOOTSTRAP_RESAMPLES * 975 / 1_000;
    (medians[low], medians[high.min(medians.len() - 1)])
}

/// Time one sample: run `f` `min_of` times and keep the MINIMUM.
pub fn sample(min_of: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..min_of.max(1) {
        let start = Instant::now();
        f();
        let secs = start.elapsed().as_secs_f64();
        if secs < best {
            best = secs;
        }
    }
    best
}

/// Run two arms INTERLEAVED inside each round, alternating which goes first, and
/// return the median of per-round ratios with a deterministic 95% bootstrap CI
/// on that median.
///
/// Call it TWICE per decision — `paired(base, base)` for the A/A null, then
/// `paired(base, cand)` — in the same invocation, so both see the same machine state.
pub fn paired(
    rounds: usize,
    min_of: usize,
    mut arm_a: impl FnMut(),
    mut arm_b: impl FnMut(),
) -> Paired {
    let rounds = rounds.max(1);
    let (mut ta, mut tb, mut ratios) = (
        Vec::with_capacity(rounds),
        Vec::with_capacity(rounds),
        Vec::with_capacity(rounds),
    );
    for round in 0..rounds {
        // Alternate which arm goes first so any within-round drift hits both equally.
        let (a, b) = if round % 2 == 0 {
            let a = sample(min_of, &mut arm_a);
            let b = sample(min_of, &mut arm_b);
            (a, b)
        } else {
            let b = sample(min_of, &mut arm_b);
            let a = sample(min_of, &mut arm_a);
            (a, b)
        };
        ta.push(a);
        tb.push(b);
        ratios.push(a / b);
    }
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let var = ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64;
    let (ratio_lo, ratio_hi) = bootstrap_median_ci(&ratios);
    Paired {
        p50_a: median(ta.clone()),
        p50_b: median(tb.clone()),
        ratio_p50: median(ratios.clone()),
        ratio_lo,
        ratio_hi,
        cv: var.sqrt() / mean,
        ratios,
    }
}

/// Apply the gate: DECIDED iff the candidate's CI lower bound clears the null's worse
/// edge with a 2× margin.
pub fn decide(null: &Paired, cand: &Paired) -> Verdict {
    let valid_evidence = |paired: &Paired| {
        paired.p50_a.is_finite()
            && paired.p50_a > 0.0
            && paired.p50_b.is_finite()
            && paired.p50_b > 0.0
            && !paired.ratios.is_empty()
            && paired
                .ratios
                .iter()
                .all(|ratio| ratio.is_finite() && *ratio > 0.0)
            && paired.ratio_p50.is_finite()
            && paired.ratio_p50 > 0.0
            && paired.ratio_lo.is_finite()
            && paired.ratio_lo > 0.0
            && paired.ratio_hi.is_finite()
            && paired.ratio_hi >= paired.ratio_lo
    };

    // The null is symmetric in principle but not in sample: take whichever side is
    // further from unity, so a lopsided null cannot flatter the candidate.
    let hi = null.ratio_hi;
    let lo_inv = if null.ratio_lo > 0.0 {
        1.0 / null.ratio_lo
    } else {
        f64::INFINITY
    };
    let null_edge = hi.max(lo_inv).max(1.0);
    let required = 1.0 + 2.0 * (null_edge - 1.0);
    Verdict {
        null_edge,
        required,
        observed_lo: cand.ratio_lo,
        ratio_p50: cand.ratio_p50,
        decided: valid_evidence(null)
            && valid_evidence(cand)
            && required.is_finite()
            && cand.ratio_lo > required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Spans a block boundary (56..64 bytes forces an extra padding block).
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn a_null_does_not_decide_and_a_real_effect_does() {
        // A/A: both arms identical work. A/B: arm B does a quarter of it.
        let work = |n: usize| {
            let mut acc = 0u64;
            for i in 0..n {
                acc = acc.wrapping_add((i as u64).wrapping_mul(2_654_435_761));
            }
            std::hint::black_box(acc);
        };
        let null = paired(15, 3, || work(200_000), || work(200_000));
        let cand = paired(15, 3, || work(200_000), || work(50_000));

        let null_verdict = decide(&null, &null);
        assert!(
            !null_verdict.decided,
            "an A/A null must never decide: {null_verdict}"
        );
        let verdict = decide(&null, &cand);
        assert!(
            verdict.decided,
            "a 4x effect must clear the gate: null={null} cand={cand} {verdict}"
        );
        assert!(
            verdict.ratio_p50 > 2.0,
            "ratio should reflect the 4x work reduction, got {}",
            verdict.ratio_p50
        );
    }

    #[test]
    fn required_bar_is_twice_the_null_excursion() {
        let mk = |lo: f64, hi: f64| Paired {
            p50_a: 1.0,
            p50_b: 1.0,
            ratios: vec![1.0],
            ratio_p50: 1.0,
            ratio_lo: lo,
            ratio_hi: hi,
            cv: 0.0,
        };
        // Null spanning [0.95, 1.01]: the LOW side is further from unity
        // (1/0.95 = 1.0526 > 1.01), so it sets the edge -> bar 1.1053. A lopsided
        // null must not be allowed to flatter the candidate through its narrow side.
        let lopsided = mk(0.95, 1.01);
        let v = decide(&lopsided, &mk(1.20, 1.30));
        assert!((v.null_edge - 1.0 / 0.95).abs() < 1e-12, "{v}");
        assert!(
            (v.required - (1.0 + 2.0 * (1.0 / 0.95 - 1.0))).abs() < 1e-12,
            "{v}"
        );
        assert!(v.decided, "1.20 CI-low clears a 1.1053 bar: {v}");
        // Same null, candidate CI-low at 1.05 -> inside the bar, not decided, even
        // though its median (1.30) looks impressive.
        assert!(!decide(&lopsided, &mk(1.05, 1.30)).decided);
        // Symmetric null [0.98, 1.03]: the HIGH side dominates (1.03 > 1.0204).
        let v = decide(&mk(0.98, 1.03), &mk(1.05, 1.05));
        assert!((v.null_edge - 1.03).abs() < 1e-12, "{v}");
        assert!((v.required - 1.06).abs() < 1e-12, "{v}");
        assert!(!v.decided, "1.05 CI-low does NOT clear a 1.06 bar: {v}");
    }

    #[test]
    fn bootstrap_interval_estimates_the_median_not_raw_sample_spread() {
        let mut ratios = vec![1.0; 24];
        ratios.extend([1000.0; 8]);

        let (lo, hi) = bootstrap_median_ci(&ratios);

        assert_eq!((lo, hi), (1.0, 1.0));
    }

    #[test]
    fn non_finite_evidence_fails_closed() {
        let invalid = Paired {
            p50_a: 1.0,
            p50_b: 1.0,
            ratios: vec![f64::NAN],
            ratio_p50: f64::NAN,
            ratio_lo: f64::NAN,
            ratio_hi: f64::NAN,
            cv: f64::NAN,
        };
        assert!(!decide(&invalid, &invalid).decided);

        let valid_candidate = Paired {
            p50_a: 2.0,
            p50_b: 1.0,
            ratios: vec![2.0],
            ratio_p50: 2.0,
            ratio_lo: 2.0,
            ratio_hi: 2.0,
            cv: f64::NAN,
        };
        assert!(
            !decide(&invalid, &valid_candidate).decided,
            "a valid candidate must not pass against a non-finite null"
        );
    }

    #[test]
    fn cv_is_provenance_only_even_when_non_finite() {
        let mk = |lo: f64, hi: f64, median: f64| Paired {
            p50_a: 1.0,
            p50_b: 1.0,
            ratios: vec![median],
            ratio_p50: median,
            ratio_lo: lo,
            ratio_hi: hi,
            cv: f64::NAN,
        };
        let verdict = decide(&mk(0.99, 1.01, 1.0), &mk(1.10, 1.20, 1.15));
        assert!(
            verdict.decided,
            "cv must never enter the verdict: {verdict}"
        );
    }
}
