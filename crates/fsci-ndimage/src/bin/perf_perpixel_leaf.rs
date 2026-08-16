//! A/B probe for the two per-pixel B-spline levers in `sample_interpolated`.
//!
//! `zoom`/`shift`/diagonal-`affine_transform` are axis-separable and precompute their per-axis
//! supports once. `rotate`, a GENERAL (non-diagonal) `affine_transform`, and `map_coordinates`
//! have coupled coordinates, so their supports are rebuilt per pixel. Two levers live there:
//!
//!   `compact` — `NDIMAGE_BSPLINE_COMPACT_DISABLE`: the cardinal tap loop spanned
//!               `floor(cc) ± order` (`2·order+1` `cardinal_bspline` calls) though only `order+1`
//!               taps can be nonzero. Shipped 6c53716ff (order3 1.37x, order5 1.53x).
//!   `offs`    — `NDIMAGE_SPLINE_OFFSET_DISABLE`: each of the `(order+1)^ndim` tensor leaves
//!               recomputed `Σ idx[d]·stride[d]` via `coeffs.get`. Premultiply taps by stride once
//!               per pixel ⇒ the leaf is a single `data[base]` load. Shipped 6347c4045.
//!
//!   `simd`    — `NDIMAGE_BSPLINE_SIMD_DISABLE`: `cardinal_bspline` is >60% self and entirely scalar
//!               FP. The compact window leaves a contiguous run of `order+1` = 2/4/6 taps, each an
//!               INDEPENDENT evaluation of the same recursion ⇒ one tap per lane, bit-identical.
//!
//! REJECTED lever (2026-07-10, do not re-add): a `fold` interior fast path skipping `rem_euclid`
//! for tap runs inside `0..=len-1`. Byte-identical but only 1.01-1.05x — `perf annotate` shows
//! ZERO `idiv` in `compute_axis_support` (61.39% self), so there was no division to remove.
//!
//! NOISE: remote rch workers cannot be `taskset`-pinned, so the `map_coords_serial` workload is
//! sized UNDER the parallel gate (`npts · (order+1)^ndim < 2^18`) and therefore runs on the
//! SERIAL path by construction — the honest way to time a per-pixel arithmetic lever.
//!
//! Each lever carries its own NULL CONTROL, a row where the knob provably cannot act:
//!   `offs`            → `order=0` (`sample_interpolated` returns before the leaf).
//!   `compact`, `simd` → `Constant` mode (routes to `fold_wrap_cubic` / `bspline_local_support`,
//!                       never the cardinal loop); `order=0` is inert for both, and `order=1` is
//!                       inert for `simd` (empty recursion ⇒ the scalar kernel is used).
//! Any run whose control drifts off 1.00x is noise-dominated and must be discarded.
//!
//! A/A NULL CONTROL (franken_whisper), decided by MEDIAN (frankenmermaid calibration). Inert-path
//! controls prove a knob cannot act; they do NOT measure the harness's noise floor. So every row also
//! times a SECOND instance of the ORIG arm, interleaved with the other two in one measured routine.
//! Per iteration we form the A/A ratio `orig_i / orig2_i` (identical code, so it must centre on 1.000)
//! and the A/B ratio `orig_i / cand_i`.
//!
//! GATE ON THE MEDIAN, NOT ON cv. A `cv < 5%` gate is UNREACHABLE on this hardware (mermaid swept
//! min_sample x min_of and never attained it). A row is DECIDABLE only when the candidate's MEDIAN
//! ratio lies clearly OUTSIDE the null's observed range `[min(null_i), max(null_i)]`. Otherwise the
//! effect is inside the floor and the row decides nothing — neither a WIN nor a REJECT. cv is still
//! reported, as information. The floor is PER-FUNCTION (frankenlibc), so it is measured per row.
//!
//! SUBSTRATE (rule v2): both arms live in THIS binary, are selected by an in-process atomic, and are
//! ALTERNATED per iteration inside one measured routine — NOT merely registered as two Criterion
//! group members, which run sequentially and therefore do NOT cancel worker/thermal drift. A single
//! `rch exec` invocation measures both on the same worker (rch picks workers non-deterministically
//! and the ORIG/CAND ratio is not worker-invariant, so an A/B split across two invocations is
//! invalid). Every input is fed THROUGH `black_box` and the whole result is consumed BY `black_box`,
//! so no arm can be dead-code-eliminated (a DCE'd arm shows 0% self-time — the integrity rule's
//! other catch; here `compute_axis_support` measures 61.39% self, so nothing was eliminated).
//!
//! Usage: `perf_perpixel_leaf <compact|offs|simd> [order] [reps] [iters]`
//!
//! GOTCHA: `.rch.env` sets `RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,FSCI_REQUIRE_SCIPY_ORACLE`, so an
//! arbitrary env var (e.g. `FSCI_AB_REPS`) is DROPPED before it reaches the remote worker — the run
//! silently uses the defaults. Pass sample counts as ARGS, which are part of the command and do
//! propagate. The emitted header echoes `reps`/`iters` so a silently-ignored setting is visible.
use fsci_ndimage::{
    BoundaryMode, NDIMAGE_BSPLINE_COMPACT_DISABLE, NDIMAGE_BSPLINE_SIMD_DISABLE,
    NDIMAGE_SPLINE_OFFSET_DISABLE, NdArray, affine_transform, map_coordinates, rotate,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Mean and coefficient-of-variation (%) of a sample set.
fn mean_cv(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 })
}

// ── Self-certification (LEDGER RULE) ───────────────────────────────────────────────────────────
// Every WIN/REJECT row must record the BINARY SHA-256, the WORKER identity, the SELF-TIME of the
// function under test, and cv_pct. `rch` picks workers non-deterministically, so without the worker
// id and the binary hash a row is not reproducible — frankenredis found 67 of its 70 reject rows
// carried no sha256 and could not be re-run. The hash is computed here, in-process, over this
// binary's own bytes, so it is recorded on the machine that actually produced the numbers.
//
// Self-contained SHA-256 (FIPS 180-4). Safe Rust, no new dependency (the workspace has only blake3,
// and a probe must not add one).
const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, c) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*c);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
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

/// Runtime ISA of the machine producing these numbers. A ratio measured on a binary compiled for
/// baseline x86-64 (SSE2, 128-bit) is not comparable to one compiled with AVX2 (256-bit), and the
/// fleet is heterogeneous — so record what the CPU offers and what the binary was built for.
fn isa() -> String {
    let has = |f: &str, on: bool| if on { format!("+{f}") } else { format!("-{f}") };
    let cpu = format!(
        "cpu:{} {} {}",
        has("avx2", is_x86_feature_detected!("avx2")),
        has("fma", is_x86_feature_detected!("fma")),
        has("avx512f", is_x86_feature_detected!("avx512f"))
    );
    let built = format!(
        "built:{} {}",
        has("avx2", cfg!(target_feature = "avx2")),
        has("fma", cfg!(target_feature = "fma"))
    );
    format!("{cpu} | {built}")
}

/// `binary_sha256`, worker identity and exe path of the process producing these numbers.
///
/// Known-answer self-test first: a certification tool that is not itself verified is worse than
/// none, because it makes an unreproducible row *look* reproducible.
fn certify() -> (String, String, String) {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256 KAT (empty) failed"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256 KAT (abc) failed"
    );
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<unknown>".into());
    let sha = std::fs::read(&exe).map_or_else(|_| "<unreadable>".into(), |b| sha256_hex(&b));
    let worker = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "<unknown>".into());
    (sha, worker, exe)
}

/// A general (non-diagonal) affine, so the diagonal separable gate misses.
const GENERAL_AFFINE: [[f64; 3]; 2] = [[0.9, 0.3, -20.0], [-0.3, 0.9, 15.0]];

/// Kept under the parallel gate: `NPTS * (order+1)^2 < 2^18` for every order <= 5 (6000*36=216000).
const NPTS: usize = 6000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lever = args.get(1).map_or("offs", |s| s.as_str());
    let only_order: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Args win over env: env does not survive `rch exec` (see the GOTCHA above).
    let reps = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_usize("FSCI_AB_REPS", 3));
    let iters = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| env_usize("FSCI_AB_ITERS", 7));

    let mut s = 1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let big_side = 512usize;
    let big = NdArray::new(
        (0..big_side * big_side).map(|_| r()).collect(),
        vec![big_side, big_side],
    )
    .unwrap();

    // Small input keeps the spline prefilter from diluting the per-point leaf work.
    let small_side = 64usize;
    let small = NdArray::new(
        (0..small_side * small_side).map(|_| r()).collect(),
        vec![small_side, small_side],
    )
    .unwrap();
    // Coupled (non-separable) coordinates.
    let coords: Vec<Vec<f64>> = {
        let (mut rr, mut cc) = (Vec::with_capacity(NPTS), Vec::with_capacity(NPTS));
        for i in 0..NPTS {
            let (y, x) = ((i / small_side) as f64, (i % small_side) as f64);
            rr.push(0.9 * y + 0.05 * x + 3.7);
            cc.push(0.9 * x - 0.05 * y + 2.3);
        }
        vec![rr, cc]
    };

    let offs_lever = lever == "offs";
    let simd_lever = lever == "simd";
    println!(
        "# same-binary A/B lever={lever}: {}",
        match lever {
            "offs" => "ORIG index-space leaf vs flat-offset leaf (per-pixel path)",
            "simd" => "ORIG per-tap scalar cardinal_bspline vs one lane-parallel run",
            _ => "ORIG full tap window vs compact support window",
        }
    );
    println!("# CONTROL rows must read ~1.00x; otherwise the run is noise-dominated.");
    // LEDGER RULE: a row without binary_sha256 + worker + cv_pct is not reproducible.
    let (sha, worker, exe) = certify();
    println!("# binary_sha256={sha}");
    println!("# worker={worker} exe={exe} reps={reps} iters={iters}");
    println!("# isa {}", isa());
    println!(
        "# self_time(function under test): cardinal_bspline is inlined into \
         compute_axis_support = 61.39% self (captured rotate order=3 profile)"
    );
    let set_orig = |orig: bool| {
        if offs_lever {
            NDIMAGE_SPLINE_OFFSET_DISABLE.store(orig, Ordering::Relaxed);
        } else if simd_lever {
            NDIMAGE_BSPLINE_SIMD_DISABLE.store(orig, Ordering::Relaxed);
        } else {
            NDIMAGE_BSPLINE_COMPACT_DISABLE.store(orig, Ordering::Relaxed);
        }
    };

    for &order in &[0usize, 1, 2, 3, 4, 5] {
        if only_order.is_some_and(|o| o != order) {
            continue;
        }
        // Order 0 has no cardinal loop, so it is inert for `compact` too — keep it as an extra
        // control row rather than skipping, since a knob that moves it is a bug.
        for &mode in &[BoundaryMode::Reflect, BoundaryMode::Constant] {
            type KernelArm<'a> = (&'a str, Box<dyn Fn() -> Vec<f64> + 'a>);
            let kernels: [KernelArm; 3] = [
                (
                    "rotate_par",
                    Box::new(|| {
                        rotate(black_box(&big), 33.0, false, order, mode, 0.0)
                            .unwrap()
                            .data
                    }),
                ),
                (
                    "affine_gen_par",
                    Box::new(|| {
                        affine_transform(
                            black_box(&big),
                            black_box(&GENERAL_AFFINE),
                            order,
                            mode,
                            0.0,
                        )
                        .unwrap()
                        .data
                    }),
                ),
                (
                    "map_coords_serial",
                    Box::new(|| {
                        map_coordinates(black_box(&small), black_box(&coords), order, mode, 0.0)
                            .unwrap()
                    }),
                ),
            ];

            for (name, f) in &kernels {
                // Parity first: same binary, both arms, bit-level compare.
                set_orig(false);
                let cand = f();
                set_orig(true);
                let orig = f();
                let md = cand
                    .iter()
                    .zip(&orig)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                let bits = cand
                    .iter()
                    .zip(&orig)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();

                let bench = |orig: bool| {
                    set_orig(orig);
                    // Consume the WHOLE Vec through black_box (not just `.len()`, which would let a
                    // fully-inlined pure call have its element computation eliminated).
                    drop(black_box(f()));
                    let t = Instant::now();
                    for _ in 0..reps {
                        drop(black_box(f()));
                    }
                    t.elapsed().as_secs_f64() / reps as f64 * 1000.0
                };
                // Interleave ORIG / candidate / ORIG-again so slow drift hits all arms equally.
                // The third arm is the A/A null control: identical code to the first.
                let (mut ov, mut cv, mut nv) = (Vec::new(), Vec::new(), Vec::new());
                let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
                for _ in 0..iters {
                    let (o, c, n) = (bench(true), bench(false), bench(true));
                    // PAIRED per-iteration ratios: drift cancels inside the pair.
                    null_r.push(o / n);
                    cand_r.push(o / c);
                    ov.push(o);
                    cv.push(c);
                    nv.push(n);
                }
                let median = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    let n = v.len();
                    if n % 2 == 1 {
                        v[n / 2]
                    } else {
                        0.5 * (v[n / 2 - 1] + v[n / 2])
                    }
                };
                let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
                let null_med = median(&mut nr);
                let cand_med = median(&mut cr);
                let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
                let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
                // DECIDABLE iff the candidate median lies clearly outside the null's observed range.
                let decidable = cand_med > null_hi || cand_med < null_lo;
                set_orig(false);
                let (om, ocv) = mean_cv(&ov);
                let (cm, ccv) = mean_cv(&cv);
                let (_nm, ncv) = mean_cv(&nv);
                let ob = ov.iter().copied().fold(f64::MAX, f64::min);
                let cb = cv.iter().copied().fold(f64::MAX, f64::min);
                let _nb = nv.iter().copied().fold(f64::MAX, f64::min);
                let is_control = if offs_lever {
                    order == 0
                } else if simd_lever {
                    // order 0 has no cardinal loop; order 1 has an empty recursion and stays scalar.
                    order <= 1 || mode == BoundaryMode::Constant
                } else {
                    order == 0 || mode == BoundaryMode::Constant
                };
                let tag = match (decidable, is_control) {
                    (_, true) => "CONTROL ",
                    (true, false) => "DECIDED ",
                    (false, false) => "IN-FLOOR",
                };
                println!(
                    "{tag} order={order} {mode:?} {name}: orig {ob:.2}ms cand {cb:.2}ms \
                     | CAND median {cand_med:.3}x | NULL median {null_med:.3}x range \
                     [{null_lo:.3}, {null_hi:.3}] | cv o/c/n {ocv:.1}/{ccv:.1}/{ncv:.1}% \
                     best {:.2}x mean {:.2}x maxdiff={md:.1e} bitmism={bits}",
                    ob / cb,
                    om / cm
                );
            }
        }
    }
}
