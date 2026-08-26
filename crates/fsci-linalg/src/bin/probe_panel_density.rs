//! What arithmetic density can a dense f64 panel kernel reach in SAFE RUST on this machine?
//!
//! WHY THIS EXISTS (frankenscipy-8m8s7, under llywn.3). llywn's residual gap against SciPy's
//! `splu` is not a fill gap and not an ordering gap -- both were measured and closed. It is an
//! ARITHMETIC DENSITY gap: on the `laplacian_3d_cubic side=16` cell SciPy retires 1.7638 FLOPs per
//! instruction and we retire 0.3334, a 5.29x difference, because SuperLU does 1.85x MORE
//! arithmetic inside dense supernodal panels and buys 2.85x fewer instructions for it.
//!
//! llywn.3 named a from-scratch safe-Rust dense panel kernel as the only remaining lever. Building
//! the supernodal machinery around such a kernel is expensive and llywn.2 already paid for one
//! negative of that shape (the existing blocked path measured 4.17x WORSE than per-pivot with its
//! symbolic phase zeroed). So this prices the KERNEL FIRST, on its own, before anything is built
//! around it:
//!
//!   * if a safe-Rust panel kernel reaches ~1.7 FLOPs/instruction, the gap is closable in
//!     principle and llywn's remaining lever is real work with a known ceiling;
//!   * if it stalls near our current 0.33, the wall is codegen/ISA rather than algorithm, and
//!     llywn is WALLED rather than open with a lever nobody can land.
//!
//! Either answer closes a question that is currently open on speculation.
//!
//! THE ANSWER, MEASURED 2026-08-26 on `thinkstation1`, 50 reps over the `panel` shapes:
//!
//!     variant            instructions         fp_ops      mac_flops   FLOPs/ins
//!     setup                   7,905,253      2,397,492      1,515,922    0.3033
//!     naive_ijk           2,492,232,630    544,093,513      1,515,922    0.2183
//!     naive_kij             367,737,825    539,792,713      1,515,922    1.4679
//!     naive_kij_fma         304,443,744    539,792,713    538,911,122    1.7730
//!     tiled4x4              630,637,146    544,093,513      1,515,922    0.8628
//!     tiled4x4_fma          563,680,121    544,093,513    538,911,122    0.9653
//!     tiled8x4_fma          560,351,722    544,093,513    538,911,122    0.9710
//!     matmul_public       1,598,296,511    544,093,513      1,515,922    0.3404  (see caveat)
//!
//! **YES: safe Rust reaches SciPy's density.** `naive_kij_fma` retires 1.7730 FLOPs/instruction
//! against SciPy's 1.7638 on the cubic cell -- and it is a plain unit-stride loop, no intrinsics,
//! no `unsafe`, no blocking.
//!
//! SECOND PASS (frankenscipy-fzk5t), AND IT CORRECTS THE READING ABOVE. Those numbers are all at
//! k=32..64. llywn's supernodes are **5.35 columns wide**, so k is about 5, and density is not the
//! only thing that changes. Counting `L1-dcache-load-misses` as well, 50 reps.
//!
//! The executed-binary SHA-256 for these tables is pinned in the ledger row rather than here, on
//! purpose: a sha written into this comment is invalidated by the next edit to this comment, so a
//! source-pinned hash reads as provenance while being self-referentially stale. The ledger row is
//! the artifact that has to name the binary that produced the numbers.
//!
//!     shapes=supernode (k=5)     instructions       fp_ops  FLOPs/ins     L1d_miss  miss/kFLOP
//!     naive_kij                   512,403,220  722,621,075     1.4103   48,846,806      67.597
//!     naive_kij_fma               423,376,788  722,621,075     1.7068   48,795,176      67.525
//!     naive_ikj                   513,507,942  722,621,075     1.4072   12,332,786      17.067
//!     naive_ikj_fma               426,075,526  722,621,075     1.6960   11,584,267      16.031
//!     tiled4x4_fma              1,032,654,036  794,710,675     0.7696   11,699,391      14.722
//!
//!     shapes=panel (k=32/64)     instructions       fp_ops  FLOPs/ins     L1d_miss  miss/kFLOP
//!     naive_kij                   367,831,569  539,792,713     1.4675   35,327,407      65.446
//!     naive_ikj                   361,338,338  539,792,713     1.4939   32,563,724      60.326
//!     naive_ikj_fma               295,945,346  539,792,713     1.8240   32,851,297      60.859
//!
//! **AT THE REAL SUPERNODE WIDTH, LOOP ORDER IS WORTH 4.22-FOLD IN D1 READ MISSES AT IDENTICAL
//! DENSITY** (67.597 -> 16.031 misses per kFLOP; density 1.41 either way, 1.70 either way with
//! `mul_add`). At k=64 the same choice is worth about 8%. Choosing `k-i-j` on the k=64 evidence
//! and carrying it to k=5 would have kept the density and thrown away the entire traffic win --
//! and llywn's own decomposition puts 78.65% of the elimination's D1 read misses in exactly that
//! streaming. The supernodal argument predicts the row is touched once per supernode instead of
//! once per column, so misses should fall by about `1 - 1/w`; at w=5 that predicts 5-fold and
//! 4.22-fold is measured, the shortfall being the A and B streams, which do not reduce.
//!
//! `tiled4x4_fma` reaches the low miss count too (14.722) and pays 0.7696 density for it. `i-k-j`
//! is the only variant that gets BOTH.
//!
//! TWO RESULTS THAT INVERT THE PREMISE llywn.3 WAS WORKING FROM:
//!
//!   1. **Hand register-tiling is WORSE, not better.** The 4x4 and 8x4 kernels -- the textbook
//!      way to buy BLAS-3 density -- land at 0.86-0.97, about 1.8x BELOW the simple contiguous
//!      loop. LLVM vectorises the plain loop into packed AVX2 and my accumulator arrays get in
//!      its way. A panel kernel for llywn should therefore NOT be written as a register-tiled
//!      micro-kernel; that is the shape that loses.
//!
//!   2. **Most of the density is NOT from FMA, so the contract block does not bite.** Going from
//!      our sparse 0.3334 to the plain dense loop is 0.3334 -> 1.4679, a 4.40x gain, and it is
//!      BIT-IDENTICAL to `naive_ijk` (asserted below). Adding `mul_add` on top is only a further
//!      1.21x. llywn.3 closed its FMA sub-lever because
//!      `supernodal_elimination_is_bit_identical_to_the_sequential_one` pins splu factor bits --
//!      but 83% of SciPy's density is reachable WITHOUT touching that contract. The gap is
//!      dominated by DENSE CONTIGUOUS DATA, not by fused arithmetic.
//!
//! WHY IT WORKS ON A LOADED HOST. Instructions and FLOPs are load-independent counts. This box has
//! held loadavg 5-190 all session; that is exactly why llywn.3 was counted rather than timed, and
//! the same applies here. NOTHING IN THIS PROBE IS A TIMING CLAIM.
//!
//! HOW TO READ IT. One variant per process, selected by argv, so `perf stat` attributes its
//! counters to that variant alone. The `setup` variant does the allocation and fill and no
//! arithmetic, so its counts are the floor to subtract rather than a number to assume:
//!
//!     perf stat -e instructions,fp_ret_sse_avx_ops.all,fp_ret_sse_avx_ops.mac_flops \
//!       probe_panel_density <variant> [reps] [panel|supernode]
//!
//! Add `-e L1-dcache-load-misses` for the traffic question, which is the one that separates the
//! loop orders at small k.
//!
//! Variants: setup, naive_ijk, naive_kij, naive_kij_fma, naive_ikj, naive_ikj_fma, tiled4x4,
//! tiled4x4_fma, tiled8x4_fma, matmul_public.
//! Shape sets: `panel` (k=32/64, default), `supernode` (k=5, llywn's actual regime).
//!
//! CORRECTNESS IS CHECKED, because a fast-but-wrong kernel would post a density number just fine.
//! Every variant is verified against `naive_ijk` before its counted region runs, and the check is
//! reported. The non-FMA variants must agree to the BIT; the `mul_add` variants are permitted a
//! tolerance, since fusing is precisely what makes them differ.
use std::hint::black_box;

/// Panel shapes for a trailing update `C[m x n] -= A[m x k] * B[k x n]`.
/// `m` and `n` are the trailing rows and columns; `k` is the supernode width.
/// All `m` and `n` are multiples of 8 so the tiled kernels need no edge handling -- edge cases are
/// a correctness concern for a shipping kernel but would only add instructions that are not part
/// of the density question being asked here.
///
/// TWO SHAPE SETS, AND THE SECOND IS THE ONE THAT MATTERS FOR llywn (frankenscipy-fzk5t).
///
/// The `panel` set uses k=32..64. That is a reasonable dense-GEMM regime and it is what the first
/// pass of this probe measured -- but IT IS NOT THE REGIME llywn RUNS IN, and reading the k=64
/// result as if it were was a real error in that pass.
///
/// The cubic cell's relaxed supernodes measure **5.35 columns wide** (ledger 2026-08-26, mean
/// 5.35, blocks 97.2% full). So llywn's panel update has k about 5, and at k=5 the LOOP ORDER
/// stops being a detail:
///
///   * `k-i-j` streams the whole target block C once per pivot column -- k passes over m*n. That
///     is exactly the traffic llywn's decomposition blames, where 78.65% of the elimination's D1
///     read misses sit in merge streaming because a target row is touched once per PIVOT COLUMN.
///   * `i-k-j` holds ONE target row and applies all k pivot updates to it before moving on, so the
///     row is touched once per SUPERNODE. That is the whole point of supernodal blocking.
///
/// At k=64 cache absorbs the difference and `k-i-j` looks fine. At k=5 it should not. The
/// `supernode` set exists to make that visible rather than assumed, which is why both sets ship
/// and the crossover is measured rather than argued.
const SHAPES_PANEL: &[(usize, usize, usize)] = &[(64, 64, 32), (128, 128, 64), (256, 256, 64)];
const SHAPES_SUPERNODE: &[(usize, usize, usize)] =
    &[(1024, 128, 5), (2048, 128, 5), (4096, 256, 5)];

fn shapes_for(set: &str) -> &'static [(usize, usize, usize)] {
    match set {
        "panel" => SHAPES_PANEL,
        "supernode" => SHAPES_SUPERNODE,
        other => {
            eprintln!("unknown shape set {other}; expected `panel` or `supernode`");
            std::process::exit(2);
        }
    }
}

fn fill(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.0131 + seed).sin() + 0.5)
        .collect()
}

/// `C -= A * B`, i-k-j order: THE SUPERNODAL SHAPE.
///
/// One target row `crow` is held while all `k` pivot updates are applied to it, so the row is
/// touched once per SUPERNODE rather than once per pivot column. `crow` is `n * 8` bytes -- 1 KiB
/// at n=128 -- so it stays resident in L1 across the whole `k` loop, which is the traffic
/// reduction supernodal blocking is supposed to buy and which `k-i-j` cannot get at small `k`.
///
/// Bit-identical to `naive_ijk` and to `naive_kij`: for a fixed `(i, j)` all three subtract the
/// same terms in the same `p` order, and negation distributes exactly through rounding.
fn naive_ikj(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        for (p, &av) in arow.iter().enumerate() {
            let brow = &b[p * n..p * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow.iter()) {
                *cv -= av * bv;
            }
        }
    }
}

/// `i-k-j` with the multiply-subtract fused. See `naive_kij_fma` for why the fusion has to be
/// written out: Rust builds with `fp-contract=off` and will not do it on our behalf.
fn naive_ikj_fma(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        let crow = &mut c[i * n..i * n + n];
        for (p, &av) in arow.iter().enumerate() {
            let brow = &b[p * n..p * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow.iter()) {
                *cv = (-av).mul_add(bv, *cv);
            }
        }
    }
}

/// `C -= A * B`, i-j-k order: the textbook triple loop, and the one that strides B badly.
fn naive_ijk(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        let arow = &a[i * k..i * k + k];
        for j in 0..n {
            let mut s = 0.0;
            for (p, &av) in arow.iter().enumerate() {
                s += av * b[p * n + j];
            }
            c[i * n + j] -= s;
        }
    }
}

/// `C -= A * B`, k-i-j order. Every access is unit-stride, but each inner iteration reaches memory
/// for its accumulator -- this is the shape our scalar sparse elimination update already has, and
/// it is here as the honest "what we do now" reference point rather than as a strawman.
fn naive_kij(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for p in 0..k {
        let brow = &b[p * n..p * n + n];
        for i in 0..m {
            let av = a[i * k + p];
            let crow = &mut c[i * n..i * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow.iter()) {
                *cv -= av * bv;
            }
        }
    }
}

/// The same k-i-j loop with the multiply-subtract written as one `mul_add`.
///
/// `*cv -= av * bv` is exactly the shape a single `vfnmadd` computes, but Rust builds with
/// `fp-contract=off` so the compiler will NOT fuse it on our behalf -- llywn.3 measured zero MAC
/// FLOPs across the whole splu cell for that reason. Writing the fusion explicitly is the only way
/// to find out what the loop is worth when it is allowed to happen, and it separates "dense
/// contiguous data" from "fused arithmetic" as two independent contributions to density.
fn naive_kij_fma(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for p in 0..k {
        let brow = &b[p * n..p * n + n];
        for i in 0..m {
            let av = a[i * k + p];
            let crow = &mut c[i * n..i * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow.iter()) {
                *cv = (-av).mul_add(bv, *cv);
            }
        }
    }
}

/// Register-tiled 4x4. Sixteen accumulators live across the whole `k` loop, so each pair of loaded
/// values feeds sixteen multiply-adds instead of one. This is the entire idea behind BLAS-3
/// density, written in safe Rust with no intrinsics.
fn tiled_4x4(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64], fma: bool) {
    let mut i0 = 0;
    while i0 < m {
        let mut j0 = 0;
        while j0 < n {
            let mut acc = [[0.0f64; 4]; 4];
            for p in 0..k {
                let brow = &b[p * n + j0..p * n + j0 + 4];
                let av = [
                    a[i0 * k + p],
                    a[(i0 + 1) * k + p],
                    a[(i0 + 2) * k + p],
                    a[(i0 + 3) * k + p],
                ];
                if fma {
                    for ii in 0..4 {
                        for jj in 0..4 {
                            acc[ii][jj] = av[ii].mul_add(brow[jj], acc[ii][jj]);
                        }
                    }
                } else {
                    for ii in 0..4 {
                        for jj in 0..4 {
                            acc[ii][jj] += av[ii] * brow[jj];
                        }
                    }
                }
            }
            for ii in 0..4 {
                let crow = &mut c[(i0 + ii) * n + j0..(i0 + ii) * n + j0 + 4];
                for jj in 0..4 {
                    crow[jj] -= acc[ii][jj];
                }
            }
            j0 += 4;
        }
        i0 += 4;
    }
}

/// Register-tiled 8x4: thirty-two accumulators. Whether going wider helps or spills is a property
/// of the register file and the codegen, which is exactly what this probe is for.
fn tiled_8x4_fma(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let mut i0 = 0;
    while i0 < m {
        let mut j0 = 0;
        while j0 < n {
            let mut acc = [[0.0f64; 4]; 8];
            for p in 0..k {
                let brow = &b[p * n + j0..p * n + j0 + 4];
                for ii in 0..8 {
                    let av = a[(i0 + ii) * k + p];
                    for jj in 0..4 {
                        acc[ii][jj] = av.mul_add(brow[jj], acc[ii][jj]);
                    }
                }
            }
            for ii in 0..8 {
                let crow = &mut c[(i0 + ii) * n + j0..(i0 + ii) * n + j0 + 4];
                for jj in 0..4 {
                    crow[jj] -= acc[ii][jj];
                }
            }
            j0 += 4;
        }
        i0 += 8;
    }
}

fn run(variant: &str, m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    match variant {
        "naive_ijk" => naive_ijk(m, n, k, a, b, c),
        "naive_ikj" => naive_ikj(m, n, k, a, b, c),
        "naive_ikj_fma" => naive_ikj_fma(m, n, k, a, b, c),
        "naive_kij" => naive_kij(m, n, k, a, b, c),
        "naive_kij_fma" => naive_kij_fma(m, n, k, a, b, c),
        "tiled4x4" => tiled_4x4(m, n, k, a, b, c, false),
        "tiled4x4_fma" => tiled_4x4(m, n, k, a, b, c, true),
        "tiled8x4_fma" => tiled_8x4_fma(m, n, k, a, b, c),
        "matmul_public" => {
            // The public GEMM takes nested Vecs and computes A*B rather than a subtracting
            // update, so it is measured on the multiply it does do; the density question is about
            // the inner kernel, and the trailing subtraction is m*n flops against m*n*k.
            //
            // READ THIS ROW WITH THE CAVEAT AND DO NOT FILE A BUG FROM IT. It measured 0.3404
            // FLOPs/instruction at these panel sizes, which looks like a damning result for our
            // dense GEMM and is NOT one: the `Vec<Vec<f64>>` construction below is an O(m*k + k*n)
            // allocate-and-copy performed on EVERY call, and `matmul` is multithreaded, so at
            // panel sizes this row is dominated by the adapter and the thread spawn rather than by
            // the kernel. The control that settles it is `perf_matmul` under the same counters at
            // n=1024/2048/4096, where the same function reaches 1.5571 -- close to the plain
            // contiguous loop and nowhere near 0.34. The row is kept because the adapter cost IS
            // real for anyone calling `matmul` on panel-sized blocks, which is exactly what a
            // supernodal path would do; it is a statement about the CALLING CONVENTION at small
            // sizes, not about the arithmetic.
            let av: Vec<Vec<f64>> = (0..m).map(|i| a[i * k..i * k + k].to_vec()).collect();
            let bv: Vec<Vec<f64>> = (0..k).map(|p| b[p * n..p * n + n].to_vec()).collect();
            let prod = fsci_linalg::matmul(&av, &bv).expect("matmul");
            for i in 0..m {
                for j in 0..n {
                    c[i * n + j] -= prod[i][j];
                }
            }
        }
        "setup" => {}
        other => {
            eprintln!("unknown variant {other}");
            std::process::exit(2);
        }
    }
}

fn main() {
    let variant = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "setup".to_string());
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let shape_set = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "panel".to_string());
    let shapes = shapes_for(&shape_set);

    // ---- correctness, BEFORE the counted region -------------------------------------------
    // A kernel that computes the wrong thing would post a density number just as happily.
    //
    // The check itself executes `naive_ijk` plus the variant once per shape, so under `perf stat`
    // it would be counted along with the kernel and would contaminate every density figure --
    // and it would contaminate the `setup` floor UNEQUALLY, since `setup` has nothing to check.
    // Counted runs therefore set `FSCI_PANEL_COUNTED=1` and verify in a separate invocation of
    // this same binary; the ELF hash is reported either way so the two invocations are provably
    // the same code.
    let counted = std::env::var("FSCI_PANEL_COUNTED").is_ok_and(|v| v == "1");
    if variant != "setup" && !counted {
        for &(m, n, k) in shapes {
            let a = fill(m * k, 0.3);
            let b = fill(k * n, 1.1);
            let mut want = vec![0.0; m * n];
            naive_ijk(m, n, k, &a, &b, &mut want);
            let mut got = vec![0.0; m * n];
            run(&variant, m, n, k, &a, &b, &mut got);
            let exact = matches!(
                variant.as_str(),
                "naive_ijk" | "naive_kij" | "naive_ikj" | "tiled4x4"
            );
            let worst = want
                .iter()
                .zip(got.iter())
                .map(|(w, g)| (w - g).abs())
                .fold(0.0f64, f64::max);
            let bit_identical = want
                .iter()
                .zip(got.iter())
                .all(|(w, g)| w.to_bits() == g.to_bits());
            if exact {
                assert!(
                    bit_identical,
                    "{variant} m={m} n={n} k={k}: expected BIT-identical to naive_ijk \
                     (no mul_add involved), worst abs diff {worst:.3e}"
                );
            } else {
                assert!(
                    worst <= 1e-9,
                    "{variant} m={m} n={n} k={k}: differs from naive_ijk by {worst:.3e}"
                );
            }
            println!(
                "check {variant} m={m} n={n} k={k}: worst_abs_diff={worst:.3e} \
                 bit_identical={bit_identical}"
            );
        }
    }

    // ---- the counted region ----------------------------------------------------------------
    let mut flops = 0.0f64;
    for &(m, n, k) in shapes {
        let a = black_box(fill(m * k, 0.3));
        let b = black_box(fill(k * n, 1.1));
        let mut c = vec![0.0; m * n];
        for _ in 0..reps {
            run(&variant, m, n, k, black_box(&a), black_box(&b), &mut c);
            black_box(&c);
        }
        // 2 flops per multiply-add, plus the m*n subtraction the update itself performs.
        if variant != "setup" {
            flops += reps as f64 * (2.0 * (m * n * k) as f64 + (m * n) as f64);
        }
    }
    println!("variant={variant} shapes={shape_set} reps={reps} analytic_flops={flops:.0}");
}
