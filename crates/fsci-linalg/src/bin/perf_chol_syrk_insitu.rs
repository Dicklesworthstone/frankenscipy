//! `frankenscipy-gykw5`: where do the missing 13–21 points go?
//!
//! Six rows on that bead compared WHOLE-FACTOR times and could only report that threading the
//! trailing update buys nothing. The accounting row narrowed it: the two updates the 32M gate
//! opens are 47.8% of an n=832 factor, and threading them at the isolated 1.3x should give a
//! whole-factor 1.133x, yet the in-situ measurement is exactly 1.000x.
//!
//! Two explanations remained, and one — cold panels — was refuted by rotating the microbench
//! through independent panel sets (speedup stayed 1.28–1.36x across a 16-fold cache-pressure
//! change). What is left is the CALL CONTEXT: the factor dispatches six times, once per panel,
//! interleaved with the panel factor and TRSM; the microbench dispatches in a tight loop.
//!
//! This settles it by measuring the SYRK portion IN THE FACTOR directly, via
//! `CHOL_SYRK_NANOS` / `CHOL_SYRK_CALLS`, at one thread and at N. It answers a question no
//! whole-factor ratio can:
//!
//!   * SYRK nanos DROP ~1.3x but the factor does not  → the saved time is being given back
//!     somewhere else, and the arithmetic in the accounting row is wrong about SYRK's share.
//!   * SYRK nanos DO NOT drop                          → the threading genuinely does not happen
//!     in situ, and the call context is the cause. That is a different bug from a slow kernel.
//!
//! The instrumentation is admissible as COST because it wraps a whole trailing update (~1.5 ms)
//! and runs `n/nb` times per factor — two clock reads per call is a part in ~10^5. This repo
//! has correctly ruled out `cfg(test)` counters for cost measurement; those sit inside hot
//! loops and act as optimisation barriers. This does not.

#[cfg(not(unix))]
fn main() {
    eprintln!("perf_chol_syrk_insitu requires a unix host");
    std::process::exit(1);
}

#[cfg(unix)]
fn main() {
    use fsci_linalg::{
        CHOL_SYRK_CALLS, CHOL_SYRK_NANOS, DecompOptions, MATMUL_PAR_DISPATCHES,
        MATMUL_PAR_MACS_OVERRIDE, MATMUL_PAR_MAX_THREADS_OVERRIDE, cholesky,
    };
    use std::hint::black_box;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let n = env_usize("INSITU_N", 832);
    let reps = env_usize("INSITU_REPS", 5);
    let cycles = env_usize("INSITU_CYCLES", 7);
    let gate: u64 = std::env::var("INSITU_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(33_554_432);

    // SPD fixture, same construction as perf_chol_vs_scipy so the sizes are comparable.
    let mut s = 0x5eed_c0de_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mm: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| next()).collect()).collect();
    let mut a = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..n).map(|t| mm[i][t] * mm[j][t]).sum();
            let v = if i == j { dot + n as f64 } else { dot };
            a[i][j] = v;
            a[j][i] = v;
        }
    }

    // One factor at a given worker cap, returning (total wall, syrk nanos, syrk calls, dispatches).
    let run = |cap: usize| -> (f64, u64, u64, usize) {
        MATMUL_PAR_MACS_OVERRIDE.store(gate, Ordering::Relaxed);
        MATMUL_PAR_MAX_THREADS_OVERRIDE.store(cap, Ordering::Relaxed);
        CHOL_SYRK_NANOS.store(0, Ordering::Relaxed);
        CHOL_SYRK_CALLS.store(0, Ordering::Relaxed);
        MATMUL_PAR_DISPATCHES.store(0, Ordering::Relaxed);
        let t0 = Instant::now();
        for _ in 0..reps {
            let out = cholesky(black_box(&a), true, DecompOptions::default()).expect("spd");
            black_box(out.factor[0][0]);
        }
        let wall = t0.elapsed().as_secs_f64();
        let out = (
            wall,
            CHOL_SYRK_NANOS.load(Ordering::Relaxed),
            CHOL_SYRK_CALLS.load(Ordering::Relaxed),
            MATMUL_PAR_DISPATCHES.load(Ordering::Relaxed),
        );
        MATMUL_PAR_MAX_THREADS_OVERRIDE.store(0, Ordering::Relaxed);
        MATMUL_PAR_MACS_OVERRIDE.store(0, Ordering::Relaxed);
        out
    };

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "n={n} gate={gate} reps={reps} cycles={cycles} host_threads={threads} loadavg={}",
        load.trim()
    );

    // Warm-up outside every timer.
    black_box(run(1));

    let (mut w1, mut wn) = (f64::INFINITY, f64::INFINITY);
    let (mut s1, mut sn) = (u64::MAX, u64::MAX);
    let (mut d1, mut dn) = (0usize, 0usize);
    let (mut c1, mut cn) = (0u64, 0u64);
    // ABBA within each cycle so drift cannot land on one arm.
    for _ in 0..cycles {
        for cap in [1usize, threads, threads, 1usize] {
            let (wall, nanos, calls, disp) = run(cap);
            if cap == 1 {
                if wall < w1 {
                    w1 = wall;
                }
                if nanos < s1 {
                    s1 = nanos;
                }
                d1 = disp;
                c1 = calls;
            } else {
                if wall < wn {
                    wn = wall;
                }
                if nanos < sn {
                    sn = nanos;
                }
                dn = disp;
                cn = calls;
            }
        }
    }

    let ms = |x: f64| x * 1000.0 / reps as f64;
    let nms = |x: u64| x as f64 / 1e6 / reps as f64;
    println!("cap=1   factor={:.4}ms syrk={:.4}ms calls={c1} dispatches={d1}", ms(w1), nms(s1));
    println!("cap={threads}  factor={:.4}ms syrk={:.4}ms calls={cn} dispatches={dn}", ms(wn), nms(sn));
    println!(
        "syrk_share_cap1={:.1}%  syrk_share_capN={:.1}%",
        nms(s1) / ms(w1) * 100.0,
        nms(sn) / ms(wn) * 100.0
    );
    println!(
        "SPEEDUP syrk={:.3}x factor={:.3}x  (>1 = threading helped)",
        nms(s1) / nms(sn),
        ms(w1) / ms(wn)
    );
    // What the factor speedup WOULD be if the measured syrk gain were the only change.
    let non_syrk = ms(w1) - nms(s1);
    println!(
        "predicted_factor_from_syrk_gain={:.3}x  (non-syrk time held at {:.4}ms)",
        ms(w1) / (non_syrk + nms(sn)),
        non_syrk
    );
}
