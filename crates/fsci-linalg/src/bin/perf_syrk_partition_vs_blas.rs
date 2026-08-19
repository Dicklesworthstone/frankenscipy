//! `frankenscipy-gykw5`: is the Cholesky basin a BANDWIDTH ceiling or a PARTITION ceiling?
//!
//! Six parallel-dispatch levers on that bead are neutral or worse, and the gap decomposes to
//! ~5 points of serial kernel rate plus ~40 points of threading SciPy gets (1.43x at n=832) and
//! we do not (1.00x). That leaves one question, and it is not a knob: when OpenBLAS threads a
//! trailing update of this exact shape, does IT get a speedup?
//!
//! Two possible answers, and they point at completely different work:
//!
//!   * dsyrk also fails to speed up at this shape  → the ceiling is memory bandwidth, the basin
//!     is not closable by ANY partitioning, and gykw5 should be closed as a wall.
//!   * dsyrk speeds up and we do not               → the ceiling is OUR PARTITION. We hand each
//!     worker a stripe of the same shared trailing matrix so every worker streams the same
//!     memory; OpenBLAS packs panels that stay resident in each core's L2. The fix is a packed
//!     2D split, which is real work but has a known shape.
//!
//! THE SHAPE IS NOT INVENTED. It is the first trailing update of a Cholesky at n=832 with the
//! shipped nb=128: C(704x704) -= A(704x128) * A(704x128)^T, 63.4M MACs — the update that misses
//! the 64M gate by 1% and is therefore the one the whole basin turns on.
//!
//! BOTH ARMS ARE TIMED AT MATCHED THREAD COUNTS in one invocation: each side is measured at 1
//! thread and at N, and what is compared is each side's OWN threading speedup. That is the
//! comparison that answers the question — an absolute ratio between a Rust GEMM and OpenBLAS
//! would confound partition efficiency with microkernel quality, which is exactly what the
//! serial arm already told us is only worth ~5%.

#[cfg(not(unix))]
fn main() {
    eprintln!("perf_syrk_partition_vs_blas requires a unix host");
    std::process::exit(1);
}

#[cfg(unix)]
fn main() {
    use fsci_linalg::{bench_trailing_syrk_prepare, bench_trailing_syrk_run};
    use std::hint::black_box;
    use std::process::Command;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let m = env_usize("SYRK_M", 704); // trailing rows at n=832, nb=128
    let k = env_usize("SYRK_K", 128); // panel width
    let reps = env_usize("SYRK_REPS", 5);
    let min_of = env_usize("SYRK_MIN_OF", 9);
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

    // Deterministic A (m x k) and its transpose, so C = A * A^T is the SYRK shape.
    let mut s = 0x5eed_c0de_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let a: Vec<Vec<f64>> = (0..m).map(|_| (0..k).map(|_| next()).collect()).collect();
    let at: Vec<Vec<f64>> = (0..k).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();

    // OUR trailing SYRK, at the same shape, timed in isolation via the bench entry point.
    // This is the arm the first version of this binary got wrong: it timed the public `matmul`,
    // which is a general GEMM on a different gate and is not what the factor calls. This calls
    // the SAME kernel the factor uses, at the SAME shape, so the self-scaling ratio is real.
    // Fixture built ONCE, outside every timer. `trailing` is destructively updated, so each
    // timed iteration works on a fresh clone — the clone is inside the timer for both arms
    // equally and is far smaller than the update itself.
    let (trailing0, l21, l21t) = bench_trailing_syrk_prepare(m, k, 0xC0DE);

    let time_ours = |nthreads: usize| -> (f64, f64) {
        let mut warm = trailing0.clone();
        let fold = bench_trailing_syrk_run(&mut warm, &l21, &l21t, m, k, nthreads);
        black_box(fold);
        let mut best = f64::INFINITY;
        for _ in 0..min_of {
            let t0 = Instant::now();
            for _ in 0..reps {
                let mut t = trailing0.clone();
                black_box(bench_trailing_syrk_run(&mut t, &l21, &l21t, m, k, nthreads));
            }
            let dt = t0.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
        }
        (best, fold)
    };

    const ORACLE: &str = r#"
import os, sys, time
import numpy as np
import scipy
m, k, reps, min_of = (int(os.environ[v]) for v in ("SYRK_M","SYRK_K","SYRK_REPS","SYRK_MIN_OF"))
rng = np.random.default_rng(0xC0DE)
A = np.asfortranarray(rng.standard_normal((m, k)))
from scipy.linalg.blas import dsyrk
C = dsyrk(1.0, A)            # warm-up, outside every timer
best = float("inf")
for _ in range(min_of):
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        C = dsyrk(1.0, A)
    dt = (time.perf_counter_ns() - t0) * 1e-9
    best = min(best, dt)
print(f"seconds={best:.9f} threads_env={os.environ.get('OPENBLAS_NUM_THREADS','unset')} "
      f"scipy={scipy.__version__} numpy={np.__version__} shape={C.shape} "
      f"tasks={len(os.listdir('/proc/self/task'))}", flush=True)
"#;

    let time_blas = |nthreads: usize| -> String {
        let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| {
            if std::path::Path::new("/usr/bin/python3.13").exists() {
                "/usr/bin/python3.13".into()
            } else {
                "python3".into()
            }
        });
        let mut c = Command::new(&python);
        let site = "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
        if std::path::Path::new(site).is_dir() {
            c.env("PYTHONPATH", site);
        }
        for key in [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ] {
            c.env(key, nthreads.to_string());
        }
        let out = c
            .arg("-u")
            .arg("-c")
            .arg(ORACLE)
            .env("SYRK_M", m.to_string())
            .env("SYRK_K", k.to_string())
            .env("SYRK_REPS", reps.to_string())
            .env("SYRK_MIN_OF", min_of.to_string())
            .output()
            .expect("spawn dsyrk arm");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "shape m={m} k={k} macs={:.1}M reps={reps} min_of={min_of} host_threads={threads} \
         loadavg={}",
        (m * k * m) as f64 / 1e6,
        load.trim()
    );

    // ABBA over the four cells so drift cannot land on one side.
    let (ours_1a, fold1) = time_ours(1);
    let blas_1a = time_blas(1);
    let (ours_na, foldn) = time_ours(threads);
    let blas_na = time_blas(threads);
    let blas_nb = time_blas(threads);
    let (ours_nb, _) = time_ours(threads);
    let blas_1b = time_blas(1);
    let (ours_1b, _) = time_ours(1);

    let parse = |s: &str| -> f64 {
        s.split_whitespace()
            .find_map(|f| f.strip_prefix("seconds="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN)
    };
    let o1 = ours_1a.min(ours_1b);
    let on = ours_na.min(ours_nb);
    let b1 = parse(&blas_1a).min(parse(&blas_1b));
    let bn = parse(&blas_na).min(parse(&blas_nb));

    println!("ours_1thread={o1:.9}s fold={fold1:.9e}");
    println!("ours_nthread={on:.9}s fold={foldn:.9e}");
    // The two arms must compute the SAME result, or the ratio compares two computations.
    assert!(
        (fold1 - foldn).abs() <= 1e-9 * fold1.abs().max(1.0),
        "serial and threaded SYRK disagree: {fold1:.17e} vs {foldn:.17e}"
    );
    println!("blas_1thread={b1:.9}s  [{}]", blas_1a);
    println!("blas_nthread={bn:.9}s  [{}]", blas_na);
    // A/A nulls: the same configuration timed twice within the interleave.
    println!(
        "null_ours_1={:.4} null_ours_n={:.4}",
        ours_1a.max(ours_1b) / ours_1a.min(ours_1b),
        ours_na.max(ours_nb) / ours_na.min(ours_nb)
    );
    println!(
        "null_blas_1={:.4} null_blas_n={:.4}",
        parse(&blas_1a).max(parse(&blas_1b)) / parse(&blas_1a).min(parse(&blas_1b)),
        parse(&blas_na).max(parse(&blas_nb)) / parse(&blas_na).min(parse(&blas_nb))
    );
    // THE ANSWER: each side's OWN threading speedup at the SAME shape, same invocation.
    println!(
        "THREADING_SPEEDUP ours={:.3}x blas={:.3}x  (>1 = threading helps that side)",
        o1 / on,
        b1 / bn
    );
    println!(
        "serial_rate_ratio ours_1/blas_1={:.3}x  (>1 = OpenBLAS faster single-threaded; note \
         dsyrk computes only the TRIANGLE while this kernel updates the full trailing block, so \
         a factor ~2 of this is shape, not rate)",
        o1 / b1
    );
}
