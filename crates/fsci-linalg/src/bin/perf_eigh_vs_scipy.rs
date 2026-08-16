//! Dense symmetric `eigh` versus a LIVE SciPy 1.17.1 incumbent, in ONE invocation.
//!
//! WHY THIS BINARY EXISTS. `frankenscipy-ll0kk` recorded eigh at 1.32x (n=256) and
//! 2.73x (n=512) SLOWER than scipy, and `frankenscipy-2o0vp` then rejected the
//! WY-blocked reduction. Every one of those numbers came from a remembered SciPy
//! figure captured in a separate run. This harness re-decides the gap under the
//! campaign's evidence rule: the incumbent runs live, on this host, inside the same
//! process tree, on the SAME matrix bytes, interleaved round-by-round with our arm.
//!
//! Contract (mirrors `perf_bdf_diag_newton`, plus the incumbent):
//!   1. line 1 is the SHA-256 of this binary's own ELF, read from `/proc/self/exe`
//!      inside the process — a shell-side hash cannot prove which build ran;
//!   2. provenance is OBSERVED, not requested: peak thread count is sampled from
//!      `/proc/self/task` by a poller thread on both sides, never assumed from a
//!      `*_NUM_THREADS` variable;
//!   3. both SciPy arms are screened: `scipy1` has every BLAS thread-count variable
//!      pinned to 1, `scipyN` is left at the deployment default. Reporting only the
//!      pinned arm would flatter a multi-threaded Rust arm;
//!   4. AGREEMENT BEFORE TIMING: our eigenvalues are compared against SciPy's on the
//!      same bytes and the run aborts if they disagree beyond the tolerance contract
//!      (`eigh` is tolerance-based, not bit-exact) — a fast arm that solved a
//!      different problem is not a result;
//!   5. every ratio is paired with its A/A null (`fsci`/`fsci` and `scipy1`/`scipy1`)
//!      measured in the SAME invocation; a null that misses 1.0 invalidates the row.
//!
//! Run: `cargo run --release --bin perf_eigh_vs_scipy --features eigh-incumbent-bench -- [sizes] [rounds] [min_of]`
//!   e.g. `... -- 256,512,768,1024 9 2`

#[cfg(feature = "eigh-incumbent-bench")]
mod bench {
    use fsci_linalg::{DecompOptions, PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE, eigh};
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Instant;

    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";

    /// Relative eigenvalue agreement our native path contracts for (it is a
    /// tolerance-based solver, so bit-equality with LAPACK is not on offer).
    const MAX_EIGENVALUE_REL_DIFF: f64 = 1.0e-8;

    const PYTHON_ORACLE: &str = r#"
import hashlib
import os
import sys
import threading
import time

import numpy as np
import scipy
from scipy import linalg
import scipy.linalg._decomp as _decomp

N = int(os.environ["FSCI_EIGH_N"])
EXPECTED_BYTES = N * N * 8
raw = sys.stdin.buffer.read(EXPECTED_BYTES)
if len(raw) != EXPECTED_BYTES:
    raise RuntimeError(f"fixture byte count mismatch: expected {EXPECTED_BYTES}, got {len(raw)}")
FIXTURE_SHA256 = hashlib.sha256(raw).hexdigest()
A = np.frombuffer(raw, dtype="<f8").reshape(N, N).copy()
if not np.array_equal(A, A.T):
    raise RuntimeError("fixture is not exactly symmetric")

PEAK_TASKS = len(os.listdir("/proc/self/task"))
STOP = threading.Event()

def poll_tasks():
    global PEAK_TASKS
    while not STOP.is_set():
        count = len(os.listdir("/proc/self/task"))
        if count > PEAK_TASKS:
            PEAK_TASKS = count
        STOP.wait(0.002)

POLLER = threading.Thread(target=poll_tasks, daemon=True)
POLLER.start()

def run():
    return linalg.eigh(A)

# Warmup is outside every timer: first call pays BLAS buffer allocation.
W, V = run()
if W.shape != (N,) or V.shape != (N, N):
    raise RuntimeError("unexpected eigh output shape")

try:
    engine_path = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
except Exception:
    engine_path = "unknown"
decomp_path = _decomp.__file__
with open(decomp_path, "rb") as handle:
    DECOMP_SHA256 = hashlib.sha256(handle.read()).hexdigest()
fsci_loaded = any(name.startswith("fsci") or name.startswith("frankenscipy") for name in sys.modules)

print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" blas={engine_path}"
    f" decomp_module={_decomp.__name__}"
    f" decomp_sha256={DECOMP_SHA256}"
    f" fixture_sha256={FIXTURE_SHA256}"
    f" n={N}"
    f" affinity={len(os.sched_getaffinity(0))}"
    f" cpus_allowed={sorted(os.sched_getaffinity(0))}"
    f" peak_tasks={PEAK_TASKS}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={scipy.__version__ == '1.17.1' and not fsci_loaded}",
    flush=True,
)

for raw_line in sys.stdin.buffer:
    fields = str(raw_line, "ascii").strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "EVALS":
        bits = np.asarray(run()[0], dtype="<f8").view("<u8")
        print("EVALS " + ",".join(f"{int(v):016x}" for v in bits), flush=True)
    elif command == "TIME" and len(fields) == 3:
        reps = int(fields[1])
        min_of = int(fields[2])
        if reps <= 0 or min_of <= 0:
            raise RuntimeError("reps and min_of must be positive")
        best = float("inf")
        checksum = 0.0
        for _ in range(min_of):
            started = time.perf_counter_ns()
            for _ in range(reps):
                w, v = run()
                checksum += float(w[0]) + float(v[0, 0])
            elapsed = (time.perf_counter_ns() - started) * 1.0e-9
            if elapsed < best:
                best = elapsed
        print(f"TIME {best:.17e} {checksum:.17e} {PEAK_TASKS}", flush=True)
    elif command == "QUIT":
        STOP.set()
        print(f"BYE {PEAK_TASKS}", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {fields!r}")
"#;

    /// A live SciPy child holding one fixture matrix.
    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
        label: &'static str,
    }

    impl Scipy {
        fn start(label: &'static str, n: usize, bytes: &[u8], pin_threads: bool) -> (Self, String) {
            let python = std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| {
                if std::path::Path::new("/usr/bin/python3.13").exists() {
                    "/usr/bin/python3.13".to_string()
                } else {
                    "python3".to_string()
                }
            });
            let mut command = Command::new(&python);
            // Prefer the campaign's pinned, screened SciPy 1.17.1 when this host has
            // it; fall back to the system interpreter's scipy (the READY line reports
            // the version either way, so the arm is never anonymous).
            if std::path::Path::new(SCIPY_SITE_PACKAGES).is_dir() {
                command.env("PYTHONPATH", SCIPY_SITE_PACKAGES);
            }
            command
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("FSCI_EIGH_N", n.to_string())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());
            if pin_threads {
                for key in [
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ] {
                    command.env(key, "1");
                }
            }
            let mut child = command
                .spawn()
                .unwrap_or_else(|error| panic!("spawn live SciPy oracle {python}: {error}"));
            let mut stdin = child.stdin.take().expect("SciPy oracle stdin");
            stdin
                .write_all(bytes)
                .and_then(|()| stdin.flush())
                .expect("send fixture to SciPy");
            let mut stdout = BufReader::new(child.stdout.take().expect("SciPy oracle stdout"));
            let mut ready = String::new();
            stdout.read_line(&mut ready).expect("read SciPy identity");
            assert!(
                ready.starts_with("READY"),
                "live SciPy oracle did not announce itself: {ready:?}"
            );
            (
                Self {
                    child,
                    stdin,
                    stdout,
                    stopped: false,
                    label,
                },
                ready.trim().to_string(),
            )
        }

        fn request(&mut self, command: &str) -> String {
            writeln!(self.stdin, "{command}")
                .and_then(|()| self.stdin.flush())
                .unwrap_or_else(|error| panic!("send {command} to {}: {error}", self.label));
            let mut response = String::new();
            self.stdout
                .read_line(&mut response)
                .unwrap_or_else(|error| panic!("read {command} from {}: {error}", self.label));
            assert!(
                !response.is_empty(),
                "SciPy oracle {} exited during {command}",
                self.label
            );
            response.trim().to_string()
        }

        fn eigenvalues(&mut self) -> Vec<f64> {
            let response = self.request("EVALS");
            let payload = response
                .strip_prefix("EVALS ")
                .unwrap_or_else(|| panic!("malformed EVALS response from {}", self.label));
            payload
                .split(',')
                .map(|value| {
                    u64::from_str_radix(value, 16)
                        .map(f64::from_bits)
                        .unwrap_or_else(|error| panic!("parse SciPy eigenvalue bits: {error}"))
                })
                .collect()
        }

        /// Seconds for the best of `min_of` replicates of `reps` decompositions,
        /// timed by SciPy's own `perf_counter` so IPC never enters the incumbent's
        /// number (the conservative direction — it can only make SciPy look faster).
        fn time(&mut self, reps: usize, min_of: usize) -> (f64, usize) {
            let response = self.request(&format!("TIME {reps} {min_of}"));
            let fields: Vec<&str> = response.split_whitespace().collect();
            assert!(
                fields.len() == 4 && fields[0] == "TIME",
                "malformed TIME response from {}: {response}",
                self.label
            );
            let elapsed: f64 = fields[1].parse().expect("SciPy elapsed");
            let peak: usize = fields[3].parse().expect("SciPy peak tasks");
            assert!(
                elapsed.is_finite() && elapsed > 0.0,
                "invalid SciPy elapsed: {response}"
            );
            (elapsed, peak)
        }

        fn stop(&mut self) -> usize {
            if self.stopped {
                return 0;
            }
            let response = self.request("QUIT");
            let peak = response
                .strip_prefix("BYE ")
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("malformed shutdown from {}: {response}", self.label));
            let status = self.child.wait().expect("wait for SciPy oracle");
            assert!(status.success(), "SciPy oracle exited with {status}");
            self.stopped = true;
            peak
        }
    }

    impl Drop for Scipy {
        fn drop(&mut self) {
            if !self.stopped {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// Samples `/proc/self/task` until stopped; the maximum it sees is the OBSERVED
    /// thread count for our arm. A requested worker count proves nothing.
    struct ThreadPoller {
        stop: Arc<AtomicBool>,
        peak: Arc<AtomicUsize>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl ThreadPoller {
        fn start() -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            let peak = Arc::new(AtomicUsize::new(1));
            let (stop_worker, peak_worker) = (Arc::clone(&stop), Arc::clone(&peak));
            let handle = std::thread::spawn(move || {
                while !stop_worker.load(Ordering::Relaxed) {
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        let count = entries.count();
                        peak_worker.fetch_max(count, Ordering::Relaxed);
                    }
                    std::thread::sleep(std::time::Duration::from_micros(500));
                }
            });
            Self {
                stop,
                peak,
                handle: Some(handle),
            }
        }

        fn finish(mut self) -> usize {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            self.peak.load(Ordering::Relaxed)
        }
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        let len = values.len();
        if len % 2 == 1 {
            values[len / 2]
        } else {
            0.5 * (values[len / 2 - 1] + values[len / 2])
        }
    }

    fn quantile(values: &[f64], q: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[index]
    }

    pub struct Paired {
        pub p50_a: f64,
        pub p50_b: f64,
        pub ratio_p50: f64,
        pub ratio_lo: f64,
        pub ratio_hi: f64,
        pub cv: f64,
        pub rounds: usize,
    }

    fn summarize(ta: Vec<f64>, tb: Vec<f64>, ratios: Vec<f64>) -> Paired {
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let var = ratios.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / ratios.len() as f64;
        Paired {
            p50_a: median(ta),
            p50_b: median(tb),
            ratio_p50: median(ratios.clone()),
            ratio_lo: quantile(&ratios, 0.025),
            ratio_hi: quantile(&ratios, 0.975),
            cv: var.sqrt() / mean,
            rounds: ratios.len(),
        }
    }

    /// Sampled again after the sweep: a row whose loadavg climbed under it is
    /// not comparable with one measured on a quiet worker, and the fleet has
    /// seen whole boards fail to certify for exactly this reason.
    fn print_loadavg_post() {
        println!("loadavg_post={}", read_trimmed("/proc/loadavg"));
    }

    /// How far the effect sits from parity, in units of the widest A/A null that
    /// certifies it. The ledger requires at least 2.0.
    ///
    /// TAKES THE NULLS EXPLICITLY, and that is the entire point. The first
    /// version of this check computed one margin per CELL from the scipy
    /// comparison and printed `margin_ok=true` while the implementation
    /// comparison sitting beside it was failing the same test at ~0.6x
    /// (frankenscipy-ll0kk). A margin is a property of a COMPARISON paired with
    /// the nulls that bound it, never of the cell it happens to live in.
    ///
    /// Bracketing 1.0 is necessary and NOT sufficient: a null that is
    /// enormously wide also brackets 1.0 -- it passes precisely BECAUSE it is
    /// wide -- so a contended host yields cells that certify while resolving
    /// nothing.
    pub fn null_margin(effect: &Paired, nulls: &[&Paired]) -> f64 {
        let deviation = |p: &Paired| (p.ratio_lo - 1.0).abs().max((p.ratio_hi - 1.0).abs());
        let null_dev = nulls
            .iter()
            .map(|p| deviation(p))
            .fold(0.0_f64, f64::max);
        let effect_dev = (effect.ratio_p50 - 1.0).abs();
        if null_dev > 0.0 {
            effect_dev / null_dev
        } else {
            f64::INFINITY
        }
    }

    pub const MIN_NULL_MARGIN: f64 = 2.0;

    /// An incumbent arm this much slower than ours on a dense symmetric
    /// eigendecomposition is not a measurement, it is a broken arm.
    ///
    /// LAPACK `dsyevd` does not lose to a pure-Rust eigensolver by an order of
    /// magnitude. When `fsci/scipyN` came back at 0.0105x-0.0192x -- claiming we
    /// were 50-100x faster -- the arm was thrashing, not losing.
    const MIN_PLAUSIBLE_INCUMBENT_RATIO: f64 = 0.1;

    /// Whether the default-BLAS SciPy arm produced a number worth reporting.
    ///
    /// A ratio far below 1 means the INCUMBENT was pathologically slow, which
    /// says nothing about our code. Kept separate from oversubscription on
    /// purpose: those two are not the same thing, and conflating them was my
    /// error (frankenscipy-ll0kk). hz2 ran scipyN at 32 threads on a cpuset of
    /// 16 -- 2x oversubscribed -- and returned perfectly sane values (2.479x).
    /// vmi1227854 ran 20 threads on a cpuset of 10 -- the SAME 2x ratio -- and
    /// returned 0.0124x. The difference was not the subscription ratio but the
    /// host: vmi1227854 was carrying external load (loadavg 9.19 rising to
    /// 22.68 on 10 cores) while hz2 was quiet. Oversubscription is a risk
    /// factor; CONTENTION is what actually breaks the arm.
    pub fn scipyn_plausible(ratio_p50: f64) -> bool {
        ratio_p50 >= MIN_PLAUSIBLE_INCUMBENT_RATIO
    }

    /// Reported alongside, never instead: a warning, not a verdict.
    pub fn scipyn_oversubscribed(peak_tasks: usize, cpuset: usize) -> bool {
        cpuset > 0 && peak_tasks > cpuset
    }

    /// How much slower our arm runs while the incumbent's processes are
    /// RESIDENT, versus with the machine to itself.
    ///
    /// WHY THIS IS NOT COVERED BY THE A/A NULL, which is the whole point.
    /// franken_numpy confirmed that its own arm slows the incumbent it is
    /// measured against. The exposure here is the mirror image: our arm is timed
    /// while the SciPy subprocesses are alive, and OpenBLAS worker threads
    /// spin-wait after a parallel region rather than sleeping immediately, so
    /// scipyN's 20-32 threads can be burning cores during our timing even
    /// though the two arms never execute simultaneously. An A/A null cannot
    /// see this: both A arms are measured under the same resident load, so they
    /// agree with each other perfectly while both are depressed.
    ///
    /// A ratio above 1 means our arm is SLOWER with the incumbent resident,
    /// i.e. we are understating ourselves and overstating the deficit.
    pub fn contention_ratio(median_alone: f64, median_resident: f64) -> f64 {
        if median_alone > 0.0 {
            median_resident / median_alone
        } else {
            f64::NAN
        }
    }

    /// Which CPU each of our threads was last observed on.
    ///
    /// Field 39 of `/proc/<pid>/stat` is `processor`. `comm` may contain spaces
    /// and parentheses, so the fields are counted from the LAST `)`.
    pub fn observed_cpus_now() -> std::collections::BTreeSet<usize> {
        let mut cpus = std::collections::BTreeSet::new();
        if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
            for e in entries.flatten() {
                let stat = e.path().join("stat");
                if let Ok(text) = std::fs::read_to_string(&stat)
                    && let Some(tail) = text.rsplit_once(')').map(|(_, t)| t)
                {
                    // After "(comm)" the next field is `state`, so `processor`
                    // (field 39 overall) is index 36 of what remains.
                    if let Some(cpu) = tail.split_whitespace().nth(36)
                        && let Ok(cpu) = cpu.parse::<usize>()
                    {
                        cpus.insert(cpu);
                    }
                }
            }
        }
        cpus
    }

    /// `cpu MHz` for one processor, read from /proc/cpuinfo.
    fn cpuinfo_mhz(cpu: usize) -> Option<f64> {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        let mut current: Option<usize> = None;
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("processor") {
                current = v.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
                    .trim()
                    .parse()
                    .ok();
            } else if let Some(v) = line.strip_prefix("cpu MHz")
                && current == Some(cpu)
            {
                return v
                    .trim_start_matches(|c: char| c == ':' || c.is_whitespace())
                    .trim()
                    .parse()
                    .ok();
            }
        }
        None
    }

    fn sysfs_usize(path: String) -> Option<usize> {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| t.trim().parse().ok())
    }

    /// `(cpu, core_id, siblings, MHz)` for each CPU, so a row can show whether
    /// two arms landed on one physical core or on its SMT sibling.
    pub fn cpu_topology(cpus: &std::collections::BTreeSet<usize>) -> Vec<(usize, usize, String, f64)> {
        cpus.iter()
            .map(|&c| {
                let core = sysfs_usize(format!(
                    "/sys/devices/system/cpu/cpu{c}/topology/core_id"
                ))
                .unwrap_or(usize::MAX);
                let sibs = std::fs::read_to_string(format!(
                    "/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
                ))
                .map(|t| t.trim().to_string())
                .unwrap_or_else(|_| "?".into());
                // cpufreq sysfs is absent on some workers (governor=unavailable),
                // so fall back to /proc/cpuinfo, which reports "cpu MHz" per
                // processor regardless. Measured: vmi1293453 has no cpufreq.
                let mhz = sysfs_usize(format!(
                    "/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq"
                ))
                .map(|k| k as f64 / 1000.0)
                .or_else(|| cpuinfo_mhz(c))
                .unwrap_or(f64::NAN);
                (c, core, sibs, mhz)
            })
            .collect()
    }

    /// Do two arms occupy the same PHYSICAL core -- either the identical CPU or
    /// its SMT sibling?
    ///
    /// frankenfs found BOTH its arms pinned to one physical core, and
    /// frankenscipy voided rows over it. Two arms on SMT siblings share
    /// execution units, so each depresses the other while every A/A null stays
    /// perfectly happy -- the same blind spot as resident-process contention,
    /// one level down in the hardware. Inputs are `(cpu, core_id)` pairs.
    pub fn arms_share_physical_core(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
        a.iter().any(|(_, core_a)| {
            *core_a != usize::MAX && b.iter().any(|(_, core_b)| core_b == core_a)
        })
    }

    /// Contention is material once it is comparable to the effects being
    /// reported. Deficits here are quoted to two decimal places, so a 5%
    /// cross-arm effect is already large enough to move a verdict.
    pub const MAX_TOLERABLE_CONTENTION: f64 = 1.05;

    fn report(label: &str, p: &Paired) {
        println!(
            "{label:<16} a={:9.3}ms b={:9.3}ms ratio_p50={:.4}x ci95=[{:.4},{:.4}] cv={:.2}% rounds={}",
            p.p50_a * 1e3,
            p.p50_b * 1e3,
            p.ratio_p50,
            p.ratio_lo,
            p.ratio_hi,
            p.cv * 100.0,
            p.rounds
        );
    }

    /// Best of `min_of` replicates of `reps` `eigh` calls, in seconds.
    /// Time `eigh` with the implementation pinned for THIS measurement.
    ///
    /// `usize::MAX` pins nalgebra's `symmetric_eigen` (no n reaches it), `1`
    /// pins `symmetric_eigh_native`. Setting the override per measurement rather
    /// than per cell is the whole point (frankenscipy-ll0kk): the previous
    /// design compared the two implementations ACROSS cells run minutes apart,
    /// with no pairing and no A/A null spanning the comparison, and the observed
    /// ordering at n=512 reversed between two runs of the same ELF on the same
    /// worker. Drift over those minutes exceeded the effect.
    fn time_fsci_impl(a: &[Vec<f64>], min_of: usize, min_dim: usize) -> f64 {
        PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE.store(min_dim, std::sync::atomic::Ordering::Relaxed);
        let t = time_fsci(a, 1, min_of);
        PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        t
    }

    fn time_fsci(a: &[Vec<f64>], reps: usize, min_of: usize) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..min_of {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(eigh(black_box(a), DecompOptions::default()).expect("fsci eigh"));
            }
            let secs = started.elapsed().as_secs_f64();
            if secs < best {
                best = secs;
            }
        }
        best
    }

    fn read_trimmed(path: &str) -> String {
        std::fs::read_to_string(path).map_or_else(
            |_| "unavailable".to_string(),
            |text| text.trim().to_string(),
        )
    }

    fn symmetric_fixture(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<u8>) {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        // Upper triangle drawn once, then mirrored, so the fixture is EXACTLY
        // symmetric bit-for-bit — the Python side rejects it otherwise.
        let mut upper = vec![vec![0.0_f64; n]; n];
        for (row, cells) in upper.iter_mut().enumerate() {
            for cell in cells.iter_mut().skip(row) {
                *cell = next();
            }
        }
        let mut a = vec![vec![0.0_f64; n]; n];
        for row in 0..n {
            for col in 0..n {
                a[row][col] = if col >= row {
                    upper[row][col]
                } else {
                    upper[col][row]
                };
            }
        }
        let mut bytes = Vec::with_capacity(n * n * 8);
        for row in &a {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        (a, bytes)
    }

    pub fn run() {
        let exe = std::env::current_exe().expect("current_exe");
        let sha = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&exe).expect("read own ELF"))
        );
        println!("elf_sha256={sha}");
        println!("elf_path={}", exe.display());

        let args: Vec<String> = std::env::args().collect();
        let sizes: Vec<usize> = args
            .get(1)
            .map_or_else(
                || "256,512,768,1024".to_string(),
                std::string::ToString::to_string,
            )
            .split(',')
            .filter_map(|value| value.trim().parse().ok())
            .collect();
        let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
        let min_of: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
        let seed: u64 = args
            .get(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0x2468_ace0_1357_9bdf);
        assert!(!sizes.is_empty(), "no sizes parsed");

        println!(
            "host={} governor={} avx2={} avx512f={} fma={} affinity={} nproc={} loadavg_pre={} rounds={rounds} min_of={min_of} seed={seed:#x}",
            read_trimmed("/proc/sys/kernel/hostname"),
            read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            std::arch::is_x86_feature_detected!("avx2"),
            std::arch::is_x86_feature_detected!("avx512f"),
            std::arch::is_x86_feature_detected!("fma"),
            std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
            std::fs::read_to_string("/proc/cpuinfo")
                .map(|c| c.matches("processor\t:").count())
                .unwrap_or(0),
            read_trimmed("/proc/loadavg"),
        );

        // frankenscipy-ll0kk. The banked 1.32x (n=256) / 2.73x (n=512) pair was
        // read as a growing ratio and therefore a scaling gap needing Cuppen
        // divide-and-conquer. But `eigh` switches implementation at
        // PUBLIC_NATIVE_EIGH_MIN_DIM = 512: n=256 was nalgebra's symmetric_eigen
        // and n=512 was symmetric_eigh_native. Those two points came from two
        // different algorithms, and a scaling law cannot be inferred across an
        // implementation switch.
        //
        // So sweep the implementation as a FIRST-CLASS dimension: force each one
        // at every size and label every row with which one ran. usize::MAX pins
        // nalgebra (no n reaches it); 1 pins native (every n reaches it).
        //
        // Read the result this way: if native-at-256 is already ~2x scipy, the
        // jump is an implementation cliff and Cuppen is the wrong target. Only
        // if native's own ratio worsens from 256 to 512 does the
        // kernel-quality-wall verdict stand.
        let cells: Vec<(&str, usize, usize)> = sizes
            .iter()
            .flat_map(|&n| [("nalgebra", usize::MAX, n), ("native", 1usize, n)])
            .collect();
        for &(impl_label, min_dim_override, n) in &cells {
            PUBLIC_NATIVE_EIGH_MIN_DIM_OVERRIDE
                .store(min_dim_override, std::sync::atomic::Ordering::Relaxed);
            let (a, bytes) = symmetric_fixture(n, seed);
            // CROSS-ARM CONTENTION PROBE (frankenscipy-ll0kk): our arm timed with
            // the machine to itself, BEFORE any SciPy process exists.
            let mut alone: Vec<f64> = (0..rounds).map(|_| time_fsci(&a, 1, min_of)).collect();
            // Placement is sampled right after a timed run, while the worker
            // threads are still on the CPUs they used.
            let fsci_cpus = {
                use std::sync::atomic::{AtomicBool, Ordering as O};
                use std::sync::{Arc, Mutex};
                let seen = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
                let stop = Arc::new(AtomicBool::new(false));
                let (s2, st2) = (Arc::clone(&seen), Arc::clone(&stop));
                let sampler = std::thread::spawn(move || {
                    while !st2.load(O::Relaxed) {
                        if let Ok(mut g) = s2.lock() {
                            g.extend(observed_cpus_now());
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                });
                // One extra timed call purely so the worker threads are ALIVE
                // while the sampler runs; its duration is discarded.
                let _ = time_fsci(&a, 1, min_of);
                stop.store(true, O::Relaxed);
                let _ = sampler.join();
                let out = seen.lock().map(|g| g.clone()).unwrap_or_default();
                out
            };

            let (mut scipy1, ready1) = Scipy::start("scipy1", n, &bytes, true);
            let (mut scipyn, readyn) = Scipy::start("scipyN", n, &bytes, false);
            println!("n={n} impl={impl_label} {ready1}");
            println!("n={n} impl={impl_label} {readyn}");

            // ── AGREEMENT BEFORE TIMING ──────────────────────────────────────────
            let ours = eigh(&a, DecompOptions::default()).expect("fsci eigh");
            let theirs = scipy1.eigenvalues();
            let theirs_n = scipyn.eigenvalues();
            assert_eq!(ours.eigenvalues.len(), theirs.len(), "eigenvalue count");
            let scale = theirs
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let (mut worst, mut worst_index) = (0.0_f64, 0usize);
            for (index, (&mine, &lapack)) in ours.eigenvalues.iter().zip(&theirs).enumerate() {
                let diff = (mine - lapack).abs() / scale;
                if diff > worst {
                    worst = diff;
                    worst_index = index;
                }
            }
            let cross_arm = theirs
                .iter()
                .zip(&theirs_n)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "n={n} agreement: worst_rel_diff={worst:.3e} at index {worst_index} \
                 scale={scale:.6e} scipy1_vs_scipyN_differing_bits={cross_arm}"
            );
            assert!(
                worst <= MAX_EIGENVALUE_REL_DIFF,
                "n={n}: fsci and live SciPy disagree ({worst:.3e} > {MAX_EIGENVALUE_REL_DIFF:.1e}) \
                 — no timing is admissible"
            );

            // ── A/A NULLS, then the incumbent ratios, all in this invocation ─────
            let poller = ThreadPoller::start();
            let (mut fa, mut fb, mut fr) = (vec![], vec![], vec![]);
            let (mut na, mut nb, mut nr) = (vec![], vec![], vec![]);
            let (mut c1a, mut c1b, mut c1r) = (vec![], vec![], vec![]);
            let (mut cna, mut cnb, mut cnr) = (vec![], vec![], vec![]);
            // Paired nalgebra-vs-native, and its OWN A/A null (nalgebra twice).
            let (mut ia, mut ib, mut ir) = (vec![], vec![], vec![]);
            let (mut ina, mut inb, mut inr) = (vec![], vec![], vec![]);
            let mut scipy1_peak = 0usize;
            let mut scipyn_peak = 0usize;
            for round in 0..rounds {
                // fsci A/A null
                let a1 = time_fsci(&a, 1, min_of);
                let a2 = time_fsci(&a, 1, min_of);
                fa.push(a1);
                fb.push(a2);
                fr.push(a1 / a2);

                // scipy1 A/A null
                let (s1, p1) = scipy1.time(1, min_of);
                let (s2, p2) = scipy1.time(1, min_of);
                scipy1_peak = scipy1_peak.max(p1).max(p2);
                na.push(s1);
                nb.push(s2);
                nr.push(s1 / s2);

                // Candidate rows, order alternating so drift hits both arms equally.
                let (ours_t, pinned_t, default_t) = if round % 2 == 0 {
                    let ours_t = time_fsci(&a, 1, min_of);
                    let (pinned_t, p) = scipy1.time(1, min_of);
                    let (default_t, pn) = scipyn.time(1, min_of);
                    scipy1_peak = scipy1_peak.max(p);
                    scipyn_peak = scipyn_peak.max(pn);
                    (ours_t, pinned_t, default_t)
                } else {
                    let (default_t, pn) = scipyn.time(1, min_of);
                    let (pinned_t, p) = scipy1.time(1, min_of);
                    let ours_t = time_fsci(&a, 1, min_of);
                    scipy1_peak = scipy1_peak.max(p);
                    scipyn_peak = scipyn_peak.max(pn);
                    (ours_t, pinned_t, default_t)
                };
                c1a.push(ours_t);
                c1b.push(pinned_t);
                c1r.push(ours_t / pinned_t);
                cna.push(ours_t);
                cnb.push(default_t);
                cnr.push(ours_t / default_t);

                // Implementation A/A null FIRST, so it is measured under the
                // same conditions as the A/B it certifies: nalgebra against
                // itself, back to back, inside this round.
                let z1 = time_fsci_impl(&a, min_of, usize::MAX);
                let z2 = time_fsci_impl(&a, min_of, usize::MAX);
                ina.push(z1);
                inb.push(z2);
                inr.push(z1 / z2);

                // Paired nalgebra vs native, A/B/B/A across rounds so drift
                // hits both implementations equally.
                let (nalg_t, nat_t) = if round % 2 == 0 {
                    let x = time_fsci_impl(&a, min_of, usize::MAX);
                    let y = time_fsci_impl(&a, min_of, 1);
                    (x, y)
                } else {
                    let y = time_fsci_impl(&a, min_of, 1);
                    let x = time_fsci_impl(&a, min_of, usize::MAX);
                    (x, y)
                };
                ia.push(nalg_t);
                ib.push(nat_t);
                ir.push(nalg_t / nat_t);
            }
            let fsci_peak_tasks = poller.finish();
            scipy1_peak = scipy1_peak.max(scipy1.stop());
            scipyn_peak = scipyn_peak.max(scipyn.stop());

            // Second alone-sample AFTER the incumbent is gone. Taking it on both
            // sides is what separates residency from drift over the cell: if the
            // machine simply got slower, pre and post disagree with each other
            // too, and the probe says nothing.
            alone.extend((0..rounds).map(|_| time_fsci(&a, 1, min_of)));

            let fsci_null = summarize(fa, fb, fr);
            let scipy_null = summarize(na, nb, nr);
            let vs_pinned = summarize(c1a, c1b, c1r);
            let vs_default = summarize(cna, cnb, cnr);
            let impl_null = summarize(ina, inb, inr);
            let impl_ab = summarize(ia, ib, ir);
            println!("--- n={n} impl={impl_label} ---");
            report("NULL fsci/fsci", &fsci_null);
            report("NULL sp1/sp1", &scipy_null);
            report("NULL nalg/nalg", &impl_null);
            report("IMPL nalg/native", &impl_ab);
            report("fsci/scipy1", &vs_pinned);
            report("fsci/scipyN", &vs_default);
            println!(
                "n={n} impl={impl_label} observed_threads: fsci_peak_tasks={fsci_peak_tasks} \
                 scipy1_peak_tasks={scipy1_peak} scipyN_peak_tasks={scipyn_peak}"
            );

            // A null whose CI excludes 1.0 means the machine moved under the
            // measurement; the row it accompanies is not reportable.
            let null_ok = |p: &Paired| p.ratio_lo <= 1.0 && p.ratio_hi >= 1.0;

            // ...but bracketing 1.0 is NECESSARY, NOT SUFFICIENT, and this gate
            // used to stop there. A null that is enormously WIDE also brackets
            // 1.0 -- it passes precisely BECAUSE it is wide -- so a contended
            // host produced cells that certified while resolving nothing. Seen
            // on vmi1227854: nulls PASS with cv 28-30% and the effect interval
            // spanning [1.2508, 2.8213]. The fleet hit the same thing as whole
            // boards reading zero-certified under host contention.
            //
            // So require the ledger's 2x margin explicitly: the effect must
            // deviate from parity by at least twice the null's own deviation.
            // One margin PER COMPARISON, each against the nulls that bound it.
            let m_scipy1 = null_margin(&vs_pinned, &[&fsci_null, &scipy_null]);
            let m_scipyn = null_margin(&vs_default, &[&fsci_null]);
            let m_impl = null_margin(&impl_ab, &[&impl_null]);
            println!(
                "n={n} impl={impl_label} margins (need >={MIN_NULL_MARGIN:.2}x): \
                 fsci/scipy1={m_scipy1:.2}x fsci/scipyN={m_scipyn:.2}x IMPL={m_impl:.2}x"
            );
            println!(
                "n={n} impl={impl_label} IMPL_CERTIFIED={} (margin {m_impl:.2}x, ci95=[{:.4},{:.4}])",
                m_impl >= MIN_NULL_MARGIN && !(impl_ab.ratio_lo <= 1.0 && impl_ab.ratio_hi >= 1.0),
                impl_ab.ratio_lo,
                impl_ab.ratio_hi
            );
            let median_of = |mut v: Vec<f64>| -> f64 {
                v.sort_by(f64::total_cmp);
                if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
            };
            let half = alone.len() / 2;
            let alone_pre = median_of(alone[..half].to_vec());
            let alone_post = median_of(alone[half..].to_vec());
            let alone_med = median_of(alone.clone());
            let resident_med = fsci_null.p50_a;
            let contention = contention_ratio(alone_med, resident_med);
            let drift = contention_ratio(alone_pre, alone_post);
            println!(
                "n={n} impl={impl_label} CONTENTION fsci_alone={:.3}ms fsci_resident={:.3}ms \
                 ratio={contention:.4}x (tolerable <={MAX_TOLERABLE_CONTENTION:.2}x) \
                 alone_pre={:.3}ms alone_post={:.3}ms drift={drift:.4}x",
                alone_med * 1e3,
                resident_med * 1e3,
                alone_pre * 1e3,
                alone_post * 1e3
            );

            let topo = cpu_topology(&fsci_cpus);
            let placement: Vec<String> = topo
                .iter()
                .map(|(c, core, sibs, mhz)| format!("cpu{c}(core{core},sibs{sibs},{mhz:.0}MHz)"))
                .collect();
            let mhz_vals: Vec<f64> =
                topo.iter().map(|(_, _, _, m)| *m).filter(|m| m.is_finite()).collect();
            // An empty set must print "unavailable", not the fold's sentinel.
            // The first version printed f64::MAX here, which is how this bug was
            // found (frankenscipy-ll0kk).
            let mhz_summary = if mhz_vals.is_empty() {
                "unavailable".to_string()
            } else {
                let lo = mhz_vals.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = mhz_vals.iter().copied().fold(0.0_f64, f64::max);
                format!("[{lo:.0}..{hi:.0}] spread={:.2}x", hi / lo)
            };
            let smt = topo
                .iter()
                .any(|(_, _, sibs, _)| sibs.contains(',') || sibs.contains('-'));
            println!(
                "n={n} impl={impl_label} PLACEMENT fsci_cpus=[{}] mhz={mhz_summary} smt_present={smt} \
                 note=arms_unpinned_both_allowed_full_cpuset",
                placement.join(" ")
            );

            let cpuset =
                std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
            let sn_over = scipyn_oversubscribed(scipyn_peak, cpuset);
            let sn_ok = scipyn_plausible(vs_default.ratio_p50);
            println!(
                "n={n} impl={impl_label} scipyN_REPORTABLE={sn_ok} \
                 (ratio={:.4}x floor={MIN_PLAUSIBLE_INCUMBENT_RATIO:.2}x) \
                 oversubscribed={sn_over} (peak_tasks={scipyn_peak} cpuset={cpuset})",
                vs_default.ratio_p50
            );
            if !sn_ok {
                println!(
                    "n={n} impl={impl_label} scipyN VOID: the incumbent arm was pathologically \
                     slow; this says nothing about our code and must not be quoted as a win"
                );
            }
            print_loadavg_post();

            let nulls_ok =
                null_ok(&fsci_null) && null_ok(&scipy_null) && m_scipy1 >= MIN_NULL_MARGIN;
            println!(
                "n={n} impl={impl_label} VERDICT vs scipy1 (1 BLAS thread) = {:.3}x {} | vs scipyN (default BLAS) = {:.3}x | nulls={}",
                vs_pinned.ratio_p50,
                if vs_pinned.ratio_p50 > 1.0 {
                    "SLOWER"
                } else {
                    "FASTER"
                },
                vs_default.ratio_p50,
                if nulls_ok { "PASS" } else { "FAIL (row void)" }
            );
        }
    }
}

#[cfg(all(test, feature = "eigh-incumbent-bench"))]
mod tests {
    use super::bench::{MIN_NULL_MARGIN, Paired, null_margin};

    fn paired(ratio_p50: f64, lo: f64, hi: f64) -> Paired {
        Paired {
            p50_a: 0.0,
            p50_b: 0.0,
            ratio_p50,
            ratio_lo: lo,
            ratio_hi: hi,
            cv: 0.0,
            rounds: 9,
        }
    }

    #[test]
    fn margin_is_per_comparison_not_per_cell() {
        // The exact cell that exposed the bug (frankenscipy-ll0kk, vmi1227854 at
        // 181% load). Both comparisons come from ONE cell, and the old
        // per-cell check reported the first one's margin for both.
        let fsci_null = paired(0.9899, 0.7187, 1.1769);
        let scipy_null = paired(1.0000, 0.9800, 1.0200);
        let impl_null = paired(1.0495, 0.6878, 1.4094);

        let vs_scipy1 = paired(3.0405, 1.4765, 4.3473);
        let impl_ab = paired(0.7656, 0.5011, 0.8621);

        let m_scipy1 = null_margin(&vs_scipy1, &[&fsci_null, &scipy_null]);
        let m_impl = null_margin(&impl_ab, &[&impl_null]);

        // The scipy comparison clears the bar...
        assert!(
            m_scipy1 >= MIN_NULL_MARGIN,
            "scipy margin {m_scipy1} should clear {MIN_NULL_MARGIN}"
        );
        // ...while the implementation comparison in the SAME cell does not.
        assert!(
            m_impl < MIN_NULL_MARGIN,
            "impl margin {m_impl} must NOT clear {MIN_NULL_MARGIN}; its null is wider than its effect"
        );
        // And that is the regression: one number cannot stand for both.
        assert!(
            m_scipy1 > m_impl * 2.0,
            "the two margins must be far apart ({m_scipy1} vs {m_impl}); if they converge this test no longer pins anything"
        );
    }

    #[test]
    fn smt_siblings_count_as_the_same_physical_core() {
        use super::bench::arms_share_physical_core;

        // The frankenfs scenario: our arm on cpu 3, the incumbent on cpu 67 --
        // different CPU numbers, SAME physical core 3. They share execution
        // units, so each depresses the other while every A/A null stays happy.
        let ours = [(3usize, 3usize)];
        let theirs = [(67usize, 3usize)];
        assert!(
            arms_share_physical_core(&ours, &theirs),
            "distinct cpu ids on one physical core must be flagged"
        );

        // Genuinely disjoint physical cores: not flagged.
        let elsewhere = [(9usize, 9usize)];
        assert!(!arms_share_physical_core(&ours, &elsewhere));

        // Identical CPU is the degenerate case of the same thing.
        assert!(arms_share_physical_core(&ours, &ours));

        // Unknown topology (core_id unreadable) must NOT produce a false alarm:
        // silence is not evidence of separation, but it is not evidence of
        // collision either, and a false positive would void good rows.
        let unknown = [(3usize, usize::MAX)];
        assert!(!arms_share_physical_core(&unknown, &unknown));

        // Overlap anywhere in the sets is enough -- an arm that touched a
        // shared core for part of its run was contended for part of its run.
        let ours_many = [(3usize, 3usize), (5usize, 5usize)];
        let theirs_one = [(69usize, 5usize)];
        assert!(arms_share_physical_core(&ours_many, &theirs_one));
    }

    #[test]
    fn contention_ratio_flags_a_depressed_arm_and_an_aa_null_would_not() {
        use super::bench::{MAX_TOLERABLE_CONTENTION, contention_ratio};

        // Our arm at 100ms alone, 118ms with the incumbent resident: an 18%
        // depression, far past the threshold.
        let c = contention_ratio(0.100, 0.118);
        assert!(c > MAX_TOLERABLE_CONTENTION, "ratio {c} should be flagged");

        // The point the A/A null misses: BOTH A arms are measured under the
        // same resident load, so they agree with each other exactly while both
        // are depressed. A null computed from them is 1.0 and certifies
        // happily.
        let a1 = 0.118_f64;
        let a2 = 0.118_f64;
        let aa_null = a1 / a2;
        assert!(
            (aa_null - 1.0).abs() < 1e-12,
            "the A/A null is perfect ({aa_null}) even though both arms are 18% slow"
        );

        // No residency effect: ratio at parity, not flagged.
        assert!(contention_ratio(0.100, 0.100) <= MAX_TOLERABLE_CONTENTION);
        // Degenerate input must not produce a verdict.
        assert!(contention_ratio(0.0, 0.118).is_nan());
    }

    #[test]
    fn scipyn_plausibility_and_oversubscription_are_independent() {
        use super::bench::{scipyn_oversubscribed, scipyn_plausible};

        // Both observed cells were 2x oversubscribed. Only one was broken.
        // hz2: 32 threads on a cpuset of 16, ratio 2.479x -- oversubscribed but
        // perfectly reportable, because the box was quiet.
        assert!(scipyn_oversubscribed(32, 16));
        assert!(scipyn_plausible(2.479));

        // vmi1227854: 20 threads on a cpuset of 10 -- the SAME 2x ratio -- but
        // ratio 0.0124x under external load. Oversubscription did not
        // distinguish these two; plausibility does.
        assert!(scipyn_oversubscribed(20, 10));
        assert!(!scipyn_plausible(0.0124));

        // So a gate keyed on oversubscription alone would have voided the hz2
        // data too. That is the mistake this pair of predicates exists to avoid.
        assert_eq!(
            scipyn_oversubscribed(32, 16),
            scipyn_oversubscribed(20, 10),
            "both cells are equally oversubscribed, so that signal cannot separate them"
        );
        assert_ne!(
            scipyn_plausible(2.479),
            scipyn_plausible(0.0124),
            "plausibility is what separates the usable cell from the broken one"
        );

        // Not oversubscribed at all, and a sane ratio: nothing flagged.
        assert!(!scipyn_oversubscribed(2, 16));
        assert!(scipyn_plausible(1.991));
        // Unknown cpuset must not raise a false alarm.
        assert!(!scipyn_oversubscribed(32, 0));
    }

    #[test]
    fn a_wide_null_cannot_certify_a_small_effect() {
        // Bracketing 1.0 is not sufficient: this null passes the old check
        // precisely because it is wide.
        let wide_null = paired(1.0, 0.60, 1.40);
        let small_effect = paired(1.10, 1.02, 1.18);
        assert!(null_margin(&small_effect, &[&wide_null]) < MIN_NULL_MARGIN);

        // The same effect against a tight null does certify.
        let tight_null = paired(1.0, 0.99, 1.01);
        assert!(null_margin(&small_effect, &[&tight_null]) >= MIN_NULL_MARGIN);
    }

    #[test]
    fn widest_null_governs_and_a_perfect_null_is_infinite() {
        let tight = paired(1.0, 0.99, 1.01);
        let wide = paired(1.0, 0.70, 1.30);
        let effect = paired(1.50, 1.40, 1.60);
        // The WIDEST null bounds the comparison, not the friendliest one.
        let both = null_margin(&effect, &[&tight, &wide]);
        let only_wide = null_margin(&effect, &[&wide]);
        assert!((both - only_wide).abs() < 1e-12, "{both} vs {only_wide}");

        let perfect = paired(1.0, 1.0, 1.0);
        assert!(null_margin(&effect, &[&perfect]).is_infinite());
    }
}

#[cfg(feature = "eigh-incumbent-bench")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "eigh-incumbent-bench"))]
fn main() {
    eprintln!("perf_eigh_vs_scipy requires --features eigh-incumbent-bench");
    std::process::exit(2);
}
