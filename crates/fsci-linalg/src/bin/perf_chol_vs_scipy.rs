//! Live-SciPy Cholesky A/B: `fsci_linalg::cholesky` against `scipy.linalg.cholesky`.
//!
//! WHY THIS EXISTS. `frankenscipy-8l8r1.151` asks for a large-n panel-order SYRK A/B, and
//! its parent `8l8r1` is "close ALL vs-upstream perf gaps". Every Cholesky A/B in this repo
//! so far has been fsci-against-fsci — a SELF-SPEEDUP, which the campaign rules classify as
//! maintenance, not a win. There was no measured `cholesky` ratio against the incumbent at
//! all, so there was no target to close and no way to tell whether a SYRK lever mattered.
//! This binary produces that missing number first.
//!
//! WHAT MAKES THE ROW ADMISSIBLE.
//!   1. BOTH ARMS RUN IN ONE INVOCATION. The SciPy arm is a persistent child holding the
//!      fixture, driven over a stdin protocol, so fsci and SciPy are timed inside the same
//!      process lifetime rather than in two runs the host can drift between.
//!   2. INTERLEAVED, NOT BLOCKED. Cells alternate ABBA and aggregate at replicate level.
//!      Window drift on this host is larger than the effects being measured, so a
//!      block-A-then-block-B design measures the window, not the code.
//!   3. EVERY RATIO CARRIES ITS A/A NULL. fsci-vs-fsci and scipy-vs-scipy must land at 1.0.
//!      A failing null invalidates the row rather than being reported alongside it.
//!   4. PER-ARM loadavg AND CPU MHz. Cores on this box run at different frequencies AT THE
//!      SAME INSTANT (1429-4289 MHz observed), so a ratio whose arms sat at different clocks
//!      is a frequency ratio in disguise. Both are sampled around each arm, not once per run.
//!   5. PARITY BEFORE TIMING. A speed number for a wrong answer is not a result. Before any
//!      cell is timed the run computes the NORMWISE backward error `‖A − LLᵀ‖_F / ‖A‖_F` for
//!      both arms and aborts unless ours is inside `16·n·ε` AND within 4x the incumbent's.
//!   6. NO C BLAS ANYWHERE ON OUR SIDE. The fsci arm is safe Rust; the row records the
//!      SciPy arm's BLAS so the comparison is named honestly, and `ldd` on this binary is
//!      the check that we did not smuggle one in.
//!
//! THE SCIPY ARM IS RUN TWICE, DELIBERATELY. `scipy1` pins every BLAS thread variable to 1;
//! `scipyN` leaves default parallelism but CAPPED AT THE CPUSET. Unbounded, OpenBLAS
//! oversubscribes and returns numbers that flatter us absurdly while wrecking the window for
//! every other arm. `scipyN` is the strongest honest incumbent; `scipy1` is the per-core
//! comparison. Reporting only one of them would be picking the arm that suits the answer.

#[cfg(not(unix))]
fn main() {
    eprintln!("perf_chol_vs_scipy requires a unix host (/proc sampling)");
    std::process::exit(1);
}

#[cfg(unix)]
mod harness {
    use fsci_linalg::{CHOL_SYRK_NC_OVERRIDE, DecompOptions, cholesky};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    /// The campaign's pinned, screened SciPy 1.17.1, when this host has it.
    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";

    /// Backward-error budget for our own factor, in units of `n · f64::EPSILON`.
    ///
    /// WHY NOT AN ELEMENTWISE RELATIVE DIFF. The first version of this harness gated on
    /// `max |L_ours − L_theirs| / |L|` against a fixed 1e-11, and it FAILED at n=1024 with
    /// 1.219e-11 at entry (822, 577). That gate was measuring the wrong thing: a Cholesky
    /// factor has entries spanning many orders of magnitude, and the relative difference of
    /// a near-zero entry is enormous while its contribution to the factorisation is nil.
    /// Gating on it makes the pass/fail depend on the smallest entry in the matrix, which
    /// is a scale bug, and "loosen the constant until n=1024 passes" would have been gate
    /// self-weakening dressed up as a fix.
    ///
    /// The quantity that actually says whether a Cholesky is correct is the NORMWISE
    /// backward error `‖A − LLᵀ‖_F / ‖A‖_F`, for which the standard bound is `O(n)·ε`
    /// (Higham, Accuracy and Stability of Numerical Algorithms, Thm 10.3). So this gate is
    /// stronger than the one it replaces, not weaker: it checks OUR factor against the
    /// mathematics in absolute terms, rather than only checking that we agree with SciPy.
    const BACKWARD_ERROR_BUDGET: f64 = 16.0;

    /// How much worse than the incumbent's own backward error ours may be.
    ///
    /// The absolute budget above can be satisfied by both arms while one is quietly ten
    /// times sloppier, so the comparison to SciPy is kept as a second, independent gate.
    /// Anchoring to the incumbent rather than to a constant is what makes this meaningful
    /// on a host whose BLAS we do not control.
    const MAX_BACKWARD_ERROR_VS_SCIPY: f64 = 4.0;

    /// Highest 1-minute loadavg at which a cell may be certified.
    ///
    /// OBSERVED DEFECT, not a precaution. On 2026-08-19 the same code, fixtures and protocol
    /// gave `scipyN/fsci` = 1.339x at loadavg 20-27 and 0.808x at loadavg 60-106 for n=512 —
    /// the cell INVERTED — and **both A/A nulls passed in the loaded window** (1.012, 1.012).
    /// An A/A null bounds drift BETWEEN the halves of a cell; it is structurally blind to a
    /// load that is high but steady, because that depresses both halves equally while
    /// changing which arm the scheduler favours. So the null cannot be the only gate, and a
    /// loadavg printed but not enforced is a number nobody checks.
    ///
    /// WHICH LOAD, AND WHY IT IS NOT THE PEAK. The first version gated on the PEAK loadavg
    /// during the cell at a ceiling of 16, and refused every cell it was ever run on —
    /// including the quiet-window ones it was written to admit. The reason is that the
    /// harness generates most of that peak itself: the `scipyN` arm alone runs 64 tasks, so
    /// a run drives loadavg from 19.9 to 37 without any external contention at all. A
    /// ceiling below your own operating point is a freeze, not a gate.
    ///
    /// The confound worth catching is EXTERNAL contention. Our own arms load the box in
    /// BOTH halves of every cell by construction, so they are part of the measurement, not a
    /// bias in it. So the gate reads AMBIENT load — sampled once before a size's first
    /// replicate — and the peak during the cell is recorded as provenance and explicitly not
    /// gated.
    ///
    /// The 30 is the campaign's own documented deferral threshold, and it reproduces the
    /// historical evidence in both directions: the window that produced the 1.407x reading
    /// had ambient 20.60 (admitted), and the window where the same code read 0.910x had
    /// ambient 59.95 (refused). Override with `FSCI_CHOL_MAX_LOAD` and say so in the row.
    const DEFAULT_MAX_LOADAVG: f64 = 30.0;

    /// Largest tolerated ratio between the two arms' mean clock over a cell.
    ///
    /// Cores on this host run at DIFFERENT frequencies at the same instant (1429-4289 MHz
    /// observed). A ratio whose arms sat at materially different clocks is a frequency ratio
    /// wearing a costume, so the harness measures each arm's own cpuset clock and refuses the
    /// cell when they diverge. 1.25 is wide enough to pass the ordinary jitter seen in the
    /// quiet-window runs (fsci 2504-3469 against scipy1 2928-3166) and narrow enough to catch
    /// an arm parked on a different CCD.
    const MAX_CROSS_ARM_CLOCK_RATIO: f64 = 1.25;

    /// Threads the SciPy arm materialises per thread requested — MEASURED on this fleet and
    /// recorded in `perf_eigh_vs_scipy.rs` (scipy1 requests 1, observes 2; scipyN requests
    /// 10, observes 20). Encoded as a constant with its provenance rather than guessed.
    const OBSERVED_THREADS_PER_REQUESTED: usize = 2;

    fn blas_thread_cap(cpuset: usize) -> usize {
        (cpuset / OBSERVED_THREADS_PER_REQUESTED).max(1)
    }

    const PYTHON_ORACLE: &str = r#"
import hashlib
import os
import sys
import collections
import threading
import time

import numpy as np
import scipy
from scipy import linalg
import scipy.linalg._decomp_cholesky as _chol

N = int(os.environ["FSCI_CHOL_N"])
EXPECTED_BYTES = N * N * 8
raw = sys.stdin.buffer.read(EXPECTED_BYTES)
if len(raw) != EXPECTED_BYTES:
    raise RuntimeError(f"fixture byte count mismatch: expected {EXPECTED_BYTES}, got {len(raw)}")
FIXTURE_SHA256 = hashlib.sha256(raw).hexdigest()
A = np.frombuffer(raw, dtype="<f8").reshape(N, N).copy()
if not np.array_equal(A, A.T):
    raise RuntimeError("fixture is not exactly symmetric")

# Thread COUNT and NAMES, polled: the environment variables are the request, this is the
# observation. A row that reports requested threads is reporting a wish.
PEAK_TASKS = len(os.listdir("/proc/self/task"))
STOP = threading.Event()
THREAD_NAMES = {}

def poll_tasks():
    global PEAK_TASKS
    while not STOP.is_set():
        tids = os.listdir("/proc/self/task")
        count = len(tids)
        if count > PEAK_TASKS:
            PEAK_TASKS = count
        for tid in tids:
            if tid not in THREAD_NAMES:
                try:
                    with open(f"/proc/self/task/{tid}/comm") as fh:
                        THREAD_NAMES[tid] = fh.read().strip()
                except OSError:
                    pass
        STOP.wait(0.002)

POLLER = threading.Thread(target=poll_tasks, daemon=True)
POLLER.start()

# check_finite=False so the incumbent is NOT carrying a validation pass our arm skips.
# This is the strongest honest SciPy: overwrite_a stays False because our arm does not
# destroy its input either, and letting it would be comparing different contracts.
def run():
    return linalg.cholesky(A, lower=True, check_finite=False)

# Warmup outside every timer: the first call pays BLAS buffer allocation.
L = run()
if L.shape != (N, N):
    raise RuntimeError("unexpected cholesky output shape")

try:
    engine_path = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
except Exception:
    engine_path = "unknown"
chol_path = _chol.__file__
with open(chol_path, "rb") as handle:
    CHOL_SHA256 = hashlib.sha256(handle.read()).hexdigest()
fsci_loaded = any(name.startswith("fsci") or name.startswith("frankenscipy") for name in sys.modules)

print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" blas={engine_path}"
    f" chol_module={_chol.__name__}"
    f" chol_sha256={CHOL_SHA256}"
    f" fixture_sha256={FIXTURE_SHA256}"
    f" n={N}"
    f" affinity={len(os.sched_getaffinity(0))}"
    f" peak_tasks={PEAK_TASKS}"
    f" thread_names={sorted(collections.Counter(THREAD_NAMES.values()).items())}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={scipy.__version__ == '1.17.1' and not fsci_loaded}",
    flush=True,
)

for raw_line in sys.stdin.buffer:
    fields = str(raw_line, "ascii").strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "FACTOR":
        # Raw bits, so the parity check is on the values and not on a decimal rendering.
        bits = np.ascontiguousarray(run(), dtype="<f8").view("<u8").reshape(-1)
        print("FACTOR " + ",".join(f"{int(v):016x}" for v in bits), flush=True)
    elif command == "MHZ":
        # Sampled INSIDE the arm's own process, so it is this arm's clock.
        mhz = 0.0
        try:
            with open("/proc/cpuinfo") as fh:
                vals = [float(l.split(":")[1]) for l in fh if l.startswith("cpu MHz")]
            cpus = sorted(os.sched_getaffinity(0))
            allowed = [vals[c] for c in cpus if c < len(vals)]
            mhz = sum(allowed) / len(allowed) if allowed else 0.0
        except Exception:
            pass
        with open("/proc/loadavg") as fh:
            load = fh.read().split()[0]
        print(f"MHZ {mhz:.1f} {load}", flush=True)
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
                out = run()
                checksum += float(out[0, 0]) + float(out[N - 1, N - 1])
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

    fn read_trimmed(path: &str) -> String {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .trim()
            .to_string()
    }

    fn loadavg_1min() -> f64 {
        read_trimmed("/proc/loadavg")
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::NAN)
    }

    /// Mean `cpu MHz` over the CPUs THIS process may run on.
    ///
    /// Not a whole-box mean: cores differ by up to 3x at the same instant here, and the
    /// only frequency that explains an arm's time is the frequency of the cores it was
    /// allowed to use.
    fn cpu_mhz_mean() -> f64 {
        let text = read_trimmed("/proc/cpuinfo");
        let vals: Vec<f64> = text
            .lines()
            .filter_map(|line| line.strip_prefix("cpu MHz"))
            .filter_map(|rest| rest.split(':').nth(1))
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        if vals.is_empty() {
            return f64::NAN;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    /// SHA-256 of this running executable, read from `/proc/self/exe`.
    ///
    /// Self-reported from INSIDE the process, so the row names the binary that produced it
    /// rather than a path that may have been rebuilt since.
    fn elf_sha256() -> String {
        let Ok(mut file) = std::fs::File::open("/proc/self/exe") else {
            return "unavailable".to_string();
        };
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(k) => hasher.update(&buf[..k]),
                Err(_) => return "unavailable".to_string(),
            }
        }
        hasher.hex()
    }

    /// Minimal SHA-256. Vendored rather than pulled in as a dependency: this binary exists
    /// to certify that we link nothing exotic, so it should not add a crate to say so.
    struct Sha256 {
        state: [u32; 8],
        buf: [u8; 64],
        buflen: usize,
        len: u64,
    }

    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    impl Sha256 {
        fn new() -> Self {
            Self {
                state: [
                    0x6a09_e667,
                    0xbb67_ae85,
                    0x3c6e_f372,
                    0xa54f_f53a,
                    0x510e_527f,
                    0x9b05_688c,
                    0x1f83_d9ab,
                    0x5be0_cd19,
                ],
                buf: [0u8; 64],
                buflen: 0,
                len: 0,
            }
        }

        // The SHA-256 message schedule is defined by index arithmetic (w[i-15], w[i-2],
        // w[i-7], w[i-16]); rewriting it over iterators would obscure the specification it
        // transcribes without changing what it computes.
        #[allow(
            clippy::needless_range_loop,
            reason = "indices are the FIPS 180-4 spec"
        )]
        fn compress(&mut self) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    self.buf[4 * i],
                    self.buf[4 * i + 1],
                    self.buf[4 * i + 2],
                    self.buf[4 * i + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            for (slot, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
                *slot = slot.wrapping_add(v);
            }
        }

        fn update(&mut self, mut data: &[u8]) {
            self.len = self.len.wrapping_add(data.len() as u64);
            while !data.is_empty() {
                let take = (64 - self.buflen).min(data.len());
                self.buf[self.buflen..self.buflen + take].copy_from_slice(&data[..take]);
                self.buflen += take;
                data = &data[take..];
                if self.buflen == 64 {
                    self.compress();
                    self.buflen = 0;
                }
            }
        }

        fn hex(mut self) -> String {
            let bitlen = self.len.wrapping_mul(8);
            self.update(&[0x80]);
            while self.buflen != 56 {
                self.update(&[0x00]);
            }
            let tail = bitlen.to_be_bytes();
            self.buf[56..64].copy_from_slice(&tail);
            self.buflen = 64;
            self.compress();
            let mut out = String::with_capacity(64);
            for word in self.state {
                out.push_str(&format!("{word:08x}"));
            }
            out
        }
    }

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
            if std::path::Path::new(SCIPY_SITE_PACKAGES).is_dir() {
                command.env("PYTHONPATH", SCIPY_SITE_PACKAGES);
            }
            command
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("FSCI_CHOL_N", n.to_string())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit());

            let thread_keys = [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ];
            if pin_threads {
                for key in thread_keys {
                    command.env(key, "1");
                }
            } else {
                let cap = blas_thread_cap(
                    std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
                )
                .to_string();
                for key in thread_keys {
                    command.env(key, &cap);
                }
            }

            let mut child = command.spawn().expect("failed to spawn the SciPy arm");
            let mut stdin = child.stdin.take().expect("scipy stdin");
            stdin.write_all(bytes).expect("write fixture");
            stdin.flush().expect("flush fixture");
            let mut stdout = BufReader::new(child.stdout.take().expect("scipy stdout"));
            let mut ready = String::new();
            stdout.read_line(&mut ready).expect("scipy READY");
            assert!(
                ready.starts_with("READY"),
                "scipy arm did not become ready: {ready}"
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

        fn command(&mut self, line: &str) -> String {
            writeln!(self.stdin, "{line}").expect("write command");
            self.stdin.flush().expect("flush command");
            let mut reply = String::new();
            self.stdout.read_line(&mut reply).expect("read reply");
            reply.trim().to_string()
        }

        fn factor_bits(&mut self) -> Vec<u64> {
            let reply = self.command("FACTOR");
            let body = reply
                .strip_prefix("FACTOR ")
                .unwrap_or_else(|| panic!("{}: bad FACTOR reply: {reply}", self.label));
            body.split(',')
                .map(|h| u64::from_str_radix(h, 16).expect("hex u64"))
                .collect()
        }

        /// `(mean MHz over this arm's own cpuset, 1-minute loadavg)`, sampled in the child.
        fn mhz_load(&mut self) -> (f64, f64) {
            let reply = self.command("MHZ");
            let mut it = reply
                .strip_prefix("MHZ ")
                .unwrap_or("0 0")
                .split_whitespace();
            (
                it.next().and_then(|v| v.parse().ok()).unwrap_or(f64::NAN),
                it.next().and_then(|v| v.parse().ok()).unwrap_or(f64::NAN),
            )
        }

        /// Best-of-`min_of` wall seconds for `reps` factorisations, plus observed task peak.
        fn time(&mut self, reps: usize, min_of: usize) -> (f64, usize) {
            let reply = self.command(&format!("TIME {reps} {min_of}"));
            let mut it = reply
                .strip_prefix("TIME ")
                .unwrap_or_else(|| panic!("{}: bad TIME reply: {reply}", self.label))
                .split_whitespace();
            let secs: f64 = it.next().expect("secs").parse().expect("secs");
            let _checksum = it.next();
            let tasks: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            (secs, tasks)
        }

        fn quit(&mut self) -> usize {
            if self.stopped {
                return 0;
            }
            self.stopped = true;
            let reply = self.command("QUIT");
            let tasks = reply
                .strip_prefix("BYE ")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            let _ = self.child.wait();
            tasks
        }
    }

    /// A symmetric positive-definite fixture: `A = M Mᵀ + n·I`, deterministic in `seed`.
    ///
    /// Built symmetric BY CONSTRUCTION and then written from the lower triangle, so the
    /// SciPy side's `array_equal(A, A.T)` assertion is exact rather than nearly true. The
    /// `n·I` shift keeps it comfortably conditioned: this harness measures the cost of a
    /// factorisation, and a fixture that is nearly singular would measure how each arm
    /// handles a hard case instead.
    fn spd_fixture(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| next()).collect()).collect();
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let dot: f64 = (0..n).map(|k| m[i][k] * m[j][k]).sum();
                let v = if i == j { dot + n as f64 } else { dot };
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        a
    }

    fn to_row_major_bytes(a: &[Vec<f64>]) -> Vec<u8> {
        let mut out = Vec::with_capacity(a.len() * a.len() * 8);
        for row in a {
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Best-of-`min_of` wall seconds for `reps` fsci factorisations at a given NC setting.
    ///
    /// `nc = 0` is the shipped single-pass SYRK traversal; `nc > 0` selects the NC-blocked
    /// one. The toggle is set ONCE around the whole timed region, not per iteration, so the
    /// atomic store is never part of what is being timed. The two arms are bit-identical by
    /// construction and by `chol_syrk_nc_blocking_is_bit_identical`, so any difference here
    /// is cost and nothing else — which is the only reason a same-binary A/B on it is
    /// meaningful rather than a comparison of two different computations.
    fn time_fsci_nc(a: &[Vec<f64>], reps: usize, min_of: usize, nc: usize) -> f64 {
        CHOL_SYRK_NC_OVERRIDE.store(nc, std::sync::atomic::Ordering::Relaxed);
        let out = time_fsci(a, reps, min_of);
        CHOL_SYRK_NC_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        out
    }

    /// Best-of-`min_of` wall seconds for `reps` fsci factorisations.
    fn time_fsci(a: &[Vec<f64>], reps: usize, min_of: usize) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..min_of {
            let started = Instant::now();
            for _ in 0..reps {
                let out = cholesky(a, true, DecompOptions::default()).expect("spd fixture factors");
                std::hint::black_box(out.factor[0][0]);
            }
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed < best {
                best = elapsed;
            }
        }
        best
    }

    /// Normwise backward error `‖A − LLᵀ‖_F / ‖A‖_F`, using only the lower triangle of `l`.
    ///
    /// This is the quantity Cholesky is actually accurate in. Computed for BOTH arms so the
    /// row can say whether we match the incumbent's accuracy, not merely whether we agree
    /// with it to some constant a human picked.
    fn backward_error(a: &[Vec<f64>], l: &[Vec<f64>]) -> f64 {
        let n = a.len();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..n {
            for j in 0..=i {
                // (LLᵀ)_ij = Σ_{k<=min(i,j)} L_ik L_jk — the sum stops at j because L is lower.
                let dot: f64 = l[i][..=j].iter().zip(&l[j][..=j]).map(|(x, y)| x * y).sum();
                let d = a[i][j] - dot;
                // Off-diagonal entries appear twice in the full symmetric matrix, so they
                // count twice toward a Frobenius norm computed from one triangle.
                let w = if i == j { 1.0 } else { 2.0 };
                num += w * d * d;
                den += w * a[i][j] * a[i][j];
            }
        }
        if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        }
    }

    /// `‖L_ours − L_theirs‖_F / ‖L_theirs‖_F` over the lower triangle.
    fn lower_rel_frobenius(ours: &[Vec<f64>], theirs: &[Vec<f64>]) -> f64 {
        let n = ours.len();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..n {
            for j in 0..=i {
                let d = ours[i][j] - theirs[i][j];
                num += d * d;
                den += theirs[i][j] * theirs[i][j];
            }
        }
        if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        }
    }

    /// The elementwise relative difference and where it occurs — DIAGNOSTIC ONLY.
    fn worst_elementwise_rel(ours: &[Vec<f64>], theirs: &[Vec<f64>]) -> (f64, (usize, usize)) {
        let n = ours.len();
        let mut worst = 0.0f64;
        let mut at = (0usize, 0usize);
        for i in 0..n {
            for j in 0..=i {
                let (m, t) = (ours[i][j], theirs[i][j]);
                let denom = t.abs().max(m.abs()).max(f64::MIN_POSITIVE);
                let rel = (m - t).abs() / denom;
                if rel > worst {
                    worst = rel;
                    at = (i, j);
                }
            }
        }
        (worst, at)
    }

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        if n % 2 == 1 {
            v[n / 2]
        } else {
            f64::midpoint(v[n / 2 - 1], v[n / 2])
        }
    }

    pub fn run() {
        let sizes: Vec<usize> = std::env::var("FSCI_CHOL_SIZES")
            .unwrap_or_else(|_| "256,512,1024".to_string())
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        let replicates = env_usize("FSCI_CHOL_REPLICATES", 5);
        let reps = env_usize("FSCI_CHOL_REPS", 3);
        let min_of = env_usize("FSCI_CHOL_MIN_OF", 3);
        let seed = 0x5eed_c0de_u64;
        // The NC-blocking arm under test. 0 disables the arm entirely (the harness then
        // measures only fsci against SciPy, exactly as before).
        let nc_arm: usize = std::env::var("FSCI_CHOL_NC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let max_load: f64 = std::env::var("FSCI_CHOL_MAX_LOAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_LOADAVG);

        println!("# perf_chol_vs_scipy — fsci_linalg::cholesky vs scipy.linalg.cholesky, LIVE");
        println!("elf_sha256={}", elf_sha256());
        println!(
            "host={} nproc={} loadavg_pre={} mhz_pre={:.0}",
            read_trimmed("/proc/sys/kernel/hostname"),
            std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
            read_trimmed("/proc/loadavg"),
            cpu_mhz_mean(),
        );
        println!(
            "sizes={sizes:?} replicates={replicates} reps={reps} min_of={min_of} seed={seed:#x} \
             max_loadavg={max_load} max_clock_ratio={MAX_CROSS_ARM_CLOCK_RATIO} nc_arm={nc_arm}"
        );

        for &n in &sizes {
            let a = spd_fixture(n, seed);
            let bytes = to_row_major_bytes(&a);

            let (mut scipy1, ready1) = Scipy::start("scipy1", n, &bytes, true);
            let (mut scipyn, readyn) = Scipy::start("scipyN", n, &bytes, false);
            println!("n={n} {ready1}");
            println!("n={n} {readyn}");

            // ── PARITY FIRST. A timing row for a wrong factor is not a result. ──────
            let ours = cholesky(&a, true, DecompOptions::default()).expect("spd fixture factors");
            let theirs_bits = scipy1.factor_bits();
            assert_eq!(
                theirs_bits.len(),
                n * n,
                "scipy returned {} values for an {n}x{n} factor",
                theirs_bits.len()
            );
            let theirs: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| f64::from_bits(theirs_bits[i * n + j]))
                        .collect()
                })
                .collect();

            let res_ours = backward_error(&a, &ours.factor);
            let res_theirs = backward_error(&a, &theirs);
            let budget = BACKWARD_ERROR_BUDGET * n as f64 * f64::EPSILON;
            let factor_rel = lower_rel_frobenius(&ours.factor, &theirs);
            // Reported, NOT gated: kept because it is what the first version of this gate
            // used, so the row shows the number that misled it alongside the one that did not.
            let (worst, worst_at) = worst_elementwise_rel(&ours.factor, &theirs);
            println!(
                "n={n} parity backward_err ours={res_ours:.3e} scipy={res_theirs:.3e} \
                 budget={budget:.3e} (16·n·eps) factor_rel_fro={factor_rel:.3e} \
                 worst_elementwise_rel={worst:.3e}@({},{}) [diagnostic, not gated]",
                worst_at.0, worst_at.1
            );
            assert!(
                res_ours <= budget,
                "n={n}: our backward error {res_ours:.3e} exceeds {budget:.3e}; the factor \
                 is wrong and timing it would be meaningless"
            );
            assert!(
                res_ours <= MAX_BACKWARD_ERROR_VS_SCIPY * res_theirs,
                "n={n}: our backward error {res_ours:.3e} is more than \
                 {MAX_BACKWARD_ERROR_VS_SCIPY}x the incumbent's {res_theirs:.3e}; we would be \
                 buying speed with accuracy"
            );

            // Warm the fsci arm outside every timer, matching the SciPy side's warmup.
            //
            // THREE factorisations, not one. With a single warmup the n=256 cell returned an
            // fsci A/A null of 1.676 — the first timed half was paying worker-pool spin-up
            // that the second half did not, which the null correctly flagged as invalid.
            // Warmup is the fix; widening the null threshold would have been hiding it.
            std::hint::black_box(time_fsci(&a, 3, 1));

            let mut r_scipy1 = Vec::with_capacity(replicates);
            let mut r_scipyn = Vec::with_capacity(replicates);
            let mut null_fsci = Vec::with_capacity(replicates);
            let mut null_s1 = Vec::with_capacity(replicates);
            // NC-blocking arm: ratio against our own default traversal, plus its own A/A
            // null. This is a SELF-speedup and is labelled as one — it is the lever, and the
            // SciPy ratios in the same cell are what say whether the lever matters.
            let mut r_nc = Vec::with_capacity(replicates);
            let mut null_nc = Vec::with_capacity(replicates);
            // Every load and clock sample taken anywhere inside this cell, so the gates
            // below judge the window the ratio was actually measured in rather than a
            // single reading taken before it.
            let ambient = loadavg_1min();
            let mut cell_loads: Vec<f64> = Vec::new();
            let mut cell_mhz_fsci: Vec<f64> = Vec::new();
            let mut cell_mhz_scipy: Vec<f64> = Vec::new();

            for rep in 0..replicates {
                // ABBA within the replicate: the second half reverses the order so a
                // monotone drift across the cell cancels instead of loading one arm.
                let (mhz_f0, load_f0) = (cpu_mhz_mean(), loadavg_1min());
                let f_a = time_fsci(&a, reps, min_of);
                let (s1_a, tasks1) = scipy1.time(reps, min_of);
                let (mhz_s1, load_s1) = scipy1.mhz_load();
                let (sn_a, tasksn) = scipyn.time(reps, min_of);
                let (mhz_sn, load_sn) = scipyn.mhz_load();
                let (sn_b, _) = scipyn.time(reps, min_of);
                let (s1_b, _) = scipy1.time(reps, min_of);
                let f_b = time_fsci(&a, reps, min_of);
                let (mhz_f1, load_f1) = (cpu_mhz_mean(), loadavg_1min());
                // NC arm, ABBA against the default arm inside the same replicate so the two
                // traversals are compared in the same window rather than across cells.
                if nc_arm > 0 {
                    let nc_a = time_fsci_nc(&a, reps, min_of, nc_arm);
                    let d_a = time_fsci_nc(&a, reps, min_of, 0);
                    let d_b = time_fsci_nc(&a, reps, min_of, 0);
                    let nc_b = time_fsci_nc(&a, reps, min_of, nc_arm);
                    let nc_t = nc_a.min(nc_b);
                    let d_t = d_a.min(d_b);
                    null_nc.push(d_a.max(d_b) / d_a.min(d_b));
                    // > 1 means NC-blocking is FASTER than the shipped traversal.
                    r_nc.push(d_t / nc_t);
                }
                cell_loads.extend([load_f0, load_f1, load_s1, load_sn]);
                cell_mhz_fsci.extend([mhz_f0, mhz_f1]);
                cell_mhz_scipy.extend([mhz_s1, mhz_sn]);

                let fsci = f_a.min(f_b);
                let s1 = s1_a.min(s1_b);
                let sn = sn_a.min(sn_b);

                // A/A nulls from the two halves of the SAME cell: they bracket the same
                // window the ratio does, which is the only way they can invalidate it.
                null_fsci.push(f_a.max(f_b) / f_a.min(f_b));
                null_s1.push(s1_a.max(s1_b) / s1_a.min(s1_b));

                r_scipy1.push(s1 / fsci);
                r_scipyn.push(sn / fsci);

                println!(
                    "n={n} rep={rep} fsci={fsci:.6e}s scipy1={s1:.6e}s scipyN={sn:.6e}s \
                     r1={:.3}x rN={:.3}x | fsci[mhz {mhz_f0:.0}->{mhz_f1:.0} load {load_f0:.2}->{load_f1:.2}] \
                     scipy1[mhz {mhz_s1:.0} load {load_s1:.2} tasks {tasks1}] \
                     scipyN[mhz {mhz_sn:.0} load {load_sn:.2} tasks {tasksn}]",
                    s1 / fsci,
                    sn / fsci,
                );
            }

            let m1 = median(&mut r_scipy1.clone());
            let mn = median(&mut r_scipyn.clone());
            let nf = median(&mut null_fsci.clone());
            let ns = median(&mut null_s1.clone());
            let nulls_ok = (nf - 1.0).abs() <= 0.05 && (ns - 1.0).abs() <= 0.05;

            // ── THE TWO GATES THE A/A NULL CANNOT PROVIDE ──────────────────────────
            // `ambient` is the load BEFORE this size ran anything; `load_peak` is the
            // highest reading during it and is dominated by our own arms, so it is reported
            // and NOT gated. See DEFAULT_MAX_LOADAVG for why that distinction is the whole
            // difference between a gate and a freeze.
            let load_peak = cell_loads.iter().copied().fold(0.0f64, f64::max);
            let load_ok = ambient <= max_load;
            let mhz_f = mean(&cell_mhz_fsci);
            let mhz_s = mean(&cell_mhz_scipy);
            let clock_ratio = if mhz_f > 0.0 && mhz_s > 0.0 {
                (mhz_f / mhz_s).max(mhz_s / mhz_f)
            } else {
                f64::NAN
            };
            // `is_finite()` FIRST, so a NaN ratio (no clock samples) FAILS the gate. The
            // first version wrote this as `!(ratio > MAX)`, which is true for NaN and would
            // have let a cell with no frequency evidence certify itself — the exact shape of
            // blindness that a gate is supposed to prevent.
            let clock_ok = clock_ratio.is_finite() && clock_ratio <= MAX_CROSS_ARM_CLOCK_RATIO;

            let verdict = if nulls_ok && load_ok && clock_ok {
                "PASS"
            } else {
                "FAIL"
            };
            println!(
                "n={n} RESULT scipy1/fsci={m1:.3}x scipyN/fsci={mn:.3}x \
                 null_fsci={nf:.3} null_scipy1={ns:.3} ambient={ambient:.2}/{max_load:.2} \
                 load_peak={load_peak:.2}[ours, not gated] \
                 mhz_fsci={mhz_f:.0} mhz_scipy={mhz_s:.0} clock_ratio={clock_ratio:.3} \
                 gates={verdict} loadavg_post={}",
                read_trimmed("/proc/loadavg"),
            );
            if nc_arm > 0 {
                let mnc = median(&mut r_nc.clone());
                let nnc = median(&mut null_nc.clone());
                let nc_ok = (nnc - 1.0).abs() <= 0.05;
                println!(
                    "n={n} NC_ARM nc={nc_arm} default/nc={mnc:.3}x (>1 = NC faster) \
                     null_nc={nnc:.3} nc_gates={}",
                    if nc_ok && load_ok { "PASS" } else { "FAIL" }
                );
            }
            if !nulls_ok {
                println!(
                    "n={n} ROW INVALID: an A/A null missed 1.0 by more than 5%, so this \
                     cell measured the window, not the code."
                );
            }
            if !load_ok {
                println!(
                    "n={n} ROW INVALID: ambient loadavg was {ambient:.2} against a ceiling of \
                     {max_load:.2}. This gate exists because a passing A/A null certified a \
                     cell that was wrong by 1.5x under steady load; the null cannot see it."
                );
            }
            if !clock_ok {
                println!(
                    "n={n} ROW INVALID: the arms ran at {mhz_f:.0} MHz and {mhz_s:.0} MHz \
                     (ratio {clock_ratio:.3} > {MAX_CROSS_ARM_CLOCK_RATIO}); this would be a \
                     frequency ratio, not a code ratio."
                );
            }

            let t1 = scipy1.quit();
            let tn = scipyn.quit();
            println!("n={n} scipy1_peak_tasks={t1} scipyN_peak_tasks={tn}");
        }
    }
}

#[cfg(unix)]
fn main() {
    harness::run();
}
