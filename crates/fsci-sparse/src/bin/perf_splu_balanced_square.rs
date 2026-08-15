//! `splu` versus a LIVE SciPy incumbent, on a CONTENDED host (balanced square).
//!
//! WHY THIS EXISTS. `perf_spsolve`'s cubic-live gate is the sanctioned splu
//! harness and its `bounded_preflight` requires twelve one-second samples in which
//! the pinned CPU, its SMT siblings, AND the mean across all 64 CPUs are each
//! ≤ 20% busy. On a box shared by a dozen agents that host-mean term is
//! effectively unsatisfiable, which is why splu — the worst vs-incumbent ratio in
//! this repo — keeps going unmeasured while integrity work accumulates instead.
//! franken_networkx root-caused the identical failure in its own harness
//! (`require_host_wide_quiescence`, 25 consecutive attempts with zero admitted)
//! and committed the fix at `scripts/balanced_square_ab.py` (72761094c). This is
//! that design ported to this repo's splu surface.
//!
//! THE SUBSTITUTION, and what it does NOT relax. It does not try to make the host
//! quiet; it makes the COMPARISON immune to the host being busy:
//!
//!   * Both arms run INSIDE one round, interleaved as a balanced square
//!     `A B B A A B B A`. Each arm occupies the same multiset of slot POSITIONS,
//!     so drift across a round hits both arms equally instead of biasing one.
//!   * Each arm carries its OWN A/A null — that arm's first-half slots over its
//!     second-half slots — which must land at 1.0. Contention is therefore caught
//!     per row, after the fact, instead of being excluded up front by a predicate
//!     that can never be satisfied.
//!   * A row whose null leaves [0.98, 1.02] is reported NULL-FAILED and its ratio
//!     is NOT a result. Refusing is the point.
//!
//! Everything the campaign requires of evidence is unchanged and still enforced
//! here: the incumbent runs LIVE in this same invocation on the identical matrix
//! bytes, each arm times itself so the pipe never enters either number, both arms
//! are parity-gated before any timing, and the ELF SHA-256 is self-reported from
//! inside this process. This is a substrate for when the quiescence gate cannot be
//! met — it does not retire `perf_spsolve`'s contract rows, which remain the
//! stronger evidence whenever the host is genuinely idle.
//!
//! Ratio convention is `t_scipy / t_fsci`, matching the ledgers: > 1 means
//! FrankenSciPy is faster.
//!
//! Run: `cargo run --release -p fsci-sparse --bin perf_splu_balanced_square \
//!        --features sparse-incumbent-bench -- [side] [rounds]`

mod bench {
    use fsci_sparse::{
        CooMatrix, CscMatrix, FormatConvertible, LuOptions, SPLU_CUBIC_SPECTRAL_DISABLE,
        SPLU_CUBIC_SPECTRAL_FACTOR_HITS, Shape2D, splu, splu_factor_payload_bytes, splu_solve,
    };
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// Slot order within one round. Each arm gets four slots, and the two arms
    /// occupy mirror-image positions, so a monotone drift across the round
    /// cancels in the per-round ratio.
    const SQUARE: [u8; 8] = *b"ABBAABBA";
    /// A per-arm A/A null must land within this of 1.0 or the row is void.
    const NULL_BOUND: f64 = 0.02;
    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
    /// Both factorizations solve the same RHS before any timing; they use
    /// different orderings and pivot thresholds, so agreement is to solve
    /// accuracy, not to bits.
    const MAX_SOLUTION_REL_DIFF: f64 = 1.0e-9;

    const PYTHON_ORACLE: &str = r#"
import hashlib
import os
import sys
import time

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.linalg._dsolve.linsolve as _linsolve

N = int(os.environ["FSCI_SPLU_N"])
NNZ = int(os.environ["FSCI_SPLU_NNZ"])

def read_exact(count):
    raw = sys.stdin.buffer.read(count)
    if len(raw) != count:
        raise RuntimeError(f"short read: wanted {count}, got {len(raw)}")
    return raw

indptr_raw = read_exact((N + 1) * 8)
indices_raw = read_exact(NNZ * 8)
data_raw = read_exact(NNZ * 8)
rhs_raw = read_exact(N * 8)
FIXTURE_SHA256 = hashlib.sha256(indptr_raw + indices_raw + data_raw + rhs_raw).hexdigest()

indptr = np.frombuffer(indptr_raw, dtype="<u8").astype(np.int32)
indices = np.frombuffer(indices_raw, dtype="<u8").astype(np.int32)
data = np.frombuffer(data_raw, dtype="<f8").copy()
rhs = np.frombuffer(rhs_raw, dtype="<f8").copy()
A = sp.csc_matrix((data, indices, indptr), shape=(N, N))
if A.nnz != NNZ:
    raise RuntimeError(f"nnz mismatch after assembly: {A.nnz} != {NNZ}")

# Warmup is outside every timer: the first factorization pays SuperLU setup.
lu = spla.splu(A)
x = lu.solve(rhs)
if x.shape != (N,) or not np.all(np.isfinite(x)):
    raise RuntimeError("warmup solve produced a non-finite solution")

with open(_linsolve.__file__, "rb") as handle:
    ENGINE_SHA256 = hashlib.sha256(handle.read()).hexdigest()
fsci_loaded = any(name.startswith("fsci") or name.startswith("franken") for name in sys.modules)

print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" engine_module={_linsolve.__name__}"
    f" scipy_engine_sha256={ENGINE_SHA256}"
    f" fixture_sha256={FIXTURE_SHA256}"
    f" n={N} nnz={NNZ}"
    f" lu_nnz={int(lu.L.nnz + lu.U.nnz)}"
    f" observed_os_tasks={len(os.listdir('/proc/self/task'))}"
    f" affinity={len(os.sched_getaffinity(0))}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={scipy.__version__ == '1.17.1' and not fsci_loaded}",
    flush=True,
)

for raw_line in sys.stdin.buffer:
    fields = str(raw_line, "ascii").strip().split()
    if not fields:
        continue
    if fields[0] == "FACTOR":
        started = time.perf_counter_ns()
        f = spla.splu(A)
        elapsed = time.perf_counter_ns() - started
        print(f"FACTOR {elapsed} {int(f.L.nnz + f.U.nnz)}", flush=True)
    elif fields[0] == "SOLVE":
        sol = spla.splu(A).solve(rhs)
        bits = np.asarray(sol, dtype="<f8").view("<u8")
        print("SOLVE " + ",".join(f"{int(v):016x}" for v in bits), flush=True)
    elif fields[0] == "QUIT":
        print("BYE", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {fields!r}")
"#;

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    impl Scipy {
        fn start(n: usize, nnz: usize, payload: &[u8]) -> (Self, String) {
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
            let mut child = command
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("FSCI_SPLU_N", n.to_string())
                .env("FSCI_SPLU_NNZ", nnz.to_string())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap_or_else(|error| panic!("spawn live SciPy oracle {python}: {error}"));
            let mut stdin = child.stdin.take().expect("SciPy stdin");
            stdin
                .write_all(payload)
                .and_then(|()| stdin.flush())
                .expect("send fixture to SciPy");
            let mut stdout = BufReader::new(child.stdout.take().expect("SciPy stdout"));
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
                },
                ready.trim().to_string(),
            )
        }

        fn request(&mut self, command: &str) -> String {
            writeln!(self.stdin, "{command}")
                .and_then(|()| self.stdin.flush())
                .unwrap_or_else(|error| panic!("send {command}: {error}"));
            let mut response = String::new();
            self.stdout
                .read_line(&mut response)
                .unwrap_or_else(|error| panic!("read {command}: {error}"));
            assert!(!response.is_empty(), "SciPy oracle exited during {command}");
            response.trim().to_string()
        }

        /// Nanoseconds for ONE SciPy factorization, timed by SciPy's own
        /// `perf_counter_ns` so the pipe round trip is outside the number.
        fn factor(&mut self) -> (u64, usize) {
            let response = self.request("FACTOR");
            let fields: Vec<&str> = response.split_whitespace().collect();
            assert!(
                fields.len() == 3 && fields[0] == "FACTOR",
                "malformed FACTOR response: {response}"
            );
            (
                fields[1].parse().expect("SciPy elapsed ns"),
                fields[2].parse().expect("SciPy LU nnz"),
            )
        }

        fn solution(&mut self) -> Vec<f64> {
            let response = self.request("SOLVE");
            let payload = response
                .strip_prefix("SOLVE ")
                .unwrap_or_else(|| panic!("malformed SOLVE response"));
            payload
                .split(',')
                .map(|value| {
                    u64::from_str_radix(value, 16)
                        .map(f64::from_bits)
                        .unwrap_or_else(|error| panic!("parse SciPy solution bits: {error}"))
                })
                .collect()
        }

        fn stop(&mut self) {
            if self.stopped {
                return;
            }
            let response = self.request("QUIT");
            assert_eq!(response, "BYE", "malformed SciPy shutdown");
            let status = self.child.wait().expect("wait for SciPy oracle");
            assert!(status.success(), "SciPy oracle exited with {status}");
            self.stopped = true;
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

    /// 3-D 7-point Laplacian on a `side³` grid — the same fixture family the
    /// cubic-live splu gate uses, so this row is comparable to that one.
    fn laplacian_3d_cubic(side: usize) -> CscMatrix {
        let n = side * side * side;
        let index = |z: usize, y: usize, x: usize| (z * side + y) * side + x;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let i = index(z, y, x);
                    rows.push(i);
                    cols.push(i);
                    data.push(6.001);
                    for (dz, dy, dx) in [
                        (-1_i64, 0_i64, 0_i64),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ] {
                        let (nz, ny, nx) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
                        if (0..side as i64).contains(&nz)
                            && (0..side as i64).contains(&ny)
                            && (0..side as i64).contains(&nx)
                        {
                            rows.push(i);
                            cols.push(index(nz as usize, ny as usize, nx as usize));
                            data.push(-1.0);
                        }
                    }
                }
            }
        }
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("cubic Laplacian triplets")
            .to_csr()
            .expect("cubic CSR")
            .to_csc()
            .expect("cubic CSC")
    }

    /// Pentadiagonal whose row/column labels are scrambled by a fixed symmetric
    /// permutation: identical nnz-per-row to a band matrix, but no grid pattern
    /// for `splu`'s structure detector to match, so this arm always measures the
    /// GENERAL sparse LU. It is the negative control for the cubic fixture — a
    /// structure-specific fast path that also "wins" here would be matching on
    /// something other than the structure it claims.
    fn scattered_pentadiagonal(side: usize) -> CscMatrix {
        let n = side * side * side;
        let mut permutation: Vec<usize> = (0..n).collect();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for index in (1..n).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            permutation.swap(index, (state >> 11) as usize % (index + 1));
        }
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            for offset in [-2_i64, -1, 0, 1, 2] {
                let j = i as i64 + offset;
                if j >= 0 && (j as usize) < n {
                    rows.push(permutation[i]);
                    cols.push(permutation[j as usize]);
                    data.push(if offset == 0 { 6.0 } else { -1.0 });
                }
            }
        }
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("scattered pentadiagonal triplets")
            .to_csr()
            .expect("scattered CSR")
            .to_csc()
            .expect("scattered CSC")
    }

    fn build_fixture(fixture: &str, side: usize) -> CscMatrix {
        match fixture {
            "cubic" => laplacian_3d_cubic(side),
            "scattered" => scattered_pentadiagonal(side),
            other => panic!("unknown fixture {other:?}"),
        }
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let len = sorted.len();
        if len % 2 == 1 {
            sorted[len / 2]
        } else {
            0.5 * (sorted[len / 2 - 1] + sorted[len / 2])
        }
    }

    /// Deterministic bootstrap percentile interval of the median. Seeded LCG so
    /// two runs of the same data report the same interval.
    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
        const ITERS: usize = 4000;
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut medians = Vec::with_capacity(ITERS);
        let mut sample = vec![0.0_f64; values.len()];
        for _ in 0..ITERS {
            for cell in &mut sample {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *cell = values[(state >> 33) as usize % values.len()];
            }
            medians.push(median(&sample));
        }
        medians.sort_by(f64::total_cmp);
        (
            medians[(0.025 * ITERS as f64) as usize],
            medians[(0.975 * ITERS as f64) as usize],
        )
    }

    /// Mean busy fraction across every CPU over one 300 ms window, from
    /// `/proc/stat`. Reported, never gated on: this substrate exists precisely
    /// because gating on host-wide quiescence is unsatisfiable here, and a row
    /// that hides how busy the box was is worse than one that states it.
    fn host_mean_busy() -> f64 {
        let sample = || -> Vec<(u64, u64)> {
            std::fs::read_to_string("/proc/stat").map_or_else(
                |_| Vec::new(),
                |text| {
                    text.lines()
                        .filter(|line| line.starts_with("cpu") && !line.starts_with("cpu "))
                        .filter_map(|line| {
                            let fields: Vec<u64> = line
                                .split_whitespace()
                                .skip(1)
                                .filter_map(|value| value.parse().ok())
                                .collect();
                            let total: u64 = fields.iter().sum();
                            fields.get(3).map(|idle| (total, *idle))
                        })
                        .collect()
                },
            )
        };
        let before = sample();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let after = sample();
        if before.is_empty() || before.len() != after.len() {
            return f64::NAN;
        }
        let busy: f64 = before
            .iter()
            .zip(&after)
            .map(|((t0, i0), (t1, i1))| {
                let total = t1.saturating_sub(*t0);
                if total == 0 {
                    0.0
                } else {
                    1.0 - i1.saturating_sub(*i0) as f64 / total as f64
                }
            })
            .sum();
        busy / before.len() as f64
    }

    /// `(physical_cores, logical_threads, ram_bytes, numa_count)`, all observed
    /// from `/proc` and `/sys` inside this process.
    fn hardware() -> (usize, usize, u64, usize) {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let logical = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        let mut physical_ids = Vec::new();
        let (mut package, mut core) = (None, None);
        for line in cpuinfo.lines() {
            if let Some(value) = line.strip_prefix("physical id") {
                package = value.split(':').nth(1).map(|v| v.trim().to_string());
            } else if let Some(value) = line.strip_prefix("core id") {
                core = value.split(':').nth(1).map(|v| v.trim().to_string());
            }
            if let (Some(p), Some(c)) = (package.as_ref(), core.as_ref()) {
                let key = format!("{p}:{c}");
                if !physical_ids.contains(&key) {
                    physical_ids.push(key);
                }
                package = None;
                core = None;
            }
        }
        let ram_bytes = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("MemTotal:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .map_or(0, |kb| kb * 1024);
        let numa = std::fs::read_dir("/sys/devices/system/node")
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .strip_prefix("node")
                            .is_some_and(|rest| rest.chars().all(|c| c.is_ascii_digit()))
                    })
                    .count()
            })
            .unwrap_or(0);
        (
            if physical_ids.is_empty() {
                logical
            } else {
                physical_ids.len()
            },
            logical,
            ram_bytes,
            numa.max(1),
        )
    }

    fn read_trimmed(path: &str) -> String {
        std::fs::read_to_string(path).map_or_else(
            |_| "unavailable".to_string(),
            |text| text.trim().to_string(),
        )
    }

    /// A busy host is admissible only when the balanced-square nulls show that
    /// its drift was shared by both arms.  This is deliberately separate from
    /// the physical-load observations printed below: those remain provenance,
    /// not an impossible admission predicate on a shared machine.
    fn balanced_square_quiescence(null_scipy: f64, null_fsci: f64) -> &'static str {
        if (null_scipy - 1.0).abs() <= NULL_BOUND && (null_fsci - 1.0).abs() <= NULL_BOUND {
            "clear"
        } else {
            "null-failed"
        }
    }

    pub fn run() {
        let exe = std::env::current_exe().expect("current_exe");
        let elf_sha256 = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&exe).expect("read own ELF"))
        );
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!("elf_path={}", exe.display());

        let args: Vec<String> = std::env::args().collect();
        let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
        let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(41);
        let warmup: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
        assert!(side >= 4, "side must be at least 4");
        assert!(
            rounds >= 9,
            "a bootstrap CI over fewer than 9 rounds is noise"
        );

        // Provenance, self-reported from inside this process. `observed_*` are
        // read, never requested — a requested thread count proves nothing.
        let (physical_cores, logical_threads, ram_bytes, numa_count) = hardware();
        println!(
            "provenance: host_identity={} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={ram_bytes} numa_count={numa_count} \
             scaling_governor={} runtime_isa={} requested_frankenscipy_threads=1 \
             actual_observed_frankenscipy_threads={} affinity={} loadavg={}",
            read_trimmed("/proc/sys/kernel/hostname"),
            read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            if std::arch::is_x86_feature_detected!("avx512f") {
                "avx512f+avx2+fma"
            } else if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                "avx2+fma"
            } else {
                "baseline"
            },
            std::fs::read_dir("/proc/self/task").map_or(0, Iterator::count),
            std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
            read_trimmed("/proc/loadavg"),
        );
        // WHICH ALGORITHM IS UNDER TEST. `splu` silently routes an exact cubic
        // Dirichlet grid to `CubicSpectralLu` — an O(n log n) structure-specific
        // solver, not a general LU. `SparseLuFactorization::backend_used` reports
        // `CubicSpectralLu` for this path, so the process output distinguishes it
        // from the general native LU. Measured 2026-08-14: on this fixture the spectral path reports
        // 204x against SuperLU while retaining a "factorization" with zero fill.
        // That is a real capability but it is NOT the general-splu number, so the
        // arm is named on the command line and proven by the hit counter below.
        let fixture = args.get(5).map(String::as_str).unwrap_or("cubic");
        let fixture_name = match fixture {
            "cubic" => "laplacian_3d_cubic",
            "scattered" => "scattered_pentadiagonal",
            other => panic!("fixture must be `cubic` or `scattered`, got {other:?}"),
        };
        let spectral_arg = args.get(4).map(String::as_str).unwrap_or("off");
        let spectral_enabled = match spectral_arg {
            "on" => true,
            "off" => false,
            other => panic!("spectral arm must be `on` or `off`, got {other:?}"),
        };
        SPLU_CUBIC_SPECTRAL_DISABLE.store(!spectral_enabled, Ordering::Relaxed);
        println!(
            "fixture={fixture_name} side={side} rounds={rounds} warmup={warmup} \
             arm={} (cubic spectral fast path {})",
            if spectral_enabled {
                "structured-fastpath"
            } else {
                "general-sparse-LU"
            },
            if spectral_enabled {
                "ENABLED"
            } else {
                "DISABLED"
            }
        );

        let spectral_hits_before = SPLU_CUBIC_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed);
        let matrix = build_fixture(fixture, side);
        let n = matrix.shape().rows;
        let indptr = matrix.indptr();
        let indices = matrix.indices();
        let data = matrix.data();
        let nnz = data.len();
        // Deterministic RHS with structure, so a wrong solve cannot hide behind a
        // constant vector.
        let rhs: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.125 * ((17 * i + 23) % 29) as f64)
            .collect();

        let mut payload = Vec::with_capacity((n + 1 + nnz) * 8 + nnz * 8 + n * 8);
        for value in indptr {
            payload.extend_from_slice(&(*value as u64).to_le_bytes());
        }
        for value in indices {
            payload.extend_from_slice(&(*value as u64).to_le_bytes());
        }
        for value in data {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        for value in &rhs {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        println!(
            "fixture_sha256={:x} n={n} nnz={nnz}",
            Sha256::digest(&payload)
        );

        let (mut scipy, ready) = Scipy::start(n, nnz, &payload);
        println!("scipy_identity: {ready}");
        assert!(
            ready.contains("genuine=True"),
            "the SciPy arm is not a genuine unpolluted scipy 1.17.1 interpreter"
        );

        // ── PARITY BEFORE TIMING ─────────────────────────────────────────────
        // A fast arm that factored a different matrix is not a result.
        let ours = splu(&matrix, LuOptions::default()).expect("fsci splu");
        let x_ours = splu_solve(&ours, &rhs).expect("fsci splu_solve");
        let x_theirs = scipy.solution();
        assert_eq!(x_ours.len(), x_theirs.len(), "solution length");
        let scale = x_theirs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let worst = x_ours
            .iter()
            .zip(&x_theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
            / scale.max(1.0);
        println!("parity: worst_rel_solution_diff={worst:.3e} scale={scale:.6e}");
        assert!(
            worst <= MAX_SOLUTION_REL_DIFF,
            "fsci and live SciPy disagree on the solution ({worst:.3e}) — no timing is admissible"
        );

        // ── BALANCED SQUARE ──────────────────────────────────────────────────
        for _ in 0..warmup {
            let _ = black_box(scipy.factor());
            let _ = black_box(splu(&matrix, LuOptions::default()).expect("fsci splu"));
        }

        let pre_busy = host_mean_busy();
        let mut ratios = Vec::with_capacity(rounds);
        let mut nulls_scipy = Vec::with_capacity(rounds);
        let mut nulls_fsci = Vec::with_capacity(rounds);
        let mut scipy_lu_nnz = 0usize;
        for _ in 0..rounds {
            let mut a_slots = Vec::with_capacity(4);
            let mut b_slots = Vec::with_capacity(4);
            for slot in SQUARE {
                if slot == b'A' {
                    let (ns, lu_nnz) = scipy.factor();
                    scipy_lu_nnz = lu_nnz;
                    a_slots.push(ns as f64);
                } else {
                    let started = Instant::now();
                    let factorization = splu(&matrix, LuOptions::default()).expect("fsci splu");
                    let elapsed = started.elapsed().as_nanos() as f64;
                    black_box(&factorization);
                    b_slots.push(elapsed);
                }
            }
            ratios.push(median(&a_slots) / median(&b_slots));
            // Each arm's own first-half / second-half ratio. The square places the
            // halves symmetrically, so a departure from 1.0 is drift or
            // contention, not slot position.
            nulls_scipy.push(median(&a_slots[..2]) / median(&a_slots[2..]));
            nulls_fsci.push(median(&b_slots[..2]) / median(&b_slots[2..]));
        }
        let post_busy = host_mean_busy();
        scipy.stop();

        let ratio = median(&ratios);
        let (low, high) = bootstrap_median_ci(&ratios);
        let null_scipy = median(&nulls_scipy);
        let null_fsci = median(&nulls_fsci);
        let quiescence = balanced_square_quiescence(null_scipy, null_fsci);
        let nulls_ok = quiescence == "clear";

        // EXECUTION PROOF. `backend_used` says which factorization actually ran:
        // `splu` densifies small or structurally-dense input to an n×n dense LU,
        // and a row that quietly measured the dense fallback against SuperLU's
        // sparse factorization is comparing two different algorithms. The retained
        // payload is the fill proxy — the dense fallback's is n²·8 bytes.
        // EXECUTION PROOF. `backend_used` distinguishes the general native sparse
        // LU from the structure-specific spectral solver. The spectral FACTOR-hit
        // counter remains an independent assertion that the selected route ran,
        // and `toggle_reads` proves the library read the toggle at all (a toggle
        // nothing loads makes both arms the same code).
        // The retained payload is the fill proxy: a "factorization" with no fill
        // did not do the elimination SuperLU did.
        let spectral_hits =
            SPLU_CUBIC_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed) - spectral_hits_before;
        println!(
            "execution_proof: fsci_backend={:?} fsci_ordering={:?} \
             fsci_lu_payload_bytes={} scipy_lu_nnz={scipy_lu_nnz} \
             cubic_spectral_factor_hits={spectral_hits} \
             cubic_spectral_toggle_reads={} arm_expected_spectral={spectral_enabled}",
            ours.backend_used,
            ours.ordering_used,
            splu_factor_payload_bytes(&ours),
            SPLU_CUBIC_SPECTRAL_DISABLE.load_count(),
        );
        assert!(
            spectral_enabled || spectral_hits == 0,
            "the spectral path ran {spectral_hits} times in an arm that disabled it — \
             SPLU_CUBIC_SPECTRAL_DISABLE did not take effect and this row measures the \
             wrong algorithm"
        );
        assert!(
            !(spectral_enabled && fixture == "cubic") || spectral_hits > 0,
            "the structured-fastpath arm never hit the spectral path on the cubic \
             fixture — the row would claim an algorithm that did not run"
        );
        // Reported, never gated on — see `host_mean_busy`.
        println!(
            "pre_measurement_quiescence=NOT_CERTIFIED(host_mean_busy={pre_busy:.3}) \
             post_measurement_quiescence=NOT_CERTIFIED(host_mean_busy={post_busy:.3})"
        );
        // The decision below is the bootstrap-median CI against a 2x null margin.
        // CV is not computed and would be provenance only if it were.
        let null_edge = (null_scipy - 1.0).abs().max((null_fsci - 1.0).abs());
        let required_low = 1.0 + 2.0 * null_edge;
        let required_high = 1.0 - 2.0 * null_edge;
        println!(
            "NULL scipy/scipy={null_scipy:.4} NULL fsci/fsci={null_fsci:.4} bound=+/-{NULL_BOUND} \
             null_edge={null_edge:.4} decided_if_ci_lo>{required_low:.4}_or_ci_hi<{required_high:.4}"
        );
        println!(
            "quiescence={quiescence} method=balanced-square \
             criterion=both_A/A_nulls_within_{NULL_BOUND:.3}"
        );
        let verdict = if !nulls_ok {
            "NULL-FAILED (row void)"
        } else if low <= required_low && high >= required_high {
            "IN-FLOOR (inside the 2x A/A-null margin)"
        } else if ratio > 1.0 {
            "ADMISSIBLE: FrankenSciPy FASTER"
        } else {
            "ADMISSIBLE: FrankenSciPy SLOWER"
        };
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio:.4}x  ci95=[{low:.4},{high:.4}]  \
             rounds={rounds}  verdict={verdict}"
        );
    }

    #[cfg(test)]
    mod tests {
        use super::balanced_square_quiescence;

        #[test]
        fn balanced_square_nulls_admit_shared_host_drift() {
            assert_eq!(balanced_square_quiescence(0.9958, 0.9976), "clear");
        }

        #[test]
        fn balanced_square_nulls_reject_order_bias() {
            assert_eq!(balanced_square_quiescence(1.021, 1.0), "null-failed");
        }
    }
}

fn main() {
    bench::run();
}
