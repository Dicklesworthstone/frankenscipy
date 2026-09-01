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
//! Run: `cargo build --release -p fsci-sparse --bin perf_splu`, then invoke the
//! executable directly. The bin is named `perf_splu`, NOT after this file — the
//! previous instruction here named the file and would simply fail. `--features
//! sparse-incumbent-bench` is not needed either; that feature is an empty
//! compatibility stub (see this crate's `Cargo.toml`).
//!
//! Do NOT measure in a window you built in, and take the executable path from
//! `--message-format=json` rather than assuming it: a stale binary from a previous
//! turn's temporary edit has been profiled by mistake once, caught only because the
//! ELF SHA-256 is checked before every profile.

mod bench {
    use fsci_sparse::{
        CooMatrix, CscMatrix, FormatConvertible, LuOptions, PermutationOrdering,
        SPLU_BACK_MERGE_ENABLE, SPLU_BACK_MERGE_FACTOR_HITS, SPLU_BANDED_ENABLE,
        SPLU_BANDED_FACTOR_HITS, SPLU_BANDED_STAGE_NANOS, SPLU_BANDED_STAGE_TIMING,
        SPLU_BANDED_UNPACK_RESERVE, SPLU_BANDED_UNPACK_RESERVE_HITS, SPLU_CONTIGUOUS_SOLVE_ENABLE,
        SPLU_CONTIGUOUS_SOLVE_HITS, SPLU_CUBIC_SPECTRAL_DISABLE, SPLU_CUBIC_SPECTRAL_FACTOR_HITS,
        SPLU_PARTIAL_INPLACE_ENABLE, SPLU_PARTIAL_INPLACE_FACTOR_HITS, SPLU_ROW_HEAD_CACHE_DISABLE,
        SPLU_ROW_HEAD_CACHE_FACTOR_HITS, SPLU_SUPERNODAL_ENABLE, SPLU_SUPERNODAL_FACTOR_HITS,
        SPLU_SWAP_WRITEBACK_ENABLE, SPLU_SWAP_WRITEBACK_HITS, Shape2D, splu,
        splu_factor_payload_bytes, splu_solve,
    };
    use sha2::{Digest, Sha256};

    /// Ordering for the fsci arm of the balanced-square factor cell, selected by
    /// `FSCI_SPLU_ORDERING` and defaulting to the library default.
    ///
    /// EXISTS TO TEST A STATED MECHANISM, NOT TO TUNE. frankenscipy-llywn records this cell's
    /// mechanism as settled -- "the fill-reducing ordering is NOT the problem" -- on the grounds
    /// that our fill is at SuperLU parity. That is true of the SHIPPING ordering: `Colamd` maps
    /// to RCM here and gives 1,188,312 factor nonzeros against SuperLU's 1,231,312, 0.965x. AMD,
    /// which did not exist when that was written, gives 648,372 -- 1.83x less. This harness
    /// hardcoded `LuOptions::default()` at every call site and so could not measure any ordering
    /// but the one whose exhaustion it was being cited to prove. Unset, every arm runs exactly as
    /// it ships.
    fn balanced_arm_options() -> LuOptions {
        let ordering = match std::env::var("FSCI_SPLU_ORDERING").ok().as_deref() {
            None | Some("") | Some("default") => LuOptions::default().ordering,
            Some("colamd") => PermutationOrdering::Colamd,
            Some("rcm") => PermutationOrdering::ReverseCuthillMcKee,
            Some("mmd-ata") => PermutationOrdering::MmdAta,
            Some("mmd-at-plus-a") => PermutationOrdering::MmdAtPlusA,
            Some("amd") => PermutationOrdering::Amd,
            Some("natural") => PermutationOrdering::Natural,
            Some(other) => panic!(
                "FSCI_SPLU_ORDERING={other:?} is not one of \
                 default|colamd|rcm|mmd-ata|mmd-at-plus-a|amd|natural"
            ),
        };
        LuOptions {
            ordering,
            ..LuOptions::default()
        }
    }

    use fsci_runtime::scipy_incumbent::ScipyIncumbent;
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// Slot order within one round. Each arm gets four slots, and the two arms
    /// occupy mirror-image positions, so a monotone drift across the round
    /// cancels in the per-round ratio.
    const SQUARE: [u8; 8] = *b"ABBAABBA";
    /// A per-arm A/A null must land within this of 1.0 or the row is void.
    const NULL_BOUND: f64 = 0.02;
    /// Both factorizations solve the same RHS before any timing; they use
    /// different orderings and pivot thresholds, so agreement is to solve
    /// accuracy, not to bits.
    const MAX_SOLUTION_REL_DIFF: f64 = 1.0e-9;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Fixture {
        Cubic,
        Scattered,
        /// run7d's cell. Added so the worst measured vs-SciPy loss in this repo can be
        /// measured at all: its sanctioned harness (`perf_spsolve --convection-splu-live`)
        /// hard-requires `TRJ_BOOKING_CLAIM_MESSAGE_ID` from the agent-mail build-slot
        /// service, which is disabled server-side (`frankenscipy-fr78g`). This harness exists
        /// precisely because those gates were unsatisfiable -- see the header -- so porting
        /// the cell here is the sanctioned route, not a way around a gate.
        Convection,
    }

    impl Fixture {
        const fn name(self) -> &'static str {
            match self {
                Self::Cubic => "laplacian_3d_cubic",
                Self::Scattered => "scattered_pentadiagonal",
                Self::Convection => "convection_diffusion_2d",
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct RunConfig {
        side: usize,
        rounds: usize,
        warmup: usize,
        spectral_enabled: bool,
        fixture: Fixture,
        /// Which side of `SPLU_ROW_HEAD_CACHE_DISABLE` the FrankenSciPy arm runs.
        ///
        /// Defaults to `true` — the shipping arm — so every previously recorded
        /// invocation of this harness means what it meant before this argument
        /// existed, and a self-A/B is opted into rather than defaulted into.
        head_cache_enabled: bool,
        /// Which side of `SPLU_BACK_MERGE_ENABLE` the FrankenSciPy arm runs.
        ///
        /// Defaults to `false`, matching the library default: the back-merge is
        /// unmeasured, so it is opted into here exactly as it is in the library.
        back_merge_enabled: bool,
        /// Which side of `SPLU_PARTIAL_INPLACE_ENABLE` the FrankenSciPy arm runs.
        ///
        /// Defaults to `true`, tracking the library default, which flipped when the
        /// lever shipped. This comment previously claimed `false` "matching the library
        /// default" and went stale the moment the library moved — which is exactly how a
        /// bare invocation came to measure the non-shipping arm while calling itself the
        /// shipping configuration.
        partial_inplace_enabled: bool,
        /// Which side of `SPLU_SUPERNODAL_ENABLE` the FrankenSciPy arm runs.
        /// Defaults to `false`, tracking the library default.
        supernodal_enabled: bool,
    }

    fn parse_optional_usize(
        args: &[String],
        index: usize,
        label: &str,
        default: usize,
    ) -> Result<usize, String> {
        args.get(index).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| format!("{label} must be an integer, got {value:?}"))
        })
    }

    const USAGE: &str = "\
usage: perf_splu [side] [rounds] [warmup] [on|off] [cubic|scattered] [on|off] [on|off] \
                 [on|off] [on|off]
       perf_splu aggregate <ratio> <ratio> <ratio> ...

  side      grid side (default 24, minimum 4)
  rounds    balanced-square rounds (default 41, minimum 9)
  warmup    untimed warmup rounds (default 4)
  on|off    cubic-spectral arm (default off)
  fixture   cubic | scattered (default cubic)
  on|off    row-head-cache arm (default on, the shipping layout)
  on|off    back-merge arm (default off, the unmeasured lever)
  on|off    partial-inplace-prefix arm (default on, tracks the library default)
  on|off    supernodal-blocking arm (default off, tracks the library default)

  aggregate combine ratios from INDEPENDENT invocations into one reportable
            figure: median with a bootstrap interval over the REPLICATES, plus
            the observed range. Refuses fewer than three. The per-run ci95 this
            harness prints bootstraps over rounds inside ONE process and is far
            too narrow to describe reproducibility -- replicates of one binary
            on one cell have been measured spanning 13-15%.

Prints elf_sha256, provenance, per-round ratios, both A/A nulls and a
bootstrap-median CI. The ELF SHA-256 is self-reported from inside the process
and is computed AFTER argument dispatch, so this message costs nothing.";

    /// Does this invocation only want the usage text?
    ///
    /// Split out so it can be tested against both arms: an argument list that
    /// MUST be treated as help, and a real measurement configuration that MUST
    /// NOT be.
    fn is_help_request(args: &[String]) -> bool {
        args.iter()
            .skip(1)
            .any(|arg| arg == "-h" || arg == "--help" || arg == "help")
    }

    fn parse_run_config(args: &[String]) -> Result<RunConfig, String> {
        if args.len() > 10 {
            return Err(format!(
                "expected at most nine arguments: [side] [rounds] [warmup] [on|off] \
                 [cubic|scattered] [on|off] [on|off] [on|off] [on|off], got {}",
                args.len() - 1
            ));
        }

        let side = parse_optional_usize(args, 1, "side", 24)?;
        let rounds = parse_optional_usize(args, 2, "rounds", 41)?;
        let warmup = parse_optional_usize(args, 3, "warmup", 4)?;
        if side < 4 {
            return Err("side must be at least 4".to_string());
        }
        if rounds < 9 {
            return Err("a bootstrap CI over fewer than 9 rounds is noise".to_string());
        }

        let spectral_enabled = match args.get(4).map(String::as_str).unwrap_or("off") {
            "on" => true,
            "off" => false,
            other => return Err(format!("spectral arm must be `on` or `off`, got {other:?}")),
        };
        let fixture = match args.get(5).map(String::as_str).unwrap_or("cubic") {
            "cubic" => Fixture::Cubic,
            "scattered" => Fixture::Scattered,
            "convection" => Fixture::Convection,
            other => {
                return Err(format!(
                    "fixture must be `cubic`, `scattered` or `convection`, got {other:?}"
                ));
            }
        };

        // The head-cache arm. `on` is the shipping layout and the default, so an
        // invocation written before this argument existed selects exactly the code
        // it selected then; `off` routes the elimination's candidate lookups back
        // through the 56-byte row headers (frankenscipy-u7biq). The two arms are
        // BIT-IDENTICAL by contract, which is what makes this a legal self-A/B:
        // the parity gate below compares the same solution on both.
        let head_cache_enabled = match args.get(6).map(String::as_str).unwrap_or("on") {
            "on" => true,
            "off" => false,
            other => {
                return Err(format!(
                    "row-head-cache arm must be `on` or `off`, got {other:?}"
                ));
            }
        };

        // EVERY DEFAULT BELOW TRACKS THE LIBRARY DEFAULT, and the tracking is pinned by
        // `bare_invocation_defaults_track_the_library_toggles` rather than by these
        // comments. This mattered once already: when the partial-inplace argument was
        // added the library default was OFF and `"off"` here matched it; the library
        // flipped to ON and this did not, so a bare invocation silently measured the
        // NON-shipping arm and reported `partial_inplace_factor_hits=0` while calling
        // itself the shipping configuration. Only the hit counter caught it, and only
        // after the fact. A harness default that has drifted from the library default is
        // worse than having no default at all, because it still prints a provenance line.
        //
        // These four are stored UNCONDITIONALLY at the run site (unlike `FSCI_SPLU_BANDED`,
        // which only overrides when asked), so the literal tokens here ARE the arm a bare
        // invocation measures. The drift test reads the library statics and fails closed if
        // any of them stops agreeing with the token on its own line.

        // The back-merge arm (frankenscipy-xup61); `SPLU_BACK_MERGE_ENABLE` defaults false.
        let back_merge_enabled = match args.get(7).map(String::as_str).unwrap_or("off") {
            "on" => true,
            "off" => false,
            other => {
                return Err(format!(
                    "back-merge arm must be `on` or `off`, got {other:?}"
                ));
            }
        };

        // The partial-inplace-prefix arm; `SPLU_PARTIAL_INPLACE_ENABLE` defaults true.
        let partial_inplace_enabled = match args.get(8).map(String::as_str).unwrap_or("on") {
            "on" => true,
            "off" => false,
            other => {
                return Err(format!(
                    "partial-inplace arm must be `on` or `off`, got {other:?}"
                ));
            }
        };

        // The supernodal-blocking arm; `SPLU_SUPERNODAL_ENABLE` defaults false. Memory of
        // this crate: the shipping arm here is BANDED, and rows have been priced against
        // supernodal twice while `supernodal_factor_hits=0` — so this default is load-bearing.
        let supernodal_enabled = match args.get(9).map(String::as_str).unwrap_or("off") {
            "on" => true,
            "off" => false,
            other => {
                return Err(format!(
                    "supernodal arm must be `on` or `off`, got {other:?}"
                ));
            }
        };

        Ok(RunConfig {
            side,
            rounds,
            warmup,
            spectral_enabled,
            head_cache_enabled,
            back_merge_enabled,
            partial_inplace_enabled,
            supernodal_enabled,
            fixture,
        })
    }

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
    f" genuine={scipy.__version__ == '1.17.1' and np.__version__ == '2.4.3' and not fsci_loaded}",
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
    elif fields[0] == "SOLVETIME":
        # Times k solves on the ALREADY-FACTORED `lu` built during warmup, so the factorization
        # is outside the timer on both arms. This is the only timed path in this harness that
        # measures the triangular solves rather than the factorization.
        k = int(fields[1])
        started = time.perf_counter_ns()
        for _ in range(k):
            y = lu.solve(rhs)
        elapsed = time.perf_counter_ns() - started
        print(f"SOLVETIME {elapsed} {int(y[0].view('<u8')):016x}", flush=True)
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

    /// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
    /// installation whose compiled submodules do not load, and that difference would
    /// otherwise only surface mid-timing.
    const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.sparse.linalg"];

    /// The one live-SciPy incumbent this process compares against.
    ///
    /// This harness is where the stale-discovery failure was first caught, and it carried the
    /// only working resolver in the workspace while 26 neighbours still hard-coded paths that
    /// no longer exist. That resolver now lives in `fsci_runtime::scipy_incumbent` so every
    /// harness answers the question the same way; this is the same mechanism, not a new one,
    /// and it additionally reports the NumPy version the row needs to be comparable
    /// (frankenscipy-m5s54).
    fn incumbent() -> &'static ScipyIncumbent {
        static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
        INCUMBENT.get_or_init(|| {
            let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
                .unwrap_or_else(|error| panic!("{error}"));
            println!("{}", resolved.provenance_line());
            resolved
        })
    }

    impl Scipy {
        fn start(n: usize, nnz: usize, payload: &[u8]) -> (Self, String) {
            let incumbent = incumbent();
            let python = incumbent.python.clone();
            let mut command = incumbent.command();
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

        /// Nanoseconds for `count` solves on the warm factor. The factorization is NOT in the
        /// timed region on either arm.
        fn solve_time(&mut self, count: usize) -> u64 {
            let response = self.request(&format!("SOLVETIME {count}"));
            let fields: Vec<&str> = response.split_whitespace().collect();
            assert!(
                fields.len() == 3 && fields[0] == "SOLVETIME",
                "malformed SOLVETIME response: {response}"
            );
            fields[1].parse().expect("SciPy solve elapsed ns")
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

    /// The same 5-point convection-diffusion stencil `perf_spsolve` builds for run7d:
    /// asymmetric off-diagonals (WEST -1.2, EAST -0.8) so the pattern is genuinely
    /// nonsymmetric, diagonal 4.001. Copied rather than shared because the two binaries do
    /// not share a module; the constants are kept identical on purpose so this measures the
    /// same cell the bead is about.
    fn convection_diffusion_2d(side: usize) -> CscMatrix {
        let n = side * side;
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..side {
            for column in 0..side {
                let index = row * side + column;
                if row > 0 {
                    rows.push(index);
                    cols.push(index - side);
                    data.push(-1.0);
                }
                if column > 0 {
                    rows.push(index);
                    cols.push(index - 1);
                    data.push(-1.2);
                }
                rows.push(index);
                cols.push(index);
                data.push(4.001);
                if column + 1 < side {
                    rows.push(index);
                    cols.push(index + 1);
                    data.push(-0.8);
                }
                if row + 1 < side {
                    rows.push(index);
                    cols.push(index + side);
                    data.push(-1.0);
                }
            }
        }
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("convection triplets")
            .to_csr()
            .expect("convection CSR")
            .to_csc()
            .expect("convection CSC")
    }

    /// Which half of the job the square times.
    ///
    /// `FSCI_SPLU_STAGE=solve` times the triangular solves on an already-built factor; anything
    /// else times the factorization, which is what this harness has always done. An env var
    /// rather than a positional argument so every existing invocation keeps its meaning.
    ///
    /// WHY THE SOLVE NEEDED ITS OWN STAGE: run7d is a whole-job bead -- factor plus sixteen
    /// solves -- and its factorization measures 0.4629x against live SciPy. Nothing in this repo
    /// had ever timed its SOLVE against a live incumbent; the `SOLVE` command here existed only
    /// to compare solution bits for parity, and the split probe that does time our solves has no
    /// SciPy arm at all. So the solve half of the worst measured loss was unmeasured.
    fn solve_stage_requested() -> bool {
        matches!(
            std::env::var("FSCI_SPLU_STAGE").ok().as_deref(),
            Some("solve")
        )
    }

    fn build_fixture(fixture: &str, side: usize) -> CscMatrix {
        match fixture {
            "cubic" => laplacian_3d_cubic(side),
            "scattered" => scattered_pentadiagonal(side),
            "convection" => convection_diffusion_2d(side),
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

    /// Aggregate ACROSS replicate runs, which is the level the variance actually lives at.
    ///
    /// WHY THIS EXISTS (frankenscipy-llywn, 2026-08-17). Every row this harness prints
    /// bootstraps over ROUNDS INSIDE ONE INVOCATION, and that interval is far too narrow
    /// to describe the cell: four admissible rows on one ELF, same fixture, same
    /// rounds=41, taken within ten minutes, read **0.5230, 0.5380, 0.5775, 0.4950** — a
    /// 1.167x span — while each individual CI was about 0.02 wide. Within-invocation
    /// bootstrap measures how stable the rounds were, **not** how reproducible the
    /// measurement is.
    ///
    /// AND THE CONVENTION BUILT ON THOSE ROWS WAS WORSE THAN THE ROWS. The standing bound
    /// was quoted as "the worst CI floor across all admissible rows", which is a **running
    /// minimum, not an estimator**: it moves one way only, so collecting more honest data
    /// makes the published number monotonically worse. On this cell it went
    /// 0.5291 → 0.5103 → 0.4905 purely by sampling more. This reports the **median of N
    /// replicates with a bootstrap interval over the replicates themselves**, which
    /// converges, plus the observed range so the spread is never hidden.
    ///
    /// Returns `(n, median, ci_lo, ci_hi, min, max)`.
    fn replicate_summary(ratios: &[f64]) -> (usize, f64, f64, f64, f64, f64) {
        let (lo, high) = bootstrap_median_ci(ratios);
        let min = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let max = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (ratios.len(), median(ratios), lo, high, min, max)
    }

    /// `perf_splu aggregate <ratio> <ratio> ...` — combine replicate ratios into one
    /// reportable figure. Deliberately a separate mode rather than a flag on a measuring
    /// run: the replicates it combines should come from independent invocations, ideally
    /// spread across windows, and folding them into one process would hide exactly the
    /// between-invocation variance this is built to expose.
    fn run_aggregate(raw: &[String]) -> Result<(), String> {
        let mut ratios = Vec::with_capacity(raw.len());
        for value in raw {
            let parsed: f64 = value
                .parse()
                .map_err(|_| format!("replicate ratio must be a number, got {value:?}"))?;
            if !parsed.is_finite() || parsed <= 0.0 {
                return Err(format!(
                    "replicate ratio must be finite and positive, got {parsed}"
                ));
            }
            ratios.push(parsed);
        }
        // Three is the floor at which a median and a range mean anything at all. Two
        // replicates cannot distinguish a spread from an outlier, and reporting a
        // confident-looking interval over them would recreate the problem this fixes.
        if ratios.len() < 3 {
            return Err(format!(
                "aggregating fewer than 3 replicates is not reportable, got {}",
                ratios.len()
            ));
        }
        let (n, med, lo, high, min, max) = replicate_summary(&ratios);
        println!(
            "replicate_aggregate: n={n} median={med:.4}x ci95_across_replicates=[{lo:.4},{high:.4}] \
             observed_range=[{min:.4},{max:.4}] spread={:.1}% \
             deficit_vs_incumbent={:.2}x",
            100.0 * (max - min) / med,
            1.0 / med
        );
        println!(
            "NOTE: the interval above is over REPLICATES, not over rounds within one \
             invocation. Do not quote a within-invocation ci95 as the reproducibility of \
             this cell, and do not quote a running worst-floor as a bound."
        );
        Ok(())
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
        // ARGUMENT DISPATCH COMES FIRST, and the ordering is the whole point.
        //
        // This function used to hash its own 1.2 MB ELF before looking at argv.
        // Measured with callgrind on the scattered cell that digest is 70,788,403
        // instructions, 30.84% of the entire process — more than `factorize_csr`
        // itself at 25.39% — so `--help` and every rejected selector paid for a
        // full SHA-256 of the binary, and any share-of-program figure taken on a
        // small cell was computed against a denominator that startup dominated.
        // It already cost one published number: a fill-crossover figure of "3.0%
        // vs 68.6% of program = 23x" is 8.9% vs 73.7% = 8.3x once measured
        // against the timed work instead (frankenscipy-ahimi).
        //
        // The digest is NOT dropped — it is the executed-ELF provenance the
        // ledger gate requires, and two sanctioned ELFs of this harness have
        // already read one cell 15% apart (frankenscipy-kapqa), which is exactly
        // why every row must name its binary. It is only moved behind the
        // dispatch, so an invocation that will not measure anything does not pay
        // for it. It remains one-time startup work outside every `Instant::now()`
        // region, as it always was, so no banked row changes.
        let args: Vec<String> = std::env::args().collect();
        if is_help_request(&args) {
            println!("{USAGE}");
            return;
        }
        if args.get(1).map(String::as_str) == Some("aggregate") {
            if let Err(error) = run_aggregate(&args[2..]) {
                eprintln!("invalid perf_splu aggregate invocation: {error}");
                std::process::exit(2);
            }
            return;
        }
        let config = match parse_run_config(&args) {
            Ok(config) => config,
            Err(error) => {
                eprintln!("invalid perf_splu measurement configuration: {error}");
                std::process::exit(2);
            }
        };

        let exe = std::env::current_exe().expect("current_exe");
        let elf_sha256 = format!(
            "{:x}",
            Sha256::digest(std::fs::read(&exe).expect("read own ELF"))
        );
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!("elf_path={}", exe.display());

        let RunConfig {
            side,
            rounds,
            warmup,
            spectral_enabled,
            fixture,
            head_cache_enabled,
            back_merge_enabled,
            partial_inplace_enabled,
            supernodal_enabled,
        } = config;

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
        let fixture_name = fixture.name();
        SPLU_CUBIC_SPECTRAL_DISABLE.store(!spectral_enabled, Ordering::Relaxed);
        // THE SELF-A/B ARM (frankenscipy-u7biq). Off routes every candidate lookup
        // in the elimination back through the 56-byte row header that carries 47.5%
        // of this kernel's D1 read misses; on reads the dense head projection
        // instead. Named on the command line and proven by the hit counter below,
        // exactly as the spectral arm is — a row that cannot show its toggle was
        // read is an A/A comparison wearing an A/B's label.
        SPLU_ROW_HEAD_CACHE_DISABLE.store(!head_cache_enabled, Ordering::Relaxed);
        let head_cache_hits_before = SPLU_ROW_HEAD_CACHE_FACTOR_HITS.load(Ordering::Relaxed);
        SPLU_ROW_HEAD_CACHE_DISABLE.reset_load_count();
        SPLU_BACK_MERGE_ENABLE.store(back_merge_enabled, Ordering::Relaxed);
        let back_merge_hits_before = SPLU_BACK_MERGE_FACTOR_HITS.load(Ordering::Relaxed);
        SPLU_BACK_MERGE_ENABLE.reset_load_count();
        SPLU_PARTIAL_INPLACE_ENABLE.store(partial_inplace_enabled, Ordering::Relaxed);
        let partial_inplace_hits_before = SPLU_PARTIAL_INPLACE_FACTOR_HITS.load(Ordering::Relaxed);
        SPLU_PARTIAL_INPLACE_ENABLE.reset_load_count();
        SPLU_SUPERNODAL_ENABLE.store(supernodal_enabled, Ordering::Relaxed);
        // `FSCI_SPLU_BANDED=1` routes the FrankenSciPy arm through the row-contiguous banded
        // factorization. It declines rather than guesses, so the hit count is printed below and a
        // silent fall-through cannot read as a measured result.
        // ONLY OVERRIDE WHEN ASKED. An unconditional `store` would pin the arm to this harness's
        // own default and silently measure something other than the shipping configuration --
        // which is exactly what happened once the library default flipped to ON: the row read
        // `banded_factor_hits=0` while the shipping path takes the banded route.
        let banded_override = match std::env::var("FSCI_SPLU_BANDED").ok().as_deref() {
            Some("1") | Some("true") => Some(true),
            Some("0") | Some("false") => Some(false),
            _ => None,
        };
        if let Some(value) = banded_override {
            SPLU_BANDED_ENABLE.store(value, Ordering::Relaxed);
        }
        // `FSCI_SPLU_SWAP_WRITEBACK=0` restores the copy-back writeback, so both arms of
        // that change live in ONE binary and can be alternated inside a single window
        // instead of compared across two builds in two windows. Same only-override-when-
        // asked rule as the banded arm directly above, and for the same reason.
        let swap_writeback_override =
            match std::env::var("FSCI_SPLU_SWAP_WRITEBACK").ok().as_deref() {
                Some("1") | Some("true") => Some(true),
                Some("0") | Some("false") => Some(false),
                _ => None,
            };
        if let Some(value) = swap_writeback_override {
            SPLU_SWAP_WRITEBACK_ENABLE.store(value, Ordering::Relaxed);
        }
        // `FSCI_SPLU_CONTIGUOUS_SOLVE=0` restores the indexed substitution, so both arms of
        // the solve change live in ONE binary. Same only-override-when-asked rule as above.
        let contiguous_solve_override =
            match std::env::var("FSCI_SPLU_CONTIGUOUS_SOLVE").ok().as_deref() {
                Some("1") | Some("true") => Some(true),
                Some("0") | Some("false") => Some(false),
                _ => None,
            };
        if let Some(value) = contiguous_solve_override {
            SPLU_CONTIGUOUS_SOLVE_ENABLE.store(value, Ordering::Relaxed);
        }
        let contiguous_solve_hits_before = SPLU_CONTIGUOUS_SOLVE_HITS.load(Ordering::Relaxed);
        let banded_requested = SPLU_BANDED_ENABLE.load(Ordering::Relaxed);
        let banded_hits_before = SPLU_BANDED_FACTOR_HITS.load(Ordering::Relaxed);
        // `FSCI_SPLU_UNPACK_RESERVE=0` restores the growing per-row vectors in the banded
        // unpack. Only-override-when-asked, so an unset variable keeps the shipping default.
        match std::env::var("FSCI_SPLU_UNPACK_RESERVE").ok().as_deref() {
            Some("1") | Some("true") => SPLU_BANDED_UNPACK_RESERVE.store(true, Ordering::Relaxed),
            Some("0") | Some("false") => {
                SPLU_BANDED_UNPACK_RESERVE.store(false, Ordering::Relaxed);
            }
            _ => {}
        }
        if std::env::var("FSCI_SPLU_BANDED_STAGES").is_ok_and(|v| v != "0") {
            for slot in &SPLU_BANDED_STAGE_NANOS {
                slot.store(0, Ordering::Relaxed);
            }
            SPLU_BANDED_STAGE_TIMING.store(true, Ordering::Relaxed);
        }
        // `FSCI_SPLU_SOLVE_STAGES=1` splits the SOLVE into forward / backward / unpermute and
        // reports CLOSURE first. Off by default so no clock is constructed on a normal row.
        if std::env::var("FSCI_SPLU_SOLVE_STAGES").is_ok_and(|v| v != "0") {
            for slot in &fsci_sparse::linalg::SPLU_SOLVE_STAGE_NANOS {
                slot.store(0, Ordering::Relaxed);
            }
            fsci_sparse::linalg::SPLU_SOLVE_STAGE_TOTAL_NANOS.store(0, Ordering::Relaxed);
            fsci_sparse::linalg::SPLU_SOLVE_STAGE_TIMING.store(true, Ordering::Relaxed);
        }

        let unpack_reserve_hits_before = SPLU_BANDED_UNPACK_RESERVE_HITS.load(Ordering::Relaxed);
        let swap_writeback_hits_before = SPLU_SWAP_WRITEBACK_HITS.load(Ordering::Relaxed);
        let supernodal_hits_before = SPLU_SUPERNODAL_FACTOR_HITS.load(Ordering::Relaxed);
        SPLU_SUPERNODAL_ENABLE.reset_load_count();
        println!(
            "supernodal_arm={}",
            if supernodal_enabled {
                "ENABLED (symbolic plan, blocked update)"
            } else {
                "DISABLED (per-pivot elimination)"
            }
        );
        println!(
            "partial_inplace_arm={}",
            if partial_inplace_enabled {
                "ENABLED (coincident prefix updated in place)"
            } else {
                "DISABLED (full merge through scratch)"
            }
        );
        println!(
            "back_merge_arm={}",
            if back_merge_enabled {
                "ENABLED (in-place merge from the back)"
            } else {
                "DISABLED (shared scratch plus copy-back)"
            }
        );
        println!(
            "row_head_cache_arm={}",
            if head_cache_enabled {
                "ENABLED (shipping dense head projection)"
            } else {
                "DISABLED (legacy 56-byte row-header lookups)"
            }
        );
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

        // Solves per timed slot. Sixteen because run7d's job is a factor plus sixteen solves,
        // so a slot here is one job's worth of solve work.
        const SOLVES_PER_SLOT: usize = 16;
        let solve_stage = solve_stage_requested();
        let spectral_hits_before = SPLU_CUBIC_SPECTRAL_FACTOR_HITS.load(Ordering::Relaxed);
        let matrix = build_fixture(
            match fixture {
                Fixture::Cubic => "cubic",
                Fixture::Scattered => "scattered",
                Fixture::Convection => "convection",
            },
            side,
        );
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
        let ours = splu(&matrix, balanced_arm_options()).expect("fsci splu");
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
        println!("fsci_arm_ordering={:?}", ours.ordering_used);
        assert!(
            worst <= MAX_SOLUTION_REL_DIFF,
            "fsci and live SciPy disagree on the solution ({worst:.3e}) — no timing is admissible"
        );

        // ── BALANCED SQUARE ──────────────────────────────────────────────────
        // In solve mode the factor is built once, outside every timer, mirroring the SciPy arm's
        // warm `lu`. `ours` above is exactly that factor and it already passed the parity gate.
        let warm_factor = &ours;
        for _ in 0..warmup {
            if solve_stage {
                let _ = black_box(splu_solve(&warm_factor, &rhs).expect("fsci splu_solve"));
                let _ = black_box(scipy.solve_time(SOLVES_PER_SLOT));
            } else {
                let _ = black_box(scipy.factor());
                let _ = black_box(splu(&matrix, balanced_arm_options()).expect("fsci splu"));
            }
        }

        let pre_busy = host_mean_busy();
        let mut ratios = Vec::with_capacity(rounds);
        let mut nulls_scipy = Vec::with_capacity(rounds);
        let mut nulls_fsci = Vec::with_capacity(rounds);
        // Absolute per-round medians, kept because a RATIO alone does not say
        // whether a cell is slow in nanoseconds or in milliseconds, and the two
        // call for completely different levers. Borrowed from frankenfs.
        let mut scipy_ns = Vec::with_capacity(rounds);
        let mut fsci_ns = Vec::with_capacity(rounds);
        let mut scipy_lu_nnz = 0usize;
        for _ in 0..rounds {
            let mut a_slots = Vec::with_capacity(4);
            let mut b_slots = Vec::with_capacity(4);
            for slot in SQUARE {
                if slot == b'A' {
                    if solve_stage {
                        a_slots.push(scipy.solve_time(SOLVES_PER_SLOT) as f64);
                    } else {
                        let (ns, lu_nnz) = scipy.factor();
                        scipy_lu_nnz = lu_nnz;
                        a_slots.push(ns as f64);
                    }
                } else if solve_stage {
                    // The factor is built ONCE outside every timer, exactly as the SciPy arm
                    // keeps its warm `lu`, so this times the triangular solves alone.
                    let started = Instant::now();
                    for _ in 0..SOLVES_PER_SLOT {
                        let solution = splu_solve(&warm_factor, &rhs).expect("fsci splu_solve");
                        black_box(&solution);
                    }
                    b_slots.push(started.elapsed().as_nanos() as f64);
                } else {
                    let started = Instant::now();
                    let factorization = splu(&matrix, balanced_arm_options()).expect("fsci splu");
                    let elapsed = started.elapsed().as_nanos() as f64;
                    black_box(&factorization);
                    b_slots.push(elapsed);
                }
            }
            scipy_ns.push(median(&a_slots));
            fsci_ns.push(median(&b_slots));
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
            !(spectral_enabled && fixture == Fixture::Cubic) || spectral_hits > 0,
            "the structured-fastpath arm never hit the spectral path on the cubic \
             fixture — the row would claim an algorithm that did not run"
        );

        // THE SAME PROOF, FOR THE HEAD-CACHE ARM, and both directions of it. The
        // enabled arm must show factorizations that took the head path; the
        // disabled arm must show NONE. Only one of those checks is a must-hit, and
        // a must-hit alone cannot tell a live control from a counter that is simply
        // always incremented. This process is single-threaded and owns the counter,
        // so the exact delta is meaningful here in a way it would not be under a
        // concurrent test binary.
        let head_cache_hits =
            SPLU_ROW_HEAD_CACHE_FACTOR_HITS.load(Ordering::Relaxed) - head_cache_hits_before;
        println!(
            "execution_proof: row_head_cache_enabled={head_cache_enabled} \
             row_head_cache_factor_hits={head_cache_hits} \
             row_head_cache_toggle_reads={}",
            SPLU_ROW_HEAD_CACHE_DISABLE.load_count(),
        );
        // THE GUARD IS SCOPED, NOT WEAKENED. Its premise is that the general elimination ran; it
        // exists to catch an A/B whose two arms are the same code. The banded path bypasses that
        // elimination entirely -- which is the whole point of it -- so the row-head-cache toggle
        // is legitimately unread there, and `banded_factor_hits` is the positive evidence that a
        // DIFFERENT path ran rather than the same one twice. Where the banded path did not take
        // the factorization the original assertion applies unchanged.
        let banded_hits = SPLU_BANDED_FACTOR_HITS.load(Ordering::Relaxed) - banded_hits_before;
        assert!(
            SPLU_ROW_HEAD_CACHE_DISABLE.load_count() > 0 || banded_hits > 0,
            "the elimination never read SPLU_ROW_HEAD_CACHE_DISABLE and the banded path never \
             ran, so both arms of this A/B are the same code and the ratio is not reportable"
        );
        // Only an EXPLICIT request may not decline. Once the library default became ON,
        // `banded_requested` is true for every run and a decline is the correct, expected outcome
        // on a cell the path is not for -- scattered declines because its factor barely fills.
        // Asserting on the default turned a working recogniser into a crash.
        assert!(
            banded_override != Some(true) || banded_hits > 0,
            "FSCI_SPLU_BANDED=1 was set and the banded arm never accepted a factorization, so \
             this row would measure the general path under a banded label"
        );
        assert!(
            banded_hits > 0 || head_cache_enabled == (head_cache_hits > 0),
            "the head-cache arm did not take effect: enabled={head_cache_enabled} but \
             {head_cache_hits} factorizations took the head path"
        );

        let back_merge_hits =
            SPLU_BACK_MERGE_FACTOR_HITS.load(Ordering::Relaxed) - back_merge_hits_before;
        println!(
            "execution_proof: back_merge_enabled={back_merge_enabled} \
             back_merge_factor_hits={back_merge_hits} \
             back_merge_toggle_reads={}",
            SPLU_BACK_MERGE_ENABLE.load_count(),
        );
        assert!(
            SPLU_BACK_MERGE_ENABLE.load_count() > 0 || banded_hits > 0,
            "the elimination never read SPLU_BACK_MERGE_ENABLE, so both arms of this \
             A/B are the same code and the ratio is not reportable"
        );
        assert!(
            banded_hits > 0 || back_merge_enabled == (back_merge_hits > 0),
            "the back-merge arm did not take effect: enabled={back_merge_enabled} but \
             {back_merge_hits} factorizations took it"
        );

        let partial_inplace_hits =
            SPLU_PARTIAL_INPLACE_FACTOR_HITS.load(Ordering::Relaxed) - partial_inplace_hits_before;
        println!(
            "execution_proof: partial_inplace_enabled={partial_inplace_enabled} \
             partial_inplace_factor_hits={partial_inplace_hits} \
             partial_inplace_toggle_reads={}",
            SPLU_PARTIAL_INPLACE_ENABLE.load_count(),
        );
        assert!(
            SPLU_PARTIAL_INPLACE_ENABLE.load_count() > 0 || banded_hits > 0,
            "the elimination never read SPLU_PARTIAL_INPLACE_ENABLE, so both arms of \
             this A/B are the same code and the ratio is not reportable"
        );
        assert!(
            banded_hits > 0 || partial_inplace_enabled == (partial_inplace_hits > 0),
            "the partial-inplace arm did not take effect: \
             enabled={partial_inplace_enabled} but {partial_inplace_hits} took it"
        );

        // DECLINING IS LEGITIMATE for this arm -- it refuses matrices with no exploitable
        // width or any row interchange -- so this REPORTS hits rather than asserting them.
        // A row where the arm was enabled and hits are zero measured the SEQUENTIAL path
        // and must be read as such, not as a supernodal result.
        let supernodal_hits =
            SPLU_SUPERNODAL_FACTOR_HITS.load(Ordering::Relaxed) - supernodal_hits_before;
        {
            let stage: Vec<u64> = SPLU_BANDED_STAGE_NANOS
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .collect();
            let total = stage.iter().sum::<u64>();
            println!(
                "execution_proof: banded_eliminate_ms={:.3} banded_unpack_ms={:.3} \
                 unpack_share={:.4} unpack_reserve={} unpack_reserve_hits={} instrumented={}",
                stage[0] as f64 / 1.0e6,
                stage[1] as f64 / 1.0e6,
                stage[1] as f64 / total.max(1) as f64,
                SPLU_BANDED_UNPACK_RESERVE.load(Ordering::Relaxed),
                SPLU_BANDED_UNPACK_RESERVE_HITS.load(Ordering::Relaxed)
                    - unpack_reserve_hits_before,
                total > 0,
            );
        }
        println!(
            "execution_proof: banded_requested={banded_requested} banded_factor_hits={}",
            SPLU_BANDED_FACTOR_HITS.load(Ordering::Relaxed) - banded_hits_before
        );
        // STAGE SHARES for frankenscipy-6940p. The vs-SciPy deficit shrinks monotonically
        // with n, and a gap that CLOSES as n grows points at amortising overhead rather
        // than a per-element kernel deficit. Shares — not totals — decide which: ordering,
        // setup and assemble are driven by n and nnz; eliminate is driven by FILL. If the
        // first three lose share as n grows while eliminate gains it, the small-n penalty
        // is one-time cost and the lever family is different from every kernel lever
        // priced so far.
        {
            let stage: Vec<u64> = fsci_sparse::linalg::SPLU_STAGE_NANOS
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .collect();
            let total = stage.iter().sum::<u64>().max(1);
            let pct = |i: usize| 100.0 * stage[i] as f64 / total as f64;
            println!(
                "splu_stages: ordering_ms={:.3} ({:.2}%) setup_ms={:.3} ({:.2}%) \
                 eliminate_ms={:.3} ({:.2}%) assemble_ms={:.3} ({:.2}%) one_time_share={:.4}",
                stage[0] as f64 / 1.0e6,
                pct(0),
                stage[1] as f64 / 1.0e6,
                pct(1),
                stage[2] as f64 / 1.0e6,
                pct(2),
                stage[3] as f64 / 1.0e6,
                pct(3),
                (stage[0] + stage[1] + stage[3]) as f64 / total as f64,
            );
        }
        // "enabled" is not "took effect". The swap writeback only fires on the FULL merge
        // path, so on a fixture the banded kernel accepts, or one the one-column arm covers,
        // this reads zero and an A/B between the two settings is a null over identical code.
        println!(
            "execution_proof: contiguous_solve_enabled={} contiguous_solve_hits={}",
            SPLU_CONTIGUOUS_SOLVE_ENABLE.load(Ordering::Relaxed),
            SPLU_CONTIGUOUS_SOLVE_HITS.load(Ordering::Relaxed) - contiguous_solve_hits_before
        );
        println!(
            "execution_proof: swap_writeback_enabled={} swap_writeback_hits={}",
            SPLU_SWAP_WRITEBACK_ENABLE.load(Ordering::Relaxed),
            SPLU_SWAP_WRITEBACK_HITS.load(Ordering::Relaxed) - swap_writeback_hits_before
        );
        println!(
            "execution_proof: supernodal_enabled={supernodal_enabled} \
             supernodal_factor_hits={supernodal_hits} \
             supernodal_toggle_reads={}",
            SPLU_SUPERNODAL_ENABLE.load_count(),
        );
        if fsci_sparse::linalg::SPLU_SOLVE_STAGE_TIMING.load(Ordering::Relaxed) {
            let stage: Vec<u64> = fsci_sparse::linalg::SPLU_SOLVE_STAGE_NANOS
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .collect();
            let parts: u64 = stage.iter().sum();
            let total = fsci_sparse::linalg::SPLU_SOLVE_STAGE_TOTAL_NANOS.load(Ordering::Relaxed);
            // CLOSURE FIRST. Parts that do not sum to the whole are three unrelated numbers,
            // and quoting a share off them would be worse than quoting nothing.
            println!(
                "solve_stages: closure={:.4} (sum {:.3} ms vs whole {:.3} ms) \
                 forward_ms={:.3} ({:.2}%) backward_ms={:.3} ({:.2}%) \
                 unpermute_ms={:.3} ({:.2}%)",
                parts as f64 / total.max(1) as f64,
                parts as f64 / 1.0e6,
                total as f64 / 1.0e6,
                stage[0] as f64 / 1.0e6,
                100.0 * stage[0] as f64 / parts.max(1) as f64,
                stage[1] as f64 / 1.0e6,
                100.0 * stage[1] as f64 / parts.max(1) as f64,
                stage[2] as f64 / 1.0e6,
                100.0 * stage[2] as f64 / parts.max(1) as f64,
            );
        }
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
        // ABSOLUTE TIMES BESIDE THE RATIO. A ratio hides the scale it was taken
        // at: 0.2x on a 40 ms factorization and 0.2x on a 40 us one are the same
        // number and completely different problems, and only one of them is worth
        // a supernodal rewrite. `ns_per_unit` normalises by retained fill so cells
        // of different sizes can be compared at all.
        let scipy_median_ns = median(&scipy_ns);
        let fsci_median_ns = median(&fsci_ns);
        let (scipy_ns_low, scipy_ns_high) = bootstrap_median_ci(&scipy_ns);
        let (fsci_ns_low, fsci_ns_high) = bootstrap_median_ci(&fsci_ns);
        println!(
            "absolute: scipy_median_ns={scipy_median_ns:.0} \
             ci95=[{scipy_ns_low:.0},{scipy_ns_high:.0}] \
             fsci_median_ns={fsci_median_ns:.0} \
             ci95=[{fsci_ns_low:.0},{fsci_ns_high:.0}] \
             scipy_median_us={:.3} fsci_median_us={:.3}",
            scipy_median_ns / 1_000.0,
            fsci_median_ns / 1_000.0
        );
        println!(
            "per_unit: scipy_ns_per_lu_nonzero={:.4} fsci_ns_per_lu_nonzero={:.4} \
             scipy_ns_per_row={:.2} fsci_ns_per_row={:.2} \
             lu_nnz={scipy_lu_nnz} n={}",
            ns_per_unit(scipy_median_ns, scipy_lu_nnz),
            ns_per_unit(fsci_median_ns, scipy_lu_nnz),
            ns_per_unit(scipy_median_ns, matrix.shape().rows),
            ns_per_unit(fsci_median_ns, matrix.shape().rows),
            matrix.shape().rows
        );
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio:.4}x  ci95=[{low:.4},{high:.4}]  \
             rounds={rounds}  verdict={verdict}"
        );
    }

    /// Nanoseconds per unit of retained work, or `NaN` when there is no work to
    /// divide by — reporting `0` there would read as "infinitely fast" rather
    /// than "undefined", which is the wrong way for a provenance field to fail.
    fn ns_per_unit(median_ns: f64, units: usize) -> f64 {
        if units == 0 {
            f64::NAN
        } else {
            median_ns / units as f64
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{
            Fixture, Ordering, RunConfig, SCIPY_REQUIRED_MODULES, SPLU_BACK_MERGE_ENABLE,
            SPLU_PARTIAL_INPLACE_ENABLE, SPLU_ROW_HEAD_CACHE_DISABLE, SPLU_SUPERNODAL_ENABLE,
            balanced_square_quiescence, is_help_request, ns_per_unit, parse_run_config,
            replicate_summary, run_aggregate,
        };
        use fsci_runtime::scipy_incumbent::{
            PYTHON_CANDIDATES, SITE_PACKAGES_CANDIDATES, interpreter_can_import_scipy,
        };

        fn args(values: &[&str]) -> Vec<String> {
            values.iter().map(ToString::to_string).collect()
        }

        #[test]
        fn replicate_summary_widens_with_between_replicate_spread() {
            // THE WHOLE POINT of aggregating at replicate level is that the interval must
            // respond to how much the replicates disagree. A summary that reported a
            // narrow interval regardless would reproduce the defect it replaces -- an
            // over-confident number that survives contradicting data.

            // TIGHT: replicates that agree. The interval must be correspondingly narrow.
            let tight = [0.5300, 0.5310, 0.5305, 0.5295, 0.5302];
            let (n, med, lo, high, min, max) = replicate_summary(&tight);
            assert_eq!(n, 5);
            assert!(
                (med - 0.5302).abs() < 1e-9,
                "median of the tight set, got {med}"
            );
            assert!(
                high - lo < 0.005,
                "agreeing replicates must give a narrow interval, got [{lo},{high}]"
            );
            assert!((min - 0.5295).abs() < 1e-9 && (max - 0.5310).abs() < 1e-9);

            // SPREAD: the four ratios actually measured on one ELF within ten minutes.
            // The interval must be visibly wider than the tight case, or the summary is
            // not measuring reproducibility at all.
            let observed = [0.5230, 0.5380, 0.5775, 0.4950];
            let (n, med, lo, high, min, max) = replicate_summary(&observed);
            assert_eq!(n, 4);
            assert!(
                (0.49..=0.58).contains(&med),
                "median must sit inside the observed range, got {med}"
            );
            assert!(
                high - lo > 0.02,
                "replicates spanning 1.167x must give a wide interval, got [{lo},{high}] \
                 -- a narrow one here means the spread is being hidden"
            );
            assert!((min - 0.4950).abs() < 1e-9 && (max - 0.5775).abs() < 1e-9);
        }

        #[test]
        fn replicate_summary_is_deterministic() {
            // Two calls on the same data must report the same interval, or a row banked
            // from it could not be reproduced from the ledger.
            let values = [0.5230, 0.5380, 0.5775, 0.4950, 0.5100];
            assert_eq!(replicate_summary(&values), replicate_summary(&values));
        }

        /// The interpreter probe answers BOTH ways -- the two-arm control.
        ///
        /// The probe itself now lives in `fsci_runtime::scipy_incumbent`, because this
        /// harness was the only one in the workspace that had it while 26 neighbours still
        /// selected an interpreter by `Path::exists`. That predicate cannot fail for the
        /// reason we care about: `python3` exists on every host here and imports SciPy on
        /// almost none of them. This test stays HERE as well as there because it is this
        /// harness's own contract that its incumbent arm is real, and because a shared
        /// helper is exactly the kind of thing that gets swapped out from under a caller.
        #[test]
        fn scipy_interpreter_probe_answers_both_ways() {
            // MUST MISS: an interpreter that is not on the box at all. This arm needs no
            // SciPy anywhere and so runs identically on a bare CI worker.
            assert!(
                !interpreter_can_import_scipy(
                    "/nonexistent/bin/python-that-is-not-installed",
                    None,
                    SCIPY_REQUIRED_MODULES
                ),
                "probe claimed a nonexistent interpreter can import scipy"
            );
            // MUST MISS for the same reason even when handed a plausible PYTHONPATH: it is
            // the IMPORT that is being proven, never the existence of a directory.
            assert!(
                !interpreter_can_import_scipy(
                    "/nonexistent/bin/python-that-is-not-installed",
                    Some("/tmp"),
                    SCIPY_REQUIRED_MODULES
                ),
                "probe was satisfied by a path rather than by an import"
            );
            // MUST HIT, but only where a live incumbent is actually installed. Asserting
            // unconditionally would make this test a host check rather than a probe check,
            // and it would fail on workers that legitimately carry no SciPy.
            let found = PYTHON_CANDIDATES
                .iter()
                .flat_map(|python| {
                    std::iter::once((*python, None)).chain(
                        SITE_PACKAGES_CANDIDATES
                            .iter()
                            .map(move |path| (*python, Some(*path))),
                    )
                })
                .find(|(python, site)| {
                    interpreter_can_import_scipy(python, *site, SCIPY_REQUIRED_MODULES)
                });
            if let Some((python, site)) = found {
                assert!(
                    interpreter_can_import_scipy(python, site, SCIPY_REQUIRED_MODULES),
                    "probe is not repeatable on {python}"
                );
                println!("must-hit arm observed on {python} site={site:?}");
            } else {
                println!(
                    "must-hit arm SKIPPED: no candidate on this host imports scipy, so only \
                     the must-miss arms ran here"
                );
            }
        }

        #[test]
        fn aggregate_refuses_inputs_that_cannot_support_a_summary() {
            // Two replicates cannot separate a spread from an outlier, and printing a
            // confident-looking interval over them would recreate the over-confidence
            // this mode exists to remove.
            assert!(
                run_aggregate(&args(&["0.52", "0.55"])).is_err(),
                "two replicates"
            );
            assert!(run_aggregate(&args(&[])).is_err(), "no replicates");
            // Non-numeric and non-physical inputs are refused rather than silently
            // coerced, since a ratio of zero or NaN would poison the median.
            assert!(
                run_aggregate(&args(&["0.52", "banana", "0.55"])).is_err(),
                "non-numeric"
            );
            assert!(
                run_aggregate(&args(&["0.52", "0", "0.55"])).is_err(),
                "zero ratio"
            );
            assert!(
                run_aggregate(&args(&["0.52", "-0.1", "0.55"])).is_err(),
                "negative ratio"
            );
            assert!(
                run_aggregate(&args(&["0.52", "NaN", "0.55"])).is_err(),
                "NaN ratio"
            );
            // And the must-hit arm: three valid replicates are accepted.
            assert!(run_aggregate(&args(&["0.5230", "0.5380", "0.5775"])).is_ok());
        }

        #[test]
        fn parse_run_config_defaults_to_cubic_general_lu() {
            assert_eq!(
                parse_run_config(&args(&["perf_splu"])),
                Ok(RunConfig {
                    side: 24,
                    rounds: 41,
                    warmup: 4,
                    spectral_enabled: false,
                    fixture: Fixture::Cubic,
                    // The shipping layout, so a bare invocation measures what it
                    // measured before this argument existed.
                    head_cache_enabled: true,
                    back_merge_enabled: false,
                    partial_inplace_enabled: true,
                    supernodal_enabled: false,
                })
            );
        }

        #[test]
        fn parse_run_config_preserves_explicit_scattered_arm() {
            assert_eq!(
                parse_run_config(&args(&["perf_splu", "24", "9", "1", "off", "scattered"])),
                Ok(RunConfig {
                    side: 24,
                    rounds: 9,
                    warmup: 1,
                    spectral_enabled: false,
                    fixture: Fixture::Scattered,
                    head_cache_enabled: true,
                    back_merge_enabled: false,
                    partial_inplace_enabled: true,
                    supernodal_enabled: false,
                })
            );
        }

        #[test]
        fn parse_run_config_selects_both_row_head_cache_arms() {
            // TWO ARMS, BOTH OBSERVED. `off` must actually reach the config — an
            // argument that parses but is dropped on the floor produces a row
            // labelled as the legacy layout while measuring the shipping one, which
            // is worse than not having the argument at all.
            let off = parse_run_config(&args(&[
                "perf_splu",
                "16",
                "11",
                "3",
                "off",
                "cubic",
                "off",
            ]))
            .expect("the legacy-layout arm must parse");
            assert!(
                !off.head_cache_enabled,
                "`off` must select the 56-byte header lookups"
            );

            let on = parse_run_config(&args(&["perf_splu", "16", "11", "3", "off", "cubic", "on"]))
                .expect("the head-cache arm must parse");
            assert!(
                on.head_cache_enabled,
                "`on` must select the head projection"
            );
            assert_eq!(
                on,
                parse_run_config(&args(&["perf_splu", "16", "11", "3", "off", "cubic"]))
                    .expect("the six-argument form must still parse"),
                "omitting the argument must be identical to passing `on`, or every \
                 previously recorded invocation of this harness changes meaning"
            );

            let error = parse_run_config(&args(&[
                "perf_splu",
                "16",
                "11",
                "3",
                "off",
                "cubic",
                "maybe",
            ]))
            .expect_err("an unrecognised arm must be refused, not defaulted");
            assert!(
                error.contains("row-head-cache arm"),
                "the refusal must name the argument it refused, got {error:?}"
            );
        }

        #[test]
        fn parse_run_config_rejects_malformed_positional_arguments() {
            let error = parse_run_config(&args(&["perf_splu", ".", "splu"]))
                .expect_err("a malformed command must not select default rows");
            assert!(error.contains("side must be an integer"));
        }

        /// The arm selection a `RunConfig` actually carries.
        ///
        /// Split out so the SAME projection is applied to the parsed config and to the
        /// deliberately drifted one below; comparing two differently-built tuples would
        /// let a typo in one of them pass as agreement.
        fn arms_of(config: &RunConfig) -> (bool, bool, bool, bool) {
            (
                config.head_cache_enabled,
                config.back_merge_enabled,
                config.partial_inplace_enabled,
                config.supernodal_enabled,
            )
        }

        /// The arms the LIBRARY ships, read from the toggles themselves.
        ///
        /// Read, never restated. A test that repeated the literals `parse_run_config`
        /// uses would agree with the harness by construction and would have stayed green
        /// through exactly the drift this crate has already paid for once.
        ///
        /// PRECONDITION, stated because it is the thing that could rot: no test in this
        /// binary calls `store` on any of these four statics, so what is read here is the
        /// construction default and not another test's leftover. `cargo test` runs a
        /// crate's tests concurrently in one process (frankenscipy-0zn0v), so if a
        /// toggle-writing test is ever added to THIS bin, it must take a shared mutex with
        /// this one or this assertion becomes a race. The library's own toggle tests live
        /// in a different test binary and cannot reach these.
        fn library_shipping_arms() -> (bool, bool, bool, bool) {
            (
                // Inverted: the toggle names the DISABLE side.
                !SPLU_ROW_HEAD_CACHE_DISABLE.load(Ordering::Relaxed),
                SPLU_BACK_MERGE_ENABLE.load(Ordering::Relaxed),
                SPLU_PARTIAL_INPLACE_ENABLE.load(Ordering::Relaxed),
                SPLU_SUPERNODAL_ENABLE.load(Ordering::Relaxed),
            )
        }

        #[test]
        fn bare_invocation_defaults_track_the_library_toggles() {
            // WHAT THIS PINS. `parse_run_config` hardcodes a token per arm and the run
            // site stores all four UNCONDITIONALLY, so a bare `perf_splu` measures the
            // harness's literals, not the library's defaults. Those agree today. The last
            // time they stopped agreeing, the harness printed a provenance line calling
            // itself the shipping configuration while measuring the other arm, and only
            // `partial_inplace_factor_hits=0` gave it away — after the row was taken.
            let bare =
                parse_run_config(&args(&["perf_splu"])).expect("a bare invocation must parse");
            let library = library_shipping_arms();

            // MUST-HIT.
            assert_eq!(
                arms_of(&bare),
                library,
                "a bare `perf_splu` no longer measures the shipping arms: harness \
                 (head_cache, back_merge, partial_inplace, supernodal) = {:?} against \
                 library {library:?}. Update the defaults in `parse_run_config` to match \
                 the library toggles -- do NOT relax this test. Every row previously taken \
                 with a bare invocation described the arm this test is now refusing.",
                arms_of(&bare)
            );

            // MUST-MISS, one arm at a time. Without this the assertion above would pass
            // just as happily if both sides were built from the same source -- "we agree"
            // proves nothing until disagreement has been shown to be detectable.
            for (index, label) in [
                (0usize, "head_cache"),
                (1, "back_merge"),
                (2, "partial_inplace"),
                (3, "supernodal"),
            ] {
                let mut drifted = bare;
                match index {
                    0 => drifted.head_cache_enabled = !drifted.head_cache_enabled,
                    1 => drifted.back_merge_enabled = !drifted.back_merge_enabled,
                    2 => drifted.partial_inplace_enabled = !drifted.partial_inplace_enabled,
                    _ => drifted.supernodal_enabled = !drifted.supernodal_enabled,
                }
                assert_ne!(
                    arms_of(&drifted),
                    library,
                    "flipping the {label} arm was not detected, so this comparison cannot \
                     see drift in it either"
                );
            }
        }

        #[test]
        fn parse_run_config_refuses_every_malformed_selector_position() {
            // The bead this closes (frankenscipy-tcg0u) is about commands that parse as
            // DEFAULTS instead of failing. One position was covered; a selector is only
            // fail-closed if EVERY position is, so each is asserted here, and each error
            // must NAME the argument it refused -- a generic refusal sends the operator
            // back to the same guess that produced the malformed command.
            for (argv, expected) in [
                (vec!["perf_splu", "24", "x"], "rounds must be an integer"),
                (
                    vec!["perf_splu", "24", "41", "x"],
                    "warmup must be an integer",
                ),
                (
                    vec!["perf_splu", "24", "41", "4", "maybe"],
                    "spectral arm must be `on` or `off`",
                ),
                (
                    vec!["perf_splu", "24", "41", "4", "off", "banded"],
                    "fixture must be `cubic`, `scattered` or `convection`",
                ),
                (
                    vec!["perf_splu", "24", "41", "4", "off", "cubic", "on", "yes"],
                    "back-merge arm must be `on` or `off`",
                ),
                (
                    vec![
                        "perf_splu",
                        "24",
                        "41",
                        "4",
                        "off",
                        "cubic",
                        "on",
                        "off",
                        "yes",
                    ],
                    "partial-inplace arm must be `on` or `off`",
                ),
                (
                    vec![
                        "perf_splu",
                        "24",
                        "41",
                        "4",
                        "off",
                        "cubic",
                        "on",
                        "off",
                        "on",
                        "yes",
                    ],
                    "supernodal arm must be `on` or `off`",
                ),
            ] {
                let error = match parse_run_config(&args(&argv)) {
                    Ok(config) => panic!("{argv:?} must be refused, but parsed as {config:?}"),
                    Err(error) => error,
                };
                assert!(
                    error.contains(expected),
                    "{argv:?} was refused without naming the argument: wanted {expected:?}, \
                     got {error:?}"
                );
            }

            // EXTRA selectors, the other half of "malformed": an operator who appends one
            // argument too many has mistaken the positional order, and every arm after the
            // mistake is then wrong. Refuse the whole command rather than the tail.
            let error = parse_run_config(&args(&[
                "perf_splu",
                "24",
                "41",
                "4",
                "off",
                "cubic",
                "on",
                "off",
                "on",
                "off",
                "off",
            ]))
            .expect_err("a tenth selector must be refused, not ignored");
            assert!(
                error.contains("expected at most nine arguments"),
                "the arity refusal must say what the limit is, got {error:?}"
            );

            // MUST-HIT for this test too: the longest LEGAL command still parses, so the
            // arity guard is refusing the tenth selector and not the ninth.
            let full = parse_run_config(&args(&[
                "perf_splu",
                "24",
                "41",
                "4",
                "off",
                "cubic",
                "on",
                "off",
                "on",
                "off",
            ]))
            .expect("the nine-selector form must still parse");
            assert_eq!(
                arms_of(&full),
                (true, false, true, false),
                "the fully explicit form must reach the config it names"
            );

            // The two numeric floors are refusals, not clamps -- a silently clamped
            // `side` would measure a different matrix than the command names.
            assert!(
                parse_run_config(&args(&["perf_splu", "3"]))
                    .expect_err("side 3 must be refused")
                    .contains("side must be at least 4")
            );
            assert!(
                parse_run_config(&args(&["perf_splu", "24", "8"]))
                    .expect_err("8 rounds must be refused")
                    .contains("fewer than 9 rounds")
            );
        }

        #[test]
        fn balanced_square_nulls_admit_shared_host_drift() {
            assert_eq!(balanced_square_quiescence(0.9958, 0.9976), "clear");
        }

        #[test]
        fn balanced_square_nulls_reject_order_bias() {
            assert_eq!(balanced_square_quiescence(1.021, 1.0), "null-failed");
        }

        #[test]
        fn per_unit_normalisation_is_undefined_rather_than_zero_on_no_work() {
            // Two arms: a case that MUST produce a finite rate and one that MUST
            // NOT, because a divide-by-zero that quietly returns 0.0 would print
            // the fastest number in the table for a cell that measured nothing.
            let rate = ns_per_unit(2_500.0, 1_000);
            assert!(
                (rate - 2.5).abs() < 1e-12,
                "1000 units of 2500 ns is 2.5 ns each"
            );
            assert!(
                ns_per_unit(2_500.0, 0).is_nan(),
                "no units must read as undefined, never as zero"
            );
        }

        #[test]
        fn help_is_recognised_before_anything_expensive_runs() {
            // Two arms, per frankenscipy-yq1k8: a case that MUST be treated as
            // help and a case that MUST NOT. A predicate tested on one arm can be
            // blind or blanket-matching and still print a clean result.
            for help in [
                vec!["perf_splu", "--help"],
                vec!["perf_splu", "-h"],
                vec!["perf_splu", "help"],
                vec!["perf_splu", "10", "9", "2", "off", "scattered", "--help"],
            ] {
                assert!(
                    is_help_request(&args(&help)),
                    "{help:?} must short-circuit before the ELF is hashed"
                );
            }

            for measurement in [
                vec!["perf_splu"],
                vec!["perf_splu", "16"],
                vec!["perf_splu", "10", "9", "2", "off", "scattered"],
            ] {
                assert!(
                    !is_help_request(&args(&measurement)),
                    "{measurement:?} is a real measurement configuration"
                );
                parse_run_config(&args(&measurement))
                    .expect("a real configuration must still parse after the help arm");
            }

            // The program name itself is never a help request, or invoking the
            // binary as `help` would silently print usage instead of measuring.
            assert!(!is_help_request(&args(&["help"])));
        }
    }
}

fn main() {
    bench::run();
}
