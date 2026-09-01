//! Whole-job solver reports versus screened SciPy 1.17.1 public deployments.
//!
//! The `newton`, `secant`, and `fixed_point` modes retire their historical
//! scalar-loop comparisons by screening the public array-valued incumbent and
//! persistent scalar/vector pool deployments.

#[cfg(feature = "opt-incumbent-bench")]
mod bench {
    use fsci_opt::{RootOptions, fixed_point_many, newton_many, secant_many};
    use fsci_runtime::scipy_incumbent::{PINNED_NUMPY, PINNED_SCIPY, ScipyIncumbent};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    const BATCH: usize = 65_536;
    const FIXTURE_COLUMNS: usize = 6;
    const PAYLOAD_COLUMNS: usize = 9;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 3;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const MAX_PRICE_ERROR: f64 = 5.0e-8;
    const MAX_VOLATILITY_ERROR: f64 = 5.0e-7;
    const MAX_CROSS_VOLATILITY_ERROR_ABS: f64 = 1.0e-8;
    const MAX_CROSS_VOLATILITY_ERROR_REL: f64 = 1.0e-8;
    const MAX_COLEBROOK_RESIDUAL: f64 = 1.0e-8;
    const MAX_CROSS_FRICTION_ERROR_ABS: f64 = 1.0e-9;
    const MAX_CROSS_FRICTION_ERROR_REL: f64 = 1.0e-8;
    const PRICE_INDEX_NOISE_FLOOR: f64 = 1.0e-11;
    const SECANT_PRICE_INDEX_NOISE_FLOOR: f64 = 1.0e-9;
    const COLEBROOK_INDEX_NOISE_FLOOR: f64 = 1.0e-10;
    const VOLATILITY_INDEX_NOISE_FLOOR: f64 = 1.0e-10;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SCIPY_ARMS: [&str; 6] = [
        "scalar_loop",
        "array_single",
        "scalar_thread",
        "scalar_process",
        "array_thread",
        "array_process",
    ];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SolverMode {
        Newton,
        Secant,
        FixedPoint,
    }

    impl SolverMode {
        const fn label(self) -> &'static str {
            match self {
                Self::Newton => "newton",
                Self::Secant => "secant",
                Self::FixedPoint => "fixed_point",
            }
        }

        const fn rust_api(self) -> &'static str {
            match self {
                Self::Newton => "FrankenSciPy newton_many",
                Self::Secant => "FrankenSciPy secant_many",
                Self::FixedPoint => "FrankenSciPy fixed_point_many",
            }
        }

        const fn collapse_boundary(self) -> f64 {
            match self {
                Self::Newton => 49.5,
                Self::Secant => 53.6,
                Self::FixedPoint => 192.0,
            }
        }

        const fn old_claim(self) -> &'static str {
            match self {
                Self::Newton => "495_to_986x",
                Self::Secant => "536x",
                Self::FixedPoint => "1920x",
            }
        }
    }

    static SOLVER_MODE: OnceLock<SolverMode> = OnceLock::new();

    fn solver_mode() -> SolverMode {
        match SOLVER_MODE.get() {
            Some(&solver) => solver,
            None => SolverMode::Newton,
        }
    }

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import math
import multiprocessing
import os
import sys
import threading
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import numpy._core._multiarray_umath as _multiarray_umath
import scipy
from scipy import optimize, special
from scipy.optimize import _minpack_py, _zeros_py
from scipy.special import _ufuncs

BATCH = 65536
COLUMNS = 9
TOL = 1.0e-10
EXPECTED_BYTES = BATCH * COLUMNS * 8
SOLVER_MODE = os.environ.get("FSCI_SOLVER_MODE")
if SOLVER_MODE not in ("newton", "secant", "fixed_point"):
    raise RuntimeError(f"invalid FSCI_SOLVER_MODE {SOLVER_MODE!r}")
MAXITER = 500 if SOLVER_MODE == "fixed_point" else 50

raw_fixture = sys.stdin.buffer.read(EXPECTED_BYTES)
if len(raw_fixture) != EXPECTED_BYTES:
    raise RuntimeError(
        f"fixture byte count mismatch: expected {EXPECTED_BYTES}, got {len(raw_fixture)}"
    )
FIXTURE_SHA256 = hashlib.sha256(raw_fixture).hexdigest()
DATA = np.frombuffer(raw_fixture, dtype="<f8").reshape(BATCH, COLUMNS).copy()
if SOLVER_MODE == "fixed_point":
    DIAMETER = DATA[:, 0]
    VELOCITY = DATA[:, 1]
    LENGTH = DATA[:, 2]
    RELATIVE_ROUGHNESS = DATA[:, 3]
    REYNOLDS = DATA[:, 4]
    DYNAMIC_PRESSURE = DATA[:, 5]
    LENGTH_OVER_DIAMETER = DATA[:, 6]
else:
    SPOT = DATA[:, 0]
    STRIKE = DATA[:, 1]
    MATURITY = DATA[:, 2]
    RATE = DATA[:, 3]
    TARGET = DATA[:, 4]
    TRUE_VOLATILITY = DATA[:, 5]
    SQRT_T = DATA[:, 6]
    LOG_FORWARD = DATA[:, 7]
    DISCOUNTED_STRIKE = DATA[:, 8]
CHECKSUM_WEIGHTS = (
    (np.arange(BATCH, dtype=np.float64) % 251.0) + 1.0
) / 251.0
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
CHUNK_COUNT = min(BATCH, WORKER_CAPACITY * 8)
CHUNK_RANGES = []
for chunk in range(CHUNK_COUNT):
    lo = chunk * BATCH // CHUNK_COUNT
    hi = (chunk + 1) * BATCH // CHUNK_COUNT
    if lo < hi:
        CHUNK_RANGES.append((lo, hi))

def vector_price(sigma, lo=0, hi=BATCH):
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt_t = sigma * SQRT_T[lo:hi]
    d1 = LOG_FORWARD[lo:hi] / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return (
        SPOT[lo:hi] * special.ndtr(d1)
        - DISCOUNTED_STRIKE[lo:hi] * special.ndtr(d2)
    )

def vector_vega(sigma, lo=0, hi=BATCH):
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_sqrt_t = sigma * SQRT_T[lo:hi]
    d1 = LOG_FORWARD[lo:hi] / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    density = np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return SPOT[lo:hi] * density * SQRT_T[lo:hi]

def scalar_price(sigma, index):
    sigma_sqrt_t = sigma * SQRT_T[index]
    d1 = LOG_FORWARD[index] / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return float(
        SPOT[index] * special.ndtr(d1)
        - DISCOUNTED_STRIKE[index] * special.ndtr(d2)
    )

def scalar_vega(sigma, index):
    sigma_sqrt_t = sigma * SQRT_T[index]
    d1 = LOG_FORWARD[index] / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return float(SPOT[index] * density * SQRT_T[index])

def vector_fixed_map(friction, lo=0, hi=BATCH):
    friction = np.asarray(friction, dtype=np.float64)
    inverse = -2.0 * np.log10(
        RELATIVE_ROUGHNESS[lo:hi] / 3.7
        + 2.51 / (REYNOLDS[lo:hi] * np.sqrt(friction))
    )
    return 1.0 / np.square(inverse)

def scalar_fixed_map(friction, index):
    inverse = -2.0 * math.log10(
        RELATIVE_ROUGHNESS[index] / 3.7
        + 2.51 / (REYNOLDS[index] * math.sqrt(friction))
    )
    return 1.0 / (inverse * inverse)

def solve_scalar_one(index):
    if SOLVER_MODE == "fixed_point":
        root = optimize.fixed_point(
            lambda friction: scalar_fixed_map(friction, index),
            0.02,
            xtol=TOL,
            maxiter=MAXITER,
            method="del2",
        )
        return (
            float(root),
            True,
            False,
            int(os.getpid()),
            int(threading.get_native_id()),
        )
    fprime = (
        (lambda sigma: scalar_vega(sigma, index))
        if SOLVER_MODE == "newton"
        else None
    )
    root, result = optimize.newton(
        lambda sigma: scalar_price(sigma, index) - TARGET[index],
        0.30,
        fprime=fprime,
        tol=TOL,
        maxiter=MAXITER,
        rtol=0.0,
        full_output=True,
        disp=False,
    )
    return (
        float(root),
        bool(result.converged),
        False,
        int(os.getpid()),
        int(threading.get_native_id()),
    )

def solve_array_range(bounds):
    lo, hi = bounds
    if SOLVER_MODE == "fixed_point":
        roots = optimize.fixed_point(
            lambda friction: vector_fixed_map(friction, lo, hi),
            np.full(hi - lo, 0.02, dtype=np.float64),
            xtol=TOL,
            maxiter=MAXITER,
            method="del2",
        )
        return (
            np.asarray(roots, dtype=np.float64),
            np.ones(hi - lo, dtype=bool),
            np.zeros(hi - lo, dtype=bool),
            int(os.getpid()),
            int(threading.get_native_id()),
        )
    fprime = (
        (lambda sigma: vector_vega(sigma, lo, hi))
        if SOLVER_MODE == "newton"
        else None
    )
    result = optimize.newton(
        lambda sigma: vector_price(sigma, lo, hi) - TARGET[lo:hi],
        np.full(hi - lo, 0.30, dtype=np.float64),
        fprime=fprime,
        tol=TOL,
        maxiter=MAXITER,
        rtol=0.0,
        full_output=True,
        disp=False,
    )
    return (
        np.asarray(result.root, dtype=np.float64),
        np.asarray(result.converged, dtype=bool),
        np.asarray(result.zero_der, dtype=bool),
        int(os.getpid()),
        int(threading.get_native_id()),
    )

def combine_scalar(results):
    roots = np.asarray([row[0] for row in results], dtype=np.float64)
    converged = np.asarray([row[1] for row in results], dtype=bool)
    zero_der = np.asarray([row[2] for row in results], dtype=bool)
    pids = {row[3] for row in results}
    tids = {row[4] for row in results}
    return roots, converged, zero_der, pids, tids

def combine_chunks(results):
    roots = np.concatenate([row[0] for row in results])
    converged = np.concatenate([row[1] for row in results])
    zero_der = np.concatenate([row[2] for row in results])
    pids = {row[3] for row in results}
    tids = {row[4] for row in results}
    return roots, converged, zero_der, pids, tids

def array_single():
    if SOLVER_MODE == "fixed_point":
        roots = optimize.fixed_point(
            vector_fixed_map,
            np.full(BATCH, 0.02, dtype=np.float64),
            xtol=TOL,
            maxiter=MAXITER,
            method="del2",
        )
        return (
            np.asarray(roots, dtype=np.float64),
            np.ones(BATCH, dtype=bool),
            np.zeros(BATCH, dtype=bool),
            {int(os.getpid())},
            {int(threading.get_native_id())},
        )
    fprime = vector_vega if SOLVER_MODE == "newton" else None
    result = optimize.newton(
        lambda sigma: vector_price(sigma) - TARGET,
        np.full(BATCH, 0.30, dtype=np.float64),
        fprime=fprime,
        tol=TOL,
        maxiter=MAXITER,
        rtol=0.0,
        full_output=True,
        disp=False,
    )
    return (
        np.asarray(result.root, dtype=np.float64),
        np.asarray(result.converged, dtype=bool),
        np.asarray(result.zero_der, dtype=bool),
        {int(os.getpid())},
        {int(threading.get_native_id())},
    )

# Pools are deliberately forked only after the immutable fixture and public
# callables exist. Construction and warmup are outside every timer.
PROCESS_POOL = multiprocessing.get_context("fork").Pool(WORKER_CAPACITY)
THREAD_POOL = ThreadPool(WORKER_CAPACITY)

def execute(arm):
    if arm == "scalar_loop":
        return combine_scalar([solve_scalar_one(index) for index in range(BATCH)])
    if arm == "array_single":
        return array_single()
    if arm == "scalar_thread":
        return combine_scalar(THREAD_POOL.map(solve_scalar_one, range(BATCH)))
    if arm == "scalar_process":
        chunksize = max(1, BATCH // (WORKER_CAPACITY * 16))
        return combine_scalar(
            PROCESS_POOL.map(solve_scalar_one, range(BATCH), chunksize)
        )
    if arm == "array_thread":
        return combine_chunks(THREAD_POOL.map(solve_array_range, CHUNK_RANGES))
    if arm == "array_process":
        return combine_chunks(PROCESS_POOL.map(solve_array_range, CHUNK_RANGES, 1))
    raise RuntimeError(f"unknown arm {arm!r}")

def summarize(raw):
    roots, converged, zero_der, pids, tids = raw
    roots = np.asarray(roots, dtype=np.float64)
    converged = np.asarray(converged, dtype=bool)
    zero_der = np.asarray(zero_der, dtype=bool)
    if roots.shape != (BATCH,):
        raise RuntimeError(f"invalid roots shape {roots.shape}")
    if converged.shape != (BATCH,) or zero_der.shape != (BATCH,):
        raise RuntimeError("invalid convergence shape")
    if SOLVER_MODE == "fixed_point":
        equation_residuals = np.abs(
            1.0 / np.sqrt(roots)
            + 2.0
            * np.log10(
                RELATIVE_ROUGHNESS / 3.7
                + 2.51 / (REYNOLDS * np.sqrt(roots))
            )
        )
        pressure_losses = roots * LENGTH_OVER_DIAMETER * DYNAMIC_PRESSURE
        primary = equation_residuals
        secondary = pressure_losses
        primary_quantiles = np.percentile(primary, [50, 95, 99])
        secondary_quantiles = np.percentile(secondary, [50, 95, 99])
        bands = np.histogram(
            roots,
            bins=np.asarray(
                [-np.inf, 0.0125, 0.020, 0.0275, 0.035, 0.045, 0.060, 0.080, np.inf],
                dtype=np.float64,
            ),
        )[0]
        severity = (
            int(np.count_nonzero(pressure_losses >= 1.0e5)),
            int(np.count_nonzero(pressure_losses >= 1.0e6)),
            int(np.count_nonzero(pressure_losses >= 1.0e7)),
        )
        secondary_mean = float(np.mean(secondary))
        checksum = float(
            np.dot(roots, CHECKSUM_WEIGHTS)
            + 1.0e-7 * np.dot(pressure_losses, CHECKSUM_WEIGHTS)
        )
    else:
        repriced = vector_price(roots)
        price_errors = np.abs(repriced - TARGET)
        volatility_errors = np.abs(roots - TRUE_VOLATILITY)
        primary = price_errors
        secondary = volatility_errors
        primary_quantiles = np.percentile(primary, [50, 95, 99])
        secondary_quantiles = np.percentile(secondary, [50, 95, 99])
        bands = np.histogram(
            roots,
            bins=np.asarray(
                [-np.inf, 0.15, 0.225, 0.30, 0.375, 0.45, 0.525, 0.60, np.inf],
                dtype=np.float64,
            ),
        )[0]
        severity = (0, 0, 0)
        secondary_mean = float(np.mean(secondary))
        checksum = float(np.dot(roots, CHECKSUM_WEIGHTS))
    summary = (
        int(np.count_nonzero(np.isfinite(roots))),
        int(np.count_nonzero(converged)),
        int(np.count_nonzero(zero_der)),
        int(np.argmax(primary)),
        float(primary_quantiles[0]),
        float(primary_quantiles[1]),
        float(primary_quantiles[2]),
        float(np.max(primary)),
        int(np.argmax(secondary)),
        float(secondary_quantiles[0]),
        float(secondary_quantiles[1]),
        float(secondary_quantiles[2]),
        float(np.max(secondary)),
        float(np.mean(roots)),
        float(np.min(roots)),
        float(np.max(roots)),
        *[int(value) for value in bands],
        secondary_mean,
        *severity,
        checksum,
    )
    return roots, summary, len(pids), len(tids)

def summary_payload(summary):
    return ",".join(
        str(value) if isinstance(value, int) else f"{value:.17e}"
        for value in summary
    )

def task_runtime():
    runtime = {}
    for name in os.listdir("/proc/self/task"):
        try:
            with open(
                f"/proc/self/task/{name}/schedstat", "r", encoding="ascii"
            ) as handle:
                runtime[int(name)] = int(handle.read().split()[0])
        except (FileNotFoundError, ProcessLookupError):
            pass
    return runtime

def run_observed(function):
    before = task_runtime()
    output = function()
    after = task_runtime()
    active = sum(after.get(tid, 0) > value for tid, value in before.items())
    active += sum(tid not in before and value > 0 for tid, value in after.items())
    return output, max(1, active), max(len(before), len(after))

def file_hash(path):
    with open(path, "rb") as handle:
        content = handle.read()
    return hashlib.sha256(content).hexdigest(), content

if SOLVER_MODE == "fixed_point":
    solver_path = inspect.getsourcefile(_minpack_py)
    numeric_path = inspect.getfile(_multiarray_umath)
    solver_module = optimize.fixed_point.__module__
    numeric_module = "numpy._core._multiarray_umath"
    engine_identity = (
        solver_module == "scipy.optimize._minpack_py"
        and np.log10 is _multiarray_umath.log10
    )
else:
    solver_path = inspect.getsourcefile(_zeros_py)
    numeric_path = inspect.getfile(_ufuncs)
    solver_module = optimize.newton.__module__
    numeric_module = "scipy.special._ufuncs"
    engine_identity = (
        solver_module == "scipy.optimize._zeros_py"
        and special.ndtr is _ufuncs.ndtr
    )
if solver_path is None or numeric_path is None:
    raise RuntimeError("cannot resolve live SciPy/numeric engine paths")
SOLVER_SHA256, solver_content = file_hash(solver_path)
NUMERIC_SHA256, numeric_content = file_hash(numeric_path)
engine_digest = hashlib.sha256()
for label, content in (("solver", solver_content), ("numeric", numeric_content)):
    engine_digest.update(label.encode("ascii"))
    engine_digest.update(b"\0")
    engine_digest.update(content)
SCIPY_ENGINE_SHA256 = engine_digest.hexdigest()

for arm in (
    "scalar_loop",
    "array_single",
    "scalar_thread",
    "scalar_process",
    "array_thread",
    "array_process",
):
    roots, summary, _pids, _tids = summarize(execute(arm))
    if not np.all(np.isfinite(roots)) or not math.isfinite(summary[-1]):
        raise RuntimeError(f"{arm} warmup failed")

fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and np.__version__ == "2.4.3"
    and engine_identity
    and not fsci_loaded
)
print(
    "READY"
    f" solver_mode={SOLVER_MODE}"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" solver_module={solver_module}"
    f" numeric_module={numeric_module}"
    f" scipy_engine_sha256={SCIPY_ENGINE_SHA256}"
    f" solver_engine_sha256={SOLVER_SHA256}"
    f" numeric_engine_sha256={NUMERIC_SHA256}"
    f" fixture_sha256={FIXTURE_SHA256}"
    f" worker_capacity={WORKER_CAPACITY}"
    f" chunk_count={len(CHUNK_RANGES)}"
    f" pool_start=fork"
    f" max_os_tasks={len(os.listdir('/proc/self/task'))}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={genuine}",
    flush=True,
)

for raw_line in sys.stdin.buffer:
    line = str(raw_line, "ascii")
    fields = line.strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "CHECK" and len(fields) == 2:
        arm = fields[1]
        raw, active, maximum = run_observed(lambda: execute(arm))
        _roots, summary, processes, threads = summarize(raw)
        print(
            f"CHECK {arm} {active} {maximum} {processes} {threads}"
            f" {summary_payload(summary)}",
            flush=True,
        )
    elif command == "ROOTS" and len(fields) == 2:
        arm = fields[1]
        roots = summarize(execute(arm))[0]
        bits = np.asarray(roots, dtype="<f8").view("<u8")
        print(
            f"ROOTS {arm} " + ",".join(f"{int(value):016x}" for value in bits),
            flush=True,
        )
    elif command == "TIME" and len(fields) == 3:
        arm = fields[1]
        repetitions = int(fields[2])
        if repetitions <= 0:
            raise RuntimeError("repetitions must be positive")
        before = task_runtime()
        checksum = 0.0
        maximum_processes = 0
        maximum_threads = 0
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            _roots, summary, processes, threads = summarize(execute(arm))
            checksum += summary[-1]
            maximum_processes = max(maximum_processes, processes)
            maximum_threads = max(maximum_threads, threads)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        after = task_runtime()
        active = sum(after.get(tid, 0) > value for tid, value in before.items())
        active += sum(tid not in before and value > 0 for tid, value in after.items())
        print(
            f"TIME {arm} {elapsed:.17e} {checksum:.17e}"
            f" {max(1, active)} {max(len(before), len(after))}"
            f" {maximum_processes} {maximum_threads}",
            flush=True,
        )
    elif command == "QUIT":
        THREAD_POOL.close()
        THREAD_POOL.join()
        PROCESS_POOL.close()
        PROCESS_POOL.join()
        print("BYE", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {line!r}")
"#;

    #[derive(Clone)]
    struct Dataset {
        fixture: Vec<[f64; FIXTURE_COLUMNS]>,
        params: Vec<Vec<f64>>,
        fixture_bytes: Vec<u8>,
        fixture_sha256: String,
    }

    #[derive(Clone, Copy, Debug)]
    struct JobSummary {
        finite: usize,
        converged: usize,
        zero_derivative: usize,
        worst_price_index: usize,
        price_p50: f64,
        price_p95: f64,
        price_p99: f64,
        max_price_error: f64,
        worst_volatility_index: usize,
        volatility_p50: f64,
        volatility_p95: f64,
        volatility_p99: f64,
        max_volatility_error: f64,
        mean_volatility: f64,
        min_volatility: f64,
        max_volatility: f64,
        bands: [usize; 8],
        secondary_mean: f64,
        pressure_counts: [usize; 3],
        checksum: f64,
    }

    #[derive(Clone, Copy)]
    struct Timing {
        elapsed: f64,
        active_tasks: usize,
        max_os_tasks: usize,
        worker_processes: usize,
        callsite_threads: usize,
    }

    #[derive(Clone)]
    struct ScipyCheck {
        arm: String,
        active_tasks: usize,
        max_os_tasks: usize,
        worker_processes: usize,
        callsite_threads: usize,
        summary: JobSummary,
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    /// Environment the live SciPy oracle spawns under.
    ///
    /// The resolver probes under exactly this, so a candidate interpreter cannot pass the
    /// probe under conditions the timed spawn will not get. The single-thread BLAS pinning
    /// lives in `fsci_runtime::scipy_incumbent::SINGLE_THREAD_ENV` and is applied by
    /// `ScipyIncumbent::command`, so it is deliberately not repeated here.
    const SPAWN_ENV: &[(&str, &str)] = &[];
    /// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
    /// installation whose compiled submodules do not load, and that difference would
    /// otherwise only surface mid-timing.
    const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.optimize"];

    /// The one live-SciPy incumbent this process compares against, resolved once and PROVEN
    /// by running the import rather than by a path existing.
    ///
    /// This harness used to name `/usr/bin/python3.13` and a pinned site-packages directory
    /// outright. Both are gone from this host, and neither absence was checked before the
    /// fixture reached the child's stdin, so a missing incumbent arrived as `BrokenPipe`
    /// several lines AFTER a well-formed provenance header had printed
    /// (frankenscipy-m5s54).
    fn incumbent() -> Result<&'static ScipyIncumbent, String> {
        static INCUMBENT: std::sync::OnceLock<Result<ScipyIncumbent, String>> =
            std::sync::OnceLock::new();
        INCUMBENT
            .get_or_init(|| {
                ScipyIncumbent::resolve_with(SPAWN_ENV, SCIPY_REQUIRED_MODULES)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    impl Scipy {
        fn start(data: &Dataset, solver: SolverMode) -> Result<(Self, String), String> {
            let incumbent = incumbent()?;
            println!("{}", incumbent.provenance_line());
            let python = incumbent.python.clone();
            let mut child = incumbent
                .command()
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("FSCI_SOLVER_MODE", solver.label())
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("spawn live SciPy oracle {python}: {error}"))?;
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy oracle has no stdin".to_string())?;
            stdin
                .write_all(&data.fixture_bytes)
                .and_then(|()| stdin.flush())
                .map_err(|error| format!("send exact fixture to SciPy: {error}"))?;
            let mut stdout = BufReader::new(
                child
                    .stdout
                    .take()
                    .ok_or_else(|| "live SciPy oracle has no stdout".to_string())?,
            );
            let mut ready = String::new();
            stdout
                .read_line(&mut ready)
                .map_err(|error| format!("read SciPy identity: {error}"))?;
            if ready.is_empty() {
                return Err("live SciPy oracle exited before identity".to_string());
            }
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                    stopped: false,
                },
                ready.trim().to_string(),
            ))
        }

        fn request(&mut self, command: &str, context: &str) -> Result<String, String> {
            writeln!(self.stdin, "{command}")
                .and_then(|()| self.stdin.flush())
                .map_err(|error| format!("send {context}: {error}"))?;
            let mut response = String::new();
            self.stdout
                .read_line(&mut response)
                .map_err(|error| format!("read {context}: {error}"))?;
            if response.is_empty() {
                return Err(format!("SciPy oracle exited during {context}"));
            }
            Ok(response.trim().to_string())
        }

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let response = self.request(&format!("CHECK {arm}"), "SciPy quality check")?;
            let fields = response.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("malformed SciPy check response: {response}"));
            }
            Ok(ScipyCheck {
                arm: arm.to_string(),
                active_tasks: parse(fields[2], "SciPy active task count")?,
                max_os_tasks: parse(fields[3], "SciPy OS task count")?,
                worker_processes: parse(fields[4], "SciPy worker process count")?,
                callsite_threads: parse(fields[5], "SciPy call-site thread count")?,
                summary: parse_summary(fields[6])?,
            })
        }

        fn roots(&mut self, arm: &str) -> Result<Vec<f64>, String> {
            let response = self.request(&format!("ROOTS {arm}"), "SciPy roots")?;
            let payload = response
                .strip_prefix(&format!("ROOTS {arm} "))
                .ok_or_else(|| format!("malformed SciPy roots response for {arm}"))?;
            let roots = payload
                .split(',')
                .map(|value| {
                    u64::from_str_radix(value, 16)
                        .map(f64::from_bits)
                        .map_err(|error| format!("parse SciPy root bits {value}: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if roots.len() != BATCH {
                return Err(format!(
                    "SciPy roots length mismatch: expected {BATCH}, got {}",
                    roots.len()
                ));
            }
            Ok(roots)
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let response =
                self.request(&format!("TIME {arm} {repetitions}"), "SciPy timing sample")?;
            let fields = response.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 8 || fields[0] != "TIME" || fields[1] != arm {
                return Err(format!("malformed SciPy timing response: {response}"));
            }
            let elapsed: f64 = parse(fields[2], "SciPy elapsed time")?;
            let checksum: f64 = parse(fields[3], "SciPy checksum")?;
            if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() {
                return Err(format!("invalid SciPy timing response: {response}"));
            }
            Ok(Timing {
                elapsed,
                active_tasks: parse(fields[4], "SciPy active task count")?,
                max_os_tasks: parse(fields[5], "SciPy OS task count")?,
                worker_processes: parse(fields[6], "SciPy worker process count")?,
                callsite_threads: parse(fields[7], "SciPy call-site thread count")?,
            })
        }

        fn stop(&mut self) -> Result<(), String> {
            if self.stopped {
                return Ok(());
            }
            let response = self.request("QUIT", "SciPy shutdown")?;
            if response != "BYE" {
                return Err(format!("malformed SciPy shutdown response: {response}"));
            }
            let status = self
                .child
                .wait()
                .map_err(|error| format!("wait for SciPy oracle: {error}"))?;
            if !status.success() {
                return Err(format!("SciPy oracle exited with {status}"));
            }
            self.stopped = true;
            Ok(())
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

    #[derive(Clone)]
    struct ScreenedArm {
        check: ScipyCheck,
        median: f64,
        max_active_tasks: usize,
        max_os_tasks: usize,
        max_worker_processes: usize,
        max_callsite_threads: usize,
    }

    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 11) as f64 / (1_u64 << 53) as f64
    }

    // Cephes' real erf/erfc rational, matching the implementation used by
    // scipy.special.ndtr and FrankenSciPy's audited scalar special-function
    // path without adding a circular fsci-opt -> fsci-special dependency.
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERF_T: [f64; 5] = [
        9.60497373987051638749E0,
        9.00260197203842689217E1,
        2.23200534594684319226E3,
        7.00332514112805075473E3,
        5.55923013010394962768E4,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERF_U: [f64; 5] = [
        3.35617141647503099647E1,
        5.21357949780152679795E2,
        4.59432382970980127987E3,
        2.26290000613890934246E4,
        4.92673942608635921086E4,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERFC_P: [f64; 9] = [
        2.46196981473530512524E-10,
        5.64189564831068821977E-1,
        7.46321056442269912687E0,
        4.86371970985681366614E1,
        1.96520832956077098242E2,
        5.26445194995477358631E2,
        9.34528527171957607540E2,
        1.02755188689515710272E3,
        5.57535335369399327526E2,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERFC_Q: [f64; 8] = [
        1.32281951154744992508E1,
        8.67072140885989742329E1,
        3.54937778887819891062E2,
        9.75708501743205489753E2,
        1.82390916687909736289E3,
        2.24633760818710981792E3,
        1.65666309194161350182E3,
        5.57535340817727675546E2,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERFC_R: [f64; 6] = [
        5.64189583547755073984E-1,
        1.27536670759978104416E0,
        5.01905042251180477414E0,
        6.16021097993053585195E0,
        7.40974269950448939160E0,
        2.97886665372100240670E0,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_ERFC_S: [f64; 6] = [
        2.26052863220117276590E0,
        9.39603524938001434673E0,
        1.20489539808096656605E1,
        1.70814450747565897222E1,
        9.60896809063285878198E0,
        3.36907645100081516050E0,
    ];
    #[allow(clippy::excessive_precision)]
    const CEPHES_MAXLOG: f64 = 7.08396418532264106224E2;

    fn cephes_polevl(x: f64, coefficients: &[f64]) -> f64 {
        coefficients
            .iter()
            .fold(0.0, |accumulator, &value| accumulator * x + value)
    }

    fn cephes_p1evl(x: f64, coefficients: &[f64]) -> f64 {
        coefficients
            .iter()
            .fold(1.0, |accumulator, &value| accumulator * x + value)
    }

    fn erf_scalar(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return x.signum();
        }
        if x < 0.0 {
            return -erf_scalar(-x);
        }
        if x > 1.0 {
            return 1.0 - erfc_scalar(x);
        }
        let square = x * x;
        x * cephes_polevl(square, &CEPHES_ERF_T) / cephes_p1evl(square, &CEPHES_ERF_U)
    }

    fn erfc_scalar(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return if x.is_sign_positive() { 0.0 } else { 2.0 };
        }
        let absolute = x.abs();
        if absolute < 1.0 {
            return 1.0 - erf_scalar(x);
        }
        let exponent = -x * x;
        if exponent < -CEPHES_MAXLOG {
            return if x < 0.0 { 2.0 } else { 0.0 };
        }
        let exponential = exponent.exp();
        let (numerator, denominator) = if absolute < 8.0 {
            (
                cephes_polevl(absolute, &CEPHES_ERFC_P),
                cephes_p1evl(absolute, &CEPHES_ERFC_Q),
            )
        } else {
            (
                cephes_polevl(absolute, &CEPHES_ERFC_R),
                cephes_p1evl(absolute, &CEPHES_ERFC_S),
            )
        };
        let tail = exponential * numerator / denominator;
        if x < 0.0 { 2.0 - tail } else { tail }
    }

    fn normal_cdf(x: f64) -> f64 {
        0.5 * erfc_scalar(-x * std::f64::consts::FRAC_1_SQRT_2)
    }

    fn option_price(sigma: f64, params: &[f64]) -> f64 {
        let spot = params[0];
        let discounted_strike = params[1];
        let sqrt_t = params[2];
        let log_forward = params[3];
        let sigma_sqrt_t = sigma * sqrt_t;
        let d1 = log_forward / sigma_sqrt_t + 0.5 * sigma_sqrt_t;
        let d2 = d1 - sigma_sqrt_t;
        spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    }

    fn option_vega(sigma: f64, params: &[f64]) -> f64 {
        let spot = params[0];
        let sqrt_t = params[2];
        let log_forward = params[3];
        let sigma_sqrt_t = sigma * sqrt_t;
        let d1 = log_forward / sigma_sqrt_t + 0.5 * sigma_sqrt_t;
        let density = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
        spot * density * sqrt_t
    }

    fn fixed_point_map(friction: f64, params: &[f64]) -> f64 {
        let reynolds = params[0];
        let relative_roughness = params[1];
        let inverse =
            -2.0 * (relative_roughness / 3.7 + 2.51 / (reynolds * friction.sqrt())).log10();
        1.0 / (inverse * inverse)
    }

    fn colebrook_residual(friction: f64, params: &[f64]) -> f64 {
        let reynolds = params[0];
        let relative_roughness = params[1];
        1.0 / friction.sqrt()
            + 2.0 * (relative_roughness / 3.7 + 2.51 / (reynolds * friction.sqrt())).log10()
    }

    fn pressure_loss(friction: f64, params: &[f64]) -> f64 {
        friction * params[2] * params[3]
    }

    fn build_option_dataset() -> Dataset {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut fixture = Vec::with_capacity(BATCH);
        let mut params = Vec::with_capacity(BATCH);
        let mut fixture_bytes = Vec::with_capacity(BATCH * PAYLOAD_COLUMNS * 8);
        for _ in 0..BATCH {
            let spot = 80.0 + 40.0 * lcg_uniform(&mut state);
            let log_moneyness = -0.15 + 0.30 * lcg_uniform(&mut state);
            let strike = spot * (-log_moneyness).exp();
            let maturity = 0.25 + 1.75 * lcg_uniform(&mut state);
            let rate = 0.005 + 0.045 * lcg_uniform(&mut state);
            let true_volatility = 0.12 + 0.48 * lcg_uniform(&mut state);
            let sqrt_t = maturity.sqrt();
            let log_forward = (spot / strike).ln() + rate * maturity;
            let discounted_strike = strike * (-rate * maturity).exp();
            let mut row_params = vec![
                spot,
                discounted_strike,
                sqrt_t,
                log_forward,
                0.0,
                true_volatility,
            ];
            let target = option_price(true_volatility, &row_params);
            row_params[4] = target;
            let row = [spot, strike, maturity, rate, target, true_volatility];
            for value in row {
                fixture_bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [sqrt_t, log_forward, discounted_strike] {
                fixture_bytes.extend_from_slice(&value.to_le_bytes());
            }
            fixture.push(row);
            params.push(row_params);
        }
        let fixture_sha256 = format!("{:x}", Sha256::digest(&fixture_bytes));
        Dataset {
            fixture,
            params,
            fixture_bytes,
            fixture_sha256,
        }
    }

    fn build_hydraulic_dataset() -> Dataset {
        const DENSITY: f64 = 1_000.0;
        const VISCOSITY: f64 = 1.0e-3;
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        let mut fixture = Vec::with_capacity(BATCH);
        let mut params = Vec::with_capacity(BATCH);
        let mut fixture_bytes = Vec::with_capacity(BATCH * PAYLOAD_COLUMNS * 8);
        for _ in 0..BATCH {
            let diameter = 0.05 + 0.95 * lcg_uniform(&mut state);
            let velocity = 0.1 + 4.9 * lcg_uniform(&mut state);
            let length = 20.0 + 1_980.0 * lcg_uniform(&mut state);
            let relative_roughness = 1.0e-6 + (0.03 - 1.0e-6) * lcg_uniform(&mut state);
            let reynolds = DENSITY * velocity * diameter / VISCOSITY;
            let dynamic_pressure = 0.5 * DENSITY * velocity * velocity;
            let length_over_diameter = length / diameter;
            let row = [
                diameter,
                velocity,
                length,
                relative_roughness,
                reynolds,
                dynamic_pressure,
            ];
            for value in row {
                fixture_bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [length_over_diameter, DENSITY, VISCOSITY] {
                fixture_bytes.extend_from_slice(&value.to_le_bytes());
            }
            fixture.push(row);
            params.push(vec![
                reynolds,
                relative_roughness,
                length_over_diameter,
                dynamic_pressure,
            ]);
        }
        let fixture_sha256 = format!("{:x}", Sha256::digest(&fixture_bytes));
        Dataset {
            fixture,
            params,
            fixture_bytes,
            fixture_sha256,
        }
    }

    fn build_dataset() -> Dataset {
        match solver_mode() {
            SolverMode::FixedPoint => build_hydraulic_dataset(),
            SolverMode::Newton | SolverMode::Secant => build_option_dataset(),
        }
    }

    fn percentile_sorted(values: &[f64], quantile: f64) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        let position = quantile * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            values[lower]
        } else {
            let fraction = position - lower as f64;
            values[lower] * (1.0 - fraction) + values[upper] * fraction
        }
    }

    fn volatility_band(value: f64) -> usize {
        if value < 0.15 {
            0
        } else if value < 0.225 {
            1
        } else if value < 0.30 {
            2
        } else if value < 0.375 {
            3
        } else if value < 0.45 {
            4
        } else if value < 0.525 {
            5
        } else if value < 0.60 {
            6
        } else {
            7
        }
    }

    fn friction_band(value: f64) -> usize {
        if value < 0.0125 {
            0
        } else if value < 0.020 {
            1
        } else if value < 0.0275 {
            2
        } else if value < 0.035 {
            3
        } else if value < 0.045 {
            4
        } else if value < 0.060 {
            5
        } else if value < 0.080 {
            6
        } else {
            7
        }
    }

    fn summarize_option(data: &Dataset, roots: &[f64], converged: &[bool]) -> JobSummary {
        let mut price_errors = Vec::with_capacity(BATCH);
        let mut volatility_errors = Vec::with_capacity(BATCH);
        let mut finite = 0usize;
        let mut worst_price_index = 0usize;
        let mut max_price_error = f64::NEG_INFINITY;
        let mut worst_volatility_index = 0usize;
        let mut max_volatility_error = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        let mut bands = [0usize; 8];
        let mut checksum = 0.0;
        for (index, (&root, params)) in roots.iter().zip(&data.params).enumerate() {
            if root.is_finite() {
                finite += 1;
            }
            let price_error = (option_price(root, params) - params[4]).abs();
            let volatility_error = (root - params[5]).abs();
            if price_error > max_price_error {
                max_price_error = price_error;
                worst_price_index = index;
            }
            if volatility_error > max_volatility_error {
                max_volatility_error = volatility_error;
                worst_volatility_index = index;
            }
            price_errors.push(price_error);
            volatility_errors.push(volatility_error);
            sum += root;
            minimum = minimum.min(root);
            maximum = maximum.max(root);
            bands[volatility_band(root)] += 1;
            let weight = (index % 251 + 1) as f64 / 251.0;
            checksum += root * weight;
        }
        let secondary_mean = volatility_errors.iter().sum::<f64>() / BATCH as f64;
        price_errors.sort_by(f64::total_cmp);
        volatility_errors.sort_by(f64::total_cmp);
        JobSummary {
            finite,
            converged: converged.iter().filter(|&&value| value).count(),
            zero_derivative: 0,
            worst_price_index,
            price_p50: percentile_sorted(&price_errors, 0.50),
            price_p95: percentile_sorted(&price_errors, 0.95),
            price_p99: percentile_sorted(&price_errors, 0.99),
            max_price_error,
            worst_volatility_index,
            volatility_p50: percentile_sorted(&volatility_errors, 0.50),
            volatility_p95: percentile_sorted(&volatility_errors, 0.95),
            volatility_p99: percentile_sorted(&volatility_errors, 0.99),
            max_volatility_error,
            mean_volatility: sum / BATCH as f64,
            min_volatility: minimum,
            max_volatility: maximum,
            bands,
            secondary_mean,
            pressure_counts: [0; 3],
            checksum,
        }
    }

    fn summarize_hydraulic(data: &Dataset, roots: &[f64], converged: &[bool]) -> JobSummary {
        let mut residuals = Vec::with_capacity(BATCH);
        let mut pressure_losses = Vec::with_capacity(BATCH);
        let mut finite = 0usize;
        let mut worst_residual_index = 0usize;
        let mut max_residual = f64::NEG_INFINITY;
        let mut worst_pressure_index = 0usize;
        let mut max_pressure = f64::NEG_INFINITY;
        let mut friction_sum = 0.0;
        let mut minimum_friction = f64::INFINITY;
        let mut maximum_friction = f64::NEG_INFINITY;
        let mut bands = [0usize; 8];
        let mut pressure_counts = [0usize; 3];
        let mut pressure_sum = 0.0;
        let mut checksum = 0.0;
        for (index, (&friction, params)) in roots.iter().zip(&data.params).enumerate() {
            if friction.is_finite() {
                finite += 1;
            }
            let residual = colebrook_residual(friction, params).abs();
            let pressure = pressure_loss(friction, params);
            if residual > max_residual {
                max_residual = residual;
                worst_residual_index = index;
            }
            if pressure > max_pressure {
                max_pressure = pressure;
                worst_pressure_index = index;
            }
            residuals.push(residual);
            pressure_losses.push(pressure);
            friction_sum += friction;
            pressure_sum += pressure;
            minimum_friction = minimum_friction.min(friction);
            maximum_friction = maximum_friction.max(friction);
            bands[friction_band(friction)] += 1;
            pressure_counts[0] += usize::from(pressure >= 1.0e5);
            pressure_counts[1] += usize::from(pressure >= 1.0e6);
            pressure_counts[2] += usize::from(pressure >= 1.0e7);
            let weight = (index % 251 + 1) as f64 / 251.0;
            checksum += friction * weight + 1.0e-7 * pressure * weight;
        }
        residuals.sort_by(f64::total_cmp);
        pressure_losses.sort_by(f64::total_cmp);
        JobSummary {
            finite,
            converged: converged.iter().filter(|&&value| value).count(),
            zero_derivative: 0,
            worst_price_index: worst_residual_index,
            price_p50: percentile_sorted(&residuals, 0.50),
            price_p95: percentile_sorted(&residuals, 0.95),
            price_p99: percentile_sorted(&residuals, 0.99),
            max_price_error: max_residual,
            worst_volatility_index: worst_pressure_index,
            volatility_p50: percentile_sorted(&pressure_losses, 0.50),
            volatility_p95: percentile_sorted(&pressure_losses, 0.95),
            volatility_p99: percentile_sorted(&pressure_losses, 0.99),
            max_volatility_error: max_pressure,
            mean_volatility: friction_sum / BATCH as f64,
            min_volatility: minimum_friction,
            max_volatility: maximum_friction,
            bands,
            secondary_mean: pressure_sum / BATCH as f64,
            pressure_counts,
            checksum,
        }
    }

    fn summarize(data: &Dataset, roots: &[f64], converged: &[bool]) -> JobSummary {
        match solver_mode() {
            SolverMode::FixedPoint => summarize_hydraulic(data, roots, converged),
            SolverMode::Newton | SolverMode::Secant => summarize_option(data, roots, converged),
        }
    }

    fn execute_batch(data: &Dataset) -> Result<(JobSummary, Vec<f64>), String> {
        let mut roots = Vec::with_capacity(BATCH);
        let mut converged = Vec::with_capacity(BATCH);
        match solver_mode() {
            SolverMode::Newton => {
                for result in newton_many(
                    |sigma, params| option_price(sigma, params) - params[4],
                    option_vega,
                    0.30,
                    &data.params,
                    1.0e-10,
                    0.0,
                    50,
                ) {
                    match result {
                        Ok(result) => {
                            roots.push(result.root);
                            converged.push(result.converged);
                        }
                        Err(_) => {
                            roots.push(f64::NAN);
                            converged.push(false);
                        }
                    }
                }
            }
            SolverMode::Secant => {
                for result in secant_many(
                    |sigma, params| option_price(sigma, params) - params[4],
                    0.30,
                    None,
                    &data.params,
                    RootOptions {
                        xtol: 1.0e-10,
                        rtol: 0.0,
                        maxiter: 50,
                        ..RootOptions::default()
                    },
                ) {
                    match result {
                        Ok(result) => {
                            roots.push(result.root);
                            converged.push(result.converged);
                        }
                        Err(_) => {
                            roots.push(f64::NAN);
                            converged.push(false);
                        }
                    }
                }
            }
            SolverMode::FixedPoint => {
                for result in fixed_point_many(fixed_point_map, 0.02, &data.params, 1.0e-10, 500) {
                    match result {
                        Ok(root) => {
                            roots.push(root);
                            converged.push(true);
                        }
                        Err(_) => {
                            roots.push(f64::NAN);
                            converged.push(false);
                        }
                    }
                }
            }
        }
        if roots.len() != BATCH {
            return Err(format!(
                "FrankenSciPy root count mismatch: expected {BATCH}, got {}",
                roots.len()
            ));
        }
        let summary = summarize(data, &roots, &converged);
        Ok((summary, roots))
    }

    fn time_batch(data: &Dataset, repetitions: usize) -> Result<f64, String> {
        let started = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..repetitions {
            let (summary, roots) = execute_batch(data)?;
            checksum += summary.checksum + roots[roots.len() / 2];
        }
        black_box(checksum);
        Ok(started.elapsed().as_secs_f64())
    }

    fn parse_summary(payload: &str) -> Result<JobSummary, String> {
        let values = payload.split(',').collect::<Vec<_>>();
        if values.len() != 29 {
            return Err(format!(
                "malformed SciPy summary: expected 29 values, got {}",
                values.len()
            ));
        }
        Ok(JobSummary {
            finite: parse(values[0], "SciPy finite count")?,
            converged: parse(values[1], "SciPy convergence count")?,
            zero_derivative: parse(values[2], "SciPy zero-derivative count")?,
            worst_price_index: parse(values[3], "SciPy worst price index")?,
            price_p50: parse(values[4], "SciPy price p50")?,
            price_p95: parse(values[5], "SciPy price p95")?,
            price_p99: parse(values[6], "SciPy price p99")?,
            max_price_error: parse(values[7], "SciPy maximum price error")?,
            worst_volatility_index: parse(values[8], "SciPy worst volatility index")?,
            volatility_p50: parse(values[9], "SciPy volatility p50")?,
            volatility_p95: parse(values[10], "SciPy volatility p95")?,
            volatility_p99: parse(values[11], "SciPy volatility p99")?,
            max_volatility_error: parse(values[12], "SciPy maximum volatility error")?,
            mean_volatility: parse(values[13], "SciPy mean volatility")?,
            min_volatility: parse(values[14], "SciPy minimum volatility")?,
            max_volatility: parse(values[15], "SciPy maximum volatility")?,
            bands: [
                parse(values[16], "SciPy band zero")?,
                parse(values[17], "SciPy band one")?,
                parse(values[18], "SciPy band two")?,
                parse(values[19], "SciPy band three")?,
                parse(values[20], "SciPy band four")?,
                parse(values[21], "SciPy band five")?,
                parse(values[22], "SciPy band six")?,
                parse(values[23], "SciPy band seven")?,
            ],
            secondary_mean: parse(values[24], "SciPy secondary mean")?,
            pressure_counts: [
                parse(values[25], "SciPy severity zero")?,
                parse(values[26], "SciPy severity one")?,
                parse(values[27], "SciPy severity two")?,
            ],
            checksum: parse(values[28], "SciPy checksum")?,
        })
    }

    fn quality_eligible(label: &str, summary: JobSummary) -> bool {
        let common = summary.finite == BATCH
            && summary.converged == BATCH
            && summary.zero_derivative == 0
            && summary.bands.iter().sum::<usize>() == BATCH
            && summary.checksum.is_finite();
        let eligible = match solver_mode() {
            SolverMode::FixedPoint => {
                common
                    && summary.max_price_error.is_finite()
                    && summary.max_price_error <= MAX_COLEBROOK_RESIDUAL
                    && summary.max_volatility_error.is_finite()
                    && summary.max_volatility_error >= 0.0
                    && summary.secondary_mean.is_finite()
                    && summary.secondary_mean >= 0.0
                    && summary.min_volatility >= 0.005
                    && summary.max_volatility <= 0.10
                    && summary
                        .pressure_counts
                        .windows(2)
                        .all(|counts| counts[0] >= counts[1])
                    && summary.pressure_counts[0] <= BATCH
            }
            SolverMode::Newton | SolverMode::Secant => {
                common
                    && summary.max_price_error.is_finite()
                    && summary.max_price_error <= MAX_PRICE_ERROR
                    && summary.max_volatility_error.is_finite()
                    && summary.max_volatility_error <= MAX_VOLATILITY_ERROR
            }
        };
        if !eligible {
            println!(
                "scientific_gate_failed label={label} finite={} converged={} \
                 zero_derivative={} max_price_error={:.17e} \
                 max_volatility_error={:.17e} root_min={:.17e} root_max={:.17e} \
                 band_total={} pressure_counts={:?} checksum={:.17e}",
                summary.finite,
                summary.converged,
                summary.zero_derivative,
                summary.max_price_error,
                summary.max_volatility_error,
                summary.min_volatility,
                summary.max_volatility,
                summary.bands.iter().sum::<usize>(),
                summary.pressure_counts,
                summary.checksum
            );
        }
        eligible
    }

    fn cross_quality(
        ours_roots: &[f64],
        ours: JobSummary,
        scipy_roots: &[f64],
        scipy: JobSummary,
    ) -> Result<f64, String> {
        if ours_roots.len() != BATCH || scipy_roots.len() != BATCH {
            return Err("cross-quality root length mismatch".to_string());
        }
        let mut maximum_error = 0.0_f64;
        for (index, (&got, &reference)) in ours_roots.iter().zip(scipy_roots).enumerate() {
            let error = (got - reference).abs();
            let tolerance = match solver_mode() {
                SolverMode::FixedPoint => {
                    MAX_CROSS_FRICTION_ERROR_ABS + MAX_CROSS_FRICTION_ERROR_REL * reference.abs()
                }
                SolverMode::Newton | SolverMode::Secant => {
                    MAX_CROSS_VOLATILITY_ERROR_ABS
                        + MAX_CROSS_VOLATILITY_ERROR_REL * reference.abs()
                }
            };
            if !error.is_finite() || error > tolerance {
                return Err(format!(
                    "cross-root mismatch at {index}: ours={got:.17e} \
                     scipy={reference:.17e} error={error:.17e} \
                     tolerance={tolerance:.17e}"
                ));
            }
            maximum_error = maximum_error.max(error);
        }
        if ours.bands != scipy.bands {
            return Err(format!(
                "root-band mismatch: ours={:?} scipy={:?}",
                ours.bands, scipy.bands
            ));
        }
        if matches!(solver_mode(), SolverMode::FixedPoint)
            && ours
                .pressure_counts
                .iter()
                .zip(scipy.pressure_counts)
                .any(|(left, right)| left.cmp(&right).is_ne())
        {
            return Err(format!(
                "pressure-count mismatch: ours={:?} scipy={:?}",
                ours.pressure_counts, scipy.pressure_counts
            ));
        }
        let price_index_noise_floor = match solver_mode() {
            SolverMode::Newton => PRICE_INDEX_NOISE_FLOOR,
            SolverMode::Secant => SECANT_PRICE_INDEX_NOISE_FLOOR,
            SolverMode::FixedPoint => COLEBROOK_INDEX_NOISE_FLOOR,
        };
        let price_below_floor = ours.max_price_error <= price_index_noise_floor
            && scipy.max_price_error <= price_index_noise_floor;
        if !price_below_floor && ours.worst_price_index != scipy.worst_price_index {
            return Err(format!(
                "worst-price indices disagree: ours={} scipy={}",
                ours.worst_price_index, scipy.worst_price_index
            ));
        }
        let volatility_below_floor = match solver_mode() {
            SolverMode::FixedPoint => {
                let scale = ours
                    .max_volatility_error
                    .abs()
                    .max(scipy.max_volatility_error.abs())
                    .max(1.0);
                (ours.max_volatility_error - scipy.max_volatility_error).abs() <= 1.0e-12 * scale
            }
            SolverMode::Newton | SolverMode::Secant => {
                ours.max_volatility_error <= VOLATILITY_INDEX_NOISE_FLOOR
                    && scipy.max_volatility_error <= VOLATILITY_INDEX_NOISE_FLOOR
            }
        };
        if !volatility_below_floor && ours.worst_volatility_index != scipy.worst_volatility_index {
            return Err(format!(
                "worst-volatility indices disagree: ours={} scipy={}",
                ours.worst_volatility_index, scipy.worst_volatility_index
            ));
        }
        Ok(maximum_error)
    }

    fn print_summary(label: &str, summary: JobSummary) {
        if matches!(solver_mode(), SolverMode::FixedPoint) {
            println!(
                "{label}: finite={}/{} converged={}/{} invalid_slope={} \
                 worst_colebrook_residual_index={} colebrook_residual_p50={:.17e} \
                 p95={:.17e} p99={:.17e} max={:.17e} \
                 worst_pressure_loss_index={} pressure_loss_p50={:.17e} \
                 p95={:.17e} p99={:.17e} max={:.17e} mean={:.17e} \
                 friction_mean={:.17e} min={:.17e} max={:.17e} \
                 friction_bands={:?} pressure_severity_counts={:?} checksum={:.17e}",
                summary.finite,
                BATCH,
                summary.converged,
                BATCH,
                summary.zero_derivative,
                summary.worst_price_index,
                summary.price_p50,
                summary.price_p95,
                summary.price_p99,
                summary.max_price_error,
                summary.worst_volatility_index,
                summary.volatility_p50,
                summary.volatility_p95,
                summary.volatility_p99,
                summary.max_volatility_error,
                summary.secondary_mean,
                summary.mean_volatility,
                summary.min_volatility,
                summary.max_volatility,
                summary.bands,
                summary.pressure_counts,
                summary.checksum
            );
            return;
        }
        println!(
            "{label}: finite={}/{} converged={}/{} zero_derivative={} \
             worst_price_index={} price_error_p50={:.17e} p95={:.17e} \
             p99={:.17e} max={:.17e} worst_volatility_index={} \
             volatility_error_p50={:.17e} p95={:.17e} p99={:.17e} \
             max={:.17e} volatility_mean={:.17e} min={:.17e} max={:.17e} \
             bands={:?} checksum={:.17e}",
            summary.finite,
            BATCH,
            summary.converged,
            BATCH,
            summary.zero_derivative,
            summary.worst_price_index,
            summary.price_p50,
            summary.price_p95,
            summary.price_p99,
            summary.max_price_error,
            summary.worst_volatility_index,
            summary.volatility_p50,
            summary.volatility_p95,
            summary.volatility_p99,
            summary.max_volatility_error,
            summary.mean_volatility,
            summary.min_volatility,
            summary.max_volatility,
            summary.bands,
            summary.checksum
        );
    }

    fn observe_batch_workers(data: &Dataset, expected: usize) -> Result<usize, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let observed_tasks = Arc::new(Mutex::new(BTreeSet::<String>::new()));
        let (ready_tx, ready_rx) = mpsc::sync_channel(0);
        let sampler_stop = Arc::clone(&stop);
        let sampler_observed_tasks = Arc::clone(&observed_tasks);
        let primary_task = std::fs::read_to_string("/proc/self/stat")
            .map_err(|error| format!("read primary task identity: {error}"))?
            .split_whitespace()
            .next()
            .ok_or_else(|| "primary task identity is empty".to_string())?
            .to_string();
        let sampler = std::thread::Builder::new()
            .name("fsci-task-sampler".to_string())
            .spawn(move || {
                let _ = ready_tx.send(());
                while !sampler_stop.load(Ordering::Acquire) {
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        for entry in entries.filter_map(Result::ok) {
                            let tid = entry.file_name().to_string_lossy().to_string();
                            if tid == primary_task {
                                continue;
                            }
                            let comm = std::fs::read_to_string(entry.path().join("comm"))
                                .unwrap_or_default();
                            if comm.trim().starts_with("fsci-task") {
                                continue;
                            }
                            if let Ok(mut observed) = sampler_observed_tasks.lock() {
                                observed.insert(tid);
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_micros(50));
                }
            })
            .map_err(|error| format!("spawn worker sampler: {error}"))?;
        ready_rx
            .recv()
            .map_err(|error| format!("start worker sampler: {error}"))?;
        for _ in 0..4 {
            black_box(execute_batch(data)?);
            let observed = observed_tasks
                .lock()
                .map_err(|_| "worker observation lock poisoned".to_string())?
                .len();
            if observed >= expected {
                break;
            }
        }
        stop.store(true, Ordering::Release);
        sampler
            .join()
            .map_err(|_| "worker sampler panicked".to_string())?;
        let observed = observed_tasks
            .lock()
            .map_err(|_| "worker observation lock poisoned".to_string())?
            .len();
        if observed == 0 {
            return Err("no FrankenSciPy worker task was directly observed".to_string());
        }
        Ok(observed)
    }

    fn screen_arm(scipy: &mut Scipy, arm: &str) -> Result<Option<ScreenedArm>, String> {
        let check = scipy.check(arm)?;
        print_summary(&format!("screen SciPy {arm}"), check.summary);
        if !quality_eligible(&format!("SciPy {arm}"), check.summary) {
            println!(
                "screen_excluded arm={arm} reason=scientific_gate \
                 time_samples=0 eligible=false observed_active_tasks={} \
                 observed_os_tasks={} observed_worker_processes={} \
                 observed_callsite_threads={}",
                check.active_tasks,
                check.max_os_tasks,
                check.worker_processes,
                check.callsite_threads
            );
            return Ok(None);
        }
        let mut samples = Vec::with_capacity(SCREEN_ROUNDS);
        let mut max_active_tasks = check.active_tasks;
        let mut max_os_tasks = check.max_os_tasks;
        let mut max_worker_processes = check.worker_processes;
        let mut max_callsite_threads = check.callsite_threads;
        for round in 0..SCREEN_ROUNDS {
            let timing = scipy.time(arm, 1)?;
            max_active_tasks = max_active_tasks.max(timing.active_tasks);
            max_os_tasks = max_os_tasks.max(timing.max_os_tasks);
            max_worker_processes = max_worker_processes.max(timing.worker_processes);
            max_callsite_threads = max_callsite_threads.max(timing.callsite_threads);
            samples.push(timing.elapsed);
            println!(
                "screen arm={arm} round={round} seconds={:.9} \
                 observed_active_tasks={} observed_os_tasks={} \
                 observed_worker_processes={} observed_callsite_threads={}",
                timing.elapsed,
                timing.active_tasks,
                timing.max_os_tasks,
                timing.worker_processes,
                timing.callsite_threads
            );
        }
        let arm_median = median(samples.clone());
        println!(
            "screen_result arm={arm} median_seconds={arm_median:.9} \
             samples={} observed_active_tasks={max_active_tasks} \
             observed_os_tasks={max_os_tasks} \
             observed_worker_processes={max_worker_processes} \
             observed_callsite_threads={max_callsite_threads}",
            csv(&samples)
        );
        Ok(Some(ScreenedArm {
            check,
            median: arm_median,
            max_active_tasks,
            max_os_tasks,
            max_worker_processes,
            max_callsite_threads,
        }))
    }

    fn screen_public_arms(scipy: &mut Scipy) -> Result<Vec<ScreenedArm>, String> {
        let mut screened = Vec::with_capacity(SCIPY_ARMS.len());
        for arm in SCIPY_ARMS {
            if let Some(result) = screen_arm(scipy, arm)? {
                screened.push(result);
            }
        }
        if screened.is_empty() {
            return Err("every SciPy public arm failed the scientific gate".to_string());
        }
        Ok(screened)
    }

    #[derive(Clone)]
    struct Measurement {
        ours: Vec<f64>,
        scipy: Vec<f64>,
        ratios: Vec<f64>,
        ours_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
        max_scipy_active_tasks: usize,
        max_scipy_os_tasks: usize,
        max_scipy_worker_processes: usize,
        max_scipy_callsite_threads: usize,
    }

    fn effect_pair(
        scipy: &mut Scipy,
        arm: &str,
        data: &Dataset,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_batch(data, repetitions)?;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
                incumbent.worker_processes,
                incumbent.callsite_threads,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_batch(data, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
                incumbent.worker_processes,
                incumbent.callsite_threads,
            ))
        }
    }

    fn ours_null_pair(data: &Dataset, repetitions: usize, round: usize) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_batch(data, repetitions)?,
                time_batch(data, repetitions)?,
            )
        } else {
            let right = time_batch(data, repetitions)?;
            let left = time_batch(data, repetitions)?;
            (left, right)
        };
        Ok(left / right)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        arm: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, usize, usize, usize, usize), String> {
        let (left, right) = if round.is_multiple_of(2) {
            (scipy.time(arm, repetitions)?, scipy.time(arm, repetitions)?)
        } else {
            let right = scipy.time(arm, repetitions)?;
            let left = scipy.time(arm, repetitions)?;
            (left, right)
        };
        Ok((
            left.elapsed / right.elapsed,
            left.active_tasks.max(right.active_tasks),
            left.max_os_tasks.max(right.max_os_tasks),
            left.worker_processes.max(right.worker_processes),
            left.callsite_threads.max(right.callsite_threads),
        ))
    }

    fn measure(
        scipy: &mut Scipy,
        arm: &str,
        data: &Dataset,
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(scipy, arm, data, repetitions, warmup)?;
        }
        let mut measurement = Measurement {
            ours: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            ratios: Vec::with_capacity(rounds),
            ours_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
            max_scipy_active_tasks: 0,
            max_scipy_os_tasks: 0,
            max_scipy_worker_processes: 0,
            max_scipy_callsite_threads: 0,
        };
        for round in 0..rounds {
            let (effect, ours_null, scipy_null) = match round % 3 {
                0 => (
                    effect_pair(scipy, arm, data, repetitions, round)?,
                    ours_null_pair(data, repetitions, round)?,
                    scipy_null_pair(scipy, arm, repetitions, round)?,
                ),
                1 => {
                    let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                    let effect = effect_pair(scipy, arm, data, repetitions, round)?;
                    let ours_null = ours_null_pair(data, repetitions, round)?;
                    (effect, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(data, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                    let effect = effect_pair(scipy, arm, data, repetitions, round)?;
                    (effect, ours_null, scipy_null)
                }
            };
            let effect_active = effect.2.max(scipy_null.1);
            let effect_tasks = effect.3.max(scipy_null.2);
            let effect_processes = effect.4.max(scipy_null.3);
            let effect_threads = effect.5.max(scipy_null.4);
            measurement.ours.push(effect.0);
            measurement.scipy.push(effect.1);
            measurement.ratios.push(effect.1 / effect.0);
            measurement.ours_nulls.push(ours_null);
            measurement.scipy_nulls.push(scipy_null.0);
            measurement.max_scipy_active_tasks =
                measurement.max_scipy_active_tasks.max(effect_active);
            measurement.max_scipy_os_tasks = measurement.max_scipy_os_tasks.max(effect_tasks);
            measurement.max_scipy_worker_processes =
                measurement.max_scipy_worker_processes.max(effect_processes);
            measurement.max_scipy_callsite_threads =
                measurement.max_scipy_callsite_threads.max(effect_threads);
            println!(
                "round={round} ours_seconds={:.9} scipy_seconds={:.9} \
                 ratio={:.9} ours_null={ours_null:.9} scipy_null={:.9} \
                 observed_scipy_active_tasks={effect_active} \
                 observed_scipy_os_tasks={effect_tasks} \
                 observed_scipy_worker_processes={effect_processes} \
                 observed_scipy_callsite_threads={effect_threads}",
                effect.0,
                effect.1,
                effect.1 / effect.0,
                scipy_null.0
            );
        }
        Ok(measurement)
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            return f64::NAN;
        }
        if values.len() % 2 == 1 {
            values[values.len() / 2]
        } else {
            0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
        }
    }

    fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        percentile_sorted(&values, quantile)
    }

    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let mut bootstraps = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut sample = Vec::with_capacity(values.len());
            for _ in 0..values.len() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                sample.push(values[(state as usize) % values.len()]);
            }
            bootstraps.push(median(sample));
        }
        bootstraps.sort_by(f64::total_cmp);
        (
            percentile_sorted(&bootstraps, 0.025),
            percentile_sorted(&bootstraps, 0.975),
        )
    }

    fn coefficient_of_variation(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().saturating_sub(1).max(1) as f64;
        variance.sqrt() / mean
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    struct Decision {
        outcome: &'static str,
        ratio_median: f64,
        ratio_low: f64,
        ratio_high: f64,
        decidable: bool,
    }

    fn print_decision(arm: &str, measurement: &Measurement, repetitions: usize) -> Decision {
        let ratio_median = median(measurement.ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&measurement.ratios);
        let ours_null_median = median(measurement.ours_nulls.clone());
        let scipy_null_median = median(measurement.scipy_nulls.clone());
        let (ours_null_low, ours_null_high) = bootstrap_median_ci(&measurement.ours_nulls);
        let (scipy_null_low, scipy_null_high) = bootstrap_median_ci(&measurement.scipy_nulls);
        let ours_p50 = median(measurement.ours.clone()) / repetitions as f64;
        let scipy_p50 = median(measurement.scipy.clone()) / repetitions as f64;
        let ours_p95 = percentile(measurement.ours.clone(), 0.95) / repetitions as f64;
        let scipy_p95 = percentile(measurement.scipy.clone(), 0.95) / repetitions as f64;
        let ours_p99 = percentile(measurement.ours.clone(), 0.99) / repetitions as f64;
        let scipy_p99 = percentile(measurement.scipy.clone(), 0.99) / repetitions as f64;
        println!(
            "whole_job_wall: selected_public_scipy_arm={arm} \
             FrankenSciPy_p50={:.9}ms p95={:.9}ms p99={:.9}ms \
             SciPy_p50={:.9}ms p95={:.9}ms p99={:.9}ms",
            ours_p50 * 1.0e3,
            ours_p95 * 1.0e3,
            ours_p99 * 1.0e3,
            scipy_p50 * 1.0e3,
            scipy_p95 * 1.0e3,
            scipy_p99 * 1.0e3
        );
        println!(
            "raw_samples_seconds: ours={} scipy={} ratios={} \
             null_ours={} null_scipy={}",
            csv(&measurement.ours),
            csv(&measurement.scipy),
            csv(&measurement.ratios),
            csv(&measurement.ours_nulls),
            csv(&measurement.scipy_nulls)
        );
        println!(
            "NULL-ours A/A median={ours_null_median:.9} \
             ci95=[{ours_null_low:.9},{ours_null_high:.9}] cv={:.3}% \
             ci_straddles_one={} provenance_only=true",
            coefficient_of_variation(&measurement.ours_nulls) * 100.0,
            ours_null_low <= 1.0 && ours_null_high >= 1.0
        );
        println!(
            "NULL-scipy A/A median={scipy_null_median:.9} \
             ci95=[{scipy_null_low:.9},{scipy_null_high:.9}] cv={:.3}% \
             ci_straddles_one={} provenance_only=true",
            coefficient_of_variation(&measurement.scipy_nulls) * 100.0,
            scipy_null_low <= 1.0 && scipy_null_high >= 1.0
        );
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_median:.9}x \
             bootstrap_median_ci95=[{ratio_low:.9},{ratio_high:.9}] \
             cv={:.3}% provenance_only=true",
            coefficient_of_variation(&measurement.ratios) * 100.0
        );

        let null_half_width = ((ours_null_high - ours_null_low) / 2.0)
            .max((scipy_null_high - scipy_null_low) / 2.0)
            .max(0.0);
        let null_endpoint_deviation = (ours_null_low - 1.0)
            .abs()
            .max((ours_null_high - 1.0).abs())
            .max((scipy_null_low - 1.0).abs())
            .max((scipy_null_high - 1.0).abs());
        let c1 = ratio_low > 1.0 || ratio_high < 1.0;
        let point_effect_deviation = (ratio_median - 1.0).abs();
        let effect_endpoint_deviation = if ratio_low > 1.0 {
            ratio_low - 1.0
        } else if ratio_high < 1.0 {
            1.0 - ratio_high
        } else {
            0.0
        };
        let c2 = point_effect_deviation > 2.0 * null_half_width;
        let c2b = effect_endpoint_deviation > 2.0 * null_endpoint_deviation;
        let c3 = (ours_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (scipy_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let decidable = c1 && c2 && c2b && c3;
        let outcome = if decidable && ratio_low > 1.0 {
            "DECIDED FRANKENSCIPY WIN"
        } else if decidable && ratio_high < 1.0 {
            "DECIDED FRANKENSCIPY LOSS"
        } else {
            "NOT DECIDED"
        };
        println!(
            "corrected_null_gate: c1_effect_ci_excludes_one={c1} \
             c2_point_effect_beats_2x_half_width={c2} \
             c2b_effect_endpoint_beats_2x_null_endpoint={c2b} \
             c3_null_medians_within_2pct={c3} decidable={decidable} \
             point_effect_deviation={point_effect_deviation:.9} \
             effect_endpoint_deviation={effect_endpoint_deviation:.9} \
             null_half_width={null_half_width:.9} \
             null_endpoint_deviation={null_endpoint_deviation:.9} \
             required_c2={:.9} required_c2b={:.9} \
             ours_null_median={ours_null_median:.9} \
             scipy_null_median={scipy_null_median:.9} \
             null_ci_veto=disabled_telemetry_only outcome={outcome}",
            2.0 * null_half_width,
            2.0 * null_endpoint_deviation
        );
        Decision {
            outcome,
            ratio_median,
            ratio_low,
            ratio_high,
            decidable,
        }
    }

    #[derive(Clone, Copy)]
    struct CpuTicks {
        total: u64,
        idle: u64,
    }

    fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
        let stat = std::fs::read_to_string("/proc/stat")
            .map_err(|error| format!("read /proc/stat: {error}"))?;
        let mut cpus = BTreeMap::new();
        for line in stat.lines() {
            let mut fields = line.split_whitespace();
            let Some(label) = fields.next() else {
                continue;
            };
            let Some(index) = label.strip_prefix("cpu") else {
                continue;
            };
            if index.is_empty() {
                continue;
            }
            let Ok(index) = index.parse::<usize>() else {
                continue;
            };
            let ticks = fields
                .map(|value| value.parse::<u64>().unwrap_or(0))
                .collect::<Vec<_>>();
            if ticks.len() < 5 {
                continue;
            }
            cpus.insert(
                index,
                CpuTicks {
                    total: ticks.iter().sum(),
                    idle: ticks[3] + ticks.get(4).copied().unwrap_or(0),
                },
            );
        }
        Ok(cpus)
    }

    /// Sample host-wide load and REPORT it. This deliberately does not abort.
    ///
    /// It used to: any single CPU above `HOST_QUIESCENCE_MAX_BUSY` failed the
    /// run. On a shared 64-way box that condition is never met, so this harness
    /// aborted before it measured anything — it has a live-SciPy incumbent arm
    /// and no ratio against it, because the gate fires first. A gate that cannot
    /// be satisfied does not enforce rigour; it converts measurable losses into
    /// unmeasured ones (frankenscipy-w1vdc).
    ///
    /// The substitution is the one `scripts/ledger_preflight.py` already
    /// sanctions: `NOT_CERTIFIED(host_mean_busy=N)` is admissible ONLY for a row
    /// carrying same-invocation A/A nulls, and this harness interleaves an
    /// independent null for both arms in every round. The null DETECTS the
    /// contention quiescence tried to exclude in advance, by measuring the
    /// interference that actually occurred rather than asserting in advance that
    /// none would.
    ///
    /// Measured support for dropping the ABSOLUTE bound: across six
    /// balanced-square runs banked in `docs/NEGATIVE_EVIDENCE.md` the busiest
    /// sample on record, a saturated box at `host_mean_busy=0.988`, produced the
    /// TIGHTEST A/A null of any run, and the quietest at 0.135 produced the
    /// loosest that still passed. A load-DELTA criterion did not reproduce there
    /// either, so none is imposed — the null is the gate.
    fn report_host_wide_quiescence(label: &str) -> Result<(), String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        let mut busiest = 0.0_f64;
        let mut busiest_cpu = 0usize;
        let mut total_busy_fraction = 0.0_f64;
        let mut sampled_cpus = 0u32;
        for (cpu, start) in before {
            let Some(end) = after.get(&cpu) else {
                continue;
            };
            let total = end.total.saturating_sub(start.total);
            let idle = end.idle.saturating_sub(start.idle);
            if total == 0 {
                continue;
            }
            let busy = 1.0 - idle as f64 / total as f64;
            total_busy_fraction += busy;
            sampled_cpus += 1;
            if busy > busiest {
                busiest = busy;
                busiest_cpu = cpu;
            }
        }
        if sampled_cpus == 0 {
            return Err("no CPU accumulated load-sample ticks".to_string());
        }
        let host_mean_busy = total_busy_fraction / f64::from(sampled_cpus);
        println!(
            "host_quiescence label={label} busiest_cpu={busiest_cpu} \
             busiest_fraction={busiest:.6} host_mean_busy={host_mean_busy:.6} \
             threshold={HOST_QUIESCENCE_MAX_BUSY:.6}"
        );
        // The token the ledger recognises. `clear` remains the strongest form and
        // is still claimed when the host genuinely is quiet; otherwise the row
        // states how busy the box was and names the A/A nulls as its gate.
        if busiest > HOST_QUIESCENCE_MAX_BUSY {
            println!(
                "host_wide_quiescence_{label}=NOT_CERTIFIED(host_mean_busy={host_mean_busy:.3}) \
                 sampled_cpus={sampled_cpus} busiest_cpu={busiest_cpu} \
                 busiest_fraction={busiest:.3} limit={HOST_QUIESCENCE_MAX_BUSY:.3} \
                 gate=same_invocation_A/A_nulls"
            );
        } else {
            println!(
                "host_wide_quiescence_{label}=clear sampled_cpus={sampled_cpus} \
                 host_mean_busy={host_mean_busy:.3} busiest_fraction={busiest:.3} \
                 busy_cpu_count_above_limit=0 limit={HOST_QUIESCENCE_MAX_BUSY:.3}"
            );
        }
        Ok(())
    }

    fn affinity_cpus() -> Result<Vec<usize>, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read process status: {error}"))?;
        let allowed = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .ok_or_else(|| "process status omitted Cpus_allowed_list".to_string())?;
        let mut cpus = Vec::new();
        for range in allowed.split(',') {
            if let Some((start, end)) = range.split_once('-') {
                let start: usize = parse(start, "affinity range start")?;
                let end: usize = parse(end, "affinity range end")?;
                if start > end {
                    return Err(format!("invalid affinity range {range}"));
                }
                cpus.extend(start..=end);
            } else {
                cpus.push(parse(range, "affinity CPU")?);
            }
        }
        if cpus.is_empty() {
            return Err("empty CPU affinity".to_string());
        }
        Ok(cpus)
    }

    fn require_performance_governor(cpus: &[usize]) -> Result<(), String> {
        let mut rows = Vec::with_capacity(cpus.len());
        for &cpu in cpus {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor");
            let governor = std::fs::read_to_string(&path)
                .map_err(|error| format!("read {path}: {error}"))?
                .trim()
                .to_string();
            if governor != "performance" {
                return Err(format!("cpu{cpu} governor is {governor}, not performance"));
            }
            rows.push(format!("{cpu}:{governor}"));
        }
        println!("affinity_governors={}", rows.join(","));
        Ok(())
    }

    fn current_hostname() -> Result<String, String> {
        std::fs::read_to_string("/etc/hostname")
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("read hostname: {error}"))
    }

    fn current_boot_id() -> Result<String, String> {
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("read boot ID: {error}"))
    }

    fn require_runtime_isa() -> Result<String, String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let flags = cpuinfo
            .lines()
            .find_map(|line| {
                line.strip_prefix("flags")
                    .and_then(|row| row.split_once(':'))
            })
            .map(|(_, flags)| flags.split_whitespace().collect::<Vec<_>>())
            .ok_or_else(|| "cpuinfo omitted runtime flags".to_string())?;
        for required in ["avx2", "fma"] {
            if !flags.contains(&required) {
                return Err(format!("runtime ISA omitted {required}"));
            }
        }
        Ok("avx2+fma".to_string())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes =
            std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn require_hex_sha(value: &str, label: &str) -> Result<(), String> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("{label} is not a SHA-256 digest: {value}"));
        }
        Ok(())
    }

    fn require_commit(value: &str) -> Result<(), String> {
        if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("invalid source commit identity: {value}"));
        }
        Ok(())
    }

    fn require_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required environment variable {name} is absent"))
    }

    fn require_booking_claim() -> Result<String, String> {
        let claim = require_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        if claim.is_empty() || !claim.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "invalid numeric trj booking claim message ID: {claim}"
            ));
        }
        Ok(claim)
    }

    fn rust_version() -> Result<String, String> {
        let output = Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|error| format!("run rustc --version: {error}"))?;
        if !output.status.success() {
            return Err(format!("rustc --version exited with {}", output.status));
        }
        String::from_utf8(output.stdout)
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("decode rustc version: {error}"))
    }

    fn parse<T>(value: &str, label: &str) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        value
            .parse()
            .map_err(|error| format!("parse {label} from {value:?}: {error}"))
    }

    fn fields(line: &str) -> HashMap<&str, &str> {
        line.split_whitespace()
            .skip(1)
            .filter_map(|field| field.split_once('='))
            .collect()
    }

    fn parse_run_arguments() -> Result<(usize, usize, bool, SolverMode), String> {
        let mut smoke = false;
        let mut solver = SolverMode::Newton;
        let mut numeric = Vec::new();
        for argument in std::env::args().skip(1) {
            if argument == "--smoke" {
                smoke = true;
            } else if argument == "--secant" || argument == "--solver=secant" {
                solver = SolverMode::Secant;
            } else if argument == "--newton" || argument == "--solver=newton" {
                solver = SolverMode::Newton;
            } else if argument == "--fixed-point"
                || argument == "--solver=fixed_point"
                || argument == "--solver=fixed-point"
            {
                solver = SolverMode::FixedPoint;
            } else if argument.starts_with("--solver=") {
                return Err(format!(
                    "invalid solver selector {argument:?}; expected newton, secant, or fixed_point"
                ));
            } else {
                numeric.push(argument);
            }
        }
        if numeric.len() > 2 {
            return Err("usage: perf_newton_many_scipy [rounds] [repetitions] \
                 [--smoke] [--solver=newton|secant|fixed_point]"
                .to_string());
        }
        let rounds = numeric
            .first()
            .map(|value| parse(value, "round count"))
            .transpose()?
            .unwrap_or(DEFAULT_ROUNDS);
        let repetitions = numeric
            .get(1)
            .map(|value| parse(value, "repetition count"))
            .transpose()?
            .unwrap_or(DEFAULT_REPETITIONS);
        if rounds < 5 || repetitions == 0 {
            return Err("rounds must be at least five and repetitions positive".to_string());
        }
        Ok((rounds, repetitions, smoke, solver))
    }

    pub fn run() -> Result<(), String> {
        let (rounds, repetitions, smoke, solver) = parse_run_arguments()?;
        SOLVER_MODE
            .set(solver)
            .map_err(|_| "solver mode was initialized more than once".to_string())?;
        let hostname = current_hostname()?;
        let boot_id = current_boot_id()?;
        let affinity = affinity_cpus()?;
        let runtime_isa = require_runtime_isa()?;
        let executable = std::env::current_exe()
            .map_err(|error| format!("resolve running executable: {error}"))?;
        let elf_sha256 = sha256_file(&executable)?;
        require_hex_sha(&elf_sha256, "running ELF SHA-256")?;
        let source_commit = require_env("BINARY_SOURCE_COMMIT")?;
        require_commit(&source_commit)?;
        let builder_identity = require_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = require_env("BINARY_BUILD_ROUTE")?;
        if build_route != "rch-exec-base-clean-overlay-no-overlay" {
            return Err(format!("invalid strict build route: {build_route}"));
        }
        let booking_claim = if smoke {
            "non-evidence-smoke".to_string()
        } else {
            require_booking_claim()?
        };
        if !smoke {
            if hostname != "threadripperje" {
                return Err(format!(
                    "evidence run requires threadripperje, observed {hostname}"
                ));
            }
            if affinity.len() != 32 {
                return Err(format!(
                    "evidence run requires exactly 32 affinity CPUs, observed {}",
                    affinity.len()
                ));
            }
            require_performance_governor(&affinity)?;
            report_host_wide_quiescence("before_screen")?;
        }
        println!(
            "PROVENANCE mode={} solver_mode={} host={} boot_id={} affinity_cpus={} \
             actual_affinity_count={} runtime_isa={} rustc={} \
             elf_path={} elf_sha256={} source_commit={} builder_identity={} \
             build_route={} trj_booking_claim_message_id={} target_dir_policy=shared_reused",
            if smoke {
                "NON_EVIDENCE_SMOKE"
            } else {
                "EVIDENCE"
            },
            solver.label(),
            hostname,
            boot_id,
            affinity
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            affinity.len(),
            runtime_isa,
            rust_version()?,
            executable.display(),
            elf_sha256,
            source_commit,
            builder_identity,
            build_route,
            booking_claim
        );

        let data = build_dataset();
        require_hex_sha(&data.fixture_sha256, "fixture SHA-256")?;
        let (mut scipy, ready) = Scipy::start(&data, solver)?;
        println!("SCIPY_IDENTITY {ready}");
        let identity = fields(&ready);
        let (expected_solver_module, expected_numeric_module) = match solver {
            SolverMode::FixedPoint => (
                "scipy.optimize._minpack_py",
                "numpy._core._multiarray_umath",
            ),
            SolverMode::Newton | SolverMode::Secant => {
                ("scipy.optimize._zeros_py", "scipy.special._ufuncs")
            }
        };
        if !ready.starts_with("READY ")
            || identity.get("scipy") != Some(&PINNED_SCIPY)
            || identity.get("numpy") != Some(&PINNED_NUMPY)
            || identity.get("genuine") != Some(&"True")
            || identity.get("fsci_loaded") != Some(&"False")
            || identity.get("solver_mode") != Some(&solver.label())
            || identity.get("solver_module") != Some(&expected_solver_module)
            || identity.get("numeric_module") != Some(&expected_numeric_module)
            || identity.get("pool_start") != Some(&"fork")
        {
            return Err(format!("live SciPy identity gate failed: {ready}"));
        }
        for field in [
            "scipy_engine_sha256",
            "solver_engine_sha256",
            "numeric_engine_sha256",
            "fixture_sha256",
        ] {
            require_hex_sha(
                identity
                    .get(field)
                    .ok_or_else(|| format!("SciPy identity omitted {field}"))?,
                field,
            )?;
        }
        if identity
            .get("fixture_sha256")
            .is_none_or(|observed| observed.cmp(&data.fixture_sha256.as_str()).is_ne())
        {
            return Err(format!(
                "fixture SHA-256 mismatch: rust={} scipy={:?}",
                data.fixture_sha256,
                identity.get("fixture_sha256")
            ));
        }
        let worker_capacity: usize = parse(
            identity
                .get("worker_capacity")
                .ok_or_else(|| "SciPy identity omitted worker_capacity".to_string())?,
            "SciPy worker capacity",
        )?;
        let chunk_count: usize = parse(
            identity
                .get("chunk_count")
                .ok_or_else(|| "SciPy identity omitted chunk_count".to_string())?,
            "SciPy chunk count",
        )?;
        if worker_capacity != affinity.len() || chunk_count != affinity.len() * 8 {
            return Err(format!(
                "SciPy pool topology mismatch: capacity={worker_capacity} \
                 affinity={} chunks={chunk_count}",
                affinity.len()
            ));
        }
        let first = data
            .fixture
            .first()
            .ok_or_else(|| "empty fixture".to_string())?;
        let last = data
            .fixture
            .last()
            .ok_or_else(|| "empty fixture".to_string())?;
        match solver {
            SolverMode::FixedPoint => println!(
                "FIXTURE pipes={} input_columns=4 payload_columns={} \
                 fixture_bytes={} fixture_sha256={} diameter_range=0.05,1.0 \
                 velocity_range=0.1,5.0 length_range=20,2000 \
                 relative_roughness_range=1e-6,0.03 reynolds_min=5000 \
                 density=1000 viscosity=1e-3 solver_mode={} x0=0.02 \
                 method=del2 tol=1e-10 maxiter=500 \
                 first_reynolds={:.17e} last_reynolds={:.17e} \
                 first_relative_roughness={:.17e} last_relative_roughness={:.17e}",
                BATCH,
                PAYLOAD_COLUMNS,
                data.fixture_bytes.len(),
                data.fixture_sha256,
                solver.label(),
                first[4],
                last[4],
                first[3],
                last[3]
            ),
            SolverMode::Newton | SolverMode::Secant => println!(
                "FIXTURE contracts={} input_columns={} payload_columns={} \
                 fixture_bytes={} fixture_sha256={} \
                 spot_range=80,120 log_moneyness_range=-0.15,0.15 \
                 maturity_range=0.25,2 rate_range=0.005,0.05 \
                 true_volatility_range=0.12,0.60 solver_mode={} x0=0.30 \
                 second_point_policy={} tol=1e-10 maxiter=50 \
                 first_target={:.17e} last_target={:.17e}",
                BATCH,
                FIXTURE_COLUMNS,
                PAYLOAD_COLUMNS,
                data.fixture_bytes.len(),
                data.fixture_sha256,
                solver.label(),
                match solver {
                    SolverMode::Newton => "analytic_vega",
                    SolverMode::Secant => "public_api_default",
                    SolverMode::FixedPoint => "not_applicable",
                },
                first[4],
                last[4]
            ),
        }

        let (ours_summary, ours_roots) = execute_batch(&data)?;
        print_summary("FrankenSciPy", ours_summary);
        if !quality_eligible("FrankenSciPy", ours_summary) {
            return Err(format!(
                "FrankenSciPy failed the scientific gate: {ours_summary:?}"
            ));
        }
        let expected_workers = affinity.len().clamp(1, BATCH / 2048);
        let observed_rust_workers = observe_batch_workers(&data, expected_workers)?;
        println!(
            "ACTUAL_OBSERVED FrankenSciPy_worker_tasks={} \
             production_gate_workers={} affinity_capacity={} \
             requested_threads_not_used=true",
            observed_rust_workers,
            expected_workers,
            affinity.len()
        );

        let screened = screen_public_arms(&mut scipy)?;
        let selected = screened
            .iter()
            .min_by(|left, right| left.median.total_cmp(&right.median))
            .ok_or_else(|| "no valid SciPy public arm".to_string())?;
        let selected_roots = scipy.roots(&selected.check.arm)?;
        let maximum_cross_root_error = cross_quality(
            &ours_roots,
            ours_summary,
            &selected_roots,
            selected.check.summary,
        )?;
        println!(
            "selected_public_scipy_arm={} selected_screen_median_seconds={:.9} \
             selected_observed_active_tasks={} selected_observed_os_tasks={} \
             selected_observed_worker_processes={} \
             selected_observed_callsite_threads={} \
             max_cross_root_error={maximum_cross_root_error:.17e}",
            selected.check.arm,
            selected.median,
            selected.max_active_tasks,
            selected.max_os_tasks,
            selected.max_worker_processes,
            selected.max_callsite_threads
        );
        let scalar_median = screened
            .iter()
            .find(|entry| entry.check.arm == "scalar_loop")
            .map(|entry| entry.median);
        let best_array = screened
            .iter()
            .filter(|entry| entry.check.arm.starts_with("array_"))
            .min_by(|left, right| left.median.total_cmp(&right.median));
        let removed_scalar_tax_ratio = scalar_median
            .zip(best_array.map(|entry| entry.median))
            .map(|(scalar, array)| scalar / array);
        if let (Some(scalar), Some(array), Some(ratio)) = (
            scalar_median,
            best_array.map(|entry| entry.median),
            removed_scalar_tax_ratio,
        ) {
            println!(
                "mechanism_screen scalar_loop_over_best_array={ratio:.9}x \
                 predicted_at_least_20={} scalar_seconds={scalar:.9} \
                 best_array_seconds={array:.9} falsified_by_quality=false",
                ratio >= 20.0
            );
        } else {
            println!(
                "mechanism_screen scalar_loop_over_best_array=unavailable \
                 predicted_at_least_20=false falsified_by_quality=true"
            );
        }

        if smoke {
            println!(
                "NON_EVIDENCE_SMOKE_COMPLETE selected_public_scipy_arm={} \
                 scalar_loop_over_best_array={} actual_observed_rust_workers={} \
                 selected_worker_processes={} selected_callsite_threads={} \
                 no_effect_verdict=true",
                selected.check.arm,
                removed_scalar_tax_ratio
                    .map(|ratio| format!("{ratio:.9}x"))
                    .unwrap_or_else(|| "unavailable_quality_falsified".to_string()),
                observed_rust_workers,
                selected.max_worker_processes,
                selected.max_callsite_threads
            );
            scipy.stop()?;
            return Ok(());
        }

        report_host_wide_quiescence("before_effect")?;
        let measurement = measure(&mut scipy, &selected.check.arm, &data, rounds, repetitions)?;
        let decision = print_decision(&selected.check.arm, &measurement, repetitions);
        println!(
            "ACTUAL_OBSERVED selected_scipy_active_tasks={} \
             selected_scipy_max_os_tasks={} selected_scipy_worker_processes={} \
             selected_scipy_callsite_threads={} \
             requested_worker_capacity_not_substituted=true",
            measurement.max_scipy_active_tasks,
            measurement.max_scipy_os_tasks,
            measurement.max_scipy_worker_processes,
            measurement.max_scipy_callsite_threads
        );

        let p1 = selected.check.arm == "array_process";
        let p2 = decision.ratio_high < solver.collapse_boundary();
        let p3 = removed_scalar_tax_ratio.is_some_and(|ratio| ratio >= 20.0);
        let (p4_condition, p4) = match solver {
            SolverMode::Newton => (
                "ci_upper_below_3",
                decision.ratio_high < DURABLE_WIN_BOUNDARY,
            ),
            SolverMode::Secant | SolverMode::FixedPoint => (
                "ci_lower_above_3",
                decision.ratio_low > DURABLE_WIN_BOUNDARY,
            ),
        };
        let selected_is_process = selected.check.arm.ends_with("_process");
        let selected_is_thread = selected.check.arm.ends_with("_thread");
        let selected_is_single = selected.check.arm == "array_single";
        let p5_rust = observed_rust_workers == 32;
        let p5_scipy = if selected_is_process {
            measurement.max_scipy_worker_processes >= 16
        } else if selected_is_thread {
            measurement.max_scipy_callsite_threads > 1
        } else if selected_is_single {
            measurement.max_scipy_callsite_threads == 1
        } else {
            true
        };
        println!(
            "PREREGISTERED_PREDICTIONS P1_array_process_fastest={} \
             P2_ci_upper_below_{:.1}={} \
             P3_scalar_over_best_array_at_least_20={} \
             P4_condition={} P4_satisfied={} P5_rust_32_workers={} \
             P5_selected_scipy_observation={} selected_is_process={} \
             selected_is_thread={} selected_is_single_array={} \
             ratio_median={:.9} ratio_ci=[{:.9},{:.9}]",
            p1,
            solver.collapse_boundary(),
            p2,
            p3,
            p4_condition,
            p4,
            p5_rust,
            p5_scipy,
            selected_is_process,
            selected_is_thread,
            selected_is_single,
            decision.ratio_median,
            decision.ratio_low,
            decision.ratio_high
        );
        let durable_frankenscipy_win =
            decision.decidable && decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if durable_frankenscipy_win {
            solver.rust_api()
        } else {
            selected.check.arm.as_str()
        };
        let (job_count, job_kind) = match solver {
            SolverMode::Newton => (
                "65,536-contract",
                "implied-volatility calibration and risk report",
            ),
            SolverMode::Secant => (
                "65,536-contract",
                "derivative-free implied-volatility calibration and risk report",
            ),
            SolverMode::FixedPoint => (
                "65,536-pipe",
                "Colebrook friction-factor and Darcy pressure-loss report",
            ),
        };
        println!(
            "CHOOSER STATEMENT: choose {chooser} for this exact {job_count} \
             {job_kind}; solver_mode={} \
             durable_frankenscipy_boundary=3x durable_frankenscipy_win={} \
             outcome={} ratio_ci_low={:.9} old_scalar_loop_claim={} retired=true",
            solver.label(),
            durable_frankenscipy_win,
            decision.outcome,
            decision.ratio_low,
            solver.old_claim()
        );
        scipy.stop()?;
        Ok(())
    }
}

#[cfg(feature = "opt-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("perf_newton_many_scipy failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "opt-incumbent-bench"))]
fn main() {
    eprintln!("perf_newton_many_scipy requires --features opt-incumbent-bench");
    std::process::exit(2);
}
