//! Real multistart optimization job versus the strongest screened SciPy arm.
//!
//! The historical `minimize_many` row compared separately timed processes.
//! This harness runs a live SciPy 1.17.1 child in the same invocation, screens
//! every pre-registered public derivative arm, and applies the corrected
//! bootstrap-median gate with independent A/A controls.

#[cfg(feature = "opt-incumbent-bench")]
mod bench {
    use fsci_opt::{MinimizeOptions, OptimizeMethod, OptimizeResult, minimize, minimize_many};
    use fsci_runtime::RuntimeMode;
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    const BATCH: usize = 128;
    const DIMENSION: usize = 6;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 3;
    const SCREEN_ROUNDS: usize = 5;
    const DIAGNOSTIC_ROUNDS: usize = 9;
    const GLOBAL_THRESHOLD: f64 = 1.0e-8;
    const BEST_OBJECTIVE_LIMIT: f64 = 1.0e-12;
    const BEST_POINT_ERROR_LIMIT: f64 = 1.0e-4;
    const QUALITY_FRACTION: f64 = 0.95;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HISTORICAL_LOWER_RATIO: f64 = 275.0;
    const COLLAPSE_BOUNDARY: f64 = HISTORICAL_LOWER_RATIO / 2.0;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
    const SCIPY_ARMS: [&str; 4] = ["fd", "fd_workers", "analytic", "fused"];

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import _optimize

BATCH = 128
DIMENSION = 6
MASK = (1 << 64) - 1
STATE = 7
VALUES = []
for _ in range(BATCH * DIMENSION):
    STATE = (STATE * 6364136223846793005 + 1) & MASK
    VALUES.append(-2.0 + 4.0 * ((STATE >> 11) / float(1 << 53)))
STARTS = np.asarray(VALUES, dtype="<f8").reshape(BATCH, DIMENSION)
STARTS_SHA256 = hashlib.sha256(STARTS.tobytes(order="C")).hexdigest()
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
POOL = ThreadPool(WORKER_CAPACITY)

def fused_rosen(x):
    return float(optimize.rosen(x)), np.asarray(optimize.rosen_der(x), dtype=np.float64)

def solve_one(arm, start):
    options = {"gtol": 1.0e-8, "maxiter": 500}
    if arm == "fd":
        return optimize.minimize(
            optimize.rosen, start, method="BFGS", jac=None, options=options
        )
    if arm == "fd_workers":
        options["workers"] = POOL.map
        return optimize.minimize(
            optimize.rosen, start, method="BFGS", jac=None, options=options
        )
    if arm == "analytic":
        return optimize.minimize(
            optimize.rosen,
            start,
            method="BFGS",
            jac=optimize.rosen_der,
            options=options,
        )
    if arm == "fused":
        return optimize.minimize(
            fused_rosen, start, method="BFGS", jac=True, options=options
        )
    raise RuntimeError(f"unknown arm {arm!r}")

def summarize(results):
    if len(results) != BATCH:
        raise RuntimeError(f"expected {BATCH} results, got {len(results)}")
    fun = np.asarray([float(result.fun) for result in results], dtype=np.float64)
    points = np.asarray([result.x for result in results], dtype=np.float64)
    if fun.shape != (BATCH,) or points.shape != (BATCH, DIMENSION):
        raise RuntimeError("unexpected result shape")
    if not np.all(np.isfinite(fun)) or not np.all(np.isfinite(points)):
        raise RuntimeError("non-finite result")
    best_index = int(np.argmin(fun))
    best_fun = float(fun[best_index])
    best_error = float(np.max(np.abs(points[best_index] - 1.0)))
    success = sum(bool(result.success) for result in results)
    global_count = int(np.count_nonzero(fun <= 1.0e-8))
    nfev = sum(int(result.nfev or 0) for result in results)
    njev = sum(int(result.njev or 0) for result in results)
    nit = sum(int(result.nit or 0) for result in results)
    weights = np.arange(1, DIMENSION + 1, dtype=np.float64)
    checksum = float(
        np.sum(fun)
        + np.sum(points * weights)
        + success
        + global_count
        + nfev * 1.0e-6
        + njev * 1.0e-7
        + nit * 1.0e-8
    )
    return (
        best_fun,
        best_error,
        success,
        global_count,
        nfev,
        njev,
        nit,
        checksum,
    )

def run_job(arm):
    return summarize([solve_one(arm, start) for start in STARTS])

def task_runtime():
    runtime = {}
    for name in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{name}/schedstat", "r", encoding="ascii") as handle:
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

for arm in ("fd", "fd_workers", "analytic", "fused"):
    output = run_job(arm)
    if not np.isfinite(output[-1]):
        raise RuntimeError(f"{arm} warmup failed")

source_path = inspect.getsourcefile(_optimize)
if source_path is None:
    raise RuntimeError("cannot resolve scipy.optimize implementation")
with open(source_path, "rb") as source_file:
    source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and optimize.minimize.__module__ == "scipy.optimize._minimize"
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" minimize_module={optimize.minimize.__module__}"
    f" scipy_engine_path={source_path}"
    f" scipy_engine_sha256={source_sha256}"
    f" starts_sha256={STARTS_SHA256}"
    f" worker_capacity={WORKER_CAPACITY}"
    f" max_os_tasks={len(os.listdir('/proc/self/task'))}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={genuine}",
    flush=True,
)

for line in sys.stdin:
    fields = line.strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "CHECK" and len(fields) == 2:
        arm = fields[1]
        output, active, maximum = run_observed(lambda: run_job(arm))
        values = ",".join(str(value) if isinstance(value, int) else f"{value:.17e}" for value in output)
        print(f"CHECK {arm} {active} {maximum} {values}", flush=True)
    elif command == "TIME" and len(fields) == 3:
        arm = fields[1]
        repetitions = int(fields[2])
        if repetitions <= 0:
            raise RuntimeError("repetitions must be positive")
        before = task_runtime()
        checksum = 0.0
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            checksum += run_job(arm)[-1]
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        after = task_runtime()
        active = sum(after.get(tid, 0) > value for tid, value in before.items())
        active += sum(tid not in before and value > 0 for tid, value in after.items())
        print(
            f"TIME {arm} {elapsed:.17e} {checksum:.17e}"
            f" {max(1, active)} {max(len(before), len(after))}",
            flush=True,
        )
    elif command == "QUIT":
        POOL.close()
        POOL.join()
        print("BYE", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {line!r}")
"#;

    #[derive(Clone, Copy, Debug)]
    struct JobSummary {
        best_fun: f64,
        best_error: f64,
        success: usize,
        global_count: usize,
        nfev: usize,
        njev: usize,
        nit: usize,
        checksum: f64,
    }

    #[derive(Clone, Copy)]
    struct Timing {
        elapsed: f64,
        active_tasks: usize,
        max_os_tasks: usize,
    }

    #[derive(Clone)]
    struct ScipyCheck {
        arm: String,
        active_tasks: usize,
        max_os_tasks: usize,
        summary: JobSummary,
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    impl Scipy {
        fn start() -> Result<(Self, String), String> {
            let python =
                std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "/usr/bin/python3.13".to_string());
            let mut child = Command::new(&python)
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("PYTHONPATH", SCIPY_SITE_PACKAGES)
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("spawn live SciPy oracle {python}: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy oracle has no stdin".to_string())?;
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
                .map_err(|error| format!("write {context}: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush {context}: {error}"))?;
            let mut reply = String::new();
            self.stdout
                .read_line(&mut reply)
                .map_err(|error| format!("read {context}: {error}"))?;
            if reply.is_empty() {
                return Err(format!("live SciPy oracle exited during {context}"));
            }
            Ok(reply.trim().to_string())
        }

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let reply = self.request(&format!("CHECK {arm}"), "SciPy quality check")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("invalid SciPy quality reply: {reply}"));
            }
            Ok(ScipyCheck {
                arm: arm.to_string(),
                active_tasks: parse(fields[2], "SciPy active tasks")?,
                max_os_tasks: parse(fields[3], "SciPy max OS tasks")?,
                summary: parse_summary(fields[4], "SciPy summary")?,
            })
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let reply = self.request(
                &format!("TIME {arm} {repetitions}"),
                "SciPy timed multistart job",
            )?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" || fields[1] != arm {
                return Err(format!("invalid SciPy timing reply: {reply}"));
            }
            let elapsed: f64 = parse(fields[2], "SciPy elapsed")?;
            let checksum: f64 = parse(fields[3], "SciPy checksum")?;
            let active_tasks: usize = parse(fields[4], "SciPy active tasks")?;
            let max_os_tasks: usize = parse(fields[5], "SciPy max OS tasks")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || !checksum.is_finite()
                || active_tasks == 0
                || max_os_tasks == 0
            {
                return Err(format!("inadmissible SciPy timing: {reply}"));
            }
            Ok(Timing {
                elapsed,
                active_tasks,
                max_os_tasks,
            })
        }

        fn stop(&mut self) -> Result<(), String> {
            let reply = self.request("QUIT", "SciPy shutdown")?;
            if reply != "BYE" {
                return Err(format!("invalid SciPy shutdown reply: {reply}"));
            }
            let status = self
                .child
                .wait()
                .map_err(|error| format!("wait for SciPy oracle: {error}"))?;
            self.stopped = true;
            if !status.success() {
                return Err(format!("SciPy oracle exited with {status}"));
            }
            Ok(())
        }
    }

    impl Drop for Scipy {
        fn drop(&mut self) {
            if !self.stopped {
                let _ = writeln!(self.stdin, "QUIT");
                let _ = self.stdin.flush();
                let _ = self.child.wait();
            }
        }
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|pair| {
                let residual = pair[1] - pair[0] * pair[0];
                100.0 * residual * residual + (1.0 - pair[0]) * (1.0 - pair[0])
            })
            .sum()
    }

    fn rosenbrock_gradient(x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len()];
        for index in 0..x.len() - 1 {
            let residual = x[index + 1] - x[index] * x[index];
            gradient[index] += -400.0 * x[index] * residual - 2.0 * (1.0 - x[index]);
            gradient[index + 1] += 200.0 * residual;
        }
        gradient
    }

    fn starts() -> Vec<Vec<f64>> {
        let mut state = 7u64;
        (0..BATCH)
            .map(|_| {
                (0..DIMENSION)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        -2.0 + 4.0 * ((state >> 11) as f64 / (1u64 << 53) as f64)
                    })
                    .collect()
            })
            .collect()
    }

    fn starts_sha256(starts: &[Vec<f64>]) -> String {
        let mut digest = Sha256::new();
        for row in starts {
            for value in row {
                digest.update(value.to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    }

    fn options() -> MinimizeOptions {
        MinimizeOptions {
            method: Some(OptimizeMethod::Bfgs),
            tol: Some(1.0e-8),
            maxiter: Some(500),
            maxfev: Some(20_000),
            gradient: Some(rosenbrock_gradient),
            gradient_available: true,
            mode: RuntimeMode::Strict,
            ..MinimizeOptions::default()
        }
    }

    fn summarize_results(
        results: &[Result<OptimizeResult, fsci_opt::OptError>],
    ) -> Result<JobSummary, String> {
        if results.len() != BATCH {
            return Err(format!(
                "expected {BATCH} FrankenSciPy results, got {}",
                results.len()
            ));
        }
        let mut best_fun = f64::INFINITY;
        let mut best_error = f64::INFINITY;
        let mut success = 0usize;
        let mut global_count = 0usize;
        let mut nfev = 0usize;
        let mut njev = 0usize;
        let mut nit = 0usize;
        let mut checksum = 0.0;
        for (row, result) in results.iter().enumerate() {
            let result = result
                .as_ref()
                .map_err(|error| format!("FrankenSciPy row {row} failed: {error:?}"))?;
            if result.x.len() != DIMENSION {
                return Err(format!(
                    "FrankenSciPy row {row} returned dimension {}",
                    result.x.len()
                ));
            }
            let fun = result
                .fun
                .ok_or_else(|| format!("FrankenSciPy row {row} omitted objective"))?;
            if !fun.is_finite() || result.x.iter().any(|value| !value.is_finite()) {
                return Err(format!("FrankenSciPy row {row} returned non-finite output"));
            }
            if fun < best_fun {
                best_fun = fun;
                best_error = result
                    .x
                    .iter()
                    .map(|value| (value - 1.0).abs())
                    .fold(0.0, f64::max);
            }
            success += usize::from(result.success);
            global_count += usize::from(fun <= GLOBAL_THRESHOLD);
            nfev += result.nfev;
            njev += result.njev;
            nit += result.nit;
            checksum += fun;
            for (index, value) in result.x.iter().enumerate() {
                checksum += (index + 1) as f64 * value;
            }
        }
        checksum += success as f64
            + global_count as f64
            + nfev as f64 * 1.0e-6
            + njev as f64 * 1.0e-7
            + nit as f64 * 1.0e-8;
        Ok(JobSummary {
            best_fun,
            best_error,
            success,
            global_count,
            nfev,
            njev,
            nit,
            checksum,
        })
    }

    fn run_batch(starts: &[Vec<f64>]) -> Result<JobSummary, String> {
        summarize_results(&minimize_many(rosenbrock, starts, options()))
    }

    fn run_serial(starts: &[Vec<f64>]) -> Result<JobSummary, String> {
        let results = starts
            .iter()
            .map(|start| minimize(rosenbrock, start, options()))
            .collect::<Vec<_>>();
        summarize_results(&results)
    }

    fn time_batch(starts: &[Vec<f64>], repetitions: usize) -> Result<Timing, String> {
        let mut checksum = 0.0;
        let started = Instant::now();
        for _ in 0..repetitions {
            checksum = black_box(checksum + run_batch(black_box(starts))?.checksum);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(checksum);
        if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() {
            return Err("inadmissible FrankenSciPy batch timing".to_string());
        }
        Ok(Timing {
            elapsed,
            active_tasks: 0,
            max_os_tasks: 0,
        })
    }

    fn time_serial(starts: &[Vec<f64>], repetitions: usize) -> Result<Timing, String> {
        let mut checksum = 0.0;
        let started = Instant::now();
        for _ in 0..repetitions {
            checksum = black_box(checksum + run_serial(black_box(starts))?.checksum);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(checksum);
        if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() {
            return Err("inadmissible FrankenSciPy serial timing".to_string());
        }
        Ok(Timing {
            elapsed,
            active_tasks: 1,
            max_os_tasks: 1,
        })
    }

    fn parse_summary(value: &str, label: &str) -> Result<JobSummary, String> {
        let fields = value.split(',').collect::<Vec<_>>();
        if fields.len() != 8 {
            return Err(format!(
                "{label} has {} fields, expected eight",
                fields.len()
            ));
        }
        Ok(JobSummary {
            best_fun: parse(fields[0], "best objective")?,
            best_error: parse(fields[1], "best point error")?,
            success: parse(fields[2], "success count")?,
            global_count: parse(fields[3], "global count")?,
            nfev: parse(fields[4], "function evaluations")?,
            njev: parse(fields[5], "gradient evaluations")?,
            nit: parse(fields[6], "iterations")?,
            checksum: parse(fields[7], "summary checksum")?,
        })
    }

    fn basic_integrity(label: &str, summary: JobSummary) -> Result<(), String> {
        if !summary.best_fun.is_finite()
            || !summary.best_error.is_finite()
            || !summary.checksum.is_finite()
            || summary.success > BATCH
            || summary.global_count > BATCH
        {
            return Err(format!("{label} failed result integrity: {summary:?}"));
        }
        Ok(())
    }

    fn basic_quality(label: &str, summary: JobSummary) -> Result<(), String> {
        basic_integrity(label, summary)?;
        if summary.best_fun > BEST_OBJECTIVE_LIMIT || summary.best_error > BEST_POINT_ERROR_LIMIT {
            return Err(format!("{label} failed selected-arm quality: {summary:?}"));
        }
        Ok(())
    }

    fn prove_quality(ours: JobSummary, scipy: &ScipyCheck) -> Result<(), String> {
        basic_quality("FrankenSciPy", ours)?;
        basic_quality(&format!("SciPy {}", scipy.arm), scipy.summary)?;
        let minimum_success = (QUALITY_FRACTION * scipy.summary.success as f64).ceil() as usize;
        let minimum_global = (QUALITY_FRACTION * scipy.summary.global_count as f64).ceil() as usize;
        if ours.success < minimum_success || ours.global_count < minimum_global {
            return Err(format!(
                "FrankenSciPy quality shortfall: success={}/{} required>={} \
                 global={}/{} required>={}",
                ours.success,
                scipy.summary.success,
                minimum_success,
                ours.global_count,
                scipy.summary.global_count,
                minimum_global
            ));
        }
        Ok(())
    }

    fn print_summary(label: &str, summary: JobSummary) {
        println!(
            "{label}: best_fun={:.17e} best_point_max_abs_error={:.17e} \
             success={}/{} global_fun_le_1e-8={}/{} total_nfev={} \
             total_njev={} total_nit={} checksum={:.17e}",
            summary.best_fun,
            summary.best_error,
            summary.success,
            BATCH,
            summary.global_count,
            BATCH,
            summary.nfev,
            summary.njev,
            summary.nit,
            summary.checksum
        );
    }

    fn observe_batch_workers(starts: &[Vec<f64>], expected: usize) -> Result<usize, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let maximum = Arc::new(Mutex::new(0usize));
        let (ready_tx, ready_rx) = mpsc::sync_channel(0);
        let sampler_stop = Arc::clone(&stop);
        let sampler_maximum = Arc::clone(&maximum);
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
                        let mut workers = 0usize;
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
                            workers += 1;
                        }
                        if let Ok(mut observed) = sampler_maximum.lock() {
                            *observed = (*observed).max(workers);
                        }
                    }
                    std::thread::sleep(Duration::from_micros(50));
                }
            })
            .map_err(|error| format!("spawn worker sampler: {error}"))?;
        ready_rx
            .recv()
            .map_err(|error| format!("start worker sampler: {error}"))?;
        for _ in 0..3 {
            black_box(run_batch(starts)?);
            let observed = *maximum
                .lock()
                .map_err(|_| "worker observation lock poisoned".to_string())?;
            if observed >= expected {
                break;
            }
        }
        stop.store(true, Ordering::Release);
        sampler
            .join()
            .map_err(|_| "worker sampler panicked".to_string())?;
        let observed = *maximum
            .lock()
            .map_err(|_| "worker observation lock poisoned".to_string())?;
        if observed == 0 {
            return Err("no FrankenSciPy worker task was directly observed".to_string());
        }
        Ok(observed)
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
    }

    fn effect_pair(
        scipy: &mut Scipy,
        arm: &str,
        starts: &[Vec<f64>],
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_batch(starts, repetitions)?.elapsed;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_batch(starts, repetitions)?.elapsed;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        }
    }

    fn ours_null_pair(
        starts: &[Vec<f64>],
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_batch(starts, repetitions)?.elapsed,
                time_batch(starts, repetitions)?.elapsed,
            )
        } else {
            let right = time_batch(starts, repetitions)?.elapsed;
            let left = time_batch(starts, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        arm: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, usize, usize), String> {
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
        ))
    }

    fn measure(
        scipy: &mut Scipy,
        arm: &str,
        starts: &[Vec<f64>],
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(scipy, arm, starts, repetitions, warmup)?;
        }
        let mut measurement = Measurement {
            ours: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            ratios: Vec::with_capacity(rounds),
            ours_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
            max_scipy_active_tasks: 0,
            max_scipy_os_tasks: 0,
        };
        for round in 0..rounds {
            let (ours, incumbent, effect_active, effect_tasks, ours_null, scipy_null) =
                match round % 3 {
                    0 => {
                        let effect = effect_pair(scipy, arm, starts, repetitions, round)?;
                        let ours_null = ours_null_pair(starts, repetitions, round)?;
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        (
                            effect.0,
                            effect.1,
                            effect.2.max(scipy_null.1),
                            effect.3.max(scipy_null.2),
                            ours_null,
                            scipy_null.0,
                        )
                    }
                    1 => {
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        let effect = effect_pair(scipy, arm, starts, repetitions, round)?;
                        let ours_null = ours_null_pair(starts, repetitions, round)?;
                        (
                            effect.0,
                            effect.1,
                            effect.2.max(scipy_null.1),
                            effect.3.max(scipy_null.2),
                            ours_null,
                            scipy_null.0,
                        )
                    }
                    _ => {
                        let ours_null = ours_null_pair(starts, repetitions, round)?;
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        let effect = effect_pair(scipy, arm, starts, repetitions, round)?;
                        (
                            effect.0,
                            effect.1,
                            effect.2.max(scipy_null.1),
                            effect.3.max(scipy_null.2),
                            ours_null,
                            scipy_null.0,
                        )
                    }
                };
            measurement.ours.push(ours);
            measurement.scipy.push(incumbent);
            measurement.ratios.push(incumbent / ours);
            measurement.ours_nulls.push(ours_null);
            measurement.scipy_nulls.push(scipy_null);
            measurement.max_scipy_active_tasks =
                measurement.max_scipy_active_tasks.max(effect_active);
            measurement.max_scipy_os_tasks = measurement.max_scipy_os_tasks.max(effect_tasks);
            println!(
                "round={round} ours_seconds={ours:.9} scipy_seconds={incumbent:.9} \
                 ratio={:.9} ours_null={ours_null:.9} scipy_null={scipy_null:.9} \
                 observed_scipy_active_tasks={effect_active} \
                 observed_scipy_os_tasks={effect_tasks}",
                incumbent / ours
            );
        }
        Ok(measurement)
    }

    struct Mechanism {
        compiled_factors: Vec<f64>,
        batch_factors: Vec<f64>,
        products: Vec<f64>,
    }

    fn measure_mechanism(
        scipy: &mut Scipy,
        arm: &str,
        starts: &[Vec<f64>],
        repetitions: usize,
    ) -> Result<Mechanism, String> {
        let mut mechanism = Mechanism {
            compiled_factors: Vec::with_capacity(DIAGNOSTIC_ROUNDS),
            batch_factors: Vec::with_capacity(DIAGNOSTIC_ROUNDS),
            products: Vec::with_capacity(DIAGNOSTIC_ROUNDS),
        };
        for round in 0..DIAGNOSTIC_ROUNDS {
            let (batch, serial, incumbent) = match round % 3 {
                0 => (
                    time_batch(starts, repetitions)?.elapsed,
                    time_serial(starts, repetitions)?.elapsed,
                    scipy.time(arm, repetitions)?.elapsed,
                ),
                1 => {
                    let incumbent = scipy.time(arm, repetitions)?.elapsed;
                    let batch = time_batch(starts, repetitions)?.elapsed;
                    let serial = time_serial(starts, repetitions)?.elapsed;
                    (batch, serial, incumbent)
                }
                _ => {
                    let serial = time_serial(starts, repetitions)?.elapsed;
                    let incumbent = scipy.time(arm, repetitions)?.elapsed;
                    let batch = time_batch(starts, repetitions)?.elapsed;
                    (batch, serial, incumbent)
                }
            };
            let compiled = incumbent / serial;
            let batching = serial / batch;
            mechanism.compiled_factors.push(compiled);
            mechanism.batch_factors.push(batching);
            mechanism.products.push(compiled * batching);
            println!(
                "mechanism_round={round} batch_seconds={batch:.9} \
                 serial_seconds={serial:.9} scipy_seconds={incumbent:.9} \
                 scipy_over_serial={compiled:.9} \
                 serial_over_batch={batching:.9} product={:.9}",
                compiled * batching
            );
        }
        Ok(mechanism)
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
        if values.is_empty() {
            return f64::NAN;
        }
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut medians = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut sample = Vec::with_capacity(values.len());
            for _ in 0..values.len() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                sample.push(values[(state as usize) % values.len()]);
            }
            medians.push(median(sample));
        }
        medians.sort_by(f64::total_cmp);
        (medians[250], medians[9_750])
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

    fn print_decision(
        arm: &str,
        measurement: &Measurement,
        repetitions: usize,
    ) -> (&'static str, f64, f64, f64) {
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
             provenance_only=true",
            coefficient_of_variation(&measurement.ours_nulls) * 100.0
        );
        println!(
            "NULL-scipy A/A median={scipy_null_median:.9} \
             ci95=[{scipy_null_low:.9},{scipy_null_high:.9}] cv={:.3}% \
             provenance_only=true",
            coefficient_of_variation(&measurement.scipy_nulls) * 100.0
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
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1.0e-12))
            .max(1.0 / scipy_null_low.max(1.0e-12))
            .max(1.0);
        let c1 = ratio_low > 1.0 || ratio_high < 1.0;
        let effect_deviation = if ratio_low > 1.0 {
            ratio_low - 1.0
        } else if ratio_high < 1.0 {
            1.0 - ratio_high
        } else {
            0.0
        };
        let c2 = effect_deviation > 2.0 * null_half_width;
        let c2b = effect_deviation > 2.0 * (null_edge - 1.0);
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
             c2_beats_2x_half_width={c2} c2b_beats_2x_endpoint={c2b} \
             c3_null_medians_within_2pct={c3} decidable={decidable} \
             effect_deviation={effect_deviation:.9} \
             null_half_width={null_half_width:.9} \
             required_c2={:.9} required_c2b={:.9} \
             ours_null_median={ours_null_median:.9} \
             scipy_null_median={scipy_null_median:.9} \
             null_ci_veto=disabled_telemetry_only outcome={outcome}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        (outcome, ratio_median, ratio_low, ratio_high)
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
            let Some(suffix) = label.strip_prefix("cpu") else {
                continue;
            };
            if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let cpu: usize = parse(suffix, "CPU index")?;
            let ticks = fields
                .map(|field| parse::<u64>(field, "CPU tick"))
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("CPU {cpu} has an incomplete /proc/stat row"));
            }
            cpus.insert(
                cpu,
                CpuTicks {
                    total: ticks.iter().sum(),
                    idle: ticks[3].saturating_add(ticks[4]),
                },
            );
        }
        if cpus.is_empty() {
            return Err("/proc/stat exposed no per-CPU rows".to_string());
        }
        Ok(cpus)
    }

    fn require_host_wide_quiescence(phase: &str) -> Result<(), String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        if before.len() != after.len() {
            return Err("CPU topology changed during host load sample".to_string());
        }
        let mut maximum_busy_fraction = 0.0f64;
        let mut busy = Vec::new();
        for (cpu, first) in &before {
            let second = after
                .get(cpu)
                .ok_or_else(|| format!("CPU {cpu} disappeared during load sample"))?;
            let total = second.total.saturating_sub(first.total);
            let idle = second.idle.saturating_sub(first.idle);
            if total == 0 {
                return Err(format!("CPU {cpu} accumulated no load-sample ticks"));
            }
            let busy_fraction = 1.0 - idle as f64 / total as f64;
            maximum_busy_fraction = maximum_busy_fraction.max(busy_fraction);
            if busy_fraction > HOST_QUIESCENCE_MAX_BUSY {
                busy.push((cpu, busy_fraction));
            }
        }
        if !busy.is_empty() {
            let detail = busy
                .iter()
                .map(|(cpu, fraction)| format!("{cpu}:{:.1}%", fraction * 100.0))
                .collect::<Vec<_>>()
                .join(",");
            return Err(format!(
                "host-wide quiescence {phase} failed: {} CPUs exceeded {:.0}% \
                 busy (maximum {:.1}%): {detail}",
                busy.len(),
                HOST_QUIESCENCE_MAX_BUSY * 100.0,
                maximum_busy_fraction * 100.0
            ));
        }
        println!(
            "host_wide_quiescence_{phase}=clear sampled_cpus={} \
             maximum_busy_fraction={maximum_busy_fraction:.3} \
             busy_cpu_count_above_limit=0 limit={HOST_QUIESCENCE_MAX_BUSY:.3}",
            before.len()
        );
        Ok(())
    }

    fn cpu_affinity() -> Result<String, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?;
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_string())
    }

    fn affinity_cpus(affinity: &str) -> Result<Vec<usize>, String> {
        let mut cpus = Vec::new();
        for segment in affinity.split(',') {
            if let Some((first, last)) = segment.split_once('-') {
                let first: usize = parse(first, "affinity range start")?;
                let last: usize = parse(last, "affinity range end")?;
                if last < first {
                    return Err(format!("invalid affinity range {segment}"));
                }
                cpus.extend(first..=last);
            } else {
                cpus.push(parse(segment, "affinity CPU")?);
            }
        }
        cpus.sort_unstable();
        cpus.dedup();
        if cpus.is_empty() {
            return Err("CPU affinity contains no CPUs".to_string());
        }
        Ok(cpus)
    }

    fn host_identity() -> Result<String, String> {
        std::fs::read_to_string("/etc/hostname")
            .map_err(|error| format!("read /etc/hostname: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn boot_id() -> Result<String, String> {
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map_err(|error| format!("read boot ID: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn cpu_topology() -> Result<(usize, usize, String), String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let mut pairs = HashSet::new();
        let mut logical = 0usize;
        let mut model = None;
        for block in cpuinfo.split("\n\n") {
            let mut physical = None;
            let mut core = None;
            let mut processor = None;
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("physical id") {
                    physical = value.split(':').nth(1).map(str::trim);
                } else if let Some(value) = line.strip_prefix("core id") {
                    core = value.split(':').nth(1).map(str::trim);
                } else if let Some(value) = line.strip_prefix("processor") {
                    processor = value.split(':').nth(1).map(str::trim);
                } else if let Some(value) = line.strip_prefix("model name") {
                    model = value
                        .split_once(':')
                        .map(|(_, text)| text.trim().to_string());
                }
            }
            if let Some(processor_id) = processor {
                logical += 1;
                pairs.insert((
                    physical.unwrap_or("0").to_string(),
                    core.unwrap_or(processor_id).to_string(),
                ));
            }
        }
        if logical == 0 || pairs.is_empty() {
            return Err("could not derive CPU topology".to_string());
        }
        Ok((
            pairs.len(),
            logical,
            model.unwrap_or_else(|| "unknown".to_string()),
        ))
    }

    fn ram_bytes() -> Result<u64, String> {
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|value| value.split_whitespace().next())
            .ok_or_else(|| "MemTotal missing from /proc/meminfo".to_string())?;
        parse::<u64>(kib, "MemTotal").map(|value| value * 1024)
    }

    fn numa_node_count() -> Result<usize, String> {
        let nodes = std::fs::read_dir("/sys/devices/system/node")
            .map_err(|error| format!("read NUMA topology: {error}"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let text = name.to_string_lossy();
                text.strip_prefix("node")
                    .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .count();
        if nodes == 0 {
            return Err("NUMA topology exposed zero nodes".to_string());
        }
        Ok(nodes)
    }

    fn runtime_isa_features() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            format!(
                "sse2={},sse4_2={},avx2={},fma={},bmi2={},vaes={},avx512f={}",
                std::is_x86_feature_detected!("sse2"),
                std::is_x86_feature_detected!("sse4.2"),
                std::is_x86_feature_detected!("avx2"),
                std::is_x86_feature_detected!("fma"),
                std::is_x86_feature_detected!("bmi2"),
                std::is_x86_feature_detected!("vaes"),
                std::is_x86_feature_detected!("avx512f")
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::env::consts::ARCH.to_string()
        }
    }

    fn read_policy_field(cpu: usize, name: &str) -> Result<String, String> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}");
        std::fs::read_to_string(&path)
            .map_err(|error| format!("read {path}: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn print_hardware_provenance(affinity: &str, cpus: &[usize]) -> Result<(), String> {
        let (physical_cores, logical_threads, model) = cpu_topology()?;
        let mut drivers = HashSet::new();
        let mut governors = HashSet::new();
        let mut preferences = HashSet::new();
        let mut minimums = Vec::new();
        let mut maximums = Vec::new();
        for &cpu in cpus {
            drivers.insert(read_policy_field(cpu, "scaling_driver")?);
            governors.insert(read_policy_field(cpu, "scaling_governor")?);
            preferences.insert(
                read_policy_field(cpu, "energy_performance_preference")
                    .unwrap_or_else(|_| "unavailable".to_string()),
            );
            minimums.push(read_policy_field(cpu, "scaling_min_freq")?);
            maximums.push(read_policy_field(cpu, "scaling_max_freq")?);
        }
        let cpuset_effective = std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective")
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|_| "unavailable".to_string());
        println!(
            "hardware_provenance: host_identity={} boot_id={} \
             cpu_model={:?} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={} numa_nodes={} \
             runtime_detected_isa={} affinity={affinity} affinity_cpu_count={} \
             cgroup_cpuset_effective={cpuset_effective}",
            host_identity()?,
            boot_id()?,
            model,
            ram_bytes()?,
            numa_node_count()?,
            runtime_isa_features(),
            cpus.len()
        );
        println!(
            "cpu_frequency_policy: scaling_drivers={drivers:?} \
             scaling_governors={governors:?} \
             energy_performance_preferences={preferences:?} \
             scaling_min_freq_khz_range={:?} scaling_max_freq_khz_range={:?}",
            minimums.iter().min(),
            maximums.iter().max()
        );
        if governors.len() != 1 || !governors.contains("performance") {
            return Err(format!(
                "every affinity CPU must use the performance governor, found {governors:?}"
            ));
        }
        Ok(())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn sha256_of_self() -> Result<String, String> {
        let executable =
            std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
        sha256_file(&executable)
    }

    fn ready_value<'a>(identity: &'a str, prefix: &str) -> Option<&'a str> {
        identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix(prefix))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("{name} is required for benchmark provenance"))
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

    fn validate_provenance(smoke: bool) -> Result<(String, String), String> {
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        if source_commit.len() != 40 || !source_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("BINARY_SOURCE_COMMIT must be a full 40-hex commit".to_string());
        }
        for token in ["rch", "base", "clean-overlay", "no-overlay"] {
            if !build_route.contains(token) {
                return Err(format!("BINARY_BUILD_ROUTE is missing {token:?}"));
            }
        }
        let booking_claim = if smoke {
            "smoke-only".to_string()
        } else {
            let value = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
            let numeric: u64 = parse(&value, "numeric trj booking claim")?;
            if numeric == 0 {
                return Err("trj booking claim must be positive".to_string());
            }
            value
        };
        println!(
            "binary_provenance: source_commit={source_commit} \
             builder_identity={builder_identity} build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");
        Ok((source_commit, booking_claim))
    }

    pub fn run() -> Result<(), String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        let smoke = arguments.first().is_some_and(|value| value == "--smoke");
        let numeric = if smoke {
            &arguments[1..]
        } else {
            &arguments[..]
        };
        let rounds = numeric
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(DEFAULT_ROUNDS);
        let repetitions = numeric
            .get(1)
            .map(|value| parse::<usize>(value, "repetitions"))
            .transpose()?
            .unwrap_or(DEFAULT_REPETITIONS);
        if rounds < 7 || repetitions == 0 {
            return Err("require rounds>=7 and repetitions>0".to_string());
        }

        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        let _ = validate_provenance(smoke)?;
        println!(
            "python_oracle_protocol_sha256={:x}",
            Sha256::digest(PYTHON_ORACLE.as_bytes())
        );

        let affinity = cpu_affinity()?;
        let cpus = affinity_cpus(&affinity)?;
        if !smoke {
            if host_identity()? != "threadripperje" {
                return Err("evidence run is restricted to threadripperje".to_string());
            }
            if cpus.len() != 32 {
                return Err(format!(
                    "evidence run requires exactly 32 affinity CPUs, found {}",
                    cpus.len()
                ));
            }
            print_hardware_provenance(&affinity, &cpus)?;
            require_host_wide_quiescence("pre")?;
        } else {
            println!(
                "smoke_hardware: host_identity={} affinity={affinity} \
                 affinity_cpu_count={} governor_gate_skipped_for_non_evidence_smoke=true",
                host_identity()?,
                cpus.len()
            );
        }

        let starts = starts();
        let starts_sha = starts_sha256(&starts);
        let ours_summary = run_batch(&starts)?;
        basic_quality("FrankenSciPy", ours_summary)?;
        print_summary("frankenscipy_quality", ours_summary);
        println!(
            "fixture=rosenbrock-6d-multistart batch={BATCH} \
             starts_domain=[-2,2]^6 starts_seed=7 starts_sha256={starts_sha} \
             method=BFGS analytic_gradient_both_arms=true tol=1e-8 maxiter=500 \
             rounds={rounds} repetitions_per_sample={repetitions}"
        );
        println!(
            "whole_job_boundary: INCLUDED=solve_all_128_fixed_starts,\
             retain_all_results,select_best,materialize_quality_and_work_counters; \
             EXCLUDED=start_generation,python_startup,scipy_import,\
             incumbent_screen,quality_serialization,pipe_transport,bootstrap"
        );

        let (mut scipy, identity) = Scipy::start()?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains("minimize_module=scipy.optimize._minimize")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine identity gate".to_string());
        }
        let scipy_engine_sha256 = ready_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "SciPy arm omitted engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("SciPy arm reported an invalid engine SHA-256".to_string());
        }
        let scipy_starts_sha = ready_value(&identity, "starts_sha256=")
            .ok_or_else(|| "SciPy arm omitted starts SHA-256".to_string())?;
        if scipy_starts_sha != starts_sha {
            return Err(format!(
                "input identity mismatch: FrankenSciPy={starts_sha} SciPy={scipy_starts_sha}"
            ));
        }
        let scipy_worker_capacity: usize = parse(
            ready_value(&identity, "worker_capacity=")
                .ok_or_else(|| "SciPy arm omitted worker capacity".to_string())?,
            "SciPy worker capacity",
        )?;
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "incumbent_screen_contract: public_arms={} \
             selection=lowest_five-sample-median_whole-job-wall \
             screen_samples_excluded_from_effect=true",
            SCIPY_ARMS.join(",")
        );

        let mut checks = Vec::with_capacity(SCIPY_ARMS.len());
        for arm in SCIPY_ARMS {
            let check = scipy.check(arm)?;
            basic_integrity(&format!("SciPy {arm}"), check.summary)?;
            print_summary(&format!("scipy_quality_{arm}"), check.summary);
            println!(
                "scipy_observation_{arm}: actual_observed_active_tasks={} \
                 max_observed_os_tasks={} workers_map_capacity={scipy_worker_capacity}",
                check.active_tasks, check.max_os_tasks
            );
            checks.push(check);
        }

        if smoke {
            scipy.stop()?;
            println!(
                "SMOKE COMPLETE: no evidence timing performed elf_sha256={elf_sha256} \
                 scipy_engine_sha256={scipy_engine_sha256}"
            );
            return Ok(());
        }

        let expected_workers = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(BATCH);
        let observed_workers = observe_batch_workers(&starts, expected_workers)?;
        if observed_workers != expected_workers {
            return Err(format!(
                "FrankenSciPy observed worker mismatch: expected from affinity/API \
                 {expected_workers}, directly observed {observed_workers}"
            ));
        }
        println!(
            "frankenscipy_task_provenance: actual_observed_concurrent_solve_workers={observed_workers} \
             available_parallelism={expected_workers} observation=untimed_proc_task_sampler"
        );

        let mut screened = Vec::with_capacity(SCIPY_ARMS.len());
        for arm in SCIPY_ARMS {
            let mut samples = Vec::with_capacity(SCREEN_ROUNDS);
            let mut maximum_active = 0usize;
            let mut maximum_tasks = 0usize;
            for _ in 0..SCREEN_ROUNDS {
                let timing = scipy.time(arm, 1)?;
                samples.push(timing.elapsed);
                maximum_active = maximum_active.max(timing.active_tasks);
                maximum_tasks = maximum_tasks.max(timing.max_os_tasks);
            }
            let screen_median = median(samples.clone());
            println!(
                "scipy_public_arm_screen: arm={arm} eligible=true raw_seconds={} \
                 median_whole_job_ms={:.9} actual_observed_active_tasks={} \
                 max_observed_os_tasks={maximum_tasks}",
                csv(&samples),
                screen_median * 1.0e3,
                maximum_active
            );
            screened.push((arm, screen_median));
        }
        screened.sort_by(|left, right| left.1.total_cmp(&right.1));
        let selected_arm = screened[0].0;
        let selected_check = checks
            .iter()
            .find(|check| check.arm == selected_arm)
            .ok_or_else(|| "selected SciPy arm has no quality record".to_string())?;
        prove_quality(ours_summary, selected_check)?;
        println!(
            "selected_public_scipy_arm={selected_arm} \
             selection_screen_outside_headline_samples=true"
        );
        println!(
            "quality_gate=PASS candidate_success={} incumbent_success={} \
             candidate_global={} incumbent_global={} \
             candidate_best_fun={:.17e} incumbent_best_fun={:.17e} \
             candidate_best_error={:.17e} incumbent_best_error={:.17e}",
            ours_summary.success,
            selected_check.summary.success,
            ours_summary.global_count,
            selected_check.summary.global_count,
            ours_summary.best_fun,
            selected_check.summary.best_fun,
            ours_summary.best_error,
            selected_check.summary.best_error
        );

        let measurement = measure(&mut scipy, selected_arm, &starts, rounds, repetitions)?;
        let (outcome, ratio_median, ratio_low, ratio_high) =
            print_decision(selected_arm, &measurement, repetitions);

        let mechanism = measure_mechanism(&mut scipy, selected_arm, &starts, repetitions)?;
        let compiled_median = median(mechanism.compiled_factors.clone());
        let batch_median = median(mechanism.batch_factors.clone());
        let product_median = compiled_median * batch_median;
        let agreement_factor = (product_median / ratio_median)
            .max(ratio_median / product_median)
            .max(1.0);
        let mechanism_confirmed =
            compiled_median > 2.0 && batch_median > 4.0 && agreement_factor <= 1.5;
        println!(
            "mechanism_summary: scipy_over_franken_serial_median={compiled_median:.9}x \
             franken_serial_over_batch_median={batch_median:.9}x \
             factor_product={product_median:.9}x primary_ratio={ratio_median:.9}x \
             product_agreement_factor={agreement_factor:.9} \
             compiled_factor_gt_2={} batch_factor_gt_4={} \
             agreement_within_1_5={} mechanism_confirmed={mechanism_confirmed} \
             compiled_raw={} batch_raw={} product_raw={}",
            compiled_median > 2.0,
            batch_median > 4.0,
            agreement_factor <= 1.5,
            csv(&mechanism.compiled_factors),
            csv(&mechanism.batch_factors),
            csv(&mechanism.products)
        );

        let p1 = matches!(selected_arm, "analytic" | "fused");
        let p2 = ratio_high < COLLAPSE_BOUNDARY;
        let p3 = outcome == "DECIDED FRANKENSCIPY WIN" && ratio_low > DURABLE_WIN_BOUNDARY;
        let p5_rust = observed_workers == 32;
        let p5_scipy = if matches!(selected_arm, "analytic" | "fused") {
            measurement.max_scipy_active_tasks == 1
        } else {
            true
        };
        println!(
            "preregistered_predictions: P1_analytic_or_fused_fastest={p1} \
             P2_old_ratio_collapses_2x={p2} collapse_boundary={COLLAPSE_BOUNDARY:.3} \
             P3_durable_ci_low_gt_3={p3} \
             P4_two_factor_mechanism={mechanism_confirmed} \
             P5_rust_observed_32={p5_rust} \
             P5_scipy_observation_matches_arm={p5_scipy}"
        );
        println!(
            "thread_provenance: actual_observed_frankenscipy_solve_workers={observed_workers} \
             actual_observed_scipy_active_tasks={} \
             actual_observed_scipy_max_os_tasks={} \
             scipy_workers_map_capacity={scipy_worker_capacity}",
            measurement.max_scipy_active_tasks, measurement.max_scipy_os_tasks
        );

        let chooser = if p3 {
            "choose FrankenSciPy minimize_many for this exact 128-start analytic-gradient job"
        } else if outcome == "DECIDED FRANKENSCIPY LOSS" {
            "choose the selected public SciPy arm for this exact multistart job"
        } else {
            "retain SciPy as the default; no durable whole-job speed claim is authorized"
        };
        println!("CHOOSER STATEMENT: {chooser}.");

        scipy.stop()?;
        require_host_wide_quiescence("post")?;
        println!(
            "FINAL: outcome={outcome} selected_public_scipy_arm={selected_arm} \
             ratio={ratio_median:.9} ci95=[{ratio_low:.9},{ratio_high:.9}] \
             actual_observed_frankenscipy_workers={observed_workers} \
             actual_observed_scipy_active_tasks={} elf_sha256={elf_sha256} \
             scipy_engine_sha256={scipy_engine_sha256}",
            measurement.max_scipy_active_tasks
        );
        Ok(())
    }
}

#[cfg(feature = "opt-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "opt-incumbent-bench"))]
fn main() {
    eprintln!("perf_minimize_many_scipy requires --features opt-incumbent-bench");
    std::process::exit(2);
}
