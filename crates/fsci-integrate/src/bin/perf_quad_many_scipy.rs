//! Whole 2,048-member damped-transform study versus screened SciPy 1.17.1.
//!
//! The historical `quad_many` row compared against a scalar Python loop.
//! This harness screens public vector and persistent-pool routes, freezes the
//! strongest valid incumbent, and applies the corrected dual-null median gate.

#[cfg(feature = "quad-incumbent-bench")]
mod bench {
    use fsci_integrate::{QuadOptions, quad_many};
    use fsci_runtime::scipy_incumbent::{PINNED_NUMPY, PINNED_SCIPY, ScipyIncumbent};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
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

    const BATCH: usize = 2_048;
    const EPSABS: f64 = 1.49e-8;
    const EPSREL: f64 = 1.49e-8;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 3;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const MAX_SCALED_ERROR: f64 = 4.0;
    const COLLAPSE_BOUNDARY: f64 = 12.2;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SCIPY_ARMS: [&str; 7] = [
        "quad_scalar",
        "quad_thread",
        "quad_process",
        "quad_vec",
        "quad_vec_process",
        "cubature_gk21",
        "cubature_gk21_process",
    ];

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
import scipy
from scipy import integrate
from scipy.integrate import _cubature, _quad_vec, _quadpack_py

BATCH = 2048
EPSABS = 1.49e-8
EPSREL = 1.49e-8
MASK64 = (1 << 64) - 1

def build_inputs():
    state = 7
    rows = []
    for _ in range(BATCH):
        state = (state * 6364136223846793005 + 1) & MASK64
        up = (state >> 11) / float(1 << 53)
        state = (state * 6364136223846793005 + 1) & MASK64
        uw = (state >> 11) / float(1 << 53)
        rows.append((2.0 + 48.0 * up, 1.0 + 34.0 * uw))
    params = np.asarray(rows, dtype="<f8")
    reference = [
        (
            p + math.exp(-p) * (-p * math.cos(w) + w * math.sin(w))
        ) / (p * p + w * w)
        for p, w in rows
    ]
    return params, np.asarray(reference, dtype="<f8")

PARAMS, REFERENCE = build_inputs()
PARAMS_SHA256 = hashlib.sha256(PARAMS.tobytes(order="C")).hexdigest()
REFERENCE_SHA256 = hashlib.sha256(REFERENCE.tobytes(order="C")).hexdigest()
INPUT_SHA256 = hashlib.sha256(
    PARAMS.tobytes(order="C") + REFERENCE.tobytes(order="C")
).hexdigest()
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))

ACTIVE_THREADS = set()
ACTIVE_LOCK = threading.Lock()
RETURNED_PIDS = set()

def reset_observations():
    global ACTIVE_THREADS, RETURNED_PIDS
    with ACTIVE_LOCK:
        ACTIVE_THREADS = set()
    RETURNED_PIDS = set()

def note_thread():
    with ACTIVE_LOCK:
        ACTIVE_THREADS.add(threading.get_native_id())

def scalar_one(pair):
    note_thread()
    p = float(pair[0])
    w = float(pair[1])
    result = integrate.quad(
        lambda x: math.exp(-p * x) * math.cos(w * x),
        0.0,
        1.0,
        epsabs=EPSABS,
        epsrel=EPSREL,
        limit=50,
        full_output=1,
    )
    value, error, info = result[:3]
    return (
        os.getpid(),
        threading.get_native_id(),
        float(value),
        float(error),
        len(result) == 3,
        int(info.get("neval", 0)),
    )

def warm_pid(_):
    return os.getpid()

def tracked_apply(payload):
    function, item = payload
    return os.getpid(), function(item)

def tracking_map(function, iterable):
    tagged = PROCESS_POOL.map(
        tracked_apply,
        [(function, item) for item in iterable],
    )
    for pid, _value in tagged:
        RETURNED_PIDS.add(int(pid))
    return [value for _pid, value in tagged]

def scalar_job(mode):
    if mode == "scalar":
        rows = [scalar_one(pair) for pair in PARAMS]
    elif mode == "thread":
        rows = THREAD_POOL.map(scalar_one, list(PARAMS))
    elif mode == "process":
        rows = PROCESS_POOL.map(scalar_one, list(PARAMS))
    else:
        raise RuntimeError(f"unknown scalar mode {mode!r}")
    pids = {int(row[0]) for row in rows}
    tids = {int(row[1]) for row in rows}
    RETURNED_PIDS.update(pids)
    values = np.asarray([row[2] for row in rows], dtype=np.float64)
    errors = np.asarray([row[3] for row in rows], dtype=np.float64)
    successes = sum(bool(row[4]) for row in rows)
    work = sum(int(row[5]) for row in rows)
    active = len(tids) if mode == "thread" else 1
    return values, errors, successes, work, active, len(pids)

def vector_integrand(x):
    note_thread()
    return np.exp(-PARAMS[:, 0] * x) * np.cos(PARAMS[:, 1] * x)

def quad_vec_job(workers):
    value, error, info = integrate.quad_vec(
        vector_integrand,
        0.0,
        1.0,
        epsabs=EPSABS,
        epsrel=EPSREL,
        norm="max",
        limit=10000,
        workers=workers,
        full_output=True,
    )
    values = np.asarray(value, dtype=np.float64)
    errors = np.full(BATCH, float(error), dtype=np.float64)
    pids = len(RETURNED_PIDS) if workers != 1 else 1
    active = pids if workers != 1 else 1
    return (
        values,
        errors,
        BATCH if bool(info.success) else 0,
        int(info.neval),
        active,
        pids,
    )

def cubature_integrand(points):
    note_thread()
    x = points[:, 0]
    return (
        np.exp(-x[:, None] * PARAMS[None, :, 0])
        * np.cos(x[:, None] * PARAMS[None, :, 1])
    )

def cubature_job(workers):
    result = integrate.cubature(
        cubature_integrand,
        np.asarray([0.0]),
        np.asarray([1.0]),
        rule="gk21",
        rtol=EPSREL,
        atol=EPSABS,
        workers=workers,
    )
    values = np.asarray(result.estimate, dtype=np.float64)
    errors = np.broadcast_to(
        np.asarray(result.error, dtype=np.float64),
        (BATCH,),
    ).copy()
    pids = len(RETURNED_PIDS) if workers != 1 else 1
    active = pids if workers != 1 else 1
    return (
        values,
        errors,
        BATCH if result.status == "converged" else 0,
        int(result.subdivisions),
        active,
        pids,
    )

def execute(arm):
    reset_observations()
    if arm == "quad_scalar":
        return scalar_job("scalar")
    if arm == "quad_thread":
        return scalar_job("thread")
    if arm == "quad_process":
        return scalar_job("process")
    if arm == "quad_vec":
        return quad_vec_job(1)
    if arm == "quad_vec_process":
        return quad_vec_job(tracking_map)
    if arm == "cubature_gk21":
        return cubature_job(1)
    if arm == "cubature_gk21_process":
        return cubature_job(tracking_map)
    raise RuntimeError(f"unknown arm {arm!r}")

def validate_raw(raw):
    values, errors, successes, work, active, worker_processes = raw
    if values.shape != (BATCH,) or errors.shape != (BATCH,):
        raise RuntimeError(
            f"unexpected shapes values={values.shape} errors={errors.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite integral value")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise RuntimeError("invalid error estimate")
    if not 0 <= successes <= BATCH:
        raise RuntimeError("invalid success count")
    if work < 0 or active < 1 or worker_processes < 1:
        raise RuntimeError("invalid work or worker observation")
    return (
        values,
        errors,
        int(successes),
        int(work),
        int(active),
        int(worker_processes),
    )

def whole_job_checksum(raw):
    values, errors, successes, work, _active, _processes = validate_raw(raw)
    weights = np.arange(1, BATCH + 1, dtype=np.float64)
    best_index = int(np.argmax(np.abs(values)))
    return float(
        np.dot(values, weights)
        + np.dot(errors, weights) * 1.0e-3
        + best_index * 1.0e-6
        + successes
        + work * 1.0e-9
    )

def quality_fields(raw):
    values, errors, successes, work, active, processes = validate_raw(raw)
    scale = EPSABS + EPSREL * np.abs(REFERENCE)
    absolute = np.abs(values - REFERENCE)
    best_index = int(np.argmax(np.abs(values)))
    return (
        successes,
        best_index,
        float(PARAMS[best_index, 0]),
        float(PARAMS[best_index, 1]),
        float(values[best_index]),
        float(np.min(errors)),
        float(np.max(errors)),
        float(np.max(absolute)),
        float(np.max(absolute / scale)),
        work,
        float(np.mean(values)),
        float(np.mean(errors)),
        whole_job_checksum(raw),
        active,
        len(os.listdir("/proc/self/task")),
        processes,
        values,
        errors,
    )

def source_sha256():
    paths = {
        inspect.getsourcefile(_quadpack_py),
        inspect.getsourcefile(_quad_vec),
        inspect.getsourcefile(_cubature),
    }
    digest = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as handle:
            payload = handle.read()
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()

# Fork only after every callable that may cross the process boundary exists,
# but still before the parent starts a thread pool.
PROCESS_POOL = multiprocessing.get_context("fork").Pool(WORKER_CAPACITY)
PROCESS_POOL.map(warm_pid, range(WORKER_CAPACITY * 2))
THREAD_POOL = ThreadPool(WORKER_CAPACITY)

SCIPY_ENGINE_SHA256 = source_sha256()
SCIPY_GENUINE = (
    scipy.__version__ == "1.17.1"
    and np.__version__ == "2.4.3"
)
FSCI_LOADED = any(name == "fsci" or name.startswith("fsci_") for name in sys.modules)
print(
    "READY"
    f"|{scipy.__version__}"
    f"|{np.__version__}"
    f"|{SCIPY_ENGINE_SHA256}"
    f"|{PARAMS_SHA256}"
    f"|{REFERENCE_SHA256}"
    f"|{INPUT_SHA256}"
    f"|{WORKER_CAPACITY}"
    f"|{len(os.listdir('/proc/self/task'))}"
    f"|{SCIPY_GENUINE}"
    f"|{FSCI_LOADED}"
    f"|{scipy.__file__}",
    flush=True,
)

SINK = 0.0
for line in sys.stdin:
    try:
        fields = line.strip().split()
        if not fields:
            continue
        command = fields[0]
        if command == "CHECK" and len(fields) == 2:
            arm = fields[1]
            quality = quality_fields(execute(arm))
            values_csv = ",".join(f"{value:.17e}" for value in quality[16])
            errors_csv = ",".join(f"{value:.17e}" for value in quality[17])
            print(
                "CHECK"
                f"|{arm}"
                f"|{quality[0]}"
                f"|{quality[1]}"
                f"|{quality[2]:.17e}"
                f"|{quality[3]:.17e}"
                f"|{quality[4]:.17e}"
                f"|{quality[5]:.17e}"
                f"|{quality[6]:.17e}"
                f"|{quality[7]:.17e}"
                f"|{quality[8]:.17e}"
                f"|{quality[9]}"
                f"|{quality[10]:.17e}"
                f"|{quality[11]:.17e}"
                f"|{quality[12]:.17e}"
                f"|{quality[13]}"
                f"|{quality[14]}"
                f"|{quality[15]}"
                f"|{values_csv}"
                f"|{errors_csv}",
                flush=True,
            )
        elif command == "TIME" and len(fields) == 3:
            arm = fields[1]
            repetitions = int(fields[2])
            if repetitions < 1:
                raise RuntimeError("repetitions must be positive")
            started = time.perf_counter()
            checksum = 0.0
            max_active = 0
            max_tasks = 0
            max_processes = 0
            for _ in range(repetitions):
                raw = execute(arm)
                checksum += whole_job_checksum(raw)
                max_active = max(max_active, int(raw[4]))
                max_tasks = max(max_tasks, len(os.listdir("/proc/self/task")))
                max_processes = max(max_processes, int(raw[5]))
            elapsed = (time.perf_counter() - started) / repetitions
            SINK += checksum
            print(
                "TIME"
                f"|{arm}"
                f"|{elapsed:.17e}"
                f"|{checksum:.17e}"
                f"|{max_active}"
                f"|{max_tasks}"
                f"|{max_processes}",
                flush=True,
            )
        elif command == "STOP":
            PROCESS_POOL.close()
            PROCESS_POOL.join()
            THREAD_POOL.close()
            THREAD_POOL.join()
            print(f"STOPPED|{SINK:.17e}", flush=True)
            break
        else:
            raise RuntimeError(f"invalid command {line!r}")
    except Exception as error:
        print(f"ERROR|{type(error).__name__}|{error}", flush=True)
"#;

    #[derive(Clone, Copy, Debug)]
    struct JobSummary {
        successes: usize,
        best_index: usize,
        best_p: f64,
        best_w: f64,
        best_integral: f64,
        min_error_estimate: f64,
        max_error_estimate: f64,
        max_abs_reference_error: f64,
        max_scaled_reference_error: f64,
        work: usize,
        mean_integral: f64,
        mean_error_estimate: f64,
        checksum: f64,
    }

    struct RawJob {
        values: Vec<f64>,
        errors: Vec<f64>,
        successes: usize,
        work: usize,
    }

    #[derive(Clone, Copy)]
    struct Timing {
        elapsed: f64,
        active_tasks: usize,
        max_os_tasks: usize,
        worker_processes: usize,
    }

    #[derive(Clone)]
    struct ScipyCheck {
        arm: String,
        active_tasks: usize,
        max_os_tasks: usize,
        worker_processes: usize,
        summary: JobSummary,
        values: Vec<f64>,
        errors: Vec<f64>,
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
    const SPAWN_ENV: &[(&str, &str)] = &[("PYTHONNOUSERSITE", "1")];
    /// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
    /// installation whose compiled submodules do not load, and that difference would
    /// otherwise only surface mid-timing.
    const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.integrate"];

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
        fn start() -> Result<(Self, String), String> {
            let incumbent = incumbent()?;
            println!("{}", incumbent.provenance_line());
            let python = incumbent.python.clone();
            let mut child = incumbent
                .command()
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
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
            let reply = reply.trim().to_string();
            if reply.starts_with("ERROR|") {
                return Err(format!("live SciPy {context} failed: {reply}"));
            }
            Ok(reply)
        }

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let reply = self.request(&format!("CHECK {arm}"), "SciPy quality check")?;
            let fields = reply.splitn(20, '|').collect::<Vec<_>>();
            if fields.len() != 20 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("invalid SciPy quality reply: {reply}"));
            }
            let values = fields[18]
                .split(',')
                .map(|field| parse(field, "SciPy integral value"))
                .collect::<Result<Vec<_>, _>>()?;
            let errors = fields[19]
                .split(',')
                .map(|field| parse(field, "SciPy error estimate"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != BATCH || errors.len() != BATCH {
                return Err(format!(
                    "SciPy {arm} returned value/error lengths {}/{} instead of {BATCH}",
                    values.len(),
                    errors.len()
                ));
            }
            Ok(ScipyCheck {
                arm: arm.to_string(),
                summary: JobSummary {
                    successes: parse(fields[2], "SciPy successes")?,
                    best_index: parse(fields[3], "SciPy best index")?,
                    best_p: parse(fields[4], "SciPy best p")?,
                    best_w: parse(fields[5], "SciPy best w")?,
                    best_integral: parse(fields[6], "SciPy best integral")?,
                    min_error_estimate: parse(fields[7], "SciPy minimum error")?,
                    max_error_estimate: parse(fields[8], "SciPy maximum error")?,
                    max_abs_reference_error: parse(fields[9], "SciPy absolute error")?,
                    max_scaled_reference_error: parse(fields[10], "SciPy scaled error")?,
                    work: parse(fields[11], "SciPy work count")?,
                    mean_integral: parse(fields[12], "SciPy mean integral")?,
                    mean_error_estimate: parse(fields[13], "SciPy mean error")?,
                    checksum: parse(fields[14], "SciPy checksum")?,
                },
                active_tasks: parse(fields[15], "SciPy active tasks")?,
                max_os_tasks: parse(fields[16], "SciPy max OS tasks")?,
                worker_processes: parse(fields[17], "SciPy worker processes")?,
                values,
                errors,
            })
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let reply = self.request(
                &format!("TIME {arm} {repetitions}"),
                "SciPy timed transform study",
            )?;
            let fields = reply.split('|').collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "TIME" || fields[1] != arm {
                return Err(format!("invalid SciPy timing reply: {reply}"));
            }
            let elapsed: f64 = parse(fields[2], "SciPy elapsed")?;
            let checksum: f64 = parse(fields[3], "SciPy checksum")?;
            let active_tasks: usize = parse(fields[4], "SciPy active tasks")?;
            let max_os_tasks: usize = parse(fields[5], "SciPy max OS tasks")?;
            let worker_processes: usize = parse(fields[6], "SciPy worker processes")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || !checksum.is_finite()
                || active_tasks == 0
                || max_os_tasks == 0
                || worker_processes == 0
            {
                return Err(format!("inadmissible SciPy timing: {reply}"));
            }
            Ok(Timing {
                elapsed,
                active_tasks,
                max_os_tasks,
                worker_processes,
            })
        }

        fn stop(&mut self) -> Result<(), String> {
            let reply = self.request("STOP", "SciPy shutdown")?;
            if !reply.starts_with("STOPPED|") {
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
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    #[derive(Clone)]
    struct Dataset {
        params: Vec<Vec<f64>>,
        reference: Vec<f64>,
        params_sha256: String,
        reference_sha256: String,
        input_sha256: String,
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

    fn sha256_f64s(values: impl IntoIterator<Item = f64>) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn build_dataset() -> Dataset {
        let mut state = 7u64;
        let mut params = Vec::with_capacity(BATCH);
        let mut reference = Vec::with_capacity(BATCH);
        for _ in 0..BATCH {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let up = (state >> 11) as f64 / (1u64 << 53) as f64;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let uw = (state >> 11) as f64 / (1u64 << 53) as f64;
            let p = 2.0 + 48.0 * up;
            let w = 1.0 + 34.0 * uw;
            params.push(vec![p, w]);
            reference.push((p + (-p).exp() * (-p * w.cos() + w * w.sin())) / (p * p + w * w));
        }
        let params_sha256 = sha256_f64s(params.iter().flatten().copied());
        let reference_sha256 = sha256_f64s(reference.iter().copied());
        let mut digest = Sha256::new();
        for value in params.iter().flatten().chain(reference.iter()) {
            digest.update(value.to_le_bytes());
        }
        let input_sha256 = format!("{:x}", digest.finalize());
        Dataset {
            params,
            reference,
            params_sha256,
            reference_sha256,
            input_sha256,
        }
    }

    fn execute_ours(data: &Dataset) -> Result<RawJob, String> {
        let results = quad_many(
            |x, params| (-params[0] * x).exp() * (params[1] * x).cos(),
            0.0,
            1.0,
            &data.params,
            QuadOptions {
                epsabs: EPSABS,
                epsrel: EPSREL,
                limit: 50,
            },
        );
        if results.len() != BATCH {
            return Err(format!(
                "FrankenSciPy returned {} rows instead of {BATCH}",
                results.len()
            ));
        }
        let mut values = Vec::with_capacity(BATCH);
        let mut errors = Vec::with_capacity(BATCH);
        let mut successes = 0usize;
        let mut work = 0usize;
        for result in results {
            let result =
                result.map_err(|error| format!("FrankenSciPy quad_many member failed: {error}"))?;
            values.push(result.integral);
            errors.push(result.error);
            successes += usize::from(result.converged);
            work = work.saturating_add(result.neval);
        }
        Ok(RawJob {
            values,
            errors,
            successes,
            work,
        })
    }

    fn summarize(raw: &RawJob, data: &Dataset) -> Result<JobSummary, String> {
        if raw.values.len() != BATCH
            || raw.errors.len() != BATCH
            || data.reference.len() != BATCH
            || data.params.len() != BATCH
        {
            return Err("whole-job vector length mismatch".to_string());
        }
        if raw.values.iter().any(|value| !value.is_finite())
            || raw
                .errors
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || raw.successes > BATCH
        {
            return Err("whole-job output integrity failure".to_string());
        }
        let mut best_index = 0usize;
        let mut max_abs_reference_error = 0.0f64;
        let mut max_scaled_reference_error = 0.0f64;
        let mut checksum = 0.0f64;
        for index in 0..BATCH {
            if raw.values[index].abs() > raw.values[best_index].abs() {
                best_index = index;
            }
            let absolute = (raw.values[index] - data.reference[index]).abs();
            let scale = EPSABS + EPSREL * data.reference[index].abs();
            max_abs_reference_error = max_abs_reference_error.max(absolute);
            max_scaled_reference_error = max_scaled_reference_error.max(absolute / scale);
            let weight = (index + 1) as f64;
            checksum += raw.values[index] * weight;
            checksum += raw.errors[index] * weight * 1.0e-3;
        }
        checksum += best_index as f64 * 1.0e-6;
        checksum += raw.successes as f64;
        checksum += raw.work as f64 * 1.0e-9;
        Ok(JobSummary {
            successes: raw.successes,
            best_index,
            best_p: data.params[best_index][0],
            best_w: data.params[best_index][1],
            best_integral: raw.values[best_index],
            min_error_estimate: raw.errors.iter().copied().fold(f64::INFINITY, f64::min),
            max_error_estimate: raw.errors.iter().copied().fold(0.0, f64::max),
            max_abs_reference_error,
            max_scaled_reference_error,
            work: raw.work,
            mean_integral: raw.values.iter().sum::<f64>() / BATCH as f64,
            mean_error_estimate: raw.errors.iter().sum::<f64>() / BATCH as f64,
            checksum,
        })
    }

    fn quality_eligible(summary: JobSummary) -> bool {
        summary.successes == BATCH
            && summary.best_index < BATCH
            && summary.best_p.is_finite()
            && summary.best_w.is_finite()
            && summary.best_integral.is_finite()
            && summary.min_error_estimate.is_finite()
            && summary.min_error_estimate >= 0.0
            && summary.max_error_estimate.is_finite()
            && summary.max_error_estimate >= summary.min_error_estimate
            && summary.max_abs_reference_error.is_finite()
            && summary.max_abs_reference_error >= 0.0
            && summary.max_scaled_reference_error.is_finite()
            && summary.max_scaled_reference_error <= MAX_SCALED_ERROR
            && summary.work > 0
            && summary.mean_integral.is_finite()
            && summary.mean_error_estimate.is_finite()
            && summary.mean_error_estimate >= 0.0
            && summary.checksum.is_finite()
    }

    fn cross_quality(
        ours: &RawJob,
        ours_summary: JobSummary,
        scipy: &ScipyCheck,
        data: &Dataset,
    ) -> Result<f64, String> {
        if !quality_eligible(ours_summary) || !quality_eligible(scipy.summary) {
            return Err(format!(
                "scientific quality gate failed: ours={ours_summary:?} scipy={:?}",
                scipy.summary
            ));
        }
        if scipy.values.len() != BATCH || scipy.errors.len() != BATCH {
            return Err("selected SciPy materialization length mismatch".to_string());
        }
        let mut maximum_scaled_disagreement = 0.0f64;
        for index in 0..BATCH {
            let scale = EPSABS + EPSREL * data.reference[index].abs();
            maximum_scaled_disagreement = maximum_scaled_disagreement
                .max((ours.values[index] - scipy.values[index]).abs() / scale);
        }
        if maximum_scaled_disagreement > MAX_SCALED_ERROR {
            return Err(format!(
                "cross-integral scaled disagreement {maximum_scaled_disagreement:.9} \
                 exceeds {MAX_SCALED_ERROR:.9}"
            ));
        }
        if ours_summary.best_index != scipy.summary.best_index {
            return Err(format!(
                "maximum-absolute-response member disagrees: ours={} scipy={}",
                ours_summary.best_index, scipy.summary.best_index
            ));
        }
        Ok(maximum_scaled_disagreement)
    }

    fn print_summary(label: &str, summary: JobSummary) {
        println!(
            "{label}: successes={}/{} best_index={} best_p={:.17e} best_w={:.17e} \
             best_integral={:.17e} error_estimate_min={:.17e} \
             error_estimate_max={:.17e} mean_integral={:.17e} \
             mean_error_estimate={:.17e} max_abs_reference_error={:.17e} \
             max_scaled_reference_error={:.9} work={} checksum={:.17e}",
            summary.successes,
            BATCH,
            summary.best_index,
            summary.best_p,
            summary.best_w,
            summary.best_integral,
            summary.min_error_estimate,
            summary.max_error_estimate,
            summary.mean_integral,
            summary.mean_error_estimate,
            summary.max_abs_reference_error,
            summary.max_scaled_reference_error,
            summary.work,
            summary.checksum
        );
    }

    fn time_ours(data: &Dataset, repetitions: usize) -> Result<f64, String> {
        let started = Instant::now();
        let mut checksum = 0.0f64;
        for _ in 0..repetitions {
            let raw = execute_ours(data)?;
            checksum += summarize(&raw, data)?.checksum;
        }
        black_box(checksum);
        let elapsed = started.elapsed().as_secs_f64() / repetitions as f64;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err("inadmissible FrankenSciPy timing".to_string());
        }
        Ok(elapsed)
    }

    fn observe_ours_workers(data: &Dataset, expected: usize) -> Result<usize, String> {
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
        for _ in 0..8 {
            black_box(execute_ours(data)?);
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
    struct ScreenedArm {
        check: ScipyCheck,
        median: f64,
        samples: Vec<f64>,
        max_active_tasks: usize,
        max_os_tasks: usize,
        max_worker_processes: usize,
    }

    fn screen_arm(scipy: &mut Scipy, arm: &str) -> Result<Option<ScreenedArm>, String> {
        let check = scipy.check(arm)?;
        print_summary(&format!("screen SciPy {arm}"), check.summary);
        if !quality_eligible(check.summary) {
            println!(
                "screen_excluded arm={arm} reason=scientific_gate \
                 time_samples=0 eligible=false observed_active_tasks={} \
                 observed_os_tasks={} observed_worker_processes={}",
                check.active_tasks, check.max_os_tasks, check.worker_processes
            );
            return Ok(None);
        }
        let mut samples = Vec::with_capacity(SCREEN_ROUNDS);
        let mut max_active_tasks = check.active_tasks;
        let mut max_os_tasks = check.max_os_tasks;
        let mut max_worker_processes = check.worker_processes;
        for round in 0..SCREEN_ROUNDS {
            let timing = scipy.time(arm, 1)?;
            max_active_tasks = max_active_tasks.max(timing.active_tasks);
            max_os_tasks = max_os_tasks.max(timing.max_os_tasks);
            max_worker_processes = max_worker_processes.max(timing.worker_processes);
            samples.push(timing.elapsed);
            println!(
                "screen arm={arm} round={round} seconds={:.9} \
                 observed_active_tasks={} observed_os_tasks={} \
                 observed_worker_processes={}",
                timing.elapsed, timing.active_tasks, timing.max_os_tasks, timing.worker_processes
            );
        }
        let arm_median = median(samples.clone());
        println!(
            "screen_result arm={arm} median_seconds={arm_median:.9} \
             samples={} observed_active_tasks={max_active_tasks} \
             observed_os_tasks={max_os_tasks} \
             observed_worker_processes={max_worker_processes}",
            csv(&samples)
        );
        Ok(Some(ScreenedArm {
            check,
            median: arm_median,
            samples,
            max_active_tasks,
            max_os_tasks,
            max_worker_processes,
        }))
    }

    fn screen_public_arms(scipy: &mut Scipy) -> Result<Vec<ScreenedArm>, String> {
        let mut screened = Vec::with_capacity(SCIPY_ARMS.len());
        for arm in SCIPY_ARMS {
            match screen_arm(scipy, arm) {
                Ok(Some(result)) => screened.push(result),
                Ok(None) => {}
                Err(error) if error.contains("quality check failed: ERROR|") => {
                    println!(
                        "screen_excluded arm={arm} reason=execution_error \
                         detail={error:?} time_samples=0 eligible=false"
                    );
                }
                Err(error) => return Err(error),
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
    }

    fn effect_pair(
        scipy: &mut Scipy,
        arm: &str,
        data: &Dataset,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_ours(data, repetitions)?;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
                incumbent.worker_processes,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_ours(data, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
                incumbent.worker_processes,
            ))
        }
    }

    fn ours_null_pair(data: &Dataset, repetitions: usize, round: usize) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (time_ours(data, repetitions)?, time_ours(data, repetitions)?)
        } else {
            let right = time_ours(data, repetitions)?;
            let left = time_ours(data, repetitions)?;
            (left, right)
        };
        Ok(left / right)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        arm: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, usize, usize, usize), String> {
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
            println!(
                "round={round} ours_seconds={:.9} scipy_seconds={:.9} \
                 ratio={:.9} ours_null={ours_null:.9} scipy_null={:.9} \
                 observed_scipy_active_tasks={effect_active} \
                 observed_scipy_os_tasks={effect_tasks} \
                 observed_scipy_worker_processes={effect_processes}",
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

    struct Decision {
        outcome: &'static str,
        ratio_median: f64,
        ratio_low: f64,
        ratio_high: f64,
        decidable: bool,
    }

    fn print_decision(arm: &str, measurement: &Measurement) -> Decision {
        let ratio_median = median(measurement.ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&measurement.ratios);
        let ours_null_median = median(measurement.ours_nulls.clone());
        let scipy_null_median = median(measurement.scipy_nulls.clone());
        let (ours_null_low, ours_null_high) = bootstrap_median_ci(&measurement.ours_nulls);
        let (scipy_null_low, scipy_null_high) = bootstrap_median_ci(&measurement.scipy_nulls);
        let ours_p50 = median(measurement.ours.clone());
        let scipy_p50 = median(measurement.scipy.clone());
        let ours_p95 = percentile(measurement.ours.clone(), 0.95);
        let scipy_p95 = percentile(measurement.scipy.clone(), 0.95);
        let ours_p99 = percentile(measurement.ours.clone(), 0.99);
        let scipy_p99 = percentile(measurement.scipy.clone(), 0.99);
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
            let mut row = line.split_whitespace();
            let Some(label) = row.next() else {
                continue;
            };
            let Some(suffix) = label.strip_prefix("cpu") else {
                continue;
            };
            if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let cpu: usize = parse(suffix, "CPU index")?;
            let ticks = row
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
    /// interference that actually occurred.
    ///
    /// Measured support for dropping the ABSOLUTE bound: across six
    /// balanced-square runs banked in `docs/NEGATIVE_EVIDENCE.md` the busiest
    /// sample on record, a saturated box at `host_mean_busy=0.988`, produced the
    /// TIGHTEST A/A null of any run, and the quietest at 0.135 produced the
    /// loosest that still passed. A load-DELTA criterion did not reproduce there
    /// either, so none is imposed — the null is the gate.
    fn report_host_wide_quiescence(phase: &str) -> Result<(), String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        if before.len() != after.len() {
            return Err("CPU topology changed during host load sample".to_string());
        }
        let mut maximum_busy_fraction = 0.0f64;
        let mut total_busy_fraction = 0.0f64;
        let mut sampled_cpus = 0u32;
        let mut busy = Vec::new();
        for (cpu, first) in &before {
            let second = after
                .get(cpu)
                .ok_or_else(|| format!("CPU {cpu} disappeared during load sample"))?;
            let total = second.total.saturating_sub(first.total);
            let idle = second.idle.saturating_sub(first.idle);
            if total == 0 {
                continue;
            }
            let busy_fraction = total.saturating_sub(idle) as f64 / total as f64;
            maximum_busy_fraction = maximum_busy_fraction.max(busy_fraction);
            total_busy_fraction += busy_fraction;
            sampled_cpus += 1;
            if busy_fraction > HOST_QUIESCENCE_MAX_BUSY {
                busy.push(format!("{cpu}:{busy_fraction:.3}"));
            }
        }
        if sampled_cpus == 0 {
            return Err("no CPU accumulated load-sample ticks".to_string());
        }
        let host_mean_busy = total_busy_fraction / f64::from(sampled_cpus);
        println!(
            "host_quiescence phase={phase} sample_ms={} threshold={:.3} \
             max_busy_fraction={maximum_busy_fraction:.3} \
             host_mean_busy={host_mean_busy:.3} busy_cpus={}",
            HOST_QUIESCENCE_SAMPLE.as_millis(),
            HOST_QUIESCENCE_MAX_BUSY,
            if busy.is_empty() {
                "none".to_string()
            } else {
                busy.join(",")
            }
        );
        // The token the ledger recognises. `clear` is still the strongest form
        // and is still printed when the host genuinely is quiet; otherwise the
        // row states how busy the box was rather than hiding it, and names the
        // A/A nulls as what it is decided on.
        if busy.is_empty() {
            println!(
                "host_wide_quiescence_{phase}=clear sampled_cpus={sampled_cpus} \
                 host_mean_busy={host_mean_busy:.3} busy_cpu_count_above_limit=0 \
                 limit={HOST_QUIESCENCE_MAX_BUSY:.3}"
            );
        } else {
            println!(
                "host_wide_quiescence_{phase}=NOT_CERTIFIED(host_mean_busy={host_mean_busy:.3}) \
                 sampled_cpus={sampled_cpus} busy_cpu_count_above_limit={} \
                 limit={HOST_QUIESCENCE_MAX_BUSY:.3} gate=same_invocation_A/A_nulls",
                busy.len()
            );
        }
        Ok(())
    }

    fn affinity_cpus() -> Result<Vec<usize>, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?;
        let allowed = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .ok_or_else(|| "missing Cpus_allowed_list".to_string())?;
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
        for cpu in cpus {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor");
            let governor =
                std::fs::read_to_string(&path).map_err(|error| format!("read {path}: {error}"))?;
            let governor = governor.trim().to_string();
            if governor != "performance" {
                return Err(format!("CPU {cpu} governor is {governor}, not performance"));
            }
            rows.push(format!("{cpu}:{governor}"));
        }
        println!("affinity_governors={}", rows.join(","));
        Ok(())
    }

    fn require_runtime_isa() -> Result<String, String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let first_flags = cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("flags"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, flags)| flags.split_whitespace().collect::<Vec<_>>())
            .ok_or_else(|| "cannot find CPU flags".to_string())?;
        for required in ["avx2", "fma"] {
            if !first_flags.contains(&required) {
                return Err(format!("runtime ISA lacks {required}"));
            }
        }
        Ok("avx2+fma".to_string())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn require_hex_sha(value: &str, label: &str) -> Result<(), String> {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("{label} is not a 64-hex SHA-256: {value}"));
        }
        Ok(())
    }

    fn require_commit(value: &str) -> Result<(), String> {
        if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "BINARY_SOURCE_COMMIT is not a full 40-hex commit: {value}"
            ));
        }
        Ok(())
    }

    fn require_booking_claim() -> Result<String, String> {
        let value = std::env::var("TRJ_BOOKING_CLAIM_MESSAGE_ID")
            .map_err(|_| "missing TRJ_BOOKING_CLAIM_MESSAGE_ID".to_string())?;
        let id: u64 = parse(&value, "TRJ booking claim message id")?;
        if id == 0 {
            return Err("TRJ booking claim message id must be non-zero".to_string());
        }
        Ok(value)
    }

    fn require_env(name: &str) -> Result<String, String> {
        let value = std::env::var(name).map_err(|_| format!("missing {name}"))?;
        if value.trim().is_empty() {
            return Err(format!("{name} is empty"));
        }
        Ok(value)
    }

    fn current_hostname() -> Result<String, String> {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("read hostname: {error}"))
    }

    fn current_boot_id() -> Result<String, String> {
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("read boot id: {error}"))
    }

    fn rust_version() -> Result<String, String> {
        let output = Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|error| format!("run rustc --version: {error}"))?;
        if !output.status.success() {
            return Err("rustc --version failed".to_string());
        }
        String::from_utf8(output.stdout)
            .map(|value| value.trim().to_string())
            .map_err(|error| format!("decode rustc --version: {error}"))
    }

    fn parse_run_arguments() -> Result<(usize, usize, bool), String> {
        let mut numeric = Vec::new();
        let mut smoke = false;
        for argument in std::env::args().skip(1) {
            if argument == "--smoke" {
                smoke = true;
            } else {
                numeric.push(argument);
            }
        }
        if numeric.len() > 2 {
            return Err("usage: perf_quad_many_scipy [rounds] [repetitions] [--smoke]".to_string());
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
        Ok((rounds, repetitions, smoke))
    }

    pub fn run() -> Result<(), String> {
        let (rounds, repetitions, smoke) = parse_run_arguments()?;
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
            "PROVENANCE mode={} host={} boot_id={} affinity_cpus={} \
             actual_affinity_count={} runtime_isa={} rustc={} \
             elf_path={} elf_sha256={} source_commit={} builder_identity={} \
             build_route={} trj_booking_claim_message_id={} \
             target_dir_policy=shared_reused",
            if smoke {
                "NON_EVIDENCE_SMOKE"
            } else {
                "EVIDENCE"
            },
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
        let (mut scipy, ready) = Scipy::start()?;
        let identity = ready.split('|').collect::<Vec<_>>();
        if identity.len() != 12
            || identity[0] != "READY"
            || identity[1] != PINNED_SCIPY
            || identity[2] != PINNED_NUMPY
            || identity[9] != "True"
            || identity[10] != "False"
        {
            return Err(format!("live SciPy identity gate failed: {ready}"));
        }
        for (value, label) in [
            (identity[3], "SciPy engine SHA-256"),
            (identity[4], "SciPy parameter SHA-256"),
            (identity[5], "SciPy reference SHA-256"),
            (identity[6], "SciPy input SHA-256"),
        ] {
            require_hex_sha(value, label)?;
        }
        let worker_capacity: usize = parse(identity[7], "SciPy worker capacity")?;
        let initial_os_tasks: usize = parse(identity[8], "SciPy initial OS tasks")?;
        if worker_capacity != affinity.len() || initial_os_tasks == 0 {
            return Err(format!(
                "SciPy execution identity mismatch: worker_capacity={worker_capacity} \
                 affinity={} initial_os_tasks={initial_os_tasks}",
                affinity.len()
            ));
        }
        for (label, rust_value, scipy_value) in [
            ("params_sha256", data.params_sha256.as_str(), identity[4]),
            (
                "reference_sha256",
                data.reference_sha256.as_str(),
                identity[5],
            ),
            ("input_sha256", data.input_sha256.as_str(), identity[6]),
        ] {
            if rust_value != scipy_value {
                return Err(format!(
                    "{label} mismatch: rust={rust_value} scipy={scipy_value}"
                ));
            }
        }
        println!(
            "SCIPY_IDENTITY scipy={} numpy={} scipy_engine_sha256={} \
             genuine={} fsci_loaded={} scipy_file={} worker_capacity={} \
             initial_os_tasks={} params_sha256={} reference_sha256={} \
             input_sha256={}",
            identity[1],
            identity[2],
            identity[3],
            identity[9],
            identity[10],
            identity[11],
            worker_capacity,
            initial_os_tasks,
            identity[4],
            identity[5],
            identity[6]
        );
        println!(
            "FIXTURE batch={} interval=0,1 p_range=2,50 w_range=1,35 \
             epsabs={EPSABS:.9e} epsrel={EPSREL:.9e} params_sha256={} \
             reference_sha256={} input_sha256={}",
            BATCH, data.params_sha256, data.reference_sha256, data.input_sha256
        );

        let ours_raw = execute_ours(&data)?;
        let ours_summary = summarize(&ours_raw, &data)?;
        print_summary("FrankenSciPy", ours_summary);
        if !quality_eligible(ours_summary) {
            return Err(format!(
                "FrankenSciPy failed the scientific gate: {ours_summary:?}"
            ));
        }
        let expected_workers = affinity.len().min(BATCH);
        let observed_rust_workers = observe_ours_workers(&data, expected_workers)?;
        println!(
            "ACTUAL_OBSERVED FrankenSciPy_worker_tasks={} \
             affinity_capacity={} requested_threads_not_used=true",
            observed_rust_workers,
            affinity.len()
        );

        let screened = screen_public_arms(&mut scipy)?;
        let selected = screened
            .iter()
            .min_by(|left, right| left.median.total_cmp(&right.median))
            .ok_or_else(|| "no valid SciPy public arm".to_string())?;
        let maximum_cross_scaled_error =
            cross_quality(&ours_raw, ours_summary, &selected.check, &data)?;
        println!(
            "selected_public_scipy_arm={} selected_screen_median_seconds={:.9} \
             selected_screen_samples={} selected_observed_active_tasks={} \
             selected_observed_os_tasks={} selected_observed_worker_processes={} \
             max_cross_scaled_error={maximum_cross_scaled_error:.9}",
            selected.check.arm,
            selected.median,
            selected.samples.len(),
            selected.max_active_tasks,
            selected.max_os_tasks,
            selected.max_worker_processes
        );
        let scalar_median = screened
            .iter()
            .find(|entry| entry.check.arm == "quad_scalar")
            .map(|entry| entry.median);
        let removed_serial_tax_ratio = scalar_median.map(|scalar| scalar / selected.median);
        if let (Some(scalar), Some(ratio)) = (scalar_median, removed_serial_tax_ratio) {
            println!(
                "mechanism_screen scalar_quad_over_selected={ratio:.9}x \
                 predicted_at_least_8={} scalar_seconds={scalar:.9} \
                 selected_seconds={:.9} falsified_by_quality=false",
                ratio >= 8.0,
                selected.median
            );
        } else {
            println!(
                "mechanism_screen scalar_quad_over_selected=unavailable \
                 predicted_at_least_8=false falsified_by_quality=true"
            );
        }

        if smoke {
            println!(
                "NON_EVIDENCE_SMOKE_COMPLETE selected_public_scipy_arm={} \
                 scalar_quad_over_selected={} actual_observed_rust_workers={} \
                 selected_worker_processes={} no_effect_verdict=true",
                selected.check.arm,
                removed_serial_tax_ratio
                    .map(|ratio| format!("{ratio:.9}x"))
                    .unwrap_or_else(|| "unavailable_quality_falsified".to_string()),
                observed_rust_workers,
                selected.max_worker_processes
            );
            scipy.stop()?;
            return Ok(());
        }

        report_host_wide_quiescence("before_effect")?;
        let selected_arm = selected.check.arm.clone();
        let measurement = measure(&mut scipy, &selected_arm, &data, rounds, repetitions)?;
        let decision = print_decision(&selected_arm, &measurement);
        println!(
            "ACTUAL_OBSERVED selected_scipy_active_tasks={} \
             selected_scipy_max_os_tasks={} selected_scipy_worker_processes={} \
             requested_worker_capacity_not_substituted=true",
            measurement.max_scipy_active_tasks,
            measurement.max_scipy_os_tasks,
            measurement.max_scipy_worker_processes
        );

        let p1 = selected_arm == "quad_vec";
        let p2 = decision.ratio_high < COLLAPSE_BOUNDARY;
        let p3 = decision.ratio_high < DURABLE_WIN_BOUNDARY;
        let p4 = removed_serial_tax_ratio.is_some_and(|ratio| ratio >= 8.0);
        let selected_is_process = selected_arm.ends_with("_process");
        let selected_is_thread = selected_arm == "quad_thread";
        let selected_is_single_vector =
            matches!(selected_arm.as_str(), "quad_vec" | "cubature_gk21");
        let p5_rust = observed_rust_workers == 32;
        let p5_scipy = if selected_is_process {
            measurement.max_scipy_worker_processes >= 16
        } else if selected_is_thread {
            measurement.max_scipy_active_tasks > 1
        } else if selected_is_single_vector {
            measurement.max_scipy_active_tasks == 1
        } else {
            true
        };
        println!(
            "PREREGISTERED_PREDICTIONS P1_single_quad_vec_fastest={p1} \
             P2_ci_upper_below_12_2={p2} P3_ci_upper_below_3={p3} \
             P4_scalar_over_selected_at_least_8={p4} \
             P5_rust_32_workers={p5_rust} \
             P5_selected_scipy_observation={p5_scipy} \
             selected_is_process={selected_is_process} \
             selected_is_thread={selected_is_thread} \
             selected_is_single_vector={selected_is_single_vector} \
             ratio_median={:.9} ratio_ci=[{:.9},{:.9}]",
            decision.ratio_median, decision.ratio_low, decision.ratio_high
        );
        let durable_frankenscipy_win =
            decision.decidable && decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if durable_frankenscipy_win {
            "FrankenSciPy quad_many".to_string()
        } else {
            selected_arm.clone()
        };
        println!(
            "CHOOSER STATEMENT: choose {chooser} for this exact 2,048-member \
             damped-frequency-response study; durable_frankenscipy_boundary=3x \
             durable_frankenscipy_win={durable_frankenscipy_win} \
             outcome={} ratio_ci_low={:.9} old_14_5_to_61_1x_retired=true",
            decision.outcome, decision.ratio_low
        );
        scipy.stop()?;
        Ok(())
    }
}

#[cfg(feature = "quad-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("perf_quad_many_scipy failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "quad-incumbent-bench"))]
fn main() {
    eprintln!("perf_quad_many_scipy requires --features quad-incumbent-bench");
    std::process::exit(2);
}
