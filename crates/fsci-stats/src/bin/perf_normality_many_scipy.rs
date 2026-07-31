//! Whole batched normality-screening report versus screened live SciPy 1.17.1.
//!
//! The historical `*_many` rows compared FrankenSciPy with scalar Python
//! loops. This harness screens every pre-registered public SciPy deployment,
//! freezes the strongest valid arm, and applies the corrected dual-null gate.

#[cfg(feature = "stats-incumbent-bench")]
mod bench {
    use fsci_stats::{jarque_bera_many, normaltest_many, shapiro_many};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    const ROWS: usize = 2_048;
    const COLS: usize = 4_096;
    const TESTS: usize = 3;
    const OUTPUTS: usize = ROWS * TESTS * 2;
    const ALPHA: f64 = 0.05;
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-10;
    const RELATIVE_TOLERANCE: f64 = 1.0e-8;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 3;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const COLLAPSE_BOUNDARY: f64 = 7.4;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
    const SCIPY_ARMS: [&str; 6] = [
        "scalar",
        "axis",
        "thread",
        "process",
        "hybrid_thread",
        "hybrid_process",
    ];

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import multiprocessing
import os
import sys
import threading
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy
from scipy import stats
from scipy.stats import (
    _ansari_swilk_statistics,
    _axis_nan_policy,
    _morestats,
    _stats_py,
)

ROWS = 2048
COLS = 4096
TESTS = 3
OUTPUTS = ROWS * TESTS * 2
ALPHA = 0.05
ABS_TOL = 1.0e-10
REL_TOL = 1.0e-8
PAYLOAD_BYTES = ROWS * COLS * 8

payload = sys.stdin.buffer.read(PAYLOAD_BYTES)
if len(payload) != PAYLOAD_BYTES:
    raise RuntimeError(
        f"input payload has {len(payload)} bytes, expected {PAYLOAD_BYTES}"
    )
INPUT_SHA256 = hashlib.sha256(payload).hexdigest()
DATA = np.frombuffer(payload, dtype="<f8").reshape((ROWS, COLS))
if DATA.shape != (ROWS, COLS) or not np.all(np.isfinite(DATA)):
    raise RuntimeError("invalid input matrix")

WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
OBSERVATION_LOCK = threading.Lock()
OBSERVED_NATIVE_IDS = set()
MAX_OS_TASKS = len(os.listdir("/proc/self/task"))

def reset_observations():
    global OBSERVED_NATIVE_IDS, MAX_OS_TASKS
    with OBSERVATION_LOCK:
        OBSERVED_NATIVE_IDS = set()
    MAX_OS_TASKS = len(os.listdir("/proc/self/task"))

def note_thread():
    global MAX_OS_TASKS
    with OBSERVATION_LOCK:
        OBSERVED_NATIVE_IDS.add(threading.get_native_id())
        MAX_OS_TASKS = max(MAX_OS_TASKS, len(os.listdir("/proc/self/task")))

def public_normaltest(values, axis=None):
    note_thread()
    return stats.normaltest(values, axis=axis)

def public_jarque_bera(values, axis=None):
    note_thread()
    return stats.jarque_bera(values, axis=axis)

def public_shapiro(values, axis=None):
    note_thread()
    return stats.shapiro(values, axis=axis)

def flat_blocks(nt_stat, nt_p, jb_stat, jb_p, sh_stat, sh_p):
    blocks = [
        np.asarray(nt_stat, dtype=np.float64).reshape(ROWS),
        np.asarray(nt_p, dtype=np.float64).reshape(ROWS),
        np.asarray(jb_stat, dtype=np.float64).reshape(ROWS),
        np.asarray(jb_p, dtype=np.float64).reshape(ROWS),
        np.asarray(sh_stat, dtype=np.float64).reshape(ROWS),
        np.asarray(sh_p, dtype=np.float64).reshape(ROWS),
    ]
    return np.concatenate(blocks)

def row_suite(index):
    row = DATA[index]
    nt = public_normaltest(row)
    jb = public_jarque_bera(row)
    sh = public_shapiro(row)
    return (
        os.getpid(),
        threading.get_native_id(),
        float(nt.statistic),
        float(nt.pvalue),
        float(jb.statistic),
        float(jb.pvalue),
        float(sh.statistic),
        float(sh.pvalue),
    )

def shapiro_row(index):
    result = public_shapiro(DATA[index])
    return (
        os.getpid(),
        threading.get_native_id(),
        float(result.statistic),
        float(result.pvalue),
    )

def warm_pid(_index):
    return os.getpid()

def rows_to_output(rows):
    matrix = np.asarray([row[2:] for row in rows], dtype=np.float64)
    if matrix.shape != (ROWS, 6):
        raise RuntimeError(f"invalid row result matrix {matrix.shape}")
    return flat_blocks(
        matrix[:, 0],
        matrix[:, 1],
        matrix[:, 2],
        matrix[:, 3],
        matrix[:, 4],
        matrix[:, 5],
    )

def axis_moments():
    nt = public_normaltest(DATA, axis=1)
    jb = public_jarque_bera(DATA, axis=1)
    return (
        np.asarray(nt.statistic, dtype=np.float64),
        np.asarray(nt.pvalue, dtype=np.float64),
        np.asarray(jb.statistic, dtype=np.float64),
        np.asarray(jb.pvalue, dtype=np.float64),
    )

def scalar_job():
    rows = [row_suite(index) for index in range(ROWS)]
    tids = {int(row[1]) for row in rows}
    return rows_to_output(rows), len(tids), 1

def axis_job():
    nt_stat, nt_p, jb_stat, jb_p = axis_moments()
    sh = public_shapiro(DATA, axis=1)
    output = flat_blocks(
        nt_stat,
        nt_p,
        jb_stat,
        jb_p,
        sh.statistic,
        sh.pvalue,
    )
    return output, len(OBSERVED_NATIVE_IDS), 1

def thread_job():
    rows = THREAD_POOL.map(row_suite, range(ROWS), chunksize=1)
    tids = {int(row[1]) for row in rows}
    return rows_to_output(rows), len(tids), 1

def process_job():
    rows = PROCESS_POOL.map(row_suite, range(ROWS), chunksize=1)
    pids = {int(row[0]) for row in rows}
    return rows_to_output(rows), len(pids), len(pids)

def hybrid_thread_job():
    nt_stat, nt_p, jb_stat, jb_p = axis_moments()
    rows = THREAD_POOL.map(shapiro_row, range(ROWS), chunksize=1)
    tids = {int(row[1]) for row in rows}
    sh_stat = np.asarray([row[2] for row in rows], dtype=np.float64)
    sh_p = np.asarray([row[3] for row in rows], dtype=np.float64)
    tids.update(OBSERVED_NATIVE_IDS)
    return (
        flat_blocks(nt_stat, nt_p, jb_stat, jb_p, sh_stat, sh_p),
        len(tids),
        1,
    )

def hybrid_process_job():
    nt_stat, nt_p, jb_stat, jb_p = axis_moments()
    rows = PROCESS_POOL.map(shapiro_row, range(ROWS), chunksize=1)
    pids = {int(row[0]) for row in rows}
    sh_stat = np.asarray([row[2] for row in rows], dtype=np.float64)
    sh_p = np.asarray([row[3] for row in rows], dtype=np.float64)
    return (
        flat_blocks(nt_stat, nt_p, jb_stat, jb_p, sh_stat, sh_p),
        1 + len(pids),
        len(pids),
    )

def execute(arm):
    reset_observations()
    if arm == "scalar":
        raw = scalar_job()
    elif arm == "axis":
        raw = axis_job()
    elif arm == "thread":
        raw = thread_job()
    elif arm == "process":
        raw = process_job()
    elif arm == "hybrid_thread":
        raw = hybrid_thread_job()
    elif arm == "hybrid_process":
        raw = hybrid_process_job()
    else:
        raise RuntimeError(f"unknown arm {arm!r}")
    output, active_tasks, worker_processes = raw
    if output.shape != (OUTPUTS,):
        raise RuntimeError(f"invalid output shape {output.shape}")
    if active_tasks < 1 or worker_processes < 1:
        raise RuntimeError("invalid observed task count")
    return (
        output,
        int(active_tasks),
        max(MAX_OS_TASKS, len(os.listdir("/proc/self/task"))),
        int(worker_processes),
    )

def classify(values):
    tolerance = ABS_TOL + REL_TOL * ALPHA
    boundary = np.abs(values - ALPHA) <= tolerance
    rejected = np.logical_and(values < ALPHA, np.logical_not(boundary))
    return rejected, boundary

def summarize(output):
    if output.shape != (OUTPUTS,) or not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite or malformed whole-job output")
    statistics = [
        output[0 * ROWS : 1 * ROWS],
        output[2 * ROWS : 3 * ROWS],
        output[4 * ROWS : 5 * ROWS],
    ]
    pvalues = [
        output[1 * ROWS : 2 * ROWS],
        output[3 * ROWS : 4 * ROWS],
        output[5 * ROWS : 6 * ROWS],
    ]
    if any(np.any(np.logical_or(values < 0.0, values > 1.0)) for values in pvalues):
        raise RuntimeError("p-value outside [0,1]")
    rejected = []
    boundary = []
    for values in pvalues:
        test_rejected, test_boundary = classify(values)
        rejected.append(test_rejected)
        boundary.append(test_boundary)
    valid = [int(np.count_nonzero(np.isfinite(values))) for values in pvalues]
    reject_counts = [int(np.count_nonzero(values)) for values in rejected]
    boundary_counts = [int(np.count_nonzero(values)) for values in boundary]
    union_rejected = int(np.count_nonzero(np.logical_or.reduce(rejected)))
    min_indices = [int(np.argmin(values)) for values in pvalues]
    min_values = [float(values[index]) for values, index in zip(pvalues, min_indices)]
    mean_statistics = [float(np.mean(values)) for values in statistics]
    mean_pvalues = [float(np.mean(values)) for values in pvalues]
    weights = np.arange(1, OUTPUTS + 1, dtype=np.float64)
    checksum = float(np.dot(output, weights))
    checksum += float(sum(reject_counts)) * 1.0e-3
    checksum += float(union_rejected) * 1.0e-5
    checksum += float(sum(min_indices)) * 1.0e-9
    checksum += float(sum(mean_statistics) + sum(mean_pvalues)) * 1.0e-12
    if valid != [ROWS, ROWS, ROWS] or not np.isfinite(checksum):
        raise RuntimeError("invalid whole-job summary")
    return (
        valid,
        reject_counts,
        boundary_counts,
        union_rejected,
        min_indices,
        min_values,
        mean_statistics,
        mean_pvalues,
        checksum,
    )

def source_sha256():
    paths = {
        inspect.getsourcefile(_axis_nan_policy),
        inspect.getsourcefile(_morestats),
        inspect.getsourcefile(_stats_py),
        _ansari_swilk_statistics.__file__,
    }
    if None in paths:
        raise RuntimeError("cannot resolve SciPy normality engine source")
    digest = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as handle:
            content = handle.read()
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()

# Fork only after DATA and every process-crossing callable exist, and before
# the parent creates any threads.
PROCESS_POOL = multiprocessing.get_context("fork").Pool(WORKER_CAPACITY)
for _warmup in range(2):
    PROCESS_POOL.map(warm_pid, range(WORKER_CAPACITY * 4), chunksize=1)
THREAD_POOL = ThreadPool(WORKER_CAPACITY)

SCIPY_ENGINE_SHA256 = source_sha256()
FSCI_LOADED = any(
    name == "fsci" or name.startswith("fsci_") or name.startswith("franken")
    for name in sys.modules
)
SCIPY_GENUINE = (
    scipy.__version__ == "1.17.1"
    and "/frankenscipy-scipy-1.17.1/site-packages/" in scipy.__file__
    and stats.normaltest.__module__ == "scipy.stats._stats_py"
    and stats.jarque_bera.__module__ == "scipy.stats._stats_py"
    and stats.shapiro.__module__ == "scipy.stats._morestats"
    and not FSCI_LOADED
)
print(
    "READY"
    f"|{scipy.__version__}"
    f"|{np.__version__}"
    f"|{SCIPY_ENGINE_SHA256}"
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
            output, active, tasks, processes = execute(arm)
            summarize(output)
            values_csv = ",".join(f"{value:.17e}" for value in output)
            print(
                "CHECK"
                f"|{arm}"
                f"|{INPUT_SHA256}"
                f"|{active}"
                f"|{tasks}"
                f"|{processes}"
                f"|{values_csv}",
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
                output, active, tasks, processes = execute(arm)
                checksum += summarize(output)[8]
                max_active = max(max_active, active)
                max_tasks = max(max_tasks, tasks)
                max_processes = max(max_processes, processes)
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

    #[derive(Clone)]
    struct Dataset {
        rows: Vec<Vec<f64>>,
        input_sha256: String,
        regime_counts: [usize; 4],
    }

    #[derive(Clone)]
    struct RawJob {
        values: Vec<f64>,
    }

    #[derive(Clone, Copy, Debug)]
    struct JobSummary {
        finite_statistics: [usize; TESTS],
        valid_pvalues: [usize; TESTS],
        rejection_counts: [usize; TESTS],
        boundary_counts: [usize; TESTS],
        union_rejections: usize,
        minimum_indices: [usize; TESTS],
        minimum_pvalues: [f64; TESTS],
        mean_statistics: [f64; TESTS],
        mean_pvalues: [f64; TESTS],
        checksum: f64,
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
        input_sha256: String,
        active_tasks: usize,
        max_os_tasks: usize,
        worker_processes: usize,
        raw: RawJob,
        summary: JobSummary,
    }

    struct Scipy {
        child: Child,
        stdin: BufWriter<ChildStdin>,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    impl Scipy {
        fn start(data: &Dataset) -> Result<(Self, String), String> {
            let python =
                std::env::var("SCIPY_PYTHON").unwrap_or_else(|_| "/usr/bin/python3.13".to_string());
            let mut child = Command::new(&python)
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("PYTHONPATH", SCIPY_SITE_PACKAGES)
                .env("PYTHONNOUSERSITE", "1")
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
            let child_stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy oracle has no stdin".to_string())?;
            let mut stdin = BufWriter::new(child_stdin);
            let mut payload = Vec::with_capacity(ROWS * COLS * size_of::<f64>());
            for value in data.rows.iter().flatten() {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            if payload.len() != ROWS * COLS * size_of::<f64>() {
                return Err(format!(
                    "Rust fixture payload has {} bytes, expected {}",
                    payload.len(),
                    ROWS * COLS * size_of::<f64>()
                ));
            }
            let payload_sha256 = format!("{:x}", Sha256::digest(&payload));
            if payload_sha256.cmp(&data.input_sha256).is_ne() {
                return Err(format!(
                    "Rust fixture payload SHA changed: built={} streamed={payload_sha256}",
                    data.input_sha256
                ));
            }
            stdin
                .write_all(&payload)
                .map_err(|error| format!("stream exact fixture to SciPy: {error}"))?;
            stdin
                .flush()
                .map_err(|error| format!("flush exact SciPy fixture: {error}"))?;
            drop(payload);

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
            let reply = self.request(&format!("CHECK {arm}"), "normality quality check")?;
            let fields = reply.splitn(7, '|').collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("invalid SciPy quality reply: {reply}"));
            }
            require_hex_sha(fields[2], "SciPy CHECK input SHA-256")?;
            let values = fields[6]
                .split(',')
                .map(|field| parse(field, "SciPy normality output"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != OUTPUTS {
                return Err(format!(
                    "SciPy {arm} returned {} outputs instead of {OUTPUTS}",
                    values.len()
                ));
            }
            let raw = RawJob { values };
            let summary = summarize(&raw)?;
            Ok(ScipyCheck {
                arm: arm.to_string(),
                input_sha256: fields[2].to_string(),
                active_tasks: parse(fields[3], "SciPy active tasks")?,
                max_os_tasks: parse(fields[4], "SciPy max OS tasks")?,
                worker_processes: parse(fields[5], "SciPy worker processes")?,
                raw,
                summary,
            })
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let reply = self.request(
                &format!("TIME {arm} {repetitions}"),
                "normality timed report",
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

    fn parse<T>(value: &str, label: &str) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        value
            .parse()
            .map_err(|error| format!("parse {label} from {value:?}: {error}"))
    }

    fn build_dataset() -> Dataset {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        let mut rows = Vec::with_capacity(ROWS);
        let mut regime_counts = [0usize; 4];
        let mut digest = Sha256::new();
        for row_index in 0..ROWS {
            let regime = row_index % regime_counts.len();
            regime_counts[regime] += 1;
            let mut row = Vec::with_capacity(COLS);
            for column in 0..COLS {
                let mut base = -6.0f64;
                for _ in 0..12 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    base += (state >> 11) as f64 / (1u64 << 53) as f64;
                }
                let value = if regime == 0 {
                    base
                } else if regime == 1 {
                    1.75 * base + 0.25
                } else if regime == 2 {
                    base.abs()
                } else if column.is_multiple_of(257) {
                    7.0 * base
                } else {
                    base
                };
                digest.update(value.to_le_bytes());
                row.push(value);
            }
            rows.push(row);
        }
        Dataset {
            rows,
            input_sha256: format!("{:x}", digest.finalize()),
            regime_counts,
        }
    }

    fn execute_ours(data: &Dataset) -> Result<RawJob, String> {
        let normal = normaltest_many(&data.rows);
        let jarque = jarque_bera_many(&data.rows);
        let shapiro = shapiro_many(&data.rows);
        if normal.len() != ROWS || jarque.len() != ROWS || shapiro.len() != ROWS {
            return Err(format!(
                "FrankenSciPy result lengths normal={} jarque={} shapiro={}, expected {ROWS}",
                normal.len(),
                jarque.len(),
                shapiro.len()
            ));
        }
        let mut values = Vec::with_capacity(OUTPUTS);
        values.extend(normal.iter().map(|result| result.statistic));
        values.extend(normal.iter().map(|result| result.pvalue));
        values.extend(jarque.iter().map(|result| result.statistic));
        values.extend(jarque.iter().map(|result| result.pvalue));
        values.extend(shapiro.iter().map(|result| result.statistic));
        values.extend(shapiro.iter().map(|result| result.pvalue));
        Ok(RawJob { values })
    }

    fn output_block(raw: &RawJob, block: usize) -> &[f64] {
        &raw.values[block * ROWS..(block + 1) * ROWS]
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum RejectionClass {
        Accept,
        Reject,
        Boundary,
    }

    fn comparison_tolerance(reference: f64) -> f64 {
        ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs()
    }

    fn rejection_class(pvalue: f64) -> RejectionClass {
        if (pvalue - ALPHA).abs() <= comparison_tolerance(ALPHA) {
            RejectionClass::Boundary
        } else if pvalue < ALPHA {
            RejectionClass::Reject
        } else {
            RejectionClass::Accept
        }
    }

    fn summarize(raw: &RawJob) -> Result<JobSummary, String> {
        if raw.values.len() != OUTPUTS {
            return Err(format!(
                "whole-job output length {} differs from {OUTPUTS}",
                raw.values.len()
            ));
        }
        let mut finite_statistics = [0usize; TESTS];
        let mut valid_pvalues = [0usize; TESTS];
        let mut rejection_counts = [0usize; TESTS];
        let mut boundary_counts = [0usize; TESTS];
        let mut minimum_indices = [0usize; TESTS];
        let mut minimum_pvalues = [f64::INFINITY; TESTS];
        let mut mean_statistics = [0.0f64; TESTS];
        let mut mean_pvalues = [0.0f64; TESTS];
        let mut union_rejections = 0usize;

        for test in 0..TESTS {
            let statistics = output_block(raw, test * 2);
            let pvalues = output_block(raw, test * 2 + 1);
            for (index, (&statistic, &pvalue)) in statistics.iter().zip(pvalues).enumerate() {
                if statistic.is_finite() {
                    finite_statistics[test] += 1;
                }
                if pvalue.is_finite() && (0.0..=1.0).contains(&pvalue) {
                    valid_pvalues[test] += 1;
                }
                match rejection_class(pvalue) {
                    RejectionClass::Accept => {}
                    RejectionClass::Reject => rejection_counts[test] += 1,
                    RejectionClass::Boundary => boundary_counts[test] += 1,
                }
                if pvalue < minimum_pvalues[test] {
                    minimum_pvalues[test] = pvalue;
                    minimum_indices[test] = index;
                }
                mean_statistics[test] += statistic;
                mean_pvalues[test] += pvalue;
            }
            mean_statistics[test] /= ROWS as f64;
            mean_pvalues[test] /= ROWS as f64;
        }
        for index in 0..ROWS {
            if (0..TESTS).any(|test| {
                rejection_class(output_block(raw, test * 2 + 1)[index]) == RejectionClass::Reject
            }) {
                union_rejections += 1;
            }
        }

        let mut checksum = 0.0f64;
        for (index, value) in raw.values.iter().enumerate() {
            checksum += *value * (index + 1) as f64;
        }
        checksum += rejection_counts.iter().sum::<usize>() as f64 * 1.0e-3;
        checksum += union_rejections as f64 * 1.0e-5;
        checksum += minimum_indices.iter().sum::<usize>() as f64 * 1.0e-9;
        checksum +=
            (mean_statistics.iter().sum::<f64>() + mean_pvalues.iter().sum::<f64>()) * 1.0e-12;

        let summary = JobSummary {
            finite_statistics,
            valid_pvalues,
            rejection_counts,
            boundary_counts,
            union_rejections,
            minimum_indices,
            minimum_pvalues,
            mean_statistics,
            mean_pvalues,
            checksum,
        };
        if !quality_eligible(summary) {
            return Err(format!("whole-job output integrity failure: {summary:?}"));
        }
        Ok(summary)
    }

    fn quality_eligible(summary: JobSummary) -> bool {
        summary.finite_statistics == [ROWS; TESTS]
            && summary.valid_pvalues == [ROWS; TESTS]
            && summary
                .rejection_counts
                .iter()
                .zip(summary.boundary_counts)
                .all(|(&rejected, boundary)| rejected + boundary <= ROWS)
            && summary.union_rejections <= ROWS
            && summary.minimum_indices.iter().all(|&index| index < ROWS)
            && summary
                .minimum_pvalues
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && summary
                .mean_statistics
                .iter()
                .all(|value| value.is_finite())
            && summary
                .mean_pvalues
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && summary.checksum.is_finite()
    }

    fn cross_quality(
        ours: &RawJob,
        ours_summary: JobSummary,
        scipy: &ScipyCheck,
        input_sha256: &str,
    ) -> Result<f64, String> {
        if scipy.input_sha256 != input_sha256 {
            return Err(format!(
                "input SHA mismatch for {}: rust={input_sha256} scipy={}",
                scipy.arm, scipy.input_sha256
            ));
        }
        if !quality_eligible(ours_summary) || !quality_eligible(scipy.summary) {
            return Err(format!(
                "scientific integrity gate failed for {}: ours={ours_summary:?} scipy={:?}",
                scipy.arm, scipy.summary
            ));
        }
        if ours_summary.finite_statistics != scipy.summary.finite_statistics
            || ours_summary.valid_pvalues != scipy.summary.valid_pvalues
            || ours_summary.rejection_counts != scipy.summary.rejection_counts
            || ours_summary.boundary_counts != scipy.summary.boundary_counts
            || ours_summary.union_rejections != scipy.summary.union_rejections
        {
            return Err(format!(
                "whole-report count disagreement for {}: ours={ours_summary:?} scipy={:?}",
                scipy.arm, scipy.summary
            ));
        }

        let mut maximum_scaled_error = 0.0f64;
        for (index, (&got, &reference)) in ours.values.iter().zip(&scipy.raw.values).enumerate() {
            let scaled = (got - reference).abs() / comparison_tolerance(reference);
            maximum_scaled_error = maximum_scaled_error.max(scaled);
            if scaled > 1.0 {
                return Err(format!(
                    "output {index} disagrees for {}: rust={got:.17e} \
                     scipy={reference:.17e} scaled_error={scaled:.9}",
                    scipy.arm
                ));
            }
        }
        for test in 0..TESTS {
            let ours_pvalues = output_block(ours, test * 2 + 1);
            let scipy_pvalues = output_block(&scipy.raw, test * 2 + 1);
            for index in 0..ROWS {
                if rejection_class(ours_pvalues[index]) != rejection_class(scipy_pvalues[index]) {
                    return Err(format!(
                        "rejection classification differs for {} test={test} row={index}",
                        scipy.arm
                    ));
                }
            }
            let ours_minimum = ours_summary.minimum_indices[test];
            let scipy_minimum = scipy.summary.minimum_indices[test];
            if ours_minimum != scipy_minimum {
                let ours_tied = (ours_pvalues[ours_minimum] - ours_pvalues[scipy_minimum]).abs()
                    <= comparison_tolerance(ours_pvalues[ours_minimum]);
                let scipy_tied = (scipy_pvalues[ours_minimum] - scipy_pvalues[scipy_minimum]).abs()
                    <= comparison_tolerance(scipy_pvalues[scipy_minimum]);
                if !ours_tied && !scipy_tied {
                    return Err(format!(
                        "minimum-p row differs for {} test={test}: rust={ours_minimum} \
                         scipy={scipy_minimum}",
                        scipy.arm
                    ));
                }
            }
        }
        Ok(maximum_scaled_error)
    }

    fn print_summary(label: &str, summary: JobSummary) {
        println!(
            "{label}: finite_statistics={:?} valid_pvalues={:?} \
             rejection_counts={:?} boundary_counts={:?} union_rejections={} \
             minimum_indices={:?} minimum_pvalues={:?} mean_statistics={:?} \
             mean_pvalues={:?} checksum={:.17e}",
            summary.finite_statistics,
            summary.valid_pvalues,
            summary.rejection_counts,
            summary.boundary_counts,
            summary.union_rejections,
            summary.minimum_indices,
            summary.minimum_pvalues,
            summary.mean_statistics,
            summary.mean_pvalues,
            summary.checksum
        );
    }

    fn time_ours(data: &Dataset, repetitions: usize) -> Result<f64, String> {
        let started = Instant::now();
        let mut checksum = 0.0f64;
        for _ in 0..repetitions {
            let raw = execute_ours(data)?;
            checksum += summarize(&raw)?.checksum;
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
        for _ in 0..4 {
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
        maximum_scaled_error: f64,
    }

    fn screen_arm(
        scipy: &mut Scipy,
        arm: &str,
        ours: &RawJob,
        ours_summary: JobSummary,
        input_sha256: &str,
    ) -> Result<Option<ScreenedArm>, String> {
        let check = match scipy.check(arm) {
            Ok(check) => check,
            Err(error) if error.contains("failed: ERROR|") => {
                println!(
                    "screen_excluded arm={arm} reason=execution_error detail={error:?} \
                     time_samples=0 eligible=false"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        print_summary(&format!("screen SciPy {arm}"), check.summary);
        let maximum_scaled_error = match cross_quality(ours, ours_summary, &check, input_sha256) {
            Ok(value) => value,
            Err(error) => {
                println!(
                    "screen_excluded arm={arm} reason=scientific_gate detail={error:?} \
                     time_samples=0 eligible=false observed_active_tasks={} \
                     observed_os_tasks={} observed_worker_processes={}",
                    check.active_tasks, check.max_os_tasks, check.worker_processes
                );
                return Ok(None);
            }
        };
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
            "screen_result arm={arm} median_seconds={arm_median:.9} samples={} \
             observed_active_tasks={max_active_tasks} observed_os_tasks={max_os_tasks} \
             observed_worker_processes={max_worker_processes} \
             max_cross_scaled_error={maximum_scaled_error:.9}",
            csv(&samples)
        );
        Ok(Some(ScreenedArm {
            check,
            median: arm_median,
            samples,
            max_active_tasks,
            max_os_tasks,
            max_worker_processes,
            maximum_scaled_error,
        }))
    }

    fn screen_public_arms(
        scipy: &mut Scipy,
        ours: &RawJob,
        ours_summary: JobSummary,
        input_sha256: &str,
    ) -> Result<Vec<ScreenedArm>, String> {
        let mut screened = Vec::with_capacity(SCIPY_ARMS.len());
        for arm in SCIPY_ARMS {
            if let Some(result) = screen_arm(scipy, arm, ours, ours_summary, input_sha256)? {
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
                "round={round} ours_seconds={:.9} scipy_seconds={:.9} ratio={:.9} \
                 ours_null={ours_null:.9} scipy_null={:.9} \
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
            "raw_samples_seconds: ours={} scipy={} ratios={} null_ours={} null_scipy={}",
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
                continue;
            }
            let busy_fraction = total.saturating_sub(idle) as f64 / total as f64;
            maximum_busy_fraction = maximum_busy_fraction.max(busy_fraction);
            if busy_fraction > HOST_QUIESCENCE_MAX_BUSY {
                busy.push(format!("{cpu}:{busy_fraction:.3}"));
            }
        }
        println!(
            "host_quiescence phase={phase} sample_ms={} threshold={:.3} \
             max_busy_fraction={maximum_busy_fraction:.3} busy_cpus={}",
            HOST_QUIESCENCE_SAMPLE.as_millis(),
            HOST_QUIESCENCE_MAX_BUSY,
            if busy.is_empty() {
                "none".to_string()
            } else {
                busy.join(",")
            }
        );
        if !busy.is_empty() {
            return Err(format!(
                "host is not quiescent for {phase}: {}",
                busy.join(",")
            ));
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
            return Err(
                "usage: perf_normality_many_scipy [rounds] [repetitions] [--smoke]".to_string(),
            );
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
            require_host_wide_quiescence("before_screen")?;
        }
        println!(
            "PROVENANCE mode={} host={} boot_id={} affinity_cpus={} \
             actual_affinity_count={} runtime_isa={} rustc={} elf_path={} \
             elf_sha256={} source_commit={} builder_identity={} build_route={} \
             trj_booking_claim_message_id={} target_dir_policy=shared_reused",
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
        require_hex_sha(&data.input_sha256, "Rust input SHA-256")?;
        println!(
            "FIXTURE rows={ROWS} observations_per_row={COLS} outputs={OUTPUTS} \
             alpha={ALPHA:.9} regime_counts={:?} input_sha256={} \
             input_bytes={} jarque_bera_large_sample_regime=true \
             shapiro_pvalue_sample_boundary_respected=true",
            data.regime_counts,
            data.input_sha256,
            ROWS * COLS * size_of::<f64>()
        );

        let (mut scipy, ready) = Scipy::start(&data)?;
        let identity = ready.split('|').collect::<Vec<_>>();
        if identity.len() != 10
            || identity[0] != "READY"
            || identity[1] != "1.17.1"
            || identity[7] != "True"
            || identity[8] != "False"
            || !identity[9].starts_with(SCIPY_SITE_PACKAGES)
        {
            return Err(format!("live SciPy identity gate failed: {ready}"));
        }
        require_hex_sha(identity[3], "SciPy engine SHA-256")?;
        require_hex_sha(identity[4], "SciPy input SHA-256")?;
        if identity[4] != data.input_sha256 {
            return Err(format!(
                "cross-process input identity mismatch: rust={} scipy={}",
                data.input_sha256, identity[4]
            ));
        }
        let worker_capacity: usize = parse(identity[5], "SciPy worker capacity")?;
        let initial_os_tasks: usize = parse(identity[6], "SciPy initial OS tasks")?;
        if worker_capacity != affinity.len() || initial_os_tasks == 0 {
            return Err(format!(
                "SciPy execution identity mismatch: worker_capacity={worker_capacity} \
                 affinity={} initial_os_tasks={initial_os_tasks}",
                affinity.len()
            ));
        }
        println!(
            "SCIPY_IDENTITY scipy={} numpy={} scipy_engine_sha256={} \
             input_sha256={} worker_capacity={} initial_os_tasks={} genuine={} \
             fsci_loaded={} scipy_file={}",
            identity[1],
            identity[2],
            identity[3],
            identity[4],
            worker_capacity,
            initial_os_tasks,
            identity[7],
            identity[8],
            identity[9]
        );

        let ours_raw = execute_ours(&data)?;
        let ours_summary = summarize(&ours_raw)?;
        print_summary("FrankenSciPy", ours_summary);
        let expected_rust_workers = affinity.len().min(ROWS / 128);
        let observed_rust_workers = observe_ours_workers(&data, expected_rust_workers)?;
        println!(
            "ACTUAL_OBSERVED FrankenSciPy_worker_tasks={} \
             affinity_capacity={} production_shapiro_fanout_ceiling={} \
             requested_threads_not_substituted=true",
            observed_rust_workers,
            affinity.len(),
            expected_rust_workers
        );

        let screened = screen_public_arms(&mut scipy, &ours_raw, ours_summary, &data.input_sha256)?;
        let selected = screened
            .iter()
            .min_by(|left, right| left.median.total_cmp(&right.median))
            .ok_or_else(|| "no valid SciPy public arm".to_string())?;
        println!(
            "selected_public_scipy_arm={} selected_screen_median_seconds={:.9} \
             selected_screen_samples={} selected_observed_active_tasks={} \
             selected_observed_os_tasks={} selected_observed_worker_processes={} \
             max_cross_scaled_error={:.9}",
            selected.check.arm,
            selected.median,
            selected.samples.len(),
            selected.max_active_tasks,
            selected.max_os_tasks,
            selected.max_worker_processes,
            selected.maximum_scaled_error
        );
        let scalar_median = screened
            .iter()
            .find(|entry| entry.check.arm == "scalar")
            .map(|entry| entry.median);
        let removed_user_loop_tax_ratio = scalar_median.map(|scalar| scalar / selected.median);
        if let (Some(scalar), Some(ratio)) = (scalar_median, removed_user_loop_tax_ratio) {
            println!(
                "mechanism_screen scalar_over_selected={ratio:.9}x \
                 predicted_at_least_8={} scalar_seconds={scalar:.9} \
                 selected_seconds={:.9} falsified_by_quality=false",
                ratio >= 8.0,
                selected.median
            );
        } else {
            println!(
                "mechanism_screen scalar_over_selected=unavailable \
                 predicted_at_least_8=false falsified_by_quality=true"
            );
        }

        if smoke {
            let selected_is_process = selected.check.arm.ends_with("process");
            let selected_is_axis = selected.check.arm == "axis";
            let p5_scipy = if selected_is_process {
                selected.max_worker_processes >= 16
            } else if selected_is_axis {
                selected.max_active_tasks == 1
            } else {
                true
            };
            println!(
                "NON_EVIDENCE_SMOKE_COMPLETE selected_public_scipy_arm={} \
                 P1_hybrid_process_fastest={} P3_scalar_over_selected_at_least_8={} \
                 P5_rust_at_least_16={} P5_selected_scipy_observation={} \
                 scalar_over_selected={} actual_observed_rust_workers={} \
                 selected_active_tasks={} selected_worker_processes={} \
                 no_primary_effect_verdict=true",
                selected.check.arm,
                selected.check.arm == "hybrid_process",
                removed_user_loop_tax_ratio.is_some_and(|ratio| ratio >= 8.0),
                observed_rust_workers >= 16,
                p5_scipy,
                removed_user_loop_tax_ratio
                    .map(|ratio| format!("{ratio:.9}x"))
                    .unwrap_or_else(|| "unavailable_quality_falsified".to_string()),
                observed_rust_workers,
                selected.max_active_tasks,
                selected.max_worker_processes
            );
            scipy.stop()?;
            return Ok(());
        }

        require_host_wide_quiescence("before_effect")?;
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

        let p1 = selected_arm == "hybrid_process";
        let p2 = decision.ratio_high < COLLAPSE_BOUNDARY;
        let p3 = removed_user_loop_tax_ratio.is_some_and(|ratio| ratio >= 8.0);
        let p4 = decision.ratio_high < DURABLE_WIN_BOUNDARY;
        let selected_is_process = selected_arm.ends_with("process");
        let selected_is_axis = selected_arm == "axis";
        let p5_rust = observed_rust_workers >= 16;
        let p5_scipy = if selected_is_process {
            measurement.max_scipy_worker_processes >= 16
        } else if selected_is_axis {
            measurement.max_scipy_active_tasks == 1
        } else {
            true
        };
        println!(
            "PREREGISTERED_PREDICTIONS P1_hybrid_process_fastest={p1} \
             P2_ci_upper_below_7_4={p2} P3_scalar_over_selected_at_least_8={p3} \
             P4_ci_upper_below_3={p4} P5_rust_at_least_16={p5_rust} \
             P5_selected_scipy_observation={p5_scipy} \
             selected_is_process={selected_is_process} selected_is_axis={selected_is_axis} \
             ratio_median={:.9} ratio_ci=[{:.9},{:.9}]",
            decision.ratio_median, decision.ratio_low, decision.ratio_high
        );
        let durable_frankenscipy_win =
            decision.decidable && decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if durable_frankenscipy_win {
            "FrankenSciPy normaltest_many + jarque_bera_many + shapiro_many".to_string()
        } else {
            format!("SciPy {selected_arm}")
        };
        println!(
            "CHOOSER STATEMENT: choose {chooser} for this exact 2,048-by-4,096 \
             many-channel normality-screening report; \
             durable_frankenscipy_boundary=3x \
             durable_frankenscipy_win={durable_frankenscipy_win} \
             outcome={} ratio_ci_low={:.9} \
             historical_74_to_2267x_scalar_loop_claims_retired=true",
            decision.outcome, decision.ratio_low
        );
        scipy.stop()?;
        require_host_wide_quiescence("after_effect")?;
        Ok(())
    }
}

#[cfg(feature = "stats-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("perf_normality_many_scipy failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "stats-incumbent-bench"))]
fn main() {
    eprintln!("perf_normality_many_scipy requires --features stats-incumbent-bench");
    std::process::exit(2);
}
