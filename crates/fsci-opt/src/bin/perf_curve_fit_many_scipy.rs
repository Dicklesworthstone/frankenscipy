//! Whole 2,000-trace curve-fit study versus screened SciPy 1.17.1.
//!
//! The historical `curve_fit_many` row used a serial numerical-Jacobian
//! Python loop. This harness screens public analytic, pooled, and joint-sparse
//! incumbent routes and applies the corrected dual-null median gate.

#[cfg(feature = "opt-incumbent-bench")]
mod bench {
    use fsci_opt::{CurveFitOptions, curve_fit_many};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashMap};
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

    const BATCH: usize = 2_000;
    const POINTS: usize = 80;
    const PARAMETERS: usize = 3;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 3;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const MAX_FIT_RMSE: f64 = 0.02;
    const MAX_CROSS_CURVE_RMSE: f64 = 5.0e-4;
    const COLLAPSE_BOUNDARY: f64 = 11.3;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SCIPY_SITE_PACKAGES: &str =
        "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
    const SCIPY_ARMS: [&str; 5] = [
        "curve_fit_numeric_scalar",
        "curve_fit_numeric_pool",
        "curve_fit_jac_scalar",
        "curve_fit_jac_pool",
        "joint_sparse",
    ];

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import math
import os
import sys
import time
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy
from scipy import optimize, sparse
from scipy.optimize import _minpack_py
from scipy.optimize._lsq import least_squares as _least_squares_module

BATCH = 2000
POINTS = 80
PARAMETERS = 3
P0 = np.asarray([1.0, 1.0, 0.0], dtype=np.float64)
X = (np.arange(POINTS, dtype=np.float64) * (5.0 / 79.0)).astype("<f8")
MASK64 = (1 << 64) - 1
state = 12345

def rng():
    global state
    state = (state * 6364136223846793005 + 1) & MASK64
    return (state >> 11) / float(1 << 53)

truth = []
rows = []
for _ in range(BATCH):
    a = 1.0 + 2.0 * rng()
    b = 0.3 + rng()
    c = rng()
    truth.append((a, b, c))
    rows.append(
        [
            a * math.exp(-b * float(x)) + c + 0.02 * (rng() - 0.5)
            for x in X
        ]
    )
TRUTH = np.asarray(truth, dtype="<f8")
Y = np.asarray(rows, dtype="<f8")
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
POOL = ThreadPool(WORKER_CAPACITY)

def array_sha(array):
    return hashlib.sha256(
        np.asarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()

X_SHA256 = array_sha(X)
TRUTH_SHA256 = array_sha(TRUTH)
Y_SHA256 = array_sha(Y)
input_digest = hashlib.sha256()
for label, array in (("x", X), ("truth", TRUTH), ("y", Y)):
    input_digest.update(label.encode("ascii"))
    input_digest.update(b"\0")
    input_digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
INPUT_SHA256 = input_digest.hexdigest()

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

def analytic_jac(x, a, b, _c):
    exponential = np.exp(-b * x)
    return np.column_stack(
        (exponential, -a * x * exponential, np.ones_like(x))
    )

def fit_one(index, analytic):
    kwargs = {
        "p0": P0,
        "method": "lm",
        "full_output": True,
        "ftol": 1.0e-8,
        "xtol": 1.0e-8,
        "gtol": 1.0e-8,
        "maxfev": 400,
    }
    if analytic:
        kwargs["jac"] = analytic_jac
    with warnings.catch_warnings():
        warnings.simplefilter("error", optimize.OptimizeWarning)
        popt, _pcov, info, _message, status = optimize.curve_fit(
            model, X, Y[index], **kwargs
        )
    return np.asarray(popt, dtype=np.float64), int(status in (1, 2, 3, 4)), int(
        info.get("nfev", 0)
    )

def independent_job(analytic, pooled):
    if pooled:
        results = POOL.map(
            lambda index: fit_one(index, analytic), range(BATCH)
        )
    else:
        results = [fit_one(index, analytic) for index in range(BATCH)]
    popt = np.asarray([result[0] for result in results], dtype=np.float64)
    successes = sum(result[1] for result in results)
    nfev = sum(result[2] for result in results)
    return popt, successes, nfev

JOINT_ROWS = np.repeat(np.arange(BATCH * POINTS, dtype=np.int64), PARAMETERS)
JOINT_COLS = np.empty(BATCH * POINTS * PARAMETERS, dtype=np.int64)
for index in range(BATCH):
    start = index * POINTS * PARAMETERS
    stop = (index + 1) * POINTS * PARAMETERS
    JOINT_COLS[start:stop] = np.tile(
        np.arange(PARAMETERS, dtype=np.int64) + PARAMETERS * index,
        POINTS,
    )

def joint_residual(flat):
    params = flat.reshape(BATCH, PARAMETERS)
    predicted = (
        params[:, 0, None] * np.exp(-params[:, 1, None] * X)
        + params[:, 2, None]
    )
    return (predicted - Y).ravel()

def joint_jacobian(flat):
    params = flat.reshape(BATCH, PARAMETERS)
    exponential = np.exp(-params[:, 1, None] * X)
    data = np.stack(
        (
            exponential,
            -params[:, 0, None] * X * exponential,
            np.ones_like(exponential),
        ),
        axis=2,
    ).ravel()
    return sparse.csr_array(
        (data, (JOINT_ROWS, JOINT_COLS)),
        shape=(BATCH * POINTS, BATCH * PARAMETERS),
    )

def joint_job():
    result = optimize.least_squares(
        joint_residual,
        np.tile(P0, BATCH),
        jac=joint_jacobian,
        method="trf",
        tr_solver="lsmr",
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=400,
    )
    return (
        np.asarray(result.x, dtype=np.float64).reshape(BATCH, PARAMETERS),
        BATCH if result.success else 0,
        int(result.nfev),
    )

def execute(arm):
    if arm == "curve_fit_numeric_scalar":
        return independent_job(False, False)
    if arm == "curve_fit_numeric_pool":
        return independent_job(False, True)
    if arm == "curve_fit_jac_scalar":
        return independent_job(True, False)
    if arm == "curve_fit_jac_pool":
        return independent_job(True, True)
    if arm == "joint_sparse":
        return joint_job()
    raise RuntimeError(f"unknown arm {arm!r}")

def summarize(raw):
    popt, successes, nfev = raw
    popt = np.asarray(popt, dtype=np.float64)
    if popt.shape != (BATCH, PARAMETERS) or not np.all(np.isfinite(popt)):
        raise RuntimeError(f"invalid parameter output shape={popt.shape}")
    predicted = (
        popt[:, 0, None] * np.exp(-popt[:, 1, None] * X)
        + popt[:, 2, None]
    )
    residual = predicted - Y
    rss = np.sum(residual * residual, axis=1)
    initial = P0[0] * np.exp(-P0[1] * X) + P0[2]
    initial_rss = np.sum(np.square(initial[None, :] - Y), axis=1)
    improved = int(np.count_nonzero(rss < initial_rss))
    rmse = np.sqrt(rss / POINTS)
    parameter_error = np.max(
        np.abs(popt - TRUTH) / np.asarray([2.0, 1.0, 1.0]), axis=1
    )
    worst_index = int(np.argmax(rss))
    weights = np.arange(1, BATCH + 1, dtype=np.float64)
    checksum = float(
        np.dot(popt[:, 0], weights)
        + np.dot(popt[:, 1], weights) * 1.0e-2
        + np.dot(popt[:, 2], weights) * 1.0e-4
        + np.dot(rss, weights) * 1.0e-6
        + successes
        + nfev * 1.0e-9
        + worst_index * 1.0e-12
    )
    return (
        popt,
        int(successes),
        improved,
        worst_index,
        float(rss[worst_index]),
        float(np.sum(rss)),
        float(np.median(rmse)),
        float(np.percentile(rmse, 95)),
        float(np.percentile(rmse, 99)),
        float(np.max(rmse)),
        float(np.median(parameter_error)),
        float(np.percentile(parameter_error, 99)),
        int(nfev),
        checksum,
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

def source_hash(module):
    source_path = inspect.getsourcefile(module)
    if source_path is None:
        raise RuntimeError(f"cannot resolve source for {module!r}")
    with open(source_path, "rb") as source_file:
        source = source_file.read()
    return source_path, hashlib.sha256(source).hexdigest(), source

engine_rows = [
    ("curve_fit",) + source_hash(_minpack_py),
    ("least_squares",) + source_hash(_least_squares_module),
]
engine_digest = hashlib.sha256()
for label, _path, _digest, source in engine_rows:
    engine_digest.update(label.encode("ascii"))
    engine_digest.update(b"\0")
    engine_digest.update(source)
SCIPY_ENGINE_SHA256 = engine_digest.hexdigest()

for arm in (
    "curve_fit_numeric_scalar",
    "curve_fit_numeric_pool",
    "curve_fit_jac_scalar",
    "curve_fit_jac_pool",
    "joint_sparse",
):
    summary = summarize(execute(arm))
    if not math.isfinite(summary[-1]):
        raise RuntimeError(f"{arm} warmup failed")

fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and optimize.curve_fit.__module__ == "scipy.optimize._minpack_py"
    and optimize.least_squares.__module__ == "scipy.optimize._lsq.least_squares"
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" curve_fit_module={optimize.curve_fit.__module__}"
    f" least_squares_module={optimize.least_squares.__module__}"
    f" scipy_engine_sha256={SCIPY_ENGINE_SHA256}"
    f" curve_fit_engine_sha256={engine_rows[0][2]}"
    f" least_squares_engine_sha256={engine_rows[1][2]}"
    f" x_sha256={X_SHA256}"
    f" truth_sha256={TRUTH_SHA256}"
    f" y_sha256={Y_SHA256}"
    f" input_sha256={INPUT_SHA256}"
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
    if command == "DATA" and len(fields) == 1:
        values = np.concatenate((X.ravel(), TRUTH.ravel(), Y.ravel()))
        print(
            "DATA " + ",".join(f"{value:.17e}" for value in values),
            flush=True,
        )
    elif command == "CHECK" and len(fields) == 2:
        arm = fields[1]
        raw, active, maximum = run_observed(lambda: execute(arm))
        summary = summarize(raw)
        values = summary[1:]
        payload = ",".join(
            str(value) if isinstance(value, int) else f"{value:.17e}"
            for value in values
        )
        print(f"CHECK {arm} {active} {maximum} {payload}", flush=True)
    elif command == "PARAMS" and len(fields) == 2:
        arm = fields[1]
        popt = summarize(execute(arm))[0]
        print(
            f"PARAMS {arm} "
            + ",".join(f"{value:.17e}" for value in popt.ravel()),
            flush=True,
        )
    elif command == "TIME" and len(fields) == 3:
        arm = fields[1]
        repetitions = int(fields[2])
        if repetitions <= 0:
            raise RuntimeError("repetitions must be positive")
        before = task_runtime()
        checksum = 0.0
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            checksum += summarize(execute(arm))[-1]
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
        successes: usize,
        improved: usize,
        worst_index: usize,
        worst_rss: f64,
        total_rss: f64,
        median_rmse: f64,
        p95_rmse: f64,
        p99_rmse: f64,
        max_rmse: f64,
        median_parameter_error: f64,
        p99_parameter_error: f64,
        nfev: usize,
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

        fn data(&mut self) -> Result<Dataset, String> {
            let response = self.request("DATA", "SciPy fixed dataset")?;
            let payload = response
                .strip_prefix("DATA ")
                .ok_or_else(|| "malformed SciPy DATA response".to_string())?;
            let values = payload
                .split(',')
                .map(|value| parse(value, "dataset value"))
                .collect::<Result<Vec<f64>, _>>()?;
            let expected = POINTS + BATCH * PARAMETERS + BATCH * POINTS;
            if values.len() != expected {
                return Err(format!(
                    "dataset length mismatch: expected {expected}, got {}",
                    values.len()
                ));
            }
            let x_end = POINTS;
            let truth_end = x_end + BATCH * PARAMETERS;
            let x = values[..x_end].to_vec();
            let truth = values[x_end..truth_end]
                .chunks_exact(PARAMETERS)
                .map(<[f64]>::to_vec)
                .collect();
            let y = values[truth_end..]
                .chunks_exact(POINTS)
                .map(<[f64]>::to_vec)
                .collect();
            Ok(Dataset { x, truth, y })
        }

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let response = self.request(&format!("CHECK {arm}"), "SciPy quality check")?;
            let fields = response.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("malformed SciPy check response: {response}"));
            }
            let values = fields[4].split(',').collect::<Vec<_>>();
            if values.len() != 13 {
                return Err(format!("malformed SciPy quality fields: {response}"));
            }
            Ok(ScipyCheck {
                arm: arm.to_string(),
                active_tasks: parse(fields[2], "SciPy active task count")?,
                max_os_tasks: parse(fields[3], "SciPy OS task count")?,
                summary: JobSummary {
                    successes: parse(values[0], "SciPy success count")?,
                    improved: parse(values[1], "SciPy improved count")?,
                    worst_index: parse(values[2], "SciPy worst index")?,
                    worst_rss: parse(values[3], "SciPy worst RSS")?,
                    total_rss: parse(values[4], "SciPy total RSS")?,
                    median_rmse: parse(values[5], "SciPy median RMSE")?,
                    p95_rmse: parse(values[6], "SciPy p95 RMSE")?,
                    p99_rmse: parse(values[7], "SciPy p99 RMSE")?,
                    max_rmse: parse(values[8], "SciPy maximum RMSE")?,
                    median_parameter_error: parse(values[9], "SciPy median parameter error")?,
                    p99_parameter_error: parse(values[10], "SciPy p99 parameter error")?,
                    nfev: parse(values[11], "SciPy function evaluations")?,
                    checksum: parse(values[12], "SciPy checksum")?,
                },
            })
        }

        fn parameters(&mut self, arm: &str) -> Result<Vec<Vec<f64>>, String> {
            let response = self.request(&format!("PARAMS {arm}"), "SciPy fitted parameters")?;
            let payload = response
                .strip_prefix(&format!("PARAMS {arm} "))
                .ok_or_else(|| format!("malformed SciPy parameter response for {arm}"))?;
            let values = payload
                .split(',')
                .map(|value| parse(value, "fitted parameter"))
                .collect::<Result<Vec<f64>, _>>()?;
            if values.len() != BATCH * PARAMETERS {
                return Err(format!(
                    "SciPy parameter length mismatch: expected {}, got {}",
                    BATCH * PARAMETERS,
                    values.len()
                ));
            }
            Ok(values
                .chunks_exact(PARAMETERS)
                .map(<[f64]>::to_vec)
                .collect())
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let response =
                self.request(&format!("TIME {arm} {repetitions}"), "SciPy timing sample")?;
            let fields = response.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" || fields[1] != arm {
                return Err(format!("malformed SciPy timing response: {response}"));
            }
            let elapsed = parse(fields[2], "SciPy elapsed time")?;
            let checksum: f64 = parse(fields[3], "SciPy checksum")?;
            if !checksum.is_finite() || elapsed <= 0.0 {
                return Err(format!("invalid SciPy timing response: {response}"));
            }
            Ok(Timing {
                elapsed,
                active_tasks: parse(fields[4], "SciPy active task count")?,
                max_os_tasks: parse(fields[5], "SciPy OS task count")?,
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
    }

    struct Dataset {
        x: Vec<f64>,
        truth: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
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
            .filter_map(|field| field.split_once('='))
            .collect()
    }

    fn f64_sha256(values: impl IntoIterator<Item = f64>) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn input_sha256(data: &Dataset) -> String {
        let mut digest = Sha256::new();
        for (label, values) in [
            ("x", data.x.iter().copied().collect::<Vec<_>>()),
            (
                "truth",
                data.truth.iter().flatten().copied().collect::<Vec<_>>(),
            ),
            ("y", data.y.iter().flatten().copied().collect::<Vec<_>>()),
        ] {
            digest.update(label.as_bytes());
            digest.update([0]);
            for value in values {
                digest.update(value.to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    }

    fn summarize_parameters(
        data: &Dataset,
        fitted: &[Vec<f64>],
        successes: usize,
        nfev: usize,
    ) -> Result<JobSummary, String> {
        if fitted.len() != BATCH
            || fitted
                .iter()
                .any(|row| row.len() != PARAMETERS || row.iter().any(|value| !value.is_finite()))
        {
            return Err("invalid fitted parameter matrix".to_string());
        }
        let initial_prediction = data.x.iter().map(|x| (-x).exp()).collect::<Vec<_>>();
        let mut rss = Vec::with_capacity(BATCH);
        let mut rmse = Vec::with_capacity(BATCH);
        let mut parameter_error = Vec::with_capacity(BATCH);
        let mut improved = 0usize;
        for index in 0..BATCH {
            let parameters = &fitted[index];
            let mut fit_rss = 0.0;
            let mut initial_rss = 0.0;
            for point in 0..POINTS {
                let prediction =
                    parameters[0] * (-parameters[1] * data.x[point]).exp() + parameters[2];
                let fit_residual = prediction - data.y[index][point];
                fit_rss += fit_residual * fit_residual;
                let initial_residual = initial_prediction[point] - data.y[index][point];
                initial_rss += initial_residual * initial_residual;
            }
            improved += usize::from(fit_rss < initial_rss);
            rss.push(fit_rss);
            rmse.push((fit_rss / POINTS as f64).sqrt());
            parameter_error.push(
                ((parameters[0] - data.truth[index][0]).abs() / 2.0)
                    .max((parameters[1] - data.truth[index][1]).abs())
                    .max((parameters[2] - data.truth[index][2]).abs()),
            );
        }
        let worst_index = rss
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "empty residual vector".to_string())?;
        let weights = (1..=BATCH).map(|value| value as f64);
        let checksum = fitted
            .iter()
            .zip(weights.clone())
            .map(|(parameters, weight)| parameters[0] * weight)
            .sum::<f64>()
            + fitted
                .iter()
                .zip(weights.clone())
                .map(|(parameters, weight)| parameters[1] * weight)
                .sum::<f64>()
                * 1.0e-2
            + fitted
                .iter()
                .zip(weights.clone())
                .map(|(parameters, weight)| parameters[2] * weight)
                .sum::<f64>()
                * 1.0e-4
            + rss
                .iter()
                .zip(weights)
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
                * 1.0e-6
            + successes as f64
            + nfev as f64 * 1.0e-9
            + worst_index as f64 * 1.0e-12;
        Ok(JobSummary {
            successes,
            improved,
            worst_index,
            worst_rss: rss[worst_index],
            total_rss: rss.iter().sum(),
            median_rmse: median(rmse.clone()),
            p95_rmse: percentile(rmse.clone(), 0.95),
            p99_rmse: percentile(rmse.clone(), 0.99),
            max_rmse: rmse.iter().copied().fold(0.0, f64::max),
            median_parameter_error: median(parameter_error.clone()),
            p99_parameter_error: percentile(parameter_error, 0.99),
            nfev,
            checksum,
        })
    }

    fn execute_batch(data: &Dataset) -> Result<(JobSummary, Vec<Vec<f64>>), String> {
        let options = CurveFitOptions {
            p0: Some(vec![1.0, 1.0, 0.0]),
            ..CurveFitOptions::default()
        };
        let fitted = curve_fit_many(
            |x, parameters| parameters[0] * (-parameters[1] * x).exp() + parameters[2],
            &data.x,
            &data.y,
            options,
        )
        .map_err(|error| format!("FrankenSciPy curve_fit_many failed: {error}"))?;
        let summary = summarize_parameters(data, &fitted, BATCH, 0)?;
        Ok((summary, fitted))
    }

    fn time_batch(data: &Dataset, repetitions: usize) -> Result<f64, String> {
        let started = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..repetitions {
            checksum += execute_batch(data)?.0.checksum;
        }
        black_box(checksum);
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed <= 0.0 || !checksum.is_finite() {
            return Err("invalid FrankenSciPy timing sample".to_string());
        }
        Ok(elapsed)
    }

    fn basic_integrity(label: &str, summary: JobSummary) -> Result<(), String> {
        if summary.successes > BATCH
            || summary.improved > BATCH
            || summary.worst_index >= BATCH
            || !summary.worst_rss.is_finite()
            || summary.worst_rss < 0.0
            || !summary.total_rss.is_finite()
            || summary.total_rss < 0.0
            || !summary.median_rmse.is_finite()
            || !summary.p95_rmse.is_finite()
            || !summary.p99_rmse.is_finite()
            || !summary.max_rmse.is_finite()
            || summary.median_rmse < 0.0
            || summary.p95_rmse < summary.median_rmse
            || summary.p99_rmse < summary.p95_rmse
            || summary.max_rmse < summary.p99_rmse
            || !summary.median_parameter_error.is_finite()
            || summary.median_parameter_error < 0.0
            || !summary.p99_parameter_error.is_finite()
            || summary.p99_parameter_error < summary.median_parameter_error
            || !summary.checksum.is_finite()
        {
            return Err(format!("{label} failed result integrity: {summary:?}"));
        }
        Ok(())
    }

    fn quality_eligible(label: &str, summary: JobSummary) -> bool {
        basic_integrity(label, summary).is_ok()
            && summary.successes == BATCH
            && summary.improved == BATCH
            && summary.max_rmse <= MAX_FIT_RMSE
    }

    fn cross_quality(
        data: &Dataset,
        ours: &[Vec<f64>],
        ours_summary: JobSummary,
        scipy: &[Vec<f64>],
        scipy_summary: JobSummary,
    ) -> Result<f64, String> {
        if !quality_eligible("FrankenSciPy", ours_summary)
            || !quality_eligible("selected SciPy", scipy_summary)
        {
            return Err(format!(
                "scientific quality gate failed: ours={ours_summary:?} scipy={scipy_summary:?}"
            ));
        }
        if ours.len() != BATCH || scipy.len() != BATCH {
            return Err("cross-quality parameter length mismatch".to_string());
        }
        let mut maximum_curve_rmse = 0.0f64;
        for index in 0..BATCH {
            if ours[index].len() != PARAMETERS || scipy[index].len() != PARAMETERS {
                return Err("cross-quality parameter width mismatch".to_string());
            }
            let mut squared = 0.0;
            for x in &data.x {
                let ours_value = ours[index][0] * (-ours[index][1] * x).exp() + ours[index][2];
                let scipy_value = scipy[index][0] * (-scipy[index][1] * x).exp() + scipy[index][2];
                squared += (ours_value - scipy_value).powi(2);
            }
            maximum_curve_rmse = maximum_curve_rmse.max((squared / POINTS as f64).sqrt());
        }
        if maximum_curve_rmse > MAX_CROSS_CURVE_RMSE {
            return Err(format!(
                "fitted-curve disagreement {maximum_curve_rmse:.17e} exceeds \
                 {MAX_CROSS_CURVE_RMSE:.17e}"
            ));
        }
        if ours_summary.worst_index != scipy_summary.worst_index {
            let worst_difference = (ours_summary.worst_rss - scipy_summary.worst_rss).abs();
            let worst_scale = ours_summary.worst_rss.max(scipy_summary.worst_rss);
            if worst_difference > 0.01 * worst_scale {
                return Err(format!(
                    "worst-fit traces disagree: ours={} scipy={} \
                     rss_difference={worst_difference:.17e} scale={worst_scale:.17e}",
                    ours_summary.worst_index, scipy_summary.worst_index
                ));
            }
        }
        Ok(maximum_curve_rmse)
    }

    fn print_summary(label: &str, summary: JobSummary) {
        println!(
            "{label}: successes={}/{} improved={}/{} worst_index={} \
             worst_rss={:.17e} total_rss={:.17e} median_rmse={:.17e} \
             p95_rmse={:.17e} p99_rmse={:.17e} max_rmse={:.17e} \
             median_parameter_error={:.17e} p99_parameter_error={:.17e} \
             nfev={} checksum={:.17e}",
            summary.successes,
            BATCH,
            summary.improved,
            BATCH,
            summary.worst_index,
            summary.worst_rss,
            summary.total_rss,
            summary.median_rmse,
            summary.p95_rmse,
            summary.p99_rmse,
            summary.max_rmse,
            summary.median_parameter_error,
            summary.p99_parameter_error,
            summary.nfev,
            summary.checksum
        );
    }

    fn observe_batch_workers(data: &Dataset, expected: usize) -> Result<usize, String> {
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
            black_box(execute_batch(data)?);
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

    fn screen_arm(scipy: &mut Scipy, arm: &str) -> Result<ScreenedArm, String> {
        let check = scipy.check(arm)?;
        print_summary(&format!("screen SciPy {arm}"), check.summary);
        if !quality_eligible(&format!("SciPy {arm}"), check.summary) {
            return Err(format!("SciPy public arm {arm} failed the scientific gate"));
        }
        let mut samples = Vec::with_capacity(SCREEN_ROUNDS);
        let mut max_active_tasks = check.active_tasks;
        let mut max_os_tasks = check.max_os_tasks;
        for round in 0..SCREEN_ROUNDS {
            let timing = scipy.time(arm, 1)?;
            max_active_tasks = max_active_tasks.max(timing.active_tasks);
            max_os_tasks = max_os_tasks.max(timing.max_os_tasks);
            samples.push(timing.elapsed);
            println!(
                "screen arm={arm} round={round} seconds={:.9} \
                 observed_active_tasks={} observed_os_tasks={}",
                timing.elapsed, timing.active_tasks, timing.max_os_tasks
            );
        }
        let arm_median = median(samples.clone());
        println!(
            "screen_result arm={arm} median_seconds={arm_median:.9} \
             samples={} observed_active_tasks={max_active_tasks} \
             observed_os_tasks={max_os_tasks}",
            csv(&samples)
        );
        Ok(ScreenedArm {
            check,
            median: arm_median,
            max_active_tasks,
            max_os_tasks,
        })
    }

    fn screen_public_arms(scipy: &mut Scipy) -> Result<Vec<ScreenedArm>, String> {
        SCIPY_ARMS
            .iter()
            .map(|arm| screen_arm(scipy, arm))
            .collect()
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
        data: &Dataset,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_batch(data, repetitions)?;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_batch(data, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
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
        };
        for round in 0..rounds {
            let (ours, incumbent, effect_active, effect_tasks, ours_null, scipy_null) =
                match round % 3 {
                    0 => {
                        let effect = effect_pair(scipy, arm, data, repetitions, round)?;
                        let ours_null = ours_null_pair(data, repetitions, round)?;
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
                        let effect = effect_pair(scipy, arm, data, repetitions, round)?;
                        let ours_null = ours_null_pair(data, repetitions, round)?;
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
                        let ours_null = ours_null_pair(data, repetitions, round)?;
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        let effect = effect_pair(scipy, arm, data, repetitions, round)?;
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
                "usage: perf_curve_fit_many_scipy [rounds] [repetitions] [--smoke]".to_string(),
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
             actual_affinity_count={} runtime_isa={} rustc={} \
             elf_path={} elf_sha256={} source_commit={} builder_identity={} \
             build_route={} trj_booking_claim_message_id={} target_dir_policy=shared_reused",
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

        let (mut scipy, ready) = Scipy::start()?;
        println!("SCIPY_IDENTITY {ready}");
        let identity = fields(&ready);
        if !ready.starts_with("READY ")
            || identity.get("scipy") != Some(&"1.17.1")
            || identity.get("genuine") != Some(&"True")
            || identity.get("fsci_loaded") != Some(&"False")
            || identity.get("curve_fit_module") != Some(&"scipy.optimize._minpack_py")
            || identity.get("least_squares_module") != Some(&"scipy.optimize._lsq.least_squares")
        {
            return Err(format!("live SciPy identity gate failed: {ready}"));
        }
        for field in [
            "scipy_engine_sha256",
            "curve_fit_engine_sha256",
            "least_squares_engine_sha256",
            "x_sha256",
            "truth_sha256",
            "y_sha256",
            "input_sha256",
        ] {
            require_hex_sha(
                identity
                    .get(field)
                    .ok_or_else(|| format!("SciPy identity omitted {field}"))?,
                field,
            )?;
        }
        let data = scipy.data()?;
        let x_sha256 = f64_sha256(data.x.iter().copied());
        let truth_sha256 = f64_sha256(data.truth.iter().flatten().copied());
        let y_sha256 = f64_sha256(data.y.iter().flatten().copied());
        let rust_input_sha256 = input_sha256(&data);
        for (field, observed) in [
            ("x_sha256", x_sha256.as_str()),
            ("truth_sha256", truth_sha256.as_str()),
            ("y_sha256", y_sha256.as_str()),
            ("input_sha256", rust_input_sha256.as_str()),
        ] {
            if identity.get(field) != Some(&observed) {
                return Err(format!(
                    "{field} mismatch: rust={observed} scipy={:?}",
                    identity.get(field)
                ));
            }
        }
        println!(
            "FIXTURE batch={} points={} x_start={:.17e} x_end={:.17e} \
             x_sha256={} truth_sha256={} y_sha256={} input_sha256={} \
             p0=1,1,0 noise_half_width=0.01",
            BATCH,
            POINTS,
            data.x[0],
            data.x[POINTS - 1],
            x_sha256,
            truth_sha256,
            y_sha256,
            rust_input_sha256
        );

        let (ours_summary, ours_parameters) = execute_batch(&data)?;
        print_summary("FrankenSciPy", ours_summary);
        if !quality_eligible("FrankenSciPy", ours_summary) {
            return Err(format!(
                "FrankenSciPy failed the scientific gate: {ours_summary:?}"
            ));
        }
        let expected_workers = affinity.len().min(BATCH);
        let observed_rust_workers = observe_batch_workers(&data, expected_workers)?;
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
        let selected_parameters = scipy.parameters(&selected.check.arm)?;
        let maximum_cross_curve_rmse = cross_quality(
            &data,
            &ours_parameters,
            ours_summary,
            &selected_parameters,
            selected.check.summary,
        )?;
        println!(
            "selected_public_scipy_arm={} selected_screen_median_seconds={:.9} \
             selected_observed_active_tasks={} selected_observed_os_tasks={} \
             max_cross_curve_rmse={maximum_cross_curve_rmse:.17e}",
            selected.check.arm, selected.median, selected.max_active_tasks, selected.max_os_tasks
        );
        let numeric_scalar_median = screened
            .iter()
            .find(|entry| entry.check.arm == "curve_fit_numeric_scalar")
            .map(|entry| entry.median)
            .ok_or_else(|| "numeric scalar curve_fit screen result missing".to_string())?;
        let incumbent_mechanism_ratio = numeric_scalar_median / selected.median;
        println!(
            "mechanism_screen numeric_scalar_over_selected={incumbent_mechanism_ratio:.9}x \
             predicted_above_8={} numeric_scalar_seconds={numeric_scalar_median:.9} \
             selected_seconds={:.9}",
            incumbent_mechanism_ratio >= 8.0,
            selected.median
        );

        if smoke {
            println!(
                "NON_EVIDENCE_SMOKE_COMPLETE selected_public_scipy_arm={} \
                 numeric_scalar_over_selected={incumbent_mechanism_ratio:.9}x \
                 actual_observed_rust_workers={} no_effect_verdict=true",
                selected.check.arm, observed_rust_workers
            );
            scipy.stop()?;
            return Ok(());
        }

        require_host_wide_quiescence("before_effect")?;
        let measurement = measure(&mut scipy, &selected.check.arm, &data, rounds, repetitions)?;
        let decision = print_decision(&selected.check.arm, &measurement, repetitions);
        println!(
            "ACTUAL_OBSERVED selected_scipy_active_tasks={} \
             selected_scipy_max_os_tasks={} requested_worker_capacity_not_substituted=true",
            measurement.max_scipy_active_tasks, measurement.max_scipy_os_tasks
        );

        let p1 = selected.check.arm == "curve_fit_jac_pool";
        let p2 = decision.ratio_high < COLLAPSE_BOUNDARY;
        let p3 = decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let p4 = incumbent_mechanism_ratio >= 8.0;
        let selected_is_pool = selected.check.arm.ends_with("_pool");
        let p5_rust = observed_rust_workers == 32;
        let p5_scipy = !selected_is_pool || measurement.max_scipy_active_tasks > 1;
        println!(
            "PREREGISTERED_PREDICTIONS P1_analytic_pool_fastest={} \
             P2_ci_upper_below_11_3={} P3_ci_lower_above_3={} \
             P4_numeric_scalar_over_selected_at_least_8={} \
             P5_rust_32_workers={} P5_selected_pool_observed_multiple_tasks={} \
             selected_is_pool={} ratio_median={:.9} ratio_ci=[{:.9},{:.9}]",
            p1,
            p2,
            p3,
            p4,
            p5_rust,
            p5_scipy,
            selected_is_pool,
            decision.ratio_median,
            decision.ratio_low,
            decision.ratio_high
        );
        let choose_frankenscipy = decision.decidable && decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if choose_frankenscipy {
            "FrankenSciPy curve_fit_many"
        } else {
            &selected.check.arm
        };
        println!(
            "CHOOSER STATEMENT: choose {chooser} for this exact 2,000-trace \
             exponential-decay fitting study; durable_frankenscipy_boundary=3x \
             outcome={} ratio_ci_low={:.9} old_113x_self_speedup_retired=true",
            decision.outcome, decision.ratio_low
        );
        scipy.stop()?;
        Ok(())
    }
}

#[cfg(feature = "opt-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("perf_curve_fit_many_scipy failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "opt-incumbent-bench"))]
fn main() {
    eprintln!("perf_curve_fit_many_scipy requires --features opt-incumbent-bench");
    std::process::exit(2);
}
