//! Whole 100-parameter triple-integration job versus screened SciPy 1.17.1.
//!
//! The historical `tplquad_many` row compared against a scalar Python loop.
//! This harness screens SciPy's public vector-valued routes, freezes the
//! strongest valid incumbent, and applies the corrected dual-null median gate.

#[cfg(feature = "tplquad-incumbent-bench")]
mod bench {
    use fsci_integrate::{DblquadOptions, DblquadResult, tplquad_many};
    use fsci_runtime::scipy_incumbent::{PINNED_NUMPY, PINNED_SCIPY, ScipyIncumbent};
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

    const BATCH: usize = 100;
    const EPSABS: f64 = 1.49e-8;
    const EPSREL: f64 = 1.49e-8;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 5;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const MAX_SCALED_REFERENCE_ERROR: f64 = 4.0;
    const COLLAPSE_BOUNDARY: f64 = 15.9;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const BASE_SCIPY_ARMS: [&str; 4] = [
        "tplquad_scalar",
        "quad_vec",
        "cubature_gk21",
        "cubature_genz_malik",
    ];

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import math
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy
from scipy import integrate, special
from scipy.integrate import _cubature, _quad_vec, _quadpack_py

BATCH = 100
EPSABS = 1.49e-8
EPSREL = 1.49e-8
PARAMS = np.asarray(
    [2.0 + 13.0 * i / (BATCH - 1) for i in range(BATCH)],
    dtype="<f8",
)
ROOT = np.sqrt(np.pi) / (2.0 * np.sqrt(PARAMS)) * special.erf(np.sqrt(PARAMS))
REFERENCE = np.power(ROOT, 3).astype("<f8")
PARAMS_SHA256 = hashlib.sha256(PARAMS.tobytes(order="C")).hexdigest()
REFERENCE_SHA256 = hashlib.sha256(REFERENCE.tobytes(order="C")).hexdigest()
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
POOL = ThreadPool(WORKER_CAPACITY)
WORKER_RULE = "genz-malik"

def scalar_job():
    values = []
    errors = []
    for p in PARAMS:
        value, error = integrate.tplquad(
            lambda z, y, x, p=float(p): math.exp(
                -p * (x * x + y * y + z * z)
            ),
            0.0,
            1.0,
            lambda _x: 0.0,
            lambda _x: 1.0,
            lambda _x, _y: 0.0,
            lambda _x, _y: 1.0,
            epsabs=EPSABS,
            epsrel=EPSREL,
        )
        values.append(value)
        errors.append(error)
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        True,
        0,
    )

def quad_vec_job():
    success = True
    errors = []
    evaluations = 0

    def middle(x):
        nonlocal success, evaluations

        def inner(y):
            nonlocal success, evaluations
            value, error, info = integrate.quad_vec(
                lambda z: np.exp(-PARAMS * (x * x + y * y + z * z)),
                0.0,
                1.0,
                epsabs=EPSABS,
                epsrel=EPSREL,
                norm="max",
                full_output=True,
            )
            success = success and bool(info.success)
            errors.append(float(error))
            evaluations += int(info.neval)
            return value

        value, error, info = integrate.quad_vec(
            inner,
            0.0,
            1.0,
            epsabs=EPSABS,
            epsrel=EPSREL,
            norm="max",
            full_output=True,
        )
        success = success and bool(info.success)
        errors.append(float(error))
        evaluations += int(info.neval)
        return value

    value, error, info = integrate.quad_vec(
        middle,
        0.0,
        1.0,
        epsabs=EPSABS,
        epsrel=EPSREL,
        norm="max",
        full_output=True,
    )
    success = success and bool(info.success)
    errors.append(float(error))
    evaluations += int(info.neval)
    combined_error = sum(errors)
    return (
        np.asarray(value, dtype=np.float64),
        np.full(BATCH, combined_error, dtype=np.float64),
        success,
        evaluations,
    )

def vector_integrand(points):
    radius_squared = (
        np.square(points[:, 0])
        + np.square(points[:, 1])
        + np.square(points[:, 2])
    )
    return np.exp(-radius_squared[:, None] * PARAMS[None, :])

def cubature_job(rule, workers):
    result = integrate.cubature(
        vector_integrand,
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 1.0]),
        rule=rule,
        rtol=EPSREL,
        atol=EPSABS,
        workers=workers,
    )
    return (
        np.asarray(result.estimate, dtype=np.float64),
        np.asarray(result.error, dtype=np.float64),
        result.status == "converged",
        int(result.subdivisions),
    )

def execute(arm):
    if arm == "tplquad_scalar":
        return scalar_job()
    if arm == "quad_vec":
        return quad_vec_job()
    if arm == "cubature_gk21":
        return cubature_job("gk21", 1)
    if arm == "cubature_genz_malik":
        return cubature_job("genz-malik", 1)
    if arm == "cubature_workers":
        return cubature_job(WORKER_RULE, POOL.map)
    raise RuntimeError(f"unknown arm {arm!r}")

def validate_raw(raw):
    values, errors, converged, work = raw
    if values.shape != (BATCH,) or errors.shape != (BATCH,):
        raise RuntimeError(
            f"unexpected output shapes values={values.shape} errors={errors.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite integral value")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise RuntimeError("invalid error estimate")
    if work < 0:
        raise RuntimeError("negative work count")
    return values, errors, bool(converged), int(work)

def whole_job_checksum(raw):
    values, errors, converged, work = validate_raw(raw)
    weights = np.arange(1, BATCH + 1, dtype=np.float64)
    best_index = int(np.argmax(values))
    return float(
        np.dot(values, weights)
        + np.dot(errors, weights) * 1.0e-3
        + best_index * 1.0e-4
        + int(converged)
        + work * 1.0e-9
    )

def quality_fields(raw):
    values, errors, converged, work = validate_raw(raw)
    best_index = int(np.argmax(values))
    scale = EPSABS + EPSREL * np.abs(REFERENCE)
    absolute = np.abs(values - REFERENCE)
    return (
        BATCH if converged else 0,
        best_index,
        float(PARAMS[best_index]),
        float(values[best_index]),
        float(np.min(errors)),
        float(np.max(errors)),
        float(np.max(absolute)),
        float(np.max(absolute / scale)),
        work,
        whole_job_checksum(raw),
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
    ("tplquad",) + source_hash(_quadpack_py),
    ("quad_vec",) + source_hash(_quad_vec),
    ("cubature",) + source_hash(_cubature),
]
engine_digest = hashlib.sha256()
for label, _path, _digest, source in engine_rows:
    engine_digest.update(label.encode("ascii"))
    engine_digest.update(b"\0")
    engine_digest.update(source)
SCIPY_ENGINE_SHA256 = engine_digest.hexdigest()

for arm in (
    "tplquad_scalar",
    "quad_vec",
    "cubature_gk21",
    "cubature_genz_malik",
    "cubature_workers",
):
    raw = execute(arm)
    if not math.isfinite(whole_job_checksum(raw)):
        raise RuntimeError(f"{arm} warmup failed")

fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and np.__version__ == "2.4.3"
    and integrate.tplquad.__module__ == "scipy.integrate._quadpack_py"
    and integrate.quad_vec.__module__ == "scipy.integrate._quad_vec"
    and integrate.cubature.__module__ == "scipy.integrate._cubature"
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" tplquad_module={integrate.tplquad.__module__}"
    f" quad_vec_module={integrate.quad_vec.__module__}"
    f" cubature_module={integrate.cubature.__module__}"
    f" scipy_engine_sha256={SCIPY_ENGINE_SHA256}"
    f" tplquad_engine_sha256={engine_rows[0][2]}"
    f" quad_vec_engine_sha256={engine_rows[1][2]}"
    f" cubature_engine_sha256={engine_rows[2][2]}"
    f" params_sha256={PARAMS_SHA256}"
    f" reference_sha256={REFERENCE_SHA256}"
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
    if command == "REFERENCE" and len(fields) == 1:
        print(
            "REFERENCE "
            + ",".join(f"{value:.17e}" for value in REFERENCE),
            flush=True,
        )
    elif command == "SET_WORKER_RULE" and len(fields) == 2:
        rule = fields[1]
        if rule not in ("gk21", "genz-malik"):
            raise RuntimeError(f"invalid worker rule {rule!r}")
        WORKER_RULE = rule
        raw = execute("cubature_workers")
        if not math.isfinite(whole_job_checksum(raw)):
            raise RuntimeError("workers warmup failed")
        print(f"WORKER_RULE {WORKER_RULE}", flush=True)
    elif command == "CHECK" and len(fields) == 2:
        arm = fields[1]
        raw, active, maximum = run_observed(lambda: execute(arm))
        output = quality_fields(raw)
        values = ",".join(
            str(value) if isinstance(value, int) else f"{value:.17e}"
            for value in output
        )
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
            checksum += whole_job_checksum(execute(arm))
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
        converged: usize,
        best_index: usize,
        best_p: f64,
        best_integral: f64,
        min_error_estimate: f64,
        max_error_estimate: f64,
        max_abs_reference_error: f64,
        max_scaled_reference_error: f64,
        work: usize,
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

        fn reference(&mut self) -> Result<Vec<f64>, String> {
            let response = self.request("REFERENCE", "SciPy closed-form reference")?;
            let values = response
                .strip_prefix("REFERENCE ")
                .ok_or_else(|| format!("malformed reference response: {response}"))?
                .split(',')
                .map(|value| parse(value, "reference value"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != BATCH {
                return Err(format!(
                    "reference length mismatch: expected {BATCH}, got {}",
                    values.len()
                ));
            }
            Ok(values)
        }

        fn set_worker_rule(&mut self, rule: &str) -> Result<(), String> {
            let response =
                self.request(&format!("SET_WORKER_RULE {rule}"), "SciPy workers-map rule")?;
            if response != format!("WORKER_RULE {rule}") {
                return Err(format!("malformed worker-rule response: {response}"));
            }
            Ok(())
        }

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let response = self.request(&format!("CHECK {arm}"), "SciPy quality check")?;
            let fields = response.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("malformed SciPy check response: {response}"));
            }
            let values = fields[4].split(',').collect::<Vec<_>>();
            if values.len() != 10 {
                return Err(format!("malformed SciPy quality fields: {response}"));
            }
            Ok(ScipyCheck {
                arm: arm.to_string(),
                active_tasks: parse(fields[2], "SciPy active task count")?,
                max_os_tasks: parse(fields[3], "SciPy OS task count")?,
                summary: JobSummary {
                    converged: parse(values[0], "SciPy converged count")?,
                    best_index: parse(values[1], "SciPy best index")?,
                    best_p: parse(values[2], "SciPy best parameter")?,
                    best_integral: parse(values[3], "SciPy best integral")?,
                    min_error_estimate: parse(values[4], "SciPy minimum error")?,
                    max_error_estimate: parse(values[5], "SciPy maximum error")?,
                    max_abs_reference_error: parse(values[6], "SciPy absolute error")?,
                    max_scaled_reference_error: parse(values[7], "SciPy scaled error")?,
                    work: parse(values[8], "SciPy work count")?,
                    checksum: parse(values[9], "SciPy checksum")?,
                },
            })
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

    fn parameters() -> Vec<Vec<f64>> {
        (0..BATCH)
            .map(|index| vec![2.0 + 13.0 * index as f64 / (BATCH - 1) as f64])
            .collect()
    }

    fn parameter_values(parameters: &[Vec<f64>]) -> Vec<f64> {
        parameters.iter().map(|row| row[0]).collect()
    }

    fn f64_sha256(values: &[f64]) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn execute_batch(
        parameters: &[Vec<f64>],
        reference: &[f64],
    ) -> Result<(JobSummary, Vec<f64>), String> {
        let options = DblquadOptions {
            epsabs: EPSABS,
            epsrel: EPSREL,
            limit: 50,
        };
        let output = tplquad_many(
            |z, y, x, parameter| (-parameter[0] * (x * x + y * y + z * z)).exp(),
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            parameters,
            options,
        );
        if output.len() != BATCH {
            return Err(format!(
                "FrankenSciPy output length mismatch: expected {BATCH}, got {}",
                output.len()
            ));
        }
        let mut values = Vec::with_capacity(BATCH);
        let mut errors = Vec::with_capacity(BATCH);
        let mut converged = 0usize;
        for result in output {
            let DblquadResult {
                integral,
                error,
                converged: did_converge,
            } = result.map_err(|error| format!("FrankenSciPy tplquad_many failed: {error}"))?;
            if !integral.is_finite() || !error.is_finite() || error < 0.0 {
                return Err("FrankenSciPy produced invalid integral/error output".to_string());
            }
            values.push(integral);
            errors.push(error);
            converged += usize::from(did_converge);
        }
        let best_index = values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "FrankenSciPy returned no values".to_string())?;
        let mut max_abs_reference_error = 0.0f64;
        let mut max_scaled_reference_error = 0.0f64;
        for (value, expected) in values.iter().zip(reference) {
            let absolute = (value - expected).abs();
            let scale = EPSABS + EPSREL * expected.abs();
            max_abs_reference_error = max_abs_reference_error.max(absolute);
            max_scaled_reference_error = max_scaled_reference_error.max(absolute / scale);
        }
        let weights = (1..=BATCH).map(|value| value as f64);
        let checksum = values
            .iter()
            .zip(weights.clone())
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            + errors
                .iter()
                .zip(weights)
                .map(|(error, weight)| error * weight)
                .sum::<f64>()
                * 1.0e-3
            + best_index as f64 * 1.0e-4
            + usize::from(converged == BATCH) as f64;
        Ok((
            JobSummary {
                converged,
                best_index,
                best_p: parameters[best_index][0],
                best_integral: values[best_index],
                min_error_estimate: errors.iter().copied().fold(f64::INFINITY, f64::min),
                max_error_estimate: errors.iter().copied().fold(0.0, f64::max),
                max_abs_reference_error,
                max_scaled_reference_error,
                work: 0,
                checksum,
            },
            values,
        ))
    }

    fn time_batch(
        parameters: &[Vec<f64>],
        reference: &[f64],
        repetitions: usize,
    ) -> Result<f64, String> {
        let started = Instant::now();
        let mut checksum = 0.0;
        for _ in 0..repetitions {
            checksum += execute_batch(parameters, reference)?.0.checksum;
        }
        black_box(checksum);
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed <= 0.0 || !checksum.is_finite() {
            return Err("invalid FrankenSciPy timing sample".to_string());
        }
        Ok(elapsed)
    }

    fn basic_integrity(label: &str, summary: JobSummary) -> Result<(), String> {
        if summary.best_index >= BATCH
            || !summary.best_p.is_finite()
            || !summary.best_integral.is_finite()
            || !summary.min_error_estimate.is_finite()
            || !summary.max_error_estimate.is_finite()
            || summary.min_error_estimate < 0.0
            || summary.max_error_estimate < summary.min_error_estimate
            || !summary.max_abs_reference_error.is_finite()
            || summary.max_abs_reference_error < 0.0
            || !summary.max_scaled_reference_error.is_finite()
            || summary.max_scaled_reference_error < 0.0
            || !summary.checksum.is_finite()
        {
            return Err(format!("{label} failed result integrity: {summary:?}"));
        }
        Ok(())
    }

    fn quality_eligible(label: &str, summary: JobSummary) -> bool {
        match basic_integrity(label, summary) {
            Ok(()) => {
                summary.converged == BATCH
                    && summary.best_index == 0
                    && summary.max_scaled_reference_error <= MAX_SCALED_REFERENCE_ERROR
            }
            Err(_) => false,
        }
    }

    fn prove_quality(ours: JobSummary, scipy: &ScipyCheck) -> Result<(), String> {
        basic_integrity("FrankenSciPy", ours)?;
        basic_integrity(&format!("SciPy {}", scipy.arm), scipy.summary)?;
        if !quality_eligible("FrankenSciPy", ours)
            || !quality_eligible(&format!("SciPy {}", scipy.arm), scipy.summary)
        {
            return Err(format!(
                "scientific quality gate failed: ours={ours:?} scipy={:?}",
                scipy.summary
            ));
        }
        let agreement_limit = (EPSABS + EPSREL * ours.best_integral.abs())
            + (EPSABS + EPSREL * scipy.summary.best_integral.abs());
        let difference = (ours.best_integral - scipy.summary.best_integral).abs();
        if ours.best_index != scipy.summary.best_index || difference > agreement_limit {
            return Err(format!(
                "selected maxima disagree: ours_index={} scipy_index={} \
                 difference={difference:.17e} limit={agreement_limit:.17e}",
                ours.best_index, scipy.summary.best_index
            ));
        }
        Ok(())
    }

    fn print_summary(label: &str, summary: JobSummary) {
        println!(
            "{label}: converged={}/{} best_index={} best_p={:.17e} \
             best_integral={:.17e} min_error_estimate={:.17e} \
             max_error_estimate={:.17e} max_abs_reference_error={:.17e} \
             max_scaled_reference_error={:.9} work={} checksum={:.17e}",
            summary.converged,
            BATCH,
            summary.best_index,
            summary.best_p,
            summary.best_integral,
            summary.min_error_estimate,
            summary.max_error_estimate,
            summary.max_abs_reference_error,
            summary.max_scaled_reference_error,
            summary.work,
            summary.checksum
        );
    }

    fn observe_batch_workers(
        parameters: &[Vec<f64>],
        reference: &[f64],
        expected: usize,
    ) -> Result<usize, String> {
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
        for _ in 0..6 {
            black_box(execute_batch(parameters, reference)?);
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
        let mut screened = Vec::with_capacity(BASE_SCIPY_ARMS.len() + 1);
        for arm in BASE_SCIPY_ARMS {
            screened.push(screen_arm(scipy, arm)?);
        }
        let worker_rule = screened
            .iter()
            .filter(|entry| {
                entry.check.arm == "cubature_gk21" || entry.check.arm == "cubature_genz_malik"
            })
            .min_by(|left, right| left.median.total_cmp(&right.median))
            .map(|entry| {
                entry
                    .check
                    .arm
                    .strip_prefix("cubature_")
                    .expect("cubature arm prefix")
                    .replace('_', "-")
            })
            .ok_or_else(|| "no valid single-worker cubature rule".to_string())?;
        scipy.set_worker_rule(&worker_rule)?;
        println!("workers_map_rule_frozen={worker_rule}");
        screened.push(screen_arm(scipy, "cubature_workers")?);
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
    }

    fn effect_pair(
        scipy: &mut Scipy,
        arm: &str,
        parameters: &[Vec<f64>],
        reference: &[f64],
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_batch(parameters, reference, repetitions)?;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_batch(parameters, reference, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        }
    }

    fn ours_null_pair(
        parameters: &[Vec<f64>],
        reference: &[f64],
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_batch(parameters, reference, repetitions)?,
                time_batch(parameters, reference, repetitions)?,
            )
        } else {
            let right = time_batch(parameters, reference, repetitions)?;
            let left = time_batch(parameters, reference, repetitions)?;
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
        parameters: &[Vec<f64>],
        reference: &[f64],
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(scipy, arm, parameters, reference, repetitions, warmup)?;
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
                        let effect =
                            effect_pair(scipy, arm, parameters, reference, repetitions, round)?;
                        let ours_null = ours_null_pair(parameters, reference, repetitions, round)?;
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
                        let effect =
                            effect_pair(scipy, arm, parameters, reference, repetitions, round)?;
                        let ours_null = ours_null_pair(parameters, reference, repetitions, round)?;
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
                        let ours_null = ours_null_pair(parameters, reference, repetitions, round)?;
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        let effect =
                            effect_pair(scipy, arm, parameters, reference, repetitions, round)?;
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
            return Err(
                "usage: perf_tplquad_many_scipy [rounds] [repetitions] [--smoke]".to_string(),
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
            report_host_wide_quiescence("before_screen")?;
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

        let parameters = parameters();
        let parameter_scalars = parameter_values(&parameters);
        let parameter_sha256 = f64_sha256(&parameter_scalars);
        let (mut scipy, ready) = Scipy::start()?;
        println!("SCIPY_IDENTITY {ready}");
        let identity = fields(&ready);
        if !ready.starts_with("READY ")
            || identity.get("scipy") != Some(&PINNED_SCIPY)
            || identity.get("numpy") != Some(&PINNED_NUMPY)
            || identity.get("genuine") != Some(&"True")
            || identity.get("fsci_loaded") != Some(&"False")
            || identity.get("tplquad_module") != Some(&"scipy.integrate._quadpack_py")
            || identity.get("quad_vec_module") != Some(&"scipy.integrate._quad_vec")
            || identity.get("cubature_module") != Some(&"scipy.integrate._cubature")
        {
            return Err(format!("live SciPy identity gate failed: {ready}"));
        }
        let scipy_engine_sha256 = identity
            .get("scipy_engine_sha256")
            .ok_or_else(|| "SciPy identity omitted engine SHA-256".to_string())?;
        require_hex_sha(scipy_engine_sha256, "SciPy engine SHA-256")?;
        if identity.get("params_sha256") != Some(&parameter_sha256.as_str()) {
            return Err(format!(
                "parameter hash mismatch: rust={parameter_sha256} scipy={:?}",
                identity.get("params_sha256")
            ));
        }
        let reference = scipy.reference()?;
        let reference_sha256 = f64_sha256(&reference);
        if identity.get("reference_sha256") != Some(&reference_sha256.as_str()) {
            return Err(format!(
                "reference hash mismatch: rust={reference_sha256} scipy={:?}",
                identity.get("reference_sha256")
            ));
        }
        println!(
            "FIXTURE batch={} p_start={:.17e} p_end={:.17e} \
             params_sha256={} reference_sha256={} epsabs={:.17e} epsrel={:.17e}",
            BATCH,
            parameter_scalars[0],
            parameter_scalars[BATCH - 1],
            parameter_sha256,
            reference_sha256,
            EPSABS,
            EPSREL
        );

        let (ours, _ours_values) = execute_batch(&parameters, &reference)?;
        print_summary("FrankenSciPy", ours);
        if !quality_eligible("FrankenSciPy", ours) {
            return Err(format!("FrankenSciPy failed the scientific gate: {ours:?}"));
        }
        let expected_workers = affinity.len().min(BATCH);
        let observed_rust_workers =
            observe_batch_workers(&parameters, &reference, expected_workers)?;
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
        prove_quality(ours, &selected.check)?;
        println!(
            "selected_public_scipy_arm={} selected_screen_median_seconds={:.9} \
             selected_observed_active_tasks={} selected_observed_os_tasks={}",
            selected.check.arm, selected.median, selected.max_active_tasks, selected.max_os_tasks
        );
        let scalar_median = screened
            .iter()
            .find(|entry| entry.check.arm == "tplquad_scalar")
            .map(|entry| entry.median)
            .ok_or_else(|| "scalar tplquad screen result missing".to_string())?;
        let callback_tax_ratio = scalar_median / selected.median;
        println!(
            "mechanism_screen scalar_tplquad_over_selected={callback_tax_ratio:.9}x \
             predicted_above_10={} scalar_seconds={scalar_median:.9} \
             selected_seconds={:.9}",
            callback_tax_ratio > 10.0,
            selected.median
        );

        if smoke {
            println!(
                "NON_EVIDENCE_SMOKE_COMPLETE selected_public_scipy_arm={} \
                 scalar_over_selected={callback_tax_ratio:.9}x \
                 actual_observed_rust_workers={} no_effect_verdict=true",
                selected.check.arm, observed_rust_workers
            );
            scipy.stop()?;
            return Ok(());
        }

        report_host_wide_quiescence("before_effect")?;
        let measurement = measure(
            &mut scipy,
            &selected.check.arm,
            &parameters,
            &reference,
            rounds,
            repetitions,
        )?;
        let decision = print_decision(&selected.check.arm, &measurement, repetitions);
        println!(
            "ACTUAL_OBSERVED selected_scipy_active_tasks={} \
             selected_scipy_max_os_tasks={} requested_worker_capacity_not_substituted=true",
            measurement.max_scipy_active_tasks, measurement.max_scipy_os_tasks
        );

        let p1 = selected.check.arm == "cubature_genz_malik";
        let p2 = decision.ratio_high < COLLAPSE_BOUNDARY;
        let p3 = decision.ratio_high < 1.0;
        let p4 = callback_tax_ratio > 10.0;
        let p5_rust = observed_rust_workers == 32;
        let p5_scipy = measurement.max_scipy_active_tasks == 1;
        println!(
            "PREREGISTERED_PREDICTIONS P1_genz_malik_fastest={} \
             P2_ci_upper_below_15_9={} P3_scipy_wins={} \
             P4_scalar_over_selected_above_10={} \
             P5_rust_32_workers={} P5_selected_scipy_one_active_task={} \
             ratio_median={:.9} ratio_ci=[{:.9},{:.9}]",
            p1,
            p2,
            p3,
            p4,
            p5_rust,
            p5_scipy,
            decision.ratio_median,
            decision.ratio_low,
            decision.ratio_high
        );
        let choose_frankenscipy = decision.decidable && decision.ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if choose_frankenscipy {
            "FrankenSciPy tplquad_many"
        } else {
            &selected.check.arm
        };
        println!(
            "CHOOSER STATEMENT: choose {chooser} for this exact 100-parameter \
             unit-cube Gaussian study; durable_frankenscipy_boundary=3x \
             outcome={} ratio_ci_low={:.9} old_83_to_159x_retired=true",
            decision.outcome, decision.ratio_low
        );
        scipy.stop()?;
        Ok(())
    }
}

#[cfg(feature = "tplquad-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("perf_tplquad_many_scipy failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "tplquad-incumbent-bench"))]
fn main() {
    eprintln!("perf_tplquad_many_scipy requires --features tplquad-incumbent-bench");
    std::process::exit(2);
}
