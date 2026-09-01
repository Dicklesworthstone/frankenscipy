//! Whole 128-parameter double-integration job versus screened SciPy 1.17.1.
//!
//! The historical `dblquad_many` row compared against a scalar Python loop.
//! This harness screens SciPy's public vector-valued routes, freezes the
//! strongest valid incumbent, and applies the corrected dual-null median gate.

#[cfg(feature = "dblquad-incumbent-bench")]
mod bench {
    use fsci_integrate::{DblquadOptions, DblquadResult, dblquad_many};
    use fsci_runtime::scipy_incumbent::{PINNED_NUMPY, PINNED_SCIPY, ScipyIncumbent};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Stdio};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::{Duration, Instant};

    const BATCH: usize = 128;
    const EPSABS: f64 = 1.49e-8;
    const EPSREL: f64 = 1.49e-8;
    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 5;
    const SCREEN_ROUNDS: usize = 5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const MAX_SCALED_REFERENCE_ERROR: f64 = 4.0;
    const COLLAPSE_BOUNDARY: f64 = 21.1;
    const DURABLE_WIN_BOUNDARY: f64 = 3.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const BASE_SCIPY_ARMS: [&str; 4] = [
        "dblquad_scalar",
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

BATCH = 128
EPSABS = 1.49e-8
EPSREL = 1.49e-8
PARAMS = np.asarray(
    [5.0 + 35.0 * i / (BATCH - 1) for i in range(BATCH)],
    dtype="<f8",
)
REFERENCE = (
    np.pi / PARAMS * np.square(special.erf(np.sqrt(PARAMS) / 2.0))
).astype("<f8")
PARAMS_SHA256 = hashlib.sha256(PARAMS.tobytes(order="C")).hexdigest()
REFERENCE_SHA256 = hashlib.sha256(REFERENCE.tobytes(order="C")).hexdigest()
WORKER_CAPACITY = max(1, len(os.sched_getaffinity(0)))
POOL = ThreadPool(WORKER_CAPACITY)
WORKER_RULE = "gk21"

def scalar_job():
    values = []
    errors = []
    for p in PARAMS:
        value, error = integrate.dblquad(
            lambda y, x, p=float(p): math.exp(
                -p * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
            ),
            0.0,
            1.0,
            lambda _x: 0.0,
            lambda _x: 1.0,
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
    inner_success = True
    inner_errors = []
    inner_evaluations = 0

    def inner(x):
        nonlocal inner_success, inner_evaluations
        value, error, info = integrate.quad_vec(
            lambda y: np.exp(
                -PARAMS * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
            ),
            0.0,
            1.0,
            epsabs=EPSABS,
            epsrel=EPSREL,
            norm="max",
            full_output=True,
        )
        inner_success = inner_success and bool(info.success)
        inner_errors.append(float(error))
        inner_evaluations += int(info.neval)
        return value

    value, outer_error, info = integrate.quad_vec(
        inner,
        0.0,
        1.0,
        epsabs=EPSABS,
        epsrel=EPSREL,
        norm="max",
        full_output=True,
    )
    combined_error = float(outer_error) + max(inner_errors, default=0.0)
    return (
        np.asarray(value, dtype=np.float64),
        np.full(BATCH, combined_error, dtype=np.float64),
        bool(info.success) and inner_success,
        int(info.neval) + inner_evaluations,
    )

def vector_integrand(points):
    radius_squared = (
        np.square(points[:, 0] - 0.5) + np.square(points[:, 1] - 0.5)
    )
    return np.exp(-radius_squared[:, None] * PARAMS[None, :])

def cubature_job(rule, workers):
    result = integrate.cubature(
        vector_integrand,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
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
    if arm == "dblquad_scalar":
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
    ("dblquad",) + source_hash(_quadpack_py),
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
    "dblquad_scalar",
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
    and integrate.dblquad.__module__ == "scipy.integrate._quadpack_py"
    and integrate.quad_vec.__module__ == "scipy.integrate._quad_vec"
    and integrate.cubature.__module__ == "scipy.integrate._cubature"
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" dblquad_module={integrate.dblquad.__module__}"
    f" quad_vec_module={integrate.quad_vec.__module__}"
    f" cubature_module={integrate.cubature.__module__}"
    f" scipy_engine_sha256={SCIPY_ENGINE_SHA256}"
    f" dblquad_engine_sha256={engine_rows[0][2]}"
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

    struct RawJob {
        values: Vec<f64>,
        errors: Vec<f64>,
        converged: usize,
        work: usize,
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

        fn reference(&mut self) -> Result<Vec<f64>, String> {
            let reply = self.request("REFERENCE", "closed-form reference transfer")?;
            let values: Vec<f64> = reply
                .strip_prefix("REFERENCE ")
                .ok_or_else(|| format!("invalid reference reply: {reply}"))?
                .split(',')
                .map(|field| parse(field, "reference value"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != BATCH || values.iter().any(|value| !value.is_finite()) {
                return Err("invalid closed-form reference vector".to_string());
            }
            Ok(values)
        }

        fn set_worker_rule(&mut self, rule: &str) -> Result<(), String> {
            let reply = self.request(
                &format!("SET_WORKER_RULE {rule}"),
                "SciPy cubature worker-rule selection",
            )?;
            if reply != format!("WORKER_RULE {rule}") {
                return Err(format!("invalid worker-rule reply: {reply}"));
            }
            Ok(())
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
                "SciPy timed double-integration job",
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

    fn parameters() -> Vec<Vec<f64>> {
        (0..BATCH)
            .map(|index| vec![5.0 + 35.0 * index as f64 / (BATCH - 1) as f64])
            .collect()
    }

    fn values_sha256(values: impl IntoIterator<Item = f64>) -> String {
        let mut digest = Sha256::new();
        for value in values {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }

    fn options() -> DblquadOptions {
        DblquadOptions {
            epsabs: EPSABS,
            epsrel: EPSREL,
            limit: 50,
        }
    }

    fn execute_batch(parameters: &[Vec<f64>]) -> Result<RawJob, String> {
        let results = dblquad_many(
            |y, x, row| {
                let radius_squared = (x - 0.5).powi(2) + (y - 0.5).powi(2);
                (-row[0] * radius_squared).exp()
            },
            0.0,
            1.0,
            0.0,
            1.0,
            parameters,
            options(),
        );
        if results.len() != BATCH {
            return Err(format!(
                "expected {BATCH} FrankenSciPy results, got {}",
                results.len()
            ));
        }
        let mut values = Vec::with_capacity(BATCH);
        let mut errors = Vec::with_capacity(BATCH);
        let mut converged = 0usize;
        for (index, result) in results.into_iter().enumerate() {
            let DblquadResult {
                integral,
                error,
                converged: did_converge,
            } = result
                .map_err(|error| format!("FrankenSciPy parameter row {index} failed: {error:?}"))?;
            if !integral.is_finite() || !error.is_finite() || error < 0.0 {
                return Err(format!(
                    "FrankenSciPy parameter row {index} returned invalid output"
                ));
            }
            values.push(integral);
            errors.push(error);
            converged += usize::from(did_converge);
        }
        Ok(RawJob {
            values,
            errors,
            converged,
            work: 0,
        })
    }

    fn whole_job_checksum(raw: &RawJob) -> Result<f64, String> {
        if raw.values.len() != BATCH || raw.errors.len() != BATCH {
            return Err("FrankenSciPy raw result has invalid shape".to_string());
        }
        let best_index = raw
            .values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "FrankenSciPy result is empty".to_string())?;
        let weighted_values = raw
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>();
        let weighted_errors = raw
            .errors
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>();
        let checksum = weighted_values
            + weighted_errors * 1.0e-3
            + best_index as f64 * 1.0e-4
            + usize::from(raw.converged == BATCH) as f64
            + raw.work as f64 * 1.0e-9;
        if !checksum.is_finite() {
            return Err("FrankenSciPy checksum is non-finite".to_string());
        }
        Ok(checksum)
    }

    fn summarize(
        raw: &RawJob,
        parameters: &[Vec<f64>],
        reference: &[f64],
    ) -> Result<JobSummary, String> {
        let checksum = whole_job_checksum(raw)?;
        if parameters.len() != BATCH || reference.len() != BATCH {
            return Err("quality fixture has invalid shape".to_string());
        }
        let best_index = raw
            .values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| "FrankenSciPy result is empty".to_string())?;
        let mut max_abs_reference_error = 0.0f64;
        let mut max_scaled_reference_error = 0.0f64;
        for (value, expected) in raw.values.iter().zip(reference) {
            let absolute = (value - expected).abs();
            let scale = EPSABS + EPSREL * expected.abs();
            max_abs_reference_error = max_abs_reference_error.max(absolute);
            max_scaled_reference_error = max_scaled_reference_error.max(absolute / scale);
        }
        Ok(JobSummary {
            converged: raw.converged,
            best_index,
            best_p: parameters[best_index][0],
            best_integral: raw.values[best_index],
            min_error_estimate: raw.errors.iter().copied().fold(f64::INFINITY, f64::min),
            max_error_estimate: raw.errors.iter().copied().fold(0.0, f64::max),
            max_abs_reference_error,
            max_scaled_reference_error,
            work: raw.work,
            checksum,
        })
    }

    fn time_batch(parameters: &[Vec<f64>], repetitions: usize) -> Result<Timing, String> {
        let mut checksum = 0.0;
        let started = Instant::now();
        for _ in 0..repetitions {
            let raw = execute_batch(black_box(parameters))?;
            checksum = black_box(checksum + whole_job_checksum(&raw)?);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(checksum);
        if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() {
            return Err("inadmissible FrankenSciPy timing".to_string());
        }
        Ok(Timing {
            elapsed,
            active_tasks: 0,
            max_os_tasks: 0,
        })
    }

    fn parse_summary(value: &str, label: &str) -> Result<JobSummary, String> {
        let fields = value.split(',').collect::<Vec<_>>();
        if fields.len() != 10 {
            return Err(format!("{label} has {} fields, expected ten", fields.len()));
        }
        Ok(JobSummary {
            converged: parse(fields[0], "converged count")?,
            best_index: parse(fields[1], "best index")?,
            best_p: parse(fields[2], "best parameter")?,
            best_integral: parse(fields[3], "best integral")?,
            min_error_estimate: parse(fields[4], "minimum error estimate")?,
            max_error_estimate: parse(fields[5], "maximum error estimate")?,
            max_abs_reference_error: parse(fields[6], "maximum absolute reference error")?,
            max_scaled_reference_error: parse(fields[7], "maximum scaled reference error")?,
            work: parse(fields[8], "work count")?,
            checksum: parse(fields[9], "summary checksum")?,
        })
    }

    fn basic_integrity(label: &str, summary: JobSummary) -> Result<(), String> {
        if summary.converged > BATCH
            || summary.best_index >= BATCH
            || !summary.best_p.is_finite()
            || !summary.best_integral.is_finite()
            || !summary.min_error_estimate.is_finite()
            || summary.min_error_estimate < 0.0
            || !summary.max_error_estimate.is_finite()
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

    fn observe_batch_workers(parameters: &[Vec<f64>], expected: usize) -> Result<usize, String> {
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
            black_box(execute_batch(parameters)?);
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
        parameters: &[Vec<f64>],
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, usize, usize), String> {
        if round.is_multiple_of(2) {
            let ours = time_batch(parameters, repetitions)?.elapsed;
            let incumbent = scipy.time(arm, repetitions)?;
            Ok((
                ours,
                incumbent.elapsed,
                incumbent.active_tasks,
                incumbent.max_os_tasks,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?;
            let ours = time_batch(parameters, repetitions)?.elapsed;
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
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_batch(parameters, repetitions)?.elapsed,
                time_batch(parameters, repetitions)?.elapsed,
            )
        } else {
            let right = time_batch(parameters, repetitions)?.elapsed;
            let left = time_batch(parameters, repetitions)?.elapsed;
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
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(scipy, arm, parameters, repetitions, warmup)?;
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
                        let effect = effect_pair(scipy, arm, parameters, repetitions, round)?;
                        let ours_null = ours_null_pair(parameters, repetitions, round)?;
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
                        let effect = effect_pair(scipy, arm, parameters, repetitions, round)?;
                        let ours_null = ours_null_pair(parameters, repetitions, round)?;
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
                        let ours_null = ours_null_pair(parameters, repetitions, round)?;
                        let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                        let effect = effect_pair(scipy, arm, parameters, repetitions, round)?;
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
    ///
    /// `clear` is still printed when the host genuinely is quiet, because it
    /// remains the strongest form and a row that can claim it should.
    fn report_host_wide_quiescence(phase: &str) -> Result<(), String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        if before.len() != after.len() {
            return Err("CPU topology changed during host load sample".to_string());
        }
        let mut maximum_busy_fraction = 0.0f64;
        let mut total_busy_fraction = 0.0f64;
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
            total_busy_fraction += busy_fraction;
            if busy_fraction > HOST_QUIESCENCE_MAX_BUSY {
                busy.push((cpu, busy_fraction));
            }
        }
        let host_mean_busy = total_busy_fraction / before.len() as f64;
        if busy.is_empty() {
            println!(
                "host_wide_quiescence_{phase}=clear sampled_cpus={} \
                 maximum_busy_fraction={maximum_busy_fraction:.3} \
                 host_mean_busy={host_mean_busy:.3} \
                 busy_cpu_count_above_limit=0 limit={HOST_QUIESCENCE_MAX_BUSY:.3}",
                before.len()
            );
            return Ok(());
        }
        let detail = busy
            .iter()
            .map(|(cpu, fraction)| format!("{cpu}:{:.1}%", fraction * 100.0))
            .collect::<Vec<_>>()
            .join(",");
        // Reported in the exact form the ledger recognises, and never hidden: a
        // row that conceals how busy the box was is worse than one that states
        // it, and the A/A nulls are what the row is decided on.
        println!(
            "host_wide_quiescence_{phase}=NOT_CERTIFIED(host_mean_busy={host_mean_busy:.3}) \
             sampled_cpus={} maximum_busy_fraction={maximum_busy_fraction:.3} \
             busy_cpu_count_above_limit={} limit={HOST_QUIESCENCE_MAX_BUSY:.3} \
             gate=same_invocation_A/A_nulls busy_cpus={detail}",
            before.len(),
            busy.len()
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
            "hardware_provenance: host_identity={} boot_id={} cpu_model={:?} \
             physical_cores={physical_cores} logical_threads={logical_threads} \
             ram_bytes={} numa_nodes={} runtime_detected_isa={} \
             affinity={affinity} affinity_cpu_count={} \
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

    fn validate_provenance(smoke: bool) -> Result<(), String> {
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        if source_commit.len() != 40 || !source_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("BINARY_SOURCE_COMMIT must be a full 40-hex commit".to_string());
        }
        // frankenscipy-2auhe. This USED to require the free-text route to contain the four
        // substrings "rch", "base", "clean-overlay", "no-overlay".
        //
        // That check was TAUTOLOGICAL: it is a substring match on a string the operator
        // types, so it cannot distinguish a genuine `rch exec --base --clean-overlay
        // --no-overlay` build from an operator who typed those four words after building
        // some other way. It verified nothing, while BLOCKING every operator who reported
        // truthfully — the exact inversion of what a provenance gate is for. Worse, the
        // route it demanded is now impossible AND forbidden: `/data/tmp/cargo-target` and
        // its frozen ELFs were reclaimed, and the standing build rules require
        // `RCH_CARGO_WRAPPER_BYPASS=1` with `env -u CARGO_TARGET_DIR` and forbid the shared
        // target. So the only way to satisfy it was to write a route string that is not
        // what happened, which the bead text explicitly forbids.
        //
        // Replaced with a DECLARED KIND from a closed set. The operator still cannot be
        // stopped from lying, but now the row states which regime it claims, the claim is
        // machine-checkable against a fixed vocabulary, and a route that is not the
        // pre-registered one LABELS ITSELF in the output instead of being indistinguishable
        // from it. The 40-hex commit check above is untouched and is the part that was ever
        // doing work.
        const PREREGISTERED_ROUTE: &str = "rch-exec-base-clean-overlay-no-overlay";
        const ROUTE_KINDS: [&str; 2] = [PREREGISTERED_ROUTE, "local-wrapper-bypass"];
        let declared_kind = ROUTE_KINDS
            .iter()
            .find(|kind| build_route.starts_with(**kind))
            .ok_or_else(|| {
                format!(
                    "BINARY_BUILD_ROUTE must START WITH one of {ROUTE_KINDS:?}, followed by the \
                     verbatim command; got {build_route:?}"
                )
            })?;
        if *declared_kind != PREREGISTERED_ROUTE {
            // Printed, not fatal: the row is allowed to exist and is required to say so.
            println!(
                "build_route_deviation: declared_kind={declared_kind} \
                 preregistered_kind={PREREGISTERED_ROUTE} \
                 reason=shared_cargo_target_reclaimed_and_forbidden_by_standing_rules \
                 (frankenscipy-2auhe)"
            );
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
        Ok(())
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
        validate_provenance(smoke)?;
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
            report_host_wide_quiescence("pre")?;
        } else {
            println!(
                "smoke_hardware: host_identity={} boot_id={} affinity={affinity} \
                 affinity_cpu_count={} runtime_detected_isa={} \
                 governor_gate_skipped_for_non_evidence_smoke=true",
                host_identity()?,
                boot_id()?,
                cpus.len(),
                runtime_isa_features()
            );
        }

        let parameters = parameters();
        let params_sha = values_sha256(parameters.iter().map(|row| row[0]));
        let (mut scipy, identity) = Scipy::start()?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with(&format!("READY scipy={PINNED_SCIPY} "))
            || !identity.contains(&format!(" numpy={PINNED_NUMPY} "))
            || !identity.contains("dblquad_module=scipy.integrate._quadpack_py")
            || !identity.contains("quad_vec_module=scipy.integrate._quad_vec")
            || !identity.contains("cubature_module=scipy.integrate._cubature")
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
        let scipy_params_sha = ready_value(&identity, "params_sha256=")
            .ok_or_else(|| "SciPy arm omitted parameter SHA-256".to_string())?;
        if scipy_params_sha != params_sha {
            return Err(format!(
                "input identity mismatch: FrankenSciPy={params_sha} SciPy={scipy_params_sha}"
            ));
        }
        let reference = scipy.reference()?;
        let reference_sha = values_sha256(reference.iter().copied());
        let scipy_reference_sha = ready_value(&identity, "reference_sha256=")
            .ok_or_else(|| "SciPy arm omitted reference SHA-256".to_string())?;
        if scipy_reference_sha != reference_sha {
            return Err(format!(
                "reference identity mismatch: received={reference_sha} \
                 self_reported={scipy_reference_sha}"
            ));
        }
        let scipy_worker_capacity: usize = parse(
            ready_value(&identity, "worker_capacity=")
                .ok_or_else(|| "SciPy arm omitted worker capacity".to_string())?,
            "SciPy worker capacity",
        )?;
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "fixture=gaussian-unit-square-parameter-sweep batch={BATCH} \
             parameter_domain=[5,40] params_sha256={params_sha} \
             closed_form_reference_sha256={reference_sha} \
             epsabs={EPSABS:.17e} epsrel={EPSREL:.17e} \
             rounds={rounds} repetitions_per_sample={repetitions}"
        );
        println!(
            "whole_job_boundary: INCLUDED=solve_all_128_fixed_parameters,\
             retain_all_integrals_errors_and_convergence,select_maximum,\
             materialize_checksum; EXCLUDED=parameter_generation,python_startup,\
             scipy_import,incumbent_screen,closed_form_quality_gate,\
             pipe_transport,bootstrap"
        );
        println!(
            "incumbent_screen_contract: public_arms=dblquad_scalar,quad_vec,\
             cubature_gk21,cubature_genz_malik,cubature_workers_<fastest_rule> \
             selection=lowest_five-sample-median_valid_whole-job-wall \
             screen_samples_excluded_from_effect=true"
        );

        let ours_raw = execute_batch(&parameters)?;
        let ours_summary = summarize(&ours_raw, &parameters, &reference)?;
        if !quality_eligible("FrankenSciPy", ours_summary) {
            return Err(format!(
                "FrankenSciPy failed pre-timing quality gate: {ours_summary:?}"
            ));
        }
        print_summary("frankenscipy_quality", ours_summary);

        let mut checks = HashMap::new();
        for arm in BASE_SCIPY_ARMS {
            let check = scipy.check(arm)?;
            basic_integrity(&format!("SciPy {arm}"), check.summary)?;
            let eligible = quality_eligible(&format!("SciPy {arm}"), check.summary);
            print_summary(&format!("scipy_quality_{arm}"), check.summary);
            println!(
                "scipy_observation_{arm}: quality_eligible={eligible} \
                 actual_observed_active_tasks={} max_observed_os_tasks={} \
                 workers_map_capacity={scipy_worker_capacity}",
                check.active_tasks, check.max_os_tasks
            );
            checks.insert(arm.to_string(), check);
        }
        scipy.set_worker_rule("gk21")?;
        let worker_check = scipy.check("cubature_workers")?;
        basic_integrity("SciPy cubature_workers_gk21", worker_check.summary)?;
        print_summary("scipy_quality_cubature_workers_gk21", worker_check.summary);
        println!(
            "scipy_observation_cubature_workers_gk21: quality_eligible={} \
             actual_observed_active_tasks={} max_observed_os_tasks={} \
             workers_map_capacity={scipy_worker_capacity}",
            quality_eligible("SciPy cubature_workers_gk21", worker_check.summary),
            worker_check.active_tasks,
            worker_check.max_os_tasks
        );

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
        let observed_workers = observe_batch_workers(&parameters, expected_workers)?;
        if observed_workers != expected_workers {
            return Err(format!(
                "FrankenSciPy observed worker mismatch: expected from affinity/API \
                 {expected_workers}, directly observed {observed_workers}"
            ));
        }
        println!(
            "frankenscipy_task_provenance: \
             actual_observed_concurrent_solve_workers={observed_workers} \
             available_parallelism={expected_workers} \
             observation=untimed_proc_task_sampler"
        );

        let mut screened = Vec::new();
        let mut screen_medians = HashMap::new();
        for arm in BASE_SCIPY_ARMS {
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
            let eligible = checks
                .get(arm)
                .is_some_and(|check| quality_eligible(arm, check.summary));
            println!(
                "scipy_public_arm_screen: arm={arm} eligible={eligible} \
                 raw_seconds={} median_whole_job_ms={:.9} \
                 actual_observed_active_tasks={maximum_active} \
                 max_observed_os_tasks={maximum_tasks}",
                csv(&samples),
                screen_median * 1.0e3
            );
            screen_medians.insert(arm.to_string(), screen_median);
            if eligible {
                screened.push((arm.to_string(), screen_median));
            }
        }

        let gk21_median = *screen_medians
            .get("cubature_gk21")
            .ok_or_else(|| "missing gk21 screen".to_string())?;
        let genz_median = *screen_medians
            .get("cubature_genz_malik")
            .ok_or_else(|| "missing genz-malik screen".to_string())?;
        let worker_rule = if gk21_median <= genz_median {
            "gk21"
        } else {
            "genz-malik"
        };
        scipy.set_worker_rule(worker_rule)?;
        let worker_arm = format!("cubature_workers_{worker_rule}");
        let worker_check = scipy.check("cubature_workers")?;
        basic_integrity(&format!("SciPy {worker_arm}"), worker_check.summary)?;
        print_summary(&format!("scipy_quality_{worker_arm}"), worker_check.summary);
        let worker_eligible =
            quality_eligible(&format!("SciPy {worker_arm}"), worker_check.summary);
        let mut worker_samples = Vec::with_capacity(SCREEN_ROUNDS);
        let mut worker_active = 0usize;
        let mut worker_tasks = 0usize;
        for _ in 0..SCREEN_ROUNDS {
            let timing = scipy.time("cubature_workers", 1)?;
            worker_samples.push(timing.elapsed);
            worker_active = worker_active.max(timing.active_tasks);
            worker_tasks = worker_tasks.max(timing.max_os_tasks);
        }
        let worker_median = median(worker_samples.clone());
        println!(
            "scipy_public_arm_screen: arm={worker_arm} \
             protocol_arm=cubature_workers eligible={worker_eligible} \
             raw_seconds={} median_whole_job_ms={:.9} \
             actual_observed_active_tasks={worker_active} \
             max_observed_os_tasks={worker_tasks}",
            csv(&worker_samples),
            worker_median * 1.0e3
        );
        if worker_eligible {
            screened.push(("cubature_workers".to_string(), worker_median));
        }
        checks.insert("cubature_workers".to_string(), worker_check);
        screened.sort_by(|left, right| left.1.total_cmp(&right.1));
        let selected_arm = screened
            .first()
            .map(|row| row.0.as_str())
            .ok_or_else(|| "no SciPy public arm passed the quality gate".to_string())?;
        let selected_label = if selected_arm == "cubature_workers" {
            worker_arm.as_str()
        } else {
            selected_arm
        };
        let selected_check = checks
            .get(selected_arm)
            .ok_or_else(|| "selected SciPy arm has no quality record".to_string())?;
        prove_quality(ours_summary, selected_check)?;
        println!(
            "selected_public_scipy_arm={selected_label} \
             protocol_arm={selected_arm} \
             selection_screen_outside_headline_samples=true"
        );
        println!(
            "quality_gate=PASS candidate_converged={} incumbent_converged={} \
             candidate_best_index={} incumbent_best_index={} \
             candidate_best_integral={:.17e} incumbent_best_integral={:.17e} \
             candidate_max_scaled_reference_error={:.9} \
             incumbent_max_scaled_reference_error={:.9}",
            ours_summary.converged,
            selected_check.summary.converged,
            ours_summary.best_index,
            selected_check.summary.best_index,
            ours_summary.best_integral,
            selected_check.summary.best_integral,
            ours_summary.max_scaled_reference_error,
            selected_check.summary.max_scaled_reference_error
        );

        let measurement = measure(&mut scipy, selected_arm, &parameters, rounds, repetitions)?;
        let (outcome, ratio_median, ratio_low, ratio_high) =
            print_decision(selected_label, &measurement, repetitions);

        let scalar_median = *screen_medians
            .get("dblquad_scalar")
            .ok_or_else(|| "missing scalar dblquad screen".to_string())?;
        let selected_screen_median = screened[0].1;
        let removed_callback_tax = scalar_median / selected_screen_median;
        let p1 = matches!(selected_arm, "cubature_gk21" | "cubature_genz_malik");
        let p2 = ratio_high < COLLAPSE_BOUNDARY;
        let p3 = outcome == "DECIDED FRANKENSCIPY LOSS" && ratio_high < 1.0;
        let p4 = removed_callback_tax > 10.0;
        let p5_rust = observed_workers == 32;
        let p5_scipy = p1 && measurement.max_scipy_active_tasks == 1;
        println!(
            "mechanism_summary: scalar_dblquad_screen_median_ms={:.9} \
             selected_scipy_screen_median_ms={:.9} \
             scalar_over_selected={removed_callback_tax:.9}x \
             callback_tax_removed_gt_10={p4}",
            scalar_median * 1.0e3,
            selected_screen_median * 1.0e3
        );
        println!(
            "preregistered_predictions: \
             P1_single_worker_vectorized_cubature_fastest={p1} \
             P2_old_ratio_collapses_10x={p2} \
             collapse_boundary={COLLAPSE_BOUNDARY:.3} \
             P3_vectorized_scipy_wins={p3} \
             P4_scalar_over_selected_gt_10={p4} \
             P5_rust_observed_32={p5_rust} \
             P5_scipy_observed_one={p5_scipy}"
        );
        println!(
            "thread_provenance: \
             actual_observed_frankenscipy_solve_workers={observed_workers} \
             actual_observed_scipy_active_tasks={} \
             actual_observed_scipy_max_os_tasks={} \
             scipy_workers_map_capacity={scipy_worker_capacity}",
            measurement.max_scipy_active_tasks, measurement.max_scipy_os_tasks
        );

        let durable_franken_win =
            outcome == "DECIDED FRANKENSCIPY WIN" && ratio_low > DURABLE_WIN_BOUNDARY;
        let chooser = if durable_franken_win {
            "choose FrankenSciPy dblquad_many for this exact 128-parameter job"
        } else {
            "choose the selected vectorized SciPy arm for this exact 128-parameter job"
        };
        println!(
            "CHOOSER STATEMENT: {chooser}; retire the historical scalar-loop \
             62.7-211x magnitude for this workload."
        );

        scipy.stop()?;
        report_host_wide_quiescence("post")?;
        println!(
            "FINAL: outcome={outcome} selected_public_scipy_arm={selected_label} \
             ratio={ratio_median:.9} ci95=[{ratio_low:.9},{ratio_high:.9}] \
             durable_frankenscipy_gt_3={durable_franken_win} \
             actual_observed_frankenscipy_workers={observed_workers} \
             actual_observed_scipy_active_tasks={} elf_sha256={elf_sha256} \
             scipy_engine_sha256={scipy_engine_sha256}",
            measurement.max_scipy_active_tasks
        );
        Ok(())
    }
}

#[cfg(feature = "dblquad-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "dblquad-incumbent-bench"))]
fn main() {
    eprintln!("perf_dblquad_many_scipy requires --features dblquad-incumbent-bench");
    std::process::exit(2);
}
