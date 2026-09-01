//! AVX2+FMA re-adjudication of the pre-ISA-floor `ndtri` central SIMD reject.

#[cfg(feature = "ndtri-isafloor-bench")]
mod bench {
    use fsci_runtime::scipy_incumbent::ScipyIncumbent;
    use fsci_special::convenience::{
        ndtri_isafloor_scalar_baseline, ndtri_isafloor_simd_candidate, ndtri_scalar,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Stdio};
    use std::time::{Duration, Instant};

    /// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
    /// installation whose compiled submodules do not load, and that difference would
    /// otherwise only surface mid-run.
    const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.special"];

    /// The one live-SciPy incumbent this process compares against, resolved once and PROVEN
    /// by running the import rather than by a path or a `PATH` name resolving.
    ///
    /// This harness used to spawn a bare `python3`, which on `thinkstation1` is 3.14
    /// with no SciPy at all, so the oracle died on its first write and the run read as a
    /// flaky pipe rather than as a missing incumbent (frankenscipy-m5s54).
    fn incumbent() -> &'static ScipyIncumbent {
        static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
        INCUMBENT.get_or_init(|| {
            let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
                .unwrap_or_else(|error| panic!("{error}"));
            println!("{}", resolved.provenance_line());
            resolved
        })
    }

    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 10;
    const DEFAULT_ELEMENTS: usize = 200_000;
    const CENTRAL_BOUNDARY: f64 = 0.135_335_283_236_612_7;
    const SCIPY_ABSOLUTE_TOLERANCE: f64 = 2.0e-13;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const BOOT_ID_PATH: &str = "/proc/sys/kernel/random/boot_id";

    const PYTHON_ORACLE: &str = r#"
import hashlib
import os
import sys
import threading
import time

import numpy as np
import scipy
import scipy.special as special
import scipy.special._ufuncs as special_ufuncs

BOUNDARY = 0.1353352832366127
PROBES = np.array([
    1.0e-300,
    1.0e-50,
    1.0e-12,
    np.nextafter(BOUNDARY, 0.0),
    BOUNDARY,
    np.nextafter(BOUNDARY, 1.0),
    0.25,
    0.5,
    0.75,
    np.nextafter(1.0 - BOUNDARY, 0.0),
    1.0 - BOUNDARY,
    np.nextafter(1.0 - BOUNDARY, 1.0),
    1.0 - 1.0e-12,
], dtype=np.float64)
FIXTURES = {}
OBSERVED_NATIVE_IDS = set()
MAX_OS_TASKS = 0

def observe():
    global MAX_OS_TASKS
    OBSERVED_NATIVE_IDS.add(threading.get_native_id())
    MAX_OS_TASKS = max(MAX_OS_TASKS, len(os.listdir("/proc/self/task")))

def fixture(kind, n):
    key = (kind, n)
    cached = FIXTURES.get(key)
    if cached is not None:
        return cached
    if kind == "central":
        index = np.arange(n, dtype=np.float64) + 0.5
        values = BOUNDARY + (1.0 - 2.0 * BOUNDARY) * index / float(n)
    elif kind == "mixed":
        values = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    elif kind == "tail":
        half = n // 2
        index = np.arange(half, dtype=np.float64) + 0.5
        lower = 1.0e-12 + (BOUNDARY - 2.0e-12) * index / float(half)
        values = np.empty(n, dtype=np.float64)
        values[0::2] = lower
        values[1::2] = 1.0 - lower
    else:
        raise RuntimeError(f"unknown fixture {kind!r}")
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid fixture")
    FIXTURES[key] = values
    return values

observe()
special.ndtri(PROBES)
observe()
engine_path = special_ufuncs.__file__
with open(engine_path, "rb") as engine_file:
    engine_sha256 = hashlib.sha256(engine_file.read()).hexdigest()
fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and isinstance(special.ndtri, np.ufunc)
    and special.ndtri is special_ufuncs.ndtri
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" ufunc_name={special.ndtri.__name__}"
    f" ufunc_type={type(special.ndtri).__module__}.{type(special.ndtri).__name__}"
    f" scipy_engine_path={engine_path}"
    f" scipy_engine_sha256={engine_sha256}"
    f" actual_observed_worker_threads={len(OBSERVED_NATIVE_IDS)}"
    f" max_observed_os_tasks={MAX_OS_TASKS}"
    f" fsci_loaded={fsci_loaded}"
    f" genuine={genuine}",
    flush=True,
)

for line in sys.stdin:
    fields = line.strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "CHECK" and len(fields) == 1:
        observe()
        output = special.ndtri(PROBES)
        observe()
        values = ",".join(f"{value:.17e}" for value in output)
        print(
            f"CHECK {len(OBSERVED_NATIVE_IDS)} {MAX_OS_TASKS} {values}",
            flush=True,
        )
    elif command == "TIME" and len(fields) == 4:
        kind = fields[1]
        n = int(fields[2])
        repetitions = int(fields[3])
        if n <= 0 or repetitions <= 0:
            raise RuntimeError("n and repetitions must be positive")
        values = fixture(kind, n)
        checksum = 0.0
        observe()
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            output = special.ndtri(values)
            checksum += float(output[0]) + float(output[n // 2]) + float(output[-1])
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        observe()
        print(
            f"TIME {kind} {elapsed:.17e} {checksum:.17e}"
            f" {len(OBSERVED_NATIVE_IDS)} {MAX_OS_TASKS}",
            flush=True,
        )
    elif command == "QUIT":
        print("BYE", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {line!r}")
"#;

    #[derive(Clone, Copy)]
    enum RustArm {
        Scalar,
        Simd,
    }

    impl RustArm {
        fn name(self) -> &'static str {
            match self {
                Self::Scalar => "scalar",
                Self::Simd => "simd",
            }
        }

        fn evaluate(self, input: &[f64]) -> Vec<f64> {
            match self {
                Self::Scalar => ndtri_isafloor_scalar_baseline(input),
                Self::Simd => ndtri_isafloor_simd_candidate(input),
            }
        }
    }

    struct Fixture {
        name: &'static str,
        values: Vec<f64>,
    }

    #[derive(Clone, Copy)]
    struct Timing {
        elapsed: f64,
    }

    struct ScipyCheck {
        observed_threads: usize,
        max_os_tasks: usize,
        output: Vec<f64>,
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    impl Scipy {
        fn start() -> Result<(Self, String), String> {
            let mut child = incumbent()
                .command()
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
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
                .map_err(|error| format!("spawn live SciPy oracle: {error}"))?;
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

        fn check(&mut self) -> Result<ScipyCheck, String> {
            let reply = self.request("CHECK", "SciPy parity check")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 4 || fields[0] != "CHECK" {
                return Err(format!("invalid SciPy parity reply: {reply}"));
            }
            let observed_threads = parse(fields[1], "SciPy observed threads")?;
            let max_os_tasks = parse(fields[2], "SciPy max OS tasks")?;
            let output = fields[3]
                .split(',')
                .map(|value| parse::<f64>(value, "SciPy output"))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ScipyCheck {
                observed_threads,
                max_os_tasks,
                output,
            })
        }

        fn time(
            &mut self,
            fixture: &str,
            elements: usize,
            repetitions: usize,
        ) -> Result<Timing, String> {
            let reply = self.request(
                &format!("TIME {fixture} {elements} {repetitions}"),
                "SciPy timed ndtri job",
            )?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" || fields[1] != fixture {
                return Err(format!("invalid SciPy timing reply: {reply}"));
            }
            let elapsed = parse::<f64>(fields[2], "SciPy elapsed")?;
            let checksum = parse::<f64>(fields[3], "SciPy checksum")?;
            let observed_threads = parse::<usize>(fields[4], "SciPy observed threads")?;
            let max_os_tasks = parse::<usize>(fields[5], "SciPy max OS tasks")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || !checksum.is_finite()
                || observed_threads != 1
                || max_os_tasks != 1
            {
                return Err(format!("inadmissible SciPy timing: {reply}"));
            }
            Ok(Timing { elapsed })
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

    #[derive(Clone)]
    struct SelfMeasurement {
        scalar: Vec<f64>,
        simd: Vec<f64>,
        ratios: Vec<f64>,
        scalar_nulls: Vec<f64>,
        simd_nulls: Vec<f64>,
    }

    #[derive(Clone)]
    struct IncumbentMeasurement {
        simd: Vec<f64>,
        scipy: Vec<f64>,
        ratios: Vec<f64>,
        simd_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
    }

    struct Gate {
        median: f64,
        low: f64,
        high: f64,
        first_null_median: f64,
        first_null_low: f64,
        first_null_high: f64,
        second_null_median: f64,
        second_null_low: f64,
        second_null_high: f64,
        c1: bool,
        c2: bool,
        c2b: bool,
        c3: bool,
        decidable: bool,
        outcome: &'static str,
    }

    fn fixtures(elements: usize) -> [Fixture; 3] {
        let central = (0..elements)
            .map(|index| {
                CENTRAL_BOUNDARY
                    + (1.0 - 2.0 * CENTRAL_BOUNDARY) * (index as f64 + 0.5) / elements as f64
            })
            .collect();
        let mixed = (0..elements)
            .map(|index| (index as f64 + 0.5) / elements as f64)
            .collect();
        let half = elements / 2;
        let mut tail = Vec::with_capacity(elements);
        for index in 0..half {
            let lower = 1.0e-12 + (CENTRAL_BOUNDARY - 2.0e-12) * (index as f64 + 0.5) / half as f64;
            tail.push(lower);
            tail.push(1.0 - lower);
        }
        [
            Fixture {
                name: "central",
                values: central,
            },
            Fixture {
                name: "tail",
                values: tail,
            },
            Fixture {
                name: "mixed",
                values: mixed,
            },
        ]
    }

    fn probe_probabilities() -> Vec<f64> {
        let lower_bits = CENTRAL_BOUNDARY.to_bits();
        let upper = 1.0 - CENTRAL_BOUNDARY;
        let upper_bits = upper.to_bits();
        vec![
            1.0e-300,
            1.0e-50,
            1.0e-12,
            f64::from_bits(lower_bits - 1),
            CENTRAL_BOUNDARY,
            f64::from_bits(lower_bits + 1),
            0.25,
            0.5,
            0.75,
            f64::from_bits(upper_bits - 1),
            upper,
            f64::from_bits(upper_bits + 1),
            1.0 - 1.0e-12,
        ]
    }

    fn prove_candidate_parity(fixtures: &[Fixture; 3]) -> Result<usize, String> {
        let mut tested = 0usize;
        for fixture in fixtures {
            let scalar = ndtri_isafloor_scalar_baseline(&fixture.values);
            let simd = ndtri_isafloor_simd_candidate(&fixture.values);
            if scalar.len() != simd.len() {
                return Err(format!("{} parity length mismatch", fixture.name));
            }
            for (index, (&left, &right)) in scalar.iter().zip(&simd).enumerate() {
                if left.to_bits() != right.to_bits() {
                    return Err(format!(
                        "{} candidate bit mismatch at {index}: scalar={left:.17e} \
                         simd={right:.17e}",
                        fixture.name
                    ));
                }
                tested += 1;
            }
        }
        let probes = probe_probabilities();
        let scalar = ndtri_isafloor_scalar_baseline(&probes);
        let simd = ndtri_isafloor_simd_candidate(&probes);
        for (index, (&left, &right)) in scalar.iter().zip(&simd).enumerate() {
            if left.to_bits() != right.to_bits() {
                return Err(format!(
                    "probe candidate bit mismatch at {index}: scalar={left:.17e} \
                     simd={right:.17e}"
                ));
            }
            tested += 1;
        }
        Ok(tested)
    }

    fn prove_scipy_parity(check: &ScipyCheck) -> Result<f64, String> {
        let probes = probe_probabilities();
        if check.output.len() != probes.len() {
            return Err(format!(
                "SciPy returned {} probes, expected {}",
                check.output.len(),
                probes.len()
            ));
        }
        let mut maximum = 0.0f64;
        for (probability, incumbent) in probes.into_iter().zip(&check.output) {
            let ours = ndtri_scalar(probability);
            maximum = maximum.max((ours - incumbent).abs());
        }
        if check.observed_threads != 1
            || check.max_os_tasks != 1
            || maximum > SCIPY_ABSOLUTE_TOLERANCE
        {
            return Err(format!(
                "SciPy parity failed: max_abs_difference={maximum:.3e} \
                 observed_threads={} max_os_tasks={}",
                check.observed_threads, check.max_os_tasks
            ));
        }
        Ok(maximum)
    }

    fn time_rust(arm: RustArm, input: &[f64], repetitions: usize) -> Result<Timing, String> {
        let threads_before = observed_os_threads()?;
        let mut checksum = 0u64;
        let started = Instant::now();
        for _ in 0..repetitions {
            let output = arm.evaluate(black_box(input));
            checksum = checksum
                .wrapping_add(output[0].to_bits())
                .wrapping_add(output[output.len() / 2].to_bits())
                .wrapping_add(output[output.len() - 1].to_bits());
            black_box(output);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let observed_threads = threads_before.max(observed_os_threads()?);
        black_box(checksum);
        if !elapsed.is_finite() || elapsed <= 0.0 || observed_threads != 1 {
            return Err(format!(
                "inadmissible {} timing: elapsed={elapsed} \
                 actual_observed_os_tasks={observed_threads}",
                arm.name()
            ));
        }
        Ok(Timing { elapsed })
    }

    fn effect_pair(
        fixture: &Fixture,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_rust(RustArm::Scalar, &fixture.values, repetitions)?.elapsed,
                time_rust(RustArm::Simd, &fixture.values, repetitions)?.elapsed,
            ))
        } else {
            let simd = time_rust(RustArm::Simd, &fixture.values, repetitions)?.elapsed;
            let scalar = time_rust(RustArm::Scalar, &fixture.values, repetitions)?.elapsed;
            Ok((scalar, simd))
        }
    }

    fn rust_null_pair(
        arm: RustArm,
        input: &[f64],
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_rust(arm, input, repetitions)?.elapsed,
                time_rust(arm, input, repetitions)?.elapsed,
            )
        } else {
            let right = time_rust(arm, input, repetitions)?.elapsed;
            let left = time_rust(arm, input, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    fn measure_self(
        fixture: &Fixture,
        rounds: usize,
        repetitions: usize,
    ) -> Result<SelfMeasurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(fixture, repetitions, warmup)?;
        }
        let mut measurement = SelfMeasurement {
            scalar: Vec::with_capacity(rounds),
            simd: Vec::with_capacity(rounds),
            ratios: Vec::with_capacity(rounds),
            scalar_nulls: Vec::with_capacity(rounds),
            simd_nulls: Vec::with_capacity(rounds),
        };
        for round in 0..rounds {
            let (scalar, simd, scalar_null, simd_null) = match round % 3 {
                0 => {
                    let pair = effect_pair(fixture, repetitions, round)?;
                    let scalar_null =
                        rust_null_pair(RustArm::Scalar, &fixture.values, repetitions, round)?;
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    (pair.0, pair.1, scalar_null, simd_null)
                }
                1 => {
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    let pair = effect_pair(fixture, repetitions, round)?;
                    let scalar_null =
                        rust_null_pair(RustArm::Scalar, &fixture.values, repetitions, round)?;
                    (pair.0, pair.1, scalar_null, simd_null)
                }
                _ => {
                    let scalar_null =
                        rust_null_pair(RustArm::Scalar, &fixture.values, repetitions, round)?;
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    let pair = effect_pair(fixture, repetitions, round)?;
                    (pair.0, pair.1, scalar_null, simd_null)
                }
            };
            measurement.scalar.push(scalar);
            measurement.simd.push(simd);
            measurement.ratios.push(scalar / simd);
            measurement.scalar_nulls.push(scalar_null);
            measurement.simd_nulls.push(simd_null);
            println!(
                "self_round fixture={} round={round} scalar_seconds={scalar:.9} \
                 simd_seconds={simd:.9} scalar_over_simd={:.9} \
                 scalar_null={scalar_null:.9} simd_null={simd_null:.9}",
                fixture.name,
                scalar / simd
            );
        }
        Ok(measurement)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        fixture: &str,
        elements: usize,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                scipy.time(fixture, elements, repetitions)?.elapsed,
                scipy.time(fixture, elements, repetitions)?.elapsed,
            )
        } else {
            let right = scipy.time(fixture, elements, repetitions)?.elapsed;
            let left = scipy.time(fixture, elements, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    fn incumbent_effect_pair(
        scipy: &mut Scipy,
        fixture: &Fixture,
        elements: usize,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_rust(RustArm::Simd, &fixture.values, repetitions)?.elapsed,
                scipy.time(fixture.name, elements, repetitions)?.elapsed,
            ))
        } else {
            let incumbent = scipy.time(fixture.name, elements, repetitions)?.elapsed;
            let simd = time_rust(RustArm::Simd, &fixture.values, repetitions)?.elapsed;
            Ok((simd, incumbent))
        }
    }

    fn measure_incumbent(
        scipy: &mut Scipy,
        fixture: &Fixture,
        elements: usize,
        rounds: usize,
        repetitions: usize,
    ) -> Result<IncumbentMeasurement, String> {
        for warmup in 0..2 {
            let _ = incumbent_effect_pair(scipy, fixture, elements, repetitions, warmup)?;
        }
        let mut measurement = IncumbentMeasurement {
            simd: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            ratios: Vec::with_capacity(rounds),
            simd_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
        };
        for round in 0..rounds {
            let (simd, incumbent, simd_null, scipy_null) = match round % 3 {
                0 => {
                    let pair = incumbent_effect_pair(scipy, fixture, elements, repetitions, round)?;
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    let scipy_null =
                        scipy_null_pair(scipy, fixture.name, elements, repetitions, round)?;
                    (pair.0, pair.1, simd_null, scipy_null)
                }
                1 => {
                    let scipy_null =
                        scipy_null_pair(scipy, fixture.name, elements, repetitions, round)?;
                    let pair = incumbent_effect_pair(scipy, fixture, elements, repetitions, round)?;
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    (pair.0, pair.1, simd_null, scipy_null)
                }
                _ => {
                    let simd_null =
                        rust_null_pair(RustArm::Simd, &fixture.values, repetitions, round)?;
                    let scipy_null =
                        scipy_null_pair(scipy, fixture.name, elements, repetitions, round)?;
                    let pair = incumbent_effect_pair(scipy, fixture, elements, repetitions, round)?;
                    (pair.0, pair.1, simd_null, scipy_null)
                }
            };
            measurement.simd.push(simd);
            measurement.scipy.push(incumbent);
            measurement.ratios.push(incumbent / simd);
            measurement.simd_nulls.push(simd_null);
            measurement.scipy_nulls.push(scipy_null);
            println!(
                "incumbent_round fixture={} round={round} simd_seconds={simd:.9} \
                 scipy_seconds={incumbent:.9} scipy_over_simd={:.9} \
                 simd_null={simd_null:.9} scipy_null={scipy_null:.9}",
                fixture.name,
                incumbent / simd
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
        let mut state = 0x510e_527f_ade6_82d1u64;
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

    fn corrected_gate(ratios: &[f64], first_nulls: &[f64], second_nulls: &[f64]) -> Gate {
        let effect_median = median(ratios.to_vec());
        let (effect_low, effect_high) = bootstrap_median_ci(ratios);
        let first_null_median = median(first_nulls.to_vec());
        let second_null_median = median(second_nulls.to_vec());
        let (first_null_low, first_null_high) = bootstrap_median_ci(first_nulls);
        let (second_null_low, second_null_high) = bootstrap_median_ci(second_nulls);
        let null_half_width = ((first_null_high - first_null_low) / 2.0)
            .max((second_null_high - second_null_low) / 2.0)
            .max(0.0);
        let widest_null_endpoint_distance = (first_null_low - 1.0)
            .abs()
            .max((first_null_high - 1.0).abs())
            .max((second_null_low - 1.0).abs())
            .max((second_null_high - 1.0).abs());
        let c1 = effect_low > 1.0 || effect_high < 1.0;
        let point_effect_distance = (effect_median - 1.0).abs();
        let nearer_effect_ci_endpoint_distance = if effect_low > 1.0 {
            effect_low - 1.0
        } else if effect_high < 1.0 {
            1.0 - effect_high
        } else {
            0.0
        };
        let c2 = point_effect_distance > 2.0 * null_half_width;
        let c2b = nearer_effect_ci_endpoint_distance > 2.0 * widest_null_endpoint_distance;
        let c3 = (first_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (second_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let decidable = c1 && c2 && c2b && c3;
        let outcome = if decidable && effect_low > 1.0 {
            "WIN"
        } else if decidable && effect_high < 1.0 {
            "LOSS"
        } else {
            "UNDECIDABLE"
        };
        Gate {
            median: effect_median,
            low: effect_low,
            high: effect_high,
            first_null_median,
            first_null_low,
            first_null_high,
            second_null_median,
            second_null_low,
            second_null_high,
            c1,
            c2,
            c2b,
            c3,
            decidable,
            outcome,
        }
    }

    fn print_gate(label: &str, gate: &Gate, first_name: &str, second_name: &str) {
        let null_half_width = ((gate.first_null_high - gate.first_null_low) / 2.0)
            .max((gate.second_null_high - gate.second_null_low) / 2.0)
            .max(0.0);
        let widest_null_endpoint_distance = (gate.first_null_low - 1.0)
            .abs()
            .max((gate.first_null_high - 1.0).abs())
            .max((gate.second_null_low - 1.0).abs())
            .max((gate.second_null_high - 1.0).abs());
        println!(
            "{label}_ratio median={:.9} ci95=[{:.9},{:.9}]",
            gate.median, gate.low, gate.high
        );
        println!(
            "{label}_NULL-{first_name} median={:.9} ci95=[{:.9},{:.9}] \
             ci_straddles_one={} telemetry_only=true",
            gate.first_null_median,
            gate.first_null_low,
            gate.first_null_high,
            gate.first_null_low <= 1.0 && gate.first_null_high >= 1.0
        );
        println!(
            "{label}_NULL-{second_name} median={:.9} ci95=[{:.9},{:.9}] \
             ci_straddles_one={} telemetry_only=true",
            gate.second_null_median,
            gate.second_null_low,
            gate.second_null_high,
            gate.second_null_low <= 1.0 && gate.second_null_high >= 1.0
        );
        println!(
            "{label}_corrected_null_gate: c1_effect_ci_excludes_one={} \
             c2_point_effect_beats_2x_half_width={} \
             c2b_nearer_effect_ci_endpoint_beats_2x_null_endpoint={} \
             c3_null_medians_within_2pct={} decidable={} \
             null_half_width={null_half_width:.9} \
             widest_null_endpoint_distance={widest_null_endpoint_distance:.9} \
             null_ci_veto=disabled_telemetry_only \
             outcome={}",
            gate.c1, gate.c2, gate.c2b, gate.c3, gate.decidable, gate.outcome
        );
    }

    fn print_self_result(
        fixture: &Fixture,
        measurement: &SelfMeasurement,
        repetitions: usize,
    ) -> Gate {
        let gate = corrected_gate(
            &measurement.ratios,
            &measurement.scalar_nulls,
            &measurement.simd_nulls,
        );
        println!(
            "self_wall fixture={} scalar_p50_us={:.9} p95_us={:.9} p99_us={:.9} \
             simd_p50_us={:.9} p95_us={:.9} p99_us={:.9}",
            fixture.name,
            median(measurement.scalar.clone()) / repetitions as f64 * 1.0e6,
            percentile(measurement.scalar.clone(), 0.95) / repetitions as f64 * 1.0e6,
            percentile(measurement.scalar.clone(), 0.99) / repetitions as f64 * 1.0e6,
            median(measurement.simd.clone()) / repetitions as f64 * 1.0e6,
            percentile(measurement.simd.clone(), 0.95) / repetitions as f64 * 1.0e6,
            percentile(measurement.simd.clone(), 0.99) / repetitions as f64 * 1.0e6
        );
        println!(
            "self_raw fixture={} scalar_seconds={} simd_seconds={} ratios={} \
             scalar_nulls={} simd_nulls={} ratio_cv={:.3}%",
            fixture.name,
            csv(&measurement.scalar),
            csv(&measurement.simd),
            csv(&measurement.ratios),
            csv(&measurement.scalar_nulls),
            csv(&measurement.simd_nulls),
            coefficient_of_variation(&measurement.ratios) * 100.0
        );
        print_gate(&format!("self_{}", fixture.name), &gate, "scalar", "simd");
        gate
    }

    fn print_incumbent_result(measurement: &IncumbentMeasurement, repetitions: usize) -> Gate {
        let gate = corrected_gate(
            &measurement.ratios,
            &measurement.simd_nulls,
            &measurement.scipy_nulls,
        );
        println!(
            "incumbent_wall fixture=mixed simd_p50_us={:.9} p95_us={:.9} p99_us={:.9} \
             scipy_p50_us={:.9} p95_us={:.9} p99_us={:.9}",
            median(measurement.simd.clone()) / repetitions as f64 * 1.0e6,
            percentile(measurement.simd.clone(), 0.95) / repetitions as f64 * 1.0e6,
            percentile(measurement.simd.clone(), 0.99) / repetitions as f64 * 1.0e6,
            median(measurement.scipy.clone()) / repetitions as f64 * 1.0e6,
            percentile(measurement.scipy.clone(), 0.95) / repetitions as f64 * 1.0e6,
            percentile(measurement.scipy.clone(), 0.99) / repetitions as f64 * 1.0e6
        );
        println!(
            "incumbent_raw fixture=mixed simd_seconds={} scipy_seconds={} ratios={} \
             simd_nulls={} scipy_nulls={} ratio_cv={:.3}%",
            csv(&measurement.simd),
            csv(&measurement.scipy),
            csv(&measurement.ratios),
            csv(&measurement.simd_nulls),
            csv(&measurement.scipy_nulls),
            coefficient_of_variation(&measurement.ratios) * 100.0
        );
        print_gate("incumbent_mixed", &gate, "simd", "scipy");
        gate
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
            let cpu = parse::<usize>(suffix, "CPU index")?;
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

    fn observed_os_threads() -> Result<usize, String> {
        std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("read /proc/self/task: {error}"))
            .map(Iterator::count)
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

    fn host_identity() -> Result<String, String> {
        std::fs::read_to_string("/etc/hostname")
            .map_err(|error| format!("read /etc/hostname: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn boot_id() -> Result<String, String> {
        std::fs::read_to_string(BOOT_ID_PATH)
            .map_err(|error| format!("read {BOOT_ID_PATH}: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn cpu_model() -> Result<String, String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("model name"))
            .and_then(|value| value.split_once(':').map(|(_, model)| model.trim()))
            .map(str::to_string)
            .ok_or_else(|| "model name missing from /proc/cpuinfo".to_string())
    }

    fn cpu_topology() -> Result<(usize, usize), String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let mut pairs = HashSet::new();
        let mut logical = 0usize;
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
        Ok((pairs.len(), logical))
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
                let text = entry.file_name().to_string_lossy().into_owned();
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

    fn read_policy_field(cpu: &str, name: &str) -> Result<String, String> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}");
        std::fs::read_to_string(&path)
            .map_err(|error| format!("read {path}: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn print_hardware_provenance(affinity: &str) -> Result<(), String> {
        let (physical_cores, logical_threads) = cpu_topology()?;
        let governor = read_policy_field(affinity, "scaling_governor")?;
        if governor != "performance" {
            return Err(format!(
                "CPU {affinity} governor must be performance, found {governor}"
            ));
        }
        let driver = read_policy_field(affinity, "scaling_driver")?;
        let preference = read_policy_field(affinity, "energy_performance_preference")
            .unwrap_or_else(|_| "unavailable".to_string());
        let minimum = read_policy_field(affinity, "scaling_min_freq")?;
        let maximum = read_policy_field(affinity, "scaling_max_freq")?;
        println!(
            "hardware_provenance: host_identity={} boot_id={} cpu_model={:?} \
             physical_cores={physical_cores} logical_threads={logical_threads} \
             ram_bytes={} numa_nodes={} runtime_detected_isa={} affinity={affinity}",
            host_identity()?,
            boot_id()?,
            cpu_model()?,
            ram_bytes()?,
            numa_node_count()?,
            runtime_isa_features()
        );
        println!(
            "cpu_frequency_policy: cpu={affinity} scaling_driver={driver} \
             scaling_governor={governor} \
             energy_performance_preference={preference} \
             scaling_min_freq_khz={minimum} scaling_max_freq_khz={maximum}"
        );
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

    pub fn run() -> Result<(), String> {
        let arguments = std::env::args().collect::<Vec<_>>();
        let rounds = arguments
            .get(1)
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(DEFAULT_ROUNDS);
        let repetitions = arguments
            .get(2)
            .map(|value| parse::<usize>(value, "repetitions"))
            .transpose()?
            .unwrap_or(DEFAULT_REPETITIONS);
        let elements = arguments
            .get(3)
            .map(|value| parse::<usize>(value, "elements"))
            .transpose()?
            .unwrap_or(DEFAULT_ELEMENTS);
        if rounds < DEFAULT_ROUNDS
            || repetitions == 0
            || elements < 1_024
            || !elements.is_multiple_of(8)
        {
            return Err(
                "require rounds>=15, repetitions>0, and elements>=1024 divisible by 8".to_string(),
            );
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        parse::<u64>(&booking_claim, "numeric booking claim")?;
        // frankenscipy-2b7tr gate 3, brought into line with `perf_minimize_many_scipy` and
        // `perf_dblquad_many_scipy` (95d6f6ef4).
        //
        // This USED to be an exact string comparison against
        // "rch-base-clean-overlay-no-overlay". That is TAUTOLOGICAL: `BINARY_BUILD_ROUTE`
        // is free text the operator types, so the comparison cannot tell a genuine
        // `rch exec --base --clean-overlay --no-overlay` build from an operator who typed
        // that literal after building some other way. It was demonstrated rather than
        // argued — MistyBear typed the literal for a LOCALLY built binary and the harness
        // accepted it. So the gate blocked every operator who reported truthfully while
        // admitting anyone who did not, which is the exact inversion of a provenance check.
        //
        // ONE CORRECTION to the rationale recorded on the sibling harnesses: that commit
        // also said the pre-registered route is "impossible AND forbidden" under the
        // standing build rules. It is not. `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR
        // rch exec --base HEAD --clean-overlay --no-overlay -- cargo build ...` exits 0,
        // and rch rewrites CARGO_TARGET_DIR to a worker-scoped path of its own, so the
        // banned shared `/data/tmp/cargo-target` is never touched. The route is available;
        // the check was still tautological. Both are true, and only the second is a reason
        // to replace it.
        //
        // Replaced with a DECLARED KIND from a closed set. An operator still cannot be
        // stopped from lying, but the row now states which regime it claims, the claim is
        // machine-checkable against a fixed vocabulary, and a route that is not the
        // pre-registered one LABELS ITSELF in the output instead of being
        // indistinguishable from it. The 40-hex `source_commit` check is untouched and is
        // the part that was ever doing work.
        const PREREGISTERED_ROUTE: &str = "rch-exec-base-clean-overlay-no-overlay";
        const ROUTE_KINDS: [&str; 2] = [PREREGISTERED_ROUTE, "local-wrapper-bypass"];
        let declared_kind = ROUTE_KINDS
            .iter()
            .find(|kind| build_route.starts_with(**kind))
            .ok_or_else(|| {
                format!(
                    "BINARY_BUILD_ROUTE must START WITH one of {ROUTE_KINDS:?}, followed by \
                     the verbatim command; got {build_route:?}"
                )
            })?;
        if *declared_kind != PREREGISTERED_ROUTE {
            // Printed, not fatal: the row is allowed to exist and is required to say so.
            println!(
                "build_route_deviation: declared_kind={declared_kind} \
                 preregistered_kind={PREREGISTERED_ROUTE} \
                 note=row_is_not_on_the_preregistered_route"
            );
        }
        println!("elf_sha256={elf_sha256}");
        println!(
            "binary_provenance: source_commit={source_commit} \
             builder_identity={builder_identity} build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");
        println!(
            "python_oracle_protocol_sha256={:x}",
            Sha256::digest(PYTHON_ORACLE.as_bytes())
        );

        let affinity = cpu_affinity()?;
        if affinity.contains(',')
            || affinity.contains('-')
            || parse::<usize>(&affinity, "single CPU affinity").is_err()
        {
            return Err(format!(
                "pin the whole invocation to exactly one CPU, found affinity={affinity}"
            ));
        }
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return Err("VOID-ISAFLOOR retry requires runtime AVX2+FMA".to_string());
        }
        print_hardware_provenance(&affinity)?;
        require_host_wide_quiescence("pre")?;

        let fixtures = fixtures(elements);
        let parity_values = prove_candidate_parity(&fixtures)?;
        let rust_observed_threads = observed_os_threads()?;
        if rust_observed_threads != 1 {
            return Err(format!(
                "Rust actual observed OS tasks={rust_observed_threads}, expected one"
            ));
        }
        println!(
            "fixture_contract: elements_per_fixture={elements} fixtures=central,tail,mixed \
             boundary={CENTRAL_BOUNDARY:.17e} rounds={rounds} \
             repetitions={repetitions} output_bytes_per_job={} \
             candidate_scalar_bit_parity_values={parity_values} bit_mismatches=0",
            elements * std::mem::size_of::<f64>()
        );
        println!(
            "job_boundary: INCLUDED=preconstructed_input,full_output_allocation,\
             every_ndtri_element,full_output_materialization; \
             EXCLUDED=input_construction,python_startup,scipy_import,pipe_transport,\
             parity_serialization,bootstrap"
        );

        let (mut scipy, identity) = Scipy::start()?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains("ufunc_name=ndtri")
            || !identity.contains("ufunc_type=numpy.ufunc")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("max_observed_os_tasks=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine identity/thread gate".to_string());
        }
        let scipy_engine_sha256 = ready_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "SciPy arm omitted engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("SciPy arm reported an invalid engine SHA-256".to_string());
        }
        let scipy_check = scipy.check()?;
        let scipy_max_abs_difference = prove_scipy_parity(&scipy_check)?;
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "parity_proof: candidate_vs_scalar_bit_mismatches=0 \
             values_tested={parity_values} scipy_probe_max_abs_difference=\
             {scipy_max_abs_difference:.3e} scipy_tolerance={SCIPY_ABSOLUTE_TOLERANCE:.3e}"
        );
        println!(
            "thread_provenance: requested_rust_threads=1 \
             actual_observed_rust_os_tasks={rust_observed_threads} \
             requested_scipy_threads=1 actual_observed_scipy_worker_threads={} \
             actual_observed_scipy_max_os_tasks={}",
            scipy_check.observed_threads, scipy_check.max_os_tasks
        );
        println!(
            "incumbent_contract: arm=scipy.special.ndtri \
             type=numpy.ufunc genuine_scipy_1_17_1=true \
             strongest_semantically_equivalent_public_arm=true"
        );

        let mut gates = Vec::with_capacity(fixtures.len());
        for fixture in &fixtures {
            let measurement = measure_self(fixture, rounds, repetitions)?;
            gates.push((
                fixture.name,
                print_self_result(fixture, &measurement, repetitions),
            ));
        }
        let mixed = fixtures
            .iter()
            .find(|fixture| fixture.name == "mixed")
            .ok_or_else(|| "mixed fixture missing".to_string())?;
        let incumbent_measurement =
            measure_incumbent(&mut scipy, mixed, elements, rounds, repetitions)?;
        let incumbent_gate = print_incumbent_result(&incumbent_measurement, repetitions);

        let central_gate = &gates[0].1;
        let tail_gate = &gates[1].1;
        let mixed_gate = &gates[2].1;
        let central_prediction_confirmed = central_gate.low > 1.25;
        let central_prediction_falsified = central_gate.low <= 1.25;
        let mixed_reject_restood = mixed_gate.outcome != "WIN";
        let mixed_reject_falsified = mixed_gate.outcome == "WIN";
        let tail_prediction_confirmed = tail_gate.outcome != "WIN";
        let tail_prediction_falsified = tail_gate.outcome == "WIN";
        let incumbent_prediction_confirmed = incumbent_gate.low <= 1.2;
        let incumbent_prediction_falsified = incumbent_gate.low > 1.2;
        println!(
            "preregistered_predictions: \
             central_ci_low={:.9} central_gt_1_25_confirmed={central_prediction_confirmed} \
             central_gt_1_25_falsified={central_prediction_falsified} \
             mixed_outcome={} old_reject_restood={mixed_reject_restood} \
             old_reject_falsified={mixed_reject_falsified} \
             tail_outcome={} tail_no_win_confirmed={tail_prediction_confirmed} \
             tail_no_win_falsified={tail_prediction_falsified} \
             incumbent_ci_low={:.9} no_durable_1_2x_incumbent_win_confirmed=\
             {incumbent_prediction_confirmed} no_durable_1_2x_incumbent_win_falsified=\
             {incumbent_prediction_falsified}",
            central_gate.low, mixed_gate.outcome, tail_gate.outcome, incumbent_gate.low
        );

        let production_decision = if mixed_gate.outcome == "WIN" && incumbent_gate.outcome != "LOSS"
        {
            "KEEP-SIMD"
        } else {
            "REJECT-SIMD"
        };
        let chooser = if production_decision == "KEEP-SIMD" {
            "use central SIMD only in ndtri's existing serial middle-size band"
        } else {
            "retain the current scalar/work-gated ndtri implementation"
        };
        println!(
            "CHOOSER STATEMENT: {chooser}; central-only evidence never creates \
             a distribution-sensitive public chooser."
        );

        scipy.stop()?;
        require_host_wide_quiescence("post")?;
        println!(
            "FINAL: production_decision={production_decision} \
             central_self_ratio={:.9} ci95=[{:.9},{:.9}] outcome={} \
             tail_self_ratio={:.9} ci95=[{:.9},{:.9}] outcome={} \
             mixed_self_ratio={:.9} ci95=[{:.9},{:.9}] outcome={} \
             mixed_scipy_over_simd={:.9} ci95=[{:.9},{:.9}] outcome={} \
             actual_observed_rust_threads=1 actual_observed_scipy_threads=1 \
             elf_sha256={elf_sha256} scipy_engine_sha256={scipy_engine_sha256}",
            central_gate.median,
            central_gate.low,
            central_gate.high,
            central_gate.outcome,
            tail_gate.median,
            tail_gate.low,
            tail_gate.high,
            tail_gate.outcome,
            mixed_gate.median,
            mixed_gate.low,
            mixed_gate.high,
            mixed_gate.outcome,
            incumbent_gate.median,
            incumbent_gate.low,
            incumbent_gate.high,
            incumbent_gate.outcome
        );
        Ok(())
    }
}

#[cfg(feature = "ndtri-isafloor-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "ndtri-isafloor-bench"))]
fn main() {
    eprintln!("perf_cdf requires --features ndtri-isafloor-bench");
    std::process::exit(2);
}
