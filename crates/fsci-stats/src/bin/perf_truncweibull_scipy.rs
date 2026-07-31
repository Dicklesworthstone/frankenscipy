//! Public `TruncWeibullMin` mean/variance summary job versus live SciPy.
//!
//! The historical approximately 370x number was a FrankenSciPy self-speedup
//! over deleted quadrature. This harness converts it into a same-invocation
//! live-incumbent result with public-arm screening and independent A/A nulls.

#[cfg(feature = "stats-incumbent-bench")]
mod bench {
    use fsci_stats::{ContinuousDistribution, TruncWeibullMin};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};

    const DEFAULT_ROUNDS: usize = 15;
    const DEFAULT_REPETITIONS: usize = 2_000;
    const SCREEN_ROUNDS: usize = 5;
    const SCREEN_REPETITIONS: usize = 200;
    const OUTPUTS: usize = 6;
    const ABSOLUTE_TOLERANCE: f64 = 1.0e-10;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HISTORICAL_SELF_SPEEDUP: f64 = 370.0;
    const COLLAPSE_BOUNDARY: f64 = HISTORICAL_SELF_SPEEDUP / 10.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(400);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const PUBLIC_ARMS: [&str; 3] = ["vectorized", "scalar", "frozen"];
    const PRIVATE_ARM: &str = "private";

    const PYTHON_ORACLE: &str = r#"
import hashlib
import inspect
import os
import sys
import threading
import time

import numpy as np
import scipy
from scipy import stats
from scipy.stats import _continuous_distns

C = np.array([1.5, 2.5, 3.0], dtype=np.float64)
A = np.array([0.5, 0.1, 1.0], dtype=np.float64)
B = np.array([5.0, 3.0, 10.0], dtype=np.float64)
PARAMETERS = tuple(zip(C.tolist(), A.tolist(), B.tolist()))
DIST = stats.truncweibull_min
FROZEN = tuple(DIST(c, a, b) for c, a, b in PARAMETERS)
OBSERVED_NATIVE_IDS = set()
MAX_OS_TASKS = 0

def observe():
    global MAX_OS_TASKS
    OBSERVED_NATIVE_IDS.add(threading.get_native_id())
    MAX_OS_TASKS = max(MAX_OS_TASKS, len(os.listdir("/proc/self/task")))

def interleave(mean, variance):
    output = np.empty(6, dtype=np.float64)
    output[0::2] = np.asarray(mean, dtype=np.float64)
    output[1::2] = np.asarray(variance, dtype=np.float64)
    return output

def vectorized_job():
    mean, variance = DIST.stats(C, A, B, moments="mv")
    return interleave(mean, variance)

def scalar_job():
    output = np.empty(6, dtype=np.float64)
    for index, (c, a, b) in enumerate(PARAMETERS):
        mean, variance = DIST.stats(c, a, b, moments="mv")
        output[2 * index] = mean
        output[2 * index + 1] = variance
    return output

def frozen_job():
    output = np.empty(6, dtype=np.float64)
    for index, distribution in enumerate(FROZEN):
        mean, variance = distribution.stats(moments="mv")
        output[2 * index] = mean
        output[2 * index + 1] = variance
    return output

def private_job():
    first = DIST._munp(1, C, A, B)
    second = DIST._munp(2, C, A, B)
    return interleave(first, second - first * first)

ARMS = {
    "vectorized": vectorized_job,
    "scalar": scalar_job,
    "frozen": frozen_job,
    "private": private_job,
}

for function in ARMS.values():
    observe()
    output = function()
    observe()
    if output.shape != (6,) or not np.all(np.isfinite(output)):
        raise RuntimeError("inadmissible warmup output")

source_path = inspect.getsourcefile(_continuous_distns.truncweibull_min_gen)
if source_path is None:
    raise RuntimeError("cannot resolve scipy truncweibull_min source")
with open(source_path, "rb") as source_file:
    source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
fsci_loaded = any(
    name.startswith("fsci") or name.startswith("frankenscipy")
    for name in sys.modules
)
genuine = (
    scipy.__version__ == "1.17.1"
    and DIST.__class__.__module__ == "scipy.stats._continuous_distns"
    and DIST.__class__.__name__ == "truncweibull_min_gen"
    and not fsci_loaded
)
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" dist_module={DIST.__class__.__module__}"
    f" dist_class={DIST.__class__.__name__}"
    f" scipy_engine_path={source_path}"
    f" scipy_engine_sha256={source_sha256}"
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
    if command == "CHECK" and len(fields) == 2:
        arm = fields[1]
        observe()
        output = ARMS[arm]()
        observe()
        values = ",".join(f"{value:.17e}" for value in output)
        print(
            f"CHECK {arm} {len(OBSERVED_NATIVE_IDS)} {MAX_OS_TASKS} {values}",
            flush=True,
        )
    elif command == "TIME" and len(fields) == 3:
        arm = fields[1]
        repetitions = int(fields[2])
        if repetitions <= 0:
            raise RuntimeError("repetitions must be positive")
        function = ARMS[arm]
        checksum = 0.0
        observe()
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            output = function()
            checksum += float(output[0]) + float(output[5])
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        observe()
        print(
            f"TIME {arm} {elapsed:.17e} {checksum:.17e}"
            f" {len(OBSERVED_NATIVE_IDS)} {MAX_OS_TASKS}",
            flush=True,
        )
    elif command == "QUIT":
        print("BYE", flush=True)
        break
    else:
        raise RuntimeError(f"invalid command: {line!r}")
"#;

    #[derive(Clone)]
    struct Timing {
        elapsed: f64,
    }

    struct ScipyCheck {
        arm: String,
        observed_threads: usize,
        max_os_tasks: usize,
        output: [f64; OUTPUTS],
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        stopped: bool,
    }

    impl Scipy {
        fn start() -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
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

        fn check(&mut self, arm: &str) -> Result<ScipyCheck, String> {
            let reply = self.request(&format!("CHECK {arm}"), "SciPy parity check")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "CHECK" || fields[1] != arm {
                return Err(format!("invalid SciPy parity reply: {reply}"));
            }
            let observed_threads: usize = parse(fields[2], "SciPy observed threads")?;
            let max_os_tasks: usize = parse(fields[3], "SciPy max OS tasks")?;
            let values = fields[4]
                .split(',')
                .map(|value| parse::<f64>(value, "SciPy output"))
                .collect::<Result<Vec<_>, _>>()?;
            let output: [f64; OUTPUTS] = values.try_into().map_err(|values: Vec<f64>| {
                format!("expected six outputs, got {}", values.len())
            })?;
            Ok(ScipyCheck {
                arm: arm.to_string(),
                observed_threads,
                max_os_tasks,
                output,
            })
        }

        fn time(&mut self, arm: &str, repetitions: usize) -> Result<Timing, String> {
            let reply = self.request(
                &format!("TIME {arm} {repetitions}"),
                "SciPy timed summary job",
            )?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" || fields[1] != arm {
                return Err(format!("invalid SciPy timing reply: {reply}"));
            }
            let elapsed: f64 = parse(fields[2], "SciPy elapsed")?;
            let checksum: f64 = parse(fields[3], "SciPy checksum")?;
            let observed_threads: usize = parse(fields[4], "SciPy observed threads")?;
            let max_os_tasks: usize = parse(fields[5], "SciPy max OS tasks")?;
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

    fn distributions() -> [TruncWeibullMin; 3] {
        [
            TruncWeibullMin::new(1.5, 0.5, 5.0),
            TruncWeibullMin::new(2.5, 0.1, 3.0),
            TruncWeibullMin::new(3.0, 1.0, 10.0),
        ]
    }

    fn summarize(distributions: &[TruncWeibullMin; 3]) -> [f64; OUTPUTS] {
        let mut output = [0.0; OUTPUTS];
        for (index, distribution) in distributions.iter().enumerate() {
            output[2 * index] = distribution.mean();
            output[2 * index + 1] = distribution.var();
        }
        output
    }

    fn time_ours(
        distributions: &[TruncWeibullMin; 3],
        repetitions: usize,
    ) -> Result<Timing, String> {
        let threads_before = observed_os_threads()?;
        let mut checksum = 0.0;
        let started = Instant::now();
        for _ in 0..repetitions {
            let output = summarize(black_box(distributions));
            checksum = black_box(checksum + output[0] + output[OUTPUTS - 1]);
            black_box(output);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let observed_threads = threads_before.max(observed_os_threads()?);
        black_box(checksum);
        if !elapsed.is_finite() || elapsed <= 0.0 || !checksum.is_finite() || observed_threads != 1
        {
            return Err(format!(
                "inadmissible FrankenSciPy timing: elapsed={elapsed} \
                 checksum={checksum} observed_threads={observed_threads}"
            ));
        }
        Ok(Timing { elapsed })
    }

    fn prove_parity(ours: &[f64; OUTPUTS], check: &ScipyCheck) -> Result<f64, String> {
        let mut max_abs_difference = 0.0f64;
        for (&left, right) in ours.iter().zip(check.output) {
            if !left.is_finite() || !right.is_finite() {
                return Err(format!("{} produced a non-finite output", check.arm));
            }
            max_abs_difference = max_abs_difference.max((left - right).abs());
        }
        if check.observed_threads != 1
            || check.max_os_tasks != 1
            || max_abs_difference > ABSOLUTE_TOLERANCE
        {
            return Err(format!(
                "{} failed parity: max_abs_difference={max_abs_difference:.3e} \
                 observed_threads={} max_os_tasks={}",
                check.arm, check.observed_threads, check.max_os_tasks
            ));
        }
        Ok(max_abs_difference)
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

    #[derive(Clone)]
    struct Measurement {
        ours: Vec<f64>,
        scipy: Vec<f64>,
        ratios: Vec<f64>,
        ours_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
    }

    fn effect_pair(
        scipy: &mut Scipy,
        arm: &str,
        distributions: &[TruncWeibullMin; 3],
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_ours(distributions, repetitions)?.elapsed,
                scipy.time(arm, repetitions)?.elapsed,
            ))
        } else {
            let incumbent = scipy.time(arm, repetitions)?.elapsed;
            let ours = time_ours(distributions, repetitions)?.elapsed;
            Ok((ours, incumbent))
        }
    }

    fn ours_null_pair(
        distributions: &[TruncWeibullMin; 3],
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_ours(distributions, repetitions)?.elapsed,
                time_ours(distributions, repetitions)?.elapsed,
            )
        } else {
            let right = time_ours(distributions, repetitions)?.elapsed;
            let left = time_ours(distributions, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        arm: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                scipy.time(arm, repetitions)?.elapsed,
                scipy.time(arm, repetitions)?.elapsed,
            )
        } else {
            let right = scipy.time(arm, repetitions)?.elapsed;
            let left = scipy.time(arm, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    fn measure(
        scipy: &mut Scipy,
        arm: &str,
        distributions: &[TruncWeibullMin; 3],
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = effect_pair(scipy, arm, distributions, repetitions, warmup)?;
        }
        let mut measurement = Measurement {
            ours: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            ratios: Vec::with_capacity(rounds),
            ours_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
        };
        for round in 0..rounds {
            let (ours, incumbent, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let pair = effect_pair(scipy, arm, distributions, repetitions, round)?;
                    let ours_null = ours_null_pair(distributions, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                    let pair = effect_pair(scipy, arm, distributions, repetitions, round)?;
                    let ours_null = ours_null_pair(distributions, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(distributions, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, arm, repetitions, round)?;
                    let pair = effect_pair(scipy, arm, distributions, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
            };
            measurement.ours.push(ours);
            measurement.scipy.push(incumbent);
            measurement.ratios.push(incumbent / ours);
            measurement.ours_nulls.push(ours_null);
            measurement.scipy_nulls.push(scipy_null);
            println!(
                "round={round} ours_seconds={ours:.9} scipy_seconds={incumbent:.9} \
                 ratio={:.9} ours_null={ours_null:.9} scipy_null={scipy_null:.9}",
                incumbent / ours
            );
        }
        Ok(measurement)
    }

    fn diagnostic_public_private(
        scipy: &mut Scipy,
        public_arm: &str,
        rounds: usize,
        repetitions: usize,
    ) -> Result<Vec<f64>, String> {
        let mut ratios = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (public, private) = if round.is_multiple_of(2) {
                (
                    scipy.time(public_arm, repetitions)?.elapsed,
                    scipy.time(PRIVATE_ARM, repetitions)?.elapsed,
                )
            } else {
                let private = scipy.time(PRIVATE_ARM, repetitions)?.elapsed;
                let public = scipy.time(public_arm, repetitions)?.elapsed;
                (public, private)
            };
            ratios.push(public / private);
        }
        Ok(ratios)
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
             FrankenSciPy_p50={:.9}us p95={:.9}us p99={:.9}us \
             SciPy_p50={:.9}us p95={:.9}us p99={:.9}us",
            ours_p50 * 1.0e6,
            ours_p95 * 1.0e6,
            ours_p99 * 1.0e6,
            scipy_p50 * 1.0e6,
            scipy_p95 * 1.0e6,
            scipy_p99 * 1.0e6
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

    fn read_policy_field(cpu: &str, name: &str) -> Result<String, String> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}");
        std::fs::read_to_string(&path)
            .map_err(|error| format!("read {path}: {error}"))
            .map(|value| value.trim().to_string())
    }

    fn print_hardware_provenance(affinity: &str) -> Result<(), String> {
        let (physical_cores, logical_threads) = cpu_topology()?;
        let driver = read_policy_field(affinity, "scaling_driver")?;
        let governor = read_policy_field(affinity, "scaling_governor")?;
        let preference = read_policy_field(affinity, "energy_performance_preference")
            .unwrap_or_else(|_| "unavailable".to_string());
        let minimum = read_policy_field(affinity, "scaling_min_freq")?;
        let maximum = read_policy_field(affinity, "scaling_max_freq")?;
        let cpuset_effective = std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective")
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|_| "unavailable".to_string());
        println!(
            "hardware_provenance: host_identity={} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={} numa_nodes={} \
             runtime_detected_isa={} affinity={affinity} \
             cgroup_cpuset_effective={cpuset_effective}",
            host_identity()?,
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
        if rounds < 7 || repetitions == 0 {
            return Err("require rounds>=7 and repetitions>0".to_string());
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
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
        print_hardware_provenance(&affinity)?;
        require_host_wide_quiescence("pre")?;

        let distributions = distributions();
        let ours_output = summarize(&distributions);
        if ours_output.iter().any(|value| !value.is_finite()) {
            return Err("FrankenSciPy produced non-finite parity output".to_string());
        }
        let ours_observed_threads = observed_os_threads()?;
        if ours_observed_threads != 1 {
            return Err(format!(
                "FrankenSciPy actual observed threads={ours_observed_threads}, expected one"
            ));
        }
        println!(
            "fixture=truncweibull_min-three-row-summary \
             parameters=1.5:0.5:5.0,2.5:0.1:3.0,3.0:1.0:10.0 \
             outputs=mean,var_per_row rounds={rounds} repetitions={repetitions} \
             requested_threads=1"
        );
        println!(
            "whole_job_boundary: INCLUDED=three_preconstructed_distributions,\
             three_public_mean_calls,three_public_var_calls,six_materialized_outputs; \
             EXCLUDED=process_startup,scipy_import,distribution_construction,\
             pipe_transport,backend_screening,parity_serialization,bootstrap"
        );
        println!(
            "frankenscipy_outputs={}",
            ours_output
                .iter()
                .map(|value| format!("{value:.17e}"))
                .collect::<Vec<_>>()
                .join(",")
        );

        let (mut scipy, identity) = Scipy::start()?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains("dist_module=scipy.stats._continuous_distns")
            || !identity.contains("dist_class=truncweibull_min_gen")
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
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "thread_provenance: requested_frankenscipy_threads=1 \
             actual_observed_frankenscipy_worker_threads={ours_observed_threads} \
             requested_scipy_threads=1 actual_observed_scipy_worker_threads=1 \
             actual_observed_scipy_max_os_tasks=1"
        );
        println!(
            "incumbent_screen_contract: public_arms={} \
             selection=lowest_five-sample-median_wall_time \
             full_six-output_parity_before_selection=true \
             private_munp_ineligible_for_incumbent=true",
            PUBLIC_ARMS.join(",")
        );

        let mut screened = Vec::with_capacity(PUBLIC_ARMS.len());
        for arm in PUBLIC_ARMS {
            let check = scipy.check(arm)?;
            let max_abs_difference = prove_parity(&ours_output, &check)?;
            let mut samples = Vec::with_capacity(SCREEN_ROUNDS);
            for _ in 0..SCREEN_ROUNDS {
                samples.push(scipy.time(arm, SCREEN_REPETITIONS)?.elapsed);
            }
            let screen_median = median(samples.clone());
            println!(
                "scipy_public_arm_screen: arm={arm} eligible=true \
                 max_abs_difference={max_abs_difference:.3e} \
                 observed_threads={} max_os_tasks={} \
                 repetitions={SCREEN_REPETITIONS} raw_seconds={} \
                 median_per_job_us={:.9}",
                check.observed_threads,
                check.max_os_tasks,
                csv(&samples),
                screen_median / SCREEN_REPETITIONS as f64 * 1.0e6
            );
            screened.push((arm, screen_median));
        }
        screened.sort_by(|left, right| left.1.total_cmp(&right.1));
        let selected_arm = screened[0].0;
        println!(
            "selected_public_scipy_arm={selected_arm} \
             selection_screen_outside_headline_samples=true"
        );

        let private_check = scipy.check(PRIVATE_ARM)?;
        let private_max_abs_difference = prove_parity(&ours_output, &private_check)?;
        println!(
            "scipy_private_diagnostic_proof: eligible_for_diagnostic=true \
             eligible_for_incumbent=false max_abs_difference={private_max_abs_difference:.3e} \
             observed_threads={} max_os_tasks={}",
            private_check.observed_threads, private_check.max_os_tasks
        );

        let measurement = measure(
            &mut scipy,
            selected_arm,
            &distributions,
            rounds,
            repetitions,
        )?;
        let (outcome, ratio_median, ratio_low, ratio_high) =
            print_decision(selected_arm, &measurement, repetitions);

        let diagnostic_ratios =
            diagnostic_public_private(&mut scipy, selected_arm, rounds, repetitions)?;
        let diagnostic_median = median(diagnostic_ratios.clone());
        let (diagnostic_low, diagnostic_high) = bootstrap_median_ci(&diagnostic_ratios);
        let wrapper_tax_attributed = diagnostic_low > 2.0;
        println!(
            "scipy_public_private_diagnostic: public_arm={selected_arm} \
             public_over_private_median={diagnostic_median:.9}x \
             bootstrap_median_ci95=[{diagnostic_low:.9},{diagnostic_high:.9}] \
             raw_ratios={} wrapper_tax_threshold_ci_low_gt_2=true \
             wrapper_tax_attributed={wrapper_tax_attributed}",
            csv(&diagnostic_ratios)
        );

        let collapse_confirmed = ratio_high < COLLAPSE_BOUNDARY;
        let collapse_falsified = ratio_low >= COLLAPSE_BOUNDARY;
        let direction_confirmed = ratio_low > 1.0;
        let direction_falsified = ratio_high <= 1.0;
        println!(
            "preregistered_predictions: historical_self_speedup={HISTORICAL_SELF_SPEEDUP:.1}x \
             collapse_boundary={COLLAPSE_BOUNDARY:.1}x \
             ratio_median={ratio_median:.9} ratio_ci_low={ratio_low:.9} \
             ratio_ci_high={ratio_high:.9} \
             collapse_confirmed={collapse_confirmed} \
             collapse_falsified={collapse_falsified} \
             direction_confirmed={direction_confirmed} \
             direction_falsified={direction_falsified}"
        );
        let chooser = if outcome == "DECIDED FRANKENSCIPY WIN" {
            "choose FrankenSciPy for this exact three-row mean/variance summary job"
        } else if outcome == "DECIDED FRANKENSCIPY LOSS" {
            "choose the selected public SciPy arm for this exact summary job"
        } else {
            "choose on deployment and API fit; make no speed claim"
        };
        println!("CHOOSER STATEMENT: {chooser}.");

        scipy.stop()?;
        require_host_wide_quiescence("post")?;
        println!(
            "FINAL: outcome={outcome} selected_public_scipy_arm={selected_arm} \
             ratio={ratio_median:.9} ci95=[{ratio_low:.9},{ratio_high:.9}] \
             actual_observed_frankenscipy_threads=1 \
             actual_observed_scipy_threads=1 elf_sha256={elf_sha256} \
             scipy_engine_sha256={scipy_engine_sha256}"
        );
        Ok(())
    }
}

#[cfg(feature = "stats-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "stats-incumbent-bench"))]
fn main() {
    eprintln!("perf_truncweibull_scipy requires --features stats-incumbent-bench");
    std::process::exit(2);
}
