#![forbid(unsafe_code)]
//! One-shot, live-incumbent completion cell for stored sparse cardinality.
//!
//! The candidate is public O(1) `sparse_nnz`, the same-ELF control is the old
//! forced-serial value scan now exposed as `sparse_count_nonzero`, and the live
//! incumbent is SciPy's `csr_matrix.nnz` property. All fixture work, hashing,
//! parity, Python startup, and calibration stay outside timing.

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_sparse::{
        CsrMatrix, SPARSE_COUNT_NONZERO_FORCE_SERIAL, Shape2D, sparse_count_nonzero, sparse_nnz,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::{Duration, Instant};

    const ROWS: usize = 1_048_576;
    const ENTRIES_PER_ROW: usize = 8;
    const EXPECTED_NNZ: usize = ROWS * ENTRIES_PER_ROW;
    const ROUNDS: usize = 24;
    const MIN_SAMPLE_SECONDS: f64 = 0.250;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(300);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const PYTHON: &str = "/usr/bin/python3.13";
    const HARNESS_SOURCE: &[u8] = include_bytes!("perf_sparse_nnz.rs");
    const PRODUCTION_SOURCE: &[u8] = include_bytes!("../linalg.rs");

    const PYTHON_ORACLE: &str = r#"
import hashlib
import os
import struct
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse._compressed as sparse_engine

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def task_count():
    return len(os.listdir("/proc/self/task"))

engine_path = sparse_engine.__file__
python_sha256 = file_sha256(sys.executable)
engine_sha256 = file_sha256(engine_path)
oracle_sha256 = os.environ["FSCI_NNZ_ORACLE_SHA256"]
affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
print(
    "READY"
    f" scipy={scipy.__version__}"
    f" numpy={np.__version__}"
    f" python={sys.executable}"
    f" python_sha256={python_sha256}"
    f" scipy_engine={engine_path}"
    f" scipy_engine_sha256={engine_sha256}"
    f" oracle_sha256={oracle_sha256}"
    f" affinity={affinity}"
    f" actual_observed_worker_threads={task_count()}"
    " fsci_loaded=False genuine=True",
    flush=True,
)

matrix = None

for raw in sys.stdin:
    fields = raw.strip().split()
    if not fields:
        continue
    command = fields[0]
    if command == "INIT":
        n = int(fields[1])
        width = int(fields[2])
        nnz = n * width
        flat = np.arange(nnz, dtype=np.int64)
        rows = flat // width
        indices = (rows // width) * width + (flat % width)
        data = 1.0 + ((rows + indices) % 17).astype(np.float64) / 64.0
        indptr = np.arange(0, nnz + 1, width, dtype=np.int64)
        matrix = csr_matrix((data, indices, indptr), shape=(n, n), copy=False)
        del flat, rows
        print(
            "CASE"
            f" rows={matrix.shape[0]} cols={matrix.shape[1]} nnz={matrix.nnz}"
            f" sorted={matrix.has_sorted_indices} canonical={matrix.has_canonical_format}"
            f" finite={bool(np.isfinite(matrix.data).all())}"
            f" numerical_nonzero={matrix.count_nonzero()}",
            flush=True,
        )
    elif command == "INPUT_SHA256":
        h = hashlib.sha256()
        h.update(struct.pack("<Q", matrix.shape[0]))
        h.update(struct.pack("<Q", matrix.nnz))
        h.update(np.asarray(matrix.data, dtype="<f8").tobytes())
        h.update(np.asarray(matrix.indices, dtype="<u8").tobytes())
        h.update(np.asarray(matrix.indptr, dtype="<u8").tobytes())
        print(f"INPUT_SHA256 {h.hexdigest()}", flush=True)
    elif command == "EXPLICIT_ZERO":
        z = csr_matrix(
            (
                np.array([0.0, 2.0], dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
                np.array([0, 1, 2], dtype=np.int64),
            ),
            shape=(2, 2),
            copy=False,
        )
        print(f"ZERO stored={z.nnz} numerical={z.count_nonzero()}", flush=True)
    elif command == "RUN":
        repetitions = int(fields[1])
        last = 0
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            last = matrix.nnz
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        print(
            f"TIME {elapsed:.17e} {matrix.nnz} {task_count()} {last}",
            flush=True,
        )
    elif command == "QUIT":
        break
    else:
        print(f"FATAL unknown-command={command}", flush=True)
        break
"#;

    #[derive(Clone, Copy)]
    enum Arm {
        Candidate,
        Control,
        Live,
    }

    const ORDERS: [[Arm; 3]; 6] = [
        [Arm::Candidate, Arm::Control, Arm::Live],
        [Arm::Candidate, Arm::Live, Arm::Control],
        [Arm::Control, Arm::Candidate, Arm::Live],
        [Arm::Control, Arm::Live, Arm::Candidate],
        [Arm::Live, Arm::Candidate, Arm::Control],
        [Arm::Live, Arm::Control, Arm::Candidate],
    ];

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    impl Scipy {
        fn start() -> Result<(Self, String, String), String> {
            let oracle_sha256 = sha256_bytes(PYTHON_ORACLE.as_bytes());
            let mut child = Command::new(PYTHON)
                .arg("-u")
                .arg("-c")
                .arg(PYTHON_ORACLE)
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .env("FSCI_NNZ_ORACLE_SHA256", &oracle_sha256)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| format!("spawn live SciPy: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy has no stdin".to_string())?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| "live SciPy has no stdout".to_string())?;
            let mut scipy = Self {
                child,
                stdin,
                stdout: BufReader::new(stdout),
            };
            let identity = scipy.read_reply("identity")?;
            Ok((scipy, identity, oracle_sha256))
        }

        fn read_reply(&mut self, context: &str) -> Result<String, String> {
            let mut line = String::new();
            self.stdout
                .read_line(&mut line)
                .map_err(|error| format!("read live SciPy {context}: {error}"))?;
            let line = line.trim().to_string();
            if line.is_empty() {
                return Err(format!("live SciPy exited while reading {context}"));
            }
            if line.starts_with("FATAL ") {
                return Err(format!("live SciPy rejected {context}: {line}"));
            }
            Ok(line)
        }

        fn initialize(&mut self) -> Result<String, String> {
            writeln!(self.stdin, "INIT {ROWS} {ENTRIES_PER_ROW}")
                .map_err(|error| format!("write live INIT: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush live INIT: {error}"))?;
            self.read_reply("fixture")
        }

        fn input_sha256(&mut self) -> Result<String, String> {
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write live INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush live INPUT_SHA256: {error}"))?;
            let reply = self.read_reply("input SHA-256")?;
            let digest = reply
                .strip_prefix("INPUT_SHA256 ")
                .ok_or_else(|| format!("malformed live input digest: {reply}"))?;
            if !is_sha256(digest) {
                return Err(format!("invalid live input digest: {digest}"));
            }
            Ok(digest.to_string())
        }

        fn explicit_zero(&mut self) -> Result<(usize, usize), String> {
            writeln!(self.stdin, "EXPLICIT_ZERO")
                .map_err(|error| format!("write live EXPLICIT_ZERO: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush live EXPLICIT_ZERO: {error}"))?;
            let reply = self.read_reply("explicit-zero parity")?;
            Ok((
                parse_field(&reply, "stored=")?,
                parse_field(&reply, "numerical=")?,
            ))
        }

        fn time_nnz(&mut self, repetitions: usize) -> Result<f64, String> {
            writeln!(self.stdin, "RUN {repetitions}")
                .map_err(|error| format!("write live RUN: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush live RUN: {error}"))?;
            let reply = self.read_reply("timing")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "TIME" {
                return Err(format!("malformed live timing: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse live elapsed: {error}"))?;
            let stored = fields[2]
                .parse::<usize>()
                .map_err(|error| format!("parse live stored count: {error}"))?;
            let threads = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse live task count: {error}"))?;
            let checksum = fields[4]
                .parse::<usize>()
                .map_err(|error| format!("parse live checksum: {error}"))?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || stored != EXPECTED_NNZ
                || checksum != EXPECTED_NNZ
                || threads != 1
            {
                return Err(format!("inadmissible live timing: {reply}"));
            }
            Ok(elapsed)
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    struct ForceSerialGuard;

    impl ForceSerialGuard {
        fn activate() -> Self {
            SPARSE_COUNT_NONZERO_FORCE_SERIAL.store(true, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for ForceSerialGuard {
        fn drop(&mut self) {
            SPARSE_COUNT_NONZERO_FORCE_SERIAL.store(false, Ordering::SeqCst);
        }
    }

    fn fixture() -> CsrMatrix {
        let mut data = Vec::with_capacity(EXPECTED_NNZ);
        let mut indices = Vec::with_capacity(EXPECTED_NNZ);
        let mut indptr = Vec::with_capacity(ROWS + 1);
        indptr.push(0);
        for row in 0..ROWS {
            let base = (row / ENTRIES_PER_ROW) * ENTRIES_PER_ROW;
            for slot in 0..ENTRIES_PER_ROW {
                let column = base + slot;
                indices.push(column);
                data.push(1.0 + ((row + column) % 17) as f64 / 64.0);
            }
            indptr.push(data.len());
        }
        CsrMatrix::from_components(Shape2D::new(ROWS, ROWS), data, indices, indptr, false)
            .expect("canonical registered CSR")
    }

    fn explicit_zero_fixture() -> CsrMatrix {
        CsrMatrix::from_components(
            Shape2D::new(2, 2),
            vec![0.0, 2.0],
            vec![0, 1],
            vec![0, 1, 2],
            false,
        )
        .expect("canonical explicit-zero CSR")
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        std::fs::read(path)
            .map(|bytes| sha256_bytes(&bytes))
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))
    }

    fn canonical_input_sha256(matrix: &CsrMatrix) -> Result<String, String> {
        let mut digest = Sha256::new();
        for (label, value) in [
            ("row count", matrix.shape().rows),
            ("stored count", matrix.nnz()),
        ] {
            let value = u64::try_from(value)
                .map_err(|error| format!("{label} does not fit canonical u64: {error}"))?;
            digest.update(value.to_le_bytes());
        }
        for &value in matrix.data() {
            digest.update(value.to_le_bytes());
        }
        for (label, values) in [
            ("column index", matrix.indices()),
            ("row pointer", matrix.indptr()),
        ] {
            for &value in values {
                let value = u64::try_from(value)
                    .map_err(|error| format!("{label} does not fit canonical u64: {error}"))?;
                digest.update(value.to_le_bytes());
            }
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn field_value<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|field| field.strip_prefix(prefix))
    }

    fn parse_field<T>(line: &str, prefix: &str) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        field_value(line, prefix)
            .ok_or_else(|| format!("missing {prefix} in {line}"))?
            .parse::<T>()
            .map_err(|error| format!("parse {prefix} in {line}: {error}"))
    }

    fn time_candidate(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let mut last = 0usize;
        let started = Instant::now();
        for _ in 0..repetitions {
            last = black_box(sparse_nnz(black_box(matrix)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        assert_eq!(last, EXPECTED_NNZ, "candidate timed checksum");
        elapsed
    }

    fn time_control(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let _guard = ForceSerialGuard::activate();
        let mut last = 0usize;
        let started = Instant::now();
        for _ in 0..repetitions {
            last = black_box(sparse_count_nonzero(black_box(matrix)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        assert_eq!(last, EXPECTED_NNZ, "control timed checksum");
        elapsed
    }

    fn calibrate_local(matrix: &CsrMatrix, arm: Arm) -> Result<(usize, f64), String> {
        let mut repetitions = 1usize;
        loop {
            let elapsed = match arm {
                Arm::Candidate => time_candidate(matrix, repetitions),
                Arm::Control => time_control(matrix, repetitions),
                Arm::Live => return Err("cannot locally calibrate the live arm".to_string()),
            };
            if elapsed >= MIN_SAMPLE_SECONDS {
                return Ok((repetitions, elapsed));
            }
            repetitions = repetitions
                .checked_mul(2)
                .ok_or_else(|| "local calibration repetition count overflowed".to_string())?;
        }
    }

    fn calibrate_live(scipy: &mut Scipy) -> Result<(usize, f64), String> {
        let mut repetitions = 1usize;
        loop {
            let elapsed = scipy.time_nnz(repetitions)?;
            if elapsed >= MIN_SAMPLE_SECONDS {
                return Ok((repetitions, elapsed));
            }
            repetitions = repetitions
                .checked_mul(2)
                .ok_or_else(|| "live calibration repetition count overflowed".to_string())?;
        }
    }

    fn time_arm(
        arm: Arm,
        matrix: &CsrMatrix,
        scipy: &mut Scipy,
        candidate_repetitions: usize,
        control_repetitions: usize,
        live_repetitions: usize,
    ) -> Result<f64, String> {
        match arm {
            Arm::Candidate => Ok(time_candidate(matrix, candidate_repetitions)),
            Arm::Control => Ok(time_control(matrix, control_repetitions)),
            Arm::Live => scipy.time_nnz(live_repetitions),
        }
    }

    fn symmetrized_null(
        arm: Arm,
        matrix: &CsrMatrix,
        scipy: &mut Scipy,
        candidate_repetitions: usize,
        control_repetitions: usize,
        live_repetitions: usize,
    ) -> Result<f64, String> {
        let left_forward = time_arm(
            arm,
            matrix,
            scipy,
            candidate_repetitions,
            control_repetitions,
            live_repetitions,
        )?;
        let right_forward = time_arm(
            arm,
            matrix,
            scipy,
            candidate_repetitions,
            control_repetitions,
            live_repetitions,
        )?;
        let right_reverse = time_arm(
            arm,
            matrix,
            scipy,
            candidate_repetitions,
            control_repetitions,
            live_repetitions,
        )?;
        let left_reverse = time_arm(
            arm,
            matrix,
            scipy,
            candidate_repetitions,
            control_repetitions,
            live_repetitions,
        )?;
        Ok(((left_forward / right_forward) * (left_reverse / right_reverse)).sqrt())
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
        } else {
            values[values.len() / 2]
        }
    }

    fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let index = ((values.len().saturating_sub(1)) as f64 * quantile).ceil() as usize;
        values[index.min(values.len().saturating_sub(1))]
    }

    fn bootstrap_median_ci(values: &[f64], seed: u64) -> (f64, f64) {
        let mut state = seed;
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

    fn cv(values: &[f64]) -> f64 {
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
            .map(|value| format!("{value:.12e}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    #[derive(Clone, Copy)]
    struct CpuTicks {
        total: u64,
        idle: u64,
    }

    fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
        let contents = std::fs::read_to_string("/proc/stat")
            .map_err(|error| format!("read /proc/stat: {error}"))?;
        let mut cpus = BTreeMap::new();
        for line in contents.lines() {
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
            let cpu = suffix
                .parse::<usize>()
                .map_err(|error| format!("parse CPU index {suffix}: {error}"))?;
            let ticks = fields
                .map(|field| {
                    field
                        .parse::<u64>()
                        .map_err(|error| format!("parse cpu{cpu} tick {field}: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("cpu{cpu} /proc/stat row is too short"));
            }
            cpus.insert(
                cpu,
                CpuTicks {
                    total: ticks.iter().copied().sum(),
                    idle: ticks[3].saturating_add(ticks[4]),
                },
            );
        }
        if cpus.is_empty() {
            return Err("no per-CPU rows found in /proc/stat".to_string());
        }
        Ok(cpus)
    }

    fn require_host_wide_quiescence(phase: &str) -> Result<bool, String> {
        let before = read_cpu_ticks()?;
        thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        let mut busy_cpus = Vec::new();
        let mut maximum_busy_fraction = 0.0f64;
        for (cpu, start) in &before {
            let end = after
                .get(cpu)
                .ok_or_else(|| format!("cpu{cpu} disappeared during quiescence sample"))?;
            let total = end.total.saturating_sub(start.total);
            let idle = end.idle.saturating_sub(start.idle);
            let busy_fraction = if total == 0 {
                1.0
            } else {
                total.saturating_sub(idle) as f64 / total as f64
            };
            maximum_busy_fraction = maximum_busy_fraction.max(busy_fraction);
            if busy_fraction > HOST_QUIESCENCE_MAX_BUSY {
                busy_cpus.push(format!("cpu{cpu}={:.1}%", busy_fraction * 100.0));
            }
        }
        if busy_cpus.is_empty() {
            println!(
                "host_wide_quiescence_{phase}=clear sampled_cpus={} \
                 maximum_busy_fraction={maximum_busy_fraction:.3} \
                 busy_cpu_count_above_limit=0 limit={HOST_QUIESCENCE_MAX_BUSY:.3}",
                before.len()
            );
            return Ok(true);
        }
        if std::env::var_os("FSCI_SPARSE_ALLOW_NON_EXCLUSIVE").is_none_or(|value| value != "1") {
            return Err(format!(
                "host-wide quiescence failed during {phase}: {}",
                busy_cpus.join(",")
            ));
        }
        println!(
            "host_wide_quiescence_{phase}=NOT_CERTIFIED evidence_class=PROVISIONAL_NON_EXCLUSIVE \
             sampled_cpus={} maximum_busy_fraction={maximum_busy_fraction:.3} \
             busy_cpu_count_above_limit={} limit={HOST_QUIESCENCE_MAX_BUSY:.3} busy_cpus={}",
            before.len(),
            busy_cpus.len(),
            busy_cpus.join(",")
        );
        Ok(false)
    }

    fn cpu_affinity() -> Result<String, String> {
        std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "Cpus_allowed_list is unavailable".to_string())
    }

    fn hardware_provenance() -> Result<(), String> {
        let affinity = cpu_affinity()?;
        if affinity != "25" {
            return Err(format!(
                "registered completion cell requires CPU 25, observed affinity={affinity}"
            ));
        }
        let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map_err(|error| format!("read hostname: {error}"))?;
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let logical_threads = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        let mut physical_cores = HashSet::new();
        for block in cpuinfo.split("\n\n") {
            let package = block.lines().find_map(|line| {
                line.strip_prefix("physical id")
                    .and_then(|field| field.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            let core = block.lines().find_map(|line| {
                line.strip_prefix("core id")
                    .and_then(|field| field.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            if let (Some(package), Some(core)) = (package, core) {
                physical_cores.insert((package.to_string(), core.to_string()));
            }
        }
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let ram_kib = meminfo
            .lines()
            .find_map(|line| {
                line.strip_prefix("MemTotal:")
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .ok_or_else(|| "MemTotal is unavailable".to_string())?;
        let numa_nodes = std::fs::read_dir("/sys/devices/system/node")
            .map_err(|error| format!("read NUMA topology: {error}"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.strip_prefix("node").is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                    })
                })
            })
            .count();
        let governor =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu25/cpufreq/scaling_governor")
                .map_err(|error| format!("read CPU 25 governor: {error}"))?;
        let observed_threads = std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("read process tasks: {error}"))?
            .count();
        let available_parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .map_err(|error| format!("read available parallelism: {error}"))?;
        if observed_threads != 1 || available_parallelism != 1 {
            return Err(format!(
                "one-thread provenance failed: observed_tasks={observed_threads} \
                 available_parallelism={available_parallelism}"
            ));
        }
        println!(
            "hardware_provenance: host_identity={} physical_cores={} logical_threads={} \
             ram_bytes={} numa_nodes={} affinity={affinity} requested_threads=1 \
             actual_observed_worker_threads=1/1/1 available_parallelism=1 \
             runtime_detected_isa=sse2:{},sse4_2:{},avx2:{},fma:{},bmi2:{},vaes:{},avx512f:{} \
             scaling_governor={}",
            hostname.trim(),
            physical_cores.len(),
            logical_threads,
            ram_kib * 1_024,
            numa_nodes,
            std::is_x86_feature_detected!("sse2"),
            std::is_x86_feature_detected!("sse4.2"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("bmi2"),
            std::is_x86_feature_detected!("vaes"),
            std::is_x86_feature_detected!("avx512f"),
            governor.trim()
        );
        Ok(())
    }

    fn build_id(executable: &Path) -> Result<String, String> {
        let output = Command::new("/usr/bin/readelf")
            .arg("-n")
            .arg(executable)
            .output()
            .map_err(|error| format!("run readelf: {error}"))?;
        if !output.status.success() {
            return Err("readelf failed to inspect the benchmark ELF".to_string());
        }
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .find_map(|line| line.trim().strip_prefix("Build ID:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "GNU Build ID is unavailable".to_string())
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required provenance variable {name} is absent"))
    }

    pub fn run() -> Result<(), String> {
        let arguments = std::env::args().collect::<Vec<_>>();
        if arguments.len() != 1 {
            return Err("registered sparse_nnz cell takes no arguments".to_string());
        }
        let rch_worker = required_env("FSCI_RCH_WORKER")?;
        let rch_route = required_env("FSCI_RCH_ROUTE")?;
        let source_commit = required_env("FSCI_SOURCE_COMMIT")?;
        let claim_message_id = required_env("FSCI_CLAIM_MESSAGE_ID")?;
        let release_message_id = required_env("FSCI_RELEASE_MESSAGE_ID")?;
        let filesystem_lock = required_env("FSCI_FILESYSTEM_LOCK_HELD")?;
        if filesystem_lock != "1" {
            return Err("registered filesystem lock is not held".to_string());
        }
        hardware_provenance()?;
        let executable = std::env::current_exe()
            .map_err(|error| format!("resolve current executable: {error}"))?;
        let elf_sha256 = sha256_file(&executable)?;
        let build_id = build_id(&executable)?;
        let harness_source_sha256 = sha256_bytes(HARNESS_SOURCE);
        let production_source_sha256 = sha256_bytes(PRODUCTION_SOURCE);
        println!(
            "build_provenance: strict_rch=true worker={rch_worker} route={rch_route} \
             source_commit={source_commit} elf={} elf_sha256={elf_sha256} \
             gnu_build_id={build_id} harness_source_sha256={harness_source_sha256} \
             production_source_sha256={production_source_sha256}",
            executable.display()
        );
        println!(
            "coordination: agent_name={} claim_message_id={claim_message_id} \
             release_message_id={release_message_id} filesystem_lock_held=true",
            std::env::var("AGENT_NAME").unwrap_or_else(|_| "unknown".to_string())
        );

        let pre_quiet = require_host_wide_quiescence("pre")?;
        let matrix = fixture();
        if matrix.shape() != Shape2D::new(ROWS, ROWS)
            || matrix.nnz() != EXPECTED_NNZ
            || matrix
                .data()
                .iter()
                .any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err("registered Rust CSR fixture contract failed".to_string());
        }
        let rust_input_sha256 = canonical_input_sha256(&matrix)?;
        let explicit_zero = explicit_zero_fixture();
        let rust_zero_stored = sparse_nnz(&explicit_zero);
        let rust_zero_numerical = sparse_count_nonzero(&explicit_zero);

        let (mut scipy, identity, oracle_sha256) = Scipy::start()?;
        println!("live_identity: {identity}");
        if !identity.starts_with("READY ")
            || !identity.contains("scipy=1.17.1")
            || !identity.contains(&format!("python={PYTHON}"))
            || !identity.contains("affinity=25")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy genuine-incumbent identity gate failed".to_string());
        }
        for prefix in ["python_sha256=", "scipy_engine_sha256=", "oracle_sha256="] {
            let digest = field_value(&identity, prefix)
                .ok_or_else(|| format!("live identity omitted {prefix}"))?;
            if !is_sha256(digest) {
                return Err(format!("live identity reported invalid {prefix}{digest}"));
            }
        }
        if field_value(&identity, "oracle_sha256=") != Some(oracle_sha256.as_str()) {
            return Err("embedded oracle identity mismatch".to_string());
        }
        let live_case = scipy.initialize()?;
        let expected_case = format!(
            "CASE rows={ROWS} cols={ROWS} nnz={EXPECTED_NNZ} sorted=True canonical=True \
             finite=True numerical_nonzero={EXPECTED_NNZ}"
        );
        if live_case != expected_case {
            return Err(format!("live SciPy fixture contract failed: {live_case}"));
        }
        println!("live_case: {live_case}");
        let live_input_sha256 = scipy.input_sha256()?;
        if rust_input_sha256 != live_input_sha256 {
            return Err(format!(
                "input digest mismatch: rust={rust_input_sha256} live={live_input_sha256}"
            ));
        }
        println!(
            "input_sha256={rust_input_sha256} rust_input_sha256={rust_input_sha256} \
             live_input_sha256={live_input_sha256} input_digest_match=true"
        );
        let (live_zero_stored, live_zero_numerical) = scipy.explicit_zero()?;
        if (
            rust_zero_stored,
            rust_zero_numerical,
            live_zero_stored,
            live_zero_numerical,
        ) != (2, 1, 2, 1)
        {
            return Err(format!(
                "explicit-zero parity failed: rust={rust_zero_stored}/{rust_zero_numerical} \
                 live={live_zero_stored}/{live_zero_numerical}"
            ));
        }
        println!(
            "explicit_zero_parity: rust_stored={rust_zero_stored} \
             rust_numerical={rust_zero_numerical} live_stored={live_zero_stored} \
             live_numerical={live_zero_numerical} pass=true"
        );

        let (candidate_repetitions, candidate_calibration_seconds) =
            calibrate_local(&matrix, Arm::Candidate)?;
        let (control_repetitions, control_calibration_seconds) =
            calibrate_local(&matrix, Arm::Control)?;
        let (live_repetitions, live_calibration_seconds) = calibrate_live(&mut scipy)?;
        let calibration_gate = candidate_calibration_seconds >= MIN_SAMPLE_SECONDS
            && control_calibration_seconds >= MIN_SAMPLE_SECONDS
            && live_calibration_seconds >= MIN_SAMPLE_SECONDS;
        println!(
            "calibration: candidate_repetitions={candidate_repetitions} \
             candidate_seconds={candidate_calibration_seconds:.9} \
             control_repetitions={control_repetitions} \
             control_seconds={control_calibration_seconds:.9} \
             live_repetitions={live_repetitions} live_seconds={live_calibration_seconds:.9} \
             min_sample_seconds={MIN_SAMPLE_SECONDS:.3} separate_per_arm=true \
             calibration_gate={calibration_gate}"
        );

        let measurement_quiet = require_host_wide_quiescence("measurement")?;
        let mut candidate_times = Vec::with_capacity(ROUNDS);
        let mut control_times = Vec::with_capacity(ROUNDS);
        let mut live_times = Vec::with_capacity(ROUNDS);
        let mut candidate_nulls = Vec::with_capacity(ROUNDS);
        let mut control_nulls = Vec::with_capacity(ROUNDS);
        let mut live_nulls = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            let mut candidate_batch = 0.0;
            let mut control_batch = 0.0;
            let mut live_batch = 0.0;
            for arm in ORDERS[round % ORDERS.len()] {
                let elapsed = time_arm(
                    arm,
                    &matrix,
                    &mut scipy,
                    candidate_repetitions,
                    control_repetitions,
                    live_repetitions,
                )?;
                match arm {
                    Arm::Candidate => candidate_batch = elapsed,
                    Arm::Control => control_batch = elapsed,
                    Arm::Live => live_batch = elapsed,
                }
            }
            candidate_times.push(candidate_batch / candidate_repetitions as f64);
            control_times.push(control_batch / control_repetitions as f64);
            live_times.push(live_batch / live_repetitions as f64);
            for arm in ORDERS[(round + 3) % ORDERS.len()] {
                let null = symmetrized_null(
                    arm,
                    &matrix,
                    &mut scipy,
                    candidate_repetitions,
                    control_repetitions,
                    live_repetitions,
                )?;
                match arm {
                    Arm::Candidate => candidate_nulls.push(null),
                    Arm::Control => control_nulls.push(null),
                    Arm::Live => live_nulls.push(null),
                }
            }
        }
        let post_quiet = require_host_wide_quiescence("post")?;
        scipy.quit();

        let control_ratios = control_times
            .iter()
            .zip(&candidate_times)
            .map(|(control, candidate)| control / candidate)
            .collect::<Vec<_>>();
        let live_ratios = live_times
            .iter()
            .zip(&candidate_times)
            .map(|(live, candidate)| live / candidate)
            .collect::<Vec<_>>();
        let candidate_p50 = median(candidate_times.clone());
        let candidate_p95 = percentile(candidate_times.clone(), 0.95);
        let candidate_p99 = percentile(candidate_times.clone(), 0.99);
        let control_p50 = median(control_times.clone());
        let control_p95 = percentile(control_times.clone(), 0.95);
        let control_p99 = percentile(control_times.clone(), 0.99);
        let live_p50 = median(live_times.clone());
        let live_p95 = percentile(live_times.clone(), 0.95);
        let live_p99 = percentile(live_times.clone(), 0.99);
        let control_ratio_median = median(control_ratios.clone());
        let live_ratio_median = median(live_ratios.clone());
        let (control_ratio_low, control_ratio_high) =
            bootstrap_median_ci(&control_ratios, 0x243f_6a88_85a3_08d3);
        let (live_ratio_low, live_ratio_high) =
            bootstrap_median_ci(&live_ratios, 0x1319_8a2e_0370_7344);
        let candidate_null_median = median(candidate_nulls.clone());
        let control_null_median = median(control_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let candidate_null_ci = bootstrap_median_ci(&candidate_nulls, 0xa409_3822_299f_31d0);
        let control_null_ci = bootstrap_median_ci(&control_nulls, 0x082e_fa98_ec4e_6c89);
        let live_null_ci = bootstrap_median_ci(&live_nulls, 0x4528_21e6_38d0_1377);
        let null_medians_ok = [candidate_null_median, control_null_median, live_null_median]
            .iter()
            .all(|median| (median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT);
        let widest_null_half_width = [candidate_null_ci, control_null_ci, live_null_ci]
            .iter()
            .map(|(low, high)| (high - low) / 2.0)
            .fold(0.0f64, f64::max);
        let widest_null_endpoint_margin = [candidate_null_ci, control_null_ci, live_null_ci]
            .iter()
            .map(|(low, high)| (high - 1.0).abs().max((1.0 / low - 1.0).abs()))
            .fold(0.0f64, f64::max);
        let control_clears_null = control_ratio_low - 1.0 > 2.0 * widest_null_half_width
            && control_ratio_low - 1.0 > 2.0 * widest_null_endpoint_margin;
        let live_clears_null = live_ratio_low - 1.0 > 2.0 * widest_null_half_width
            && live_ratio_low - 1.0 > 2.0 * widest_null_endpoint_margin;
        let tail_gate = candidate_p95 < control_p95
            && candidate_p95 < live_p95
            && candidate_p99 < control_p99
            && candidate_p99 < live_p99;
        let performance_gate = calibration_gate
            && null_medians_ok
            && control_clears_null
            && live_clears_null
            && control_ratio_low > 100.0
            && live_ratio_low > 5.0
            && tail_gate;
        let exclusivity_certified = pre_quiet && measurement_quiet && post_quiet;
        let coordination_certified = claim_message_id != "0" && release_message_id != "0";
        let decided_keep = performance_gate && exclusivity_certified && coordination_certified;

        println!(
            "timing_ms: candidate_p50={:.9} candidate_p95={:.9} candidate_p99={:.9} \
             control_p50={:.9} control_p95={:.9} control_p99={:.9} \
             live_p50={:.9} live_p95={:.9} live_p99={:.9} \
             candidate_cv={:.6} control_cv={:.6} live_cv={:.6}",
            candidate_p50 * 1_000.0,
            candidate_p95 * 1_000.0,
            candidate_p99 * 1_000.0,
            control_p50 * 1_000.0,
            control_p95 * 1_000.0,
            control_p99 * 1_000.0,
            live_p50 * 1_000.0,
            live_p95 * 1_000.0,
            live_p99 * 1_000.0,
            cv(&candidate_times),
            cv(&control_times),
            cv(&live_times)
        );
        println!(
            "effects: control_over_candidate_median={control_ratio_median:.6} \
             control_over_candidate_ci95=[{control_ratio_low:.6},{control_ratio_high:.6}] \
             live_over_candidate_median={live_ratio_median:.6} \
             live_over_candidate_ci95=[{live_ratio_low:.6},{live_ratio_high:.6}] \
             control_ratio_cv={:.6} live_ratio_cv={:.6}",
            cv(&control_ratios),
            cv(&live_ratios)
        );
        println!(
            "nulls: design=four-call-forward-reverse-geometric-symmetrization \
             candidate_median={candidate_null_median:.6} \
             candidate_ci95=[{:.6},{:.6}] control_median={control_null_median:.6} \
             control_ci95=[{:.6},{:.6}] live_median={live_null_median:.6} \
             live_ci95=[{:.6},{:.6}] widest_half_width={widest_null_half_width:.6} \
             widest_endpoint_margin={widest_null_endpoint_margin:.6} \
             null_medians_within_2pct={null_medians_ok}",
            candidate_null_ci.0,
            candidate_null_ci.1,
            control_null_ci.0,
            control_null_ci.1,
            live_null_ci.0,
            live_null_ci.1
        );
        println!(
            "registered_gate: performance_gate={performance_gate} decided_keep={decided_keep} \
             calibration_gate={calibration_gate} control_clears_2x_null={control_clears_null} \
             live_clears_2x_null={live_clears_null} control_ci_low_gt_100={} \
             live_ci_low_gt_5={} tail_gate={tail_gate} \
             exclusivity_certified={exclusivity_certified} \
             coordination_certified={coordination_certified}",
            control_ratio_low > 100.0,
            live_ratio_low > 5.0
        );
        println!("raw_candidate_per_call_seconds={}", csv(&candidate_times));
        println!("raw_control_per_call_seconds={}", csv(&control_times));
        println!("raw_live_per_call_seconds={}", csv(&live_times));
        println!("raw_control_over_candidate={}", csv(&control_ratios));
        println!("raw_live_over_candidate={}", csv(&live_ratios));
        println!("raw_candidate_aa={}", csv(&candidate_nulls));
        println!("raw_control_aa={}", csv(&control_nulls));
        println!("raw_live_aa={}", csv(&live_nulls));
        println!(
            "verdict={} semantic_correction=KEEP exact_cell_rerun_forbidden=true \
             timing_evidence_class={} external_resource_accounting=gnu-time-v",
            if decided_keep {
                "KEEP"
            } else if performance_gate {
                "PROVISIONAL_KEEP"
            } else {
                "REVERT_PERFORMANCE_CLAIM"
            },
            if exclusivity_certified && coordination_certified {
                "DECIDED"
            } else {
                "PROVISIONAL_NON_EXCLUSIVE"
            }
        );
        Ok(())
    }
}

#[cfg(feature = "sparse-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("FATAL: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "sparse-incumbent-bench"))]
fn main() {
    eprintln!("perf_sparse_nnz requires --features sparse-incumbent-bench");
    std::process::exit(2);
}
