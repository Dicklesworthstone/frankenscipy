//! Profiling-only harness for sparse hot paths.
//!
//! NOT a product binary. It exists so RCH, hyperfine, and sha256 checks can
//! attach to deterministic sparse arithmetic scenarios.
//!
//! Usage:
//!   `perf_sparse add-csr <n> <density> <repeats>`
//!   `perf_sparse add-csr-golden [path]`
//!   `perf_sparse spilu <n> <half_bandwidth> <repeats>`
//!   `perf_sparse spilu-golden [path]`
//!   `perf_sparse expm-current-profile <n> <repeats>`
//!   `perf_sparse expm-vs-scipy <n> <rounds> [oracle]`
//!   `perf_sparse laplacian-current-profile <n> <repeats>`
//!   `perf_sparse laplacian-vs-scipy <n> <rounds> [oracle]`
//!   `perf_sparse laplacian-cycle-current-profile <n> <repeats>`
//!   `perf_sparse laplacian-cycle-vs-scipy <n> <rounds> [oracle]`
//!   `perf_sparse laplacian-torus-candidate-vs-scipy <side> <rounds> [oracle]`
//!   `perf_sparse csc-add-current-profile <n> <repeats>`
//!   `perf_sparse csc-add-vs-scipy <n> <rounds> [oracle]`
//!   `perf_sparse transpose-view-vs-scipy <rows> <rounds> [oracle]`

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, IluOptions, Shape2D, add_csr, diags,
    random, scale_csr, spilu,
};

#[cfg(feature = "sparse-incumbent-bench")]
mod expm_bench {
    use fsci_sparse::linalg::{ExpmOptions, LAPLACIAN_FORCE_DENSE_REFERENCE, expm, laplacian};
    use fsci_sparse::{
        CscMatrix, CsrMatrix, Shape2D, add_csc, sparse_transpose, sparse_transpose_view,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::{Duration, Instant};

    const REGISTERED_N: usize = 384;
    const REGISTERED_ROUNDS: usize = 21;
    const LAPLACIAN_REGISTERED_N: usize = 4_096;
    const LAPLACIAN_RESULT_NNZ: usize = 3 * LAPLACIAN_REGISTERED_N - 2;
    const CYCLE_REGISTERED_N: usize = 6_144;
    const CYCLE_REGISTERED_ROUNDS: usize = 22;
    const CYCLE_RESULT_NNZ: usize = 3 * CYCLE_REGISTERED_N;
    const TORUS_SIDE: usize = 96;
    const TORUS_N: usize = TORUS_SIDE * TORUS_SIDE;
    const TORUS_INPUT_NNZ: usize = 4 * TORUS_N;
    const TORUS_RESULT_NNZ: usize = 5 * TORUS_N;
    const TORUS_REGISTERED_ROUNDS: usize = 24;
    const CSC_ADD_REGISTERED_N: usize = 4_096;
    const CSC_ADD_ENTRIES_PER_COLUMN: usize = 24;
    const CSC_ADD_REGISTERED_ROUNDS: usize = 24;
    const TRANSPOSE_ROWS: usize = 262_144;
    const TRANSPOSE_COLS: usize = 131_072;
    const TRANSPOSE_ENTRIES_PER_ROW: usize = 8;
    const TRANSPOSE_NNZ: usize = TRANSPOSE_ROWS * TRANSPOSE_ENTRIES_PER_ROW;
    const TRANSPOSE_REGISTERED_ROUNDS: usize = 24;
    const MIN_SAMPLE_SECONDS: f64 = 0.005;
    const CYCLE_MIN_SAMPLE_SECONDS: f64 = 0.050;
    const CSC_ADD_MIN_SAMPLE_SECONDS: f64 = 0.020;
    const TRANSPOSE_MIN_SAMPLE_SECONDS: f64 = 0.100;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_secs(1);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const HARNESS_SOURCE: &[u8] = include_bytes!("perf_sparse.rs");
    const FORMATS_SOURCE: &[u8] = include_bytes!("../formats.rs");
    const LINALG_SOURCE: &[u8] = include_bytes!("../linalg.rs");
    const ORACLE_SOURCE: &[u8] = include_bytes!("../../python/scipy_sparse_arm.py");

    fn diagonal_fixture(n: usize) -> CsrMatrix {
        let data = (0..n)
            .map(|index| ((index % 23) as f64 - 11.0) / 64.0 + 1.0 / 256.0)
            .collect::<Vec<_>>();
        let indices = (0..n).collect::<Vec<_>>();
        let indptr = (0..=n).collect::<Vec<_>>();
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical diagonal CSR")
    }

    fn path_fixture(n: usize) -> CsrMatrix {
        let mut data = Vec::with_capacity(2 * n.saturating_sub(1));
        let mut indices = Vec::with_capacity(data.capacity());
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..n {
            if row > 0 {
                indices.push(row - 1);
                data.push(1.0 + ((row - 1) % 17) as f64 / 32.0);
            }
            if row + 1 < n {
                indices.push(row + 1);
                data.push(1.0 + (row % 17) as f64 / 32.0);
            }
            indptr.push(data.len());
        }
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical undirected path CSR")
    }

    fn cycle_fixture(n: usize) -> CsrMatrix {
        let mut rows = vec![Vec::<(usize, f64)>::with_capacity(2); n];
        for index in 0..n {
            let neighbor = (index + 1) % n;
            let weight = 1.0 + (index % 29) as f64 / 64.0;
            rows[index].push((neighbor, weight));
            rows[neighbor].push((index, weight));
        }
        let mut data = Vec::with_capacity(2 * n);
        let mut indices = Vec::with_capacity(2 * n);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in &mut rows {
            row.sort_unstable_by_key(|entry| entry.0);
            for &(column, value) in row.iter() {
                indices.push(column);
                data.push(value);
            }
            indptr.push(data.len());
        }
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical undirected cycle CSR")
    }

    fn torus_fixture(side: usize) -> CsrMatrix {
        let n = side * side;
        let mut data = Vec::with_capacity(4 * n);
        let mut indices = Vec::with_capacity(4 * n);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..side {
            for column in 0..side {
                let mut neighbors = [
                    ((row + side - 1) % side) * side + column,
                    ((row + 1) % side) * side + column,
                    row * side + (column + side - 1) % side,
                    row * side + (column + 1) % side,
                ];
                neighbors.sort_unstable();
                indices.extend_from_slice(&neighbors);
                data.extend_from_slice(&[1.0; 4]);
                indptr.push(data.len());
            }
        }
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical periodic two-dimensional grid CSR")
    }

    fn csc_add_operand(n: usize, side: usize) -> CscMatrix {
        let nnz = n * CSC_ADD_ENTRIES_PER_COLUMN;
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for column in 0..n {
            let mut entries = (0..CSC_ADD_ENTRIES_PER_COLUMN)
                .map(|slot| {
                    let row = (173 * slot + 17 * column + 89 * side) % n;
                    let numerator = ((column + 3 * slot + 11 * side) % 37) as i64 - 18;
                    (row, numerator as f64 / 32.0)
                })
                .collect::<Vec<_>>();
            entries.sort_unstable_by_key(|entry| entry.0);
            for (row, value) in entries {
                indices.push(row);
                data.push(value);
            }
            indptr.push(data.len());
        }
        CscMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical CSC-add operand")
    }

    fn csc_add_fixture(n: usize) -> (CscMatrix, CscMatrix) {
        (csc_add_operand(n, 0), csc_add_operand(n, 1))
    }

    fn transpose_fixture() -> CsrMatrix {
        let mut data = Vec::with_capacity(TRANSPOSE_NNZ);
        let mut indices = Vec::with_capacity(TRANSPOSE_NNZ);
        let mut indptr = Vec::with_capacity(TRANSPOSE_ROWS + 1);
        indptr.push(0);
        for row in 0..TRANSPOSE_ROWS {
            let base = (17 * row) % (TRANSPOSE_COLS - TRANSPOSE_ENTRIES_PER_ROW + 1);
            for slot in 0..TRANSPOSE_ENTRIES_PER_ROW {
                let column = base + slot;
                indices.push(column);
                data.push(1.0 + ((row + column) % 17) as f64 / 64.0);
            }
            indptr.push(data.len());
        }
        CsrMatrix::from_components(
            Shape2D::new(TRANSPOSE_ROWS, TRANSPOSE_COLS),
            data,
            indices,
            indptr,
            false,
        )
        .expect("canonical rectangular transpose fixture")
    }

    fn compressed_parts_sha256(
        shape: Shape2D,
        data: &[f64],
        indices: &[usize],
        indptr: &[usize],
    ) -> Result<String, String> {
        let mut digest = Sha256::new();
        for (label, value) in [
            ("row count", shape.rows),
            ("column count", shape.cols),
            ("nonzero count", data.len()),
        ] {
            let value = u64::try_from(value)
                .map_err(|error| format!("{label} does not fit u64: {error}"))?;
            digest.update(value.to_le_bytes());
        }
        for &value in data {
            digest.update(value.to_le_bytes());
        }
        for (label, values) in [("compressed index", indices), ("pointer", indptr)] {
            for &value in values {
                let value = u64::try_from(value)
                    .map_err(|error| format!("{label} does not fit u64: {error}"))?;
                digest.update(value.to_le_bytes());
            }
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn canonical_input_sha256(matrix: &CsrMatrix) -> Result<String, String> {
        let mut digest = Sha256::new();
        for (label, value) in [
            ("row count", matrix.shape().rows),
            ("nonzero count", matrix.nnz()),
        ] {
            let value = u64::try_from(value)
                .map_err(|error| format!("{label} does not fit u64: {error}"))?;
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
                    .map_err(|error| format!("{label} does not fit u64: {error}"))?;
                digest.update(value.to_le_bytes());
            }
        }
        Ok(format!("{:x}", digest.finalize()))
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

    fn sample_host_wide_quiescence(phase: &str) -> Result<bool, String> {
        let before = read_cpu_ticks()?;
        thread::sleep(HOST_QUIESCENCE_SAMPLE);
        let after = read_cpu_ticks()?;
        let mut busy_cpus = Vec::new();
        let mut maximum_busy_fraction = 0.0_f64;
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
        let clear = busy_cpus.is_empty();
        println!(
            "host_wide_quiescence_{phase}={} evidence_class=PROVISIONAL_NON_EXCLUSIVE \
             sampled_cpus={} maximum_busy_fraction={maximum_busy_fraction:.3} \
             busy_cpu_count_above_limit={} limit={HOST_QUIESCENCE_MAX_BUSY:.3} \
             busy_cpus={}",
            if clear { "clear" } else { "NOT_CERTIFIED" },
            before.len(),
            busy_cpus.len(),
            if clear {
                "none".to_string()
            } else {
                busy_cpus.join(",")
            }
        );
        Ok(clear)
    }

    fn affinity_list() -> Result<String, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?;
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_string())
    }

    fn affinity_cpu_count(list: &str) -> Result<usize, String> {
        let mut count = 0usize;
        for segment in list.split(',') {
            if let Some((start, end)) = segment.split_once('-') {
                let start = start
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity start {start}: {error}"))?;
                let end = end
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity end {end}: {error}"))?;
                count = count
                    .checked_add(end.saturating_sub(start).saturating_add(1))
                    .ok_or_else(|| "affinity CPU count overflowed".to_string())?;
            } else {
                segment
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity CPU {segment}: {error}"))?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| "affinity CPU count overflowed".to_string())?;
            }
        }
        Ok(count)
    }

    fn print_hardware_provenance() -> Result<(), String> {
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map_err(|error| format!("read hostname: {error}"))?;
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let logical_threads = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        let mut physical_cores = BTreeSet::new();
        for block in cpuinfo.split("\n\n") {
            let physical = block.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(name, _)| name.trim() == "physical id")
                    .map(|(_, value)| value.trim().to_string())
            });
            let core = block.lines().find_map(|line| {
                line.split_once(':')
                    .filter(|(name, _)| name.trim() == "core id")
                    .map(|(_, value)| value.trim().to_string())
            });
            if let (Some(physical), Some(core)) = (physical, core) {
                physical_cores.insert((physical, core));
            }
        }
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let ram_kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|value| value.split_whitespace().next())
            .ok_or_else(|| "MemTotal missing from /proc/meminfo".to_string())?
            .parse::<u64>()
            .map_err(|error| format!("parse MemTotal: {error}"))?;
        let numa_count = std::fs::read_dir("/sys/devices/system/node")
            .map_err(|error| format!("read NUMA topology: {error}"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.strip_prefix("node")
                    .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .count();
        let affinity = affinity_list()?;
        if affinity_cpu_count(&affinity)? != 1 {
            return Err(format!(
                "candidate completion must be pinned to one CPU, got affinity={affinity}"
            ));
        }
        let governor_path =
            format!("/sys/devices/system/cpu/cpu{affinity}/cpufreq/scaling_governor");
        let governor = std::fs::read_to_string(&governor_path)
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let mut isa = Vec::new();
        if std::is_x86_feature_detected!("avx2") {
            isa.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            isa.push("fma");
        }
        println!(
            "host_identity={} physical_cores={} logical_threads={logical_threads} \
             ram_bytes={} numa_count={numa_count} requested_threads=1 \
             actual_observed_worker_threads=1/1/1 affinity={affinity} \
             runtime_isa={} scaling_governor={governor} \
             claim_message_id=0 release_message_id=0 coordination=agent-mail-unavailable",
            host.trim(),
            physical_cores.len(),
            ram_kib * 1_024,
            if isa.is_empty() {
                "baseline".to_string()
            } else {
                isa.join("+")
            }
        );
        Ok(())
    }

    fn csc_pair_input_sha256(lhs: &CscMatrix, rhs: &CscMatrix) -> Result<String, String> {
        let mut digest = Sha256::new();
        let n = u64::try_from(lhs.shape().rows)
            .map_err(|error| format!("CSC dimension does not fit u64: {error}"))?;
        digest.update(n.to_le_bytes());
        for (label, matrix) in [("left", lhs), ("right", rhs)] {
            let nnz = u64::try_from(matrix.nnz())
                .map_err(|error| format!("{label} CSC nnz does not fit u64: {error}"))?;
            digest.update(nnz.to_le_bytes());
            for &value in matrix.data() {
                digest.update(value.to_le_bytes());
            }
            for &index in matrix.indices() {
                let index = u64::try_from(index)
                    .map_err(|error| format!("{label} CSC index does not fit u64: {error}"))?;
                digest.update(index.to_le_bytes());
            }
            for &pointer in matrix.indptr() {
                let pointer = u64::try_from(pointer)
                    .map_err(|error| format!("{label} CSC pointer does not fit u64: {error}"))?;
                digest.update(pointer.to_le_bytes());
            }
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn oracle_path(explicit: Option<&String>) -> Result<PathBuf, String> {
        if let Some(path) = explicit {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!("SciPy oracle does not exist: {}", path.display()));
        }
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest.join("python/scipy_sparse_arm.py");
        if path.is_file() {
            return Ok(path);
        }
        Err("SciPy sparse oracle is unavailable".to_string())
    }

    fn field_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|field| field.strip_prefix(name))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    struct ScipyExpm {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    type SparseParity = (String, Vec<usize>, Vec<usize>, Vec<f64>);

    impl ScipyExpm {
        fn start(script: &Path) -> Result<(Self, String), String> {
            Self::start_mode(script, "--live-expm")
        }

        fn start_laplacian(script: &Path) -> Result<(Self, String), String> {
            Self::start_mode(script, "--live-laplacian")
        }

        fn start_laplacian_cycle(script: &Path) -> Result<(Self, String), String> {
            Self::start_mode(script, "--live-laplacian-cycle")
        }

        fn start_csc_add(script: &Path) -> Result<(Self, String), String> {
            Self::start_mode(script, "--live-csc-add")
        }

        fn start_transpose(script: &Path) -> Result<(Self, String), String> {
            Self::start_mode(script, "--live-transpose")
        }

        fn start_mode(script: &Path, mode: &str) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg(mode)
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| format!("spawn live SciPy: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy arm has no stdin".to_string())?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| "live SciPy arm has no stdout".to_string())?;
            let mut result = Self {
                child,
                stdin,
                stdout: BufReader::new(stdout),
            };
            let identity = result.read_line()?;
            Ok((result, identity))
        }

        fn read_line(&mut self) -> Result<String, String> {
            let mut line = String::new();
            self.stdout
                .read_line(&mut line)
                .map_err(|error| format!("read live SciPy: {error}"))?;
            let line = line.trim().to_string();
            if line.is_empty() {
                return Err("live SciPy closed its protocol stream".to_string());
            }
            if line.starts_with("FATAL ") {
                return Err(format!("live SciPy rejected request: {line}"));
            }
            Ok(line)
        }

        fn write_usize_vector(&mut self, label: &str, values: &[usize]) -> Result<(), String> {
            let mut line = String::with_capacity(label.len() + 2 + values.len() * 8);
            write!(line, "{label} ").map_err(|error| format!("format {label}: {error}"))?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    line.push(',');
                }
                write!(line, "{value}")
                    .map_err(|error| format!("format {label} value: {error}"))?;
            }
            line.push('\n');
            self.stdin
                .write_all(line.as_bytes())
                .map_err(|error| format!("write {label}: {error}"))
        }

        fn write_f64_vector(&mut self, label: &str, values: &[f64]) -> Result<(), String> {
            let mut line = String::with_capacity(label.len() + 2 + values.len() * 24);
            write!(line, "{label} ").map_err(|error| format!("format {label}: {error}"))?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    line.push(',');
                }
                write!(line, "{value:.17e}")
                    .map_err(|error| format!("format {label} value: {error}"))?;
            }
            line.push('\n');
            self.stdin
                .write_all(line.as_bytes())
                .map_err(|error| format!("write {label}: {error}"))
        }

        fn initialize(&mut self, matrix: &CsrMatrix) -> Result<String, String> {
            writeln!(self.stdin, "INIT {} {}", matrix.shape().rows, matrix.nnz())
                .map_err(|error| format!("write INIT: {error}"))?;
            self.write_usize_vector("INDPTR", matrix.indptr())?;
            self.write_usize_vector("INDICES", matrix.indices())?;
            self.write_f64_vector("DATA", matrix.data())?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INIT: {error}"))?;
            self.read_line()
        }

        fn initialize_csc_pair(
            &mut self,
            lhs: &CscMatrix,
            rhs: &CscMatrix,
        ) -> Result<String, String> {
            writeln!(
                self.stdin,
                "INIT_CSC_ADD {} {} {}",
                lhs.shape().rows,
                lhs.nnz(),
                rhs.nnz()
            )
            .map_err(|error| format!("write INIT_CSC_ADD: {error}"))?;
            self.write_usize_vector("LHS_INDPTR", lhs.indptr())?;
            self.write_usize_vector("LHS_INDICES", lhs.indices())?;
            self.write_f64_vector("LHS_DATA", lhs.data())?;
            self.write_usize_vector("RHS_INDPTR", rhs.indptr())?;
            self.write_usize_vector("RHS_INDICES", rhs.indices())?;
            self.write_f64_vector("RHS_DATA", rhs.data())?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INIT_CSC_ADD: {error}"))?;
            self.read_line()
        }

        fn initialize_transpose(&mut self, matrix: &CsrMatrix) -> Result<String, String> {
            writeln!(
                self.stdin,
                "INIT_TRANSPOSE {} {} {}",
                matrix.shape().rows,
                matrix.shape().cols,
                matrix.nnz()
            )
            .map_err(|error| format!("write INIT_TRANSPOSE: {error}"))?;
            self.write_usize_vector("INDPTR", matrix.indptr())?;
            self.write_usize_vector("INDICES", matrix.indices())?;
            self.write_f64_vector("DATA", matrix.data())?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INIT_TRANSPOSE: {error}"))?;
            self.read_line()
        }

        fn input_sha256(&mut self) -> Result<String, String> {
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INPUT_SHA256: {error}"))?;
            let reply = self.read_line()?;
            let digest = reply
                .strip_prefix("INPUT_SHA256 ")
                .ok_or_else(|| format!("malformed INPUT_SHA256 reply: {reply}"))?;
            if !is_sha256(digest) {
                return Err(format!("invalid live input SHA-256: {digest}"));
            }
            Ok(digest.to_string())
        }

        fn parity(&mut self, n: usize) -> Result<(String, Vec<f64>), String> {
            writeln!(self.stdin, "PARITY").map_err(|error| format!("write PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush PARITY: {error}"))?;
            let result = self.read_line()?;
            let diagonal_line = self.read_line()?;
            let payload = diagonal_line
                .strip_prefix("DIAG ")
                .ok_or_else(|| format!("malformed live diagonal: {diagonal_line}"))?;
            let diagonal = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse live diagonal: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if diagonal.len() != n {
                return Err(format!("live diagonal length {} != {n}", diagonal.len()));
            }
            Ok((result, diagonal))
        }

        fn read_usize_reply(&mut self, label: &str) -> Result<Vec<usize>, String> {
            let line = self.read_line()?;
            let payload = line
                .strip_prefix(label)
                .and_then(|rest| rest.strip_prefix(' '))
                .ok_or_else(|| format!("malformed {label} reply: {line}"))?;
            payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|error| format!("parse {label}: {error}"))
                })
                .collect()
        }

        fn read_f64_reply(&mut self, label: &str) -> Result<Vec<f64>, String> {
            let line = self.read_line()?;
            let payload = line
                .strip_prefix(label)
                .and_then(|rest| rest.strip_prefix(' '))
                .ok_or_else(|| format!("malformed {label} reply: {line}"))?;
            payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse {label}: {error}"))
                })
                .collect()
        }

        fn laplacian_parity(&mut self) -> Result<SparseParity, String> {
            writeln!(self.stdin, "PARITY").map_err(|error| format!("write PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush PARITY: {error}"))?;
            let result = self.read_line()?;
            let indptr = self.read_usize_reply("OUT_INDPTR")?;
            let indices = self.read_usize_reply("OUT_INDICES")?;
            let data = self.read_f64_reply("OUT_DATA")?;
            Ok((result, indptr, indices, data))
        }

        fn csc_add_parity(&mut self) -> Result<SparseParity, String> {
            self.laplacian_parity()
        }

        fn transpose_parity(&mut self) -> Result<(String, String), String> {
            writeln!(self.stdin, "PARITY").map_err(|error| format!("write PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush PARITY: {error}"))?;
            let result = self.read_line()?;
            let digest_line = self.read_line()?;
            let digest = digest_line
                .strip_prefix("OUTPUT_SHA256 ")
                .ok_or_else(|| format!("malformed transpose output digest: {digest_line}"))?;
            if !is_sha256(digest) {
                return Err(format!("invalid transpose output SHA-256: {digest}"));
            }
            Ok((result, digest.to_string()))
        }

        fn solve(&mut self, repetitions: usize, expected_nnz: usize) -> Result<f64, String> {
            writeln!(self.stdin, "SOLVE {repetitions}")
                .map_err(|error| format!("write SOLVE: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SOLVE: {error}"))?;
            let reply = self.read_line()?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 5 || fields[0] != "TIME" {
                return Err(format!("malformed live timing: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse live elapsed: {error}"))?;
            let nnz = fields[2]
                .parse::<usize>()
                .map_err(|error| format!("parse live result nnz: {error}"))?;
            let threads = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse live worker count: {error}"))?;
            let checksum = fields[4]
                .parse::<f64>()
                .map_err(|error| format!("parse live checksum: {error}"))?;
            black_box(checksum);
            if !elapsed.is_finite() || elapsed <= 0.0 || nnz != expected_nnz || threads != 1 {
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

    fn time_current(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let started = Instant::now();
        for _ in 0..repetitions {
            let result =
                expm(black_box(matrix), ExpmOptions::default()).expect("FrankenSciPy sparse expm");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn time_current_laplacian(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = laplacian(black_box(matrix), true)
                .expect("FrankenSciPy normalized sparse laplacian");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn time_current_cycle_laplacian(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        LAPLACIAN_FORCE_DENSE_REFERENCE.store(false, Ordering::SeqCst);
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = laplacian(black_box(matrix), false)
                .expect("FrankenSciPy unnormalized sparse laplacian");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    struct DenseReferenceGuard;

    impl DenseReferenceGuard {
        fn activate() -> Self {
            LAPLACIAN_FORCE_DENSE_REFERENCE.store(true, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for DenseReferenceGuard {
        fn drop(&mut self) {
            LAPLACIAN_FORCE_DENSE_REFERENCE.store(false, Ordering::SeqCst);
        }
    }

    fn time_torus_candidate(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        LAPLACIAN_FORCE_DENSE_REFERENCE.store(false, Ordering::SeqCst);
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = laplacian(black_box(matrix), true)
                .expect("FrankenSciPy direct normalized torus Laplacian");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn time_torus_dense_reference(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let _guard = DenseReferenceGuard::activate();
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = laplacian(black_box(matrix), true)
                .expect("FrankenSciPy dense-reference normalized torus Laplacian");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn four_call_geometric_null<F>(mut sample: F) -> Result<f64, String>
    where
        F: FnMut() -> Result<f64, String>,
    {
        let left_first = sample()?;
        let right_first = sample()?;
        let right_second = sample()?;
        let left_second = sample()?;
        Ok(((left_first / right_first) * (left_second / right_second)).sqrt())
    }

    fn time_current_csc_add(lhs: &CscMatrix, rhs: &CscMatrix, repetitions: usize) -> f64 {
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = add_csc(black_box(lhs), black_box(rhs))
                .expect("FrankenSciPy canonical CSC addition");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn time_transpose_view(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let mut checksum = 0usize;
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = sparse_transpose_view(black_box(matrix));
            checksum ^= result
                .nnz()
                .wrapping_add(result.shape().rows)
                .wrapping_add(result.shape().cols);
            black_box(result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(checksum);
        elapsed
    }

    fn time_materialized_transpose(matrix: &CsrMatrix, repetitions: usize) -> f64 {
        let mut checksum = 0usize;
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = sparse_transpose(black_box(matrix));
            checksum ^= result
                .nnz()
                .wrapping_add(result.shape().rows)
                .wrapping_add(result.shape().cols);
            black_box(result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(checksum);
        elapsed
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required environment variable {name} is absent"))
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            let middle = values.len() / 2;
            0.5 * (values[middle - 1] + values[middle])
        } else {
            values[values.len() / 2]
        }
    }

    fn percentile(mut values: Vec<f64>, numerator: usize, denominator: usize) -> f64 {
        values.sort_by(f64::total_cmp);
        let scaled = numerator.saturating_mul(values.len().saturating_sub(1));
        values[scaled.div_ceil(denominator)]
    }

    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
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

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().saturating_sub(1).max(1) as f64;
        variance.sqrt() / mean
    }

    fn validate_current(matrix: &CsrMatrix, live_diagonal: &[f64]) -> Result<(f64, f64), String> {
        let current = expm(matrix, ExpmOptions::default())
            .map_err(|error| format!("FrankenSciPy parity expm: {error}"))?;
        let n = matrix.shape().rows;
        if current.len() != n || current.iter().any(|row| row.len() != n) {
            return Err("FrankenSciPy returned the wrong dense shape".to_string());
        }
        let mut offdiag_max = 0.0_f64;
        let mut difference_squared = 0.0_f64;
        let mut live_squared = 0.0_f64;
        for row in 0..n {
            for column in 0..n {
                let value = current[row][column];
                if !value.is_finite() {
                    return Err("FrankenSciPy returned a non-finite value".to_string());
                }
                if row != column {
                    offdiag_max = offdiag_max.max(value.abs());
                }
            }
            let expected = matrix.data()[row].exp();
            let tolerance = 4.0 * f64::EPSILON * expected.abs().max(1.0);
            if (current[row][row] - expected).abs() > tolerance
                || (live_diagonal[row] - expected).abs() > tolerance
            {
                return Err(format!("diagonal parity failed at index {row}"));
            }
            let difference = current[row][row] - live_diagonal[row];
            difference_squared += difference * difference;
            live_squared += live_diagonal[row] * live_diagonal[row];
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if offdiag_max > 1.0e-13 || relative_l2 > 1.0e-14 {
            return Err(format!(
                "parity thresholds failed: offdiag_max={offdiag_max:e} relative_l2={relative_l2:e}"
            ));
        }
        Ok((offdiag_max, relative_l2))
    }

    fn validate_current_laplacian(
        matrix: &CsrMatrix,
        live_indptr: &[usize],
        live_indices: &[usize],
        live_data: &[f64],
    ) -> Result<(f64, f64), String> {
        let current = laplacian(matrix, true)
            .map_err(|error| format!("FrankenSciPy parity laplacian: {error}"))?;
        let n = matrix.shape().rows;
        let expected_nnz = 3 * n - 2;
        if current.shape() != Shape2D::new(n, n)
            || current.nnz() != expected_nnz
            || live_indptr.len() != n + 1
            || live_indices.len() != expected_nnz
            || live_data.len() != expected_nnz
            || live_indptr[n] != expected_nnz
        {
            return Err("live SciPy returned malformed sparse Laplacian arrays".to_string());
        }
        if current.indptr() != live_indptr || current.indices() != live_indices {
            return Err("current/live sparse Laplacian structures differ".to_string());
        }
        let mut degrees = vec![0.0_f64; n];
        for row in 0..n {
            degrees[row] = matrix.data()[matrix.indptr()[row]..matrix.indptr()[row + 1]]
                .iter()
                .map(|value| value.abs())
                .sum();
        }
        let mut difference_squared = 0.0_f64;
        let mut live_squared = 0.0_f64;
        for row in 0..n {
            let expected_columns = if row == 0 {
                vec![0, 1]
            } else if row + 1 == n {
                vec![n - 2, n - 1]
            } else {
                vec![row - 1, row, row + 1]
            };
            let start = live_indptr[row];
            let end = live_indptr[row + 1];
            if live_indices[start..end] != expected_columns {
                return Err(format!("live CSR structure mismatch at row {row}"));
            }
            for offset in start..end {
                let column = live_indices[offset];
                let expected = if column == row {
                    1.0
                } else {
                    let edge = column.min(row);
                    let weight = 1.0 + (edge % 17) as f64 / 32.0;
                    -weight / (degrees[row] * degrees[column]).sqrt()
                };
                let current_value = current.data()[offset];
                if !current_value.is_finite() {
                    return Err("FrankenSciPy returned a non-finite Laplacian".to_string());
                }
                let live_value = live_data[offset];
                let tolerance = 8.0 * f64::EPSILON * expected.abs().max(1.0);
                if (current_value - expected).abs() > tolerance
                    || (live_value - expected).abs() > tolerance
                {
                    return Err(format!(
                        "Laplacian value mismatch at ({row},{column}): \
                         current={current_value:e} live={live_value:e} expected={expected:e}"
                    ));
                }
                let difference = current_value - live_value;
                difference_squared += difference * difference;
                live_squared += live_value * live_value;
            }
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if relative_l2 > 1.0e-14 {
            return Err(format!(
                "Laplacian parity threshold failed: relative_l2={relative_l2:e}"
            ));
        }
        Ok((0.0, relative_l2))
    }

    fn validate_current_cycle_laplacian(
        matrix: &CsrMatrix,
        live_indptr: &[usize],
        live_indices: &[usize],
        live_data: &[f64],
    ) -> Result<(f64, f64), String> {
        let current = laplacian(matrix, false)
            .map_err(|error| format!("FrankenSciPy parity cycle laplacian: {error}"))?;
        let n = matrix.shape().rows;
        let expected_nnz = 3 * n;
        if current.shape() != Shape2D::new(n, n)
            || current.nnz() != expected_nnz
            || live_indptr.len() != n + 1
            || live_indices.len() != expected_nnz
            || live_data.len() != expected_nnz
            || live_indptr[n] != expected_nnz
        {
            return Err("live SciPy returned malformed cycle Laplacian arrays".to_string());
        }
        if current.indptr() != live_indptr || current.indices() != live_indices {
            return Err("current/live cycle Laplacian structures differ".to_string());
        }
        let mut degrees = vec![0.0_f64; n];
        for row in 0..n {
            degrees[row] = matrix.data()[matrix.indptr()[row]..matrix.indptr()[row + 1]]
                .iter()
                .map(|value| value.abs())
                .sum();
        }
        let mut difference_squared = 0.0_f64;
        let mut live_squared = 0.0_f64;
        for row in 0..n {
            let mut expected_columns = vec![(row + n - 1) % n, row, (row + 1) % n];
            expected_columns.sort_unstable();
            let start = live_indptr[row];
            let end = live_indptr[row + 1];
            if live_indices[start..end] != expected_columns {
                return Err(format!("live cycle CSR structure mismatch at row {row}"));
            }
            for offset in start..end {
                let column = live_indices[offset];
                let expected = if column == row {
                    degrees[row]
                } else if column == (row + 1) % n {
                    -(1.0 + (row % 29) as f64 / 64.0)
                } else {
                    -(1.0 + (column % 29) as f64 / 64.0)
                };
                let current_value = current.data()[offset];
                if !current_value.is_finite() {
                    return Err("FrankenSciPy returned a non-finite cycle Laplacian".to_string());
                }
                let live_value = live_data[offset];
                let tolerance = 4.0 * f64::EPSILON * expected.abs().max(1.0);
                if (current_value - expected).abs() > tolerance
                    || (live_value - expected).abs() > tolerance
                {
                    return Err(format!(
                        "cycle Laplacian mismatch at ({row},{column}): \
                         current={current_value:e} live={live_value:e} expected={expected:e}"
                    ));
                }
                let difference = current_value - live_value;
                difference_squared += difference * difference;
                live_squared += live_value * live_value;
            }
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if relative_l2 > 1.0e-15 {
            return Err(format!(
                "cycle parity threshold failed: relative_l2={relative_l2:e}"
            ));
        }
        Ok((0.0, relative_l2))
    }

    fn validate_torus_candidate(
        matrix: &CsrMatrix,
        live_indptr: &[usize],
        live_indices: &[usize],
        live_data: &[f64],
    ) -> Result<(f64, f64, usize), String> {
        LAPLACIAN_FORCE_DENSE_REFERENCE.store(false, Ordering::SeqCst);
        let candidate = laplacian(matrix, true)
            .map_err(|error| format!("FrankenSciPy candidate torus Laplacian: {error}"))?;
        let dense_reference = {
            let _guard = DenseReferenceGuard::activate();
            laplacian(matrix, true)
                .map_err(|error| format!("FrankenSciPy dense-reference torus: {error}"))?
        };
        let candidate_meta = candidate.canonical_meta();
        let dense_meta = dense_reference.canonical_meta();
        if candidate.shape() != Shape2D::new(TORUS_N, TORUS_N)
            || candidate.nnz() != TORUS_RESULT_NNZ
            || !candidate_meta.sorted_indices
            || !candidate_meta.deduplicated
            || !dense_meta.sorted_indices
            || !dense_meta.deduplicated
        {
            return Err("candidate/dense returned the wrong canonical torus contract".to_string());
        }
        if live_indptr.len() != TORUS_N + 1
            || live_indices.len() != TORUS_RESULT_NNZ
            || live_data.len() != TORUS_RESULT_NNZ
            || live_indptr[TORUS_N] != TORUS_RESULT_NNZ
        {
            return Err("live SciPy returned malformed normalized torus arrays".to_string());
        }
        if candidate.indptr() != live_indptr || candidate.indices() != live_indices {
            return Err("candidate/live normalized torus structures differ".to_string());
        }
        if candidate.indptr() != dense_reference.indptr()
            || candidate.indices() != dense_reference.indices()
            || candidate_meta != dense_meta
        {
            return Err("candidate/dense-reference normalized torus structures differ".to_string());
        }
        let dense_bit_mismatches = candidate
            .data()
            .iter()
            .zip(dense_reference.data())
            .filter(|(candidate, reference)| candidate.to_bits() != reference.to_bits())
            .count();
        if dense_bit_mismatches != 0 {
            return Err(format!(
                "candidate/dense-reference torus values have {dense_bit_mismatches} bit mismatches"
            ));
        }

        let mut maximum_absolute_difference = 0.0_f64;
        let mut difference_squared = 0.0_f64;
        let mut live_squared = 0.0_f64;
        for row in 0..TORUS_N {
            let start = candidate.indptr()[row];
            let end = candidate.indptr()[row + 1];
            if end - start != 5 {
                return Err(format!(
                    "normalized torus row {row} has {} entries",
                    end - start
                ));
            }
            for offset in start..end {
                let column = candidate.indices()[offset];
                let expected = if column == row { 1.0 } else { -0.25 };
                let candidate_value = candidate.data()[offset];
                let live_value = live_data[offset];
                if !candidate_value.is_finite() || !live_value.is_finite() {
                    return Err(
                        "candidate/live normalized torus returned non-finite data".to_string()
                    );
                }
                let tolerance = 4.0 * f64::EPSILON * live_value.abs().max(1.0);
                if (candidate_value - expected).abs() > tolerance
                    || (live_value - expected).abs() > tolerance
                    || (candidate_value - live_value).abs() > tolerance
                {
                    return Err(format!(
                        "normalized torus mismatch at ({row},{column}): candidate={candidate_value:e} \
                         live={live_value:e} expected={expected:e} tolerance={tolerance:e}"
                    ));
                }
                let difference = candidate_value - live_value;
                maximum_absolute_difference = maximum_absolute_difference.max(difference.abs());
                difference_squared += difference * difference;
                live_squared += live_value * live_value;
            }
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if relative_l2 > 1.0e-15 {
            return Err(format!(
                "normalized torus relative L2 {relative_l2:e} exceeds 1e-15"
            ));
        }
        Ok((
            maximum_absolute_difference,
            relative_l2,
            dense_bit_mismatches,
        ))
    }

    fn validate_current_csc_add(
        lhs: &CscMatrix,
        rhs: &CscMatrix,
        live_indptr: &[usize],
        live_indices: &[usize],
        live_data: &[f64],
    ) -> Result<(usize, f64, f64), String> {
        let current = add_csc(lhs, rhs)
            .map_err(|error| format!("FrankenSciPy parity CSC addition: {error}"))?;
        let meta = current.canonical_meta();
        if !meta.sorted_indices || !meta.deduplicated {
            return Err("FrankenSciPy CSC-add result is not canonical".to_string());
        }
        if current.indptr() != live_indptr || current.indices() != live_indices {
            return Err(format!(
                "CSC-add structure mismatch: current nnz={} live nnz={}",
                current.nnz(),
                live_data.len()
            ));
        }
        if current.data().len() != live_data.len() {
            return Err(format!(
                "CSC-add data length mismatch: current={} live={}",
                current.data().len(),
                live_data.len()
            ));
        }
        let mut max_abs_difference = 0.0_f64;
        let mut difference_squared = 0.0_f64;
        let mut live_squared = 0.0_f64;
        for (offset, (&current_value, &live_value)) in
            current.data().iter().zip(live_data).enumerate()
        {
            if !current_value.is_finite() || !live_value.is_finite() {
                return Err(format!("CSC-add non-finite output at offset {offset}"));
            }
            let difference = current_value - live_value;
            let tolerance = 4.0 * f64::EPSILON * live_value.abs().max(1.0);
            if difference.abs() > tolerance {
                return Err(format!(
                    "CSC-add value mismatch at offset {offset}: current={current_value:e} \
                     live={live_value:e} tolerance={tolerance:e}"
                ));
            }
            max_abs_difference = max_abs_difference.max(difference.abs());
            difference_squared += difference * difference;
            live_squared += live_value * live_value;
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if relative_l2 > 1.0e-15 {
            return Err(format!("CSC-add relative L2 {relative_l2:e} exceeds 1e-15"));
        }
        Ok((current.nnz(), max_abs_difference, relative_l2))
    }

    fn sha256_of_self() -> Result<String, String> {
        let executable =
            std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
        let bytes = std::fs::read(executable).map_err(|error| format!("read own ELF: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn run_current_profile(n: usize, repetitions: usize) -> Result<(), String> {
        if n < 1 || repetitions < 1 {
            return Err("require n>=1 and repetitions>=1".to_string());
        }
        let matrix = diagonal_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elapsed = time_current(&matrix, repetitions);
        println!(
            "EXPM_FSCI_PROFILE n={n} nnz={} repetitions={repetitions} \
             elapsed_seconds={elapsed:.9} result_format=dense \
             result_elements={} actual_observed_worker_threads=1 \
             input_sha256={input_sha256}",
            matrix.nnz(),
            n * n
        );
        Ok(())
    }

    pub fn run_vs_scipy(
        n: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if n != REGISTERED_N || rounds != REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot registration requires n={REGISTERED_N} rounds={REGISTERED_ROUNDS}"
            ));
        }
        let matrix = diagonal_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=exact-diagonal-csr n={n} nnz={} data=((i%23)-11)/64+1/256 \
             rounds={rounds} construction_outside_timing=true \
             serialization_outside_timing=true requested_threads=1 \
             actual_observed_frankenscipy_worker_threads=1",
            matrix.nnz()
        );
        let script = oracle_path(explicit_oracle)?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = ScipyExpm::start(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("method=expm")
            || !identity.contains("solver_mod=scipy.sparse.linalg._matfuncs")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine-incumbent identity gate".to_string());
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        let case = scipy.initialize(&matrix)?;
        let expected_case =
            format!("CASE method=expm n={n} nnz={n} sorted=True canonical=True finite=True");
        if case != expected_case {
            return Err(format!("live SciPy constructed the wrong fixture: {case}"));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let (live_result, live_diagonal) = scipy.parity(n)?;
        let expected_result =
            format!("RESULT rows={n} cols={n} nnz={n} sorted=True canonical=True offdiag_max=0.0");
        if live_result != expected_result {
            return Err(format!("live SciPy result contract failed: {live_result}"));
        }
        let (offdiag_max, relative_l2) = validate_current(&matrix, &live_diagonal)?;
        println!(
            "agreement: diagonal_components={n}/{n} current_offdiag_max={offdiag_max:.3e} \
             live_result_format=csr live_result_nnz={n} diagonal_relative_l2={relative_l2:.3e} \
             tolerance=4*EPSILON*max(1,abs(expected)) pass=true"
        );

        let mut repetitions = 1usize;
        loop {
            let current = time_current(&matrix, repetitions);
            let live = scipy.solve(repetitions, n)?;
            if current >= MIN_SAMPLE_SECONDS && live >= MIN_SAMPLE_SECONDS {
                break;
            }
            repetitions = repetitions
                .checked_mul(2)
                .ok_or_else(|| "calibration repetition count overflowed".to_string())?;
        }
        println!(
            "calibration: repetitions={repetitions} min_sample_ms={} whole_public_calls=true",
            MIN_SAMPLE_SECONDS * 1_000.0
        );

        let mut current_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut current_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (current, live) = if round.is_multiple_of(2) {
                (
                    time_current(&matrix, repetitions),
                    scipy.solve(repetitions, n)?,
                )
            } else {
                let live = scipy.solve(repetitions, n)?;
                let current = time_current(&matrix, repetitions);
                (current, live)
            };
            let (current_left, current_right) = if round.is_multiple_of(2) {
                (
                    time_current(&matrix, repetitions),
                    time_current(&matrix, repetitions),
                )
            } else {
                let right = time_current(&matrix, repetitions);
                let left = time_current(&matrix, repetitions);
                (left, right)
            };
            let (live_left, live_right) = if round.is_multiple_of(2) {
                (scipy.solve(repetitions, n)?, scipy.solve(repetitions, n)?)
            } else {
                let right = scipy.solve(repetitions, n)?;
                let left = scipy.solve(repetitions, n)?;
                (left, right)
            };
            current_times.push(current / repetitions as f64);
            live_times.push(live / repetitions as f64);
            ratios.push(live / current);
            current_nulls.push(current_left / current_right);
            live_nulls.push(live_left / live_right);
        }
        scipy.quit();

        let current_p50 = median(current_times.clone());
        let live_p50 = median(live_times.clone());
        let ratio_median = median(ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios);
        let current_null_median = median(current_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (current_null_low, current_null_high) = bootstrap_median_ci(&current_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = current_null_high
            .max(live_null_high)
            .max(1.0 / current_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((current_null_high - current_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let effect_deviation = if ratio_high < 1.0 {
            1.0 - ratio_high
        } else if ratio_low > 1.0 {
            ratio_low - 1.0
        } else {
            0.0
        };
        let null_medians_ok =
            (current_null_median - 1.0).abs() <= 0.02 && (live_null_median - 1.0).abs() <= 0.02;
        let decidable = (ratio_high < 1.0 || ratio_low > 1.0)
            && effect_deviation > 2.0 * null_half_width
            && effect_deviation > 2.0 * (null_edge - 1.0)
            && null_medians_ok;
        println!(
            "timing: current_p50_ms={:.6} live_scipy_p50_ms={:.6} \
             incumbent_ratio_scipy_over_frankenscipy={ratio_median:.6} \
             bootstrap_median_ci95=[{ratio_low:.6},{ratio_high:.6}] \
             current_cv={:.6} live_cv={:.6} ratio_cv={:.6}",
            current_p50 * 1_000.0,
            live_p50 * 1_000.0,
            cv(&current_times),
            cv(&live_times),
            cv(&ratios)
        );
        println!(
            "nulls: current_median={current_null_median:.6} \
             current_ci95=[{current_null_low:.6},{current_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "corrected_null_gate: decidable={decidable} \
             effect_deviation={effect_deviation:.6} \
             required_half_width_margin={:.6} required_endpoint_margin={:.6}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!("raw_current_seconds={current_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_scipy_over_frankenscipy={ratios:?}");
        println!("raw_current_null={current_nulls:?}");
        println!("raw_live_null={live_nulls:?}");
        println!(
            "verdict={} (non-exclusive host; NOT DECIDED-class evidence)",
            if decidable && ratio_high < 0.90 {
                "PROVISIONAL FRANKENSCIPY LOSS; PROFILE ADMITTED"
            } else if decidable && ratio_low > 1.0 {
                "PROVISIONAL FRANKENSCIPY WIN; NO PROFILE"
            } else {
                "PROVISIONAL INDETERMINATE; NO PROFILE"
            }
        );
        Ok(())
    }

    pub fn run_laplacian_current_profile(n: usize, repetitions: usize) -> Result<(), String> {
        if n < 2 || repetitions < 1 {
            return Err("require n>=2 and repetitions>=1".to_string());
        }
        let matrix = path_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elapsed = time_current_laplacian(&matrix, repetitions);
        println!(
            "LAPLACIAN_FSCI_PROFILE n={n} input_nnz={} repetitions={repetitions} \
             elapsed_seconds={elapsed:.9} result_format=dense \
             result_elements={} actual_observed_worker_threads=1 \
             input_sha256={input_sha256}",
            matrix.nnz(),
            n * n
        );
        Ok(())
    }

    pub fn run_laplacian_vs_scipy(
        n: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if n != LAPLACIAN_REGISTERED_N || rounds != REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot registration requires n={LAPLACIAN_REGISTERED_N} \
                 rounds={REGISTERED_ROUNDS}"
            ));
        }
        let matrix = path_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=weighted-undirected-path n={n} input_nnz={} \
             edge_weight=1+(i%17)/32 normed=true rounds={rounds} \
             construction_outside_timing=true serialization_outside_timing=true \
             requested_threads=1 actual_observed_frankenscipy_worker_threads=1",
            matrix.nnz()
        );
        let script = oracle_path(explicit_oracle)?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = ScipyExpm::start_laplacian(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("method=laplacian")
            || !identity.contains("solver_mod=scipy.sparse.csgraph._laplacian")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy Laplacian failed genuine-incumbent identity gate".to_string());
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        let case = scipy.initialize(&matrix)?;
        let expected_case = format!(
            "CASE method=laplacian n={n} nnz={} sorted=True canonical=True \
             finite=True normed=True form=array",
            matrix.nnz()
        );
        if case != expected_case {
            return Err(format!("live SciPy constructed the wrong fixture: {case}"));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let (live_result, live_indptr, live_indices, live_data) = scipy.laplacian_parity()?;
        let expected_result = format!(
            "RESULT rows={n} cols={n} nnz={LAPLACIAN_RESULT_NNZ} \
             sorted=True canonical=True"
        );
        if live_result != expected_result {
            return Err(format!("live SciPy result contract failed: {live_result}"));
        }
        let (outside_pattern_max, relative_l2) =
            validate_current_laplacian(&matrix, &live_indptr, &live_indices, &live_data)?;
        println!(
            "agreement: structural_components={LAPLACIAN_RESULT_NNZ}/{} \
             current_outside_pattern_max={outside_pattern_max:.3e} \
             live_result_format=sparse live_result_nnz={LAPLACIAN_RESULT_NNZ} \
             structural_relative_l2={relative_l2:.3e} \
             tolerance=8*EPSILON*max(1,abs(expected)) pass=true",
            n * n
        );

        let mut current_repetitions = 1usize;
        while time_current_laplacian(&matrix, current_repetitions) < MIN_SAMPLE_SECONDS {
            current_repetitions = current_repetitions
                .checked_mul(2)
                .ok_or_else(|| "current calibration repetition count overflowed".to_string())?;
        }
        let mut live_repetitions = 1usize;
        while scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)? < MIN_SAMPLE_SECONDS {
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live calibration repetition count overflowed".to_string())?;
        }
        println!(
            "calibration: current_repetitions={current_repetitions} \
             live_repetitions={live_repetitions} min_sample_ms={} \
             whole_public_calls=true",
            MIN_SAMPLE_SECONDS * 1_000.0
        );

        let mut current_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut current_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (current_batch, live_batch) = if round.is_multiple_of(2) {
                (
                    time_current_laplacian(&matrix, current_repetitions),
                    scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?,
                )
            } else {
                let live = scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?;
                let current = time_current_laplacian(&matrix, current_repetitions);
                (current, live)
            };
            let (current_left, current_right) = if round.is_multiple_of(2) {
                (
                    time_current_laplacian(&matrix, current_repetitions),
                    time_current_laplacian(&matrix, current_repetitions),
                )
            } else {
                let right = time_current_laplacian(&matrix, current_repetitions);
                let left = time_current_laplacian(&matrix, current_repetitions);
                (left, right)
            };
            let (live_left, live_right) = if round.is_multiple_of(2) {
                (
                    scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?,
                    scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?,
                )
            } else {
                let right = scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?;
                let left = scipy.solve(live_repetitions, LAPLACIAN_RESULT_NNZ)?;
                (left, right)
            };
            let current = current_batch / current_repetitions as f64;
            let live = live_batch / live_repetitions as f64;
            current_times.push(current);
            live_times.push(live);
            ratios.push(live / current);
            current_nulls.push(current_left / current_right);
            live_nulls.push(live_left / live_right);
        }
        scipy.quit();

        let current_p50 = median(current_times.clone());
        let live_p50 = median(live_times.clone());
        let ratio_median = median(ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios);
        let current_null_median = median(current_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (current_null_low, current_null_high) = bootstrap_median_ci(&current_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = current_null_high
            .max(live_null_high)
            .max(1.0 / current_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((current_null_high - current_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let effect_deviation = if ratio_high < 1.0 {
            1.0 - ratio_high
        } else if ratio_low > 1.0 {
            ratio_low - 1.0
        } else {
            0.0
        };
        let null_medians_ok =
            (current_null_median - 1.0).abs() <= 0.02 && (live_null_median - 1.0).abs() <= 0.02;
        let decidable = (ratio_high < 1.0 || ratio_low > 1.0)
            && effect_deviation > 2.0 * null_half_width
            && effect_deviation > 2.0 * (null_edge - 1.0)
            && null_medians_ok;
        println!(
            "timing: current_p50_ms={:.6} live_scipy_p50_ms={:.6} \
             incumbent_ratio_scipy_over_frankenscipy={ratio_median:.6} \
             bootstrap_median_ci95=[{ratio_low:.6},{ratio_high:.6}] \
             current_cv={:.6} live_cv={:.6} ratio_cv={:.6}",
            current_p50 * 1_000.0,
            live_p50 * 1_000.0,
            cv(&current_times),
            cv(&live_times),
            cv(&ratios)
        );
        println!(
            "nulls: current_median={current_null_median:.6} \
             current_ci95=[{current_null_low:.6},{current_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "corrected_null_gate: decidable={decidable} \
             effect_deviation={effect_deviation:.6} \
             required_half_width_margin={:.6} required_endpoint_margin={:.6}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!("raw_current_seconds={current_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_scipy_over_frankenscipy={ratios:?}");
        println!("raw_current_null={current_nulls:?}");
        println!("raw_live_null={live_nulls:?}");
        println!(
            "verdict={} (non-exclusive host; NOT DECIDED-class evidence)",
            if decidable && ratio_high < 0.25 {
                "PROVISIONAL FRANKENSCIPY LOSS; PROFILE ADMITTED"
            } else if decidable && ratio_low > 1.0 {
                "PROVISIONAL FRANKENSCIPY WIN; NO PROFILE"
            } else {
                "PROVISIONAL INDETERMINATE; NO PROFILE"
            }
        );
        Ok(())
    }

    pub fn run_laplacian_cycle_current_profile(n: usize, repetitions: usize) -> Result<(), String> {
        if n < 3 || repetitions < 1 {
            return Err("require n>=3 and repetitions>=1".to_string());
        }
        let matrix = cycle_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elapsed = time_current_cycle_laplacian(&matrix, repetitions);
        println!(
            "LAPLACIAN_CYCLE_FSCI_PROFILE n={n} input_nnz={} repetitions={repetitions} \
             elapsed_seconds={elapsed:.9} normed=false result_format=dense \
             result_elements={} actual_observed_worker_threads=1 \
             input_sha256={input_sha256}",
            matrix.nnz(),
            n * n
        );
        Ok(())
    }

    pub fn run_laplacian_cycle_vs_scipy(
        n: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if n != CYCLE_REGISTERED_N || rounds != CYCLE_REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot registration requires n={CYCLE_REGISTERED_N} \
                 rounds={CYCLE_REGISTERED_ROUNDS}"
            ));
        }
        let matrix = cycle_fixture(n);
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=weighted-undirected-cycle n={n} input_nnz={} \
             edge=(i,(i+1)%n) edge_weight=1+(i%29)/64 normed=false rounds={rounds} \
             construction_outside_timing=true serialization_outside_timing=true \
             requested_threads=1 actual_observed_frankenscipy_worker_threads=1 \
             null_design=four-call-forward-reverse-geometric-symmetrization",
            matrix.nnz()
        );
        let script = oracle_path(explicit_oracle)?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = ScipyExpm::start_laplacian_cycle(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("method=laplacian")
            || !identity.contains("solver_mod=scipy.sparse.csgraph._laplacian")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err(
                "live SciPy cycle Laplacian failed genuine-incumbent identity gate".to_string(),
            );
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        let case = scipy.initialize(&matrix)?;
        let expected_case = format!(
            "CASE method=laplacian n={n} nnz={} sorted=True canonical=True \
             finite=True normed=False form=array",
            matrix.nnz()
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy constructed the wrong cycle fixture: {case}"
            ));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let (live_result, live_indptr, live_indices, live_data) = scipy.laplacian_parity()?;
        let expected_result = format!(
            "RESULT rows={n} cols={n} nnz={CYCLE_RESULT_NNZ} \
             sorted=True canonical=True"
        );
        if live_result != expected_result {
            return Err(format!(
                "live SciPy cycle result contract failed: {live_result}"
            ));
        }
        let (outside_pattern_max, relative_l2) =
            validate_current_cycle_laplacian(&matrix, &live_indptr, &live_indices, &live_data)?;
        println!(
            "agreement: structural_components={CYCLE_RESULT_NNZ}/{} \
             current_outside_pattern_max={outside_pattern_max:.3e} \
             live_result_format=sparse live_result_nnz={CYCLE_RESULT_NNZ} \
             structural_relative_l2={relative_l2:.3e} \
             tolerance=4*EPSILON*max(1,abs(expected)) pass=true",
            n * n
        );

        let mut current_repetitions = 1usize;
        while time_current_cycle_laplacian(&matrix, current_repetitions) < CYCLE_MIN_SAMPLE_SECONDS
        {
            current_repetitions = current_repetitions
                .checked_mul(2)
                .ok_or_else(|| "current calibration repetition count overflowed".to_string())?;
        }
        let mut live_repetitions = 1usize;
        while scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)? < CYCLE_MIN_SAMPLE_SECONDS {
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live calibration repetition count overflowed".to_string())?;
        }
        println!(
            "calibration: current_repetitions={current_repetitions} \
             live_repetitions={live_repetitions} min_sample_ms={} \
             whole_public_calls=true separate_per_arm_repetitions=true",
            CYCLE_MIN_SAMPLE_SECONDS * 1_000.0
        );

        let mut current_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut current_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (current_batch, live_batch) = if round.is_multiple_of(2) {
                (
                    time_current_cycle_laplacian(&matrix, current_repetitions),
                    scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?,
                )
            } else {
                let live = scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?;
                let current = time_current_cycle_laplacian(&matrix, current_repetitions);
                (current, live)
            };

            let current_left_first = time_current_cycle_laplacian(&matrix, current_repetitions);
            let current_right_first = time_current_cycle_laplacian(&matrix, current_repetitions);
            let current_right_second = time_current_cycle_laplacian(&matrix, current_repetitions);
            let current_left_second = time_current_cycle_laplacian(&matrix, current_repetitions);
            let current_null = ((current_left_first / current_right_first)
                * (current_left_second / current_right_second))
                .sqrt();

            let live_left_first = scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?;
            let live_right_first = scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?;
            let live_right_second = scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?;
            let live_left_second = scipy.solve(live_repetitions, CYCLE_RESULT_NNZ)?;
            let live_null = ((live_left_first / live_right_first)
                * (live_left_second / live_right_second))
                .sqrt();

            let current = current_batch / current_repetitions as f64;
            let live = live_batch / live_repetitions as f64;
            current_times.push(current);
            live_times.push(live);
            ratios.push(live / current);
            current_nulls.push(current_null);
            live_nulls.push(live_null);
        }
        scipy.quit();

        let current_p50 = median(current_times.clone());
        let live_p50 = median(live_times.clone());
        let ratio_median = median(ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios);
        let current_null_median = median(current_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (current_null_low, current_null_high) = bootstrap_median_ci(&current_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = current_null_high
            .max(live_null_high)
            .max(1.0 / current_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((current_null_high - current_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let effect_deviation = (1.0 - ratio_high).max(0.0);
        let null_medians_ok =
            (current_null_median - 1.0).abs() <= 0.02 && (live_null_median - 1.0).abs() <= 0.02;
        let clears_null =
            effect_deviation > 2.0 * null_half_width && effect_deviation > 2.0 * (null_edge - 1.0);
        let profile_admitted = ratio_high < 0.25 && null_medians_ok && clears_null;
        println!(
            "timing: current_p50_ms={:.6} live_scipy_p50_ms={:.6} \
             incumbent_ratio_scipy_over_frankenscipy={ratio_median:.6} \
             bootstrap_median_ci95=[{ratio_low:.6},{ratio_high:.6}] \
             current_cv={:.6} live_cv={:.6} ratio_cv={:.6}",
            current_p50 * 1_000.0,
            live_p50 * 1_000.0,
            cv(&current_times),
            cv(&live_times),
            cv(&ratios)
        );
        println!(
            "nulls: design=four-call-forward-reverse-geometric-symmetrization \
             current_median={current_null_median:.6} \
             current_ci95=[{current_null_low:.6},{current_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "registered_loss_gate: profile_admitted={profile_admitted} ratio_ci_high={ratio_high:.6} \
             required_ratio_ci_high_lt=0.250000 effect_deviation={effect_deviation:.6} \
             clears_2x_null={clears_null} required_half_width_margin={:.6} \
             required_endpoint_margin={:.6}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!("raw_current_seconds={current_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_scipy_over_frankenscipy={ratios:?}");
        println!("raw_current_symmetrized_null={current_nulls:?}");
        println!("raw_live_symmetrized_null={live_nulls:?}");
        println!(
            "verdict={} (non-exclusive host; NOT DECIDED-class evidence)",
            if profile_admitted {
                "PROVISIONAL FRANKENSCIPY LOSS; PROFILE ADMITTED"
            } else {
                "PROVISIONAL INDETERMINATE; NO PROFILE"
            }
        );
        Ok(())
    }

    pub fn run_csc_add_current_profile(n: usize, repetitions: usize) -> Result<(), String> {
        if n != CSC_ADD_REGISTERED_N || repetitions < 1 {
            return Err(format!(
                "registered CSC-add profile requires n={CSC_ADD_REGISTERED_N} and repetitions>=1"
            ));
        }
        let (lhs, rhs) = csc_add_fixture(n);
        let input_sha256 = csc_pair_input_sha256(&lhs, &rhs)?;
        let result =
            add_csc(&lhs, &rhs).map_err(|error| format!("FrankenSciPy CSC-add warmup: {error}"))?;
        let elapsed = time_current_csc_add(&lhs, &rhs, repetitions);
        let available_threads = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let selected_workers = available_threads.min(16).min(n / 256).max(1);
        println!(
            "CSC_ADD_FSCI_PROFILE n={n} lhs_nnz={} rhs_nnz={} repetitions={repetitions} \
             elapsed_seconds={elapsed:.9} result_format=csc result_nnz={} \
             available_threads={available_threads} selected_worker_threads={selected_workers} \
             input_sha256={input_sha256}",
            lhs.nnz(),
            rhs.nnz(),
            result.nnz()
        );
        Ok(())
    }

    pub fn run_csc_add_vs_scipy(
        n: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if n != CSC_ADD_REGISTERED_N || rounds != CSC_ADD_REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot registration requires n={CSC_ADD_REGISTERED_N} \
                 rounds={CSC_ADD_REGISTERED_ROUNDS}"
            ));
        }
        let (lhs, rhs) = csc_add_fixture(n);
        let input_sha256 = csc_pair_input_sha256(&lhs, &rhs)?;
        let elf_sha256 = sha256_of_self()?;
        let available_threads = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let selected_workers = available_threads.min(16).min(n / 256).max(1);
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=deterministic-canonical-csc-pair n={n} entries_per_column={} \
             lhs_nnz={} rhs_nnz={} row=(173*j+17*c+89*side)%n \
             value=((c+3*j+11*side)%37-18)/32 rounds={rounds} \
             construction_outside_timing=true serialization_outside_timing=true \
             requested_threads={available_threads} selected_frankenscipy_worker_threads={selected_workers} \
             requested_live_threads=1 null_design=four-call-forward-reverse-geometric-symmetrization",
            CSC_ADD_ENTRIES_PER_COLUMN,
            lhs.nnz(),
            rhs.nnz()
        );
        let script = oracle_path(explicit_oracle)?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = ScipyExpm::start_csc_add(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("method=csc_add")
            || !identity.contains("solver_mod=scipy.sparse._compressed")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy CSC-add failed genuine-incumbent identity gate".to_string());
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its CSC-add engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid CSC-add engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        let case = scipy.initialize_csc_pair(&lhs, &rhs)?;
        let expected_case = format!(
            "CASE method=csc_add n={n} lhs_nnz={} rhs_nnz={} \
             lhs_sorted=True rhs_sorted=True lhs_canonical=True rhs_canonical=True finite=True",
            lhs.nnz(),
            rhs.nnz()
        );
        if case != expected_case {
            return Err(format!("live SciPy constructed the wrong CSC pair: {case}"));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let (live_result, live_indptr, live_indices, live_data) = scipy.csc_add_parity()?;
        let live_nnz = field_value(&live_result, "nnz=")
            .ok_or_else(|| format!("live CSC-add result omitted nnz: {live_result}"))?
            .parse::<usize>()
            .map_err(|error| format!("parse live CSC-add nnz: {error}"))?;
        let expected_result =
            format!("RESULT rows={n} cols={n} nnz={live_nnz} sorted=True canonical=True");
        if live_result != expected_result
            || live_indptr.len() != n + 1
            || live_indices.len() != live_nnz
            || live_data.len() != live_nnz
        {
            return Err(format!(
                "live SciPy CSC-add result contract failed: {live_result}"
            ));
        }
        let (current_nnz, max_abs_difference, relative_l2) =
            validate_current_csc_add(&lhs, &rhs, &live_indptr, &live_indices, &live_data)?;
        println!(
            "agreement: structural_components={current_nnz}/{live_nnz} \
             max_abs_difference={max_abs_difference:.3e} relative_l2={relative_l2:.3e} \
             current_result_format=csc live_result_format=csc \
             tolerance=4*EPSILON*max(1,abs(live)) pass=true"
        );

        let mut current_repetitions = 1usize;
        while time_current_csc_add(&lhs, &rhs, current_repetitions) < CSC_ADD_MIN_SAMPLE_SECONDS {
            current_repetitions = current_repetitions
                .checked_mul(2)
                .ok_or_else(|| "current CSC-add calibration overflowed".to_string())?;
        }
        let mut live_repetitions = 1usize;
        while scipy.solve(live_repetitions, live_nnz)? < CSC_ADD_MIN_SAMPLE_SECONDS {
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live CSC-add calibration overflowed".to_string())?;
        }
        println!(
            "calibration: current_repetitions={current_repetitions} \
             live_repetitions={live_repetitions} min_sample_ms={} \
             whole_public_calls=true separate_per_arm_repetitions=true",
            CSC_ADD_MIN_SAMPLE_SECONDS * 1_000.0
        );

        let mut current_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut current_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (current_batch, live_batch) = if round.is_multiple_of(2) {
                (
                    time_current_csc_add(&lhs, &rhs, current_repetitions),
                    scipy.solve(live_repetitions, live_nnz)?,
                )
            } else {
                let live = scipy.solve(live_repetitions, live_nnz)?;
                let current = time_current_csc_add(&lhs, &rhs, current_repetitions);
                (current, live)
            };

            let current_left_first = time_current_csc_add(&lhs, &rhs, current_repetitions);
            let current_right_first = time_current_csc_add(&lhs, &rhs, current_repetitions);
            let current_right_second = time_current_csc_add(&lhs, &rhs, current_repetitions);
            let current_left_second = time_current_csc_add(&lhs, &rhs, current_repetitions);
            let current_null = ((current_left_first / current_right_first)
                * (current_left_second / current_right_second))
                .sqrt();

            let live_left_first = scipy.solve(live_repetitions, live_nnz)?;
            let live_right_first = scipy.solve(live_repetitions, live_nnz)?;
            let live_right_second = scipy.solve(live_repetitions, live_nnz)?;
            let live_left_second = scipy.solve(live_repetitions, live_nnz)?;
            let live_null = ((live_left_first / live_right_first)
                * (live_left_second / live_right_second))
                .sqrt();

            let current = current_batch / current_repetitions as f64;
            let live = live_batch / live_repetitions as f64;
            current_times.push(current);
            live_times.push(live);
            ratios.push(live / current);
            current_nulls.push(current_null);
            live_nulls.push(live_null);
        }
        scipy.quit();

        let current_p50 = median(current_times.clone());
        let current_p95 = percentile(current_times.clone(), 95, 100);
        let current_p99 = percentile(current_times.clone(), 99, 100);
        let live_p50 = median(live_times.clone());
        let live_p95 = percentile(live_times.clone(), 95, 100);
        let live_p99 = percentile(live_times.clone(), 99, 100);
        let ratio_median = median(ratios.clone());
        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios);
        let current_null_median = median(current_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (current_null_low, current_null_high) = bootstrap_median_ci(&current_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = current_null_high
            .max(live_null_high)
            .max(1.0 / current_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((current_null_high - current_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let effect_deviation = (1.0 - ratio_high).max(0.0);
        let null_medians_ok =
            (current_null_median - 1.0).abs() <= 0.02 && (live_null_median - 1.0).abs() <= 0.02;
        let clears_null =
            effect_deviation > 2.0 * null_half_width && effect_deviation > 2.0 * (null_edge - 1.0);
        let profile_admitted = ratio_high < 0.50 && null_medians_ok && clears_null;
        println!(
            "timing: current_p50_ms={:.6} current_p95_ms={:.6} current_p99_ms={:.6} \
             live_scipy_p50_ms={:.6} live_scipy_p95_ms={:.6} live_scipy_p99_ms={:.6} \
             incumbent_ratio_scipy_over_frankenscipy={ratio_median:.6} \
             bootstrap_median_ci95=[{ratio_low:.6},{ratio_high:.6}] \
             current_cv={:.6} live_cv={:.6} ratio_cv={:.6}",
            current_p50 * 1_000.0,
            current_p95 * 1_000.0,
            current_p99 * 1_000.0,
            live_p50 * 1_000.0,
            live_p95 * 1_000.0,
            live_p99 * 1_000.0,
            cv(&current_times),
            cv(&live_times),
            cv(&ratios)
        );
        println!(
            "nulls: design=four-call-forward-reverse-geometric-symmetrization \
             current_median={current_null_median:.6} \
             current_ci95=[{current_null_low:.6},{current_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "registered_loss_gate: profile_admitted={profile_admitted} ratio_ci_high={ratio_high:.6} \
             required_ratio_ci_high_lt=0.500000 effect_deviation={effect_deviation:.6} \
             clears_2x_null={clears_null} required_half_width_margin={:.6} \
             required_endpoint_margin={:.6}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!("raw_current_seconds={current_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_scipy_over_frankenscipy={ratios:?}");
        println!("raw_current_symmetrized_null={current_nulls:?}");
        println!("raw_live_symmetrized_null={live_nulls:?}");
        println!(
            "verdict={} (non-exclusive host; NOT DECIDED-class evidence)",
            if profile_admitted {
                "PROVISIONAL FRANKENSCIPY LOSS; PROFILE ADMITTED"
            } else {
                "PROVISIONAL INDETERMINATE; NO PROFILE"
            }
        );
        Ok(())
    }

    pub fn run_laplacian_torus_candidate_vs_scipy(
        side: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if side != TORUS_SIDE || rounds != TORUS_REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot completion requires side={TORUS_SIDE} rounds={TORUS_REGISTERED_ROUNDS}"
            ));
        }
        print_hardware_provenance()?;
        let matrix = torus_fixture(side);
        if matrix.shape() != Shape2D::new(TORUS_N, TORUS_N)
            || matrix.nnz() != TORUS_INPUT_NNZ
            || !matrix.canonical_meta().sorted_indices
            || !matrix.canonical_meta().deduplicated
        {
            return Err("registered torus fixture has the wrong canonical contract".to_string());
        }
        let input_sha256 = canonical_input_sha256(&matrix)?;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=normalized-periodic-two-dimensional-grid side={side} n={TORUS_N} \
             input_nnz={TORUS_INPUT_NNZ} result_nnz={TORUS_RESULT_NNZ} \
             four_wraparound_neighbors=true edge_weight=1 normed=true rounds={rounds} \
             construction_outside_timing=true serialization_outside_timing=true \
             three_arm_order=six-permutation-rotation \
             null_design=four-call-forward-reverse-geometric-symmetrization \
             same_invocation=true side_by_side=true"
        );

        let script = oracle_path(explicit_oracle)?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = ScipyExpm::start_laplacian(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("method=laplacian")
            || !identity.contains("solver_mod=scipy.sparse.csgraph._laplacian")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy torus failed genuine-incumbent identity gate".to_string());
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its Laplacian engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid Laplacian engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");

        let case = scipy.initialize(&matrix)?;
        let expected_case = format!(
            "CASE method=laplacian n={TORUS_N} nnz={TORUS_INPUT_NNZ} \
             sorted=True canonical=True finite=True normed=True form=array"
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy constructed the wrong torus fixture: {case}"
            ));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let (live_result, live_indptr, live_indices, live_data) = scipy.laplacian_parity()?;
        let expected_result = format!(
            "RESULT rows={TORUS_N} cols={TORUS_N} nnz={TORUS_RESULT_NNZ} \
             sorted=True canonical=True"
        );
        if live_result != expected_result {
            return Err(format!(
                "live SciPy torus result contract failed: {live_result}"
            ));
        }
        let (maximum_absolute_difference, relative_l2, dense_bit_mismatches) =
            validate_torus_candidate(&matrix, &live_indptr, &live_indices, &live_data)?;
        println!(
            "agreement: structural_components={TORUS_RESULT_NNZ}/{TORUS_RESULT_NNZ} \
             candidate_result_format=csr candidate_result_nnz={TORUS_RESULT_NNZ} \
             candidate_canonical=true forced_dense_result_format=csr \
             dense_reference_bit_mismatches={dense_bit_mismatches} \
             live_result_format=sparse max_abs_difference={maximum_absolute_difference:.3e} \
             structural_relative_l2={relative_l2:.3e} \
             tolerance=4*EPSILON*max(1,abs(live)) pass=true"
        );

        let mut candidate_repetitions = 1usize;
        while time_torus_candidate(&matrix, candidate_repetitions) < CYCLE_MIN_SAMPLE_SECONDS {
            candidate_repetitions = candidate_repetitions
                .checked_mul(2)
                .ok_or_else(|| "candidate calibration repetition count overflowed".to_string())?;
        }
        let mut dense_repetitions = 1usize;
        while time_torus_dense_reference(&matrix, dense_repetitions) < CYCLE_MIN_SAMPLE_SECONDS {
            dense_repetitions = dense_repetitions
                .checked_mul(2)
                .ok_or_else(|| "dense calibration repetition count overflowed".to_string())?;
        }
        let mut live_repetitions = 1usize;
        while scipy.solve(live_repetitions, TORUS_RESULT_NNZ)? < CYCLE_MIN_SAMPLE_SECONDS {
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live calibration repetition count overflowed".to_string())?;
        }
        println!(
            "calibration: candidate_repetitions={candidate_repetitions} \
             forced_dense_repetitions={dense_repetitions} live_repetitions={live_repetitions} \
             min_sample_ms={} whole_public_calls=true separate_per_arm_repetitions=true",
            CYCLE_MIN_SAMPLE_SECONDS * 1_000.0
        );
        let quiescence_pre = sample_host_wide_quiescence("pre")?;

        const ORDERS: [[u8; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut candidate_times = Vec::with_capacity(rounds);
        let mut dense_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut dense_over_candidate = Vec::with_capacity(rounds);
        let mut live_over_candidate = Vec::with_capacity(rounds);
        let mut candidate_nulls = Vec::with_capacity(rounds);
        let mut dense_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let mut candidate_batch = 0.0;
            let mut dense_batch = 0.0;
            let mut live_batch = 0.0;
            for arm in ORDERS[round % ORDERS.len()] {
                match arm {
                    0 => candidate_batch = time_torus_candidate(&matrix, candidate_repetitions),
                    1 => {
                        dense_batch = time_torus_dense_reference(&matrix, dense_repetitions);
                    }
                    2 => live_batch = scipy.solve(live_repetitions, TORUS_RESULT_NNZ)?,
                    _ => unreachable!(),
                }
            }

            let mut candidate_null = 0.0;
            let mut dense_null = 0.0;
            let mut live_null = 0.0;
            for arm in ORDERS[round % ORDERS.len()] {
                match arm {
                    0 => {
                        candidate_null = four_call_geometric_null(|| {
                            Ok(time_torus_candidate(&matrix, candidate_repetitions))
                        })?;
                    }
                    1 => {
                        dense_null = four_call_geometric_null(|| {
                            Ok(time_torus_dense_reference(&matrix, dense_repetitions))
                        })?;
                    }
                    2 => {
                        live_null = four_call_geometric_null(|| {
                            scipy.solve(live_repetitions, TORUS_RESULT_NNZ)
                        })?;
                    }
                    _ => unreachable!(),
                }
            }

            let candidate = candidate_batch / candidate_repetitions as f64;
            let dense = dense_batch / dense_repetitions as f64;
            let live = live_batch / live_repetitions as f64;
            candidate_times.push(candidate);
            dense_times.push(dense);
            live_times.push(live);
            dense_over_candidate.push(dense / candidate);
            live_over_candidate.push(live / candidate);
            candidate_nulls.push(candidate_null);
            dense_nulls.push(dense_null);
            live_nulls.push(live_null);
        }
        let quiescence_post = sample_host_wide_quiescence("post")?;
        scipy.quit();

        let candidate_p50 = median(candidate_times.clone());
        let candidate_p95 = percentile(candidate_times.clone(), 95, 100);
        let candidate_p99 = percentile(candidate_times.clone(), 99, 100);
        let dense_p50 = median(dense_times.clone());
        let dense_p95 = percentile(dense_times.clone(), 95, 100);
        let dense_p99 = percentile(dense_times.clone(), 99, 100);
        let live_p50 = median(live_times.clone());
        let live_p95 = percentile(live_times.clone(), 95, 100);
        let live_p99 = percentile(live_times.clone(), 99, 100);
        let dense_ratio_median = median(dense_over_candidate.clone());
        let live_ratio_median = median(live_over_candidate.clone());
        let (dense_ratio_low, dense_ratio_high) = bootstrap_median_ci(&dense_over_candidate);
        let (live_ratio_low, live_ratio_high) = bootstrap_median_ci(&live_over_candidate);

        let candidate_null_median = median(candidate_nulls.clone());
        let dense_null_median = median(dense_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (candidate_null_low, candidate_null_high) = bootstrap_median_ci(&candidate_nulls);
        let (dense_null_low, dense_null_high) = bootstrap_median_ci(&dense_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = candidate_null_high
            .max(dense_null_high)
            .max(live_null_high)
            .max(1.0 / candidate_null_low.max(1.0e-12))
            .max(1.0 / dense_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((candidate_null_high - candidate_null_low) / 2.0)
            .max((dense_null_high - dense_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let half_width_margin = 2.0 * null_half_width;
        let endpoint_margin = 2.0 * (null_edge - 1.0);
        let dense_effect_deviation = (dense_ratio_low - 1.0).max(0.0);
        let live_effect_deviation = (live_ratio_low - 1.0).max(0.0);
        let null_medians_ok = (candidate_null_median - 1.0).abs() <= 0.02
            && (dense_null_median - 1.0).abs() <= 0.02
            && (live_null_median - 1.0).abs() <= 0.02;
        let dense_clears_null =
            dense_effect_deviation > half_width_margin && dense_effect_deviation > endpoint_margin;
        let live_clears_null =
            live_effect_deviation > half_width_margin && live_effect_deviation > endpoint_margin;
        let tails_pass = candidate_p95 < live_p95 && candidate_p99 < live_p99;
        let keep = dense_ratio_low > 100.0
            && live_ratio_low > 1.10
            && tails_pass
            && null_medians_ok
            && dense_clears_null
            && live_clears_null;
        let quiescence_all_clear = quiescence_pre && quiescence_post;

        println!(
            "timing: candidate_p50_ms={:.6} candidate_p95_ms={:.6} candidate_p99_ms={:.6} \
             forced_dense_p50_ms={:.6} forced_dense_p95_ms={:.6} forced_dense_p99_ms={:.6} \
             live_scipy_p50_ms={:.6} live_scipy_p95_ms={:.6} live_scipy_p99_ms={:.6}",
            candidate_p50 * 1_000.0,
            candidate_p95 * 1_000.0,
            candidate_p99 * 1_000.0,
            dense_p50 * 1_000.0,
            dense_p95 * 1_000.0,
            dense_p99 * 1_000.0,
            live_p50 * 1_000.0,
            live_p95 * 1_000.0,
            live_p99 * 1_000.0
        );
        println!(
            "ratios: forced_dense_over_candidate={dense_ratio_median:.6} \
             forced_dense_over_candidate_ci95=[{dense_ratio_low:.6},{dense_ratio_high:.6}] \
             incumbent_ratio_scipy_over_candidate={live_ratio_median:.6} \
             incumbent_ratio_ci95=[{live_ratio_low:.6},{live_ratio_high:.6}] \
             candidate_cv={:.6} forced_dense_cv={:.6} live_cv={:.6} \
             forced_dense_ratio_cv={:.6} incumbent_ratio_cv={:.6}",
            cv(&candidate_times),
            cv(&dense_times),
            cv(&live_times),
            cv(&dense_over_candidate),
            cv(&live_over_candidate)
        );
        println!(
            "nulls: candidate_median={candidate_null_median:.6} \
             candidate_ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] \
             forced_dense_median={dense_null_median:.6} \
             forced_dense_ci95=[{dense_null_low:.6},{dense_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "registered_keep_gate: keep={keep} dense_ci_low={dense_ratio_low:.6} \
             required_dense_ci_low_gt=100.000000 incumbent_ci_low={live_ratio_low:.6} \
             required_incumbent_ci_low_gt=1.100000 candidate_tails_below_live={tails_pass} \
             dense_effect_deviation={dense_effect_deviation:.6} \
             live_effect_deviation={live_effect_deviation:.6} \
             dense_clears_2x_null={dense_clears_null} live_clears_2x_null={live_clears_null} \
             required_half_width_margin={half_width_margin:.6} \
             required_endpoint_margin={endpoint_margin:.6} \
             host_wide_quiescence_all_clear={quiescence_all_clear}"
        );
        println!("raw_candidate_seconds={candidate_times:?}");
        println!("raw_forced_dense_seconds={dense_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_forced_dense_over_candidate={dense_over_candidate:?}");
        println!("raw_scipy_over_candidate={live_over_candidate:?}");
        println!("raw_candidate_symmetrized_null={candidate_nulls:?}");
        println!("raw_forced_dense_symmetrized_null={dense_nulls:?}");
        println!("raw_live_symmetrized_null={live_nulls:?}");
        println!(
            "verdict={} evidence_class=PROVISIONAL_NON_EXCLUSIVE \
             competitive_campaign_win_forbidden=true",
            if keep {
                "STRUCTURAL-API-WIN GATES PASS; KEEP"
            } else {
                "CANDIDATE GATE FAILED; REVERT"
            }
        );
        Ok(())
    }

    pub fn run_transpose_view_vs_scipy(
        rows: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if rows != TRANSPOSE_ROWS || rounds != TRANSPOSE_REGISTERED_ROUNDS {
            return Err(format!(
                "one-shot completion requires rows={TRANSPOSE_ROWS} \
                 rounds={TRANSPOSE_REGISTERED_ROUNDS}"
            ));
        }
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let claim_id = required_env("COORDINATION_CLAIM_ID")?;
        let release_id = required_env("COORDINATION_RELEASE_ID")?;
        let lock_held = required_env("FSCI_BENCH_LOCK_HELD")?;
        if lock_held != "1" {
            return Err("filesystem benchmark lock must be held".to_string());
        }
        print_hardware_provenance()?;

        let matrix = transpose_fixture();
        if matrix.shape() != Shape2D::new(TRANSPOSE_ROWS, TRANSPOSE_COLS)
            || matrix.nnz() != TRANSPOSE_NNZ
            || !matrix.canonical_meta().sorted_indices
            || !matrix.canonical_meta().deduplicated
        {
            return Err(
                "registered transpose fixture has the wrong canonical contract".to_string(),
            );
        }
        let input_sha256 = compressed_parts_sha256(
            matrix.shape(),
            matrix.data(),
            matrix.indices(),
            matrix.indptr(),
        )?;
        let candidate = sparse_transpose_view(&matrix);
        let candidate_output_sha256 = compressed_parts_sha256(
            candidate.shape(),
            candidate.data(),
            candidate.indices(),
            candidate.indptr(),
        )?;
        let candidate_buffers_shared =
            std::ptr::eq(candidate.data().as_ptr(), matrix.data().as_ptr())
                && std::ptr::eq(candidate.indices().as_ptr(), matrix.indices().as_ptr())
                && std::ptr::eq(candidate.indptr().as_ptr(), matrix.indptr().as_ptr());
        if candidate.shape() != Shape2D::new(TRANSPOSE_COLS, TRANSPOSE_ROWS)
            || candidate.nnz() != TRANSPOSE_NNZ
            || candidate.canonical_meta() != matrix.canonical_meta()
            || !candidate_buffers_shared
        {
            return Err("borrowed transpose view failed its representation contract".to_string());
        }
        let control = sparse_transpose(&matrix);
        if control.shape() != candidate.shape() || control.nnz() != candidate.nnz() {
            return Err("materialized transpose control has the wrong shape or nnz".to_string());
        }
        for row in [0, 1, TRANSPOSE_ROWS / 2, TRANSPOSE_ROWS - 1] {
            let base = (17 * row) % (TRANSPOSE_COLS - TRANSPOSE_ENTRIES_PER_ROW + 1);
            for slot in 0..TRANSPOSE_ENTRIES_PER_ROW {
                let column = base + slot;
                let expected = 1.0 + ((row + column) % 17) as f64 / 64.0;
                if candidate
                    .get(column, row)
                    .map_err(|error| format!("candidate sampled lookup: {error}"))?
                    .to_bits()
                    != expected.to_bits()
                {
                    return Err(format!("candidate transpose mismatch at ({column},{row})"));
                }
                let start = control.indptr()[column];
                let end = control.indptr()[column + 1];
                let offset = control.indices()[start..end]
                    .binary_search(&row)
                    .map_err(|_| format!("control transpose omitted ({column},{row})"))?;
                if control.data()[start + offset].to_bits() != expected.to_bits() {
                    return Err(format!("control transpose mismatch at ({column},{row})"));
                }
            }
        }
        drop(control);

        let elf_sha256 = sha256_of_self()?;
        let harness_source_sha256 = format!("{:x}", Sha256::digest(HARNESS_SOURCE));
        let formats_source_sha256 = format!("{:x}", Sha256::digest(FORMATS_SOURCE));
        let linalg_source_sha256 = format!("{:x}", Sha256::digest(LINALG_SOURCE));
        let embedded_oracle_sha256 = format!("{:x}", Sha256::digest(ORACLE_SOURCE));
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "build_identity: source_commit={source_commit} builder_identity={builder_identity} \
             build_route={build_route} coordination_claim_id={claim_id} \
             coordination_release_id={release_id} filesystem_lock_held={lock_held}"
        );
        println!(
            "source_identity: harness_sha256={harness_source_sha256} \
             formats_sha256={formats_source_sha256} linalg_sha256={linalg_source_sha256} \
             embedded_oracle_sha256={embedded_oracle_sha256}"
        );
        println!(
            "fixture=deterministic-canonical-rectangular-csr rows={TRANSPOSE_ROWS} \
             cols={TRANSPOSE_COLS} entries_per_row={TRANSPOSE_ENTRIES_PER_ROW} \
             nnz={TRANSPOSE_NNZ} column_base=(17*row)%(cols-width+1) \
             value=1+((row+column)%17)/64 rounds={rounds} \
             construction_outside_timing=true serialization_outside_timing=true \
             parity_outside_timing=true three_arm_order=six-permutation-rotation \
             null_design=four-call-forward-reverse-geometric-symmetrization \
             same_invocation=true side_by_side=true"
        );

        let script = oracle_path(explicit_oracle)?;
        let oracle_bytes = std::fs::read(&script)
            .map_err(|error| format!("read transferred SciPy oracle: {error}"))?;
        let transferred_oracle_sha256 = format!("{:x}", Sha256::digest(&oracle_bytes));
        if transferred_oracle_sha256 != embedded_oracle_sha256 {
            return Err(format!(
                "transferred oracle SHA-256 mismatch: embedded={embedded_oracle_sha256} \
                 transferred={transferred_oracle_sha256}"
            ));
        }
        println!(
            "scipy_oracle_script={} transferred_oracle_sha256={transferred_oracle_sha256} \
             oracle_hash_match=true",
            script.display()
        );
        let (mut scipy, identity) = ScipyExpm::start_transpose(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains("method=csr_transpose_view")
            || !identity.contains("solver_mod=scipy.sparse._csr")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy transpose failed genuine-incumbent identity gate".to_string());
        }
        let scipy_engine_sha256 = field_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy omitted its transpose engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy reported an invalid transpose engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");

        let case = scipy.initialize_transpose(&matrix)?;
        let expected_case = format!(
            "CASE method=csr_transpose_view rows={TRANSPOSE_ROWS} cols={TRANSPOSE_COLS} \
             nnz={TRANSPOSE_NNZ} sorted=True canonical=True finite=True result_format=csc \
             result_rows={TRANSPOSE_COLS} result_cols={TRANSPOSE_ROWS} data_shared=True \
             indices_shared=True indptr_shared=True"
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy constructed the wrong transpose case: {case}"
            ));
        }
        println!("scipy_case: {case}");
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !input_sha256.bytes().eq(scipy_input_sha256.bytes()) {
            return Err(format!(
                "input digest mismatch: frankenscipy={input_sha256} scipy={scipy_input_sha256}"
            ));
        }
        let (live_result, live_output_sha256) = scipy.transpose_parity()?;
        let expected_result = format!(
            "RESULT rows={TRANSPOSE_COLS} cols={TRANSPOSE_ROWS} nnz={TRANSPOSE_NNZ} \
             format=csc sorted=True canonical=True data_shared=True indices_shared=True \
             indptr_shared=True"
        );
        if live_result != expected_result || live_output_sha256 != candidate_output_sha256 {
            return Err(format!(
                "transpose output contract mismatch: result={live_result} \
                 candidate_sha256={candidate_output_sha256} live_sha256={live_output_sha256}"
            ));
        }
        println!(
            "input_sha256={input_sha256} frankenscipy_input_sha256={input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        println!(
            "agreement: candidate_output_sha256={candidate_output_sha256} \
             scipy_output_sha256={live_output_sha256} output_digest_match=true \
             candidate_result_format=csc_view live_result_format=csc \
             shape={TRANSPOSE_COLS}x{TRANSPOSE_ROWS} nnz={TRANSPOSE_NNZ} \
             candidate_data_shared=true candidate_indices_shared=true \
             candidate_indptr_shared=true live_data_shared=true \
             live_indices_shared=true live_indptr_shared=true exact=true"
        );

        let quiescence_pre = sample_host_wide_quiescence("pre")?;
        let mut candidate_repetitions = 1usize;
        let candidate_calibration_seconds = loop {
            let elapsed = time_transpose_view(&matrix, candidate_repetitions);
            if elapsed >= TRANSPOSE_MIN_SAMPLE_SECONDS {
                break elapsed;
            }
            candidate_repetitions = candidate_repetitions
                .checked_mul(2)
                .ok_or_else(|| "candidate calibration repetition count overflowed".to_string())?;
        };
        let mut control_repetitions = 1usize;
        let control_calibration_seconds = loop {
            let elapsed = time_materialized_transpose(&matrix, control_repetitions);
            if elapsed >= TRANSPOSE_MIN_SAMPLE_SECONDS {
                break elapsed;
            }
            control_repetitions = control_repetitions
                .checked_mul(2)
                .ok_or_else(|| "control calibration repetition count overflowed".to_string())?;
        };
        let mut live_repetitions = 1usize;
        let live_calibration_seconds = loop {
            let elapsed = scipy.solve(live_repetitions, TRANSPOSE_NNZ)?;
            if elapsed >= TRANSPOSE_MIN_SAMPLE_SECONDS {
                break elapsed;
            }
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live calibration repetition count overflowed".to_string())?;
        };
        println!(
            "calibration: candidate_repetitions={candidate_repetitions} \
             materialized_control_repetitions={control_repetitions} \
             live_repetitions={live_repetitions} min_sample_ms={} \
             candidate_seconds={candidate_calibration_seconds:.9} \
             materialized_control_seconds={control_calibration_seconds:.9} \
             live_seconds={live_calibration_seconds:.9} whole_public_calls=true \
             separate_per_arm_repetitions=true",
            TRANSPOSE_MIN_SAMPLE_SECONDS * 1_000.0
        );
        for _ in 0..2 {
            let _ = time_transpose_view(&matrix, candidate_repetitions);
            let _ = time_materialized_transpose(&matrix, control_repetitions);
            let _ = scipy.solve(live_repetitions, TRANSPOSE_NNZ)?;
        }
        let quiescence_measurement = sample_host_wide_quiescence("measurement")?;

        const ORDERS: [[u8; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut candidate_times = Vec::with_capacity(rounds);
        let mut control_times = Vec::with_capacity(rounds);
        let mut live_times = Vec::with_capacity(rounds);
        let mut control_over_candidate = Vec::with_capacity(rounds);
        let mut live_over_candidate = Vec::with_capacity(rounds);
        let mut candidate_nulls = Vec::with_capacity(rounds);
        let mut control_nulls = Vec::with_capacity(rounds);
        let mut live_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let mut candidate_batch = 0.0;
            let mut control_batch = 0.0;
            let mut live_batch = 0.0;
            for arm in ORDERS[round % ORDERS.len()] {
                match arm {
                    0 => candidate_batch = time_transpose_view(&matrix, candidate_repetitions),
                    1 => {
                        control_batch = time_materialized_transpose(&matrix, control_repetitions);
                    }
                    2 => live_batch = scipy.solve(live_repetitions, TRANSPOSE_NNZ)?,
                    _ => unreachable!(),
                }
            }
            let mut candidate_null = 0.0;
            let mut control_null = 0.0;
            let mut live_null = 0.0;
            for arm in ORDERS[round % ORDERS.len()] {
                match arm {
                    0 => {
                        candidate_null = four_call_geometric_null(|| {
                            Ok(time_transpose_view(&matrix, candidate_repetitions))
                        })?;
                    }
                    1 => {
                        control_null = four_call_geometric_null(|| {
                            Ok(time_materialized_transpose(&matrix, control_repetitions))
                        })?;
                    }
                    2 => {
                        live_null = four_call_geometric_null(|| {
                            scipy.solve(live_repetitions, TRANSPOSE_NNZ)
                        })?;
                    }
                    _ => unreachable!(),
                }
            }
            let candidate_seconds = candidate_batch / candidate_repetitions as f64;
            let control_seconds = control_batch / control_repetitions as f64;
            let live_seconds = live_batch / live_repetitions as f64;
            candidate_times.push(candidate_seconds);
            control_times.push(control_seconds);
            live_times.push(live_seconds);
            control_over_candidate.push(control_seconds / candidate_seconds);
            live_over_candidate.push(live_seconds / candidate_seconds);
            candidate_nulls.push(candidate_null);
            control_nulls.push(control_null);
            live_nulls.push(live_null);
        }
        let quiescence_post = sample_host_wide_quiescence("post")?;
        scipy.quit();

        let candidate_p50 = median(candidate_times.clone());
        let candidate_p95 = percentile(candidate_times.clone(), 95, 100);
        let candidate_p99 = percentile(candidate_times.clone(), 99, 100);
        let control_p50 = median(control_times.clone());
        let control_p95 = percentile(control_times.clone(), 95, 100);
        let control_p99 = percentile(control_times.clone(), 99, 100);
        let live_p50 = median(live_times.clone());
        let live_p95 = percentile(live_times.clone(), 95, 100);
        let live_p99 = percentile(live_times.clone(), 99, 100);
        let control_ratio_median = median(control_over_candidate.clone());
        let live_ratio_median = median(live_over_candidate.clone());
        let (control_ratio_low, control_ratio_high) = bootstrap_median_ci(&control_over_candidate);
        let (live_ratio_low, live_ratio_high) = bootstrap_median_ci(&live_over_candidate);
        let candidate_null_median = median(candidate_nulls.clone());
        let control_null_median = median(control_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        let (candidate_null_low, candidate_null_high) = bootstrap_median_ci(&candidate_nulls);
        let (control_null_low, control_null_high) = bootstrap_median_ci(&control_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let null_edge = candidate_null_high
            .max(control_null_high)
            .max(live_null_high)
            .max(1.0 / candidate_null_low.max(1.0e-12))
            .max(1.0 / control_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let null_half_width = ((candidate_null_high - candidate_null_low) / 2.0)
            .max((control_null_high - control_null_low) / 2.0)
            .max((live_null_high - live_null_low) / 2.0);
        let half_width_margin = 2.0 * null_half_width;
        let endpoint_margin = 2.0 * (null_edge - 1.0);
        let control_effect_deviation = (control_ratio_low - 1.0).max(0.0);
        let live_effect_deviation = (live_ratio_low - 1.0).max(0.0);
        let null_medians_ok = (candidate_null_median - 1.0).abs() <= 0.02
            && (control_null_median - 1.0).abs() <= 0.02
            && (live_null_median - 1.0).abs() <= 0.02;
        let control_clears_null = control_effect_deviation > half_width_margin
            && control_effect_deviation > endpoint_margin;
        let live_clears_null =
            live_effect_deviation > half_width_margin && live_effect_deviation > endpoint_margin;
        let tails_pass = candidate_p95 < control_p95
            && candidate_p99 < control_p99
            && candidate_p95 < live_p95
            && candidate_p99 < live_p99;
        let duration_pass = candidate_calibration_seconds >= TRANSPOSE_MIN_SAMPLE_SECONDS
            && control_calibration_seconds >= TRANSPOSE_MIN_SAMPLE_SECONDS
            && live_calibration_seconds >= TRANSPOSE_MIN_SAMPLE_SECONDS;
        let keep = control_ratio_low > 1_000.0
            && live_ratio_low > 20.0
            && tails_pass
            && duration_pass
            && null_medians_ok
            && control_clears_null
            && live_clears_null;
        let quiescence_all_clear = quiescence_pre && quiescence_measurement && quiescence_post;

        println!(
            "timing: candidate_p50_ns={:.6} candidate_p95_ns={:.6} \
             candidate_p99_ns={:.6} materialized_control_p50_ms={:.6} \
             materialized_control_p95_ms={:.6} materialized_control_p99_ms={:.6} \
             live_scipy_p50_us={:.6} live_scipy_p95_us={:.6} live_scipy_p99_us={:.6}",
            candidate_p50 * 1.0e9,
            candidate_p95 * 1.0e9,
            candidate_p99 * 1.0e9,
            control_p50 * 1.0e3,
            control_p95 * 1.0e3,
            control_p99 * 1.0e3,
            live_p50 * 1.0e6,
            live_p95 * 1.0e6,
            live_p99 * 1.0e6
        );
        println!(
            "ratios: materialized_control_over_candidate={control_ratio_median:.6} \
             materialized_control_over_candidate_ci95=[{control_ratio_low:.6},{control_ratio_high:.6}] \
             incumbent_ratio_scipy_over_candidate={live_ratio_median:.6} \
             incumbent_ratio_ci95=[{live_ratio_low:.6},{live_ratio_high:.6}] \
             candidate_cv={:.6} materialized_control_cv={:.6} live_cv={:.6} \
             materialized_control_ratio_cv={:.6} incumbent_ratio_cv={:.6}",
            cv(&candidate_times),
            cv(&control_times),
            cv(&live_times),
            cv(&control_over_candidate),
            cv(&live_over_candidate)
        );
        println!(
            "nulls: candidate_median={candidate_null_median:.6} \
             candidate_ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] \
             materialized_control_median={control_null_median:.6} \
             materialized_control_ci95=[{control_null_low:.6},{control_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "registered_keep_gate: keep={keep} control_ci_low={control_ratio_low:.6} \
             required_control_ci_low_gt=1000.000000 incumbent_ci_low={live_ratio_low:.6} \
             required_incumbent_ci_low_gt=20.000000 candidate_tails_below_both={tails_pass} \
             duration_pass={duration_pass} control_effect_deviation={control_effect_deviation:.6} \
             live_effect_deviation={live_effect_deviation:.6} \
             control_clears_2x_null={control_clears_null} \
             live_clears_2x_null={live_clears_null} \
             required_half_width_margin={half_width_margin:.6} \
             required_endpoint_margin={endpoint_margin:.6} \
             host_wide_quiescence_all_clear={quiescence_all_clear}"
        );
        println!("raw_candidate_seconds={candidate_times:?}");
        println!("raw_materialized_control_seconds={control_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_materialized_control_over_candidate={control_over_candidate:?}");
        println!("raw_scipy_over_candidate={live_over_candidate:?}");
        println!("raw_candidate_symmetrized_null={candidate_nulls:?}");
        println!("raw_materialized_control_symmetrized_null={control_nulls:?}");
        println!("raw_live_symmetrized_null={live_nulls:?}");
        println!(
            "verdict={} evidence_class=PROVISIONAL_NON_EXCLUSIVE \
             competitive_campaign_win_forbidden=true",
            if keep {
                "STRUCTURAL-API-WIN GATES PASS; KEEP"
            } else {
                "CANDIDATE GATE FAILED; REVERT"
            }
        );
        Ok(())
    }
}

const SEED: u64 = 0xBEEF_CAFE;

fn make_add_inputs(n: usize, density: f64) -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(n, n);
    let lhs = random(shape, density, SEED)
        .expect("random lhs")
        .to_csr()
        .expect("lhs csr");
    let rhs = random(shape, density, SEED ^ 0x5EED_1234)
        .expect("random rhs")
        .to_csr()
        .expect("rhs csr");
    (lhs, rhs)
}

fn cancellation_inputs() -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(3, 4);
    let lhs = CooMatrix::from_triplets(
        shape,
        vec![1.0, 2.0, -4.0, 5.0],
        vec![0, 1, 1, 2],
        vec![1, 0, 3, 2],
        false,
    )
    .expect("lhs coo")
    .to_csr()
    .expect("lhs csr");
    let rhs = CooMatrix::from_triplets(
        shape,
        vec![3.0, 4.0, -5.0, 6.0],
        vec![0, 1, 2, 2],
        vec![2, 3, 2, 3],
        false,
    )
    .expect("rhs coo")
    .to_csr()
    .expect("rhs csr");
    (lhs, rhs)
}

fn write_csr(output: &mut String, label: &str, matrix: &CsrMatrix) {
    let meta = matrix.canonical_meta();
    write!(
        output,
        "case={label} shape={}x{} nnz={} sorted={} deduplicated={} indptr=",
        matrix.shape().rows,
        matrix.shape().cols,
        matrix.nnz(),
        meta.sorted_indices,
        meta.deduplicated,
    )
    .expect("write header");
    for value in matrix.indptr() {
        write!(output, "{value},").expect("write indptr");
    }
    output.push_str(" indices=");
    for value in matrix.indices() {
        write!(output, "{value},").expect("write indices");
    }
    output.push_str(" data=");
    for value in matrix.data() {
        write!(output, "{:016x},", value.to_bits()).expect("write data");
    }
    output.push('\n');
}

fn add_csr_golden_text() -> String {
    let mut output = String::new();
    let cases = [(8usize, 0.25), (64, 0.05), (1024, 0.001)];
    for (n, density) in cases {
        let (lhs, rhs) = make_add_inputs(n, density);
        let sum = add_csr(&lhs, &rhs).expect("add csr");
        write_csr(&mut output, &format!("random-{n}-{density}"), &sum);
    }
    let (lhs, rhs) = cancellation_inputs();
    let sum = add_csr(&lhs, &rhs).expect("add csr cancellation");
    write_csr(&mut output, "cancellation", &sum);
    output
}

fn diags_golden_text() -> String {
    let mut output = String::new();

    let small = diags(
        &[
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
            vec![2.0; 6],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        &[-1, 0, 1],
        Some(Shape2D::new(6, 6)),
    )
    .expect("small tridiag");
    write_csr(&mut output, "diags-tridiag-6", &small);

    let rectangular = diags(
        &[vec![0.0, 3.0, -2.0], vec![4.0, 0.0]],
        &[1, -2],
        Some(Shape2D::new(4, 5)),
    )
    .expect("rectangular explicit-zero diags");
    write_csr(&mut output, "diags-rect-explicit-zero", &rectangular);

    let n = 10_000usize;
    let sub = vec![-1.0; n - 1];
    let main = vec![2.0; n];
    let sup = vec![-1.0; n - 1];
    let large =
        diags(&[sub, main, sup], &[-1, 0, 1], Some(Shape2D::new(n, n))).expect("large tridiag");
    write_csr(&mut output, "diags-tridiag-10000", &large);

    output
}

fn coo_csr_golden_text() -> String {
    let mut output = String::new();

    let duplicate = CooMatrix::from_triplets(
        Shape2D::new(4, 5),
        vec![7.0, 1.5, 0.0, -2.0, 3.25, -7.0, 2.0],
        vec![3, 0, 2, 0, 2, 3, 0],
        vec![1, 4, 2, 1, 2, 1, 1],
        false,
    )
    .expect("duplicate coo");
    write_csr(
        &mut output,
        "coo-csr-unsorted-duplicates",
        &duplicate.to_csr().expect("duplicate csr"),
    );

    let rectangular = CooMatrix::from_triplets(
        Shape2D::new(3, 6),
        vec![0.0, -4.0, 9.0, 1.25, -1.25, 5.5],
        vec![2, 1, 0, 1, 1, 2],
        vec![5, 2, 0, 4, 4, 1],
        false,
    )
    .expect("rectangular coo");
    write_csr(
        &mut output,
        "coo-csr-rect-explicit-zero",
        &rectangular.to_csr().expect("rectangular csr"),
    );

    let seeded = random(Shape2D::new(32, 32), 0.08, SEED)
        .expect("seeded coo")
        .to_csr()
        .expect("seeded csr");
    write_csr(&mut output, "coo-csr-seeded-32", &seeded);

    output
}

fn scale_csr_golden_text() -> String {
    let mut output = String::new();

    let canonical = CooMatrix::from_triplets(
        Shape2D::new(4, 5),
        vec![1.0, -2.0, -0.0, 3.5, 0.0, 9.25],
        vec![0, 1, 1, 2, 3, 3],
        vec![4, 0, 3, 2, 1, 4],
        false,
    )
    .expect("canonical scale coo")
    .to_csr()
    .expect("canonical scale csr");
    write_csr(
        &mut output,
        "scale-csr-canonical-neg",
        &scale_csr(&canonical, -2.5).expect("scale canonical"),
    );
    write_csr(
        &mut output,
        "scale-csr-canonical-zero-alpha",
        &scale_csr(&canonical, 0.0).expect("scale canonical zero"),
    );

    let unsorted = CsrMatrix::from_components(
        Shape2D::new(3, 4),
        vec![1.0, -0.0, 4.5, -3.0, 2.0],
        vec![3, 1, 1, 0, 0],
        vec![0, 2, 4, 5],
        false,
    )
    .expect("valid unsorted csr");
    write_csr(
        &mut output,
        "scale-csr-unsorted-preserve-meta",
        &scale_csr(&unsorted, -1.5).expect("scale unsorted"),
    );

    let duplicate = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![2.0, -2.0, 5.0],
        vec![1, 1, 2],
        vec![0, 2, 3],
        false,
    )
    .expect("valid duplicate csr");
    write_csr(
        &mut output,
        "scale-csr-duplicates-preserve-meta",
        &scale_csr(&duplicate, 3.0).expect("scale duplicate"),
    );

    output
}

fn make_spilu_banded_csc(n: usize, half_bandwidth: usize) -> CscMatrix {
    let entries_per_row = half_bandwidth.saturating_mul(2).saturating_add(1);
    let mut data = Vec::with_capacity(n.saturating_mul(entries_per_row));
    let mut rows = Vec::with_capacity(data.capacity());
    let mut cols = Vec::with_capacity(data.capacity());

    for row in 0..n {
        let start = row.saturating_sub(half_bandwidth);
        let end = row.saturating_add(half_bandwidth).min(n.saturating_sub(1));
        for col in start..=end {
            rows.push(row);
            cols.push(col);
            if row == col {
                data.push(entries_per_row as f64 + 2.0 + (row % 17) as f64 * 0.001);
            } else {
                data.push(-1.0 / (row.abs_diff(col) + 1) as f64);
            }
        }
    }

    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("spilu banded coo")
        .to_csc()
        .expect("spilu banded csc")
}

fn spilu_rhs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i % 23) as f64 - 11.0) * 0.125 + 1.0)
        .collect()
}

fn spilu_golden_text() -> String {
    let mut output = String::new();
    for &(n, half_bandwidth) in &[(16usize, 3usize), (64, 5), (160, 7)] {
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        let ilu = spilu(&matrix, IluOptions::default()).expect("spilu golden");
        let solution = ilu.solve(&spilu_rhs(n)).expect("spilu solve");
        write!(
            output,
            "case=banded-{n}-{half_bandwidth} shape={}x{} solution=",
            ilu.shape.0, ilu.shape.1
        )
        .expect("write spilu golden header");
        for value in solution {
            write!(output, "{:016x},", value.to_bits()).expect("write spilu solve bits");
        }
        output.push('\n');
    }
    output
}

fn write_or_print_golden(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output.as_bytes()).expect("write golden artifact");
    }
    print!("{output}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("add-csr");
    if mode == "transpose-view-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let rows = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(262_144);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(24);
            if let Err(error) = expm_bench::run_transpose_view_vs_scipy(rows, rounds, args.get(4)) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("transpose-view comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "laplacian-torus-candidate-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let side = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(96);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(24);
            if let Err(error) =
                expm_bench::run_laplacian_torus_candidate_vs_scipy(side, rounds, args.get(4))
            {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("torus Laplacian comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "csc-add-current-profile" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(4_096);
            let repetitions = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
            if let Err(error) = expm_bench::run_csc_add_current_profile(n, repetitions) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("CSC-add profiling requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "csc-add-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(4_096);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(24);
            if let Err(error) = expm_bench::run_csc_add_vs_scipy(n, rounds, args.get(4)) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("CSC-add comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "laplacian-cycle-current-profile" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(6_144);
            let repetitions = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
            if let Err(error) = expm_bench::run_laplacian_cycle_current_profile(n, repetitions) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("cycle Laplacian profiling requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "laplacian-cycle-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(6_144);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(22);
            if let Err(error) = expm_bench::run_laplacian_cycle_vs_scipy(n, rounds, args.get(4)) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("cycle Laplacian comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "laplacian-current-profile" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(4_096);
            let repetitions = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
            if let Err(error) = expm_bench::run_laplacian_current_profile(n, repetitions) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("laplacian profiling requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "laplacian-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(4_096);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(21);
            if let Err(error) = expm_bench::run_laplacian_vs_scipy(n, rounds, args.get(4)) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("laplacian comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "expm-current-profile" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(384);
            let repetitions = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1);
            if let Err(error) = expm_bench::run_current_profile(n, repetitions) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("expm profiling requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "expm-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(384);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(21);
            if let Err(error) = expm_bench::run_vs_scipy(n, rounds, args.get(4)) {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("expm comparison requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }
    if mode == "add-csr-golden" {
        write_or_print_golden(add_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "diags-golden" {
        write_or_print_golden(diags_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "coo-csr-golden" {
        write_or_print_golden(coo_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "scale-csr-golden" {
        write_or_print_golden(scale_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "spilu-golden" {
        write_or_print_golden(spilu_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "spilu" {
        let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_024);
        let half_bandwidth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
        let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        let rhs = spilu_rhs(n);

        let t0 = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..repeats {
            let ilu = spilu(black_box(&matrix), IluOptions::default()).expect("spilu");
            checksum += ilu.shape.0 as f64 + ilu.shape.1 as f64;
            checksum += ilu.solve(black_box(&rhs)).expect("spilu solve")[n / 2];
            black_box(&ilu);
        }
        let elapsed = t0.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1e3;
        let per_call_ms = total_ms / repeats as f64;
        println!(
            "{{\"mode\":\"{mode}\",\"n\":{n},\"half_bandwidth\":{half_bandwidth},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
        );
        return;
    }
    if mode != "add-csr" {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }

    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let density: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let (lhs, rhs) = make_add_inputs(n, density);

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        let sum = add_csr(black_box(&lhs), black_box(&rhs)).expect("add csr");
        checksum += sum.data().iter().sum::<f64>() + sum.nnz() as f64;
        black_box(&sum);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_ms = total_ms / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"density\":{density},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
    );
}
