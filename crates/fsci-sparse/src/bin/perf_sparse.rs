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
//!   `perf_sparse csc-add-current-profile <n> <repeats>`
//!   `perf_sparse csc-add-vs-scipy <n> <rounds> [oracle]`
//!   `perf_sparse csc-add-serial-gate-vs-scipy <n> <rounds> [oracle]`

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
    use fsci_sparse::linalg::{ExpmOptions, expm, laplacian};
    use fsci_sparse::{
        CSC_COMBINE_FORCE_PARALLEL, CSC_COMBINE_LAST_WORKERS, CscMatrix, CsrMatrix, Shape2D,
        add_csc,
    };
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    const REGISTERED_N: usize = 384;
    const REGISTERED_ROUNDS: usize = 21;
    const LAPLACIAN_REGISTERED_N: usize = 4_096;
    const LAPLACIAN_RESULT_NNZ: usize = 3 * LAPLACIAN_REGISTERED_N - 2;
    const CYCLE_REGISTERED_N: usize = 6_144;
    const CYCLE_REGISTERED_ROUNDS: usize = 22;
    const CYCLE_RESULT_NNZ: usize = 3 * CYCLE_REGISTERED_N;
    const CSC_ADD_REGISTERED_N: usize = 4_096;
    const CSC_ADD_ENTRIES_PER_COLUMN: usize = 24;
    const CSC_ADD_REGISTERED_ROUNDS: usize = 24;
    const CSC_ADD_COMPLETION_N: usize = 3_072;
    const CSC_ADD_COMPLETION_ENTRIES_PER_COLUMN: usize = 32;
    const CSC_ADD_COMPLETION_RESULT_NNZ: usize = 150_528;
    const CSC_ADD_COMPLETION_ROUNDS: usize = 24;
    const MIN_SAMPLE_SECONDS: f64 = 0.005;
    const CYCLE_MIN_SAMPLE_SECONDS: f64 = 0.050;
    const CSC_ADD_MIN_SAMPLE_SECONDS: f64 = 0.020;

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

    fn csc_add_completion_operand(n: usize, side: usize) -> CscMatrix {
        let nnz = n * CSC_ADD_COMPLETION_ENTRIES_PER_COLUMN;
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for column in 0..n {
            let mut entries = (0..CSC_ADD_COMPLETION_ENTRIES_PER_COLUMN)
                .map(|slot| {
                    let row = (211 * slot + 29 * column + 515 * side) % n;
                    let numerator = 1 + (5 * column + 7 * slot + 13 * side) % 47;
                    (row, numerator as f64 / 64.0)
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
            .expect("canonical CSC-add completion operand")
    }

    fn csc_add_completion_fixture(n: usize) -> (CscMatrix, CscMatrix) {
        (
            csc_add_completion_operand(n, 0),
            csc_add_completion_operand(n, 1),
        )
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
            write!(self.stdin, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    write!(self.stdin, ",")
                        .map_err(|error| format!("write {label} separator: {error}"))?;
                }
                write!(self.stdin, "{value}")
                    .map_err(|error| format!("write {label} value: {error}"))?;
            }
            writeln!(self.stdin).map_err(|error| format!("finish {label}: {error}"))
        }

        fn write_f64_vector(&mut self, label: &str, values: &[f64]) -> Result<(), String> {
            write!(self.stdin, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    write!(self.stdin, ",")
                        .map_err(|error| format!("write {label} separator: {error}"))?;
                }
                write!(self.stdin, "{value:.17e}")
                    .map_err(|error| format!("write {label} value: {error}"))?;
            }
            writeln!(self.stdin).map_err(|error| format!("finish {label}: {error}"))
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
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = laplacian(black_box(matrix), false)
                .expect("FrankenSciPy unnormalized sparse laplacian");
            black_box(result);
        }
        started.elapsed().as_secs_f64()
    }

    fn time_current_csc_add(lhs: &CscMatrix, rhs: &CscMatrix, repetitions: usize) -> f64 {
        time_current_csc_add_route(lhs, rhs, repetitions, false).0
    }

    fn time_current_csc_add_route(
        lhs: &CscMatrix,
        rhs: &CscMatrix,
        repetitions: usize,
        force_parallel: bool,
    ) -> (f64, usize) {
        CSC_COMBINE_FORCE_PARALLEL.store(force_parallel, Ordering::Relaxed);
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = add_csc(black_box(lhs), black_box(rhs))
                .expect("FrankenSciPy canonical CSC addition");
            black_box(result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let workers = CSC_COMBINE_LAST_WORKERS.load(Ordering::Relaxed);
        CSC_COMBINE_FORCE_PARALLEL.store(false, Ordering::Relaxed);
        (elapsed, workers)
    }

    fn four_call_geometric_null<F>(mut measure: F) -> Result<f64, String>
    where
        F: FnMut() -> Result<f64, String>,
    {
        let left_first = measure()?;
        let right_first = measure()?;
        let right_second = measure()?;
        let left_second = measure()?;
        Ok(((left_first / right_first) * (left_second / right_second)).sqrt())
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
        if current.len() != n || current.iter().any(|row| row.len() != n) {
            return Err("FrankenSciPy returned the wrong dense Laplacian shape".to_string());
        }
        if live_indptr.len() != n + 1
            || live_indices.len() != expected_nnz
            || live_data.len() != expected_nnz
            || live_indptr[n] != expected_nnz
        {
            return Err("live SciPy returned malformed sparse Laplacian arrays".to_string());
        }
        let mut degrees = vec![0.0_f64; n];
        for row in 0..n {
            degrees[row] = matrix.data()[matrix.indptr()[row]..matrix.indptr()[row + 1]]
                .iter()
                .map(|value| value.abs())
                .sum();
        }
        let mut outside_pattern_max = 0.0_f64;
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
            let mut structural_offset = 0usize;
            for column in 0..n {
                let expected = if column == row {
                    1.0
                } else if column.abs_diff(row) == 1 {
                    let edge = column.min(row);
                    let weight = 1.0 + (edge % 17) as f64 / 32.0;
                    -weight / (degrees[row] * degrees[column]).sqrt()
                } else {
                    0.0
                };
                let current_value = current[row][column];
                if !current_value.is_finite() {
                    return Err("FrankenSciPy returned a non-finite Laplacian".to_string());
                }
                if expected == 0.0 {
                    outside_pattern_max = outside_pattern_max.max(current_value.abs());
                    continue;
                }
                let live_value = live_data[start + structural_offset];
                structural_offset += 1;
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
            if structural_offset != end - start {
                return Err(format!("live CSR row {row} has an unexpected entry count"));
            }
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if outside_pattern_max != 0.0 || relative_l2 > 1.0e-14 {
            return Err(format!(
                "Laplacian parity thresholds failed: outside_pattern_max={outside_pattern_max:e} \
                 relative_l2={relative_l2:e}"
            ));
        }
        Ok((outside_pattern_max, relative_l2))
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
        if current.len() != n || current.iter().any(|row| row.len() != n) {
            return Err("FrankenSciPy returned the wrong dense cycle shape".to_string());
        }
        if live_indptr.len() != n + 1
            || live_indices.len() != expected_nnz
            || live_data.len() != expected_nnz
            || live_indptr[n] != expected_nnz
        {
            return Err("live SciPy returned malformed cycle Laplacian arrays".to_string());
        }
        let mut degrees = vec![0.0_f64; n];
        for row in 0..n {
            degrees[row] = matrix.data()[matrix.indptr()[row]..matrix.indptr()[row + 1]]
                .iter()
                .map(|value| value.abs())
                .sum();
        }
        let mut outside_pattern_max = 0.0_f64;
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
            let mut structural_offset = 0usize;
            for column in 0..n {
                let expected = if column == row {
                    degrees[row]
                } else if column == (row + 1) % n {
                    -(1.0 + (row % 29) as f64 / 64.0)
                } else if column == (row + n - 1) % n {
                    -(1.0 + (column % 29) as f64 / 64.0)
                } else {
                    0.0
                };
                let current_value = current[row][column];
                if !current_value.is_finite() {
                    return Err("FrankenSciPy returned a non-finite cycle Laplacian".to_string());
                }
                if expected == 0.0 {
                    outside_pattern_max = outside_pattern_max.max(current_value.abs());
                    continue;
                }
                let live_value = live_data[start + structural_offset];
                structural_offset += 1;
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
            if structural_offset != end - start {
                return Err(format!(
                    "live cycle CSR row {row} has an unexpected entry count"
                ));
            }
        }
        let relative_l2 = difference_squared.sqrt() / live_squared.sqrt().max(f64::EPSILON);
        if outside_pattern_max != 0.0 || relative_l2 > 1.0e-15 {
            return Err(format!(
                "cycle parity thresholds failed: outside_pattern_max={outside_pattern_max:e} \
                 relative_l2={relative_l2:e}"
            ));
        }
        Ok((outside_pattern_max, relative_l2))
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

    pub fn run_csc_add_serial_gate_vs_scipy(
        n: usize,
        rounds: usize,
        explicit_oracle: Option<&String>,
    ) -> Result<(), String> {
        if n != CSC_ADD_COMPLETION_N || rounds != CSC_ADD_COMPLETION_ROUNDS {
            return Err(format!(
                "one-shot completion requires n={CSC_ADD_COMPLETION_N} \
                 rounds={CSC_ADD_COMPLETION_ROUNDS}"
            ));
        }
        let available_threads = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        if available_threads != 32 {
            return Err(format!(
                "one-shot completion requires exactly 32 available CPUs, observed {available_threads}"
            ));
        }

        let (lhs, rhs) = csc_add_completion_fixture(n);
        let input_sha256 = csc_pair_input_sha256(&lhs, &rhs)?;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "fixture=deterministic-canonical-csc-serial-gate n={n} \
             entries_per_column={} lhs_nnz={} rhs_nnz={} \
             row=(211*j+29*c+515*side)%n \
             value=(1+((5*c+7*j+13*side)%47))/64 expected_result_nnz={} \
             rounds={rounds} construction_outside_timing=true \
             serialization_outside_timing=true requested_threads={available_threads} \
             requested_live_threads=1 \
             null_design=four-call-forward-reverse-geometric-symmetrization",
            CSC_ADD_COMPLETION_ENTRIES_PER_COLUMN,
            lhs.nnz(),
            rhs.nnz(),
            CSC_ADD_COMPLETION_RESULT_NNZ
        );

        CSC_COMBINE_FORCE_PARALLEL.store(false, Ordering::Relaxed);
        let candidate = add_csc(&lhs, &rhs)
            .map_err(|error| format!("FrankenSciPy candidate CSC addition: {error}"))?;
        let candidate_workers = CSC_COMBINE_LAST_WORKERS.load(Ordering::Relaxed);
        CSC_COMBINE_FORCE_PARALLEL.store(true, Ordering::Relaxed);
        let control = add_csc(&lhs, &rhs)
            .map_err(|error| format!("FrankenSciPy forced-old CSC addition: {error}"))?;
        let control_workers = CSC_COMBINE_LAST_WORKERS.load(Ordering::Relaxed);
        CSC_COMBINE_FORCE_PARALLEL.store(false, Ordering::Relaxed);
        let candidate_meta = candidate.canonical_meta();
        let control_meta = control.canonical_meta();
        let exact_control_match = candidate.indptr() == control.indptr()
            && candidate.indices() == control.indices()
            && candidate
                .data()
                .iter()
                .zip(control.data())
                .all(|(left, right)| left.to_bits() == right.to_bits())
            && candidate_meta == control_meta;
        if candidate.nnz() != CSC_ADD_COMPLETION_RESULT_NNZ
            || control.nnz() != CSC_ADD_COMPLETION_RESULT_NNZ
            || !candidate_meta.sorted_indices
            || !candidate_meta.deduplicated
            || !control_meta.sorted_indices
            || !control_meta.deduplicated
            || !exact_control_match
            || candidate_workers != 1
            || control_workers != 12
        {
            return Err(format!(
                "candidate/control contract failed: candidate_nnz={} control_nnz={} \
                 candidate_workers={candidate_workers} control_workers={control_workers} \
                 exact_control_match={exact_control_match}",
                candidate.nnz(),
                control.nnz()
            ));
        }
        println!(
            "same_elf_control: candidate_workers={candidate_workers} \
             forced_old_workers={control_workers} candidate_nnz={} control_nnz={} \
             indptr_exact=true indices_exact=true data_bits_exact=true metadata_exact=true",
            candidate.nnz(),
            control.nnz()
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
        let expected_result = format!(
            "RESULT rows={n} cols={n} nnz={CSC_ADD_COMPLETION_RESULT_NNZ} \
             sorted=True canonical=True"
        );
        if live_result != expected_result
            || live_indptr.len() != n + 1
            || live_indices.len() != CSC_ADD_COMPLETION_RESULT_NNZ
            || live_data.len() != CSC_ADD_COMPLETION_RESULT_NNZ
        {
            return Err(format!(
                "live SciPy CSC-add result contract failed: {live_result}"
            ));
        }
        let (current_nnz, max_abs_difference, relative_l2) =
            validate_current_csc_add(&lhs, &rhs, &live_indptr, &live_indices, &live_data)?;
        println!(
            "agreement: structural_components={current_nnz}/{} \
             max_abs_difference={max_abs_difference:.3e} relative_l2={relative_l2:.3e} \
             candidate_result_format=csc control_result_format=csc live_result_format=csc \
             tolerance=4*EPSILON*max(1,abs(live)) pass=true",
            CSC_ADD_COMPLETION_RESULT_NNZ
        );

        let mut candidate_repetitions = 1usize;
        while time_current_csc_add_route(&lhs, &rhs, candidate_repetitions, false).0
            < CSC_ADD_MIN_SAMPLE_SECONDS
        {
            candidate_repetitions = candidate_repetitions
                .checked_mul(2)
                .ok_or_else(|| "candidate CSC-add calibration overflowed".to_string())?;
        }
        let mut control_repetitions = 1usize;
        while time_current_csc_add_route(&lhs, &rhs, control_repetitions, true).0
            < CSC_ADD_MIN_SAMPLE_SECONDS
        {
            control_repetitions = control_repetitions
                .checked_mul(2)
                .ok_or_else(|| "forced-old CSC-add calibration overflowed".to_string())?;
        }
        let mut live_repetitions = 1usize;
        while scipy.solve(live_repetitions, CSC_ADD_COMPLETION_RESULT_NNZ)?
            < CSC_ADD_MIN_SAMPLE_SECONDS
        {
            live_repetitions = live_repetitions
                .checked_mul(2)
                .ok_or_else(|| "live CSC-add calibration overflowed".to_string())?;
        }
        println!(
            "calibration: candidate_repetitions={candidate_repetitions} \
             forced_old_repetitions={control_repetitions} \
             live_repetitions={live_repetitions} min_sample_ms={} \
             whole_public_calls=true separate_per_arm_repetitions=true",
            CSC_ADD_MIN_SAMPLE_SECONDS * 1_000.0
        );

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
                    0 => {
                        let (elapsed, workers) =
                            time_current_csc_add_route(&lhs, &rhs, candidate_repetitions, false);
                        if workers != 1 {
                            return Err(format!(
                                "candidate timing selected {workers} workers instead of 1"
                            ));
                        }
                        candidate_batch = elapsed;
                    }
                    1 => {
                        let (elapsed, workers) =
                            time_current_csc_add_route(&lhs, &rhs, control_repetitions, true);
                        if workers != 12 {
                            return Err(format!(
                                "forced-old timing selected {workers} workers instead of 12"
                            ));
                        }
                        control_batch = elapsed;
                    }
                    2 => {
                        live_batch =
                            scipy.solve(live_repetitions, CSC_ADD_COMPLETION_RESULT_NNZ)?;
                    }
                    _ => unreachable!(),
                }
            }

            let candidate_null = four_call_geometric_null(|| {
                let (elapsed, workers) =
                    time_current_csc_add_route(&lhs, &rhs, candidate_repetitions, false);
                if workers != 1 {
                    return Err(format!("candidate null selected {workers} workers"));
                }
                Ok(elapsed)
            })?;
            let control_null = four_call_geometric_null(|| {
                let (elapsed, workers) =
                    time_current_csc_add_route(&lhs, &rhs, control_repetitions, true);
                if workers != 12 {
                    return Err(format!("forced-old null selected {workers} workers"));
                }
                Ok(elapsed)
            })?;
            let live_null = four_call_geometric_null(|| {
                scipy.solve(live_repetitions, CSC_ADD_COMPLETION_RESULT_NNZ)
            })?;

            let candidate = candidate_batch / candidate_repetitions as f64;
            let control = control_batch / control_repetitions as f64;
            let live = live_batch / live_repetitions as f64;
            candidate_times.push(candidate);
            control_times.push(control);
            live_times.push(live);
            control_over_candidate.push(control / candidate);
            live_over_candidate.push(live / candidate);
            candidate_nulls.push(candidate_null);
            control_nulls.push(control_null);
            live_nulls.push(live_null);
        }
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
        let tails_pass = candidate_p95 < live_p95 && candidate_p99 < live_p99;
        let keep = control_ratio_low > 1.50
            && live_ratio_low > 1.05
            && tails_pass
            && null_medians_ok
            && control_clears_null
            && live_clears_null;

        println!(
            "timing: candidate_p50_ms={:.6} candidate_p95_ms={:.6} candidate_p99_ms={:.6} \
             forced_old_p50_ms={:.6} forced_old_p95_ms={:.6} forced_old_p99_ms={:.6} \
             live_scipy_p50_ms={:.6} live_scipy_p95_ms={:.6} live_scipy_p99_ms={:.6}",
            candidate_p50 * 1_000.0,
            candidate_p95 * 1_000.0,
            candidate_p99 * 1_000.0,
            control_p50 * 1_000.0,
            control_p95 * 1_000.0,
            control_p99 * 1_000.0,
            live_p50 * 1_000.0,
            live_p95 * 1_000.0,
            live_p99 * 1_000.0
        );
        println!(
            "ratios: forced_old_over_candidate={control_ratio_median:.6} \
             forced_old_over_candidate_ci95=[{control_ratio_low:.6},{control_ratio_high:.6}] \
             live_scipy_over_candidate={live_ratio_median:.6} \
             live_scipy_over_candidate_ci95=[{live_ratio_low:.6},{live_ratio_high:.6}] \
             candidate_cv={:.6} forced_old_cv={:.6} live_cv={:.6} \
             forced_old_ratio_cv={:.6} live_ratio_cv={:.6}",
            cv(&candidate_times),
            cv(&control_times),
            cv(&live_times),
            cv(&control_over_candidate),
            cv(&live_over_candidate)
        );
        println!(
            "nulls: candidate_median={candidate_null_median:.6} \
             candidate_ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] \
             forced_old_median={control_null_median:.6} \
             forced_old_ci95=[{control_null_low:.6},{control_null_high:.6}] \
             live_median={live_null_median:.6} \
             live_ci95=[{live_null_low:.6},{live_null_high:.6}] \
             worst_null_edge={null_edge:.6} null_half_width={null_half_width:.6} \
             null_medians_within_2pct={null_medians_ok}"
        );
        println!(
            "registered_keep_gate: keep={keep} \
             forced_old_ci_low={control_ratio_low:.6} required_gt=1.500000 \
             live_ci_low={live_ratio_low:.6} required_gt=1.050000 \
             candidate_tails_below_live={tails_pass} \
             forced_old_effect_deviation={control_effect_deviation:.6} \
             live_effect_deviation={live_effect_deviation:.6} \
             forced_old_clears_2x_null={control_clears_null} \
             live_clears_2x_null={live_clears_null} \
             required_half_width_margin={half_width_margin:.6} \
             required_endpoint_margin={endpoint_margin:.6}"
        );
        println!("raw_candidate_seconds={candidate_times:?}");
        println!("raw_forced_old_seconds={control_times:?}");
        println!("raw_live_scipy_seconds={live_times:?}");
        println!("raw_forced_old_over_candidate={control_over_candidate:?}");
        println!("raw_live_scipy_over_candidate={live_over_candidate:?}");
        println!("raw_candidate_symmetrized_null={candidate_nulls:?}");
        println!("raw_forced_old_symmetrized_null={control_nulls:?}");
        println!("raw_live_symmetrized_null={live_nulls:?}");
        println!(
            "verdict={} (Agent Mail unavailable => PROVISIONAL_NON_EXCLUSIVE)",
            if keep { "KEEP" } else { "REVERT" }
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
    if mode == "csc-add-serial-gate-vs-scipy" {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            let n = args
                .get(2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(3_072);
            let rounds = args
                .get(3)
                .and_then(|value| value.parse().ok())
                .unwrap_or(24);
            if let Err(error) = expm_bench::run_csc_add_serial_gate_vs_scipy(n, rounds, args.get(4))
            {
                eprintln!("fatal: {error}");
                std::process::exit(2);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("CSC-add serial-gate comparison requires --features sparse-incumbent-bench");
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
