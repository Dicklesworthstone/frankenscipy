//! Sparse Krylov solvers versus a genuine live SciPy incumbent.
//!
//! Rust transmits the exact deterministic CSR and RHS once to a persistent Python
//! co-process. Construction, serialization, callback iteration counting, and parity
//! checks are outside timing. The timed arms are interleaved in one invocation with
//! independent A/A nulls, executable identities, full hardware/thread/frequency
//! provenance, and fail-closed host-wide quiescence checks.
//!
//! Run: `cargo run --profile release-perf --bin perf_sparse_vs_scipy \
//!       --features sparse-incumbent-bench -- \
//!       [side] [rounds] [cg|gmres|lgmres|bicg|cgs|bicgstab|lsqr|lsmr|qmr|qmr-batch|lgmres-batch]`

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::linalg::{
        ITERATIVE_BATCH_LAST_WORKERS, IterativeSolveOptions, LGMRES_BATCH_FORCE_SEQUENTIAL,
        LgmresOptions, QMR_BATCH_FORCE_SEQUENTIAL, bicg, bicgstab, cg, cgs, gmres, lgmres,
        lgmres_batch, lsmr, lsqr, qmr, qmr_batch,
    };
    use fsci_sparse::{CsrMatrix, Shape2D};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    const DIAGONAL: f64 = 4.001;
    const WEST: f64 = -1.2;
    const EAST: f64 = -0.8;
    const VERTICAL: f64 = -1.0;
    const RTOL: f64 = 1.0e-5;
    const MIN_SAMPLE_MS: f64 = 2.0;
    const QMR_BATCH_SIZE: usize = 32;
    const QMR_BATCH_SIDE: usize = 48;
    const LGMRES_BATCH_SIDE: usize = 64;
    const QMR_BATCH_ROUNDS: usize = 24;
    const QMR_BATCH_MIN_SAMPLE_MS: f64 = 20.0;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(300);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    /// Corrected-gate clause 3: each A/A null median must sit within this
    /// fraction of 1.0. Bounds arm-order bias without coupling the verdict to
    /// the null's precision.
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;

    #[derive(Clone, Copy)]
    enum Method {
        Cg,
        Gmres,
        Lgmres,
        Bicg,
        Cgs,
        Bicgstab,
        Lsqr,
        Lsmr,
        Qmr,
    }

    #[derive(Clone, Copy)]
    enum IterativeBatchMethod {
        Qmr,
        Lgmres,
    }

    impl IterativeBatchMethod {
        const fn label(self) -> &'static str {
            match self {
                Self::Qmr => "qmr",
                Self::Lgmres => "lgmres",
            }
        }

        const fn batch_label(self) -> &'static str {
            match self {
                Self::Qmr => "qmr-batch",
                Self::Lgmres => "lgmres-batch",
            }
        }

        const fn method(self) -> Method {
            match self {
                Self::Qmr => Method::Qmr,
                Self::Lgmres => Method::Lgmres,
            }
        }

        const fn registered_side(self) -> usize {
            match self {
                Self::Qmr => QMR_BATCH_SIDE,
                Self::Lgmres => LGMRES_BATCH_SIDE,
            }
        }
    }

    impl Method {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "cg" => Ok(Self::Cg),
                "gmres" => Ok(Self::Gmres),
                "lgmres" => Ok(Self::Lgmres),
                "bicg" => Ok(Self::Bicg),
                "cgs" => Ok(Self::Cgs),
                "bicgstab" => Ok(Self::Bicgstab),
                "lsqr" => Ok(Self::Lsqr),
                "lsmr" => Ok(Self::Lsmr),
                "qmr" => Ok(Self::Qmr),
                _ => Err(format!(
                    "unknown method {value:?}; expected cg, gmres, lgmres, bicg, cgs, \
                     bicgstab, lsqr, lsmr, or qmr"
                )),
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::Cg => "cg",
                Self::Gmres => "gmres",
                Self::Lgmres => "lgmres",
                Self::Bicg => "bicg",
                Self::Cgs => "cgs",
                Self::Bicgstab => "bicgstab",
                Self::Lsqr => "lsqr",
                Self::Lsmr => "lsmr",
                Self::Qmr => "qmr",
            }
        }

        /// SciPy's success code is method-dependent. The `_isolve` solvers
        /// return `info == 0`, but `lsqr` returns `istop`, where 1 means
        /// "Ax - b is small enough" and 2 means the least-squares solution is
        /// good enough. With `atol=0` and `conlim=0` we expect `istop == 1`.
        /// Single definition so the parity path and the timed path cannot drift.
        const fn scipy_status_is_converged(self, status: i32) -> bool {
            match self {
                Self::Lsqr | Self::Lsmr => status == 1 || status == 2,
                Self::Cg
                | Self::Gmres
                | Self::Lgmres
                | Self::Bicg
                | Self::Cgs
                | Self::Bicgstab
                | Self::Qmr => status == 0,
            }
        }
    }

    /// Five-point convection-diffusion stencil. The horizontal coefficients differ
    /// by direction, making the matrix genuinely non-symmetric while preserving
    /// strict diagonal dominance for a stable, deterministic general-solver fixture.
    fn convection_diffusion_2d(side: usize) -> CsrMatrix {
        let n = side * side;
        let expected_nnz = 5 * n - 4 * side;
        let mut data = Vec::with_capacity(expected_nnz);
        let mut indices = Vec::with_capacity(expected_nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..side {
            for column in 0..side {
                let index = row * side + column;
                if row > 0 {
                    indices.push(index - side);
                    data.push(VERTICAL);
                }
                if column > 0 {
                    indices.push(index - 1);
                    data.push(WEST);
                }
                indices.push(index);
                data.push(DIAGONAL);
                if column + 1 < side {
                    indices.push(index + 1);
                    data.push(EAST);
                }
                if row + 1 < side {
                    indices.push(index + side);
                    data.push(VERTICAL);
                }
                indptr.push(data.len());
            }
        }
        assert_eq!(data.len(), expected_nnz);
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical non-symmetric CSR")
    }

    fn dirichlet_laplacian_2d(side: usize) -> CsrMatrix {
        let n = side * side;
        let expected_nnz = 5 * n - 4 * side;
        let mut data = Vec::with_capacity(expected_nnz);
        let mut indices = Vec::with_capacity(expected_nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..side {
            for column in 0..side {
                let index = row * side + column;
                if row > 0 {
                    indices.push(index - side);
                    data.push(VERTICAL);
                }
                if column > 0 {
                    indices.push(index - 1);
                    data.push(VERTICAL);
                }
                indices.push(index);
                data.push(DIAGONAL);
                if column + 1 < side {
                    indices.push(index + 1);
                    data.push(VERTICAL);
                }
                if row + 1 < side {
                    indices.push(index + side);
                    data.push(VERTICAL);
                }
                indptr.push(data.len());
            }
        }
        assert_eq!(data.len(), expected_nnz);
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical symmetric positive-definite CSR")
    }

    fn rhs(n: usize) -> Vec<f64> {
        (0..n)
            .map(|index| 1.0 + 0.01 * (index % 17) as f64)
            .collect()
    }

    fn canonical_input_sha256(matrix: &CsrMatrix, rhs: &[f64]) -> Result<String, String> {
        let mut digest = Sha256::new();
        for (label, value) in [
            ("row count", matrix.shape().rows),
            ("nonzero count", matrix.nnz()),
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
        for &value in rhs {
            digest.update(value.to_le_bytes());
        }
        let digest = digest.finalize();
        Ok(format!("{digest:x}"))
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        method: Method,
    }

    struct ScipyParity {
        info: i32,
        iterations: usize,
        residual: f64,
        solution: Vec<f64>,
    }

    struct ScipyTiming {
        elapsed: f64,
    }

    impl Scipy {
        fn start(script: &Path, method: Method) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--live")
                .arg(method.label())
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| format!("spawn python3: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy arm has no stdin".to_string())?;
            let mut stdout = BufReader::new(
                child
                    .stdout
                    .take()
                    .ok_or_else(|| "live SciPy arm has no stdout".to_string())?,
            );
            let mut ready = String::new();
            stdout
                .read_line(&mut ready)
                .map_err(|error| format!("read live SciPy identity: {error}"))?;
            if ready.is_empty() {
                return Err("live SciPy arm exited before reporting identity".to_string());
            }
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                    method,
                },
                ready.trim().to_string(),
            ))
        }

        fn read_reply(&mut self, context: &str) -> Result<String, String> {
            let mut output = String::new();
            self.stdout
                .read_line(&mut output)
                .map_err(|error| format!("read {context}: {error}"))?;
            if output.is_empty() {
                return Err(format!("live SciPy arm exited while reading {context}"));
            }
            Ok(output.trim().to_string())
        }

        fn write_usize_vector(&mut self, label: &str, values: &[usize]) -> Result<(), String> {
            write!(self.stdin, "{label} ")
                .map_err(|error| format!("write {label} marker: {error}"))?;
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
            write!(self.stdin, "{label} ")
                .map_err(|error| format!("write {label} marker: {error}"))?;
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

        fn initialize(
            &mut self,
            matrix: &CsrMatrix,
            rhs: &[f64],
            max_iter: usize,
        ) -> Result<String, String> {
            writeln!(
                self.stdin,
                "INIT {} {} {RTOL:.17e} {max_iter}",
                matrix.shape().rows,
                matrix.nnz()
            )
            .map_err(|error| format!("write INIT: {error}"))?;
            self.write_usize_vector("INDPTR", matrix.indptr())?;
            self.write_usize_vector("INDICES", matrix.indices())?;
            self.write_f64_vector("DATA", matrix.data())?;
            self.write_f64_vector("B", rhs)?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INIT: {error}"))?;
            self.read_reply("SciPy fixture identity")
        }

        fn input_sha256(&mut self) -> Result<String, String> {
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush INPUT_SHA256: {error}"))?;
            let reply = self.read_reply("SciPy input SHA-256")?;
            let digest = reply
                .strip_prefix("INPUT_SHA256 ")
                .ok_or_else(|| format!("invalid SciPy input SHA-256: {reply}"))?;
            if !is_sha256(digest) {
                return Err(format!("invalid SciPy input SHA-256: {reply}"));
            }
            Ok(digest.to_string())
        }

        fn parity(&mut self, expected_components: usize) -> Result<ScipyParity, String> {
            writeln!(self.stdin, "PARITY").map_err(|error| format!("write PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush PARITY: {error}"))?;
            let result = self.read_reply("SciPy parity header")?;
            if !result.starts_with("RESULT ") {
                return Err(format!("invalid SciPy parity header: {result}"));
            }
            let field = |name: &str| -> Result<&str, String> {
                result
                    .split_whitespace()
                    .find_map(|item| item.strip_prefix(&format!("{name}=")))
                    .ok_or_else(|| format!("missing {name} in {result}"))
            };
            let info = field("info")?
                .parse::<i32>()
                .map_err(|error| format!("parse SciPy info: {error}"))?;
            let iterations = field("iterations")?
                .parse::<usize>()
                .map_err(|error| format!("parse SciPy iterations: {error}"))?;
            let residual = field("residual")?
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy residual: {error}"))?;
            let components = field("components")?
                .parse::<usize>()
                .map_err(|error| format!("parse SciPy components: {error}"))?;
            let vector = self.read_reply("SciPy parity vector")?;
            let payload = vector
                .strip_prefix("X ")
                .ok_or_else(|| format!("invalid SciPy parity vector: {vector}"))?;
            let solution = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse SciPy solution component: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if components != expected_components || solution.len() != expected_components {
                return Err(format!(
                    "SciPy parity components {components}/{} != {expected_components}",
                    solution.len()
                ));
            }
            Ok(ScipyParity {
                info,
                iterations,
                residual,
                solution,
            })
        }

        fn solve(
            &mut self,
            repetitions: usize,
            expected_components: usize,
        ) -> Result<ScipyTiming, String> {
            writeln!(self.stdin, "SOLVE {repetitions}")
                .map_err(|error| format!("write SOLVE: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SOLVE: {error}"))?;
            let reply = self.read_reply("timed SciPy result")?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.len() != 6 || fields.first() != Some(&"TIME") {
                return Err(format!("invalid timed SciPy result: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy elapsed time: {error}"))?;
            let info = fields[2]
                .parse::<i32>()
                .map_err(|error| format!("parse SciPy info: {error}"))?;
            let components = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse SciPy component count: {error}"))?;
            let observed_threads = fields[4]
                .parse::<usize>()
                .map_err(|error| format!("parse SciPy observed threads: {error}"))?;
            let checksum = fields[5]
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy checksum: {error}"))?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || !self.method.scipy_status_is_converged(info)
                || components != expected_components
                || observed_threads != 1
            {
                return Err(format!("inadmissible timed SciPy result: {reply}"));
            }
            Ok(ScipyTiming { elapsed })
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return f64::NAN;
        }
        if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
        }
    }

    /// Deterministic percentile-bootstrap CI on the median — the campaign gate.
    fn boot_ci(v: &[f64]) -> (f64, f64) {
        if v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mut st = 0x6a09_e667_f3bc_c909u64;
        let mut meds = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut s = Vec::with_capacity(v.len());
            for _ in 0..v.len() {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                s.push(v[(st as usize) % v.len()]);
            }
            meds.push(median(s));
        }
        meds.sort_by(f64::total_cmp);
        (meds[250], meds[9_750])
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

    fn solve_ours(
        method: Method,
        matrix: &CsrMatrix,
        rhs: &[f64],
        max_iter: usize,
    ) -> fsci_sparse::IterativeSolveResult {
        let options = IterativeSolveOptions {
            mode: RuntimeMode::Strict,
            check_finite: true,
            tol: RTOL,
            max_iter: Some(max_iter),
        };
        match method {
            Method::Cg => cg(matrix, rhs, None, options),
            Method::Gmres => gmres(matrix, rhs, None, options),
            Method::Lgmres => lgmres(
                matrix,
                rhs,
                None,
                LgmresOptions {
                    tol: RTOL,
                    max_iter: Some(max_iter),
                    inner_m: 30,
                    outer_k: 3,
                },
            ),
            Method::Bicg => bicg(matrix, rhs, None, options),
            Method::Cgs => cgs(matrix, rhs, None, options),
            Method::Bicgstab => bicgstab(matrix, rhs, None, options),
            // lsqr minimizes ||Ax - b||_2 and takes no initial guess; its
            // stopping test is |phi_bar| / ||b|| < tol, which the SciPy arm
            // mirrors with atol=0, btol=tol, conlim=0.
            Method::Lsqr => lsqr(matrix, rhs, options),
            // The current public implementation delegates to LSQR. The live
            // arm runs genuine SciPy LSMR with its matching residual stop.
            Method::Lsmr => lsmr(matrix, rhs, options),
            // qmr solves the same square system as gmres/bicgstab. Its stopping
            // rule is ||b - Ax|| / ||b|| < tol on the recomputed true residual;
            // SciPy tests the recursively carried r against
            // atol = max(0, rtol*||b||), the same relative-residual criterion.
            Method::Qmr => qmr(matrix, rhs, None, options),
        }
        .expect("FrankenSciPy iterative solve")
    }

    fn time_ours(
        method: Method,
        matrix: &CsrMatrix,
        rhs: &[f64],
        max_iter: usize,
        repetitions: usize,
    ) -> f64 {
        let mut result = None;
        let started = Instant::now();
        for _ in 0..repetitions {
            result = Some(black_box(solve_ours(
                method,
                black_box(matrix),
                black_box(rhs),
                max_iter,
            )));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let result = result.expect("positive repetition count");
        assert!(result.converged, "timed FrankenSciPy solve must converge");
        let checksum = result
            .solution
            .iter()
            .fold(0u64, |state, value| state ^ value.to_bits());
        black_box(checksum);
        elapsed
    }

    fn calibrate_repetitions(
        scipy: &mut Scipy,
        method: Method,
        matrix: &CsrMatrix,
        rhs: &[f64],
        max_iter: usize,
    ) -> Result<usize, String> {
        let mut repetitions = 1usize;
        loop {
            let ours = time_ours(method, matrix, rhs, max_iter, repetitions);
            let incumbent = scipy.solve(repetitions, rhs.len())?;
            if ours * 1_000.0 >= MIN_SAMPLE_MS && incumbent.elapsed * 1_000.0 >= MIN_SAMPLE_MS {
                return Ok(repetitions);
            }
            repetitions = repetitions
                .checked_mul(2)
                .ok_or_else(|| "calibration repetition count overflowed".to_string())?;
        }
    }

    fn incumbent_pair(
        scipy: &mut Scipy,
        method: Method,
        matrix: &CsrMatrix,
        rhs: &[f64],
        max_iter: usize,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_ours(method, matrix, rhs, max_iter, repetitions),
                scipy.solve(repetitions, rhs.len())?.elapsed,
            ))
        } else {
            let incumbent = scipy.solve(repetitions, rhs.len())?.elapsed;
            let ours = time_ours(method, matrix, rhs, max_iter, repetitions);
            Ok((ours, incumbent))
        }
    }

    fn ours_null_pair(
        method: Method,
        matrix: &CsrMatrix,
        rhs: &[f64],
        max_iter: usize,
        repetitions: usize,
        round: usize,
    ) -> f64 {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_ours(method, matrix, rhs, max_iter, repetitions),
                time_ours(method, matrix, rhs, max_iter, repetitions),
            )
        } else {
            let right = time_ours(method, matrix, rhs, max_iter, repetitions);
            let left = time_ours(method, matrix, rhs, max_iter, repetitions);
            (left, right)
        };
        left / right
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        components: usize,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                scipy.solve(repetitions, components)?.elapsed,
                scipy.solve(repetitions, components)?.elapsed,
            )
        } else {
            let right = scipy.solve(repetitions, components)?.elapsed;
            let left = scipy.solve(repetitions, components)?.elapsed;
            (left, right)
        };
        Ok(left / right)
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

    /// Set once if any quiescence phase was waived via
    /// `FSCI_SPARSE_ALLOW_NON_EXCLUSIVE=1`. Once set, the run can never print a
    /// `DECIDED` verdict, so a provisional number cannot be laundered into a
    /// gated one by quoting the verdict line.
    static EXCLUSIVITY_WAIVED: AtomicBool = AtomicBool::new(false);

    fn non_exclusive_waiver_requested() -> bool {
        std::env::var_os("FSCI_SPARSE_ALLOW_NON_EXCLUSIVE").is_some_and(|value| value == "1")
    }

    fn require_host_wide_quiescence(phase: &str) -> Result<(), String> {
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
        if !busy_cpus.is_empty() {
            if !non_exclusive_waiver_requested() {
                return Err(format!(
                    "host-wide benchmark exclusivity failed during {phase}; CPUs above {:.1}% busy: {}",
                    HOST_QUIESCENCE_MAX_BUSY * 100.0,
                    busy_cpus.join(",")
                ));
            }
            // Explicitly waived. Record the contamination level in the artifact
            // itself rather than suppressing it, and permanently downgrade this
            // run's evidence class.
            EXCLUSIVITY_WAIVED.store(true, Ordering::SeqCst);
            println!(
                "host_wide_quiescence_{phase}=NOT_CERTIFIED \
                 evidence_class=PROVISIONAL_NON_EXCLUSIVE waiver=FSCI_SPARSE_ALLOW_NON_EXCLUSIVE \
                 sampled_cpus={} maximum_busy_fraction={maximum_busy_fraction:.3} \
                 busy_cpu_count_above_limit={} limit={HOST_QUIESCENCE_MAX_BUSY:.3} \
                 busy_cpus={}",
                before.len(),
                busy_cpus.len(),
                busy_cpus.join(",")
            );
            return Ok(());
        }
        println!(
            "host_wide_quiescence_{phase}=clear sampled_cpus={} \
             maximum_busy_fraction={maximum_busy_fraction:.3} \
             busy_cpu_count_above_limit=0 limit={HOST_QUIESCENCE_MAX_BUSY:.3}",
            before.len()
        );
        Ok(())
    }

    fn host_identity() -> Result<String, String> {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|hostname| hostname.trim().replace(char::is_whitespace, "_"))
            .map_err(|error| format!("read host identity: {error}"))
    }

    fn cpu_topology() -> Result<(usize, usize), String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let logical_threads = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        let mut physical_cores = HashSet::new();
        for block in cpuinfo.split("\n\n") {
            let physical_id = block.lines().find_map(|line| {
                line.strip_prefix("physical id")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            let core_id = block.lines().find_map(|line| {
                line.strip_prefix("core id")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            if let (Some(physical_id), Some(core_id)) = (physical_id, core_id) {
                physical_cores.insert((physical_id.to_string(), core_id.to_string()));
            }
        }
        if physical_cores.is_empty() || logical_threads == 0 {
            return Err("CPU topology is unavailable".to_string());
        }
        Ok((physical_cores.len(), logical_threads))
    }

    fn ram_bytes() -> Result<u64, String> {
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let kibibytes = meminfo
            .lines()
            .find_map(|line| {
                line.strip_prefix("MemTotal:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .ok_or_else(|| "MemTotal is unavailable".to_string())?;
        kibibytes
            .checked_mul(1_024)
            .ok_or_else(|| "RAM byte count overflowed".to_string())
    }

    fn numa_node_count() -> Result<usize, String> {
        let count = std::fs::read_dir("/sys/devices/system/node")
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
        if count == 0 {
            return Err("NUMA node count is unavailable".to_string());
        }
        Ok(count)
    }

    fn runtime_isa_features() -> String {
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

    fn observed_os_threads() -> Result<usize, String> {
        std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("observe process threads: {error}"))
            .map(Iterator::count)
    }

    fn observed_peak_worker_tasks<T>(work: impl FnOnce() -> T + Send) -> (T, usize)
    where
        T: Send,
    {
        let done = AtomicBool::new(false);
        thread::scope(|scope| {
            let watcher = scope.spawn(|| {
                let mut peak = 2usize;
                while !done.load(Ordering::Relaxed) {
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        peak = peak.max(entries.count());
                    }
                    thread::sleep(Duration::from_micros(200));
                }
                peak
            });
            let value = work();
            done.store(true, Ordering::Relaxed);
            let peak = watcher.join().unwrap_or(2).saturating_sub(2).max(1);
            (value, peak)
        })
    }

    fn cpu_affinity() -> String {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
                    .map(str::trim)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn affinity_cpus(affinity: &str) -> Result<Vec<usize>, String> {
        let mut cpus = Vec::new();
        for field in affinity.split(',') {
            if let Some((first, last)) = field.split_once('-') {
                let first = first
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity start {first}: {error}"))?;
                let last = last
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity end {last}: {error}"))?;
                if last < first {
                    return Err(format!("descending CPU affinity range {field}"));
                }
                cpus.extend(first..=last);
            } else {
                cpus.push(
                    field
                        .parse::<usize>()
                        .map_err(|error| format!("parse affinity CPU {field}: {error}"))?,
                );
            }
        }
        cpus.sort_unstable();
        cpus.dedup();
        if cpus.is_empty() {
            return Err("CPU affinity names no CPUs".to_string());
        }
        Ok(cpus)
    }

    fn read_policy_field(base: &Path, name: &str, required: bool) -> Result<String, String> {
        let path = base.join(name);
        match std::fs::read_to_string(&path) {
            Ok(value) => Ok(value.trim().replace(char::is_whitespace, "_")),
            Err(error) if !required => Ok(format!("unavailable({error})").replace(' ', "_")),
            Err(error) => Err(format!("read {}: {error}", path.display())),
        }
    }

    fn print_hardware_provenance(affinity: &str) -> Result<(), String> {
        let affinity_cpus = affinity_cpus(affinity)?;
        let cpu = affinity_cpus[0];
        let host = host_identity()?;
        let (physical_cores, logical_threads) = cpu_topology()?;
        let ram_bytes = ram_bytes()?;
        let numa_nodes = numa_node_count()?;
        let policy = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq"));
        let scaling_driver = read_policy_field(&policy, "scaling_driver", true)?;
        let scaling_governor = read_policy_field(&policy, "scaling_governor", true)?;
        let energy_performance_preference =
            read_policy_field(&policy, "energy_performance_preference", false)?;
        let scaling_min_freq_khz = read_policy_field(&policy, "scaling_min_freq", false)?;
        let scaling_max_freq_khz = read_policy_field(&policy, "scaling_max_freq", false)?;
        println!(
            "hardware_provenance: host_identity={host} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={ram_bytes} numa_nodes={numa_nodes} \
             runtime_detected_isa={} affinity={affinity} cpuset_logical_cap={}",
            runtime_isa_features(),
            affinity_cpus.len()
        );
        println!(
            "cpu_frequency_policy: cpu={cpu} scaling_driver={scaling_driver} \
             scaling_governor={scaling_governor} \
             energy_performance_preference={energy_performance_preference} \
             scaling_min_freq_khz={scaling_min_freq_khz} \
             scaling_max_freq_khz={scaling_max_freq_khz}"
        );
        Ok(())
    }

    fn ready_value<'a>(identity: &'a str, prefix: &str) -> Option<&'a str> {
        identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix(prefix))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn scipy_oracle_script(argument: Option<&String>) -> Result<PathBuf, String> {
        if let Some(path) = argument {
            let script = PathBuf::from(path);
            if script.is_file() {
                return Ok(script);
            }
            return Err(format!(
                "oracle argument is not a file: {}",
                script.display()
            ));
        }
        if let Some(path) = std::env::var_os("FSCI_SCIPY_SPARSE_ORACLE") {
            let script = PathBuf::from(path);
            if script.is_file() {
                return Ok(script);
            }
            return Err(format!(
                "FSCI_SCIPY_SPARSE_ORACLE is not a file: {}",
                script.display()
            ));
        }
        let compiled = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/scipy_sparse_arm.py");
        if compiled.is_file() {
            return Ok(compiled);
        }
        let workspace_relative = PathBuf::from("crates/fsci-sparse/python/scipy_sparse_arm.py");
        if workspace_relative.is_file() {
            return Ok(workspace_relative);
        }
        Err("SciPy sparse oracle is unavailable".to_string())
    }

    fn sha256_of_self() -> Result<String, String> {
        let executable =
            std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
        let bytes = std::fs::read(executable).map_err(|error| format!("read own ELF: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn qmr_batch_required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required provenance variable {name} is absent"))
    }

    fn qmr_batch_percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let index = ((values.len().saturating_sub(1)) as f64 * quantile).ceil() as usize;
        values[index.min(values.len().saturating_sub(1))]
    }

    fn qmr_batch_csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn require_iterative_batch_affinity(affinity: &str) -> Result<(), String> {
        let cpus = affinity_cpus(affinity)?;
        let expected = (0..QMR_BATCH_SIZE).collect::<Vec<_>>();
        if cpus != expected {
            return Err(format!(
                "registered iterative batch requires affinity 0-31, observed {affinity}"
            ));
        }
        let (physical_cores, logical_threads) = cpu_topology()?;
        if physical_cores != QMR_BATCH_SIZE || logical_threads != 2 * QMR_BATCH_SIZE {
            return Err(format!(
                "registered host requires 32 physical/64 logical CPUs, observed \
                 {physical_cores}/{logical_threads}"
            ));
        }
        let mut selected_cores = HashSet::new();
        for cpu in &cpus {
            let topology = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/topology"));
            let package = read_policy_field(&topology, "physical_package_id", true)?;
            let core = read_policy_field(&topology, "core_id", true)?;
            selected_cores.insert((package, core));
        }
        if selected_cores.len() != QMR_BATCH_SIZE {
            return Err(format!(
                "affinity selected {} unique physical cores; expected {QMR_BATCH_SIZE}",
                selected_cores.len()
            ));
        }
        let available = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        if available != QMR_BATCH_SIZE {
            return Err(format!(
                "available_parallelism={available}; expected {QMR_BATCH_SIZE}"
            ));
        }
        println!(
            "batch_topology: selected_logical_cpus={} selected_unique_physical_cores={} \
             smt_siblings_selected=0 available_parallelism={available}",
            cpus.len(),
            selected_cores.len()
        );
        Ok(())
    }

    fn qmr_batch_rhs(rhs: &[f64]) -> Vec<Vec<f64>> {
        vec![rhs.to_vec(); QMR_BATCH_SIZE]
    }

    fn solve_ours_iterative_batch(
        batch_method: IterativeBatchMethod,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
        force_sequential: bool,
    ) -> Result<Vec<fsci_sparse::IterativeSolveResult>, String> {
        let result = match batch_method {
            IterativeBatchMethod::Qmr => {
                QMR_BATCH_FORCE_SEQUENTIAL.store(force_sequential, Ordering::SeqCst);
                let result = qmr_batch(
                    matrix,
                    rhses,
                    None,
                    IterativeSolveOptions {
                        mode: RuntimeMode::Strict,
                        check_finite: true,
                        tol: RTOL,
                        max_iter: Some(max_iter),
                    },
                );
                QMR_BATCH_FORCE_SEQUENTIAL.store(false, Ordering::SeqCst);
                result
            }
            IterativeBatchMethod::Lgmres => {
                LGMRES_BATCH_FORCE_SEQUENTIAL.store(force_sequential, Ordering::SeqCst);
                let result = lgmres_batch(
                    matrix,
                    rhses,
                    None,
                    LgmresOptions {
                        tol: RTOL,
                        max_iter: Some(max_iter),
                        inner_m: 30,
                        outer_k: 3,
                    },
                );
                LGMRES_BATCH_FORCE_SEQUENTIAL.store(false, Ordering::SeqCst);
                result
            }
        };
        result.map_err(|error| {
            format!(
                "FrankenSciPy {} batch: {error}",
                batch_method.label().to_ascii_uppercase()
            )
        })
    }

    fn valid_iterative_batch(results: &[fsci_sparse::IterativeSolveResult]) -> bool {
        results.len() == QMR_BATCH_SIZE
            && results.iter().all(|result| {
                result.converged
                    && result.iterations > 0
                    && result.residual_norm.is_finite()
                    && result.residual_norm <= 1.25 * RTOL
                    && result.solution.iter().all(|value| value.is_finite())
            })
    }

    fn time_ours_iterative_batch(
        batch_method: IterativeBatchMethod,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
        repetitions: usize,
        force_sequential: bool,
    ) -> Result<f64, String> {
        let mut results = None;
        let started = Instant::now();
        for _ in 0..repetitions {
            results = Some(black_box(solve_ours_iterative_batch(
                batch_method,
                black_box(matrix),
                black_box(rhses),
                max_iter,
                force_sequential,
            )?));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let results =
            results.ok_or_else(|| "iterative batch repetition count was zero".to_string())?;
        if !valid_iterative_batch(&results) {
            return Err(format!(
                "timed {} did not converge exactly as registered",
                batch_method.batch_label()
            ));
        }
        let checksum = results
            .iter()
            .flat_map(|result| &result.solution)
            .fold(0u64, |state, value| state.rotate_left(7) ^ value.to_bits());
        black_box(checksum);
        Ok(elapsed)
    }

    fn calibrate_iterative_batch(
        batch_method: IterativeBatchMethod,
        scipy: &mut Scipy,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
    ) -> Result<usize, String> {
        let mut repetitions = 1usize;
        loop {
            let candidate = time_ours_iterative_batch(
                batch_method,
                matrix,
                rhses,
                max_iter,
                repetitions,
                false,
            )?;
            let sequential = time_ours_iterative_batch(
                batch_method,
                matrix,
                rhses,
                max_iter,
                repetitions,
                true,
            )?;
            let incumbent = scipy.solve(
                repetitions
                    .checked_mul(QMR_BATCH_SIZE)
                    .ok_or_else(|| "SciPy iterative batch repetition overflowed".to_string())?,
                matrix.shape().rows,
            )?;
            if candidate * 1_000.0 >= QMR_BATCH_MIN_SAMPLE_MS
                && sequential * 1_000.0 >= QMR_BATCH_MIN_SAMPLE_MS
                && incumbent.elapsed * 1_000.0 >= QMR_BATCH_MIN_SAMPLE_MS
            {
                return Ok(repetitions);
            }
            repetitions = repetitions.checked_mul(2).ok_or_else(|| {
                "iterative batch calibration repetition count overflowed".to_string()
            })?;
        }
    }

    #[derive(Clone, Copy)]
    enum QmrBatchArm {
        Candidate,
        Sequential,
        Scipy,
    }

    const QMR_BATCH_ORDERS: [[QmrBatchArm; 3]; 6] = [
        [
            QmrBatchArm::Candidate,
            QmrBatchArm::Sequential,
            QmrBatchArm::Scipy,
        ],
        [
            QmrBatchArm::Candidate,
            QmrBatchArm::Scipy,
            QmrBatchArm::Sequential,
        ],
        [
            QmrBatchArm::Sequential,
            QmrBatchArm::Candidate,
            QmrBatchArm::Scipy,
        ],
        [
            QmrBatchArm::Sequential,
            QmrBatchArm::Scipy,
            QmrBatchArm::Candidate,
        ],
        [
            QmrBatchArm::Scipy,
            QmrBatchArm::Candidate,
            QmrBatchArm::Sequential,
        ],
        [
            QmrBatchArm::Scipy,
            QmrBatchArm::Sequential,
            QmrBatchArm::Candidate,
        ],
    ];

    fn time_qmr_batch_arm(
        batch_method: IterativeBatchMethod,
        arm: QmrBatchArm,
        scipy: &mut Scipy,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
        repetitions: usize,
    ) -> Result<f64, String> {
        match arm {
            QmrBatchArm::Candidate => {
                time_ours_iterative_batch(batch_method, matrix, rhses, max_iter, repetitions, false)
            }
            QmrBatchArm::Sequential => {
                time_ours_iterative_batch(batch_method, matrix, rhses, max_iter, repetitions, true)
            }
            QmrBatchArm::Scipy => scipy
                .solve(
                    repetitions
                        .checked_mul(QMR_BATCH_SIZE)
                        .ok_or_else(|| "SciPy iterative batch repetition overflowed".to_string())?,
                    matrix.shape().rows,
                )
                .map(|timing| timing.elapsed),
        }
    }

    // Keep the complete frozen arm state explicit at this benchmark boundary.
    #[allow(clippy::too_many_arguments)]
    fn qmr_batch_null_pair(
        batch_method: IterativeBatchMethod,
        arm: QmrBatchArm,
        scipy: &mut Scipy,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_qmr_batch_arm(
                    batch_method,
                    arm,
                    scipy,
                    matrix,
                    rhses,
                    max_iter,
                    repetitions,
                )?,
                time_qmr_batch_arm(
                    batch_method,
                    arm,
                    scipy,
                    matrix,
                    rhses,
                    max_iter,
                    repetitions,
                )?,
            )
        } else {
            let right = time_qmr_batch_arm(
                batch_method,
                arm,
                scipy,
                matrix,
                rhses,
                max_iter,
                repetitions,
            )?;
            let left = time_qmr_batch_arm(
                batch_method,
                arm,
                scipy,
                matrix,
                rhses,
                max_iter,
                repetitions,
            )?;
            (left, right)
        };
        Ok(left / right)
    }

    struct QmrBatchMeasurement {
        candidate: Vec<f64>,
        sequential: Vec<f64>,
        scipy: Vec<f64>,
        maintenance_ratios: Vec<f64>,
        incumbent_ratios: Vec<f64>,
        candidate_nulls: Vec<f64>,
        sequential_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
    }

    fn measure_qmr_batch(
        batch_method: IterativeBatchMethod,
        scipy: &mut Scipy,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        max_iter: usize,
        rounds: usize,
        repetitions: usize,
    ) -> Result<QmrBatchMeasurement, String> {
        let mut measurement = QmrBatchMeasurement {
            candidate: Vec::with_capacity(rounds),
            sequential: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            maintenance_ratios: Vec::with_capacity(rounds),
            incumbent_ratios: Vec::with_capacity(rounds),
            candidate_nulls: Vec::with_capacity(rounds),
            sequential_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
        };
        for round in 0..rounds {
            let order = QMR_BATCH_ORDERS[round % QMR_BATCH_ORDERS.len()];
            let mut arm_times = [0.0f64; 3];
            for arm in order {
                let elapsed = time_qmr_batch_arm(
                    batch_method,
                    arm,
                    scipy,
                    matrix,
                    rhses,
                    max_iter,
                    repetitions,
                )?;
                match arm {
                    QmrBatchArm::Candidate => arm_times[0] = elapsed,
                    QmrBatchArm::Sequential => arm_times[1] = elapsed,
                    QmrBatchArm::Scipy => arm_times[2] = elapsed,
                }
            }
            let mut arm_nulls = [0.0f64; 3];
            for arm in order.into_iter().rev() {
                let null = qmr_batch_null_pair(
                    batch_method,
                    arm,
                    scipy,
                    matrix,
                    rhses,
                    max_iter,
                    repetitions,
                    round,
                )?;
                match arm {
                    QmrBatchArm::Candidate => arm_nulls[0] = null,
                    QmrBatchArm::Sequential => arm_nulls[1] = null,
                    QmrBatchArm::Scipy => arm_nulls[2] = null,
                }
            }
            measurement.candidate.push(arm_times[0]);
            measurement.sequential.push(arm_times[1]);
            measurement.scipy.push(arm_times[2]);
            measurement
                .maintenance_ratios
                .push(arm_times[1] / arm_times[0]);
            measurement
                .incumbent_ratios
                .push(arm_times[2] / arm_times[0]);
            measurement.candidate_nulls.push(arm_nulls[0]);
            measurement.sequential_nulls.push(arm_nulls[1]);
            measurement.scipy_nulls.push(arm_nulls[2]);
        }
        Ok(measurement)
    }

    struct QmrBatchGate {
        low: f64,
        high: f64,
        effect_pass: bool,
    }

    fn print_qmr_batch_gate(
        label: &str,
        ratios: &[f64],
        left_nulls: &[f64],
        right_nulls: &[f64],
    ) -> QmrBatchGate {
        let (low, high) = boot_ci(ratios);
        let (left_null_low, left_null_high) = boot_ci(left_nulls);
        let (right_null_low, right_null_high) = boot_ci(right_nulls);
        let left_null_median = median(left_nulls.to_vec());
        let right_null_median = median(right_nulls.to_vec());
        let null_half_width = ((left_null_high - left_null_low) / 2.0)
            .max((right_null_high - right_null_low) / 2.0)
            .max(0.0);
        let null_edge = left_null_high
            .max(right_null_high)
            .max(1.0 / left_null_low.max(1.0e-9))
            .max(1.0 / right_null_low.max(1.0e-9))
            .max(1.0);
        let c1 = low > 1.0;
        let effect_deviation = (low - 1.0).max(0.0);
        let c2 = effect_deviation > 2.0 * null_half_width;
        let c2b = effect_deviation > 2.0 * (null_edge - 1.0);
        let c3 = (left_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (right_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let effect_pass = c1 && c2 && c2b && c3;
        let waived = EXCLUSIVITY_WAIVED.load(Ordering::SeqCst);
        let outcome = if effect_pass && waived {
            "PROVISIONAL FRANKENSCIPY WIN (non-exclusive host; NOT DECIDED-class evidence)"
        } else if effect_pass {
            "DECIDED FRANKENSCIPY WIN"
        } else if waived {
            "PROVISIONAL INDETERMINATE (non-exclusive host; NOT DECIDED-class evidence)"
        } else {
            "NOT DECIDED"
        };
        println!(
            "{label}_corrected_null_gate: c1_effect_ci_above_one={c1} \
             c2_beats_2x_half_width={c2} c2b_beats_2x_endpoint={c2b} \
             c3_null_medians_within_2pct={c3} effect_pass={effect_pass} \
             effect_deviation={effect_deviation:.6} null_half_width={null_half_width:.6} \
             required_c2={:.6} required_c2b={:.6} \
             left_null_median={left_null_median:.6} \
             right_null_median={right_null_median:.6} => {outcome}",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!(
            "{label}_ratio_median={:.6} ci95=[{low:.6},{high:.6}] cv={:.3}% \
             cv_used_for_decision=false",
            median(ratios.to_vec()),
            cv(ratios) * 100.0
        );
        QmrBatchGate {
            low,
            high,
            effect_pass,
        }
    }

    fn run_qmr_batch(
        batch_method: IterativeBatchMethod,
        elf_sha256: &str,
        side: usize,
        rounds: usize,
        max_iter: usize,
        script_argument: Option<&String>,
    ) -> Result<(), String> {
        let registered_side = batch_method.registered_side();
        if side != registered_side || rounds != QMR_BATCH_ROUNDS {
            return Err(format!(
                "registered {} cell requires side={registered_side} and rounds={QMR_BATCH_ROUNDS}",
                batch_method.batch_label()
            ));
        }
        let source_commit = qmr_batch_required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = qmr_batch_required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = qmr_batch_required_env("BINARY_BUILD_ROUTE")?;
        let claim_id = qmr_batch_required_env("COORDINATION_CLAIM_ID")?;
        let release_id = qmr_batch_required_env("COORDINATION_RELEASE_ID")?;
        let lock_held = qmr_batch_required_env("FSCI_BENCH_LOCK_HELD")?;
        if lock_held != "1" {
            return Err("FSCI_BENCH_LOCK_HELD must equal 1".to_string());
        }
        println!(
            "binary_provenance: source_commit={source_commit} builder_identity={builder_identity} \
             build_route={build_route} elf_sha256={elf_sha256}"
        );
        println!(
            "coordination: claim_id={claim_id} release_id={release_id} \
             filesystem_lock_state=held"
        );

        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        print_hardware_provenance(&affinity)?;
        require_iterative_batch_affinity(&affinity)?;
        if observed_os_threads()? != 1 {
            return Err("iterative batch harness started with more than one OS thread".to_string());
        }
        require_host_wide_quiescence("pre")?;

        let n = side * side;
        let matrix = convection_diffusion_2d(side);
        let rhs = rhs(n);
        let rhses = qmr_batch_rhs(&rhs);
        println!(
            "fixture=nonsymmetric-convection-diffusion-2d method={} side={side} \
             n={n} nnz={} scenarios={QMR_BATCH_SIZE} \
             rhs=32_identical_copies_of_1+0.01*(i%17) rtol={RTOL} atol=0 \
             maxiter={max_iter} x0=zeros rounds={rounds} \
             construction_outside_timing=true rhs_cloning_outside_timing=true \
             serialization_outside_timing=true",
            batch_method.batch_label(),
            matrix.nnz()
        );
        println!(
            "whole_job_boundary: INCLUDED=1_public_{}_32_solves_or_32_scalar_calls; \
             EXCLUDED=construction,cloning,pool_warmup,python_startup,scipy_import,\
             serialization,parity,hashing,bootstrap",
            batch_method.batch_label().replace('-', "_")
        );

        ITERATIVE_BATCH_LAST_WORKERS.store(0, Ordering::SeqCst);
        let (candidate_result, actual_candidate_workers) = observed_peak_worker_tasks(|| {
            solve_ours_iterative_batch(batch_method, &matrix, &rhses, max_iter, false)
        });
        let candidate = candidate_result?;
        let selected_candidate_workers = ITERATIVE_BATCH_LAST_WORKERS.load(Ordering::SeqCst);
        let sequential = solve_ours_iterative_batch(batch_method, &matrix, &rhses, max_iter, true)?;
        let selected_sequential_workers = ITERATIVE_BATCH_LAST_WORKERS.load(Ordering::SeqCst);
        if candidate != sequential || !valid_iterative_batch(&candidate) {
            return Err(format!(
                "candidate and same-ELF sequential {} batches differ",
                batch_method.label().to_ascii_uppercase()
            ));
        }
        if actual_candidate_workers != QMR_BATCH_SIZE
            || selected_candidate_workers != QMR_BATCH_SIZE
            || selected_sequential_workers != 1
        {
            return Err(format!(
                "{} routing mismatch: actual_candidate={actual_candidate_workers} \
                 selected_candidate={selected_candidate_workers} \
                 selected_sequential={selected_sequential_workers}",
                batch_method.batch_label()
            ));
        }
        if candidate
            .iter()
            .skip(1)
            .any(|result| result != &candidate[0])
        {
            return Err(format!(
                "identical {} scenarios produced different ordered results",
                batch_method.label().to_ascii_uppercase()
            ));
        }
        let b_norm = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
        let mut maximum_true_residual = 0.0f64;
        for result in &candidate {
            let ax = matrix
                .matvec(&result.solution)
                .map_err(|error| format!("iterative batch residual matvec: {error}"))?;
            let residual = rhs
                .iter()
                .zip(&ax)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt()
                / b_norm;
            maximum_true_residual = maximum_true_residual.max(residual);
        }
        if maximum_true_residual > 1.25 * RTOL {
            return Err(format!(
                "candidate true residual {maximum_true_residual:.3e} exceeds contract"
            ));
        }

        let script = scipy_oracle_script(script_argument)?;
        println!("scipy_oracle_script={}", script.display());
        let scalar_method = batch_method.method();
        let (mut scipy, identity) = Scipy::start(&script, scalar_method)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains(&format!("method={}", batch_method.label()))
            || !identity.contains("solver_mod=scipy.sparse.linalg._isolve")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine 1.17.1 identity gate".to_string());
        }
        let scipy_engine_sha256 = ready_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy arm omitted its engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy arm reported an invalid engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "thread_provenance: requested_frankenscipy_threads={QMR_BATCH_SIZE} \
             actual_observed_frankenscipy_worker_threads={actual_candidate_workers} \
             selected_candidate_workers={selected_candidate_workers} \
             forced_sequential_active_tasks={selected_sequential_workers} \
             dormant_cached_pool_threads={} requested_scipy_threads=1 \
             actual_observed_scipy_worker_threads=1 python_blas_thread_cap=1",
            QMR_BATCH_SIZE - 1
        );
        println!(
            "Legacy incumbent arm: SciPy 1.17.1 public {} called {QMR_BATCH_SIZE} \
             times sequentially; side-by-side same-invocation",
            batch_method.label()
        );

        let case = scipy.initialize(&matrix, &rhs, max_iter)?;
        let expected_case = format!(
            "CASE method={} n={n} nnz={} sorted=True canonical=True finite=True \
             nonsymmetric=True",
            batch_method.label(),
            matrix.nnz()
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy arm constructed the wrong fixture: {case}"
            ));
        }
        let frankenscipy_input_sha256 = canonical_input_sha256(&matrix, &rhs)?;
        let scipy_input_sha256 = scipy.input_sha256()?;
        if frankenscipy_input_sha256 != scipy_input_sha256 {
            return Err(format!(
                "canonical input digest mismatch: frankenscipy={frankenscipy_input_sha256} \
                 scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={frankenscipy_input_sha256} \
             frankenscipy_input_sha256={frankenscipy_input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );
        let theirs = scipy.parity(n)?;
        let ours = &candidate[0];
        let mut maximum_scaled_difference = 0.0f64;
        let mut difference_squared = 0.0f64;
        let mut scipy_squared = 0.0f64;
        let mut tolerance_mismatches = 0usize;
        for result in &candidate {
            for (&left, &right) in result.solution.iter().zip(&theirs.solution) {
                let difference = (left - right).abs();
                let tolerance = 10.0 * RTOL * right.abs().max(1.0);
                maximum_scaled_difference = maximum_scaled_difference.max(difference / tolerance);
                tolerance_mismatches +=
                    usize::from(!difference.is_finite() || difference > tolerance);
                difference_squared += difference * difference;
                scipy_squared += right * right;
            }
        }
        let relative_l2_difference =
            difference_squared.sqrt() / scipy_squared.sqrt().max(f64::EPSILON);
        println!(
            "agreement: batch_results={} components_per_result={} \
             candidate_control_exact=true candidate_control_bit_mismatches=0 \
             relative_l2_diff={relative_l2_difference:.3e} \
             maximum_scaled_diff={maximum_scaled_difference:.3e} \
             tolerance_mismatches={tolerance_mismatches} ours_iterations={} \
             scipy_iterations={} ours_true_residual={maximum_true_residual:.3e} \
             scipy_true_residual={:.3e}",
            candidate.len(),
            ours.solution.len(),
            ours.iterations,
            theirs.iterations,
            theirs.residual
        );
        if !scalar_method.scipy_status_is_converged(theirs.info)
            || theirs.residual > 1.25 * RTOL
            || !relative_l2_difference.is_finite()
            || relative_l2_difference > 5.0e-4
            || tolerance_mismatches != 0
        {
            return Err("batch arms did not solve numerically comparable systems".to_string());
        }

        let repetitions =
            calibrate_iterative_batch(batch_method, &mut scipy, &matrix, &rhses, max_iter)?;
        println!(
            "calibration whole_batch_repetitions={repetitions} \
             min_sample_ms={QMR_BATCH_MIN_SAMPLE_MS}"
        );
        for warmup in 0..3 {
            let _ = time_ours_iterative_batch(
                batch_method,
                &matrix,
                &rhses,
                max_iter,
                repetitions,
                false,
            )?;
            let _ = time_ours_iterative_batch(
                batch_method,
                &matrix,
                &rhses,
                max_iter,
                repetitions,
                true,
            )?;
            let _ = scipy.solve(repetitions * QMR_BATCH_SIZE, n)?;
            black_box(warmup);
        }
        require_host_wide_quiescence("measurement")?;
        let measurement = measure_qmr_batch(
            batch_method,
            &mut scipy,
            &matrix,
            &rhses,
            max_iter,
            rounds,
            repetitions,
        )?;
        require_host_wide_quiescence("post")?;

        let per_batch = repetitions as f64;
        let candidate_p50 = median(measurement.candidate.clone()) * 1.0e3 / per_batch;
        let candidate_p95 =
            qmr_batch_percentile(measurement.candidate.clone(), 0.95) * 1.0e3 / per_batch;
        let candidate_p99 =
            qmr_batch_percentile(measurement.candidate.clone(), 0.99) * 1.0e3 / per_batch;
        let sequential_p50 = median(measurement.sequential.clone()) * 1.0e3 / per_batch;
        let sequential_p95 =
            qmr_batch_percentile(measurement.sequential.clone(), 0.95) * 1.0e3 / per_batch;
        let sequential_p99 =
            qmr_batch_percentile(measurement.sequential.clone(), 0.99) * 1.0e3 / per_batch;
        let scipy_p50 = median(measurement.scipy.clone()) * 1.0e3 / per_batch;
        let scipy_p95 = qmr_batch_percentile(measurement.scipy.clone(), 0.95) * 1.0e3 / per_batch;
        let scipy_p99 = qmr_batch_percentile(measurement.scipy.clone(), 0.99) * 1.0e3 / per_batch;
        println!(
            "whole_batch_wall: candidate_p50={candidate_p50:.6}ms \
             p95={candidate_p95:.6}ms p99={candidate_p99:.6}ms | \
             sequential_p50={sequential_p50:.6}ms p95={sequential_p95:.6}ms \
             p99={sequential_p99:.6}ms | scipy_p50={scipy_p50:.6}ms \
             p95={scipy_p95:.6}ms p99={scipy_p99:.6}ms"
        );
        println!(
            "arm_cv_provenance: candidate={:.3}% sequential={:.3}% scipy={:.3}% \
             cv_used_for_decision=false",
            cv(&measurement.candidate) * 100.0,
            cv(&measurement.sequential) * 100.0,
            cv(&measurement.scipy) * 100.0
        );
        println!(
            "raw_samples_seconds: candidate={} sequential={} scipy={} \
             candidate_null={} sequential_null={} scipy_null={}",
            qmr_batch_csv(&measurement.candidate),
            qmr_batch_csv(&measurement.sequential),
            qmr_batch_csv(&measurement.scipy),
            qmr_batch_csv(&measurement.candidate_nulls),
            qmr_batch_csv(&measurement.sequential_nulls),
            qmr_batch_csv(&measurement.scipy_nulls)
        );
        let maintenance = print_qmr_batch_gate(
            "maintenance_sequential_over_candidate",
            &measurement.maintenance_ratios,
            &measurement.candidate_nulls,
            &measurement.sequential_nulls,
        );
        let competitive = print_qmr_batch_gate(
            "competitive_scipy_over_candidate",
            &measurement.incumbent_ratios,
            &measurement.candidate_nulls,
            &measurement.scipy_nulls,
        );
        let tail_pass = candidate_p95 < sequential_p95
            && candidate_p99 < sequential_p99
            && candidate_p95 < scipy_p95
            && candidate_p99 < scipy_p99;
        let duration_pass = candidate_p50 >= 5.0;
        let maintenance_pass = maintenance.effect_pass && maintenance.low > 4.0;
        let competitive_pass = competitive.effect_pass && competitive.low > 2.0;
        let keep = maintenance_pass && competitive_pass && tail_pass && duration_pass;
        let evidence_class = if EXCLUSIVITY_WAIVED.load(Ordering::SeqCst) {
            "PROVISIONAL_NON_EXCLUSIVE"
        } else {
            "CAMPAIGN-WIN"
        };
        println!(
            "registered_timing_decision: maintenance_ci=[{:.6},{:.6}] threshold_low_gt_4 \
             maintenance_pass={maintenance_pass} competitive_ci=[{:.6},{:.6}] \
             threshold_low_gt_2 competitive_pass={competitive_pass} \
             tail_pass={tail_pass} candidate_duration_pass={duration_pass} \
             evidence_class={evidence_class} => {}",
            maintenance.low,
            maintenance.high,
            competitive.low,
            competitive.high,
            if keep { "KEEP" } else { "REVERT" }
        );
        scipy.quit();
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        let args: Vec<String> = std::env::args().collect();
        let side = args
            .get(1)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);
        let rounds = args
            .get(2)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(21);
        let method_argument = args.get(3).map_or("gmres", String::as_str);
        let batch_method = match method_argument {
            "qmr-batch" => Some(IterativeBatchMethod::Qmr),
            "lgmres-batch" => Some(IterativeBatchMethod::Lgmres),
            _ => None,
        };
        let method = batch_method.map_or_else(
            || Method::parse(method_argument),
            |batch| Ok(batch.method()),
        )?;
        if !(2..=1_024).contains(&side) || rounds < 5 {
            return Err("require 2<=side<=1024 and rounds>=5".to_string());
        }
        let n = side
            .checked_mul(side)
            .ok_or_else(|| "fixture dimension overflowed".to_string())?;
        let max_iter = n
            .checked_mul(10)
            .ok_or_else(|| "maximum iteration count overflowed".to_string())?;
        if let Some(batch_method) = batch_method {
            return run_qmr_batch(
                batch_method,
                &elf_sha256,
                side,
                rounds,
                max_iter,
                args.get(4),
            );
        }
        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        let multi_profile = matches!(method, Method::Bicgstab)
            && std::env::var_os("FSCI_BICGSTAB_PROFILE_MULTI").is_some_and(|value| value == "1");
        if affinity == "unknown" {
            return Err("CPU affinity is unreadable".to_string());
        }
        let affinity_cpu_count = affinity_cpus(&affinity)?.len();
        if !multi_profile && affinity_cpu_count != 1 {
            return Err("pin this invocation to exactly one CPU with taskset".to_string());
        }
        if multi_profile && affinity_cpu_count < 2 {
            return Err(
                "multi-CPU BiCGSTAB profile requires at least two affinity CPUs".to_string(),
            );
        }
        print_hardware_provenance(&affinity)?;
        println!(
            "evidence_protocol={} bicgstab_multi_profile={multi_profile}",
            if multi_profile {
                "PROFILE_ONLY_MULTI_CPU"
            } else {
                "DEFAULT_SINGLE_CPU"
            }
        );
        require_host_wide_quiescence("pre")?;

        let (matrix, fixture_label, west, east, nonsymmetric) = if matches!(method, Method::Cg) {
            (
                dirichlet_laplacian_2d(side),
                "dirichlet-five-point-laplacian-2d",
                VERTICAL,
                VERTICAL,
                "False",
            )
        } else {
            (
                convection_diffusion_2d(side),
                "nonsymmetric-convection-diffusion-2d",
                WEST,
                EAST,
                "True",
            )
        };
        let rhs = rhs(n);
        println!(
            "fixture={fixture_label} method={} side={side} n={n} \
             nnz={} diagonal={DIAGONAL} west={west} east={east} vertical={VERTICAL} \
             rhs=1+0.01*(i%17) rtol={RTOL} atol=0 maxiter={max_iter} x0=zeros \
             rounds={rounds} construction_outside_timing=true \
             serialization_outside_timing=true",
            method.label(),
            matrix.nnz()
        );

        let script = scipy_oracle_script(args.get(4))?;
        println!("scipy_oracle_script={}", script.display());
        let (mut scipy, identity) = Scipy::start(&script, method)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains(&format!("method={}", method.label()))
            || !identity.contains("solver_mod=scipy.sparse.linalg._isolve")
            || !identity.contains("scipy_engine_sha256=")
            || !identity.contains("actual_observed_worker_threads=")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine-incumbent identity gate".to_string());
        }
        let scipy_version = ready_value(&identity, "scipy=")
            .ok_or_else(|| "live SciPy arm omitted its version".to_string())?;
        let scipy_engine_sha256 = ready_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "live SciPy arm omitted its engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256) {
            return Err("live SciPy arm reported an invalid engine SHA-256".to_string());
        }
        let actual_scipy_workers = ready_value(&identity, "actual_observed_worker_threads=")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| "live SciPy arm omitted its actual observed worker count".to_string())?;
        if actual_scipy_workers != 1 {
            return Err(format!(
                "live SciPy arm observed {actual_scipy_workers} threads; expected one"
            ));
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "Legacy incumbent arm: SciPy {scipy_version}; side-by-side same-invocation; \
             child-side scipy.sparse.linalg.{}-only timing",
            method.label()
        );

        let case = scipy.initialize(&matrix, &rhs, max_iter)?;
        let expected_case = format!(
            "CASE method={} n={n} nnz={} sorted=True canonical=True finite=True \
             nonsymmetric={nonsymmetric}",
            method.label(),
            matrix.nnz()
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy arm constructed the wrong fixture: {case}"
            ));
        }
        println!("scipy_case: {case}");
        let frankenscipy_input_sha256 = canonical_input_sha256(&matrix, &rhs)?;
        let scipy_input_sha256 = scipy.input_sha256()?;
        if !frankenscipy_input_sha256
            .bytes()
            .eq(scipy_input_sha256.bytes())
        {
            return Err(format!(
                "canonical input digest mismatch: frankenscipy={frankenscipy_input_sha256} \
                 scipy={scipy_input_sha256}"
            ));
        }
        println!(
            "input_sha256={frankenscipy_input_sha256} \
             frankenscipy_input_sha256={frankenscipy_input_sha256} \
             scipy_input_sha256={scipy_input_sha256} input_digest_match=true"
        );

        let actual_ours_before = observed_os_threads()?;
        let (ours, actual_ours_workers) = if multi_profile {
            observed_peak_worker_tasks(|| solve_ours(method, &matrix, &rhs, max_iter))
        } else {
            (solve_ours(method, &matrix, &rhs, max_iter), 1)
        };
        let actual_ours_after = observed_os_threads()?;
        if actual_ours_before != 1 || actual_ours_after != 1 {
            return Err(format!(
                "FrankenSciPy arm leaked tasks across the solve: before={actual_ours_before} \
                 after={actual_ours_after}"
            ));
        }
        if multi_profile && actual_ours_workers < 2 {
            return Err(format!(
                "multi-CPU BiCGSTAB profile observed only {actual_ours_workers} worker task"
            ));
        }
        let requested_ours = if multi_profile { "auto" } else { "1" };
        println!(
            "thread_provenance: requested_threads={affinity_cpu_count} \
             requested_frankenscipy_threads={requested_ours} \
             actual_observed_frankenscipy_worker_threads={actual_ours_workers} \
             requested_scipy_threads=1 \
             actual_observed_scipy_worker_threads={actual_scipy_workers} \
             python_blas_thread_cap=1"
        );
        let theirs = scipy.parity(n)?;
        let ax = matrix
            .matvec(&ours.solution)
            .map_err(|error| format!("FrankenSciPy residual matvec: {error}"))?;
        let residual_numerator = rhs
            .iter()
            .zip(&ax)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt();
        let residual_denominator = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
        let ours_true_residual = residual_numerator / residual_denominator;
        let mut max_abs_difference = 0.0f64;
        let mut maximum_scaled_difference = 0.0f64;
        let mut difference_squared = 0.0f64;
        let mut scipy_squared = 0.0f64;
        let mut tolerance_mismatches = 0usize;
        for (&left, &right) in ours.solution.iter().zip(&theirs.solution) {
            let difference = (left - right).abs();
            let tolerance = 10.0 * RTOL * right.abs().max(1.0);
            max_abs_difference = max_abs_difference.max(difference);
            maximum_scaled_difference = maximum_scaled_difference.max(difference / tolerance);
            tolerance_mismatches += usize::from(!difference.is_finite() || difference > tolerance);
            difference_squared += difference * difference;
            scipy_squared += right * right;
        }
        let relative_l2_difference =
            difference_squared.sqrt() / scipy_squared.sqrt().max(f64::EPSILON);
        println!(
            "agreement: components={}/{} max_abs_diff={max_abs_difference:.3e} \
             relative_l2_diff={relative_l2_difference:.3e} \
             maximum_scaled_diff={maximum_scaled_difference:.3e} \
             tolerance_mismatches={tolerance_mismatches} \
             component_tolerance=10*rtol*max(1,abs(scipy)) \
             true_residual_ours={ours_true_residual:.3e} \
             true_residual_scipy={:.3e}",
            ours.solution.len(),
            theirs.solution.len(),
            theirs.residual
        );
        if matches!(method, Method::Lgmres) {
            println!(
                "execution: ours converged={} inner_arnoldi_steps={} \
                 reported_residual={:.3e} | scipy info={} counted_outer_cycles={} \
                 iteration_counts_comparable=false",
                ours.converged, ours.iterations, ours.residual_norm, theirs.info, theirs.iterations
            );
        } else {
            println!(
                "execution: ours converged={} iterations={} reported_residual={:.3e} | \
                 scipy info={} counted_inner_iterations={} \
                 iteration_ratio={:.4}",
                ours.converged,
                ours.iterations,
                ours.residual_norm,
                theirs.info,
                theirs.iterations,
                ours.iterations as f64 / theirs.iterations.max(1) as f64
            );
        }
        if matches!(method, Method::Gmres) {
            println!(
                "solver_schedule: frankenscipy_restart=20 scipy_restart=default_20 \
                 both_public_defaults=true scipy_callback_type=pr_norm_counting_outside_timing"
            );
        }
        if matches!(method, Method::Cg) {
            println!(
                "solver_schedule: both_public_cg=true both_unpreconditioned=true \
                 matvecs_per_iteration_ours=1 matvecs_per_iteration_scipy=1 \
                 scipy_callback_type=per_iteration_x_counting_outside_timing"
            );
        }
        if matches!(method, Method::Lgmres) {
            println!(
                "solver_schedule: frankenscipy_inner_m=30 scipy_inner_m=30 \
                 frankenscipy_outer_k=3 scipy_outer_k=3 \
                 scipy_store_outer_Av=true scipy_prepend_outer_v=false \
                 scipy_callback_type=per_outer_cycle_x_counting_outside_timing"
            );
        }
        if matches!(method, Method::Bicg) {
            println!(
                "solver_schedule: frankenscipy_transpose=materialized_csr_once_per_solve \
                 scipy_transpose=csr_rmatvec_operator \
                 matvecs_per_iteration_ours=2 matvecs_per_iteration_scipy=2 \
                 scipy_callback_type=per_iteration_x_counting_outside_timing"
            );
        }
        if matches!(method, Method::Cgs) {
            println!(
                "solver_schedule: matvecs_per_iteration_ours=2 \
                 matvecs_per_iteration_scipy=2 \
                 scipy_callback_type=per_iteration_x_counting_outside_timing"
            );
        }
        if matches!(method, Method::Lsqr | Method::Lsmr) {
            // Both least-squares APIs return the exact iteration count, so the
            // counted trajectory needs no callback instrumentation.
            println!(
                "solver_schedule: frankenscipy_method={} \
                 scipy_atol=0 scipy_btol=rtol scipy_conlim=0 scipy_damp=0 \
                 scipy_iteration_source=returned_itn_no_callback",
                if matches!(method, Method::Lsmr) {
                    "delegated_lsqr"
                } else {
                    "lsqr"
                }
            );
        }
        if matches!(method, Method::Qmr) {
            // Both arms run un-preconditioned QMR with algebraically identical
            // recurrences (verified line-by-line against SciPy 1.17.1 in the
            // pre-registration). They differ in residual bookkeeping only:
            // SciPy carries r recursively and tests ||r|| < max(0, rtol*||b||);
            // we recompute the true residual b - Ax and test the same relative
            // quantity, which costs us a third matvec per iteration against
            // SciPy's two. SciPy's callback fires once per completed loop body,
            // so the counted trajectory is the executed work.
            println!(
                "solver_schedule: frankenscipy_stop=true_residual_over_bnorm_lt_tol \
                 scipy_stop=recursive_residual_lt_atol scipy_atol=0 scipy_rtol=tol \
                 scipy_M1=None scipy_M2=None scipy_identity_preconditioner_dispatches=4 \
                 matvecs_per_iteration_ours=3 matvecs_per_iteration_scipy=2 \
                 scipy_callback_type=per_iteration_x_counting_outside_timing"
            );
        }
        let scipy_converged = method.scipy_status_is_converged(theirs.info);
        let cg_trajectory_matches = !matches!(method, Method::Cg)
            || (ours.iterations == theirs.iterations && relative_l2_difference <= 1.0e-10);
        if matches!(method, Method::Cg) {
            println!(
                "cg_trajectory: same_iterations={} relative_l2_limit=1e-10 pass={cg_trajectory_matches}",
                ours.iterations == theirs.iterations
            );
        }
        if !ours.converged
            || !scipy_converged
            || ours.solution.len() != n
            || theirs.solution.len() != n
            || ours.iterations == 0
            || theirs.iterations == 0
            || !ours_true_residual.is_finite()
            || !theirs.residual.is_finite()
            || ours_true_residual > 1.25 * RTOL
            || theirs.residual > 1.25 * RTOL
            || !relative_l2_difference.is_finite()
            || relative_l2_difference > 5.0e-4
            || tolerance_mismatches != 0
            || !cg_trajectory_matches
        {
            return Err("arms did not solve a numerically comparable system".to_string());
        }

        let repetitions = calibrate_repetitions(&mut scipy, method, &matrix, &rhs, max_iter)?;
        println!("calibration repetitions={repetitions} min_sample_ms={MIN_SAMPLE_MS}");
        for warmup in 0..4 {
            let _ = incumbent_pair(
                &mut scipy,
                method,
                &matrix,
                &rhs,
                max_iter,
                repetitions,
                warmup,
            )?;
        }
        require_host_wide_quiescence("measurement")?;

        let mut ours_times = Vec::with_capacity(rounds);
        let mut scipy_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut ours_nulls = Vec::with_capacity(rounds);
        let mut scipy_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (ours_time, scipy_time, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent = incumbent_pair(
                        &mut scipy,
                        method,
                        &matrix,
                        &rhs,
                        max_iter,
                        repetitions,
                        round,
                    )?;
                    let ours_null =
                        ours_null_pair(method, &matrix, &rhs, max_iter, repetitions, round);
                    let scipy_null = scipy_null_pair(&mut scipy, n, repetitions, round)?;
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(&mut scipy, n, repetitions, round)?;
                    let incumbent = incumbent_pair(
                        &mut scipy,
                        method,
                        &matrix,
                        &rhs,
                        max_iter,
                        repetitions,
                        round,
                    )?;
                    let ours_null =
                        ours_null_pair(method, &matrix, &rhs, max_iter, repetitions, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null =
                        ours_null_pair(method, &matrix, &rhs, max_iter, repetitions, round);
                    let scipy_null = scipy_null_pair(&mut scipy, n, repetitions, round)?;
                    let incumbent = incumbent_pair(
                        &mut scipy,
                        method,
                        &matrix,
                        &rhs,
                        max_iter,
                        repetitions,
                        round,
                    )?;
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_times.push(ours_time);
            scipy_times.push(scipy_time);
            ratios.push(scipy_time / ours_time);
            ours_nulls.push(ours_null);
            scipy_nulls.push(scipy_null);
        }
        require_host_wide_quiescence("post")?;

        let (ratio_low, ratio_high) = boot_ci(&ratios);
        let (ours_null_low, ours_null_high) = boot_ci(&ours_nulls);
        let (scipy_null_low, scipy_null_high) = boot_ci(&scipy_nulls);
        println!(
            "OURS p50={:.6}ms/rep SCIPY p50={:.6}ms/rep",
            median(ours_times) * 1_000.0 / repetitions as f64,
            median(scipy_times) * 1_000.0 / repetitions as f64
        );
        println!(
            "NULL-ours A/A median={:.6} ci95=[{ours_null_low:.6},{ours_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(ours_nulls.clone()),
            cv(&ours_nulls) * 100.0
        );
        println!(
            "NULL-scipy A/A median={:.6} ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(scipy_nulls.clone()),
            cv(&scipy_nulls) * 100.0
        );
        let ratio_p50 = median(ratios.clone());
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_p50:.4}x \
             (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
             cv={:.3}% provenance only)",
            cv(&ratios) * 100.0
        );
        // ---- Corrected A/A null gate (fleet standard, adopted 2026-07-30) ----
        //
        // A row is decidable when ALL of:
        //   c1  the effect CI excludes 1.0
        //   c2  the effect deviation exceeds 2x the LARGER null half-width
        //   c3  each null MEDIAN is within 2% of 1.0
        //
        // c3 is the substance: it bounds arm-order bias WITHOUT coupling the
        // verdict to how precise the measurement is. Null CIs are still printed,
        // but they no longer veto. There was never a CI-straddle veto here (a
        // null CI excluding 1.0 has never blocked a row in this harness), so
        // adopting this is a strengthening via c3 rather than a repair.
        //
        // c2b is RETAINED and is NOT part of the fleet rule. It is this
        // harness's original margin, built on the largest multiplicative
        // deviation of any null CI *endpoint* from 1.0 rather than on the
        // half-width. For a null CI offset from 1.0 the endpoint deviation
        // always exceeds the half-width, so c2b is strictly stricter than c2 —
        // measured looser in 23 of 23 audited cells. Dropping it in favour of
        // c2 alone would therefore be a loosening, which the standard
        // explicitly warns against, so both must hold.
        let null_half_width = ((ours_null_high - ours_null_low) / 2.0)
            .max((scipy_null_high - scipy_null_low) / 2.0)
            .max(0.0);
        let ours_null_median = median(ours_nulls.clone());
        let scipy_null_median = median(scipy_nulls.clone());
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1e-9))
            .max(1.0 / scipy_null_low.max(1e-9))
            .max(1.0);

        let c1_effect_ci_excludes_one = ratio_low > 1.0 || ratio_high < 1.0;
        // Deviation of the effect CI's near edge from 1.0 — conservative, and
        // consistent with c1 already demanding the whole CI clear 1.0.
        let effect_deviation = if ratio_low > 1.0 {
            ratio_low - 1.0
        } else if ratio_high < 1.0 {
            1.0 - ratio_high
        } else {
            0.0
        };
        let c2_beats_half_width_margin = effect_deviation > 2.0 * null_half_width;
        let c2b_beats_endpoint_margin = effect_deviation > 2.0 * (null_edge - 1.0);
        let c3_null_medians_unbiased = (ours_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (scipy_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let decidable = c1_effect_ci_excludes_one
            && c2_beats_half_width_margin
            && c2b_beats_endpoint_margin
            && c3_null_medians_unbiased;

        println!(
            "corrected-null-gate: c1_effect_ci_excludes_one={c1_effect_ci_excludes_one} \
             c2_beats_2x_half_width={c2_beats_half_width_margin} \
             c2b_beats_2x_endpoint={c2b_beats_endpoint_margin} \
             c3_null_medians_within_{:.0}pct={c3_null_medians_unbiased} \
             decidable={decidable} effect_deviation={effect_deviation:.6} \
             null_half_width={null_half_width:.6} required_c2={:.6} \
             required_c2b={:.6} ours_null_median={ours_null_median:.6} \
             scipy_null_median={scipy_null_median:.6} \
             null_ci_veto=disabled_telemetry_only",
            NULL_MEDIAN_BIAS_LIMIT * 100.0,
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );

        let required = 1.0 + 2.0 * (null_edge - 1.0);
        // A run whose environmental exclusivity was waived must never emit the
        // token "DECIDED", so that no provisional row can be laundered by
        // grepping the verdict line.
        let waived = EXCLUSIVITY_WAIVED.load(Ordering::SeqCst);
        let outcome = match (
            decidable && ratio_low > 1.0,
            decidable && ratio_high < 1.0,
            waived,
        ) {
            (true, _, false) => "DECIDED FRANKENSCIPY WIN",
            (_, true, false) => "DECIDED FRANKENSCIPY LOSS",
            (false, false, false) => "NOT DECIDED",
            (true, _, true) => {
                "PROVISIONAL FRANKENSCIPY WIN (non-exclusive host; NOT DECIDED-class evidence)"
            }
            (_, true, true) => {
                "PROVISIONAL FRANKENSCIPY LOSS (non-exclusive host; NOT DECIDED-class evidence)"
            }
            (false, false, true) => {
                "PROVISIONAL INDETERMINATE (non-exclusive host; NOT DECIDED-class evidence)"
            }
        };
        println!(
            "median-CI gate: worst_null_edge={null_edge:.4} required={required:.4} \
             ratio_ci=[{ratio_low:.4},{ratio_high:.4}] \
             null_margin=2x cv_used_for_decision=false => {outcome}"
        );
        scipy.quit();
        Ok(())
    }
}

#[cfg(feature = "sparse-incumbent-bench")]
fn main() {
    if let Err(error) = bench::run() {
        eprintln!("ABORT: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(feature = "sparse-incumbent-bench"))]
fn main() {
    eprintln!("perf_sparse_vs_scipy requires --features sparse-incumbent-bench");
    std::process::exit(2);
}
