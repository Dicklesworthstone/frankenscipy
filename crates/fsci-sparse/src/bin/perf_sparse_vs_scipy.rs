//! Sparse GMRES/BiCGSTAB versus a genuine live SciPy incumbent.
//!
//! Rust transmits the exact deterministic CSR and RHS once to a persistent Python
//! co-process. Construction, serialization, callback iteration counting, and parity
//! checks are outside timing. The timed arms are interleaved in one invocation with
//! independent A/A nulls, executable identities, full hardware/thread/frequency
//! provenance, and fail-closed host-wide quiescence checks.
//!
//! Run: `cargo run --profile release-perf --bin perf_sparse_vs_scipy \
//!       --features sparse-incumbent-bench -- [side] [rounds] [gmres|bicgstab|lsqr]`

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::linalg::{IterativeSolveOptions, bicgstab, gmres, lsqr};
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
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(300);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    /// Corrected-gate clause 3: each A/A null median must sit within this
    /// fraction of 1.0. Bounds arm-order bias without coupling the verdict to
    /// the null's precision.
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;

    #[derive(Clone, Copy)]
    enum Method {
        Gmres,
        Bicgstab,
        Lsqr,
    }

    impl Method {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "gmres" => Ok(Self::Gmres),
                "bicgstab" => Ok(Self::Bicgstab),
                "lsqr" => Ok(Self::Lsqr),
                _ => Err(format!(
                    "unknown method {value:?}; expected gmres, bicgstab, or lsqr"
                )),
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::Gmres => "gmres",
                Self::Bicgstab => "bicgstab",
                Self::Lsqr => "lsqr",
            }
        }

        /// SciPy's success code is method-dependent. The `_isolve` solvers
        /// return `info == 0`, but `lsqr` returns `istop`, where 1 means
        /// "Ax - b is small enough" and 2 means the least-squares solution is
        /// good enough. With `atol=0` and `conlim=0` we expect `istop == 1`.
        /// Single definition so the parity path and the timed path cannot drift.
        const fn scipy_status_is_converged(self, status: i32) -> bool {
            match self {
                Self::Lsqr => status == 1 || status == 2,
                Self::Gmres | Self::Bicgstab => status == 0,
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

    fn rhs(n: usize) -> Vec<f64> {
        (0..n)
            .map(|index| 1.0 + 0.01 * (index % 17) as f64)
            .collect()
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
            Method::Gmres => gmres(matrix, rhs, None, options),
            Method::Bicgstab => bicgstab(matrix, rhs, None, options),
            // lsqr minimizes ||Ax - b||_2 and takes no initial guess; its
            // stopping test is |phi_bar| / ||b|| < tol, which the SciPy arm
            // mirrors with atol=0, btol=tol, conlim=0.
            Method::Lsqr => lsqr(matrix, rhs, options),
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
        std::env::var_os("FSCI_SPARSE_ALLOW_NON_EXCLUSIVE")
            .is_some_and(|value| value == "1")
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

    fn read_policy_field(base: &Path, name: &str, required: bool) -> Result<String, String> {
        let path = base.join(name);
        match std::fs::read_to_string(&path) {
            Ok(value) => Ok(value.trim().replace(char::is_whitespace, "_")),
            Err(error) if !required => Ok(format!("unavailable({error})").replace(' ', "_")),
            Err(error) => Err(format!("read {}: {error}", path.display())),
        }
    }

    fn print_hardware_provenance(affinity: &str) -> Result<(), String> {
        let cpu = affinity
            .parse::<usize>()
            .map_err(|error| format!("affinity is not one CPU: {error}"))?;
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
             runtime_detected_isa={} affinity={affinity} cpuset_logical_cap=1",
            runtime_isa_features()
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
        let method = Method::parse(args.get(3).map_or("gmres", String::as_str))?;
        if !(2..=1_024).contains(&side) || rounds < 5 {
            return Err("require 2<=side<=1024 and rounds>=5".to_string());
        }
        let n = side
            .checked_mul(side)
            .ok_or_else(|| "fixture dimension overflowed".to_string())?;
        let max_iter = n
            .checked_mul(10)
            .ok_or_else(|| "maximum iteration count overflowed".to_string())?;
        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        if affinity == "unknown" || affinity.contains(',') || affinity.contains('-') {
            return Err("pin this invocation to exactly one CPU with taskset".to_string());
        }
        print_hardware_provenance(&affinity)?;
        require_host_wide_quiescence("pre")?;

        let matrix = convection_diffusion_2d(side);
        let rhs = rhs(n);
        println!(
            "fixture=nonsymmetric-convection-diffusion-2d method={} side={side} n={n} \
             nnz={} diagonal={DIAGONAL} west={WEST} east={EAST} vertical={VERTICAL} \
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
             nonsymmetric=True",
            method.label(),
            matrix.nnz()
        );
        if case != expected_case {
            return Err(format!(
                "live SciPy arm constructed the wrong fixture: {case}"
            ));
        }
        println!("scipy_case: {case}");

        let actual_ours_before = observed_os_threads()?;
        let ours = solve_ours(method, &matrix, &rhs, max_iter);
        let actual_ours_workers = actual_ours_before.max(observed_os_threads()?);
        if actual_ours_workers != 1 {
            return Err(format!(
                "FrankenSciPy arm observed {actual_ours_workers} threads; expected one"
            ));
        }
        println!(
            "thread_provenance: requested_threads=1 \
             requested_frankenscipy_threads=1 \
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
        if matches!(method, Method::Gmres) {
            println!(
                "solver_schedule: frankenscipy_restart=20 scipy_restart=default_20 \
                 both_public_defaults=true scipy_callback_type=pr_norm_counting_outside_timing"
            );
        }
        if matches!(method, Method::Lsqr) {
            // lsqr exposes no callback; SciPy returns the exact iteration count
            // as itn, so the counted trajectory needs no instrumentation. Both
            // arms stop on the same rule: |phi_bar| / ||b|| <= tol.
            println!(
                "solver_schedule: frankenscipy_stop=phi_bar_over_bnorm_lt_tol \
                 scipy_atol=0 scipy_btol=rtol scipy_conlim=0 scipy_damp=0 \
                 stopping_rule_matched=true scipy_iteration_source=returned_itn_no_callback"
            );
        }
        let scipy_converged = method.scipy_status_is_converged(theirs.info);
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
        let c2b_beats_endpoint_margin =
            effect_deviation > 2.0 * (null_edge - 1.0);
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
