//! Whole-job steady convection-diffusion iterative-solver screen versus live SciPy.
//!
//! This is deliberately separate from the single-solve kernel harness. Every
//! timed repetition assembles one 32x32 sparse operator, constructs twelve
//! localized source fields, constructs the selected SciPy preconditioner, runs
//! public solves, materializes every field, and computes three scientific
//! summaries per field. The historical default remains GMRES; `bicgstab`
//! selects the 128-source batch-widening fixture.
//!
//! Run:
//! `perf_gmres_job_vs_scipy [rounds] [scipy_gmres_job_arm.py] [gmres|bicgstab]`

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::linalg::{
        GMRES_BATCH_FORCE_SEQUENTIAL, IterativeSolveOptions, bicgstab, gmres_batch,
    };
    use fsci_sparse::{CsrMatrix, IterativeSolveResult, Shape2D};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};

    const SIDE: usize = 32;
    const GMRES_SCENARIOS: usize = 12;
    const BICGSTAB_SCENARIOS: usize = 128;
    const SUMMARIES_PER_SCENARIO: usize = 3;
    const DIAGONAL: f64 = 4.001;
    const WEST: f64 = -1.2;
    const EAST: f64 = -0.8;
    const VERTICAL: f64 = -1.0;
    const RTOL: f64 = 1.0e-5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_millis(300);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const SOURCE_ROWS: [usize; 3] = [6, 16, 25];
    const SOURCE_COLUMNS: [usize; 4] = [5, 12, 20, 27];
    const CONFIGURATIONS: [&str; 6] = [
        "csr-matrix-none",
        "csr-array-none",
        "csc-matrix-none",
        "csc-array-none",
        "csr-matrix-jacobi",
        "csc-matrix-spilu",
    ];

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Method {
        Gmres,
        Bicgstab,
    }

    impl Method {
        fn parse(value: &str) -> Result<Self, String> {
            match value {
                "gmres" => Ok(Self::Gmres),
                "bicgstab" => Ok(Self::Bicgstab),
                _ => Err(format!(
                    "unknown whole-job method {value:?}; expected gmres or bicgstab"
                )),
            }
        }

        const fn label(self) -> &'static str {
            match self {
                Self::Gmres => "gmres",
                Self::Bicgstab => "bicgstab",
            }
        }

        const fn display(self) -> &'static str {
            match self {
                Self::Gmres => "GMRES",
                Self::Bicgstab => "BiCGSTAB",
            }
        }

        const fn scenarios(self) -> usize {
            match self {
                Self::Gmres => GMRES_SCENARIOS,
                Self::Bicgstab => BICGSTAB_SCENARIOS,
            }
        }
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        method: Method,
    }

    struct ScipyJobCheck {
        configuration: String,
        successes: usize,
        input_sha256: String,
        observed_threads: usize,
        infos: Vec<i32>,
        iterations: Vec<usize>,
        residuals: Vec<f64>,
        fields: Vec<f64>,
        summaries: Vec<f64>,
    }

    struct ScipyJobTiming {
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
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("spawn live SciPy arm: {error}"))?;
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
            let mut reply = String::new();
            self.stdout
                .read_line(&mut reply)
                .map_err(|error| format!("read {context}: {error}"))?;
            if reply.is_empty() {
                return Err(format!("live SciPy arm exited while reading {context}"));
            }
            Ok(reply.trim().to_string())
        }

        fn job_check(&mut self, configuration: &str) -> Result<ScipyJobCheck, String> {
            writeln!(self.stdin, "JOB_CHECK {configuration} {SIDE}")
                .map_err(|error| format!("write JOB_CHECK: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush JOB_CHECK: {error}"))?;
            let header = self.read_reply("SciPy job-check header")?;
            let fields = header.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 10 || fields[0] != "JOB_CHECK" {
                return Err(format!("invalid SciPy job-check header: {header}"));
            }
            let reported_configuration = fields[1].to_string();
            if reported_configuration != configuration {
                return Err(format!(
                    "SciPy checked {reported_configuration}, expected {configuration}"
                ));
            }
            let successes = parse(fields[2], "SciPy job successes")?;
            let components: usize = parse(fields[3], "SciPy job components")?;
            let summary_count: usize = parse(fields[4], "SciPy job summaries")?;
            let input_sha256 = fields[5].to_string();
            let observed_threads = parse(fields[6], "SciPy observed threads")?;
            let infos = parse_csv(fields[7], "SciPy infos")?;
            let iterations = parse_csv(fields[8], "SciPy iteration counts")?;
            let residuals = parse_csv(fields[9], "SciPy residuals")?;
            let fields_line = self.read_reply("SciPy job fields")?;
            let summaries_line = self.read_reply("SciPy job summaries")?;
            let output_fields =
                parse_f64_line(&fields_line, "JOB_X", components, "SciPy job fields")?;
            let summaries = parse_f64_line(
                &summaries_line,
                "JOB_SUMMARIES",
                summary_count,
                "SciPy job summaries",
            )?;
            Ok(ScipyJobCheck {
                configuration: reported_configuration,
                successes,
                input_sha256,
                observed_threads,
                infos,
                iterations,
                residuals,
                fields: output_fields,
                summaries,
            })
        }

        fn job_time(
            &mut self,
            configuration: &str,
            repetitions: usize,
        ) -> Result<ScipyJobTiming, String> {
            let scenarios = self.method.scenarios();
            writeln!(self.stdin, "JOB_TIME {configuration} {SIDE} {repetitions}")
                .map_err(|error| format!("write JOB_TIME: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush JOB_TIME: {error}"))?;
            let reply = self.read_reply("timed SciPy whole job")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "JOB_TIME" {
                return Err(format!("invalid timed SciPy whole job: {reply}"));
            }
            let elapsed: f64 = parse(fields[1], "SciPy whole-job elapsed")?;
            let successes: usize = parse(fields[2], "SciPy timed successes")?;
            let components: usize = parse(fields[3], "SciPy timed components")?;
            let summaries: usize = parse(fields[4], "SciPy timed summaries")?;
            let threads: usize = parse(fields[5], "SciPy timed threads")?;
            let checksum: f64 = parse(fields[6], "SciPy timed checksum")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || successes != scenarios
                || components != scenarios * SIDE * SIDE
                || summaries != scenarios * SUMMARIES_PER_SCENARIO
                || threads != 1
                || !checksum.is_finite()
            {
                return Err(format!("inadmissible timed SciPy whole job: {reply}"));
            }
            Ok(ScipyJobTiming { elapsed })
        }

        fn solve_only_time(
            &mut self,
            configuration: &str,
            repetitions: usize,
        ) -> Result<ScipyJobTiming, String> {
            let scenarios = self.method.scenarios();
            writeln!(
                self.stdin,
                "JOB_SOLVE_ONLY_TIME {configuration} {SIDE} {repetitions}"
            )
            .map_err(|error| format!("write JOB_SOLVE_ONLY_TIME: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush JOB_SOLVE_ONLY_TIME: {error}"))?;
            let reply = self.read_reply("SciPy solve-only timing")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "JOB_SOLVE_ONLY_TIME" {
                return Err(format!("invalid SciPy solve-only timing: {reply}"));
            }
            let elapsed: f64 = parse(fields[1], "SciPy solve-only elapsed")?;
            let successes: usize = parse(fields[2], "SciPy solve-only successes")?;
            let components: usize = parse(fields[3], "SciPy solve-only components")?;
            let threads: usize = parse(fields[4], "SciPy solve-only threads")?;
            let checksum: f64 = parse(fields[5], "SciPy solve-only checksum")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || successes != scenarios
                || components != scenarios * SIDE * SIDE
                || threads != 1
                || !checksum.is_finite()
            {
                return Err(format!("inadmissible SciPy solve-only timing: {reply}"));
            }
            Ok(ScipyJobTiming { elapsed })
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn parse<T>(value: &str, label: &str) -> Result<T, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        value
            .parse::<T>()
            .map_err(|error| format!("parse {label}: {error}"))
    }

    fn parse_csv<T>(values: &str, label: &str) -> Result<Vec<T>, String>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        values.split(',').map(|value| parse(value, label)).collect()
    }

    fn parse_f64_line(
        line: &str,
        prefix: &str,
        expected: usize,
        label: &str,
    ) -> Result<Vec<f64>, String> {
        let values = line
            .strip_prefix(prefix)
            .and_then(|value| value.strip_prefix(' '))
            .ok_or_else(|| format!("invalid {label}: {line}"))?;
        let parsed = parse_csv(values, label)?;
        if parsed.len() != expected {
            return Err(format!(
                "{label} length {} != expected {expected}",
                parsed.len()
            ));
        }
        Ok(parsed)
    }

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
            .expect("canonical whole-job convection-diffusion CSR")
    }

    fn source_fields(method: Method, side: usize) -> Vec<Vec<f64>> {
        let scenarios = method.scenarios();
        let mut fields = Vec::with_capacity(scenarios);
        match method {
            Method::Gmres => {
                for source_row in SOURCE_ROWS {
                    for source_column in SOURCE_COLUMNS {
                        let mut rhs = Vec::with_capacity(side * side);
                        for row in 0..side {
                            let row_weight = 4usize.saturating_sub(row.abs_diff(source_row));
                            for column in 0..side {
                                let column_weight =
                                    4usize.saturating_sub(column.abs_diff(source_column));
                                rhs.push((1 + row_weight * column_weight) as f64 / 16.0);
                            }
                        }
                        fields.push(rhs);
                    }
                }
            }
            Method::Bicgstab => {
                for scenario in 0..BICGSTAB_SCENARIOS {
                    let rhs = (0..side * side)
                        .map(|index| {
                            1.0 + 0.01 * ((index + 7 * scenario) % 17) as f64
                                + 0.0001 * scenario as f64
                        })
                        .collect();
                    fields.push(rhs);
                }
            }
        }
        assert_eq!(fields.len(), scenarios);
        fields
    }

    fn input_sha256(matrix: &CsrMatrix, rhses: &[Vec<f64>], side: usize) -> String {
        let mut hasher = Sha256::new();
        hasher.update((side as u64).to_le_bytes());
        hasher.update((rhses.len() as u64).to_le_bytes());
        for value in matrix.data() {
            hasher.update(value.to_le_bytes());
        }
        for &index in matrix.indices() {
            hasher.update((index as u64).to_le_bytes());
        }
        for &index in matrix.indptr() {
            hasher.update((index as u64).to_le_bytes());
        }
        for rhs in rhses {
            for value in rhs {
                hasher.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn scientific_summaries(
        solutions: &[IterativeSolveResult],
        rhses: &[Vec<f64>],
        side: usize,
    ) -> Vec<f64> {
        let spacing = 1.0 / (side + 1) as f64;
        let cell_area = spacing * spacing;
        let mut summaries = Vec::with_capacity(solutions.len() * SUMMARIES_PER_SCENARIO);
        for (solution, rhs) in solutions.iter().zip(rhses) {
            let inventory = solution.solution.iter().sum::<f64>() * cell_area;
            let outlet = (0..side)
                .map(|row| solution.solution[row * side + side - 1])
                .sum::<f64>()
                * spacing;
            let exposure = solution
                .solution
                .iter()
                .zip(rhs)
                .map(|(value, source)| value * source)
                .sum::<f64>()
                * cell_area;
            summaries.extend([inventory, outlet, exposure]);
        }
        summaries
    }

    struct GmresJobResult {
        solutions: Vec<IterativeSolveResult>,
        summaries: Vec<f64>,
    }

    fn solve_inputs(
        method: Method,
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        postprocess: bool,
    ) -> GmresJobResult {
        let options = IterativeSolveOptions {
            mode: RuntimeMode::Strict,
            check_finite: true,
            tol: RTOL,
            max_iter: Some(10 * SIDE * SIDE),
        };
        let solutions = match method {
            Method::Gmres => gmres_batch(matrix, rhses, None, options),
            Method::Bicgstab => rhses
                .iter()
                .map(|rhs| bicgstab(matrix, rhs, None, options))
                .collect::<Result<Vec<_>, _>>(),
        }
        .expect("FrankenSciPy whole-job iterative batch");
        let summaries = if postprocess {
            scientific_summaries(&solutions, rhses, SIDE)
        } else {
            Vec::new()
        };
        GmresJobResult {
            solutions,
            summaries,
        }
    }

    fn run_whole_job(method: Method) -> GmresJobResult {
        let matrix = convection_diffusion_2d(SIDE);
        let rhses = source_fields(method, SIDE);
        solve_inputs(method, &matrix, &rhses, true)
    }

    fn valid_job(method: Method, result: &GmresJobResult, require_summaries: bool) -> bool {
        let scenarios = method.scenarios();
        result.solutions.len() == scenarios
            && (!require_summaries || result.summaries.len() == scenarios * SUMMARIES_PER_SCENARIO)
            && result.summaries.iter().all(|value| value.is_finite())
            && result.solutions.iter().all(|solution| {
                solution.converged
                    && solution.solution.len() == SIDE * SIDE
                    && solution.iterations > 0
                    && solution.residual_norm.is_finite()
                    && solution.residual_norm <= 1.25 * RTOL
                    && solution.solution.iter().all(|value| value.is_finite())
            })
    }

    fn flatten_fields(result: &GmresJobResult) -> Vec<f64> {
        result
            .solutions
            .iter()
            .flat_map(|solution| solution.solution.iter().copied())
            .collect()
    }

    fn checksum(result: &GmresJobResult) -> u64 {
        result
            .solutions
            .iter()
            .flat_map(|solution| &solution.solution)
            .chain(&result.summaries)
            .fold(0u64, |state, value| state.rotate_left(1) ^ value.to_bits())
    }

    fn maximum_true_residual(
        matrix: &CsrMatrix,
        rhses: &[Vec<f64>],
        result: &GmresJobResult,
    ) -> Result<f64, String> {
        rhses
            .iter()
            .zip(&result.solutions)
            .map(|(rhs, solution)| {
                let ax = matrix
                    .matvec(&solution.solution)
                    .map_err(|error| format!("profile residual matvec: {error}"))?;
                let numerator = rhs
                    .iter()
                    .zip(ax)
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let denominator = rhs.iter().map(|value| value * value).sum::<f64>().sqrt();
                Ok(numerator / denominator.max(f64::EPSILON))
            })
            .try_fold(0.0f64, |maximum, residual| {
                residual.map(|value| maximum.max(value))
            })
    }

    fn run_bicgstab_batch_profile(repetitions: usize) -> Result<(), String> {
        if repetitions == 0 {
            return Err("BiCGSTAB batch profile repetitions must be positive".to_string());
        }
        let affinity = cpu_affinity()?;
        if affinity_cpu_count(&affinity)? != 1 {
            return Err("pin the untouched BiCGSTAB batch profile to one CPU".to_string());
        }
        if observed_os_threads()? != 1 {
            return Err("BiCGSTAB batch profile started with more than one thread".to_string());
        }

        let method = Method::Bicgstab;
        let matrix = convection_diffusion_2d(SIDE);
        let rhses = source_fields(method, SIDE);
        let input_sha256 = input_sha256(&matrix, &rhses, SIDE);
        let warm = solve_inputs(method, &matrix, &rhses, true);
        if !valid_job(method, &warm, true) {
            let failures = warm
                .solutions
                .iter()
                .enumerate()
                .filter(|(_, solution)| {
                    !solution.converged
                        || solution.iterations == 0
                        || !solution.residual_norm.is_finite()
                        || solution.residual_norm > 1.25 * RTOL
                })
                .map(|(index, solution)| {
                    format!(
                        "{index}:converged={},iterations={},residual={:.3e}",
                        solution.converged, solution.iterations, solution.residual_norm
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            return Err(format!(
                "untouched serial BiCGSTAB profile failed convergence: {failures}"
            ));
        }
        let maximum_residual = maximum_true_residual(&matrix, &rhses, &warm)?;
        let mut folded = checksum(&warm);
        let started = Instant::now();
        for _ in 0..repetitions {
            let result = black_box(solve_inputs(method, &matrix, &rhses, true));
            folded = folded.rotate_left(1) ^ checksum(&result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(folded);
        if observed_os_threads()? != 1 {
            return Err("untouched BiCGSTAB profile leaked worker threads".to_string());
        }
        println!("elf_sha256={}", sha256_of_self()?);
        println!(
            "BICGSTAB_BATCH_PROFILE method=bicgstab side={SIDE} n={} nnz={} \
             scenarios={} repetitions={repetitions} elapsed_seconds={elapsed:.9} \
             maximum_true_residual={maximum_residual:.17e} checksum={folded} \
             actual_observed_worker_threads=1 cpu_affinity={affinity} \
             input_sha256={input_sha256}",
            SIDE * SIDE,
            matrix.nnz(),
            method.scenarios()
        );
        Ok(())
    }

    fn time_ours_whole_job(method: Method, repetitions: usize) -> Result<f64, String> {
        let mut result = None;
        let started = Instant::now();
        for _ in 0..repetitions {
            result = Some(black_box(run_whole_job(method)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let result = result.ok_or_else(|| "whole-job repetitions must be positive".to_string())?;
        if !valid_job(method, &result, true) {
            return Err("timed FrankenSciPy whole job was incomplete".to_string());
        }
        black_box(checksum(&result));
        Ok(elapsed)
    }

    fn time_ours_solve_only(method: Method, repetitions: usize) -> Result<f64, String> {
        let matrix = convection_diffusion_2d(SIDE);
        let rhses = source_fields(method, SIDE);
        let mut result = None;
        let started = Instant::now();
        for _ in 0..repetitions {
            result = Some(black_box(solve_inputs(method, &matrix, &rhses, false)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let result = result.ok_or_else(|| "solve-only repetitions must be positive".to_string())?;
        if !valid_job(method, &result, false) {
            return Err("timed FrankenSciPy solve-only job was incomplete".to_string());
        }
        black_box(checksum(&result));
        Ok(elapsed)
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

    fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            return f64::NAN;
        }
        let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    fn boot_ci(values: &[f64]) -> (f64, f64) {
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

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().saturating_sub(1).max(1) as f64;
        variance.sqrt() / mean
    }

    fn incumbent_pair(
        scipy: &mut Scipy,
        method: Method,
        configuration: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_ours_whole_job(method, repetitions)?,
                scipy.job_time(configuration, repetitions)?.elapsed,
            ))
        } else {
            let incumbent = scipy.job_time(configuration, repetitions)?.elapsed;
            let ours = time_ours_whole_job(method, repetitions)?;
            Ok((ours, incumbent))
        }
    }

    fn ours_null_pair(method: Method, repetitions: usize, round: usize) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_ours_whole_job(method, repetitions)?,
                time_ours_whole_job(method, repetitions)?,
            )
        } else {
            let right = time_ours_whole_job(method, repetitions)?;
            let left = time_ours_whole_job(method, repetitions)?;
            (left, right)
        };
        Ok(left / right)
    }

    fn scipy_null_pair(
        scipy: &mut Scipy,
        configuration: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                scipy.job_time(configuration, repetitions)?.elapsed,
                scipy.job_time(configuration, repetitions)?.elapsed,
            )
        } else {
            let right = scipy.job_time(configuration, repetitions)?.elapsed;
            let left = scipy.job_time(configuration, repetitions)?.elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    #[derive(Clone)]
    struct Measurement {
        ours: Vec<f64>,
        scipy: Vec<f64>,
        ratios: Vec<f64>,
        ours_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
    }

    struct Decision {
        outcome: &'static str,
        ratio_median: f64,
        ratio_low: f64,
        ratio_high: f64,
        ours_p50: f64,
        scipy_p50: f64,
    }

    fn measure_configuration(
        scipy: &mut Scipy,
        method: Method,
        configuration: &str,
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = incumbent_pair(scipy, method, configuration, repetitions, warmup)?;
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
                    let pair = incumbent_pair(scipy, method, configuration, repetitions, round)?;
                    let ours_null = ours_null_pair(method, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    let pair = incumbent_pair(scipy, method, configuration, repetitions, round)?;
                    let ours_null = ours_null_pair(method, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(method, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    let pair = incumbent_pair(scipy, method, configuration, repetitions, round)?;
                    (pair.0, pair.1, ours_null, scipy_null)
                }
            };
            measurement.ours.push(ours);
            measurement.scipy.push(incumbent);
            measurement.ratios.push(incumbent / ours);
            measurement.ours_nulls.push(ours_null);
            measurement.scipy_nulls.push(scipy_null);
        }
        Ok(measurement)
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn print_measurement(
        label: &str,
        configuration: &str,
        measurement: &Measurement,
        repetitions: usize,
        headline: bool,
    ) -> Decision {
        let (ratio_low, ratio_high) = boot_ci(&measurement.ratios);
        let (ours_null_low, ours_null_high) = boot_ci(&measurement.ours_nulls);
        let (scipy_null_low, scipy_null_high) = boot_ci(&measurement.scipy_nulls);
        let ours_p50 = median(measurement.ours.clone()) / repetitions as f64;
        let scipy_p50 = median(measurement.scipy.clone()) / repetitions as f64;
        let ours_p95 = percentile(measurement.ours.clone(), 0.95) / repetitions as f64;
        let scipy_p95 = percentile(measurement.scipy.clone(), 0.95) / repetitions as f64;
        let ours_p99 = percentile(measurement.ours.clone(), 0.99) / repetitions as f64;
        let scipy_p99 = percentile(measurement.scipy.clone(), 0.99) / repetitions as f64;
        println!(
            "{label}_whole_job_wall: configuration={configuration} \
             FrankenSciPy p50={:.6}ms p95={:.6}ms p99={:.6}ms | \
             SciPy p50={:.6}ms p95={:.6}ms p99={:.6}ms",
            ours_p50 * 1e3,
            ours_p95 * 1e3,
            ours_p99 * 1e3,
            scipy_p50 * 1e3,
            scipy_p95 * 1e3,
            scipy_p99 * 1e3
        );
        println!(
            "{label}_raw_samples_seconds: ours={} scipy={} ratios={} \
             null_ours={} null_scipy={}",
            csv(&measurement.ours),
            csv(&measurement.scipy),
            csv(&measurement.ratios),
            csv(&measurement.ours_nulls),
            csv(&measurement.scipy_nulls)
        );
        let ours_null_median = median(measurement.ours_nulls.clone());
        let scipy_null_median = median(measurement.scipy_nulls.clone());
        println!(
            "{label}_NULL-ours A/A median={ours_null_median:.6} \
             ci95=[{ours_null_low:.6},{ours_null_high:.6}] cv={:.3}% \
             (provenance only)",
            cv(&measurement.ours_nulls) * 100.0
        );
        println!(
            "{label}_NULL-scipy A/A median={scipy_null_median:.6} \
             ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] cv={:.3}% \
             (provenance only)",
            cv(&measurement.scipy_nulls) * 100.0
        );
        let ratio_median = median(measurement.ratios.clone());
        if headline {
            println!(
                "Incumbent ratio: SciPy / FrankenSciPy = {ratio_median:.4}x \
                 (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
                 cv={:.3}% provenance only)",
                cv(&measurement.ratios) * 100.0
            );
        } else {
            println!(
                "Unpreconditioned ratio: SciPy / FrankenSciPy = {ratio_median:.4}x \
                 (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
                 cv={:.3}% provenance only)",
                cv(&measurement.ratios) * 100.0
            );
        }

        let null_half_width = ((ours_null_high - ours_null_low) / 2.0)
            .max((scipy_null_high - scipy_null_low) / 2.0)
            .max(0.0);
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1e-9))
            .max(1.0 / scipy_null_low.max(1e-9))
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
            "{label}_corrected-null-gate: c1_effect_ci_excludes_one={c1} \
             c2_beats_2x_half_width={c2} c2b_beats_2x_endpoint={c2b} \
             c3_null_medians_within_2pct={c3} decidable={decidable} \
             effect_deviation={effect_deviation:.6} \
             null_half_width={null_half_width:.6} \
             required_c2={:.6} required_c2b={:.6} \
             ours_null_median={ours_null_median:.6} \
             scipy_null_median={scipy_null_median:.6} \
             null_ci_veto=disabled_telemetry_only",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!(
            "{label}_median-CI gate: worst_null_edge={null_edge:.4} \
             required={:.4} ratio_ci=[{ratio_low:.4},{ratio_high:.4}] \
             null_margin=2x cv_used_for_decision=false => {outcome}",
            1.0 + 2.0 * (null_edge - 1.0)
        );
        Decision {
            outcome,
            ratio_median,
            ratio_low,
            ratio_high,
            ours_p50,
            scipy_p50,
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
            let cpu = parse(suffix, "CPU index")?;
            let ticks = fields
                .map(|field| parse::<u64>(field, "CPU tick"))
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("CPU {cpu} has an incomplete /proc/stat row"));
            }
            let total = ticks.iter().sum();
            let idle = ticks[3].saturating_add(ticks[4]);
            cpus.insert(cpu, CpuTicks { total, idle });
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
            return Err("CPU topology changed during host-wide load sample".to_string());
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
                return Err(format!("CPU {cpu} accumulated no ticks during load sample"));
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
                "host-wide quiescence {phase} failed: {} CPUs exceeded {:.0}% busy \
                 (maximum {:.1}%): {detail}",
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

    fn observed_os_threads() -> Result<usize, String> {
        std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("read /proc/self/task: {error}"))
            .map(Iterator::count)
    }

    fn affinity_cpu_count(affinity: &str) -> Result<usize, String> {
        let mut cpus = HashSet::new();
        for segment in affinity.split(',') {
            if let Some((start, end)) = segment.split_once('-') {
                let start = parse::<usize>(start, "affinity range start")?;
                let end = parse::<usize>(end, "affinity range end")?;
                if end < start {
                    return Err(format!("invalid CPU affinity range {segment}"));
                }
                cpus.extend(start..=end);
            } else {
                cpus.insert(parse::<usize>(segment, "affinity CPU")?);
            }
        }
        Ok(cpus.len())
    }

    fn observed_peak_worker_threads<T>(work: impl FnOnce() -> T + Send) -> (T, usize)
    where
        T: Send,
    {
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let watcher = scope.spawn(|| {
                let mut peak = 1usize;
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
                        peak = peak.max(entries.count());
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
                peak
            });
            let value = work();
            done.store(true, std::sync::atomic::Ordering::Relaxed);
            // Exclude the watcher and main thread; keep one as the serial floor.
            let workers = watcher.join().unwrap_or(1).saturating_sub(2).max(1);
            (value, workers)
        })
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
            if processor.is_some() {
                logical += 1;
                pairs.insert((
                    physical.unwrap_or("0").to_string(),
                    core.unwrap_or_else(|| processor.expect("processor exists"))
                        .to_string(),
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

    fn print_hardware_provenance(affinity: &str, affinity_cpus: usize) -> Result<(), String> {
        let (physical_cores, logical_threads) = cpu_topology()?;
        let cpu = affinity
            .split(',')
            .next()
            .and_then(|segment| segment.split('-').next())
            .ok_or_else(|| "CPU affinity set is empty".to_string())?;
        let driver = read_policy_field(cpu, "scaling_driver")?;
        let governor = read_policy_field(cpu, "scaling_governor")?;
        let preference = read_policy_field(cpu, "energy_performance_preference")
            .unwrap_or_else(|_| "unavailable".to_string());
        let minimum = read_policy_field(cpu, "scaling_min_freq")?;
        let maximum = read_policy_field(cpu, "scaling_max_freq")?;
        println!(
            "hardware_provenance: host_identity={} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={} numa_nodes={} \
             runtime_detected_isa={} affinity={affinity} \
             cpuset_logical_cap={affinity_cpus}",
            host_identity()?,
            ram_bytes()?,
            numa_node_count()?,
            runtime_isa_features()
        );
        println!(
            "cpu_frequency_policy: cpu={cpu} scaling_driver={driver} \
             scaling_governor={governor} \
             energy_performance_preference={preference} \
             scaling_min_freq_khz={minimum} scaling_max_freq_khz={maximum}"
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

    fn scipy_oracle_script(argument: Option<&String>) -> Result<PathBuf, String> {
        if let Some(argument) = argument {
            let path = PathBuf::from(argument);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!(
                "explicit SciPy whole-job oracle is unavailable: {}",
                path.display()
            ));
        }
        let compiled =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/scipy_gmres_job_arm.py");
        if compiled.is_file() {
            return Ok(compiled);
        }
        let workspace_relative = PathBuf::from("crates/fsci-sparse/python/scipy_gmres_job_arm.py");
        if workspace_relative.is_file() {
            return Ok(workspace_relative);
        }
        Err("SciPy whole-job GMRES oracle is unavailable".to_string())
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("{name} is required for whole-job provenance"))
    }

    struct Proof {
        eligible: bool,
        max_abs_difference: f64,
        relative_l2_difference: f64,
        maximum_scaled_difference: f64,
        tolerance_mismatches: usize,
        maximum_summary_scaled_difference: f64,
    }

    fn prove_candidate(
        method: Method,
        ours: &GmresJobResult,
        ours_fields: &[f64],
        expected_input_sha256: &str,
        check: &ScipyJobCheck,
    ) -> Proof {
        let scenarios = method.scenarios();
        let expected_components = scenarios * SIDE * SIDE;
        let expected_summaries = scenarios * SUMMARIES_PER_SCENARIO;
        let structural = check.configuration.as_str() != ""
            && check.successes == scenarios
            && check.input_sha256 == expected_input_sha256
            && check.observed_threads == 1
            && check.infos.len() == scenarios
            && check.infos.iter().all(|info| *info == 0)
            && check.iterations.len() == scenarios
            && check.iterations.iter().all(|iterations| *iterations > 0)
            && check.residuals.len() == scenarios
            && check
                .residuals
                .iter()
                .all(|value| value.is_finite() && *value <= 1.25 * RTOL)
            && ours_fields.len() == expected_components
            && check.fields.len() == expected_components
            && ours.summaries.len() == expected_summaries
            && check.summaries.len() == expected_summaries;

        let mut max_abs_difference = 0.0f64;
        let mut maximum_scaled_difference = 0.0f64;
        let mut tolerance_mismatches = 0usize;
        let mut difference_squared = 0.0f64;
        let mut scipy_squared = 0.0f64;
        if ours_fields.len() == check.fields.len() {
            for (&left, &right) in ours_fields.iter().zip(&check.fields) {
                let difference = (left - right).abs();
                let tolerance = 10.0 * RTOL * right.abs().max(1.0);
                max_abs_difference = max_abs_difference.max(difference);
                maximum_scaled_difference = maximum_scaled_difference.max(difference / tolerance);
                tolerance_mismatches +=
                    usize::from(!difference.is_finite() || difference > tolerance);
                difference_squared += difference * difference;
                scipy_squared += right * right;
            }
        } else {
            tolerance_mismatches = expected_components;
        }
        let relative_l2_difference =
            difference_squared.sqrt() / scipy_squared.sqrt().max(f64::EPSILON);
        let mut maximum_summary_scaled_difference = 0.0f64;
        if ours.summaries.len() == check.summaries.len() {
            for (&left, &right) in ours.summaries.iter().zip(&check.summaries) {
                let tolerance = 10.0 * RTOL * right.abs().max(1.0);
                maximum_summary_scaled_difference =
                    maximum_summary_scaled_difference.max((left - right).abs() / tolerance);
            }
        } else {
            maximum_summary_scaled_difference = f64::INFINITY;
        }
        let eligible = structural
            && tolerance_mismatches == 0
            && relative_l2_difference.is_finite()
            && relative_l2_difference <= 5.0e-4
            && maximum_summary_scaled_difference.is_finite()
            && maximum_summary_scaled_difference <= 1.0;
        Proof {
            eligible,
            max_abs_difference,
            relative_l2_difference,
            maximum_scaled_difference,
            tolerance_mismatches,
            maximum_summary_scaled_difference,
        }
    }

    struct Candidate {
        configuration: &'static str,
        check: ScipyJobCheck,
        screen_elapsed: f64,
    }

    fn diagnostic_decomposition(
        scipy: &mut Scipy,
        method: Method,
        configuration: &str,
        whole_ours_p50: f64,
        whole_scipy_p50: f64,
    ) -> Result<(), String> {
        let mut ours = Vec::with_capacity(5);
        let mut incumbent = Vec::with_capacity(5);
        for round in 0_usize..5 {
            if round.is_multiple_of(2) {
                ours.push(time_ours_solve_only(method, 1)?);
                incumbent.push(scipy.solve_only_time(configuration, 1)?.elapsed);
            } else {
                incumbent.push(scipy.solve_only_time(configuration, 1)?.elapsed);
                ours.push(time_ours_solve_only(method, 1)?);
            }
        }
        let ours_solve = median(ours);
        let scipy_solve = median(incumbent);
        let ours_non_solver_fraction =
            ((whole_ours_p50 - ours_solve).max(0.0) / whole_ours_p50).min(1.0);
        let scipy_non_solver_fraction =
            ((whole_scipy_p50 - scipy_solve).max(0.0) / whole_scipy_p50).min(1.0);
        println!(
            "decomposition: selected_configuration={configuration} \
             ours_solve_only_p50={:.6}ms scipy_solve_only_p50={:.6}ms \
             ours_non_solver_fraction={:.2}% scipy_non_solver_fraction={:.2}% \
             p5_both_under_25pct={}",
            ours_solve * 1e3,
            scipy_solve * 1e3,
            ours_non_solver_fraction * 100.0,
            scipy_non_solver_fraction * 100.0,
            ours_non_solver_fraction < 0.25 && scipy_non_solver_fraction < 0.25
        );
        Ok(())
    }

    pub fn run() -> Result<(), String> {
        let arguments = std::env::args().collect::<Vec<_>>();
        if arguments.get(1).map(String::as_str) == Some("--profile-bicgstab-batch") {
            let repetitions = arguments
                .get(2)
                .map(|value| parse::<usize>(value, "profile repetitions"))
                .transpose()?
                .unwrap_or(10);
            return run_bicgstab_batch_profile(repetitions);
        }
        let method = Method::parse(arguments.get(3).map_or("gmres", String::as_str))?;
        let scenarios = method.scenarios();
        let sequential_control = std::env::var_os("FSCI_GMRES_BATCH_FORCE_SEQUENTIAL").is_some();
        GMRES_BATCH_FORCE_SEQUENTIAL
            .store(sequential_control, std::sync::atomic::Ordering::Relaxed);
        let rounds = arguments
            .get(1)
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(11);
        if rounds < 7 {
            return Err("whole-job median-CI gate requires rounds>=7".to_string());
        }
        let repetitions = 1usize;
        let elf_sha256 = sha256_of_self()?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!(
            "binary_provenance: builder_identity={builder_identity} \
             source_commit={source_commit} build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");

        let affinity = cpu_affinity()?;
        let affinity_cpus = affinity_cpu_count(&affinity)?;
        if affinity_cpus == 0 {
            return Err("CPU affinity set is empty".to_string());
        }
        println!("cpu_affinity={affinity}");
        println!(
            "iterative_batch_scheduler={} method={} affinity_cpu_count={affinity_cpus}",
            if method == Method::Gmres && sequential_control {
                "same-elf-sequential-control"
            } else if method == Method::Bicgstab {
                "untouched-serial-public-solves"
            } else {
                "shared-nothing-auto"
            },
            method.label()
        );
        print_hardware_provenance(&affinity, affinity_cpus)?;
        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        require_host_wide_quiescence("pre")?;

        println!(
            "fixture=steady-convection-diffusion-source-screen side={SIDE} n={} \
             scenarios={scenarios} rounds={rounds} repetitions={repetitions} \
             method={} restart={} rtol={RTOL} atol=0 maxiter={} x0=zeros \
             diagonal={DIAGONAL} west={WEST} east={EAST} vertical={VERTICAL} \
             source_layout={} source_radius=4 \
             requested_frankenscipy_threads=auto requested_scipy_threads=1",
            SIDE * SIDE,
            method.display(),
            if method == Method::Gmres {
                "20"
            } else {
                "none"
            },
            10 * SIDE * SIDE,
            if method == Method::Gmres {
                "rows=6,16,25;columns=5,12,20,27"
            } else {
                "dense=1+0.01*((index+7*scenario)%17)+0.0001*scenario"
            },
        );
        println!(
            "whole_job_boundary: INCLUDED=operator_assembly,{scenarios}_source_fields,\
             selected_preconditioner_construction,{scenarios}_public_{}_solves,\
             {}_field_values,domain_inventory,east_outlet_integral,\
             source_weighted_exposure; EXCLUDED=python_interpreter_startup,\
             scipy_import,pipe_transport,backend_screening,parity_serialization,\
             provenance_collection,bootstrap_calculation",
            method.label(),
            scenarios * SIDE * SIDE
        );
        println!(
            "work_units: operator_nnz={} scenario_solves={scenarios} \
             materialized_field_values={} scientific_summaries={}",
            5 * SIDE * SIDE - 4 * SIDE,
            scenarios * SIDE * SIDE,
            scenarios * SUMMARIES_PER_SCENARIO
        );

        let matrix = convection_diffusion_2d(SIDE);
        let rhses = source_fields(method, SIDE);
        let expected_input_sha256 = input_sha256(&matrix, &rhses, SIDE);
        let (ours, ours_threads) =
            observed_peak_worker_threads(|| solve_inputs(method, &matrix, &rhses, true));
        if ours_threads > affinity_cpus || !valid_job(method, &ours, true) {
            return Err("FrankenSciPy whole-job parity arm was inadmissible".to_string());
        }
        let ours_fields = flatten_fields(&ours);

        let script = scipy_oracle_script(arguments.get(2))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let (mut scipy, identity) = Scipy::start(&script, method)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains(&format!("method={}", method.label()))
            || !identity.contains("solver_mod=scipy.sparse.linalg._isolve")
            || !identity.contains("actual_observed_worker_threads=1")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy arm failed genuine 1.17.1 identity gate".to_string());
        }
        let scipy_engine_sha256 = ready_value(&identity, "scipy_engine_sha256=")
            .ok_or_else(|| "SciPy arm omitted GMRES engine SHA-256".to_string())?;
        let spilu_source_sha256 = ready_value(&identity, "spilu_source_sha256=")
            .ok_or_else(|| "SciPy arm omitted spilu source SHA-256".to_string())?;
        let superlu_engine_sha256 = ready_value(&identity, "superlu_engine_sha256=")
            .ok_or_else(|| "SciPy arm omitted SuperLU engine SHA-256".to_string())?;
        if !is_sha256(scipy_engine_sha256)
            || !is_sha256(spilu_source_sha256)
            || !is_sha256(superlu_engine_sha256)
        {
            return Err("live SciPy arm reported an invalid engine SHA-256".to_string());
        }
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!("spilu_source_sha256={spilu_source_sha256}");
        println!("superlu_engine_sha256={superlu_engine_sha256}");
        println!(
            "Legacy incumbent arm: SciPy 1.17.1; side-by-side same-invocation; \
             strongest valid public {} configuration selected live",
            method.display()
        );
        println!(
            "thread_provenance: requested_frankenscipy_threads=auto \
             actual_observed_frankenscipy_worker_threads={ours_threads} \
             requested_scipy_threads=1 actual_observed_scipy_worker_threads=1 \
             python_blas_thread_cap=1"
        );
        println!(
            "incumbent_backend_screen_contract: method={} configurations={}; \
             selection=lowest_valid_live_whole_job_wall_time; \
             full_output_eligibility_before_selection=true \
             screen_outside_headline_samples=true",
            method.label(),
            CONFIGURATIONS.join(",")
        );

        let mut candidates = Vec::with_capacity(CONFIGURATIONS.len());
        for configuration in CONFIGURATIONS {
            let check = scipy.job_check(configuration)?;
            let proof =
                prove_candidate(method, &ours, &ours_fields, &expected_input_sha256, &check);
            println!(
                "scipy_backend_proof: configuration={configuration} \
                 eligible={} successes={}/{} input_sha_match={} \
                 max_abs_diff={:.3e} relative_l2_diff={:.3e} \
                 maximum_scaled_diff={:.3e} tolerance_mismatches={} \
                 maximum_summary_scaled_diff={:.3e} \
                 iterations={}",
                proof.eligible,
                check.successes,
                scenarios,
                check.input_sha256 == expected_input_sha256,
                proof.max_abs_difference,
                proof.relative_l2_difference,
                proof.maximum_scaled_difference,
                proof.tolerance_mismatches,
                proof.maximum_summary_scaled_difference,
                check
                    .iterations
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            if proof.eligible {
                let screen_elapsed = scipy.job_time(configuration, 1)?.elapsed;
                println!(
                    "scipy_backend_screen: configuration={configuration} \
                     whole_job_ms={:.6}",
                    screen_elapsed * 1e3
                );
                candidates.push(Candidate {
                    configuration,
                    check,
                    screen_elapsed,
                });
            } else {
                println!(
                    "scipy_backend_screen: configuration={configuration} \
                     ineligible_full_result_contract=true"
                );
            }
        }
        if candidates.is_empty() {
            return Err(format!(
                "no SciPy {} configuration passed full-result eligibility",
                method.display()
            ));
        }
        candidates.sort_by(|left, right| left.screen_elapsed.total_cmp(&right.screen_elapsed));
        let selected = &candidates[0];
        let unpreconditioned = candidates
            .iter()
            .filter(|candidate| candidate.configuration.ends_with("-none"))
            .min_by(|left, right| left.screen_elapsed.total_cmp(&right.screen_elapsed))
            .ok_or_else(|| "no unpreconditioned SciPy configuration was eligible".to_string())?;
        println!(
            "selected_scipy_incumbent: configuration={} screened_whole_job_ms={:.6} \
             reason=fastest_valid_live_SciPy_{}_configuration",
            selected.configuration,
            selected.screen_elapsed * 1e3,
            method.display()
        );
        println!(
            "selected_unpreconditioned_scipy: configuration={} \
             screened_whole_job_ms={:.6}",
            unpreconditioned.configuration,
            unpreconditioned.screen_elapsed * 1e3
        );

        let ours_iterations = ours
            .solutions
            .iter()
            .map(|solution| solution.iterations)
            .collect::<Vec<_>>();
        let unpreconditioned_count_parity = ours_iterations == unpreconditioned.check.iterations;
        println!(
            "operation_counts: ours_iterations={} | \
             unpreconditioned_scipy_iterations={} exact_scenario_parity={} | \
             selected_scipy_configuration={} selected_scipy_iterations={}",
            ours_iterations
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            unpreconditioned
                .check
                .iterations
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            unpreconditioned_count_parity,
            selected.configuration,
            selected
                .check
                .iterations
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "agreement: scenarios={scenarios}/{scenarios} \
             compared_field_values={}/{} compared_summaries={}/{} \
             input_sha256={expected_input_sha256}",
            ours_fields.len(),
            scenarios * SIDE * SIDE,
            ours.summaries.len(),
            scenarios * SUMMARIES_PER_SCENARIO
        );

        require_host_wide_quiescence("measurement")?;
        let unpreconditioned_measurement = measure_configuration(
            &mut scipy,
            method,
            unpreconditioned.configuration,
            rounds,
            repetitions,
        )?;
        let headline_measurement = if selected.configuration == unpreconditioned.configuration {
            unpreconditioned_measurement.clone()
        } else {
            measure_configuration(
                &mut scipy,
                method,
                selected.configuration,
                rounds,
                repetitions,
            )?
        };
        require_host_wide_quiescence("post")?;

        let unpreconditioned_decision = print_measurement(
            "unpreconditioned",
            unpreconditioned.configuration,
            &unpreconditioned_measurement,
            repetitions,
            false,
        );
        let headline_decision = print_measurement(
            "headline",
            selected.configuration,
            &headline_measurement,
            repetitions,
            true,
        );
        diagnostic_decomposition(
            &mut scipy,
            method,
            selected.configuration,
            headline_decision.ours_p50,
            headline_decision.scipy_p50,
        )?;
        let durable_three_x = headline_decision.outcome == "DECIDED FRANKENSCIPY WIN"
            && headline_decision.ratio_low >= 3.0;
        println!(
            "prediction_scorecard: p1_unpreconditioned_ci_low_ge_3={} \
             p2_exact_unpreconditioned_iteration_parity={} \
             p3_spilu_selected={} p4_headline_unpreconditioned_3x_does_not_survive={} \
             durable_three_x_whole_job_claim={durable_three_x} \
             unpreconditioned_ratio={:.4} headline_ratio={:.4} \
             headline_ci=[{:.4},{:.4}]",
            unpreconditioned_decision.ratio_low >= 3.0,
            unpreconditioned_count_parity,
            selected.configuration == "csc-matrix-spilu",
            !durable_three_x,
            unpreconditioned_decision.ratio_median,
            headline_decision.ratio_median,
            headline_decision.ratio_low,
            headline_decision.ratio_high
        );
        println!(
            "scope_guard: this result applies only to the serial 32x32, \
             twelve-source, steady convection-diffusion GMRES job; direct sparse \
             solvers, other Krylov methods, other preconditioners, matrices, \
             sizes, and thread counts are not decided"
        );
        if durable_three_x {
            println!(
                "CHOOSER STATEMENT: Pick FrankenSciPy GMRES for this serial 32x32 \
                 twelve-source steady convection-diffusion job: it clears a durable \
                 3x whole-job threshold against the fastest screened valid SciPy \
                 1.17.1 configuration ({}). Pick SciPy for the separately measured \
                 unpreconditioned side-96/n=9216 shape, where FrankenSciPy loses; \
                 benchmark direct solvers and every unmeasured shape.",
                selected.configuration
            );
        } else if headline_decision.outcome == "DECIDED FRANKENSCIPY LOSS" {
            println!(
                "CHOOSER STATEMENT: Pick SciPy 1.17.1 GMRES with {} for this serial \
                 32x32 twelve-source steady convection-diffusion job; the strongest \
                 valid incumbent reverses the favorable unpreconditioned kernel \
                 result. SciPy also wins the separately measured unpreconditioned \
                 side-96/n=9216 shape. Direct sparse solvers and every other matrix, \
                 size, preconditioner, and thread count remain unmeasured.",
                selected.configuration
            );
        } else if headline_decision.outcome == "DECIDED FRANKENSCIP WIN" {
            println!(
                "CHOOSER STATEMENT: Pick FrankenSciPy GMRES for this exact serial \
                 32x32 twelve-source job, but do not call it a durable 3x claim: the \
                 fastest screened SciPy 1.17.1 configuration ({}) narrows the win \
                 below that threshold. Pick SciPy for the separately measured \
                 unpreconditioned side-96/n=9216 shape; benchmark every other shape.",
                selected.configuration
            );
        } else {
            println!(
                "CHOOSER STATEMENT: This run does not distinguish FrankenSciPy from \
                 the fastest screened SciPy 1.17.1 GMRES configuration beyond its \
                 corrected A/A noise gate. Pick SciPy for the separately measured \
                 unpreconditioned side-96/n=9216 shape and benchmark every other \
                 matrix, size, preconditioner, and solver family."
            );
        }
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
    eprintln!("perf_gmres_job_vs_scipy requires --features sparse-incumbent-bench");
    std::process::exit(2);
}
