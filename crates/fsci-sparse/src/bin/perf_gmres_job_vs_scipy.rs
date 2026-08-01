//! Persistent-operator convection-diffusion GMRES job versus live SciPy.
//!
//! One operator and one ILU factor serve twelve localized source fields. Every
//! timed repetition runs the twelve public GMRES solves, materializes every
//! field, and computes three scientific summaries per field. Operator, source,
//! and preconditioner construction are measured separately.
//!
//! Run:
//! `perf_gmres_job_vs_scipy [rounds] [scipy_gmres_job_arm.py]`

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::linalg::{
        GMRES_BATCH_FORCE_SEQUENTIAL, GMRES_PRECONDITIONED_FUSED_DISABLE,
        GMRES_PRECONDITIONED_FUSED_HITS, IterativeSolveOptions, gmres_batch_preconditioned, spilu,
    };
    use fsci_sparse::{
        CsrMatrix, FormatConvertible, IluOptions, IterativeSolveResult, Shape2D,
        SparseIluFactorization,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};

    const SIDE: usize = 192;
    const SCENARIOS: usize = 12;
    const SUMMARIES_PER_SCENARIO: usize = 3;
    const DIAGONAL: f64 = 4.001;
    const WEST: f64 = -1.2;
    const EAST: f64 = -0.8;
    const VERTICAL: f64 = -1.0;
    const RTOL: f64 = 1.0e-5;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_secs(1);
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const HOST_PREFLIGHT_ATTEMPTS: usize = 12;
    const SOURCE_ROWS: [usize; 3] = [36, 96, 150];
    const SOURCE_COLUMNS: [usize; 4] = [30, 72, 120, 162];
    const CONFIGURATIONS: [&str; 6] = [
        "csr-matrix-none",
        "csr-array-none",
        "csc-matrix-none",
        "csc-array-none",
        "csr-matrix-jacobi",
        "csc-matrix-spilu",
    ];

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
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
        fn start(script: &Path) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--live")
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
                || successes != SCENARIOS
                || components != SCENARIOS * SIDE * SIDE
                || summaries != SCENARIOS * SUMMARIES_PER_SCENARIO
                || threads != 1
                || !checksum.is_finite()
            {
                return Err(format!("inadmissible timed SciPy whole job: {reply}"));
            }
            Ok(ScipyJobTiming { elapsed })
        }

        fn persistent_job_time(
            &mut self,
            configuration: &str,
            repetitions: usize,
        ) -> Result<ScipyJobTiming, String> {
            writeln!(
                self.stdin,
                "JOB_SOLVE_ONLY_TIME {configuration} {SIDE} {repetitions}"
            )
            .map_err(|error| format!("write JOB_SOLVE_ONLY_TIME: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush JOB_SOLVE_ONLY_TIME: {error}"))?;
            let reply = self.read_reply("SciPy persistent-job timing")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 7 || fields[0] != "JOB_SOLVE_ONLY_TIME" {
                return Err(format!("invalid SciPy persistent-job timing: {reply}"));
            }
            let elapsed: f64 = parse(fields[1], "SciPy persistent-job elapsed")?;
            let successes: usize = parse(fields[2], "SciPy persistent-job successes")?;
            let components: usize = parse(fields[3], "SciPy persistent-job components")?;
            let summaries: usize = parse(fields[4], "SciPy persistent-job summaries")?;
            let threads: usize = parse(fields[5], "SciPy persistent-job threads")?;
            let checksum: f64 = parse(fields[6], "SciPy persistent-job checksum")?;
            black_box(checksum);
            if !elapsed.is_finite()
                || elapsed <= 0.0
                || successes != SCENARIOS
                || components != SCENARIOS * SIDE * SIDE
                || summaries != SCENARIOS * SUMMARIES_PER_SCENARIO
                || threads != 1
                || !checksum.is_finite()
            {
                return Err(format!("inadmissible SciPy persistent-job timing: {reply}"));
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

    fn source_fields(side: usize) -> Vec<Vec<f64>> {
        let mut fields = Vec::with_capacity(SCENARIOS);
        for source_row in SOURCE_ROWS {
            for source_column in SOURCE_COLUMNS {
                let mut rhs = Vec::with_capacity(side * side);
                for row in 0..side {
                    let row_weight = 4usize.saturating_sub(row.abs_diff(source_row));
                    for column in 0..side {
                        let column_weight = 4usize.saturating_sub(column.abs_diff(source_column));
                        rhs.push((1 + row_weight * column_weight) as f64 / 16.0);
                    }
                }
                fields.push(rhs);
            }
        }
        assert_eq!(fields.len(), SCENARIOS);
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
        let mut summaries = Vec::with_capacity(SCENARIOS * SUMMARIES_PER_SCENARIO);
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

    struct PersistentJob {
        matrix: CsrMatrix,
        rhses: Vec<Vec<f64>>,
        preconditioner: SparseIluFactorization,
    }

    fn build_persistent_job() -> Result<PersistentJob, String> {
        let matrix = convection_diffusion_2d(SIDE);
        let rhses = source_fields(SIDE);
        let csc = matrix
            .to_csc()
            .map_err(|error| format!("persistent-job CSC conversion: {error}"))?;
        let preconditioner = spilu(&csc, IluOptions::default())
            .map_err(|error| format!("persistent-job ILU(0): {error}"))?;
        Ok(PersistentJob {
            matrix,
            rhses,
            preconditioner,
        })
    }

    fn solve_inputs(job: &PersistentJob) -> GmresJobResult {
        let solutions = gmres_batch_preconditioned(
            &job.matrix,
            &job.rhses,
            &job.preconditioner,
            None,
            IterativeSolveOptions {
                mode: RuntimeMode::Strict,
                check_finite: true,
                tol: RTOL,
                max_iter: Some(10 * SIDE * SIDE),
            },
        )
        .expect("FrankenSciPy persistent preconditioned GMRES batch");
        let summaries = scientific_summaries(&solutions, &job.rhses, SIDE);
        GmresJobResult {
            solutions,
            summaries,
        }
    }

    fn valid_job(result: &GmresJobResult) -> bool {
        result.solutions.len() == SCENARIOS
            && result.summaries.len() == SCENARIOS * SUMMARIES_PER_SCENARIO
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

    #[derive(Clone, Copy)]
    enum FrankenArm {
        Fused,
        Separate,
    }

    impl FrankenArm {
        const fn disable_fusion(self) -> bool {
            matches!(self, Self::Separate)
        }
    }

    fn set_franken_arm(arm: FrankenArm) {
        GMRES_PRECONDITIONED_FUSED_DISABLE
            .store(arm.disable_fusion(), std::sync::atomic::Ordering::Relaxed);
    }

    fn time_franken_arm(
        job: &PersistentJob,
        arm: FrankenArm,
        repetitions: usize,
    ) -> Result<f64, String> {
        set_franken_arm(arm);
        let mut result = None;
        let started = Instant::now();
        for _ in 0..repetitions {
            result = Some(black_box(solve_inputs(job)));
        }
        let elapsed = started.elapsed().as_secs_f64();
        let result =
            result.ok_or_else(|| "persistent-job repetitions must be positive".to_string())?;
        if !valid_job(&result) {
            return Err("timed FrankenSciPy persistent job was incomplete".to_string());
        }
        black_box(checksum(&result));
        Ok(elapsed)
    }

    fn time_franken_cold_job(arm: FrankenArm) -> Result<f64, String> {
        set_franken_arm(arm);
        let started = Instant::now();
        let job = build_persistent_job()?;
        let result = black_box(solve_inputs(&job));
        let elapsed = started.elapsed().as_secs_f64();
        if !valid_job(&result) {
            return Err("cold FrankenSciPy setup-plus-solve job was incomplete".to_string());
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

    fn rust_null_pair(
        job: &PersistentJob,
        arm: FrankenArm,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_franken_arm(job, arm, repetitions)?,
                time_franken_arm(job, arm, repetitions)?,
            )
        } else {
            let right = time_franken_arm(job, arm, repetitions)?;
            let left = time_franken_arm(job, arm, repetitions)?;
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
                scipy
                    .persistent_job_time(configuration, repetitions)?
                    .elapsed,
                scipy
                    .persistent_job_time(configuration, repetitions)?
                    .elapsed,
            )
        } else {
            let right = scipy
                .persistent_job_time(configuration, repetitions)?
                .elapsed;
            let left = scipy
                .persistent_job_time(configuration, repetitions)?
                .elapsed;
            (left, right)
        };
        Ok(left / right)
    }

    #[derive(Clone)]
    struct Measurement {
        candidate: Vec<f64>,
        control: Vec<f64>,
        scipy: Vec<f64>,
        control_candidate_ratios: Vec<f64>,
        scipy_candidate_ratios: Vec<f64>,
        candidate_nulls: Vec<f64>,
        control_nulls: Vec<f64>,
        scipy_nulls: Vec<f64>,
    }

    struct Decision {
        outcome: &'static str,
        ratio_median: f64,
        ratio_low: f64,
        ratio_high: f64,
        candidate_p50: f64,
        comparator_p50: f64,
        candidate_null_median: f64,
        comparator_null_median: f64,
    }

    fn timed_triplet(
        job: &PersistentJob,
        scipy: &mut Scipy,
        configuration: &str,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64, f64), String> {
        match round % 3 {
            0 => Ok((
                time_franken_arm(job, FrankenArm::Fused, repetitions)?,
                time_franken_arm(job, FrankenArm::Separate, repetitions)?,
                scipy
                    .persistent_job_time(configuration, repetitions)?
                    .elapsed,
            )),
            1 => {
                let control = time_franken_arm(job, FrankenArm::Separate, repetitions)?;
                let incumbent = scipy
                    .persistent_job_time(configuration, repetitions)?
                    .elapsed;
                let candidate = time_franken_arm(job, FrankenArm::Fused, repetitions)?;
                Ok((candidate, control, incumbent))
            }
            _ => {
                let incumbent = scipy
                    .persistent_job_time(configuration, repetitions)?
                    .elapsed;
                let candidate = time_franken_arm(job, FrankenArm::Fused, repetitions)?;
                let control = time_franken_arm(job, FrankenArm::Separate, repetitions)?;
                Ok((candidate, control, incumbent))
            }
        }
    }

    fn measure_configuration(
        job: &PersistentJob,
        scipy: &mut Scipy,
        configuration: &str,
        rounds: usize,
        repetitions: usize,
    ) -> Result<Measurement, String> {
        for warmup in 0..2 {
            let _ = timed_triplet(job, scipy, configuration, repetitions, warmup)?;
        }
        let mut measurement = Measurement {
            candidate: Vec::with_capacity(rounds),
            control: Vec::with_capacity(rounds),
            scipy: Vec::with_capacity(rounds),
            control_candidate_ratios: Vec::with_capacity(rounds),
            scipy_candidate_ratios: Vec::with_capacity(rounds),
            candidate_nulls: Vec::with_capacity(rounds),
            control_nulls: Vec::with_capacity(rounds),
            scipy_nulls: Vec::with_capacity(rounds),
        };
        for round in 0..rounds {
            let (candidate, control, incumbent) =
                timed_triplet(job, scipy, configuration, repetitions, round)?;
            let (candidate_null, control_null, scipy_null) = match round % 3 {
                0 => {
                    let candidate_null =
                        rust_null_pair(job, FrankenArm::Fused, repetitions, round)?;
                    let control_null =
                        rust_null_pair(job, FrankenArm::Separate, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    (candidate_null, control_null, scipy_null)
                }
                1 => {
                    let control_null =
                        rust_null_pair(job, FrankenArm::Separate, repetitions, round)?;
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    let candidate_null =
                        rust_null_pair(job, FrankenArm::Fused, repetitions, round)?;
                    (candidate_null, control_null, scipy_null)
                }
                _ => {
                    let scipy_null = scipy_null_pair(scipy, configuration, repetitions, round)?;
                    let candidate_null =
                        rust_null_pair(job, FrankenArm::Fused, repetitions, round)?;
                    let control_null =
                        rust_null_pair(job, FrankenArm::Separate, repetitions, round)?;
                    (candidate_null, control_null, scipy_null)
                }
            };
            measurement.candidate.push(candidate);
            measurement.control.push(control);
            measurement.scipy.push(incumbent);
            measurement
                .control_candidate_ratios
                .push(control / candidate);
            measurement
                .scipy_candidate_ratios
                .push(incumbent / candidate);
            measurement.candidate_nulls.push(candidate_null);
            measurement.control_nulls.push(control_null);
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

    struct RatioSamples<'a> {
        candidate: &'a [f64],
        comparator: &'a [f64],
        ratios: &'a [f64],
        candidate_nulls: &'a [f64],
        comparator_nulls: &'a [f64],
    }

    fn print_ratio_decision(
        label: &str,
        comparator_name: &str,
        samples: RatioSamples<'_>,
        repetitions: usize,
    ) -> Decision {
        let RatioSamples {
            candidate,
            comparator,
            ratios,
            candidate_nulls,
            comparator_nulls,
        } = samples;
        let (ratio_low, ratio_high) = boot_ci(ratios);
        let (candidate_null_low, candidate_null_high) = boot_ci(candidate_nulls);
        let (comparator_null_low, comparator_null_high) = boot_ci(comparator_nulls);
        let candidate_p50 = median(candidate.to_vec()) / repetitions as f64;
        let comparator_p50 = median(comparator.to_vec()) / repetitions as f64;
        let candidate_p95 = percentile(candidate.to_vec(), 0.95) / repetitions as f64;
        let comparator_p95 = percentile(comparator.to_vec(), 0.95) / repetitions as f64;
        let candidate_p99 = percentile(candidate.to_vec(), 0.99) / repetitions as f64;
        let comparator_p99 = percentile(comparator.to_vec(), 0.99) / repetitions as f64;
        println!(
            "{label}_persistent_job_wall: fused-candidate p50={:.6}ms p95={:.6}ms \
             p99={:.6}ms | {comparator_name} p50={:.6}ms p95={:.6}ms p99={:.6}ms",
            candidate_p50 * 1e3,
            candidate_p95 * 1e3,
            candidate_p99 * 1e3,
            comparator_p50 * 1e3,
            comparator_p95 * 1e3,
            comparator_p99 * 1e3
        );
        println!(
            "{label}_raw_samples_seconds: candidate={} comparator={} ratios={} \
             null_candidate={} null_comparator={}",
            csv(candidate),
            csv(comparator),
            csv(ratios),
            csv(candidate_nulls),
            csv(comparator_nulls)
        );
        let candidate_null_median = median(candidate_nulls.to_vec());
        let comparator_null_median = median(comparator_nulls.to_vec());
        println!(
            "{label}_NULL-candidate A/A median={candidate_null_median:.6} \
             ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] cv={:.3}% \
             (provenance only)",
            cv(candidate_nulls) * 100.0
        );
        println!(
            "{label}_NULL-{comparator_name} A/A median={comparator_null_median:.6} \
             ci95=[{comparator_null_low:.6},{comparator_null_high:.6}] cv={:.3}% \
             (provenance only)",
            cv(comparator_nulls) * 100.0
        );
        let ratio_median = median(ratios.to_vec());
        println!(
            "{label} ratio: {comparator_name} / fused-candidate = {ratio_median:.4}x \
             (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
             cv={:.3}% provenance only)",
            cv(ratios) * 100.0
        );

        let null_half_width = ((candidate_null_high - candidate_null_low) / 2.0)
            .max((comparator_null_high - comparator_null_low) / 2.0)
            .max(0.0);
        let null_edge = candidate_null_high
            .max(comparator_null_high)
            .max(1.0 / candidate_null_low.max(1e-9))
            .max(1.0 / comparator_null_low.max(1e-9))
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
        let c3 = (candidate_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (comparator_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
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
             candidate_null_median={candidate_null_median:.6} \
             comparator_null_median={comparator_null_median:.6} \
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
            candidate_p50,
            comparator_p50,
            candidate_null_median,
            comparator_null_median,
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

    fn require_bounded_preflight() -> Result<(), String> {
        let mut last_error = String::new();
        for attempt in 1..=HOST_PREFLIGHT_ATTEMPTS {
            match require_host_wide_quiescence("pre") {
                Ok(()) => {
                    println!(
                        "host_wide_quiescence_preflight_attempt={attempt}/{} accepted=true",
                        HOST_PREFLIGHT_ATTEMPTS
                    );
                    return Ok(());
                }
                Err(error) => {
                    println!(
                        "host_wide_quiescence_preflight_attempt={attempt}/{} accepted=false \
                         detail={error}",
                        HOST_PREFLIGHT_ATTEMPTS
                    );
                    last_error = error;
                }
            }
        }
        Err(format!(
            "bounded host-wide preflight exhausted {HOST_PREFLIGHT_ATTEMPTS} load-only samples: \
             {last_error}"
        ))
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
        ours: &GmresJobResult,
        ours_fields: &[f64],
        expected_input_sha256: &str,
        check: &ScipyJobCheck,
    ) -> Proof {
        let expected_components = SCENARIOS * SIDE * SIDE;
        let expected_summaries = SCENARIOS * SUMMARIES_PER_SCENARIO;
        let structural = check.configuration.as_str() != ""
            && check.successes == SCENARIOS
            && check.input_sha256 == expected_input_sha256
            && check.observed_threads == 1
            && check.infos.len() == SCENARIOS
            && check.infos.iter().all(|info| *info == 0)
            && check.iterations.len() == SCENARIOS
            && check.iterations.iter().all(|iterations| *iterations > 0)
            && check.residuals.len() == SCENARIOS
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
            && relative_l2_difference <= 1.0e-8
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
        persistent_screen_elapsed: f64,
        cold_screen_elapsed: f64,
    }

    struct FrankenProof {
        eligible: bool,
        solutions_bit_identical: bool,
        summaries_bit_identical: bool,
        iterations_identical: bool,
        residuals_bit_identical: bool,
    }

    fn prove_franken_pair(candidate: &GmresJobResult, control: &GmresJobResult) -> FrankenProof {
        let solutions_bit_identical = candidate
            .solutions
            .iter()
            .zip(&control.solutions)
            .all(|(left, right)| left.solution == right.solution);
        let summaries_bit_identical = candidate.summaries == control.summaries;
        let iterations_identical =
            candidate
                .solutions
                .iter()
                .zip(&control.solutions)
                .all(|(left, right)| {
                    left.iterations == right.iterations && left.converged == right.converged
                });
        let residuals_bit_identical = candidate
            .solutions
            .iter()
            .zip(&control.solutions)
            .all(|(left, right)| left.residual_norm.to_bits() == right.residual_norm.to_bits());
        let eligible = valid_job(candidate)
            && valid_job(control)
            && candidate.solutions.len() == control.solutions.len()
            && solutions_bit_identical
            && summaries_bit_identical
            && iterations_identical
            && residuals_bit_identical;
        FrankenProof {
            eligible,
            solutions_bit_identical,
            summaries_bit_identical,
            iterations_identical,
            residuals_bit_identical,
        }
    }

    pub fn run() -> Result<(), String> {
        let arguments = std::env::args().collect::<Vec<_>>();
        let sequential_control = std::env::var_os("FSCI_GMRES_BATCH_FORCE_SEQUENTIAL").is_some();
        GMRES_BATCH_FORCE_SEQUENTIAL
            .store(sequential_control, std::sync::atomic::Ordering::Relaxed);
        let rounds = arguments
            .get(1)
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(21);
        if rounds < 21 {
            return Err(
                "pre-registered persistent-job median-CI gate requires rounds>=21".to_string(),
            );
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
            "gmres_batch_scheduler={} affinity_cpu_count={affinity_cpus}",
            if sequential_control {
                "same-elf-sequential-control"
            } else {
                "shared-nothing-auto"
            }
        );
        print_hardware_provenance(&affinity, affinity_cpus)?;
        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        require_bounded_preflight()?;

        println!(
            "fixture=persistent-steady-convection-diffusion-source-job side={SIDE} n={} \
             scenarios={SCENARIOS} rounds={rounds} repetitions={repetitions} \
             method=GMRES restart=20 rtol={RTOL} atol=0 maxiter={} x0=zeros \
             diagonal={DIAGONAL} west={WEST} east={EAST} vertical={VERTICAL} \
             source_rows=36,96,150 source_columns=30,72,120,162 source_radius=4 \
             requested_frankenscipy_threads=auto requested_scipy_threads=1",
            SIDE * SIDE,
            10 * SIDE * SIDE
        );
        println!(
            "persistent_job_boundary: INCLUDED=1_public_preconditioned_gmres_batch_12_solves,\
             442368_field_values,domain_inventory,east_outlet_integral,\
             source_weighted_exposure; EXCLUDED=operator_assembly,12_source_fields,\
             csc_conversion,one_reused_preconditioner,python_interpreter_startup,\
             scipy_import,pipe_transport,backend_screening,parity_serialization,\
             provenance_collection,bootstrap_calculation; \
             secondary_cold_screen_includes=operator_sources_csc_factor_solve_summaries"
        );
        println!(
            "work_units: operator_nnz={} scenario_solves={SCENARIOS} \
             materialized_field_values={} scientific_summaries={}",
            5 * SIDE * SIDE - 4 * SIDE,
            SCENARIOS * SIDE * SIDE,
            SCENARIOS * SUMMARIES_PER_SCENARIO
        );

        let setup_started = Instant::now();
        let job = build_persistent_job()?;
        let franken_setup_seconds = setup_started.elapsed().as_secs_f64();
        let expected_input_sha256 = input_sha256(&job.matrix, &job.rhses, SIDE);
        println!(
            "frankenscipy_persistent_setup_seconds={franken_setup_seconds:.9} \
             input_sha256={expected_input_sha256}"
        );

        GMRES_PRECONDITIONED_FUSED_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        set_franken_arm(FrankenArm::Fused);
        let (candidate, candidate_threads) = observed_peak_worker_threads(|| solve_inputs(&job));
        let candidate_fused_hits =
            GMRES_PRECONDITIONED_FUSED_HITS.load(std::sync::atomic::Ordering::Relaxed);
        GMRES_PRECONDITIONED_FUSED_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        set_franken_arm(FrankenArm::Separate);
        let (control, control_threads) = observed_peak_worker_threads(|| solve_inputs(&job));
        let control_fused_hits =
            GMRES_PRECONDITIONED_FUSED_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let expected_workers = SCENARIOS.min(affinity_cpus);
        if candidate_threads != expected_workers
            || control_threads != expected_workers
            || candidate_threads != control_threads
            || !valid_job(&candidate)
            || !valid_job(&control)
        {
            return Err(format!(
                "FrankenSciPy parity arms were inadmissible: candidate_threads={candidate_threads} \
                 control_threads={control_threads} expected_workers={expected_workers}"
            ));
        }
        let franken_proof = prove_franken_pair(&candidate, &control);
        println!(
            "candidate_control_exact_proof: eligible={} solutions_bit_identical={} \
             summaries_bit_identical={} iterations_identical={} \
             residuals_bit_identical={} candidate_fused_hits={} control_fused_hits={}",
            franken_proof.eligible,
            franken_proof.solutions_bit_identical,
            franken_proof.summaries_bit_identical,
            franken_proof.iterations_identical,
            franken_proof.residuals_bit_identical,
            candidate_fused_hits,
            control_fused_hits
        );
        if !franken_proof.eligible || candidate_fused_hits == 0 || control_fused_hits != 0 {
            return Err(
                "candidate/control exactness or fused-dispatch contract failed".to_string(),
            );
        }
        let candidate_fields = flatten_fields(&candidate);

        let script = scipy_oracle_script(arguments.get(2))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let (mut scipy, identity) = Scipy::start(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=1.17.1 ")
            || !identity.contains("gmres_mod=scipy.sparse.linalg._isolve")
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
             strongest valid public GMRES configuration selected live"
        );
        println!(
            "thread_provenance: requested_frankenscipy_threads=auto \
             actual_observed_candidate_worker_threads={candidate_threads} \
             actual_observed_control_worker_threads={control_threads} \
             requested_scipy_threads=1 actual_observed_scipy_worker_threads=1 \
             python_blas_thread_cap=1"
        );
        println!(
            "incumbent_backend_screen_contract: configurations={}; \
             selection=lowest_valid_live_persistent_job_wall_time; \
             full_output_eligibility_before_selection=true \
             cold_setup_plus_solve_screen=secondary \
             screen_outside_headline_samples=true",
            CONFIGURATIONS.join(",")
        );

        require_host_wide_quiescence("measurement")?;
        let candidate_cold_seconds = time_franken_cold_job(FrankenArm::Fused)?;
        let control_cold_seconds = time_franken_cold_job(FrankenArm::Separate)?;
        println!(
            "frankenscipy_cold_setup_plus_solve_seconds: fused={candidate_cold_seconds:.9} \
             separate={control_cold_seconds:.9}"
        );

        let mut candidates = Vec::with_capacity(CONFIGURATIONS.len());
        for configuration in CONFIGURATIONS {
            let check = scipy.job_check(configuration)?;
            let proof = prove_candidate(
                &candidate,
                &candidate_fields,
                &expected_input_sha256,
                &check,
            );
            println!(
                "scipy_backend_proof: configuration={configuration} \
                 eligible={} successes={}/{} input_sha_match={} \
                 max_abs_diff={:.3e} relative_l2_diff={:.3e} \
                 maximum_scaled_diff={:.3e} tolerance_mismatches={} \
                 maximum_summary_scaled_diff={:.3e} \
                 iterations={}",
                proof.eligible,
                check.successes,
                SCENARIOS,
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
                let cold_screen_elapsed = scipy.job_time(configuration, 1)?.elapsed;
                let persistent_samples = (0..3)
                    .map(|_| {
                        scipy
                            .persistent_job_time(configuration, 1)
                            .map(|timing| timing.elapsed)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let persistent_screen_elapsed = median(persistent_samples);
                println!(
                    "scipy_backend_screen: configuration={configuration} \
                     persistent_job_p50_ms={:.6} cold_setup_plus_solve_ms={:.6}",
                    persistent_screen_elapsed * 1e3,
                    cold_screen_elapsed * 1e3
                );
                candidates.push(Candidate {
                    configuration,
                    check,
                    persistent_screen_elapsed,
                    cold_screen_elapsed,
                });
            } else {
                println!(
                    "scipy_backend_screen: configuration={configuration} \
                     ineligible_full_result_contract=true"
                );
            }
        }
        if candidates.is_empty() {
            return Err("no SciPy GMRES configuration passed full-result eligibility".to_string());
        }
        candidates.sort_by(|left, right| {
            left.persistent_screen_elapsed
                .total_cmp(&right.persistent_screen_elapsed)
        });
        let selected = &candidates[0];
        println!(
            "selected_scipy_incumbent: configuration={} screened_persistent_job_ms={:.6} \
             screened_cold_setup_plus_solve_ms={:.6} \
             reason=fastest_valid_live_SciPy_GMRES_configuration",
            selected.configuration,
            selected.persistent_screen_elapsed * 1e3,
            selected.cold_screen_elapsed * 1e3
        );

        let candidate_iterations = candidate
            .solutions
            .iter()
            .map(|solution| solution.iterations)
            .collect::<Vec<_>>();
        let control_iterations = control
            .solutions
            .iter()
            .map(|solution| solution.iterations)
            .collect::<Vec<_>>();
        println!(
            "operation_counts: candidate_iterations={} control_iterations={} \
             exact_candidate_control_parity={} selected_scipy_configuration={} \
             selected_scipy_iterations={}",
            candidate_iterations
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            control_iterations
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            candidate_iterations == control_iterations,
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
            "agreement: scenarios={SCENARIOS}/{SCENARIOS} \
             compared_field_values={}/{} compared_summaries={}/{} \
             input_sha256={expected_input_sha256}",
            candidate_fields.len(),
            SCENARIOS * SIDE * SIDE,
            candidate.summaries.len(),
            SCENARIOS * SUMMARIES_PER_SCENARIO
        );

        let measurement = measure_configuration(
            &job,
            &mut scipy,
            selected.configuration,
            rounds,
            repetitions,
        )?;
        require_host_wide_quiescence("post")?;

        let maintenance = print_ratio_decision(
            "maintenance",
            "same-ELF-separate-control",
            RatioSamples {
                candidate: &measurement.candidate,
                comparator: &measurement.control,
                ratios: &measurement.control_candidate_ratios,
                candidate_nulls: &measurement.candidate_nulls,
                comparator_nulls: &measurement.control_nulls,
            },
            repetitions,
        );
        let competitive = print_ratio_decision(
            "competitive",
            selected.configuration,
            RatioSamples {
                candidate: &measurement.candidate,
                comparator: &measurement.scipy,
                ratios: &measurement.scipy_candidate_ratios,
                candidate_nulls: &measurement.candidate_nulls,
                comparator_nulls: &measurement.scipy_nulls,
            },
            repetitions,
        );
        let all_null_medians_within_2pct = (maintenance.candidate_null_median - 1.0).abs()
            <= NULL_MEDIAN_BIAS_LIMIT
            && (maintenance.comparator_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (competitive.comparator_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let maintenance_keep = all_null_medians_within_2pct
            && maintenance.outcome == "DECIDED FRANKENSCIPY WIN"
            && maintenance.ratio_low >= 1.10;
        let competitive_win =
            competitive.outcome == "DECIDED FRANKENSCIPY WIN" && competitive.ratio_low > 1.0;
        println!(
            "prediction_scorecard: exact_candidate_control={} fused_hits_nonzero={} \
             control_hits_zero={} all_null_medians_within_2pct={} spilu_selected={} \
             maintenance_ratio={:.4} maintenance_ci=[{:.4},{:.4}] \
             live_ratio={:.4} live_ci=[{:.4},{:.4}] maintenance_keep={} \
             competitive_win={} candidate_p50_ms={:.6} control_p50_ms={:.6} \
             live_p50_ms={:.6}",
            franken_proof.eligible,
            candidate_fused_hits > 0,
            control_fused_hits == 0,
            all_null_medians_within_2pct,
            selected.configuration == "csc-matrix-spilu",
            maintenance.ratio_median,
            maintenance.ratio_low,
            maintenance.ratio_high,
            competitive.ratio_median,
            competitive.ratio_low,
            competitive.ratio_high,
            maintenance_keep,
            competitive_win,
            maintenance.candidate_p50 * 1e3,
            maintenance.comparator_p50 * 1e3,
            competitive.comparator_p50 * 1e3
        );
        println!(
            "scope_guard: this result applies only to the side-192, twelve-source, \
             persistent-operator convection-diffusion GMRES job with one reused \
             FrankenSciPy ILU(0); fill-capable ILUT, direct solvers, other matrices, \
             sizes, source counts, and worker counts are not decided"
        );
        if !maintenance_keep {
            println!(
                "CHOOSER STATEMENT: REVERT the fused CSR/ILU-L traversal for this \
                 side-192 persistent job because it did not clear the registered \
                 1.10x exact same-ELF maintenance gate; pick live SciPy 1.17.1 {} \
                 whenever its measured persistent-job wall is lower.",
                selected.configuration,
            );
        } else if competitive_win {
            println!(
                "CHOOSER STATEMENT: KEEP and pick FrankenSciPy fused ILU-GMRES for \
                 this side-192 twelve-source persistent job; it clears both the \
                 same-ELF maintenance gate and the strongest live SciPy 1.17.1 {} \
                 comparison.",
                selected.configuration
            );
        } else if competitive.outcome == "DECIDED FRANKENSCIPY LOSS" {
            println!(
                "CHOOSER STATEMENT: KEEP the fused traversal as a measured same-ELF \
                 maintenance win, but pick live SciPy 1.17.1 {} for this exact \
                 persistent job because the incumbent remains faster.",
                selected.configuration
            );
        } else {
            println!(
                "CHOOSER STATEMENT: KEEP the fused traversal as a measured same-ELF \
                 maintenance win; the live incumbent comparison is undecided, so \
                 make no competitive chooser claim."
            );
        }
        set_franken_arm(FrankenArm::Fused);
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
