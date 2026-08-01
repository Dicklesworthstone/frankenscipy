//! Persistent row-band Lanczos MINRES versus its same-ELF scoped-iteration
//! control and versus live SciPy.
//!
//! Three arms per cell, all in one invocation so host drift hits every arm:
//!
//! * `persistent-minres` — one row-band worker team per public solve
//! * `scoped-control` — same ELF, forcing the inherited worker scopes per SpMV
//! * `live-scipy-minres` — genuine `scipy.sparse.linalg.minres` child process
//!
//! The operator is `A = L - shift*I` with `L` the canonical Dirichlet
//! five-point Laplacian, so `shift = 0` gives the SPD control and a shift inside
//! `L`'s spectrum gives the symmetric-indefinite case MINRES exists for. Only
//! the diagonal changes, so both fixtures share sparsity and nnz.
//!
//! Run:
//! `perf_minres_vs_scipy <side> <shift> [rounds] [reps] [docs/perf_oracle_minres.py]`

#[cfg(feature = "live-scipy-bench")]
mod live_minres {
    use fsci_sparse::linalg::{
        MINRES_FORCE_ITERATION_SCOPES, MINRES_LAST_PERSISTENT_WORKERS, MINRES_PERSISTENT_SOLVE_HITS,
    };
    use fsci_sparse::{CsrMatrix, IterativeSolveOptions, IterativeSolveResult, Shape2D, minres};
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    const BASE_DIAGONAL: f64 = 4.001;
    const RTOL: f64 = 1e-8;

    /// Canonical row-sorted five-point Laplacian with `diagonal` on the diagonal.
    fn laplacian_2d(side: usize, diagonal: f64) -> CsrMatrix {
        let n = side * side;
        let expected_nnz = 5 * n - 4 * side;
        let mut data = Vec::with_capacity(expected_nnz);
        let mut indices = Vec::with_capacity(expected_nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..side {
            for col in 0..side {
                let index = row * side + col;
                if row > 0 {
                    indices.push(index - side);
                    data.push(-1.0);
                }
                if col > 0 {
                    indices.push(index - 1);
                    data.push(-1.0);
                }
                indices.push(index);
                data.push(diagonal);
                if col + 1 < side {
                    indices.push(index + 1);
                    data.push(-1.0);
                }
                if row + 1 < side {
                    indices.push(index + side);
                    data.push(-1.0);
                }
                indptr.push(data.len());
            }
        }
        assert_eq!(data.len(), expected_nnz);
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical Laplacian CSR")
    }

    fn rhs(n: usize) -> Vec<f64> {
        (0..n)
            .map(|index| 1.0 + 0.01 * (index % 17) as f64)
            .collect()
    }

    fn options(max_iter: usize) -> IterativeSolveOptions {
        IterativeSolveOptions {
            tol: RTOL,
            max_iter: Some(max_iter),
            ..Default::default()
        }
    }

    fn solve_persistent(a: &CsrMatrix, b: &[f64], max_iter: usize) -> IterativeSolveResult {
        MINRES_FORCE_ITERATION_SCOPES.store(false, std::sync::atomic::Ordering::Relaxed);
        minres(a, b, None, options(max_iter)).expect("FrankenSciPy MINRES solve")
    }

    fn solve_scoped_control(a: &CsrMatrix, b: &[f64], max_iter: usize) -> IterativeSolveResult {
        MINRES_FORCE_ITERATION_SCOPES.store(true, std::sync::atomic::Ordering::Relaxed);
        let result = minres(a, b, None, options(max_iter)).expect("same-ELF scoped MINRES solve");
        MINRES_FORCE_ITERATION_SCOPES.store(false, std::sync::atomic::Ordering::Relaxed);
        result
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    struct ScipyParity {
        info: i32,
        iterations: usize,
        residual: f64,
        solution: Vec<f64>,
    }

    impl Scipy {
        fn start(script: &str) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--minres-live")
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| format!("spawn python3: {error}"))?;
            let stdin = child.stdin.take().ok_or("no stdin")?;
            let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
            let mut ready = String::new();
            stdout
                .read_line(&mut ready)
                .map_err(|error| format!("read READY: {error}"))?;
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                },
                ready.trim().to_string(),
            ))
        }

        fn line(&mut self, command: &str) -> Result<String, String> {
            writeln!(self.stdin, "{command}")
                .map_err(|error| format!("write {command}: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush: {error}"))?;
            let mut output = String::new();
            self.stdout
                .read_line(&mut output)
                .map_err(|error| format!("read reply to {command}: {error}"))?;
            Ok(output.trim().to_string())
        }

        fn prepare(
            &mut self,
            side: usize,
            diagonal: f64,
            max_iter: usize,
            expected_n: usize,
            expected_nnz: usize,
        ) -> Result<String, String> {
            let reply = self.line(&format!("PREP {side} {diagonal} {RTOL} {max_iter}"))?;
            if !reply.starts_with("CASE ")
                || !reply.contains(&format!("n={expected_n}"))
                || !reply.contains(&format!("nnz={expected_nnz}"))
                || !reply.contains("sorted=True")
            {
                return Err(format!("unmatched SciPy case: {reply}"));
            }
            Ok(reply)
        }

        fn parity(&mut self) -> Result<ScipyParity, String> {
            let result = self.line("PARITY")?;
            if !result.starts_with("RESULT ") {
                return Err(format!("bad PARITY result: {result}"));
            }
            let field = |name: &str| -> Result<&str, String> {
                result
                    .split_whitespace()
                    .find_map(|item| item.strip_prefix(&format!("{name}=")))
                    .ok_or_else(|| format!("missing {name} in {result}"))
            };
            let info = field("info")?
                .parse::<i32>()
                .map_err(|error| format!("parse info: {error}"))?;
            let iterations = field("iterations")?
                .parse::<usize>()
                .map_err(|error| format!("parse iterations: {error}"))?;
            let residual = field("residual")?
                .parse::<f64>()
                .map_err(|error| format!("parse residual: {error}"))?;
            let components = field("components")?
                .parse::<usize>()
                .map_err(|error| format!("parse components: {error}"))?;
            let mut values = String::new();
            self.stdout
                .read_line(&mut values)
                .map_err(|error| format!("read PARITY vector: {error}"))?;
            let payload = values
                .trim()
                .strip_prefix("X ")
                .ok_or_else(|| format!("bad PARITY vector: {}", values.trim()))?
                .to_string();
            let solution = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse solution component: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if solution.len() != components {
                return Err(format!(
                    "PARITY vector length {} != {components}",
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

        fn solve(&mut self, reps: usize, expected_n: usize) -> Result<f64, String> {
            let reply = self.line(&format!("SOLVE {reps}"))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"TIME") || fields.len() != 4 {
                return Err(format!("bad SOLVE reply: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse elapsed time: {error}"))?;
            let components = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse components: {error}"))?;
            if !elapsed.is_finite() || elapsed <= 0.0 || components != expected_n {
                return Err(format!("invalid timed SciPy result: {reply}"));
            }
            Ok(elapsed)
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn true_relative_residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
        let ax = a.matvec(x).expect("residual matvec");
        let numerator = b
            .iter()
            .zip(&ax)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt();
        let denominator = b.iter().map(|value| value * value).sum::<f64>().sqrt();
        numerator / denominator
    }

    fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    fn relative_l2_diff(left: &[f64], right: &[f64]) -> f64 {
        let numerator = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let denominator = right.iter().map(|value| value * value).sum::<f64>().sqrt();
        numerator / denominator.max(f64::MIN_POSITIVE)
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
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    fn ratios(numerators: &[f64], denominators: &[f64]) -> Vec<f64> {
        numerators
            .iter()
            .zip(denominators)
            .map(|(numerator, denominator)| numerator / denominator)
            .collect()
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
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
            / values.len() as f64;
        variance.sqrt() / mean
    }

    fn time_arm(
        solve: impl Fn(&CsrMatrix, &[f64], usize) -> IterativeSolveResult,
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
    ) -> f64 {
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(solve(black_box(a), black_box(b), max_iter)));
        }
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        elapsed
    }

    fn time_null_pair(
        solve: impl Fn(&CsrMatrix, &[f64], usize) -> IterativeSolveResult + Copy,
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
    ) -> (f64, f64) {
        (
            time_arm(solve, a, b, max_iter, reps),
            time_arm(solve, a, b, max_iter, reps),
        )
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

    /// Mean busy fraction across all CPUs over a short sample, reported rather
    /// than gated on: a hard quiescence gate is unsatisfiable on this shared box
    /// and would produce zero measurements instead of caveated ones.
    fn host_busy_fraction() -> f64 {
        let read = || -> Option<(u64, u64)> {
            let stat = std::fs::read_to_string("/proc/stat").ok()?;
            let line = stat.lines().next()?;
            let fields: Vec<u64> = line
                .split_whitespace()
                .skip(1)
                .filter_map(|value| value.parse().ok())
                .collect();
            if fields.len() < 5 {
                return None;
            }
            let total: u64 = fields.iter().sum();
            let idle = fields[3] + fields[4];
            Some((total, idle))
        };
        let Some((total0, idle0)) = read() else {
            return f64::NAN;
        };
        std::thread::sleep(std::time::Duration::from_millis(300));
        let Some((total1, idle1)) = read() else {
            return f64::NAN;
        };
        let delta_total = total1.saturating_sub(total0) as f64;
        if delta_total <= 0.0 {
            return f64::NAN;
        }
        1.0 - (idle1.saturating_sub(idle0) as f64) / delta_total
    }

    pub fn run() {
        let executable = std::env::current_exe().expect("current executable");
        let sha = {
            let mut hasher = Sha256::new();
            hasher.update(std::fs::read(&executable).expect("read own ELF"));
            format!("{:x}", hasher.finalize())
        };
        println!("elf_sha256={sha}");

        let arguments: Vec<String> = std::env::args().skip(1).collect();
        let side: usize = arguments
            .first()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);
        let shift: f64 = arguments
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.0);
        let rounds: usize = arguments
            .get(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(11);
        let reps: usize = arguments
            .get(3)
            .and_then(|value| value.parse().ok())
            .unwrap_or(3);
        let script = arguments
            .get(4)
            .cloned()
            .unwrap_or_else(|| "docs/perf_oracle_minres.py".to_string());
        let max_iter: usize = std::env::var("FSCI_MINRES_MAX_ITER")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(20_000);
        if side < 2 || rounds < 3 || reps == 0 {
            eprintln!("ABORT: require side>=2, rounds>=3, reps>=1");
            std::process::exit(2);
        }

        println!("cpu_affinity={}", cpu_affinity());
        let host_busy_pre = host_busy_fraction();
        println!("host_busy_fraction_pre={host_busy_pre:.3}");
        if !host_busy_pre.is_finite() || host_busy_pre > 0.20 {
            eprintln!("ABORT: preflight host busy fraction exceeds 0.20");
            std::process::exit(2);
        }

        let diagonal = BASE_DIAGONAL - shift;
        let a = laplacian_2d(side, diagonal);
        let n = a.shape().rows;
        let b = rhs(n);
        println!(
            "fixture side={side} n={n} nnz={} shift={shift} diagonal={diagonal} rtol={RTOL} \
             max_iter={max_iter} rounds={rounds} reps={reps}",
            a.nnz()
        );

        // ── Scientific and routing admission, before any timing ─────────────
        MINRES_PERSISTENT_SOLVE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        let candidate = solve_persistent(&a, &b, max_iter);
        let candidate_hits =
            MINRES_PERSISTENT_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let candidate_workers =
            MINRES_LAST_PERSISTENT_WORKERS.load(std::sync::atomic::Ordering::Relaxed);
        MINRES_PERSISTENT_SOLVE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        let control = solve_scoped_control(&a, &b, max_iter);
        let control_hits = MINRES_PERSISTENT_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let candidate_true_residual = true_relative_residual(&a, &b, &candidate.solution);
        let control_true_residual = true_relative_residual(&a, &b, &control.solution);
        let candidate_control_relative_l2 =
            relative_l2_diff(&candidate.solution, &control.solution);
        println!(
            "persistent-minres iterations={} converged={} reported_residual={:.6e} \
             true_residual={candidate_true_residual:.6e} route_hits={candidate_hits} \
             actual_workers={candidate_workers}",
            candidate.iterations, candidate.converged, candidate.residual_norm,
        );
        println!(
            "scoped-control iterations={} converged={} reported_residual={:.6e} \
             true_residual={control_true_residual:.6e} route_hits={control_hits}",
            control.iterations, control.converged, control.residual_norm,
        );
        println!(
            "candidate_control_max_abs={:.6e} candidate_control_relative_l2={:.6e}",
            max_abs_diff(&candidate.solution, &control.solution),
            candidate_control_relative_l2,
        );
        let rust_admission = candidate.converged
            && control.converged
            && candidate.iterations == control.iterations
            && candidate_true_residual <= RTOL
            && control_true_residual <= RTOL
            && candidate_control_relative_l2 <= 1e-10
            && candidate.solution.iter().all(|value| value.is_finite())
            && control.solution.iter().all(|value| value.is_finite())
            && candidate_hits == 1
            && control_hits == 0
            && candidate_workers >= 2;
        if !rust_admission {
            eprintln!("ABORT: candidate/control scientific or routing admission failed");
            std::process::exit(2);
        }

        let (mut scipy, ready) = match Scipy::start(&script) {
            Ok(pair) => pair,
            Err(error) => {
                eprintln!("ABORT: live SciPy arm unavailable: {error}");
                std::process::exit(2);
            }
        };
        println!("scipy_identity={ready}");
        if !ready.contains("genuine=True") {
            eprintln!("ABORT: SciPy arm is not genuine");
            std::process::exit(2);
        }
        let case = scipy
            .prepare(side, diagonal, max_iter, n, a.nnz())
            .expect("SciPy case preparation");
        println!("scipy_case={case}");

        let parity = scipy.parity().expect("SciPy parity solve");
        println!(
            "scipy-minres info={} iterations={} true_residual={:.6e}",
            parity.info, parity.iterations, parity.residual
        );
        println!(
            "agreement_candidate_vs_scipy_max_abs={:.6e}",
            max_abs_diff(&candidate.solution, &parity.solution)
        );
        if parity.info != 0
            || !parity.residual.is_finite()
            || parity.solution.len() != n
            || !parity.solution.iter().all(|value| value.is_finite())
        {
            eprintln!("ABORT: live SciPy scientific admission failed");
            std::process::exit(2);
        }

        // ── Timing: balanced headlines and an independent A/A for every arm ─
        let host_busy_measurement = host_busy_fraction();
        println!("host_busy_fraction_measurement={host_busy_measurement:.3}");
        if !host_busy_measurement.is_finite() || host_busy_measurement > 0.20 {
            eprintln!("ABORT: measurement host busy fraction exceeds 0.20");
            std::process::exit(2);
        }

        let mut candidate_times = Vec::with_capacity(rounds);
        let mut control_times = Vec::with_capacity(rounds);
        let mut scipy_times = Vec::with_capacity(rounds);
        let mut candidate_null_left = Vec::with_capacity(rounds);
        let mut candidate_null_right = Vec::with_capacity(rounds);
        let mut control_null_left = Vec::with_capacity(rounds);
        let mut control_null_right = Vec::with_capacity(rounds);
        let mut scipy_null_left = Vec::with_capacity(rounds);
        let mut scipy_null_right = Vec::with_capacity(rounds);

        // Untimed warm-up.
        black_box(solve_persistent(&a, &b, max_iter));
        black_box(solve_scoped_control(&a, &b, max_iter));
        let _ = scipy.solve(1, n).expect("SciPy warm-up");
        MINRES_PERSISTENT_SOLVE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);

        for round in 0..rounds {
            // Rotate arm order every round so no arm keeps a cache position.
            match round % 3 {
                0 => {
                    candidate_times.push(time_arm(solve_persistent, &a, &b, max_iter, reps));
                    control_times.push(time_arm(solve_scoped_control, &a, &b, max_iter, reps));
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                }
                1 => {
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                    candidate_times.push(time_arm(solve_persistent, &a, &b, max_iter, reps));
                    control_times.push(time_arm(solve_scoped_control, &a, &b, max_iter, reps));
                }
                _ => {
                    control_times.push(time_arm(solve_scoped_control, &a, &b, max_iter, reps));
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                    candidate_times.push(time_arm(solve_persistent, &a, &b, max_iter, reps));
                }
            }

            let mut candidate_null = || {
                let (left, right) = time_null_pair(solve_persistent, &a, &b, max_iter, reps);
                candidate_null_left.push(left);
                candidate_null_right.push(right);
            };
            let mut control_null = || {
                let (left, right) = time_null_pair(solve_scoped_control, &a, &b, max_iter, reps);
                control_null_left.push(left);
                control_null_right.push(right);
            };
            match round % 3 {
                0 => {
                    candidate_null();
                    control_null();
                    scipy_null_left.push(scipy.solve(reps, n).expect("SciPy null left"));
                    scipy_null_right.push(scipy.solve(reps, n).expect("SciPy null right"));
                }
                1 => {
                    scipy_null_left.push(scipy.solve(reps, n).expect("SciPy null left"));
                    scipy_null_right.push(scipy.solve(reps, n).expect("SciPy null right"));
                    candidate_null();
                    control_null();
                }
                _ => {
                    control_null();
                    scipy_null_left.push(scipy.solve(reps, n).expect("SciPy null left"));
                    scipy_null_right.push(scipy.solve(reps, n).expect("SciPy null right"));
                    candidate_null();
                }
            }
        }
        let timed_candidate_hits =
            MINRES_PERSISTENT_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let expected_candidate_hits = rounds * reps * 3;
        scipy.quit();
        if timed_candidate_hits != expected_candidate_hits {
            eprintln!(
                "ABORT: timed candidate hits {timed_candidate_hits} != {expected_candidate_hits}"
            );
            std::process::exit(2);
        }

        let host_busy_post = host_busy_fraction();
        println!("host_busy_fraction_post={host_busy_post:.3}");
        if !host_busy_post.is_finite() || host_busy_post > 0.20 {
            eprintln!("ABORT: postflight host busy fraction exceeds 0.20");
            std::process::exit(2);
        }

        let report = |label: &str, values: &[f64]| {
            let per_solve: Vec<f64> = values
                .iter()
                .map(|value| value * 1e3 / reps as f64)
                .collect();
            println!(
                "{label} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} cv_percent={:.3}",
                median(per_solve.clone()),
                percentile(per_solve.clone(), 0.95),
                percentile(per_solve.clone(), 0.99),
                cv(&per_solve) * 100.0,
            );
        };

        report("persistent-minres", &candidate_times);
        report("scoped-control", &control_times);
        report("live-scipy-minres", &scipy_times);
        println!(
            "raw_samples_seconds candidate={} control={} live={} candidate_null_left={} \
             candidate_null_right={} control_null_left={} control_null_right={} \
             live_null_left={} live_null_right={}",
            csv(&candidate_times),
            csv(&control_times),
            csv(&scipy_times),
            csv(&candidate_null_left),
            csv(&candidate_null_right),
            csv(&control_null_left),
            csv(&control_null_right),
            csv(&scipy_null_left),
            csv(&scipy_null_right),
        );

        let control_ratios = ratios(&control_times, &candidate_times);
        let live_ratios = ratios(&scipy_times, &candidate_times);
        let candidate_nulls = ratios(&candidate_null_left, &candidate_null_right);
        let control_nulls = ratios(&control_null_left, &control_null_right);
        let live_nulls = ratios(&scipy_null_left, &scipy_null_right);
        let (control_low, control_high) = bootstrap_median_ci(&control_ratios);
        let (live_low, live_high) = bootstrap_median_ci(&live_ratios);
        let (candidate_null_low, candidate_null_high) = bootstrap_median_ci(&candidate_nulls);
        let (control_null_low, control_null_high) = bootstrap_median_ci(&control_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);
        let candidate_null_median = median(candidate_nulls.clone());
        let control_null_median = median(control_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        println!(
            "candidate_A/A median={candidate_null_median:.6} \
             ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] raw={}",
            csv(&candidate_nulls),
        );
        println!(
            "control_A/A median={control_null_median:.6} \
             ci95=[{control_null_low:.6},{control_null_high:.6}] raw={}",
            csv(&control_nulls),
        );
        println!(
            "live_A/A median={live_null_median:.6} \
             ci95=[{live_null_low:.6},{live_null_high:.6}] raw={}",
            csv(&live_nulls),
        );

        let widest_null_edge = candidate_null_high
            .max(control_null_high)
            .max(live_null_high)
            .max(1.0 / candidate_null_low.max(1e-12))
            .max(1.0 / control_null_low.max(1e-12))
            .max(1.0 / live_null_low.max(1e-12))
            .max(1.0);
        let twice_null_threshold = 1.0 + 2.0 * (widest_null_edge - 1.0);
        let null_medians_pass = [candidate_null_median, control_null_median, live_null_median]
            .into_iter()
            .all(|value| (value - 1.0).abs() <= 0.02);
        let maintenance_pass = control_low >= 1.20 && control_low > twice_null_threshold;
        let competitive_pass = live_low > twice_null_threshold;
        println!(
            "maintenance_ratio control/candidate median={:.6} \
             bootstrap_median_ci95=[{control_low:.6},{control_high:.6}] \
             registered_minimum=1.200000 twice_widest_null_threshold={twice_null_threshold:.6}",
            median(control_ratios),
        );
        println!(
            "competitive_ratio live_scipy/candidate median={:.6} \
             bootstrap_median_ci95=[{live_low:.6},{live_high:.6}] \
             required_above_twice_null={twice_null_threshold:.6}",
            median(live_ratios),
        );
        let keep = null_medians_pass && maintenance_pass && competitive_pass;
        println!(
            "decision_gate null_medians_within_2pct={null_medians_pass} \
             maintenance_pass={maintenance_pass} competitive_flip_pass={competitive_pass} \
             cv_used_for_decision=false"
        );
        println!("DECISION={}", if keep { "KEEP" } else { "REVERT" });
    }
}

fn main() {
    #[cfg(feature = "live-scipy-bench")]
    {
        live_minres::run();
    }
    #[cfg(not(feature = "live-scipy-bench"))]
    {
        eprintln!("perf_minres_vs_scipy requires --features live-scipy-bench");
        std::process::exit(2);
    }
}
