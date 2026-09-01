//! Lanczos MINRES versus the GMRES(20) delegate it replaces, and versus live SciPy.
//!
//! Three arms per cell, all in one invocation so host drift hits every arm:
//!
//! * `ours-minres` — `fsci_sparse::minres` (three-term Lanczos + Givens)
//! * `ours-gmres20` — `fsci_sparse::gmres`, which is exactly what `minres`
//!   delegated to before this change
//! * `scipy-minres` — live `scipy.sparse.linalg.minres` in a child process
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
    use fsci_runtime::scipy_incumbent::ScipyIncumbent;
    use fsci_sparse::{
        CsrMatrix, IterativeSolveOptions, IterativeSolveResult, Shape2D, gmres, minres,
    };
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    /// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
    /// installation whose compiled submodules do not load, and that difference would
    /// otherwise only surface mid-run.
    const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.sparse.linalg"];

    /// The one live-SciPy incumbent this process compares against, resolved once and PROVEN
    /// by running the import rather than by a name resolving on `PATH`.
    ///
    /// This harness used to spawn a bare `python3`, which on `thinkstation1` is 3.14 with no
    /// SciPy at all, so the oracle died on its first write and the run read as a flaky pipe
    /// rather than as a missing incumbent (frankenscipy-m5s54).
    fn incumbent() -> &'static ScipyIncumbent {
        static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
        INCUMBENT.get_or_init(|| {
            let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
                .unwrap_or_else(|error| panic!("{error}"));
            println!("{}", resolved.provenance_line());
            resolved
        })
    }

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

    fn solve_minres(a: &CsrMatrix, b: &[f64], max_iter: usize) -> IterativeSolveResult {
        minres(a, b, None, options(max_iter)).expect("FrankenSciPy MINRES solve")
    }

    fn solve_gmres20(a: &CsrMatrix, b: &[f64], max_iter: usize) -> IterativeSolveResult {
        gmres(a, b, None, options(max_iter)).expect("FrankenSciPy GMRES delegate solve")
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
            let mut child = incumbent()
                .command()
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
        println!("host_busy_fraction={:.3}", host_busy_fraction());

        let diagonal = BASE_DIAGONAL - shift;
        let a = laplacian_2d(side, diagonal);
        let n = a.shape().rows;
        let b = rhs(n);
        println!(
            "fixture side={side} n={n} nnz={} shift={shift} diagonal={diagonal} rtol={RTOL} \
             max_iter={max_iter} rounds={rounds} reps={reps}",
            a.nnz()
        );

        // ── Scouting: what each arm actually does, before any timing ────────
        let ours = solve_minres(&a, &b, max_iter);
        let delegate = solve_gmres20(&a, &b, max_iter);
        println!(
            "ours-minres  iterations={} converged={} reported_residual={:.6e} true_residual={:.6e}",
            ours.iterations,
            ours.converged,
            ours.residual_norm,
            true_relative_residual(&a, &b, &ours.solution)
        );
        println!(
            "ours-gmres20 iterations={} converged={} reported_residual={:.6e} true_residual={:.6e}",
            delegate.iterations,
            delegate.converged,
            delegate.residual_norm,
            true_relative_residual(&a, &b, &delegate.solution)
        );
        println!(
            "a_applications_ratio_gmres20_over_minres={:.4}",
            delegate.iterations as f64 / ours.iterations.max(1) as f64
        );

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
            "agreement_minres_vs_scipy_max_abs={:.6e}",
            max_abs_diff(&ours.solution, &parity.solution)
        );
        println!(
            "agreement_gmres20_vs_scipy_max_abs={:.6e}",
            max_abs_diff(&delegate.solution, &parity.solution)
        );

        // ── Timing: interleaved so host drift hits all three arms ───────────
        let mut minres_times = Vec::with_capacity(rounds);
        let mut delegate_times = Vec::with_capacity(rounds);
        let mut scipy_times = Vec::with_capacity(rounds);
        let mut null_ratios = Vec::with_capacity(rounds);

        // Untimed warm-up.
        black_box(solve_minres(&a, &b, max_iter));
        black_box(solve_gmres20(&a, &b, max_iter));
        let _ = scipy.solve(1, n).expect("SciPy warm-up");

        for round in 0..rounds {
            // Rotate arm order every round so no arm keeps a cache position.
            match round % 3 {
                0 => {
                    minres_times.push(time_arm(solve_minres, &a, &b, max_iter, reps));
                    delegate_times.push(time_arm(solve_gmres20, &a, &b, max_iter, reps));
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                }
                1 => {
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                    minres_times.push(time_arm(solve_minres, &a, &b, max_iter, reps));
                    delegate_times.push(time_arm(solve_gmres20, &a, &b, max_iter, reps));
                }
                _ => {
                    delegate_times.push(time_arm(solve_gmres20, &a, &b, max_iter, reps));
                    scipy_times.push(scipy.solve(reps, n).expect("timed SciPy MINRES"));
                    minres_times.push(time_arm(solve_minres, &a, &b, max_iter, reps));
                }
            }
            // Null pair: same arm against itself. Its median should sit at 1.0;
            // how far it strays is this cell's noise floor.
            let left = time_arm(solve_minres, &a, &b, max_iter, reps);
            let right = time_arm(solve_minres, &a, &b, max_iter, reps);
            null_ratios.push(left / right);
        }
        scipy.quit();

        let report = |label: &str, values: &[f64]| -> f64 {
            let per_solve: Vec<f64> = values
                .iter()
                .map(|value| value * 1e3 / reps as f64)
                .collect();
            let med = median(per_solve.clone());
            let (low, high) = bootstrap_median_ci(&per_solve);
            println!(
                "{label} median_ms_per_solve={med:.4} ci95=[{low:.4},{high:.4}] cv={:.4}",
                cv(&per_solve)
            );
            med
        };

        let minres_ms = report("ours-minres ", &minres_times);
        let delegate_ms = report("ours-gmres20", &delegate_times);
        let scipy_ms = report("scipy-minres", &scipy_times);
        let null_med = median(null_ratios.clone());
        println!(
            "null_pair_median={null_med:.4} null_bias={:.4}",
            (null_med - 1.0).abs()
        );

        println!(
            "SPEEDUP minres_vs_gmres20_delegate={:.4}x",
            delegate_ms / minres_ms
        );
        println!(
            "SPEEDUP minres_vs_scipy_minres={:.4}x",
            scipy_ms / minres_ms
        );
        println!(
            "SPEEDUP gmres20_delegate_vs_scipy_minres={:.4}x",
            scipy_ms / delegate_ms
        );
        if !ours.converged || !delegate.converged {
            println!(
                "WARNING at least one arm hit the {max_iter}-iteration cap; \
                 wall-clock ratios involving it are bounds, not measurements"
            );
        }
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
