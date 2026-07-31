//! Same-process A/B for parallel CSR SpMV (the inner kernel of every Krylov
//! solver, eigsh/eigs/svds, onenormest). Each output row is an independent dot
//! product, so parallelizing across row chunks is byte-identical. This settles
//! whether SpMV is bandwidth-bound (<2x) or scales on this 64-core box.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_csr_matvec`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{CsrMatrix, FormatConvertible, Shape2D, random, spmv_csr};

#[cfg(feature = "live-scipy-bench")]
mod live_cg {
    use fsci_sparse::linalg::{
        CG_FORCE_ITERATION_SCOPES, CG_NARROW_INDICES_DISABLE, CG_WORKER_NNZ_SHIFT,
        CG_WORKER_NNZ_SHIFT_DEFAULT,
    };
    use fsci_sparse::{CsrMatrix, IterativeSolveOptions, Shape2D, cg};
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    const DIAGONAL: f64 = 4.001;
    const RTOL: f64 = 1e-5;

    fn laplacian_2d(side: usize) -> CsrMatrix {
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
                data.push(DIAGONAL);
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

    fn solve_ours(a: &CsrMatrix, b: &[f64], max_iter: usize) -> fsci_sparse::IterativeSolveResult {
        let force_iteration_scopes = std::env::var_os("FSCI_CG_FORCE_ITERATION_SCOPES").is_some();
        CG_FORCE_ITERATION_SCOPES
            .store(force_iteration_scopes, std::sync::atomic::Ordering::Relaxed);
        // Worker budget under test. Reported by the caller so the cell records
        // which budget produced its number.
        CG_NARROW_INDICES_DISABLE.store(
            std::env::var_os("FSCI_CG_NARROW_INDICES_DISABLE").is_some(),
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(shift) = std::env::var("FSCI_CG_WORKER_NNZ_SHIFT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            CG_WORKER_NNZ_SHIFT.store(shift, std::sync::atomic::Ordering::Relaxed);
        }
        let result = cg(
            a,
            b,
            None,
            IterativeSolveOptions {
                tol: RTOL,
                max_iter: Some(max_iter),
                ..Default::default()
            },
        )
        .expect("FrankenSciPy CG solve");
        CG_FORCE_ITERATION_SCOPES.store(false, std::sync::atomic::Ordering::Relaxed);
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
                .arg("--cg-live")
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
            self.read_reply(command)
        }

        fn read_reply(&mut self, context: &str) -> Result<String, String> {
            let mut output = String::new();
            self.stdout
                .read_line(&mut output)
                .map_err(|error| format!("read reply to {context}: {error}"))?;
            Ok(output.trim().to_string())
        }

        fn prepare(
            &mut self,
            side: usize,
            max_iter: usize,
            expected_n: usize,
            expected_nnz: usize,
        ) -> Result<String, String> {
            let reply = self.line(&format!("PREP {side} {DIAGONAL} {RTOL} {max_iter}"))?;
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
            let values = self.read_reply("PARITY vector")?;
            let payload = values
                .strip_prefix("X ")
                .ok_or_else(|| format!("bad PARITY vector: {values}"))?;
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
            let info = fields[2]
                .parse::<i32>()
                .map_err(|error| format!("parse info: {error}"))?;
            let components = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse components: {error}"))?;
            if !elapsed.is_finite() || elapsed <= 0.0 || info != 0 || components != expected_n {
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

    fn time_ours(a: &CsrMatrix, b: &[f64], max_iter: usize, reps: usize) -> f64 {
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(solve_ours(black_box(a), black_box(b), max_iter)));
        }
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        elapsed
    }

    fn incumbent_pair(
        scipy: &mut Scipy,
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
        round: usize,
    ) -> (f64, f64) {
        if round % 2 == 0 {
            (
                time_ours(a, b, max_iter, reps),
                scipy.solve(reps, b.len()).expect("timed SciPy CG"),
            )
        } else {
            let incumbent = scipy.solve(reps, b.len()).expect("timed SciPy CG");
            let ours = time_ours(a, b, max_iter, reps);
            (ours, incumbent)
        }
    }

    fn ours_null_pair(a: &CsrMatrix, b: &[f64], max_iter: usize, reps: usize, round: usize) -> f64 {
        let (left, right) = if round % 2 == 0 {
            (
                time_ours(a, b, max_iter, reps),
                time_ours(a, b, max_iter, reps),
            )
        } else {
            let right = time_ours(a, b, max_iter, reps);
            let left = time_ours(a, b, max_iter, reps);
            (left, right)
        };
        left / right
    }

    fn scipy_null_pair(scipy: &mut Scipy, n: usize, reps: usize, round: usize) -> f64 {
        let (left, right) = if round % 2 == 0 {
            (
                scipy.solve(reps, n).expect("SciPy CG null A"),
                scipy.solve(reps, n).expect("SciPy CG null B"),
            )
        } else {
            let right = scipy.solve(reps, n).expect("SciPy CG null B");
            let left = scipy.solve(reps, n).expect("SciPy CG null A");
            (left, right)
        };
        left / right
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

    /// Count the CPUs named by a `Cpus_allowed_list` string ("5", "0-63", "0,4-7").
    fn affinity_cpu_count(list: &str) -> usize {
        list.split(',')
            .filter(|part| !part.is_empty())
            .map(|part| match part.split_once('-') {
                Some((low, high)) => {
                    let low: usize = low.trim().parse().unwrap_or(0);
                    let high: usize = high.trim().parse().unwrap_or(0);
                    high.saturating_sub(low) + 1
                }
                None => 1,
            })
            .sum()
    }

    /// Peak OS tasks this process actually ran, sampled off the timing path.
    ///
    /// Requested threads are not evidence. This polls `/proc/self/task` while an
    /// untimed warm-up solve runs and reports the maximum it saw.
    fn observed_peak_tasks<T>(work: impl FnOnce() -> T + Send) -> (T, usize)
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
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                peak
            });
            let value = work();
            done.store(true, std::sync::atomic::Ordering::Relaxed);
            // Subtract the watcher itself and the main thread.
            let peak = watcher.join().unwrap_or(1).saturating_sub(2).max(1);
            (value, peak)
        })
    }

    pub fn run() {
        let executable = std::env::current_exe().expect("current executable");
        let sha = {
            let mut hasher = Sha256::new();
            hasher.update(std::fs::read(&executable).expect("read own ELF"));
            format!("{:x}", hasher.finalize())
        };
        println!("elf_sha256={sha}");

        let arguments: Vec<String> = std::env::args().skip(2).collect();
        let side = arguments
            .first()
            .and_then(|value| value.parse().ok())
            .unwrap_or(120);
        let rounds = arguments
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(11);
        let reps = arguments
            .get(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(3);
        let script = arguments
            .get(3)
            .cloned()
            .unwrap_or_else(|| "docs/perf_oracle_spmv.py".to_string());
        if side < 2 || rounds < 3 || reps == 0 {
            eprintln!("ABORT: require side>=2, rounds>=3, reps>=1");
            std::process::exit(2);
        }

        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        if affinity == "unknown" {
            eprintln!("ABORT: pin this invocation with taskset; affinity is unreadable");
            std::process::exit(2);
        }
        // A multi-CPU cell is admitted deliberately: the candidate under test is
        // per-iteration parallelism, and a one-CPU pin cannot measure it. The
        // SciPy arm keeps its single-thread caps either way, so the cell must
        // report what each side actually ran on rather than what it asked for.
        let cpus = affinity_cpu_count(&affinity);
        if cpus == 0 {
            eprintln!("ABORT: affinity {affinity} names no CPU");
            std::process::exit(2);
        }
        println!(
            "cg_worker_nnz_shift={} (default {CG_WORKER_NNZ_SHIFT_DEFAULT}) narrow_indices={}",
            std::env::var("FSCI_CG_WORKER_NNZ_SHIFT")
                .unwrap_or_else(|_| CG_WORKER_NNZ_SHIFT_DEFAULT.to_string()),
            !std::env::var_os("FSCI_CG_NARROW_INDICES_DISABLE").is_some()
        );
        println!(
            "affinity_cpu_count={cpus} available_parallelism={}",
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(0)
        );

        let a = laplacian_2d(side);
        let n = side * side;
        let expected_nnz = 5 * n - 4 * side;
        let b = rhs(n);
        let max_iter = n * 10;
        println!(
            "fixture=dirichlet-five-point-laplacian side={side} n={n} nnz={} \
             diagonal={DIAGONAL} rhs=1+0.01*(i%17) rtol={RTOL} atol=0 \
             maxiter={max_iter} x0=zeros",
            a.nnz()
        );

        let (mut scipy, identity) = Scipy::start(&script).unwrap_or_else(|error| {
            eprintln!("ABORT: cannot start SciPy arm: {error}");
            std::process::exit(3);
        });
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("cg_mod=scipy.sparse.linalg._isolve.iterative")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            eprintln!("ABORT: SciPy arm is not genuine");
            std::process::exit(4);
        }
        let scipy_version = identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix("scipy="))
            .expect("SciPy version");
        println!(
            "Legacy incumbent arm: SciPy {scipy_version}; side-by-side \
             same-invocation; child-side cg-only timing"
        );
        let case = scipy
            .prepare(side, max_iter, n, expected_nnz)
            .unwrap_or_else(|error| {
                eprintln!("ABORT: {error}");
                std::process::exit(5);
            });
        println!("scipy_case: {case}");

        let (ours, ours_peak_tasks) = observed_peak_tasks(|| solve_ours(&a, &b, max_iter));
        println!(
            "thread_provenance: actual_observed_frankenscipy_worker_tasks={ours_peak_tasks} \
             requested_frankenscipy_threads=auto scipy_thread_caps=1 \
             observation_outside_timing=true"
        );
        let theirs = scipy.parity().unwrap_or_else(|error| {
            eprintln!("ABORT: SciPy parity solve failed: {error}");
            std::process::exit(6);
        });
        let ours_residual = true_relative_residual(&a, &b, &ours.solution);
        let mut max_abs_diff = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut scipy_sq = 0.0f64;
        for (left, right) in ours.solution.iter().zip(&theirs.solution) {
            let difference = left - right;
            max_abs_diff = max_abs_diff.max(difference.abs());
            diff_sq += difference * difference;
            scipy_sq += right * right;
        }
        let relative_l2_diff = diff_sq.sqrt() / scipy_sq.sqrt().max(f64::EPSILON);
        let iteration_ratio = ours.iterations as f64 / theirs.iterations.max(1) as f64;
        println!(
            "agreement: components={}/{} max_abs_diff={max_abs_diff:.3e} \
             relative_l2_diff={relative_l2_diff:.3e} true_residual_ours={ours_residual:.3e} \
             true_residual_scipy={:.3e}",
            ours.solution.len(),
            theirs.solution.len(),
            theirs.residual
        );
        println!(
            "execution: ours converged={} iterations={} reported_residual={:.3e} | \
             scipy info={} iterations={} iteration_ratio={iteration_ratio:.4}",
            ours.converged, ours.iterations, ours.residual_norm, theirs.info, theirs.iterations
        );
        if !ours.converged
            || theirs.info != 0
            || ours.solution.len() != n
            || theirs.solution.len() != n
            || ours.iterations == 0
            || theirs.iterations == 0
            || !ours_residual.is_finite()
            || !theirs.residual.is_finite()
            || ours_residual > 1.25 * RTOL
            || theirs.residual > 1.25 * RTOL
            || !relative_l2_diff.is_finite()
            || relative_l2_diff > 0.05
            || !(0.75..=1.25).contains(&iteration_ratio)
        {
            eprintln!("ABORT: arms did not solve a numerically comparable CG problem");
            std::process::exit(7);
        }

        let (mut ours_times, mut scipy_times, mut ratios) = (vec![], vec![], vec![]);
        let (mut ours_nulls, mut scipy_nulls) = (vec![], vec![]);
        for round in 0..rounds {
            let (ours_time, scipy_time, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent = incumbent_pair(&mut scipy, &a, &b, max_iter, reps, round);
                    let ours_null = ours_null_pair(&a, &b, max_iter, reps, round);
                    let scipy_null = scipy_null_pair(&mut scipy, n, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(&mut scipy, n, reps, round);
                    let incumbent = incumbent_pair(&mut scipy, &a, &b, max_iter, reps, round);
                    let ours_null = ours_null_pair(&a, &b, max_iter, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(&a, &b, max_iter, reps, round);
                    let scipy_null = scipy_null_pair(&mut scipy, n, reps, round);
                    let incumbent = incumbent_pair(&mut scipy, &a, &b, max_iter, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_times.push(ours_time);
            scipy_times.push(scipy_time);
            ratios.push(scipy_time / ours_time);
            ours_nulls.push(ours_null);
            scipy_nulls.push(scipy_null);
        }

        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios);
        let (ours_null_low, ours_null_high) = bootstrap_median_ci(&ours_nulls);
        let (scipy_null_low, scipy_null_high) = bootstrap_median_ci(&scipy_nulls);
        let ours_p50 = median(ours_times);
        let scipy_p50 = median(scipy_times);
        println!(
            "OURS p50={:.6}ms/rep SCIPY p50={:.6}ms/rep",
            ours_p50 * 1e3 / reps as f64,
            scipy_p50 * 1e3 / reps as f64
        );
        println!(
            "NULL-ours median={:.6} ci95=[{ours_null_low:.6},{ours_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(ours_nulls.clone()),
            cv(&ours_nulls) * 100.0
        );
        println!(
            "NULL-scipy median={:.6} ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] \
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
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1e-9))
            .max(1.0 / scipy_null_low.max(1e-9));
        let required = 1.0 + 2.0 * (null_edge - 1.0);
        let outcome = if ratio_low > required {
            "DECIDED FRANKENSCIPY WIN"
        } else if ratio_high < 1.0 / required {
            "DECIDED FRANKENSCIPY LOSS"
        } else {
            "NOT DECIDED"
        };
        println!(
            "median-CI gate: worst_null_edge={null_edge:.4} required={required:.4} \
             ratio_ci=[{ratio_low:.4},{ratio_high:.4}] => {outcome}"
        );
        scipy.quit();
    }
}

fn legacy_public_spmv_csr(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.shape().rows];
    for (row, output) in result.iter_mut().enumerate().take(a.shape().rows) {
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            *output += a.data()[idx] * x[a.indices()[idx]];
        }
    }
    result
}

fn public_spmv_ab() {
    println!("public spmv_csr legacy-row-sweep vs current; bit identity + timing\n");
    for &(n, density, reps) in &[
        (100usize, 0.05f64, 200_000u32),
        (1_000, 0.01, 50_000),
        (10_000, 0.001, 5_000),
    ] {
        let a = random(Shape2D::new(n, n), density, 0xA11CE ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect();

        let legacy = legacy_public_spmv_csr(&a, &x);
        let current = spmv_csr(&a, &x).unwrap();
        let identical = legacy
            .iter()
            .zip(&current)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        let mut acc = 0.0;
        let _ = legacy_public_spmv_csr(&a, &x);
        let _ = spmv_csr(&a, &x).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            acc += legacy_public_spmv_csr(black_box(&a), black_box(&x))[0];
        }
        let legacy_time = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..reps {
            acc += spmv_csr(black_box(&a), black_box(&x)).unwrap()[0];
        }
        let current_time = t1.elapsed();
        println!(
            "n={n:>5} nnz={:>7} reps={reps:>6} identical={identical} legacy={:>9.3?} current={:>9.3?} ratio={:>5.2}x (acc={acc:.3})",
            a.nnz(),
            legacy_time / reps,
            current_time / reps,
            legacy_time.as_secs_f64() / current_time.as_secs_f64(),
        );
    }
}

fn serial(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for idx in indptr[i]..indptr[i + 1] {
            s += data[idx] * x[indices[idx]];
        }
        out[i] = s;
    }
    out
}

fn parallel(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let nnz = data.len();
    // Scale workers by WORK (nnz), ~128K nnz/thread, so medium matrices don't
    // over-spawn; serial below ~256K nnz where spawn cost isn't amortized.
    let nthreads = if nnz < 1 << 18 || n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(nnz / (1 << 17))
            .max(1)
    };
    if nthreads <= 1 {
        return serial(a, x);
    }
    let mut out = vec![0.0; n];
    let chunk = n.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (t, slot) in out.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (r, o) in slot.iter_mut().enumerate() {
                    let i = base + r;
                    let mut s = 0.0;
                    for idx in indptr[i]..indptr[i + 1] {
                        s += data[idx] * x[indices[idx]];
                    }
                    *o = s;
                }
            });
        }
    });
    out
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("cg-vs-scipy") {
        #[cfg(feature = "live-scipy-bench")]
        {
            live_cg::run();
            return;
        }
        #[cfg(not(feature = "live-scipy-bench"))]
        {
            eprintln!("cg-vs-scipy requires --features live-scipy-bench");
            std::process::exit(2);
        }
    }

    if std::env::var_os("FSCI_PUBLIC_SPMV_AB").is_some() {
        public_spmv_ab();
        return;
    }

    println!("nproc check below; A/B byte-identity + timing\n");
    for &(n, density) in &[
        (20_000usize, 0.0005f64),
        (50_000, 0.0004),
        (200_000, 0.0001),
        (500_000, 0.00004),
    ] {
        let a = random(Shape2D::new(n, n), density, 0xC0FFEE ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let nnz = a.data().len();
        let mut x = vec![0.0; n];
        let mut s = 0x1234_5678u64;
        for xi in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *xi = (s >> 11) as f64 / (1u64 << 53) as f64;
        }

        // Byte-identity
        let ser = serial(&a, &x);
        let par = parallel(&a, &x);
        let identical = ser
            .iter()
            .zip(&par)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        let reps = 200;
        let _ = parallel(&a, &x);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += serial(black_box(&a), black_box(&x))[0];
        }
        let ts = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..reps {
            acc += parallel(black_box(&a), black_box(&x))[0];
        }
        let tp = t1.elapsed();
        println!(
            "n={n:>7} nnz={nnz:>9} identical={identical}  serial={:>9.3?}  parallel={:>9.3?}  ratio={:>5.1}x (acc={acc:.3})",
            ts / reps,
            tp / reps,
            ts.as_secs_f64() / tp.as_secs_f64()
        );
    }
    println!("\n--- end-to-end (lib eigsh, uses lib csr_matvec) ---");
    end_to_end();
}

// End-to-end: eigsh (power iteration) does many matvecs; the lib csr_matvec
// change speeds them. Run this binary after stashing the lib change to compare.
fn end_to_end() {
    use fsci_sparse::{EigsOptions, eigsh};
    for &(n, density) in &[(100_000usize, 0.00015f64), (300_000, 0.00005)] {
        // Symmetric-ish: A + A^T would be ideal but random is fine for matvec timing.
        let a = random(Shape2D::new(n, n), density, 0xBEEF ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let t0 = Instant::now();
        let r = eigsh(
            &a,
            2,
            EigsOptions {
                tol: 1e-8,
                max_iter: 80,
            },
        );
        let dt = t0.elapsed();
        println!(
            "eigsh n={n} nnz={} -> {:?} in {:?}",
            a.data().len(),
            r.is_ok(),
            dt
        );
    }
}
