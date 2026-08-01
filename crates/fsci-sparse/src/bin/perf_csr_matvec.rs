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
        CG_FORCE_ITERATION_SCOPES, CG_NARROW_INDICES_DISABLE, CG_NARROW_VALUE_SOLVE_HITS,
        CG_NARROW_VALUES_DISABLE, CG_WORKER_NNZ_SHIFT, CG_WORKER_NNZ_SHIFT_DEFAULT,
    };
    use fsci_sparse::{CsrMatrix, IterativeSolveOptions, Shape2D, cg};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::{Duration, Instant};

    const DIAGONAL: f64 = 4.001;
    const WIDE_DIAGONAL: f64 = 0.625;
    const WIDE_OFF_DIAGONAL: f64 = -1.0 / 512.0;
    const RTOL: f64 = 1e-5;
    const HOST_QUIESCENCE_MAX_BUSY: f64 = 0.20;
    const HOST_QUIESCENCE_SAMPLE: Duration = Duration::from_secs(1);

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

    fn wide_band_nnz(n: usize, half_bandwidth: usize) -> Option<usize> {
        n.checked_mul(half_bandwidth.checked_mul(2)?.checked_add(1)?)?
            .checked_sub(half_bandwidth.checked_mul(half_bandwidth.checked_add(1)?)?)
    }

    fn wide_band_spd(n: usize, half_bandwidth: usize) -> CsrMatrix {
        let expected_nnz = wide_band_nnz(n, half_bandwidth).expect("wide-band nnz fits usize");
        let mut data = Vec::with_capacity(expected_nnz);
        let mut indices = Vec::with_capacity(expected_nnz);
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in 0..n {
            let first = row.saturating_sub(half_bandwidth);
            let last = row.saturating_add(half_bandwidth).min(n - 1);
            for column in first..=last {
                indices.push(column);
                data.push(if column == row {
                    WIDE_DIAGONAL
                } else {
                    WIDE_OFF_DIAGONAL
                });
            }
            indptr.push(data.len());
        }
        assert_eq!(data.len(), expected_nnz);
        CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
            .expect("canonical wide-band SPD CSR")
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

    fn solve_ours_value_mode(
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        force_f64_values: bool,
    ) -> fsci_sparse::IterativeSolveResult {
        CG_NARROW_VALUES_DISABLE.store(force_f64_values, std::sync::atomic::Ordering::Relaxed);
        let result = solve_ours(a, b, max_iter);
        CG_NARROW_VALUES_DISABLE.store(false, std::sync::atomic::Ordering::Relaxed);
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

        fn prepare_wide(
            &mut self,
            n: usize,
            half_bandwidth: usize,
            max_iter: usize,
            expected_nnz: usize,
        ) -> Result<String, String> {
            let reply = self.line(&format!(
                "PREP_WIDE {n} {half_bandwidth} {WIDE_DIAGONAL} {WIDE_OFF_DIAGONAL} \
                 {RTOL} {max_iter}"
            ))?;
            if !reply.starts_with("CASE ")
                || !reply.contains(&format!("n={n}"))
                || !reply.contains(&format!("nnz={expected_nnz}"))
                || !reply.contains("sorted=True")
                || !reply.contains("dtype=float64")
            {
                return Err(format!("unmatched SciPy wide-band case: {reply}"));
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

    fn time_value_mode(
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
        force_f64_values: bool,
    ) -> (f64, usize) {
        let before = CG_NARROW_VALUE_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(solve_ours_value_mode(
                black_box(a),
                black_box(b),
                max_iter,
                force_f64_values,
            )));
        }
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        let after = CG_NARROW_VALUE_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        (elapsed, after.saturating_sub(before))
    }

    fn value_null_pair(
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
        round: usize,
        force_f64_values: bool,
    ) -> (f64, usize) {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_value_mode(a, b, max_iter, reps, force_f64_values),
                time_value_mode(a, b, max_iter, reps, force_f64_values),
            )
        } else {
            let right = time_value_mode(a, b, max_iter, reps, force_f64_values);
            let left = time_value_mode(a, b, max_iter, reps, force_f64_values);
            (left, right)
        };
        (left.0 / right.0, left.1 + right.1)
    }

    fn percentile(mut values: Vec<f64>, probability: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let index = ((values.len() - 1) as f64 * probability).ceil() as usize;
        values[index]
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn incumbent_pair(
        scipy: &mut Scipy,
        a: &CsrMatrix,
        b: &[f64],
        max_iter: usize,
        reps: usize,
        round: usize,
    ) -> (f64, f64) {
        if round.is_multiple_of(2) {
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
        let (left, right) = if round.is_multiple_of(2) {
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
        let (left, right) = if round.is_multiple_of(2) {
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
                .map_err(|error| format!("parse CPU index: {error}"))?;
            let ticks = fields
                .map(|field| {
                    field
                        .parse::<u64>()
                        .map_err(|error| format!("parse CPU tick: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("CPU {cpu} has an incomplete /proc/stat row"));
            }
            cpus.insert(
                cpu,
                CpuTicks {
                    total: ticks.iter().sum(),
                    idle: ticks[3].saturating_add(ticks[4]),
                },
            );
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
                busy.push((*cpu, busy_fraction));
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

    fn host_identity() -> String {
        std::fs::read_to_string("/etc/hostname")
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    fn cpu_topology() -> (usize, usize) {
        let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
            return (0, 0);
        };
        let mut cores = HashSet::new();
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
            if let Some(processor) = processor {
                logical += 1;
                cores.insert((
                    physical.unwrap_or("0").to_string(),
                    core.unwrap_or(processor).to_string(),
                ));
            }
        }
        (cores.len(), logical)
    }

    fn ram_bytes() -> u64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|meminfo| {
                meminfo
                    .lines()
                    .find_map(|line| line.strip_prefix("MemTotal:"))
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .unwrap_or(0)
            * 1024
    }

    fn numa_node_count() -> usize {
        std::fs::read_dir("/sys/devices/system/node")
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .strip_prefix("node")
                            .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
                    })
                    .count()
            })
            .unwrap_or(0)
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

    fn first_affinity_cpu(list: &str) -> Option<usize> {
        list.split(',')
            .find(|part| !part.is_empty())?
            .split('-')
            .next()?
            .trim()
            .parse()
            .ok()
    }

    fn cpu_frequency_policy(cpu: usize) -> String {
        let directory = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
        let read = |name: &str| {
            std::fs::read_to_string(format!("{directory}/{name}"))
                .map(|value| value.trim().to_string())
                .unwrap_or_else(|_| "unavailable".to_string())
        };
        format!(
            "sample_cpu={cpu} scaling_driver={} scaling_governor={} \
             energy_performance_preference={} scaling_min_freq_khz={} \
             scaling_max_freq_khz={}",
            read("scaling_driver"),
            read("scaling_governor"),
            read("energy_performance_preference"),
            read("scaling_min_freq"),
            read("scaling_max_freq")
        )
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

    pub fn run_mixed() {
        let executable = std::env::current_exe().expect("current executable");
        let sha = {
            let mut hasher = Sha256::new();
            hasher.update(std::fs::read(&executable).expect("read own ELF"));
            format!("{:x}", hasher.finalize())
        };
        println!("elf_sha256={sha}");
        println!("frankenscipy_engine_sha256={sha}");
        for name in [
            "BINARY_BUILDER_IDENTITY",
            "BINARY_SOURCE_COMMIT",
            "BINARY_BUILD_ROUTE",
            "TRJ_BOOKING_CLAIM_MESSAGE_ID",
        ] {
            let value = std::env::var(name).unwrap_or_else(|_| {
                eprintln!("ABORT: {name} is required for benchmark provenance");
                std::process::exit(2);
            });
            println!("{}={value}", name.to_ascii_lowercase());
        }

        let arguments: Vec<String> = std::env::args().skip(2).collect();
        let n = arguments
            .first()
            .and_then(|value| value.parse().ok())
            .unwrap_or(65_536usize);
        let half_bandwidth = arguments
            .get(1)
            .and_then(|value| value.parse().ok())
            .unwrap_or(128usize);
        let rounds = arguments
            .get(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(21usize);
        let reps = arguments
            .get(3)
            .and_then(|value| value.parse().ok())
            .unwrap_or(1usize);
        let script = arguments
            .get(4)
            .cloned()
            .unwrap_or_else(|| "docs/perf_oracle_spmv.py".to_string());
        if n < 2
            || half_bandwidth == 0
            || half_bandwidth >= n
            || rounds < 21
            || reps == 0
            || wide_band_nnz(n, half_bandwidth).is_none()
        {
            eprintln!(
                "ABORT: require n>=2, 0<half_bandwidth<n, rounds>=21, reps>=1, and fitting nnz"
            );
            std::process::exit(2);
        }

        for name in [
            "FSCI_CG_FORCE_ITERATION_SCOPES",
            "FSCI_CG_NARROW_INDICES_DISABLE",
            "FSCI_CG_WORKER_NNZ_SHIFT",
        ] {
            if std::env::var_os(name).is_some() {
                eprintln!("ABORT: {name} would override the registered solver configuration");
                std::process::exit(2);
            }
        }

        let affinity = cpu_affinity();
        let cpus = affinity_cpu_count(&affinity);
        if affinity == "unknown" || cpus == 0 {
            eprintln!("ABORT: pin this invocation to a non-empty physical-core affinity");
            std::process::exit(2);
        }
        let (physical_cores, logical_threads) = cpu_topology();
        println!(
            "hardware_provenance: host_identity={} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={} numa_nodes={} \
             runtime_detected_isa={} affinity={affinity} cpuset_logical_cap={cpus}",
            host_identity(),
            ram_bytes(),
            numa_node_count(),
            runtime_isa_features()
        );
        println!(
            "cpu_frequency_policy: {}",
            cpu_frequency_policy(first_affinity_cpu(&affinity).expect("validated affinity"))
        );
        println!(
            "cg_worker_nnz_shift={} narrow_indices=true candidate_values=f32-exact \
             control_values=f64 affinity_cpu_count={cpus} available_parallelism={}",
            CG_WORKER_NNZ_SHIFT_DEFAULT,
            std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(0)
        );
        require_host_wide_quiescence("pre").unwrap_or_else(|error| {
            eprintln!("ABORT: {error}");
            std::process::exit(2);
        });

        let expected_nnz = wide_band_nnz(n, half_bandwidth).expect("validated nnz");
        let a = wide_band_spd(n, half_bandwidth);
        let b = rhs(n);
        let max_iter = n.saturating_mul(10);
        println!(
            "fixture=symmetric-strictly-diagonally-dominant-wide-band n={n} \
             half_bandwidth={half_bandwidth} nnz={} diagonal={WIDE_DIAGONAL} \
             off_diagonal={WIDE_OFF_DIAGONAL} rhs=1+0.01*(i%17) rtol={RTOL} \
             atol=0 maxiter={max_iter} x0=zeros construction_outside_timing=true",
            a.nnz()
        );
        if a.nnz() != expected_nnz {
            eprintln!("ABORT: Rust fixture nnz does not match the registered cell");
            std::process::exit(3);
        }

        let (mut scipy, identity) = Scipy::start(&script).unwrap_or_else(|error| {
            eprintln!("ABORT: cannot start SciPy arm: {error}");
            std::process::exit(3);
        });
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("cg_mod=scipy.sparse.linalg._isolve.iterative")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
            || !identity.contains("cg_engine_sha256=")
        {
            eprintln!("ABORT: SciPy arm is not a hashed genuine incumbent");
            std::process::exit(4);
        }
        println!(
            "Legacy incumbent arm: genuine live SciPy; side-by-side same-invocation; \
             child-side float64 cg-only timing; BLAS thread cap supplied by caller"
        );
        let case = scipy
            .prepare_wide(n, half_bandwidth, max_iter, expected_nnz)
            .unwrap_or_else(|error| {
                eprintln!("ABORT: {error}");
                std::process::exit(5);
            });
        println!("scipy_case: {case}");

        CG_NARROW_VALUE_SOLVE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        let (candidate, candidate_workers) =
            observed_peak_tasks(|| solve_ours_value_mode(&a, &b, max_iter, false));
        let candidate_warm_hits =
            CG_NARROW_VALUE_SOLVE_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let before_control = candidate_warm_hits;
        let (control, control_workers) =
            observed_peak_tasks(|| solve_ours_value_mode(&a, &b, max_iter, true));
        let control_warm_hits = CG_NARROW_VALUE_SOLVE_HITS
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(before_control);
        let theirs = scipy.parity().unwrap_or_else(|error| {
            eprintln!("ABORT: SciPy parity solve failed: {error}");
            std::process::exit(6);
        });
        println!(
            "thread_provenance: requested_candidate_threads=auto \
             actual_observed_candidate_worker_tasks={candidate_workers} \
             requested_control_threads=auto actual_observed_control_worker_tasks={control_workers} \
             requested_scipy_threads=1 actual_observed_scipy_worker_threads=1"
        );

        let candidate_control_bit_mismatches = candidate
            .solution
            .iter()
            .zip(&control.solution)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        let candidate_residual = true_relative_residual(&a, &b, &candidate.solution);
        let control_residual = true_relative_residual(&a, &b, &control.solution);
        let mut max_abs_diff = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut scipy_sq = 0.0f64;
        let mut tolerance_mismatches = 0usize;
        for (left, right) in candidate.solution.iter().zip(&theirs.solution) {
            let difference = left - right;
            max_abs_diff = max_abs_diff.max(difference.abs());
            diff_sq += difference * difference;
            scipy_sq += right * right;
            let tolerance = 10.0 * RTOL * right.abs().max(1.0);
            tolerance_mismatches += usize::from(difference.abs() > tolerance);
        }
        let relative_l2_diff = diff_sq.sqrt() / scipy_sq.sqrt().max(f64::EPSILON);
        println!(
            "agreement: candidate_control_bit_mismatches={candidate_control_bit_mismatches} \
             candidate_live_max_abs_diff={max_abs_diff:.3e} \
             candidate_live_relative_l2_diff={relative_l2_diff:.3e} \
             candidate_live_tolerance_mismatches={tolerance_mismatches} \
             candidate_true_residual={candidate_residual:.3e} \
             control_true_residual={control_residual:.3e} \
             scipy_true_residual={:.3e}",
            theirs.residual
        );
        println!(
            "execution: candidate converged={} iterations={} reported_residual={:.3e} | \
             control converged={} iterations={} reported_residual={:.3e} | \
             scipy info={} iterations={} compact_warm_hits={candidate_warm_hits} \
             forced_f64_warm_hits={control_warm_hits}",
            candidate.converged,
            candidate.iterations,
            candidate.residual_norm,
            control.converged,
            control.iterations,
            control.residual_norm,
            theirs.info,
            theirs.iterations
        );
        if !candidate.converged
            || !control.converged
            || theirs.info != 0
            || candidate.solution.len() != n
            || control.solution.len() != n
            || theirs.solution.len() != n
            || candidate.iterations == 0
            || candidate.iterations != control.iterations
            || candidate.residual_norm.to_bits() != control.residual_norm.to_bits()
            || candidate_control_bit_mismatches != 0
            || candidate_residual > 1.25 * RTOL
            || control_residual > 1.25 * RTOL
            || theirs.residual > 1.25 * RTOL
            || tolerance_mismatches != 0
            || relative_l2_diff > 1e-8
            || candidate_warm_hits != 1
            || control_warm_hits != 0
            || candidate_workers != control_workers
        {
            eprintln!("ABORT: correctness, exactness, dispatch, or worker-parity gate failed");
            std::process::exit(7);
        }

        require_host_wide_quiescence("measurement").unwrap_or_else(|error| {
            eprintln!("ABORT: {error}");
            std::process::exit(2);
        });
        CG_NARROW_VALUE_SOLVE_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut candidate_times = Vec::with_capacity(rounds);
        let mut control_times = Vec::with_capacity(rounds);
        let mut scipy_times = Vec::with_capacity(rounds);
        let mut maintenance_ratios = Vec::with_capacity(rounds);
        let mut competitive_ratios = Vec::with_capacity(rounds);
        let mut candidate_nulls = Vec::with_capacity(rounds);
        let mut control_nulls = Vec::with_capacity(rounds);
        let mut scipy_nulls = Vec::with_capacity(rounds);
        let mut candidate_hits = 0usize;
        let mut control_hits = 0usize;
        for round in 0..rounds {
            let order = match round % 3 {
                0 => [0u8, 1, 2],
                1 => [2u8, 0, 1],
                _ => [1u8, 2, 0],
            };
            let mut candidate_time = 0.0;
            let mut control_time = 0.0;
            let mut scipy_time = 0.0;
            for arm in order {
                match arm {
                    0 => {
                        let timed = time_value_mode(&a, &b, max_iter, reps, false);
                        candidate_time = timed.0;
                        candidate_hits += timed.1;
                    }
                    1 => {
                        let timed = time_value_mode(&a, &b, max_iter, reps, true);
                        control_time = timed.0;
                        control_hits += timed.1;
                    }
                    _ => {
                        scipy_time = scipy.solve(reps, n).expect("timed SciPy CG");
                    }
                }
            }
            let candidate_null = value_null_pair(&a, &b, max_iter, reps, round, false);
            candidate_hits += candidate_null.1;
            let control_null = value_null_pair(&a, &b, max_iter, reps, round, true);
            control_hits += control_null.1;
            let scipy_null = scipy_null_pair(&mut scipy, n, reps, round);
            candidate_times.push(candidate_time / reps as f64);
            control_times.push(control_time / reps as f64);
            scipy_times.push(scipy_time / reps as f64);
            maintenance_ratios.push(control_time / candidate_time);
            competitive_ratios.push(scipy_time / candidate_time);
            candidate_nulls.push(candidate_null.0);
            control_nulls.push(control_null.0);
            scipy_nulls.push(scipy_null);
        }
        require_host_wide_quiescence("post").unwrap_or_else(|error| {
            eprintln!("ABORT: {error}");
            std::process::exit(2);
        });
        scipy.quit();

        let expected_candidate_hits = rounds * reps * 3;
        println!(
            "mechanism: candidate_compact_solve_hits={candidate_hits} \
             expected_candidate_hits={expected_candidate_hits} \
             forced_f64_compact_solve_hits={control_hits}"
        );
        if candidate_hits != expected_candidate_hits || control_hits != 0 {
            eprintln!("ABORT: timed dispatch counters do not prove the registered mechanism");
            std::process::exit(7);
        }

        for (label, values) in [
            ("candidate", &candidate_times),
            ("control", &control_times),
            ("scipy", &scipy_times),
        ] {
            println!(
                "{label}_p50_ms={:.6} {label}_p95_ms={:.6} {label}_p99_ms={:.6} \
                 {label}_cv={:.3}%",
                median(values.clone()) * 1e3,
                percentile(values.clone(), 0.95) * 1e3,
                percentile(values.clone(), 0.99) * 1e3,
                cv(values) * 100.0
            );
            println!("raw_{label}_seconds={}", csv(values));
        }

        let print_decision = |label: &str,
                              ratios: &[f64],
                              first_nulls: &[f64],
                              second_nulls: &[f64],
                              minimum: f64| {
            let ratio_median = median(ratios.to_vec());
            let (ratio_low, ratio_high) = bootstrap_median_ci(ratios);
            let first_null_median = median(first_nulls.to_vec());
            let second_null_median = median(second_nulls.to_vec());
            let (first_null_low, first_null_high) = bootstrap_median_ci(first_nulls);
            let (second_null_low, second_null_high) = bootstrap_median_ci(second_nulls);
            let widest_null_endpoint = (first_null_low - 1.0)
                .abs()
                .max((first_null_high - 1.0).abs())
                .max((second_null_low - 1.0).abs())
                .max((second_null_high - 1.0).abs());
            let null_half_width = ((first_null_high - first_null_low) / 2.0)
                .max((second_null_high - second_null_low) / 2.0);
            let c1 = ratio_low >= minimum;
            let c2 = ratio_median - 1.0 > 2.0 * null_half_width;
            let c2b = ratio_low - 1.0 > 2.0 * widest_null_endpoint;
            let c3 =
                (first_null_median - 1.0).abs() <= 0.02 && (second_null_median - 1.0).abs() <= 0.02;
            println!(
                "{label}_ratio median={ratio_median:.6} \
                 ci95=[{ratio_low:.6},{ratio_high:.6}] minimum={minimum:.2}"
            );
            println!(
                "{label}_null_first median={first_null_median:.6} \
                 ci95=[{first_null_low:.6},{first_null_high:.6}]"
            );
            println!(
                "{label}_null_second median={second_null_median:.6} \
                 ci95=[{second_null_low:.6},{second_null_high:.6}]"
            );
            println!(
                "{label}_corrected_null_gate: c1_ci_low_at_least_minimum={c1} \
                 c2_point_effect_beats_2x_half_width={c2} \
                 c2b_nearer_ci_endpoint_beats_2x_null_endpoint={c2b} \
                 c3_null_medians_within_2pct={c3} decidable={}",
                c1 && c2 && c2b && c3
            );
        };
        print_decision(
            "maintenance_control_over_candidate",
            &maintenance_ratios,
            &candidate_nulls,
            &control_nulls,
            1.10,
        );
        print_decision(
            "competitive_scipy_over_candidate",
            &competitive_ratios,
            &candidate_nulls,
            &scipy_nulls,
            1.0,
        );
        println!("raw_maintenance_ratios={}", csv(&maintenance_ratios));
        println!("raw_competitive_ratios={}", csv(&competitive_ratios));
        println!("raw_candidate_nulls={}", csv(&candidate_nulls));
        println!("raw_control_nulls={}", csv(&control_nulls));
        println!("raw_scipy_nulls={}", csv(&scipy_nulls));
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
    if std::env::args().nth(1).as_deref() == Some("cg-mixed-vs-scipy") {
        #[cfg(feature = "live-scipy-bench")]
        {
            live_cg::run_mixed();
            return;
        }
        #[cfg(not(feature = "live-scipy-bench"))]
        {
            eprintln!("cg-mixed-vs-scipy requires --features live-scipy-bench");
            std::process::exit(2);
        }
    }

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
