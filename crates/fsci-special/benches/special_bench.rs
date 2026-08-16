use criterion::{BenchmarkId, Criterion, criterion_group};
use fsci_runtime::RuntimeMode;
use fsci_special::{
    MATHIEU_PERIODIC_CACHE_DISABLE_FOR_BENCH, SpecialTensor, bei_zeros, beip_zeros, ber_zeros,
    berp_zeros, beta, ellipe, ellipeinc, ellipk, ellipkinc, erf, erfc, erfinv, factorialk, gamma,
    gammainc, gammaln, hyperu, hyperu_scalar, j0, j1, jn_zeros, jnjnp_zeros, jnp_zeros, jv,
    kei_zeros, keip_zeros, kelvin_zeros, ker_zeros, kerp_zeros, log_ndtr, log_ndtr_scalar,
    mathieu_cem, mathieu_sem, ndtri, pbdv, pbdv_many, rgamma, riccati_jn, riccati_yn,
    spence_scalar, y0, zeta, zeta_scalar,
};
use std::f64::consts::{FRAC_2_PI, LN_2, PI};
use std::hint::black_box;
use std::io::Write;
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::Ordering;
use std::time::Duration;

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn real_vec(values: &[f64]) -> SpecialTensor {
    SpecialTensor::RealVec(values.to_vec())
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

fn consume_tensor(t: SpecialTensor) {
    match t {
        SpecialTensor::RealScalar(v) => {
            black_box(v);
        }
        SpecialTensor::RealVec(values) => {
            black_box(values);
        }
        _ => panic!("unexpected tensor shape"),
    }
}

mod live_kv {
    use fsci_runtime::RuntimeMode;
    use fsci_special::bessel::kv;
    use fsci_special::types::SpecialTensor;
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    const ORDER: f64 = 1.5;
    const POINT_COUNT: usize = 2_000_000;
    const Z_LOW: f64 = 0.3;
    const Z_HIGH: f64 = 50.0;
    const MIN_SAMPLE_MS: f64 = 2.0;
    const ABS_TOLERANCE: f64 = 1.0e-13;
    const REL_TOLERANCE: f64 = 1.0e-12;
    const BOOTSTRAP_RESAMPLES: usize = 10_000;

    struct ScipyKv {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    impl ScipyKv {
        fn start(script: &Path) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--kv-live")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|error| format!("failed to spawn live SciPy arm: {error}"))?;
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
                .map_err(|error| format!("failed to read live SciPy identity: {error}"))?;
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
            let mut output = String::new();
            self.stdout
                .read_line(&mut output)
                .map_err(|error| format!("failed to read {context}: {error}"))?;
            if output.is_empty() {
                return Err(format!("live SciPy arm exited while reading {context}"));
            }
            Ok(output.trim().to_string())
        }

        fn prepare(&mut self, values: &[f64]) -> Result<String, String> {
            writeln!(self.stdin, "PREP {ORDER:.17e} {}", values.len())
                .map_err(|error| format!("failed to write PREP: {error}"))?;
            let mut payload = Vec::with_capacity(std::mem::size_of_val(values));
            for value in values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            self.stdin
                .write_all(&payload)
                .map_err(|error| format!("failed to write exact kv fixture: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("failed to flush PREP: {error}"))?;
            let reply = self.read_reply("SciPy fixture identity")?;
            let input_sha = {
                let mut hasher = Sha256::new();
                hasher.update(&payload);
                format!("{:x}", hasher.finalize())
            };
            if !reply.starts_with("CASE ")
                || !reply.contains(&format!("order={ORDER}"))
                || !reply.contains(&format!("points={}", values.len()))
                || !reply.contains("sorted=True")
                || !reply.contains("finite=True")
                || !reply.contains("positive=True")
                || !reply.contains(&format!("input_sha256={input_sha}"))
            {
                return Err(format!(
                    "live SciPy arm constructed the wrong fixture: {reply}"
                ));
            }
            Ok(reply)
        }

        fn parity(&mut self, expected_components: usize) -> Result<Vec<f64>, String> {
            writeln!(self.stdin, "PARITY")
                .map_err(|error| format!("failed to write PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("failed to flush PARITY: {error}"))?;
            let result = self.read_reply("SciPy parity header")?;
            let components = result
                .strip_prefix("RESULT components=")
                .ok_or_else(|| format!("invalid SciPy parity header: {result}"))?
                .parse::<usize>()
                .map_err(|error| format!("invalid SciPy parity component count: {error}"))?;
            if components != expected_components {
                return Err(format!(
                    "SciPy parity component count {components} != {expected_components}"
                ));
            }
            let mut payload = vec![0_u8; components * size_of::<f64>()];
            self.stdout
                .read_exact(&mut payload)
                .map_err(|error| format!("failed to read SciPy parity vector: {error}"))?;
            let mut terminator = [0_u8; 1];
            self.stdout
                .read_exact(&mut terminator)
                .map_err(|error| format!("failed to read SciPy parity terminator: {error}"))?;
            if &terminator != b"\n" {
                return Err("invalid SciPy parity vector terminator".to_string());
            }
            Ok(payload
                .as_chunks::<{ size_of::<f64>() }>()
                .0
                .iter()
                .map(|bytes| {
                    let value = *bytes;
                    f64::from_le_bytes(value)
                })
                .collect())
        }

        fn solve(&mut self, repetitions: usize, expected_components: usize) -> Result<f64, String> {
            writeln!(self.stdin, "SOLVE {repetitions}")
                .map_err(|error| format!("failed to write SOLVE: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("failed to flush SOLVE: {error}"))?;
            let reply = self.read_reply("timed SciPy kv result")?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.len() != 4 || fields.first() != Some(&"TIME") {
                return Err(format!("invalid timed SciPy kv result: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("invalid SciPy elapsed time: {error}"))?;
            let components = fields[2]
                .parse::<usize>()
                .map_err(|error| format!("invalid SciPy component count: {error}"))?;
            if !elapsed.is_finite() || elapsed <= 0.0 || components != expected_components {
                return Err(format!("invalid timed SciPy kv result: {reply}"));
            }
            black_box(fields[3]);
            Ok(elapsed)
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            self.0 = value;
            value
        }
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let midpoint = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            0.5 * (sorted[midpoint - 1] + sorted[midpoint])
        } else {
            sorted[midpoint]
        }
    }

    fn bootstrap_median_ci(values: &[f64], seed: u64) -> (f64, f64) {
        let mut generator = XorShift64(seed);
        let mut sample = Vec::with_capacity(values.len());
        let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
        for _ in 0..BOOTSTRAP_RESAMPLES {
            sample.clear();
            for _ in 0..values.len() {
                sample.push(values[generator.next() as usize % values.len()]);
            }
            medians.push(median(&sample));
        }
        medians.sort_by(f64::total_cmp);
        (medians[250], medians[9_750])
    }

    fn coefficient_of_variation(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sum_squared_deviations = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (sum_squared_deviations / values.len().saturating_sub(1).max(1) as f64).sqrt() / mean
    }

    fn solve_ours(order: &SpecialTensor, values: &SpecialTensor) -> Vec<f64> {
        match kv(order, values, RuntimeMode::Strict).expect("FrankenSciPy kv") {
            SpecialTensor::RealVec(result) => result,
            _ => panic!("FrankenSciPy kv returned a non-real-vector result"),
        }
    }

    fn time_ours(order: &SpecialTensor, values: &SpecialTensor, repetitions: usize) -> f64 {
        let started = Instant::now();
        let mut components = 0usize;
        for _ in 0..repetitions {
            let result = black_box(solve_ours(black_box(order), black_box(values)));
            components ^= result.len();
            black_box(result);
        }
        black_box(components);
        started.elapsed().as_secs_f64()
    }

    fn calibrate_repetitions(
        scipy: &mut ScipyKv,
        order: &SpecialTensor,
        values: &SpecialTensor,
    ) -> Result<usize, String> {
        let mut repetitions = 1usize;
        loop {
            let ours = time_ours(order, values, repetitions);
            let incumbent = scipy.solve(repetitions, POINT_COUNT)?;
            if ours * 1_000.0 >= MIN_SAMPLE_MS && incumbent * 1_000.0 >= MIN_SAMPLE_MS {
                return Ok(repetitions);
            }
            repetitions = repetitions
                .checked_mul(2)
                .ok_or_else(|| "kv calibration repetition count overflowed".to_string())?;
        }
    }

    fn incumbent_pair(
        scipy: &mut ScipyKv,
        order: &SpecialTensor,
        values: &SpecialTensor,
        repetitions: usize,
        round: usize,
    ) -> Result<(f64, f64), String> {
        if round.is_multiple_of(2) {
            Ok((
                time_ours(order, values, repetitions),
                scipy.solve(repetitions, POINT_COUNT)?,
            ))
        } else {
            let incumbent = scipy.solve(repetitions, POINT_COUNT)?;
            let ours = time_ours(order, values, repetitions);
            Ok((ours, incumbent))
        }
    }

    fn ours_null_pair(
        order: &SpecialTensor,
        values: &SpecialTensor,
        repetitions: usize,
        round: usize,
    ) -> f64 {
        let (left, right) = if round.is_multiple_of(2) {
            (
                time_ours(order, values, repetitions),
                time_ours(order, values, repetitions),
            )
        } else {
            let right = time_ours(order, values, repetitions);
            let left = time_ours(order, values, repetitions);
            (left, right)
        };
        left / right
    }

    fn scipy_null_pair(
        scipy: &mut ScipyKv,
        repetitions: usize,
        round: usize,
    ) -> Result<f64, String> {
        let (left, right) = if round.is_multiple_of(2) {
            (
                scipy.solve(repetitions, POINT_COUNT)?,
                scipy.solve(repetitions, POINT_COUNT)?,
            )
        } else {
            let right = scipy.solve(repetitions, POINT_COUNT)?;
            let left = scipy.solve(repetitions, POINT_COUNT)?;
            (left, right)
        };
        Ok(left / right)
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

    pub fn run(arguments: &[String]) -> Result<(), String> {
        let executable = std::env::current_exe()
            .map_err(|error| format!("failed to locate current executable: {error}"))?;
        let mut hasher = Sha256::new();
        hasher.update(
            std::fs::read(&executable)
                .map_err(|error| format!("failed to read current executable: {error}"))?,
        );
        println!("elf_sha256={:x}", hasher.finalize());

        let live_index = arguments
            .iter()
            .position(|argument| argument == "--live-scipy-kv")
            .ok_or_else(|| "missing --live-scipy-kv dispatch".to_string())?;
        let rounds = arguments
            .get(live_index + 1)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(21);
        if rounds < 5 {
            return Err("live kv arm requires at least five rounds".to_string());
        }

        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        if affinity == "unknown" || affinity.contains(',') || affinity.contains('-') {
            return Err("pin the live kv invocation to exactly one CPU with taskset".to_string());
        }

        let values: Vec<f64> = (0..POINT_COUNT)
            .map(|index| Z_LOW + (Z_HIGH - Z_LOW) * index as f64 / (POINT_COUNT - 1) as f64)
            .collect();
        let order = SpecialTensor::RealScalar(ORDER);
        let value_tensor = SpecialTensor::RealVec(values.clone());
        println!(
            "fixture=kv-half-integer-structural-translation order={ORDER} \
             points={POINT_COUNT} z=[{Z_LOW},{Z_HIGH}] exact_input_binary_transfer=true \
             expected_path=half_integer_closed_form construction_outside_timing=true"
        );

        let script = arguments
            .get(live_index + 2)
            .filter(|argument| !argument.starts_with('-'))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../docs/perf_oracle_special_cephes.py")
            });
        let (mut scipy, identity) = ScipyKv::start(&script)?;
        println!("scipy_arm: {identity}");
        if !identity.starts_with("READY scipy=")
            || !identity.contains("kv_name=kv")
            || !identity.contains("kv_type=ufunc")
            || !identity.contains("fsci_loaded=False")
            || !identity.contains("genuine=True")
        {
            return Err("live SciPy kv arm failed genuine-incumbent identity gate".to_string());
        }
        let scipy_version = identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix("scipy="))
            .ok_or_else(|| "live SciPy arm omitted its version".to_string())?;
        println!(
            "Legacy incumbent arm: SciPy {scipy_version}; side-by-side \
             same-invocation; child-side scipy.special.kv-only timing"
        );

        let case = scipy.prepare(&values)?;
        println!("scipy_case: {case}");
        let ours = solve_ours(&order, &value_tensor);
        let incumbent = scipy.parity(POINT_COUNT)?;
        let mut max_abs_difference = 0.0f64;
        let mut max_rel_difference = 0.0f64;
        let mut mismatch_count = 0usize;
        for (&left, &right) in ours.iter().zip(&incumbent) {
            let difference = (left - right).abs();
            let tolerance = ABS_TOLERANCE + REL_TOLERANCE * right.abs();
            max_abs_difference = max_abs_difference.max(difference);
            if right != 0.0 {
                max_rel_difference = max_rel_difference.max(difference / right.abs());
            }
            mismatch_count += usize::from(!difference.is_finite() || difference > tolerance);
        }
        println!(
            "agreement: components={}/{} max_abs_diff={max_abs_difference:.3e} \
             max_rel_diff={max_rel_difference:.3e} abs_tolerance={ABS_TOLERANCE:.1e} \
             rel_tolerance={REL_TOLERANCE:.1e} tolerance_mismatches={mismatch_count}",
            ours.len(),
            incumbent.len()
        );
        if ours.len() != POINT_COUNT
            || incumbent.len() != POINT_COUNT
            || mismatch_count != 0
            || !max_abs_difference.is_finite()
            || !max_rel_difference.is_finite()
        {
            return Err("kv arms failed full-vector SciPy conformance".to_string());
        }

        let repetitions = calibrate_repetitions(&mut scipy, &order, &value_tensor)?;
        println!("calibration repetitions={repetitions} min_sample_ms={MIN_SAMPLE_MS}");
        for warmup in 0..4 {
            let _ = incumbent_pair(&mut scipy, &order, &value_tensor, repetitions, warmup)?;
        }

        let mut ours_times = Vec::with_capacity(rounds);
        let mut scipy_times = Vec::with_capacity(rounds);
        let mut ratios = Vec::with_capacity(rounds);
        let mut ours_nulls = Vec::with_capacity(rounds);
        let mut scipy_nulls = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (ours_time, scipy_time, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent =
                        incumbent_pair(&mut scipy, &order, &value_tensor, repetitions, round)?;
                    let ours_null = ours_null_pair(&order, &value_tensor, repetitions, round);
                    let scipy_null = scipy_null_pair(&mut scipy, repetitions, round)?;
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(&mut scipy, repetitions, round)?;
                    let incumbent =
                        incumbent_pair(&mut scipy, &order, &value_tensor, repetitions, round)?;
                    let ours_null = ours_null_pair(&order, &value_tensor, repetitions, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(&order, &value_tensor, repetitions, round);
                    let scipy_null = scipy_null_pair(&mut scipy, repetitions, round)?;
                    let incumbent =
                        incumbent_pair(&mut scipy, &order, &value_tensor, repetitions, round)?;
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_times.push(ours_time);
            scipy_times.push(scipy_time);
            ratios.push(scipy_time / ours_time);
            ours_nulls.push(ours_null);
            scipy_nulls.push(scipy_null);
        }

        let (ratio_low, ratio_high) = bootstrap_median_ci(&ratios, 0x510e_527f_ade6_82d1);
        let (ours_null_low, ours_null_high) =
            bootstrap_median_ci(&ours_nulls, 0x9b05_688c_2b3e_6c1f);
        let (scipy_null_low, scipy_null_high) =
            bootstrap_median_ci(&scipy_nulls, 0x1f83_d9ab_fb41_bd6b);
        let ours_p50 = median(&ours_times);
        let scipy_p50 = median(&scipy_times);
        println!(
            "OURS p50={:.6}ms/rep SCIPY p50={:.6}ms/rep",
            ours_p50 * 1_000.0 / repetitions as f64,
            scipy_p50 * 1_000.0 / repetitions as f64
        );
        println!(
            "NULL-ours A/A median={:.6} ci95=[{ours_null_low:.6},{ours_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(&ours_nulls),
            coefficient_of_variation(&ours_nulls) * 100.0
        );
        println!(
            "NULL-scipy A/A median={:.6} ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(&scipy_nulls),
            coefficient_of_variation(&scipy_nulls) * 100.0
        );
        let ratio_p50 = median(&ratios);
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_p50:.4}x \
             (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
             cv={:.3}% provenance only)",
            coefficient_of_variation(&ratios) * 100.0
        );
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1.0e-9))
            .max(1.0 / scipy_null_low.max(1.0e-9));
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
        Ok(())
    }
}

fn factorialk_product(n: i64, k: i64) -> f64 {
    if k < 1 || n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0;
    let mut step = n;
    while step > 0 {
        result *= step as f64;
        step -= k;
    }
    result
}

fn spherical_jn_forward_reference(n: u32, x: f64) -> f64 {
    let mut j_prev = x.sin() / x;
    if n == 0 {
        return j_prev;
    }
    let mut j_curr = x.sin() / (x * x) - x.cos() / x;
    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = next;
    }
    j_curr
}

fn riccati_jn_repeated_forward_reference(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut s = Vec::with_capacity(n as usize + 1);
    let mut sp = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        s.push(x * spherical_jn_forward_reference(k, x));
    }
    sp.push(x.cos());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        sp.push(-((k as f64) * s[k] * inv_x) + s[k - 1]);
    }
    (s, sp)
}

fn spherical_yn_repeated_reference(n: u32, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    let mut y_prev = -x.cos() / x;
    if n == 0 {
        return y_prev;
    }
    let mut y_curr = -x.cos() / (x * x) - x.sin() / x;
    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) / x * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = next;
    }
    y_curr
}

fn riccati_yn_repeated_reference(n: u32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut c = Vec::with_capacity(n as usize + 1);
    let mut cp = Vec::with_capacity(n as usize + 1);
    for k in 0..=n {
        c.push(x * spherical_yn_repeated_reference(k, x));
    }
    cp.push(x.sin());
    for k in 1..=n as usize {
        let inv_x = if x.abs() < 1.0e-300 { 0.0 } else { 1.0 / x };
        cp.push(-((k as f64) * c[k] * inv_x) + c[k - 1]);
    }
    (c, cp)
}

fn bench_factorialk_k1(c: &mut Criterion) {
    let n = 128;
    let candidate = factorialk(n, 1);
    let original = factorialk_product(n, 1);
    let rel_err = (candidate - original).abs() / original.abs();
    assert!(
        rel_err <= 2.0e-13,
        "factorialk benchmark rel_err={rel_err:e}"
    );

    let mut group = c.benchmark_group("factorialk_k1_delegate");
    group.bench_function("candidate/128", |b| {
        b.iter(|| black_box(factorialk(black_box(n), black_box(1))))
    });
    group.bench_function("original/128", |b| {
        b.iter(|| black_box(factorialk_product(black_box(n), black_box(1))))
    });
    group.finish();
}

const GAMMA_INPUTS: &[f64] = &[0.5, 1.0, 2.5, 5.0, 10.0, 50.0, 100.0];
const ERF_INPUTS: &[f64] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
const BESSEL_INPUTS: &[f64] = &[0.1, 1.0, 5.0, 10.0, 20.0, 50.0];
const ELLIPTIC_M_INPUTS: &[f64] = &[0.0, 0.5, 0.9];
const ELLIPTIC_INCOMPLETE_INPUTS: &[(f64, f64)] =
    &[(PI / 6.0, 0.0), (PI / 4.0, 0.5), (PI / 3.0, 0.9)];

fn bench_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gamma(black_box(input), RuntimeMode::Strict).expect("gamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammaln(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammaln");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammaln", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = gammaln(black_box(input), RuntimeMode::Strict).expect("gammaln");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_rgamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_rgamma");

    for &x in GAMMA_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("rgamma", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = rgamma(black_box(input), RuntimeMode::Strict).expect("rgamma");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_gammainc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_gammainc");

    let pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (10.0, 10.0)];
    for &(a, x) in pairs {
        let sa = scalar(a);
        let sx = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("gammainc", format!("a{a}_x{x}")),
            &(sa, sx),
            |b, (sa, sx)| {
                b.iter(|| {
                    let out = gammainc(black_box(sa), black_box(sx), RuntimeMode::Strict)
                        .expect("gammainc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erf(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erf");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erf", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erf(black_box(input), RuntimeMode::Strict).expect("erf");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfc(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfc");

    for &x in ERF_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfc", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfc(black_box(input), RuntimeMode::Strict).expect("erfc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_erfinv(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfinv");

    let inputs: &[f64] = &[-0.9, -0.5, 0.0, 0.5, 0.9];
    for &x in inputs {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("erfinv", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = erfinv(black_box(input), RuntimeMode::Strict).expect("erfinv");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn scipy_erfinv_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
y = np.linspace(-0.95, 0.95, n, dtype=np.float64)
sc.erfinv(y)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.erfinv(y)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy erfinv oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy erfinv oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy erfinv oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy erfinv oracle");
    if !output.status.success() {
        eprintln!(
            "scipy erfinv oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy erfinv timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy erfinv timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_erfinv_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_erfinv_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 100_000usize;
    let denom = (n - 1) as f64;
    let y: Vec<f64> = (0..n).map(|i| -0.95 + 1.9 * (i as f64) / denom).collect();
    let input = real_vec(&y);

    group.bench_function("rust_current_n100000", |b| {
        b.iter(|| {
            let out = erfinv(black_box(&input), RuntimeMode::Strict).expect("erfinv");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n100000", |b| {
            b.iter_custom(|iters| {
                scipy_erfinv_duration(n, iters)
                    .expect("scipy erfinv oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_erfinv_n100000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn bench_beta(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_beta");

    let pairs: &[(f64, f64)] = &[(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (5.0, 5.0)];
    for &(a, b_val) in pairs {
        let sa = scalar(a);
        let sb = scalar(b_val);
        group.bench_with_input(
            BenchmarkId::new("beta", format!("a{a}_b{b_val}")),
            &(sa, sb),
            |b, (sa, sb)| {
                b.iter(|| {
                    let out =
                        beta(black_box(sa), black_box(sb), RuntimeMode::Strict).expect("beta");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_jv_array(c: &mut Criterion) {
    // Array J_v(z): scalar order, large real vector — the par_map_indices fan-out
    // path. Head-to-head vs scipy.special.jv(2, z) (~104 ms at n=200k).
    let mut group = c.benchmark_group("special_bessel_jv_array");
    for &n in &[50_000usize, 200_000] {
        let zs: Vec<f64> = (0..n)
            .map(|i| (i as f64 / n as f64) * 50.0 + 0.01)
            .collect();
        let z = SpecialTensor::RealVec(zs);
        let order = SpecialTensor::RealScalar(2.0);
        group.bench_function(BenchmarkId::new("v2", n), |b| {
            b.iter(|| {
                let out = jv(black_box(&order), black_box(&z), RuntimeMode::Strict).expect("jv");
                black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_bessel_j(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_j");

    for &x in BESSEL_INPUTS {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("j0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j0(black_box(input), RuntimeMode::Strict).expect("j0");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("j1", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = j1(black_box(input), RuntimeMode::Strict).expect("j1");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_bessel_y0(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_bessel_y");

    for &x in &[0.1, 1.0, 5.0, 10.0, 20.0] {
        let input = scalar(x);
        group.bench_with_input(
            BenchmarkId::new("y0", format!("{x}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = y0(black_box(input), RuntimeMode::Strict).expect("y0");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn legacy_y0_series_small(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;
    const MAX_TERMS: usize = 96;

    let z = x * x * 0.25;
    let mut j0_term = 1.0;
    let mut j0 = 1.0;
    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        j0_term *= -z / (kf * kf);
        j0 += j0_term;
        if j0_term.abs() <= f64::EPSILON * j0.abs().max(1.0) {
            break;
        }
    }

    let mut harmonic = 0.0;
    let mut term = 1.0;
    let mut correction = 0.0;
    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        harmonic += 1.0 / kf;
        term *= -z / (kf * kf);
        let addend = -harmonic * term;
        correction += addend;
        if addend.abs() <= f64::EPSILON * correction.abs().max(1.0) {
            break;
        }
    }
    FRAC_2_PI * ((x.ln() - LN_2 + EULER_MASCHERONI) * j0 + correction)
}

fn bench_y0_small_cephes_ab(c: &mut Criterion) {
    let xs: Vec<f64> = (0..4096).map(|i| 0.01 + 4.99 * i as f64 / 4095.0).collect();
    let input = real_vec(&xs);
    let mut group = c.benchmark_group("y0_small_cephes_ab");
    group.bench_function("original_series", |b| {
        b.iter(|| {
            let out: Vec<f64> = xs
                .iter()
                .map(|&x| legacy_y0_series_small(black_box(x)))
                .collect();
            black_box(out)
        });
    });
    group.bench_function("candidate_rational", |b| {
        b.iter(|| black_box(y0(black_box(&input), RuntimeMode::Strict).expect("y0")));
    });
    group.finish();
}

fn bench_riccati_yn_recurrence(c: &mut Criterion) {
    let order = 512;
    let x = 1024.0;
    let candidate = riccati_yn(order, x);
    let original = riccati_yn_repeated_reference(order, x);
    assert!(
        candidate
            .0
            .iter()
            .chain(&candidate.1)
            .zip(original.0.iter().chain(&original.1))
            .all(|(&got, &want)| got.to_bits() == want.to_bits())
    );

    let mut group = c.benchmark_group("riccati_yn_recurrence_ab");
    group.bench_function("512/candidate", |bench| {
        bench.iter(|| black_box(riccati_yn(black_box(order), black_box(x))))
    });
    group.bench_function("512/original", |bench| {
        bench.iter(|| {
            black_box(riccati_yn_repeated_reference(
                black_box(order),
                black_box(x),
            ))
        })
    });
    group.finish();
}

fn bench_riccati_jn_recurrence(c: &mut Criterion) {
    let order = 512;
    let x = 1024.0;
    let candidate = riccati_jn(order, x);
    let original = riccati_jn_repeated_forward_reference(order, x);
    assert!(
        candidate
            .0
            .iter()
            .chain(&candidate.1)
            .zip(original.0.iter().chain(&original.1))
            .all(|(&got, &want)| got.to_bits() == want.to_bits())
    );

    let mut group = c.benchmark_group("riccati_jn_recurrence_ab");
    group.bench_function("512/candidate", |bench| {
        bench.iter(|| black_box(riccati_jn(black_box(order), black_box(x))))
    });
    group.bench_function("512/original", |bench| {
        bench.iter(|| {
            black_box(riccati_jn_repeated_forward_reference(
                black_box(order),
                black_box(x),
            ))
        })
    });
    group.finish();
}

fn bench_complete_elliptic(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_complete_elliptic");

    for &m in ELLIPTIC_M_INPUTS {
        let input = scalar(m);
        group.bench_with_input(
            BenchmarkId::new("ellipk", format!("m{m}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = ellipk(black_box(input), RuntimeMode::Strict).expect("ellipk");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ellipe", format!("m{m}")),
            &input,
            |b, input| {
                b.iter(|| {
                    let out = ellipe(black_box(input), RuntimeMode::Strict).expect("ellipe");
                    black_box(real_val(&out));
                });
            },
        );
    }

    group.finish();
}

fn bench_incomplete_elliptic(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_incomplete_elliptic");

    for &(phi, m) in ELLIPTIC_INCOMPLETE_INPUTS {
        let ellipkinc_phi_input = scalar(phi);
        let ellipkinc_m_input = scalar(m);
        let ellipeinc_phi_input = scalar(phi);
        let ellipeinc_m_input = scalar(m);
        let case = format!("phi{phi:.3}_m{m:.1}");
        group.bench_with_input(
            BenchmarkId::new("ellipkinc_scalar", &case),
            &(ellipkinc_phi_input, ellipkinc_m_input),
            |b, (phi_input, m_input)| {
                b.iter(|| {
                    let out = ellipkinc(
                        black_box(phi_input),
                        black_box(m_input),
                        RuntimeMode::Strict,
                    )
                    .expect("ellipkinc");
                    black_box(real_val(&out));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ellipeinc_scalar", case),
            &(ellipeinc_phi_input, ellipeinc_m_input),
            |b, (phi_input, m_input)| {
                b.iter(|| {
                    let out = ellipeinc(
                        black_box(phi_input),
                        black_box(m_input),
                        RuntimeMode::Strict,
                    )
                    .expect("ellipeinc");
                    black_box(real_val(&out));
                });
            },
        );
    }

    let broadcast_m = (scalar(PI / 3.0), real_vec(&[0.0, 0.25, 0.5, 0.75]));
    group.bench_with_input(
        BenchmarkId::new("ellipkinc_broadcast_m", "scalar_phi_over_m_vec"),
        &broadcast_m,
        |b, (phi_input, m_input)| {
            b.iter(|| {
                let out = ellipkinc(
                    black_box(phi_input),
                    black_box(m_input),
                    RuntimeMode::Strict,
                )
                .expect("ellipkinc broadcast over m");
                consume_tensor(out);
            });
        },
    );

    let pairwise = (
        real_vec(&[PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0 - 0.1]),
        real_vec(&[0.0, 0.25, 0.5, 0.75]),
    );
    group.bench_with_input(
        BenchmarkId::new("ellipeinc_pairwise_vec", "phi_vec_m_vec"),
        &pairwise,
        |b, (phi_input, m_input)| {
            b.iter(|| {
                let out = ellipeinc(
                    black_box(phi_input),
                    black_box(m_input),
                    RuntimeMode::Strict,
                )
                .expect("ellipeinc pairwise vector");
                consume_tensor(out);
            });
        },
    );

    group.finish();
}

fn scipy_special_available() -> bool {
    let script = "import scipy.special\n";
    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn scipy special availability check");
    child
        .stdin
        .as_mut()
        .expect("open scipy special availability stdin")
        .write_all(script.as_bytes())
        .expect("write scipy special availability script");
    child.wait().is_ok_and(|status| status.success())
}

fn scipy_ndtri_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
q = np.linspace(1e-12, 1.0 - 1e-12, n, dtype=np.float64)
sc.ndtri(q)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.ndtri(q)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy ndtri oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy ndtri oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy ndtri oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy ndtri oracle");
    if !output.status.success() {
        eprintln!(
            "scipy ndtri oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy ndtri timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy ndtri timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_ndtri_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_ndtri_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 500_000usize;
    let denom = (n - 1) as f64;
    let q: Vec<f64> = (0..n)
        .map(|i| 1.0e-12 + (1.0 - 2.0e-12) * (i as f64) / denom)
        .collect();
    let input = real_vec(&q);

    group.bench_function("rust_current_n500000", |b| {
        b.iter(|| {
            let out = ndtri(black_box(&input), RuntimeMode::Strict).expect("ndtri");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n500000", |b| {
            b.iter_custom(|iters| {
                scipy_ndtri_duration(n, iters)
                    .expect("scipy ndtri oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_ndtri_n500000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn bench_special_log_ndtr_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_log_ndtr_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    let n = 500_000usize;
    let denom = (n - 1) as f64;
    let x: Vec<f64> = (0..n).map(|i| -8.0 + 16.0 * (i as f64) / denom).collect();
    let input = real_vec(&x);

    group.bench_function("fast_erfc_simd_n500000", |b| {
        b.iter(|| {
            let out = log_ndtr(black_box(&input), RuntimeMode::Strict).expect("log_ndtr");
            black_box(out);
        });
    });

    group.bench_function("orig_scalar_map_n500000", |b| {
        b.iter(|| {
            let out: Vec<f64> = black_box(&x)
                .iter()
                .map(|&value| log_ndtr_scalar(value))
                .collect();
            black_box(out);
        });
    });

    group.finish();
}

fn scipy_zeta_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
s = 1.1 + np.arange(n, dtype=np.float64) * (8.9 / max(n - 1, 1))
sc.zeta(s)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.zeta(s)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy zeta oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy zeta oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy zeta oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy zeta oracle");
    if !output.status.success() {
        eprintln!(
            "scipy zeta oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy zeta timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy zeta timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_zeta_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_zeta_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 100_000usize;
    let denom = (n - 1).max(1) as f64;
    let values: Vec<f64> = (0..n).map(|i| 1.1 + 8.9 * (i as f64) / denom).collect();
    let input = real_vec(&values);

    group.bench_function("rust_scalar_loop_n100000", |b| {
        b.iter(|| {
            let out: Vec<f64> = values.iter().copied().map(zeta_scalar).collect();
            black_box(out);
        });
    });

    group.bench_function("rust_tensor_n100000", |b| {
        b.iter(|| {
            let out = zeta(black_box(&input), RuntimeMode::Strict).expect("zeta");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n100000", |b| {
            b.iter_custom(|iters| {
                scipy_zeta_duration(n, iters)
                    .expect("scipy zeta oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_zeta_n100000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn scipy_hyperu_a1_gamma_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

n = int(sys.argv[1])
iters = int(sys.argv[2])
x = np.linspace(0.5, 8.5, n, dtype=np.float64)
sc.hyperu(1.0, 1.25, x)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    out = sc.hyperu(1.0, 1.25, x)
    checksum += float(out[0] + out[n // 2] + out[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy hyperu oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy hyperu oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy hyperu oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy hyperu oracle");
    if !output.status.success() {
        eprintln!(
            "scipy hyperu oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy hyperu timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy hyperu timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_special_hyperu_a1_gamma_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_hyperu_a1_gamma_array");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let n = 50_000usize;
    let denom = (n - 1).max(1) as f64;
    let x_values: Vec<f64> = (0..n).map(|i| 0.5 + 8.0 * (i as f64) / denom).collect();
    let a = scalar(1.0);
    let b_param = scalar(1.25);
    let x = real_vec(&x_values);

    group.bench_function("rust_current_n50000", |b| {
        b.iter(|| {
            let out = hyperu(
                black_box(&a),
                black_box(&b_param),
                black_box(&x),
                RuntimeMode::Strict,
            )
            .expect("hyperu");
            black_box(out);
        });
    });

    if scipy_special_available() {
        group.bench_function("scipy_n50000", |b| {
            b.iter_custom(|iters| {
                scipy_hyperu_a1_gamma_duration(n, iters)
                    .expect("scipy hyperu oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping scipy_hyperu_a1_gamma_n50000: python3 cannot import scipy.special");
    }

    group.finish();
}

fn scipy_jnjnp_zeros_duration(nt: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.special as sc

nt = int(sys.argv[1])
iters = int(sys.argv[2])
sc.jnjnp_zeros(nt)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    zo, n, m, t = sc.jnjnp_zeros(nt)
    checksum += float(zo[-1]) + float(n[-1]) + float(m[-1]) + float(t[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &nt.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy jnjnp_zeros oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy jnjnp_zeros oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy jnjnp_zeros oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy jnjnp_zeros oracle");
    if !output.status.success() {
        eprintln!(
            "scipy jnjnp_zeros oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy jnjnp_zeros timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy jnjnp_zeros timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn legacy_duplicate_jnjnp_zeros(nt: usize) -> (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>) {
    if nt == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let mut cands: Vec<(f64, i32, i32, i32)> = Vec::new();
    cands.push((0.0, 0, 0, 1));
    let per = nt + 2;
    let n_max = nt as u32 + 2;
    for n in 0..=n_max {
        for (i, &x) in jn_zeros(n, per).iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 0));
        }
        let jp = if n == 0 {
            jn_zeros(1, per)
        } else {
            jnp_zeros(n, per)
        };
        for (i, &x) in jp.iter().enumerate() {
            cands.push((x, n as i32, (i + 1) as i32, 1));
        }
    }
    cands.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("Bessel zeros are finite")
            .then(a.3.cmp(&b.3))
    });
    cands.truncate(nt);
    let zo = cands.iter().map(|c| c.0).collect();
    let n = cands.iter().map(|c| c.1).collect();
    let m = cands.iter().map(|c| c.2).collect();
    let t = cands.iter().map(|c| c.3).collect();
    (zo, n, m, t)
}

fn bench_acoco_gauntlet_jnjnp_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("acoco_gauntlet_jnjnp_zeros");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for &nt in &[64_usize, 128_usize] {
        group.bench_function(format!("rust_current_nt{nt}"), |b| {
            b.iter(|| {
                let (zo, n, m, t) = jnjnp_zeros(black_box(nt));
                black_box((zo, n, m, t));
            });
        });
        group.bench_function(format!("rust_legacy_duplicate_nt{nt}"), |b| {
            b.iter(|| {
                let (zo, n, m, t) = legacy_duplicate_jnjnp_zeros(black_box(nt));
                black_box((zo, n, m, t));
            });
        });
        if scipy_special_available() {
            group.bench_function(format!("scipy_nt{nt}"), |b| {
                b.iter_custom(|iters| {
                    scipy_jnjnp_zeros_duration(nt, iters)
                        .expect("scipy jnjnp_zeros oracle should run after availability check")
                });
            });
        } else {
            eprintln!("skipping scipy_nt{nt}: python3 cannot import scipy.special");
        }
    }
    group.finish();
}

/// Array (RealVec) dispatch — the realistic ufunc workload. fsci parallelizes the per-family
/// array path; scipy.special is vectorized single-core C. Head-to-head vs scipy.
fn bench_array(c: &mut Criterion) {
    let xs: Vec<f64> = (0..65536).map(|i| 0.5 + (i as f64) * 0.0001).collect();
    let t = real_vec(&xs);
    let mut group = c.benchmark_group("special_array_65536");
    group.bench_function("gamma", |b| {
        b.iter(|| gamma(black_box(&t), RuntimeMode::Strict).expect("gamma"))
    });
    group.bench_function("erf", |b| {
        b.iter(|| erf(black_box(&t), RuntimeMode::Strict).expect("erf"))
    });
    group.bench_function("j0", |b| {
        b.iter(|| j0(black_box(&t), RuntimeMode::Strict).expect("j0"))
    });
    group.finish();
}

fn legacy_pbdv_positive(v: f64, x: f64) -> (f64, f64) {
    let d = legacy_parabolic_cylinder_d_positive(v, x);
    let d_next = legacy_parabolic_cylinder_d_positive(v + 1.0, x);
    (d, 0.5 * x * d - d_next)
}

fn legacy_parabolic_cylinder_d_positive(v: f64, x: f64) -> f64 {
    let coef = 2.0_f64.powf(v / 2.0) * (-x * x / 4.0).exp();
    let z = x * x / 2.0;
    coef * hyperu_scalar(-v / 2.0, 0.5, z, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

fn bench_pbdv_integer_gauntlet(c: &mut Criterion) {
    let xs: Vec<f64> = (0..20_000)
        .map(|i| 0.1 + 9.9 * (i as f64) / 19_999.0)
        .collect();
    let mut group = c.benchmark_group("pbdv_integer_gauntlet");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("v2_current_integer_scalar_loop", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &x in &xs {
                let (d, dp) = pbdv(black_box(2.0), black_box(x));
                acc += d + dp;
            }
            black_box(acc);
        })
    });

    group.bench_function("v2_orig_hyperu_scalar_loop", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &x in &xs {
                let (d, dp) = legacy_pbdv_positive(black_box(2.0), black_box(x));
                acc += d + dp;
            }
            black_box(acc);
        })
    });

    group.bench_function("v2_current_many", |b| {
        b.iter(|| black_box(pbdv_many(black_box(2.0), black_box(&xs))))
    });

    group.finish();
}

fn legacy_spence_scalar(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x < 0.0 {
        return f64::NAN;
    }
    legacy_dilog_real(1.0 - x)
}

fn legacy_dilog_real(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == 1.0 {
        return PI * PI / 6.0;
    }
    if z == 0.0 {
        return 0.0;
    }
    if z < 0.0 {
        let log_term = (1.0 - z).ln();
        let transformed = z / (z - 1.0);
        return -legacy_dilog_real(transformed) - 0.5 * log_term * log_term;
    }
    if z > 0.5 {
        let complement = 1.0 - z;
        return PI * PI / 6.0 - z.ln() * complement.ln() - legacy_dilog_series(complement);
    }
    legacy_dilog_series(z)
}

fn legacy_dilog_series(z: f64) -> f64 {
    let mut term = z;
    let mut sum = z;
    for k in 2..=128usize {
        term *= z;
        let kf = k as f64;
        let addend = term / (kf * kf);
        sum += addend;
        if addend.abs() <= f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

fn bench_spence_cephes_gauntlet(c: &mut Criterion) {
    let xs: Vec<f64> = (0..200_000)
        .map(|i| 0.05 + 9.95 * (i as f64) / 199_999.0)
        .collect();
    let mut group = c.benchmark_group("spence_cephes_gauntlet");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("current_cephes_scalar_loop", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &x in &xs {
                acc += spence_scalar(black_box(x));
            }
            black_box(acc);
        })
    });

    group.bench_function("orig_series_scalar_loop", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &x in &xs {
                acc += legacy_spence_scalar(black_box(x));
            }
            black_box(acc);
        })
    });

    group.finish();
}

fn bench_ncfdtr(c: &mut Criterion) {
    let mut group = c.benchmark_group("ncfdtr");
    // cost scales with nc: the Poisson(nc/2) mixture spans ~O(√nc) incomplete-beta
    // terms. Evaluate a batch of f for each (dfn, dfd, nc).
    let fs: Vec<f64> = (0..256).map(|i| 0.05 + (i as f64) * 0.02).collect();
    for &(dfn, dfd, nc) in &[
        (10.0_f64, 10.0_f64, 2.0_f64),
        (20.0, 30.0, 200.0),
        (50.0, 50.0, 2000.0),
    ] {
        group.bench_function(
            BenchmarkId::new("cdf", format!("dfn{dfn}_dfd{dfd}_nc{nc}")),
            |b| {
                b.iter(|| {
                    fs.iter()
                        .map(|&f| {
                            fsci_special::ncfdtr(
                                black_box(dfn),
                                black_box(dfd),
                                black_box(nc),
                                black_box(f),
                            )
                        })
                        .sum::<f64>()
                })
            },
        );
    }
    group.finish();
}

fn bench_mathieu_periodic_cache_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("mathieu_periodic_cache_gauntlet");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    let xs: Vec<f64> = (0..256).map(|i| (i as f64) * 180.0 / 255.0).collect();
    let cases: &[(&str, u32, f64, bool)] = &[
        ("cem_m3_q10", 3, 10.0, true),
        ("cem_m4_q50", 4, 50.0, true),
        ("sem_m3_q50", 3, 50.0, false),
        ("sem_m4_q20", 4, 20.0, false),
    ];

    for &(name, m, q, even) in cases {
        MATHIEU_PERIODIC_CACHE_DISABLE_FOR_BENCH.store(false, Ordering::Relaxed);
        for &x in xs.iter().take(4) {
            let _ = if even {
                mathieu_cem(m, q, x)
            } else {
                mathieu_sem(m, q, x)
            };
        }

        group.bench_function(BenchmarkId::new("cached", name), |b| {
            MATHIEU_PERIODIC_CACHE_DISABLE_FOR_BENCH.store(false, Ordering::Relaxed);
            b.iter(|| {
                let mut acc = 0.0_f64;
                for &x in &xs {
                    let (value, deriv) = if even {
                        mathieu_cem(black_box(m), black_box(q), black_box(x))
                    } else {
                        mathieu_sem(black_box(m), black_box(q), black_box(x))
                    };
                    acc += value + deriv;
                }
                black_box(acc);
            });
        });
        group.bench_function(BenchmarkId::new("orig_recompute", name), |b| {
            MATHIEU_PERIODIC_CACHE_DISABLE_FOR_BENCH.store(true, Ordering::Relaxed);
            b.iter(|| {
                let mut acc = 0.0_f64;
                for &x in &xs {
                    let (value, deriv) = if even {
                        mathieu_cem(black_box(m), black_box(q), black_box(x))
                    } else {
                        mathieu_sem(black_box(m), black_box(q), black_box(x))
                    };
                    acc += value + deriv;
                }
                black_box(acc);
            });
        });
    }

    MATHIEU_PERIODIC_CACHE_DISABLE_FOR_BENCH.store(false, Ordering::Relaxed);
    group.finish();
}

fn kelvin_zeros_serial(nt: u32) -> [Vec<f64>; 8] {
    [
        ber_zeros(nt),
        bei_zeros(nt),
        ker_zeros(nt),
        kei_zeros(nt),
        berp_zeros(nt),
        beip_zeros(nt),
        kerp_zeros(nt),
        keip_zeros(nt),
    ]
}

fn bench_kelvin_zeros_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("kelvin_zeros_ab");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for nt in [4, 10, 32] {
        group.bench_with_input(BenchmarkId::new("serial_families", nt), &nt, |b, &nt| {
            b.iter(|| black_box(kelvin_zeros_serial(black_box(nt))));
        });
        group.bench_with_input(BenchmarkId::new("wrapper", nt), &nt, |b, &nt| {
            b.iter(|| black_box(kelvin_zeros(black_box(nt))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_factorialk_k1,
    bench_kelvin_zeros_ab,
    bench_mathieu_periodic_cache_gauntlet,
    bench_spence_cephes_gauntlet,
    bench_pbdv_integer_gauntlet,
    bench_ncfdtr,
    bench_array,
    bench_gamma,
    bench_gammaln,
    bench_rgamma,
    bench_gammainc,
    bench_erf,
    bench_erfc,
    bench_erfinv,
    bench_special_erfinv_array,
    bench_special_log_ndtr_array,
    bench_special_ndtri_array,
    bench_special_zeta_array,
    bench_special_hyperu_a1_gamma_array,
    bench_beta,
    bench_bessel_jv_array,
    bench_bessel_j,
    bench_bessel_y0,
    bench_y0_small_cephes_ab,
    bench_riccati_jn_recurrence,
    bench_riccati_yn_recurrence,
    bench_complete_elliptic,
    bench_incomplete_elliptic,
    bench_acoco_gauntlet_jnjnp_zeros
);
fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments
        .iter()
        .any(|argument| argument == "--live-scipy-kv")
    {
        return match live_kv::run(&arguments) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("ABORT: {error}");
                ExitCode::FAILURE
            }
        };
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
    ExitCode::SUCCESS
}
