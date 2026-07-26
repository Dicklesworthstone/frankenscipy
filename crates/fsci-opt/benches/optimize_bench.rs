#![forbid(unsafe_code)]
//! Criterion benchmarks for fsci-opt (P2C-003-H).
//!
//! Groups: bfgs, lbfgsb, cg, powell, brentq, brenth, bisect, ridder

use std::hint::black_box;
use std::io::{self, Write as _};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group};
use fsci_opt::DifferentialEvolutionOptions;
use fsci_opt::{
    LeastSquaresOptions, MinimizeOptions, OptimizeMethod, RootMethod, RootOptions, bfgs, bisect,
    brenth, brentq, cg_pr_plus, differential_evolution, lbfgsb, least_squares,
    linear_sum_assignment, numerical_gradient, numerical_jacobian, powell, ridder,
};
use fsci_runtime::RuntimeMode;
use rand::{Rng, RngExt, SeedableRng};
use sha2::{Digest, Sha256};

const FRONTIER_ROUNDS: usize = 41;
const FRONTIER_MIN_OF: usize = 3;
const FRONTIER_BOOTSTRAP_RESAMPLES: usize = 10_000;
const FRONTIER_MIN_SAMPLE_MS: f64 = 2.0;

// ── Test functions ────────────────────────────────────────────────────

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    s
}

fn rosenbrock_gradient(x: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() - 1 {
        let residual = x[i + 1] - x[i] * x[i];
        grad[i] += -400.0 * x[i] * residual - 2.0 * (1.0 - x[i]);
        grad[i + 1] += 200.0 * residual;
    }
    grad
}

fn quadratic(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn cubic_root(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

fn sin_root(x: f64) -> f64 {
    x.sin()
}

fn opts(method: OptimizeMethod) -> MinimizeOptions {
    MinimizeOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

#[derive(Clone, Copy)]
enum TrustExactArm {
    Pivoted,
    Cholesky,
}

impl TrustExactArm {
    const fn cholesky_disabled(self) -> bool {
        matches!(self, Self::Pivoted)
    }
}

struct FrontierPairedStats {
    arm_a_median_ms: f64,
    arm_b_median_ms: f64,
    arm_a_mad_ms: f64,
    arm_b_mad_ms: f64,
    arm_a_cv: f64,
    arm_b_cv: f64,
    ratio_median: f64,
    ratio_mad: f64,
    ratio_cv: f64,
    ratio_ci_low: f64,
    ratio_ci_high: f64,
    ratios: Vec<f64>,
    checksum: u64,
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
    assert!(!values.is_empty(), "median requires at least one sample");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) * 0.5
    } else {
        sorted[midpoint]
    }
}

fn median_absolute_deviation(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations)
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squared_deviations = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    let standard_deviation =
        (sum_squared_deviations / values.len().saturating_sub(1).max(1) as f64).sqrt();
    standard_deviation / mean
}

fn bootstrap_median_ci(values: &[f64], seed: u64) -> (f64, f64) {
    let mut generator = XorShift64(seed);
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(FRONTIER_BOOTSTRAP_RESAMPLES);
    for _ in 0..FRONTIER_BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            sample.push(values[generator.next() as usize % values.len()]);
        }
        medians.push(median(&sample));
    }
    medians.sort_by(f64::total_cmp);
    let low = FRONTIER_BOOTSTRAP_RESAMPLES * 25 / 1_000;
    let high = FRONTIER_BOOTSTRAP_RESAMPLES * 975 / 1_000;
    (medians[low], medians[high.min(medians.len() - 1)])
}

fn trust_exact_checksum(result: &fsci_opt::OptimizeResult) -> u64 {
    let mut checksum = 0x6a09_e667_f3bc_c909_u64;
    for &value in &result.x {
        checksum = checksum.rotate_left(9) ^ value.to_bits();
    }
    if let Some(value) = result.fun {
        checksum = checksum.rotate_left(11) ^ value.to_bits();
    }
    for &value in result.jac.as_deref().unwrap_or_default() {
        checksum = checksum.rotate_left(13) ^ value.to_bits();
    }
    checksum ^= (result.success as u64) << 63;
    checksum ^= (result.nfev as u64).rotate_left(7);
    checksum ^= (result.njev as u64).rotate_left(17);
    checksum ^= (result.nhev as u64).rotate_left(29);
    checksum ^ (result.nit as u64).rotate_left(41)
}

fn time_trust_exact_arm(
    x0: &[f64],
    arm: TrustExactArm,
    repetitions: usize,
) -> Result<(f64, u64), String> {
    use fsci_opt::{TRUST_EXACT_CHOLESKY_DISABLE, trust_exact};
    use std::sync::atomic::Ordering;

    TRUST_EXACT_CHOLESKY_DISABLE.store(arm.cholesky_disabled(), Ordering::Relaxed);
    let started = Instant::now();
    let mut checksum = 0xbb67_ae85_84ca_a73b_u64;
    for repetition in 0..repetitions {
        let result = black_box(
            trust_exact(&rosenbrock, black_box(x0), opts(OptimizeMethod::TrustExact))
                .map_err(|error| format!("timed trust-exact solve failed: {error:?}"))?,
        );
        checksum = checksum
            .rotate_left(7)
            .wrapping_add(trust_exact_checksum(&result))
            .wrapping_add(repetition as u64 + 1);
        black_box(checksum);
    }
    Ok((started.elapsed().as_secs_f64() * 1_000.0, checksum))
}

fn min_trust_exact_sample(
    x0: &[f64],
    arm: TrustExactArm,
    repetitions: usize,
) -> Result<(f64, u64), String> {
    let mut best_ms = f64::INFINITY;
    let mut best_checksum = 0;
    for _ in 0..FRONTIER_MIN_OF {
        let (elapsed_ms, checksum) = time_trust_exact_arm(x0, arm, repetitions)?;
        if elapsed_ms < best_ms {
            best_ms = elapsed_ms;
            best_checksum = checksum;
        }
    }
    Ok((best_ms, best_checksum))
}

fn calibrate_trust_exact_repetitions(x0: &[f64]) -> Result<usize, String> {
    let mut repetitions = 1usize;
    loop {
        let (elapsed_ms, checksum) =
            time_trust_exact_arm(x0, TrustExactArm::Cholesky, repetitions)?;
        black_box(checksum);
        if elapsed_ms >= FRONTIER_MIN_SAMPLE_MS {
            return Ok(repetitions);
        }
        repetitions = repetitions
            .checked_mul(2)
            .ok_or_else(|| String::from("frontier calibration repetition count overflowed"))?;
    }
}

fn paired_trust_exact(
    x0: &[f64],
    arm_a: TrustExactArm,
    arm_b: TrustExactArm,
    repetitions: usize,
    bootstrap_seed: u64,
) -> Result<FrontierPairedStats, String> {
    let mut arm_a_ms = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut arm_b_ms = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut ratios = Vec::with_capacity(FRONTIER_ROUNDS);
    let mut combined_checksum = 0u64;

    for round in 0..FRONTIER_ROUNDS {
        let ((a_ms, a_checksum), (b_ms, b_checksum)) = if round.is_multiple_of(2) {
            (
                min_trust_exact_sample(x0, arm_a, repetitions)?,
                min_trust_exact_sample(x0, arm_b, repetitions)?,
            )
        } else {
            let b = min_trust_exact_sample(x0, arm_b, repetitions)?;
            let a = min_trust_exact_sample(x0, arm_a, repetitions)?;
            (a, b)
        };
        arm_a_ms.push(a_ms);
        arm_b_ms.push(b_ms);
        ratios.push(a_ms / b_ms);
        combined_checksum = combined_checksum
            .rotate_left(7)
            .wrapping_add(a_checksum.rotate_left(17))
            .wrapping_add(b_checksum.rotate_right(11))
            .wrapping_add(round as u64 + 1);
    }

    let (ratio_ci_low, ratio_ci_high) = bootstrap_median_ci(&ratios, bootstrap_seed);
    Ok(FrontierPairedStats {
        arm_a_median_ms: median(&arm_a_ms),
        arm_b_median_ms: median(&arm_b_ms),
        arm_a_mad_ms: median_absolute_deviation(&arm_a_ms),
        arm_b_mad_ms: median_absolute_deviation(&arm_b_ms),
        arm_a_cv: coefficient_of_variation(&arm_a_ms),
        arm_b_cv: coefficient_of_variation(&arm_b_ms),
        ratio_median: median(&ratios),
        ratio_mad: median_absolute_deviation(&ratios),
        ratio_cv: coefficient_of_variation(&ratios),
        ratio_ci_low,
        ratio_ci_high,
        ratios,
        checksum: combined_checksum,
    })
}

fn print_frontier_paired_stats(label: &str, stats: &FrontierPairedStats) {
    println!(
        "paired label={label} rounds={FRONTIER_ROUNDS} min_of={FRONTIER_MIN_OF} \
         arm_a_median_ms={:.9} arm_b_median_ms={:.9} arm_a_mad_ms={:.9} \
         arm_b_mad_ms={:.9} ratio_median={:.9} ratio_mad={:.9} \
         ratio_median_ci95=[{:.9},{:.9}] arm_a_cv_provenance={:.6} \
         arm_b_cv_provenance={:.6} ratio_cv_provenance={:.6} checksum={:016x}",
        stats.arm_a_median_ms,
        stats.arm_b_median_ms,
        stats.arm_a_mad_ms,
        stats.arm_b_mad_ms,
        stats.ratio_median,
        stats.ratio_mad,
        stats.ratio_ci_low,
        stats.ratio_ci_high,
        stats.arm_a_cv,
        stats.arm_b_cv,
        stats.ratio_cv,
        stats.checksum
    );
    print!("paired_ratios label={label}");
    for ratio in &stats.ratios {
        print!(" {ratio:.9}");
    }
    println!();
}

fn prove_trust_exact_cholesky_contract(x0: &[f64]) -> Result<(), String> {
    use fsci_opt::{TRUST_EXACT_CHOLESKY_DISABLE, trust_exact};
    use std::sync::atomic::Ordering;

    TRUST_EXACT_CHOLESKY_DISABLE.store(true, Ordering::Relaxed);
    let baseline = trust_exact(&rosenbrock, x0, opts(OptimizeMethod::TrustExact))
        .map_err(|error| format!("pivoted proof solve failed: {error:?}"))?;
    TRUST_EXACT_CHOLESKY_DISABLE.store(false, Ordering::Relaxed);
    let candidate = trust_exact(&rosenbrock, x0, opts(OptimizeMethod::TrustExact))
        .map_err(|error| format!("Cholesky proof solve failed: {error:?}"))?;

    if candidate.success != baseline.success
        || candidate.status != baseline.status
        || candidate.message != baseline.message
    {
        return Err(format!(
            "trust-exact terminal contract changed: baseline={:?}/{:?}/{:?}, \
             candidate={:?}/{:?}/{:?}",
            baseline.success,
            baseline.status,
            baseline.message,
            candidate.success,
            candidate.status,
            candidate.message
        ));
    }
    let max_abs_x = candidate
        .x
        .iter()
        .zip(&baseline.x)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let fun_abs = match (candidate.fun, baseline.fun) {
        (Some(left), Some(right)) => (left - right).abs(),
        (None, None) => 0.0,
        _ => return Err(String::from("trust-exact objective presence changed")),
    };
    if max_abs_x > 1.0e-5 || fun_abs > 1.0e-10 {
        return Err(format!(
            "trust-exact numerical contract changed: max_abs_x={max_abs_x:.3e}, \
             fun_abs={fun_abs:.3e}"
        ));
    }
    println!(
        "frontier_contract terminal_equal=true max_abs_x={max_abs_x:.12e} \
         fun_abs={fun_abs:.12e} baseline_nfev={} candidate_nfev={} \
         baseline_njev={} candidate_njev={} baseline_nhev={} candidate_nhev={} \
         baseline_nit={} candidate_nit={}",
        baseline.nfev,
        candidate.nfev,
        baseline.njev,
        candidate.njev,
        baseline.nhev,
        candidate.nhev,
        baseline.nit,
        candidate.nit
    );
    Ok(())
}

fn run_trust_exact_cholesky_frontier() -> Result<bool, String> {
    use fsci_opt::TRUST_EXACT_CHOLESKY_DISABLE;
    use std::sync::atomic::Ordering;

    const DIMENSION: usize = 20;
    let x0 = vec![0.0; DIMENSION];
    prove_trust_exact_cholesky_contract(&x0)?;
    let repetitions = calibrate_trust_exact_repetitions(&x0)?;

    for round in 0usize..4 {
        if round.is_multiple_of(2) {
            black_box(time_trust_exact_arm(
                &x0,
                TrustExactArm::Pivoted,
                repetitions,
            )?);
            black_box(time_trust_exact_arm(
                &x0,
                TrustExactArm::Cholesky,
                repetitions,
            )?);
        } else {
            black_box(time_trust_exact_arm(
                &x0,
                TrustExactArm::Cholesky,
                repetitions,
            )?);
            black_box(time_trust_exact_arm(
                &x0,
                TrustExactArm::Pivoted,
                repetitions,
            )?);
        }
    }

    println!(
        "frontier_fixture name=trust_exact_spd_cholesky dimension={DIMENSION} \
         repetitions={repetitions} min_sample_ms={FRONTIER_MIN_SAMPLE_MS} \
         solver_fallback=pivoted_gauss_jordan"
    );
    let null = paired_trust_exact(
        &x0,
        TrustExactArm::Pivoted,
        TrustExactArm::Pivoted,
        repetitions,
        0x243f_6a88_85a3_08d3,
    )?;
    let candidate = paired_trust_exact(
        &x0,
        TrustExactArm::Pivoted,
        TrustExactArm::Cholesky,
        repetitions,
        0x1319_8a2e_0370_7344,
    )?;
    TRUST_EXACT_CHOLESKY_DISABLE.store(false, Ordering::Relaxed);
    print_frontier_paired_stats("aa_pivoted_pivoted", &null);
    print_frontier_paired_stats("ab_pivoted_cholesky", &candidate);

    let null_half_width = (1.0 - null.ratio_ci_low)
        .abs()
        .max((null.ratio_ci_high - 1.0).abs());
    let decision_floor = (1.0 + 2.0 * null_half_width).max(1.01);
    let keep = candidate.ratio_ci_low > decision_floor;
    println!(
        "frontier_gate decision={} decision_basis=bootstrap_median_ci_vs_2x_aa \
         bootstrap_resamples={FRONTIER_BOOTSTRAP_RESAMPLES} aa_ci95=[{:.9},{:.9}] \
         aa_half_width={null_half_width:.9} decision_floor={decision_floor:.9} \
         candidate_ratio_median={:.9} candidate_ratio_median_ci95=[{:.9},{:.9}] \
         cv_used_for_decision=false",
        if keep { "KEEP" } else { "REJECT" },
        null.ratio_ci_low,
        null.ratio_ci_high,
        candidate.ratio_median,
        candidate.ratio_ci_low,
        candidate.ratio_ci_high
    );
    Ok(keep)
}

fn report_bench_elf_sha256() -> Result<(), String> {
    let identity = (|| {
        let executable = std::env::current_exe()?;
        let bytes = std::fs::read(executable)?;
        Ok::<_, io::Error>(format!("{:x}", Sha256::digest(bytes)))
    })();
    match &identity {
        Ok(hash) => println!("bench_elf_sha256={hash}"),
        Err(_) => println!("bench_elf_sha256=unavailable"),
    }
    io::stdout()
        .flush()
        .map_err(|error| format!("failed to flush benchmark identity: {error}"))?;
    identity
        .map(|_| ())
        .map_err(|error| format!("failed to hash benchmark executable: {error}"))
}

fn lbfgsb_opts() -> MinimizeOptions {
    MinimizeOptions {
        method: Some(OptimizeMethod::LBfgsB),
        mode: RuntimeMode::Strict,
        tol: Some(1.0e-8),
        maxiter: Some(2000),
        ..Default::default()
    }
}

fn root_opts(method: RootMethod) -> RootOptions {
    RootOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
        ..Default::default()
    }
}

// ── Minimize benchmarks ──────────────────────────────────────────────

fn bench_bfgs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfgs");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = bfgs(&rosenbrock, x0, opts(OptimizeMethod::Bfgs));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_exact_gradient", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let mut options = opts(OptimizeMethod::Bfgs);
                    options.gradient = Some(rosenbrock_gradient);
                    let _ = bfgs(&rosenbrock, x0, options);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = bfgs(&quadratic, x0, opts(OptimizeMethod::Bfgs));
                });
            },
        );
    }
    group.finish();
}

fn bench_lbfgsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgsb");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    let x0 = vec![-1.2, 1.0];
    group.bench_with_input(
        BenchmarkId::new("rosenbrock_unconstrained_fd", 2usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&rosenbrock, x0, lbfgsb_opts(), None);
            });
        },
    );

    let x0: Vec<f64> = (0..10)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect();
    group.bench_with_input(
        BenchmarkId::new("rosenbrock_unconstrained_fd", 10usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&rosenbrock, x0, lbfgsb_opts(), None);
            });
        },
    );

    let x0: Vec<f64> = (0..32).map(|i| (i as f64 % 7.0) - 3.0).collect();
    group.bench_with_input(
        BenchmarkId::new("quadratic_unconstrained_fd", 32usize),
        &x0,
        |b, x0| {
            b.iter(|| {
                let _ = lbfgsb(&quadratic, x0, lbfgsb_opts(), None);
            });
        },
    );
    group.finish();
}

fn bench_cg(c: &mut Criterion) {
    let mut group = c.benchmark_group("cg");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = cg_pr_plus(&rosenbrock, x0, opts(OptimizeMethod::ConjugateGradient));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_exact_gradient", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let mut options = opts(OptimizeMethod::ConjugateGradient);
                    options.gradient = Some(rosenbrock_gradient);
                    let _ = cg_pr_plus(&rosenbrock, x0, options);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = cg_pr_plus(&quadratic, x0, opts(OptimizeMethod::ConjugateGradient));
                });
            },
        );
    }
    group.finish();
}

fn bench_powell(c: &mut Criterion) {
    let mut group = c.benchmark_group("powell");
    for &dim in &[2usize, 5, 10] {
        let x0: Vec<f64> = vec![0.0; dim];
        group.bench_with_input(
            BenchmarkId::new("rosenbrock", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = powell(&rosenbrock, x0, opts(OptimizeMethod::Powell));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quadratic", dim),
            &(x0.clone()),
            |b, x0| {
                b.iter(|| {
                    let _ = powell(&quadratic, x0, opts(OptimizeMethod::Powell));
                });
            },
        );
    }
    group.finish();
}

fn bench_brentq(c: &mut Criterion) {
    let mut group = c.benchmark_group("brentq");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = brentq(sin_root, (3.0, 4.0), root_opts(RootMethod::Brentq));
        });
    });
    group.finish();
}

fn bench_brenth(c: &mut Criterion) {
    let mut group = c.benchmark_group("brenth");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = brenth(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brenth));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = brenth(sin_root, (3.0, 4.0), root_opts(RootMethod::Brenth));
        });
    });
    group.finish();
}

fn bench_bisect(c: &mut Criterion) {
    let mut group = c.benchmark_group("bisect");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = bisect(cubic_root, (1.0, 3.0), root_opts(RootMethod::Bisect));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = bisect(sin_root, (3.0, 4.0), root_opts(RootMethod::Bisect));
        });
    });
    group.finish();
}

fn bench_ridder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ridder");
    group.bench_function("cubic", |b| {
        b.iter(|| {
            let _ = ridder(cubic_root, (1.0, 3.0), root_opts(RootMethod::Ridder));
        });
    });
    group.bench_function("sin", |b| {
        b.iter(|| {
            let _ = ridder(sin_root, (3.0, 4.0), root_opts(RootMethod::Ridder));
        });
    });
    group.finish();
}

fn bench_least_squares(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_squares");
    group.bench_function("rosenbrock_residual", |b| {
        b.iter(|| {
            let residuals = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
            let _ = least_squares(residuals, &[-1.2, 1.0], LeastSquaresOptions::default());
        });
    });

    let xs: Vec<f64> = (0..64).map(|i| i as f64 * 0.125).collect();
    let truth = [2.5_f64, 1.3, 0.5];
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| truth[0] * (-truth[1] * x).exp() + truth[2])
        .collect();
    group.bench_function("exp_curve_64", |b| {
        b.iter(|| {
            let residuals = |p: &[f64]| {
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| p[0] * (-p[1] * x).exp() + p[2] - y)
                    .collect::<Vec<_>>()
            };
            let _ = least_squares(residuals, &[1.0, 1.0, 1.0], LeastSquaresOptions::default());
        });
    });

    let xs: Vec<f64> = (0..128).map(|i| i as f64 * 0.05).collect();
    let truth = [2.0_f64, 0.7, 0.25, -0.03];
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| truth[0] * (-truth[1] * x).exp() + truth[2] + truth[3] * x)
        .collect();
    group.bench_function("exp_linear_curve_128", |b| {
        b.iter(|| {
            let residuals = |p: &[f64]| {
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| p[0] * (-p[1] * x).exp() + p[2] + p[3] * x - y)
                    .collect::<Vec<_>>()
            };
            let _ = least_squares(
                residuals,
                &[1.0, 0.5, 0.0, 0.0],
                LeastSquaresOptions::default(),
            );
        });
    });
    group.finish();
}

/// Hungarian assignment — head-to-head vs scipy.optimize.linear_sum_assignment.
fn bench_assignment(c: &mut Criterion) {
    use criterion::BenchmarkId;
    let mut group = c.benchmark_group("linear_sum_assignment");
    for &n in &[500usize, 1000] {
        let cost: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        // LCG-based continuous values (few ties), matching scipy's uniform.
                        let s = (i
                            .wrapping_mul(1103515245)
                            .wrapping_add(j)
                            .wrapping_mul(12345)
                            ^ (i.wrapping_mul(2654435761) >> 7))
                            as u64;
                        (s as f64 / u64::MAX as f64) * (i % 9 + 1) as f64
                    })
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("dense", n), |b| {
            b.iter(|| linear_sum_assignment(std::hint::black_box(&cost)).expect("lsa"))
        });
    }
    group.finish();
}

fn bench_differential_evolution(c: &mut Criterion) {
    // Global optimizer over a user objective evaluated INLINE in Rust (vs scipy's
    // Python callback per nfev). Rosenbrock-5d, matched config to the scipy run
    // (maxiter=100, popsize=15, tol=1e-8, seed=1). scipy ~271 ms (nfev=7689).
    let rosen = |x: &[f64]| -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() - 1 {
            s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        s
    };
    let bounds = vec![(-5.0, 5.0); 5];
    let mut group = c.benchmark_group("differential_evolution");
    group.sample_size(20);
    group.bench_function("rosen_5d", |b| {
        b.iter(|| {
            let opts = DifferentialEvolutionOptions {
                maxiter: 100,
                popsize: 15,
                tol: 1e-8,
                seed: Some(1),
                ..Default::default()
            };
            differential_evolution(rosen, &bounds, opts).expect("DE")
        })
    });
    group.finish();
}

fn select_three_fixed(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = [0; 3];
    let mut len = 0;
    let mut attempts = 0;
    while len < indices.len() {
        let idx = rng.random_range(0..n);
        if idx != exclude && !indices[..len].contains(&idx) {
            indices[len] = idx;
            len += 1;
        }
        attempts += 1;
        if attempts > 1000 {
            for k in 0..n {
                if len >= indices.len() {
                    break;
                }
                if !indices[..len].contains(&k) {
                    indices[len] = k;
                    len += 1;
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

fn select_three_allocating(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut indices = Vec::with_capacity(3);
    let mut attempts = 0;
    while indices.len() < 3 {
        let idx = rng.random_range(0..n);
        if idx != exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
        attempts += 1;
        if attempts > 1000 {
            for k in 0..n {
                if indices.len() >= 3 {
                    break;
                }
                if !indices.contains(&k) {
                    indices.push(k);
                }
            }
            break;
        }
    }
    (indices[0], indices[1], indices[2])
}

fn select_three_checksum(
    mut rng: rand::rngs::StdRng,
    select: fn(&mut rand::rngs::StdRng, usize, usize) -> (usize, usize, usize),
) -> usize {
    let mut checksum = 0usize;
    for trial in 0..7_500 {
        let (r0, r1, r2) = select(&mut rng, 75, trial % 75);
        checksum = checksum
            .wrapping_mul(31)
            .wrapping_add(r0)
            .wrapping_add(r1.wrapping_mul(3))
            .wrapping_add(r2.wrapping_mul(7));
    }
    checksum
}

fn bench_select_three_ab(c: &mut Criterion) {
    let mut fixed_rng = rand::rngs::StdRng::seed_from_u64(1);
    let mut allocating_rng = rand::rngs::StdRng::seed_from_u64(1);
    for trial in 0..7_500 {
        assert_eq!(
            select_three_fixed(&mut fixed_rng, 75, trial % 75),
            select_three_allocating(&mut allocating_rng, 75, trial % 75),
        );
    }

    let mut group = c.benchmark_group("select_three_ab");
    group.bench_function("fixed_array/7500", |b| {
        b.iter(|| {
            black_box(select_three_checksum(
                rand::rngs::StdRng::seed_from_u64(1),
                select_three_fixed,
            ))
        });
    });
    group.bench_function("vec_capacity_3/7500", |b| {
        b.iter(|| {
            black_box(select_three_checksum(
                rand::rngs::StdRng::seed_from_u64(1),
                select_three_allocating,
            ))
        });
    });
    group.finish();
}

fn reference_clone_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let f0 = f(x);
    let mut grad = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += eps;
        grad.push((f(&xp) - f0) / eps);
    }
    grad
}

fn reference_clone_jacobian<F>(f: F, x: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let f0 = f(x);
    let mut jac = vec![vec![0.0; x.len()]; f0.len()];
    for j in 0..x.len() {
        let mut xp = x.to_vec();
        xp[j] += eps;
        let fp = f(&xp);
        for i in 0..f0.len() {
            jac[i][j] = (fp[i] - f0[i]) / eps;
        }
    }
    jac
}

fn finite_difference_scalar(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(i, &value)| {
            let weight = (i % 7 + 1) as f64;
            weight * value * value + 0.125 * value
        })
        .sum()
}

fn finite_difference_vector(x: &[f64]) -> Vec<f64> {
    let mut sums = [0.0; 4];
    for (i, &value) in x.iter().enumerate() {
        sums[i & 3] += value * ((i % 11 + 1) as f64);
    }
    sums.to_vec()
}

fn bench_finite_difference_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("finite_difference_helpers");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(1));

    for &dim in &[256usize, 512] {
        let x: Vec<f64> = (0..dim)
            .map(|i| ((i % 29) as f64 - 14.0) * 0.03125)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("gradient_clone_reference", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(reference_clone_gradient(
                        finite_difference_scalar,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gradient_scratch_reuse", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(numerical_gradient(
                        finite_difference_scalar,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
    }

    for &dim in &[128usize, 256] {
        let x: Vec<f64> = (0..dim)
            .map(|i| ((i % 31) as f64 - 15.0) * 0.015625)
            .collect();
        group.bench_with_input(
            BenchmarkId::new("jacobian_clone_reference", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(reference_clone_jacobian(
                        finite_difference_vector,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("jacobian_scratch_reuse", dim),
            &x,
            |b, x| {
                b.iter(|| {
                    black_box(numerical_jacobian(
                        finite_difference_vector,
                        black_box(x),
                        1.0e-6,
                    ))
                });
            },
        );
    }
    group.finish();
}

// ── lm_root J^T J + J^T F build: strided-original vs symmetric-row-outer A/B ──
//
// Same-process A/B isolating the per-iteration normal-equation build inside
// `lm_root` (root.rs). The `original` arm is the shipped-before naive triple-nest
// (row-reduction innermost = n² cache-missing strided passes over `jac`); the
// `candidate` arm is the current lib code (single row-outer pass, J^T J symmetry).
// Both produce bit-identical output (asserted), so this measures pure cache/flop win.

fn jtj_jtf_original(jac: &[Vec<f64>], fx: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.len();
    let mut jtj = vec![vec![0.0; n]; n];
    let mut jtf = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            for row in jac.iter().take(n) {
                jtj[i][j] += row[i] * row[j];
            }
        }
        for (row, fx_value) in jac.iter().zip(fx.iter()).take(n) {
            jtf[i] += row[i] * *fx_value;
        }
    }
    (jtj, jtf)
}

fn jtj_jtf_candidate(jac: &[Vec<f64>], fx: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jac.len();
    let mut jtj = vec![vec![0.0; n]; n];
    let mut jtf = vec![0.0; n];
    for (row, &fx_value) in jac.iter().zip(fx.iter()) {
        for i in 0..n {
            let ri = row[i];
            for j in i..n {
                let v = ri * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
            jtf[i] += ri * fx_value;
        }
    }
    (jtj, jtf)
}

fn bench_lm_jtj_build_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("lm_root_jtj_build_ab");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[64usize, 128, 256, 512] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xB1AC_7 ^ n as u64);
        let jac: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
            .collect();
        let fx: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();

        // Byte-identical proof: candidate output matches original bit-for-bit.
        let (o_jtj, o_jtf) = jtj_jtf_original(&jac, &fx);
        let (c_jtj, c_jtf) = jtj_jtf_candidate(&jac, &fx);
        for (ro, rc) in o_jtj.iter().zip(c_jtj.iter()) {
            for (a, b) in ro.iter().zip(rc.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "jtj mismatch at n={n}");
            }
        }
        for (a, b) in o_jtf.iter().zip(c_jtf.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "jtf mismatch at n={n}");
        }

        group.bench_function(BenchmarkId::new("original_strided", n), |b| {
            b.iter(|| black_box(jtj_jtf_original(black_box(&jac), black_box(&fx))));
        });
        group.bench_function(BenchmarkId::new("candidate_symmetric", n), |b| {
            b.iter(|| black_box(jtj_jtf_candidate(black_box(&jac), black_box(&fx))));
        });
    }
    group.finish();
}

/// Same-binary A/B for folding the `+λI` shift directly into the exact trust-region
/// subproblem solve (dropping the redundant `shifted_matrix` copy per λ trial). Rosenbrock
/// from the origin drives many constrained subproblem steps (each a bracketing + bisection
/// sweep of `(H+λI)` solves). Byte-identical — final `x` and `fun` asserted bit-equal.
fn bench_trust_exact_fold_shift_ab(c: &mut Criterion) {
    use fsci_opt::{TRUST_EXACT_FOLD_SHIFT_DISABLE, trust_exact};
    use std::sync::atomic::Ordering;

    const PROFILE_PARENT: &str = "FSCI_TRUST_EXACT_PROFILE";
    const PROFILE_CHILD: &str = "FSCI_TRUST_EXACT_PROFILE_CHILD";
    if std::env::var_os(PROFILE_CHILD).is_some() {
        let x0 = vec![0.0; 20];
        TRUST_EXACT_FOLD_SHIFT_DISABLE.store(false, Ordering::Relaxed);
        for _ in 0..4 {
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
        }
        let started = std::time::Instant::now();
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..160 {
            let result = trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            )
            .expect("profiled trust-exact solve");
            for value in result.x {
                digest ^= value.to_bits();
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(result.fun);
        }
        eprintln!(
            "FSCI_TRUST_EXACT_PROFILE solves=160 mean_ms={:.6} digest={digest:#018x}",
            started.elapsed().as_secs_f64() * 1000.0 / 160.0
        );
        std::process::exit(0);
    }
    if std::env::var_os(PROFILE_PARENT).is_some() {
        let exe = std::env::current_exe().expect("current release benchmark binary");
        let perf_path = format!(
            "/dev/shm/fsci-trust-exact-{}-{}.perf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        );
        let profile_status = Command::new("perf")
            .args([
                "record", "-e", "cycles:u", "-F", "997", "-o", &perf_path, "--",
            ])
            .arg(&exe)
            .args(["trust_exact_fold_shift_ab", "--noplot"])
            .env(PROFILE_CHILD, "1")
            .status()
            .expect("spawn trust-exact perf child");
        assert!(
            profile_status.success(),
            "trust-exact perf child failed: {profile_status}"
        );
        let report = Command::new("perf")
            .args([
                "report",
                "--stdio",
                "--no-children",
                "--percent-limit",
                "0.5",
                "--sort",
                "symbol",
                "-i",
                &perf_path,
            ])
            .output()
            .expect("run trust-exact perf report");
        assert!(
            report.status.success(),
            "trust-exact perf report failed: {}",
            String::from_utf8_lossy(&report.stderr)
        );
        eprintln!(
            "FSCI_TRUST_EXACT_REPORT_BEGIN\n{}FSCI_TRUST_EXACT_REPORT_END",
            String::from_utf8(report.stdout).expect("utf8 trust-exact perf report")
        );
    }

    let mut group = c.benchmark_group("trust_exact_fold_shift_ab");
    group.sample_size(20);
    for &dim in &[5usize, 10, 20] {
        let x0: Vec<f64> = vec![0.0; dim];
        TRUST_EXACT_FOLD_SHIFT_DISABLE.store(false, Ordering::Relaxed);
        let folded = trust_exact(&rosenbrock, &x0, opts(OptimizeMethod::TrustExact)).unwrap();
        TRUST_EXACT_FOLD_SHIFT_DISABLE.store(true, Ordering::Relaxed);
        let orig = trust_exact(&rosenbrock, &x0, opts(OptimizeMethod::TrustExact)).unwrap();
        assert!(
            folded.x.len() == orig.x.len()
                && folded
                    .x
                    .iter()
                    .zip(&orig.x)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && folded.fun.unwrap().to_bits() == orig.fun.unwrap().to_bits(),
            "trust_exact fold-shift not byte-identical (dim={dim})"
        );
        group.bench_with_input(BenchmarkId::new("current_folded", dim), &x0, |b, x0| {
            b.iter(|| {
                TRUST_EXACT_FOLD_SHIFT_DISABLE.store(false, Ordering::Relaxed);
                black_box(trust_exact(
                    &rosenbrock,
                    x0,
                    opts(OptimizeMethod::TrustExact),
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_shifted_copy", dim), &x0, |b, x0| {
            b.iter(|| {
                TRUST_EXACT_FOLD_SHIFT_DISABLE.store(true, Ordering::Relaxed);
                black_box(trust_exact(
                    &rosenbrock,
                    x0,
                    opts(OptimizeMethod::TrustExact),
                ))
            });
        });
    }
    TRUST_EXACT_FOLD_SHIFT_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn sample_stats(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / sorted.len() as f64;
    let percentile = |numerator: usize| {
        let rank = (numerator * sorted.len()).div_ceil(100).saturating_sub(1);
        sorted[rank]
    };
    (
        percentile(50),
        percentile(95),
        percentile(99),
        variance.sqrt() / mean * 100.0,
    )
}

/// Same-binary A/B for storing the hot trust-exact augmented system in one contiguous region.
/// The opt-in probe runs strict A/B/A triplets so the two original measurements form a null
/// control around every candidate sample.
fn bench_trust_exact_flat_augmented_ab(c: &mut Criterion) {
    use fsci_opt::{TRUST_EXACT_FLAT_AUGMENTED_DISABLE, trust_exact};
    use std::sync::atomic::Ordering;

    let x0 = vec![0.0; 20];
    TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(false, Ordering::Relaxed);
    let flat = trust_exact(&rosenbrock, &x0, opts(OptimizeMethod::TrustExact)).unwrap();
    TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(true, Ordering::Relaxed);
    let nested = trust_exact(&rosenbrock, &x0, opts(OptimizeMethod::TrustExact)).unwrap();
    assert_eq!(
        flat, nested,
        "flat augmented solve changed trust-exact result"
    );
    for (index, (&flat, &nested)) in flat.x.iter().zip(nested.x.iter()).enumerate() {
        assert_eq!(
            flat.to_bits(),
            nested.to_bits(),
            "flat augmented solve changed x[{index}] bits"
        );
    }
    assert_eq!(
        flat.fun.map(f64::to_bits),
        nested.fun.map(f64::to_bits),
        "flat augmented solve changed objective bits"
    );

    if std::env::var_os("FSCI_TRUST_EXACT_FLAT_PROBE").is_some() {
        for _ in 0..3 {
            TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(true, Ordering::Relaxed);
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
            TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(false, Ordering::Relaxed);
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
        }

        let mut original_before_ms = Vec::with_capacity(13);
        let mut flat_ms = Vec::with_capacity(13);
        let mut original_after_ms = Vec::with_capacity(13);
        for _ in 0..13 {
            TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(true, Ordering::Relaxed);
            let started = std::time::Instant::now();
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
            original_before_ms.push(started.elapsed().as_secs_f64() * 1000.0);

            TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(false, Ordering::Relaxed);
            let started = std::time::Instant::now();
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
            flat_ms.push(started.elapsed().as_secs_f64() * 1000.0);

            TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(true, Ordering::Relaxed);
            let started = std::time::Instant::now();
            let _ = black_box(trust_exact(
                &rosenbrock,
                black_box(&x0),
                opts(OptimizeMethod::TrustExact),
            ));
            original_after_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let candidate_ratios = original_before_ms
            .iter()
            .zip(original_after_ms.iter())
            .zip(flat_ms.iter())
            .map(|((&before, &after), &flat)| ((before + after) * 0.5) / flat)
            .collect::<Vec<_>>();
        let null_ratios = original_before_ms
            .iter()
            .zip(original_after_ms.iter())
            .map(|(&before, &after)| before / after)
            .collect::<Vec<_>>();
        let (original_p50, original_p95, original_p99, original_cv) =
            sample_stats(&original_before_ms);
        let (flat_p50, flat_p95, flat_p99, flat_cv) = sample_stats(&flat_ms);
        let (candidate_ratio, _, _, _) = sample_stats(&candidate_ratios);
        let (null_ratio, _, _, null_cv) = sample_stats(&null_ratios);
        let null_low = null_ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let null_high = null_ratios
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "FSCI_TRUST_EXACT_FLAT_INTERLEAVED dim=20 rounds=13 \
             original_ms[p50={original_p50:.6},p95={original_p95:.6},p99={original_p99:.6},cv={original_cv:.3}%] \
             flat_ms[p50={flat_p50:.6},p95={flat_p95:.6},p99={flat_p99:.6},cv={flat_cv:.3}%] \
             candidate_median={candidate_ratio:.6}x \
             null_median={null_ratio:.6}x null_range=[{null_low:.6},{null_high:.6}] null_cv={null_cv:.3}%"
        );
    }

    let mut group = c.benchmark_group("trust_exact_flat_augmented_ab");
    group.sample_size(20);
    for &dim in &[10usize, 20] {
        let x0 = vec![0.0; dim];
        group.bench_with_input(BenchmarkId::new("flat", dim), &x0, |b, x0| {
            b.iter(|| {
                TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(false, Ordering::Relaxed);
                black_box(trust_exact(
                    &rosenbrock,
                    x0,
                    opts(OptimizeMethod::TrustExact),
                ))
            });
        });
        group.bench_with_input(BenchmarkId::new("nested", dim), &x0, |b, x0| {
            b.iter(|| {
                TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(true, Ordering::Relaxed);
                black_box(trust_exact(
                    &rosenbrock,
                    x0,
                    opts(OptimizeMethod::TrustExact),
                ))
            });
        });
    }
    TRUST_EXACT_FLAT_AUGMENTED_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(
    benches,
    bench_trust_exact_flat_augmented_ab,
    bench_trust_exact_fold_shift_ab,
    bench_lm_jtj_build_ab,
    bench_select_three_ab,
    bench_finite_difference_helpers,
    bench_assignment,
    bench_differential_evolution,
    bench_bfgs,
    bench_lbfgsb,
    bench_cg,
    bench_powell,
    bench_brentq,
    bench_brenth,
    bench_bisect,
    bench_ridder,
    bench_least_squares,
);

fn main() -> ExitCode {
    if let Err(error) = report_bench_elf_sha256() {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }

    if std::env::args().any(|argument| argument == "--frontier-trust-exact-cholesky") {
        return match run_trust_exact_cholesky_frontier() {
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
    ExitCode::SUCCESS
}
