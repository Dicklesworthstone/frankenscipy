//! P2C-001-H: Performance profiling and behavior-isomorphism evidence for IVP.
//!
//! Produces structured JSON artifact at:
//!   fixtures/artifacts/FSCI-P2C-001/perf/perf_profile_report.json
//!
//! Focuses on validation + initial-step hotspots and includes:
//! - p50/p95/p99 timing
//! - top-3 hotspot ranking
//! - optimization attempt log fields required by the bead
//! - behavior-isomorphism checks for the optimized path

use fsci_integrate::{
    InitialStepRequest, OdeSolver, OdeSolverState, RK45_TABLEAU, RkSolver, RkSolverConfig,
    ToleranceValue, validate_first_step, validate_max_step, validate_tol,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::time::Instant;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 25;
const HOTSPOT_SELECT_INITIAL_STEP: &str = "select_initial_step";

#[derive(Serialize)]
struct PerfReport {
    generated_at: String,
    packet_id: String,
    optimization_name: String,
    benchmark_rows: Vec<BenchmarkRow>,
    hotspot_ranking: Vec<HotspotEntry>,
    optimization_attempts: Vec<OptimizationAttempt>,
    failed_optimization_attempts: Vec<FailedOptimizationAttempt>,
    memory_delta: MemoryDelta,
    isomorphism_check: IsomorphismCheck,
    methodology: Vec<String>,
}

#[derive(Serialize)]
struct BenchmarkRow {
    hotspot_function: String,
    scenario: String,
    before: TimingStats,
    after: TimingStats,
    p95_delta_pct: f64,
}

#[derive(Serialize, Clone)]
struct TimingStats {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    min_ns: u128,
    max_ns: u128,
    mean_ns: u128,
}

#[derive(Serialize)]
struct HotspotEntry {
    rank: usize,
    hotspot_function: String,
    scenario: String,
    after_p95_ns: u128,
    share_of_total_p95: f64,
}

#[derive(Serialize)]
struct OptimizationAttempt {
    optimization_name: String,
    hotspot_function: String,
    before_p95_ns: u128,
    after_p95_ns: u128,
    delta_pct: f64,
}

#[derive(Serialize)]
struct FailedOptimizationAttempt {
    optimization_name: String,
    hotspot_function: String,
    reason: String,
    rollback_evidence: String,
}

#[derive(Serialize)]
struct MemoryDelta {
    method: String,
    before_estimated_alloc_events: usize,
    after_estimated_alloc_events: usize,
    alloc_count_delta: i64,
    note: String,
}

#[derive(Serialize)]
struct IsomorphismCheck {
    all_cases_pass: bool,
    details: Vec<IsomorphismDetail>,
}

#[derive(Serialize)]
struct IsomorphismDetail {
    case_name: String,
    legacy_h_abs: f64,
    optimized_h_abs: f64,
    abs_diff: f64,
    passes: bool,
}

fn quantile_index(n: usize, q: f64) -> usize {
    if n == 0 {
        return 0;
    }
    let raw = (n as f64 * q).ceil() as usize;
    raw.saturating_sub(1).min(n - 1)
}

fn time_operation<F: FnMut()>(mut f: F) -> TimingStats {
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let mut timings = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_nanos());
    }
    timings.sort_unstable();

    let n = timings.len();
    let p50 = timings[quantile_index(n, 0.50)];
    let p95 = timings[quantile_index(n, 0.95)];
    let p99 = timings[quantile_index(n, 0.99)];
    let min = timings[0];
    let max = timings[n - 1];
    let mean = timings.iter().sum::<u128>() / n as u128;

    TimingStats {
        p50_ns: p50,
        p95_ns: p95,
        p99_ns: p99,
        min_ns: min,
        max_ns: max,
        mean_ns: mean,
    }
}

fn p95_delta_pct(before: &TimingStats, after: &TimingStats) -> f64 {
    if before.p95_ns == 0 {
        return 0.0;
    }
    (after.p95_ns as f64 - before.p95_ns as f64) * 100.0 / before.p95_ns as f64
}

fn exponential_decay_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![-0.5 * y[0]]
}

fn lorenz_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2],
    ]
}

fn legacy_rms_norm(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f64).sqrt()
}

fn legacy_select_initial_step<F>(
    fun: &mut F,
    request: &InitialStepRequest<'_>,
) -> Result<f64, fsci_integrate::IntegrateValidationError>
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let n = request.y0.len();
    if n == 0 {
        return Ok(f64::INFINITY);
    }

    if request.mode == RuntimeMode::Hardened {
        if !request.y0.iter().all(|v| v.is_finite()) {
            return Err(
                fsci_integrate::IntegrateValidationError::NotYetImplemented {
                    function: "select_initial_step: non-finite y0 in Hardened mode",
                },
            );
        }
        if !request.f0.iter().all(|v| v.is_finite()) {
            return Err(
                fsci_integrate::IntegrateValidationError::NotYetImplemented {
                    function: "select_initial_step: non-finite f0 in Hardened mode",
                },
            );
        }
    }

    let interval_length = (request.t_bound - request.t0).abs();
    if interval_length == 0.0 {
        return Ok(0.0);
    }

    let scale: Vec<f64> = match &request.atol {
        ToleranceValue::Scalar(atol) => request
            .y0
            .iter()
            .map(|y| atol + y.abs() * request.rtol)
            .collect(),
        ToleranceValue::Vector(atol_vec) => request
            .y0
            .iter()
            .zip(atol_vec.iter())
            .map(|(y, a)| a + y.abs() * request.rtol)
            .collect(),
    };

    let scaled_y0: Vec<f64> = request
        .y0
        .iter()
        .zip(scale.iter())
        .map(|(y, s)| y / s)
        .collect();
    let scaled_f0: Vec<f64> = request
        .f0
        .iter()
        .zip(scale.iter())
        .map(|(f, s)| f / s)
        .collect();
    let d0 = legacy_rms_norm(&scaled_y0);
    let d1 = legacy_rms_norm(&scaled_f0);

    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    let h0 = h0.min(interval_length);

    let y1: Vec<f64> = request
        .y0
        .iter()
        .zip(request.f0.iter())
        .map(|(y, f)| y + h0 * request.direction * f)
        .collect();

    let f1 = fun(request.t0 + h0 * request.direction, &y1);

    let diff_scaled: Vec<f64> = f1
        .iter()
        .zip(request.f0.iter())
        .zip(scale.iter())
        .map(|((f1v, f0v), s)| (f1v - f0v) / s)
        .collect();
    let d2 = legacy_rms_norm(&diff_scaled) / h0;

    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (1e-6_f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (request.order + 1.0))
    };

    Ok((100.0 * h0)
        .min(h1)
        .min(interval_length)
        .min(request.max_step))
}

fn build_hotspot_ranking(rows: &[BenchmarkRow]) -> Vec<HotspotEntry> {
    let mut indexed: Vec<(usize, &BenchmarkRow)> = rows.iter().enumerate().collect();
    indexed.sort_by_key(|(_, row)| std::cmp::Reverse(row.after.p95_ns));

    let total_after_p95: u128 = rows.iter().map(|row| row.after.p95_ns).sum();
    indexed
        .iter()
        .enumerate()
        .map(|(rank_idx, (_, row))| HotspotEntry {
            rank: rank_idx + 1,
            hotspot_function: row.hotspot_function.clone(),
            scenario: row.scenario.clone(),
            after_p95_ns: row.after.p95_ns,
            share_of_total_p95: if total_after_p95 == 0 {
                0.0
            } else {
                row.after.p95_ns as f64 / total_after_p95 as f64
            },
        })
        .collect()
}

#[test]
fn perf_p2c001_ivp_profile_and_isomorphism() {
    let mut rows = Vec::new();

    let validate_tol_scalar = time_operation(|| {
        let _ = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Scalar(1e-9),
            128,
            RuntimeMode::Strict,
        )
        .expect("scalar tolerance validation should succeed");
    });
    rows.push(BenchmarkRow {
        hotspot_function: "validate_tol".to_string(),
        scenario: "scalar_n128".to_string(),
        before: validate_tol_scalar.clone(),
        after: validate_tol_scalar.clone(),
        p95_delta_pct: 0.0,
    });

    let atol_vec = vec![1e-9; 256];
    let validate_tol_vector = time_operation(|| {
        let _ = validate_tol(
            ToleranceValue::Scalar(1e-6),
            ToleranceValue::Vector(atol_vec.clone()),
            256,
            RuntimeMode::Strict,
        )
        .expect("vector tolerance validation should succeed");
    });
    rows.push(BenchmarkRow {
        hotspot_function: "validate_tol".to_string(),
        scenario: "vector_n256".to_string(),
        before: validate_tol_vector.clone(),
        after: validate_tol_vector.clone(),
        p95_delta_pct: 0.0,
    });

    let validate_first_step_stats = time_operation(|| {
        let _ = validate_first_step(0.01, 0.0, 10.0).expect("first_step validation should pass");
    });
    rows.push(BenchmarkRow {
        hotspot_function: "validate_first_step".to_string(),
        scenario: "positive_within_bounds".to_string(),
        before: validate_first_step_stats.clone(),
        after: validate_first_step_stats.clone(),
        p95_delta_pct: 0.0,
    });

    let validate_max_step_stats = time_operation(|| {
        let _ = validate_max_step(0.5).expect("max_step validation should pass");
    });
    rows.push(BenchmarkRow {
        hotspot_function: "validate_max_step".to_string(),
        scenario: "finite_positive".to_string(),
        before: validate_max_step_stats.clone(),
        after: validate_max_step_stats.clone(),
        p95_delta_pct: 0.0,
    });

    let y0 = vec![1.0, 1.0, 1.0];
    let rhs_for_f0 = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
    let f0 = rhs_for_f0(0.0, &y0);
    let request = InitialStepRequest {
        t0: 0.0,
        y0: &y0,
        t_bound: 1.0,
        max_step: f64::INFINITY,
        f0: &f0,
        direction: 1.0,
        order: 4.0,
        rtol: 1e-6,
        atol: ToleranceValue::Vector(vec![1e-9, 1e-9, 1e-9]),
        mode: RuntimeMode::Strict,
    };

    let before_select = time_operation(|| {
        let mut rhs = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let _ = legacy_select_initial_step(&mut rhs, &request)
            .expect("legacy initial step selection should succeed");
    });
    let after_select = time_operation(|| {
        let mut rhs = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let _ = fsci_integrate::select_initial_step(&mut rhs, &request)
            .expect("optimized initial step selection should succeed");
    });
    rows.push(BenchmarkRow {
        hotspot_function: HOTSPOT_SELECT_INITIAL_STEP.to_string(),
        scenario: "lorenz_vector_atol".to_string(),
        p95_delta_pct: p95_delta_pct(&before_select, &after_select),
        before: before_select.clone(),
        after: after_select.clone(),
    });

    let rk_step_loop = time_operation(|| {
        let y0_local = [1.0, 1.0, 1.0];
        let mut rhs = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &y0_local,
            t_bound: 0.2,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-9),
            max_step: f64::INFINITY,
            first_step: Some(1e-3),
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver =
            RkSolver::new(&mut rhs, config).expect("solver construction should succeed");
        for _ in 0..20 {
            if solver.state() != OdeSolverState::Running {
                break;
            }
            let _ = solver
                .step_with(&mut rhs)
                .expect("RK step loop should remain stable");
        }
    });
    rows.push(BenchmarkRow {
        hotspot_function: "RkSolver::step_with".to_string(),
        scenario: "lorenz_20_steps".to_string(),
        before: rk_step_loop.clone(),
        after: rk_step_loop.clone(),
        p95_delta_pct: 0.0,
    });

    let hotspot_ranking = build_hotspot_ranking(&rows);
    let top3 = hotspot_ranking.iter().take(3).collect::<Vec<_>>();
    assert_eq!(top3.len(), 3, "expected at least three ranked hotspots");

    let optimization_attempts = vec![OptimizationAttempt {
        optimization_name: "remove_intermediate_rms_vectors_in_select_initial_step".to_string(),
        hotspot_function: HOTSPOT_SELECT_INITIAL_STEP.to_string(),
        before_p95_ns: before_select.p95_ns,
        after_p95_ns: after_select.p95_ns,
        delta_pct: p95_delta_pct(&before_select, &after_select),
    }];

    let failed_optimization_attempts = Vec::new();

    let memory_delta = MemoryDelta {
        method: "algorithmic allocation-count estimate".to_string(),
        // Legacy path allocated 5 temporary Vecs in select_initial_step:
        // scale, scaled_y0, scaled_f0, y1, diff_scaled.
        before_estimated_alloc_events: 5,
        // Optimized path allocates only scale and y1 as temporaries.
        after_estimated_alloc_events: 2,
        alloc_count_delta: -3,
        note: "Estimate applies to select_initial_step temporaries only; allocator-level instrumentation is not used in this harness.".to_string(),
    };

    let mut iso_details = Vec::new();

    {
        let y0_case = vec![1.0];
        let rhs_for_f0_case = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let f0_case = rhs_for_f0_case(0.0, &y0_case);
        let req = InitialStepRequest {
            t0: 0.0,
            y0: &y0_case,
            t_bound: 10.0,
            max_step: f64::INFINITY,
            f0: &f0_case,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let mut rhs_legacy = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let mut rhs_optimized = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let legacy = legacy_select_initial_step(&mut rhs_legacy, &req)
            .expect("legacy select_initial_step should succeed");
        let optimized = fsci_integrate::select_initial_step(&mut rhs_optimized, &req)
            .expect("optimized select_initial_step should succeed");
        let abs_diff = (legacy - optimized).abs();
        iso_details.push(IsomorphismDetail {
            case_name: "scalar_tol_exponential".to_string(),
            legacy_h_abs: legacy,
            optimized_h_abs: optimized,
            abs_diff,
            passes: abs_diff <= 1e-15,
        });
    }

    {
        let y0_case = vec![1.0, 1.0, 1.0];
        let rhs_for_f0_case = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let f0_case = rhs_for_f0_case(0.0, &y0_case);
        let req = InitialStepRequest {
            t0: 0.0,
            y0: &y0_case,
            t_bound: 1.0,
            max_step: f64::INFINITY,
            f0: &f0_case,
            direction: 1.0,
            order: 4.0,
            rtol: 1e-6,
            atol: ToleranceValue::Vector(vec![1e-9, 1e-9, 1e-9]),
            mode: RuntimeMode::Strict,
        };
        let mut rhs_legacy = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let mut rhs_optimized = lorenz_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let legacy = legacy_select_initial_step(&mut rhs_legacy, &req)
            .expect("legacy select_initial_step should succeed");
        let optimized = fsci_integrate::select_initial_step(&mut rhs_optimized, &req)
            .expect("optimized select_initial_step should succeed");
        let abs_diff = (legacy - optimized).abs();
        iso_details.push(IsomorphismDetail {
            case_name: "vector_tol_lorenz".to_string(),
            legacy_h_abs: legacy,
            optimized_h_abs: optimized,
            abs_diff,
            passes: abs_diff <= 1e-15,
        });
    }

    {
        let y0_case = vec![1.0];
        let rhs_for_f0_case = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let f0_case = rhs_for_f0_case(1.0, &y0_case);
        let req = InitialStepRequest {
            t0: 1.0,
            y0: &y0_case,
            t_bound: 0.0,
            max_step: 0.5,
            f0: &f0_case,
            direction: -1.0,
            order: 4.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            mode: RuntimeMode::Strict,
        };
        let mut rhs_legacy = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let mut rhs_optimized = exponential_decay_rhs as fn(f64, &[f64]) -> Vec<f64>;
        let legacy = legacy_select_initial_step(&mut rhs_legacy, &req)
            .expect("legacy select_initial_step should succeed");
        let optimized = fsci_integrate::select_initial_step(&mut rhs_optimized, &req)
            .expect("optimized select_initial_step should succeed");
        let abs_diff = (legacy - optimized).abs();
        iso_details.push(IsomorphismDetail {
            case_name: "backward_integration_scalar_tol".to_string(),
            legacy_h_abs: legacy,
            optimized_h_abs: optimized,
            abs_diff,
            passes: abs_diff <= 1e-15,
        });
    }

    let all_iso_pass = iso_details.iter().all(|detail| detail.passes);
    assert!(
        all_iso_pass,
        "optimized select_initial_step diverged from legacy behavior"
    );

    let report = PerfReport {
        generated_at: chrono_lite_now(),
        packet_id: "FSCI-P2C-001".to_string(),
        optimization_name: "remove_intermediate_rms_vectors_in_select_initial_step".to_string(),
        benchmark_rows: rows,
        hotspot_ranking,
        optimization_attempts,
        failed_optimization_attempts,
        memory_delta,
        isomorphism_check: IsomorphismCheck {
            all_cases_pass: all_iso_pass,
            details: iso_details,
        },
        methodology: vec![
            format!("warmup_iters={WARMUP_ITERS}"),
            format!("bench_iters={BENCH_ITERS}"),
            "before_select_initial_step=legacy algorithm with temporary scaled vectors".to_string(),
            "after_select_initial_step=single-pass RMS accumulation over scale/f0 differences"
                .to_string(),
            "top-3 hotspots ranked by after.p95_ns share of total".to_string(),
        ],
    };

    let artifact_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-001/perf");
    std::fs::create_dir_all(&artifact_dir).expect("perf artifact directory should be creatable");

    let pretty_json = serde_json::to_string_pretty(&report).expect("perf report should serialize");
    std::fs::write(artifact_dir.join("perf_profile_report.json"), pretty_json)
        .expect("perf profile report should be written");

    let compact_json =
        serde_json::to_string(&report).expect("compact perf report should serialize");
    println!("P2C001_PERF_REPORT_JSON={compact_json}");
}

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs();
    format!("unix:{secs}")
}
