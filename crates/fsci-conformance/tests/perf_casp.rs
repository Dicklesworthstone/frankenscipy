//! P2C-008-H: Performance profiling tests for CASP runtime operations.
//!
//! Produces structured JSON artifact at:
//!   fixtures/artifacts/FSCI-P2C-008/perf/perf_profile_report.json
//!
//! Covers PolicyController, SolverPortfolio, ConformalCalibrator, DecisionSignals.

use fsci_runtime::{
    ConformalCalibrator, DecisionSignals, MatrixConditionState, PolicyController, RuntimeMode,
    SolverAction, SolverPortfolio,
};
use serde::Serialize;
use std::time::Instant;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 100;

// ── Data structures ────────────────────────────────────────────────────

#[derive(Serialize)]
struct PerfReport {
    generated_at: String,
    operation_benchmarks: Vec<OperationBenchmark>,
    hotspot_ranking: Vec<HotspotEntry>,
    memory_notes: Vec<MemoryNote>,
    isomorphism_check: IsomorphismCheck,
}

#[derive(Serialize)]
struct OperationBenchmark {
    operation: String,
    input_desc: String,
    iterations: usize,
    median_ns: u128,
    p95_ns: u128,
    min_ns: u128,
    max_ns: u128,
    mean_ns: u128,
}

#[derive(Serialize)]
struct HotspotEntry {
    rank: usize,
    operation: String,
    input_desc: String,
    median_ns: u128,
    fraction_of_total: f64,
}

#[derive(Serialize)]
struct MemoryNote {
    operation: String,
    notes: String,
}

#[derive(Serialize)]
struct IsomorphismCheck {
    all_operations_pass: bool,
    details: Vec<IsomorphismDetail>,
}

#[derive(Serialize)]
struct IsomorphismDetail {
    operation: String,
    passes: bool,
    note: String,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn time_operation<F: FnMut()>(mut f: F) -> Vec<u128> {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let mut timings = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_nanos());
    }
    timings.sort();
    timings
}

fn compute_stats(timings: &[u128]) -> (u128, u128, u128, u128, u128) {
    let n = timings.len();
    let median = timings[n / 2];
    let p95 = timings[(n as f64 * 0.95) as usize];
    let min_v = timings[0];
    let max_v = timings[n - 1];
    let mean = timings.iter().sum::<u128>() / n as u128;
    (median, p95, min_v, max_v, mean)
}

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}

fn state_to_rcond(state: &MatrixConditionState) -> f64 {
    match state {
        MatrixConditionState::WellConditioned => 1e-2,
        MatrixConditionState::ModerateCondition => 1e-6,
        MatrixConditionState::IllConditioned => 1e-12,
        MatrixConditionState::NearSingular => 1e-18,
    }
}

// ── Main test ──────────────────────────────────────────────────────────

#[test]
fn perf_p2c008_casp_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();

    // ── PolicyController.decide benchmarks ───────────────────────────

    // Strict mode, benign signal
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        let sig = DecisionSignals::new(2.0, 0.0, 0.0);
        let timings = time_operation(|| {
            let _ = ctrl.decide(sig);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "policy_decide".into(),
            input_desc: "strict_benign".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // Strict mode, high-risk signal
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        let sig = DecisionSignals::new(16.0, 1.0, 1.0);
        let timings = time_operation(|| {
            let _ = ctrl.decide(sig);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "policy_decide".into(),
            input_desc: "strict_high_risk".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // Hardened mode, benign signal
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Hardened, 64);
        let sig = DecisionSignals::new(2.0, 0.0, 0.0);
        let timings = time_operation(|| {
            let _ = ctrl.decide(sig);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "policy_decide".into(),
            input_desc: "hardened_benign".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── SolverPortfolio.select_action benchmarks ─────────────────────

    for state in &MatrixConditionState::ALL {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let rcond = state_to_rcond(state);
        let timings = time_operation(|| {
            let _ = portfolio.select_action(rcond, None);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "solver_select".into(),
            input_desc: format!("{state:?}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── ConformalCalibrator benchmarks ────────────────────────────────

    for &count in &[10usize, 100, 500] {
        let mut cal = ConformalCalibrator::new(0.05, 200);
        let timings = time_operation(|| {
            for i in 0..count {
                cal.observe(1e-12 * (i as f64 + 1.0));
            }
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "calibrator_observe".into(),
            input_desc: format!("{count}_observations"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // calibrator should_fallback
    {
        let mut cal = ConformalCalibrator::new(0.05, 200);
        for i in 0..100 {
            cal.observe(1e-12 * (i as f64 + 1.0));
        }
        let timings = time_operation(|| {
            let _ = cal.should_fallback();
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "calibrator_fallback_check".into(),
            input_desc: "100_obs".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── DecisionSignals creation ─────────────────────────────────────
    {
        let timings = time_operation(|| {
            let _ = DecisionSignals::new(8.0, 0.5, 0.3);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "signal_create".into(),
            input_desc: "new(8.0,0.5,0.3)".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Serialization benchmark ──────────────────────────────────────
    {
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        portfolio.observe_backward_error(1e-10);
        portfolio.observe_backward_error(1e-8);
        let timings = time_operation(|| {
            let _ = portfolio.serialize_jsonl();
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "portfolio_serialize".into(),
            input_desc: "jsonl_2entries".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Hotspot ranking ──────────────────────────────────────────────
    let representative_ops = [
        "policy_decide",
        "solver_select",
        "calibrator_observe",
        "calibrator_fallback_check",
        "signal_create",
        "portfolio_serialize",
    ];
    let mut reps: Vec<&OperationBenchmark> = Vec::new();
    for op in &representative_ops {
        if let Some(b) = benchmarks.iter().rev().find(|b| b.operation == *op) {
            reps.push(b);
        }
    }
    reps.sort_by_key(|b| std::cmp::Reverse(b.median_ns));

    let total_ns: u128 = reps.iter().map(|b| b.median_ns).sum();
    let hotspot_ranking: Vec<HotspotEntry> = reps
        .iter()
        .enumerate()
        .map(|(i, b)| HotspotEntry {
            rank: i + 1,
            operation: b.operation.clone(),
            input_desc: b.input_desc.clone(),
            median_ns: b.median_ns,
            fraction_of_total: if total_ns > 0 {
                b.median_ns as f64 / total_ns as f64
            } else {
                0.0
            },
        })
        .collect();

    // ── Memory notes ─────────────────────────────────────────────────
    let memory_notes = vec![
        MemoryNote {
            operation: "policy_decide".into(),
            notes: "Stack-only: 3 logits, 3-element posterior, 3-element losses. No heap allocation per call.".into(),
        },
        MemoryNote {
            operation: "solver_select".into(),
            notes: "Stack-only: 4-element posterior, 5-element losses. O(1) per call.".into(),
        },
        MemoryNote {
            operation: "calibrator".into(),
            notes: "Bounded VecDeque with capacity. O(capacity) total memory.".into(),
        },
    ];

    // ── Isomorphism checks ───────────────────────────────────────────
    let mut iso_details = Vec::new();

    // Posterior normalization
    {
        let mut ctrl = PolicyController::new(RuntimeMode::Strict, 64);
        let dec = ctrl.decide(DecisionSignals::new(5.0, 0.3, 0.1));
        let sum: f64 = dec.posterior.iter().sum();
        let pass = (sum - 1.0).abs() < 1e-9;
        iso_details.push(IsomorphismDetail {
            operation: "posterior_normalization".into(),
            passes: pass,
            note: format!("sum={sum:.15}"),
        });
    }

    // SolverPortfolio well-conditioned → DirectLU
    {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-2, None);
        let pass = matches!(action, SolverAction::DirectLU);
        iso_details.push(IsomorphismDetail {
            operation: "well_conditioned_directlu".into(),
            passes: pass,
            note: format!("action={action:?}"),
        });
    }

    // IllConditioned → SVDFallback
    {
        let portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let (action, _, _, _) = portfolio.select_action(1e-12, None);
        let pass = matches!(action, SolverAction::SVDFallback);
        iso_details.push(IsomorphismDetail {
            operation: "ill_conditioned_svd".into(),
            passes: pass,
            note: format!("action={action:?}"),
        });
    }

    // Calibrator starts with no fallback
    {
        let cal = ConformalCalibrator::new(0.05, 200);
        let pass = !cal.should_fallback();
        iso_details.push(IsomorphismDetail {
            operation: "calibrator_initial_no_fallback".into(),
            passes: pass,
            note: format!("should_fallback={}", cal.should_fallback()),
        });
    }

    // Signal finiteness check
    {
        let s1 = DecisionSignals::new(1.0, 0.5, 0.3);
        let s2 = DecisionSignals::new(f64::NAN, 0.0, 0.0);
        let pass = s1.is_finite() && !s2.is_finite();
        iso_details.push(IsomorphismDetail {
            operation: "signal_finiteness".into(),
            passes: pass,
            note: format!(
                "finite(1,0.5,0.3)={}, finite(NaN,0,0)={}",
                s1.is_finite(),
                s2.is_finite()
            ),
        });
    }

    // Decision determinism
    {
        let signals = DecisionSignals::new(5.0, 0.2, 0.1);
        let mut c1 = PolicyController::new(RuntimeMode::Strict, 64);
        let mut c2 = PolicyController::new(RuntimeMode::Strict, 64);
        let d1 = c1.decide(signals);
        let d2 = c2.decide(signals);
        let pass = d1.action == d2.action && d1.top_state == d2.top_state;
        iso_details.push(IsomorphismDetail {
            operation: "decision_determinism".into(),
            passes: pass,
            note: format!("action1={:?}, action2={:?}", d1.action, d2.action),
        });
    }

    let all_pass = iso_details.iter().all(|d| d.passes);

    let report = PerfReport {
        generated_at: chrono_lite_now(),
        operation_benchmarks: benchmarks,
        hotspot_ranking,
        memory_notes,
        isomorphism_check: IsomorphismCheck {
            all_operations_pass: all_pass,
            details: iso_details,
        },
    };

    let artifact_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-008/perf");
    std::fs::create_dir_all(&artifact_dir).unwrap();

    let json = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(artifact_dir.join("perf_profile_report.json"), &json).unwrap();

    for d in &report.isomorphism_check.details {
        if d.passes {
            eprintln!("  PASS: {} — {}", d.operation, d.note);
        } else {
            eprintln!("  FAIL: {} — {}", d.operation, d.note);
        }
    }
    assert!(all_pass, "Behavior-isomorphism check failed");

    eprintln!("\n── Top-3 Hotspots ──");
    for h in report.hotspot_ranking.iter().take(3) {
        eprintln!(
            "  #{}: {} ({}) — {:.0}ns ({:.1}%)",
            h.rank,
            h.operation,
            h.input_desc,
            h.median_ns as f64,
            h.fraction_of_total * 100.0,
        );
    }
    eprintln!();
}
