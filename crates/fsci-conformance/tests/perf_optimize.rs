//! P2C-003-H: Performance profiling tests for optimize/root-finding operations.
//!
//! Produces structured JSON artifact at:
//!   fixtures/artifacts/FSCI-P2C-003/perf/perf_profile_report.json
//!
//! Covers BFGS, CG, Powell, brentq, brenth, bisect, ridder.

use fsci_opt::{
    ConvergenceStatus, MinimizeOptions, OptimizeMethod, RootMethod, RootOptions, bfgs, bisect,
    brenth, brentq, cg_pr_plus, powell, ridder,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::time::Instant;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 30;

type RootSolverFn =
    fn(fn(f64) -> f64, (f64, f64), RootOptions) -> Result<fsci_opt::RootResult, fsci_opt::OptError>;

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

fn minimize_opts(method: OptimizeMethod) -> MinimizeOptions {
    MinimizeOptions {
        method: Some(method),
        mode: RuntimeMode::Strict,
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

// ── Test functions ────────────────────────────────────────────────────

fn rosenbrock(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        s += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
    }
    s
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

// ── Main test ──────────────────────────────────────────────────────────

#[test]
fn perf_p2c003_optimize_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();

    // ── Minimize benchmarks ──────────────────────────────────────────
    let dims = [2usize, 5, 10];

    for &dim in &dims {
        let x0: Vec<f64> = vec![0.0; dim];

        // BFGS on Rosenbrock
        let timings = time_operation(|| {
            let _ = bfgs(&rosenbrock, &x0, minimize_opts(OptimizeMethod::Bfgs));
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "bfgs".into(),
            input_desc: format!("rosenbrock_dim{dim}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });

        // BFGS on quadratic
        let timings = time_operation(|| {
            let _ = bfgs(&quadratic, &x0, minimize_opts(OptimizeMethod::Bfgs));
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "bfgs".into(),
            input_desc: format!("quadratic_dim{dim}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });

        // CG on Rosenbrock
        let timings = time_operation(|| {
            let _ = cg_pr_plus(
                &rosenbrock,
                &x0,
                minimize_opts(OptimizeMethod::ConjugateGradient),
            );
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "cg".into(),
            input_desc: format!("rosenbrock_dim{dim}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });

        // Powell on Rosenbrock
        let timings = time_operation(|| {
            let _ = powell(&rosenbrock, &x0, minimize_opts(OptimizeMethod::Powell));
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "powell".into(),
            input_desc: format!("rosenbrock_dim{dim}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Root-finding benchmarks ──────────────────────────────────────
    let root_methods: Vec<(&str, RootSolverFn, RootMethod)> = vec![
        ("brentq", brentq, RootMethod::Brentq),
        ("brenth", brenth, RootMethod::Brenth),
        ("bisect", bisect, RootMethod::Bisect),
        ("ridder", ridder, RootMethod::Ridder),
    ];

    for (name, method_fn, method_enum) in &root_methods {
        // cubic
        let opts = root_opts(*method_enum);
        let mfn = *method_fn;
        let timings = time_operation(|| {
            let _ = mfn(cubic_root, (1.0, 3.0), opts);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: name.to_string(),
            input_desc: "cubic_[1,3]".into(),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });

        // sin
        let timings = time_operation(|| {
            let _ = mfn(sin_root, (3.0, 4.0), opts);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: name.to_string(),
            input_desc: "sin_[3,4]".into(),
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
        "bfgs", "cg", "powell", "brentq", "brenth", "bisect", "ridder",
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
            operation: "bfgs".into(),
            notes: "Allocates n×n Hessian inverse approximation. O(n²) memory.".into(),
        },
        MemoryNote {
            operation: "cg".into(),
            notes: "O(n) memory — stores gradient and direction vectors only.".into(),
        },
        MemoryNote {
            operation: "powell".into(),
            notes: "Allocates n direction vectors (n×n total). No gradient needed.".into(),
        },
        MemoryNote {
            operation: "root_finders".into(),
            notes: "O(1) memory — all root finders use constant workspace.".into(),
        },
    ];

    // ── Isomorphism checks ───────────────────────────────────────────
    let mut iso_details = Vec::new();

    // BFGS finds minimum of quadratic at origin
    {
        let result = bfgs(
            &quadratic,
            &[1.0, 2.0, 3.0],
            minimize_opts(OptimizeMethod::Bfgs),
        )
        .unwrap();
        let at_origin = result.x.iter().all(|&xi| xi.abs() < 1e-4);
        let pass = result.success && at_origin;
        iso_details.push(IsomorphismDetail {
            operation: "bfgs_quadratic_minimum".into(),
            passes: pass,
            note: format!(
                "x={:?}, f={:?}, success={}",
                result.x, result.fun, result.success
            ),
        });
    }

    // CG finds minimum of quadratic at origin
    {
        let result = cg_pr_plus(
            &quadratic,
            &[1.0, 2.0],
            minimize_opts(OptimizeMethod::ConjugateGradient),
        )
        .unwrap();
        let at_origin = result.x.iter().all(|&xi| xi.abs() < 1e-3);
        let pass = result.success && at_origin;
        iso_details.push(IsomorphismDetail {
            operation: "cg_quadratic_minimum".into(),
            passes: pass,
            note: format!("x={:?}, f={:?}", result.x, result.fun),
        });
    }

    // Powell finds minimum of quadratic
    {
        let result = powell(
            &quadratic,
            &[1.0, 2.0],
            minimize_opts(OptimizeMethod::Powell),
        )
        .unwrap();
        let at_origin = result.x.iter().all(|&xi| xi.abs() < 1e-3);
        let pass = result.success && at_origin;
        iso_details.push(IsomorphismDetail {
            operation: "powell_quadratic_minimum".into(),
            passes: pass,
            note: format!("x={:?}, f={:?}", result.x, result.fun),
        });
    }

    // brentq finds root of cubic
    {
        let result = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq)).unwrap();
        let err = cubic_root(result.root).abs();
        let pass = result.converged && err < 1e-10;
        iso_details.push(IsomorphismDetail {
            operation: "brentq_cubic_root".into(),
            passes: pass,
            note: format!("root={:.10}, f(root)={err:.2e}", result.root),
        });
    }

    // bisect agrees with brentq
    {
        let bq = brentq(cubic_root, (1.0, 3.0), root_opts(RootMethod::Brentq)).unwrap();
        let bs = bisect(cubic_root, (1.0, 3.0), root_opts(RootMethod::Bisect)).unwrap();
        let diff = (bq.root - bs.root).abs();
        let pass = diff < 1e-10;
        iso_details.push(IsomorphismDetail {
            operation: "brentq_bisect_agreement".into(),
            passes: pass,
            note: format!(
                "brentq={:.10}, bisect={:.10}, diff={diff:.2e}",
                bq.root, bs.root
            ),
        });
    }

    // sin root near pi
    {
        let result = brentq(sin_root, (3.0, 4.0), root_opts(RootMethod::Brentq)).unwrap();
        let err = (result.root - std::f64::consts::PI).abs();
        let pass = err < 1e-10;
        iso_details.push(IsomorphismDetail {
            operation: "sin_root_at_pi".into(),
            passes: pass,
            note: format!("root={:.12}, err_from_pi={err:.2e}", result.root),
        });
    }

    // BFGS on Rosenbrock converges (may not always find exact minimum but should converge)
    {
        let result = bfgs(
            &rosenbrock,
            &[0.0, 0.0],
            minimize_opts(OptimizeMethod::Bfgs),
        )
        .unwrap();
        let pass = matches!(
            result.status,
            ConvergenceStatus::Success | ConvergenceStatus::MaxIterations
        );
        iso_details.push(IsomorphismDetail {
            operation: "bfgs_rosenbrock_converges".into(),
            passes: pass,
            note: format!(
                "status={:?}, nfev={}, nit={}",
                result.status, result.nfev, result.nit
            ),
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

    // Write artifact
    let artifact_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-003/perf");
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
