//! P2C-006-H: Performance profiling tests for special function operations.
//!
//! Produces structured JSON artifacts at:
//!   fixtures/artifacts/FSCI-P2C-006/perf/
//!
//! Covers gamma, gammaln, rgamma, gammainc, erf, erfc, erfinv, beta, j0, j1, y0.

use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialTensor, beta, erf, erfc, erfinv, gamma, gammainc, gammaincc, gammaln, j0, j1, rgamma, y0,
};
use serde::Serialize;
use std::time::Instant;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 50;

// ── Data structures ────────────────────────────────────────────────────────────

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

// ── Helpers ─────────────────────────────────────────────────────────────────────

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

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

// ── Benchmark a single function call pattern ──────────────────────────────────

fn bench_unary(
    operation: &str,
    inputs: &[f64],
    f: impl Fn(&SpecialTensor) -> SpecialTensor,
) -> Vec<OperationBenchmark> {
    let mut results = Vec::new();
    for &x in inputs {
        let input = scalar(x);
        let timings = time_operation(|| {
            let _ = f(&input);
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        results.push(OperationBenchmark {
            operation: operation.into(),
            input_desc: format!("x={x}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }
    results
}

// ── Main test ──────────────────────────────────────────────────────────────────

#[test]
fn perf_p2c006_full_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();

    let gamma_inputs: &[f64] = &[0.5, 1.0, 2.5, 5.0, 10.0, 50.0, 100.0];
    let erf_inputs: &[f64] = &[-3.0, -1.0, 0.0, 0.5, 1.0, 3.0];
    let bessel_inputs: &[f64] = &[0.1, 1.0, 5.0, 10.0, 20.0];

    // gamma
    benchmarks.extend(bench_unary("gamma", gamma_inputs, |x| {
        gamma(x, RuntimeMode::Strict).expect("gamma")
    }));

    // gammaln
    benchmarks.extend(bench_unary("gammaln", gamma_inputs, |x| {
        gammaln(x, RuntimeMode::Strict).expect("gammaln")
    }));

    // rgamma
    benchmarks.extend(bench_unary("rgamma", gamma_inputs, |x| {
        rgamma(x, RuntimeMode::Strict).expect("rgamma")
    }));

    // erf
    benchmarks.extend(bench_unary("erf", erf_inputs, |x| {
        erf(x, RuntimeMode::Strict).expect("erf")
    }));

    // erfc
    benchmarks.extend(bench_unary("erfc", erf_inputs, |x| {
        erfc(x, RuntimeMode::Strict).expect("erfc")
    }));

    // erfinv
    let erfinv_inputs: &[f64] = &[-0.9, -0.5, 0.0, 0.5, 0.9];
    benchmarks.extend(bench_unary("erfinv", erfinv_inputs, |x| {
        erfinv(x, RuntimeMode::Strict).expect("erfinv")
    }));

    // j0
    benchmarks.extend(bench_unary("j0", bessel_inputs, |x| {
        j0(x, RuntimeMode::Strict).expect("j0")
    }));

    // j1
    benchmarks.extend(bench_unary("j1", bessel_inputs, |x| {
        j1(x, RuntimeMode::Strict).expect("j1")
    }));

    // y0
    benchmarks.extend(bench_unary("y0", bessel_inputs, |x| {
        y0(x, RuntimeMode::Strict).expect("y0")
    }));

    // gammainc (binary)
    let inc_pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (10.0, 10.0)];
    for &(a, x) in inc_pairs {
        let sa = scalar(a);
        let sx = scalar(x);
        let timings = time_operation(|| {
            let _ = gammainc(&sa, &sx, RuntimeMode::Strict).expect("gammainc");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "gammainc".into(),
            input_desc: format!("a={a},x={x}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // beta (binary)
    let beta_pairs: &[(f64, f64)] = &[(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (5.0, 5.0)];
    for &(a, b_val) in beta_pairs {
        let sa = scalar(a);
        let sb = scalar(b_val);
        let timings = time_operation(|| {
            let _ = beta(&sa, &sb, RuntimeMode::Strict).expect("beta");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "beta".into(),
            input_desc: format!("a={a},b={b_val}"),
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Hotspot ranking ────────────────────────────────────────────────────────
    // Pick one representative input per operation, rank by median
    let representative_ops = [
        "gamma", "gammaln", "rgamma", "erf", "erfc", "erfinv", "j0", "j1", "y0", "gammainc", "beta",
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

    // ── Memory notes ───────────────────────────────────────────────────────────
    let memory_notes = vec![
        MemoryNote {
            operation: "all_scalar".into(),
            notes: "Scalar operations: O(1) memory. SpecialTensor enum is 24 bytes.".into(),
        },
        MemoryNote {
            operation: "gammainc".into(),
            notes: "Series expansion may allocate temporary accumulators for convergence.".into(),
        },
    ];

    // ── Isomorphism check ──────────────────────────────────────────────────────
    let mut iso_details = Vec::new();

    // gamma(1)=1, gamma(5)=24
    {
        let g1 = real_val(&gamma(&scalar(1.0), RuntimeMode::Strict).unwrap());
        let g5 = real_val(&gamma(&scalar(5.0), RuntimeMode::Strict).unwrap());
        let pass = (g1 - 1.0).abs() < 1e-12 && (g5 - 24.0).abs() < 1e-10;
        iso_details.push(IsomorphismDetail {
            operation: "gamma_contract".into(),
            passes: pass,
            note: format!("gamma(1)={g1:.6e}, gamma(5)={g5:.6e}"),
        });
    }

    // erf+erfc=1
    {
        let e = real_val(&erf(&scalar(1.5), RuntimeMode::Strict).unwrap());
        let ec = real_val(&erfc(&scalar(1.5), RuntimeMode::Strict).unwrap());
        let sum = e + ec;
        let pass = (sum - 1.0).abs() < 1e-12;
        iso_details.push(IsomorphismDetail {
            operation: "erf_erfc_identity".into(),
            passes: pass,
            note: format!("erf(1.5)+erfc(1.5)={sum:.6e}"),
        });
    }

    // erfinv roundtrip
    {
        let x = 0.7;
        let inv = real_val(&erfinv(&scalar(x), RuntimeMode::Strict).unwrap());
        let roundtrip = real_val(&erf(&scalar(inv), RuntimeMode::Strict).unwrap());
        let err = (roundtrip - x).abs();
        iso_details.push(IsomorphismDetail {
            operation: "erfinv_roundtrip".into(),
            passes: err < 1e-10,
            note: format!("err={err:.2e}"),
        });
    }

    // gammainc + gammaincc = 1
    {
        let a = scalar(3.0);
        let x = scalar(2.0);
        let inc = real_val(&gammainc(&a, &x, RuntimeMode::Strict).unwrap());
        let incc = real_val(&gammaincc(&a, &x, RuntimeMode::Strict).unwrap());
        let sum = inc + incc;
        let pass = (sum - 1.0).abs() < 1e-10;
        iso_details.push(IsomorphismDetail {
            operation: "gammainc_complement".into(),
            passes: pass,
            note: format!("gammainc(3,2)+gammaincc(3,2)={sum:.6e}"),
        });
    }

    // beta-gamma relation: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    {
        let a_val = 2.0;
        let b_val = 3.0;
        let b_ab = real_val(&beta(&scalar(a_val), &scalar(b_val), RuntimeMode::Strict).unwrap());
        let ga = real_val(&gamma(&scalar(a_val), RuntimeMode::Strict).unwrap());
        let gb = real_val(&gamma(&scalar(b_val), RuntimeMode::Strict).unwrap());
        let gab = real_val(&gamma(&scalar(a_val + b_val), RuntimeMode::Strict).unwrap());
        let expected = ga * gb / gab;
        let rel_err = (b_ab - expected).abs() / expected.abs().max(1e-30);
        iso_details.push(IsomorphismDetail {
            operation: "beta_gamma_relation".into(),
            passes: rel_err < 1e-10,
            note: format!("rel_err={rel_err:.2e}"),
        });
    }

    // j0(0)=1, j1(0)=0
    {
        let j0_0 = real_val(&j0(&scalar(0.0), RuntimeMode::Strict).unwrap());
        let j1_0 = real_val(&j1(&scalar(0.0), RuntimeMode::Strict).unwrap());
        let pass = (j0_0 - 1.0).abs() < 1e-8 && j1_0.abs() < 1e-8;
        iso_details.push(IsomorphismDetail {
            operation: "bessel_boundary".into(),
            passes: pass,
            note: format!("j0(0)={j0_0:.6e}, j1(0)={j1_0:.6e}"),
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
        .join("fixtures/artifacts/FSCI-P2C-006/perf");
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
