//! P2C-005-H: Performance profiling tests for FFT operations.
//!
//! Produces structured JSON artifacts at:
//!   fixtures/artifacts/FSCI-P2C-005/perf/
//!
//! Covers fft, ifft, rfft, irfft, fft2, and FFT polynomial multiplication at
//! sizes [16, 64, 256, 1024].

use fsci_fft::{FftOptions, fft, fft2, ifft, irfft, polynomial_multiply_fft, rfft};
use serde::Serialize;
use std::time::Instant;

type Complex64 = (f64, f64);

const SIZES: &[usize] = &[16, 64, 256, 1024];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 20;

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
    size_desc: String,
    n: usize,
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
    size_desc: String,
    median_ns: u128,
    fraction_of_total: f64,
}

#[derive(Serialize)]
struct MemoryNote {
    operation: String,
    size: String,
    estimated_input_bytes: usize,
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

fn make_complex_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            ((2.0 * std::f64::consts::PI * t).sin(), 0.0)
        })
        .collect()
}

fn make_real_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * t).sin()
        })
        .collect()
}

fn make_polynomial_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (3.0 * t).sin() - 0.5 * (5.0 * t).cos()
        })
        .collect()
}

fn default_opts() -> FftOptions {
    FftOptions::default()
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

fn complex_abs(c: Complex64) -> f64 {
    (c.0 * c.0 + c.1 * c.1).sqrt()
}

fn direct_polynomial_multiply(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; lhs.len() + rhs.len() - 1];
    for (i, &left) in lhs.iter().enumerate() {
        for (j, &right) in rhs.iter().enumerate() {
            output[i + j] += left * right;
        }
    }
    output
}

// ── Main test ──────────────────────────────────────────────────────────────────

#[test]
fn perf_p2c005_full_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();
    let opts = default_opts();

    // fft
    for &n in SIZES {
        let input = make_complex_input(n);
        let timings = time_operation(|| {
            let _ = fft(&input, &opts).expect("fft");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "fft".into(),
            size_desc: format!("n={n}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ifft
    for &n in SIZES {
        let input = make_complex_input(n);
        let spectrum = fft(&input, &opts).expect("fft");
        let timings = time_operation(|| {
            let _ = ifft(&spectrum, &opts).expect("ifft");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "ifft".into(),
            size_desc: format!("n={n}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // rfft
    for &n in SIZES {
        let input = make_real_input(n);
        let timings = time_operation(|| {
            let _ = rfft(&input, &opts).expect("rfft");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "rfft".into(),
            size_desc: format!("n={n}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // irfft
    for &n in SIZES {
        let input = make_real_input(n);
        let spectrum = rfft(&input, &opts).expect("rfft");
        let timings = time_operation(|| {
            let _ = irfft(&spectrum, Some(n), &opts).expect("irfft");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "irfft".into(),
            size_desc: format!("n={n}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // fft2
    for &side in &[8, 16, 32] {
        let n = side * side;
        let input = make_complex_input(n);
        let timings = time_operation(|| {
            let _ = fft2(&input, (side, side), &opts).expect("fft2");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "fft2".into(),
            size_desc: format!("{side}x{side}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // polynomial_multiply_fft
    for &n in SIZES {
        let lhs = make_polynomial_input(n);
        let rhs = make_polynomial_input(n);
        let timings = time_operation(|| {
            let _ = polynomial_multiply_fft(&lhs, &rhs, &opts).expect("polynomial_multiply_fft");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "polynomial_multiply_fft".into(),
            size_desc: format!("degree<{n} x degree<{n}"),
            n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Hotspot ranking ────────────────────────────────────────────────────────
    // Rank at n=1024
    let mut largest: Vec<_> = benchmarks.iter().filter(|b| b.n == 1024).collect();
    largest.sort_by_key(|b| std::cmp::Reverse(b.median_ns));

    let total_ns: u128 = largest.iter().map(|b| b.median_ns).sum();
    let hotspot_ranking: Vec<HotspotEntry> = largest
        .iter()
        .enumerate()
        .map(|(i, b)| HotspotEntry {
            rank: i + 1,
            operation: b.operation.clone(),
            size_desc: b.size_desc.clone(),
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
            operation: "fft".into(),
            size: "n=1024".into(),
            estimated_input_bytes: 1024 * 16,
            notes: "Input: 1024 complex64 (16KB). O(n²) naive DFT allocates output vec.".into(),
        },
        MemoryNote {
            operation: "fft2".into(),
            size: "32x32".into(),
            estimated_input_bytes: 32 * 32 * 16,
            notes: "Input: 1024 complex64 (16KB). Row + column transforms, two allocations.".into(),
        },
        MemoryNote {
            operation: "polynomial_multiply_fft".into(),
            size: "degree<1024 x degree<1024".into(),
            estimated_input_bytes: 1024 * 8 * 2,
            notes: "Two real coefficient buffers; FFT path pads to next power-of-two product length and compares against direct O(n^2) oracle.".into(),
        },
    ];

    // ── Isomorphism check ──────────────────────────────────────────────────────
    let mut iso_details = Vec::new();

    // fft-ifft roundtrip
    {
        let input = make_complex_input(64);
        let spectrum = fft(&input, &opts).expect("fft");
        let recovered = ifft(&spectrum, &opts).expect("ifft");
        let max_err: f64 = input
            .iter()
            .zip(&recovered)
            .map(|(a, b)| complex_abs((a.0 - b.0, a.1 - b.1)))
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        iso_details.push(IsomorphismDetail {
            operation: "fft_ifft_roundtrip".into(),
            passes: max_err < 1e-10,
            note: format!("max_err={max_err:.2e}"),
        });
    }

    // rfft-irfft roundtrip
    {
        let input = make_real_input(64);
        let spectrum = rfft(&input, &opts).expect("rfft");
        let recovered = irfft(&spectrum, Some(64), &opts).expect("irfft");
        let max_err: f64 = input
            .iter()
            .zip(&recovered)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        iso_details.push(IsomorphismDetail {
            operation: "rfft_irfft_roundtrip".into(),
            passes: max_err < 1e-10,
            note: format!("max_err={max_err:.2e}"),
        });
    }

    // Parseval's theorem: sum|x|² = (1/N) sum|X|²
    {
        let input = make_complex_input(64);
        let spectrum = fft(&input, &opts).expect("fft");
        let energy_time: f64 = input.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum();
        let energy_freq: f64 = spectrum.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum::<f64>() / 64.0;
        let rel_err = (energy_time - energy_freq).abs() / energy_time.max(1e-30);
        iso_details.push(IsomorphismDetail {
            operation: "parseval_energy".into(),
            passes: rel_err < 1e-10,
            note: format!("rel_err={rel_err:.2e}"),
        });
    }

    // FFT polynomial multiplication equals direct coefficient convolution.
    {
        let lhs = make_polynomial_input(64);
        let rhs = make_polynomial_input(64);
        let fast = polynomial_multiply_fft(&lhs, &rhs, &opts).expect("polynomial_multiply_fft");
        let direct = direct_polynomial_multiply(&lhs, &rhs);
        let max_err: f64 =
            fast.iter()
                .zip(&direct)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
        iso_details.push(IsomorphismDetail {
            operation: "polynomial_multiply_fft_direct_convolution".into(),
            passes: max_err < 1e-9,
            note: format!("max_err={max_err:.2e}"),
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
        .join("fixtures/artifacts/FSCI-P2C-005/perf");
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

    eprintln!("\n── Top-3 Hotspots (at n=1024) ──");
    for h in report.hotspot_ranking.iter().take(3) {
        eprintln!(
            "  #{}: {} {} — {:.3}ms ({:.1}%)",
            h.rank,
            h.operation,
            h.size_desc,
            h.median_ns as f64 / 1_000_000.0,
            h.fraction_of_total * 100.0,
        );
    }
    eprintln!();
}
