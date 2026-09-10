//! P2C-004-H: Performance profiling tests for sparse matrix operations.
//!
//! Produces structured JSON artifacts at:
//!   fixtures/artifacts/FSCI-P2C-004/perf/
//!
//! Covers CSR construction, spmv, format conversion, arithmetic across
//! [100×100 5%, 1000×1000 1%, 10000×10000 0.1%] configurations.

use fsci_sparse::{
    CsrMatrix, FormatConvertible, Shape2D, add_csr, diags, eye, random, scale_csr, spmv_csr,
};
use serde::Serialize;
use std::time::Instant;

const CONFIGS: &[(usize, f64, &str)] = &[
    (100, 0.05, "100x100_d5"),
    (1_000, 0.01, "1000x1000_d1"),
    (10_000, 0.001, "10000x10000_d01"),
];
const SEED: u64 = 0xBEEF_CAFE;
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
    matrix_desc: String,
    matrix_n: usize,
    nnz: usize,
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
    matrix_desc: String,
    median_ns: u128,
    fraction_of_total: f64,
}

#[derive(Serialize)]
struct MemoryNote {
    operation: String,
    matrix_size: String,
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

fn make_random_csr(n: usize, density: f64) -> CsrMatrix {
    random(Shape2D::new(n, n), density, SEED)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect()
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

// ── Main test ──────────────────────────────────────────────────────────────────

#[test]
fn perf_p2c004_full_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();

    // CSR construction from COO
    for &(n, density, label) in CONFIGS {
        let coo = random(Shape2D::new(n, n), density, SEED).expect("random coo");
        let nnz = coo.nnz();
        let timings = time_operation(|| {
            let _ = coo.to_csr().expect("coo->csr");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "coo_to_csr".into(),
            matrix_desc: label.into(),
            matrix_n: n,
            nnz,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // spmv_csr
    for &(n, density, label) in CONFIGS {
        let csr = make_random_csr(n, density);
        let nnz = csr.nnz();
        let vec = make_vector(n);
        let timings = time_operation(|| {
            let _ = spmv_csr(&csr, &vec).expect("spmv");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "spmv_csr".into(),
            matrix_desc: label.into(),
            matrix_n: n,
            nnz,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // CSR -> CSC conversion
    for &(n, density, label) in CONFIGS {
        let csr = make_random_csr(n, density);
        let nnz = csr.nnz();
        let timings = time_operation(|| {
            let _ = csr.to_csc().expect("csr->csc");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "csr_to_csc".into(),
            matrix_desc: label.into(),
            matrix_n: n,
            nnz,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // add_csr
    for &(n, density, label) in CONFIGS {
        let a = make_random_csr(n, density);
        let b = random(Shape2D::new(n, n), density, SEED ^ 0xFF)
            .expect("random b")
            .to_csr()
            .expect("b csr");
        let nnz = a.nnz() + b.nnz();
        let timings = time_operation(|| {
            let _ = add_csr(&a, &b).expect("add");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "add_csr".into(),
            matrix_desc: label.into(),
            matrix_n: n,
            nnz,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // scale_csr
    for &(n, density, label) in CONFIGS {
        let a = make_random_csr(n, density);
        let nnz = a.nnz();
        let timings = time_operation(|| {
            let _ = scale_csr(&a, 2.5).expect("scale");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "scale_csr".into(),
            matrix_desc: label.into(),
            matrix_n: n,
            nnz,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // eye construction
    for &n in &[100usize, 1_000, 10_000] {
        let timings = time_operation(|| {
            let _ = eye(n).expect("eye");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "eye".into(),
            matrix_desc: format!("{n}x{n}"),
            matrix_n: n,
            nnz: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // diags (tridiagonal)
    for &n in &[100usize, 1_000, 10_000] {
        let sub = vec![-1.0; n.saturating_sub(1)];
        let main_d = vec![2.0; n];
        let sup = vec![-1.0; n.saturating_sub(1)];
        let timings = time_operation(|| {
            let _ = diags(
                &[sub.clone(), main_d.clone(), sup.clone()],
                &[-1, 0, 1],
                Some(Shape2D::new(n, n)),
            )
            .expect("tridiag");
        });
        let (median, p95, min_v, max_v, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "diags_tridiag".into(),
            matrix_desc: format!("{n}x{n}"),
            matrix_n: n,
            nnz: 3 * n - 2,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min_v,
            max_ns: max_v,
            mean_ns: mean,
        });
    }

    // ── Hotspot ranking ────────────────────────────────────────────────────────
    // Rank at the largest config (10000x10000)
    let mut largest: Vec<_> = benchmarks.iter().filter(|b| b.matrix_n == 10_000).collect();
    largest.sort_by_key(|b| std::cmp::Reverse(b.median_ns));

    let total_ns: u128 = largest.iter().map(|b| b.median_ns).sum();
    let hotspot_ranking: Vec<HotspotEntry> = largest
        .iter()
        .enumerate()
        .map(|(i, b)| HotspotEntry {
            rank: i + 1,
            operation: b.operation.clone(),
            matrix_desc: b.matrix_desc.clone(),
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
            operation: "spmv_csr".into(),
            matrix_size: "10000x10000_d01".into(),
            estimated_input_bytes: 10_000 * 10 * 16 + 10_000 * 8,
            notes: "CSR: ~10k nnz × (8B data + 8B index) + 10k indptr. Vector: 80KB.".into(),
        },
        MemoryNote {
            operation: "csr_to_csc".into(),
            matrix_size: "10000x10000_d01".into(),
            estimated_input_bytes: 10_000 * 10 * 16,
            notes: "Transpose allocates new indptr/indices/data arrays of same nnz.".into(),
        },
        MemoryNote {
            operation: "add_csr".into(),
            matrix_size: "10000x10000_d01".into(),
            estimated_input_bytes: 2 * 10_000 * 10 * 16,
            notes: "Two CSR inputs. Output nnz <= sum of input nnz.".into(),
        },
    ];

    // ── Isomorphism check ──────────────────────────────────────────────────────
    let mut iso_details = Vec::new();

    // spmv isomorphism: identity * v = v
    {
        let id = eye(100).expect("eye");
        let v = make_vector(100);
        let result = spmv_csr(&id, &v).expect("spmv");
        let max_err: f64 =
            result
                .iter()
                .zip(&v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
        iso_details.push(IsomorphismDetail {
            operation: "spmv_identity".into(),
            passes: max_err < 1e-12,
            note: format!("max_err={max_err:.2e}"),
        });
    }

    // format roundtrip isomorphism: csr -> csc -> csr preserves data
    {
        let csr = make_random_csr(100, 0.05);
        let roundtrip = csr.to_csc().expect("csr->csc").to_csr().expect("csc->csr");
        let v = make_vector(100);
        let orig = spmv_csr(&csr, &v).expect("spmv orig");
        let rt = spmv_csr(&roundtrip, &v).expect("spmv roundtrip");
        let max_err: f64 =
            orig.iter()
                .zip(&rt)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
        iso_details.push(IsomorphismDetail {
            operation: "format_roundtrip".into(),
            passes: max_err < 1e-12,
            note: format!("spmv_diff={max_err:.2e}"),
        });
    }

    // add isomorphism: A + 0 = A
    {
        let a = make_random_csr(100, 0.05);
        let zero = random(Shape2D::new(100, 100), 0.0, 1)
            .expect("zero")
            .to_csr()
            .expect("zero csr");
        let sum = add_csr(&a, &zero).expect("add zero");
        let v = make_vector(100);
        let orig = spmv_csr(&a, &v).expect("spmv a");
        let sum_v = spmv_csr(&sum, &v).expect("spmv sum");
        let max_err: f64 =
            orig.iter()
                .zip(&sum_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
        iso_details.push(IsomorphismDetail {
            operation: "add_zero_identity".into(),
            passes: max_err < 1e-12,
            note: format!("max_err={max_err:.2e}"),
        });
    }

    // scale isomorphism: scale(A, 1.0) = A
    {
        let a = make_random_csr(100, 0.05);
        let scaled = scale_csr(&a, 1.0).expect("scale 1.0");
        let v = make_vector(100);
        let orig = spmv_csr(&a, &v).expect("spmv a");
        let sc = spmv_csr(&scaled, &v).expect("spmv scaled");
        let max_err: f64 =
            orig.iter()
                .zip(&sc)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
        iso_details.push(IsomorphismDetail {
            operation: "scale_identity".into(),
            passes: max_err < 1e-12,
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
        .join("fixtures/artifacts/FSCI-P2C-004/perf");
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

    eprintln!("\n── Top-3 Hotspots (at 10000-size) ──");
    for h in report.hotspot_ranking.iter().take(3) {
        eprintln!(
            "  #{}: {} {} — {:.3}ms ({:.1}%)",
            h.rank,
            h.operation,
            h.matrix_desc,
            h.median_ns as f64 / 1_000_000.0,
            h.fraction_of_total * 100.0,
        );
    }
    eprintln!();
}
