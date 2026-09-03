//! P2C-002-H: Performance profiling tests for linalg operations.
//!
//! Produces structured JSON artifacts at:
//!   fixtures/artifacts/P2C-002/perf/
//!
//! Covers all 7 operations × 4 matrix sizes with:
//! - Median/p95 timing in nanoseconds
//! - Hotspot ranking by CPU time
//! - Memory delta estimation for large matrices
//! - Behavior-isomorphism verification (conformance still passes)

use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, TriangularSolveOptions, det, inv, lstsq,
    pinv, solve, solve_banded, solve_triangular,
};
use fsci_runtime::RuntimeMode;
use serde::Serialize;
use std::time::Instant;

const SIZES: &[usize] = &[4, 16, 64, 256];
const WARMUP_ITERS: usize = 2;
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
    rows: usize,
    cols: usize,
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

// ── Matrix generators ──────────────────────────────────────────────────────────

fn make_diag_dominant(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (n as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_upper_triangular(n: usize) -> Vec<Vec<f64>> {
    let mut a = make_diag_dominant(n);
    for (i, row) in a.iter_mut().enumerate() {
        for cell in row.iter_mut().take(i) {
            *cell = 0.0;
        }
    }
    a
}

fn make_overdetermined(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols]; rows];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (cols as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_tridiag_banded(n: usize) -> ((usize, usize), Vec<Vec<f64>>) {
    // LAPACK band storage: ab[nupper + i - j][j] = A[i][j]
    // For tridiag (l=1, u=1):
    //   ab[0][j] = A[j-1][j]  (superdiag, j >= 1)
    //   ab[1][j] = A[j][j]    (main diagonal)
    //   ab[2][j] = A[j+1][j]  (subdiag, j <= n-2)
    let mut ab = vec![vec![0.0; n]; 3];
    ab[1].fill(4.0); // main diagonal
    for col in ab[0].iter_mut().skip(1) {
        *col = -1.0; // superdiag: A[j-1][j], j >= 1
    }
    for col in ab[2].iter_mut().take(n.saturating_sub(1)) {
        *col = -1.0; // subdiag: A[j+1][j], j <= n-2
    }
    ((1, 1), ab)
}

fn make_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

// ── Timing harness ─────────────────────────────────────────────────────────────

fn time_operation<F: FnMut()>(mut f: F) -> Vec<u128> {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        f();
    }
    // Measure
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
    let min = timings[0];
    let max = timings[n - 1];
    let mean = timings.iter().sum::<u128>() / n as u128;
    (median, p95, min, max, mean)
}

// ── Main test ──────────────────────────────────────────────────────────────────

// Runs ~354s on CI; gate it behind `--ignored` so that
// `cargo test --workspace --all-targets` stays reliable on rch-offloaded
// builds where long-running binaries were being reaped before later
// tests could launch (br-smhw). Opt-in with
// `cargo test -p fsci-conformance --test perf_linalg -- --ignored`.
#[test]
#[ignore = "long-running perf baseline — run with --ignored (see br-smhw)"]
fn perf_p2c002_full_profile() {
    let mut benchmarks: Vec<OperationBenchmark> = Vec::new();

    // solve
    for &n in SIZES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        let timings = time_operation(|| {
            let _ = solve(&a, &b, SolveOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "solve".into(),
            matrix_desc: format!("{n}x{n}"),
            rows: n,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // solve_triangular
    for &n in SIZES {
        let a = make_upper_triangular(n);
        let b = make_rhs(n);
        let timings = time_operation(|| {
            let _ = solve_triangular(&a, &b, TriangularSolveOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "solve_triangular".into(),
            matrix_desc: format!("{n}x{n}"),
            rows: n,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // solve_banded
    for &n in SIZES {
        let (l_u, ab) = make_tridiag_banded(n);
        let b = make_rhs(n);
        let timings = time_operation(|| {
            let _ = solve_banded(l_u, &ab, &b, SolveOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "solve_banded".into(),
            matrix_desc: format!("{n}x{n}"),
            rows: n,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // inv
    for &n in SIZES {
        let a = make_diag_dominant(n);
        let timings = time_operation(|| {
            let _ = inv(&a, InvOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "inv".into(),
            matrix_desc: format!("{n}x{n}"),
            rows: n,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // det
    for &n in SIZES {
        let a = make_diag_dominant(n);
        let timings = time_operation(|| {
            let _ = det(&a, RuntimeMode::Strict, true);
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "det".into(),
            matrix_desc: format!("{n}x{n}"),
            rows: n,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // lstsq (overdetermined: 2n x n)
    for &n in SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let b = make_rhs(rows);
        let timings = time_operation(|| {
            let _ = lstsq(&a, &b, LstsqOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "lstsq".into(),
            matrix_desc: format!("{rows}x{n}"),
            rows,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // pinv (overdetermined: 2n x n)
    for &n in SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let timings = time_operation(|| {
            let _ = pinv(&a, PinvOptions::default());
        });
        let (median, p95, min, max, mean) = compute_stats(&timings);
        benchmarks.push(OperationBenchmark {
            operation: "pinv".into(),
            matrix_desc: format!("{rows}x{n}"),
            rows,
            cols: n,
            iterations: BENCH_ITERS,
            median_ns: median,
            p95_ns: p95,
            min_ns: min,
            max_ns: max,
            mean_ns: mean,
        });
    }

    // ── Hotspot ranking ────────────────────────────────────────────────────────

    // Rank by median time at the largest size (256x256 or 512x256)
    let mut largest: Vec<_> = benchmarks.iter().filter(|b| b.cols == 256).collect();
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
            fraction_of_total: b.median_ns as f64 / total_ns as f64,
        })
        .collect();

    // ── Memory notes ───────────────────────────────────────────────────────────

    let memory_notes = vec![
        MemoryNote {
            operation: "solve".into(),
            matrix_size: "256x256".into(),
            estimated_input_bytes: 256 * 256 * 8 + 256 * 8,
            notes: "Input: A (512KB) + b (2KB). LU decomposition allocates DMatrix clone.".into(),
        },
        MemoryNote {
            operation: "inv".into(),
            matrix_size: "256x256".into(),
            estimated_input_bytes: 256 * 256 * 8,
            notes: "Input: A (512KB). LU + solve for each column of identity = n solves.".into(),
        },
        MemoryNote {
            operation: "lstsq".into(),
            matrix_size: "512x256".into(),
            estimated_input_bytes: 512 * 256 * 8 + 512 * 8,
            notes: "Input: A (1MB) + b (4KB). SVD (Gelsd) allocates U, S, Vt matrices.".into(),
        },
        MemoryNote {
            operation: "pinv".into(),
            matrix_size: "512x256".into(),
            estimated_input_bytes: 512 * 256 * 8,
            notes: "Input: A (1MB). Full SVD + matrix multiply for pseudoinverse.".into(),
        },
    ];

    // ── Isomorphism check ──────────────────────────────────────────────────────
    // Verify operations still produce correct results (behavior unchanged)

    let mut iso_details = Vec::new();

    // solve isomorphism
    {
        let a = make_diag_dominant(16);
        let b = make_rhs(16);
        let r = solve(&a, &b, SolveOptions::default()).unwrap();
        // Verify Ax = b
        let mut residual = 0.0_f64;
        for i in 0..16 {
            let ax_i: f64 = (0..16).map(|j| a[i][j] * r.x[j]).sum();
            residual = residual.max((ax_i - b[i]).abs());
        }
        iso_details.push(IsomorphismDetail {
            operation: "solve".into(),
            passes: residual < 1e-10,
            note: format!("max_residual={residual:.2e}"),
        });
    }

    // solve_triangular isomorphism
    {
        let a = make_upper_triangular(16);
        let b = make_rhs(16);
        let r = solve_triangular(&a, &b, TriangularSolveOptions::default()).unwrap();
        let mut residual = 0.0_f64;
        for i in 0..16 {
            let ax_i: f64 = (0..16).map(|j| a[i][j] * r.x[j]).sum();
            residual = residual.max((ax_i - b[i]).abs());
        }
        iso_details.push(IsomorphismDetail {
            operation: "solve_triangular".into(),
            passes: residual < 1e-10,
            note: format!("max_residual={residual:.2e}"),
        });
    }

    // solve_banded isomorphism: compare with dense solve
    {
        let n = 16;
        let (l_u, ab) = make_tridiag_banded(n);
        let b = make_rhs(n);
        let banded_r = solve_banded(l_u, &ab, &b, SolveOptions::default()).unwrap();
        // Build equivalent dense tridiag and solve directly
        let mut dense = vec![vec![0.0; n]; n];
        for j in 0..n {
            dense[j][j] = 4.0;
            if j > 0 {
                dense[j][j - 1] = -1.0;
            }
            if j + 1 < n {
                dense[j][j + 1] = -1.0;
            }
        }
        let dense_r = solve(&dense, &b, SolveOptions::default()).unwrap();
        let mut diff = 0.0_f64;
        for i in 0..n {
            diff = diff.max((banded_r.x[i] - dense_r.x[i]).abs());
        }
        iso_details.push(IsomorphismDetail {
            operation: "solve_banded".into(),
            passes: diff < 1e-10,
            note: format!("banded_vs_dense_diff={diff:.2e}"),
        });
    }

    // inv isomorphism: A @ inv(A) ≈ I
    {
        let a = make_diag_dominant(16);
        let r = inv(&a, InvOptions::default()).unwrap();
        let mut max_err = 0.0_f64;
        for (i, a_row) in a.iter().enumerate() {
            for j in 0..16 {
                let val: f64 = a_row
                    .iter()
                    .enumerate()
                    .map(|(k, &a_ik)| a_ik * r.inverse[k][j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((val - expected).abs());
            }
        }
        iso_details.push(IsomorphismDetail {
            operation: "inv".into(),
            passes: max_err < 1e-10,
            note: format!("max_identity_err={max_err:.2e}"),
        });
    }

    // det isomorphism: det(A) * det(inv(A)) ≈ 1
    {
        let a = make_diag_dominant(16);
        let d1 = det(&a, RuntimeMode::Strict, true).unwrap();
        let inv_a = inv(&a, InvOptions::default()).unwrap();
        let d2 = det(&inv_a.inverse, RuntimeMode::Strict, true).unwrap();
        let product = d1 * d2;
        iso_details.push(IsomorphismDetail {
            operation: "det".into(),
            passes: (product - 1.0).abs() < 1e-6,
            note: format!("det(A)*det(inv(A))={product:.6e}"),
        });
    }

    // lstsq isomorphism
    {
        let a = make_overdetermined(32, 16);
        let b = make_rhs(32);
        let r = lstsq(&a, &b, LstsqOptions::default()).unwrap();
        // Check A^T A x ≈ A^T b (normal equations)
        let mut max_err = 0.0_f64;
        for j in 0..16 {
            let mut atax_j = 0.0_f64;
            let mut atb_j = 0.0_f64;
            for i in 0..32 {
                atb_j += a[i][j] * b[i];
                for k in 0..16 {
                    atax_j += a[i][j] * a[i][k] * r.x[k];
                }
            }
            max_err = max_err.max((atax_j - atb_j).abs());
        }
        iso_details.push(IsomorphismDetail {
            operation: "lstsq".into(),
            passes: max_err < 1e-6,
            note: format!("normal_eq_err={max_err:.2e}"),
        });
    }

    // pinv isomorphism: pinv(A) @ b ≈ lstsq(A, b)
    {
        let a = make_overdetermined(32, 16);
        let r = pinv(&a, PinvOptions::default()).unwrap();
        // pseudo_inverse is 16×32 (n rows, m cols)
        let b = make_rhs(32);
        let lstsq_r = lstsq(&a, &b, LstsqOptions::default()).unwrap();
        let pinv_x: Vec<f64> = r
            .pseudo_inverse
            .iter()
            .map(|row| row.iter().zip(&b).map(|(&p, &bi)| p * bi).sum())
            .collect();
        let diff = pinv_x
            .iter()
            .zip(&lstsq_r.x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, |a: f64, b: f64| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        iso_details.push(IsomorphismDetail {
            operation: "pinv".into(),
            passes: diff < 1e-8,
            note: format!("pinv_vs_lstsq_diff={diff:.2e}"),
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
    let artifact_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/P2C-002/perf");
    std::fs::create_dir_all(&artifact_dir).unwrap();

    let json = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write(artifact_dir.join("perf_profile_report.json"), &json).unwrap();

    // Assertions
    for d in &report.isomorphism_check.details {
        if !d.passes {
            eprintln!("  FAIL: {} — {}", d.operation, d.note);
        } else {
            eprintln!("  PASS: {} — {}", d.operation, d.note);
        }
    }
    assert!(all_pass, "Behavior-isomorphism check failed");

    // Print top-3 hotspots for human review
    eprintln!("\n── Top-3 Hotspots (at 256-col size) ──");
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

/// Minimal timestamp without pulling in chrono.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}
