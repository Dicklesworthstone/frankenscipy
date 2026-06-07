//! Profiling-only harness for the dense linear-solve hot path.
//!
//! NOT a product binary — it exists so `samply`/`perf`/`hyperfine` have a
//! tight, deterministic scenario to attach to (see
//! `tests/artifacts/perf/`). Build under the `release-perf` profile:
//!
//! ```bash
//! RUSTFLAGS="-C force-frame-pointers=yes" \
//!   cargo build -p fsci-linalg --profile release-perf --bin perf_solve
//! ```
//!
//! Usage: `perf_solve <mode> <n> <repeats> [seed]`
//!   mode    = solve | lu_factor | lu_solve | lu_solve_cached | solve_triangular
//!             | backward_error_probe
//!   n       = matrix dimension
//!   repeats = number of timed iterations
//!
//! Prints one JSON line with per-call timing so it doubles as a wall-clock
//! probe. Matrix is deterministic (diagonally dominant => well-conditioned,
//! so CASP takes the DirectLU fast path and we profile the kernel, not the
//! fallback ladder).

use std::hint::black_box;
use std::time::Instant;

use fsci_linalg::{
    DecompOptions, SolveOptions, TriangularSolveOptions, lu_factor, lu_solve, solve,
    solve_toeplitz, solve_triangular, toeplitz,
};
use nalgebra::{DMatrix, DVector};

/// Deterministic diagonally-dominant matrix; no RNG crate so the workload is
/// byte-for-byte reproducible across runs/hosts.
fn make_matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        // SplitMix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) as f64) / (u64::MAX as f64) - 0.5
    };
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        let mut rowsum = 0.0;
        for (j, cell) in row.iter_mut().enumerate() {
            let v = next();
            *cell = v;
            if i != j {
                rowsum += v.abs();
            }
        }
        // Force diagonal dominance for a well-conditioned system.
        row[i] = rowsum + 1.0;
    }
    a
}

fn make_rhs(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0xDEAD_BEEF;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        })
        .collect()
}

/// Deterministic, strongly diagonally-dominant (well-conditioned) non-symmetric
/// Toeplitz system: c[0] dominates so every leading principal minor is
/// non-singular (Levinson is stable). r[0] is ignored by both paths.
fn make_toeplitz(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let c: Vec<f64> = (0..n)
        .map(|k| if k == 0 { 4.0 } else { 0.5f64.powi(k as i32) })
        .collect();
    let r: Vec<f64> = (0..n)
        .map(|k| if k == 0 { 0.0 } else { 0.3f64.powi(k as i32) })
        .collect();
    let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
    (c, r, b)
}

fn make_upper_triangular(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate().skip(i) {
            *cell = if i == j {
                (n as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

fn make_linear_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

fn dmatrix_from_rows_for_probe(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let row_count = rows.len();
    let column_count = rows.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(row_count * column_count);
    for row in rows {
        data.extend_from_slice(row);
    }
    DMatrix::from_row_slice(row_count, column_count, &data)
}

fn compute_backward_error_probe(
    matrix: &DMatrix<f64>,
    x: &DVector<f64>,
    rhs: &DVector<f64>,
) -> f64 {
    let residual = matrix * x - rhs;
    let residual_norm = residual.norm();
    let denom = matrix.norm() * x.norm() + rhs.norm();
    if !residual_norm.is_finite() || !denom.is_finite() {
        return f64::INFINITY;
    }
    if denom > 0.0 {
        residual_norm / denom
    } else {
        0.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("solve");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let repeats: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(42);

    // Bit-exact golden capture for isomorphism proofs. Prints the raw f64 bit
    // pattern of every solution element across a fixed set of (n, seed) pairs,
    // for both the non-transposed (borrowed) and transposed (owned) solve
    // paths. Any change that alters numerics flips the sha256 of this output.
    if mode == "golden" {
        for &gn in &[3usize, 8, 17, 32, 64, 100] {
            for &gseed in &[1u64, 42, 12345] {
                let ga = make_matrix(gn, gseed);
                let gb = make_rhs(gn, gseed);
                for transposed in [false, true] {
                    let opts = SolveOptions {
                        transposed,
                        ..SolveOptions::default()
                    };
                    let r = solve(&ga, &gb, opts).unwrap();
                    print!("n={gn} seed={gseed} t={} ", transposed as u8);
                    for v in &r.x {
                        print!("{:016x} ", v.to_bits());
                    }
                    println!("be={:016x}", r.backward_error.unwrap_or(0.0).to_bits());
                }
            }
        }
        return;
    }

    if mode == "backward_error_probe" {
        let a = make_matrix(n, seed);
        let b = make_rhs(n, seed);
        let solved = solve(&a, &b, SolveOptions::default()).unwrap();
        let matrix = dmatrix_from_rows_for_probe(&a);
        let x = DVector::from_column_slice(&solved.x);
        let rhs = DVector::from_column_slice(&b);
        let t0 = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..repeats {
            checksum +=
                compute_backward_error_probe(black_box(&matrix), black_box(&x), black_box(&rhs));
        }
        let elapsed = t0.elapsed();
        let per_call_ms = elapsed.as_secs_f64() * 1e3 / repeats as f64;
        println!(
            "{{\"mode\":\"{mode}\",\"n\":{n},\"repeats\":{repeats},\"total_ms\":{:.3},\"per_call_ms\":{:.4},\"checksum\":{:.6e}}}",
            elapsed.as_secs_f64() * 1e3,
            per_call_ms,
            checksum
        );
        return;
    }

    // Toeplitz A/B: `toeplitz_levinson` is the live O(n²) Levinson–Durbin
    // path; `toeplitz_dense` reconstructs the previous O(n³) behaviour
    // (materialise the dense n×n Toeplitz, then run a general LU solve). Same
    // binary, same workload — the ratio of per_call_ms is the algorithmic win.
    if mode == "toeplitz_levinson" || mode == "toeplitz_dense" {
        let (tc, tr, tb) = make_toeplitz(n);
        let t0 = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..repeats {
            let x = if mode == "toeplitz_levinson" {
                solve_toeplitz(black_box(&tc), Some(black_box(&tr)), black_box(&tb)).unwrap()
            } else {
                let matrix = toeplitz(black_box(&tc), Some(black_box(&tr)));
                solve(black_box(&matrix), black_box(&tb), SolveOptions::default())
                    .unwrap()
                    .x
            };
            checksum += x.iter().sum::<f64>();
        }
        let elapsed = t0.elapsed();
        let per_call_ms = elapsed.as_secs_f64() * 1e3 / repeats as f64;
        println!(
            "{{\"mode\":\"{mode}\",\"n\":{n},\"repeats\":{repeats},\"total_ms\":{:.3},\"per_call_ms\":{:.4},\"checksum\":{:.6e}}}",
            elapsed.as_secs_f64() * 1e3,
            per_call_ms,
            checksum
        );
        return;
    }

    let (a, b) = if mode == "solve_triangular" {
        (make_upper_triangular(n), make_linear_rhs(n))
    } else {
        (make_matrix(n, seed), make_rhs(n, seed))
    };

    let cached_lu = if mode == "lu_solve_cached" {
        Some(lu_factor(&a, DecompOptions::default()).unwrap())
    } else {
        None
    };

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        match mode {
            "solve" => {
                let r = solve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap();
                checksum += r.x.iter().sum::<f64>();
            }
            "lu_factor" => {
                // LuFactorResult fields are opaque (wraps nalgebra LU); just
                // keep the result live so the factorization isn't elided.
                let r = lu_factor(black_box(&a), DecompOptions::default()).unwrap();
                black_box(&r);
                checksum += 1.0;
            }
            "lu_solve" => {
                let f = lu_factor(&a, DecompOptions::default()).unwrap();
                let r = lu_solve(black_box(&f), black_box(&b)).unwrap();
                checksum += r.x.iter().sum::<f64>();
            }
            "lu_solve_cached" => {
                let f = cached_lu.as_ref().unwrap();
                let r = lu_solve(black_box(f), black_box(&b)).unwrap();
                checksum += r.x.iter().sum::<f64>();
            }
            "solve_triangular" => {
                let r = solve_triangular(
                    black_box(&a),
                    black_box(&b),
                    TriangularSolveOptions::default(),
                )
                .unwrap();
                checksum += r.x.iter().sum::<f64>() + r.backward_error.unwrap_or(0.0);
            }
            other => {
                eprintln!("unknown mode: {other}");
                std::process::exit(2);
            }
        }
    }
    let elapsed = t0.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1e3 / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"repeats\":{repeats},\"total_ms\":{:.3},\"per_call_ms\":{:.4},\"checksum\":{:.6e}}}",
        elapsed.as_secs_f64() * 1e3,
        per_call_ms,
        checksum
    );
}
