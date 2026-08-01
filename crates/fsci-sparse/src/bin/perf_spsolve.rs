//! Same-process A/B + parity harness for spsolve routing.
//!
//! Before: spsolve densified any sparse A (n<=32768) into an n×n dense matrix and
//! ran O(n³) nalgebra dense LU. After: genuinely-sparse A routes to the native
//! sparse LU (~O(n·fill)). On a diagonally-dominant pentadiagonal system the fill
//! is O(n), so the sparse path is orders of magnitude cheaper. The solution is
//! unique, so x matches the dense path to rounding (PARITY block prints max|Δx|).
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_spsolve`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, LuOptions, PermutationOrdering, Shape2D,
    SolveOptions, splu, splu_solve, spsolve,
};
use nalgebra::{DMatrix, DVector};

// Pentadiagonal whose row/col labels are scrambled by a fixed pseudo-random symmetric
// permutation: same nnz (~5/row) but huge bandwidth in natural order, so natural-order
// sparse LU fills toward dense — while a fill-reducing reorder (RCM) recovers the band.
fn scattered_pentadiagonal(n: usize, seed: u64) -> CsrMatrix {
    // Fisher-Yates shuffle of 0..n with an LCG.
    let mut q: Vec<usize> = (0..n).collect();
    let mut s = seed;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (s >> 11) as usize % (i + 1);
        q.swap(i, j);
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(q[i]);
                cols.push(q[j as usize]);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

fn opts_with(ordering: PermutationOrdering) -> SolveOptions {
    SolveOptions {
        ordering,
        ..SolveOptions::default()
    }
}

// Diagonally-dominant banded matrix, half-bandwidth `hb` (2·hb+1 nnz/row). For hb>8 this
// exceeds the old nnz<=16n gate and used to densify to O(n³); the bandwidth gate now
// routes it to the sparse LU (fill bounded by the band).
fn banded(n: usize, hb: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(hb);
        let hi = (i + hb).min(n - 1);
        for j in lo..=hi {
            rows.push(i);
            cols.push(j);
            data.push(if i == j { 2.0 * hb as f64 + 2.0 } else { -1.0 });
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// 2D 5-point Laplacian on a k×k grid (n=k²): the canonical fill-reduction benchmark.
// RCM keeps bandwidth ~k -> fill O(n·k)=O(n^1.5); minimum-degree/nested-dissection
// achieve O(n log n) fill. Diagonally dominant (diag 4+eps, neighbors -1) -> stable.
fn laplacian_2d(k: usize) -> CsrMatrix {
    let n = k * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |r: usize, c: usize| r * k + c;
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                if nr >= 0 && nr < k as i64 && nc >= 0 && nc < k as i64 {
                    rows.push(i);
                    cols.push(idx(nr as usize, nc as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Rectangular counterpart used by the profile-first widening campaign. The
// insertion order matches the square fixture above and produces canonical CSR.
fn laplacian_2d_rectangular(rows_count: usize, cols_count: usize) -> CsrMatrix {
    let n = rows_count * cols_count;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |row: usize, col: usize| row * cols_count + col;
    for row in 0..rows_count {
        for col in 0..cols_count {
            let i = idx(row, col);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (row_delta, col_delta) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let neighbor_row = row as i64 + row_delta;
                let neighbor_col = col as i64 + col_delta;
                if neighbor_row >= 0
                    && neighbor_row < rows_count as i64
                    && neighbor_col >= 0
                    && neighbor_col < cols_count as i64
                {
                    rows.push(i);
                    cols.push(idx(neighbor_row as usize, neighbor_col as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Arrowhead: diagonal + a dense hub row/col through node 0. nnz ~= 3n. Eliminating the
// hub early (natural/RCM, which can't isolate it) fills the whole trailing block O(n²);
// minimum-degree eliminates the degree-1 spokes first (no fill) and the hub last (no
// fill) -> O(n). The showcase where min-degree crushes bandwidth ordering.
fn arrowhead(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(n as f64 + 4.0); // strong diagonal -> diagonally dominant, stable
        if i != 0 {
            rows.push(0);
            cols.push(i);
            data.push(-1.0);
            rows.push(i);
            cols.push(0);
            data.push(-1.0);
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Diagonally-dominant pentadiagonal A (bandwidth 2): diag 6, off-diagonals -1 at
// ±1, ±2. nnz/row ~= 5, so a.nnz() <= 16n -> routes to the native sparse LU.
fn pentadiagonal(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(i);
                cols.push(j as usize);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Verbatim of the OLD dense path: densify the CSR and solve with nalgebra LU.
fn dense_solve_baseline(a: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let mut dense = vec![0.0f64; n * n];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * n + indices[idx]] = data[idx];
        }
    }
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let x = matrix.lu().solve(&rhs).expect("dense lu");
    x.iter().copied().collect()
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn profile_rectangular_rust(repetitions: usize) {
    let rows = 32usize;
    let cols = 128usize;
    let n = rows * cols;
    let matrix = laplacian_2d_rectangular(rows, cols);
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
    let warm = spsolve(&matrix, &rhs, SolveOptions::default()).expect("rectangular warmup");
    let mut checksum = warm.solution.iter().sum::<f64>();
    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("rectangular profile solve");
        checksum += black_box(solved.solution[n / 2]);
    }
    println!(
        "RECTANGULAR_PROFILE rows={rows} cols={cols} n={n} nnz={} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
    );
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    if arguments.next().as_deref() == Some("--profile-rectangular-rust") {
        let repetitions = arguments
            .next()
            .map(|value| value.parse::<usize>().expect("positive repetition count"))
            .unwrap_or(50);
        assert!(repetitions > 0, "repetition count must be positive");
        profile_rectangular_rust(repetitions);
        return;
    }

    // Wider-banded routing: matrices with >16 nnz/row but a narrow band now route to the
    // sparse LU (bandwidth gate) instead of densifying to an O(n³) dense LU.
    println!("--- wider-banded routing: dense(old) vs sparse(bandwidth gate) ---");
    for &(n, hb) in &[(1024usize, 16usize), (2048, 24), (3000, 30)] {
        let a = banded(n, hb);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(&x_dense)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps_s = (20_000_000 / (n + 1)).clamp(10, 3000);
        let t_sparse = time(reps_s, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_d = if n >= 2048 { 2 } else { 4 };
        let t_dense = time(reps_d, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });
        println!(
            "banded n={n:>5} hb={hb:>3} ({} nnz/row): dense={t_dense:>10.4}ms  sparse={t_sparse:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            2 * hb + 1,
            t_dense / t_sparse,
        );
    }

    println!("===PARITY+AB===");
    for &n in &[512usize, 1024, 2048] {
        let a = pentadiagonal(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        // correctness: sparse-routed result vs old dense result
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps_sparse = (50_000_000 / (n + 1)).clamp(20, 5000);
        let t_after = time(reps_sparse, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_dense = if n >= 2048 { 1 } else { 3 };
        let t_before = time(reps_dense, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });

        println!(
            "spsolve n={n:>5}: dense={t_before:>10.4}ms  sparse={t_after:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_before / t_after
        );

        // splu factorization: same routing. Time factorize-only (the dominant cost).
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let fac = splu(&a_csc, LuOptions::default()).expect("splu");
        let x_splu = splu_solve(&fac, &b).expect("splu_solve");
        let max_dx2 = x_splu
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let t_splu = time(reps_sparse, || {
            black_box(splu(black_box(&a_csc), LuOptions::default()).unwrap());
        });
        let t_dense_fac = time(reps_dense, || {
            let n = a.shape().rows;
            let mut dense = vec![0.0f64; n * n];
            let indptr = a.indptr();
            let indices = a.indices();
            let data = a.data();
            for i in 0..n {
                for idx in indptr[i]..indptr[i + 1] {
                    dense[i * n + indices[idx]] = data[idx];
                }
            }
            black_box(DMatrix::from_row_slice(n, n, &dense).lu());
        });
        println!(
            "splu    n={n:>5}: dense={t_dense_fac:>10.4}ms  sparse={t_splu:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx2:.2e}",
            t_dense_fac / t_splu
        );
    }

    // ── NEW LEVER: fill-reducing ordering on a SCATTERED sparse matrix ──
    // natural-order sparse LU fills toward dense; RCM (default Colamd→RCM) recovers
    // the band. Both routes solve the SAME unique system (parity to rounding).
    println!("--- fill-reducing ordering (scattered pentadiagonal) ---");
    for &n in &[300usize, 600, 1000] {
        let a = scattered_pentadiagonal(n, 0x1234 ^ n as u64);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_nat = spsolve(&a, &b, opts_with(PermutationOrdering::Natural))
            .expect("spsolve natural")
            .solution;
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let max_dx = x_nat
            .iter()
            .zip(x_rcm.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_nat = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Natural),
                )
                .unwrap(),
            );
        });
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        println!(
            "ordering n={n:>5}: natural={t_nat:>10.4}ms  rcm={t_rcm:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_nat / t_rcm
        );
    }

    // ── NEW LEVER: minimum-degree (MmdAtPlusA) vs RCM on a 2D Laplacian ──
    println!("--- minimum-degree vs RCM (2D 5-point Laplacian) ---");
    for &k in &[20usize, 32, 45, 64] {
        let a = laplacian_2d(k);
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(&x_mmd)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps = (8_000_000 / (n + 1)).clamp(3, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "lap2d k={k:>3} n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>7.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    // ── factor-once-solve-many: min-degree's smaller factor pays off per-solve ──
    println!("--- splu factor + 200 solves: RCM vs min-degree (2D Laplacian) ---");
    for &k in &[32usize, 45, 64] {
        let a = laplacian_2d(k);
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let m = 200usize;

        let lu_rcm = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::Colamd,
                ..LuOptions::default()
            },
        )
        .expect("rcm");
        let lu_mmd = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::MmdAtPlusA,
                ..LuOptions::default()
            },
        )
        .expect("mmd");
        let xr = splu_solve(&lu_rcm, &b).unwrap();
        let xm = splu_solve(&lu_mmd, &b).unwrap();
        let max_dx = xr
            .iter()
            .zip(&xm)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = 30;
        let t_rcm = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_rcm), black_box(&b)).unwrap());
            }
        });
        let t_mmd = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_mmd), black_box(&b)).unwrap());
            }
        });
        println!(
            "solve×{m} k={k:>3} n={n:>5}: rcm={t_rcm:>9.4}ms  mmd={t_mmd:>9.4}ms  per-solve speedup={:>6.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    println!("--- minimum-degree ordering (arrowhead) ---");
    for &n in &[300usize, 600, 1000] {
        let a = arrowhead(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("spsolve mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(x_mmd.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "arrowhd n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }
}
