use std::{
    hint::black_box,
    io::Write,
    process::{Command, Stdio},
    time::Duration,
};

use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "chol-wall-bench")]
use fsci_linalg::{
    CHOL_NB_OVERRIDE, CHOL_PANEL_INNER_OVERRIDE, CHOL_PANEL_TRSM_FORCE_SERIAL,
    CHOL_PANEL_TRSM_PAR_PANELS,
};
use fsci_linalg::{
    DecompOptions, EIGH_INVITER_FORCE_SERIAL, EXPM_ADAPTIVE_ORDER_DISABLE, HELMERT_FORCE_SERIAL,
    IS_DIAGONAL_FORCE_SERIAL, InvOptions, KHATRI_RAO_FORCE_SERIAL, LstsqOptions, MatrixAssumption,
    PASCAL_FORCE_SERIAL, PinvOptions, SolveOptions, TANHM_SHARED_PADE_DISABLE,
    TriangularSolveOptions, cho_factor, cho_solve, coshm, det, dft, eigh, expm, frobenius_norm,
    inv, is_diagonal, lstsq, lu_factor, lu_solve, mat_flatten, matmul, orthogonal_procrustes,
    pascal, pinv, randomized_eigh, solve, solve_banded, solve_triangular, svd, tanhm, vdot,
};
#[cfg(feature = "chol-wall-bench")]
use fsci_linalg::{
    cholesky_wall_bundle_candidate, cholesky_wall_mr4_nr4_candidate,
    cholesky_wall_mr4_nr8_fma_candidate, cholesky_wall_mr4_nr8_orig,
    cholesky_wall_trsm_blocked_fma_candidate,
};
use fsci_runtime::RuntimeMode;
use nalgebra::{DMatrix, DVector, Dyn, LU};
use std::sync::atomic::Ordering;

// Per SPEC §17, baseline sizes for dense solve family.
// SIZES: quick smoke tests; BASELINE_SIZES: full p50/p95/p99 capture
const SIZES: &[usize] = &[4, 16, 64, 256];
const BASELINE_SIZES: &[usize] = &[100, 500, 1000, 2000, 4000];
const MATMUL_SIZES: &[usize] = &[256, 512, 768, 1024];
const EIGH_SIZES: &[usize] = &[256, 512];
const RANDOMIZED_EIGH_CASES: &[(usize, usize)] = &[(256, 16), (512, 24)];
const DFT_GAUNTLET_SIZES: &[usize] = &[256, 512, 1024];
const CHO_FACTOR_GAUNTLET_SIZES: &[usize] = &[500, 1000];
const LU_FACTOR_GAUNTLET_SIZES: &[usize] = &[1000];
const ORTHOGONAL_PROCRUSTES_CASES: &[(usize, usize)] = &[(3000, 150), (5000, 200)];
const LU_FACTOR_SOLVE_CASES: &[usize] = &[1000, 1500];

/// Diagonally-dominant matrix: guaranteed non-singular, well-conditioned.
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

/// Upper-triangular matrix extracted from diag-dominant.
fn make_upper_triangular(n: usize) -> Vec<Vec<f64>> {
    let mut a = make_diag_dominant(n);
    for (i, row) in a.iter_mut().enumerate() {
        for cell in row.iter_mut().take(i) {
            *cell = 0.0;
        }
    }
    a
}

/// Tall rectangular matrix (rows > cols) for lstsq/pinv.
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

/// Wide full-row-rank matrix for minimum-norm lstsq/pinv routes.
fn make_underdetermined(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; cols]; rows];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                (rows as f64) * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

/// Tridiagonal banded matrix in LAPACK band storage format.
/// Returns (l_and_u, ab) where ab has shape (3, n).
fn make_tridiag_banded(n: usize) -> ((usize, usize), Vec<Vec<f64>>) {
    // LAPACK band storage: ab[nupper + i - j][j] = A[i][j]
    // For tridiag (l=1, u=1):
    //   ab[0][j] = A[j-1][j]  (superdiag, j >= 1)
    //   ab[1][j] = A[j][j]    (main diagonal)
    //   ab[2][j] = A[j+1][j]  (subdiag, j <= n-2)
    let mut ab = vec![vec![0.0; n]; 3];
    for col in &mut ab[1] {
        *col = 4.0;
    }
    for col in ab[0].iter_mut().skip(1) {
        *col = -1.0;
    }
    for col in ab[2].iter_mut().take(n.saturating_sub(1)) {
        *col = -1.0;
    }
    ((1, 1), ab)
}

fn make_rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i + 1) as f64).collect()
}

fn make_matmul_matrix(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| ((i * 31 + j * 17 + seed) % 97) as f64 * 0.01)
                .collect()
        })
        .collect()
}

fn frobenius_norm_scalar_reference(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|row| row.iter())
        .map(|&value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn vdot_scalar_reference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&left, &right)| left * right).sum()
}

fn mat_flatten_iterator_reference(a: &[Vec<f64>]) -> Vec<f64> {
    a.iter().flat_map(|row| row.iter().copied()).collect()
}

fn bench_mat_flatten_contiguous_rows(c: &mut Criterion) {
    let matrix = make_matmul_matrix(2048, 1024, 0x6f31);
    let candidate = mat_flatten(&matrix);
    let original = mat_flatten_iterator_reference(&matrix);
    assert_eq!(candidate, original);

    let mut group = c.benchmark_group("mat_flatten_contiguous_rows");
    group.bench_function("candidate/2048x1024", |bencher| {
        bencher.iter(|| black_box(mat_flatten(black_box(&matrix))))
    });
    group.bench_function("original/2048x1024", |bencher| {
        bencher.iter(|| black_box(mat_flatten_iterator_reference(black_box(&matrix))))
    });
    group.finish();
}

fn bench_frobenius_norm_simd(c: &mut Criterion) {
    let matrix = make_matmul_matrix(2048, 1024, 0x51ad);
    let mut group = c.benchmark_group("frobenius_norm_simd");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("scalar_reference_2048x1024", |bencher| {
        bencher.iter(|| black_box(frobenius_norm_scalar_reference(black_box(&matrix))))
    });
    group.bench_function("candidate_2048x1024", |bencher| {
        bencher.iter(|| black_box(frobenius_norm(black_box(&matrix))))
    });
    group.finish();
}

fn bench_vdot_simd_ab(c: &mut Criterion) {
    let len = 2_097_152usize;
    let left: Vec<f64> = (0..len)
        .map(|idx| ((idx % 251) as f64 - 125.0) / 31.0)
        .collect();
    let right: Vec<f64> = (0..len)
        .map(|idx| ((idx % 239) as f64 - 119.0) / 29.0)
        .collect();
    let mut group = c.benchmark_group("vdot_simd_ab");
    group.sample_size(15);
    group.bench_function("current_simd_n2097152", |bencher| {
        bencher.iter(|| black_box(vdot(black_box(&left), black_box(&right))))
    });
    group.bench_function("orig_scalar_n2097152", |bencher| {
        bencher.iter(|| black_box(vdot_scalar_reference(black_box(&left), black_box(&right))))
    });
    group.finish();
}

fn bench_is_diagonal(c: &mut Criterion) {
    let matrix = vec![vec![0.0; 5_000]; 5_000];
    IS_DIAGONAL_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let reference = is_diagonal(&matrix, 0.0);
    IS_DIAGONAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
    assert_eq!(is_diagonal(&matrix, 0.0), reference);

    let mut group = c.benchmark_group("is_diagonal");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    IS_DIAGONAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.bench_function("5000x5000", |bencher| {
        bencher.iter(|| black_box(is_diagonal(black_box(&matrix), black_box(0.0))))
    });
    IS_DIAGONAL_FORCE_SERIAL.store(true, Ordering::Relaxed);
    group.bench_function("orig_serial/5000x5000", |bencher| {
        bencher.iter(|| black_box(is_diagonal(black_box(&matrix), black_box(0.0))))
    });
    IS_DIAGONAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.bench_function("current_repeat/5000x5000", |bencher| {
        bencher.iter(|| black_box(is_diagonal(black_box(&matrix), black_box(0.0))))
    });
    IS_DIAGONAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_dmatrix_from_rows(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let m = rows.len();
    let n = rows.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in rows {
            data.push(row[col]);
        }
    }
    DMatrix::from_vec(m, n, data)
}

fn bench_matrix_norm1(matrix: &DMatrix<f64>) -> f64 {
    let mut max_col_sum = 0.0_f64;
    for col in 0..matrix.ncols() {
        let mut sum = 0.0_f64;
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)].abs();
            if !value.is_finite() {
                return f64::NAN;
            }
            sum += value;
        }
        max_col_sum = max_col_sum.max(sum);
    }
    max_col_sum
}

fn bench_solve_lu_transpose(
    lu: &LU<f64, Dyn, Dyn>,
    u_t: &DMatrix<f64>,
    l_t: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Option<DVector<f64>> {
    let y = u_t.solve_lower_triangular(b)?;
    let z = l_t.solve_upper_triangular(&y)?;
    let mut x = z;
    lu.p().inv_permute_rows(&mut x);
    Some(x)
}

fn bench_fast_rcond_from_lu(lu: &LU<f64, Dyn, Dyn>, a_norm: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if a_norm == 0.0 || !a_norm.is_finite() {
        return 0.0;
    }

    let mut x = DVector::from_element(n, 1.0 / (n as f64));
    let mut inv_a_norm = 0.0;
    let u_t = lu.u().transpose();
    let l_t = lu.l().transpose();

    for _ in 0..5 {
        let sign_x = x.map(|value| if value >= 0.0 { 1.0 } else { -1.0 });
        let Some(w) = bench_solve_lu_transpose(lu, &u_t, &l_t, &sign_x) else {
            return 0.0;
        };
        let sign_w = w.map(|value| if value >= 0.0 { 1.0 } else { -1.0 });
        let Some(x_new) = lu.solve(&sign_w) else {
            return 0.0;
        };

        let new_norm = x_new.lp_norm(1);
        let direction_delta = x_new
            .iter()
            .zip(x.iter())
            .map(|(&new, &old)| (new - old).abs())
            .sum::<f64>();
        if (new_norm - inv_a_norm).abs() <= 1e-10 * new_norm {
            inv_a_norm = new_norm;
            break;
        }
        inv_a_norm = new_norm;
        x = x_new;

        if direction_delta <= f64::EPSILON * new_norm {
            break;
        }
    }

    if inv_a_norm <= 0.0 {
        return 0.0;
    }
    let rcond = 1.0 / (a_norm * inv_a_norm);
    if rcond.is_nan() { 0.0 } else { rcond.min(1.0) }
}

fn lu_factor_solve_original_nalgebra(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let matrix = bench_dmatrix_from_rows(a);
    let a_norm = bench_matrix_norm1(&matrix);
    let lu = matrix.lu();
    black_box(bench_fast_rcond_from_lu(&lu, a_norm, a.len()));
    let rhs = DVector::from_column_slice(b);
    lu.solve(&rhs)
        .expect("original nalgebra lu solve")
        .iter()
        .copied()
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn make_symmetric_eigh_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j {
                (n as f64) * 3.0 + (i as f64) * 0.01
            } else {
                1.0 / ((i - j + 1) as f64)
            };
            a[i][j] = value;
            a[j][i] = value;
        }
    }
    a
}

#[allow(clippy::needless_range_loop)]
fn make_low_rank_symmetric_eigh_matrix(n: usize, rank: usize) -> Vec<Vec<f64>> {
    let mut factors = vec![vec![0.0; n]; rank];
    for (r, row) in factors.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let x = ((r + 3) * (i + 5)) as f64;
            *value = (x.sin() * 0.5) + (x.cos() * 0.25);
        }
    }

    let mut a = vec![vec![0.0; n]; n];
    for r in 0..rank {
        let weight = (rank - r) as f64;
        for i in 0..n {
            let scaled = weight * factors[r][i];
            for j in 0..=i {
                a[i][j] += scaled * factors[r][j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            a[j][i] = a[i][j];
        }
    }
    a
}

// ── solve ──────────────────────────────────────────────────────────────────────

fn bench_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── solve_triangular ───────────────────────────────────────────────────────────

fn bench_solve_triangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_triangular");
    for &n in SIZES {
        let a = make_upper_triangular(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve_triangular(&a, &b, TriangularSolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── solve_banded ───────────────────────────────────────────────────────────────

fn bench_solve_banded(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_banded");
    for &n in SIZES {
        let (l_u, ab) = make_tridiag_banded(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| solve_banded(l_u, &ab, &b, SolveOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── inv ────────────────────────────────────────────────────────────────────────

fn bench_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| inv(&a, InvOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── det ────────────────────────────────────────────────────────────────────────

fn bench_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("det");
    for &n in SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| det(&a, RuntimeMode::Strict, true).unwrap());
        });
    }
    group.finish();
}

// ── lstsq ──────────────────────────────────────────────────────────────────────

fn bench_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstsq");
    for &n in SIZES {
        // Overdetermined: 2n rows x n cols
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── pinv ───────────────────────────────────────────────────────────────────────

fn bench_pinv(c: &mut Criterion) {
    let mut group = c.benchmark_group("pinv");
    for &n in SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
        });
    }
    group.finish();
}

// ── matmul ────────────────────────────────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in MATMUL_SIZES {
        let a = make_matmul_matrix(n, n, 0);
        let b = make_matmul_matrix(n, n, 11);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| matmul(std::hint::black_box(&a), std::hint::black_box(&b)).unwrap());
        });
    }
    group.finish();
}

// ── eigh ──────────────────────────────────────────────────────────────────────

/// Parallel vs serial inverse-iteration eigenvector sweep in native `eigh` (n≥512).
/// One scoped spawn over independent columns; BYTE-IDENTICAL (asserted before timing).
/// `EIGH_INVITER_FORCE_SERIAL` A/B.
fn bench_eigh_inviter_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigh_inviter_parallel_ab");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in &[512usize, 800, 1200] {
        let a = make_symmetric_eigh_matrix(n);
        EIGH_INVITER_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let par = eigh(&a, DecompOptions::default()).unwrap();
        EIGH_INVITER_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let ser = eigh(&a, DecompOptions::default()).unwrap();
        let identical = par
            .eigenvalues
            .iter()
            .zip(&ser.eigenvalues)
            .all(|(p, s)| p.to_bits() == s.to_bits());
        assert!(
            identical,
            "n{n}: parallel inviter not byte-identical to serial"
        );
        group.bench_function(format!("serial_n{n}"), |bencher| {
            bencher.iter(|| {
                EIGH_INVITER_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(eigh(black_box(&a), DecompOptions::default()).unwrap())
            });
        });
        group.bench_function(format!("parallel_n{n}"), |bencher| {
            bencher.iter(|| {
                EIGH_INVITER_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(eigh(black_box(&a), DecompOptions::default()).unwrap())
            });
        });
    }
    EIGH_INVITER_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_eigh_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigh_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &n in EIGH_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| eigh(std::hint::black_box(&a), DecompOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_randomized_eigh(c: &mut Criterion) {
    let mut group = c.benchmark_group("randomized_eigh");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for &(n, k) in RANDOMIZED_EIGH_CASES {
        let a = make_low_rank_symmetric_eigh_matrix(n, k + 8);
        group.bench_function(format!("{n}x{n}_k{k}"), |bencher| {
            bencher.iter(|| {
                randomized_eigh(
                    std::hint::black_box(&a),
                    std::hint::black_box(k),
                    8,
                    2,
                    0x5eed_1234,
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

// ══════════════════════════════════════════════════════════════════════════════
// BASELINE BENCHMARKS - Per SPEC §17, capture p50/p95/p99 at standard sizes
// Run with: cargo bench --bench linalg_bench -- baseline
// ══════════════════════════════════════════════════════════════════════════════

fn bench_baseline_solve(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("baseline_solve");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        // Mixed-precision (default) arm.
        group.bench_function(format!("{n}x{n}"), |bencher| {
            fsci_linalg::DISABLE_MIXED_LU.store(false, Relaxed);
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
        });
        // Exact-f64 arm on the SAME worker/binary — the A/B baseline for the mixed route.
        group.bench_function(format!("{n}x{n}_f64"), |bencher| {
            fsci_linalg::DISABLE_MIXED_LU.store(true, Relaxed);
            bencher.iter(|| solve(&a, &b, SolveOptions::default()).unwrap());
            fsci_linalg::DISABLE_MIXED_LU.store(false, Relaxed);
        });
    }
    group.finish();
}

fn bench_lu_factor_gauntlet(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("lu_factor_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &n in LU_FACTOR_GAUNTLET_SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}_flat_factor"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
            bencher.iter(|| {
                black_box(lu_factor(black_box(&a), black_box(DecompOptions::default())).unwrap())
            });
        });
        group.bench_function(format!("{n}x{n}_orig_nalgebra_factor"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(true, Relaxed);
            bencher.iter(|| {
                black_box(lu_factor(black_box(&a), black_box(DecompOptions::default())).unwrap())
            });
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
        });
    }
    group.finish();
}

/// Well-conditioned, finite-determinant matrix (diagonal 1.5): its `∏ pivots` stays
/// under `f64::MAX`, so `det`'s flat blocked-LU path is exercised rather than falling
/// back to nalgebra on the overflow guard (which `make_diag_dominant`'s huge diagonal
/// would trip).
fn make_det_finite(n: usize) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i == j {
                1.5
            } else {
                0.4 / ((i as f64 - j as f64).abs() + 1.0)
            };
        }
    }
    a
}

/// Same-binary A/B for `det`: the flat blocked-LU determinant vs the original nalgebra
/// `.lu().determinant()`, toggled through `DISABLE_FLAT_LU_FACTOR`.
fn bench_det_gauntlet(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("det_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &n in LU_FACTOR_GAUNTLET_SIZES {
        let a = make_det_finite(n);
        group.bench_function(format!("{n}x{n}_flat_det"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
            bencher.iter(|| {
                black_box(det(black_box(&a), RuntimeMode::Strict, black_box(true)).unwrap())
            });
        });
        group.bench_function(format!("{n}x{n}_orig_nalgebra_det"), |bencher| {
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(true, Relaxed);
            bencher.iter(|| {
                black_box(det(black_box(&a), RuntimeMode::Strict, black_box(true)).unwrap())
            });
            fsci_linalg::DISABLE_FLAT_LU_FACTOR.store(false, Relaxed);
        });
    }
    group.finish();
}

fn bench_baseline_solve_pos(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_solve_pos");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| {
                solve(
                    std::hint::black_box(&a),
                    std::hint::black_box(&b),
                    SolveOptions {
                        assume_a: Some(MatrixAssumption::PositiveDefinite),
                        ..SolveOptions::default()
                    },
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_baseline_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_inv");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let a = make_diag_dominant(n);
        group.bench_function(format!("{n}x{n}"), |bencher| {
            bencher.iter(|| inv(&a, InvOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_lu_factor_solve_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_factor_solve_gauntlet");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));
    for &n in LU_FACTOR_SOLVE_CASES {
        let a = make_diag_dominant(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}_original_nalgebra"), |bencher| {
            bencher.iter(|| lu_factor_solve_original_nalgebra(black_box(&a), black_box(&b)));
        });
        group.bench_function(format!("{n}x{n}_current_public"), |bencher| {
            bencher.iter(|| {
                let factor = lu_factor(black_box(&a), DecompOptions::default()).expect("lu_factor");
                lu_solve(black_box(&factor), black_box(&b))
                    .expect("lu_solve")
                    .x
            });
        });
    }
    group.finish();
}

fn bench_baseline_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_lstsq");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_u0ucw_wide_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("u0ucw_wide_lstsq");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &(rows, cols) in &[(500usize, 1000usize), (1000, 2000)] {
        let a = make_underdetermined(rows, cols);
        let b = make_rhs(rows);
        group.bench_function(format!("{rows}x{cols}_materialized_transpose"), |bencher| {
            bencher.iter(|| lstsq(&a, &b, LstsqOptions::default()).unwrap());
        });
    }
    group.finish();
}

fn bench_u0ucw_wide_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;

    let mut group = c.benchmark_group("u0ucw_wide_pinv");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &(rows, cols) in &[(500usize, 1000usize), (1000, 2000)] {
        let a = make_underdetermined(rows, cols);
        group.bench_function(
            format!("{rows}x{cols}_normal_equation_cholesky"),
            |bencher| {
                fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
                fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
                bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            },
        );
        group.bench_function(format!("{rows}x{cols}_eigen_rcond_gate"), |bencher| {
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(true, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{cols}_svd_fallback"), |bencher| {
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(true, Relaxed);
            fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        });
    }
    group.finish();
}

fn scipy_pinv_duration(rows: usize, cols: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

rows = int(sys.argv[1])
cols = int(sys.argv[2])
iters = int(sys.argv[3])
i = np.arange(rows, dtype=np.float64)[:, None]
j = np.arange(cols, dtype=np.float64)[None, :]
a = 1.0 / (np.abs(i - j) + 1.0)
d = np.arange(rows)
a[d, d] = rows * 2.0
la.pinv(a, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    result = la.pinv(a, check_finite=False)
    checksum += float(result[0, 0])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &rows.to_string(),
            &cols.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy pinv oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy oracle script");
    let output = child.wait_with_output().expect("wait for scipy oracle");
    if !output.status.success() {
        eprintln!(
            "scipy pinv oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy timing");
    let seconds: f64 = stdout.trim().parse().expect("parse scipy timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_pinv_available() -> bool {
    let mut child = match Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };
    let Some(stdin) = child.stdin.as_mut() else {
        return false;
    };
    if stdin.write_all(b"import scipy.linalg\n").is_err() {
        return false;
    }
    child.wait().is_ok_and(|status| status.success())
}

fn scipy_lstsq_duration(rows: usize, cols: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

rows = int(sys.argv[1])
cols = int(sys.argv[2])
iters = int(sys.argv[3])
i = np.arange(rows, dtype=np.float64)[:, None]
j = np.arange(cols, dtype=np.float64)[None, :]
a = 1.0 / (np.abs(i - j) + 1.0)
d = np.arange(rows)
a[d, d] = rows * 2.0
b = np.arange(1, rows + 1, dtype=np.float64)
la.lstsq(a, b, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    x, residuals, rank, s = la.lstsq(a, b, check_finite=False)
    checksum += float(x[0]) + float(rank)
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &rows.to_string(),
            &cols.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy lstsq oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy lstsq oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy lstsq oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy lstsq oracle");
    if !output.status.success() {
        eprintln!(
            "scipy lstsq oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy lstsq timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy lstsq timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_eigh_subset_duration(n: usize, k: usize, rank: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
k = int(sys.argv[2])
rank = int(sys.argv[3])
iters = int(sys.argv[4])
i = np.arange(n, dtype=np.float64)
factors = []
for r in range(rank):
    x = float(r + 3) * (i + 5.0)
    factors.append(np.sin(x) * 0.5 + np.cos(x) * 0.25)
a = np.zeros((n, n), dtype=np.float64)
for r, f in enumerate(factors):
    weight = float(rank - r)
    a += weight * np.outer(f, f)
lo = max(0, n - k)
la.eigh(a, subset_by_index=(lo, n - 1), check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    w, v = la.eigh(a, subset_by_index=(lo, n - 1), check_finite=False)
    checksum += float(w[-1]) + float(v[0, -1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args([
            "-",
            &n.to_string(),
            &k.to_string(),
            &rank.to_string(),
            &iters.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy eigh oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy eigh oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy eigh oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy eigh oracle");
    if !output.status.success() {
        eprintln!(
            "scipy eigh oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy eigh timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy eigh timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_dft_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
la.dft(n)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    m = la.dft(n)
    checksum += float(np.real(m[1, 1])) + float(np.imag(m[1, 1]))
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy dft oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy dft oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy dft oracle script");
    let output = child.wait_with_output().expect("wait for scipy dft oracle");
    if !output.status.success() {
        eprintln!(
            "scipy dft oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy dft timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy dft timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_cho_factor_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
i = np.arange(n, dtype=np.float64)
a = 1.0 / (np.abs(i[:, None] - i[None, :]) + 1.0)
a[np.arange(n), np.arange(n)] = n * 3.0 + i * 0.01
la.cho_factor(a, lower=True, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    c, lower = la.cho_factor(a, lower=True, check_finite=False)
    checksum += float(c[0, 0]) + float(c[-1, -1]) + float(lower)
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy cho_factor oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy cho_factor oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy cho_factor oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy cho_factor oracle");
    if !output.status.success() {
        eprintln!(
            "scipy cho_factor oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy cho_factor timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy cho_factor timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn scipy_cho_factor_solve_duration(n: usize, iters: u64) -> Option<Duration> {
    let script = r#"
import sys
import time
import numpy as np
import scipy.linalg as la

n = int(sys.argv[1])
iters = int(sys.argv[2])
i = np.arange(n, dtype=np.float64)
a = 1.0 / (np.abs(i[:, None] - i[None, :]) + 1.0)
a[np.arange(n), np.arange(n)] = n * 3.0 + i * 0.01
b = np.arange(1, n + 1, dtype=np.float64)
factor = la.cho_factor(a, lower=True, check_finite=False)
la.cho_solve(factor, b, check_finite=False)
start = time.perf_counter()
checksum = 0.0
for _ in range(iters):
    factor = la.cho_factor(a, lower=True, check_finite=False)
    x = la.cho_solve(factor, b, check_finite=False)
    checksum += float(x[0]) + float(x[-1])
elapsed = time.perf_counter() - start
if not np.isfinite(checksum):
    raise SystemExit("non-finite checksum")
print(f"{elapsed:.17f}")
"#;
    let mut child = Command::new("python3")
        .args(["-", &n.to_string(), &iters.to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn scipy cho_factor+cho_solve oracle");
    child
        .stdin
        .as_mut()
        .expect("open scipy cho_factor+cho_solve oracle stdin")
        .write_all(script.as_bytes())
        .expect("write scipy cho_factor+cho_solve oracle script");
    let output = child
        .wait_with_output()
        .expect("wait for scipy cho_factor+cho_solve oracle");
    if !output.status.success() {
        eprintln!(
            "scipy cho_factor+cho_solve oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8(output.stdout).expect("utf8 scipy cho_factor+cho_solve timing");
    let seconds: f64 = stdout
        .trim()
        .parse()
        .expect("parse scipy cho_factor+cho_solve timing seconds");
    Some(Duration::from_secs_f64(seconds))
}

fn bench_u0ucw_gauntlet_scipy_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;

    const ROWS: usize = 500;
    const COLS: usize = 1000;
    let a = make_underdetermined(ROWS, COLS);
    let mut group = c.benchmark_group("u0ucw_gauntlet_scipy_pinv");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("500x1000_rust_current_diag_gate", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
    });
    group.bench_function("500x1000_rust_eigen_rcond_gate", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(true, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
    });
    group.bench_function("500x1000_rust_svd_fallback", |bencher| {
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(true, Relaxed);
        fsci_linalg::DISABLE_WIDE_PINV_DIAG_RCOND_GATE.store(false, Relaxed);
        bencher.iter(|| black_box(pinv(black_box(&a), PinvOptions::default()).unwrap()));
        fsci_linalg::DISABLE_WIDE_PINV_CHOLESKY.store(false, Relaxed);
    });
    if scipy_pinv_available() {
        group.bench_function("500x1000_scipy_pinv", |bencher| {
            bencher.iter_custom(|iters| {
                scipy_pinv_duration(ROWS, COLS, iters)
                    .expect("scipy pinv oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping 500x1000_scipy_pinv: python3 cannot import scipy.linalg");
    }
    group.finish();
}

fn bench_u0ucw_gauntlet_scipy_lstsq(c: &mut Criterion) {
    const ROWS: usize = 500;
    const COLS: usize = 1000;
    let a = make_underdetermined(ROWS, COLS);
    let b = make_rhs(ROWS);
    let mut group = c.benchmark_group("u0ucw_gauntlet_scipy_lstsq");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("500x1000_rust_current_materialized_transpose", |bencher| {
        bencher.iter(|| {
            black_box(lstsq(black_box(&a), black_box(&b), LstsqOptions::default())).unwrap()
        });
    });
    if scipy_pinv_available() {
        group.bench_function("500x1000_scipy_lstsq", |bencher| {
            bencher.iter_custom(|iters| {
                scipy_lstsq_duration(ROWS, COLS, iters)
                    .expect("scipy lstsq oracle should run after availability check")
            });
        });
    } else {
        eprintln!("skipping 500x1000_scipy_lstsq: python3 cannot import scipy.linalg");
    }
    group.finish();
}

fn bench_randomized_eigh_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("randomized_eigh_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for &(n, k) in RANDOMIZED_EIGH_CASES {
        let rank = k + 8;
        let a = make_low_rank_symmetric_eigh_matrix(n, rank);
        group.bench_function(format!("{n}x{n}_k{k}_rust_randomized"), |bencher| {
            bencher.iter(|| {
                black_box(
                    randomized_eigh(
                        black_box(&a),
                        black_box(k),
                        black_box(8),
                        black_box(2),
                        black_box(0x5eed_1234),
                    )
                    .unwrap(),
                )
            });
        });
        group.bench_function(format!("{n}x{n}_k{k}_rust_full_eigh"), |bencher| {
            bencher.iter(|| black_box(eigh(black_box(&a), DecompOptions::default()).unwrap()));
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}x{n}_k{k}_scipy_subset_eigh"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_eigh_subset_duration(n, k, rank, iters)
                        .expect("scipy eigh oracle should run after availability check")
                });
            });
        } else {
            eprintln!(
                "skipping {n}x{n}_k{k}_scipy_subset_eigh: python3 cannot import scipy.linalg"
            );
        }
    }
    group.finish();
}

fn bench_dft_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("dft_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for &n in DFT_GAUNTLET_SIZES {
        group.bench_function(format!("{n}_rust_dft"), |bencher| {
            bencher.iter(|| black_box(dft(black_box(n), black_box(Option::<&str>::None)).unwrap()));
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}_scipy_dft"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_dft_duration(n, iters)
                        .expect("scipy dft oracle should run after availability check")
                });
            });
        } else {
            eprintln!("skipping {n}_scipy_dft: python3 cannot import scipy.linalg");
        }
    }
    group.finish();
}

/// One-binary A/B for the AVX2+FMA MR4×NR8 trailing-SYRK micro-kernel:
/// production (MR4×NR4 mul+add) vs candidate (MR4×NR8 `mul_add`), interleaved per
/// factor inside one routine, with a paired production-vs-production A/A null pass,
/// a candidate-binary perf self-time child, and a differing-bits execution proof.
/// The candidate is 1e-10-tolerant vs production (FMA single-rounding), NOT bit-exact.
/// One-binary BUNDLE A/B: baseline = production (fresh scratch + dot in-panel) vs
/// candidate = hoisted scratch + inner=32 nested in-panel — the composite of two
/// individually sub-floor levers (~+3% and ~+2.5%), measured together per their
/// recorded retry predicates. 1e-10 contract + differing-bits execution proof.
#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_bundle_ab(c: &mut Criterion) {
    if std::env::var_os("FSCI_CHOL_BUNDLE_AB_SKIP").is_some() {
        return;
    }
    let summarize = |values: &[f64]| -> (f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        (
            mean,
            variance.sqrt() / mean * 100.0,
            sorted[sorted.len() / 2],
        )
    };

    for &(n, samples, factors_per_sample) in &[(1000usize, 20usize, 28usize), (2048, 12, 6)] {
        let a = make_symmetric_eigh_matrix(n);
        // Baseline arm pins the PRE-bundle behavior: fresh scratch (REUSE=false
        // straddler) + FORCED dot loop (override sentinel 1 — plain 0 now means
        // "size default", which is the nested path at n ≥ 1000 post-flip).
        let measure_baseline = || -> Duration {
            CHOL_PANEL_INNER_OVERRIDE.store(1, Ordering::Relaxed);
            let start = std::time::Instant::now();
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                    .expect("measured baseline factor"),
            );
            start.elapsed()
        };
        let measure_bundle = || -> Duration {
            CHOL_PANEL_INNER_OVERRIDE.store(32, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let out =
                black_box(cholesky_wall_bundle_candidate(black_box(&a))).expect("bundle factor");
            let elapsed = start.elapsed();
            black_box(out);
            elapsed
        };

        // Correctness: 1e-10 vs pre-bundle baseline + execution proof.
        CHOL_PANEL_INNER_OVERRIDE.store(1, Ordering::Relaxed);
        let baseline_factor =
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("baseline factor");
        CHOL_PANEL_INNER_OVERRIDE.store(32, Ordering::Relaxed);
        let bundle_factor = cholesky_wall_bundle_candidate(black_box(&a)).expect("bundle factor");
        CHOL_PANEL_INNER_OVERRIDE.store(0, Ordering::Relaxed);
        let mut differing = 0usize;
        let mut max_rel = 0.0f64;
        for (index, (&expected, &actual)) in baseline_factor.iter().zip(&bundle_factor).enumerate()
        {
            if actual.to_bits() != expected.to_bits() {
                differing += 1;
            }
            let rel = (actual - expected).abs() / expected.abs().max(1.0);
            max_rel = max_rel.max(rel);
            assert!(
                rel <= 1e-10,
                "bundle factor diverged past 1e-10 at flat index {index}, n={n}"
            );
        }
        assert!(
            differing > 0,
            "execution proof failed: bundle bit-identical to baseline at n={n} (inner path dead?)"
        );
        eprintln!(
            "FSCI_CHOL_BUNDLE_AB n={n} exec_proof differing_elements={differing} max_rel={max_rel:.3e}"
        );
        black_box(baseline_factor);
        black_box(bundle_factor);

        for _ in 0..4 {
            black_box(measure_baseline());
            black_box(measure_bundle());
        }
        let mut null_ratios = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut first_elapsed = Duration::ZERO;
            let mut second_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    first_elapsed += measure_baseline();
                    second_elapsed += measure_baseline();
                } else {
                    second_elapsed += measure_baseline();
                    first_elapsed += measure_baseline();
                }
            }
            null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
        }
        let mut null_sorted = null_ratios.clone();
        null_sorted.sort_by(f64::total_cmp);
        let null_summary = summarize(&null_ratios);
        eprintln!(
            "FSCI_CHOL_BUNDLE_AB n={n} NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
            null_summary.2,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1],
            null_summary.1
        );

        let mut base_ms = Vec::with_capacity(samples);
        let mut cand_ms = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut base_elapsed = Duration::ZERO;
            let mut cand_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    base_elapsed += measure_baseline();
                    cand_elapsed += measure_bundle();
                } else {
                    cand_elapsed += measure_bundle();
                    base_elapsed += measure_baseline();
                }
            }
            base_ms.push(base_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
            cand_ms.push(cand_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
        }
        let base = summarize(&base_ms);
        let cand = summarize(&cand_ms);
        let paired_ratios = base_ms
            .iter()
            .zip(&cand_ms)
            .map(|(before, after)| before / after)
            .collect::<Vec<_>>();
        let paired = summarize(&paired_ratios);
        let gflops = |ms: f64| (n as f64).powi(3) / 3.0 / (ms * 1e6);
        eprintln!(
            "FSCI_CHOL_BUNDLE_AB n={n} BASE mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} gflops_p50={:.2}",
            base.0,
            base.1,
            base.2,
            gflops(base.2)
        );
        eprintln!(
            "FSCI_CHOL_BUNDLE_AB n={n} CAND mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} gflops_p50={:.2}",
            cand.0,
            cand.1,
            cand.2,
            gflops(cand.2)
        );
        let decided = paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
        eprintln!(
            "FSCI_CHOL_BUNDLE_AB n={n} paired_median={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided} samples={samples} factors_per_sample={factors_per_sample}",
            paired.2,
            paired.1,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1]
        );
    }
    let _ = c;
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_bundle_ab(_c: &mut Criterion) {}

/// One-binary in-panel inner-block sweep (frankenscipy-dcx11): candidates {16, 32}
/// each paired against the unblocked dot loop (inner=0) via
/// `CHOL_PANEL_INNER_OVERRIDE`, with per-size A/A (0,0) nulls. 1e-10 factor
/// contract + per-arm differing-bits execution proofs.
#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_panel_inner_ab(c: &mut Criterion) {
    if std::env::var_os("FSCI_CHOL_PANEL_INNER_AB_SKIP").is_some() {
        return;
    }
    let summarize = |values: &[f64]| -> (f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        (
            mean,
            variance.sqrt() / mean * 100.0,
            sorted[sorted.len() / 2],
        )
    };

    for &(n, samples, factors_per_sample) in &[(1000usize, 16usize, 24usize), (2048, 10, 6)] {
        let a = make_symmetric_eigh_matrix(n);
        let measure_with_inner = |inner: usize| -> Duration {
            CHOL_PANEL_INNER_OVERRIDE.store(inner, Ordering::Relaxed);
            let start = std::time::Instant::now();
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("measured factor"),
            );
            start.elapsed()
        };

        for _ in 0..4 {
            black_box(measure_with_inner(1));
        }
        let mut null_ratios = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut first_elapsed = Duration::ZERO;
            let mut second_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    first_elapsed += measure_with_inner(1);
                    second_elapsed += measure_with_inner(1);
                } else {
                    second_elapsed += measure_with_inner(1);
                    first_elapsed += measure_with_inner(1);
                }
            }
            null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
        }
        let mut null_sorted = null_ratios.clone();
        null_sorted.sort_by(f64::total_cmp);
        let null_summary = summarize(&null_ratios);
        eprintln!(
            "FSCI_CHOL_PANEL_INNER_AB n={n} NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
            null_summary.2,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1],
            null_summary.1
        );

        CHOL_PANEL_INNER_OVERRIDE.store(1, Ordering::Relaxed);
        let baseline_factor =
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("baseline factor");
        for &inner in &[16usize, 32] {
            CHOL_PANEL_INNER_OVERRIDE.store(inner, Ordering::Relaxed);
            let candidate_factor =
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("candidate factor");
            let mut differing = 0usize;
            let mut max_rel = 0.0f64;
            for (index, (&expected, &actual)) in
                baseline_factor.iter().zip(&candidate_factor).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    differing += 1;
                }
                let rel = (actual - expected).abs() / expected.abs().max(1.0);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 1e-10,
                    "inner={inner} factor diverged past 1e-10 at flat index {index}"
                );
            }
            assert!(
                differing > 0,
                "execution proof failed: inner={inner} arm bit-identical to dot loop (override dead?)"
            );
            eprintln!(
                "FSCI_CHOL_PANEL_INNER_AB n={n} inner={inner} exec_proof differing_elements={differing} max_rel={max_rel:.3e}"
            );
            black_box(&candidate_factor);

            let mut base_ms = Vec::with_capacity(samples);
            let mut cand_ms = Vec::with_capacity(samples);
            for sample in 0..samples {
                let mut base_elapsed = Duration::ZERO;
                let mut cand_elapsed = Duration::ZERO;
                for factor in 0..factors_per_sample {
                    if (sample + factor) % 2 == 0 {
                        base_elapsed += measure_with_inner(1);
                        cand_elapsed += measure_with_inner(inner);
                    } else {
                        cand_elapsed += measure_with_inner(inner);
                        base_elapsed += measure_with_inner(1);
                    }
                }
                base_ms.push(base_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
                cand_ms.push(cand_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
            }
            let base = summarize(&base_ms);
            let cand = summarize(&cand_ms);
            let paired_ratios = base_ms
                .iter()
                .zip(&cand_ms)
                .map(|(before, after)| before / after)
                .collect::<Vec<_>>();
            let paired = summarize(&paired_ratios);
            let decided =
                paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
            eprintln!(
                "FSCI_CHOL_PANEL_INNER_AB n={n} inner={inner} base_p50_ms={:.6} cand_p50_ms={:.6} paired_median={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided}",
                base.2,
                cand.2,
                paired.2,
                paired.1,
                null_sorted[0],
                null_sorted[null_sorted.len() - 1]
            );
        }
        CHOL_PANEL_INNER_OVERRIDE.store(0, Ordering::Relaxed);
    }
    let _ = c;
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_panel_inner_ab(_c: &mut Criterion) {}

/// One-binary NB panel-width sweep: candidates {96, 160, 192} each paired against the
/// NB=128 default via `CHOL_NB_OVERRIDE`, with an A/A (128,128) null. Same symbols in
/// every arm (runtime const), so provenance = per-arm differing-bits execution proof
/// + recorded override value. 1e-10 factor contract; winner must clear the null range.
#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_nb_sweep(c: &mut Criterion) {
    if std::env::var_os("FSCI_CHOL_NB_SWEEP_SKIP").is_some() {
        return;
    }
    // Canonical staircase-evidence sweep (2026-07-22 verdicts in the artifact dir):
    // NB=192 regresses n ≤ 768, wins n=1000; NB=256 wins n ≥ 1536.
    let sweep: &[(usize, &[usize], usize, usize)] = &[
        (256, &[192], 12, 96),
        (384, &[192], 12, 64),
        (512, &[192], 12, 48),
        (768, &[160, 192], 14, 32),
        (1000, &[96, 160, 192, 224, 256], 16, 24),
        (1536, &[192, 256], 10, 10),
        (2048, &[192, 256], 10, 6),
    ];

    let summarize = |values: &[f64]| -> (f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        (
            mean,
            variance.sqrt() / mean * 100.0,
            sorted[sorted.len() / 2],
        )
    };

    for &(n, nb_list, samples, factors_per_sample) in sweep {
        let a = make_symmetric_eigh_matrix(n);
        let measure_with_nb = |nb: usize| -> Duration {
            CHOL_NB_OVERRIDE.store(nb, Ordering::Relaxed);
            let start = std::time::Instant::now();
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("measured factor"),
            );
            start.elapsed()
        };

        // A/A null at NB=128.
        for _ in 0..4 {
            black_box(measure_with_nb(128));
        }
        let mut null_ratios = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut first_elapsed = Duration::ZERO;
            let mut second_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    first_elapsed += measure_with_nb(128);
                    second_elapsed += measure_with_nb(128);
                } else {
                    second_elapsed += measure_with_nb(128);
                    first_elapsed += measure_with_nb(128);
                }
            }
            null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
        }
        let mut null_sorted = null_ratios.clone();
        null_sorted.sort_by(f64::total_cmp);
        let null_summary = summarize(&null_ratios);
        eprintln!(
            "FSCI_CHOL_NB_SWEEP n={n} NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
            null_summary.2,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1],
            null_summary.1
        );

        CHOL_NB_OVERRIDE.store(0, Ordering::Relaxed);
        let baseline_factor =
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("baseline factor");
        for &nb in nb_list {
            CHOL_NB_OVERRIDE.store(nb, Ordering::Relaxed);
            let candidate_factor =
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("candidate factor");
            let mut differing = 0usize;
            let mut max_rel = 0.0f64;
            for (index, (&expected, &actual)) in
                baseline_factor.iter().zip(&candidate_factor).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    differing += 1;
                }
                let rel = (actual - expected).abs() / expected.abs().max(1.0);
                max_rel = max_rel.max(rel);
                assert!(
                    rel <= 1e-10,
                    "NB={nb} factor diverged past 1e-10 at flat index {index}"
                );
            }
            assert!(
                differing > 0,
                "execution proof failed: NB={nb} arm bit-identical to NB=128 (override dead?)"
            );
            eprintln!(
                "FSCI_CHOL_NB_SWEEP n={n} nb={nb} exec_proof differing_elements={differing} max_rel={max_rel:.3e}"
            );
            black_box(&candidate_factor);

            let mut base_ms = Vec::with_capacity(samples);
            let mut cand_ms = Vec::with_capacity(samples);
            for sample in 0..samples {
                let mut base_elapsed = Duration::ZERO;
                let mut cand_elapsed = Duration::ZERO;
                for factor in 0..factors_per_sample {
                    if (sample + factor) % 2 == 0 {
                        base_elapsed += measure_with_nb(128);
                        cand_elapsed += measure_with_nb(nb);
                    } else {
                        cand_elapsed += measure_with_nb(nb);
                        base_elapsed += measure_with_nb(128);
                    }
                }
                base_ms.push(base_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
                cand_ms.push(cand_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
            }
            let base = summarize(&base_ms);
            let cand = summarize(&cand_ms);
            let paired_ratios = base_ms
                .iter()
                .zip(&cand_ms)
                .map(|(before, after)| before / after)
                .collect::<Vec<_>>();
            let paired = summarize(&paired_ratios);
            let decided =
                paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
            eprintln!(
                "FSCI_CHOL_NB_SWEEP n={n} nb={nb} base128_p50_ms={:.6} cand_p50_ms={:.6} paired_median={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided}",
                base.2,
                cand.2,
                paired.2,
                paired.1,
                null_sorted[0],
                null_sorted[null_sorted.len() - 1]
            );
        }
        CHOL_NB_OVERRIDE.store(0, Ordering::Relaxed);
    }
    let _ = c;
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_nb_sweep(_c: &mut Criterion) {}

/// One-binary A/B for the work-gated parallel blocked-FMA panel TRSM: both arms run
/// the SAME production factor path; the `CHOL_PANEL_TRSM_FORCE_SERIAL` static picks
/// serial (baseline) vs fanned (candidate) per call. BIT-IDENTICAL (4-row-aligned
/// chunks); execution proof via the `CHOL_PANEL_TRSM_PAR_PANELS` counter.
#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_trsm_par_ab(c: &mut Criterion) {
    const PROFILE_CHILD: &str = "FSCI_CHOL_TRSM_PAR_PROFILE_CHILD";
    if std::env::var_os("FSCI_CHOL_TRSM_PAR_AB_SKIP").is_some() {
        return;
    }

    if std::env::var_os(PROFILE_CHILD).is_some() {
        let a = make_symmetric_eigh_matrix(1000);
        CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
        for _ in 0..4 {
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                    .expect("warm parallel factor"),
            );
        }
        let start = std::time::Instant::now();
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..160 {
            let factor = cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                .expect("profiled parallel factor");
            for &value in factor.iter().step_by(4096) {
                digest ^= value.to_bits();
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(factor);
        }
        eprintln!(
            "FSCI_CHOL_TRSM_PAR_PROFILE factors=160 mean_ms={:.6} digest={digest:#018x}",
            start.elapsed().as_secs_f64() * 1000.0 / 160.0
        );
        std::process::exit(0);
    }

    let exe = std::env::current_exe().expect("current release-perf benchmark binary");
    let perf_path = format!(
        "/dev/shm/fsc-chol-trsm-par-{}-{}.perf",
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
        .args(["cholesky_wall_trsm_par_ab", "--noplot"])
        .env(PROFILE_CHILD, "1")
        .status()
        .expect("spawn trsm-par perf child");
    assert!(
        profile_status.success(),
        "trsm-par perf child failed: {profile_status}"
    );
    let report = Command::new("perf")
        .args([
            "report",
            "--stdio",
            "--no-children",
            "--percent-limit",
            "0.1",
            "--sort",
            "symbol",
            "-i",
            &perf_path,
        ])
        .output()
        .expect("run trsm-par perf report");
    assert!(
        report.status.success(),
        "trsm-par perf report failed: {}",
        String::from_utf8_lossy(&report.stderr)
    );
    let report = String::from_utf8(report.stdout).expect("utf8 trsm-par perf report");
    eprintln!("FSCI_CHOL_TRSM_PAR_REPORT_BEGIN\n{report}FSCI_CHOL_TRSM_PAR_REPORT_END");
    assert!(
        report.contains("cholesky_panel_trsm_blocked_fma_rows"),
        "chunk-level TRSM symbol must have non-zero profiled self-time"
    );

    let summarize = |values: &[f64]| -> (f64, f64, f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let cv_pct = variance.sqrt() / mean * 100.0;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |pct: usize| sorted[(sorted.len() - 1) * pct / 100];
        (mean, cv_pct, percentile(50), percentile(95), percentile(99))
    };

    // Under the size-adaptive NB staircase (2026-07-22), n=1000 runs nb=192 whose
    // first panels carry ~14.9M TRSM MACs — ABOVE the 8M gate — so both sizes now
    // expect the fan-out (the old zero-spawn expectation held only at nb=128).
    for &(n, samples, factors_per_sample, expect_spawn) in
        &[(1000usize, 20usize, 32usize, true), (2048, 12, 6, true)]
    {
        let a = make_symmetric_eigh_matrix(n);

        CHOL_PANEL_TRSM_FORCE_SERIAL.store(true, Ordering::Relaxed);
        let serial_before = CHOL_PANEL_TRSM_PAR_PANELS.load(Ordering::Relaxed);
        let serial_factor =
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("serial factor");
        assert_eq!(
            CHOL_PANEL_TRSM_PAR_PANELS.load(Ordering::Relaxed),
            serial_before,
            "forced-serial arm must not take the parallel branch"
        );
        CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let parallel_before = CHOL_PANEL_TRSM_PAR_PANELS.load(Ordering::Relaxed);
        let parallel_factor =
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("parallel factor");
        let spawned_panels = CHOL_PANEL_TRSM_PAR_PANELS.load(Ordering::Relaxed) - parallel_before;
        // Gate proof: at adaptive NB both sizes' largest TRSM panels exceed 8M MACs
        // (n=1000@nb=192: 14.9M; n=2048@nb=256: 31M+).
        if expect_spawn {
            assert!(
                spawned_panels > 0,
                "execution proof failed: no panel took the parallel branch at n={n}"
            );
        } else {
            assert_eq!(
                spawned_panels, 0,
                "gate proof failed: n={n} must stay fully serial under the 8M gate"
            );
        }
        for (index, (&expected, &actual)) in serial_factor.iter().zip(&parallel_factor).enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "parallel TRSM changed n={n} factor bits at flat index {index}"
            );
        }
        eprintln!("FSCI_CHOL_TRSM_PAR_AB n={n} bit_identical=true spawned_panels={spawned_panels}");
        black_box(serial_factor);
        black_box(parallel_factor);

        let measure_serial = || -> Duration {
            CHOL_PANEL_TRSM_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let start = std::time::Instant::now();
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                    .expect("measured serial factor"),
            );
            start.elapsed()
        };
        let measure_parallel = || -> Duration {
            CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let start = std::time::Instant::now();
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                    .expect("measured parallel factor"),
            );
            start.elapsed()
        };

        for candidate_first in [false, true] {
            if candidate_first {
                black_box(measure_parallel());
                black_box(measure_serial());
            } else {
                black_box(measure_serial());
                black_box(measure_parallel());
            }
        }

        let mut null_ratios = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut first_elapsed = Duration::ZERO;
            let mut second_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    first_elapsed += measure_serial();
                    second_elapsed += measure_serial();
                } else {
                    second_elapsed += measure_serial();
                    first_elapsed += measure_serial();
                }
            }
            null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
        }
        let mut null_sorted = null_ratios.clone();
        null_sorted.sort_by(f64::total_cmp);
        let null_summary = summarize(&null_ratios);
        eprintln!(
            "FSCI_CHOL_TRSM_PAR_AB n={n} NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
            null_summary.2,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1],
            null_summary.1
        );

        let mut serial_ms = Vec::with_capacity(samples);
        let mut parallel_ms = Vec::with_capacity(samples);
        for sample in 0..samples {
            let mut serial_elapsed = Duration::ZERO;
            let mut parallel_elapsed = Duration::ZERO;
            for factor in 0..factors_per_sample {
                if (sample + factor) % 2 == 0 {
                    serial_elapsed += measure_serial();
                    parallel_elapsed += measure_parallel();
                } else {
                    parallel_elapsed += measure_parallel();
                    serial_elapsed += measure_serial();
                }
            }
            serial_ms.push(serial_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
            parallel_ms.push(parallel_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
        }
        let prod = summarize(&serial_ms);
        let cand = summarize(&parallel_ms);
        let paired_ratios = serial_ms
            .iter()
            .zip(&parallel_ms)
            .map(|(before, after)| before / after)
            .collect::<Vec<_>>();
        let paired = summarize(&paired_ratios);
        let gflops = |ms: f64| (n as f64).powi(3) / 3.0 / (ms * 1e6);
        eprintln!(
            "FSCI_CHOL_TRSM_PAR_AB n={n} PROD mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
            prod.0,
            prod.1,
            prod.2,
            prod.3,
            prod.4,
            gflops(prod.2)
        );
        eprintln!(
            "FSCI_CHOL_TRSM_PAR_AB n={n} CAND mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
            cand.0,
            cand.1,
            cand.2,
            cand.3,
            cand.4,
            gflops(cand.2)
        );
        let decided = paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
        eprintln!(
            "FSCI_CHOL_TRSM_PAR_AB n={n} paired_median={:.6} paired_mean={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided} samples={samples} factors_per_sample={factors_per_sample}",
            paired.2,
            paired.0,
            paired.1,
            null_sorted[0],
            null_sorted[null_sorted.len() - 1]
        );
    }
    CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);

    let a = make_symmetric_eigh_matrix(1000);
    let mut group = c.benchmark_group("cholesky_wall_trsm_par_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let candidate_first = std::cell::Cell::new(false);
    group.bench_function("1000x1000_paired_alternating", |bencher| {
        bencher.iter(|| {
            let first = candidate_first.get();
            candidate_first.set(!first);
            if first {
                CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion parallel factor"),
                );
                CHOL_PANEL_TRSM_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion serial factor"),
                );
            } else {
                CHOL_PANEL_TRSM_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion serial factor"),
                );
                CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion parallel factor"),
                );
            }
        });
    });
    group.finish();
    CHOL_PANEL_TRSM_FORCE_SERIAL.store(false, Ordering::Relaxed);
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_trsm_par_ab(_c: &mut Criterion) {}

/// One-binary A/B for the blocked GEMM-shaped panel TRSM: production (rows2 dot TRSM +
/// FMA SYRK) vs candidate (blocked left-looking FMA TRSM + FMA SYRK), interleaved per
/// factor with a production/production A/A null pass, a candidate-binary perf self-time
/// child, and a differing-bits execution proof. 1e-10-tolerant vs production, NOT bit-exact.
#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_trsm_fma_ab(c: &mut Criterion) {
    const PROFILE_CHILD: &str = "FSCI_CHOL_TRSM_FMA_PROFILE_CHILD";
    if std::env::var_os("FSCI_CHOL_TRSM_FMA_AB_SKIP").is_some() {
        return;
    }
    let n = 1000usize;
    let a = make_symmetric_eigh_matrix(n);

    if std::env::var_os(PROFILE_CHILD).is_some() {
        for _ in 0..4 {
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("warm trsm factor"),
            );
        }
        let start = std::time::Instant::now();
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..160 {
            let factor = cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                .expect("profiled trsm factor");
            for &value in factor.iter().step_by(4096) {
                digest ^= value.to_bits();
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(factor);
        }
        eprintln!(
            "FSCI_CHOL_TRSM_FMA_PROFILE factors=160 mean_ms={:.6} digest={digest:#018x}",
            start.elapsed().as_secs_f64() * 1000.0 / 160.0
        );
        std::process::exit(0);
    }

    let exe = std::env::current_exe().expect("current release-perf benchmark binary");
    let perf_path = format!(
        "/dev/shm/fsc-chol-trsm-fma-{}-{}.perf",
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
        .args(["cholesky_wall_trsm_fma_ab", "--noplot"])
        .env(PROFILE_CHILD, "1")
        .status()
        .expect("spawn trsm perf child");
    assert!(
        profile_status.success(),
        "trsm perf child failed: {profile_status}"
    );
    let report = Command::new("perf")
        .args([
            "report",
            "--stdio",
            "--no-children",
            "--percent-limit",
            "0.1",
            "--sort",
            "symbol",
            "-i",
            &perf_path,
        ])
        .output()
        .expect("run trsm perf report");
    assert!(
        report.status.success(),
        "trsm perf report failed: {}",
        String::from_utf8_lossy(&report.stderr)
    );
    let report = String::from_utf8(report.stdout).expect("utf8 trsm perf report");
    eprintln!("FSCI_CHOL_TRSM_FMA_REPORT_BEGIN\n{report}FSCI_CHOL_TRSM_FMA_REPORT_END");
    assert!(
        report.contains("cholesky_panel_trsm_blocked_fma"),
        "candidate-specific blocked-TRSM symbol must have non-zero profiled self-time"
    );

    let production = cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("production factor");
    let candidate =
        cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("trsm candidate factor");
    let mut differing_elements = 0usize;
    let mut max_rel = 0.0f64;
    for (index, (&expected, &actual)) in production.iter().zip(&candidate).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            differing_elements += 1;
        }
        let rel = (actual - expected).abs() / expected.abs().max(1.0);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= 1e-10,
            "blocked TRSM factor diverged past 1e-10 at flat index {index}: {expected} vs {actual}"
        );
    }
    assert!(
        differing_elements > 0,
        "execution proof failed: blocked TRSM arm bit-identical to production (dead arm?)"
    );
    eprintln!(
        "FSCI_CHOL_TRSM_FMA_AB exec_proof differing_elements={differing_elements} of {} max_rel={max_rel:.3e}",
        production.len()
    );
    black_box(production);
    black_box(candidate);

    for candidate_first in [false, true, false, true] {
        if candidate_first {
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("warm trsm factor"),
            );
            black_box(
                cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("warm production factor"),
            );
        } else {
            black_box(
                cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("warm production factor"),
            );
            black_box(
                cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("warm trsm factor"),
            );
        }
    }

    let samples = 20usize;
    let factors_per_sample = 32usize;
    let measure_production = || -> Duration {
        let start = std::time::Instant::now();
        black_box(
            cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("measured production factor"),
        );
        start.elapsed()
    };
    let measure_trsm = || -> Duration {
        let start = std::time::Instant::now();
        black_box(
            cholesky_wall_trsm_blocked_fma_candidate(black_box(&a)).expect("measured trsm factor"),
        );
        start.elapsed()
    };

    let summarize = |values: &[f64]| -> (f64, f64, f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let cv_pct = variance.sqrt() / mean * 100.0;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |pct: usize| sorted[(sorted.len() - 1) * pct / 100];
        (mean, cv_pct, percentile(50), percentile(95), percentile(99))
    };

    let mut null_ratios = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut first_elapsed = Duration::ZERO;
        let mut second_elapsed = Duration::ZERO;
        for factor in 0..factors_per_sample {
            if (sample + factor) % 2 == 0 {
                first_elapsed += measure_production();
                second_elapsed += measure_production();
            } else {
                second_elapsed += measure_production();
                first_elapsed += measure_production();
            }
        }
        null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
    }
    let mut null_sorted = null_ratios.clone();
    null_sorted.sort_by(f64::total_cmp);
    let null_summary = summarize(&null_ratios);
    eprintln!(
        "FSCI_CHOL_TRSM_FMA_AB NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
        null_summary.2,
        null_sorted[0],
        null_sorted[null_sorted.len() - 1],
        null_summary.1
    );

    let mut production_ms = Vec::with_capacity(samples);
    let mut trsm_ms = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut production_elapsed = Duration::ZERO;
        let mut trsm_elapsed = Duration::ZERO;
        for factor in 0..factors_per_sample {
            if (sample + factor) % 2 == 0 {
                production_elapsed += measure_production();
                trsm_elapsed += measure_trsm();
            } else {
                trsm_elapsed += measure_trsm();
                production_elapsed += measure_production();
            }
        }
        production_ms.push(production_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
        trsm_ms.push(trsm_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
    }
    let prod = summarize(&production_ms);
    let cand = summarize(&trsm_ms);
    let paired_ratios = production_ms
        .iter()
        .zip(&trsm_ms)
        .map(|(before, after)| before / after)
        .collect::<Vec<_>>();
    let paired = summarize(&paired_ratios);
    let gflops = |ms: f64| (n as f64).powi(3) / 3.0 / (ms * 1e6);
    eprintln!(
        "FSCI_CHOL_TRSM_FMA_AB PROD mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
        prod.0,
        prod.1,
        prod.2,
        prod.3,
        prod.4,
        gflops(prod.2)
    );
    eprintln!(
        "FSCI_CHOL_TRSM_FMA_AB CAND mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
        cand.0,
        cand.1,
        cand.2,
        cand.3,
        cand.4,
        gflops(cand.2)
    );
    let decided = paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
    eprintln!(
        "FSCI_CHOL_TRSM_FMA_AB paired_median={:.6} paired_mean={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided} samples={samples} factors_per_sample={factors_per_sample}",
        paired.2,
        paired.0,
        paired.1,
        null_sorted[0],
        null_sorted[null_sorted.len() - 1]
    );

    let mut group = c.benchmark_group("cholesky_wall_trsm_fma_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let candidate_first = std::cell::Cell::new(false);
    group.bench_function("1000x1000_paired_alternating", |bencher| {
        bencher.iter(|| {
            let first = candidate_first.get();
            candidate_first.set(!first);
            if first {
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion trsm factor"),
                );
                black_box(
                    cholesky_wall_mr4_nr8_fma_candidate(black_box(&a))
                        .expect("criterion production factor"),
                );
            } else {
                black_box(
                    cholesky_wall_mr4_nr8_fma_candidate(black_box(&a))
                        .expect("criterion production factor"),
                );
                black_box(
                    cholesky_wall_trsm_blocked_fma_candidate(black_box(&a))
                        .expect("criterion trsm factor"),
                );
            }
        });
    });
    group.finish();
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_trsm_fma_ab(_c: &mut Criterion) {}

#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_nr8_fma_ab(c: &mut Criterion) {
    const PROFILE_CHILD: &str = "FSCI_CHOL_NR8_FMA_PROFILE_CHILD";
    if std::env::var_os("FSCI_CHOL_NR8_FMA_AB_SKIP").is_some() {
        return;
    }
    let n = 1000usize;
    let a = make_symmetric_eigh_matrix(n);

    if std::env::var_os(PROFILE_CHILD).is_some() {
        for _ in 0..4 {
            black_box(cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("warm fma factor"));
        }
        let start = std::time::Instant::now();
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..160 {
            let factor =
                cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("profiled fma factor");
            for &value in factor.iter().step_by(4096) {
                digest ^= value.to_bits();
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(factor);
        }
        eprintln!(
            "FSCI_CHOL_NR8_FMA_PROFILE factors=160 mean_ms={:.6} digest={digest:#018x}",
            start.elapsed().as_secs_f64() * 1000.0 / 160.0
        );
        std::process::exit(0);
    }

    let exe = std::env::current_exe().expect("current release-perf benchmark binary");
    let perf_path = format!(
        "/dev/shm/fsc-chol-nr8-fma-{}-{}.perf",
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
        .args(["cholesky_wall_nr8_fma_ab", "--noplot"])
        .env(PROFILE_CHILD, "1")
        .status()
        .expect("spawn fma perf child");
    assert!(
        profile_status.success(),
        "fma perf child failed: {profile_status}"
    );
    let report = Command::new("perf")
        .args([
            "report",
            "--stdio",
            "--no-children",
            "--percent-limit",
            "0.1",
            "--sort",
            "symbol",
            "-i",
            &perf_path,
        ])
        .output()
        .expect("run fma perf report");
    assert!(
        report.status.success(),
        "fma perf report failed: {}",
        String::from_utf8_lossy(&report.stderr)
    );
    let report = String::from_utf8(report.stdout).expect("utf8 fma perf report");
    eprintln!("FSCI_CHOL_NR8_FMA_REPORT_BEGIN\n{report}FSCI_CHOL_NR8_FMA_REPORT_END");
    assert!(
        report.contains("cholesky_syrk_flat_rows_mr4_nr8_fma"),
        "candidate-specific FMA SYRK symbol must have non-zero profiled self-time"
    );

    // Correctness (1e-10 factor-uniqueness contract) + execution proof: the FMA arm
    // must produce a factor that differs in SOME bits (else the switch never flipped)
    // while staying inside the blocked path's documented tolerance vs production.
    let production = cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("production factor");
    let candidate = cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("fma factor");
    let mut differing_elements = 0usize;
    let mut max_rel = 0.0f64;
    for (index, (&expected, &actual)) in production.iter().zip(&candidate).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            differing_elements += 1;
        }
        let rel = (actual - expected).abs() / expected.abs().max(1.0);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= 1e-10,
            "FMA factor diverged past 1e-10 at flat index {index}: {expected} vs {actual}"
        );
    }
    assert!(
        differing_elements > 0,
        "execution proof failed: FMA arm bit-identical to production (dead arm?)"
    );
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for &value in &candidate {
        for byte in value.to_bits().to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    eprintln!(
        "FSCI_CHOL_NR8_FMA_AB exec_proof differing_elements={differing_elements} of {} max_rel={max_rel:.3e} digest={digest:#018x}",
        production.len()
    );
    black_box(production);
    black_box(candidate);

    for candidate_first in [false, true, false, true] {
        if candidate_first {
            black_box(cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("warm fma factor"));
            black_box(
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("warm production factor"),
            );
        } else {
            black_box(
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("warm production factor"),
            );
            black_box(cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("warm fma factor"));
        }
    }

    let samples = 20usize;
    let factors_per_sample = 32usize;
    let measure_production = || -> Duration {
        let start = std::time::Instant::now();
        black_box(
            cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("measured production factor"),
        );
        start.elapsed()
    };
    let measure_fma = || -> Duration {
        let start = std::time::Instant::now();
        black_box(cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)).expect("measured fma factor"));
        start.elapsed()
    };

    let summarize = |values: &[f64]| -> (f64, f64, f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let cv_pct = variance.sqrt() / mean * 100.0;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |pct: usize| sorted[(sorted.len() - 1) * pct / 100];
        (mean, cv_pct, percentile(50), percentile(95), percentile(99))
    };

    // A/A NULL pass: production vs production, paired inside the same routine. The
    // candidate is DECIDABLE iff its paired-ratio median lies outside the null's
    // observed [min, max] (median null gate; cv reported, not gated).
    let mut null_ratios = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut first_elapsed = Duration::ZERO;
        let mut second_elapsed = Duration::ZERO;
        for factor in 0..factors_per_sample {
            if (sample + factor) % 2 == 0 {
                first_elapsed += measure_production();
                second_elapsed += measure_production();
            } else {
                second_elapsed += measure_production();
                first_elapsed += measure_production();
            }
        }
        null_ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
    }
    let mut null_sorted = null_ratios.clone();
    null_sorted.sort_by(f64::total_cmp);
    let null_summary = summarize(&null_ratios);
    eprintln!(
        "FSCI_CHOL_NR8_FMA_AB NULL median={:.6} min={:.6} max={:.6} cv_pct={:.3}",
        null_summary.2,
        null_sorted[0],
        null_sorted[null_sorted.len() - 1],
        null_summary.1
    );

    let mut production_ms = Vec::with_capacity(samples);
    let mut fma_ms = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut production_elapsed = Duration::ZERO;
        let mut fma_elapsed = Duration::ZERO;
        for factor in 0..factors_per_sample {
            if (sample + factor) % 2 == 0 {
                production_elapsed += measure_production();
                fma_elapsed += measure_fma();
            } else {
                fma_elapsed += measure_fma();
                production_elapsed += measure_production();
            }
        }
        production_ms.push(production_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
        fma_ms.push(fma_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
    }
    let prod = summarize(&production_ms);
    let cand = summarize(&fma_ms);
    let paired_ratios = production_ms
        .iter()
        .zip(&fma_ms)
        .map(|(before, after)| before / after)
        .collect::<Vec<_>>();
    let paired = summarize(&paired_ratios);
    let gflops = |ms: f64| (n as f64).powi(3) / 3.0 / (ms * 1e6);
    eprintln!(
        "FSCI_CHOL_NR8_FMA_AB PROD mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
        prod.0,
        prod.1,
        prod.2,
        prod.3,
        prod.4,
        gflops(prod.2)
    );
    eprintln!(
        "FSCI_CHOL_NR8_FMA_AB CAND mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} gflops_p50={:.2}",
        cand.0,
        cand.1,
        cand.2,
        cand.3,
        cand.4,
        gflops(cand.2)
    );
    let decided = paired.2 < null_sorted[0] || paired.2 > null_sorted[null_sorted.len() - 1];
    eprintln!(
        "FSCI_CHOL_NR8_FMA_AB paired_median={:.6} paired_mean={:.6} paired_cv_pct={:.3} null=[{:.6},{:.6}] DECIDED={decided} samples={samples} factors_per_sample={factors_per_sample}",
        paired.2,
        paired.0,
        paired.1,
        null_sorted[0],
        null_sorted[null_sorted.len() - 1]
    );

    let mut group = c.benchmark_group("cholesky_wall_nr8_fma_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let candidate_first = std::cell::Cell::new(false);
    group.bench_function("1000x1000_paired_alternating", |bencher| {
        bencher.iter(|| {
            let first = candidate_first.get();
            candidate_first.set(!first);
            if first {
                black_box(
                    cholesky_wall_mr4_nr8_fma_candidate(black_box(&a))
                        .expect("criterion fma factor"),
                );
                black_box(
                    cholesky_wall_mr4_nr4_candidate(black_box(&a))
                        .expect("criterion production factor"),
                );
            } else {
                black_box(
                    cholesky_wall_mr4_nr4_candidate(black_box(&a))
                        .expect("criterion production factor"),
                );
                black_box(
                    cholesky_wall_mr4_nr8_fma_candidate(black_box(&a))
                        .expect("criterion fma factor"),
                );
            }
        });
    });
    group.finish();
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_nr8_fma_ab(_c: &mut Criterion) {}

#[cfg(feature = "chol-wall-bench")]
fn bench_cholesky_wall_mr4_nr4_ab(c: &mut Criterion) {
    const PROFILE_CHILD: &str = "FSCI_CHOL_MR4_NR4_PROFILE_CHILD";
    // Allow the FMA A/B run to isolate itself from this preamble (it is expensive
    // and spawns its own perf child); the guard is opt-in via env and changes nothing
    // for existing invocations.
    if std::env::var_os("FSCI_CHOL_MR4_NR4_AB_SKIP").is_some() {
        return;
    }
    let n = 1000usize;
    let a = make_symmetric_eigh_matrix(n);

    if std::env::var_os(PROFILE_CHILD).is_some() {
        for _ in 0..4 {
            black_box(
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("warm candidate factor"),
            );
        }
        let start = std::time::Instant::now();
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..160 {
            let factor =
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("profiled candidate factor");
            for &value in factor.iter().step_by(4096) {
                digest ^= value.to_bits();
                digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
            }
            black_box(factor);
        }
        eprintln!(
            "FSCI_CHOL_MR4_NR4_PROFILE factors=160 mean_ms={:.6} digest={digest:#018x}",
            start.elapsed().as_secs_f64() * 1000.0 / 160.0
        );
        std::process::exit(0);
    }

    let exe = std::env::current_exe().expect("current release-perf benchmark binary");
    let perf_path = format!(
        "/dev/shm/fsc-chol-mr4-nr4-{}-{}.perf",
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
        .args(["cholesky_wall_mr4_nr4_ab", "--noplot"])
        .env(PROFILE_CHILD, "1")
        .status()
        .expect("spawn candidate perf child");
    assert!(
        profile_status.success(),
        "candidate perf child failed: {profile_status}"
    );
    let report = Command::new("perf")
        .args([
            "report",
            "--stdio",
            "--no-children",
            "--percent-limit",
            "0.1",
            "--sort",
            "symbol",
            "-i",
            &perf_path,
        ])
        .output()
        .expect("run candidate perf report");
    assert!(
        report.status.success(),
        "candidate perf report failed: {}",
        String::from_utf8_lossy(&report.stderr)
    );
    let report = String::from_utf8(report.stdout).expect("utf8 candidate perf report");
    eprintln!("FSCI_CHOL_MR4_NR4_REPORT_BEGIN\n{report}FSCI_CHOL_MR4_NR4_REPORT_END");
    assert!(
        report.contains("cholesky_syrk_flat_rows_mr4_nr4"),
        "candidate-specific SYRK symbol must have non-zero profiled self-time"
    );

    let original = cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("original factor");
    let candidate = cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("candidate factor");
    for (index, (&expected, &actual)) in original.iter().zip(&candidate).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "MR4xNR4 changed n=1000 factor bits at flat index {index}"
        );
    }
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for &value in &candidate {
        for byte in value.to_bits().to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    black_box(original);
    black_box(candidate);

    for candidate_first in [false, true, false, true] {
        if candidate_first {
            black_box(
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("warm candidate factor"),
            );
            black_box(cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("warm original factor"));
        } else {
            black_box(cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("warm original factor"));
            black_box(
                cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("warm candidate factor"),
            );
        }
    }

    let samples = 20usize;
    let factors_per_sample = 64usize;
    let measure_original = || -> Duration {
        let start = std::time::Instant::now();
        black_box(cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("measured original factor"));
        start.elapsed()
    };
    let measure_candidate = || -> Duration {
        let start = std::time::Instant::now();
        black_box(
            cholesky_wall_mr4_nr4_candidate(black_box(&a)).expect("measured candidate factor"),
        );
        start.elapsed()
    };

    let mut original_ms = Vec::with_capacity(samples);
    let mut candidate_ms = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut original_elapsed = Duration::ZERO;
        let mut candidate_elapsed = Duration::ZERO;
        for factor in 0..factors_per_sample {
            if (sample + factor) % 2 == 0 {
                original_elapsed += measure_original();
                candidate_elapsed += measure_candidate();
            } else {
                candidate_elapsed += measure_candidate();
                original_elapsed += measure_original();
            }
        }
        original_ms.push(original_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
        candidate_ms.push(candidate_elapsed.as_secs_f64() * 1000.0 / factors_per_sample as f64);
    }
    let summarize = |values: &[f64]| -> (f64, f64, f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let cv_pct = variance.sqrt() / mean * 100.0;
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |pct: usize| sorted[(sorted.len() - 1) * pct / 100];
        (mean, cv_pct, percentile(50), percentile(95), percentile(99))
    };
    let orig = summarize(&original_ms);
    let cand = summarize(&candidate_ms);
    let paired_ratios = original_ms
        .iter()
        .zip(&candidate_ms)
        .map(|(before, after)| before / after)
        .collect::<Vec<_>>();
    let paired = summarize(&paired_ratios);
    eprintln!(
        "FSCI_CHOL_MR4_NR4_AB ORIG mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6}",
        orig.0, orig.1, orig.2, orig.3, orig.4
    );
    eprintln!(
        "FSCI_CHOL_MR4_NR4_AB CAND mean_ms={:.6} cv_pct={:.3} p50_ms={:.6} p95_ms={:.6} p99_ms={:.6}",
        cand.0, cand.1, cand.2, cand.3, cand.4
    );
    eprintln!(
        "FSCI_CHOL_MR4_NR4_AB speedup_mean={:.6} paired_ratio_mean={:.6} paired_cv_pct_provenance={:.3} cv_used_for_decision=false verdict=NOT_DECIDED reason=legacy_harness_has_no_same_invocation_aa_null digest={digest:#018x} samples={samples} factors_per_sample={factors_per_sample}",
        orig.0 / cand.0,
        paired.0,
        paired.1
    );

    let mut group = c.benchmark_group("cholesky_wall_mr4_nr4_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let candidate_first = std::cell::Cell::new(false);
    group.bench_function("1000x1000_paired_alternating", |bencher| {
        bencher.iter(|| {
            let first = candidate_first.get();
            candidate_first.set(!first);
            if first {
                black_box(
                    cholesky_wall_mr4_nr4_candidate(black_box(&a))
                        .expect("criterion candidate factor"),
                );
                black_box(
                    cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("criterion original factor"),
                );
            } else {
                black_box(
                    cholesky_wall_mr4_nr8_orig(black_box(&a)).expect("criterion original factor"),
                );
                black_box(
                    cholesky_wall_mr4_nr4_candidate(black_box(&a))
                        .expect("criterion candidate factor"),
                );
            }
        });
    });
    group.finish();
}

#[cfg(not(feature = "chol-wall-bench"))]
fn bench_cholesky_wall_mr4_nr4_ab(_c: &mut Criterion) {}

fn bench_cho_factor_gauntlet_scipy(c: &mut Criterion) {
    let mut group = c.benchmark_group("cho_factor_gauntlet_scipy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for &n in CHO_FACTOR_GAUNTLET_SIZES {
        let a = make_symmetric_eigh_matrix(n);
        let b = make_rhs(n);
        group.bench_function(format!("{n}x{n}_rust_cho_factor"), |bencher| {
            bencher
                .iter(|| black_box(cho_factor(black_box(&a), DecompOptions::default()).unwrap()));
        });
        group.bench_function(format!("{n}x{n}_rust_cho_factor_solve"), |bencher| {
            bencher.iter(|| {
                let factor = cho_factor(black_box(&a), DecompOptions::default()).unwrap();
                black_box(cho_solve(black_box(&factor), black_box(&b)).unwrap())
            });
        });
        if scipy_pinv_available() {
            group.bench_function(format!("{n}x{n}_scipy_cho_factor"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_cho_factor_duration(n, iters)
                        .expect("scipy cho_factor oracle should run after availability check")
                });
            });
            group.bench_function(format!("{n}x{n}_scipy_cho_factor_solve"), |bencher| {
                bencher.iter_custom(|iters| {
                    scipy_cho_factor_solve_duration(n, iters).expect(
                        "scipy cho_factor+cho_solve oracle should run after availability check",
                    )
                });
            });
        } else {
            eprintln!("skipping {n}x{n}_scipy_cho_factor: python3 cannot import scipy.linalg");
        }
    }
    group.finish();
}

fn bench_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cols = a.first().map_or(0, Vec::len);
    (0..cols)
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn orthogonal_procrustes_original_bench(a: &[Vec<f64>], b: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
    let at = bench_transpose(a);
    let m_mat = matmul(&at, b).unwrap();
    let svd_result = svd(&m_mat, DecompOptions::default()).unwrap();
    let r = matmul(&svd_result.u, &svd_result.vt).unwrap();
    let scale = svd_result.s.iter().sum();
    (r, scale)
}

fn bench_orthogonal_procrustes_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthogonal_procrustes_gauntlet");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &(rows, cols) in ORTHOGONAL_PROCRUSTES_CASES {
        let a = make_matmul_matrix(rows, cols, 0x33aa);
        let b = make_matmul_matrix(rows, cols, 0x5eed);
        group.bench_function(format!("{rows}x{cols}_orig"), |bencher| {
            bencher.iter(|| {
                black_box(orthogonal_procrustes_original_bench(
                    black_box(&a),
                    black_box(&b),
                ))
            });
        });
        group.bench_function(format!("{rows}x{cols}_candidate"), |bencher| {
            bencher
                .iter(|| black_box(orthogonal_procrustes(black_box(&a), black_box(&b)).unwrap()));
        });
    }
    group.finish();
}

fn bench_khatri_rao_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("khatri_rao_parallel_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &(m, p, r) in &[(200usize, 200usize, 10usize), (1000, 1000, 8)] {
        let a = make_matmul_matrix(m, r, 0x4b52);
        let b = make_matmul_matrix(p, r, 0x9a77);
        group.bench_function(format!("current_parallel_m{m}_p{p}_r{r}"), |bencher| {
            bencher.iter(|| {
                KHATRI_RAO_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(fsci_linalg::khatri_rao(black_box(&a), black_box(&b)).unwrap())
            });
        });
        group.bench_function(format!("orig_strided_m{m}_p{p}_r{r}"), |bencher| {
            bencher.iter(|| {
                KHATRI_RAO_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(fsci_linalg::khatri_rao(black_box(&a), black_box(&b)).unwrap())
            });
        });
    }
    KHATRI_RAO_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

/// Same-binary A/B for `tanhm` sharing the [13/13] Padé polynomials between
/// `expm(A)` and `expm(-A)` via `expm_pm` (6 matmuls + 2 solves) versus two
/// independent `expm` runs (~12 matmuls). Matrices are scaled to 1-norm well
/// under theta13 so the scaling-squaring exponent s=0 (no squarings dilute the
/// shared-matmul win). Both arms asserted byte-identical before timing.
/// Public `expm` adaptive Padé order (m∈{3,5,7,9,13} by ‖A‖₁, SciPy's algorithm)
/// versus the degree-13-only kernel. Small-norm matrices only need m=3/5/7 (2-4
/// matmuls) instead of m=13's 6 + squarings. Matrices scaled into each θ band; both
/// arms asserted tolerance-parity before timing.
fn bench_expm_adaptive_order_ab(c: &mut Criterion) {
    // Scale a base matrix so ‖A‖₁ ≈ target (lands in a specific θ band → order m).
    fn scaled_to_norm(n: usize, seed: usize, target: f64) -> Vec<Vec<f64>> {
        let base = make_matmul_matrix(n, n, seed);
        let cur = (0..n)
            .map(|j| (0..n).map(|i| base[i][j].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let s = target / cur;
        base.iter()
            .map(|r| r.iter().map(|&v| v * s).collect())
            .collect()
    }
    let mut group = c.benchmark_group("expm_adaptive_order_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    // (label, target ‖A‖₁, adaptive order): θ3≈.015 θ5≈.25 θ7≈.95 θ9≈2.10.
    for &(label, target) in &[("m3", 0.01), ("m5", 0.2), ("m7", 0.8), ("m9", 1.8)] {
        for &n in &[128usize, 256usize] {
            let a = scaled_to_norm(n, 0x5b3d, target);
            EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
            let adaptive = expm(&a, DecompOptions::default()).unwrap();
            EXPM_ADAPTIVE_ORDER_DISABLE.store(true, Ordering::Relaxed);
            let deg13 = expm(&a, DecompOptions::default()).unwrap();
            let max_abs = adaptive
                .iter()
                .zip(&deg13)
                .flat_map(|(ra, rd)| ra.iter().zip(rd).map(|(x, y)| (x - y).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs < 1e-12,
                "{label} n{n}: adaptive vs deg13 diverged {max_abs:e}"
            );
            group.bench_function(format!("adaptive_{label}_n{n}"), |bencher| {
                bencher.iter(|| {
                    EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
                    black_box(expm(black_box(&a), DecompOptions::default()).unwrap())
                });
            });
            group.bench_function(format!("deg13_{label}_n{n}"), |bencher| {
                bencher.iter(|| {
                    EXPM_ADAPTIVE_ORDER_DISABLE.store(true, Ordering::Relaxed);
                    black_box(expm(black_box(&a), DecompOptions::default()).unwrap())
                });
            });
        }
    }
    EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

/// `coshm`/`tanhm` (which build on `expm_pm`) with adaptive Padé order on vs off —
/// the small-norm branch of the same θ-order lever as `expm`. Small-norm matrices
/// use a lower degree (fewer shared matmuls). Tolerance-parity asserted before timing.
fn bench_expm_pm_adaptive_ab(c: &mut Criterion) {
    fn scaled_to_norm(n: usize, seed: usize, target: f64) -> Vec<Vec<f64>> {
        let base = make_matmul_matrix(n, n, seed);
        let cur = (0..n)
            .map(|j| (0..n).map(|i| base[i][j].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let s = target / cur;
        base.iter()
            .map(|r| r.iter().map(|&v| v * s).collect())
            .collect()
    }
    let mut group = c.benchmark_group("expm_pm_adaptive_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &(label, target) in &[("m3", 0.01), ("m5", 0.2)] {
        for &n in &[128usize, 256usize] {
            let a = scaled_to_norm(n, 0x71c4, target);
            EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
            let adaptive = coshm(&a, DecompOptions::default()).unwrap();
            EXPM_ADAPTIVE_ORDER_DISABLE.store(true, Ordering::Relaxed);
            let deg13 = coshm(&a, DecompOptions::default()).unwrap();
            let max_abs = adaptive
                .iter()
                .zip(&deg13)
                .flat_map(|(ra, rd)| ra.iter().zip(rd).map(|(x, y)| (x - y).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs < 1e-11,
                "{label} n{n}: coshm adaptive diverged {max_abs:e}"
            );
            group.bench_function(format!("adaptive_{label}_n{n}"), |bencher| {
                bencher.iter(|| {
                    EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
                    black_box(coshm(black_box(&a), DecompOptions::default()).unwrap())
                });
            });
            group.bench_function(format!("deg13_{label}_n{n}"), |bencher| {
                bencher.iter(|| {
                    EXPM_ADAPTIVE_ORDER_DISABLE.store(true, Ordering::Relaxed);
                    black_box(coshm(black_box(&a), DecompOptions::default()).unwrap())
                });
            });
        }
    }
    EXPM_ADAPTIVE_ORDER_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_tanhm_shared_pade_ab(c: &mut Criterion) {
    fn small_norm_matrix(n: usize, seed: usize) -> Vec<Vec<f64>> {
        let base = make_matmul_matrix(n, n, seed);
        let scale = 1.0 / n as f64; // 1-norm ~ 0.48 => s = 0
        base.iter()
            .map(|r| r.iter().map(|&v| v * scale).collect())
            .collect()
    }
    let mut group = c.benchmark_group("tanhm_shared_pade_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[48usize, 96usize] {
        let a = small_norm_matrix(n, 0x5b3d);
        TANHM_SHARED_PADE_DISABLE.store(false, Ordering::Relaxed);
        let shared = tanhm(&a, DecompOptions::default()).unwrap();
        TANHM_SHARED_PADE_DISABLE.store(true, Ordering::Relaxed);
        let orig = tanhm(&a, DecompOptions::default()).unwrap();
        assert!(
            shared
                .iter()
                .zip(&orig)
                .all(|(rs, ro)| rs.iter().zip(ro).all(|(x, y)| x.to_bits() == y.to_bits())),
            "tanhm shared-Pade must be byte-identical to the two-expm path"
        );
        group.bench_function(format!("current_shared_n{n}"), |bencher| {
            bencher.iter(|| {
                TANHM_SHARED_PADE_DISABLE.store(false, Ordering::Relaxed);
                black_box(tanhm(black_box(&a), DecompOptions::default()).unwrap())
            });
        });
        group.bench_function(format!("orig_two_expm_n{n}"), |bencher| {
            bencher.iter(|| {
                TANHM_SHARED_PADE_DISABLE.store(true, Ordering::Relaxed);
                black_box(tanhm(black_box(&a), DecompOptions::default()).unwrap())
            });
        });
    }
    TANHM_SHARED_PADE_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_helmert_parallel_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("helmert_parallel_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[1000usize, 2000usize] {
        group.bench_function(format!("current_sub_n{n}"), |bencher| {
            bencher.iter(|| {
                HELMERT_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(fsci_linalg::helmert(black_box(n)))
            });
        });
        group.bench_function(format!("orig_sub_n{n}"), |bencher| {
            bencher.iter(|| {
                HELMERT_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(fsci_linalg::helmert(black_box(n)))
            });
        });
        group.bench_function(format!("current_full_n{n}"), |bencher| {
            bencher.iter(|| {
                HELMERT_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(fsci_linalg::helmert_full(black_box(n)))
            });
        });
        group.bench_function(format!("orig_full_n{n}"), |bencher| {
            bencher.iter(|| {
                HELMERT_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(fsci_linalg::helmert_full(black_box(n)))
            });
        });
    }
    HELMERT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_pascal_symmetric_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("pascal_symmetric_ab");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for &n in &[192usize, 384usize] {
        group.bench_function(format!("current_quadratic_n{n}"), |bencher| {
            bencher.iter(|| {
                PASCAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
                black_box(pascal(black_box(n), true))
            });
        });
        group.bench_function(format!("orig_gram_n{n}"), |bencher| {
            bencher.iter(|| {
                PASCAL_FORCE_SERIAL.store(true, Ordering::Relaxed);
                black_box(pascal(black_box(n), true))
            });
        });
    }
    PASCAL_FORCE_SERIAL.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_baseline_pinv(c: &mut Criterion) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut group = c.benchmark_group("baseline_pinv");
    group.sample_size(100);
    for &n in BASELINE_SIZES {
        let rows = n * 2;
        let a = make_overdetermined(rows, n);
        group.bench_function(format!("{rows}x{n}"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
        });
        group.bench_function(format!("{rows}x{n}_trsm_serial"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_materialized_rhs"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(true, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_dmatrix_solve"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_dmatrix_cert"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_materialized_cert"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
        });
        group.bench_function(format!("{rows}x{n}_chol_solve"), |bencher| {
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(true, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRSM_THREADS.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_STREAMING_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_CERT.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_ROW_MAJOR_SOLVE.store(false, Relaxed);
            fsci_linalg::DISABLE_TALL_PINV_TRANSPOSE_RHS_COPY.store(false, Relaxed);
            bencher.iter(|| pinv(&a, PinvOptions::default()).unwrap());
            fsci_linalg::DISABLE_TALL_PINV_TRSM.store(false, Relaxed);
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_solve,
    bench_solve_triangular,
    bench_solve_banded,
    bench_inv,
    bench_det,
    bench_lstsq,
    bench_pinv,
    bench_matmul,
    bench_mat_flatten_contiguous_rows,
    bench_frobenius_norm_simd,
    bench_vdot_simd_ab,
    bench_is_diagonal,
    bench_eigh_inviter_parallel_ab,
    bench_eigh_dense,
    bench_randomized_eigh
);

criterion_group!(
    baseline_benches,
    bench_baseline_solve,
    bench_baseline_solve_pos,
    bench_baseline_inv,
    bench_baseline_lstsq,
    bench_u0ucw_wide_lstsq,
    bench_u0ucw_wide_pinv,
    bench_u0ucw_gauntlet_scipy_pinv,
    bench_u0ucw_gauntlet_scipy_lstsq,
    bench_randomized_eigh_gauntlet_scipy,
    bench_dft_gauntlet_scipy,
    bench_cho_factor_gauntlet_scipy,
    bench_lu_factor_gauntlet,
    bench_det_gauntlet,
    bench_orthogonal_procrustes_gauntlet,
    bench_khatri_rao_parallel_ab,
    bench_expm_adaptive_order_ab,
    bench_expm_pm_adaptive_ab,
    bench_tanhm_shared_pade_ab,
    bench_helmert_parallel_ab,
    bench_pascal_symmetric_ab,
    bench_lu_factor_solve_gauntlet,
    bench_baseline_pinv
);

criterion_group!(
    cholesky_wall_benches,
    bench_cholesky_wall_bundle_ab,
    bench_cholesky_wall_panel_inner_ab,
    bench_cholesky_wall_nb_sweep,
    bench_cholesky_wall_trsm_par_ab,
    bench_cholesky_wall_trsm_fma_ab,
    bench_cholesky_wall_nr8_fma_ab,
    bench_cholesky_wall_mr4_nr4_ab
);

criterion_main!(cholesky_wall_benches, benches, baseline_benches);
