use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_sparse::{
    BMAT_FORCE_GENERIC, COO_SUM_DUPLICATES_RADIX_DISABLE, CooMatrix, CscMatrix, CsrMatrix,
    DIAGS_VALIDATE, FormatConvertible, IluOptions, KRON_VALIDATE, Shape2D, SolveOptions,
    VSTACK_FORCE_GENERIC, add_csr, block_diag, bmat, diags, eye, eye_array, find, kron, random,
    scale_coo, scale_csc, scale_csr, sparse_diagonal, sparse_frobenius_inner, sparse_is_symmetric,
    sparse_norm, sparse_sum, sparse_trace, sparse_transpose, spilu, spmm, spmv_csr, spsolve, tril,
    vstack,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Matrix configurations: (rows/cols, density).
const CONFIGS: &[(usize, f64)] = &[
    (100, 0.05),     // 100×100, 5%
    (1_000, 0.01),   // 1000×1000, 1%
    (10_000, 0.001), // 10000×10000, 0.1%
];
const TINY_DENSITY_CASES: &[(usize, f64)] = &[(1_000_000_000, 1e-19), (2_000_000_000, 1e-20)];

const SEED: u64 = 0xBEEF_CAFE;

fn make_random_csr(n: usize, density: f64) -> CsrMatrix {
    random(Shape2D::new(n, n), density, SEED)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_random_rect_csr(rows: usize, cols: usize, density: f64, seed: u64) -> CsrMatrix {
    random(Shape2D::new(rows, cols), density, seed)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect()
}

fn make_skewed_spmv_csr(rows: usize, cols: usize) -> CsrMatrix {
    let hot_rows = rows / 8;
    let mut data = Vec::with_capacity(hot_rows * 80 + (rows - hot_rows) * 2);
    let mut indices = Vec::with_capacity(data.capacity());
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for row in 0..rows {
        let degree = if row < hot_rows { 80 } else { 2 };
        for k in 0..degree {
            data.push((k + 1) as f64 * 0.001);
            indices.push((row.wrapping_mul(37) + k * 9_973) % cols);
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
        .expect("valid skewed CSR")
}

fn sparse_trace_materialized_reference(matrix: &CsrMatrix) -> f64 {
    sparse_diagonal(matrix).iter().sum()
}

fn sparse_norm_scalar_reference(matrix: &CsrMatrix) -> f64 {
    matrix
        .data()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn sparse_sum_scalar_reference(matrix: &CsrMatrix) -> f64 {
    matrix.data().iter().sum()
}

fn sparse_frobenius_inner_nested_reference(a: &CsrMatrix, b: &CsrMatrix) -> f64 {
    let mut sum = 0.0;
    for row in 0..a.shape().rows {
        for a_idx in a.indptr()[row]..a.indptr()[row + 1] {
            for b_idx in b.indptr()[row]..b.indptr()[row + 1] {
                if b.indices()[b_idx] == a.indices()[a_idx] {
                    sum += a.data()[a_idx] * b.data()[b_idx];
                    break;
                }
            }
        }
    }
    sum
}

fn make_inner_bench_matrix(n: usize, row_width: usize, value_bias: f64) -> CsrMatrix {
    let mut data = Vec::with_capacity(n * row_width);
    let mut indices = Vec::with_capacity(n * row_width);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for row in 0..n {
        for col in 0..row_width {
            indices.push(col);
            data.push(value_bias + ((row * 17 + col * 13) % 101) as f64 * 0.001);
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
        .expect("canonical inner-product matrix")
}

fn sparse_is_symmetric_linear_reference(a: &CsrMatrix, tol: f64) -> bool {
    let n = a.shape().rows;
    if n != a.shape().cols {
        return false;
    }
    for row in 0..n {
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            let col = a.indices()[idx];
            let value = a.data()[idx];
            let mut found = false;
            for transpose_idx in a.indptr()[col]..a.indptr()[col + 1] {
                if a.indices()[transpose_idx] == row {
                    if (a.data()[transpose_idx] - value).abs() > tol {
                        return false;
                    }
                    found = true;
                    break;
                }
            }
            if !found && value.abs() > tol {
                return false;
            }
        }
    }
    true
}

fn make_symmetric_block_matrix(n: usize, block_width: usize) -> CsrMatrix {
    assert!(block_width > 0 && n.is_multiple_of(block_width));
    let mut data = Vec::with_capacity(n * block_width);
    let mut indices = Vec::with_capacity(n * block_width);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for row in 0..n {
        let block_start = (row / block_width) * block_width;
        for col in block_start..block_start + block_width {
            indices.push(col);
            data.push(1.0 + ((row + col) % 101) as f64 * 0.001);
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(n, n), data, indices, indptr, false)
        .expect("canonical symmetric block matrix")
}

fn make_wide_transpose_matrix(rows: usize, cols: usize, row_width: usize) -> CsrMatrix {
    let mut data = Vec::with_capacity(rows * row_width);
    let mut indices = Vec::with_capacity(rows * row_width);
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for row in 0..rows {
        for offset in 0..row_width {
            indices.push((row * 131 + offset * 977) % cols);
            data.push((row * row_width + offset) as f64 * 0.001 - 2.0);
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, false)
        .expect("wide transpose matrix")
}

fn sparse_transpose_separate_counts_reference(a: &CsrMatrix) -> CsrMatrix {
    let (rows, cols) = (a.shape().rows, a.shape().cols);
    let nnz = a.data().len();
    let mut counts = vec![0usize; cols];
    for &col in a.indices() {
        counts[col] += 1;
    }
    let mut indptr = vec![0usize; cols + 1];
    for col in 0..cols {
        indptr[col + 1] = indptr[col] + counts[col];
    }
    let mut indices = vec![0usize; nnz];
    let mut data = vec![0.0; nnz];
    let mut positions = vec![0usize; cols];
    for row in 0..rows {
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            let col = a.indices()[idx];
            let dest = indptr[col] + positions[col];
            indices[dest] = row;
            data[dest] = a.data()[idx];
            positions[col] += 1;
        }
    }
    CsrMatrix::from_components(Shape2D::new(cols, rows), data, indices, indptr, false)
        .expect("reference transpose")
}

fn scale_csc_checked_reference(matrix: &CscMatrix, alpha: f64) -> CscMatrix {
    assert!(alpha.is_finite());
    let data = matrix.data().iter().map(|value| value * alpha).collect();
    CscMatrix::from_components(
        matrix.shape(),
        data,
        matrix.indices().to_vec(),
        matrix.indptr().to_vec(),
        false,
    )
    .expect("checked scale csc")
}

fn scale_coo_checked_reference(matrix: &CooMatrix, alpha: f64) -> CooMatrix {
    assert!(alpha.is_finite());
    let data = matrix.data().iter().map(|value| value * alpha).collect();
    CooMatrix::from_triplets(
        matrix.shape(),
        data,
        matrix.row_indices().to_vec(),
        matrix.col_indices().to_vec(),
        false,
    )
    .expect("checked COO reference")
}

// Reproduces the old `tril(.., 0)` path: identical to_coo + lower-triangle filter,
// differing only in routing the result through from_triplets' redundant bounds scan.
fn tril0_checked_reference(matrix: &CsrMatrix) -> CooMatrix {
    let coo = matrix.to_coo().expect("to_coo");
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for idx in 0..coo.nnz() {
        let row = coo.row_indices()[idx];
        let col = coo.col_indices()[idx];
        if row >= col {
            rows.push(row);
            cols.push(col);
            data.push(coo.data()[idx]);
        }
    }
    CooMatrix::from_triplets(coo.shape(), data, rows, cols, false).expect("checked tril")
}

fn csr_to_coo_checked_reference(matrix: &CsrMatrix) -> CooMatrix {
    let mut rows = Vec::with_capacity(matrix.nnz());
    let mut cols = Vec::with_capacity(matrix.nnz());
    let mut data = Vec::with_capacity(matrix.nnz());
    for row in 0..matrix.shape().rows {
        for idx in matrix.indptr()[row]..matrix.indptr()[row + 1] {
            rows.push(row);
            cols.push(matrix.indices()[idx]);
            data.push(matrix.data()[idx]);
        }
    }
    CooMatrix::from_triplets(matrix.shape(), data, rows, cols, false).expect("checked CSR->COO")
}

fn csc_to_coo_checked_reference(matrix: &CscMatrix) -> CooMatrix {
    let mut rows = Vec::with_capacity(matrix.nnz());
    let mut cols = Vec::with_capacity(matrix.nnz());
    let mut data = Vec::with_capacity(matrix.nnz());
    for col in 0..matrix.shape().cols {
        for idx in matrix.indptr()[col]..matrix.indptr()[col + 1] {
            rows.push(matrix.indices()[idx]);
            cols.push(col);
            data.push(matrix.data()[idx]);
        }
    }
    CooMatrix::from_triplets(matrix.shape(), data, rows, cols, false).expect("checked CSC->COO")
}

fn bench_csr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_csr_construction");

    for &(n, density) in CONFIGS {
        let coo = random(Shape2D::new(n, n), density, SEED).expect("random coo");
        group.bench_with_input(
            BenchmarkId::new(format!("{n}x{n}_d{}", (density * 100.0) as u32), n),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let csr = black_box(coo).to_csr().expect("coo->csr");
                    black_box(csr.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmv");

    for &(n, density) in CONFIGS {
        let csr = make_random_csr(n, density);
        let vec = make_vector(n);
        group.bench_with_input(
            BenchmarkId::new(
                format!("{n}x{n}_d{}_nnz{}", (density * 100.0) as u32, csr.nnz()),
                n,
            ),
            &(csr, vec),
            |b, (csr, vec)| {
                b.iter(|| {
                    let result = spmv_csr(black_box(csr), black_box(vec)).expect("spmv");
                    black_box(result.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_spmv_query_gate_ab(c: &mut Criterion) {
    let csr = make_random_csr(100, 0.05);
    let vector = make_vector(100);

    let candidate = spmv_csr(&csr, &vector).expect("candidate spmv");
    let queried_cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(csr.shape().rows.max(1));
    black_box(queried_cores);
    let query_first = spmv_csr(&csr, &vector).expect("query-first spmv");
    assert!(
        candidate
            .iter()
            .zip(&query_first)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    let mut group = c.benchmark_group("spmv_query_gate_ab");
    group.bench_function("candidate_skip_query", |b| {
        b.iter(|| {
            let result = spmv_csr(black_box(&csr), black_box(&vector)).expect("candidate spmv");
            black_box(result);
        });
    });
    group.bench_function("baseline_query_first", |b| {
        b.iter(|| {
            let cores = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1)
                .min(csr.shape().rows.max(1));
            black_box(cores);
            let result = spmv_csr(black_box(&csr), black_box(&vector)).expect("query-first spmv");
            black_box(result);
        });
    });
    group.finish();
}

fn bench_spmv_skewed_rows(c: &mut Criterion) {
    let csr = make_skewed_spmv_csr(100_000, 100_000);
    let vector = make_vector(100_000);
    let mut group = c.benchmark_group("spmv_skewed_rows");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("rows100k_nnz1175k", |b| {
        b.iter(|| {
            let result = spmv_csr(black_box(&csr), black_box(&vector)).expect("skewed spmv");
            black_box(result);
        });
    });
    group.finish();
}

fn bench_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_format_conversion");

    for &(n, density) in CONFIGS {
        let csr = make_random_csr(n, density);
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csr_to_csc"), n),
            &csr,
            |b, csr: &CsrMatrix| {
                b.iter(|| {
                    let csc = black_box(csr).to_csc().expect("csr->csc");
                    black_box(csc.nnz());
                });
            },
        );

        let csc = csr.to_csc().expect("csr->csc");
        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csc_to_csr"), n),
            &csc,
            |b, csc: &CscMatrix| {
                b.iter(|| {
                    let csr = black_box(csc).to_csr().expect("csc->csr");
                    black_box(csr.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_arithmetic");

    for &(n, density) in CONFIGS {
        let a = make_random_csr(n, density);
        let a_csc = a.to_csc().expect("a csc");
        let a_coo = a.to_coo().expect("a coo");
        let b = random(Shape2D::new(n, n), density, SEED ^ 0xFF)
            .expect("random b")
            .to_csr()
            .expect("b csr");
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_add"), n),
            &(a.clone(), b),
            |b_iter, (a, b)| {
                b_iter.iter(|| {
                    let sum = add_csr(black_box(a), black_box(b)).expect("add");
                    black_box(sum.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_find"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let found = find(black_box(a)).expect("find");
                    black_box(found);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_csr(black_box(a), black_box(2.5)).expect("scale");
                    black_box(scaled.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale_csc"), n),
            &a_csc,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_csc(black_box(a), black_box(2.5)).expect("scale csc");
                    black_box(scaled.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale_coo"), n),
            &a_coo,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_coo(black_box(a), black_box(2.5)).expect("scale coo");
                    black_box(scaled.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale_coo_checked_reference"), n),
            &a_coo,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_coo_checked_reference(black_box(a), black_box(2.5));
                    black_box(scaled.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_tril"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let lower = tril(black_box(a), black_box(0)).expect("tril");
                    black_box(lower.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_tril_checked_reference"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let lower = tril0_checked_reference(black_box(a));
                    black_box(lower.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csr_to_coo"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let coo = black_box(a).to_coo().expect("csr to_coo");
                    black_box(coo.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csr_to_coo_checked_reference"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let coo = csr_to_coo_checked_reference(black_box(a));
                    black_box(coo.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csc_to_coo"), n),
            &a_csc,
            |b_iter, a| {
                b_iter.iter(|| {
                    let coo = black_box(a).to_coo().expect("csc to_coo");
                    black_box(coo.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csc_to_coo_checked_reference"), n),
            &a_csc,
            |b_iter, a| {
                b_iter.iter(|| {
                    let coo = csc_to_coo_checked_reference(black_box(a));
                    black_box(coo.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale_csc_checked_reference"), n),
            &a_csc,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_csc_checked_reference(black_box(a), black_box(2.5));
                    black_box(scaled.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_eye(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_eye");

    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("eye", n), &n, |b, &n| {
            b.iter(|| {
                let id = eye(black_box(n)).expect("eye");
                black_box(id.nnz());
            });
        });
    }

    group.finish();
}

fn bench_diags(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_diags");

    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("tridiag", n), &n, |b, &n: &usize| {
            let sub = vec![-1.0; n.saturating_sub(1)];
            let main = vec![2.0; n];
            let sup = vec![-1.0; n.saturating_sub(1)];
            b.iter(|| {
                let csr = diags(
                    black_box(&[sub.clone(), main.clone(), sup.clone()]),
                    &[-1, 0, 1],
                    Some(Shape2D::new(n, n)),
                )
                .expect("tridiag");
                black_box(csr.nnz());
            });
        });
    }

    group.finish();
}

// Reproduces the old `eye_array`/`eye_rectangular` path: build the single diagonal as a COO then
// canonicalize via to_csr. The current path builds the CSR directly.
fn eye_rect_via_coo_reference(rows: usize, cols: usize, k: isize) -> CsrMatrix {
    let shape = Shape2D::new(rows, cols);
    let (row_start, length) = if k >= 0 {
        let k_us = k as usize;
        if k_us >= cols {
            (0usize, 0usize)
        } else {
            (0usize, rows.min(cols - k_us))
        }
    } else {
        let k_abs = (-k) as usize;
        if k_abs >= rows {
            (0usize, 0usize)
        } else {
            (k_abs, (rows - k_abs).min(cols))
        }
    };
    let data = vec![1.0; length];
    let r: Vec<usize> = (row_start..row_start + length).collect();
    let c: Vec<usize> = if k >= 0 {
        (k as usize..k as usize + length).collect()
    } else {
        (0..length).collect()
    };
    CooMatrix::from_triplets(shape, data, r, c, false)
        .expect("eye ref coo")
        .to_csr()
        .expect("eye ref to_csr")
}

fn bench_eye_rectangular_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_eye_rectangular_ab");
    group.sample_size(10);
    for &n in &[100_000usize, 1_000_000usize] {
        group.bench_with_input(BenchmarkId::new("current_direct", n), &n, |b, &n| {
            b.iter(|| {
                let m = eye_array(black_box(n), black_box(n), black_box(1)).expect("eye_array");
                black_box(m.nnz());
            });
        });
        group.bench_with_input(BenchmarkId::new("orig_coo_to_csr", n), &n, |b, &n| {
            b.iter(|| {
                let m = eye_rect_via_coo_reference(black_box(n), black_box(n), black_box(1));
                black_box(m.nnz());
            });
        });
    }
    group.finish();
}

fn bench_bmat_canonical_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_bmat_canonical_ab");
    group.sample_size(10);
    // 4x4 grid of canonical CSR blocks, ~11k nnz each (~180k total over 6000x6000): the generic
    // path builds a full COO (3 x nnz) + from_triplets + to_csr the direct row-shift build skips.
    let g = 4usize;
    let bs = 1_500usize;
    let mats: Vec<CsrMatrix> = (0..(g * g) as u64)
        .map(|i| make_random_rect_csr(bs, bs, 0.005, SEED ^ (i.wrapping_mul(0x9E37))))
        .collect();
    let grid: Vec<Vec<Option<&CsrMatrix>>> = (0..g)
        .map(|i| (0..g).map(|j| Some(&mats[i * g + j])).collect())
        .collect();

    group.bench_function("current_direct", |bn| {
        bn.iter(|| {
            BMAT_FORCE_GENERIC.store(false, Ordering::Relaxed);
            let m = bmat(black_box(&grid)).expect("bmat");
            black_box(m.nnz());
        });
    });
    group.bench_function("orig_generic", |bn| {
        bn.iter(|| {
            BMAT_FORCE_GENERIC.store(true, Ordering::Relaxed);
            let m = bmat(black_box(&grid)).expect("bmat");
            black_box(m.nnz());
        });
    });
    BMAT_FORCE_GENERIC.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_vstack_canonical_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vstack_canonical_ab");
    group.sample_size(10);
    // 16 canonical CSR blocks sharing 4000 cols, ~20k nnz each (~320k total over 16k rows): the
    // generic path pays N to_coo conversions + COO alloc + to_csr rebuild the direct concat skips.
    let cols = 4_000usize;
    let block_rows = 1_000usize;
    let blocks: Vec<CsrMatrix> = (0..16u64)
        .map(|i| make_random_rect_csr(block_rows, cols, 0.005, SEED ^ i))
        .collect();
    let refs: Vec<&dyn FormatConvertible> =
        blocks.iter().map(|m| m as &dyn FormatConvertible).collect();

    group.bench_function("current_direct", |bn| {
        bn.iter(|| {
            VSTACK_FORCE_GENERIC.store(false, Ordering::Relaxed);
            let m = vstack(black_box(&refs)).expect("vstack");
            black_box(m.nnz());
        });
    });
    group.bench_function("orig_generic", |bn| {
        bn.iter(|| {
            VSTACK_FORCE_GENERIC.store(true, Ordering::Relaxed);
            let m = vstack(black_box(&refs)).expect("vstack");
            black_box(m.nnz());
        });
    });
    VSTACK_FORCE_GENERIC.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_kron_validate_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_kron_validate_ab");
    group.sample_size(10);
    // total_nnz ≈ 5000 * 200 = 1M over 50000 output rows: clears the parallel-fill gate, so the
    // redundant O(nnz) validation scan (bounds + detect_canonical) is a serial tail worth trimming.
    let a = make_random_csr(1_000, 0.005);
    let b = make_random_csr(50, 0.08);
    group.bench_function("current_trusted", |bn| {
        bn.iter(|| {
            KRON_VALIDATE.store(false, Ordering::Relaxed);
            let m = kron(black_box(&a), black_box(&b)).expect("kron");
            black_box(m.nnz());
        });
    });
    group.bench_function("orig_validated", |bn| {
        bn.iter(|| {
            KRON_VALIDATE.store(true, Ordering::Relaxed);
            let m = kron(black_box(&a), black_box(&b)).expect("kron");
            black_box(m.nnz());
        });
    });
    KRON_VALIDATE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_diags_validate_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_diags_validate_ab");
    group.sample_size(10);
    // Wide band so capacity clears the parallel-fill gate (~1M entries): the redundant O(nnz)
    // validation scans (bounds + detect_canonical) are serial while the fill fans across cores.
    let n = 50_000usize;
    let band = 10isize;
    let offsets: Vec<isize> = (-band..=band).collect();
    let diagonals: Vec<Vec<f64>> = offsets
        .iter()
        .map(|&o| vec![1.0; n - o.unsigned_abs()])
        .collect();
    let shape = Some(Shape2D::new(n, n));

    group.bench_function(BenchmarkId::new("current_trusted", n), |b| {
        b.iter(|| {
            DIAGS_VALIDATE.store(false, Ordering::Relaxed);
            let csr = diags(black_box(&diagonals), black_box(&offsets), shape).expect("diags");
            black_box(csr.nnz());
        });
    });
    group.bench_function(BenchmarkId::new("orig_validated", n), |b| {
        b.iter(|| {
            DIAGS_VALIDATE.store(true, Ordering::Relaxed);
            let csr = diags(black_box(&diagonals), black_box(&offsets), shape).expect("diags");
            black_box(csr.nnz());
        });
    });
    DIAGS_VALIDATE.store(false, Ordering::Relaxed);
    group.finish();
}

fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmm");
    for &(n, density) in &[(500usize, 0.02f64), (1_000, 0.01), (2_000, 0.01)] {
        let a = make_random_csr(n, density);
        let b = random(Shape2D::new(n, n), density, SEED ^ 0x1234)
            .expect("b coo")
            .to_csr()
            .expect("b csr");
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);
        group.bench_with_input(BenchmarkId::new(label, n), &(a, b), |bi, (a, b)| {
            bi.iter(|| {
                let prod = spmm(black_box(a), black_box(b));
                black_box(prod.nnz());
            });
        });
    }
    group.finish();
}

fn bench_kron(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_kron");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    let a = make_random_rect_csr(400, 400, 0.02, SEED);
    let b = make_random_rect_csr(120, 120, 0.05, SEED ^ 0x5C1F);
    let label = format!("400x400d2_120x120d5_nnz{}_{}", a.nnz(), b.nnz());

    group.bench_with_input(
        BenchmarkId::new(label, a.nnz() * b.nnz()),
        &(a, b),
        |bi, (a, b)| {
            bi.iter(|| {
                let product = kron(black_box(a), black_box(b)).expect("kron");
                black_box(product.nnz());
            });
        },
    );

    group.finish();
}

fn make_spilu_banded_csc(n: usize, half_bandwidth: usize) -> CscMatrix {
    let entries_per_row = half_bandwidth.saturating_mul(2).saturating_add(1);
    let mut data = Vec::with_capacity(n.saturating_mul(entries_per_row));
    let mut rows = Vec::with_capacity(data.capacity());
    let mut cols = Vec::with_capacity(data.capacity());

    for row in 0..n {
        let start = row.saturating_sub(half_bandwidth);
        let end = row.saturating_add(half_bandwidth).min(n.saturating_sub(1));
        for col in start..=end {
            rows.push(row);
            cols.push(col);
            if row == col {
                data.push(entries_per_row as f64 + 2.0 + (row % 17) as f64 * 0.001);
            } else {
                data.push(-1.0 / (row.abs_diff(col) + 1) as f64);
            }
        }
    }

    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("spilu banded coo")
        .to_csc()
        .expect("spilu banded csc")
}

fn make_laplacian_2d(k: usize) -> CsrMatrix {
    let n = k * k;
    let mut rows = Vec::with_capacity(n * 5);
    let mut cols = Vec::with_capacity(n * 5);
    let mut data = Vec::with_capacity(n * 5);
    let idx = |r: usize, c: usize| r * k + c;
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i64 + dr;
                let nc = c as i64 + dc;
                if nr >= 0 && nr < k as i64 && nc >= 0 && nc < k as i64 {
                    rows.push(i);
                    cols.push(idx(nr as usize, nc as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("laplacian coo")
        .to_csr()
        .expect("laplacian csr")
}

fn bench_spilu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spilu");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for &(n, half_bandwidth) in &[(512usize, 16usize), (1_024, 32)] {
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_bw{half_bandwidth}"), n),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let ilu = spilu(black_box(matrix), IluOptions::default()).expect("spilu");
                    black_box(ilu.shape);
                });
            },
        );
    }

    group.finish();
}

fn bench_spsolve_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spsolve_laplacian");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for &side in &[40usize, 70] {
        let matrix = make_laplacian_2d(side);
        let n = side * side;
        let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        group.bench_with_input(
            BenchmarkId::new("spsolve", n),
            &(matrix, rhs),
            |b, (a, rhs)| {
                b.iter(|| {
                    let result = spsolve(black_box(a), black_box(rhs), SolveOptions::default())
                        .expect("spsolve laplacian");
                    black_box(result.solution.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_random_tiny_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_random_tiny_density");
    for &(n, density) in TINY_DENSITY_CASES {
        group.bench_with_input(
            BenchmarkId::new(format!("{n}x{n}_d{density:.0e}"), n),
            &(n, density),
            |b, &(n, density)| {
                b.iter(|| {
                    let coo = random(
                        Shape2D::new(black_box(n), black_box(n)),
                        black_box(density),
                        black_box(SEED),
                    )
                    .expect("random tiny density");
                    black_box(coo.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_coo_to_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_to_csr");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        // Deterministic unsorted triplets (LCG), with some duplicates.
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            rows.push(next() % n);
            cols.push(next() % n);
            data.push((next() % 1000) as f64 + 1.0);
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), nnz),
            &coo,
            |bi, coo| {
                bi.iter(|| {
                    let csr = FormatConvertible::to_csr(black_box(coo)).expect("to_csr");
                    black_box(csr.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_coo_sum_duplicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_sum_duplicates");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        let mut st = 0xC0FF_EE00_1234_5678u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let rows: Vec<usize> = (0..nnz).map(|_| next() % n).collect();
        let cols: Vec<usize> = (0..nnz).map(|_| next() % n).collect();
        let data: Vec<f64> = (0..nnz).map(|_| (next() % 1000) as f64 + 1.0).collect();
        let fixture = (rows, cols, data);
        for (label, disable_radix) in [("current", false), ("legacy_original", true)] {
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_n{n}_nnz{nnz}"), nnz),
                &fixture,
                |bi, (rows, cols, data)| {
                    bi.iter(|| {
                        COO_SUM_DUPLICATES_RADIX_DISABLE.store(disable_radix, Ordering::Relaxed);
                        let coo = CooMatrix::from_triplets(
                            Shape2D::new(n, n),
                            black_box(data).clone(),
                            black_box(rows).clone(),
                            black_box(cols).clone(),
                            true,
                        )
                        .expect("coo");
                        black_box(coo.nnz());
                    });
                },
            );
        }
        COO_SUM_DUPLICATES_RADIX_DISABLE.store(false, Ordering::Relaxed);
    }
    group.finish();
}

fn bench_coo_to_csc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_to_csc");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        let mut st = 0x0FE1_DEAD_5678_1234u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            rows.push(next() % n);
            cols.push(next() % n);
            data.push((next() % 1000) as f64 + 1.0);
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), nnz),
            &coo,
            |bi, coo| {
                bi.iter(|| {
                    let csc = FormatConvertible::to_csc(black_box(coo)).expect("to_csc");
                    black_box(csc.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_block_diag(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_block_diag");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (nblocks, bs, dens) in [(2000usize, 40usize, 0.1f64), (500, 120, 0.05)] {
        let blocks: Vec<CsrMatrix> = (0..nblocks)
            .map(|i| make_random_rect_csr(bs, bs, dens, SEED ^ (i as u64).wrapping_mul(0x9E37)))
            .collect();
        let refs: Vec<&CsrMatrix> = blocks.iter().collect();
        let nnz: usize = blocks.iter().map(CsrMatrix::nnz).sum();
        group.bench_with_input(
            BenchmarkId::new(format!("nblocks{nblocks}_bs{bs}_nnz{nnz}"), nnz),
            &refs,
            |bi, refs| {
                bi.iter(|| {
                    let out = block_diag(black_box(refs)).expect("block_diag");
                    black_box(out.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_sparse_trace_direct_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_trace_direct_ab");
    group.sample_size(15);
    let n = 65_535usize;
    let matrix = CsrMatrix::from_components(
        Shape2D::new(n, n),
        (0..n).map(|i| 1.0 + (i % 17) as f64).collect(),
        (0..n).collect(),
        (0..=n).collect(),
        false,
    )
    .expect("diagonal csr");

    group.bench_function("current_direct_n65535", |bencher| {
        bencher.iter(|| black_box(sparse_trace(black_box(&matrix))));
    });
    group.bench_function("orig_materialized_n65535", |bencher| {
        bencher.iter(|| black_box(sparse_trace_materialized_reference(black_box(&matrix))));
    });
    group.finish();
}

fn bench_sparse_norm_fro_simd_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_norm_fro_simd_ab");
    group.sample_size(15);
    let matrix = make_inner_bench_matrix(32_768, 64, 1.0);

    group.bench_function("current_simd_nnz2097152", |bencher| {
        bencher.iter(|| black_box(sparse_norm(black_box(&matrix), black_box("fro"))));
    });
    group.bench_function("orig_scalar_nnz2097152", |bencher| {
        bencher.iter(|| black_box(sparse_norm_scalar_reference(black_box(&matrix))));
    });
    group.finish();
}

fn bench_sparse_sum_simd_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_sum_simd_ab");
    group.sample_size(15);
    let matrix = make_inner_bench_matrix(32_768, 64, 1.0);

    group.bench_function("current_simd_nnz2097152", |bencher| {
        bencher.iter(|| black_box(sparse_sum(black_box(&matrix))));
    });
    group.bench_function("orig_scalar_nnz2097152", |bencher| {
        bencher.iter(|| black_box(sparse_sum_scalar_reference(black_box(&matrix))));
    });
    group.finish();
}

fn bench_sparse_frobenius_inner_merge_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_frobenius_inner_merge_ab");
    group.sample_size(15);
    let n = 4_096usize;
    let row_width = 64usize;
    let a = make_inner_bench_matrix(n, row_width, 1.0);
    let b = make_inner_bench_matrix(n, row_width, 2.0);

    group.bench_function("current_merge_n4096_w64", |bencher| {
        bencher.iter(|| black_box(sparse_frobenius_inner(black_box(&a), black_box(&b))));
    });
    group.bench_function("orig_nested_n4096_w64", |bencher| {
        bencher.iter(|| {
            black_box(sparse_frobenius_inner_nested_reference(
                black_box(&a),
                black_box(&b),
            ))
        });
    });
    group.finish();
}

fn bench_sparse_is_symmetric_lookup_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_is_symmetric_lookup_ab");
    group.sample_size(15);
    let matrix = make_symmetric_block_matrix(4_096, 64);

    group.bench_function("current_binary_n4096_w64", |bencher| {
        bencher.iter(|| black_box(sparse_is_symmetric(black_box(&matrix), black_box(0.0))));
    });
    group.bench_function("orig_linear_n4096_w64", |bencher| {
        bencher.iter(|| {
            black_box(sparse_is_symmetric_linear_reference(
                black_box(&matrix),
                black_box(0.0),
            ))
        });
    });
    group.finish();
}

fn bench_sparse_transpose_bookkeeping_ab(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_transpose_bookkeeping_ab");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    let matrix = make_wide_transpose_matrix(512, 32_768, 16);

    group.bench_function("current_in_place_counts", |bencher| {
        bencher.iter(|| black_box(sparse_transpose(black_box(&matrix))));
    });
    group.bench_function("orig_separate_counts", |bencher| {
        bencher.iter(|| {
            black_box(sparse_transpose_separate_counts_reference(black_box(
                &matrix,
            )))
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_coo_sum_duplicates,
    bench_coo_to_csr,
    bench_coo_to_csc,
    bench_block_diag,
    bench_csr_construction,
    bench_spmv,
    bench_spmv_query_gate_ab,
    bench_spmv_skewed_rows,
    bench_format_conversion,
    bench_arithmetic,
    bench_eye,
    bench_eye_rectangular_ab,
    bench_diags,
    bench_diags_validate_ab,
    bench_kron_validate_ab,
    bench_vstack_canonical_ab,
    bench_bmat_canonical_ab,
    bench_spmm,
    bench_kron,
    bench_spilu,
    bench_spsolve_laplacian,
    bench_random_tiny_density,
    bench_sparse_trace_direct_ab,
    bench_sparse_norm_fro_simd_ab,
    bench_sparse_sum_simd_ab,
    bench_sparse_frobenius_inner_merge_ab,
    bench_sparse_is_symmetric_lookup_ab,
    bench_sparse_transpose_bookkeeping_ab,
);
criterion_main!(benches);
