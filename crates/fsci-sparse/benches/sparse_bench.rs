use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, IluOptions, Shape2D, add_csr, diags, eye,
    random, scale_csr, spilu, spmm, spmv_csr,
};
use std::hint::black_box;
use std::time::Duration;

/// Matrix configurations: (rows/cols, density).
const CONFIGS: &[(usize, f64)] = &[
    (100, 0.05),     // 100×100, 5%
    (1_000, 0.01),   // 1000×1000, 1%
    (10_000, 0.001), // 10000×10000, 0.1%
];

const SEED: u64 = 0xBEEF_CAFE;

fn make_random_csr(n: usize, density: f64) -> CsrMatrix {
    random(Shape2D::new(n, n), density, SEED)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect()
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
            BenchmarkId::new(format!("{label}_scale"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_csr(black_box(a), black_box(2.5)).expect("scale");
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

criterion_group!(
    benches,
    bench_csr_construction,
    bench_spmv,
    bench_format_conversion,
    bench_arithmetic,
    bench_eye,
    bench_diags,
    bench_spmm,
    bench_spilu
);
criterion_main!(benches);
