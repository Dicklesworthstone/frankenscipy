use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_ndimage::{
    BoundaryMode, NdArray, binary_dilation, binary_erosion, maximum_filter, median_filter,
    minimum_filter, rank_filter,
};
use std::hint::black_box;

fn image(side: usize) -> NdArray {
    let data: Vec<f64> = (0..side * side)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() * 100.0 + (x * 0.0031).cos() * 37.0 + (i % 53) as f64
        })
        .collect();
    NdArray::new(data, vec![side, side]).expect("image")
}

fn bench_minmax_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_filter");
    let img = image(256);
    for &size in &[7usize, 15, 31] {
        group.bench_function(BenchmarkId::new("minimum_256x256", size), |b| {
            b.iter(|| {
                minimum_filter(black_box(&img), size, BoundaryMode::Reflect, 0.0).expect("min")
            })
        });
        group.bench_function(BenchmarkId::new("maximum_256x256", size), |b| {
            b.iter(|| {
                maximum_filter(black_box(&img), size, BoundaryMode::Reflect, 0.0).expect("max")
            })
        });
    }
    group.finish();
}

fn binary_image(side: usize) -> NdArray {
    // Mostly-set image (worst case for the footprint scan / scatter).
    let data: Vec<f64> = (0..side * side)
        .map(|i| if i % 13 == 0 { 0.0 } else { 1.0 })
        .collect();
    NdArray::new(data, vec![side, side]).expect("binary image")
}

fn bench_binary_morph(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_morph");
    let img = binary_image(256);
    for &size in &[7usize, 15] {
        group.bench_function(BenchmarkId::new("erosion_256x256", size), |b| {
            b.iter(|| binary_erosion(black_box(&img), size, 1).expect("erosion"))
        });
        group.bench_function(BenchmarkId::new("dilation_256x256", size), |b| {
            b.iter(|| binary_dilation(black_box(&img), size, 1).expect("dilation"))
        });
    }
    group.finish();
}

fn bench_rank_filter(c: &mut Criterion) {
    let img = image(160);
    let mut group = c.benchmark_group("rank_filter");
    for &size in &[7usize, 15] {
        group.bench_function(BenchmarkId::new("median_160x160", size), |b| {
            b.iter(|| {
                median_filter(black_box(&img), size, BoundaryMode::Reflect, 0.0).expect("med")
            })
        });
        let kt = size * size;
        group.bench_function(BenchmarkId::new("rank_q25_160x160", size), |b| {
            b.iter(|| {
                rank_filter(
                    black_box(&img),
                    (kt / 4) as isize,
                    size,
                    BoundaryMode::Reflect,
                    0.0,
                )
                .expect("rank")
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_minmax_filter,
    bench_binary_morph,
    bench_rank_filter
);
criterion_main!(benches);
