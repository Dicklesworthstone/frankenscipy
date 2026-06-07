use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_stats::{
    HaltonSampler, SobolSampler, SomersDInput, acf, argsort, centered_discrepancy, ecdf, histogram,
    kendalltau, l2_star_discrepancy, mannkendall, mixture_discrepancy, pacf, psd_welch, rand_index,
    somersd, wraparound_discrepancy,
};

fn deterministic_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() + (x * 0.031).cos() * 0.25 + (i % 17) as f64 * 0.001
        })
        .collect()
}

fn qmc_halton_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = HaltonSampler::new(dimension).expect("valid Halton dimension");
    sampler.sample(n)
}

fn qmc_sobol_sample(n: usize, dimension: usize) -> Vec<f64> {
    let mut sampler = SobolSampler::new(dimension).expect("valid Sobol dimension");
    sampler.sample(n)
}

fn bench_qmc_discrepancy(c: &mut Criterion) {
    let sample = qmc_halton_sample(512, 2);
    let mut group = c.benchmark_group("qmc_discrepancy");
    group.bench_function("centered/512x2", |b| {
        b.iter(|| centered_discrepancy(&sample, 2).expect("centered discrepancy"))
    });
    group.bench_function("mixture/512x2", |b| {
        b.iter(|| mixture_discrepancy(&sample, 2).expect("mixture discrepancy"))
    });
    group.bench_function("l2_star/512x2", |b| {
        b.iter(|| l2_star_discrepancy(&sample, 2).expect("l2 star discrepancy"))
    });
    group.bench_function("wraparound/512x2", |b| {
        b.iter(|| wraparound_discrepancy(&sample, 2).expect("wraparound discrepancy"))
    });
    group.finish();
}

fn bench_qmc_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("qmc_sampling");
    for n in [1024usize, 4096] {
        group.bench_function(BenchmarkId::new("halton_4d", n), |b| {
            b.iter(|| qmc_halton_sample(n, 4))
        });
        group.bench_function(BenchmarkId::new("sobol_2d", n), |b| {
            b.iter(|| qmc_sobol_sample(n, 2))
        });
    }
    group.finish();
}

fn bench_ordering_and_bins(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let x_eval = deterministic_data(512);
    let data_big = deterministic_data(65536);
    let mut group = c.benchmark_group("ordering_and_bins");
    group.bench_function("argsort/4096", |b| b.iter(|| argsort(&data)));
    group.bench_function("argsort/65536", |b| b.iter(|| argsort(&data_big)));
    group.bench_function("histogram/4096x64", |b| b.iter(|| histogram(&data, 64)));
    group.bench_function("ecdf/4096x512", |b| b.iter(|| ecdf(&data, &x_eval)));
    group.finish();
}

fn bench_time_series(c: &mut Criterion) {
    let data = deterministic_data(4096);
    let mut group = c.benchmark_group("time_series");
    group.bench_function("acf/4096x64", |b| b.iter(|| acf(&data, 64)));
    group.bench_function("pacf/4096x64", |b| b.iter(|| pacf(&data, 64)));
    group.bench_function("psd_welch/4096_w128_o64", |b| {
        b.iter(|| psd_welch(&data, 128, 64, 1.0))
    });
    group.finish();
}

fn bench_rank_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_correlation");
    for &n in &[2048usize, 4096] {
        // x roughly monotone, y a noisy/tied transform — exercises ties.
        let x = deterministic_data(n);
        let y: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 0.013).cos() + (i % 11) as f64 * 0.1).round())
            .collect();
        group.bench_function(BenchmarkId::new("kendalltau", n), |b| {
            b.iter(|| kendalltau(&x, &y))
        });
        group.bench_function(BenchmarkId::new("mannkendall", n), |b| {
            b.iter(|| mannkendall(&x))
        });
    }
    group.finish();
}

fn bench_somersd(c: &mut Criterion) {
    let mut group = c.benchmark_group("somersd");
    // Distinct ranks -> an n x n contingency table, the worst case for the
    // per-cell quadrant re-summation (O((R*C)^2)).
    for &n in &[64usize, 128] {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % n) as f64).collect();
        group.bench_function(BenchmarkId::new("rankings", n), |b| {
            b.iter(|| somersd(SomersDInput::Rankings(&x, &y), None))
        });
    }
    group.finish();
}

fn bench_rand_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_index");
    for &n in &[2000usize, 8000] {
        let lt: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let lp: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64).collect();
        group.bench_function(BenchmarkId::new("k10", n), |b| {
            b.iter(|| rand_index(&lt, &lp))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qmc_discrepancy,
    bench_qmc_sampling,
    bench_ordering_and_bins,
    bench_time_series,
    bench_rank_correlation,
    bench_somersd,
    bench_rand_index
);
criterion_main!(benches);
