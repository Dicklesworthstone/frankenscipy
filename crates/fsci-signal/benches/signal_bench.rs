use criterion::{Criterion, criterion_group, criterion_main};
use fsci_signal::{
    ConvolveMode, FirWindow, SosSection, cwt, fftconvolve, filtfilt, firls, firwin, lfilter,
    medfilt, order_filter, remez, ricker, sosfilt, welch,
};
use std::hint::black_box;

fn deterministic_signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            (37.0 * t).sin() + 0.35 * (91.0 * t).cos() + 0.1 * ((i * 17 % 29) as f64 - 14.0)
        })
        .collect()
}

fn bench_convolution(c: &mut Criterion) {
    let x = deterministic_signal(4096);
    let h = deterministic_signal(257);

    c.bench_function("convolution/fftconvolve/4096x257_same", |b| {
        b.iter(|| {
            black_box(fftconvolve(
                black_box(&x),
                black_box(&h),
                black_box(ConvolveMode::Same),
            ))
        })
    });
}

fn bench_filtering(c: &mut Criterion) {
    let x = deterministic_signal(4096);
    let b_coeffs = vec![0.067_455_27, 0.134_910_55, 0.067_455_27];
    let a_coeffs = vec![1.0, -1.142_980_5, 0.412_801_6];
    let sos: Vec<SosSection> = vec![
        [
            0.067_455_27,
            0.134_910_55,
            0.067_455_27,
            1.0,
            -1.142_980_5,
            0.412_801_6,
        ],
        [
            0.067_455_27,
            0.134_910_55,
            0.067_455_27,
            1.0,
            -1.142_980_5,
            0.412_801_6,
        ],
    ];

    c.bench_function("filtering/lfilter/4096_biquad", |b| {
        b.iter(|| {
            black_box(lfilter(
                black_box(&b_coeffs),
                black_box(&a_coeffs),
                black_box(&x),
                black_box(None),
            ))
        })
    });

    c.bench_function("filtering/filtfilt/4096_biquad", |b| {
        b.iter(|| {
            black_box(filtfilt(
                black_box(&b_coeffs),
                black_box(&a_coeffs),
                black_box(&x),
            ))
        })
    });

    c.bench_function("filtering/sosfilt/4096_two_sections", |b| {
        b.iter(|| black_box(sosfilt(black_box(&sos), black_box(&x))))
    });
}

fn bench_spectral(c: &mut Criterion) {
    let x = deterministic_signal(4096);

    c.bench_function("spectral/welch/4096_w256_o128", |b| {
        b.iter(|| {
            black_box(welch(
                black_box(&x),
                black_box(1.0),
                black_box(Some("hann")),
                black_box(Some(256)),
                black_box(Some(128)),
            ))
        })
    });
}

fn bench_wavelets(c: &mut Criterion) {
    let x = deterministic_signal(2048);
    let widths: Vec<f64> = (1..=32).map(|w| w as f64).collect();

    c.bench_function("wavelets/cwt_ricker/2048x32", |b| {
        b.iter(|| black_box(cwt(black_box(&x), ricker, black_box(&widths))))
    });
}

fn bench_design(c: &mut Criterion) {
    let bands = vec![0.0, 0.2, 0.3, 0.5];
    let desired = vec![1.0, 1.0, 0.0, 0.0];
    let remez_desired = vec![1.0, 0.0];
    let weights = vec![1.0, 10.0];

    c.bench_function("design/firwin/513_hamming", |b| {
        b.iter(|| {
            black_box(firwin(
                black_box(513),
                black_box(&[0.2]),
                black_box(FirWindow::Hamming),
                black_box(true),
            ))
        })
    });

    c.bench_function("design/firls/257_two_band", |b| {
        b.iter(|| {
            black_box(firls(
                black_box(257),
                black_box(&bands),
                black_box(&desired),
                black_box(None),
            ))
        })
    });

    c.bench_function("design/remez/257_two_band", |b| {
        b.iter(|| {
            black_box(remez(
                black_box(257),
                black_box(&bands),
                black_box(&remez_desired),
                black_box(Some(&weights)),
            ))
        })
    });
}

fn bench_medfilt(c: &mut Criterion) {
    let signal = deterministic_signal(8192);
    let mut group = c.benchmark_group("medfilt");
    for &k in &[65usize, 257, 513] {
        group.bench_function(format!("8192_k{k}"), |b| {
            b.iter(|| medfilt(black_box(&signal), k).expect("medfilt"))
        });
    }
    group.finish();
}

fn bench_order_filter(c: &mut Criterion) {
    let signal = deterministic_signal(8192);
    let mut group = c.benchmark_group("order_filter");
    for &ws in &[65usize, 257] {
        group.bench_function(format!("8192_w{ws}_q25"), |b| {
            b.iter(|| order_filter(black_box(&signal), ws, ws / 4))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_convolution,
    bench_filtering,
    bench_spectral,
    bench_wavelets,
    bench_design,
    bench_medfilt,
    bench_order_filter
);
criterion_main!(benches);
