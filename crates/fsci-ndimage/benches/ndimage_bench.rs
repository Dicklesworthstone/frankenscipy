use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_ndimage::{
    BoundaryMode, NdArray, binary_closing, binary_dilation, binary_erosion, binary_opening,
    diff_array, maximum_filter, maximum_filter1d, mean, median_filter, minimum_filter,
    minimum_filter1d, rank_filter, watershed_ift,
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

fn bench_minmax_filter1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_filter1d");
    let n = 65536usize;
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.01).sin() * 100.0 + (i % 7) as f64)
        .collect();
    let line = NdArray::new(data, vec![n]).expect("line");
    for &size in &[31usize, 101] {
        group.bench_function(BenchmarkId::new("maximum_65536", size), |b| {
            b.iter(|| {
                maximum_filter1d(black_box(&line), size, 0, BoundaryMode::Reflect, 0.0)
                    .expect("max1d")
            })
        });
        group.bench_function(BenchmarkId::new("minimum_65536", size), |b| {
            b.iter(|| {
                minimum_filter1d(black_box(&line), size, 0, BoundaryMode::Reflect, 0.0)
                    .expect("min1d")
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
        // opening = erosion∘dilation, closing = dilation∘erosion — both inherit the 2D
        // bit-pack speedup (frankenscipy-9l5oo). scipy's binary_opening/closing don't
        // decompose the box structure (full footprint), so they are very slow here.
        group.bench_function(BenchmarkId::new("opening_256x256", size), |b| {
            b.iter(|| binary_opening(black_box(&img), size, 1).expect("opening"))
        });
        group.bench_function(BenchmarkId::new("closing_256x256", size), |b| {
            b.iter(|| binary_closing(black_box(&img), size, 1).expect("closing"))
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

/// Correlation (precomputed tap-delta table, frankenscipy-e3r7e) and the separable
/// Gaussian filter — the common dense-filter workloads.
fn bench_correlate_gaussian(c: &mut Criterion) {
    use fsci_ndimage::{correlate, gaussian_filter};
    let img = image(256);
    let weights = NdArray::new(vec![1.0; 25], vec![5, 5]).expect("weights");
    let mut group = c.benchmark_group("correlate_gaussian");
    group.bench_function("correlate_5x5/256", |b| {
        b.iter(|| correlate(black_box(&img), &weights, BoundaryMode::Reflect, 0.0).expect("corr"))
    });
    group.bench_function("gaussian_sigma2/256", |b| {
        b.iter(|| gaussian_filter(black_box(&img), 2.0, BoundaryMode::Reflect, 0.0).expect("gauss"))
    });
    group.finish();
}

fn label_mean_case(side: usize, label_count: usize) -> (NdArray, NdArray, Vec<usize>) {
    let n = side * side;
    let labels: Vec<f64> = (0..n)
        .map(|idx| {
            let mixed = idx.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) % label_count;
            (mixed + 1) as f64
        })
        .collect();
    let values: Vec<f64> = (0..n)
        .map(|idx| {
            let x = idx as f64;
            (x * 0.011).sin() * 13.0 + (x * 0.0007).cos() * 17.0
        })
        .collect();
    let input = NdArray::new(values, vec![side, side]).expect("label mean input");
    let labels = NdArray::new(labels, vec![side, side]).expect("label mean labels");
    let index = (1..=label_count).collect();
    (input, labels, index)
}

fn bench_label_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_mean");
    for &(side, label_count) in &[(256usize, 512usize), (512, 1024), (512, 2048), (768, 4096)] {
        let (input, labels, index) = label_mean_case(side, label_count);
        group.bench_function(
            BenchmarkId::new("one_based", format!("n{}_k{}", side * side, label_count)),
            |b| b.iter(|| mean(black_box(&input), Some(&labels), Some(&index)).expect("mean")),
        );
    }
    group.finish();
}

fn watershed_case(side: usize, marker_count: usize) -> (NdArray, NdArray) {
    let n = side * side;
    let costs: Vec<f64> = (0..n)
        .map(|idx| {
            let row = idx / side;
            let col = idx % side;
            ((row.wrapping_mul(17) + col.wrapping_mul(31) + idx.wrapping_mul(7)) & 255) as f64
        })
        .collect();
    let mut markers = vec![0.0; n];
    let mut state = 0x9E37_79B9usize;
    for label in 1..=marker_count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut idx = state % n;
        while markers[idx] != 0.0 {
            idx = (idx + 1) % n;
        }
        markers[idx] = label as f64;
    }
    (
        NdArray::new(costs, vec![side, side]).expect("watershed input"),
        NdArray::new(markers, vec![side, side]).expect("watershed markers"),
    )
}

fn bench_watershed_ift(c: &mut Criterion) {
    let (input, markers) = watershed_case(512, 50);
    c.benchmark_group("watershed_ift")
        .bench_function("uint8_512x512_m50", |b| {
            b.iter(|| watershed_ift(black_box(&input), black_box(&markers), None).expect("ift"))
        });
}

fn diff_array_clone_reference(input: &NdArray) -> NdArray {
    let data: Vec<f64> = input
        .data
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    NdArray {
        data: data.clone(),
        shape: vec![data.len()],
        strides: vec![1],
    }
}

fn bench_diff_array_ownership(c: &mut Criterion) {
    let n = 1usize << 20;
    let data: Vec<f64> = (0..n)
        .map(|idx| {
            let x = idx as f64;
            (x * 0.0017).sin() * 31.0 + (x * 0.00013).cos() * 17.0
        })
        .collect();
    let input = NdArray::new(data, vec![n]).expect("diff input");
    let mut group = c.benchmark_group("diff_array_ownership");
    group.bench_function("move_output", |b| {
        b.iter(|| black_box(diff_array(black_box(&input))))
    });
    group.bench_function("clone_reference", |b| {
        b.iter(|| black_box(diff_array_clone_reference(black_box(&input))))
    });
    group.finish();
}

/// A/B for the `convolve_axes` tap-delta / `nd_filter_apply` conversion
/// (frankenscipy-dn3i6 family): the old scalar path allocated `weights.unravel`
/// plus a fresh `in_idx` Vec per pixel per tap, while the new path precomputes
/// each tap's full-ndim delta once and gathers interior pixels straight from the
/// flat buffer. Both arms produce byte-identical output (see the bitwise test).
fn bench_convolve_axes_ab(c: &mut Criterion) {
    use fsci_ndimage::{convolve_axes, convolve_axes_scalar_reference};
    let img = image(256);
    let weights = NdArray::new(vec![1.0; 25], vec![5, 5]).expect("weights");
    let axes = [-2isize, -1];
    let mut group = c.benchmark_group("convolve_axes_ab");
    group.bench_function("tap_delta_5x5/256", |b| {
        b.iter(|| {
            convolve_axes(
                black_box(&img),
                black_box(&weights),
                black_box(&axes),
                BoundaryMode::Reflect,
                0.0,
            )
            .expect("convolve_axes")
        })
    });
    group.bench_function("scalar_reference_5x5/256", |b| {
        b.iter(|| {
            convolve_axes_scalar_reference(
                black_box(&img),
                black_box(&weights),
                black_box(&axes),
                BoundaryMode::Reflect,
                0.0,
            )
            .expect("convolve_axes_scalar_reference")
        })
    });
    group.finish();
}

/// A/B for the `correlate_axes` `nd_filter_apply` conversion (frankenscipy-dn3i6):
/// the scalar reference already precomputes `tap_delta` but still does a per-pixel
/// `unravel` + `in_idx` and routes every tap through `get_boundary`; the new path
/// gathers interior pixels straight from the flat buffer. Byte-identical output.
fn bench_correlate_axes_ab(c: &mut Criterion) {
    use fsci_ndimage::{correlate_axes, correlate_axes_scalar_reference};
    let img = image(256);
    let weights = NdArray::new(vec![1.0; 25], vec![5, 5]).expect("weights");
    let axes = [-2isize, -1];
    let mut group = c.benchmark_group("correlate_axes_ab");
    group.bench_function("nd_filter_5x5/256", |b| {
        b.iter(|| {
            correlate_axes(
                black_box(&img),
                black_box(&weights),
                black_box(&axes),
                BoundaryMode::Reflect,
                0.0,
            )
            .expect("correlate_axes")
        })
    });
    group.bench_function("scalar_reference_5x5/256", |b| {
        b.iter(|| {
            correlate_axes_scalar_reference(
                black_box(&img),
                black_box(&weights),
                black_box(&axes),
                BoundaryMode::Reflect,
                0.0,
            )
            .expect("correlate_axes_scalar_reference")
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_minmax_filter,
    bench_minmax_filter1d,
    bench_binary_morph,
    bench_rank_filter,
    bench_correlate_gaussian,
    bench_label_mean,
    bench_watershed_ift,
    bench_diff_array_ownership,
    bench_convolve_axes_ab,
    bench_correlate_axes_ab,
    bench_zoom,
    bench_rotate
);
criterion_main!(benches);

/// Geometric rotate — shares `sample_interpolated` with zoom, so it benefits from the
/// order=1 cardinal fast path (frankenscipy-wm14d). Head-to-head vs scipy.ndimage.rotate.
fn bench_rotate(c: &mut Criterion) {
    use fsci_ndimage::rotate;
    let img = image(256);
    let mut group = c.benchmark_group("rotate");
    for &order in &[1usize, 3] {
        group.bench_function(BenchmarkId::new("30deg_256", order), |b| {
            b.iter(|| {
                rotate(
                    black_box(&img),
                    30.0,
                    false,
                    order,
                    BoundaryMode::Reflect,
                    0.0,
                )
                .expect("rotate")
            })
        });
    }
    group.finish();
}

/// Geometric zoom (output-pixel loop parallelized). order=1 is cheap per pixel
/// (bilinear) — a regression candidate if the parallel gate is over-eager; order=3
/// is heavier (cubic spline). Head-to-head vs scipy.ndimage.zoom.
fn bench_zoom(c: &mut Criterion) {
    use fsci_ndimage::zoom;
    let img = image(256);
    let mut group = c.benchmark_group("zoom");
    for &order in &[1usize, 3] {
        group.bench_function(BenchmarkId::new("2x_256", order), |b| {
            b.iter(|| {
                zoom(
                    black_box(&img),
                    &[2.0, 2.0],
                    order,
                    BoundaryMode::Reflect,
                    0.0,
                )
                .expect("zoom")
            })
        });
    }
    group.finish();
}
