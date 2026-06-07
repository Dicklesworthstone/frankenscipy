use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_interpolate::{
    BarycentricInterpolator, CloughTocher2DInterpolator, CubicSplineStandalone, GriddataMethod,
    Interp1d, Interp1dOptions, InterpKind, LinearNDInterpolator, PchipInterpolator,
    RectBivariateSpline, RegularGridInterpolator, RegularGridMethod, SplineBc, griddata,
    interp1d_linear, lagrange, polymul, polyroots,
};
use fsci_runtime::RuntimeMode;

fn grid_1d(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn values_1d(xs: &[f64]) -> Vec<f64> {
    xs.iter()
        .map(|&x| (x * 11.0).sin() + 0.25 * (x * 7.0).cos())
        .collect()
}

fn query_1d(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            0.001 + 0.998 * t
        })
        .collect()
}

fn query_1d_unsorted(n: usize) -> Vec<f64> {
    let mut xs = query_1d(n);
    xs.reverse();
    xs
}

fn points_2d(side: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut points = Vec::with_capacity(side * side);
    let mut values = Vec::with_capacity(side * side);
    for iy in 0..side {
        for ix in 0..side {
            let x = (ix as f64 + 0.3 * (iy % 3) as f64) / side as f64;
            let y = (iy as f64 + 0.2 * (ix % 5) as f64) / side as f64;
            points.push(vec![x, y]);
            values.push((x * 5.0).sin() + (y * 3.0).cos());
        }
    }
    (points, values)
}

fn queries_2d(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = ((i * 37) % 997) as f64 / 997.0;
            let y = ((i * 53 + 17) % 991) as f64 / 991.0;
            vec![x, y]
        })
        .collect()
}

fn rect_grid(side: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let x = grid_1d(side);
    let y = grid_1d(side);
    let z = x
        .iter()
        .map(|&xi| {
            y.iter()
                .map(|&yi| (xi * 6.0).sin() * (yi * 4.0).cos())
                .collect()
        })
        .collect();
    (x, y, z)
}

fn regular_grid_values(points: &[Vec<f64>]) -> Vec<f64> {
    let nx = points[0].len();
    let ny = points[1].len();
    let nz = points[2].len();
    let mut values = Vec::with_capacity(nx * ny * nz);
    for &x in &points[0] {
        for &y in &points[1] {
            for &z in &points[2] {
                values.push((x * 5.0).sin() + (y * 3.0).cos() + z * z);
            }
        }
    }
    values
}

fn bench_interp1d(c: &mut Criterion) {
    let x = grid_1d(4096);
    let y = values_1d(&x);
    let sorted = query_1d(8192);
    let unsorted = query_1d_unsorted(8192);
    let interp = Interp1d::new(
        &x,
        &y,
        Interp1dOptions {
            kind: InterpKind::Linear,
            mode: RuntimeMode::Strict,
            fill_value: None,
            bounds_error: false,
            spline_bc: SplineBc::Natural,
        },
    )
    .expect("linear interpolator");

    let mut group = c.benchmark_group("interp1d");
    group.bench_function("linear_sorted_eval_many/4096x8192", |b| {
        b.iter(|| interp.eval_many(&sorted).expect("sorted eval"))
    });
    group.bench_function("linear_unsorted_eval_many/4096x8192", |b| {
        b.iter(|| interp.eval_many(&unsorted).expect("unsorted eval"))
    });
    group.bench_function("linear_one_shot/4096x8192", |b| {
        b.iter(|| interp1d_linear(&x, &y, &sorted).expect("one-shot linear"))
    });
    group.finish();
}

fn bench_splines(c: &mut Criterion) {
    let x = grid_1d(1024);
    let y = values_1d(&x);
    let x_new = query_1d(4096);
    let cubic = CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic spline");
    let pchip = PchipInterpolator::new(&x, &y).expect("pchip");

    let mut group = c.benchmark_group("splines");
    group.bench_function("cubic_eval_many/1024x4096", |b| {
        b.iter(|| cubic.eval_many(&x_new))
    });
    group.bench_function("pchip_eval_many/1024x4096", |b| {
        b.iter(|| pchip.eval_many(&x_new))
    });
    group.bench_function("cubic_construct/1024", |b| {
        b.iter(|| CubicSplineStandalone::new(&x, &y, SplineBc::Natural).expect("cubic construct"))
    });
    group.finish();
}

fn bench_polynomial(c: &mut Criterion) {
    let x = grid_1d(128);
    let y = values_1d(&x);
    let bary = BarycentricInterpolator::new(&x, &y).expect("barycentric");
    let x_new = query_1d(2048);
    let a = values_1d(&grid_1d(512));
    let b = values_1d(&grid_1d(512));
    let roots_coeffs = vec![1.0, -2.5, 1.1, -0.25, 0.03];

    let mut group = c.benchmark_group("polynomial");
    group.bench_function("barycentric_eval_many/128x2048", |b| {
        b.iter(|| bary.eval_many(&x_new))
    });
    group.bench_function("lagrange_construct/128", |b| {
        b.iter(|| lagrange(&x, &y).expect("lagrange"))
    });
    group.bench_function("polymul/512x512", |bench| bench.iter(|| polymul(&a, &b)));
    group.bench_function("polyroots/degree4", |b| b.iter(|| polyroots(&roots_coeffs)));
    group.finish();
}

fn bench_regular_grid(c: &mut Criterion) {
    let points = vec![grid_1d(32), grid_1d(32), grid_1d(16)];
    let values = regular_grid_values(&points);
    let queries = queries_2d(4096)
        .into_iter()
        .map(|q| vec![q[0], q[1], (q[0] * 0.7 + q[1] * 0.3).fract()])
        .collect::<Vec<_>>();
    let linear = RegularGridInterpolator::new(
        points.clone(),
        values.clone(),
        RegularGridMethod::Linear,
        false,
        None,
    )
    .expect("regular linear");
    let nearest =
        RegularGridInterpolator::new(points, values, RegularGridMethod::Nearest, false, None)
            .expect("regular nearest");

    let mut group = c.benchmark_group("regular_grid");
    group.bench_function("linear_eval_many/32x32x16_4096", |b| {
        b.iter(|| linear.eval_many(&queries).expect("linear grid"))
    });
    group.bench_function("nearest_eval_many/32x32x16_4096", |b| {
        b.iter(|| nearest.eval_many(&queries).expect("nearest grid"))
    });
    group.finish();
}

fn bench_scattered(c: &mut Criterion) {
    let (points, values) = points_2d(24);
    let queries = queries_2d(1024);
    let linear = LinearNDInterpolator::new(&points, &values).expect("linear nd");
    let clough = CloughTocher2DInterpolator::new(&points, &values).expect("clough-tocher");

    let mut group = c.benchmark_group("scattered_2d");
    group.bench_function("linear_nd_eval_many/576x1024", |b| {
        b.iter(|| linear.eval_many(&queries).expect("linear nd eval"))
    });
    group.bench_function("clough_tocher_eval_many/576x1024", |b| {
        b.iter(|| clough.eval_many(&queries).expect("clough eval"))
    });
    group.bench_function("griddata_linear/576x1024", |b| {
        b.iter(|| griddata(&points, &values, &queries, GriddataMethod::Linear).expect("griddata"))
    });
    group.finish();
}

fn bench_rbf_and_rect(c: &mut Criterion) {
    let (x, y, z) = rect_grid(32);
    let rect = RectBivariateSpline::new(&x, &y, &z, 3, 3).expect("rect bivariate");
    let xi = query_1d(64);
    let yi = query_1d(64);

    let mut group = c.benchmark_group("rbf_rect");
    group.bench_function("rect_eval_grid/32x32_to_64x64", |b| {
        b.iter(|| rect.eval_grid(&xi, &yi))
    });
    group.bench_function(BenchmarkId::new("rect_integral", "32x32"), |b| {
        b.iter(|| rect.integral(0.05, 0.95, 0.05, 0.95))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_interp1d,
    bench_splines,
    bench_polynomial,
    bench_regular_grid,
    bench_scattered,
    bench_rbf_and_rect
);
criterion_main!(benches);
