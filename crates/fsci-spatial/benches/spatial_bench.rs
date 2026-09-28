use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_spatial::{RigidTransform, Rotation};

/// Batch point-cloud transform vs mapping the scalar apply (the "original").
/// Quantifies the loop-invariant hoist: apply_many builds the rotation matrix
/// (and, for the rigid transform, the inverse rotation) ONCE instead of per point.
fn bench_transform_batch(c: &mut Criterion) {
    let n = 8192usize;
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64;
            [t * 0.001, (t * 0.7).sin(), (t * 0.3).cos()]
        })
        .collect();
    let mut group = c.benchmark_group("transform_batch");

    let r = Rotation::from_quat([
        0.022_260_026_714_733_816,
        0.439_679_739_540_909_55,
        0.360_423_405_650_355_9,
        0.822_363_171_905_999_4,
    ]);
    group.bench_function("rotation/apply_many", |b| b.iter(|| r.apply_many(&pts)));
    group.bench_function("rotation/map_apply", |b| {
        b.iter(|| pts.iter().map(|&p| r.apply(p)).collect::<Vec<_>>())
    });

    let tf = RigidTransform::from_components([1.0, 2.0, 3.0], r);
    group.bench_function("rigid/apply_many", |b| {
        b.iter(|| tf.apply_many(&pts, false))
    });
    group.bench_function("rigid/map_apply", |b| {
        b.iter(|| pts.iter().map(|&p| tf.apply(p, false)).collect::<Vec<_>>())
    });

    group.finish();
}

/// Validation of a large, valid dense distance matrix. This isolates the
/// full-input success path where every element must be inspected.
fn bench_distance_validation(c: &mut Criterion) {
    use fsci_spatial::is_valid_dm;
    use std::hint::black_box;

    let n = 4096usize;
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let value = ((i * 131 + j * 17) as f64 * 0.013).sin().abs();
            matrix[i][j] = value;
            matrix[j][i] = value;
        }
    }

    let mut group = c.benchmark_group("distance_validation");
    group.bench_function(BenchmarkId::new("is_valid_dm", "4096x4096"), |b| {
        b.iter(|| black_box(is_valid_dm(black_box(&matrix), black_box(0.0))))
    });
    group.finish();
}

/// All-pairs distance matrix — the dominant O(n²·d) spatial workload. Cosine/
/// Correlation exercise the per-vector precompute path; Euclidean the SIMD
/// partial-distance path. Both are parallel above a work threshold.
fn bench_pdist(c: &mut Criterion) {
    use fsci_spatial::{DistanceMetric, pdist};
    let mut group = c.benchmark_group("pdist");
    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64;
                vec![
                    (t * 0.1).sin(),
                    (t * 0.2).cos(),
                    t * 0.001,
                    (t * 0.05).sin(),
                ]
            })
            .collect();
        group.bench_function(BenchmarkId::new("euclidean", n), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Euclidean))
        });
        group.bench_function(BenchmarkId::new("chebyshev_repeat", n), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Chebyshev))
        });
        group.bench_function(BenchmarkId::new("cosine", n), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Cosine))
        });
        group.bench_function(BenchmarkId::new("chebyshev", n), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Chebyshev))
        });
    }
    group.finish();
}

/// Rectangular all-pairs distances for the small-d metrics that share the
/// `cdist_metric` generic path unless a metric-specific row kernel is present.
fn bench_cdist_small_metrics(c: &mut Criterion) {
    use fsci_spatial::{DistanceMetric, cdist_metric};
    let mut group = c.benchmark_group("cdist_small_metrics");
    for &d in &[3usize, 5, 7] {
        let xa: Vec<Vec<f64>> = (0..1000usize)
            .map(|i| {
                (0..d)
                    .map(|k| ((i * 97 + k * 51) as f64 * 0.017 + 0.3).sin())
                    .collect()
            })
            .collect();
        let xb: Vec<Vec<f64>> = (0..800usize)
            .map(|i| {
                (0..d)
                    .map(|k| ((i * 89 + k * 43) as f64 * 0.019 + 1.1).cos())
                    .collect()
            })
            .collect();
        for metric in [
            DistanceMetric::SqEuclidean,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
            DistanceMetric::Canberra,
            DistanceMetric::Braycurtis,
            DistanceMetric::Correlation,
        ] {
            group.bench_function(
                BenchmarkId::new(format!("{metric:?}"), format!("d{d}")),
                |b| b.iter(|| cdist_metric(&xa, &xb, metric)),
            );
        }
        // seuclidean has its own signature (variance vector V); benched here too.
        let vvar: Vec<f64> = (0..d).map(|k| 0.5 + (k as f64) * 0.3).collect();
        group.bench_function(BenchmarkId::new("Seuclidean", format!("d{d}")), |b| {
            b.iter(|| fsci_spatial::cdist_seuclidean(&xa, &xb, &vvar))
        });
    }
    group.finish();
}

/// cdist Hamming/Jaccard on high-dimensional binary data (the realistic regime for
/// these set-overlap metrics) — the generic per-pair arm vs a SoA-across-pairs count.
fn bench_cdist_boolean_metrics(c: &mut Criterion) {
    use fsci_spatial::{DistanceMetric, cdist_metric};
    let mut group = c.benchmark_group("cdist_boolean_metrics");
    for &d in &[64usize, 256] {
        // Deterministic pseudo-binary {0.0,1.0} data.
        let bit = |i: usize, k: usize| -> f64 {
            (((i.wrapping_mul(2654435761)
                .wrapping_add(k.wrapping_mul(40503)))
                >> 13)
                & 1) as f64
        };
        let xa: Vec<Vec<f64>> = (0..400usize)
            .map(|i| (0..d).map(|k| bit(i, k)).collect())
            .collect();
        let xb: Vec<Vec<f64>> = (0..400usize)
            .map(|i| (0..d).map(|k| bit(i + 7919, k * 3 + 1)).collect())
            .collect();
        for metric in [DistanceMetric::Hamming, DistanceMetric::Jaccard] {
            group.bench_function(
                BenchmarkId::new(format!("{metric:?}"), format!("d{d}")),
                |b| b.iter(|| cdist_metric(&xa, &xb, metric)),
            );
        }
    }
    group.finish();
}

/// KDTree build (O(n) median via select_nth) and nearest-neighbour query sweep.
fn bench_kdtree(c: &mut Criterion) {
    use fsci_spatial::KDTree;
    let n = 4096usize;
    let pt = |t: f64| vec![(t * 0.1).sin(), (t * 0.2).cos(), (t * 0.05).sin()];
    let data: Vec<Vec<f64>> = (0..n).map(|i| pt(i as f64)).collect();
    let mut group = c.benchmark_group("kdtree");
    group.bench_function("build/4096", |b| b.iter(|| KDTree::new(&data)));
    let tree = KDTree::new(&data).expect("kdtree");
    let queries: Vec<Vec<f64>> = (0..n).map(|i| pt(i as f64 + 0.5)).collect();
    group.bench_function("query/4096", |b| {
        b.iter(|| queries.iter().map(|q| tree.query(q)).collect::<Vec<_>>())
    });
    // Head-to-head with scipy.spatial.cKDTree.query(Q, k=1) at scipy's sizes.
    for &(nn, d) in &[(10000usize, 3usize), (10000, 8)] {
        let mk = |seed: f64| -> Vec<Vec<f64>> {
            (0..nn)
                .map(|i| {
                    (0..d)
                        .map(|k| ((i * 97 + k * 51) as f64 * 0.017 + seed).sin())
                        .collect()
                })
                .collect()
        };
        let pdata = mk(0.0);
        let qdata = mk(0.5);
        let t2 = KDTree::new(&pdata).expect("kdtree");
        group.bench_function(BenchmarkId::new("query_seq", format!("n{nn}_d{d}")), |b| {
            b.iter(|| qdata.iter().map(|q| t2.query(q)).collect::<Vec<_>>())
        });
        group.bench_function(BenchmarkId::new("query_many", format!("n{nn}_d{d}")), |b| {
            b.iter(|| t2.query_many(&qdata).expect("query_many"))
        });
        group.bench_function(
            BenchmarkId::new("query_k_seq", format!("n{nn}_d{d}_k10")),
            |b| b.iter(|| qdata.iter().map(|q| t2.query_k(q, 10)).collect::<Vec<_>>()),
        );
        group.bench_function(
            BenchmarkId::new("query_k_many", format!("n{nn}_d{d}_k10")),
            |b| b.iter(|| t2.query_k_many(&qdata, 10).expect("query_k_many")),
        );
        if d <= 3 {
            group.bench_function(
                BenchmarkId::new("ball_seq", format!("n{nn}_d{d}_r03")),
                |b| {
                    b.iter(|| {
                        qdata
                            .iter()
                            .map(|q| t2.query_ball_point(q, 0.3))
                            .collect::<Vec<_>>()
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new("ball_many", format!("n{nn}_d{d}_r03")),
                |b| b.iter(|| t2.query_ball_point_many(&qdata, 0.3).expect("ball_many")),
            );
        }
    }
    group.finish();
}

fn bench_sparse_dm(c: &mut Criterion) {
    use fsci_spatial::KDTree;
    let mut group = c.benchmark_group("sparse_distance_matrix");
    for &(n, r) in &[(5000usize, 0.05f64), (10000, 0.04)] {
        let mk = |seed: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    vec![
                        (t * 0.0137 + seed).sin() * 0.5 + 0.5,
                        (t * 0.0291 + seed).cos() * 0.5 + 0.5,
                    ]
                })
                .collect()
        };
        let a = KDTree::new(&mk(0.0)).expect("a");
        let b = KDTree::new(&mk(1.3)).expect("b");
        group.bench_function(BenchmarkId::new("sdm", n), |bn| {
            bn.iter(|| a.sparse_distance_matrix(&b, r).expect("sdm"))
        });
    }
    group.finish();
}

fn bench_find_simplex(c: &mut Criterion) {
    use fsci_spatial::Delaunay;
    let mut group = c.benchmark_group("find_simplex");
    for &npts in &[2000usize, 5000] {
        let pts: Vec<Vec<f64>> = (0..npts)
            .map(|i| {
                let t = i as f64;
                vec![(t * 0.137).sin() * 0.5 + 0.5, (t * 0.071).cos() * 0.5 + 0.5]
            })
            .collect();
        let tri = Delaunay::new(&pts).expect("delaunay");
        let q: Vec<Vec<f64>> = (0..50000)
            .map(|i| {
                let t = i as f64;
                vec![(t * 0.0191).fract(), (t * 0.0233).fract()]
            })
            .collect();
        group.bench_function(BenchmarkId::new("walk", npts), |b| {
            b.iter(|| tri.find_simplex(&q, false, None).expect("walk"))
        });
    }
    group.finish();
}

/// Delaunay triangulation (the lifted exact-predicate Quickhull) vs SciPy's Qhull
/// (docs/perf_oracle_delaunay.py).
fn bench_delaunay(c: &mut Criterion) {
    use fsci_spatial::Delaunay;
    let mut group = c.benchmark_group("delaunay");
    for &n in &[1000usize, 2000, 4000, 8000] {
        // Deterministic scattered 2-D points (low-discrepancy-ish, no exact duplicates).
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64;
                vec![
                    (t * 0.6180339887).fract() * 100.0,
                    (t * 0.4142135624).fract() * 100.0,
                ]
            })
            .collect();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| Delaunay::new(&pts))
        });
    }
    group.finish();
}

/// High-dimension pdist euclidean — head-to-head vs scipy.spatial.distance.pdist
/// at d >> 4 (the dim-4 SoA fast path does not apply here).
fn bench_pdist_highdim(c: &mut Criterion) {
    use fsci_spatial::{DistanceMetric, pdist};
    let mut group = c.benchmark_group("pdist_highdim");
    for &(n, d) in &[(1000usize, 64usize), (2000, 64), (1000, 128), (2000, 16)] {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|k| ((i * 131 + k * 17) as f64 * 0.013).sin() * 3.0 + (k as f64) * 0.1)
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::new("euclidean", format!("n{n}_d{d}")), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Euclidean))
        });
        group.bench_function(BenchmarkId::new("chebyshev", format!("n{n}_d{d}")), |b| {
            b.iter(|| pdist(&data, DistanceMetric::Chebyshev))
        });
    }
    group.finish();
}

/// SphericalVoronoi: the on-sphere 3-D convex hull (incremental insertion) +
/// region build. Probes the O(n^2) hull scan against scipy's Qhull O(n log n).
fn bench_spherical_voronoi(c: &mut Criterion) {
    use fsci_spatial::SphericalVoronoi;
    let mut group = c.benchmark_group("spherical_voronoi");
    group.sample_size(10);
    for &n in &[200usize, 1000, 4000] {
        // Deterministic quasi-uniform points on the unit sphere (golden-spiral).
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let k = i as f64 + 0.5;
                let phi = (1.0 - 2.0 * k / n as f64).acos();
                let theta = std::f64::consts::PI * (1.0 + 5.0f64.sqrt()) * k;
                [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]
            })
            .collect();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| SphericalVoronoi::new(&pts, [0.0, 0.0, 0.0], 1.0))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transform_batch,
    bench_distance_validation,
    bench_pdist,
    bench_cdist_small_metrics,
    bench_cdist_boolean_metrics,
    bench_pdist_highdim,
    bench_kdtree,
    bench_delaunay,
    bench_spherical_voronoi,
    bench_find_simplex,
    bench_sparse_dm
);
criterion_main!(benches);
