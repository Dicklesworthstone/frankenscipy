use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_cluster::{LinkageMethod, gaussian_mixture, kmeans, kmeans2, linkage, vq};
use std::hint::black_box;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Deterministic blobs: `n` points in `d` dims drawn around 4 cluster centres.
fn blobs(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let centre = (i % 4) as f64;
            (0..d)
                .map(|j| {
                    let t = (i * (j + 1)) as f64;
                    centre * 5.0 + (t * 0.013).sin() * 0.5 + ((i + j) % 7) as f64 * 0.05
                })
                .collect()
        })
        .collect()
}

fn sq_dist_bench(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn legacy_agglo_nearest(
    inter_dist: &[Vec<f64>],
    active: &[bool],
    i: usize,
    total: usize,
) -> (usize, f64) {
    let mut best_j = i;
    let mut best_d = f64::INFINITY;
    for j in (i + 1)..total {
        if active[j] && inter_dist[i][j] < best_d {
            best_d = inter_dist[i][j];
            best_j = j;
        }
    }
    (best_j, best_d)
}

fn legacy_agglomerate_nnarray(
    n: usize,
    mut inter_dist: Vec<Vec<f64>>,
    method: LinkageMethod,
) -> Vec<[f64; 4]> {
    let total = 2 * n - 1;
    let mut active = vec![false; total];
    active[..n].fill(true);
    let mut cluster_size = vec![1usize; total];
    let mut nn = vec![0usize; total];
    let mut d_nn = vec![f64::INFINITY; total];
    for i in 0..n {
        let (j, d) = legacy_agglo_nearest(&inter_dist, &active, i, total);
        nn[i] = j;
        d_nn[i] = d;
    }

    let mut result = Vec::with_capacity(n - 1);
    for step in 0..n - 1 {
        let new_id = n + step;

        let mut min_d = f64::INFINITY;
        let mut mi = 0;
        for i in 0..new_id {
            if active[i] && d_nn[i] < min_d {
                min_d = d_nn[i];
                mi = i;
            }
        }
        let mj = nn[mi];

        let new_size = cluster_size[mi] + cluster_size[mj];
        result.push([mi as f64, mj as f64, min_d, new_size as f64]);

        active[mi] = false;
        active[mj] = false;
        active[new_id] = true;
        cluster_size[new_id] = new_size;

        for k in 0..new_id {
            if !active[k] {
                continue;
            }
            let d_ki = inter_dist[k][mi];
            let d_kj = inter_dist[k][mj];
            let new_dist = match method {
                LinkageMethod::Single => d_ki.min(d_kj),
                LinkageMethod::Complete => d_ki.max(d_kj),
                LinkageMethod::Average => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    (ni * d_ki + nj * d_kj) / (ni + nj)
                }
                LinkageMethod::Ward => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nk = cluster_size[k] as f64;
                    let nt = ni + nj + nk;
                    (((nk + ni) * d_ki * d_ki + (nk + nj) * d_kj * d_kj - nk * min_d * min_d) / nt)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Weighted => 0.5 * (d_ki + d_kj),
                LinkageMethod::Centroid => {
                    let ni = cluster_size[mi] as f64;
                    let nj = cluster_size[mj] as f64;
                    let nt = ni + nj;
                    let alpha_i = ni / nt;
                    let alpha_j = nj / nt;
                    let beta = -(ni * nj) / (nt * nt);
                    (alpha_i * d_ki * d_ki + alpha_j * d_kj * d_kj + beta * min_d * min_d)
                        .max(0.0)
                        .sqrt()
                }
                LinkageMethod::Median => (0.5 * d_ki * d_ki + 0.5 * d_kj * d_kj
                    - 0.25 * min_d * min_d)
                    .max(0.0)
                    .sqrt(),
            };
            inter_dist[k][new_id] = new_dist;
            inter_dist[new_id][k] = new_dist;
        }

        d_nn[new_id] = f64::INFINITY;
        nn[new_id] = new_id;

        for k in 0..new_id {
            if !active[k] {
                continue;
            }
            if nn[k] == mi || nn[k] == mj {
                let (j, d) = legacy_agglo_nearest(&inter_dist, &active, k, total);
                nn[k] = j;
                d_nn[k] = d;
            } else if inter_dist[k][new_id] < d_nn[k] {
                d_nn[k] = inter_dist[k][new_id];
                nn[k] = new_id;
            }
        }
    }

    result
}

fn legacy_linkage_nested(data: &[Vec<f64>], method: LinkageMethod) -> Vec<[f64; 4]> {
    let n = data.len();
    let total = 2 * n - 1;
    let mut inter_dist = vec![vec![f64::INFINITY; total]; total];
    for i in 0..n {
        inter_dist[i][i] = 0.0;
        for j in i + 1..n {
            let d = sq_dist_bench(&data[i], &data[j]).sqrt();
            inter_dist[i][j] = d;
            inter_dist[j][i] = d;
        }
    }
    legacy_agglomerate_nnarray(n, inter_dist, method)
}

/// Compare the production linkage against the legacy nested baseline, and
/// anchor production to the INCUMBENT.
///
/// This used to demand bit-identity with `legacy_linkage_nested` and had been
/// failing for weeks (frankenscipy-r4cis). The disagreement was settled by
/// measurement, not preference. The fixture is a closed form with no RNG, so it
/// replicates exactly in Python; running `scipy.cluster.hierarchy.linkage` on it
/// with `method="average"`:
///
///   row 352: SciPy = 4596322736578931239 = production;  legacy is 1 ULP below
///   row 406: SciPy = 0x3fcbc9f13543281a  = production;  legacy is 2 ULP below
///
/// In both disputed rows SciPy is bit-equal to PRODUCTION and the legacy nested
/// baseline is the side that is wrong, by a drift that grows with the row. So
/// bit-identity to the baseline was the wrong contract twice over: it pinned a
/// SciPy-exact implementation to a less accurate legacy artifact, and it would
/// have reported any future move TOWARD SciPy as a regression. The baseline is
/// here to be TIMED, not to adjudicate correctness — and widening the bound
/// until it passes would just be choosing a tolerance that lets a wrong oracle
/// win.
///
/// What replaces it is stricter where correctness lives and honest where it
/// does not:
///   * merge STRUCTURE — which clusters join, in what order, with what member
///     counts — must match the baseline EXACTLY. A different dendrogram is a
///     real defect and still fails.
///   * merge DISTANCES are compared to the baseline with a relative tolerance,
///     because the baseline is not an oracle for them.
///   * production is pinned BIT-EXACTLY to the two SciPy values above, so
///     production drifting away from the incumbent fails immediately — which
///     is the check the old assertion only appeared to provide.
const SCIPY_AVERAGE_ANCHORS: [(usize, u64); 2] = [
    (352, 4_596_322_736_578_931_239),
    (406, 0x3fcb_c9f1_3543_281a),
];

fn assert_linkage_matches_baseline(left: &[[f64; 4]], right: &[[f64; 4]]) {
    assert_eq!(left.len(), right.len(), "linkage row count");
    for (row, (a, b)) in left.iter().zip(right).enumerate() {
        for column in [0usize, 1, 3] {
            assert_eq!(
                a[column].to_bits(),
                b[column].to_bits(),
                "linkage row {row} column {column}: merge structure differs ({} vs {})",
                a[column],
                b[column]
            );
        }
        let (x, y) = (a[2], b[2]);
        assert!(
            x.is_finite() && y.is_finite(),
            "linkage row {row} distance is not finite ({x} vs {y})"
        );
        let scale = x.abs().max(y.abs()).max(f64::MIN_POSITIVE);
        let relative = (x - y).abs() / scale;
        assert!(
            relative <= 1.0e-12,
            "linkage row {row} distance drifted {relative:.3e} from the baseline \
             ({x} vs {y}); production is the SciPy-exact side, so a drift this \
             large means production moved"
        );
    }
}

/// Pin production to SciPy on the rows where the legacy baseline was measured to
/// be wrong. This is the real oracle check (frankenscipy-r4cis).
fn assert_scipy_average_anchors(current: &[[f64; 4]]) {
    for (row, bits) in SCIPY_AVERAGE_ANCHORS {
        assert!(
            row < current.len(),
            "anchor row {row} beyond linkage of {} rows",
            current.len()
        );
        assert_eq!(
            current[row][2].to_bits(),
            bits,
            "row {row}: production {} must be bit-equal to scipy.cluster.hierarchy.linkage \
             (method=average) on this fixture",
            current[row][2]
        );
    }
}

fn scipy_linkage_duration(n: usize, d: usize, method: &str, iters: u64) -> Duration {
    let script = r#"
import math
import sys
import time
import numpy as np
from scipy.cluster.hierarchy import linkage

n = int(sys.argv[1])
d = int(sys.argv[2])
method = sys.argv[3]
iters = int(sys.argv[4])
data = np.array([
    [
        (i % 4) * 5.0 + math.sin(i * (j + 1) * 0.013) * 0.5 + ((i + j) % 7) * 0.05
        for j in range(d)
    ]
    for i in range(n)
], dtype=np.float64)

start = time.perf_counter()
acc = 0.0
for _ in range(iters):
    z = linkage(data, method=method)
    acc += float(z[0, 2])
elapsed = time.perf_counter() - start
print(f"{elapsed:.17g} {acc:.17g}")
"#;
    let mut child = Command::new("python3")
        .arg("-")
        .arg(n.to_string())
        .arg(d.to_string())
        .arg(method)
        .arg(iters.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn SciPy linkage oracle");
    child
        .stdin
        .as_mut()
        .expect("SciPy stdin")
        .write_all(script.as_bytes())
        .expect("write SciPy linkage oracle script");
    let output = child.wait_with_output().expect("run SciPy linkage oracle");
    assert!(
        output.status.success(),
        "SciPy linkage oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("SciPy stdout is utf8");
    let seconds: f64 = stdout
        .split_whitespace()
        .next()
        .expect("SciPy elapsed seconds")
        .parse()
        .expect("parse SciPy elapsed seconds");
    Duration::from_secs_f64(seconds)
}

/// kmeans (Lloyd) and kmeans2 (double-buffered Lloyd loop, frankenscipy-4ylee).
fn legacy_kmeans2_vq(
    data: &[Vec<f64>],
    init: &[Vec<f64>],
    iter: usize,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let d = data[0].len();
    let nc = init.len();
    let mut code_book = init.to_vec();
    let mut label = vec![0usize; data.len()];
    let mut sums = vec![vec![0.0_f64; d]; nc];
    let mut counts = vec![0usize; nc];
    let mut next_cb = code_book.clone();
    for _ in 0..iter {
        label = vq(data, &code_book).expect("legacy vq kmeans2").0;
        for row in sums.iter_mut() {
            row.iter_mut().for_each(|x| *x = 0.0);
        }
        counts.iter_mut().for_each(|x| *x = 0);
        for (i, &lab) in label.iter().enumerate() {
            counts[lab] += 1;
            for c in 0..d {
                sums[lab][c] += data[i][c];
            }
        }
        for j in 0..nc {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f64;
                for c in 0..d {
                    next_cb[j][c] = sums[j][c] * inv;
                }
            } else {
                next_cb[j].clone_from(&code_book[j]);
            }
        }
        std::mem::swap(&mut code_book, &mut next_cb);
    }
    (code_book, label)
}

fn bench_kmeans(c: &mut Criterion) {
    let data = blobs(2000, 4);
    let init: Vec<Vec<f64>> = (0..4).map(|k| vec![k as f64 * 5.0; 4]).collect();
    let mut group = c.benchmark_group("kmeans");
    group.bench_function("k4/n2000", |b| b.iter(|| kmeans(&data, 4, 50, 42)));
    group.bench_function("kmeans2_legacy_vq/k4/n2000", |b| {
        b.iter(|| legacy_kmeans2_vq(&data, &init, 50))
    });
    group.bench_function("kmeans2/k4/n2000", |b| b.iter(|| kmeans2(&data, &init, 50)));
    group.finish();
}

/// Gaussian-mixture EM. n=1000 is below the E-step work-gate (serial path);
/// n=5000/20000 exercise the parallel E-step (frankenscipy-yw7ts). Sizes mirror
/// docs/perf_oracle_gmm.py for the sklearn head-to-head.
fn bench_gmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gmm");
    for &(n, d, k) in &[(1000usize, 3usize, 3usize), (5000, 8, 5), (20000, 16, 8)] {
        let data = blobs(n, d);
        group.bench_function(BenchmarkId::new("diag", format!("n{n}_d{d}_k{k}")), |b| {
            b.iter(|| gaussian_mixture(&data, k, 50, 1e-4, 1e-6, 42))
        });
    }
    group.finish();
}

/// Hierarchical clustering: NN-chain linkage + cophenetic distances (the cophenet
/// member-list move-instead-of-clone win, frankenscipy-jphzn).
fn bench_hierarchical(c: &mut Criterion) {
    use fsci_cluster::cophenet;
    let data = blobs(400, 4);
    let z = linkage(&data, LinkageMethod::Average).expect("linkage");
    let mut group = c.benchmark_group("hierarchical");
    group.bench_function("linkage_average/n400", |b| {
        b.iter(|| linkage(&data, LinkageMethod::Average))
    });
    group.bench_function("cophenet/n400", |b| b.iter(|| cophenet(&z)));
    group.finish();
}

/// frankenscipy-va60h gauntlet: current flat row-major linkage arena against
/// the pre-optimization nested-row NN-array route and the original SciPy API.
fn bench_va60h_linkage_gauntlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("va60h_gauntlet_linkage");
    group.sample_size(10);

    for &(method, method_name) in &[
        (LinkageMethod::Average, "average"),
        (LinkageMethod::Ward, "ward"),
    ] {
        let n = 800usize;
        let d = 4usize;
        let data = blobs(n, d);
        let current = linkage(&data, method).expect("current linkage");
        let legacy = legacy_linkage_nested(&data, method);
        assert_linkage_matches_baseline(&current, &legacy);
        if matches!(method, LinkageMethod::Average) {
            assert_scipy_average_anchors(&current);
        }

        let workload = format!("{method_name}_n{n}_d{d}");
        group.bench_function(BenchmarkId::new("rust_current_flat", &workload), |b| {
            b.iter(|| black_box(linkage(black_box(&data), method).expect("linkage")))
        });
        group.bench_function(BenchmarkId::new("rust_legacy_nested", &workload), |b| {
            b.iter(|| black_box(legacy_linkage_nested(black_box(&data), method)))
        });
        group.bench_function(BenchmarkId::new("scipy_original", &workload), |b| {
            b.iter_custom(|iters| scipy_linkage_duration(n, d, method_name, iters))
        });
    }

    group.finish();
}

/// Affinity propagation — the O(n²)-per-iteration responsibility/availability
/// message passing (responsibility update parallelized, frankenscipy-yw7ts).
fn bench_affinity_propagation(c: &mut Criterion) {
    use fsci_cluster::affinity_propagation;
    // n=300 is below the responsibility-update gate (n²<2¹⁸, serial); n=1000/2000
    // exercise the parallel update (frankenscipy-yw7ts). Mirrors docs/perf_oracle_ap.py.
    let mut group = c.benchmark_group("affinity_propagation");
    for &n in &[300usize, 1000, 2000] {
        let data = blobs(n, 4);
        let sim: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        -data[i]
                            .iter()
                            .zip(&data[j])
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| affinity_propagation(&sim, -50.0, 0.9, 80, 15))
        });
    }
    group.finish();
}

/// Silhouette score — O(n²) all-pairs, parallelized per-anchor. Small + large n to
/// check the parallel gate doesn't regress small inputs. Head-to-head vs
/// sklearn.metrics.silhouette_score (docs/perf_oracle_silhouette.py).
fn bench_silhouette(c: &mut Criterion) {
    use fsci_cluster::silhouette_score;
    let mut group = c.benchmark_group("silhouette");
    for &n in &[500usize, 2000] {
        let data = blobs(n, 4);
        let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| silhouette_score(&data, &labels))
        });
    }
    group.finish();
}

/// Same-binary A/B for fusing kmedoids' intra-cluster distance row-sum directly into
/// per-candidate cost accumulation (dropping the M×M matrix + its second O(M²) pass).
/// Byte-identical — the full result (labels/centroids/inertia) is asserted bit-equal
/// between the fused and matrix arms before timing.
fn bench_kmedoids_fuse_ab(c: &mut Criterion) {
    use fsci_cluster::{KMEDOIDS_COST_FUSE_DISABLE, kmedoids};
    use std::sync::atomic::Ordering;
    let mut group = c.benchmark_group("kmedoids_fuse_ab");
    for &(n, k, d) in &[(2000usize, 4usize, 4usize), (3000, 6, 8)] {
        let data = blobs(n, d);
        KMEDOIDS_COST_FUSE_DISABLE.store(false, Ordering::Relaxed);
        let f = kmedoids(&data, k, 50, 42).expect("kmedoids fused");
        KMEDOIDS_COST_FUSE_DISABLE.store(true, Ordering::Relaxed);
        let o = kmedoids(&data, k, 50, 42).expect("kmedoids orig");
        assert_eq!(
            f.labels, o.labels,
            "kmedoids fuse labels mismatch (n={n} k={k})"
        );
        assert_eq!(
            f.inertia.to_bits(),
            o.inertia.to_bits(),
            "kmedoids fuse inertia mismatch"
        );
        assert!(
            f.centroids
                .iter()
                .zip(&o.centroids)
                .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())),
            "kmedoids fuse centroids not byte-identical (n={n} k={k})"
        );
        group.bench_function(
            BenchmarkId::new("current_fused", format!("n{n}_k{k}_d{d}")),
            |b| {
                b.iter(|| {
                    KMEDOIDS_COST_FUSE_DISABLE.store(false, Ordering::Relaxed);
                    kmedoids(&data, k, 50, 42).expect("km")
                })
            },
        );
        group.bench_function(
            BenchmarkId::new("orig_matrix", format!("n{n}_k{k}_d{d}")),
            |b| {
                b.iter(|| {
                    KMEDOIDS_COST_FUSE_DISABLE.store(true, Ordering::Relaxed);
                    kmedoids(&data, k, 50, 42).expect("km")
                })
            },
        );
    }
    KMEDOIDS_COST_FUSE_DISABLE.store(false, Ordering::Relaxed);
    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans,
    bench_kmedoids_fuse_ab,
    bench_gmm,
    bench_hierarchical,
    bench_va60h_linkage_gauntlet,
    bench_affinity_propagation,
    bench_silhouette
);
criterion_main!(benches);
