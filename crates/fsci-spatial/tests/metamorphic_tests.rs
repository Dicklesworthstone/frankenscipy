//! Metamorphic tests for `fsci-spatial`.
//!
//! Each relation is oracle-free: distance symmetry, identity of
//! indiscernibles, triangle inequality, pdist/squareform round-trip,
//! cdist symmetry, and KDTree nearest-distance correctness.
//!
//! Run with: `cargo test -p fsci-spatial --test metamorphic_tests`

use fsci_spatial::{
    ConvexHull, Delaunay, DistanceMetric, KDTree, Rotation, Voronoi, angle_between,
    cartesian_to_cylindrical, cartesian_to_spherical, cdist_metric, centroid, chebyshev, cityblock,
    cosine, cross_3d, cylindrical_to_cartesian, diameter, dice, distance_matrix, dot, euclidean,
    geometric_slerp, hausdorff_distance, jensenshannon, k_nearest_neighbors, kulsinski,
    mahalanobis, matching, medoid, metric_distance, minkowski, nearest_neighbors, normalize, pdist,
    procrustes, rogerstanimoto, rotate_point, rotation_matrix, russellrao, sokalmichener,
    sokalsneath, spherical_to_cartesian, spread, squareform_to_condensed, squareform_to_matrix,
    yule,
};

const ATOL: f64 = 1e-12;
const RTOL: f64 = 1e-10;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

/// 2-D points as the `(npoints, ndim)` rows the Qhull-equivalent types take.
fn as_rows(points: &[(f64, f64)]) -> Vec<Vec<f64>> {
    points.iter().map(|&(x, y)| vec![x, y]).collect()
}

const ALL_METRICS: &[DistanceMetric] = &[
    DistanceMetric::Euclidean,
    DistanceMetric::SqEuclidean,
    DistanceMetric::Cityblock,
    DistanceMetric::Chebyshev,
    DistanceMetric::Cosine,
    DistanceMetric::Correlation,
    DistanceMetric::Hamming,
    DistanceMetric::Jaccard,
    DistanceMetric::Canberra,
    DistanceMetric::Braycurtis,
];

fn sample_points() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 1.0, 2.0, 5.0],
        vec![-1.0, 0.0, 5.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
        vec![0.0, -2.0, 4.0, 1.0],
        vec![2.5, 1.5, 0.5, 3.5],
        vec![-3.0, 4.0, -2.0, 0.0],
    ]
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — distance symmetry: d(a, b) = d(b, a) for every metric
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_symmetric() {
    let pts = sample_points();
    for &metric in ALL_METRICS {
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                let dab = metric_distance(&pts[i], &pts[j], metric);
                let dba = metric_distance(&pts[j], &pts[i], metric);
                // Correlation/Cosine are undefined (NaN — as in scipy) for the
                // zero-variance constant vector; "both NaN" is still symmetric.
                assert!(
                    close(dab, dba) || (dab.is_nan() && dba.is_nan()),
                    "MR1 {metric:?} d({i},{j})={dab} d({j},{i})={dba}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — identity of indiscernibles: d(a, a) = 0 for every metric
//        (Hamming/Jaccard return 0 for identical points)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_self_zero() {
    let pts = sample_points();
    for &metric in ALL_METRICS {
        for (i, p) in pts.iter().enumerate() {
            let d = metric_distance(p, p, metric);
            // d(x,x)=0 holds only where the metric is defined. Correlation needs
            // non-zero variance and Cosine a non-zero norm; for the constant /
            // zero vectors in `sample_points` they are NaN — exactly as
            // scipy.spatial.distance returns — so accept NaN for those two.
            let undefined_here = d.is_nan()
                && matches!(metric, DistanceMetric::Correlation | DistanceMetric::Cosine);
            assert!(
                close(d, 0.0) || undefined_here,
                "MR2 {metric:?} d({i},{i}) = {d}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — triangle inequality for true metrics: d(a,c) ≤ d(a,b) + d(b,c)
//        Holds for Euclidean, Cityblock, Chebyshev (and Minkowski-p≥1)
//        but NOT for Correlation/Cosine/Bray-Curtis in general — those
//        are excluded from this relation.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_triangle_inequality_metric_distances() {
    let pts = sample_points();
    let metrics = [
        DistanceMetric::Euclidean,
        DistanceMetric::Cityblock,
        DistanceMetric::Chebyshev,
    ];
    let n = pts.len();
    for &metric in &metrics {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let dab = metric_distance(&pts[i], &pts[j], metric);
                    let dbc = metric_distance(&pts[j], &pts[k], metric);
                    let dac = metric_distance(&pts[i], &pts[k], metric);
                    // Allow a small relative slack for f64 round-off when
                    // dac is close to the sum.
                    let bound = (dab + dbc) * (1.0 + 1e-12) + 1e-15;
                    assert!(
                        dac <= bound,
                        "MR3 {metric:?} triangle violated for ({i},{j},{k}): d_ac={dac} > d_ab+d_bc={}, bound={bound}",
                        dab + dbc
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — pdist ↔ squareform round-trip: condensed → square → condensed
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pdist_squareform_roundtrip() {
    let pts = sample_points();
    let condensed = pdist(&pts, DistanceMetric::Euclidean).unwrap();
    let square = squareform_to_matrix(&condensed).unwrap();
    let back = squareform_to_condensed(&square).unwrap();

    assert_eq!(back.len(), condensed.len(), "MR4 round-trip length");
    for (i, (got, want)) in back.iter().zip(&condensed).enumerate() {
        assert!(
            close(*got, *want),
            "MR4 round-trip element {i}: got={got} expected={want}"
        );
    }
    // Square matrix must be symmetric with zero diagonal.
    let n = pts.len();
    assert_eq!(square.len(), n);
    for (i, row) in square.iter().enumerate() {
        assert!((row[i]).abs() < 1e-15, "diagonal not zero at {i}");
        for j in (i + 1)..n {
            assert!(
                close(row[j], square[j][i]),
                "square not symmetric at ({i},{j})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — cdist(X, X) is symmetric and diagonal == 0 for true metrics.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_self_symmetric_zero_diagonal() {
    let pts = sample_points();
    let m = cdist_metric(&pts, &pts, DistanceMetric::Euclidean).unwrap();
    let n = pts.len();
    assert_eq!(m.len(), n);
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row.len(), n);
        assert!(
            row[i].abs() < 1e-12,
            "MR5 cdist diagonal not zero at {i}: {}",
            row[i]
        );
        for j in (i + 1)..n {
            assert!(
                close(row[j], m[j][i]),
                "MR5 cdist not symmetric at ({i},{j}): {} vs {}",
                row[j],
                m[j][i]
            );
            assert!(
                m[i][j] >= 0.0,
                "MR5 cdist negative at ({i},{j}): {}",
                m[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — KDTree.query returns the true Euclidean nearest neighbor.
//
// Build a tree, query each input point — the nearest must be the
// point itself (distance 0). Then query a perturbed point and check
// its returned distance equals the brute-force minimum.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_matches_brute_force() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();

    // 1. Self-query: each input must find itself with distance 0.
    for (i, p) in pts.iter().enumerate() {
        let (idx, d) = tree.query(p).unwrap();
        assert_eq!(idx, i, "MR6 self-query: expected {i}, got {idx}");
        assert!(d.abs() < 1e-12, "MR6 self-query distance: {d}");
    }

    // 2. Perturbed-query: distance must equal the true nearest.
    let q = vec![1.5, 2.5, 2.5, 3.5];
    let (kd_idx, kd_dist) = tree.query(&q).unwrap();
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, p) in pts.iter().enumerate() {
        let d = euclidean(&q, p);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    assert_eq!(
        kd_idx, best_idx,
        "MR6 KDTree returned wrong index: {kd_idx} vs brute {best_idx}"
    );
    assert!(
        close(kd_dist, best_dist),
        "MR6 KDTree distance: {kd_dist} vs brute {best_dist}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — KDTree.query_ball_point returns *all* points within radius r.
//
// Brute-force the ball and verify both returned-vs-true sets agree
// after sorting.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_ball_point_matches_brute_force() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    let q = vec![1.5, 2.5, 2.5, 3.5];
    let r = 4.0;

    let mut from_tree = tree.query_ball_point(&q, r).unwrap();
    let mut from_brute: Vec<usize> = pts
        .iter()
        .enumerate()
        .filter(|(_, p)| euclidean(&q, p) <= r)
        .map(|(i, _)| i)
        .collect();

    from_tree.sort();
    from_brute.sort();
    assert_eq!(
        from_tree, from_brute,
        "MR7 KDTree ball: tree={from_tree:?} brute={from_brute:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — squared Euclidean equals Euclidean squared (per element).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sqeuclidean_equals_euclidean_squared() {
    let pts = sample_points();
    for i in 0..pts.len() {
        for j in 0..pts.len() {
            let e = metric_distance(&pts[i], &pts[j], DistanceMetric::Euclidean);
            let sq = metric_distance(&pts[i], &pts[j], DistanceMetric::SqEuclidean);
            assert!(
                close(sq, e * e),
                "MR8 sqeuclidean({i},{j})={sq} vs euclidean²={}",
                e * e
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — ConvexHull encloses every input point: each point must be on
// or inside the hull (signed distance from each hull edge ≥ −ε).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_encloses_all_inputs() {
    // 7 points forming a square plus interior points.
    let pts: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 3.0),
        (0.0, 3.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 1.5),
    ];
    let hull = ConvexHull::new(&as_rows(&pts)).unwrap();

    // Get hull vertices in CCW order.
    let hull_pts: Vec<(f64, f64)> = hull.vertices.iter().map(|&i| pts[i]).collect();
    let m = hull_pts.len();
    assert!(m >= 3, "convex hull must have at least 3 vertices");

    // For every input point, signed distance from each hull edge must be
    // ≥ 0 (point is on the same side as the interior, or on the edge).
    // For a CCW-oriented polygon, interior points yield positive cross
    // products for every edge.
    for (idx, &p) in pts.iter().enumerate() {
        for k in 0..m {
            let a = hull_pts[k];
            let b = hull_pts[(k + 1) % m];
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            assert!(
                cross >= -1e-12,
                "MR9 point {idx} ({p:?}) outside hull edge ({a:?} → {b:?}): cross={cross}"
            );
        }
    }

    // Hull area (SciPy's `volume` in 2-D) for the 4x3 rectangle must be 12.
    let expected_area = 12.0_f64;
    assert!(
        (hull.volume - expected_area).abs() < 1e-10,
        "MR9 hull area: {} vs {expected_area}",
        hull.volume
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — ConvexHull of a triangle returns the triangle (3 vertices)
// and the area of the triangle.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_triangle() {
    let pts = vec![(0.0_f64, 0.0), (4.0, 0.0), (0.0, 3.0)];
    let hull = ConvexHull::new(&as_rows(&pts)).unwrap();
    assert_eq!(
        hull.vertices.len(),
        3,
        "triangle should have 3 hull vertices"
    );
    let expected_area = 6.0_f64; // 1/2 · base · height = 1/2 · 4 · 3
    assert!(
        (hull.volume - expected_area).abs() < 1e-10,
        "MR10 triangle area: {} vs {expected_area}",
        hull.volume
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — ConvexHull is invariant under translation: shifting every
// point by (Δx, Δy) leaves area and the set of hull vertices unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_translation_invariance() {
    let base: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 4.0),
        (0.0, 4.0),
        (2.0, 2.0),
        (3.5, 1.0),
    ];
    let hull_a = ConvexHull::new(&as_rows(&base)).unwrap();
    let dx = 7.5_f64;
    let dy = -3.2_f64;
    let shifted: Vec<(f64, f64)> = base.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
    let hull_b = ConvexHull::new(&as_rows(&shifted)).unwrap();
    assert!(
        (hull_a.volume - hull_b.volume).abs() < 1e-10,
        "MR11 area changed under translation: {} vs {}",
        hull_a.volume,
        hull_b.volume
    );
    assert_eq!(
        hull_a.vertices.len(),
        hull_b.vertices.len(),
        "MR11 vertex count differs"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — KDTree.query_k returns k results sorted by ascending distance
// and contains the brute-force k-nearest set.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_k_sorted_and_correct() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    let q = vec![1.5_f64, 2.5, 2.5, 3.5];
    let k = 3;
    let result = tree.query_k(&q, k).unwrap();
    assert_eq!(result.len(), k, "MR12 query_k length");

    // Distances are non-decreasing.
    for w in result.windows(2) {
        assert!(
            w[0].1 <= w[1].1 + 1e-12,
            "MR12 distances not sorted ascending: {} > {}",
            w[0].1,
            w[1].1
        );
    }

    // Cross-check: sort all brute-force distances and compare top-k.
    let mut brute: Vec<(usize, f64)> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| (i, euclidean(&q, p)))
        .collect();
    brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for i in 0..k {
        assert!(
            close(result[i].1, brute[i].1),
            "MR12 distance #{i}: tree={} brute={}",
            result[i].1,
            brute[i].1
        );
        assert_eq!(
            result[i].0, brute[i].0,
            "MR12 index #{i}: tree={} brute={}",
            result[i].0, brute[i].0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — KDTree.size matches input length.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_size_matches_input() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).unwrap();
    assert_eq!(tree.size(), pts.len(), "MR13 KDTree.size mismatch");
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — cdist row count matches xa.len() and column count matches
// xb.len(); all entries are non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_dimensions_and_nonneg() {
    let xa = vec![vec![0.0_f64, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let xb = vec![vec![5.0_f64, 5.0], vec![5.0, 6.0]];
    let m = cdist_metric(&xa, &xb, DistanceMetric::Euclidean).unwrap();
    assert_eq!(m.len(), xa.len(), "MR14 cdist row count");
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row.len(), xb.len(), "MR14 cdist row {i} col count");
        for &v in row {
            assert!(v >= 0.0, "MR14 cdist negative entry: {v}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — cdist is translation-invariant: shifting both xa and xb by
// the same (Δ, Δ) leaves the distance matrix unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_translation_invariance() {
    let xa = vec![vec![0.0_f64, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let xb = vec![vec![5.0_f64, 5.0], vec![5.0, 6.0]];
    let m_base = cdist_metric(&xa, &xb, DistanceMetric::Euclidean).unwrap();

    let dx = 7.5_f64;
    let dy = -3.2_f64;
    let xa_t: Vec<Vec<f64>> = xa.iter().map(|p| vec![p[0] + dx, p[1] + dy]).collect();
    let xb_t: Vec<Vec<f64>> = xb.iter().map(|p| vec![p[0] + dx, p[1] + dy]).collect();
    let m_shift = cdist_metric(&xa_t, &xb_t, DistanceMetric::Euclidean).unwrap();
    for (r1, r2) in m_base.iter().zip(&m_shift) {
        for (a, b) in r1.iter().zip(r2) {
            assert!(close(*a, *b), "MR15 cdist translation: {a} vs {b}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — cdist between points and themselves equals pdist as a square
// matrix on the off-diagonal entries.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cdist_self_matches_pdist() {
    let pts = sample_points();
    let cd = cdist_metric(&pts, &pts, DistanceMetric::Euclidean).unwrap();
    let condensed = pdist(&pts, DistanceMetric::Euclidean).unwrap();
    let n = pts.len();
    let mut idx = 0;
    for (i, row) in cd.iter().enumerate() {
        for (j, from_cdist) in row.iter().enumerate().take(n).skip(i + 1) {
            let from_pdist = condensed[idx];
            assert!(
                close(*from_cdist, from_pdist),
                "MR16 cdist[{i},{j}]={from_cdist} pdist[{idx}]={from_pdist}"
            );
            idx += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — Minkowski with p = 2 equals Euclidean; with p = 1 equals
// Cityblock; with p → ∞ approaches Chebyshev.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minkowski_specializes_to_classic_metrics() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let mp2 = minkowski(a, b, 2.0);
            let euc = euclidean(a, b);
            assert!(
                close(mp2, euc),
                "MR17 minkowski(p=2) = {mp2} != euclidean = {euc}"
            );
            let mp1 = minkowski(a, b, 1.0);
            let cb = cityblock(a, b);
            assert!(
                close(mp1, cb),
                "MR17 minkowski(p=1) = {mp1} != cityblock = {cb}"
            );
            // p = 50 should be very close to Chebyshev (within 5% relative on
            // non-zero distances).
            let mp_hi = minkowski(a, b, 50.0);
            let cheb = chebyshev(a, b);
            if cheb > 1e-6 {
                let rel = (mp_hi - cheb).abs() / cheb;
                assert!(
                    rel < 0.05,
                    "MR17 minkowski(p=50) = {mp_hi} vs chebyshev = {cheb} (rel diff {rel})"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — chebyshev ≤ euclidean ≤ cityblock for any pair (in Rᵈ).
// (Standard p-norm inequality between p=∞, p=2, p=1.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_metric_norm_ordering() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let cheb = chebyshev(a, b);
            let euc = euclidean(a, b);
            let cb = cityblock(a, b);
            assert!(
                cheb <= euc + 1e-10,
                "MR18 chebyshev = {cheb} > euclidean = {euc}"
            );
            assert!(
                euc <= cb + 1e-10,
                "MR18 euclidean = {euc} > cityblock = {cb}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Cosine distance is non-negative and bounded above by 2 for
// any pair of non-zero vectors, and is 0 for v vs c·v with c > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cosine_distance_bounds_and_scale_invariance() {
    let pts = sample_points();
    for a in &pts {
        for b in &pts {
            let d = cosine(a, b);
            assert!(
                (-1e-12..=2.0 + 1e-12).contains(&d),
                "MR19 cosine = {d}, expected in [0, 2]"
            );
        }
        // a vs 3·a should be ≈ 0 (parallel).
        let scaled: Vec<f64> = a.iter().map(|&x| 3.0 * x).collect();
        if a.iter().any(|&x| x.abs() > 1e-9) {
            let d_par = cosine(a, &scaled);
            assert!(
                d_par.abs() < 1e-9,
                "MR19 cosine(a, 3a) = {d_par}, expected ≈ 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — Jensen-Shannon distance is symmetric and non-negative on two
// probability distributions.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_jensenshannon_symmetric_nonneg() {
    let dists = [
        vec![0.5_f64, 0.3, 0.2],
        vec![0.1_f64, 0.6, 0.3],
        vec![0.25_f64, 0.25, 0.5],
        vec![1.0_f64 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ];
    for p in &dists {
        for q in &dists {
            let pq = jensenshannon(p, q, None);
            let qp = jensenshannon(q, p, None);
            assert!(pq >= -1e-12, "MR20 JS = {pq} negative");
            assert!(close(pq, qp), "MR20 JS not symmetric: pq = {pq}, qp = {qp}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — procrustes(A, A) yields disparity ≈ 0 (Procrustes distance
// of a configuration to itself is zero).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_procrustes_self_disparity_zero() {
    let a: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
    ];
    let res = procrustes(&a, &a).expect("procrustes self");
    assert!(
        res.disparity < 1e-9,
        "MR21 procrustes(A, A) disparity = {}, expected ≈ 0",
        res.disparity
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — KDTree.query returns a self-match with distance 0 when
// querying a point already in the tree.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kdtree_query_self_match_distance_zero() {
    let pts = sample_points();
    let tree = KDTree::new(&pts).expect("kdtree");
    for (i, p) in pts.iter().enumerate() {
        let (idx, dist) = tree.query(p).expect("query");
        assert_eq!(idx, i, "MR22 self-match index for point {i}: got {idx}");
        assert!(
            dist < 1e-12,
            "MR22 self-match distance for point {i}: {dist}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — spherical→cartesian→spherical preserves r and yields the same
// (x, y, z) when round-tripped via cartesian_to_spherical.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spherical_cartesian_roundtrip() {
    use std::f64::consts::PI;
    let cases = [
        (1.0_f64, PI / 4.0, 0.5),
        (3.5, PI / 3.0, 1.2),
        (2.0, PI / 2.0, 0.0),
        (5.0, PI / 6.0, -1.0),
    ];
    for &(r, theta, phi) in &cases {
        let (x, y, z) = spherical_to_cartesian(r, theta, phi);
        let (r2, theta2, phi2) = cartesian_to_spherical(x, y, z);
        assert!((r - r2).abs() < 1e-12, "MR23 r mismatch: {r} vs {r2}");
        assert!(
            (theta - theta2).abs() < 1e-12,
            "MR23 theta mismatch: {theta} vs {theta2}"
        );
        // φ may differ by ±2π — compare via cos and sin.
        assert!(
            (phi.cos() - phi2.cos()).abs() < 1e-9 && (phi.sin() - phi2.sin()).abs() < 1e-9,
            "MR23 phi mismatch: {phi} vs {phi2}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — cylindrical→cartesian→cylindrical preserves ρ and z; θ
// matches modulo 2π.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cylindrical_cartesian_roundtrip() {
    use std::f64::consts::PI;
    let cases = [
        (1.0_f64, 0.0, 0.0),
        (2.5, PI / 4.0, 1.0),
        (3.0, PI / 2.0, -2.5),
        (0.5, PI, 5.0),
    ];
    for &(rho, theta, z) in &cases {
        let (x, y, zc) = cylindrical_to_cartesian(rho, theta, z);
        let (rho2, theta2, z2) = cartesian_to_cylindrical(x, y, zc);
        assert!(
            (rho - rho2).abs() < 1e-12,
            "MR24 rho mismatch: {rho} vs {rho2}"
        );
        assert!((z - z2).abs() < 1e-12, "MR24 z mismatch: {z} vs {z2}");
        assert!(
            (theta.cos() - theta2.cos()).abs() < 1e-9 && (theta.sin() - theta2.sin()).abs() < 1e-9,
            "MR24 theta mismatch: {theta} vs {theta2}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — A rotation about a unit axis preserves the L2 norm of any
// vector it is applied to.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_preserves_norm() {
    use std::f64::consts::PI;
    let axes: &[[f64; 3]] = &[
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [
            1.0 / (3.0_f64).sqrt(),
            1.0 / (3.0_f64).sqrt(),
            1.0 / (3.0_f64).sqrt(),
        ],
    ];
    let points: &[[f64; 3]] = &[[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0], [2.5, -3.0, 1.0]];
    for axis in axes {
        for angle in &[0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0, PI] {
            let r = rotation_matrix(axis, *angle);
            for p in points {
                let q = rotate_point(&r, p);
                let np: f64 = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let nq: f64 = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                assert!(
                    (np - nq).abs() < 1e-9,
                    "MR25 rotation broke norm: {np} vs {nq}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — cross_3d is antisymmetric: a × b = -(b × a).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cross_3d_antisymmetric() {
    let pairs: &[([f64; 3], [f64; 3])] = &[
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([-1.0, 0.5, 2.5], [3.0, -1.5, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    ];
    for (a, b) in pairs {
        let ab = cross_3d(a, b);
        let ba = cross_3d(b, a);
        for k in 0..3 {
            assert!(
                (ab[k] + ba[k]).abs() < 1e-12,
                "MR26 a×b + b×a [{k}] = {} != 0",
                ab[k] + ba[k]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — dot(a, a) = ‖a‖² and dot is symmetric in its arguments.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dot_self_norm_squared_and_symmetry() {
    let vectors = sample_points();
    for a in &vectors {
        let aa = dot(a, a);
        let norm_sq: f64 = a.iter().map(|&v| v * v).sum();
        assert!(
            (aa - norm_sq).abs() < 1e-9,
            "MR27 dot(a, a) = {aa} vs ‖a‖² = {norm_sq}"
        );
        for b in &vectors {
            let ab = dot(a, b);
            let ba = dot(b, a);
            assert!((ab - ba).abs() < 1e-12, "MR27 dot symmetry: {ab} vs {ba}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — angle_between is in [0, π] for any two non-zero vectors.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_angle_between_in_zero_pi() {
    use std::f64::consts::PI;
    let vectors = sample_points();
    for a in &vectors {
        for b in &vectors {
            let ang = angle_between(a, b);
            if ang.is_finite() {
                assert!(
                    (-1e-12..=PI + 1e-12).contains(&ang),
                    "MR28 angle_between = {ang} not in [0, π]"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — normalize produces a unit-length vector for any non-zero
// input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_normalize_unit_length() {
    let vectors = sample_points();
    for v in &vectors {
        let nv = normalize(v);
        let len: f64 = nv.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if v.iter().any(|&x| x.abs() > 1e-9) {
            assert!(
                (len - 1.0).abs() < 1e-9,
                "MR29 normalize length = {len}, expected 1"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — hausdorff_distance is symmetric: H(A, B) = H(B, A).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hausdorff_distance_symmetric() {
    let a: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let b: Vec<Vec<f64>> = vec![vec![3.0, 3.0], vec![3.5, 4.0], vec![4.0, 3.5]];
    let hab = hausdorff_distance(&a, &b).unwrap();
    let hba = hausdorff_distance(&b, &a).unwrap();
    assert!(
        (hab - hba).abs() < 1e-12,
        "MR30 hausdorff not symmetric: {hab} vs {hba}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — Yule and Dice boolean distances on identical inputs are 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_boolean_metric_self_zero() {
    let bs: &[Vec<bool>] = &[
        vec![true, false, true, false, true],
        vec![false, true, true, true, false],
        vec![true, true, true, true, true],
    ];
    for b in bs {
        // Yule of x with itself; dice of x with itself.
        let y = yule(b, b);
        let d = dice(b, b);
        assert!(
            y.abs() < 1e-12,
            "MR31 yule(x, x) = {y}, expected 0 on {b:?}"
        );
        assert!(
            d.abs() < 1e-12,
            "MR31 dice(x, x) = {d}, expected 0 on {b:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — centroid of n copies of a single point equals that point.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_centroid_of_constant_points() {
    let p = vec![1.5_f64, -2.5, 3.0];
    let pts: Vec<Vec<f64>> = (0..7).map(|_| p.clone()).collect();
    let c = centroid(&pts);
    assert_eq!(c.len(), p.len(), "MR32 centroid dim");
    for (i, (&a, &b)) in c.iter().zip(&p).enumerate() {
        assert!((a - b).abs() < 1e-12, "MR32 centroid[{i}] = {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — k_nearest_neighbors returns lists of length k for each point.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_k_nearest_neighbors_lengths() {
    let pts = sample_points();
    for k in 1..=3 {
        let (idx, dist) = k_nearest_neighbors(&pts, k);
        assert_eq!(idx.len(), pts.len(), "MR33 idx length");
        assert_eq!(dist.len(), pts.len(), "MR33 dist length");
        for (i, lst) in idx.iter().enumerate() {
            assert_eq!(lst.len(), k, "MR33 idx[{i}].len = {} vs k = {k}", lst.len());
        }
        for (i, lst) in dist.iter().enumerate() {
            assert_eq!(
                lst.len(),
                k,
                "MR33 dist[{i}].len = {} vs k = {k}",
                lst.len()
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — diameter (max pairwise distance) is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_diameter_nonneg() {
    let pts = sample_points();
    let d = diameter(&pts);
    assert!(d >= -1e-12, "MR34 diameter = {d} < 0");
    // Spread (min pairwise distance) ≤ diameter.
    if pts.len() >= 2 {
        let mut mn = f64::INFINITY;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let dij = euclidean(&pts[i], &pts[j]);
                if dij < mn {
                    mn = dij;
                }
            }
        }
        assert!(mn <= d + 1e-9, "MR34 min pairwise = {mn} > diameter = {d}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — matching(x, x) = 0 and rogerstanimoto(x, x) = 0 for any
// boolean vector.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_matching_rogers_self_zero() {
    let bs: &[Vec<bool>] = &[
        vec![true, false, true, false, true],
        vec![false, true, true, true, false],
        vec![true, true, true, true, true],
    ];
    for b in bs {
        let m = matching(b, b);
        let r = rogerstanimoto(b, b);
        assert!(m.abs() < 1e-12, "MR35 matching(x, x) = {m} on {b:?}");
        assert!(r.abs() < 1e-12, "MR35 rogerstanimoto(x, x) = {r} on {b:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — russellrao distance lies in [0, 1] for any pair of boolean
// vectors.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_russellrao_in_unit_interval() {
    let pairs: &[(Vec<bool>, Vec<bool>)] = &[
        (
            vec![true, false, true, false],
            vec![true, true, false, false],
        ),
        (vec![false; 8], vec![false; 8]),
        (vec![true; 8], vec![true; 8]),
        (
            vec![true, false, true, true, false, true, false],
            vec![false, true, false, true, true, false, true],
        ),
    ];
    for (u, v) in pairs {
        let r = russellrao(u, v);
        assert!(
            (-1e-12..=1.0 + 1e-12).contains(&r),
            "MR36 russellrao = {r} outside [0, 1]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — medoid returns Some(idx) with idx < n on a non-empty point
// set.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_medoid_index_in_range() {
    let pts = sample_points();
    let m = medoid(&pts);
    assert!(m.is_some(), "MR37 medoid returned None on non-empty input");
    let idx = m.unwrap();
    assert!(idx < pts.len(), "MR37 medoid index = {idx} ≥ {}", pts.len());
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — spread (minimum pairwise distance) ≤ diameter.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_spread_dominated_by_diameter() {
    let pts = sample_points();
    let s = spread(&pts);
    let d = diameter(&pts);
    assert!(s <= d + 1e-9, "MR38 spread = {s} > diameter = {d}");
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — nearest_neighbors returns vectors of length n; every assigned
// neighbour index is in 0..n and != self.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_nearest_neighbors_length_and_validity() {
    let pts = sample_points();
    let n = pts.len();
    let (nn, dists) = nearest_neighbors(&pts);
    assert_eq!(nn.len(), n, "MR39 nearest_neighbors length");
    assert_eq!(dists.len(), n, "MR39 nearest_neighbors dist length");
    for (i, &maybe_j) in nn.iter().enumerate() {
        if let Some(j) = maybe_j {
            assert!(j < n, "MR39 nearest_neighbors[{i}] = {j} ≥ {n}");
            assert_ne!(j, i, "MR39 nearest_neighbors[{i}] = self");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — Hausdorff distance is non-negative for any non-empty point sets.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_hausdorff_nonneg() {
    let a: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![0.0, 0.0]];
    let b: Vec<Vec<f64>> = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let d = hausdorff_distance(&a, &b).unwrap();
    assert!(d >= -1e-12, "MR40 hausdorff = {d} < 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — distance_matrix(x, y) has shape (len(x), len(y)).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_distance_matrix_shape() {
    let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, (i * 2) as f64]).collect();
    let y: Vec<Vec<f64>> = (0..3)
        .map(|j| vec![j as f64 + 0.5, (j * 3) as f64 - 1.0])
        .collect();
    let d = distance_matrix(&x, &y).unwrap();
    assert_eq!(d.len(), x.len(), "MR41 distance_matrix rows");
    for row in &d {
        assert_eq!(row.len(), y.len(), "MR41 distance_matrix cols");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — geometric_slerp at t = 0 returns start; at t = 1 returns end.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_geometric_slerp_endpoints() {
    let start = vec![1.0_f64, 0.0, 0.0];
    let end = vec![0.0_f64, 1.0, 0.0];
    let res = geometric_slerp(&start, &end, &[0.0_f64, 1.0_f64]).unwrap();
    assert_eq!(res.len(), 2, "MR42 slerp output length");
    for (k, &expected) in start.iter().enumerate() {
        assert!(
            (res[0][k] - expected).abs() < 1e-12,
            "MR42 slerp(t=0)[{k}] = {} vs {expected}",
            res[0][k]
        );
    }
    for (k, &expected) in end.iter().enumerate() {
        assert!(
            (res[1][k] - expected).abs() < 1e-12,
            "MR42 slerp(t=1)[{k}] = {} vs {expected}",
            res[1][k]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — Mahalanobis distance with the identity inverse covariance
// equals Euclidean distance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mahalanobis_identity_inv_cov_equals_euclidean() {
    let pts = sample_points();
    let d = pts[0].len();
    let mut id = vec![vec![0.0; d]; d];
    for (i, row) in id.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for a in &pts {
        for b in &pts {
            let m = mahalanobis(a, b, &id);
            let e = euclidean(a, b);
            assert!(
                (m - e).abs() < 1e-9,
                "MR43 mahalanobis(I) = {m} vs euclidean = {e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — kulsinski distance is finite (well-defined) for any
// non-degenerate pair of boolean vectors.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kulsinski_finite() {
    let pairs: &[(Vec<bool>, Vec<bool>)] = &[
        (
            vec![true, false, true, false],
            vec![true, true, false, false],
        ),
        (
            vec![true, false, false, true],
            vec![false, true, true, false],
        ),
    ];
    for (u, v) in pairs {
        let k = kulsinski(u, v);
        assert!(
            k.is_finite(),
            "MR44 kulsinski non-finite: {k} on {u:?} vs {v:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — sokalmichener(x, x) = 0 for any boolean vector.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sokalmichener_self_zero() {
    let bs: &[Vec<bool>] = &[
        vec![true, false, true, false, true],
        vec![false, true, true, true, false],
    ];
    for b in bs {
        let s = sokalmichener(b, b);
        assert!(s.abs() < 1e-12, "MR45 sokalmichener(x, x) = {s} on {b:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — sokalsneath(x, x) = 0 for any boolean vector with at least
// one true.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_sokalsneath_self_zero() {
    let bs: &[Vec<bool>] = &[
        vec![true, false, true, false, true],
        vec![false, true, true, true, false],
    ];
    for b in bs {
        let s = sokalsneath(b, b);
        assert!(s.abs() < 1e-12, "MR46 sokalsneath(x, x) = {s} on {b:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — ConvexHull area is positive on a triangle with non-collinear
// vertices.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_area_positive() {
    let points = vec![(0.0, 0.0), (4.0, 0.0), (0.0, 3.0), (1.0, 1.0)];
    let hull = ConvexHull::new(&as_rows(&points)).unwrap();
    assert!(
        hull.volume > 0.0,
        "MR47 ConvexHull area = {} ≤ 0",
        hull.volume
    );
    // Triangle area is 6 (base 4, height 3, area = 6); SciPy names it `volume` in 2-D.
    assert!(
        (hull.volume - 6.0).abs() < 1e-9,
        "MR47 ConvexHull area = {} expected 6",
        hull.volume
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — ConvexHull vertex indices are unique.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_unique_vertices() {
    let points = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5)];
    let hull = ConvexHull::new(&as_rows(&points)).unwrap();
    let mut seen = vec![false; points.len()];
    for &v in &hull.vertices {
        assert!(v < points.len(), "MR48 vertex idx {v} out of range");
        assert!(!seen[v], "MR48 duplicate vertex {v}");
        seen[v] = true;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — ConvexHull perimeter is positive for any non-degenerate input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_perimeter_positive() {
    let points = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
    let hull = ConvexHull::new(&as_rows(&points)).unwrap();
    // SciPy's `area` of a 2-D hull is its perimeter.
    assert!(hull.area > 0.0, "MR49 ConvexHull perimeter = {}", hull.area);
    // 4-3-4-3 rectangle perimeter = 14.
    assert!(
        (hull.area - 14.0).abs() < 1e-9,
        "MR49 perimeter = {} expected 14",
        hull.area
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — Delaunay triangulation simplex indices are all in [0, n).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_delaunay_simplices_in_range() {
    let points = vec![
        (0.0_f64, 0.0_f64),
        (4.0, 0.0),
        (4.0, 3.0),
        (0.0, 3.0),
        (2.0, 1.5),
    ];
    let d = Delaunay::new(&as_rows(&points)).unwrap();
    let n = points.len();
    for simplex in &d.simplices {
        assert_eq!(simplex.len(), 3, "MR50 a 2-D simplex has 3 vertices");
        for &idx in simplex {
            assert!(idx < n, "MR50 simplex idx {idx} ≥ {n}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — Voronoi regions cover all input points (each point belongs to
// some region).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_voronoi_regions_exist() {
    let points = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5)];
    let v = Voronoi::new(&as_rows(&points)).unwrap();
    assert_eq!(
        v.point_region.len(),
        points.len(),
        "MR51 voronoi point_region length"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — ConvexHull on a subset of points has area ≤ that of the full
// hull (monotonicity in inclusion).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_convex_hull_subset_area_dominated() {
    let outer = vec![
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
        (5.0, 5.0),
    ];
    let subset = vec![(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)];
    let hull_outer = ConvexHull::new(&as_rows(&outer)).unwrap();
    let hull_subset = ConvexHull::new(&as_rows(&subset)).unwrap();
    assert!(
        hull_subset.volume <= hull_outer.volume + 1e-9,
        "MR52 subset area = {} > outer area = {}",
        hull_subset.volume,
        hull_outer.volume
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — Rotation::identity().apply(v) = v.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_identity_apply_returns_input() {
    let id = Rotation::identity();
    let vs: &[[f64; 3]] = &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.5, 0.5, -2.0],
    ];
    for v in vs {
        let r = id.apply(*v);
        for k in 0..3 {
            assert!(
                (r[k] - v[k]).abs() < 1e-12,
                "MR53 identity rotation changed v[{k}]: {} vs {}",
                r[k],
                v[k]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — Rotation.inv().apply(R.apply(v)) = v for any unit quaternion.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_inverse_undoes_apply() {
    let r = Rotation::from_quat([0.5, 0.5, 0.5, 0.5]);
    let r_inv = r.inv();
    let v = [1.0_f64, 2.0, 3.0];
    let rotated = r.apply(v);
    let recovered = r_inv.apply(rotated);
    for k in 0..3 {
        assert!(
            (recovered[k] - v[k]).abs() < 1e-9,
            "MR54 R.inv()(R(v))[{k}] = {} vs {}",
            recovered[k],
            v[k]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — from_quat then as_quat round-trip preserves the (normalized)
// quaternion (modulo sign).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_quat_roundtrip() {
    let q_in = [0.7071067811865475, 0.0, 0.0, 0.7071067811865475]; // 90° about x.
    let r = Rotation::from_quat(q_in);
    let q_out = r.as_quat();
    let same = (q_in[0] - q_out[0]).abs() < 1e-9
        && (q_in[1] - q_out[1]).abs() < 1e-9
        && (q_in[2] - q_out[2]).abs() < 1e-9
        && (q_in[3] - q_out[3]).abs() < 1e-9;
    let neg = (q_in[0] + q_out[0]).abs() < 1e-9
        && (q_in[1] + q_out[1]).abs() < 1e-9
        && (q_in[2] + q_out[2]).abs() < 1e-9
        && (q_in[3] + q_out[3]).abs() < 1e-9;
    assert!(
        same || neg,
        "MR55 quat roundtrip: in = {q_in:?}, out = {q_out:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — Rotation preserves vector magnitude (isometry).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_preserves_magnitude() {
    let r = Rotation::from_quat([0.5, 0.5, 0.5, 0.5]);
    let vs: &[[f64; 3]] = &[[1.0, 2.0, 3.0], [-2.0, 0.5, 1.5], [0.0, 0.0, 1.0]];
    for v in vs {
        let nv: f64 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let rv = r.apply(*v);
        let nrv: f64 = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
        assert!(
            (nv - nrv).abs() < 1e-9,
            "MR56 rotation broke magnitude: {nv} vs {nrv}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR57 — as_rotvec then from_rotvec round-trip preserves the rotation
// (verified via apply on a probe vector).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_rotvec_roundtrip() {
    let r1 = Rotation::from_quat([0.3, 0.4, 0.5, std::f64::consts::FRAC_1_SQRT_2]);
    let rv = r1.as_rotvec();
    let r2 = Rotation::from_rotvec(rv);
    let probe = [1.0_f64, 2.0, -1.5];
    let v1 = r1.apply(probe);
    let v2 = r2.apply(probe);
    for k in 0..3 {
        assert!(
            (v1[k] - v2[k]).abs() < 1e-7,
            "MR57 rotvec roundtrip differs at {k}: {} vs {}",
            v1[k],
            v2[k]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR58 — Rotation preserves dot products: u·v = (R·u)·(R·v).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rotation_preserves_dot_product() {
    let r = Rotation::from_quat([0.2, 0.3, 0.4, 0.5]);
    let u = [1.0_f64, 2.0, 3.0];
    let v = [0.5_f64, -1.0, 2.0];
    let original = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let ru = r.apply(u);
    let rv = r.apply(v);
    let rotated = ru[0] * rv[0] + ru[1] * rv[1] + ru[2] * rv[2];
    assert!(
        (original - rotated).abs() < 1e-9,
        "MR58 rotation changed dot product: {original} vs {rotated}"
    );
}
