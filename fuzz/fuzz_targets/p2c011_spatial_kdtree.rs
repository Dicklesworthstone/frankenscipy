#![no_main]

use arbitrary::Arbitrary;
use fsci_spatial::KDTree;
use libfuzzer_sys::fuzz_target;

const MAX_POINTS: usize = 64;
const MAX_DIM: usize = 8;
const DIST_TOL: f64 = 1e-12;

#[derive(Debug, Arbitrary)]
struct KDTreeInput {
    raw_points: Vec<Vec<f64>>,
    raw_query: Vec<f64>,
    k: u8,
    radius: f64,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fuzz_target!(|input: KDTreeInput| {
    if input.raw_points.is_empty() {
        return;
    }

    let first_dim = input.raw_points[0].len();
    if first_dim == 0 || first_dim > MAX_DIM {
        return;
    }

    let points: Vec<Vec<f64>> = input
        .raw_points
        .iter()
        .take(MAX_POINTS)
        .filter(|p| p.len() == first_dim)
        .map(|p| p.iter().map(|&v| sanitize(v)).collect())
        .collect();

    if points.len() < 2 {
        return;
    }

    let tree = match KDTree::new(&points) {
        Ok(t) => t,
        Err(_) => return,
    };

    if input.raw_query.len() != first_dim {
        return;
    }
    let query: Vec<f64> = input.raw_query.iter().map(|&v| sanitize(v)).collect();

    // Oracle 1: query() returns the actual nearest neighbor
    if let Ok((idx, dist)) = tree.query(&query) {
        assert!(idx < points.len(), "query returned invalid index");

        let actual_dist = euclidean_dist(&query, &points[idx]);
        let dist_diff = (dist - actual_dist).abs();
        assert!(
            dist_diff < DIST_TOL,
            "query distance mismatch: returned {} but computed {}",
            dist,
            actual_dist
        );

        // Verify it's actually the nearest
        for (i, pt) in points.iter().enumerate() {
            let d = euclidean_dist(&query, pt);
            assert!(
                d >= dist - DIST_TOL,
                "point {} at distance {} is closer than returned nearest {} at distance {}",
                i,
                d,
                idx,
                dist
            );
        }
    }

    // Oracle 2: query_k() returns sorted k nearest neighbors
    let k = (input.k as usize).min(points.len()).max(1);
    if let Ok(results) = tree.query_k(&query, k) {
        assert_eq!(results.len(), k, "query_k returned wrong number of results");

        let mut prev_dist = 0.0;
        for (idx, dist) in &results {
            assert!(*idx < points.len(), "query_k returned invalid index");

            let actual_dist = euclidean_dist(&query, &points[*idx]);
            let dist_diff = (dist - actual_dist).abs();
            assert!(
                dist_diff < DIST_TOL,
                "query_k distance mismatch for index {}: returned {} but computed {}",
                idx,
                dist,
                actual_dist
            );

            assert!(
                *dist >= prev_dist - DIST_TOL,
                "query_k results not sorted: {} >= {}",
                prev_dist,
                dist
            );
            prev_dist = *dist;
        }

        // The k-th result should have all other points at >= distance
        if let Some(&(_, k_dist)) = results.last() {
            let mut distances: Vec<f64> = points
                .iter()
                .map(|pt| euclidean_dist(&query, pt))
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if k <= distances.len() {
                let expected_k_dist = distances[k - 1];
                assert!(
                    (k_dist - expected_k_dist).abs() < DIST_TOL,
                    "query_k k-th distance {} != expected {}",
                    k_dist,
                    expected_k_dist
                );
            }
        }
    }

    // Oracle 3: query_ball_point() returns all points within radius
    let radius = sanitize(input.radius).abs().min(1e4);
    if radius > 0.0
        && let Ok(indices) = tree.query_ball_point(&query, radius)
    {
            // All returned points must be within radius
            for &idx in &indices {
                assert!(idx < points.len(), "query_ball_point returned invalid index");
                let d = euclidean_dist(&query, &points[idx]);
                assert!(
                    d <= radius + DIST_TOL,
                    "query_ball_point returned point {} at distance {} > radius {}",
                    idx,
                    d,
                    radius
                );
            }

        // No point outside the result should be within radius
        for (i, pt) in points.iter().enumerate() {
            let d = euclidean_dist(&query, pt);
            if d <= radius - DIST_TOL && !indices.contains(&i) {
                panic!(
                    "query_ball_point missed point {} at distance {} <= radius {}",
                    i, d, radius
                );
            }
        }
    }

    // Oracle 4: query_pairs() consistency
    if let Ok(pairs) = tree.query_pairs(radius) {
        for &(i, j) in &pairs {
            assert!(i < points.len() && j < points.len(), "query_pairs invalid indices");
            assert!(i < j, "query_pairs should return i < j");
            let d = euclidean_dist(&points[i], &points[j]);
            assert!(
                d <= radius + DIST_TOL,
                "query_pairs returned pair ({}, {}) at distance {} > radius {}",
                i,
                j,
                d,
                radius
            );
        }
    }
});
