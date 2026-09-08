#![no_main]

use arbitrary::Arbitrary;
use fsci_cluster::{kmedoids, vq};
use libfuzzer_sys::fuzz_target;

// k-medoids robustness oracle.
//
// Resolves frankenscipy-mpf66.
//
// Verifies five invariants of the (post-frankenscipy-ebx8l)
// kmedoids implementation:
//
//   1. labels.len() == data.len()
//   2. each label is in 0..k
//   3. centroids.len() == k and each is dim-conforming
//   4. each centroid is a member of data (medoid invariant —
//      kmedoids selects actual data points, not means)
//   5. n_iter <= max_iter
//
// Also pins a metamorphic property: vq(data, centroids) must
// reproduce the labels kmedoids assigned, since the medoid for
// each cluster minimizes the L1 sum-of-distances and vq finds
// the nearest centroid.

const MAX_POINTS: usize = 32;
const MAX_DIM: usize = 6;
const MAX_K: usize = 4;
const MAX_ITER: usize = 50;

#[derive(Debug, Arbitrary)]
struct KmedoidsInput {
    raw_points: Vec<Vec<f64>>,
    k_choice: u8,
    seed: u64,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9 + 1e-9 * a.abs().max(b.abs())
}

fn vec_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
}

fuzz_target!(|input: KmedoidsInput| {
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
    let n = points.len();
    if n < 2 {
        return;
    }

    let k = 1 + (input.k_choice as usize % MAX_K.min(n));
    let Ok(result) = kmedoids(&points, k, MAX_ITER, input.seed) else {
        return;
    };

    // Property 1: labels length.
    if result.labels.len() != n {
        panic!(
            "kmedoids labels length {} != n={n}",
            result.labels.len()
        );
    }

    // Property 2: every label ∈ [0, k).
    for (i, &lbl) in result.labels.iter().enumerate() {
        if lbl >= k {
            panic!("kmedoids label[{i}] = {lbl} >= k={k}");
        }
    }

    // Property 3: centroids count and shape.
    if result.centroids.len() != k {
        panic!(
            "kmedoids centroids.len() = {} != k={k}",
            result.centroids.len()
        );
    }
    for (c, centroid) in result.centroids.iter().enumerate() {
        if centroid.len() != first_dim {
            panic!(
                "centroid[{c}] dim {} != input dim {first_dim}",
                centroid.len()
            );
        }
    }

    // Property 4: each centroid is a member of data (medoid).
    for (c, centroid) in result.centroids.iter().enumerate() {
        if !points.iter().any(|p| vec_eq(p, centroid)) {
            panic!(
                "kmedoids centroid[{c}] = {centroid:?} is not a data member"
            );
        }
    }

    // Property 5: iteration count bounded.
    if result.n_iter > MAX_ITER {
        panic!("kmedoids n_iter {} > max_iter {MAX_ITER}", result.n_iter);
    }

    // Property 6 (metamorphic): vq(points, centroids) reproduces
    // the labels kmedoids assigned. Both choose the nearest
    // centroid by Euclidean distance.
    if let Ok((vq_labels, _vq_dists)) = vq(&points, &result.centroids)
        && vq_labels != result.labels
    {
        // It's possible kmedoids ties broke differently than vq's
        // first-min selection; only fail if any point's vq label
        // gives a strictly larger distance than its kmedoids label.
        for (i, (&km_lbl, &vq_lbl)) in
            result.labels.iter().zip(vq_labels.iter()).enumerate()
        {
            if km_lbl != vq_lbl {
                let d_km = result.centroids[km_lbl]
                    .iter()
                    .zip(points[i].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>();
                let d_vq = result.centroids[vq_lbl]
                    .iter()
                    .zip(points[i].iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>();
                if d_km > d_vq + 1e-9 {
                    panic!(
                        "kmedoids label[{i}] = {km_lbl} sq_dist {d_km} > \
                         vq label {vq_lbl} sq_dist {d_vq}"
                    );
                }
            }
        }
    }
});
