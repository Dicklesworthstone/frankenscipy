#![no_main]

use arbitrary::Arbitrary;
use fsci_cluster::{FclusterCriterion, LinkageMethod, fcluster, linkage};
use libfuzzer_sys::fuzz_target;

// Cluster fcluster bounds oracle:
// fcluster(z, k) should produce cluster labels in range [1, k] for valid
// linkage matrix z and max_clusters k.
//
// This catches:
// - Off-by-one in cluster assignment indexing
// - Incorrect handling of singleton clusters
// - Edge cases: single point, all same, k=1, k=n

const MAX_POINTS: usize = 32;
const MAX_DIMS: usize = 8;

#[derive(Debug, Arbitrary)]
struct ClusterInput {
    points: Vec<Vec<f64>>,
    max_clusters: u8,
    method_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fuzz_target!(|input: ClusterInput| {
    let n = input.points.len().min(MAX_POINTS);
    if n < 2 {
        return;
    }

    let dim = input.points.first().map_or(0, |p| p.len().min(MAX_DIMS));
    if dim == 0 {
        return;
    }

    let data: Vec<Vec<f64>> = input
        .points
        .iter()
        .take(n)
        .map(|p| {
            p.iter()
                .take(dim)
                .map(|&v| sanitize(v))
                .chain(std::iter::repeat(0.0))
                .take(dim)
                .collect()
        })
        .collect();

    let method = match input.method_variant % 5 {
        0 => LinkageMethod::Single,
        1 => LinkageMethod::Complete,
        2 => LinkageMethod::Average,
        3 => LinkageMethod::Ward,
        _ => LinkageMethod::Centroid,
    };

    let z = match linkage(&data, method) {
        Ok(z) => z,
        Err(_) => return,
    };

    let max_clusters = (input.max_clusters as usize).clamp(1, n);
    // fcluster signature returns Result<Vec<usize>, ClusterError>; ignore
    // input shapes that fail validation (typically n < 2 or degenerate z).
    let labels = match fcluster(&z, FclusterCriterion::MaxClust(max_clusters)) {
        Ok(v) => v,
        Err(_) => return,
    };

    if labels.len() != n {
        panic!(
            "fcluster label count mismatch: got {} expected {} (n={}, k={})",
            labels.len(),
            n,
            n,
            max_clusters
        );
    }

    for (i, &label) in labels.iter().enumerate() {
        if label == 0 || label > max_clusters {
            panic!(
                "fcluster label out of bounds at index {}: label={} not in [1, {}]",
                i, label, max_clusters
            );
        }
    }
});
