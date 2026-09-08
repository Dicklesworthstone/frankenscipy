#![no_main]

use arbitrary::Arbitrary;
use fsci_cluster::{
    fcluster, is_monotonic, is_valid_linkage, leaves_list, linkage, num_obs_linkage, LinkageMethod,
};
use libfuzzer_sys::fuzz_target;

const MAX_POINTS: usize = 32;
const MAX_DIM: usize = 8;

#[derive(Debug, Arbitrary)]
struct LinkageInput {
    data: Vec<Vec<f64>>,
    method_idx: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e4, 1e4)
    } else {
        0.0
    }
}

fn get_method(idx: u8) -> LinkageMethod {
    match idx % 7 {
        0 => LinkageMethod::Single,
        1 => LinkageMethod::Complete,
        2 => LinkageMethod::Average,
        3 => LinkageMethod::Weighted,
        4 => LinkageMethod::Centroid,
        5 => LinkageMethod::Median,
        _ => LinkageMethod::Ward,
    }
}

fuzz_target!(|input: LinkageInput| {
    let n = input.data.len().min(MAX_POINTS);
    if n < 2 {
        return;
    }

    let dim = input.data.first().map_or(0, |r| r.len().min(MAX_DIM));
    if dim == 0 {
        return;
    }

    let data: Vec<Vec<f64>> = input
        .data
        .iter()
        .take(n)
        .map(|row| {
            row.iter()
                .take(dim)
                .map(|&v| sanitize(v))
                .chain(std::iter::repeat(0.0))
                .take(dim)
                .collect()
        })
        .collect();

    let method = get_method(input.method_idx);

    let z = match linkage(&data, method) {
        Ok(z) => z,
        Err(_) => return,
    };

    if !is_valid_linkage(&z) {
        panic!(
            "linkage produced invalid linkage matrix: n={}, method={:?}",
            n, method
        );
    }

    if !is_monotonic(&z) {
        panic!(
            "linkage produced non-monotonic linkage matrix: n={}, method={:?}",
            n, method
        );
    }

    let n_obs = num_obs_linkage(&z);
    if n_obs != n {
        panic!(
            "num_obs_linkage mismatch: expected {}, got {} (method={:?})",
            n, n_obs, method
        );
    }

    if z.len() != n - 1 {
        panic!(
            "linkage matrix rows: expected {}, got {} (n={}, method={:?})",
            n - 1,
            z.len(),
            n,
            method
        );
    }

    let leaves = leaves_list(&z);
    if leaves.len() != n {
        panic!(
            "leaves_list length: expected {}, got {} (method={:?})",
            n,
            leaves.len(),
            method
        );
    }

    let mut sorted_leaves = leaves.clone();
    sorted_leaves.sort();
    let expected: Vec<usize> = (0..n).collect();
    if sorted_leaves != expected {
        panic!(
            "leaves_list not a permutation of 0..{}: got {:?} (method={:?})",
            n, sorted_leaves, method
        );
    }

    for row in &z {
        if !row[0].is_finite() || !row[1].is_finite() || !row[2].is_finite() || !row[3].is_finite()
        {
            panic!(
                "linkage matrix contains non-finite values: {:?} (method={:?})",
                row, method
            );
        }
        if row[2] < 0.0 {
            panic!(
                "linkage distance is negative: {} (method={:?})",
                row[2], method
            );
        }
        if (row[3] as usize) < 2 {
            panic!(
                "linkage cluster size < 2: {} (method={:?})",
                row[3], method
            );
        }
    }

    if n <= 16 {
        for max_clusters in [2, 3, n / 2 + 1, n] {
            if (1..=n).contains(&max_clusters)
                && let Ok(labels) = fcluster(&z, max_clusters)
            {
                if labels.len() != n {
                    panic!(
                        "fcluster labels length: expected {}, got {} (max_clusters={})",
                        n,
                        labels.len(),
                        max_clusters
                    );
                }
                for &label in &labels {
                    if label == 0 || label > max_clusters {
                        panic!(
                            "fcluster label out of range: {} not in 1..={} (max_clusters={})",
                            label, max_clusters, max_clusters
                        );
                    }
                }
            }
        }
    }
});
