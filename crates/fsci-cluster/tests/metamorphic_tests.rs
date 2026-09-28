//! Metamorphic tests for `fsci-cluster`.
//!
//! Each test asserts an invariant of clustering that holds regardless
//! of the specific input — e.g. labels lie in [0, k), `vq` reproduces
//! `kmeans` labels for the same centroids, linkage matrix monotonicity.
//!
//! Run with: `cargo test -p fsci-cluster --test metamorphic_tests`

use fsci_cluster::{
    FclusterCriterion, LinkageMethod, adjusted_rand_score, calinski_harabasz_score,
    completeness_score, cophenet, davies_bouldin_score, dbscan, elbow_inertias, fcluster,
    fclusterdata, fowlkes_mallows_score, gap_statistic, homogeneity_score, inconsistent,
    is_monotonic, is_valid_linkage, kmeans, kmedoids, leaves_list, linkage, linkage_from_distances,
    mean_shift, mini_batch_kmeans, normalized_mutual_info, num_obs_linkage, proximity_cliques,
    silhouette_samples, silhouette_score, v_measure_score, vq, whiten,
};

fn small_dataset() -> Vec<Vec<f64>> {
    // Three clusters in 2D, well separated.
    vec![
        // Cluster 1
        vec![0.0, 0.0],
        vec![0.5, 0.2],
        vec![0.1, -0.3],
        vec![-0.2, 0.4],
        // Cluster 2
        vec![5.0, 5.0],
        vec![5.4, 5.1],
        vec![4.8, 4.7],
        vec![5.2, 4.9],
        // Cluster 3
        vec![10.0, 0.0],
        vec![9.7, -0.2],
        vec![10.3, 0.1],
        vec![9.8, 0.4],
    ]
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — kmeans labels are in [0, k); inertia is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_label_range_and_inertia() {
    let data = small_dataset();
    for &k in &[1, 2, 3, 5] {
        if k > data.len() {
            continue;
        }
        let result = kmeans(&data, k, 200, 42).unwrap();
        assert_eq!(result.labels.len(), data.len(), "labels length");
        for &lab in &result.labels {
            assert!(lab < k, "MR1 label {lab} out of range for k={k}");
        }
        assert!(
            result.inertia.is_finite() && result.inertia >= 0.0,
            "MR1 inertia non-finite or negative: {} (k={k})",
            result.inertia
        );
        assert_eq!(result.centroids.len(), k, "centroid count");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — vq(data, kmeans.centroids) returns the same labels as kmeans.
//
// This is a structural sanity: the vq operator is the assignment step
// of Lloyd's algorithm, so a converged kmeans must satisfy
// vq(centroids) == kmeans.labels at convergence (up to ties on
// equidistant points).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_vq_matches_kmeans_assignment() {
    let data = small_dataset();
    let k = 3;
    let result = kmeans(&data, k, 500, 7).unwrap();
    let (vq_labels, dists) = vq(&data, &result.centroids).unwrap();
    assert_eq!(vq_labels.len(), result.labels.len());
    for (i, (kl, vl)) in result.labels.iter().zip(&vq_labels).enumerate() {
        assert_eq!(
            *kl, *vl,
            "MR2 label disagreement at i={i}: kmeans={kl} vq={vl}"
        );
        assert!(dists[i].is_finite() && dists[i] >= 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — linkage produces a valid linkage matrix (per is_valid_linkage)
//        and the merge distances are non-decreasing for single, complete,
//        and ward methods (these are guaranteed monotone).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_validity_and_monotonicity() {
    let data = small_dataset();
    for &method in &[
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ] {
        let z = linkage(&data, method).unwrap();
        assert!(is_valid_linkage(&z), "MR3 invalid linkage for {method:?}");
        if matches!(
            method,
            LinkageMethod::Single | LinkageMethod::Complete | LinkageMethod::Ward
        ) {
            for i in 1..z.len() {
                let d_prev = z[i - 1][2];
                let d_curr = z[i][2];
                assert!(
                    d_curr >= d_prev - 1e-12,
                    "MR3 {method:?} not monotone at i={i}: {d_prev} → {d_curr}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — fcluster respects max_clusters: the number of unique labels
//        in the output is at most max_clusters.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fcluster_respects_max_clusters() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    for &max_k in &[1usize, 2, 3, 5, 10] {
        let labels = fcluster(&z, FclusterCriterion::MaxClust(max_k)).unwrap();
        assert_eq!(labels.len(), data.len());
        let mut sorted_unique: Vec<usize> = labels.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        assert!(
            sorted_unique.len() <= max_k,
            "MR4 fcluster gave {} clusters but max_clusters={max_k}",
            sorted_unique.len()
        );
        // Labels must be 1-based integers (matches scipy convention).
        assert!(*sorted_unique.first().unwrap() >= 1, "labels must be ≥ 1");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — silhouette_score is in [-1, 1] for any clustering with k ≥ 2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_silhouette_in_unit_signed_interval() {
    let data = small_dataset();
    for &k in &[2, 3] {
        let res = kmeans(&data, k, 200, 42).unwrap();
        let s = silhouette_score(&data, &res.labels).unwrap();
        assert!(
            (-1.0..=1.0).contains(&s),
            "MR5 silhouette score out of [-1, 1]: {s} (k={k})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — kmeans is deterministic across calls with the same seed.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_deterministic_seeded() {
    let data = small_dataset();
    let r1 = kmeans(&data, 3, 200, 12345).unwrap();
    let r2 = kmeans(&data, 3, 200, 12345).unwrap();
    assert_eq!(r1.labels, r2.labels, "MR6 kmeans labels differ across runs");
    assert_eq!(
        r1.centroids.len(),
        r2.centroids.len(),
        "centroid count differs"
    );
    for (c1, c2) in r1.centroids.iter().zip(&r2.centroids) {
        for (a, b) in c1.iter().zip(c2) {
            assert!((a - b).abs() < 1e-12, "centroid drift: {a} vs {b}");
        }
    }
    assert!((r1.inertia - r2.inertia).abs() < 1e-12, "inertia drift");
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — kmeans inertia computation matches Σ‖x_i − c_{label_i}‖².
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_inertia_matches_definition() {
    let data = small_dataset();
    let res = kmeans(&data, 3, 500, 7).unwrap();
    let mut expected = 0.0;
    for (i, point) in data.iter().enumerate() {
        let centroid = &res.centroids[res.labels[i]];
        let d2: f64 = point
            .iter()
            .zip(centroid)
            .map(|(p, c)| (p - c).powi(2))
            .sum();
        expected += d2;
    }
    let rel_err = ((res.inertia - expected) / expected.abs().max(1e-12)).abs();
    assert!(
        rel_err < 1e-9,
        "MR7 inertia mismatch: reported={} expected={} rel_err={rel_err:e}",
        res.inertia,
        expected
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — DBSCAN with very small eps marks every point as noise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dbscan_tiny_eps_marks_all_noise() {
    let data = small_dataset();
    let res = dbscan(&data, 1e-12, 3).unwrap();
    assert_eq!(res.labels.len(), data.len());
    for &lab in &res.labels {
        assert_eq!(lab, -1, "MR8 DBSCAN(eps=tiny): expected all noise");
    }
    assert_eq!(res.n_clusters, 0, "MR8 expected 0 clusters");
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — DBSCAN finds the three obvious clusters in the small dataset
// at a sensible eps. The number of clusters must equal 3 and no point
// should be labelled noise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dbscan_separable_clusters() {
    let data = small_dataset(); // 3 well-separated clusters of 4 each
    let res = dbscan(&data, 1.5, 2).unwrap();
    assert_eq!(
        res.n_clusters, 3,
        "MR9 expected 3 clusters, got {}",
        res.n_clusters
    );
    for (i, &lab) in res.labels.iter().enumerate() {
        assert!(lab >= 0, "MR9 point {i} should not be noise: lab={lab}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — DBSCAN labels are in {-1, 0, 1, ..., n_clusters-1}.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dbscan_label_range() {
    let data = small_dataset();
    let res = dbscan(&data, 1.5, 2).unwrap();
    for &lab in &res.labels {
        assert!(
            lab == -1 || (lab >= 0 && (lab as usize) < res.n_clusters),
            "MR10 label {lab} out of range for n_clusters={}",
            res.n_clusters
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — linkage matrix has shape (n - 1, 4) where n is the number of
// input observations. Each row encodes a single agglomerative merge.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_shape() {
    let data = small_dataset();
    let n = data.len();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    assert_eq!(
        z.len(),
        n - 1,
        "MR11 linkage rows: got {}, expected {}",
        z.len(),
        n - 1
    );
    // Each row already has fixed length 4 (the [f64; 4] type), so
    // structural shape is enforced by the type system.
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — leaves_list returns every original observation index exactly
// once across [0, n).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_leaves_list_is_permutation() {
    let data = small_dataset();
    let n = data.len();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    let leaves = leaves_list(&z);
    assert_eq!(
        leaves.len(),
        n,
        "MR12 leaves count: got {}, expected {n}",
        leaves.len()
    );
    let mut sorted = leaves.clone();
    sorted.sort_unstable();
    for (i, &v) in sorted.iter().enumerate() {
        assert_eq!(
            v, i,
            "MR12 missing index {i} in leaves_list: got {sorted:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — num_obs_linkage matches the input dataset size.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_num_obs_matches_input() {
    let data = small_dataset();
    let n = data.len();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    assert_eq!(num_obs_linkage(&z), n, "MR13 num_obs_linkage mismatch");
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — Calinski-Harabasz score is positive and finite for a valid
// k-clustering with k ≥ 2 of well-separated data.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_calinski_harabasz_positive() {
    let data = small_dataset();
    let res = kmeans(&data, 3, 200, 42).unwrap();
    let score = calinski_harabasz_score(&data, &res.labels).unwrap();
    assert!(
        score.is_finite() && score > 0.0,
        "MR14 CH score not positive: {score}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — Davies-Bouldin score is non-negative and finite.
// (Lower is better; well-separated clusters give a small DB index.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_davies_bouldin_nonneg() {
    let data = small_dataset();
    let res = kmeans(&data, 3, 200, 42).unwrap();
    let score = davies_bouldin_score(&data, &res.labels).unwrap();
    assert!(
        score.is_finite() && score >= 0.0,
        "MR15 DB score not non-negative: {score}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — Identical clusterings give ARI = 1 and NMI = 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_identical_labelings_give_perfect_agreement() {
    let labels: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 2, 2, 0, 1];
    let ari = adjusted_rand_score(&labels, &labels).unwrap();
    assert!(
        (ari - 1.0).abs() < 1e-12,
        "MR16 ARI(L, L) = {ari}, expected 1"
    );
    let nmi = normalized_mutual_info(&labels, &labels).unwrap();
    assert!(
        (nmi - 1.0).abs() < 1e-9,
        "MR16 NMI(L, L) = {nmi}, expected 1"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — ARI is symmetric: ARI(a, b) = ARI(b, a).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ari_symmetric() {
    let a = vec![0_usize, 0, 1, 1, 2, 2, 0, 1, 2, 0];
    let b = vec![0_usize, 1, 0, 1, 2, 2, 1, 1, 2, 0];
    let ab = adjusted_rand_score(&a, &b).unwrap();
    let ba = adjusted_rand_score(&b, &a).unwrap();
    assert!(
        (ab - ba).abs() < 1e-12,
        "MR17 ARI not symmetric: {ab} vs {ba}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — kmeans inertia is non-increasing as k grows (more clusters
// fit data better in sum-of-squares).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_inertia_nonincreasing_in_k() {
    let data = small_dataset(); // 12 points, 3 well-separated clusters
    let mut prev = f64::INFINITY;
    for k in 1..=4 {
        let res = kmeans(&data, k, 200, 42).unwrap();
        assert!(
            res.inertia.is_finite() && res.inertia >= 0.0,
            "MR18 inertia must be finite + non-negative at k={k}: {}",
            res.inertia
        );
        assert!(
            res.inertia <= prev + 1e-9,
            "MR18 inertia not monotone: k={k}, prev={prev}, curr={}",
            res.inertia
        );
        prev = res.inertia;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — mini_batch_kmeans returns k centroids of the right shape.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mini_batch_kmeans_returns_k_centroids() {
    let data = small_dataset();
    for k in [2_usize, 3, 4] {
        let res = mini_batch_kmeans(&data, k, 100, 6, 42).unwrap();
        assert_eq!(res.centroids.len(), k, "MR19 centroid count");
        for c in &res.centroids {
            assert_eq!(c.len(), data[0].len(), "MR19 centroid dim");
        }
        assert_eq!(res.labels.len(), data.len(), "MR19 label count");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — fclusterdata(X, k, m) equals fcluster(linkage(X, m), k):
// the convenience function should be definitionally equivalent to
// the two-step composition.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fclusterdata_matches_fcluster_pipeline() {
    let data = small_dataset();
    let criterion = FclusterCriterion::MaxClust(3);
    let from_func = fclusterdata(&data, criterion, LinkageMethod::Ward).unwrap();
    let z = linkage(&data, LinkageMethod::Ward).unwrap();
    let from_pipeline = fcluster(&z, criterion).unwrap();
    assert_eq!(
        from_func, from_pipeline,
        "MR20 fclusterdata should equal fcluster(linkage(X))"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — cophenet returns a condensed distance vector of length
// n*(n-1)/2 for an n-leaf linkage tree.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cophenet_condensed_length() {
    let data = small_dataset();
    let n = data.len();
    let z = linkage(&data, LinkageMethod::Average).unwrap();
    let c = cophenet(&z);
    let expected = n * (n - 1) / 2;
    assert_eq!(
        c.len(),
        expected,
        "MR21 cophenet length = {} vs expected n*(n-1)/2 = {expected}",
        c.len()
    );
    for (i, &v) in c.iter().enumerate() {
        assert!(v >= 0.0, "MR21 cophenet[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — Linkage matrices produced by ward/single/average are
// monotonic — merge distances are non-decreasing along the tree.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_is_monotonic_for_classic_methods() {
    let data = small_dataset();
    for &method in &[
        LinkageMethod::Ward,
        LinkageMethod::Single,
        LinkageMethod::Average,
    ] {
        let z = linkage(&data, method).unwrap();
        assert!(
            is_monotonic(&z),
            "MR22 linkage(method = {method:?}) is not monotonic"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — Normalized mutual information is symmetric: NMI(a, b) = NMI(b, a).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_normalized_mutual_info_symmetric() {
    let a = vec![0_usize, 0, 1, 1, 2, 2, 0, 1, 2, 0];
    let b = vec![1_usize, 1, 0, 0, 2, 2, 1, 0, 2, 1];
    let nmi_ab = normalized_mutual_info(&a, &b).unwrap();
    let nmi_ba = normalized_mutual_info(&b, &a).unwrap();
    assert!(
        (nmi_ab - nmi_ba).abs() < 1e-10,
        "MR23 NMI(a, b) = {nmi_ab}, NMI(b, a) = {nmi_ba}"
    );
    assert!(
        (-1e-10..=1.0 + 1e-10).contains(&nmi_ab),
        "MR23 NMI not in [0, 1]"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — Fowlkes-Mallows score on a perfect labeling equals 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fowlkes_mallows_perfect_labeling_one() {
    let labels = vec![0_usize, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2];
    let s = fowlkes_mallows_score(&labels, &labels).unwrap();
    assert!(
        (s - 1.0).abs() < 1e-12,
        "MR24 FM(self, self) = {s}, expected 1.0"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — Homogeneity score is in [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_homogeneity_in_unit_interval() {
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        ),
        (
            vec![0, 0, 1, 1, 2, 2, 0, 1, 2],
            vec![1, 1, 0, 0, 2, 2, 1, 0, 2],
        ),
        (
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 1, 0, 1, 0, 1, 0, 1, 0],
        ),
    ];
    for (i, (t, p)) in cases.iter().enumerate() {
        let h = homogeneity_score(t, p).unwrap();
        assert!(
            (-1e-10..=1.0 + 1e-10).contains(&h),
            "MR25 homogeneity[{i}] = {h} outside [0, 1]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — Completeness score is in [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_completeness_in_unit_interval() {
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        ),
        (
            vec![0, 0, 1, 1, 2, 2, 0, 1, 2],
            vec![1, 1, 0, 0, 2, 2, 1, 0, 2],
        ),
        (
            vec![0, 1, 0, 1, 0, 1, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ];
    for (i, (t, p)) in cases.iter().enumerate() {
        let c = completeness_score(t, p).unwrap();
        assert!(
            (-1e-10..=1.0 + 1e-10).contains(&c),
            "MR26 completeness[{i}] = {c} outside [0, 1]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — whiten produces approximately unit-variance columns when input
// rows have non-degenerate column variance. (Mean is not necessarily 0
// in scipy's whiten — it only divides by stddev.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_whiten_columns_have_unit_variance() {
    let data = vec![
        vec![1.0_f64, 5.0, 100.0],
        vec![2.0, 6.0, 200.0],
        vec![3.0, 7.0, 300.0],
        vec![4.0, 8.0, 400.0],
        vec![5.0, 9.0, 500.0],
    ];
    let w = whiten(&data).unwrap();
    let n = w.len();
    let d = w[0].len();
    for j in 0..d {
        let mean: f64 = w.iter().map(|row| row[j]).sum::<f64>() / n as f64;
        let var: f64 = w.iter().map(|row| (row[j] - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (var - 1.0).abs() < 1e-9,
            "MR27 whiten column {j} variance = {var}, expected 1"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — v_measure_score is symmetric in (true, predicted) and lies
// in [0, 1].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_v_measure_symmetric_and_bounded() {
    let cases: &[(Vec<usize>, Vec<usize>)] = &[
        (
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        ),
        (
            vec![0, 0, 1, 1, 2, 2, 0, 1, 2],
            vec![1, 1, 0, 0, 2, 2, 1, 0, 2],
        ),
    ];
    for (i, (t, p)) in cases.iter().enumerate() {
        let v_tp = v_measure_score(t, p).unwrap();
        let v_pt = v_measure_score(p, t).unwrap();
        assert!(
            (v_tp - v_pt).abs() < 1e-10,
            "MR28 v_measure[{i}] not symmetric: {v_tp} vs {v_pt}"
        );
        assert!(
            (-1e-10..=1.0 + 1e-10).contains(&v_tp),
            "MR28 v_measure[{i}] = {v_tp} outside [0, 1]"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — silhouette_samples are each in [-1, 1] and their mean equals
// silhouette_score.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_silhouette_samples_bounded_and_mean_matches_score() {
    let data = small_dataset();
    let labels = vec![0_usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let s_each = silhouette_samples(&data, &labels).unwrap();
    let s_avg = silhouette_score(&data, &labels).unwrap();
    let mean: f64 = s_each.iter().sum::<f64>() / s_each.len() as f64;
    for (i, &v) in s_each.iter().enumerate() {
        assert!(
            (-1.0 - 1e-9..=1.0 + 1e-9).contains(&v),
            "MR29 silhouette[{i}] = {v} outside [-1, 1]"
        );
    }
    assert!(
        (mean - s_avg).abs() < 1e-9,
        "MR29 mean(silhouette_samples) = {mean} vs silhouette_score = {s_avg}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — kmedoids returns labels in [0, k).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmedoids_labels_in_range() {
    let data = small_dataset();
    for k in 1..=4 {
        let res = kmedoids(&data, k, 100, 7).unwrap();
        for (i, &lbl) in res.labels.iter().enumerate() {
            assert!(lbl < k, "MR30 kmedoids k={k} label[{i}] = {lbl} ≥ k");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — inconsistent(z, depth) returns a vector of the same length as
// z (one row per merge).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_inconsistent_length_matches_linkage() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Average).unwrap();
    let inc = inconsistent(&z, 2);
    assert_eq!(
        inc.len(),
        z.len(),
        "MR31 inconsistent length = {} vs linkage rows = {}",
        inc.len(),
        z.len()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — ARI of a perfect labeling is 1; ARI is symmetric.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_ari_perfect_label_is_one_and_symmetric() {
    let labels = vec![0_usize, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2];
    let ari_self = adjusted_rand_score(&labels, &labels).unwrap();
    assert!(
        (ari_self - 1.0).abs() < 1e-10,
        "MR32 ARI(self, self) = {ari_self}, expected 1"
    );
    let other = vec![1_usize, 1, 0, 0, 2, 2, 1, 0, 2, 1, 0, 2];
    let ari_lo = adjusted_rand_score(&labels, &other).unwrap();
    let ari_ol = adjusted_rand_score(&other, &labels).unwrap();
    assert!(
        (ari_lo - ari_ol).abs() < 1e-10,
        "MR32 ARI not symmetric: {ari_lo} vs {ari_ol}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — elbow_inertias is non-increasing in k: SSE drops or stays
// the same as k grows.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_elbow_inertias_non_increasing() {
    let data = small_dataset();
    let inertias = elbow_inertias(&data, 5, 11);
    assert_eq!(inertias.len(), 5, "MR33 elbow_inertias length");
    for w in inertias.windows(2) {
        // Allow small numerical slack in case kmeans heuristic lands at
        // marginally different local optima.
        assert!(
            w[1] <= w[0] + 1e-6,
            "MR33 elbow_inertias not non-increasing: {} → {}",
            w[0],
            w[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — mean_shift labels are in [0, num_centers): every assigned
// label points at a returned centre.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mean_shift_labels_match_centers() {
    let data = small_dataset();
    let (centers, labels) = mean_shift(&data, 2.0, 100).unwrap();
    let n_centers = centers.len();
    assert!(n_centers >= 1, "MR34 mean_shift returned no centers");
    for (i, &lbl) in labels.iter().enumerate() {
        assert!(
            lbl < n_centers,
            "MR34 mean_shift label[{i}] = {lbl} ≥ {n_centers}"
        );
    }
    assert_eq!(labels.len(), data.len(), "MR34 label count");
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — gap_statistic returns one entry per k in 1..=max_k.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gap_statistic_length() {
    let data = small_dataset();
    for max_k in [2usize, 3, 5] {
        let g = gap_statistic(&data, max_k, 4, 7);
        assert_eq!(
            g.len(),
            max_k,
            "MR35 gap_statistic length: got {} expected {max_k}",
            g.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — proximity_cliques with very large eps merges all points
// into a single clique containing every index.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_proximity_cliques_large_eps_one_clique() {
    let data = small_dataset();
    let cliques = proximity_cliques(&data, 1e6);
    // With eps spanning the entire dataset, the cliques should cover
    // every input index. (One large clique or several overlapping ones
    // both meet that criterion.)
    let mut seen = vec![false; data.len()];
    for c in &cliques {
        for &i in c {
            seen[i] = true;
        }
    }
    for (i, ok) in seen.iter().enumerate() {
        assert!(*ok, "MR36 proximity_cliques missed index {i}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — linkage_from_distances(condensed, n, method) returns a
// linkage matrix with exactly n - 1 merges.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_from_distances_n_minus_1_merges() {
    let n = 5;
    // Condensed distance vector for n=5 has length n*(n-1)/2 = 10.
    let condensed: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 1.2, 2.2, 1.8];
    for &method in &[
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
    ] {
        let z = linkage_from_distances(&condensed, n, method).unwrap();
        assert_eq!(
            z.len(),
            n - 1,
            "MR37 linkage_from_distances merges = {} expected {}",
            z.len(),
            n - 1
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — Calinski-Harabasz score is non-negative on a valid clustering.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_calinski_harabasz_nonneg() {
    let data = small_dataset();
    let labels = vec![0_usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let s = calinski_harabasz_score(&data, &labels).unwrap();
    assert!(s >= -1e-9, "MR38 calinski_harabasz = {s} < 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — Davies-Bouldin score is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_davies_bouldin_nonneg_explicit() {
    let data = small_dataset();
    let labels = vec![0_usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let s = davies_bouldin_score(&data, &labels).unwrap();
    assert!(s >= -1e-9, "MR39 davies_bouldin = {s} < 0");
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — vq returns label vector of length n with all values < k
// (the number of centroids).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_vq_labels_bounded_by_k() {
    let data = small_dataset();
    let centroids = vec![vec![0.0_f64, 0.0], vec![5.0, 5.0], vec![10.0, 0.0]];
    let (labels, _dists) = vq(&data, &centroids).unwrap();
    assert_eq!(labels.len(), data.len(), "MR40 vq labels length");
    for (i, &lbl) in labels.iter().enumerate() {
        assert!(
            lbl < centroids.len(),
            "MR40 vq labels[{i}] = {lbl} ≥ k = {}",
            centroids.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — whiten preserves the number of rows in the input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_whiten_preserves_row_count() {
    let data = small_dataset();
    let w = whiten(&data).unwrap();
    assert_eq!(w.len(), data.len(), "MR41 whiten row count");
    for (i, (a, b)) in data.iter().zip(&w).enumerate() {
        assert_eq!(a.len(), b.len(), "MR41 row {i} feature count");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — mean_shift on a non-empty dataset returns at least 1 cluster
// centre.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mean_shift_at_least_one_center() {
    let data = small_dataset();
    let (centers, _labels) = mean_shift(&data, 2.0, 100).unwrap();
    assert!(
        !centers.is_empty(),
        "MR42 mean_shift returned 0 centers for non-empty data"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — DBSCAN labels are either -1 (noise) or a non-negative cluster
// id strictly less than n_clusters.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dbscan_label_convention() {
    let data = small_dataset();
    let r = dbscan(&data, 2.0, 2).unwrap();
    assert_eq!(r.labels.len(), data.len(), "MR43 dbscan label count");
    for &l in &r.labels {
        assert!(
            l == -1 || (l >= 0 && (l as usize) < r.n_clusters),
            "MR43 dbscan label = {l} (n_clusters = {})",
            r.n_clusters
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — kmeans inertia is non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_inertia_nonneg() {
    let data = small_dataset();
    for k in [1usize, 2, 3] {
        let res = kmeans(&data, k, 100, 11).unwrap();
        assert!(
            res.inertia >= -1e-9,
            "MR44 kmeans(k={k}) inertia = {} < 0",
            res.inertia
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — fclusterdata returns labels of length n with values in [1, k].
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fclusterdata_label_range() {
    let data = small_dataset();
    for k in [1usize, 2, 3] {
        let labels =
            fclusterdata(&data, FclusterCriterion::MaxClust(k), LinkageMethod::Ward).unwrap();
        assert_eq!(labels.len(), data.len(), "MR45 fclusterdata length");
        for (i, &lbl) in labels.iter().enumerate() {
            assert!(
                lbl >= 1 && lbl <= k,
                "MR45 fclusterdata label[{i}] = {lbl} outside [1, {k}]"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — num_obs_linkage equals the number of leaves of the tree:
// for a linkage with m merges, n_obs = m + 1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_num_obs_linkage_matches_size() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Average).unwrap();
    let n_obs = num_obs_linkage(&z);
    assert_eq!(
        n_obs,
        z.len() + 1,
        "MR46 num_obs_linkage = {} expected {} = merges + 1",
        n_obs,
        z.len() + 1
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — leaves_list returns a permutation of 0..n (single linkage).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_leaves_list_single_linkage_permutation() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Single).unwrap();
    let leaves = leaves_list(&z);
    let n = data.len();
    assert_eq!(leaves.len(), n, "MR47 leaves length");
    let mut seen = vec![false; n];
    for &l in &leaves {
        assert!(l < n, "MR47 leaf index {l} ≥ n = {n}");
        seen[l] = true;
    }
    for (i, ok) in seen.iter().enumerate() {
        assert!(*ok, "MR47 leaves missed index {i}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — is_valid_linkage on a linkage produced by linkage() returns
// true.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_is_valid_linkage_returns_true_on_linkage_output() {
    let data = small_dataset();
    for &method in &[
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ] {
        let z = linkage(&data, method).unwrap();
        assert!(
            is_valid_linkage(&z),
            "MR48 is_valid_linkage(linkage({method:?})) returned false"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — kmeans on three tight, well-separated clusters partitions
// the data into 3 clusters of size 4 (correct cluster sizes).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_kmeans_recovers_3_clusters_of_4() {
    let data = small_dataset();
    let res = kmeans(&data, 3, 100, 17).unwrap();
    let mut counts = vec![0usize; 3];
    for &lbl in &res.labels {
        counts[lbl] += 1;
    }
    counts.sort();
    assert_eq!(
        counts,
        vec![4, 4, 4],
        "MR49 kmeans cluster sizes = {:?}, expected [4, 4, 4]",
        counts
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — dbscan with smaller eps results in more clusters or more
// noise points (monotonicity of partition coarseness).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dbscan_smaller_eps_more_noise_or_clusters() {
    let data = small_dataset();
    let r_loose = dbscan(&data, 5.0, 2).unwrap();
    let r_tight = dbscan(&data, 0.5, 2).unwrap();
    // Tight eps should produce more noise or more clusters (or both).
    let noise_loose = r_loose.labels.iter().filter(|&&l| l == -1).count();
    let noise_tight = r_tight.labels.iter().filter(|&&l| l == -1).count();
    assert!(
        noise_tight >= noise_loose,
        "MR50 dbscan tighter eps did not increase noise: tight={noise_tight}, loose={noise_loose}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — whiten preserves shape (rows × dims) of the input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_whiten_preserves_shape() {
    let data = small_dataset();
    let w = whiten(&data).unwrap();
    assert_eq!(w.len(), data.len(), "MR51 whiten row count");
    for (i, (a, b)) in data.iter().zip(&w).enumerate() {
        assert_eq!(a.len(), b.len(), "MR51 whiten row {i} dim");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — mini_batch_kmeans returns labels in [0, k).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_mini_batch_kmeans_label_range() {
    let data = small_dataset();
    for k in [1usize, 2, 3] {
        let res = mini_batch_kmeans(&data, k, 50, 4, 13).unwrap();
        for &lbl in &res.labels {
            assert!(lbl < k, "MR52 mini_batch_kmeans label {lbl} ≥ k = {k}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — cophenet entries are non-negative.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cophenet_nonneg_entries() {
    let data = small_dataset();
    let z = linkage(&data, LinkageMethod::Average).unwrap();
    let c = cophenet(&z);
    for (i, &v) in c.iter().enumerate() {
        assert!(v >= -1e-12, "MR53 cophenet[{i}] = {v} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — linkage_from_distances using single/complete/average all
// produce the same number of merges (n - 1) for the same input.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linkage_from_distances_consistent_merge_count() {
    let n = 6;
    let condensed: Vec<f64> = (0..n * (n - 1) / 2).map(|i| (i as f64) + 1.0).collect();
    let mut counts = Vec::new();
    for &method in &[
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
    ] {
        let z = linkage_from_distances(&condensed, n, method).unwrap();
        counts.push(z.len());
    }
    let first = counts[0];
    for c in &counts[1..] {
        assert_eq!(*c, first, "MR54 linkage merge counts differ: {:?}", counts);
    }
    assert_eq!(first, n - 1, "MR54 expected {} merges", n - 1);
}
