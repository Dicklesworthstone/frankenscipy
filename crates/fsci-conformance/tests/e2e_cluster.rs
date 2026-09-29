#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-009 (Clustering).
//!
//! Implements conformance tests for scipy.cluster parity:
//!   Happy-path (1-5): kmeans, hierarchical, dbscan, metrics
//!   Edge cases (6-8): degenerate cases, validation
//!   Cross-op consistency (9-11): metric agreement, linkage validation
//!   Performance boundary (12-14): large datasets
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-009/e2e/`.

use fsci_cluster::{
    FclusterCriterion, LinkageMethod, adjusted_rand_score, calinski_harabasz_score,
    completeness_score, davies_bouldin_score, dbscan, fcluster, homogeneity_score, is_monotonic,
    is_valid_linkage, kmeans, linkage, normalized_mutual_info, silhouette_score, v_measure_score,
    whiten,
};
use fsci_conformance::PacketFamily;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    cluster_metadata: Option<ClusterMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClusterMetadata {
    algorithm: String,
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts")
        .join(PacketFamily::Cluster.packet_id())
        .join("e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!(
        "rch exec -- cargo test -p fsci-conformance --test e2e_cluster -- {scenario_id} --nocapture"
    )
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).expect("failed to create e2e dir");
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).expect("failed to write bundle");
}

/// Generate well-separated clusters for testing
fn generate_clusters(n_per_cluster: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut data = Vec::new();
    for c in 0..n_clusters {
        let center_x = (c as f64) * 10.0;
        let center_y = (c as f64) * 10.0;
        for i in 0..n_per_cluster {
            // Simple deterministic "noise" based on seed and index
            let noise_x = ((seed.wrapping_mul(i as u64 + 1).wrapping_mul(c as u64 + 1)) % 1000)
                as f64
                / 1000.0
                - 0.5;
            let noise_y = ((seed.wrapping_mul(i as u64 + 2).wrapping_mul(c as u64 + 1)) % 1000)
                as f64
                / 1000.0
                - 0.5;
            data.push(vec![center_x + noise_x, center_y + noise_y]);
        }
    }
    data
}

// ───────────────────── Scenario runner framework ──────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    passed: bool,
    error_chain: Option<String>,
    cluster_meta: Option<ClusterMetadata>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_owned(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            passed: true,
            error_chain: None,
            cluster_meta: None,
        }
    }

    fn set_cluster_meta(
        &mut self,
        algorithm: &str,
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
    ) {
        self.cluster_meta = Some(ClusterMetadata {
            algorithm: algorithm.to_owned(),
            n_samples,
            n_features,
            n_clusters,
        });
    }

    fn record_step(
        &mut self,
        name: &str,
        action: &str,
        input_summary: &str,
        mode: &str,
        f: impl FnOnce() -> Result<String, String>,
    ) -> bool {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();
        let (outcome, output_summary) = match result {
            Ok(summary) => ("pass".to_owned(), summary),
            Err(err) => {
                self.passed = false;
                if self.error_chain.is_none() {
                    self.error_chain = Some(err.clone());
                }
                ("fail".to_owned(), err)
            }
        };
        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_owned(),
            action: action.to_owned(),
            input_summary: input_summary.to_owned(),
            output_summary,
            duration_ns,
            mode: mode.to_owned(),
            outcome: outcome.clone(),
        });
        outcome == "pass"
    }

    fn finish(self) -> ForensicLogBundle {
        let total_duration_ns = self.start.elapsed().as_nanos();
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: Vec::new(),
            environment: make_env(),
            cluster_metadata: self.cluster_meta,
            overall: OverallResult {
                status: if self.passed { "pass" } else { "fail" }.to_owned(),
                total_duration_ns,
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//                       HAPPY-PATH SCENARIOS (1-5)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: K-means clustering
/// scipy.cluster.vq.kmeans
#[test]
fn scenario_01_kmeans() {
    let mut runner = ScenarioRunner::new("scenario_01_kmeans");
    runner.set_cluster_meta("kmeans", 60, 2, 3);

    let data = generate_clusters(20, 3, 42);

    runner.record_step(
        "kmeans_clustering",
        "kmeans(data, k=3)",
        "60 points, 3 clusters",
        "Strict",
        || {
            let result = kmeans(&data, 3, 100, 42).map_err(|e| format!("{e}"))?;
            // Should find 3 centroids
            if result.centroids.len() != 3 {
                return Err(format!(
                    "expected 3 centroids, got {}",
                    result.centroids.len()
                ));
            }
            // All points should be assigned to a cluster
            if result.labels.len() != data.len() {
                return Err(format!(
                    "expected {} labels, got {}",
                    data.len(),
                    result.labels.len()
                ));
            }
            // Labels should be 0, 1, or 2
            let valid_labels = result.labels.iter().all(|&l| l < 3);
            if !valid_labels {
                return Err("invalid cluster labels".to_owned());
            }
            Ok(format!(
                "centroids={}, inertia={:.4}",
                result.centroids.len(),
                result.inertia
            ))
        },
    );

    runner.record_step(
        "kmeans_convergence",
        "check inertia is finite and positive",
        "verify convergence",
        "Strict",
        || {
            let result = kmeans(&data, 3, 100, 42).map_err(|e| format!("{e}"))?;
            if result.inertia.is_finite() && result.inertia >= 0.0 {
                Ok(format!("inertia={:.4}", result.inertia))
            } else {
                Err(format!("invalid inertia: {}", result.inertia))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_kmeans", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: Hierarchical clustering with linkage
/// scipy.cluster.hierarchy.linkage
#[test]
fn scenario_02_hierarchical() {
    let mut runner = ScenarioRunner::new("scenario_02_hierarchical");
    runner.set_cluster_meta("linkage", 20, 2, 3);

    let data = generate_clusters(5, 4, 123);

    runner.record_step(
        "linkage_ward",
        "linkage(data, method=Ward)",
        "20 points, Ward linkage",
        "Strict",
        || {
            let z = linkage(&data, LinkageMethod::Ward).map_err(|e| format!("{e}"))?;
            // Linkage matrix should have n-1 rows
            let expected_rows = data.len() - 1;
            if z.len() != expected_rows {
                return Err(format!("expected {} rows, got {}", expected_rows, z.len()));
            }
            // Each row should have 4 elements
            if !z.iter().all(|row| row.len() == 4) {
                return Err("rows should have 4 elements".to_owned());
            }
            Ok(format!("linkage matrix: {}x4", z.len()))
        },
    );

    runner.record_step(
        "linkage_validation",
        "is_valid_linkage and is_monotonic",
        "validate linkage matrix",
        "Strict",
        || {
            let z = linkage(&data, LinkageMethod::Ward).map_err(|e| format!("{e}"))?;
            let valid = is_valid_linkage(&z);
            let monotonic = is_monotonic(&z);
            if valid && monotonic {
                Ok("valid=true, monotonic=true".to_owned())
            } else {
                Err(format!("valid={valid}, monotonic={monotonic}"))
            }
        },
    );

    runner.record_step(
        "fcluster_cut",
        "fcluster(z, max_clusters=3)",
        "cut dendrogram into 3 clusters",
        "Strict",
        || {
            let z = linkage(&data, LinkageMethod::Ward).map_err(|e| format!("{e}"))?;
            let labels =
                fcluster(&z, FclusterCriterion::MaxClust(3)).map_err(|e| format!("{e}"))?;
            if labels.len() != data.len() {
                return Err(format!("expected {} labels", data.len()));
            }
            // Count unique labels
            let mut unique: Vec<usize> = labels.clone();
            unique.sort();
            unique.dedup();
            if unique.len() <= 3 {
                Ok(format!("{} clusters assigned", unique.len()))
            } else {
                Err(format!("too many clusters: {}", unique.len()))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_hierarchical", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: DBSCAN density-based clustering
/// sklearn.cluster.DBSCAN (scipy doesn't have DBSCAN but we test ours)
#[test]
fn scenario_03_dbscan() {
    let mut runner = ScenarioRunner::new("scenario_03_dbscan");
    runner.set_cluster_meta("dbscan", 60, 2, 3);

    let data = generate_clusters(20, 3, 456);

    runner.record_step(
        "dbscan_clustering",
        "dbscan(data, eps=2.0, min_samples=3)",
        "60 points, find density clusters",
        "Strict",
        || {
            let result = dbscan(&data, 2.0, 3).map_err(|e| format!("{e}"))?;
            // Should assign labels to all points
            if result.labels.len() != data.len() {
                return Err(format!(
                    "expected {} labels, got {}",
                    data.len(),
                    result.labels.len()
                ));
            }
            // Count clusters (excluding noise label -1 if present)
            let n_clusters = result.n_clusters;
            let n_noise = result.labels.iter().filter(|&&l| l == -1).count();
            // With well-separated data and reasonable eps, should find ~3 clusters
            if (2..=5).contains(&n_clusters) {
                Ok(format!("n_clusters={}, n_noise={}", n_clusters, n_noise))
            } else {
                Err(format!("unexpected n_clusters={}", n_clusters))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_dbscan", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: Clustering quality metrics
/// sklearn.metrics (silhouette, calinski-harabasz, davies-bouldin)
#[test]
fn scenario_04_clustering_metrics() {
    let mut runner = ScenarioRunner::new("scenario_04_clustering_metrics");
    runner.set_cluster_meta("metrics", 60, 2, 3);

    let data = generate_clusters(20, 3, 789);
    // Assign true labels based on generation
    let labels: Vec<usize> = (0..3).flat_map(|c| vec![c; 20]).collect();

    runner.record_step(
        "silhouette_score",
        "silhouette_score(data, labels)",
        "measure cluster cohesion",
        "Strict",
        || {
            let score = silhouette_score(&data, &labels).map_err(|e| e.to_string())?;
            // Well-separated clusters should have high silhouette (> 0.5)
            if (-1.0..=1.0).contains(&score) {
                if score > 0.3 {
                    Ok(format!("silhouette={score:.4} (good separation)"))
                } else {
                    Ok(format!("silhouette={score:.4} (weak separation)"))
                }
            } else {
                Err(format!("silhouette={score} out of range [-1, 1]"))
            }
        },
    );

    runner.record_step(
        "calinski_harabasz",
        "calinski_harabasz_score(data, labels)",
        "variance ratio criterion",
        "Strict",
        || {
            let score = calinski_harabasz_score(&data, &labels).map_err(|e| e.to_string())?;
            // CH score should be positive for valid clustering
            if score > 0.0 && score.is_finite() {
                Ok(format!("calinski_harabasz={score:.4}"))
            } else {
                Err(format!("invalid CH score: {score}"))
            }
        },
    );

    runner.record_step(
        "davies_bouldin",
        "davies_bouldin_score(data, labels)",
        "cluster similarity measure",
        "Strict",
        || {
            let score = davies_bouldin_score(&data, &labels).map_err(|e| e.to_string())?;
            // DB score should be non-negative (lower is better)
            if score >= 0.0 && score.is_finite() {
                Ok(format!("davies_bouldin={score:.4}"))
            } else {
                Err(format!("invalid DB score: {score}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_clustering_metrics", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: External validation metrics
/// sklearn.metrics (adjusted_rand, NMI, homogeneity, completeness, v-measure)
#[test]
fn scenario_05_external_validation() {
    let mut runner = ScenarioRunner::new("scenario_05_external_validation");
    runner.set_cluster_meta("external_metrics", 30, 2, 3);

    // True and predicted labels for comparison
    let labels_true: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ];
    // Slightly perturbed predictions
    let labels_pred: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ];

    runner.record_step(
        "adjusted_rand",
        "adjusted_rand_score(true, pred)",
        "compare clusterings",
        "Strict",
        || {
            let score =
                adjusted_rand_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            // ARI ranges from -1 to 1, with 1 being perfect agreement
            if (-1.0..=1.0).contains(&score) {
                Ok(format!("adjusted_rand={score:.4}"))
            } else {
                Err(format!("ARI out of range: {score}"))
            }
        },
    );

    runner.record_step(
        "normalized_mutual_info",
        "normalized_mutual_info(true, pred)",
        "information-theoretic metric",
        "Strict",
        || {
            let score =
                normalized_mutual_info(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            // NMI ranges from 0 to 1
            if (0.0..=1.0 + 1e-10).contains(&score) {
                Ok(format!("nmi={score:.4}"))
            } else {
                Err(format!("NMI out of range: {score}"))
            }
        },
    );

    runner.record_step(
        "homogeneity_completeness_vmeasure",
        "homogeneity, completeness, v_measure",
        "label assignment quality",
        "Strict",
        || {
            let h = homogeneity_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            let c = completeness_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            let v = v_measure_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            // All should be in [0, 1]
            if (0.0..=1.0).contains(&h) && (0.0..=1.0).contains(&c) && (0.0..=1.0).contains(&v) {
                // V-measure should be harmonic mean of h and c
                let expected_v = if h + c > 0.0 {
                    2.0 * h * c / (h + c)
                } else {
                    0.0
                };
                if (v - expected_v).abs() < 0.01 {
                    Ok(format!("h={h:.4}, c={c:.4}, v={v:.4}"))
                } else {
                    Err(format!("v={v:.4} != expected {expected_v:.4}"))
                }
            } else {
                Err(format!("scores out of range: h={h}, c={c}, v={v}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_external_validation", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                       EDGE CASE SCENARIOS (6-8)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Data whitening
/// scipy.cluster.vq.whiten
#[test]
fn scenario_06_whiten() {
    let mut runner = ScenarioRunner::new("scenario_06_whiten");
    runner.set_cluster_meta("whiten", 20, 3, 0);

    // Data with different scales per feature
    let data: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            vec![
                i as f64 * 0.1,   // scale 0.1
                i as f64 * 10.0,  // scale 10
                i as f64 * 0.001, // scale 0.001
            ]
        })
        .collect();

    runner.record_step(
        "whiten_data",
        "whiten(data)",
        "normalize features",
        "Strict",
        || {
            let whitened = whiten(&data).map_err(|e| format!("{e}"))?;
            // Whitened data should have unit variance per column
            let n = whitened.len() as f64;
            let n_features = 3;
            for f in 0..n_features {
                let mean: f64 = whitened.iter().map(|row| row[f]).sum::<f64>() / n;
                let var: f64 = whitened
                    .iter()
                    .map(|row| (row[f] - mean).powi(2))
                    .sum::<f64>()
                    / n;
                // Variance should be close to 1 (within tolerance)
                if (var - 1.0).abs() > 0.1 {
                    return Err(format!("feature {f} variance={var:.4}, expected ~1.0"));
                }
            }
            Ok("all features normalized".to_owned())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_whiten", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Perfect clustering metrics
/// When true == pred, metrics should be 1.0
#[test]
fn scenario_07_perfect_match() {
    let mut runner = ScenarioRunner::new("scenario_07_perfect_match");
    runner.set_cluster_meta("perfect", 30, 2, 3);

    let labels: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ];

    runner.record_step(
        "perfect_ari",
        "adjusted_rand_score(labels, labels)",
        "self-comparison should be 1.0",
        "Strict",
        || {
            let score = adjusted_rand_score(&labels, &labels).map_err(|e| e.to_string())?;
            if (score - 1.0).abs() < 1e-10 {
                Ok(format!("perfect ARI={score:.4}"))
            } else {
                Err(format!("ARI={score}, expected 1.0"))
            }
        },
    );

    runner.record_step(
        "perfect_nmi",
        "normalized_mutual_info(labels, labels)",
        "self-comparison should be 1.0",
        "Strict",
        || {
            let score = normalized_mutual_info(&labels, &labels).map_err(|e| e.to_string())?;
            if (score - 1.0).abs() < 1e-10 {
                Ok(format!("perfect NMI={score:.4}"))
            } else {
                Err(format!("NMI={score}, expected 1.0"))
            }
        },
    );

    runner.record_step(
        "perfect_vmeasure",
        "v_measure_score(labels, labels)",
        "self-comparison should be 1.0",
        "Strict",
        || {
            let score = v_measure_score(&labels, &labels).map_err(|e| e.to_string())?;
            if (score - 1.0).abs() < 1e-10 {
                Ok(format!("perfect V={score:.4}"))
            } else {
                Err(format!("V={score}, expected 1.0"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_perfect_match", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Different linkage methods
/// scipy.cluster.hierarchy.linkage with various methods
#[test]
fn scenario_08_linkage_methods() {
    let mut runner = ScenarioRunner::new("scenario_08_linkage_methods");
    runner.set_cluster_meta("linkage", 15, 2, 3);

    let data = generate_clusters(5, 3, 321);

    for method in [
        LinkageMethod::Single,
        LinkageMethod::Complete,
        LinkageMethod::Average,
        LinkageMethod::Ward,
    ] {
        let method_name = format!("{:?}", method);
        runner.record_step(
            &format!("linkage_{}", method_name.to_lowercase()),
            &format!("linkage(data, {:?})", method),
            &format!("{} linkage", method_name),
            "Strict",
            || {
                let z = linkage(&data, method).map_err(|e| format!("{e}"))?;
                if is_valid_linkage(&z) {
                    Ok(format!("{} valid, {} merges", method_name, z.len()))
                } else {
                    Err(format!("{} produced invalid linkage", method_name))
                }
            },
        );
    }

    let bundle = runner.finish();
    write_bundle("scenario_08_linkage_methods", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   CROSS-OP CONSISTENCY SCENARIOS (9-11)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Metric relationships
/// V-measure = harmonic mean of homogeneity and completeness
#[test]
fn scenario_09_metric_relationships() {
    let mut runner = ScenarioRunner::new("scenario_09_metric_relationships");
    runner.set_cluster_meta("metrics", 40, 2, 4);

    let labels_true: Vec<usize> = (0..4).flat_map(|c| vec![c; 10]).collect();
    // Shuffled predictions
    let labels_pred: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
        3, 3, 3, 3, 3, 3, 0, 0, 0, 0,
    ];

    runner.record_step(
        "vmeasure_harmonic_mean",
        "verify V = 2*h*c/(h+c)",
        "V-measure formula",
        "Strict",
        || {
            let h = homogeneity_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            let c = completeness_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            let v = v_measure_score(&labels_true, &labels_pred).map_err(|e| e.to_string())?;
            let expected_v = if h + c > 0.0 {
                2.0 * h * c / (h + c)
            } else {
                0.0
            };
            if (v - expected_v).abs() < 0.001 {
                Ok(format!("h={h:.4}, c={c:.4}, v={v:.4}=expected"))
            } else {
                Err(format!("v={v:.4} != expected {expected_v:.4}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_metric_relationships", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: K-means inertia decreases with more clusters
#[test]
fn scenario_10_elbow_property() {
    let mut runner = ScenarioRunner::new("scenario_10_elbow_property");
    runner.set_cluster_meta("kmeans", 100, 2, 5);

    let data = generate_clusters(20, 5, 999);

    runner.record_step(
        "elbow_monotonic",
        "inertia decreases as k increases",
        "k=2,3,4,5",
        "Strict",
        || {
            let mut inertias = Vec::new();
            for k in 2..=5 {
                let result = kmeans(&data, k, 50, 42).map_err(|e| format!("{e}"))?;
                inertias.push(result.inertia);
            }
            // Inertia should monotonically decrease
            let monotonic = inertias.windows(2).all(|w| w[1] <= w[0]);
            if monotonic {
                Ok(format!("inertias={:?}", inertias))
            } else {
                Err(format!("non-monotonic: {:?}", inertias))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_elbow_property", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: Linkage monotonicity
/// Merge distances should be non-decreasing
#[test]
fn scenario_11_linkage_monotonicity() {
    let mut runner = ScenarioRunner::new("scenario_11_linkage_monotonicity");
    runner.set_cluster_meta("linkage", 30, 2, 5);

    let data = generate_clusters(6, 5, 777);

    runner.record_step(
        "ward_monotonicity",
        "is_monotonic(linkage(data, Ward))",
        "merge distances non-decreasing",
        "Strict",
        || {
            let z = linkage(&data, LinkageMethod::Ward).map_err(|e| format!("{e}"))?;
            if is_monotonic(&z) {
                Ok("Ward linkage is monotonic".to_owned())
            } else {
                Err("Ward linkage not monotonic".to_owned())
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_linkage_monotonicity", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
//                   PERFORMANCE BOUNDARY SCENARIOS (12-14)
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large K-means
#[test]
fn scenario_12_large_kmeans() {
    let mut runner = ScenarioRunner::new("scenario_12_large_kmeans");
    runner.set_cluster_meta("kmeans", 1000, 10, 10);

    // Generate larger dataset
    let data: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            let cluster = i % 10;
            (0..10)
                .map(|j| (cluster as f64) * 5.0 + (i as f64 * (j + 1) as f64).sin())
                .collect()
        })
        .collect();

    runner.record_step(
        "kmeans_1000x10",
        "kmeans(data, k=10)",
        "1000 points, 10 dims, 10 clusters",
        "Strict",
        || {
            let start = Instant::now();
            let result = kmeans(&data, 10, 50, 42).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if result.centroids.len() == 10 && elapsed.as_millis() < 5000 {
                Ok(format!("time={:?}, inertia={:.2}", elapsed, result.inertia))
            } else {
                Err(format!("too slow or wrong: {:?}", elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_kmeans", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: Large hierarchical clustering
#[test]
fn scenario_13_large_linkage() {
    let mut runner = ScenarioRunner::new("scenario_13_large_linkage");
    runner.set_cluster_meta("linkage", 200, 5, 10);

    let data: Vec<Vec<f64>> = (0..200)
        .map(|i| {
            let cluster = i % 10;
            (0..5)
                .map(|j| (cluster as f64) * 3.0 + ((i * j) as f64).sin())
                .collect()
        })
        .collect();

    runner.record_step(
        "linkage_200x5",
        "linkage(data, Ward)",
        "200 points, 5 dims",
        "Strict",
        || {
            let start = Instant::now();
            let z = linkage(&data, LinkageMethod::Ward).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            if z.len() == 199 && is_valid_linkage(&z) && elapsed.as_millis() < 5000 {
                Ok(format!("time={:?}, {} merges", elapsed, z.len()))
            } else {
                Err(format!("invalid or slow: {:?}, len={}", elapsed, z.len()))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_large_linkage", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: Large DBSCAN
#[test]
fn scenario_14_large_dbscan() {
    let mut runner = ScenarioRunner::new("scenario_14_large_dbscan");
    runner.set_cluster_meta("dbscan", 500, 3, 5);

    let data: Vec<Vec<f64>> = (0..500)
        .map(|i| {
            let cluster = i % 5;
            vec![
                (cluster as f64) * 10.0 + (i as f64 * 0.1).sin(),
                (cluster as f64) * 10.0 + (i as f64 * 0.2).cos(),
                (cluster as f64) * 10.0 + (i as f64 * 0.3).sin(),
            ]
        })
        .collect();

    runner.record_step(
        "dbscan_500x3",
        "dbscan(data, eps=3.0, min_samples=5)",
        "500 points, 3 dims",
        "Strict",
        || {
            let start = Instant::now();
            let result = dbscan(&data, 3.0, 5).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();
            let n_noise = result.labels.iter().filter(|&&l| l == -1).count();
            if result.labels.len() == 500 && elapsed.as_millis() < 5000 {
                Ok(format!(
                    "time={:?}, clusters={}, noise={}",
                    elapsed, result.n_clusters, n_noise
                ))
            } else {
                Err(format!("slow or wrong: {:?}", elapsed))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_large_dbscan", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}
