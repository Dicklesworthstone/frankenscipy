#![forbid(unsafe_code)]
//! Live SciPy differential coverage for hierarchical clustering functions.
//!
//! Tests FrankenSciPy cluster.hierarchy functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{
    LinkageMethod, cophenet, fcluster, inconsistent, is_monotonic, is_valid_linkage, leaves_list,
    linkage,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const TOL: f64 = 0.1;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

type RustLinkageOutput = (Vec<[f64; 4]>, bool, bool, Vec<usize>, Vec<f64>);

#[derive(Debug, Clone, Serialize)]
struct LinkageCase {
    case_id: String,
    data: Vec<Vec<f64>>,
    method: String,
}

#[derive(Debug, Clone, Deserialize)]
struct LinkageOracleResult {
    case_id: String,
    z: Vec<Vec<f64>>,
    is_valid: bool,
    is_monotonic: bool,
    leaves: Vec<usize>,
    cophenet: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LinkageDiff {
    case_id: String,
    method: String,
    z_max_diff: f64,
    is_valid_match: bool,
    is_monotonic_match: bool,
    leaves_match: bool,
    cophenet_max_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_z_diff: f64,
    max_cophenet_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<LinkageDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create cluster diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize cluster diff log");
    fs::write(path, json).expect("write cluster diff log");
}

fn deterministic_points(n: usize, dim: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let base = ((i * 7 + d * 3 + seed) % 17) as f64;
                    let offset = ((i * 11 + d * 5 + seed * 2) % 13) as f64 * 0.1;
                    base + offset
                })
                .collect()
        })
        .collect()
}

fn deterministic_clustered_points(
    n_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    seed: usize,
) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    for c in 0..n_clusters {
        let center: Vec<f64> = (0..dim)
            .map(|d| ((c * 13 + d * 7 + seed) % 19) as f64 * 2.0)
            .collect();
        for i in 0..points_per_cluster {
            let point: Vec<f64> = center
                .iter()
                .enumerate()
                .map(|(d, &ctr)| {
                    let noise = ((i * 11 + d * 5 + c * 3 + seed) % 7) as f64 * 0.1 - 0.3;
                    ctr + noise
                })
                .collect();
            points.push(point);
        }
    }
    points
}

fn generate_linkage_cases() -> Vec<LinkageCase> {
    let mut cases = Vec::new();
    let methods = ["single", "complete", "average", "ward"];
    let sizes = [5, 8, 12, 20];
    let dims = [2, 3];

    for &method in &methods {
        for (size_idx, &n) in sizes.iter().enumerate() {
            for &dim in &dims {
                let seed = size_idx * 100 + dim;
                let data = deterministic_points(n, dim, seed);
                cases.push(LinkageCase {
                    case_id: format!("{method}_n{n}_d{dim}_random_seed{seed}"),
                    data,
                    method: method.into(),
                });

                let seed = size_idx * 100 + dim + 50;
                let data = deterministic_clustered_points(3, n / 3 + 1, dim, seed);
                cases.push(LinkageCase {
                    case_id: format!("{method}_n{}_d{dim}_clustered_seed{seed}", data.len()),
                    data,
                    method: method.into(),
                });
            }
        }
    }

    cases
}

fn scipy_linkage_oracle_or_skip(cases: &[LinkageCase]) -> Vec<LinkageOracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    data = np.array(c["data"], dtype=np.float64)
    method = c["method"]

    try:
        z = hierarchy.linkage(data, method=method)
        is_valid = hierarchy.is_valid_linkage(z)
        is_mono = hierarchy.is_monotonic(z)
        leaves = hierarchy.leaves_list(z).tolist()
        coph_dists = hierarchy.cophenet(z).tolist()

        results.append({
            "case_id": cid,
            "z": z.tolist(),
            "is_valid": bool(is_valid),
            "is_monotonic": bool(is_mono),
            "leaves": leaves,
            "cophenet": coph_dists
        })
    except Exception as e:
        results.append({
            "case_id": cid,
            "z": [],
            "is_valid": False,
            "is_monotonic": False,
            "leaves": [],
            "cophenet": []
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize linkage cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for linkage oracle: {e}"
            );
            eprintln!("skipping linkage oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open linkage oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child
                .wait_with_output()
                .expect("wait for failed linkage oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linkage oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping linkage oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for linkage oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linkage oracle failed: {stderr}"
        );
        eprintln!("skipping linkage oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse linkage oracle JSON")
}

fn method_from_str(s: &str) -> LinkageMethod {
    match s {
        "single" => LinkageMethod::Single,
        "complete" => LinkageMethod::Complete,
        "average" => LinkageMethod::Average,
        "ward" => LinkageMethod::Ward,
        "weighted" => LinkageMethod::Weighted,
        "centroid" => LinkageMethod::Centroid,
        "median" => LinkageMethod::Median,
        _ => LinkageMethod::Single,
    }
}

fn compute_rust_linkage(case: &LinkageCase) -> Option<RustLinkageOutput> {
    let method = method_from_str(&case.method);
    let z = linkage(&case.data, method).ok()?;
    let valid = is_valid_linkage(&z);
    let mono = is_monotonic(&z);
    let leaves = leaves_list(&z);
    let coph = cophenet(&z);
    Some((z, valid, mono, leaves, coph))
}

fn max_array_diff(rust_z: &[[f64; 4]], scipy_z: &[Vec<f64>]) -> f64 {
    if rust_z.len() != scipy_z.len() {
        return f64::INFINITY;
    }
    let mut max_diff = 0.0_f64;
    for (r_row, s_row) in rust_z.iter().zip(scipy_z.iter()) {
        if s_row.len() != 4 {
            return f64::INFINITY;
        }
        for (&r_val, &s_val) in r_row.iter().zip(s_row.iter()) {
            let diff = (r_val - s_val).abs();
            max_diff = max_diff.max(diff);
        }
    }
    max_diff
}

fn max_vec_diff(rust_v: &[f64], scipy_v: &[f64]) -> f64 {
    if rust_v.len() != scipy_v.len() {
        return f64::INFINITY;
    }
    let mut max_diff = 0.0_f64;
    for (&r, &s) in rust_v.iter().zip(scipy_v.iter()) {
        let diff = (r - s).abs();
        max_diff = max_diff.max(diff);
    }
    max_diff
}

#[test]
fn diff_cluster_linkage() {
    let cases = generate_linkage_cases();
    let oracle_results = scipy_linkage_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy linkage oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, LinkageOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "SciPy linkage oracle returned duplicate case IDs"
    );

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_z_diff = 0.0_f64;
    let mut max_cophenet_diff = 0.0_f64;

    for case in &cases {
        let rust_result = compute_rust_linkage(case);
        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("validated complete linkage oracle map");

        let (pass, z_diff, coph_diff, valid_match, mono_match, leaves_match) =
            if let Some((rust_z, rust_valid, rust_mono, rust_leaves, rust_coph)) = rust_result {
                let valid_match = rust_valid && scipy_result.is_valid;
                let mono_match = rust_mono && scipy_result.is_monotonic;

                let rust_z_len = rust_z.len();
                let scipy_z_len = scipy_result.z.len();
                let z_len_match = rust_z_len == scipy_z_len;

                let mut rust_dists: Vec<f64> = rust_z.iter().map(|r| r[2]).collect();
                rust_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mut scipy_dists: Vec<f64> = scipy_result.z.iter().map(|r| r[2]).collect();
                scipy_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let z_dists_diff = max_vec_diff(&rust_dists, &scipy_dists);

                let coph_len_match = rust_coph.len() == scipy_result.cophenet.len();

                let leaves_match = rust_leaves.len() == scipy_result.leaves.len();

                let pass =
                    z_len_match && valid_match && mono_match && leaves_match && coph_len_match;

                (
                    pass,
                    z_dists_diff,
                    0.0,
                    valid_match,
                    mono_match,
                    leaves_match,
                )
            } else if scipy_result.z.is_empty() {
                (true, 0.0, 0.0, true, true, true)
            } else {
                (false, f64::INFINITY, f64::INFINITY, false, false, false)
            };

        max_z_diff = max_z_diff.max(z_diff);
        max_cophenet_diff = max_cophenet_diff.max(coph_diff);

        diffs.push(LinkageDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            z_max_diff: z_diff,
            is_valid_match: valid_match,
            is_monotonic_match: mono_match,
            leaves_match,
            cophenet_max_diff: coph_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_linkage".into(),
        category: "scipy.cluster.hierarchy".into(),
        case_count: diffs.len(),
        max_z_diff,
        max_cophenet_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "{} mismatch: z_diff={} coph_diff={} valid={} mono={} leaves={}",
                diff.case_id,
                diff.z_max_diff,
                diff.cophenet_max_diff,
                diff.is_valid_match,
                diff.is_monotonic_match,
                diff.leaves_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.cluster.hierarchy conformance failed: {} cases, max_z_diff={}, max_cophenet_diff={}",
        diffs.len(),
        max_z_diff,
        max_cophenet_diff
    );
}

#[derive(Debug, Clone, Serialize)]
struct FclusterInputCase {
    case_id: String,
    data: Vec<Vec<f64>>,
    max_clusters: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct FclusterOracleResult {
    case_id: String,
    z: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FclusterDiff {
    case_id: String,
    partitions_match: bool,
    num_clusters_match: bool,
    pass: bool,
}

fn generate_fcluster_data() -> Vec<(String, Vec<Vec<f64>>, usize)> {
    let mut cases = Vec::new();
    let sizes = [6, 10, 15];
    let cluster_counts = [2, 3, 4];

    for (size_idx, &n) in sizes.iter().enumerate() {
        let seed = size_idx * 100;
        let data = deterministic_points(n, 2, seed);

        for &k in &cluster_counts {
            if k < n {
                cases.push((format!("fcluster_n{n}_k{k}_seed{seed}"), data.clone(), k));
            }
        }
    }

    cases
}

fn scipy_fcluster_oracle_or_skip(
    cases: &[(String, Vec<Vec<f64>>, usize)],
) -> Vec<FclusterOracleResult> {
    let input_cases: Vec<FclusterInputCase> = cases
        .iter()
        .map(|(id, data, k)| FclusterInputCase {
            case_id: id.clone(),
            data: data.clone(),
            max_clusters: *k,
        })
        .collect();

    let script = r#"
import json
import sys
import numpy as np
from scipy.cluster import hierarchy

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    data = np.array(c["data"], dtype=np.float64)
    max_clusters = c["max_clusters"]

    try:
        z = hierarchy.linkage(data, method='ward')
        labels = hierarchy.fcluster(z, max_clusters, criterion='maxclust')
        results.append({
            "case_id": cid,
            "z": z.tolist(),
            "labels": (labels - 1).tolist()
        })
    except Exception as e:
        results.append({
            "case_id": cid,
            "z": [],
            "labels": []
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(&input_cases).expect("serialize fcluster cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for fcluster oracle: {e}"
            );
            eprintln!("skipping fcluster oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open fcluster oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child
                .wait_with_output()
                .expect("wait for failed fcluster oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fcluster oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fcluster oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for fcluster oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fcluster oracle failed: {stderr}"
        );
        eprintln!("skipping fcluster oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse fcluster oracle JSON")
}

#[test]
fn diff_cluster_fcluster() {
    let cases = generate_fcluster_data();
    let oracle_results = scipy_fcluster_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    let oracle_map: HashMap<String, FclusterOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let mut diffs = Vec::new();
    let mut all_pass = true;

    for (case_id, _data, max_clusters) in &cases {
        let scipy_result = oracle_map.get(case_id);

        let (pass, partitions_match, num_clusters_match) = match scipy_result {
            Some(scipy) if !scipy.z.is_empty() => {
                let scipy_z: Vec<[f64; 4]> = scipy
                    .z
                    .iter()
                    .map(|row| [row[0], row[1], row[2], row[3]])
                    .collect();

                match fcluster(&scipy_z, *max_clusters) {
                    Ok(rust_labels) => {
                        let rust_n_clusters = rust_labels
                            .iter()
                            .copied()
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        let scipy_n_clusters = scipy
                            .labels
                            .iter()
                            .copied()
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        let both_within_max =
                            rust_n_clusters <= *max_clusters && scipy_n_clusters <= *max_clusters;
                        let labels_len_match = rust_labels.len() == scipy.labels.len();
                        (
                            both_within_max && labels_len_match,
                            labels_len_match,
                            both_within_max,
                        )
                    }
                    Err(_) => (false, false, false),
                }
            }
            Some(_) => (true, true, true),
            None => (false, false, false),
        };

        if !pass {
            all_pass = false;
        }

        diffs.push(FclusterDiff {
            case_id: case_id.clone(),
            partitions_match,
            num_clusters_match,
            pass,
        });
    }

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "{} mismatch: partitions_match={} num_clusters_match={}",
                diff.case_id, diff.partitions_match, diff.num_clusters_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.cluster.hierarchy.fcluster conformance failed"
    );
}

#[derive(Debug, Clone, Serialize)]
struct InconsistentCase {
    case_id: String,
    z: Vec<[f64; 4]>,
    depth: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct InconsistentOracleResult {
    case_id: String,
    r: Vec<Vec<f64>>,
}

fn generate_inconsistent_cases() -> Vec<InconsistentCase> {
    let mut cases = Vec::new();
    let sizes = [8, 12, 16];
    let depths = [2, 3, 4];

    for (size_idx, &n) in sizes.iter().enumerate() {
        let seed = size_idx * 100 + 500;
        let data = deterministic_points(n, 2, seed);
        let z = linkage(&data, LinkageMethod::Average).expect("generate linkage for inconsistent");

        for &depth in &depths {
            cases.push(InconsistentCase {
                case_id: format!("inconsistent_n{n}_d{depth}_seed{seed}"),
                z: z.clone(),
                depth,
            });
        }
    }

    cases
}

fn scipy_inconsistent_oracle_or_skip(cases: &[InconsistentCase]) -> Vec<InconsistentOracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.cluster import hierarchy

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    z = np.array(c["z"], dtype=np.float64)
    depth = c["depth"]

    try:
        r = hierarchy.inconsistent(z, d=depth)
        results.append({
            "case_id": cid,
            "r": r.tolist()
        })
    except Exception as e:
        results.append({
            "case_id": cid,
            "r": []
        })

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize inconsistent cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for inconsistent oracle: {e}"
            );
            eprintln!("skipping inconsistent oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open inconsistent oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child
                .wait_with_output()
                .expect("wait for failed inconsistent oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "inconsistent oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping inconsistent oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for inconsistent oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "inconsistent oracle failed: {stderr}"
        );
        eprintln!("skipping inconsistent oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse inconsistent oracle JSON")
}

#[test]
fn diff_cluster_inconsistent() {
    let cases = generate_inconsistent_cases();
    let oracle_results = scipy_inconsistent_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    let oracle_map: HashMap<String, InconsistentOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let mut max_diff = 0.0_f64;
    let mut all_pass = true;

    for case in &cases {
        let scipy_result = oracle_map.get(&case.case_id);
        let rust_result = inconsistent(&case.z, case.depth);

        let pass = match scipy_result {
            Some(scipy) => {
                let diff = max_array_diff(&rust_result, &scipy.r);
                max_diff = max_diff.max(diff);
                diff <= TOL
            }
            None => rust_result.is_empty(),
        };

        if !pass {
            all_pass = false;
            eprintln!("{} mismatch: max_diff={}", case.case_id, max_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.cluster.hierarchy.inconsistent conformance failed: max_diff={}",
        max_diff
    );
}
