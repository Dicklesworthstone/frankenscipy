#![forbid(unsafe_code)]
//! Live SciPy differential coverage for hierarchical clustering functions.
//!
//! Tests FrankenSciPy cluster.hierarchy functions against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{
    FclusterCriterion, LinkageMethod, cophenet, fcluster, inconsistent, is_monotonic,
    is_valid_linkage, leaves_list, linkage,
};
use fsci_conformance::{ArmCounts, CompareLedger};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
/// Merge heights, cophenetic distances and inconsistency statistics are computed from the
/// same condensed distances SciPy builds, so they agree to rounding: relative to max(|s|, 1).
const REL_TOL: f64 = 1e-12;
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
    compared: BTreeMap<String, ArmCounts>,
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

const LINKAGE_METHODS: [&str; 4] = ["single", "complete", "average", "ward"];

fn generate_linkage_cases() -> Vec<LinkageCase> {
    let mut cases = Vec::new();
    let methods = LINKAGE_METHODS;
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

/// `|r - s| / max(|s|, 1)`, and infinite when exactly one side is NaN, so a max fold cannot
/// swallow it.
fn rel_diff(r: f64, s: f64) -> f64 {
    match (r.is_nan(), s.is_nan()) {
        (true, true) => 0.0,
        (false, false) => (r - s).abs() / s.abs().max(1.0),
        _ => f64::INFINITY,
    }
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
            max_diff = max_diff.max(rel_diff(r_val, s_val));
        }
    }
    max_diff
}

fn max_vec_diff(rust_v: &[f64], scipy_v: &[f64]) -> f64 {
    if rust_v.len() != scipy_v.len() {
        return f64::INFINITY;
    }
    rust_v
        .iter()
        .zip(scipy_v)
        .fold(0.0_f64, |acc, (&r, &s)| acc.max(rel_diff(r, s)))
}

/// Row-for-row linkage comparison: the merged pair of cluster ids and the member count must
/// be equal and the height within the returned relative difference. Infinite on any id or
/// count mismatch.
fn linkage_rows_diff(rust_z: &[[f64; 4]], scipy_z: &[Vec<f64>]) -> f64 {
    let ids_match = rust_z.len() == scipy_z.len()
        && rust_z.iter().zip(scipy_z).all(|(r, s)| {
            s.len() == 4
                && r[0].to_bits() == s[0].to_bits()
                && r[1].to_bits() == s[1].to_bits()
                && r[3].to_bits() == s[3].to_bits()
        });
    if !ids_match {
        return f64::INFINITY;
    }
    rust_z
        .iter()
        .zip(scipy_z)
        .fold(0.0_f64, |acc, (r, s)| acc.max(rel_diff(r[2], s[2])))
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
    let mut ledger = CompareLedger::new("diff_cluster_linkage", &LINKAGE_METHODS);

    for case in &cases {
        let rust_result = compute_rust_linkage(case);
        let scipy_result = oracle_map
            .get(&case.case_id)
            .expect("validated complete linkage oracle map");
        // The oracle sends an empty `z` when SciPy raised.
        let scipy_value = (!scipy_result.z.is_empty()).then_some(scipy_result);
        let Some((scipy_result, (rust_z, rust_valid, rust_mono, rust_leaves, rust_coph))) =
            ledger.both(&case.method, &case.case_id, scipy_value, rust_result)
        else {
            continue;
        };

        let (pass, z_diff, coph_diff, valid_match, mono_match, leaves_match) = {
            let valid_match = rust_valid && scipy_result.is_valid;
            let mono_match = rust_mono && scipy_result.is_monotonic;
            // Whole rows: SciPy's merge order, its (smaller id, larger id) columns and counts.
            let z_diff = linkage_rows_diff(&rust_z, &scipy_result.z);
            let coph_diff = max_vec_diff(&rust_coph, &scipy_result.cophenet);
            let leaves_match = rust_leaves == scipy_result.leaves;
            let pass = valid_match
                && mono_match
                && z_diff <= REL_TOL
                && coph_diff <= REL_TOL
                && leaves_match;
            (
                pass,
                z_diff,
                coph_diff,
                valid_match,
                mono_match,
                leaves_match,
            )
        };
        ledger.compared(&case.method, &case.case_id, pass);

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
        compared: ledger.counts().clone(),
        max_z_diff,
        max_cophenet_diff,
        tolerance: REL_TOL,
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
    let min_per_arm = LINKAGE_METHODS
        .iter()
        .map(|method| cases.iter().filter(|c| c.method == *method).count())
        .min()
        .expect("LINKAGE_METHODS is non-empty");
    ledger.finish(min_per_arm);
}

/// One `fcluster(ward_linkage(data), t, criterion, depth)` call. `t` is the cluster count for
/// `maxclust` and the threshold otherwise; `depth` is read by `inconsistent` only.
#[derive(Debug, Clone, Serialize)]
struct FclusterInputCase {
    case_id: String,
    data: Vec<Vec<f64>>,
    criterion: &'static str,
    t: f64,
    depth: usize,
}

const FCLUSTER_CRITERIA: [&str; 3] = ["maxclust", "distance", "inconsistent"];

#[derive(Debug, Clone, Deserialize)]
struct FclusterOracleResult {
    case_id: String,
    z: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FclusterDiff {
    case_id: String,
    labels_match: bool,
    partitions_match: bool,
    pass: bool,
}

/// True when both labelings group the observations identically, whatever the label values.
fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let mut a_to_b = HashMap::new();
    let mut b_to_a = HashMap::new();
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            *a_to_b.entry(x).or_insert(y) == y && *b_to_a.entry(y).or_insert(x) == x
        })
}

/// The lattice-like points tie many ward merges (n = 10, seed 100 ties rows 5 and 6), which
/// is where maxclust's whole-tie cut and SciPy's label numbering show.
fn generate_fcluster_cases() -> Vec<FclusterInputCase> {
    let mut cases = Vec::new();
    let sizes = [6, 10, 15, 24];

    for (size_idx, &n) in sizes.iter().enumerate() {
        let seed = size_idx * 100;
        let data = deterministic_points(n, 2, seed);
        let mut push = |criterion: &'static str, t: f64, depth: usize| {
            cases.push(FclusterInputCase {
                case_id: format!("fcluster_{criterion}_t{t}_d{depth}_n{n}_seed{seed}"),
                data: data.clone(),
                criterion,
                t,
                depth,
            });
        };
        for k in [1.0, 2.0, 3.0, 4.0, 7.0] {
            push("maxclust", k, 2);
        }
        for t in [1.5, 4.0, 8.0, 15.0] {
            push("distance", t, 2);
        }
        for (t, depth) in [(0.7, 2), (1.0, 2), (1.15, 2), (0.9, 3), (1.15, 3)] {
            push("inconsistent", t, depth);
        }
    }

    cases
}

fn scipy_fcluster_oracle_or_skip(input_cases: &[FclusterInputCase]) -> Vec<FclusterOracleResult> {
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
    criterion = c["criterion"]
    t = int(c["t"]) if criterion == "maxclust" else c["t"]

    try:
        z = hierarchy.linkage(data, method='ward')
        labels = hierarchy.fcluster(z, t, criterion=criterion, depth=c["depth"])
        results.append({
            "case_id": cid,
            "z": z.tolist(),
            "labels": labels.tolist()
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
    let cases = generate_fcluster_cases();
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
    let mut ledger = CompareLedger::new("diff_cluster_fcluster", &FCLUSTER_CRITERIA);

    for case in &cases {
        let case_id = &case.case_id;
        // The oracle sends an empty `z` when SciPy raised; fsci clusters SciPy's own `z`.
        let scipy_result = oracle_map.get(case_id).filter(|scipy| !scipy.z.is_empty());
        let rust_labels = scipy_result.and_then(|scipy| {
            let scipy_z: Vec<[f64; 4]> = scipy
                .z
                .iter()
                .map(|row| [row[0], row[1], row[2], row[3]])
                .collect();
            let criterion = match case.criterion {
                "maxclust" => FclusterCriterion::MaxClust(case.t as usize),
                "distance" => FclusterCriterion::Distance(case.t),
                _ => FclusterCriterion::Inconsistent {
                    t: case.t,
                    depth: case.depth,
                    r: None,
                },
            };
            fcluster(&scipy_z, criterion).ok()
        });
        let Some((scipy, rust_labels)) =
            ledger.both(case.criterion, case_id, scipy_result, rust_labels)
        else {
            continue;
        };

        // Both sides are 1-based. The labels themselves are compared, not only the
        // partition: SciPy numbers clusters in its own depth-first tree order.
        let labels_match = rust_labels == scipy.labels;
        let partitions_match = same_partition(&rust_labels, &scipy.labels);
        let pass = labels_match;
        ledger.compared(case.criterion, case_id, pass);

        if !pass {
            all_pass = false;
        }

        diffs.push(FclusterDiff {
            case_id: case_id.clone(),
            labels_match,
            partitions_match,
            pass,
        });
    }

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "{} mismatch: labels_match={} partitions_match={}",
                diff.case_id, diff.labels_match, diff.partitions_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.cluster.hierarchy.fcluster conformance failed"
    );
    let min_per_arm = FCLUSTER_CRITERIA
        .iter()
        .map(|criterion| cases.iter().filter(|c| c.criterion == *criterion).count())
        .min()
        .expect("FCLUSTER_CRITERIA is non-empty");
    ledger.finish(min_per_arm);
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
    let mut ledger = CompareLedger::new("diff_cluster_inconsistent", &["inconsistent"]);

    for case in &cases {
        // The oracle sends an empty `r` when SciPy raised.
        let scipy_result = oracle_map
            .get(&case.case_id)
            .filter(|scipy| !scipy.r.is_empty());
        let rust_result = inconsistent(&case.z, case.depth);
        let Some((scipy, rust_result)) = ledger.both(
            "inconsistent",
            &case.case_id,
            scipy_result,
            Some(rust_result),
        ) else {
            continue;
        };

        let pass = {
            let diff = max_array_diff(&rust_result, &scipy.r);
            max_diff = max_diff.max(diff);
            // The max fold in `max_array_diff` swallows a NaN in fsci's matrix.
            let no_nan = rust_result.iter().flatten().all(|v| !v.is_nan());
            diff <= REL_TOL && no_nan
        };
        ledger.compared("inconsistent", &case.case_id, pass);

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
    ledger.finish(cases.len());
}
