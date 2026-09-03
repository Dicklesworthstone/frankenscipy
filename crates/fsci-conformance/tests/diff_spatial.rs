#![forbid(unsafe_code)]
//! Live SciPy differential coverage for scipy.spatial.distance functions.
//!
//! Tests FrankenSciPy distance metrics against SciPy subprocess oracle
//! across deterministic input families.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{
    braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, hamming, jaccard,
    jensenshannon, mahalanobis, minkowski, seuclidean, sqeuclidean, wminkowski,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DistanceCase {
    case_id: String,
    metric: String,
    u: Vec<f64>,
    v: Vec<f64>,
    p: Option<f64>,
    w: Option<Vec<f64>>,
    vi: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    metric: String,
    dim: usize,
    rust_value: f64,
    scipy_value: f64,
    abs_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create spatial diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize spatial diff log");
    fs::write(path, json).expect("write spatial diff log");
}

fn deterministic_vector(dim: usize, seed: usize) -> Vec<f64> {
    (0..dim)
        .map(|idx| {
            let base = ((idx + seed) % 7) as f64 * 0.3 - 0.9;
            let wave = (((idx * 3 + seed) % 11) as f64 * 0.15) - 0.7;
            base + wave + (seed % 5) as f64 * 0.1
        })
        .collect()
}

fn deterministic_positive_vector(dim: usize, seed: usize) -> Vec<f64> {
    deterministic_vector(dim, seed)
        .iter()
        .map(|&x| x.abs() + 0.01)
        .collect()
}

fn deterministic_binary_vector(dim: usize, seed: usize) -> Vec<f64> {
    (0..dim)
        .map(|idx| {
            if (idx + seed).is_multiple_of(3) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn deterministic_identity_vi(dim: usize) -> Vec<Vec<f64>> {
    (0..dim)
        .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn distance_cases() -> Vec<DistanceCase> {
    let dims = [2, 3, 5, 8, 10];
    let mut cases = Vec::new();

    for (dim_idx, &dim) in dims.iter().enumerate() {
        for seed_offset in 0..5 {
            let seed = dim_idx * 10 + seed_offset;
            let u = deterministic_vector(dim, seed);
            let v = deterministic_vector(dim, seed + 100);

            cases.push(DistanceCase {
                case_id: format!("euclidean_dim{dim}_seed{seed}"),
                metric: "euclidean".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("sqeuclidean_dim{dim}_seed{seed}"),
                metric: "sqeuclidean".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("cityblock_dim{dim}_seed{seed}"),
                metric: "cityblock".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("chebyshev_dim{dim}_seed{seed}"),
                metric: "chebyshev".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("cosine_dim{dim}_seed{seed}"),
                metric: "cosine".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("correlation_dim{dim}_seed{seed}"),
                metric: "correlation".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("canberra_dim{dim}_seed{seed}"),
                metric: "canberra".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("braycurtis_dim{dim}_seed{seed}"),
                metric: "braycurtis".into(),
                u: deterministic_positive_vector(dim, seed),
                v: deterministic_positive_vector(dim, seed + 100),
                p: None,
                w: None,
                vi: None,
            });

            for p in [1.0, 2.0, 3.0, 0.5] {
                cases.push(DistanceCase {
                    case_id: format!("minkowski_dim{dim}_seed{seed}_p{p:.1}"),
                    metric: "minkowski".into(),
                    u: u.clone(),
                    v: v.clone(),
                    p: Some(p),
                    w: None,
                    vi: None,
                });
            }
        }
    }

    for &dim in &[3, 5, 8] {
        for seed_offset in 0..3 {
            let seed = 200 + dim * 10 + seed_offset;
            let u = deterministic_binary_vector(dim, seed);
            let v = deterministic_binary_vector(dim, seed + 50);

            cases.push(DistanceCase {
                case_id: format!("hamming_dim{dim}_seed{seed}"),
                metric: "hamming".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });

            cases.push(DistanceCase {
                case_id: format!("jaccard_dim{dim}_seed{seed}"),
                metric: "jaccard".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: None,
            });
        }
    }

    for &dim in &[3, 5] {
        for seed_offset in 0..3 {
            let seed = 300 + dim * 10 + seed_offset;
            let u = deterministic_positive_vector(dim, seed);
            let v = deterministic_positive_vector(dim, seed + 50);

            cases.push(DistanceCase {
                case_id: format!("jensenshannon_dim{dim}_seed{seed}"),
                metric: "jensenshannon".into(),
                u,
                v,
                p: None,
                w: None,
                vi: None,
            });
        }
    }

    for &dim in &[3, 5] {
        for seed_offset in 0..3 {
            let seed = 400 + dim * 10 + seed_offset;
            let u = deterministic_vector(dim, seed);
            let v = deterministic_vector(dim, seed + 50);
            let variance = deterministic_positive_vector(dim, seed + 200);

            cases.push(DistanceCase {
                case_id: format!("seuclidean_dim{dim}_seed{seed}"),
                metric: "seuclidean".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: Some(variance),
                vi: None,
            });

            let vi = deterministic_identity_vi(dim);
            cases.push(DistanceCase {
                case_id: format!("mahalanobis_dim{dim}_seed{seed}"),
                metric: "mahalanobis".into(),
                u: u.clone(),
                v: v.clone(),
                p: None,
                w: None,
                vi: Some(vi),
            });
        }
    }

    for &dim in &[3, 5] {
        for seed_offset in 0..2 {
            let seed = 500 + dim * 10 + seed_offset;
            let u = deterministic_vector(dim, seed);
            let v = deterministic_vector(dim, seed + 50);
            let w = deterministic_positive_vector(dim, seed + 100);

            for p in [1.5, 2.0, 3.0] {
                cases.push(DistanceCase {
                    case_id: format!("wminkowski_dim{dim}_seed{seed}_p{p:.1}"),
                    metric: "wminkowski".into(),
                    u: u.clone(),
                    v: v.clone(),
                    p: Some(p),
                    w: Some(w.clone()),
                    vi: None,
                });
            }
        }
    }

    cases
}

fn rust_output(case: &DistanceCase) -> Option<f64> {
    let result = match case.metric.as_str() {
        "euclidean" => euclidean(&case.u, &case.v),
        "sqeuclidean" => sqeuclidean(&case.u, &case.v),
        "cityblock" => cityblock(&case.u, &case.v),
        "chebyshev" => chebyshev(&case.u, &case.v),
        "cosine" => cosine(&case.u, &case.v),
        "correlation" => correlation(&case.u, &case.v),
        "canberra" => canberra(&case.u, &case.v),
        "braycurtis" => braycurtis(&case.u, &case.v),
        "minkowski" => minkowski(&case.u, &case.v, case.p?),
        "hamming" => hamming(&case.u, &case.v),
        "jaccard" => jaccard(&case.u, &case.v),
        "jensenshannon" => jensenshannon(&case.u, &case.v, None),
        "seuclidean" => seuclidean(&case.u, &case.v, case.w.as_ref()?),
        "mahalanobis" => mahalanobis(&case.u, &case.v, case.vi.as_ref()?),
        "wminkowski" => wminkowski(&case.u, &case.v, case.p?, case.w.as_ref()?),
        _ => return None,
    };
    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

fn run_scipy_oracle(cases: &[DistanceCase]) -> Option<Vec<OracleCase>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy.spatial import distance

cases = json.load(sys.stdin)
results = []

for c in cases:
    cid = c["case_id"]
    metric = c["metric"]
    u = np.array(c["u"], dtype=np.float64)
    v = np.array(c["v"], dtype=np.float64)
    p = c.get("p")
    w = c.get("w")
    vi = c.get("vi")

    try:
        if metric == "euclidean":
            val = distance.euclidean(u, v)
        elif metric == "sqeuclidean":
            val = distance.sqeuclidean(u, v)
        elif metric == "cityblock":
            val = distance.cityblock(u, v)
        elif metric == "chebyshev":
            val = distance.chebyshev(u, v)
        elif metric == "cosine":
            val = distance.cosine(u, v)
        elif metric == "correlation":
            val = distance.correlation(u, v)
        elif metric == "canberra":
            val = distance.canberra(u, v)
        elif metric == "braycurtis":
            val = distance.braycurtis(u, v)
        elif metric == "minkowski":
            val = distance.minkowski(u, v, p=p)
        elif metric == "hamming":
            val = distance.hamming(u, v)
        elif metric == "jaccard":
            val = distance.jaccard(u, v)
        elif metric == "jensenshannon":
            val = distance.jensenshannon(u, v)
        elif metric == "seuclidean":
            val = distance.seuclidean(u, v, V=np.array(w))
        elif metric == "mahalanobis":
            val = distance.mahalanobis(u, v, VI=np.array(vi))
        elif metric == "wminkowski":
            weights = np.array(w)
            if hasattr(distance, "wminkowski"):
                val = distance.wminkowski(u, v, p=p, w=weights)
            else:
                val = (np.sum((weights * np.abs(u - v)) ** p)) ** (1.0 / p)
        else:
            continue

        if np.isfinite(val):
            results.append({"case_id": cid, "value": float(val)})
    except Exception:
        pass

json.dump(results, sys.stdout)
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let json_input = serde_json::to_string(cases).ok()?;
        stdin.write_all(json_input.as_bytes()).ok()?;
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_oracle_or_skip(cases: &[DistanceCase]) -> Vec<OracleCase> {
    match run_scipy_oracle(cases) {
        Some(results) => results,
        None => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "SciPy oracle required but not available"
            );
            eprintln!("SciPy oracle not available, skipping diff test");
            Vec::new()
        }
    }
}

fn complete_oracle_map(
    test_id: &str,
    cases: &[DistanceCase],
    oracle_results: &[OracleCase],
) -> HashMap<String, f64> {
    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "{test_id} SciPy oracle returned partial coverage"
    );

    let oracle_map: HashMap<_, _> = oracle_results
        .iter()
        .map(|o| (o.case_id.clone(), o.value))
        .collect();
    assert_eq!(
        oracle_map.len(),
        cases.len(),
        "{test_id} SciPy oracle returned duplicate or missing case ids"
    );

    let missing_rust_outputs: Vec<&str> = cases
        .iter()
        .filter(|case| rust_output(case).is_none())
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_rust_outputs.is_empty(),
        "{test_id} missing Rust distance outputs: {:?}",
        missing_rust_outputs
    );

    let missing_oracle_cases: Vec<&str> = cases
        .iter()
        .filter(|case| !oracle_map.contains_key(&case.case_id))
        .map(|case| case.case_id.as_str())
        .collect();
    assert!(
        missing_oracle_cases.is_empty(),
        "{test_id} missing SciPy distance oracle results: {:?}",
        missing_oracle_cases
    );

    oracle_map
}

#[test]
fn diff_001_basic_distances() {
    let start = Instant::now();
    let cases = distance_cases();

    let basic_metrics = [
        "euclidean",
        "sqeuclidean",
        "cityblock",
        "chebyshev",
        "cosine",
        "correlation",
        "canberra",
        "braycurtis",
    ];

    let basic_cases: Vec<_> = cases
        .iter()
        .filter(|c| basic_metrics.contains(&c.metric.as_str()))
        .cloned()
        .collect();

    let oracle_results = scipy_oracle_or_skip(&basic_cases);
    if oracle_results.is_empty() {
        return;
    }

    let oracle_map = complete_oracle_map("diff_001_basic_distances", &basic_cases, &oracle_results);

    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &basic_cases {
        let rust_val = match rust_output(case) {
            Some(v) => v,
            None => continue,
        };
        let scipy_val = match oracle_map.get(&case.case_id) {
            Some(&v) => v,
            None => continue,
        };

        let abs_diff = (rust_val - scipy_val).abs();
        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            dim: case.u.len(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: TOL,
            pass: abs_diff <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_001_basic_distances".into(),
        category: "scipy.spatial.distance".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for diff in &diffs {
        assert!(
            diff.pass,
            "{} mismatch: rust={} scipy={} diff={}",
            diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
        );
    }
}

#[test]
fn diff_002_minkowski_variants() {
    let start = Instant::now();
    let cases = distance_cases();

    let minkowski_cases: Vec<_> = cases
        .iter()
        .filter(|c| c.metric == "minkowski")
        .cloned()
        .collect();

    let oracle_results = scipy_oracle_or_skip(&minkowski_cases);
    if oracle_results.is_empty() {
        return;
    }

    let oracle_map = complete_oracle_map(
        "diff_002_minkowski_variants",
        &minkowski_cases,
        &oracle_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &minkowski_cases {
        let rust_val = match rust_output(case) {
            Some(v) => v,
            None => continue,
        };
        let scipy_val = match oracle_map.get(&case.case_id) {
            Some(&v) => v,
            None => continue,
        };

        let abs_diff = (rust_val - scipy_val).abs();
        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            dim: case.u.len(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: TOL,
            pass: abs_diff <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_002_minkowski_variants".into(),
        category: "scipy.spatial.distance.minkowski".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for diff in &diffs {
        assert!(
            diff.pass,
            "{} mismatch: rust={} scipy={} diff={}",
            diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
        );
    }
}

#[test]
fn diff_003_binary_distances() {
    let start = Instant::now();
    let cases = distance_cases();

    let binary_cases: Vec<_> = cases
        .iter()
        .filter(|c| c.metric == "hamming" || c.metric == "jaccard")
        .cloned()
        .collect();

    let oracle_results = scipy_oracle_or_skip(&binary_cases);
    if oracle_results.is_empty() {
        return;
    }

    let oracle_map =
        complete_oracle_map("diff_003_binary_distances", &binary_cases, &oracle_results);

    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &binary_cases {
        let rust_val = match rust_output(case) {
            Some(v) => v,
            None => continue,
        };
        let scipy_val = match oracle_map.get(&case.case_id) {
            Some(&v) => v,
            None => continue,
        };

        let abs_diff = (rust_val - scipy_val).abs();
        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            dim: case.u.len(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: TOL,
            pass: abs_diff <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_003_binary_distances".into(),
        category: "scipy.spatial.distance.binary".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for diff in &diffs {
        assert!(
            diff.pass,
            "{} mismatch: rust={} scipy={} diff={}",
            diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
        );
    }
}

#[test]
fn diff_004_weighted_distances() {
    let start = Instant::now();
    let cases = distance_cases();

    let weighted_metrics = ["seuclidean", "mahalanobis", "wminkowski", "jensenshannon"];

    let weighted_cases: Vec<_> = cases
        .iter()
        .filter(|c| weighted_metrics.contains(&c.metric.as_str()))
        .cloned()
        .collect();

    let oracle_results = scipy_oracle_or_skip(&weighted_cases);
    if oracle_results.is_empty() {
        return;
    }

    let oracle_map = complete_oracle_map(
        "diff_004_weighted_distances",
        &weighted_cases,
        &oracle_results,
    );

    let mut diffs = Vec::new();
    let mut max_diff = 0.0_f64;

    for case in &weighted_cases {
        let rust_val = match rust_output(case) {
            Some(v) => v,
            None => continue,
        };
        let scipy_val = match oracle_map.get(&case.case_id) {
            Some(&v) => v,
            None => continue,
        };

        let abs_diff = (rust_val - scipy_val).abs();
        max_diff = max_diff.max(abs_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            dim: case.u.len(),
            rust_value: rust_val,
            scipy_value: scipy_val,
            abs_diff,
            tolerance: TOL,
            pass: abs_diff <= TOL,
        });
    }
    assert_eq!(
        diffs.len(),
        weighted_cases.len(),
        "weighted distance diff silently skipped {} cases",
        weighted_cases.len() - diffs.len()
    );

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_004_weighted_distances".into(),
        category: "scipy.spatial.distance.weighted".into(),
        case_count: diffs.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for diff in &diffs {
        assert!(
            diff.pass,
            "{} mismatch: rust={} scipy={} diff={}",
            diff.case_id, diff.rust_value, diff.scipy_value, diff.abs_diff
        );
    }
}
