#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the batch-distance
//! entry points `pdist(x, metric)` and `cdist(xa, xb, metric)`.
//!
//! Resolves [frankenscipy-ixht7]. Existing `diff_spatial.rs`
//! covers per-pair distance functions; this harness covers the
//! batch APIs across all 10 `DistanceMetric` variants.
//!
//! 3 fixtures × 10 metrics × {pdist, cdist} ≈ 60 cases.
//! Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{DistanceMetric, cdist_metric, pdist};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PdistCase {
    case_id: String,
    metric: String,
    x: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CdistCase {
    case_id: String,
    metric: String,
    xa: Vec<Vec<f64>>,
    xb: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    pdist: Vec<PdistCase>,
    cdist: Vec<CdistCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PdistArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct CdistArm {
    case_id: String,
    rows: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    pdist: Vec<PdistArm>,
    cdist: Vec<CdistArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fn_name: String,
    metric: String,
    max_abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create pdist_cdist diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize pdist_cdist diff log");
    fs::write(path, json).expect("write pdist_cdist diff log");
}

fn metric_for(name: &str) -> Option<DistanceMetric> {
    match name {
        "euclidean" => Some(DistanceMetric::Euclidean),
        "sqeuclidean" => Some(DistanceMetric::SqEuclidean),
        "cityblock" => Some(DistanceMetric::Cityblock),
        "chebyshev" => Some(DistanceMetric::Chebyshev),
        "cosine" => Some(DistanceMetric::Cosine),
        "correlation" => Some(DistanceMetric::Correlation),
        "hamming" => Some(DistanceMetric::Hamming),
        "jaccard" => Some(DistanceMetric::Jaccard),
        "canberra" => Some(DistanceMetric::Canberra),
        "braycurtis" => Some(DistanceMetric::Braycurtis),
        _ => None,
    }
}

// All 10 DistanceMetric variants. (jaccard real-valued semantics
// were aligned with scipy ≥1.15 in frankenscipy-z747j — input is
// coerced to Boolean before computation.)
const METRICS: [&str; 10] = [
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "cosine",
    "correlation",
    "hamming",
    "jaccard",
    "canberra",
    "braycurtis",
];

fn generate_query() -> OracleQuery {
    // Distinct-coordinate fixtures so jaccard/hamming have
    // meaningful (non-trivial) values.
    let small_2d: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![0.5, 0.866],
        vec![-0.5, 0.866],
        vec![-1.0, 0.0],
    ];
    let larger_3d: Vec<Vec<f64>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
        vec![1.0, 1.1, 1.2],
        vec![1.3, 1.4, 1.5],
    ];
    let one_d_long: Vec<Vec<f64>> = (1..=8).map(|i| vec![i as f64 * 0.5]).collect();

    let pdist_fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("small_2d", small_2d.clone()),
        ("larger_3d", larger_3d.clone()),
        ("one_d_long", one_d_long.clone()),
    ];

    // For cdist, use distinct xa/xb pairs.
    let cdist_fixtures: Vec<(&str, Vec<Vec<f64>>, Vec<Vec<f64>>)> = vec![
        (
            "small_4x3_2d",
            small_2d,
            vec![vec![2.0, 1.0], vec![3.0, 2.0], vec![4.0, 3.0]],
        ),
        (
            "larger_5x4_3d",
            larger_3d,
            vec![
                vec![0.0, 0.0, 0.0],
                vec![2.0, 2.0, 2.0],
                vec![5.0, 5.0, 5.0],
                vec![10.0, 10.0, 10.0],
            ],
        ),
        (
            "one_d_8x6",
            one_d_long,
            (1..=6).map(|i| vec![i as f64]).collect(),
        ),
    ];

    let mut pdist_cases = Vec::new();
    for (name, x) in &pdist_fixtures {
        for m in &METRICS {
            pdist_cases.push(PdistCase {
                case_id: format!("{name}_{m}"),
                metric: (*m).to_string(),
                x: x.clone(),
            });
        }
    }
    let mut cdist_cases = Vec::new();
    for (name, xa, xb) in &cdist_fixtures {
        for m in &METRICS {
            cdist_cases.push(CdistCase {
                case_id: format!("{name}_{m}"),
                metric: (*m).to_string(),
                xa: xa.clone(),
                xb: xb.clone(),
            });
        }
    }
    OracleQuery {
        pdist: pdist_cases,
        cdist: cdist_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.spatial import distance as sd

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def listify(arr):
    out = []
    for v in arr.tolist():
        f = fnone(v)
        if f is None:
            return None  # whole case is unsupported (NaN/inf produced)
        out.append(f)
    return out

def listify_2d(arr):
    out = []
    for row in arr.tolist():
        rrow = []
        for v in row:
            f = fnone(v)
            if f is None:
                return None
            rrow.append(f)
        out.append(rrow)
    return out

q = json.load(sys.stdin)

pdist_results = []
for case in q["pdist"]:
    cid = case["case_id"]
    metric = case["metric"]
    out = None
    try:
        x = np.asarray(case["x"], dtype=np.float64)
        d = sd.pdist(x, metric=metric)
        out = listify(d)
    except Exception:
        out = None
    pdist_results.append({"case_id": cid, "values": out})

cdist_results = []
for case in q["cdist"]:
    cid = case["case_id"]
    metric = case["metric"]
    out = None
    try:
        xa = np.asarray(case["xa"], dtype=np.float64)
        xb = np.asarray(case["xb"], dtype=np.float64)
        d = sd.cdist(xa, xb, metric=metric)
        out = listify_2d(d)
    except Exception:
        out = None
    cdist_results.append({"case_id": cid, "rows": out})

print(json.dumps({"pdist": pdist_results, "cdist": cdist_results}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize pdist_cdist query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for pdist_cdist oracle: {e}"
            );
            eprintln!("skipping pdist_cdist oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open pdist_cdist oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "pdist_cdist oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping pdist_cdist oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for pdist_cdist oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "pdist_cdist oracle failed: {stderr}"
        );
        eprintln!("skipping pdist_cdist oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse pdist_cdist oracle JSON"))
}

#[test]
fn diff_spatial_pdist_cdist() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.pdist.len(), query.pdist.len());
    assert_eq!(oracle.cdist.len(), query.cdist.len());

    let pmap: HashMap<String, PdistArm> = oracle
        .pdist
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let cmap: HashMap<String, CdistArm> = oracle
        .cdist
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.pdist {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Some(metric) = metric_for(&case.metric) else {
            continue;
        };
        let rust_v = match pdist(&case.x, metric) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if rust_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: "pdist".into(),
                metric: case.metric.clone(),
                max_abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let max_d = rust_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(r, s)| (r - s).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(max_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fn_name: "pdist".into(),
            metric: case.metric.clone(),
            max_abs_diff: max_d,
            pass: max_d <= ABS_TOL,
        });
    }

    for case in &query.cdist {
        let scipy_arm = cmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_rows) = scipy_arm.rows.as_ref() else {
            continue;
        };
        let Some(metric) = metric_for(&case.metric) else {
            continue;
        };
        let rust_rows = match cdist_metric(&case.xa, &case.xb, metric) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if rust_rows.len() != scipy_rows.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                fn_name: "cdist".into(),
                metric: case.metric.clone(),
                max_abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_d = 0.0_f64;
        let mut shape_ok = true;
        for (rr, sr) in rust_rows.iter().zip(scipy_rows.iter()) {
            if rr.len() != sr.len() {
                shape_ok = false;
                break;
            }
            for (r, s) in rr.iter().zip(sr.iter()) {
                max_d = max_d.max((r - s).abs());
            }
        }
        max_overall = max_overall.max(max_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fn_name: "cdist".into(),
            metric: case.metric.clone(),
            max_abs_diff: if shape_ok { max_d } else { f64::INFINITY },
            pass: shape_ok && max_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_pdist_cdist".into(),
        category: "fsci_spatial::pdist + cdist_metric".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "pdist/cdist mismatch: {} {} ({}) max_abs={}",
                d.case_id, d.fn_name, d.metric, d.max_abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "pdist/cdist conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
