#![forbid(unsafe_code)]
//! Live scipy.spatial.distance parity for fsci_spatial::metric_distance
//! dispatcher across all DistanceMetric variants.
//!
//! Resolves [frankenscipy-tkjro]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_spatial::{DistanceMetric, metric_distance};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    metric: String,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    metric: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create md diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn parse_metric(s: &str) -> Option<DistanceMetric> {
    match s {
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

fn generate_query() -> OracleQuery {
    let pair_a = vec![1.0, 2.0, 3.0, 4.0];
    let pair_b = vec![2.0, 4.0, 6.0, 8.0];
    let pair_c = vec![1.0, 0.0, 1.0, 1.0];
    let pair_d = vec![0.0, 1.0, 1.0, 0.0];
    let pair_e = vec![5.0, 1.0, -2.0, 3.0];
    let pair_f = vec![3.0, 2.0, 4.0, -1.0];

    let mut points = Vec::new();
    let metrics = [
        "euclidean",
        "sqeuclidean",
        "cityblock",
        "chebyshev",
        "cosine",
        "correlation",
        "canberra",
        "braycurtis",
    ];
    // Boolean-style for hamming/jaccard
    let hj_a = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let hj_b = vec![1.0, 1.0, 0.0, 0.0, 1.0];

    for &m in &metrics {
        for (label, a, b) in [("ab", &pair_a, &pair_b), ("ef", &pair_e, &pair_f)] {
            points.push(Case {
                case_id: format!("{m}_{label}"),
                metric: m.into(),
                a: a.clone(),
                b: b.clone(),
            });
        }
        // canberra not great when components hit zero; add a third pair for robustness
        if !matches!(m, "correlation") {
            points.push(Case {
                case_id: format!("{m}_cd"),
                metric: m.into(),
                a: pair_c.clone(),
                b: pair_d.clone(),
            });
        }
    }
    for m in ["hamming", "jaccard"] {
        points.push(Case {
            case_id: format!("{m}_bool"),
            metric: m.into(),
            a: hj_a.clone(),
            b: hj_b.clone(),
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.spatial import distance as sd

def f(metric, a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    if metric == "euclidean":   return float(sd.euclidean(a, b))
    if metric == "sqeuclidean": return float(sd.sqeuclidean(a, b))
    if metric == "cityblock":   return float(sd.cityblock(a, b))
    if metric == "chebyshev":   return float(sd.chebyshev(a, b))
    if metric == "cosine":      return float(sd.cosine(a, b))
    if metric == "correlation": return float(sd.correlation(a, b))
    if metric == "hamming":     return float(sd.hamming(a, b))
    if metric == "jaccard":     return float(sd.jaccard(a, b))
    if metric == "canberra":    return float(sd.canberra(a, b))
    if metric == "braycurtis":  return float(sd.braycurtis(a, b))
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; m = case["metric"]
    a = case["a"]; b = case["b"]
    try:
        v = f(m, a, b)
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for md oracle: {e}"
            );
            eprintln!("skipping md oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "md oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping md oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for md oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "md oracle failed: {stderr}"
        );
        eprintln!("skipping md oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse md oracle JSON"))
}

#[test]
fn diff_spatial_metric_distance_dispatch() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let Some(metric) = parse_metric(&case.metric) else {
            continue;
        };
        let actual = metric_distance(&case.a, &case.b, metric);
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            metric: case.metric.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_spatial_metric_distance_dispatch".into(),
        category: "fsci_spatial::metric_distance dispatcher vs scipy.spatial.distance".into(),
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
                "{} mismatch: {} abs_diff={}",
                d.metric, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "md conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
