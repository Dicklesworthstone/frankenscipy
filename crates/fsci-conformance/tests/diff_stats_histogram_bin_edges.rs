#![forbid(unsafe_code)]
//! Live SciPy / NumPy differential coverage for the
//! histogram-binning rules `numpy.histogram_bin_edges(data,
//! bins=method)` across all five fsci-supported methods:
//!   • "sqrt"     — n_bins = ceil(sqrt(n))
//!   • "rice"     — n_bins = ceil(2 * cbrt(n))
//!   • "scott"    — Scott's rule (n_bins from std / cbrt(n))
//!   • "fd"       — Freedman-Diaconis (IQR / cbrt(n))
//!   • "sturges"  — n_bins = ceil(1 + log2(n))
//!
//! Resolves [frankenscipy-lucey]. 3 datasets × 5 methods = 15
//! cases via subprocess. Each case compares the edge vector
//! element-wise (length match + max-abs aggregation).
//! Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::histogram_bin_edges;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    method: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    edges: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
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
    fs::create_dir_all(output_dir()).expect("create histogram_bin_edges diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize histogram_bin_edges diff log");
    fs::write(path, json).expect("write histogram_bin_edges diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=20).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0, 15.0, 20.0, 25.0,
                30.0,
            ],
        ),
        (
            "ties",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
    ];
    let methods = ["sqrt", "rice", "scott", "fd", "sturges"];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for method in methods {
            points.push(PointCase {
                case_id: format!("{name}_{method}"),
                method: method.into(),
                data: data.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def vec_or_none(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; method = case["method"]
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        val = vec_or_none(np.histogram_bin_edges(data, bins=method).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "edges": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize histogram_bin_edges query");
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
                "failed to spawn python3 for histogram_bin_edges oracle: {e}"
            );
            eprintln!("skipping histogram_bin_edges oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open histogram_bin_edges oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "histogram_bin_edges oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping histogram_bin_edges oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for histogram_bin_edges oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "histogram_bin_edges oracle failed: {stderr}"
        );
        eprintln!("skipping histogram_bin_edges oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse histogram_bin_edges oracle JSON"))
}

#[test]
fn diff_stats_histogram_bin_edges() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_edges) = &scipy_arm.edges else {
            continue;
        };
        let rust_edges = histogram_bin_edges(&case.data, &case.method);
        if rust_edges.len() != scipy_edges.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                method: case.method.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        for (a, b) in rust_edges.iter().zip(scipy_edges.iter()) {
            if a.is_finite() {
                max_local = max_local.max((a - b).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_histogram_bin_edges".into(),
        category: "numpy.histogram_bin_edges".into(),
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
                "histogram_bin_edges {} mismatch: {} abs={}",
                d.method, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "histogram_bin_edges conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
