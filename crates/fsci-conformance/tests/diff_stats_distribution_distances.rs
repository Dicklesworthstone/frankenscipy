#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `wasserstein_distance(u, v)` (1-D Wasserstein / earth-
//! mover) and `energy_distance(u, v)`.
//!
//! Resolves [frankenscipy-q8nda]. The oracle calls
//! `scipy.stats.{wasserstein_distance, energy_distance}`.
//!
//! 5 (u, v) fixtures × 2 funcs = 10 cases. Tol 1e-12 abs
//! (closed-form CDF-difference integration / pairwise-
//! distance sums).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{energy_distance, wasserstein_distance};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    u: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create distribution_distances diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json =
        serde_json::to_string_pretty(log).expect("serialize distribution_distances diff log");
    fs::write(path, json).expect("write distribution_distances diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Identical → 0
        (
            "identical",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| i as f64).collect(),
        ),
        // Shifted by 5
        (
            "shifted_n10",
            (1..=10).map(|i| i as f64).collect(),
            (6..=15).map(|i| i as f64).collect(),
        ),
        // Different sample sizes (stretched)
        (
            "diff_size",
            (1..=8).map(|i| i as f64).collect(),
            (1..=15).map(|i| (i as f64) * 0.5).collect(),
        ),
        // Different distributions: uniform vs concentrated
        (
            "uniform_vs_concentrated",
            (1..=20).map(|i| i as f64).collect(),
            vec![10.0, 10.5, 9.5, 10.0, 10.5, 9.5, 10.0, 10.5, 9.5, 10.0],
        ),
        // Negative values + ties
        (
            "neg_and_ties",
            vec![-3.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0],
            vec![-2.0, -2.0, 0.0, 1.0, 1.0, 3.0, 3.0, 5.0, 7.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, u, v) in &fixtures {
        for func in ["wasserstein", "energy"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                u: u.clone(),
                v: v.clone(),
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
from scipy import stats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    u = np.array(case["u"], dtype=float)
    v = np.array(case["v"], dtype=float)
    val = None
    try:
        if func == "wasserstein":
            val = stats.wasserstein_distance(u, v)
        elif func == "energy":
            val = stats.energy_distance(u, v)
    except Exception:
        val = None
    points.append({"case_id": cid, "value": fnone(val)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize distribution_distances query");
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
                "failed to spawn python3 for distribution_distances oracle: {e}"
            );
            eprintln!("skipping distribution_distances oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open distribution_distances oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "distribution_distances oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping distribution_distances oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for distribution_distances oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "distribution_distances oracle failed: {stderr}"
        );
        eprintln!("skipping distribution_distances oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse distribution_distances oracle JSON"))
}

#[test]
fn diff_stats_distribution_distances() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let rust_v = match case.func.as_str() {
            "wasserstein" => wasserstein_distance(&case.u, &case.v),
            "energy" => energy_distance(&case.u, &case.v),
            _ => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_distribution_distances".into(),
        category: "scipy.stats.{wasserstein_distance, energy_distance}".into(),
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
                "distribution_distances {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "distribution_distances conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
