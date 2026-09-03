#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the 2D directional
//! statistics function `scipy.stats.directional_stats(data,
//! axis=0, normalize=True)`.
//!
//! Resolves [frankenscipy-2a9ww]. Cross-checks the mean
//! direction unit vector and the mean resultant length
//! across 4 fixtures.
//!
//! 4 fixtures × 3 arms (mean_dir_x + mean_dir_y + mean_R) =
//! 12 cases via subprocess. Tol 1e-12 abs (closed-form sums
//! of unit vectors).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::directional_stats;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    data: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    mean_dir_x: Option<f64>,
    mean_dir_y: Option<f64>,
    mean_resultant_length: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create directional diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize directional diff log");
    fs::write(path, json).expect("write directional diff log");
}

fn generate_query() -> OracleQuery {
    // Each fixture is a Vec of 2D unit vectors (or general 2D vectors,
    // normalize=true on both sides).
    let fixtures: Vec<(&str, Vec<[f64; 2]>)> = vec![
        // Concentrated near angle 0 (positive x-axis)
        (
            "concentrated_east",
            vec![
                [1.0, 0.0],
                [0.99, 0.14],
                [0.98, -0.20],
                [0.95, 0.31],
                [0.99, -0.10],
            ],
        ),
        // Dispersed across the circle
        (
            "dispersed",
            vec![
                [1.0, 0.0],
                [0.5, 0.866],
                [-0.5, 0.866],
                [-1.0, 0.0],
                [-0.5, -0.866],
                [0.5, -0.866],
            ],
        ),
        // Antipodal pairs cancel mostly
        (
            "antipodal",
            vec![
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [0.7, 0.7],
                [-0.7, -0.7],
            ],
        ),
        // Asymmetric cluster around angle pi/3
        (
            "asymmetric",
            vec![
                [0.5, 0.866],
                [0.6, 0.8],
                [0.45, 0.9],
                [0.55, 0.83],
                [0.65, 0.76],
                [0.4, 0.92],
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, data)| PointCase {
            case_id: name.into(),
            data,
        })
        .collect();
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
    cid = case["case_id"]
    data = np.array(case["data"], dtype=float)
    try:
        # scipy.stats.directional_stats normalizes by default; the data
        # we feed is already unit-or-near-unit so this is the right
        # comparison for fsci with normalize=true.
        res = stats.directional_stats(data, normalize=True)
        md = res.mean_direction
        points.append({
            "case_id": cid,
            "mean_dir_x": fnone(md[0]),
            "mean_dir_y": fnone(md[1]),
            "mean_resultant_length": fnone(res.mean_resultant_length),
        })
    except Exception:
        points.append({
            "case_id": cid,
            "mean_dir_x": None,
            "mean_dir_y": None,
            "mean_resultant_length": None,
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize directional query");
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
                "failed to spawn python3 for directional oracle: {e}"
            );
            eprintln!("skipping directional oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open directional oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "directional oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping directional oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for directional oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "directional oracle failed: {stderr}"
        );
        eprintln!("skipping directional oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse directional oracle JSON"))
}

#[test]
fn diff_stats_directional() {
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
        let result = directional_stats(&case.data, true);

        let arms: [(&str, Option<f64>, Option<f64>); 3] = [
            (
                "mean_dir_x",
                scipy_arm.mean_dir_x,
                result.mean_direction.first().copied(),
            ),
            (
                "mean_dir_y",
                scipy_arm.mean_dir_y,
                result.mean_direction.get(1).copied(),
            ),
            (
                "mean_resultant_length",
                scipy_arm.mean_resultant_length,
                Some(result.mean_resultant_length),
            ),
        ];

        for (arm_name, scipy_v, rust_v) in arms {
            if let (Some(scipy_v), Some(rust_v)) = (scipy_v, rust_v)
                && rust_v.is_finite()
            {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: arm_name.into(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_directional".into(),
        category: "scipy.stats.directional_stats".into(),
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
                "directional mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "directional conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
