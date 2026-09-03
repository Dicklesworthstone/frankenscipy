#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Siegel-slopes
//! robust regression `scipy.stats.siegelslopes(y, x)`.
//!
//! Resolves [frankenscipy-whsm4]. Distinct from
//! diff_stats_theilslopes.rs (Theil-Sen), Siegel uses the
//! median of per-point medians of pairwise slopes — more
//! robust to outliers in both x and y.
//!
//! 4 (x, y) fixtures × 2 arms (slope + intercept) = 8 cases
//! via subprocess. Tol 1e-12 abs (closed-form median of
//! medians).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::siegelslopes;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    slope: Option<f64>,
    intercept: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create siegelslopes diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize siegelslopes diff log");
    fs::write(path, json).expect("write siegelslopes diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Clean linear: y = 2x + 1
        (
            "clean_linear",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect(),
        ),
        // Mild noise on a positive slope
        (
            "noisy",
            (1..=12).map(|i| i as f64).collect(),
            vec![
                2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 14.3, 16.05, 18.2, 19.8, 21.9, 24.1,
            ],
        ),
        // Single severe outlier — Siegel should be robust
        (
            "outlier",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 7.0, 8.0, 9.0, 10.0],
        ),
        // Negative slope plus noise
        (
            "neg_slope",
            (1..=11).map(|i| i as f64).collect(),
            vec![10.05, 9.1, 7.85, 6.9, 6.05, 5.1, 4.0, 3.05, 2.1, 0.9, 0.1],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, x, y)| PointCase {
            case_id: name.into(),
            x,
            y,
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
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    try:
        # scipy.stats.siegelslopes signature: (y, x)
        res = stats.siegelslopes(y, x)
        # Returned as a SiegelslopesResult namedtuple (slope, intercept)
        points.append({
            "case_id": cid,
            "slope": fnone(res.slope),
            "intercept": fnone(res.intercept),
        })
    except Exception:
        points.append({"case_id": cid, "slope": None, "intercept": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize siegelslopes query");
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
                "failed to spawn python3 for siegelslopes oracle: {e}"
            );
            eprintln!("skipping siegelslopes oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open siegelslopes oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "siegelslopes oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping siegelslopes oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for siegelslopes oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "siegelslopes oracle failed: {stderr}"
        );
        eprintln!("skipping siegelslopes oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse siegelslopes oracle JSON"))
}

#[test]
fn diff_stats_siegelslopes() {
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
        let result = siegelslopes(&case.x, &case.y);

        if let Some(scipy_slope) = scipy_arm.slope
            && result.slope.is_finite()
        {
            let abs_diff = (result.slope - scipy_slope).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "slope".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_int) = scipy_arm.intercept
            && result.intercept.is_finite()
        {
            let abs_diff = (result.intercept - scipy_int).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "intercept".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_siegelslopes".into(),
        category: "scipy.stats.siegelslopes".into(),
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
                "siegelslopes mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "siegelslopes conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
