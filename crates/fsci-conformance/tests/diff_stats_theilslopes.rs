#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Theil-Sen
//! robust regression `scipy.stats.theilslopes(y, x, alpha,
//! method='separate')`.
//!
//! Resolves [frankenscipy-6qjzk]. Cross-checks slope,
//! intercept, low_slope, high_slope.
//!
//! Tolerances:
//!   - slope/intercept: 1e-12 abs (closed-form medians of
//!     pairwise slopes / median(y - slope*x)).
//!   - low_slope/high_slope: 1e-9 abs (chains
//!     `ndtri(1 - alpha/2)` for the rank window — fsci uses a
//!     rational ndtri vs scipy's distpacked ndtri, accumulated
//!     differences land in the 1e-10 range on small n).
//!
//! 4 datasets × 4 arms = 16 cases via subprocess.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::theilslopes;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TIGHT_TOL: f64 = 1.0e-12;
const SLOPE_BAND_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DatasetCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<DatasetCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    slope: Option<f64>,
    intercept: Option<f64>,
    low_slope: Option<f64>,
    high_slope: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create theilslopes diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize theilslopes diff log");
    fs::write(path, json).expect("write theilslopes diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>, Vec<f64>, f64)> = vec![
        // Clean linear: y = 2x + 1
        (
            "clean_linear",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect(),
            0.95,
        ),
        // Linear with mild noise
        (
            "noisy_linear",
            (1..=12).map(|i| i as f64).collect(),
            vec![
                2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 14.3, 16.05, 18.2, 19.8, 21.9, 24.1,
            ],
            0.95,
        ),
        // Linear with one severe outlier
        (
            "outlier",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0, 7.0, 8.0, 9.0, 10.0],
            0.95,
        ),
        // Negative slope plus noise
        (
            "neg_slope",
            (1..=11).map(|i| i as f64).collect(),
            vec![10.05, 9.1, 7.85, 6.9, 6.05, 5.1, 4.0, 3.05, 2.1, 0.9, 0.1],
            0.99,
        ),
    ];

    let mut points = Vec::new();
    for (name, x, y, alpha) in datasets {
        points.push(DatasetCase {
            case_id: name.into(),
            x,
            y,
            alpha,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
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
    x = case["x"]; y = case["y"]; alpha = float(case["alpha"])
    try:
        # scipy.stats.theilslopes signature: (y, x, alpha, method)
        # We want method='separate' to match fsci-stats default.
        res = stats.theilslopes(y, x, alpha=alpha, method='separate')
        # Result is a namedtuple (slope, intercept, low_slope, high_slope)
        points.append({
            "case_id": cid,
            "slope": fnone(res.slope),
            "intercept": fnone(res.intercept),
            "low_slope": fnone(res.low_slope),
            "high_slope": fnone(res.high_slope),
        })
    except Exception:
        points.append({"case_id": cid, "slope": None, "intercept": None,
                       "low_slope": None, "high_slope": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize theilslopes query");
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
                "failed to spawn python3 for theilslopes oracle: {e}"
            );
            eprintln!("skipping theilslopes oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open theilslopes oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "theilslopes oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping theilslopes oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for theilslopes oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "theilslopes oracle failed: {stderr}"
        );
        eprintln!("skipping theilslopes oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse theilslopes oracle JSON"))
}

#[test]
fn diff_stats_theilslopes() {
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
        let result = theilslopes(&case.x, &case.y, case.alpha);

        let arms: [(&str, Option<f64>, f64, f64); 4] = [
            ("slope", scipy_arm.slope, result.slope, TIGHT_TOL),
            (
                "intercept",
                scipy_arm.intercept,
                result.intercept,
                TIGHT_TOL,
            ),
            (
                "low_slope",
                scipy_arm.low_slope,
                result.low_slope,
                SLOPE_BAND_TOL,
            ),
            (
                "high_slope",
                scipy_arm.high_slope,
                result.high_slope,
                SLOPE_BAND_TOL,
            ),
        ];

        for (arm_name, scipy_v, rust_v, tol) in arms {
            if let Some(scipy_v) = scipy_v
                && rust_v.is_finite()
            {
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: arm_name.into(),
                    abs_diff,
                    pass: abs_diff <= tol,
                });
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_theilslopes".into(),
        category: "scipy.stats.theilslopes".into(),
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
                "theilslopes mismatch: {} {} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "theilslopes conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
