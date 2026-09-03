#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the linregress
//! "extras" not exercised by `diff_stats_correlation.rs`:
//!
//!   • `linregress(x, y).stderr`           — slope std err
//!   • `linregress(x, y).intercept_stderr` — intercept std err
//!   • `linregress_ci(x, y, alpha)`         — slope/intercept CI
//!     bounds at alpha=0.05 (4 endpoints)
//!
//! Resolves [frankenscipy-0yjf5]. The existing correlation
//! harness already pins slope/intercept/rvalue/pvalue, so this
//! file targets the orthogonal closed-form residual std-error
//! arithmetic plus the StudentT::ppf chain used by the CI.
//!
//! 4 (x, y) datasets × 6 arms = 24 cases via subprocess.
//! Tolerances:
//!   - stderr / intercept_stderr : 1e-12 abs (closed-form
//!     residual sums).
//!   - slope_lo/hi, intercept_lo/hi : 1e-9 abs (chains scipy's
//!     `t.ppf` vs fsci's `StudentT::ppf`; small drift expected
//!     at the 1e-11 level on small df).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{linregress, linregress_ci};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STDERR_TOL: f64 = 1.0e-12;
const CI_TOL: f64 = 1.0e-9;
const ALPHA: f64 = 0.05;
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
    stderr: Option<f64>,
    intercept_stderr: Option<f64>,
    slope_lo: Option<f64>,
    slope_hi: Option<f64>,
    intercept_lo: Option<f64>,
    intercept_hi: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create linregress-extras diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize linregress-extras diff log");
    fs::write(path, json).expect("write linregress-extras diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // Clean linear: y = 3x + 2
        (
            "clean_linear",
            (1..=8).map(|i| i as f64).collect(),
            (1..=8).map(|i| 3.0 * i as f64 + 2.0).collect(),
        ),
        // Mild residual noise
        (
            "noisy_linear",
            (1..=10).map(|i| i as f64).collect(),
            vec![1.05, 1.95, 3.10, 3.90, 5.20, 5.85, 7.05, 7.95, 9.10, 9.95],
        ),
        // Larger residual variance — exercises bigger stderr branch
        (
            "wide_noise",
            (1..=12).map(|i| i as f64).collect(),
            vec![
                2.0, 5.5, 4.0, 8.0, 6.5, 10.5, 9.0, 13.0, 11.5, 15.5, 14.0, 18.0,
            ],
        ),
        // Negative slope
        (
            "neg_slope",
            (0..9).map(|i| 0.5 * i as f64).collect(),
            vec![10.05, 9.0, 8.05, 7.0, 6.05, 5.0, 4.05, 3.0, 2.05],
        ),
    ];

    let points = datasets
        .into_iter()
        .map(|(name, x, y)| DatasetCase {
            case_id: name.into(),
            x,
            y,
            alpha: ALPHA,
        })
        .collect();
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
        res = stats.linregress(x, y)
        n = len(x)
        # scipy doesn't expose a `linregress_ci`; compute manually
        # the same way fsci does: bounds = est ± t_crit * stderr.
        df = n - 2
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        slope_margin = t_crit * res.stderr
        intercept_margin = t_crit * res.intercept_stderr
        points.append({
            "case_id": cid,
            "stderr": fnone(res.stderr),
            "intercept_stderr": fnone(res.intercept_stderr),
            "slope_lo": fnone(res.slope - slope_margin),
            "slope_hi": fnone(res.slope + slope_margin),
            "intercept_lo": fnone(res.intercept - intercept_margin),
            "intercept_hi": fnone(res.intercept + intercept_margin),
        })
    except Exception:
        points.append({"case_id": cid, "stderr": None,
                       "intercept_stderr": None, "slope_lo": None,
                       "slope_hi": None, "intercept_lo": None,
                       "intercept_hi": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize linregress-extras query");
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
                "failed to spawn python3 for linregress-extras oracle: {e}"
            );
            eprintln!("skipping linregress-extras oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open linregress-extras oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linregress-extras oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping linregress-extras oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for linregress-extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linregress-extras oracle failed: {stderr}"
        );
        eprintln!("skipping linregress-extras oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse linregress-extras oracle JSON"))
}

#[test]
fn diff_stats_linregress_extras() {
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
        let lr = linregress(&case.x, &case.y);
        let ci = linregress_ci(&case.x, &case.y, case.alpha);

        let arms: [(&str, Option<f64>, f64, f64); 6] = [
            ("stderr", scipy_arm.stderr, lr.stderr, STDERR_TOL),
            (
                "intercept_stderr",
                scipy_arm.intercept_stderr,
                lr.intercept_stderr,
                STDERR_TOL,
            ),
            ("slope_lo", scipy_arm.slope_lo, ci.slope_lo, CI_TOL),
            ("slope_hi", scipy_arm.slope_hi, ci.slope_hi, CI_TOL),
            (
                "intercept_lo",
                scipy_arm.intercept_lo,
                ci.intercept_lo,
                CI_TOL,
            ),
            (
                "intercept_hi",
                scipy_arm.intercept_hi,
                ci.intercept_hi,
                CI_TOL,
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
        test_id: "diff_stats_linregress_extras".into(),
        category: "scipy.stats.linregress (stderr + CI)".into(),
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
                "linregress-extras mismatch: {} {} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "linregress-extras conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
