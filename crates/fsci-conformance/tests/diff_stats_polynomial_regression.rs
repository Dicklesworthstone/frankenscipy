#![forbid(unsafe_code)]
//! Live SciPy / NumPy differential coverage for the
//! polynomial-regression coefficient solver
//! `numpy.polyfit(x, y, degree)`.
//!
//! Resolves [frankenscipy-3lbml]. fsci's
//! `polynomial_regression(x, y, degree)` returns
//! `[c_0, c_1, ..., c_degree]` (low-to-high power order, with
//! intercept first). numpy.polyfit returns the same
//! coefficients in REVERSE order (high-to-low). The harness
//! reverses numpy's output before comparison.
//!
//! 3 (x, y) fixtures × 3 degrees = 9 cases via subprocess.
//! Each case compares the coefficient vector with max-abs
//! aggregation. Tol 1e-9 abs (linear-system solve precision).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::polynomial_regression;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    degree: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    coeffs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create polynomial_regression diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize polynomial_regression diff log");
    fs::write(path, json).expect("write polynomial_regression diff log");
}

fn generate_query() -> OracleQuery {
    // Each fixture is a (name, x, y) tuple. We test multiple degrees on
    // each. y is constructed as a known polynomial so the recovered
    // coefficients should match the construction (modulo solve precision).
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        // y = 2x + 1
        (
            "linear",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect(),
        ),
        // y = 0.5x² - x + 2
        (
            "quadratic",
            (1..=12).map(|i| i as f64).collect(),
            (1..=12)
                .map(|i| {
                    let x = i as f64;
                    0.5 * x * x - x + 2.0
                })
                .collect(),
        ),
        // y = 0.1x³ + x² - 0.5x + 1
        (
            "cubic",
            (1..=15).map(|i| i as f64).collect(),
            (1..=15)
                .map(|i| {
                    let x = i as f64;
                    0.1 * x * x * x + x * x - 0.5 * x + 1.0
                })
                .collect(),
        ),
    ];
    let degrees: [u64; 3] = [1, 2, 3];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for &degree in &degrees {
            points.push(PointCase {
                case_id: format!("{name}_d{degree}"),
                x: x.clone(),
                y: y.clone(),
                degree,
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
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    degree = int(case["degree"])
    val = None
    try:
        # numpy returns coefficients in high-to-low power order; fsci
        # returns them low-to-high. Reverse here so the order matches.
        coeffs_hi_to_lo = np.polyfit(x, y, degree).tolist()
        val = vec_or_none(list(reversed(coeffs_hi_to_lo)))
    except Exception:
        val = None
    points.append({"case_id": cid, "coeffs": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize polynomial_regression query");
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
                "failed to spawn python3 for polynomial_regression oracle: {e}"
            );
            eprintln!("skipping polynomial_regression oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open polynomial_regression oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "polynomial_regression oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping polynomial_regression oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for polynomial_regression oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "polynomial_regression oracle failed: {stderr}"
        );
        eprintln!("skipping polynomial_regression oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse polynomial_regression oracle JSON"))
}

#[test]
fn diff_stats_polynomial_regression() {
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
        let Some(scipy_coeffs) = &scipy_arm.coeffs else {
            continue;
        };
        let rust_coeffs = polynomial_regression(&case.x, &case.y, case.degree as usize);
        if rust_coeffs.len() != scipy_coeffs.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        for (a, b) in rust_coeffs.iter().zip(scipy_coeffs.iter()) {
            if a.is_finite() {
                max_local = max_local.max((a - b).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_polynomial_regression".into(),
        category: "numpy.polyfit (compared via fsci's polynomial_regression)".into(),
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
                "polynomial_regression mismatch: {} abs={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "polynomial_regression conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
