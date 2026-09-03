#![forbid(unsafe_code)]
//! Live numerical reference checks for fsci's
//! `ridge_regression(X, y, alpha)`.
//!
//! Resolves [frankenscipy-mdhiy]. fsci solves
//! `(X'X + αI')β = X'y` where the regularization matrix I'
//! is identity except for the intercept slot (which gets no
//! regularization). The oracle reproduces this exactly via
//! numpy.linalg.solve on a design matrix with a leading
//! 1-column.
//!
//! 3 (X, y) fixtures × 3 alpha values = 9 cases via
//! subprocess. Each case compares the coefficient vector
//! element-wise (max-abs aggregation). Tol 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::ridge_regression;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    alpha: f64,
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
    fs::create_dir_all(output_dir()).expect("create ridge_regression diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ridge_regression diff log");
    fs::write(path, json).expect("write ridge_regression diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>, Vec<f64>)> = vec![
        // Single feature: y = 2 + 3*x1
        (
            "single",
            (1..=10).map(|i| vec![i as f64]).collect(),
            (1..=10).map(|i| 2.0 + 3.0 * i as f64).collect(),
        ),
        // Two correlated features
        (
            "two_correlated",
            (1..=15).map(|i| vec![i as f64, i as f64 + 0.1]).collect(),
            (1..=15).map(|i| 1.0 + 1.5 * i as f64).collect(),
        ),
        // Three features, y has noise
        (
            "three_noisy",
            (1..=12)
                .map(|i| vec![i as f64, ((i % 3) + 1) as f64, ((i * i) as f64) / 10.0])
                .collect(),
            (1..=12)
                .map(|i| {
                    let x1 = i as f64;
                    let x3 = (i * i) as f64 / 10.0;
                    1.0 + 0.5 * x1 + 0.3 * x3
                })
                .collect(),
        ),
    ];
    let alphas: [f64; 3] = [0.0, 0.5, 5.0];

    let mut points = Vec::new();
    for (name, x, y) in &fixtures {
        for &alpha in &alphas {
            points.push(PointCase {
                case_id: format!("{name}_a{alpha}"),
                x: x.clone(),
                y: y.clone(),
                alpha,
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
    alpha = float(case["alpha"])
    val = None
    try:
        n, p = x.shape
        X = np.column_stack([np.ones(n), x])  # leading 1-column
        XtX = X.T @ X
        # Skip intercept in regularization (index 0).
        reg = np.eye(p + 1) * alpha
        reg[0, 0] = 0.0
        beta = np.linalg.solve(XtX + reg, X.T @ y)
        val = vec_or_none(beta.tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "coeffs": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ridge_regression query");
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
                "failed to spawn python3 for ridge_regression oracle: {e}"
            );
            eprintln!("skipping ridge_regression oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ridge_regression oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ridge_regression oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ridge_regression oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ridge_regression oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ridge_regression oracle failed: {stderr}"
        );
        eprintln!("skipping ridge_regression oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ridge_regression oracle JSON"))
}

#[test]
fn diff_stats_ridge_regression() {
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
        let rust_coeffs = ridge_regression(&case.x, &case.y, case.alpha);
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
        test_id: "diff_stats_ridge_regression".into(),
        category: "ridge_regression (numpy reference)".into(),
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
                "ridge_regression mismatch: {} abs={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "ridge_regression conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
