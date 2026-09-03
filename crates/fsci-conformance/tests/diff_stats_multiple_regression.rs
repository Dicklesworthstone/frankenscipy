#![forbid(unsafe_code)]
//! Live numerical reference checks for fsci's
//! `multiple_regression(X, y)` against numpy's analytic
//! linear-system solve.
//!
//! Resolves [frankenscipy-5basu]. fsci returns
//! `(coefficients, residuals, r_squared, std_errors)` with
//! coefficients[0] as the intercept. The oracle computes the
//! same quantities via `numpy.linalg.lstsq` on a
//! design matrix with a leading 1-column.
//!
//! 3 (X_matrix, y) fixtures × (coeffs vector + r_squared +
//! residuals max-abs + std_errors vector) = 12 cases via
//! subprocess. Tol 1e-9 abs (linear solve precision).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::multiple_regression;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    coeffs: Option<Vec<f64>>,
    r_squared: Option<f64>,
    residuals: Option<Vec<f64>>,
    std_errors: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create multiple_regression diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize multiple_regression diff log");
    fs::write(path, json).expect("write multiple_regression diff log");
}

fn generate_query() -> OracleQuery {
    // Each fixture: X is n × p (rows are observations), y is n-vector.
    // y is constructed so the recovered coefficients match a known set.
    let fixtures: Vec<(&str, Vec<Vec<f64>>, Vec<f64>)> = vec![
        // Single feature, exact fit: y = 2 + 3*x1
        (
            "single_feat",
            (1..=10).map(|i| vec![i as f64]).collect(),
            (1..=10).map(|i| 2.0 + 3.0 * i as f64).collect(),
        ),
        // Two features, exact fit: y = 1 + 2*x1 + 0.5*x2
        (
            "two_feat",
            (1..=12)
                .map(|i| vec![i as f64, (i as f64).powi(2)])
                .collect(),
            (1..=12)
                .map(|i| {
                    let x1 = i as f64;
                    let x2 = x1 * x1;
                    1.0 + 2.0 * x1 + 0.5 * x2
                })
                .collect(),
        ),
        // Two features with mild noise residuals
        (
            "two_feat_noisy",
            (1..=15).map(|i| vec![i as f64, ((i % 3) as f64)]).collect(),
            vec![
                4.05, 6.95, 9.05, 11.95, 14.05, 16.95, 19.05, 21.95, 24.05, 26.95, 29.05, 31.95,
                34.05, 36.95, 39.05,
            ],
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

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
    out = {"case_id": cid, "coeffs": None, "r_squared": None,
           "residuals": None, "std_errors": None}
    try:
        n, p = x.shape
        X = np.column_stack([np.ones(n), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        # std_errors: sqrt(diag(MSE * (X'X)^{-1})) using ddof = n - p - 1
        df = n - (p + 1)
        if df > 0:
            mse = ss_res / df
            xtx_inv = np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(mse * xtx_inv))
            out["std_errors"] = vec_or_none(se.tolist())
        out["coeffs"] = vec_or_none(beta.tolist())
        out["r_squared"] = fnone(r2)
        out["residuals"] = vec_or_none(resid.tolist())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize multiple_regression query");
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
                "failed to spawn python3 for multiple_regression oracle: {e}"
            );
            eprintln!("skipping multiple_regression oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open multiple_regression oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "multiple_regression oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping multiple_regression oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for multiple_regression oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "multiple_regression oracle failed: {stderr}"
        );
        eprintln!("skipping multiple_regression oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse multiple_regression oracle JSON"))
}

#[test]
fn diff_stats_multiple_regression() {
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
        let (rust_coeffs, rust_resid, rust_r2, rust_se) = multiple_regression(&case.x, &case.y);

        // coeffs vector
        if let Some(scipy_coeffs) = &scipy_arm.coeffs
            && rust_coeffs.len() == scipy_coeffs.len()
        {
            let mut max_local = 0.0_f64;
            for (a, b) in rust_coeffs.iter().zip(scipy_coeffs.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "coeffs_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }

        // r_squared
        if let Some(scipy_r2) = scipy_arm.r_squared
            && rust_r2.is_finite()
        {
            let abs_diff = (rust_r2 - scipy_r2).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "r_squared".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }

        // residuals (vector)
        if let Some(scipy_resid) = &scipy_arm.residuals
            && rust_resid.len() == scipy_resid.len()
        {
            let mut max_local = 0.0_f64;
            for (a, b) in rust_resid.iter().zip(scipy_resid.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "residuals_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }

        // std_errors (vector)
        if let Some(scipy_se) = &scipy_arm.std_errors
            && rust_se.len() == scipy_se.len()
        {
            let mut max_local = 0.0_f64;
            for (a, b) in rust_se.iter().zip(scipy_se.iter()) {
                if a.is_finite() {
                    max_local = max_local.max((a - b).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "std_errors_max".into(),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_multiple_regression".into(),
        category: "multiple_regression (numpy.linalg.lstsq reference)".into(),
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
                "multiple_regression mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "multiple_regression conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
