#![forbid(unsafe_code)]
//! Live numerical reference checks for two time-series
//! diagnostic utilities that scipy doesn't expose directly:
//!   • `durbin_watson(residuals)` — Σ(diffs²) / Σ(r²)
//!   • `acf(data, max_lag)` — biased autocorrelation function
//!
//! Resolves [frankenscipy-c5s8h]. The Durbin-Watson statistic
//! is in statsmodels (not always available); the formula is
//! one line so the oracle reproduces it analytically in numpy.
//! The ACF formula matches numpy's `np.correlate(centered,
//! centered, mode='full')[n-1:n-1+max_lag+1] / var`.
//!
//! 4 DW residual fixtures + 3 acf (data, max_lag) fixtures =
//! 7 cases via subprocess. Tol 1e-12 abs (closed-form ratios).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{acf, durbin_watson};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    max_lag: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    vector: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create dw_acf diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dw_acf diff log");
    fs::write(path, json).expect("write dw_acf diff log");
}

fn generate_query() -> OracleQuery {
    // DW residual fixtures
    let dw_fixtures: Vec<(&str, Vec<f64>)> = vec![
        // Independent residuals (DW ≈ 2)
        (
            "iid_like",
            vec![1.0, -1.0, 0.5, -0.5, 1.2, -0.9, 0.3, -0.7, 1.1, -1.0],
        ),
        // Positive autocorrelation (DW < 2)
        (
            "pos_autocorr",
            vec![0.1, 0.2, 0.4, 0.7, 1.1, 1.4, 1.5, 1.3, 0.9, 0.4],
        ),
        // Negative autocorrelation (DW > 2)
        (
            "neg_autocorr",
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ),
        // Trending residuals
        (
            "trending",
            (1..=12).map(|i| (i as f64) * 0.5 - 3.0).collect(),
        ),
    ];

    // ACF fixtures (data, max_lag)
    let acf_fixtures: Vec<(&str, Vec<f64>, u64)> = vec![
        ("compact_lag5", (1..=20).map(|i| i as f64).collect(), 5),
        (
            "noisy_lag3",
            vec![1.0, 1.5, 0.8, 1.7, 0.9, 1.4, 1.0, 1.6, 1.1, 1.5, 0.7, 1.4],
            3,
        ),
        (
            "alternating_lag4",
            vec![
                1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            ],
            4,
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &dw_fixtures {
        points.push(PointCase {
            case_id: format!("dw_{name}"),
            func: "durbin_watson".into(),
            data: data.clone(),
            max_lag: 0,
        });
    }
    for (name, data, max_lag) in &acf_fixtures {
        points.push(PointCase {
            case_id: format!("acf_{name}"),
            func: "acf".into(),
            data: data.clone(),
            max_lag: *max_lag,
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
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "scalar": None, "vector": None}
    try:
        if func == "durbin_watson":
            num = float(np.sum(np.diff(data) ** 2))
            den = float(np.sum(data ** 2))
            out["scalar"] = fnone(num / den) if den != 0 else None
        elif func == "acf":
            ml = int(case["max_lag"])
            n = len(data)
            mean = float(data.mean())
            cent = data - mean
            var = float(np.sum(cent * cent))
            if var == 0:
                out["vector"] = [1.0] * (ml + 1)
            else:
                acf_vals = []
                for lag in range(min(ml, n - 1) + 1):
                    s = float(np.sum(cent[:n - lag] * cent[lag:]))
                    acf_vals.append(s / var)
                out["vector"] = vec_or_none(acf_vals)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize dw_acf query");
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
                "failed to spawn python3 for dw_acf oracle: {e}"
            );
            eprintln!("skipping dw_acf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open dw_acf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dw_acf oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dw_acf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for dw_acf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dw_acf oracle failed: {stderr}"
        );
        eprintln!("skipping dw_acf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dw_acf oracle JSON"))
}

#[test]
fn diff_stats_dw_acf() {
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
        match case.func.as_str() {
            "durbin_watson" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = durbin_watson(&case.data);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "acf" => {
                if let Some(scipy_vec) = &scipy_arm.vector {
                    let rust_vec = acf(&case.data, case.max_lag as usize);
                    if rust_vec.len() == scipy_vec.len() {
                        let mut max_local = 0.0_f64;
                        for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
                            if a.is_finite() {
                                max_local = max_local.max((a - b).abs());
                            }
                        }
                        max_overall = max_overall.max(max_local);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_dw_acf".into(),
        category: "durbin_watson + acf (numpy reference)".into(),
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
                "dw_acf {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "dw_acf conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
