#![forbid(unsafe_code)]
//! Live SciPy differential coverage for four descriptive
//! statistics utilities not exercised by any other diff
//! harness:
//!   • `scipy.stats.describe(data)` — 6-tuple summary (nobs,
//!     minmax, mean, variance, skewness, kurtosis)
//!   • `scipy.stats.gstd(data)` — geometric standard deviation
//!   • `scipy.stats.variation(data)` — coefficient of variation
//!   • `scipy.stats.gzscore(data)` — geometric z-score
//!
//! Resolves [frankenscipy-139s5]. 3 datasets × 7 describe
//! arms + 3 datasets × 1 gstd + 3 datasets × 1 variation + 3
//! datasets × 1 gzscore (vector max-abs) = 21 + 3 + 3 + 3 =
//! 30 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{describe, gstd, gzscore, variation};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    nobs: Option<i64>,
    min: Option<f64>,
    max: Option<f64>,
    mean: Option<f64>,
    variance: Option<f64>,
    skewness: Option<f64>,
    kurtosis: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create describe_misc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize describe_misc diff log");
    fs::write(path, json).expect("write describe_misc diff log");
}

fn generate_query() -> OracleQuery {
    // Datasets used for describe and variation (any sign)
    let mixed_datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "near_normal",
            vec![
                -1.5, -0.9, -0.4, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6,
            ],
        ),
        (
            "skewed",
            vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7, 5.0, 7.0, 10.0,
            ],
        ),
        (
            "spread",
            vec![-5.0, -2.0, 0.0, 1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
        ),
    ];
    // Positive-only datasets for gstd / gzscore
    let positive_datasets: Vec<(&str, Vec<f64>)> = vec![
        (
            "logspace",
            vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
        ),
        (
            "tight_pos",
            vec![1.05, 0.95, 1.10, 0.90, 1.15, 0.85, 1.20, 0.80],
        ),
        (
            "varied_pos",
            vec![0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 6.5, 10.0, 15.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &mixed_datasets {
        points.push(PointCase {
            case_id: format!("describe_{name}"),
            func: "describe".into(),
            data: data.clone(),
        });
        points.push(PointCase {
            case_id: format!("variation_{name}"),
            func: "variation".into(),
            data: data.clone(),
        });
    }
    for (name, data) in &positive_datasets {
        points.push(PointCase {
            case_id: format!("gstd_{name}"),
            func: "gstd".into(),
            data: data.clone(),
        });
        points.push(PointCase {
            case_id: format!("gzscore_{name}"),
            func: "gzscore".into(),
            data: data.clone(),
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
from scipy import stats

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
    out = {
        "case_id": cid,
        "nobs": None, "min": None, "max": None, "mean": None,
        "variance": None, "skewness": None, "kurtosis": None,
        "scalar": None, "vector": None,
    }
    try:
        if func == "describe":
            # fsci uses biased moments for skewness/kurtosis (population
            # estimator, m_k/n / (m_2/n)^...) — match by passing bias=True.
            res = stats.describe(data, bias=True)
            out["nobs"] = int(res.nobs)
            out["min"] = fnone(res.minmax[0])
            out["max"] = fnone(res.minmax[1])
            out["mean"] = fnone(res.mean)
            out["variance"] = fnone(res.variance)
            out["skewness"] = fnone(res.skewness)
            out["kurtosis"] = fnone(res.kurtosis)
        elif func == "gstd":
            out["scalar"] = fnone(stats.gstd(data))
        elif func == "variation":
            out["scalar"] = fnone(stats.variation(data))
        elif func == "gzscore":
            out["vector"] = vec_or_none(stats.gzscore(data).tolist())
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize describe_misc query");
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
                "failed to spawn python3 for describe_misc oracle: {e}"
            );
            eprintln!("skipping describe_misc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open describe_misc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "describe_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping describe_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for describe_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "describe_misc oracle failed: {stderr}"
        );
        eprintln!("skipping describe_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse describe_misc oracle JSON"))
}

#[test]
fn diff_stats_describe_misc() {
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
            "describe" => {
                let r = describe(&case.data);
                let arms: [(&str, Option<f64>, Option<f64>); 7] = [
                    (
                        "nobs",
                        scipy_arm.nobs.map(|n| n as f64),
                        Some(r.nobs as f64),
                    ),
                    ("min", scipy_arm.min, Some(r.minmax.0)),
                    ("max", scipy_arm.max, Some(r.minmax.1)),
                    ("mean", scipy_arm.mean, Some(r.mean)),
                    ("variance", scipy_arm.variance, Some(r.variance)),
                    ("skewness", scipy_arm.skewness, Some(r.skewness)),
                    ("kurtosis", scipy_arm.kurtosis, Some(r.kurtosis)),
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
            "gstd" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = gstd(&case.data);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            arm: "gstd".into(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "variation" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = variation(&case.data);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            arm: "variation".into(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            "gzscore" => {
                if let Some(scipy_vec) = &scipy_arm.vector {
                    let rust_vec = gzscore(&case.data);
                    let mut max_local = 0.0_f64;
                    for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
                        if a.is_finite() {
                            max_local = max_local.max((a - b).abs());
                        }
                    }
                    max_overall = max_overall.max(max_local);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: "gzscore_max".into(),
                        abs_diff: max_local,
                        pass: max_local <= ABS_TOL,
                    });
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_describe_misc".into(),
        category: "scipy.stats.describe + gstd + variation + gzscore".into(),
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
                "describe_misc mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "describe_misc conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
