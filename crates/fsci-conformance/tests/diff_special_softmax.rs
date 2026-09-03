#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the vector-input
//! log-sum-exp family in `scipy.special`:
//!   - `scipy.special.softmax(x)`     → exp(x − logsumexp(x))
//!   - `scipy.special.log_softmax(x)` → x − logsumexp(x)
//!   - `scipy.special.logsumexp(x)`   → ln Σ exp(x_i)
//!
//! Resolves [frankenscipy-yoo58]. fsci_special::softmax /
//! log_softmax / logsumexp are exposed via `convenience.rs` and
//! are exercised by lib tests, but had no dedicated diff_special_*
//! harness; spot checks matched scipy to ~1e-15.
//!
//! 7 input arrays × 3 functions = case sweep at 1e-12 abs tol.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{log_softmax, logsumexp, softmax};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Vector result for softmax / log_softmax.
    vector: Option<Vec<f64>>,
    /// Scalar result for logsumexp.
    scalar: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create softmax diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize softmax diff log");
    fs::write(path, json).expect("write softmax diff log");
}

fn generate_query() -> OracleQuery {
    // Input vectors span:
    // - single element (degenerate)
    // - balanced positive values
    // - balanced symmetric (incl. negatives)
    // - extreme positive (exp-overflow without log-sum-exp's shift)
    // - extreme negative (exp-underflow probe)
    // - mixed regimes
    let inputs: &[(&str, &[f64])] = &[
        ("single", &[5.0]),
        ("balanced_small", &[1.0, 2.0, 3.0, -1.0]),
        ("symmetric", &[-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("large_positive", &[100.0, 99.0, 98.0]),
        ("large_negative", &[-100.0, -101.0, -99.0]),
        ("mixed_extreme", &[1e3, 1.0, -1e3]),
        ("near_zero", &[0.001, 0.002, -0.001]),
    ];
    let funcs = ["softmax", "log_softmax", "logsumexp"];
    let mut points = Vec::new();
    for (label, xs) in inputs {
        for func in funcs {
            points.push(PointCase {
                case_id: format!("{func}_{label}"),
                func: func.to_string(),
                x: xs.to_vec(),
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
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
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
    x = np.array(case["x"], dtype=float)
    try:
        if func == "softmax":
            points.append({"case_id": cid, "vector": finite_vec_or_none(special.softmax(x)), "scalar": None})
        elif func == "log_softmax":
            points.append({"case_id": cid, "vector": finite_vec_or_none(special.log_softmax(x)), "scalar": None})
        elif func == "logsumexp":
            points.append({"case_id": cid, "vector": None, "scalar": finite_or_none(special.logsumexp(x))})
        else:
            points.append({"case_id": cid, "vector": None, "scalar": None})
    except Exception:
        points.append({"case_id": cid, "vector": None, "scalar": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize softmax query");
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
                "failed to spawn python3 for softmax oracle: {e}"
            );
            eprintln!("skipping softmax oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open softmax oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "softmax oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping softmax oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for softmax oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "softmax oracle failed: {stderr}"
        );
        eprintln!("skipping softmax oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse softmax oracle JSON"))
}

#[test]
fn diff_special_softmax() {
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
            "softmax" => {
                let Some(scipy_v) = scipy_arm.vector.as_ref() else {
                    continue;
                };
                let fsci_v = softmax(&case.x);
                if fsci_v.len() != scipy_v.len() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
                let abs_d = fsci_v
                    .iter()
                    .zip(scipy_v.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "log_softmax" => {
                let Some(scipy_v) = scipy_arm.vector.as_ref() else {
                    continue;
                };
                let fsci_v = log_softmax(&case.x);
                if fsci_v.len() != scipy_v.len() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
                let abs_d = fsci_v
                    .iter()
                    .zip(scipy_v.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "logsumexp" => {
                let Some(scipy_v) = scipy_arm.scalar else {
                    continue;
                };
                let fsci_v = logsumexp(&case.x);
                if !fsci_v.is_finite() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    continue;
                }
                let abs_d = (fsci_v - scipy_v).abs();
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_softmax".into(),
        category: "scipy.special.softmax/log_softmax/logsumexp".into(),
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
                "softmax-family {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special softmax/log_softmax/logsumexp conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
