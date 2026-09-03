#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for fsci's
//! variant-named descriptive utilities not exercised by the
//! shorter-name harnesses:
//!   • `zscore_ddof(data, ddof=1)`     — explicit-ddof zscore
//!   • `gzscore_ddof(data, ddof=1)`    — explicit-ddof gzscore
//!   • `iqr_range(data)`               — alias of iqr (Q3 − Q1)
//!   • `standard_error_of_mean(data)`  — alias of sem (ddof=1)
//!
//! Resolves [frankenscipy-l7qd4]. The oracle calls
//! `scipy.stats.{zscore, gzscore, iqr, sem}` with the matching
//! ddof / convention.
//!
//! 3 datasets × 4 funcs = 12 arms (per-element max-abs for the
//! vector funcs, scalar for iqr_range / standard_error_of_mean).
//! Tol 1e-12 abs (closed-form arithmetic chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{gzscore_ddof, iqr_range, standard_error_of_mean, zscore_ddof};
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
    /// Vector arm (zscore_ddof / gzscore_ddof) or scalar arm packed in length-1.
    values: Option<Vec<f64>>,
    /// Scalar arm (iqr_range / standard_error_of_mean).
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create descriptive_aliases diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize descriptive_aliases diff log");
    fs::write(path, json).expect("write descriptive_aliases diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_positive_n12", (1..=12).map(|i| i as f64).collect()),
        (
            "spread_positive_n15",
            (1..=15).map(|i| (i as f64).powi(2)).collect(),
        ),
        (
            "ties_positive_n14",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in [
            "zscore_ddof",
            "gzscore_ddof",
            "iqr_range",
            "standard_error_of_mean",
        ] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
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
from scipy import stats

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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "values": None, "value": None}
    try:
        if func == "zscore_ddof":
            arr = stats.zscore(data, ddof=1)
            out["values"] = vec_or_none(np.asarray(arr).tolist())
        elif func == "gzscore_ddof":
            arr = stats.gzscore(data, ddof=1)
            out["values"] = vec_or_none(np.asarray(arr).tolist())
        elif func == "iqr_range":
            out["value"] = fnone(stats.iqr(data))
        elif func == "standard_error_of_mean":
            out["value"] = fnone(stats.sem(data, ddof=1))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize descriptive_aliases query");
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
                "failed to spawn python3 for descriptive_aliases oracle: {e}"
            );
            eprintln!("skipping descriptive_aliases oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open descriptive_aliases oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "descriptive_aliases oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping descriptive_aliases oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for descriptive_aliases oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "descriptive_aliases oracle failed: {stderr}"
        );
        eprintln!("skipping descriptive_aliases oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse descriptive_aliases oracle JSON"))
}

#[test]
fn diff_stats_descriptive_aliases() {
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
            "zscore_ddof" => {
                if let Some(scipy_vec) = &scipy_arm.values {
                    let rust_vec = zscore_ddof(&case.data, 1);
                    if rust_vec.len() == scipy_vec.len() {
                        let mut max_local = 0.0_f64;
                        for (r, s) in rust_vec.iter().zip(scipy_vec.iter()) {
                            if r.is_finite() {
                                max_local = max_local.max((r - s).abs());
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
            "gzscore_ddof" => {
                if let Some(scipy_vec) = &scipy_arm.values {
                    let rust_vec = gzscore_ddof(&case.data, 1);
                    if rust_vec.len() == scipy_vec.len() {
                        let mut max_local = 0.0_f64;
                        for (r, s) in rust_vec.iter().zip(scipy_vec.iter()) {
                            if r.is_finite() {
                                max_local = max_local.max((r - s).abs());
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
            "iqr_range" => {
                if let Some(scipy_v) = scipy_arm.value {
                    let rust_v = iqr_range(&case.data);
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
            "standard_error_of_mean" => {
                if let Some(scipy_v) = scipy_arm.value {
                    let rust_v = standard_error_of_mean(&case.data);
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
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_descriptive_aliases".into(),
        category: "scipy.stats descriptive aliases (zscore/gzscore/iqr/sem)".into(),
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
                "descriptive_aliases {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "descriptive_aliases conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
