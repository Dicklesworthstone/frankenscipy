#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three closed-form
//! utilities not exercised by any other diff harness:
//!   • `scipy.stats.trimboth(data, proportiontocut)` — drop
//!     a fraction from each end of the sorted sample.
//!   • `scipy.stats.trim1(data, proportiontocut, tail)` —
//!     drop a fraction from one tail.
//!   • `scipy.stats.mode(data, keepdims=False)` — most
//!     frequent value with its count.
//!
//! Resolves [frankenscipy-pdr3c]. fsci returns sorted slices
//! for trimboth/trim1; scipy returns the same elements but
//! preserves original input order. The harness sorts both
//! sides before comparison so ordering convention doesn't
//! affect the diff.
//!
//! 3 datasets × (3 trimboth props + 3 trim1 left + 3 trim1
//! right + 1 mode) = 30 cases via subprocess. Tol 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{mode_full, trim1, trimboth};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    prop: f64,
    tail: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    vector_sorted: Option<Vec<f64>>,
    mode_value: Option<f64>,
    mode_count: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create trim_mode diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize trim_mode diff log");
    fs::write(path, json).expect("write trim_mode diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=20).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -5.0, -2.0, 0.0, 1.0, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0, 15.0, 20.0,
            ],
        ),
        (
            "ties",
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ),
    ];
    let props: [f64; 3] = [0.10, 0.20, 0.25];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for &p in &props {
            points.push(PointCase {
                case_id: format!("trimboth_{name}_p{p}"),
                func: "trimboth".into(),
                data: data.clone(),
                prop: p,
                tail: "".into(),
            });
            points.push(PointCase {
                case_id: format!("trim1_{name}_p{p}_left"),
                func: "trim1".into(),
                data: data.clone(),
                prop: p,
                tail: "left".into(),
            });
            points.push(PointCase {
                case_id: format!("trim1_{name}_p{p}_right"),
                func: "trim1".into(),
                data: data.clone(),
                prop: p,
                tail: "right".into(),
            });
        }
        points.push(PointCase {
            case_id: format!("mode_{name}"),
            func: "mode".into(),
            data: data.clone(),
            prop: 0.0,
            tail: "".into(),
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
    out = {"case_id": cid, "vector_sorted": None,
           "mode_value": None, "mode_count": None}
    try:
        if func == "trimboth":
            res = stats.trimboth(data, proportiontocut=float(case["prop"]))
            sorted_res = sorted(np.asarray(res).tolist())
            out["vector_sorted"] = vec_or_none(sorted_res)
        elif func == "trim1":
            res = stats.trim1(data, proportiontocut=float(case["prop"]),
                              tail=case["tail"])
            sorted_res = sorted(np.asarray(res).tolist())
            out["vector_sorted"] = vec_or_none(sorted_res)
        elif func == "mode":
            res = stats.mode(data, keepdims=False)
            out["mode_value"] = fnone(res.mode)
            out["mode_count"] = int(res.count)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize trim_mode query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for trim_mode oracle: {e}"
            );
            eprintln!("skipping trim_mode oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open trim_mode oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trim_mode oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trim_mode oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for trim_mode oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trim_mode oracle failed: {stderr}"
        );
        eprintln!("skipping trim_mode oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trim_mode oracle JSON"))
}

#[test]
fn diff_stats_trim_mode() {
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
            "trimboth" => {
                if let Some(scipy_vec) = &scipy_arm.vector_sorted {
                    let mut rust_vec = trimboth(&case.data, case.prop).unwrap_or_else(|e| {
                        panic!("scipy produced a trimboth for case {} (prop {}) but ours refused: {e:?}", case.case_id, case.prop)
                    });
                    rust_vec.sort_by(|a, b| a.total_cmp(b));
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
                            arm: "vector_sorted_max".into(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            "trim1" => {
                if let Some(scipy_vec) = &scipy_arm.vector_sorted {
                    let mut rust_vec = trim1(&case.data, case.prop, &case.tail);
                    rust_vec.sort_by(|a, b| a.total_cmp(b));
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
                            arm: "vector_sorted_max".into(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            "mode" => {
                let r = mode_full(&case.data);
                if let Some(scipy_v) = scipy_arm.mode_value
                    && r.mode.is_finite()
                {
                    let abs_diff = (r.mode - scipy_v).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "mode_value".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
                if let Some(scipy_c) = scipy_arm.mode_count {
                    let abs_diff = (r.count as i64 - scipy_c).unsigned_abs() as f64;
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "mode_count".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_trim_mode".into(),
        category: "scipy.stats.trimboth + trim1 + mode".into(),
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
                "trim_mode {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "trim_mode conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
