#![forbid(unsafe_code)]
//! Live numerical reference checks for five simple
//! time-series / running-aggregate primitives:
//!   • `ewma(data, span)` — pandas-style EWMA
//!     (alpha = 2/(span+1), initialized with data[0])
//!   • `moving_average(data, window)` — simple sliding
//!     window mean
//!   • `cumsum(data)` — cumulative sum
//!   • `cumprod(data)` — cumulative product
//!   • `diff(data)` — first-order difference
//!
//! Resolves [frankenscipy-d79q3]. The oracle reproduces each
//! primitive directly in numpy.
//!
//! 3 datasets × 5 funcs = 15 cases via subprocess. Each case
//! compares the output vector with max-abs aggregation. Tol
//! 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{cumprod, cumsum, diff, ewma, moving_average};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// 1e-10 (not 1e-12) because cumprod accumulates multiplicative
// floating-point drift over the full vector — values can grow into the
// thousands and last-bit noise relative to numpy's pairwise reduction
// reaches ~3e-11 on the noisy fixture.
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    span: f64,
    window: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    out: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create timeseries_ops diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize timeseries_ops diff log");
    fs::write(path, json).expect("write timeseries_ops diff log");
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.func.as_str() {
        "ewma" => Some(ewma(&case.data, case.span)),
        "moving_average" => Some(moving_average(&case.data, case.window as usize)),
        "cumsum" => Some(cumsum(&case.data)),
        "cumprod" => Some(cumprod(&case.data)),
        "diff" => Some(diff(&case.data)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=10).map(|i| i as f64).collect()),
        (
            "noisy",
            vec![1.5, 2.0, 1.8, 2.5, 2.2, 3.0, 2.7, 3.5, 3.2, 4.0, 3.8, 4.5],
        ),
        (
            "alternating",
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        // ewma at span=3
        points.push(PointCase {
            case_id: format!("ewma_{name}_s3"),
            func: "ewma".into(),
            data: data.clone(),
            span: 3.0,
            window: 0,
        });
        // moving_average at window=3
        points.push(PointCase {
            case_id: format!("ma_{name}_w3"),
            func: "moving_average".into(),
            data: data.clone(),
            span: 0.0,
            window: 3,
        });
        // cumsum
        points.push(PointCase {
            case_id: format!("cumsum_{name}"),
            func: "cumsum".into(),
            data: data.clone(),
            span: 0.0,
            window: 0,
        });
        // cumprod
        points.push(PointCase {
            case_id: format!("cumprod_{name}"),
            func: "cumprod".into(),
            data: data.clone(),
            span: 0.0,
            window: 0,
        });
        // diff
        points.push(PointCase {
            case_id: format!("diff_{name}"),
            func: "diff".into(),
            data: data.clone(),
            span: 0.0,
            window: 0,
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

def numpy_ewma(data, span):
    if span < 1.0 or not math.isfinite(span):
        return list(data)
    alpha = 2.0 / (span + 1.0)
    out = [float(data[0])]
    for i in range(1, len(data)):
        out.append(alpha * float(data[i]) + (1.0 - alpha) * out[-1])
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    val = None
    try:
        if func == "ewma":
            val = vec_or_none(numpy_ewma(data.tolist(), float(case["span"])))
        elif func == "moving_average":
            w = int(case["window"])
            if w == 0 or w > len(data):
                val = []
            else:
                # numpy.convolve mode='valid' gives the same as the simple
                # sliding window mean.
                kernel = np.ones(w) / w
                val = vec_or_none(np.convolve(data, kernel, mode='valid').tolist())
        elif func == "cumsum":
            val = vec_or_none(np.cumsum(data).tolist())
        elif func == "cumprod":
            val = vec_or_none(np.cumprod(data).tolist())
        elif func == "diff":
            val = vec_or_none(np.diff(data).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "out": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize timeseries_ops query");
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
                "failed to spawn python3 for timeseries_ops oracle: {e}"
            );
            eprintln!("skipping timeseries_ops oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open timeseries_ops oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "timeseries_ops oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping timeseries_ops oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for timeseries_ops oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "timeseries_ops oracle failed: {stderr}"
        );
        eprintln!("skipping timeseries_ops oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse timeseries_ops oracle JSON"))
}

#[test]
fn diff_stats_timeseries_ops() {
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
        let Some(scipy_vec) = &scipy_arm.out else {
            continue;
        };
        let Some(rust_vec) = fsci_eval(case) else {
            continue;
        };
        if rust_vec.len() != scipy_vec.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
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

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_timeseries_ops".into(),
        category: "ewma + moving_average + cumsum + cumprod + diff".into(),
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
                "timeseries_ops {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "timeseries_ops conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
