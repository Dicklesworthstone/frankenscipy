#![forbid(unsafe_code)]
//! Live numpy parity for fsci_ndimage array statistics:
//! array_sum, array_mean, array_variance, array_std, array_max,
//! array_min.
//!
//! Resolves [frankenscipy-90er3]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    NdArray, array_max, array_mean, array_min, array_std, array_sum, array_variance,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create array_stats diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let v1d: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let v2d: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5 - 2.0).collect();
    let v3d: Vec<f64> = (0..24).map(|i| ((i as f64) * 0.3).sin() + 0.5).collect();
    let v_neg: Vec<f64> = vec![-3.5, -1.0, 0.0, 1.5, 7.0, -0.25];

    let fixtures: Vec<(&str, Vec<f64>, Vec<usize>)> = vec![
        ("seq10_1d", v1d, vec![10]),
        ("range_2d_4x4", v2d, vec![4, 4]),
        ("sin_3d_2x3x4", v3d, vec![2, 3, 4]),
        ("mixed_neg_1d", v_neg, vec![6]),
    ];

    let mut points = Vec::new();
    for (label, data, shape) in &fixtures {
        for op in ["sum", "mean", "variance", "std", "max", "min"] {
            points.push(Case {
                case_id: format!("{label}_{op}"),
                op: op.into(),
                shape: shape.clone(),
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    shape = tuple(int(s) for s in case["shape"])
    a = np.array(case["data"], dtype=float).reshape(shape)
    try:
        if op == "sum":       v = float(np.sum(a))
        elif op == "mean":    v = float(np.mean(a))
        elif op == "variance": v = float(np.var(a))   # ddof=0 (population) matches fsci
        elif op == "std":     v = float(np.std(a))    # ddof=0
        elif op == "max":     v = float(np.max(a))
        elif op == "min":     v = float(np.min(a))
        else:                  v = float("nan")
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for array_stats oracle: {e}"
            );
            eprintln!("skipping array_stats oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "array_stats oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping array_stats oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for array_stats oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "array_stats oracle failed: {stderr}"
        );
        eprintln!("skipping array_stats oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse array_stats oracle JSON"))
}

#[test]
fn diff_ndimage_array_stats() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let Ok(arr) = NdArray::new(case.data.clone(), case.shape.clone()) else {
            continue;
        };
        let actual = match case.op.as_str() {
            "sum" => array_sum(&arr),
            "mean" => array_mean(&arr),
            "variance" => array_variance(&arr),
            "std" => array_std(&arr),
            "max" => array_max(&arr),
            "min" => array_min(&arr),
            _ => continue,
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_array_stats".into(),
        category: "fsci_ndimage array_sum/mean/variance/std/max/min vs numpy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "array_stats conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
