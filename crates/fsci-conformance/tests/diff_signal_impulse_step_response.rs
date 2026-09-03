#![forbid(unsafe_code)]
//! Live scipy.signal-derived parity for fsci_signal::{impulse_response,
//! step_response}.
//!
//! Resolves [frankenscipy-0cj1i]. impulse_response is the filter output
//! for δ[n]; step_response for u[n]. Both can be computed via numpy
//! lfilter (scipy.signal.lfilter) directly without needing dimpulse/dstep.
//!
//! Tolerance: 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{impulse_response, step_response};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "impulse" | "step"
    b: Vec<f64>,
    a: Vec<f64>,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create impulse_step diff dir");
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
    // Includes filters with len(b) < len(a): impulse/step_response now
    // apply b directly (no leading-zero delay) — frankenscipy-xw68h.
    let filters: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "moving_avg_3",
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0, 0.0, 0.0],
        ),
        ("biquad_lp", vec![0.04, 0.08, 0.04], vec![1.0, -1.4, 0.6]),
        (
            "fir_5tap",
            vec![0.1, 0.2, 0.4, 0.2, 0.1],
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("one_pole", vec![0.2], vec![1.0, -0.8]),
        ("two_pole_scalar_num", vec![0.5], vec![1.0, -0.3, 0.1]),
        ("first_order_lp", vec![0.1, 0.1], vec![1.0, -0.8]),
    ];
    let mut points = Vec::new();
    for (label, b, a) in &filters {
        for op in ["impulse", "step"] {
            for &n in &[16_usize, 32, 64] {
                points.push(Case {
                    case_id: format!("{op}_{label}_n{n}"),
                    op: op.into(),
                    b: b.clone(),
                    a: a.clone(),
                    n,
                });
            }
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
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    n = int(case["n"])
    try:
        if op == "impulse":
            x = np.zeros(n); x[0] = 1.0
        elif op == "step":
            x = np.ones(n)
        else:
            x = None
        if x is None:
            points.append({"case_id": cid, "values": None})
            continue
        y = signal.lfilter(b, a, x)
        flat = [float(v) for v in y.tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "values": flat})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for impulse_step oracle: {e}"
            );
            eprintln!("skipping impulse_step oracle: python3 not available ({e})");
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
                "impulse_step oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping impulse_step oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for impulse_step oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "impulse_step oracle failed: {stderr}"
        );
        eprintln!("skipping impulse_step oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse impulse_step oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_impulse_step_response() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let res = match case.op.as_str() {
            "impulse" => impulse_response(&case.b, &case.a, case.n),
            "step" => step_response(&case.b, &case.a, case.n),
            _ => continue,
        };
        let Ok(y) = res else { continue };
        let abs_d = vec_max_diff(&y, expected);
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
        test_id: "diff_signal_impulse_step_response".into(),
        category: "fsci_signal::impulse_response + step_response vs scipy.signal.lfilter".into(),
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
        "impulse_step conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
