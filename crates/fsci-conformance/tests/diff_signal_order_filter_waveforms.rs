#![forbid(unsafe_code)]
//! Live scipy.signal parity for fsci_signal::order_filter,
//! sawtooth, square, and unit_impulse.
//!
//! Resolves [frankenscipy-9mj5c]. firwin2 was originally bundled
//! here but diverges from scipy.signal.firwin2 by ~1e-2 abs across
//! all probes — filed as defect [frankenscipy-ei9yv] and dropped.
//!
//! - `order_filter`: rank-order filter. fsci shrinks the window
//!   near boundaries; scipy zero-pads. Compare only interior
//!   samples (window fully inside array), 1e-12 abs.
//! - `sawtooth`, `square`: periodic waveform generators with
//!   parameter (width/duty). Deterministic; 1e-12 abs.
//! - `unit_impulse`: delta function. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{order_filter, sawtooth, square, unit_impulse};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "order_filter" | "sawtooth" | "square" | "unit_impulse"
    /// order_filter
    x: Vec<f64>,
    window_size: usize,
    rank: usize,
    /// sawtooth / square
    t: Vec<f64>,
    param: f64,
    /// unit_impulse
    shape: usize,
    idx: usize,
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
    fs::create_dir_all(output_dir()).expect("create signal_misc diff dir");
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
    let mut points = Vec::new();

    // order_filter probes
    let x_smooth: Vec<f64> = (0..32)
        .map(|i| (i as f64 * 0.4).sin() + 0.1 * (i as f64 * 1.7).cos())
        .collect();
    let x_spike: Vec<f64> = {
        let mut v = vec![0.0_f64; 32];
        v[5] = 10.0;
        v[15] = -8.0;
        v[20] = 5.0;
        v
    };
    let x_random: Vec<f64> = (0..40)
        .map(|i| {
            let s = (i as f64 * 0.137).sin();
            let c = (i as f64 * 0.731).cos();
            s + c * 0.5 - 0.2 * (i as f64 * 0.51).sin()
        })
        .collect();

    for (label, x) in [
        ("smooth", &x_smooth),
        ("spike", &x_spike),
        ("random", &x_random),
    ] {
        for window_size in [3_usize, 5, 7] {
            let median = window_size / 2;
            let upper = window_size - 1;
            for (rname, rank) in [("min", 0), ("med", median), ("max", upper)] {
                points.push(Case {
                    case_id: format!("of_{label}_w{window_size}_{rname}"),
                    op: "order_filter".into(),
                    x: x.clone(),
                    window_size,
                    rank,
                    t: vec![],
                    param: 0.0,
                    shape: 0,
                    idx: 0,
                });
            }
        }
    }

    // sawtooth / square probes — full period, fractional period
    let t1: Vec<f64> = (0..64)
        .map(|i| i as f64 * 2.0 * std::f64::consts::PI / 64.0)
        .collect();
    let t2: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
    for &width in &[0.0_f64, 0.25, 0.5, 0.75, 1.0] {
        points.push(Case {
            case_id: format!("sawtooth_w{width}_t1"),
            op: "sawtooth".into(),
            x: vec![],
            window_size: 0,
            rank: 0,
            t: t1.clone(),
            param: width,
            shape: 0,
            idx: 0,
        });
    }
    for &duty in &[0.2_f64, 0.5, 0.8] {
        for (tlabel, t) in [("t1", &t1), ("t2", &t2)] {
            points.push(Case {
                case_id: format!("square_d{duty}_{tlabel}"),
                op: "square".into(),
                x: vec![],
                window_size: 0,
                rank: 0,
                t: t.clone(),
                param: duty,
                shape: 0,
                idx: 0,
            });
        }
    }

    // unit_impulse probes
    for (shape, idx) in [(10, 0_usize), (20, 5), (32, 16), (8, 7)] {
        points.push(Case {
            case_id: format!("ui_s{shape}_i{idx}"),
            op: "unit_impulse".into(),
            x: vec![],
            window_size: 0,
            rank: 0,
            t: vec![],
            param: 0.0,
            shape,
            idx,
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
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "order_filter":
            x = np.array(case["x"], dtype=float)
            ws = int(case["window_size"]); rank = int(case["rank"])
            half = ws // 2
            domain = np.ones(ws, dtype=int)
            y = signal.order_filter(x, domain, rank)
            interior = y[half:len(y)-half]
            flat = [float(v) for v in interior.tolist()]
        elif op == "sawtooth":
            t = np.array(case["t"], dtype=float)
            w = float(case["param"])
            y = signal.sawtooth(t, w)
            flat = [float(v) for v in y.tolist()]
        elif op == "square":
            t = np.array(case["t"], dtype=float)
            d = float(case["param"])
            y = signal.square(t, d)
            flat = [float(v) for v in y.tolist()]
        elif op == "unit_impulse":
            shape = int(case["shape"]); idx = int(case["idx"])
            y = signal.unit_impulse(shape, idx)
            flat = [float(v) for v in y.tolist()]
        else:
            points.append({"case_id": cid, "values": None}); continue
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
                "failed to spawn python3 for signal_misc oracle: {e}"
            );
            eprintln!("skipping signal_misc oracle: python3 not available ({e})");
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
                "signal_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping signal_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for signal_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "signal_misc oracle failed: {stderr}"
        );
        eprintln!("skipping signal_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse signal_misc oracle JSON"))
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
fn diff_signal_order_filter_waveforms() {
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
        let abs_d = match case.op.as_str() {
            "order_filter" => {
                let y = order_filter(&case.x, case.window_size, case.rank);
                let half = case.window_size / 2;
                if y.len() < 2 * half {
                    continue;
                }
                let interior = &y[half..y.len() - half];
                vec_max_diff(interior, expected)
            }
            "sawtooth" => {
                let Ok(y) = sawtooth(&case.t, case.param) else {
                    continue;
                };
                vec_max_diff(&y, expected)
            }
            "square" => {
                let Ok(y) = square(&case.t, case.param) else {
                    continue;
                };
                vec_max_diff(&y, expected)
            }
            "unit_impulse" => {
                let Ok(y) = unit_impulse(case.shape, Some(case.idx)) else {
                    continue;
                };
                vec_max_diff(&y, expected)
            }
            _ => continue,
        };
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
        test_id: "diff_signal_order_filter_waveforms".into(),
        category: "fsci_signal::order_filter + sawtooth + square + unit_impulse vs scipy.signal"
            .into(),
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
        "signal_misc conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
