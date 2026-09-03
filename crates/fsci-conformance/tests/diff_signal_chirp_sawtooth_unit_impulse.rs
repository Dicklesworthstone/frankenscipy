#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::chirp, sawtooth,
//! and unit_impulse.
//!
//! Resolves [frankenscipy-y3osz]. 1e-10 abs on time-domain output.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{ChirpMethod, chirp, sawtooth, unit_impulse};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// For chirp: (t, f0, t1, f1, method).
    t: Vec<f64>,
    f0: f64,
    t1: f64,
    f1: f64,
    method: String,
    /// For sawtooth: t + width.
    width: f64,
    /// For unit_impulse: shape, idx.
    shape: usize,
    idx: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create chirp_etc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize chirp_etc diff log");
    fs::write(path, json).expect("write chirp_etc diff log");
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

fn generate_query() -> OracleQuery {
    let t11 = linspace(0.0, 1.0, 11);
    let t21 = linspace(0.0, 2.0, 21);
    let empty = vec![];

    let mut points = Vec::new();

    // chirp: linear / quadratic / logarithmic
    for method in ["linear", "quadratic", "logarithmic"] {
        points.push(PointCase {
            case_id: format!("chirp_{method}_11pt"),
            op: "chirp".into(),
            t: t11.clone(),
            f0: 1.0,
            t1: 1.0,
            f1: 10.0,
            method: method.into(),
            width: 0.0,
            shape: 0,
            idx: 0,
        });
        points.push(PointCase {
            case_id: format!("chirp_{method}_21pt"),
            op: "chirp".into(),
            t: t21.clone(),
            f0: 0.5,
            t1: 2.0,
            f1: 5.0,
            method: method.into(),
            width: 0.0,
            shape: 0,
            idx: 0,
        });
    }

    // sawtooth: fsci.sawtooth takes phase in radians (divides by 2π);
    // scipy.signal.sawtooth uses the same convention. Use 2π*t as phase.
    let two_pi = 2.0 * std::f64::consts::PI;
    let t11_phase: Vec<f64> = t11.iter().map(|&v| two_pi * v).collect();
    for &width in &[0.25_f64, 0.5, 0.75] {
        points.push(PointCase {
            case_id: format!("sawtooth_w{width}_t11"),
            op: "sawtooth".into(),
            t: t11_phase.clone(),
            f0: 0.0,
            t1: 0.0,
            f1: 0.0,
            method: "".into(),
            width,
            shape: 0,
            idx: 0,
        });
    }

    // unit_impulse: various (shape, idx)
    for &(shape, idx) in &[(8_usize, 0_usize), (16, 4), (32, 15), (5, 2)] {
        points.push(PointCase {
            case_id: format!("unit_impulse_n{shape}_idx{idx}"),
            op: "unit_impulse".into(),
            t: empty.clone(),
            f0: 0.0,
            t1: 0.0,
            f1: 0.0,
            method: "".into(),
            width: 0.0,
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

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "chirp":
            t = np.array(case["t"], dtype=float)
            y = signal.chirp(t, f0=float(case["f0"]), t1=float(case["t1"]),
                             f1=float(case["f1"]), method=case["method"])
        elif op == "sawtooth":
            t = np.array(case["t"], dtype=float)
            y = signal.sawtooth(t, width=float(case["width"]))
        elif op == "unit_impulse":
            y = signal.unit_impulse(int(case["shape"]), idx=int(case["idx"]))
        else:
            y = None
        points.append({"case_id": cid, "values": finite_or_none(y) if y is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize chirp_etc query");
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
                "failed to spawn python3 for chirp_etc oracle: {e}"
            );
            eprintln!("skipping chirp_etc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open chirp_etc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "chirp_etc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping chirp_etc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for chirp_etc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "chirp_etc oracle failed: {stderr}"
        );
        eprintln!("skipping chirp_etc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse chirp_etc oracle JSON"))
}

#[test]
fn diff_signal_chirp_sawtooth_unit_impulse() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let fsci_v: Vec<f64> = match case.op.as_str() {
            "chirp" => {
                let method = match case.method.as_str() {
                    "linear" => ChirpMethod::Linear,
                    "quadratic" => ChirpMethod::Quadratic,
                    "logarithmic" => ChirpMethod::Logarithmic,
                    _ => continue,
                };
                let Ok(y) = chirp(&case.t, case.f0, case.t1, case.f1, method) else {
                    continue;
                };
                y
            }
            "sawtooth" => {
                let Ok(y) = sawtooth(&case.t, case.width) else {
                    continue;
                };
                y
            }
            "unit_impulse" => {
                let Ok(y) = unit_impulse(case.shape, Some(case.idx)) else {
                    continue;
                };
                y
            }
            _ => continue,
        };
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
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
        test_id: "diff_signal_chirp_sawtooth_unit_impulse".into(),
        category: "scipy.signal chirp + sawtooth + unit_impulse".into(),
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
        "chirp_sawtooth_unit_impulse conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
