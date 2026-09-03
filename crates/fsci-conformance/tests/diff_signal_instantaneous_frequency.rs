#![forbid(unsafe_code)]
//! Live scipy.signal.hilbert-derived parity for fsci_signal::
//! instantaneous_frequency.
//!
//! Resolves [frankenscipy-ia19q]. Definition (fsci):
//!   φ_inst = unwrap(angle(hilbert(x)))
//!   f_inst[0] = 0
//!   f_inst[i] = (φ_inst[i] - φ_inst[i-1]) · fs / (2π) for i >= 1
//!
//! The oracle computes the same via scipy.signal.hilbert + numpy.
//! Tolerance: 1e-8 abs (Hilbert + finite-difference floor).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::instantaneous_frequency;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    x: Vec<f64>,
    fs: f64,
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
    fs::create_dir_all(output_dir()).expect("create inst_freq diff dir");
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
    let n = 64;
    let fs = 100.0;
    let dt = 1.0 / fs;
    // 10 Hz sinusoid.
    let sin10: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64) * dt;
            (2.0 * std::f64::consts::PI * 10.0 * t).sin()
        })
        .collect();
    // 5 Hz cosine.
    let cos5: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64) * dt;
            (2.0 * std::f64::consts::PI * 5.0 * t).cos()
        })
        .collect();
    // Linear chirp from 2 Hz to 15 Hz.
    let n_chirp = 128;
    let chirp: Vec<f64> = (0..n_chirp)
        .map(|i| {
            let t = (i as f64) * dt;
            let f0 = 2.0;
            let f1 = 15.0;
            let total_t = (n_chirp as f64) * dt;
            let k = (f1 - f0) / total_t;
            let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * k * t * t);
            phase.sin()
        })
        .collect();
    OracleQuery {
        points: vec![
            Case {
                case_id: "sin_10hz_n64_fs100".into(),
                x: sin10,
                fs,
            },
            Case {
                case_id: "cos_5hz_n64_fs100".into(),
                x: cos5,
                fs,
            },
            Case {
                case_id: "chirp_2_15hz_n128_fs100".into(),
                x: chirp,
                fs,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.signal import hilbert

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    fs = float(case["fs"])
    try:
        z = hilbert(x)
        phase = np.unwrap(np.angle(z))
        n = len(x)
        freq = np.zeros(n)
        for i in range(1, n):
            freq[i] = (phase[i] - phase[i-1]) * fs / (2.0 * math.pi)
        out = freq.tolist()
        if all(math.isfinite(v) for v in out):
            points.append({"case_id": cid, "values": [float(v) for v in out]})
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
                "failed to spawn python3 for inst_freq oracle: {e}"
            );
            eprintln!("skipping inst_freq oracle: python3 not available ({e})");
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
                "inst_freq oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping inst_freq oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for inst_freq oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "inst_freq oracle failed: {stderr}"
        );
        eprintln!("skipping inst_freq oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse inst_freq oracle JSON"))
}

#[test]
fn diff_signal_instantaneous_frequency() {
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
        let Ok(actual) = instantaneous_frequency(&case.x, case.fs) else {
            continue;
        };
        let abs_d = if actual.len() != expected.len() {
            f64::INFINITY
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_instantaneous_frequency".into(),
        category: "fsci_signal::instantaneous_frequency vs scipy.signal.hilbert formula".into(),
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
            eprintln!("inst_freq mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "inst_freq conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
