#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two scipy.signal peak
//! detection primitives that fsci-signal exposes but that had no
//! dedicated conformance harness:
//!   - `scipy.signal.find_peaks(x)` (default options)
//!   - `scipy.signal.argrelextrema(x, np.greater | np.less)`
//!
//! Resolves [frankenscipy-z4an6]. find_peaks was previously covered
//! only via e2e_signal scenarios; argrelextrema had no test at all.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{FindPeaksOptions, argrelextrema, find_peaks};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
    order: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    indices: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    pass: bool,
    fsci: Vec<usize>,
    scipy: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create find_peaks diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_peaks diff log");
    fs::write(path, json).expect("write find_peaks diff log");
}

fn generate_query() -> OracleQuery {
    // Signal shapes covering: simple oscillation, alternating peaks,
    // monotonic rise (no peaks), short signal, plateau-style peaks.
    let signals: &[(&str, Vec<f64>)] = &[
        (
            "oscillation",
            vec![0.0, 1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 3.5, 0.0],
        ),
        (
            "alternating",
            vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0, 0.0],
        ),
        ("monotonic_rise", vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("monotonic_fall", vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        ("plateau", vec![0.0, 1.0, 3.0, 3.0, 3.0, 1.0, 0.0, 2.0, 0.0]),
        ("trivial_short", vec![1.0, 2.0, 1.0]),
        (
            "noisy_sine_n11",
            // Discrete sample of sin(x) at evenly spaced x.
            (0..11)
                .map(|i| (i as f64 * std::f64::consts::PI / 5.0).sin())
                .collect(),
        ),
    ];
    let mut points = Vec::new();
    for (label, sig) in signals {
        for func in ["find_peaks", "argrelextrema_max", "argrelextrema_min"] {
            points.push(PointCase {
                case_id: format!("{func}_{label}"),
                func: func.into(),
                x: sig.clone(),
                order: 1,
            });
        }
    }
    // order >= 2 coverage for argrelextrema: extrema within `order` of an
    // edge must still be detected (scipy mode='clip'; frankenscipy-mufw9).
    for (label, sig) in signals {
        for func in ["argrelextrema_max", "argrelextrema_min"] {
            for order in [2_usize, 3] {
                points.push(PointCase {
                    case_id: format!("{func}_{label}_o{order}"),
                    func: func.into(),
                    x: sig.clone(),
                    order,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    x = np.array(case["x"], dtype=float)
    order = int(case["order"])
    try:
        if func == "find_peaks":
            peaks, _ = signal.find_peaks(x)
            points.append({"case_id": cid, "indices": [int(p) for p in peaks]})
        elif func == "argrelextrema_max":
            idx = signal.argrelextrema(x, np.greater, order=order)
            points.append({"case_id": cid, "indices": [int(p) for p in idx[0]]})
        elif func == "argrelextrema_min":
            idx = signal.argrelextrema(x, np.less, order=order)
            points.append({"case_id": cid, "indices": [int(p) for p in idx[0]]})
        else:
            points.append({"case_id": cid, "indices": None})
    except Exception:
        points.append({"case_id": cid, "indices": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize find_peaks query");
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
                "failed to spawn python3 for find_peaks oracle: {e}"
            );
            eprintln!("skipping find_peaks oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open find_peaks oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_peaks oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_peaks oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_peaks oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_peaks oracle failed: {stderr}"
        );
        eprintln!("skipping find_peaks oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_peaks oracle JSON"))
}

fn fsci_eval(func: &str, x: &[f64], order: usize) -> Vec<usize> {
    match func {
        "find_peaks" => find_peaks(x, FindPeaksOptions::default()).peaks,
        "argrelextrema_max" => argrelextrema(x, order, true),
        "argrelextrema_min" => argrelextrema(x, order, false),
        _ => Vec::new(),
    }
}

#[test]
fn diff_signal_find_peaks_argrelextrema() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_idx) = scipy_arm.indices.as_ref() else {
            continue;
        };
        let fsci_idx = fsci_eval(&case.func, &case.x, case.order);
        let pass = fsci_idx == *scipy_idx;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            pass,
            fsci: fsci_idx,
            scipy: scipy_idx.clone(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_find_peaks_argrelextrema".into(),
        category: "scipy.signal.find_peaks / argrelextrema".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "find_peaks/argrelextrema {} mismatch: case={} fsci={:?} scipy={:?}",
                d.func, d.case_id, d.fsci, d.scipy
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal find_peaks/argrelextrema conformance failed: {} cases",
        diffs.len()
    );
}
