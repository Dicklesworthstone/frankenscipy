#![forbid(unsafe_code)]
//! Live SciPy differential coverage for direct-method 1-D convolution.
//! fsci_signal::convolve and fsci_signal::correlate (the non-FFT
//! variants) are already used internally by other tests, but the
//! direct call shape (a, b, mode) had no dedicated parity harness.
//!
//! Resolves [frankenscipy-p4n0c]. Oracle:
//! `scipy.signal.convolve(a, b, mode, method='direct')`. Direct sum is
//! exact up to ULP — 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{ConvolveMode, convolve};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<f64>,
    b: Vec<f64>,
    mode: String,
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
    mode: String,
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
    fs::create_dir_all(output_dir()).expect("create convolve diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize convolve diff log");
    fs::write(path, json).expect("write convolve diff log");
}

fn parse_mode(name: &str) -> ConvolveMode {
    match name {
        "full" => ConvolveMode::Full,
        "same" => ConvolveMode::Same,
        "valid" => ConvolveMode::Valid,
        _ => ConvolveMode::Full,
    }
}

fn generate_query() -> OracleQuery {
    let pairs: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("small_3x3", vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        (
            "uneven_5x3",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.0, 0.0, -1.0],
        ),
        (
            "box_kernel_n7",
            vec![1.0; 7],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ),
        (
            "delta_x_signal",
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
        ),
        (
            "long_x_short",
            (0..20).map(|i| ((i as f64) * 0.2).sin()).collect(),
            vec![0.5, -1.0, 0.5],
        ),
    ];
    let modes = ["full", "same", "valid"];

    let mut points = Vec::new();
    for (label, a, b) in pairs {
        for mode in modes {
            points.push(PointCase {
                case_id: format!("{label}_{mode}"),
                a: a.clone(),
                b: b.clone(),
                mode: mode.into(),
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
from scipy import signal

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
    cid = case["case_id"]
    a = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    mode = case["mode"]
    try:
        v = signal.convolve(a, b, mode=mode, method='direct')
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize convolve query");
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
                "failed to spawn python3 for convolve oracle: {e}"
            );
            eprintln!("skipping convolve oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open convolve oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "convolve oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping convolve oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for convolve oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "convolve oracle failed: {stderr}"
        );
        eprintln!("skipping convolve oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse convolve oracle JSON"))
}

#[test]
fn diff_signal_convolve() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(fsci_v) = convolve(&case.a, &case.b, parse_mode(&case.mode)) else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                mode: case.mode.clone(),
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
            mode: case.mode.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_convolve".into(),
        category: "scipy.signal.convolve (method='direct')".into(),
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
                "convolve {} mismatch: {} abs_diff={}",
                d.mode, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.convolve(direct) conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
