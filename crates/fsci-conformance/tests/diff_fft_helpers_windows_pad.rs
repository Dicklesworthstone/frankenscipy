#![forbid(unsafe_code)]
//! Live SciPy/numpy differential coverage for fsci_fft helper functions:
//! hann_window, hamming_window, blackman_window, apply_window,
//! zero_pad_pow2.
//!
//! Resolves [frankenscipy-jmkwv]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{apply_window, blackman_window, hamming_window, hann_window, zero_pad_pow2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    n: usize,
    /// For apply_window: input signal.
    x: Vec<f64>,
    /// For apply_window: window vector.
    window: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create fft_helpers diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fft_helpers diff log");
    fs::write(path, json).expect("write fft_helpers diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    for n in [8_usize, 16, 32, 64] {
        for op in ["hann_window", "hamming_window", "blackman_window"] {
            points.push(PointCase {
                case_id: format!("{op}_n{n}"),
                op: op.into(),
                n,
                x: vec![],
                window: vec![],
            });
        }
    }

    // apply_window
    let signal_16: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.4).sin()).collect();
    let window_hann: Vec<f64> = {
        let n = 16;
        let nf = (n - 1) as f64;
        (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / nf).cos()))
            .collect()
    };
    points.push(PointCase {
        case_id: "apply_window_hann_16".into(),
        op: "apply_window".into(),
        n: 16,
        x: signal_16.clone(),
        window: window_hann,
    });

    // zero_pad_pow2: input length not a power of 2
    for &len in &[3_usize, 5, 7, 10, 100] {
        let x: Vec<f64> = (0..len).map(|i| (i + 1) as f64).collect();
        points.push(PointCase {
            case_id: format!("zero_pad_pow2_len{len}"),
            op: "zero_pad_pow2".into(),
            n: 0,
            x,
            window: vec![],
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
from scipy.signal import windows

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    n = int(case["n"])
    try:
        if op == "hann_window":
            v = windows.hann(n, sym=False)
            # fsci uses sym (periodic-vs-symmetric) — windows.hann(n, sym=True)
            # produces the same value as fsci. Try sym=True instead.
            v = windows.hann(n, sym=True)
        elif op == "hamming_window":
            v = windows.hamming(n, sym=True)
        elif op == "blackman_window":
            v = windows.blackman(n, sym=True)
        elif op == "apply_window":
            x = np.array(case["x"], dtype=float)
            w = np.array(case["window"], dtype=float)
            v = x * w
        elif op == "zero_pad_pow2":
            x = np.array(case["x"], dtype=float)
            np2 = next_pow2(len(x)) if len(x) > 0 else 1
            v = np.zeros(np2, dtype=float)
            v[:len(x)] = x
        else:
            v = None
        points.append({"case_id": cid, "values": finite_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fft_helpers query");
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
                "failed to spawn python3 for fft_helpers oracle: {e}"
            );
            eprintln!("skipping fft_helpers oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open fft_helpers oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fft_helpers oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fft_helpers oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fft_helpers oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fft_helpers oracle failed: {stderr}"
        );
        eprintln!("skipping fft_helpers oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fft_helpers oracle JSON"))
}

#[test]
fn diff_fft_helpers_windows_pad() {
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
            "hann_window" => hann_window(case.n),
            "hamming_window" => hamming_window(case.n),
            "blackman_window" => blackman_window(case.n),
            "apply_window" => apply_window(&case.x, &case.window),
            "zero_pad_pow2" => zero_pad_pow2(&case.x),
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
        test_id: "diff_fft_helpers_windows_pad".into(),
        category: "fsci_fft helper windows + apply_window + zero_pad_pow2".into(),
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
        "fft_helpers conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
