#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.zoom_fft`.
//!
//! Resolves [frankenscipy-mhiog]. `fsci_signal::zoom_fft` exposes the
//! SciPy default `fs=2` convention: frequency 1.0 is Nyquist, and the
//! range [0, 2) spans the full DFT unit circle.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::zoom_fft;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-8;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    f1: f64,
    f2: f64,
    m: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened [re, im, re, im, ...] of M complex outputs.
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
    fs::create_dir_all(output_dir()).expect("create zoom_fft diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zoom_fft diff log");
    fs::write(path, json).expect("write zoom_fft diff log");
}

fn generate_query() -> OracleQuery {
    let signals: &[(&str, Vec<f64>)] = &[
        ("linear_n8", (1..=8).map(|i| i as f64).collect()),
        (
            "sine_n16",
            (0..16).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "decay_n12",
            (0..12).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
        (
            "mixed_n32",
            (0..32)
                .map(|i| {
                    let t = i as f64;
                    (0.25 * t).cos() + 0.5 * (0.9 * t).sin()
                })
                .collect(),
        ),
    ];

    let configs: &[(&str, f64, f64, Option<usize>)] = &[
        ("low_band_m5", 0.1, 0.4, Some(5)),
        ("mid_band_m9", 0.35, 0.95, Some(9)),
        ("nyquist_crossing_m7", 0.8, 1.2, Some(7)),
        ("full_fft_m_eq_n", 0.0, 2.0, None),
    ];

    let mut points = Vec::new();
    for (signal_label, x) in signals {
        for (config_label, f1, f2, m) in configs {
            points.push(PointCase {
                case_id: format!("{signal_label}_{config_label}"),
                x: x.clone(),
                f1: *f1,
                f2: *f2,
                m: m.unwrap_or(x.len()),
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

def finite_pack_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            re = float(v.real); im = float(v.imag)
        except Exception:
            return None
        if not (math.isfinite(re) and math.isfinite(im)):
            return None
        out.append(re); out.append(im)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    f1 = float(case["f1"])
    f2 = float(case["f2"])
    m = int(case["m"])
    try:
        v = signal.zoom_fft(x, [f1, f2], m=m, fs=2.0)
        points.append({"case_id": cid, "values": finite_pack_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize zoom_fft query");
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
                "failed to spawn python3 for zoom_fft oracle: {e}"
            );
            eprintln!("skipping zoom_fft oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open zoom_fft oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zoom_fft oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zoom_fft oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for zoom_fft oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zoom_fft oracle failed: {stderr}"
        );
        eprintln!("skipping zoom_fft oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zoom_fft oracle JSON"))
}

#[test]
fn diff_signal_zoom_fft() {
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
        let Ok(complex_out) = zoom_fft(&case.x, (case.f1, case.f2), case.m) else {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        };
        let mut fsci_v = Vec::with_capacity(complex_out.len() * 2);
        for (re, im) in complex_out {
            fsci_v.push(re);
            fsci_v.push(im);
        }
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
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
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_zoom_fft".into(),
        category: "scipy.signal.zoom_fft".into(),
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
            eprintln!("zoom_fft mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.zoom_fft conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
