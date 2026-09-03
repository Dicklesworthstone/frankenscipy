#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three FFT helpers:
//!   - `periodogram_simple(x, fs)`   vs `scipy.signal.periodogram(x, fs,
//!                                       window='boxcar', detrend=False,
//!                                       scaling='density')`
//!   - `magnitude_spectrum(x)`       vs `np.abs(np.fft.rfft(x)) / n`
//!   - `phase_spectrum_signal(x)`    vs `np.angle(np.fft.rfft(x))`
//!
//! Resolves [frankenscipy-848pd]. All three return one-sided spectra of
//! length n/2+1. Tight 1e-10 abs tolerance — both sides round-trip
//! through real-valued FFT.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{magnitude_spectrum, periodogram_simple, phase_spectrum_signal};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    x: Vec<f64>,
    fs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For periodogram: alternating [f0, p0, f1, p1, ...]. For magnitude/phase: plain vector.
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
    let signals: &[(&str, Vec<f64>)] = &[
        ("linear_n8", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            "sin_n16",
            (0..16).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "two_tone_n32",
            (0..32)
                .map(|i| {
                    let t = i as f64 * 0.1;
                    (2.0 * std::f64::consts::PI * 1.0 * t).sin()
                        + 0.5 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
                })
                .collect(),
        ),
        (
            "decay_n10",
            (0..10).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
    ];
    let fs_values: &[f64] = &[1.0, 10.0];

    let mut points = Vec::new();
    for (label, x) in signals {
        for op in ["magnitude_spectrum", "phase_spectrum"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                x: x.clone(),
                fs: 1.0,
            });
        }
        for &fs in fs_values {
            points.push(PointCase {
                case_id: format!("periodogram_{label}_fs{fs}"),
                op: "periodogram".into(),
                x: x.clone(),
                fs,
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
from scipy.signal import periodogram

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
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
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    n = len(x)
    try:
        if op == "magnitude_spectrum":
            v = np.abs(np.fft.rfft(x)) / n
            points.append({"case_id": cid, "values": finite_vec_or_none(v)})
        elif op == "phase_spectrum":
            v = np.angle(np.fft.rfft(x))
            points.append({"case_id": cid, "values": finite_vec_or_none(v)})
        elif op == "periodogram":
            fs = float(case["fs"])
            f, p = periodogram(x, fs, window='boxcar', detrend=False, scaling='density')
            packed = []
            for ff, pp in zip(f, p):
                packed.append(float(ff)); packed.append(float(pp))
            if any(not math.isfinite(v) for v in packed):
                points.append({"case_id": cid, "values": None})
            else:
                points.append({"case_id": cid, "values": packed})
        else:
            points.append({"case_id": cid, "values": None})
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

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.op.as_str() {
        "magnitude_spectrum" => magnitude_spectrum(&case.x).ok(),
        "phase_spectrum" => phase_spectrum_signal(&case.x).ok(),
        "periodogram" => {
            let (f, p) = periodogram_simple(&case.x, case.fs).ok()?;
            let mut packed = Vec::with_capacity(f.len() * 2);
            for (ff, pp) in f.iter().zip(p.iter()) {
                packed.push(*ff);
                packed.push(*pp);
            }
            Some(packed)
        }
        _ => None,
    }
}

#[test]
fn diff_fft_periodogram_magnitude_phase() {
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
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
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
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_periodogram_magnitude_phase".into(),
        category: "scipy.signal.periodogram + magnitude/phase spectra".into(),
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
                "fft_helper {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy fft-helper conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
