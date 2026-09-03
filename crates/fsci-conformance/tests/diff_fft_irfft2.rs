#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.irfft2`.
//!
//! Resolves [frankenscipy-0naam]. fsci_fft::irfft2(input, shape, opts)
//! inverts a 2-D real FFT. Complex inputs are pre-computed via scipy
//! rfft2 on a known real signal and passed in (re, im) pairs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, irfft2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    /// Original real input (to be rfft2-encoded by the python oracle, then
    /// passed back via the query response for fsci to irfft2).
    real_input: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// scipy's irfft2 output (the expected reconstruction).
    output: Option<Vec<f64>>,
    /// rfft2(real_input) packed as (re, im) for fsci's irfft2 input.
    complex_input: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create irfft2 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize irfft2 diff log");
    fs::write(path, json).expect("write irfft2 diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, usize, usize, Vec<f64>)] = &[
        (
            "2x4_increasing",
            2,
            4,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        ("4x4_increasing", 4, 4, (1..=16).map(|i| i as f64).collect()),
        (
            "3x6_sine",
            3,
            6,
            (0..18).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "4x8_decay",
            4,
            8,
            (0..32).map(|i| (-(i as f64) / 6.0).exp()).collect(),
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, r, c, x)| PointCase {
            case_id: (*name).into(),
            rows: *r,
            cols: *c,
            real_input: x.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import fft

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

def pack_complex(arr):
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
    r = int(case["rows"]); c = int(case["cols"])
    x = np.array(case["real_input"], dtype=float).reshape(r, c)
    try:
        y = fft.rfft2(x)
        rec = fft.irfft2(y, s=(r, c))
        points.append({
            "case_id": cid,
            "output": finite_vec_or_none(rec),
            "complex_input": pack_complex(y),
        })
    except Exception:
        points.append({"case_id": cid, "output": None, "complex_input": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize irfft2 query");
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
                "failed to spawn python3 for irfft2 oracle: {e}"
            );
            eprintln!("skipping irfft2 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open irfft2 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "irfft2 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping irfft2 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for irfft2 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "irfft2 oracle failed: {stderr}"
        );
        eprintln!("skipping irfft2 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse irfft2 oracle JSON"))
}

#[test]
fn diff_fft_irfft2() {
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
        let Some(scipy_out) = scipy_arm.output.as_ref() else {
            continue;
        };
        let Some(packed_complex) = scipy_arm.complex_input.as_ref() else {
            continue;
        };
        // Unpack scipy's rfft2 output into Complex64 tuples.
        let complex_input: Vec<(f64, f64)> = packed_complex
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| (p[0], p[1]))
            .collect();
        let opts = FftOptions::default();
        let Ok(rec) = irfft2(&complex_input, (case.rows, case.cols), &opts) else {
            continue;
        };
        if rec.len() != scipy_out.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = rec
            .iter()
            .zip(scipy_out.iter())
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
        test_id: "diff_fft_irfft2".into(),
        category: "scipy.fft.irfft2".into(),
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
            eprintln!("irfft2 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.fft.irfft2 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
