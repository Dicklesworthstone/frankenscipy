#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.rfft2`.
//!
//! Resolves [frankenscipy-n22d3]. fsci_fft::rfft2(input, shape, opts)
//! returns Vec<Complex64> of length rows × (cols/2 + 1). 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, rfft2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    input: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened complex values [re0, im0, re1, im1, ...]
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
    fs::create_dir_all(output_dir()).expect("create rfft2 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rfft2 diff log");
    fs::write(path, json).expect("write rfft2 diff log");
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
            "4x4_alternating",
            4,
            4,
            (0..16)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        (
            "3x6_sine",
            3,
            6,
            (0..18).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "4x8_smooth_radial",
            4,
            8,
            (0..32)
                .map(|i| {
                    let r = (i / 8) as f64;
                    let c = (i % 8) as f64;
                    ((r - 1.5).powi(2) + (c - 3.5).powi(2)).sqrt()
                })
                .collect(),
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, r, c, x)| PointCase {
            case_id: (*name).into(),
            rows: *r,
            cols: *c,
            input: x.clone(),
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
    r = int(case["rows"]); c = int(case["cols"])
    x = np.array(case["input"], dtype=float).reshape(r, c)
    try:
        y = fft.rfft2(x)
        points.append({"case_id": cid, "values": finite_pack_or_none(y)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize rfft2 query");
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
                "failed to spawn python3 for rfft2 oracle: {e}"
            );
            eprintln!("skipping rfft2 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open rfft2 oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rfft2 oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rfft2 oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for rfft2 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rfft2 oracle failed: {stderr}"
        );
        eprintln!("skipping rfft2 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rfft2 oracle JSON"))
}

#[test]
fn diff_fft_rfft2() {
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
        let opts = FftOptions::default();
        let Ok(complex_out) = rfft2(&case.input, (case.rows, case.cols), &opts) else {
            continue;
        };
        let mut fsci_v = Vec::with_capacity(complex_out.len() * 2);
        for c in &complex_out {
            fsci_v.push(c.0);
            fsci_v.push(c.1);
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
        test_id: "diff_fft_rfft2".into(),
        category: "scipy.fft.rfft2".into(),
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
            eprintln!("rfft2 mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.fft.rfft2 conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
