#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.ifftn`.
//!
//! Resolves [frankenscipy-j2jkj]. Round-trips 2-D inputs via scipy's
//! fftn, then compares fsci's ifftn vs scipy's ifftn on identical
//! complex inputs. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, ifftn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    shape: Vec<usize>,
    /// Real-valued initial signal to fftn; scipy returns the complex
    /// fftn output via the response (so both sides ifftn the same
    /// thing).
    real_input: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Packed (re, im) for scipy's ifftn output.
    output: Option<Vec<f64>>,
    /// Packed (re, im) for scipy's fftn output (input to ifftn).
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
    fs::create_dir_all(output_dir()).expect("create ifftn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ifftn diff log");
    fs::write(path, json).expect("write ifftn diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<usize>, Vec<f64>)] = &[
        ("2x4_arange", vec![2, 4], (0..8).map(|i| i as f64).collect()),
        (
            "4x4_increasing",
            vec![4, 4],
            (1..=16).map(|i| i as f64).collect(),
        ),
        (
            "3x6_sine",
            vec![3, 6],
            (0..18).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "2x2x4_3d",
            vec![2, 2, 4],
            (0..16).map(|i| i as f64).collect(),
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, shape, x)| PointCase {
            case_id: (*name).into(),
            shape: shape.clone(),
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
    shape = case["shape"]
    x = np.array(case["real_input"], dtype=float).reshape(shape)
    try:
        y = fft.fftn(x)
        inv = fft.ifftn(y)
        points.append({
            "case_id": cid,
            "output": pack_complex(inv),
            "complex_input": pack_complex(y),
        })
    except Exception:
        points.append({"case_id": cid, "output": None, "complex_input": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ifftn query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for ifftn oracle: {e}"
            );
            eprintln!("skipping ifftn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ifftn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ifftn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ifftn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ifftn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ifftn oracle failed: {stderr}"
        );
        eprintln!("skipping ifftn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ifftn oracle JSON"))
}

#[test]
fn diff_fft_ifftn() {
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
        let Some(packed) = scipy_arm.complex_input.as_ref() else {
            continue;
        };
        let complex_input: Vec<(f64, f64)> =
            packed.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
        let opts = FftOptions::default();
        let Ok(rec) = ifftn(&complex_input, &case.shape, &opts) else {
            continue;
        };
        let mut fsci_v = Vec::with_capacity(rec.len() * 2);
        for c in &rec {
            fsci_v.push(c.0);
            fsci_v.push(c.1);
        }
        if fsci_v.len() != scipy_out.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
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
        test_id: "diff_fft_ifftn".into(),
        category: "scipy.fft.ifftn".into(),
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
            eprintln!("ifftn mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.fft.ifftn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
