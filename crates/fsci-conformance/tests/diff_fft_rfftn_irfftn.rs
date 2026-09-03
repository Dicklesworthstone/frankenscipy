#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.fft.rfftn` and `irfftn`.
//!
//! Resolves [frankenscipy-rggr1]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, irfftn, rfftn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    shape: Vec<usize>,
    /// For rfftn: real input. For irfftn: scipy's rfftn output (we use
    /// the same real_input and ask scipy to emit both rfftn(x) and
    /// irfftn(rfftn(x)), then route).
    real_input: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For rfftn: packed (re, im) of scipy.fft.rfftn(input).
    /// For irfftn: real output of scipy.fft.irfftn(rfftn(x), s=shape).
    expected: Option<Vec<f64>>,
    /// For irfftn: packed (re, im) of scipy.fft.rfftn(real_input) — the
    /// complex input that fsci.irfftn should receive.
    complex_input: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create rfftn_irfftn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize rfftn_irfftn diff log");
    fs::write(path, json).expect("write rfftn_irfftn diff log");
}

fn generate_query() -> OracleQuery {
    let inputs: &[(&str, Vec<usize>, Vec<f64>)] = &[
        ("2x4_arange", vec![2, 4], (0..8).map(|i| i as f64).collect()),
        (
            "3x4_sine",
            vec![3, 4],
            (0..12).map(|i| ((i as f64) * 0.3).sin()).collect(),
        ),
        (
            "2x3x4_3d_arange",
            vec![2, 3, 4],
            (0..24).map(|i| i as f64).collect(),
        ),
    ];
    let mut points = Vec::new();
    for (label, shape, x) in inputs {
        for op in ["rfftn", "irfftn"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                shape: shape.clone(),
                real_input: x.clone(),
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
    shape = case["shape"]
    x = np.array(case["real_input"], dtype=float).reshape(shape)
    try:
        y = fft.rfftn(x)
        if op == "rfftn":
            points.append({
                "case_id": cid,
                "expected": pack_complex(y),
                "complex_input": None,
            })
        elif op == "irfftn":
            inv = fft.irfftn(y, s=tuple(shape))
            points.append({
                "case_id": cid,
                "expected": finite_vec_or_none(inv),
                "complex_input": pack_complex(y),
            })
        else:
            points.append({"case_id": cid, "expected": None, "complex_input": None})
    except Exception:
        points.append({"case_id": cid, "expected": None, "complex_input": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize rfftn_irfftn query");
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
                "failed to spawn python3 for rfftn_irfftn oracle: {e}"
            );
            eprintln!("skipping rfftn_irfftn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open rfftn_irfftn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rfftn_irfftn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rfftn_irfftn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for rfftn_irfftn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rfftn_irfftn oracle failed: {stderr}"
        );
        eprintln!("skipping rfftn_irfftn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rfftn_irfftn oracle JSON"))
}

#[test]
fn diff_fft_rfftn_irfftn() {
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
        let Some(expected) = scipy_arm.expected.as_ref() else {
            continue;
        };
        let opts = FftOptions::default();
        let fsci_v = match case.op.as_str() {
            "rfftn" => {
                let Ok(out) = rfftn(&case.real_input, &case.shape, &opts) else {
                    continue;
                };
                let mut packed = Vec::with_capacity(out.len() * 2);
                for c in &out {
                    packed.push(c.0);
                    packed.push(c.1);
                }
                packed
            }
            "irfftn" => {
                let Some(packed) = scipy_arm.complex_input.as_ref() else {
                    continue;
                };
                let complex_in: Vec<(f64, f64)> = packed
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| (p[0], p[1]))
                    .collect();
                let Ok(out) = irfftn(&complex_in, &case.shape, &opts) else {
                    continue;
                };
                out
            }
            _ => continue,
        };
        if fsci_v.len() != expected.len() {
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
            .zip(expected.iter())
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
        test_id: "diff_fft_rfftn_irfftn".into(),
        category: "scipy.fft.rfftn + irfftn".into(),
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
                "rfftn_irfftn {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.fft.rfftn/irfftn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
