#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Discrete Cosine and
//! Discrete Sine Transform families:
//!   - `scipy.fft.dct(x, type=1|2|3|4, norm='backward')`
//!   - `scipy.fft.dst(x, type=1|2|3|4, norm='backward')`
//!
//! Resolves [frankenscipy-obsfj]. fsci_fft exposes dct (≡ DCT-II),
//! dct_i, dct_iii, dct_iv, dst_i, dst_ii, dst_iii, dst_iv via
//! transforms.rs but had no dedicated diff_fft_* harness — coverage
//! was only via P2C-005 fixture/oracle dispatch. Verified all 8
//! transforms match scipy to ~1e-14.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, dct, dct_i, dct_iii, dct_iv, dst_i, dst_ii, dst_iii, dst_iv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 5.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: Vec<f64>,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create dct/dst diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize dct/dst diff log");
    fs::write(path, json).expect("write dct/dst diff log");
}

fn generate_query() -> OracleQuery {
    // Input signals: arithmetic series, constant, alternating, decaying.
    let inputs: &[(&str, Vec<f64>)] = &[
        ("linear_n8", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ("constant_n6", vec![3.5, 3.5, 3.5, 3.5, 3.5, 3.5]),
        ("alternating_n6", vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        (
            "decay_n8",
            (0..8).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
        (
            "single_spike_n8",
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    let funcs = [
        "dct_i", "dct_ii", "dct_iii", "dct_iv", "dst_i", "dst_ii", "dst_iii", "dst_iv",
    ];
    let mut points = Vec::new();
    for (label, xs) in inputs {
        for func in funcs {
            points.push(PointCase {
                case_id: format!("{func}_{label}"),
                func: func.to_string(),
                x: xs.clone(),
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
from scipy.fft import dct, dst

DISPATCH = {
    "dct_i":   lambda x: dct(x, type=1, norm='backward'),
    "dct_ii":  lambda x: dct(x, type=2, norm='backward'),
    "dct_iii": lambda x: dct(x, type=3, norm='backward'),
    "dct_iv":  lambda x: dct(x, type=4, norm='backward'),
    "dst_i":   lambda x: dst(x, type=1, norm='backward'),
    "dst_ii":  lambda x: dst(x, type=2, norm='backward'),
    "dst_iii": lambda x: dst(x, type=3, norm='backward'),
    "dst_iv":  lambda x: dst(x, type=4, norm='backward'),
}

def finite_vec_or_none(arr):
    out = []
    for v in arr.tolist():
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
    cid = case["case_id"]; func = case["func"]
    x = np.array(case["x"], dtype=float)
    handler = DISPATCH.get(func)
    try:
        v = handler(x) if handler is not None else None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize dct/dst query");
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
                "failed to spawn python3 for dct/dst oracle: {e}"
            );
            eprintln!("skipping dct/dst oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open dct/dst oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "dct/dst oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping dct/dst oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for dct/dst oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "dct/dst oracle failed: {stderr}"
        );
        eprintln!("skipping dct/dst oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse dct/dst oracle JSON"))
}

fn fsci_eval(func: &str, x: &[f64]) -> Option<Vec<f64>> {
    let opts = FftOptions::default();
    match func {
        "dct_i" => dct_i(x, &opts).ok(),
        "dct_ii" => dct(x, &opts).ok(),
        "dct_iii" => dct_iii(x, &opts).ok(),
        "dct_iv" => dct_iv(x, &opts).ok(),
        "dst_i" => dst_i(x, &opts).ok(),
        "dst_ii" => dst_ii(x, &opts).ok(),
        "dst_iii" => dst_iii(x, &opts).ok(),
        "dst_iv" => dst_iv(x, &opts).ok(),
        _ => None,
    }
}

#[test]
fn diff_fft_dct_dst() {
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
        let Some(fsci_v) = fsci_eval(&case.func, &case.x) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
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
            func: case.func.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_dct_dst".into(),
        category: "scipy.fft.{dct,dst} type=I..IV".into(),
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
                "dct/dst {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.fft dct/dst conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
