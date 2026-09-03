#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two FFT helpers:
//!   - `polynomial_multiply_fft(a, b)` vs `np.convolve(a, b)`
//!     (ascending-coefficient polynomial product)
//!   - `analytic_signal(x)` vs `scipy.signal.hilbert(x)` (analytic
//!     signal as (re, im) pairs)
//!
//! Resolves [frankenscipy-pqaz0]. Both round-trip through real-valued
//! FFT; 1e-10 abs tolerance dominates O(log N) ULP accumulation.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, analytic_signal, polynomial_multiply_fft};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// polymul: convolved vector. analytic_signal: alternating [re, im, re, im, ...]
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
    fs::create_dir_all(output_dir()).expect("create polymul_analytic diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize polymul_analytic diff log");
    fs::write(path, json).expect("write polymul_analytic diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // polymul cases
    let polymul_cases: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("simple_3x3", vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        ("small_2x4", vec![1.0, -1.0], vec![1.0, 2.0, 3.0, 4.0]),
        (
            "longer_8x6",
            vec![1.0, 0.0, -1.0, 2.0, 1.0, -3.0, 0.5, 1.0],
            vec![0.5, 1.0, 1.5, -1.0, 0.0, 2.0],
        ),
        ("single_element_each", vec![3.0], vec![7.0]),
        (
            "delta_kernel",
            vec![0.0, 0.0, 1.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0],
        ),
    ];
    for (label, a, b) in polymul_cases {
        points.push(PointCase {
            case_id: format!("polymul_{label}"),
            op: "polymul".into(),
            a: a.clone(),
            b: b.clone(),
        });
    }

    // analytic_signal cases
    let analytic_cases: &[(&str, Vec<f64>)] = &[
        ("linear_n8", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        (
            "sine_n16",
            (0..16).map(|i| ((i as f64) * 0.4).sin()).collect(),
        ),
        (
            "cosine_n32",
            (0..32).map(|i| ((i as f64) * 0.2).cos()).collect(),
        ),
        (
            "decay_n12",
            (0..12).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
    ];
    for (label, x) in analytic_cases {
        points.push(PointCase {
            case_id: format!("analytic_{label}"),
            op: "analytic".into(),
            a: x.clone(),
            b: vec![],
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
from scipy import signal

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
    a = np.array(case["a"], dtype=float)
    try:
        if op == "polymul":
            b = np.array(case["b"], dtype=float)
            v = np.convolve(a, b)
            points.append({"case_id": cid, "values": finite_vec_or_none(v)})
        elif op == "analytic":
            c = signal.hilbert(a)
            packed = []
            for v in c:
                packed.append(float(np.real(v)))
                packed.append(float(np.imag(v)))
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
    let query_json = serde_json::to_string(query).expect("serialize polymul_analytic query");
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
                "failed to spawn python3 for polymul_analytic oracle: {e}"
            );
            eprintln!("skipping polymul_analytic oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open polymul_analytic oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "polymul_analytic oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping polymul_analytic oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for polymul_analytic oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "polymul_analytic oracle failed: {stderr}"
        );
        eprintln!("skipping polymul_analytic oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse polymul_analytic oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    let opts = FftOptions::default();
    match case.op.as_str() {
        "polymul" => polynomial_multiply_fft(&case.a, &case.b, &opts).ok(),
        "analytic" => {
            let v = analytic_signal(&case.a).ok()?;
            let mut packed = Vec::with_capacity(v.len() * 2);
            for (re, im) in v {
                packed.push(re);
                packed.push(im);
            }
            Some(packed)
        }
        _ => None,
    }
}

#[test]
fn diff_fft_polymul_analytic_signal() {
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
        let Some(fsci_v) = fsci_eval(case) else {
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
        test_id: "diff_fft_polymul_analytic_signal".into(),
        category: "polynomial_multiply_fft + analytic_signal".into(),
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
                "polymul_analytic {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "polymul/analytic_signal conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
