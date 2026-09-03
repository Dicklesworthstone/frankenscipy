#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.sweep_poly(t, poly)`.
//!
//! Resolves [frankenscipy-dxmih]. fsci_signal::sweep_poly generates
//! cos(2π · ∫₀ᵗ poly(τ)dτ) for a polynomial frequency profile. Closed-
//! form, so 1e-12 abs tolerance is appropriate. scipy's `poly` argument
//! follows numpy.poly1d convention (leading coefficient first, highest
//! power); fsci uses the same convention.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::sweep_poly;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    t: Vec<f64>,
    poly: Vec<f64>,
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
    fs::create_dir_all(output_dir()).expect("create sweep_poly diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sweep_poly diff log");
    fs::write(path, json).expect("write sweep_poly diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<f64>)] = &[
        // (label, t, poly)
        (
            "linear_freq",
            (0..20).map(|i| i as f64 * 0.05).collect(),
            vec![2.0, 1.0], // f(t) = 2t + 1
        ),
        (
            "quadratic_freq",
            (0..15).map(|i| i as f64 * 0.1).collect(),
            vec![1.0, 0.5, 2.0], // f(t) = t² + 0.5t + 2
        ),
        (
            "constant_freq",
            (0..10).map(|i| i as f64 * 0.2).collect(),
            vec![3.0], // f(t) = 3 (constant carrier)
        ),
        (
            "cubic_freq",
            (0..12).map(|i| i as f64 * 0.08).collect(),
            vec![0.5, -1.0, 2.0, 1.5], // f(t) = 0.5t³ - t² + 2t + 1.5
        ),
        (
            "neg_coeffs",
            (0..8).map(|i| i as f64 * 0.15).collect(),
            vec![-1.5, 0.5],
        ),
        (
            "high_order",
            (0..10).map(|i| i as f64 * 0.1).collect(),
            vec![0.1, -0.2, 0.3, -0.1, 1.0], // degree 4
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, t, p)| PointCase {
            case_id: (*name).into(),
            t: t.clone(),
            poly: p.clone(),
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
from scipy.signal import sweep_poly

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
    t = np.array(case["t"], dtype=float)
    poly = np.array(case["poly"], dtype=float)
    try:
        v = sweep_poly(t, poly)
        points.append({"case_id": cid, "values": finite_vec_or_none(v)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sweep_poly query");
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
                "failed to spawn python3 for sweep_poly oracle: {e}"
            );
            eprintln!("skipping sweep_poly oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open sweep_poly oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sweep_poly oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sweep_poly oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for sweep_poly oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sweep_poly oracle failed: {stderr}"
        );
        eprintln!("skipping sweep_poly oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sweep_poly oracle JSON"))
}

#[test]
fn diff_signal_sweep_poly() {
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
        let fsci_v = sweep_poly(&case.t, &case.poly);
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
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
        test_id: "diff_signal_sweep_poly".into(),
        category: "scipy.signal.sweep_poly".into(),
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
            eprintln!("sweep_poly mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.sweep_poly conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
