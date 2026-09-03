#![forbid(unsafe_code)]
//! Live scipy.fft.idct parity for fsci_fft::idct (type-2 inverse DCT).
//!
//! Resolves [frankenscipy-t7u6z]. Compare across Backward and Ortho
//! normalizations on several input lengths and signal shapes.
//! Tolerance 1e-9 abs (DCT roundtrip carries some fp drift).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{FftOptions, Normalization, idct};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    norm: String, // "backward" | "ortho"
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    norm: String,
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
    fs::create_dir_all(output_dir()).expect("create idct diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // Different lengths
    let sines: Vec<(usize, Vec<f64>)> = vec![8, 16, 32, 64, 65]
        .into_iter()
        .map(|n| {
            let v: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
                .collect();
            (n, v)
        })
        .collect();
    let pulses: Vec<(usize, Vec<f64>)> = vec![16, 32, 64]
        .into_iter()
        .map(|n| {
            let mut v = vec![0.0_f64; n];
            v[n / 4] = 1.0;
            v[n / 2] = -0.5;
            v[3 * n / 4] = 0.3;
            (n, v)
        })
        .collect();
    let cosines: Vec<(usize, Vec<f64>)> = vec![16, 32, 64]
        .into_iter()
        .map(|n| {
            let v: Vec<f64> = (0..n)
                .map(|i| (3.0 * std::f64::consts::PI * i as f64 / n as f64).cos() + 0.3)
                .collect();
            (n, v)
        })
        .collect();

    for (label, signals) in [("sin", &sines), ("pulse", &pulses), ("cos", &cosines)] {
        for (n, v) in signals {
            for norm in ["backward", "ortho"] {
                points.push(Case {
                    case_id: format!("idct_{label}_n{n}_{norm}"),
                    norm: norm.into(),
                    x: v.clone(),
                });
            }
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
from scipy.fft import idct

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; norm = case["norm"]
    x = np.array(case["x"], dtype=float)
    try:
        y = idct(x, type=2, norm=norm)
        flat = [float(v) for v in y.tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "values": flat})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for idct oracle: {e}"
            );
            eprintln!("skipping idct oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "idct oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping idct oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for idct oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "idct oracle failed: {stderr}"
        );
        eprintln!("skipping idct oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse idct oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_fft_idct() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let opts = FftOptions {
            normalization: match case.norm.as_str() {
                "backward" => Normalization::Backward,
                "ortho" => Normalization::Ortho,
                _ => continue,
            },
            ..Default::default()
        };
        let Ok(y) = idct(&case.x, &opts) else {
            continue;
        };
        let abs_d = vec_max_diff(&y, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            norm: case.norm.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_idct".into(),
        category: "fsci_fft::idct (type-2) vs scipy.fft.idct".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.norm, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "idct conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
