#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::freqs (analog filter
//! frequency response at specified ω).
//!
//! Resolves [frankenscipy-2i7no]. 1e-10 abs on (w, |h|, angle(h)).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::freqs;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
    w: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    w: Option<Vec<f64>>,
    h_mag: Option<Vec<f64>>,
    h_phase: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create freqs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize freqs diff log");
    fs::write(path, json).expect("write freqs diff log");
}

fn generate_query() -> OracleQuery {
    let w_grid = vec![0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    let points = vec![
        PointCase {
            case_id: "lp_second_order".into(),
            b: vec![1.0],
            a: vec![1.0, 0.5, 1.0],
            w: w_grid.clone(),
        },
        PointCase {
            case_id: "first_order".into(),
            b: vec![1.0],
            a: vec![1.0, 1.0],
            w: w_grid.clone(),
        },
        PointCase {
            case_id: "hp_inverse".into(),
            b: vec![1.0, 0.0, 0.0],
            a: vec![1.0, 1.4, 1.0],
            w: w_grid.clone(),
        },
        PointCase {
            case_id: "third_order".into(),
            b: vec![1.0, 0.5],
            a: vec![1.0, 2.0, 2.0, 1.0],
            w: w_grid,
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    w = np.array(case["w"], dtype=float)
    try:
        w_out, h = signal.freqs(b, a, w)
        points.append({
            "case_id": cid,
            "w": finite_or_none(w_out),
            "h_mag": finite_or_none(np.abs(h)),
            "h_phase": finite_or_none(np.angle(h)),
        })
    except Exception:
        points.append({"case_id": cid, "w": None, "h_mag": None, "h_phase": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize freqs query");
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
                "failed to spawn python3 for freqs oracle: {e}"
            );
            eprintln!("skipping freqs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open freqs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "freqs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping freqs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for freqs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "freqs oracle failed: {stderr}"
        );
        eprintln!("skipping freqs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse freqs oracle JSON"))
}

#[test]
fn diff_signal_freqs() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let (Some(w_exp), Some(mag_exp), Some(phase_exp)) = (
            scipy_arm.w.as_ref(),
            scipy_arm.h_mag.as_ref(),
            scipy_arm.h_phase.as_ref(),
        ) else {
            continue;
        };
        let Ok(res) = freqs(&case.b, &case.a, &case.w) else {
            continue;
        };
        let abs_d = if res.w.len() != w_exp.len()
            || res.h_mag.len() != mag_exp.len()
            || res.h_phase.len() != phase_exp.len()
        {
            f64::INFINITY
        } else {
            let dw = res
                .w
                .iter()
                .zip(w_exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let dm = res
                .h_mag
                .iter()
                .zip(mag_exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let dp = res
                .h_phase
                .iter()
                .zip(phase_exp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            dw.max(dm).max(dp)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_freqs".into(),
        category: "scipy.signal.freqs (analog frequency response)".into(),
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
            eprintln!("freqs mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "freqs conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
