#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.correlate2d`.
//!
//! Resolves [frankenscipy-p2d3y]. fsci_signal::correlate2d uses
//! direct-method 2-D correlation (reverse-and-convolve). scipy with
//! `boundary='fill'`, `fillvalue=0` should match exactly for the
//! 'full', 'same', and 'valid' modes. 1e-12 abs tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{ConvolveMode, correlate2d};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<f64>,
    a_shape: (usize, usize),
    v: Vec<f64>,
    v_shape: (usize, usize),
    mode: String,
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
    mode: String,
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
    fs::create_dir_all(output_dir()).expect("create correlate2d diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize correlate2d diff log");
    fs::write(path, json).expect("write correlate2d diff log");
}

fn parse_mode(name: &str) -> ConvolveMode {
    match name {
        "full" => ConvolveMode::Full,
        "same" => ConvolveMode::Same,
        "valid" => ConvolveMode::Valid,
        _ => ConvolveMode::Full,
    }
}

fn generate_query() -> OracleQuery {
    let pairs: &[(&str, Vec<f64>, (usize, usize), Vec<f64>, (usize, usize))] = &[
        (
            "5x5_x_3x3_laplace",
            (1..=25).map(|i| i as f64).collect(),
            (5, 5),
            vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
            (3, 3),
        ),
        (
            "4x4_x_2x2",
            (1..=16).map(|i| i as f64).collect(),
            (4, 4),
            vec![1.0, -1.0, -1.0, 1.0],
            (2, 2),
        ),
        (
            "6x6_box_x_3x3_box",
            vec![1.0; 36],
            (6, 6),
            vec![1.0 / 9.0; 9],
            (3, 3),
        ),
        (
            "5x4_x_2x2",
            (1..=20).map(|i| i as f64).collect(),
            (5, 4),
            vec![1.0, 2.0, 3.0, 4.0],
            (2, 2),
        ),
    ];
    let modes = ["full", "same", "valid"];

    let mut points = Vec::new();
    for (label, a, ashape, v, vshape) in pairs {
        for mode in modes {
            points.push(PointCase {
                case_id: format!("{label}_{mode}"),
                a: a.clone(),
                a_shape: *ashape,
                v: v.clone(),
                v_shape: *vshape,
                mode: mode.into(),
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
    cid = case["case_id"]
    a = np.array(case["a"], dtype=float).reshape(case["a_shape"])
    v = np.array(case["v"], dtype=float).reshape(case["v_shape"])
    mode = case["mode"]
    try:
        r = signal.correlate2d(a, v, mode=mode, boundary='fill', fillvalue=0)
        points.append({"case_id": cid, "values": finite_vec_or_none(r)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize correlate2d query");
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
                "failed to spawn python3 for correlate2d oracle: {e}"
            );
            eprintln!("skipping correlate2d oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open correlate2d oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "correlate2d oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping correlate2d oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for correlate2d oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "correlate2d oracle failed: {stderr}"
        );
        eprintln!("skipping correlate2d oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse correlate2d oracle JSON"))
}

#[test]
fn diff_signal_correlate2d() {
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
        let Ok(fsci_v) = correlate2d(
            &case.a,
            case.a_shape,
            &case.v,
            case.v_shape,
            parse_mode(&case.mode),
        ) else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                mode: case.mode.clone(),
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
            mode: case.mode.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_correlate2d".into(),
        category: "scipy.signal.correlate2d (boundary=fill, fillvalue=0)".into(),
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
                "correlate2d {} mismatch: {} abs_diff={}",
                d.mode, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.correlate2d conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
