#![forbid(unsafe_code)]
//! Live numpy parity for fsci_signal::frame_signal.
//!
//! Resolves [frankenscipy-ubwx7]. frame_signal(x, frame_len, hop_len)
//! produces overlapping windows of length frame_len strided by
//! hop_len. Reference computed via numpy slicing.
//!
//! Tolerance: exact (no floating-point ops applied to data).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::frame_signal;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    x: Vec<f64>,
    frame_len: usize,
    hop_len: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened row-major frames.
    flat: Option<Vec<f64>>,
    n_frames: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create frame diff dir");
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
    let s10: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let s20: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5 + 0.1).collect();
    let s32: Vec<f64> = (0..32).map(|i| ((i as f64) * 0.25).sin()).collect();

    OracleQuery {
        points: vec![
            Case {
                case_id: "s10_f4_h2".into(),
                x: s10.clone(),
                frame_len: 4,
                hop_len: 2,
            },
            Case {
                case_id: "s10_f3_h1".into(),
                x: s10.clone(),
                frame_len: 3,
                hop_len: 1,
            },
            Case {
                case_id: "s10_f5_h5".into(),
                x: s10,
                frame_len: 5,
                hop_len: 5,
            },
            Case {
                case_id: "s20_f6_h3".into(),
                x: s20.clone(),
                frame_len: 6,
                hop_len: 3,
            },
            Case {
                case_id: "s20_f10_h4".into(),
                x: s20,
                frame_len: 10,
                hop_len: 4,
            },
            Case {
                case_id: "s32_f8_h2".into(),
                x: s32,
                frame_len: 8,
                hop_len: 2,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    f = int(case["frame_len"]); h = int(case["hop_len"])
    try:
        if f == 0 or h == 0 or len(x) < f:
            points.append({"case_id": cid, "flat": [], "n_frames": 0})
            continue
        frames = []
        i = 0
        while i + f <= len(x):
            frames.append(x[i:i+f].tolist())
            i += h
        n_frames = len(frames)
        flat = [float(v) for row in frames for v in row]
        points.append({"case_id": cid, "flat": flat, "n_frames": int(n_frames)})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "flat": None, "n_frames": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for frame oracle: {e}"
            );
            eprintln!("skipping frame oracle: python3 not available ({e})");
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
                "frame oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping frame oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for frame oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "frame oracle failed: {stderr}"
        );
        eprintln!("skipping frame oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse frame oracle JSON"))
}

#[test]
fn diff_signal_frame_signal() {
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
        let (Some(eflat), Some(enf)) = (arm.flat.as_ref(), arm.n_frames) else {
            continue;
        };
        let frames = frame_signal(&case.x, case.frame_len, case.hop_len);
        let flat: Vec<f64> = frames.iter().flatten().copied().collect();
        let abs_d = if frames.len() != enf || flat.len() != eflat.len() {
            f64::INFINITY
        } else {
            flat.iter()
                .zip(eflat.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
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
        test_id: "diff_signal_frame_signal".into(),
        category: "fsci_signal::frame_signal vs numpy slicing".into(),
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
            eprintln!("frame mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "frame conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
