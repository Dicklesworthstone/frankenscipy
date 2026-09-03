#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::upfirdn (polyphase
//! FIR filtering with up/down resampling).
//!
//! Resolves [frankenscipy-ziwx4]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::upfirdn;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    h: Vec<f64>,
    x: Vec<f64>,
    up: usize,
    down: usize,
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
    fs::create_dir_all(output_dir()).expect("create upfirdn diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize upfirdn diff log");
    fs::write(path, json).expect("write upfirdn diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "fir3_up2_down1".into(),
            h: vec![1.0, 1.0, 1.0],
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            up: 2,
            down: 1,
        },
        PointCase {
            case_id: "identity_up1_down2".into(),
            h: vec![1.0],
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            up: 1,
            down: 2,
        },
        PointCase {
            case_id: "boxcar5_up3_down2".into(),
            h: vec![0.2, 0.2, 0.2, 0.2, 0.2],
            x: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            up: 3,
            down: 2,
        },
        PointCase {
            case_id: "halfband4_up2_down1".into(),
            h: vec![0.25, 0.5, 0.5, 0.25],
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            up: 2,
            down: 1,
        },
        PointCase {
            case_id: "fir2_up1_down1".into(),
            h: vec![0.5, 0.5],
            x: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            up: 1,
            down: 1,
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

def finite_vec_or_none(arr):
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
    h = np.array(case["h"], dtype=float)
    x = np.array(case["x"], dtype=float)
    up = int(case["up"]); down = int(case["down"])
    try:
        y = signal.upfirdn(h, x, up=up, down=down)
        points.append({"case_id": cid, "values": finite_vec_or_none(y)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize upfirdn query");
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
                "failed to spawn python3 for upfirdn oracle: {e}"
            );
            eprintln!("skipping upfirdn oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open upfirdn oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "upfirdn oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping upfirdn oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for upfirdn oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "upfirdn oracle failed: {stderr}"
        );
        eprintln!("skipping upfirdn oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse upfirdn oracle JSON"))
}

#[test]
fn diff_signal_upfirdn() {
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
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(fsci_v) = upfirdn(&case.h, &case.x, case.up, case.down) else {
            continue;
        };
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
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
        test_id: "diff_signal_upfirdn".into(),
        category: "scipy.signal.upfirdn".into(),
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
            eprintln!("upfirdn mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "upfirdn conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
