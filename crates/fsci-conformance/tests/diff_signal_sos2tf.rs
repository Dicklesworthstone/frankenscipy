#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::sos2tf
//! (second-order section → transfer function).
//!
//! Resolves [frankenscipy-m5p9v]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::sos2tf;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    /// Flattened SOS sections: each section is [b0, b1, b2, a0, a1, a2].
    sos_flat: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    b: Option<Vec<f64>>,
    a: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create sos2tf diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize sos2tf diff log");
    fs::write(path, json).expect("write sos2tf diff log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        PointCase {
            case_id: "one_section".into(),
            // H(z) = (1 + 0.5z^-1) / (1 - 0.3z^-1)
            sos_flat: vec![1.0, 0.5, 0.0, 1.0, -0.3, 0.0],
        },
        PointCase {
            case_id: "two_sections".into(),
            sos_flat: vec![
                1.0, 0.5, 0.0, 1.0, -0.3, 0.0, 1.0, -0.5, 0.25, 1.0, 0.2, 0.1,
            ],
        },
        PointCase {
            case_id: "three_sections".into(),
            sos_flat: vec![
                0.5, 0.0, 0.0, 1.0, 0.1, 0.0, 1.0, -0.1, 0.05, 1.0, 0.4, 0.2, 1.0, 0.3, 0.1, 1.0,
                -0.5, 0.25,
            ],
        },
        PointCase {
            case_id: "fir_2sections".into(),
            // Pure FIR: a coefficients are [1, 0, 0]
            sos_flat: vec![
                0.25, 0.5, 0.25, 1.0, 0.0, 0.0, 1.0, -1.0, 0.5, 1.0, 0.0, 0.0,
            ],
        },
        PointCase {
            case_id: "iir_resonator".into(),
            sos_flat: vec![0.049, 0.099, 0.049, 1.0, -1.279, 0.478],
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
    sos_flat = np.array(case["sos_flat"], dtype=float)
    n_sec = sos_flat.size // 6
    sos = sos_flat.reshape(n_sec, 6)
    try:
        b, a = signal.sos2tf(sos)
        points.append({"case_id": cid,
                       "b": finite_vec_or_none(b),
                       "a": finite_vec_or_none(a)})
    except Exception:
        points.append({"case_id": cid, "b": None, "a": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize sos2tf query");
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
                "failed to spawn python3 for sos2tf oracle: {e}"
            );
            eprintln!("skipping sos2tf oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open sos2tf oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "sos2tf oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping sos2tf oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for sos2tf oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sos2tf oracle failed: {stderr}"
        );
        eprintln!("skipping sos2tf oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse sos2tf oracle JSON"))
}

#[test]
fn diff_signal_sos2tf() {
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
        let (Some(b_exp), Some(a_exp)) = (scipy_arm.b.as_ref(), scipy_arm.a.as_ref()) else {
            continue;
        };
        // Reshape SOS flat into [f64; 6] sections
        let sos: Vec<[f64; 6]> = case
            .sos_flat
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
            .collect();
        let ba = sos2tf(&sos);
        let abs_d = if ba.b.len() != b_exp.len() || ba.a.len() != a_exp.len() {
            f64::INFINITY
        } else {
            let db =
                ba.b.iter()
                    .zip(b_exp.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
            let da =
                ba.a.iter()
                    .zip(a_exp.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
            db.max(da)
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
        test_id: "diff_signal_sos2tf".into(),
        category: "scipy.signal.sos2tf".into(),
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
            eprintln!("sos2tf mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "sos2tf conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
